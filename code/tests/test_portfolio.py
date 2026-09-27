"""Tests for ``advisor.evaluation.portfolio`` -- the shared portfolio simulator.

Hand-checkable arithmetic on tiny price paths pins down the cost model and the
mark-to-market rule (equity starts at the initial capital, costs are charged
on one-way turnover, weights drift between rebalances). Synthetic geometric
random walks -- built by :func:`make_prices`, which the other evaluation test
modules import -- exercise the helpers at realistic sizes. Everything here is
offline and deterministic.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from advisor.evaluation.metrics import max_drawdown
from advisor.evaluation.portfolio import (
    decision_dates,
    drawdown_series,
    period_returns,
    pivot_prices,
    simulate_weights,
    slice_window,
    summarise,
    to_timestamp,
)

OHLCV_COLUMNS: list[str] = [
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
]


def make_prices(
    tickers: list[str],
    *,
    n: int = 400,
    start: date = date(2021, 1, 4),
    seed: int = 7,
    drifts: list[float] | None = None,
    vol: float = 0.01,
    start_price: float = 100.0,
) -> pd.DataFrame:
    """Build a deterministic multi-ticker long-form OHLCV frame.

    Each ticker follows a seeded geometric random walk (seed ``seed + j``) with
    its own daily drift, so tests can construct trending cross-sections.

    Args:
        tickers: Symbols to generate.
        n: Trading days per ticker (business-day spacing from ``start``).
        start: Calendar date of the first row.
        seed: Base RNG seed.
        drifts: Optional per-ticker daily drift (default ``0.0005`` each).
        vol: Daily volatility of the log-returns.
        start_price: Initial close.

    Returns:
        Long-form frame with :data:`OHLCV_COLUMNS`, sorted by ``(ticker, date)``.
    """
    frames = []
    for j, ticker in enumerate(tickers):
        rng = np.random.default_rng(seed + j)
        mu = 0.0005 if drifts is None else float(drifts[j])
        daily = rng.normal(loc=mu, scale=vol, size=n)
        close = start_price * np.exp(np.cumsum(daily))
        dates = pd.bdate_range(start=pd.Timestamp(start), periods=n)
        open_ = np.concatenate([[start_price], close[:-1]])
        span = np.abs(rng.normal(0.0, 0.5, size=n)) + 0.25
        frames.append(
            pd.DataFrame(
                {
                    "date": dates.date,
                    "ticker": ticker,
                    "open": open_,
                    "high": np.maximum(open_, close) + span,
                    "low": np.minimum(open_, close) - span,
                    "close": close,
                    "adj_close": close,
                    "volume": rng.integers(1_000_000, 5_000_000, size=n).astype("int64"),
                }
            )
        )
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)[OHLCV_COLUMNS]


@pytest.fixture
def prices() -> pd.DataFrame:
    """Three-ticker synthetic frame, 400 business days from 2021-01-04."""
    return make_prices(["AAA", "BBB", "CCC"])


def _two_asset_path() -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """A 30-day, two-asset linear price path (A rises, B falls)."""
    idx = pd.bdate_range("2025-01-01", periods=30)
    prices = pd.DataFrame(
        {"A": np.linspace(100.0, 130.0, 30), "B": np.linspace(50.0, 40.0, 30)}, index=idx
    )
    return prices, idx


# --------------------------------------------------------------------------- #
# pivot / slice / decision dates
# --------------------------------------------------------------------------- #
def test_to_timestamp_normalises_inputs() -> None:
    """Dates, strings and Timestamps all map to a naive midnight Timestamp."""
    expected = pd.Timestamp("2025-03-04")
    assert to_timestamp(date(2025, 3, 4)) == expected
    assert to_timestamp("2025-03-04") == expected
    assert to_timestamp(pd.Timestamp("2025-03-04 15:30")) == expected


def test_pivot_prices_shape_index_and_values(prices: pd.DataFrame) -> None:
    """The wide frame is date x sorted-ticker with a monotonic DatetimeIndex."""
    wide = pivot_prices(prices)
    assert isinstance(wide.index, pd.DatetimeIndex)
    assert wide.index.is_monotonic_increasing
    assert list(wide.columns) == ["AAA", "BBB", "CCC"]
    assert wide.shape == (400, 3)
    bbb = prices[prices["ticker"] == "BBB"]
    assert np.allclose(wide["BBB"].to_numpy(), bbb["adj_close"].to_numpy())


def test_pivot_prices_ffill_and_drops_empty_tickers(prices: pd.DataFrame) -> None:
    """Gaps stay NaN unless ``ffill=True``; all-NaN tickers are dropped."""
    frame = prices.copy()
    gap_date = sorted(frame["date"].unique())[5]
    frame.loc[(frame["ticker"] == "AAA") & (frame["date"] == gap_date), "adj_close"] = np.nan
    empty = frame[frame["ticker"] == "CCC"].copy()
    empty["ticker"] = "ZZZ"
    empty["adj_close"] = np.nan
    frame = pd.concat([frame, empty], ignore_index=True)

    wide = pivot_prices(frame)
    assert "ZZZ" not in wide.columns
    assert np.isnan(wide["AAA"].iloc[5])
    wide_f = pivot_prices(frame, ffill=True)
    assert wide_f["AAA"].iloc[5] == wide_f["AAA"].iloc[4]


def test_pivot_prices_validates_input(prices: pd.DataFrame) -> None:
    """Missing columns and empty frames raise ``ValueError``."""
    with pytest.raises(ValueError):
        pivot_prices(prices.drop(columns=["adj_close"]))
    with pytest.raises(ValueError):
        pivot_prices(prices.iloc[0:0])


def test_slice_window_is_start_inclusive_end_exclusive(prices: pd.DataFrame) -> None:
    """``slice_window`` accepts ``date`` bounds and uses ``[start, end)``."""
    wide = pivot_prices(prices)
    window = slice_window(wide, date(2021, 3, 1), date(2021, 4, 1))
    assert window.index[0] >= pd.Timestamp("2021-03-01")
    assert window.index[-1] < pd.Timestamp("2021-04-01")
    expected = ((wide.index >= "2021-03-01") & (wide.index < "2021-04-01")).sum()
    assert len(window) == expected
    assert len(slice_window(wide, None, None)) == len(wide)


def test_decision_dates_weekly_count_and_weekday() -> None:
    """Weekly decisions fall on the last trading day of each week (Fridays here)."""
    idx = pd.bdate_range("2021-01-04", periods=400)
    weekly = decision_dates(idx, "W-FRI")
    assert len(weekly) == len(idx.to_period("W-FRI").unique())
    assert all(d.weekday() == 4 for d in weekly[:-1])
    assert weekly[-1] == idx[-1]  # trailing partial week contributes its last day
    assert weekly == sorted(weekly) and len(set(weekly)) == len(weekly)
    assert all(isinstance(d, pd.Timestamp) for d in weekly)


def test_decision_dates_monthly_step_and_daily() -> None:
    """Monthly = last trading day per month; int = every n days; D = every day."""
    idx = pd.bdate_range("2021-01-04", periods=400)
    monthly = decision_dates(idx, "M")
    assert monthly == decision_dates(idx, "ME")
    assert len(monthly) == len(idx.to_period("M").unique())
    positions = idx.get_indexer(pd.DatetimeIndex(monthly))
    for pos in positions[:-1]:
        assert idx[pos + 1].month != idx[pos].month
    assert decision_dates(idx, 5) == list(idx[::5])
    assert decision_dates(idx, "D") == list(idx)
    assert decision_dates([], "W-FRI") == []


def test_decision_dates_rejects_bad_freq() -> None:
    """Unknown aliases and non-positive steps are rejected."""
    idx = pd.bdate_range("2021-01-04", periods=20)
    with pytest.raises(ValueError):
        decision_dates(idx, "X")
    with pytest.raises(ValueError):
        decision_dates(idx, 0)


# --------------------------------------------------------------------------- #
# simulate_weights
# --------------------------------------------------------------------------- #
def test_simulate_starts_at_capital_and_matches_hand_computation() -> None:
    """Day-1 equity equals the closed-form value; turnover matches the trades."""
    prices, idx = _two_asset_path()
    weights = pd.DataFrame({"A": [0.5, 1.0], "B": [0.5, 0.0]}, index=[idx[0], idx[10]])
    out = simulate_weights(prices, weights, cost_bps=10.0, initial_capital=1000.0)

    expected_keys = {
        "equity_curve",
        "daily_returns",
        "turnover",
        "costs",
        "trades",
        "weights",
        "holdings",
    }
    assert expected_keys <= set(out)
    eq = out["equity_curve"]
    assert eq.iloc[0] == 1000.0
    assert eq.index.equals(idx)
    ratio_a = prices["A"].iloc[1] / prices["A"].iloc[0]
    ratio_b = prices["B"].iloc[1] / prices["B"].iloc[0]
    assert eq.iloc[1] == pytest.approx(1000.0 * (1 - 0.001) * (0.5 * ratio_a + 0.5 * ratio_b))

    # First rebalance from cash: one-way turnover 1.0; cost = 10 bps of equity.
    assert out["turnover"].iloc[0] == pytest.approx(1.0)
    assert out["costs"].iloc[0] == pytest.approx(1.0)
    # Second rebalance: turnover is the distance from the drifted weights.
    trades = out["trades"]
    second = trades[trades["date"] == idx[10]].set_index("ticker")
    drifted_a, drifted_b = second.loc["A", "prev_weight"], second.loc["B", "prev_weight"]
    assert drifted_a + drifted_b == pytest.approx(1.0)
    assert out["turnover"].iloc[1] == pytest.approx(abs(1.0 - drifted_a) + abs(0.0 - drifted_b))
    assert len(out["daily_returns"]) == len(idx) - 1


def test_weights_drift_between_rebalances() -> None:
    """Between rebalances holdings follow the price-ratio drift and sum to one."""
    prices, idx = _two_asset_path()
    weights = pd.DataFrame({"A": [0.5], "B": [0.5]}, index=[idx[0]])
    out = simulate_weights(prices, weights, cost_bps=0.0, initial_capital=1000.0)
    holdings = out["holdings"]
    growth_a = prices["A"] / prices["A"].iloc[0]
    growth_b = prices["B"] / prices["B"].iloc[0]
    expected_a = 0.5 * growth_a / (0.5 * growth_a + 0.5 * growth_b)
    assert np.allclose(holdings["A"].to_numpy(), expected_a.to_numpy())
    assert np.allclose(holdings.sum(axis=1).to_numpy(), 1.0)
    # Zero-cost buy-and-hold equity equals the weighted price growth exactly.
    assert np.allclose(
        out["equity_curve"].to_numpy(), 1000.0 * (0.5 * growth_a + 0.5 * growth_b).to_numpy()
    )


def test_costs_reduce_terminal_equity(prices: pd.DataFrame) -> None:
    """Higher turnover costs monotonically lower the final equity."""
    wide = pivot_prices(prices)
    dates = decision_dates(wide.index, "W-FRI")
    rng = np.random.default_rng(0)
    raw = rng.random((len(dates), 3))
    weights = pd.DataFrame(raw / raw.sum(axis=1, keepdims=True), index=dates, columns=wide.columns)
    finals = [
        simulate_weights(wide, weights, cost_bps=c)["equity_curve"].iloc[-1]
        for c in (0.0, 10.0, 50.0)
    ]
    assert finals[0] > finals[1] > finals[2]


def test_cash_when_no_weights(prices: pd.DataFrame) -> None:
    """All-zero weights keep the portfolio flat in cash with zero turnover."""
    wide = pivot_prices(prices)
    weights = pd.DataFrame(0.0, index=[wide.index[0]], columns=wide.columns)
    out = simulate_weights(wide, weights, cost_bps=10.0, initial_capital=500.0)
    assert (out["equity_curve"] == 500.0).all()
    assert (out["daily_returns"] == 0.0).all()
    assert out["turnover"].sum() == 0.0
    assert out["trades"].empty


def test_rejects_shorts_and_leverage() -> None:
    """Negative weights or rows summing above one are rejected."""
    prices, idx = _two_asset_path()
    with pytest.raises(ValueError):
        simulate_weights(prices, pd.DataFrame({"A": [-0.2], "B": [0.5]}, index=[idx[0]]))
    with pytest.raises(ValueError):
        simulate_weights(prices, pd.DataFrame({"A": [0.8], "B": [0.5]}, index=[idx[0]]))
    with pytest.raises(ValueError):
        simulate_weights(prices.iloc[0:0], pd.DataFrame({"A": [1.0]}, index=[idx[0]]))


def test_rebalance_on_non_trading_day_snaps_forward() -> None:
    """A weekend rebalance date becomes the next trading day; before it we hold cash."""
    prices, idx = _two_asset_path()
    saturday = pd.Timestamp("2025-01-04")
    assert saturday.weekday() == 5
    out = simulate_weights(prices, pd.DataFrame({"A": [1.0]}, index=[saturday]), cost_bps=0.0)
    monday = pd.Timestamp("2025-01-06")
    assert list(out["weights"].index) == [monday]
    assert (out["equity_curve"].loc[:monday] == 100_000.0).all()
    assert out["equity_curve"].iloc[-1] > 100_000.0


def test_rebalance_on_final_day_is_ignored() -> None:
    """A rebalance with no subsequent mark is dropped rather than charged."""
    prices, idx = _two_asset_path()
    out = simulate_weights(prices, pd.DataFrame({"A": [1.0]}, index=[idx[-1]]), cost_bps=10.0)
    assert out["trades"].empty
    assert (out["equity_curve"] == 100_000.0).all()


def test_missing_price_ticker_gets_zero_weight() -> None:
    """A ticker with no price on the rebalance date is left in cash."""
    prices, idx = _two_asset_path()
    prices = prices.copy()
    prices.loc[idx[:5], "B"] = np.nan
    weights = pd.DataFrame({"A": [0.5], "B": [0.5]}, index=[idx[0]])
    out = simulate_weights(prices, weights, cost_bps=10.0, initial_capital=1000.0)
    assert out["weights"].loc[idx[0], "B"] == 0.0
    ratio_a = prices["A"].iloc[1] / prices["A"].iloc[0]
    assert out["equity_curve"].iloc[1] == pytest.approx(
        1000.0 * (1 - 0.001 * 0.5) * (0.5 + 0.5 * ratio_a)
    )


# --------------------------------------------------------------------------- #
# summarise and friends
# --------------------------------------------------------------------------- #
def test_summarise_keys_and_sanity(prices: pd.DataFrame) -> None:
    """The summary carries every headline metric with sensible values."""
    wide = pivot_prices(prices)
    weights = pd.DataFrame({"AAA": [0.4], "BBB": [0.3], "CCC": [0.3]}, index=[wide.index[0]])
    out = simulate_weights(wide, weights)
    stats = summarise(out["equity_curve"], out["daily_returns"], n_trials=3)
    expected = {
        "total_return",
        "cagr",
        "annualised_vol",
        "sharpe",
        "sortino",
        "max_drawdown",
        "deflated_sharpe",
        "calmar",
        "n_days",
        "n_trials",
        "initial_equity",
        "final_equity",
    }
    assert expected <= set(stats)
    assert stats["n_days"] == len(wide)
    assert stats["n_trials"] == 3
    assert stats["total_return"] == pytest.approx(
        out["equity_curve"].iloc[-1] / out["equity_curve"].iloc[0] - 1.0
    )
    assert 0.0 <= stats["deflated_sharpe"] <= 1.0
    assert stats["max_drawdown"] <= 0.0
    assert stats["annualised_vol"] > 0.0
    # More trials compared -> a harder hurdle -> a lower deflated Sharpe.
    fewer = summarise(out["equity_curve"], out["daily_returns"], n_trials=1)["deflated_sharpe"]
    assert fewer >= stats["deflated_sharpe"]


def test_summarise_degenerate_curves() -> None:
    """Monotone curves have no drawdown (Calmar nan); empty input is all-nan."""
    idx = pd.bdate_range("2025-01-01", periods=10)
    curve = pd.Series(np.linspace(100.0, 110.0, 10), index=idx)
    stats = summarise(curve, curve.pct_change().dropna())
    assert stats["max_drawdown"] == 0.0
    assert np.isnan(stats["calmar"])
    empty = summarise(pd.Series(dtype="float64"), pd.Series(dtype="float64"))
    assert empty["n_days"] == 0 and np.isnan(empty["sharpe"])


def test_drawdown_series_and_period_returns() -> None:
    """Drawdown minimum equals ``max_drawdown``; period returns compound to the total."""
    idx = pd.bdate_range("2025-01-01", periods=60)
    curve = pd.Series(100.0 + 10.0 * np.sin(np.arange(60) / 5.0), index=idx)
    dd = drawdown_series(curve)
    assert dd.max() <= 0.0
    assert dd.min() == pytest.approx(max_drawdown(curve))
    monthly = period_returns(curve.pct_change().dropna(), "M")
    assert isinstance(monthly.index, pd.PeriodIndex)
    assert np.prod(1.0 + monthly.to_numpy()) - 1.0 == pytest.approx(
        curve.iloc[-1] / curve.iloc[0] - 1.0
    )
    assert period_returns(pd.Series(dtype="float64")).empty
