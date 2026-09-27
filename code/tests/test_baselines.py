"""Tests for ``advisor.evaluation.baselines`` -- the four comparison baselines.

Every baseline must produce an equity curve on the evaluation window's
trading calendar, start at the initial capital and be reproducible. Strategy
logic is checked on constructed data: momentum must long the strongest
trending name, Markowitz weights must lie on the simplex and switch to
minimum-variance when no asset has a positive mean, the random ensemble must
yield ordered percentile bands, and signals must be point-in-time (perturbing
the future must not change today's weights). All offline and deterministic.
"""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd
import pytest

from advisor.evaluation.baselines import (
    BASELINE_NAMES,
    buy_and_hold,
    markowitz_mean_variance,
    momentum_12_1,
    permutation_p_value,
    random_allocation,
    random_allocation_ensemble,
    run_all_baselines,
)
from advisor.evaluation.portfolio import decision_dates, pivot_prices, slice_window
from tests.test_portfolio import make_prices

START = date(2022, 6, 1)
END = date(2023, 6, 1)
TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE"]
N_DAYS = 760  # ~3 years of business days from 2020-09-01: history + window


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    """Five-ticker synthetic universe with history well before the window."""
    return make_prices(TICKERS, n=N_DAYS, start=date(2020, 9, 1), seed=11)


@pytest.fixture(scope="module")
def spy() -> pd.DataFrame:
    """Synthetic benchmark series on the same calendar."""
    return make_prices(["SPY"], n=N_DAYS, start=date(2020, 9, 1), seed=99)


def _window_index(frame: pd.DataFrame) -> pd.DatetimeIndex:
    return slice_window(pivot_prices(frame), START, END).index


# --------------------------------------------------------------------------- #
# Shared contract
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "fn", [buy_and_hold, random_allocation, momentum_12_1, markowitz_mean_variance]
)
def test_curve_has_window_index_and_starts_at_capital(prices: pd.DataFrame, fn) -> None:
    """Each baseline returns a named, positive equity Series on the window calendar."""
    curve = fn(prices, start=START, end=END)
    assert isinstance(curve, pd.Series)
    assert curve.index.equals(_window_index(prices))
    assert curve.iloc[0] == 100_000.0
    assert curve.name == fn.__name__
    assert np.isfinite(curve.to_numpy()).all() and (curve > 0).all()


@pytest.mark.parametrize(
    "fn", [buy_and_hold, random_allocation, momentum_12_1, markowitz_mean_variance]
)
def test_return_details_contract(prices: pd.DataFrame, fn) -> None:
    """``return_details=True`` yields the simulation dict plus a summary."""
    details = fn(prices, start=START, end=END, return_details=True)
    for key in (
        "equity_curve",
        "daily_returns",
        "turnover",
        "trades",
        "weights",
        "summary",
        "name",
    ):
        assert key in details
    assert details["name"] == fn.__name__
    assert "deflated_sharpe" in details["summary"]
    assert details["equity_curve"].iloc[0] == 100_000.0


def test_baselines_are_deterministic(prices: pd.DataFrame) -> None:
    """Running a baseline twice gives identical curves."""
    for fn in (momentum_12_1, markowitz_mean_variance, random_allocation):
        first = fn(prices, start=START, end=END)
        second = fn(prices, start=START, end=END)
        pd.testing.assert_series_equal(first, second)


# --------------------------------------------------------------------------- #
# Buy-and-hold
# --------------------------------------------------------------------------- #
def test_buy_and_hold_uses_benchmark_and_never_rebalances(
    prices: pd.DataFrame, spy: pd.DataFrame
) -> None:
    """With a benchmark frame the curve tracks SPY after a single 10 bps cost."""
    details = buy_and_hold(
        prices, benchmark_prices=spy, start=START, end=END, cost_bps=10.0, return_details=True
    )
    assert details["mode"] == "benchmark"
    assert details["tickers"] == ["SPY"]
    assert len(details["turnover"]) == 1
    assert details["weights"].iloc[0]["SPY"] == 1.0
    spy_window = slice_window(pivot_prices(spy), START, END)["SPY"]
    expected = 100_000.0 * (1 - 0.001) * spy_window / spy_window.iloc[0]
    assert np.allclose(details["equity_curve"].iloc[1:].to_numpy(), expected.iloc[1:].to_numpy())


def test_buy_and_hold_falls_back_to_equal_weight(
    prices: pd.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    """Without a benchmark: equal weight across the universe, and a warning."""
    with caplog.at_level(logging.WARNING, logger="advisor.evaluation.baselines"):
        details = buy_and_hold(prices, start=START, end=END, return_details=True)
    assert details["mode"] == "equal_weight_universe"
    assert sorted(details["tickers"]) == TICKERS
    assert np.allclose(details["weights"].iloc[0].to_numpy(), 0.2)
    assert len(details["turnover"]) == 1
    assert any("falling back" in rec.message for rec in caplog.records)


# --------------------------------------------------------------------------- #
# Random allocation + ensemble
# --------------------------------------------------------------------------- #
def test_random_allocation_is_seeded_and_equal_weight(prices: pd.DataFrame) -> None:
    """Same seed -> identical curve; each week holds ``n_holdings`` names at 1/n."""
    a = random_allocation(prices, start=START, end=END, seed=1, n_holdings=2, return_details=True)
    b = random_allocation(prices, start=START, end=END, seed=1, n_holdings=2, return_details=True)
    c = random_allocation(prices, start=START, end=END, seed=2, n_holdings=2)
    pd.testing.assert_series_equal(a["equity_curve"], b["equity_curve"])
    assert not np.allclose(a["equity_curve"].to_numpy(), c.to_numpy())

    weights = a["weights"]
    assert ((weights > 0).sum(axis=1) == 2).all()
    values = weights.to_numpy()
    assert np.allclose(values[values > 0], 0.5)
    expected_dates = decision_dates(_window_index(prices), "W-FRI")
    first = _window_index(prices)[0]
    if expected_dates[0] != first:
        expected_dates.insert(0, first)
    # The final decision date carries no subsequent mark and is dropped.
    assert len(weights) in (len(expected_dates), len(expected_dates) - 1)
    assert a["seed"] == 1 and a["n_holdings"] == 2


def test_random_ensemble_percentiles_ordered(prices: pd.DataFrame) -> None:
    """Ensemble bands are ordered and every member reports a terminal Sharpe."""
    ens = random_allocation_ensemble(prices, n_seeds=8, seed=100, start=START, end=END)
    pct = ens["percentiles"]
    assert (pct["p05"] <= pct["p50"] + 1e-12).all()
    assert (pct["p50"] <= pct["p95"] + 1e-12).all()
    assert len(ens["terminal_sharpe"]) == 8
    assert list(ens["terminal_sharpe"].index) == list(range(100, 108))
    assert ens["seeds"] == list(range(100, 108))
    assert ens["mean_equity_curve"].index.equals(_window_index(prices))
    assert ens["curves"].shape == (len(_window_index(prices)), 8)
    assert {"sharpe", "sortino", "total_return", "max_drawdown"} <= set(
        ens["terminal_stats"].columns
    )
    with pytest.raises(ValueError):
        random_allocation_ensemble(prices, n_seeds=0, start=START, end=END)


def test_permutation_p_value() -> None:
    """Add-one one-sided p-value against a null sample."""
    null = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0])
    assert permutation_p_value(2.5, null) == pytest.approx(3 / 6)
    assert permutation_p_value(10.0, null) == pytest.approx(1 / 6)
    assert permutation_p_value(-1.0, null) == pytest.approx(1.0)
    assert np.isnan(permutation_p_value(1.0, pd.Series(dtype="float64")))
    assert np.isnan(permutation_p_value(float("nan"), null))


# --------------------------------------------------------------------------- #
# 12-1 momentum
# --------------------------------------------------------------------------- #
def test_momentum_longs_top_names_in_trending_data() -> None:
    """With clearly separated drifts the top quantile is the strongest trender."""
    drifts = [0.004, 0.0, -0.004, 0.0005, -0.0005]
    trending = make_prices(
        TICKERS, n=N_DAYS, start=date(2020, 9, 1), seed=5, drifts=drifts, vol=0.005
    )

    one = momentum_12_1(trending, start=START, end=END, top_quantile=0.2, return_details=True)
    assert (one["n_valid"] > 0).all()
    assert (one["weights"]["AAA"] == 1.0).all()
    assert (one["weights"].drop(columns=["AAA"]) == 0.0).all().all()

    two = momentum_12_1(trending, start=START, end=END, top_quantile=0.4, return_details=True)
    assert (two["weights"][["AAA", "DDD"]] == 0.5).all().all()
    assert (two["weights"].drop(columns=["AAA", "DDD"]) == 0.0).all().all()
    assert one["lookback_days"] == 252 and one["skip_days"] == 21


def test_momentum_rebalances_monthly(prices: pd.DataFrame) -> None:
    """Weights are set on the first window day and then at month-ends."""
    details = momentum_12_1(prices, start=START, end=END, return_details=True)
    monthly = decision_dates(_window_index(prices), "M")
    first = _window_index(prices)[0]
    expected = [first, *[d for d in monthly if d != first]]
    assert list(details["weights"].index) == expected[: len(details["weights"])]


def test_momentum_without_history_sits_in_cash(
    prices: pd.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    """No pre-window history -> no valid score -> cash, with a warning."""
    short = prices[pd.to_datetime(prices["date"]) >= pd.Timestamp(START)]
    with caplog.at_level(logging.WARNING, logger="advisor.evaluation.baselines"):
        details = momentum_12_1(short, start=START, end=END, return_details=True)
    assert details["n_valid"].iloc[0] == 0
    assert (details["equity_curve"].iloc[:200] == 100_000.0).all()
    assert any("history" in rec.message for rec in caplog.records)


def test_momentum_rejects_bad_parameters(prices: pd.DataFrame) -> None:
    """Quantile outside (0, 1] or lookback <= skip is an error."""
    with pytest.raises(ValueError):
        momentum_12_1(prices, start=START, end=END, top_quantile=0.0)
    with pytest.raises(ValueError):
        momentum_12_1(prices, start=START, end=END, lookback_days=10, skip_days=21)


def test_momentum_signal_is_point_in_time(prices: pd.DataFrame) -> None:
    """Changing prices after a decision date leaves that date's weights unchanged."""
    base = momentum_12_1(prices, start=START, end=END, return_details=True)
    cutoff = base["weights"].index[2]
    shocked = prices.copy()
    later = pd.to_datetime(shocked["date"]) > cutoff
    shocked.loc[later, "adj_close"] = shocked.loc[later, "adj_close"] * np.where(
        shocked.loc[later, "ticker"] == "CCC", 3.0, 0.5
    )
    other = momentum_12_1(shocked, start=START, end=END, return_details=True)
    pd.testing.assert_frame_equal(base["weights"].loc[:cutoff], other["weights"].loc[:cutoff])


# --------------------------------------------------------------------------- #
# Markowitz mean-variance
# --------------------------------------------------------------------------- #
def test_markowitz_weights_on_simplex(prices: pd.DataFrame) -> None:
    """Long-only weights sum to one whenever invested; the optimiser converges."""
    details = markowitz_mean_variance(prices, start=START, end=END, return_details=True)
    weights = details["target_weights"]
    sums = weights.sum(axis=1)
    invested = details["modes"] != "cash"
    assert invested.all()
    assert np.allclose(sums[invested].to_numpy(), 1.0, atol=1e-6)
    assert (weights.to_numpy() >= -1e-9).all()
    assert set(details["modes"].unique()) <= {"max_sharpe", "min_variance"}
    assert details["optimiser_success"].all()
    assert (details["n_assets"] == 5).all()


def test_markowitz_min_variance_when_all_means_negative() -> None:
    """Without a positive expected excess return the min-variance portfolio is used."""
    bearish = make_prices(TICKERS, n=N_DAYS, start=date(2020, 9, 1), seed=8, drifts=[-0.003] * 5)
    details = markowitz_mean_variance(bearish, start=START, end=END, return_details=True)
    assert (details["modes"] == "min_variance").all()
    weights = details["target_weights"]
    assert np.allclose(weights.sum(axis=1).to_numpy(), 1.0, atol=1e-6)
    assert (weights.to_numpy() >= -1e-9).all()


def test_markowitz_prefers_high_sharpe_asset() -> None:
    """The tangency portfolio leans on the asset with the best risk/return."""
    drifts = [0.003, 0.0, 0.0, 0.0, 0.0]
    skewed = make_prices(
        TICKERS, n=N_DAYS, start=date(2020, 9, 1), seed=13, drifts=drifts, vol=0.01
    )
    details = markowitz_mean_variance(skewed, start=START, end=END, return_details=True)
    weights = details["target_weights"]
    assert (weights["AAA"] > weights.drop(columns=["AAA"]).max(axis=1)).all()


def test_markowitz_signal_is_point_in_time(prices: pd.DataFrame) -> None:
    """Future prices must not influence today's mean-variance weights."""
    base = markowitz_mean_variance(prices, start=START, end=END, return_details=True)
    cutoff = base["target_weights"].index[1]
    shocked = prices.copy()
    later = pd.to_datetime(shocked["date"]) > cutoff
    shocked.loc[later, "adj_close"] = shocked.loc[later, "adj_close"] * 0.5
    other = markowitz_mean_variance(shocked, start=START, end=END, return_details=True)
    pd.testing.assert_frame_equal(
        base["target_weights"].loc[:cutoff], other["target_weights"].loc[:cutoff]
    )


def test_markowitz_insufficient_history_sits_in_cash(
    prices: pd.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    """Fewer than ``min_history_days`` rows before the first date -> cash."""
    short = prices[pd.to_datetime(prices["date"]) >= pd.Timestamp(START) - pd.Timedelta(days=10)]
    with caplog.at_level(logging.WARNING, logger="advisor.evaluation.baselines"):
        details = markowitz_mean_variance(short, start=START, end=END, return_details=True)
    assert details["modes"].iloc[0] == "cash"
    assert any("insufficient history" in rec.message for rec in caplog.records)
    with pytest.raises(ValueError):
        markowitz_mean_variance(prices, start=START, end=END, lookback_days=1)


# --------------------------------------------------------------------------- #
# Convenience runner
# --------------------------------------------------------------------------- #
def test_run_all_baselines_keys(prices: pd.DataFrame, spy: pd.DataFrame) -> None:
    """The bundle contains the four baselines and the random ensemble."""
    out = run_all_baselines(prices, benchmark_prices=spy, start=START, end=END, n_seeds=3)
    assert set(out) == set(BASELINE_NAMES) | {"random_ensemble"}
    assert out["buy_and_hold"]["mode"] == "benchmark"
    assert out["random_ensemble"]["n_seeds"] == 3
    for name in BASELINE_NAMES:
        assert out[name]["equity_curve"].index.equals(_window_index(prices))
