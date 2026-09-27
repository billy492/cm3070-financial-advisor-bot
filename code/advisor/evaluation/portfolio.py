"""Shared portfolio-simulation helpers for the evaluation layer.

Every strategy in the evaluation -- the LLM advisor's dynamic portfolio, the
four baselines of ADR-0002 and the random null ensemble -- is reduced to the
same primitive: a table of *target weights* indexed by rebalance dates. This
module turns such a table into an equity curve under one shared, documented
cost model, so that all strategies are compared like-for-like (Arnott, Harvey &
Markowitz 2019, protocol point on realistic transaction costs), and computes
the summary statistics reported in the thesis (Sharpe, Sortino, maximum
drawdown, and the Deflated Sharpe Ratio of Bailey & Lopez de Prado 2014).

Simulation semantics (see :func:`simulate_weights`):

* Target weights are set at the **close** of each rebalance date and held
  until the next rebalance date. Between rebalances the weights drift with
  prices (no intra-period rebalancing).
* The portfolio is marked to market daily using ``adj_close`` returns.
* Transaction cost is ``cost_bps`` basis points of the one-way turnover
  ``sum |w_target - w_drifted|`` at each rebalance; cash earns 0.
* Equity on a rebalance date is the pre-trade mark; the cost is deducted from
  the value carried into the next day, so the curve starts at exactly
  ``initial_capital`` and every cost is still fully charged.
* Missing prices are forward-filled within a ticker (a stale price is the only
  point-in-time-safe fill); tickers with no data are dropped, and a ticker
  without a valid price on a rebalance date receives weight 0 (that capital
  stays in cash).

All functions are pure pandas/numpy, vectorised within rebalance segments, and
deterministic.

References:
    Bailey, D. H. & Lopez de Prado, M. (2014). The Deflated Sharpe Ratio:
        Correcting for Selection Bias, Backtest Overfitting and Non-Normality.
        *Journal of Portfolio Management*, 40(5), 94-107.
    Arnott, R., Harvey, C. R. & Markowitz, H. (2019). A Backtesting Protocol
        in the Era of Machine Learning. *Journal of Financial Data Science*,
        1(1), 64-74.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from advisor.evaluation.metrics import (
    deflated_sharpe_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)

__all__ = [
    "TRADING_DAYS_PER_YEAR",
    "to_timestamp",
    "pivot_prices",
    "slice_window",
    "decision_dates",
    "simulate_weights",
    "summarise",
    "drawdown_series",
    "period_returns",
]

#: Annualisation constant for daily US-equity returns.
TRADING_DAYS_PER_YEAR: int = 252

#: Weight tolerance used when validating long-only, unlevered weight tables.
_WEIGHT_TOL: float = 1e-6

_MONTHLY_ALIASES = frozenset({"M", "ME", "BM", "BME", "MONTH", "MONTHLY"})
_QUARTERLY_ALIASES = frozenset({"Q", "QE", "BQ", "BQE", "QUARTER", "QUARTERLY"})
_DAILY_ALIASES = frozenset({"D", "B", "DAY", "DAILY"})


def to_timestamp(value: date | datetime | str | pd.Timestamp) -> pd.Timestamp:
    """Coerce a date-like value to a timezone-naive midnight ``pd.Timestamp``.

    pandas 3 refuses to compare ``datetime64`` indexes with ``datetime.date``
    objects, so every date boundary in the evaluation layer is normalised
    through this helper before being used in a comparison or lookup.

    Args:
        value: A ``datetime.date``, ``datetime.datetime``, ISO string or
            ``pd.Timestamp``.

    Returns:
        The equivalent ``pd.Timestamp`` at midnight, without timezone.
    """
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts.normalize()


def pivot_prices(
    prices_long: pd.DataFrame,
    column: str = "adj_close",
    *,
    ffill: bool = False,
) -> pd.DataFrame:
    """Pivot a long-form price frame into a wide ``date x ticker`` matrix.

    Args:
        prices_long: Long-form frame with at least ``date``, ``ticker`` and
            ``column`` columns (the schema emitted by
            :func:`advisor.data.loader.load_prices`).
        column: Price field to pivot (default ``adj_close``, the total-return
            series used for mark-to-market).
        ffill: When ``True``, forward-fill gaps within each ticker. Forward
            filling only ever uses *past* observations, so it introduces no
            look-ahead; leading gaps (before a ticker's first print) stay NaN.

    Returns:
        A ``pd.DataFrame`` indexed by a sorted ``DatetimeIndex`` named
        ``date`` with one float column per ticker (sorted). Tickers with no
        finite price at all are dropped.

    Raises:
        ValueError: If a required column is missing or the frame is empty.
    """
    required = {"date", "ticker", column}
    missing = required - set(prices_long.columns)
    if missing:
        raise ValueError(f"pivot_prices: missing columns {sorted(missing)!r}")
    if len(prices_long) == 0:
        raise ValueError("pivot_prices: received an empty price frame.")

    frame = prices_long.loc[:, ["date", "ticker", column]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["ticker"] = frame["ticker"].astype(str)
    frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    # Keep the last print when a (date, ticker) pair is duplicated.
    frame = frame.drop_duplicates(subset=["date", "ticker"], keep="last")

    wide = frame.pivot(index="date", columns="ticker", values=column)
    wide = wide.sort_index()
    wide = wide.reindex(columns=sorted(wide.columns))
    wide.index.name = "date"
    wide.columns.name = None
    wide = wide.dropna(axis=1, how="all")
    if ffill:
        wide = wide.ffill()
    return wide


def slice_window(
    prices_wide: pd.DataFrame,
    start: date | str | pd.Timestamp | None,
    end: date | str | pd.Timestamp | None,
) -> pd.DataFrame:
    """Restrict a wide price frame to ``start <= date < end``.

    The end bound is exclusive, matching the convention of
    :func:`advisor.data.loader.load_prices` and :data:`advisor.config.TEST_END`.

    Args:
        prices_wide: Wide frame indexed by ``DatetimeIndex``.
        start: Inclusive start (``None`` = from the first row).
        end: Exclusive end (``None`` = through the last row).

    Returns:
        The sliced frame (a copy).
    """
    idx = pd.DatetimeIndex(prices_wide.index)
    mask = np.ones(len(idx), dtype=bool)
    if start is not None:
        mask &= idx >= to_timestamp(start)
    if end is not None:
        mask &= idx < to_timestamp(end)
    return prices_wide.loc[mask].copy()


def decision_dates(
    index: Iterable[Any],
    freq: str | int = "W-FRI",
) -> list[pd.Timestamp]:
    """Return the decision/rebalance dates implied by a trading calendar.

    Args:
        index: Trading dates (typically ``prices_wide.index``). Unsorted or
            duplicated inputs are sorted and de-duplicated.
        freq: One of

            * ``"W-FRI"`` (default) or any ``"W-<DAY>"`` anchor: the **last
              trading day of each week** ending on that weekday (so a Friday
              holiday yields the Thursday);
            * ``"M"`` / ``"ME"``: the last trading day of each calendar month;
            * ``"Q"`` / ``"QE"``: the last trading day of each calendar quarter;
            * ``"D"`` / ``"B"``: every trading day;
            * an ``int`` ``n``: every ``n``-th trading day, starting from the
              first date.

            A trailing partial period (e.g. the window ends mid-week) still
            contributes its last available day.

    Returns:
        A chronologically sorted list of ``pd.Timestamp`` decision dates.

    Raises:
        ValueError: For an unknown frequency string or a non-positive step.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(list(index))).unique().sort_values()
    if len(idx) == 0:
        return []

    if isinstance(freq, (int, np.integer)) and not isinstance(freq, bool):
        step = int(freq)
        if step < 1:
            raise ValueError(f"decision_dates: step must be >= 1, got {step}.")
        return [pd.Timestamp(ts) for ts in idx[::step]]

    if not isinstance(freq, str):
        raise ValueError(f"decision_dates: unsupported freq {freq!r}.")

    key = freq.strip().upper()
    if key in _DAILY_ALIASES:
        return [pd.Timestamp(ts) for ts in idx]
    if key in _MONTHLY_ALIASES:
        periods = idx.to_period("M")
    elif key in _QUARTERLY_ALIASES:
        periods = idx.to_period("Q")
    elif key.startswith("W"):
        anchor = key if "-" in key else "W-FRI"
        periods = idx.to_period(anchor)
    else:
        raise ValueError(
            f"decision_dates: unknown freq {freq!r}; use 'W-FRI', 'M', 'Q', 'D' or an int."
        )

    positions = pd.Series(np.arange(len(idx)), index=periods)
    last_positions = positions.groupby(level=0, sort=True).max().to_numpy()
    return [pd.Timestamp(idx[p]) for p in np.sort(last_positions)]


def _validate_weights(weights: pd.DataFrame) -> pd.DataFrame:
    """Check a weight table is long-only and unlevered; clean tiny numerical noise.

    Args:
        weights: Target weights (rows = rebalance dates, columns = tickers).

    Returns:
        A float copy with negative round-off clipped to zero.

    Raises:
        ValueError: If any weight is materially negative or any row sums to
            more than one (short positions and leverage are out of scope for
            a recommendation-only research system).
    """
    clean = weights.astype("float64").fillna(0.0)
    if (clean.to_numpy() < -_WEIGHT_TOL).any():
        raise ValueError("simulate_weights: negative weights are not supported (long-only).")
    row_sums = clean.sum(axis=1)
    if (row_sums > 1.0 + _WEIGHT_TOL).any():
        worst = float(row_sums.max())
        raise ValueError(
            f"simulate_weights: weights must sum to <= 1 per rebalance (max {worst:.6f})."
        )
    clean = clean.clip(lower=0.0)
    # Scale rows that exceed one by round-off back onto the simplex boundary.
    over = row_sums > 1.0
    if over.any():
        clean.loc[over] = clean.loc[over].div(row_sums[over], axis=0)
    return clean


def simulate_weights(
    prices_wide: pd.DataFrame,
    weights: pd.DataFrame,
    cost_bps: float = 10.0,
    initial_capital: float = 100_000.0,
) -> dict[str, Any]:
    """Mark a rebalanced long-only portfolio to market under a turnover cost model.

    Semantics are documented in the module docstring. In brief: on each
    rebalance date ``d_k`` the target weights ``w_k`` are established at the
    close; for every subsequent day ``t`` up to and including the next
    rebalance date the equity is

    ``V(t) = V(d_k) * (1 - c * turnover_k) * [ (1 - sum(w_k)) + sum_i w_k,i * P_i(t) / P_i(d_k) ]``

    where ``c = cost_bps / 10_000`` and ``turnover_k = sum_i |w_k,i - w^drift_i(d_k)|``
    is the one-way turnover against the weights that had drifted from the
    previous target. Segments are vectorised in numpy; the only Python loop
    is over rebalance dates.

    Args:
        prices_wide: Wide ``date x ticker`` price matrix (see
            :func:`pivot_prices`). Gaps are forward-filled per ticker; tickers
            with no data are dropped. The equity curve spans this index.
        weights: Target weights indexed by rebalance dates (any date-like);
            columns are tickers. Missing tickers are treated as weight 0;
            columns absent from ``prices_wide`` are ignored. Rows must be
            non-negative and sum to at most 1 (the remainder is cash). A
            rebalance date that is not a trading day snaps to the next trading
            day; one falling on or after the final day is ignored because no
            subsequent mark exists.
        cost_bps: Transaction cost in basis points of one-way turnover.
        initial_capital: Starting portfolio value (held in cash until the first
            rebalance).

    Returns:
        A dict with keys

        * ``equity_curve``: ``pd.Series`` of portfolio value per trading day
          (starts at ``initial_capital``).
        * ``daily_returns``: ``pd.Series`` of simple daily returns (the first
          day is dropped).
        * ``turnover``: ``pd.Series`` of one-way turnover per effective
          rebalance date.
        * ``costs``: ``pd.Series`` of cash cost per effective rebalance date.
        * ``trades``: ``pd.DataFrame`` with one row per non-zero weight change
          (``date, ticker, prev_weight, target_weight, delta_weight, notional``).
        * ``weights``: the effective target weights actually used (aligned,
          invalid tickers zeroed).
        * ``holdings``: ``pd.DataFrame`` of drifted weights at every close.

    Raises:
        ValueError: On an empty price frame, invalid weights, or a mismatched
            weight table with no usable tickers.
    """
    if prices_wide is None or len(prices_wide) == 0:
        raise ValueError("simulate_weights: prices_wide is empty.")
    if cost_bps < 0:
        raise ValueError("simulate_weights: cost_bps must be non-negative.")
    if initial_capital <= 0:
        raise ValueError("simulate_weights: initial_capital must be positive.")

    prices = prices_wide.astype("float64").sort_index()
    prices.index = pd.DatetimeIndex(pd.to_datetime(prices.index))
    prices = prices.dropna(axis=1, how="all").ffill()
    idx = prices.index
    n_days = len(idx)
    tickers = list(prices.columns)
    n_assets = len(tickers)

    # --- Align the weight table to the price calendar ------------------------
    w_table = weights.copy()
    w_table.index = pd.DatetimeIndex(pd.to_datetime(w_table.index))
    w_table = w_table.sort_index()
    w_table = w_table.reindex(columns=tickers, fill_value=0.0)
    w_table = _validate_weights(w_table)

    snapped = idx.searchsorted(w_table.index.to_numpy(), side="left")
    usable = snapped < (n_days - 1)  # need at least one subsequent mark
    w_table = w_table.iloc[usable]
    snapped = snapped[usable]
    if len(w_table) > 0:
        w_table.index = idx[snapped]
        w_table = w_table[~w_table.index.duplicated(keep="last")]
        positions = np.unique(snapped)
    else:
        positions = np.array([], dtype=int)

    price_arr = prices.to_numpy()
    target_arr = w_table.to_numpy() if len(w_table) else np.zeros((0, n_assets))
    cost_rate = float(cost_bps) / 10_000.0

    equity = np.full(n_days, float(initial_capital), dtype="float64")
    holdings = np.zeros((n_days, n_assets), dtype="float64")
    turnover = np.zeros(len(positions), dtype="float64")
    costs = np.zeros(len(positions), dtype="float64")
    used_targets = np.zeros((len(positions), n_assets), dtype="float64")
    trade_rows: list[dict[str, Any]] = []

    drifted = np.zeros(n_assets, dtype="float64")  # weights just before rebalancing
    for k, p in enumerate(positions):
        value_pre = equity[p]
        target = target_arr[k].copy()
        valid = np.isfinite(price_arr[p])
        target[~valid] = 0.0
        used_targets[k] = target

        delta = target - drifted
        turn = float(np.abs(delta).sum())
        cost = cost_rate * turn * value_pre
        value_post = value_pre - cost
        turnover[k] = turn
        costs[k] = cost

        for i in np.flatnonzero(np.abs(delta) > 1e-12):
            trade_rows.append(
                {
                    "date": idx[p],
                    "ticker": tickers[i],
                    "prev_weight": float(drifted[i]),
                    "target_weight": float(target[i]),
                    "delta_weight": float(delta[i]),
                    "notional": float(delta[i] * value_pre),
                }
            )

        seg_end = positions[k + 1] if k + 1 < len(positions) else n_days - 1
        holdings[p] = target
        if seg_end > p:
            base = price_arr[p]
            growth = price_arr[p + 1 : seg_end + 1] / base  # (m x n_assets)
            growth = np.where(target > 0.0, growth, 0.0)
            growth = np.nan_to_num(growth, nan=0.0, posinf=0.0, neginf=0.0)
            asset_values = value_post * (growth * target)  # dollar value per asset
            cash_value = value_post * (1.0 - target.sum())
            total = cash_value + asset_values.sum(axis=1)
            equity[p + 1 : seg_end + 1] = total
            with np.errstate(divide="ignore", invalid="ignore"):
                seg_holdings = np.where(total[:, None] > 0.0, asset_values / total[:, None], 0.0)
            holdings[p + 1 : seg_end + 1] = seg_holdings
            drifted = seg_holdings[-1].copy()
        else:  # pragma: no cover - guarded by the `usable` mask above
            drifted = target.copy()

    equity_curve = pd.Series(equity, index=idx, name="equity")
    daily_returns = equity_curve.pct_change().dropna().rename("return")
    reb_index = idx[positions] if len(positions) else pd.DatetimeIndex([], name="date")
    trades = pd.DataFrame(
        trade_rows,
        columns=["date", "ticker", "prev_weight", "target_weight", "delta_weight", "notional"],
    )
    return {
        "equity_curve": equity_curve,
        "daily_returns": daily_returns,
        "turnover": pd.Series(turnover, index=reb_index, name="turnover"),
        "costs": pd.Series(costs, index=reb_index, name="cost"),
        "trades": trades,
        "weights": pd.DataFrame(used_targets, index=reb_index, columns=tickers),
        "holdings": pd.DataFrame(holdings, index=idx, columns=tickers),
    }


def summarise(
    equity_curve: pd.Series,
    daily_returns: pd.Series,
    n_trials: int = 1,
    *,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    trial_sharpes: Sequence[float] | None = None,
) -> dict[str, float]:
    """Summarise an equity curve with the thesis's headline statistics.

    Args:
        equity_curve: Portfolio value per trading day.
        daily_returns: Simple daily returns of the same portfolio.
        n_trials: Number of strategies/configurations whose performance was
            compared before this one was selected; passed to
            :func:`advisor.evaluation.metrics.deflated_sharpe_ratio`, which
            computes the return series' skewness and kurtosis internally
            (Bailey & Lopez de Prado 2014). Use ``1`` for a stand-alone run.
        periods_per_year: Annualisation constant (252 for daily data).
        trial_sharpes: Annualised Sharpe ratios of every compared strategy
            (including this one); their spread scales the DSR's
            expected-maximum benchmark. Optional.

    Returns:
        A dict with ``total_return``, ``cagr``, ``annualised_vol``, ``sharpe``,
        ``sortino``, ``max_drawdown``, ``deflated_sharpe``, ``calmar``,
        ``n_days``, ``n_trials``, ``initial_equity`` and ``final_equity``.
        Undefined ratios (e.g. Calmar with zero drawdown) are ``nan``.
    """
    eq = pd.Series(equity_curve, dtype="float64").dropna()
    rets = pd.Series(daily_returns, dtype="float64").dropna()
    n_days = int(len(eq))
    n_periods = int(len(rets))
    nan = float("nan")

    if n_days == 0:
        return {
            "total_return": nan,
            "cagr": nan,
            "annualised_vol": nan,
            "sharpe": nan,
            "sortino": nan,
            "max_drawdown": nan,
            "deflated_sharpe": nan,
            "calmar": nan,
            "n_days": 0,
            "n_trials": int(n_trials),
            "initial_equity": nan,
            "final_equity": nan,
        }

    initial = float(eq.iloc[0])
    final = float(eq.iloc[-1])
    total_return = final / initial - 1.0 if initial > 0 else nan

    years = n_periods / float(periods_per_year)
    if years > 0 and math.isfinite(total_return) and (1.0 + total_return) > 0:
        cagr = (1.0 + total_return) ** (1.0 / years) - 1.0
    else:
        cagr = nan

    if n_periods > 1:
        annualised_vol = float(rets.std(ddof=1)) * math.sqrt(periods_per_year)
    else:
        annualised_vol = nan

    mdd = float(max_drawdown(eq))
    calmar = cagr / abs(mdd) if (mdd < 0 and math.isfinite(cagr)) else nan

    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "annualised_vol": float(annualised_vol),
        "sharpe": float(sharpe_ratio(rets, periods_per_year=periods_per_year)),
        "sortino": float(sortino_ratio(rets, periods_per_year=periods_per_year)),
        "max_drawdown": mdd,
        "deflated_sharpe": float(
            deflated_sharpe_ratio(
                rets,
                n_trials=int(n_trials),
                periods_per_year=periods_per_year,
                trial_sharpes=trial_sharpes,
            )
        ),
        "calmar": float(calmar),
        "n_days": n_days,
        "n_trials": int(n_trials),
        "initial_equity": initial,
        "final_equity": final,
    }


def drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """Return the running drawdown ``equity / running_max - 1`` (non-positive).

    Args:
        equity_curve: Portfolio value per trading day.

    Returns:
        A ``pd.Series`` aligned with ``equity_curve``.
    """
    eq = pd.Series(equity_curve, dtype="float64")
    running_max = eq.cummax()
    return (eq / running_max - 1.0).rename("drawdown")


def period_returns(daily_returns: pd.Series, freq: str = "M") -> pd.Series:
    """Compound daily returns into calendar-period returns.

    Args:
        daily_returns: Simple daily returns indexed by ``DatetimeIndex``.
        freq: Period alias understood by ``DatetimeIndex.to_period`` (``"M"``
            for calendar months, ``"Q"`` for quarters, ``"W-FRI"`` for weeks).

    Returns:
        A ``pd.Series`` indexed by ``pd.Period`` with the compounded return of
        each period.
    """
    rets = pd.Series(daily_returns, dtype="float64").dropna()
    if rets.empty:
        return pd.Series(dtype="float64")
    periods = pd.DatetimeIndex(rets.index).to_period(freq)
    return rets.groupby(periods).apply(lambda r: float(np.prod(1.0 + r.to_numpy()) - 1.0))
