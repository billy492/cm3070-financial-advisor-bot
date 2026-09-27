"""Baseline strategies for backtest comparison (ADR-0002, Table 1 of the report).

The advisor is evaluated against exactly four pre-registered baselines that
span the passive, naive, factor-based and optimisation-based families:

* :func:`buy_and_hold` -- **buy-and-hold S&P 500** (the report's headline
  passive benchmark, tracked via the ``SPY`` ETF's adjusted close). If no
  benchmark series is supplied the function falls back to an equal-weight
  buy-and-hold of the universe and records ``mode`` in its details.
* :func:`random_allocation` -- a seeded **random allocation** null model:
  every week an equal-weight random subset of the universe is held.
  :func:`random_allocation_ensemble` repeats it over many seeds to build the
  null distribution of Sharpe ratios used for a permutation-style p-value
  (:func:`permutation_p_value`) of the advisor's Sharpe.
* :func:`momentum_12_1` -- cross-sectional **12-1 momentum**
  (Jegadeesh & Titman 1993): rank on the return from ``t-252`` to ``t-21``
  trading days (skip the most recent month to avoid short-term reversal),
  go long the top quantile equal-weight, rebalance monthly.
* :func:`markowitz_mean_variance` -- long-only **mean-variance** portfolio
  (Markowitz 1952): maximum-Sharpe (or minimum-variance when no asset has a
  positive expected excess return) over a 252-day lookback with a
  Ledoit-Wolf shrunk covariance, rebalanced monthly.

Every baseline is driven through the same
:func:`advisor.evaluation.portfolio.simulate_weights` engine -- the same
turnover cost model, the same mark-to-market rule and the same test window --
so that comparisons with the advisor are like-for-like (Arnott, Harvey &
Markowitz 2019). All signals are computed point-in-time: the weights set on a
date only use prices up to and including that date's close. Callers should
pass price history that starts *before* the test window (the runner passes
data from ``TRAIN_START``) so momentum and Markowitz have their lookbacks;
otherwise they sit in cash until enough history accrues.

The four public functions share the stub-era signature
``fn(prices_long, **kw) -> pd.Series`` (an equity curve indexed by date);
passing ``return_details=True`` instead returns the full simulation dict
(equity curve, daily returns, turnover, trades, weights, summary, and any
strategy-specific diagnostics).

References:
    Markowitz, H. (1952). Portfolio Selection. *Journal of Finance*, 7(1),
        77-91.
    Jegadeesh, N. & Titman, S. (1993). Returns to Buying Winners and Selling
        Losers: Implications for Stock Market Efficiency. *Journal of
        Finance*, 48(1), 65-91.
    Bailey, D. H. & Lopez de Prado, M. (2014). The Deflated Sharpe Ratio.
        *Journal of Portfolio Management*, 40(5), 94-107.
    Arnott, R., Harvey, C. R. & Markowitz, H. (2019). A Backtesting Protocol
        in the Era of Machine Learning. *Journal of Financial Data Science*,
        1(1), 64-74.
    Ledoit, O. & Wolf, M. (2004). A well-conditioned estimator for
        large-dimensional covariance matrices. *Journal of Multivariate
        Analysis*, 88(2), 365-411.
"""

from __future__ import annotations

import logging
import math
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from advisor.config import RANDOM_SEED, TEST_END, TEST_START
from advisor.evaluation.portfolio import (
    decision_dates,
    pivot_prices,
    simulate_weights,
    slice_window,
    summarise,
)

__all__ = [
    "BASELINE_NAMES",
    "buy_and_hold",
    "random_allocation",
    "random_allocation_ensemble",
    "permutation_p_value",
    "momentum_12_1",
    "markowitz_mean_variance",
    "run_all_baselines",
]

log = logging.getLogger(__name__)

#: Canonical order of the four baselines in every table and figure.
BASELINE_NAMES: tuple[str, ...] = (
    "buy_and_hold",
    "random_allocation",
    "momentum_12_1",
    "markowitz_mean_variance",
)

DateLike = date | str | pd.Timestamp


# --------------------------------------------------------------------------- #
# Shared plumbing
# --------------------------------------------------------------------------- #
def _prepare(
    prices_long: pd.DataFrame,
    start: DateLike,
    end: DateLike,
    price_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pivot (forward-filled) prices and slice the evaluation window.

    Args:
        prices_long: Long-form price frame (may start before ``start``).
        start: Inclusive window start.
        end: Exclusive window end.
        price_column: Price field to simulate on.

    Returns:
        ``(full_wide, window_wide)`` -- the full-history wide frame used for
        point-in-time signals and its ``[start, end)`` slice used for
        simulation.

    Raises:
        ValueError: If the window contains no trading days.
    """
    full = pivot_prices(prices_long, price_column, ffill=True)
    window = slice_window(full, start, end)
    if window.empty:
        raise ValueError(
            f"No price rows between {pd.Timestamp(start).date()} and "
            f"{pd.Timestamp(end).date()}; check the window and the loaded history."
        )
    return full, window


def _rebalance_dates(window_index: pd.DatetimeIndex, freq: str | int) -> list[pd.Timestamp]:
    """Decision dates inside the window, always including its first trading day.

    Args:
        window_index: Trading days of the evaluation window.
        freq: Rebalance frequency (see
            :func:`advisor.evaluation.portfolio.decision_dates`).

    Returns:
        Sorted list of rebalance dates starting at ``window_index[0]``.
    """
    dates = decision_dates(window_index, freq)
    first = pd.Timestamp(window_index[0])
    if not dates or dates[0] != first:
        dates.insert(0, first)
    return dates


def _finish(
    sim: dict[str, Any],
    *,
    return_details: bool,
    name: str,
    **extra: Any,
) -> pd.Series | dict[str, Any]:
    """Attach a summary and diagnostics to a simulation, or return its curve.

    Args:
        sim: Output of :func:`simulate_weights`.
        return_details: When ``False`` return just the equity curve.
        name: Strategy name stamped onto the curve and details.
        **extra: Strategy-specific diagnostics merged into the details.

    Returns:
        Either the equity ``pd.Series`` (named ``name``) or the details dict.
    """
    curve = sim["equity_curve"].rename(name)
    if not return_details:
        return curve
    details = dict(sim)
    details["equity_curve"] = curve
    details["name"] = name
    details["summary"] = summarise(curve, sim["daily_returns"])
    details.update(extra)
    return details


# --------------------------------------------------------------------------- #
# 1. Buy-and-hold (S&P 500 via SPY)
# --------------------------------------------------------------------------- #
def buy_and_hold(
    prices_long: pd.DataFrame,
    *,
    benchmark_prices: pd.DataFrame | None = None,
    start: DateLike = TEST_START,
    end: DateLike = TEST_END,
    cost_bps: float = 10.0,
    initial_capital: float = 100_000.0,
    price_column: str = "adj_close",
    return_details: bool = False,
    **_: Any,
) -> pd.Series | dict[str, Any]:
    """Passive buy-and-hold benchmark.

    The report's headline baseline is **buy-and-hold of the S&P 500**: pass
    ``benchmark_prices`` (a long-form frame for ``"SPY"`` -- or ``"^GSPC"`` --
    loaded by the caller via :func:`advisor.data.loader.load_prices`) and the
    strategy buys it with 100% of capital on the first trading day of the
    window and never trades again (``mode="benchmark"``). When no benchmark
    frame is supplied the function degrades to an equal-weight buy-and-hold of
    every ticker in ``prices_long`` with a valid price on the first day
    (``mode="equal_weight_universe"``); weights then drift with prices and are
    never rebalanced.

    Args:
        prices_long: Long-form universe prices (used only in fallback mode).
        benchmark_prices: Long-form benchmark prices (``SPY``); ``None`` for
            the equal-weight fallback.
        start: Inclusive window start.
        end: Exclusive window end.
        cost_bps: One-way turnover cost in basis points (charged once, on
            the initial purchase).
        initial_capital: Starting capital.
        price_column: Price field to simulate on.
        return_details: Return the full simulation dict instead of the curve.
        **_: Ignored (keeps the stub-era ``**kw`` signature).

    Returns:
        The equity ``pd.Series`` or, with ``return_details=True``, a dict that
        additionally carries ``mode`` and ``tickers``.
    """
    if benchmark_prices is not None and len(benchmark_prices) > 0:
        _, window = _prepare(benchmark_prices, start, end, price_column)
        mode = "benchmark"
    else:
        log.warning(
            "buy_and_hold: no benchmark_prices supplied; falling back to an "
            "equal-weight buy-and-hold of the universe."
        )
        _, window = _prepare(prices_long, start, end, price_column)
        mode = "equal_weight_universe"

    first = window.index[0]
    valid = window.columns[window.iloc[0].notna()]
    if len(valid) == 0:
        raise ValueError("buy_and_hold: no ticker has a valid price on the first window day.")
    weights = pd.DataFrame([[1.0 / len(valid)] * len(valid)], index=[first], columns=list(valid))
    sim = simulate_weights(window, weights, cost_bps=cost_bps, initial_capital=initial_capital)
    return _finish(
        sim,
        return_details=return_details,
        name="buy_and_hold",
        mode=mode,
        tickers=list(valid),
    )


# --------------------------------------------------------------------------- #
# 2. Random allocation (null model) + ensemble
# --------------------------------------------------------------------------- #
def _random_weights(
    window: pd.DataFrame,
    dates: list[pd.Timestamp],
    n_holdings: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Draw an equal-weight random subset of tickers at each rebalance date.

    Args:
        window: Wide price frame of the evaluation window.
        dates: Rebalance dates (subset of ``window.index``).
        n_holdings: Number of names to hold (capped by availability).
        rng: Seeded numpy generator.

    Returns:
        Target weights (rows = ``dates``, columns = ``window.columns``).
    """
    rows = np.zeros((len(dates), window.shape[1]), dtype="float64")
    available = window.notna().to_numpy()
    positions = window.index.get_indexer(pd.DatetimeIndex(dates))
    for r, pos in enumerate(positions):
        candidates = np.flatnonzero(available[pos])
        k = min(int(n_holdings), len(candidates))
        if k <= 0:
            continue
        chosen = rng.choice(candidates, size=k, replace=False)
        rows[r, chosen] = 1.0 / k
    return pd.DataFrame(rows, index=pd.DatetimeIndex(dates), columns=window.columns)


def random_allocation(
    prices_long: pd.DataFrame,
    *,
    n_holdings: int = 10,
    seed: int = RANDOM_SEED,
    freq: str | int = "W-FRI",
    start: DateLike = TEST_START,
    end: DateLike = TEST_END,
    cost_bps: float = 10.0,
    initial_capital: float = 100_000.0,
    price_column: str = "adj_close",
    return_details: bool = False,
    **_: Any,
) -> pd.Series | dict[str, Any]:
    """Seeded random-allocation null model.

    At each rebalance date (weekly by default, matching the advisor's decision
    cadence) the portfolio holds an equal-weight random subset of
    ``n_holdings`` tickers drawn without replacement -- from a
    ``numpy.random.default_rng(seed)`` generator -- among the tickers with a
    valid price that day. It controls for luck: a strategy that cannot beat
    the distribution of such portfolios (see
    :func:`random_allocation_ensemble`) has shown no skill.

    Args:
        prices_long: Long-form universe prices.
        n_holdings: Number of names held each period (default 10).
        seed: Generator seed (default :data:`advisor.config.RANDOM_SEED`).
        freq: Rebalance frequency (default weekly, ``"W-FRI"``).
        start: Inclusive window start.
        end: Exclusive window end.
        cost_bps: One-way turnover cost in basis points.
        initial_capital: Starting capital.
        price_column: Price field to simulate on.
        return_details: Return the full simulation dict instead of the curve.
        **_: Ignored (keeps the stub-era ``**kw`` signature).

    Returns:
        The equity ``pd.Series`` or the details dict (with ``seed`` and
        ``n_holdings``).
    """
    _, window = _prepare(prices_long, start, end, price_column)
    dates = _rebalance_dates(window.index, freq)
    rng = np.random.default_rng(int(seed))
    weights = _random_weights(window, dates, n_holdings, rng)
    sim = simulate_weights(window, weights, cost_bps=cost_bps, initial_capital=initial_capital)
    return _finish(
        sim,
        return_details=return_details,
        name="random_allocation",
        seed=int(seed),
        n_holdings=int(n_holdings),
        freq=freq,
    )


def random_allocation_ensemble(
    prices_long: pd.DataFrame,
    n_seeds: int = 100,
    *,
    seed: int = RANDOM_SEED,
    n_holdings: int = 10,
    freq: str | int = "W-FRI",
    start: DateLike = TEST_START,
    end: DateLike = TEST_END,
    cost_bps: float = 10.0,
    initial_capital: float = 100_000.0,
    price_column: str = "adj_close",
) -> dict[str, Any]:
    """Null distribution of random-allocation portfolios over many seeds.

    Runs :func:`random_allocation` with seeds ``seed, seed+1, ..., seed+n_seeds-1``
    (independent ``default_rng`` streams) and aggregates the resulting equity
    curves and terminal statistics. The ``terminal_sharpe`` series is the
    empirical null distribution used by :func:`permutation_p_value` to test
    whether the advisor's Sharpe ratio is distinguishable from luck -- the
    multiple-testing spirit of Bailey & Lopez de Prado (2014) applied as a
    Monte-Carlo permutation test.

    Args:
        prices_long: Long-form universe prices.
        n_seeds: Number of ensemble members (default 100).
        seed: Base seed; member ``i`` uses ``seed + i``.
        n_holdings: Names held per period.
        freq: Rebalance frequency.
        start: Inclusive window start.
        end: Exclusive window end.
        cost_bps: One-way turnover cost in basis points.
        initial_capital: Starting capital.
        price_column: Price field to simulate on.

    Returns:
        A dict with

        * ``mean_equity_curve``: ``pd.Series`` mean curve across members;
        * ``percentiles``: ``pd.DataFrame`` with columns ``p05``, ``p50``,
          ``p95`` (per-day cross-sectional percentiles of equity);
        * ``terminal_sharpe``: ``pd.Series`` of each member's annualised
          Sharpe ratio, indexed by seed;
        * ``terminal_stats``: ``pd.DataFrame`` (seed, sharpe, sortino,
          total_return, max_drawdown);
        * ``curves``: ``pd.DataFrame`` of all member equity curves;
        * ``n_seeds``, ``seeds``, ``n_holdings``, ``freq``.

    Raises:
        ValueError: If ``n_seeds < 1``.
    """
    if n_seeds < 1:
        raise ValueError("random_allocation_ensemble: n_seeds must be >= 1.")
    _, window = _prepare(prices_long, start, end, price_column)
    dates = _rebalance_dates(window.index, freq)

    seeds = [int(seed) + i for i in range(int(n_seeds))]
    curves: dict[int, pd.Series] = {}
    stats: list[dict[str, float]] = []
    for s in seeds:
        rng = np.random.default_rng(s)
        weights = _random_weights(window, dates, n_holdings, rng)
        sim = simulate_weights(window, weights, cost_bps=cost_bps, initial_capital=initial_capital)
        curves[s] = sim["equity_curve"]
        summary = summarise(sim["equity_curve"], sim["daily_returns"])
        stats.append(
            {
                "seed": s,
                "sharpe": summary["sharpe"],
                "sortino": summary["sortino"],
                "total_return": summary["total_return"],
                "max_drawdown": summary["max_drawdown"],
            }
        )

    curves_df = pd.DataFrame(curves)
    curves_df.columns.name = "seed"
    stats_df = pd.DataFrame(stats).set_index("seed")
    percentiles = pd.DataFrame(
        {
            "p05": curves_df.quantile(0.05, axis=1),
            "p50": curves_df.quantile(0.50, axis=1),
            "p95": curves_df.quantile(0.95, axis=1),
        }
    )
    return {
        "mean_equity_curve": curves_df.mean(axis=1).rename("random_ensemble_mean"),
        "percentiles": percentiles,
        "terminal_sharpe": stats_df["sharpe"].rename("sharpe"),
        "terminal_stats": stats_df,
        "curves": curves_df,
        "n_seeds": int(n_seeds),
        "seeds": seeds,
        "n_holdings": int(n_holdings),
        "freq": freq,
    }


def permutation_p_value(observed: float, null_samples: pd.Series | np.ndarray) -> float:
    """One-sided Monte-Carlo p-value of an observed statistic against a null sample.

    Uses the standard add-one estimator ``(1 + #{null >= observed}) / (1 + n)``
    (Phipson & Smyth 2010), which never returns exactly zero.

    Args:
        observed: The strategy's statistic (e.g. annualised Sharpe).
        null_samples: Statistics of the null ensemble members.

    Returns:
        The p-value in ``(0, 1]``; ``nan`` if the null sample is empty or the
        observation is not finite.
    """
    null = np.asarray(pd.Series(null_samples, dtype="float64").dropna(), dtype="float64")
    if null.size == 0 or not math.isfinite(observed):
        return float("nan")
    return float((1.0 + np.count_nonzero(null >= observed)) / (1.0 + null.size))


# --------------------------------------------------------------------------- #
# 3. 12-1 momentum (Jegadeesh & Titman 1993)
# --------------------------------------------------------------------------- #
def momentum_12_1(
    prices_long: pd.DataFrame,
    *,
    top_quantile: float = 0.2,
    lookback_days: int = 252,
    skip_days: int = 21,
    freq: str | int = "M",
    start: DateLike = TEST_START,
    end: DateLike = TEST_END,
    cost_bps: float = 10.0,
    initial_capital: float = 100_000.0,
    price_column: str = "adj_close",
    return_details: bool = False,
    **_: Any,
) -> pd.Series | dict[str, Any]:
    """Cross-sectional 12-1 momentum (Jegadeesh & Titman 1993).

    On each monthly rebalance date ``t`` every ticker is scored by
    ``P(t - skip_days) / P(t - lookback_days) - 1`` -- the trailing
    twelve-month return with the most recent month skipped, the classic
    "12-1" formation that side-steps the one-month reversal effect. The top
    ``top_quantile`` fraction (``ceil(q * n_valid)``, at least one name) is
    held equal-weight until the next rebalance. The signal is computed with
    backward positional shifts on the full price history, so it is strictly
    point-in-time; dates without a full lookback score NaN and are excluded.
    If no ticker has a valid score the portfolio sits in cash, which is what
    happens when the caller supplies fewer than ``lookback_days`` days of
    pre-window history.

    Args:
        prices_long: Long-form prices **starting well before** ``start``
            (at least ``lookback_days`` trading days earlier).
        top_quantile: Fraction of the ranked universe to hold (default 0.2).
        lookback_days: Formation-period length in trading days (12 months).
        skip_days: Most-recent days skipped (1 month).
        freq: Rebalance frequency (default monthly).
        start: Inclusive window start.
        end: Exclusive window end.
        cost_bps: One-way turnover cost in basis points.
        initial_capital: Starting capital.
        price_column: Price field to simulate on.
        return_details: Return the full simulation dict instead of the curve.
        **_: Ignored (keeps the stub-era ``**kw`` signature).

    Returns:
        The equity ``pd.Series`` or the details dict (with the ``signal``
        frame at the decision dates and ``n_valid`` per date).

    Raises:
        ValueError: For an invalid quantile or lookback configuration.
    """
    if not 0.0 < top_quantile <= 1.0:
        raise ValueError("momentum_12_1: top_quantile must be in (0, 1].")
    if lookback_days <= skip_days or skip_days < 0:
        raise ValueError("momentum_12_1: require lookback_days > skip_days >= 0.")

    full, window = _prepare(prices_long, start, end, price_column)
    dates = _rebalance_dates(window.index, freq)

    # Backward-looking shifts only: signal at t uses P[t-skip] and P[t-lookback].
    signal_all = full.shift(skip_days) / full.shift(lookback_days) - 1.0
    signal = signal_all.loc[pd.DatetimeIndex(dates)]
    current_valid = window.loc[pd.DatetimeIndex(dates)].notna()
    signal = signal.where(current_valid)

    rows = np.zeros((len(dates), window.shape[1]), dtype="float64")
    n_valid: list[int] = []
    col_index = {c: i for i, c in enumerate(window.columns)}
    for r, d in enumerate(dates):
        scores = signal.loc[d].dropna()
        n_valid.append(int(len(scores)))
        if scores.empty:
            continue
        k = max(1, int(math.ceil(top_quantile * len(scores))))
        top = scores.sort_values(ascending=False, kind="mergesort").index[:k]
        for ticker in top:
            rows[r, col_index[ticker]] = 1.0 / k
    if n_valid and n_valid[0] == 0:
        log.warning(
            "momentum_12_1: no valid momentum score on %s -- supply at least %d "
            "trading days of history before the window (sitting in cash until then).",
            dates[0].date(),
            lookback_days,
        )

    weights = pd.DataFrame(rows, index=pd.DatetimeIndex(dates), columns=window.columns)
    sim = simulate_weights(window, weights, cost_bps=cost_bps, initial_capital=initial_capital)
    return _finish(
        sim,
        return_details=return_details,
        name="momentum_12_1",
        signal=signal,
        n_valid=pd.Series(n_valid, index=pd.DatetimeIndex(dates), name="n_valid"),
        top_quantile=float(top_quantile),
        lookback_days=int(lookback_days),
        skip_days=int(skip_days),
        freq=freq,
    )


# --------------------------------------------------------------------------- #
# 4. Markowitz mean-variance (1952)
# --------------------------------------------------------------------------- #
def _solve_long_only(
    mu: np.ndarray,
    sigma: np.ndarray,
    risk_free: float = 0.0,
) -> tuple[np.ndarray, str, bool]:
    """Long-only maximum-Sharpe (or minimum-variance) weights via SLSQP.

    Maximising the Sharpe ratio directly is non-convex, so when at least one
    asset has a positive expected excess return the problem is solved in its
    convex Charnes-Cooper form: ``min y' Sigma y`` subject to
    ``(mu - rf)' y = 1, y >= 0`` and ``w = y / sum(y)``. When every expected
    excess return is non-positive the tangency portfolio is undefined and the
    global minimum-variance portfolio (``min w' Sigma w, w >= 0, sum w = 1``)
    is used instead. If the optimiser fails the routine falls back to
    minimum-variance and finally to equal weight, and reports which.

    Args:
        mu: Expected (per-period) returns, shape ``(n,)``.
        sigma: Covariance matrix, shape ``(n, n)`` (positive semi-definite).
        risk_free: Per-period risk-free rate.

    Returns:
        ``(weights, mode, success)`` where ``mode`` is ``"max_sharpe"``,
        ``"min_variance"`` or ``"equal_weight"`` and ``success`` reports
        whether the optimiser converged.
    """
    from scipy.optimize import minimize  # lazy: keep import light

    n = int(len(mu))
    if n == 1:
        return np.array([1.0]), "single_asset", True
    excess = np.asarray(mu, dtype="float64") - float(risk_free)
    sigma = np.asarray(sigma, dtype="float64")
    equal = np.full(n, 1.0 / n)

    def _normalise(x: np.ndarray) -> np.ndarray:
        x = np.clip(np.asarray(x, dtype="float64"), 0.0, None)
        total = x.sum()
        return x / total if total > 0 else equal.copy()

    if np.max(excess) > 0.0:
        k = int(np.argmax(excess))
        y0 = np.zeros(n)
        y0[k] = 1.0 / excess[k]
        res = minimize(
            lambda y: float(y @ sigma @ y),
            y0,
            jac=lambda y: 2.0 * sigma @ y,
            method="SLSQP",
            bounds=[(0.0, None)] * n,
            constraints=[
                {"type": "eq", "fun": lambda y: float(excess @ y) - 1.0, "jac": lambda y: excess}
            ],
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        if res.success and np.isfinite(res.x).all() and res.x.sum() > 0:
            return _normalise(res.x), "max_sharpe", True
        log.debug("max-Sharpe SLSQP failed (%s); falling back to min-variance.", res.message)

    res = minimize(
        lambda w: float(w @ sigma @ w),
        equal,
        jac=lambda w: 2.0 * sigma @ w,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[
            {"type": "eq", "fun": lambda w: float(w.sum()) - 1.0, "jac": lambda w: np.ones(n)}
        ],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if res.success and np.isfinite(res.x).all():
        return _normalise(res.x), "min_variance", True
    log.debug("min-variance SLSQP failed (%s); falling back to equal weight.", res.message)
    return equal.copy(), "equal_weight", False


def markowitz_mean_variance(
    prices_long: pd.DataFrame,
    *,
    lookback_days: int = 252,
    min_history_days: int = 60,
    freq: str | int = "M",
    risk_free: float = 0.0,
    start: DateLike = TEST_START,
    end: DateLike = TEST_END,
    cost_bps: float = 10.0,
    initial_capital: float = 100_000.0,
    price_column: str = "adj_close",
    return_details: bool = False,
    **_: Any,
) -> pd.Series | dict[str, Any]:
    """Long-only Markowitz (1952) mean-variance portfolio, rebalanced monthly.

    On each rebalance date the trailing ``lookback_days`` daily simple returns
    of every ticker with a complete history in that window are used to
    estimate the sample mean vector and a **Ledoit-Wolf shrunk** covariance
    matrix (``sklearn.covariance.LedoitWolf``). The long-only maximum-Sharpe
    portfolio is solved with ``scipy.optimize.minimize`` (SLSQP, ``w >= 0``,
    ``sum w = 1``); if no asset has a positive expected excess return the
    minimum-variance portfolio is used instead. Inputs are annualised (x252)
    purely for numerical conditioning -- the optimal weights are invariant to
    that scaling.

    Limitations (documented rather than patched, because the baseline is meant
    to be the textbook method): mean-variance weights are notoriously
    sensitive to estimation error in the expected returns (Markowitz 1952;
    see also DeMiguel, Garlappi & Uppal 2009 on the difficulty of beating
    1/N), and sample means over one year of daily data are noisy. Shrinking
    the covariance tames the second-moment error; the first-moment error is
    left as-is, which is why the baseline can concentrate in a handful of
    names.

    Args:
        prices_long: Long-form prices **starting before** ``start`` so the
            first rebalance has its lookback.
        lookback_days: Estimation window in trading days (default 252).
        min_history_days: Minimum rows required to optimise; with fewer the
            portfolio stays in cash.
        freq: Rebalance frequency (default monthly).
        risk_free: Per-day risk-free rate used for the excess return.
        start: Inclusive window start.
        end: Exclusive window end.
        cost_bps: One-way turnover cost in basis points.
        initial_capital: Starting capital.
        price_column: Price field to simulate on.
        return_details: Return the full simulation dict instead of the curve.
        **_: Ignored (keeps the stub-era ``**kw`` signature).

    Returns:
        The equity ``pd.Series`` or the details dict (with ``modes`` per
        date, ``n_assets`` per date and the ``target_weights`` frame).

    Raises:
        ValueError: If ``lookback_days`` or ``min_history_days`` is invalid.
    """
    from sklearn.covariance import LedoitWolf  # lazy: keep import light

    if lookback_days < 2 or min_history_days < 2 or min_history_days > lookback_days:
        raise ValueError("markowitz_mean_variance: require 2 <= min_history_days <= lookback_days.")

    full, window = _prepare(prices_long, start, end, price_column)
    dates = _rebalance_dates(window.index, freq)
    returns_all = full.pct_change()

    rows = np.zeros((len(dates), window.shape[1]), dtype="float64")
    modes: list[str] = []
    n_assets: list[int] = []
    successes: list[bool] = []
    col_index = {c: i for i, c in enumerate(window.columns)}
    for r, d in enumerate(dates):
        pos = int(full.index.get_loc(d))
        lo = max(1, pos - lookback_days + 1)  # row 0 of pct_change is NaN
        block = returns_all.iloc[lo : pos + 1]
        block = block.dropna(axis=1, how="any")
        # Only tickers priced today can be bought.
        priced = window.loc[d]
        block = block.loc[:, [c for c in block.columns if pd.notna(priced.get(c))]]
        if len(block) < min_history_days or block.shape[1] == 0:
            modes.append("cash")
            n_assets.append(0)
            successes.append(False)
            continue
        values = block.to_numpy(dtype="float64")
        mu = values.mean(axis=0) * 252.0
        sigma = LedoitWolf().fit(values).covariance_ * 252.0
        w, mode, ok = _solve_long_only(mu, sigma, risk_free=float(risk_free) * 252.0)
        for c, wi in zip(block.columns, w, strict=True):
            rows[r, col_index[c]] = float(wi)
        modes.append(mode)
        n_assets.append(int(block.shape[1]))
        successes.append(bool(ok))
    if modes and modes[0] == "cash":
        log.warning(
            "markowitz_mean_variance: insufficient history on %s (need >= %d days); "
            "sitting in cash until enough accrues.",
            dates[0].date(),
            min_history_days,
        )

    weights = pd.DataFrame(rows, index=pd.DatetimeIndex(dates), columns=window.columns)
    sim = simulate_weights(window, weights, cost_bps=cost_bps, initial_capital=initial_capital)
    return _finish(
        sim,
        return_details=return_details,
        name="markowitz_mean_variance",
        target_weights=weights,
        modes=pd.Series(modes, index=pd.DatetimeIndex(dates), name="mode"),
        n_assets=pd.Series(n_assets, index=pd.DatetimeIndex(dates), name="n_assets"),
        optimiser_success=pd.Series(successes, index=pd.DatetimeIndex(dates), name="success"),
        lookback_days=int(lookback_days),
        freq=freq,
    )


# --------------------------------------------------------------------------- #
# Convenience: run everything the report compares against
# --------------------------------------------------------------------------- #
def run_all_baselines(
    prices_long: pd.DataFrame,
    *,
    benchmark_prices: pd.DataFrame | None = None,
    start: DateLike = TEST_START,
    end: DateLike = TEST_END,
    cost_bps: float = 10.0,
    initial_capital: float = 100_000.0,
    seed: int = RANDOM_SEED,
    n_seeds: int = 100,
    n_holdings: int = 10,
    random_freq: str | int = "W-FRI",
    price_column: str = "adj_close",
) -> dict[str, dict[str, Any]]:
    """Run the four baselines plus the random null ensemble with shared settings.

    Args:
        prices_long: Long-form universe prices (history from before ``start``).
        benchmark_prices: Long-form ``SPY`` prices for :func:`buy_and_hold`.
        start: Inclusive window start.
        end: Exclusive window end.
        cost_bps: One-way turnover cost in basis points (shared by all).
        initial_capital: Starting capital (shared by all).
        seed: Base random seed.
        n_seeds: Ensemble size for the random null.
        n_holdings: Names held by the random allocation.
        random_freq: Rebalance frequency of the random allocation.
        price_column: Price field to simulate on.

    Returns:
        ``{name: details}`` for each of :data:`BASELINE_NAMES` plus
        ``"random_ensemble"`` (the output of :func:`random_allocation_ensemble`).
    """
    common: dict[str, Any] = {
        "start": start,
        "end": end,
        "cost_bps": cost_bps,
        "initial_capital": initial_capital,
        "price_column": price_column,
        "return_details": True,
    }
    results: dict[str, dict[str, Any]] = {}
    log.info("Baseline: buy_and_hold")
    results["buy_and_hold"] = buy_and_hold(prices_long, benchmark_prices=benchmark_prices, **common)
    log.info("Baseline: random_allocation (seed=%d)", seed)
    results["random_allocation"] = random_allocation(
        prices_long, seed=seed, n_holdings=n_holdings, freq=random_freq, **common
    )
    log.info("Baseline: momentum_12_1")
    results["momentum_12_1"] = momentum_12_1(prices_long, **common)
    log.info("Baseline: markowitz_mean_variance")
    results["markowitz_mean_variance"] = markowitz_mean_variance(prices_long, **common)
    log.info("Random null ensemble (%d seeds)", n_seeds)
    results["random_ensemble"] = random_allocation_ensemble(
        prices_long,
        n_seeds=n_seeds,
        seed=seed,
        n_holdings=n_holdings,
        freq=random_freq,
        start=start,
        end=end,
        cost_bps=cost_bps,
        initial_capital=initial_capital,
        price_column=price_column,
    )
    return results
