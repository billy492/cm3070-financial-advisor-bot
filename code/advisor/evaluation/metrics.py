"""Evaluation metrics for the financial advisor backtest and calibration layer.

This module implements the five core evaluation metrics used across the thesis
evaluation pipeline (ADR-0002):

* :func:`sharpe_ratio`   -- annualised risk-adjusted return.
* :func:`sortino_ratio`  -- downside-risk-adjusted return.
* :func:`max_drawdown`   -- worst peak-to-trough decline of an equity curve.
* :func:`deflated_sharpe_ratio` -- Sharpe ratio corrected for multiple testing,
  following Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting and Non-Normality",
  *Journal of Portfolio Management*, 40(5), 94-107.
* :func:`expected_calibration_error` -- binned ECE following Naeini, Cooper &
  Hauskrecht (2015), "Obtaining Well Calibrated Probabilities Using Bayesian
  Binning", *AAAI*.

All functions use only numpy and pandas, accept array-likes (lists, numpy
arrays, or pandas Series), and guard against empty / degenerate inputs by
returning a well-defined float (typically ``0.0`` or ``nan``) rather than
raising, so they are safe to call inside a backtest loop.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

__all__ = [
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "deflated_sharpe_ratio",
    "expected_calibration_error",
    "brier_score",
    "sharpness",
    "adaptive_ece",
    "hit_rate",
    "annualised_return",
    "calmar_ratio",
]


def _to_1d_array(values: Sequence[float] | np.ndarray | pd.Series) -> np.ndarray:
    """Coerce an array-like of numbers into a clean 1-D float numpy array.

    Drops NaN / inf entries so downstream statistics are well defined.

    Args:
        values: Any array-like of numbers (list, tuple, ndarray, or Series).

    Returns:
        A 1-D ``float64`` numpy array with non-finite values removed.
    """
    arr = np.asarray(values, dtype="float64").ravel()
    if arr.size == 0:
        return arr
    return arr[np.isfinite(arr)]


def sharpe_ratio(
    returns: Sequence[float] | np.ndarray | pd.Series,
    periods_per_year: int = 252,
    risk_free: float = 0.0,
) -> float:
    """Annualised Sharpe ratio of a series of periodic returns.

    Computes ``mean(excess) / std(excess) * sqrt(periods_per_year)`` where
    ``excess`` is the per-period return minus the per-period risk-free rate.
    The standard deviation uses the sample estimator (``ddof=1``).

    Args:
        returns: Per-period (e.g. daily) simple returns.
        periods_per_year: Number of return periods per year used to annualise
            (252 for daily trading days).
        risk_free: Per-period risk-free rate, in the same units as ``returns``.

    Returns:
        The annualised Sharpe ratio as a float. Returns ``0.0`` when fewer than
        two finite observations are available or when the return volatility is
        zero (degenerate / constant series).
    """
    r = _to_1d_array(returns)
    if r.size < 2:
        return 0.0
    excess = r - risk_free
    sd = float(np.std(excess, ddof=1))
    if sd == 0.0 or not math.isfinite(sd):
        return 0.0
    return float(np.mean(excess) / sd * math.sqrt(periods_per_year))


def sortino_ratio(
    returns: Sequence[float] | np.ndarray | pd.Series,
    periods_per_year: int = 252,
    risk_free: float = 0.0,
) -> float:
    """Annualised Sortino ratio of a series of periodic returns.

    Like the Sharpe ratio but the denominator is the *downside* deviation --
    the root-mean-square of the negative parts of the excess returns -- so only
    harmful volatility is penalised. The downside deviation is computed over all
    observations (zeros for non-negative excess), consistent with the standard
    Sortino definition.

    Args:
        returns: Per-period (e.g. daily) simple returns.
        periods_per_year: Number of return periods per year used to annualise.
        risk_free: Per-period risk-free / target rate, in the same units as
            ``returns``.

    Returns:
        The annualised Sortino ratio as a float. Returns ``0.0`` when fewer than
        two finite observations are available. Returns ``+inf`` when there is no
        downside risk but the mean excess return is positive (no negative
        excess returns), and ``0.0`` when the mean excess return is also zero.
    """
    r = _to_1d_array(returns)
    if r.size < 2:
        return 0.0
    excess = r - risk_free
    downside = np.minimum(excess, 0.0)
    downside_dev = float(math.sqrt(np.mean(downside ** 2)))
    if downside_dev == 0.0:
        mean_excess = float(np.mean(excess))
        if mean_excess > 0.0:
            return float("inf")
        return 0.0
    return float(np.mean(excess) / downside_dev * math.sqrt(periods_per_year))


def max_drawdown(
    equity_curve: Sequence[float] | np.ndarray | pd.Series,
) -> float:
    """Maximum drawdown of an equity curve.

    The drawdown at time ``t`` is ``equity[t] / running_max[t] - 1``; the
    maximum drawdown is the most negative such value over the whole curve.

    Args:
        equity_curve: Sequence of portfolio values (an equity / wealth curve),
            in chronological order. Should be strictly positive.

    Returns:
        The maximum drawdown as a non-positive float (e.g. ``-0.25`` for a 25%
        peak-to-trough decline). Returns ``0.0`` for an empty curve, a curve
        with fewer than two finite points, or a monotonically non-decreasing
        curve.
    """
    eq = _to_1d_array(equity_curve)
    if eq.size < 2:
        return 0.0
    running_max = np.maximum.accumulate(eq)
    # Avoid division by zero / sign issues on non-positive peaks.
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdowns = eq / running_max - 1.0
    drawdowns = drawdowns[np.isfinite(drawdowns)]
    if drawdowns.size == 0:
        return 0.0
    return float(min(drawdowns.min(), 0.0))


def deflated_sharpe_ratio(
    returns: Sequence[float] | np.ndarray | pd.Series,
    n_trials: int,
    *,
    periods_per_year: int = 252,
    trial_sharpes: Sequence[float] | np.ndarray | pd.Series | None = None,
    sr_variance: float | None = None,
) -> float:
    r"""Deflated Sharpe Ratio (DSR) after Bailey & Lopez de Prado (2014).

    ``trial_sharpes`` are the *annualised* Sharpe ratios of all ``n_trials``
    strategies that were compared (including this one); their cross-trial
    variance (in per-period units) scales the expected-maximum benchmark
    ``SR0``. ``sr_variance`` overrides it with an explicit per-period variance.
    When neither is given, the asymptotic sampling variance of one Sharpe
    estimate, ``(1 + SR_hat^2 / 2) / T`` (Lo 2002), is used, i.e. the spread
    that ``n_trials`` skill-less strategies of the same length would show.

    The DSR is the probability that the *true* Sharpe ratio of a strategy is
    positive, after correcting the observed Sharpe ratio for (a) the number of
    independent trials/configurations tried (multiple-testing / selection bias),
    (b) the sample length, and (c) the skewness and kurtosis (non-normality) of
    the return distribution.

    The procedure (Bailey & Lopez de Prado 2014, eqs. 9-11):

    1.  Estimate the (non-annualised, per-period) observed Sharpe ratio
        ``SR_hat = mean(r) / std(r)``.
    2.  Estimate the *expected maximum* Sharpe ratio under the null of zero
        skill across ``N`` trials, where Sharpe ratios are drawn i.i.d.
        :math:`\\mathcal{N}(0, \\sigma_{SR}^2)`. With the variance of the
        independent trial Sharpe estimates approximated as 1 (standardised),
        this benchmark is

        ``SR0 = sqrt(Var) * [ (1 - gamma) * Z^{-1}(1 - 1/N)
                              + gamma * Z^{-1}(1 - 1/(N*e)) ]``

        where ``gamma`` is the Euler-Mascheroni constant, ``Z^{-1}`` is the
        inverse standard-normal CDF, and ``Var`` is the cross-trial variance of
        the estimated Sharpe ratios (taken as 1 in this standardised form).
    3.  The DSR is the probability that the deflated, standardised Sharpe ratio
        exceeds this benchmark:

        ``DSR = Z( (SR_hat - SR0) * sqrt(T - 1)
                   / sqrt(1 - skew*SR_hat + (kurt - 1)/4 * SR_hat^2) )``

        where ``T`` is the number of observations, ``skew`` and ``kurt`` are the
        sample skewness and (non-excess) kurtosis of the returns, and ``Z`` is
        the standard-normal CDF.

    All Sharpe quantities above are computed on a *per-period* basis; the
    ``periods_per_year`` argument is accepted for interface symmetry with the
    other ratios and does not change the (scale-free) DSR probability.

    Args:
        returns: Per-period (e.g. daily) simple returns of the selected
            strategy.
        n_trials: Number of independent strategy configurations / trials that
            were searched before selecting this one. Must be ``>= 1``.
        periods_per_year: Accepted for interface symmetry; the DSR probability
            is invariant to this value.
        trial_sharpes: Annualised Sharpe ratios of every compared strategy;
            their cross-trial variance scales the benchmark ``SR0``.
        sr_variance: Explicit per-period variance of the trial Sharpe ratios;
            overrides ``trial_sharpes`` when given.

    Returns:
        The Deflated Sharpe Ratio as a probability in ``[0, 1]``. Returns
        ``0.0`` for fewer than two finite observations, zero volatility, or a
        non-positive ``n_trials`` benchmark that cannot be formed.
    """
    r = _to_1d_array(returns)
    t = r.size
    if t < 2:
        return 0.0

    mean = float(np.mean(r))
    sd = float(np.std(r, ddof=1))
    if sd == 0.0 or not math.isfinite(sd):
        return 0.0

    sr_hat = mean / sd  # per-period observed Sharpe ratio

    # Sample skewness and (non-excess) kurtosis of the returns.
    centred = r - mean
    pop_sd = float(np.std(r, ddof=0))
    if pop_sd == 0.0:
        return 0.0
    skew = float(np.mean(centred ** 3) / pop_sd ** 3)
    kurt = float(np.mean(centred ** 4) / pop_sd ** 4)  # non-excess (normal -> 3)

    n = max(int(n_trials), 1)

    # Expected maximum Sharpe ratio across N i.i.d. trials under H0 (zero skill).
    # Bailey & Lopez de Prado scale the benchmark by the cross-trial variance of
    # the *estimated* per-period Sharpe ratios, V[{SR_n}]. Priority: an explicit
    # ``sr_variance``; else the sample variance of ``trial_sharpes`` (annualised
    # values converted back to per-period units); else the asymptotic sampling
    # variance of a single Sharpe estimate, (1 + SR_hat^2 / 2) / T (Lo 2002),
    # which is the null-hypothesis spread of trial estimates of this length.
    if sr_variance is not None:
        var_sr = float(sr_variance)
    else:
        var_sr = float("nan")
        if trial_sharpes is not None:
            trials = _to_1d_array(trial_sharpes) / math.sqrt(periods_per_year)
            if trials.size >= 2:
                var_sr = float(np.var(trials, ddof=1))
        if not math.isfinite(var_sr) or var_sr <= 0.0:
            var_sr = (1.0 + 0.5 * sr_hat**2) / t
    gamma = 0.5772156649015329  # Euler-Mascheroni constant
    if n == 1:
        sr0 = 0.0
    else:
        z1 = _norm_ppf(1.0 - 1.0 / n)
        z2 = _norm_ppf(1.0 - 1.0 / (n * math.e))
        sr0 = math.sqrt(var_sr) * ((1.0 - gamma) * z1 + gamma * z2)

    # Variance term capturing non-normality of the return distribution.
    denom_var = 1.0 - skew * sr_hat + (kurt - 1.0) / 4.0 * sr_hat ** 2
    if denom_var <= 0.0 or not math.isfinite(denom_var):
        return 0.0

    numerator = (sr_hat - sr0) * math.sqrt(max(t - 1, 1))
    z_stat = numerator / math.sqrt(denom_var)
    return float(_norm_cdf(z_stat))


def expected_calibration_error(
    confidences: Sequence[float] | np.ndarray | pd.Series,
    correct: Sequence[float] | np.ndarray | pd.Series,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE) over equal-width confidence bins.

    Following Naeini, Cooper & Hauskrecht (2015) and the now-standard
    formulation in Guo et al. (2017), predictions are partitioned into
    ``n_bins`` equal-width bins over the confidence range ``[0, 1]``. For each
    non-empty bin the absolute gap between the average confidence and the
    empirical accuracy is computed, and the ECE is the sample-size-weighted mean
    of those gaps:

    ``ECE = sum_b (|B_b| / n) * | acc(B_b) - conf(B_b) |``

    Args:
        confidences: Predicted confidences (calibrated probabilities) in
            ``[0, 1]``, one per prediction.
        correct: Binary correctness labels (1 / True if the prediction was
            correct, else 0 / False), aligned element-wise with ``confidences``.
        n_bins: Number of equal-width bins spanning ``[0, 1]`` (default 10).

    Returns:
        The Expected Calibration Error as a float in ``[0, 1]``. Returns ``0.0``
        when there are no valid (finite, paired) observations or when
        ``n_bins < 1``.
    """
    conf = np.asarray(confidences, dtype="float64").ravel()
    corr = np.asarray(correct, dtype="float64").ravel()

    if conf.size == 0 or corr.size == 0 or conf.size != corr.size:
        return 0.0
    if n_bins < 1:
        return 0.0

    finite = np.isfinite(conf) & np.isfinite(corr)
    conf = conf[finite]
    corr = corr[finite]
    n = conf.size
    if n == 0:
        return 0.0

    # Clip confidences into [0, 1] so out-of-range values still land in a bin.
    conf = np.clip(conf, 0.0, 1.0)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Assign each confidence to a bin index in [0, n_bins-1].
    # Use the right edge as inclusive for the topmost bin.
    bin_ids = np.digitize(conf, bin_edges[1:-1], right=False)

    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        count = int(np.count_nonzero(mask))
        if count == 0:
            continue
        avg_conf = float(np.mean(conf[mask]))
        avg_acc = float(np.mean(corr[mask]))
        ece += (count / n) * abs(avg_acc - avg_conf)

    return float(ece)


# ---------------------------------------------------------------------------
# Small self-contained standard-normal helpers (numpy/pandas-only constraint;
# avoids a scipy dependency).
# ---------------------------------------------------------------------------


def _norm_cdf(x: float) -> float:
    """Standard-normal cumulative distribution function via ``math.erf``."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (quantile function).

    Uses the Acklam rational approximation, accurate to roughly 1.15e-9 over the
    full open interval ``(0, 1)``. Used to compute the expected-maximum-Sharpe
    benchmark in :func:`deflated_sharpe_ratio` without a scipy dependency.

    Args:
        p: Probability in the open interval ``(0, 1)``.

    Returns:
        The value ``z`` such that ``Phi(z) = p``. Clamps extreme inputs to a
        large finite magnitude rather than returning ``+/-inf``.
    """
    if p <= 0.0:
        return -1.0e10
    if p >= 1.0:
        return 1.0e10

    # Coefficients for Acklam's algorithm.
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
            ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
        )
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
    )


# ---------------------------------------------------------------------------
# Additional calibration / performance metrics (Chapter 4.5 of the preliminary
# report: report sharpness and a proper scoring rule next to ECE, and use
# adaptive equal-mass bins, so a base-rate collapse cannot score well).
# ---------------------------------------------------------------------------


def _paired_arrays(
    confidences: Sequence[float] | np.ndarray | pd.Series,
    labels: Sequence[float] | np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned finite ``(confidences, labels)`` arrays, confidences clipped to [0, 1].

    Mismatched lengths yield two empty arrays so callers can fall back to their
    degenerate-input value without raising.
    """
    conf = np.asarray(confidences, dtype="float64").ravel()
    lab = np.asarray(labels, dtype="float64").ravel()
    if conf.size == 0 or conf.size != lab.size:
        empty = np.empty(0, dtype="float64")
        return empty, empty
    keep = np.isfinite(conf) & np.isfinite(lab)
    return np.clip(conf[keep], 0.0, 1.0), lab[keep]


def brier_score(
    confidences: Sequence[float] | np.ndarray | pd.Series,
    labels: Sequence[float] | np.ndarray | pd.Series,
) -> float:
    """Brier score: mean squared error between confidence and outcome.

    ``BS = mean((p_i - y_i)^2)`` (Brier 1950). A strictly proper scoring rule,
    so - unlike ECE - it cannot be improved by discarding information: it
    rewards calibration *and* sharpness/resolution together (Murphy 1973
    decomposition). ``0`` is perfect, ``1`` is the worst possible, and always
    predicting the base rate ``r`` scores ``r(1 - r)``.

    Args:
        confidences: Predicted probabilities in ``[0, 1]`` (clipped if outside).
        labels: Binary outcomes (1/True = correct) aligned with ``confidences``.

    Returns:
        The Brier score in ``[0, 1]``; ``nan`` when there are no valid paired
        observations (``0`` would falsely read as a perfect score).
    """
    conf, lab = _paired_arrays(confidences, labels)
    if conf.size == 0:
        return float("nan")
    return float(np.mean((conf - lab) ** 2))


def sharpness(confidences: Sequence[float] | np.ndarray | pd.Series) -> float:
    """Sharpness: mean absolute distance of the confidences from ``0.5``.

    Sharpness is a property of the forecasts alone - how concentrated (far
    from the uninformative coin-flip value) they are - and calibration should
    be maximised *subject to* it (Gneiting, Balabdaoui & Raftery 2007). For a
    binary correctness forecast the natural summary is ``mean(|p - 0.5|)``,
    which ranges from ``0`` (every confidence is 0.5 - the base-rate-collapse
    failure mode seen with an over-large temperature) to ``0.5`` (every
    confidence is exactly 0 or 1). Reported next to ECE so that a calibrator
    cannot look good merely by flattening the confidences.

    Args:
        confidences: Predicted probabilities in ``[0, 1]`` (clipped if outside).

    Returns:
        A float in ``[0, 0.5]``; ``nan`` when there are no finite observations.
    """
    conf = _to_1d_array(confidences)
    if conf.size == 0:
        return float("nan")
    return float(np.mean(np.abs(np.clip(conf, 0.0, 1.0) - 0.5)))


def adaptive_ece(
    confidences: Sequence[float] | np.ndarray | pd.Series,
    labels: Sequence[float] | np.ndarray | pd.Series,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error with equal-mass (quantile) bins.

    Identical to :func:`expected_calibration_error` except that the
    predictions are sorted by confidence and split into ``n_bins`` contiguous
    groups of (near-)equal size, the *adaptive* binning of Nixon et al. (2019).
    Equal-width bins leave the tails nearly empty and the crowded middle bins
    dominate; equal-mass bins give every bin the same statistical weight, so the
    estimate is less sensitive to the bin count.

    Args:
        confidences: Predicted probabilities in ``[0, 1]`` (clipped if outside).
        labels: Binary outcomes aligned with ``confidences``.
        n_bins: Number of equal-mass bins (default 10). With fewer observations
            than bins some bins are empty and are skipped.

    Returns:
        The adaptive ECE as a float in ``[0, 1]``. Returns ``0.0`` when there
        are no valid paired observations or ``n_bins < 1`` (mirroring
        :func:`expected_calibration_error`).
    """
    conf, lab = _paired_arrays(confidences, labels)
    n = conf.size
    if n == 0 or n_bins < 1:
        return 0.0
    order = np.argsort(conf, kind="stable")
    ece = 0.0
    for idx in np.array_split(order, n_bins):
        if idx.size == 0:
            continue
        gap = abs(float(np.mean(lab[idx])) - float(np.mean(conf[idx])))
        ece += (idx.size / n) * gap
    return float(ece)


def hit_rate(labels: Sequence[float] | np.ndarray | pd.Series) -> float:
    """Fraction of recommendations that were correct.

    Args:
        labels: Binary correctness labels (1/True = correct). Any non-zero
            value counts as a hit; non-finite entries are ignored.

    Returns:
        The hit rate in ``[0, 1]``; ``nan`` when there are no finite labels
        (``0/0`` is undefined and ``0.0`` would claim "always wrong").
    """
    lab = _to_1d_array(labels)
    if lab.size == 0:
        return float("nan")
    return float(np.mean(lab != 0.0))


def annualised_return(
    equity_curve: Sequence[float] | np.ndarray | pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Compound annual growth rate (CAGR) of an equity curve.

    ``(end / start) ** (periods_per_year / (n - 1)) - 1`` where ``n`` is the
    number of finite points, i.e. the curve spans ``n - 1`` periods.

    Args:
        equity_curve: Portfolio values in chronological order (positive).
        periods_per_year: Periods per year (252 for daily trading days).

    Returns:
        The annualised return as a float (``0.10`` = +10 % a year). Returns
        ``0.0`` for fewer than two finite points or a non-positive starting
        value, and ``-1.0`` (total loss) when the final value is non-positive.
    """
    eq = _to_1d_array(equity_curve)
    if eq.size < 2 or periods_per_year <= 0:
        return 0.0
    start, end = float(eq[0]), float(eq[-1])
    if start <= 0.0:
        return 0.0
    if end <= 0.0:
        return -1.0
    years = (eq.size - 1) / periods_per_year
    return float((end / start) ** (1.0 / years) - 1.0)


def calmar_ratio(
    equity_curve: Sequence[float] | np.ndarray | pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Calmar ratio: annualised return divided by the magnitude of max drawdown.

    Args:
        equity_curve: Portfolio values in chronological order (positive).
        periods_per_year: Periods per year used by :func:`annualised_return`.

    Returns:
        ``annualised_return / |max_drawdown|``. Mirrors :func:`sortino_ratio`
        for a curve that never draws down: ``+inf`` when the annualised return
        is positive and ``0.0`` otherwise. Returns ``0.0`` for degenerate input.
    """
    ann = annualised_return(equity_curve, periods_per_year=periods_per_year)
    mdd = max_drawdown(equity_curve)
    if mdd == 0.0:
        return float("inf") if ann > 0.0 else 0.0
    return float(ann / abs(mdd))
