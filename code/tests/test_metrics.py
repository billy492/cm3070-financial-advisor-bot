"""Tests for ``advisor.evaluation.metrics``.

Hand-computed expected values guard the core evaluation statistics:

* :func:`sharpe_ratio` -- annualised mean/stdev ratio (invariant properties +
  tolerance band covering the ddof=0/ddof=1 convention).
* :func:`sortino_ratio` -- positive, finite for the mixed-sign fixture.
* :func:`max_drawdown` -- exact, unambiguous expected value on a crafted curve.
* :func:`expected_calibration_error` -- exact values on crafted bin layouts
  (Naeini 2015 equal-width binning).

All inputs are small and deterministic; comparisons use ``pytest.approx`` to
tolerate floating-point and (for Sharpe) the population-vs-sample stdev choice.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from advisor.evaluation.metrics import (
    expected_calibration_error,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)


# --------------------------------------------------------------------------- #
# Sharpe ratio
# --------------------------------------------------------------------------- #
def test_sharpe_ratio_known_band(returns_series: pd.Series) -> None:
    """Annualised Sharpe of the fixture lies in the ddof-agnostic band.

    mean = 0.005, ann factor = sqrt(252).
    - sample stdev (ddof=1) -> Sharpe ~= 4.756
    - population stdev (ddof=0) -> Sharpe ~= 5.084
    Either convention is acceptable; assert membership in the spanning band.
    """
    s = sharpe_ratio(returns_series)
    assert 4.5 <= s <= 5.2


def test_sharpe_ratio_zero_mean_is_zero() -> None:
    """A symmetric zero-mean series has a Sharpe ratio of (approximately) zero."""
    r = pd.Series([0.01, -0.01, 0.01, -0.01, 0.01, -0.01])
    assert sharpe_ratio(r) == pytest.approx(0.0, abs=1e-9)


def test_sharpe_ratio_sign_follows_mean() -> None:
    """A net-positive drift gives a positive Sharpe; net-negative gives negative."""
    pos = pd.Series([0.02, 0.01, 0.03, 0.015, 0.005])
    neg = pd.Series([-0.02, -0.01, -0.03, -0.015, -0.005])
    assert sharpe_ratio(pos) > 0
    assert sharpe_ratio(neg) < 0


def test_sharpe_ratio_risk_free_reduces_value(returns_series: pd.Series) -> None:
    """Subtracting a positive risk-free rate lowers the Sharpe ratio."""
    base = sharpe_ratio(returns_series, risk_free=0.0)
    with_rf = sharpe_ratio(returns_series, risk_free=0.001)
    assert with_rf < base


def test_sharpe_ratio_annualisation_scales_with_periods() -> None:
    """Higher periods_per_year scales Sharpe by sqrt(ratio of periods)."""
    r = pd.Series([0.01, -0.005, 0.02, 0.0, 0.015, -0.01])
    s_252 = sharpe_ratio(r, periods_per_year=252)
    s_63 = sharpe_ratio(r, periods_per_year=63)
    # sqrt(252/63) == 2 exactly.
    assert s_252 == pytest.approx(s_63 * 2.0, rel=1e-9)


# --------------------------------------------------------------------------- #
# Sortino ratio
# --------------------------------------------------------------------------- #
def test_sortino_ratio_positive_for_net_positive_drift(
    returns_series: pd.Series,
) -> None:
    """The fixture (net positive mean) yields a positive, finite Sortino ratio."""
    val = sortino_ratio(returns_series)
    assert math.isfinite(val)
    assert val > 0


def test_sortino_ratio_geq_zero_band(returns_series: pd.Series) -> None:
    """Sortino exceeds Sharpe here because downside-only dispersion is smaller."""
    assert sortino_ratio(returns_series) >= sharpe_ratio(returns_series)


# --------------------------------------------------------------------------- #
# Max drawdown
# --------------------------------------------------------------------------- #
def test_max_drawdown_exact_value() -> None:
    """Crafted equity curve: worst peak-to-trough is 150 -> 75 == -0.5."""
    equity = pd.Series([100.0, 120.0, 90.0, 150.0, 75.0, 150.0])
    assert max_drawdown(equity) == pytest.approx(-0.5)


def test_max_drawdown_monotonic_increasing_is_zero() -> None:
    """A strictly increasing curve never draws down: result is 0."""
    equity = pd.Series([100.0, 101.0, 102.0, 110.0, 200.0])
    assert max_drawdown(equity) == pytest.approx(0.0)


def test_max_drawdown_is_non_positive() -> None:
    """Drawdown is always <= 0 by definition."""
    equity = pd.Series([50.0, 40.0, 60.0, 30.0, 35.0])
    assert max_drawdown(equity) <= 0.0


def test_max_drawdown_full_wipeout() -> None:
    """A drop from peak 200 to 0 is a -100% (-1.0) drawdown."""
    equity = pd.Series([100.0, 200.0, 0.0])
    assert max_drawdown(equity) == pytest.approx(-1.0)


# --------------------------------------------------------------------------- #
# Expected Calibration Error (Naeini 2015)
# --------------------------------------------------------------------------- #
def test_ece_perfect_calibration_is_zero() -> None:
    """Confidences that exactly match per-bin accuracy give ECE == 0."""
    confidences = [0.0, 0.0, 1.0, 1.0]
    correct = [False, False, True, True]
    assert expected_calibration_error(confidences, correct, n_bins=5) == pytest.approx(
        0.0, abs=1e-9
    )


def test_ece_confident_but_wrong_is_high() -> None:
    """All predictions at 0.9 confidence but always wrong -> ECE == 0.9."""
    confidences = [0.9, 0.9, 0.9, 0.9]
    correct = [False, False, False, False]
    assert expected_calibration_error(confidences, correct, n_bins=10) == pytest.approx(
        0.9, abs=1e-9
    )


def test_ece_mixed_known_value() -> None:
    """Crafted 5-bin layout with one confidence per bin interior.

    bins (width 0.2): [0,0.2) [0.2,0.4) [0.4,0.6) [0.6,0.8) [0.8,1.0]
      conf 0.05 wrong  -> |0 - 0.05| = 0.05
      conf 0.25 wrong  -> |0 - 0.25| = 0.25
      conf 0.45 right  -> |1 - 0.45| = 0.55
      conf 0.65 right  -> |1 - 0.65| = 0.35
      conf 0.85 right  -> top bin, two samples (0.85, 0.95), both right ->
                          avg_conf=0.90, acc=1.0, |1-0.90| = 0.10
    weights: first four bins 1/6 each, top bin 2/6.
    ECE = (0.05+0.25+0.55+0.35)/6 + (2/6)*0.10 = 0.233333...
    """
    confidences = [0.05, 0.25, 0.45, 0.65, 0.85, 0.95]
    correct = [False, False, True, True, True, True]
    assert expected_calibration_error(confidences, correct, n_bins=5) == pytest.approx(
        0.233333, abs=1e-5
    )


def test_ece_is_bounded_unit_interval() -> None:
    """ECE always lies in [0, 1] for any valid input."""
    rng = np.random.default_rng(0)
    confidences = rng.uniform(0.0, 1.0, size=50).tolist()
    correct = (rng.uniform(0.0, 1.0, size=50) > 0.5).tolist()
    val = expected_calibration_error(confidences, correct, n_bins=10)
    assert 0.0 <= val <= 1.0


def test_ece_accepts_numpy_and_list_inputs() -> None:
    """ECE handles both python lists and numpy arrays for its inputs."""
    conf_list = [0.2, 0.8, 0.6, 0.4]
    correct_list = [False, True, True, False]
    as_list = expected_calibration_error(conf_list, correct_list, n_bins=4)
    as_np = expected_calibration_error(
        np.asarray(conf_list), np.asarray(correct_list), n_bins=4
    )
    assert as_list == pytest.approx(as_np)
