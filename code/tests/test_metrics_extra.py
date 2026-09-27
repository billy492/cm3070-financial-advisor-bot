"""Tests for the calibration / performance metrics added to ``advisor.evaluation.metrics``.

Covers the proper scoring rule and sharpness summary that Chapter 4.5 of the
preliminary report asks for alongside ECE (so a base-rate collapse cannot
score well), the equal-mass adaptive ECE, and the equity-curve helpers.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from advisor.evaluation import metrics
from advisor.evaluation.metrics import (
    adaptive_ece,
    annualised_return,
    brier_score,
    calmar_ratio,
    expected_calibration_error,
    hit_rate,
    max_drawdown,
    sharpness,
)


def test_new_metrics_are_exported() -> None:
    """The additions are part of the module's public API."""
    for name in (
        "brier_score",
        "sharpness",
        "adaptive_ece",
        "hit_rate",
        "annualised_return",
        "calmar_ratio",
    ):
        assert name in metrics.__all__


# --------------------------------------------------------------------------- #
# Brier score
# --------------------------------------------------------------------------- #
def test_brier_perfect_and_worst_predictions() -> None:
    """Certain-and-right scores 0; certain-and-wrong scores 1."""
    assert brier_score([1.0, 0.0, 1.0], [1, 0, 1]) == pytest.approx(0.0)
    assert brier_score([1.0, 0.0], [0, 1]) == pytest.approx(1.0)


def test_brier_known_value_and_base_rate_property() -> None:
    """Hand-computed value, and constant 0.5 forecasts score exactly 0.25."""
    assert brier_score([0.8, 0.3], [1, 0]) == pytest.approx((0.04 + 0.09) / 2)
    assert brier_score([0.5] * 6, [1, 0, 1, 1, 0, 0]) == pytest.approx(0.25)


def test_brier_degenerate_inputs_are_nan() -> None:
    """Empty or misaligned inputs are undefined, not a misleading zero."""
    assert math.isnan(brier_score([], []))
    assert math.isnan(brier_score([0.5, 0.6], [1]))


# --------------------------------------------------------------------------- #
# Sharpness
# --------------------------------------------------------------------------- #
def test_sharpness_bounds() -> None:
    """Coin-flip confidences give 0, certain confidences give the maximum 0.5."""
    assert sharpness([0.5, 0.5, 0.5]) == pytest.approx(0.0)
    assert sharpness([0.0, 1.0, 1.0]) == pytest.approx(0.5)
    assert sharpness([0.75, 0.25]) == pytest.approx(0.25)
    rng = np.random.default_rng(0)
    assert 0.0 <= sharpness(rng.uniform(size=200)) <= 0.5
    assert math.isnan(sharpness([]))


def test_softening_lowers_sharpness_but_not_below_zero() -> None:
    """Pulling confidences towards 0.5 (as a large temperature does) reduces sharpness."""
    raw = np.array([0.9, 0.8, 0.2, 0.1])
    softened = 0.5 + 0.5 * (raw - 0.5)
    assert sharpness(softened) < sharpness(raw)


# --------------------------------------------------------------------------- #
# Adaptive (equal-mass) ECE
# --------------------------------------------------------------------------- #
def test_adaptive_ece_near_zero_for_calibrated_data() -> None:
    """Outcomes drawn from the stated probabilities are calibrated: adaptive ECE ~ 0."""
    rng = np.random.default_rng(1)
    conf = rng.beta(2.0, 2.0, size=20000)
    labels = (rng.uniform(size=conf.size) < conf).astype(int)
    assert adaptive_ece(conf, labels, n_bins=10) < 0.02


def test_adaptive_ece_detects_gross_miscalibration() -> None:
    """Always 0.9 confident and always wrong gives an ECE of 0.9 under any binning."""
    assert adaptive_ece([0.9] * 4, [0, 0, 0, 0]) == pytest.approx(0.9)
    assert adaptive_ece([0.9] * 4, [0, 0, 0, 0]) == pytest.approx(
        expected_calibration_error([0.9] * 4, [0, 0, 0, 0])
    )


def test_adaptive_ece_uses_equal_mass_bins() -> None:
    """Equal-mass bins weight the tails like the middle, unlike equal-width bins."""
    conf = np.array([0.01, 0.02, 0.03, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.99])
    labels = np.array([1, 1, 1, 1, 0, 1, 0, 1, 0, 0])
    # Two bins of five: [0.01..0.52] acc 4/5 conf 0.218 ; [0.53..0.99] acc 2/5 conf 0.634.
    expected = 0.5 * abs(0.8 - np.mean(conf[:5])) + 0.5 * abs(0.4 - np.mean(conf[5:]))
    assert adaptive_ece(conf, labels, n_bins=2) == pytest.approx(expected)


def test_adaptive_ece_degenerate_inputs() -> None:
    """Empty, misaligned or zero-bin inputs return 0.0 like ``expected_calibration_error``."""
    assert adaptive_ece([], []) == 0.0
    assert adaptive_ece([0.5, 0.6], [1]) == 0.0
    assert adaptive_ece([0.5], [1], n_bins=0) == 0.0
    assert 0.0 <= adaptive_ece([0.2, 0.9], [0, 1], n_bins=10) <= 1.0  # fewer points than bins


# --------------------------------------------------------------------------- #
# Hit rate
# --------------------------------------------------------------------------- #
def test_hit_rate() -> None:
    """Fraction of correct calls, accepting ints or booleans; undefined when empty."""
    assert hit_rate([1, 0, 1, 1]) == pytest.approx(0.75)
    assert hit_rate([True, False]) == pytest.approx(0.5)
    assert hit_rate(pd.Series([0, 0, 0])) == pytest.approx(0.0)
    assert math.isnan(hit_rate([]))


# --------------------------------------------------------------------------- #
# Equity-curve helpers
# --------------------------------------------------------------------------- #
def test_annualised_return_doubling_in_one_year() -> None:
    """A curve that doubles over 252 periods has a CAGR of exactly 100 %, over any window."""
    equity = 100.0 * 2.0 ** (np.arange(253) / 252.0)
    assert annualised_return(equity) == pytest.approx(1.0)
    # Half the window (126 periods, growth sqrt(2)) annualises to the same 100 %.
    assert annualised_return(pd.Series(equity[:127])) == pytest.approx(1.0)
    # Counting 126 periods per year, the same curve spans two years: CAGR = sqrt(2) - 1.
    assert annualised_return(equity, periods_per_year=126) == pytest.approx(2.0**0.5 - 1.0)


def test_annualised_return_degenerate_cases() -> None:
    """Flat curves give 0, too-short curves give 0, a wipe-out gives -1."""
    assert annualised_return([100.0, 100.0, 100.0]) == pytest.approx(0.0)
    assert annualised_return([100.0]) == 0.0
    assert annualised_return([]) == 0.0
    assert annualised_return([100.0, 50.0, 0.0]) == -1.0


def test_calmar_ratio_known_curve() -> None:
    """Calmar = annualised return / |max drawdown| on the crafted drawdown curve."""
    equity = [100.0, 120.0, 90.0, 150.0, 75.0, 150.0]
    assert max_drawdown(equity) == pytest.approx(-0.5)
    assert calmar_ratio(equity) == pytest.approx(annualised_return(equity) / 0.5)


def test_calmar_ratio_no_drawdown_and_degenerate() -> None:
    """No drawdown mirrors Sortino: +inf with positive growth, 0 otherwise."""
    assert calmar_ratio([100.0, 101.0, 102.0]) == math.inf
    assert calmar_ratio([100.0, 100.0]) == 0.0
    assert calmar_ratio([]) == 0.0
