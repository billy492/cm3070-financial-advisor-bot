"""Regression tests for the Deflated Sharpe Ratio's cross-trial variance.

The June scaffold hard-coded the cross-trial variance of the Sharpe estimates
to 1 (a per-period Sharpe of ~0.05 is then always deflated to 0). These tests
pin the corrected behaviour: the benchmark ``SR0`` scales with the spread of
the compared strategies' Sharpe ratios (Bailey & Lopez de Prado 2014), and a
skilful strategy is no longer deflated to zero.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from advisor.evaluation.metrics import deflated_sharpe_ratio, sharpe_ratio


@pytest.fixture
def skilful_returns() -> np.ndarray:
    """Daily returns with an annualised Sharpe of roughly 1.3 over one year."""
    rng = np.random.default_rng(7)
    noise = rng.normal(size=352)
    # Standardise the draw so the series has an exact mean and standard
    # deviation regardless of sampling noise: per-period SR = 0.08.
    noise = (noise - noise.mean()) / noise.std(ddof=1)
    return 0.0008 + 0.01 * noise


def test_dsr_is_not_degenerate_for_a_skilful_strategy(skilful_returns: np.ndarray) -> None:
    """A Sharpe near 1.3 with five trials should keep a healthy DSR, not 0."""
    dsr = deflated_sharpe_ratio(skilful_returns, n_trials=5)
    assert 0.5 < dsr < 1.0


def test_trial_sharpes_spread_lowers_dsr(skilful_returns: np.ndarray) -> None:
    """A wider spread of trial Sharpes raises SR0 and therefore lowers the DSR."""
    sr = sharpe_ratio(skilful_returns)
    tight = deflated_sharpe_ratio(
        skilful_returns, n_trials=5, trial_sharpes=[sr, sr * 0.9, sr * 1.05, sr * 0.95, sr]
    )
    wide = deflated_sharpe_ratio(
        skilful_returns, n_trials=5, trial_sharpes=[sr, -1.0, 2.5, 0.2, 1.8]
    )
    assert 0.0 <= wide < tight <= 1.0


def test_explicit_sr_variance_overrides_trial_sharpes(skilful_returns: np.ndarray) -> None:
    """``sr_variance`` takes priority and a huge variance deflates towards zero."""
    small = deflated_sharpe_ratio(skilful_returns, n_trials=5, sr_variance=1e-6)
    huge = deflated_sharpe_ratio(
        skilful_returns, n_trials=5, sr_variance=1.0, trial_sharpes=[1, 1, 1, 1, 1]
    )
    assert huge < 0.05 < small


def test_single_trial_has_no_deflation_benchmark(skilful_returns: np.ndarray) -> None:
    """With one trial SR0 = 0, so the DSR is simply P(true SR > 0)."""
    one = deflated_sharpe_ratio(skilful_returns, n_trials=1)
    many = deflated_sharpe_ratio(skilful_returns, n_trials=50)
    assert many < one


def test_degenerate_inputs_still_return_zero() -> None:
    """Constant or too-short series keep the documented 0.0 guard."""
    assert deflated_sharpe_ratio([0.01, 0.01, 0.01], n_trials=3) == 0.0
    assert deflated_sharpe_ratio([0.01], n_trials=3) == 0.0
    assert math.isfinite(deflated_sharpe_ratio([0.01, -0.02, 0.03, 0.0], n_trials=2))
