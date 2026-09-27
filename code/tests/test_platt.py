"""Tests for ``advisor.calibration.platt``.

Platt scaling is exercised on synthetic data whose miscalibration is known by
construction: true probabilities are drawn from a Beta(2, 2), outcomes are
Bernoulli draws from them, and the *reported* confidences are distorted in
logit space by a known slope and shift. The scaler must recover the inverse
map, lower the held-out ECE, and - the property that motivates it - fit a
correctness rate *below one half*, which temperature scaling structurally
cannot. Everything is offline and seeded.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from advisor.calibration import PlattScaler, TemperatureScaler
from advisor.evaluation.metrics import expected_calibration_error


def _synthetic(n: int, slope: float, shift: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(confidences, labels)`` with reported logit ``slope * z + shift``."""
    rng = np.random.default_rng(seed)
    p_true = rng.beta(2.0, 2.0, size=n)
    labels = (rng.uniform(size=n) < p_true).astype(int)
    logits = np.log(p_true) - np.log1p(-p_true)
    confidences = 1.0 / (1.0 + np.exp(-(slope * logits + shift)))
    return confidences, labels


def _halves(
    conf: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split into a fitting half and a held-out half."""
    half = conf.size // 2
    return conf[:half], labels[:half], conf[half:], labels[half:]


# --------------------------------------------------------------------------- #
# Fitting behaviour
# --------------------------------------------------------------------------- #
def test_recovers_inverse_affine_map() -> None:
    """Reported logit ``2z + 1`` needs ``a = 1/2, b = -1/2`` to undo it."""
    conf, labels = _synthetic(20000, slope=2.0, shift=1.0)
    scaler = PlattScaler(ridge=0.0).fit(conf, labels)
    assert scaler.slope == pytest.approx(0.5, abs=0.06)
    assert scaler.intercept == pytest.approx(-0.5, abs=0.06)
    assert scaler.n_fit_ == 20000


def test_already_calibrated_source_stays_near_identity() -> None:
    """Honest confidences give a slope near 1 and an intercept near 0."""
    conf, labels = _synthetic(20000, slope=1.0, shift=0.0, seed=1)
    scaler = PlattScaler().fit(conf, labels)
    assert scaler.slope == pytest.approx(1.0, abs=0.06)
    assert scaler.intercept == pytest.approx(0.0, abs=0.06)


def test_fit_never_increases_in_sample_nll() -> None:
    """The fitted map is at least as good as the identity on the fitting data."""
    conf, labels = _synthetic(3000, slope=1.7, shift=-0.4, seed=2)
    scaler = PlattScaler().fit(conf, labels)
    assert scaler.nll_before_ is not None and scaler.nll_after_ is not None
    assert scaler.nll_after_ <= scaler.nll_before_ + 1e-9


def test_lowers_held_out_ece() -> None:
    """Calibrating on one half lowers the ECE on the other half."""
    conf, labels = _synthetic(8000, slope=2.5, shift=0.8, seed=3)
    c_fit, y_fit, c_test, y_test = _halves(conf, labels)
    scaler = PlattScaler().fit(c_fit, y_fit)
    ece_raw = expected_calibration_error(c_test, y_test)
    ece_cal = expected_calibration_error(scaler.transform(c_test), y_test)
    assert ece_cal < ece_raw * 0.5


def test_fits_base_rate_below_half_where_temperature_cannot() -> None:
    """The motivating case: confident advisor, correct 30% of the time.

    Temperature scaling fixes 0.5 and can only shrink towards it, so its mean
    calibrated confidence stays at or above 0.5. Platt scaling has an intercept
    and lands near the true 30% hit rate.
    """
    rng = np.random.default_rng(4)
    conf = np.clip(rng.normal(0.72, 0.05, size=3000), 0.55, 0.9)
    labels = (rng.uniform(size=3000) < 0.30).astype(int)

    platt = PlattScaler().fit(conf, labels)
    temp = TemperatureScaler().fit(conf, labels)
    mean_platt = float(np.mean(platt.transform(conf)))
    mean_temp = float(np.mean(temp.transform(conf)))

    assert mean_platt == pytest.approx(0.30, abs=0.03)
    assert mean_temp >= 0.5 - 1e-9
    assert temp.temperature == pytest.approx(temp.t_max, rel=1e-3)
    assert expected_calibration_error(platt.transform(conf), labels) < 0.05


def test_map_is_monotone_for_positive_slope() -> None:
    """Ranking of confidences is preserved when the slope is positive."""
    conf, labels = _synthetic(2000, slope=1.5, shift=0.3, seed=5)
    scaler = PlattScaler().fit(conf, labels)
    assert scaler.slope > 0.0
    grid = np.linspace(0.01, 0.99, 50)
    out = np.asarray(scaler.transform(grid))
    assert np.all(np.diff(out) > 0.0)


def test_fit_transform_matches_fit_then_transform() -> None:
    """``fit_transform`` is ``fit`` followed by ``transform`` on the same data."""
    conf, labels = _synthetic(500, slope=1.3, shift=0.2, seed=6)
    a = PlattScaler().fit_transform(conf, labels)
    b = PlattScaler().fit(conf, labels).transform(conf)
    assert a == pytest.approx(b)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_transform_before_fit_raises() -> None:
    """An unfitted scaler cannot transform."""
    with pytest.raises(RuntimeError, match="not fitted"):
        PlattScaler().transform([0.5])


def test_single_class_labels_rejected() -> None:
    """Both outcomes are needed for a meaningful fit."""
    with pytest.raises(ValueError, match="both classes"):
        PlattScaler().fit([0.6, 0.7, 0.8], [1, 1, 1])


@pytest.mark.parametrize(
    ("conf", "labels", "match"),
    [
        ([], [], "at least one"),
        ([0.5, 0.6], [1], "same length"),
        ([0.5, 1.5], [0, 1], "lie in"),
        ([0.5, float("nan")], [0, 1], "finite"),
        ([0.5, 0.6], [0, 2], "binary"),
    ],
)
def test_bad_inputs_rejected(conf: list[float], labels: list[int], match: str) -> None:
    """Empty, misaligned, out-of-range, non-finite and non-binary inputs fail."""
    with pytest.raises(ValueError, match=match):
        PlattScaler().fit(conf, labels)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ridge": -1.0},
        {"eps": 0.5},
        {"slope": 1.0},
        {"intercept": 0.0},
        {"slope": float("inf"), "intercept": 0.0},
    ],
)
def test_bad_constructor_arguments_rejected(kwargs: dict[str, float]) -> None:
    """Negative ridge, bad eps, half-specified or non-finite parameters fail."""
    with pytest.raises(ValueError):
        PlattScaler(**kwargs)


def test_preset_parameters_work_without_fit() -> None:
    """Slope 1, intercept 0 is the identity map."""
    scaler = PlattScaler(slope=1.0, intercept=0.0)
    assert scaler.transform([0.2, 0.5, 0.8]) == pytest.approx([0.2, 0.5, 0.8], abs=1e-6)
    assert scaler.transform([]) == []


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def test_json_round_trip(tmp_path: Path) -> None:
    """``to_dict`` / ``from_dict`` survive a trip through JSON unchanged."""
    conf, labels = _synthetic(1000, slope=1.8, shift=0.5, seed=7)
    scaler = PlattScaler().fit(conf, labels)
    path = tmp_path / "platt.json"
    path.write_text(json.dumps(scaler.to_dict()))
    restored = PlattScaler.from_dict(json.loads(path.read_text()))
    assert restored.slope == pytest.approx(scaler.slope)
    assert restored.intercept == pytest.approx(scaler.intercept)
    assert restored.n_fit_ == scaler.n_fit_
    assert restored.nll_after_ == pytest.approx(scaler.nll_after_)
    assert restored.transform(conf[:20]) == pytest.approx(scaler.transform(conf[:20]))
    assert "slope=" in repr(restored)
    assert "unfitted" in repr(PlattScaler())
