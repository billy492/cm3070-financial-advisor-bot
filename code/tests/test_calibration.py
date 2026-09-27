"""Tests for ``advisor.calibration.temperature``.

The calibrator is exercised on synthetic data whose miscalibration is known
by construction: true probabilities are drawn from a Beta(2, 2), outcomes are
Bernoulli draws from them, and the *reported* confidences are sharpened
(over-confident, factor > 1) or flattened (under-confident, factor < 1) in
logit space. Temperature scaling must then learn ``T`` above / below one,
recover the sharpening factor, lower the held-out Expected Calibration Error,
and never move a confidence across 0.5 (the action-preserving property of
Guo et al. 2017). Everything is offline and seeded.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from advisor.calibration import (
    TemperatureScaler,
    plot_reliability_diagram,
    reliability_curve,
)
from advisor.evaluation.metrics import expected_calibration_error


def _synthetic(n: int, sharpen: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(confidences, labels)`` with logit-space sharpening ``sharpen``."""
    rng = np.random.default_rng(seed)
    p_true = rng.beta(2.0, 2.0, size=n)
    labels = (rng.uniform(size=n) < p_true).astype(int)
    logits = np.log(p_true) - np.log1p(-p_true)
    confidences = 1.0 / (1.0 + np.exp(-sharpen * logits))
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
def test_overconfident_source_learns_temperature_above_one() -> None:
    """Sharpened confidences need softening: the fitted T exceeds 1."""
    conf, labels = _synthetic(4000, sharpen=2.0)
    scaler = TemperatureScaler().fit(conf, labels)
    assert scaler.temperature is not None
    assert scaler.temperature > 1.0
    assert scaler.n_fit_ == 4000


def test_underconfident_source_learns_temperature_below_one() -> None:
    """Flattened confidences need sharpening: the fitted T is below 1."""
    conf, labels = _synthetic(4000, sharpen=0.5)
    assert TemperatureScaler().fit(conf, labels).temperature < 1.0


@pytest.mark.parametrize("sharpen", [2.0, 0.5])
def test_recovers_known_sharpening_factor(sharpen: float) -> None:
    """The NLL-optimal T undoes the construction: T is close to the sharpening factor."""
    conf, labels = _synthetic(20000, sharpen=sharpen, seed=1)
    scaler = TemperatureScaler().fit(conf, labels)
    assert scaler.temperature == pytest.approx(sharpen, rel=0.15)


def test_calibrated_source_keeps_temperature_near_one() -> None:
    """Already-calibrated confidences are left (almost) alone."""
    conf, labels = _synthetic(20000, sharpen=1.0, seed=2)
    assert 0.85 <= TemperatureScaler().fit(conf, labels).temperature <= 1.15


def test_held_out_ece_decreases_after_calibration() -> None:
    """Fitting on one half lowers the ECE measured on the other half (H2 of ADR-0002)."""
    conf, labels = _synthetic(10000, sharpen=2.0, seed=3)
    fit_c, fit_y, test_c, test_y = _halves(conf, labels)
    scaler = TemperatureScaler().fit(fit_c, fit_y)
    before = expected_calibration_error(test_c, test_y)
    after = expected_calibration_error(scaler.transform(test_c), test_y)
    assert after < before
    assert after < 0.03


def test_nll_after_is_not_above_nll_before() -> None:
    """With T = 1 inside the search bounds the optimum can only improve the NLL."""
    conf, labels = _synthetic(3000, sharpen=2.0, seed=4)
    scaler = TemperatureScaler().fit(conf, labels)
    assert scaler.nll_before_ is not None and scaler.nll_after_ is not None
    assert scaler.nll_after_ <= scaler.nll_before_ + 1e-12


def test_temperature_respects_search_bounds() -> None:
    """A too-narrow search interval pins T to its ceiling, mirroring the report's T = 20 case."""
    conf, labels = _synthetic(3000, sharpen=3.0, seed=5)
    scaler = TemperatureScaler(t_min=0.5, t_max=1.5).fit(conf, labels)
    assert scaler.temperature == pytest.approx(1.5, abs=1e-6)


# --------------------------------------------------------------------------- #
# Transform properties
# --------------------------------------------------------------------------- #
def test_transform_preserves_action_and_ranking() -> None:
    """``p > 0.5`` iff calibrated ``> 0.5``, and the ordering of confidences is unchanged."""
    conf, labels = _synthetic(2000, sharpen=2.0, seed=6)
    scaler = TemperatureScaler().fit(conf, labels)
    raw = np.linspace(0.01, 0.99, 99)
    calibrated = np.asarray(scaler.transform(raw))
    assert np.array_equal(raw > 0.5, calibrated > 0.5)
    assert np.all(np.diff(calibrated) > 0.0)


def test_half_is_a_fixed_point_for_any_temperature() -> None:
    """The sigmoid crosses one-half where the logit crosses zero, so 0.5 never moves."""
    for temperature in (0.1, 1.0, 7.5):
        assert TemperatureScaler(temperature=temperature).transform([0.5])[0] == pytest.approx(
            0.5, abs=1e-12
        )


def test_identity_temperature_is_a_no_op() -> None:
    """``T = 1`` returns the (clipped) inputs unchanged."""
    raw = [0.05, 0.3, 0.5, 0.77, 0.99]
    assert TemperatureScaler(temperature=1.0).transform(raw) == pytest.approx(raw, abs=1e-9)


def test_transform_softens_when_temperature_above_one() -> None:
    """``T > 1`` pulls confidences towards 0.5; ``T < 1`` pushes them away."""
    assert TemperatureScaler(temperature=2.0).transform([0.9])[0] < 0.9
    assert TemperatureScaler(temperature=0.5).transform([0.9])[0] > 0.9


def test_transform_handles_endpoints_and_empty_input() -> None:
    """Exact 0 and 1 are clipped (finite logits) and an empty input yields an empty list."""
    scaler = TemperatureScaler(temperature=2.0)
    out = scaler.transform([0.0, 1.0])
    assert 0.0 < out[0] < 0.5 < out[1] < 1.0
    assert scaler.transform([]) == []


def test_fit_transform_matches_fit_then_transform() -> None:
    """``fit_transform`` is exactly ``fit(...).transform(...)`` on the same data."""
    conf, labels = _synthetic(500, sharpen=1.5, seed=7)
    a = TemperatureScaler().fit_transform(conf, labels)
    b = TemperatureScaler().fit(conf, labels).transform(conf)
    assert a == pytest.approx(b)


# --------------------------------------------------------------------------- #
# Validation errors
# --------------------------------------------------------------------------- #
def test_transform_before_fit_raises() -> None:
    """An unfitted scaler refuses to transform."""
    with pytest.raises(RuntimeError):
        TemperatureScaler().transform([0.6])


@pytest.mark.parametrize(
    ("confidences", "labels"),
    [
        ([], []),
        ([0.5, 0.6], [1]),
        ([0.5, 0.6], [0, 2]),
        ([0.5, 0.6, 0.7], [1, 1, 1]),
        ([0.5, float("nan")], [0, 1]),
        ([0.5, 1.5], [0, 1]),
    ],
    ids=["empty", "mismatched", "non-binary", "single-class", "nan", "out-of-range"],
)
def test_fit_rejects_bad_input(confidences: list[float], labels: list[int]) -> None:
    """Empty, misaligned, non-binary, single-class or invalid confidences are errors."""
    with pytest.raises(ValueError):
        TemperatureScaler().fit(confidences, labels)


@pytest.mark.parametrize(
    "kwargs",
    [{"t_min": 2.0, "t_max": 1.0}, {"t_min": 0.0}, {"eps": 0.0}, {"temperature": -1.0}],
)
def test_constructor_rejects_bad_parameters(kwargs: dict) -> None:
    """Bounds, clipping margin and a pre-set temperature are validated."""
    with pytest.raises(ValueError):
        TemperatureScaler(**kwargs)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def test_to_dict_from_dict_round_trip_through_json() -> None:
    """A fitted scaler survives ``json.dumps`` / ``json.loads`` with identical behaviour."""
    conf, labels = _synthetic(1000, sharpen=2.0, seed=8)
    scaler = TemperatureScaler(t_min=0.1, t_max=10.0).fit(conf, labels)
    restored = TemperatureScaler.from_dict(json.loads(json.dumps(scaler.to_dict())))
    assert restored.temperature == pytest.approx(scaler.temperature)
    assert restored.n_fit_ == scaler.n_fit_
    assert restored.nll_before_ == pytest.approx(scaler.nll_before_)
    assert restored.nll_after_ == pytest.approx(scaler.nll_after_)
    assert (restored.t_min, restored.t_max, restored.eps) == (0.1, 10.0, scaler.eps)
    assert restored.transform([0.2, 0.9]) == pytest.approx(scaler.transform([0.2, 0.9]))


def test_from_dict_unfitted_payload() -> None:
    """An unfitted payload restores an unfitted scaler."""
    restored = TemperatureScaler.from_dict(TemperatureScaler().to_dict())
    assert restored.temperature is None
    assert "unfitted" in repr(restored)


# --------------------------------------------------------------------------- #
# Reliability diagrams
# --------------------------------------------------------------------------- #
def test_reliability_curve_uniform_shapes_and_totals() -> None:
    """Equal-width bins: aligned arrays, counts summing to n, ascending confidences."""
    conf, labels = _synthetic(1000, sharpen=1.0, seed=9)
    mean_conf, mean_acc, count = reliability_curve(conf, labels, n_bins=10)
    assert mean_conf.shape == mean_acc.shape == count.shape
    assert 1 <= count.size <= 10
    assert int(count.sum()) == 1000
    assert np.all(np.diff(mean_conf) > 0.0)
    assert np.all((mean_acc >= 0.0) & (mean_acc <= 1.0))


def test_reliability_curve_quantile_bins_have_equal_mass() -> None:
    """Equal-mass bins differ in size by at most one observation."""
    conf, labels = _synthetic(1003, sharpen=2.0, seed=10)
    _, _, count = reliability_curve(conf, labels, n_bins=10, strategy="quantile")
    assert count.size == 10
    assert count.max() - count.min() <= 1


def test_reliability_curve_matches_ece_definition() -> None:
    """The count-weighted gap of the uniform curve is exactly the ECE."""
    conf, labels = _synthetic(800, sharpen=2.0, seed=11)
    mean_conf, mean_acc, count = reliability_curve(conf, labels, n_bins=10)
    ece_from_curve = float(np.sum(count / count.sum() * np.abs(mean_acc - mean_conf)))
    assert ece_from_curve == pytest.approx(expected_calibration_error(conf, labels, 10))


def test_reliability_curve_degenerate_and_invalid_inputs() -> None:
    """Empty input gives empty arrays; bad strategy, bin count or lengths raise."""
    mean_conf, mean_acc, count = reliability_curve([], [])
    assert mean_conf.size == mean_acc.size == count.size == 0
    with pytest.raises(ValueError):
        reliability_curve([0.5], [1], strategy="sturges")
    with pytest.raises(ValueError):
        reliability_curve([0.5], [1], n_bins=0)
    with pytest.raises(ValueError):
        reliability_curve([0.5, 0.6], [1])


def test_plot_reliability_diagram_writes_png_and_svg(tmp_path: Path) -> None:
    """Both image formats are written next to each other, under a created folder."""
    conf, labels = _synthetic(600, sharpen=2.0, seed=12)
    scaler = TemperatureScaler().fit(conf, labels)
    curves = {
        "raw": reliability_curve(conf, labels),
        "calibrated": reliability_curve(scaler.transform(conf), labels),
    }
    png, svg = plot_reliability_diagram(curves, tmp_path / "figs" / "reliability.png")
    assert png == tmp_path / "figs" / "reliability.png"
    assert svg == tmp_path / "figs" / "reliability.svg"
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert b"<svg" in svg.read_bytes()


def test_plot_reliability_diagram_requires_a_curve(tmp_path: Path) -> None:
    """An empty curve mapping is an error rather than an empty figure."""
    with pytest.raises(ValueError):
        plot_reliability_diagram({}, tmp_path / "empty")
