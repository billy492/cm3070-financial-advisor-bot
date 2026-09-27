"""Tests for ``advisor.recommender.heuristic`` -- the transparent rule baseline.

The rule table (BUY / SELL / HOLD), the confidence formula, the wording of the
reason and counterfactual, input validation, calibration and the
``predict`` interface are all exercised offline.
"""

from __future__ import annotations

from datetime import date

import pytest

from advisor.recommender.heuristic import (
    HeuristicAdvisor,
    heuristic_action,
    heuristic_confidence,
)
from advisor.recommender.schema import Action, Recommendation

AS_OF = date(2025, 6, 2)


def _feats(mom_10: float, close: float, sma_50: float, **extra: float) -> dict[str, float]:
    base = {"close": close, "sma_50": sma_50, "mom_10": mom_10, "rsi_14": 55.0, "vol_20": 0.015}
    base.update(extra)
    return base


# --------------------------------------------------------------------------- #
# Rule table
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("mom_10", "close", "sma_50", "expected"),
    [
        (0.03, 105.0, 100.0, Action.BUY),  # momentum up, price above average
        (-0.03, 95.0, 100.0, Action.SELL),  # momentum down, price below average
        (0.03, 95.0, 100.0, Action.HOLD),  # up momentum but below average: mixed
        (-0.03, 105.0, 100.0, Action.HOLD),  # down momentum but above average: mixed
        (0.0, 105.0, 100.0, Action.HOLD),  # zero momentum never BUY/SELL
        (0.03, 100.0, 100.0, Action.HOLD),  # price exactly at the average
        (-0.03, 100.0, 100.0, Action.HOLD),
    ],
)
def test_rule_table(mom_10: float, close: float, sma_50: float, expected: Action) -> None:
    """``heuristic_action`` and ``recommend`` agree with the documented rule."""
    assert heuristic_action(close, sma_50, mom_10) is expected
    rec = HeuristicAdvisor().recommend("AAA", _feats(mom_10, close, sma_50), as_of=AS_OF)
    assert rec.action is expected


@pytest.mark.parametrize(
    ("mom_10", "expected"),
    [(0.0, 0.5), (0.05, 0.7), (-0.05, 0.7), (0.1125, 0.95), (0.5, 0.95), (-1.0, 0.95)],
)
def test_confidence_formula(mom_10: float, expected: float) -> None:
    """Confidence is ``clip(0.5 + 4|mom_10|, 0.5, 0.95)``."""
    assert heuristic_confidence(mom_10) == pytest.approx(expected)
    rec = HeuristicAdvisor().recommend("AAA", _feats(mom_10, 105.0, 100.0), as_of=AS_OF)
    assert rec.confidence == pytest.approx(expected)
    assert rec.raw_confidence == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# Recommendation contents
# --------------------------------------------------------------------------- #
def test_recommendation_fields_and_wording() -> None:
    """Ticker is upper-cased, reason cites it, counterfactual has the required opening."""
    feats = _feats(0.034, 187.15, 179.2)
    rec = HeuristicAdvisor().recommend("aapl", feats, as_of=AS_OF)

    assert isinstance(rec, Recommendation)
    assert rec.ticker == "AAPL"
    assert rec.as_of == AS_OF
    assert "AAPL" in rec.reason
    assert "+0.0340" in rec.reason
    assert rec.counterfactual.startswith("I would change my mind if")
    assert rec.features == feats


def test_counterfactual_names_the_flipping_indicator() -> None:
    """Each branch's counterfactual points at momentum / the 50-day average."""
    advisor = HeuristicAdvisor()
    buy = advisor.recommend("A", _feats(0.05, 110, 100), as_of=AS_OF).counterfactual
    sell = advisor.recommend("A", _feats(-0.05, 90, 100), as_of=AS_OF).counterfactual
    hold = advisor.recommend("A", _feats(0.05, 90, 100), as_of=AS_OF).counterfactual
    assert "momentum turned negative" in buy
    assert "momentum turned positive" in sell
    assert "agreed in the same direction" in hold


def test_predict_returns_plain_action_string() -> None:
    """``predict`` mirrors ``recommend`` but returns only the action string."""
    advisor = HeuristicAdvisor()
    assert advisor.predict(_feats(0.05, 110, 100)) == "BUY"
    assert advisor.predict(_feats(-0.05, 90, 100)) == "SELL"
    assert advisor.predict(_feats(0.0, 90, 100)) == "HOLD"
    assert type(advisor.predict(_feats(0.05, 110, 100))) is str


def test_metadata_and_health() -> None:
    """Model tag / prompt version are fixed strings and health is always True."""
    advisor = HeuristicAdvisor()
    assert advisor.model_tag == "heuristic-v1"
    assert advisor.model == "heuristic-v1"
    assert advisor.prompt_version == "rule-v1"
    assert advisor.health() is True


# --------------------------------------------------------------------------- #
# Validation and calibration
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("missing", ["close", "sma_50", "mom_10"])
def test_missing_required_feature_raises(missing: str) -> None:
    """Each of the three rule inputs is mandatory."""
    feats = _feats(0.05, 110, 100)
    del feats[missing]
    with pytest.raises(ValueError, match=missing):
        HeuristicAdvisor().recommend("A", feats, as_of=AS_OF)


def test_nan_and_non_numeric_features_raise() -> None:
    """NaN (unwarmed window) and non-numeric values are rejected."""
    with pytest.raises(ValueError, match="NaN"):
        HeuristicAdvisor().recommend("A", _feats(float("nan"), 110, 100), as_of=AS_OF)
    with pytest.raises(ValueError, match="not numeric"):
        HeuristicAdvisor().recommend("A", {**_feats(0.05, 110, 100), "close": "n/a"}, as_of=AS_OF)


class _HalfCalibrator:
    def transform(self, confidences: list[float]) -> list[float]:
        return [0.5 + (c - 0.5) / 2 for c in confidences]


def test_calibrator_is_applied_to_confidence_only() -> None:
    """With a calibrator, ``confidence`` is transformed and ``raw_confidence`` kept."""
    rec = HeuristicAdvisor(calibrator=_HalfCalibrator()).recommend(
        "A", _feats(0.05, 110, 100), as_of=AS_OF
    )
    assert rec.raw_confidence == pytest.approx(0.7)
    assert rec.confidence == pytest.approx(0.6)


def test_accepts_pandas_row(single_ticker_prices) -> None:  # type: ignore[no-untyped-def]
    """A feature row from ``compute_features`` (a Series) is accepted directly."""
    from advisor.features.indicators import compute_features

    row = compute_features(single_ticker_prices).dropna().iloc[-1]
    rec = HeuristicAdvisor().recommend("AAA", row, as_of=AS_OF)
    assert rec.action in (Action.BUY, Action.HOLD, Action.SELL)
    assert rec.features is not None and "rsi_14" in rec.features
