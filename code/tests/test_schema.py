"""Tests for ``advisor.recommender.schema``.

Covers construction of the :class:`~advisor.recommender.schema.Recommendation`
dataclass and the JSON-serialisability contract of its ``to_dict`` method
(``Action`` -> ``str``, ``date`` -> ISO-8601 string).
"""

from __future__ import annotations

import dataclasses
import json
from datetime import date

import pytest

from advisor.recommender.schema import Action, Recommendation


def _make_recommendation(**overrides: object) -> Recommendation:
    """Build a baseline :class:`Recommendation`, applying any field overrides."""
    kwargs: dict = {
        "ticker": "AAA",
        "action": Action.BUY,
        "confidence": 0.71,
        "raw_confidence": 0.80,
        "reason": "Momentum positive and RSI not overbought.",
        "counterfactual": "I would change my mind if RSI exceeded 70.",
        "as_of": date(2025, 3, 14),
        "features": {"rsi_14": 57.3, "mom_10": 0.034},
    }
    kwargs.update(overrides)
    return Recommendation(**kwargs)  # type: ignore[arg-type]


def test_action_enum_values_are_strings() -> None:
    """``Action`` members are str-valued and round-trip from their value."""
    assert Action.BUY.value == "BUY"
    assert Action.HOLD.value == "HOLD"
    assert Action.SELL.value == "SELL"
    # str-Enum: members compare equal to their string value.
    assert Action.BUY == "BUY"
    assert Action("SELL") is Action.SELL


def test_recommendation_construction_fields() -> None:
    """All contract fields are stored verbatim on construction."""
    rec = _make_recommendation()
    assert rec.ticker == "AAA"
    assert rec.action is Action.BUY
    assert rec.confidence == pytest.approx(0.71)
    assert rec.raw_confidence == pytest.approx(0.80)
    assert rec.reason.startswith("Momentum")
    assert rec.counterfactual.startswith("I would change my mind")
    assert rec.as_of == date(2025, 3, 14)
    assert rec.features == {"rsi_14": 57.3, "mom_10": 0.034}


def test_recommendation_is_frozen() -> None:
    """The dataclass is frozen (immutable) per the contract."""
    rec = _make_recommendation()
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.confidence = 0.0  # type: ignore[misc]


def test_features_default_is_none() -> None:
    """``features`` defaults to ``None`` when omitted."""
    rec = _make_recommendation(features=None)
    assert rec.features is None


def test_to_dict_serialises_action_and_date() -> None:
    """``to_dict`` maps Action -> str and date -> ISO-8601 string."""
    rec = _make_recommendation()
    d = rec.to_dict()

    assert d["action"] == "BUY"
    assert isinstance(d["action"], str)
    assert d["as_of"] == "2025-03-14"
    assert isinstance(d["as_of"], str)

    # Scalar fields carried through unchanged.
    assert d["ticker"] == "AAA"
    assert d["confidence"] == pytest.approx(0.71)
    assert d["raw_confidence"] == pytest.approx(0.80)
    assert d["reason"] == rec.reason
    assert d["counterfactual"] == rec.counterfactual


def test_to_dict_is_json_serialisable() -> None:
    """The dict returned by ``to_dict`` survives a JSON round-trip."""
    rec = _make_recommendation()
    payload = json.dumps(rec.to_dict())
    restored = json.loads(payload)

    assert restored["action"] == "BUY"
    assert restored["as_of"] == "2025-03-14"
    assert restored["ticker"] == "AAA"


@pytest.mark.parametrize("action", [Action.BUY, Action.HOLD, Action.SELL])
def test_to_dict_round_trips_each_action(action: Action) -> None:
    """Every Action value serialises to its string name in ``to_dict``."""
    rec = _make_recommendation(action=action)
    d = rec.to_dict()
    assert d["action"] == action.value
    # And reconstructs back to the same enum member.
    assert Action(d["action"]) is action


def test_to_dict_with_none_features() -> None:
    """``to_dict`` handles a ``None`` features field cleanly."""
    rec = _make_recommendation(features=None)
    d = rec.to_dict()
    assert d["features"] is None
    json.dumps(d)  # must not raise
