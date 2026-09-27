"""Typed data structures for advisor recommendations.

This module defines the canonical output contract of the recommender layer:
the :class:`Action` enum and the immutable :class:`Recommendation` dataclass.
Every component downstream of the LLM (calibration, counterfactual generation,
backtesting, the web UI) consumes :class:`Recommendation` objects, so this
module is deliberately dependency-free and import-safe (standard library only).

See ADR-0002 for the decision to emit ``BUY``/``HOLD``/``SELL`` actions with a
verbal confidence, plain-English reason, and a counterfactual ("I would change
my mind if ...") per recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum

__all__ = ["Action", "Recommendation"]


class Action(str, Enum):  # noqa: UP042 -- str mixin kept: StrEnum changes str(), which cache keys rely on
    """Discrete advisor action.

    Subclasses :class:`str` so that members compare equal to their string
    value (``Action.BUY == "BUY"``) and serialize transparently to JSON.
    """

    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"

    def __str__(self) -> str:  # pragma: no cover - trivial
        """Return the bare action string (e.g. ``"BUY"``) rather than ``Action.BUY``."""
        return self.value


@dataclass(frozen=True)
class Recommendation:
    """A single, immutable advisor recommendation for one ticker.

    Attributes:
        ticker: Upper-case US equity symbol the recommendation concerns.
        action: The recommended :class:`Action` (BUY/HOLD/SELL).
        confidence: Post-hoc calibrated probability in ``[0, 1]`` that the
            action is "correct" (temperature-scaled per Guo et al. 2017,
            ADR-0002). Until calibration is fitted this equals ``raw_confidence``.
        raw_confidence: The model's verbal / pre-calibration confidence in
            ``[0, 1]`` as emitted by the LLM, before any calibration map.
        reason: Plain-English justification suitable for a non-technical user.
        counterfactual: A "I would change my mind if ..." statement describing
            the minimal feature change that would flip the action
            (Wachter et al. 2018 framing; DiCE per Mothilal et al. 2020).
        as_of: The trading date the recommendation was generated for. No
            feature used to produce it may post-date this day (no lookahead).
        features: Optional snapshot of the engineered features the model saw,
            kept for auditing, calibration datasets, and counterfactual search.
    """

    ticker: str
    action: Action
    confidence: float
    raw_confidence: float
    reason: str
    counterfactual: str
    as_of: date
    features: dict | None = field(default=None)

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict representation.

        Enum members are reduced to their string value and the ``as_of`` date
        to an ISO-8601 string so the result can be passed directly to
        :func:`json.dumps` without a custom encoder.

        Returns:
            A plain ``dict`` with primitive, JSON-safe values.
        """
        return {
            "ticker": self.ticker,
            "action": self.action.value,
            "confidence": float(self.confidence),
            "raw_confidence": float(self.raw_confidence),
            "reason": self.reason,
            "counterfactual": self.counterfactual,
            "as_of": self.as_of.isoformat(),
            "features": self.features,
        }
