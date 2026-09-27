"""Transparent rule-based advisor (the "no AI" baseline).

:class:`HeuristicAdvisor` implements the momentum/trend rule that
``python -m advisor.reproduce`` has always used for its offline smoke test,
behind exactly the same ``recommend`` / ``predict`` interface as
:class:`advisor.recommender.llm.OllamaAdvisor`. That lets the Streamlit UI,
the LLM cache, the counterfactual search and the walk-forward backtest swap
the LLM for a fully deterministic, explainable rule - which is also the
"LLM stage removed" arm of the ablation study in the evaluation design.

Rule:
    * ``mom_10 > 0`` and ``close > sma_50``  ->  ``BUY``
    * ``mom_10 < 0`` and ``close < sma_50``  ->  ``SELL``
    * otherwise                              ->  ``HOLD``

Confidence is a crude, monotone function of momentum magnitude,
``clip(0.5 + 4 * |mom_10|, 0.5, 0.95)``; it is *not* calibrated unless a
``calibrator`` is supplied, exactly as for the LLM advisor.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Any

from advisor.recommender.schema import Action, Recommendation

__all__ = ["HeuristicAdvisor", "heuristic_action", "heuristic_confidence"]

#: Features the rule needs. Others (rsi_14, vol_20, ...) are passed through
#: untouched onto ``Recommendation.features`` for auditing.
REQUIRED_KEYS: tuple[str, ...] = ("close", "sma_50", "mom_10")


def heuristic_action(close: float, sma_50: float, mom_10: float) -> Action:
    """Apply the trend-following rule to three numbers.

    Args:
        close: Latest closing price.
        sma_50: 50-day simple moving average of the close.
        mom_10: 10-day momentum (fractional change over 10 trading days).

    Returns:
        ``Action.BUY`` when momentum is positive and price is above the 50-day
        average, ``Action.SELL`` when both are negative/below, else
        ``Action.HOLD``.
    """
    if mom_10 > 0 and close > sma_50:
        return Action.BUY
    if mom_10 < 0 and close < sma_50:
        return Action.SELL
    return Action.HOLD


def heuristic_confidence(mom_10: float) -> float:
    """Map momentum magnitude to a confidence in ``[0.5, 0.95]``.

    Args:
        mom_10: 10-day momentum (fractional change over 10 trading days).

    Returns:
        ``clip(0.5 + 4 * |mom_10|, 0.5, 0.95)``.
    """
    return float(min(0.95, max(0.5, 0.5 + abs(float(mom_10)) * 4.0)))


class HeuristicAdvisor:
    """Deterministic rule-based advisor with the ``OllamaAdvisor`` interface.

    Args:
        calibrator: Optional object with ``transform([p]) -> [p']``. When set,
            ``Recommendation.confidence`` is the calibrated value;
            ``raw_confidence`` always keeps the rule's confidence.

    Attributes:
        model_tag: ``"heuristic-v1"`` - the identifier used in cache keys and
            result file names.
        prompt_version: ``"rule-v1"``; bump if the rule changes.
    """

    model_tag: str = "heuristic-v1"
    prompt_version: str = "rule-v1"

    def __init__(self, *, calibrator: Any | None = None) -> None:
        """Create the rule-based advisor, optionally with a confidence calibrator."""
        self.calibrator = calibrator

    @property
    def model(self) -> str:
        """Alias of :attr:`model_tag` for parity with :class:`OllamaAdvisor`."""
        return self.model_tag

    def health(self, *, require_model: bool = True) -> bool:
        """Always ``True``: the rule needs no server.

        Args:
            require_model: Ignored; present for interface parity.

        Returns:
            ``True``.
        """
        return True

    def recommend(
        self,
        ticker: str,
        features: Mapping[str, Any],
        *,
        as_of: date,
    ) -> Recommendation:
        """Apply the rule and wrap the outcome in a :class:`Recommendation`.

        Args:
            ticker: US equity symbol (stamped upper-case on the result).
            features: Mapping containing at least ``close``, ``sma_50`` and
                ``mom_10``. The full mapping (values cast to ``float`` where
                possible) is stored on ``Recommendation.features``.
            as_of: Trading date the recommendation is generated for.

        Returns:
            A :class:`Recommendation` with a plain-English reason and a
            counterfactual naming the indicator that would flip the call.

        Raises:
            ValueError: If a required feature is missing or not numeric.
        """
        values = self._required_values(features)
        close, sma_50, mom_10 = values["close"], values["sma_50"], values["mom_10"]
        symbol = ticker.upper()

        action = heuristic_action(close, sma_50, mom_10)
        if action is Action.BUY:
            reason = (
                f"{symbol} shows positive 10-day momentum ({mom_10:+.4f}) and trades "
                f"above its 50-day average price ({close:.2f} > {sma_50:.2f}), a "
                "bullish trend-following signal."
            )
            counterfactual = (
                "I would change my mind if 10-day momentum turned negative or the "
                "close fell below the 50-day average price."
            )
        elif action is Action.SELL:
            reason = (
                f"{symbol} shows negative 10-day momentum ({mom_10:+.4f}) and trades "
                f"below its 50-day average price ({close:.2f} < {sma_50:.2f}), a "
                "bearish trend-following signal."
            )
            counterfactual = (
                "I would change my mind if 10-day momentum turned positive or the "
                "close rose back above the 50-day average price."
            )
        else:
            reason = (
                f"{symbol} sends mixed signals (momentum {mom_10:+.4f}; close "
                f"{close:.2f} vs 50-day average price {sma_50:.2f}); no decisive trend."
            )
            counterfactual = (
                "I would change my mind if momentum and the 50-day average price "
                "trend agreed in the same direction."
            )

        raw_confidence = heuristic_confidence(mom_10)
        confidence = raw_confidence
        if self.calibrator is not None:
            confidence = float(self.calibrator.transform([raw_confidence])[0])
            confidence = min(1.0, max(0.0, confidence))

        return Recommendation(
            ticker=symbol,
            action=action,
            confidence=confidence,
            raw_confidence=raw_confidence,
            reason=reason,
            counterfactual=counterfactual,
            as_of=as_of,
            features=self._snapshot(features),
        )

    def predict(self, features: Mapping[str, Any]) -> str:
        """Return only the action string for a feature vector.

        Args:
            features: Mapping with at least ``close``, ``sma_50``, ``mom_10``.

        Returns:
            ``"BUY"``, ``"HOLD"`` or ``"SELL"``.
        """
        return self.recommend("X", features, as_of=date.today()).action.value

    @staticmethod
    def _required_values(features: Mapping[str, Any]) -> dict[str, float]:
        """Extract and validate the three inputs of the rule.

        Args:
            features: Feature mapping.

        Returns:
            ``{"close": ..., "sma_50": ..., "mom_10": ...}`` as floats.

        Raises:
            ValueError: If a key is missing, non-numeric or NaN.
        """
        out: dict[str, float] = {}
        for key in REQUIRED_KEYS:
            if key not in features:
                raise ValueError(
                    f"HeuristicAdvisor needs feature {key!r}; got {sorted(features)!r}."
                )
            try:
                value = float(features[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Feature {key!r} is not numeric: {features[key]!r}") from exc
            if value != value:  # NaN check without numpy
                raise ValueError(f"Feature {key!r} is NaN (indicator window not warmed up).")
            out[key] = value
        return out

    @staticmethod
    def _snapshot(features: Mapping[str, Any]) -> dict[str, Any]:
        """Copy the feature mapping with numeric values as plain floats."""
        snap: dict[str, Any] = {}
        for key, value in features.items():
            try:
                snap[str(key)] = float(value)
            except (TypeError, ValueError):
                snap[str(key)] = value
        return snap
