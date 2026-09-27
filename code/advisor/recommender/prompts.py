"""Prompt construction for the LLM stock advisor.

Exposes the cautious-advisor :data:`SYSTEM_PROMPT`, the feature glossary, the
strict JSON response schema, and :func:`build_advisor_prompt`, which turns a
ticker plus an engineered-feature dictionary into a chat ``messages`` list
(role/content dicts) understood by both Ollama's native ``/api/chat`` endpoint
and OpenAI-style ``/chat/completions`` endpoints.

The prompt is written for *small local models* (Llama 3.1 8B, Qwen3 8B per
ADR-0001): every feature is explained in plain words with its typical range,
the model is walked through a fixed decision policy (trend -> momentum -> RSI),
confidence is defined as a probability of being right over the next ~5 trading
days (elicited verbally, after Tian et al. 2023), and the counterfactual must
name exactly one indicator and one threshold so the DiCE-style search
(Mothilal et al. 2020) has something concrete to verify.

Design notes:
    * The system prompt frames the model as a *cautious* advisor and forbids
      any prose outside the JSON object - this keeps parsing robust and aligns
      with the research-only, no-execution scope of the project.
    * Confidence is requested as a single float in ``[0, 1]``. It is treated as
      a *verbal* (uncalibrated) confidence; downstream temperature scaling
      (Guo et al. 2017, ADR-0002) maps it to a calibrated probability.
    * The user message carries the "as of" date so the model never assumes the
      wall-clock "today" when a backtest asks about a historical day.
    * :data:`PROMPT_VERSION` is part of every LLM cache key
      (:mod:`advisor.recommender.cache`), so editing the prompt invalidates
      cached answers automatically.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import date
from typing import Any

__all__ = [
    "PROMPT_VERSION",
    "SYSTEM_PROMPT",
    "STRICT_JSON_REMINDER",
    "FEATURE_GLOSSARY",
    "ADVISOR_FEATURE_KEYS",
    "RESPONSE_JSON_SCHEMA",
    "build_advisor_prompt",
    "format_features",
    "select_features",
]

#: Bump whenever the wording of the prompt changes; cached answers are keyed on it.
PROMPT_VERSION: str = "2.0"

#: Feature keys the advisor is shown, in display order. ``close`` is the raw
#: price; the rest are the engineered columns from
#: :data:`advisor.features.indicators.FEATURE_COLUMNS`.
ADVISOR_FEATURE_KEYS: tuple[str, ...] = (
    "close",
    "sma_10",
    "sma_50",
    "mom_10",
    "ret_1d",
    "vol_20",
    "rsi_14",
)

#: Plain-English meaning and typical range of every feature, for the prompt and
#: for UI tooltips. Keep the wording jargon-light: it is read by an 8B model.
FEATURE_GLOSSARY: dict[str, str] = {
    "close": "latest closing price in US dollars.",
    "sma_10": (
        "average closing price over the last 10 trading days "
        "(the short-term trend line)."
    ),
    "sma_50": (
        "average closing price over the last 50 trading days "
        "(the medium-term trend line)."
    ),
    "mom_10": (
        "10-day momentum: price change over the last 10 trading days as a "
        "fraction (0.05 = +5%). Positive = rising, negative = falling. "
        "Usually between -0.15 and +0.15."
    ),
    "ret_1d": "price change since the previous close as a fraction (0.01 = +1%).",
    "vol_20": (
        "20-day volatility: standard deviation of daily returns. "
        "Around 0.01 is calm; 0.03 or more is turbulent."
    ),
    "rsi_14": (
        "14-day Relative Strength Index on a 0-100 scale. Above 70 = "
        "overbought (stretched, may pull back); below 30 = oversold "
        "(may bounce); near 50 = neutral."
    ),
}

SYSTEM_PROMPT: str = (
    "You are a cautious, evidence-driven stock advisor writing for people "
    "with no finance background. You look at ONE US stock at a time, using "
    "only the technical-analysis numbers you are given, and recommend exactly "
    "one action: BUY, HOLD or SELL. This is research only: you never place "
    "trades, never handle money, and never give personalised financial "
    "advice.\n"
    "\n"
    "FEATURE GLOSSARY (name: meaning and typical range)\n"
    + "\n".join(f"- {name}: {meaning}" for name, meaning in FEATURE_GLOSSARY.items())
    + "\n"
    "\n"
    "DECISION POLICY (apply the steps in this order)\n"
    "1. Trend: is close above both sma_10 and sma_50 (uptrend), below both "
    "(downtrend), or in between (no clear trend)?\n"
    "2. Momentum: does mom_10 confirm the trend (positive in an uptrend, "
    "negative in a downtrend)?\n"
    "3. RSI check: rsi_14 above 70 argues AGAINST buying (already stretched); "
    "rsi_14 below 30 argues AGAINST selling (already washed out).\n"
    "4. BUY only when trend and momentum agree upward and rsi_14 is not above "
    "70. SELL only when trend and momentum agree downward and rsi_14 is not "
    "below 30. Otherwise HOLD. When signals are weak, mixed or contradictory, "
    "prefer HOLD.\n"
    "5. High vol_20 (0.03 or more) means the outcome is less predictable: "
    "lower your confidence.\n"
    "\n"
    "CONFIDENCE\n"
    '"confidence" is a decimal between 0 and 1: the probability that your '
    "action turns out to be right over the next 5 trading days. Be honest and "
    "modest: 0.5 means a coin flip, 0.6-0.7 is a reasonable call, above 0.8 "
    "needs strongly aligned signals. Never write it as a percentage or a "
    "string.\n"
    "\n"
    "OUTPUT FORMAT\n"
    "Reply with ONE strict JSON object and nothing else: no markdown, no code "
    "fences, no commentary before or after it. It must have EXACTLY these "
    "four keys:\n"
    '  "action": one of the strings "BUY", "HOLD" or "SELL".\n'
    '  "confidence": a decimal between 0 and 1 (for example 0.62).\n'
    '  "reason": one or two plain-English sentences a non-technical person '
    "can understand, quoting the numbers that drove the decision. No jargon.\n"
    '  "counterfactual": ONE sentence that starts with "I would change my '
    'mind if" and names exactly ONE indicator (say "10-day momentum", '
    '"14-day RSI", "50-day average price", "20-day volatility") together with '
    "a specific threshold value, for example: \"I would change my mind if the "
    '14-day RSI rose above 70."\n'
    "Do not invent data you were not given. Output ONLY the JSON object."
)

#: Follow-up sent once when the first answer could not be parsed/validated.
STRICT_JSON_REMINDER: str = (
    "Your previous reply was not a valid answer. Reply again with ONLY a JSON "
    'object with exactly the keys "action", "confidence", "reason" and '
    '"counterfactual". "action" must be exactly "BUY", "HOLD" or "SELL"; '
    '"confidence" must be a bare decimal between 0 and 1 (for example 0.62), '
    "not a percentage and not a string. No markdown, no code fences, no text "
    "outside the JSON object."
)

#: JSON schema of the four-key answer, passed as Ollama's ``format`` so the
#: server constrains decoding to this shape (structured outputs).
RESPONSE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["BUY", "HOLD", "SELL"]},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
        "counterfactual": {"type": "string"},
    },
    "required": ["action", "confidence", "reason", "counterfactual"],
}


def _is_missing(value: Any) -> bool:
    """Return ``True`` for ``None``/NaN-like values."""
    if value is None:
        return True
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def select_features(
    row: Mapping[str, Any],
    *,
    keys: tuple[str, ...] = ADVISOR_FEATURE_KEYS,
    strict: bool = True,
) -> dict[str, float]:
    """Pick the advisor's feature subset out of a feature row.

    Works on any mapping-like row, including a ``pandas.Series`` produced by
    :func:`advisor.features.indicators.compute_features`. Values are cast to
    plain ``float`` so the result is JSON-serialisable and hashable for the
    LLM cache.

    Args:
        row: Mapping of column name to value (a DataFrame row / Series works).
        keys: Feature names to extract, in order. Defaults to
            :data:`ADVISOR_FEATURE_KEYS`.
        strict: When ``True`` (default) every key must be present and
            non-NaN; otherwise missing/NaN keys are silently dropped.

    Returns:
        Dict of feature name to float, in ``keys`` order.

    Raises:
        ValueError: If ``strict`` and a key is missing or NaN (typically the
            indicator windows have not warmed up yet for that date).
    """
    out: dict[str, float] = {}
    for key in keys:
        value = row[key] if key in row else None
        if _is_missing(value):
            if strict:
                raise ValueError(
                    f"Feature {key!r} is missing or NaN - not enough price history "
                    "before this date to compute the indicators."
                )
            continue
        out[key] = float(value)
    return out


def _format_value(name: str, value: Any) -> str:
    """Render one feature value for the prompt (rounded, with a % gloss)."""
    if _is_missing(value):
        return "not available"
    if not isinstance(value, (int, float)):
        return str(value)
    v = float(value)
    if name in ("mom_10", "ret_1d"):
        return f"{v:+.4f} ({v * 100:+.2f}%)"
    if name in ("vol_20",):
        return f"{v:.4f}"
    if name in ("rsi_14",):
        return f"{v:.1f}"
    if name in ("close", "sma_10", "sma_50", "open", "high", "low", "adj_close"):
        return f"{v:.2f}"
    return f"{v:.6g}"


def format_features(features: Mapping[str, Any]) -> str:
    """Render a feature mapping as a bullet list for the user message.

    Known features (:data:`ADVISOR_FEATURE_KEYS`) come first in their
    canonical order; any extra keys follow in sorted order so the text is
    deterministic for identical inputs.

    Args:
        features: Mapping of feature name to value.

    Returns:
        Multi-line string, one ``- name: value`` bullet per feature.
    """
    ordered = [k for k in ADVISOR_FEATURE_KEYS if k in features]
    ordered += sorted(k for k in features if k not in ADVISOR_FEATURE_KEYS)
    return "\n".join(f"- {k}: {_format_value(k, features[k])}" for k in ordered)


def build_advisor_prompt(
    ticker: str,
    features: Mapping[str, Any],
    *,
    as_of: date | None = None,
) -> list[dict[str, str]]:
    """Build the chat ``messages`` list for one advisory call.

    Args:
        ticker: US equity symbol to advise on (e.g. ``"NVDA"``). Embedded
            verbatim, upper-cased, in the user message.
        features: Mapping of engineered feature names to values (``close``,
            ``sma_10``, ``sma_50``, ``mom_10``, ``ret_1d``, ``vol_20``,
            ``rsi_14`` from :mod:`advisor.features.indicators`). Rendered as a
            readable bullet list via :func:`format_features`.
        as_of: The trading date the features describe. Written into the user
            message as "today" so the model never assumes the wall-clock date.
            Defaults to :func:`datetime.date.today` when omitted.

    Returns:
        A two-element list ``[{"role": "system", ...}, {"role": "user", ...}]``
        accepted by Ollama's ``/api/chat`` and by OpenAI-style endpoints.
    """
    day = as_of or date.today()
    user_content = (
        f"Date: {day.isoformat()} ({day.strftime('%A')}). Treat this as today; "
        "you know nothing that happened after this date.\n"
        f"Stock: {ticker.upper()}\n"
        "Latest technical-analysis features (as of the close on this date):\n"
        f"{format_features(features)}\n\n"
        "Using ONLY these numbers and the decision policy, reply with the "
        "strict JSON object."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
