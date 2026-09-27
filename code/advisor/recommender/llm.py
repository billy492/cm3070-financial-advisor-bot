"""Local-LLM advisor client for Ollama.

:class:`OllamaAdvisor` is the runtime bridge between the engineered features
and a :class:`~advisor.recommender.schema.Recommendation`. By default it POSTs
to Ollama's *native* ``/api/chat`` endpoint with ``stream=false``, a JSON
``format`` constraint (structured output), ``think=false`` (disables the
hidden reasoning of "thinking" models such as Qwen3; harmless for Llama), and
deterministic decoding options (``temperature=0``, fixed ``seed``). An
OpenAI-compatible ``/v1/chat/completions`` path is kept behind the
``openai_compat`` flag for servers that only expose that API.

Robustness:
    * transient failures (connection errors, timeouts, HTTP 408/429/5xx) are
      retried up to ``max_retries`` times with a linear back-off;
    * the assistant text is parsed leniently (code fences, ``<think>`` blocks
      and surrounding prose are tolerated) and then validated strictly
      (action in {BUY, HOLD, SELL}; confidence a number in ``[0, 1]``);
    * a malformed or invalid answer is re-asked exactly once with a stricter
      reminder; a second failure raises :class:`ValueError`.

No network call happens at import time; every request is made inside a
method. Confidence calibration (Guo et al. 2017, ADR-0002) is applied through
an optional ``calibrator`` object exposing ``transform([p]) -> [p']``.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from collections.abc import Mapping
from datetime import date
from typing import Any

import requests

from advisor.config import OLLAMA_BASE_URL, OLLAMA_MODEL, RANDOM_SEED
from advisor.recommender.prompts import (
    PROMPT_VERSION,
    RESPONSE_JSON_SCHEMA,
    STRICT_JSON_REMINDER,
    build_advisor_prompt,
)
from advisor.recommender.schema import Action, Recommendation

__all__ = ["OllamaAdvisor", "ollama_host", "parse_advisor_json"]

_log = logging.getLogger(__name__)

# HTTP statuses treated as transient (worth a retry).
_TRANSIENT_STATUS: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504})
# ``<think>...</think>`` blocks some reasoning models emit inside the content.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
# Leading/trailing markdown code fences (``` or ```json).
_FENCE_OPEN = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*\n?")
_FENCE_CLOSE = re.compile(r"\n?\s*```\s*$")


def ollama_host(base_url: str) -> str:
    """Return the bare Ollama host URL, stripping a trailing ``/v1``.

    Args:
        base_url: Either the native host (``http://localhost:11434``) or the
            OpenAI-compatible base (``http://localhost:11434/v1``), with or
            without a trailing slash.

    Returns:
        ``scheme://host[:port]`` with no path and no trailing slash.
    """
    host = base_url.strip().rstrip("/")
    if host.lower().endswith("/v1"):
        host = host[: -len("/v1")]
    return host.rstrip("/")


def parse_advisor_json(content: str) -> dict[str, Any]:
    """Recover the JSON object from an assistant message.

    Tolerates ``<think>`` blocks, markdown code fences and prose before/after
    the object: after stripping fences the whole text is tried first, then
    every ``{`` position is tried with a raw JSON decode so the *first*
    well-formed object wins even when the surrounding prose contains braces.

    Args:
        content: Raw assistant message text.

    Returns:
        The decoded JSON object.

    Raises:
        ValueError: If the content is empty or contains no JSON object.
    """
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Advisor model returned empty content.")

    text = _THINK_BLOCK.sub("", content).strip()
    text = _FENCE_CLOSE.sub("", _FENCE_OPEN.sub("", text)).strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj

    raise ValueError(f"Advisor model did not return a JSON object: {text[:200]!r}")


class OllamaAdvisor:
    """Advisor backed by a local Ollama model.

    Args:
        model: Ollama model tag to query (default :data:`OLLAMA_MODEL`).
        base_url: Ollama base URL; a trailing ``/v1`` (OpenAI-compatible base)
            is stripped automatically (default :data:`OLLAMA_BASE_URL`).
        timeout: Seconds to wait for one HTTP response.
        temperature: Sampling temperature; ``0.0`` for deterministic output.
        max_retries: Extra attempts after a *transient* HTTP failure.
        calibrator: Optional object with ``transform([p]) -> [p']`` (e.g. a
            fitted :class:`advisor.calibration.temperature.TemperatureScaler`).
            When set, ``Recommendation.confidence`` is the calibrated value;
            ``raw_confidence`` always keeps the model's verbal confidence.
        openai_compat: Use ``/v1/chat/completions`` instead of the native
            ``/api/chat`` endpoint. Thinking cannot be disabled there, so it
            is a fallback only.
        json_schema: Send :data:`RESPONSE_JSON_SCHEMA` as Ollama's ``format``
            (structured output). When ``False`` the plain ``"json"`` mode is
            used instead. Ignored on the OpenAI-compatible path.
        num_predict: Maximum tokens the model may generate per answer.
        seed: Decoding seed passed to Ollama for reproducibility.
        retry_backoff: Base seconds slept between transient retries
            (multiplied by the attempt number).

    Attributes:
        prompt_version: The :data:`PROMPT_VERSION` this client speaks; part of
            the LLM cache key.
    """

    prompt_version: str = PROMPT_VERSION

    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        *,
        timeout: float = 120,
        temperature: float = 0.0,
        max_retries: int = 2,
        calibrator: Any | None = None,
        openai_compat: bool = False,
        json_schema: bool = True,
        num_predict: int = 300,
        seed: int = RANDOM_SEED,
        retry_backoff: float = 1.0,
    ) -> None:
        """Configure the client; no request is made until a method is called."""
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.max_retries = max(0, int(max_retries))
        self.calibrator = calibrator
        self.openai_compat = openai_compat
        self.json_schema = json_schema
        self.num_predict = num_predict
        self.seed = seed
        self.retry_backoff = retry_backoff

    # ------------------------------------------------------------------ URLs
    @property
    def host(self) -> str:
        """Bare Ollama host (``scheme://host:port``) derived from ``base_url``."""
        return ollama_host(self.base_url)

    @property
    def model_tag(self) -> str:
        """Identifier of the underlying model, used in LLM cache keys."""
        return self.model

    @property
    def chat_url(self) -> str:
        """Full URL of the chat endpoint in use (native or OpenAI-compatible)."""
        if self.openai_compat:
            return f"{self.host}/v1/chat/completions"
        return f"{self.host}/api/chat"

    @property
    def tags_url(self) -> str:
        """URL of Ollama's model-listing endpoint."""
        return f"{self.host}/api/tags"

    # --------------------------------------------------------------- health
    def available_models(self) -> list[str]:
        """List the model tags installed on the Ollama server.

        Returns:
            Model names as reported by ``GET /api/tags`` (e.g.
            ``["llama3.1:8b", "qwen3:8b"]``).

        Raises:
            ValueError: If the server cannot be reached or answers with an
                unexpected shape.
        """
        try:
            resp = requests.get(self.tags_url, timeout=min(self.timeout, 10))
            resp.raise_for_status()
            body = resp.json()
            return [str(m["name"]) for m in body.get("models", [])]
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"Cannot list Ollama models at {self.tags_url}: {exc}") from exc

    def health(self, *, require_model: bool = True) -> bool:
        """Check that the Ollama server is reachable (and the model installed).

        Args:
            require_model: Also require ``self.model`` to appear in the
                server's installed models (``name`` or ``name:latest``).

        Returns:
            ``True`` when healthy, ``False`` otherwise. Never raises.
        """
        try:
            models = self.available_models()
        except ValueError:
            return False
        if not require_model:
            return True
        wanted = {self.model, f"{self.model}:latest"}
        return any(m in wanted for m in models)

    # ----------------------------------------------------------- inference
    def recommend(
        self,
        ticker: str,
        features: Mapping[str, Any],
        *,
        as_of: date,
    ) -> Recommendation:
        """Query the model and return a parsed :class:`Recommendation`.

        Builds the advisory prompt (stamped with ``as_of``), POSTs it to the
        chat endpoint, parses and validates the strict JSON answer, re-asks
        once with a stricter reminder if the answer is unusable, and maps the
        result onto a :class:`Recommendation`.

        Args:
            ticker: US equity symbol to advise on (e.g. ``"AAPL"``).
            features: Engineered-feature mapping shown to the model. The exact
                dict sent (values cast to ``float``) is stored on
                ``Recommendation.features``.
            as_of: Trading date the recommendation is generated for. Written
                into the prompt and stamped onto the result; the caller must
                only pass features observable on or before this date.

        Returns:
            A :class:`Recommendation`. ``raw_confidence`` is the model's verbal
            confidence; ``confidence`` is the calibrated value when a
            ``calibrator`` is configured, else equal to ``raw_confidence``.

        Raises:
            ValueError: If the HTTP call fails after retries, or the model
                returns an unusable answer twice.
        """
        features_sent = self._clean_features(features)
        messages = build_advisor_prompt(ticker, features_sent, as_of=as_of)

        content = self._post(self._build_payload(messages))
        try:
            return self._to_recommendation(
                parse_advisor_json(content),
                ticker=ticker,
                as_of=as_of,
                features=features_sent,
            )
        except ValueError as first_err:
            _log.warning("Re-asking %s after unusable answer: %s", self.model, first_err)
            retry_messages = [
                *messages,
                {"role": "assistant", "content": content},
                {"role": "user", "content": STRICT_JSON_REMINDER},
            ]
            content2 = self._post(self._build_payload(retry_messages))
            try:
                return self._to_recommendation(
                    parse_advisor_json(content2),
                    ticker=ticker,
                    as_of=as_of,
                    features=features_sent,
                )
            except ValueError as second_err:
                raise ValueError(
                    f"Model {self.model!r} returned an unusable answer twice for "
                    f"{ticker.upper()} as of {as_of.isoformat()}: first: {first_err}; "
                    f"second: {second_err}"
                ) from second_err

    def predict(self, features: Mapping[str, Any]) -> str:
        """Return only the action string for a feature vector.

        Used as the black-box decision function probed by the counterfactual
        search. It reuses exactly the same prompt path as :meth:`recommend`
        (placeholder ticker ``"X"`` and today's date).

        Args:
            features: Feature mapping (same keys as for :meth:`recommend`).

        Returns:
            ``"BUY"``, ``"HOLD"`` or ``"SELL"``.
        """
        return self.recommend("X", features, as_of=date.today()).action.value

    # ------------------------------------------------------------ internals
    @staticmethod
    def _clean_features(features: Mapping[str, Any]) -> dict[str, Any]:
        """Copy the feature mapping with numeric values cast to ``float``.

        Args:
            features: Raw feature mapping (may contain numpy scalars).

        Returns:
            A plain ``dict`` safe for JSON serialisation and cache hashing.

        Raises:
            ValueError: If a numeric feature is NaN/inf.
        """
        clean: dict[str, Any] = {}
        for key, value in features.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    clean[str(key)] = str(value)
                    continue
            else:
                number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"Feature {key!r} is not finite ({value!r}).")
            clean[str(key)] = number
        return clean

    def _build_payload(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Assemble the request body for the configured endpoint.

        Args:
            messages: Chat messages from :func:`build_advisor_prompt`.

        Returns:
            JSON-serialisable request body.
        """
        if self.openai_compat:
            return {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "seed": self.seed,
                "max_tokens": self.num_predict,
                "response_format": {"type": "json_object"},
            }
        return {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": RESPONSE_JSON_SCHEMA if self.json_schema else "json",
            "think": False,
            "options": {
                "temperature": self.temperature,
                "seed": self.seed,
                "num_predict": self.num_predict,
            },
        }

    def _post(self, payload: dict[str, Any]) -> str:
        """POST a payload, retrying transient failures, and return the text.

        Args:
            payload: Request body from :meth:`_build_payload`.

        Returns:
            The assistant message content.

        Raises:
            ValueError: On a non-transient HTTP error, after exhausting
                retries, or if the response body has an unexpected shape.
        """
        last_error: str = ""
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(self.chat_url, json=payload, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            except requests.RequestException as exc:  # non-retryable client error
                raise ValueError(f"Ollama request to {self.chat_url} failed: {exc}") from exc
            else:
                if resp.status_code in _TRANSIENT_STATUS:
                    last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                elif resp.status_code >= 400:
                    raise ValueError(
                        f"Ollama rejected the request (HTTP {resp.status_code}) at "
                        f"{self.chat_url}: {resp.text[:300]}. Is model {self.model!r} "
                        f"installed? Try `ollama pull {self.model}`."
                    )
                else:
                    return self._extract_content(resp)
            if attempt < self.max_retries:
                delay = self.retry_backoff * (attempt + 1)
                _log.warning(
                    "Transient Ollama failure (%s); retry %d/%d in %.1fs",
                    last_error,
                    attempt + 1,
                    self.max_retries,
                    delay,
                )
                if delay > 0:
                    time.sleep(delay)
        raise ValueError(
            f"Ollama request to {self.chat_url} failed after "
            f"{self.max_retries + 1} attempt(s): {last_error}. Is Ollama running "
            f"at {self.host}?"
        )

    def _extract_content(self, resp: Any) -> str:
        """Pull the assistant text out of a native or OpenAI-style response.

        Args:
            resp: A ``requests.Response``-like object with ``.json()``.

        Returns:
            The assistant message content string.

        Raises:
            ValueError: If the body is not JSON or lacks the expected keys, or
                if the model produced only hidden "thinking" and no answer.
        """
        try:
            body = resp.json()
        except ValueError as exc:
            raise ValueError(f"Non-JSON response body from {self.chat_url}: {exc}") from exc
        try:
            if isinstance(body, dict) and "message" in body:  # native /api/chat
                message = body["message"]
                content = message.get("content", "")
                if not str(content).strip() and message.get("thinking"):
                    raise ValueError(
                        "Model returned only hidden thinking and no answer; "
                        "increase num_predict or use a non-thinking model."
                    )
                return str(content)
            return str(body["choices"][0]["message"]["content"])  # OpenAI style
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise ValueError(
                f"Unexpected chat response shape from {self.chat_url}: {exc}"
            ) from exc

    @staticmethod
    def _parse_content(content: str) -> dict[str, Any]:
        """Parse assistant text into a JSON object (see :func:`parse_advisor_json`).

        Args:
            content: Raw assistant message string.

        Returns:
            The decoded JSON object.

        Raises:
            ValueError: If no JSON object can be recovered.
        """
        return parse_advisor_json(content)

    def _to_recommendation(
        self,
        parsed: Mapping[str, Any],
        *,
        ticker: str,
        as_of: date,
        features: dict[str, Any] | None,
    ) -> Recommendation:
        """Validate a parsed answer and map it onto a :class:`Recommendation`.

        Args:
            parsed: Decoded JSON object with ``action``, ``confidence``,
                ``reason`` and ``counterfactual`` keys.
            ticker: Symbol to stamp onto the recommendation (upper-cased).
            as_of: Trading date to stamp onto the recommendation.
            features: The feature dict that was sent to the model.

        Returns:
            A populated :class:`Recommendation`.

        Raises:
            ValueError: If the action is missing/invalid or the confidence is
                not a number in ``[0, 1]``.
        """
        raw_action = parsed.get("action")
        if not isinstance(raw_action, str):
            raise ValueError(f"Missing or non-string 'action': {raw_action!r}")
        try:
            action = Action(raw_action.strip().upper())
        except ValueError as exc:
            raise ValueError(f"Invalid action {raw_action!r}; expected BUY/HOLD/SELL.") from exc

        raw_conf = parsed.get("confidence")
        if isinstance(raw_conf, bool):
            raise ValueError(f"Confidence must be a number, got {raw_conf!r}.")
        try:
            raw_confidence = float(raw_conf)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Confidence must be a number, got {raw_conf!r}.") from exc
        if not math.isfinite(raw_confidence) or not 0.0 <= raw_confidence <= 1.0:
            raise ValueError(f"Confidence {raw_confidence} out of range [0, 1].")

        reason = str(parsed.get("reason") or "").strip()
        counterfactual = str(parsed.get("counterfactual") or "").strip()

        confidence = raw_confidence
        if self.calibrator is not None:
            confidence = float(self.calibrator.transform([raw_confidence])[0])
            confidence = min(1.0, max(0.0, confidence))

        return Recommendation(
            ticker=ticker.upper(),
            action=action,
            confidence=confidence,
            raw_confidence=raw_confidence,
            reason=reason,
            counterfactual=counterfactual,
            as_of=as_of,
            features=dict(features) if features is not None else None,
        )
