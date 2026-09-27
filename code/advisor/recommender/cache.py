"""Persistent, append-only cache of advisor answers.

:class:`CachedAdvisor` wraps any advisor exposing the
``recommend(ticker, features, *, as_of)`` / ``predict(features)`` interface
(:class:`~advisor.recommender.llm.OllamaAdvisor`,
:class:`~advisor.recommender.heuristic.HeuristicAdvisor`, ...) and memoises
every answer in a JSON-Lines file. Because a local 8B model needs several
seconds per call, a walk-forward backtest over ~50 tickers and hundreds of
days takes hours; with the cache it becomes *resumable* (kill it, restart it,
only the missing days are queried) and *reproducible offline* (a second run
returns byte-identical answers without Ollama).

Cache key: SHA-256 of the JSON encoding of ``(model tag, ticker, as_of ISO
date, sorted features rounded to 6 dp, prompt version)``. Any change to the
model, the prompt wording (:data:`~advisor.recommender.prompts.PROMPT_VERSION`)
or the inputs therefore yields a fresh key. The calibrator is deliberately
*not* part of the key: calibration is post-hoc and is re-applied on every hit
from the stored ``raw_confidence``, so a fitted calibrator never forces the
LLM to be re-queried.

Thread-safety: a single :class:`threading.Lock` guards the in-memory dict and
the file append, which is sufficient for a ``ThreadPoolExecutor`` with a few
workers (the inner advisor is called outside the lock so calls can overlap).
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections.abc import Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from advisor.recommender.prompts import PROMPT_VERSION
from advisor.recommender.schema import Action, Recommendation

__all__ = ["CachedAdvisor", "make_cache_key", "recommendation_from_dict"]

_log = logging.getLogger(__name__)

#: Decimal places features are rounded to before hashing.
_FEATURE_DECIMALS: int = 6


def _normalise_features(features: Mapping[str, Any]) -> dict[str, Any]:
    """Round numeric features to 6 dp (and fold ``-0.0`` into ``0.0``).

    Args:
        features: Feature mapping (numbers, numpy scalars or strings).

    Returns:
        A plain dict with sorted keys, safe for deterministic JSON encoding.
    """
    out: dict[str, Any] = {}
    for key in sorted(features):
        value = features[key]
        if isinstance(value, bool):
            out[str(key)] = value
            continue
        try:
            number = round(float(value), _FEATURE_DECIMALS)
        except (TypeError, ValueError):
            out[str(key)] = str(value)
            continue
        out[str(key)] = 0.0 if number == 0 else number
    return out


def make_cache_key(
    model_tag: str,
    ticker: str,
    as_of: date,
    features: Mapping[str, Any],
    prompt_version: str = PROMPT_VERSION,
) -> str:
    """Compute the SHA-256 cache key for one advisor query.

    Args:
        model_tag: Identifier of the inner model (e.g. ``"llama3.1:8b"``).
        ticker: Ticker symbol (case-insensitive).
        as_of: Trading date of the query.
        features: Feature mapping; rounded to 6 dp and key-sorted.
        prompt_version: Prompt wording version.

    Returns:
        64-character hexadecimal digest.
    """
    material = json.dumps(
        [
            model_tag,
            ticker.upper(),
            as_of.isoformat(),
            _normalise_features(features),
            prompt_version,
        ],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=True,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def recommendation_from_dict(data: Mapping[str, Any]) -> Recommendation:
    """Rebuild a :class:`Recommendation` from :meth:`Recommendation.to_dict` output.

    Args:
        data: Dict with the keys produced by ``Recommendation.to_dict``.

    Returns:
        The reconstructed, immutable :class:`Recommendation`.

    Raises:
        ValueError: If a mandatory key is missing or malformed.
    """
    try:
        return Recommendation(
            ticker=str(data["ticker"]),
            action=Action(str(data["action"]).upper()),
            confidence=float(data["confidence"]),
            raw_confidence=float(data["raw_confidence"]),
            reason=str(data.get("reason", "")),
            counterfactual=str(data.get("counterfactual", "")),
            as_of=date.fromisoformat(str(data["as_of"])[:10]),
            features=dict(data["features"]) if data.get("features") is not None else None,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Malformed cached recommendation: {exc}") from exc


class CachedAdvisor:
    """Memoising wrapper around an advisor, persisted as JSON Lines.

    Args:
        inner: The advisor to wrap. Must expose ``recommend`` and may expose
            ``model_tag``/``model``, ``prompt_version`` and ``calibrator``.
        cache_path: JSONL file to load from and append to. Created (with
            parent directories) on first write.

    Attributes:
        hits: Number of ``recommend`` calls served from the cache.
        misses: Number of ``recommend`` calls forwarded to ``inner``.
    """

    def __init__(self, inner: Any, cache_path: str | Path) -> None:
        """Wrap ``inner`` and load any existing answers from ``cache_path``."""
        self.inner = inner
        self.cache_path = Path(cache_path)
        self.hits: int = 0
        self.misses: int = 0
        self._lock = threading.Lock()
        self._store: dict[str, dict[str, Any]] = {}
        self._load()

    # ------------------------------------------------------------ metadata
    @property
    def model_tag(self) -> str:
        """Model identifier of the wrapped advisor (used in the key)."""
        tag = getattr(self.inner, "model_tag", None) or getattr(self.inner, "model", None)
        return str(tag) if tag else type(self.inner).__name__

    @property
    def prompt_version(self) -> str:
        """Prompt version of the wrapped advisor (used in the key)."""
        return str(getattr(self.inner, "prompt_version", PROMPT_VERSION))

    @property
    def calibrator(self) -> Any | None:
        """Calibrator of the wrapped advisor, if any (re-applied on hits)."""
        return getattr(self.inner, "calibrator", None)

    @calibrator.setter
    def calibrator(self, value: Any | None) -> None:
        self.inner.calibrator = value

    def __len__(self) -> int:
        """Number of cached answers currently loaded."""
        return len(self._store)

    def health(self, *, require_model: bool = True) -> bool:
        """Delegate to the wrapped advisor's ``health`` (``True`` if absent).

        Args:
            require_model: Forwarded to the inner advisor.

        Returns:
            The inner advisor's health, or ``True`` if it has no such method.
        """
        probe = getattr(self.inner, "health", None)
        return bool(probe(require_model=require_model)) if callable(probe) else True

    # ------------------------------------------------------------ interface
    def has(self, ticker: str, features: Mapping[str, Any], *, as_of: date) -> bool:
        """Report whether an answer for this exact query is already cached.

        Lets callers skip a server health check (or a slow LLM) when the
        answer will be served from disk anyway.

        Args:
            ticker: Ticker symbol.
            features: Feature mapping (part of the cache key).
            as_of: Trading date (part of the cache key).

        Returns:
            ``True`` if :meth:`recommend` would be a cache hit.
        """
        key = make_cache_key(self.model_tag, ticker, as_of, features, self.prompt_version)
        with self._lock:
            return key in self._store

    def recommend(
        self,
        ticker: str,
        features: Mapping[str, Any],
        *,
        as_of: date,
    ) -> Recommendation:
        """Return the cached answer for this query, or ask ``inner`` and store it.

        Args:
            ticker: Ticker symbol.
            features: Feature mapping (part of the cache key).
            as_of: Trading date (part of the cache key).

        Returns:
            A :class:`Recommendation`. On a hit, ``confidence`` is recomputed
            from the stored ``raw_confidence`` with the inner advisor's
            current calibrator (or equals ``raw_confidence`` when none).
        """
        key = make_cache_key(self.model_tag, ticker, as_of, features, self.prompt_version)
        with self._lock:
            cached = self._store.get(key)
            if cached is not None:
                self.hits += 1
        if cached is not None:
            return self._apply_calibration(recommendation_from_dict(cached))

        rec = self.inner.recommend(ticker, features, as_of=as_of)
        with self._lock:
            self.misses += 1
            if key not in self._store:
                self._store[key] = rec.to_dict()
                self._append(key, rec)
        return rec

    def predict(self, features: Mapping[str, Any]) -> str:
        """Return only the action string (cached like :meth:`recommend`).

        Args:
            features: Feature mapping.

        Returns:
            ``"BUY"``, ``"HOLD"`` or ``"SELL"``.
        """
        return self.recommend("X", features, as_of=date.today()).action.value

    # ------------------------------------------------------------ internals
    def _apply_calibration(self, rec: Recommendation) -> Recommendation:
        """Recompute ``confidence`` from ``raw_confidence`` with the current calibrator."""
        calibrator = self.calibrator
        if calibrator is None:
            confidence = rec.raw_confidence
        else:
            confidence = float(calibrator.transform([rec.raw_confidence])[0])
            confidence = min(1.0, max(0.0, confidence))
        if confidence == rec.confidence:
            return rec
        return Recommendation(
            ticker=rec.ticker,
            action=rec.action,
            confidence=confidence,
            raw_confidence=rec.raw_confidence,
            reason=rec.reason,
            counterfactual=rec.counterfactual,
            as_of=rec.as_of,
            features=rec.features,
        )

    def _load(self) -> None:
        """Read every valid JSONL record into memory (malformed lines are skipped)."""
        if not self.cache_path.exists():
            return
        loaded = 0
        with self.cache_path.open("r", encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    self._store[str(record["key"])] = dict(record["recommendation"])
                    loaded += 1
                except (ValueError, KeyError, TypeError):
                    _log.warning("Skipping malformed cache line %d in %s", lineno, self.cache_path)
        _log.info("Loaded %d cached answers from %s", loaded, self.cache_path)

    def _append(self, key: str, rec: Recommendation) -> None:
        """Append one record to the JSONL file (caller holds the lock)."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "key": key,
            "model_tag": self.model_tag,
            "prompt_version": self.prompt_version,
            "ticker": rec.ticker,
            "as_of": rec.as_of.isoformat(),
            "cached_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "recommendation": rec.to_dict(),
        }
        with self.cache_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
