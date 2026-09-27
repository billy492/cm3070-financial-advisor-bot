"""Tests for ``advisor.recommender.cache`` -- the JSONL memoising wrapper.

Covers hit/miss accounting, persistence across instances, key sensitivity
(model tag, ticker, date, features rounded to 6 dp, prompt version), tolerance
of malformed lines, re-application of a calibrator on hits, ``predict`` and
thread-safety under a small ``ThreadPoolExecutor``.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from advisor.recommender.cache import CachedAdvisor, make_cache_key, recommendation_from_dict
from advisor.recommender.heuristic import HeuristicAdvisor
from advisor.recommender.schema import Action, Recommendation

AS_OF = date(2025, 3, 14)
FEATS: dict[str, float] = {
    "close": 101.0,
    "sma_10": 100.5,
    "sma_50": 98.2,
    "mom_10": 0.034,
    "ret_1d": 0.012,
    "vol_20": 0.018,
    "rsi_14": 57.3,
}


class _FakeAdvisor:
    """Deterministic inner advisor that counts calls."""

    model_tag = "fake-model:1b"
    prompt_version = "p1"

    def __init__(self) -> None:
        self.calls: list[tuple[str, date]] = []
        self.calibrator: Any | None = None

    def recommend(self, ticker: str, features: Any, *, as_of: date) -> Recommendation:
        self.calls.append((ticker, as_of))
        raw = float(features["rsi_14"]) / 100.0
        conf = raw if self.calibrator is None else float(self.calibrator.transform([raw])[0])
        return Recommendation(
            ticker=ticker.upper(),
            action=Action.BUY if features["mom_10"] > 0 else Action.SELL,
            confidence=conf,
            raw_confidence=raw,
            reason=f"reason for {ticker}",
            counterfactual="I would change my mind if the 14-day RSI rose above 70.",
            as_of=as_of,
            features=dict(features),
        )


class _HalfCalibrator:
    def transform(self, confidences: list[float]) -> list[float]:
        return [0.5 + (c - 0.5) / 2 for c in confidences]


@pytest.fixture
def cache_path(tmp_path: Path) -> Path:
    """A fresh JSONL path inside a temporary directory."""
    return tmp_path / "llm_cache" / "fake.jsonl"


# --------------------------------------------------------------------------- #
# Keys
# --------------------------------------------------------------------------- #
def test_make_cache_key_is_deterministic_sha256() -> None:
    """Same inputs -> same 64-hex key; ticker case and key order are irrelevant."""
    k1 = make_cache_key("m", "AAPL", AS_OF, FEATS, "p1")
    k2 = make_cache_key("m", "aapl", AS_OF, dict(reversed(list(FEATS.items()))), "p1")
    assert k1 == k2
    assert len(k1) == 64 and int(k1, 16) >= 0


@pytest.mark.parametrize(
    "variant",
    [
        lambda: make_cache_key("other", "AAPL", AS_OF, FEATS, "p1"),
        lambda: make_cache_key("m", "MSFT", AS_OF, FEATS, "p1"),
        lambda: make_cache_key("m", "AAPL", date(2025, 3, 15), FEATS, "p1"),
        lambda: make_cache_key("m", "AAPL", AS_OF, {**FEATS, "rsi_14": 57.301}, "p1"),
        lambda: make_cache_key("m", "AAPL", AS_OF, FEATS, "p2"),
    ],
    ids=["model", "ticker", "as_of", "feature", "prompt_version"],
)
def test_make_cache_key_changes_with_each_component(variant: Any) -> None:
    """Every component of the key matters."""
    assert variant() != make_cache_key("m", "AAPL", AS_OF, FEATS, "p1")


def test_make_cache_key_rounds_features_to_6dp() -> None:
    """Differences beyond 6 dp (and -0.0 vs 0.0) do not change the key."""
    base = make_cache_key("m", "AAPL", AS_OF, {**FEATS, "mom_10": 0.0}, "p1")
    assert make_cache_key("m", "AAPL", AS_OF, {**FEATS, "mom_10": -0.0}, "p1") == base
    assert make_cache_key("m", "AAPL", AS_OF, {**FEATS, "mom_10": 1e-9}, "p1") == base
    assert make_cache_key("m", "AAPL", AS_OF, {**FEATS, "mom_10": 1e-5}, "p1") != base


# --------------------------------------------------------------------------- #
# Hit / miss / persistence
# --------------------------------------------------------------------------- #
def test_miss_then_hit(cache_path: Path) -> None:
    """The first call reaches the inner advisor; the identical second one does not."""
    inner = _FakeAdvisor()
    cached = CachedAdvisor(inner, cache_path)

    first = cached.recommend("AAPL", FEATS, as_of=AS_OF)
    second = cached.recommend("aapl", FEATS, as_of=AS_OF)

    assert first == second
    assert inner.calls == [("AAPL", AS_OF)]
    assert (cached.hits, cached.misses) == (1, 1)
    assert cached.model_tag == "fake-model:1b"
    assert cached.prompt_version == "p1"
    assert len(cached) == 1


def test_persists_across_instances(cache_path: Path) -> None:
    """A new wrapper over the same file serves the answer without the inner advisor."""
    CachedAdvisor(_FakeAdvisor(), cache_path).recommend("AAPL", FEATS, as_of=AS_OF)
    assert cache_path.exists()

    inner2 = _FakeAdvisor()
    cached2 = CachedAdvisor(inner2, cache_path)
    rec = cached2.recommend("AAPL", FEATS, as_of=AS_OF)

    assert inner2.calls == []
    assert (cached2.hits, cached2.misses) == (1, 0)
    assert rec.action is Action.BUY
    assert rec.features == FEATS
    assert rec.as_of == AS_OF


def test_jsonl_records_are_append_only_and_self_describing(cache_path: Path) -> None:
    """Each line is a JSON record with key, model tag, prompt version and the answer."""
    cached = CachedAdvisor(_FakeAdvisor(), cache_path)
    cached.recommend("AAPL", FEATS, as_of=AS_OF)
    cached.recommend("MSFT", FEATS, as_of=AS_OF)

    lines = [json.loads(line) for line in cache_path.read_text().splitlines() if line]
    assert len(lines) == 2
    assert {rec["ticker"] for rec in lines} == {"AAPL", "MSFT"}
    assert all(rec["model_tag"] == "fake-model:1b" for rec in lines)
    assert all(rec["prompt_version"] == "p1" for rec in lines)
    assert lines[0]["recommendation"]["action"] == "BUY"


def test_malformed_lines_are_skipped(cache_path: Path) -> None:
    """Corrupt lines do not break loading; valid ones are still served."""
    cached = CachedAdvisor(_FakeAdvisor(), cache_path)
    cached.recommend("AAPL", FEATS, as_of=AS_OF)
    with cache_path.open("a") as handle:
        handle.write("this is not json\n{\"key\": \"x\"}\n")

    inner = _FakeAdvisor()
    reloaded = CachedAdvisor(inner, cache_path)
    assert len(reloaded) == 1
    reloaded.recommend("AAPL", FEATS, as_of=AS_OF)
    assert inner.calls == []


def test_different_inputs_miss(cache_path: Path) -> None:
    """A different date, ticker or feature value reaches the inner advisor."""
    inner = _FakeAdvisor()
    cached = CachedAdvisor(inner, cache_path)
    cached.recommend("AAPL", FEATS, as_of=AS_OF)
    cached.recommend("AAPL", FEATS, as_of=date(2025, 3, 17))
    cached.recommend("MSFT", FEATS, as_of=AS_OF)
    cached.recommend("AAPL", {**FEATS, "rsi_14": 71.0}, as_of=AS_OF)
    assert len(inner.calls) == 4
    assert (cached.hits, cached.misses) == (0, 4)


def test_prompt_version_change_invalidates(cache_path: Path) -> None:
    """Bumping the inner advisor's prompt version yields fresh misses."""
    CachedAdvisor(_FakeAdvisor(), cache_path).recommend("AAPL", FEATS, as_of=AS_OF)
    inner = _FakeAdvisor()
    inner.prompt_version = "p2"
    cached = CachedAdvisor(inner, cache_path)
    cached.recommend("AAPL", FEATS, as_of=AS_OF)
    assert inner.calls == [("AAPL", AS_OF)]


# --------------------------------------------------------------------------- #
# Calibration on hits, predict, integration
# --------------------------------------------------------------------------- #
def test_calibrator_reapplied_on_hit(cache_path: Path) -> None:
    """Fitting a calibrator later changes ``confidence`` on hits, not ``raw_confidence``."""
    inner = _FakeAdvisor()
    cached = CachedAdvisor(inner, cache_path)
    raw_rec = cached.recommend("AAPL", FEATS, as_of=AS_OF)
    assert raw_rec.confidence == pytest.approx(0.573)

    cached.calibrator = _HalfCalibrator()
    assert inner.calibrator is not None
    hit = cached.recommend("AAPL", FEATS, as_of=AS_OF)

    assert inner.calls == [("AAPL", AS_OF)]  # no re-query
    assert hit.raw_confidence == pytest.approx(0.573)
    assert hit.confidence == pytest.approx(0.5 + 0.073 / 2)

    cached.calibrator = None
    assert cached.recommend("AAPL", FEATS, as_of=AS_OF).confidence == pytest.approx(0.573)


def test_predict_uses_cache(cache_path: Path) -> None:
    """``predict`` returns the action string and is memoised like ``recommend``."""
    inner = _FakeAdvisor()
    cached = CachedAdvisor(inner, cache_path)
    assert cached.predict(FEATS) == "BUY"
    assert cached.predict(FEATS) == "BUY"
    assert cached.predict({**FEATS, "mom_10": -0.02}) == "SELL"
    assert len(inner.calls) == 2
    assert inner.calls[0][0] == "X"


def test_wraps_heuristic_advisor_end_to_end(cache_path: Path) -> None:
    """The wrapper is advisor-agnostic: the rule baseline works and is served on reload."""
    cached = CachedAdvisor(HeuristicAdvisor(), cache_path)
    rec = cached.recommend("AAPL", FEATS, as_of=AS_OF)
    assert cached.model_tag == "heuristic-v1"
    assert rec.action is Action.BUY
    again = CachedAdvisor(HeuristicAdvisor(), cache_path)
    assert again.recommend("AAPL", FEATS, as_of=AS_OF) == rec
    assert again.hits == 1
    assert again.health() is True


def test_thread_safety_each_key_computed_once(cache_path: Path) -> None:
    """Concurrent distinct queries are each computed once and all persisted."""
    inner = _FakeAdvisor()
    cached = CachedAdvisor(inner, cache_path)
    tickers = [f"T{i:02d}" for i in range(24)]

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda t: cached.recommend(t, FEATS, as_of=AS_OF), tickers))
    with ThreadPoolExecutor(max_workers=4) as pool:
        again = list(pool.map(lambda t: cached.recommend(t, FEATS, as_of=AS_OF), tickers))

    assert results == again
    assert len(inner.calls) == 24
    assert len(cached) == 24
    assert (cached.hits, cached.misses) == (24, 24)
    assert len([line for line in cache_path.read_text().splitlines() if line]) == 24


def test_recommendation_from_dict_round_trip() -> None:
    """``to_dict`` -> ``recommendation_from_dict`` is the identity."""
    rec = _FakeAdvisor().recommend("AAPL", FEATS, as_of=AS_OF)
    assert recommendation_from_dict(rec.to_dict()) == rec
    with pytest.raises(ValueError):
        recommendation_from_dict({"ticker": "AAPL"})


def test_has_reports_cached_queries_without_calling_inner(cache_path: Path) -> None:
    """``has`` is a pure lookup: True only for an identical, already-answered query."""
    inner = _FakeAdvisor()
    cached = CachedAdvisor(inner, cache_path)
    assert cached.has("AAPL", FEATS, as_of=AS_OF) is False
    cached.recommend("AAPL", FEATS, as_of=AS_OF)
    assert cached.has("aapl", FEATS, as_of=AS_OF) is True
    assert cached.has("AAPL", FEATS, as_of=date(2025, 3, 17)) is False
    assert len(inner.calls) == 1
    assert (cached.hits, cached.misses) == (0, 1)
