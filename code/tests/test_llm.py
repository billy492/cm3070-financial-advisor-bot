"""Tests for ``advisor.recommender.llm`` -- offline, HTTP fully mocked.

Every test monkeypatches ``requests.post`` / ``requests.get`` so no network
access ever happens. Coverage:

* JSON parsing: clean, code-fenced, prose-wrapped and ``<think>``-polluted
  answers all yield a :class:`Recommendation`;
* validation: invalid actions / confidences are rejected;
* the re-ask protocol: one stricter retry on a malformed answer, then
  ``ValueError``;
* transient HTTP failures are retried, hard failures raise immediately;
* calibration is applied to ``confidence`` and never to ``raw_confidence``;
* the native payload disables thinking and constrains the output to JSON.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import pytest
import requests

from advisor.recommender.llm import OllamaAdvisor, ollama_host, parse_advisor_json
from advisor.recommender.prompts import (
    PROMPT_VERSION,
    RESPONSE_JSON_SCHEMA,
    STRICT_JSON_REMINDER,
)
from advisor.recommender.schema import Action, Recommendation

GOOD_ANSWER: dict[str, Any] = {
    "action": "BUY",
    "confidence": 0.72,
    "reason": "Price is above both averages and momentum is positive.",
    "counterfactual": "I would change my mind if the 14-day RSI rose above 70.",
}
AS_OF = date(2025, 3, 14)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class _FakeResponse:
    """Minimal stand-in for ``requests.Response``."""

    def __init__(self, status_code: int = 200, body: Any = None, text: str = "") -> None:
        self.status_code = status_code
        self._body = body
        self.text = text or (json.dumps(body) if body is not None else "")

    def json(self) -> Any:
        if self._body is None:
            raise ValueError("no json")
        return self._body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def _native(content: str) -> _FakeResponse:
    """Wrap assistant text in Ollama's native ``/api/chat`` response shape."""
    return _FakeResponse(200, {"model": "m", "message": {"role": "assistant", "content": content}})


def _openai(content: str) -> _FakeResponse:
    """Wrap assistant text in an OpenAI-style ``/chat/completions`` shape."""
    return _FakeResponse(200, {"choices": [{"message": {"role": "assistant", "content": content}}]})


class _Recorder:
    """Callable replacing ``requests.post`` that replays scripted responses."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, url: str, *, json: Any = None, timeout: Any = None, **kw: Any) -> Any:
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def advisor() -> OllamaAdvisor:
    """Client with zero back-off so retry tests do not sleep."""
    return OllamaAdvisor(
        model="test-model:8b",
        base_url="http://localhost:11434/v1",
        max_retries=1,
        retry_backoff=0.0,
    )


def _install(monkeypatch: pytest.MonkeyPatch, responses: list[Any]) -> _Recorder:
    recorder = _Recorder(responses)
    monkeypatch.setattr(requests, "post", recorder)
    return recorder


# --------------------------------------------------------------------------- #
# ollama_host / URLs
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("http://localhost:11434/v1", "http://localhost:11434"),
        ("http://localhost:11434/v1/", "http://localhost:11434"),
        ("http://localhost:11434", "http://localhost:11434"),
        ("http://box:11434/", "http://box:11434"),
    ],
)
def test_ollama_host_strips_v1(base_url: str, expected: str) -> None:
    """A trailing ``/v1`` (OpenAI-compatible base) is removed."""
    assert ollama_host(base_url) == expected


def test_default_endpoint_is_native_chat(advisor: OllamaAdvisor) -> None:
    """The native ``/api/chat`` endpoint is used by default."""
    assert advisor.chat_url == "http://localhost:11434/api/chat"
    assert advisor.tags_url == "http://localhost:11434/api/tags"
    assert advisor.model_tag == "test-model:8b"
    assert advisor.prompt_version == PROMPT_VERSION


def test_openai_compat_endpoint_flag() -> None:
    """``openai_compat=True`` switches to ``/v1/chat/completions``."""
    client = OllamaAdvisor(model="m", base_url="http://h:1/v1", openai_compat=True)
    assert client.chat_url == "http://h:1/v1/chat/completions"


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def test_parse_clean_json() -> None:
    """A bare JSON object parses as-is."""
    assert parse_advisor_json(json.dumps(GOOD_ANSWER)) == GOOD_ANSWER


def test_parse_fenced_json() -> None:
    """Markdown code fences around the object are tolerated."""
    text = "```json\n" + json.dumps(GOOD_ANSWER, indent=2) + "\n```"
    assert parse_advisor_json(text) == GOOD_ANSWER


def test_parse_prose_wrapped_json() -> None:
    """Prose before/after the object (even containing braces) is tolerated."""
    text = "Sure! Here {is} my answer:\n" + json.dumps(GOOD_ANSWER) + "\nHope it helps {}"
    assert parse_advisor_json(text) == GOOD_ANSWER


def test_parse_strips_think_block() -> None:
    """A ``<think>...</think>`` block is removed before parsing."""
    text = '<think>{"action": "SELL"} thinking...</think>\n' + json.dumps(GOOD_ANSWER)
    assert parse_advisor_json(text)["action"] == "BUY"


@pytest.mark.parametrize("text", ["", "   ", "no json here", "[1, 2, 3]", "{not json"])
def test_parse_rejects_non_objects(text: str) -> None:
    """Empty text, arrays and broken braces raise ``ValueError``."""
    with pytest.raises(ValueError):
        parse_advisor_json(text)


# --------------------------------------------------------------------------- #
# recommend(): happy paths
# --------------------------------------------------------------------------- #
def test_recommend_parses_good_json(
    advisor: OllamaAdvisor, features_dict: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clean answer maps onto a Recommendation with the sent features attached."""
    recorder = _install(monkeypatch, [_native(json.dumps(GOOD_ANSWER))])

    rec = advisor.recommend("aapl", features_dict, as_of=AS_OF)

    assert isinstance(rec, Recommendation)
    assert rec.ticker == "AAPL"
    assert rec.action is Action.BUY
    assert rec.confidence == pytest.approx(0.72)
    assert rec.raw_confidence == pytest.approx(0.72)
    assert rec.reason == GOOD_ANSWER["reason"]
    assert rec.counterfactual == GOOD_ANSWER["counterfactual"]
    assert rec.as_of == AS_OF
    assert rec.features == {k: float(v) for k, v in features_dict.items()}
    assert len(recorder.calls) == 1


@pytest.mark.parametrize(
    "wrapper",
    [
        lambda s: "```json\n" + s + "\n```",
        lambda s: "Here is my recommendation:\n" + s + "\nLet me know.",
    ],
    ids=["fenced", "prose"],
)
def test_recommend_tolerates_fenced_and_prose(
    advisor: OllamaAdvisor,
    features_dict: dict,
    monkeypatch: pytest.MonkeyPatch,
    wrapper: Any,
) -> None:
    """Fenced or prose-wrapped answers still yield a Recommendation on the first call."""
    recorder = _install(monkeypatch, [_native(wrapper(json.dumps(GOOD_ANSWER)))])
    rec = advisor.recommend("MSFT", features_dict, as_of=AS_OF)
    assert rec.action is Action.BUY
    assert len(recorder.calls) == 1


def test_recommend_parses_openai_compat_shape(
    features_dict: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The OpenAI-compatible fallback path parses ``choices[0].message.content``."""
    client = OllamaAdvisor(model="m", base_url="http://h:1", openai_compat=True)
    recorder = _install(monkeypatch, [_openai(json.dumps(GOOD_ANSWER))])

    rec = client.recommend("NVDA", features_dict, as_of=AS_OF)

    assert rec.action is Action.BUY
    payload = recorder.calls[0]["json"]
    assert recorder.calls[0]["url"] == "http://h:1/v1/chat/completions"
    assert payload["response_format"] == {"type": "json_object"}
    assert "think" not in payload


# --------------------------------------------------------------------------- #
# recommend(): validation + re-ask protocol
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "bad",
    [
        {**GOOD_ANSWER, "action": "MAYBE"},
        {**GOOD_ANSWER, "action": 3},
        {k: v for k, v in GOOD_ANSWER.items() if k != "action"},
    ],
    ids=["unknown-action", "non-string-action", "missing-action"],
)
def test_recommend_rejects_invalid_action(
    advisor: OllamaAdvisor, features_dict: dict, monkeypatch: pytest.MonkeyPatch, bad: dict
) -> None:
    """An invalid action is re-asked once, then raises ``ValueError``."""
    recorder = _install(monkeypatch, [_native(json.dumps(bad)), _native(json.dumps(bad))])
    with pytest.raises(ValueError, match="action"):
        advisor.recommend("AAPL", features_dict, as_of=AS_OF)
    assert len(recorder.calls) == 2


@pytest.mark.parametrize(
    "bad_conf",
    [1.5, -0.1, "high", None, True, "62%"],
    ids=["gt1", "lt0", "word", "none", "bool", "percent-string"],
)
def test_recommend_rejects_invalid_confidence(
    advisor: OllamaAdvisor, features_dict: dict, monkeypatch: pytest.MonkeyPatch, bad_conf: Any
) -> None:
    """A confidence outside [0, 1] or non-numeric is rejected after the re-ask."""
    bad = {**GOOD_ANSWER, "confidence": bad_conf}
    recorder = _install(monkeypatch, [_native(json.dumps(bad)), _native(json.dumps(bad))])
    with pytest.raises(ValueError, match="[Cc]onfidence"):
        advisor.recommend("AAPL", features_dict, as_of=AS_OF)
    assert len(recorder.calls) == 2


def test_recommend_accepts_numeric_string_confidence(
    advisor: OllamaAdvisor, features_dict: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A confidence given as the string "0.62" is coerced leniently."""
    _install(monkeypatch, [_native(json.dumps({**GOOD_ANSWER, "confidence": "0.62"}))])
    rec = advisor.recommend("AAPL", features_dict, as_of=AS_OF)
    assert rec.raw_confidence == pytest.approx(0.62)


def test_recommend_reasks_once_with_stricter_reminder(
    advisor: OllamaAdvisor, features_dict: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed first answer triggers exactly one re-ask carrying the reminder."""
    recorder = _install(
        monkeypatch,
        [_native("I think BUY is right, roughly 70%."), _native(json.dumps(GOOD_ANSWER))],
    )

    rec = advisor.recommend("AAPL", features_dict, as_of=AS_OF)

    assert rec.action is Action.BUY
    assert len(recorder.calls) == 2
    second_messages = recorder.calls[1]["json"]["messages"]
    assert second_messages[-1] == {"role": "user", "content": STRICT_JSON_REMINDER}
    assert second_messages[-2]["role"] == "assistant"
    assert second_messages[:2] == recorder.calls[0]["json"]["messages"]


def test_recommend_raises_after_two_malformed_answers(
    advisor: OllamaAdvisor, features_dict: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two unusable answers -> ValueError; the model is not asked a third time."""
    recorder = _install(monkeypatch, [_native("nope"), _native("still nope")])
    with pytest.raises(ValueError, match="unusable answer twice"):
        advisor.recommend("AAPL", features_dict, as_of=AS_OF)
    assert len(recorder.calls) == 2


def test_recommend_treats_thinking_only_answer_as_malformed(
    advisor: OllamaAdvisor, features_dict: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty content with a ``thinking`` field is unusable (re-ask, then error)."""
    body = {"message": {"role": "assistant", "content": "", "thinking": "hmm..."}}
    _install(monkeypatch, [_FakeResponse(200, body), _FakeResponse(200, body)])
    with pytest.raises(ValueError, match="thinking"):
        advisor.recommend("AAPL", features_dict, as_of=AS_OF)


def test_recommend_rejects_nan_feature_before_any_call(
    advisor: OllamaAdvisor, features_dict: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A NaN feature is a caller bug: raise without touching the network."""
    recorder = _install(monkeypatch, [])
    with pytest.raises(ValueError, match="not finite"):
        advisor.recommend("AAPL", {**features_dict, "rsi_14": float("nan")}, as_of=AS_OF)
    assert recorder.calls == []


# --------------------------------------------------------------------------- #
# Transport: retries and hard failures
# --------------------------------------------------------------------------- #
def test_transient_connection_error_is_retried(
    advisor: OllamaAdvisor, features_dict: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ConnectionError followed by success yields a Recommendation."""
    recorder = _install(
        monkeypatch, [requests.ConnectionError("refused"), _native(json.dumps(GOOD_ANSWER))]
    )
    rec = advisor.recommend("AAPL", features_dict, as_of=AS_OF)
    assert rec.action is Action.BUY
    assert len(recorder.calls) == 2


def test_transient_http_503_is_retried(
    advisor: OllamaAdvisor, features_dict: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An HTTP 503 is retried like a connection error."""
    recorder = _install(
        monkeypatch, [_FakeResponse(503, text="busy"), _native(json.dumps(GOOD_ANSWER))]
    )
    advisor.recommend("AAPL", features_dict, as_of=AS_OF)
    assert len(recorder.calls) == 2


def test_connection_errors_exhaust_retries(
    advisor: OllamaAdvisor, features_dict: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Persistent connection errors raise a clear ValueError naming the host."""
    recorder = _install(
        monkeypatch, [requests.ConnectionError("refused"), requests.Timeout("slow")]
    )
    with pytest.raises(ValueError, match="Is Ollama running"):
        advisor.recommend("AAPL", features_dict, as_of=AS_OF)
    assert len(recorder.calls) == 2  # max_retries=1 -> two attempts


def test_http_404_is_not_retried(
    advisor: OllamaAdvisor, features_dict: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 4xx (e.g. model not installed) fails immediately with a hint."""
    recorder = _install(monkeypatch, [_FakeResponse(404, text='{"error":"model not found"}')])
    with pytest.raises(ValueError, match="ollama pull"):
        advisor.recommend("AAPL", features_dict, as_of=AS_OF)
    assert len(recorder.calls) == 1


# --------------------------------------------------------------------------- #
# Calibration, predict(), payload
# --------------------------------------------------------------------------- #
class _HalfCalibrator:
    """Toy calibrator: halves the distance to 0.5 (softens confidence)."""

    def transform(self, confidences: list[float]) -> list[float]:
        return [0.5 + (c - 0.5) / 2 for c in confidences]


def test_calibrator_applies_to_confidence_only(
    features_dict: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``confidence`` is calibrated; ``raw_confidence`` keeps the verbal value."""
    client = OllamaAdvisor(model="m", base_url="http://h:1", calibrator=_HalfCalibrator())
    _install(monkeypatch, [_native(json.dumps({**GOOD_ANSWER, "confidence": 0.9}))])

    rec = client.recommend("AAPL", features_dict, as_of=AS_OF)

    assert rec.raw_confidence == pytest.approx(0.9)
    assert rec.confidence == pytest.approx(0.7)


def test_predict_returns_action_string_via_same_prompt_path(
    advisor: OllamaAdvisor, features_dict: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``predict`` returns a bare action string using today's date and ticker X."""
    recorder = _install(monkeypatch, [_native(json.dumps({**GOOD_ANSWER, "action": "sell"}))])

    action = advisor.predict(features_dict)

    assert action == "SELL"
    assert isinstance(action, str)
    user_msg = recorder.calls[0]["json"]["messages"][1]["content"]
    assert "Stock: X" in user_msg
    assert date.today().isoformat() in user_msg


def test_native_payload_disables_thinking_and_forces_json(
    advisor: OllamaAdvisor, features_dict: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The native body carries think=false, a JSON format, stream=false and seeded options."""
    recorder = _install(monkeypatch, [_native(json.dumps(GOOD_ANSWER))])
    advisor.recommend("AAPL", features_dict, as_of=AS_OF)

    call = recorder.calls[0]
    payload = call["json"]
    assert call["url"] == "http://localhost:11434/api/chat"
    assert call["timeout"] == advisor.timeout
    assert payload["model"] == "test-model:8b"
    assert payload["think"] is False
    assert payload["stream"] is False
    assert payload["format"] in ("json", RESPONSE_JSON_SCHEMA)
    assert payload["options"]["temperature"] == 0.0
    assert payload["options"]["num_predict"] == 300
    assert isinstance(payload["options"]["seed"], int)
    assert [m["role"] for m in payload["messages"]] == ["system", "user"]
    assert AS_OF.isoformat() in payload["messages"][1]["content"]


def test_json_schema_flag_switches_to_plain_json_mode(
    features_dict: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``json_schema=False`` sends Ollama's plain ``"json"`` format."""
    client = OllamaAdvisor(model="m", base_url="http://h:1", json_schema=False)
    recorder = _install(monkeypatch, [_native(json.dumps(GOOD_ANSWER))])
    client.recommend("AAPL", features_dict, as_of=AS_OF)
    assert recorder.calls[0]["json"]["format"] == "json"


# --------------------------------------------------------------------------- #
# available_models() / health()
# --------------------------------------------------------------------------- #
def test_available_models_and_health(monkeypatch: pytest.MonkeyPatch) -> None:
    """Model tags come from GET /api/tags; health checks the configured model."""
    body = {"models": [{"name": "llama3.1:8b"}, {"name": "qwen3:8b"}]}
    calls: list[str] = []

    def fake_get(url: str, *, timeout: Any = None, **kw: Any) -> _FakeResponse:
        calls.append(url)
        return _FakeResponse(200, body)

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(requests, "post", lambda *a, **k: pytest.fail("post must not be called"))

    present = OllamaAdvisor(model="qwen3:8b", base_url="http://h:1/v1")
    missing = OllamaAdvisor(model="phi4:latest", base_url="http://h:1/v1")

    assert present.available_models() == ["llama3.1:8b", "qwen3:8b"]
    assert calls == ["http://h:1/api/tags"]
    assert present.health() is True
    assert missing.health() is False
    assert missing.health(require_model=False) is True


def test_health_false_when_server_down(monkeypatch: pytest.MonkeyPatch) -> None:
    """A connection error makes ``health`` return False rather than raise."""

    def fake_get(url: str, **kw: Any) -> Any:
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(requests, "get", fake_get)
    client = OllamaAdvisor(model="m", base_url="http://h:1")
    assert client.health() is False
    with pytest.raises(ValueError):
        client.available_models()
