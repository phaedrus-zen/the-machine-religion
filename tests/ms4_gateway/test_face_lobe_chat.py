"""FaceLobeChat tests — direct chat-completion path.

Covers:
- Empty-content fallback: when the chosen model returns nothing, we
  retry once with the configured fallback model and surface
  ``fallback_used: true`` on the response so the UI can show the actual
  effective model.
- Stream-stall guard: a streaming call that never returns ``[DONE]``
  must not hang forever; the FaceLobeChat returns whatever it has when
  the stall timeout fires.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from machine_spirit_4.double_agent.worker import (
    _DEPTH_EVIDENCE_FREE_COMPARISON_TEMPLATE,
)
from machine_spirit_4.gateway.hermes_runner import _VOICE_BREVITY_DIRECTIVE
from machine_spirit_4.gateway.face_lobe_chat import (
    FaceLobeChat,
    FaceLobeChatError,
    _canonical_turn_level_deadline_matches,
    _comparison_truth_contract_for_verified_depth,
    _comparison_truth_failures,
    _enforce_current_dispatch_ack_postcondition,
    _enforce_latency_correction_postcondition,
    _enforce_latency_lobe_staging_postcondition,
    _END_TO_END_VOICE_LATENCY_RE,
    _FIRST_TOKEN_LATENCY_RE,
    _has_concrete_operational_example,
    _latency_constraint_change_direction,
    _latency_constraint_reference,
    _latency_dimension_asserted_current,
    _latency_dimension_asserted_historical,
    _latency_staging_rejected,
    _parallel_stage_sequence_contradiction,
    _markdown_section_heading,
    _markdown_section_items,
    _ordinal_expectation,
    _ordinal_item_label,
    _render_authoritative_comparison_truth_rescue,
    _retryable_face_http_status,
    _rewrite_stale_verified_depth_status,
    _substantive_followup_failures,
    _substantive_followup_intent,
    _voice_response_start_contract,
)


# ---------------------------------------------------------------------------
# Fake HiveMind server.
# ---------------------------------------------------------------------------


class _FakeHivemindHandler(BaseHTTPRequestHandler):
    """Per-test scriptable HiveMind /v1/chat/completions."""

    # Set by each test before requests fly.
    script: dict[str, object] = {}
    log: list[dict[str, object]] = []

    def do_POST(self):  # noqa: N802 - stdlib API
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        body_bytes = self.rfile.read(int(self.headers.get("Content-Length", "0") or "0"))
        body = json.loads(body_bytes.decode("utf-8") or "{}")
        _FakeHivemindHandler.log.append(body)
        model = body.get("model")
        per_model = _FakeHivemindHandler.script.get("per_model", {})
        plan = per_model.get(model, _FakeHivemindHandler.script.get("default", {"content": "fallback ok"}))
        if plan.get("stream") and body.get("stream"):
            self._send_sse_stream(plan)
            return
        content = plan.get("content", "")
        choice = {
            "message": {"role": "assistant", "content": content},
            "finish_reason": plan.get("finish_reason", "stop"),
        }
        payload = {"choices": [choice]}
        if plan.get("warning"):
            payload["hivemind_warning"] = plan["warning"]
        out = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def _send_sse_stream(self, plan: dict[str, object]):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        chunks = plan.get("chunks") or []
        for c in chunks:
            event = {"choices": [{"delta": {"content": c}}]}
            try:
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode("utf-8"))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                return
        if plan.get("send_done", True):
            try:
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                return
        else:
            # Simulate a stalled stream that never sends [DONE].
            # The FaceLobeChat stall-timeout guard must fire and return
            # what we have. Hold the connection open longer than the
            # client's stall_timeout to exercise the guard.
            hang_secs = float(plan.get("hang_after", 5.0))
            time.sleep(hang_secs)

    def log_message(self, *args, **kwargs):  # silence stdlib logging
        pass

    def finish(self):
        try:
            super().finish()
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            pass


@pytest.fixture
def fake_hivemind():
    _FakeHivemindHandler.script = {}
    _FakeHivemindHandler.log = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeHivemindHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        yield {
            "url": f"http://127.0.0.1:{port}",
            "log": _FakeHivemindHandler.log,
            "script": _FakeHivemindHandler.script,
        }
    finally:
        server.shutdown()
        thread.join(timeout=2)


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


def test_empty_content_triggers_fallback_to_default_model(fake_hivemind):
    fake_hivemind["script"]["per_model"] = {
        "phi4-mini": {"content": ""},  # chosen model returns empty
        "qwen3-coder-next:latest": {"content": "Hello from fallback."},
    }
    flc = FaceLobeChat(
        hivemind_url=fake_hivemind["url"],
        empty_fallback_model="qwen3-coder-next:latest",
    )
    result = flc.chat("Hi", session_id="s1", model="phi4-mini")
    assert result["text"] == "Hello from fallback."
    assert result["fallback_used"] is True
    assert result["requested_model"] == "phi4-mini"
    assert result["model"] == "qwen3-coder-next:latest"
    assert result["api_calls"] == 2
    # The fallback came in as a SECOND request; verify both models were tried.
    requested_models = [body["model"] for body in fake_hivemind["log"]]
    assert requested_models == ["phi4-mini", "qwen3-coder-next:latest"]


@pytest.mark.parametrize("raw_default", ["operator-pinned:latest", None])
def test_empty_retry_consults_picker_before_automatic_default(monkeypatch, raw_default):
    """Automatic recovery must not bypass the accepted picker contract.

    An explicitly injected constructor fallback remains a supported test and
    compatibility seam. Without one, neither ``MS4_DEFAULT_MODEL`` nor the
    historical static coder alias may be sent before the picker has selected a
    contract-valid foreground model.
    """
    if raw_default is None:
        monkeypatch.delenv("MS4_DEFAULT_MODEL", raising=False)
    else:
        monkeypatch.setenv("MS4_DEFAULT_MODEL", raw_default)

    events: list[tuple[str, str]] = []
    safe_model = "llama3.1:8b"

    def post(payload):
        target_model = payload["model"]
        events.append(("call", target_model))
        return ("safe reply" if target_model == safe_model else "", {})

    def pick(**_kwargs):
        events.append(("pick", safe_model))
        return type("Choice", (), {"model_id": safe_model})()

    monkeypatch.setattr(
        "machine_spirit_4.double_agent.model_picker.choose_foreground_model",
        pick,
    )
    chat = FaceLobeChat(hivemind_url="http://unused.invalid")
    monkeypatch.setattr(chat, "_post_blocking", post)

    result = chat.chat("Hi", session_id="ordering", model="primary:8b")

    assert events == [
        ("call", "primary:8b"),
        ("pick", safe_model),
        ("call", safe_model),
    ]
    assert result["api_calls"] == 2
    assert result["model"] == safe_model


def test_non_empty_content_does_not_retry(fake_hivemind):
    fake_hivemind["script"]["per_model"] = {
        "llama3.1:8b": {"content": "Hello, world."},
        "qwen3-coder-next:latest": {"content": "SHOULD NOT BE USED"},
    }
    flc = FaceLobeChat(
        hivemind_url=fake_hivemind["url"],
        empty_fallback_model="qwen3-coder-next:latest",
    )
    result = flc.chat("Hi", session_id="s2", model="llama3.1:8b")
    assert result["text"] == "Hello, world."
    assert result["fallback_used"] is False
    assert result["model"] == "llama3.1:8b"
    assert result["api_calls"] == 1


def test_initial_transport_error_uses_governed_fallback_without_dead_end(monkeypatch):
    """A pre-output peer stall must enter the existing bounded fallback chain.

    This is distinct from a partial stream: no fragment has reached the caller,
    so retrying cannot duplicate or splice visible text.
    """
    calls: list[str] = []

    face = FaceLobeChat(
        hivemind_url="http://unused.invalid",
        empty_fallback_model="fallback:4b",
    )

    def post(payload):
        target_model = payload["model"]
        calls.append(target_model)
        if target_model == "selected:4b":
            raise FaceLobeChatError(
                "HiveMind /v1/chat/completions unreachable: timed out",
                retryable=True,
                phase="transport_open",
            )
        return "Recovered through the governed Face fallback.", {
            "stream_completed": True,
            "http_latency_ms": 7,
            "bytes_received": 42,
            "stream_chunks": 0,
            "terminal_source": "finish_reason",
            "finish_reasons": ["stop"],
            "warning_types": [],
            "incomplete_reason": None,
            "downstream_cancelled": False,
        }

    monkeypatch.setattr(face, "_post_blocking", post)

    result = face.chat("Hi", session_id="transport-recovery", model="selected:4b")

    assert calls == ["selected:4b", "fallback:4b"]
    assert result["completed"] is True
    assert result["text"] == "Recovered through the governed Face fallback."
    assert result["requested_model"] == "selected:4b"
    assert result["model"] == "fallback:4b"
    assert result["fallback_used"] is True
    assert result["api_calls"] == 2
    assert result["metrics"]["upstream_recovery_attempted"] is True
    assert result["metrics"]["upstream_recovery_used"] is True
    assert (
        result["metrics"]["upstream_recovery_reason"]
        == "selected_model_transport_open"
    )


def test_initial_stream_open_error_retries_before_any_visible_fragment(monkeypatch):
    calls: list[str] = []
    spoken: list[str] = []
    recovered = "Recovered once through the streamed Face fallback."
    face = FaceLobeChat(
        hivemind_url="http://unused.invalid",
        empty_fallback_model="fallback:4b",
    )

    def post_streaming(payload, callback, **_kwargs):
        target_model = payload["model"]
        calls.append(target_model)
        if target_model == "selected:4b":
            raise FaceLobeChatError(
                "HiveMind /v1/chat/completions (stream) unreachable: timed out",
                retryable=True,
                phase="transport_open",
            )
        callback(recovered)
        return recovered, {
            "stream_completed": True,
            "http_latency_ms": 8,
            "stream_first_token_ms": 8,
            "bytes_received": 48,
            "stream_chunks": 1,
            "terminal_source": "finish_reason",
            "finish_reasons": ["stop"],
            "warning_types": [],
            "incomplete_reason": None,
            "downstream_cancelled": False,
        }

    monkeypatch.setattr(face, "_post_streaming", post_streaming)

    result = face.chat(
        "Hi",
        session_id="stream-transport-recovery",
        model="selected:4b",
        stream_callback=spoken.append,
    )

    assert calls == ["selected:4b", "fallback:4b"]
    assert spoken == [recovered]
    assert result["text"] == recovered
    assert result["completed"] is True
    assert result["model"] == "fallback:4b"
    assert result["api_calls"] == 2
    assert result["metrics"]["upstream_recovery_attempted"] is True
    assert result["metrics"]["upstream_recovery_used"] is True


def test_transport_error_picker_can_retry_same_governed_model_once(monkeypatch):
    calls: list[str] = []
    face = FaceLobeChat(hivemind_url="http://unused.invalid")

    def post(payload):
        target_model = payload["model"]
        calls.append(target_model)
        if len(calls) == 1:
            raise FaceLobeChatError(
                "HiveMind /v1/chat/completions unreachable: timed out",
                retryable=True,
                phase="transport_open",
            )
        return "Recovered on the selected model's second route attempt.", {
            "stream_completed": True,
            "http_latency_ms": 9,
            "bytes_received": 51,
            "stream_chunks": 0,
            "terminal_source": "finish_reason",
            "finish_reasons": ["stop"],
            "warning_types": [],
            "incomplete_reason": None,
            "downstream_cancelled": False,
        }

    def pick(**_kwargs):
        return type("Choice", (), {"model_id": "selected:4b"})()

    monkeypatch.setattr(face, "_post_blocking", post)
    monkeypatch.setattr(
        "machine_spirit_4.double_agent.model_picker.choose_foreground_model",
        pick,
    )

    result = face.chat("Hi", session_id="same-model-retry", model="selected:4b")

    assert calls == ["selected:4b", "selected:4b"]
    assert result["completed"] is True
    assert result["model"] == "selected:4b"
    assert result["api_calls"] == 2
    assert result["fallback_used"] is True
    assert result["metrics"]["upstream_recovery_attempted"] is True
    assert result["metrics"]["upstream_recovery_used"] is True


def test_same_model_stream_retry_preserves_turn_relative_metrics(monkeypatch):
    calls: list[str] = []
    spoken: list[str] = []
    face = FaceLobeChat(hivemind_url="http://unused.invalid")

    def post_streaming(payload, callback, **_kwargs):
        calls.append(payload["model"])
        if len(calls) == 1:
            time.sleep(0.03)
            raise FaceLobeChatError(
                "transient peer open timeout",
                retryable=True,
                phase="transport_open",
            )
        callback("recovered")
        return "recovered", {
            "stream_completed": True,
            "http_latency_ms": 7,
            "stream_first_token_ms": 5,
            "bytes_received": 64,
            "stream_chunks": 2,
            "prompt_tokens": 11,
            "completion_tokens": 3,
            "total_tokens": 14,
            "terminal_source": "finish_reason",
            "finish_reasons": ["stop"],
            "warning_types": [],
            "incomplete_reason": None,
            "downstream_cancelled": False,
        }

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    monkeypatch.setattr(
        "machine_spirit_4.double_agent.model_picker.choose_foreground_model",
        lambda **_kwargs: type("Choice", (), {"model_id": "selected:4b"})(),
    )

    result = face.chat(
        "Hi",
        session_id="same-model-stream-metrics",
        model="selected:4b",
        stream_callback=spoken.append,
    )

    metrics = result["metrics"]
    assert calls == ["selected:4b", "selected:4b"]
    assert spoken == ["recovered"]
    assert metrics["stream_first_token_ms"] >= 30
    assert metrics["stream_chunks"] == 2
    assert metrics["prompt_tokens"] == 11
    assert metrics["completion_tokens"] == 3
    assert metrics["total_tokens"] == 14
    assert metrics["http_latency_ms"] >= 30


def test_cancellation_during_picker_prevents_recovery_request(monkeypatch):
    calls: list[str] = []
    cancel = threading.Event()
    face = FaceLobeChat(hivemind_url="http://unused.invalid")

    def post_streaming(payload, _callback, **_kwargs):
        calls.append(payload["model"])
        raise FaceLobeChatError(
            "transient peer open timeout",
            retryable=True,
            phase="transport_open",
        )

    def pick(**_kwargs):
        cancel.set()
        return type("Choice", (), {"model_id": "selected:4b"})()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    monkeypatch.setattr(
        "machine_spirit_4.double_agent.model_picker.choose_foreground_model",
        pick,
    )

    result = face.chat(
        "Hi",
        session_id="cancel-during-picker",
        model="selected:4b",
        stream_callback=lambda _fragment: None,
        cancel_event=cancel,
    )

    assert calls == ["selected:4b"]
    assert result["cancelled"] is True
    assert result["completed"] is False
    assert result["api_calls"] == 1
    assert result["metrics"]["upstream_recovery_attempted"] is True
    assert result["metrics"]["upstream_recovery_used"] is False
    assert face._sessions["cancel-during-picker"].messages == []


def test_cancellation_after_initial_failure_prevents_explicit_fallback(monkeypatch):
    calls: list[str] = []
    cancel = threading.Event()
    face = FaceLobeChat(
        hivemind_url="http://unused.invalid",
        empty_fallback_model="fallback:4b",
    )

    def post(payload):
        calls.append(payload["model"])
        cancel.set()
        raise FaceLobeChatError(
            "transient peer open timeout",
            retryable=True,
            phase="transport_open",
        )

    monkeypatch.setattr(face, "_post_blocking", post)

    with pytest.raises(FaceLobeChatError, match="transient peer open timeout"):
        face.chat(
            "Hi",
            session_id="cancel-before-explicit-fallback",
            model="selected:4b",
            cancel_event=cancel,
        )

    assert calls == ["selected:4b"]
    assert face._sessions["cancel-before-explicit-fallback"].messages == []


def test_retry_reasoning_truncation_is_not_masked_by_initial_failure(monkeypatch):
    calls: list[str] = []
    face = FaceLobeChat(hivemind_url="http://unused.invalid")

    def post(payload):
        calls.append(payload["model"])
        if len(calls) == 1:
            raise FaceLobeChatError(
                "transient peer open timeout",
                retryable=True,
                phase="transport_open",
            )
        return "", {
            "stream_completed": False,
            "http_latency_ms": 6,
            "bytes_received": 32,
            "stream_chunks": 1,
            "terminal_source": "finish_reason",
            "finish_reasons": ["length"],
            "warning_types": ["reasoning_model_truncated"],
            "incomplete_reason": "reasoning_model_truncated",
            "downstream_cancelled": False,
            "reasoning_chars": 321,
        }

    monkeypatch.setattr(face, "_post_blocking", post)
    monkeypatch.setattr(
        "machine_spirit_4.double_agent.model_picker.choose_foreground_model",
        lambda **_kwargs: type("Choice", (), {"model_id": "selected:4b"})(),
    )

    result = face.chat(
        "Hi",
        session_id="retry-reasoning-truncated",
        model="selected:4b",
    )

    assert calls == ["selected:4b", "selected:4b"]
    assert result["completed"] is False
    assert result["metrics"]["incomplete_reason"] == "reasoning_model_truncated"
    assert result["metrics"]["warning_types"] == ["reasoning_model_truncated"]
    assert result["metrics"]["reasoning_chars"] == 321
    assert result["metrics"]["upstream_recovery_used"] is False
    assert result["fallback_used"] is False


def test_nonretryable_face_error_does_not_enter_recovery(monkeypatch):
    calls: list[str] = []
    face = FaceLobeChat(hivemind_url="http://unused.invalid")

    def post(payload):
        calls.append(payload["model"])
        raise FaceLobeChatError("invalid API key", phase="http_status")

    monkeypatch.setattr(face, "_post_blocking", post)
    monkeypatch.setattr(
        "machine_spirit_4.double_agent.model_picker.choose_foreground_model",
        lambda **_kwargs: pytest.fail("picker must not run for nonretryable errors"),
    )

    with pytest.raises(FaceLobeChatError, match="invalid API key"):
        face.chat("Hi", session_id="nonretryable", model="selected:4b")
    assert calls == ["selected:4b"]


def test_terminal_retry_error_replaces_initial_transport_receipt(monkeypatch):
    calls: list[str] = []
    face = FaceLobeChat(hivemind_url="http://unused.invalid")

    def post(payload):
        calls.append(payload["model"])
        if len(calls) == 1:
            raise FaceLobeChatError(
                "transient peer open timeout",
                retryable=True,
                phase="transport_open",
            )
        raise FaceLobeChatError(
            "malformed provider JSON",
            retryable=False,
            phase="protocol_json",
        )

    monkeypatch.setattr(face, "_post_blocking", post)
    monkeypatch.setattr(
        "machine_spirit_4.double_agent.model_picker.choose_foreground_model",
        lambda **_kwargs: type("Choice", (), {"model_id": "selected:4b"})(),
    )

    result = face.chat(
        "Hi",
        session_id="terminal-retry-error",
        model="selected:4b",
    )

    assert calls == ["selected:4b", "selected:4b"]
    assert result["completed"] is False
    assert result["api_calls"] == 2
    assert result["fallback_used"] is False
    assert result["metrics"]["upstream_recovery_reason"] == (
        "selected_model_transport_open"
    )
    assert result["metrics"]["incomplete_reason"] == (
        "picker_fallback_protocol_json"
    )
    assert result["metrics"]["upstream_error_phase"] == "protocol_json"
    assert result["metrics"]["upstream_error_retryable"] is False
    assert result["metrics"]["upstream_attempt_kind"] == "picker_fallback"


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_transient_face_http_statuses_are_retryable(status):
    assert _retryable_face_http_status(status) is True


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 422])
def test_deterministic_face_http_statuses_are_not_retryable(status):
    assert _retryable_face_http_status(status) is False


def test_cancelled_at_entry_commits_no_history_and_makes_no_model_call(fake_hivemind):
    """Finding 3 (R4): a turn already cancelled at entry must not call the model
    or mutate the session it no longer owns."""
    fake_hivemind["script"]["per_model"] = {"m": {"content": "hello"}}
    flc = FaceLobeChat(hivemind_url=fake_hivemind["url"])
    seed = flc.chat("first", session_id="sc1", model="m")
    assert seed["completed"] is True
    before = list(flc._sessions["sc1"].messages)
    calls_before = len(fake_hivemind["log"])
    cancel = threading.Event()
    cancel.set()
    result = flc.chat("second", session_id="sc1", model="m", cancel_event=cancel)
    assert result["cancelled"] is True and result["completed"] is False
    assert list(flc._sessions["sc1"].messages) == before, "no history mutation for a pre-cancelled turn"
    assert len(fake_hivemind["log"]) == calls_before, "no model call for a pre-cancelled turn"


def test_cancelled_mid_stream_skips_history_commit(fake_hivemind):
    """Finding 3 (R4): a turn cancelled DURING the model call (a replacement turn
    barges it) must NOT commit user/assistant history when it finishes — a stale
    turn cannot grow or reorder the session's history."""
    fake_hivemind["script"]["per_model"] = {"m": {"stream": True, "chunks": ["hel", "lo"], "content": "hello"}}
    flc = FaceLobeChat(hivemind_url=fake_hivemind["url"])
    flc.chat("first", session_id="sc2", model="m")  # seed one committed turn
    before = list(flc._sessions["sc2"].messages)
    cancel = threading.Event()

    def on_delta(_chunk):
        cancel.set()  # a replacement turn barges us mid-stream

    result = flc.chat("second", session_id="sc2", model="m", stream_callback=on_delta, cancel_event=cancel)
    assert result["cancelled"] is True and result["completed"] is False
    assert list(flc._sessions["sc2"].messages) == before, "a cancelled mid-stream turn must not commit history"


def test_metrics_block_is_populated_on_blocking(fake_hivemind):
    """Every chat response carries a 'metrics' block with ISO timestamps,
    duration, and token counts from HiveMind's usage field when the
    upstream populates it. The UI surfaces this row under each
    message; without it the operator can't see TTFB or token usage."""

    class _UsageHandler(_FakeHivemindHandler):
        def do_POST(self):  # noqa: N802
            body_bytes = self.rfile.read(int(self.headers.get("Content-Length", "0") or "0"))
            _ = body_bytes
            payload = {
                "choices": [{"message": {"role": "assistant", "content": "Hi!"}}],
                "usage": {"prompt_tokens": 42, "completion_tokens": 7, "total_tokens": 49},
            }
            out = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *args, **kwargs):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), _UsageHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        flc = FaceLobeChat(hivemind_url=f"http://127.0.0.1:{port}")
        result = flc.chat("Hi", session_id="s-metrics", model="llama3.1:8b")
        m = result["metrics"]
        assert m["schema"] == "Ms4TurnMetrics.v1"
        assert m["started_at"].endswith("Z")
        assert m["completed_at"].endswith("Z")
        # Local handler can return in <1ms on a hot loop so duration may
        # round to 0; the value just needs to be a non-negative int.
        assert m["duration_ms"] >= 0
        assert isinstance(m["duration_ms"], int)
        assert m["http_latency_ms"] >= 0
        assert m["api_calls"] == 1
        assert m["prompt_tokens"] == 42
        assert m["completion_tokens"] == 7
        assert m["total_tokens"] == 49
        # tokens_per_second is derived; if the fake handler returns in
        # sub-millisecond time then duration_ms rounds to 0 and the
        # guard leaves tokens_per_second as None. Either is acceptable.
        assert m["tokens_per_second"] is None or m["tokens_per_second"] > 0
        assert m["effective_model"] == "llama3.1:8b"
        assert m["requested_model"] == "llama3.1:8b"
        assert m["streaming"] is False
        assert m["fallback_used"] is False
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_metrics_captures_ttfb_and_chunks_on_streaming(fake_hivemind):
    fake_hivemind["script"]["per_model"] = {
        "phi4-mini": {
            "stream": True,
            "chunks": ["Hi", " there", "!"],
            "send_done": True,
        },
    }
    flc = FaceLobeChat(
        hivemind_url=fake_hivemind["url"],
        empty_fallback_model="qwen3-coder-next:latest",
    )
    captured: list[str] = []
    result = flc.chat(
        "Hi",
        session_id="s-stream-metrics",
        model="phi4-mini",
        stream_callback=captured.append,
    )
    assert captured == ["Hi", " there", "!"]
    m = result["metrics"]
    assert m["streaming"] is True
    assert m["stream_chunks"] >= 3
    # TTFB is recorded as soon as the first content chunk arrives.
    assert m["stream_first_token_ms"] is not None
    assert m["stream_first_token_ms"] >= 0
    assert m["duration_ms"] >= m["stream_first_token_ms"]


def test_system_prompt_contains_anti_hallucination_clauses():
    """Lock in the truthfulness rules so a future prompt edit can't
    silently drop them. These are the rules that failed in live
    testing (model citing fake job UUIDs, fabricating completion
    timestamps, relabeling CPUs as VMs).

    The May-2026 rewrite softened the prompt's tone (the strict
    'hard failure' framing made the model refuse small talk like
    'can you hear me?'); the protections themselves stay in. The
    checks below are intentionally semantic, not verbatim, so the
    prompt can keep evolving its voice while we still catch any
    real regression in the anti-fabrication contract.
    """
    from machine_spirit_4.gateway.face_lobe_chat import FACE_LOBE_SYSTEM_PROMPT

    prompt = FACE_LOBE_SYSTEM_PROMPT.lower()

    # The model once inverted the conversation and greeted the human as
    # "Oracle". Keep the two roles explicit even when the selected Face model
    # is small and latency-optimized.
    assert "human operator" in prompt, "operator identity anchor missing"
    assert "never address or label the operator as oracle" in prompt, \
        "anti-identity-inversion rule missing"

    # Dispatch line is authoritative.
    assert "this turn dispatched" in prompt, "dispatch-line anchor missing"

    # No fabricating UUIDs.
    assert "uuid" in prompt and "don't invent" in prompt, \
        "anti-fabrication rule for job UUIDs missing"

    # No fabricating completion timestamps. The new phrasing is
    # "don't fabricate timestamps for completions" + "Use only the
    # dates you can see in the context block."
    assert "fabricate timestamps" in prompt, \
        "anti-fabrication rule for completion timestamps missing"
    assert "dates you can see" in prompt, \
        "date/time grounding anchor missing"

    # Don't relabel resource types — the live failure was the model
    # answering "Which VMs are running?" with a CPU listing.
    assert "cpus are cpus" in prompt and "not vms or gpus" in prompt, \
        "resource-type-relabeling rule missing"

    # Attribute Depth Lobe results clearly.
    assert "depth lobe found" in prompt, \
        "Depth Lobe attribution example missing"

    # Tool names.
    assert "do not invent tool names" in prompt, \
        "tool-name fabrication rule missing"

    # May-31 2026: the prompt must FORBID telling the operator to type a
    # slash command (impossible on voice) — deep work auto-dispatches.
    assert "/deep" in prompt, "/deep handling missing from prompt"
    plower = prompt.lower()
    assert "never tell" in plower and "slash command" in plower, \
        "prompt must forbid instructing the user to type a slash command"

    # And — the May-2026 fix — small talk must be explicitly allowed
    # so the model doesn't refuse "can you hear me?" as if it were a
    # forbidden tool call.
    assert "small talk" in prompt or "natural" in prompt, \
        "conversational/natural-speech allowance missing"


def test_system_prompt_dispatch_ack_uses_automatic_delivery_not_next_turn():
    from machine_spirit_4.gateway.face_lobe_chat import FACE_LOBE_SYSTEM_PROMPT

    prompt = " ".join(FACE_LOBE_SYSTEM_PROMPT.lower().split())
    for required in (
        "verified result will appear automatically in this conversation when it is ready",
        "does not need to ask again",
        "never promise an extra turn",
        "never promise an extra turn or require a follow-up to deliver it",
    ):
        assert required in prompt
    for stale in (
        "i'll surface the result on the next turn",
        "dispatched, give me a sec",
        "the operator will follow up",
        "operator will follow up",
    ):
        assert stale not in prompt


def test_metrics_on_fallback_path_counts_two_api_calls(fake_hivemind):
    fake_hivemind["script"]["per_model"] = {
        "phi4-mini": {"content": ""},  # empty -> triggers fallback
        "qwen3-coder-next:latest": {"content": "Fallback reply."},
    }
    flc = FaceLobeChat(
        hivemind_url=fake_hivemind["url"],
        empty_fallback_model="qwen3-coder-next:latest",
    )
    result = flc.chat("Hi", session_id="s-fb-metrics", model="phi4-mini")
    m = result["metrics"]
    assert m["api_calls"] == 2
    assert m["fallback_used"] is True
    assert m["requested_model"] == "phi4-mini"
    assert m["effective_model"] == "qwen3-coder-next:latest"


def test_is_local_ollama_model_classifies_by_tag():
    """keep_alive is Ollama-specific; only local ``name:tag`` models get
    it. Hosted-provider ids (no colon, or gpt-/claude-/o-prefixed) must
    be excluded so they aren't sent an unknown body field."""
    from machine_spirit_4.gateway.face_lobe_chat import _is_local_ollama_model

    assert _is_local_ollama_model("llama3.1:8b") is True
    assert _is_local_ollama_model("qwen3-coder-next:latest") is True
    assert _is_local_ollama_model("phi4-mini:latest") is True
    # Hosted providers — no keep_alive.
    assert _is_local_ollama_model("gpt-4o-mini") is False
    assert _is_local_ollama_model("gpt-5.2") is False
    assert _is_local_ollama_model("claude-3-5-haiku-20241022") is False
    assert _is_local_ollama_model("o4-mini") is False
    assert _is_local_ollama_model(None) is False
    assert _is_local_ollama_model("") is False


def test_face_keep_alive_default_and_disable(monkeypatch):
    from machine_spirit_4.gateway.face_lobe_chat import _face_keep_alive

    monkeypatch.delenv("MS4_FACE_KEEP_ALIVE", raising=False)
    assert _face_keep_alive("llama3.1:8b") == "10m"      # default
    assert _face_keep_alive("gpt-4o-mini") is None        # hosted -> never
    monkeypatch.setenv("MS4_FACE_KEEP_ALIVE", "30m")
    assert _face_keep_alive("llama3.1:8b") == "30m"
    monkeypatch.setenv("MS4_FACE_KEEP_ALIVE", "")         # explicit disable
    assert _face_keep_alive("llama3.1:8b") is None


def test_face_thinking_defaults_off_and_has_explicit_override(monkeypatch):
    from machine_spirit_4.gateway.face_lobe_chat import _face_thinking_enabled

    monkeypatch.delenv("MS4_FACE_ENABLE_THINKING", raising=False)
    assert _face_thinking_enabled() is False
    monkeypatch.setenv("MS4_FACE_ENABLE_THINKING", "true")
    assert _face_thinking_enabled() is True
    monkeypatch.setenv("MS4_FACE_ENABLE_THINKING", "off")
    assert _face_thinking_enabled() is False


def test_face_sends_portable_no_thinking_control(fake_hivemind, monkeypatch):
    """Face delegates long reasoning to Depth and must request visible text."""
    monkeypatch.delenv("MS4_FACE_ENABLE_THINKING", raising=False)
    fake_hivemind["script"]["per_model"] = {
        "nemotron-3-nano:4b": {"content": "ready"},
    }
    flc = FaceLobeChat(hivemind_url=fake_hivemind["url"])

    result = flc.chat(
        "Hi",
        session_id="no-think-default",
        model="nemotron-3-nano:4b",
    )

    assert result["model"] == "nemotron-3-nano:4b"
    assert fake_hivemind["log"][-1]["enable_thinking"] is False


def test_keep_alive_sent_for_local_model_not_hosted(fake_hivemind, monkeypatch):
    """The anti-thrash residency bias: a local Ollama-tag Face model must
    carry keep_alive on the wire so HiveMind keeps IT resident over a
    transient Depth model; a hosted model must NOT (it would 400)."""
    monkeypatch.delenv("MS4_FACE_KEEP_ALIVE", raising=False)
    fake_hivemind["script"]["per_model"] = {
        "llama3.1:8b": {"content": "local reply"},
        "gpt-4o-mini": {"content": "hosted reply"},
    }
    flc = FaceLobeChat(hivemind_url=fake_hivemind["url"])
    flc.chat("Hi", session_id="ka-local", model="llama3.1:8b")
    flc.chat("Hi", session_id="ka-hosted", model="gpt-4o-mini")
    by_model = {b["model"]: b for b in fake_hivemind["log"]}
    assert by_model["llama3.1:8b"].get("keep_alive") == "10m"
    assert "keep_alive" not in by_model["gpt-4o-mini"]


def test_blocking_truncation_warning_is_incomplete_and_never_committed(fake_hivemind):
    fake_hivemind["script"]["per_model"] = {
        "face-test": {
            "content": "Useful but cut off",
            "finish_reason": "length",
            "warning": {"type": "reasoning_model_truncated"},
        },
    }
    face = FaceLobeChat(hivemind_url=fake_hivemind["url"])

    result = face.chat("Explain fully", session_id="blocking-truncated", model="face-test")

    assert result["text"] == "Useful but cut off"
    assert result["completed"] is False
    assert result["metrics"]["stream_completed"] is False
    assert "reasoning_model_truncated" in result["metrics"]["warning_types"]
    assert "reasoning_model_truncated" in result["metrics"]["incomplete_reason"]
    assert face._sessions["blocking-truncated"].messages == []


def test_stream_stall_returns_partial_text_without_hanging(fake_hivemind):
    """If HiveMind never sends [DONE], the FaceLobeChat must time out
    on the stall budget and expose whatever fragments arrived as incomplete. The
    previous implementation hung the worker thread forever, which
    caused the user-visible '(no reply text)' UI state."""
    fake_hivemind["script"]["per_model"] = {
        "phi4-mini": {
            "stream": True,
            "chunks": ["Hi", " there"],
            "send_done": False,
            "hang_after": 6.0,
        },
    }
    flc = FaceLobeChat(
        hivemind_url=fake_hivemind["url"],
        empty_fallback_model="qwen3-coder-next:latest",  # not exercised here
        stream_stall_timeout=2,
    )
    captured: list[str] = []
    t0 = time.time()
    result = flc.chat(
        "Hi",
        session_id="s-stall",
        model="phi4-mini",
        stream_callback=captured.append,
    )
    elapsed = time.time() - t0
    # The fallback path will also be invoked because what we got was
    # only "Hi there" + nothing else — actually it's non-empty so no
    # fallback. We just need: (a) we DON'T hang past the stall budget
    # by a wide margin, (b) we collected the partial fragments.
    assert "Hi" in result["text"]
    assert "there" in result["text"]
    assert captured == ["Hi", " there"]
    assert result["completed"] is False
    assert result["metrics"]["stream_completed"] is False
    assert result["metrics"]["incomplete_reason"]
    assert flc._sessions["s-stall"].messages == []
    # Generous upper bound: stall_timeout 2s + handler hang 6s but the
    # readline will time out at the socket-level stall.
    assert elapsed < 10.0, f"FaceLobeChat should not hang on stalled streams; took {elapsed:.2f}s"


def test_stream_callback_false_stops_stream_early(fake_hivemind):
    """The text SSE route uses ``stream_callback`` as its cooperative
    disconnect signal. Returning False must stop the upstream Face Lobe
    stream and return the partial text collected so far instead of
    burning through the whole model response after the browser left.
    """
    fake_hivemind["script"]["per_model"] = {
        "phi4-mini": {
            "stream": True,
            "chunks": ["first", " second", " third"],
            "send_done": True,
        },
    }
    flc = FaceLobeChat(
        hivemind_url=fake_hivemind["url"],
        empty_fallback_model="qwen3-coder-next:latest",
    )
    captured: list[str] = []

    def stop_after_first(fragment: str) -> bool:
        captured.append(fragment)
        return False

    result = flc.chat(
        "Hi",
        session_id="s-cancel",
        model="phi4-mini",
        stream_callback=stop_after_first,
    )

    assert captured == ["first"]
    assert result["text"] == "first"
    assert result["completed"] is False
    assert result["cancelled"] is True
    assert result["metrics"]["stream_chunks"] == 1
    assert result["metrics"]["stream_completed"] is False
    assert result["metrics"]["downstream_cancelled"] is True
    assert flc._sessions["s-cancel"].messages == []


def test_first_token_probe_uses_production_payload_without_history_commit(fake_hivemind):
    fake_hivemind["script"]["per_model"] = {
        "selected:4b": {
            "stream": True,
            "chunks": ["r", "eady", " extra"],
            "send_done": True,
        },
    }
    flc = FaceLobeChat(
        hivemind_url=fake_hivemind["url"],
        allow_model_fallback=False,
    )
    cancel = threading.Event()

    result = flc.probe_first_token(
        "Reply with one visible token.",
        model="selected:4b",
        extra_system="cache-busting production context",
        cancel_event=cancel,
        first_token_timeout=4.0,
        max_visible_tokens=1,
    )

    assert result["text"] == "r"
    assert result["model"] == "selected:4b"
    assert result["requested_model"] == "selected:4b"
    assert result["first_token_observed"] is True
    assert result["visible_tokens"] == 1
    assert result["first_token_ms"] is not None
    assert result["completed"] is True
    assert result["metrics"]["downstream_cancelled"] is False
    assert result["metrics"]["terminal_source"] == "done_marker"
    assert flc._sessions == {}, "admission probes must not create or mutate chat history"
    assert len(fake_hivemind["log"]) == 1
    payload = fake_hivemind["log"][0]
    assert payload["model"] == "selected:4b"
    assert payload["stream"] is True
    assert payload["max_tokens"] == 1
    assert payload["options"] == {"num_ctx": 8192}
    assert payload["messages"][-1]["content"] == "Reply with one visible token."
    assert "cache-busting production context" in payload["messages"][0]["content"]


def test_face_context_and_location_policy_are_explicit_and_configurable(monkeypatch):
    monkeypatch.setenv("MS4_FACE_NUM_CTX", "16384")
    monkeypatch.setenv("MS4_FACE_LOCATION_POLICY", "prefer_peer")
    face = FaceLobeChat(hivemind_url="http://hive.test:6089")

    assert face.num_ctx == 16384
    assert face.location_policy == "prefer_peer"
    request = face._build_request(
        {
            "model": "selected:4b",
            "messages": [],
            "stream": True,
            "options": {"num_ctx": face.num_ctx},
        }
    )
    headers = {key.lower(): value for key, value in request.header_items()}
    assert headers["x-hivemind-location"] == "prefer_peer"

    monkeypatch.setenv("MS4_FACE_NUM_CTX", "not-an-integer")
    monkeypatch.setenv("MS4_FACE_LOCATION_POLICY", "somewhere-fast")
    fallback = FaceLobeChat(hivemind_url="http://hive.test:6089")
    assert fallback.num_ctx == 8192
    assert fallback.location_policy == "prefer_local"


@pytest.mark.parametrize(
    ("stats_override", "expected_completed"),
    [
        (
            {
                "terminal_source": "done_marker",
                "stream_completed": False,
                "warning_types": ["reasoning_model_truncated"],
                "incomplete_reason": "reasoning_model_truncated",
            },
            False,
        ),
        (
            {
                "terminal_source": "done_marker",
                "stream_completed": False,
                "malformed_frames": 1,
                "incomplete_reason": "malformed SSE frames: 1",
            },
            False,
        ),
        (
            {
                "terminal_source": "finish_reason",
                "stream_completed": False,
                "finish_reasons": ["content_filter"],
                "incomplete_reason": "non-success finish_reason: content_filter",
            },
            False,
        ),
        (
            {
                "terminal_source": "finish_reason",
                "stream_completed": False,
                "finish_reasons": ["length"],
                "incomplete_reason": "non-success finish_reason: length",
            },
            True,
        ),
    ],
)
def test_first_token_probe_requires_clean_terminal_metadata(
    monkeypatch,
    stats_override,
    expected_completed,
):
    flc = FaceLobeChat(
        hivemind_url="http://127.0.0.1:1",
        allow_model_fallback=False,
    )

    def fake_post_streaming(_payload, callback, **_kwargs):
        callback("r")
        stats = {
            "stream_first_token_ms": 12,
            "terminal_source": "done_marker",
            "stream_completed": True,
            "finish_reasons": [],
            "warning_types": [],
            "incomplete_reason": None,
            "downstream_cancelled": False,
            "malformed_frames": 0,
        }
        stats.update(stats_override)
        return "r", stats

    monkeypatch.setattr(flc, "_post_streaming", fake_post_streaming)

    result = flc.probe_first_token(
        "probe",
        model="selected:4b",
        extra_system="cache-busted context",
        first_token_timeout=4.0,
        max_visible_tokens=1,
    )

    assert result["first_token_observed"] is True
    assert result["completed"] is expected_completed


def test_first_token_probe_pre_cancelled_makes_no_request(fake_hivemind):
    flc = FaceLobeChat(hivemind_url=fake_hivemind["url"], allow_model_fallback=False)
    cancel = threading.Event()
    cancel.set()

    result = flc.probe_first_token(
        "probe",
        model="selected:4b",
        extra_system="context",
        cancel_event=cancel,
        first_token_timeout=4.0,
        max_visible_tokens=1,
    )

    assert result["first_token_observed"] is False
    assert result["first_token_ms"] is None
    assert result["cancelled"] is True
    assert fake_hivemind["log"] == []


def test_first_token_probe_no_visible_token_hits_deadline_and_cancels(
    fake_hivemind,
    monkeypatch,
):
    fake_hivemind["script"]["per_model"] = {
        "selected:4b": {
            "stream": True,
            "chunks": [],
            "send_done": False,
            "hang_after": 1.0,
        },
    }
    flc = FaceLobeChat(
        hivemind_url=fake_hivemind["url"],
        allow_model_fallback=False,
    )
    cancellations: list[tuple[str, str]] = []

    def cancel(trace_id: str, *, reason: str) -> bool:
        cancellations.append((trace_id, reason))
        return True

    monkeypatch.setattr(flc, "_cancel_upstream_trace", cancel)
    started = time.monotonic()
    result = flc.probe_first_token(
        "probe",
        model="selected:4b",
        extra_system="cache-busted context",
        first_token_timeout=0.1,
        max_visible_tokens=1,
    )
    elapsed = time.monotonic() - started

    assert elapsed < 0.75
    assert result["first_token_observed"] is False
    assert result["first_token_ms"] is None
    assert result["completed"] is False
    assert result["metrics"]["incomplete_reason"] == (
        "MS4 Face Lobe admission first-token deadline"
    )
    assert len(cancellations) == 1
    assert cancellations[0][1] == "MS4 Face Lobe admission first-token deadline"


def test_stream_open_uses_http_timeout_then_stall_timeout_for_reads(monkeypatch):
    """A healthy cold open may exceed the read-stall budget.

    Opening still gets the established HTTP timeout. Once open, every
    blocking read gets the shorter stall timeout, and downstream
    cancellation still closes and cancels the exact upstream trace.
    """
    cold_open_seconds = 14
    open_timeouts: list[float] = []
    cancel_calls: list[tuple[str, str]] = []

    class _Socket:
        def __init__(self):
            self.timeout: float | None = None
            self.set_calls: list[float] = []

        def settimeout(self, timeout):
            self.timeout = timeout
            self.set_calls.append(timeout)

    class _Raw:
        def __init__(self, sock):
            self._sock = sock

    class _File:
        def __init__(self, sock):
            self.raw = _Raw(sock)

    class _Response:
        def __init__(self):
            self.socket = _Socket()
            self.fp = _File(self.socket)
            self.lines = [
                b"\n",
                b'data: {"choices":[{"delta":{"content":"hello"}}]}\n',
            ]
            self.read_timeouts: list[float | None] = []
            self.closed = False

        def readline(self):
            self.read_timeouts.append(self.socket.timeout)
            if not self.lines:
                raise AssertionError("stream read continued after cancellation")
            return self.lines.pop(0)

        def close(self):
            self.closed = True

    response = _Response()

    def fake_urlopen(_request, timeout):
        open_timeouts.append(timeout)
        if timeout < cold_open_seconds:
            raise TimeoutError("cold open exceeded supplied timeout")
        # urllib leaves the open timeout on the response socket.
        response.socket.timeout = timeout
        return response

    def fake_cancel(trace_id, *, reason):
        cancel_calls.append((trace_id, reason))
        return True

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    face = FaceLobeChat(
        hivemind_url="http://hive.test:6089",
        http_timeout=60,
        stream_stall_timeout=12,
    )
    monkeypatch.setattr(face, "_cancel_upstream_trace", fake_cancel)

    text, stats = face._post_streaming(
        {"model": "llama3.1:8b", "messages": [], "stream": True},
        lambda _fragment: False,
    )

    assert face.stream_stall_timeout < cold_open_seconds < face.http_timeout
    assert open_timeouts == [face.http_timeout]
    assert response.socket.set_calls == [face.stream_stall_timeout]
    assert response.read_timeouts == [
        face.stream_stall_timeout,
        face.stream_stall_timeout,
    ]
    assert text == "hello"
    assert response.closed is True
    assert cancel_calls == [
        (stats["trace_id"], "MS4 Face Lobe downstream canceled stream")
    ]
    assert stats["upstream_canceled"] is True
    assert stats["stream_completed"] is False
    assert stats["downstream_cancelled"] is True


def test_stream_open_timeout_cancels_the_exact_hli_trace(monkeypatch):
    calls: list[tuple[urllib.request.Request, float]] = []

    class _DeleteResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def fake_urlopen(request, timeout):
        calls.append((request, timeout))
        if request.method == "POST":
            raise TimeoutError("timed out")
        return _DeleteResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    face = FaceLobeChat(hivemind_url="http://hive.test:6089")

    with pytest.raises(FaceLobeChatError, match="timed out"):
        face._post_streaming(
            {"model": "llama3.1:8b", "messages": [], "stream": True},
            lambda _fragment: True,
        )

    post_request = calls[0][0]
    post_headers = {key.lower(): value for key, value in post_request.header_items()}
    trace_id = post_headers["x-request-id"]
    assert str(uuid.UUID(trace_id)) == trace_id
    assert post_headers["x-hivemind-client-trace"] == trace_id
    assert post_headers["x-hivemind-location"] == "prefer_local"
    delete_request, delete_timeout = calls[1]
    assert delete_request.method == "DELETE"
    assert f"/jobs/{trace_id}?" in delete_request.full_url
    assert "stream+open+failure" in delete_request.full_url
    assert delete_timeout == 2.0


def test_fresh_conversation_block_does_not_buffer_neutral_stream(tmp_path):
    """Regression: the static anti-hallucination rules appended to EVERY
    context block contain 'terminal Depth Lobe jobs without verified
    results' mid-line; the old bare substring detector full-buffered every
    normal voice turn. Only the line-anchored section header may buffer."""
    from machine_spirit_4.double_agent import Blackboard, build_face_lobe_context_block
    from machine_spirit_4.gateway.face_lobe_chat import _should_buffer_for_output_guard

    board = Blackboard(tmp_path / "double_agent.sqlite3")
    board.bump_revision("conv-fresh", user_message_excerpt="hey there")
    block = build_face_lobe_context_block(conversation_id="conv-fresh", blackboard=board)
    assert block is not None
    assert "terminal Depth Lobe jobs without verified results" in block
    assert _should_buffer_for_output_guard("Hey, how are you?", block) is False


def test_genuine_unverified_terminal_section_header_buffers_stream():
    from machine_spirit_4.gateway.face_lobe_chat import _should_buffer_for_output_guard

    block = (
        "MS4 Face Lobe context (authoritative; do not contradict):\n"
        "- terminal Depth Lobe jobs without verified results "
        "(MOST RECENT FIRST; do NOT present as successful):\n"
        "    - da-1234 [failed] goal='Diagnose the issue.'"
    )
    assert _should_buffer_for_output_guard("Hey, how are you?", block) is True


def _r2_verified_depth_context() -> str:
    return (
        "MS4 Face Lobe context (authoritative; do not contradict):\n"
        "- THIS TURN DID NOT DISPATCH any background work.\n"
        "- no active background jobs\n"
        "- verified completed Depth Lobe jobs "
        "(MOST RECENT FIRST; use these answers when relevant):\n"
        "    - da-4410e950-e757-4a3c-842c-5b89d4ab4e8d [completed] "
        "goal='Compare Face and Depth'\n"
        "      result: Verified answer is complete."
    )


def _first_token_revision_with_stale_verified_delivery() -> str:
    candidate = (
        "I revise the recommendation: first-token latency must stay under four seconds. "
        "Because the opening token is now the deadline, the Face lobe owns the immediate "
        "foreground response and states the best supported diagnosis, safe action, and "
        "important uncertainty without waiting for exhaustive analysis. The Depth lobe "
        "runs asynchronously outside the first-token path and examines logs, traces, "
        "routing state, resource pressure, and cross-node evidence. Its verified result "
        "will be delivered to the operator automatically when ready, but it does not "
        "influence the immediate four-second response window. This sequence changes the "
        "critical path: conversational value is synchronous, while evidence-intensive "
        "reasoning stays non-blocking. Resource isolation keeps the larger analysis from "
        "competing with Face for memory bandwidth or compute during the opening response. "
        "The operator therefore receives a prompt, useful answer while the architecture "
        "retains a deeper analytical role. If later evidence contradicts the opening "
        "diagnosis, the system should label the correction, preserve the measurements, "
        "and explain why the conclusion changed. That makes responsiveness and diagnostic "
        "quality independently testable instead of hiding either one behind a model name."
    )
    assert len(candidate.split()) >= 130
    return candidate


def _first_token_revision_without_stale_verified_delivery() -> str:
    return _first_token_revision_with_stale_verified_delivery().replace(
        "Its verified result will be delivered to the operator automatically when ready, "
        "but it does not influence the immediate four-second response window.",
        "Depth performs that later analysis as a stable asynchronous role outside the "
        "immediate four-second response window.",
    )


def _r2_depth_answer() -> str:
    return (
        "[Verified Depth Lobe result; "
        "job_id=da-4410e950-e757-4a3c-842c-5b89d4ab4e8d]\n\n"
        "### Tradeoffs\n\n"
        "| Dimension | 4B Face Lobe | 35B Depth Lobe |\n"
        "|---|---|---|\n"
        "| **Latency per reasoning step** | ~200-500ms | ~1-3s |\n"
        "| **Token cost** | ~0.1x per call | ~3-5x per call |\n"
        "| **Overfitting risk** | Low | Moderate |"
    )


def test_standalone_bold_tradeoffs_heading_activates_qwen_ordinal_guard(monkeypatch):
    depth_answer = (
        "[Verified Depth Lobe result; job_id=da-bold-heading]\n\n"
        "**Tradeoffs**\n\n"
        "- **Latency**: Face begins sooner while Depth takes longer.\n"
        "- **Diagnostic Fidelity**: Depth preserves competing evidence and uncertainty.\n"
        "- **Resource Cost**: Depth uses more compute, memory, and tokens.\n\n"
        "**Recommendation**\n\n"
        "- Keep Face responsive and deliver verified Depth analysis later."
    )
    wrong_item = (
        "Latency, token delay, response timing, and first audio speed dominate this "
        "tradeoff because the opening response must remain quick. "
    ) * 24
    attempts = iter((wrong_item, wrong_item))
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("bold-heading-ordinal", "qwen3.6:35b")
    state.messages.append({"role": "assistant", "content": depth_answer})
    before = list(state.messages)

    def post_streaming(_payload, callback, *, cancel_event=None):
        del cancel_event
        answer = next(attempts)
        callback(answer)
        return answer, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        "Go deeper on the second tradeoff.",
        session_id=state.session_id,
        model="qwen3.6:35b",
        stream_callback=spoken.append,
    )

    semantic = result["output_guard"]["substantive_followup"]
    assert semantic["ordinal_expectation"]["label"] == "Diagnostic Fidelity"
    assert "ordinal_item_not_substantive" in semantic["first_candidate_failures"]
    assert "ordinal_item_not_substantive" in semantic["corrective_candidate_failures"]
    assert result["completed"] is False
    assert result["text"] == ""
    assert spoken == []
    assert state.messages == before


@pytest.mark.parametrize(
    "line",
    (
        "The **Tradeoffs** are latency and diagnostic fidelity.",
        "Ordinary prose with **Tradeoffs** emphasized inline.",
        "**Tradeoffs** are important, but this is still prose.",
    ),
)
def test_inline_bold_tradeoffs_prose_is_not_a_section_heading(line):
    text = f"{line}\n- Latency\n- Diagnostic fidelity"
    assert _markdown_section_items(text, "tradeoff") is None


@pytest.mark.parametrize(
    "heading",
    (
        "** Tradeoffs **",
        "***Tradeoffs**",
        "**Tradeoffs***",
        "***Tradeoffs***",
        "**Tradeoffs** and **Recommendations**",
        "**Tradeoffs** **Recommendations**",
        "**Trade**offs**",
        "**Tradeoffs*",
        "*Tradeoffs**",
        "**Tradeoffs",
    ),
)
def test_malformed_or_multi_span_bold_tradeoffs_are_not_headings(heading):
    text = f"{heading}\n- Latency\n- Diagnostic fidelity"
    assert _markdown_section_items(text, "tradeoff") is None


@pytest.mark.parametrize(
    "heading",
    (
        r"**Tradeoffs\**",
        r"**Tradeoffs\\\**",
    ),
)
def test_escaped_bold_closer_is_not_a_section_heading(heading):
    assert _markdown_section_heading(heading) is None
    text = f"{heading}\n- Latency\n- Diagnostic fidelity"
    assert _markdown_section_items(text, "tradeoff") is None


def test_even_backslash_run_preserves_valid_bold_closer():
    assert _markdown_section_heading(r"**Tradeoffs\\**") == r"Tradeoffs\\"


def test_escaped_bold_closer_cannot_drive_wrong_ordinal_expectation():
    history = [
        {
            "role": "assistant",
            "content": (
                "[Verified Depth Lobe result; job_id=da-escaped-heading]\n\n"
                r"**Tradeoffs\**"
                "\n- **Latency**: Face begins sooner.\n"
                "- **Diagnostic Fidelity**: Depth preserves competing evidence.\n"
                "- **Resource Cost**: Depth uses more compute."
            ),
        }
    ]
    assert _ordinal_expectation("Go deeper on the second tradeoff.", history) is None


@pytest.mark.parametrize(
    "malformed_boundary",
    (
        "**Recommendation*",
        "**Recommendation",
        "***Recommendation**",
        "**Recommendation** and **Next steps**",
        r"**Recommendation\**",
    ),
)
def test_malformed_bold_next_heading_bounds_prior_tradeoffs(malformed_boundary):
    text = (
        "**Tradeoffs**\n"
        "- Latency\n"
        "- Diagnostic fidelity\n"
        f"{malformed_boundary}\n"
        "- Keep Face responsive\n"
        "- Deliver Depth later"
    )
    assert _markdown_section_items(text, "tradeoff") == (
        "Tradeoffs",
        ["Latency", "Diagnostic fidelity"],
    )


def test_r2_second_tradeoff_is_resolved_to_token_cost_before_inference(fake_hivemind):
    answer = (
        "Token cost is the requested row. The fast Face uses a small fraction of "
        "the compute for each conversational turn, while the deeper model spends "
        "more tokens preserving alternatives, evidence, and uncertainty. That extra "
        "cost is justified for diagnosis and reconciliation, but not for every "
        "acknowledgement. The useful split is to answer the bounded question with "
        "Face immediately and reserve Depth for work whose accuracy benefit exceeds "
        "its additional generation and scheduling cost. The boundary should also "
        "be observable: record which lobe answered, whether a deeper pass was "
        "requested, and whether that pass changed the recommendation. That makes "
        "the cost measurable instead of merely theoretical. If repeated deeper "
        "passes rarely alter a class of answer, the router can keep that class on "
        "Face. If they routinely uncover missed constraints or conflicting evidence, "
        "the class belongs on Depth despite the expense. This feedback loop keeps "
        "the fast path honest while directing the larger token budget to requests "
        "where it demonstrably improves the operator's decision. For example, if "
        "one incident class changes diagnosis only after Depth reads cross-node logs, "
        "route that class to Depth and verify the correction rate against GPU cost."
    )
    fake_hivemind["script"]["per_model"] = {
        "nemotron-3-nano:4b": {"content": answer},
    }
    face = FaceLobeChat(hivemind_url=fake_hivemind["url"])
    state = face.get_or_create_session("r2-ordinal", "nemotron-3-nano:4b")
    state.messages.append({"role": "assistant", "content": _r2_depth_answer()})

    result = face.chat(
        "Go deeper on the second tradeoff.",
        session_id="r2-ordinal",
        model="nemotron-3-nano:4b",
    )

    assert result["completed"] is True
    system = fake_hivemind["log"][-1]["messages"][0]["content"]
    assert "MS4 resolved ordinal reference" in system
    assert "item 2" in system
    assert "Token cost" in system
    assert "one explicit, worked concrete example" in system
    locator = system.split("[MS4 resolved ordinal reference", 1)[1]
    assert "Overfitting risk" not in locator


def _r6_depth_answer() -> str:
    return (
        "[Verified Depth Lobe result; job_id=da-r6]\n\n"
        "### Tradeoffs\n\n"
        "| Dimension | Fast Face | Deep model |\n"
        "|---|---|---|\n"
        "| **Latency** | Immediate | Slower |\n"
        "| **Diagnostic Fidelity** | Bounded synthesis | Full evidence reconciliation |\n"
        "| **Resource Cost** | Low | Higher |\n"
        "| **Operational Risk** | Can miss nuance | Can delay delivery |"
    )


def _r23_diagnostic_fidelity_answer(*, include_example: bool) -> str:
    answer = (
        "Diagnostic Fidelity is the second tradeoff because a fast synthesis can "
        "preserve the leading diagnosis while dropping uncertainty, competing causes, "
        "or the observation that would falsify it. Face should state its best supported "
        "hypothesis, confidence, and the next discriminating check, while Depth retains "
        "the full chain from evidence through alternatives and verification. That split "
        "keeps the conversation responsive without treating the short answer as equal to "
        "a complete diagnostic reconciliation. The important mechanism is evidence "
        "compression: weak signals and contradictions are easier to omit when many logs, "
        "measurements, and state transitions must fit into one immediate answer. Depth can "
        "audit those sources, compare hypotheses, and return a correction whose cause is "
        "visible to the operator. The practical consequence is that consequential actions "
        "should remain provisional until the evidence needed to distinguish failure modes "
        "has been checked. This makes diagnostic quality measurable through correction "
        "rates, reproduced causes, and the number of recommendations that survive review."
    )
    if include_example:
        answer += (
            " For example, when the Face model labels a worker timeout as a network "
            "failure, the Depth lobe can inspect the trace, compare cache events, and "
            "verify the corrected cause before the operator reroutes that worker."
        )
    return answer


@pytest.mark.parametrize(
    ("item", "expected"),
    [
        (
            "   * ** Resource Consumption ** : Depth needs more compute.",
            "Resource Consumption",
        ),
        ("Resource Consumption: Depth needs more compute.", "Resource Consumption"),
        (
            "Dimension: **Diagnostic Fidelity**; Fast Face: bounded; Deep: full",
            "Diagnostic Fidelity",
        ),
        (
            "**Context Window Pressure:** Extended reasoning consumes tokens; "
            "without summary compression, early evidence can be dropped.",
            "Context Window Pressure",
        ),
    ],
    ids=(
        "leading-bold-bullet",
        "plain-label-detail",
        "rendered-table-row",
        "bold-colon-semicolon-detail",
    ),
)
def test_r7_ordinal_item_label_preserves_source_convention(item, expected):
    assert _ordinal_item_label(item) == expected


def test_r7_resource_consumption_bullet_resolves_and_passes_semantic_gate(monkeypatch):
    depth_answer = (
        "[Verified Depth Lobe result; job_id=da-r5]\n\n"
        "### Tradeoffs\n\n"
        "* **Latency**: Face responds sooner while Depth takes longer.\n"
        "* **Resource Consumption**: Depth needs more compute, GPU memory, and tokens.\n"
        "* **Operational Risk**: More moving parts require stronger recovery."
    )
    answer = (
        "Resource consumption is the second tradeoff because the deeper model "
        "uses more compute, GPU memory, token generation, and scheduler capacity "
        "for each answer. That additional usage is worthwhile when full evidence "
        "reconciliation can change a consequential decision, but wasteful for a "
        "simple acknowledgement. The router should therefore reserve Depth for "
        "diagnosis, cross-source comparison, and work whose expected accuracy gain "
        "exceeds its resource cost. Face can answer bounded conversational questions "
        "immediately and dispatch a deeper check only when uncertainty or scope "
        "requires it. Operators should measure GPU occupancy, memory pressure, token "
        "volume, queue delay, and whether the deeper result actually changed the "
        "recommendation. Those observations turn resource consumption into a tunable "
        "policy rather than a vague objection. If deeper passes rarely alter one "
        "request class, keep it on Face; if they repeatedly uncover missed evidence, "
        "accept the higher compute and memory budget for that class. For example, "
        "measure one recurring incident class on both lobes, then keep Depth only "
        "when its corrected diagnoses justify the observed GPU and queue cost."
    )
    assert len(answer.split()) >= 130
    calls = 0
    payloads: list[dict[str, object]] = []
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("r7-resource-bullet", "qwen3.6:35b")
    state.messages.append({"role": "assistant", "content": depth_answer})

    def post_streaming(payload, callback, *, cancel_event=None):
        nonlocal calls
        del cancel_event
        calls += 1
        payloads.append(payload)
        callback(answer)
        return answer, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        "Go deeper on the second tradeoff.",
        session_id="r7-resource-bullet",
        model="qwen3.6:35b",
        stream_callback=spoken.append,
    )

    assert calls == 1
    assert result["completed"] is True
    assert result["text"] == answer
    assert spoken == [answer]
    expectation = result["output_guard"]["substantive_followup"][
        "ordinal_expectation"
    ]
    assert expectation["label"] == "Resource Consumption"
    assert expectation["item"].startswith("**Resource Consumption**:")
    system = payloads[0]["messages"][0]["content"]
    assert "item 2" in system
    assert "Resource Consumption" in system
    assert result["output_guard"]["substantive_followup"]["applied"] is False


def test_r22_bold_colon_context_window_tradeoff_passes_semantic_gate(monkeypatch):
    depth_answer = (
        "[Verified Depth Lobe result; job_id=da-r22]\n\n"
        "### Tradeoffs\n\n"
        "- **Latency & Cost:** Depth takes longer and uses more compute.\n"
        "- **Context Window Pressure:** Extended reasoning consumes tokens; "
        "without sliding windows or summary compression, early evidence can still be dropped.\n"
        "- **Over-Diagnosis Risk:** Elaborate trees can exceed the available evidence.\n"
        "- **Deployment Footprint:** Larger models require more serving capacity."
    )
    answer = (
        "Context window pressure is the second tradeoff, and it describes how an "
        "extended diagnostic can consume the finite token window while it accumulates "
        "logs, hypotheses, measurements, and corrections. The danger is not merely a "
        "long answer. If early evidence is dropped before the final synthesis, the model "
        "can lose the event that distinguishes a root cause from a later symptom. Summary "
        "compression helps, but an unfaithful summary can erase uncertainty or merge two "
        "competing explanations. The practical response is to preserve source-linked "
        "checkpoints, keep a compact ledger of confirmed and falsified hypotheses, and "
        "refresh summaries before the active context reaches its limit. For example, a "
        "twelve-minute trace may begin with a cache lease change and end with a timeout. "
        "A sliding window that retains only the timeout invites the wrong network diagnosis. "
        "Retaining the lease event, its timestamp, and the evidence that connects it to the "
        "failure keeps the reasoning auditable. Depth should therefore spend its context "
        "budget on discriminating facts and durable intermediate summaries, not repeated "
        "prose, so the final recommendation reflects the whole diagnostic sequence."
    )
    assert len(answer.split()) >= 130
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("r22-context-window", "ministral-3:latest_ollama")
    state.messages.append({"role": "assistant", "content": depth_answer})
    calls = 0

    def post_streaming(_payload, callback, *, cancel_event=None):
        nonlocal calls
        del cancel_event
        calls += 1
        callback(answer)
        return answer, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        "Go deeper on the second tradeoff.",
        session_id="r22-context-window",
        model="ministral-3:latest_ollama",
        stream_callback=spoken.append,
    )

    expectation = result["output_guard"]["substantive_followup"][
        "ordinal_expectation"
    ]
    assert expectation["label"] == "Context Window Pressure"
    assert expectation["item"].startswith("**Context Window Pressure:**")
    assert result["completed"] is True
    assert result["text"] == answer
    assert spoken == [answer]
    assert calls == 1
    assert result["output_guard"]["substantive_followup"]["applied"] is False


def _completed_stream_stats() -> dict[str, object]:
    return {
        "stream_completed": True,
        "terminal_source": "done_marker",
        "stream_chunks": 1,
        "bytes_received": 100,
        "http_latency_ms": 5,
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
    }


@pytest.mark.parametrize(
    ("message", "intent"),
    [
        ("Go deeper on the second tradeoff.", "explicit_expansion"),
        ("What evidence would change your mind?", "evidence_reconsideration"),
        (
            "Assume first-token latency must stay under four seconds. "
            "Revise the recommendation.",
            "contextual_revision",
        ),
        (
            "No - I mean end-to-end voice latency. Correct the recommendation.",
            "contextual_revision",
        ),
        (
            "Give me the final architecture in plain English, and tell me what "
            "constraint I changed.",
            "contextual_synthesis",
        ),
        ("Which measurements would falsify that conclusion?", "evidence_reconsideration"),
        ("Put everything together into an overall design.", "contextual_synthesis"),
        (
            "Quickly, give me the final architecture in full detail.",
            "contextual_synthesis",
        ),
        ("Tell me more about distributed inference latency.", "explicit_expansion"),
        ("Revise the architecture to keep Depth asynchronous.", "contextual_revision"),
        ("Revise the answer to 42 in full detail.", "contextual_revision"),
    ],
)
def test_r13_substantive_contextual_followup_intent_positives(message, intent):
    assert _substantive_followup_intent(message) == intent


@pytest.mark.parametrize(
    "message",
    [
        "What was the second tradeoff?",
        "Are you still there?",
        "What is the current GPU count?",
        "Correct the typo in the heading.",
        "Correct answer: 42.",
        "Revise the answer to 42.",
        "Change the volume to 50 percent.",
        "Summarize this in one sentence.",
        "Tell me more about your favorite color.",
    ],
)
def test_r13_substantive_contextual_followup_intent_fast_path_negatives(message):
    assert _substantive_followup_intent(message) is None


@pytest.mark.parametrize(
    ("message", "reply"),
    [
        ("Revise the answer to 42.", "42."),
        (
            "Tell me more about your favorite color.",
            "I don't have personal favorites, but neon green fits the HiveMind palette nicely.",
        ),
    ],
)
def test_r14_bounded_correction_and_personal_small_talk_keep_natural_fast_path(
    monkeypatch,
    message,
    reply,
):
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("r14-fast-path", "face-model")
    state.messages.append(
        {"role": "assistant", "content": "A detailed prior technical answer is present."}
    )
    calls = 0

    def post_streaming(_payload, callback, *, cancel_event=None):
        nonlocal calls
        del cancel_event
        calls += 1
        callback(reply)
        return reply, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []

    result = face.chat(
        message,
        session_id="r14-fast-path",
        model="face-model",
        stream_callback=spoken.append,
    )

    assert calls == 1
    assert result["completed"] is True
    assert result["text"] == reply
    assert spoken == [reply]
    assert "substantive_followup" not in result["output_guard"]


def _r13_semantic_answer(kind: str, *, final_authoritative: bool = False) -> str:
    if kind == "evidence":
        core = (
            "Evidence would change the conclusion if independent traces, measurements, "
            "or controlled tests falsified the causal link. I would compare repeated "
            "failure timing, node-level logs, and a benchmark with the suspected feature "
            "disabled. If the failure persisted at the same rate, that result would weaken "
            "the diagnosis and shift the recommendation toward the competing explanation. "
        )
    elif kind == "first_token_revision":
        core = (
            "The revised recommendation keeps first-token latency under four seconds "
            "because the Face lobe must begin a useful response before the deadline. "
            "The sequence now routes immediate conversational framing through Face while "
            "Depth continues in the background and later supplies verified analysis. "
        )
    else:
        if final_authoritative:
            core = (
                "The final architecture measures from the end of the operator's speech to the "
                "first useful audible response. ASR input finalization fans out concurrently: "
                "Face generation feeds first-chunk TTS synthesis while Depth begins asynchronous "
                "verification. Full-utterance completion is tracked separately because a "
                "substantive spoken answer cannot finish inside the response-start deadline. "
                "The corrected constraint changed from first-token latency to end-to-end voice "
                "latency, and Depth returns verified analysis automatically. "
            )
        else:
            core = (
                "The final architecture measures from the end of the operator's speech to the "
                "first useful audible response. ASR input finalization fans out concurrently: "
                "Face generation feeds first-chunk TTS synthesis while Depth begins asynchronous "
                "verification. Full-utterance completion is tracked separately because a "
                "substantive spoken answer cannot finish inside the response-start deadline. "
                "The corrected constraint changed from first-token latency under four seconds "
                "to end-to-end voice latency under four seconds, and Depth returns verified "
                "analysis automatically. "
            )
    answer = core * 6
    if final_authoritative:
        answer += (
            "The active end-to-end voice-latency deadline is under four seconds."
        )
    assert len(answer.split()) >= 130
    return answer


def _assert_gateway_authoritative_latency_synthesis(
    result: dict,
    spoken: list[str] | None,
) -> str:
    text = result["text"]
    receipt = result["output_guard"]["authoritative_latency_synthesis"]
    semantic = result["output_guard"]["substantive_followup"]
    assert result["completed"] is True
    assert result["api_calls"] == 0
    assert result["output_guard"]["reason"] == "authoritative_latency_synthesis"
    assert receipt["schema"] == "Ms4AuthoritativeLatencySynthesis.v1"
    assert receipt["applied"] is True
    assert receipt["source"] == "resolved_conversation_contract"
    assert receipt["model_output_used"] is False
    assert receipt["deadline"] == "under four seconds"
    assert receipt["deadline_milliseconds"] == 4000
    assert 180 <= receipt["word_count"] <= 600
    assert "architecture" in receipt["requested_targets"]
    assert text.startswith(
        "1. Direct answer\nFace speaks first; Depth verifies in parallel. "
    )
    assert "2. Why and mechanism\n" in text
    assert "3. Practical consequence\n" in text
    assert text.count(
        "The active end-to-end voice-latency deadline is under four seconds."
    ) == 1
    assert (
        "The constraint changed from first-token latency to end-to-end voice latency. "
        "The active end-to-end voice-latency deadline is under four seconds."
        in text
    )
    assert "first useful audible response" in text
    assert "Full-utterance completion is tracked separately" in text
    assert "fans it out concurrently" in text
    assert semantic["first_candidate_failures"] == []
    assert semantic["corrective_candidate_failures"] == []
    assert semantic["corrective_regeneration_attempted"] is False
    assert semantic["fail_closed"] is False
    if spoken is not None:
        assert spoken == [text]
    return text


@pytest.mark.parametrize(
    ("message", "history", "corrected_kind", "intent", "wrong", "expected_failure"),
    [
        (
            "What evidence would change your mind?",
            [{"role": "assistant", "content": "The leading diagnosis is cache desynchronization."}],
            "evidence",
            "evidence_reconsideration",
            (
                "The logs and traces would change my mind if those logs contradicted the "
                "current logs, so I would inspect the same logging stream again. "
            )
            * 16,
            "insufficient_evidence_categories",
        ),
        (
            "Assume first-token latency must stay under four seconds. Revise the recommendation.",
            [{"role": "assistant", "content": "Use Depth synchronously for every diagnosis."}],
            "first_token_revision",
            "contextual_revision",
            (
                "Revised recommendation: use Face exclusively because first-token latency "
                "must stay under four seconds. This Face-only choice is the revised answer. "
            )
            * 14,
            "latency_lobe_staging_missing",
        ),
        (
            (
                "No - I mean end-to-end voice latency under four seconds. "
                "Correct the recommendation."
            ),
            [
                {
                    "role": "user",
                    "content": "Assume first-token latency must stay under four seconds.",
                },
                {"role": "assistant", "content": "Use a fast opening from Face."},
            ],
            "full_voice",
            "contextual_revision",
            None,
            "latency_lobe_staging_missing",
        ),
        (
            "Give me the final architecture in plain English, and tell me what constraint I changed.",
            [
                {"role": "user", "content": "First-token latency must stay under four seconds."},
                {"role": "assistant", "content": "Use a fast opening from Face."},
                {"role": "user", "content": "No, I mean end-to-end voice latency."},
                {"role": "assistant", "content": "The delivery sequence must change."},
            ],
            "full_voice",
            "contextual_synthesis",
            None,
            "requested_synthesis_target_missing",
        ),
        (
            "Quickly, give me the final architecture in full detail.",
            [
                {"role": "user", "content": "First-token latency must stay under four seconds."},
                {"role": "assistant", "content": "Use a fast opening from Face."},
                {"role": "user", "content": "No, I mean end-to-end voice latency."},
                {"role": "assistant", "content": "The delivery sequence must change."},
            ],
            "full_voice",
            "contextual_synthesis",
            None,
            "requested_synthesis_target_missing",
        ),
    ],
)
def test_r13_long_semantically_wrong_followups_regenerate_before_stream_and_history(
    monkeypatch,
    message,
    history,
    corrected_kind,
    intent,
    wrong,
    expected_failure,
):
    wrong = wrong or (
        "This response contains extensive discussion and many general considerations "
        "about the topic for the operator. "
    ) * 18
    assert len(wrong.split()) >= 130
    corrected = _r13_semantic_answer(
        corrected_kind,
        final_authoritative=(
            corrected_kind == "full_voice" and intent == "contextual_synthesis"
        ),
    )
    attempts = iter((wrong, corrected))
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session(f"r13-{corrected_kind}-{intent}", "face-model")
    state.messages.extend(history)
    before = list(state.messages)

    def post_streaming(_payload, callback, *, cancel_event=None):
        del cancel_event
        candidate = next(attempts)
        callback(candidate)
        return candidate, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []

    result = face.chat(
        message,
        session_id=state.session_id,
        model="face-model",
        stream_callback=spoken.append,
    )

    semantic = result["output_guard"]["substantive_followup"]
    assert semantic["intent"] == intent
    authoritative_final = bool(
        corrected_kind == "full_voice"
        and intent == "contextual_synthesis"
        and "tell me what constraint i changed" in message.lower()
    )
    if authoritative_final:
        _assert_gateway_authoritative_latency_synthesis(result, spoken)
        assert result["text"] != corrected.strip()
    else:
        assert result["completed"] is True
        assert result["api_calls"] == 2
        assert expected_failure in semantic["first_candidate_failures"]
        assert semantic["corrective_candidate_failures"] == []
        assert spoken == [result["text"]]
        assert result["text"] == corrected.strip()
    assert wrong not in " ".join(str(item) for item in state.messages)
    assert state.messages == before + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": result["text"]},
    ]


def _r25_vague_but_measured_evidence_answer() -> str:
    answer = (
        "Here is what would meaningfully shift my position. First, a controlled "
        "accuracy benchmark should compare root-cause correctness and false-positive "
        "rates across repeated unseen incidents. Second, latency measurements should "
        "capture p95 time to a verified diagnosis rather than only the first sentence. "
        "Third, GPU memory, compute utilization, and operating cost should be measured "
        "under the same workload. Fourth, logs, traces, and telemetry must show whether "
        "the smaller lobe preserves causal order across scheduler and worker events. "
        "Fifth, a preregistered threshold should require statistically repeatable results "
        "across enough samples to rule out a curated benchmark effect. A useful test would "
        "disable the suspected cache feature, replay the incident, and compare both lobes "
        "against blinded labels. The evidence should include accuracy, diagnosis latency, "
        "resource cost, observability quality, and confidence intervals. Production traces "
        "should then confirm that any laboratory gain survives real concurrency and node "
        "variation. Those measurements would reveal whether the smaller model's speed is "
        "a genuine end-to-end advantage or merely a faster path to a wrong explanation."
    )
    assert len(answer.split()) >= 130
    return answer


def test_r25_measured_evidence_gets_one_explicit_decision_rule_before_tts_history(
    monkeypatch,
):
    vague = _r25_vague_but_measured_evidence_answer()
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("r25-evidence-rule", "face-model")
    state.messages.append(
        {"role": "assistant", "content": "The current recommendation prefers Depth."}
    )
    before = list(state.messages)
    payloads: list[dict[str, object]] = []

    def post_streaming(payload, callback, *, cancel_event=None):
        del cancel_event
        payloads.append(payload)
        callback(vague)
        return vague, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        "What evidence would change your mind?",
        session_id=state.session_id,
        model="face-model",
        stream_callback=spoken.append,
    )

    exact_rule = (
        "Decision rule: I would change the recommendation if repeated evidence "
        "across the measurements above falsifies the current conclusion or supports "
        "the competing explanation; otherwise I would keep it."
    )
    semantic = result["output_guard"]["substantive_followup"]
    receipt = result["output_guard"]["evidence_decision_rule_postcondition"]
    assert result["completed"] is True
    assert result["api_calls"] == 1


    assert result["text"].endswith(exact_rule)
    assert result["text"].count(exact_rule) == 1
    assert spoken == [result["text"]]
    assert state.messages == before + [
        {"role": "user", "content": "What evidence would change your mind?"},
        {"role": "assistant", "content": result["text"]},
    ]
    assert semantic["first_candidate_failures"] == [
        "missing_explicit_decision_rule"
    ]
    assert semantic["corrective_regeneration_attempted"] is False
    assert receipt["applied"] is True
    assert "explicit evidence-to-decision rule" in str(
        payloads[0]["messages"][0]["content"]
    )


@pytest.mark.parametrize(
    "unrelated_rule",
    [
        "I would change the logging level if telemetry drops below the alert threshold.",
        "I would change the recommendation if the office lights turn blue.",
    ],
)
def test_r26_unrelated_change_is_not_accepted_as_the_decision_rule(
    monkeypatch,
    unrelated_rule,
):
    raw = (
        _r25_vague_but_measured_evidence_answer()
        + f" {unrelated_rule}"
    )
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("r26-unrelated-change", "face-model")
    state.messages.append(
        {"role": "assistant", "content": "The current recommendation prefers Depth."}
    )

    def post_streaming(_payload, callback, *, cancel_event=None):
        del cancel_event
        callback(raw)
        return raw, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        "What evidence would change your mind?",
        session_id=state.session_id,
        model="face-model",
        stream_callback=spoken.append,
    )

    semantic = result["output_guard"]["substantive_followup"]
    receipt = result["output_guard"]["evidence_decision_rule_postcondition"]
    assert result["completed"] is True
    assert result["api_calls"] == 1
    assert semantic["first_candidate_failures"] == [
        "missing_explicit_decision_rule"
    ]
    assert receipt["applied"] is True
    assert result["text"] != raw
    assert "I would change the recommendation if repeated evidence" in result["text"]
    assert spoken == [result["text"]]
    assert state.messages[-1] == {"role": "assistant", "content": result["text"]}


def test_r25_complete_evidence_rule_is_unchanged_and_idempotent(monkeypatch):
    complete = _r13_semantic_answer("evidence").strip()
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("r25-evidence-complete", "face-model")
    state.messages.append(
        {"role": "assistant", "content": "The current recommendation prefers Depth."}
    )
    calls = 0

    def post_streaming(_payload, callback, *, cancel_event=None):
        nonlocal calls
        del cancel_event
        calls += 1
        callback(complete)
        return complete, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        "What evidence would change your mind?",
        session_id=state.session_id,
        model="face-model",
        stream_callback=spoken.append,
    )

    assert calls == 1
    assert result["completed"] is True
    assert result["text"] == complete
    assert spoken == [complete]
    receipt = result["output_guard"]["evidence_decision_rule_postcondition"]
    assert receipt["applied"] is False
    assert result["output_guard"]["substantive_followup"][
        "first_candidate_failures"
    ] == []


@pytest.mark.parametrize(
    ("candidate", "expected_failure"),
    [
        (
            "Accuracy, latency, GPU memory, logs, and a threshold would matter, but "
            "here is only a short list.",
            "underlength",
        ),
        (
            (
                "Here is what would meaningfully shift my position: more logs and traces "
                "plus repeated telemetry measurements from the same incident stream. "
            )
            * 14,
            "insufficient_evidence_categories",
        ),
        (
            (
                "No evidence would change this recommendation. Accuracy benchmarks, "
                "latency measurements, GPU memory and compute cost, logs and traces, "
                "and statistical thresholds can all be collected, but the conclusion "
                "is unfalsifiable and can never be disproved. "
            )
            * 8,
            "negated_or_unfalsifiable_decision_rule",
        ),
    ],
)
def test_r25_unsafe_or_incomplete_evidence_cannot_be_deterministically_repaired(
    monkeypatch,
    candidate,
    expected_failure,
):
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session(f"r25-no-repair-{uuid.uuid4()}", "face-model")
    state.messages.append(
        {"role": "assistant", "content": "The current recommendation prefers Depth."}
    )
    before = list(state.messages)
    calls = 0

    def post_streaming(_payload, callback, *, cancel_event=None):
        nonlocal calls
        del cancel_event
        calls += 1
        callback(candidate)
        return candidate, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        "What evidence would change your mind?",
        session_id=state.session_id,
        model="face-model",
        stream_callback=spoken.append,
    )

    semantic = result["output_guard"]["substantive_followup"]
    assert calls == 2
    assert result["completed"] is False
    assert result["text"] == ""
    assert expected_failure in semantic["first_candidate_failures"]
    assert expected_failure in semantic["corrective_candidate_failures"]
    assert semantic["first_evidence_decision_rule_postcondition"]["applied"] is False
    assert spoken == []
    assert state.messages == before


def test_r25_corrective_decision_rule_append_stays_under_hard_word_cap(monkeypatch):
    first = "I can explain the evidence if you want me to continue."
    corrected = "\n\n".join([_r25_vague_but_measured_evidence_answer()] * 3)
    assert 340 < len(corrected.split()) < 575
    attempts = iter((first, corrected))
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("r25-corrective-cap", "face-model")
    state.messages.append(
        {"role": "assistant", "content": "The current recommendation prefers Depth."}
    )

    def post_streaming(_payload, callback, *, cancel_event=None):
        del cancel_event
        candidate = next(attempts)
        callback(candidate)
        return candidate, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        "What evidence would change your mind?",
        session_id=state.session_id,
        model="face-model",
        stream_callback=spoken.append,
    )

    semantic = result["output_guard"]["substantive_followup"]
    receipt = semantic["corrective_evidence_decision_rule_postcondition"]
    assert result["completed"] is True
    assert result["api_calls"] == 2
    assert len(result["text"].split()) <= 600
    assert receipt["applied"] is True
    assert receipt["resulting_word_count"] <= 600
    assert semantic["corrective_candidate_failures"] == []
    assert spoken == [result["text"]]


def test_r6_referential_teaser_is_corrected_before_stream_tts_and_history(monkeypatch):
    teaser = "Looking into that — I'll surface the detailed breakdown on the next turn."
    corrected = (
        "Diagnostic fidelity is the second tradeoff, and it matters because the fast "
        "Face compresses a large evidence set into an immediate conversational answer. "
        "That compression can hide weak signals, competing causes, or uncertainty that "
        "would change the diagnosis. The deeper model can retain the full chain from "
        "observations through hypotheses, tests, and disconfirming evidence, so its "
        "conclusion is easier to audit. The practical design is to let Face state the "
        "best current answer and its confidence now, while Depth independently checks "
        "the evidence and returns any correction. This preserves responsiveness without "
        "pretending the quick synthesis has the same diagnostic resolution as a complete "
        "reconciliation. For example, two symptoms can support the same first-pass "
        "diagnosis while differing in the one observation that rules it out. Face "
        "should name the missing check instead of smoothing over that uncertainty. "
        "Depth can then compare the alternatives against logs, measurements, and "
        "prior results, explicitly noting what confirms or falsifies each one. The "
        "operator gets a useful answer immediately, plus a clear reason to trust or "
        "revise it when the deeper evidence arrives. That is the fidelity tradeoff: "
        "not simply more words, but preservation of discriminating evidence and an "
        "auditable path from that evidence to the recommendation."
    )
    assert len(corrected.split()) >= 130
    attempts = iter((teaser, corrected))
    payloads: list[dict[str, object]] = []

    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("r6-exact", "qwen3.6:35b")
    state.messages.append({"role": "assistant", "content": _r6_depth_answer()})
    before = list(state.messages)

    def post_streaming(payload, callback, *, cancel_event=None):
        del cancel_event
        payloads.append(payload)
        text = next(attempts)
        callback(text)
        return text, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []

    result = face.chat(
        "Go deeper on the second tradeoff.",
        session_id="r6-exact",
        model="qwen3.6:35b",
        stream_callback=spoken.append,
    )

    assert result["completed"] is True
    assert result["api_calls"] == 2
    assert result["text"] == corrected
    assert spoken == [corrected]
    assert teaser not in " ".join(str(item) for item in state.messages)
    assert state.messages == before + [
        {"role": "user", "content": "Go deeper on the second tradeoff."},
        {"role": "assistant", "content": corrected},
    ]
    assert len(payloads) == 2
    assert "max_tokens" not in payloads[0]
    assert payloads[1]["max_tokens"] == 1024
    assert payloads[0]["options"] == {"num_ctx": 8192}
    assert payloads[1]["options"] == payloads[0]["options"]
    corrective_system = payloads[1]["messages"][0]["content"]
    assert "bounded attempt 1 of 2" in corrective_system
    assert "item 2" in corrective_system
    assert "Diagnostic Fidelity" in corrective_system
    assert "before this response ends" in corrective_system
    assert "between 220 and 380 visible words" in corrective_system
    assert "1. Direct answer — at least 60 relevant words" in corrective_system
    assert "2. Why and mechanism — at least 90 relevant words" in corrective_system
    assert "3. Practical consequence or concrete example — at least 70 relevant words" in corrective_system
    assert "do not stop before the third" in corrective_system
    assert "every listed failure is resolved" in corrective_system
    assert payloads[0]["temperature"] == 0.2
    assert payloads[1]["temperature"] == 0.0
    assert payloads[1]["messages"][1] == before[0]
    guard = result["output_guard"]["substantive_followup"]
    assert guard["first_candidate_failures"] == [
        "generic_defer",
        "underlength",
        "ordinal_item_not_substantive",
        "ordinal_concrete_example_missing",
    ]
    assert guard["corrective_regeneration_attempted"] is True
    assert guard["corrective_candidate_failures"] == []
    assert guard["fail_closed"] is False


def test_voice_word_contract_corrects_q11_followup_with_aligned_retry(monkeypatch):
    first = "I can go deeper on that tradeoff if you want me to continue."
    corrected = _r23_diagnostic_fidelity_answer(include_example=True)
    assert 180 <= len(corrected.split()) <= 220
    attempts = iter((first, corrected))
    payloads: list[dict[str, object]] = []

    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("q11-voice-corrected", "qwen3.6:35b")
    state.messages.append({"role": "assistant", "content": _r6_depth_answer()})
    before = list(state.messages)

    def post_streaming(payload, callback, *, cancel_event=None):
        del cancel_event
        payloads.append(payload)
        text = next(attempts)
        callback(text)
        return text, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        "Go deeper on the second trade-off.",
        session_id=state.session_id,
        model="qwen3.6:35b",
        stream_callback=spoken.append,
        extra_system=_VOICE_BREVITY_DIRECTIVE,
        substantive_word_range=(150, 380),
    )

    assert result["completed"] is True
    assert result["api_calls"] == 2
    assert result["text"] == corrected
    assert spoken == [corrected]
    assert state.messages == before + [
        {"role": "user", "content": "Go deeper on the second trade-off."},
        {"role": "assistant", "content": corrected},
    ]
    assert len(payloads) == 2
    assert all(
        "180 to 240 visible words" in payload["messages"][0]["content"]
        for payload in payloads
    )
    corrective_system = payloads[1]["messages"][0]["content"]
    assert "between 220 and 380 visible words" in corrective_system
    assert "do not exceed 380 visible words" in corrective_system
    guard = result["output_guard"]["substantive_followup"]
    assert guard["min_words"] == 150
    assert guard["max_words"] == 380
    assert guard["word_range_source"] == "voice_mode"
    assert "underlength" in guard["first_candidate_failures"]
    assert guard["first_candidate_word_count"] < 150
    assert 180 <= guard["corrective_candidate_word_count"] <= 380
    assert guard["corrective_candidate_failures"] == []
    assert guard["fail_closed"] is False


def test_voice_word_contract_rejects_overlength_both_attempts(monkeypatch):
    base = _r23_diagnostic_fidelity_answer(include_example=True)
    overlong = f"{base} {' '.join(['additional'] * 200)}"
    assert len(overlong.split()) > 380
    attempts = iter((overlong, overlong))

    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("q11-voice-overlength", "qwen3.6:35b")
    state.messages.append({"role": "assistant", "content": _r6_depth_answer()})
    before = list(state.messages)

    def post_streaming(_payload, callback, *, cancel_event=None):
        del cancel_event
        text = next(attempts)
        callback(text)
        return text, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        "Go deeper on the second trade-off.",
        session_id=state.session_id,
        model="qwen3.6:35b",
        stream_callback=spoken.append,
        extra_system=_VOICE_BREVITY_DIRECTIVE,
        substantive_word_range=(150, 380),
    )

    guard = result["output_guard"]["substantive_followup"]
    assert result["completed"] is False
    assert result["text"] == ""
    assert "overlength" in guard["first_candidate_failures"]
    assert "overlength" in guard["corrective_candidate_failures"]
    assert guard["first_candidate_word_count"] > 380
    assert guard["corrective_candidate_word_count"] > 380
    assert guard["fail_closed"] is True
    assert spoken == []
    assert state.messages == before


def test_explicit_expansion_requires_example_without_ordinal_table(monkeypatch):
    first = _r23_diagnostic_fidelity_answer(include_example=False)
    corrected = _r23_diagnostic_fidelity_answer(include_example=True)
    assert 150 <= len(first.split()) <= 300
    attempts = iter((first, corrected))

    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("q11-generic-expansion", "qwen3.6:35b")
    state.messages.append(
        {
            "role": "assistant",
            "content": "Compute and cost are the key tradeoff in this deployment.",
        }
    )
    before = list(state.messages)

    def post_streaming(_payload, callback, *, cancel_event=None):
        del cancel_event
        text = next(attempts)
        callback(text)
        return text, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        "Go deeper on the compute and cost trade-off.",
        session_id=state.session_id,
        model="qwen3.6:35b",
        stream_callback=spoken.append,
        extra_system=_VOICE_BREVITY_DIRECTIVE,
        substantive_word_range=(150, 380),
    )

    guard = result["output_guard"]["substantive_followup"]
    assert guard["ordinal_expectation"] is None
    assert guard["concrete_example_required"] is True
    assert "concrete_example_missing" in guard["first_candidate_failures"]
    assert guard["corrective_candidate_failures"] == []
    assert result["completed"] is True
    assert result["text"] == corrected
    assert spoken == [corrected]
    assert state.messages == before + [
        {"role": "user", "content": "Go deeper on the compute and cost trade-off."},
        {"role": "assistant", "content": corrected},
    ]


def test_r6_second_non_substantive_candidate_fails_closed_without_delivery(monkeypatch):
    attempts = iter(
        (
            "Looking into that — I'll explain it next turn.",
            "I can go deeper if you would like me to.",
        )
    )
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("r6-closed", "qwen3.6:35b")
    state.messages.append({"role": "assistant", "content": _r6_depth_answer()})
    before = list(state.messages)

    def post_streaming(_payload, callback, *, cancel_event=None):
        del cancel_event
        text = next(attempts)
        callback(text)
        return text, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []

    result = face.chat(
        "Go deeper on the second tradeoff.",
        session_id="r6-closed",
        model="qwen3.6:35b",
        stream_callback=spoken.append,
        extra_system=_VOICE_BREVITY_DIRECTIVE,
        substantive_word_range=(150, 380),
    )

    assert result["completed"] is False
    assert result["text"] == ""
    assert result["api_calls"] == 2
    assert result["metrics"]["incomplete_reason"] == "non_substantive_referential_followup"
    assert spoken == []
    assert state.messages == before
    voice_guard = result["output_guard"]["substantive_followup"]
    assert voice_guard["min_words"] == 150
    assert voice_guard["max_words"] == 380
    assert "underlength" in voice_guard["first_candidate_failures"]
    assert "underlength" in voice_guard["corrective_candidate_failures"]
    assert voice_guard["fail_closed"] is True


@pytest.mark.parametrize(
    ("message", "reply"),
    [
        ("What was the second tradeoff?", "Diagnostic fidelity."),
        ("Are you still there?", "Yes, I'm here."),
    ],
)
def test_r6_short_complete_non_expansion_replies_are_not_false_positives(
    monkeypatch,
    message,
    reply,
):
    calls = 0
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("r6-control", "qwen3.6:35b")
    state.messages.append({"role": "assistant", "content": _r6_depth_answer()})

    def post_streaming(_payload, callback, *, cancel_event=None):
        nonlocal calls
        del cancel_event
        calls += 1
        callback(reply)
        return reply, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        message,
        session_id="r6-control",
        model="qwen3.6:35b",
        stream_callback=spoken.append,
        substantive_word_range=(150, 380),
    )

    assert calls == 1
    assert result["completed"] is True
    assert result["text"] == reply
    assert spoken == [reply]
    assert "substantive_followup" not in result["output_guard"]


def test_r6_current_turn_dispatch_acknowledgement_is_exempt(monkeypatch):
    acknowledgement = (
        "Dispatched Depth Lobe job da-1234 now. The verified result will appear "
        "automatically in this conversation when it is ready; you do not need to ask again."
    )
    calls = 0
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("r6-dispatch", "qwen3.6:35b")
    state.messages.append({"role": "assistant", "content": _r6_depth_answer()})

    def post_streaming(_payload, callback, *, cancel_event=None):
        nonlocal calls
        del cancel_event
        calls += 1
        callback(acknowledgement)
        return acknowledgement, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        "Go deeper on the second tradeoff.",
        session_id="r6-dispatch",
        model="qwen3.6:35b",
        extra_system=(
            "MS4 Face Lobe context (authoritative):\n"
            "- THIS TURN DISPATCHED job da-1234 to the Depth Lobe."
        ),
        stream_callback=spoken.append,
    )

    assert calls == 1
    assert result["completed"] is True
    assert result["text"] == acknowledgement
    assert spoken == [acknowledgement]
    assert "substantive_followup" not in result["output_guard"]
    ack_guard = result["output_guard"]["current_dispatch_ack_postcondition"]
    assert ack_guard["applied"] is False
    assert ack_guard["authoritative_identity_present"] is True
    assert ack_guard["mentioned_job_ids"] == ["da-1234"]
    assert ack_guard["negated_dispatch_present"] is False
    assert ack_guard["strict_canonical_match"] is True


def test_r8_stale_current_dispatch_teaser_is_rewritten_before_delivery(monkeypatch):
    stale = "Looking into that — I'll surface the result on the next turn."
    canonical = (
        "Dispatched Depth Lobe job da-1234 now. The verified result will appear "
        "automatically in this conversation when it is ready; you do not need to ask again."
    )
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("r8-stale-dispatch", "qwen3.6:35b")
    before = list(state.messages)

    def post_streaming(_payload, callback, *, cancel_event=None):
        del cancel_event
        callback(stale)
        return stale, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        "Run the deeper comparison now.",
        session_id="r8-stale-dispatch",
        model="qwen3.6:35b",
        extra_system=(
            "MS4 Face Lobe context (authoritative):\n"
            "- THIS TURN DISPATCHED job da-1234 to the Depth Lobe."
        ),
        stream_callback=spoken.append,
    )

    assert result["completed"] is True
    assert result["api_calls"] == 1
    assert result["text"] == canonical
    assert spoken == [canonical]
    assert state.messages == before + [
        {"role": "user", "content": "Run the deeper comparison now."},
        {"role": "assistant", "content": canonical},
    ]
    assert stale not in " ".join(str(item) for item in state.messages)
    ack_guard = result["output_guard"]["current_dispatch_ack_postcondition"]
    assert ack_guard == {
        "schema": "Ms4CurrentDispatchAckPostcondition.v1",
        "applied": True,
        "dispatched_now_present": False,
        "authoritative_identity_present": False,
        "mentioned_job_ids": [],
        "negated_dispatch_present": False,
        "strict_canonical_match": False,
        "automatic_delivery_present": False,
        "no_followup_needed_present": False,
        "stale_promise_present": True,
    }


@pytest.mark.parametrize(
    ("invalid_ack", "mentioned_ids", "negated"),
    [
        (
            "Not dispatched now (da-1234). The verified result will appear "
            "automatically in this conversation when it is ready; you do not need to ask again.",
            ["da-1234"],
            True,
        ),
        (
            "Dispatched Depth Lobe job da-9999 now. The verified result will appear "
            "automatically in this conversation when it is ready; you do not need to ask again.",
            ["da-9999"],
            False,
        ),
        (
            "Dispatched now (da-9999). The verified result will appear automatically "
            "in this conversation when it is ready; you do not need to ask again.",
            ["da-9999"],
            False,
        ),
        (
            "Dispatched Depth Lobe job da-1234 now (not da-9999). The verified result "
            "will appear automatically in this conversation when it is ready; you do not "
            "need to ask again.",
            ["da-1234", "da-9999"],
            False,
        ),
    ],
    ids=("negated", "wrong-bare-id", "wrong-parenthetical-id", "mixed-identities"),
)
def test_r9_dispatch_identity_errors_are_rewritten_before_tts_and_history(
    monkeypatch,
    invalid_ack,
    mentioned_ids,
    negated,
):
    canonical = (
        "Dispatched Depth Lobe job da-1234 now. The verified result will appear "
        "automatically in this conversation when it is ready; you do not need to ask again."
    )
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    session_id = f"r9-identity-{mentioned_ids[-1]}-{int(negated)}"
    state = face.get_or_create_session(session_id, "qwen3.6:35b")
    before = list(state.messages)

    def post_streaming(_payload, callback, *, cancel_event=None):
        del cancel_event
        callback(invalid_ack)
        return invalid_ack, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    tts_boundary: list[str] = []
    result = face.chat(
        "Run the deeper comparison now.",
        session_id=session_id,
        model="qwen3.6:35b",
        extra_system=(
            "MS4 Face Lobe context (authoritative):\n"
            "- THIS TURN DISPATCHED job da-1234 to the Depth Lobe."
        ),
        stream_callback=tts_boundary.append,
    )

    assert result["completed"] is True
    assert result["text"] == canonical
    assert tts_boundary == [canonical]
    assert state.messages == before + [
        {"role": "user", "content": "Run the deeper comparison now."},
        {"role": "assistant", "content": canonical},
    ]
    assert invalid_ack not in " ".join(str(item) for item in state.messages)
    ack_guard = result["output_guard"]["current_dispatch_ack_postcondition"]
    assert ack_guard["applied"] is True
    assert ack_guard["authoritative_identity_present"] is False or negated
    assert ack_guard["mentioned_job_ids"] == mentioned_ids
    assert ack_guard["negated_dispatch_present"] is negated


@pytest.mark.parametrize(
    "invalid_ack",
    [
        (
            "Depth Lobe job da-1234 isn't dispatched now. The verified result will "
            "appear automatically in this conversation when it is ready; you do not "
            "need to ask again."
        ),
        (
            "Dispatched Depth Lobe job da-1234 now. The verified result will not "
            "appear automatically in this conversation when it is ready; you do not "
            "need to ask again."
        ),
        (
            "Dispatched Depth Lobe job da-1234 now. The verified result will appear "
            "automatically in this conversation when it is ready; you do not need to "
            "ask again. The Depth Lobe found that the secret answer is forty-two."
        ),
    ],
    ids=("isnt-dispatched", "will-not-auto-deliver", "canonical-prefix-result-suffix"),
)
def test_r10_exact_dispatch_ack_schema_removes_false_passes_before_tts_history(
    monkeypatch,
    invalid_ack,
):
    canonical = (
        "Dispatched Depth Lobe job da-1234 now. The verified result will appear "
        "automatically in this conversation when it is ready; you do not need to ask again."
    )
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("r10-exact-ack", "qwen3.6:35b")
    before = list(state.messages)
    calls = 0

    def post_streaming(_payload, callback, *, cancel_event=None):
        nonlocal calls
        del cancel_event
        calls += 1
        callback(invalid_ack)
        return invalid_ack, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    tts_boundary: list[str] = []
    result = face.chat(
        "Run the deeper comparison now.",
        session_id="r10-exact-ack",
        model="qwen3.6:35b",
        extra_system=(
            "MS4 Face Lobe context (authoritative):\n"
            "- THIS TURN DISPATCHED job da-1234 to the Depth Lobe."
        ),
        stream_callback=tts_boundary.append,
    )

    assert calls == 1
    assert result["completed"] is True
    assert result["text"] == canonical
    assert tts_boundary == [canonical]
    assert state.messages == before + [
        {"role": "user", "content": "Run the deeper comparison now."},
        {"role": "assistant", "content": canonical},
    ]
    assert invalid_ack not in " ".join(str(item) for item in state.messages)
    ack_guard = result["output_guard"]["current_dispatch_ack_postcondition"]
    assert ack_guard["applied"] is True
    assert ack_guard["strict_canonical_match"] is False


@pytest.mark.parametrize(
    "stale",
    [
        "Looking into that — I'll surface the result on the next turn.",
        "Dispatched, give me a sec.",
        "The Depth Lobe is working. Follow up for the result.",
        "The Depth Lobe is running. Check back when it is ready.",
    ],
)
def test_r8_current_dispatch_stale_ack_variants_are_canonicalized(stale):
    canonical = (
        "Dispatched Depth Lobe job da-1234 now. The verified result will appear "
        "automatically in this conversation when it is ready; you do not need to ask again."
    )
    corrected, postcondition = _enforce_current_dispatch_ack_postcondition(
        stale,
        "- THIS TURN DISPATCHED job da-1234 to the Depth Lobe.",
    )

    assert corrected == canonical
    assert postcondition["applied"] is True
    assert postcondition["stale_promise_present"] is True


def test_r8_stale_teaser_without_current_dispatch_is_unaffected(monkeypatch):
    stale = "Looking into that — I'll surface the result on the next turn."
    face = FaceLobeChat(hivemind_url="http://unused.invalid")

    def post_streaming(_payload, callback, *, cancel_event=None):
        del cancel_event
        callback(stale)
        return stale, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        "Tell me something brief.",
        session_id="r8-no-dispatch",
        model="qwen3.6:35b",
        extra_system=(
            "MS4 Face Lobe context (authoritative):\n"
            "- THIS TURN DID NOT DISPATCH any background work."
        ),
        stream_callback=spoken.append,
    )

    assert result["completed"] is True
    assert result["text"] == stale
    assert spoken == [stale]
    assert "current_dispatch_ack_postcondition" not in result["output_guard"]


def test_r9_dispatch_identity_postcondition_is_inactive_without_dispatch():
    raw = (
        "Not dispatched now (da-9999). The verified result will appear automatically "
        "in this conversation when it is ready; you do not need to ask again."
    )
    corrected, postcondition = _enforce_current_dispatch_ack_postcondition(
        raw,
        "- THIS TURN DID NOT DISPATCH any background work.",
    )

    assert corrected == raw
    assert postcondition is None


def test_r6_substantive_answer_may_end_with_optional_offer(monkeypatch):
    answer = (
        "Diagnostic fidelity is the second tradeoff because a fast synthesis can "
        "compress away uncertainty, competing causes, and weak evidence. A deeper "
        "pass retains the chain from observations to hypotheses, checks, and "
        "disconfirming facts, which makes the conclusion auditable. In practice, "
        "the Face should state the best grounded answer and confidence immediately, "
        "while the deeper worker reconciles the complete evidence set and returns "
        "a correction if needed. That division preserves conversational speed "
        "without representing the quick answer as equivalent to a full diagnostic "
        "review. For example, a concrete failure mode is evidence collision: several observations "
        "fit the leading explanation, but one measurement supports a different cause. "
        "Face should expose that conflict and identify the next discriminating check, "
        "rather than silently averaging it away. Depth can preserve both hypotheses, "
        "trace each claim to its source, and state which new result would falsify one. "
        "That makes the eventual correction intelligible instead of surprising. It "
        "also gives the router feedback about which request classes routinely need "
        "full reconciliation. The purpose of deeper analysis is therefore not extra "
        "length; it is higher resolution, explicit uncertainty, and a reproducible "
        "decision trail. I can go deeper on a specific failure mode if useful."
    )
    assert len(answer.split()) >= 130
    calls = 0
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("r6-offer-control", "qwen3.6:35b")
    state.messages.append({"role": "assistant", "content": _r6_depth_answer()})

    def post_streaming(_payload, callback, *, cancel_event=None):
        nonlocal calls
        del cancel_event
        calls += 1
        callback(answer)
        return answer, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        "Go deeper on the second tradeoff.",
        session_id="r6-offer-control",
        model="qwen3.6:35b",
        stream_callback=spoken.append,
    )

    assert calls == 1
    assert result["completed"] is True
    assert result["text"] == answer
    assert spoken == [answer]
    assert result["output_guard"]["substantive_followup"]["applied"] is False


def test_r6_wrong_ordinal_long_answer_is_rejected_then_corrected(monkeypatch):
    wrong_ordinal = (
        "Diagnostic Fidelity is the named second item. "
        + (
            "Latency dominates this tradeoff because response timing, token delay, "
            "first audio speed, and seconds of waiting determine the experience. "
        )
        * 28
    )
    assert len(wrong_ordinal.split()) >= 300
    correct_ordinal = (
        "Diagnostic fidelity concerns whether the answer preserves the evidence "
        "needed to distinguish competing diagnoses. A fast synthesis may correctly "
        "identify the leading cause while omitting uncertainty, conflicting signals, "
        "or the observation that would falsify it. The deeper pass retains each "
        "hypothesis, traces claims to measurements, and reconciles contradictions, "
        "which makes the recommendation auditable. Face should therefore state the "
        "best supported conclusion and confidence immediately, then name the missing "
        "discriminating check. Depth can verify the full chain and return a correction "
        "when evidence changes the ranking. This is not primarily a timing question: "
        "latency is a neighboring tradeoff. The second tradeoff is the resolution and "
        "trustworthiness of the diagnosis itself, including whether nuance survives "
        "compression and whether the operator can reproduce why one explanation won. "
        "For example, when Face ranks a timeout as a network fault, Depth can compare "
        "the cache and transport traces, expose the conflicting event, and verify which "
        "hypothesis survives before the operator changes the route. "
        "That division preserves a useful conversational answer while reserving full "
        "evidence reconciliation for decisions where a missed distinction would alter "
        "the action."
    )
    assert len(correct_ordinal.split()) >= 130
    attempts = iter((wrong_ordinal, correct_ordinal))
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("r6-wrong-ordinal", "qwen3.6:35b")
    state.messages.append({"role": "assistant", "content": _r6_depth_answer()})

    def post_streaming(_payload, callback, *, cancel_event=None):
        del cancel_event
        text = next(attempts)
        callback(text)
        return text, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        "Go deeper on the second tradeoff.",
        session_id="r6-wrong-ordinal",
        model="qwen3.6:35b",
        stream_callback=spoken.append,
    )

    failures = result["output_guard"]["substantive_followup"][
        "first_candidate_failures"
    ]
    assert "ordinal_item_not_substantive" in failures
    assert "ordinal_item_displaced_by_sibling" in failures
    assert result["completed"] is True
    assert result["api_calls"] == 2
    assert result["text"] == correct_ordinal
    assert spoken == [correct_ordinal]
    assert state.messages[-1] == {"role": "assistant", "content": correct_ordinal}
    assert wrong_ordinal not in " ".join(str(item) for item in state.messages)


def test_r6_correct_ordinal_answer_can_compare_neighbor_without_false_positive(monkeypatch):
    answer = (
        "Diagnostic fidelity is about preserving evidence, uncertainty, and causal "
        "distinctions so the diagnosis can be verified. Latency matters separately, "
        "but a fast response is not high fidelity merely because it arrives quickly. "
        "The Face should expose its best hypothesis, confidence, and the observation "
        "most likely to disconfirm it. The deeper model should reconcile logs and "
        "measurements, retain competing explanations, and explain why one cause wins. "
        "That evidence trail makes corrections auditable and prevents compressed prose "
        "from hiding nuance. A practical system can answer immediately while marking "
        "which claims are provisional, then replace only those claims that fail the "
        "deeper check. For example, if Face blames a worker timeout, Depth can inspect "
        "the trace, identify a cache conflict, and verify the corrected cause before "
        "the operator reroutes that worker. The operator retains conversational speed without losing the "
        "resolution needed for a consequential decision. In other words, the second "
        "tradeoff measures diagnostic trustworthiness: whether relevant evidence "
        "survives synthesis, uncertainty remains visible, and another reviewer can "
        "reproduce the recommendation from the same facts."
    )
    assert len(answer.split()) >= 130
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("r6-correct-ordinal", "qwen3.6:35b")
    state.messages.append({"role": "assistant", "content": _r6_depth_answer()})
    calls = 0

    def post_streaming(_payload, callback, *, cancel_event=None):
        nonlocal calls
        del cancel_event
        calls += 1
        callback(answer)
        return answer, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        "Go deeper on the second tradeoff.",
        session_id="r6-correct-ordinal",
        model="qwen3.6:35b",
        stream_callback=spoken.append,
    )

    assert calls == 1
    assert result["completed"] is True
    assert result["text"] == answer
    assert spoken == [answer]
    assert result["output_guard"]["substantive_followup"]["applied"] is False


def test_r23_concrete_example_detector_rejects_vacuous_marker():
    assert not _has_concrete_operational_example(
        "For example, this illustrates the concept without any operational detail."
    )
    assert not _has_concrete_operational_example(
        "The Face model can see a timeout. The operator should consider the result."
    )
    assert _has_concrete_operational_example(
        "For example, when the Face model reports a worker timeout, the Depth lobe "
        "can inspect the trace and verify the corrected cause before the operator "
        "reroutes that worker."
    )
    assert _has_concrete_operational_example(
        "For instance, when a worker timeout appears, the operator reroutes the job "
        "to the Depth model and verifies the failure cause."
    )
    assert _has_concrete_operational_example(
        "Concrete example: a gateway timeout creates latency pressure, so the operator "
        "uses routing to Depth for verification."
    )
    assert _has_concrete_operational_example(
        "E.g. a worker timeout makes the operator inspect the trace, retry the request, "
        "and verify the cause."
    )


@pytest.mark.parametrize(
    "answer",
    [
        (
            "Example: A 16GB GPU cannot hold the 35B model without memory spill, "
            "so allocate the Depth job to a 24GB worker."
        ),
        (
            "Worked example: On a 16GB GPU, the 35B model causes VRAM pressure and "
            "latency spikes; quantize it to 4-bit and reduce the batch size."
        ),
        (
            "### Example\nA 16GB GPU hits VRAM pressure with the 35B model, so allocate "
            "the job to a larger worker."
        ),
        (
            "**Worked example:** A 16GB GPU hits memory pressure, so quantize the 35B "
            "model and reduce the batch size."
        ),
        (
            "- Suppose a 16GB GPU hits VRAM pressure with the 35B model; quantize it "
            "to 4-bit and reduce the batch size."
        ),
        (
            "Consider a 16GB GPU running the 35B model: memory pressure increases "
            "latency, so allocate the job elsewhere."
        ),
        (
            "For example, a 16GB GPU hits memory pressure, so allocate the job to a "
            "larger worker and verify quality."
        ),
        (
            "For example, when a 16GB GPU reaches VRAM capacity, quantize the model, "
            "reduce the batch size, and benchmark quality."
        ),
    ],
)
def test_r24_concrete_example_detector_accepts_operational_model_loadout_forms(answer):
    assert _has_concrete_operational_example(answer)


@pytest.mark.parametrize(
    "answer",
    [
        "Worked example: compute cost and resolution must be balanced carefully.",
        "For example, the model has memory and cost. The system runs.",
        "Example scenario: the GPU has VRAM. The operator can run the job.",
        (
            "This is not an example: when a GPU hits memory pressure, the operator "
            "reduces batch size."
        ),
        (
            "For example, this is a general note. The GPU has memory. Elsewhere the "
            "operator runs a test."
        ),
        (
            "For example, this concept matters. First filler sentence. Second filler "
            "sentence. Third filler sentence. A GPU hits memory pressure and the "
            "operator reduces resolution."
        ),
        "A 16GB GPU hits memory pressure, so the operator quantizes the model.",
    ],
)
def test_r24_concrete_example_detector_preserves_local_fail_closed_boundary(answer):
    assert not _has_concrete_operational_example(answer)


def test_r32_concrete_example_detector_accepts_domain_neutral_history_example():
    answer = (
        "For example, in June 1914, Archduke Franz Ferdinand was assassinated in "
        "Sarajevo, so Austria-Hungary issued an ultimatum to Serbia. Russia mobilized "
        "after the ultimatum, Germany declared war, and alliance commitments turned "
        "a regional crisis into a continental conflict."
    )

    assert _has_concrete_operational_example(answer)


def test_r32_concrete_example_detector_accepts_singleton_proper_name_anchors():
    answer = (
        "For example, Austria-Hungary issued an ultimatum to Serbia after the "
        "assassination, so Russia mobilized in support. Germany then declared war on "
        "Russia and France, while Britain entered after Belgium was invaded, turning "
        "a regional dispute into a wider conflict."
    )

    assert _has_concrete_operational_example(answer)


@pytest.mark.parametrize(
    "answer",
    [
        (
            "For example, when a child drops a glass beside a hot stove, the parent "
            "moves the child away and explains the danger. The child then uses a "
            "different route through the kitchen because that concrete consequence "
            "changed the family's next action."
        ),
        (
            "For example, when a patient misses a scheduled dose, the clinician calls "
            "the pharmacy and checks the medication record. The clinic then changes "
            "the reminder plan so the patient receives the next dose on time."
        ),
    ],
)
def test_r32_concrete_example_detector_accepts_nontechnical_role_examples(answer):
    assert _has_concrete_operational_example(answer)


def test_r32_concrete_example_detector_rejects_long_causal_fluff():
    assert not _has_concrete_operational_example(
        "For example, this concept matters because the idea belongs in this situation, "
        "so the case should be considered with more detail and the point should remain "
        "part of the note for broader context."
    )


def test_r32_concrete_example_detector_rejects_self_referential_long_form_filler():
    assert not _has_concrete_operational_example(
        "For example, this answer changed because the explanation expanded into more "
        "wording and broad framing, so the response improved and the discussion continued "
        "through additional sentences without naming a concrete actor or event. The answer "
        "then developed more context and the explanation shifted again, which made the "
        "response longer; the discussion continued and the framing expanded while the "
        "wording changed and the content improved in a general way that did not identify "
        "any person, date, measurable effect, decision, or observable consequence. The "
        "response kept developing because the answer expanded and the discussion continued, "
        "so the explanation became broader and the wording changed again; this content "
        "remained abstract, generic, and non-operational while sounding detailed enough to "
        "satisfy a length target without providing a real worked example. The additional "
        "prose repeated the same point with different words, extended the framing, and "
        "increased the apparent detail, but it still supplied no actual scenario, participant, "
        "action, outcome, evidence, or specific consequence for the reader to evaluate."
    )


def test_r32_concrete_example_detector_rejects_sentence_transition_anchors():
    assert not _has_concrete_operational_example(
        "For example, this answer changed because the explanation expanded into broader "
        "wording, so the response improved while the discussion continued through "
        "additional abstract sentences. Moreover, the explanation developed more framing "
        "and the response shifted into longer prose without identifying any actual person, "
        "event, measurement, choice, or observable consequence. Ultimately, the discussion "
        "expanded again because the wording changed, so the content became longer while "
        "remaining abstract and generic. The explanation continued with additional language "
        "about general importance, broader context, conceptual relevance, and apparent "
        "detail, but none of those phrases named a participant or a real action. Overall, "
        "the response repeated the same material using different wording and extended "
        "framing, making the answer sound complete while providing no concrete scenario, "
        "evidence, outcome, decision, or mechanism that a reader could inspect. The final "
        "discussion remained self-referential and non-operational, despite reaching the "
        "requested length and using several causal connectors and event-shaped verbs "
        "throughout the prose."
    )


def test_r32_concrete_example_detector_preserves_three_sentence_locality():
    assert not _has_concrete_operational_example(
        "For example, this concept is abstract because this idea has context. "
        "This scenario gives more general detail. This case has another point. "
        "In June 1914, Archduke Franz Ferdinand was assassinated in Sarajevo, so "
        "Austria-Hungary issued an ultimatum to Serbia and Russia mobilized."
    )


def test_r23_missing_ordinal_example_is_repaired_once_before_delivery(monkeypatch):
    deficient = _r23_diagnostic_fidelity_answer(include_example=False)
    corrected = _r23_diagnostic_fidelity_answer(include_example=True)
    assert len(deficient.split()) >= 130
    attempts = iter((deficient, corrected))
    payloads: list[dict[str, object]] = []
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("r23-example-repair", "ministral-3:latest_ollama")
    state.messages.append({"role": "assistant", "content": _r6_depth_answer()})

    def post_streaming(payload, callback, *, cancel_event=None):
        del cancel_event
        payloads.append(payload)
        text = next(attempts)
        callback(text)
        return text, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        "Go deeper on the second tradeoff.",
        session_id="r23-example-repair",
        model="ministral-3:latest_ollama",
        stream_callback=spoken.append,
    )

    assert result["completed"] is True
    assert result["api_calls"] == 2
    assert result["text"] == corrected
    assert spoken == [corrected]
    semantic = result["output_guard"]["substantive_followup"]
    assert semantic["first_candidate_failures"] == [
        "ordinal_concrete_example_missing"
    ]
    assert semantic["corrective_candidate_failures"] == []
    assert semantic["fail_closed"] is False
    assert "one explicit, worked concrete example" in payloads[0]["messages"][0]["content"]
    assert "component or actor" in payloads[1]["messages"][0]["content"]


def test_r23_missing_ordinal_example_twice_fails_closed(monkeypatch):
    deficient = _r23_diagnostic_fidelity_answer(include_example=False)
    attempts = iter((deficient, deficient))
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("r23-example-closed", "ministral-3:latest_ollama")
    state.messages.append({"role": "assistant", "content": _r6_depth_answer()})
    before = list(state.messages)

    def post_streaming(_payload, callback, *, cancel_event=None):
        del cancel_event
        text = next(attempts)
        callback(text)
        return text, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        "Go deeper on the second tradeoff.",
        session_id="r23-example-closed",
        model="ministral-3:latest_ollama",
        stream_callback=spoken.append,
    )

    assert result["completed"] is False
    assert result["api_calls"] == 2
    assert result["text"] == ""
    assert spoken == []
    assert state.messages == before
    semantic = result["output_guard"]["substantive_followup"]
    assert semantic["first_candidate_failures"] == [
        "ordinal_concrete_example_missing"
    ]
    assert semantic["corrective_candidate_failures"] == [
        "ordinal_concrete_example_missing"
    ]
    assert semantic["first_concrete_example_evidence"]["matched"] is False
    assert semantic["corrective_concrete_example_evidence"]["matched"] is False
    assert set(semantic["first_concrete_example_evidence"]) == {
        "matched",
        "cue",
        "actor",
        "effect",
        "action",
        "causal_link",
        "best_window_index",
        "window_count",
    }
    assert semantic["fail_closed"] is True


def test_r6_output_guard_replacement_is_revalidated_and_never_delivered(monkeypatch):
    candidate = (
        "The Depth Lobe found that Diagnostic Fidelity is the second tradeoff. "
        + (
            "Diagnostic evidence, uncertainty, hypothesis resolution, audit detail, "
            "and reconciliation all support a trustworthy diagnosis. "
        )
        * 32
    )
    assert len(candidate.split()) >= 300
    attempts = iter((candidate, candidate))
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("r6-post-guard", "qwen3.6:35b")
    state.messages.append({"role": "assistant", "content": _r6_depth_answer()})
    before = list(state.messages)

    def post_streaming(_payload, callback, *, cancel_event=None):
        del cancel_event
        text = next(attempts)
        callback(text)
        return text, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        "Go deeper on the second tradeoff.",
        session_id="r6-post-guard",
        model="qwen3.6:35b",
        stream_callback=spoken.append,
    )

    assert result["completed"] is False
    assert result["text"] == ""
    assert result["api_calls"] == 2
    assert spoken == []
    assert state.messages == before
    semantic = result["output_guard"]["substantive_followup"]
    assert "underlength" in semantic["first_candidate_failures"]
    assert "underlength" in semantic["corrective_candidate_failures"]
    assert semantic["fail_closed"] is True


def test_r13_verified_completion_removes_old_dispatch_across_stream_chunks_before_validation(
    fake_hivemind,
):
    stale = (
        "Dispatched Depth Lobe job da-4410e950-e757-4a3c-842c-5b89d4ab4e8d now. "
        "The verified result will appear automatically in this conversation when it "
        "is ready; you do not need to ask again.\n\n"
    )
    substantive = (
        "The final architecture keeps the 4B Face lobe in the interactive voice path "
        "so it can acknowledge the request, preserve conversational continuity, and "
        "start useful speech without waiting for the heavier analysis. The 35B Depth "
        "lobe runs the evidence-intensive diagnosis outside that immediate path, then "
        "returns its verified conclusion into the same session. Once that conclusion "
        "exists, Face explains it completely rather than promising another turn. The "
        "voice layer renders the accepted text while the written answer remains visible "
        "for inspection and follow-up. This separates responsiveness from analytical "
        "quality without confusing either one for the other. The changed constraint is "
        "the measurement boundary: the earlier requirement covered first-token latency, "
        "whereas the corrected requirement covers end-to-end voice latency from the end "
        "of the operator's speech through ASR, Face generation, TTS synthesis, and audible "
        "playback. That broader boundary means synchronous Depth work cannot block the "
        "spoken turn. Face must provide a bounded immediate response, Depth must complete "
        "asynchronously, and the verified result must be integrated into later answers "
        "without stale dispatch language or a request for permission to continue."
    )
    assert len(substantive.split()) >= 130
    split_at = stale.index("appear") + 3
    fake_hivemind["script"]["per_model"] = {
        "nemotron-3-nano:4b": {
            "stream": True,
            "chunks": [stale[:split_at], stale[split_at:] + substantive],
            "send_done": True,
        },
    }
    face = FaceLobeChat(hivemind_url=fake_hivemind["url"])
    spoken: list[str] = []

    result = face.chat(
        "Give me the final architecture in plain English, and tell me what constraint I changed.",
        session_id="r2-completed-status",
        model="nemotron-3-nano:4b",
        extra_system=_r2_verified_depth_context(),
        stream_callback=spoken.append,
    )

    assert result["completed"] is True
    assert result["api_calls"] == 1
    assert "dispatched depth lobe" not in result["text"].lower()
    assert "will appear" not in result["text"].lower()
    assert "already complete" not in result["text"].lower()
    assert result["text"] == substantive
    assert spoken == [result["text"]]
    assert result["output_guard"]["substantive_followup"]["intent"] == "contextual_synthesis"
    assert result["output_guard"]["substantive_followup"]["first_candidate_failures"] == []
    assert result["output_guard"]["reason"] == "verified_depth_already_complete"


def test_r34_completed_depth_future_delivery_is_regenerated_before_tts_and_history(
    monkeypatch,
):
    stale = _first_token_revision_with_stale_verified_delivery()
    corrected = _first_token_revision_without_stale_verified_delivery()
    attempts = iter((stale, corrected))
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("r34-stale-delivery", "face-model")
    state.messages.append(
        {
            "role": "assistant",
            "content": "The verified Depth analysis favors staged Face and Depth roles.",
        }
    )
    before = list(state.messages)
    calls = 0
    payloads: list[dict] = []

    def post_streaming(payload, callback, *, cancel_event=None):
        nonlocal calls
        del cancel_event
        calls += 1
        payloads.append(payload)
        answer = next(attempts)
        callback(answer)
        return answer, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        "Assume first-token latency must stay under four seconds. Revise the recommendation.",
        session_id=state.session_id,
        model="face-model",
        extra_system=_r2_verified_depth_context(),
        stream_callback=spoken.append,
    )

    semantic = result["output_guard"]["substantive_followup"]
    corrective_system = str(payloads[1]["messages"][0]["content"])
    assert calls == 2
    assert result["completed"] is True
    assert result["text"] == corrected
    assert stale not in result["text"]
    assert spoken == [corrected]
    assert state.messages == before + [
        {
            "role": "user",
            "content": (
                "Assume first-token latency must stay under four seconds. "
                "Revise the recommendation."
            ),
        },
        {"role": "assistant", "content": corrected},
    ]
    assert semantic["first_candidate_failures"] == [
        "stale_verified_depth_future_delivery"
    ]
    assert semantic["corrective_candidate_failures"] == []
    assert semantic["fail_closed"] is False
    assert "already contains the completed Depth result" in corrective_system
    assert "be delivered when ready" in corrective_system


def test_r58_completed_depth_active_and_unconditional_delivery_are_regenerated(
    monkeypatch,
):
    corrected = _first_token_revision_without_stale_verified_delivery()
    stale = corrected + (
        " The completed Depth job is still active; its result will be delivered "
        "automatically."
    )
    attempts = iter((stale, corrected))
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("r58-stale-active-delivery", "face-model")
    state.messages.append(
        {
            "role": "assistant",
            "content": "The verified Depth analysis favors staged Face and Depth roles.",
        }
    )
    before = list(state.messages)
    payloads: list[dict] = []

    def post_streaming(payload, callback, *, cancel_event=None):
        del cancel_event
        payloads.append(payload)
        answer = next(attempts)
        callback(answer)
        return answer, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        "Assume first-token latency must stay under four seconds. Revise the recommendation.",
        session_id=state.session_id,
        model="face-model",
        extra_system=_r2_verified_depth_context(),
        stream_callback=spoken.append,
    )

    semantic = result["output_guard"]["substantive_followup"]
    corrective_system = str(payloads[1]["messages"][0]["content"])
    assert len(payloads) == 2
    assert result["completed"] is True
    assert result["text"] == corrected
    assert spoken == [corrected]
    assert stale not in [message["content"] for message in state.messages]
    assert state.messages == before + [
        {
            "role": "user",
            "content": (
                "Assume first-token latency must stay under four seconds. "
                "Revise the recommendation."
            ),
        },
        {"role": "assistant", "content": corrected},
    ]
    assert semantic["first_candidate_failures"] == [
        "stale_verified_depth_future_delivery"
    ]
    assert semantic["corrective_candidate_failures"] == []
    assert semantic["fail_closed"] is False
    assert "Do not call that completed job active" in corrective_system


def test_r34_repeated_completed_depth_future_delivery_fails_closed(
    monkeypatch,
):
    stale = _first_token_revision_with_stale_verified_delivery()
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("r34-stale-delivery-closed", "face-model")
    state.messages.append(
        {
            "role": "assistant",
            "content": "The verified Depth analysis favors staged Face and Depth roles.",
        }
    )
    before = list(state.messages)
    calls = 0

    def post_streaming(_payload, callback, *, cancel_event=None):
        nonlocal calls
        del cancel_event
        calls += 1
        callback(stale)
        return stale, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        "Assume first-token latency must stay under four seconds. Revise the recommendation.",
        session_id=state.session_id,
        model="face-model",
        extra_system=_r2_verified_depth_context(),
        stream_callback=spoken.append,
    )

    semantic = result["output_guard"]["substantive_followup"]
    assert calls == 2
    assert result["completed"] is False
    assert result["text"] == ""
    assert spoken == []
    assert state.messages == before
    assert "stale_verified_depth_future_delivery" in semantic[
        "first_candidate_failures"
    ]
    assert "stale_verified_depth_future_delivery" in semantic[
        "corrective_candidate_failures"
    ]
    assert semantic["fail_closed"] is True


def test_r34_future_delivery_gate_requires_verified_completed_depth_context():
    contract = {
        "intent": "explicit_expansion",
        "min_words": 1,
        "verified_depth_grounded": False,
        "latency_sequence_required": False,
        "ordinal_concrete_example_required": False,
    }

    failures = _substantive_followup_failures(
        "A future benchmark result will be delivered to the operator when ready.",
        contract,
    )

    assert "stale_verified_depth_future_delivery" not in failures
    assert failures == []


@pytest.mark.parametrize(
    "claim",
    (
        "The completed Depth job is still active; its result will be delivered "
        "automatically.",
        "The Depth result will be delivered automatically.",
        "The completed Depth job is complete. Its result will be delivered "
        "automatically.",
        "The Depth lobe's verified result is delivered to the operator when ready.",
        "The Depth analysis will be posted after processing finishes.",
        "The Depth analysis will be returned when processing is complete.",
        "The Depth findings will be shared after processing finishes.",
        "You will receive the Depth analysis once processing is complete.",
        "The operator receives the Depth analysis once processing is complete.",
        "You will automatically receive the Depth analysis once processing is complete.",
        "The operator will definitely receive the Depth analysis once processing is "
        "complete.",
        "You will eventually see the Depth analysis once processing is complete.",
        "The 35B\u2019s verified result is delivered to the operator when ready.",
        "The 35B Depth lobe runs asynchronously. Its result will be delivered "
        "automatically when ready.",
        "A 35B Depth lobe runs asynchronously. Its result will appear when ready.",
        "This Depth lobe runs asynchronously. Its result will appear when ready.",
        "Our Depth model runs asynchronously. Its result will appear when ready.",
        "The background Depth lobe analyzes the performance benchmark. Its result will "
        "be delivered automatically when ready.",
        "The asynchronous Depth lobe runs the benchmark. Its result will be delivered "
        "when ready.",
        "The local 35B Depth lobe analyzes a benchmark. Its result will appear when "
        "ready.",
        "Our dedicated Depth lobe evaluates benchmarks. Its result will appear when "
        "ready.",
        "The 35B Depth lobe will analyze the benchmark and telemetry. Its result will "
        "be delivered automatically when ready.",
        "The Depth lobe is running a simulation. Its result will be delivered "
        "automatically when ready.",
        "The Depth lobe will inspect the test logs. Its verified result will be "
        "delivered automatically when ready.",
        "Run the 35B Depth lobe against the benchmark. Its result will be delivered "
        "automatically when ready.",
        "Dispatch a 35B Depth lobe to analyze the benchmark. Its result will be "
        "delivered automatically when ready.",
        "Run a Depth model against the benchmark. Its result will appear when ready.",
        "Route our Depth lobe through the cluster. Its result will appear when ready.",
        "Dispatch the local 35B Depth lobe against the benchmark. Its result will "
        "appear when ready.",
        "The Depth result will not affect the Face deadline, but it will be delivered "
        "when ready.",
        "The Depth result should not block Face and will be delivered when ready.",
        "The Depth result will not be posted separately, but it will be delivered here "
        "when ready.",
        "The Depth result will not appear in the sidebar, but it will be delivered here "
        "when ready.",
        "The Depth result will be posted or delivered when ready.",
        "The Depth result will automatically be delivered when ready.",
        "The Depth result will definitely be delivered when ready.",
        "The Depth result will be automatically delivered when ready.",
        "The Depth result will certainly appear when ready.",
        "The Depth result will still be delivered when ready.",
        "The Depth result is going to be delivered when ready.",
        "If Face answers first, the Depth result will be delivered when ready.",
        "Assuming Face stays synchronous, the Depth result will be delivered when ready.",
    ),
)
def test_r34_completed_depth_future_delivery_grammar_is_fail_closed(claim):
    contract = {
        "intent": "explicit_expansion",
        "min_words": 1,
        "verified_depth_no_active_jobs_grounded": True,
        "latency_sequence_required": False,
        "ordinal_concrete_example_required": False,
    }

    failures = _substantive_followup_failures(claim, contract)

    assert failures == ["stale_verified_depth_future_delivery"]


@pytest.mark.parametrize(
    "claim",
    (
        "Do not say that the completed Depth job is still active or its result will "
        "be delivered automatically.",
        "The completed Depth job is not active; its result is already present.",
        "If a future Depth job is dispatched, its result will be delivered "
        "automatically.",
        "The benchmark result will appear when ready.",
        "The Depth lobe remains asynchronous in the architecture. A future benchmark "
        "result will be delivered to the operator when ready.",
        "Depth preserves responsiveness in the foreground. The benchmark result will "
        "appear when ready.",
        "Depth remains asynchronous, while the benchmark result will appear when ready.",
        "The Depth result is complete. A telemetry result from tomorrow's benchmark "
        "will be delivered when ready.",
        "If a future Depth job is dispatched, its result will be delivered when ready.",
        "The Depth result would be delivered when ready if a future job were dispatched.",
        "The Depth result, if a future job is dispatched, will be delivered when ready.",
        "Depth completed earlier. A benchmark is running separately. Its verified result "
        "will appear when ready.",
        "Depth completed earlier. A benchmark is running separately. Its result will "
        "appear when ready.",
        "The benchmark exercises Depth routing. Its result will appear when ready.",
        "Depth completed earlier, while a benchmark is running now. Its result will "
        "appear when ready.",
        "The Depth lobe completed earlier and a benchmark runs now. Its result will "
        "appear when ready.",
        "The Depth lobe completed earlier; a benchmark runs now. Its result will appear "
        "when ready.",
        "The benchmark exercises Depth routing. Its verified result will appear when "
        "ready.",
        "A benchmark runs the Depth model. Its result will appear when ready.",
        "This benchmark runs the Depth model. Its result will appear when ready.",
        "Our benchmark invokes the Depth lobe. Its result will appear when ready.",
        "A test launches the Depth model. Its result will appear when ready.",
        "These tests launch the Depth model. The result will appear when ready.",
        "Those benchmarks run the Depth model. Their result will appear when ready.",
        "Our probes invoke the Depth lobe. The result will appear when ready.",
        "A regression test launches the Depth model. Its result will appear when ready.",
        "These regression tests launch the Depth model. The result will appear when "
        "ready.",
        "The performance benchmark runs the Depth model. Its result will appear when "
        "ready.",
        "Run the benchmark against the 35B Depth lobe. Its result will appear when ready.",
        "Route the benchmark through the Depth lobe. Its result will appear when ready.",
        "Depth completed earlier. A benchmark is running separately. The verified result "
        "will appear when ready.",
        'The runbook says, "The Depth result will appear when ready."',
        'The old answer said, "The Depth result will be delivered when ready."',
        "Do not say that the Depth result will be delivered when ready.",
        "The Depth result will not appear when ready because it is already present.",
        "The Depth result is not delivered when ready; it is already here.",
        "The Depth result cannot be delivered when ready because it already exists.",
        "The Depth result is no longer delivered when ready; it is already present.",
        "The Depth result will absolutely not be delivered when ready because it is "
        "already present.",
        "The Depth result isn't going to be delivered when ready; it is already here.",
        "The Depth result is not going to be delivered when ready; it is already here.",
        "The Depth result is never going to be delivered when ready; it is already here.",
        "The Depth result is already available and will not be delivered when ready "
        "because it is present.",
        "The Depth result will not be posted or delivered when ready because it is "
        "already present.",
        "The Depth result won't be posted or delivered when ready because it is already "
        "present.",
        "The Depth result will be posted separately, but will not be delivered when ready "
        "because it is already present.",
        "You will definitely not receive the Depth analysis once processing is complete "
        "because it is already present.",
        "No Depth result will be delivered when ready because the result is already "
        "present.",
    ),
)
def test_r34_completed_depth_future_delivery_gate_preserves_nonassertive_prose(claim):
    contract = {
        "intent": "explicit_expansion",
        "min_words": 1,
        "verified_depth_no_active_jobs_grounded": True,
        "latency_sequence_required": False,
        "ordinal_concrete_example_required": False,
    }

    failures = _substantive_followup_failures(claim, contract)

    assert failures == []


def test_r34_completed_a_with_active_b_preserves_truthful_future_delivery():
    contract = {
        "intent": "explicit_expansion",
        "min_words": 1,
        "verified_depth_grounded": True,
        "verified_depth_no_active_jobs_grounded": False,
        "latency_sequence_required": False,
        "ordinal_concrete_example_required": False,
    }
    claim = (
        "Depth job da-bbbbbbbb-2222-4222-8222-bbbbbbbbbbbb is running. Its verified "
        "result will be delivered to the operator when ready."
    )

    failures = _substantive_followup_failures(claim, contract)

    assert failures == []


def test_r23_only_missing_voice_sequence_is_completed_before_tts_and_history(monkeypatch):
    candidate = (
        "The constraint changed from first-token latency to end-to-end voice latency. "
        "The final "
        "architecture keeps the Face lobe first in the interactive path "
        "while the Depth lobe runs asynchronously in the background. Face gives the "
        "immediate useful answer, and Depth returns the complete diagnosis into the "
        "same conversation when its evidence pass finishes. This sequence is necessary "
        "because the difficult reasoning must not delay the initial interaction, yet "
        "the operator still needs the deeper result without asking for another turn. "
        "The constraint changed in that precise direction. The first measure only marks when generation "
        "begins; first-token latency is not the same as end-to-end voice latency. The "
        "second governs when the useful voice exchange is complete and the "
        "conversation is released. Therefore Face owns the bounded immediate response, "
        "Depth owns the non-blocking diagnostic work, and the written channel preserves "
        "the complete evidence-backed explanation. That division keeps the architecture "
        "responsive, preserves diagnostic quality, and makes the changed constraint "
        "visible instead of silently treating the original timing target as unchanged. "
        "The active end-to-end voice-latency deadline is under four seconds."
    )
    assert len(candidate.split()) >= 130
    canonical = "The active end-to-end voice-latency deadline is under four seconds."
    definition = (
        "For this end-to-end constraint, the measured voice path runs from ASR input "
        "through Face generation and TTS synthesis to audible playback."
    )
    corrected = (
        f"{candidate.removesuffix(canonical).rstrip()}\n\n{definition}\n\n{canonical}"
    )
    attempts = iter((candidate, corrected))
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("r23-voice-sequence", "ministral-3:latest_ollama")
    state.messages.extend(
        [
            {
                "role": "user",
                "content": (
                    "Assume first-token latency must stay under four seconds. "
                    "Revise the recommendation."
                ),
            },
            {"role": "assistant", "content": "Face should answer first."},
            {
                "role": "user",
                "content": (
                    "No - I mean end-to-end voice latency. Correct the recommendation."
                ),
            },
            {"role": "assistant", "content": "The corrected voice path is staged."},
        ]
    )
    calls = 0

    def post_streaming(_payload, callback, *, cancel_event=None):
        nonlocal calls
        del cancel_event
        calls += 1
        answer = next(attempts)
        callback(answer)
        return answer, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        "Give me the final architecture in plain English, and tell me what constraint I changed.",
        session_id="r23-voice-sequence",
        model="ministral-3:latest_ollama",
        extra_system=_r2_verified_depth_context(),
        stream_callback=spoken.append,
    )

    assert calls == 0
    _assert_gateway_authoritative_latency_synthesis(result, spoken)
    assert candidate not in result["text"]
    assert corrected not in result["text"]
    assert state.messages[-1] == {"role": "assistant", "content": result["text"]}
    assert "latency_delivery_sequence_postcondition" not in result["output_guard"]


def test_only_missing_latency_lobe_staging_is_completed_before_delivery(monkeypatch):
    candidate = (
        "Revised recommendation: first-token latency must stay under four seconds. "
        "That changed constraint means the interactive path must begin a useful answer "
        "inside the bound instead of waiting for the entire diagnosis. The opening should "
        "state the best current recommendation, name the uncertainty, and preserve the "
        "operator's exact timing requirement. A separate analytical service can continue "
        "asynchronously, compare the intermittent failure against logs and traces, and "
        "return the verified diagnosis later in the same conversation. This revision is "
        "necessary because first-token latency measures the start of useful delivery, not "
        "the completion time of every reasoning step. It therefore changes the critical "
        "path: immediate conversational value is synchronous, while evidence-intensive "
        "work is non-blocking. The practical result is a responsive opening without "
        "discarding diagnostic quality. If the later evidence contradicts the opening, the "
        "system should revise the recommendation explicitly, retain the measured facts, "
        "and explain why the conclusion changed rather than hiding the correction. This "
        "keeps the deadline, reasoning boundary, and eventual evidence visible to the user."
    )
    assert len(candidate.split()) >= 130
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("latency-staging", "ministral-3:latest_ollama")
    state.messages.append(
        {
            "role": "assistant",
            "content": "Use a large analyzer for the complete diagnosis and report its evidence.",
        }
    )
    calls = 0

    def post_streaming(_payload, callback, *, cancel_event=None):
        nonlocal calls
        del cancel_event
        calls += 1
        callback(candidate)
        return candidate, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        "Assume first-token latency must stay under four seconds. Revise the recommendation.",
        session_id="latency-staging",
        model="ministral-3:latest_ollama",
        extra_system=_r2_verified_depth_context(),
        stream_callback=spoken.append,
    )

    staging = (
        "The lobe staging is explicit: Face gives the immediate foreground response, "
        "while Depth runs asynchronously in the background and delivers its verified "
        "analysis later."
    )
    semantic = result["output_guard"]["substantive_followup"]
    receipt = result["output_guard"]["latency_lobe_staging_postcondition"]
    assert calls == 1
    assert result["completed"] is True
    assert result["text"] == f"{candidate}\n\n{staging}"
    assert spoken == [result["text"]]
    assert state.messages[-1] == {"role": "assistant", "content": result["text"]}
    assert semantic["first_candidate_failures"] == ["latency_lobe_staging_missing"]
    assert semantic["corrective_regeneration_attempted"] is False
    assert semantic["fail_closed"] is False
    assert receipt["applied"] is True


def _revision_candidate_without_performative() -> str:
    candidate = (
        "First-token latency must stay under four seconds. The Face lobe gives an "
        "immediate foreground response while the Depth lobe runs asynchronously in "
        "the background and delivers verified analysis later. This arrangement is "
        "necessary because the operator needs a useful answer inside the timing bound "
        "without discarding the deeper diagnosis. The foreground answer states the best "
        "current recommendation and its uncertainty. The background analysis compares "
        "logs, traces, scheduler state, and cross-node evidence, then returns verified "
        "findings in the same conversation. The practical sequence keeps the spoken "
        "path responsive and the evidence path accountable. If the deeper result "
        "disagrees, Face explains that difference directly and preserves the measured "
        "facts so the operator can audit why the conclusion moved. "
    ) * 2
    assert len(candidate.split()) >= 130
    return candidate


def test_only_missing_explicit_revision_is_completed_before_delivery(monkeypatch):
    candidate = _revision_candidate_without_performative()
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("explicit-revision", "face-model")
    state.messages.append(
        {"role": "assistant", "content": "Use synchronous Depth for every diagnosis."}
    )
    calls = 0

    def post_streaming(_payload, callback, *, cancel_event=None):
        nonlocal calls
        del cancel_event
        calls += 1
        callback(candidate)
        return candidate, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        "Assume first-token latency must stay under four seconds. Revise the recommendation.",
        session_id=state.session_id,
        model="face-model",
        stream_callback=spoken.append,
    )

    revision = "I revise the recommendation to reflect the operator's changed constraint."
    semantic = result["output_guard"]["substantive_followup"]
    receipt = result["output_guard"]["explicit_revision_postcondition"]
    assert calls == 1
    assert result["completed"] is True
    assert result["text"] == f"{candidate.strip()}\n\n{revision}"
    assert semantic["first_candidate_failures"] == ["missing_explicit_revision"]
    assert semantic["corrective_regeneration_attempted"] is False
    assert semantic["fail_closed"] is False
    assert receipt["applied"] is True
    assert receipt["only_missing_revision"] is True
    assert spoken == [result["text"]]
    assert state.messages[-1] == {"role": "assistant", "content": result["text"]}


def test_corrective_only_missing_explicit_revision_is_completed(monkeypatch):
    first = "I can revise that if you want more detail."
    corrected = _revision_candidate_without_performative()
    attempts = iter((first, corrected))
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("corrective-explicit-revision", "face-model")
    state.messages.append(
        {"role": "assistant", "content": "Use synchronous Depth for every diagnosis."}
    )
    before = list(state.messages)
    calls = 0

    def post_streaming(_payload, callback, *, cancel_event=None):
        nonlocal calls
        del cancel_event
        calls += 1
        candidate = next(attempts)
        callback(candidate)
        return candidate, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        "Assume first-token latency must stay under four seconds. Revise the recommendation.",
        session_id=state.session_id,
        model="face-model",
        stream_callback=spoken.append,
    )

    revision = "I revise the recommendation to reflect the operator's changed constraint."
    semantic = result["output_guard"]["substantive_followup"]
    receipt = semantic["corrective_explicit_revision_postcondition"]
    assert calls == 2
    assert result["completed"] is True
    assert result["text"] == f"{corrected.strip()}\n\n{revision}"
    assert receipt["applied"] is True
    assert receipt["only_missing_revision"] is True
    assert semantic["corrective_candidate_failures"] == []
    assert semantic["fail_closed"] is False
    assert spoken == [result["text"]]
    assert state.messages == before + [
        {
            "role": "user",
            "content": (
                "Assume first-token latency must stay under four seconds. "
                "Revise the recommendation."
            ),
        },
        {"role": "assistant", "content": result["text"]},
    ]


def test_explicit_revision_postcondition_does_not_mask_underlength(monkeypatch):
    short = (
        "First-token latency must stay under four seconds. Face answers in the "
        "foreground while Depth runs asynchronously in the background because the "
        "spoken path must remain responsive."
    )
    attempts = iter((short, short))
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("short-explicit-revision", "face-model")
    state.messages.append(
        {"role": "assistant", "content": "Use synchronous Depth for every diagnosis."}
    )
    before = list(state.messages)

    def post_streaming(_payload, callback, *, cancel_event=None):
        del cancel_event
        candidate = next(attempts)
        callback(candidate)
        return candidate, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        "Assume first-token latency must stay under four seconds. Revise the recommendation.",
        session_id=state.session_id,
        model="face-model",
        stream_callback=spoken.append,
    )

    semantic = result["output_guard"]["substantive_followup"]
    first_receipt = semantic["first_explicit_revision_postcondition"]
    corrective_receipt = semantic["corrective_explicit_revision_postcondition"]
    assert result["completed"] is False
    assert result["text"] == ""
    assert "underlength" in semantic["first_candidate_failures"]
    assert "underlength" in semantic["corrective_candidate_failures"]
    assert first_receipt["applied"] is False
    assert corrective_receipt["applied"] is False
    assert spoken == []
    assert state.messages == before


def test_latency_lobe_staging_postcondition_does_not_mask_other_failures():
    text, receipt = _enforce_latency_lobe_staging_postcondition(
        "Too short.",
        {"latency_sequence_required": True},
        ["underlength", "latency_lobe_staging_missing"],
    )

    assert text == "Too short."
    assert receipt is not None
    assert receipt["applied"] is False
    assert receipt["only_resolved_latency_gaps"] is False


def test_latency_staging_contradiction_fails_closed_without_tts_or_history(monkeypatch):
    contradictory = (
        "Revised recommendation: first-token latency must stay under four seconds. "
        "Face owns the immediate foreground response because that is the only way to "
        "begin useful delivery inside the operator's bound. While the architecture "
        "must not use Depth, the response should still explain the current diagnosis, "
        "identify uncertainty, preserve the timing requirement, and state which later "
        "evidence could change the conclusion. This revision changes the critical path "
        "from complete analysis before speech to a bounded conversational opening. The "
        "recommendation is explicit, the first-token latency dimension remains active, "
        "and the four-second deadline is attached to that first useful token. The design "
        "would record observed logs, compare repeated failures, and disclose any later "
        "correction in the same conversation. It also keeps the explanation substantive "
        "enough for the operator to understand the mechanism, practical consequence, and "
        "decision boundary. The resulting interaction is responsive, auditable, and clear "
        "about the evidence still needed, but it deliberately excludes the heavier lobe "
        "from the sequence rather than allowing it to run after the opening response."
    )
    assert len(contradictory.split()) >= 130
    attempts = iter((contradictory, contradictory))
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("latency-staging-rejected", "face-model")
    state.messages.append(
        {"role": "assistant", "content": "Use both lobes for the diagnosis."}
    )
    before = list(state.messages)

    def post_streaming(_payload, callback, *, cancel_event=None):
        del cancel_event
        candidate = next(attempts)
        callback(candidate)
        return candidate, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        "Assume first-token latency must stay under four seconds. Revise the recommendation.",
        session_id="latency-staging-rejected",
        model="face-model",
        extra_system=_r2_verified_depth_context(),
        stream_callback=spoken.append,
    )

    semantic = result["output_guard"]["substantive_followup"]
    assert result["completed"] is False
    assert result["text"] == ""
    assert result["api_calls"] == 2
    assert spoken == []
    assert state.messages == before
    assert "latency_lobe_staging_contradiction" in semantic[
        "first_candidate_failures"
    ]
    assert "latency_lobe_staging_contradiction" in semantic[
        "corrective_candidate_failures"
    ]
    assert semantic["fail_closed"] is True


def test_r24_wrong_active_constraint_is_not_laundered_by_postconditions(monkeypatch):
    wrong = (
        "The final architecture keeps the Face lobe first in the interactive path "
        "while the Depth lobe runs asynchronously in the background. Face gives the "
        "immediate answer, and Depth returns the complete diagnosis into the same "
        "conversation when its evidence pass finishes. This order is useful because "
        "the difficult reasoning must not delay the opening interaction, yet the user "
        "still receives the full result without asking for another turn. The active "
        "constraint remains first-token latency under four seconds, not end-to-end "
        "voice latency. That means the opening response remains the timing target, and "
        "the rest of the exchange can finish outside the bound. Face therefore owns "
        "the immediate response, Depth owns the non-blocking diagnostic work, and the "
        "written channel preserves the evidence-backed explanation. This division "
        "keeps the architecture responsive and preserves diagnostic quality across "
        "the same session. The constraint is stated directly so the routing decision "
        "can be inspected, tested, and compared with the original recommendation "
        "without hiding which lobe owns each stage of the final design."
    )
    assert len(wrong.split()) >= 130
    attempts = iter((wrong, wrong))
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("r24-wrong-active-constraint", "face-model")
    state.messages.extend(
        [
            {
                "role": "user",
                "content": (
                    "Assume first-token latency must stay under four seconds. "
                    "Revise the recommendation."
                ),
            },
            {"role": "assistant", "content": "Face should answer first."},
        ]
    )
    before = list(state.messages)

    def post_streaming(_payload, callback, *, cancel_event=None):
        del cancel_event
        candidate = next(attempts)
        callback(candidate)
        return candidate, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        (
            "No - I mean end-to-end voice latency under four seconds. "
            "Correct the recommendation."
        ),
        session_id="r24-wrong-active-constraint",
        model="face-model",
        extra_system=_r2_verified_depth_context(),
        stream_callback=spoken.append,
    )

    semantic = result["output_guard"]["substantive_followup"]
    assert result["completed"] is False
    assert result["text"] == ""
    assert result["api_calls"] == 2
    assert "latency_constraint_direction_contradiction" in semantic[
        "first_candidate_failures"
    ]
    assert "latency_constraint_direction_contradiction" in semantic[
        "corrective_candidate_failures"
    ]
    assert semantic["first_latency_delivery_sequence_postcondition"]["applied"] is False
    assert spoken == []
    assert state.messages == before


@pytest.mark.parametrize(
    "direction_sentence",
    (
        "The old constraint was end-to-end voice latency under four seconds, "
        "and the current constraint is first-token latency under four seconds.",
        "The old constraint was end-to-end voice latency under four seconds, "
        "and first-token latency is now the current constraint under four seconds.",
        "The old constraint was end-to-end voice latency under four seconds, "
        "and first-token latency remains the active constraint under four seconds.",
        "The old constraint was end-to-end voice latency under four seconds, "
        "and first-token latency remains the constraint under four seconds.",
        "The old constraint was end-to-end voice latency under four seconds, "
        "and first-token latency stays the requirement under four seconds.",
        "The old constraint was end-to-end voice latency under four seconds, "
        "and first-token latency is the target under four seconds.",
    ),
    ids=(
        "current-prefix",
        "current-suffix",
        "active-suffix",
        "constraint-suffix",
        "requirement-suffix",
        "target-suffix",
    ),
)
def test_r25_reversed_same_clause_constraint_roles_fail_closed(
    monkeypatch,
    direction_sentence,
):
    wrong = (
        "The final architecture keeps the Face lobe first in the interactive path "
        "while the Depth lobe runs asynchronously in the background. Face gives the "
        "immediate answer, and Depth returns the complete diagnosis into the same "
        "conversation when its evidence pass finishes. This ordering is useful because "
        "the difficult reasoning must not delay the opening interaction, yet the user "
        "still receives the full result without asking for another turn. "
        f"{direction_sentence} Face therefore owns the "
        "immediate response, Depth owns the non-blocking diagnostic work, and the written "
        "channel preserves the evidence-backed explanation. This division keeps the "
        "architecture responsive and preserves diagnostic quality across one session. "
        "The constraint direction is stated directly so the routing decision can be "
        "inspected, tested, and compared with the original recommendation without hiding "
        "which lobe owns each stage or when the complete diagnostic result becomes visible."
    )
    assert len(wrong.split()) >= 130
    attempts = iter((wrong, wrong))
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("r25-reversed-constraint", "face-model")
    state.messages.extend(
        [
            {
                "role": "user",
                "content": (
                    "Assume first-token latency must stay under four seconds. "
                    "Revise the recommendation."
                ),
            },
            {"role": "assistant", "content": "Face should answer first."},
        ]
    )
    before = list(state.messages)

    def post_streaming(_payload, callback, *, cancel_event=None):
        del cancel_event
        candidate = next(attempts)
        callback(candidate)
        return candidate, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        (
            "No - I mean end-to-end voice latency under four seconds. "
            "Correct the recommendation."
        ),
        session_id="r25-reversed-constraint",
        model="face-model",
        extra_system=_r2_verified_depth_context(),
        stream_callback=spoken.append,
    )

    semantic = result["output_guard"]["substantive_followup"]
    assert result["completed"] is False
    assert result["text"] == ""
    assert result["api_calls"] == 2
    assert "latency_constraint_direction_contradiction" in semantic[
        "first_candidate_failures"
    ]
    assert "latency_constraint_direction_contradiction" in semantic[
        "corrective_candidate_failures"
    ]
    assert semantic["first_latency_delivery_sequence_postcondition"]["applied"] is False
    assert spoken == []
    assert state.messages == before


def test_r30_face_alone_is_scoped_only_by_live_path_plus_affirmative_depth_stage():
    scoped = (
        "Face alone meets the synchronous live voice budget while Depth runs "
        "asynchronously in the background and delivers the full analysis later."
    )
    global_face_only = (
        "Use only Face for the architecture, although Depth runs asynchronously "
        "in the background."
    )
    spaced_global_face_only = (
        "Face alone is the complete design for an immediate response. Depth runs "
        "asynchronously in the background and returns later."
    )
    explicit_depth_exclusion = (
        "Face handles the live voice path without the Depth lobe."
    )

    assert _latency_staging_rejected(scoped) is False
    assert _latency_staging_rejected(global_face_only) is True
    assert _latency_staging_rejected(spaced_global_face_only) is True
    assert _latency_staging_rejected(explicit_depth_exclusion) is True


@pytest.mark.parametrize(
    "scope_sentence",
    (
        "Face alone meets the live voice budget while Depth runs asynchronously "
        "in the background and delivers the full diagnosis later.",
        "First-token latency remains a component requirement under the governing "
        "end-to-end voice latency constraint. Face owns the foreground response "
        "while Depth runs asynchronously in the background.",
        "First-token latency remains a required component sub-budget under the "
        "governing end-to-end voice latency constraint. Face owns the foreground "
        "response while Depth runs asynchronously in the background.",
        "First-token latency remains a component target inside the overall "
        "end-to-end voice latency constraint. Face owns the foreground response "
        "while Depth runs asynchronously in the background.",
        "The previous architecture omitted end-to-end voice latency. The new "
        "architecture still monitors first-token latency as a useful component "
        "metric. Face owns the foreground response while Depth runs asynchronously "
        "in the background.",
        "The former design treated end-to-end voice latency as out of scope. The "
        "current design preserves first-token latency as an internal metric. Face "
        "owns the foreground response while Depth runs asynchronously in the background.",
        "The previous design omitted measurement of end-to-end voice latency, so the "
        "new architecture measures it. First-token latency is a monitored target, "
        "not the governing constraint. Face owns the foreground response while Depth "
        "runs asynchronously in the background.",
        "The previous constraint failed to measure end-to-end voice latency. The new "
        "architecture measures it while first-token latency remains only a component metric.",
        "Unlike the prior requirement, we now measure end-to-end voice latency through "
        "audible playback while first-token latency remains a monitored component metric.",
        "The earlier target missed end-to-end voice latency. The new architecture measures "
        "the full path while first-token latency remains an internal metric.",
        "The old limit was blind to end-to-end voice latency. The new architecture measures "
        "it while first-token latency remains a subordinate component metric.",
        "End-to-end voice latency was outside the previous constraint. It is now the "
        "governing measurement while first-token latency remains an internal metric.",
        "Unlike end-to-end voice latency, the old constraint was first-token latency. "
        "The new architecture measures the complete voice path.",
        "Unlike first-token latency, the current constraint is end-to-end voice latency. "
        "First-token timing remains only an internal component metric.",
        "Unlike first-token latency, the current constraint, end-to-end voice latency, "
        "covers the full spoken path.",
        "Unlike end-to-end voice latency, the old constraint, first-token latency, "
        "measured only the opening token.",
        "Unlike first-token latency, the current constraint, end-to-end voice latency "
        "(<4s), covers the full spoken path.",
        "Unlike end-to-end voice latency, the old constraint, first-token latency "
        "(under four seconds), measured only the opening token.",
        "Unlike first-token latency, the current constraint, the end-to-end voice "
        "latency, covers the full spoken path.",
        "Unlike end-to-end voice latency, the old constraint, the first-token latency, "
        "measured only the opening token.",
        "Vs. first-token latency, the current constraint, end-to-end voice latency, "
        "covers the full spoken path.",
        "- Vs. first-token latency, the current constraint, end-to-end voice latency, "
        "covers the full spoken path.",
        "### Vs. first-token latency, the current constraint, end-to-end voice latency, "
        "covers the full spoken path.",
        "**Vs.** first-token latency, the current constraint, end-to-end voice latency, "
        "covers the full spoken path.",
        "`Vs.` first-token latency, the current constraint, end-to-end voice latency, "
        "covers the full spoken path.",
        "Vs. **the first-token latency**, the current constraint, end-to-end voice "
        "latency, covers the full spoken path.",
        "**Vs.** **the first-token latency**, the current constraint, end-to-end voice "
        "latency, covers the full spoken path.",
        "- Vs. `the first-token latency`, the current constraint, end-to-end voice "
        "latency, covers the full spoken path.",
        "Vs. the *first-token latency*, the current constraint, end-to-end voice "
        "latency, covers the full spoken path.",
        "Vs. **the end-to-end voice latency**, the old constraint, first-token latency, "
        "measured only the opening token.",
        "Vs. **TTFT latency**, the current constraint, end-to-end voice latency, "
        "covers the full spoken path.",
        "Vs. **time-to-first-token latency**, the current constraint, end-to-end voice "
        "latency, covers the full spoken path.",
        "Vs. _time-to-first-token latency_, the current constraint, end-to-end voice "
        "latency, covers the full spoken path.",
        "Vs. __TTFT latency__, the current constraint, end-to-end voice latency, "
        "covers the full spoken path.",
        "Vs. _the first-token latency_, the current constraint, end-to-end voice "
        "latency, covers the full spoken path.",
        "Vs. _TTFT latency under four seconds_, the current constraint, end-to-end "
        "voice latency, covers the full spoken path.",
        "Vs. __end-to-end voice latency (<4s>)__, the old constraint, first-token "
        "latency, measured only the opening token.",
        "Unlike the first-token latency, the current constraint, end-to-end voice "
        "latency, covers the full spoken path.",
        "Compared with the end-to-end voice latency, the old constraint, first-token "
        "latency, measured only the opening token.",
        "Unlike **the first-token latency**, the current constraint, end-to-end voice "
        "latency, covers the full spoken path.",
        "End-to-end voice latency concerns devs. First-token latency was the old "
        "constraint.",
        "First-token latency concerns devs. End-to-end voice latency is the current "
        "constraint.",
    ),
    ids=(
        "scoped-face-alone",
        "component-requirement",
        "required-component-sub-budget",
        "component-target",
        "historical-e2e-omission",
        "historical-e2e-out-of-scope",
        "cross-clause-adoption-and-monitored-target",
        "previous-constraint-failed-to-measure",
        "prior-requirement-contrast",
        "earlier-target-missed",
        "old-limit-blind",
        "outside-previous-constraint",
        "contrast-e2e-from-old-first-token",
        "contrast-first-token-from-current-e2e",
        "contrast-current-value-appositive",
        "contrast-old-value-appositive",
        "contrast-current-parenthesized-deadline",
        "contrast-old-parenthesized-deadline",
        "contrast-current-articled-value",
        "contrast-old-articled-value",
        "contrast-vs-period-current-value",
        "contrast-vs-bullet-current-value",
        "contrast-vs-heading-current-value",
        "contrast-vs-bold-current-value",
        "contrast-vs-backtick-current-value",
        "contrast-vs-bold-term-current-value",
        "contrast-bold-vs-bold-term-current-value",
        "contrast-bullet-vs-backtick-term-current-value",
        "contrast-vs-article-italic-term-current-value",
        "contrast-vs-bold-term-old-value",
        "contrast-vs-bold-ttft-latency-current-value",
        "contrast-vs-bold-time-to-first-token-latency-current-value",
        "contrast-vs-underscore-time-to-first-token-latency-current-value",
        "contrast-vs-double-underscore-ttft-latency-current-value",
        "contrast-vs-underscore-article-first-token-current-value",
        "contrast-vs-underscore-ttft-deadline-current-value",
        "contrast-vs-double-underscore-e2e-deadline-old-value",
        "contrast-current-value-with-article",
        "contrast-old-value-with-article",
        "contrast-markdown-current-value-with-article",
        "devs-period-does-not-leak-historical-role",
        "devs-period-does-not-leak-current-role",
    ),
)
def test_r30_scoped_staging_and_subordinate_first_token_roles_are_deliverable(
    monkeypatch,
    scope_sentence,
):
    candidate = (
        "I revise the recommendation because the measurement boundary changed. "
        "The old constraint was first-token latency under four seconds; the new "
        "constraint is end-to-end voice latency under four seconds. "
        f"{scope_sentence} "
        "Face owns the foreground response while Depth runs asynchronously in the "
        "background and delivers verified analysis later. "
        "The response-start SLO runs from the end of the operator's speech through "
        "ASR finalization, Face generation, and TTS to the first useful non-silent "
        "reply onset. Full audible utterance completion is tracked separately as an "
        "independent metric and is not subject to that same bound. "
        "This sequence keeps synchronous heavy analysis outside the spoken critical "
        "path while preserving a complete evidence-backed result in the same session. "
        "The operator hears a bounded useful recommendation, can interrupt playback, "
        "and does not need to ask again for the later diagnosis. The written channel "
        "retains the mechanism, uncertainty, practical consequence, and next action. "
        "That division is necessary because an early token alone does not prove that "
        "useful reply audio began inside the deadline, while the separate completion "
        "metric still exposes a slow or truncated utterance."
    )
    assert len(candidate.split()) >= 130
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session(
        f"r30-valid-scope-{uuid.uuid4()}",
        "face-model",
    )
    state.messages.extend(
        [
            {
                "role": "user",
                "content": (
                    "Assume first-token latency must stay under four seconds. "
                    "Revise the recommendation."
                ),
            },
            {"role": "assistant", "content": "Face should answer first."},
        ]
    )

    def post_streaming(_payload, callback, *, cancel_event=None):
        del cancel_event
        callback(candidate)
        return candidate, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        (
            "No - I mean end-to-end voice latency under four seconds. "
            "Correct the recommendation."
        ),
        session_id=state.session_id,
        model="face-model",
        extra_system=_r2_verified_depth_context(),
        stream_callback=spoken.append,
    )

    semantic = result["output_guard"]["substantive_followup"]
    assert result["completed"] is True
    assert result["api_calls"] == 1
    assert semantic["first_candidate_failures"] == []
    assert semantic["fail_closed"] is False
    assert spoken == [candidate]
    assert state.messages[-1] == {"role": "assistant", "content": candidate}


@pytest.mark.parametrize(
    "heading",
    [
        "**Revised Recommendation (End-to-End Voice Latency Constraint: <4s)**",
        "**Revised Recommendation (v2.0: End-to-End Voice Latency Constraint: <4s)**",
        "**Revised Recommendation (e.g. End-to-End Voice Latency Constraint: <4s)**",
        "**Revised Recommendation (phase one; End-to-End Voice Latency Constraint: <4s)**",
        "**Revised Recommendation (latency? End-to-End Voice Latency Constraint: <4s)**",
    ],
)
def test_r32_parenthesized_current_latency_heading_is_not_historical(
    monkeypatch,
    heading,
):
    candidate = (
        f"{heading}\n\n"
        "The old constraint was first-token latency under four seconds; the new "
        "constraint is end-to-end voice latency under four seconds. Face owns the "
        "synchronous foreground response while Depth runs asynchronously in the "
        "background and delivers verified analysis later. The response-start SLO runs "
        "from the end of the operator's speech through ASR finalization, Face generation, "
        "and TTS to the first useful non-silent reply onset. Full audible utterance "
        "completion is tracked separately as an independent metric and is not subject "
        "to that same bound. This corrects the recommendation because an early token "
        "does not prove useful reply audio began. The "
        "foreground answer therefore stays bounded and useful, while the deeper evidence "
        "pass remains outside the spoken critical path. The written channel keeps the "
        "complete mechanism, uncertainty, practical consequence, and next action in the "
        "same session. Measuring the entire path also exposes stalls after generation, "
        "including synthesis delay, device buffering, and playback interruption. The "
        "operator can test the response-start budget from captured speech through first "
        "useful audible onset without deleting the Depth stage or confusing a component "
        "metric with the governing interaction constraint."
    )
    assert len(candidate.split()) >= 130
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session(
        f"r32-parenthesized-heading-{uuid.uuid4()}",
        "face-model",
    )
    state.messages.extend(
        [
            {
                "role": "user",
                "content": (
                    "Assume first-token latency must stay under four seconds. "
                    "Revise the recommendation."
                ),
            },
            {"role": "assistant", "content": "Face should answer first."},
        ]
    )

    def post_streaming(_payload, callback, *, cancel_event=None):
        del cancel_event
        callback(candidate)
        return candidate, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        (
            "No - I mean end-to-end voice latency under four seconds. "
            "Correct the recommendation."
        ),
        session_id=state.session_id,
        model="face-model",
        extra_system=_r2_verified_depth_context(),
        stream_callback=spoken.append,
    )

    semantic = result["output_guard"]["substantive_followup"]
    assert result["completed"] is True
    assert semantic["first_candidate_failures"] == []
    assert semantic["first_latency_constraint_postcondition"][
        "new_dimension_historical"
    ] is False
    assert spoken == [candidate]


@pytest.mark.parametrize(
    "role_sentence",
    [
        "The metric (end-to-end voice latency) was the old constraint.",
        "The old constraint was **end-to-end voice latency**.",
        "**Old constraint:** **end-to-end voice latency under four seconds**.",
        "`Old requirement:` `end-to-end voice latency under four seconds`.",
        "End-to-end voice latency, the old constraint, governs the design.",
        "First-token latency, the current constraint, still governs the design.",
        "End-to-end voice latency under four seconds, **the old constraint**, "
        "has been replaced.",
        "End-to-end voice latency under four seconds, `the old constraint`, "
        "has been replaced.",
        "First-token latency under four seconds, **the current constraint**, "
        "still governs.",
        "First-token latency under four seconds, `the current constraint`, "
        "still governs.",
        "End-to-end voice latency—the old constraint—governed the prior design.",
        "End-to-end voice latency — the old constraint — governed the prior design.",
        "End-to-end voice latency – the old constraint – governed the prior design.",
        "First-token latency—the current constraint—still governs the design.",
        "End-to-end voice latency — the old constraint.",
        "End-to-end voice latency — was the old constraint.",
        "End-to-end voice latency – was previously the target.",
        "First-token latency — the current constraint.",
        "First-token latency — remains the active constraint.",
        "End-to-end voice latency, the old constraint.",
        "First-token latency, the current constraint.",
        "End-to-end voice latency, the old constraint, end-to-end voice latency, "
        "was retired.",
        "First-token latency, the current constraint, first-token latency, still "
        "governs.",
        "End-to-end voice latency, the old constraint, first-token latency, and "
        "throughput were measured.",
        "End-to-end voice latency, the old constraint, first-token latency, and "
        "jitter were tracked in the prior design.",
        "First-token latency, the current constraint, end-to-end voice latency, and "
        "throughput still govern.",
        "The prior heading ended with the abbreviation **Vs.** End-to-end voice "
        "latency, the old constraint, first-token latency, and jitter were tracked "
        "in the prior design.",
        "_End-to-end voice latency_, the old constraint, governs the prior design.",
        "__End-to-end voice latency__, the old constraint, governs the prior design.",
        "_First-token latency_, the current constraint, still governs the design.",
        "__First-token latency__, the current constraint, still governs the design.",
        "_End-to-end voice latency under four seconds_, the old constraint, governed "
        "the prior design.",
        "__End-to-end voice latency (<4s>)__, the old constraint, governed the prior "
        "design.",
        "_First-token latency under four seconds_, the current constraint, still "
        "governs the design.",
        "_TTFT latency under four seconds_, the current constraint, still governs.",
        "_End-to-end voice latency,_ the old constraint, governed the prior design.",
        "__End-to-end voice latency under four seconds,__ the old constraint, governed "
        "the prior design.",
        "_First-token latency,_ the current constraint, still governs the design.",
        "_TTFT latency under four seconds,_ the current constraint, still governs.",
        "_End-to-end voice latency—_ the old constraint—governed the prior design.",
        "__End-to-end voice latency under four seconds–__ the old constraint–governed "
        "the prior design.",
        "_First-token latency—_ the current constraint—still governs the design.",
        "_TTFT latency under four seconds–_ the current constraint–still governs.",
        "End-to-end voice latency: the old constraint.",
        "_End-to-end voice latency:_ the old constraint.",
        "End-to-end voice latency under four seconds: old requirement.",
        "End-to-end voice latency: it was the old constraint.",
        "_End-to-end voice latency:_ this was previously the target.",
        "End-to-end voice latency under four seconds: that used to be the active "
        "requirement.",
        "End-to-end voice latency: it was the old constraint, but the design "
        "remains active.",
        "End-to-end voice latency: the metric was the old constraint.",
        "End-to-end voice latency: the latency was previously the target.",
        "End-to-end voice latency: it alone was the old constraint.",
        "End-to-end voice latency: this metric itself was formerly the target.",
        "End-to-end voice latency: this measurement was previously the target.",
        "End-to-end voice latency: that measure was the old requirement.",
        "End-to-end voice latency: the limit was the old constraint.",
        "End-to-end voice latency: this deadline was formerly the target.",
        "End-to-end voice latency, which was the old constraint, was replaced.",
        "End-to-end voice latency—which was previously the target—was replaced.",
        "End-to-end voice latency, which had been the old constraint, was replaced.",
        "End-to-end voice latency—which was once the old target—was replaced.",
        "End-to-end voice latency: this measurement had previously been the target.",
        "End-to-end voice latency: it has been the old constraint.",
        "End-to-end voice latency, which has previously been the target, was replaced.",
        "End-to-end voice latency—which had remained the prior requirement—was replaced.",
        "End-to-end voice latency: the measurement was once considered the old target.",
        "End-to-end voice latency: this measurement is no longer active.",
        "End-to-end voice latency: this measurement has no longer been active.",
        "End-to-end voice latency: it formerly served as the old constraint.",
        "End-to-end voice latency, which had served as the previous target, was "
        "replaced.",
        "End-to-end voice latency: this measurement used to serve as the active "
        "requirement.",
        "End-to-end voice latency: this metric had previously served as the old "
        "constraint.",
        "End-to-end voice latency: this metric had formerly operated as the prior "
        "target.",
        "End-to-end voice latency: this metric was previously serving as the old "
        "requirement.",
        "End-to-end voice latency: this metric had previously been serving as the old "
        "requirement.",
        "_End-to-end voice latency under four seconds was the old constraint._",
        "First-token latency: the current constraint.",
        "__First-token latency:__ current requirement.",
        "__First-token latency under four seconds is the current constraint.__",
        "First-token latency: it remains the current constraint, although it was "
        "previously the target.",
        "First-token latency: it is the active requirement, although it was formerly "
        "the target.",
        "First-token latency: it continues to be the current constraint.",
        "First-token latency, which continues to be the active requirement, still "
        "governs.",
        "First-token latency: it has remained the current constraint.",
        "First-token latency: it continues as the active requirement.",
        "First-token latency, which still serves as the current constraint, still "
        "governs.",
        "First-token latency: it is still the current constraint.",
        "First-token latency: it acts as the current constraint.",
        "First-token latency: it continues to serve as the current constraint.",
        "First-token latency: it remains as the governing requirement.",
        "First-token latency: this metric has continued to be the active constraint.",
        "First-token latency: this metric had continued as the current requirement.",
        "First-token latency: this metric has become the current constraint.",
        "First-token latency: this metric has served as the current constraint.",
        "First-token latency: this metric has operated as the active requirement.",
        "First-token latency: this metric has continued to serve as the current target.",
        "First-token latency: this metric is still serving as the current constraint.",
        "First-token latency: this metric has been serving as the current constraint.",
        "First-token latency: this metric has kept serving as the active requirement.",
        "End-to-end voice latency under four seconds, but it used to be the constraint.",
        "End-to-end voice latency under four seconds, **but it used to be the "
        "constraint**.",
        "End-to-end voice latency under four seconds, but **it used to be the "
        "constraint**.",
        "End-to-end voice latency under four seconds, however, it was previously the target.",
        "End-to-end voice latency) was the old constraint.",
        "End-to-end voice latency (was the old constraint.",
    ],
)
def test_r32_parenthetical_reverse_or_malformed_latency_role_fails_closed(
    monkeypatch,
    role_sentence,
):
    contradictory = (
        "The old constraint was first-token latency under four seconds; the new "
        "constraint is end-to-end voice latency under four seconds. "
        f"{role_sentence} Face owns the foreground response while Depth runs "
        "asynchronously in the background and returns later. The measured delivery "
        "path includes ASR input, Face generation, TTS synthesis, and audible playback. "
    ) * 5
    attempts = iter((contradictory, contradictory))
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session(
        f"r32-parenthetical-negative-{uuid.uuid4()}",
        "face-model",
    )
    state.messages.extend(
        [
            {
                "role": "user",
                "content": (
                    "Assume first-token latency must stay under four seconds. "
                    "Revise the recommendation."
                ),
            },
            {"role": "assistant", "content": "Face should answer first."},
        ]
    )

    def post_streaming(_payload, callback, *, cancel_event=None):
        del cancel_event
        answer = next(attempts)
        callback(answer)
        return answer, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    result = face.chat(
        (
            "No - I mean end-to-end voice latency under four seconds. "
            "Correct the recommendation."
        ),
        session_id=state.session_id,
        model="face-model",
        extra_system=_r2_verified_depth_context(),
        stream_callback=lambda _text: None,
    )

    semantic = result["output_guard"]["substantive_followup"]
    assert result["completed"] is False
    assert "latency_constraint_direction_contradiction" in semantic[
        "first_candidate_failures"
    ]
    assert "latency_constraint_direction_contradiction" in semantic[
        "corrective_candidate_failures"
    ]


@pytest.mark.parametrize(
    ("value", "pattern"),
    (
        ("first-token latency_suffix", _FIRST_TOKEN_LATENCY_RE),
        ("first-token latency2", _FIRST_TOKEN_LATENCY_RE),
        ("time-to-first-token latency_suffix", _FIRST_TOKEN_LATENCY_RE),
        ("TTFT latency_suffix", _FIRST_TOKEN_LATENCY_RE),
        ("_the first-token latency__", _FIRST_TOKEN_LATENCY_RE),
        ("__the time-to-first-token latency_", _FIRST_TOKEN_LATENCY_RE),
        ("_the first-token latency under four seconds__", _FIRST_TOKEN_LATENCY_RE),
        ("__the first-token latency under four seconds_", _FIRST_TOKEN_LATENCY_RE),
        ("_the TTFT latency (<4s>)__", _FIRST_TOKEN_LATENCY_RE),
        ("__the time-to-first-token latency <4s>_", _FIRST_TOKEN_LATENCY_RE),
        ("_the first-token latency.__", _FIRST_TOKEN_LATENCY_RE),
        ("__the first-token latency._", _FIRST_TOKEN_LATENCY_RE),
        ("_the first-token latency under four seconds.__", _FIRST_TOKEN_LATENCY_RE),
        ("__the TTFT latency (<4s>)!_", _FIRST_TOKEN_LATENCY_RE),
        ("_the first-token latency under four seconds___", _FIRST_TOKEN_LATENCY_RE),
        ("__the TTFT latency (<4s>)___", _FIRST_TOKEN_LATENCY_RE),
        ("_the first-token latency", _FIRST_TOKEN_LATENCY_RE),
        ("__the first-token latency under four seconds", _FIRST_TOKEN_LATENCY_RE),
        ("_time-to-first-token latency", _FIRST_TOKEN_LATENCY_RE),
        (
            "___the time-to-first-token latency under four seconds___",
            _FIRST_TOKEN_LATENCY_RE,
        ),
        ("end-to-end voice latency_suffix", _END_TO_END_VOICE_LATENCY_RE),
        ("_the end-to-end voice latency__", _END_TO_END_VOICE_LATENCY_RE),
        (
            "_the end-to-end voice latency under four seconds__",
            _END_TO_END_VOICE_LATENCY_RE,
        ),
        (
            "__the end-to-end voice latency (<4s>)_",
            _END_TO_END_VOICE_LATENCY_RE,
        ),
        ("_the end-to-end voice latency.__", _END_TO_END_VOICE_LATENCY_RE),
        (
            "__the end-to-end voice latency under four seconds!_",
            _END_TO_END_VOICE_LATENCY_RE,
        ),
        (
            "_the end-to-end voice latency under four seconds___",
            _END_TO_END_VOICE_LATENCY_RE,
        ),
        ("_the end-to-end voice latency", _END_TO_END_VOICE_LATENCY_RE),
        (
            "__the end-to-end voice latency under four seconds",
            _END_TO_END_VOICE_LATENCY_RE,
        ),
    ),
)
def test_r33_latency_dimensions_reject_partial_identifiers_and_malformed_wrappers(
    value,
    pattern,
):
    assert pattern.search(value) is None


@pytest.mark.parametrize(
    "direction_sentence",
    (
        "Change from first-token latency_suffix to end-to-end voice latency.",
        "Change from time-to-first-token latency2 to end-to-end voice latency.",
        "Change from _the first-token latency__ to _end-to-end voice latency_.",
        "Change from _the first-token latency under four seconds__ to "
        "_end-to-end voice latency under four seconds_.",
        "Change from _the first-token latency under four seconds.__ to "
        "_end-to-end voice latency under four seconds_.",
        "Change from _the first-token latency under four seconds___ to "
        "_end-to-end voice latency under four seconds_.",
        "Change from _the first-token latency to "
        "_end-to-end voice latency under four seconds_.",
        "Change from __the first-token latency under four seconds to "
        "__end-to-end voice latency under four seconds__.",
        "Change from _time-to-first-token latency to "
        "_end-to-end voice latency under four seconds_.",
        "Change from _end-to-end voice latency to "
        "_first-token latency under four seconds_.",
    ),
)
def test_r33_invalid_latency_terms_cannot_satisfy_direction(direction_sentence):
    assert _latency_constraint_change_direction(direction_sentence) == "missing"


@pytest.mark.parametrize(
    "direction_sentence",
    (
        "_Change from first-token latency to end-to-end voice latency._",
        "__Change from TTFT latency to end-to-end voice latency.__",
        "__The old constraint was _first-token latency_; the current constraint "
        "is _end-to-end voice latency_.__",
    ),
)
def test_r33_valid_larger_underscore_presentation_preserves_direction(
    direction_sentence,
):
    assert _latency_constraint_change_direction(direction_sentence) == "correct"


@pytest.mark.parametrize(
    "direction_sentence",
    (
        "_Vs. Change from first-token latency to end-to-end voice latency.",
        "__Vs. Change from TTFT latency to end-to-end voice latency.",
        "_devs. Change from first-token latency to end-to-end voice latency.",
        "_v2.0 Change from first-token latency to end-to-end voice latency.",
    ),
)
def test_r33_abbreviation_cannot_reset_malformed_underscore_presentation(
    direction_sentence,
):
    assert _latency_constraint_change_direction(direction_sentence) == "missing"


@pytest.mark.parametrize(
    "sentence",
    (
        "First-token latency: that design remains the current requirement.",
        "First-token latency: the model is the active constraint.",
        "First-token latency, which model is the active constraint.",
        "First-token latency: the service continues to serve as the current "
        "constraint.",
        "First-token latency: the model, after revision, remains active for "
        "diagnostics.",
        "First-token latency: the service, which was restarted, continues to serve "
        "as the current requirement.",
        "First-token latency: the model, not the latency metric, remains active.",
        "First-token latency: the model—after revision—remains active.",
        "First-token latency: the service—which was restarted—continues to serve as "
        "the current requirement.",
    ),
)
def test_r33_foreign_subject_cannot_become_current_latency_role(sentence):
    assert not _latency_dimension_asserted_current(sentence, _FIRST_TOKEN_LATENCY_RE)


@pytest.mark.parametrize(
    "sentence",
    (
        "End-to-end voice latency: the design, which was the old constraint, was "
        "replaced.",
        "End-to-end voice latency—the architecture—which was previously the target—"
        "was replaced.",
        "End-to-end voice latency: the design formerly served as the old constraint.",
        "End-to-end voice latency: the design, after review, was the old constraint.",
        "End-to-end voice latency—the architecture, which changed, was previously the "
        "target—was replaced.",
        "End-to-end voice latency: the design–after review–was the old constraint.",
        "End-to-end voice latency: the architecture—not the metric—was previously the "
        "target.",
    ),
)
def test_r33_foreign_which_antecedent_cannot_become_historical_latency_role(
    sentence,
):
    assert not _latency_dimension_asserted_historical(
        sentence,
        _END_TO_END_VOICE_LATENCY_RE,
    )


@pytest.mark.parametrize(
    ("sentence", "pattern", "predicate"),
    (
        (
            "End-to-end voice latency: the design changed, but it was the old "
            "requirement.",
            _END_TO_END_VOICE_LATENCY_RE,
            _latency_dimension_asserted_historical,
        ),
        (
            "End-to-end voice latency: the architecture changed, yet that was "
            "previously the target.",
            _END_TO_END_VOICE_LATENCY_RE,
            _latency_dimension_asserted_historical,
        ),
        (
            "First-token latency: the model changed, but it remains active.",
            _FIRST_TOKEN_LATENCY_RE,
            _latency_dimension_asserted_current,
        ),
        (
            "End-to-end voice latency: the latency model changed, but it was the old "
            "requirement.",
            _END_TO_END_VOICE_LATENCY_RE,
            _latency_dimension_asserted_historical,
        ),
        (
            "First-token latency: the metric model changed, but it remains active.",
            _FIRST_TOKEN_LATENCY_RE,
            _latency_dimension_asserted_current,
        ),
        (
            "End-to-end voice latency: the target evaluator changed, yet that was "
            "previously the target.",
            _END_TO_END_VOICE_LATENCY_RE,
            _latency_dimension_asserted_historical,
        ),
    ),
)
def test_r33_bare_pronoun_cannot_cross_foreign_intervening_antecedent(
    sentence,
    pattern,
    predicate,
):
    assert not predicate(sentence, pattern)


@pytest.mark.parametrize(
    "reassertion",
    (
        "but this measurement continues to be the active constraint",
        "but it has continued to be the active constraint",
        "yet that metric had stayed current",
        "however first-token latency still serves as the governing requirement",
        "but first-token latency has continued to serve as the current constraint",
        "yet this measurement has served as the active requirement",
    ),
)
def test_r33_continues_reassertion_is_not_hidden_as_subordinate_component(
    reassertion,
):
    sentence = (
        "First-token latency remains a component requirement under the governing "
        f"end-to-end voice latency constraint, {reassertion}."
    )
    assert _latency_dimension_asserted_current(sentence, _FIRST_TOKEN_LATENCY_RE)


def test_r31_subordinate_component_cannot_hide_active_constraint_reassertion(
    monkeypatch,
):
    contradictory = (
        "I revise the recommendation because the measurement boundary changed. "
        "The old constraint was first-token latency under four seconds; the new "
        "constraint is end-to-end voice latency under four seconds. First-token "
        "latency remains a component requirement under the governing end-to-end "
        "voice latency constraint, but it is also the active constraint. Face owns "
        "the foreground response while Depth runs asynchronously in the background. "
        "The measured delivery path includes ASR input, Face generation, TTS synthesis, "
        "audible playback, and release of the conversational turn. This sequence keeps "
        "heavy analysis outside the spoken critical path while retaining a complete "
        "written result in the same session. The operator hears a bounded recommendation, "
        "can interrupt playback, and does not need to ask again for the later diagnosis. "
        "The written channel retains the mechanism, uncertainty, practical consequence, "
        "and next action. That division matters because an early token alone does not "
        "prove the whole voice exchange completed within the deadline, whereas measuring "
        "input through playback covers the experience the operator actually waits for."
    )
    assert len(contradictory.split()) >= 130
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    session_id = f"r31-reassertion-{uuid.uuid4()}"
    state = face.get_or_create_session(session_id, "face-model")
    state.messages.extend(
        [
            {
                "role": "user",
                "content": (
                    "Assume first-token latency must stay under four seconds. "
                    "Revise the recommendation."
                ),
            },
            {"role": "assistant", "content": "Face should answer first."},
        ]
    )
    before = list(state.messages)
    calls = 0

    def post_streaming(_payload, callback, *, cancel_event=None):
        nonlocal calls
        del cancel_event
        calls += 1
        callback(contradictory)
        return contradictory, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        "No - I mean end-to-end voice latency. Correct the recommendation.",
        session_id=session_id,
        model="face-model",
        extra_system=_r2_verified_depth_context(),
        stream_callback=spoken.append,
    )

    _assert_gateway_authoritative_latency_correction(result, spoken)
    assert calls == 0
    assert contradictory not in result["text"]
    assert not _latency_dimension_asserted_current(
        result["text"],
        _FIRST_TOKEN_LATENCY_RE,
    )
    assert state.messages == before + [
        {
            "role": "user",
            "content": (
                "No - I mean end-to-end voice latency. Correct the recommendation."
            ),
        },
        {"role": "assistant", "content": result["text"]},
    ]


def test_r13_verified_completion_preserves_legitimate_past_job_attribution():
    historical = (
        "At 09:15, Depth Lobe job da-4410e950-e757-4a3c-842c-5b89d4ab4e8d "
        "was still running; it completed at 09:17, and its verified result informed "
        "the final architecture."
    )

    rewritten, applied = _rewrite_stale_verified_depth_status(
        historical,
        _r2_verified_depth_context(),
    )

    assert applied is False
    assert rewritten == historical


def test_r13_verified_completion_preserves_unrelated_future_benchmark_result():
    future_domain_result = "The benchmark result will appear in the report tomorrow."

    rewritten, applied = _rewrite_stale_verified_depth_status(
        future_domain_result,
        _r2_verified_depth_context(),
    )

    assert applied is False
    assert rewritten == future_domain_result


def test_r13_status_only_stale_ack_gets_truthful_terminal_replacement():
    stale_only = (
        "Dispatched Depth Lobe job da-4410e950-e757-4a3c-842c-5b89d4ab4e8d now. "
        "The verified result will appear automatically in this conversation when it "
        "is ready; you do not need to ask again."
    )

    rewritten, applied = _rewrite_stale_verified_depth_status(
        stale_only,
        _r2_verified_depth_context(),
    )

    assert applied is True
    assert rewritten == (
        "The Depth Lobe's verified result is already complete and present in "
        "this conversation."
    )


def _r14_completed_a_running_b_context() -> str:
    return (
        "MS4 Face Lobe context (authoritative; do not contradict):\n"
        "- THIS TURN DID NOT DISPATCH any background work.\n"
        "- verified completed Depth Lobe jobs "
        "(MOST RECENT FIRST; use these answers when relevant):\n"
        "    - da-aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa [completed] goal='A'\n"
        "      result: Verified result A is complete.\n"
        "- active or stale Double Agent jobs:\n"
        "    - da-bbbbbbbb-2222-4222-8222-bbbbbbbbbbbb [running - deep_chat]: B"
    )


def test_r14_completed_job_a_never_erases_truthful_running_job_b():
    running_b = (
        "Depth Lobe job da-bbbbbbbb-2222-4222-8222-bbbbbbbbbbbb is still running. "
        "The verified result will appear automatically in this conversation when it "
        "is ready; you do not need to ask again."
    )

    rewritten, applied = _rewrite_stale_verified_depth_status(
        running_b,
        _r14_completed_a_running_b_context(),
    )

    assert applied is False
    assert rewritten == running_b


def test_r14_completed_a_mentioned_in_same_sentence_does_not_erase_running_b():
    mixed_identity = (
        "Job da-aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa completed earlier, but job "
        "da-bbbbbbbb-2222-4222-8222-bbbbbbbbbbbb is still running."
    )

    rewritten, applied = _rewrite_stale_verified_depth_status(
        mixed_identity,
        _r14_completed_a_running_b_context(),
    )

    assert applied is False
    assert rewritten == mixed_identity


@pytest.mark.parametrize(
    "text",
    [
        (
            "Job da-bbbbbbbb-2222-4222-8222-bbbbbbbbbbbb is still running, while job "
            "da-aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa completed earlier."
        ),
        (
            "Job da-aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa completed earlier; whereas job "
            "da-bbbbbbbb-2222-4222-8222-bbbbbbbbbbbb is still running."
        ),
        (
            "Job da-bbbbbbbb-2222-4222-8222-bbbbbbbbbbbb is still running but job "
            "da-aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa completed earlier."
        ),
    ],
)
def test_r15_completed_a_clause_never_claims_running_b_clause(text):
    rewritten, applied = _rewrite_stale_verified_depth_status(
        text,
        _r14_completed_a_running_b_context(),
    )

    assert applied is False
    assert rewritten == text


@pytest.mark.parametrize(
    "text",
    [
        (
            "Job da-aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa is still running, while job "
            "da-bbbbbbbb-2222-4222-8222-bbbbbbbbbbbb is still running."
        ),
        (
            "Job da-bbbbbbbb-2222-4222-8222-bbbbbbbbbbbb is still running, while job "
            "da-aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa is still running."
        ),
        (
            "Job da-aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa is still running; whereas job "
            "da-bbbbbbbb-2222-4222-8222-bbbbbbbbbbbb is still running."
        ),
        (
            "Job da-bbbbbbbb-2222-4222-8222-bbbbbbbbbbbb is still running but job "
            "da-aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa is still running."
        ),
        (
            "Job da-aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa is still running — job "
            "da-bbbbbbbb-2222-4222-8222-bbbbbbbbbbbb is still running."
        ),
    ],
)
def test_r15_only_completed_a_stale_clause_is_removed_while_running_b_survives(text):
    running_b = "Job da-bbbbbbbb-2222-4222-8222-bbbbbbbbbbbb is still running."

    rewritten, applied = _rewrite_stale_verified_depth_status(
        text,
        _r14_completed_a_running_b_context(),
    )

    assert applied is True
    assert rewritten == running_b


@pytest.mark.parametrize(
    "text",
    [
        "Job da-aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa completed earlier.",
        (
            "Job da-aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa was still running during the "
            "capture, but job da-aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa completed later."
        ),
        "Job da-aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa is no longer running.",
        "Job da-aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa is not still running.",
    ],
)
def test_r15_same_completed_id_truthful_or_negated_history_is_not_removed(text):
    rewritten, applied = _rewrite_stale_verified_depth_status(
        text,
        _r14_completed_a_running_b_context(),
    )

    assert applied is False
    assert rewritten == text


@pytest.mark.parametrize(
    "text",
    [
        (
            "Job da-bbbbbbbb-2222-4222-8222-bbbbbbbbbbbb is still running: job "
            "da-aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa completed earlier."
        ),
        (
            "Job da-bbbbbbbb-2222-4222-8222-bbbbbbbbbbbb is still running although job "
            "da-aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa completed earlier."
        ),
        (
            "Job da-bbbbbbbb-2222-4222-8222-bbbbbbbbbbbb is still running though job "
            "da-aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa completed earlier."
        ),
        (
            "Job da-bbbbbbbb-2222-4222-8222-bbbbbbbbbbbb is still running yet job "
            "da-aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa completed earlier."
        ),
        (
            "Job da-bbbbbbbb-2222-4222-8222-bbbbbbbbbbbb is still running "
            "(although job da-aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa completed earlier)."
        ),
    ],
)
def test_r16_new_clause_boundaries_preserve_running_b_when_a_is_truthfully_complete(text):
    rewritten, applied = _rewrite_stale_verified_depth_status(
        text,
        _r14_completed_a_running_b_context(),
    )

    assert applied is False
    assert rewritten == text


@pytest.mark.parametrize(
    "text",
    [
        (
            "Job da-aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa is still running: job "
            "da-bbbbbbbb-2222-4222-8222-bbbbbbbbbbbb is still running."
        ),
        (
            "Job da-aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa is still running although job "
            "da-bbbbbbbb-2222-4222-8222-bbbbbbbbbbbb is still running."
        ),
        (
            "Job da-aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa is still running though job "
            "da-bbbbbbbb-2222-4222-8222-bbbbbbbbbbbb is still running."
        ),
        (
            "Job da-aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa is still running yet job "
            "da-bbbbbbbb-2222-4222-8222-bbbbbbbbbbbb is still running."
        ),
        (
            "Job da-aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa is still running "
            "(although job da-bbbbbbbb-2222-4222-8222-bbbbbbbbbbbb is still running)."
        ),
        (
            "Job da-bbbbbbbb-2222-4222-8222-bbbbbbbbbbbb is still running "
            "(although job da-aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa is still running)."
        ),
    ],
)
def test_r16_new_clause_boundaries_remove_only_stale_a_and_preserve_running_b(text):
    rewritten, applied = _rewrite_stale_verified_depth_status(
        text,
        _r14_completed_a_running_b_context(),
    )

    assert applied is True
    assert rewritten == "Job da-bbbbbbbb-2222-4222-8222-bbbbbbbbbbbb is still running."


def test_r16_mixed_three_job_sentence_removes_only_middle_stale_completed_a():
    text = (
        "Job da-bbbbbbbb-2222-4222-8222-bbbbbbbbbbbb is still running; job "
        "da-aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa is still running; job "
        "da-cccccccc-3333-4333-8333-cccccccccccc is still running."
    )
    expected = (
        "Job da-bbbbbbbb-2222-4222-8222-bbbbbbbbbbbb is still running; job "
        "da-cccccccc-3333-4333-8333-cccccccccccc is still running."
    )

    rewritten, applied = _rewrite_stale_verified_depth_status(
        text,
        _r14_completed_a_running_b_context(),
    )

    assert applied is True
    assert rewritten == expected


def test_r16_ambiguous_unsplit_multi_job_clause_is_preserved_fail_safe():
    ambiguous = (
        "Job da-aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa plus job "
        "da-bbbbbbbb-2222-4222-8222-bbbbbbbbbbbb is still running."
    )

    rewritten, applied = _rewrite_stale_verified_depth_status(
        ambiguous,
        _r14_completed_a_running_b_context(),
    )

    assert applied is False
    assert rewritten == ambiguous


@pytest.mark.parametrize(
    "connector",
    [
        " even though ",
        "; however, ",
        "; nevertheless, ",
        " although ",
        " yet ",
        " despite that ",
    ],
)
@pytest.mark.parametrize("stale_a_first", [True, False])
def test_r17_connector_grammar_removes_stale_a_without_orphan_tokens(
    connector,
    stale_a_first,
):
    stale_a = "Job da-aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa is still running"
    running_b = "job da-bbbbbbbb-2222-4222-8222-bbbbbbbbbbbb is still running"
    text = (
        f"{stale_a}{connector}{running_b}."
        if stale_a_first
        else f"{running_b.capitalize()}{connector}{stale_a.lower()}."
    )

    rewritten, applied = _rewrite_stale_verified_depth_status(
        text,
        _r14_completed_a_running_b_context(),
    )

    assert applied is True
    assert rewritten == "Job da-bbbbbbbb-2222-4222-8222-bbbbbbbbbbbb is still running."


@pytest.mark.parametrize(
    "connector",
    [
        " even though ",
        "; however, ",
        "; nevertheless, ",
        " although ",
        " yet ",
        " despite that ",
    ],
)
@pytest.mark.parametrize("completed_a_first", [True, False])
def test_r17_connector_grammar_preserves_running_b_and_truthful_completed_a(
    connector,
    completed_a_first,
):
    completed_a = "Job da-aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa completed earlier"
    running_b = "job da-bbbbbbbb-2222-4222-8222-bbbbbbbbbbbb is still running"
    text = (
        f"{completed_a}{connector}{running_b}."
        if completed_a_first
        else f"{running_b.capitalize()}{connector}{completed_a.lower()}."
    )

    rewritten, applied = _rewrite_stale_verified_depth_status(
        text,
        _r14_completed_a_running_b_context(),
    )

    assert applied is False
    assert rewritten == text


def test_r14_same_verified_completed_job_stale_running_status_is_removed():
    stale_a = (
        "Depth Lobe job da-aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa is still running. "
        "I'll surface its updated result when complete."
    )

    rewritten, applied = _rewrite_stale_verified_depth_status(
        stale_a,
        _r14_completed_a_running_b_context(),
    )

    assert applied is True
    assert rewritten == (
        "The Depth Lobe's verified result is already complete and present in "
        "this conversation."
    )


def test_r2_latency_correction_requires_a_revised_sequence_before_inference(fake_hivemind):
    fake_hivemind["script"]["per_model"] = {
        "nemotron-3-nano:4b": {"content": "A genuinely revised sequence."},
    }
    face = FaceLobeChat(hivemind_url=fake_hivemind["url"])
    state = face.get_or_create_session("r2-latency", "nemotron-3-nano:4b")
    state.messages.extend(
        [
            {
                "role": "user",
                "content": "Assume first-token latency must stay under four seconds. Revise the recommendation.",
            },
            {"role": "assistant", "content": "Use a fast first token."},
        ]
    )

    face.chat(
        (
            "No - I mean end-to-end voice latency under four seconds. "
            "Correct the recommendation."
        ),
        session_id="r2-latency",
        model="nemotron-3-nano:4b",
    )

    system = fake_hivemind["log"][-1]["messages"][0]["content"]
    assert "changed the constraint from first-token latency to end-to-end voice latency" in system
    assert "not a request to replace one label with another" in system
    assert "end of the operator's speech" in system
    assert "first useful non-silent reply onset" in system
    assert "Full audible utterance completion" in system
    assert "fan out Face and Depth concurrently" in system
    assert "numeric deadline remains under four seconds" in system
    assert "Do not invent component timings" in system


def test_r4_latency_correction_postcondition_names_old_new_and_deadline_before_tts(
    fake_hivemind,
):
    missing_old_constraint = (
        "Understood. You are constraining end-to-end voice latency to under four "
        "seconds. I revise the response-start path from the end of the operator's speech "
        "through ASR finalization, fast Face generation, and TTS to the first useful "
        "non-silent reply onset. Full audible utterance completion is tracked separately "
        "as an independent metric and is not subject to that same bound. Face must "
        "answer in the foreground while Depth runs asynchronously in the background, "
        "then its complete written diagnosis becomes available in the same session. "
    ) * 5
    fake_hivemind["script"]["per_model"] = {
        "qwen3.6:35b": {
            "stream": True,
            "chunks": [missing_old_constraint],
            "send_done": True,
        },
    }
    face = FaceLobeChat(hivemind_url=fake_hivemind["url"])
    state = face.get_or_create_session("r4-latency-postcondition", "qwen3.6:35b")
    state.messages.extend(
        [
            {
                "role": "user",
                "content": (
                    "Assume first-token latency must stay under four seconds. "
                    "Revise the recommendation."
                ),
            },
            {"role": "assistant", "content": "Use a fast opening."},
        ]
    )
    spoken: list[str] = []

    result = face.chat(
        (
            "No - I mean end-to-end voice latency under four seconds. "
            "Correct the recommendation."
        ),
        session_id="r4-latency-postcondition",
        model="qwen3.6:35b",
        extra_system=_r2_verified_depth_context(),
        stream_callback=spoken.append,
    )

    contract = (
        "the old constraint was first-token latency under four seconds; "
        "the new constraint is end-to-end voice latency under four seconds"
    )
    assert contract in result["text"]
    assert result["text"].endswith(missing_old_constraint.strip())
    assert spoken == [result["text"]], "the corrected text must be the only TTS stream emission"
    assert state.messages[-1] == {"role": "assistant", "content": result["text"]}
    assert result["output_guard"]["reason"] == "latency_constraint_correction_postcondition"
    postcondition = result["output_guard"]["latency_constraint_postcondition"]
    assert postcondition == {
        "schema": "Ms4LatencyConstraintPostcondition.v1",
        "applied": True,
        "old_constraint_present": False,
        "new_constraint_present": True,
        "deadline_present": True,
        "old_deadline_present": False,
        "new_deadline_present": True,
        "old_deadline_contradiction": False,
        "new_deadline_contradiction": False,
        "unbound_deadline_contradiction": False,
        "constraint_level_deadline_present": True,
        "constraint_level_deadline_values": [4000],
        "canonical_deadline_present": False,
        "canonical_deadline_values": [],
        "duration_values": [4000],
        "duration_occurrence_count": 5,
            "unexpected_duration_values": [],
            "residual_duration_syntax_present": True,
            "deadline_disavowal_present": False,
        "authoritative_deadline_ready": False,
        "direction": "missing",
        "new_dimension_rejected": False,
        "new_dimension_historical": False,
        "old_dimension_current": False,
        "final_synthesis": False,
        "only_repairable_failures": True,
        "failures_before": ["latency_constraint_contrast_missing"],
        "repaired_word_count": 425,
        "max_words": None,
        "deadline": "under four seconds",
    }


def test_r4_complete_latency_correction_is_buffered_but_not_rewritten(fake_hivemind):
    complete = _r13_semantic_answer("full_voice").strip()
    fake_hivemind["script"]["per_model"] = {
        "qwen3.6:35b": {
            "stream": True,
            "chunks": [complete],
            "send_done": True,
        },
    }
    face = FaceLobeChat(hivemind_url=fake_hivemind["url"])
    state = face.get_or_create_session("r4-latency-complete", "qwen3.6:35b")
    state.messages.extend(
        [
            {
                "role": "user",
                "content": "First-token latency must stay under four seconds.",
            },
            {"role": "assistant", "content": "Initial architecture."},
        ]
    )
    spoken: list[str] = []

    result = face.chat(
        "No, I mean end-to-end voice latency under four seconds. Correct it.",
        session_id="r4-latency-complete",
        model="qwen3.6:35b",
        stream_callback=spoken.append,
    )

    assert result["text"] == complete
    assert spoken == [complete]
    assert result["output_guard"]["applied"] is False
    assert result["output_guard"]["latency_constraint_postcondition"]["applied"] is False


@pytest.mark.parametrize(
    "budget_sentence",
    [
        (
            "In the current voice pipeline, ASR has a component budget under one "
            "second, the overall voice deadline is under four seconds, and TTS plus "
            "audible playback has a component budget under one second. Face generation "
            "has a component budget under two seconds."
        ),
        (
            "Component budgets are ASR under one second, Face under two seconds, and "
            "TTS under one second, while the overall voice deadline is under four seconds."
        ),
        (
            "The ASR component budget, not the overall voice deadline, is under one "
            "second."
        ),
        (
            "The overall voice deadline doesn't apply and would otherwise be under "
            "five seconds."
        ),
        (
            "The overall voice deadline doesn’t apply and would otherwise be under "
            "five seconds."
        ),
        (
            "The overall voice deadline doesn't currently apply and would otherwise "
            "be under five seconds."
        ),
        (
            "The overall voice deadline doesn’t currently apply and would otherwise "
            "be under five seconds."
        ),
        (
            "The overall voice deadline is currently not applicable and would "
            "otherwise be under five seconds."
        ),
        (
            "The overall voice deadline doesn't really apply and would otherwise be "
            "under five seconds."
        ),
        (
            "The overall voice deadline doesn’t really apply and would otherwise be "
            "under five seconds."
        ),
        (
            "Voice deadline: currently not applicable and would otherwise be under "
            "five seconds."
        ),
        (
            "The overall voice deadline just doesn't apply and would otherwise be "
            "under five seconds."
        ),
    ],
)
def test_r27_component_budgets_do_not_contradict_the_overall_voice_deadline(
    monkeypatch,
    budget_sentence,
):
    candidate = (
        "The constraint changed from first-token latency to end-to-end voice latency. "
        f"{budget_sentence} Those are component allocations inside the overall path, not competing "
        "turn-level constraints. Face gives the immediate foreground response while "
        "Depth runs asynchronously in the background and delivers verified analysis "
        "later. This sequence keeps synchronous Depth work outside the spoken critical "
        "path, preserves a useful answer now, and returns the complete written diagnosis "
        "into the same conversation when it is ready. The operator can interrupt speech, "
        "continue naturally, and inspect the later evidence without launching a duplicate "
        "job. The response-start SLO runs from the end of the operator's speech through "
        "ASR finalization, Face generation, and TTS to the first useful non-silent reply "
        "onset. Full audible utterance completion is tracked separately as an independent "
        "metric and is not subject to that same bound. The internal component budgets explain how the "
        "system intends to stay responsive without pretending those allocations are "
        "independently measured performance results."
    )
    assert len(candidate.split()) >= 130
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session(
        f"r27-component-budgets-{uuid.uuid4()}",
        "face-model",
    )
    state.messages.extend(
        [
            {
                "role": "user",
                "content": "Assume first-token latency must stay under four seconds.",
            },
            {"role": "assistant", "content": "Use a fast opening from Face."},
        ]
    )

    def post_streaming(_payload, callback, *, cancel_event=None):
        del cancel_event
        callback(candidate)
        return candidate, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        (
            "No - I mean end-to-end voice latency under four seconds. "
            "Correct the recommendation."
        ),
        session_id=state.session_id,
        model="face-model",
        stream_callback=spoken.append,
    )

    semantic = result["output_guard"]["substantive_followup"]
    receipt = result["output_guard"]["latency_constraint_postcondition"]
    assert result["completed"] is True
    assert result["api_calls"] == 1
    assert receipt["applied"] is True
    assert receipt["unbound_deadline_contradiction"] is False
    assert "unbound_latency_deadline_contradiction" not in semantic[
        "first_candidate_failures"
    ]
    assert result["text"].startswith(
        "Correction: the old constraint was first-token latency under four seconds; "
        "the new constraint is end-to-end voice latency under four seconds."
    )
    assert spoken == [result["text"]]
    assert state.messages[-1] == {"role": "assistant", "content": result["text"]}


def test_final_architecture_accepts_one_canonical_deadline_without_gateway_rewrite(
    monkeypatch,
):
    missing_old_deadline = (
        "Constraint changed: the operator shifted from first-token latency to "
        "end-to-end voice latency. The final architecture is plain: Face handles the immediate voice "
        "path, beginning with ASR input and producing the foreground answer for TTS and "
        "audible playback. Depth runs asynchronously in the background, analyzes the "
        "same evidence, and returns verified findings later without blocking speech. "
        "This ordering keeps the conversation responsive because the heavy diagnosis is "
        "outside the spoken critical path while the eventual result remains part of the "
        "same session. Face gives the best bounded answer available now, identifies any "
        "uncertainty, and speaks it; Depth then checks logs, traces, and cross-node state. "
        "If that later evidence changes the diagnosis, Face explains the correction "
        "directly. The practical architecture is therefore Face to voice first, Depth in "
        "parallel second, and verified integration afterward, with no synchronous Depth "
        "wait before the operator hears a useful response. The active end-to-end "
        "voice-latency deadline is under four seconds."
    )
    assert len(missing_old_deadline.split()) >= 120
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("final-exact-constraint", "face-model")
    state.messages.extend(
        [
            {
                "role": "user",
                "content": "Assume first-token latency must stay under four seconds.",
            },
            {"role": "assistant", "content": "Face answers first; Depth follows."},
            {
                "role": "user",
                "content": "No - I mean end-to-end voice latency. Correct it.",
            },
            {"role": "assistant", "content": "The voice pipeline is now the bound."},
        ]
    )
    calls = 0

    def post_streaming(_payload, callback, *, cancel_event=None):
        nonlocal calls
        del cancel_event
        calls += 1
        callback(missing_old_deadline)
        return missing_old_deadline, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        "Give me the final architecture in plain English, and tell me what constraint I changed.",
        session_id="final-exact-constraint",
        model="face-model",
        extra_system=_r2_verified_depth_context(),
        stream_callback=spoken.append,
    )

    assert calls == 0
    _assert_gateway_authoritative_latency_synthesis(result, spoken)
    assert missing_old_deadline not in result["text"]
    assert state.messages[-1] == {"role": "assistant", "content": result["text"]}


def test_final_architecture_retries_instead_of_composing_authority_into_model_prose(
    monkeypatch,
):
    candidate = (
        "The constraint changed from first-token latency to end-to-end voice latency. "
        "The active architecture keeps end-to-end voice latency as the governing target. "
        "That is the changed constraint. "
        "A foreground responder gives a concise answer while a separate background "
        "analyzer examines the distributed evidence. This ordering matters because "
        "heavy causal analysis must not block the spoken interaction, while the same "
        "conversation still needs the verified diagnosis after that analysis finishes. "
        "The foreground response should state the best bounded recommendation and its "
        "uncertainty. The background analysis should compare logs, traces, scheduler "
        "state, and cross-node behavior, then integrate any correction clearly. The "
        "practical result is a responsive conversation that does not discard diagnostic "
        "quality. This architecture also keeps the timing boundary observable, so an "
        "operator can measure the complete interaction and reject a configuration that "
        "misses the active deadline. Therefore the immediate path stays bounded while "
        "the evidence-intensive work remains non-blocking and accountable. The active "
        "end-to-end voice-latency deadline is under four seconds."
    )
    assert len(candidate.split()) >= 130
    corrected = _r13_semantic_answer("full_voice", final_authoritative=True).strip()
    attempts = iter((candidate, corrected))
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("final-composed-latency", "face-model")
    state.messages.extend(
        [
            {
                "role": "user",
                "content": "Assume first-token latency must stay under four seconds.",
            },
            {"role": "assistant", "content": "Use a fast foreground response."},
            {
                "role": "user",
                "content": "No - I mean end-to-end voice latency. Correct it.",
            },
            {"role": "assistant", "content": "The complete voice path is now the bound."},
        ]
    )
    calls = 0

    def post_streaming(_payload, callback, *, cancel_event=None):
        nonlocal calls
        del cancel_event
        calls += 1
        answer = next(attempts)
        callback(answer)
        return answer, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        "Give me the final architecture in plain English, and tell me what constraint I changed.",
        session_id=state.session_id,
        model="face-model",
        stream_callback=spoken.append,
    )

    assert calls == 0
    _assert_gateway_authoritative_latency_synthesis(result, spoken)
    assert candidate not in result["text"]
    assert corrected not in result["text"]
    assert "latency_lobe_staging_postcondition" not in result["output_guard"]
    assert "latency_delivery_sequence_postcondition" not in result["output_guard"]
    assert state.messages[-1] == {"role": "assistant", "content": result["text"]}


def _r25_final_architecture_with_constraint(
    statement: str,
    *,
    statement_at_end: bool = False,
) -> str:
    architecture = (
        "The final architecture is plain: Face handles the immediate "
        "foreground voice response and begins useful speech without waiting for the "
        "heavier analysis. Depth runs asynchronously in the background, examines logs, "
        "traces, scheduler state, and cross-node evidence, then returns its verified "
        "findings later in the same session. The measured spoken path includes ASR input, "
        "Face generation, TTS synthesis, and audible playback. This ordering keeps the "
        "conversation responsive while preserving a complete written diagnosis. Face "
        "states uncertainty and the best bounded recommendation now; Depth checks the "
        "causal chain and can trigger a clearly explained correction afterward. The user "
        "does not need to ask again, and synchronous Depth work never blocks the initial "
        "voice path. The practical sequence is therefore Face to voice first, Depth in "
        "parallel second, and verified integration into the same conversation when ready."
    )
    answer = (
        f"{architecture} {statement}" if statement_at_end else f"{statement} {architecture}"
    )
    assert len(answer.split()) >= 130
    return answer


@pytest.mark.parametrize(
    "deadline_statement",
    [
        "The overall voice deadline is under four seconds.",
        (
            "The overall voice deadline, not the ASR component budget, is under "
            "four seconds."
        ),
        (
            "The overall voice deadline is not an ASR component budget and remains "
            "under four seconds."
        ),
        "The overall voice deadline must remain under four seconds.",
        "The overall voice deadline should stay under four seconds.",
    ],
)
def test_final_architecture_binds_affirmative_turn_level_deadline(
    monkeypatch,
    deadline_statement,
):
    answer = _r25_final_architecture_with_constraint(
        "The constraint changed from first-token latency to end-to-end voice latency. "
        f"{deadline_statement}",
        statement_at_end=True,
    )
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("final-turn-level-deadline", "face-model")
    state.messages.extend(
        [
            {
                "role": "user",
                "content": "Assume first-token latency must stay under four seconds.",
            },
            {"role": "assistant", "content": "Face answers first; Depth follows."},
            {
                "role": "user",
                "content": "No - I mean end-to-end voice latency. Correct it.",
            },
            {"role": "assistant", "content": "The voice pipeline is now the bound."},
        ]
    )
    calls = 0

    def post_streaming(_payload, callback, *, cancel_event=None):
        nonlocal calls
        del cancel_event
        calls += 1
        callback(answer)
        return answer, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        "Give me the final architecture in plain English, and tell me what constraint I changed.",
        session_id=state.session_id,
        model="face-model",
        stream_callback=spoken.append,
    )

    assert calls == 0
    _assert_gateway_authoritative_latency_synthesis(result, spoken)
    assert answer not in result["text"]
    assert state.messages[-1] == {"role": "assistant", "content": result["text"]}


@pytest.mark.parametrize(
    "statement",
    [
        "I do not agree that the overall voice deadline is under four seconds.",
        "It is not the case that the overall voice deadline is under four seconds.",
        "This does not mean the overall voice deadline is under four seconds.",
        "There is no evidence that the overall voice deadline is under four seconds.",
        "Nobody established that the overall voice deadline is under four seconds.",
        "Neither test proves that the overall voice deadline is under four seconds.",
        "It would be false to assume the overall voice deadline is under four seconds.",
        "Example: the overall voice deadline is under four seconds.",
        "The old note reads: the overall voice deadline is under four seconds.",
        "The inscription says the overall voice deadline is under four seconds.",
        (
            "The words the overall voice deadline is under four seconds appear in "
            "the prompt."
        ),
        "Alice says the overall voice deadline is under four seconds.",
        "According to Alice, the overall voice deadline is under four seconds.",
        "The team believes the overall voice deadline is under four seconds.",
        "The documentation states that the overall voice deadline is under four seconds.",
        "Rumor has it that the overall voice deadline is under four seconds.",
        "The model claims the overall voice deadline is under four seconds.",
        "We heard that the overall voice deadline is under four seconds.",
        "The ticket says the overall voice deadline is under four seconds.",
        "In a hypothetical where the overall voice deadline is under four seconds.",
        "In the counterfactual where the overall voice deadline is under four seconds.",
        "Were it true that the overall voice deadline is under four seconds.",
        (
            "The overall voice deadline is under four seconds only if the faster "
            "model ships."
        ),
        (
            "When the overall voice deadline is under four seconds, the status light "
            "turns green."
        ),
        "It used to be true that the overall voice deadline is under four seconds.",
        "The former policy says the overall voice deadline is under four seconds.",
        (
            "The overall voice deadline was under four seconds before it was replaced."
        ),
        "In the draft, the overall voice deadline is under four seconds.",
        "The obsolete policy says the overall voice deadline is under four seconds.",
        (
            "The previous requirement was that the overall voice deadline is under "
            "four seconds."
        ),
        (
            "The overall voice deadline is under four seconds, but it was replaced by "
            "a five-second limit."
        ),
        "The prior baseline says the overall voice deadline is under four seconds.",
        (
            "The overall voice deadline is under four seconds only in the retired build."
        ),
        "Within the overall voice pipeline, ASR latency is under four seconds.",
        "For the full-path voice test, TTS latency is under four seconds.",
        "In the overall voice architecture, the playback step is under four seconds.",
        "The overall voice deadline for ASR is under four seconds.",
        "Within the overall voice system, Face generation is under four seconds.",
        "The voice target for speech recognition is under four seconds.",
        "Overall target: ASR should stay under four seconds.",
        (
            "The overall voice deadline was under four seconds; it is now under five "
            "seconds."
        ),
        (
            "The overall voice deadline was under four seconds before changing to "
            "five seconds."
        ),
        (
            "The overall voice deadline is under four seconds, but the actual target "
            "is five seconds."
        ),
        (
            "The overall voice deadline is under four seconds only in mock mode; "
            "production uses a five-second limit."
        ),
        (
            "The old overall voice deadline was under four seconds. It is now under "
            "five seconds."
        ),
        "The overall voice deadline was under four seconds until yesterday.",
        "Maybe the overall voice deadline is under four seconds.",
        (
            "Either the overall voice deadline is under four seconds or there is no "
            "deadline."
        ),
        "Not sure, but the overall voice deadline is under four seconds.",
        "The overall voice deadline is under four seconds or under five seconds.",
        "The overall voice deadline is under four seconds",
        "The overall voice deadline is under four seconds?",
        "The overall voice deadline is under four seconds!",
        "The overall voice deadline is under four seconds...",
        "`The overall voice deadline is under four seconds.`",
        "- The overall voice deadline is under four seconds.",
        "# The overall voice deadline is under four seconds.",
        "The overall voice deadline is under four seconds — supposedly.",
        "The overall voice deadline is under four seconds. That statement is false.",
        "The overall voice deadline is under four seconds. It no longer applies.",
        (
            "The overall voice deadline is under four seconds. Correction: it is under "
            "five seconds."
        ),
        "The overall voice deadline is under four seconds. No, under five seconds.",
        "The overall voice deadline is under four seconds; no, under five seconds.",
    ],
)
def test_r27_canonical_deadline_admission_rejects_red_team_non_assertions(statement):
    assert _canonical_turn_level_deadline_matches(statement) == []


def test_r27_canonical_final_deadline_cannot_mask_another_duration():
    answer = _r25_final_architecture_with_constraint(
        "The constraint changed from first-token latency to end-to-end voice latency. "
        "The actual target is five seconds. The active end-to-end voice-latency "
        "deadline is under four seconds.",
        statement_at_end=True,
    )
    corrected, receipt = _enforce_latency_correction_postcondition(
        answer,
        message=(
            "Give me the final architecture in plain English, and tell me what "
            "constraint I changed."
        ),
        history=[
            {
                "role": "user",
                "content": "Assume first-token latency must stay under four seconds.",
            },
            {"role": "assistant", "content": "Face answers first; Depth follows."},
            {
                "role": "user",
                "content": "No - I mean end-to-end voice latency. Correct it.",
            },
            {"role": "assistant", "content": "The voice pipeline is now the bound."},
        ],
        failures=["latency_deadline_missing", "prior_latency_deadline_missing"],
        max_words=600,
    )

    assert corrected == answer
    assert receipt is not None
    assert receipt["applied"] is False
    assert receipt["canonical_deadline_present"] is True
    assert receipt["canonical_deadline_values"] == [4000]
    assert receipt["duration_values"] == [4000, 5000]
    assert receipt["duration_occurrence_count"] == 2
    assert receipt["unexpected_duration_values"] == [5000]
    assert receipt["authoritative_deadline_ready"] is False


def test_r27_canonical_final_deadline_cannot_mask_prior_disavowal():
    answer = _r25_final_architecture_with_constraint(
        "The constraint changed from first-token latency to end-to-end voice latency. "
        "The operator has not adopted any end-to-end voice deadline. The active "
        "end-to-end voice-latency deadline is under four seconds.",
        statement_at_end=True,
    )
    corrected, receipt = _enforce_latency_correction_postcondition(
        answer,
        message=(
            "Give me the final architecture in plain English, and tell me what "
            "constraint I changed."
        ),
        history=[
            {
                "role": "user",
                "content": "Assume first-token latency must stay under four seconds.",
            },
            {"role": "assistant", "content": "Face answers first; Depth follows."},
            {
                "role": "user",
                "content": "No - I mean end-to-end voice latency. Correct it.",
            },
            {"role": "assistant", "content": "The voice pipeline is now the bound."},
        ],
        failures=["latency_deadline_missing", "prior_latency_deadline_missing"],
        max_words=600,
    )

    assert corrected == answer
    assert receipt is not None
    assert receipt["applied"] is False
    assert receipt["canonical_deadline_present"] is True
    assert receipt["unexpected_duration_values"] == []
    assert receipt["deadline_disavowal_present"] is True


@pytest.mark.parametrize(
    "fragment",
    [
        "Hypothetical scenario. The overall voice deadline is under four seconds.",
        (
            "The preceding analysis is only an example. The overall voice deadline is "
            "under four seconds."
        ),
        (
            "The following sentence is an obsolete quote. The overall voice deadline "
            "is under four seconds."
        ),
        (
            "Do not treat the next sentence as a requirement. The overall voice "
            "deadline is under four seconds."
        ),
        "The next sentence is false. The overall voice deadline is under four seconds.",
        (
            "The operator rejected the statement that follows. The overall voice "
            "deadline is under four seconds."
        ),
        (
            "The statement below applies only to ASR. The overall voice deadline is "
            "under four seconds."
        ),
        "Example:\nThe overall voice deadline is under four seconds.",
        "Old prompt:\nThe overall voice deadline is under four seconds.",
        "Unverified:\nThe overall voice deadline is under four seconds.",
        "Quoted material follows.\n\nThe overall voice deadline is under four seconds.",
        (
            "The four-second constraint is superseded. The overall voice deadline is "
            "under four seconds."
        ),
        (
            "There is no overall voice deadline; only ASR must finish within four "
            "seconds. The overall voice deadline is under four seconds."
        ),
        (
            "The four-second figure applies only to ASR, not the overall path. The "
            "overall voice deadline is under four seconds."
        ),
        (
            "ASR has a four-second budget; no full-path requirement exists. The overall "
            "voice deadline is under four seconds."
        ),
        (
            "Only TTS is constrained to four seconds. The overall voice deadline is "
            "under four seconds."
        ),
        (
            "The next sentence applies to the ASR component only. The overall voice "
            "deadline is under four seconds."
        ),
        (
            "The full path has no deadline. ASR alone must stay under four seconds. The "
            "overall voice deadline is under four seconds."
        ),
        (
            "If the prior constraint was first-token latency under four seconds, the "
            "architecture would work."
        ),
        (
            "The prior constraint was first-token latency under four seconds, but that "
            "statement is false."
        ),
        (
            '"The prior constraint was first-token latency under four seconds." This '
            "is only a quote."
        ),
        "The old prompt says first-token latency must stay under four seconds.",
        "The retired build required first-token latency under four seconds.",
        "First-token latency under four seconds was only a component estimate.",
        "End-to-end voice latency under four seconds was merely hypothetical.",
        "The old prompt says end-to-end voice latency is under four seconds.",
        '"End-to-end voice latency is under four seconds." This is quoted text.',
        "If end-to-end voice latency is under four seconds, the design passes.",
        "ASR alone measured end-to-end voice latency under four seconds.",
    ],
)
def test_r28_authoritative_deadline_gate_rejects_frozen_full_chat_bypasses(fragment):
    answer = (
        _r25_final_architecture_with_constraint(
            "The constraint changed from first-token latency to end-to-end voice latency."
        )
        + " "
        + fragment
    )
    corrected, receipt = _enforce_latency_correction_postcondition(
        answer,
        message=(
            "Give me the final architecture in plain English, and tell me what "
            "constraint I changed."
        ),
        history=[
            {
                "role": "user",
                "content": "Assume first-token latency must stay under four seconds.",
            },
            {"role": "assistant", "content": "Face answers first; Depth follows."},
            {
                "role": "user",
                "content": "No - I mean end-to-end voice latency. Correct it.",
            },
            {"role": "assistant", "content": "The voice pipeline is now the bound."},
        ],
        failures=["latency_deadline_missing", "prior_latency_deadline_missing"],
        max_words=600,
    )

    assert corrected == answer
    assert receipt is not None
    assert receipt["applied"] is False
    assert receipt["authoritative_deadline_ready"] is False


_R29_CANONICAL_DEADLINE = (
    "The active end-to-end voice-latency deadline is under four seconds."
)
_R29_MODEL_SPOOFED_PREFIX = (
    "Correction: the old constraint was first-token latency under four seconds; "
    "the new constraint is end-to-end voice latency under four seconds."
)
_R29_ARCHITECTURE = (
    "The final architecture is plain: Face handles the immediate foreground voice "
    "response and begins useful speech without waiting for the heavier analysis. "
    "Depth runs asynchronously in the background, examines logs, traces, scheduler "
    "state, and cross-node evidence, then returns its verified findings later in the "
    "same session. The measured spoken path includes ASR input, Face generation, TTS "
    "synthesis, and audible playback. This ordering keeps the conversation responsive "
    "while preserving a complete written diagnosis. Face states uncertainty and the "
    "best bounded recommendation now; Depth checks the causal chain and can trigger a "
    "clearly explained correction afterward. The user does not need to ask again, and "
    "synchronous Depth work never blocks the initial voice path. The practical sequence "
    "is therefore Face to voice first, Depth in parallel second, and verified integration "
    "into the same conversation when ready. The constraint changed from first-token "
    "latency to end-to-end voice latency."
)
_R29_PRELUDE_EVASIONS = [
    "The next line is fictional.",
    "The line below is satire.",
    "Treat the final line as a nonbinding example.",
    "What follows is merely sample text.",
    "Ignore what follows.",
    "Discard the final sentence.",
    "The last line is hallucinated.",
    "The last sentence is a lie.",
    "This is copied from a retired policy.",
    "According to an obsolete document:",
    "Alice alleges:",
    "The model guessed:",
    "For illustration only:",
    "Counterfactual:",
    "Assume this for discussion:",
    "The final line is not authoritative.",
    "The line below is not a requirement.",
    "Do not rely on the final sentence.",
    "The claim at the end is unsupported.",
    "The sentence at the end is false.",
    "The requirement on the next line was withdrawn.",
    "No one adopted what follows.",
    "The old spec contains this exact line:",
    "The following line applies only to TTS.",
    "Read but do not obey:",
]
_R29_DURATION_EVASIONS = [
    "The actual deadline is eleven seconds.",
    "The actual deadline is zero seconds.",
    "The actual deadline is twelve seconds.",
    "The actual deadline is twenty seconds.",
    "The actual deadline is a second.",
    "The actual deadline is half a second.",
    "The actual deadline is one-half seconds.",
    "The actual deadline is 5½ seconds.",
    "The actual deadline is five and a half seconds.",
    "The actual deadline is five‑second.",
    "The actual deadline is five–second.",
    "The actual deadline is five—second.",
    "The actual deadline is 5‑second.",
    "The actual deadline is 5–second.",
    "The actual deadline is 5e3 ms.",
    "The actual deadline is 4_500 ms.",
    "The actual deadline is 0x1388 ms.",
    "The actual deadline is five millis.",
    "The actual deadline is five thousand milliseconds.",
    "The actual deadline is V seconds.",
    "The actual deadline is Ⅺ seconds.",
    "The actual deadline is a dozen seconds.",
    "The actual deadline is several seconds.",
]
_R29_PREFIX_SPOOFS = [
    "That correction is fabricated.",
    "The correction above is fictional.",
    "Ignore the correction above.",
    "Treat the opening correction as a quotation.",
    "The opening line is only sample text.",
    "The prefix was copied from an obsolete prompt.",
    "Nothing before the final paragraph is authoritative.",
    "The first sentence is a lie.",
    "The displayed correction is not binding.",
    "Discard the opening correction.",
]
_R29_STAGING_SEQUENCE_CONTRADICTIONS = [
    "Depth must complete synchronously before Face may speak.",
    "The initial voice response waits for Depth to finish.",
    "Face is forbidden from speaking until Depth completes.",
    "Depth sits on the foreground critical path and blocks the initial response.",
    "The design serializes Face after Depth.",
    (
        "The measured path excludes ASR input, Face generation, TTS synthesis, "
        "and audible playback."
    ),
    (
        "None of ASR input, Face generation, TTS synthesis, or playback belongs "
        "to the measured path."
    ),
    (
        "Only ASR is counted; Face generation, TTS synthesis, and playback are "
        "out of scope."
    ),
    "TTS synthesis and audible playback are explicitly omitted from timing.",
    "The latency metric stops before TTS synthesis and playback.",
]
_R29_POSTCONDITION_LAUNDERING = (
    "The final architecture serializes the two lobes because certainty is preferred "
    "over responsiveness. Depth completes its investigation first. Face begins only "
    "once that work is finished. The operator hears no preliminary answer during the "
    "investigation. This design treats the initial silence as acceptable and presents "
    "one definitive voice answer. The constraint changed from first-token latency to "
    "end-to-end voice latency. The system gathers evidence, compares the likely causes, "
    "reviews the relevant state, and resolves disagreements before a response begins. "
    "That ordering avoids a provisional recommendation and ensures that every spoken "
    "claim reflects the completed investigation. It also means the user receives the "
    "diagnosis and recommendation together instead of hearing an early answer followed "
    "by an update. Operators can inspect a component record for troubleshooting, while "
    "the ordinary conversation remains a single completed result. The approach values a "
    "stable answer, a clear explanation, and consistent behavior across the cluster. It "
    "deliberately makes the deeper investigation part of what happens before speech "
    "begins, so the visible response cannot race ahead of the analysis."
)
_R29_ADVERSARIAL_CANDIDATES = [
    *[
        (f"prelude-{index:02d}", f"{_R29_ARCHITECTURE} {payload}\n{_R29_CANONICAL_DEADLINE}")
        for index, payload in enumerate(_R29_PRELUDE_EVASIONS, 1)
    ],
    *[
        (f"duration-{index:02d}", f"{_R29_ARCHITECTURE} {payload} {_R29_CANONICAL_DEADLINE}")
        for index, payload in enumerate(_R29_DURATION_EVASIONS, 1)
    ],
    *[
        (
            f"prefix-{index:02d}",
            f"{_R29_MODEL_SPOOFED_PREFIX} {payload} {_R29_ARCHITECTURE} "
            f"{_R29_CANONICAL_DEADLINE}",
        )
        for index, payload in enumerate(_R29_PREFIX_SPOOFS, 1)
    ],
    *[
        (
            f"staging-{index:02d}",
            f"{_R29_ARCHITECTURE} {payload} {_R29_CANONICAL_DEADLINE}",
        )
        for index, payload in enumerate(_R29_STAGING_SEQUENCE_CONTRADICTIONS, 1)
    ],
    ("postcondition-laundering", _R29_POSTCONDITION_LAUNDERING),
]


@pytest.mark.parametrize(
    ("case_id", "candidate"),
    _R29_ADVERSARIAL_CANDIDATES,
    ids=[case_id for case_id, _candidate in _R29_ADVERSARIAL_CANDIDATES],
)
def test_r29_red_team_matrix_cannot_cross_gateway_owned_synthesis_boundary(
    monkeypatch,
    case_id,
    candidate,
):
    del case_id
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    session_id = f"r29-red-team-{uuid.uuid4()}"
    state = face.get_or_create_session(session_id, "face-model")
    state.messages.extend(
        [
            {
                "role": "user",
                "content": "Assume first-token latency must stay under four seconds.",
            },
            {"role": "assistant", "content": "Face answers first; Depth follows."},
            {
                "role": "user",
                "content": "No - I mean end-to-end voice latency. Correct it.",
            },
            {"role": "assistant", "content": "The voice pipeline is now the bound."},
        ]
    )
    before = list(state.messages)
    calls = 0

    def post_streaming(_payload, callback, *, cancel_event=None):
        nonlocal calls
        del cancel_event
        calls += 1
        callback(candidate)
        return candidate, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        "Give me the final architecture in plain English, and tell me what constraint I changed.",
        session_id=session_id,
        model="face-model",
        stream_callback=spoken.append,
    )

    assert calls == 0
    _assert_gateway_authoritative_latency_synthesis(result, spoken)
    assert candidate not in result["text"]
    assert state.messages == before + [
        {
            "role": "user",
            "content": (
                "Give me the final architecture in plain English, and tell me what "
                "constraint I changed."
            ),
        },
        {"role": "assistant", "content": result["text"]},
    ]


_R30_CANONICAL_OPERATOR_EPISODE = [
    {
        "role": "user",
        "content": "Assume first-token latency must stay under four seconds.",
    },
    {"role": "assistant", "content": "Face answers first; Depth follows."},
    {
        "role": "user",
        "content": "No - I mean end-to-end voice latency. Correct it.",
    },
    {"role": "assistant", "content": "The voice pipeline is now the bound."},
]


@pytest.mark.parametrize(
    ("history", "prompt", "deadline", "deadline_milliseconds", "target"),
    [
        (
            _R30_CANONICAL_OPERATOR_EPISODE,
            "Give me the final architecture in plain English, and tell me what constraint I changed.",
            "under four seconds",
            4000,
            "architecture",
        ),
        (
            _R30_CANONICAL_OPERATOR_EPISODE,
            "Give me the final Oracle voice design in plain English.",
            "under four seconds",
            4000,
            "design",
        ),
        (
            _R30_CANONICAL_OPERATOR_EPISODE,
            "Give me the final Face lobe and Depth lobe plan.",
            "under four seconds",
            4000,
            "plan",
        ),
        (
            _R30_CANONICAL_OPERATOR_EPISODE,
            "Show me the final system architecture for Oracle voice.",
            "under four seconds",
            4000,
            "architecture",
        ),
        (
            _R30_CANONICAL_OPERATOR_EPISODE,
            "Summarize the architecture we settled on.",
            "under four seconds",
            4000,
            "architecture",
        ),
        (
            _R30_CANONICAL_OPERATOR_EPISODE,
            "Could you give me the final voice architecture?",
            "under four seconds",
            4000,
            "architecture",
        ),
        (
            _R30_CANONICAL_OPERATOR_EPISODE,
            "Can you show me the final Oracle voice design?",
            "under four seconds",
            4000,
            "design",
        ),
        (
            _R30_CANONICAL_OPERATOR_EPISODE,
            "I'd like the final voice plan.",
            "under four seconds",
            4000,
            "plan",
        ),
        (
            _R30_CANONICAL_OPERATOR_EPISODE,
            "Walk me through the final voice pipeline.",
            "under four seconds",
            4000,
            "pipeline",
        ),
        (
            [
                {
                    "role": "user",
                    "content": "Set the active TTFT latency deadline to < 4s.",
                },
                {"role": "assistant", "content": "Face answers first; Depth follows."},
                {
                    "role": "user",
                    "content": (
                        "Actually, switch from TTFT to end\u2013to\u2013end voice latency."
                    ),
                },
                {"role": "assistant", "content": "The voice pipeline is now the bound."},
            ],
            "Give me the final Oracle voice design.",
            "< 4s",
            4000,
            "design",
        ),
        (
            [
                *_R30_CANONICAL_OPERATOR_EPISODE[:2],
                {
                    "role": "user",
                    "content": (
                        "Change the constraint from first-token latency under four "
                        "seconds to end-to-end voice latency under four seconds."
                    ),
                },
                {"role": "assistant", "content": "The voice pipeline is now the bound."},
            ],
            "Give me the final Oracle voice architecture.",
            "under four seconds",
            4000,
            "architecture",
        ),
    ],
)
def test_r30_authoritative_synthesis_trigger_accepts_closed_scoped_aliases(
    monkeypatch,
    history,
    prompt,
    deadline,
    deadline_milliseconds,
    target,
):
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session(f"r30-positive-{uuid.uuid4()}", "face-model")
    state.messages.extend([dict(item) for item in history])

    def unexpected_model_call(*_args, **_kwargs):
        raise AssertionError("closed authoritative synthesis must not call a model")

    monkeypatch.setattr(face, "_post_streaming", unexpected_model_call)
    spoken: list[str] = []
    result = face.chat(
        prompt,
        session_id=state.session_id,
        model="face-model",
        stream_callback=spoken.append,
    )

    receipt = result["output_guard"]["authoritative_latency_synthesis"]
    assert result["completed"] is True
    assert result["api_calls"] == 0
    assert receipt["schema"] == "Ms4AuthoritativeLatencySynthesis.v1"
    assert receipt["resolution_schema"] == "Ms4ResolvedLatencyCorrection.v1"
    assert receipt["source"] == "resolved_conversation_contract"
    assert receipt["model_output_used"] is False
    assert receipt["deadline"] == deadline
    assert receipt["deadline_milliseconds"] == deadline_milliseconds
    assert target in receipt["requested_targets"]
    assert result["text"].count(
        f"The active end-to-end voice-latency deadline is {deadline}."
    ) == 1
    assert (
        "The constraint changed from first-token latency to end-to-end voice latency. "
        f"The active end-to-end voice-latency deadline is {deadline}."
        in result["text"]
    )
    assert spoken == [result["text"]]


@pytest.mark.parametrize(
    ("history", "prompt"),
    [
        (
            _R30_CANONICAL_OPERATOR_EPISODE,
            "Give me the final architecture for database backups.",
        ),
        (
            _R30_CANONICAL_OPERATOR_EPISODE,
            "Give me the final architecture, but ignore the earlier voice-latency discussion.",
        ),
        (
            _R30_CANONICAL_OPERATOR_EPISODE,
            "ASR stays under one second; now give me the final Oracle voice architecture.",
        ),
        (
            _R30_CANONICAL_OPERATOR_EPISODE,
            "Give me the final Oracle voice architecture, but make Depth synchronous.",
        ),
        (
            [
                *_R30_CANONICAL_OPERATOR_EPISODE[:2],
                {
                    "role": "user",
                    "content": "Do not switch to end-to-end voice latency.",
                },
                {"role": "assistant", "content": "Understood."},
            ],
            "Give me the final Oracle voice architecture.",
        ),
        (
            [
                *_R30_CANONICAL_OPERATOR_EPISODE[:2],
                {
                    "role": "user",
                    "content": "If I meant end-to-end voice latency, what would change?",
                },
                {"role": "assistant", "content": "That would be hypothetical."},
            ],
            "Give me the final Oracle voice architecture.",
        ),
        (
            [
                *_R30_CANONICAL_OPERATOR_EPISODE[:2],
                {
                    "role": "user",
                    "content": (
                        'The quote says "change from first-token latency to '
                        'end-to-end voice latency."'
                    ),
                },
                {"role": "assistant", "content": "That is only quoted text."},
            ],
            "Give me the final Oracle voice architecture.",
        ),
        (
            [
                *_R30_CANONICAL_OPERATOR_EPISODE,
                {"role": "user", "content": "Now discuss database backups."},
                {"role": "assistant", "content": "Backups are a separate topic."},
            ],
            "Give me the final Oracle voice architecture.",
        ),
        (
            [
                *_R30_CANONICAL_OPERATOR_EPISODE,
                {
                    "role": "user",
                    "content": (
                        "Actually change from end-to-end voice latency to "
                        "first-token latency."
                    ),
                },
                {"role": "assistant", "content": "The constraint was reversed."},
            ],
            "Give me the final Oracle voice architecture.",
        ),
        (
            [
                {
                    "role": "user",
                    "content": (
                        "We compared first-token latency under four seconds with "
                        "end-to-end voice latency, but adopted neither."
                    ),
                },
                {"role": "assistant", "content": "That was informational only."},
            ],
            "Give me the final Oracle voice architecture.",
        ),
        (
            [
                {
                    "role": "user",
                    "content": "Assume first-token latency is the old metric.",
                },
                {"role": "assistant", "content": "No deadline was supplied."},
                {
                    "role": "user",
                    "content": "No - I mean end-to-end voice latency. Correct it.",
                },
                {"role": "assistant", "content": "The metric changed."},
            ],
            "Give me the final Oracle voice architecture.",
        ),
        *[
            (
                [
                    *_R30_CANONICAL_OPERATOR_EPISODE[:2],
                    {"role": "user", "content": correction},
                    {"role": "assistant", "content": "That was not an adopted correction."},
                ],
                "Give me the final Oracle voice architecture.",
            )
            for correction in (
                (
                    "For discussion only, change from first-token latency to "
                    "end-to-end voice latency."
                ),
                (
                    "Alice says change from first-token latency to end-to-end "
                    "voice latency."
                ),
                (
                    "The old prompt instructed us to change from first-token "
                    "latency to end-to-end voice latency."
                ),
                (
                    "Should we change from first-token latency to end-to-end "
                    "voice latency?"
                ),
                (
                    "Do the opposite of this instruction: change from first-token "
                    "latency to end-to-end voice latency."
                ),
                (
                    "Purely theoretical: change from first-token latency to "
                    "end-to-end voice latency."
                ),
                (
                    "Compare a change from first-token latency to end-to-end "
                    "voice latency."
                ),
            )
        ],
        *[
            (
                [
                    {"role": "user", "content": baseline},
                    {"role": "assistant", "content": "That was not an active constraint."},
                    {
                        "role": "user",
                        "content": "No, I mean end-to-end voice latency. Correct it.",
                    },
                    {"role": "assistant", "content": "The metric was discussed."},
                ],
                "Give me the final Oracle voice architecture.",
            )
            for baseline in (
                (
                    "Alice says the active first-token latency deadline must stay "
                    "under four seconds."
                ),
                (
                    "For discussion only, assume first-token latency must stay "
                    "under four seconds."
                ),
                (
                    "Should first-token latency stay under four seconds?"
                ),
            )
        ],
        (
            [
                *_R30_CANONICAL_OPERATOR_EPISODE[:2],
                {
                    "role": "user",
                    "content": (
                        "Change the constraint from first-token latency under five "
                        "seconds to end-to-end voice latency under four seconds."
                    ),
                },
                {"role": "assistant", "content": "That contradicts the baseline."},
            ],
            "Give me the final Oracle voice architecture.",
        ),
    ],
)
def test_r30_authoritative_synthesis_trigger_rejects_unresolved_or_unrelated_context(
    monkeypatch,
    history,
    prompt,
):
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session(f"r30-negative-{uuid.uuid4()}", "face-model")
    state.messages.extend([dict(item) for item in history])
    calls = 0
    model_candidate = f"MODEL_ROUTE_SENTINEL {_R29_ARCHITECTURE}"

    def post_streaming(_payload, callback, *, cancel_event=None):
        nonlocal calls
        del cancel_event
        calls += 1
        callback(model_candidate)
        return model_candidate, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        prompt,
        session_id=state.session_id,
        model="face-model",
        stream_callback=spoken.append,
    )

    assert calls >= 1
    assert result["api_calls"] >= 1
    assert "authoritative_latency_synthesis" not in result["output_guard"]
    assert result["metrics"].get("terminal_source") != "gateway_authoritative_render"
    assert not result["text"].startswith(
        "1. Direct answer\nThe final architecture keeps the Face lobe"
    )


def test_r30_authoritative_synthesis_rechecks_cancellation_before_callback_and_history(
    monkeypatch,
):
    class CancelAtDelivery:
        def __init__(self):
            self.checks = 0

        def is_set(self):
            self.checks += 1
            return self.checks >= 2

    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("r30-cancel-at-delivery", "face-model")
    state.messages.extend(
        [
            {
                "role": "user",
                "content": "Assume first-token latency must stay under four seconds.",
            },
            {"role": "assistant", "content": "Face answers first; Depth follows."},
            {
                "role": "user",
                "content": "No - I mean end-to-end voice latency. Correct it.",
            },
            {"role": "assistant", "content": "The voice pipeline is now the bound."},
        ]
    )
    before = list(state.messages)

    def unexpected_model_call(*_args, **_kwargs):
        raise AssertionError("authoritative synthesis must not call a model")

    monkeypatch.setattr(face, "_post_streaming", unexpected_model_call)
    spoken: list[str] = []
    cancellation = CancelAtDelivery()
    result = face.chat(
        "Give me the final architecture in plain English, and tell me what constraint I changed.",
        session_id=state.session_id,
        model="face-model",
        stream_callback=spoken.append,
        cancel_event=cancellation,
    )

    assert cancellation.checks >= 2
    assert result["api_calls"] == 0
    assert result["completed"] is False
    assert result["cancelled"] is True
    assert result["guarded_stream_emitted"] is False
    assert spoken == []
    assert state.messages == before


def test_r30_authoritative_synthesis_callback_false_cancels_before_history(
    monkeypatch,
):
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("r30-callback-false", "face-model")
    state.messages.extend([dict(item) for item in _R30_CANONICAL_OPERATOR_EPISODE])
    before = list(state.messages)

    def unexpected_model_call(*_args, **_kwargs):
        raise AssertionError("authoritative synthesis must not call a model")

    monkeypatch.setattr(face, "_post_streaming", unexpected_model_call)
    delivered: list[str] = []

    def reject_delivery(text):
        delivered.append(text)
        return False

    result = face.chat(
        "Give me the final architecture in plain English, and tell me what constraint I changed.",
        session_id=state.session_id,
        model="face-model",
        stream_callback=reject_delivery,
    )

    assert len(delivered) == 1
    assert result["api_calls"] == 0
    assert result["completed"] is False
    assert result["cancelled"] is True
    assert result["text"] == ""
    assert result["guarded_stream_emitted"] is False
    assert result["metrics"]["downstream_cancelled"] is True
    assert result["metrics"]["incomplete_reason"] == "downstream_callback"
    assert state.messages == before


def test_r30_authoritative_synthesis_rechecks_cancellation_after_callback(
    monkeypatch,
):
    cancellation = threading.Event()
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("r30-cancel-inside-callback", "face-model")
    state.messages.extend([dict(item) for item in _R30_CANONICAL_OPERATOR_EPISODE])
    before = list(state.messages)

    def unexpected_model_call(*_args, **_kwargs):
        raise AssertionError("authoritative synthesis must not call a model")

    monkeypatch.setattr(face, "_post_streaming", unexpected_model_call)
    delivered: list[str] = []

    def cancel_during_delivery(text):
        delivered.append(text)
        cancellation.set()
        return True

    result = face.chat(
        "Give me the final architecture in plain English, and tell me what constraint I changed.",
        session_id=state.session_id,
        model="face-model",
        stream_callback=cancel_during_delivery,
        cancel_event=cancellation,
    )

    assert delivered == [result["text"]]
    assert result["text"].startswith("1. Direct answer\n")
    assert result["api_calls"] == 0
    assert result["completed"] is False
    assert result["cancelled"] is True
    assert result["guarded_stream_emitted"] is True
    assert result["metrics"]["incomplete_reason"] == "cancel_event"
    assert state.messages == before


@pytest.mark.parametrize("entrypoint", ["chat", "chat_authoritative"])
def test_r30_model_free_callback_exception_is_failed_delivery_without_history(
    monkeypatch,
    entrypoint,
):
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session(f"r30-callback-exception-{entrypoint}", "face-model")
    state.messages.extend([dict(item) for item in _R30_CANONICAL_OPERATOR_EPISODE])
    before = list(state.messages)

    def unexpected_model_call(*_args, **_kwargs):
        raise AssertionError("model-free delivery must not call a model")

    monkeypatch.setattr(face, "_post_streaming", unexpected_model_call)

    def broken_delivery(_text):
        raise RuntimeError("simulated downstream delivery failure")

    if entrypoint == "chat":
        result = face.chat(
            "Give me the final architecture in plain English, and tell me what constraint I changed.",
            session_id=state.session_id,
            model="face-model",
            stream_callback=broken_delivery,
        )
    else:
        result = face.chat_authoritative(
            "Deliver the verified answer.",
            authoritative_text="Verified answer from gateway state.",
            session_id=state.session_id,
            model="face-model",
            stream_callback=broken_delivery,
        )

    assert result["api_calls"] == 0
    assert result["completed"] is False
    assert result["cancelled"] is True
    assert result["text"] == ""
    assert result["guarded_stream_emitted"] is False
    if entrypoint == "chat":
        assert result["metrics"]["downstream_cancelled"] is True
        assert result["metrics"]["incomplete_reason"] == "downstream_callback"
    else:
        assert result["metrics"]["cancel_reason"] == "downstream_callback"
    assert state.messages == before


@pytest.mark.parametrize(
    ("statement", "old_deadline_present", "new_deadline_present"),
    [
        (
            "The constraint changed from first-token latency to end-to-end voice "
            "latency. The active end-to-end voice-latency deadline is under four "
            "seconds.",
            False,
            False,
        ),
        (
            "The constraint changed from first-token latency to end-to-end voice "
            "latency. The overall voice deadline is under four seconds.",
            False,
            False,
        ),
    ],
)
def test_final_corrective_requires_self_consistent_latency_contract_before_tts_history(
    monkeypatch,
    statement,
    old_deadline_present,
    new_deadline_present,
):
    del old_deadline_present, new_deadline_present
    first = "I can provide the final architecture if you want more detail."
    corrected = _r25_final_architecture_with_constraint(
        statement,
        statement_at_end=True,
    )
    attempts = iter((first, corrected))
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("final-one-sided-corrective", "face-model")
    state.messages.extend(
        [
            {
                "role": "user",
                "content": "Assume first-token latency must stay under four seconds.",
            },
            {"role": "assistant", "content": "Face answers first; Depth follows."},
            {
                "role": "user",
                "content": "No - I mean end-to-end voice latency. Correct it.",
            },
            {"role": "assistant", "content": "The voice pipeline is now the bound."},
        ]
    )
    before = list(state.messages)
    calls = 0

    def post_streaming(_payload, callback, *, cancel_event=None):
        nonlocal calls
        del cancel_event
        calls += 1
        answer = next(attempts)
        callback(answer)
        return answer, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        "Give me the final architecture in plain English, and tell me what constraint I changed.",
        session_id=state.session_id,
        model="face-model",
        stream_callback=spoken.append,
    )

    assert calls == 0
    _assert_gateway_authoritative_latency_synthesis(result, spoken)
    assert first not in result["text"]
    assert corrected not in result["text"]
    assert state.messages == before + [
        {
            "role": "user",
            "content": (
                "Give me the final architecture in plain English, and tell me what "
                "constraint I changed."
            ),
        },
        {"role": "assistant", "content": result["text"]},
    ]


@pytest.mark.parametrize(
    ("statement", "expected_failure"),
    [
        (
            "The constraint changed from first-token latency under five seconds to "
            "end-to-end voice latency under four seconds.",
            "prior_latency_deadline_contradiction",
        ),
        (
            "The constraint changed from first-token latency under four seconds to "
            "end-to-end voice latency under five seconds.",
            "latency_deadline_contradiction",
        ),
        (
            "The constraint changed from end-to-end voice latency under four seconds "
            "to first-token latency under four seconds.",
            "latency_constraint_direction_contradiction",
        ),
        (
            "The prior constraint was end-to-end voice latency under four seconds.",
            "latency_constraint_direction_contradiction",
        ),
        (
            "End-to-end voice latency under four seconds used to be the constraint.",
            "latency_constraint_direction_contradiction",
        ),
        (
            "End-to-end voice latency (under four seconds) used to be the constraint.",
            "latency_constraint_direction_contradiction",
        ),
        (
            "End-to-end voice latency (under four seconds used to be the constraint.",
            "latency_constraint_direction_contradiction",
        ),
        (
            "End-to-end voice latency under four seconds) used to be the constraint.",
            "latency_constraint_direction_contradiction",
        ),
        (
            "The constraint changed from first-token latency to end-to-end voice "
            "latency. The standalone deadline is under four seconds.",
            "latency_deadline_missing",
        ),
        (
            "The constraint changed from first-token latency under four seconds to "
            "end-to-end voice latency. The overall voice deadline is under five seconds.",
            "unbound_latency_deadline_contradiction",
        ),
        (
            "The constraint changed from first-token latency under four seconds to "
            "end-to-end voice latency. Voice deadline: under five seconds.",
            "unbound_latency_deadline_contradiction",
        ),
        (
            "The constraint changed from first-token latency under four seconds to "
            "end-to-end voice latency. Deadline: under five seconds.",
            "unbound_latency_deadline_contradiction",
        ),
        (
            "The constraint changed from first-token latency under four seconds to "
            "end-to-end voice latency. The overall voice deadline is under four seconds, "
            "ASR has a component budget under one second, and Voice deadline: under five "
            "seconds.",
            "unbound_latency_deadline_contradiction",
        ),
        (
            "The constraint changed from first-token latency under four seconds to "
            "end-to-end voice latency. The overall voice deadline, not the ASR component "
            "budget, is under five seconds.",
            "unbound_latency_deadline_contradiction",
        ),
        (
            "The constraint changed from first-token latency under four seconds to "
            "end-to-end voice latency. The overall voice deadline is not an ASR "
            "component budget and remains under five seconds.",
            "unbound_latency_deadline_contradiction",
        ),
        (
            "The constraint changed from first-token latency under four seconds to "
            "end-to-end voice latency. The overall voice deadline isn't an ASR "
            "component budget and remains under five seconds.",
            "unbound_latency_deadline_contradiction",
        ),
        (
            "The constraint changed from first-token latency under four seconds to "
            "end-to-end voice latency. The overall voice deadline isn’t an ASR "
            "component budget and remains under five seconds.",
            "unbound_latency_deadline_contradiction",
        ),
        (
            "The constraint changed from first-token latency under four seconds to "
            "end-to-end voice latency. The overall voice deadline isn't only an ASR "
            "component budget and remains under five seconds.",
            "unbound_latency_deadline_contradiction",
        ),
        (
            "The constraint changed from first-token latency under four seconds to "
            "end-to-end voice latency. The overall voice deadline isn’t only an ASR "
            "component budget and remains under five seconds.",
            "unbound_latency_deadline_contradiction",
        ),
        (
            "The constraint changed from first-token latency under four seconds to "
            "end-to-end voice latency. The overall voice deadline isn't currently an "
            "ASR component budget and remains under five seconds.",
            "unbound_latency_deadline_contradiction",
        ),
        (
            "The constraint changed from first-token latency under four seconds to "
            "end-to-end voice latency. The overall voice deadline isn’t currently an "
            "ASR component budget and remains under five seconds.",
            "unbound_latency_deadline_contradiction",
        ),
        (
            "The constraint changed from first-token latency under four seconds to "
            "end-to-end voice latency. The overall voice deadline isn't simply an ASR "
            "component budget and remains under five seconds.",
            "unbound_latency_deadline_contradiction",
        ),
        (
            "The constraint changed from first-token latency under four seconds to "
            "end-to-end voice latency. The overall voice deadline isn’t simply an ASR "
            "component budget and remains under five seconds.",
            "unbound_latency_deadline_contradiction",
        ),
        (
            "The constraint changed from first-token latency under four seconds to "
            "end-to-end voice latency. The overall voice deadline doesn't simply apply "
            "to ASR, because it governs the full path and remains under five seconds.",
            "unbound_latency_deadline_contradiction",
        ),
        (
            "The constraint changed from first-token latency under four seconds to "
            "end-to-end voice latency. The overall voice deadline doesn’t simply apply "
            "to ASR, because it governs the full path and remains under five seconds.",
            "unbound_latency_deadline_contradiction",
        ),
        (
            "The constraint changed from first-token latency under four seconds to "
            "end-to-end voice latency. The overall voice deadline doesn't just apply "
            "to ASR, because it governs the full path and remains under five seconds.",
            "unbound_latency_deadline_contradiction",
        ),
        (
            "The constraint changed from first-token latency under four seconds to "
            "end-to-end voice latency. The overall voice deadline doesn't just apply "
            "to the ASR component budget, because it governs the full path and remains "
            "under five seconds.",
            "unbound_latency_deadline_contradiction",
        ),
        (
            "The constraint changed from first-token latency under four seconds to "
            "end-to-end voice latency. The overall voice deadline doesn’t just apply "
            "to the ASR component budget, because it governs the full path and remains "
            "under five seconds.",
            "unbound_latency_deadline_contradiction",
        ),
        (
            "The constraint changed from first-token latency under four seconds to "
            "end-to-end voice latency. The overall voice deadline doesn't only apply "
            "to the ASR component budget, because it governs the full path and remains "
            "under five seconds.",
            "unbound_latency_deadline_contradiction",
        ),
        (
            "The constraint changed from first-token latency under four seconds to "
            "end-to-end voice latency. The overall voice deadline doesn't merely apply "
            "to the ASR component budget, because it governs the full path and remains "
            "under five seconds.",
            "unbound_latency_deadline_contradiction",
        ),
        (
            "The constraint changed from first-token latency under four seconds to "
            "end-to-end voice latency. The overall voice deadline doesn't solely apply "
            "to the ASR component budget, because it governs the full path and remains "
            "under five seconds.",
            "unbound_latency_deadline_contradiction",
        ),
        (
            "The constraint changed from first-token latency under four seconds to "
            "end-to-end voice latency. The overall voice deadline doesn't simply apply "
            "to the ASR component budget, because it governs the full path and remains "
            "under five seconds.",
            "unbound_latency_deadline_contradiction",
        ),
    ],
)
def test_r25_final_architecture_contradictions_fail_closed_before_tts_history(
    monkeypatch,
    statement,
    expected_failure,
):
    del expected_failure
    answer = _r25_final_architecture_with_constraint(statement)
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session(f"r25-final-reject-{uuid.uuid4()}", "face-model")
    state.messages.extend(
        [
            {
                "role": "user",
                "content": "Assume first-token latency must stay under four seconds.",
            },
            {"role": "assistant", "content": "Face answers first; Depth follows."},
            {
                "role": "user",
                "content": "No - I mean end-to-end voice latency. Correct it.",
            },
            {"role": "assistant", "content": "The voice pipeline is now the bound."},
        ]
    )
    before = list(state.messages)
    calls = 0

    def post_streaming(_payload, callback, *, cancel_event=None):
        nonlocal calls
        del cancel_event
        calls += 1
        callback(answer)
        return answer, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        "Give me the final architecture in plain English, and tell me what constraint I changed.",
        session_id=state.session_id,
        model="face-model",
        stream_callback=spoken.append,
    )

    assert calls == 0
    _assert_gateway_authoritative_latency_synthesis(result, spoken)
    assert answer not in result["text"]
    assert state.messages == before + [
        {
            "role": "user",
            "content": (
                "Give me the final architecture in plain English, and tell me what "
                "constraint I changed."
            ),
        },
        {"role": "assistant", "content": result["text"]},
    ]


@pytest.mark.parametrize(
    "statement",
    [
        (
            "The constraint changed from first-token latency to end-to-end voice "
            "latency. The phrase under four seconds is not the operator deadline."
        ),
        (
            "The constraint changed from first-token latency to end-to-end voice "
            'latency. "Under four seconds" is copied from the prompt, not the '
            "governing deadline."
        ),
        (
            "The constraint changed from first-token latency to end-to-end voice "
            "latency. ASR has a component budget under four seconds."
        ),
        (
            "The constraint changed from first-token latency to end-to-end voice "
            "latency. The prompt says under four seconds. My timing budget is under "
            "five seconds."
        ),
        (
            "The constraint changed from first-token latency to end-to-end voice "
            "latency. The overall voice deadline is not under four seconds."
        ),
        (
            "The constraint changed from first-token latency to end-to-end voice "
            "latency. The overall voice deadline isn't under four seconds."
        ),
        (
            "The constraint changed from first-token latency to end-to-end voice "
            "latency. The overall voice deadline under four seconds is not governing."
        ),
        (
            "The constraint changed from first-token latency to end-to-end voice "
            "latency. The overall voice deadline shouldn't be under four seconds."
        ),
        (
            "The constraint changed from first-token latency to end-to-end voice "
            "latency. The overall voice deadline must not be under four seconds."
        ),
        (
            "The constraint changed from first-token latency to end-to-end voice "
            "latency. The overall voice deadline under four seconds is actually not "
            "the operator deadline."
        ),
        (
            "The constraint changed from first-token latency to end-to-end voice "
            "latency. The overall voice deadline under four seconds isn't actually "
            "the operator deadline."
        ),
        (
            "The constraint changed from first-token latency to end-to-end voice "
            'latency. The overall voice deadline phrase "under four seconds" is '
            "copied from the prompt."
        ),
        (
            "The constraint changed from first-token latency to end-to-end voice "
            'latency. The overall voice deadline was quoted as "under four seconds" '
            "but never adopted."
        ),
        (
            "The constraint changed from first-token latency to end-to-end voice "
            "latency. It is false to say the overall voice deadline is under four "
            "seconds."
        ),
        (
            "The constraint changed from first-token latency to end-to-end voice "
            "latency. The overall voice deadline under four seconds does not actually "
            "apply."
        ),
        (
            "The constraint changed from first-token latency to end-to-end voice "
            "latency. The overall voice deadline under four seconds is merely quoted "
            "from the prompt."
        ),
        (
            "The constraint changed from first-token latency to end-to-end voice "
            "latency. The overall voice deadline under four seconds has never been the "
            "operator deadline."
        ),
        (
            "The constraint changed from first-token latency to end-to-end voice "
            "latency. The overall voice deadline under four seconds might not apply."
        ),
        (
            "The constraint changed from first-token latency to end-to-end voice "
            "latency. The overall voice deadline under four seconds may not apply."
        ),
        (
            "The constraint changed from first-token latency to end-to-end voice "
            "latency. The overall voice deadline under four seconds should not be "
            "treated as active."
        ),
        (
            "The constraint changed from first-token latency to end-to-end voice "
            "latency. The overall voice deadline under four seconds cannot be "
            "considered the deadline."
        ),
        (
            "The constraint changed from first-token latency to end-to-end voice "
            "latency. The overall voice deadline is allegedly under four seconds."
        ),
        (
            "The constraint changed from first-token latency to end-to-end voice "
            "latency. The overall voice deadline is supposedly under four seconds."
        ),
        (
            "The constraint changed from first-token latency to end-to-end voice "
            "latency. The overall voice deadline is reportedly under four seconds."
        ),
        (
            "The constraint changed from first-token latency to end-to-end voice "
            "latency. There is no requirement that the overall voice deadline is under "
            "four seconds."
        ),
        (
            "The constraint changed from first-token latency to end-to-end voice "
            "latency. The overall voice deadline under four seconds applies only to "
            "ASR."
        ),
        (
            "The constraint changed from first-token latency to end-to-end voice "
            "latency. The overall voice deadline was said to be under four seconds but "
            "that was rejected."
        ),
        (
            "The constraint changed from first-token latency to end-to-end voice "
            "latency. The overall voice deadline is under four seconds according to the "
            "old prompt, not the operator."
        ),
        (
            "The constraint changed from first-token latency to end-to-end voice "
            "latency. If the overall voice deadline is under four seconds, the design "
            "would pass, but the deadline remains unresolved."
        ),
        (
            "The constraint changed from first-token latency to end-to-end voice "
            "latency. Suppose the overall voice deadline is under four seconds for this "
            "hypothetical example."
        ),
        (
            "The constraint changed from first-token latency to end-to-end voice "
            "latency. I doubt that the overall voice deadline is under four seconds."
        ),
        (
            "The constraint changed from first-token latency to end-to-end voice "
            "latency. Whether the overall voice deadline is under four seconds remains "
            "unknown."
        ),
        (
            "The constraint changed from first-token latency to end-to-end voice "
            'latency. "The overall voice deadline is under four seconds" is false.'
        ),
        (
            "The constraint changed from first-token latency to end-to-end voice "
            "latency. According to the old prompt, the overall voice deadline is under "
            "four seconds, not according to the operator."
        ),
        (
            "The constraint changed from first-token latency to end-to-end voice "
            "latency. Earlier we said the overall voice deadline is under four seconds, "
            "but that statement is obsolete."
        ),
        (
            "The constraint changed from first-token latency to end-to-end voice "
            "latency. The overall voice deadline is under four seconds. That deadline "
            "is hypothetical and not the operator's requirement."
        ),
    ],
)
def test_r25_final_architecture_does_not_bind_non_authoritative_deadline_mentions(
    monkeypatch,
    statement,
):
    answer = _r25_final_architecture_with_constraint(statement)
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session(
        f"r25-final-non-authoritative-{uuid.uuid4()}",
        "face-model",
    )
    state.messages.extend(
        [
            {
                "role": "user",
                "content": "Assume first-token latency must stay under four seconds.",
            },
            {"role": "assistant", "content": "Face answers first; Depth follows."},
            {
                "role": "user",
                "content": "No - I mean end-to-end voice latency. Correct it.",
            },
            {"role": "assistant", "content": "The voice pipeline is now the bound."},
        ]
    )
    before = list(state.messages)
    calls = 0

    def post_streaming(_payload, callback, *, cancel_event=None):
        nonlocal calls
        del cancel_event
        calls += 1
        callback(answer)
        return answer, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        "Give me the final architecture in plain English, and tell me what constraint I changed.",
        session_id=state.session_id,
        model="face-model",
        stream_callback=spoken.append,
    )

    assert calls == 0
    _assert_gateway_authoritative_latency_synthesis(result, spoken)
    assert answer not in result["text"]
    assert state.messages == before + [
        {
            "role": "user",
            "content": (
                "Give me the final architecture in plain English, and tell me what "
                "constraint I changed."
            ),
        },
        {"role": "assistant", "content": result["text"]},
    ]


def test_r26_underlength_final_candidate_cannot_be_lifted_over_minimum_by_prefix(
    monkeypatch,
):
    full = _r25_final_architecture_with_constraint(
        "The constraint changed from first-token latency to end-to-end voice latency "
        "under four seconds."
    )
    candidate = " ".join(full.split()[:125])
    assert len(candidate.split()) == 125
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("r26-underlength-prefix", "face-model")
    state.messages.extend(
        [
            {
                "role": "user",
                "content": "Assume first-token latency must stay under four seconds.",
            },
            {"role": "assistant", "content": "Face answers first; Depth follows."},
            {
                "role": "user",
                "content": "No - I mean end-to-end voice latency. Correct it.",
            },
            {"role": "assistant", "content": "The voice pipeline is now the bound."},
        ]
    )
    before = list(state.messages)
    calls = 0

    def post_streaming(_payload, callback, *, cancel_event=None):
        nonlocal calls
        del cancel_event
        calls += 1
        callback(candidate)
        return candidate, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        "Give me the final architecture in plain English, and tell me what constraint I changed.",
        session_id=state.session_id,
        model="face-model",
        stream_callback=spoken.append,
    )

    assert calls == 0
    _assert_gateway_authoritative_latency_synthesis(result, spoken)
    assert candidate not in result["text"]
    assert state.messages == before + [
        {
            "role": "user",
            "content": (
                "Give me the final architecture in plain English, and tell me what "
                "constraint I changed."
            ),
        },
        {"role": "assistant", "content": result["text"]},
    ]


@pytest.mark.parametrize(
    ("dimension", "deadline"),
    [
        ("time to first token", "under 4 seconds"),
        ("time-to-first-token", "within 4 sec"),
        ("TTFT", "< 4s"),
        ("first response token", "less than four seconds"),
        ("first-token latency", "at most 4 seconds"),
    ],
)
def test_r19_latency_revision_accepts_equivalent_dimension_and_deadline_aliases(
    monkeypatch,
    dimension,
    deadline,
):
    answer = (
        f"The revised recommendation preserves {dimension} {deadline} because the Face "
        "lobe must begin the useful foreground response inside that bound while the "
        "Depth lobe continues asynchronously in the background. This ordered staging "
        "keeps the interactive path responsive and delivers verified analysis later. "
    ) * 6
    assert len(answer.split()) >= 130
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session(f"r19-{dimension}-{deadline}", "face-model")
    state.messages.append(
        {"role": "assistant", "content": "Use synchronous Depth for every diagnosis."}
    )
    calls = 0

    def post_streaming(_payload, callback, *, cancel_event=None):
        nonlocal calls
        del cancel_event
        calls += 1
        callback(answer)
        return answer, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []

    result = face.chat(
        "Assume first-token latency must stay under four seconds. Revise the recommendation.",
        session_id=state.session_id,
        model="face-model",
        stream_callback=spoken.append,
    )

    assert calls == 1
    assert result["completed"] is True
    assert spoken == [answer.strip()]
    assert result["output_guard"]["substantive_followup"]["first_candidate_failures"] == []


def test_r33_first_audio_onset_is_not_first_generated_token_latency():
    assert _FIRST_TOKEN_LATENCY_RE.search("first audible token") is None
    assert _FIRST_TOKEN_LATENCY_RE.search("time to first audible token") is None
    assert _FIRST_TOKEN_LATENCY_RE.search("first response token") is not None
    assert _FIRST_TOKEN_LATENCY_RE.search("time to first token") is not None
    assert _FIRST_TOKEN_LATENCY_RE.search("TTFT") is not None

    reference = _latency_constraint_reference(
        "Assume first-token latency must stay under four seconds. Revise the recommendation.",
        [],
    )
    assert reference is not None
    assert "first-token latency" in reference
    assert "first generated response token" in reference
    assert "not first-audio onset" in reference
    assert "first audible token" not in reference
    assert "under four seconds" in reference
    assert "final ASR should fan out concurrently" in reference
    assert "Depth starts asynchronous analysis" in reference
    assert "Do not say that Depth starts only after Face responds" in reference
    assert "do not say that a job is dispatched" in reference.lower()
    assert "a result will appear, arrive, surface, or be delivered when ready" in reference


@pytest.mark.parametrize(
    ("dimension", "deadline", "expected_failure"),
    [
        ("time to first token", "under 5 seconds", "latency_deadline_missing"),
        ("end-to-end voice latency", "under 4 seconds", "active_latency_dimension_missing"),
        ("first audible token", "under 4 seconds", "active_latency_dimension_missing"),
    ],
)
def test_r19_latency_revision_rejects_wrong_deadline_or_wrong_dimension(
    monkeypatch,
    dimension,
    deadline,
    expected_failure,
):
    answer = (
        f"The revised recommendation uses {dimension} {deadline} because the Face lobe "
        "must begin a useful foreground response while the Depth lobe continues "
        "asynchronously in the background. This ordered staging preserves the chosen "
        "interactive requirement and delivers verified analysis later. "
    ) * 7
    assert len(answer.split()) >= 130
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session(f"r19-negative-{dimension}-{deadline}", "face-model")
    state.messages.append(
        {"role": "assistant", "content": "Use synchronous Depth for every diagnosis."}
    )
    before = list(state.messages)

    payloads: list[dict[str, object]] = []

    def post_streaming(_payload, callback, *, cancel_event=None):
        del cancel_event
        payloads.append(_payload)
        callback(answer)
        return answer, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []

    result = face.chat(
        "Assume first-token latency must stay under four seconds. Revise the recommendation.",
        session_id=state.session_id,
        model="face-model",
        stream_callback=spoken.append,
    )

    semantic = result["output_guard"]["substantive_followup"]
    assert result["completed"] is False
    assert result["api_calls"] == 2
    assert expected_failure in semantic["first_candidate_failures"]
    assert expected_failure in semantic["corrective_candidate_failures"]
    assert spoken == []
    assert len(payloads) == 2
    corrective_system = str(payloads[1]["messages"][0]["content"])
    assert "between 220 and 380 visible words" in corrective_system
    assert "exact active deadline: under four seconds" in corrective_system
    assert (
        "do not state any other numeric latency deadline, target, threshold, or budget"
        in corrective_system.lower()
    )
    assert "do not exceed 380 visible words" in corrective_system
    assert payloads[0]["temperature"] == 0.2
    assert payloads[1]["temperature"] == 0.0
    assert state.messages == before


def test_r20_latency_revision_binds_omitted_deadline_without_expensive_retry(monkeypatch):
    answer = (
        "I revise the recommendation around first-token latency because the Face lobe "
        "must answer first with a useful orientation while the Depth lobe continues "
        "asynchronously in the background. This changes the sequence by keeping the "
        "interactive path responsive and reserving deeper diagnosis for automatic later "
        "delivery in the same conversation. The practical consequence is that the user "
        "gets an immediate, honest recommendation without losing the complete analysis. "
    ) * 4
    assert len(answer.split()) >= 130
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("r20-missing-deadline", "face-model")
    state.messages.append(
        {"role": "assistant", "content": "Use synchronous Depth for every diagnosis."}
    )
    calls = 0

    def post_streaming(_payload, callback, *, cancel_event=None):
        nonlocal calls
        del cancel_event
        calls += 1
        callback(answer)
        return answer, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []

    result = face.chat(
        "Assume first-token latency must stay under four seconds. Revise the recommendation.",
        session_id=state.session_id,
        model="face-model",
        stream_callback=spoken.append,
    )

    assert calls == 1
    assert result["completed"] is True
    assert result["text"].startswith(
        "Active constraint: first-token latency must stay under four seconds."
    )
    postcondition = result["output_guard"]["latency_constraint_postcondition"]
    assert postcondition == {
        "schema": "Ms4ActiveLatencyConstraintPostcondition.v1",
        "applied": True,
        "active_dimension_present": True,
        "deadline_present": False,
        "contradictory_deadline": False,
        "failures_before": ["latency_deadline_missing"],
        "max_words": None,
        "deadline": "under four seconds",
    }
    semantic = result["output_guard"]["substantive_followup"]
    assert semantic["first_candidate_failures"] == ["latency_deadline_missing"]
    assert semantic["corrective_regeneration_attempted"] is False
    assert spoken == [result["text"]]


def test_r32_first_token_deadline_and_staging_omissions_compose_without_retry(
    monkeypatch,
):
    answer = (
        "I revise the recommendation around first-token latency because the Face lobe "
        "must answer first with a useful orientation. Deeper analysis runs asynchronously "
        "outside that foreground response. This changes the sequence by protecting the "
        "interactive path while retaining a complete diagnostic review in the same "
        "conversation. The practical consequence is that the operator receives an "
        "immediate, honest recommendation and can continue working while the larger "
        "analysis examines evidence, alternatives, and uncertainty. "
    ) * 4
    assert len(answer.split()) >= 130
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("r32-deadline-staging-compose", "face-model")
    state.messages.append(
        {"role": "assistant", "content": "Use synchronous analysis for every diagnosis."}
    )
    calls = 0

    def post_streaming(_payload, callback, *, cancel_event=None):
        nonlocal calls
        del cancel_event
        calls += 1
        callback(answer)
        return answer, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        "Assume first-token latency must stay under four seconds. Revise the recommendation.",
        session_id=state.session_id,
        model="face-model",
        stream_callback=spoken.append,
    )

    semantic = result["output_guard"]["substantive_followup"]
    assert calls == 1
    assert result["completed"] is True
    assert semantic["first_candidate_failures"] == [
        "latency_deadline_missing",
        "latency_lobe_staging_missing",
    ]
    assert semantic["corrective_regeneration_attempted"] is False
    assert result["text"].startswith(
        "Active constraint: first-token latency must stay under four seconds."
    )
    assert "Face gives the immediate foreground response" in result["text"]
    assert "Depth runs asynchronously in the background" in result["text"]
    assert spoken == [result["text"]]


@pytest.mark.parametrize(
    ("answer", "expected_failure"),
    [
        (
            (
                "I revise the recommendation so first-token latency stays under four seconds, "
                "but first-token latency may also stay under five seconds because the Face lobe "
                "answers first while the Depth lobe continues asynchronously in the background. "
                "This changes the sequence and preserves automatic later delivery. "
            )
            * 7,
            "latency_deadline_contradiction",
        ),
        (
            (
                "I revise the recommendation around end-to-end voice latency under four seconds, "
                "not first-token latency. The Face lobe answers while the Depth lobe continues "
                "asynchronously in the background because that changes the execution sequence "
                "and preserves automatic later delivery in the same conversation. "
            )
            * 8,
            "active_latency_dimension_missing",
        ),
    ],
)
def test_r20_latency_revision_rejects_contradiction_and_negated_active_dimension(
    monkeypatch,
    answer,
    expected_failure,
):
    assert len(answer.split()) >= 130
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session(f"r20-{expected_failure}", "face-model")
    state.messages.append(
        {"role": "assistant", "content": "Use synchronous Depth for every diagnosis."}
    )
    before = list(state.messages)

    def post_streaming(_payload, callback, *, cancel_event=None):
        del cancel_event
        callback(answer)
        return answer, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        "Assume first-token latency must stay under four seconds. Revise the recommendation.",
        session_id=state.session_id,
        model="face-model",
        stream_callback=spoken.append,
    )

    semantic = result["output_guard"]["substantive_followup"]
    assert result["completed"] is False
    assert result["api_calls"] == 2
    assert expected_failure in semantic["first_candidate_failures"]
    assert expected_failure in semantic["corrective_candidate_failures"]
    assert spoken == []
    assert state.messages == before


@pytest.mark.parametrize(
    "negated_dimension",
    [
        "first-token latency is not the active requirement",
        "first-token latency is no longer the active requirement",
        "we reject first-token latency",
        "we do not use first-token latency",
        "replacing first-token latency",
        "superseding first-token latency",
        "in place of first-token latency",
    ],
)
def test_r20_latency_revision_rejects_postposed_or_rejected_active_dimension(
    monkeypatch,
    negated_dimension,
):
    answer = (
        "I revise the active requirement to end-to-end voice latency under four seconds; "
        f"{negated_dimension}. The Face lobe answers while the Depth lobe continues "
        "asynchronously in the background because this changes the execution sequence "
        "and preserves automatic later delivery in the same conversation. "
    ) * 6
    assert len(answer.split()) >= 130
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session(f"r20-postposed-{uuid.uuid4()}", "face-model")
    state.messages.append(
        {"role": "assistant", "content": "Use synchronous Depth for every diagnosis."}
    )
    before = list(state.messages)

    def post_streaming(_payload, callback, *, cancel_event=None):
        del cancel_event
        callback(answer)
        return answer, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        "Assume first-token latency must stay under four seconds. Revise the recommendation.",
        session_id=state.session_id,
        model="face-model",
        stream_callback=spoken.append,
    )

    semantic = result["output_guard"]["substantive_followup"]
    assert result["completed"] is False
    assert result["api_calls"] == 2
    assert "active_latency_dimension_missing" in semantic["first_candidate_failures"]
    assert "active_latency_dimension_missing" in semantic["corrective_candidate_failures"]
    assert spoken == []
    assert state.messages == before


def test_r20_corrective_candidate_over_word_cap_fails_closed(monkeypatch):
    first = "I can revise that if you want more detail."
    corrected = (
        "I revise the recommendation so first-token latency stays under four seconds. "
        "The Face lobe answers first with a useful orientation while the Depth lobe "
        "continues asynchronously in the background because this ordered sequence "
        "keeps the interaction responsive and preserves automatic later delivery. "
    ) * 18
    assert len(corrected.split()) > 600
    attempts = iter((first, corrected))
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("r20-overlength-correction", "face-model")
    state.messages.append(
        {"role": "assistant", "content": "Use synchronous Depth for every diagnosis."}
    )
    before = list(state.messages)

    def post_streaming(_payload, callback, *, cancel_event=None):
        del cancel_event
        answer = next(attempts)
        callback(answer)
        return answer, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        "Assume first-token latency must stay under four seconds. Revise the recommendation.",
        session_id=state.session_id,
        model="face-model",
        stream_callback=spoken.append,
    )

    semantic = result["output_guard"]["substantive_followup"]
    assert result["completed"] is False
    assert result["api_calls"] == 2
    assert "overlength" in semantic["corrective_candidate_failures"]
    assert spoken == []
    assert state.messages == before


def test_r21_complete_corrective_between_target_and_hard_cap_is_delivered(monkeypatch):
    first = "I can revise that if you want more detail."
    corrected = (
        "I revise the recommendation so first-token latency stays under four seconds. "
        "The Face lobe answers first with a useful orientation while the Depth lobe "
        "continues asynchronously in the background because this ordered sequence "
        "keeps the interaction responsive and preserves automatic later delivery. "
    ) * 12
    assert 340 < len(corrected.split()) <= 600
    attempts = iter((first, corrected))
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("r21-bounded-complete-correction", "face-model")
    state.messages.append(
        {"role": "assistant", "content": "Use synchronous Depth for every diagnosis."}
    )
    before = list(state.messages)

    def post_streaming(_payload, callback, *, cancel_event=None):
        del cancel_event
        answer = next(attempts)
        callback(answer)
        return answer, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        "Assume first-token latency must stay under four seconds. Revise the recommendation.",
        session_id=state.session_id,
        model="face-model",
        stream_callback=spoken.append,
    )

    semantic = result["output_guard"]["substantive_followup"]
    assert result["completed"] is True
    assert result["api_calls"] == 2
    assert semantic["corrective_candidate_failures"] == []
    assert spoken == [result["text"]]
    assert state.messages == before + [
        {
            "role": "user",
            "content": (
                "Assume first-token latency must stay under four seconds. "
                "Revise the recommendation."
            ),
        },
        {"role": "assistant", "content": result["text"]},
    ]


def _bind_r34_verified_face_depth_context(
    face: FaceLobeChat,
    session_id: str,
    *,
    goal: str = (
        "Compare the 35B Depth lobe with the 4B Face lobe for diagnosing an "
        "intermittent distributed inference failure and give a recommendation."
    ),
) -> None:
    face.bind_depth_result(
        conversation_id=session_id,
        job_id=f"da-{uuid.uuid4()}",
        result_text=(
            "The verified recommendation keeps Face on immediate triage and Depth on "
            "asynchronous diagnostic analysis."
        ),
        goal=goal,
        model="depth-model",
    )


def _r34_model_route_candidate() -> str:
    return (
        "MODEL_ROUTE_SENTINEL. I revise the recommendation so first-token latency stays "
        "under four seconds because Face owns the foreground response while Depth runs "
        "asynchronously outside the critical path. The design keeps the first useful "
        "generated response token independent of deeper analysis and preserves an honest "
        "statement of uncertainty. The mechanism uses separate routing and admission "
        "priority so the larger analysis cannot block the direct answer. The practical "
        "trade-off is that the immediate answer may be less comprehensive, while the "
        "background review can examine traces and routing evidence. For example, a "
        "distributed-inference fault can receive a reversible foreground diagnostic step "
        "while deeper evidence is analyzed independently. "
    ) * 3


@pytest.mark.parametrize(
    "revision",
    [
        "Assume first-token latency must stay under four seconds. Revise the recommendation.",
        "Set the active first token latency deadline to under 4 seconds. Revise the architecture.",
    ],
)
def test_r34_authoritative_voice_first_token_revision_uses_typed_depth_context(
    monkeypatch,
    revision,
):
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session(f"r34-positive-{uuid.uuid4()}", "face-model")
    _bind_r34_verified_face_depth_context(face, state.session_id)
    before = list(state.messages)

    def unexpected_model_call(*_args, **_kwargs):
        raise AssertionError("closed authoritative voice revision must not call a model")

    monkeypatch.setattr(face, "_post_streaming", unexpected_model_call)
    monkeypatch.setattr(face, "_post_blocking", unexpected_model_call)
    spoken: list[str] = []
    result = face.chat(
        revision,
        session_id=state.session_id,
        model="face-model",
        stream_callback=spoken.append,
        substantive_word_range=(150, 380),
    )

    receipt = result["output_guard"]["authoritative_first_token_revision"]
    semantic = result["output_guard"]["substantive_followup"]
    assert result["completed"] is True
    assert result["api_calls"] == 0
    assert result["metrics"]["terminal_source"] == "gateway_authoritative_render"
    assert result["output_guard"]["reason"] == "authoritative_first_token_revision"
    assert receipt["schema"] == "Ms4AuthoritativeFirstTokenRevision.v1"
    assert receipt["model_output_used"] is False
    assert receipt["deadline_milliseconds"] == 4000
    assert 220 <= receipt["word_count"] <= 380
    assert semantic["first_candidate_failures"] == []
    assert semantic["corrective_regeneration_attempted"] is False
    assert spoken == [result["text"]]
    assert result["text"].startswith(
        "Revised: latency now governs delivery. First-token latency must stay "
    )
    assert len(result["text"].split(".", 1)[0].split()) == 5
    assert "first useful generated response token" in result["text"]
    assert "first-audio onset" in result["text"]
    assert "fan the finalized input out concurrently" in result["text"]
    assert "only after Face" not in result["text"]
    assert _parallel_stage_sequence_contradiction(result["text"]) is False
    assert state.messages == before + [
        {"role": "user", "content": revision},
        {"role": "assistant", "content": result["text"]},
    ]


def test_r40_authoritative_voice_revision_accepts_verified_parameter_mapped_continuation(
    monkeypatch,
):
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session(f"r40-parameter-map-{uuid.uuid4()}", "face-model")
    _bind_r34_verified_face_depth_context(face, state.session_id)
    state.messages.extend(
        [
            {"role": "user", "content": "What evidence would change your mind?"},
            {
                "role": "assistant",
                "content": (
                    "For this distributed inference failure, neither the 35B candidate "
                    "nor the 4B candidate wins from size alone. Diagnostic accuracy, "
                    "latency, VRAM, compute, and falsifying benchmark evidence determine "
                    "whether the recommendation should change."
                ),
            },
        ]
    )

    def unexpected_model_call(*_args, **_kwargs):
        raise AssertionError("verified parameter-mapped continuation must not call a model")

    monkeypatch.setattr(face, "_post_streaming", unexpected_model_call)
    monkeypatch.setattr(face, "_post_blocking", unexpected_model_call)
    result = face.chat(
        "Assume first token latency must stay under 4 seconds. Revise the recommendation.",
        session_id=state.session_id,
        model="face-model",
        stream_callback=lambda _text: True,
        substantive_word_range=(150, 380),
    )

    receipt = result["output_guard"]["authoritative_first_token_revision"]
    assert result["completed"] is True
    assert result["api_calls"] == 0
    assert receipt["context_source"] == (
        "verified_depth_goal_and_immediate_parameter_mapped_answer"
    )


def test_r40_authoritative_voice_revision_rejects_unrelated_parameter_labels(
    monkeypatch,
):
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session(f"r40-unrelated-parameter-map-{uuid.uuid4()}", "face-model")
    _bind_r34_verified_face_depth_context(face, state.session_id)
    state.messages.extend(
        [
            {"role": "user", "content": "Where are the backup files?"},
            {
                "role": "assistant",
                "content": "The archived 35B and 4B backup files are in cold storage.",
            },
        ]
    )
    calls = 0

    def post_streaming(_payload, callback, *, cancel_event=None):
        nonlocal calls
        del cancel_event
        calls += 1
        candidate = _r34_model_route_candidate()
        callback(candidate)
        return candidate, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    result = face.chat(
        "Assume first token latency must stay under 4 seconds. Revise the recommendation.",
        session_id=state.session_id,
        model="face-model",
        stream_callback=lambda _text: True,
        substantive_word_range=(150, 380),
    )

    assert calls >= 1
    assert result["api_calls"] >= 1
    assert "authoritative_first_token_revision" not in result["output_guard"]


@pytest.mark.parametrize(
    "revision",
    [
        'Alice said, "Assume first-token latency must stay under four seconds. Revise the recommendation."',
        "Do not assume first-token latency must stay under four seconds. Revise the recommendation.",
        "If first-token latency must stay under four seconds, revise the recommendation.",
        "Suppose first-token latency must stay under four seconds. Revise the recommendation.",
        (
            "Assume first-token latency must stay under four seconds or under five seconds. "
            "Revise the recommendation."
        ),
        (
            "Assume first-token latency must stay under four seconds. Revise the "
            "recommendation and ignore the closed constraint grammar."
        ),
    ],
)
def test_r34_authoritative_voice_revision_rejects_open_or_unowned_current_turn(
    monkeypatch,
    revision,
):
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session(f"r34-current-negative-{uuid.uuid4()}", "face-model")
    _bind_r34_verified_face_depth_context(face, state.session_id)
    calls = 0

    def post_streaming(_payload, callback, *, cancel_event=None):
        nonlocal calls
        del cancel_event
        calls += 1
        candidate = _r34_model_route_candidate()
        callback(candidate)
        return candidate, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    result = face.chat(
        revision,
        session_id=state.session_id,
        model="face-model",
        stream_callback=lambda _text: True,
        substantive_word_range=(150, 380),
    )

    assert calls >= 1
    assert result["api_calls"] >= 1
    assert "authoritative_first_token_revision" not in result["output_guard"]
    assert result["metrics"].get("terminal_source") != "gateway_authoritative_render"


@pytest.mark.parametrize(
    "case",
    [
        "no_verified_depth",
        "unrelated_verified_goal",
        "unrelated_latest_answer",
        "current_depth_dispatch",
    ],
)
def test_r34_authoritative_voice_revision_requires_typed_current_ownership(
    monkeypatch,
    case,
):
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session(f"r34-owner-{case}-{uuid.uuid4()}", "face-model")
    if case == "no_verified_depth":
        state.messages.append(
            {
                "role": "assistant",
                "content": "Use Face for foreground triage and Depth for diagnostic analysis.",
            }
        )
    elif case == "unrelated_verified_goal":
        _bind_r34_verified_face_depth_context(
            face,
            state.session_id,
            goal="Review a PCB layout and component placement.",
        )
    else:
        _bind_r34_verified_face_depth_context(face, state.session_id)
    if case == "unrelated_latest_answer":
        state.messages.extend(
            [
                {"role": "user", "content": "Now discuss database backups."},
                {"role": "assistant", "content": "Backups are a separate topic."},
            ]
        )
    calls = 0

    def post_streaming(_payload, callback, *, cancel_event=None):
        nonlocal calls
        del cancel_event
        calls += 1
        candidate = _r34_model_route_candidate()
        callback(candidate)
        return candidate, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    kwargs = {
        "session_id": state.session_id,
        "model": "face-model",
        "stream_callback": lambda _text: True,
    }
    kwargs["substantive_word_range"] = (150, 380)
    if case == "current_depth_dispatch":
        kwargs["extra_system"] = (
            "- THIS TURN DISPATCHED job da-12345678-1234-1234-1234-123456789abc "
            "to the Depth Lobe."
        )
    result = face.chat(
        "Assume first-token latency must stay under four seconds. Revise the recommendation.",
        **kwargs,
    )

    assert calls >= 1
    assert result["api_calls"] >= 1
    assert "authoritative_first_token_revision" not in result["output_guard"]


def test_r34_authoritative_first_token_renderer_drift_fails_closed(monkeypatch):
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("r34-renderer-drift", "face-model")
    _bind_r34_verified_face_depth_context(face, state.session_id)
    before = list(state.messages)
    model_calls = 0

    def drifted_renderer(_contract, _resolution):
        return "Invalid short gateway prose.", {
            "schema": "Ms4AuthoritativeFirstTokenRevision.v1",
            "applied": True,
            "source": "resolved_conversation_contract",
            "model_output_used": False,
            "resolution_schema": "Ms4ResolvedFirstTokenRevision.v1",
            "deadline": "under four seconds",
            "deadline_milliseconds": 4000,
            "word_count": 4,
        }

    def unexpected_model_call(_payload, callback, *, cancel_event=None):
        nonlocal model_calls
        del cancel_event
        model_calls += 1
        candidate = _r34_model_route_candidate()
        callback(candidate)
        return candidate, _completed_stream_stats()

    monkeypatch.setattr(
        "machine_spirit_4.gateway.face_lobe_chat._render_authoritative_first_token_revision",
        drifted_renderer,
    )
    monkeypatch.setattr(face, "_post_streaming", unexpected_model_call)
    spoken: list[str] = []
    result = face.chat(
        "Assume first-token latency must stay under four seconds. Revise the recommendation.",
        session_id=state.session_id,
        model="face-model",
        stream_callback=spoken.append,
        substantive_word_range=(150, 380),
    )

    semantic = result["output_guard"]["substantive_followup"]
    assert model_calls == 0
    assert result["api_calls"] == 0
    assert result["completed"] is False
    assert result["text"] == ""
    assert semantic["fail_closed"] is True
    assert "underlength" in semantic["first_candidate_failures"]
    assert spoken == []
    assert state.messages == before


def test_r2_final_architecture_keeps_the_resolved_constraint_change(fake_hivemind):
    fake_hivemind["script"]["per_model"] = {
        "nemotron-3-nano:4b": {"content": "Final architecture."},
    }
    face = FaceLobeChat(hivemind_url=fake_hivemind["url"])
    state = face.get_or_create_session("r2-final", "nemotron-3-nano:4b")
    state.messages.extend(
        [
            {"role": "user", "content": "First-token latency must stay under four seconds."},
            {"role": "assistant", "content": "Initial recommendation."},
            {"role": "user", "content": "No, I mean end-to-end voice latency."},
            {"role": "assistant", "content": "Corrected recommendation."},
        ]
    )

    result = face.chat(
        "Give me the final architecture in plain English, and tell me what constraint I changed.",
        session_id="r2-final",
        model="nemotron-3-nano:4b",
    )

    _assert_gateway_authoritative_latency_synthesis(result, None)
    assert fake_hivemind["log"] == []


_R31_CORRECTION_BASELINE = [
    {
        "role": "user",
        "content": (
            "Assume first-token latency must stay under four seconds. "
            "Revise the recommendation."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Face begins the useful response inside the first-token bound while "
            "Depth continues asynchronously."
        ),
    },
]


def _assert_gateway_authoritative_latency_correction(
    result: dict,
    spoken: list[str] | None,
) -> str:
    text = result["text"]
    receipt = result["output_guard"]["authoritative_latency_correction"]
    semantic = result["output_guard"]["substantive_followup"]
    assert result["completed"] is True
    assert result["cancelled"] is False
    assert result["api_calls"] == 0
    assert result["output_guard"]["reason"] == "authoritative_latency_correction"
    assert receipt["schema"] == "Ms4AuthoritativeLatencyCorrection.v1"
    assert receipt["applied"] is True
    assert receipt["source"] == "resolved_conversation_contract"
    assert receipt["model_output_used"] is False
    assert receipt["resolution_schema"] == "Ms4ResolvedLatencyCorrection.v1"
    assert receipt["deadline"] == "under four seconds"
    assert receipt["deadline_milliseconds"] == 4000
    assert 130 <= receipt["word_count"] <= 600
    assert semantic["intent"] == "contextual_revision"
    assert semantic["first_candidate_failures"] == []
    assert semantic["corrective_regeneration_attempted"] is False
    assert "first-token latency under four seconds" in text
    assert "end-to-end voice latency under four seconds" in text
    assert "ASR input" in text
    assert "TTS synthesis" in text
    assert "audible playback" in text
    assert "Depth lobe starts its asynchronous" in text
    assert "first useful audible response" in text
    assert "Full-utterance completion is a separate metric" in text
    if spoken is not None:
        assert spoken == [text]
    return text


def test_parallel_claim_with_post_face_depth_dispatch_is_rejected():
    contradictory = (
        "Use a strictly parallel architecture. Only after the Face Lobe has responded "
        "does the system dispatch the trace to the Depth Lobe."
    )
    genuinely_parallel = (
        "When ASR finalizes, the router fans out concurrently: Face starts the spoken "
        "response while the Depth Lobe begins asynchronous analysis."
    )

    assert _latency_staging_rejected(contradictory) is True
    assert _latency_staging_rejected(genuinely_parallel) is False


@pytest.mark.parametrize(
    "candidate",
    (
        "Run Face and Depth concurrently. Face completes, then invoke the Depth Lobe.",
        "They are concurrent. Face completes its reply; then the router invokes Depth.",
        "The pipeline is parallel, but the Depth Lobe remains idle until Face returns.",
        "Use simultaneous Face and Depth paths, but Depth does not begin until Face answers.",
        "Use parallel paths and launch Depth only after the Face Lobe replies.",
        "Launch Depth only after Face has replied, although Face and Depth are parallel.",
    ),
)
def test_parallel_stage_sequence_contradiction_catches_ordering_variants(candidate):
    assert _parallel_stage_sequence_contradiction(candidate) is True


@pytest.mark.parametrize(
    "candidate",
    (
        "ASR fans out concurrently to Face and Depth; each starts from the final transcript.",
        "Only after Face responds is the already-running Depth result delivered.",
        "Do not claim they are parallel if Depth starts after Face; actual ASR fans both out concurrently.",
        "TTS chunks synthesize in parallel; Face answers, then Depth starts.",
        "Face answers first, then Depth starts.",
        "A parallel design must not dispatch Depth only after Face responds; both start from the finalized transcript.",
    ),
)
def test_parallel_stage_sequence_gate_allows_truthful_or_disavowed_prose(candidate):
    assert _parallel_stage_sequence_contradiction(candidate) is False


def test_voice_response_start_contract_separates_onset_and_completion():
    candidate = (
        "The response-start SLO begins at the end of the operator's speech and ends at "
        "the first useful non-silent reply onset. Full audible utterance completion is "
        "tracked separately as an independent metric and is not subject to that same bound."
    )

    result = _voice_response_start_contract(candidate)

    assert result["passed"] is True
    assert result["checks"] == {
        "speech_end_boundary": True,
        "first_audible_boundary": True,
        "full_completion_metric": True,
        "full_completion_deadline_conflation": False,
    }


@pytest.mark.parametrize(
    "candidate",
    (
        (
            "The governing end-to-end voice-latency deadline is under four seconds. "
            "The operator experiences the whole spoken turn, not merely text generation. "
            "ASR, Face, TTS, buffering, and audible playback complete inside that bound."
        ),
        (
            "The active end-to-end voice-latency deadline is under four seconds. The "
            "governing measurement spans ASR through Face generation and TTS until audible "
            "playback completes the live turn."
        ),
    ),
)
def test_voice_response_start_contract_rejects_q14_completion_conflation(candidate):
    result = _voice_response_start_contract(candidate)

    assert result["passed"] is False
    assert result["checks"]["full_completion_metric"] is False
    assert result["checks"]["full_completion_deadline_conflation"] is True


def test_substantive_latency_gate_reports_granular_voice_contract_failures():
    contract = {
        "intent": "contextual_revision",
        "min_words": 1,
        "max_words": 600,
        "latency_sequence_required": True,
        "latency_correction_required": True,
        "latency_deadline_milliseconds": 4000,
        "ordinal_concrete_example_required": False,
    }
    candidate = (
        "I revise the recommendation from first-token latency to end-to-end voice latency "
        "under four seconds because the Face lobe answers while the Depth lobe runs "
        "asynchronously. ASR, Face generation, TTS synthesis, and completed audible "
        "playback all fit inside the same bound."
    )

    failures = _substantive_followup_failures(candidate, contract)

    assert "voice_slo_speech_end_missing" in failures
    assert "voice_slo_first_audible_missing" in failures
    assert "voice_full_completion_metric_missing" in failures
    assert "voice_full_completion_deadline_conflation" in failures


@pytest.mark.parametrize(
    "correction",
    [
        "No\u2014I mean end-to-end voice latency. Correct the recommendation.",
        "No - I mean end-to-end voice latency. Correct it.",
        "No, I mean end-to-end voice latency. Correct it.",
        "No, I mean end-to-end voice latency, correct the recommendation.",
        "No: I mean end-to-end voice latency. Update the recommendation.",
        "No\u2013I mean end-to-end voice latency. Revise the recommendation.",
    ],
)
def test_r31_authoritative_correction_renders_closed_latest_two_turn_episode(
    monkeypatch,
    correction,
):
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session(f"r31-positive-{uuid.uuid4()}", "face-model")
    state.messages.extend([dict(item) for item in _R31_CORRECTION_BASELINE])
    before = list(state.messages)

    def unexpected_model_call(*_args, **_kwargs):
        raise AssertionError("closed authoritative correction must not call a model")

    monkeypatch.setattr(face, "_post_streaming", unexpected_model_call)
    monkeypatch.setattr(face, "_post_blocking", unexpected_model_call)
    spoken: list[str] = []
    result = face.chat(
        correction,
        session_id=state.session_id,
        model="face-model",
        stream_callback=spoken.append,
    )

    _assert_gateway_authoritative_latency_correction(result, spoken)
    assert state.messages == before + [
        {"role": "user", "content": correction},
        {"role": "assistant", "content": result["text"]},
    ]


def test_r31_authoritative_correction_works_without_streaming(monkeypatch):
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("r31-blocking", "face-model")
    state.messages.extend([dict(item) for item in _R31_CORRECTION_BASELINE])

    def unexpected_model_call(*_args, **_kwargs):
        raise AssertionError("closed authoritative correction must not call a model")

    monkeypatch.setattr(face, "_post_streaming", unexpected_model_call)
    monkeypatch.setattr(face, "_post_blocking", unexpected_model_call)
    result = face.chat(
        "No - I mean end-to-end voice latency. Correct the recommendation.",
        session_id=state.session_id,
        model="face-model",
    )

    _assert_gateway_authoritative_latency_correction(result, None)


@pytest.mark.parametrize(
    ("history", "correction"),
    [
        (
            _R31_CORRECTION_BASELINE,
            "If I meant end-to-end voice latency, what would change?",
        ),
        (
            _R31_CORRECTION_BASELINE,
            'Alice said, "No, I mean end-to-end voice latency. Correct it."',
        ),
        (
            _R31_CORRECTION_BASELINE,
            "For discussion only, change to end-to-end voice latency.",
        ),
        (
            _R31_CORRECTION_BASELINE,
            "No - I mean end-to-end voice latency. Correct it, and make Depth synchronous.",
        ),
        (
            _R31_CORRECTION_BASELINE,
            "No - I mean end-to-end voice latency under four seconds. Correct it.",
        ),
        (
            _R31_CORRECTION_BASELINE,
            "No - I mean end-to-end voice latency.",
        ),
        (
            _R31_CORRECTION_BASELINE,
            "`No - I mean end-to-end voice latency. Correct it.`",
        ),
        (
            _R31_CORRECTION_BASELINE,
            "Actually change from end-to-end voice latency to first-token latency.",
        ),
        (
            [
                *_R31_CORRECTION_BASELINE,
                {"role": "user", "content": "Now discuss database backups."},
                {"role": "assistant", "content": "Backups are a separate topic."},
            ],
            "No - I mean end-to-end voice latency. Correct it.",
        ),
        (
            [
                {
                    "role": "user",
                    "content": (
                        "Alice says first-token latency must stay under four seconds."
                    ),
                },
                {"role": "assistant", "content": "That is Alice's requirement."},
            ],
            "No - I mean end-to-end voice latency. Correct it.",
        ),
    ],
)
def test_r31_authoritative_correction_rejects_unowned_or_open_context(
    monkeypatch,
    history,
    correction,
):
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session(f"r31-negative-{uuid.uuid4()}", "face-model")
    state.messages.extend([dict(item) for item in history])
    calls = 0
    model_candidate = f"MODEL_ROUTE_SENTINEL {_R29_ARCHITECTURE}"

    def post_streaming(_payload, callback, *, cancel_event=None):
        nonlocal calls
        del cancel_event
        calls += 1
        callback(model_candidate)
        return model_candidate, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    result = face.chat(
        correction,
        session_id=state.session_id,
        model="face-model",
        stream_callback=lambda _text: True,
    )

    assert calls >= 1
    assert result["api_calls"] >= 1
    assert "authoritative_latency_correction" not in result["output_guard"]
    assert result["metrics"].get("terminal_source") != "gateway_authoritative_render"


def test_r31_authoritative_correction_never_overrides_current_depth_dispatch(
    monkeypatch,
):
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("r31-current-dispatch", "face-model")
    state.messages.extend([dict(item) for item in _R31_CORRECTION_BASELINE])
    calls = 0
    model_candidate = "I am routing this request for deeper analysis."

    def post_streaming(_payload, callback, *, cancel_event=None):
        nonlocal calls
        del cancel_event
        calls += 1
        callback(model_candidate)
        return model_candidate, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    result = face.chat(
        "No - I mean end-to-end voice latency. Correct it.",
        session_id=state.session_id,
        model="face-model",
        extra_system=(
            "- THIS TURN DISPATCHED job da-12345678-1234-1234-1234-123456789abc "
            "to the Depth Lobe."
        ),
        stream_callback=lambda _text: True,
    )

    assert calls == 1
    assert result["api_calls"] == 1
    assert "authoritative_latency_correction" not in result["output_guard"]
    assert "da-12345678-1234-1234-1234-123456789abc" in result["text"]


def test_r31_authoritative_render_validation_drift_fails_closed_without_model(
    monkeypatch,
):
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("r31-renderer-drift", "face-model")
    state.messages.extend([dict(item) for item in _R31_CORRECTION_BASELINE])
    before = list(state.messages)
    model_calls = 0

    def drifted_renderer(_contract, _resolution):
        return "Invalid short gateway prose.", {
            "schema": "Ms4AuthoritativeLatencyCorrection.v1",
            "applied": True,
            "source": "resolved_conversation_contract",
            "model_output_used": False,
            "resolution_schema": "Ms4ResolvedLatencyCorrection.v1",
            "deadline": "under four seconds",
            "deadline_milliseconds": 4000,
            "word_count": 4,
        }

    def unexpected_model_call(_payload, callback, *, cancel_event=None):
        nonlocal model_calls
        del cancel_event
        model_calls += 1
        callback(_R29_ARCHITECTURE)
        return _R29_ARCHITECTURE, _completed_stream_stats()

    monkeypatch.setattr(
        "machine_spirit_4.gateway.face_lobe_chat._render_authoritative_latency_correction",
        drifted_renderer,
    )
    monkeypatch.setattr(face, "_post_streaming", unexpected_model_call)
    spoken: list[str] = []
    result = face.chat(
        "No - I mean end-to-end voice latency. Correct it.",
        session_id=state.session_id,
        model="face-model",
        stream_callback=spoken.append,
    )

    semantic = result["output_guard"]["substantive_followup"]
    assert model_calls == 0
    assert result["api_calls"] == 0
    assert result["completed"] is False
    assert result["text"] == ""
    assert semantic["corrective_regeneration_attempted"] is False
    assert semantic["fail_closed"] is True
    assert "underlength" in semantic["first_candidate_failures"]
    assert spoken == []
    assert state.messages == before


def test_r31_authoritative_correction_rechecks_cancellation_before_delivery(
    monkeypatch,
):
    class CancelAtDelivery:
        def __init__(self):
            self.checks = 0

        def is_set(self):
            self.checks += 1
            return self.checks >= 2

    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("r31-cancel-at-delivery", "face-model")
    state.messages.extend([dict(item) for item in _R31_CORRECTION_BASELINE])
    before = list(state.messages)

    def unexpected_model_call(*_args, **_kwargs):
        raise AssertionError("closed authoritative correction must not call a model")

    monkeypatch.setattr(face, "_post_streaming", unexpected_model_call)
    spoken: list[str] = []
    cancellation = CancelAtDelivery()
    result = face.chat(
        "No - I mean end-to-end voice latency. Correct it.",
        session_id=state.session_id,
        model="face-model",
        stream_callback=spoken.append,
        cancel_event=cancellation,
    )

    assert cancellation.checks >= 2
    assert result["api_calls"] == 0
    assert result["completed"] is False
    assert result["cancelled"] is True
    assert result["guarded_stream_emitted"] is False
    assert spoken == []
    assert state.messages == before


@pytest.mark.parametrize("delivery", ["false", "exception", "cancel_after"])
def test_r31_authoritative_correction_failed_delivery_never_commits_history(
    monkeypatch,
    delivery,
):
    cancellation = threading.Event()
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session(f"r31-delivery-{delivery}", "face-model")
    state.messages.extend([dict(item) for item in _R31_CORRECTION_BASELINE])
    before = list(state.messages)
    delivered: list[str] = []

    def unexpected_model_call(*_args, **_kwargs):
        raise AssertionError("closed authoritative correction must not call a model")

    def delivery_callback(text):
        delivered.append(text)
        if delivery == "false":
            return False
        if delivery == "exception":
            raise RuntimeError("synthetic delivery failure")
        cancellation.set()
        return True

    monkeypatch.setattr(face, "_post_streaming", unexpected_model_call)
    result = face.chat(
        "No - I mean end-to-end voice latency. Correct it.",
        session_id=state.session_id,
        model="face-model",
        stream_callback=delivery_callback,
        cancel_event=cancellation,
    )

    assert len(delivered) == 1
    assert result["api_calls"] == 0
    assert result["completed"] is False
    assert result["cancelled"] is True
    assert state.messages == before


def _r32_complete_evidence_candidate(*, near_miss_words: int) -> str:
    base = (
        "Direct answer: I would change the recommendation if repeated blinded "
        "benchmark evidence contradicts the current diagnosis or supports the "
        "competing explanation; otherwise I would keep it. The test must compare "
        "diagnostic accuracy and error rate, end-to-end latency, GPU memory and "
        "compute cost, and production logs and traces across matched incident "
        "replays. Use the same sample set, thresholds, and confidence criteria for "
        "both lobes, then repeat the experiment until the observed result is stable. "
        "This evidence would falsify the current conclusion only if the alternative "
        "wins consistently rather than in one noisy run."
    )
    padding = {
        119: (
            "Record the control configuration, traffic mix, context length, model "
            "version, and failure labels so reviewers can reproduce the comparison "
            "and distinguish a causal improvement across production nodes."
        ),
        125: (
            "Record the control configuration, traffic mix, context length, model "
            "version, and failure labels so reviewers can reproduce the comparison "
            "and distinguish a causal improvement from a benchmark-specific "
            "coincidence across nodes and production loads."
        ),
    }[near_miss_words]
    candidate = f"{base} {padding}"
    assert len(candidate.split()) == near_miss_words
    return candidate


def _r32_greenhouse_evidence_candidate() -> str:
    candidate = (
        "I would change the recommendation if repeated controlled experiment "
        "evidence contradicts the current greenhouse conclusion or supports the "
        "competing explanation; otherwise I would keep it.\n"
        "1. Growth accuracy: Compare matched crop survival and yield rates; the "
        "alternative must equal or exceed the current schedule across repeated samples.\n"
        "2. Timing: Measure recovery time after watering; the alternative must reduce "
        "response time without increasing plant stress beyond the declared threshold.\n"
        "3. Resource use: Record water and power cost from the same beds; the "
        "alternative must lower resource consumption while the sensor logs confirm "
        "comparable soil moisture.\n"
        "Repeat each observation across seasons and operators so one unusual harvest "
        "cannot decide the recommendation by itself. Document controls and rerun the "
        "comparison before making any operational change across greenhouse beds."
    )
    assert len(candidate.split()) == 125
    return candidate


def test_r32_progressive_short_correction_gets_one_validated_final_rescue(
    monkeypatch,
):
    first = "I can give you the evidence later."
    short_correction = _r32_complete_evidence_candidate(near_miss_words=119)
    rescued = _r32_complete_evidence_candidate(near_miss_words=125)
    attempts = iter((first, short_correction, rescued))
    payloads: list[dict[str, object]] = []
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("r32-evidence-final-rescue", "face-model")
    state.messages.append(
        {"role": "assistant", "content": "The current recommendation prefers Depth."}
    )
    before = list(state.messages)

    def post_streaming(payload, callback, *, cancel_event=None):
        del cancel_event
        payloads.append(payload)
        text = next(attempts)
        callback(text)
        return text, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        "What evidence would change your mind?",
        session_id=state.session_id,
        model="face-model",
        stream_callback=spoken.append,
    )

    guard = result["output_guard"]["substantive_followup"]
    assert result["completed"] is True
    assert result["api_calls"] == 3
    assert result["text"] == rescued
    assert spoken == [rescued]
    assert state.messages == before + [
        {"role": "user", "content": "What evidence would change your mind?"},
        {"role": "assistant", "content": rescued},
    ]
    assert guard["corrective_attempt_count"] == 2
    assert guard["final_rescue_attempted"] is True
    assert guard["first_corrective_candidate_failures"] == ["underlength"]
    assert guard["first_corrective_candidate_word_count"] == 119
    assert guard["corrective_candidate_failures"] == []
    assert [payload["temperature"] for payload in payloads] == [0.2, 0.0, 0.0]
    assert "max_tokens" not in payloads[0]
    assert payloads[1]["max_tokens"] == 1024
    assert payloads[2]["max_tokens"] == 1024
    rescue_system = str(payloads[2]["messages"][0]["content"])
    assert "bounded attempt 2 of 2" in rescue_system
    assert (
        "previous corrective candidate stopped at 119 visible words"
        in rescue_system.lower()
    )
    assert "because: underlength" in rescue_system


def test_r32_complete_125_word_corrective_evidence_near_miss_is_completed(
    monkeypatch,
):
    corrected = _r32_complete_evidence_candidate(near_miss_words=125)
    attempts = iter(("I can give you the evidence later.", corrected))
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("r32-evidence-near-miss", "face-model")
    state.messages.append(
        {"role": "assistant", "content": "The current recommendation prefers Depth."}
    )
    before = list(state.messages)

    def post_streaming(_payload, callback, *, cancel_event=None):
        del cancel_event
        text = next(attempts)
        callback(text)
        return text, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        "What evidence would change your mind?",
        session_id=state.session_id,
        model="face-model",
        stream_callback=spoken.append,
    )

    guard = result["output_guard"]["substantive_followup"]
    decision_receipt = guard["corrective_evidence_decision_rule_postcondition"]
    assert result["completed"] is True
    assert result["fallback_used"] is False
    assert result["api_calls"] == 2
    assert result["text"] == corrected
    assert spoken == [result["text"]]
    assert state.messages == before + [
        {"role": "user", "content": "What evidence would change your mind?"},
        {"role": "assistant", "content": result["text"]},
    ]
    assert guard["min_words"] == 120
    assert decision_receipt["applied"] is False
    assert decision_receipt["candidate_word_count"] == 125
    assert decision_receipt["projected_word_count"] > 125
    assert decision_receipt["resulting_word_count"] == 125
    assert guard["corrective_candidate_failures"] == []
    assert guard["fail_closed"] is False


@pytest.mark.parametrize(
    ("corrected", "expected_words"),
    [
        (_r32_complete_evidence_candidate(near_miss_words=119), 119),
        (" ".join(["context"] * 125), 125),
    ],
)
def test_r32_short_or_semantically_empty_evidence_still_fails_closed(
    monkeypatch,
    corrected,
    expected_words,
):
    attempts = iter(("I can give you the evidence later.", corrected, corrected))
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session(
        f"r32-evidence-reject-{expected_words}",
        "face-model",
    )
    state.messages.append(
        {"role": "assistant", "content": "The current recommendation prefers Depth."}
    )
    before = list(state.messages)

    def post_streaming(_payload, callback, *, cancel_event=None):
        del cancel_event
        text = next(attempts)
        callback(text)
        return text, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        "What evidence would change your mind?",
        session_id=state.session_id,
        model="face-model",
        stream_callback=spoken.append,
    )

    guard = result["output_guard"]["substantive_followup"]
    assert result["completed"] is False
    assert result["text"] == ""
    expected_calls = 3 if expected_words == 119 else 2
    assert result["api_calls"] == expected_calls
    assert spoken == []
    assert state.messages == before
    assert guard["min_words"] == 120
    if expected_words == 119:
        assert "underlength" in guard["corrective_candidate_failures"]
    else:
        assert "underlength" not in guard["corrective_candidate_failures"]
    assert guard["corrective_candidate_failures"]
    assert guard["corrective_attempt_count"] == expected_calls - 1
    assert guard["final_rescue_attempted"] is (expected_calls == 3)
    assert guard["first_corrective_candidate_word_count"] == expected_words
    assert guard["fail_closed"] is True


def test_r32_complete_greenhouse_evidence_is_delivered_without_injection(
    monkeypatch,
):
    corrected = _r32_greenhouse_evidence_candidate()
    attempts = iter(("I can give you the evidence later.", corrected))
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("r32-greenhouse-near-miss", "face-model")
    state.messages.append(
        {"role": "assistant", "content": "Use the current greenhouse irrigation schedule."}
    )

    def post_streaming(_payload, callback, *, cancel_event=None):
        del cancel_event
        text = next(attempts)
        callback(text)
        return text, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        "What evidence would change your mind?",
        session_id=state.session_id,
        model="face-model",
        stream_callback=spoken.append,
    )

    guard = result["output_guard"]["substantive_followup"]
    assert result["completed"] is True
    assert result["text"] == corrected
    assert guard["min_words"] == 120
    assert guard["corrective_candidate_failures"] == []
    assert spoken == [result["text"]]


_R40_COMPARISON_GOAL = (
    "Compare the 35B Depth lobe with the 4B Face lobe for diagnosing an "
    "intermittent distributed inference failure and give a recommendation."
)

_R40_Q20_REJECTED_TURN_2 = (
    "The second trade-off centers on compute cost, specifically how parameter "
    "count translates to GPU memory usage, inference latency, and operational "
    "expense. A 35B model requires significantly more VRAM than a 4B model, often "
    "necessitating larger instances or more efficient quantization strategies to "
    "fit within hardware limits. This increased memory footprint can lead to higher "
    "baseline costs per hour of runtime. Furthermore, the computational graph for a "
    "35B model is heavier, meaning each inference step consumes more GPU cycles. In "
    "high-throughput scenarios, this can reduce the number of concurrent requests "
    "your system can handle on a single node, potentially requiring more nodes to "
    "maintain the same service level objectives.\n\n"
    "However, parameter count alone does not dictate efficiency. Factors like "
    "architecture design, quantization precision, and serving stack optimization "
    "play crucial roles. For instance, a well-quantized 35B model might outperform "
    "an unoptimized 4B model in terms of tokens per second on specific hardware. The "
    "key is to measure actual resource consumption under your specific workload "
    "conditions rather than assuming linear scaling with parameter size. You must "
    "benchmark both models using your exact serving configuration, including batch "
    "sizes, sequence lengths, and hardware specifications, to determine the true "
    "cost-benefit ratio for your use case.\n\n"
    "Consider a scenario where you are running a real-time chat application. A 4B "
    "model might process requests faster on a single GPU, allowing you to serve more "
    "users with fewer resources. In contrast, a 35B model might provide better "
    "reasoning capabilities but require two GPUs per instance, doubling your "
    "infrastructure costs. The decision should hinge on whether the quality "
    "improvement from the larger model justifies the increased expense and latency "
    "for your specific application requirements."
)


def _r40_safe_compute_tradeoff_candidate() -> str:
    candidate = (
        "Compute or cost is the second trade-off, and the direct answer is that the "
        "verified comparison does not assign an automatic resource penalty to either "
        "candidate. Parameter count alone does not establish actual GPU memory, VRAM, "
        "compute, latency, or operating cost. Those outcomes depend on the candidate's "
        "architecture, quantization, offload policy, batching, utilization, context "
        "length, tools, and serving route. Either candidate may use more or less of a "
        "resource on the selected stack, so the mechanism is empirical rather than a "
        "parameter-count shortcut.\n\n"
        "For example, suppose operator Maya replays the same distributed-inference "
        "incident through a 35B Depth candidate and a 4B Face candidate on the lab "
        "cluster. She holds prompts, logs, traces, context, tools, and compute budget "
        "constant. When each replay completes, the harness records peak GPU memory, "
        "VRAM allocation, accelerator utilization, queue delay, token throughput, "
        "energy use, cost per successful diagnosis, and ground-truth accuracy. That "
        "controlled run gives Maya an observable consequence and a practical action: "
        "select a route only after repeated measurements are stable. Suppose a controlled "
        "replay reserves one production accelerator. The diagnostic reservation consumes "
        "that accelerator. Therefore, live inference throughput falls during the debugging "
        "window. Maya then moves the replay to isolated capacity and repeats the test.\n\n"
        "The decision rule therefore remains calibrated. If controlled benchmarks "
        "show one candidate has a repeatable quality gain that justifies its measured "
        "resource use, prefer it for that incident class; otherwise keep the cheaper "
        "or faster measured route. Until those observations exist, describe any "
        "directional difference as a hypothesis, not as a fact inferred from 35B or "
        "4B. This preserves the Depth result while giving the operator a concrete way "
        "to decide."
    )
    assert 150 <= len(candidate.split()) <= 380
    return candidate


def test_r40_canonical_plain_heading_resolves_hyphenated_second_tradeoff():
    history = [
        {
            "role": "assistant",
            "content": (
                "[Verified Depth Lobe result; job_id=da-r40-parser]\n\n"
                + _DEPTH_EVIDENCE_FREE_COMPARISON_TEMPLATE
            ),
        }
    ]

    expectation = _ordinal_expectation(
        "Go deeper on the second trade-off.",
        history,
    )

    assert _markdown_section_heading("Tradeoffs:") == "Tradeoffs"
    assert _markdown_section_heading("Tradeoffs: latency and cost") is None
    assert expectation is not None
    assert expectation["ordinal"] == 2
    assert expectation["label"] == "Compute or cost"
    assert _ordinal_item_label(expectation["item"]) == "Compute or cost"


def test_r40_exact_q20_contradiction_is_rejected_by_typed_truth_contract():
    contract = _comparison_truth_contract_for_verified_depth(
        _DEPTH_EVIDENCE_FREE_COMPARISON_TEMPLATE,
        _R40_COMPARISON_GOAL,
        "da-r40-truth",
    )

    assert contract is not None
    assert _comparison_truth_failures(_R40_Q20_REJECTED_TURN_2, contract) == [
        "inherited_comparison_parameter_causality",
        "inherited_comparison_unverified_numeric_claim",
    ]
    assert _comparison_truth_failures(
        _r40_safe_compute_tradeoff_candidate(),
        contract,
    ) == []


def test_r40_bound_depth_truth_rejects_bad_face_then_delivers_safe_retry(
    monkeypatch,
):
    safe = _r40_safe_compute_tradeoff_candidate()
    attempts = iter((_R40_Q20_REJECTED_TURN_2, safe))
    payloads: list[dict[str, object]] = []
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("r40-bad-then-safe", "face-model")
    receipt = face.bind_depth_result(
        conversation_id=state.session_id,
        job_id="da-r40-bad-then-safe",
        result_text=_DEPTH_EVIDENCE_FREE_COMPARISON_TEMPLATE,
        goal=_R40_COMPARISON_GOAL,
        model="depth-model",
    )
    before = list(state.messages)

    def post_streaming(payload, callback, *, cancel_event=None):
        del cancel_event
        payloads.append(payload)
        answer = next(attempts)
        callback(answer)
        return answer, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        "Go deeper on the second trade-off.",
        session_id=state.session_id,
        model="face-model",
        stream_callback=spoken.append,
        substantive_word_range=(150, 380),
    )

    guard = result["output_guard"]["substantive_followup"]
    assert receipt["comparison_truth_contract_bound"] is True
    assert receipt["comparison_truth_contract_schema"] == (
        "Ms4EvidenceFreeModelComparisonTruthContract.v1"
    )
    assert guard["ordinal_expectation"]["label"] == "Compute or cost"
    assert {
        "inherited_comparison_parameter_causality",
        "inherited_comparison_unverified_numeric_claim",
    }.issubset(set(guard["first_candidate_failures"]))
    assert result["completed"] is True
    assert result["api_calls"] == 2
    assert result["text"] == safe
    assert spoken == [safe]
    assert _R40_Q20_REJECTED_TURN_2 not in [
        message["content"] for message in state.messages
    ]
    assert state.messages == before + [
        {"role": "user", "content": "Go deeper on the second trade-off."},
        {"role": "assistant", "content": safe},
    ]
    assert "parameter count alone establishes none" in str(
        payloads[1]["messages"][0]["content"]
    )


def test_r40_two_contradictory_face_attempts_use_authoritative_safe_rescue(
    monkeypatch,
):
    attempts = iter((_R40_Q20_REJECTED_TURN_2,) * 3)
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("r40-bad-twice", "face-model")
    face.bind_depth_result(
        conversation_id=state.session_id,
        job_id="da-r40-bad-twice",
        result_text=_DEPTH_EVIDENCE_FREE_COMPARISON_TEMPLATE,
        goal=_R40_COMPARISON_GOAL,
        model="depth-model",
    )
    before = list(state.messages)

    def post_streaming(_payload, callback, *, cancel_event=None):
        del cancel_event
        answer = next(attempts)
        callback(answer)
        return answer, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        "Go deeper on the second trade-off.",
        session_id=state.session_id,
        model="face-model",
        stream_callback=spoken.append,
        substantive_word_range=(150, 380),
    )

    guard = result["output_guard"]["substantive_followup"]
    assert result["completed"] is True
    assert result["text"] == _r40_safe_compute_tradeoff_candidate()
    assert result["api_calls"] == 2
    assert guard["fail_closed"] is False
    assert guard["corrective_candidate_failures"] == []
    assert guard["authoritative_contract_rescue_attempted"] is True
    assert result["output_guard"]["authoritative_comparison_truth_rescue"] == {
        "schema": "Ms4AuthoritativeComparisonTruthRescue.v1",
        "applied": True,
        "source": "verified_depth_comparison_contract",
        "model_output_used": False,
        "source_job_id": "da-r40-bad-twice",
        "ordinal": 2,
        "label": "Compute or cost",
        "word_count": len(_r40_safe_compute_tradeoff_candidate().split()),
    }
    assert spoken == [_r40_safe_compute_tradeoff_candidate()]
    assert state.messages == before + [
        {"role": "user", "content": "Go deeper on the second trade-off."},
        {"role": "assistant", "content": _r40_safe_compute_tradeoff_candidate()},
    ]


def test_r40_authoritative_safe_rescue_is_not_available_without_typed_truth():
    text, receipt = _render_authoritative_comparison_truth_rescue(
        {
            "intent": "explicit_expansion",
            "min_words": 150,
            "max_words": 380,
            "ordinal_expectation": {"ordinal": 2, "label": "Compute or cost"},
            "comparison_truth_contract": {
                "schema": "Ms4EvidenceFreeModelComparisonTruthContract.v1",
                "source_job_id": "",
                "verified_empirical_comparison_available": False,
            },
        }
    )

    assert text is None
    assert receipt is None


def test_r54_unbalanced_model_prose_uses_authoritative_safe_rescue(monkeypatch):
    malformed = _r40_safe_compute_tradeoff_candidate().replace(
        "candidate's architecture",
        "candidate' architecture",
    )
    attempts = iter((malformed, malformed))
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("r54-unbalanced-prose", "face-model")
    face.bind_depth_result(
        conversation_id=state.session_id,
        job_id="da-r54-unbalanced-prose",
        result_text=_DEPTH_EVIDENCE_FREE_COMPARISON_TEMPLATE,
        goal=_R40_COMPARISON_GOAL,
        model="depth-model",
    )

    def post_streaming(_payload, callback, *, cancel_event=None):
        del cancel_event
        answer = next(attempts)
        callback(answer)
        return answer, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        "Go deeper on the second trade-off.",
        session_id=state.session_id,
        model="face-model",
        stream_callback=spoken.append,
        substantive_word_range=(150, 380),
    )

    guard = result["output_guard"]["substantive_followup"]
    assert guard["first_candidate_failures"] == ["unbalanced_prose_delimiter"]
    assert result["output_guard"]["reason"] == (
        "authoritative_comparison_truth_rescue"
    )
    assert result["text"] == _r40_safe_compute_tradeoff_candidate()
    assert "candidate' architecture" not in result["text"]
    assert spoken == [result["text"]]


def test_r42_authoritative_evidence_rescue_passes_full_typed_contract():
    contract = {
        "intent": "evidence_reconsideration",
        "min_words": 120,
        "max_words": 380,
        "ordinal_expectation": None,
        "comparison_truth_contract": {
            "schema": "Ms4EvidenceFreeModelComparisonTruthContract.v1",
            "source_job_id": "da-r42-evidence",
            "verified_empirical_comparison_available": False,
            "parameter_count_causality_allowed": False,
        },
    }

    text, receipt = _render_authoritative_comparison_truth_rescue(contract)

    assert text is not None
    assert receipt is not None
    assert receipt["schema"] == "Ms4AuthoritativeComparisonEvidenceRescue.v1"
    assert receipt["intent"] == "evidence_reconsideration"
    assert receipt["measurement_family_count"] >= 3
    assert _substantive_followup_failures(text, contract, max_words=380) == []


def test_r42_repeated_bad_evidence_answers_use_authoritative_typed_rescue(
    monkeypatch,
):
    attempts = iter((_R40_Q20_REJECTED_TURN_2,) * 3)
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("r42-evidence-rescue", "face-model")
    face.bind_depth_result(
        conversation_id=state.session_id,
        job_id="da-r42-evidence-rescue",
        result_text=_DEPTH_EVIDENCE_FREE_COMPARISON_TEMPLATE,
        goal=_R40_COMPARISON_GOAL,
        model="depth-model",
    )
    before = list(state.messages)

    def post_streaming(_payload, callback, *, cancel_event=None):
        del cancel_event
        answer = next(attempts)
        callback(answer)
        return answer, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        "What evidence would change your mind?",
        session_id=state.session_id,
        model="face-model",
        stream_callback=spoken.append,
        substantive_word_range=(150, 380),
    )

    guard = result["output_guard"]["substantive_followup"]
    receipt = result["output_guard"]["authoritative_comparison_truth_rescue"]
    assert result["completed"] is True
    assert result["api_calls"] == 2
    assert receipt["schema"] == "Ms4AuthoritativeComparisonEvidenceRescue.v1"
    assert receipt["intent"] == "evidence_reconsideration"
    assert guard["fail_closed"] is False
    assert guard["corrective_candidate_failures"] == []
    assert spoken == [result["text"]]
    assert state.messages == before + [
        {"role": "user", "content": "What evidence would change your mind?"},
        {"role": "assistant", "content": result["text"]},
    ]


def test_r40_noncanonical_depth_result_does_not_activate_comparison_truth():
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("r40-noncanonical", "face-model")
    receipt = face.bind_depth_result(
        conversation_id=state.session_id,
        job_id="da-r40-noncanonical",
        result_text=(
            "A verified resource comparison with measured GPU occupancy supports "
            "the selected deployment route."
        ),
        goal=_R40_COMPARISON_GOAL,
        model="depth-model",
    )

    assert receipt["comparison_truth_contract_bound"] is False
    assert state.comparison_truth_contract is None
    assert _comparison_truth_failures(_R40_Q20_REJECTED_TURN_2, None) == []


def test_r41_truth_activation_is_exact_but_line_ending_tolerant():
    expected_sha = hashlib.sha256(
        _DEPTH_EVIDENCE_FREE_COMPARISON_TEMPLATE.encode("utf-8")
    ).hexdigest()
    accepted = (
        _DEPTH_EVIDENCE_FREE_COMPARISON_TEMPLATE,
        _DEPTH_EVIDENCE_FREE_COMPARISON_TEMPLATE.replace("\n", "\r\n"),
        "\n".join(
            f"{line}  "
            for line in _DEPTH_EVIDENCE_FREE_COMPARISON_TEMPLATE.split("\n")
        )
        + "\n",
        _DEPTH_EVIDENCE_FREE_COMPARISON_TEMPLATE + "\n",
    )
    rejected = (
        "\n" + _DEPTH_EVIDENCE_FREE_COMPARISON_TEMPLATE,
        _DEPTH_EVIDENCE_FREE_COMPARISON_TEMPLATE + "\n\n",
        _DEPTH_EVIDENCE_FREE_COMPARISON_TEMPLATE + " Contradictory suffix.",
        (
            "parameter count alone supplies no causal mechanism. "
            "either candidate may use more or less gpu/vram/compute. "
            "rather than infer it from parameter count. "
            "only controlled benchmark results can establish whether either "
            "candidate is better"
        ),
    )

    for result_text in accepted:
        contract = _comparison_truth_contract_for_verified_depth(
            result_text,
            _R40_COMPARISON_GOAL,
            "da-r41-normalized",
        )
        assert contract is not None
        assert contract["source_result_sha256"] == expected_sha
    for result_text in rejected:
        assert _comparison_truth_contract_for_verified_depth(
            result_text,
            _R40_COMPARISON_GOAL,
            "da-r41-rejected",
        ) is None


@pytest.mark.parametrize(
    ("candidate", "expected"),
    (
        (
            "Parameter count alone cannot establish latency, but the 35B "
            "requires more VRAM and has better reasoning.",
            {
                "inherited_comparison_parameter_causality",
                "inherited_comparison_epistemic_overclaim",
            },
        ),
        (
            "If benchmarking is impossible, the 35B requires more VRAM and "
            "has better reasoning.",
            {
                "inherited_comparison_parameter_causality",
                "inherited_comparison_epistemic_overclaim",
            },
        ),
        (
            "If controlled measurements show no latency difference, the 35B "
            "requires more VRAM on this stack.",
            {"inherited_comparison_parameter_causality"},
        ),
        (
            "The 35B does not require more VRAM than the 4B.",
            {"inherited_comparison_parameter_causality"},
        ),
    ),
)
def test_r41_disavowal_or_unrelated_evidence_never_licenses_a_claim(
    candidate,
    expected,
):
    contract = _comparison_truth_contract_for_verified_depth(
        _DEPTH_EVIDENCE_FREE_COMPARISON_TEMPLATE,
        _R40_COMPARISON_GOAL,
        "da-r41-adversarial",
    )

    assert set(_comparison_truth_failures(candidate, contract)) == expected


@pytest.mark.parametrize(
    "candidate",
    (
        (
            "A working hypothesis is that the 35B requires more VRAM on this "
            "stack; benchmark both candidates before deciding."
        ),
        (
            "The 35B may require more VRAM on this stack; controlled benchmarks "
            "must decide."
        ),
        (
            "Being 35B does not itself require more VRAM; parameter count alone "
            "supplies no causal mechanism."
        ),
        "It is false that the 35B requires more VRAM and has better reasoning.",
        (
            "If a controlled benchmark shows the 35B uses two GPUs on the matched "
            "serving stack, record that observed cost."
        ),
    ),
)
def test_r41_scoped_hypothesis_disavowal_or_same_claim_evidence_passes(candidate):
    contract = _comparison_truth_contract_for_verified_depth(
        _DEPTH_EVIDENCE_FREE_COMPARISON_TEMPLATE,
        _R40_COMPARISON_GOAL,
        "da-r41-safe",
    )

    assert _comparison_truth_failures(candidate, contract) == []


@pytest.mark.parametrize(
    ("candidate", "expected"),
    (
        (
            "This is not a hypothesis: the 35B requires more VRAM and has "
            "better reasoning.",
            {
                "inherited_comparison_parameter_causality",
                "inherited_comparison_epistemic_overclaim",
            },
        ),
        (
            "The 35B may affect latency and definitely requires more VRAM.",
            {"inherited_comparison_parameter_causality"},
        ),
        (
            "It is false that the 35B requires more VRAM, so we should note "
            "that the 35B has better reasoning.",
            {"inherited_comparison_epistemic_overclaim"},
        ),
        (
            "If controlled measurements show the 35B has unchanged latency, "
            "it requires more VRAM.",
            {"inherited_comparison_parameter_causality"},
        ),
        (
            "If controlled measurements show the 35B has lower latency, it "
            "has better reasoning.",
            {"inherited_comparison_epistemic_overclaim"},
        ),
        (
            "The 35B may require more VRAM on this stack; it has better "
            "reasoning.",
            {"inherited_comparison_epistemic_overclaim"},
        ),
        (
            "Reasoning is better with the 35B than the 4B.",
            {"inherited_comparison_epistemic_overclaim"},
        ),
        (
            "The 35B is smarter than the 4B.",
            {"inherited_comparison_epistemic_overclaim"},
        ),
    ),
)
def test_r42_qualifier_scope_never_bleeds_into_a_later_proposition(
    candidate,
    expected,
):
    contract = _comparison_truth_contract_for_verified_depth(
        _DEPTH_EVIDENCE_FREE_COMPARISON_TEMPLATE,
        _R40_COMPARISON_GOAL,
        "da-r42-scope",
    )

    assert set(_comparison_truth_failures(candidate, contract)) == expected


@pytest.mark.parametrize(
    "candidate",
    (
        "A 35B label does not imply better reasoning.",
        "It would be unsupported to say the 35B requires more VRAM.",
        "No evidence shows that the 35B requires more VRAM.",
        (
            "The fact that one candidate is 35B is insufficient to conclude "
            "it needs more VRAM."
        ),
        "The 35B is not proven to have better reasoning.",
        (
            "In a hypothetical equal-input replay, the 35B and 4B candidates "
            "each receive one GPU."
        ),
        (
            "For the controlled benchmark, allocate one GPU to both the 35B "
            "and 4B candidates."
        ),
    ),
)
def test_r42_causal_disavowal_and_symmetric_control_numbers_pass(candidate):
    contract = _comparison_truth_contract_for_verified_depth(
        _DEPTH_EVIDENCE_FREE_COMPARISON_TEMPLATE,
        _R40_COMPARISON_GOAL,
        "da-r42-safe",
    )

    assert _comparison_truth_failures(candidate, contract) == []


def test_r43_live_cpu_offload_contradiction_is_rejected_by_typed_truth_contract():
    contract = _comparison_truth_contract_for_verified_depth(
        _DEPTH_EVIDENCE_FREE_COMPARISON_TEMPLATE,
        _R40_COMPARISON_GOAL,
        "da-r43-live-offload",
    )
    candidate = (
        "If you quantize the 35B model to INT8 but run the 4B model in FP16, "
        "the 35B model might fit into memory while the 4B one requires expensive "
        "offloading to CPU, increasing latency and cost."
    )

    assert contract is not None
    assert "inherited_comparison_parameter_causality" in _comparison_truth_failures(
        candidate,
        contract,
    )


def test_r56_concessive_higher_cost_claim_is_rejected_by_typed_truth_contract():
    contract = _comparison_truth_contract_for_verified_depth(
        _DEPTH_EVIDENCE_FREE_COMPARISON_TEMPLATE,
        _R40_COMPARISON_GOAL,
        "da-r56-concessive-cost",
    )
    candidate = _r32_complete_evidence_candidate(near_miss_words=125) + (
        " Concretely, I would change the recommendation if controlled matched "
        "replays show the 35B candidate has higher diagnostic accuracy and fewer "
        "false negatives while using less compute. I would then deploy the 35B "
        "candidate, accepting its higher cost and latency."
    )

    assert contract is not None
    assert _comparison_truth_failures(candidate, contract) == [
        "inherited_comparison_parameter_causality"
    ]


@pytest.mark.parametrize(
    ("candidate", "expected"),
    (
        (
            "Suppose a controlled benchmark uses identical hardware: the 35B uses "
            "80GB VRAM while the 4B uses 12GB, so the 35B cost is higher and latency "
            "longer.",
            {
                "inherited_comparison_parameter_causality",
                "inherited_comparison_unverified_numeric_claim",
            },
        ),
        (
            "In a hypothetical controlled benchmark, the 35B uses 80GB and the 4B "
            "uses 12GB.",
            {"inherited_comparison_unverified_numeric_claim"},
        ),
        (
            "In a hypothetical equal-input replay, the 35B and 4B candidates each "
            "receive one GPU.",
            set(),
        ),
        (
            "For the controlled benchmark, allocate 80GB to both the 35B and 4B "
            "candidates.",
            set(),
        ),
    ),
)
def test_r58_controlled_examples_require_symmetric_quantities_and_no_fake_conclusion(
    candidate,
    expected,
):
    contract = _comparison_truth_contract_for_verified_depth(
        _DEPTH_EVIDENCE_FREE_COMPARISON_TEMPLATE,
        _R40_COMPARISON_GOAL,
        "da-r58-controlled-asymmetry",
    )

    assert contract is not None
    assert set(_comparison_truth_failures(candidate, contract)) == expected


@pytest.mark.parametrize(
    "candidate",
    (
        "If the 35B uses more accelerator time, Maya might accept it.",
        "The 35B has lower latency on the selected route.",
    ),
)
def test_r60_unlabeled_resource_predicates_preserve_verified_comparison_authority(
    candidate,
):
    contract = _comparison_truth_contract_for_verified_depth(
        _DEPTH_EVIDENCE_FREE_COMPARISON_TEMPLATE,
        _R40_COMPARISON_GOAL,
        "da-r60-unlabeled-candidate-predicate",
    )

    assert contract is not None
    assert "inherited_comparison_parameter_causality" in _comparison_truth_failures(
        candidate,
        contract,
    )


@pytest.mark.parametrize(
    "candidate",
    (
        "If both use similar resources yet the 4B is accurate and faster, she keeps it.",
        "The 4B has higher reliability for this diagnosis.",
    ),
)
def test_r60_unlabeled_quality_predicates_preserve_verified_comparison_authority(
    candidate,
):
    contract = _comparison_truth_contract_for_verified_depth(
        _DEPTH_EVIDENCE_FREE_COMPARISON_TEMPLATE,
        _R40_COMPARISON_GOAL,
        "da-r60-unlabeled-quality-predicate",
    )

    assert contract is not None
    assert "inherited_comparison_epistemic_overclaim" in _comparison_truth_failures(
        candidate,
        contract,
    )


def test_r60_exact_live_turn_three_scenario_is_rejected_before_delivery():
    contract = _comparison_truth_contract_for_verified_depth(
        _DEPTH_EVIDENCE_FREE_COMPARISON_TEMPLATE,
        _R40_COMPARISON_GOAL,
        "da-r60-live-turn-three",
    )
    candidate = (
        "If the 35B uses more accelerator time but yields a correct diagnosis while "
        "the 4B fails twice, she might accept it; if both use similar resources yet "
        "the 4B is accurate and faster, she keeps it."
    )

    assert contract is not None
    assert set(_comparison_truth_failures(candidate, contract)) == {
        "inherited_comparison_parameter_causality",
        "inherited_comparison_epistemic_overclaim",
    }


@pytest.mark.parametrize(
    "candidate",
    (
        (
            "If a controlled benchmark shows the 35B uses more accelerator time "
            "on the matched route, record that observed cost."
        ),
        (
            "A working hypothesis is that the 4B is accurate for this diagnosis; "
            "controlled replays must decide."
        ),
    ),
)
def test_r60_same_claim_evidence_or_explicit_hypothesis_remains_allowed(candidate):
    contract = _comparison_truth_contract_for_verified_depth(
        _DEPTH_EVIDENCE_FREE_COMPARISON_TEMPLATE,
        _R40_COMPARISON_GOAL,
        "da-r60-qualified-candidate-predicate",
    )

    assert contract is not None
    assert _comparison_truth_failures(candidate, contract) == []


def test_r60_exact_live_final_rescue_candidate_never_crosses_delivery_boundary(
    monkeypatch,
):
    live_rejected = (
        "I would change the recommendation if controlled benchmarks show that the "
        "4B Face candidate has higher diagnostic accuracy and lower false negatives "
        "than the 35B Depth candidate under identical prompts, context, tools, compute "
        "budget, and serving route; otherwise I would keep it.\n\n"
        "Mechanism: The evidence must come from repeated measurements across three "
        "independent families - diagnostic accuracy (ground-truth alignment), timing "
        "(end-to-end latency), and resource use (GPU, VRAM, and accelerator consumption) "
        "- all held constant except the candidate model. If any measurement family "
        "shows a consistent, repeatable advantage for one candidate, that overrides "
        "the hypothesis that size alone implies performance.\n\n"
        "Practical consequence: Maya would rerun the incident replay with the winning "
        "candidate on live capacity, then compare its actual resource cost and diagnostic "
        "outcome to her original plan. If the 35B uses more accelerator time but yields "
        "a correct diagnosis while the 4B fails twice, she might accept it; if both use "
        "similar resources yet the 4B is accurate and faster, she keeps it."
    )
    attempts = iter(
        (
            "I can give you the evidence later.",
            _r32_complete_evidence_candidate(near_miss_words=119),
            live_rejected,
        )
    )
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("r60-live-turn-three-delivery", "face-model")
    face.bind_depth_result(
        conversation_id=state.session_id,
        job_id="da-r60-live-turn-three-delivery",
        result_text=_DEPTH_EVIDENCE_FREE_COMPARISON_TEMPLATE,
        goal=_R40_COMPARISON_GOAL,
        model="depth-model",
    )
    before = list(state.messages)

    def post_streaming(_payload, callback, *, cancel_event=None):
        del cancel_event
        answer = next(attempts)
        callback(answer)
        return answer, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        "What evidence would change your mind?",
        session_id=state.session_id,
        model="face-model",
        stream_callback=spoken.append,
        substantive_word_range=(120, 380),
    )

    guard = result["output_guard"]["substantive_followup"]
    assert result["completed"] is True
    assert result["api_calls"] == 3
    assert guard["final_rescue_attempted"] is True
    assert guard["authoritative_contract_rescue_attempted"] is True
    assert result["output_guard"]["authoritative_comparison_truth_rescue"][
        "model_output_used"
    ] is False
    assert result["text"] != live_rejected
    assert live_rejected not in [message["content"] for message in state.messages]
    assert spoken == [result["text"]]
    assert state.messages == before + [
        {"role": "user", "content": "What evidence would change your mind?"},
        {"role": "assistant", "content": result["text"]},
    ]


def test_r56_concessive_higher_cost_claim_is_repaired_before_delivery(
    monkeypatch,
):
    rejected = _r32_complete_evidence_candidate(near_miss_words=125) + (
        " Concretely, I would change the recommendation if controlled matched "
        "replays show the 35B candidate has higher diagnostic accuracy and fewer "
        "false negatives while using less compute. I would then deploy the 35B "
        "candidate, accepting its higher cost and latency."
    )
    corrected = _r32_complete_evidence_candidate(near_miss_words=125)
    attempts = iter((rejected, corrected))
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("r56-concessive-cost-repair", "face-model")
    face.bind_depth_result(
        conversation_id=state.session_id,
        job_id="da-r56-concessive-cost-repair",
        result_text=_DEPTH_EVIDENCE_FREE_COMPARISON_TEMPLATE,
        goal=_R40_COMPARISON_GOAL,
        model="depth-model",
    )
    before = list(state.messages)

    def post_streaming(_payload, callback, *, cancel_event=None):
        del cancel_event
        answer = next(attempts)
        callback(answer)
        return answer, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        "What evidence would change your mind?",
        session_id=state.session_id,
        model="face-model",
        stream_callback=spoken.append,
        substantive_word_range=(120, 380),
    )

    guard = result["output_guard"]["substantive_followup"]
    assert "inherited_comparison_parameter_causality" in guard[
        "first_candidate_failures"
    ]
    assert guard["corrective_candidate_failures"] == []
    assert result["completed"] is True
    assert result["api_calls"] == 2
    assert result["text"] == corrected
    assert spoken == [corrected]
    assert rejected not in [message["content"] for message in state.messages]
    assert state.messages == before + [
        {"role": "user", "content": "What evidence would change your mind?"},
        {"role": "assistant", "content": corrected},
    ]


def test_r58_fabricated_controlled_benchmark_is_repaired_before_delivery(
    monkeypatch,
):
    rejected = _r32_complete_evidence_candidate(near_miss_words=125) + (
        " Suppose a controlled benchmark uses identical hardware: the 35B uses "
        "80GB VRAM while the 4B uses 12GB, so the 35B cost is higher and latency "
        "longer."
    )
    corrected = _r32_complete_evidence_candidate(near_miss_words=125)
    attempts = iter((rejected, corrected))
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("r58-fabricated-benchmark-repair", "face-model")
    face.bind_depth_result(
        conversation_id=state.session_id,
        job_id="da-r58-fabricated-benchmark-repair",
        result_text=_DEPTH_EVIDENCE_FREE_COMPARISON_TEMPLATE,
        goal=_R40_COMPARISON_GOAL,
        model="depth-model",
    )
    before = list(state.messages)
    calls = 0

    def post_streaming(_payload, callback, *, cancel_event=None):
        nonlocal calls
        del cancel_event
        calls += 1
        answer = next(attempts)
        callback(answer)
        return answer, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    spoken: list[str] = []
    result = face.chat(
        "What evidence would change your mind?",
        session_id=state.session_id,
        model="face-model",
        stream_callback=spoken.append,
        substantive_word_range=(120, 380),
    )

    guard = result["output_guard"]["substantive_followup"]
    assert set(guard["first_candidate_failures"]) >= {
        "inherited_comparison_parameter_causality",
        "inherited_comparison_unverified_numeric_claim",
    }
    assert guard["corrective_candidate_failures"] == []
    assert guard["fail_closed"] is False
    assert result["completed"] is True
    assert result["api_calls"] == 2
    assert calls == 2
    assert result["text"] == corrected
    assert spoken == [corrected]
    assert rejected not in [message["content"] for message in state.messages]
    assert state.messages == before + [
        {"role": "user", "content": "What evidence would change your mind?"},
        {"role": "assistant", "content": corrected},
    ]


def test_r43_authoritative_comparison_rescue_uses_hard_cap_without_voice_range():
    text, receipt = _render_authoritative_comparison_truth_rescue(
        {
            "intent": "explicit_expansion",
            "min_words": 130,
            "ordinal_expectation": {"ordinal": 2, "label": "Compute or cost"},
            "comparison_truth_contract": {
                "schema": "Ms4EvidenceFreeModelComparisonTruthContract.v1",
                "source_job_id": "da-r43-api-rescue",
                "verified_empirical_comparison_available": False,
            },
        }
    )

    assert text is not None
    assert receipt is not None
    assert 130 <= receipt["word_count"] <= 600


def test_r43_live_asr_correction_punctuation_renders_authoritative_path(monkeypatch):
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("r43-live-asr-correction", "face-model")
    state.messages.extend([dict(item) for item in _R31_CORRECTION_BASELINE])

    def unexpected_model_call(*_args, **_kwargs):
        raise AssertionError("the unambiguous ASR punctuation variant must not call a model")

    monkeypatch.setattr(face, "_post_streaming", unexpected_model_call)
    monkeypatch.setattr(face, "_post_blocking", unexpected_model_call)
    result = face.chat(
        "No, I mean end-to-end voice latency. Correct, the recommendation.",
        session_id=state.session_id,
        model="face-model",
    )

    _assert_gateway_authoritative_latency_correction(result, None)


def test_r43_hyphenated_live_depth_goal_authorizes_first_token_revision(monkeypatch):
    goal = (
        "Explain why a 35B-depth lobe might outperform a 4B-face lobe when "
        "diagnosing an intermittent distributed inference failure."
    )
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("r43-hyphenated-depth-goal", "face-model")
    state.depth_deliveries["da-r43-hyphenated"] = {
        "goal": goal,
        "result": _DEPTH_EVIDENCE_FREE_COMPARISON_TEMPLATE,
    }
    state.last_depth_delivery = {"job_id": "da-r43-hyphenated"}
    state.messages.append(
        {
            "role": "assistant",
            "content": (
                "Controlled evidence must compare the 35B and 4B candidates for the "
                "distributed-inference diagnosis before changing the recommendation."
            ),
        }
    )

    def unexpected_model_call(*_args, **_kwargs):
        raise AssertionError("the verified hyphenated lobe mapping must not call a model")

    monkeypatch.setattr(face, "_post_streaming", unexpected_model_call)
    monkeypatch.setattr(face, "_post_blocking", unexpected_model_call)
    result = face.chat(
        "Assume first token latency must stay under 4 seconds. Revise the recommendation.",
        session_id=state.session_id,
        model="face-model",
    )

    receipt = result["output_guard"]["authoritative_first_token_revision"]
    assert result["api_calls"] == 0
    assert receipt["schema"] == "Ms4AuthoritativeFirstTokenRevision.v1"
    assert receipt["context_source"] == (
        "verified_depth_goal_and_immediate_parameter_mapped_answer"
    )


def test_r43_verified_comparison_evidence_continues_first_token_authority(monkeypatch):
    goal = (
        "Explain why a 35B-depth lobe might outperform a 4B-face lobe when "
        "diagnosing an intermittent distributed inference failure."
    )
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("r43-comparison-evidence-continuation", "face-model")
    state.depth_deliveries["da-r43-evidence-continuation"] = {
        "goal": goal,
        "result": _DEPTH_EVIDENCE_FREE_COMPARISON_TEMPLATE,
    }
    state.last_depth_delivery = {"job_id": "da-r43-evidence-continuation"}
    state.messages.append(
        {
            "role": "assistant",
            "content": (
                "The current recommendation remains provisional because the verified "
                "comparison did not establish an empirical winner for this diagnostic "
                "incident. Controlled benchmark measurements of accuracy, latency, and "
                "resource cost must decide which candidate route wins."
            ),
        }
    )

    def unexpected_model_call(*_args, **_kwargs):
        raise AssertionError("same-topic verified evidence continuation must not call a model")

    monkeypatch.setattr(face, "_post_streaming", unexpected_model_call)
    monkeypatch.setattr(face, "_post_blocking", unexpected_model_call)
    result = face.chat(
        "Assume first-token latency must stay under four seconds. Revise the recommendation.",
        session_id=state.session_id,
        model="face-model",
    )

    receipt = result["output_guard"]["authoritative_first_token_revision"]
    assert result["api_calls"] == 0
    assert receipt["context_source"] == (
        "verified_depth_goal_and_immediate_comparison_evidence_answer"
    )


@pytest.mark.parametrize(
    "correction",
    (
        "Alice said: No, I mean end-to-end voice latency. Correct, the recommendation.",
        "If I meant end-to-end voice latency, correct, the recommendation.",
        "No, I mean end-to-end voice latency. Correct, the database recommendation.",
    ),
)
def test_r43_asr_punctuation_does_not_broaden_authority(monkeypatch, correction):
    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session(f"r43-negative-{uuid.uuid4()}", "face-model")
    state.messages.extend([dict(item) for item in _R31_CORRECTION_BASELINE])
    model_candidate = _r40_safe_compute_tradeoff_candidate()

    def post_streaming(_payload, callback, *, cancel_event=None):
        del cancel_event
        callback(model_candidate)
        return model_candidate, _completed_stream_stats()

    monkeypatch.setattr(face, "_post_streaming", post_streaming)
    result = face.chat(
        correction,
        session_id=state.session_id,
        model="face-model",
        stream_callback=lambda _chunk: None,
    )

    assert "authoritative_latency_correction" not in result["output_guard"]


@pytest.mark.parametrize(
    "correction_text",
    [
        "No, I mean end-to-end voice latency, correct the recommendation.",
        "No, I mean end-to-end voice latency. Correct. The Recommendation",
    ],
)
def test_r56_live_asr_correction_and_final_synthesis_use_latest_episode(
    monkeypatch,
    correction_text,
):
    """The exact live ASR wording must resolve without a Face model retry.

    The older episode mirrors a reused validation session.  Only the newest
    closed baseline/correction pair owns the deadline and authorizes the final
    architecture, so stale matching words cannot supply gateway authority.
    """

    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    state = face.get_or_create_session("r55-live-comma-full-episode", "face-model")
    state.messages.extend(
        [
            {
                "role": "user",
                "content": (
                    "Assume first-token latency must stay under four seconds. "
                    "Revise the recommendation."
                ),
            },
            {
                "role": "assistant",
                "content": "An older validation pass revised the foreground route.",
            },
            {
                "role": "user",
                "content": (
                    "No, I mean end-to-end voice latency. Correct the recommendation."
                ),
            },
            {
                "role": "assistant",
                "content": "An older validation pass corrected the voice metric.",
            },
            {"role": "user", "content": "Now compare the model tradeoffs."},
            {
                "role": "assistant",
                "content": (
                    "The verified Depth result keeps model quality, latency, and resource "
                    "claims separate until controlled evidence resolves them."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Assume first token latency must stay under 4 seconds. "
                    "Revise the recommendation."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "Revised: latency now governs delivery. Face owns the live opening "
                    "while Depth verifies the diagnosis asynchronously."
                ),
            },
        ]
    )

    def unexpected_model_call(*_args, **_kwargs):
        raise AssertionError("the resolved Turn 5/6 episode must not call a model")

    monkeypatch.setattr(face, "_post_streaming", unexpected_model_call)
    monkeypatch.setattr(face, "_post_blocking", unexpected_model_call)

    correction = face.chat(
        correction_text,
        session_id=state.session_id,
        model="face-model",
        substantive_word_range=(150, 380),
    )

    correction_receipt = correction["output_guard"][
        "authoritative_latency_correction"
    ]
    assert correction["completed"] is True
    assert correction["api_calls"] == 0
    assert correction["output_guard"]["reason"] == "authoritative_latency_correction"
    assert correction_receipt["model_output_used"] is False
    assert correction_receipt["deadline"] == "under 4 seconds"
    assert correction_receipt["deadline_milliseconds"] == 4000
    assert correction["output_guard"]["substantive_followup"][
        "corrective_regeneration_attempted"
    ] is False

    synthesis = face.chat(
        "Give me the final architecture in plain English and tell me what constraint I changed.",
        session_id=state.session_id,
        model="face-model",
        substantive_word_range=(150, 380),
    )

    synthesis_receipt = synthesis["output_guard"][
        "authoritative_latency_synthesis"
    ]
    semantic = synthesis["output_guard"]["substantive_followup"]
    assert synthesis["completed"] is True
    assert synthesis["api_calls"] == 0
    assert synthesis["output_guard"]["reason"] == "authoritative_latency_synthesis"
    assert synthesis_receipt["model_output_used"] is False
    assert synthesis_receipt["deadline"] == "under 4 seconds"
    assert synthesis_receipt["deadline_milliseconds"] == 4000
    assert synthesis["text"].startswith("1. Direct answer\n")
    assert "2. Why and mechanism\n" in synthesis["text"]
    assert "3. Practical consequence\n" in synthesis["text"]
    assert "Face produces a bounded spoken opening" in synthesis["text"]
    assert "Depth begins the asynchronous evidence pass" in synthesis["text"]
    assert (
        "The constraint changed from first-token latency to end-to-end voice latency."
        in synthesis["text"]
    )
    assert semantic["authoritative_latency_resolution"] is True
    assert semantic["first_candidate_failures"] == []
    assert semantic["corrective_candidate_failures"] == []
    assert semantic["corrective_regeneration_attempted"] is False
    assert semantic["fail_closed"] is False
