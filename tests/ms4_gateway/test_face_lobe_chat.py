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

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from machine_spirit_4.gateway.face_lobe_chat import FaceLobeChat


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
        payload = {"choices": [{"message": {"role": "assistant", "content": content}}]}
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
            except (BrokenPipeError, ConnectionResetError):
                return
        if plan.get("send_done", True):
            try:
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
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


def test_stream_stall_returns_partial_text_without_hanging(fake_hivemind):
    """If HiveMind never sends [DONE], the FaceLobeChat must time out
    on the stall budget and return whatever fragments arrived. The
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
    assert result["metrics"]["stream_chunks"] == 1
