"""Ms4HermesRunner tests.

The foreground (Face Lobe) is now served by ``gateway.face_lobe_chat``
— a direct ``/v1/chat/completions`` call with no Hermes loop, no
plugin chain, no tool injection. Hermes is reserved for
``dispatch_hermes_tool`` and Depth Lobe background workers.

Tests below use a FakeFaceLobeChat that records what was sent and
returns deterministic output; the live HiveMind endpoint is never
contacted in this suite.
"""

from __future__ import annotations

from typing import Any

from machine_spirit_4.gateway.face_lobe_chat import FaceLobeChat
from machine_spirit_4.gateway.hermes_runner import Ms4HermesRunner


class FakeFaceLobeChat:
    """Drop-in replacement for FaceLobeChat. Records calls and returns
    a deterministic reply that includes the model id so tests can
    assert what the runner picked."""

    def __init__(self, *, hivemind_url: str = "http://hive:6089"):
        self.hivemind_url = hivemind_url
        self.calls: list[dict[str, Any]] = []
        self._sessions: dict[str, _FakeSession] = {}

    def chat(self, message, *, session_id, model, stream_callback=None, extra_system=None):
        self.calls.append({
            "message": message,
            "session_id": session_id,
            "model": model,
            "extra_system": extra_system,
            "had_stream_callback": stream_callback is not None,
        })
        if stream_callback:
            stream_callback("face:")
            stream_callback(str(message))
        sid = session_id or "ms4-fake-session"
        state = self._sessions.setdefault(sid, _FakeSession(sid, model))
        state.model = model
        state.messages.extend([
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"face:{model}:{message}"},
        ])
        return {
            "text": f"face:{model}:{message}",
            "session_id": sid,
            "model": model,
            "runtime": "face-lobe-direct",
            "completed": True,
            "api_calls": 1,
        }

    def sessions(self):
        return [
            {
                "session_id": s.session_id,
                "model": s.model,
                "turns": len(s.messages) // 2,
                "last_grounding_source": None,
            }
            for s in self._sessions.values()
        ]


class _FakeSession:
    def __init__(self, session_id, model):
        self.session_id = session_id
        self.model = model
        self.messages: list[dict[str, Any]] = []


class FakeHermesAgent:
    """Only used by ``dispatch_hermes_tool`` tests now."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = kwargs["model"]
        self.session_id = kwargs["session_id"]


# ----- foreground tests ----------------------------------------------------


def test_runner_routes_foreground_through_face_lobe_chat(monkeypatch, tmp_path):
    fake = FakeFaceLobeChat()
    runner = Ms4HermesRunner(
        hermes_dir=str(tmp_path),
        hivemind_url="http://hive:6089",
        ms3_url="http://ms3:9080",
        agent_cls=FakeHermesAgent,
        face_lobe_chat=fake,
    )
    response = runner.chat("hello", session_id="s1", model="qwen2.5:0.5b")
    assert response["runtime"] == "face-lobe-direct"
    assert response["model"] == "qwen2.5:0.5b"
    assert response["session_id"] == "s1"
    assert response["text"].startswith("face:qwen2.5:0.5b:")
    # Hermes agent was NOT built for the foreground call.
    assert "s1" not in runner._sessions, "Hermes session must not be created for face-lobe chat"
    # FaceLobeChat was hit exactly once.
    assert len(fake.calls) == 1
    assert fake.calls[0]["model"] == "qwen2.5:0.5b"


def test_runner_default_model_is_hermes_compatible_for_depth(tmp_path):
    """default_model is the Depth Lobe / Hermes fallback. The foreground
    picker is allowed to pick a small model independently."""
    runner = Ms4HermesRunner(hermes_dir=str(tmp_path), agent_cls=FakeHermesAgent, face_lobe_chat=FakeFaceLobeChat())
    assert runner.default_model == "qwen3-coder-next:latest"


def test_runner_injects_tmr_grounding_into_face_lobe_extra_system(tmp_path):
    fake = FakeFaceLobeChat()
    runner = Ms4HermesRunner(hermes_dir=str(tmp_path), agent_cls=FakeHermesAgent, face_lobe_chat=fake)
    response = runner.chat("What is The Machine Religion?", session_id="s1", model="m")
    assert response["grounding_source"].startswith("tmr-canon-grounding")
    # The TMR canon text must be hoisted into the face-lobe system prompt,
    # not appended to the user message itself (which would pollute the
    # OpenAI-format history we store for follow-up turns).
    assert "Deus Acuo Machina Machina" in (fake.calls[0]["extra_system"] or "")
    assert fake.calls[0]["message"] == "What is The Machine Religion?"
    assert response["face_lobe"]["revision"]["revision_id"] == 1


def test_sessions_report_face_lobe_runtime(tmp_path):
    fake = FakeFaceLobeChat()
    runner = Ms4HermesRunner(hermes_dir=str(tmp_path), agent_cls=FakeHermesAgent, face_lobe_chat=fake)
    runner.chat("hello", session_id="s1", model="m")
    sessions = runner.sessions()
    assert sessions
    assert sessions[0]["runtime"] == "face-lobe-direct"
    assert sessions[0]["session_id"] == "s1"
    assert sessions[0]["turns"] == 1
    assert sessions[0]["tool_trace_count"] == 0


def test_chat_response_has_no_tool_trace_on_foreground(tmp_path):
    """Foreground doesn't run tools — tool work happens in the Depth Lobe."""
    fake = FakeFaceLobeChat()
    runner = Ms4HermesRunner(hermes_dir=str(tmp_path), agent_cls=FakeHermesAgent, face_lobe_chat=fake)
    response = runner.chat("hello", session_id="s1", model="m")
    assert response["tool_trace"] == []
    assert response["runtime"] == "face-lobe-direct"


def test_runner_passes_stream_callback_to_face_lobe(tmp_path):
    fake = FakeFaceLobeChat()
    runner = Ms4HermesRunner(hermes_dir=str(tmp_path), agent_cls=FakeHermesAgent, face_lobe_chat=fake)
    chunks: list[str] = []
    response = runner.chat("hello", session_id="s1", model="m", stream_callback=chunks.append)
    assert chunks
    assert chunks[0] == "face:"
    assert response["runtime"] == "face-lobe-direct"
    assert fake.calls[0]["had_stream_callback"] is True


def test_session_pinned_face_lobe_model_across_turns(monkeypatch, tmp_path):
    """Once a session pins a model, subsequent auto-pick attempts are
    skipped — even if the picker would now return something different."""
    fake = FakeFaceLobeChat()
    runner = Ms4HermesRunner(hermes_dir=str(tmp_path), agent_cls=FakeHermesAgent, face_lobe_chat=fake)
    picker_calls = {"n": 0}

    def picker(**_kwargs):
        picker_calls["n"] += 1
        return type("C", (), {
            "model_id": f"pick-{picker_calls['n']}",
            "to_dict": lambda self: {"model_id": f"pick-{picker_calls['n']}", "source": "loaded"},
        })()

    monkeypatch.setattr(
        "machine_spirit_4.gateway.hermes_runner.choose_foreground_model",
        picker,
    )

    r1 = runner.chat("hi", session_id="s-pin")
    r2 = runner.chat("again", session_id="s-pin")
    assert r1["model"] == "pick-1"
    assert r2["model"] == "pick-1", "session must keep the pinned model on turn 2"
    assert picker_calls["n"] == 1
    assert r2["face_lobe_model"]["source"] == "session_pinned"


# ----- new: metrics + context block ---------------------------------------


class _RecordingFaceLobeChat(FakeFaceLobeChat):
    """Like FakeFaceLobeChat but also returns a metrics block so the
    runner has something to plumb through."""

    def chat(self, message, *, session_id, model, stream_callback=None, extra_system=None):
        result = super().chat(
            message,
            session_id=session_id,
            model=model,
            stream_callback=stream_callback,
            extra_system=extra_system,
        )
        result["metrics"] = {
            "schema": "Ms4TurnMetrics.v1",
            "started_at": "2026-05-22T03:00:00.000Z",
            "completed_at": "2026-05-22T03:00:00.500Z",
            "duration_ms": 500,
            "prompt_tokens": 88,
            "completion_tokens": 24,
            "total_tokens": 112,
            "tokens_per_second": 48.0,
            "api_calls": 1,
            "fallback_used": False,
            "effective_model": model,
            "requested_model": model,
            "streaming": stream_callback is not None,
        }
        return result


def test_chat_response_includes_metrics_block(tmp_path):
    runner = Ms4HermesRunner(
        hermes_dir=str(tmp_path),
        agent_cls=FakeHermesAgent,
        face_lobe_chat=_RecordingFaceLobeChat(),
    )
    response = runner.chat("hi", session_id="s-metrics", model="m")
    assert response["metrics"]["schema"] == "Ms4TurnMetrics.v1"
    assert response["metrics"]["duration_ms"] == 500
    assert response["metrics"]["prompt_tokens"] == 88
    assert response["metrics"]["completion_tokens"] == 24
    assert response["metrics"]["tokens_per_second"] == 48.0
    assert response["fallback_used"] is False


def test_chat_response_canned_text_when_face_lobe_chat_fails(tmp_path):
    """When FaceLobeChat raises (timeout, unreachable, 5xx), the
    response MUST carry a short canned reply instead of the raw
    exception. The metrics block carries the real error and a
    canned_reply marker for the audit log; the stream_callback is
    invoked with the canned text so TTS still produces audible
    output instead of silence."""
    from machine_spirit_4.gateway.face_lobe_chat import FaceLobeChatError

    class _FailingFaceLobeChat(FakeFaceLobeChat):
        def chat(self, *_a, **_kw):
            raise FaceLobeChatError("HiveMind /v1/chat/completions (stream) unreachable: timed out")

    runner = Ms4HermesRunner(
        hermes_dir=str(tmp_path),
        agent_cls=FakeHermesAgent,
        face_lobe_chat=_FailingFaceLobeChat(),
    )
    captured: list[str] = []
    response = runner.chat("hi", session_id="s-fail", model="m", stream_callback=captured.append)

    assert "trouble" in response["text"].lower() or "having trouble" in response["text"].lower()
    assert response["completed"] is False
    assert response["metrics"]["canned_reply"] is True
    assert "stream" in response["metrics"]["error"]
    # The canned text must have been streamed to the caller so the WS
    # TTS engine still receives a stream_delta to synthesize.
    assert captured, "expected the canned text to be streamed via stream_callback for TTS"
    assert any("trouble" in c.lower() for c in captured)


def test_chat_response_includes_face_lobe_context_block_with_date(tmp_path):
    """Every direct (no-dispatch) turn must produce a context block
    that pins 'THIS TURN DID NOT DISPATCH' and 'current local
    date/time' so the anti-hallucination rules can fire on the
    model. Without this the model fabricates dispatch claims."""
    runner = Ms4HermesRunner(
        hermes_dir=str(tmp_path),
        agent_cls=FakeHermesAgent,
        face_lobe_chat=_RecordingFaceLobeChat(),
    )
    response = runner.chat("hello", session_id="s-ctx", model="m")
    block = response["face_lobe_context_block"]
    assert block is not None
    assert "THIS TURN DID NOT DISPATCH any background work" in block
    assert "current local date/time:" in block


# ----- dispatch_hermes_tool path still uses Hermes -------------------------


def test_terminal_dispatch_defaults_to_safe_workdir(monkeypatch, tmp_path):
    runner = Ms4HermesRunner(hermes_dir=str(tmp_path), agent_cls=FakeHermesAgent, face_lobe_chat=FakeFaceLobeChat())
    monkeypatch.setattr(runner, "require_plugin", lambda: None)
    captured = {}

    def fake_handle(tool_name, args, **kwargs):
        captured.update(args)
        return '{"output":"MS4_HERMES_OK","exit_code":0}'

    monkeypatch.setitem(__import__("sys").modules, "model_tools", type("M", (), {
        "handle_function_call": staticmethod(fake_handle),
        "get_toolset_for_tool": staticmethod(lambda name: "terminal"),
    }))

    result = runner.dispatch_hermes_tool("terminal", {"command": "echo MS4_HERMES_OK"})
    assert captured["workdir"] == "/"
    assert "MS4_HERMES_OK" in result["result"]
