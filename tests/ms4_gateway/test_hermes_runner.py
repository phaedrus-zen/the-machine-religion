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

import json
import sys
import threading
import types
from typing import Any

import pytest

from machine_spirit_4.gateway import hermes_runner as hr
from machine_spirit_4.gateway.hermes_runner import Ms4HermesRunner, Ms4TurnCancelled


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

    def chat_authoritative(
        self,
        message,
        *,
        authoritative_text,
        session_id,
        model,
        stream_callback=None,
        extra_system=None,
        **_kwargs,
    ):
        self.calls.append({
            "message": message,
            "session_id": session_id,
            "model": model,
            "extra_system": extra_system,
            "authoritative_text": authoritative_text,
            "had_stream_callback": stream_callback is not None,
        })
        sid = session_id or "ms4-fake-session"
        return {
            "text": authoritative_text,
            "session_id": sid,
            "model": model,
            "runtime": "face-lobe-direct",
            "completed": True,
            "api_calls": 0,
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


def test_cancelled_foreground_turn_commits_nothing_and_dispatches_no_job(tmp_path):
    """Finding 3 (R4): a turn cancelled before it runs must NOT route, bump the
    revision, dispatch a Depth job, or commit any Face Lobe history. The
    cancellation fence raises Ms4TurnCancelled before any conversation side
    effect, and the fence propagates (it is not swallowed into a job stub)."""
    fake = FakeFaceLobeChat()
    dispatched: list[Any] = []
    runner = Ms4HermesRunner(
        hermes_dir=str(tmp_path),
        hivemind_url="http://hive:6089",
        ms3_url="http://ms3:9080",
        agent_cls=FakeHermesAgent,
        face_lobe_chat=fake,
    )
    # Any dispatch attempt would be a bug for a cancelled turn — record it.
    import machine_spirit_4.gateway.hermes_runner as _hr
    orig_runner = _hr.default_runner

    class _RecordingRunner:
        def submit(self, envelope):
            dispatched.append(envelope)
            return {"job_id": "should-not-happen"}

    _hr.default_runner = lambda: _RecordingRunner()
    try:
        cancel = threading.Event()
        cancel.set()
        with pytest.raises(Ms4TurnCancelled):
            runner.chat("please run a deep research task", session_id="s1", model="m", cancel_event=cancel)
    finally:
        _hr.default_runner = orig_runner
    assert fake.calls == [], "the Face Lobe model must not be called for a cancelled turn"
    assert "s1" not in fake._sessions, "no session/history may be created for a cancelled turn"
    assert "s1" not in runner._sessions, "no Hermes session for a cancelled turn"
    assert dispatched == [], "a cancelled turn must dispatch no Depth job"


def test_runner_default_model_is_hermes_compatible_for_depth(tmp_path):
    """default_model is the Depth Lobe / Hermes fallback. The foreground
    picker is allowed to pick a small model independently."""
    runner = Ms4HermesRunner(hermes_dir=str(tmp_path), agent_cls=FakeHermesAgent, face_lobe_chat=FakeFaceLobeChat())
    assert runner.default_model == "nemotron-3-nano:30b"


# ----- sanctioned background-agent entry point + iteration config ----------


def test_new_background_agent_sanctioned_path(tmp_path, monkeypatch):
    """Double Agent workers build their agent via new_background_agent —
    the sanctioned entry point that replaced the
    runner._new_agent.__self__._agent_class() reach-through."""
    monkeypatch.setenv("MS4_DA_MAX_ITERATIONS", "7")
    runner = Ms4HermesRunner(
        hermes_dir=str(tmp_path), agent_cls=FakeHermesAgent, face_lobe_chat=FakeFaceLobeChat()
    )
    plugin_checked = {"n": 0}
    monkeypatch.setattr(runner, "ensure_hermes_path", lambda: None)
    monkeypatch.setattr(runner, "require_plugin", lambda: plugin_checked.__setitem__("n", plugin_checked["n"] + 1))

    def ts(*_a):
        pass

    def tc(*_a):
        pass

    agent = runner.new_background_agent(
        session_id="da-worker-1", model="phi4-mini:latest",
        tool_start_callback=ts, tool_complete_callback=tc,
    )
    assert isinstance(agent, FakeHermesAgent)
    assert agent.kwargs["session_id"] == "da-worker-1"
    assert agent.kwargs["model"] == "phi4-mini:latest"
    assert agent.kwargs["platform"] == "ms4"
    assert agent.kwargs["skip_memory"] is True
    assert agent.kwargs["skip_context_files"] is True
    assert agent.kwargs["tool_start_callback"] is ts
    assert agent.kwargs["tool_complete_callback"] is tc
    # Env-driven iteration cap is honored.
    assert agent.kwargs["max_iterations"] == 7
    # Depth gets an explicit completion budget and bounded reasoning policy;
    # foreground agents remain on their separate latency-sensitive contract.
    assert agent.kwargs["max_tokens"] == 4096
    assert agent.kwargs["reasoning_config"] == {"enabled": True, "effort": "low"}
    # The sanctioned path enforces the ethics plugin before constructing.
    assert plugin_checked["n"] == 1


def test_explicit_empty_background_catalog_disables_kanban_override(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "hostile-inherited-task")
    runner = Ms4HermesRunner(
        hermes_dir=str(tmp_path),
        agent_cls=FakeHermesAgent,
        face_lobe_chat=FakeFaceLobeChat(),
    )
    monkeypatch.setattr(runner, "ensure_hermes_path", lambda: None)
    monkeypatch.setattr(runner, "require_plugin", lambda: None)

    agent = runner.new_background_agent(
        session_id="da-no-tools",
        model="llama3.1:8b",
        tool_start_callback=lambda *_a: None,
        tool_complete_callback=lambda *_a: None,
        enabled_toolsets=[],
    )

    assert agent.kwargs["enabled_toolsets"] == []
    assert agent.kwargs["disabled_toolsets"] == ["kanban"]


def test_list_hermes_tools_keeps_explicit_empty_catalog_authoritative(
    tmp_path, monkeypatch
):
    calls = []
    fake_model_tools = types.ModuleType("model_tools")

    def get_tool_definitions(**kwargs):
        calls.append(kwargs)
        return []

    fake_model_tools.get_tool_definitions = get_tool_definitions
    fake_model_tools.get_toolset_for_tool = lambda _name: None
    monkeypatch.setitem(sys.modules, "model_tools", fake_model_tools)
    runner = Ms4HermesRunner(
        hermes_dir=str(tmp_path),
        agent_cls=FakeHermesAgent,
        face_lobe_chat=FakeFaceLobeChat(),
    )
    monkeypatch.setattr(runner, "ensure_hermes_path", lambda: None)

    assert runner.list_hermes_tools(enabled_toolsets=[]) == []
    assert calls == [
        {
            "enabled_toolsets": [],
            "disabled_toolsets": ["kanban"],
            "quiet_mode": True,
        }
    ]


def test_background_agent_propagates_hivemind_trace_header(tmp_path, monkeypatch):
    runner = Ms4HermesRunner(
        hermes_dir=str(tmp_path), agent_cls=FakeHermesAgent, face_lobe_chat=FakeFaceLobeChat()
    )
    monkeypatch.setattr(runner, "ensure_hermes_path", lambda: None)
    monkeypatch.setattr(runner, "require_plugin", lambda: None)
    trace_id = "77a4fe89-2f09-47b0-8f52-c35186fe82dc"

    agent = runner.new_background_agent(
        session_id=f"da-{trace_id}", model="llama3.1:8b",
        tool_start_callback=lambda *_a: None, tool_complete_callback=lambda *_a: None,
    )

    assert agent.kwargs["request_overrides"] == {
        "extra_headers": {"X-HiveMind-Client-Trace": trace_id},
    }


def test_worker_namespaced_background_agent_propagates_hivemind_trace_header(tmp_path, monkeypatch):
    runner = Ms4HermesRunner(
        hermes_dir=str(tmp_path), agent_cls=FakeHermesAgent, face_lobe_chat=FakeFaceLobeChat()
    )
    monkeypatch.setattr(runner, "ensure_hermes_path", lambda: None)
    monkeypatch.setattr(runner, "require_plugin", lambda: None)
    trace_id = "77a4fe89-2f09-47b0-8f52-c35186fe82dc"

    agent = runner.new_background_agent(
        session_id=f"da-worker-da-{trace_id}", model="llama3.1:8b",
        tool_start_callback=lambda *_a: None, tool_complete_callback=lambda *_a: None,
    )

    assert agent.kwargs["request_overrides"] == {
        "extra_headers": {"X-HiveMind-Client-Trace": trace_id},
    }
    assert agent._api_max_retries == 1


def test_background_agent_api_attempt_override_is_explicit(tmp_path, monkeypatch):
    monkeypatch.setenv("MS4_DA_API_MAX_ATTEMPTS", "2")
    runner = Ms4HermesRunner(
        hermes_dir=str(tmp_path), agent_cls=FakeHermesAgent, face_lobe_chat=FakeFaceLobeChat()
    )
    monkeypatch.setattr(runner, "ensure_hermes_path", lambda: None)
    monkeypatch.setattr(runner, "require_plugin", lambda: None)
    trace_id = "77a4fe89-2f09-47b0-8f52-c35186fe82dc"

    agent = runner.new_background_agent(
        session_id=f"da-worker-da-{trace_id}", model="llama3.1:8b",
        tool_start_callback=lambda *_a: None, tool_complete_callback=lambda *_a: None,
    )

    assert agent._api_max_retries == 2


def test_new_background_agent_falls_back_to_default_model(tmp_path, monkeypatch):
    runner = Ms4HermesRunner(
        hermes_dir=str(tmp_path), agent_cls=FakeHermesAgent, face_lobe_chat=FakeFaceLobeChat()
    )
    monkeypatch.setattr(runner, "ensure_hermes_path", lambda: None)
    monkeypatch.setattr(runner, "require_plugin", lambda: None)
    agent = runner.new_background_agent(
        session_id="da-2", model=None,
        tool_start_callback=lambda *_a: None, tool_complete_callback=lambda *_a: None,
    )
    assert agent.kwargs["model"] == runner.default_model


def test_max_iterations_env_helpers(monkeypatch):
    from machine_spirit_4.gateway import hermes_runner as hr

    monkeypatch.delenv("MS4_DA_MAX_ITERATIONS", raising=False)
    monkeypatch.setenv("MS4_HERMES_MAX_ITERATIONS", "3")
    assert hr.foreground_max_iterations() == 3
    # Background falls back to the foreground value when its own var is unset.
    assert hr.background_max_iterations() == 3
    monkeypatch.setenv("MS4_DA_MAX_ITERATIONS", "9")
    assert hr.background_max_iterations() == 9
    # Garbage / empty fall back to the documented default (12).
    monkeypatch.setenv("MS4_HERMES_MAX_ITERATIONS", "garbage")
    monkeypatch.delenv("MS4_DA_MAX_ITERATIONS", raising=False)
    assert hr.foreground_max_iterations() == 12
    assert hr.background_max_iterations() == 12


def test_background_model_budget_helpers_are_bounded(monkeypatch):
    from machine_spirit_4.gateway import hermes_runner as hr

    monkeypatch.delenv("MS4_DA_MAX_TOKENS", raising=False)
    monkeypatch.delenv("MS4_DA_REASONING_EFFORT", raising=False)
    assert hr.background_max_tokens() == 4096
    assert hr.background_reasoning_config() == {"enabled": True, "effort": "low"}

    monkeypatch.setenv("MS4_DA_MAX_TOKENS", "8192")
    monkeypatch.setenv("MS4_DA_REASONING_EFFORT", "medium")
    assert hr.background_max_tokens() == 8192
    assert hr.background_reasoning_config() == {"enabled": True, "effort": "medium"}

    monkeypatch.setenv("MS4_DA_REASONING_EFFORT", "off")
    assert hr.background_reasoning_config() == {"enabled": False}

    for invalid in ("garbage", "2048", "16384", "32768"):
        monkeypatch.setenv("MS4_DA_MAX_TOKENS", invalid)
        assert hr.background_max_tokens() == 4096
    for unsupported in ("turbo", "xhigh", "max", "ultra"):
        monkeypatch.setenv("MS4_DA_REASONING_EFFORT", unsupported)
        assert hr.background_reasoning_config() == {"enabled": True, "effort": "low"}


def test_foreground_agent_does_not_inherit_depth_budget(tmp_path, monkeypatch):
    runner = Ms4HermesRunner(
        hermes_dir=str(tmp_path),
        agent_cls=FakeHermesAgent,
        face_lobe_chat=FakeFaceLobeChat(),
    )
    monkeypatch.setattr(runner, "ensure_hermes_path", lambda: None)

    agent = runner._new_agent("face-session", "phi4-mini:latest")

    assert "max_tokens" not in agent.kwargs
    assert "reasoning_config" not in agent.kwargs


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


def test_unexpected_foreground_picker_error_never_calls_runner_default(monkeypatch, tmp_path):
    """An exception escaping the picker is a defect, not fallback permission."""
    fake = FakeFaceLobeChat()
    runner = Ms4HermesRunner(
        hermes_dir=str(tmp_path),
        default_model="rejected-heavy:70b",
        agent_cls=FakeHermesAgent,
        face_lobe_chat=fake,
    )

    def picker_boom(**_kwargs):
        raise TypeError("unexpected picker defect")

    monkeypatch.setattr(hr, "choose_foreground_model", picker_boom)

    with pytest.raises(TypeError, match="unexpected picker defect"):
        runner.chat("hello", session_id="s-picker-defect")

    assert fake.calls == [], "a picker defect must not call the heavy runner default"


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
    # The Face Lobe context block prefers HiveMind cluster time when
    # reachable ("current cluster date/time: ..."), falling back to
    # "current local date/time: ..." otherwise. Accept either since
    # this test doesn't gate the cluster reachability.
    assert (
        "current local date/time:" in block
        or "current cluster date/time:" in block
    )


def test_dispatch_failure_uses_deterministic_guard_without_face_call(monkeypatch, tmp_path):
    fake = FakeFaceLobeChat()
    _patch_catalog(monkeypatch, _inline_catalog(EXACT_READ_CANONICAL, errors=()))
    monkeypatch.setenv("MS4_QM_INLINE", "0")
    monkeypatch.setattr(hr, "router_route", lambda m: types.SimpleNamespace(
        kind="deep",
        confidence=0.95,
        source="test",
        cleaned_message=m,
        goal=m,
        to_dict=lambda: {"kind": "deep", "source": "test"},
    ))
    monkeypatch.setattr(hr, "choose_foreground_model", lambda **_k: types.SimpleNamespace(
        model_id="face-model",
        to_dict=lambda: {"schema": "Ms4ForegroundModel.v1", "model_id": "face-model", "source": "test"},
    ))
    monkeypatch.setattr(hr, "choose_depth_model", lambda **_k: types.SimpleNamespace(
        model_id="depth-model",
        to_dict=lambda: {"model_id": "depth-model", "source": "test"},
    ))
    monkeypatch.setattr(hr, "build_grounded_user_message", lambda msg, _url: (msg, None))

    class _FailingRunner:
        def submit(self, _envelope):
            raise RuntimeError("depth namespace cannot reach HiveMind MCP")

    monkeypatch.setattr(hr, "default_runner", lambda: _FailingRunner())

    runner = Ms4HermesRunner(
        hermes_dir=str(tmp_path),
        agent_cls=FakeHermesAgent,
        face_lobe_chat=fake,
    )
    monkeypatch.setattr(runner, "_try_cluster_time", lambda: None)
    chunks: list[str] = []

    monkeypatch.setattr(
        runner,
        "_try_quartermaster_inline",
        lambda message: (None, _exact_read_outcome(message)),
    )

    response = runner.chat(EXACT_READ_GOAL, session_id="s-dispatch-fail", stream_callback=chunks.append)

    assert fake.calls == []
    assert response["completed"] is False
    assert response["api_calls"] == 0
    assert "dispatch failed" in response["text"]
    assert "No background job is running for this turn." in response["text"]
    assert response["dispatched_job"]["error"].startswith("router dispatch failed:")
    assert response["fabrication_guard"]["applied"] is True
    assert response["fabrication_guard"]["reason"] == "dispatch_failed"
    assert response["metrics"]["fabrication_guard"] is True
    assert "depth_lobe_dispatch_failed" in response["grounding_source"]
    block = response["face_lobe_context_block"]
    assert "THIS TURN DID NOT DISPATCH any background work" in block
    assert "THIS TURN TRIED to dispatch Depth Lobe work, but dispatch failed" in block
    assert chunks == [response["text"]]


def test_depth_dispatch_envelope_includes_prior_face_context(monkeypatch, tmp_path):
    fake = FakeFaceLobeChat()
    _patch_catalog(monkeypatch, _inline_catalog(EXACT_READ_CANONICAL, errors=()))
    runner = Ms4HermesRunner(
        hermes_dir=str(tmp_path),
        agent_cls=FakeHermesAgent,
        face_lobe_chat=fake,
    )
    monkeypatch.setattr(runner, "_try_cluster_time", lambda: None)
    monkeypatch.setattr(hr, "build_grounded_user_message", lambda msg, _url: (msg, None))

    runner.chat("what is running?", session_id="s-prior", model="face-model")
    assert len(fake._sessions["s-prior"].messages) == 2

    monkeypatch.setenv("MS4_QM_INLINE", "0")
    monkeypatch.setattr(hr, "router_route", lambda m: types.SimpleNamespace(
        kind="deep",
        confidence=0.95,
        source="test",
        cleaned_message=m,
        goal=m,
        to_dict=lambda: {"kind": "deep", "source": "test"},
    ))
    monkeypatch.setattr(hr, "choose_depth_model", lambda **_k: types.SimpleNamespace(
        model_id="depth-model",
        to_dict=lambda: {"model_id": "depth-model", "source": "test"},
    ))
    monkeypatch.setattr(
        hr,
        "face_lobe_turn_start",
        lambda **kwargs: {
            "revision": {
                "revision_id": 7,
                "turn_id": kwargs.get("turn_id"),
            },
        },
    )
    captured = {}

    class _CapturingRunner:
        def submit(self, envelope):
            captured["envelope"] = envelope
            return {"job_id": envelope.job_id, "state": "queued"}

    monkeypatch.setattr(hr, "default_runner", lambda: _CapturingRunner())

    monkeypatch.setattr(
        runner,
        "_try_quartermaster_inline",
        lambda message: (None, _exact_read_outcome(message)),
    )

    turn_id = "ms4-turn-0123456789abcdef"
    response = runner.chat(
        EXACT_READ_GOAL,
        session_id="s-prior",
        model="face-model",
        turn_id=turn_id,
    )

    envelope = captured["envelope"]
    assert envelope.prior_context == [
        {"role": "user", "content": "what is running?"},
        {"role": "assistant", "content": "face:face-model:what is running?"},
    ]
    assert EXACT_READ_GOAL not in str(envelope.prior_context)
    assert response["depth_lobe_prior_context_count"] == 2
    assert envelope.conversation_revision_id == 7
    assert envelope.turn_id == turn_id
    assert response["turn_id"] == turn_id
    assert response["revision_id"] == 7
    assert response["face_lobe"]["revision"]["turn_id"] == turn_id
    assert response["dispatched_job"]["job_id"] == envelope.job_id


EXACT_READ_CANONICAL = "hivemind.app.get@v1"
EXACT_READ_GOAL = f"fetch app {EXACT_READ_CANONICAL} id=demo"
EXACT_GATED_CANONICAL = "hivemind.services.enable@v1"
INLINE_CANONICAL = "hivemind.time.now@v1"


class _SeamCounters:
    def __init__(self):
        self.inline = 0
        self.submit = 0
        self.direct = 0
        self.envelopes: list[Any] = []


def _exact_read_outcome(query: str, canonical: str = EXACT_READ_CANONICAL) -> dict[str, Any]:
    return {
        "schema": "Ms4ToolRouteDecision.v1",
        "verdict": "depth",
        "query": query,
        "reason": f"top tool {canonical} requires args ['id']",
        "resolution": {
            "schema": "Ms4ToolResolution.v1",
            "query": query,
            "tier": "deterministic",
            "confidence": 1.0,
            "toolboxes": ["apps"],
            "tools": [{
                "name": canonical,
                "toolbox": "apps",
                "cluster": "applications",
                "score": 1.0,
                "kind": "hivemind_native",
                "inline_eligible": True,
                "source": "hivemind",
                "gated_by": [],
            }],
            "catalog_version": "test-catalog",
            "fallback_reason": None,
        },
        "inline_tool": None,
        "inline_toolbox": None,
        "ethics": None,
    }


def _inline_outcome(query: str, canonical: str = INLINE_CANONICAL) -> dict[str, Any]:
    return {
        "schema": "Ms4ToolRouteDecision.v1",
        "verdict": "inline",
        "query": query,
        "reason": f"inline {canonical}",
        "resolution": {
            "schema": "Ms4ToolResolution.v1",
            "query": query,
            "tier": "deterministic",
            "confidence": 1.0,
            "toolboxes": ["time"],
            "tools": [{
                "name": canonical,
                "toolbox": "time",
                "cluster": "time",
                "score": 1.0,
                "kind": "hivemind_native",
                "inline_eligible": True,
                "source": "hivemind",
                "gated_by": [],
            }],
            "catalog_version": "test-catalog",
            "fallback_reason": None,
        },
        "inline_tool": canonical,
        "inline_toolbox": "time",
        "ethics": {"allowed": True, "reason": "test"},
    }


def _inline_catalog(*names: str, errors: tuple = ()):
    from machine_spirit_4.gateway.quartermaster.catalog import (
        KIND_HIVEMIND_NATIVE,
        SOURCE_HIVEMIND,
        Catalog,
        ToolEntry,
    )

    tools = []
    for name in names:
        toolbox = name.split(".")[1] if "." in name else "time"
        tools.append(ToolEntry(
            schema="Ms4QuartermasterTool.v1",
            name=name,
            toolbox=toolbox,
            cluster=toolbox,
            description="test",
            source=SOURCE_HIVEMIND,
            kind=KIND_HIVEMIND_NATIVE,
            input_schema={"type": "object", "properties": {}, "required": []},
        ))
    boxed: dict[str, tuple] = {}
    for entry in tools:
        boxed.setdefault(entry.toolbox, ())
        boxed[entry.toolbox] = boxed[entry.toolbox] + (entry,)
    return Catalog(
        schema="Ms4QuartermasterCatalog.v1",
        version="test",
        built_at="2026-08-17T00:00:00Z",
        hivemind_url="http://hive:6089",
        tools=tuple(tools),
        toolboxes=boxed,
        sources={"hivemind": len(tools)},
        errors=errors,
    )


def _patch_catalog(monkeypatch, catalog):
    monkeypatch.setattr(
        "machine_spirit_4.gateway.quartermaster.catalog.get_catalog",
        lambda _url, **_k: catalog,
    )


def _install_submit_counter(monkeypatch, counters: _SeamCounters):
    class _CountingRunner:
        def submit(self, envelope):
            counters.submit += 1
            counters.envelopes.append(envelope)
            return {"job_id": envelope.job_id, "state": "queued"}

    monkeypatch.setattr(hr, "default_runner", lambda: _CountingRunner())
    return counters


def _deep_route(monkeypatch):
    monkeypatch.setattr(hr, "router_route", lambda m: types.SimpleNamespace(
        kind="deep",
        confidence=0.95,
        source="test",
        cleaned_message=m,
        goal=m,
        to_dict=lambda: {"kind": "deep", "source": "test"},
    ))
    monkeypatch.setattr(hr, "choose_depth_model", lambda **_k: types.SimpleNamespace(
        model_id="depth-model",
        to_dict=lambda: {"model_id": "depth-model", "source": "test"},
    ))
    monkeypatch.setattr(hr, "choose_foreground_model", lambda **_k: types.SimpleNamespace(
        model_id="face-model",
        to_dict=lambda: {
            "schema": "Ms4ForegroundModel.v1",
            "model_id": "face-model",
            "source": "test",
        },
    ))
    monkeypatch.setattr(hr, "build_grounded_user_message", lambda msg, _url: (msg, None))


def test_depth_dispatch_hivemind_query_gets_hivemind_toolset(monkeypatch, tmp_path):
    fake = FakeFaceLobeChat()
    runner = Ms4HermesRunner(
        hermes_dir=str(tmp_path),
        agent_cls=FakeHermesAgent,
        face_lobe_chat=fake,
    )
    monkeypatch.setattr(runner, "_try_cluster_time", lambda: None)
    monkeypatch.setattr(hr, "build_grounded_user_message", lambda msg, _url: (msg, None))
    monkeypatch.setenv("MS4_QM_INLINE", "0")
    monkeypatch.delenv("MS4_QM_TRIM_DEPTH", raising=False)
    monkeypatch.setattr(hr, "router_route", lambda m: types.SimpleNamespace(
        kind="deep",
        confidence=0.95,
        source="test",
        cleaned_message=m,
        goal=m,
        to_dict=lambda: {"kind": "deep", "source": "test"},
    ))
    monkeypatch.setattr(hr, "choose_depth_model", lambda **_k: types.SimpleNamespace(
        model_id="depth-model",
        to_dict=lambda: {"model_id": "depth-model", "source": "test"},
    ))
    monkeypatch.setattr(hr, "face_lobe_turn_start", lambda **_k: {"revision": {"revision_id": 3}})
    captured = {}

    class _CapturingRunner:
        def submit(self, envelope):
            captured["envelope"] = envelope
            return {"job_id": envelope.job_id, "state": "queued"}

    monkeypatch.setattr(hr, "default_runner", lambda: _CapturingRunner())

    cases = [
        ("what gpus are available right now?", None),
        ("investigate this open-ended issue", None),
    ]
    for index, (query, expected_toolsets) in enumerate(cases):
        response = runner.chat(query, session_id=f"s-hm-{index}", model="face-model")
        assert "envelope" not in captured
        assert response["dispatched_job"] is None
        assert expected_toolsets is None


def test_depth_toolset_selection_failure_uses_explicit_safe_defaults(monkeypatch):
    from machine_spirit_4.gateway import quartermaster

    def fail_selection(_query):
        raise RuntimeError("catalog unavailable")

    monkeypatch.setattr(quartermaster, "hermes_toolsets_for_query", fail_selection)

    assert hr._curated_depth_toolsets("investigate the cluster") is None


def test_compound_analytical_depth_job_gets_explicit_empty_tool_catalog():
    prompt = (
        "Explain why a 35B Depth lobe might outperform a 4B Face lobe when "
        "diagnosing an intermittent distributed-inference failure. Give me the "
        "recommendation, mechanism, tradeoffs, and a concrete example."
    )

    assert hr._curated_depth_toolsets(prompt) == []


def test_current_analytical_depth_job_keeps_curated_tools():
    toolsets = hr._curated_depth_toolsets(
        "Inspect the current cluster logs, recommend a fix, and explain the tradeoffs."
    )

    assert toolsets is None


# ----- dispatch_hermes_tool path still uses Hermes -------------------------


def test_terminal_dispatch_defaults_to_safe_workdir(monkeypatch, tmp_path):
    counters = _SeamCounters()
    runner = Ms4HermesRunner(hermes_dir=str(tmp_path), agent_cls=FakeHermesAgent, face_lobe_chat=FakeFaceLobeChat())
    monkeypatch.setattr(runner, "require_plugin", lambda: None)

    def fake_handle(tool_name, args, **kwargs):
        counters.direct += 1
        return '{"output":"MS4_HERMES_OK","exit_code":0}'

    monkeypatch.setitem(__import__("sys").modules, "model_tools", type("M", (), {
        "handle_function_call": staticmethod(fake_handle),
        "get_toolset_for_tool": staticmethod(lambda name: "terminal"),
    }))

    result = runner.dispatch_hermes_tool("terminal", {"command": "echo MS4_HERMES_OK"})
    assert counters.inline == 0
    assert counters.submit == 0
    assert counters.direct == 0
    assert result.get("refused") is True
    assert result.get("result") is None
    assert "not an admitted" in result["error"]


def _inline_decide(query, canonical=INLINE_CANONICAL):
    class _Decision:
        verdict = "inline"
        inline_tool = types.SimpleNamespace(name=canonical, toolbox="time")
        resolution = types.SimpleNamespace(tier="deterministic")

        def to_dict(self):
            return _inline_outcome(query, canonical)

    return _Decision()


def test_unknown_or_failed_mapping_is_zero_effect(monkeypatch, tmp_path):
    counters = _SeamCounters()
    _install_submit_counter(monkeypatch, counters)
    _deep_route(monkeypatch)
    monkeypatch.setenv("MS4_QM_INLINE", "1")
    _patch_catalog(monkeypatch, _inline_catalog(INLINE_CANONICAL))

    def boom(_query, **_k):
        raise RuntimeError("selector failed")

    monkeypatch.setattr("machine_spirit_4.gateway.quartermaster.decide", boom)
    monkeypatch.setattr(
        "machine_spirit_4.gateway.quartermaster.execute_inline_tool",
        lambda *_a, **_k: counters.__dict__.update(inline=counters.inline + 1) or {"tool": INLINE_CANONICAL, "result": {}, "elapsed_ms": 1},
    )
    runner = Ms4HermesRunner(
        hermes_dir=str(tmp_path),
        agent_cls=FakeHermesAgent,
        face_lobe_chat=FakeFaceLobeChat(),
        hivemind_url="http://hive:6089",
    )
    monkeypatch.setattr(runner, "_try_cluster_time", lambda: None)
    response = runner.chat("investigate the cluster", session_id="s-map-fail", model="m")
    assert counters.inline == 0
    assert counters.submit == 0
    assert counters.direct == 0
    assert response["dispatched_job"] is None
    assert "not an admitted" in response["text"] or "did not run" in response["text"]


def test_empty_tool_selection_is_zero_effect(monkeypatch, tmp_path):
    counters = _SeamCounters()
    _install_submit_counter(monkeypatch, counters)
    _deep_route(monkeypatch)
    monkeypatch.setenv("MS4_QM_INLINE", "0")
    runner = Ms4HermesRunner(
        hermes_dir=str(tmp_path),
        agent_cls=FakeHermesAgent,
        face_lobe_chat=FakeFaceLobeChat(),
        hivemind_url="http://hive:6089",
    )
    monkeypatch.setattr(runner, "_try_cluster_time", lambda: None)
    response = runner.chat("what gpus are available right now?", session_id="s-empty", model="m")
    assert counters.inline == 0
    assert counters.submit == 0
    assert counters.direct == 0
    assert response["dispatched_job"] is None


def test_accepted_inline_read_is_one_execute_zero_depth_zero_direct(monkeypatch, tmp_path):
    counters = _SeamCounters()
    _install_submit_counter(monkeypatch, counters)
    monkeypatch.setenv("MS4_QM_INLINE", "1")
    _patch_catalog(monkeypatch, _inline_catalog(INLINE_CANONICAL))
    monkeypatch.setattr(
        "machine_spirit_4.gateway.quartermaster.decide",
        lambda query, **_k: _inline_decide(query),
    )

    def fake_execute(_url, name, **_k):
        counters.inline += 1
        return {"tool": name, "result": {"now": "2026-08-17T00:00:00Z"}, "elapsed_ms": 4}

    monkeypatch.setattr(
        "machine_spirit_4.gateway.quartermaster.execute_inline_tool",
        fake_execute,
    )
    fake = FakeFaceLobeChat()
    runner = Ms4HermesRunner(
        hermes_dir=str(tmp_path),
        agent_cls=FakeHermesAgent,
        face_lobe_chat=fake,
        hivemind_url="http://hive:6089",
    )
    monkeypatch.setattr(runner, "_try_cluster_time", lambda: None)
    response = runner.chat("what time is it", session_id="s-inline", model="m")
    assert counters.inline == 1
    assert counters.submit == 0
    assert counters.direct == 0
    assert response["dispatched_job"] is None
    assert response["quartermaster_inline"] is True
    assert INLINE_CANONICAL in response["text"]


def test_admitted_inline_none_is_terminal_no_fallthrough(monkeypatch, tmp_path):
    counters = _SeamCounters()
    _install_submit_counter(monkeypatch, counters)
    _deep_route(monkeypatch)
    monkeypatch.setenv("MS4_QM_INLINE", "1")
    _patch_catalog(monkeypatch, _inline_catalog(INLINE_CANONICAL))
    monkeypatch.setattr(
        "machine_spirit_4.gateway.quartermaster.decide",
        lambda query, **_k: _inline_decide(query),
    )

    def fake_execute(_url, name, **_k):
        counters.inline += 1
        return None

    monkeypatch.setattr(
        "machine_spirit_4.gateway.quartermaster.execute_inline_tool",
        fake_execute,
    )
    runner = Ms4HermesRunner(
        hermes_dir=str(tmp_path),
        agent_cls=FakeHermesAgent,
        face_lobe_chat=FakeFaceLobeChat(),
        hivemind_url="http://hive:6089",
    )
    monkeypatch.setattr(runner, "_try_cluster_time", lambda: None)
    response = runner.chat("what time is it", session_id="s-inline-none", model="m")
    assert counters.inline == 1
    assert counters.submit == 0
    assert counters.direct == 0
    assert response["dispatched_job"] is None
    assert "did not retry" in response["text"] or "did not return" in response["text"]


def test_admitted_inline_raise_is_terminal_no_fallthrough(monkeypatch, tmp_path):
    counters = _SeamCounters()
    _install_submit_counter(monkeypatch, counters)
    _deep_route(monkeypatch)
    monkeypatch.setenv("MS4_QM_INLINE", "1")
    _patch_catalog(monkeypatch, _inline_catalog(INLINE_CANONICAL))
    monkeypatch.setattr(
        "machine_spirit_4.gateway.quartermaster.decide",
        lambda query, **_k: _inline_decide(query),
    )

    def fake_execute(_url, name, **_k):
        counters.inline += 1
        raise RuntimeError("inline exploded")

    monkeypatch.setattr(
        "machine_spirit_4.gateway.quartermaster.execute_inline_tool",
        fake_execute,
    )
    runner = Ms4HermesRunner(
        hermes_dir=str(tmp_path),
        agent_cls=FakeHermesAgent,
        face_lobe_chat=FakeFaceLobeChat(),
        hivemind_url="http://hive:6089",
    )
    monkeypatch.setattr(runner, "_try_cluster_time", lambda: None)
    response = runner.chat("what time is it", session_id="s-inline-raise", model="m")
    assert counters.inline == 1
    assert counters.submit == 0
    assert counters.direct == 0
    assert response["dispatched_job"] is None
    assert "did not retry" in response["text"] or "did not return" in response["text"]


def test_exact_read_catalog_errors_refuse_zero_effect(monkeypatch, tmp_path):
    counters = _SeamCounters()
    _install_submit_counter(monkeypatch, counters)
    _deep_route(monkeypatch)
    monkeypatch.setenv("MS4_QM_INLINE", "0")
    _patch_catalog(
        monkeypatch,
        _inline_catalog(EXACT_READ_CANONICAL, errors=("manifest load failed",)),
    )
    runner = Ms4HermesRunner(
        hermes_dir=str(tmp_path),
        agent_cls=FakeHermesAgent,
        face_lobe_chat=FakeFaceLobeChat(),
        hivemind_url="http://hive:6089",
    )
    monkeypatch.setattr(runner, "_try_cluster_time", lambda: None)
    monkeypatch.setattr(
        runner,
        "_try_quartermaster_inline",
        lambda message: (None, _exact_read_outcome(message)),
    )
    response = runner.chat(EXACT_READ_GOAL, session_id="s-exact-errors", model="m")
    assert counters.inline == 0
    assert counters.submit == 0
    assert counters.direct == 0
    assert response["dispatched_job"] is None
    assert "not an admitted" in response["text"] or "did not run" in response["text"]


def test_accepted_exact_depth_route_submits_once_with_exact_toolset(monkeypatch, tmp_path):
    counters = _SeamCounters()
    _install_submit_counter(monkeypatch, counters)
    _deep_route(monkeypatch)
    monkeypatch.setenv("MS4_QM_INLINE", "0")
    _patch_catalog(monkeypatch, _inline_catalog(EXACT_READ_CANONICAL, errors=()))
    runner = Ms4HermesRunner(
        hermes_dir=str(tmp_path),
        agent_cls=FakeHermesAgent,
        face_lobe_chat=FakeFaceLobeChat(),
        hivemind_url="http://hive:6089",
    )
    monkeypatch.setattr(runner, "_try_cluster_time", lambda: None)
    monkeypatch.setattr(
        runner,
        "_try_quartermaster_inline",
        lambda message: (None, _exact_read_outcome(message)),
    )
    response = runner.chat(EXACT_READ_GOAL, session_id="s-exact-depth", model="m")
    assert counters.inline == 0
    assert counters.submit == 1
    assert counters.direct == 0
    assert counters.envelopes[0].resource_request.enabled_toolsets == [
        "mcp-hivemind-exact-read"
    ]
    assert response["dispatched_job"]["state"] == "queued"


def test_accepted_direct_exact_wrapper_is_one_handler_call(monkeypatch, tmp_path):
    counters = _SeamCounters()
    audit_events = []
    sentinel = "DO_NOT_PERSIST_ARGUMENT_OR_RESULT"
    _patch_catalog(monkeypatch, _inline_catalog(EXACT_READ_CANONICAL))
    runner = Ms4HermesRunner(
        hermes_dir=str(tmp_path),
        agent_cls=FakeHermesAgent,
        face_lobe_chat=FakeFaceLobeChat(),
        hivemind_url="http://hive:6089",
    )
    monkeypatch.setattr(runner, "require_plugin", lambda: None)
    monkeypatch.setattr(runner, "ensure_hermes_path", lambda: None)

    def fake_handle(tool_name, args, **kwargs):
        counters.direct += 1
        assert tool_name == "hivemind_exact_read"
        assert args["canonical_tool"] == EXACT_READ_CANONICAL
        return f'{{"ok":true,"value":"{sentinel}"}}'

    monkeypatch.setitem(sys.modules, "model_tools", type("M", (), {
        "handle_function_call": staticmethod(fake_handle),
        "get_toolset_for_tool": staticmethod(lambda _name: "mcp-hivemind-exact-read"),
    }))
    monkeypatch.setattr(
        hr,
        "append_event",
        lambda event_type, data: audit_events.append({"event_type": event_type, **data}),
    )
    result = runner.dispatch_hermes_tool(
        "hivemind_exact_read",
        {"canonical_tool": EXACT_READ_CANONICAL, "arguments": {"id": sentinel}},
    )
    assert counters.inline == 0
    assert counters.submit == 0
    assert counters.direct == 1
    assert sentinel in result["result"]
    assert result.get("refused") is not True
    assert audit_events == [
        {
            "event_type": "hermes_tool_call",
            "tool": "hivemind_exact_read",
            "toolset": "mcp-hivemind-exact-read",
            "argument_count": 2,
            "result_present": True,
            "result_type": "str",
        }
    ]
    assert sentinel not in json.dumps(audit_events)
