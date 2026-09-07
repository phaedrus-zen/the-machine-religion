"""Phase D — Face Lobe inline fast-path wiring in Ms4HermesRunner.chat().

Two layers of test:

1. ``chat()`` integration via the patchable ``_try_quartermaster_inline``
   seam: inline-hit skips Depth and Face inference and emits the
   authoritative payload directly; malformed or unadmitted mappings are
   refused without a Depth or Face effect.
2. ``_try_quartermaster_inline`` end-to-end against a fake MCP server
   (catalog tools/list + tool exec) with an injected ethics evaluator,
   proving a real read-only query runs inline and surfaces a block.
"""

from __future__ import annotations

import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from machine_spirit_4.double_agent import (
    Blackboard,
    DepthChoice,
    JobEnvelope,
    JobResult,
    ResourceRequest,
    safety as double_agent_safety,
)
from machine_spirit_4.double_agent.safety import INTERNAL_GOAL_MAX_CHARS
from machine_spirit_4.gateway import face_lobe_chat as face_lobe_chat_module
from machine_spirit_4.gateway import hermes_runner as hermes_runner_module
from machine_spirit_4.gateway.face_lobe_chat import FaceLobeChat
from machine_spirit_4.gateway.hermes_runner import Ms4HermesRunner, Ms4TurnCancelled


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeSession:
    def __init__(self, session_id, model):
        self.session_id = session_id
        self.model = model
        self.messages: list[dict[str, Any]] = []


class FakeFaceLobeChat:
    """Records Face inference and authoritative-result calls separately."""

    def __init__(self, *, hivemind_url: str = "http://hive:6089"):
        self.hivemind_url = hivemind_url
        self.calls: list[dict[str, Any]] = []
        self.authoritative_calls: list[dict[str, Any]] = []
        self._sessions: dict[str, _FakeSession] = {}

    def chat(
        self,
        message,
        *,
        session_id,
        model,
        stream_callback=None,
        extra_system=None,
        cancel_event=None,
    ):
        self.calls.append({"message": message, "extra_system": extra_system or ""})
        sid = session_id or "s"
        state = self._sessions.setdefault(sid, _FakeSession(sid, model))
        state.messages.extend([
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"reply:{message}"},
        ])
        return {
            "text": f"reply:{message}",
            "session_id": sid,
            "model": model,
            "runtime": "face-lobe-direct",
            "completed": True,
            "api_calls": 1,
            "metrics": {"schema": "Ms4TurnMetrics.v1"},
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
        cancel_event=None,
    ):
        self.authoritative_calls.append({
            "message": message,
            "authoritative_text": authoritative_text,
            "extra_system": extra_system or "",
        })
        sid = session_id or "s"
        state = self._sessions.setdefault(sid, _FakeSession(sid, model))
        state.messages.extend([
            {"role": "user", "content": message},
            {"role": "assistant", "content": authoritative_text},
        ])
        if stream_callback is not None:
            stream_callback(authoritative_text)
        return {
            "text": authoritative_text,
            "session_id": sid,
            "model": model,
            "runtime": "face-lobe-direct",
            "completed": True,
            "cancelled": False,
            "api_calls": 0,
            "fallback_used": False,
            "metrics": {
                "schema": "Ms4TurnMetrics.v1",
                "api_calls": 0,
                "streaming": stream_callback is not None,
            },
            "output_guard": {
                "schema": "Ms4FaceLobeOutputGuard.v1",
                "applied": False,
                "reason": None,
            },
        }

    def sessions(self):
        return []


class FakeHermesAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = kwargs.get("model")


def _runner(tmp_path, monkeypatch, hivemind_url="http://hive:6089"):
    monkeypatch.setenv("MS4_QM_INLINE", "1")  # opt back in for these tests
    return Ms4HermesRunner(
        hermes_dir=str(tmp_path),
        hivemind_url=hivemind_url,
        agent_cls=FakeHermesAgent,
        face_lobe_chat=FakeFaceLobeChat(hivemind_url=hivemind_url),
    )


_TWO_GPU_RESULT = {
    "gpus": [
        {"node_id": "GPU-NODE-ALPHA", "gpu_id": "GPU-ALPHA-0"},
        {"node_id": "GPU-NODE-BETA", "gpu_id": "GPU-BETA-1"},
    ]
}
_INLINE_TOOL = "hivemind.gpu.availability@v1"
_TWO_GPU_TEXT = json.dumps(_TWO_GPU_RESULT, ensure_ascii=False)
_AUTHORITATIVE_TEXT = f"Authoritative result from {_INLINE_TOOL}: {_TWO_GPU_TEXT}"
_TWO_GPU_BLOCK = (
    "MS4 Quartermaster inline tool result (authoritative; quote this faithfully, "
    "do not invent additional tools, data, or fields):\n"
    f"- tool: {_INLINE_TOOL}  (read-only, executed just now in 4 ms)\n"
    f"- result: {_TWO_GPU_TEXT}\n"
    "This is the answer to the operator's request. Summarise it in plain language."
)


def _patch_deep_route(monkeypatch):
    monkeypatch.setattr(
        "machine_spirit_4.gateway.hermes_runner.router_route",
        lambda message: type("R", (), {
            "kind": "deep", "cleaned_message": message, "goal": message,
            "to_dict": lambda self: {"kind": "deep"},
        })(),
    )


def _patch_direct_route(monkeypatch):
    monkeypatch.setattr(
        "machine_spirit_4.gateway.hermes_runner.router_route",
        lambda message: type("R", (), {
            "kind": "direct", "cleaned_message": message, "goal": "",
            "to_dict": lambda self: {"kind": "direct"},
        })(),
    )


def _valid_depth_outcome(query: str) -> dict[str, Any]:
    return {
        "schema": "Ms4ToolRouteDecision.v1",
        "verdict": "depth",
        "query": query,
        "reason": "top tool requires Depth",
        "resolution": {
            "schema": "Ms4ToolResolution.v1",
            "query": query,
            "tier": "deterministic",
            "confidence": 1.0,
            "toolboxes": ["jobs"],
            "tools": [
                {
                    "name": "hivemind.jobs.cancel@v1",
                    "toolbox": "jobs",
                    "cluster": "jobs",
                    "score": 1.0,
                    "kind": "hivemind_native",
                    "inline_eligible": False,
                }
            ],
            "catalog_version": "test",
            "fallback_reason": None,
        },
        "inline_tool": None,
        "inline_toolbox": None,
        "ethics": None,
    }


def _exact_read_depth_outcome(
    query: str,
    *,
    canonical_tool: str = "hivemind.app.get@v1",
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if tools is None:
        tools = [
            {
                "name": canonical_tool,
                "toolbox": "apps",
                "cluster": "applications",
                "score": 1.0,
                "kind": "hivemind_native",
                "inline_eligible": True,
            }
        ]
    return {
        "schema": "Ms4ToolRouteDecision.v1",
        "verdict": "depth",
        "query": query,
        "reason": f"top tool {canonical_tool} requires args ['id']",
        "resolution": {
            "schema": "Ms4ToolResolution.v1",
            "query": query,
            "tier": "deterministic",
            "confidence": 1.0,
            "toolboxes": ["apps"],
            "tools": tools,
            "catalog_version": "test",
            "fallback_reason": None,
        },
        "inline_tool": None,
        "inline_toolbox": None,
        "ethics": None,
    }


def _exact_gated_depth_outcome(
    query: str,
    *,
    canonical_tool: str = "hivemind.services.enable@v1",
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if tools is None:
        tools = [
            {
                "name": canonical_tool,
                "toolbox": "services",
                "cluster": "operations",
                "score": 1.0,
                "kind": "hivemind_native",
                "inline_eligible": False,
            }
        ]
    return {
        "schema": "Ms4ToolRouteDecision.v1",
        "verdict": "depth",
        "query": query,
        "reason": f"top tool {canonical_tool} is not inline-eligible",
        "resolution": {
            "schema": "Ms4ToolResolution.v1",
            "query": query,
            "tier": "deterministic",
            "confidence": 1.0,
            "toolboxes": ["services"],
            "tools": tools,
            "catalog_version": "test",
            "fallback_reason": None,
        },
        "inline_tool": None,
        "inline_toolbox": None,
        "ethics": None,
    }


def _valid_none_outcome(query: str) -> dict[str, Any]:
    return {
        "schema": "Ms4ToolRouteDecision.v1",
        "verdict": "none",
        "query": query,
        "reason": "no tool resolved",
        "resolution": {
            "schema": "Ms4ToolResolution.v1",
            "query": query,
            "tier": "none",
            "confidence": 0.0,
            "toolboxes": [],
            "tools": [],
            "catalog_version": "test",
            "fallback_reason": "no tool resolved",
        },
        "inline_tool": None,
        "inline_toolbox": None,
        "ethics": None,
    }


# ---------------------------------------------------------------------------
# chat() integration via the patchable seam
# ---------------------------------------------------------------------------


def test_inline_success_preserves_two_gpu_records_without_face_inference(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch)
    monkeypatch.setattr(
        runner, "_try_quartermaster_inline",
        lambda message: (
            _TWO_GPU_BLOCK,
            {
                "verdict": "inline",
                "inline_invoked": True,
                "inline_executed": True,
                "inline_tool": _INLINE_TOOL,
                "inline_result": _TWO_GPU_RESULT,
                "inline_result_text": _TWO_GPU_TEXT,
            },
        ),
    )
    _patch_deep_route(monkeypatch)
    monkeypatch.setattr(runner, "_try_cluster_time", lambda: None)

    resp = runner.chat("what gpus are available", session_id="s1", model="m")

    assert resp["text"] == _AUTHORITATIVE_TEXT
    assert _INLINE_TOOL in resp["text"]
    assert resp["text"] != _TWO_GPU_TEXT, "inline output must not be a bare JSON payload"
    assert not resp["text"].lstrip().startswith(("{", "["))
    assert resp["text"].index("GPU-NODE-ALPHA") < resp["text"].index("GPU-NODE-BETA")
    assert resp["text"].index("GPU-ALPHA-0") < resp["text"].index("GPU-BETA-1")
    assert runner.face_lobe_chat.calls == [], "successful inline execution must not call Face inference"
    assert len(runner.face_lobe_chat.authoritative_calls) == 1
    assert resp["quartermaster_inline"] is True
    assert resp["dispatched_job"] is None
    assert resp["depth_lobe_model"] is None
    assert resp["api_calls"] == 0
    assert "Depth Lobe" not in resp["text"]
    assert "depth_lobe_dispatched" not in resp["grounding_source"]


def test_unreceipted_result_shaped_inline_outcome_has_zero_effects(
    tmp_path,
    monkeypatch,
):
    runner = _runner(tmp_path, monkeypatch)
    query = "what gpus are available"
    monkeypatch.setattr(
        runner,
        "_try_quartermaster_inline",
        lambda _message: (
            _TWO_GPU_BLOCK,
            {
                "schema": "Ms4ToolRouteDecision.v1",
                "verdict": "inline",
                "query": query,
                "inline_tool": _INLINE_TOOL,
                "inline_executed": True,
                "inline_executed_tool": _INLINE_TOOL,
                "inline_result": _TWO_GPU_RESULT,
                "inline_result_text": _TWO_GPU_TEXT,
            },
        ),
    )
    _patch_deep_route(monkeypatch)
    monkeypatch.setattr(runner, "_try_cluster_time", lambda: None)
    monkeypatch.setattr(
        hermes_runner_module,
        "_catalog_proves_inline_execute",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        hermes_runner_module,
        "face_lobe_turn_start",
        lambda **_kwargs: {"revision": {"revision_id": 1}, "marked_stale": []},
    )

    class _NoEffectRunner:
        def list(self, **_kwargs):
            return []

        def submit(self, _envelope):
            raise AssertionError("unreceipted inline outcome dispatched Depth")

    monkeypatch.setattr(hermes_runner_module, "default_runner", lambda: _NoEffectRunner())

    response = runner.chat(query, session_id="unreceipted-inline", model="m")

    assert response["api_calls"] == 0
    assert response["dispatched_job"] is None
    assert response["quartermaster_inline"] is False
    assert response["tool_trace"] == []
    assert response["metrics"]["block_reason"] == "admission_refused"
    assert "inline execution receipt missing" in response["text"]
    assert runner.face_lobe_chat.calls == []
    assert runner.face_lobe_chat.authoritative_calls == []
    assert runner.face_lobe_chat._sessions == {}


class _ModelTrapFaceLobeChat(FaceLobeChat):
    def __init__(self):
        super().__init__(hivemind_url="http://hive:6089", empty_fallback_model="")
        self.model_calls = 0

    def _post_blocking(self, _payload):
        self.model_calls += 1
        raise AssertionError("Face/model inference was called after successful inline execution")

    def _post_streaming(self, _payload, _stream_callback, *, cancel_event=None):
        self.model_calls += 1
        raise AssertionError("Face/model inference was called after successful inline execution")


def _guarded_inline_runner(tmp_path, monkeypatch, *, hivemind_url="http://hive:6089"):
    face = _ModelTrapFaceLobeChat()
    face.hivemind_url = hivemind_url
    runner = Ms4HermesRunner(
        hermes_dir=str(tmp_path),
        hivemind_url=hivemind_url,
        agent_cls=FakeHermesAgent,
        face_lobe_chat=face,
    )
    monkeypatch.setenv("MS4_QM_INLINE", "1")
    _patch_deep_route(monkeypatch)
    monkeypatch.setattr(runner, "_try_cluster_time", lambda: None)
    monkeypatch.setattr(
        "machine_spirit_4.gateway.hermes_runner.face_lobe_turn_start",
        lambda **_kwargs: {"revision": {"revision_id": 1}, "marked_stale": []},
    )
    monkeypatch.setattr(
        "machine_spirit_4.gateway.hermes_runner.build_face_lobe_context_block",
        lambda **_kwargs: (
            "MS4 Face Lobe context (authoritative):\n"
            "- THIS TURN DID NOT DISPATCH any background work."
        ),
    )
    monkeypatch.setattr(
        "machine_spirit_4.gateway.hermes_runner.build_grounded_user_message",
        lambda message, _url: (message, "none"),
    )

    class _NoDepthRunner:
        def submit(self, _envelope):
            raise AssertionError("Depth dispatch occurred after successful inline execution")

    monkeypatch.setattr(
        "machine_spirit_4.gateway.hermes_runner.default_runner",
        lambda: _NoDepthRunner(),
    )
    return runner, face


def test_inline_stream_uses_guard_once_commits_truthful_history_and_zero_api_calls(
    tmp_path,
    monkeypatch,
):
    face = _ModelTrapFaceLobeChat()
    runner = Ms4HermesRunner(
        hermes_dir=str(tmp_path),
        hivemind_url="http://hive:6089",
        agent_cls=FakeHermesAgent,
        face_lobe_chat=face,
    )
    monkeypatch.setenv("MS4_QM_INLINE", "1")
    monkeypatch.setattr(
        runner,
        "_try_quartermaster_inline",
        lambda _message: (
            _TWO_GPU_BLOCK,
            {
                "verdict": "inline",
                "inline_invoked": True,
                "inline_executed": True,
                "inline_tool": _INLINE_TOOL,
                "inline_result": _TWO_GPU_RESULT,
                "inline_result_text": _TWO_GPU_TEXT,
            },
        ),
    )
    _patch_deep_route(monkeypatch)
    monkeypatch.setattr(runner, "_try_cluster_time", lambda: None)
    monkeypatch.setattr(
        "machine_spirit_4.gateway.hermes_runner.build_grounded_user_message",
        lambda message, _url: (message, "none"),
    )

    class _NoDepthRunner:
        def submit(self, _envelope):
            raise AssertionError("Depth dispatch occurred after successful inline execution")

    monkeypatch.setattr(
        "machine_spirit_4.gateway.hermes_runner.default_runner",
        lambda: _NoDepthRunner(),
    )
    real_guard = face_lobe_chat_module.apply_face_lobe_output_guard
    guard_calls: list[dict[str, Any]] = []

    def recording_guard(text, *, message, extra_system):
        guard_calls.append({
            "text": text,
            "message": message,
            "extra_system": extra_system,
        })
        return real_guard(text, message=message, extra_system=extra_system)

    monkeypatch.setattr(face_lobe_chat_module, "apply_face_lobe_output_guard", recording_guard)
    streamed: list[str] = []

    resp = runner.chat(
        "what gpus are available",
        session_id="s-stream",
        model="m",
        stream_callback=streamed.append,
    )

    assert face.model_calls == 0
    assert len(guard_calls) == 1
    assert guard_calls[0]["text"] == _AUTHORITATIVE_TEXT
    assert _TWO_GPU_BLOCK in guard_calls[0]["extra_system"]
    assert streamed == [_AUTHORITATIVE_TEXT]
    assert resp["text"] == _AUTHORITATIVE_TEXT
    assert _INLINE_TOOL in resp["text"]
    assert resp["text"] != _TWO_GPU_TEXT, "inline output must not be a bare JSON payload"
    assert not resp["text"].lstrip().startswith(("{", "["))
    assert resp["completed"] is True
    assert resp["api_calls"] == 0
    assert resp["metrics"]["api_calls"] == 0
    assert resp["metrics"]["streaming"] is True
    assert resp["face_lobe_output_guard"]["applied"] is False
    assert resp["fabrication_guard"]["applied"] is False
    assert face._sessions["s-stream"].messages == [
        {"role": "user", "content": "what gpus are available"},
        {"role": "assistant", "content": _AUTHORITATIVE_TEXT},
    ]
    assert resp["dispatched_job"] is None
    assert resp["depth_lobe_model"] is None
    assert resp["tool_trace"] == []
    assert "Depth Lobe" not in resp["text"]
    assert "depth_lobe_dispatched" not in resp["grounding_source"]


def test_voice_inline_verbose_oracle_status_is_one_bounded_grounded_sentence(
    tmp_path,
    monkeypatch,
):
    runner, face = _guarded_inline_runner(tmp_path, monkeypatch)
    executed_tool = "hivemind.oracle.status"
    correlation_marker = "ORACLE72-CORR-8F3C2A1D"
    ungrounded_claim = "EVERY subsystem is perfect! Launch anything you want."
    verbose_result = {
        "schema": "OracleStatus.v1",
        "healthy": True,
        "state": "ready",
        "active_requests": 0,
        "running_plans": 3,
        "summary": ungrounded_claim,
        "requests": [
            {
                "id": f"request-{index:03d}",
                "detail": "Verbose operator-only request detail that must remain structured.",
            }
            for index in range(24)
        ],
    }
    verbose_result_text = json.dumps(verbose_result, ensure_ascii=False)
    inline_block = (
        "MS4 Quartermaster inline tool result (authoritative; quote this faithfully, "
        "do not invent additional tools, data, or fields):\n"
        f"- tool: {executed_tool}  (read-only, executed just now in 46 ms)\n"
        f"- result: {verbose_result_text}"
    )
    monkeypatch.setattr(
        runner,
        "_try_quartermaster_inline",
        lambda _message: (
            inline_block,
            {
                "verdict": "inline",
                "inline_invoked": True,
                "inline_executed": True,
                "inline_tool": executed_tool,
                "inline_executed_tool": executed_tool,
                "inline_elapsed_ms": 46,
                "inline_result": verbose_result,
                "inline_result_text": verbose_result_text,
            },
        ),
    )
    real_guard = face_lobe_chat_module.apply_face_lobe_output_guard
    guard_calls: list[dict[str, Any]] = []

    def recording_guard(text, *, message, extra_system):
        guard_calls.append({
            "text": text,
            "message": message,
            "extra_system": extra_system,
        })
        return real_guard(text, message=message, extra_system=extra_system)

    monkeypatch.setattr(face_lobe_chat_module, "apply_face_lobe_output_guard", recording_guard)
    streamed: list[str] = []
    prompt = (
        f"Correlation marker: {correlation_marker}. "
        "In one short sentence, confirm the current Oracle status."
    )

    response = runner.chat(
        prompt,
        session_id="s-voice-inline-status",
        model="m",
        voice_mode=True,
        stream_callback=streamed.append,
    )

    spoken = response["text"]
    assert streamed == [spoken]
    assert len(spoken) <= 240
    assert re.findall(r"[.!?](?=\s|$)", spoken) == ["."]
    assert correlation_marker in spoken
    assert executed_tool in spoken
    for grounded_scalar in (
        "healthy=true",
        "state=ready",
        "active_requests=0",
        "running_plans=3",
    ):
        assert grounded_scalar in spoken
    assert ungrounded_claim not in spoken
    assert "request-023" not in spoken
    assert response["quartermaster"]["inline_result"] == verbose_result
    assert response["quartermaster"]["inline_result_text"] == verbose_result_text
    assert len(guard_calls) == 1
    assert verbose_result_text in guard_calls[0]["extra_system"]
    assert response["api_calls"] == 0
    assert response["metrics"]["api_calls"] == 0
    assert response["face_lobe_output_guard"]["applied"] is False
    assert face.model_calls == 0
    assert response["dispatched_job"] is None
    assert response["depth_lobe_model"] is None
    assert response["tool_trace"][0]["tool"] == executed_tool


def test_inline_public_trace_reports_actual_tool_metadata_without_private_fields(
    tmp_path,
    monkeypatch,
):
    runner, face = _guarded_inline_runner(tmp_path, monkeypatch)
    executed_tool = "hivemind.time.now@v1"
    result = {"iso": "2026-07-11T13:40:00Z", "timezone": "UTC"}
    result_text = json.dumps(result, ensure_ascii=False)
    authoritative_text = f"Authoritative result from {executed_tool}: {result_text}"
    block = (
        "MS4 Quartermaster inline tool result (authoritative; quote this faithfully, "
        "do not invent additional tools, data, or fields):\n"
        f"- tool: {executed_tool}  (read-only, executed just now in 17 ms)\n"
        f"- result: {result_text}"
    )
    monkeypatch.setattr(
        runner,
        "_try_quartermaster_inline",
        lambda _message: (
            block,
            {
                "verdict": "inline",
                "inline_invoked": True,
                "inline_executed": True,
                "inline_tool": "hivemind.cluster.summary@v1",
                "inline_executed_tool": executed_tool,
                "inline_elapsed_ms": 17,
                "inline_result": result,
                "inline_result_text": result_text,
                "raw_prompt": "private operator prompt",
                "credential": "sk-private-value",
                "reasoning": "hidden chain-of-thought",
            },
        ),
    )
    audit_events: list[dict[str, Any]] = []
    monkeypatch.setattr(
        hermes_runner_module,
        "append_event",
        lambda event_type, data: audit_events.append(
            {"event_type": event_type, **data}
        ),
    )
    streamed: list[str] = []

    resp = runner.chat(
        "what time is it",
        session_id="s-public-inline-trace",
        model="m",
        stream_callback=streamed.append,
    )

    assert resp["text"] == authoritative_text
    assert streamed == [authoritative_text]
    assert resp["api_calls"] == 0
    assert resp["quartermaster_inline"] is True
    assert resp["quartermaster"]["inline_executed"] is True
    assert resp["dispatched_job"] is None
    assert face.model_calls == 0
    assert resp["tool_trace"] == [
        {
            "event": "complete",
            "tool": executed_tool,
            "args": {},
            "success": True,
            "duration_ms": 17,
            "result_type": "object",
            "result_bytes": len(result_text.encode("utf-8")),
        }
    ]
    chat_turn = next(
        event for event in audit_events if event["event_type"] == "chat_turn"
    )
    assert chat_turn["tool_trace_count"] == 1
    serialized_trace = json.dumps(resp["tool_trace"], sort_keys=True)
    for forbidden in (
        "raw_prompt",
        "private operator prompt",
        "credential",
        "sk-private-value",
        "reasoning",
        "hidden chain-of-thought",
    ):
        assert forbidden not in serialized_trace


def test_inline_stream_callback_false_cancels_without_history_or_followup_chunks(
    tmp_path,
    monkeypatch,
):
    runner, face = _guarded_inline_runner(tmp_path, monkeypatch)
    monkeypatch.setattr(
        runner,
        "_try_quartermaster_inline",
        lambda _message: (
            _TWO_GPU_BLOCK,
            {
                "verdict": "inline",
                "inline_invoked": True,
                "inline_executed": True,
                "inline_tool": _INLINE_TOOL,
                "inline_executed_tool": _INLINE_TOOL,
                "inline_result": _TWO_GPU_RESULT,
                "inline_result_text": _TWO_GPU_TEXT,
            },
        ),
    )
    streamed: list[str] = []

    def reject_first_chunk(chunk):
        streamed.append(chunk)
        return False

    resp = runner.chat(
        "what gpus are available",
        session_id="s-stream-cancel",
        model="m",
        stream_callback=reject_first_chunk,
    )

    assert streamed == [_AUTHORITATIVE_TEXT]
    assert resp["text"] == ""
    assert resp["completed"] is False
    assert resp["cancelled"] is True
    assert resp["api_calls"] == 0
    assert resp["metrics"]["api_calls"] == 0
    assert resp["metrics"]["cancelled"] is True
    assert resp["metrics"]["cancel_reason"] == "downstream_callback"
    assert "trace_id" not in resp["metrics"]
    assert face.model_calls == 0
    assert face._sessions["s-stream-cancel"].messages == []
    assert resp["dispatched_job"] is None
    assert resp["depth_lobe_model"] is None
    assert "Depth Lobe" not in "".join(streamed)


def test_inline_wrapper_prefers_actual_executed_tool_identity(tmp_path, monkeypatch):
    runner, face = _guarded_inline_runner(tmp_path, monkeypatch)
    selected_tool = "hivemind.gpu.availability@v1"
    executed_tool = "hivemind.cluster.summary@v1"
    result_text = "actual executed result"
    block = (
        "MS4 Quartermaster inline tool result (authoritative; quote this faithfully, "
        "do not invent additional tools, data, or fields):\n"
        f"- tool: {executed_tool}  (read-only, executed just now in 4 ms)\n"
        f"- result: {result_text}"
    )
    monkeypatch.setattr(
        runner,
        "_try_quartermaster_inline",
        lambda _message: (
            block,
            {
                "verdict": "inline",
                "inline_invoked": True,
                "inline_executed": True,
                "inline_tool": selected_tool,
                "inline_executed_tool": executed_tool,
                "inline_result": result_text,
                "inline_result_text": result_text,
            },
        ),
    )

    resp = runner.chat("inspect the cluster", session_id="s-identity", model="m")

    assert resp["text"] == f"Authoritative result from {executed_tool}: {result_text}"
    assert executed_tool in resp["text"]
    assert selected_tool not in resp["text"]
    assert face.model_calls == 0
    assert resp["api_calls"] == 0
    assert resp["dispatched_job"] is None


def test_pre_cancelled_inline_turn_emits_nothing_and_changes_no_history(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch)
    state = _FakeSession("s-cancelled", "m")
    state.messages = [
        {"role": "user", "content": "prior"},
        {"role": "assistant", "content": "prior answer"},
    ]
    runner.face_lobe_chat._sessions[state.session_id] = state
    before = list(state.messages)
    streamed: list[str] = []
    cancel_event = threading.Event()
    cancel_event.set()

    def inline_must_not_run(_message):
        raise AssertionError("pre-cancelled turn reached Quartermaster execution")

    monkeypatch.setattr(runner, "_try_quartermaster_inline", inline_must_not_run)
    _patch_deep_route(monkeypatch)

    class _NoDepthRunner:
        def submit(self, _envelope):
            raise AssertionError("pre-cancelled turn dispatched Depth")

    monkeypatch.setattr(
        "machine_spirit_4.gateway.hermes_runner.default_runner",
        lambda: _NoDepthRunner(),
    )

    with pytest.raises(Ms4TurnCancelled):
        runner.chat(
            "what gpus are available",
            session_id=state.session_id,
            model="m",
            stream_callback=streamed.append,
            cancel_event=cancel_event,
        )

    assert streamed == []
    assert runner.face_lobe_chat.calls == []
    assert runner.face_lobe_chat.authoritative_calls == []
    assert state.messages == before


def test_malformed_depth_mapping_is_refused_without_effects(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch)
    monkeypatch.setattr(
        runner, "_try_quartermaster_inline",
        lambda message: (None, {"verdict": "depth", "reason": "needs args"}),
    )
    monkeypatch.setattr(
        "machine_spirit_4.gateway.hermes_runner.router_route",
        lambda message: type("R", (), {
            "kind": "deep", "cleaned_message": message, "goal": "g",
            "to_dict": lambda self: {"kind": "deep"},
        })(),
    )

    resp = runner.chat("refactor the whole module and run the tests", session_id="s2", model="m")
    assert resp["quartermaster_inline"] is False
    assert resp["dispatched_job"] is None
    assert resp["quartermaster"] == {"verdict": "depth", "reason": "needs args"}
    assert resp["api_calls"] == 0
    assert resp["metrics"]["block_reason"] == "admission_refused"
    assert runner.face_lobe_chat.calls == []


def test_inline_helper_exception_is_fail_soft(tmp_path, monkeypatch):
    """If the inline helper itself raises, chat() must still complete
    via the Depth path (the helper swallows internally, but assert the
    turn survives a None return)."""
    runner = _runner(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "_try_quartermaster_inline", lambda message: (None, {"error": "boom"}))
    monkeypatch.setattr(
        "machine_spirit_4.gateway.hermes_runner.router_route",
        lambda message: type("R", (), {
            "kind": "deep", "cleaned_message": message, "goal": "g",
            "to_dict": lambda self: {"kind": "deep"},
        })(),
    )
    resp = runner.chat("do something tool-ish", session_id="s3", model="m")
    assert resp["completed"] is True
    assert resp["quartermaster_inline"] is False


@pytest.mark.parametrize(
    "quartermaster_outcome",
    [
        pytest.param(_valid_none_outcome("hello there"), id="no-resolution"),
        pytest.param({"error": "catalog unavailable"}, id="failed"),
        pytest.param({"verdict": "depth"}, id="malformed-depth"),
        pytest.param(_valid_depth_outcome("another turn"), id="mismatched-depth"),
    ],
)
def test_direct_route_without_inline_result_keeps_ordinary_face_chat(
    tmp_path,
    monkeypatch,
    quartermaster_outcome,
):
    runner = _runner(tmp_path, monkeypatch)
    calls: list[str] = []

    def spy(message):
        calls.append(message)
        return None, quartermaster_outcome

    monkeypatch.setattr(runner, "_try_quartermaster_inline", spy)
    _patch_direct_route(monkeypatch)
    monkeypatch.setattr(runner, "_try_cluster_time", lambda: None)

    resp = runner.chat("hello there", session_id="s4", model="m")

    assert calls == ["hello there"]
    assert len(runner.face_lobe_chat.calls) == 1
    assert runner.face_lobe_chat.authoritative_calls == []
    assert resp["text"] == "reply:hello there"
    assert resp["quartermaster_inline"] is False
    assert resp["quartermaster"] == quartermaster_outcome
    assert resp["api_calls"] == 1
    assert resp["dispatched_job"] is None


@pytest.mark.parametrize(
    "message",
    [
        pytest.param(
            "Remember these session facts and reply exactly READY.",
            id="memory-priming",
        ),
        pytest.param(
            "Explain Action 2 from the completed answer in at least 100 words.",
            id="referential-followup",
        ),
        pytest.param(
            "Which exact action mitigates ALPHA, and what owner name did I give you?",
            id="memory-recall",
        ),
    ],
)
def test_low_confidence_quartermaster_depth_cannot_upgrade_direct_conversation(
    tmp_path,
    monkeypatch,
    message,
):
    runner = _runner(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "_try_cluster_time", lambda: None)

    outcome = _valid_depth_outcome(message)
    outcome["reason"] = "confidence 0.140 < inline bar 0.250"
    outcome["resolution"]["tier"] = "embeddings"
    outcome["resolution"]["confidence"] = 0.14
    monkeypatch.setattr(
        runner,
        "_try_quartermaster_inline",
        lambda _message: (None, outcome),
    )

    class _NoDispatchRunner:
        def list(self, **_kwargs):
            return []

        def submit(self, _envelope):
            raise AssertionError("ordinary direct conversation dispatched Depth")

    monkeypatch.setattr(
        "machine_spirit_4.gateway.hermes_runner.default_runner",
        lambda: _NoDispatchRunner(),
    )

    resp = runner.chat(message, session_id="direct-conversation", model="m")

    assert resp["router"]["kind"] == "direct"
    assert resp["quartermaster"]["verdict"] == "depth"
    assert resp["dispatched_job"] is None
    assert resp["depth_lobe_model"] is None
    assert resp["text"] == f"reply:{message}"
    assert len(runner.face_lobe_chat.calls) == 1


def test_slash_direct_cannot_be_reinterpreted_as_quartermaster_depth(
    tmp_path,
    monkeypatch,
):
    runner = _runner(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "_try_cluster_time", lambda: None)

    def depth_outcome(message):
        return None, _exact_read_depth_outcome(message)

    monkeypatch.setattr(runner, "_try_quartermaster_inline", depth_outcome)
    monkeypatch.setattr(
        "machine_spirit_4.gateway.hermes_runner.default_runner",
        lambda: pytest.fail("/direct must not create or inspect a Depth runner"),
    )

    resp = runner.chat(
        "/direct use hivemind.app.get@v1 to inspect one registered app",
        session_id="slash-direct-no-depth",
        model="m",
    )

    assert resp["router"]["kind"] == "direct"
    assert resp["router"]["override"] is True
    assert resp["dispatched_job"] is None
    assert resp["runtime"] == "face-lobe-direct"
    assert len(runner.face_lobe_chat.calls) == 1
    assert runner.face_lobe_chat.calls[0]["message"] == (
        "use hivemind.app.get@v1 to inspect one registered app"
    )


def test_cancelled_after_strict_exact_read_outcome_submits_no_job(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch)
    _patch_direct_route(monkeypatch)
    monkeypatch.setattr(runner, "_try_cluster_time", lambda: None)
    monkeypatch.setattr(
        "machine_spirit_4.gateway.hermes_runner.face_lobe_turn_start",
        lambda **_kwargs: {"revision": {"revision_id": 8}, "marked_stale": []},
    )
    cancel_event = threading.Event()

    def cancel_with_depth_outcome(message):
        cancel_event.set()
        return None, _exact_read_depth_outcome(message)

    monkeypatch.setattr(runner, "_try_quartermaster_inline", cancel_with_depth_outcome)
    submitted: list[Any] = []

    class _CapturingRunner:
        def list(self, **_kwargs):
            return []

        def submit(self, envelope):
            submitted.append(envelope)
            return {"job_id": envelope.job_id, "state": "queued"}

    monkeypatch.setattr(
        "machine_spirit_4.gateway.hermes_runner.default_runner",
        lambda: _CapturingRunner(),
    )

    query = "Oracle, use hivemind.app.get@v1 to show me one registered app."
    with pytest.raises(Ms4TurnCancelled, match="after Quartermaster routing"):
        runner.chat(
            query,
            session_id="cancel-after-qm",
            model="m",
            cancel_event=cancel_event,
        )

    assert submitted == []
    assert runner.face_lobe_chat.calls == []
    assert runner.face_lobe_chat.authoritative_calls == []


def test_confirmation_depth_outcome_submits_offered_goal_once(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch)
    _patch_direct_route(monkeypatch)
    monkeypatch.setenv("MS4_DA_CONFIRM_DISPATCH", "1")
    monkeypatch.setattr(runner, "_try_cluster_time", lambda: None)
    monkeypatch.setattr(
        "machine_spirit_4.gateway.hermes_runner.face_lobe_turn_start",
        lambda **_kwargs: {"revision": {"revision_id": 9}, "marked_stale": []},
    )
    monkeypatch.setattr(
        "machine_spirit_4.gateway.hermes_runner.choose_depth_model",
        lambda **_kwargs: type("D", (), {
            "model_id": "depth-model",
            "to_dict": lambda self: {"model_id": "depth-model", "source": "test"},
        })(),
    )
    monkeypatch.setattr(
        "machine_spirit_4.gateway.hermes_runner.build_grounded_user_message",
        lambda message, _url: (message, "none"),
    )
    offered_goal = (
        "Explain why a 35B Depth lobe might outperform a 4B Face lobe. "
        "Give the recommendation, mechanism, tradeoffs, and a concrete example."
    )
    runner._pending_deep_goal["confirm-depth"] = offered_goal
    quartermaster_calls: list[str] = []

    def depth_outcome(message):
        quartermaster_calls.append(message)
        return None, _valid_none_outcome(message)

    monkeypatch.setattr(runner, "_try_quartermaster_inline", depth_outcome)
    submitted: list[Any] = []

    class _CapturingRunner:
        def submit(self, envelope):
            submitted.append(envelope)
            return {"job_id": envelope.job_id, "state": "queued"}

    monkeypatch.setattr(
        "machine_spirit_4.gateway.hermes_runner.default_runner",
        lambda: _CapturingRunner(),
    )

    resp = runner.chat("yes please", session_id="confirm-depth", model="m")

    assert resp["confirm_dispatch"] is True
    assert quartermaster_calls == [offered_goal]
    assert len(submitted) == 1
    assert submitted[0].internal_goal == offered_goal
    assert submitted[0].user_visible_goal == offered_goal
    assert runner.face_lobe_chat.calls[0]["message"] == "yes please"
    assert resp["dispatched_job"]["job_id"] == submitted[0].job_id


def _capture_depth_envelope(
    tmp_path,
    monkeypatch,
    *,
    query: str,
    outcome: dict[str, Any],
    route_kind: str,
    face_model: str = "face-model",
    depth_model: str | None = None,
):
    runner = _runner(tmp_path, monkeypatch)
    if route_kind == "direct":
        _patch_direct_route(monkeypatch)
    else:
        _patch_deep_route(monkeypatch)
    monkeypatch.setattr(runner, "_try_quartermaster_inline", lambda _message: (None, outcome))
    monkeypatch.setattr(runner, "_try_cluster_time", lambda: None)
    monkeypatch.setattr(
        "machine_spirit_4.gateway.hermes_runner.face_lobe_turn_start",
        lambda **_kwargs: {"revision": {"revision_id": 12}, "marked_stale": []},
    )
    depth_picker_calls: list[dict[str, Any]] = []

    def choose_depth(**kwargs):
        depth_picker_calls.append(kwargs)
        override = kwargs.get("envelope_override")
        if isinstance(override, str) and override.strip():
            return DepthChoice(
                model_id=override.strip(),
                source="envelope_override",
                detail="job envelope resource_request.model_override",
            )
        return DepthChoice(
            model_id="picked-depth-model",
            source="loaded",
            detail="test depth picker",
        )

    monkeypatch.setattr(
        "machine_spirit_4.gateway.hermes_runner.choose_depth_model",
        choose_depth,
    )
    monkeypatch.setattr(
        "machine_spirit_4.gateway.hermes_runner.build_grounded_user_message",
        lambda message, _url: (message, "none"),
    )
    monkeypatch.setattr(
        "machine_spirit_4.gateway.hermes_runner.build_face_lobe_context_block",
        lambda **_kwargs: "MS4 Face Lobe context (authoritative).",
    )
    submitted: list[Any] = []

    class _CapturingRunner:
        def submit(self, envelope):
            submitted.append(envelope)
            return {"job_id": envelope.job_id, "state": "queued"}

    monkeypatch.setattr(
        "machine_spirit_4.gateway.hermes_runner.default_runner",
        lambda: _CapturingRunner(),
    )

    response = runner.chat(
        query,
        session_id="exact-read-depth",
        model=face_model,
        depth_model=depth_model,
    )
    return runner, response, submitted, depth_picker_calls


class _PersistedCapturingRunner:
    def __init__(self, blackboard: Blackboard):
        self.blackboard = blackboard
        self.submitted: list[JobEnvelope] = []

    def list(self, **kwargs):
        return self.blackboard.list_jobs(**kwargs)

    def get_result(self, job_id):
        return self.blackboard.get_result(job_id)

    def submit(self, envelope):
        self.blackboard.insert_job(envelope)
        self.submitted.append(envelope)
        return self.blackboard.get_job_snapshot(envelope.job_id)


def _seed_persisted_exact_read_missing_job(
    board: Blackboard,
    *,
    conversation_id: str,
    canonical_tool: str = "hivemind.app.get@v1",
    missing_keys: tuple[str, ...] = ("id",),
    state: str = "completed",
    toolsets: list[str] | None = None,
    result_status: str = "success",
    evidence_mode: str = "verified",
) -> str:
    job_id = double_agent_safety.new_job_id()
    envelope = JobEnvelope(
        job_id=job_id,
        parent_conversation_id=conversation_id,
        conversation_revision_id=1,
        background_lobe_type="deep_chat",
        user_visible_goal=f"Use {canonical_tool}.",
        internal_goal=f"Use {canonical_tool}.",
        resource_request=ResourceRequest(
            model_override="face-model",
            enabled_toolsets=(
                list(toolsets)
                if toolsets is not None
                else ["mcp-hivemind-exact-read"]
            ),
        ),
    )
    board.insert_job(envelope)

    tool_call_id = f"call-{job_id}"
    bridge_result: dict[str, Any] = {
        "status": "missing_required_arguments",
        "success": False,
        "canonical_tool": canonical_tool,
        "missing_required_keys": list(missing_keys),
    }
    result_excerpt = json.dumps(bridge_result, separators=(",", ":"))
    evidence = [
        {
            "kind": "verified_tool_result",
            "tool": "hivemind_exact_read",
            "tool_call_id": tool_call_id,
            "evidence_ref": f"tool_call:{tool_call_id}",
            "result_excerpt": result_excerpt,
        }
    ]
    actions_taken = [
        {"tool": "hivemind_exact_read", "tool_call_id": tool_call_id}
    ]
    if evidence_mode == "malformed":
        evidence[0]["result_excerpt"] = "{not-json"
    elif evidence_mode == "unverified":
        evidence[0]["kind"] = "unverified_tool_result"
    elif evidence_mode == "ambiguous":
        second_call_id = f"{tool_call_id}-second"
        evidence.append({
            **evidence[0],
            "tool_call_id": second_call_id,
            "evidence_ref": f"tool_call:{second_call_id}",
        })
        actions_taken.append(
            {"tool": "hivemind_exact_read", "tool_call_id": second_call_id}
        )
    elif evidence_mode == "missing-canonical":
        parsed = json.loads(evidence[0]["result_excerpt"])
        parsed.pop("canonical_tool")
        evidence[0]["result_excerpt"] = json.dumps(parsed, separators=(",", ":"))
    elif evidence_mode == "no-result":
        evidence = []
        actions_taken = []

    if evidence_mode != "no-result":
        board.insert_result(
            JobResult(
                job_id=job_id,
                status=result_status,
                summary="Exact read requires one argument.",
                text="Please provide the missing argument.",
                evidence=evidence,
                actions_taken=actions_taken,
                confidence="high",
                conversation_revision_id=1,
            )
        )
    if state != "queued":
        board.update_job_state(
            job_id,
            state=state,
            finished_at="2026-07-11T06:00:00+00:00" if state == "completed" else None,
        )
    return job_id


def test_persisted_exact_read_refuses_foreign_voice_turn_result() -> None:
    job_id = double_agent_safety.new_job_id()
    owner_turn_id = "ms4-turn-aaaaaaaaaaaaaaaa"
    envelope = JobEnvelope(
        job_id=job_id,
        parent_conversation_id="conv-turn-owner",
        conversation_revision_id=1,
        background_lobe_type="deep_chat",
        user_visible_goal="Use hivemind.app.get@v1.",
        internal_goal="Use hivemind.app.get@v1.",
        resource_request=ResourceRequest(
            enabled_toolsets=["mcp-hivemind-exact-read"],
        ),
        turn_id=owner_turn_id,
    )
    job = envelope.to_dict()
    job.update(state="completed", finished_at="2026-07-11T06:00:00+00:00")
    tool_call_id = "call-foreign-turn"
    result = JobResult(
        job_id=job_id,
        status="success",
        summary="FOREIGN_TURN_SUMMARY",
        text="FOREIGN_TURN_PRIVATE_RESULT",
        evidence=[
            {
                "kind": "verified_tool_result",
                "tool": "hivemind_exact_read",
                "tool_call_id": tool_call_id,
                "evidence_ref": f"tool_call:{tool_call_id}",
                "result_excerpt": json.dumps(
                    {
                        "status": "missing_required_arguments",
                        "success": False,
                        "canonical_tool": "hivemind.app.get@v1",
                        "missing_required_keys": ["id"],
                    },
                    separators=(",", ":"),
                ),
            }
        ],
        actions_taken=[
            {"tool": "hivemind_exact_read", "tool_call_id": tool_call_id}
        ],
        confidence="high",
        conversation_revision_id=1,
        turn_id="ms4-turn-bbbbbbbbbbbbbbbb",
    ).to_dict()

    assert (
        hermes_runner_module._persisted_exact_read_missing_source(job, result)
        is None
    )


def _patch_persisted_continuation_catalog(
    monkeypatch,
    *tool_specs: dict[str, Any],
) -> None:
    from machine_spirit_4.gateway.quartermaster import catalog as catalog_module

    entries = []
    for spec in tool_specs:
        canonical_tool = spec["name"]
        entries.append(
            catalog_module.ToolEntry(
                schema="Ms4QuartermasterTool.v1",
                name=canonical_tool,
                toolbox="apps",
                cluster="applications",
                description=f"Test entry for {canonical_tool}",
                source=spec.get("source", catalog_module.SOURCE_HIVEMIND),
                kind=spec.get("kind", catalog_module.KIND_HIVEMIND_NATIVE),
                input_schema={
                    "type": "object",
                    "properties": {
                        key: {"type": "string"}
                        for key in spec.get("required", ("id",))
                    },
                    "required": list(spec.get("required", ("id",))),
                },
                gated_by=tuple(spec.get("gated_by", ())),
            )
        )
    catalog = catalog_module.Catalog(
        schema="Ms4QuartermasterCatalog.v1",
        version="continuation-test",
        built_at="2026-07-11T06:00:00+00:00",
        hivemind_url="http://hive:6089",
        tools=tuple(entries),
        toolboxes={"apps": tuple(entries)},
        sources={
            "hivemind": sum(
                entry.source == catalog_module.SOURCE_HIVEMIND for entry in entries
            ),
            "ms4": 0,
            "external_mcp": 0,
        },
        errors=(),
    )
    monkeypatch.setattr(catalog_module, "get_catalog", lambda _url: catalog)


def _patch_strict_gated_catalog(
    monkeypatch,
    *,
    canonical_tool: str = "hivemind.services.enable@v1",
    native_kind: str = "hivemind_native",
    proxy_kind: str = "runtime_action",
    gated_by: tuple[str, ...] = ("operator-level",),
    include_native: bool = True,
) -> None:
    from machine_spirit_4.gateway.quartermaster import catalog as catalog_module

    entries = ()
    if include_native:
        entries = (
            catalog_module.ToolEntry(
                schema="Ms4QuartermasterTool.v1",
                name=canonical_tool,
                toolbox="services",
                cluster="operations",
                description=f"Test gated entry for {canonical_tool}",
                source=catalog_module.SOURCE_HIVEMIND,
                kind=native_kind,
                input_schema={
                    "type": "object",
                    "properties": {"service_name": {"type": "string"}},
                    "required": ["service_name"],
                },
                gated_by=(),
            ),
        )
    catalog = catalog_module.Catalog(
        schema="Ms4QuartermasterCatalog.v1",
        version="gated-test",
        built_at="2026-07-11T06:00:00+00:00",
        hivemind_url="http://hive:6089",
        tools=entries,
        toolboxes={"services": entries},
        sources={"hivemind": len(entries), "ms4": 0, "external_mcp": 0},
        errors=(),
    )
    proxy = {
        "name": f"ms4.{canonical_tool}",
        "kind": proxy_kind,
        "gated_by": list(gated_by),
    }
    monkeypatch.setattr(catalog_module, "get_catalog", lambda _url: catalog)
    monkeypatch.setattr(
        catalog_module,
        "_load_ms4_manifest",
        lambda: ([proxy], []),
    )
    return catalog


def _configure_persisted_continuation_turn(
    tmp_path,
    monkeypatch,
    *,
    persisted_runner: _PersistedCapturingRunner,
    literal_message: str,
    expected_canonical: str,
    route_kind: str = "direct",
    cancel_event: threading.Event | None = None,
):
    runner = _runner(tmp_path, monkeypatch)
    if route_kind == "direct":
        _patch_direct_route(monkeypatch)
    else:
        _patch_deep_route(monkeypatch)
    monkeypatch.setattr(
        "machine_spirit_4.gateway.hermes_runner.default_runner",
        lambda: persisted_runner,
    )
    monkeypatch.setattr(runner, "_try_cluster_time", lambda: None)
    monkeypatch.setattr(
        "machine_spirit_4.gateway.hermes_runner.face_lobe_turn_start",
        lambda **_kwargs: {"revision": {"revision_id": 2}, "marked_stale": []},
    )
    monkeypatch.setattr(
        "machine_spirit_4.gateway.hermes_runner.choose_depth_model",
        lambda **_kwargs: DepthChoice(
            model_id="picked-depth-model",
            source="loaded",
            detail="test depth picker",
        ),
    )
    monkeypatch.setattr(
        "machine_spirit_4.gateway.hermes_runner.build_grounded_user_message",
        lambda message, _url: (message, "none"),
    )
    monkeypatch.setattr(
        "machine_spirit_4.gateway.hermes_runner.build_face_lobe_context_block",
        lambda **_kwargs: "MS4 Face Lobe context (authoritative).",
    )
    quartermaster_calls: list[str] = []

    def route_with_controlled_outcome(dispatch_intent):
        quartermaster_calls.append(dispatch_intent)
        if dispatch_intent != literal_message:
            if cancel_event is not None:
                cancel_event.set()
            return None, _exact_read_depth_outcome(
                dispatch_intent,
                canonical_tool=expected_canonical,
            )
        return None, _valid_none_outcome(dispatch_intent)

    monkeypatch.setattr(
        runner,
        "_try_quartermaster_inline",
        route_with_controlled_outcome,
    )
    return runner, quartermaster_calls


def _persisted_continuation_links(board: Blackboard, source_job_id: str):
    return [
        event
        for event in board.list_events(source_job_id, limit=1000)
        if isinstance(event.get("payload"), dict)
        and event["payload"].get("source")
        == "hermes_runner_exact_read_continuation"
    ]


def test_explicit_single_exact_read_narrows_depth_envelope_to_bridge_only(
    tmp_path,
    monkeypatch,
):
    query = "Oracle, use hivemind.app.get@v1 to show me one registered app."
    canonical_tool = "hivemind.app.get@v1"
    outcome = _exact_read_depth_outcome(query, canonical_tool=canonical_tool)
    _patch_persisted_continuation_catalog(
        monkeypatch,
        {"name": canonical_tool, "required": ("id",)},
    )

    runner, response, submitted, depth_picker_calls = _capture_depth_envelope(
        tmp_path,
        monkeypatch,
        query=query,
        outcome=outcome,
        route_kind="direct",
    )

    assert len(submitted) == 1
    envelope = submitted[0]
    assert envelope.resource_request.enabled_toolsets == ["mcp-hivemind-exact-read"]
    assert envelope.to_dict()["resource_request"]["enabled_toolsets"] == [
        "mcp-hivemind-exact-read"
    ]
    assert envelope.resource_request.model_override == "face-model"
    assert depth_picker_calls == []
    assert response["depth_lobe_model"] == {
        "schema": "Ms4DepthModel.v1",
        "model_id": "face-model",
        "source": "foreground_reuse_exact_read",
        "detail": "strict exact-read reused selected foreground model",
    }
    assert envelope.user_visible_goal == query
    assert query in envelope.internal_goal
    assert (
        "Call `hivemind_exact_read` exactly once with `canonical_tool` set to "
        f"`{canonical_tool}`."
    ) in envelope.internal_goal
    assert "Never call or substitute any other tool." in envelope.internal_goal
    assert "`missing_required_keys`" in envelope.internal_goal
    assert "report those keys and ask the user for them" in envelope.internal_goal
    assert len(envelope.internal_goal) <= INTERNAL_GOAL_MAX_CHARS
    assert runner.face_lobe_chat.calls[0]["message"] == query
    assert response["quartermaster"] == outcome
    assert response["dispatched_job"]["job_id"] == envelope.job_id


def test_explicit_depth_model_wins_verbatim_for_strict_exact_read(
    tmp_path,
    monkeypatch,
):
    query = "Oracle, use hivemind.app.get@v1 to show me one registered app."
    outcome = _exact_read_depth_outcome(query)
    _patch_persisted_continuation_catalog(
        monkeypatch,
        {"name": "hivemind.app.get@v1", "required": ("id",)},
    )

    _runner_instance, response, submitted, depth_picker_calls = _capture_depth_envelope(
        tmp_path,
        monkeypatch,
        query=query,
        outcome=outcome,
        route_kind="direct",
        face_model="face-model",
        depth_model="llama3.1:8b",
    )

    assert len(submitted) == 1
    envelope = submitted[0]
    assert envelope.resource_request.enabled_toolsets == ["mcp-hivemind-exact-read"]
    assert envelope.resource_request.model_override == "llama3.1:8b"
    assert depth_picker_calls == [
        {
            "hivemind_url": "http://hive:6089",
            "envelope_override": "llama3.1:8b",
        }
    ]
    assert response["depth_lobe_model"] == {
        "schema": "Ms4DepthModel.v1",
        "model_id": "llama3.1:8b",
        "source": "envelope_override",
        "detail": "job envelope resource_request.model_override",
    }


def test_invalid_foreground_model_falls_back_to_depth_picker_for_exact_read(
    tmp_path,
    monkeypatch,
):
    query = "Oracle, use hivemind.app.get@v1 to show me one registered app."
    outcome = _exact_read_depth_outcome(query)
    _patch_persisted_continuation_catalog(
        monkeypatch,
        {"name": "hivemind.app.get@v1", "required": ("id",)},
    )

    _runner_instance, response, submitted, depth_picker_calls = _capture_depth_envelope(
        tmp_path,
        monkeypatch,
        query=query,
        outcome=outcome,
        route_kind="direct",
        face_model="",
    )

    assert len(submitted) == 1
    assert submitted[0].resource_request.model_override == "picked-depth-model"
    assert len(depth_picker_calls) == 1
    assert depth_picker_calls[0]["envelope_override"] is None
    assert response["depth_lobe_model"]["source"] == "loaded"
    assert response["depth_lobe_model"]["source"] != "foreground_reuse_exact_read"


def test_explicit_single_gated_tool_uses_background_only_gated_bridge(
    tmp_path,
    monkeypatch,
):
    query = (
        "Oracle, use hivemind.services.enable@v1 to enable "
        "menta_human_bridge."
    )
    canonical_tool = "hivemind.services.enable@v1"
    outcome = _exact_gated_depth_outcome(query, canonical_tool=canonical_tool)
    _patch_strict_gated_catalog(
        monkeypatch,
        canonical_tool=canonical_tool,
    )

    runner, response, submitted, depth_picker_calls = _capture_depth_envelope(
        tmp_path,
        monkeypatch,
        query=query,
        outcome=outcome,
        route_kind="deep",
    )

    assert len(submitted) == 1
    envelope = submitted[0]
    assert envelope.resource_request.enabled_toolsets == [
        "mcp-hivemind-exact-gated"
    ]
    assert envelope.resource_request.model_override == "face-model"
    assert depth_picker_calls == []
    assert response["depth_lobe_model"] == {
        "schema": "Ms4DepthModel.v1",
        "model_id": "face-model",
        "source": "foreground_reuse_exact_gated",
        "detail": "strict exact-gated reused selected foreground model",
    }
    assert envelope.user_visible_goal == query
    assert query in envelope.internal_goal
    assert "[MS4 TRUSTED EXACT-GATED DIRECTIVE]" in envelope.internal_goal
    assert (
        "Call `hivemind_exact_gated` exactly once with `canonical_tool` set to "
        f"`{canonical_tool}`."
    ) in envelope.internal_goal
    assert "Never call or substitute any other tool." in envelope.internal_goal
    assert "Human Bridge approval" in envelope.internal_goal
    assert runner.face_lobe_chat.calls[0]["message"] == query
    assert response["dispatched_job"]["job_id"] == envelope.job_id
    assert response["api_calls"] == 1


def test_natural_deterministic_gated_result_uses_gated_toolset_and_warm_face_model(
    tmp_path,
    monkeypatch,
):
    query = "Oracle, please enable menta_human_bridge."
    canonical_tool = "hivemind.services.enable@v1"
    outcome = _exact_gated_depth_outcome(query, canonical_tool=canonical_tool)
    outcome["resolution"]["deterministic_arguments"] = {
        "service_name": "menta_human_bridge"
    }
    _patch_strict_gated_catalog(
        monkeypatch,
        canonical_tool=canonical_tool,
    )

    runner, response, submitted, depth_picker_calls = _capture_depth_envelope(
        tmp_path,
        monkeypatch,
        query=query,
        outcome=outcome,
        route_kind="deep",
    )

    assert len(submitted) == 1
    envelope = submitted[0]
    assert envelope.resource_request.enabled_toolsets == [
        "mcp-hivemind-exact-gated"
    ]
    assert envelope.resource_request.model_override == "face-model"
    assert depth_picker_calls == []
    assert response["depth_lobe_model"] == {
        "schema": "Ms4DepthModel.v1",
        "model_id": "face-model",
        "source": "foreground_reuse_exact_gated",
        "detail": "strict exact-gated reused selected foreground model",
    }
    assert query in envelope.internal_goal
    assert (
        'Validated deterministic arguments: {"service_name": '
        '"menta_human_bridge"}'
    ) in envelope.internal_goal
    assert "Do not rediscover or alter these arguments" in envelope.internal_goal
    assert runner.face_lobe_chat.calls[0]["message"] == query
    assert envelope.user_visible_goal == query


def test_real_cascade_natural_service_result_feeds_exact_gated_directive(
    tmp_path,
    monkeypatch,
):
    from machine_spirit_4.gateway.quartermaster import (
        BACKEND_TFIDF,
        ToolRouter,
        build_index,
    )

    query = "Enable the menta_human_bridge service."
    canonical_tool = "hivemind.services.enable@v1"
    catalog = _patch_strict_gated_catalog(
        monkeypatch,
        canonical_tool=canonical_tool,
    )
    index = build_index(catalog, backend=BACKEND_TFIDF)
    decision = ToolRouter(
        ethics_evaluator=lambda _intent: {"allowed": True},
        audit=False,
    ).decide(query, catalog=catalog, index=index)
    outcome = decision.to_dict()

    assert outcome["verdict"] == "depth"
    assert outcome["resolution"]["tier"] == "deterministic"
    assert outcome["resolution"]["deterministic_arguments"] == {
        "service_name": "menta_human_bridge"
    }

    _runner_instance, _response, submitted, _picker_calls = _capture_depth_envelope(
        tmp_path,
        monkeypatch,
        query=query,
        outcome=outcome,
        route_kind="deep",
    )

    assert len(submitted) == 1
    envelope = submitted[0]
    assert envelope.resource_request.enabled_toolsets == [
        "mcp-hivemind-exact-gated"
    ]
    assert '"service_name": "menta_human_bridge"' in envelope.internal_goal


def test_natural_embeddings_gated_result_is_refused_without_broad_fallback(
    tmp_path,
    monkeypatch,
):
    query = "Enable the menta_human_bridge service."
    canonical_tool = "hivemind.services.enable@v1"
    outcome = _exact_gated_depth_outcome(query, canonical_tool=canonical_tool)
    outcome["resolution"]["tier"] = "embeddings"
    outcome["resolution"]["confidence"] = 0.473832
    outcome["resolution"]["deterministic_arguments"] = {
        "service_name": "menta_human_bridge"
    }
    _patch_strict_gated_catalog(
        monkeypatch,
        canonical_tool=canonical_tool,
    )

    _runner_instance, response, submitted, depth_picker_calls = _capture_depth_envelope(
        tmp_path,
        monkeypatch,
        query=query,
        outcome=outcome,
        route_kind="deep",
    )

    assert submitted == []
    assert depth_picker_calls == []
    assert response["dispatched_job"] is None
    assert response["depth_lobe_model"] is None
    assert response["api_calls"] == 0
    assert response["metrics"]["block_reason"] == "admission_refused"


def test_natural_strict_gated_explicit_depth_override_wins(
    tmp_path,
    monkeypatch,
):
    query = "Enable the menta_human_bridge service."
    canonical_tool = "hivemind.services.enable@v1"
    outcome = _exact_gated_depth_outcome(query, canonical_tool=canonical_tool)
    outcome["resolution"]["deterministic_arguments"] = {
        "service_name": "menta_human_bridge"
    }
    _patch_strict_gated_catalog(
        monkeypatch,
        canonical_tool=canonical_tool,
    )

    _runner_instance, response, submitted, depth_picker_calls = _capture_depth_envelope(
        tmp_path,
        monkeypatch,
        query=query,
        outcome=outcome,
        route_kind="deep",
        depth_model="llama3.1:8b",
    )

    assert submitted[0].resource_request.model_override == "llama3.1:8b"
    assert depth_picker_calls[0]["envelope_override"] == "llama3.1:8b"
    assert response["depth_lobe_model"]["source"] == "envelope_override"


def test_natural_strict_gated_invalid_face_model_falls_back_to_picker(
    tmp_path,
    monkeypatch,
):
    query = "Restart the menta_human_bridge service."
    canonical_tool = "hivemind.services.restart@v1"
    outcome = _exact_gated_depth_outcome(query, canonical_tool=canonical_tool)
    outcome["resolution"]["deterministic_arguments"] = {
        "service_name": "menta_human_bridge"
    }
    _patch_strict_gated_catalog(
        monkeypatch,
        canonical_tool=canonical_tool,
    )

    _runner_instance, response, submitted, depth_picker_calls = _capture_depth_envelope(
        tmp_path,
        monkeypatch,
        query=query,
        outcome=outcome,
        route_kind="deep",
        face_model="",
    )

    assert submitted[0].resource_request.model_override == "picked-depth-model"
    assert len(depth_picker_calls) == 1
    assert response["depth_lobe_model"]["source"] == "loaded"


@pytest.mark.parametrize(
    "case",
    ["missing", "extra", "mismatched", "malformed"],
)
def test_natural_gated_invalid_deterministic_arguments_block_without_submit(
    tmp_path,
    monkeypatch,
    case,
):
    query = "Enable the menta_human_bridge service."
    canonical_tool = "hivemind.services.enable@v1"
    outcome = _exact_gated_depth_outcome(query, canonical_tool=canonical_tool)
    if case == "extra":
        outcome["resolution"]["deterministic_arguments"] = {
            "service_name": "menta_human_bridge",
            "other": "value",
        }
    elif case == "mismatched":
        outcome["resolution"]["deterministic_arguments"] = {
            "service_name": "menta_oracle"
        }
    elif case == "malformed":
        outcome["resolution"]["deterministic_arguments"] = {
            "service_name": ["menta_human_bridge"]
        }
    _patch_strict_gated_catalog(
        monkeypatch,
        canonical_tool=canonical_tool,
    )

    runner, response, submitted, _picker_calls = _capture_depth_envelope(
        tmp_path,
        monkeypatch,
        query=query,
        outcome=outcome,
        route_kind="deep",
    )

    assert submitted == []
    assert response["dispatched_job"] is None
    assert response["api_calls"] == 0
    assert "did not dispatch" in response["text"].lower()
    assert "gated" in response["text"].lower()
    assert runner.face_lobe_chat.calls == []


@pytest.mark.parametrize(
    "case",
    ["catalog-unavailable", "malformed-catalog", "non-gated-proof"],
)
def test_gated_candidate_proof_failure_blocks_without_broad_fallback(
    tmp_path,
    monkeypatch,
    case,
):
    from machine_spirit_4.gateway.quartermaster import catalog as catalog_module

    query = "Enable the menta_human_bridge service."
    canonical_tool = "hivemind.services.enable@v1"
    outcome = _exact_gated_depth_outcome(query, canonical_tool=canonical_tool)
    outcome["resolution"]["deterministic_arguments"] = {
        "service_name": "menta_human_bridge"
    }
    _patch_strict_gated_catalog(
        monkeypatch,
        canonical_tool=canonical_tool,
        gated_by=() if case == "non-gated-proof" else ("operator-level",),
    )
    if case == "catalog-unavailable":
        monkeypatch.setattr(
            catalog_module,
            "get_catalog",
            lambda _url: (_ for _ in ()).throw(RuntimeError("catalog offline")),
        )
    elif case == "malformed-catalog":
        monkeypatch.setattr(
            catalog_module,
            "get_catalog",
            lambda _url: object(),
        )

    runner, response, submitted, _picker_calls = _capture_depth_envelope(
        tmp_path,
        monkeypatch,
        query=query,
        outcome=outcome,
        route_kind="deep",
    )

    assert submitted == []
    assert response["dispatched_job"] is None
    assert response["api_calls"] == 0
    assert "did not dispatch" in response["text"].lower()
    assert runner.face_lobe_chat.calls == []


def test_real_deep_service_conflict_vetoes_all_submission(tmp_path, monkeypatch):
    from machine_spirit_4.gateway.quartermaster import ToolRouter
    from machine_spirit_4.gateway.quartermaster import router as qm_router

    _patch_strict_gated_catalog(
        monkeypatch,
        canonical_tool="hivemind.services.enable@v1",
    )
    monkeypatch.setattr(
        qm_router,
        "_DEFAULT_ROUTER",
        ToolRouter(ethics_evaluator=lambda _intent: {"allowed": True}, audit=False),
    )
    runner = _runner(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "_try_cluster_time", lambda: None)
    monkeypatch.setattr(
        "machine_spirit_4.gateway.hermes_runner.face_lobe_turn_start",
        lambda **_kwargs: {"revision": {"revision_id": 15}, "marked_stale": []},
    )
    picker_calls: list[dict[str, Any]] = []

    def choose_depth(**kwargs):
        picker_calls.append(kwargs)
        return DepthChoice(
            model_id="picked-depth-model",
            source="loaded",
            detail="test picker",
        )

    monkeypatch.setattr(
        "machine_spirit_4.gateway.hermes_runner.choose_depth_model",
        choose_depth,
    )
    submitted: list[Any] = []

    class _CapturingRunner:
        def submit(self, envelope):
            submitted.append(envelope)
            return {"job_id": envelope.job_id, "state": "queued"}

    monkeypatch.setattr(
        "machine_spirit_4.gateway.hermes_runner.default_runner",
        lambda: _CapturingRunner(),
    )
    query = (
        "/deep Enable menta_human_bridge --tool "
        "hivemind.services.disable@v1"
    )

    response = runner.chat(query, session_id="deep-conflict", model="face-model")

    assert response["router"]["kind"] == "deep"
    assert response["quartermaster"]["verdict"] == "none"
    assert response["quartermaster"]["reason"] == (
        "natural service action conflicts with explicit canonical tool"
    )
    assert response["quartermaster"]["resolution"]["fallback_reason"] == (
        "natural service action conflicts with explicit canonical tool"
    )
    assert picker_calls == []
    assert submitted == []
    assert response["dispatched_job"] is None
    assert response["depth_lobe_model"] is None
    assert response["api_calls"] == 0
    assert "did not dispatch" in response["text"].lower()
    assert "no background job" in response["text"].lower()
    assert runner.face_lobe_chat.calls == []


def test_real_unrelated_deep_prompt_is_refused_without_broad_fallback(tmp_path, monkeypatch):
    from machine_spirit_4.gateway.quartermaster import ToolRouter
    from machine_spirit_4.gateway.quartermaster import router as qm_router

    _patch_strict_gated_catalog(
        monkeypatch,
        canonical_tool="hivemind.services.enable@v1",
    )
    monkeypatch.setattr(
        qm_router,
        "_DEFAULT_ROUTER",
        ToolRouter(ethics_evaluator=lambda _intent: {"allowed": True}, audit=False),
    )
    runner = _runner(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "_try_cluster_time", lambda: None)
    monkeypatch.setattr(
        "machine_spirit_4.gateway.hermes_runner.face_lobe_turn_start",
        lambda **_kwargs: {"revision": {"revision_id": 16}, "marked_stale": []},
    )
    monkeypatch.setattr(
        "machine_spirit_4.gateway.hermes_runner.choose_depth_model",
        lambda **_kwargs: DepthChoice(
            model_id="picked-depth-model",
            source="loaded",
            detail="test picker",
        ),
    )
    submitted: list[Any] = []

    class _CapturingRunner:
        def submit(self, envelope):
            submitted.append(envelope)
            return {"job_id": envelope.job_id, "state": "queued"}

    monkeypatch.setattr(
        "machine_spirit_4.gateway.hermes_runner.default_runner",
        lambda: _CapturingRunner(),
    )
    query = "/deep Investigate this unrelated open-ended cluster issue."

    response = runner.chat(query, session_id="deep-control", model="face-model")

    assert response["router"]["kind"] == "deep"
    assert submitted == []
    assert response["dispatched_job"] is None
    assert response["depth_lobe_model"] is None
    assert response["api_calls"] == 0
    assert response["metrics"]["block_reason"] == "admission_refused"
    assert runner.face_lobe_chat.calls == []


@pytest.mark.parametrize(
    "case",
    [
        "read-only",
        "multiple-resolved",
        "multiple-explicit",
        "conflicting-explicit",
        "unknown",
        "inline-verdict",
        "non-gated",
        "dry-run-only",
        "malformed",
        "mismatched",
    ],
)
def test_non_strict_outcomes_do_not_activate_exact_gated_bridge(
    monkeypatch,
    case,
):
    canonical_tool = "hivemind.services.enable@v1"
    query = f"Use {canonical_tool} to enable menta_human_bridge."
    outcome = _exact_gated_depth_outcome(query, canonical_tool=canonical_tool)
    _patch_strict_gated_catalog(
        monkeypatch,
        canonical_tool=canonical_tool,
    )

    if case == "read-only":
        canonical_tool = "hivemind.app.get@v1"
        query = f"Use {canonical_tool} to read app-7."
        outcome = _exact_read_depth_outcome(query, canonical_tool=canonical_tool)
        _patch_persisted_continuation_catalog(
            monkeypatch,
            {"name": canonical_tool, "required": ("id",)},
        )
    elif case == "multiple-resolved":
        outcome["resolution"]["tools"].append({
            "name": "hivemind.services.disable@v1",
            "toolbox": "services",
            "cluster": "operations",
            "score": 0.9,
            "kind": "hivemind_native",
            "inline_eligible": False,
        })
    elif case == "multiple-explicit":
        query = (
            "Use hivemind.services.enable@v1 and "
            "hivemind.services.disable@v1."
        )
        outcome["query"] = query
        outcome["resolution"]["query"] = query
    elif case == "conflicting-explicit":
        query = (
            "Enable menta_human_bridge with "
            "hivemind.services.disable@v1."
        )
        outcome["query"] = query
        outcome["resolution"]["query"] = query
    elif case == "unknown":
        _patch_strict_gated_catalog(
            monkeypatch,
            canonical_tool=canonical_tool,
            include_native=False,
        )
    elif case == "inline-verdict":
        outcome["verdict"] = "inline"
        outcome["inline_tool"] = canonical_tool
    elif case == "non-gated":
        _patch_strict_gated_catalog(
            monkeypatch,
            canonical_tool=canonical_tool,
            gated_by=(),
        )
    elif case == "dry-run-only":
        _patch_strict_gated_catalog(
            monkeypatch,
            canonical_tool=canonical_tool,
            proxy_kind="dry_run_only",
        )
    elif case == "malformed":
        outcome["resolution"].pop("schema")
    elif case == "mismatched":
        outcome["query"] = "another turn"

    assert hermes_runner_module._strict_exact_gated_canonical(
        query,
        outcome,
        inline_block=None,
        hivemind_url="http://hive:6089",
    ) is None


def test_cancelled_strict_gated_turn_submits_no_job(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch)
    _patch_direct_route(monkeypatch)
    monkeypatch.setattr(runner, "_try_cluster_time", lambda: None)
    monkeypatch.setattr(
        "machine_spirit_4.gateway.hermes_runner.face_lobe_turn_start",
        lambda **_kwargs: {"revision": {"revision_id": 13}, "marked_stale": []},
    )
    canonical_tool = "hivemind.services.enable@v1"
    query = "Enable the menta_human_bridge service."
    _patch_strict_gated_catalog(
        monkeypatch,
        canonical_tool=canonical_tool,
    )
    cancel_event = threading.Event()

    def cancel_with_gated_outcome(message):
        cancel_event.set()
        outcome = _exact_gated_depth_outcome(
            message,
            canonical_tool=canonical_tool,
        )
        outcome["resolution"]["deterministic_arguments"] = {
            "service_name": "menta_human_bridge"
        }
        return None, outcome

    monkeypatch.setattr(
        runner,
        "_try_quartermaster_inline",
        cancel_with_gated_outcome,
    )
    submitted: list[Any] = []

    class _CapturingRunner:
        def submit(self, envelope):
            submitted.append(envelope)
            return {"job_id": envelope.job_id, "state": "queued"}

    monkeypatch.setattr(
        "machine_spirit_4.gateway.hermes_runner.default_runner",
        lambda: _CapturingRunner(),
    )

    with pytest.raises(Ms4TurnCancelled, match="after Quartermaster routing"):
        runner.chat(
            query,
            session_id="cancel-gated",
            model="face-model",
            cancel_event=cancel_event,
        )

    assert submitted == []
    assert runner.face_lobe_chat.calls == []


def test_final_pre_submit_cancellation_seam_submits_no_job(
    tmp_path,
    monkeypatch,
):
    runner = _runner(tmp_path, monkeypatch)
    _patch_deep_route(monkeypatch)
    monkeypatch.setattr(runner, "_try_cluster_time", lambda: None)
    query = (
        "Explain why a 35B Depth lobe might outperform a 4B Face lobe. "
        "Give the recommendation, mechanism, tradeoffs, and a concrete example."
    )
    monkeypatch.setattr(
        runner,
        "_try_quartermaster_inline",
        lambda message: (None, _valid_none_outcome(message)),
    )
    revisions: list[dict[str, Any]] = []

    def start_turn(**kwargs):
        revisions.append(kwargs)
        return {"revision": {"revision_id": 14}, "marked_stale": []}

    monkeypatch.setattr(
        "machine_spirit_4.gateway.hermes_runner.face_lobe_turn_start",
        start_turn,
    )
    monkeypatch.setattr(
        "machine_spirit_4.gateway.hermes_runner.choose_depth_model",
        lambda **_kwargs: DepthChoice(
            model_id="picked-depth-model",
            source="loaded",
            detail="test picker",
        ),
    )
    cancel_event = threading.Event()
    submitted: list[Any] = []

    class _CapturingRunner:
        def submit(self, envelope):
            submitted.append(envelope)
            return {"job_id": envelope.job_id, "state": "queued"}

    capturing_runner = _CapturingRunner()

    def cancel_at_dispatch_runner_lookup():
        cancel_event.set()
        return capturing_runner

    monkeypatch.setattr(
        "machine_spirit_4.gateway.hermes_runner.default_runner",
        cancel_at_dispatch_runner_lookup,
    )

    with pytest.raises(Ms4TurnCancelled, match="final Depth dispatch"):
        runner.chat(
            query,
            session_id="final-cancel",
            model="face-model",
            cancel_event=cancel_event,
        )

    assert len(revisions) == 1
    assert submitted == []
    assert runner.face_lobe_chat.calls == []


def test_persisted_missing_id_reply_resumes_one_exact_read_with_literal_face_text(
    tmp_path,
    monkeypatch,
):
    board = Blackboard(tmp_path / "continuation.sqlite3")
    conversation_id = "same-session-continuation"
    canonical_tool = "hivemind.app.get@v1"
    source_job_id = _seed_persisted_exact_read_missing_job(
        board,
        conversation_id=conversation_id,
        canonical_tool=canonical_tool,
    )
    _patch_persisted_continuation_catalog(
        monkeypatch,
        {"name": canonical_tool, "required": ("id",)},
    )
    persisted_runner = _PersistedCapturingRunner(board)
    value = "ed5d7377-7daf-4cbf-aba2-aab40e74dc0b"
    literal_message = f"The ID is {value}."
    runner, quartermaster_calls = _configure_persisted_continuation_turn(
        tmp_path,
        monkeypatch,
        persisted_runner=persisted_runner,
        literal_message=literal_message,
        expected_canonical=canonical_tool,
    )

    response = runner.chat(
        literal_message,
        session_id=conversation_id,
        model="face-model",
    )

    assert len(persisted_runner.submitted) == 1
    envelope = persisted_runner.submitted[0]
    synthetic_intent = (
        f"Use {canonical_tool} with id={value} to complete the prior exact read."
    )
    assert quartermaster_calls == [synthetic_intent]
    assert envelope.resource_request.enabled_toolsets == [
        "mcp-hivemind-exact-read"
    ]
    assert envelope.resource_request.model_override == "face-model"
    assert synthetic_intent in envelope.internal_goal
    assert envelope.user_visible_goal == literal_message
    assert runner.face_lobe_chat.calls[0]["message"] == literal_message
    assert runner.face_lobe_chat._sessions[conversation_id].messages[0] == {
        "role": "user",
        "content": literal_message,
    }
    assert response["depth_lobe_model"]["source"] == "foreground_reuse_exact_read"
    links = _persisted_continuation_links(board, source_job_id)
    assert len(links) == 1
    assert links[0]["visibility"] == "operator_only"
    assert links[0]["payload"] == {
        "source": "hermes_runner_exact_read_continuation",
        "source_job_id": source_job_id,
        "continuation_job_id": envelope.job_id,
        "missing_key": "id",
    }


@pytest.mark.parametrize(
    ("literal_message", "expected_value"),
    [
        pytest.param("ID=app-7", "app-7", id="equals"),
        pytest.param("id: app-7.", "app-7", id="colon"),
        pytest.param(
            "ed5d7377-7daf-4cbf-aba2-aab40e74dc0b",
            "ed5d7377-7daf-4cbf-aba2-aab40e74dc0b",
            id="bare-uuid",
        ),
        pytest.param("id=0", "0", id="numeric-zero"),
        pytest.param("id is false", "false", id="boolean-looking-false"),
    ],
)
def test_persisted_exact_read_continuation_accepts_only_bounded_key_forms(
    tmp_path,
    monkeypatch,
    literal_message,
    expected_value,
):
    board = Blackboard(tmp_path / f"continuation-{expected_value}.sqlite3")
    conversation_id = "bounded-key-forms"
    canonical_tool = "hivemind.app.get@v1"
    _seed_persisted_exact_read_missing_job(
        board,
        conversation_id=conversation_id,
        canonical_tool=canonical_tool,
    )
    _patch_persisted_continuation_catalog(
        monkeypatch,
        {"name": canonical_tool, "required": ("id",)},
    )
    persisted_runner = _PersistedCapturingRunner(board)
    runner, quartermaster_calls = _configure_persisted_continuation_turn(
        tmp_path,
        monkeypatch,
        persisted_runner=persisted_runner,
        literal_message=literal_message,
        expected_canonical=canonical_tool,
    )

    runner.chat(literal_message, session_id=conversation_id, model="face-model")

    assert len(persisted_runner.submitted) == 1
    assert quartermaster_calls == [
        f"Use {canonical_tool} with id={expected_value} "
        "to complete the prior exact read."
    ]


@pytest.mark.parametrize(
    "case",
    [
        "multiple-missing-keys",
        "zero-missing-keys",
        "unknown-canonical",
        "mutating-canonical",
        "runtime-action",
        "confirmation-gated",
        "malformed-evidence",
        "unverified-evidence",
        "ambiguous-evidence",
        "missing-canonical",
        "failed-result",
        "no-result",
        "running-source",
        "failed-source",
        "canceled-source",
        "wrong-toolset",
        "cross-conversation",
        "no-candidate",
        "topic-change",
        "extra-prose",
        "multiple-assignments",
        "json-value",
        "another-canonical",
    ],
)
def test_persisted_exact_read_continuation_rejections_have_zero_effects(
    tmp_path,
    monkeypatch,
    case,
):
    board = Blackboard(tmp_path / f"continuation-reject-{case}.sqlite3")
    conversation_id = "continuation-rejections"
    canonical_tool = "hivemind.app.get@v1"
    missing_keys = ("id",)
    state = "completed"
    toolsets = None
    result_status = "success"
    evidence_mode = "verified"
    source_conversation = conversation_id
    literal_message = "id=app-7"
    catalog_specs: list[dict[str, Any]] = [
        {"name": canonical_tool, "required": ("id",)}
    ]
    seed_source = True

    if case == "multiple-missing-keys":
        missing_keys = ("id", "zone")
    elif case == "zero-missing-keys":
        missing_keys = ()
    elif case == "unknown-canonical":
        catalog_specs = []
    elif case == "mutating-canonical":
        canonical_tool = "hivemind.jobs.cancel@v1"
        missing_keys = ("job_id",)
        literal_message = "job_id=job-7"
        catalog_specs = [{"name": canonical_tool, "required": ("job_id",)}]
    elif case == "runtime-action":
        catalog_specs[0]["kind"] = "runtime_action"
    elif case == "confirmation-gated":
        catalog_specs[0]["gated_by"] = ("confirm_operator",)
    elif case == "malformed-evidence":
        evidence_mode = "malformed"
    elif case == "unverified-evidence":
        evidence_mode = "unverified"
    elif case == "ambiguous-evidence":
        evidence_mode = "ambiguous"
    elif case == "missing-canonical":
        evidence_mode = "missing-canonical"
    elif case == "failed-result":
        result_status = "failed"
    elif case == "no-result":
        evidence_mode = "no-result"
    elif case == "running-source":
        state = "running"
    elif case == "failed-source":
        state = "failed"
    elif case == "canceled-source":
        state = "canceled"
    elif case == "wrong-toolset":
        toolsets = ["mcp-hivemind"]
    elif case == "cross-conversation":
        source_conversation = "different-conversation"
    elif case == "no-candidate":
        seed_source = False
    elif case == "topic-change":
        literal_message = "What is the weather?"
    elif case == "extra-prose":
        literal_message = "The ID is app-7. Also list the VMs."
    elif case == "multiple-assignments":
        literal_message = "id=app-7 id=app-8"
    elif case == "json-value":
        literal_message = 'id={"value":"app-7"}'
    elif case == "another-canonical":
        literal_message = "id=hivemind.vm.get@v1"

    source_job_id = None
    if seed_source:
        source_job_id = _seed_persisted_exact_read_missing_job(
            board,
            conversation_id=source_conversation,
            canonical_tool=canonical_tool,
            missing_keys=missing_keys,
            state=state,
            toolsets=toolsets,
            result_status=result_status,
            evidence_mode=evidence_mode,
        )
    _patch_persisted_continuation_catalog(monkeypatch, *catalog_specs)
    persisted_runner = _PersistedCapturingRunner(board)
    runner, quartermaster_calls = _configure_persisted_continuation_turn(
        tmp_path,
        monkeypatch,
        persisted_runner=persisted_runner,
        literal_message=literal_message,
        expected_canonical=canonical_tool,
        route_kind="deep",
    )

    response = runner.chat(
        literal_message,
        session_id=conversation_id,
        model="face-model",
    )

    assert quartermaster_calls == [literal_message]
    assert persisted_runner.submitted == []
    assert response["dispatched_job"] is None
    assert response["api_calls"] == 0
    assert response["metrics"]["block_reason"] == "admission_refused"
    assert runner.face_lobe_chat.calls == []
    if source_job_id is not None:
        assert _persisted_continuation_links(board, source_job_id) == []


def test_most_recent_eligible_persisted_exact_read_source_wins(
    tmp_path,
    monkeypatch,
):
    board = Blackboard(tmp_path / "continuation-most-recent.sqlite3")
    conversation_id = "continuation-most-recent"
    older_canonical = "hivemind.app.get@v1"
    newer_canonical = "hivemind.app.status@v1"
    older_job_id = _seed_persisted_exact_read_missing_job(
        board,
        conversation_id=conversation_id,
        canonical_tool=older_canonical,
    )
    time.sleep(0.002)
    newer_job_id = _seed_persisted_exact_read_missing_job(
        board,
        conversation_id=conversation_id,
        canonical_tool=newer_canonical,
    )
    _patch_persisted_continuation_catalog(
        monkeypatch,
        {"name": older_canonical, "required": ("id",)},
        {"name": newer_canonical, "required": ("id",)},
    )
    persisted_runner = _PersistedCapturingRunner(board)
    literal_message = "id=app-7"
    runner, quartermaster_calls = _configure_persisted_continuation_turn(
        tmp_path,
        monkeypatch,
        persisted_runner=persisted_runner,
        literal_message=literal_message,
        expected_canonical=newer_canonical,
    )

    runner.chat(literal_message, session_id=conversation_id, model="face-model")

    assert quartermaster_calls == [
        f"Use {newer_canonical} with id=app-7 to complete the prior exact read."
    ]
    assert _persisted_continuation_links(board, older_job_id) == []
    assert len(_persisted_continuation_links(board, newer_job_id)) == 1


def test_persisted_exact_read_source_is_consumed_across_fresh_runner_view(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "continuation-consumed.sqlite3"
    board = Blackboard(db_path)
    conversation_id = "continuation-consumed"
    canonical_tool = "hivemind.app.get@v1"
    source_job_id = _seed_persisted_exact_read_missing_job(
        board,
        conversation_id=conversation_id,
        canonical_tool=canonical_tool,
    )
    _patch_persisted_continuation_catalog(
        monkeypatch,
        {"name": canonical_tool, "required": ("id",)},
    )
    first_persisted_runner = _PersistedCapturingRunner(board)
    first_literal = "id=app-7"
    first_runner, first_qm_calls = _configure_persisted_continuation_turn(
        tmp_path,
        monkeypatch,
        persisted_runner=first_persisted_runner,
        literal_message=first_literal,
        expected_canonical=canonical_tool,
    )

    first_runner.chat(first_literal, session_id=conversation_id, model="face-model")

    assert len(first_persisted_runner.submitted) == 1
    assert "complete the prior exact read" in first_qm_calls[0]
    assert len(_persisted_continuation_links(board, source_job_id)) == 1

    fresh_board = Blackboard(db_path)
    fresh_persisted_runner = _PersistedCapturingRunner(fresh_board)
    second_literal = "id=app-8"
    second_runner, second_qm_calls = _configure_persisted_continuation_turn(
        tmp_path,
        monkeypatch,
        persisted_runner=fresh_persisted_runner,
        literal_message=second_literal,
        expected_canonical=canonical_tool,
        route_kind="deep",
    )

    second_response = second_runner.chat(
        second_literal,
        session_id=conversation_id,
        model="face-model",
    )

    assert second_qm_calls == [second_literal]
    assert fresh_persisted_runner.submitted == []
    assert second_response["dispatched_job"] is None
    assert second_response["api_calls"] == 0
    assert second_response["metrics"]["block_reason"] == "admission_refused"
    assert second_runner.face_lobe_chat.calls == []
    assert len(_persisted_continuation_links(fresh_board, source_job_id)) == 1


def test_cancelled_persisted_exact_read_continuation_submits_and_consumes_nothing(
    tmp_path,
    monkeypatch,
):
    board = Blackboard(tmp_path / "continuation-cancel.sqlite3")
    conversation_id = "continuation-cancel"
    canonical_tool = "hivemind.app.get@v1"
    source_job_id = _seed_persisted_exact_read_missing_job(
        board,
        conversation_id=conversation_id,
        canonical_tool=canonical_tool,
    )
    _patch_persisted_continuation_catalog(
        monkeypatch,
        {"name": canonical_tool, "required": ("id",)},
    )
    persisted_runner = _PersistedCapturingRunner(board)
    literal_message = "id=app-7"
    cancel_event = threading.Event()
    runner, _quartermaster_calls = _configure_persisted_continuation_turn(
        tmp_path,
        monkeypatch,
        persisted_runner=persisted_runner,
        literal_message=literal_message,
        expected_canonical=canonical_tool,
        cancel_event=cancel_event,
    )

    with pytest.raises(Ms4TurnCancelled, match="after Quartermaster routing"):
        runner.chat(
            literal_message,
            session_id=conversation_id,
            model="face-model",
            cancel_event=cancel_event,
        )

    assert persisted_runner.submitted == []
    assert _persisted_continuation_links(board, source_job_id) == []


def test_exact_read_internal_goal_bounds_long_original_without_losing_directive():
    canonical_tool = "hivemind.app.get@v1"
    original_goal = (
        f"Oracle, use {canonical_tool} to inspect the registered app. "
        + ("detail " * 400)
    )

    internal_goal = hermes_runner_module._exact_read_internal_goal(
        original_goal,
        canonical_tool,
    )

    assert len(internal_goal) == INTERNAL_GOAL_MAX_CHARS
    assert internal_goal.startswith("[MS4 TRUSTED EXACT-READ DIRECTIVE]")
    assert f"`{canonical_tool}`" in internal_goal
    assert "`missing_required_keys`" in internal_goal
    assert "Oracle, use hivemind.app.get@v1" in internal_goal


def test_exact_read_match_is_case_insensitive_with_safe_token_boundaries():
    query = "Oracle, use HIVEMIND.APP.GET@V1, please."
    outcome = _exact_read_depth_outcome(query)

    assert hermes_runner_module._strict_exact_read_canonical(
        query,
        outcome,
        inline_block=None,
    ) == "hivemind.app.get@v1"


@pytest.mark.parametrize(
    "case",
    [
        "unknown-zero",
        "zero-depth",
        "multiple-resolved",
        "multiple-explicit",
        "fuzzy-only",
        "embedded-leading-token",
        "embedded-trailing-token",
        "runtime-action",
        "dry-run-only",
        "confirmation-gated",
        "non-inline",
        "none-verdict",
        "inline-verdict",
        "inline-result",
        "inline-block",
        "malformed",
        "mismatched-decision",
        "mismatched-resolution",
    ],
)
def test_non_strict_quartermaster_outcomes_do_not_activate_exact_read_bridge(case):
    canonical_tool = "hivemind.app.get@v1"
    query = f"Oracle, use {canonical_tool} to show one app."
    outcome = _exact_read_depth_outcome(query, canonical_tool=canonical_tool)
    inline_block = None

    if case == "unknown-zero":
        query = "Oracle, use hivemind.app.lookup_magic@v1 to show one app."
        outcome = _valid_none_outcome(query)
    elif case == "zero-depth":
        outcome["resolution"]["tools"] = []
    elif case == "multiple-resolved":
        outcome["resolution"]["tools"].append({
            "name": "hivemind.app.list@v1",
            "toolbox": "apps",
            "cluster": "applications",
            "score": 0.9,
            "kind": "hivemind_native",
            "inline_eligible": True,
        })
    elif case == "multiple-explicit":
        query = (
            f"Use {canonical_tool} and hivemind.app.list@v1 "
            "to show registered apps."
        )
        outcome["query"] = query
        outcome["resolution"]["query"] = query
    elif case == "fuzzy-only":
        query = "Show me one registered app."
        outcome["query"] = query
        outcome["resolution"]["query"] = query
    elif case == "embedded-leading-token":
        query = f"Use prefix{canonical_tool} to show one app."
        outcome["query"] = query
        outcome["resolution"]["query"] = query
    elif case == "embedded-trailing-token":
        query = f"Use {canonical_tool}.shadow to show one app."
        outcome["query"] = query
        outcome["resolution"]["query"] = query
    elif case in {"runtime-action", "dry-run-only"}:
        outcome["resolution"]["tools"][0]["kind"] = case.replace("-", "_")
    elif case == "confirmation-gated":
        outcome["resolution"]["tools"][0]["inline_eligible"] = False
        outcome["resolution"]["tools"][0]["gated_by"] = ["confirm_operator"]
    elif case == "non-inline":
        outcome["resolution"]["tools"][0]["inline_eligible"] = False
    elif case == "none-verdict":
        outcome["verdict"] = "none"
    elif case == "inline-verdict":
        outcome["verdict"] = "inline"
        outcome["inline_tool"] = canonical_tool
    elif case == "inline-result":
        outcome["inline_executed"] = True
        outcome["inline_result"] = {"id": "app-7"}
    elif case == "inline-block":
        inline_block = "authoritative inline result"
    elif case == "malformed":
        outcome["resolution"].pop("schema")
    elif case == "mismatched-decision":
        outcome["query"] = "another turn"
    elif case == "mismatched-resolution":
        outcome["resolution"]["query"] = "another turn"

    assert hermes_runner_module._strict_exact_read_canonical(
        query,
        outcome,
        inline_block=inline_block,
    ) is None


def test_normal_broad_depth_request_is_refused_without_default_toolsets(
    tmp_path,
    monkeypatch,
):
    query = "investigate this open-ended issue"
    outcome = _valid_none_outcome(query)

    _runner_instance, response, submitted, depth_picker_calls = _capture_depth_envelope(
        tmp_path,
        monkeypatch,
        query=query,
        outcome=outcome,
        route_kind="deep",
    )

    assert submitted == []
    assert depth_picker_calls == []
    assert response["dispatched_job"] is None
    assert response["depth_lobe_model"] is None
    assert response["api_calls"] == 0
    assert response["metrics"]["block_reason"] == "admission_refused"


# ---------------------------------------------------------------------------
# _try_quartermaster_inline end-to-end against a fake MCP server
# ---------------------------------------------------------------------------


class _FakeCluster:
    """Answers tools/list (for the catalog) and tools/call (for inline
    execution). Lets us drive the whole inline path with no real cluster."""

    def __init__(self):
        self.tools: list[dict[str, Any]] = []
        self.call_results: dict[str, Any] = {}
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.port = 0

    def start(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a, **_k):
                return

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0") or "0")
                raw = self.rfile.read(length).decode("utf-8") if length else "{}"
                body = json.loads(raw)
                method = body.get("method")
                if method == "tools/list":
                    result = {"tools": outer.tools}
                else:  # tools/call
                    name = (body.get("params") or {}).get("name")
                    payload = outer.call_results.get(name, {"ok": True})
                    is_error = (
                        isinstance(payload, dict)
                        and payload.get("__fake_mcp_is_error__") is True
                    )
                    payload_text = (
                        str(payload.get("message") or "tool error")
                        if is_error
                        else json.dumps(payload)
                    )
                    result = {
                        "content": [{"type": "text", "text": payload_text}],
                        "isError": is_error,
                    }
                env = {"jsonrpc": "2.0", "id": body.get("id"), "result": result}
                wire = json.dumps(env).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(wire)))
                self.end_headers()
                self.wfile.write(wire)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]

    def stop(self):
        if self.server:
            try:
                self.server.shutdown()
            except Exception:
                pass


@pytest.fixture
def fake_cluster(monkeypatch):
    c = _FakeCluster()
    c.start()
    monkeypatch.setenv("MS4_HIVEMIND_MCP_URL", f"http://127.0.0.1:{c.port}/mcp")
    monkeypatch.setenv("MS4_QM_INLINE", "1")
    monkeypatch.delenv("MS4_HIVEMIND_API_KEY", raising=False)
    from machine_spirit_4.gateway.quartermaster import (
        reset_catalog_cache_for_tests,
        reset_index_cache_for_tests,
    )
    reset_catalog_cache_for_tests()
    reset_index_cache_for_tests()
    yield c
    c.stop()
    reset_catalog_cache_for_tests()
    reset_index_cache_for_tests()


@pytest.mark.parametrize(
    ("query", "expected_tool", "payload"),
    [
        (
            "active jobs",
            "hivemind.jobs.active@v1",
            {"total_active": 2, "summary": "2 jobs running"},
        ),
        (
            "list the cluster networks",
            "hivemind.network.list@v1",
            {"networks": [{"id": "cluster-net-1"}]},
        ),
        (
            "list my virtual machines",
            "hivemind.vm.list@v1",
            {"vms": [{"name": "oracle-dev", "running": True}]},
        ),
        (
            "Show current GPU availability.",
            "hivemind.gpu.availability@v1",
            {"free_gpus": 2},
        ),
        (
            "what is the grid status",
            "hivemind.grid.status@v1",
            {"healthy": True, "state": "ready"},
        ),
    ],
)
def test_direct_safe_reads_execute_inline_before_legacy_dispatch_gate(
    tmp_path,
    monkeypatch,
    fake_cluster,
    query,
    expected_tool,
    payload,
):
    fake_cluster.tools = [
        {
            "name": "hivemind.jobs.active@v1",
            "description": "Show all currently active jobs and their progress.",
            "inputSchema": {},
        },
        {
            "name": "hivemind.jobs.cancel@v1",
            "description": "Cancel an active job.",
            "inputSchema": {"required": ["job_id"]},
        },
        {
            "name": "hivemind.network.list@v1",
            "description": "List the cluster virtual networks and attachments.",
            "inputSchema": {},
        },
        {
            "name": "hivemind.network.delete@v1",
            "description": "Delete a cluster virtual network.",
            "inputSchema": {"required": ["id"]},
        },
        {
            "name": "hivemind.vm.list@v1",
            "description": "List configured virtual machines and running state.",
            "inputSchema": {},
        },
        {
            "name": "hivemind.vm.start@v1",
            "description": "Start a virtual machine.",
            "inputSchema": {"required": ["name"]},
        },
        {
            "name": "hivemind.gpu.availability@v1",
            "description": "Show current GPU availability for scheduling.",
            "inputSchema": {},
        },
        {
            "name": "hivemind.grid.status@v1",
            "description": "Report the current grid status.",
            "inputSchema": {},
        },
        {
            "name": "hivemind.grid.control@v1",
            "description": "Change grid control state.",
            "inputSchema": {"required": ["state"]},
        },
    ]
    fake_cluster.call_results[expected_tool] = payload
    url = f"http://127.0.0.1:{fake_cluster.port}"
    runner = _runner(tmp_path, monkeypatch, hivemind_url=url)
    from machine_spirit_4.gateway.quartermaster import ToolRouter
    from machine_spirit_4.gateway.quartermaster import router as qm_router
    monkeypatch.setattr(
        qm_router,
        "_DEFAULT_ROUTER",
        ToolRouter(ethics_evaluator=lambda _intent: {"allowed": True}),
    )
    monkeypatch.setattr(runner, "_try_cluster_time", lambda: None)
    monkeypatch.setattr(
        "machine_spirit_4.gateway.hermes_runner.build_grounded_user_message",
        lambda message, _url: (message, "none"),
    )

    resp = runner.chat(query, session_id=f"direct-{expected_tool}", model="m")

    assert resp["router"]["kind"] == "direct"
    assert resp["quartermaster_inline"] is True
    assert resp["quartermaster"]["inline_tool"] == expected_tool
    assert resp["quartermaster"]["inline_executed_tool"] == expected_tool
    assert expected_tool in resp["text"]
    assert json.dumps(payload, ensure_ascii=False) in resp["text"]
    assert resp["api_calls"] == 0
    assert resp["dispatched_job"] is None
    assert resp["depth_lobe_model"] is None
    assert runner.face_lobe_chat.calls == []
    assert len(runner.face_lobe_chat.authoritative_calls) == 1


def test_ambiguous_direct_mutation_depth_verdict_stays_in_face_chat(
    tmp_path,
    monkeypatch,
    fake_cluster,
):
    fake_cluster.tools = [
        {
            "name": "hivemind.jobs.active@v1",
            "description": "Show all currently active jobs.",
            "inputSchema": {},
        },
        {
            "name": "hivemind.jobs.cancel@v1",
            "description": "Cancel an active job.",
            "inputSchema": {"required": ["job_id"]},
        },
    ]
    fake_cluster.call_results["hivemind.jobs.cancel@v1"] = {
        "unexpected_mutation": True,
    }
    url = f"http://127.0.0.1:{fake_cluster.port}"
    runner = _runner(tmp_path, monkeypatch, hivemind_url=url)
    from machine_spirit_4.gateway.quartermaster import ToolRouter
    from machine_spirit_4.gateway.quartermaster import router as qm_router
    monkeypatch.setattr(
        qm_router,
        "_DEFAULT_ROUTER",
        ToolRouter(ethics_evaluator=lambda _intent: {"allowed": True}),
    )
    monkeypatch.setattr(runner, "_try_cluster_time", lambda: None)
    monkeypatch.setattr(
        "machine_spirit_4.gateway.hermes_runner.build_grounded_user_message",
        lambda message, _url: (message, "none"),
    )
    monkeypatch.setattr(
        "machine_spirit_4.gateway.hermes_runner.choose_depth_model",
        lambda **_kwargs: type("D", (), {
            "model_id": "depth-model",
            "to_dict": lambda self: {"model_id": "depth-model", "source": "test"},
        })(),
    )
    monkeypatch.setattr(
        "machine_spirit_4.gateway.hermes_runner.face_lobe_turn_start",
        lambda **_kwargs: {"revision": {"revision_id": 4}, "marked_stale": []},
    )
    submitted: list[Any] = []

    class _CapturingRunner:
        def submit(self, envelope):
            submitted.append(envelope)
            return {"job_id": envelope.job_id, "state": "queued"}

    monkeypatch.setattr(
        "machine_spirit_4.gateway.hermes_runner.default_runner",
        lambda: _CapturingRunner(),
    )
    query = "cancel active jobs"
    resp = runner.chat(query, session_id="direct-mutation", model="m")

    assert resp["router"]["kind"] == "direct"
    assert resp["quartermaster_inline"] is False
    assert resp["quartermaster"]["verdict"] == "depth"
    assert resp["quartermaster"]["schema"] == "Ms4ToolRouteDecision.v1"
    assert resp["quartermaster"]["inline_tool"] is None
    assert runner.face_lobe_chat.authoritative_calls == []
    assert len(runner.face_lobe_chat.calls) == 1
    assert resp["api_calls"] == 1
    assert submitted == []
    assert resp["dispatched_job"] is None
    assert resp["depth_lobe_model"] is None
    assert resp["text"] == f"reply:{query}"


def test_inline_end_to_end_readonly_query(tmp_path, monkeypatch, fake_cluster):
    fake_cluster.tools = [
        {"name": "hivemind.gpu.availability@v1", "description": "Report free GPUs available for scheduling", "inputSchema": {}},
        {"name": "hivemind.vm.start@v1", "description": "Start a virtual machine", "inputSchema": {"required": ["name"]}},
    ]
    fake_cluster.call_results["hivemind.gpu.availability@v1"] = {"free_gpus": 2, "nodes": ["a", "b"]}

    url = f"http://127.0.0.1:{fake_cluster.port}"
    runner = Ms4HermesRunner(
        hermes_dir=str(tmp_path), hivemind_url=url,
        agent_cls=FakeHermesAgent, face_lobe_chat=FakeFaceLobeChat(hivemind_url=url),
    )
    # Inject an allow-everything ethics evaluator into the default router.
    from machine_spirit_4.gateway.quartermaster import ToolRouter
    from machine_spirit_4.gateway.quartermaster import router as qm_router
    monkeypatch.setattr(qm_router, "_DEFAULT_ROUTER", ToolRouter(ethics_evaluator=lambda _i: {"allowed": True}))

    block, outcome = runner._try_quartermaster_inline("what gpus are available")
    assert block is not None
    assert "hivemind.gpu.availability@v1" in block
    assert "free_gpus" in block
    assert outcome["verdict"] == "inline"
    assert outcome["inline_executed"] is True


def test_inline_long_payload_is_lossless_through_guarded_reply(
    tmp_path,
    monkeypatch,
    fake_cluster,
):
    late_gpu = {
        "node_id": "GPU-NODE-LATE-SENTINEL",
        "gpu_id": "GPU-LATE-SENTINEL-9",
    }
    long_result = {
        "notes": "BEGIN  DOUBLE   SPACES|" + ("x" * 1900),
        "gpus": [
            {"node_id": "GPU-NODE-EARLY", "gpu_id": "GPU-EARLY-0"},
            late_gpu,
        ],
    }
    expected_result_text = json.dumps(long_result, ensure_ascii=False, default=str)
    assert len(expected_result_text) > 1800
    fake_cluster.tools = [
        {
            "name": _INLINE_TOOL,
            "description": "Report free GPUs available for scheduling",
            "inputSchema": {},
        },
    ]
    fake_cluster.call_results[_INLINE_TOOL] = long_result
    url = f"http://127.0.0.1:{fake_cluster.port}"
    runner, face = _guarded_inline_runner(tmp_path, monkeypatch, hivemind_url=url)
    from machine_spirit_4.gateway.quartermaster import ToolRouter
    from machine_spirit_4.gateway.quartermaster import router as qm_router
    monkeypatch.setattr(
        qm_router,
        "_DEFAULT_ROUTER",
        ToolRouter(ethics_evaluator=lambda _intent: {"allowed": True}),
    )

    block, outcome = runner._try_quartermaster_inline("what gpus are available")

    assert block is not None
    assert outcome["inline_result_text"] == expected_result_text
    assert "BEGIN  DOUBLE   SPACES" in outcome["inline_result_text"]
    assert late_gpu["node_id"] in outcome["inline_result_text"]
    assert late_gpu["gpu_id"] in outcome["inline_result_text"]
    assert not outcome["inline_result_text"].endswith("...")

    monkeypatch.setattr(runner, "_try_quartermaster_inline", lambda _message: (block, outcome))
    resp = runner.chat("what gpus are available", session_id="s-long", model="m")
    expected_reply = f"Authoritative result from {_INLINE_TOOL}: {expected_result_text}"

    assert resp["text"] == expected_reply
    assert len(resp["text"]) == len(expected_reply)
    assert "BEGIN  DOUBLE   SPACES" in resp["text"]
    assert resp["text"].index("GPU-NODE-EARLY") < resp["text"].index(late_gpu["node_id"])
    assert late_gpu["gpu_id"] in resp["text"]
    assert face.model_calls == 0
    assert resp["api_calls"] == 0
    assert resp["dispatched_job"] is None


def test_inline_falls_back_when_top_tool_needs_args(tmp_path, monkeypatch, fake_cluster):
    fake_cluster.tools = [
        {"name": "hivemind.vm.start@v1", "description": "Start a virtual machine by name", "inputSchema": {"required": ["name"]}},
    ]
    url = f"http://127.0.0.1:{fake_cluster.port}"
    runner = Ms4HermesRunner(
        hermes_dir=str(tmp_path), hivemind_url=url,
        agent_cls=FakeHermesAgent, face_lobe_chat=FakeFaceLobeChat(hivemind_url=url),
    )
    from machine_spirit_4.gateway.quartermaster import ToolRouter
    from machine_spirit_4.gateway.quartermaster import router as qm_router
    monkeypatch.setattr(qm_router, "_DEFAULT_ROUTER", ToolRouter(ethics_evaluator=lambda _i: {"allowed": True}))

    block, outcome = runner._try_quartermaster_inline("start the virtual machine")
    assert block is None  # needs args -> depth
    assert outcome["verdict"] == "depth"


def test_inline_disabled_by_env(tmp_path, monkeypatch, fake_cluster):
    monkeypatch.setenv("MS4_QM_INLINE", "0")
    url = f"http://127.0.0.1:{fake_cluster.port}"
    runner = Ms4HermesRunner(
        hermes_dir=str(tmp_path), hivemind_url=url,
        agent_cls=FakeHermesAgent, face_lobe_chat=FakeFaceLobeChat(hivemind_url=url),
    )
    block, outcome = runner._try_quartermaster_inline("what gpus are available")
    assert block is None
    assert outcome is None


_TIME_TOOL = "hivemind.time.now@v1"
_TIME_QUERY = f"Return the authoritative current time from {_TIME_TOOL}."
_INLINE_TIME = "2026-07-11T15:45:22.061333100+00:00"
_FALLBACK_TIME = "2026-07-11T15:45:22.064308900+00:00"


class _SequentialToolResults:
    """Count concrete fake-MCP tools/call executions and sequence results."""

    def __init__(
        self,
        tool_name: str,
        *results: Any,
        other_results: dict[str, Any] | None = None,
    ):
        self.tool_name = tool_name
        self.results = list(results)
        self.other_results = dict(other_results or {})
        self.call_names: list[str] = []

    def get(self, tool_name: str, default: Any = None) -> Any:
        self.call_names.append(tool_name)
        if tool_name != self.tool_name:
            return self.other_results.get(tool_name, default)
        if not self.results:
            return default
        return self.results.pop(0)


def _record_cluster_time_lookups(runner, monkeypatch) -> list[None]:
    calls: list[None] = []
    existing_lookup = runner._try_cluster_time

    def recording_lookup():
        calls.append(None)
        return existing_lookup()

    monkeypatch.setattr(runner, "_try_cluster_time", recording_lookup)
    return calls


def _real_time_turn(
    tmp_path,
    monkeypatch,
    fake_cluster,
    tool_results: _SequentialToolResults,
):
    fake_cluster.tools = [
        {
            "name": _TIME_TOOL,
            "description": "Get current time in UTC and local timezone.",
            "inputSchema": {"type": "object", "properties": {}},
            "annotations": {
                "readOnlyHint": True,
                "destructiveHint": False,
                "idempotentHint": True,
            },
        }
    ]
    fake_cluster.call_results = tool_results
    runner = _runner(
        tmp_path,
        monkeypatch,
        hivemind_url=f"http://127.0.0.1:{fake_cluster.port}",
    )
    _patch_direct_route(monkeypatch)
    monkeypatch.setattr(
        hermes_runner_module,
        "face_lobe_turn_start",
        lambda **_kwargs: {"revision": {"revision_id": 1}, "marked_stale": []},
    )
    monkeypatch.setattr(
        hermes_runner_module,
        "build_grounded_user_message",
        lambda message, _url: (message, "none"),
    )
    context_calls: list[dict[str, Any]] = []

    def build_context(**kwargs):
        context_calls.append(kwargs)
        return "MS4 Face Lobe context (authoritative):\n" + "\n".join(
            f"- {line}" for line in kwargs["extra_authoritative_lines"]
        )

    monkeypatch.setattr(
        hermes_runner_module,
        "build_face_lobe_context_block",
        build_context,
    )
    audit_events: list[dict[str, Any]] = []
    monkeypatch.setattr(
        hermes_runner_module,
        "append_event",
        lambda event_type, data: audit_events.append(
            {"event_type": event_type, **data}
        ),
    )
    from machine_spirit_4.gateway.quartermaster import ToolRouter
    from machine_spirit_4.gateway.quartermaster import router as qm_router

    monkeypatch.setattr(
        qm_router,
        "_DEFAULT_ROUTER",
        ToolRouter(
            ethics_evaluator=lambda _intent: {"allowed": True},
            audit=False,
        ),
    )
    return runner, context_calls, audit_events


@pytest.mark.parametrize("streaming", [False, True], ids=["blocking", "streaming"])
def test_inline_time_reuses_registered_result_for_same_turn_date_grounding(
    tmp_path,
    monkeypatch,
    fake_cluster,
    streaming,
):
    inline_result = {"raw_text": _INLINE_TIME}
    tool_results = _SequentialToolResults(
        _TIME_TOOL,
        inline_result,
        {"iso": _FALLBACK_TIME},
    )
    runner, context_calls, audit_events = _real_time_turn(
        tmp_path,
        monkeypatch,
        fake_cluster,
        tool_results,
    )
    grounding_lookups = _record_cluster_time_lookups(runner, monkeypatch)
    streamed: list[str] = []

    response = runner.chat(
        _TIME_QUERY,
        session_id=f"inline-time-{'stream' if streaming else 'blocking'}",
        model="m",
        stream_callback=streamed.append if streaming else None,
    )

    assert tool_results.call_names == [_TIME_TOOL]
    assert grounding_lookups == []
    result_text = json.dumps(inline_result, ensure_ascii=False)
    authoritative_text = f"Authoritative result from {_TIME_TOOL}: {result_text}"
    assert response["text"] == authoritative_text
    assert streamed == ([authoritative_text] if streaming else [])
    assert response["completed"] is True
    assert response["cancelled"] is False
    assert response["quartermaster_inline"] is True
    assert response["quartermaster"]["inline_executed"] is True
    assert response["quartermaster"]["inline_executed_tool"] == _TIME_TOOL
    assert response["api_calls"] == 0
    assert response["metrics"]["api_calls"] == 0
    assert response["metrics"]["streaming"] is streaming
    assert response["fallback_used"] is False
    assert response["quartermaster_result_fallback"] is False
    assert response["dispatched_job"] is None
    assert response["depth_lobe_model"] is None
    assert runner.face_lobe_chat.calls == []
    assert len(runner.face_lobe_chat.authoritative_calls) == 1
    assert context_calls[0]["extra_authoritative_lines"][0].startswith(
        f"current cluster date/time: {_INLINE_TIME} "
    )
    assert response["tool_trace"] == [
        {
            "event": "complete",
            "tool": _TIME_TOOL,
            "args": {},
            "success": True,
            "duration_ms": response["quartermaster"]["inline_elapsed_ms"],
            "result_type": "object",
            "result_bytes": len(result_text.encode("utf-8")),
        }
    ]
    assert set(response["tool_trace"][0]) == {
        "event",
        "tool",
        "args",
        "success",
        "duration_ms",
        "result_type",
        "result_bytes",
    }
    serialized_trace = json.dumps(response["tool_trace"], sort_keys=True)
    assert _INLINE_TIME not in serialized_trace
    assert "payload" not in serialized_trace
    assert "reasoning" not in serialized_trace
    terminal_events = [
        event for event in audit_events if event["event_type"] == "chat_turn"
    ]
    assert len(terminal_events) == 1
    assert terminal_events[0]["tool_trace_count"] == 1


@pytest.mark.parametrize(
    "inline_result",
    [
        pytest.param(None, id="failed"),
        pytest.param({"unexpected": "value"}, id="malformed"),
        pytest.param({"raw_text": ""}, id="unusable"),
        pytest.param({"raw_text": "clock unavailable"}, id="invalid-non-time"),
    ],
)
def test_unusable_inline_time_result_keeps_existing_date_grounding_lookup(
    tmp_path,
    monkeypatch,
    fake_cluster,
    inline_result,
):
    tool_results = _SequentialToolResults(
        _TIME_TOOL,
        inline_result,
        {"iso": _FALLBACK_TIME},
    )
    runner, context_calls, _audit_events = _real_time_turn(
        tmp_path,
        monkeypatch,
        fake_cluster,
        tool_results,
    )
    grounding_lookups = _record_cluster_time_lookups(runner, monkeypatch)

    response = runner.chat(
        _TIME_QUERY,
        session_id=f"inline-time-fallback-{inline_result!r}",
        model="m",
    )

    assert tool_results.call_names == [_TIME_TOOL, _TIME_TOOL]
    assert grounding_lookups == [None]
    assert context_calls[0]["extra_authoritative_lines"][0].startswith(
        f"current cluster date/time: {_FALLBACK_TIME} "
    )
    assert response["quartermaster"]["inline_executed"] is (inline_result is not None)


def test_inline_time_protocol_error_keeps_existing_date_grounding_lookup(
    tmp_path,
    monkeypatch,
    fake_cluster,
):
    tool_results = _SequentialToolResults(
        _TIME_TOOL,
        {
            "__fake_mcp_is_error__": True,
            "message": "registered time tool failed",
        },
        {"iso": _FALLBACK_TIME},
    )
    runner, context_calls, _audit_events = _real_time_turn(
        tmp_path,
        monkeypatch,
        fake_cluster,
        tool_results,
    )
    grounding_lookups = _record_cluster_time_lookups(runner, monkeypatch)

    response = runner.chat(
        _TIME_QUERY,
        session_id="inline-time-protocol-error",
        model="m",
    )

    assert grounding_lookups == [None]
    assert tool_results.call_names == [_TIME_TOOL, _TIME_TOOL]
    assert context_calls[0]["extra_authoritative_lines"][0].startswith(
        f"current cluster date/time: {_FALLBACK_TIME} "
    )
    assert response["quartermaster"]["inline_executed"] is False
    assert response["quartermaster_inline"] is False
    assert response["tool_trace"] == []
    assert response["api_calls"] == 0
    assert response["metrics"]["block_reason"] == "inline_terminal_failure"
    assert runner.face_lobe_chat.calls == []


def test_other_inline_tool_keeps_existing_registered_time_grounding_lookup(
    tmp_path,
    monkeypatch,
    fake_cluster,
):
    other_tool = "hivemind.gpu.availability@v1"
    tool_results = _SequentialToolResults(
        other_tool,
        {"free_gpus": 2},
        other_results={_TIME_TOOL: {"iso": _FALLBACK_TIME}},
    )
    runner, context_calls, _audit_events = _real_time_turn(
        tmp_path,
        monkeypatch,
        fake_cluster,
        tool_results,
    )
    fake_cluster.tools.append(
        {
            "name": other_tool,
            "description": "Report free GPUs available for scheduling.",
            "inputSchema": {"type": "object", "properties": {}},
            "annotations": {
                "readOnlyHint": True,
                "destructiveHint": False,
                "idempotentHint": True,
            },
        }
    )
    grounding_lookups = _record_cluster_time_lookups(runner, monkeypatch)

    response = runner.chat(
        "What GPUs are available?",
        session_id="other-inline-tool-time-grounding",
        model="m",
    )

    assert grounding_lookups == [None]
    assert tool_results.call_names == [other_tool, _TIME_TOOL]
    assert context_calls[0]["extra_authoritative_lines"][0].startswith(
        f"current cluster date/time: {_FALLBACK_TIME} "
    )
    assert response["quartermaster_inline"] is True
    assert response["quartermaster"]["inline_executed_tool"] == other_tool
    assert response["tool_trace"][0]["tool"] == other_tool
    assert response["api_calls"] == 0


@pytest.mark.parametrize(
    ("grounding_result", "expected_prefix"),
    [
        pytest.param(
            {"iso": _FALLBACK_TIME},
            f"current cluster date/time: {_FALLBACK_TIME} ",
            id="cluster-time",
        ),
        pytest.param(None, "current local date/time: ", id="absent-result-local-fallback"),
    ],
)
def test_non_inline_turn_keeps_registered_time_grounding_and_fallback(
    tmp_path,
    monkeypatch,
    fake_cluster,
    grounding_result,
    expected_prefix,
):
    tool_results = _SequentialToolResults(_TIME_TOOL, grounding_result)
    runner, context_calls, _audit_events = _real_time_turn(
        tmp_path,
        monkeypatch,
        fake_cluster,
        tool_results,
    )
    grounding_lookups = _record_cluster_time_lookups(runner, monkeypatch)
    monkeypatch.setenv("MS4_QM_INLINE", "0")

    response = runner.chat(
        "Hello without an inline tool.",
        session_id=f"non-inline-time-{grounding_result!r}",
        model="m",
    )

    assert tool_results.call_names == [_TIME_TOOL]
    assert grounding_lookups == [None]
    assert context_calls[0]["extra_authoritative_lines"][0].startswith(expected_prefix)
    assert response["quartermaster"] is None
    assert response["quartermaster_inline"] is False
    assert response["tool_trace"] == []
    assert response["api_calls"] == 1
    assert len(runner.face_lobe_chat.calls) == 1
    assert runner.face_lobe_chat.authoritative_calls == []
