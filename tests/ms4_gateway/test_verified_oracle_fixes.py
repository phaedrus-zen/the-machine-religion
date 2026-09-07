from __future__ import annotations

import http.client
import json
import threading
import time
import types
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

from machine_spirit_4.gateway import hermes_runner as hermes_runner_module
from machine_spirit_4.gateway import oracle_admin
from machine_spirit_4.gateway import server as server_module
from machine_spirit_4.gateway.face_lobe_chat import FaceLobeChatError
from machine_spirit_4.gateway.hermes_runner import Ms4HermesRunner


ROOT = Path(__file__).resolve().parents[2]


# A lifecycle_state snapshot whose chat plane is up. Oracle readiness now
# fails closed against HiveMind's /api/v1/cluster/lifecycle_state, so tests
# that assert the Oracle-status / voice-input gates pin a healthy chat plane
# here to keep exercising exactly those gates rather than the new one.
_HEALTHY_CHAT_LIFECYCLE = {
    "source": "menta_hli.cluster.lifecycle_state.v1",
    "ai_plane_ready": True,
    "boot_stage": "ready",
    "workload_phase": "inference_loaded",
    "signals": {"loaded_models_count": 2},
    "capabilities_available": ["chat", "embedding"],
    "capabilities_unavailable": [],
}


def _patch_healthy_foreground_model(monkeypatch) -> None:
    monkeypatch.setattr(
        oracle_admin,
        "choose_foreground_model",
        lambda **_kwargs: types.SimpleNamespace(
            model_id="nemotron-3-nano:4b",
            source="loaded",
            detail="test catalog admission",
        ),
    )


def test_oracle_readiness_inspects_nested_asr_service(monkeypatch) -> None:
    _patch_healthy_foreground_model(monkeypatch)
    monkeypatch.setattr(
        oracle_admin,
        "status",
        lambda *_args, **_kwargs: {"schema": "Ms4OracleSnapshot.v1", "healthy": True},
    )
    monkeypatch.setattr(
        oracle_admin.hivemind_state,
        "get_combined_snapshot",
        lambda *_args, **_kwargs: {
            "mcp_base_url": "http://hivemind.test:6105/mcp",
            "active_jobs": {"total_active": 0},
            "cluster_load": {},
            "service_health": {
                "healthy": True,
                "services": {
                    "oracle": {"name": "menta_oracle", "healthy": True},
                    "asr": {
                        "service": "ASR",
                        "healthy": False,
                        "provisioning_state": "configured_not_provisioned",
                        "detail": "No endpoints configured",
                    },
                },
            },
            "errors": [],
        },
    )
    monkeypatch.setattr(oracle_admin.hivemind_state, "hivemind_auth_configured", lambda: True)
    monkeypatch.setattr(
        oracle_admin.hivemind_state,
        "get_lifecycle_state",
        lambda *_args, **_kwargs: dict(_HEALTHY_CHAT_LIFECYCLE),
    )

    snapshot = oracle_admin.readiness("http://hivemind.test:6089", timeout=1)

    assert snapshot["ready"] is True
    assert snapshot["readiness"] == "ready"
    assert snapshot["voice_input"]["ready"] is False
    assert snapshot["voice_input"]["services"][0]["key"] == "services.asr"
    assert "No endpoints configured" in snapshot["voice_input"]["issues"][0]
    assert any(
        check["name"] == "voice_input" and check["state"] == "fail"
        for check in snapshot["checks"]
    )
    # Chat plane is healthy here; ASR unavailability is voice_input truth only.
    assert snapshot["chat_plane"]["status"] == "ready"


def test_oracle_readiness_accepts_one_healthy_asr_alternative(monkeypatch) -> None:
    _patch_healthy_foreground_model(monkeypatch)
    monkeypatch.setattr(
        oracle_admin,
        "status",
        lambda *_args, **_kwargs: {"schema": "Ms4OracleSnapshot.v1", "healthy": True},
    )
    monkeypatch.setattr(
        oracle_admin.hivemind_state,
        "get_combined_snapshot",
        lambda *_args, **_kwargs: {
            "mcp_base_url": "http://hivemind.test:6105/mcp",
            "active_jobs": {"total_active": 0},
            "cluster_load": {},
            "service_health": {
                "healthy": True,
                "services": {
                    "asr": {
                        "service": "ASR",
                        "healthy": True,
                        "provisioning_state": "running",
                    },
                    "asr_super": {
                        "service": "ASR_SUPER",
                        "healthy": False,
                        "provisioning_state": "configured_not_provisioned",
                        "detail": "No endpoints configured",
                    },
                },
            },
            "errors": [],
        },
    )
    monkeypatch.setattr(oracle_admin.hivemind_state, "hivemind_auth_configured", lambda: True)
    monkeypatch.setattr(
        oracle_admin.hivemind_state,
        "get_lifecycle_state",
        lambda *_args, **_kwargs: dict(_HEALTHY_CHAT_LIFECYCLE),
    )

    snapshot = oracle_admin.readiness("http://hivemind.test:6089", timeout=1)

    assert snapshot["ready"] is True
    assert snapshot["readiness"] == "ready"
    assert snapshot["voice_input"]["ready"] is True
    assert snapshot["voice_input"]["issues"] == []
    assert snapshot["voice_input"]["unavailable_alternates"] == [
        "ASR_SUPER: No endpoints configured"
    ]
    assert any(
        check["name"] == "voice_input" and check["state"] == "pass"
        for check in snapshot["checks"]
    )
    assert snapshot["chat_plane"]["status"] == "ready"
    assert any(
        check["name"] == "chat_plane" and check["state"] == "pass"
        for check in snapshot["checks"]
    )


def _patch_healthy_oracle_and_voice(monkeypatch) -> None:
    """Pin every non-chat-plane gate healthy so a readiness test isolates
    the chat-plane / lifecycle behaviour under test."""
    _patch_healthy_foreground_model(monkeypatch)
    monkeypatch.setattr(
        oracle_admin,
        "status",
        lambda *_args, **_kwargs: {"schema": "Ms4OracleSnapshot.v1", "healthy": True},
    )
    monkeypatch.setattr(
        oracle_admin.hivemind_state,
        "get_combined_snapshot",
        lambda *_args, **_kwargs: {
            "mcp_base_url": "http://hivemind.test:6105/mcp",
            "active_jobs": {"total_active": 0},
            "cluster_load": {},
            "service_health": {
                "menta_oracle": {"name": "menta_oracle", "healthy": True},
                "asr": {"service": "ASR", "healthy": True, "provisioning_state": "running"},
            },
            "errors": [],
        },
    )
    monkeypatch.setattr(oracle_admin.hivemind_state, "hivemind_auth_configured", lambda: True)


def test_oracle_readiness_blocks_when_chat_plane_down(monkeypatch) -> None:
    """Live P1 contradiction: HLI lifecycle_state reports ai_plane_ready=false
    with zero loaded models and chat unavailable while every other gate is
    healthy. Oracle readiness must fail closed to blocked, never ready."""
    _patch_healthy_oracle_and_voice(monkeypatch)
    monkeypatch.setattr(
        oracle_admin.hivemind_state,
        "get_lifecycle_state",
        lambda *_args, **_kwargs: {
            "source": "menta_hli.cluster.lifecycle_state.v1",
            "ai_plane_ready": False,
            "boot_stage": "degraded",
            "workload_phase": "idle",
            "signals": {"loaded_models_count": 0},
            "capabilities_available": [],
            "capabilities_unavailable": ["chat", "vision"],
        },
    )

    snap = oracle_admin.readiness("http://hivemind.test:6089", timeout=1)

    assert snap["ready"] is False
    assert snap["readiness"] == "blocked"
    assert snap["chat_plane"]["status"] == "down"
    assert snap["chat_plane"]["ai_plane_ready"] is False
    assert snap["chat_plane"]["chat_available"] is False
    assert snap["chat_plane"]["loaded_models_count"] == 0
    assert any(
        check["name"] == "chat_plane" and check["state"] == "fail"
        for check in snap["checks"]
    )
    assert any("chat plane is not ready" in blocker for blocker in snap["blockers"])


def test_oracle_readiness_degraded_when_lifecycle_missing(monkeypatch) -> None:
    """When lifecycle_state evidence cannot be fetched, readiness must degrade
    (never ready, never a fabricated hard blocker) and surface the reason."""
    _patch_healthy_oracle_and_voice(monkeypatch)

    def _raise(*_args, **_kwargs):
        raise oracle_admin.hivemind_state.HivemindStateError(
            "HiveMind lifecycle_state at http://hivemind.test:6089/... unreachable: timed out"
        )

    monkeypatch.setattr(oracle_admin.hivemind_state, "get_lifecycle_state", _raise)

    snap = oracle_admin.readiness("http://hivemind.test:6089", timeout=1)

    assert snap["ready"] is False
    assert snap["readiness"] == "degraded"
    assert snap["chat_plane"]["status"] == "unknown"
    assert snap["chat_plane"]["error"]
    assert any(
        check["name"] == "chat_plane" and check["state"] == "warn"
        for check in snap["checks"]
    )
    # Missing evidence degrades but must not fabricate a hard chat-plane blocker.
    assert not any("chat plane" in blocker.lower() for blocker in snap["blockers"])


def test_oracle_readiness_degraded_when_lifecycle_invalid(monkeypatch) -> None:
    """A lifecycle payload lacking the authoritative ai_plane_ready flag is
    treated as unknown evidence — readiness degrades, never ready."""
    _patch_healthy_oracle_and_voice(monkeypatch)
    monkeypatch.setattr(
        oracle_admin.hivemind_state,
        "get_lifecycle_state",
        lambda *_args, **_kwargs: {"source": "menta_hli.cluster.lifecycle_state.v1", "signals": {}},
    )

    snap = oracle_admin.readiness("http://hivemind.test:6089", timeout=1)

    assert snap["ready"] is False
    assert snap["readiness"] == "degraded"
    assert snap["chat_plane"]["status"] == "unknown"


def test_oracle_readiness_ready_when_chat_plane_healthy(monkeypatch) -> None:
    """Positive path: a healthy chat lifecycle plus the existing Oracle and
    voice gates passing yields ready=True."""
    _patch_healthy_oracle_and_voice(monkeypatch)
    monkeypatch.setattr(
        oracle_admin.hivemind_state,
        "get_lifecycle_state",
        lambda *_args, **_kwargs: dict(_HEALTHY_CHAT_LIFECYCLE),
    )

    snap = oracle_admin.readiness("http://hivemind.test:6089", timeout=1)

    assert snap["ready"] is True
    assert snap["readiness"] == "ready"
    assert snap["chat_plane"]["status"] == "ready"
    assert snap["chat_plane"]["ai_plane_ready"] is True
    assert snap["chat_plane"]["chat_available"] is True
    assert any(
        check["name"] == "chat_plane" and check["state"] == "pass"
        for check in snap["checks"]
    )
    assert snap["model_admission"] == {
        "status": "ready",
        "model_id": "nemotron-3-nano:4b",
        "source": "loaded",
        "detail": "test catalog admission",
        "error": None,
    }


def test_oracle_readiness_degrades_when_foreground_model_admission_fails(
    monkeypatch,
) -> None:
    _patch_healthy_oracle_and_voice(monkeypatch)
    monkeypatch.setattr(
        oracle_admin.hivemind_state,
        "get_lifecycle_state",
        lambda *_args, **_kwargs: dict(_HEALTHY_CHAT_LIFECYCLE),
    )

    def picker_failure(**_kwargs):
        raise RuntimeError("no admitted foreground model")

    monkeypatch.setattr(oracle_admin, "choose_foreground_model", picker_failure)

    snap = oracle_admin.readiness("http://hivemind.test:6089", timeout=1)

    assert snap["ready"] is False
    assert snap["readiness"] == "degraded"
    assert snap["chat_plane"]["status"] == "ready"
    assert snap["model_admission"]["status"] == "failed"
    assert snap["model_admission"]["model_id"] is None
    assert "no admitted foreground model" in snap["model_admission"]["error"]
    assert any(
        check["name"] == "model_admission" and check["state"] == "fail"
        for check in snap["checks"]
    )
    assert not any("model admission" in blocker.lower() for blocker in snap["blockers"])


class _FailingFaceLobe:
    def __init__(self) -> None:
        self._sessions: dict[str, Any] = {}
        self.session_ids: list[str] = []

    def chat(self, _message: str, **kwargs: Any) -> dict[str, Any]:
        self.session_ids.append(kwargs["session_id"])
        raise FaceLobeChatError("HiveMind /v1/chat/completions (stream) unreachable: timed out")

    chat_authoritative = chat

    def sessions(self) -> list[dict[str, Any]]:
        return []


def test_quartermaster_result_survives_face_lobe_timeout(monkeypatch, tmp_path) -> None:
    face_lobe = _FailingFaceLobe()
    runner = Ms4HermesRunner(
        hermes_dir=str(tmp_path),
        hivemind_url="http://hivemind.test:6089",
        face_lobe_chat=face_lobe,
    )
    result = {"free_gpus": 2, "nodes": ["alpha", "beta"]}
    block = (
        "MS4 Quartermaster inline tool result (authoritative):\n"
        '- tool: hivemind.gpu.availability@v1\n- result: {"free_gpus": 2}'
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
                "inline_executed_tool": "hivemind.gpu.availability@v1",
                "inline_result": result,
                "inline_result_text": json.dumps(result, ensure_ascii=False),
            },
        ),
    )
    monkeypatch.setattr(runner, "_try_cluster_time", lambda: None)
    monkeypatch.setattr(
        hermes_runner_module,
        "router_route",
        lambda message: types.SimpleNamespace(
            kind="deep",
            cleaned_message=message,
            goal=message,
            to_dict=lambda: {"kind": "deep"},
        ),
    )
    monkeypatch.setattr(
        hermes_runner_module,
        "face_lobe_turn_start",
        lambda **_kwargs: {"revision": {"revision_id": 1}, "marked_stale": []},
    )
    monkeypatch.setattr(
        hermes_runner_module,
        "build_face_lobe_context_block",
        lambda **_kwargs: "MS4 Face Lobe context (authoritative)",
    )
    monkeypatch.setattr(
        hermes_runner_module,
        "build_grounded_user_message",
        lambda message, _url: (message, None),
    )
    streamed: list[str] = []

    response = runner.chat(
        "Which GPUs are available?",
        model="face-model",
        stream_callback=streamed.append,
    )

    assert response["completed"] is True
    assert response["quartermaster_result_fallback"] is True
    assert response["quartermaster"]["inline_result"] == result
    assert "free_gpus" in response["text"]
    assert "trouble reaching" not in response["text"].lower()
    assert response["metrics"]["error"].endswith("timed out")
    assert streamed[-1] == response["text"]
    assert response["session_id"].startswith("ms4-")
    assert face_lobe.session_ids == [response["session_id"]]


def test_quartermaster_inline_outcome_retains_successful_tool_payload(monkeypatch, tmp_path) -> None:
    from machine_spirit_4.gateway import quartermaster

    monkeypatch.setenv("MS4_QM_INLINE", "1")
    result = {"free_gpus": 2, "nodes": ["alpha", "beta"]}
    decision = types.SimpleNamespace(
        verdict=quartermaster.VERDICT_INLINE,
        inline_tool=types.SimpleNamespace(
            name="hivemind.gpu.availability@v1",
            toolbox="gpu",
        ),
        resolution=types.SimpleNamespace(tier="deterministic"),
        to_dict=lambda: {
            "schema": "Ms4ToolRouteDecision.v1",
            "verdict": "inline",
            "query": "Which GPUs are available?",
            "inline_tool": "hivemind.gpu.availability@v1",
        },
    )
    monkeypatch.setattr(
        hermes_runner_module,
        "_catalog_proves_inline_execute",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(quartermaster, "decide", lambda *_args, **_kwargs: decision)
    monkeypatch.setattr(
        quartermaster,
        "execute_inline_tool",
        lambda *_args, **_kwargs: {
            "tool": "hivemind.gpu.availability@v1",
            "result": result,
            "elapsed_ms": 4,
        },
    )
    monkeypatch.setattr(
        quartermaster,
        "format_inline_block",
        lambda *_args, **_kwargs: "- result: retained",
    )
    runner = Ms4HermesRunner(
        hermes_dir=str(tmp_path),
        hivemind_url="http://hivemind.test:6089",
        face_lobe_chat=_FailingFaceLobe(),
    )

    block, outcome = runner._try_quartermaster_inline("Which GPUs are available?")

    assert block == "- result: retained"
    assert outcome["inline_result"] == result
    assert '"free_gpus": 2' in outcome["inline_result_text"]


class _StreamRunner:
    def __init__(self, *, error: bool = False) -> None:
        self.error = error

    def chat(self, message: str, *, session_id: str | None, stream_callback, **_kwargs: Any):
        if self.error:
            raise ValueError("model rejected")
        stream_callback("answer")
        return {
            "text": f"answer:{message}",
            "session_id": session_id or "session-created",
            "completed": True,
        }


def _chat_stream(runner: _StreamRunner) -> tuple[float, str]:
    class Handler(server_module.Ms4GatewayHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args: Any) -> None:
            return

    Handler.runner = runner
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=2)
    body = json.dumps({"message": "hello", "session_id": "session-kept"})
    started = time.monotonic()
    try:
        connection.request("POST", "/chat/stream", body=body, headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        wire = response.read().decode("utf-8")
        return time.monotonic() - started, wire
    finally:
        connection.close()
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


def test_chat_stream_done_reaches_eof_and_preserves_session() -> None:
    elapsed, wire = _chat_stream(_StreamRunner())

    assert elapsed < 1.0
    assert "event: done" in wire
    assert '"session_id": "session-kept"' in wire


def test_chat_stream_error_reaches_eof() -> None:
    elapsed, wire = _chat_stream(_StreamRunner(error=True))

    assert elapsed < 1.0
    assert "event: error" in wire
    assert "model rejected" in wire


def test_voice_predicates_use_user_copy_and_keep_diagnostics_visible() -> None:
    raw = "voice_input_ready=false asr.status=unhealthy detail=No endpoints configured"
    payload = server_module._voice_error_payload(
        server_module.VoiceUnavailable(raw),
        fail_closed=True,
    )

    assert payload["error"] == "Voice input is not ready. Provision ASR and try again."
    assert "voice_input_ready=false" not in payload["error"]
    assert payload["diagnostics"] == raw
    assert payload["fail_closed"] is True

    html = (ROOT / "machine_spirit_4" / "web" / "index.html").read_text(encoding="utf-8")
    for required in (
        "function voiceUserFacingCopy(",
        "voiceError.voiceDiagnostics = payload.diagnostics || payload.error || '';",
        "voiceStatusLine.title = diagnostics ? String(diagnostics) : '';",
        "console.warn('Voice turn failed diagnostics:', diagnostics);",
        "if (eventData.session_id) {",
        "const finalText = String(eventData.text || state.text || '');",
    ):
        assert required in html


def test_oracle_front_door_checks_readiness_before_claiming_ready() -> None:
    """The front door must open on "Checking readiness" (not a premature
    "Ready"), probe /hivemind/oracle/readiness during startup, and reflect
    ready / degraded / blocked without clobbering an active or engaged turn."""
    html = (ROOT / "machine_spirit_4" / "web" / "index.html").read_text(encoding="utf-8")

    # Initial copy + chip say "Checking readiness", never a premature "Ready".
    assert "<strong>Checking readiness</strong>" in html
    assert "Checking readiness…" in html
    assert 'data-turn-phase="checking"' in html
    assert "<strong>Ready</strong>" not in html

    # A readiness probe is defined and fired during initial page startup.
    assert "async function probeOracleReadinessOnStartup(" in html
    assert "function applyOracleReadinessSnapshot(" in html
    assert "probeOracleReadinessOnStartup();" in html
    assert "fetch('/hivemind/oracle/readiness'" in html

    # Each readiness state is reflected honestly after the fetch.
    assert "setOracleStageState('idle', 'Ready.'" in html
    assert "setOracleStageState('error', 'Readiness degraded.'" in html
    assert "setOracleStageState('blocked', 'Oracle chat is not ready.'" in html

    # ...but the probe never overwrites an active/engaged voice or chat turn.
    assert "oracleTurnEngaged" in html
    assert "if (!oracleStage || oracleTurnEngaged) return;" in html
