"""Unit + route tests for the May 26 2026 HiveMind admin expansion.

Covers the four new admin modules (``oracle_admin``, ``training_admin``,
``adapter_admin``, ``loadout_admin``), the screenshot-with-image
upgrade in ``vm_admin.get_screenshot``, and the new REST routes /
MCP proxies they back. Reuses the same in-process MCP fake the rest
of the gateway test suite uses.
"""

from __future__ import annotations

import base64
import io
import json
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from machine_spirit_4.gateway import (
    adapter_admin,
    hivemind_tools,
    loadout_admin,
    oracle_admin,
    server as srv_module,
    training_admin,
    vm_admin,
)
from machine_spirit_4.gateway.adapter_admin import AdapterAdminError
from machine_spirit_4.gateway.loadout_admin import LoadoutAdminError
from machine_spirit_4.gateway.oracle_admin import OracleAdminError
from machine_spirit_4.gateway.training_admin import TrainingAdminError


# ---------------------------------------------------------------------------
# In-process fake MCP server (mirrors the pattern in test_hivemind_state.py)
# ---------------------------------------------------------------------------


class _FakeMcp:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []
        self.responses: dict[str, Any] = {}
        self.image_responses: dict[str, dict[str, str]] = {}
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.port: int = 0

    def set_response(self, tool: str, payload: Any) -> None:
        self.responses[tool] = payload

    def set_image_response(self, tool: str, *, json_payload: Any, image_base64: str, mime: str) -> None:
        """Emit BOTH a text JSON content block AND an MCP image content
        block — mirrors the May-26 HiveMind upgrade for vm.screenshot."""
        self.responses[tool] = json_payload
        self.image_responses[tool] = {"data": image_base64, "mimeType": mime}

    def start(self) -> None:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args, **_kwargs):
                return

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0") or "0")
                raw = self.rfile.read(length).decode("utf-8") if length else "{}"
                try:
                    body = json.loads(raw)
                except json.JSONDecodeError:
                    body = {}
                outer.calls.append({"body": body})
                tool_name = (body.get("params") or {}).get("name") if isinstance(body, dict) else None
                content: list[dict[str, Any]] = []
                payload = outer.responses.get(tool_name or "", {})
                content.append({"type": "text", "text": json.dumps(payload)})
                img = outer.image_responses.get(tool_name or "")
                if img:
                    content.append({"type": "image", "data": img["data"], "mimeType": img["mimeType"]})
                envelope = {
                    "jsonrpc": "2.0",
                    "id": body.get("id", 1),
                    "result": {"content": content, "isError": False},
                }
                wire = json.dumps(envelope).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(wire)))
                self.end_headers()
                self.wfile.write(wire)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]

    def stop(self) -> None:
        if self.server:
            try:
                self.server.shutdown()
            except Exception:
                pass


@pytest.fixture
def fake_mcp(monkeypatch):
    server = _FakeMcp()
    server.start()
    monkeypatch.setenv("MS4_HIVEMIND_MCP_URL", f"http://127.0.0.1:{server.port}/mcp")
    monkeypatch.delenv("MS4_HIVEMIND_API_KEY", raising=False)
    yield server
    server.stop()


def hurl(fake: _FakeMcp) -> str:
    return f"http://127.0.0.1:{fake.port}"


# A lifecycle_state snapshot whose chat plane is up. Oracle readiness fails
# closed against HiveMind's /api/v1/cluster/lifecycle_state, so readiness
# tests that assert the Oracle-status / voice-input gates pin a healthy chat
# plane here to keep exercising exactly those gates, not the chat-plane gate.
_HEALTHY_CHAT_LIFECYCLE = {
    "source": "menta_hli.cluster.lifecycle_state.v1",
    "ai_plane_ready": True,
    "boot_stage": "ready",
    "workload_phase": "inference_loaded",
    "signals": {"loaded_models_count": 2},
    "capabilities_available": ["chat", "embedding"],
    "capabilities_unavailable": [],
}


# ===========================================================================
# Wrappers + admin modules
# ===========================================================================


def test_oracle_admin_status_wraps_schema(fake_mcp):
    fake_mcp.set_response("hivemind.oracle.status", {"healthy": True, "running_plans": 0})
    snap = oracle_admin.status(hurl(fake_mcp))
    assert snap["schema"] == "Ms4OracleSnapshot.v1"
    assert snap["healthy"] is True
    assert snap["running_plans"] == 0


def test_oracle_admin_readiness_surfaces_provisioning_gap(fake_mcp):
    fake_mcp.set_response("hivemind.oracle.status", {"healthy": True, "running_plans": 0})
    fake_mcp.set_response("hivemind.jobs.active@v1", {"total_active": 0, "summary": ""})
    fake_mcp.set_response("hivemind.cluster.load@v1", {"trackers": {"amplification_ratio": 1.0}})
    fake_mcp.set_response(
        "hivemind.service_health@v1",
        {
            "menta_hli": {"name": "menta_hli", "healthy": True},
            "menta_oracle": {"name": "menta_oracle", "healthy": False, "error": "not provisioned"},
        },
    )
    snap = oracle_admin.readiness(hurl(fake_mcp))
    assert snap["schema"] == "Ms4OracleReadiness.v1"
    assert snap["ready"] is False
    assert snap["readiness"] == "blocked"
    assert snap["provisioning"]["automatic"] is False
    assert snap["provisioning"]["state"] == "approval_required"
    assert any("menta_oracle" in item for item in snap["blockers"])
    assert any(
        check["name"] == "oracle_service_health" and check["state"] == "fail"
        for check in snap["checks"]
    )


def test_oracle_admin_readiness_degraded_is_not_labeled_safe(monkeypatch):
    monkeypatch.setattr(
        oracle_admin,
        "status",
        lambda *_args, **_kwargs: {"schema": "Ms4OracleSnapshot.v1", "status": "starting"},
    )
    monkeypatch.setattr(
        oracle_admin.hivemind_state,
        "get_combined_snapshot",
        lambda *_args, **_kwargs: {
            "mcp_base_url": "http://127.0.0.1:6105/mcp",
            "active_jobs": {"total_active": 0},
            "cluster_load": {},
            "service_health": {"menta_hli": {"name": "menta_hli", "healthy": True}},
            "errors": [],
        },
    )
    monkeypatch.setattr(oracle_admin.hivemind_state, "hivemind_auth_configured", lambda: True)
    monkeypatch.setattr(
        oracle_admin.hivemind_state,
        "get_lifecycle_state",
        lambda *_args, **_kwargs: dict(_HEALTHY_CHAT_LIFECYCLE),
    )

    snap = oracle_admin.readiness("http://hivemind.test:6089", timeout=1)
    assert snap["readiness"] == "degraded"
    assert snap["ready"] is False
    assert snap["provisioning"]["state"] == "diagnostic_required"
    labels = [action["label"] for action in snap["next_actions"]]
    assert not any("safe to use" in label for label in labels)
    assert any("Resolve readiness warnings" in label for label in labels)


@pytest.mark.parametrize(
    "voice_entry",
    [
        {
            "name": "ASR",
            "healthy": False,
            "provisioning_state": "configured_not_provisioned",
            "detail": "No endpoints configured",
        },
        {
            "name": "voice_input",
            "voice_input_ready": False,
            "status": "unavailable",
        },
    ],
)
def test_oracle_admin_readiness_keeps_chat_ready_when_voice_input_unavailable(monkeypatch, voice_entry):
    monkeypatch.setattr(
        oracle_admin,
        "status",
        lambda *_args, **_kwargs: {"schema": "Ms4OracleSnapshot.v1", "healthy": True},
    )
    monkeypatch.setattr(
        oracle_admin.hivemind_state,
        "get_combined_snapshot",
        lambda *_args, **_kwargs: {
            "mcp_base_url": "http://127.0.0.1:6105/mcp",
            "active_jobs": {"total_active": 0},
            "cluster_load": {},
            "service_health": {
                "menta_hli": {"name": "menta_hli", "healthy": True},
                "voice_input": voice_entry,
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

    snap = oracle_admin.readiness("http://hivemind.test:6089", timeout=1)

    assert snap["readiness"] == "ready"
    assert snap["ready"] is True
    assert snap["voice_input"]["ready"] is False
    assert snap["voice_input"]["issues"]
    assert snap["blockers"] == []
    assert snap["chat_plane"]["status"] == "ready"
    assert any(
        check["name"] == "voice_input" and check["state"] == "fail"
        for check in snap["checks"]
    )


def test_oracle_admin_readiness_accepts_running_asr(monkeypatch):
    monkeypatch.setattr(
        oracle_admin,
        "status",
        lambda *_args, **_kwargs: {"schema": "Ms4OracleSnapshot.v1", "healthy": True},
    )
    monkeypatch.setattr(
        oracle_admin.hivemind_state,
        "get_combined_snapshot",
        lambda *_args, **_kwargs: {
            "mcp_base_url": "http://127.0.0.1:6105/mcp",
            "active_jobs": {"total_active": 0},
            "cluster_load": {},
            "service_health": {
                "menta_hli": {"name": "menta_hli", "healthy": True},
                "ASR": {
                    "name": "ASR",
                    "healthy": True,
                    "provisioning_state": "running",
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

    snap = oracle_admin.readiness("http://hivemind.test:6089", timeout=1)

    assert snap["readiness"] == "ready"
    assert snap["ready"] is True
    assert snap["voice_input"]["ready"] is True


def test_oracle_admin_chat_passes_message(fake_mcp):
    fake_mcp.set_response("hivemind.oracle.chat", {"reply": "Run inventory first."})
    result = oracle_admin.chat(hurl(fake_mcp), "what should I do next?")
    assert result["reply"] == "Run inventory first."
    args = fake_mcp.calls[-1]["body"]["params"]["arguments"]
    assert args["message"] == "what should I do next?"


def test_oracle_admin_chat_rejects_empty(fake_mcp):
    with pytest.raises(ValueError):
        oracle_admin.chat(hurl(fake_mcp), "")


def test_training_admin_combined_snapshot(fake_mcp):
    fake_mcp.set_response("hivemind.training.backends@v1", {"backends": [{"name": "forge"}, {"name": "peft"}]})
    fake_mcp.set_response("hivemind.training.status@v1", {"jobs": [{"id": "t-1", "state": "running"}]})
    snap = training_admin.combined_snapshot(hurl(fake_mcp))
    assert snap["schema"] == "Ms4TrainingSnapshot.v1"
    assert len(snap["backends"]) == 2
    assert snap["status"]["jobs"][0]["id"] == "t-1"
    assert snap["errors"] == []


def test_training_admin_start_rejects_empty_recipe(fake_mcp):
    with pytest.raises(ValueError):
        training_admin.start_job(hurl(fake_mcp), {})


def test_adapter_admin_list_and_deploy(fake_mcp):
    fake_mcp.set_response(
        "hivemind.adapters.list@v1",
        {"adapters": [{"id": "lora-1", "name": "MyLora", "base_model": "llama3.1:8b"}]},
    )
    fake_mcp.set_response("hivemind.adapters.deploy@v1", {"ok": True, "adapter_id": "lora-1"})
    adapters = adapter_admin.list_adapters(hurl(fake_mcp))
    assert len(adapters) == 1
    assert adapters[0]["id"] == "lora-1"
    result = adapter_admin.deploy(hurl(fake_mcp), "lora-1", "llama3.1:8b")
    assert result["ok"] is True


def test_adapter_admin_deploy_rejects_empty_id(fake_mcp):
    with pytest.raises(ValueError):
        adapter_admin.deploy(hurl(fake_mcp), "")


def test_loadout_admin_list_and_apply(fake_mcp):
    fake_mcp.set_response(
        "hivemind.loadout.profiles@v1",
        {"profiles": [{"id": "voice-stack", "active": False}, {"id": "depth", "active": True}]},
    )
    fake_mcp.set_response("hivemind.loadout.apply@v1", {"ok": True})
    profiles = loadout_admin.list_profiles(hurl(fake_mcp))
    assert len(profiles) == 2
    result = loadout_admin.apply(hurl(fake_mcp), "voice-stack")
    assert result["ok"] is True


def test_loadout_admin_normalises_hli_tier_map(fake_mcp):
    fake_mcp.set_response(
        "hivemind.loadout.profiles@v1",
        {
            "hardware": {"gpu_count": 1, "total_vram_gb": 32.0},
            "recommended_tier": "super",
            "active_loadout": {"tier": "medium", "quality": "balanced"},
            "profiles": {
                "tiers": {
                    "medium": {"label": "Medium", "vram_gb": 16},
                    "super": {"label": "Super", "vram_gb": 32},
                },
                "presets": {
                    "medium": {
                        "chat": {"model": "nemotron-3-nano:4b", "source": "ollama"},
                    },
                    "super": {
                        "chat": {"model": "qwen3.6:35b", "source": "ollama"},
                        "asr": {"model": "whisper-large-v3-turbo", "source": "gim"},
                    },
                },
            },
        },
    )

    snapshot = loadout_admin.combined_snapshot(hurl(fake_mcp))

    assert snapshot["hardware"]["total_vram_gb"] == 32.0
    assert snapshot["recommended_tier"] == "super"
    assert [p["id"] for p in snapshot["profiles"]] == ["medium", "super"]
    medium, super_profile = snapshot["profiles"]
    assert medium["active"] is True
    assert medium["models"] == ["nemotron-3-nano:4b"]
    assert super_profile["recommended"] is True
    assert super_profile["models"] == ["qwen3.6:35b", "whisper-large-v3-turbo"]


def test_loadout_admin_apply_rejects_empty_id(fake_mcp):
    with pytest.raises(ValueError):
        loadout_admin.apply(hurl(fake_mcp), "")


# ===========================================================================
# Screenshot — image content fallthrough
# ===========================================================================


def test_vm_screenshot_surfaces_image_content_block(fake_mcp):
    """The May 26 HiveMind upgrade made vm.screenshot emit both a text
    JSON envelope AND an MCP-native ``image`` content block. MS4's
    wrapper must surface BOTH so the UI gets the binary without
    double-round-tripping through base64."""
    fake_mcp.set_image_response(
        "hivemind.vm.screenshot@v1",
        json_payload={"format": "png", "width": 800, "height": 600},
        image_base64="ABC123",
        mime="image/png",
    )
    result = vm_admin.get_screenshot(hurl(fake_mcp), "vm-1")
    assert result["schema"] == "Ms4VmScreenshot.v1"
    assert result["vm_id"] == "vm-1"
    assert result["image_base64"] == "ABC123"
    assert result["mime_type"] == "image/png"
    assert result["metadata"]["format"] == "png"
    assert result["metadata"]["width"] == 800


def test_vm_screenshot_fallback_to_metadata_data_base64(fake_mcp):
    """Backwards-compatible fallback: clusters that don't emit the
    image content block but DO carry data_base64 in the JSON metadata
    still produce a usable screenshot."""
    fake_mcp.set_response(
        "hivemind.vm.screenshot@v1",
        {"format": "png", "data_base64": "LEGACY_DATA"},
    )
    result = vm_admin.get_screenshot(hurl(fake_mcp), "vm-1")
    assert result["image_base64"] == "LEGACY_DATA"
    assert result["mime_type"] == "image/png"


# ===========================================================================
# call_tool_with_image direct test
# ===========================================================================


def test_call_tool_with_image_unwraps_both_blocks(fake_mcp):
    fake_mcp.set_image_response(
        "hivemind.vm.screenshot@v1",
        json_payload={"format": "jpeg", "vm_id": "vm-1"},
        image_base64="JPGBYTES",
        mime="image/jpeg",
    )
    result = hivemind_tools.call_tool_with_image(
        hurl(fake_mcp), "hivemind.vm.screenshot@v1", {"vm_id": "vm-1"}
    )
    assert result["json"]["format"] == "jpeg"
    assert result["image_base64"] == "JPGBYTES"
    assert result["image_mime_type"] == "image/jpeg"


# ===========================================================================
# REST route tests
# ===========================================================================


class _DummyRunner:
    ms3_url = "http://ms3:9080"
    default_model = "qwen3-coder-next:latest"
    face_lobe_chat = types.SimpleNamespace(_sessions={})

    def __init__(self, hivemind_url: str):
        self.hivemind_url = hivemind_url


def _make_handler(runner, method: str, path: str, body: bytes = b"") -> srv_module.Ms4GatewayHandler:
    handler = srv_module.Ms4GatewayHandler.__new__(srv_module.Ms4GatewayHandler)
    handler.runner = runner
    handler.command = method
    handler.path = path
    headers = {"Host": "127.0.0.1:9180"}
    if body:
        headers["Content-Length"] = str(len(body))
        headers["Content-Type"] = "application/json"
    handler.headers = types.SimpleNamespace(get=headers.get)
    handler.rfile = io.BytesIO(body)
    handler.wfile = io.BytesIO()
    handler.requestline = f"{method} {path} HTTP/1.1"
    handler.request_version = "HTTP/1.1"
    handler.client_address = ("127.0.0.1", 0)
    handler.server = types.SimpleNamespace(server_name="test", server_port=0)
    handler.protocol_version = "HTTP/1.1"
    return handler


def _read_response(handler) -> tuple[int, dict[str, Any]]:
    raw = handler.wfile.getvalue()
    status_line, _, rest = raw.partition(b"\r\n")
    status = int(status_line.split(b" ")[1])
    head, _, body = rest.partition(b"\r\n\r\n")
    return status, json.loads(body.decode("utf-8") or "{}")


def test_route_oracle_status(fake_mcp):
    fake_mcp.set_response("hivemind.oracle.status", {"healthy": True})
    handler = _make_handler(_DummyRunner(hurl(fake_mcp)), "GET", "/hivemind/oracle/status")
    handler._hivemind_oracle_status_get()
    status, body = _read_response(handler)
    assert status == 200
    assert body["schema"] == "Ms4OracleSnapshot.v1"


def test_route_oracle_status_derives_health_when_tool_omits_field(fake_mcp):
    fake_mcp.set_response("hivemind.oracle.status", {"active_requests": 0, "last_error": None})
    handler = _make_handler(_DummyRunner(hurl(fake_mcp)), "GET", "/hivemind/oracle/status")
    handler._hivemind_oracle_status_get()
    status, body = _read_response(handler)
    assert status == 200
    assert body["schema"] == "Ms4OracleSnapshot.v1"
    assert body["healthy"] is True
    assert body["state"] == "ready"


def test_route_oracle_readiness(fake_mcp):
    fake_mcp.set_response("hivemind.oracle.status", {"healthy": True})
    fake_mcp.set_response("hivemind.jobs.active@v1", {"total_active": 0, "summary": ""})
    fake_mcp.set_response("hivemind.cluster.load@v1", {"trackers": {"amplification_ratio": 1.0}})
    fake_mcp.set_response("hivemind.service_health@v1", {"menta_hli": {"healthy": True}})
    handler = _make_handler(_DummyRunner(hurl(fake_mcp)), "GET", "/hivemind/oracle/readiness")
    handler._hivemind_oracle_readiness_get()
    status, body = _read_response(handler)
    assert status == 200
    assert body["schema"] == "Ms4OracleReadiness.v1"
    assert body["provisioning"]["automatic"] is False


def test_route_oracle_readiness_dispatches_get(fake_mcp):
    fake_mcp.set_response("hivemind.oracle.status", {"healthy": True})
    fake_mcp.set_response("hivemind.jobs.active@v1", {"total_active": 0, "summary": ""})
    fake_mcp.set_response("hivemind.cluster.load@v1", {"trackers": {"amplification_ratio": 1.0}})
    fake_mcp.set_response("hivemind.service_health@v1", {"menta_hli": {"healthy": True}})
    handler = _make_handler(_DummyRunner(hurl(fake_mcp)), "GET", "/hivemind/oracle/readiness")
    handler.do_GET()
    status, body = _read_response(handler)
    assert status == 200
    assert body["schema"] == "Ms4OracleReadiness.v1"


def test_route_oracle_chat_validates_message(fake_mcp):
    handler = _make_handler(_DummyRunner(hurl(fake_mcp)), "POST", "/hivemind/oracle/chat", body=b'{}')
    handler._hivemind_oracle_chat()
    status, body = _read_response(handler)
    assert status == 400
    assert "message" in body["error"]


def test_route_training_snapshot(fake_mcp):
    fake_mcp.set_response("hivemind.training.backends@v1", {"backends": []})
    fake_mcp.set_response("hivemind.training.status@v1", {"jobs": []})
    handler = _make_handler(_DummyRunner(hurl(fake_mcp)), "GET", "/hivemind/training")
    handler._hivemind_training_get()
    status, body = _read_response(handler)
    assert status == 200
    assert body["schema"] == "Ms4TrainingSnapshot.v1"


def test_route_training_start_rejects_missing_recipe(fake_mcp):
    handler = _make_handler(_DummyRunner(hurl(fake_mcp)), "POST", "/hivemind/training/start", body=b'{}')
    handler._hivemind_training_start()
    status, body = _read_response(handler)
    assert status == 400
    assert "recipe" in body["error"]


def test_route_adapters_and_loadout(fake_mcp):
    fake_mcp.set_response("hivemind.adapters.list@v1", {"adapters": []})
    fake_mcp.set_response("hivemind.loadout.profiles@v1", {"profiles": []})

    handler1 = _make_handler(_DummyRunner(hurl(fake_mcp)), "GET", "/hivemind/adapters")
    handler1._hivemind_adapters_get()
    status1, body1 = _read_response(handler1)
    assert status1 == 200
    assert body1["schema"] == "Ms4AdapterSnapshot.v1"

    handler2 = _make_handler(_DummyRunner(hurl(fake_mcp)), "GET", "/hivemind/loadout")
    handler2._hivemind_loadout_get()
    status2, body2 = _read_response(handler2)
    assert status2 == 200
    assert body2["schema"] == "Ms4LoadoutSnapshot.v1"


def test_route_jobs_cancel_requires_confirm(fake_mcp):
    handler = _make_handler(_DummyRunner(hurl(fake_mcp)), "POST", "/hivemind/jobs/cancel", body=b'{}')
    handler._hivemind_jobs_cancel()
    status, body = _read_response(handler)
    assert status == 400
    assert "confirm" in body["error"]


def test_route_jobs_cancel_with_confirm_fires(fake_mcp):
    job_id = "77a4fe89-2f09-47b0-8f52-c35186fe82dc"
    fake_mcp.set_response("hivemind.jobs.cancel@v1", {"cancelled": True, "trace_id": job_id})
    body = json.dumps({"confirm": True, "job_id": job_id, "reason": "test"}).encode()
    handler = _make_handler(_DummyRunner(hurl(fake_mcp)), "POST", "/hivemind/jobs/cancel", body=body)
    handler._hivemind_jobs_cancel()
    status, body = _read_response(handler)
    assert status == 200
    assert body == {"cancelled": True, "trace_id": job_id}


def test_route_jobs_cancel_requires_job_id(fake_mcp):
    handler = _make_handler(
        _DummyRunner(hurl(fake_mcp)), "POST", "/hivemind/jobs/cancel",
        body=b'{"confirm": true}',
    )
    handler._hivemind_jobs_cancel()
    status, body = _read_response(handler)
    assert status == 400
    assert "job_id" in body["error"]


def test_route_service_lifecycle_dispatchers(fake_mcp):
    """Enable / disable / restart all share the same shape: forward the
    service name to the corresponding hivemind.services.* tool."""
    fake_mcp.set_response("hivemind.services.enable@v1", {"ok": True})
    fake_mcp.set_response("hivemind.services.disable@v1", {"ok": True})
    fake_mcp.set_response("hivemind.services.restart@v1", {"ok": True})

    for action in ("enable", "disable", "restart"):
        handler = _make_handler(
            _DummyRunner(hurl(fake_mcp)), "POST", f"/hivemind/services/menta_hli/{action}"
        )
        getattr(handler, f"_hivemind_service_{action}")("menta_hli")
        status, body = _read_response(handler)
        assert status == 200, f"{action} failed: {body}"
        assert body["ok"] is True
        last_call = fake_mcp.calls[-1]["body"]["params"]
        assert last_call["name"] == f"hivemind.services.{action}@v1"
        assert last_call["arguments"]["service_name"] == "menta_hli"


def test_route_inference_models(fake_mcp):
    fake_mcp.set_response("hivemind.inference.models@v1", {"data": [{"id": "llama3.1:8b"}]})
    handler = _make_handler(_DummyRunner(hurl(fake_mcp)), "GET", "/hivemind/inference/models")
    handler._hivemind_inference_models_get()
    status, body = _read_response(handler)
    assert status == 200
    assert body["data"][0]["id"] == "llama3.1:8b"


def test_route_inference_chat_validates_required(fake_mcp):
    handler = _make_handler(_DummyRunner(hurl(fake_mcp)), "POST", "/hivemind/inference/chat", body=b'{"model": "x"}')
    handler._hivemind_inference_chat()
    status, body = _read_response(handler)
    assert status == 400


def test_route_deploy_gim_validates_name(fake_mcp):
    handler = _make_handler(_DummyRunner(hurl(fake_mcp)), "POST", "/hivemind/deploy/gim", body=b'{}')
    handler._hivemind_deploy_gim()
    status, body = _read_response(handler)
    assert status == 400
    assert "gim_name" in body["error"]
