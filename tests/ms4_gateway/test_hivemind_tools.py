"""Unit tests for the typed HiveMind tool wrappers.

Covers every public function in ``machine_spirit_4.gateway.hivemind_tools``
against a fake HiveMind MCP server. The fake records every call so we
can verify the right tool name + arguments went on the wire (it's the
contract surface other modules build on, so a regression here would
silently break the gateway routes + MCP proxy + admin modules).
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from machine_spirit_4.gateway import hivemind_tools
from machine_spirit_4.gateway.hivemind_tools import HivemindToolError


class _FakeMcp:
    """In-process MCP server. Records every call. Returns either the
    pre-canned response set per tool, or an ``isError`` envelope if
    ``set_error`` was used."""

    def __init__(self):
        self.calls: list[dict[str, Any]] = []
        self.responses: dict[str, Any] = {}
        self.errors: dict[str, str] = {}
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.port: int = 0

    def set_response(self, tool: str, payload: Any) -> None:
        self.responses[tool] = payload

    def set_error(self, tool: str, message: str) -> None:
        self.errors[tool] = message

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
                outer.calls.append({"path": self.path, "body": body, "headers": dict(self.headers)})
                tool_name = (body.get("params") or {}).get("name") if isinstance(body, dict) else None
                if tool_name in outer.errors:
                    envelope = {
                        "jsonrpc": "2.0",
                        "id": body.get("id", 1),
                        "result": {
                            "content": [{"type": "text", "text": outer.errors[tool_name]}],
                            "isError": True,
                        },
                    }
                else:
                    payload = outer.responses.get(tool_name or "", {})
                    envelope = {
                        "jsonrpc": "2.0",
                        "id": body.get("id", 1),
                        "result": {
                            "content": [{"type": "text", "text": json.dumps(payload)}],
                            "isError": False,
                        },
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


def hm_url(fake: _FakeMcp) -> str:
    """The HLI proxy URL we hand to wrappers. The MS4_HIVEMIND_MCP_URL
    pin is what actually routes the request to the fake."""
    return f"http://127.0.0.1:{fake.port}"


# ---------------------------------------------------------------------------
# One representative happy-path test per domain. The transport + auth +
# fallback logic is already exercised by test_hivemind_state.py; here we
# just confirm each wrapper hits the right tool name and passes the
# right arguments.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn_name,tool_name,args,expected_response,call_args",
    [
        # apps
        ("app_list", "hivemind.app.list@v1", (), {"apps": [{"id": "a1"}]}, {}),
        ("app_get", "hivemind.app.get@v1", ("a1",), {"id": "a1"}, {"id": "a1"}),
        ("app_start", "hivemind.app.start@v1", ("a1",), {"ok": True}, {"id": "a1"}),
        ("app_stop", "hivemind.app.stop@v1", ("a1",), {"ok": True}, {"id": "a1"}),
        ("apps_discover", "hivemind.apps.discover@v1", (), {"discovered": []}, {}),
        # vm — live contract keys the VM by ``name`` (not ``vm_id``);
        # screenshot additionally requires ``width``+``height``.
        ("vm_list", "hivemind.vm.list@v1", (), {"vms": [{"id": "vm-1"}]}, {}),
        ("vm_start", "hivemind.vm.start@v1", ("vm-1",), {"ok": True}, {"name": "vm-1"}),
        ("vm_stop", "hivemind.vm.stop@v1", ("vm-1",), {"ok": True}, {"name": "vm-1"}),
        ("vm_force_stop", "hivemind.vm.force_stop@v1", ("vm-1",), {"ok": True}, {"name": "vm-1"}),
        ("vm_delete", "hivemind.vm.delete@v1", ("vm-1",), {"ok": True}, {"name": "vm-1"}),
        ("vm_screenshot", "hivemind.vm.screenshot@v1", ("vm-1",), {"format": "png", "data_base64": "abc"}, {"name": "vm-1", "width": 1280, "height": 720}),
        # storage
        ("storage_status", "hivemind.storage.status@v1", (), {"healthy": True}, {}),
        ("storage_pools", "hivemind.storage.pools@v1", (), {"pools": []}, {}),
        ("storage_volumes", "hivemind.storage.volumes@v1", (), {"volumes": []}, {}),
        ("storage_snapshots", "hivemind.storage.snapshots@v1", (), {"snapshots": []}, {}),
        # network
        ("network_list", "hivemind.network.list@v1", (), {"networks": []}, {}),
        ("network_bridges", "hivemind.network.bridges@v1", (), {"bridges": []}, {}),
        ("network_interfaces", "hivemind.network.interfaces@v1", (), {"interfaces": []}, {}),
        # gpu_mode
        ("gpu_mode_capabilities", "hivemind.gpu_mode.capabilities@v1", (), {"modes": ["passthrough", "vgpu"]}, {}),
        ("gpu_availability", "hivemind.gpu.availability@v1", (), {"available": []}, {}),
        # voice_identities
        ("voice_identities_list", "hivemind.voice_identities.list@v1", (), {"identities": []}, {}),
        # human
        ("human_telegram_poll", "hivemind.human.telegram.poll@v1", (), {"messages": []}, {}),
        # services maintenance
        ("services_maintenance_clear", "hivemind.services.maintenance.clear@v1", ("ASR",), {"ok": True}, {"service_name": "ASR"}),
        # time + models
        ("time_now", "hivemind.time.now@v1", (), {"iso": "2026-05-26T12:00:00Z"}, {}),
        ("models_list", "hivemind.models.list@v1", (), {"data": []}, {}),
        # capability matrix
        ("capability_matrix", "hivemind.capability.matrix@v1", (), {"nodes": []}, {}),
        # files / search / web / http
        ("files_read", "hivemind.files.read@v1", ("/etc/hosts",), {"text": "127.0.0.1 localhost"}, {"path": "/etc/hosts"}),
        ("ollama_tags", "hivemind.ollama.tags@v1", (), {"models": []}, {}),
        # crown
        ("crown_status", "hivemind.crown.status@v1", (), {"connected": True}, {}),
        ("crown_latest", "hivemind.crown.latest@v1", (), {"event": None}, {}),
        ("crown_signal_quality", "hivemind.crown.signal_quality@v1", (), {"score": 0.92}, {}),
        # api keys
        ("api_keys_status", "hivemind.api_keys.status@v1", (), {"keys": []}, {}),
    ],
)
def test_wrapper_hits_correct_tool(fake_mcp, fn_name, tool_name, args, expected_response, call_args):
    fake_mcp.set_response(tool_name, expected_response)
    fn = getattr(hivemind_tools, fn_name)
    result = fn(hm_url(fake_mcp), *args)
    assert result == expected_response
    last = fake_mcp.calls[-1]["body"]
    assert last["params"]["name"] == tool_name, f"wrong tool: {last['params']['name']}"
    assert last["params"]["arguments"] == call_args, f"wrong arguments: {last['params']['arguments']}"


def test_iserror_surfaces_as_hivemind_tool_error(fake_mcp):
    fake_mcp.set_error("hivemind.vm.list@v1", "cluster offline")
    with pytest.raises(HivemindToolError) as exc:
        hivemind_tools.vm_list(hm_url(fake_mcp))
    assert "cluster offline" in str(exc.value)


def test_transport_failure_surfaces_as_hivemind_tool_error(monkeypatch):
    monkeypatch.setenv("MS4_HIVEMIND_MCP_URL", "http://127.0.0.1:1/mcp")
    # Use time_now() which accepts a timeout — vm_list doesn't expose
    # a per-call timeout, the underlying transport defaults to 10s.
    # Either function exercises the same transport-failure path.
    with pytest.raises(HivemindToolError):
        hivemind_tools.time_now("http://127.0.0.1:1")


def test_auth_header_attached_when_env_set(fake_mcp, monkeypatch):
    monkeypatch.setenv("MS4_HIVEMIND_API_KEY", "tok-abc")
    fake_mcp.set_response("hivemind.time.now@v1", {"iso": "ok"})
    hivemind_tools.time_now(hm_url(fake_mcp))
    headers = fake_mcp.calls[-1]["headers"]
    assert any(k.lower() == "authorization" and "Bearer tok-abc" in v for k, v in headers.items())


def test_is_service_in_maintenance_fail_soft(monkeypatch):
    """Even when service_health is unreachable, is_service_in_maintenance
    returns False rather than blocking the caller. Keeps voice_admin's
    maintenance check non-fatal."""
    monkeypatch.setenv("MS4_HIVEMIND_MCP_URL", "http://127.0.0.1:1/mcp")
    assert hivemind_tools.is_service_in_maintenance("http://127.0.0.1:1", "ASR") is False


# ---------------------------------------------------------------------------
# Compose helpers: voice_identities.enroll/refine/identify do a
# base64-encode step before hitting the tool. Lock that in so a future
# refactor can't accidentally regress to passing raw bytes.
# ---------------------------------------------------------------------------


def test_voice_identities_enroll_base64_encodes_audio(fake_mcp):
    fake_mcp.set_response("hivemind.voice_identities.enroll@v1", {"identity_id": "id-1"})
    body = hivemind_tools.voice_identities_enroll(
        hm_url(fake_mcp), audio_base64="abcd==", name="Alice"
    )
    assert body["identity_id"] == "id-1"
    args = fake_mcp.calls[-1]["body"]["params"]["arguments"]
    assert args["audio_base64"] == "abcd=="
    assert args["name"] == "Alice"


def test_models_recommend_passes_workload_and_constraints(fake_mcp):
    fake_mcp.set_response(
        "hivemind.models.recommend@v1",
        {"recommended": [{"model_id": "phi4-mini:latest", "score": 0.9, "reason": "fast + tool-use"}]},
    )
    body = hivemind_tools.models_recommend(
        hm_url(fake_mcp),
        workload="foreground_chat_small",
        constraints={"max_size_b": 8, "instruct": True},
    )
    assert body["recommended"][0]["model_id"] == "phi4-mini:latest"
    args = fake_mcp.calls[-1]["body"]["params"]["arguments"]
    assert args["workload"] == "foreground_chat_small"
    assert args["constraints"] == {"max_size_b": 8, "instruct": True}


def test_human_approval_request_passes_summary_and_risk(fake_mcp):
    fake_mcp.set_response(
        "hivemind.human.approval.request@v1",
        {"request_id": "req-1", "status": "pending"},
    )
    body = hivemind_tools.human_approval_request(
        hm_url(fake_mcp),
        action_id="act-1",
        summary="restart asr_gim",
        risk_level="high",
        details={"node": "dgx-0"},
    )
    assert body["request_id"] == "req-1"
    args = fake_mcp.calls[-1]["body"]["params"]["arguments"]
    assert args["action_id"] == "act-1"
    assert args["risk_level"] == "high"
    assert args["details"] == {"node": "dgx-0"}


def test_vlm_chat_passes_messages(fake_mcp):
    fake_mcp.set_response(
        "hivemind.vlm.chat@v1",
        {"choices": [{"message": {"content": "a cat"}}]},
    )
    msgs = [{"role": "user", "content": [{"type": "text", "text": "what?"}]}]
    body = hivemind_tools.vlm_chat(hm_url(fake_mcp), messages=msgs, model="qwen3-vl:4b")
    assert body["choices"][0]["message"]["content"] == "a cat"
    args = fake_mcp.calls[-1]["body"]["params"]["arguments"]
    assert args["messages"] == msgs
    assert args["model"] == "qwen3-vl:4b"


# ---------------------------------------------------------------------------
# PsyKyo game-session benchmark bridge + moonlight (June 2026) — the
# wrappers the Depth Lobe gamestream_benchmark plan template drives.
# ---------------------------------------------------------------------------


def test_psykyo_gs_list_capable_gpus(fake_mcp):
    fake_mcp.set_response(
        "hivemind.psykyo.game_session.list_capable_gpus@v1",
        {"gpus": [{"uuid": "GPU-abc", "host": "desktop-1"}]},
    )
    body = hivemind_tools.psykyo_game_session_list_capable_gpus(
        hm_url(fake_mcp), host_filter="desktop-1"
    )
    assert body["gpus"][0]["uuid"] == "GPU-abc"
    last = fake_mcp.calls[-1]["body"]["params"]
    assert last["name"] == "hivemind.psykyo.game_session.list_capable_gpus@v1"
    assert last["arguments"] == {"host_filter": "desktop-1"}


def test_psykyo_gs_start_benchmark_forces_benchmark_mode(fake_mcp):
    fake_mcp.set_response(
        "hivemind.psykyo.game_session.start_benchmark@v1",
        {"job_id": "gs-1", "plan": {"job_id": "gs-1"}},
    )
    body = hivemind_tools.psykyo_game_session_start_benchmark(
        hm_url(fake_mcp),
        game="cyberpunk-2077",
        benchmark_runs=3,
        requires_stream=False,
        gpu_uuid="GPU-abc",
        host="desktop-1",
        client="ms4-depth-lobe",
    )
    assert body["job_id"] == "gs-1"
    last = fake_mcp.calls[-1]["body"]["params"]
    assert last["name"] == "hivemind.psykyo.game_session.start_benchmark@v1"
    args = last["arguments"]
    assert args["game"] == "cyberpunk-2077"
    assert args["mode"] == "benchmark"
    assert args["benchmark_runs"] == 3
    assert args["requires_stream"] is False
    assert args["gpu_uuid"] == "GPU-abc"
    assert args["host"] == "desktop-1"
    assert args["client"] == "ms4-depth-lobe"


def test_psykyo_gs_start_benchmark_requires_game(fake_mcp):
    with pytest.raises(ValueError):
        hivemind_tools.psykyo_game_session_start_benchmark(hm_url(fake_mcp), game="")


def test_psykyo_gs_run_benchmark_one_shot(fake_mcp):
    fake_mcp.set_response(
        "hivemind.psykyo.game_session.run_benchmark@v1",
        {"final_state": "COMPLETE", "benchmark_summary": {"avg_fps": 100.0}},
    )
    body = hivemind_tools.psykyo_game_session_run_benchmark(
        hm_url(fake_mcp), game="cp2077", benchmark_runs=2
    )
    assert body["final_state"] == "COMPLETE"
    last = fake_mcp.calls[-1]["body"]["params"]
    assert last["name"] == "hivemind.psykyo.game_session.run_benchmark@v1"
    assert last["arguments"]["mode"] == "benchmark"
    assert last["arguments"]["benchmark_runs"] == 2


def test_psykyo_gs_get_results_and_stop_session(fake_mcp):
    fake_mcp.set_response(
        "hivemind.psykyo.game_session.get_results@v1",
        {"benchmark_summary": {"avg_fps": 99.9}},
    )
    body = hivemind_tools.psykyo_game_session_get_results(hm_url(fake_mcp), job_id="gs-1")
    assert body["benchmark_summary"]["avg_fps"] == 99.9
    assert fake_mcp.calls[-1]["body"]["params"]["arguments"] == {"job_id": "gs-1"}

    fake_mcp.set_response(
        "hivemind.psykyo.game_session.stop_session@v1", {"state": "CANCELLED"}
    )
    body = hivemind_tools.psykyo_game_session_stop_session(hm_url(fake_mcp), job_id="gs-1")
    assert body["state"] == "CANCELLED"
    last = fake_mcp.calls[-1]["body"]["params"]
    assert last["name"] == "hivemind.psykyo.game_session.stop_session@v1"
    assert last["arguments"] == {"job_id": "gs-1"}

    with pytest.raises(ValueError):
        hivemind_tools.psykyo_game_session_get_results(hm_url(fake_mcp), job_id="")


def test_moonlight_stream_passes_verify_seconds(fake_mcp):
    fake_mcp.set_response("hivemind.moonlight.stream@v1", {"ok": True, "verified": True})
    body = hivemind_tools.moonlight_stream(
        hm_url(fake_mcp),
        host="192.168.1.50",
        app="Desktop",
        width=1920,
        height=1080,
        fps=60,
        verify_seconds=10,
    )
    assert body["ok"] is True
    last = fake_mcp.calls[-1]["body"]["params"]
    assert last["name"] == "hivemind.moonlight.stream@v1"
    args = last["arguments"]
    assert args == {
        "host": "192.168.1.50",
        "app": "Desktop",
        "width": 1920,
        "height": 1080,
        "fps": 60,
        "verify_seconds": 10,
    }

    with pytest.raises(ValueError):
        hivemind_tools.moonlight_stream(hm_url(fake_mcp), host="")
