"""Unit + route + MCP + wrapper-shape tests for GPU passthrough.

Covers the May 27 2026 GPU-P / DDA / vGPU work:

1. **Wrapper shape regressions** — locks in the corrected wire shape
   for vm.* and gpu_mode.* (vm uses ``name`` not ``vm_id``; gpu_mode
   uses ``gpu_pci_id``/``desired_mode``/``count``). These tests will
   fail loudly if a future refactor reverts to the legacy shape.
2. **gpu_passthrough admin module** — combined snapshot + prepare +
   vgpu_create + game_stream_vm happy + error paths.
3. **REST routes** — full set under /hivemind/gpu/passthrough/*.
4. **MS4 MCP proxies** — registry coverage + handler dispatch.
5. **UI asset** — Settings panel DOM ids + JS function names.

Mirrors the in-process MCP fake pattern from the other admin tests.
"""

from __future__ import annotations

import io
import json
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from machine_spirit_4.gateway import (
    gpu_mode_admin,
    gpu_passthrough,
    hivemind_tools,
    server as srv_module,
    vm_admin,
)
from machine_spirit_4.gateway.gpu_passthrough import GpuPassthroughError
from machine_spirit_4.mcp import tools as mcp_tools


# ---------------------------------------------------------------------------
# In-process fake MCP server
# ---------------------------------------------------------------------------


class _FakeMcp:
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
                outer.calls.append({"body": body})
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


def hurl(fake: _FakeMcp) -> str:
    return f"http://127.0.0.1:{fake.port}"


def _last_args(fake: _FakeMcp) -> dict[str, Any]:
    return fake.calls[-1]["body"]["params"]["arguments"]


# ===========================================================================
# Wrapper shape regressions — locks in the May 27 2026 wire contract
# ===========================================================================


def test_vm_start_passes_name_not_vm_id(fake_mcp):
    fake_mcp.set_response("hivemind.vm.start@v1", {"ok": True})
    hivemind_tools.vm_start(hurl(fake_mcp), "cyberpunk-stream-01")
    args = _last_args(fake_mcp)
    assert args == {"name": "cyberpunk-stream-01"}
    assert "vm_id" not in args


def test_vm_stop_force_stop_delete_undeploy_pass_name(fake_mcp):
    """All four destructive-ish ops must use ``name`` on the wire."""
    for tool, fn in [
        ("hivemind.vm.stop@v1", hivemind_tools.vm_stop),
        ("hivemind.vm.force_stop@v1", hivemind_tools.vm_force_stop),
        ("hivemind.vm.delete@v1", hivemind_tools.vm_delete),
        ("hivemind.vm.undeploy@v1", hivemind_tools.vm_undeploy),
    ]:
        fake_mcp.set_response(tool, {"ok": True})
        fn(hurl(fake_mcp), "vmX")
        args = _last_args(fake_mcp)
        assert args == {"name": "vmX"}, f"{tool} sent wrong args: {args}"


def test_vm_deploy_uses_name_no_target_node_required(fake_mcp):
    fake_mcp.set_response("hivemind.vm.deploy@v1", {"ok": True})
    hivemind_tools.vm_deploy(hurl(fake_mcp), "vmX")
    args = _last_args(fake_mcp)
    assert args == {"name": "vmX"}
    # opts are forwarded transparently for forward-compat:
    hivemind_tools.vm_deploy(hurl(fake_mcp), "vmY", target_node="hostA")
    args = _last_args(fake_mcp)
    assert args == {"name": "vmY", "target_node": "hostA"}


def test_vm_screenshot_passes_name_width_height(fake_mcp):
    fake_mcp.set_response("hivemind.vm.screenshot@v1", {"format": "png"})
    hivemind_tools.vm_screenshot(hurl(fake_mcp), "vmX")
    args = _last_args(fake_mcp)
    assert args == {"name": "vmX", "width": 1280, "height": 720}
    hivemind_tools.vm_screenshot(hurl(fake_mcp), "vmY", width=640, height=360)
    args = _last_args(fake_mcp)
    assert args == {"name": "vmY", "width": 640, "height": 360}


def test_vm_create_prebuilt_requires_vm_type_and_name(fake_mcp):
    fake_mcp.set_response("hivemind.vm.create_prebuilt@v1", {"ok": True})
    hivemind_tools.vm_create_prebuilt(
        hurl(fake_mcp),
        "windows_game_stream_prebuilt",
        "cyberpunk-stream-01",
    )
    args = _last_args(fake_mcp)
    assert args["vm_type"] == "windows_game_stream_prebuilt"
    assert args["name"] == "cyberpunk-stream-01"
    assert "template" not in args, "legacy 'template' key must NOT leak through"


def test_vm_gpus_passes_no_args(fake_mcp):
    """May-26 contract: vm.gpus takes no arguments. Earlier MS4
    wrappers passed an optional ``vm_id`` filter — lock that out."""
    fake_mcp.set_response("hivemind.vm.gpus@v1", {"gpus": []})
    hivemind_tools.vm_gpus(hurl(fake_mcp))
    args = _last_args(fake_mcp)
    assert args == {}


def test_gpu_mode_set_passes_pci_id_and_desired_mode(fake_mcp):
    fake_mcp.set_response("hivemind.gpu_mode.set@v1", {"ok": True})
    hivemind_tools.gpu_mode_set(
        hurl(fake_mcp),
        gpu_pci_id="0000:01:00.0",
        desired_mode="passthrough",
    )
    args = _last_args(fake_mcp)
    assert args == {"gpu_pci_id": "0000:01:00.0", "desired_mode": "passthrough"}
    # node_id / gpu_id (legacy) must NOT appear:
    assert "node_id" not in args
    assert "gpu_id" not in args
    assert "mode" not in args


def test_gpu_mode_set_rejects_invalid_mode():
    with pytest.raises(ValueError):
        hivemind_tools.gpu_mode_set(
            "http://x", gpu_pci_id="0000:01:00.0", desired_mode="gpu_p"
        )


def test_gpu_mode_vgpu_create_passes_pci_id_profile_count(fake_mcp):
    fake_mcp.set_response("hivemind.gpu_mode.vgpu_create@v1", {"ok": True})
    hivemind_tools.gpu_mode_vgpu_create(
        hurl(fake_mcp), gpu_pci_id="0000:01:00.0", profile="nvidia-256", count=4
    )
    args = _last_args(fake_mcp)
    assert args == {"gpu_pci_id": "0000:01:00.0", "profile": "nvidia-256", "count": 4}


def test_gpu_mode_vgpu_create_rejects_count_out_of_range():
    with pytest.raises(ValueError):
        hivemind_tools.gpu_mode_vgpu_create(
            "http://x", gpu_pci_id="0000:01:00.0", profile="nvidia-256", count=0
        )
    with pytest.raises(ValueError):
        hivemind_tools.gpu_mode_vgpu_create(
            "http://x", gpu_pci_id="0000:01:00.0", profile="nvidia-256", count=100
        )


def test_gpu_mode_vgpu_status_takes_no_args(fake_mcp):
    fake_mcp.set_response("hivemind.gpu_mode.vgpu_status@v1", {"vgpus": []})
    hivemind_tools.gpu_mode_vgpu_status(hurl(fake_mcp))
    args = _last_args(fake_mcp)
    assert args == {}


# ===========================================================================
# gpu_passthrough admin module
# ===========================================================================


def test_snapshot_combines_all_four_reads(fake_mcp):
    fake_mcp.set_response("hivemind.gpu_mode.capabilities@v1", {"modes_supported": ["passthrough", "vgpu"]})
    fake_mcp.set_response("hivemind.vm.gpus@v1", {"gpus": [{"pci_id": "0000:01:00.0", "vendor": "NVIDIA"}]})
    fake_mcp.set_response("hivemind.gpu_mode.vgpu_status@v1", {"healthy": True})
    fake_mcp.set_response("hivemind.gpu.availability@v1", {"free_gpus": 1})
    snap = gpu_passthrough.snapshot(hurl(fake_mcp))
    assert snap["schema"] == "Ms4GpuPassthroughSnapshot.v1"
    assert snap["capabilities"]["modes_supported"] == ["passthrough", "vgpu"]
    assert snap["vm_visible_gpus"]["gpus"][0]["pci_id"] == "0000:01:00.0"
    assert snap["vgpu_stack_status"]["healthy"] is True
    assert snap["availability"]["free_gpus"] == 1
    assert snap["errors"] == []
    # Mode metadata always populated even when sub-calls failed:
    assert "gpu_p" in snap["modes"]
    assert "dda" in snap["modes"]
    assert "vgpu" in snap["modes"]
    assert "Windows Server" in snap["modes"]["dda"]["needs"]


def test_snapshot_fail_soft_per_subcall(fake_mcp):
    """One subcall errors -> snapshot still returns; error captured."""
    fake_mcp.set_response("hivemind.gpu_mode.capabilities@v1", {"x": 1})
    fake_mcp.set_error("hivemind.vm.gpus@v1", "vm_manager unreachable")
    fake_mcp.set_response("hivemind.gpu_mode.vgpu_status@v1", {"healthy": True})
    fake_mcp.set_response("hivemind.gpu.availability@v1", {"free_gpus": 0})
    snap = gpu_passthrough.snapshot(hurl(fake_mcp))
    assert snap["capabilities"] == {"x": 1}
    assert snap["vm_visible_gpus"] is None
    assert any("vm.gpus" in e for e in snap["errors"])


def test_prepare_mode_requires_confirm():
    with pytest.raises(ValueError, match="confirm=True"):
        gpu_passthrough.prepare_mode(
            "http://x", gpu_pci_id="0000:01:00.0", desired_mode="passthrough", confirm=False
        )


def test_prepare_mode_rejects_gpu_p_mode():
    """GPU-P is not a gpu_mode — must reject it with a clear hint."""
    with pytest.raises(ValueError, match="passthrough.*vgpu"):
        gpu_passthrough.prepare_mode(
            "http://x", gpu_pci_id="0000:01:00.0", desired_mode="gpu_p", confirm=True
        )


def test_prepare_mode_dda_intent(fake_mcp):
    fake_mcp.set_response("hivemind.gpu_mode.set@v1", {"ok": True})
    result = gpu_passthrough.prepare_mode(
        hurl(fake_mcp),
        gpu_pci_id="0000:01:00.0",
        desired_mode="passthrough",
        confirm=True,
    )
    assert result["schema"] == "Ms4GpuPassthroughAction.v1"
    assert result["intent"] == "dda_prepare"
    assert result["desired_mode"] == "passthrough"
    args = _last_args(fake_mcp)
    assert args == {"gpu_pci_id": "0000:01:00.0", "desired_mode": "passthrough"}


def test_prepare_mode_vgpu_intent(fake_mcp):
    fake_mcp.set_response("hivemind.gpu_mode.set@v1", {"ok": True})
    result = gpu_passthrough.prepare_mode(
        hurl(fake_mcp),
        gpu_pci_id="0000:01:00.0",
        desired_mode="vgpu",
        vm_uuid="abc-123",
        confirm=True,
    )
    assert result["intent"] == "vgpu_prepare"
    assert result["vm_uuid"] == "abc-123"
    args = _last_args(fake_mcp)
    assert args["vm_uuid"] == "abc-123"


def test_create_vgpu_requires_confirm():
    with pytest.raises(ValueError, match="confirm=True"):
        gpu_passthrough.create_vgpu(
            "http://x", gpu_pci_id="0000:01:00.0", profile="nvidia-256", confirm=False
        )


def test_create_vgpu_happy(fake_mcp):
    fake_mcp.set_response("hivemind.gpu_mode.vgpu_create@v1", {"uuids": ["a", "b"]})
    result = gpu_passthrough.create_vgpu(
        hurl(fake_mcp),
        gpu_pci_id="0000:01:00.0",
        profile="nvidia-256",
        count=2,
        confirm=True,
    )
    assert result["intent"] == "vgpu_create"
    assert result["count"] == 2
    assert result["uuids"] == ["a", "b"]


def test_create_game_stream_vm_requires_confirm():
    with pytest.raises(ValueError, match="confirm=True"):
        gpu_passthrough.create_game_stream_vm("http://x", name="vm-1", confirm=False)


def test_create_game_stream_vm_rejects_invalid_name():
    with pytest.raises(ValueError):
        gpu_passthrough.create_game_stream_vm("http://x", name="", confirm=True)


def test_create_game_stream_vm_happy_uses_prebuilt_template(fake_mcp):
    fake_mcp.set_response("hivemind.vm.create_prebuilt@v1", {"created": "cyberpunk-stream-01"})
    fake_mcp.set_response("hivemind.vm.deploy@v1", {"deployed": True})
    result = gpu_passthrough.create_game_stream_vm(
        hurl(fake_mcp), name="cyberpunk-stream-01", confirm=True
    )
    assert result["schema"] == "Ms4GpuPassthroughAction.v1"
    assert result["intent"] == "gpu_p_game_stream_vm"
    assert result["vm_type"] == "windows_game_stream_prebuilt"
    assert result["create"]["created"] == "cyberpunk-stream-01"
    assert result["deploy"]["deployed"] is True
    assert result["errors"] == []
    create_call = fake_mcp.calls[-2]["body"]["params"]
    deploy_call = fake_mcp.calls[-1]["body"]["params"]
    assert create_call["name"] == "hivemind.vm.create_prebuilt@v1"
    assert create_call["arguments"]["vm_type"] == "windows_game_stream_prebuilt"
    assert create_call["arguments"]["name"] == "cyberpunk-stream-01"
    assert deploy_call["name"] == "hivemind.vm.deploy@v1"
    assert deploy_call["arguments"]["name"] == "cyberpunk-stream-01"


def test_create_game_stream_vm_fail_soft_on_deploy(fake_mcp):
    """create succeeds but deploy errors -> create preserved + error
    captured. The operator can retry deploy manually."""
    fake_mcp.set_response("hivemind.vm.create_prebuilt@v1", {"created": "vm-x"})
    fake_mcp.set_error("hivemind.vm.deploy@v1", "hypervisor not ready")
    result = gpu_passthrough.create_game_stream_vm(
        hurl(fake_mcp), name="vm-x", confirm=True
    )
    assert result["create"]["created"] == "vm-x"
    assert result["deploy"] is None
    assert any("vm.deploy" in e for e in result["errors"])


# ===========================================================================
# REST routes
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
    _head, _, body = rest.partition(b"\r\n\r\n")
    return status, json.loads(body.decode("utf-8") or "{}")


def test_route_passthrough_snapshot(fake_mcp):
    fake_mcp.set_response("hivemind.gpu_mode.capabilities@v1", {})
    fake_mcp.set_response("hivemind.vm.gpus@v1", {"gpus": []})
    fake_mcp.set_response("hivemind.gpu_mode.vgpu_status@v1", {})
    fake_mcp.set_response("hivemind.gpu.availability@v1", {})
    handler = _make_handler(
        _DummyRunner(hurl(fake_mcp)), "GET", "/hivemind/gpu/passthrough/snapshot"
    )
    handler._hivemind_gpu_passthrough_snapshot()
    status, body = _read_response(handler)
    assert status == 200
    assert body["schema"] == "Ms4GpuPassthroughSnapshot.v1"
    assert "gpu_p" in body["modes"]


def test_route_prepare_requires_confirm(fake_mcp):
    handler = _make_handler(
        _DummyRunner(hurl(fake_mcp)),
        "POST",
        "/hivemind/gpu/passthrough/prepare",
        body=b'{"gpu_pci_id": "0000:01:00.0", "desired_mode": "passthrough"}',
    )
    handler._hivemind_gpu_passthrough_prepare()
    status, body = _read_response(handler)
    assert status == 400
    assert "confirm" in body["error"]


def test_route_prepare_rejects_gpu_p_mode(fake_mcp):
    handler = _make_handler(
        _DummyRunner(hurl(fake_mcp)),
        "POST",
        "/hivemind/gpu/passthrough/prepare",
        body=b'{"gpu_pci_id": "0000:01:00.0", "desired_mode": "gpu_p", "confirm": true}',
    )
    handler._hivemind_gpu_passthrough_prepare()
    status, body = _read_response(handler)
    assert status == 400
    assert "GPU-P" in body.get("note", "") or "passthrough" in body["error"]


def test_route_prepare_happy(fake_mcp):
    fake_mcp.set_response("hivemind.gpu_mode.set@v1", {"ok": True})
    handler = _make_handler(
        _DummyRunner(hurl(fake_mcp)),
        "POST",
        "/hivemind/gpu/passthrough/prepare",
        body=b'{"gpu_pci_id": "0000:01:00.0", "desired_mode": "passthrough", "confirm": true}',
    )
    handler._hivemind_gpu_passthrough_prepare()
    status, body = _read_response(handler)
    assert status == 200
    assert body["intent"] == "dda_prepare"


def test_route_vgpu_requires_confirm(fake_mcp):
    handler = _make_handler(
        _DummyRunner(hurl(fake_mcp)),
        "POST",
        "/hivemind/gpu/passthrough/vgpu",
        body=b'{"gpu_pci_id": "0000:01:00.0", "profile": "nvidia-256"}',
    )
    handler._hivemind_gpu_passthrough_vgpu()
    status, body = _read_response(handler)
    assert status == 400
    assert "confirm" in body["error"]


def test_route_game_stream_vm_requires_confirm(fake_mcp):
    handler = _make_handler(
        _DummyRunner(hurl(fake_mcp)),
        "POST",
        "/hivemind/gpu/passthrough/game-stream-vm",
        body=b'{"name": "cyberpunk-stream-01"}',
    )
    handler._hivemind_gpu_passthrough_game_stream_vm()
    status, body = _read_response(handler)
    assert status == 400
    assert "confirm" in body["error"]


def test_route_game_stream_vm_happy(fake_mcp):
    fake_mcp.set_response("hivemind.vm.create_prebuilt@v1", {"created": "cyberpunk-stream-01"})
    fake_mcp.set_response("hivemind.vm.deploy@v1", {"deployed": True})
    handler = _make_handler(
        _DummyRunner(hurl(fake_mcp)),
        "POST",
        "/hivemind/gpu/passthrough/game-stream-vm",
        body=b'{"name": "cyberpunk-stream-01", "confirm": true}',
    )
    handler._hivemind_gpu_passthrough_game_stream_vm()
    status, body = _read_response(handler)
    assert status == 200
    assert body["intent"] == "gpu_p_game_stream_vm"
    assert body["vm_type"] == "windows_game_stream_prebuilt"


def test_route_game_stream_vm_validates_name(fake_mcp):
    handler = _make_handler(
        _DummyRunner(hurl(fake_mcp)),
        "POST",
        "/hivemind/gpu/passthrough/game-stream-vm",
        body=b'{"name": "", "confirm": true}',
    )
    handler._hivemind_gpu_passthrough_game_stream_vm()
    status, body = _read_response(handler)
    assert status == 400


# ===========================================================================
# MS4 MCP proxies
# ===========================================================================


def test_mcp_registry_exposes_passthrough_proxies():
    reg = mcp_tools.build_tool_registry()
    expected = {
        "ms4.hivemind.gpu.passthrough.snapshot@v1",
        "ms4.hivemind.gpu.passthrough.prepare@v1",
        "ms4.hivemind.gpu.passthrough.vgpu@v1",
        "ms4.hivemind.gpu.passthrough.game_stream_vm@v1",
    }
    assert expected <= set(reg.keys())


def test_mcp_passthrough_snapshot_dispatches(fake_mcp):
    fake_mcp.set_response("hivemind.gpu_mode.capabilities@v1", {})
    fake_mcp.set_response("hivemind.vm.gpus@v1", {})
    fake_mcp.set_response("hivemind.gpu_mode.vgpu_status@v1", {})
    fake_mcp.set_response("hivemind.gpu.availability@v1", {})
    reg = mcp_tools.build_tool_registry()
    tool = reg["ms4.hivemind.gpu.passthrough.snapshot@v1"]
    runtime = types.SimpleNamespace(hivemind_url=hurl(fake_mcp))
    result = tool.handler(runtime, {})
    assert result["schema"] == "Ms4GpuPassthroughSnapshot.v1"


def test_mcp_passthrough_prepare_requires_confirm(fake_mcp):
    reg = mcp_tools.build_tool_registry()
    tool = reg["ms4.hivemind.gpu.passthrough.prepare@v1"]
    runtime = types.SimpleNamespace(hivemind_url=hurl(fake_mcp))
    with pytest.raises(ValueError, match="confirm"):
        tool.handler(runtime, {"gpu_pci_id": "0000:01:00.0", "desired_mode": "passthrough"})


def test_mcp_game_stream_vm_proxy(fake_mcp):
    fake_mcp.set_response("hivemind.vm.create_prebuilt@v1", {"created": "vm-x"})
    fake_mcp.set_response("hivemind.vm.deploy@v1", {"deployed": True})
    reg = mcp_tools.build_tool_registry()
    tool = reg["ms4.hivemind.gpu.passthrough.game_stream_vm@v1"]
    runtime = types.SimpleNamespace(hivemind_url=hurl(fake_mcp))
    result = tool.handler(runtime, {"name": "vm-x", "confirm": True})
    assert result["intent"] == "gpu_p_game_stream_vm"
    assert result["create"]["created"] == "vm-x"
