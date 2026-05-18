from __future__ import annotations

import json
from pathlib import Path

import machine_spirit_4.mcp.tools as mcp_tools
from machine_spirit_4.mcp.server import handle_jsonrpc
from machine_spirit_4.mcp.tools import build_tool_registry


ROOT = Path(__file__).resolve().parents[2]


class FakeRunner:
    def sessions(self):
        return []


class FakeRuntime:
    def __init__(self) -> None:
        self.runner = FakeRunner()
        self.ms3_url = "http://ms3"
        self.ethics_payloads = []

    def post_json(self, url, payload, timeout=30):
        if url.endswith("/ethics/evaluate"):
            self.ethics_payloads.append(payload)
            return 200, {"schema": "EthicsDecision.v1", "decision": "allow"}
        raise AssertionError(url)


class FakeController:
    def status(self):
        return {"schema": "Ms4DesktopStatus.v1"}

    def capture(self, arguments):
        return {"schema": "Ms4DesktopCapture.v1", "read_only": True}

    def act(self, arguments):
        return {"schema": "Ms4DesktopActionResult.v1", "ok": True, "action": arguments["action"]}


def test_gateway_asset_exposes_desktop_routes():
    server = (ROOT / "machine_spirit_4" / "gateway" / "server.py").read_text(encoding="utf-8")

    assert '"/desktop/status"' in server
    assert '"/desktop/capture"' in server
    assert '"/desktop/action"' in server


def test_mcp_registry_exposes_desktop_tools():
    registry = build_tool_registry()

    assert "ms4.desktop.status@v1" in registry
    assert "ms4.desktop.capture@v1" in registry
    assert "ms4.desktop.action@v1" in registry


def test_desktop_action_tool_runs_ethics_before_action(monkeypatch):
    monkeypatch.setenv("MS4_DESKTOP_CONTROL", "1")
    runtime = FakeRuntime()
    response = handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": 8,
            "method": "tools/call",
            "params": {
                "name": "ms4.desktop.action@v1",
                "arguments": {"action": "wait", "seconds": 0},
            },
        },
        runtime,
    )

    assert response["result"]["isError"] is False
    assert runtime.ethics_payloads
    assert runtime.ethics_payloads[0]["schema"] == "ActionIntent.v1"
    assert runtime.ethics_payloads[0]["action_type"] == "desktop_ui"
    assert runtime.ethics_payloads[0]["risk_class"] == "low"
    assert runtime.ethics_payloads[0]["requires_safety_clearance"] is False
    assert "Ms4DesktopActionResult.v1" in response["result"]["content"][0]["text"]


def test_desktop_type_action_is_ethics_evaluated_without_physical_safety_clearance(monkeypatch):
    monkeypatch.setenv("MS4_DESKTOP_CONTROL", "1")
    monkeypatch.setattr(mcp_tools, "desktop_controller", lambda: FakeController())
    runtime = FakeRuntime()
    response = handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": 9,
            "method": "tools/call",
            "params": {
                "name": "ms4.desktop.action@v1",
                "arguments": {"action": "type", "text": "MS4 desktop smoke"},
            },
        },
        runtime,
    )

    assert response["result"]["isError"] is False
    assert runtime.ethics_payloads[0]["risk_class"] == "medium"
    assert runtime.ethics_payloads[0]["requires_safety_clearance"] is False


def test_manifest_lists_desktop_tools():
    manifest = json.loads((ROOT / "machine_spirit_4" / "mcp" / "manifest.json").read_text(encoding="utf-8"))
    names = {tool["name"] for tool in manifest["tools"]}

    assert "ms4.desktop.status@v1" in names
    assert "ms4.desktop.capture@v1" in names
    assert "ms4.desktop.action@v1" in names
