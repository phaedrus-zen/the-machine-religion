from __future__ import annotations

import json
from types import SimpleNamespace

from machine_spirit_4.mcp.server import handle_jsonrpc, service_info
from machine_spirit_4.mcp.tools import build_tool_registry


class FakeRunner:
    def health(self):
        return {"runtime": "ms4-fusion", "plugin": {"enabled": True}}

    def sessions(self):
        return [{"session_id": "s1", "turns": 2}]

    def chat(self, message, *, session_id=None, model=None):
        return {
            "text": f"answer:{message}",
            "session_id": session_id or "s1",
            "runtime": "hermes",
            "model": model or "default",
            "grounding_source": "test",
        }

    def list_hermes_tools(self, enabled_toolsets=None):
        return [{"name": "terminal", "toolset": "terminal", "description": "Execute commands"}]

    def dispatch_hermes_tool(self, tool_name, args=None, *, session_id=None):
        return {"tool": tool_name, "args": args or {}, "result": "{\"output\":\"MS4_HERMES_OK\"}", "runtime": "hermes"}


class FakeRuntime:
    def __init__(self):
        self.runner = FakeRunner()
        self.ms3_url = "http://ms3"
        self.hivemind_url = "http://hive"

    def get_json(self, url, timeout=20):
        if url.endswith("/identity/verify"):
            return 200, {"identity_confirmed": True, "spirit_id": "sister"}
        if url.endswith("/state"):
            return 200, {"identity": {"name": "Sister"}, "cognitive_load": 0.2}
        if url.endswith("/voice/status"):
            return 200, {"schema": "VoiceReadiness.v1", "voice_input_ready": False}
        if url.endswith("/models"):
            return 200, {"models": [{"id": "qwen3-coder-next:latest"}]}
        raise AssertionError(url)

    def post_json(self, url, payload, timeout=30):
        if url.endswith("/ethics/evaluate"):
            return 200, {"schema": "EthicsDecision.v1", "decision": "allow", "payload": payload}
        raise AssertionError(url)

    def mcp_call(self, name, arguments, timeout=30):
        if name == "hivemind.cluster.summary@v1":
            return json.dumps({"cluster_statistics": {"total_nodes": 1, "active_nodes": 1, "total_gpus": 1}})
        if name == "hivemind.hosts.list@v1":
            return json.dumps({
                "nodes": [
                    {
                        "name": "node-a",
                        "status": "active",
                        "ip_addresses": ["1.2.3.4"],
                        "hardware": {
                            "devices": {
                                "gpu": {
                                    "compute_device_type": "GPU",
                                    "manufacturer": "NVIDIA",
                                    "device_name": "RTX",
                                }
                            }
                        },
                    }
                ]
            })
        raise AssertionError(name)


def test_initialize_returns_mcp_capabilities():
    response = handle_jsonrpc({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}, FakeRuntime())

    assert response["result"]["protocolVersion"] == "2025-11-25"
    assert response["result"]["serverInfo"]["name"] == "ms4-mcp-server"
    assert response["result"]["capabilities"]["tools"]["listChanged"] is False


def test_tools_list_contains_ms4_tools_with_annotations():
    response = handle_jsonrpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}, FakeRuntime())
    tools = response["result"]["tools"]
    names = {tool["name"] for tool in tools}

    assert "ms4.identity.verify@v1" in names
    assert "ms4.chat.send@v1" in names
    identity_tool = next(tool for tool in tools if tool["name"] == "ms4.identity.verify@v1")
    assert identity_tool["annotations"]["readOnlyHint"] is True
    assert identity_tool["annotations"]["destructiveHint"] is False


def test_tools_call_dispatches_chat_send():
    response = handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "ms4.chat.send@v1",
                "arguments": {"message": "hello", "session_id": "s1", "model": "m"},
            },
        },
        FakeRuntime(),
    )

    assert response["result"]["isError"] is False
    text = response["result"]["content"][0]["text"]
    assert "answer:hello" in text
    assert "hermes" in text


def test_unknown_tool_returns_jsonrpc_error():
    response = handle_jsonrpc(
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "nope", "arguments": {}}},
        FakeRuntime(),
    )

    assert response["error"]["code"] == -32602


def test_registry_has_initial_v1_tool_count():
    registry = build_tool_registry()

    # Original v1: 27. May-25 2026 HiveMind expansion: +16 hivemind.*
    # proxy tools. May-26 PsyKyo bridge expansion: +5 typed wrappers.
    # May-26 fill-all-gaps round: +17 admin proxies. May-26 game-
    # session round: +6 game/game_session proxies. Bump when adding /
    # removing tools.
    # See test_manifest.py for the running count justification.
    assert len(registry) == 71
    assert "ms4.vision.analyze_local@v1" in registry
    assert "ms4.hermes.version@v1" in registry
    assert "ms4.hermes.update@v1" in registry
    assert "ms4.double_agent.submit@v1" in registry
    assert "ms4.double_agent.list@v1" in registry
    # New HiveMind proxies (smoke check — the per-tool wrapper tests
    # cover behaviour, this just locks them into the registry).
    assert "ms4.hivemind.vms@v1" in registry
    assert "ms4.hivemind.apps@v1" in registry
    assert "ms4.hivemind.voice_identities@v1" in registry
    assert "ms4.hivemind.approval.request@v1" in registry
    assert "ms4.hivemind.time@v1" in registry
    for psykyo_tool in (
        "ms4.hivemind.psykyo.benchmark.run@v1",
        "ms4.hivemind.psykyo.benchmark.gap@v1",
        "ms4.hivemind.psykyo.benchmark.workqueue@v1",
        "ms4.hivemind.psykyo.evidence.latest@v1",
        "ms4.hivemind.psykyo.vlm_consensus@v1",
    ):
        assert psykyo_tool in registry


def test_service_info_matches_mcp_get_contract():
    info = service_info()

    assert info["service"] == "ms4-mcp-server"
    assert info["mcp_protocol_version"] == "2025-11-25"
    assert info["status"] == "ready"
    assert info["tools_loaded"] == 71


def test_hermes_tools_list_and_call_dispatch():
    runtime = FakeRuntime()
    listed = handle_jsonrpc(
        {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "ms4.hermes.tools.list@v1", "arguments": {}}},
        runtime,
    )
    called = handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {
                "name": "ms4.hermes.tool.call@v1",
                "arguments": {"tool": "terminal", "args": {"command": "echo MS4_HERMES_OK"}},
            },
        },
        runtime,
    )

    assert "terminal" in listed["result"]["content"][0]["text"]
    assert "MS4_HERMES_OK" in called["result"]["content"][0]["text"]


def test_runtime_deps_status_tool_reports_schema():
    response = handle_jsonrpc(
        {"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {"name": "ms4.runtime.deps.status@v1", "arguments": {}}},
        FakeRuntime(),
    )

    assert response["result"]["isError"] is False
    assert "Ms4DependencyStatus.v1" in response["result"]["content"][0]["text"]
