from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from machine_spirit_4.desktop import DesktopSafetyError
from machine_spirit_4.mcp import server as server_module
from machine_spirit_4.mcp.manifest_policy import load_manifest, validate_manifest
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
    chat_tool = next(tool for tool in tools if tool["name"] == "ms4.chat.send@v1")
    chat_policy = chat_tool["_meta"]["ms4/runtimeActionPolicy"]
    assert chat_policy["gated_by"] == ["none"]
    assert chat_policy["idempotency"]["mode"] == "none"
    assert chat_policy["cancellation"]["mode"] == "none"
    assert chat_policy["audit"]["mode"] == "required"
    assert set(chat_policy["action_states"]) == {
        "approval_required",
        "denied",
        "running",
        "committed",
        "cancelled",
        "reconciliation_required",
    }
    hermes_tool = next(tool for tool in tools if tool["name"] == "ms4.hermes.tool.call@v1")
    assert "confirm" in hermes_tool["inputSchema"]["required"]
    assert hermes_tool["inputSchema"]["properties"]["confirm"]["const"] is True
    gpu_prepare = next(
        tool for tool in tools if tool["name"] == "ms4.hivemind.gpu.passthrough.prepare@v1"
    )
    assert gpu_prepare["annotations"].get("destructiveHint") is not False


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
    legacy = json.loads(response["result"]["content"][0]["text"])
    action = response["result"]["structuredContent"]
    assert action["action_state"] == "committed"
    assert legacy["text"] == action["result"]["text"] == "answer:hello"
    assert legacy["runtime"] == action["result"]["runtime"] == "hermes"


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
    # session round: +6 game/game_session proxies. May-27 GPU
    # passthrough round: +4 hivemind.gpu.passthrough.* proxies. Bump
    # when adding / removing tools.
    # See test_manifest.py for the running count justification.
    assert len(registry) == 75
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
    assert info["tools_loaded"] == 75


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
                "arguments": {
                    "tool": "terminal",
                    "args": {"command": "echo MS4_HERMES_OK"},
                    "confirm": True,
                },
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


def _action_payload(response):
    return response["result"]["structuredContent"]


def _recording_registry(calls):
    registry = build_tool_registry()

    def handler_for(name):
        def handler(_runtime, _arguments):
            calls.append(name)
            return {"ok": True}

        return handler

    for name, tool in tuple(registry.items()):
        registry[name] = replace(tool, handler=handler_for(name))
    return registry


def test_every_declared_runtime_gate_is_enforced_before_handler(monkeypatch):
    policies = validate_manifest(load_manifest())
    actions = {
        name: policy for name, policy in policies.items() if policy.get("kind") == "runtime_action"
    }
    calls = []
    registry = _recording_registry(calls)
    monkeypatch.setattr(server_module, "build_tool_registry", lambda: registry)

    for request_id, (name, policy) in enumerate(sorted(actions.items()), start=100):
        calls.clear()
        gate = policy["gated_by"][0]
        response = server_module.tool_call_response(
            request_id,
            {"name": name, "arguments": {}},
            FakeRuntime(),
        )
        state = _action_payload(response)["action_state"]
        if gate == "none":
            assert calls == [name]
            assert state in {"committed", "cancelled"}
        elif gate == "confirm":
            assert calls == []
            assert state == "approval_required"
            confirmed = server_module.tool_call_response(
                request_id + 1000,
                {"name": name, "arguments": {"confirm": True}},
                FakeRuntime(),
            )
            assert calls == [name]
            assert _action_payload(confirmed)["action_state"] in {"committed", "cancelled"}
        else:
            assert gate == "deny_until_implemented"
            assert calls == []
            assert state == "denied"


def test_none_gate_is_bounded_before_handler(monkeypatch):
    calls = []
    registry = _recording_registry(calls)
    monkeypatch.setattr(server_module, "build_tool_registry", lambda: registry)

    response = server_module.tool_call_response(
        200,
        {
            "name": "ms4.chat.send@v1",
            "arguments": {"message": "x" * (1024 * 1024)},
        },
        FakeRuntime(),
    )
    action = _action_payload(response)

    assert calls == []
    assert action["action_state"] == "denied"
    assert action["reason_code"] == "arguments_too_large"


def test_open_inference_wrapper_keeps_fixed_timeout_and_declared_options(monkeypatch):
    from machine_spirit_4.gateway import hivemind_tools
    from machine_spirit_4.mcp import tools as mcp_tools

    captured = {}

    def inference_chat(url, *, messages, model, timeout=120, **options):
        captured.update(
            url=url,
            messages=messages,
            model=model,
            timeout=timeout,
            options=options,
        )
        return {"ok": True}

    monkeypatch.setattr(hivemind_tools, "inference_chat", inference_chat)
    result = mcp_tools.hivemind_inference_chat(
        FakeRuntime(),
        {
            "messages": [{"role": "user", "content": "hello"}],
            "model": "m",
            "max_tokens": 32,
            "temperature": 0.2,
            "timeout": 999999,
            "unexpected": "ignored",
        },
    )

    assert result == {"ok": True}
    assert captured["timeout"] == 120
    assert captured["options"] == {"temperature": 0.2, "max_tokens": 32}


def test_runtime_action_result_distinguishes_all_contract_states(monkeypatch):
    registry = build_tool_registry()
    monkeypatch.setattr(server_module, "build_tool_registry", lambda: registry)

    def set_handler(name, handler):
        registry[name] = replace(registry[name], handler=handler)

    set_handler("ms4.hermes.tool.call@v1", lambda _runtime, _args: {"ok": True})
    approval_required = server_module.tool_call_response(
        301,
        {"name": "ms4.hermes.tool.call@v1", "arguments": {}},
        FakeRuntime(),
    )
    set_handler("ms4.hivemind.vm.start@v1", lambda _runtime, _args: {"ok": True})
    denied = server_module.tool_call_response(
        302,
        {"name": "ms4.hivemind.vm.start@v1", "arguments": {}},
        FakeRuntime(),
    )

    set_handler("ms4.chat.send@v1", lambda _runtime, _args: {"status": "queued"})
    running = server_module.tool_call_response(
        303, {"name": "ms4.chat.send@v1", "arguments": {}}, FakeRuntime()
    )
    set_handler("ms4.chat.send@v1", lambda _runtime, _args: {"ok": True})
    committed = server_module.tool_call_response(
        304, {"name": "ms4.chat.send@v1", "arguments": {}}, FakeRuntime()
    )
    set_handler("ms4.chat.send@v1", lambda _runtime, _args: {"status": "cancelled"})
    cancelled = server_module.tool_call_response(
        305, {"name": "ms4.chat.send@v1", "arguments": {}}, FakeRuntime()
    )

    def uncertain(_runtime, _args):
        raise RuntimeError("transport failed after dispatch")

    set_handler("ms4.chat.send@v1", uncertain)
    reconciliation_required = server_module.tool_call_response(
        306, {"name": "ms4.chat.send@v1", "arguments": {}}, FakeRuntime()
    )

    states = {
        _action_payload(response)["action_state"]
        for response in (
            approval_required,
            denied,
            running,
            committed,
            cancelled,
            reconciliation_required,
        )
    }
    assert states == {
        "approval_required",
        "denied",
        "running",
        "committed",
        "cancelled",
        "reconciliation_required",
    }


def test_runtime_action_failed_substate_requires_reconciliation(monkeypatch):
    registry = build_tool_registry()
    registry["ms4.chat.send@v1"] = replace(
        registry["ms4.chat.send@v1"],
        handler=lambda _runtime, _args: {"state": "FAILED_EXECUTION"},
    )
    monkeypatch.setattr(server_module, "build_tool_registry", lambda: registry)

    response = server_module.tool_call_response(
        307, {"name": "ms4.chat.send@v1", "arguments": {}}, FakeRuntime()
    )

    assert _action_payload(response)["action_state"] == "reconciliation_required"


@pytest.mark.parametrize(
    ("result", "expected"),
    (
        ({"completed": False, "cancelled": False, "error": "upstream failed"}, "reconciliation_required"),
        ({"completed": False, "cancelled": True}, "cancelled"),
        (
            {"completed": False, "fail_closed": True, "error": "turn failed closed"},
            "reconciliation_required",
        ),
        ({"refused": True, "error": "not admitted", "result": None}, "denied"),
        ({"completed": True, "cancelled": False}, "committed"),
    ),
)
def test_runtime_action_maps_real_hermes_result_shapes(monkeypatch, result, expected):
    registry = build_tool_registry()
    registry["ms4.chat.send@v1"] = replace(
        registry["ms4.chat.send@v1"],
        handler=lambda _runtime, _args: result,
    )
    monkeypatch.setattr(server_module, "build_tool_registry", lambda: registry)

    response = server_module.tool_call_response(
        308, {"name": "ms4.chat.send@v1", "arguments": {}}, FakeRuntime()
    )

    assert _action_payload(response)["action_state"] == expected


def test_desktop_safety_denial_is_not_reported_as_uncertain(monkeypatch):
    from machine_spirit_4.mcp import tools as tools_module

    class BlockedDesktop:
        def act(self, _arguments):
            raise DesktopSafetyError("desktop control disabled")

    monkeypatch.setattr(tools_module, "desktop_controller", lambda: BlockedDesktop())
    response = server_module.tool_call_response(
        309,
        {
            "name": "ms4.desktop.action@v1",
            "arguments": {"action": "click", "confirm": True},
        },
        FakeRuntime(),
    )

    action = _action_payload(response)
    assert action["action_state"] == "denied"
    assert action["reason_code"] == "action_policy_denied"


def test_runtime_action_audit_is_required_and_secret_safe(monkeypatch, tmp_path):
    audit_path = tmp_path / "runtime-actions.jsonl"
    monkeypatch.setenv("MS4_AUDIT_LOG", str(audit_path))
    registry = build_tool_registry()
    sentinel = "DO_NOT_AUDIT_THIS_ARGUMENT_OR_RESULT"
    registry["ms4.chat.send@v1"] = replace(
        registry["ms4.chat.send@v1"],
        handler=lambda _runtime, _args: {"echo": sentinel},
    )
    monkeypatch.setattr(server_module, "build_tool_registry", lambda: registry)

    response = server_module.tool_call_response(
        400,
        {"name": "ms4.chat.send@v1", "arguments": {"message": sentinel}},
        FakeRuntime(),
    )
    audit_text = Path(audit_path).read_text(encoding="utf-8")
    events = [json.loads(line) for line in audit_text.splitlines()]

    assert _action_payload(response)["action_state"] == "committed"
    assert [event["phase"] for event in events] == ["admission", "outcome"]
    assert all(event["event_type"] == "ms4_mcp_runtime_action" for event in events)
    assert sentinel not in audit_text


def test_required_admission_audit_failure_denies_before_handler(monkeypatch):
    calls = []
    registry = _recording_registry(calls)
    monkeypatch.setattr(server_module, "build_tool_registry", lambda: registry)

    def fail_audit(*_args, **_kwargs):
        raise OSError("audit unavailable")

    monkeypatch.setattr(server_module.audit, "append_event", fail_audit)
    response = server_module.tool_call_response(
        401,
        {"name": "ms4.chat.send@v1", "arguments": {"message": "hello"}},
        FakeRuntime(),
    )
    action = _action_payload(response)

    assert calls == []
    assert action["action_state"] == "denied"
    assert action["reason_code"] == "required_audit_unavailable"


def test_required_outcome_audit_failure_requires_reconciliation(monkeypatch):
    calls = []
    registry = _recording_registry(calls)
    monkeypatch.setattr(server_module, "build_tool_registry", lambda: registry)
    real_append = server_module.audit.append_event

    def fail_outcome(event_type, data):
        if data["phase"] == "outcome":
            raise OSError("audit unavailable")
        return real_append(event_type, data)

    monkeypatch.setattr(server_module.audit, "append_event", fail_outcome)
    response = server_module.tool_call_response(
        402,
        {"name": "ms4.chat.send@v1", "arguments": {"message": "hello"}},
        FakeRuntime(),
    )
    action = _action_payload(response)

    assert calls == ["ms4.chat.send@v1"]
    assert action["action_state"] == "reconciliation_required"
    assert action["reason_code"] == "committed_result_not_audited"
