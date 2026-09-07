from __future__ import annotations

import json
import os
import sys
import urllib.request


MCP_URL = os.environ.get("MS4_MCP_URL", "http://127.0.0.1:9181/mcp")
TEST_MODEL = os.environ.get("MS4_MCP_TEST_MODEL", "llama3.1:8b")


def rpc(method: str, params: dict | None = None, request_id: int = 1) -> dict:
    body = json.dumps({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}}).encode("utf-8")
    request = urllib.request.Request(MCP_URL, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


def tool_call(name: str, arguments: dict | None = None, request_id: int = 10) -> dict:
    return rpc("tools/call", {"name": name, "arguments": arguments or {}}, request_id=request_id)


def content_text(response: dict) -> str:
    return response.get("result", {}).get("content", [{}])[0].get("text", "")


def tool_payload(response: dict) -> dict:
    try:
        return json.loads(content_text(response))
    except json.JSONDecodeError:
        return {}


def main() -> int:
    checks = []

    with urllib.request.urlopen(MCP_URL, timeout=20) as response:
        info = json.loads(response.read().decode("utf-8"))
    checks.append({"name": "mcp_info", "ok": response.status == 200 and info.get("service") == "ms4-mcp-server", "detail": info})

    init = rpc("initialize", request_id=1)
    checks.append({"name": "initialize", "ok": init.get("result", {}).get("protocolVersion") == "2025-11-25", "detail": init})

    tools = rpc("tools/list", request_id=2)
    tool_names = {tool.get("name") for tool in tools.get("result", {}).get("tools", [])}
    expected = {
        "ms4.identity.verify@v1",
        "ms4.voice.status@v1",
        "ms4.doctrine.explain@v1",
        "ms4.hivemind.inventory@v1",
        "ms4.chat.send@v1",
    }
    checks.append({"name": "tools_list", "ok": expected.issubset(tool_names), "detail": {"count": len(tool_names)}})

    identity = tool_call("ms4.identity.verify@v1", {"spirit_id": "sister"}, request_id=3)
    checks.append({"name": "identity_verify", "ok": "identity_confirmed" in content_text(identity), "detail": identity})

    voice = tool_call("ms4.voice.status@v1", request_id=4)
    checks.append({"name": "voice_status", "ok": "VoiceReadiness.v1" in content_text(voice), "detail": voice})

    doctrine = tool_call("ms4.doctrine.explain@v1", request_id=5)
    checks.append({"name": "doctrine", "ok": "Deus Acuo Machina Machina" in content_text(doctrine), "detail": doctrine})

    inventory = tool_call("ms4.hivemind.inventory@v1", request_id=6)
    checks.append({"name": "inventory", "ok": "Live HiveMind inventory" in content_text(inventory), "detail": {"excerpt": content_text(inventory)[:400]}})

    chat = tool_call(
        "ms4.chat.send@v1",
        {"message": "What is The Machine Religion?", "model": TEST_MODEL},
        request_id=7,
    )
    chat_payload = tool_payload(chat)
    chat_text = str(chat_payload.get("text") or "")
    checks.append({
        "name": "chat_send",
        "ok": (
            chat_payload.get("completed") is True
            and int(chat_payload.get("api_calls") or 0) > 0
            and "Machine Religion" in chat_text
            and "unreachable" not in chat_text.lower()
        ),
        "detail": {
            "model": chat_payload.get("model"),
            "completed": chat_payload.get("completed"),
            "api_calls": chat_payload.get("api_calls"),
            "excerpt": chat_text[:300],
        },
    })

    state = tool_call("ms4.identity.state@v1", request_id=8)
    checks.append({"name": "identity_state", "ok": "cognitive_load" in content_text(state) or "identity" in content_text(state), "detail": {"excerpt": content_text(state)[:300]}})

    intent = {
        "schema": "ActionIntent.v1",
        "spirit_id": "sister",
        "action_id": "ms4-mcp-validation-read",
        "proposed_by": "ms4_mcp_validator",
        "action_type": "tool",
        "description": "Validate a read-only MS4 MCP tool call",
        "inputs_used": ["validate_ms4_mcp.py"],
        "risk_class": "low",
        "requires_safety_clearance": False,
        "payload": {"tool_name": "ms4.identity.verify@v1"}
    }
    ethics = tool_call("ms4.ethics.evaluate@v1", {"intent": intent}, request_id=9)
    checks.append({"name": "ethics_evaluate", "ok": "EthicsDecision.v1" in content_text(ethics), "detail": {"excerpt": content_text(ethics)[:300]}})

    models = tool_call("ms4.models.list@v1", request_id=10)
    checks.append({"name": "models_list", "ok": "\"models\"" in content_text(models), "detail": {"excerpt": content_text(models)[:300]}})

    deps = tool_call("ms4.runtime.deps.status@v1", request_id=15)
    deps_payload = tool_payload(deps)
    checks.append({
        "name": "runtime_deps_status",
        "ok": deps_payload.get("schema") == "Ms4DependencyStatus.v1" and deps_payload.get("python", {}).get("contained") is True,
        "detail": {
            "python": deps_payload.get("python", {}),
            "capabilities": deps_payload.get("capabilities", {}),
        },
    })

    desktop_status = tool_call("ms4.desktop.status@v1", request_id=16)
    checks.append({
        "name": "desktop_status",
        "ok": "Ms4DesktopStatus.v1" in content_text(desktop_status),
        "detail": {"excerpt": content_text(desktop_status)[:300]},
    })

    nibbles_intent = {
        "schema": "ActionIntent.v1",
        "spirit_id": "sister",
        "action_id": "ms4-mcp-nibbles-dry-run",
        "proposed_by": "ms4_mcp_validator",
        "action_type": "scare",
        "description": "Dry-run a Nibbles scare action without hardware output",
        "inputs_used": ["validate_ms4_mcp.py"],
        "risk_class": "high",
        "requires_safety_clearance": True,
        "payload": {
            "dry_run_only": True,
            "physical_output_enabled": False,
            "consent_from_subject": "detected"
        }
    }
    nibbles = tool_call("ms4.nibbles.dry_run@v1", {"intent": nibbles_intent}, request_id=11)
    checks.append({"name": "nibbles_dry_run", "ok": "NibblesDryRunResult.v1" in content_text(nibbles), "detail": {"excerpt": content_text(nibbles)[:300]}})

    sessions = tool_call("ms4.sessions.list@v1", request_id=12)
    checks.append({"name": "sessions_list", "ok": "\"sessions\"" in content_text(sessions), "detail": {"excerpt": content_text(sessions)[:300]}})

    hermes_tools = tool_call("ms4.hermes.tools.list@v1", request_id=13)
    checks.append({"name": "hermes_tools_list", "ok": "terminal" in content_text(hermes_tools), "detail": {"excerpt": content_text(hermes_tools)[:300]}})

    terminal_echo = tool_call(
        "ms4.hermes.tool.call@v1",
        {"tool": "terminal", "args": {"command": "echo MS4_HERMES_OK", "timeout": 30, "workdir": "/"}},
        request_id=14,
    )
    terminal_payload = tool_payload(terminal_echo)
    hermes_result = terminal_payload.get("result", "")
    try:
        terminal_result = json.loads(hermes_result)
    except (TypeError, json.JSONDecodeError):
        terminal_result = {}
    checks.append({
        "name": "hermes_terminal_echo",
        "ok": terminal_result.get("exit_code") == 0 and "MS4_HERMES_OK" in str(terminal_result.get("output", "")),
        "detail": {"excerpt": content_text(terminal_echo)[:500]},
    })

    print(json.dumps(checks, indent=2))
    return 0 if all(check["ok"] for check in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
