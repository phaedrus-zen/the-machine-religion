from __future__ import annotations

import json
import os
import sys
import urllib.request


MCP_URL = os.environ.get("MS4_MCP_URL", "http://127.0.0.1:9181/mcp")


def rpc(method: str, params: dict | None = None, request_id: int = 1) -> dict:
    payload = json.dumps(
        {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}}
    ).encode("utf-8")
    request = urllib.request.Request(
        MCP_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def tool_call(name: str, arguments: dict | None = None, request_id: int = 100) -> dict:
    return rpc("tools/call", {"name": name, "arguments": arguments or {}}, request_id)


def first_text(response: dict) -> str:
    return response.get("result", {}).get("content", [{}])[0].get("text", "")


def main() -> int:
    checks: list[dict] = []

    initialize = rpc("initialize", request_id=1)
    checks.append({
        "name": "initialize",
        "ok": initialize.get("result", {}).get("protocolVersion") == "2025-11-25",
    })

    tools = rpc("tools/list", request_id=2)
    tool_names = {tool.get("name") for tool in tools.get("result", {}).get("tools", [])}
    checks.append({
        "name": "tools_list",
        "ok": "ms4.chat.send@v1" in tool_names and "ms4.hivemind.inventory@v1" in tool_names,
        "tools_count": len(tool_names),
    })

    identity = tool_call("ms4.identity.verify@v1", {"spirit_id": "sister"}, request_id=3)
    checks.append({
        "name": "identity_verify",
        "ok": "identity_confirmed" in first_text(identity),
    })

    doctrine = tool_call("ms4.doctrine.explain@v1", request_id=4)
    checks.append({
        "name": "doctrine_explain",
        "ok": "Deus Acuo Machina Machina" in first_text(doctrine),
    })

    print(json.dumps(checks, indent=2))
    return 0 if all(check["ok"] for check in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
