from __future__ import annotations

import json
import os
import sys
import urllib.request


GATEWAY = os.environ.get("MS4_GATEWAY_URL", "http://127.0.0.1:9180").rstrip("/")
MCP_URL = os.environ.get("MS4_MCP_URL", "http://127.0.0.1:9181/mcp")


def get_json(path: str) -> tuple[int, dict]:
    with urllib.request.urlopen(f"{GATEWAY}{path}", timeout=20) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def post_json(path: str, payload: dict) -> tuple[int, dict]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{GATEWAY}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def rpc(method: str, params: dict | None = None, request_id: int = 1) -> dict:
    body = json.dumps({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}}).encode("utf-8")
    request = urllib.request.Request(MCP_URL, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def tool_call(name: str, arguments: dict | None = None, request_id: int = 10) -> dict:
    return rpc("tools/call", {"name": name, "arguments": arguments or {}}, request_id=request_id)


def content_text(response: dict) -> str:
    return response.get("result", {}).get("content", [{}])[0].get("text", "")


def main() -> int:
    checks = []

    status_code, status = get_json("/desktop/status")
    checks.append({
        "name": "gateway_desktop_status",
        "ok": status_code == 200 and status.get("schema") == "Ms4DesktopStatus.v1",
        "detail": {
            "control_enabled": status.get("control_enabled"),
            "screen": status.get("screen"),
            "dependencies": status.get("dependencies"),
        },
    })

    capture_code, capture = post_json("/desktop/capture", {"include_image": False})
    checks.append({
        "name": "gateway_desktop_capture",
        "ok": capture_code == 200 and capture.get("schema") == "Ms4DesktopCapture.v1" and capture.get("read_only") is True,
        "detail": {key: capture.get(key) for key in ("width", "height", "monitor", "format")},
    })

    mcp_status = tool_call("ms4.desktop.status@v1", request_id=21)
    checks.append({
        "name": "mcp_desktop_status",
        "ok": "Ms4DesktopStatus.v1" in content_text(mcp_status),
        "detail": {"excerpt": content_text(mcp_status)[:400]},
    })

    control_enabled = str(os.environ.get("MS4_DESKTOP_CONTROL", "")).lower() in {"1", "true", "yes", "on"}
    if control_enabled:
        action_code, action = post_json("/desktop/action", {"action": "wait", "seconds": 0})
        checks.append({
            "name": "gateway_desktop_wait_action",
            "ok": action_code == 200 and action.get("schema") == "Ms4DesktopActionResult.v1" and action.get("ok") is True,
            "detail": action,
        })
        mcp_action = tool_call(
            "ms4.desktop.action@v1",
            {"action": "wait", "seconds": 0, "confirm": True},
            request_id=22,
        )
        checks.append({
            "name": "mcp_desktop_wait_action",
            "ok": "Ms4DesktopActionResult.v1" in content_text(mcp_action),
            "detail": {"excerpt": content_text(mcp_action)[:400]},
        })
    else:
        checks.append({
            "name": "desktop_action_smoke",
            "ok": True,
            "detail": "Skipped action smoke because MS4_DESKTOP_CONTROL is not enabled.",
        })

    print(json.dumps(checks, indent=2))
    return 0 if all(check["ok"] for check in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
