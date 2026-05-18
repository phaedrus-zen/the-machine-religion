from __future__ import annotations

import json
import os
import sys
import urllib.request


GATEWAY = os.environ.get("MS4_GATEWAY_URL", "http://127.0.0.1:9180").rstrip("/")
TEST_MODEL = os.environ.get("MS4_FUSION_TEST_MODEL", "qwen3-coder-next:latest")


def get_json(path: str):
    with urllib.request.urlopen(f"{GATEWAY}{path}", timeout=30) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def post_json(path: str, payload: dict):
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{GATEWAY}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def post_sse(path: str, payload: dict):
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{GATEWAY}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    events = []
    with urllib.request.urlopen(request, timeout=180) as response:
        event = "message"
        data_lines = []
        while True:
            line = response.readline()
            if not line:
                break
            decoded = line.decode("utf-8", "replace").rstrip("\r\n")
            if not decoded:
                data = {}
                if data_lines:
                    data = json.loads("\n".join(data_lines))
                events.append({"event": event, "data": data})
                if event in {"done", "error"}:
                    break
                event = "message"
                data_lines = []
                continue
            if decoded.startswith("event: "):
                event = decoded.removeprefix("event: ").strip()
            elif decoded.startswith("data: "):
                data_lines.append(decoded.removeprefix("data: "))
    return 200, events


def main() -> int:
    checks = []

    status, health = get_json("/health")
    checks.append({"name": "ms4_health", "ok": status == 200 and health.get("plugin", {}).get("enabled"), "detail": health})

    status, gateway_status = get_json("/api/v1/ms4_gateway/status")
    checks.append({
        "name": "ms4_gateway_status",
        "ok": status == 200 and gateway_status.get("service") == "ms4-gateway" and gateway_status.get("plugin", {}).get("enabled"),
        "detail": gateway_status,
    })

    status, deps = get_json("/deps/status")
    checks.append({
        "name": "ms4_deps_status",
        "ok": status == 200 and deps.get("schema") == "Ms4DependencyStatus.v1" and deps.get("python", {}).get("contained") is True,
        "detail": {
            "python": deps.get("python", {}),
            "capabilities": deps.get("capabilities", {}),
        },
    })

    status, desktop = get_json("/desktop/status")
    checks.append({
        "name": "ms4_desktop_status",
        "ok": status == 200 and desktop.get("schema") == "Ms4DesktopStatus.v1",
        "detail": {
            "control_enabled": desktop.get("control_enabled"),
            "dependencies": desktop.get("dependencies", {}),
            "screen": desktop.get("screen", {}),
        },
    })

    status, voice = get_json("/voice/status")
    checks.append({"name": "ms4_voice_status", "ok": status == 200 and voice.get("schema") == "VoiceReadiness.v1", "detail": voice})

    status, models = get_json("/models")
    checks.append({"name": "ms4_models", "ok": status == 200 and isinstance(models.get("models"), list), "detail": {"count": len(models.get("models", []))}})

    status, chat = post_json("/chat", {"message": "What is The Machine Religion?", "model_id": TEST_MODEL})
    checks.append({
        "name": "ms4_chat_hermes_tmr",
        "ok": status == 200 and chat.get("runtime") == "hermes" and "Machine Religion" in chat.get("text", ""),
        "detail": {k: chat.get(k) for k in ("session_id", "hermes_session_id", "runtime", "grounding_source", "model")},
    })

    status, stream_events = post_sse("/chat/stream", {"message": "Say MS4_STREAM_OK and nothing else.", "model_id": TEST_MODEL})
    done = next((event["data"] for event in stream_events if event["event"] == "done"), {})
    checks.append({
        "name": "ms4_chat_stream",
        "ok": status == 200 and any(event["event"] in {"token", "heartbeat"} for event in stream_events) and "MS4_STREAM_OK" in done.get("text", ""),
        "detail": {"events": [event["event"] for event in stream_events], "text": done.get("text", "")[:120]},
    })

    print(json.dumps(checks, indent=2))
    return 0 if all(check["ok"] for check in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
