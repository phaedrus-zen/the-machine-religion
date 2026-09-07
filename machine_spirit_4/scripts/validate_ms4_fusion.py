from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
import uuid


GATEWAY = os.environ.get("MS4_GATEWAY_URL", "http://127.0.0.1:9180").rstrip("/")
# Keep the default smoke on the deployment's resident Face model. Operators
# can still opt into a large cold-load model explicitly, but validation must
# not manufacture a remote qwen3-coder-next backlog merely to prove routing.
TEST_MODEL = os.environ.get("MS4_FUSION_TEST_MODEL", "nemotron-3-nano:4b")
DEPTH_TIMEOUT_SECONDS = float(os.environ.get("MS4_FUSION_DEPTH_TIMEOUT", "180"))
DEPTH_POLL_SECONDS = float(os.environ.get("MS4_FUSION_DEPTH_POLL", "2"))
CONTEXT_MARKER = "MS4_FUSION_CONTEXT_OK"


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


def wait_for_depth_job(job_id: str) -> tuple[dict, bool]:
    deadline = time.monotonic() + max(1.0, DEPTH_TIMEOUT_SECONDS)
    snapshot: dict = {}
    while time.monotonic() < deadline:
        _, snapshot = get_json(f"/api/v1/double-agent/jobs/{job_id}")
        if snapshot.get("state") in {"completed", "failed", "canceled", "stale"}:
            return snapshot, False
        time.sleep(max(0.1, DEPTH_POLL_SECONDS))

    # A validation run owns this exact job. Do not leave its worker or model
    # load behind when the terminal-state contract times out.
    try:
        _, snapshot = post_json(f"/api/v1/double-agent/jobs/{job_id}/cancel", {})
    except Exception:
        pass
    return snapshot, True


def main() -> int:
    checks = []
    session_id = f"ms4-fusion-validator-{uuid.uuid4().hex[:12]}"

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

    # A plain knowledge question is answered on the fast Face Lobe, so /chat
    # returns runtime "face-lobe-direct" (only the explicit /hermes/tool path
    # ever returns "hermes"). The TMR contract here is: a real answer, produced
    # via the Face Lobe, grounded in TMR canon.
    status, chat = post_json("/chat", {
        "message": f"What is The Machine Religion? Remember this validation marker: {CONTEXT_MARKER}",
        "model_id": TEST_MODEL,
        "session_id": session_id,
    })
    chat_text = chat.get("text") or ""
    chat_grounding = chat.get("grounding_source") or ""
    checks.append({
        "name": "ms4_chat_facelobe_tmr",
        "ok": (
            status == 200
            and chat.get("runtime") == "face-lobe-direct"
            and "Machine Religion" in chat_text
            and "tmr-canon" in chat_grounding
        ),
        "detail": {k: chat.get(k) for k in ("session_id", "hermes_session_id", "runtime", "grounding_source", "model")},
    })

    # Exercise the Depth Lobe / Hermes route with a reasoning-only analytical
    # /deep prompt. Admission grants an empty tool catalog, not the broad
    # mcp-hivemind set; do not weaken that contract to keep this smoke green.
    status, deep = post_json("/chat", {
        "message": (
            "/deep Compare Face and Depth as an architecture, then give the "
            "recommendation, mechanism, tradeoffs, and a concrete example. "
            "Include the exact marker from the prior turn."
        ),
        "model_id": TEST_MODEL,
        "depth_model_id": TEST_MODEL,
        "session_id": session_id,
    })
    deep_grounding = deep.get("grounding_source") or ""
    deep_job = deep.get("dispatched_job") if isinstance(deep.get("dispatched_job"), dict) else {}
    checks.append({
        "name": "ms4_chat_deep_dispatch",
        "ok": (
            status == 200
            and "depth_lobe_dispatched" in deep_grounding
            and bool(deep_job.get("job_id"))
            and (deep.get("depth_lobe_model") or {}).get("model_id") == TEST_MODEL
        ),
        "detail": {
            "runtime": deep.get("runtime"),
            "grounding_source": deep_grounding,
            "dispatched_job_id": deep_job.get("job_id"),
            "depth_lobe_model": (deep.get("depth_lobe_model") or {}).get("model_id"),
        },
    })

    depth_snapshot: dict = {}
    depth_timed_out = False
    if deep_job.get("job_id"):
        depth_snapshot, depth_timed_out = wait_for_depth_job(str(deep_job["job_id"]))
    depth_result = depth_snapshot.get("result") if isinstance(depth_snapshot.get("result"), dict) else {}
    depth_text = str(depth_result.get("text") or depth_result.get("summary") or "")
    depth_resource = depth_snapshot.get("resource_request") if isinstance(depth_snapshot.get("resource_request"), dict) else {}
    checks.append({
        "name": "ms4_chat_deep_completion",
        "ok": (
            not depth_timed_out
            and depth_snapshot.get("state") == "completed"
            and depth_result.get("status") == "success"
            and depth_resource.get("model_override") == TEST_MODEL
            and (depth_resource.get("enabled_toolsets") or []) == []
            and CONTEXT_MARKER in depth_text
        ),
        "detail": {
            "job_id": deep_job.get("job_id"),
            "state": depth_snapshot.get("state"),
            "result_status": depth_result.get("status"),
            "model_override": depth_resource.get("model_override"),
            "enabled_toolsets": depth_resource.get("enabled_toolsets"),
            "prior_context_turns": len(depth_snapshot.get("prior_context") or []),
            "timed_out": depth_timed_out,
            "text": depth_text[:240],
        },
    })

    status, stream_events = post_sse("/chat/stream", {
        "message": "Say MS4_STREAM_OK and nothing else.",
        "model_id": TEST_MODEL,
        "session_id": session_id,
    })
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
