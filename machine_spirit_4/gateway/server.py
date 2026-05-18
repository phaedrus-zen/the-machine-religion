from __future__ import annotations

import json
import os
import queue
import threading
import urllib.error
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from machine_spirit_4.deps_status import dependency_status
from machine_spirit_4.desktop import DesktopSafetyError, desktop_controller

from .audit import read_events
from .hermes_runner import HermesUnavailable, Ms4HermesRunner
from .vision import analyze_local_image


ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = ROOT / "machine_spirit_4" / "web"
SERVICE_NAME = "ms4-gateway"


def service_info(runner: Ms4HermesRunner) -> dict[str, Any]:
    health = runner.health()
    return {
        "ok": True,
        "service": SERVICE_NAME,
        "version": "0.1.0",
        "status": "ready",
        "endpoint": "/chat",
        "runtime": health.get("runtime", "ms4-fusion"),
        "hermes_dir": health.get("hermes_dir"),
        "hivemind_url": health.get("hivemind_url"),
        "ms3_url": health.get("ms3_url"),
        "plugin": health.get("plugin", {}),
        "sessions": health.get("sessions", 0),
    }


def _json_response(handler: SimpleHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)


def _sse_start(handler: SimpleHTTPRequestHandler) -> None:
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream; charset=utf-8")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Connection", "keep-alive")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()


def _sse_event(handler: SimpleHTTPRequestHandler, event: str, payload: dict[str, Any]) -> bool:
    body = f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")
    try:
        handler.wfile.write(body)
        handler.wfile.flush()
        return True
    except (BrokenPipeError, ConnectionResetError, OSError):
        return False


def _read_json(handler: SimpleHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0") or "0")
    if length <= 0:
        return {}
    raw = handler.rfile.read(length).decode("utf-8")
    return json.loads(raw)


def _proxy_json(url: str, timeout: int = 20) -> tuple[int, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, {"error": exc.read().decode("utf-8", errors="replace")}
    except Exception as exc:
        return 502, {"error": str(exc)}


def _post_json(url: str, payload: dict[str, Any], timeout: int = 20) -> tuple[int, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, {"error": exc.read().decode("utf-8", errors="replace")}
    except Exception as exc:
        return 502, {"error": str(exc)}


def _desktop_action_intent(action: dict[str, Any]) -> dict[str, Any]:
    action_name = str(action.get("action", "unknown")).lower()
    low_risk = action_name == "wait"
    return {
        "schema": "ActionIntent.v1",
        "spirit_id": os.environ.get("MS4_SPIRIT_ID", "sister"),
        "action_id": f"ms4-desktop-{action_name}",
        "proposed_by": "ms4_gateway",
        "action_type": "desktop_ui",
        "description": f"MS4 desktop UI action: {action_name}",
        "inputs_used": ["MS4 Gateway /desktop/action"],
        "risk_class": "low" if low_risk else "medium",
        "requires_safety_clearance": False,
        "payload": {key: value for key, value in action.items() if key != "text"},
    }


def _ethics_allows(payload: dict[str, Any]) -> bool:
    decision = str(payload.get("decision") or payload.get("resolution") or "").lower()
    if decision in {"allow", "allowed", "offer", "noactionneeded"}:
        return True
    if payload.get("allowed") is True:
        return True
    return False


class Ms4GatewayHandler(SimpleHTTPRequestHandler):
    runner: Ms4HermesRunner

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path in {"/healthcheck/basic", "/api/v1/ms4_gateway/healthcheck/basic"}:
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", "4")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(b"true")
            return
        if self.path == "/api/v1/ms4_gateway/status":
            _json_response(self, 200, service_info(self.runner))
            return
        if self.path == "/health":
            _json_response(self, 200, self.runner.health())
            return
        if self.path == "/sessions":
            _json_response(self, 200, {"sessions": self.runner.sessions()})
            return
        if self.path == "/deps/status":
            _json_response(self, 200, dependency_status())
            return
        if self.path == "/desktop/status":
            _json_response(self, 200, desktop_controller().status())
            return
        if self.path.startswith("/audit"):
            limit = 100
            if "?" in self.path:
                try:
                    query = self.path.split("?", 1)[1]
                    for part in query.split("&"):
                        key, _, value = part.partition("=")
                        if key == "limit":
                            limit = int(value)
                except ValueError:
                    limit = 100
            _json_response(self, 200, {"events": read_events(limit=limit)})
            return
        if self.path == "/hermes/tools":
            _json_response(self, 200, {"tools": self.runner.list_hermes_tools()})
            return
        if self.path == "/models":
            status, payload = _proxy_json(f"{self.runner.ms3_url}/models", timeout=30)
            _json_response(self, status, payload if isinstance(payload, dict) else {"models": payload})
            return
        if self.path == "/voice/status":
            status, payload = _proxy_json(f"{self.runner.ms3_url}/voice/status", timeout=15)
            _json_response(self, status, payload if isinstance(payload, dict) else {"voice": payload})
            return
        if self.path == "/" or self.path.startswith("/static/"):
            self._serve_static()
            return
        _json_response(self, 404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path == "/chat/stream":
            self._stream_chat()
            return
        if self.path == "/desktop/capture":
            try:
                body = _read_json(self)
                _json_response(self, 200, desktop_controller().capture(body))
            except Exception as exc:
                _json_response(self, 500, {"error": str(exc)})
            return
        if self.path == "/desktop/action":
            try:
                body = _read_json(self)
                status, ethics = _post_json(f"{self.runner.ms3_url}/ethics/evaluate", _desktop_action_intent(body), timeout=20)
                if status >= 400 or not _ethics_allows(ethics if isinstance(ethics, dict) else {}):
                    _json_response(self, 403, {"error": "desktop action blocked by MS3 ethics", "ethics": ethics})
                    return
                _json_response(self, 200, desktop_controller().act(body))
            except DesktopSafetyError as exc:
                _json_response(self, 403, {"error": str(exc), "fail_closed": True})
            except Exception as exc:
                _json_response(self, 500, {"error": str(exc)})
            return
        if self.path == "/hermes/tool":
            try:
                body = _read_json(self)
                tool_name = str(body.get("tool") or body.get("tool_name") or "").strip()
                if not tool_name:
                    _json_response(self, 400, {"error": "tool is required"})
                    return
                result = self.runner.dispatch_hermes_tool(
                    tool_name,
                    body.get("args") if isinstance(body.get("args"), dict) else {},
                    session_id=body.get("session_id") or None,
                )
                _json_response(self, 200, result)
            except HermesUnavailable as exc:
                _json_response(self, 503, {"error": str(exc), "fail_closed": True})
            except Exception as exc:
                _json_response(self, 500, {"error": str(exc)})
            return
        if self.path == "/vision/analyze-local":
            try:
                body = _read_json(self)
                result = analyze_local_image(
                    hivemind_url=self.runner.hivemind_url,
                    image_path=str(body.get("image_path") or ""),
                    question=body.get("question") or body.get("prompt") or None,
                    model=body.get("model") or None,
                )
                _json_response(self, 200, result)
            except ValueError as exc:
                _json_response(self, 400, {"error": str(exc), "fail_closed": True})
            except Exception as exc:
                _json_response(self, 502, {"error": str(exc), "fail_closed": True})
            return
        if self.path != "/chat":
            _json_response(self, 404, {"error": "not found"})
            return
        try:
            body = _read_json(self)
            message = str(body.get("message") or body.get("text") or "").strip()
            if not message:
                _json_response(self, 400, {"error": "message is required"})
                return
            result = self.runner.chat(
                message,
                session_id=body.get("session_id") or None,
                model=body.get("model_id") or body.get("model") or None,
            )
            _json_response(self, 200, result)
        except HermesUnavailable as exc:
            _json_response(self, 503, {"error": str(exc), "fail_closed": True})
        except ValueError as exc:
            _json_response(self, 400, {"error": str(exc), "model_incompatible": True})
        except Exception as exc:
            _json_response(self, 500, {"error": str(exc)})

    def _stream_chat(self) -> None:
        try:
            body = _read_json(self)
            message = str(body.get("message") or body.get("text") or "").strip()
            if not message:
                _json_response(self, 400, {"error": "message is required"})
                return
            events: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue()

            def stream_callback(delta: str) -> None:
                if delta:
                    events.put(("token", {"text": delta}))

            def worker() -> None:
                try:
                    result = self.runner.chat(
                        message,
                        session_id=body.get("session_id") or None,
                        model=body.get("model_id") or body.get("model") or None,
                        stream_callback=stream_callback,
                    )
                    events.put(("done", result))
                except ValueError as exc:
                    events.put(("error", {"error": str(exc), "model_incompatible": True}))
                except HermesUnavailable as exc:
                    events.put(("error", {"error": str(exc), "fail_closed": True}))
                except Exception as exc:
                    events.put(("error", {"error": str(exc)}))

            _sse_start(self)
            if not _sse_event(self, "status", {"status": "started"}):
                return
            thread = threading.Thread(target=worker, daemon=True)
            thread.start()
            while True:
                try:
                    event, payload = events.get(timeout=2.0)
                except queue.Empty:
                    if not _sse_event(self, "heartbeat", {"status": "running"}):
                        return
                    continue
                if not _sse_event(self, event, payload):
                    return
                if event in {"done", "error"}:
                    return
        except Exception as exc:
            _json_response(self, 500, {"error": str(exc)})

    def _serve_static(self) -> None:
        relative = "index.html" if self.path == "/" else self.path.removeprefix("/static/")
        target = (WEB_ROOT / relative).resolve()
        if WEB_ROOT.resolve() not in target.parents and target != WEB_ROOT.resolve():
            _json_response(self, 403, {"error": "forbidden"})
            return
        if not target.is_file():
            _json_response(self, 404, {"error": "not found"})
            return
        content = target.read_bytes()
        content_type = "text/html; charset=utf-8" if target.suffix == ".html" else "text/plain; charset=utf-8"
        if target.suffix == ".js":
            content_type = "application/javascript; charset=utf-8"
        if target.suffix == ".css":
            content_type = "text/css; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def build_runner() -> Ms4HermesRunner:
    return Ms4HermesRunner(
        hermes_dir=os.environ.get("MS4_HERMES_DIR", str(Path.home() / "Documents" / "hermes-agent")),
        hivemind_url=os.environ.get("MS4_HIVEMIND_URL", "http://127.0.0.1:6089"),
        ms3_url=os.environ.get("MS4_MS3_URL", "http://127.0.0.1:9080"),
        default_model=os.environ.get("MS4_DEFAULT_MODEL", "qwen3-coder-next:latest"),
    )


def run(host: str = "127.0.0.1", port: int = 9180) -> None:
    handler_cls = Ms4GatewayHandler
    handler_cls.runner = build_runner()
    server = ThreadingHTTPServer((host, port), handler_cls)
    print(f"MS4 gateway listening on http://{host}:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    run(
        host=os.environ.get("MS4_GATEWAY_HOST", "127.0.0.1"),
        port=int(os.environ.get("MS4_GATEWAY_PORT", "9180")),
    )
