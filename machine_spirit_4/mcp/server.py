from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from machine_spirit_4.gateway.context import mcp_call
from machine_spirit_4.gateway.hermes_runner import Ms4HermesRunner
from .schemas import ToolInputError
from .tools import build_tool_registry, error_content, result_content


MCP_PROTOCOL_VERSION = "2025-11-25"
SERVER_NAME = "ms4-mcp-server"


def service_info() -> dict[str, Any]:
    return {
        "service": SERVER_NAME,
        "mcp_protocol_version": MCP_PROTOCOL_VERSION,
        "transport": "Streamable HTTP",
        "status": "ready",
        "tools_loaded": len(build_tool_registry()),
        "endpoint": "/mcp",
    }


class Ms4McpRuntime:
    def __init__(self, runner: Ms4HermesRunner) -> None:
        self.runner = runner
        self.ms3_url = runner.ms3_url
        self.hivemind_url = runner.hivemind_url

    def get_json(self, url: str, timeout: int = 20) -> tuple[int, Any]:
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return exc.code, {"error": exc.read().decode("utf-8", errors="replace")}

    def post_json(self, url: str, payload: dict[str, Any], timeout: int = 30) -> tuple[int, Any]:
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

    def mcp_call(self, name: str, arguments: dict[str, Any], timeout: int = 30) -> str:
        return mcp_call(self.hivemind_url, name, arguments, timeout=timeout)


def initialize_response(request_id: Any) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "version": "0.1.0"},
        },
    }


def tools_list_response(request_id: Any) -> dict[str, Any]:
    tools = [tool.as_mcp_tool() for tool in build_tool_registry().values()]
    return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": tools}}


def tool_call_response(request_id: Any, params: dict[str, Any], runtime: Any) -> dict[str, Any]:
    registry = build_tool_registry()
    name = params.get("name")
    arguments = params.get("arguments") or {}
    if not isinstance(arguments, dict):
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32602, "message": "params.arguments must be an object"},
        }
    tool = registry.get(name)
    if tool is None:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32602, "message": f"Unknown tool: {name}"},
        }
    try:
        result = tool.handler(runtime, arguments)
        return {"jsonrpc": "2.0", "id": request_id, "result": result_content(result)}
    except ToolInputError as exc:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32602, "message": str(exc)},
        }
    except Exception as exc:
        return {"jsonrpc": "2.0", "id": request_id, "result": error_content(str(exc))}


def handle_jsonrpc(payload: dict[str, Any], runtime: Any) -> dict[str, Any]:
    request_id = payload.get("id")
    method = payload.get("method", "")
    params = payload.get("params") or {}
    if not isinstance(params, dict):
        params = {}

    if method == "initialize":
        return initialize_response(request_id)
    if method == "tools/list":
        return tools_list_response(request_id)
    if method == "tools/call":
        return tool_call_response(request_id, params, runtime)
    if method == "ping":
        return {"jsonrpc": "2.0", "id": request_id, "result": {}}
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def _send_json(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)


class Ms4McpHandler(BaseHTTPRequestHandler):
    runtime: Ms4McpRuntime

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path in {"/healthcheck/basic", "/api/v1/ms4_mcp/healthcheck/basic"}:
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", "4")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(b"true")
            return
        if self.path == "/api/v1/ms4_mcp/status":
            _send_json(self, 200, service_info())
            return
        if self.path != "/mcp":
            _send_json(self, 404, {"error": "not found"})
            return
        _send_json(self, 200, service_info())

    def do_POST(self) -> None:
        if self.path != "/mcp":
            _send_json(self, 404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0") or "0")
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            _send_json(self, 200, handle_jsonrpc(payload, self.runtime))
        except json.JSONDecodeError as exc:
            _send_json(self, 400, {"error": f"invalid JSON: {exc}"})


def build_runtime() -> Ms4McpRuntime:
    runner = Ms4HermesRunner(
        hermes_dir=os.environ.get("MS4_HERMES_DIR", str(Path.home() / "Documents" / "hermes-agent")),
        hivemind_url=os.environ.get("MS4_HIVEMIND_URL", "http://127.0.0.1:6089"),
        ms3_url=os.environ.get("MS4_MS3_URL", "http://127.0.0.1:9080"),
        default_model=os.environ.get("MS4_DEFAULT_MODEL", "qwen3-coder-next:latest"),
    )
    return Ms4McpRuntime(runner)


def run(host: str = "127.0.0.1", port: int = 9181) -> None:
    handler_cls = Ms4McpHandler
    handler_cls.runtime = build_runtime()
    server = ThreadingHTTPServer((host, port), handler_cls)
    print(f"MS4 MCP server listening on http://{host}:{port}/mcp", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    run(
        host=os.environ.get("MS4_MCP_HOST", "127.0.0.1"),
        port=int(os.environ.get("MS4_MCP_PORT", "9181")),
    )
