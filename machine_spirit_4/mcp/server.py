from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from machine_spirit_4.contained import require_contained_runtime
from machine_spirit_4.desktop import DesktopSafetyError
from machine_spirit_4.gateway import audit
from machine_spirit_4.scripts import runtime_common
from machine_spirit_4.gateway.context import mcp_call
from machine_spirit_4.gateway.hermes_runner import Ms4HermesRunner
from .manifest_policy import (
    ManifestPolicyError,
    admission_state,
    decorate_mcp_tool,
    load_and_validate_manifest,
)
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
    registry = build_tool_registry()
    try:
        _manifest, policies = load_and_validate_manifest(
            registry_names=registry,
            registry_runtime_actions=(
                name for name, tool in registry.items() if tool.is_runtime_action
            ),
        )
    except ManifestPolicyError:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32603, "message": "MS4 MCP manifest policy is invalid"},
        }
    tools = []
    for name, tool in registry.items():
        serialized = tool.as_mcp_tool()
        policy = policies.get(name)
        if policy is not None and policy.get("kind") == "runtime_action":
            serialized = decorate_mcp_tool(serialized, policy)
        tools.append(serialized)
    return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": tools}}


def _action_content(
    *,
    action_id: str,
    tool_name: str,
    state: str,
    result: Any = None,
    reason_code: str | None = None,
    detail: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "Ms4McpRuntimeActionResult.v1",
        "action_id": action_id,
        "tool": tool_name,
        "action_state": state,
    }
    if result is not None:
        payload["result"] = result
    if reason_code is not None:
        payload["reason_code"] = reason_code
    if detail is not None:
        payload["detail"] = detail
    # Keep the v1 text payload backward compatible for successful handlers;
    # policy-aware clients consume the additive structured action envelope.
    text_payload = result if result is not None else payload
    content = result_content(text_payload)
    content["structuredContent"] = payload
    content["isError"] = state in {
        "approval_required",
        "denied",
        "reconciliation_required",
    }
    return content


def _audit_action(
    policy: dict[str, Any],
    *,
    action_id: str,
    tool_name: str,
    phase: str,
    state: str,
    reason_code: str | None = None,
) -> None:
    data = {
        "schema": "Ms4McpRuntimeActionAudit.v1",
        "action_id": action_id,
        "tool": tool_name,
        "phase": phase,
        "action_state": state,
    }
    if reason_code is not None:
        data["reason_code"] = reason_code
    audit.append_event(policy["audit"]["event"], data)


def _normalized_state(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"approval_required", "awaiting_approval", "pending_approval"}:
        return "approval_required"
    if normalized in {"denied", "rejected"} or normalized.startswith("denied_"):
        return "denied"
    if normalized in {"cancelled", "canceled"} or normalized.startswith(("cancelled_", "canceled_")):
        return "cancelled"
    if normalized in {"failed", "error", "timeout", "timed_out", "unknown"} or normalized.startswith(
        ("failed_", "error_")
    ):
        return "reconciliation_required"
    if normalized in {"queued", "pending", "in_progress", "running"}:
        return "running"
    if normalized in {"complete", "completed", "committed", "success", "succeeded"}:
        return "committed"
    return None


def _result_state(result: Any) -> str:
    if isinstance(result, dict):
        if result.get("refused") is True:
            return "denied"
        if result.get("cancelled") is True or result.get("canceled") is True:
            return "cancelled"
        for key in ("action_state", "state", "status", "phase"):
            state = _normalized_state(result.get(key))
            if state is not None:
                return state
        if result.get("completed") is False or result.get("ok") is False or result.get("error"):
            return "reconciliation_required"
        if result.get("completed") is True:
            return "committed"
    return "committed"


def _runtime_action_response(
    request_id: Any,
    *,
    name: str,
    arguments: dict[str, Any],
    runtime: Any,
    tool: Any,
    policy: dict[str, Any],
) -> dict[str, Any]:
    action_id = str(uuid.uuid4())
    state, reason_code = admission_state(policy, arguments)
    try:
        _audit_action(
            policy,
            action_id=action_id,
            tool_name=name,
            phase="admission",
            state=state,
            reason_code=reason_code,
        )
    except Exception:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": _action_content(
                action_id=action_id,
                tool_name=name,
                state="denied",
                reason_code="required_audit_unavailable",
            ),
        }

    if state != "running":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": _action_content(
                action_id=action_id,
                tool_name=name,
                state=state,
                reason_code=reason_code,
            ),
        }

    try:
        result = tool.handler(runtime, arguments)
    except (ToolInputError, DesktopSafetyError) as exc:
        state = "denied"
        reason_code = (
            "action_policy_denied"
            if isinstance(exc, DesktopSafetyError)
            else "invalid_action_arguments"
        )
        try:
            _audit_action(
                policy,
                action_id=action_id,
                tool_name=name,
                phase="outcome",
                state=state,
                reason_code=reason_code,
            )
        except Exception:
            state = "reconciliation_required"
            reason_code = "required_audit_unavailable"
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": _action_content(
                action_id=action_id,
                tool_name=name,
                state=state,
                reason_code=reason_code,
                detail=str(exc),
            ),
        }
    except Exception:
        state = "reconciliation_required"
        reason_code = "handler_outcome_unknown"
        try:
            _audit_action(
                policy,
                action_id=action_id,
                tool_name=name,
                phase="outcome",
                state=state,
                reason_code=reason_code,
            )
        except Exception:
            pass
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": _action_content(
                action_id=action_id,
                tool_name=name,
                state=state,
                reason_code=reason_code,
                detail="The handler outcome is uncertain; reconcile before retrying.",
            ),
        }

    state = _result_state(result)
    try:
        _audit_action(
            policy,
            action_id=action_id,
            tool_name=name,
            phase="outcome",
            state=state,
        )
    except Exception:
        state = "reconciliation_required"
        reason_code = "committed_result_not_audited"
    else:
        reason_code = None
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": _action_content(
            action_id=action_id,
            tool_name=name,
            state=state,
            result=result,
            reason_code=reason_code,
        ),
    }


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
        _manifest, policies = load_and_validate_manifest(
            registry_names=registry,
            registry_runtime_actions=(
                registered_name
                for registered_name, registered_tool in registry.items()
                if registered_tool.is_runtime_action
            ),
        )
    except ManifestPolicyError:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32603, "message": "MS4 MCP manifest policy is invalid"},
        }
    policy = policies.get(name)
    if policy is not None and policy.get("kind") == "runtime_action":
        return _runtime_action_response(
            request_id,
            name=name,
            arguments=arguments,
            runtime=runtime,
            tool=tool,
            policy=policy,
        )
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
        hermes_dir=str(runtime_common.hermes_dir()),
        hivemind_url=(
            os.environ.get("MS4_HIVEMIND_URL")
            or os.environ.get("MS4_HIVEMIND_HLI_URL")
            or "http://127.0.0.1:6089"
        ),
        ms3_url=os.environ.get("MS4_MS3_URL", "http://127.0.0.1:9080"),
        default_model=os.environ.get("MS4_DEFAULT_MODEL", "qwen3-coder-next:latest"),
    )
    return Ms4McpRuntime(runner)


def run(host: str = "127.0.0.1", port: int = 9181) -> None:
    require_contained_runtime(SERVER_NAME)
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
