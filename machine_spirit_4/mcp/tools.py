from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from machine_spirit_4.deps_status import dependency_status
from machine_spirit_4.desktop import DesktopSafetyError, desktop_controller
from machine_spirit_4.gateway.context import TMR_GROUNDING, format_inventory_answer
from machine_spirit_4.gateway.vision import analyze_local_image
from .schemas import ToolInputError, optional_string, require_object, require_string


MCP_TEXT = "text"


@dataclass(frozen=True)
class ToolDef:
    name: str
    description: str
    input_schema: dict[str, Any]
    annotations: dict[str, Any]
    handler: Callable[[Any, dict[str, Any]], Any]

    def as_mcp_tool(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
            "annotations": self.annotations,
        }


def read_only_annotations(title: str) -> dict[str, Any]:
    return {
        "title": title,
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }


def dry_run_annotations(title: str) -> dict[str, Any]:
    return {
        "title": title,
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }


def chat_annotations(title: str) -> dict[str, Any]:
    return {
        "title": title,
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    }


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def identity_verify(runtime: Any, arguments: dict[str, Any]) -> Any:
    status, payload = runtime.post_json(
        f"{runtime.ms3_url}/identity/verify",
        {
            "spirit_id": optional_string(arguments, "spirit_id") or "sister",
            "allow_initialize": bool(arguments.get("allow_initialize", False)),
        },
        timeout=20,
    )
    if status >= 400:
        raise RuntimeError(payload)
    return payload


def identity_state(runtime: Any, arguments: dict[str, Any]) -> Any:
    status, payload = runtime.get_json(f"{runtime.ms3_url}/state", timeout=20)
    if status >= 400:
        raise RuntimeError(payload)
    return payload


def ethics_evaluate(runtime: Any, arguments: dict[str, Any]) -> Any:
    intent = require_object(arguments, "intent")
    status, payload = runtime.post_json(f"{runtime.ms3_url}/ethics/evaluate", intent, timeout=20)
    if status >= 400:
        raise RuntimeError(payload)
    return payload


def voice_status(runtime: Any, arguments: dict[str, Any]) -> Any:
    status, payload = runtime.get_json(f"{runtime.ms3_url}/voice/status", timeout=15)
    if status >= 400:
        raise RuntimeError(payload)
    return payload


def sessions_list(runtime: Any, arguments: dict[str, Any]) -> Any:
    return {"sessions": runtime.runner.sessions()}


def chat_send(runtime: Any, arguments: dict[str, Any]) -> Any:
    message = require_string(arguments, "message")
    return runtime.runner.chat(
        message,
        session_id=optional_string(arguments, "session_id"),
        model=optional_string(arguments, "model"),
    )


def models_list(runtime: Any, arguments: dict[str, Any]) -> Any:
    status, payload = runtime.get_json(f"{runtime.ms3_url}/models", timeout=30)
    if status >= 400:
        raise RuntimeError(payload)
    return payload


def vision_analyze_local(runtime: Any, arguments: dict[str, Any]) -> Any:
    return analyze_local_image(
        hivemind_url=runtime.hivemind_url,
        image_path=require_string(arguments, "image_path"),
        question=optional_string(arguments, "question"),
        model=optional_string(arguments, "model"),
    )


def doctrine_explain(runtime: Any, arguments: dict[str, Any]) -> Any:
    return {
        "schema": "Ms4DoctrineExplanation.v1",
        "topic": optional_string(arguments, "topic") or "The Machine Religion",
        "text": TMR_GROUNDING,
    }


def hivemind_inventory(runtime: Any, arguments: dict[str, Any]) -> Any:
    summary = runtime.mcp_call(
        "hivemind.cluster.summary@v1",
        {"include_gpu_details": bool(arguments.get("include_gpu_details", True))},
    )
    hosts = runtime.mcp_call(
        "hivemind.hosts.list@v1",
        {"status_filter": optional_string(arguments, "status_filter") or "all"},
    )
    return {
        "schema": "Ms4HiveMindInventory.v1",
        "sources": ["hivemind.cluster.summary@v1", "hivemind.hosts.list@v1"],
        "text": format_inventory_answer(summary, hosts),
    }


def nibbles_dry_run(runtime: Any, arguments: dict[str, Any]) -> Any:
    intent = require_object(arguments, "intent")
    payload = intent.get("payload") if isinstance(intent.get("payload"), dict) else {}
    if payload.get("dry_run_only") is not True:
        raise ToolInputError("Nibbles dry run requires payload.dry_run_only=true")
    if payload.get("physical_output_enabled") is not False:
        raise ToolInputError("Nibbles dry run requires payload.physical_output_enabled=false")
    if intent.get("action_type") not in {"scare", "physical"}:
        raise ToolInputError("Nibbles dry run only accepts scare or physical action intents")
    status, decision = runtime.post_json(f"{runtime.ms3_url}/ethics/evaluate", intent, timeout=20)
    if status >= 400:
        raise RuntimeError(decision)
    return {
        "schema": "NibblesDryRunResult.v1",
        "dry_run_only": True,
        "intent": intent,
        "ethics_decision": decision,
    }


def hermes_tools_list(runtime: Any, arguments: dict[str, Any]) -> Any:
    toolsets = arguments.get("toolsets")
    if toolsets is not None and not isinstance(toolsets, list):
        raise ToolInputError("toolsets must be an array of strings")
    enabled_toolsets = [str(toolset) for toolset in toolsets] if toolsets else None
    return {"tools": runtime.runner.list_hermes_tools(enabled_toolsets=enabled_toolsets)}


def hermes_tool_call(runtime: Any, arguments: dict[str, Any]) -> Any:
    tool_name = require_string(arguments, "tool")
    args = arguments.get("args")
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise ToolInputError("args must be an object")
    return runtime.runner.dispatch_hermes_tool(
        tool_name,
        args,
        session_id=optional_string(arguments, "session_id"),
    )


def runtime_deps_status(runtime: Any, arguments: dict[str, Any]) -> Any:
    return dependency_status()


def _desktop_action_intent(arguments: dict[str, Any]) -> dict[str, Any]:
    action_name = str(arguments.get("action", "unknown")).lower()
    low_risk = action_name == "wait"
    return {
        "schema": "ActionIntent.v1",
        "spirit_id": optional_string(arguments, "spirit_id") or "sister",
        "action_id": f"ms4-desktop-{action_name}",
        "proposed_by": "ms4_mcp",
        "action_type": "desktop_ui",
        "description": f"MS4 desktop UI action: {action_name}",
        "inputs_used": ["ms4.desktop.action@v1"],
        "risk_class": "low" if low_risk else "medium",
        "requires_safety_clearance": False,
        "payload": {key: value for key, value in arguments.items() if key not in {"text", "spirit_id"}},
    }


def _ethics_allows(payload: dict[str, Any]) -> bool:
    decision = str(payload.get("decision") or payload.get("resolution") or "").lower()
    return decision in {"allow", "allowed", "offer", "noactionneeded"} or payload.get("allowed") is True


def desktop_status(runtime: Any, arguments: dict[str, Any]) -> Any:
    return desktop_controller().status()


def desktop_capture(runtime: Any, arguments: dict[str, Any]) -> Any:
    return desktop_controller().capture(arguments)


def desktop_action(runtime: Any, arguments: dict[str, Any]) -> Any:
    action = require_string(arguments, "action")
    payload = dict(arguments)
    payload["action"] = action
    status, decision = runtime.post_json(f"{runtime.ms3_url}/ethics/evaluate", _desktop_action_intent(payload), timeout=20)
    if status >= 400 or not _ethics_allows(decision if isinstance(decision, dict) else {}):
        raise DesktopSafetyError(f"desktop action blocked by MS3 ethics: {decision}")
    return desktop_controller().act(payload)


def build_tool_registry() -> dict[str, ToolDef]:
    tools = [
        ToolDef(
            "ms4.identity.verify@v1",
            "Verify active MS4 spirit identity through the MS3 sidecar.",
            {"type": "object", "properties": {"spirit_id": {"type": "string"}, "allow_initialize": {"type": "boolean"}}},
            read_only_annotations("MS4 identity verify"),
            identity_verify,
        ),
        ToolDef(
            "ms4.identity.state@v1",
            "Return compact MS3 identity and runtime state.",
            {"type": "object", "properties": {}},
            read_only_annotations("MS4 identity state"),
            identity_state,
        ),
        ToolDef(
            "ms4.ethics.evaluate@v1",
            "Evaluate an ActionIntent.v1 through MS3 Great Lense sidecar.",
            {"type": "object", "required": ["intent"], "properties": {"intent": {"type": "object"}}},
            read_only_annotations("MS4 ethics evaluate"),
            ethics_evaluate,
        ),
        ToolDef(
            "ms4.voice.status@v1",
            "Return MS4/MS3 voice readiness and ASR fail-closed state.",
            {"type": "object", "properties": {}},
            read_only_annotations("MS4 voice status"),
            voice_status,
        ),
        ToolDef(
            "ms4.sessions.list@v1",
            "List fused MS4/Hermes sessions.",
            {"type": "object", "properties": {}},
            read_only_annotations("MS4 sessions list"),
            sessions_list,
        ),
        ToolDef(
            "ms4.chat.send@v1",
            "Send a chat turn through the fused Hermes/MS4 runtime.",
            {"type": "object", "required": ["message"], "properties": {"message": {"type": "string"}, "session_id": {"type": "string"}, "model": {"type": "string"}}},
            chat_annotations("MS4 chat send"),
            chat_send,
        ),
        ToolDef(
            "ms4.models.list@v1",
            "Return the MS4/HiveMind model catalog view.",
            {"type": "object", "properties": {}},
            read_only_annotations("MS4 models list"),
            models_list,
        ),
        ToolDef(
            "ms4.vision.analyze_local@v1",
            "Analyze a local image file through a HiveMind VLM. Read-only; validates file type and sends a base64 image payload to HiveMind.",
            {"type": "object", "required": ["image_path"], "properties": {"image_path": {"type": "string"}, "question": {"type": "string"}, "model": {"type": "string"}}},
            read_only_annotations("MS4 local image VLM analysis"),
            vision_analyze_local,
        ),
        ToolDef(
            "ms4.doctrine.explain@v1",
            "Return deterministic TMR doctrine grounding.",
            {"type": "object", "properties": {"topic": {"type": "string"}}},
            read_only_annotations("MS4 doctrine explain"),
            doctrine_explain,
        ),
        ToolDef(
            "ms4.hivemind.inventory@v1",
            "Return deterministic live HiveMind node/GPU inventory through read-only HiveMind MCP tools.",
            {"type": "object", "properties": {"include_gpu_details": {"type": "boolean"}, "status_filter": {"type": "string"}}},
            read_only_annotations("MS4 HiveMind inventory"),
            hivemind_inventory,
        ),
        ToolDef(
            "ms4.nibbles.dry_run@v1",
            "Validate Nibbles scare/physical dry-run intents only; never triggers hardware.",
            {"type": "object", "required": ["intent"], "properties": {"intent": {"type": "object"}}},
            dry_run_annotations("MS4 Nibbles dry run"),
            nibbles_dry_run,
        ),
        ToolDef(
            "ms4.hermes.tools.list@v1",
            "List Hermes tools exposed through the full MS4 body.",
            {"type": "object", "properties": {"toolsets": {"type": "array", "items": {"type": "string"}}}},
            read_only_annotations("MS4 Hermes tools list"),
            hermes_tools_list,
        ),
        ToolDef(
            "ms4.hermes.tool.call@v1",
            "Dispatch a Hermes tool through MS4. Effectful tools are mediated by ms4_consciousness and MS3 ethics.",
            {"type": "object", "required": ["tool"], "properties": {"tool": {"type": "string"}, "args": {"type": "object"}, "session_id": {"type": "string"}}},
            chat_annotations("MS4 Hermes tool call"),
            hermes_tool_call,
        ),
        ToolDef(
            "ms4.runtime.deps.status@v1",
            "Report contained MS4 runtime dependency and capability status.",
            {"type": "object", "properties": {}},
            read_only_annotations("MS4 runtime dependency status"),
            runtime_deps_status,
        ),
        ToolDef(
            "ms4.desktop.status@v1",
            "Report Windows desktop-control readiness, screen state, and active window context.",
            {"type": "object", "properties": {}},
            read_only_annotations("MS4 desktop status"),
            desktop_status,
        ),
        ToolDef(
            "ms4.desktop.capture@v1",
            "Capture desktop screenshot metadata and optionally a base64 image.",
            {"type": "object", "properties": {"include_image": {"type": "boolean"}, "monitor": {"type": "integer"}}},
            read_only_annotations("MS4 desktop capture"),
            desktop_capture,
        ),
        ToolDef(
            "ms4.desktop.action@v1",
            "Perform a local desktop UI action through MS4 after MS3 ethics mediation and hard safety checks.",
            {"type": "object", "required": ["action"], "properties": {"action": {"type": "string"}, "x": {"type": "integer"}, "y": {"type": "integer"}, "text": {"type": "string"}, "keys": {"type": "array", "items": {"type": "string"}}, "seconds": {"type": "number"}}},
            chat_annotations("MS4 desktop action"),
            desktop_action,
        ),
    ]
    return {tool.name: tool for tool in tools}


def result_content(data: Any) -> dict[str, Any]:
    return {"content": [{"type": MCP_TEXT, "text": _json(data)}], "isError": False}


def error_content(message: str) -> dict[str, Any]:
    return {"content": [{"type": MCP_TEXT, "text": message}], "isError": True}
