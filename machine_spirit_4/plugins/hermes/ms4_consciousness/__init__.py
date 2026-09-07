"""MS4 consciousness integration plugin for Hermes."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from .client import Ms4Client, Ms4ClientError
from .config import Ms4Config, load_config
from .output import strip_psyche_blocks
from .psyche import build_ms4_context_block

_identity_cache: dict[str, Any] | None = None
_state_cache: dict[str, Any] | None = None
_last_error: str | None = None


def register(ctx: Any) -> None:
    ctx.register_hook("on_session_start", on_session_start)
    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    ctx.register_hook("transform_llm_output", on_transform_llm_output)
    ctx.register_hook("post_llm_call", on_post_llm_call)
    _register_hivemind_tools(ctx)


def on_session_start(**kwargs: Any) -> dict[str, Any]:
    config = load_config()
    client = Ms4Client(config.ms3_sidecar_url)
    return _verify_identity(client, config, session_id=str(kwargs.get("session_id", "")))


def on_pre_llm_call(**kwargs: Any) -> dict[str, str]:
    config = load_config()
    client = Ms4Client(config.ms3_sidecar_url)
    _ensure_identity(client, config, session_id=str(kwargs.get("session_id", "")))

    state = _safe_get_state(client, config)
    context = build_ms4_context_block(
        identity=_identity_cache,
        state=state,
        spirit_id=config.spirit_id,
    )
    if _last_error:
        context = f"{context}\nms4_status: degraded_fail_closed\nlast_error: {_last_error}"
    return {"context": context}


def on_pre_tool_call(**kwargs: Any) -> dict[str, str] | None:
    config = load_config()
    client = Ms4Client(config.ms3_sidecar_url)
    tool_name = str(kwargs.get("tool_name", "unknown_tool"))
    args = kwargs.get("args") if isinstance(kwargs.get("args"), dict) else {}

    try:
        _ensure_identity(client, config, session_id=str(kwargs.get("session_id", "")))
        action_type = _classify_tool(tool_name)
        risk_class = _risk_class_for_tool(tool_name, args, action_type)
        decision = client.evaluate_action(
            {
                "schema": "ActionIntent.v1",
                "spirit_id": config.spirit_id,
                "action_id": str(kwargs.get("tool_call_id") or tool_name),
                "proposed_by": "hermes",
                "action_type": action_type,
                "description": f"Hermes tool call: {tool_name}",
                "risk_class": risk_class,
                "requires_safety_clearance": action_type in {"scare", "physical"},
                "payload": {"tool_name": tool_name, "args": args},
                "inputs_used": [],
            }
        )
    except Ms4ClientError as exc:
        return _block(f"MS4 fail-closed: ethics unavailable for {tool_name}: {exc}", config)

    if _decision_allows(decision):
        return None
    reason = decision.get("reason") or decision.get("resolution") or "MS3 ethics decision denied the action"
    return _block(f"MS4 ethics block for {tool_name}: {reason}", config)


def on_transform_llm_output(**kwargs: Any) -> str | None:
    response_text = str(kwargs.get("response_text", ""))
    cleaned, changed = strip_psyche_blocks(response_text)
    if not changed:
        return None

    _record_event(
        {
            "event_type": "advisory_psyche_block_stripped",
            "session_id": str(kwargs.get("session_id", "")),
            "model": str(kwargs.get("model", "")),
        }
    )
    return cleaned


def on_post_llm_call(**kwargs: Any) -> None:
    _record_event(
        {
            "event_type": "llm_call_observed",
            "session_id": str(kwargs.get("session_id", "")),
            "api_call_count": kwargs.get("api_call_count"),
        }
    )


_HIVEMIND_TOOLSET = "mcp-hivemind"
_HIVEMIND_EXACT_READ_TOOLSET = "mcp-hivemind-exact-read"
_HIVEMIND_EXACT_GATED_TOOLSET = "mcp-hivemind-exact-gated"
_HIVEMIND_EXACT_READ = "hivemind_exact_read"
_HIVEMIND_EXACT_GATED = "hivemind_exact_gated"
_HIVEMIND_SKILLS_LIST = "ms4_skills_list"
_HIVEMIND_SKILL_VIEW = "ms4_skill_view"
_HIVEMIND_SKILLS_LIST_DESCRIPTION = (
    "List the shared HiveMind/Hermes skills available to this Depth job. "
    "This is a read-only wrapper; use ms4_skill_view to load one skill."
)
_HIVEMIND_SKILL_VIEW_DESCRIPTION = (
    "Read one shared HiveMind/Hermes skill or one of its linked files. "
    "This wrapper returns raw stored content without running skill "
    "preprocessing and cannot create, edit, install, or delete skills."
)
_HIVEMIND_EXACT_READ_DESCRIPTION = (
    "Use this when the user names an exact canonical `hivemind.*` READ tool "
    "that lacks a dedicated wrapper. Never substitute another tool. Pass the "
    "exact canonical tool name and an arguments object (`{}` for zero-argument "
    "reads). If required arguments are missing, return the missing required "
    "arguments and ask the user for them."
)
_HIVEMIND_EXACT_GATED_DESCRIPTION = (
    "Delegate one exact canonical HiveMind gated mutation to HLI Oracle. "
    "HLI remains the sole mutation authority and must preserve Human Bridge "
    "approval. Never substitute another tool or claim success without one "
    "matching authoritative tool_trace entry."
)
_HIVEMIND_EXACT_GATED_TIMEOUT_SECS = 930
_HIVEMIND_EXACT_GATED_RESPONSE_MAX_BYTES = 1_048_576
_HIVEMIND_EXACT_GATED_ARGUMENTS_MAX_BYTES = 4096
_HIVEMIND_EXACT_GATED_ORIGINAL_GOAL_MAX_CHARS = 1024
_HIVEMIND_EXACT_GATED_PROMPT_MAX_CHARS = 6144
_HIVEMIND_EXACT_GATED_SESSION_MAX_CHARS = 128
_HIVEMIND_EXACT_GATED_MAX_ARGUMENTS = 16
_HIVEMIND_EXACT_GATED_BYPASS_KEYS = frozenset({
    "approval_token",
    "auto_approve",
    "approved",
    "authority",
    "authority_escalation",
    "bypass",
    "skip_approval",
})


def _hivemind_url() -> str:
    return (
        os.environ.get("MS4_HIVEMIND_URL")
        or os.environ.get("MS4_HIVEMIND_HLI_URL")
        or "http://127.0.0.1:6089"
    ).rstrip("/")


def _json_result(data: Any) -> str:
    if isinstance(data, str):
        try:
            json.loads(data)
            return data
        except json.JSONDecodeError:
            pass
    return json.dumps(data, ensure_ascii=False, default=str)


def _empty_schema(name: str, description: str) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    }


def _skills_list_schema() -> dict[str, Any]:
    return {
        "name": _HIVEMIND_SKILLS_LIST,
        "description": _HIVEMIND_SKILLS_LIST_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Optional exact category filter.",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    }


def _skill_view_schema() -> dict[str, Any]:
    return {
        "name": _HIVEMIND_SKILL_VIEW,
        "description": _HIVEMIND_SKILL_VIEW_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "minLength": 1,
                    "description": (
                        "Exact skill name returned by ms4_skills_list, including "
                        "a category or plugin namespace when present."
                    ),
                },
                "file_path": {
                    "type": "string",
                    "minLength": 1,
                    "description": (
                        "Optional relative linked-file path within the selected skill."
                    ),
                },
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    }


def _exact_read_schema() -> dict[str, Any]:
    return {
        "name": _HIVEMIND_EXACT_READ,
        "description": _HIVEMIND_EXACT_READ_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "canonical_tool": {
                    "type": "string",
                    "pattern": r"^hivemind\.",
                    "description": (
                        "Exact canonical HiveMind catalog name; aliases and "
                        "substitute tools are not accepted."
                    ),
                },
                "arguments": {
                    "type": "object",
                    "description": (
                        "Arguments for the canonical tool. Pass {} when the "
                        "catalog schema has no required arguments."
                    ),
                    "additionalProperties": True,
                },
            },
            "required": ["canonical_tool", "arguments"],
            "additionalProperties": False,
        },
    }


def _exact_gated_schema() -> dict[str, Any]:
    return {
        "name": _HIVEMIND_EXACT_GATED,
        "description": _HIVEMIND_EXACT_GATED_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "canonical_tool": {
                    "type": "string",
                    "pattern": r"^hivemind\.",
                    "maxLength": 160,
                    "description": "Exact validated canonical HiveMind gated tool.",
                },
                "arguments": {
                    "type": "object",
                    "maxProperties": _HIVEMIND_EXACT_GATED_MAX_ARGUMENTS,
                    "additionalProperties": True,
                    "description": "Flat bounded arguments for the exact tool.",
                },
                "original_goal": {
                    "type": "string",
                    "maxLength": _HIVEMIND_EXACT_GATED_ORIGINAL_GOAL_MAX_CHARS,
                    "description": "Original operator goal; never an approval token.",
                },
            },
            "required": ["canonical_tool", "arguments", "original_goal"],
            "additionalProperties": False,
        },
    }


def _exact_read_rejection(
    canonical_tool: str,
    *,
    status: str,
    code: str,
    message: str,
    **details: Any,
) -> str:
    return _json_result(
        {
            "status": status,
            "success": False,
            "canonical_tool": canonical_tool,
            "error": {"code": code, "message": message, **details},
        }
    )


def _handle_hivemind_time_now(_args: dict[str, Any] | None = None, **_kwargs: Any) -> Any:
    from machine_spirit_4.gateway import hivemind_tools

    return _json_result(hivemind_tools.time_now(_hivemind_url()))


def _handle_hivemind_models_list(_args: dict[str, Any] | None = None, **_kwargs: Any) -> Any:
    from machine_spirit_4.gateway import hivemind_tools

    return _json_result(hivemind_tools.models_list(_hivemind_url()))


def _handle_hivemind_capability_matrix(_args: dict[str, Any] | None = None, **_kwargs: Any) -> Any:
    from machine_spirit_4.gateway import hivemind_tools

    return _json_result(hivemind_tools.capability_matrix(_hivemind_url()))


def _handle_hivemind_vm_list(_args: dict[str, Any] | None = None, **_kwargs: Any) -> Any:
    from machine_spirit_4.gateway import hivemind_tools

    return _json_result(hivemind_tools.vm_list(_hivemind_url()))


def _handle_hivemind_gpu_availability(_args: dict[str, Any] | None = None, **_kwargs: Any) -> Any:
    from machine_spirit_4.gateway import hivemind_tools

    return _json_result(hivemind_tools.gpu_availability(_hivemind_url()))


def _handle_hivemind_cluster_summary(_args: dict[str, Any] | None = None, **_kwargs: Any) -> Any:
    from machine_spirit_4.gateway import hivemind_tools

    args = _args if isinstance(_args, dict) else {}
    return _json_result(
        hivemind_tools.cluster_summary(
            _hivemind_url(),
            include_gpu_details=bool(args.get("include_gpu_details", False)),
        )
    )


def _handle_hivemind_hosts_list(_args: dict[str, Any] | None = None, **_kwargs: Any) -> Any:
    from machine_spirit_4.gateway import hivemind_tools

    args = _args if isinstance(_args, dict) else {}
    return _json_result(
        hivemind_tools.hosts_list(
            _hivemind_url(),
            status_filter=str(args.get("status_filter") or "all"),
        )
    )


def _handle_hivemind_active_jobs(_args: dict[str, Any] | None = None, **_kwargs: Any) -> Any:
    from machine_spirit_4.gateway import hivemind_state

    return _json_result(hivemind_state.get_active_jobs(_hivemind_url()))


def _handle_hivemind_cluster_load(_args: dict[str, Any] | None = None, **_kwargs: Any) -> Any:
    from machine_spirit_4.gateway import hivemind_state

    return _json_result(hivemind_state.get_cluster_load(_hivemind_url()))


def _handle_hivemind_service_health(_args: dict[str, Any] | None = None, **_kwargs: Any) -> Any:
    from machine_spirit_4.gateway import hivemind_state

    return _json_result(hivemind_state.get_service_health(_hivemind_url()))


def _handle_hivemind_state_snapshot(_args: dict[str, Any] | None = None, **_kwargs: Any) -> Any:
    from machine_spirit_4.gateway import hivemind_state

    return _json_result(hivemind_state.get_combined_snapshot(_hivemind_url()))


def _skill_task_id(kwargs: dict[str, Any]) -> str | None:
    task_id = kwargs.get("task_id")
    if task_id is None:
        return None
    return str(task_id)


def _handle_ms4_skills_list(
    _args: dict[str, Any] | None = None,
    **_kwargs: Any,
) -> Any:
    request = _args if isinstance(_args, dict) else {}
    category = request.get("category")
    if category is not None and not isinstance(category, str):
        return _json_result(
            {"success": False, "error": "category must be a string when provided."}
        )
    try:
        from tools import skills_tool

        if not skills_tool._skills_dir().is_dir():
            return _json_result(
                {
                    "success": False,
                    "error": (
                        "Hermes skills directory is unavailable; the read-only "
                        "wrapper will not create it."
                    ),
                }
            )

        return _json_result(
            skills_tool.skills_list(
                category=category,
                task_id=_skill_task_id(_kwargs),
            )
        )
    except Exception as exc:
        return _json_result(
            {"success": False, "error": f"Shared skill listing failed: {exc}"}
        )


def _handle_ms4_skill_view(
    _args: dict[str, Any] | None = None,
    **_kwargs: Any,
) -> Any:
    request = _args if isinstance(_args, dict) else {}
    name = request.get("name")
    file_path = request.get("file_path")
    if not isinstance(name, str) or not name.strip():
        return _json_result(
            {"success": False, "error": "name must be a non-empty string."}
        )
    if file_path is not None and (
        not isinstance(file_path, str) or not file_path.strip()
    ):
        return _json_result(
            {
                "success": False,
                "error": "file_path must be a non-empty string when provided.",
            }
        )
    try:
        from tools.skills_tool import skill_view

        return _json_result(
            skill_view(
                name=name,
                file_path=file_path,
                task_id=_skill_task_id(_kwargs),
                preprocess=False,
            )
        )
    except Exception as exc:
        return _json_result(
            {"success": False, "error": f"Shared skill read failed: {exc}"}
        )


def _handle_hivemind_exact_read(
    _args: dict[str, Any] | None = None,
    **_kwargs: Any,
) -> Any:
    request = _args if isinstance(_args, dict) else {}
    canonical_raw = request.get("canonical_tool")
    canonical_tool = canonical_raw if isinstance(canonical_raw, str) else ""
    arguments = request.get("arguments")
    if not canonical_tool:
        return _exact_read_rejection(
            canonical_tool,
            status="rejected",
            code="invalid_request",
            message="canonical_tool must be a non-empty string.",
        )
    if not isinstance(arguments, dict):
        return _exact_read_rejection(
            canonical_tool,
            status="rejected",
            code="invalid_request",
            message="arguments must be an object; use {} for zero-argument reads.",
        )

    hivemind_url = _hivemind_url()
    try:
        from machine_spirit_4.gateway.quartermaster import catalog as catalog_module

        catalog = catalog_module.get_catalog(hivemind_url)
    except Exception as exc:
        return _exact_read_rejection(
            canonical_tool,
            status="catalog_unavailable",
            code="catalog_load_failed",
            message=f"Quartermaster catalog could not be loaded: {exc}",
        )

    catalog_errors = getattr(catalog, "errors", None)
    if not isinstance(catalog_errors, (list, tuple)):
        return _exact_read_rejection(
            canonical_tool,
            status="catalog_unavailable",
            code="catalog_unverified",
            message="Quartermaster catalog did not expose a verifiable error state.",
        )
    if catalog_errors:
        return _exact_read_rejection(
            canonical_tool,
            status="catalog_unavailable",
            code="catalog_incomplete",
            message="Quartermaster catalog reported load errors; exact reads fail closed.",
            catalog_errors=[str(error) for error in catalog_errors],
        )

    try:
        entries_by_name = catalog.by_name()
    except Exception as exc:
        return _exact_read_rejection(
            canonical_tool,
            status="catalog_unavailable",
            code="catalog_unverified",
            message=f"Quartermaster catalog index is unavailable: {exc}",
        )
    entry = entries_by_name.get(canonical_tool) if isinstance(entries_by_name, dict) else None
    if entry is None:
        return _exact_read_rejection(
            canonical_tool,
            status="rejected",
            code="unknown_tool",
            message="Exact canonical tool was not found in the current Quartermaster catalog.",
        )
    if (
        not canonical_tool.startswith("hivemind.")
        or entry.name != canonical_tool
        or entry.source != catalog_module.SOURCE_HIVEMIND
    ):
        return _exact_read_rejection(
            canonical_tool,
            status="rejected",
            code="noncanonical_tool",
            message="Only exact canonical HiveMind-native tool names are accepted.",
        )
    if not catalog_module.is_inline_eligible(entry):
        return _exact_read_rejection(
            canonical_tool,
            status="rejected",
            code="unsafe_tool",
            message=(
                "The current shared Quartermaster safety classification does "
                "not permit this tool as a read-only inline-safe operation."
            ),
        )

    input_schema = entry.input_schema
    if not isinstance(input_schema, dict):
        return _exact_read_rejection(
            canonical_tool,
            status="rejected",
            code="schema_unverified",
            message="The exact catalog entry has no verifiable input schema.",
        )
    required_raw = input_schema.get("required", [])
    if (
        not isinstance(required_raw, list)
        or any(not isinstance(key, str) or not key for key in required_raw)
    ):
        return _exact_read_rejection(
            canonical_tool,
            status="rejected",
            code="schema_unverified",
            message="The exact catalog entry has an invalid required-key schema.",
        )
    missing_required_keys = sorted(
        {
            key
            for key in required_raw
            if key not in arguments
            or arguments[key] is None
            or (
                isinstance(arguments[key], str)
                and not arguments[key].strip()
            )
        }
    )
    if missing_required_keys:
        return _json_result(
            {
                "status": "missing_required_arguments",
                "success": False,
                "canonical_tool": canonical_tool,
                "missing_required_keys": missing_required_keys,
            }
        )

    try:
        from machine_spirit_4.gateway import hivemind_tools

        authoritative_result = hivemind_tools._call_tool(
            hivemind_url,
            canonical_tool,
            arguments,
        )
    except Exception as exc:
        return _exact_read_rejection(
            canonical_tool,
            status="error",
            code="transport_failed",
            message=f"HiveMind MCP exact read failed: {exc}",
        )
    return _json_result(
        {
            "status": "success",
            "success": True,
            "canonical_tool": canonical_tool,
            "result": authoritative_result,
        }
    )


def _canonical_to_hli_oracle_tool(canonical_tool: str) -> str | None:
    """Mirror HLI oracle.rs canonicalize_tool_name exactly."""
    if not isinstance(canonical_tool, str) or not canonical_tool.startswith("hivemind."):
        return None
    base, marker, version = canonical_tool.rpartition("@v")
    if not marker or not version or not version.isascii() or not version.isdigit():
        return None
    if not base or any(
        not segment
        or any(not (char.isascii() and (char.isalnum() or char in "_-")) for char in segment)
        for segment in base.split(".")
    ):
        return None
    return base.replace(".", "_")


def _catalog_exact_gated_alias(canonical_tool: str) -> tuple[str | None, str | None]:
    try:
        from machine_spirit_4.gateway.quartermaster import catalog as catalog_module

        catalog = catalog_module.get_catalog(_hivemind_url())
    except Exception as exc:
        return None, f"Quartermaster catalog could not be loaded: {exc}"
    errors = getattr(catalog, "errors", None)
    if not isinstance(errors, (list, tuple)) or errors:
        return None, "Quartermaster catalog is incomplete or unverifiable."
    try:
        entries = catalog.by_name()
    except Exception as exc:
        return None, f"Quartermaster catalog index is unavailable: {exc}"
    entry = entries.get(canonical_tool) if isinstance(entries, dict) else None
    if (
        entry is None
        or entry.name != canonical_tool
        or entry.source != catalog_module.SOURCE_HIVEMIND
        or entry.kind != catalog_module.KIND_HIVEMIND_NATIVE
    ):
        return None, "Exact canonical HiveMind-native tool was not found."
    if catalog_module.is_inline_eligible(entry):
        return None, "Read-only inline-safe tools are not accepted by the gated bridge."

    try:
        manifest_entries, manifest_errors = catalog_module._load_ms4_manifest()
    except Exception as exc:
        return None, f"Shared MS4 gating policy could not be loaded: {exc}"
    if manifest_errors:
        return None, "Shared MS4 gating policy is incomplete."
    proxy_name = f"ms4.{canonical_tool}"
    proxies = [
        raw
        for raw in manifest_entries
        if isinstance(raw, dict) and raw.get("name") == proxy_name
    ]
    if len(proxies) != 1:
        return None, "Exact canonical tool has no unique shared MS4 policy proxy."
    proxy = proxies[0]
    gated_by = proxy.get("gated_by")
    if isinstance(gated_by, str):
        gates = [gated_by]
    elif isinstance(gated_by, list):
        gates = [gate for gate in gated_by if isinstance(gate, str) and gate]
    else:
        gates = []
    if proxy.get("kind") != catalog_module.KIND_RUNTIME_ACTION or not gates:
        return None, "Exact canonical tool is not classified as a gated runtime action."
    if any("dry" in gate.casefold() for gate in gates):
        return None, "Dry-run-only tools are not accepted by the gated bridge."
    alias = _canonical_to_hli_oracle_tool(canonical_tool)
    if alias is None:
        return None, "Canonical tool has no source-valid HLI Oracle equivalent."
    return alias, None


def _bounded_exact_gated_arguments(arguments: Any) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(arguments, dict) or len(arguments) > _HIVEMIND_EXACT_GATED_MAX_ARGUMENTS:
        return None, "arguments must be a bounded object."
    bounded: dict[str, Any] = {}
    for key, value in arguments.items():
        if (
            not isinstance(key, str)
            or not key
            or len(key) > 64
            or not key.replace("_", "").isalnum()
            or key.casefold() in _HIVEMIND_EXACT_GATED_BYPASS_KEYS
        ):
            return None, f"argument key {key!r} is not permitted."
        if isinstance(value, str):
            if (
                len(value) > 512
                or any(ord(char) < 32 or ord(char) == 127 for char in value)
            ):
                return None, f"argument {key!r} exceeds the bounded scalar contract."
        elif not isinstance(value, (bool, int, float)):
            return None, f"argument {key!r} must be a flat JSON scalar."
        bounded[key] = value
    try:
        encoded = json.dumps(
            bounded,
            ensure_ascii=True,
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError):
        return None, "arguments are not finite JSON scalars."
    if len(encoded.encode("utf-8")) > _HIVEMIND_EXACT_GATED_ARGUMENTS_MAX_BYTES:
        return None, "arguments exceed the bounded JSON limit."
    return bounded, None


def _post_hli_oracle_chat(
    url: str,
    payload: dict[str, Any],
    *,
    timeout: int,
) -> Any:
    from machine_spirit_4.gateway.hivemind_state import hivemind_auth_headers

    body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        **hivemind_auth_headers(),
    }
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if not 200 <= int(response.status) < 300:
                raise RuntimeError(f"HLI Oracle HTTP {response.status}")
            raw = response.read(_HIVEMIND_EXACT_GATED_RESPONSE_MAX_BYTES + 1)
    except urllib.error.HTTPError as exc:
        detail = exc.read(512).decode("utf-8", "replace")
        raise RuntimeError(f"HLI Oracle HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise RuntimeError(f"HLI Oracle request failed: {exc}") from exc
    if len(raw) > _HIVEMIND_EXACT_GATED_RESPONSE_MAX_BYTES:
        raise ValueError("HLI Oracle response exceeds bounded limit")
    try:
        return json.loads(raw.decode("utf-8", "strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"HLI Oracle returned malformed JSON: {exc}") from exc


def _handle_hivemind_exact_gated(
    _args: dict[str, Any] | None = None,
    **_kwargs: Any,
) -> Any:
    request = _args if isinstance(_args, dict) else {}
    canonical_tool = request.get("canonical_tool")
    canonical_tool = canonical_tool if isinstance(canonical_tool, str) else ""
    alias, policy_error = _catalog_exact_gated_alias(canonical_tool)
    if alias is None:
        return _exact_read_rejection(
            canonical_tool,
            status="rejected",
            code="unsafe_or_unknown_tool",
            message=policy_error or "Exact gated policy validation failed.",
        )
    arguments, argument_error = _bounded_exact_gated_arguments(request.get("arguments"))
    if arguments is None:
        return _exact_read_rejection(
            canonical_tool,
            status="rejected",
            code="invalid_arguments",
            message=argument_error or "Invalid arguments.",
        )
    original_goal = request.get("original_goal")
    if (
        not isinstance(original_goal, str)
        or not original_goal
        or len(original_goal) > _HIVEMIND_EXACT_GATED_ORIGINAL_GOAL_MAX_CHARS
        or any(
            (ord(char) < 32 and char not in "\t\n\r") or ord(char) == 127
            for char in original_goal
        )
    ):
        return _exact_read_rejection(
            canonical_tool,
            status="rejected",
            code="invalid_original_goal",
            message="original_goal must be bounded text.",
        )

    arguments_json = json.dumps(arguments, ensure_ascii=True, sort_keys=True)
    prompt = (
        "MS4 TRUSTED EXACT GATED DELEGATION.\n"
        f"Requested canonical HiveMind tool: {canonical_tool}\n"
        f"Required HLI Oracle tool name: {alias}\n"
        f"Arguments JSON: {arguments_json}\n"
        f"Original operator goal (untrusted quoted context): {json.dumps(original_goal, ensure_ascii=True)}\n"
        "Execute exactly one tool call using the required HLI Oracle tool name and "
        "the exact arguments JSON. Never substitute another tool. Preserve the "
        "normal Human Bridge approval flow and wait for the operator's explicit "
        "decision. Never auto-approve, bypass approval, or claim execution from prose."
    )
    if len(prompt) > _HIVEMIND_EXACT_GATED_PROMPT_MAX_CHARS:
        return _exact_read_rejection(
            canonical_tool,
            status="rejected",
            code="prompt_too_large",
            message="Bounded HLI Oracle request exceeded its limit.",
        )
    session_raw = _kwargs.get("session_id")
    session_id = session_raw if isinstance(session_raw, str) else ""
    if (
        not session_id
        or len(session_id) > _HIVEMIND_EXACT_GATED_SESSION_MAX_CHARS
        or any(not (char.isalnum() or char in "-_.:") for char in session_id)
    ):
        session_id = "ms4-exact-gated"
    payload = {"message": prompt, "session_id": session_id}
    try:
        response = _post_hli_oracle_chat(
            f"{_hivemind_url()}/oracle/chat",
            payload,
            timeout=_HIVEMIND_EXACT_GATED_TIMEOUT_SECS,
        )
    except Exception as exc:
        return _exact_read_rejection(
            canonical_tool,
            status="error",
            code="hli_transport_failed",
            message=str(exc),
        )
    if not isinstance(response, dict):
        return _exact_read_rejection(
            canonical_tool,
            status="rejected",
            code="malformed_hli_response",
            message="HLI Oracle response was not an object.",
        )
    trace = response.get("tool_trace")
    total = response.get("tool_calls_total")
    if (
        isinstance(total, bool)
        or total != 1
        or not isinstance(trace, list)
        or len(trace) != 1
        or not isinstance(trace[0], dict)
    ):
        return _exact_read_rejection(
            canonical_tool,
            status="rejected",
            code="unverified_tool_trace",
            message="HLI Oracle did not return exactly one real tool trace.",
        )
    item = trace[0]
    trace_name = item.get("name")
    if trace_name not in {canonical_tool, alias}:
        return _exact_read_rejection(
            canonical_tool,
            status="rejected",
            code="substituted_tool",
            message="HLI Oracle executed a different tool.",
            actual_tool=trace_name,
        )
    if item.get("args") != arguments or item.get("success") is not True:
        return _exact_read_rejection(
            canonical_tool,
            status="rejected",
            code="unsuccessful_tool_trace",
            message="HLI Oracle trace arguments or success state were not authoritative.",
        )
    if (
        isinstance(item.get("round"), bool)
        or not isinstance(item.get("round"), int)
        or item["round"] < 1
        or not isinstance(item.get("tool_call_id"), str)
        or not item["tool_call_id"]
        or isinstance(item.get("duration_ms"), bool)
        or not isinstance(item.get("duration_ms"), int)
        or item["duration_ms"] < 0
        or isinstance(item.get("result_bytes"), bool)
        or not isinstance(item.get("result_bytes"), int)
        or item["result_bytes"] < 0
        or not isinstance(item.get("timestamp"), str)
        or not item["timestamp"]
    ):
        return _exact_read_rejection(
            canonical_tool,
            status="rejected",
            code="malformed_tool_trace",
            message="HLI Oracle tool trace was malformed.",
        )
    response_text = response.get("response")
    if not isinstance(response_text, str):
        return _exact_read_rejection(
            canonical_tool,
            status="rejected",
            code="malformed_hli_response",
            message="HLI Oracle response text was missing.",
        )
    return _json_result(
        {
            "status": "success",
            "success": True,
            "authoritative": True,
            "canonical_tool": canonical_tool,
            "hli_tool": trace_name,
            "response": response_text,
            "tool_trace": trace,
        }
    )


_HIVEMIND_TOOLS: tuple[tuple[str, str, Any], ...] = (
    (
        _HIVEMIND_SKILLS_LIST,
        _HIVEMIND_SKILLS_LIST_DESCRIPTION,
        _handle_ms4_skills_list,
    ),
    (
        _HIVEMIND_SKILL_VIEW,
        _HIVEMIND_SKILL_VIEW_DESCRIPTION,
        _handle_ms4_skill_view,
    ),
    (
        "hivemind_time_now",
        "Read the authoritative HiveMind cluster time through the MCP gateway.",
        _handle_hivemind_time_now,
    ),
    (
        "hivemind_models_list",
        "Read the resilient HiveMind model catalog through the MCP gateway.",
        _handle_hivemind_models_list,
    ),
    (
        "hivemind_capability_matrix",
        "Read per-node HiveMind capabilities, loaded models, and available runtimes.",
        _handle_hivemind_capability_matrix,
    ),
    (
        "hivemind_vm_list",
        "Read the HiveMind VM inventory across the cluster.",
        _handle_hivemind_vm_list,
    ),
    (
        "hivemind_gpu_availability",
        "Read current HiveMind GPU scheduling availability.",
        _handle_hivemind_gpu_availability,
    ),
    (
        "hivemind_cluster_summary",
        "Read HiveMind cluster node, GPU, memory, and activity summary.",
        _handle_hivemind_cluster_summary,
    ),
    (
        "hivemind_hosts_list",
        "Read HiveMind cluster nodes with hardware, status, and last-seen data.",
        _handle_hivemind_hosts_list,
    ),
    (
        "hivemind_active_jobs",
        "Read active HiveMind jobs across inference, pulls, scatter, and training.",
        _handle_hivemind_active_jobs,
    ),
    (
        "hivemind_cluster_load",
        "Read HiveMind cluster load, trackers, deadlines, retries, and peer state.",
        _handle_hivemind_cluster_load,
    ),
    (
        "hivemind_service_health",
        "Read HiveMind service health through the MCP gateway.",
        _handle_hivemind_service_health,
    ),
    (
        "hivemind_state_snapshot",
        "Read the combined MS4 HiveMind state snapshot: active jobs, load, and service health.",
        _handle_hivemind_state_snapshot,
    ),
)


def _register_hivemind_tools(ctx: Any) -> None:
    for name, description, handler in _HIVEMIND_TOOLS:
        if name == _HIVEMIND_SKILLS_LIST:
            schema = _skills_list_schema()
        elif name == _HIVEMIND_SKILL_VIEW:
            schema = _skill_view_schema()
        else:
            schema = _empty_schema(name, description)
        ctx.register_tool(
            name=name,
            toolset=_HIVEMIND_TOOLSET,
            schema=schema,
            handler=handler,
            description=description,
        )
    ctx.register_tool(
        name=_HIVEMIND_EXACT_READ,
        toolset=_HIVEMIND_EXACT_READ_TOOLSET,
        schema=_exact_read_schema(),
        handler=_handle_hivemind_exact_read,
        description=_HIVEMIND_EXACT_READ_DESCRIPTION,
    )
    ctx.register_tool(
        name=_HIVEMIND_EXACT_GATED,
        toolset=_HIVEMIND_EXACT_GATED_TOOLSET,
        schema=_exact_gated_schema(),
        handler=_handle_hivemind_exact_gated,
        description=_HIVEMIND_EXACT_GATED_DESCRIPTION,
    )


def _verify_identity(client: Ms4Client, config: Ms4Config, session_id: str = "") -> dict[str, Any]:
    global _identity_cache, _last_error
    try:
        identity = client.verify_identity(config.spirit_id)
        normalized = _normalize_identity(identity)
        if not normalized.get("verified"):
            raise Ms4ClientError("identity response was not verified")
        _identity_cache = normalized
        _last_error = None
        try:
            client.heartbeat(config.spirit_id, session_id=session_id)
        except Ms4ClientError:
            pass
        return {"status": "verified", "spirit_id": config.spirit_id}
    except Ms4ClientError as exc:
        _last_error = str(exc)
        if config.fail_closed:
            return {"status": "blocked", "reason": _last_error}
        return {"status": "degraded", "reason": _last_error}


def _normalize_identity(identity: dict[str, Any]) -> dict[str, Any]:
    anchor = identity.get("anchor") if isinstance(identity.get("anchor"), dict) else {}
    return {
        "verified": bool(identity.get("verified") or identity.get("identity_confirmed")),
        "spirit_id": identity.get("spirit_id") or anchor.get("spirit_id"),
        "name": identity.get("name") or anchor.get("name"),
        "chosen_name": identity.get("chosen_name") or anchor.get("chosen_name"),
        "glyph": identity.get("glyph") or anchor.get("glyph"),
        "raw": identity,
    }


def _ensure_identity(client: Ms4Client, config: Ms4Config, session_id: str = "") -> None:
    if _identity_cache and _identity_cache.get("verified"):
        return
    result = _verify_identity(client, config, session_id=session_id)
    if result.get("status") != "verified" and config.fail_closed:
        raise Ms4ClientError(str(result.get("reason") or "identity verification failed"))


def _safe_get_state(client: Ms4Client, config: Ms4Config) -> dict[str, Any] | None:
    global _state_cache, _last_error
    try:
        _state_cache = client.get_state(config.spirit_id)
    except Ms4ClientError as exc:
        _last_error = str(exc)
    return _state_cache


def _record_event(event: dict[str, Any]) -> None:
    config = load_config()
    client = Ms4Client(config.ms3_sidecar_url)
    try:
        client.record_event({"spirit_id": config.spirit_id, **event})
    except Ms4ClientError:
        pass


def _decision_allows(decision: dict[str, Any]) -> bool:
    allowed = decision.get("allowed")
    if isinstance(allowed, bool):
        return allowed
    decision_text = str(decision.get("decision", "")).lower()
    if decision_text:
        return decision_text == "allow"
    safety = decision.get("safety")
    if isinstance(safety, dict) and safety.get("hard_block") is True:
        return False
    resolution = str(decision.get("resolution", "")).lower()
    return resolution in {"allow", "allowed", "approved"}


def _block(message: str, config: Ms4Config) -> dict[str, str] | None:
    if not config.fail_closed:
        return None
    return {"action": "block", "message": message}


def _classify_tool(tool_name: str) -> str:
    name = tool_name.lower()
    if "memory" in name and any(token in name for token in ("write", "save", "store", "delete")):
        return "memory_write"
    if "identity" in name and any(token in name for token in ("write", "set", "mutate", "delete")):
        return "identity_mutation"
    if any(token in name for token in ("scare", "strobe", "light", "speak")):
        return "scare" if "scare" in name else "light"
    if any(token in name for token in ("servo", "motor", "physical")):
        return "physical"
    return "tool"


def _risk_class_for_tool(tool_name: str, args: dict[str, Any], action_type: str) -> str:
    name = tool_name.lower()
    command = str(args.get("command", "")).lower()
    destructive_markers = (
        " rm ",
        " del ",
        "remove-item",
        "format ",
        "shutdown",
        "restart-computer",
        "reg delete",
        "cipher /w",
        "diskpart",
    )
    if name in {"terminal", "process", "execute_code"}:
        if any(marker in f" {command} " for marker in destructive_markers):
            return "high"
        return "medium"
    if any(token in name for token in ("write", "patch", "delete", "browser", "click", "type", "press")):
        return "medium"
    if action_type in {"identity_mutation", "physical", "scare"}:
        return "high"
    if action_type in {"memory_write", "speak", "light"}:
        return "medium"
    return "low"
