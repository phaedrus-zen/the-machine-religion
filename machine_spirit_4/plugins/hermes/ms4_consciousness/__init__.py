"""MS4 consciousness integration plugin for Hermes."""

from __future__ import annotations

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
