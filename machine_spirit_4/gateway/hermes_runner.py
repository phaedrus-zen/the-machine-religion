from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


log = logging.getLogger("ms4.gateway.hermes_runner")


class Ms4TurnCancelled(RuntimeError):
    """Finding 3: raised when a foreground turn is cancelled (barge/disconnect)
    before it commits any conversation side effect — no revision bump, no Depth
    dispatch, no history append. A stale/cancelled turn can never mutate the
    session it lost ownership of."""


# Default Hermes agent iteration cap. Foreground (Face Lobe direct chat)
# rarely loops, but the Depth Lobe worker can. Both are env-tunable so an
# operator can trade depth for latency without a code change.
_DEFAULT_MAX_ITERATIONS = 12

# Depth answers have a stricter completion contract than the latency-sensitive
# Face path.  Leaving Hermes at the provider default can strand a reasoning
# model inside a 2K context before it emits any visible final answer.  Give
# background agents an explicit, bounded output budget while leaving foreground
# agents unchanged.
_DEFAULT_DEPTH_MAX_TOKENS = 4096
_MIN_DEPTH_MAX_TOKENS = 4096
_MAX_DEPTH_MAX_TOKENS = 8192
_DEFAULT_DEPTH_REASONING_EFFORT = "low"
_DEPTH_REASONING_EFFORTS = frozenset({"minimal", "low", "medium", "high"})
_DEPTH_REASONING_DISABLED = frozenset(
    {"0", "disabled", "false", "none", "no", "off"}
)

_VOICE_SUBSTANTIVE_MIN_WORDS = 150
_VOICE_SUBSTANTIVE_MAX_WORDS = 380

# Appended to the Face Lobe system prompt on voice turns only. Keep spoken
# delivery efficient without replacing requested substance with a teaser.
_VOICE_BREVITY_DIRECTIVE = (
    "VOICE MODE — your reply will be spoken aloud by text-to-speech:\n"
    "- Lead with the answer. Do not omit requested substance or replace it with an invitation.\n"
    "- For every substantive question or contextual follow-up, aim for 180 to 240 visible words. "
    "150 is a hard floor and 380 is a hard ceiling. This includes requests to explain, "
    "compare, diagnose, recommend, expand, revise, synthesize, or go deeper. Only greetings, "
    "acknowledgements, and single-fact lookups may be shorter.\n"
    "- For very long lists, tables, inventories, URLs, paths, or code, speak a meaningful "
    "bounded synthesis and state that the complete answer is visible in the transcript.\n"
    "- For ordinary multi-part questions, answer every requested part even when concise.\n"
    "- No headings, bullets, or markdown — this is being spoken, not read."
)

_VOICE_INLINE_MAX_CHARS = 240
_VOICE_INLINE_TRUSTED_SCALARS = (
    "healthy",
    "state",
    "status",
    "available",
    "ready",
    "active_requests",
    "queued_requests",
    "running_plans",
    "running_jobs",
    "total_active",
)
_CORRELATION_MARKER_RE = re.compile(
    r"\bcorrelation\s+marker\b\s*[:#-]?\s*([^.!?;\r\n]{1,80})",
    re.IGNORECASE,
)


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    """Read a positive int from the environment, clamped to ``minimum``.

    Falls back to ``default`` on missing/empty/non-integer values so a
    typo in an env var can never crash the runtime — it just keeps the
    documented default.
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return max(minimum, int(raw.strip()))
    except (TypeError, ValueError):
        log.warning("ms4: ignoring non-integer %s=%r; using default %d", name, raw, default)
        return default


def foreground_max_iterations() -> int:
    """Hermes ``max_iterations`` for the foreground/session agent path."""
    return _env_int("MS4_HERMES_MAX_ITERATIONS", _DEFAULT_MAX_ITERATIONS)


def background_max_iterations() -> int:
    """Hermes ``max_iterations`` for Double Agent Depth Lobe workers.
    Falls back to the foreground value when ``MS4_DA_MAX_ITERATIONS`` is
    unset so a single override still affects both."""
    return _env_int("MS4_DA_MAX_ITERATIONS", foreground_max_iterations())


def background_max_tokens() -> int:
    """Bounded output budget for Double Agent Depth model calls.

    The explicit top-level budget also lets HLI size the upstream context before
    OpenAI-to-Ollama translation.  Invalid or dangerous overrides fall back to
    the completion-safe default instead of silently creating a tiny or runaway
    request.
    """
    name = "MS4_DA_MAX_TOKENS"
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return _DEFAULT_DEPTH_MAX_TOKENS
    try:
        value = int(raw.strip())
    except (TypeError, ValueError):
        value = 0
    if not (_MIN_DEPTH_MAX_TOKENS <= value <= _MAX_DEPTH_MAX_TOKENS):
        log.warning(
            "ms4: ignoring out-of-range %s=%r; using default %d",
            name,
            raw,
            _DEFAULT_DEPTH_MAX_TOKENS,
        )
        return _DEFAULT_DEPTH_MAX_TOKENS
    return value


def background_reasoning_config() -> dict[str, Any]:
    """Explicit reasoning policy for Double Agent Depth model calls.

    Low reasoning preserves the value of a large Depth lobe without allowing
    hidden reasoning to consume the entire answer budget.  Operators can turn
    reasoning off or select another supported effort without changing Face.
    """
    name = "MS4_DA_REASONING_EFFORT"
    raw = os.environ.get(name)
    normalized = (raw or _DEFAULT_DEPTH_REASONING_EFFORT).strip().lower()
    if normalized in _DEPTH_REASONING_DISABLED:
        return {"enabled": False}
    if normalized in {"1", "enabled", "on", "true", "yes"}:
        normalized = _DEFAULT_DEPTH_REASONING_EFFORT
    if normalized not in _DEPTH_REASONING_EFFORTS:
        log.warning(
            "ms4: ignoring unsupported %s=%r; using default %s",
            name,
            raw,
            _DEFAULT_DEPTH_REASONING_EFFORT,
        )
        normalized = _DEFAULT_DEPTH_REASONING_EFFORT
    return {"enabled": True, "effort": normalized}


def _try_cluster_time_now(hivemind_url: str) -> str | None:
    """Best-effort cluster-time lookup via ``hivemind.time.now@v1``.

    Returns the cluster's ISO timestamp string when reachable, ``None``
    on any failure (offline cluster, missing tool, malformed response).
    Capped at 2s so it can't delay a chat turn.
    """
    try:
        from . import hivemind_tools

        body = hivemind_tools.time_now(hivemind_url)
    except Exception:
        return None
    if isinstance(body, dict):
        for key in ("iso", "iso8601", "now", "datetime", "time"):
            value = body.get(key)
            if isinstance(value, str) and value:
                return value
    if isinstance(body, str):
        return body
    return None


def _now_local_iso() -> str:
    """Return the current local date/time as ISO-8601 with timezone.

    Used to ground the Face Lobe on every turn so questions like
    "what is the date?" don't require a Depth Lobe dispatch. Local
    time (with tzinfo) is what the operator actually wants to see.
    """
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _canned_chat_failure_text(exc: Exception) -> str:
    """Translate a FaceLobeChat exception into a short user-visible
    reply that's safe to feed into TTS. The raw exception string
    (full URL, timeout numbers, etc.) is operator detail and goes to
    metrics.error / the audit log, not the user.

    Tailored to the three failure modes seen live:
      - HiveMind unreachable
      - Streaming chat completion stalled / timed out
      - Server returned a non-2xx (5xx)
    """
    msg = str(exc).lower()
    if "stream" in msg and ("timed out" in msg or "unreachable" in msg):
        return (
            "I'm having trouble reaching the language cluster's streaming "
            "chat endpoint right now. Give it a moment and try again, or "
            "switch the Face Lobe to a lighter chat model."
        )
    if "unreachable" in msg or "connection refused" in msg or "name or service" in msg:
        return (
            "HiveMind looks unreachable from MS4 right now. Check that "
            "the cluster is running on the configured URL and try again."
        )
    if "5" in msg and ("502" in msg or "503" in msg or "500" in msg):
        return (
            "HiveMind returned a server error on the chat completion. "
            "It's usually transient — try the same prompt again in a "
            "few seconds."
        )
    # Catch-all canned text. The operator can dig the real exception out
    # of metrics.error.
    return (
        "Something went wrong reaching the language model. The operator "
        "can check the audit log for the exact error."
    )


def _combine_grounding(
    grounding_source: str | None,
    face_lobe_block: str | None,
    dispatched_job: dict[str, Any] | None,
) -> str:
    """Combine the grounding-source labels so the operator can see at a
    glance what context layers were applied to a turn."""
    parts: list[str] = []
    if grounding_source:
        parts.append(grounding_source)
    if face_lobe_block:
        parts.append("face_lobe")
    if dispatched_job and isinstance(dispatched_job, dict) and dispatched_job.get("job_id"):
        parts.append("depth_lobe_dispatched")
    elif dispatched_job and isinstance(dispatched_job, dict) and dispatched_job.get("error"):
        parts.append("depth_lobe_dispatch_failed")
    return "+".join(parts) if parts else "none"

from machine_spirit_4.double_agent import (
    DepthChoice,
    JobEnvelope,
    JobEvent,
    ResourceRequest,
    SchemaError,
    build_face_lobe_context_block,
    choose_depth_model,
    choose_foreground_model,
    depth_fallback_model,
    default_runner,
    face_lobe_turn_start,
    is_reasoning_only_analytical,
    router_route,
)
from machine_spirit_4.double_agent.continuation import phrase_is_continuation
from machine_spirit_4.double_agent.safety import INTERNAL_GOAL_MAX_CHARS, new_job_id
from machine_spirit_4.double_agent.schemas import sanitize_prior_context

from .audit import append_event
from .context import build_grounded_user_message
from .face_lobe_chat import FaceLobeChat, FaceLobeChatError


import re as _re

# Markers that a Face Lobe reply OFFERED to run a Depth Lobe job (so the
# user's next "ok do that" should actually dispatch it). Deliberately
# specific to avoid false positives on incidental mentions.
_DEEP_OFFER_RE = _re.compile(
    r"(/deep\b|prefix\s+[`'\"]?/?deep|dispatch(?:ing)?\s+(?:a\s+)?(?:deep|depth|background)"
    r"|depth[ -]?lobe\s+job|spin\s+(?:one|it)\s+up|kick\s+(?:one|it)\s+off"
    r"|run\s+it\s+in\s+the\s+background"
    r"|i['’]?ll\s+(?:run|handle|fetch|get|look\s+into|take\s+care\s+of|pull\s+up|check|grab|dispatch)"
    r"|let\s+me\s+(?:run|check|pull|fetch|grab|look))",
    _re.IGNORECASE,
)


def _confirm_dispatch_enabled() -> bool:
    """When on (default), a bare affirmation ('ok do that', 'yes please')
    that follows a Face Lobe offer to run deep work actually dispatches
    that work — instead of the model falsely claiming it dispatched.
    Critical for voice, where the operator can't type '/deep'."""
    return os.environ.get("MS4_DA_CONFIRM_DISPATCH", "1").strip().lower() not in {"0", "false", "no"}


def _looks_like_deep_offer(text: str) -> bool:
    return bool(text) and bool(_DEEP_OFFER_RE.search(text))


def _safe_dispatch_error(dispatched_job: dict[str, Any] | None) -> str | None:
    """Return a short, single-line dispatch error fit for prompts/UI."""
    if not isinstance(dispatched_job, dict):
        return None
    raw = dispatched_job.get("error")
    if not raw:
        return None
    text = " ".join(str(raw).split())
    if len(text) > 240:
        text = text[:237].rstrip() + "..."
    return text


def _dispatch_failure_authority_line(dispatched_job: dict[str, Any] | None) -> str | None:
    error = _safe_dispatch_error(dispatched_job)
    if not error:
        return None
    return (
        "THIS TURN TRIED to dispatch Depth Lobe work, but dispatch failed: "
        f"{error}. No Depth Lobe job is running for this turn unless a job id is listed."
    )


def _dispatch_failure_user_text(error: str) -> str:
    return (
        "I tried to hand that to the Depth Lobe, but dispatch failed: "
        f"{error}. No background job is running for this turn."
    )


def _gated_proof_block_user_text() -> str:
    return (
        "I did not dispatch that gated action because its exact current "
        "catalog and approval-policy proof could not be validated. "
        "No background job or mutation was started."
    )


_SERVICE_ACTION_CONFLICT_REASON = (
    "natural service action conflicts with explicit canonical tool"
)


def _quartermaster_service_action_conflict(
    dispatch_intent: str,
    outcome: dict[str, Any] | None,
) -> bool:
    """Authenticate the cascade's unique deterministic conflict marker."""
    if (
        not isinstance(outcome, dict)
        or outcome.get("schema") != "Ms4ToolRouteDecision.v1"
        or outcome.get("verdict") != "none"
        or outcome.get("query") != dispatch_intent
        or outcome.get("reason") != _SERVICE_ACTION_CONFLICT_REASON
        or outcome.get("inline_tool") is not None
    ):
        return False
    resolution = outcome.get("resolution")
    return bool(
        isinstance(resolution, dict)
        and resolution.get("schema") == "Ms4ToolResolution.v1"
        and resolution.get("query") == dispatch_intent
        and resolution.get("tier") == "none"
        and resolution.get("tools") == []
        and resolution.get("fallback_reason") == _SERVICE_ACTION_CONFLICT_REASON
        and isinstance(resolution.get("catalog_version"), str)
        and resolution.get("catalog_version")
    )


def _render_inline_result(result: Any, *, limit: int = 1800) -> str:
    if isinstance(result, str):
        rendered = result
    else:
        try:
            rendered = json.dumps(result, ensure_ascii=False, default=str)
        except Exception:
            rendered = str(result)
    rendered = " ".join(rendered.split())
    if len(rendered) > limit:
        rendered = rendered[: limit - 3].rstrip() + "..."
    return rendered or "(no data)"


def _serialize_authoritative_inline_result(result: Any) -> str:
    """Serialize successful inline output without normalization or truncation."""
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception:
        return str(result)


def _quartermaster_public_tool_trace(
    outcome: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Describe one completed inline call without exposing its payload."""
    if (
        not isinstance(outcome, dict)
        or outcome.get("inline_executed") is not True
        or "inline_result" not in outcome
    ):
        return []
    tool = outcome.get("inline_executed_tool")
    elapsed_ms = outcome.get("inline_elapsed_ms")
    if (
        not isinstance(tool, str)
        or not tool.strip()
        or isinstance(elapsed_ms, bool)
        or not isinstance(elapsed_ms, int)
        or elapsed_ms < 0
    ):
        return []

    result = outcome["inline_result"]
    if result is None:
        result_type = "null"
    elif isinstance(result, bool):
        result_type = "boolean"
    elif isinstance(result, dict):
        result_type = "object"
    elif isinstance(result, (list, tuple)):
        result_type = "array"
    elif isinstance(result, str):
        result_type = "string"
    elif isinstance(result, (int, float)):
        result_type = "number"
    else:
        result_type = "other"
    result_bytes = len(
        _serialize_authoritative_inline_result(result).encode("utf-8")
    )
    return [
        {
            "event": "complete",
            "tool": tool.strip(),
            "args": {},
            "success": True,
            "duration_ms": elapsed_ms,
            "result_type": result_type,
            "result_bytes": result_bytes,
        }
    ]


def _quartermaster_authoritative_text(outcome: dict[str, Any] | None) -> str | None:
    """Build the deterministic model-free reply from executed inline data."""
    if not isinstance(outcome, dict) or outcome.get("inline_executed") is not True:
        return None
    tool = outcome.get("inline_executed_tool") or outcome.get("inline_tool")
    if not isinstance(tool, str) or not tool.strip():
        return None
    result_text = outcome.get("inline_result_text")
    if not isinstance(result_text, str):
        if "inline_result" not in outcome:
            return None
        result_text = _serialize_authoritative_inline_result(outcome["inline_result"])
    return f"Authoritative result from {tool.strip()}: {result_text}"


def _voice_inline_scalar(value: Any) -> str | None:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        rendered = repr(value)
        return rendered if re.fullmatch(r"-?\d+(?:\.\d+)?", rendered) else None
    if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_:/+\-]{1,48}", value):
        return value
    return None


def _quartermaster_voice_authoritative_text(
    message: str,
    outcome: dict[str, Any] | None,
    *,
    limit: int = _VOICE_INLINE_MAX_CHARS,
) -> str | None:
    """Render one bounded voice sentence from executed inline metadata."""
    if not isinstance(outcome, dict) or outcome.get("inline_executed") is not True:
        return None
    tool = outcome.get("inline_executed_tool") or outcome.get("inline_tool")
    if not isinstance(tool, str) or not tool.strip():
        return None

    marker_match = _CORRELATION_MARKER_RE.search(message)
    marker = None
    if marker_match is not None:
        candidate = " ".join(marker_match.group(1).split())
        candidate = re.sub(r"[^A-Za-z0-9_:/,+\- ]", "", candidate).strip(" ,:-")
        marker = candidate or None

    prefix = f"Correlation marker {marker}: " if marker else ""
    stem = f"{prefix}{tool.strip()} reported"
    result = outcome.get("inline_result")
    trusted: list[str] = []
    if isinstance(result, dict):
        for key in _VOICE_INLINE_TRUSTED_SCALARS:
            if key not in result:
                continue
            rendered = _voice_inline_scalar(result[key])
            if rendered is None:
                continue
            candidate = f"{key}={rendered}"
            combined = ", ".join([*trusted, candidate])
            if len(f"{stem} {combined}.") <= limit:
                trusted.append(candidate)

    sentence = f"{stem} {', '.join(trusted)}." if trusted else f"{prefix}{tool.strip()} completed."
    return sentence if len(sentence) <= limit else None


def _quartermaster_authoritative_guard_block(
    outcome: dict[str, Any] | None,
) -> str | None:
    """Ground the full authoritative payload for the existing output guard."""
    if not isinstance(outcome, dict) or outcome.get("inline_executed") is not True:
        return None
    tool = outcome.get("inline_executed_tool") or outcome.get("inline_tool")
    result_text = outcome.get("inline_result_text")
    if not isinstance(tool, str) or not tool.strip() or not isinstance(result_text, str):
        return None
    return (
        "MS4 Quartermaster inline tool result (authoritative; quote this faithfully, "
        "do not invent additional tools, data, or fields):\n"
        f"- tool: {tool.strip()}  (read-only, executed inline)\n"
        f"- result: {result_text}"
    )


def _usable_rfc3339_datetime(value: Any) -> str | None:
    """Return an exact timezone-aware RFC3339/ISO datetime, or ``None``."""
    if (
        not isinstance(value, str)
        or value != value.strip()
        or not 20 <= len(value) <= 64
        or value[10] != "T"
    ):
        return None
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return value


def _quartermaster_inline_cluster_time(
    outcome: dict[str, Any] | None,
) -> str | None:
    """Reuse a usable same-turn result from the exact cluster-time tool."""
    if (
        not isinstance(outcome, dict)
        or outcome.get("inline_executed") is not True
        or outcome.get("inline_executed_tool") != "hivemind.time.now@v1"
        or "inline_result" not in outcome
    ):
        return None
    result = outcome["inline_result"]
    if isinstance(result, str):
        return _usable_rfc3339_datetime(result)
    if isinstance(result, dict):
        for key in ("iso", "iso8601", "now", "datetime", "time", "raw_text"):
            value = result.get(key)
            usable = _usable_rfc3339_datetime(value)
            if usable is not None:
                return usable
    return None


def _quartermaster_result_fallback_text(
    outcome: dict[str, Any] | None,
    context_block: str | None,
) -> str | None:
    """Return authoritative inline output when narration itself fails."""
    sentinel = object()
    result: Any = sentinel
    if isinstance(outcome, dict) and "inline_result" in outcome:
        result = outcome["inline_result"]

    if result is sentinel and context_block:
        for line in context_block.splitlines():
            if line.lstrip().startswith("- result:"):
                result = line.split(":", 1)[1].strip()
                break
    if result is sentinel:
        return None

    return (
        "The requested tool completed successfully, but the Face Lobe timed "
        "out while summarizing it. Here is the result: "
        f"{_render_inline_result(result)}"
    )


def _face_lobe_prior_context(face_lobe_chat: Any, session_id: str | None) -> list[dict[str, str]]:
    if not session_id:
        return []
    sessions = getattr(face_lobe_chat, "_sessions", None)
    if not isinstance(sessions, dict):
        return []
    lock = getattr(face_lobe_chat, "_lock", None)
    if lock is not None:
        try:
            with lock:
                state = sessions.get(session_id)
                messages = list(getattr(state, "messages", []) or []) if state is not None else []
        except Exception:
            return []
    else:
        state = sessions.get(session_id)
        messages = list(getattr(state, "messages", []) or []) if state is not None else []
    return sanitize_prior_context(messages)


_DEPTH_REQUIRED_TOOLSET = "mcp-hivemind"
_DEPTH_DEFAULT_TOOLSETS = ("hermes-cli", _DEPTH_REQUIRED_TOOLSET)
_DEPTH_EXACT_READ_TOOLSET = "mcp-hivemind-exact-read"
_DEPTH_EXACT_GATED_TOOLSET = "mcp-hivemind-exact-gated"
_EXACT_READ_CANONICAL_MAX_CHARS = 160
_EXACT_READ_DIRECTIVE_MAX_CHARS = 640
_EXACT_GATED_DIRECTIVE_MAX_CHARS = 768
_CATALOG_NAME_SEGMENT_CHARS = r"A-Za-z0-9_@-"
_HIVEMIND_CANONICAL_NAME_RE = _re.compile(
    r"\Ahivemind(?:\.[A-Za-z0-9_-]+)+@v[1-9][0-9]*\Z",
    _re.IGNORECASE,
)
_HIVEMIND_CANONICAL_TOKEN_RE = _re.compile(
    rf"(?<![{_CATALOG_NAME_SEGMENT_CHARS}])"
    rf"(?<![{_CATALOG_NAME_SEGMENT_CHARS}]\.)"
    r"(hivemind(?:\.[A-Za-z0-9_-]+)+@v[1-9][0-9]*)"
    rf"(?![{_CATALOG_NAME_SEGMENT_CHARS}])"
    rf"(?!\.[{_CATALOG_NAME_SEGMENT_CHARS}])",
    _re.IGNORECASE,
)
_EXACT_READ_DIRECTIVE_TEMPLATE = (
    "[MS4 TRUSTED EXACT-READ DIRECTIVE]\n"
    "Call `hivemind_exact_read` exactly once with `canonical_tool` set to "
    "`{canonical_tool}`. Never call or substitute any other tool. Build its "
    "`arguments` object only from the original user goal; do not guess missing "
    "values. If the bridge returns structured `missing_required_keys`, report "
    "those keys and ask the user for them."
)
_EXACT_READ_GOAL_SEPARATOR = "\n\n[ORIGINAL USER GOAL]\n"
_EXACT_GATED_DIRECTIVE_TEMPLATE = (
    "[MS4 TRUSTED EXACT-GATED DIRECTIVE]\n"
    "Call `hivemind_exact_gated` exactly once with `canonical_tool` set to "
    "`{canonical_tool}`. Pass a bounded `arguments` object derived only from "
    "the original user goal and pass that goal unchanged as `original_goal`. "
    "Never call or substitute any other tool. Never execute the mutation "
    "directly. HLI Oracle must preserve Human Bridge approval and is the sole "
    "mutation authority."
)


def _strict_exact_read_canonical(
    dispatch_intent: str,
    outcome: dict[str, Any] | None,
    *,
    inline_block: str | None,
) -> str | None:
    """Return the sole trusted canonical read explicitly named by this turn."""
    if (
        not isinstance(dispatch_intent, str)
        or not dispatch_intent
        or inline_block is not None
        or not isinstance(outcome, dict)
        or outcome.get("schema") != "Ms4ToolRouteDecision.v1"
        or outcome.get("verdict") != "depth"
        or outcome.get("query") != dispatch_intent
        or not isinstance(outcome.get("reason"), str)
        or outcome.get("inline_tool") is not None
        or outcome.get("inline_toolbox") is not None
    ):
        return None

    inline_executed = outcome.get("inline_executed")
    if inline_executed is not None and inline_executed is not False:
        return None
    if (
        "inline_result" in outcome
        or "inline_result_text" in outcome
        or outcome.get("inline_executed_tool") is not None
        or outcome.get("inline_elapsed_ms") is not None
    ):
        return None

    resolution = outcome.get("resolution")
    if (
        not isinstance(resolution, dict)
        or resolution.get("schema") != "Ms4ToolResolution.v1"
        or resolution.get("query") != dispatch_intent
        or not isinstance(resolution.get("tier"), str)
        or resolution.get("tier") == "none"
        or not isinstance(resolution.get("catalog_version"), str)
        or not resolution.get("catalog_version")
    ):
        return None
    confidence = resolution.get("confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        return None
    fallback_reason = resolution.get("fallback_reason")
    if fallback_reason is not None and not isinstance(fallback_reason, str):
        return None
    toolboxes = resolution.get("toolboxes")
    if (
        not isinstance(toolboxes, list)
        or not toolboxes
        or any(not isinstance(toolbox, str) or not toolbox for toolbox in toolboxes)
    ):
        return None
    tools = resolution.get("tools")
    if not isinstance(tools, list) or not tools:
        return None

    canonical_by_fold: dict[str, str] = {}
    for tool in tools:
        if not isinstance(tool, dict):
            return None
        canonical = tool.get("name")
        toolbox = tool.get("toolbox")
        cluster = tool.get("cluster")
        score = tool.get("score")
        if (
            not isinstance(canonical, str)
            or not canonical
            or len(canonical) > _EXACT_READ_CANONICAL_MAX_CHARS
            or _HIVEMIND_CANONICAL_NAME_RE.fullmatch(canonical) is None
            or not isinstance(toolbox, str)
            or not toolbox
            or toolbox not in toolboxes
            or not isinstance(cluster, str)
            or not cluster
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not 0.0 <= float(score) <= 1.0
            or tool.get("kind") != "hivemind_native"
            or tool.get("inline_eligible") is not True
        ):
            return None
        source = tool.get("source")
        if source is not None and source != "hivemind":
            return None
        gated_by = tool.get("gated_by")
        if gated_by:
            return None
        canonical_by_fold.setdefault(canonical.casefold(), canonical)
    if len(canonical_by_fold) != 1:
        return None

    canonical_fold, canonical = next(iter(canonical_by_fold.items()))
    explicit_names = {
        match.group(1).casefold()
        for match in _HIVEMIND_CANONICAL_TOKEN_RE.finditer(dispatch_intent)
    }
    if explicit_names != {canonical_fold}:
        return None
    return canonical


def _gate_is_non_admitting(gate: str) -> bool:
    """Manifest gate strings that must NOT admit a tool to the gated bridge.

    'dry' gates describe dry-run behaviour and an explicit 'none: <rationale>'
    gate declares a policy of *no* gate (bounded inference, no world mutation).
    Neither is a confirmation/authority gate, so neither may newly admit the
    tool to the exact-gated bridge.
    """
    folded = gate.casefold().strip()
    return "dry" in folded or folded.startswith("none")


def _catalog_proves_exact_gated(canonical_tool: str, hivemind_url: str) -> bool:
    try:
        from .quartermaster import catalog as catalog_module

        catalog = catalog_module.get_catalog(hivemind_url)
        errors = getattr(catalog, "errors", None)
        if not isinstance(errors, (list, tuple)) or errors:
            return False
        entries = catalog.by_name()
        entry = entries.get(canonical_tool) if isinstance(entries, dict) else None
        if (
            entry is None
            or entry.name != canonical_tool
            or entry.source != catalog_module.SOURCE_HIVEMIND
            or entry.kind != catalog_module.KIND_HIVEMIND_NATIVE
            or catalog_module.is_inline_eligible(entry)
        ):
            return False
        manifest_entries, manifest_errors = catalog_module._load_ms4_manifest()
        if manifest_errors:
            return False
        proxies = [
            raw
            for raw in manifest_entries
            if isinstance(raw, dict)
            and raw.get("name") == f"ms4.{canonical_tool}"
        ]
        if len(proxies) != 1:
            return False
        proxy = proxies[0]
        gated_by = proxy.get("gated_by")
        if isinstance(gated_by, str):
            gates = [gated_by]
        elif isinstance(gated_by, list):
            gates = [gate for gate in gated_by if isinstance(gate, str) and gate]
        else:
            gates = []
        return bool(
            proxy.get("kind") == catalog_module.KIND_RUNTIME_ACTION
            and gates
            and not any(_gate_is_non_admitting(gate) for gate in gates)
        )
    except Exception as exc:
        log.warning("strict exact-gated catalog validation failed: %s", exc)
        return False


def _catalog_proves_exact_read_entry(canonical_tool: str, hivemind_url: str) -> bool:
    """Current catalog proof for an exact-read HiveMind canonical tool."""
    try:
        from .quartermaster import catalog as catalog_module

        catalog = catalog_module.get_catalog(hivemind_url)
        errors = getattr(catalog, "errors", None)
        if not isinstance(errors, (list, tuple)) or errors:
            return False
        entries = catalog.by_name()
        entry = entries.get(canonical_tool) if isinstance(entries, dict) else None
        if (
            entry is None
            or entry.name != canonical_tool
            or entry.source != catalog_module.SOURCE_HIVEMIND
            or entry.kind != catalog_module.KIND_HIVEMIND_NATIVE
            or not catalog_module.is_inline_eligible(entry)
        ):
            return False
        return True
    except Exception as exc:
        log.warning("strict exact-read catalog validation failed: %s", exc)
        return False


def _catalog_proves_inline_execute(canonical_tool: str, hivemind_url: str) -> bool:
    """Current catalog proof for a zero-arg eligible inline read."""
    try:
        from .quartermaster import catalog as catalog_module

        if not _catalog_proves_exact_read_entry(canonical_tool, hivemind_url):
            return False
        catalog = catalog_module.get_catalog(hivemind_url)
        entries = catalog.by_name()
        entry = entries.get(canonical_tool) if isinstance(entries, dict) else None
        if entry is None:
            return False
        schema = entry.input_schema or {}
        required = schema.get("required") if isinstance(schema, dict) else None
        if isinstance(required, list) and any(
            isinstance(item, str) and item for item in required
        ):
            return False
        return True
    except Exception as exc:
        log.warning("inline catalog validation failed: %s", exc)
        return False


_ADMIT_INLINE_READ = "inline_read"
_ADMIT_EXACT_READ = "exact_read"
_ADMIT_EXACT_GATED = "exact_gated"
_ADMIT_NO_TOOL = "no_tool"
_ADMIT_REFUSE = "refuse"


def _oracle_supported_admission(
    dispatch_intent: str,
    outcome: dict[str, Any] | None,
    *,
    inline_block: str | None,
    hivemind_url: str,
    inline_invoked: bool = False,
) -> dict[str, Any]:
    """One fail-closed admission for inline, Depth, and direct seams.

    Never grants full/default catalog authority. ``None`` mapping is not
    a tool grant. An explicit no-tool analytical job is distinct from an
    empty or failed mapping.
    """
    if inline_invoked:
        return {
            "decision": _ADMIT_REFUSE,
            "reason": "inline already invoked",
            "canonical": None,
            "toolsets": None,
        }
    if (
        isinstance(outcome, dict)
        and outcome.get("schema") == "Ms4ToolRouteDecision.v1"
        and outcome.get("verdict") == "inline"
        and outcome.get("query") == dispatch_intent
        and isinstance(outcome.get("inline_tool"), str)
        and outcome.get("inline_tool")
    ):
        canonical = outcome["inline_tool"]
        if _catalog_proves_inline_execute(canonical, hivemind_url):
            return {
                "decision": _ADMIT_INLINE_READ,
                "reason": "catalog-backed inline read",
                "canonical": canonical,
                "toolsets": None,
            }
        return {
            "decision": _ADMIT_REFUSE,
            "reason": "inline catalog or argument proof failed",
            "canonical": None,
            "toolsets": None,
        }

    exact_read = _strict_exact_read_canonical(
        dispatch_intent,
        outcome,
        inline_block=inline_block,
    )
    if exact_read is not None:
        if not _catalog_proves_exact_read_entry(exact_read, hivemind_url):
            return {
                "decision": _ADMIT_REFUSE,
                "reason": "catalog does not admit exact-read canonical",
                "canonical": None,
                "toolsets": None,
            }
        return {
            "decision": _ADMIT_EXACT_READ,
            "reason": "exact hivemind_exact_read",
            "canonical": exact_read,
            "toolsets": [_DEPTH_EXACT_READ_TOOLSET],
        }

    exact_gated = _strict_exact_gated_canonical(
        dispatch_intent,
        outcome,
        inline_block=inline_block,
        hivemind_url=hivemind_url,
    )
    if exact_gated is not None:
        return {
            "decision": _ADMIT_EXACT_GATED,
            "reason": "exact hivemind_exact_gated",
            "canonical": exact_gated,
            "toolsets": [_DEPTH_EXACT_GATED_TOOLSET],
        }

    if is_reasoning_only_analytical(dispatch_intent):
        return {
            "decision": _ADMIT_NO_TOOL,
            "reason": "recognized no-tool analytical job",
            "canonical": None,
            "toolsets": [],
        }
    return {
        "decision": _ADMIT_REFUSE,
        "reason": "unsupported, unknown, empty, or failed tool mapping",
        "canonical": None,
        "toolsets": None,
    }


def _oracle_supported_direct_admission(
    tool_name: str,
    tool_args: dict[str, Any],
    hivemind_url: str,
) -> dict[str, Any]:
    """Admit only an exact Oracle wrapper plus a current canonical argument."""
    if tool_name == "hivemind_exact_read":
        canonical = tool_args.get("canonical_tool")
        if (
            not isinstance(canonical, str)
            or not canonical
            or _HIVEMIND_CANONICAL_NAME_RE.fullmatch(canonical) is None
        ):
            return {
                "decision": _ADMIT_REFUSE,
                "reason": "missing or invalid canonical_tool",
                "canonical": None,
            }
        if not _catalog_proves_exact_read_entry(canonical, hivemind_url):
            return {
                "decision": _ADMIT_REFUSE,
                "reason": "catalog does not admit exact-read canonical",
                "canonical": None,
            }
        return {
            "decision": _ADMIT_EXACT_READ,
            "reason": "direct exact-read wrapper",
            "canonical": canonical,
        }
    if tool_name == "hivemind_exact_gated":
        canonical = tool_args.get("canonical_tool")
        if (
            not isinstance(canonical, str)
            or not canonical
            or _HIVEMIND_CANONICAL_NAME_RE.fullmatch(canonical) is None
        ):
            return {
                "decision": _ADMIT_REFUSE,
                "reason": "missing or invalid canonical_tool",
                "canonical": None,
            }
        if not _catalog_proves_exact_gated(canonical, hivemind_url):
            return {
                "decision": _ADMIT_REFUSE,
                "reason": "catalog or manifest does not admit exact-gated canonical",
                "canonical": None,
            }
        return {
            "decision": _ADMIT_EXACT_GATED,
            "reason": "direct exact-gated wrapper",
            "canonical": canonical,
        }
    return {
        "decision": _ADMIT_REFUSE,
        "reason": f"unsupported tool {tool_name}",
        "canonical": None,
    }


def _admission_refused_user_text(reason: str) -> str:
    return (
        "I did not run that tool request because it is not an admitted "
        f"Oracle-supported route: {reason}. No inline call, Depth job, or "
        "direct tool handler was started."
    )


def _inline_terminal_user_text() -> str:
    return (
        "The admitted inline tool ran once but did not return a usable "
        "result. I did not retry or fall through to Depth or a direct handler."
    )


def _shared_manifest_marks_gated_candidate(canonical_tool: str) -> bool:
    try:
        from .quartermaster import catalog as catalog_module
        from .quartermaster.cascade import _SERVICE_ACTION_CANONICAL

        if canonical_tool in set(_SERVICE_ACTION_CANONICAL.values()):
            return True
        manifest_entries, manifest_errors = catalog_module._load_ms4_manifest()
        if manifest_errors:
            return False
        proxies = [
            raw
            for raw in manifest_entries
            if isinstance(raw, dict)
            and raw.get("name") == f"ms4.{canonical_tool}"
        ]
        if len(proxies) != 1:
            return False
        proxy = proxies[0]
        gated_by = proxy.get("gated_by")
        gates = (
            [gated_by]
            if isinstance(gated_by, str) and gated_by
            else [
                gate
                for gate in gated_by
                if isinstance(gated_by, list)
                and isinstance(gate, str)
                and gate
            ]
            if isinstance(gated_by, list)
            else []
        )
        return bool(
            proxy.get("kind") == catalog_module.KIND_RUNTIME_ACTION
            and gates
            and not any(_gate_is_non_admitting(gate) for gate in gates)
        )
    except Exception:
        return False


def _exact_gated_candidate_canonical(
    dispatch_intent: str,
    outcome: dict[str, Any] | None,
    *,
    inline_block: str | None,
) -> str | None:
    """Identify a narrow gated candidate before live catalog proof."""
    if (
        not isinstance(dispatch_intent, str)
        or not dispatch_intent
        or inline_block is not None
        or not isinstance(outcome, dict)
        or outcome.get("schema") != "Ms4ToolRouteDecision.v1"
        or outcome.get("verdict") != "depth"
        or outcome.get("query") != dispatch_intent
        or not isinstance(outcome.get("reason"), str)
        or outcome.get("inline_tool") is not None
        or outcome.get("inline_toolbox") is not None
        or outcome.get("inline_executed") is True
        or "inline_result" in outcome
        or "inline_result_text" in outcome
    ):
        return None
    resolution = outcome.get("resolution")
    if (
        not isinstance(resolution, dict)
        or resolution.get("schema") != "Ms4ToolResolution.v1"
        or resolution.get("query") != dispatch_intent
        or not isinstance(resolution.get("tools"), list)
        or not resolution["tools"]
    ):
        return None
    canonical_by_fold: dict[str, str] = {}
    for tool in resolution["tools"]:
        if (
            not isinstance(tool, dict)
            or not isinstance(tool.get("name"), str)
            or _HIVEMIND_CANONICAL_NAME_RE.fullmatch(tool["name"]) is None
            or tool.get("kind") != "hivemind_native"
            or tool.get("inline_eligible") is not False
        ):
            return None
        canonical_by_fold.setdefault(tool["name"].casefold(), tool["name"])
    if len(canonical_by_fold) != 1:
        return None
    canonical_fold, canonical = next(iter(canonical_by_fold.items()))
    explicit_names = {
        match.group(1).casefold()
        for match in _HIVEMIND_CANONICAL_TOKEN_RE.finditer(dispatch_intent)
    }
    if explicit_names:
        if explicit_names != {canonical_fold}:
            return None
    else:
        if resolution.get("tier") != "deterministic":
            return None
        try:
            from .quartermaster.cascade import _parse_service_action_command

            command = _parse_service_action_command(dispatch_intent)
        except Exception:
            return None
        if (
            command is None
            or command.conflicts
            or command.canonical_tool.casefold() != canonical_fold
        ):
            return None
    return canonical if _shared_manifest_marks_gated_candidate(canonical) else None


def _strict_exact_gated_canonical(
    dispatch_intent: str,
    outcome: dict[str, Any] | None,
    *,
    inline_block: str | None,
    hivemind_url: str,
) -> str | None:
    """Return one explicit catalog-proven gated HiveMind canonical tool."""
    if (
        not isinstance(dispatch_intent, str)
        or not dispatch_intent
        or inline_block is not None
        or not isinstance(outcome, dict)
        or outcome.get("schema") != "Ms4ToolRouteDecision.v1"
        or outcome.get("verdict") != "depth"
        or outcome.get("query") != dispatch_intent
        or not isinstance(outcome.get("reason"), str)
        or outcome.get("inline_tool") is not None
        or outcome.get("inline_toolbox") is not None
    ):
        return None
    inline_executed = outcome.get("inline_executed")
    if inline_executed is not None and inline_executed is not False:
        return None
    if (
        "inline_result" in outcome
        or "inline_result_text" in outcome
        or outcome.get("inline_executed_tool") is not None
        or outcome.get("inline_elapsed_ms") is not None
    ):
        return None
    resolution = outcome.get("resolution")
    if (
        not isinstance(resolution, dict)
        or resolution.get("schema") != "Ms4ToolResolution.v1"
        or resolution.get("query") != dispatch_intent
        or not isinstance(resolution.get("tier"), str)
        or resolution.get("tier") == "none"
        or not isinstance(resolution.get("catalog_version"), str)
        or not resolution.get("catalog_version")
    ):
        return None
    confidence = resolution.get("confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        return None
    toolboxes = resolution.get("toolboxes")
    tools = resolution.get("tools")
    if (
        not isinstance(toolboxes, list)
        or not toolboxes
        or any(not isinstance(toolbox, str) or not toolbox for toolbox in toolboxes)
        or not isinstance(tools, list)
        or not tools
    ):
        return None
    canonical_by_fold: dict[str, str] = {}
    for tool in tools:
        if not isinstance(tool, dict):
            return None
        canonical = tool.get("name")
        toolbox = tool.get("toolbox")
        score = tool.get("score")
        if (
            not isinstance(canonical, str)
            or not canonical
            or len(canonical) > _EXACT_READ_CANONICAL_MAX_CHARS
            or _HIVEMIND_CANONICAL_NAME_RE.fullmatch(canonical) is None
            or not isinstance(toolbox, str)
            or toolbox not in toolboxes
            or not isinstance(tool.get("cluster"), str)
            or not tool.get("cluster")
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not 0.0 <= float(score) <= 1.0
            or tool.get("kind") != "hivemind_native"
            or tool.get("inline_eligible") is not False
        ):
            return None
        canonical_by_fold.setdefault(canonical.casefold(), canonical)
    if len(canonical_by_fold) != 1:
        return None
    canonical_fold, canonical = next(iter(canonical_by_fold.items()))
    explicit_names = {
        match.group(1).casefold()
        for match in _HIVEMIND_CANONICAL_TOKEN_RE.finditer(dispatch_intent)
    }
    deterministic_arguments = resolution.get("deterministic_arguments")
    if explicit_names:
        if explicit_names != {canonical_fold}:
            return None
        if deterministic_arguments is not None:
            try:
                from .quartermaster.cascade import _parse_service_action_command

                command = _parse_service_action_command(dispatch_intent)
            except Exception:
                return None
            if (
                command is None
                or command.conflicts
                or command.canonical_tool.casefold() != canonical_fold
                or deterministic_arguments
                != {"service_name": command.service_name}
            ):
                return None
    else:
        if resolution.get("tier") != "deterministic":
            return None
        try:
            from .quartermaster.cascade import _parse_service_action_command

            command = _parse_service_action_command(dispatch_intent)
        except Exception:
            return None
        if (
            command is None
            or command.conflicts
            or command.canonical_tool.casefold() != canonical_fold
            or deterministic_arguments
            != {"service_name": command.service_name}
        ):
            return None
    if not _catalog_proves_exact_gated(canonical, hivemind_url):
        return None
    return canonical


def _exact_read_internal_goal(original_goal: str, canonical_tool: str) -> str:
    """Add a static trusted directive while preserving a bounded user goal."""
    if (
        not isinstance(canonical_tool, str)
        or len(canonical_tool) > _EXACT_READ_CANONICAL_MAX_CHARS
        or _HIVEMIND_CANONICAL_NAME_RE.fullmatch(canonical_tool) is None
    ):
        raise ValueError("invalid exact-read canonical tool")
    directive = _EXACT_READ_DIRECTIVE_TEMPLATE.format(canonical_tool=canonical_tool)
    if len(directive) > _EXACT_READ_DIRECTIVE_MAX_CHARS:
        raise ValueError("exact-read directive exceeded its trusted length bound")

    goal = str(original_goal or "")
    goal_budget = INTERNAL_GOAL_MAX_CHARS - len(directive) - len(_EXACT_READ_GOAL_SEPARATOR)
    if len(goal) > goal_budget:
        goal = goal[: goal_budget - 1] + "…"
    return f"{directive}{_EXACT_READ_GOAL_SEPARATOR}{goal}"


def _exact_gated_internal_goal(
    original_goal: str,
    canonical_tool: str,
    deterministic_arguments: dict[str, Any] | None = None,
) -> str:
    if (
        not isinstance(canonical_tool, str)
        or len(canonical_tool) > _EXACT_READ_CANONICAL_MAX_CHARS
        or _HIVEMIND_CANONICAL_NAME_RE.fullmatch(canonical_tool) is None
    ):
        raise ValueError("invalid exact-gated canonical tool")
    if deterministic_arguments is None:
        directive = _EXACT_GATED_DIRECTIVE_TEMPLATE.format(
            canonical_tool=canonical_tool
        )
    else:
        arguments_json = json.dumps(
            deterministic_arguments,
            ensure_ascii=True,
            sort_keys=True,
        )
        directive = (
            "[MS4 TRUSTED EXACT-GATED DIRECTIVE]\n"
            "Call `hivemind_exact_gated` exactly once with `canonical_tool` "
            f"set to `{canonical_tool}`. Validated deterministic arguments: "
            f"{arguments_json}. Pass exactly these `arguments` and pass the "
            "original user goal unchanged as `original_goal`. Do not "
            "rediscover or alter these arguments. Never call or substitute "
            "any other tool. Never execute the mutation directly. HLI Oracle "
            "must preserve Human Bridge approval and is the sole mutation authority."
        )
    if len(directive) > _EXACT_GATED_DIRECTIVE_MAX_CHARS:
        raise ValueError("exact-gated directive exceeded its trusted length bound")
    goal = str(original_goal or "")
    goal_budget = INTERNAL_GOAL_MAX_CHARS - len(directive) - len(
        _EXACT_READ_GOAL_SEPARATOR
    )
    if len(goal) > goal_budget:
        goal = goal[: goal_budget - 1] + "…"
    return f"{directive}{_EXACT_READ_GOAL_SEPARATOR}{goal}"


_EXACT_READ_CONTINUATION_MESSAGE_MAX_CHARS = 160
_EXACT_READ_CONTINUATION_KEY_MAX_CHARS = 64
_EXACT_READ_CONTINUATION_VALUE_MAX_CHARS = 128
_EXACT_READ_CONTINUATION_GOAL_MAX_CHARS = 512
_EXACT_READ_CONTINUATION_JOB_LIMIT = 20
_EXACT_READ_CONTINUATION_EVENT_SOURCE = "hermes_runner_exact_read_continuation"
_EXACT_READ_CONTINUATION_KEY_RE = _re.compile(
    r"\A[A-Za-z_][A-Za-z0-9_]{0,63}\Z",
    _re.ASCII,
)
_EXACT_READ_CONTINUATION_VALUE_PATTERN = (
    r"[A-Za-z0-9]"
    r"(?:[A-Za-z0-9_:@/+~.\-]{0,126}[A-Za-z0-9_:@/+~\-])?"
)
_EXACT_READ_CONTINUATION_BARE_UUID_RE = _re.compile(
    r"\A[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
    r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}\Z",
    _re.ASCII,
)
_EXACT_READ_CONTINUATION_LINK_STATUS = (
    "Exact-read continuation source linkage recorded."
)


def _could_be_exact_read_continuation_reply(message: str) -> bool:
    if not isinstance(message, str):
        return False
    text = message.strip(" ")
    if (
        not text
        or len(text) > _EXACT_READ_CONTINUATION_MESSAGE_MAX_CHARS
        or any(ord(char) < 32 or ord(char) == 127 for char in text)
        or _HIVEMIND_CANONICAL_TOKEN_RE.search(text) is not None
    ):
        return False
    key = r"[A-Za-z_][A-Za-z0-9_]{0,63}"
    value = _EXACT_READ_CONTINUATION_VALUE_PATTERN
    return any(
        _re.fullmatch(pattern, text, _re.IGNORECASE | _re.ASCII) is not None
        for pattern in (
            rf"(?:the[ ]+)?{key}[ ]+is[ ]+{value}[.!?]?",
            rf"{key}[ ]*(?:=|:)[ ]*{value}[.!?]?",
            rf"{value}[.!?]?",
        )
    )


def _parse_exact_read_continuation_value(
    message: str,
    missing_key: str,
) -> str | None:
    if (
        not _could_be_exact_read_continuation_reply(message)
        or not isinstance(missing_key, str)
        or len(missing_key) > _EXACT_READ_CONTINUATION_KEY_MAX_CHARS
        or _EXACT_READ_CONTINUATION_KEY_RE.fullmatch(missing_key) is None
    ):
        return None
    text = message.strip(" ")
    escaped_key = _re.escape(missing_key)
    value_pattern = _EXACT_READ_CONTINUATION_VALUE_PATTERN
    named_patterns = (
        rf"(?:the[ ]+)?{escaped_key}[ ]+is[ ]+"
        rf"(?P<value>{value_pattern})[.!]?",
        rf"{escaped_key}[ ]*(?:=|:)[ ]*"
        rf"(?P<value>{value_pattern})[.!]?",
    )
    for pattern in named_patterns:
        match = _re.fullmatch(pattern, text, _re.IGNORECASE | _re.ASCII)
        if match is not None:
            return match.group("value")

    bare_match = _re.fullmatch(
        rf"(?P<value>{value_pattern})[.!]?",
        text,
        _re.ASCII,
    )
    if bare_match is None:
        return None
    value = bare_match.group("value")
    bare_has_identity_shape = (
        _EXACT_READ_CONTINUATION_BARE_UUID_RE.fullmatch(value) is not None
        or value.casefold() in {"true", "false"}
        or any(char.isdigit() for char in value)
        or any(char in "-_.:@/+~" for char in value)
    )
    return value if bare_has_identity_shape else None


def _persisted_exact_read_missing_source(
    job: dict[str, Any],
    result: dict[str, Any] | None,
) -> tuple[str, str] | None:
    if (
        not isinstance(job, dict)
        or job.get("schema") != "DoubleAgentJobEnvelope.v1"
        or job.get("state") != "completed"
        or not isinstance(job.get("job_id"), str)
        or not job.get("job_id")
        or not isinstance(job.get("finished_at"), str)
        or not job.get("finished_at")
    ):
        return None
    resource_request = job.get("resource_request")
    if (
        not isinstance(resource_request, dict)
        or resource_request.get("enabled_toolsets")
        != [_DEPTH_EXACT_READ_TOOLSET]
    ):
        return None
    if (
        not isinstance(result, dict)
        or result.get("schema") != "DoubleAgentJobResult.v1"
        or result.get("job_id") != job.get("job_id")
        or result.get("status") != "success"
        or result.get("conversation_revision_id")
        != job.get("conversation_revision_id")
    ):
        return None
    evidence = result.get("evidence")
    actions = result.get("actions_taken")
    if (
        not isinstance(evidence, list)
        or len(evidence) != 1
        or not isinstance(actions, list)
        or len(actions) != 1
    ):
        return None
    item = evidence[0]
    action = actions[0]
    if not isinstance(item, dict) or not isinstance(action, dict):
        return None
    tool_call_id = item.get("tool_call_id")
    if (
        item.get("kind") != "verified_tool_result"
        or item.get("tool") != "hivemind_exact_read"
        or not isinstance(tool_call_id, str)
        or not tool_call_id
        or item.get("evidence_ref") != f"tool_call:{tool_call_id}"
        or action
        != {"tool": "hivemind_exact_read", "tool_call_id": tool_call_id}
    ):
        return None
    result_excerpt = item.get("result_excerpt")
    if not isinstance(result_excerpt, str) or not result_excerpt:
        return None
    try:
        bridge_result = json.loads(result_excerpt)
    except (TypeError, ValueError):
        return None
    if (
        not isinstance(bridge_result, dict)
        or set(bridge_result) != {
            "status",
            "success",
            "canonical_tool",
            "missing_required_keys",
        }
        or bridge_result.get("status") != "missing_required_arguments"
        or bridge_result.get("success") is not False
    ):
        return None
    canonical_tool = bridge_result.get("canonical_tool")
    missing_keys = bridge_result.get("missing_required_keys")
    if (
        not isinstance(canonical_tool, str)
        or len(canonical_tool) > _EXACT_READ_CANONICAL_MAX_CHARS
        or _HIVEMIND_CANONICAL_NAME_RE.fullmatch(canonical_tool) is None
        or not isinstance(missing_keys, list)
        or len(missing_keys) != 1
    ):
        return None
    missing_key = missing_keys[0]
    if (
        not isinstance(missing_key, str)
        or len(missing_key) > _EXACT_READ_CONTINUATION_KEY_MAX_CHARS
        or _EXACT_READ_CONTINUATION_KEY_RE.fullmatch(missing_key) is None
    ):
        return None
    internal_goal = job.get("internal_goal")
    if not isinstance(internal_goal, str):
        return None
    source_canonicals = {
        match.group(1).casefold()
        for match in _HIVEMIND_CANONICAL_TOKEN_RE.finditer(internal_goal)
    }
    if source_canonicals != {canonical_tool.casefold()}:
        return None
    return canonical_tool, missing_key


def _exact_read_source_was_consumed(blackboard: Any, source_job_id: str) -> bool:
    list_events = getattr(blackboard, "list_events", None)
    if not callable(list_events):
        return True
    try:
        events = list_events(source_job_id, limit=1000)
    except Exception:
        return True
    if not isinstance(events, list):
        return True
    for event in events:
        if not isinstance(event, dict):
            continue
        payload = event.get("payload")
        if (
            event.get("type") == "job.checkpoint"
            and isinstance(payload, dict)
            and payload.get("source") == _EXACT_READ_CONTINUATION_EVENT_SOURCE
            and payload.get("source_job_id") == source_job_id
            and isinstance(payload.get("continuation_job_id"), str)
            and payload.get("continuation_job_id")
        ):
            return True
    return False


def _current_catalog_allows_exact_read(
    canonical_tool: str,
    *,
    hivemind_url: str,
) -> bool:
    try:
        from .quartermaster import catalog as catalog_module

        catalog = catalog_module.get_catalog(hivemind_url)
        catalog_errors = getattr(catalog, "errors", None)
        if not isinstance(catalog_errors, (list, tuple)) or catalog_errors:
            return False
        entries = catalog.by_name()
        entry = entries.get(canonical_tool) if isinstance(entries, dict) else None
        return bool(
            entry is not None
            and entry.name == canonical_tool
            and entry.source == catalog_module.SOURCE_HIVEMIND
            and entry.kind == catalog_module.KIND_HIVEMIND_NATIVE
            and catalog_module.is_inline_eligible(entry)
        )
    except Exception as exc:
        log.warning("exact-read continuation catalog validation failed: %s", exc)
        return False


def _build_exact_read_continuation_intent(
    canonical_tool: str,
    missing_key: str,
    value: str,
) -> str | None:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > _EXACT_READ_CONTINUATION_VALUE_MAX_CHARS
    ):
        return None
    intent = (
        f"Use {canonical_tool} with {missing_key}={value} "
        "to complete the prior exact read."
    )
    return (
        intent
        if len(intent) <= _EXACT_READ_CONTINUATION_GOAL_MAX_CHARS
        else None
    )


def _persisted_exact_read_continuation(
    runner: Any,
    *,
    conversation_id: str,
    message: str,
    hivemind_url: str,
) -> dict[str, str] | None:
    if not _could_be_exact_read_continuation_reply(message):
        return None
    list_jobs = getattr(runner, "list", None)
    get_result = getattr(runner, "get_result", None)
    blackboard = getattr(runner, "blackboard", None)
    if not callable(list_jobs) or not callable(get_result) or blackboard is None:
        return None
    try:
        jobs = list_jobs(
            conversation_id=conversation_id,
            states=("completed",),
            limit=_EXACT_READ_CONTINUATION_JOB_LIMIT,
        )
    except Exception:
        return None
    if not isinstance(jobs, list):
        return None
    for job in jobs:
        if not isinstance(job, dict):
            continue
        source_job_id = job.get("job_id")
        try:
            result = get_result(source_job_id)
        except Exception:
            continue
        source = _persisted_exact_read_missing_source(job, result)
        if source is None:
            continue
        canonical_tool, missing_key = source
        if _exact_read_source_was_consumed(blackboard, source_job_id):
            return None
        if not _current_catalog_allows_exact_read(
            canonical_tool,
            hivemind_url=hivemind_url,
        ):
            return None
        value = _parse_exact_read_continuation_value(message, missing_key)
        if value is None:
            return None
        dispatch_intent = _build_exact_read_continuation_intent(
            canonical_tool,
            missing_key,
            value,
        )
        if dispatch_intent is None:
            return None
        return {
            "source_job_id": source_job_id,
            "canonical_tool": canonical_tool,
            "missing_key": missing_key,
            "value": value,
            "dispatch_intent": dispatch_intent,
        }
    return None


def _persist_exact_read_continuation_link(
    runner: Any,
    continuation: dict[str, str],
    *,
    continuation_job_id: str,
) -> None:
    blackboard = getattr(runner, "blackboard", None)
    insert_event = getattr(blackboard, "insert_event", None)
    if not callable(insert_event):
        raise SchemaError("exact-read continuation linkage store unavailable")
    source_job_id = continuation["source_job_id"]
    insert_event(
        JobEvent.make(
            job_id=source_job_id,
            type="job.checkpoint",
            safe_user_status=_EXACT_READ_CONTINUATION_LINK_STATUS,
            visibility="operator_only",
            payload={
                "source": _EXACT_READ_CONTINUATION_EVENT_SOURCE,
                "source_job_id": source_job_id,
                "continuation_job_id": continuation_job_id,
                "missing_key": continuation["missing_key"],
            },
        )
    )


def _curated_depth_toolsets(query: str) -> list[str] | None:
    """Return an admitted no-tool catalog or refuse.

    Only a recognized pure analytical job may use explicit ``[]``.
    Unknown, empty, or failed mapping is ``None`` — never full/default
    catalog, ``hermes-cli``, or raw ``mcp-hivemind``.
    """
    if is_reasoning_only_analytical(query):
        return []
    return None


# Strong back-references to a just-offered action. Their presence (when
# a deep action is already PENDING) is a confident confirmation signal.
_BACKREF_TOKENS = (
    "do that", "do it", "do this", "go ahead", "go for it", "that for me",
    "deep lobe", "yes please", "please do", "run it", "go on it", "do so",
)
# Negations that flip a would-be confirmation into a decline.
_DECLINE_TOKENS = (
    "don't", "do not", "never mind", "nevermind", "stop", "cancel",
    "not now", "no thanks", "forget it", "hold off", "wait",
)


def _is_confirmation(message: str) -> bool:
    """Lenient confirmation detector, used ONLY when a deep action is
    already PENDING (offered last turn). Catches both ultra-short
    continuations ('ok', 'sure') and natural affirmations that
    back-reference the offer ('okay can you do that for me please', 'I
    need you to do that for me', 'yeah run it'). A topic change ('okay
    what's the weather') has no back-reference and does NOT fire; an
    explicit decline ('no, don't') is rejected."""
    if not message:
        return False
    text = message.strip().lower().rstrip(".!?, \t\n")
    if not text or len(text) > 160:
        return False
    if any(neg in text for neg in _DECLINE_TOKENS):
        return False
    if phrase_is_continuation(message):
        return True
    if any(tok in text for tok in _BACKREF_TOKENS):
        return True
    return False


@dataclass
class SessionState:
    session_id: str
    agent: Any
    history: list[dict[str, Any]] = field(default_factory=list)
    last_grounding_source: str | None = None
    tool_trace: list[dict[str, Any]] = field(default_factory=list)


class HermesUnavailable(RuntimeError):
    pass


class Ms4HermesRunner:
    def __init__(
        self,
        *,
        hermes_dir: str,
        hivemind_url: str = "http://127.0.0.1:6089",
        ms3_url: str = "http://127.0.0.1:9080",
        default_model: str | None = None,
        agent_cls: Any | None = None,
        face_lobe_chat: FaceLobeChat | None = None,
    ) -> None:
        self.hermes_dir = Path(hermes_dir)
        self.hivemind_url = hivemind_url.rstrip("/")
        self.ms3_url = ms3_url.rstrip("/")
        # Cache the cluster-time probe so we don't pay an MCP round
        # trip on every chat turn. The probe is best-effort + capped
        # at ~2s; we cache successful results for 30s and negative
        # results for 5s so a transiently-offline cluster doesn't
        # force the slow path on every turn.
        self._cluster_time_cache: tuple[float, str | None] = (0.0, None)
        self._cluster_time_lock = threading.Lock()
        self.default_model = default_model or depth_fallback_model()
        self._agent_cls = agent_cls
        self._sessions: dict[str, SessionState] = {}
        # Face Lobe direct-chat path: per artifact §5.1/§16.5 the foreground
        # is "status and routing focused" and should NOT pay the Hermes
        # tool-loop tax on every turn. We keep the Hermes plumbing for
        # `dispatch_hermes_tool` and Depth Lobe workers.
        self.face_lobe_chat = face_lobe_chat or FaceLobeChat(hivemind_url=self.hivemind_url)
        # Persistent Face model ownership belongs to the runner. The chat
        # session's model field records the most recently used model, including
        # one-turn overrides, so it cannot also be the durable session pin.
        self._face_model_pins: dict[str, str] = {}
        self._face_model_pins_lock = threading.Lock()
        # Per-conversation memory of a deep action the Face Lobe OFFERED
        # but didn't dispatch, keyed by session id. If the next user turn
        # is a bare affirmation we dispatch this goal (confirmation-
        # triggered dispatch). See `_confirm_dispatch_enabled`.
        self._pending_deep_goal: dict[str, str] = {}
        self._pending_deep_lock = threading.Lock()

    def deliver_depth_result(self, job_id: str, conversation_id: str) -> dict[str, Any]:
        """Verify and bind one persisted Depth result to its exact Face session."""
        job_id = str(job_id or "").strip()
        conversation_id = str(conversation_id or "").strip()
        if not job_id:
            raise ValueError("job_id is required")
        if not conversation_id:
            raise ValueError("conversation_id is required")
        depth_runner = default_runner()
        job = depth_runner.get(job_id)
        if not isinstance(job, dict):
            raise KeyError(job_id)
        job_conversation_id = str(job.get("parent_conversation_id") or "")
        if job_conversation_id != conversation_id:
            raise ValueError("job does not belong to the requested conversation")
        state = str(job.get("state") or "")
        if state not in {"completed", "failed", "canceled"}:
            raise RuntimeError(f"Depth job is not terminal: {state or 'unknown'}")
        result = depth_runner.get_result(job_id)
        base: dict[str, Any] = {
            "schema": "Ms4DepthCompletionDelivery.v1",
            "job_id": job_id,
            "conversation_id": conversation_id,
            "job_state": state,
            "history_bound": False,
            "already_bound": False,
            "history_truncated": False,
            "result": result,
        }
        if state != "completed":
            return {**base, "delivery_kind": "failure"}
        if not isinstance(result, dict):
            return {**base, "delivery_kind": "incomplete", "reason": "missing_result"}
        result_text = str(result.get("text") or "").strip()
        valid_result = (
            result.get("schema") == "DoubleAgentJobResult.v1"
            and result.get("job_id") == job_id
            and result.get("status") == "success"
            and result.get("conversation_revision_id")
            == job.get("conversation_revision_id")
            and bool(result_text)
        )
        if not valid_result:
            return {
                **base,
                "delivery_kind": "incomplete",
                "reason": "unverified_or_incomplete_result",
            }
        with self._face_model_pins_lock:
            face_model = self._face_model_pins.get(conversation_id)
        binding = self.face_lobe_chat.bind_depth_result(
            conversation_id=conversation_id,
            job_id=job_id,
            result_text=result_text,
            goal=str(job.get("user_visible_goal") or ""),
            model=face_model,
            sort_key=str(job.get("finished_at") or job.get("created_at") or ""),
        )
        append_event(
            "depth_result_delivered",
            {
                "job_id": job_id,
                "conversation_id": conversation_id,
                "history_bound": binding["history_bound"],
                "already_bound": binding["already_bound"],
                "history_truncated": binding["history_truncated"],
                "result_chars": len(result_text),
            },
        )
        return {**base, **binding, "delivery_kind": "answer"}

    def _reconcile_depth_results(self, conversation_id: str) -> list[str]:
        """Fail-soft recovery for clients that missed the explicit delivery call."""
        try:
            jobs = default_runner().list(
                conversation_id=conversation_id,
                states=("completed",),
                limit=16,
            )
        except Exception as exc:
            log.warning("Depth delivery reconciliation list failed: %s", exc)
            return []
        reconciled: list[str] = []
        ordered_jobs = sorted(
            (job for job in jobs if isinstance(job, dict)),
            key=lambda job: str(job.get("finished_at") or job.get("created_at") or ""),
        )
        for job in ordered_jobs:
            job_id = str(job.get("job_id") or "")
            if not job_id:
                continue
            try:
                delivery = self.deliver_depth_result(job_id, conversation_id)
            except Exception as exc:
                log.warning("Depth delivery reconciliation skipped %s: %s", job_id, exc)
                continue
            if delivery.get("delivery_kind") == "answer":
                reconciled.append(job_id)
        return reconciled

    def ensure_hermes_path(self) -> None:
        hermes_path = str(self.hermes_dir)
        if hermes_path not in sys.path:
            sys.path.insert(0, hermes_path)

    def plugin_status(self) -> dict[str, Any]:
        self.ensure_hermes_path()
        try:
            from hermes_cli.plugins import PluginManager

            manager = PluginManager()
            manager.discover_and_load()
            loaded = manager._plugins.get("ms4_consciousness")
            hooks = sorted(manager._hooks.keys())
            return {
                "found": loaded is not None,
                "enabled": bool(loaded and loaded.enabled),
                "hooks": hooks,
            }
        except Exception as exc:
            return {"found": False, "enabled": False, "hooks": [], "error": str(exc)}

    def require_plugin(self) -> None:
        status = self.plugin_status()
        if not status.get("enabled"):
            raise HermesUnavailable(f"ms4_consciousness plugin is not enabled: {status}")

    def _agent_class(self):
        if self._agent_cls is not None:
            return self._agent_cls
        self.ensure_hermes_path()
        from run_agent import AIAgent

        return AIAgent

    def _construct_agent(
        self,
        *,
        session_id: str,
        model: str,
        tool_start_callback: Any,
        tool_complete_callback: Any,
        max_iterations: int,
        enabled_toolsets: list[str] | None = None,
        max_tokens: int | None = None,
        reasoning_config: dict[str, Any] | None = None,
    ):
        """Single place that constructs a Hermes ``AIAgent``.

        Both the foreground session path (:meth:`_new_agent`) and the
        Double Agent background worker path
        (:meth:`new_background_agent`) funnel through here so the agent
        construction contract (base_url, auth, platform tagging,
        memory/context skipping) lives in exactly one location.

        ``enabled_toolsets`` (Phase E) optionally restricts the Hermes
        tool catalog the agent loads to a subset of toolsets, trimming
        per-turn tool context. ``None`` = the full catalog (the escape
        hatch / default), matching Hermes' own default.
        """
        agent_cls = self._agent_class()
        kwargs: dict[str, Any] = dict(
            base_url=f"{self.hivemind_url}/v1",
            api_key=os.environ.get("MS4_HIVEMIND_API_KEY", "local-not-needed"),
            provider="custom",
            api_mode="chat_completions",
            model=model,
            session_id=session_id,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            platform="ms4",
            max_iterations=max_iterations,
            tool_start_callback=tool_start_callback,
            tool_complete_callback=tool_complete_callback,
        )
        # Depth workers use their ``da-<uuid>`` session id as the HLI trace.
        # This gives the parent runner a stable identifier it can cancel when
        # the subprocess is stopped, instead of leaving an anonymous HLI job
        # in /jobs/active until the upstream request times out.
        # The worker deliberately namespaces its Hermes session as
        # ``da-worker-da-<uuid>``. Accept both that runtime form and the
        # direct ``da-<uuid>`` form used by focused callers/tests, while still
        # requiring the suffix to parse as an exact UUID.
        trace_id = ""
        if "da-" in session_id:
            try:
                trace_id = str(uuid.UUID(session_id.rsplit("da-", 1)[1]))
            except (ValueError, AttributeError):
                trace_id = ""
            if trace_id:
                kwargs["request_overrides"] = {
                    "extra_headers": {"X-HiveMind-Client-Trace": trace_id},
                }
        if enabled_toolsets is not None:
            kwargs["enabled_toolsets"] = list(enabled_toolsets)
            if not enabled_toolsets:
                # Hermes normally treats [] as an empty catalog, but a worker
                # process carrying HERMES_KANBAN_TASK adds the kanban toolset
                # back in. Subtract that exception explicitly so a no-tools
                # authority remains authoritative under inherited env state.
                kwargs["disabled_toolsets"] = ["kanban"]
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if reasoning_config is not None:
            kwargs["reasoning_config"] = dict(reasoning_config)
        agent = agent_cls(**kwargs)
        if trace_id:
            # HLI already owns endpoint retries, peer fallback, and cold-load
            # recovery. Letting Hermes retry the entire API call multiplies
            # expensive work and can leave several cluster requests running
            # after the parent depth job is canceled. Hermes defines this
            # value as total attempts (1 == no retry).
            try:
                api_attempts = int(os.environ.get("MS4_DA_API_MAX_ATTEMPTS", "1"))
            except ValueError:
                api_attempts = 1
            agent._api_max_retries = max(1, api_attempts)
        return agent

    def _new_agent(self, session_id: str, model: str):
        state_ref = self._sessions.get(session_id)

        def on_tool_start(tool_call_id: str, name: str, args: dict[str, Any]) -> None:
            state = self._sessions.get(session_id) or state_ref
            if state is not None:
                state.tool_trace.append({"event": "start", "tool_call_id": tool_call_id, "tool": name, "args": args})

        def on_tool_complete(tool_call_id: str, name: str, args: dict[str, Any], result: str) -> None:
            state = self._sessions.get(session_id) or state_ref
            if state is not None:
                state.tool_trace.append({
                    "event": "complete",
                    "tool_call_id": tool_call_id,
                    "tool": name,
                    "args": args,
                    "result_excerpt": str(result)[:2000],
                })

        return self._construct_agent(
            session_id=session_id,
            model=model,
            tool_start_callback=on_tool_start,
            tool_complete_callback=on_tool_complete,
            max_iterations=foreground_max_iterations(),
        )

    def new_background_agent(
        self,
        *,
        session_id: str,
        model: str,
        tool_start_callback: Any,
        tool_complete_callback: Any,
        max_iterations: int | None = None,
        enabled_toolsets: list[str] | None = None,
        reasoning_config_override: dict[str, Any] | None = None,
    ):
        """Sanctioned entry point for Double Agent Depth Lobe workers.

        Builds a fresh, isolated ``AIAgent`` whose lifecycle callbacks
        the caller owns end-to-end. This replaces the previous
        ``runner._new_agent.__self__._agent_class()`` reach-through:
        the worker no longer needs to know how the runner stores or
        imports the agent class. Ensures the Hermes path is on
        ``sys.path`` and the ms4_consciousness plugin is enabled before
        constructing (so tool authority still routes through MS3 ethics).

        ``enabled_toolsets`` (Phase E) optionally trims the worker's
        Hermes tool catalog to a subset; ``None`` keeps the full
        catalog (escape hatch). ``reasoning_config_override`` lets a
        bounded corrective pass explicitly disable hidden reasoning while
        retaining the normal nontruncating Depth output-token budget.
        """
        self.ensure_hermes_path()
        self.require_plugin()
        return self._construct_agent(
            session_id=session_id,
            model=model or self.default_model,
            tool_start_callback=tool_start_callback,
            tool_complete_callback=tool_complete_callback,
            max_iterations=max_iterations if max_iterations is not None else background_max_iterations(),
            enabled_toolsets=enabled_toolsets,
            max_tokens=background_max_tokens(),
            reasoning_config=(
                background_reasoning_config()
                if reasoning_config_override is None
                else dict(reasoning_config_override)
            ),
        )

    def get_or_create_session(self, session_id: str | None, model: str) -> SessionState:
        sid = session_id or f"ms4-{uuid.uuid4()}"
        state = self._sessions.get(sid)
        if state is None:
            state = SessionState(session_id=sid, agent=self._new_agent(sid, model))
            self._sessions[sid] = state
        elif getattr(state.agent, "model", model) != model:
            state.agent = self._new_agent(sid, model)
            state.history = []
            state.tool_trace = []
        return state

    def _try_quartermaster_inline(self, message: str) -> tuple[str | None, dict[str, Any] | None]:
        """Quartermaster inline fast-path (Phase D).

        On each foreground turn, ask the Quartermaster whether the request
        resolves to a single safe, read-only, zero-arg, ethics-cleared
        tool. Admission is fail-closed against the current catalog before
        any execute. A None, exception, or malformed result after that
        one invoke is terminal — no Depth or direct fallthrough.
        """
        if os.environ.get("MS4_QM_INLINE", "1").strip().lower() in {"0", "false", "no"}:
            return None, None
        try:
            from .quartermaster import (
                VERDICT_INLINE,
                decide as qm_decide,
                execute_inline_tool,
                format_inline_block,
            )

            decision = qm_decide(message, hivemind_url=self.hivemind_url)
            outcome = decision.to_dict()
            if decision.verdict != VERDICT_INLINE or decision.inline_tool is None:
                return None, outcome

            admission = _oracle_supported_admission(
                message,
                outcome,
                inline_block=None,
                hivemind_url=self.hivemind_url,
            )
            if admission["decision"] != _ADMIT_INLINE_READ:
                outcome["admission_refused"] = True
                outcome["admission_reason"] = admission["reason"]
                return None, outcome

            outcome["inline_invoked"] = True
            try:
                executed = execute_inline_tool(self.hivemind_url, decision.inline_tool.name)
            except Exception as exc:  # noqa: BLE001 — one invoke is terminal
                log.warning("quartermaster inline invoke failed (terminal): %s", exc)
                outcome["inline_executed"] = False
                outcome["inline_terminal_failure"] = True
                outcome["inline_error"] = str(exc)
                return None, outcome
            if not executed:
                outcome["inline_executed"] = False
                outcome["inline_terminal_failure"] = True
                return None, outcome

            outcome["inline_executed"] = True
            outcome["inline_elapsed_ms"] = executed.get("elapsed_ms")
            executed_tool = executed.get("tool")
            outcome["inline_executed_tool"] = (
                executed_tool
                if isinstance(executed_tool, str) and executed_tool.strip()
                else decision.inline_tool.name
            )
            # Keep the authoritative payload on the response so chat() can
            # emit it directly without a second model interpretation.
            outcome["inline_result"] = executed.get("result")
            outcome["inline_result_text"] = _serialize_authoritative_inline_result(
                executed.get("result")
            )
            append_event("quartermaster_inline_exec", {
                "query": message[:240],
                "tool": decision.inline_tool.name,
                "toolbox": decision.inline_tool.toolbox,
                "tier": decision.resolution.tier,
                "elapsed_ms": executed.get("elapsed_ms"),
            })
            return format_inline_block(executed, query=message), outcome
        except Exception as exc:  # noqa: BLE001 — decide/mapping failure is not an invoke
            log.warning("quartermaster inline path failed (no invoke): %s", exc)
            return None, {"error": str(exc)}

    def chat(
        self,
        message: str,
        *,
        session_id: str | None = None,
        model: str | None = None,
        depth_model: str | None = None,
        voice_mode: bool = False,
        stream_callback: Any | None = None,
        cancel_event: threading.Event | None = None,
        client_id: str | None = None,
        recommend_receipt: Any | None = None,
    ) -> dict[str, Any]:
        """Foreground (Face Lobe) chat turn.

        Routes ALL foreground turns through the direct FaceLobeChat path
        (no Hermes loop, no plugin chain, no tool injection) so even
        trivial turns return in seconds instead of tens of seconds.
        Tool-requiring work is dispatched in parallel to a Depth Lobe
        background job by the auto-router; the foreground reply mentions
        the dispatch but doesn't wait on it.
        """
        os.environ["MS4_MS3_SIDECAR_URL"] = self.ms3_url
        os.environ.setdefault("MS4_SPIRIT_ID", "sister")
        # Establish one conversation id before routing, dispatch, or model
        # work. This keeps first-turn failures and background jobs attached
        # to the same session the client receives in the terminal response.
        session_id = session_id or f"ms4-{uuid.uuid4()}"

        def _turn_cancelled() -> bool:
            return cancel_event is not None and cancel_event.is_set()

        # Finding 3: the cancellation fence guards EVERY conversation side effect,
        # not only the final model call. A turn cancelled before this point must
        # not route, bump the revision, dispatch a Depth job, or touch history.
        if _turn_cancelled():
            raise Ms4TurnCancelled("foreground turn cancelled before routing")

        # ---- Face Lobe model resolution (precedence: per-turn > session-pinned > picker > default)
        face_lobe_model: dict[str, Any]
        if model is not None:
            selected_model = model
            face_lobe_model = {
                "schema": "Ms4ForegroundModel.v1",
                "model_id": selected_model,
                "source": "per_turn_override",
                "detail": "request specified model",
            }
        else:
            with self._face_model_pins_lock:
                selected_model = self._face_model_pins.get(session_id)
            if selected_model is not None:
                face_lobe_model = {
                    "schema": "Ms4ForegroundModel.v1",
                    "model_id": selected_model,
                    "source": "session_pinned",
                    "detail": "model pinned on first automatic selection; one-turn overrides do not replace it",
                }
            else:
                choice = choose_foreground_model(hivemind_url=self.hivemind_url)
                with self._face_model_pins_lock:
                    selected_model = self._face_model_pins.setdefault(
                        session_id,
                        choice.model_id,
                    )
                if selected_model == choice.model_id:
                    face_lobe_model = choice.to_dict()
                else:
                    face_lobe_model = {
                        "schema": "Ms4ForegroundModel.v1",
                        "model_id": selected_model,
                        "source": "session_pinned",
                        "detail": "concurrent first turn reused the winning session pin",
                    }

        reconciled_depth_job_ids: list[str] = []

        # A narrowly shaped follow-up may satisfy one persisted exact-read
        # missing-key result. Inspect durable records before the ordinary
        # router can reinterpret that short token as an unrelated request.
        depth_runner: Any | None = None
        exact_read_continuation: dict[str, str] | None = None
        if _could_be_exact_read_continuation_reply(message):
            try:
                depth_runner = default_runner()
                exact_read_continuation = _persisted_exact_read_continuation(
                    depth_runner,
                    conversation_id=session_id,
                    message=message,
                    hivemind_url=self.hivemind_url,
                )
            except Exception as exc:
                log.warning("persisted exact-read continuation lookup failed: %s", exc)

        # ---- Auto-router decision (and revision bump + stale-older-jobs)
        route_decision = router_route(message)
        message_for_chat = route_decision.cleaned_message or message
        dispatched_job: dict[str, Any] | None = None
        depth_choice_dict: dict[str, Any] | None = None
        depth_prior_context = _face_lobe_prior_context(self.face_lobe_chat, session_id)

        # Confirmation-triggered dispatch: if the prior Face Lobe turn
        # OFFERED a deep action and this turn is a bare affirmation
        # ("ok do that", "yes please"), dispatch the offered goal instead
        # of letting the model falsely claim it dispatched. Essential for
        # voice (no way to type "/deep").
        confirm_goal: str | None = None
        if (
            _confirm_dispatch_enabled()
            and route_decision.kind != "deep"
            and session_id
            and _is_confirmation(message)
        ):
            with self._pending_deep_lock:
                confirm_goal = self._pending_deep_goal.pop(session_id, None)
        effective_deep = route_decision.kind == "deep" or confirm_goal is not None
        active_exact_read_continuation = (
            exact_read_continuation if confirm_goal is None else None
        )
        # The thing we actually dispatch (the offered goal on confirm).
        dispatch_intent = (
            confirm_goal
            or (
                active_exact_read_continuation["dispatch_intent"]
                if active_exact_read_continuation is not None
                else message_for_chat
            )
        )

        face_lobe_outcome: dict[str, Any] = {}
        face_lobe_block: str | None = None
        # We need to know dispatched_job before we build the context
        # block (so the block can include the THIS TURN DISPATCHED /
        # DID NOT DISPATCH line), so handle revision bump first, then
        # auto-router dispatch, then build the block.
        # Finding 3: re-check ownership before the revision bump — this is the
        # first mutating conversation side effect, and a cancelled turn must not
        # advance or stale the shared session's revision state.
        if _turn_cancelled():
            raise Ms4TurnCancelled("foreground turn cancelled before revision bump")
        try:
            face_lobe_outcome = face_lobe_turn_start(
                conversation_id=session_id or f"ms4-{uuid.uuid4()}",
                user_message=message_for_chat,
            )
        except Exception as exc:
            face_lobe_outcome = {"error": str(exc)}

        quartermaster_outcome: dict[str, Any] | None = None
        quartermaster_block: str | None = None

        # Offer every foreground request to the existing safe inline path
        # before the legacy direct/deep dispatch gate. Fail-soft — a None
        # block retains the route decision's prior behavior.
        quartermaster_block, quartermaster_outcome = self._try_quartermaster_inline(dispatch_intent)

        authoritative_inline_text = (
            _quartermaster_authoritative_text(quartermaster_outcome)
            if quartermaster_block is not None
            else None
        )
        authoritative_inline_guard_block = (
            _quartermaster_authoritative_guard_block(quartermaster_outcome)
            if authoritative_inline_text is not None
            else None
        )
        if voice_mode and authoritative_inline_text is not None:
            authoritative_inline_text = _quartermaster_voice_authoritative_text(
                message_for_chat,
                quartermaster_outcome,
            )
        inline_invoked = bool(
            isinstance(quartermaster_outcome, dict)
            and quartermaster_outcome.get("inline_invoked") is True
        )
        inline_terminal_failure = bool(
            isinstance(quartermaster_outcome, dict)
            and quartermaster_outcome.get("inline_terminal_failure") is True
        )
        if (
            quartermaster_block is not None
            and isinstance(quartermaster_outcome, dict)
            and quartermaster_outcome.get("inline_executed") is True
            and authoritative_inline_text is None
        ):
            log.warning(
                "quartermaster inline result lacked tool identity or payload; terminal"
            )
            quartermaster_block = None
            quartermaster_outcome["inline_terminal_failure"] = True
            inline_terminal_failure = True
        if _turn_cancelled():
            raise Ms4TurnCancelled("foreground turn cancelled after Quartermaster routing")

        quartermaster_resolution = (
            quartermaster_outcome.get("resolution")
            if isinstance(quartermaster_outcome, dict)
            else None
        )
        quartermaster_requests_depth = (
            quartermaster_block is None
            and not inline_invoked
            and not inline_terminal_failure
            and isinstance(quartermaster_outcome, dict)
            and quartermaster_outcome.get("schema") == "Ms4ToolRouteDecision.v1"
            and quartermaster_outcome.get("verdict") == "depth"
            and quartermaster_outcome.get("query") == dispatch_intent
            and isinstance(quartermaster_outcome.get("reason"), str)
            and isinstance(quartermaster_resolution, dict)
            and quartermaster_resolution.get("schema") == "Ms4ToolResolution.v1"
            and quartermaster_resolution.get("query") == dispatch_intent
            and quartermaster_outcome.get("inline_tool") is None
            and quartermaster_outcome.get("inline_executed") is not True
        )

        service_action_conflict_blocked = _quartermaster_service_action_conflict(
            dispatch_intent,
            quartermaster_outcome,
        )
        if service_action_conflict_blocked:
            effective_deep = False

        admission = _oracle_supported_admission(
            dispatch_intent,
            quartermaster_outcome,
            inline_block=quartermaster_block,
            hivemind_url=self.hivemind_url,
            inline_invoked=inline_invoked,
        )
        exact_read_canonical = (
            admission["canonical"]
            if admission["decision"] == _ADMIT_EXACT_READ
            else None
        )
        exact_gated_candidate = _exact_gated_candidate_canonical(
            dispatch_intent,
            quartermaster_outcome,
            inline_block=quartermaster_block,
        )
        exact_gated_canonical = (
            admission["canonical"]
            if admission["decision"] == _ADMIT_EXACT_GATED
            else None
        )
        explicit_direct_override = (
            route_decision.kind == "direct"
            and bool(getattr(route_decision, "override", False))
        )
        # Exact-read may promote a uniquely named current canonical on a
        # direct turn. Exact-gated stays gated: Face-direct offers only
        # store a pending intent until confirmation or an explicit /deep.
        quartermaster_strict_tool_depth = bool(
            quartermaster_requests_depth
            and exact_read_canonical is not None
            and exact_gated_canonical is None
            and not explicit_direct_override
        )
        if quartermaster_strict_tool_depth:
            effective_deep = True
        gated_proof_blocked = (
            exact_gated_candidate is not None
            and exact_gated_canonical is None
        )
        # A refused or low-confidence Quartermaster mapping must not block an
        # ordinary direct conversation. Explicit /deep stays fail-closed below.
        no_action_blocked = (
            service_action_conflict_blocked
            or gated_proof_blocked
            or inline_terminal_failure
            or (
                effective_deep
                and admission["decision"] == _ADMIT_REFUSE
                and not inline_invoked
            )
        )
        if no_action_blocked:
            effective_deep = False
        # Ordinary follow-ups reconcile before Face inference so they see exact
        # completed Depth answers. Explicit /direct is a hard no-Depth contract;
        # deep turns reconcile only after dispatch so the dispatch runner lookup
        # remains the final cancellation seam.
        if not effective_deep and not explicit_direct_override:
            reconciled_depth_job_ids = self._reconcile_depth_results(session_id)
        exact_gated_arguments = (
            quartermaster_resolution.get("deterministic_arguments")
            if exact_gated_canonical is not None
            and isinstance(quartermaster_resolution, dict)
            and isinstance(
                quartermaster_resolution.get("deterministic_arguments"),
                dict,
            )
            else None
        )
        if effective_deep and quartermaster_block is None:
            try:
                revision_id = 1
                rev_dict = face_lobe_outcome.get("revision") if isinstance(face_lobe_outcome.get("revision"), dict) else None
                if rev_dict and isinstance(rev_dict.get("revision_id"), int):
                    revision_id = int(rev_dict["revision_id"])
                reusable_foreground_model = (
                    depth_model is None
                    and isinstance(selected_model, str)
                    and bool(selected_model)
                    and selected_model.strip() == selected_model
                )
                if reusable_foreground_model and exact_read_canonical is not None:
                    depth_choice = DepthChoice(
                        model_id=selected_model,
                        source="foreground_reuse_exact_read",
                        detail="strict exact-read reused selected foreground model",
                    )
                elif reusable_foreground_model and exact_gated_canonical is not None:
                    depth_choice = DepthChoice(
                        model_id=selected_model,
                        source="foreground_reuse_exact_gated",
                        detail="strict exact-gated reused selected foreground model",
                    )
                else:
                    # Explicit Depth selection still wins verbatim; invalid or
                    # absent foreground ids retain the established picker path.
                    depth_choice = choose_depth_model(
                        hivemind_url=self.hivemind_url,
                        envelope_override=(depth_model or None),
                    )
                depth_choice_dict = depth_choice.to_dict()
                if exact_gated_canonical is not None:
                    depth_toolsets = [_DEPTH_EXACT_GATED_TOOLSET]
                    depth_internal_goal = _exact_gated_internal_goal(
                        dispatch_intent,
                        exact_gated_canonical,
                        exact_gated_arguments,
                    )
                elif exact_read_canonical is not None:
                    depth_toolsets = [_DEPTH_EXACT_READ_TOOLSET]
                    depth_internal_goal = _exact_read_internal_goal(
                        dispatch_intent,
                        exact_read_canonical,
                    )
                elif admission["decision"] == _ADMIT_NO_TOOL:
                    depth_toolsets = []
                    depth_internal_goal = dispatch_intent
                else:
                    raise SchemaError(
                        "oracle_supported_admission_refused: "
                        f"{admission['reason']}"
                    )
                envelope = JobEnvelope(
                    job_id=new_job_id(),
                    parent_conversation_id=session_id or "ms4-conv",
                    conversation_revision_id=revision_id,
                    background_lobe_type="deep_chat",
                    user_visible_goal=(confirm_goal or route_decision.goal or message_for_chat)[:240],
                    internal_goal=depth_internal_goal,
                    prior_context=depth_prior_context,
                    resource_request=ResourceRequest(
                        model_override=depth_choice.model_id,
                        enabled_toolsets=depth_toolsets,
                    ),
                )
                envelope.validate()
                # Finding 3: last ownership gate before we hand a background job
                # to the runner. A cancelled turn must dispatch NO Depth job.
                if _turn_cancelled():
                    raise Ms4TurnCancelled("foreground turn cancelled before Depth dispatch")
                if (
                    active_exact_read_continuation is not None
                    and exact_read_canonical is not None
                    and exact_read_canonical.casefold()
                    == active_exact_read_continuation["canonical_tool"].casefold()
                ):
                    _persist_exact_read_continuation_link(
                        depth_runner,
                        active_exact_read_continuation,
                        continuation_job_id=envelope.job_id,
                    )
                dispatch_runner = depth_runner or default_runner()
                if _turn_cancelled():
                    raise Ms4TurnCancelled(
                        "foreground turn cancelled at final Depth dispatch"
                    )
                dispatched_job = dispatch_runner.submit(envelope)
            except Ms4TurnCancelled:
                # Finding 3: cancellation is NOT a dispatch error — it must
                # propagate so the turn commits nothing, not be swallowed into a
                # {"error": ...} job stub.
                raise
            except SchemaError as exc:
                dispatched_job = {"error": f"router refused to dispatch: {exc}"}
            except Exception as exc:
                dispatched_job = {"error": f"router dispatch failed: {exc}"}

        if effective_deep and not _turn_cancelled():
            reconciled_depth_job_ids = self._reconcile_depth_results(session_id)

        # Current date/time as authoritative grounding so simple
        # questions like "what is the date?" don't need a dispatch.
        # Prefer the HiveMind cluster's clock when reachable so
        # operators running MS4 in a different timezone don't get
        # answers that disagree with the cluster's own logs/job
        # timestamps. Local time stays as the fail-soft fallback.
        now_local = _now_local_iso()
        cluster_time = (
            _quartermaster_inline_cluster_time(quartermaster_outcome)
            if quartermaster_block is not None
            else None
        )
        if cluster_time is None:
            cluster_time = self._try_cluster_time()
        if cluster_time:
            date_grounding_line = (
                f"current cluster date/time: {cluster_time} "
                f"(local: {now_local})"
            )
        else:
            date_grounding_line = f"current local date/time: {now_local}"
        dispatched_job_id = (
            dispatched_job["job_id"]
            if isinstance(dispatched_job, dict) and dispatched_job.get("job_id")
            else None
        )
        dispatch_error = _safe_dispatch_error(dispatched_job)
        authoritative_lines = [date_grounding_line]
        dispatch_failure_line = _dispatch_failure_authority_line(dispatched_job)
        if dispatch_failure_line:
            authoritative_lines.append(dispatch_failure_line)
        try:
            face_lobe_block = build_face_lobe_context_block(
                conversation_id=session_id or "",
                dispatched_this_turn_job_id=dispatched_job_id,
                extra_authoritative_lines=authoritative_lines,
            )
        except Exception as exc:
            log.warning("face_lobe context block failed: %s", exc)
            face_lobe_block = (
                "MS4 Face Lobe context (authoritative):\n"
                f"- {date_grounding_line}\n"
                + (
                    f"- THIS TURN DISPATCHED job {dispatched_job_id} to the Depth Lobe.\n"
                    if dispatched_job_id
                    else "- THIS TURN DID NOT DISPATCH any background work.\n"
                )
            )
            if dispatch_failure_line:
                face_lobe_block += f"- {dispatch_failure_line}\n"

        # ---- Grounded user message (TMR canon, inventory, tools-list, etc.)
        grounded_message, grounding_source = build_grounded_user_message(message_for_chat, self.hivemind_url)
        extra_system_blocks: list[str] = []
        if face_lobe_block:
            extra_system_blocks.append(face_lobe_block)
        # Quartermaster inline result remains guard grounding even though its
        # successful payload now bypasses Face inference.
        if quartermaster_block:
            extra_system_blocks.append(quartermaster_block)
        if authoritative_inline_guard_block:
            extra_system_blocks.append(authoritative_inline_guard_block)
        if grounded_message != message_for_chat:
            extra_system_blocks.append(grounded_message.split("\n\nUser request:", 1)[0])
        # Voice turns are spoken aloud via TTS: request efficient delivery
        # without dropping requested substance or deferring it to another turn.
        if voice_mode:
            extra_system_blocks.append(_VOICE_BREVITY_DIRECTIVE)
        extra_system = "\n\n".join(extra_system_blocks) if extra_system_blocks else None

        # ---- Direct chat call (the actual latency-sensitive bit)
        completed = True
        api_calls: int | None = 1
        metrics: dict[str, Any] | None = None
        face_result: dict[str, Any] = {}
        effective_model = selected_model
        fallback_used = False
        quartermaster_result_fallback = False
        wall_start = time.monotonic()
        fabrication_guard: dict[str, Any] = {
            "schema": "Ms4FabricationGuard.v1",
            "applied": False,
            "reason": None,
        }
        if no_action_blocked:
            if service_action_conflict_blocked:
                block_reason = "service_action_conflict"
                text = _gated_proof_block_user_text()
            elif gated_proof_blocked:
                block_reason = "gated_proof_blocked"
                text = _gated_proof_block_user_text()
            elif inline_terminal_failure:
                block_reason = "inline_terminal_failure"
                text = _inline_terminal_user_text()
            else:
                block_reason = "admission_refused"
                text = _admission_refused_user_text(admission["reason"])
            if stream_callback is not None:
                try:
                    stream_callback(text)
                except Exception:
                    pass
            completed = True
            api_calls = 0
            chat_session_id = session_id
            metrics = {
                "schema": "Ms4TurnMetrics.v1",
                "duration_ms": int((time.monotonic() - wall_start) * 1000),
                "requested_model": selected_model,
                "streaming": stream_callback is not None,
                "no_action_blocked": True,
                "block_reason": block_reason,
            }
            fabrication_guard = {
                "schema": "Ms4FabricationGuard.v1",
                "applied": True,
                "reason": block_reason,
            }
        elif dispatch_error:
            text = _dispatch_failure_user_text(dispatch_error)
            log.warning("Depth Lobe dispatch failed: %s", dispatch_error)
            if stream_callback is not None:
                try:
                    stream_callback(text)
                except Exception:
                    pass
            completed = False
            api_calls = 0
            chat_session_id = session_id
            metrics = {
                "schema": "Ms4TurnMetrics.v1",
                "duration_ms": int((time.monotonic() - wall_start) * 1000),
                "dispatch_error": dispatch_error,
                "requested_model": selected_model,
                "streaming": stream_callback is not None,
                "fabrication_guard": True,
            }
            fabrication_guard = {
                "schema": "Ms4FabricationGuard.v1",
                "applied": True,
                "reason": "dispatch_failed",
                "dispatch_error": dispatch_error,
            }
        else:
            try:
                face_chat_kwargs: dict[str, Any] = {
                    "session_id": session_id,
                    "model": selected_model,
                    "stream_callback": stream_callback,
                    "extra_system": extra_system,
                }
                # Keep the established FaceLobeChat/fake-client contract for
                # ordinary turns. Voice adds this only while its first-token
                # watchdog is active.
                if cancel_event is not None:
                    face_chat_kwargs["cancel_event"] = cancel_event
                if authoritative_inline_text is not None:
                    face_result = self.face_lobe_chat.chat_authoritative(
                        message_for_chat,
                        authoritative_text=authoritative_inline_text,
                        **face_chat_kwargs,
                    )
                else:
                    if voice_mode:
                        face_chat_kwargs["substantive_word_range"] = (
                            _VOICE_SUBSTANTIVE_MIN_WORDS,
                            _VOICE_SUBSTANTIVE_MAX_WORDS,
                        )
                    lookup = getattr(self.face_lobe_chat, "recommend_receipt_for", None)
                    face_receipt = recommend_receipt
                    if not face_receipt and callable(lookup):
                        face_receipt = lookup(session_id, client_id, selected_model)
                    if face_receipt:
                        face_chat_kwargs["recommend_receipt"] = face_receipt
                        face_chat_kwargs["recommend_validate_only"] = False
                    face_result = self.face_lobe_chat.chat(message_for_chat, **face_chat_kwargs)
                text = face_result["text"]
                chat_session_id = face_result["session_id"]
                effective_model = face_result.get("model") or selected_model
                fallback_used = bool(face_result.get("fallback_used"))
                raw_api_calls = face_result.get("api_calls")
                api_calls = int(raw_api_calls) if raw_api_calls is not None else 1
                completed = bool(face_result.get("completed", False))
                metrics = face_result.get("metrics")
                output_guard = face_result.get("output_guard")
                if isinstance(output_guard, dict) and output_guard.get("applied"):
                    fabrication_guard = {
                        "schema": "Ms4FabricationGuard.v1",
                        "applied": True,
                        "reason": "face_lobe_output_guard",
                        "output_guard": output_guard,
                    }
            except FaceLobeChatError as exc:
                if getattr(exc, "fail_closed", False):
                    log.warning("Face Lobe recommend lease failed closed: %s", exc)
                    return {
                        "text": "",
                        "session_id": session_id,
                        "model": selected_model,
                        "completed": False,
                        "cancelled": False,
                        "fail_closed": True,
                        "error": str(exc),
                        "metrics": {
                            "schema": "Ms4TurnMetrics.v1",
                            "error": str(exc),
                            "requested_model": selected_model,
                            "fail_closed": True,
                        },
                    }
                # User-visible canned reply: the raw "HiveMind /v1/chat/completions
                # (stream) unreachable: timed out" is ugly and meaningless to a
                # voice user. Translate it into something the operator can act on
                # AND something the TTS pipeline can synthesize into a sensible
                # audible reply. The raw exception goes into metrics.error for
                # debug + the audit log.
                authoritative_fallback = (
                    authoritative_inline_text
                    if voice_mode and authoritative_inline_text is not None
                    else _quartermaster_result_fallback_text(
                        quartermaster_outcome,
                        quartermaster_block,
                    )
                )
                if authoritative_fallback:
                    text = authoritative_fallback
                    quartermaster_result_fallback = True
                else:
                    text = _canned_chat_failure_text(exc)
                log.warning("Face Lobe chat call failed: %s", exc)
                # Stream the canned text to the caller's stream_callback so the
                # WS_SUPER TTS path still gets audio to synthesize and the user
                # actually hears the apology instead of silence.
                if stream_callback is not None:
                    try:
                        stream_callback(text)
                    except Exception:
                        pass
                completed = quartermaster_result_fallback
                api_calls = 0
                chat_session_id = session_id
                metrics = {
                    "schema": "Ms4TurnMetrics.v1",
                    "duration_ms": int((time.monotonic() - wall_start) * 1000),
                    "error": str(exc),
                    "requested_model": selected_model,
                    "streaming": stream_callback is not None,
                    "canned_reply": not quartermaster_result_fallback,
                    "quartermaster_result_fallback": quartermaster_result_fallback,
                }

        # Confirmation-triggered-dispatch bookkeeping: if this turn did NOT
        # dispatch but the Face Lobe OFFERED deep work, remember the goal so
        # the next affirmation runs it. Otherwise clear any stale offer.
        if _confirm_dispatch_enabled() and session_id:
            if dispatched_job is None and not effective_deep and _looks_like_deep_offer(text):
                with self._pending_deep_lock:
                    self._pending_deep_goal[session_id] = message_for_chat[:240]
            else:
                with self._pending_deep_lock:
                    self._pending_deep_goal.pop(session_id, None)

        public_tool_trace = (
            _quartermaster_public_tool_trace(quartermaster_outcome)
            if quartermaster_block is not None
            else []
        )
        response = {
            "text": text,
            "session_id": chat_session_id,
            "hermes_session_id": chat_session_id,  # kept for backward compat with old UI fields
            "model": effective_model,
            "grounding_source": _combine_grounding(grounding_source, face_lobe_block, dispatched_job),
            "api_calls": api_calls,
            "completed": completed,
            "cancelled": bool((face_result or {}).get("cancelled", False)),
            "tool_trace": public_tool_trace,
            "runtime": "face-lobe-direct",
            "ms3_sidecar_url": self.ms3_url,
            "hivemind_url": self.hivemind_url,
            "face_lobe": face_lobe_outcome,
            "face_lobe_model": face_lobe_model,
            "face_lobe_context_block": face_lobe_block,
            "router": route_decision.to_dict(),
            "dispatched_job": dispatched_job,
            "confirm_dispatch": confirm_goal is not None,
            "depth_lobe_model": depth_choice_dict,
            "depth_lobe_prior_context_count": len(depth_prior_context),
            "reconciled_depth_job_ids": reconciled_depth_job_ids,
            "quartermaster": quartermaster_outcome,
            "quartermaster_inline": bool(quartermaster_block),
            "quartermaster_result_fallback": quartermaster_result_fallback,
            "fallback_used": fallback_used,
            "requested_model": selected_model if fallback_used else None,
            "metrics": metrics,
            "fabrication_guard": fabrication_guard,
            "face_lobe_output_guard": (face_result or {}).get("output_guard"),
        }
        recommend_evidence = (face_result or {}).get("recommend")
        if isinstance(recommend_evidence, dict) and recommend_evidence:
            response["recommend"] = recommend_evidence
        append_event("chat_turn", {
            "session_id": chat_session_id,
            "model": effective_model,
            "requested_model": selected_model,
            "fallback_used": fallback_used,
            "grounding_source": response["grounding_source"],
            "tool_trace_count": len(public_tool_trace),
            "completed": completed,
            "runtime": "face-lobe-direct",
            "router_kind": route_decision.kind,
            "dispatched_job_id": dispatched_job_id,
            "depth_lobe_prior_context_count": len(depth_prior_context),
            "quartermaster_inline": bool(quartermaster_block),
            "quartermaster_result_fallback": quartermaster_result_fallback,
            "quartermaster_verdict": (quartermaster_outcome or {}).get("verdict") if isinstance(quartermaster_outcome, dict) else None,
            "revision_id": face_lobe_outcome.get("revision", {}).get("revision_id") if isinstance(face_lobe_outcome.get("revision"), dict) else None,
            "marked_stale_jobs": face_lobe_outcome.get("marked_stale", []),
            "metrics": metrics,
            "fabrication_guard": fabrication_guard,
            "recommend_owner": (response.get("recommend") or {}).get("owner"),
            "recommend_endpoint": (response.get("recommend") or {}).get("endpoint"),
            "recommend_generation": (response.get("recommend") or {}).get("generation"),
            "recommend_lease_id": (response.get("recommend") or {}).get("lease_id"),
        })
        return response

    def _try_cluster_time(self) -> str | None:
        """Cached, fail-soft wrapper around ``hivemind.time.now@v1``.

        Successful probes are cached for 30 s; failures for 5 s so a
        transiently-offline cluster forces only one slow probe per
        ~5 s window. The probe itself is capped at ~2 s by
        :func:`_try_cluster_time_now` and the underlying MCP timeout.
        """
        now = time.time()
        with self._cluster_time_lock:
            cached_at, cached_value = self._cluster_time_cache
            age = now - cached_at
            if cached_value is not None and age < 30.0:
                return cached_value
            if cached_value is None and age < 5.0:
                return None
        value = _try_cluster_time_now(self.hivemind_url)
        with self._cluster_time_lock:
            self._cluster_time_cache = (time.time(), value)
        return value

    def sessions(self) -> list[dict[str, Any]]:
        # Foreground sessions live in face_lobe_chat now; Hermes-backed
        # ad-hoc tool sessions (from dispatch_hermes_tool) live in
        # self._sessions for backward compatibility. List both with a
        # `runtime` discriminator.
        out: list[dict[str, Any]] = []
        for state in self.face_lobe_chat.sessions():
            out.append({
                "session_id": state["session_id"],
                "hermes_session_id": state["session_id"],
                "turns": state["turns"],
                "model": state["model"],
                "last_grounding_source": state["last_grounding_source"],
                "tool_trace_count": 0,
                "runtime": "face-lobe-direct",
            })
        for state in self._sessions.values():
            out.append({
                "session_id": state.session_id,
                "hermes_session_id": getattr(state.agent, "session_id", state.session_id),
                "turns": len(state.history),
                "model": getattr(state.agent, "model", self.default_model),
                "last_grounding_source": state.last_grounding_source,
                "tool_trace_count": len(state.tool_trace),
                "runtime": "hermes-tool",
            })
        return out

    def health(self) -> dict[str, Any]:
        return {
            "runtime": "ms4-fusion",
            "hermes_dir": str(self.hermes_dir),
            "hivemind_url": self.hivemind_url,
            "ms3_url": self.ms3_url,
            "plugin": self.plugin_status(),
            "sessions": len(self._sessions),
        }

    def list_hermes_tools(self, enabled_toolsets: list[str] | None = None) -> list[dict[str, Any]]:
        self.ensure_hermes_path()
        from model_tools import get_tool_definitions, get_toolset_for_tool

        disabled_toolsets = (
            ["kanban"]
            if enabled_toolsets is not None and not enabled_toolsets
            else None
        )
        definitions = get_tool_definitions(
            enabled_toolsets=enabled_toolsets,
            disabled_toolsets=disabled_toolsets,
            quiet_mode=True,
        )
        tools = []
        for definition in definitions:
            function = definition.get("function", {})
            name = function.get("name")
            if not name:
                continue
            tools.append({
                "name": name,
                "description": function.get("description", ""),
                "toolset": get_toolset_for_tool(name),
                "schema": function,
            })
        return tools

    def dispatch_hermes_tool(
        self,
        tool_name: str,
        args: dict[str, Any] | None = None,
        *,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        tool_args = dict(args or {})
        admission = _oracle_supported_direct_admission(
            tool_name,
            tool_args,
            self.hivemind_url,
        )
        if admission["decision"] not in {_ADMIT_EXACT_READ, _ADMIT_EXACT_GATED}:
            sid = session_id or f"ms4-tool-{uuid.uuid4()}"
            return {
                "tool": tool_name,
                "toolset": None,
                "args": tool_args,
                "result": None,
                "error": _admission_refused_user_text(admission["reason"]),
                "refused": True,
                "session_id": sid,
                "runtime": "hermes",
            }
        self.require_plugin()
        os.environ["MS4_MS3_SIDECAR_URL"] = self.ms3_url
        os.environ.setdefault("MS4_SPIRIT_ID", "sister")
        self.ensure_hermes_path()
        from model_tools import handle_function_call, get_toolset_for_tool

        sid = session_id or f"ms4-tool-{uuid.uuid4()}"
        result = handle_function_call(
            tool_name,
            tool_args,
            task_id=sid,
            session_id=sid,
            tool_call_id=f"ms4-direct-{uuid.uuid4()}",
        )
        response = {
            "tool": tool_name,
            "toolset": get_toolset_for_tool(tool_name),
            "args": tool_args,
            "result": result,
            "session_id": sid,
            "runtime": "hermes",
        }
        append_event("hermes_tool_call", {
            "session_id": sid,
            "tool": tool_name,
            "toolset": response["toolset"],
            "args": tool_args,
            "result_excerpt": str(result)[:1000],
        })
        return response
