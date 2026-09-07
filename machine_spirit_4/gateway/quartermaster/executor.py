"""Inline tool executor — runs one resolved read-only tool and formats
its result into an authoritative Face Lobe context block.

Only ever called by the Phase-D fast path for a tool the
:mod:`router` cleared as ``inline`` (read-only, zero required args,
ethics-allowed). Execution is by MCP tool name with empty arguments
via the shared :func:`hivemind_state.post_mcp_envelope` transport
(direct MCP 6105 -> HLI proxy fallback + bearer auth) — the same path
the typed ``hivemind_tools`` wrappers use, just name-dispatched.

Fail-soft contract: :func:`execute_inline_tool` returns ``None`` on
ANY failure so the caller falls back to the Depth Lobe. We never
fabricate a result (anti-hallucination): if the tool didn't return
data, the inline path is abandoned, not faked.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any


log = logging.getLogger("ms4.gateway.quartermaster.executor")


def _exec_timeout() -> float:
    try:
        return float(os.environ.get("MS4_QM_INLINE_TIMEOUT", "8.0"))
    except (TypeError, ValueError):
        return 8.0


# Cap the rendered result so a chatty tool can't blow up the Face Lobe
# prompt. Operators who want the full payload use the Depth Lobe.
_RESULT_CHAR_LIMIT = 1800


def execute_inline_tool(
    hivemind_url: str,
    tool_name: str,
    *,
    arguments: dict[str, Any] | None = None,
    timeout: float | None = None,
) -> dict[str, Any] | None:
    """Execute ``tool_name`` (empty args by default) and return
    ``{tool, result, elapsed_ms}`` on success, or ``None`` on any
    failure / empty result."""
    t0 = time.monotonic()

    # Imported 3rd-party MCP tools (ext.<server>.<tool>@v1) execute via
    # the mcp_bridge proxy, NOT HiveMind's MCP. (They only reach here if
    # the router cleared them as inline — i.e. read_only + opted-in +
    # ethics-allowed.)
    if tool_name.startswith("ext."):
        try:
            from ...mcp_bridge.proxy import call_imported_tool
        except Exception as exc:  # pragma: no cover — import guard
            log.debug("quartermaster executor mcp_bridge import failed: %s", exc)
            return None
        out = call_imported_tool(tool_name, arguments or {})
        if not isinstance(out, dict) or not out.get("ok") or out.get("is_error"):
            log.info("quartermaster inline exec %s via bridge failed (-> depth)", tool_name)
            return None
        result = out.get("result")
        if result is None:
            return None
        return {"tool": tool_name, "result": result, "elapsed_ms": int((time.monotonic() - t0) * 1000)}

    try:
        from ..hivemind_state import post_mcp_envelope
    except Exception as exc:  # pragma: no cover — import guard
        log.debug("quartermaster executor import failed: %s", exc)
        return None

    payload = {
        "jsonrpc": "2.0",
        "id": f"ms4-qm-{uuid.uuid4().hex[:8]}",
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments or {}},
    }
    try:
        envelope = post_mcp_envelope(
            hivemind_url, payload, timeout=int(timeout or _exec_timeout())
        )
    except Exception as exc:
        log.info("quartermaster inline exec %s failed (-> depth): %s", tool_name, exc)
        return None

    result = _unwrap(envelope)
    if result is None:
        log.info("quartermaster inline exec %s returned no usable result (-> depth)", tool_name)
        return None
    return {
        "tool": tool_name,
        "result": result,
        "elapsed_ms": int((time.monotonic() - t0) * 1000),
    }


def _unwrap(envelope: Any) -> Any:
    """Unwrap an MCP tools/call envelope to the tool's JSON payload.
    Returns None on protocol error, isError, or empty content."""
    if not isinstance(envelope, dict):
        return None
    if envelope.get("error"):
        return None
    result = envelope.get("result")
    if not isinstance(result, dict):
        return None
    if result.get("isError"):
        return None
    content = result.get("content")
    if isinstance(content, list):
        for chunk in content:
            if isinstance(chunk, dict) and chunk.get("type") == "text":
                inner = chunk.get("text") or ""
                if not inner.strip():
                    continue
                try:
                    return json.loads(inner)
                except json.JSONDecodeError:
                    return {"raw_text": inner}
    # Some tools return a bare result dict with no content wrapper.
    if result and "content" not in result:
        return result
    return None


def format_inline_block(executed: dict[str, Any], *, query: str | None = None) -> str:
    """Render an executed inline result as an authoritative Face Lobe
    context block. The freshness is ``just now`` because we executed
    it this turn (mirrors the ``cached(age=Ns)`` qualifier vocabulary
    the Face Lobe prompt already understands)."""
    tool = executed.get("tool", "?")
    result = executed.get("result")
    elapsed = executed.get("elapsed_ms")
    rendered = _render_result(result)
    lines = [
        "MS4 Quartermaster inline tool result (authoritative; quote this faithfully, "
        "do not invent additional tools, data, or fields):",
        f"- tool: {tool}  (read-only, executed just now in {elapsed} ms)",
        f"- result: {rendered}",
        "This is the answer to the operator's request. Summarise it in plain language. "
        "If the result is empty or says nothing is present, say so plainly — do not fabricate.",
    ]
    return "\n".join(lines)


def _render_result(result: Any) -> str:
    if result is None:
        return "(no data)"
    if isinstance(result, str):
        text = result
    else:
        try:
            text = json.dumps(result, ensure_ascii=False, default=str)
        except Exception:
            text = str(result)
    text = " ".join(text.split())
    if len(text) > _RESULT_CHAR_LIMIT:
        text = text[: _RESULT_CHAR_LIMIT - 3].rstrip() + "..."
    return text
