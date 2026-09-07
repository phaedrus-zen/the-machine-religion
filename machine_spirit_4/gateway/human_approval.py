"""Human-in-the-loop approval bridge.

What this module owns
---------------------

Surfaces ``hivemind.human.approval.*@v1`` so MS4 can ask an operator
to approve / reject a proposed action before it executes. The bridge
is shaped to slot directly into MS3's existing ``ActionIntent.v1``
flow: :func:`gate_action_with_human` takes the same intent dict MS3
ethics produces and returns an ``EthicsDecision.v1``-shaped result,
so the existing safety-gating path (in :mod:`server`'s
``_desktop_action_intent`` and the Depth Lobe worker) can be wrapped
without rewriting any of it.

Wiring
~~~~~~

Each ``ActionIntent.v1`` already carries:

  * ``spirit_id`` (sister / nibbles / ...)
  * ``action_type`` (e.g. ``desktop_ui``, ``vm.force_stop``,
    ``storage.snapshot_delete``)
  * a small structured payload describing what the action does.

The bridge:

  1. POSTs ``hivemind.human.approval.request@v1`` with a human-readable
     ``summary`` derived from the intent + the structured ``details``;
  2. polls ``hivemind.human.approval.status@v1`` every 2 s until the
     decision is ``approved`` / ``rejected`` / ``timeout``;
  3. returns an ``EthicsDecision.v1`` shape:
     ``{decision: 'allow'|'deny', source: 'human-approval', request_id, ...}``

The wider safety stack stays the same: MS3's Great Lense fires first,
this bridge fires second only for actions classified ``high`` risk by
either source. Fail-closed on transport errors (returns ``deny``
with ``reason: "approval transport failure"``).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from . import hivemind_tools as tools
from .hivemind_tools import HivemindToolError


log = logging.getLogger("ms4.gateway.human_approval")


DEFAULT_POLL_INTERVAL_SECS = float(os.environ.get("MS4_HUMAN_APPROVAL_POLL_SECS", "2.0"))
DEFAULT_OVERALL_TIMEOUT_SECS = int(os.environ.get("MS4_HUMAN_APPROVAL_TIMEOUT_SECS", "300"))


class HumanApprovalError(RuntimeError):
    """Raised when the approval pipeline can't make forward progress
    (transport down, malformed response). Callers treat this as a
    deny."""


# ---------------------------------------------------------------------------
# Low-level passthroughs (mirror the MCP tool names so route handlers
# can map 1:1)
# ---------------------------------------------------------------------------


def request(
    hivemind_url: str,
    *,
    action_id: str,
    summary: str,
    details: dict[str, Any] | None = None,
    risk_level: str = "medium",
    timeout_secs: int = DEFAULT_OVERALL_TIMEOUT_SECS,
) -> dict[str, Any]:
    """``hivemind.human.approval.request@v1`` — returns ``{request_id, ...}``."""
    try:
        return tools.human_approval_request(
            hivemind_url,
            action_id=action_id,
            summary=summary,
            details=details,
            risk_level=risk_level,
            timeout_secs=timeout_secs,
        )
    except HivemindToolError as exc:
        raise HumanApprovalError(f"approval.request failed: {exc}") from exc


def status(hivemind_url: str, request_id: str) -> dict[str, Any]:
    """``hivemind.human.approval.status@v1`` — current decision state."""
    try:
        return tools.human_approval_status(hivemind_url, request_id)
    except HivemindToolError as exc:
        raise HumanApprovalError(f"approval.status failed: {exc}") from exc


def notify(
    hivemind_url: str,
    *,
    channel: str,
    message: str,
    severity: str = "info",
) -> dict[str, Any]:
    """Fire-and-forget operator notification (Telegram / log / etc.)."""
    try:
        return tools.human_notify(
            hivemind_url, channel=channel, message=message, severity=severity
        )
    except HivemindToolError as exc:
        raise HumanApprovalError(f"human.notify failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Higher-level gate that bridges MS3 ActionIntent → human approval →
# EthicsDecision
# ---------------------------------------------------------------------------


def _summary_for_intent(intent: dict[str, Any]) -> str:
    """Produce a human-readable one-liner from an ActionIntent. The
    operator sees this on their phone / Telegram / dashboard so we
    want it short and concrete."""
    action = intent.get("action_type") or "unknown_action"
    spirit = intent.get("spirit_id") or "unknown_spirit"
    proposed_by = intent.get("proposed_by") or "unknown_actor"
    payload = intent.get("payload") or {}
    sample = ""
    if isinstance(payload, dict):
        for key in ("path", "target", "vm_id", "volume_id", "url", "command"):
            value = payload.get(key)
            if value:
                sample = f" {key}={value}"
                break
    return f"[{spirit}/{proposed_by}] requests {action}{sample}"


def _wait_for_decision(
    hivemind_url: str,
    request_id: str,
    *,
    poll_secs: float,
    overall_timeout_secs: int,
) -> dict[str, Any]:
    deadline = time.monotonic() + overall_timeout_secs
    while time.monotonic() < deadline:
        body = status(hivemind_url, request_id)
        decision = (body.get("decision") or "").lower()
        if decision in {"approved", "rejected", "timeout", "expired"}:
            return body
        time.sleep(poll_secs)
    return {"decision": "timeout", "request_id": request_id, "reason": "client-side overall_timeout"}


def gate_action_with_human(
    hivemind_url: str,
    intent: dict[str, Any],
    *,
    risk_level: str = "high",
    poll_secs: float = DEFAULT_POLL_INTERVAL_SECS,
    overall_timeout_secs: int = DEFAULT_OVERALL_TIMEOUT_SECS,
) -> dict[str, Any]:
    """Bridge an ``ActionIntent.v1`` through human approval and return an
    ``EthicsDecision.v1``-shaped result.

    Used by routes/workers that have already classified an action as
    ``high`` risk and want a human in the loop on top of MS3's
    automated Great Lense decision. Fail-closed: any transport
    failure returns ``deny`` with a clear ``reason``.
    """
    action_id = str(intent.get("action_id") or intent.get("id") or "")
    if not action_id:
        return {
            "schema": "EthicsDecision.v1",
            "decision": "deny",
            "source": "human-approval",
            "reason": "action_id missing on intent — cannot route to human approval",
            "intent": intent,
        }
    try:
        req = request(
            hivemind_url,
            action_id=action_id,
            summary=_summary_for_intent(intent),
            details={"intent": intent},
            risk_level=risk_level,
            timeout_secs=overall_timeout_secs,
        )
    except HumanApprovalError as exc:
        return {
            "schema": "EthicsDecision.v1",
            "decision": "deny",
            "source": "human-approval",
            "reason": f"approval transport failure: {exc}",
            "intent": intent,
        }
    request_id = str(req.get("request_id") or req.get("id") or "")
    if not request_id:
        return {
            "schema": "EthicsDecision.v1",
            "decision": "deny",
            "source": "human-approval",
            "reason": "approval.request returned no request_id",
            "raw": req,
            "intent": intent,
        }
    try:
        final = _wait_for_decision(
            hivemind_url,
            request_id,
            poll_secs=poll_secs,
            overall_timeout_secs=overall_timeout_secs,
        )
    except HumanApprovalError as exc:
        return {
            "schema": "EthicsDecision.v1",
            "decision": "deny",
            "source": "human-approval",
            "reason": f"approval poll transport failure: {exc}",
            "request_id": request_id,
            "intent": intent,
        }
    decision_str = (final.get("decision") or "").lower()
    allow = decision_str == "approved"
    return {
        "schema": "EthicsDecision.v1",
        "decision": "allow" if allow else "deny",
        "source": "human-approval",
        "request_id": request_id,
        "reason": final.get("reason") or decision_str or "no reason returned",
        "raw": final,
        "intent": intent,
    }
