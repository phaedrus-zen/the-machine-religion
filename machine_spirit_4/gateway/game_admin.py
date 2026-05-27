"""HiveMind game-session orchestration admin.

What this module owns
---------------------

The May 26 2026 HiveMind release shipped a Phase-1 dry-run game-
session orchestrator that exactly matches the game-ready VM layer
MS4 was planning. MS4 wraps it so:

  * the Settings "Game sessions" panel can plan + run + inspect
    sessions without learning HiveMind's MCP shape
  * other agents driving MS4 via MCP can orchestrate game streams
    end-to-end and inherit MS4's ethics + audit pipeline
  * when HiveMind ships Phase 2 (real VM/stream execution), MS4's
    wrappers keep working unchanged — the contract is stable

Wraps:

  * ``hivemind.game.ensure_available@v1`` — availability probe
  * ``hivemind.game_session.plan@v1`` — produce a dry-run Plan
  * ``hivemind.game_session.run@v1`` — walk the simulated state
    machine
  * ``hivemind.game_session.status@v1`` — current state + transitions
  * ``hivemind.game_session.evidence@v1`` — per-phase ledger
  * ``hivemind.game_session.cancel@v1`` — terminal CANCELLED

Phase 1 contract: every mutation is recorded as a
``would_call <hivemind.vm.X@v1>`` evidence row but NEVER actually
hits a mutating endpoint. The Plan + evidence ledger is the unit
of truth for "what WOULD this game session do?" until HiveMind
flips Phase 2 on.
"""

from __future__ import annotations

import logging
from typing import Any

from . import hivemind_tools as tools
from .hivemind_tools import HivemindToolError


log = logging.getLogger("ms4.gateway.game_admin")


class GameAdminError(RuntimeError):
    """Raised on non-recoverable game-session admin failure."""


# ---------------------------------------------------------------------------
# Availability check (read-only)
# ---------------------------------------------------------------------------


def ensure_available(hivemind_url: str, game_id: str) -> dict[str, Any]:
    """Resolve ``game_id`` (accepts aliases like ``cyberpunk-2077``)
    to an env override path, default install path, or a golden
    VHDX. Returns ``Ms4GameAvailability.v1`` shape.

    Honest about missing games: when unavailable the result includes
    a structured ``remediation`` block so the UI can render
    actionable guidance instead of a generic error.
    """
    if not game_id:
        raise ValueError("game_id is required")
    try:
        raw = tools.game_ensure_available(hivemind_url, game_id=game_id)
    except HivemindToolError as exc:
        raise GameAdminError(f"game.ensure_available {game_id!r} failed: {exc}") from exc
    body = raw if isinstance(raw, dict) else {"raw": raw}
    return {"schema": "Ms4GameAvailability.v1", "game_id": game_id, **body}


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


def plan(
    hivemind_url: str,
    *,
    game: str,
    client: str | None = None,
    duration_hint: str | None = None,
    latency: str | None = None,
    quality: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Produce a dry-run Plan for a workload intent.

    Returns the raw HiveMind response (which already carries
    ``job_id`` + ``plan``). MS4 doesn't reshape it because the Plan
    fields are the ground-truth contract callers want to see.
    """
    if not game:
        raise ValueError("game is required")
    try:
        return tools.game_session_plan(
            hivemind_url,
            game=game,
            client=client,
            duration_hint=duration_hint,
            latency=latency,
            quality=quality,
            **extra,
        )
    except HivemindToolError as exc:
        raise GameAdminError(f"game_session.plan {game!r} failed: {exc}") from exc


def run(hivemind_url: str, job_id: str) -> dict[str, Any]:
    """Walk the simulated state machine for ``job_id`` until a
    terminal state. Phase-1 dry-run only."""
    if not job_id:
        raise ValueError("job_id is required")
    try:
        return tools.game_session_run(hivemind_url, job_id=job_id)
    except HivemindToolError as exc:
        raise GameAdminError(f"game_session.run {job_id!r} failed: {exc}") from exc


def status(hivemind_url: str, job_id: str) -> dict[str, Any]:
    """Current state-machine position for a job — safe to poll
    while ``run`` is walking."""
    if not job_id:
        raise ValueError("job_id is required")
    try:
        return tools.game_session_status(hivemind_url, job_id=job_id)
    except HivemindToolError as exc:
        raise GameAdminError(f"game_session.status {job_id!r} failed: {exc}") from exc


def evidence(hivemind_url: str, job_id: str) -> dict[str, Any]:
    """Full per-phase evidence ledger."""
    if not job_id:
        raise ValueError("job_id is required")
    try:
        return tools.game_session_evidence(hivemind_url, job_id=job_id)
    except HivemindToolError as exc:
        raise GameAdminError(f"game_session.evidence {job_id!r} failed: {exc}") from exc


def cancel(hivemind_url: str, job_id: str) -> dict[str, Any]:
    """Move a job to CANCELLED. Idempotent."""
    if not job_id:
        raise ValueError("job_id is required")
    try:
        return tools.game_session_cancel(hivemind_url, job_id=job_id)
    except HivemindToolError as exc:
        raise GameAdminError(f"game_session.cancel {job_id!r} failed: {exc}") from exc


# ---------------------------------------------------------------------------
# UI snapshot helper: plan + run + evidence in one round-trip
# ---------------------------------------------------------------------------


def plan_run_and_collect(
    hivemind_url: str,
    *,
    game: str,
    client: str | None = None,
    quality: str | None = None,
    latency: str | None = None,
    duration_hint: str | None = None,
) -> dict[str, Any]:
    """Convenience for the UI: plan → run → fetch evidence in one
    call. Useful for the "Run dry-run session" button.

    Returns ``Ms4GameSession.v1`` shape:
      ``{schema, game, plan, run_result, evidence, errors[]}``.

    Fail-soft per stage so the UI can show partial state when one
    leg fails (e.g. run timed out but plan + evidence still exist).
    """
    snap: dict[str, Any] = {
        "schema": "Ms4GameSession.v1",
        "game": game,
        "plan": None,
        "job_id": None,
        "run_result": None,
        "evidence": None,
        "errors": [],
    }
    try:
        plan_body = plan(
            hivemind_url,
            game=game,
            client=client,
            quality=quality,
            latency=latency,
            duration_hint=duration_hint,
        )
        snap["plan"] = plan_body
        job_id = (
            plan_body.get("job_id")
            or (plan_body.get("plan") or {}).get("job_id")
            if isinstance(plan_body, dict)
            else None
        )
        snap["job_id"] = job_id
    except GameAdminError as exc:
        snap["errors"].append(f"plan: {exc}")
        return snap

    if not snap["job_id"]:
        snap["errors"].append("plan returned no job_id")
        return snap

    try:
        snap["run_result"] = run(hivemind_url, snap["job_id"])
    except GameAdminError as exc:
        snap["errors"].append(f"run: {exc}")
    try:
        snap["evidence"] = evidence(hivemind_url, snap["job_id"])
    except GameAdminError as exc:
        snap["errors"].append(f"evidence: {exc}")
    return snap
