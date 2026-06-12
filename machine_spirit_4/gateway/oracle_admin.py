"""HiveMind Oracle planner admin.

What this module owns
---------------------

Wraps the ``hivemind.oracle.*`` tool family (planner / coordinator
running at port 6089's Oracle route + the dedicated MCP tools). The
Oracle is HiveMind's reasoning surface: given a goal, it plans steps,
checks cluster capacity, and chooses tools. MS4 exposes it so the
operator can:

  * see Oracle's current state (`status`)
  * push runtime config (`configure`)
  * ask Oracle to plan a task (`chat`)

This is intentionally a thin pass-through. MS4 doesn't substitute its
own planner; if Oracle is up, MS4 surfaces it. If Oracle is down or
disabled in the cluster config, the per-method calls raise
:class:`OracleAdminError` and the UI/REST routes degrade gracefully
to "Oracle unavailable" rather than blocking other functionality.
"""

from __future__ import annotations

import logging
from typing import Any

from . import hivemind_tools as tools
from . import hivemind_state
from .hivemind_tools import HivemindToolError


log = logging.getLogger("ms4.gateway.oracle_admin")


class OracleAdminError(RuntimeError):
    """Raised on non-recoverable Oracle admin failure."""


def _state(value: Any) -> str:
    return str(value or "").strip()


def _healthy(value: Any) -> bool | None:
    if not isinstance(value, dict):
        return None
    if "healthy" in value:
        return bool(value.get("healthy"))
    if "ok" in value:
        return bool(value.get("ok"))
    if "available" in value:
        return bool(value.get("available"))
    status_value = _state(value.get("status") or value.get("state")).lower()
    if status_value in {"ready", "running", "healthy", "ok", "online", "enabled"}:
        return True
    if status_value in {"blocked", "disabled", "error", "failed", "unhealthy", "offline"}:
        return False
    return None


def _service_name(key: str, body: Any) -> str:
    if isinstance(body, dict):
        return _state(body.get("name") or body.get("service") or key)
    return key


def _oracle_relevant_services(service_health: Any) -> list[dict[str, Any]]:
    if not isinstance(service_health, dict):
        return []
    out: list[dict[str, Any]] = []
    for key, body in service_health.items():
        name = _service_name(str(key), body)
        low = name.lower()
        if not any(token in low for token in ("oracle", "hli", "mcp")):
            continue
        entry = body.copy() if isinstance(body, dict) else {"raw": body}
        entry.setdefault("name", name)
        entry["key"] = str(key)
        out.append(entry)
    return out


def readiness(hivemind_url: str, *, timeout: int = 5) -> dict[str, Any]:
    """Read-only Oracle operator readiness snapshot.

    This is intentionally not an auto-provisioning entry point. It gives
    the UI a single, stable preflight explaining whether Oracle is usable,
    which prerequisites look unhealthy, and which follow-up actions cross
    into approval-gated mutation territory.
    """
    errors: list[str] = []
    checks: list[dict[str, Any]] = []
    blockers: list[str] = []
    next_actions: list[dict[str, Any]] = []

    oracle_snapshot: dict[str, Any] | None = None
    try:
        oracle_snapshot = status(hivemind_url, timeout=timeout)
    except OracleAdminError as exc:
        err = str(exc)
        errors.append(err)
        oracle_snapshot = {"schema": "Ms4OracleSnapshot.v1", "available": False, "error": err}

    oracle_health = _healthy(oracle_snapshot)
    if oracle_health is True:
        checks.append({"name": "oracle_status", "state": "pass", "detail": "Oracle status responded healthy."})
    elif oracle_health is False:
        detail = _state((oracle_snapshot or {}).get("error") or (oracle_snapshot or {}).get("status"))
        checks.append({"name": "oracle_status", "state": "fail", "detail": detail or "Oracle reported unhealthy."})
        blockers.append("Oracle status is unavailable or unhealthy.")
    else:
        checks.append({"name": "oracle_status", "state": "warn", "detail": "Oracle status did not include health."})

    cluster = hivemind_state.get_combined_snapshot(hivemind_url, timeout=timeout)
    cluster_errors = [str(e) for e in cluster.get("errors", [])] if isinstance(cluster, dict) else []
    if cluster_errors:
        errors.extend(cluster_errors)
        checks.append({
            "name": "hivemind_state",
            "state": "warn",
            "detail": "; ".join(cluster_errors[:3]),
        })
    else:
        checks.append({"name": "hivemind_state", "state": "pass", "detail": "Cluster state snapshot available."})

    relevant_services = _oracle_relevant_services(cluster.get("service_health") if isinstance(cluster, dict) else None)
    unhealthy_services: list[str] = []
    for service in relevant_services:
        service_health = _healthy(service)
        if service_health is False:
            name = _state(service.get("name") or service.get("key"))
            reason = _state(service.get("error") or service.get("status") or service.get("state"))
            unhealthy_services.append(f"{name}: {reason or 'unhealthy'}")
    if unhealthy_services:
        checks.append({
            "name": "oracle_service_health",
            "state": "fail",
            "detail": "; ".join(unhealthy_services[:4]),
        })
        blockers.extend(unhealthy_services)
    elif relevant_services:
        checks.append({
            "name": "oracle_service_health",
            "state": "pass",
            "detail": f"{len(relevant_services)} relevant service(s) healthy or nonblocking.",
        })
    else:
        checks.append({
            "name": "oracle_service_health",
            "state": "warn",
            "detail": "No Oracle/HLI/MCP service-health entries were returned.",
        })

    if not hivemind_state.hivemind_auth_configured():
        next_actions.append({
            "label": "Set MS4_HIVEMIND_API_KEY if this cluster requires bearer auth.",
            "risk": "config",
            "requires_approval": False,
        })
    if blockers:
        readiness_state = "blocked"
    elif cluster_errors or oracle_health is not True:
        readiness_state = "degraded"
    else:
        readiness_state = "ready"

    if readiness_state == "blocked":
        next_actions.append({
            "label": "Provision, enable, or restart missing Oracle prerequisites through approved HiveMind service routes.",
            "risk": "mutating",
            "requires_approval": True,
        })
    elif readiness_state == "degraded":
        next_actions.append({
            "label": "Resolve readiness warnings before relying on Oracle chat.",
            "risk": "diagnostic",
            "requires_approval": False,
        })
    else:
        next_actions.append({
            "label": "Oracle chat is safe to use from MS4.",
            "risk": "read_only",
            "requires_approval": False,
        })

    return {
        "schema": "Ms4OracleReadiness.v1",
        "hivemind_url": hivemind_url,
        "mcp_base_url": cluster.get("mcp_base_url") if isinstance(cluster, dict) else None,
        "auth_configured": hivemind_state.hivemind_auth_configured(),
        "ready": readiness_state == "ready",
        "readiness": readiness_state,
        "oracle": oracle_snapshot,
        "checks": checks,
        "blockers": blockers,
        "next_actions": next_actions,
        "provisioning": {
            "automatic": False,
            "state": (
                "approval_required"
                if readiness_state == "blocked"
                else "diagnostic_required"
                if readiness_state == "degraded"
                else "not_needed"
            ),
            "reason": (
                "MS4 readiness is read-only; enable/restart/provision calls are mutating "
                "and stay behind explicit approval."
            ),
            "mutating_route_hints": [
                "/hivemind/services/{service}/enable",
                "/hivemind/services/{service}/restart",
                "/voice/services/{service}/provision",
            ],
        },
        "cluster": {
            "active_jobs": cluster.get("active_jobs") if isinstance(cluster, dict) else None,
            "cluster_load": cluster.get("cluster_load") if isinstance(cluster, dict) else None,
            "service_health": relevant_services,
            "errors": cluster_errors,
        },
        "errors": errors,
    }


def status(hivemind_url: str, *, timeout: int = 15) -> dict[str, Any]:
    """Oracle planner state. Schema ``Ms4OracleSnapshot.v1``."""
    try:
        raw = tools.oracle_status(hivemind_url, timeout=timeout)
    except HivemindToolError as exc:
        raise OracleAdminError(f"oracle.status failed: {exc}") from exc
    body = raw if isinstance(raw, dict) else {"raw": raw}
    return {"schema": "Ms4OracleSnapshot.v1", **body}


def configure(hivemind_url: str, config: dict[str, Any]) -> dict[str, Any]:
    """Push runtime config to the Oracle planner."""
    if not isinstance(config, dict):
        raise ValueError("config must be an object")
    try:
        raw = tools.oracle_configure(hivemind_url, config=config)
    except HivemindToolError as exc:
        raise OracleAdminError(f"oracle.configure failed: {exc}") from exc
    return raw if isinstance(raw, dict) else {"raw": raw}


def chat(hivemind_url: str, message: str, **opts: Any) -> dict[str, Any]:
    """Ask Oracle to reason about / plan for ``message``.

    Used by MS4 operator workflows like "what should I run next on the
    cluster?". The Face Lobe can also dispatch to Oracle for planning
    when its router decides a question is best handled there rather
    than by Hermes."""
    if not message or not isinstance(message, str):
        raise ValueError("message is required")
    try:
        raw = tools.oracle_chat(hivemind_url, message=message, **opts)
    except HivemindToolError as exc:
        raise OracleAdminError(f"oracle.chat failed: {exc}") from exc
    return raw if isinstance(raw, dict) else {"raw": raw}
