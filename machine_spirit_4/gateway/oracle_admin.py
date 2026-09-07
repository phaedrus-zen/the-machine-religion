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


def _service_health_entries(service_health: Any) -> list[tuple[str, Any]]:
    """Return service entries from both supported health envelopes.

    Older HiveMind builds returned ``{service_name: status}`` directly.
    Current builds wrap that map under ``service_health.services`` and keep
    envelope metadata beside it. Readiness must inspect the actual service
    map in either shape instead of treating a healthy envelope as healthy
    ASR.
    """
    if not isinstance(service_health, dict):
        return []

    entries: list[tuple[str, Any]] = []
    nested = service_health.get("services")
    for key, body in service_health.items():
        if key == "services" and isinstance(nested, dict):
            continue
        entries.append((str(key), body))
    if isinstance(nested, dict):
        entries.extend((f"services.{key}", body) for key, body in nested.items())
    return entries


def _oracle_relevant_services(service_health: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key, body in _service_health_entries(service_health):
        leaf_key = key.rsplit(".", 1)[-1]
        name = _service_name(leaf_key, body)
        low = name.lower()
        if not any(token in low for token in ("oracle", "hli", "mcp")):
            continue
        entry = body.copy() if isinstance(body, dict) else {"raw": body}
        entry.setdefault("name", name)
        entry["key"] = key
        out.append(entry)
    return out


_VOICE_INPUT_UNAVAILABLE_STATES = {
    "configured_not_provisioned",
    "not_provisioned",
    "not_ready",
    "unavailable",
    "unreachable",
    "disabled",
    "error",
    "failed",
    "unhealthy",
    "offline",
    "stopped",
}


def _voice_input_services(service_health: Any) -> list[dict[str, Any]]:
    """Return service-health entries that can accept Oracle voice input."""
    out: list[dict[str, Any]] = []
    for key, body in _service_health_entries(service_health):
        leaf_key = key.rsplit(".", 1)[-1]
        name = _service_name(leaf_key, body)
        identifiers = {leaf_key.lower(), name.lower()}
        normalized = {
            value.replace("-", "_").replace(" ", "_")
            for value in identifiers
        }
        body_has_voice_flag = isinstance(body, dict) and "voice_input_ready" in body
        is_voice_input = body_has_voice_flag or any(
            "voice_input" in value
            or "speech_to_text" in value
            or "transcrib" in value
            or "asr" in value.split("_")
            or value.startswith("asr")
            for value in normalized
        )
        if not is_voice_input:
            continue
        entry = body.copy() if isinstance(body, dict) else {"raw": body}
        entry.setdefault("name", name)
        entry["key"] = key
        out.append(entry)
    return out


def _voice_input_issue(service: dict[str, Any]) -> str | None:
    state = _state(
        service.get("provisioning_state")
        or service.get("status")
        or service.get("state")
    ).lower()
    voice_ready = service.get("voice_input_ready")
    unavailable = (
        voice_ready is False
        or _healthy(service) is False
        or state in _VOICE_INPUT_UNAVAILABLE_STATES
        or state.endswith("not_provisioned")
    )
    if not unavailable:
        return None
    name = _state(service.get("name") or service.get("key")) or "ASR"
    detail = _state(
        service.get("detail")
        or service.get("error")
        or service.get("last_error")
        or state
    )
    return f"{name}: {detail or 'voice input unavailable'}"


def _chat_plane_from_lifecycle(lifecycle: Any) -> dict[str, Any]:
    """Interpret a HiveMind ``lifecycle_state`` snapshot for the Oracle chat plane.

    Oracle chat is only safe when HiveMind's own authoritative lifecycle
    view says the AI plane can serve a chat turn. This condenses that
    snapshot into a stable, concise block:

    ``status``:
      * ``"ready"``   — valid evidence AND a chat capability is up
        (``ai_plane_ready`` true and ``chat`` not marked unavailable);
      * ``"down"``    — valid evidence but chat is explicitly unavailable
        (``ai_plane_ready`` false / ``chat`` in ``capabilities_unavailable``);
      * ``"unknown"`` — evidence is missing or not the expected shape
        (no ``ai_plane_ready`` boolean), so readiness must fail soft.

    ``ai_plane_ready`` on the wire is defined as "``chat`` in
    ``capabilities_available``"; we still cross-check the capability lists
    so a future server divergence can't silently pass the gate.
    """
    unknown = {
        "status": "unknown",
        "ai_plane_ready": None,
        "chat_available": None,
        "loaded_models_count": None,
        "boot_stage": None,
        "workload_phase": None,
        "source": None,
        "detail": "No lifecycle_state evidence returned.",
    }
    if not isinstance(lifecycle, dict):
        return unknown

    signals = lifecycle.get("signals")
    signals = signals if isinstance(signals, dict) else {}
    loaded_models_count = signals.get("loaded_models_count")
    boot_stage = _state(lifecycle.get("boot_stage")) or None
    workload_phase = _state(lifecycle.get("workload_phase")) or None
    source = _state(lifecycle.get("source")) or None

    ai_plane_ready = lifecycle.get("ai_plane_ready")
    if not isinstance(ai_plane_ready, bool):
        return {
            **unknown,
            "loaded_models_count": loaded_models_count,
            "boot_stage": boot_stage,
            "workload_phase": workload_phase,
            "source": source,
            "detail": "lifecycle_state missing ai_plane_ready flag.",
        }

    caps_unavailable = lifecycle.get("capabilities_unavailable")
    caps_available = lifecycle.get("capabilities_available")
    caps_unavailable = caps_unavailable if isinstance(caps_unavailable, list) else []
    caps_available = caps_available if isinstance(caps_available, list) else []
    chat_available = bool(
        ai_plane_ready
        and "chat" not in caps_unavailable
        and (not caps_available or "chat" in caps_available)
    )

    if chat_available:
        detail = "HiveMind chat plane is ready."
    else:
        bits = [f"ai_plane_ready={str(ai_plane_ready).lower()}"]
        if "chat" in caps_unavailable or (caps_available and "chat" not in caps_available):
            bits.append("chat capability unavailable")
        if isinstance(loaded_models_count, int):
            bits.append(f"{loaded_models_count} model(s) loaded")
        detail = "HiveMind chat plane is not ready: " + ", ".join(bits) + "."

    return {
        "status": "ready" if chat_available else "down",
        "ai_plane_ready": ai_plane_ready,
        "chat_available": chat_available,
        "loaded_models_count": loaded_models_count,
        "boot_stage": boot_stage,
        "workload_phase": workload_phase,
        "source": source,
        "detail": detail,
    }


def readiness(hivemind_url: str, *, timeout: int = 5) -> dict[str, Any]:
    """Read-only Oracle operator readiness snapshot.

    This is intentionally not an auto-provisioning entry point. It gives
    the UI a single, stable preflight explaining whether Oracle is usable,
    which prerequisites look unhealthy, and which follow-up actions cross
    into approval-gated mutation territory.

    Oracle chat readiness fails closed against HiveMind's authoritative
    ``/api/v1/cluster/lifecycle_state`` chat plane: an explicit down chat
    plane is a ``blocked`` blocker, and missing/invalid lifecycle evidence
    forces at least ``degraded`` — MS4 must never report ``ready`` when the
    cluster cannot actually serve a chat turn.
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

    voice_input_services = _voice_input_services(
        cluster.get("service_health") if isinstance(cluster, dict) else None
    )
    voice_input_service_issues = [
        _voice_input_issue(service) for service in voice_input_services
    ]
    voice_input_available_count = sum(
        issue is None for issue in voice_input_service_issues
    )
    unavailable_voice_input_alternates = [
        issue for issue in voice_input_service_issues if issue is not None
    ]
    # ASR and ASR_SUPER are alternative providers, not cumulative
    # prerequisites. One healthy input path is sufficient for Oracle voice.
    voice_input_issues = (
        [] if voice_input_available_count else unavailable_voice_input_alternates
    )
    if voice_input_issues:
        checks.append({
            "name": "voice_input",
            "state": "fail",
            "detail": "; ".join(voice_input_issues[:4]),
        })
    elif voice_input_available_count:
        checks.append({
            "name": "voice_input",
            "state": "pass",
            "detail": (
                f"{voice_input_available_count} of {len(voice_input_services)} "
                "ASR/voice-input service(s) available."
            ),
        })
    else:
        checks.append({
            "name": "voice_input",
            "state": "warn",
            "detail": "No ASR/voice-input service-health entry was returned.",
        })

    # Chat-plane truth gate. Oracle chat cannot be "ready" unless HiveMind's
    # authoritative lifecycle_state confirms the AI plane can serve a chat
    # turn. An explicit down plane blocks; missing/invalid evidence degrades.
    chat_plane_error: str | None = None
    lifecycle_snapshot: dict[str, Any] | None = None
    try:
        lifecycle_snapshot = hivemind_state.get_lifecycle_state(hivemind_url, timeout=timeout)
    except hivemind_state.HivemindStateError as exc:
        chat_plane_error = str(exc)
        errors.append(f"lifecycle_state: {exc}")
    chat_plane = _chat_plane_from_lifecycle(lifecycle_snapshot)
    if chat_plane_error and chat_plane["status"] == "unknown":
        chat_plane["detail"] = f"lifecycle_state unavailable: {chat_plane_error}"
    chat_plane_down = chat_plane["status"] == "down"
    chat_plane_unknown = chat_plane["status"] == "unknown"
    if chat_plane_down:
        checks.append({"name": "chat_plane", "state": "fail", "detail": chat_plane["detail"]})
        blockers.append(chat_plane["detail"])
    elif chat_plane_unknown:
        checks.append({"name": "chat_plane", "state": "warn", "detail": chat_plane["detail"]})
    else:
        checks.append({"name": "chat_plane", "state": "pass", "detail": chat_plane["detail"]})

    if not hivemind_state.hivemind_auth_configured():
        next_actions.append({
            "label": "Set MS4_HIVEMIND_API_KEY if this cluster requires bearer auth.",
            "risk": "config",
            "requires_approval": False,
        })
    if blockers:
        readiness_state = "blocked"
    elif chat_plane_unknown or cluster_errors or oracle_health is not True:
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
        "voice_input": {
            "ready": (
                True
                if voice_input_available_count
                else False
                if voice_input_services
                else None
            ),
            "services": voice_input_services,
            "issues": voice_input_issues,
            "unavailable_alternates": (
                unavailable_voice_input_alternates
                if voice_input_available_count
                else []
            ),
        },
        "output_delivery": {
            "delivered": False,
            "source": "not-claimed-from-input-or-readiness",
            "detail": "Oracle chat readiness and voice input readiness do not imply audible output delivery.",
        },
        "chat_plane": {
            "status": chat_plane["status"],
            "ai_plane_ready": chat_plane["ai_plane_ready"],
            "chat_available": chat_plane["chat_available"],
            "loaded_models_count": chat_plane["loaded_models_count"],
            "boot_stage": chat_plane["boot_stage"],
            "workload_phase": chat_plane["workload_phase"],
            "source": chat_plane["source"],
            "detail": chat_plane["detail"],
            "error": chat_plane_error,
        },
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
    snapshot = {"schema": "Ms4OracleSnapshot.v1", **body}
    if _healthy(snapshot) is None:
        has_error = bool(snapshot.get("error") or snapshot.get("last_error"))
        snapshot["healthy"] = not has_error
    if not snapshot.get("state"):
        snapshot["state"] = "ready" if _healthy(snapshot) is True else "degraded"
    return snapshot


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
