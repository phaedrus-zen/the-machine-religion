"""HiveMind voice service lifecycle admin.

What this module does
---------------------

MS4 used to surface ``ASR not ready: No endpoints configured`` as a
dead-end error. That error means HiveMind has the service in its
catalog but no node has been assigned to run it — and HiveMind has
all the APIs to fix that. This module drives those APIs so the UI
can present a single "Provision now" button instead of asking the
operator to ssh into HiveMind.

Surface used
~~~~~~~~~~~~

* ``GET  http://<hivemind>/provision/status/{ASR|TTS|TTS_SUPER}``
  Real-time health + provisioning state for a single voice service.

* ``POST http://<hivemind>/v1/mcp`` with ``hivemind.resources.request@v1``
  The named-resource API. Pass ``capability=asr|tts|tts_super`` and
  HiveMind picks an appropriate GPU/node and starts provisioning.
  Returns immediately with ``provision_id`` + ``poll_url``; the
  actual provisioning is async on HiveMind's side.

* ``POST http://<hivemind>/v1/mcp`` with ``hivemind.resources.release@v1``
  Release a provisioned resource (unload model / disable service).

We deliberately do NOT use HiveMind's older ``/edit-config`` shape.
Per live evidence and prior operator correction, the named-resource
API is the canonical way to drive cluster lifecycle.

Status shape
------------

``get_provision_status`` returns a stable dict (``Ms4VoiceServiceStatus.v1``)
that's safe to feed straight to the UI:

  {
    "schema": "Ms4VoiceServiceStatus.v1",
    "service": "ASR" | "TTS" | "TTS_SUPER",
    "healthy": bool,
    "detail": str,                       # e.g. "No endpoints configured", "running"
    "endpoint": str,                     # may be empty
    "endpoints_configured": int,
    "provisioning_state": str,           # configured_not_provisioned | provisioning | running | ...
    "raw": dict,                         # full HiveMind body for transparency
  }
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
import uuid
from typing import Any


log = logging.getLogger("ms4.gateway.voice_admin")


VOICE_SERVICES: tuple[str, ...] = ("ASR", "TTS", "TTS_SUPER")
# How the HiveMind ``capability`` argument maps to a service id we
# can poll with /provision/status/{SERVICE}. ``superskill:`` prefixes
# are how HiveMind names extended capabilities like TTS_SUPER.
CAPABILITY_TO_SERVICE: dict[str, str] = {
    "asr": "ASR",
    "tts": "TTS",
    "tts_super": "TTS_SUPER",
    "superskill:tts_super": "TTS_SUPER",
}
SERVICE_TO_CAPABILITY: dict[str, str] = {
    "ASR": "asr",
    "TTS": "tts",
    "TTS_SUPER": "superskill:tts_super",
}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class VoiceAdminError(RuntimeError):
    """Raised when HiveMind returns an unexpected error from a voice
    provisioning call. The gateway converts this into 502 / 503."""


class VoiceServiceUnknown(ValueError):
    """Raised when the caller asks about a service id that isn't a
    voice service we manage."""


# ---------------------------------------------------------------------------
# HTTP / MCP transport (stdlib so this module stays drop-in)
# ---------------------------------------------------------------------------


def _http_get_json(url: str, *, timeout: int = 10) -> tuple[int, dict[str, Any] | None]:
    # Delegate to the shared helper so every MS4 → HiveMind HTTP
    # call obeys MS4_HIVEMIND_API_KEY from exactly one place.
    from .hivemind_state import hivemind_auth_headers

    req = urllib.request.Request(url, headers=hivemind_auth_headers(), method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
            try:
                return r.status, json.loads(body.decode("utf-8", "replace"))
            except json.JSONDecodeError:
                return r.status, None
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read().decode("utf-8", "replace"))
        except Exception:
            return exc.code, None
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise VoiceAdminError(f"HiveMind {url} unreachable: {exc}") from exc


def _mcp_call(hivemind_url: str, tool_name: str, arguments: dict[str, Any], *, timeout: int = 30) -> dict[str, Any]:
    """Call a HiveMind MCP tool.

    Delegates to :func:`hivemind_state.post_mcp_envelope` so we get
    direct-MCP (port 6105) + HLI-proxy fallback + bearer auth for
    free. Returns the parsed JSON (the inner JSON-encoded content
    block, not the JSON-RPC envelope) when the tool returns
    structured data.
    """
    from .hivemind_state import HivemindStateError, post_mcp_envelope

    payload = {
        "jsonrpc": "2.0",
        "id": f"ms4-voice-admin-{uuid.uuid4().hex[:8]}",
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
    }
    try:
        envelope = post_mcp_envelope(hivemind_url, payload, timeout=timeout)
    except HivemindStateError as exc:
        raise VoiceAdminError(f"HiveMind MCP {tool_name!r} call failed: {exc}") from exc

    if isinstance(envelope, dict) and "error" in envelope and envelope["error"]:
        raise VoiceAdminError(f"HiveMind MCP {tool_name!r} returned error: {envelope['error']}")

    result = envelope.get("result") if isinstance(envelope, dict) else None
    if not isinstance(result, dict):
        return {"raw_envelope": envelope}
    content = result.get("content")
    if isinstance(content, list):
        for chunk in content:
            if isinstance(chunk, dict) and chunk.get("type") == "text" and isinstance(chunk.get("text"), str):
                inner = chunk["text"]
                try:
                    parsed = json.loads(inner)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    return {"text": inner}
    return result


# ---------------------------------------------------------------------------
# Status + listing
# ---------------------------------------------------------------------------


def _normalize_service_name(service: str) -> str:
    """Accept ``asr``, ``ASR``, ``tts_super``, ``TTS_SUPER`` etc. and
    return the canonical uppercase id. Raise VoiceServiceUnknown for
    anything else."""
    s = (service or "").strip().upper()
    if s not in VOICE_SERVICES:
        raise VoiceServiceUnknown(
            f"unknown voice service {service!r}; expected one of {VOICE_SERVICES}"
        )
    return s


def get_provision_status(hivemind_url: str, service: str, *, timeout: int = 10) -> dict[str, Any]:
    """Fetch /provision/status/{service} and project it onto the
    stable ``Ms4VoiceServiceStatus.v1`` shape."""
    svc = _normalize_service_name(service)
    status, body = _http_get_json(
        f"{hivemind_url.rstrip('/')}/provision/status/{svc}",
        timeout=timeout,
    )
    if body is None:
        body = {}
    service_health = body.get("service_health") if isinstance(body.get("service_health"), dict) else {}
    healthy = bool(service_health.get("healthy"))
    provisioning_state = str(
        service_health.get("provisioning_state")
        or body.get("status")
        or ("running" if healthy else "unknown")
    )
    return {
        "schema": "Ms4VoiceServiceStatus.v1",
        "service": svc,
        "healthy": healthy,
        "detail": str(body.get("detail") or service_health.get("last_error") or ""),
        "endpoint": str(body.get("endpoint") or ""),
        "endpoints_configured": int(body.get("endpoints_configured") or 0),
        "provisioning_state": provisioning_state,
        "http_status": status,
        "raw": body,
    }


def list_voice_services(hivemind_url: str, *, timeout: int = 10) -> list[dict[str, Any]]:
    """Status for every service in :data:`VOICE_SERVICES`, in order.

    The UI's voice banner + Settings dialog rendering depend on this
    being stable order so positions don't jitter between refreshes.
    """
    out: list[dict[str, Any]] = []
    for svc in VOICE_SERVICES:
        try:
            out.append(get_provision_status(hivemind_url, svc, timeout=timeout))
        except VoiceAdminError as exc:
            out.append({
                "schema": "Ms4VoiceServiceStatus.v1",
                "service": svc,
                "healthy": False,
                "detail": f"status fetch failed: {exc}",
                "endpoint": "",
                "endpoints_configured": 0,
                "provisioning_state": "unreachable",
                "http_status": 0,
                "raw": {},
            })
    return out


# ---------------------------------------------------------------------------
# Provisioning + release
# ---------------------------------------------------------------------------


def request_voice_service(
    hivemind_url: str,
    service: str,
    *,
    model: str | None = None,
    tier: str | None = None,
    mode: str | None = None,
    backend: str | None = None,
    timeout: int = 30,
    allow_during_maintenance: bool = False,
) -> dict[str, Any]:
    """Ask HiveMind to provision a voice capability.

    Returns immediately with ``status: 'provisioning'`` (or 'running'
    if the service was already up). The caller polls
    :func:`get_provision_status` to track the transition.

    Maintenance check (May 25 2026): if ``hivemind.service_health@v1``
    reports the service is in a maintenance window, this raises
    :class:`VoiceAdminError` unless the caller passes
    ``allow_during_maintenance=True``. The Settings UI sets that
    flag when the operator clicks "Provision anyway" on the
    maintenance pill. Best-effort: the check itself is fail-soft
    (an error querying maintenance state never blocks provisioning).
    """
    svc = _normalize_service_name(service)
    if not allow_during_maintenance:
        try:
            from . import hivemind_tools as _tools

            if _tools.is_service_in_maintenance(hivemind_url, svc):
                raise VoiceAdminError(
                    f"{svc} is in a HiveMind maintenance window; "
                    f"pass allow_during_maintenance=True to override"
                )
        except VoiceAdminError:
            raise
        except Exception as exc:
            log.warning("maintenance probe failed (ignoring): %s", exc)
    capability = SERVICE_TO_CAPABILITY[svc]
    arguments: dict[str, Any] = {"capability": capability}
    if model:
        arguments["model"] = model
    if tier:
        arguments["tier"] = tier
    if mode:
        arguments["mode"] = mode
    if backend:
        arguments["backend"] = backend
    body = _mcp_call(hivemind_url, "hivemind.resources.request@v1", arguments, timeout=timeout)
    body.setdefault("service", svc)
    body.setdefault("capability", capability)
    return body


def release_voice_service(
    hivemind_url: str,
    service: str,
    *,
    model: str | None = None,
    provision_id: str | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    """Release a previously-provisioned voice resource.

    For services this typically unloads the active endpoint. Pass
    ``provision_id`` (from the original request response) for the
    most precise release; ``model`` is accepted for symmetry with
    HiveMind's tool signature but services usually don't need it.
    """
    svc = _normalize_service_name(service)
    arguments: dict[str, Any] = {}
    if model:
        arguments["model"] = model
    if provision_id:
        arguments["provision_id"] = provision_id
    body = _mcp_call(hivemind_url, "hivemind.resources.release@v1", arguments, timeout=timeout)
    body.setdefault("service", svc)
    return body


def poll_until_healthy(
    hivemind_url: str,
    service: str,
    *,
    timeout_secs: float = 120.0,
    poll_interval_secs: float = 1.5,
) -> dict[str, Any]:
    """Block until /provision/status/{service} reports ``healthy:true``
    or ``timeout_secs`` elapses. Returns the final status snapshot
    either way; check ``status['healthy']`` to know which.

    Synchronous and intentionally simple — callers that want SSE
    progress should drive ``get_provision_status`` themselves on a
    timer thread.
    """
    deadline = time.monotonic() + timeout_secs
    status: dict[str, Any] = get_provision_status(hivemind_url, service)
    while not status["healthy"] and time.monotonic() < deadline:
        time.sleep(poll_interval_secs)
        try:
            status = get_provision_status(hivemind_url, service)
        except VoiceAdminError as exc:
            log.warning("poll_until_healthy(%s) error: %s", service, exc)
            continue
    return status
