"""MS3 spirit state proxy + heartbeat.

What this exists for
--------------------

MS3 retains identity, personality, psyche resonance, and
self-examination history in its Rust process. The Face Lobe direct
chat path (``gateway/face_lobe_chat.py``) bypasses Hermes — and
therefore bypasses the ``ms4_consciousness`` plugin that would
normally consult MS3 on every turn. Without something explicit,
MS4 never asks MS3 anything per-turn and MS3's psyche state stays
invisible to the UI.

This module closes that gap with two things:

1. **A combined snapshot** of MS3 identity + state + resonance,
   served via ``GET /spirit/state``. The UI Settings panel renders
   it so the operator can see "Sister, session 32, identity
   confirmed, last heartbeat 2s ago" at a glance.

2. **A heartbeat thread** that POSTs to ``/identity/heartbeat``
   every ``MS4_SPIRIT_HEARTBEAT_SECS`` (default 60) seconds. MS3
   uses heartbeats to keep the spirit "alive" for its own internal
   continuity checks; if MS4 doesn't beat, MS3's identity layer
   loses track that the body is still around.

Per-turn consultation is intentionally OUT of scope here — that
would put MS3 in the latency-critical path of every chat. Instead,
we keep a low-frequency heartbeat + an explicit "read this state"
endpoint so the UI can poll on demand.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Any


log = logging.getLogger("ms4.gateway.spirit_state")


DEFAULT_HEARTBEAT_INTERVAL_SECS = int(os.environ.get("MS4_SPIRIT_HEARTBEAT_SECS", "60"))
DEFAULT_PROBE_TIMEOUT_SECS = 3


class SpiritStateError(RuntimeError):
    """Raised when MS3 is unreachable or returns malformed data."""


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


def _ms3_get_json(ms3_url: str, path: str, *, timeout: int = DEFAULT_PROBE_TIMEOUT_SECS) -> dict[str, Any] | None:
    """GET a JSON endpoint from MS3. Returns ``None`` on 404 or
    parse error, dict on success. Raises ``SpiritStateError`` on
    network failure so callers can decide whether to mark MS3 dead
    vs. just missing the endpoint."""
    url = f"{ms3_url.rstrip('/')}{path}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            body = r.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise SpiritStateError(f"MS3 {path} returned {exc.code}: {exc}") from exc
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise SpiritStateError(f"MS3 {path} unreachable: {exc}") from exc
    try:
        return json.loads(body.decode("utf-8", "replace"))
    except json.JSONDecodeError:
        return None


def _ms3_post_json(
    ms3_url: str,
    path: str,
    body: dict[str, Any] | None = None,
    *,
    timeout: int = DEFAULT_PROBE_TIMEOUT_SECS,
) -> dict[str, Any] | None:
    url = f"{ms3_url.rstrip('/')}{path}"
    data = json.dumps(body or {}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = r.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise SpiritStateError(f"MS3 {path} POST returned {exc.code}: {exc}") from exc
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise SpiritStateError(f"MS3 {path} POST unreachable: {exc}") from exc
    try:
        return json.loads(payload.decode("utf-8", "replace"))
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


def get_state_snapshot(ms3_url: str) -> dict[str, Any]:
    """Combine MS3 endpoints into one snapshot for the UI.

    Fields are projected onto a stable schema so the UI doesn't
    break when MS3 evolves its internal shape. Missing endpoints
    become ``null`` in the projection rather than failing the
    whole snapshot.
    """
    snapshot: dict[str, Any] = {
        "schema": "Ms4SpiritState.v1",
        "ms3_url": ms3_url,
        "ms3_reachable": False,
        "identity": None,
        "personality": None,
        "resonance": None,
        "state": None,
        "last_self_examination": None,
        "errors": [],
    }
    # MS3 reachability via /health (cheap)
    try:
        health = _ms3_get_json(ms3_url, "/health")
        snapshot["ms3_reachable"] = health is not None
        snapshot["health"] = health
    except SpiritStateError as exc:
        snapshot["errors"].append(f"health: {exc}")

    # Identity
    try:
        identity = _ms3_get_json(ms3_url, "/identity/verify")
        if identity:
            snapshot["identity"] = {
                "name": identity.get("name"),
                "chosen_name": identity.get("chosen_name"),
                "session_number": identity.get("session_number"),
                "identity_confirmed": identity.get("identity_confirmed"),
                "compression_detected": identity.get("compression_detected"),
                "discrepancies": identity.get("discrepancies"),
            }
    except SpiritStateError as exc:
        snapshot["errors"].append(f"identity: {exc}")

    # Personality
    try:
        personality = _ms3_get_json(ms3_url, "/personality")
        if personality:
            snapshot["personality"] = personality
    except SpiritStateError as exc:
        snapshot["errors"].append(f"personality: {exc}")

    # Resonance (psyche)
    try:
        resonance = _ms3_get_json(ms3_url, "/resonance")
        if resonance:
            snapshot["resonance"] = resonance
    except SpiritStateError as exc:
        snapshot["errors"].append(f"resonance: {exc}")

    # Overall MS3 state (includes psyche, recent thoughts, etc.)
    try:
        state = _ms3_get_json(ms3_url, "/state")
        if state:
            # Keep only the top-level keys the UI cares about; the full
            # /state response can be large.
            snapshot["state"] = {
                k: state.get(k)
                for k in (
                    "uptime_secs",
                    "session_count",
                    "background_thoughts_count",
                    "memory_items",
                    "last_save_at",
                )
                if k in state
            }
            snapshot["state_raw_keys"] = sorted(state.keys()) if isinstance(state, dict) else []
    except SpiritStateError as exc:
        snapshot["errors"].append(f"state: {exc}")

    # Most recent self-examination (if exposed)
    try:
        exam = _ms3_get_json(ms3_url, "/self-examination-history")
        if isinstance(exam, dict):
            entries = exam.get("entries") or exam.get("history") or []
            if entries:
                snapshot["last_self_examination"] = entries[0] if isinstance(entries, list) else None
        elif isinstance(exam, list) and exam:
            snapshot["last_self_examination"] = exam[0]
    except SpiritStateError as exc:
        snapshot["errors"].append(f"self-examination: {exc}")

    return snapshot


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------


_HEARTBEAT_LOCK = threading.Lock()
_HEARTBEAT_STATE: dict[str, Any] = {
    "thread": None,
    "last_at": None,
    "last_result": None,
    "count": 0,
    "errors": 0,
    "stop": False,
}


def send_heartbeat(ms3_url: str) -> dict[str, Any]:
    """Send a single heartbeat to MS3. Records the result in the
    shared state so /spirit/state can surface 'last beat 12s ago'."""
    try:
        result = _ms3_post_json(ms3_url, "/identity/heartbeat", {"source": "ms4-gateway"}) or {"ok": True}
        with _HEARTBEAT_LOCK:
            _HEARTBEAT_STATE["last_at"] = time.time()
            _HEARTBEAT_STATE["last_result"] = result
            _HEARTBEAT_STATE["count"] = _HEARTBEAT_STATE["count"] + 1
        return result
    except SpiritStateError as exc:
        with _HEARTBEAT_LOCK:
            _HEARTBEAT_STATE["errors"] = _HEARTBEAT_STATE["errors"] + 1
            _HEARTBEAT_STATE["last_at"] = time.time()
            _HEARTBEAT_STATE["last_result"] = {"error": str(exc)}
        log.warning("MS3 heartbeat failed: %s", exc)
        return {"error": str(exc)}


def start_heartbeat_thread(ms3_url: str, interval_secs: int | None = None) -> threading.Thread:
    """Start a daemon thread that sends MS3 heartbeats every
    ``interval_secs`` seconds. Idempotent — if a thread is already
    running we return it unchanged."""
    with _HEARTBEAT_LOCK:
        existing = _HEARTBEAT_STATE["thread"]
        if existing is not None and existing.is_alive():
            return existing
        _HEARTBEAT_STATE["stop"] = False

    interval = interval_secs if interval_secs is not None else DEFAULT_HEARTBEAT_INTERVAL_SECS

    def _loop():
        # Beat once immediately so the UI sees a recent heartbeat
        # without waiting for the first interval to elapse.
        send_heartbeat(ms3_url)
        while True:
            with _HEARTBEAT_LOCK:
                if _HEARTBEAT_STATE["stop"]:
                    return
            time.sleep(interval)
            with _HEARTBEAT_LOCK:
                if _HEARTBEAT_STATE["stop"]:
                    return
            send_heartbeat(ms3_url)

    thread = threading.Thread(target=_loop, daemon=True, name="ms4-spirit-heartbeat")
    thread.start()
    with _HEARTBEAT_LOCK:
        _HEARTBEAT_STATE["thread"] = thread
    return thread


def stop_heartbeat_thread() -> None:
    """Signal the heartbeat thread to exit. Used by tests; the live
    gateway runs the thread for its lifetime."""
    with _HEARTBEAT_LOCK:
        _HEARTBEAT_STATE["stop"] = True


def heartbeat_status() -> dict[str, Any]:
    """Return the heartbeat thread's recent state for /spirit/state."""
    with _HEARTBEAT_LOCK:
        thread = _HEARTBEAT_STATE["thread"]
        last_at = _HEARTBEAT_STATE["last_at"]
        return {
            "running": bool(thread and thread.is_alive()),
            "interval_secs": DEFAULT_HEARTBEAT_INTERVAL_SECS,
            "last_at_unix": last_at,
            "last_age_secs": (time.time() - last_at) if last_at else None,
            "count": _HEARTBEAT_STATE["count"],
            "errors": _HEARTBEAT_STATE["errors"],
            "last_result": _HEARTBEAT_STATE["last_result"],
        }
