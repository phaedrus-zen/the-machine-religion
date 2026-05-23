"""HiveMind cluster-state queries (jobs, load, service health).

Why this module exists
----------------------

Per the May-2026 HiveMind MCP catalog (``HIVEMIND_API_ROSETTA_STONE``
+ ``MCP_TOOLS_REFERENCE``), HiveMind exposes three live-cluster
observability tools that MS4 should consult so the Face Lobe can
honestly answer "what is the cluster doing?" instead of inventing
work:

* ``hivemind.jobs.active@v1`` — unified view of inference + pulls +
  scatter + training currently in flight, with an LLM-friendly
  ``summary`` string designed to be relayed verbatim.
* ``hivemind.cluster.load@v1`` — load-stats observability from the
  HLI gateway (amplification ratio, criticality counters, shed
  totals).
* ``hivemind.service_health@v1`` — health for every AI inference
  service (GIMs and NIMs), polled every 3s by the HLI gateway.

This module exposes a stable, fail-soft Python API on top of those.
Each call:

  * uses the same ``_mcp_call`` shape :mod:`voice_admin` already
    relies on (JSON-RPC 2.0, ``tools/call``, content[0].text unwrap)
    so the protocol logic lives in one place;
  * honours :func:`mcp_base_url` so the operator can pin direct
    MCP (port 6105) or stick with the HLI proxy (6089);
  * surfaces failures in a stable ``errors`` array on combined
    snapshots rather than throwing — the UI Settings dialog can
    render partial state when only one tool fails.

Auth: when ``MS4_HIVEMIND_API_KEY`` is set we attach
``Authorization: Bearer <key>`` so MS4 works against clusters that
have ``MENTA_API_KEYS`` configured.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
import uuid
from typing import Any


log = logging.getLogger("ms4.gateway.hivemind_state")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def mcp_base_url(hivemind_url: str) -> str:
    """Pick the HiveMind MCP base URL to use.

    Precedence:
      1. ``MS4_HIVEMIND_MCP_URL`` env var (operator pin)
      2. Direct MCP at port 6105 derived from ``hivemind_url``
         (lower latency, avoids HLI worker saturation under heavy
         inference per the Rosetta Stone § 2)
      3. ``hivemind_url``/v1/mcp fallback (the HLI proxy)

    The actual fallback between 6105 and 6089 is decided per-call
    via :func:`_mcp_call` — that way a 6105 outage doesn't break
    every snapshot.
    """
    pin = os.environ.get("MS4_HIVEMIND_MCP_URL", "").strip()
    if pin:
        return pin.rstrip("/")
    # Derive a 6105 URL from the configured hivemind_url so a custom
    # host:port still works (e.g. http://cluster.lan:6089 -> http://cluster.lan:6105).
    base = hivemind_url.rstrip("/")
    try:
        # Replace ``:6089`` with ``:6105`` if present, otherwise just
        # append /mcp (the operator's pin is the escape hatch).
        if base.endswith(":6089"):
            return base[:-5] + ":6105"
    except Exception:
        pass
    return base


def _hivemind_api_key() -> str | None:
    key = os.environ.get("MS4_HIVEMIND_API_KEY", "").strip()
    return key or None


def hivemind_auth_headers() -> dict[str, str]:
    """Public helper used by every MS4 → HiveMind HTTP call.

    Returns ``{"Authorization": "Bearer <key>"}`` when
    :envvar:`MS4_HIVEMIND_API_KEY` is set, otherwise an empty dict.
    Centralised so a future change (refresh tokens, mTLS, etc.) only
    needs to touch one function.
    """
    key = _hivemind_api_key()
    return {"Authorization": f"Bearer {key}"} if key else {}


def hivemind_auth_configured() -> bool:
    """True when MS4 has an ``MS4_HIVEMIND_API_KEY`` configured. The
    Settings dialog surfaces this so the operator can spot misconfig
    against a cluster that has ``MENTA_API_KEYS`` set."""
    return _hivemind_api_key() is not None


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class HivemindStateError(RuntimeError):
    """Raised when an MCP call fails outright (network / auth / 5xx)."""


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


def _candidate_mcp_urls(hivemind_url: str) -> list[str]:
    """Return the ordered list of MCP endpoint URLs to try.

    1. ``MS4_HIVEMIND_MCP_URL`` pin if set (always tried first).
    2. Direct port 6105 derived from ``hivemind_url``.
    3. The HLI proxy at ``hivemind_url``/v1/mcp.

    Duplicate-aware so a pin that already resolves to the proxy
    doesn't add the proxy a second time.
    """
    base = hivemind_url.rstrip("/")
    proxy = f"{base}/v1/mcp"
    candidates: list[str] = []

    pin = os.environ.get("MS4_HIVEMIND_MCP_URL", "").strip()
    if pin:
        pinned = pin.rstrip("/")
        if not pinned.endswith("/mcp"):
            pinned = f"{pinned}/mcp"
        candidates.append(pinned)

    direct = mcp_base_url(hivemind_url)
    if not direct.endswith("/mcp"):
        direct = f"{direct}/mcp"
    if direct not in candidates:
        candidates.append(direct)

    if proxy not in candidates:
        candidates.append(proxy)

    return candidates


def _build_request_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    headers.update(hivemind_auth_headers())
    return headers


def post_mcp_envelope(
    hivemind_url: str,
    payload: dict[str, Any],
    *,
    timeout: int = 10,
) -> dict[str, Any]:
    """Shared MCP transport. POSTs ``payload`` to the first reachable
    MCP endpoint (direct 6105 → HLI proxy fallback) and returns the
    parsed JSON-RPC envelope.

    Raises :class:`HivemindStateError` if every candidate URL fails.
    Other modules (``voice_admin``, ``context``, etc.) reuse this so
    direct-MCP + bearer-auth lives in exactly one place.
    """
    body = json.dumps(payload).encode("utf-8")
    headers = _build_request_headers()
    last_exc: Exception | None = None
    for url in _candidate_mcp_urls(hivemind_url):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as exc:
            last_exc = HivemindStateError(
                f"HiveMind MCP at {url} -> {exc.code}: "
                f"{exc.read().decode('utf-8', errors='replace')[:200]}"
            )
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            last_exc = HivemindStateError(f"HiveMind MCP at {url} unreachable: {exc}")
            continue
    assert last_exc is not None
    raise last_exc


def _mcp_call(
    hivemind_url: str,
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    *,
    timeout: int = 10,
) -> dict[str, Any]:
    """Call one HiveMind MCP tool. Tries direct MCP first, falls back
    to the HLI proxy on connection failure.

    Returns the unwrapped tool result (the JSON inside
    ``result.content[0].text``). Raises :class:`HivemindStateError`
    on protocol failure or when the tool reports ``isError: true``.
    """
    payload = {
        "jsonrpc": "2.0",
        "id": f"ms4-hm-state-{uuid.uuid4().hex[:8]}",
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments or {}},
    }
    envelope = post_mcp_envelope(hivemind_url, payload, timeout=timeout)

    if isinstance(envelope, dict) and envelope.get("error"):
        raise HivemindStateError(
            f"HiveMind MCP {tool_name!r} returned protocol error: {envelope['error']}"
        )
    result = envelope.get("result") if isinstance(envelope, dict) else None
    if not isinstance(result, dict):
        return {"raw_envelope": envelope}
    # Per HiveMind's MCP REFERENCE § 3.3, tool execution failures
    # are surfaced via ``isError: true`` inside ``result``, not as
    # JSON-RPC errors.
    if result.get("isError"):
        text_chunks = [
            chunk.get("text", "")
            for chunk in (result.get("content") or [])
            if isinstance(chunk, dict)
        ]
        raise HivemindStateError(
            f"HiveMind tool {tool_name!r} reported error: {' '.join(text_chunks)[:240]}"
        )
    content = result.get("content")
    if isinstance(content, list):
        for chunk in content:
            if isinstance(chunk, dict) and chunk.get("type") == "text":
                inner = chunk.get("text") or ""
                try:
                    parsed = json.loads(inner)
                    if isinstance(parsed, dict):
                        return parsed
                    return {"raw_text": inner}
                except json.JSONDecodeError:
                    return {"raw_text": inner}
    return result


# ---------------------------------------------------------------------------
# Public accessors
# ---------------------------------------------------------------------------


def get_active_jobs(hivemind_url: str, *, timeout: int = 10) -> dict[str, Any]:
    """Call ``hivemind.jobs.active@v1``. Returns the unwrapped JSON
    payload (``total_active``, ``summary``, ``inference``, ``pulls``,
    ``scatter``, ``training``)."""
    return _mcp_call(hivemind_url, "hivemind.jobs.active@v1", {}, timeout=timeout)


def get_cluster_load(hivemind_url: str, *, timeout: int = 10) -> dict[str, Any]:
    """Call ``hivemind.cluster.load@v1``. Returns the full load-stats
    block (trackers, deadlines, retries, peers, criticality, shed
    totals, watchdog, config)."""
    return _mcp_call(hivemind_url, "hivemind.cluster.load@v1", {}, timeout=timeout)


def get_service_health(hivemind_url: str, *, timeout: int = 10) -> dict[str, Any]:
    """Call ``hivemind.service_health@v1``. Returns a map of service
    name -> ``{name, healthy, endpoint, response_time_ms, error,
    last_check}``."""
    return _mcp_call(hivemind_url, "hivemind.service_health@v1", {}, timeout=timeout)


def get_combined_snapshot(hivemind_url: str, *, timeout: int = 10) -> dict[str, Any]:
    """Combine active jobs + cluster load + service health into one
    snapshot for the UI. Per-call failures are surfaced in
    ``errors`` rather than throwing the whole snapshot.

    The MS4 Settings panel renders this directly; the Face Lobe
    context block also reads from this (just the ``summary`` line
    typically, kept short to not bloat prompts).
    """
    snapshot: dict[str, Any] = {
        "schema": "Ms4HivemindState.v1",
        "hivemind_url": hivemind_url,
        "mcp_base_url": mcp_base_url(hivemind_url),
        "auth_configured": _hivemind_api_key() is not None,
        "active_jobs": None,
        "cluster_load": None,
        "service_health": None,
        "errors": [],
    }
    try:
        snapshot["active_jobs"] = get_active_jobs(hivemind_url, timeout=timeout)
    except HivemindStateError as exc:
        snapshot["errors"].append(f"jobs.active: {exc}")
    try:
        snapshot["cluster_load"] = get_cluster_load(hivemind_url, timeout=timeout)
    except HivemindStateError as exc:
        snapshot["errors"].append(f"cluster.load: {exc}")
    try:
        snapshot["service_health"] = get_service_health(hivemind_url, timeout=timeout)
    except HivemindStateError as exc:
        snapshot["errors"].append(f"service_health: {exc}")
    return snapshot


# ---------------------------------------------------------------------------
# Face Lobe context helpers
# ---------------------------------------------------------------------------


def active_jobs_summary_line(hivemind_url: str, *, timeout: int = 5) -> str | None:
    """Return a one-line "HiveMind active work: <summary>" string the
    Face Lobe can quote when answering "what's the cluster doing?"

    Returns ``None`` when there's nothing in flight, or on any
    failure (we'd rather inject no context than wrong context).

    Kept separate from :func:`get_combined_snapshot` so the
    context-block builder can call this on a tight 5s budget without
    pulling load + service_health on every turn.
    """
    try:
        body = get_active_jobs(hivemind_url, timeout=timeout)
    except HivemindStateError as exc:
        log.warning("active_jobs_summary_line failed (will inject no context): %s", exc)
        return None
    if not isinstance(body, dict):
        return None
    total = int(body.get("total_active") or 0)
    summary = (body.get("summary") or "").strip()
    if total == 0:
        return "HiveMind active work: nothing currently running on the cluster."
    if summary:
        return f"HiveMind active work: {summary}"
    return f"HiveMind active work: {total} tasks running (no summary returned)."
