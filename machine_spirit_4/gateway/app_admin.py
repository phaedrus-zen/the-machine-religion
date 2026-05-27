"""HiveMind App Registry admin (the ``hivemind.app.*@v1`` surface).

The App Registry on port 6110 lets HiveMind nodes register named
apps with versioned manifests and resource requirements. The May-2026
catalog exposes the whole lifecycle as MCP tools — MS4 wraps them so
the Settings panel can list running apps, surface metrics, and
start/stop them with the same UX pattern the voice-services panel
already uses.
"""

from __future__ import annotations

import logging
from typing import Any

from . import hivemind_tools as tools
from .hivemind_tools import HivemindToolError


log = logging.getLogger("ms4.gateway.app_admin")


class AppAdminError(RuntimeError):
    """Raised on a non-recoverable app admin failure."""


def _normalize_list(raw: Any, key: str = "apps") -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        cand = raw.get(key) or raw.get("data") or []
        if isinstance(cand, list):
            return [a for a in cand if isinstance(a, dict)]
    if isinstance(raw, list):
        return [a for a in raw if isinstance(a, dict)]
    return []


def list_apps(hivemind_url: str) -> list[dict[str, Any]]:
    """Local app registry contents (vs. ``discover_apps`` which is
    cluster-wide)."""
    try:
        return _normalize_list(tools.app_list(hivemind_url))
    except HivemindToolError as exc:
        raise AppAdminError(f"app.list failed: {exc}") from exc


def discover_apps(hivemind_url: str) -> list[dict[str, Any]]:
    """Cluster-wide app discovery via mDNS / registry mesh."""
    try:
        return _normalize_list(tools.apps_discover(hivemind_url))
    except HivemindToolError as exc:
        raise AppAdminError(f"apps.discover failed: {exc}") from exc


def get_app(hivemind_url: str, app_id: str) -> dict[str, Any]:
    try:
        raw = tools.app_get(hivemind_url, app_id)
    except HivemindToolError as exc:
        raise AppAdminError(f"app.get {app_id!r} failed: {exc}") from exc
    return raw if isinstance(raw, dict) else {"raw": raw}


def get_app_status(hivemind_url: str, app_id: str) -> dict[str, Any]:
    try:
        raw = tools.app_status(hivemind_url, app_id)
    except HivemindToolError as exc:
        raise AppAdminError(f"app.status {app_id!r} failed: {exc}") from exc
    return raw if isinstance(raw, dict) else {"raw": raw}


def get_app_metrics(hivemind_url: str, app_id: str) -> dict[str, Any]:
    try:
        raw = tools.app_metrics(hivemind_url, app_id)
    except HivemindToolError as exc:
        raise AppAdminError(f"app.metrics {app_id!r} failed: {exc}") from exc
    return raw if isinstance(raw, dict) else {"raw": raw}


def start_app(hivemind_url: str, app_id: str) -> dict[str, Any]:
    try:
        return tools.app_start(hivemind_url, app_id)
    except HivemindToolError as exc:
        raise AppAdminError(f"app.start {app_id!r} failed: {exc}") from exc


def stop_app(hivemind_url: str, app_id: str) -> dict[str, Any]:
    try:
        return tools.app_stop(hivemind_url, app_id)
    except HivemindToolError as exc:
        raise AppAdminError(f"app.stop {app_id!r} failed: {exc}") from exc


def resolve_app(hivemind_url: str, name: str) -> dict[str, Any]:
    """Look up an app_id by name. Useful for ``/apps/by-name/<name>``
    style URLs."""
    try:
        return tools.app_resolve(hivemind_url, name)
    except HivemindToolError as exc:
        raise AppAdminError(f"app.resolve {name!r} failed: {exc}") from exc


def register_app(hivemind_url: str, manifest: dict[str, Any]) -> dict[str, Any]:
    """Register a new app from a manifest dict."""
    try:
        return tools.app_register(hivemind_url, manifest)
    except HivemindToolError as exc:
        raise AppAdminError(f"app.register failed: {exc}") from exc


def list_with_status_and_metrics(hivemind_url: str) -> dict[str, Any]:
    """UI-friendly snapshot: list apps, then fan out status + metrics
    per app. Fail-soft per app — a single broken app doesn't fail
    the whole snapshot."""
    snapshot: dict[str, Any] = {
        "schema": "Ms4AppSnapshot.v1",
        "apps": [],
        "errors": [],
    }
    try:
        apps = list_apps(hivemind_url)
    except AppAdminError as exc:
        snapshot["errors"].append(f"app.list: {exc}")
        return snapshot
    for app in apps:
        app_id = str(app.get("id") or app.get("app_id") or "")
        if not app_id:
            snapshot["apps"].append({"app": app, "status": None, "metrics": None})
            continue
        entry: dict[str, Any] = {"app": app, "status": None, "metrics": None}
        try:
            entry["status"] = get_app_status(hivemind_url, app_id)
        except AppAdminError as exc:
            snapshot["errors"].append(f"app.status {app_id}: {exc}")
        try:
            entry["metrics"] = get_app_metrics(hivemind_url, app_id)
        except AppAdminError as exc:
            snapshot["errors"].append(f"app.metrics {app_id}: {exc}")
        snapshot["apps"].append(entry)
    return snapshot
