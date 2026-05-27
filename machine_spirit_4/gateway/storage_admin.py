"""HiveMind storage admin (``hivemind.storage.*@v1``).

Thin, typed feature surface over the storage tools added in the
May-2026 catalog. Backs MS4's Settings "Storage" panel and a small
REST surface so other agents driving MS4's MCP can list/create/delete
volumes and snapshots without learning HiveMind's MCP shape.

Mutations are guarded with explicit ``confirm`` flags (delete /
restore) so an accidental UI click can't wipe a volume.
"""

from __future__ import annotations

import logging
from typing import Any

from . import hivemind_tools as tools
from .hivemind_tools import HivemindToolError


log = logging.getLogger("ms4.gateway.storage_admin")


class StorageAdminError(RuntimeError):
    """Raised on a non-recoverable storage admin failure."""


def status(hivemind_url: str) -> dict[str, Any]:
    try:
        raw = tools.storage_status(hivemind_url)
    except HivemindToolError as exc:
        raise StorageAdminError(f"storage.status failed: {exc}") from exc
    return raw if isinstance(raw, dict) else {"raw": raw}


def list_pools(hivemind_url: str) -> list[dict[str, Any]]:
    try:
        raw = tools.storage_pools(hivemind_url)
    except HivemindToolError as exc:
        raise StorageAdminError(f"storage.pools failed: {exc}") from exc
    if isinstance(raw, dict):
        return [p for p in (raw.get("pools") or raw.get("data") or []) if isinstance(p, dict)]
    if isinstance(raw, list):
        return [p for p in raw if isinstance(p, dict)]
    return []


def list_volumes(hivemind_url: str, pool_id: str | None = None) -> list[dict[str, Any]]:
    try:
        raw = tools.storage_volumes(hivemind_url, pool_id)
    except HivemindToolError as exc:
        raise StorageAdminError(f"storage.volumes failed: {exc}") from exc
    if isinstance(raw, dict):
        return [v for v in (raw.get("volumes") or raw.get("data") or []) if isinstance(v, dict)]
    if isinstance(raw, list):
        return [v for v in raw if isinstance(v, dict)]
    return []


def list_snapshots(hivemind_url: str, volume_id: str | None = None) -> list[dict[str, Any]]:
    try:
        raw = tools.storage_snapshots(hivemind_url, volume_id)
    except HivemindToolError as exc:
        raise StorageAdminError(f"storage.snapshots failed: {exc}") from exc
    if isinstance(raw, dict):
        return [
            s for s in (raw.get("snapshots") or raw.get("data") or []) if isinstance(s, dict)
        ]
    if isinstance(raw, list):
        return [s for s in raw if isinstance(s, dict)]
    return []


def combined_snapshot(hivemind_url: str) -> dict[str, Any]:
    """Status + pools + volumes + snapshots in one snapshot for the
    Settings panel. Per-call failures land in ``errors``."""
    snap: dict[str, Any] = {
        "schema": "Ms4StorageSnapshot.v1",
        "status": None,
        "pools": [],
        "volumes": [],
        "snapshots": [],
        "errors": [],
    }
    try:
        snap["status"] = status(hivemind_url)
    except StorageAdminError as exc:
        snap["errors"].append(f"status: {exc}")
    try:
        snap["pools"] = list_pools(hivemind_url)
    except StorageAdminError as exc:
        snap["errors"].append(f"pools: {exc}")
    try:
        snap["volumes"] = list_volumes(hivemind_url)
    except StorageAdminError as exc:
        snap["errors"].append(f"volumes: {exc}")
    try:
        snap["snapshots"] = list_snapshots(hivemind_url)
    except StorageAdminError as exc:
        snap["errors"].append(f"snapshots: {exc}")
    return snap


# ---- Mutations ----------------------------------------------------------


def create_volume(hivemind_url: str, **opts: Any) -> dict[str, Any]:
    try:
        return tools.storage_volume_create(hivemind_url, **opts)
    except HivemindToolError as exc:
        raise StorageAdminError(f"volume_create failed: {exc}") from exc


def delete_volume(hivemind_url: str, volume_id: str, *, confirm: bool) -> dict[str, Any]:
    if not confirm:
        raise ValueError("delete_volume requires confirm=True (irreversible)")
    try:
        return tools.storage_volume_delete(hivemind_url, volume_id)
    except HivemindToolError as exc:
        raise StorageAdminError(f"volume_delete {volume_id!r} failed: {exc}") from exc


def attach_volume(hivemind_url: str, volume_id: str, target: str) -> dict[str, Any]:
    try:
        return tools.storage_volume_attach(hivemind_url, volume_id, target)
    except HivemindToolError as exc:
        raise StorageAdminError(f"volume_attach {volume_id!r}->{target!r} failed: {exc}") from exc


def detach_volume(hivemind_url: str, volume_id: str) -> dict[str, Any]:
    try:
        return tools.storage_volume_detach(hivemind_url, volume_id)
    except HivemindToolError as exc:
        raise StorageAdminError(f"volume_detach {volume_id!r} failed: {exc}") from exc


def resize_volume(hivemind_url: str, volume_id: str, new_size_bytes: int) -> dict[str, Any]:
    if new_size_bytes <= 0:
        raise ValueError("new_size_bytes must be positive")
    try:
        return tools.storage_volume_resize(hivemind_url, volume_id, new_size_bytes)
    except HivemindToolError as exc:
        raise StorageAdminError(f"volume_resize {volume_id!r} failed: {exc}") from exc


def create_snapshot(hivemind_url: str, volume_id: str, label: str | None = None) -> dict[str, Any]:
    try:
        return tools.storage_snapshot_create(hivemind_url, volume_id, label)
    except HivemindToolError as exc:
        raise StorageAdminError(f"snapshot_create {volume_id!r} failed: {exc}") from exc


def delete_snapshot(hivemind_url: str, snapshot_id: str, *, confirm: bool) -> dict[str, Any]:
    if not confirm:
        raise ValueError("delete_snapshot requires confirm=True (irreversible)")
    try:
        return tools.storage_snapshot_delete(hivemind_url, snapshot_id)
    except HivemindToolError as exc:
        raise StorageAdminError(f"snapshot_delete {snapshot_id!r} failed: {exc}") from exc


def restore_snapshot(hivemind_url: str, snapshot_id: str, *, confirm: bool) -> dict[str, Any]:
    """Roll a volume back to a snapshot. Requires ``confirm=True`` —
    data written after the snapshot is discarded."""
    if not confirm:
        raise ValueError("restore_snapshot requires confirm=True (discards post-snapshot data)")
    try:
        return tools.storage_snapshot_restore(hivemind_url, snapshot_id)
    except HivemindToolError as exc:
        raise StorageAdminError(f"snapshot_restore {snapshot_id!r} failed: {exc}") from exc
