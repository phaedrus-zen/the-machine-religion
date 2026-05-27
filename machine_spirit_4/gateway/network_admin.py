"""HiveMind network admin (``hivemind.network.*@v1``).

Wraps the 10 network tools added in the May-2026 catalog: lists,
creates, deletes, attaches, detaches, bridges, interfaces, isolation
toggle, and per-network status. Used by the Settings panel's
"Networks" tab and the MS4 MCP proxy tools.
"""

from __future__ import annotations

import logging
from typing import Any

from . import hivemind_tools as tools
from .hivemind_tools import HivemindToolError


log = logging.getLogger("ms4.gateway.network_admin")


class NetworkAdminError(RuntimeError):
    pass


def _list(raw: Any, key: str) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        cand = raw.get(key) or raw.get("data") or []
        return [n for n in cand if isinstance(n, dict)]
    if isinstance(raw, list):
        return [n for n in raw if isinstance(n, dict)]
    return []


def list_networks(hivemind_url: str) -> list[dict[str, Any]]:
    try:
        return _list(tools.network_list(hivemind_url), "networks")
    except HivemindToolError as exc:
        raise NetworkAdminError(f"network.list failed: {exc}") from exc


def network_status(hivemind_url: str, network_id: str | None = None) -> Any:
    try:
        return tools.network_status(hivemind_url, network_id)
    except HivemindToolError as exc:
        raise NetworkAdminError(f"network.status failed: {exc}") from exc


def list_bridges(hivemind_url: str) -> list[dict[str, Any]]:
    try:
        return _list(tools.network_bridges(hivemind_url), "bridges")
    except HivemindToolError as exc:
        raise NetworkAdminError(f"network.bridges failed: {exc}") from exc


def list_interfaces(hivemind_url: str) -> list[dict[str, Any]]:
    try:
        return _list(tools.network_interfaces(hivemind_url), "interfaces")
    except HivemindToolError as exc:
        raise NetworkAdminError(f"network.interfaces failed: {exc}") from exc


def list_attachments(hivemind_url: str, network_id: str | None = None) -> list[dict[str, Any]]:
    try:
        return _list(tools.network_attachments(hivemind_url, network_id), "attachments")
    except HivemindToolError as exc:
        raise NetworkAdminError(f"network.attachments failed: {exc}") from exc


def create_network(hivemind_url: str, **opts: Any) -> dict[str, Any]:
    try:
        return tools.network_create(hivemind_url, **opts)
    except HivemindToolError as exc:
        raise NetworkAdminError(f"network.create failed: {exc}") from exc


def delete_network(hivemind_url: str, network_id: str, *, confirm: bool) -> dict[str, Any]:
    if not confirm:
        raise ValueError("delete_network requires confirm=True (severs attachments)")
    try:
        return tools.network_delete(hivemind_url, network_id)
    except HivemindToolError as exc:
        raise NetworkAdminError(f"network.delete {network_id!r} failed: {exc}") from exc


def attach(hivemind_url: str, network_id: str, target: str) -> dict[str, Any]:
    try:
        return tools.network_attach(hivemind_url, network_id, target)
    except HivemindToolError as exc:
        raise NetworkAdminError(f"network.attach {network_id!r}->{target!r} failed: {exc}") from exc


def detach(hivemind_url: str, network_id: str, target: str) -> dict[str, Any]:
    try:
        return tools.network_detach(hivemind_url, network_id, target)
    except HivemindToolError as exc:
        raise NetworkAdminError(f"network.detach {network_id!r}->{target!r} failed: {exc}") from exc


def set_isolation(hivemind_url: str, network_id: str, isolated: bool) -> dict[str, Any]:
    try:
        return tools.network_isolate(hivemind_url, network_id, isolated)
    except HivemindToolError as exc:
        raise NetworkAdminError(f"network.isolate {network_id!r} failed: {exc}") from exc


def combined_snapshot(hivemind_url: str) -> dict[str, Any]:
    snap: dict[str, Any] = {
        "schema": "Ms4NetworkSnapshot.v1",
        "networks": [],
        "bridges": [],
        "interfaces": [],
        "attachments": [],
        "errors": [],
    }
    for key, fn in (
        ("networks", list_networks),
        ("bridges", list_bridges),
        ("interfaces", list_interfaces),
        ("attachments", lambda h: list_attachments(h)),
    ):
        try:
            snap[key] = fn(hivemind_url)
        except NetworkAdminError as exc:
            snap["errors"].append(f"{key}: {exc}")
    return snap
