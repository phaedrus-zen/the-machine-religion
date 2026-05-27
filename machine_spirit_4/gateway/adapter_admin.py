"""HiveMind adapter admin (LoRA / PEFT adapter deployment).

Wraps ``hivemind.adapters.{list,deploy}@v1`` so MS4 can let the
operator see what adapters HiveMind knows about (typically the
output of training jobs) and deploy one to a model runtime so it
can be loaded for inference.
"""

from __future__ import annotations

import logging
from typing import Any

from . import hivemind_tools as tools
from .hivemind_tools import HivemindToolError


log = logging.getLogger("ms4.gateway.adapter_admin")


class AdapterAdminError(RuntimeError):
    """Raised on a non-recoverable adapter admin failure."""


def list_adapters(hivemind_url: str) -> list[dict[str, Any]]:
    """Available trained adapters across the cluster."""
    try:
        raw = tools.adapters_list(hivemind_url)
    except HivemindToolError as exc:
        raise AdapterAdminError(f"adapters.list failed: {exc}") from exc
    if isinstance(raw, dict):
        cand = raw.get("adapters") or raw.get("data") or []
        return [a for a in cand if isinstance(a, dict)]
    if isinstance(raw, list):
        return [a for a in raw if isinstance(a, dict)]
    return []


def deploy(hivemind_url: str, adapter_id: str, target_model: str | None = None) -> dict[str, Any]:
    """Deploy an adapter to a model runtime."""
    if not adapter_id:
        raise ValueError("adapter_id is required")
    try:
        raw = tools.adapters_deploy(hivemind_url, adapter_id=adapter_id, target_model=target_model)
    except HivemindToolError as exc:
        raise AdapterAdminError(f"adapters.deploy {adapter_id!r} failed: {exc}") from exc
    return raw if isinstance(raw, dict) else {"raw": raw}


def combined_snapshot(hivemind_url: str) -> dict[str, Any]:
    """Snapshot for the Settings panel. Schema ``Ms4AdapterSnapshot.v1``."""
    snap: dict[str, Any] = {
        "schema": "Ms4AdapterSnapshot.v1",
        "adapters": [],
        "errors": [],
    }
    try:
        snap["adapters"] = list_adapters(hivemind_url)
    except AdapterAdminError as exc:
        snap["errors"].append(f"adapters.list: {exc}")
    return snap
