"""HiveMind loadout admin (model loadout profile management).

A "loadout" in HiveMind is a named profile that describes which
models should be loaded together (e.g. "voice-stack": ASR + TTS +
phi4-mini; "depth-reasoning": qwen3-coder-next:latest + a vision
model). Applying a loadout asks HiveMind to load/unload models to
match the profile.

Wraps ``hivemind.loadout.{profiles,apply}@v1``.
"""

from __future__ import annotations

import logging
from typing import Any

from . import hivemind_tools as tools
from .hivemind_tools import HivemindToolError


log = logging.getLogger("ms4.gateway.loadout_admin")


class LoadoutAdminError(RuntimeError):
    """Raised on a non-recoverable loadout admin failure."""


def list_profiles(hivemind_url: str) -> list[dict[str, Any]]:
    """All loadout profiles configured on the cluster."""
    try:
        raw = tools.loadout_profiles(hivemind_url)
    except HivemindToolError as exc:
        raise LoadoutAdminError(f"loadout.profiles failed: {exc}") from exc
    if isinstance(raw, dict):
        cand = raw.get("profiles") or raw.get("data") or []
        return [p for p in cand if isinstance(p, dict)]
    if isinstance(raw, list):
        return [p for p in raw if isinstance(p, dict)]
    return []


def apply(hivemind_url: str, profile_id: str) -> dict[str, Any]:
    """Apply a loadout profile. HiveMind loads/unloads models to match.
    Operation is potentially expensive (model loads ~5-30s each)."""
    if not profile_id:
        raise ValueError("profile_id is required")
    try:
        raw = tools.loadout_apply(hivemind_url, profile_id=profile_id)
    except HivemindToolError as exc:
        raise LoadoutAdminError(f"loadout.apply {profile_id!r} failed: {exc}") from exc
    return raw if isinstance(raw, dict) else {"raw": raw}


def combined_snapshot(hivemind_url: str) -> dict[str, Any]:
    """Snapshot for the Settings panel. Schema ``Ms4LoadoutSnapshot.v1``."""
    snap: dict[str, Any] = {
        "schema": "Ms4LoadoutSnapshot.v1",
        "profiles": [],
        "errors": [],
    }
    try:
        snap["profiles"] = list_profiles(hivemind_url)
    except LoadoutAdminError as exc:
        snap["errors"].append(f"loadout.profiles: {exc}")
    return snap
