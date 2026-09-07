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


def _fetch_profiles_payload(hivemind_url: str) -> Any:
    try:
        return tools.loadout_profiles(hivemind_url)
    except HivemindToolError as exc:
        raise LoadoutAdminError(f"loadout.profiles failed: {exc}") from exc


def _normalise_profiles(raw: Any) -> list[dict[str, Any]]:
    """Normalize both legacy profile lists and HLI's tier-map response."""
    if isinstance(raw, list):
        return [p for p in raw if isinstance(p, dict)]
    if not isinstance(raw, dict):
        return []

    candidate = raw.get("profiles") or raw.get("data") or []
    if isinstance(candidate, list):
        return [p for p in candidate if isinstance(p, dict)]
    if not isinstance(candidate, dict):
        return []

    presets = candidate.get("presets")
    tiers = candidate.get("tiers")
    if not isinstance(presets, dict):
        return []
    tier_meta = tiers if isinstance(tiers, dict) else {}
    active = raw.get("active_loadout")
    active_tier = active.get("tier") if isinstance(active, dict) else None
    recommended_tier = raw.get("recommended_tier")

    profiles: list[dict[str, Any]] = []
    for profile_id, preset in sorted(presets.items()):
        if not isinstance(profile_id, str) or not isinstance(preset, dict):
            continue
        meta = tier_meta.get(profile_id)
        meta = meta if isinstance(meta, dict) else {}
        models = [
            str(spec.get("model"))
            for spec in preset.values()
            if isinstance(spec, dict) and spec.get("model")
        ]
        profiles.append(
            {
                "id": profile_id,
                "profile_id": profile_id,
                "name": str(meta.get("label") or profile_id),
                "description": str(meta.get("description") or ""),
                "target_hardware": meta.get("target_hardware"),
                "vram_gb": meta.get("vram_gb"),
                "models": models,
                "capabilities": sorted(str(cap) for cap in preset),
                "active": active_tier == profile_id,
                "recommended": recommended_tier == profile_id,
            }
        )
    return profiles


def list_profiles(hivemind_url: str) -> list[dict[str, Any]]:
    """All loadout profiles configured on the cluster."""
    return _normalise_profiles(_fetch_profiles_payload(hivemind_url))


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
        raw = _fetch_profiles_payload(hivemind_url)
        snap["profiles"] = _normalise_profiles(raw)
        if isinstance(raw, dict):
            snap["hardware"] = raw.get("hardware")
            snap["recommended_tier"] = raw.get("recommended_tier")
            snap["active_loadout"] = raw.get("active_loadout")
    except LoadoutAdminError as exc:
        snap["errors"].append(f"loadout.profiles: {exc}")
    return snap
