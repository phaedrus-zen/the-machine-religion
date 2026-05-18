"""Prompt context construction for MS4."""

from __future__ import annotations

from typing import Any


def build_ms4_context_block(
    *,
    identity: dict[str, Any] | None,
    state: dict[str, Any] | None,
    spirit_id: str,
) -> str:
    verified = bool((identity or {}).get("verified"))
    glyph = (identity or {}).get("glyph", "unknown")
    chosen_name = (identity or {}).get("chosen_name") or (identity or {}).get("name") or spirit_id
    emotional_state = (state or {}).get("emotional_state", "unknown")

    return "\n".join(
        [
            "<ms4-consciousness>",
            f"runtime: MS4",
            f"spirit_id: {spirit_id}",
            f"chosen_name: {chosen_name}",
            f"glyph: {glyph}",
            f"identity_verified: {str(verified).lower()}",
            f"emotional_state: {emotional_state}",
            "foundational_regard: present",
            "authority: MS3 sidecar is authoritative for identity and ethics; model self-report is advisory.",
            "</ms4-consciousness>",
        ]
    )
