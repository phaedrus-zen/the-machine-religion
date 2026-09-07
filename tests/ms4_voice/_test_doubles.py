"""Shared, contract-complete doubles for isolated voice-stream tests."""

from __future__ import annotations

from typing import Any


def completed_chat_response(text: str, session_id: str) -> dict[str, Any]:
    """Return a successful Face Lobe terminal response.

    Capacity and scheduling tests use this helper so they exercise their
    intended REST-TTS behavior without bypassing the production completion
    guard or silently relying on a legacy partial response shape.
    """

    return {
        "text": text,
        "session_id": session_id,
        "completed": True,
        "cancelled": False,
    }
