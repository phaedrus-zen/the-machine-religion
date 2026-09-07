"""Voice pipeline stage truth.

Input readiness, ASR, reasoning, synthesized bytes, playback/delivery,
persisted recent-turn rows, and UI status are separate facts. Input-ready
never implies output-delivered.
"""

from __future__ import annotations

from typing import Any


def classify_voice_pipeline_truth(
    *,
    input_ready: bool,
    asr_transcript: str | None,
    reasoning_result: str | None,
    tts_bytes: bytes | None,
    playback_receipt: dict[str, Any] | None,
    persisted_turn: dict[str, Any] | None,
    ui_status: str | None = None,
) -> dict[str, Any]:
    asr_ok = bool(str(asr_transcript or "").strip())
    reasoning_ok = bool(str(reasoning_result or "").strip())
    tts_bytes_ok = bool(tts_bytes) and len(tts_bytes) > 0
    playback = playback_receipt if isinstance(playback_receipt, dict) else None
    playback_ok = bool(playback) and (
        playback.get("client_written") is True
        or int(playback.get("bytes_played") or 0) > 0
    )
    persisted = bool(persisted_turn)
    output_delivered = bool(tts_bytes_ok and playback_ok)
    return {
        "input_ready": bool(input_ready),
        "asr_ok": asr_ok,
        "reasoning_ok": reasoning_ok,
        "tts_bytes_ok": tts_bytes_ok,
        "playback_ok": playback_ok,
        "persisted": persisted,
        "output_delivered": output_delivered,
        "ui_status": ui_status,
    }
