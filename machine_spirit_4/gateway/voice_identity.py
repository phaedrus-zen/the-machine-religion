"""HiveMind voice-identity admin + per-turn speaker identification.

What this module owns
---------------------

* CRUD over enrolled voice identities (``hivemind.voice_identities.*@v1``).
* A per-turn helper :func:`identify_speaker_from_wav` that the
  hands-free voice loop calls with the captured WAV so the UI can
  display the speaker's name on each turn.

Design notes
------------

We treat the HiveMind ``identify`` call as **best-effort**: it runs
inside the voice loop on a tight ~1.5 s budget so it can't add
perceptible latency. On any failure (timeout, low confidence, no
match) the loop proceeds without a speaker name. The identity ID
is also persisted on the FaceLobeChat session so the model can
address the speaker by name once enough turns have been recognized.
"""

from __future__ import annotations

import base64
import logging
from typing import Any

from . import hivemind_tools as tools
from .hivemind_tools import HivemindToolError


log = logging.getLogger("ms4.gateway.voice_identity")


# Default minimum score required to claim a "this speaker is X"
# identification. HiveMind's identify call returns a score in
# [0.0, 1.0]; 0.65 is a sane middle ground (too strict and every
# turn is "unknown"; too loose and the UI flaps between identities
# turn-over-turn). Tunable via MS4_VOICE_IDENTITY_MIN_SCORE.
DEFAULT_MIN_SCORE = 0.65


class VoiceIdentityError(RuntimeError):
    """Raised on a non-recoverable voice-identity admin failure."""


# ---------------------------------------------------------------------------
# Read paths
# ---------------------------------------------------------------------------


def list_identities(hivemind_url: str) -> list[dict[str, Any]]:
    """Return enrolled identities as a list of dicts. Normalises
    HiveMind's response which is sometimes wrapped in ``{identities: [...]}``."""
    try:
        raw = tools.voice_identities_list(hivemind_url)
    except HivemindToolError as exc:
        raise VoiceIdentityError(f"voice_identities.list failed: {exc}") from exc
    if isinstance(raw, dict):
        cand = raw.get("identities") or raw.get("data") or []
        return [i for i in cand if isinstance(i, dict)]
    if isinstance(raw, list):
        return [i for i in raw if isinstance(i, dict)]
    return []


# ---------------------------------------------------------------------------
# Enrollment + refinement + deletion
# ---------------------------------------------------------------------------


def enroll(
    hivemind_url: str,
    *,
    name: str,
    audio: bytes,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Enroll a new voice. ``audio`` should be a WAV blob (recommended
    ≥5 s, 16 kHz mono); we base64-encode for the JSON-RPC envelope."""
    if not name:
        raise ValueError("enroll requires a non-empty name")
    if not audio:
        raise ValueError("enroll requires non-empty audio")
    b64 = base64.b64encode(audio).decode("ascii")
    try:
        return tools.voice_identities_enroll(
            hivemind_url, audio_base64=b64, name=name, metadata=metadata
        )
    except HivemindToolError as exc:
        raise VoiceIdentityError(f"voice_identities.enroll failed: {exc}") from exc


def refine(hivemind_url: str, *, identity_id: str, audio: bytes) -> dict[str, Any]:
    """Append a new audio sample to an existing identity to improve
    future recognition."""
    if not identity_id:
        raise ValueError("refine requires an identity_id")
    if not audio:
        raise ValueError("refine requires non-empty audio")
    b64 = base64.b64encode(audio).decode("ascii")
    try:
        return tools.voice_identities_refine(
            hivemind_url, identity_id=identity_id, audio_base64=b64
        )
    except HivemindToolError as exc:
        raise VoiceIdentityError(f"voice_identities.refine failed: {exc}") from exc


def delete(hivemind_url: str, identity_id: str) -> dict[str, Any]:
    if not identity_id:
        raise ValueError("delete requires an identity_id")
    try:
        return tools.voice_identities_delete(hivemind_url, identity_id)
    except HivemindToolError as exc:
        raise VoiceIdentityError(f"voice_identities.delete failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Per-turn identification (called inside the hands-free voice loop)
# ---------------------------------------------------------------------------


def identify_speaker_from_wav(
    hivemind_url: str,
    audio: bytes,
    *,
    top_k: int = 1,
    min_score: float = DEFAULT_MIN_SCORE,
    timeout: int = 5,
) -> dict[str, Any] | None:
    """Best-effort speaker identification from a captured WAV.

    Returns ``{identity_id, name, score, accepted: True}`` when the
    top match's score crosses ``min_score``, otherwise ``None`` (the
    voice loop displays "unknown speaker" in that case rather than
    guessing).

    Swallows every failure mode (transport, isError, malformed
    response) and returns ``None`` — we'd rather drop the speaker
    name than crash a working voice turn.
    """
    if not audio:
        return None
    try:
        b64 = base64.b64encode(audio).decode("ascii")
        raw = tools.voice_identities_identify(
            hivemind_url, audio_base64=b64, top_k=top_k
        )
    except HivemindToolError as exc:
        log.info("speaker identify failed (no speaker label this turn): %s", exc)
        return None
    except Exception as exc:  # noqa: BLE001 (deliberately fail-soft)
        log.warning("speaker identify unexpected error: %s", exc)
        return None

    if not isinstance(raw, dict):
        return None
    matches = raw.get("matches") or raw.get("results") or []
    if not isinstance(matches, list) or not matches:
        return None
    top = matches[0] if isinstance(matches[0], dict) else None
    if not top:
        return None
    score = float(top.get("score") or 0.0)
    if score < min_score:
        return None
    return {
        "schema": "Ms4SpeakerIdentification.v1",
        "identity_id": str(top.get("identity_id") or top.get("id") or ""),
        "name": str(top.get("name") or ""),
        "score": score,
        "accepted": True,
        "min_score": min_score,
        "top_k": top_k,
    }
