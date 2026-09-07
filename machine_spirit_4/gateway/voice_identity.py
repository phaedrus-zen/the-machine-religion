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
is emitted as a late voice event; session persistence is not yet
wired, so this module does not claim recognition survives the turn.
"""

from __future__ import annotations

import base64
import logging
import math
import threading
from typing import Any

from . import hivemind_tools as tools
from .hivemind_tools import HivemindToolError


log = logging.getLogger("ms4.gateway.voice_identity")


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


def refine(
    hivemind_url: str,
    *,
    name: str,
    embedding: list[float],
    blend_alpha: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Blend a pre-computed voice embedding into an existing identity.

    HLI's current refine contract is ``{name, embedding}``; unlike enroll,
    it does not accept raw audio. Embedding extraction therefore remains the
    caller's responsibility instead of pretending an audio upload can work.
    """
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("refine requires a non-empty name")
    if not isinstance(embedding, list) or not embedding:
        raise ValueError("refine requires a non-empty embedding array; audio is not supported")
    clean_embedding: list[float] = []
    for value in embedding:
        if isinstance(value, bool):
            raise ValueError("refine embedding values must be finite numbers")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("refine embedding values must be finite numbers") from exc
        if not math.isfinite(number):
            raise ValueError("refine embedding values must be finite numbers")
        clean_embedding.append(number)
    try:
        return tools.voice_identities_refine(
            hivemind_url,
            name=clean_name,
            embedding=clean_embedding,
            blend_alpha=blend_alpha,
            metadata=metadata,
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
    timeout: float = 5,
    cancel_event: threading.Event | None = None,
) -> dict[str, Any] | None:
    """Best-effort speaker identification from a captured WAV.

    Returns a normalized accepted result when HLI's canonical flat response
    has ``below_threshold=false`` and a name. HLI is the sole threshold owner;
    TMR never re-scores or overrides that verdict. Unknown, below-threshold,
    or malformed replies return ``None`` rather than guessing.

    Swallows every failure mode (transport, isError, malformed
    response) and returns ``None`` — we'd rather drop the speaker
    name than crash a working voice turn.
    """
    if not audio or (cancel_event is not None and cancel_event.is_set()):
        return None
    try:
        b64 = base64.b64encode(audio).decode("ascii")
        raw = tools.voice_identities_identify(
            hivemind_url,
            audio_base64=b64,
            top_k=top_k,
            timeout=timeout,
            cancel_event=cancel_event,
        )
        if cancel_event is not None and cancel_event.is_set():
            return None
    except HivemindToolError as exc:
        log.info("speaker identify failed (no speaker label this turn): %s", exc)
        return None
    except Exception as exc:  # noqa: BLE001 (deliberately fail-soft)
        log.warning("speaker identify unexpected error: %s", exc)
        return None

    if not isinstance(raw, dict):
        return None
    if raw.get("ok") is not True:
        return None
    below_threshold = raw.get("below_threshold")
    name = raw.get("name")
    if below_threshold is not False or not isinstance(name, str) or not name.strip():
        return None
    try:
        confidence = float(raw["confidence"])
        threshold = float(raw["threshold"])
    except (KeyError, TypeError, ValueError):
        return None
    if (
        not math.isfinite(confidence)
        or not math.isfinite(threshold)
        or not 0.0 <= confidence <= 1.0
        or not 0.0 <= threshold <= 1.0
    ):
        return None
    return {
        "schema": "Ms4SpeakerIdentification.v2",
        "name": name.strip(),
        # ``score`` remains as a display compatibility alias; both values
        # come directly from HLI's canonical ``confidence`` field.
        "score": confidence,
        "confidence": confidence,
        "threshold": threshold,
        "below_threshold": False,
        "accepted": True,
    }
