"""Pre-rendered "reflex" audio for instant UI playback.

What this is
------------

A small, curated catalog of short phrases ("Mhm?", "One moment.",
"I'm having trouble reaching the language cluster…") that MS4
synthesizes ONCE via HiveMind TTS, writes to disk, and serves
from the gateway. The browser preloads them on page load and can
play them instantly — no round-trip to HiveMind, no SSE stream,
no waiting for ASR.

Three reasons this exists:

1. **Latency under cold load.** When the cluster cold-starts the TTS
   model, the first audio chunk can take 4–10s. A reflex like
   "Let me check…" plays in <50ms while the real reply is still
   synthesizing.

2. **Graceful failure.** When ``FaceLobeChat`` raises (HiveMind
   unreachable, stream stalled, 5xx), the canned error reflex plays
   immediately so the user hears an apology instead of silence.

3. **Barge-in acknowledgments.** When the operator presses the mic
   to interrupt MS4 mid-utterance, an "Mhm?" or "Yes?" reflex plays
   instantly — the same affordance ChatGPT-mobile uses.

Storage layout
~~~~~~~~~~~~~~

Reflexes are persisted under ``machine_spirit_4/canned_audio/<voice>/<id>.wav``
so different TTS voices can each have a generated set. The default
voice (``MS4_REFLEX_VOICE``, default ``alloy``) is generated on
gateway boot in a background thread. Operators can regenerate at
any time via ``POST /reflexes/regenerate``.

Adding a new reflex
~~~~~~~~~~~~~~~~~~~

Append a ``Reflex`` entry to :data:`REFLEXES` with a unique snake_case
id, the text to synthesize, and the category. Then either restart
the gateway or hit ``POST /reflexes/regenerate`` to render it.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .voice import DEFAULT_TTS_FORMAT, DEFAULT_TTS_MODEL, DEFAULT_TTS_VOICE, synthesize


log = logging.getLogger("ms4.gateway.canned_reflexes")


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


CATEGORY_ACK = "ack"
CATEGORY_THINKING = "thinking"
CATEGORY_ERROR = "error"
CATEGORY_CONFIRM = "confirm"
CATEGORY_IDENTITY = "identity"


@dataclass(frozen=True)
class Reflex:
    """One canned phrase entry. Immutable so the catalog is safe to
    iterate in the route handlers without locking."""

    id: str
    text: str
    category: str
    description: str = ""


REFLEXES: tuple[Reflex, ...] = (
    # ---- acknowledgments — used on barge-in and "I heard you" --------------
    Reflex("ack_mhm", "Mhm?", CATEGORY_ACK, "Soft acknowledgment when the user interrupts."),
    Reflex("ack_yes", "Yes?", CATEGORY_ACK, "Slightly more attentive ack."),
    Reflex("ack_go_ahead", "Go ahead.", CATEGORY_ACK, "Open prompt to continue speaking."),
    Reflex("ack_listening", "I'm listening.", CATEGORY_ACK, "Longer reassurance the user has the floor."),
    # ---- thinking covers — fill the dead air during a cold call ------------
    Reflex("thinking_one_moment", "One moment.", CATEGORY_THINKING, "Short stall while the model wakes up."),
    Reflex("thinking_let_me_check", "Let me check.", CATEGORY_THINKING, "Stall when about to dispatch a deep job."),
    Reflex("thinking_working_on_it", "Working on it.", CATEGORY_THINKING, "Mid-flight job confirmation."),
    # ---- errors — replace ugly raw exceptions with apologies ---------------
    Reflex(
        "err_cluster_unreachable",
        "I'm having trouble reaching the language cluster. Try again in a moment.",
        CATEGORY_ERROR,
        "Spoken when FaceLobeChat raises a HiveMind-unreachable error.",
    ),
    Reflex(
        "err_didnt_catch",
        "I didn't catch that. Could you repeat?",
        CATEGORY_ERROR,
        "Spoken when ASR returns an empty transcript.",
    ),
    Reflex(
        "err_no_input",
        "I didn't hear anything.",
        CATEGORY_ERROR,
        "Spoken when the mic captured nothing.",
    ),
    Reflex(
        "err_asr_not_ready",
        "Speech recognition isn't provisioned yet. You can provision it from the settings dialog.",
        CATEGORY_ERROR,
        "Spoken when voice_input_ready=false.",
    ),
    # ---- confirmations — for tool completions, etc. ------------------------
    Reflex("conf_okay", "Okay.", CATEGORY_CONFIRM),
    Reflex("conf_got_it", "Got it.", CATEGORY_CONFIRM),
    Reflex("conf_done", "Done.", CATEGORY_CONFIRM),
    # ---- identity ----------------------------------------------------------
    Reflex("id_online", "MS4 online.", CATEGORY_IDENTITY, "Played when the UI first becomes ready."),
    Reflex("id_face_lobe", "Face Lobe ready.", CATEGORY_IDENTITY),
)


# Quick lookup by id.
_REFLEX_BY_ID: dict[str, Reflex] = {r.id: r for r in REFLEXES}


def get_reflex(reflex_id: str) -> Reflex | None:
    return _REFLEX_BY_ID.get(reflex_id)


# ---------------------------------------------------------------------------
# Config + paths
# ---------------------------------------------------------------------------


DEFAULT_REFLEX_VOICE = os.environ.get("MS4_REFLEX_VOICE", DEFAULT_TTS_VOICE)
DEFAULT_REFLEX_MODEL = os.environ.get("MS4_REFLEX_MODEL", DEFAULT_TTS_MODEL)
DEFAULT_REFLEX_FORMAT = os.environ.get("MS4_REFLEX_FORMAT", DEFAULT_TTS_FORMAT)

# Path discipline: the cache lives under the MS4 package so it
# travels with the contained runtime. Operators can override via
# MS4_REFLEX_DIR for a shared cache across MS4 installs.
DEFAULT_REFLEX_DIR = Path(__file__).resolve().parent.parent / "canned_audio"


def reflex_dir() -> Path:
    custom = os.environ.get("MS4_REFLEX_DIR", "").strip()
    return Path(custom) if custom else DEFAULT_REFLEX_DIR


# Voice id sanitization for the on-disk path. We don't want a
# malicious or typo'd voice name to write outside the cache dir.
_SAFE_VOICE_RE = re.compile(r"[^A-Za-z0-9._-]")


def _safe_voice(voice: str) -> str:
    voice = (voice or "").strip()
    safe = _SAFE_VOICE_RE.sub("_", voice)
    if not safe:
        return DEFAULT_REFLEX_VOICE
    if safe.startswith(".") or ".." in safe:
        return DEFAULT_REFLEX_VOICE
    return safe


def reflex_path(voice: str, reflex_id: str) -> Path:
    safe_voice = _safe_voice(voice)
    if not _REFLEX_BY_ID.get(reflex_id):
        raise ValueError(f"unknown reflex id: {reflex_id!r}")
    return reflex_dir() / safe_voice / f"{reflex_id}.wav"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ReflexUnknown(ValueError):
    """Caller asked about a reflex id that isn't in the catalog."""


class ReflexGenerationError(RuntimeError):
    """HiveMind TTS failed for at least one reflex. Aggregates per-id errors."""

    def __init__(self, failures: dict[str, str]) -> None:
        super().__init__(f"reflex generation failed for {sorted(failures)}")
        self.failures = failures


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_reflex(
    *,
    hivemind_url: str,
    reflex_id: str,
    voice: str | None = None,
    model: str | None = None,
    response_format: str | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    """Synthesize one reflex via HiveMind TTS and persist the WAV
    bytes. Returns ``{id, voice, path, size_bytes}``.

    Idempotent: writing over an existing file is fine. Callers that
    want only-if-missing semantics should check ``reflex_path`` first.
    """
    reflex = _REFLEX_BY_ID.get(reflex_id)
    if reflex is None:
        raise ReflexUnknown(f"unknown reflex id: {reflex_id!r}")
    use_voice = voice or DEFAULT_REFLEX_VOICE
    use_model = model or DEFAULT_REFLEX_MODEL
    use_format = response_format or DEFAULT_REFLEX_FORMAT

    result = synthesize(
        hivemind_url=hivemind_url,
        text=reflex.text,
        model=use_model,
        voice=use_voice,
        response_format=use_format,
        timeout=timeout,
    )
    audio_bytes = result["audio_bytes"]
    path = reflex_path(use_voice, reflex_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(audio_bytes)
    return {
        "id": reflex_id,
        "voice": _safe_voice(use_voice),
        "model": use_model,
        "format": use_format,
        "path": str(path),
        "size_bytes": len(audio_bytes),
    }


def generate_all(
    *,
    hivemind_url: str,
    voice: str | None = None,
    model: str | None = None,
    response_format: str | None = None,
    force: bool = False,
    timeout: int = 60,
) -> dict[str, Any]:
    """Synthesize every reflex in :data:`REFLEXES` for ``voice``.

    By default skips reflexes that already exist on disk. Pass
    ``force=True`` to re-render them (e.g. after the voice or model
    has changed).
    """
    use_voice = voice or DEFAULT_REFLEX_VOICE
    generated: list[dict[str, Any]] = []
    skipped: list[str] = []
    failed: dict[str, str] = {}
    for reflex in REFLEXES:
        path = reflex_path(use_voice, reflex.id)
        if path.exists() and not force:
            skipped.append(reflex.id)
            continue
        try:
            info = generate_reflex(
                hivemind_url=hivemind_url,
                reflex_id=reflex.id,
                voice=use_voice,
                model=model,
                response_format=response_format,
                timeout=timeout,
            )
            generated.append(info)
        except Exception as exc:
            log.warning("reflex %r generation failed: %s", reflex.id, exc)
            failed[reflex.id] = str(exc)[:240]
    return {
        "schema": "Ms4ReflexGeneration.v1",
        "voice": _safe_voice(use_voice),
        "model": model or DEFAULT_REFLEX_MODEL,
        "force": force,
        "generated": generated,
        "skipped": skipped,
        "failed": failed,
        "total": len(REFLEXES),
    }


def generate_all_async(
    *,
    hivemind_url: str,
    voice: str | None = None,
    model: str | None = None,
    response_format: str | None = None,
    force: bool = False,
    timeout: int = 60,
) -> threading.Thread:
    """Fire-and-forget wrapper used by the gateway boot prewarm."""

    def _run() -> None:
        try:
            result = generate_all(
                hivemind_url=hivemind_url,
                voice=voice,
                model=model,
                response_format=response_format,
                force=force,
                timeout=timeout,
            )
            log.info(
                "reflex generate_all complete: voice=%s generated=%d skipped=%d failed=%d",
                result.get("voice"),
                len(result.get("generated") or []),
                len(result.get("skipped") or []),
                len(result.get("failed") or {}),
            )
        except Exception as exc:
            log.warning("reflex generate_all_async crashed: %s", exc)

    thread = threading.Thread(target=_run, daemon=True, name="ms4-reflex-prewarm")
    thread.start()
    return thread


# ---------------------------------------------------------------------------
# Read + list (gateway routes use these)
# ---------------------------------------------------------------------------


def read_reflex(*, reflex_id: str, voice: str | None = None) -> bytes | None:
    """Return the cached WAV bytes for a reflex, or ``None`` if it
    hasn't been generated yet. Raises :class:`ReflexUnknown` for an
    unknown id so the gateway can map to 400 vs 404 correctly."""
    if not _REFLEX_BY_ID.get(reflex_id):
        raise ReflexUnknown(f"unknown reflex id: {reflex_id!r}")
    path = reflex_path(voice or DEFAULT_REFLEX_VOICE, reflex_id)
    if not path.exists():
        return None
    return path.read_bytes()


def list_reflexes(*, voice: str | None = None) -> dict[str, Any]:
    """Catalog snapshot + on-disk availability for one voice."""
    use_voice = _safe_voice(voice or DEFAULT_REFLEX_VOICE)
    entries: list[dict[str, Any]] = []
    available = 0
    total_bytes = 0
    for reflex in REFLEXES:
        path = reflex_path(use_voice, reflex.id)
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        if exists:
            available += 1
            total_bytes += size
        entries.append({
            "id": reflex.id,
            "text": reflex.text,
            "category": reflex.category,
            "description": reflex.description,
            "voice": use_voice,
            "url": f"/reflexes/{use_voice}/{reflex.id}.wav",
            "available": exists,
            "size_bytes": size,
        })
    return {
        "schema": "Ms4ReflexCatalog.v1",
        "voice": use_voice,
        "default_voice": DEFAULT_REFLEX_VOICE,
        "total": len(REFLEXES),
        "available": available,
        "total_bytes": total_bytes,
        "categories": sorted({r.category for r in REFLEXES}),
        "reflexes": entries,
    }
