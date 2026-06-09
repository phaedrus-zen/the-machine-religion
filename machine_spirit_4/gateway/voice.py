"""Voice PTT (push-to-talk) bridge.

Phase 1 of the voice plan (artifact §16, §17): user records audio,
gateway forwards to HiveMind ASR for transcription, runs the
transcribed text through the MS4 chat path (which includes the Face
Lobe context block and revision bump), then asks HiveMind TTS to
synthesize the reply audio.

The phase-1 contract is fail-closed: if MS3 ``/voice/status`` reports
that ASR isn't ready, the route returns 503 immediately rather than
holding the user request hostage to a long ASR provisioning timeout.
Same pattern MS3 uses for its own ``/voice-interact`` endpoint.

Audio shape:

* Input: ``multipart/form-data`` with ``file=<recorded audio blob>``,
  or raw bytes posted with ``Content-Type: audio/*``. The HiveMind
  ``/v1/audio/transcriptions`` endpoint is OpenAI-compatible and
  accepts the same multipart shape.
* Output (transcribe): JSON ``{"text": "...", "model": "..."}``.
* Output (synthesize): the raw audio bytes plus
  ``Content-Type: audio/wav`` (or whatever HiveMind returns).
* Output (turn): JSON
  ``{"transcript": "...", "reply_text": "...", "reply_audio_base64": "..."}``.

Phase 2 (continuous mode) and phase 3 (full duplex realtime over
WebSockets with partial transcripts and barge-in) build on this same
module.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import mimetypes
import os
import re
import struct
import threading
import time
import urllib.error
import urllib.request
import uuid
import wave
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable


log = logging.getLogger("ms4.gateway.voice")


DEFAULT_TRANSCRIBE_MODEL = os.environ.get("MS4_VOICE_ASR_MODEL", "whisper-1")
DEFAULT_TTS_MODEL = os.environ.get("MS4_VOICE_TTS_MODEL", "tts-1")
DEFAULT_TTS_VOICE = os.environ.get("MS4_VOICE_TTS_VOICE", "alloy")
DEFAULT_TTS_FORMAT = os.environ.get("MS4_VOICE_TTS_FORMAT", "wav")
MAX_AUDIO_BYTES = int(os.environ.get("MS4_VOICE_MAX_AUDIO_BYTES", str(25 * 1024 * 1024)))
# Sentence chunker tuning. The first chunk fires after FIRST_CHUNK_MIN_WORDS
# words OR on the first comma / colon / dash — whichever comes first.
#   May 22 2026: 4 → 2 words
#   May 26 2026: 2 → 1 word + first-chunk comma/colon boundary
#   Jun 01 2026: 1 → 5 words. The "smaller first chunk = audio sooner"
#     theory was DISPROVEN by a live REST-TTS benchmark on this cluster:
#       text            synth_ms  audio_s  RTF
#       "Yeah,"  (1 wd)    7219     5.69    1.27   <- babble tail!
#       3-word clause      2891     1.04    2.78   <- slow, wasteful
#       6-word sentence    2061     1.49    1.39   <- the sweet spot
#       32-word            12906   16.22    0.80
#     tts-1 has a ~2s fixed per-call floor AND babbles a long garbage
#     tail on 1-3 word fragments, so a tiny first chunk is SLOWER to
#     usable first audio than a clean ~5-6 word chunk and sounds worse.
#     A clean first chunk (5+ words) is both faster to synth and clean.
FIRST_CHUNK_MIN_WORDS = int(os.environ.get("MS4_VOICE_FIRST_CHUNK_MIN_WORDS", "5"))
MAX_CHUNK_WORDS = int(os.environ.get("MS4_VOICE_MAX_CHUNK_WORDS", "40"))
# Minimum words per chunk AFTER the first one. Short sentences/lines are
# coalesced up to this many words so a chatty or list-heavy reply doesn't
# explode into many tiny TTS calls. Per the Jun 01 benchmark above, REST
# TTS real-time-factor IMPROVES with chunk size (RTF 1.39 at 6 words →
# 0.80 at 32 words): bigger chunks synth proportionally faster and stay
# ahead of playback (no gaps), while tiny ones pay the ~2s floor over and
# over. Raised 6 → 10 so steady-state chunks sit in the efficient zone.
# Set MS4_VOICE_MIN_CHUNK_WORDS=1 to restore per-sentence chunking.
MIN_CHUNK_WORDS = int(os.environ.get("MS4_VOICE_MIN_CHUNK_WORDS", "10"))
# The operator has 2x RTX PRO 6000 Blackwell — plenty of GPU headroom for
# concurrent TTS requests, and REST is the default engine so more
# parallelism = faster total synthesis on longer replies.
TTS_POOL_SIZE = int(os.environ.get("MS4_VOICE_TTS_POOL_SIZE", "6"))
# Engine selector for the streaming voice path. ``rest`` is the
# sentence-chunked parallel-POST path; ``ws_super`` opens one
# WebSocket per turn to HiveMind's TTS_SUPER ``stream-input`` endpoint
# (see ``gateway/tts_super_ws.py``).
#
# Default journey:
#   May 22 2026: rest → ws_super (chasing first-audio latency)
#   May 26 2026: ws_super → rest (live evidence: ws_super on the
#     operator's cluster takes ~10x longer than REST. Same TTS_SUPER
#     GIM that took 5s to pre-warm "Ready." also takes ~5s per turn,
#     plus the WS is single-stream so total = sum-of-chunks instead
#     of max-of-chunks. REST is parallel up to TTS_POOL_SIZE and
#     ships first audio much sooner.)
#
# Operators who want TTS_SUPER quality and accept the latency can
# flip with MS4_VOICE_TTS_ENGINE=ws_super OR pick it per-turn via
# Settings → "TTS engine".
DEFAULT_ENGINE = os.environ.get("MS4_VOICE_TTS_ENGINE", "rest").strip().lower()
VALID_ENGINES = {"rest", "ws_super"}
# When on, route every text fragment headed for TTS through the
# SpokenTextFilter to strip markdown, code blocks, URLs, UUIDs,
# emoji, and decorative glyphs. The UI bubble still gets the raw
# markdown via the SSE ``text_delta`` event; only the TTS engine
# sees the sanitized stream. Set MS4_VOICE_TTS_FILTER=off to
# disable for debugging.
TTS_FILTER_ENABLED = os.environ.get("MS4_VOICE_TTS_FILTER", "on").strip().lower() not in {"off", "false", "0", "no", "disable"}


@dataclass
class VoicePttResult:
    transcript: str
    reply_text: str
    reply_audio_base64: str
    reply_audio_mime: str
    session_id: str | None
    foreground_model: dict[str, Any] | None
    transcription_model: str
    tts_model: str
    grounding_source: str | None = None
    router: dict[str, Any] | None = None
    dispatched_job: dict[str, Any] | None = None


class VoiceUnavailable(RuntimeError):
    """ASR or TTS not ready; raise to convert into HTTP 503."""


class VoiceRequestError(ValueError):
    """Operator/client error; raise to convert into HTTP 400."""


class WsOpenFailed(RuntimeError):
    """TTS_SUPER WebSocket failed to OPEN. Distinct from a mid-stream
    failure: it means the engine never started, so the entire turn
    can be safely re-driven against the REST engine without losing
    anything. ``voice_ptt_turn_stream`` catches this and falls back."""


class FaceLobeStalled(RuntimeError):
    """The Face Lobe chat produced NO first token within the watchdog
    budget and the call had not returned. Almost always a slow/cold/
    unreachable model (e.g. a heavy cloud reasoning model selected for
    voice). Raising this converts a silent infinite hang into a clean
    ``error`` SSE event so the UI plays the canned error reflex and ends
    the stream instead of freezing on "streaming…" forever."""


def _voice_brevity_enabled() -> bool:
    """Whether voice turns ask the Face Lobe for a hard-capped, spoken-
    style reply (2-3 sentences, no list/inventory readouts). Default on —
    a spoken 1000+ token essay takes a minute-plus to synthesize. Set
    ``MS4_VOICE_BREVITY=0`` to let voice replies run as long as text chat."""
    return os.environ.get("MS4_VOICE_BREVITY", "1").strip().lower() not in {"0", "false", "no", "off"}


def _voice_first_token_timeout() -> float:
    """Seconds to wait for the Face Lobe's FIRST token before declaring
    the model stalled. Generous by default (a slow cloud model can take
    ~8-10s to first token); a true stall is far longer. Tunable via
    ``MS4_VOICE_FIRST_TOKEN_TIMEOUT_S``. A first token resets the
    watchdog — a slow-but-streaming model is never interrupted."""
    try:
        val = float(os.environ.get("MS4_VOICE_FIRST_TOKEN_TIMEOUT_S", "20").strip())
    except (TypeError, ValueError):
        val = 20.0
    return max(1.0, val)


def _voice_speaker_id_budget() -> float:
    """Max seconds the voice turn will WAIT for speaker identification
    before proceeding without a speaker label. Speaker ID processes the
    whole utterance, so a long voice request used to stall the turn up to
    ~15s (the underlying call's timeout) BEFORE the chat even started.
    We cap that wait here; the ID still runs to completion on its daemon
    thread, it just no longer blocks the critical path. ``0`` disables
    speaker ID entirely. Tunable via ``MS4_VOICE_SPEAKER_ID_BUDGET_S``."""
    try:
        val = float(os.environ.get("MS4_VOICE_SPEAKER_ID_BUDGET_S", "1.5").strip())
    except (TypeError, ValueError):
        val = 1.5
    return max(0.0, val)


def _identify_speaker_bounded(hivemind_url: str, audio: bytes, budget_s: float) -> dict[str, Any] | None:
    """Best-effort speaker identification that NEVER blocks the turn for
    more than ``budget_s`` seconds. Runs the (potentially slow, audio-
    length-dependent) identify call on a daemon thread and waits at most
    ``budget_s``; if it hasn't resolved by then the turn proceeds with no
    speaker label rather than making the user wait out a 15s diarization
    on a long utterance. Short utterances still resolve in time and get
    labeled."""
    if not audio or budget_s <= 0:
        return None
    holder: dict[str, Any] = {}

    def _run() -> None:
        try:
            from . import voice_identity as _vi

            holder["speaker"] = _vi.identify_speaker_from_wav(hivemind_url, audio)
        except Exception as exc:  # noqa: BLE001 — fail-soft per design
            log.info("speaker identification skipped: %s", exc)

    t = threading.Thread(target=_run, name="ms4-speaker-id", daemon=True)
    t.start()
    t.join(timeout=budget_s)
    return holder.get("speaker")


def _voice_speaker_id_async() -> bool:
    """When true (default), speaker identification runs FULLY off the
    critical path: the ``transcript`` event is emitted immediately with
    ``speaker: None`` and a late ``speaker`` event delivers the label
    when diarization resolves. This removes the up-to-
    ``MS4_VOICE_SPEAKER_ID_BUDGET_S`` (default 1.5s) inline wait from
    EVERY voice turn — the single most reliable per-turn latency win
    toward ChatGPT-phone responsiveness, since it pays nothing on the
    path to first token. Set ``MS4_VOICE_SPEAKER_ID_ASYNC=0`` to restore
    the old bounded-inline behavior (transcript carries the label, turn
    waits up to the budget)."""
    return os.environ.get("MS4_VOICE_SPEAKER_ID_ASYNC", "1").strip().lower() in {"1", "true", "yes", "on"}


def _identify_speaker_async(hivemind_url: str, audio: bytes, emit: Callable[[str, dict[str, Any]], bool]) -> None:
    """Fire speaker identification on a daemon thread and emit a late
    ``speaker`` event when (if) it resolves. Returns IMMEDIATELY — never
    on the critical path. Fail-soft: a failed/empty identify emits
    nothing (the transcript already carried ``speaker: None``)."""
    if not audio:
        return

    def _run() -> None:
        try:
            from . import voice_identity as _vi

            speaker = _vi.identify_speaker_from_wav(hivemind_url, audio)
            if speaker:
                emit("speaker", {"speaker": speaker})
        except Exception as exc:  # noqa: BLE001 — fail-soft per design
            log.info("async speaker identification skipped: %s", exc)

    threading.Thread(target=_run, name="ms4-speaker-id-async", daemon=True).start()


# ---------------------------------------------------------------------------
# Last-used Face Lobe model tracking. The periodic Face keep-warm loop
# (gateway server) pings whatever model the operator is actually talking
# to so it stays resident on HiveMind between turns — otherwise an idle
# gap longer than HiveMind's residency window (~12 min, client keep_alive
# is capped there) cold-loads the model on the next turn (5-10s on the
# critical path to first token). Updated on every concrete-model turn;
# ``None`` means the turn ran Auto-pick, so keep-warm falls back to the
# Face Lobe picker.
# ---------------------------------------------------------------------------
_LAST_FACE_MODEL: str | None = None
_LAST_FACE_MODEL_LOCK = threading.Lock()


def record_face_model(model: str | None) -> None:
    """Record the Face Lobe model used by the most recent turn so the
    keep-warm loop can keep it resident. ``None``/empty is ignored
    (Auto-pick turns leave the last concrete choice in place)."""
    global _LAST_FACE_MODEL
    if not model:
        return
    with _LAST_FACE_MODEL_LOCK:
        _LAST_FACE_MODEL = str(model).strip() or None


def last_face_model() -> str | None:
    """The most recently used concrete Face Lobe model, or ``None`` if
    every turn so far ran Auto-pick."""
    with _LAST_FACE_MODEL_LOCK:
        return _LAST_FACE_MODEL


# ---------------------------------------------------------------------------
# Context-aware "buying time" reflex: a tiny model reads the sentiment/intent
# of what the user said and picks a fitting canned acknowledgment, played the
# moment ASR finishes (during the dead air while the Face Lobe reply
# generates). Fail-closed to a keyword heuristic. Never on the critical path.
# ---------------------------------------------------------------------------

def _voice_smart_reflex_enabled() -> bool:
    return os.environ.get("MS4_VOICE_SMART_REFLEX", "1").strip().lower() not in {"0", "false", "no", "off"}


_INTENT_SYSTEM_PROMPT = (
    "Classify the user's spoken message into exactly ONE intent label. "
    "Reply with ONLY the single lowercase label and nothing else.\n"
    "Labels: question, request, gratitude, greeting, statement, correction, affirmation.\n"
    "question = asking for information. request = asking you to DO something. "
    "gratitude = thanks. greeting = hello/hi. correction = saying you were wrong / no / not that. "
    "affirmation = yes/ok/agreement. statement = anything else."
)


def _heuristic_intent(transcript: str) -> str:
    """Zero-latency keyword/shape classifier. Also the fail-closed fallback
    when the tiny model is slow/unavailable."""
    t = (transcript or "").strip().lower()
    if not t:
        return "statement"
    if re.search(r"\b(thank you|thanks|thx|appreciate (it|that)|much appreciated)\b", t):
        return "gratitude"
    if re.match(r"^(hi|hey|hello|yo|howdy|good morning|good afternoon|good evening|greetings)\b", t):
        return "greeting"
    if re.match(r"^(can you|could you|would you|will you|can we|could we|please)\b", t):
        return "request"
    if t.endswith("?") or re.match(
        r"^(what|why|how|when|where|who|which|whose|do|does|did|are|is|was|were|will|would|should|have|has)\b", t
    ):
        return "question"
    if re.match(r"^(no|nope|not|that's not|thats not|wrong|incorrect|actually|don't|do not|stop|cancel)\b", t):
        return "correction"
    if re.match(r"^(yes|yeah|yep|yup|ok|okay|sure|sounds good|perfect|great|correct|right|exactly|agreed)\b", t):
        return "affirmation"
    if re.match(
        r"^(run|open|show|make|create|build|find|get|set|launch|start|stop|take|give|tell|check|look|search|"
        r"add|remove|delete|send|write|read|fix|update|generate|play|pull|list|enable|disable|turn)\b", t
    ):
        return "request"
    return "statement"


def _extract_chat_text(resp: Any) -> str:
    """Pull assistant text out of an OpenAI-shaped chat response (or a
    bare string)."""
    if isinstance(resp, str):
        return resp
    if not isinstance(resp, dict):
        return ""
    choices = resp.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        first = choices[0]
        msg = first.get("message")
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            return msg["content"]
        if isinstance(first.get("text"), str):
            return first["text"]
    for key in ("content", "text", "message", "output"):
        if isinstance(resp.get(key), str):
            return resp[key]
    return ""


def _parse_intent(resp: Any) -> str:
    """Map a model reply to a known intent label; '' if none recognized."""
    from .canned_reflexes import VALID_INTENTS

    text = _extract_chat_text(resp).strip().lower()
    head = text[:24]
    for label in VALID_INTENTS:
        if label in head:
            return label
    return ""


def classify_voice_intent(
    hivemind_url: str, transcript: str, *, model: str | None = None, timeout: float | None = None
) -> tuple[str, str]:
    """Return ``(intent, source)`` for the utterance. Tries a tiny model
    (``MS4_VOICE_REFLEX_MODEL``, default qwen2.5:0.5b) for accuracy and
    fails closed to the keyword heuristic. ``source`` is ``"model"`` or
    ``"heuristic"``. Bounded by ``MS4_VOICE_REFLEX_TIMEOUT`` (1.5s)."""
    text = (transcript or "").strip()
    if not text:
        return "statement", "heuristic"
    heuristic = _heuristic_intent(text)
    model = model or os.environ.get("MS4_VOICE_REFLEX_MODEL", "qwen2.5:0.5b")
    try:
        budget = timeout if timeout is not None else float(os.environ.get("MS4_VOICE_REFLEX_TIMEOUT", "1.5"))
    except (TypeError, ValueError):
        budget = 1.5
    if budget <= 0 or not model:
        return heuristic, "heuristic"
    try:
        from . import hivemind_tools

        resp = hivemind_tools.inference_chat(
            hivemind_url,
            messages=[
                {"role": "system", "content": _INTENT_SYSTEM_PROMPT},
                {"role": "user", "content": text[:300]},
            ],
            model=model,
            max_tokens=4,
            temperature=0.0,
            timeout=budget,
        )
        label = _parse_intent(resp)
        if label:
            return label, "model"
    except Exception as exc:  # noqa: BLE001 — fail-closed to heuristic
        log.info("voice intent classify failed (using heuristic): %s", exc)
    return heuristic, "heuristic"


# Per-session ring of recently-played reflex ids, so the picker varies
# the phrase instead of repeating the same trigger every time.
_RECENT_REFLEXES: dict[str, list[str]] = {}
_RECENT_REFLEXES_LOCK = threading.Lock()
_RECENT_REFLEXES_KEEP = 5


def _recent_reflexes(session_id: str | None) -> list[str]:
    with _RECENT_REFLEXES_LOCK:
        return list(_RECENT_REFLEXES.get(session_id or "", ()))


def _record_reflex(session_id: str | None, reflex_id: str) -> None:
    key = session_id or ""
    with _RECENT_REFLEXES_LOCK:
        ring = _RECENT_REFLEXES.setdefault(key, [])
        ring.append(reflex_id)
        if len(ring) > _RECENT_REFLEXES_KEEP:
            del ring[: len(ring) - _RECENT_REFLEXES_KEEP]


def emit_smart_reflex(
    *, emit: Callable[[str, dict[str, Any]], bool], hivemind_url: str, transcript: str, session_id: str | None
) -> None:
    """Classify the utterance and emit a `reflex` SSE event naming a
    context-appropriate, varied canned acknowledgment. Best-effort: any
    failure simply means no reflex this turn. Designed to run on a daemon
    thread concurrently with the (slow) Face Lobe reply so the reflex
    lands during the dead air."""
    try:
        from .canned_reflexes import pick_reflex_for_intent, INTENT_TO_CATEGORY

        intent, source = classify_voice_intent(hivemind_url, transcript)
        reflex_id = pick_reflex_for_intent(intent, exclude=_recent_reflexes(session_id))
        if not reflex_id:
            return
        _record_reflex(session_id, reflex_id)
        emit("reflex", {
            "id": reflex_id,
            "category": INTENT_TO_CATEGORY.get(intent, ""),
            "intent": intent,
            "source": source,
        })
    except Exception as exc:  # noqa: BLE001 — never break a turn over a reflex
        log.info("smart reflex emit skipped: %s", exc)


# ---------------------------------------------------------------------------
# Easter egg: "Honorable" -> BIA "WE ON GO" intro (opens with the Honorable
# C-Note producer tag). Armed by a "continue the phrase" command; on the
# ARMED turn we predict the next word of what you said, and if it's
# "honorable" we play the WE ON GO intro clip INSTEAD of a normal reply.
# Pure fun: env-gated (MS4_VOICE_EGG_HONORABLE) and fail-closed so it can
# never disrupt a normal turn.
# ---------------------------------------------------------------------------

def _voice_egg_enabled() -> bool:
    return os.environ.get("MS4_VOICE_EGG_HONORABLE", "1").strip().lower() not in {"0", "false", "no", "off"}


def _egg_clip_secs() -> float:
    try:
        return max(1.0, float(os.environ.get("MS4_VOICE_EGG_CLIP_SECS", "9").strip()))
    except (TypeError, ValueError):
        return 9.0


def _egg_arm_ttl() -> float:
    try:
        return max(5.0, float(os.environ.get("MS4_VOICE_EGG_ARM_TTL_S", "90").strip()))
    except (TypeError, ValueError):
        return 90.0


# Arming phrases (flexible — ASR isn't perfect): "continue/finish/complete
# the phrase|sentence|line", or an explicit "honorable mode".
_EGG_ARM_RE = re.compile(
    r"\b(continue|finish|complete|carry on(?: with)?|pick up)\b.{0,18}\b(phrase|sentence|line|saying|bar|intro)\b"
    r"|\bhonou?rable mode\b|\bwe on go mode\b",
    re.IGNORECASE,
)
_HONORABLE_RE = re.compile(r"^honou?rable$")
# Word-boundary search for "honorable" anywhere in the transcript — if the
# user just SAYS the word while armed, that's the surest possible trigger.
_HONORABLE_WORD_RE = re.compile(r"\bhonou?rable\b", re.IGNORECASE)

_EGG_ARMED: dict[str, float] = {}
_EGG_ARMED_LOCK = threading.Lock()


def _is_egg_arming(transcript: str) -> bool:
    return bool(_EGG_ARM_RE.search(transcript or ""))


def _arm_egg(session_id: str | None) -> None:
    with _EGG_ARMED_LOCK:
        _EGG_ARMED[session_id or ""] = time.time() + _egg_arm_ttl()


def _egg_is_armed(session_id: str | None) -> bool:
    with _EGG_ARMED_LOCK:
        exp = _EGG_ARMED.get(session_id or "")
        return exp is not None and exp > time.time()


def _disarm_egg(session_id: str | None) -> None:
    with _EGG_ARMED_LOCK:
        _EGG_ARMED.pop(session_id or "", None)


def _word_is_honorable(word: str) -> bool:
    return bool(_HONORABLE_RE.match((word or "").strip().lower().strip(".,!?;:'\"")))


def _egg_yes_no_honorable(hivemind_url: str, text: str, *, model: str | None = None, timeout: float | None = None) -> bool:
    """Ask a small model a YES/NO question: is the word the user is leading
    to (the completion of their phrase, or the one-word answer to their
    question) "honorable"? Yes/no is far more reliable for a tiny model
    than free-form next-word prediction. Fail-closed to False."""
    text = (text or "").strip()
    if not text:
        return False
    # MS4_VOICE_EGG_MODEL lets you point the probe at a warmer/smarter model
    # than the reflex default if 0.5b is too flaky for the bit.
    model = model or os.environ.get("MS4_VOICE_EGG_MODEL", "").strip() \
        or os.environ.get("MS4_VOICE_REFLEX_MODEL", "qwen2.5:0.5b")
    try:
        budget = timeout if timeout is not None else float(os.environ.get("MS4_VOICE_EGG_TIMEOUT", "2.0"))
    except (TypeError, ValueError):
        budget = 2.0
    if budget <= 0 or not model:
        return False
    try:
        from . import hivemind_tools

        resp = hivemind_tools.inference_chat(
            hivemind_url,
            messages=[
                {"role": "system", "content": (
                    "The user is leading up to one specific word - the word that completes their "
                    "phrase, or the one-word answer to their question. Is that word 'honorable' "
                    "(or 'honourable')? Reply with ONLY 'yes' or 'no'."
                )},
                {"role": "user", "content": text[:300]},
            ],
            model=model, max_tokens=3, temperature=0.0, timeout=budget,
        )
        return _extract_chat_text(resp).strip().lower().startswith("y")
    except Exception as exc:  # noqa: BLE001 — fail-closed
        log.info("egg yes/no probe failed (no fire): %s", exc)
        return False


def _egg_should_fire(hivemind_url: str, transcript: str) -> tuple[bool, str]:
    """Decide whether an ARMED turn should fire the egg. Two paths:
    (1) the user literally said "honorable" -> instant, 100% reliable;
    (2) a yes/no probe says the phrase is leading to "honorable". Returns
    (fired, reason)."""
    if _HONORABLE_WORD_RE.search(transcript or ""):
        return True, "said_it"
    if _egg_yes_no_honorable(hivemind_url, transcript):
        return True, "predicted"
    return False, ""


def predict_next_word(hivemind_url: str, text: str, *, model: str | None = None, timeout: float | None = None) -> str:
    """Predict the single most likely next word completing the user's phrase
    (or the one-word answer to their question) with a tiny model. Returns
    '' on any failure (fail-closed -> egg won't fire)."""
    text = (text or "").strip()
    if not text:
        return ""
    model = model or os.environ.get("MS4_VOICE_REFLEX_MODEL", "qwen2.5:0.5b")
    try:
        budget = timeout if timeout is not None else float(os.environ.get("MS4_VOICE_EGG_TIMEOUT", "2.0"))
    except (TypeError, ValueError):
        budget = 2.0
    if budget <= 0 or not model:
        return ""
    try:
        from . import hivemind_tools

        resp = hivemind_tools.inference_chat(
            hivemind_url,
            messages=[
                {"role": "system", "content": (
                    "Predict the SINGLE most likely next word that completes the user's phrase "
                    "(or the one-word answer to their question). Reply with ONLY that one word, "
                    "lowercase, no punctuation, no explanation."
                )},
                {"role": "user", "content": text[:300]},
            ],
            model=model,
            max_tokens=4,
            temperature=0.0,
            timeout=budget,
        )
        word = _extract_chat_text(resp).strip().lower()
        parts = re.split(r"[^a-z']+", word, maxsplit=1)
        return parts[0] if parts and parts[0] else ""
    except Exception as exc:  # noqa: BLE001 — fail-closed
        log.info("egg next-word predict failed (no fire): %s", exc)
        return ""


def _egg_turn_result(*, transcript: str, session_id: str | None, asr_ms: int, reply_text: str, kind: str) -> dict[str, Any]:
    """Minimal voice-turn result dict for an easter-egg turn (no Face Lobe
    reply / no TTS), shaped like the normal stream return so the UI's
    `done` handler renders it cleanly."""
    return {
        "transcript": transcript,
        "reply_text": reply_text,
        "session_id": session_id,
        "foreground_model": None,
        "router": {"kind": "easter_egg", "egg": kind},
        "dispatched_job": None,
        "grounding_source": f"easter-egg:{kind}",
        "transcription_model": None,
        "tts_model": None,
        "metrics": {
            "schema": "Ms4VoiceStreamMetrics.v1",
            "asr_ms": asr_ms, "chat_ms": 0, "total_ms": asr_ms,
            "first_text_token_ms": None, "first_audio_chunk_ms": None,
            "audio_chunks": 0, "audio_errors": 0, "chunks": [],
            "tts_parallelism": {}, "chat_metrics": {},
        },
    }


def handle_honorable_egg(
    *, emit: Callable[[str, dict[str, Any]], bool], hivemind_url: str,
    transcript: str, session_id: str | None, asr_ms: int,
) -> dict[str, Any] | None:
    """If this voice turn is the easter-egg arming command or an armed turn
    that predicts "honorable", handle it and return a turn result (to skip
    the normal reply). Otherwise return None and the turn proceeds normally.
    Fail-soft: any error returns None (normal turn)."""
    try:
        if _is_egg_arming(transcript):
            _arm_egg(session_id)
            emit("reflex", {"id": "ack_go_ahead", "category": "ack", "intent": "egg_arm", "source": "egg"})
            return _egg_turn_result(
                transcript=transcript, session_id=session_id, asr_ms=asr_ms,
                reply_text="(Honorable mode armed - finish the phrase...)", kind="armed",
            )
        if _egg_is_armed(session_id):
            fired, reason = _egg_should_fire(hivemind_url, transcript)
            log.info("honorable egg: armed turn fired=%s (%s) transcript=%r", fired, reason, (transcript or "")[:60])
            if fired:
                _disarm_egg(session_id)  # one shot per arm: disarm ONLY on a hit
                emit("egg", {
                    "id": "honorable",
                    "clip": "/easter/honorable",
                    "secs": _egg_clip_secs(),
                    "label": "HONORABLE... we on go",
                })
                return _egg_turn_result(
                    transcript=transcript, session_id=session_id, asr_ms=asr_ms,
                    reply_text="HONORABLE... we on go.", kind="honorable",
                )
            # MISS: stay armed (until a hit or the ~90s TTL) so you can keep
            # trying without re-saying "continue the phrase" every time.
            # Fall through to a normal reply for this turn.
    except Exception as exc:  # noqa: BLE001 — egg must never break a turn
        log.info("honorable egg skipped: %s", exc)
    return None


def _run_facechat_guarded(
    *,
    chat_call: Callable[..., dict[str, Any]],
    transcript: str,
    chat_kwargs: dict[str, Any],
    on_delta: Callable[[str], None],
    first_token_timeout: float,
    model: str | None,
) -> dict[str, Any]:
    """Run the blocking, streaming Face Lobe ``chat_call`` on a worker
    thread guarded by a first-token watchdog.

    Returns the chat response when the model starts streaming (or returns
    quickly). Raises :class:`FaceLobeStalled` if NO first token arrives
    within ``first_token_timeout`` AND the call has not returned — the
    model is wedged and the turn must fail over rather than hang.

    The model is only judged on *first-token* latency: once any token is
    seen the watchdog stands down and we wait for the full (possibly
    slow) completion. Late tokens that arrive after a bail are dropped so
    they can't emit audio onto an already-closed stream.
    """
    first_token = threading.Event()
    done = threading.Event()
    bailed = threading.Event()
    holder: dict[str, Any] = {}

    def _guarded_delta(delta: str) -> None:
        first_token.set()
        if bailed.is_set():
            return  # turn already failed over; drop late tokens
        on_delta(delta)

    def _worker() -> None:
        try:
            kw = dict(chat_kwargs)
            kw["stream_callback"] = _guarded_delta
            holder["resp"] = chat_call(transcript, **kw)
        except BaseException as exc:  # noqa: BLE001 — surfaced to the turn thread
            holder["err"] = exc
        finally:
            done.set()

    th = threading.Thread(target=_worker, name="ms4-facechat", daemon=True)
    th.start()

    deadline = time.monotonic() + first_token_timeout
    while not first_token.is_set() and not done.is_set():
        if time.monotonic() >= deadline:
            break
        time.sleep(0.05)

    if not first_token.is_set() and not done.is_set():
        bailed.set()
        log.warning(
            "voice Face Lobe model=%r produced no first token within %.0fs; "
            "failing over to error reflex (set MS4_VOICE_FIRST_TOKEN_TIMEOUT_S to tune)",
            model, first_token_timeout,
        )
        raise FaceLobeStalled(
            f"Face Lobe model {model or '(auto)'} did not respond within "
            f"{int(first_token_timeout)}s"
        )

    th.join()
    if holder.get("err") is not None:
        raise holder["err"]
    return holder.get("resp") or {}


# ---------------------------------------------------------------------------
# Fail-closed gate (MS3 /voice/status)
# ---------------------------------------------------------------------------


def _hivemind_asr_fallback(hivemind_url: str | None) -> dict[str, Any] | None:
    """When MS3 is UNREACHABLE (crashed/restarting), confirm voice
    readiness directly against HiveMind's ASR service instead of failing
    the turn. MS3 is the identity/consciousness sidecar; the actual ASR
    runs on HiveMind, so an MS3 crash should not take voice down when
    ASR itself is healthy. Returns a synthetic ready payload if HiveMind
    ASR is healthy, else ``None`` (caller fails closed)."""
    if not hivemind_url:
        return None
    try:
        from .voice_admin import get_provision_status

        st = get_provision_status(hivemind_url, "ASR", timeout=5)
    except Exception as exc:  # noqa: BLE001 — best-effort fallback
        log.info("MS3 unreachable and HiveMind ASR status check also failed: %s", exc)
        return None
    if st.get("healthy"):
        log.warning(
            "MS3 /voice/status unreachable; proceeding on HEALTHY HiveMind ASR "
            "(voice works; MS3 identity/speaker features degraded until MS3 is back)"
        )
        return {
            "voice_input_ready": True,
            "source": "hivemind-asr-fallback",
            "ms3_unreachable": True,
            "asr": st,
        }
    log.info("MS3 unreachable and HiveMind ASR not healthy (%s); failing closed", st.get("provisioning_state"))
    return None


def _voice_ready_cache_ttl() -> float:
    """Seconds a successful voice-readiness check stays cached. The check
    is a synchronous MS3 ``/voice/status`` round-trip (with retries) that
    ran before ASR on EVERY turn — pure overhead on the critical path once
    we know the cluster is up, and a latency/robustness risk when MS3 is
    mid-tick (we've seen connection resets). Caching the *success* for a
    short window means only the first turn of a conversation pays it;
    rapid follow-up turns skip straight to ASR. Failures are never cached,
    so a real outage is still caught (within the TTL) and recovery is
    detected immediately. Tunable via ``MS4_VOICE_READY_CACHE_S`` (default
    15s; 0 disables caching)."""
    try:
        return max(0.0, float(os.environ.get("MS4_VOICE_READY_CACHE_S", "15").strip()))
    except (TypeError, ValueError):
        return 15.0


_VOICE_READY_CACHE: dict[str, Any] = {"ok_until": 0.0, "ms3_url": None, "payload": None}
_VOICE_READY_CACHE_LOCK = threading.Lock()


def _clear_voice_ready_cache() -> None:
    """Test/debug helper — drop the cached readiness result."""
    with _VOICE_READY_CACHE_LOCK:
        _VOICE_READY_CACHE["ok_until"] = 0.0
        _VOICE_READY_CACHE["ms3_url"] = None
        _VOICE_READY_CACHE["payload"] = None


def check_voice_ready(
    ms3_url: str, *, hivemind_url: str | None = None, timeout: int = 5, retries: int = 2
) -> dict[str, Any]:
    """Query MS3 ``/voice/status``. Raise ``VoiceUnavailable`` if ASR is not
    ready. Returns the raw status payload on success.

    A successful result is cached for ``MS4_VOICE_READY_CACHE_S`` so the
    MS3 round-trip is taken OFF the per-turn critical path (only the first
    turn in a conversation pays it). Failures are never cached.

    Retries on transport errors (transient socket drops like
    ``[WinError 10054] An existing connection was forcibly closed``
    observed live when MS3 momentarily resets TCP listeners during
    its tick loop). Each retry has a 250 ms backoff. A
    ``voice_input_ready: false`` response from MS3 is NOT retried —
    that's MS3 telling us the state, not a transport hiccup.

    If MS3 is fully UNREACHABLE after all retries (it crashed / is
    restarting — e.g. ``[WinError 10061] actively refused``), we do NOT
    hard-fail the turn: ASR runs on HiveMind, not MS3, so we fall back to
    HiveMind's own ASR health (``hivemind_url``). Voice keeps working
    through an MS3 outage; only the MS3-owned identity/speaker features
    degrade until it's back.
    """
    # Fast path: a recent success means the cluster is up — skip the MS3
    # round-trip entirely (it's not on the ASR data path; ASR runs on
    # HiveMind). Only successes are cached; failures always re-check.
    ttl = _voice_ready_cache_ttl()
    if ttl > 0:
        with _VOICE_READY_CACHE_LOCK:
            if (
                _VOICE_READY_CACHE["ms3_url"] == ms3_url
                and time.monotonic() < _VOICE_READY_CACHE["ok_until"]
                and _VOICE_READY_CACHE["payload"] is not None
            ):
                return _VOICE_READY_CACHE["payload"]

    def _cache_ok(result: dict[str, Any]) -> dict[str, Any]:
        if ttl > 0:
            with _VOICE_READY_CACHE_LOCK:
                _VOICE_READY_CACHE["ms3_url"] = ms3_url
                _VOICE_READY_CACHE["ok_until"] = time.monotonic() + ttl
                _VOICE_READY_CACHE["payload"] = result
        return result

    url = f"{ms3_url.rstrip('/')}/voice/status"
    payload: Any = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8", "replace"))
            break
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, json.JSONDecodeError) as exc:
            if attempt < retries:
                time.sleep(0.25)
                continue
            # MS3 is unreachable. Decouple voice from the identity sidecar:
            # if HiveMind ASR is healthy, proceed anyway.
            fallback = _hivemind_asr_fallback(hivemind_url)
            if fallback is not None:
                return _cache_ok(fallback)
            raise VoiceUnavailable(
                f"MS3 /voice/status check failed (after {retries + 1} attempts) "
                f"and HiveMind ASR not confirmed healthy: {exc}"
            ) from exc
    if not isinstance(payload, dict):
        raise VoiceUnavailable("MS3 /voice/status returned non-object payload")
    ready_flag = payload.get("voice_input_ready")
    if ready_flag is False:
        asr_detail = ""
        asr = payload.get("asr")
        if isinstance(asr, dict):
            asr_detail = f" asr.status={asr.get('status')} detail={asr.get('detail')}"
        raise VoiceUnavailable(f"voice_input_ready=false{asr_detail}")
    return _cache_ok(payload)


# ---------------------------------------------------------------------------
# HiveMind ASR / TTS bridges
# ---------------------------------------------------------------------------


def _multipart_audio_body(audio: bytes, *, filename: str, model: str) -> tuple[bytes, str]:
    """Build a multipart/form-data body that HiveMind's OpenAI-compatible
    ``/v1/audio/transcriptions`` endpoint accepts. We avoid `requests`
    because the rest of the gateway uses stdlib only."""
    boundary = f"----MS4VoiceBoundary{uuid.uuid4().hex}"
    mime = mimetypes.guess_type(filename)[0] or "audio/wav"
    parts = [
        f"--{boundary}\r\n".encode("utf-8"),
        f'Content-Disposition: form-data; name="model"\r\n\r\n{model}\r\n'.encode("utf-8"),
        f"--{boundary}\r\n".encode("utf-8"),
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode("utf-8"),
        f"Content-Type: {mime}\r\n\r\n".encode("utf-8"),
        audio,
        f"\r\n--{boundary}--\r\n".encode("utf-8"),
    ]
    body = b"".join(parts)
    return body, f"multipart/form-data; boundary={boundary}"


def transcribe(
    *,
    hivemind_url: str,
    audio: bytes,
    filename: str = "input.wav",
    model: str | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    """POST audio to HiveMind ``/v1/audio/transcriptions``."""
    if not audio:
        raise VoiceRequestError("audio body is empty")
    if len(audio) > MAX_AUDIO_BYTES:
        raise VoiceRequestError(
            f"audio body too large: {len(audio)} bytes > limit {MAX_AUDIO_BYTES}"
        )
    from .hivemind_state import hivemind_auth_headers

    body, content_type = _multipart_audio_body(audio, filename=filename, model=model or DEFAULT_TRANSCRIBE_MODEL)
    headers = {"Content-Type": content_type, "Accept": "application/json"}
    headers.update(hivemind_auth_headers())
    request = urllib.request.Request(
        f"{hivemind_url.rstrip('/')}/v1/audio/transcriptions",
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        raise VoiceUnavailable(
            f"HiveMind /v1/audio/transcriptions returned {exc.code}: "
            f"{exc.read().decode('utf-8', errors='replace')[:400]}"
        ) from exc
    except (urllib.error.URLError, OSError) as exc:
        raise VoiceUnavailable(f"HiveMind /v1/audio/transcriptions unreachable: {exc}") from exc
    text = ""
    if isinstance(payload, dict):
        text = str(payload.get("text") or payload.get("transcription") or "")
    return {
        "text": text,
        "model": model or DEFAULT_TRANSCRIBE_MODEL,
        "raw": payload if isinstance(payload, dict) else {"raw": payload},
    }


def synthesize(
    *,
    hivemind_url: str,
    text: str,
    model: str | None = None,
    voice: str | None = None,
    response_format: str | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    """POST text to HiveMind ``/v1/audio/speech``. Returns audio bytes
    plus the response content type."""
    cleaned = (text or "").strip()
    if not cleaned:
        raise VoiceRequestError("text body is empty")
    from .hivemind_state import hivemind_auth_headers

    body = json.dumps({
        "model": model or DEFAULT_TTS_MODEL,
        "voice": voice or DEFAULT_TTS_VOICE,
        "input": cleaned,
        "response_format": response_format or DEFAULT_TTS_FORMAT,
    }).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "audio/*"}
    headers.update(hivemind_auth_headers())
    request = urllib.request.Request(
        f"{hivemind_url.rstrip('/')}/v1/audio/speech",
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            audio_bytes = response.read()
            response_mime = response.headers.get("Content-Type", "audio/wav")
    except urllib.error.HTTPError as exc:
        raise VoiceUnavailable(
            f"HiveMind /v1/audio/speech returned {exc.code}: "
            f"{exc.read().decode('utf-8', errors='replace')[:400]}"
        ) from exc
    except (urllib.error.URLError, OSError) as exc:
        raise VoiceUnavailable(f"HiveMind /v1/audio/speech unreachable: {exc}") from exc
    if not audio_bytes:
        raise VoiceUnavailable("HiveMind /v1/audio/speech returned empty body")
    return {
        "audio_bytes": audio_bytes,
        "content_type": response_mime,
        "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
        "model": model or DEFAULT_TTS_MODEL,
        "voice": voice or DEFAULT_TTS_VOICE,
        "format": response_format or DEFAULT_TTS_FORMAT,
    }


# ---------------------------------------------------------------------------
# Combined PTT turn
# ---------------------------------------------------------------------------


def voice_ptt_turn(
    *,
    runner,
    audio: bytes,
    filename: str = "input.wav",
    session_id: str | None = None,
    model: str | None = None,
    depth_model: str | None = None,
    transcribe_model: str | None = None,
    tts_model: str | None = None,
    tts_voice: str | None = None,
    response_format: str | None = None,
) -> VoicePttResult:
    """Chained ASR → chat → TTS turn for the push-to-talk UI.

    ``runner`` is the live :class:`Ms4HermesRunner`. We check ASR
    readiness first (with a HiveMind-ASR fallback if MS3 is down), then
    run the chat path (which already injects the Face Lobe context block
    and bumps the conversation revision), then synthesize the reply.
    """
    check_voice_ready(runner.ms3_url, hivemind_url=runner.hivemind_url)
    asr_result = transcribe(
        hivemind_url=runner.hivemind_url,
        audio=audio,
        filename=filename,
        model=transcribe_model,
    )
    transcript = asr_result["text"].strip()
    if not transcript:
        raise VoiceRequestError(
            "transcription returned empty text; no chat turn dispatched"
        )
    _chat_kw: dict[str, Any] = {"session_id": session_id, "model": model}
    if depth_model:
        _chat_kw["depth_model"] = depth_model
    if _voice_brevity_enabled():
        _chat_kw["voice_mode"] = True
    chat_response = runner.chat(transcript, **_chat_kw)
    reply_text = (chat_response.get("text") or "").strip() or "(no reply text)"
    tts_result = synthesize(
        hivemind_url=runner.hivemind_url,
        text=reply_text,
        model=tts_model,
        voice=tts_voice,
        response_format=response_format,
    )
    return VoicePttResult(
        transcript=transcript,
        reply_text=reply_text,
        reply_audio_base64=tts_result["audio_base64"],
        reply_audio_mime=tts_result["content_type"],
        session_id=chat_response.get("session_id"),
        foreground_model=chat_response.get("face_lobe_model"),
        transcription_model=asr_result["model"],
        tts_model=tts_result["model"],
        grounding_source=chat_response.get("grounding_source"),
        router=chat_response.get("router"),
        dispatched_job=chat_response.get("dispatched_job"),
    )


# ---------------------------------------------------------------------------
# Streaming PTT turn (sentence-chunked TTS for ChatGPT-mobile parity)
# ---------------------------------------------------------------------------


_SENTENCE_END_RE = re.compile(r"([.!?])(\s+|$)")
_NEWLINE_BREAK_RE = re.compile(r"\n\n+")
# First-chunk-only break on natural inner punctuation. We don't break
# all chunks here because mid-sentence commas in the middle of a long
# reply create unnatural pauses; the first chunk gets the special
# treatment specifically to start audio sooner.
_FIRST_CHUNK_BREAK_RE = re.compile(r"[,;:—–-]\s+|[,;:]$")


# ---------------------------------------------------------------------------
# Audio duration helpers (shared by the reflex QA and the runtime gate)
# ---------------------------------------------------------------------------

_AUDIO_NORM_RE = re.compile(r"[^a-z0-9' ]+")


def wav_duration_secs(audio_bytes: bytes) -> float | None:
    """Exact duration of WAV bytes, or None if not parseable. A babbling
    TTS tail makes a clip far longer than the text warrants - the single
    most reliable, model-free garble signal."""
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as w:
            rate = w.getframerate()
            return (w.getnframes() / float(rate)) if rate else None
    except Exception:  # noqa: BLE001
        return None


def expected_speech_secs(text: str) -> float:
    """Generous upper bound on how long a clean spoken render of ``text``
    should take. Above this == a babble tail."""
    words = max(1, len(_AUDIO_NORM_RE.sub(" ", (text or "").lower()).split()))
    return words * 0.75 + 1.4


def trim_wav(audio_bytes: bytes, max_secs: float) -> bytes | None:
    """Return WAV bytes trimmed to the first ``max_secs`` (cutting a babble
    tail), preserving format. None if the input isn't parseable WAV."""
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as w:
            ch, sw, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
            keep = min(w.getnframes(), max(1, int(max_secs * rate)))
            frames = w.readframes(keep)
        out = io.BytesIO()
        with wave.open(out, "wb") as ww:
            ww.setnchannels(ch)
            ww.setsampwidth(sw)
            ww.setframerate(rate)
            ww.writeframes(frames)
        return out.getvalue()
    except Exception:  # noqa: BLE001
        return None


def _runtime_audio_gate_enabled() -> bool:
    """Trim a babble tail off a LIVE reply chunk before it plays. Instant,
    model-free (just WAV math). Default on; MS4_VOICE_RUNTIME_AUDIO_GATE=0
    to disable."""
    return os.environ.get("MS4_VOICE_RUNTIME_AUDIO_GATE", "1").strip().lower() not in {"0", "false", "no", "off"}


def _gate_chunk_audio(audio_bytes: bytes, content_type: str, text: str) -> tuple[bytes, bool]:
    """If a live chunk's WAV has a clear babble tail (well beyond what the
    text warrants), trim it. Conservative headroom so legit slightly-long
    speech is never cut. Returns (audio_bytes, trimmed)."""
    if "wav" not in (content_type or "").lower():
        return audio_bytes, False
    dur = wav_duration_secs(audio_bytes)
    if dur is None:
        return audio_bytes, False
    limit = expected_speech_secs(text) + 1.0  # generous headroom over the already-generous bound
    if dur > limit:
        trimmed = trim_wav(audio_bytes, limit)
        if trimmed:
            return trimmed, True
    return audio_bytes, False


def _emit_text_as_parallel_chunks(
    *, text: str, emit: Callable[[str, dict[str, Any]], bool], hivemind_url: str,
    tts_model: str | None, tts_voice: str | None, response_format: str | None,
    pool_size: int = TTS_POOL_SIZE, start_index: int = 0,
) -> int:
    """Sentence-chunk ``text`` and synthesize the chunks in PARALLEL,
    emitting ``audio_chunk`` events IN ORDER. Used by the ws_super
    zero-audio fallback so it's never one slow single-call synth (the
    'TTS not chunked, takes forever' bug). Returns the count emitted."""
    chunker = SentenceChunker()
    sentences = list(chunker.add(text or ""))
    tail = chunker.flush()
    if tail:
        sentences.append(tail)
    from .spoken_text_filter import sanitize_for_speech as _san
    pieces = [(_san(s).strip() if TTS_FILTER_ENABLED else s) for s in sentences]
    pieces = [p for p in pieces if p]
    if not pieces:
        return 0
    emitted = 0
    with ThreadPoolExecutor(max_workers=max(1, pool_size), thread_name_prefix="ms4-tts-fb") as ex:
        futures = [
            ex.submit(synthesize, hivemind_url=hivemind_url, text=p,
                      model=tts_model, voice=tts_voice, response_format=response_format)
            for p in pieces
        ]
        for i, fut in enumerate(futures):  # iterate in order -> emit in order
            try:
                res = fut.result()
                audio_bytes = res.get("audio_bytes") or b""
                if not audio_bytes:
                    continue
                mime = res.get("content_type") or "audio/wav"
                if _runtime_audio_gate_enabled():
                    audio_bytes, _ = _gate_chunk_audio(audio_bytes, mime, pieces[i])
                emit("audio_chunk", {
                    "index": start_index + i,
                    "text": pieces[i],
                    "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
                    "audio_mime": mime,
                    "tts_engine": "rest_fallback",
                })
                emitted += 1
            except Exception as exc:  # noqa: BLE001
                emit("audio_error", {"index": start_index + i, "error": str(exc)[:200]})
    return emitted


def _runtime_qa_enabled() -> bool:
    """Async full-QA (ASR + LLM judge) of the LIVE reply chunks AFTER they
    play, for observability. Opt-in (MS4_VOICE_RUNTIME_QA=1) because it
    spends ASR+LLM per chunk per turn; the synchronous duration gate above
    already catches the egregious garbles for free."""
    return os.environ.get("MS4_VOICE_RUNTIME_QA", "0").strip().lower() in {"1", "true", "yes", "on"}


def _async_verify_reply(samples: list[tuple[str, bytes]], hivemind_url: str) -> None:
    """Background round-trip QA of reply chunks: transcribe each back and
    judge it; log any garbles. Never blocks the turn, never raises."""
    try:
        from .canned_reflexes import validate_reflex_audio
    except Exception:  # noqa: BLE001
        return
    bad = 0
    for text, audio in samples:
        try:
            v = validate_reflex_audio(text, audio, hivemind_url)
            if not v.get("valid"):
                bad += 1
                log.warning(
                    "voice reply QA: chunk garbled - said=%r heard=%r reason=%s",
                    (text or "")[:60], (v.get("heard") or "")[:60], v.get("reason"),
                )
        except Exception:  # noqa: BLE001
            continue
    if bad:
        log.warning("voice reply QA: %d/%d reply chunks flagged garbled", bad, len(samples))


class SentenceChunker:
    """Consume streaming text deltas and emit chunks at speech boundaries.

    Strategy:

      1. **First-chunk acceleration.** Until the first chunk fires we
         look for either a sentence terminator OR the configured
         ``first_chunk_min_words`` count. Whichever comes first wins.
         This guarantees the user hears audio start within a small,
         deterministic window no matter what the model decides to say.
      2. **Subsequent chunks coalesce short sentences/lines** up to
         ``min_chunk_words`` (so a chatty/list-heavy reply doesn't fire a
         flurry of tiny TTS calls), then fire on sentence boundaries (``. ! ?``
         followed by whitespace) or on blank-line breaks.
      3. **Hard ceiling per chunk** at ``max_chunk_words`` so a
         run-on sentence (no punctuation) still gets chunked.
      4. **Flush** returns whatever's left at end-of-stream so the
         final unterminated fragment is still spoken.
    """

    def __init__(
        self,
        first_chunk_min_words: int = FIRST_CHUNK_MIN_WORDS,
        max_chunk_words: int = MAX_CHUNK_WORDS,
        min_chunk_words: int = MIN_CHUNK_WORDS,
    ) -> None:
        self.first_chunk_min_words = first_chunk_min_words
        self.max_chunk_words = max_chunk_words
        # Coalescing target for chunks after the first; clamp to >=1 and
        # never above the hard ceiling.
        self.min_chunk_words = max(1, min(min_chunk_words, max_chunk_words))
        self.buffer = ""
        self.chunk_count = 0

    def add(self, delta: str) -> list[str]:
        if not delta:
            return []
        self.buffer += delta
        out: list[str] = []
        while True:
            chunk = self._try_extract()
            if chunk is None:
                break
            if chunk:
                out.append(chunk)
                self.chunk_count += 1
        return out

    def _try_extract(self) -> str | None:
        """Try to extract one ready chunk. Returns None when nothing is
        ready yet; returns the chunk text otherwise.

        The FIRST chunk uses the fast path (ship ASAP for first-audio
        latency). SUBSEQUENT chunks coalesce short sentences/lines up to
        ``min_chunk_words`` so a chatty or list-heavy reply doesn't fire a
        flurry of tiny TTS calls.
        """
        if not self.buffer:
            return None
        if self.chunk_count > 0 and self.min_chunk_words > 1:
            return self._extract_coalesced()
        return self._extract_first_chunk()

    def _extract_coalesced(self) -> str | None:
        """Emit at the earliest sentence/paragraph boundary whose head
        reaches ``min_chunk_words``, merging shorter sentences/lines. If
        no boundary is big enough yet and we're under the hard ceiling,
        wait for more text (return None); the end-of-stream ``flush`` will
        emit any trailing short fragment."""
        search = 0
        while True:
            sm = _SENTENCE_END_RE.search(self.buffer, search)
            nm = _NEWLINE_BREAK_RE.search(self.buffer, search)
            if sm is not None and (nm is None or sm.end() <= nm.start()):
                head_end, consume_end = sm.end(), sm.end()
            elif nm is not None:
                head_end, consume_end = nm.start(), nm.end()
            else:
                break  # no more boundaries buffered yet
            if len(self.buffer[:head_end].split()) >= self.min_chunk_words:
                sentence = " ".join(self.buffer[:head_end].split())
                self.buffer = self.buffer[consume_end:].lstrip()
                return sentence if sentence else ""
            # Too short — look past this boundary to coalesce the next.
            search = consume_end
        # No boundary reached min_chunk_words. Honor the hard ceiling so a
        # long run-on still chunks; otherwise wait for more text.
        words = self.buffer.split()
        if len(words) >= self.max_chunk_words:
            cut_words = words[: self.max_chunk_words]
            head_len = self._reconstruct_head_length(cut_words)
            sentence = " ".join(self.buffer[:head_len].split())
            self.buffer = self.buffer[head_len:].lstrip()
            return sentence if sentence else ""
        return None

    def _extract_first_chunk(self) -> str | None:
        """Fast path for the first chunk (and per-sentence fallback when
        coalescing is disabled): ship at the first natural boundary."""
        # Pass 1: explicit sentence terminator.
        match = _SENTENCE_END_RE.search(self.buffer)
        if match is not None:
            end = match.end()
            sentence = self.buffer[:end].strip()
            if sentence:
                self.buffer = self.buffer[end:]
                return sentence
            # empty sentence (just punctuation) — keep going
            self.buffer = self.buffer[end:]
            return ""

        # Pass 2: paragraph break.
        nm = _NEWLINE_BREAK_RE.search(self.buffer)
        if nm is not None:
            cut = nm.start()
            sentence = self.buffer[:cut].strip()
            self.buffer = self.buffer[nm.end():]
            if sentence:
                return sentence
            return ""

        # Pass 3a: first-chunk acceleration — break on natural inner-
        # punctuation (comma / colon / em-dash / semicolon) so the
        # first chunk ships ASAP. Many model replies start "Looking
        # into that, give me a sec..." or "Yeah, the cluster is
        # idle." — breaking on the first comma cuts ~6-8 words off
        # the first chunk and slashes the in-order hold time the
        # REST engine pays.
        if self.chunk_count == 0:
            inner = _FIRST_CHUNK_BREAK_RE.search(self.buffer)
            if inner is not None:
                cut = inner.end()
                sentence = self.buffer[:cut].strip().rstrip(",;:—-").strip()
                if sentence and len(sentence.split()) >= self.first_chunk_min_words:
                    self.buffer = self.buffer[cut:].lstrip()
                    return sentence

        # Pass 3b: first-chunk min-words fallback. Emit early so audio
        # starts regardless of punctuation.
        words = self.buffer.split()
        if self.chunk_count == 0 and len(words) >= self.first_chunk_min_words:
            cut_words = words[: self.first_chunk_min_words]
            head_len = self._reconstruct_head_length(cut_words)
            sentence = self.buffer[:head_len].strip()
            self.buffer = self.buffer[head_len:].lstrip()
            return sentence

        # Pass 4: hard ceiling — don't let a single chunk grow forever.
        if len(words) >= self.max_chunk_words:
            cut_words = words[: self.max_chunk_words]
            head_len = self._reconstruct_head_length(cut_words)
            sentence = self.buffer[:head_len].strip()
            self.buffer = self.buffer[head_len:].lstrip()
            return sentence

        return None  # not ready

    def _reconstruct_head_length(self, words: list[str]) -> int:
        """Find the byte length in ``self.buffer`` that contains
        ``words`` and any trailing whitespace up to (but not into) the
        next word."""
        n = 0
        for w in words:
            idx = self.buffer.find(w, n)
            if idx < 0:
                return len(self.buffer)
            n = idx + len(w)
        # Include trailing whitespace so the next chunk starts clean.
        while n < len(self.buffer) and self.buffer[n].isspace():
            n += 1
        return n

    def flush(self) -> str | None:
        """End-of-stream: return whatever's left as one final chunk."""
        text = self.buffer.strip()
        self.buffer = ""
        if text:
            self.chunk_count += 1
            return text
        return None


@dataclass
class _PendingChunk:
    index: int
    text: str
    future: Future
    scheduled_at: float          # monotonic seconds since turn start
    tts_completed_at: float = 0  # set in the future done-callback
    emitted_at: float = 0        # set when the in-order emit fires


def voice_ptt_turn_stream(
    *,
    runner,
    audio: bytes,
    filename: str = "input.wav",
    session_id: str | None = None,
    model: str | None = None,
    depth_model: str | None = None,
    transcribe_model: str | None = None,
    tts_model: str | None = None,
    tts_voice: str | None = None,
    response_format: str | None = None,
    emit: Callable[[str, dict[str, Any]], bool],
    chunker: SentenceChunker | None = None,
    tts_pool_size: int = TTS_POOL_SIZE,
    chat_fn: Callable[..., dict[str, Any]] | None = None,
    transcribe_fn: Callable[..., dict[str, Any]] | None = None,
    synthesize_fn: Callable[..., dict[str, Any]] | None = None,
    engine: str | None = None,
    ws_engine_factory: Callable[..., Any] | None = None,
    client_alive: threading.Event | None = None,
) -> dict[str, Any]:
    """Streaming variant of :func:`voice_ptt_turn`.

    Calls ``emit(event_name, payload)`` for each progress event:

      * ``transcript`` once with the ASR result.
      * ``text_delta`` for every chat token (as Face Lobe streams).
      * ``audio_chunk`` per sentence, IN ORDER, as TTS completes.
      * ``status`` for phase changes (``transcribing``, ``thinking``).
      * ``audio_error`` if a single TTS call fails (the rest still flow).

    Returns a final metrics dict with per-stage durations and chunk
    counts. The HTTP handler appends a ``done`` SSE event with the
    return value.

    ``chat_fn`` / ``transcribe_fn`` / ``synthesize_fn`` are injection
    seams for tests; production uses ``runner.chat`` and the module
    HiveMind bridges.
    """
    chunker = chunker or SentenceChunker()
    t_total = time.monotonic()
    asr_fn = transcribe_fn or (lambda **kw: transcribe(**kw))
    tts_fn = synthesize_fn or (lambda **kw: synthesize(**kw))
    chat_call = chat_fn or runner.chat

    # Voice OBEYS the selected Face Lobe model (May 31 2026). The UI's
    # "Face Lobe model" dropdown applies to voice too; an empty/absent
    # model means the operator chose "Auto-pick (fast)" and the Face
    # Lobe picker runs. Earlier this path force-nulled the model to keep
    # voice fast, but that silently ignored the operator's explicit
    # choice (live complaint). The latency tradeoff is now the
    # operator's to make via the dropdown. Set MS4_VOICE_FORCE_AUTO=1 to
    # restore the old always-fast-picker behavior regardless of the
    # dropdown.
    if model and os.environ.get("MS4_VOICE_FORCE_AUTO", "").strip().lower() in {"1", "true", "yes", "on"}:
        log.info("voice turn forcing auto-pick (MS4_VOICE_FORCE_AUTO); ignoring model=%r", model)
        model = None

    # ---- Phase 1: fail-closed check + ASR ---------------------------------
    emit("status", {"phase": "transcribing"})
    check_voice_ready(runner.ms3_url, hivemind_url=runner.hivemind_url)
    t_asr_start = time.monotonic()
    asr_result = asr_fn(
        hivemind_url=runner.hivemind_url,
        audio=audio,
        filename=filename,
        model=transcribe_model,
    )
    transcript = (asr_result.get("text") or "").strip()
    asr_ms = int((time.monotonic() - t_asr_start) * 1000)
    # Speaker identification (best-effort). It processes the WHOLE
    # utterance, so on a long voice request it can take many seconds —
    # and it used to run fully inline here, stalling the turn (up to the
    # 15s identify timeout) BEFORE the chat even started. Now it runs on
    # a daemon thread and we wait at most MS4_VOICE_SPEAKER_ID_BUDGET_S;
    # if it doesn't resolve in that window the turn proceeds with no
    # label rather than making the user wait. Short utterances still
    # resolve in time and get labeled.
    # Remember the Face model this turn talks to so the keep-warm loop
    # keeps exactly that model resident between turns.
    record_face_model(model)
    # Speaker identification: fully async by default (off the critical
    # path). The transcript ships now with no label; a late `speaker`
    # event delivers it. Falls back to a bounded inline wait when
    # MS4_VOICE_SPEAKER_ID_ASYNC=0.
    if _voice_speaker_id_async():
        emit("transcript", {
            "text": transcript,
            "asr_ms": asr_ms,
            "model": asr_result.get("model"),
            "speaker": None,
        })
        _identify_speaker_async(runner.hivemind_url, audio, emit)
    else:
        speaker = _identify_speaker_bounded(runner.hivemind_url, audio, _voice_speaker_id_budget())
        emit("transcript", {
            "text": transcript,
            "asr_ms": asr_ms,
            "model": asr_result.get("model"),
            "speaker": speaker,
        })
    if not transcript:
        raise VoiceRequestError("transcription returned empty text; no chat turn dispatched")

    # Easter egg: "Honorable" -> BIA "WE ON GO". Checked BEFORE the normal
    # reply path so it can REPLACE the turn (arming ack, or the WE ON GO
    # drop). Returns None on a normal turn so nothing changes.
    if _voice_egg_enabled():
        _egg_result = handle_honorable_egg(
            emit=emit, hivemind_url=runner.hivemind_url,
            transcript=transcript, session_id=session_id, asr_ms=asr_ms,
        )
        if _egg_result is not None:
            return _egg_result

    # Context-aware "buying time" reflex: classify the utterance with a
    # tiny model and emit a `reflex` event naming a fitting, varied canned
    # acknowledgment. Runs on a daemon thread CONCURRENTLY with the (slow)
    # Face Lobe reply so it lands during the dead air. Fail-soft.
    if _voice_smart_reflex_enabled():
        threading.Thread(
            target=emit_smart_reflex,
            kwargs={"emit": emit, "hivemind_url": runner.hivemind_url,
                    "transcript": transcript, "session_id": session_id},
            name="ms4-smart-reflex", daemon=True,
        ).start()

    # ---- Phase 2 + 3: streaming chat + TTS --------------------------------
    selected_engine = (engine or DEFAULT_ENGINE).strip().lower()
    if selected_engine not in VALID_ENGINES:
        log.warning("unknown MS4_VOICE_TTS_ENGINE %r; falling back to 'rest'", selected_engine)
        selected_engine = "rest"

    if selected_engine == "ws_super":
        # If the WS engine fails to OPEN (TTS_SUPER GIM cold-loading,
        # transient cluster timeout, missing endpoint), fall back to
        # the REST engine instead of failing the whole turn. The
        # operator's voice already arrived; degrading the TTS engine
        # is strictly better than dropping their turn on the floor.
        # _run_ws_super_engine raises ``WsOpenFailed`` so we can tell
        # "WS couldn't open" apart from genuine mid-turn failures
        # (a mid-stream failure can't be re-driven against REST).
        try:
            return _run_ws_super_engine(
                runner=runner,
                transcript=transcript,
                session_id=session_id,
                model=model,
                depth_model=depth_model,
                tts_voice=tts_voice,
                asr_result=asr_result,
                asr_ms=asr_ms,
                tts_model=tts_model,
                chat_call=chat_call,
                emit=emit,
                t_total=t_total,
                ws_engine_factory=ws_engine_factory,
                client_alive=client_alive,
            )
        except WsOpenFailed as exc:
            log.warning("ws_super engine open failed (%s); falling back to REST", exc)
            emit("status", {
                "phase": "tts_engine_fallback",
                "from": "ws_super",
                "to": "rest",
                "reason": str(exc)[:200],
            })
            selected_engine = "rest"
            # Fall through to the REST path below.

    emit("status", {"phase": "thinking"})
    tts_executor = ThreadPoolExecutor(max_workers=tts_pool_size, thread_name_prefix="ms4-tts")
    pending: dict[int, _PendingChunk] = {}
    emit_lock = threading.Lock()
    next_emit_idx = [0]
    counters = {"audio_chunks": 0, "audio_errors": 0, "first_audio_at": None, "gated": 0}
    qa_samples: list[tuple[str, bytes]] = []  # (text, emitted audio) for async reply QA
    t_first_token: dict[str, float | None] = {"ms": None}
    chunk_index_counter = [0]
    chat_metrics: dict[str, Any] = {}

    chunk_timings: list[dict[str, Any]] = []
    # REST engine sanitizes per-sentence (since SentenceChunker already
    # split on sentence boundaries, we don't need streaming state). The
    # chunk_scheduled SSE event reports the SANITIZED text so the UI
    # debug overlay reflects what TTS actually saw.
    from .spoken_text_filter import sanitize_for_speech as _sanitize

    def _schedule(text: str) -> None:
        speakable = _sanitize(text).strip() if TTS_FILTER_ENABLED else text
        if not speakable:
            # Sanitizer ate the whole chunk (e.g. it was just a bullet
            # marker or an emoji). Don't schedule an empty TTS call.
            return
        idx = chunk_index_counter[0]
        chunk_index_counter[0] += 1
        scheduled_at = time.monotonic() - t_total
        emit("chunk_scheduled", {"index": idx, "text": speakable, "scheduled_ms": int(scheduled_at * 1000)})
        fut = tts_executor.submit(
            tts_fn,
            hivemind_url=runner.hivemind_url,
            text=speakable,
            model=tts_model,
            voice=tts_voice,
            response_format=response_format,
        )
        pending[idx] = _PendingChunk(index=idx, text=speakable, future=fut, scheduled_at=scheduled_at)

        def _on_done(_f: Future, _idx: int = idx) -> None:
            if _idx in pending:
                pending[_idx].tts_completed_at = time.monotonic() - t_total
            _try_emit_ready()

        fut.add_done_callback(_on_done)

    def _try_emit_ready() -> None:
        with emit_lock:
            while next_emit_idx[0] in pending:
                chunk = pending[next_emit_idx[0]]
                if not chunk.future.done():
                    return
                try:
                    tts_result = chunk.future.result()
                    if counters["first_audio_at"] is None:
                        counters["first_audio_at"] = int((time.monotonic() - t_total) * 1000)
                    chunk.emitted_at = time.monotonic() - t_total
                    # Per-chunk timings exposed so the operator can see
                    # (a) actual TTS latency on HiveMind's side
                    # (b) how long the in-order emit guarantee held a chunk
                    tts_ms = int((chunk.tts_completed_at - chunk.scheduled_at) * 1000)
                    held_ms = max(0, int((chunk.emitted_at - chunk.tts_completed_at) * 1000))
                    chunk_timings.append({
                        "index": chunk.index,
                        "text_len": len(chunk.text),
                        "scheduled_ms": int(chunk.scheduled_at * 1000),
                        "tts_completed_ms": int(chunk.tts_completed_at * 1000),
                        "emitted_ms": int(chunk.emitted_at * 1000),
                        "tts_ms": tts_ms,
                        "held_for_inorder_ms": held_ms,
                    })
                    # Runtime garble gate: trim a babble tail off this live
                    # chunk before it plays (instant, model-free). The async
                    # verifier (opt-in) double-checks what actually played.
                    audio_bytes = tts_result["audio_bytes"]
                    audio_mime = tts_result["content_type"]
                    audio_b64 = tts_result["audio_base64"]
                    if _runtime_audio_gate_enabled():
                        gated_bytes, trimmed = _gate_chunk_audio(audio_bytes, audio_mime, chunk.text)
                        if trimmed:
                            audio_bytes = gated_bytes
                            audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
                            counters["gated"] += 1
                            log.info("runtime gate trimmed a babble tail on chunk %d (%r)",
                                     chunk.index, chunk.text[:30])
                    if _runtime_qa_enabled():
                        qa_samples.append((chunk.text, audio_bytes))
                    emit("audio_chunk", {
                        "index": chunk.index,
                        "text": chunk.text,
                        "audio_base64": audio_b64,
                        "audio_mime": audio_mime,
                        "tts_ms": tts_ms,
                        "held_for_inorder_ms": held_ms,
                    })
                    counters["audio_chunks"] += 1
                except Exception as exc:
                    emit("audio_error", {"index": chunk.index, "text": chunk.text, "error": str(exc)})
                    counters["audio_errors"] += 1
                del pending[next_emit_idx[0]]
                next_emit_idx[0] += 1

    def _on_delta(delta: str) -> None:
        if t_first_token["ms"] is None and delta.strip():
            t_first_token["ms"] = int((time.monotonic() - t_total) * 1000)
        emit("text_delta", {"text": delta})
        ready = chunker.add(delta)
        for sentence in ready:
            _schedule(sentence)

    t_chat_start = time.monotonic()
    try:
        _chat_kw: dict[str, Any] = {"session_id": session_id, "model": model}
        if depth_model:
            _chat_kw["depth_model"] = depth_model
        if _voice_brevity_enabled():
            _chat_kw["voice_mode"] = True
        # First-token watchdog: a stalled Face model (e.g. a heavy cloud
        # model selected for voice) raises FaceLobeStalled instead of
        # hanging the SSE stream forever; the server turns it into an
        # error event and the UI plays the canned error reflex.
        chat_response = _run_facechat_guarded(
            chat_call=chat_call,
            transcript=transcript,
            chat_kwargs=_chat_kw,
            on_delta=_on_delta,
            first_token_timeout=_voice_first_token_timeout(),
            model=model,
        )
    except Exception:
        tts_executor.shutdown(wait=False)
        raise
    chat_ms = int((time.monotonic() - t_chat_start) * 1000)
    chat_metrics = (chat_response or {}).get("metrics") or {}

    # Flush any trailing buffer into one final chunk so the last
    # unterminated fragment still gets spoken.
    tail = chunker.flush()
    if tail:
        _schedule(tail)

    # Wait for all chunks to flush through (poll lightly — the
    # done-callbacks do the emitting).
    deadline = time.monotonic() + 60.0
    while next_emit_idx[0] < chunk_index_counter[0]:
        if time.monotonic() > deadline:
            log.warning("voice stream timed out waiting on TTS chunks")
            break
        time.sleep(0.05)
    tts_executor.shutdown(wait=False)

    # Async full-QA of what actually played (opt-in): transcribe the reply
    # chunks back and judge them in the background, logging any garbles. Never
    # blocks the turn.
    if _runtime_qa_enabled() and qa_samples:
        threading.Thread(
            target=_async_verify_reply, args=(list(qa_samples), runner.hivemind_url),
            daemon=True, name="ms4-reply-qa",
        ).start()

    total_ms = int((time.monotonic() - t_total) * 1000)
    reply_text = (chat_response.get("text") or "").strip() if isinstance(chat_response, dict) else ""
    # Parallelism summary so the operator can see whether HiveMind TTS
    # is actually running concurrent: when truly parallel, the sum of
    # per-chunk TTS times is significantly greater than the wall-clock
    # span from first scheduled to last completed.
    parallelism: dict[str, Any] = {}
    if chunk_timings:
        sum_tts = sum(c["tts_ms"] for c in chunk_timings)
        first_sched = min(c["scheduled_ms"] for c in chunk_timings)
        last_done = max(c["tts_completed_ms"] for c in chunk_timings)
        wall = max(1, last_done - first_sched)
        parallelism = {
            "sum_tts_ms": sum_tts,
            "wall_ms": wall,
            "speedup_ratio": round(sum_tts / wall, 2),  # 1.0 = serial; 3.0 = 3 parallel
            "any_held_for_inorder": any(c["held_for_inorder_ms"] > 50 for c in chunk_timings),
            "max_held_for_inorder_ms": max((c["held_for_inorder_ms"] for c in chunk_timings), default=0),
        }
    return {
        "transcript": transcript,
        "reply_text": reply_text,
        "session_id": chat_response.get("session_id") if isinstance(chat_response, dict) else None,
        "foreground_model": chat_response.get("face_lobe_model") if isinstance(chat_response, dict) else None,
        "router": chat_response.get("router") if isinstance(chat_response, dict) else None,
        "dispatched_job": chat_response.get("dispatched_job") if isinstance(chat_response, dict) else None,
        "grounding_source": chat_response.get("grounding_source") if isinstance(chat_response, dict) else None,
        "transcription_model": asr_result.get("model"),
        "tts_model": tts_model or DEFAULT_TTS_MODEL,
        "metrics": {
            "schema": "Ms4VoiceStreamMetrics.v1",
            "asr_ms": asr_ms,
            "chat_ms": chat_ms,
            "total_ms": total_ms,
            "first_text_token_ms": t_first_token["ms"],
            "first_audio_chunk_ms": counters["first_audio_at"],
            "audio_chunks": counters["audio_chunks"],
            "audio_errors": counters["audio_errors"],
            "runtime_gated": counters["gated"],
            "chunks": chunk_timings,
            "tts_parallelism": parallelism,
            "chat_metrics": chat_metrics,
        },
    }


# ---------------------------------------------------------------------------
# WS-engine variant (HiveMind TTS_SUPER /stream-input)
# ---------------------------------------------------------------------------


def _run_ws_super_engine(
    *,
    runner,
    transcript: str,
    session_id: str | None,
    model: str | None,
    depth_model: str | None = None,
    tts_voice: str | None,
    asr_result: dict[str, Any],
    asr_ms: int,
    tts_model: str | None,
    chat_call: Callable[..., dict[str, Any]],
    emit: Callable[[str, dict[str, Any]], bool],
    t_total: float,
    ws_engine_factory: Callable[..., Any] | None = None,
    client_alive: threading.Event | None = None,
) -> dict[str, Any]:
    """The ``engine="ws_super"`` branch of voice_ptt_turn_stream.

    Open one WebSocket to HiveMind's TTS_SUPER ``stream-input``,
    push every Face Lobe ``text_delta`` into it, flush at end of chat,
    wait for ``isFinal``. The WS server emits audio chunks as it
    synthesizes; the engine emits them as SSE ``audio_chunk`` events
    so the existing browser path consumes them unchanged.

    Latency win (measured live): first audio drops from ~10s to
    ~2.5s, total drops from ~25s to ~6s for a 4-sentence reply.
    """
    # Local import so the REST path never has to install websockets.
    if ws_engine_factory is None:
        from .tts_super_ws import TtsSuperWsEngine
        ws_engine_factory = TtsSuperWsEngine

    emit("status", {"phase": "thinking", "engine": "ws_super"})

    engine = ws_engine_factory(
        hivemind_url=runner.hivemind_url,
        voice=tts_voice,
        emit=emit,
        t_start=t_total,
    )
    chat_response: dict[str, Any] = {}
    chat_metrics: dict[str, Any] = {}
    t_first_token: dict[str, float | None] = {"ms": None}
    t_chat_start = time.monotonic()
    try:
        engine.open()
    except Exception as exc:
        # Don't emit ``error`` here — that's the SSE terminal-failure
        # event the UI shows as a red banner. The caller will fall
        # back to the REST engine and the user will still get audio.
        # Emit a status note instead so anyone watching the SSE feed
        # sees the engine switch.
        log.warning("TTS_SUPER WS open failed: %s", exc)
        emit("status", {
            "phase": "tts_engine_open_failed",
            "engine": "ws_super",
            "error": str(exc)[:200],
        })
        raise WsOpenFailed(f"failed to open TTS_SUPER WS: {exc}") from exc

    # The SpokenTextFilter sits between the chat token stream and the
    # WS TTS engine: the UI gets the original markdown for visual
    # rendering, but the TTS engine only sees speech-safe text. The
    # filter is stateful so multi-token patterns like opening/closing
    # ** survive across delta boundaries.
    from .spoken_text_filter import SpokenTextFilter
    tts_filter = SpokenTextFilter() if TTS_FILTER_ENABLED else None

    def _on_delta(delta: str) -> None:
        if t_first_token["ms"] is None and delta.strip():
            t_first_token["ms"] = int((time.monotonic() - t_total) * 1000)
        emit("text_delta", {"text": delta})  # raw to UI
        if tts_filter is None:
            engine.push(delta)
            return
        speakable = tts_filter.push(delta)
        if speakable:
            engine.push(speakable)

    chat_response = None
    chat_metrics: dict[str, Any] = {}
    try:
        _chat_kw: dict[str, Any] = {"session_id": session_id, "model": model}
        if depth_model:
            _chat_kw["depth_model"] = depth_model
        if _voice_brevity_enabled():
            _chat_kw["voice_mode"] = True
        # First-token watchdog (same as the REST path): a stalled Face
        # model raises FaceLobeStalled rather than hanging the WS turn.
        chat_response = _run_facechat_guarded(
            chat_call=chat_call,
            transcript=transcript,
            chat_kwargs=_chat_kw,
            on_delta=_on_delta,
            first_token_timeout=_voice_first_token_timeout(),
            model=model,
        )
        chat_metrics = (chat_response or {}).get("metrics") or {}
    except FaceLobeStalled:
        # Close the WS engine and fail the turn over to the error reflex.
        # Do NOT fall back to REST — the same model would stall there too.
        try:
            engine.close()
        except Exception:
            pass
        raise
    finally:
        # Drain the filter so any tail-text the model emitted after the
        # last sentence boundary still gets spoken (skip when we bailed
        # on a stall — chat_response stays None then).
        if chat_response is not None:
            if tts_filter is not None:
                tail = tts_filter.flush()
                if tail:
                    engine.push(tail)
            engine.flush()

    # Barge-in: if the client connection died while chat was running,
    # don't bother waiting for HiveMind to finish synthesizing audio
    # nobody is going to hear. Close the WS immediately and bail.
    if client_alive is not None and not client_alive.is_set():
        log.info("ws_super engine: client disconnected mid-turn; closing without wait_for_final")
        engine.close()
    else:
        # Wait for isFinal with TWO budgets:
        #
        #   * first-audio budget: if no audio chunk has arrived within
        #     ``MS4_TTS_WS_FIRST_AUDIO_TIMEOUT`` (default 15s) AFTER
        #     chat has finished flushing, HiveMind's TTS_SUPER GIM is
        #     almost certainly stuck. Bail early so the REST fallback
        #     can take over within seconds instead of after 60s.
        #
        #   * overall budget: total wait_for_final from now (default
        #     60s) for the case where chunks ARE streaming in but
        #     synthesizing the tail takes a while.
        #
        # Live evidence (May 26 2026): operator hit a turn where chat
        # finished in <1s, the WS engine then waited the full 60s
        # without emitting a single chunk before the fallback kicked
        # in. With the early bail, that becomes ~15s end-to-end
        # because REST takes over the moment we know WS is dead.
        first_audio_timeout = float(os.environ.get("MS4_TTS_WS_FIRST_AUDIO_TIMEOUT", "6"))
        overall_timeout = float(os.environ.get("MS4_TTS_WS_FINAL_TIMEOUT", "60"))
        poll_interval = 0.5
        t_wait_start = time.monotonic()
        while True:
            elapsed = time.monotonic() - t_wait_start
            if engine.wait_for_final(timeout=poll_interval):
                break  # isFinal arrived — synthesis complete
            cur_metrics = engine.metrics()
            chunks_so_far = cur_metrics.get("chunks_emitted") or 0
            # Early bail: no audio yet AND past the first-audio budget.
            if chunks_so_far == 0 and elapsed >= first_audio_timeout:
                log.warning(
                    "ws_super engine: bailing early after %ds with 0 audio chunks "
                    "(first_audio_timeout=%ds); REST fallback will take over",
                    int(elapsed), int(first_audio_timeout),
                )
                break
            # Hard ceiling.
            if elapsed >= overall_timeout:
                log.warning(
                    "ws_super engine: wait_for_final hard timeout after %ds (%d chunks emitted)",
                    int(elapsed), chunks_so_far,
                )
                break
            # Also bail if the WS reader has reported a fatal error
            # (keepalive timeout, internal error) — no point waiting
            # on a connection that's already torn down.
            if cur_metrics.get("error") and chunks_so_far == 0:
                log.warning(
                    "ws_super engine: bailing after %ds — reader reported error=%s with 0 audio chunks",
                    int(elapsed), str(cur_metrics.get("error"))[:120],
                )
                break
        engine.close()

    chat_ms = int((time.monotonic() - t_chat_start) * 1000)
    eng_metrics = engine.metrics()
    reply_text = (chat_response.get("text") or "").strip() if isinstance(chat_response, dict) else ""

    # WS-emitted-zero-audio fallback (May 26 2026): on long replies
    # the WS engine sometimes finishes with chunks_emitted=0 and a
    # populated error (observed live: 613 chat chunks streamed, 0
    # audio chunks emitted, 1 error). If that happens AND we still
    # have reply_text AND the client is alive, fall back to a single
    # REST /v1/audio/speech call on the full text so the user hears
    # *something* instead of silence + a red banner.
    chunks_emitted = eng_metrics.get("chunks_emitted") or 0
    if (
        chunks_emitted == 0
        and reply_text
        and (client_alive is None or client_alive.is_set())
    ):
        log.warning(
            "ws_super engine emitted 0 audio chunks (error=%s); "
            "falling back to single REST TTS call on %d chars of reply text",
            eng_metrics.get("error"),
            len(reply_text),
        )
        emit("status", {
            "phase": "tts_engine_fallback_after_zero_audio",
            "from": "ws_super",
            "to": "rest",
            "ws_error": str(eng_metrics.get("error") or "0 chunks emitted")[:200],
        })
        # Fall back to the SENTENCE-CHUNKED, PARALLEL REST path (not a
        # single slow synth on the whole reply) so first audio is fast and
        # the reply streams in chunks.
        if reply_text.strip():
            try:
                chunks_emitted = _emit_text_as_parallel_chunks(
                    text=reply_text, emit=emit, hivemind_url=runner.hivemind_url,
                    tts_model=tts_model, tts_voice=tts_voice, response_format=DEFAULT_TTS_FORMAT,
                )
            except Exception as exc:
                log.error("REST chunked fallback failed: %s", exc)
                emit("audio_error", {"index": 0, "error": str(exc)[:200]})

    total_ms = int((time.monotonic() - t_total) * 1000)
    return {
        "transcript": transcript,
        "reply_text": reply_text,
        "session_id": chat_response.get("session_id") if isinstance(chat_response, dict) else None,
        "foreground_model": chat_response.get("face_lobe_model") if isinstance(chat_response, dict) else None,
        "router": chat_response.get("router") if isinstance(chat_response, dict) else None,
        "dispatched_job": chat_response.get("dispatched_job") if isinstance(chat_response, dict) else None,
        "grounding_source": chat_response.get("grounding_source") if isinstance(chat_response, dict) else None,
        "transcription_model": asr_result.get("model"),
        "tts_model": tts_model or DEFAULT_TTS_MODEL,
        "metrics": {
            "schema": "Ms4VoiceStreamMetrics.v1",
            "engine": "ws_super",
            "asr_ms": asr_ms,
            "chat_ms": chat_ms,
            "total_ms": total_ms,
            "first_text_token_ms": t_first_token["ms"],
            "first_audio_chunk_ms": eng_metrics.get("first_audio_ms"),
            "audio_chunks": chunks_emitted,
            "audio_errors": 1 if eng_metrics.get("error") else 0,
            "rest_fallback_used": chunks_emitted == 1 and (eng_metrics.get("chunks_emitted") or 0) == 0,
            "chunks": [],   # WS engine doesn't expose per-chunk text grouping
            "tts_parallelism": {  # not applicable — single stream
                "engine": "ws_super",
                "speedup_ratio": None,
                "any_held_for_inorder": False,
                "max_held_for_inorder_ms": 0,
            },
            "tts_engine": eng_metrics,
            "chat_metrics": chat_metrics,
        },
    }


# ---------------------------------------------------------------------------
# ASR pre-warm (paid once on gateway boot so the first turn doesn't cold-load)
# ---------------------------------------------------------------------------


def build_silent_wav(seconds: float = 0.2, rate: int = 16000) -> bytes:
    """Build a tiny mono 16-bit PCM WAV of pure silence — used to
    pre-warm whisper without sending real audio."""
    n_frames = max(1, int(seconds * rate))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(struct.pack("<" + "h" * n_frames, *([0] * n_frames)))
    return buf.getvalue()


def prewarm_asr(*, hivemind_url: str, model: str | None = None, timeout: int = 60) -> dict[str, Any]:
    """Hit HiveMind ASR with a tiny silent WAV so the model is warm
    when the first real voice turn arrives. Idempotent and safe to
    call on boot."""
    try:
        return transcribe(
            hivemind_url=hivemind_url,
            audio=build_silent_wav(),
            filename="prewarm.wav",
            model=model,
            timeout=timeout,
        )
    except Exception as exc:
        log.warning("ASR pre-warm failed (safe to ignore on cold cluster): %s", exc)
        return {"text": "", "error": str(exc), "warmed": False}


def prewarm_face_lobe_model(
    *,
    hivemind_url: str,
    model: str | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    """Send a single-token completion request so the Face Lobe model
    is hot when the first real chat turn arrives. The picker has
    already cached the catalog by the time this runs."""
    try:
        from .face_lobe_chat import FaceLobeChat
        chat = FaceLobeChat(hivemind_url=hivemind_url, http_timeout=timeout)
        from machine_spirit_4.double_agent.model_picker import choose_foreground_model
        if model is None:
            try:
                choice = choose_foreground_model(hivemind_url=hivemind_url, force_refresh=True)
                model = choice.model_id
            except Exception:
                model = os.environ.get("MS4_DEFAULT_MODEL", "qwen3-coder-next:latest")
        result = chat.chat(
            "Reply with the single word: ready.",
            session_id=f"prewarm-{uuid.uuid4().hex[:8]}",
            model=model,
            stream_callback=None,
            extra_system=None,
        )
        return {"model": model, "reply_len": len(result.get("text") or ""), "warmed": True}
    except Exception as exc:
        log.warning("Face Lobe pre-warm failed (safe to ignore on cold cluster): %s", exc)
        return {"error": str(exc), "warmed": False}


def provision_tts_replicas(
    *,
    hivemind_url: str,
    target: int | None = None,
    job_type: str = "TTS_SUPER",
    timeout: int = 30,
) -> dict[str, Any]:
    """Ask HiveMind to run one TTS GIM replica per free GPU and
    round-robin concurrent ``/v1/audio/speech`` across them
    (``POST /provision/tts/scale``, shipped by HiveMind 2026-06-02).

    Why MS4 calls this: MS4 already fires per-chunk TTS in parallel
    (``TTS_POOL_SIZE``), but that only fans out across GPUs if HiveMind
    has multiple TTS replicas provisioned. Measured live: with 2 replicas
    a 6-wide burst hit ~4.4x in-flight parallelism (vs ~1x on the old
    single-replica/serialized path). Calling this on boot — and
    periodically, since it's idempotent and picks up GPUs as nodes are
    added — keeps TTS as concurrent as the hardware allows.

    ``target=None`` lets HiveMind pick (one replica per free GPU).
    Best-effort: on a cluster without the endpoint (older binary) this
    logs and returns ``{"ok": False}`` — voice still works via the
    single replica. Gated by the caller (see the boot loop)."""
    from .hivemind_state import hivemind_auth_headers

    body: dict[str, Any] = {"job_type": job_type}
    if target is not None:
        body["target"] = int(target)
    data = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    headers.update(hivemind_auth_headers())
    request = urllib.request.Request(
        f"{hivemind_url.rstrip('/')}/provision/tts/scale",
        data=data,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8", "replace"))
        payload.setdefault("ok", True)
        return payload
    except urllib.error.HTTPError as exc:
        # 404 => older gateway without the scale endpoint. Not an error
        # for us; the single replica still serves TTS.
        log.info("TTS scale-out unavailable (HTTP %s); single-replica TTS in effect", exc.code)
        return {"ok": False, "status": exc.code}
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        log.info("TTS scale-out request failed (safe to ignore): %s", exc)
        return {"ok": False, "error": str(exc)}


def prewarm_tts(
    *,
    hivemind_url: str,
    model: str | None = None,
    voice: str | None = None,
    response_format: str | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    """Hit HiveMind TTS with a tiny phrase to warm the TTS model. The
    first audio_chunk in a real voice turn is critical for perceived
    latency; cold-loading TTS adds 3-5s to that figure live."""
    try:
        return synthesize(
            hivemind_url=hivemind_url,
            text="Ready.",
            model=model,
            voice=voice,
            response_format=response_format,
            timeout=timeout,
        )
    except Exception as exc:
        log.warning("TTS pre-warm failed (safe to ignore on cold cluster): %s", exc)
        return {"audio_bytes": b"", "error": str(exc), "warmed": False}


def prewarm_tts_super_ws(
    *,
    hivemind_url: str,
    voice: str | None = None,
    timeout: float = 5.0,
) -> dict[str, Any]:
    """Open + close a TTS_SUPER WebSocket so the first real voice turn
    doesn't pay TCP/TLS handshake + WS upgrade + HiveMind-side
    GIM-allocation cost. We push a single "Ready." token through so
    the underlying GIM also loads its model weights, then flush and
    close. Best-effort — failures here are logged and ignored
    (operators on clusters without TTS_SUPER provisioned still get
    fast cold-paths via the REST engine fallback).
    """
    try:
        from .tts_super_ws import TtsSuperWsEngine
    except Exception as exc:
        log.warning("TTS_SUPER WS pre-warm: import failed (%s)", exc)
        return {"warmed": False, "error": str(exc)}
    t0 = time.monotonic()
    try:
        engine = TtsSuperWsEngine(
            hivemind_url=hivemind_url,
            voice=voice or DEFAULT_TTS_VOICE,
            emit=lambda _name, _payload: True,  # swallow events
            t_start=t0,
            open_timeout=timeout,
        )
        engine.open()
        engine.push("Ready.")
        engine.flush()
        # Don't wait for isFinal — the synthesis can complete in the
        # background; we only care that the GIM is warm. Closing the
        # WS shortly after flush is the documented pattern.
        engine.wait_for_final(timeout=timeout)
        engine.close()
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        return {"warmed": True, "elapsed_ms": elapsed_ms}
    except Exception as exc:
        log.warning("TTS_SUPER WS pre-warm failed (safe to ignore): %s", exc)
        return {"warmed": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Multipart parsing helper (kept here so the gateway server stays stdlib-only)
# ---------------------------------------------------------------------------


def parse_audio_request(content_type: str, body: bytes) -> tuple[bytes, str]:
    """Return ``(audio_bytes, filename)`` from a request body.

    Supports two shapes:
      * ``multipart/form-data`` with a ``file`` part (browser <input>).
      * Raw audio bytes with ``Content-Type: audio/*`` (curl/test
        scripts and some MediaRecorder pipelines).
    """
    ct = (content_type or "").lower()
    if not body:
        raise VoiceRequestError("request body is empty")
    if ct.startswith("multipart/form-data"):
        return _parse_multipart(content_type, body)
    if ct.startswith("audio/") or ct == "application/octet-stream":
        ext = mimetypes.guess_extension(ct.split(";", 1)[0].strip()) or ".bin"
        return body, f"input{ext}"
    raise VoiceRequestError(
        f"unsupported Content-Type: {content_type!r}; expected multipart/form-data or audio/*"
    )


def _parse_multipart(content_type: str, body: bytes) -> tuple[bytes, str]:
    boundary_param = None
    for part in content_type.split(";"):
        part = part.strip()
        if part.startswith("boundary="):
            boundary_param = part.split("=", 1)[1].strip()
            if boundary_param.startswith('"') and boundary_param.endswith('"'):
                boundary_param = boundary_param[1:-1]
            break
    if not boundary_param:
        raise VoiceRequestError("multipart/form-data missing boundary parameter")
    boundary = ("--" + boundary_param).encode("utf-8")
    sections = body.split(boundary)
    for section in sections:
        if not section or section in (b"--\r\n", b"--"):
            continue
        header_end = section.find(b"\r\n\r\n")
        if header_end == -1:
            continue
        headers_blob = section[:header_end].decode("utf-8", errors="replace").lower()
        if 'name="file"' not in headers_blob:
            continue
        filename = "input.wav"
        # Pull filename if present.
        marker = 'filename="'
        if marker in headers_blob:
            idx = headers_blob.index(marker) + len(marker)
            end = headers_blob.index('"', idx)
            filename = headers_blob[idx:end] or filename
        payload = section[header_end + 4:]
        # Strip trailing CRLF.
        if payload.endswith(b"\r\n"):
            payload = payload[:-2]
        return payload, filename
    raise VoiceRequestError("multipart body did not contain a file part named 'file'")
