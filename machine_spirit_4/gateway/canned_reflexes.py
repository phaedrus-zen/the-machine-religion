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

import hashlib
import json
import logging
import os
import random
import re
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .voice import (
    DEFAULT_TTS_FORMAT, DEFAULT_TTS_MODEL, DEFAULT_TTS_VOICE,
    expected_speech_secs, synthesize, transcribe, wav_duration_secs,
)


log = logging.getLogger("ms4.gateway.canned_reflexes")
_REFLEX_IO_LOCK = threading.RLock()


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


CATEGORY_ACK = "ack"
CATEGORY_THINKING = "thinking"
CATEGORY_ERROR = "error"
CATEGORY_CONFIRM = "confirm"
CATEGORY_IDENTITY = "identity"

# Intent/sentiment categories (Jun 1 2026) for the post-utterance
# "buying time" acknowledgment. The tiny-model classifier in voice.py
# maps the user's utterance to one of these intents; the UI plays a
# varied phrase from the matching set the moment ASR finishes, while the
# (slow) Face Lobe reply generates. These play AFTER the user stops
# speaking, never over them.
CATEGORY_Q = "intent_question"
CATEGORY_REQUEST = "intent_request"
CATEGORY_GRATITUDE = "intent_gratitude"
CATEGORY_GREETING = "intent_greeting"
CATEGORY_STATEMENT = "intent_statement"
CATEGORY_CORRECTION = "intent_correction"
CATEGORY_AFFIRM = "intent_affirm"
# Easter-egg category (the "Honorable -> WE ON GO" bit). The real bit
# plays a clip; this TTS line is only the fallback if the clip is missing.
CATEGORY_EGG = "egg"


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
    # ---- intent-routed "buying time" acks (post-utterance, Jun 1 2026) -----
    # Chosen by the tiny sentiment classifier the moment ASR finishes, with
    # no-repeat variety, to cover the dead air while the reply generates.
    Reflex("q_let_me_look", "Let me look into that.", CATEGORY_Q, "Question intent."),
    Reflex("q_good_question", "Good question, one sec.", CATEGORY_Q),
    Reflex("q_let_me_check", "Let me check on that for you.", CATEGORY_Q),
    Reflex("q_digging_in", "Alright, digging into that now.", CATEGORY_Q),

    Reflex("req_on_it", "On it.", CATEGORY_REQUEST, "Request/command intent."),
    Reflex("req_getting_that", "Sure, getting that going.", CATEGORY_REQUEST),
    Reflex("req_right_away", "Right away.", CATEGORY_REQUEST),
    Reflex("req_handling", "Okay, handling that now.", CATEGORY_REQUEST),

    Reflex("grat_anytime", "Anytime!", CATEGORY_GRATITUDE, "Gratitude intent."),
    Reflex("grat_of_course", "Of course.", CATEGORY_GRATITUDE),
    Reflex("grat_happy", "Happy to help.", CATEGORY_GRATITUDE),
    Reflex("grat_youbet", "You bet.", CATEGORY_GRATITUDE),

    Reflex("greet_hey", "Hey!", CATEGORY_GREETING, "Greeting intent."),
    Reflex("greet_hi_there", "Hi there.", CATEGORY_GREETING),
    Reflex("greet_whats_up", "Hey, what's up?", CATEGORY_GREETING),
    Reflex("greet_good_to_hear", "Good to hear from you.", CATEGORY_GREETING),

    Reflex("stmt_mhm", "Mhm.", CATEGORY_STATEMENT, "Statement/chit-chat intent."),
    Reflex("stmt_i_hear_you", "I hear you.", CATEGORY_STATEMENT),
    Reflex("stmt_right", "Right.", CATEGORY_STATEMENT),
    Reflex("stmt_makes_sense", "Makes sense.", CATEGORY_STATEMENT),

    Reflex("corr_got_it_fix", "Got it, let me fix that.", CATEGORY_CORRECTION, "Correction/negative intent."),
    Reflex("corr_my_bad", "My bad, one sec.", CATEGORY_CORRECTION),
    Reflex("corr_understood", "Understood, adjusting.", CATEGORY_CORRECTION),

    Reflex("affirm_great", "Great.", CATEGORY_AFFIRM, "Affirmation/agreement intent."),
    Reflex("affirm_sounds_good", "Sounds good.", CATEGORY_AFFIRM),
    Reflex("affirm_perfect", "Perfect.", CATEGORY_AFFIRM),
    # ---- easter egg fallback (spoken if the WE ON GO clip is missing) ------
    Reflex("egg_honorable", "We on go!", CATEGORY_EGG, "Spoken fallback for the Honorable easter egg."),
)


# Intent label (tiny classifier output) -> reflex category.
INTENT_TO_CATEGORY: dict[str, str] = {
    "question": CATEGORY_Q,
    "request": CATEGORY_REQUEST,
    "gratitude": CATEGORY_GRATITUDE,
    "greeting": CATEGORY_GREETING,
    "statement": CATEGORY_STATEMENT,
    "correction": CATEGORY_CORRECTION,
    "affirmation": CATEGORY_AFFIRM,
}

# The exact label set the classifier is allowed to emit (also used to
# validate model output and build the heuristic fallback).
VALID_INTENTS: tuple[str, ...] = tuple(INTENT_TO_CATEGORY.keys())


def reflex_ids_for_category(category: str) -> list[str]:
    """All reflex ids in a category, in catalog order."""
    return [r.id for r in REFLEXES if r.category == category]


def pick_reflex_for_intent(intent: str, *, exclude: Iterable[str] = ()) -> str | None:
    """Pick a reflex id for the classified intent, avoiding recently-used
    ids for variety. Unknown/empty intent falls back to the generic
    'thinking' set. Returns None only if the catalog is empty."""
    category = INTENT_TO_CATEGORY.get((intent or "").strip().lower(), CATEGORY_THINKING)
    ids = reflex_ids_for_category(category) or reflex_ids_for_category(CATEGORY_THINKING)
    if not ids:
        return None
    exclude_set = set(exclude or ())
    fresh = [i for i in ids if i not in exclude_set]
    return random.choice(fresh if fresh else ids)


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
    qa: bool | None = None,
    retries: int | None = None,
) -> dict[str, Any]:
    """Synthesize one reflex and promote it only after fail-closed QA.

    Every render is first written to a same-directory candidate. Real ASR
    evidence and an affirmative semantic judge are both required before the
    candidate WAV and its hash-bound sidecar replace the live pair. HOLD leaves
    any last-known-good pair untouched.
    """
    reflex = _REFLEX_BY_ID.get(reflex_id)
    if reflex is None:
        raise ReflexUnknown(f"unknown reflex id: {reflex_id!r}")
    use_voice = voice or DEFAULT_REFLEX_VOICE
    use_model = model or DEFAULT_REFLEX_MODEL
    use_format = response_format or DEFAULT_REFLEX_FORMAT
    qa = _reflex_qa_enabled() if qa is None else qa
    retries = _reflex_qa_retries() if retries is None else retries
    original = reflex.text
    safe_voice = _safe_voice(use_voice)
    path = reflex_path(safe_voice, reflex_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _synth(text: str) -> bytes:
        return synthesize(
            hivemind_url=hivemind_url, text=text, model=use_model,
            voice=use_voice, response_format=use_format, timeout=timeout,
        )["audio_bytes"]

    qa_attempts: list[dict[str, Any]] = []
    text = original
    max_attempts = (max(0, retries) + 1) if qa else 1
    hold_reason = "qa_disabled" if not qa else "retry_exhausted"
    last_heard = ""
    last_candidate_size = 0

    for attempt in range(max_attempts):
        audio = _synth(text)
        last_candidate_size = len(audio)
        candidate_path = _write_candidate_audio(path, audio)
        try:
            candidate_audio = candidate_path.read_bytes()
            if not qa:
                qa_attempts.append({
                    "text": text,
                    "valid": False,
                    "heard": "",
                    "reason": "qa_disabled",
                })
                break

            verdict = validate_reflex_audio(original, candidate_audio, hivemind_url)
            last_heard = str(verdict.get("heard") or "")
            reason = str(verdict.get("reason") or "qa_inconclusive")
            qa_attempts.append({
                "text": text,
                "valid": verdict.get("valid") is True,
                "heard": last_heard[:120],
                "reason": reason,
            })
            if verdict.get("valid") is True:
                meta = {
                    "schema": "Ms4ReflexMeta.v1",
                    "id": reflex_id,
                    "original_text": original,
                    "effective_text": text,
                    "validated": True,
                    "heard": last_heard,
                    "qa_attempts": qa_attempts,
                    "qa_version": _QA_VERSION,
                    "audio_sha256": _audio_sha256(candidate_audio),
                    "rendered_at": int(time.time()),
                }
                _promote_candidate(
                    candidate_path=candidate_path,
                    voice=safe_voice,
                    reflex_id=reflex_id,
                    meta=meta,
                )
                return {
                    "id": reflex_id,
                    "voice": safe_voice,
                    "model": use_model,
                    "format": use_format,
                    "path": str(path),
                    "size_bytes": len(candidate_audio),
                    "validated": True,
                    "effective_text": text,
                    "heard": last_heard,
                    "status": "promoted",
                    "promoted": True,
                    "hold_reason": None,
                    "qa_attempts": qa_attempts,
                }

            hold_reason = reason
            if reason in _QA_TERMINAL_HOLD_REASONS:
                break
            if attempt + 1 >= max_attempts:
                hold_reason = "retry_exhausted"
                break
            text = rephrase_reflex_text(
                original,
                hivemind_url,
                avoid=[item["text"] for item in qa_attempts],
            )
        finally:
            candidate_path.unlink(missing_ok=True)

    live = _read_reflex_bundle(safe_voice, reflex_id)
    live_audio = live[0] if live is not None else None
    return {
        "id": reflex_id,
        "voice": safe_voice,
        "model": use_model,
        "format": use_format,
        "path": str(path),
        "size_bytes": len(live_audio) if live_audio is not None else 0,
        "validated": False,
        "effective_text": text,
        "heard": last_heard,
        "status": "hold",
        "promoted": False,
        "hold_reason": hold_reason,
        "candidate_size_bytes": last_candidate_size,
        "qa_attempts": qa_attempts,
        "last_known_good_preserved": live is not None,
    }


def generate_all(
    *,
    hivemind_url: str,
    voice: str | None = None,
    model: str | None = None,
    response_format: str | None = None,
    force: bool = False,
    timeout: int = 60,
    qa: bool | None = None,
) -> dict[str, Any]:
    """Synthesize every reflex in :data:`REFLEXES` for ``voice``.

    By default skips reflexes that already exist on disk. Pass
    ``force=True`` to re-render them (e.g. after the voice or model
    has changed). With QA on, existing renders that haven't passed QA
    are validated first and only re-rendered if they fail.
    """
    use_voice = voice or DEFAULT_REFLEX_VOICE
    qa = _reflex_qa_enabled() if qa is None else qa
    generated: list[dict[str, Any]] = []
    held: list[dict[str, Any]] = []
    skipped: list[str] = []
    revalidated: list[str] = []
    failed: dict[str, str] = {}
    for reflex in REFLEXES:
        path = reflex_path(use_voice, reflex.id)
        if path.exists() and not force:
            # A cache hit is usable only when the current sidecar validates and
            # hashes to these exact bytes.
            if _read_reflex_bundle(use_voice, reflex.id) is not None:
                skipped.append(reflex.id)
                continue
            if qa:
                # Validate an existing unbound/stale render before spending a
                # TTS call. Inconclusive evidence is HOLD, not permission to
                # replace it or bless it.
                try:
                    with _REFLEX_IO_LOCK:
                        existing_audio = path.read_bytes()
                    verdict = validate_reflex_audio(reflex.text, existing_audio, hivemind_url)
                    reason = str(verdict.get("reason") or "qa_inconclusive")
                    if verdict.get("valid") is True:
                        meta = {
                            "schema": "Ms4ReflexMeta.v1",
                            "id": reflex.id,
                            "original_text": reflex.text,
                            "effective_text": reflex.text,
                            "validated": True,
                            "heard": verdict.get("heard") or "",
                            "qa_attempts": [],
                            "qa_version": _QA_VERSION,
                            "audio_sha256": _audio_sha256(existing_audio),
                            "rendered_at": int(time.time()),
                        }
                        if _bind_existing_audio(
                            voice=use_voice,
                            reflex_id=reflex.id,
                            expected_audio=existing_audio,
                            meta=meta,
                        ):
                            revalidated.append(reflex.id)
                            skipped.append(reflex.id)
                        elif _read_reflex_bundle(use_voice, reflex.id) is not None:
                            # Another generator promoted a complete valid pair
                            # while ASR was checking the prior bytes.
                            skipped.append(reflex.id)
                        else:
                            held.append({
                                "id": reflex.id,
                                "voice": _safe_voice(use_voice),
                                "status": "hold",
                                "promoted": False,
                                "hold_reason": "concurrent_audio_change",
                            })
                        continue
                    if reason in _QA_TERMINAL_HOLD_REASONS:
                        held.append({
                            "id": reflex.id,
                            "voice": _safe_voice(use_voice),
                            "status": "hold",
                            "promoted": False,
                            "hold_reason": reason,
                        })
                        continue
                    log.info(
                        "reflex %r existing render failed QA (%s; heard=%r) -> regenerating",
                        reflex.id,
                        reason,
                        str(verdict.get("heard") or "")[:40],
                    )
                except Exception as exc:  # noqa: BLE001
                    log.info("reflex %r existing validation errored; holding: %s", reflex.id, exc)
                    held.append({
                        "id": reflex.id,
                        "voice": _safe_voice(use_voice),
                        "status": "hold",
                        "promoted": False,
                        "hold_reason": "validation_error",
                    })
                    continue
        try:
            info = generate_reflex(
                hivemind_url=hivemind_url,
                reflex_id=reflex.id,
                voice=use_voice,
                model=model,
                response_format=response_format,
                timeout=timeout,
                qa=qa,
            )
            if info.get("promoted") is True:
                generated.append(info)
            else:
                held.append(info)
        except Exception as exc:
            log.warning("reflex %r generation failed: %s", reflex.id, exc)
            failed[reflex.id] = str(exc)[:240]
    return {
        "schema": "Ms4ReflexGeneration.v1",
        "voice": _safe_voice(use_voice),
        "model": model or DEFAULT_REFLEX_MODEL,
        "force": force,
        "qa": qa,
        "generated": generated,
        "held": held,
        "skipped": skipped,
        "revalidated": revalidated,
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
    qa: bool | None = None,
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
                qa=qa,
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
# QA: round-trip validation of generated canned speech
#
# Render candidate -> require real Whisper ASR evidence -> require a tiny-LLM
# semantic acceptance. Deterministically bad renders may be retried with a
# TTS-stable equivalent. Missing or ambiguous evidence is HOLD and can never
# promote.
# ---------------------------------------------------------------------------


def _reflex_qa_enabled() -> bool:
    return os.environ.get("MS4_REFLEX_QA", "1").strip().lower() not in {"0", "false", "no", "off"}


def _reflex_qa_retries() -> int:
    try:
        return max(0, int(os.environ.get("MS4_REFLEX_QA_RETRIES", "3")))
    except (TypeError, ValueError):
        return 3


def _reflex_qa_model() -> str:
    return (os.environ.get("MS4_REFLEX_QA_MODEL", "").strip()
            or os.environ.get("MS4_VOICE_REFLEX_MODEL", "qwen2.5:0.5b"))


# Bump when the validator logic changes so already-"validated" reflexes get
# re-checked on the next pass (the earlier versions were fail-open).
_QA_VERSION = 3

_QA_PUNCT_RE = re.compile(r"[^a-z0-9' ]+")
_QA_TERMINAL_HOLD_REASONS = frozenset({
    "asr_unavailable",
    "asr_evidence_unavailable",
    "asr_evidence_ambiguous",
    "transcript_ambiguous",
    "semantic_judge_unavailable",
})

# Audio duration helpers live in voice.py now (shared with the runtime
# gate); these aliases keep the QA code readable.
_wav_duration_secs = wav_duration_secs
_expected_max_secs = expected_speech_secs


def _qa_norm(text: str) -> str:
    return _QA_PUNCT_RE.sub(" ", (text or "").lower()).strip()


def _chat_text(resp: Any) -> str:
    """Extract assistant text from an OpenAI-shaped chat response."""
    if isinstance(resp, str):
        return resp
    if not isinstance(resp, dict):
        return ""
    choices = resp.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        msg = choices[0].get("message")
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            return msg["content"]
        if isinstance(choices[0].get("text"), str):
            return choices[0]["text"]
    for key in ("content", "text", "message", "output"):
        if isinstance(resp.get(key), str):
            return resp[key]
    return ""


def _qa_fuzzy(intended: str, heard: str) -> bool | None:
    """Fast, model-free verdict: True (clearly matches), False (clearly
    garbled/empty/repeated), or None (ambiguous -> ask the LLM)."""
    ni, nh = _qa_norm(intended), _qa_norm(heard)
    if not nh:
        return False
    if ni == nh:
        return True
    iw, hw = ni.split(), nh.split()
    if not iw:
        return None
    # Repetition: the phrase echoed more than once ("on it on it") is a
    # garble, not a match.
    if ni and nh.count(ni) > 1:
        return False
    # Length blow-up == babble.
    if len(hw) > max(3, len(iw) * 2):
        return False
    # Clean match: all words present and length barely longer.
    if all(w in hw for w in iw) and len(hw) <= len(iw) + 1:
        return True
    return None


def _qa_llm_judge(intended: str, heard: str, hivemind_url: str) -> bool | None:
    """Tiny-LLM yes/no: does the heard audio cleanly say the intended
    phrase? None if the model is unavailable."""
    try:
        from . import hivemind_tools

        resp = hivemind_tools.inference_chat(
            hivemind_url,
            messages=[
                {"role": "system", "content": (
                    "You check text-to-speech output. Given the INTENDED phrase and what was "
                    "actually HEARD (auto-transcribed from the audio), answer 'yes' if the audio "
                    "cleanly and correctly says the intended phrase (ignore punctuation/case and "
                    "tiny transcription slips), or 'no' if it is garbled, wrong, padded with extra "
                    "words, or empty. Reply with ONLY 'yes' or 'no'."
                )},
                {"role": "user", "content": f"INTENDED: {intended!r}\nHEARD: {heard!r}"},
            ],
            model=_reflex_qa_model(), max_tokens=3, temperature=0.0, timeout=4.0,
        )
        ans = _chat_text(resp).strip().lower()
        if ans == "yes":
            return True
        if ans == "no":
            return False
        return None
    except Exception as exc:  # noqa: BLE001
        log.info("reflex QA llm judge unavailable: %s", exc)
        return None


def _asr_has_untrusted_hint(value: Any) -> bool:
    """Reject transcript hints or fabricated fixtures masquerading as ASR."""
    markers = ("hint", "fabricat", "synthetic", "mock")
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key).lower()
            if any(marker in key_text for marker in markers) and item not in (None, False, ""):
                return True
            if key_text in {"source", "origin", "evidence_source"}:
                item_text = str(item).lower()
                if any(marker in item_text for marker in markers):
                    return True
            if _asr_has_untrusted_hint(item):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_asr_has_untrusted_hint(item) for item in value)
    return False


def _real_asr_text(asr: Any) -> tuple[str, str | None]:
    """Extract text only from the raw ASR response retained by voice.transcribe."""
    if not isinstance(asr, dict):
        return "", "asr_evidence_unavailable"
    raw = asr.get("raw")
    if not isinstance(raw, dict):
        return "", "asr_evidence_unavailable"
    if _asr_has_untrusted_hint(raw):
        return "", "asr_evidence_ambiguous"
    outer = asr.get("text")
    raw_text = raw.get("text") or raw.get("transcription")
    if not isinstance(outer, str) or not isinstance(raw_text, str):
        return "", "asr_evidence_unavailable"
    outer = outer.strip()
    raw_text = raw_text.strip()
    if not outer or not raw_text:
        return "", "transcript_ambiguous"
    if outer != raw_text:
        return "", "asr_evidence_ambiguous"
    return outer, None


def _qa_result(valid: bool, heard: str, reason: str) -> dict[str, Any]:
    return {
        "valid": valid,
        "heard": heard,
        "reason": reason,
        "disposition": "accept" if valid else "hold",
    }


def validate_reflex_audio(intended_text: str, audio_bytes: bytes, hivemind_url: str) -> dict[str, Any]:
    """Round-trip QA for one candidate, failing closed on uncertainty.

    Order of evidence (strongest first):
    1. Duration gate (model-free) — a clip much longer than the phrase
       warrants is a babble tail; reject outright.
    2. Real ASR round-trip evidence (not a hint or caller-fabricated text).
    3. Deterministic transcript sanity; ambiguity is HOLD.
    4. Mandatory tiny-LLM semantic acceptance.
    """
    # 1. Duration sanity — catches the oversized babble renders directly.
    dur = _wav_duration_secs(audio_bytes)
    expected_max = _expected_max_secs(intended_text)
    if dur is not None and dur > expected_max:
        return _qa_result(False, "", f"too_long_{dur:.1f}s_max_{expected_max:.1f}s")
    # 2. ASR round-trip.
    try:
        asr = transcribe(hivemind_url=hivemind_url, audio=audio_bytes, filename="reflex_qa.wav", model=None)
    except Exception as exc:  # noqa: BLE001
        log.info("reflex QA ASR unavailable (holding candidate): %s", exc)
        return _qa_result(False, "", "asr_unavailable")
    heard, evidence_error = _real_asr_text(asr)
    if evidence_error is not None:
        return _qa_result(False, heard, evidence_error)

    fuzzy = _qa_fuzzy(intended_text, heard)
    if fuzzy is False:
        return _qa_result(False, heard, "fuzzy_bad")
    if fuzzy is None:
        return _qa_result(False, heard, "transcript_ambiguous")

    # A clean transcript is necessary but not sufficient: semantic acceptance
    # must be an explicit yes from the judge.
    judged = _qa_llm_judge(intended_text, heard, hivemind_url)
    if judged is None:
        return _qa_result(False, heard, "semantic_judge_unavailable")
    if judged is False:
        return _qa_result(False, heard, "semantic_rejected")
    return _qa_result(True, heard, "semantic_accept")


_MANUAL_REPHRASE = {
    "On it.": "I'm on it.", "Mhm.": "Mm hmm.", "Mhm?": "Mm hmm?",
    "Right away.": "Coming right up.", "Done.": "All done.", "Okay.": "Okay then.",
    "Great.": "Sounds great.", "Right.": "Right, got it.", "Of course.": "Yeah, of course.",
    "Yes?": "Yes, go ahead?", "Perfect.": "That's perfect.", "Anytime!": "Anytime, happy to.",
}


def rephrase_reflex_text(original: str, hivemind_url: str, *, avoid: Iterable[str] = ()) -> str:
    """A short, TTS-stable equivalent of ``original`` (same meaning/tone).
    LLM first, then a manual map for known troublemakers, then padding."""
    avoid_set = {original.lower()} | {a.lower() for a in (avoid or ())}
    try:
        from . import hivemind_tools

        resp = hivemind_tools.inference_chat(
            hivemind_url,
            messages=[
                {"role": "system", "content": (
                    "A very short phrase garbles in text-to-speech. Give ONE short, natural spoken "
                    "equivalent (3 to 6 words) with the SAME meaning and tone. Reply with ONLY the "
                    "phrase - no quotes, no explanation."
                )},
                {"role": "user", "content": original},
            ],
            model=_reflex_qa_model(), max_tokens=16, temperature=0.5, timeout=4.0,
        )
        cand = _chat_text(resp).strip().strip('"\'').splitlines()[0].strip() if _chat_text(resp) else ""
        if cand and 0 < len(cand) <= 60 and cand.lower() not in avoid_set:
            return cand
    except Exception as exc:  # noqa: BLE001
        log.info("reflex rephrase llm unavailable: %s", exc)
    manual = _MANUAL_REPHRASE.get(original.strip())
    if manual and manual.lower() not in avoid_set:
        return manual
    padded = f"{original.rstrip('.!?').strip()}, sure."
    return padded if padded.lower() not in avoid_set else original


def _meta_path(voice: str, reflex_id: str) -> Path:
    return reflex_path(voice, reflex_id).with_suffix(".meta.json")


def _audio_sha256(audio: bytes) -> str:
    return hashlib.sha256(audio).hexdigest()


def _write_temp_bytes(
    *,
    directory: Path,
    prefix: str,
    suffix: str,
    data: bytes,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=str(directory),
        prefix=prefix,
        suffix=suffix,
        delete=False,
    ) as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
        return Path(handle.name)


def _write_candidate_audio(final_path: Path, audio: bytes) -> Path:
    return _write_temp_bytes(
        directory=final_path.parent,
        prefix=f".{final_path.stem}.",
        suffix=".candidate.wav",
        data=audio,
    )


def _replace_bytes(path: Path, data: bytes, *, suffix: str) -> None:
    temp_path = _write_temp_bytes(
        directory=path.parent,
        prefix=f".{path.name}.",
        suffix=suffix,
        data=data,
    )
    try:
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _write_meta(voice: str, reflex_id: str, meta: dict[str, Any]) -> None:
    payload = json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8")
    with _REFLEX_IO_LOCK:
        _replace_bytes(_meta_path(voice, reflex_id), payload, suffix=".meta.tmp")


def _bind_existing_audio(
    *,
    voice: str,
    reflex_id: str,
    expected_audio: bytes,
    meta: dict[str, Any],
) -> bool:
    """Atomically bind metadata only if the validated WAV is still current."""
    if not _meta_matches_audio(meta, expected_audio, reflex_id):
        raise ValueError("reflex metadata does not match validated audio")
    path = reflex_path(voice, reflex_id)
    payload = json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8")
    with _REFLEX_IO_LOCK:
        try:
            current_audio = path.read_bytes()
        except (FileNotFoundError, OSError):
            return False
        if current_audio != expected_audio:
            return False
        _replace_bytes(_meta_path(voice, reflex_id), payload, suffix=".meta.tmp")
        return True


def _read_meta_unlocked(voice: str, reflex_id: str) -> dict[str, Any] | None:
    try:
        path = _meta_path(voice, reflex_id)
        if path.exists():
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else None
    except Exception:  # noqa: BLE001
        return None
    return None


def _read_meta(voice: str, reflex_id: str) -> dict[str, Any] | None:
    with _REFLEX_IO_LOCK:
        return _read_meta_unlocked(voice, reflex_id)


def _meta_matches_audio(meta: Any, audio: bytes, reflex_id: str) -> bool:
    if not isinstance(meta, dict):
        return False
    digest = meta.get("audio_sha256")
    return bool(
        meta.get("schema") == "Ms4ReflexMeta.v1"
        and meta.get("id") == reflex_id
        and meta.get("validated") is True
        and meta.get("qa_version") == _QA_VERSION
        and isinstance(digest, str)
        and re.fullmatch(r"[0-9a-f]{64}", digest)
        and digest == _audio_sha256(audio)
    )


def _read_reflex_bundle(
    voice: str,
    reflex_id: str,
) -> tuple[bytes, dict[str, Any]] | None:
    path = reflex_path(voice, reflex_id)
    with _REFLEX_IO_LOCK:
        meta = _read_meta_unlocked(voice, reflex_id)
        try:
            audio = path.read_bytes()
        except (FileNotFoundError, OSError):
            return None
        if not _meta_matches_audio(meta, audio, reflex_id):
            return None
        return audio, meta


def _promote_candidate(
    *,
    candidate_path: Path,
    voice: str,
    reflex_id: str,
    meta: dict[str, Any],
) -> None:
    """Promote a validated pair with same-directory replaces under one lock.

    The lock is shared by all gateway readers. If either replace fails, the
    previous bytes are restored before readers can proceed.
    """
    final_path = reflex_path(voice, reflex_id)
    meta_path = _meta_path(voice, reflex_id)
    if candidate_path.parent.resolve() != final_path.parent.resolve():
        raise ValueError("reflex candidate must share the live file directory")
    candidate_audio = candidate_path.read_bytes()
    if not _meta_matches_audio(meta, candidate_audio, reflex_id):
        raise ValueError("reflex candidate metadata does not match candidate audio")
    meta_candidate = _write_temp_bytes(
        directory=meta_path.parent,
        prefix=f".{meta_path.name}.",
        suffix=".candidate.meta.json",
        data=json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"),
    )
    try:
        with _REFLEX_IO_LOCK:
            old_audio = final_path.read_bytes() if final_path.exists() else None
            old_meta = meta_path.read_bytes() if meta_path.exists() else None
            try:
                os.replace(candidate_path, final_path)
                os.replace(meta_candidate, meta_path)
            except Exception:
                if old_audio is None:
                    final_path.unlink(missing_ok=True)
                else:
                    _replace_bytes(final_path, old_audio, suffix=".rollback.wav")
                if old_meta is None:
                    meta_path.unlink(missing_ok=True)
                else:
                    _replace_bytes(meta_path, old_meta, suffix=".rollback.meta.json")
                raise
    finally:
        candidate_path.unlink(missing_ok=True)
        meta_candidate.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Read + list (gateway routes use these)
# ---------------------------------------------------------------------------


def read_reflex(*, reflex_id: str, voice: str | None = None) -> bytes | None:
    """Return only a current, validated, hash-bound cached WAV."""
    if not _REFLEX_BY_ID.get(reflex_id):
        raise ReflexUnknown(f"unknown reflex id: {reflex_id!r}")
    bundle = _read_reflex_bundle(voice or DEFAULT_REFLEX_VOICE, reflex_id)
    return bundle[0] if bundle is not None else None


def list_reflexes(*, voice: str | None = None) -> dict[str, Any]:
    """Catalog snapshot + on-disk availability for one voice."""
    use_voice = _safe_voice(voice or DEFAULT_REFLEX_VOICE)
    entries: list[dict[str, Any]] = []
    available = 0
    total_bytes = 0
    for reflex in REFLEXES:
        bundle = _read_reflex_bundle(use_voice, reflex.id)
        audio = bundle[0] if bundle is not None else None
        meta = bundle[1] if bundle is not None else {}
        available_on_disk = audio is not None
        size = len(audio) if audio is not None else 0
        if available_on_disk:
            available += 1
            total_bytes += size
        entries.append({
            "id": reflex.id,
            "text": reflex.text,
            "category": reflex.category,
            "description": reflex.description,
            "voice": use_voice,
            "url": f"/reflexes/{use_voice}/{reflex.id}.wav",
            "available": available_on_disk,
            "size_bytes": size,
            # QA fields (None until validated): whether the audio passed the
            # round-trip check, what the ASR heard, and the effective spoken
            # text (may differ from `text` if it was auto-rephrased).
            "validated": meta.get("validated"),
            "heard": meta.get("heard"),
            "effective_text": meta.get("effective_text") or reflex.text,
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
