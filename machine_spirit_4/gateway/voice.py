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
import inspect
import io
import json
import logging
import math
import mimetypes
import os
import queue
import re
import socket
import ssl
import struct
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import wave
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


log = logging.getLogger("ms4.gateway.voice")


DEFAULT_TRANSCRIBE_MODEL = os.environ.get("MS4_VOICE_ASR_MODEL", "whisper-1")
DEFAULT_TTS_MODEL = os.environ.get("MS4_VOICE_TTS_MODEL", "tts-1")


def _default_tts_voice() -> str:
    """Resolved default TTS voice (env override, else the built-in default).

    HiveMind/Oracle narration is delivered through the regular REST TTS path
    (see ``_default_tts_engine`` / ``DEFAULT_ENGINE``, which stays ``rest``
    with model ``tts-1``) using the ``vega`` voice by default;
    ``MS4_VOICE_TTS_VOICE`` overrides it per deployment. Exposed as a
    function (mirroring ``_default_tts_engine``) so tests can assert the
    built-in fallback independent of the pinned ``DEFAULT_TTS_VOICE`` module
    constant.
    """
    return os.environ.get("MS4_VOICE_TTS_VOICE", "vega")


DEFAULT_TTS_VOICE = _default_tts_voice()
DEFAULT_TTS_FORMAT = os.environ.get("MS4_VOICE_TTS_FORMAT", "wav")
MAX_AUDIO_BYTES = int(os.environ.get("MS4_VOICE_MAX_AUDIO_BYTES", str(25 * 1024 * 1024)))


def _tts_location_policy() -> str | None:
    """Optional HLI placement pin for REST speech requests.

    This is intentionally opt-in: ordinary Oracle deployments may use the
    cluster, while a deployment that has detected a recursive peer route can
    pin speech to the local node without changing the shared HiveMind URL used
    by ASR, chat, tools, and health checks.
    """
    raw = os.environ.get("MS4_TTS_LOCATION_POLICY", "").strip().lower()
    if not raw:
        return None
    aliases = {
        "local": "local_only",
        "local-only": "local_only",
        "ondevice": "local_only",
        "peer-only": "peer_only",
        "prefer-local": "prefer_local",
        "prefer-peer": "prefer_peer",
    }
    policy = aliases.get(raw, raw)
    allowed = {"local_only", "peer_only", "prefer_local", "prefer_peer", "cluster"}
    if policy not in allowed:
        raise VoiceRequestError(
            "MS4_TTS_LOCATION_POLICY must be one of: local_only, peer_only, "
            "prefer_local, prefer_peer, cluster"
        )
    return policy
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
XTTS_HARD_TOKEN_LIMIT = 400
DEFAULT_XTTS_TOKEN_BUDGET = 360
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
# concurrent REST TTS requests, so that fallback can still fan out with
# parallelism = faster total synthesis on longer replies.
TTS_POOL_SIZE = int(os.environ.get("MS4_VOICE_TTS_POOL_SIZE", "6"))
# Engine selector for the streaming voice path. ``rest`` is the
# sentence-chunked parallel-POST path; ``ws_super`` opens one
# WebSocket per turn to HiveMind's TTS_SUPER ``stream-input`` endpoint
# (see ``gateway/tts_super_ws.py``).
#
# Default journey:
#   May 22 2026: rest → ws_super (chasing first-audio latency)
#   May 26 2026: ws_super → rest (ws_super ~10x slower on this cluster)
#   Jun 30 2026: briefly re-defaulted to ws_super on a warm first-audio result
#   Jul 06 2026: ws_super → rest (production audio-stutter evidence). A physical
#     production turn showed ws_super's single-stream TTS_SUPER GIM producing
#     audio SLOWER than real time (12 chunks 477-640 ms each, arriving
#     984-1391 ms apart), starving the browser scheduler into 10 audible gaps
#     totalling 4.42 s. REST (parallel up to TTS_POOL_SIZE) stayed ahead of
#     playback with zero gaps. See AUDIO_STUTTER_DIAGNOSIS.md (2026-07-06).
#
# REST is the production/default AND the automatic compatibility fallback.
# Operators can still force ws_super per-turn in Settings or with
# MS4_VOICE_TTS_ENGINE=ws_super for deliberate diagnostics.
def _default_tts_engine() -> str:
    """Resolved default streaming TTS engine (env override, else the built-in
    default). REST is the production default (see the audio-stutter journey
    above); ws_super is a deliberate per-turn / env opt-in. Exposed as a
    function so tests can assert the fallback independent of any pinned
    module constant."""
    return os.environ.get("MS4_VOICE_TTS_ENGINE", "rest").strip().lower()


DEFAULT_ENGINE = _default_tts_engine()
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
    raw_transcript: str | None = None
    grounding_source: str | None = None
    router: dict[str, Any] | None = None
    dispatched_job: dict[str, Any] | None = None


class VoiceUnavailable(RuntimeError):
    """ASR or TTS not ready; raise to convert into HTTP 503."""


class VoiceRequestError(ValueError):
    """Operator/client error; raise to convert into HTTP 400."""


class WsOpenFailed(RuntimeError):
    """TTS_SUPER failed before chat started.

    Import, construction, open, and sender-capability failures are safe to
    re-drive through the REST engine because no Face turn or audio happened.
    Post-chat failures are handled inside ``_run_ws_super_engine`` so chat is
    never replayed.
    """


class FaceLobeStalled(RuntimeError):
    """The Face Lobe chat produced NO first token within the watchdog
    budget and the call had not returned. Almost always a slow/cold/
    unreachable model (e.g. a heavy cloud reasoning model selected for
    voice). Raising this converts a silent infinite hang into a clean
    ``error`` SSE event so the UI plays the canned error reflex and ends
    the stream instead of freezing on "streaming…" forever."""


def _voice_brevity_enabled() -> bool:
    """Whether voice explicitly opts into the legacy shortened-answer mode.

    Complete conversational answers are the default. Operators who knowingly
    prefer a shorter spoken synthesis may set ``MS4_VOICE_BREVITY=1``; the
    directive is never injected merely because the transport happens to be
    voice.
    """
    return os.environ.get("MS4_VOICE_BREVITY", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


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


def _new_voice_turn_id() -> str:
    """Mint the server-owned correlation id for one voice turn."""
    return "ms4-turn-" + uuid.uuid4().hex[:16]


SELECTED_FACE_ADMISSION_PROFILE = "ms4-face-voice-production-uncached-700-v1"
_SELECTED_FACE_ADMISSION_DEFAULT_TTFT_MS = 15_000
_SELECTED_FACE_ADMISSION_WATCHDOG_MARGIN_MS = 1000
_SELECTED_FACE_ADMISSION_PADDING_WORDS = 144
_SELECTED_FACE_ADMISSION_PADDING_TERMS = (
    "distributed",
    "inference",
    "latency",
    "routing",
    "evidence",
    "recommendation",
    "mechanism",
    "tradeoff",
    "example",
    "context",
    "cluster",
    "foreground",
    "background",
    "revision",
    "delivery",
    "verification",
)


def selected_face_admission_latency_budget_ms() -> int:
    """Visible-token budget for selected-Face production admission.

    The default is a conservative warm-path ceiling from the live production-
    shaped discriminator (354ms healthy warm versus Q9's 28,079ms failure).
    An operator may tune it explicitly, but it can never exceed the live voice
    watchdog minus a one-second handoff margin; admitting a model that the next
    voice turn is guaranteed to kill would make the readiness receipt false.
    """
    raw = os.environ.get(
        "MS4_FACE_SELECTED_ADMISSION_TTFT_MS",
        str(_SELECTED_FACE_ADMISSION_DEFAULT_TTFT_MS),
    )
    try:
        configured = int(str(raw).strip())
    except (TypeError, ValueError):
        configured = _SELECTED_FACE_ADMISSION_DEFAULT_TTFT_MS
    configured = min(60_000, max(250, configured))
    watchdog_cap = max(
        250,
        int(_voice_first_token_timeout() * 1000)
        - _SELECTED_FACE_ADMISSION_WATCHDOG_MARGIN_MS,
    )
    return min(configured, watchdog_cap)


class _SelectedFaceAdmissionBlackboard:
    """Side-effect-free context source for a representative Face probe."""

    @staticmethod
    def current_revision(_conversation_id: str) -> int:
        return 1

    @staticmethod
    def list_jobs(**_kwargs: Any) -> list[dict[str, Any]]:
        return []


def _selected_face_admission_probe_payload() -> tuple[str, str]:
    """Build a cache-busted production-sized Face request without side effects.

    Q9 proved that an identical one-word warm can reuse 849/850 prompt tokens
    while a real voice turn still evaluates roughly 700 uncached tokens.  The
    nonce is therefore the first appended system content, invalidating only the
    synthetic suffix while preserving the normal Face base-prefix cache.  The
    remainder mirrors the authoritative context and voice directive used by a
    real turn, then adds a bounded representative payload near Q9's uncached
    size.  No real session, revision, route, or Depth job is created.
    """
    from machine_spirit_4.double_agent import build_face_lobe_context_block
    from .hermes_runner import _VOICE_BREVITY_DIRECTIVE

    nonce = uuid.uuid4().hex
    context = build_face_lobe_context_block(
        conversation_id=f"admission-{nonce}",
        blackboard=_SelectedFaceAdmissionBlackboard(),
        include_completed=False,
        dispatched_this_turn_job_id=None,
        extra_authoritative_lines=[
            f"selected Face admission profile: {SELECTED_FACE_ADMISSION_PROFILE}",
            f"cache-busting probe nonce: {nonce}",
            "this is a side-effect-free latency measurement; no real work was dispatched",
        ],
    ) or "MS4 Face Lobe context (admission probe only)."
    padding = " ".join(
        _SELECTED_FACE_ADMISSION_PADDING_TERMS[
            index % len(_SELECTED_FACE_ADMISSION_PADDING_TERMS)
        ]
        for index in range(_SELECTED_FACE_ADMISSION_PADDING_WORDS)
    )
    extra_system = (
        f"SELECTED FACE ADMISSION CACHE BUSTER: {nonce}\n"
        f"profile: {SELECTED_FACE_ADMISSION_PROFILE}\n\n"
        f"{context}\n\n"
        f"{_VOICE_BREVITY_DIRECTIVE}\n\n"
        "Representative uncached production context follows. It is bounded, "
        "non-authoritative, and exists only to measure prompt evaluation under "
        "the same latency-sensitive request shape as an Oracle voice turn:\n"
        f"{padding}"
    )
    message = (
        "Explain why a larger Depth Lobe can outperform a smaller Face Lobe "
        "when diagnosing intermittent distributed inference failures, including "
        "the recommendation, mechanism, trade-offs, and a concrete example. "
        "For this admission measurement, output exactly one visible token: r"
    )
    return message, extra_system


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


def _voice_speaker_id_terminal_budget() -> float:
    """Bounded cancellation/join budget for the per-turn speaker worker."""
    try:
        val = float(os.environ.get("MS4_VOICE_SPEAKER_ID_TERMINAL_BUDGET_S", "0.25").strip())
    except (TypeError, ValueError):
        val = 0.25
    return max(0.0, val)


def _voice_ws_send_timeout() -> float:
    """Maximum time to drain all queued WS text after chat completes."""
    try:
        val = float(os.environ.get("MS4_TTS_WS_SEND_TIMEOUT_S", "6.0").strip())
    except (TypeError, ValueError):
        val = 6.0
    return max(0.05, val)


def _voice_ws_sender_cleanup_timeout() -> float:
    """Maximum post-cancel join budget for the ordered WS sender."""
    try:
        val = float(os.environ.get("MS4_TTS_WS_SENDER_CLEANUP_TIMEOUT_S", "1.0").strip())
    except (TypeError, ValueError):
        val = 1.0
    return max(0.05, val)


def _voice_ws_first_audio_timeout() -> float:
    """Absolute wall-clock budget for the FIRST WS audio chunk (measured from
    chat flush) before the turn bails to the REST fallback. Raised to 10s from
    6s: a cold / post-idle TTS_SUPER GIM whose first audio lands well after the
    warm ~5-6s must not be killed mid-synthesis. The caller pins this as a fixed
    monotonic deadline so the 0.5s poll cadence can neither extend nor shorten
    it. ``0`` (used by tests) means "bail immediately if no audio yet"."""
    try:
        val = float(os.environ.get("MS4_TTS_WS_FIRST_AUDIO_TIMEOUT", "10").strip())
    except (TypeError, ValueError):
        val = 10.0
    return max(0.0, val)


def _voice_ws_queue_max_items() -> int:
    """Maximum queued and in-flight text fragments for one WS voice turn."""
    try:
        val = int(os.environ.get("MS4_TTS_WS_QUEUE_MAX_ITEMS", "24").strip())
    except (TypeError, ValueError):
        val = 24
    return max(1, val)


def _voice_ws_queue_max_bytes() -> int:
    """Maximum queued and in-flight UTF-8 text bytes for one WS voice turn."""
    try:
        val = int(os.environ.get("MS4_TTS_WS_QUEUE_MAX_BYTES", "65536").strip())
    except (TypeError, ValueError):
        val = 65536
    return max(1, val)


def _voice_ws_queue_put_timeout() -> float:
    """Maximum callback backpressure before a saturated WS turn is cancelled."""
    try:
        val = float(os.environ.get("MS4_TTS_WS_QUEUE_PUT_TIMEOUT_S", "0.25").strip())
    except (TypeError, ValueError):
        val = 0.25
    return max(0.01, val)


def _voice_first_chunk_deadline() -> float:
    """Wall-clock budget after first speakable text for the first TTS chunk."""
    try:
        val = float(os.environ.get("MS4_VOICE_FIRST_CHUNK_DEADLINE_S", "0.35").strip())
    except (TypeError, ValueError):
        val = 0.35
    return max(0.0, val)


def _voice_rest_cleanup_timeout() -> float:
    """Maximum time to retire turn-owned REST synthesis workers."""
    try:
        val = float(os.environ.get("MS4_TTS_REST_CLEANUP_TIMEOUT_S", "1.0").strip())
    except (TypeError, ValueError):
        val = 1.0
    return max(0.05, val)


def _voice_facechat_join_timeout() -> float:
    """Finding 2: bounded grace for the Face Lobe worker thread to retire after a
    turn is cancelled (barge/disconnect) or declared stalled. A cancelled turn
    must never wait on an unbounded join. Tunable via
    ``MS4_VOICE_FACECHAT_JOIN_TIMEOUT_S``."""
    try:
        val = float(os.environ.get("MS4_VOICE_FACECHAT_JOIN_TIMEOUT_S", "2.0").strip())
    except (TypeError, ValueError):
        val = 2.0
    return max(0.05, val)


def _voice_rest_final_timeout(
    *,
    piece_count: int | None = None,
    capacity: int | None = None,
) -> float:
    """Hard ceiling for REST synthesis work owned by one voice turn.

    An explicit ``MS4_TTS_REST_FINAL_TIMEOUT_S`` remains an exact operator
    ceiling.  With the default, scale the tail budget by the number of
    capacity-sized synthesis waves.  The live 22-chunk Depth follow-up reached
    only 10 client-written chunks before the old fixed 60-second ceiling, so a
    complete answer failed after already speaking its first half.
    """

    configured = os.environ.get("MS4_TTS_REST_FINAL_TIMEOUT_S")
    if configured is not None:
        try:
            configured_value = float(configured.strip())
        except (TypeError, ValueError):
            return 60.0
        if not math.isfinite(configured_value):
            return 60.0
        return max(0.05, configured_value)
    base_timeout = 60.0
    if not piece_count or not capacity:
        return base_timeout
    try:
        per_wave = float(
            os.environ.get("MS4_TTS_REST_FINAL_TIMEOUT_PER_WAVE_S", "15.0").strip()
        )
    except (TypeError, ValueError):
        per_wave = 15.0
    if not math.isfinite(per_wave):
        per_wave = 15.0
    try:
        maximum = float(
            os.environ.get("MS4_TTS_REST_FINAL_TIMEOUT_MAX_S", "600.0").strip()
        )
    except (TypeError, ValueError):
        maximum = 600.0
    if not math.isfinite(maximum):
        maximum = 600.0
    waves = (max(1, int(piece_count)) + max(1, int(capacity)) - 1) // max(
        1, int(capacity)
    )
    scaled = max(base_timeout, waves * max(0.05, per_wave))
    return min(scaled, max(base_timeout, maximum))


def _voice_rest_progress_timeout() -> float:
    """Maximum idle interval while REST speech still has pending chunks.

    This is deliberately separate from ``_voice_rest_final_timeout``.  A long
    answer may take more than one minute in total while still delivering a new
    audio chunk every few seconds; only an interval with no client-written
    audio is a stall.  The final timeout remains the finite, workload-scaled
    absolute ceiling for the whole drain.
    """

    try:
        value = float(
            os.environ.get("MS4_TTS_REST_PROGRESS_TIMEOUT_S", "60.0").strip()
        )
    except (TypeError, ValueError):
        value = 60.0
    if not math.isfinite(value):
        value = 60.0
    return max(0.05, value)


_VOICE_REST_CAPACITY_OBSERVATION_LOCK = threading.Lock()
_VOICE_REST_CAPACITY_OBSERVATION: tuple[int, str, float | None] = (
    1,
    "fail_closed_default",
    None,
)


def _voice_rest_capacity_proof_ttl_s() -> float:
    """Maximum age of a measured capacity-two admission.

    The default is one autoscale interval plus a small scheduling reserve. A
    stale result is not evidence that two independently generating endpoints
    still exist, so browser response-start scheduling falls back to one.
    """

    try:
        value = float(
            os.environ.get("MS4_VOICE_TTS_CAPACITY_PROOF_TTL_S", "660").strip()
        )
    except (TypeError, ValueError):
        value = 660.0
    if not math.isfinite(value):
        value = 660.0
    return max(1.0, min(value, 3600.0))


def _voice_rest_capacity_min_speedup() -> float:
    """Measured n=2 speedup floor; operators may tighten, never weaken, it."""

    try:
        value = float(
            os.environ.get("MS4_VOICE_TTS_CAPACITY_MIN_SPEEDUP", "1.5").strip()
        )
    except (TypeError, ValueError):
        value = 1.5
    if not math.isfinite(value):
        value = 1.5
    return max(1.5, min(value, 2.0))


def _set_voice_rest_capacity_observation(
    capacity: int,
    provenance: str,
    *,
    observed_at: float | None = None,
) -> int:
    global _VOICE_REST_CAPACITY_OBSERVATION
    value = max(1, min(int(capacity), 2))
    timestamp = time.monotonic() if observed_at is None else observed_at
    with _VOICE_REST_CAPACITY_OBSERVATION_LOCK:
        _VOICE_REST_CAPACITY_OBSERVATION = (value, provenance, timestamp)
    return value


def _record_voice_rest_capacity_observation(payload: dict[str, Any] | None) -> int:
    """Record scale discovery without mistaking it for concurrency proof.

    ``MS4_VOICE_TTS_CAPACITY`` is a desired client-side ceiling, not evidence
    that that many independently generating GIM processes are available.  A
    scale response claiming two replicas can still front one serialized origin.
    It therefore resets effective capacity to one until a same-route warm
    single-vs-n=2 measurement proves at least 1.5x throughput. A failed,
    malformed, or unavailable response also resets to one.
    """

    replicas = payload.get("replicas") if isinstance(payload, dict) else None
    if (
        isinstance(payload, dict)
        and payload.get("ok") is True
        and type(replicas) is int
        and replicas == 1
    ):
        provenance = "scale_endpoint"
    elif (
        isinstance(payload, dict)
        and payload.get("ok") is True
        and type(replicas) is int
        and replicas > 1
    ):
        provenance = "scale_response_unverified"
    else:
        provenance = "fail_closed_default"
    return _set_voice_rest_capacity_observation(1, provenance)


def _voice_rest_concurrency_verdict(
    *,
    single_ms: Any,
    concurrent_wall_ms: Any,
    concurrent_requests: Any,
    valid_audio_results: Any,
    warmup_audio_valid: Any,
    single_audio_valid: Any,
    provenance_sample_count: Any,
    provenance_non_cloud_count: Any,
    location_policy: Any,
) -> dict[str, Any]:
    """Evaluate a sanitized warm-single versus n=2 REST speech measurement."""

    floor = _voice_rest_capacity_min_speedup()
    strict_policy = location_policy in {"local_only", "peer_only"}
    request_count_valid = type(concurrent_requests) is int and concurrent_requests == 2
    pair_audio_valid = type(valid_audio_results) is int and valid_audio_results == 2
    warmup_valid = warmup_audio_valid is True
    timed_single_valid = single_audio_valid is True
    provenance_valid = (
        type(provenance_sample_count) is int
        and provenance_sample_count == 4
        and type(provenance_non_cloud_count) is int
        and provenance_non_cloud_count == provenance_sample_count
    )
    timing_valid = (
        isinstance(single_ms, (int, float))
        and not isinstance(single_ms, bool)
        and math.isfinite(float(single_ms))
        and float(single_ms) > 0
        and isinstance(concurrent_wall_ms, (int, float))
        and not isinstance(concurrent_wall_ms, bool)
        and math.isfinite(float(concurrent_wall_ms))
        and float(concurrent_wall_ms) > 0
    )
    speedup = (
        (2.0 * float(single_ms)) / float(concurrent_wall_ms)
        if timing_valid
        else 0.0
    )
    passed = (
        strict_policy
        and request_count_valid
        and pair_audio_valid
        and warmup_valid
        and timed_single_valid
        and provenance_valid
        and speedup >= floor
    )
    if not strict_policy:
        reason = "explicit_non_cloud_location_policy_required"
    elif not warmup_valid or not timed_single_valid or not pair_audio_valid:
        reason = "nonempty_audio_required_for_all_samples"
    elif not provenance_valid:
        reason = "non_cloud_hli_provenance_required_for_all_samples"
    elif not request_count_valid or not timing_valid:
        reason = "invalid_probe_measurement"
    elif speedup < floor:
        reason = "speedup_below_floor"
    else:
        reason = "measured_capacity_two"
    return {
        "passed": passed,
        "reason": reason,
        "single_ms": round(float(single_ms), 3) if timing_valid else None,
        "concurrent_requests": 2 if request_count_valid else 0,
        "concurrent_wall_ms": (
            round(float(concurrent_wall_ms), 3) if timing_valid else None
        ),
        "observed_speedup": round(speedup, 3),
        "minimum_speedup": floor,
        "valid_audio_results": valid_audio_results if pair_audio_valid else 0,
        "warmup_audio_valid": warmup_valid,
        "single_audio_valid": timed_single_valid,
        "provenance_sample_count": (
            provenance_sample_count if type(provenance_sample_count) is int else 0
        ),
        "provenance_non_cloud_count": (
            provenance_non_cloud_count
            if type(provenance_non_cloud_count) is int
            else 0
        ),
        "location_policy": location_policy if strict_policy else None,
        "route": "/v1/audio/speech",
    }


def _record_voice_rest_concurrency_observation(proof: dict[str, Any] | None) -> int:
    """Admit capacity two only from a complete, recomputed measurement."""

    candidate = proof if isinstance(proof, dict) else {}
    verdict = _voice_rest_concurrency_verdict(
        single_ms=candidate.get("single_ms"),
        concurrent_wall_ms=candidate.get("concurrent_wall_ms"),
        concurrent_requests=candidate.get("concurrent_requests"),
        valid_audio_results=candidate.get("valid_audio_results"),
        warmup_audio_valid=candidate.get("warmup_audio_valid"),
        single_audio_valid=candidate.get("single_audio_valid"),
        provenance_sample_count=candidate.get("provenance_sample_count"),
        provenance_non_cloud_count=candidate.get("provenance_non_cloud_count"),
        location_policy=candidate.get("location_policy"),
    )
    if candidate.get("passed") is True and verdict["passed"] is True:
        return _set_voice_rest_capacity_observation(
            2,
            "measured_concurrency_probe",
        )
    return _set_voice_rest_capacity_observation(1, "concurrency_probe_failed")


def _voice_rest_provenance_is_non_cloud(
    result: dict[str, Any] | None,
    *,
    location_policy: str,
) -> bool:
    """Require sanitized HLI routing provenance and reject cloud origins."""

    if location_policy not in {"local_only", "peer_only"}:
        return False
    provenance = _sanitize_hli_provenance(
        result.get("provenance") if isinstance(result, dict) else None
    )
    if not provenance:
        return False
    routing_anchor = any(
        provenance.get(key) not in (None, "", [], {})
        for key in ("served_by", "location", "source", "peer", "node_id", "gateway")
    )
    if not routing_anchor:
        return False
    try:
        serialized = json.dumps(provenance, sort_keys=True, allow_nan=False).lower()
    except (TypeError, ValueError):
        return False
    cloud_markers = (
        "api.openai.com",
        "azure",
        "bedrock",
        "vertex",
        "anthropic",
        "gemini",
        "cloudflare",
        "cloud_provider",
        "cloud-provider",
    )
    if any(marker in serialized for marker in cloud_markers):
        return False
    explicit_location = str(provenance.get("location") or "").strip().lower()
    allowed_locations = {
        "peer_only": {"lan", "peer", "peer_only", "remote"},
        "local_only": {"local", "local_only", "ondevice", "on_device"},
    }
    # Marker absence is not proof. The HLI response must explicitly bind every
    # sample to the same non-cloud location class requested on the wire.
    return explicit_location in allowed_locations[location_policy]


_VOICE_REST_WAV_CONTENT_TYPES = frozenset(
    {"audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave"}
)


def _voice_rest_wav_audio_is_valid(result: dict[str, Any] | None) -> bool:
    """Validate one strict, complete, nonempty PCM WAV response.

    Capacity evidence controls browser playback admission, so JSON/HTML error
    bodies, header-only ``RIFF`` prefixes, truncated RIFF containers, and a WAV
    body paired with a non-WAV content type all fail closed.  The capacity
    proof deliberately accepts exactly one ``fmt `` and one final ``data``
    chunk: bytes or chunks after audio are not useful-reply evidence.
    """

    if not isinstance(result, dict):
        return False
    content_type = str(result.get("content_type") or "").split(";", 1)[0].strip().lower()
    if content_type not in _VOICE_REST_WAV_CONTENT_TYPES:
        return False
    audio = result.get("audio_bytes")
    if not isinstance(audio, (bytes, bytearray)):
        return False
    data = bytes(audio)
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return False
    # RIFF's declared container size excludes the leading 8 bytes. Exact size
    # binding rejects both truncation and an appended non-audio response tail.
    if int.from_bytes(data[4:8], "little", signed=False) + 8 != len(data):
        return False

    offset = 12
    fmt_geometry: tuple[int, int, int, int] | None = None
    data_bytes: int | None = None
    while offset < len(data):
        # A four-byte tail such as ``EVIL`` is not a chunk: every chunk needs
        # its complete id+size header even if RIFF's outer size was rewritten.
        if len(data) - offset < 8:
            return False
        chunk_id = data[offset : offset + 4]
        chunk_bytes = int.from_bytes(
            data[offset + 4 : offset + 8], "little", signed=False
        )
        body_start = offset + 8
        body_end = body_start + chunk_bytes
        if body_end > len(data):
            return False
        padded_end = body_end + (chunk_bytes & 1)
        if padded_end > len(data):
            return False
        if chunk_bytes & 1 and data[body_end] != 0:
            return False

        if chunk_id == b"fmt ":
            if fmt_geometry is not None or chunk_bytes < 16:
                return False
            encoding, channels, rate, byte_rate, block_align, bits_per_sample = (
                struct.unpack_from("<HHIIHH", data, body_start)
            )
            if (
                encoding != 1
                or channels <= 0
                or rate <= 0
                or bits_per_sample not in {8, 16, 24, 32}
                or block_align != channels * (bits_per_sample // 8)
                or byte_rate != rate * block_align
            ):
                return False
            fmt_geometry = (channels, rate, bits_per_sample // 8, block_align)
        elif chunk_id == b"data":
            if data_bytes is not None or fmt_geometry is None or chunk_bytes <= 0:
                return False
            data_bytes = chunk_bytes
            # For this fail-closed capacity proof, audio must be the terminal
            # RIFF chunk.  A well-formed unknown chunk after data is still an
            # appended tail and therefore cannot prove a clean audio response.
            if padded_end != len(data):
                return False
        offset = padded_end

    if offset != len(data) or fmt_geometry is None or data_bytes is None:
        return False
    channels, rate, sample_width, block_align = fmt_geometry
    if data_bytes % block_align != 0:
        return False
    frames = data_bytes // block_align
    if frames <= 0:
        return False
    try:
        with wave.open(io.BytesIO(data), "rb") as reader:
            if reader.getcomptype() != "NONE":
                return False
            if (
                reader.getnframes() != frames
                or reader.getframerate() != rate
                or reader.getnchannels() != channels
                or reader.getsampwidth() != sample_width
            ):
                return False
            decoded = reader.readframes(frames)
            if len(decoded) != data_bytes or reader.readframes(1):
                return False
    except (EOFError, OSError, ValueError, wave.Error):
        return False
    return True


def probe_voice_rest_concurrency(
    *,
    hivemind_url: str,
    model: str | None = None,
    voice: str | None = None,
    response_format: str | None = None,
    timeout: float | None = None,
    synthesize_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Measure warm single versus n=2 throughput on the production REST route.

    Billing safety is fail closed: the probe runs only under an explicit
    ``local_only`` or ``peer_only`` TTS location policy, uses the selected model
    without fallback or provisioning, and requires non-cloud HLI provenance on
    the warm single and both concurrent samples. Only >=1.5x measured speedup
    admits capacity two.
    """

    location_policy = _tts_location_policy()
    if location_policy not in {"local_only", "peer_only"}:
        verdict = _voice_rest_concurrency_verdict(
            single_ms=None,
            concurrent_wall_ms=None,
            concurrent_requests=2,
            valid_audio_results=0,
            warmup_audio_valid=False,
            single_audio_valid=False,
            provenance_sample_count=0,
            provenance_non_cloud_count=0,
            location_policy=location_policy,
        )
        _record_voice_rest_concurrency_observation(verdict)
        return verdict
    requested_format = str(response_format or DEFAULT_TTS_FORMAT).strip().lower()
    if requested_format not in {"wav", "wave"}:
        verdict = _voice_rest_concurrency_verdict(
            single_ms=None,
            concurrent_wall_ms=None,
            concurrent_requests=2,
            valid_audio_results=0,
            warmup_audio_valid=False,
            single_audio_valid=False,
            provenance_sample_count=0,
            provenance_non_cloud_count=0,
            location_policy=location_policy,
        )
        verdict["reason"] = "wav_response_format_required"
        verdict["requested_response_format"] = requested_format or None
        _record_voice_rest_concurrency_observation(verdict)
        return verdict
    if timeout is None:
        try:
            timeout = float(
                os.environ.get(
                    "MS4_VOICE_TTS_CAPACITY_PROBE_TIMEOUT_S",
                    "30",
                ).strip()
            )
        except (TypeError, ValueError):
            timeout = 30.0
    if not math.isfinite(timeout):
        timeout = 30.0
    total_timeout = max(0.05, min(float(timeout), 120.0))
    call_synthesize = synthesize if synthesize_fn is None else synthesize_fn
    nonce = uuid.uuid4().hex[:12]
    probe_deadline = time.monotonic() + total_timeout

    def remaining_budget() -> float:
        remaining = probe_deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("REST TTS capacity probe exhausted its total budget")
        return max(0.01, remaining)

    def run_sample(label: str) -> dict[str, Any]:
        # Labels are equal length and the nonce defeats any response cache while
        # keeping synthesis work comparable across all three requests.
        return call_synthesize(
            hivemind_url=hivemind_url,
            text=f"Oracle capacity probe {nonce} {label}.",
            model=model,
            voice=voice,
            response_format="wav",
            timeout=remaining_budget(),
        )

    try:
        # Pay any cold-start cost before the timed baseline. Without this bound
        # warmup, a slow cold single followed by two warm but serialized calls
        # can manufacture a false >1.5x speedup.
        warmup = run_sample("ten")
        single_started = time.monotonic()
        single = run_sample("one")
        single_ms = (time.monotonic() - single_started) * 1000.0
        concurrent_started = time.monotonic()
        executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ms4-tts-capacity")
        futures = [executor.submit(run_sample, label) for label in ("two", "six")]
        concurrent: list[dict[str, Any]] = []
        try:
            for future in futures:
                remaining = probe_deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "REST TTS capacity probe exhausted its total budget"
                    )
                concurrent.append(
                    future.result(timeout=remaining)
                )
        finally:
            for future in futures:
                if not future.done():
                    future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
        concurrent_wall_ms = (time.monotonic() - concurrent_started) * 1000.0
        samples = [warmup, single, *concurrent]
        audio_valid = [_voice_rest_wav_audio_is_valid(item) for item in samples]
        non_cloud = [
            _voice_rest_provenance_is_non_cloud(
                item,
                location_policy=location_policy,
            )
            for item in samples
        ]
        # Detect a policy change during the measurement rather than attributing
        # samples potentially sent under different routing constraints.
        policy_stable = _tts_location_policy() == location_policy
        verdict = _voice_rest_concurrency_verdict(
            single_ms=single_ms,
            concurrent_wall_ms=concurrent_wall_ms,
            concurrent_requests=2,
            valid_audio_results=sum(audio_valid[2:]),
            warmup_audio_valid=audio_valid[0],
            single_audio_valid=audio_valid[1],
            provenance_sample_count=len(samples),
            provenance_non_cloud_count=sum(non_cloud) if policy_stable else 0,
            location_policy=location_policy if policy_stable else None,
        )
        verdict["requested_response_format"] = "wav"
        verdict["audio_validation"] = "strict_final_data_pcm_wav_v2"
    except Exception as exc:
        verdict = _voice_rest_concurrency_verdict(
            single_ms=None,
            concurrent_wall_ms=None,
            concurrent_requests=2,
            valid_audio_results=0,
            warmup_audio_valid=False,
            single_audio_valid=False,
            provenance_sample_count=0,
            provenance_non_cloud_count=0,
            location_policy=location_policy,
        )
        verdict["reason"] = "probe_request_failed"
        verdict["error_type"] = type(exc).__name__
        verdict["requested_response_format"] = "wav"
        verdict["audio_validation"] = "strict_final_data_pcm_wav_v2"
    _record_voice_rest_concurrency_observation(verdict)
    return verdict


def _voice_rest_capacity_snapshot() -> tuple[int, str]:
    """Return observed REST-TTS capacity clamped by the operator ceiling."""

    try:
        configured = int(os.environ.get("MS4_VOICE_TTS_CAPACITY", "2").strip())
    except (TypeError, ValueError):
        configured = 2
    configured = max(1, min(configured, 64))
    with _VOICE_REST_CAPACITY_OBSERVATION_LOCK:
        observed, provenance, observed_at = _VOICE_REST_CAPACITY_OBSERVATION
    if observed > 1:
        age = (
            time.monotonic() - observed_at
            if isinstance(observed_at, (int, float))
            else math.inf
        )
        if not math.isfinite(age) or age < 0 or age > _voice_rest_capacity_proof_ttl_s():
            return 1, "concurrency_proof_stale"
    return max(1, min(configured, observed)), provenance


def _voice_rest_capacity() -> int:
    """Observed concurrent REST-TTS capacity, never configuration alone."""

    return _voice_rest_capacity_snapshot()[0]


def _voice_rest_busy_retries() -> int:
    """Bounded retries after an explicit upstream HTTP 429/model-busy result."""

    try:
        value = int(os.environ.get("MS4_TTS_REST_BUSY_RETRIES", "1").strip())
    except (TypeError, ValueError):
        value = 1
    return max(0, min(value, 3))


def _voice_rest_busy_retry_delay() -> float:
    try:
        value = float(os.environ.get("MS4_TTS_REST_BUSY_RETRY_DELAY_S", "0.075").strip())
    except (TypeError, ValueError):
        value = 0.075
    return max(0.0, min(value, 1.0))


def _xtts_token_budget() -> int:
    """Conservative per-fragment budget, always below XTTS's hard 400."""

    try:
        value = int(
            os.environ.get(
                "MS4_VOICE_XTTS_TOKEN_BUDGET",
                str(DEFAULT_XTTS_TOKEN_BUDGET),
            ).strip()
        )
    except (TypeError, ValueError):
        value = DEFAULT_XTTS_TOKEN_BUDGET
    return max(16, min(value, XTTS_HARD_TOKEN_LIMIT - 1))


_XTTS_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_XTTS_NATURAL_TOKEN_BREAKS = frozenset(".!?;:,)]}\u2014\u2013-")
_XTTS_TOKENIZER_LOCK = threading.Lock()
_XTTS_TOKENIZER: Any | None = None
_XTTS_TOKENIZER_RESOLVED = False


def _xtts_vocab_candidates() -> list[Path]:
    candidates: list[Path] = []
    if explicit := os.environ.get("MS4_VOICE_XTTS_VOCAB"):
        candidates.append(Path(explicit))
    roots = [
        os.environ.get("HIVEMIND_MODEL_REPO"),
        r"C:\HiveMind\model_repo",
        r"E:\HiveMind\model_repo",
    ]
    for root in roots:
        if root:
            candidates.append(
                Path(root)
                / "tts"
                / "tts_models--multilingual--multi-dataset--xtts_v2"
                / "vocab.json"
            )
    return candidates


def _native_xtts_tokenizer() -> Any | None:
    """Lazily load the exact tokenizer vocabulary used by deployed XTTS-v2."""

    global _XTTS_TOKENIZER, _XTTS_TOKENIZER_RESOLVED
    if _XTTS_TOKENIZER_RESOLVED:
        return _XTTS_TOKENIZER
    with _XTTS_TOKENIZER_LOCK:
        if _XTTS_TOKENIZER_RESOLVED:
            return _XTTS_TOKENIZER
        try:
            from tokenizers import Tokenizer

            vocab = next(
                (candidate for candidate in _xtts_vocab_candidates() if candidate.is_file()),
                None,
            )
            if vocab is not None:
                _XTTS_TOKENIZER = Tokenizer.from_file(str(vocab))
        except Exception as exc:  # noqa: BLE001 - fail-closed fallback below
            log.warning("native XTTS tokenizer unavailable; using fail-closed splitting: %s", exc)
        _XTTS_TOKENIZER_RESOLVED = True
        return _XTTS_TOKENIZER


def _fail_closed_xtts_token_upper_bound(text: str) -> int:
    """Conservative bound used when the runtime has no native tokenizer.

    BPE cannot emit more units than its prepared UTF-8 byte alphabet. Account
    for XTTS's ``[en]`` / ``[SPACE]`` preparation exactly, then reserve another
    25 percent plus eight units for cleaner expansion and fixed markers. With
    the default 360-unit budget this admits at most 281 prepared bytes, leaving
    at least 39 units below XTTS's hard 400-token ceiling without degenerating
    into one-code-point fragments.
    """

    cleaned = " ".join((text or "").split()).lower()
    prepared = f"[en]{cleaned}".replace(" ", "[SPACE]")
    prepared_bytes = len(prepared.encode("utf-8", errors="replace"))
    return ((prepared_bytes * 5 + 3) // 4) + 8


def _estimate_xtts_tokens(text: str) -> int:
    tokenizer = _native_xtts_tokenizer()
    if tokenizer is None:
        return _fail_closed_xtts_token_upper_bound(text)
    try:
        # VoiceBpeTokenizer's English path lowercases/collapses whitespace,
        # prepends [en], then replaces spaces before invoking this tokenizer.
        cleaned = " ".join((text or "").split()).lower()
        prepared = f"[en]{cleaned}".replace(" ", "[SPACE]")
        return len(tokenizer.encode(prepared).ids)
    except Exception as exc:  # noqa: BLE001 - never undercount on tokenizer error
        log.warning("native XTTS token count failed; using fail-closed splitting: %s", exc)
        return _fail_closed_xtts_token_upper_bound(text)


def _xtts_safe_cut(text: str, token_budget: int | None = None) -> int | None:
    """Return a token-boundary cut when ``text`` exceeds the XTTS budget.

    The latest natural speech boundary in the safe half of the budget wins;
    otherwise the latest complete lexical token wins. A single pathological
    token is split by a bounded binary search using the same token estimator.
    """

    budget = token_budget or _xtts_token_budget()
    if _estimate_xtts_tokens(text) <= budget:
        return None
    last_end = 0
    natural_end = 0
    for match in _XTTS_TOKEN_RE.finditer(text):
        end = match.end()
        while end < len(text) and text[end].isspace():
            end += 1
        if end >= len(text):
            continue
        if _estimate_xtts_tokens(text[:end]) <= budget:
            last_end = end
            if match.group(0) in _XTTS_NATURAL_TOKEN_BREAKS:
                natural_end = end
    if (
        natural_end
        and _estimate_xtts_tokens(text[:natural_end]) >= max(1, budget // 2)
    ):
        return natural_end
    if last_end:
        return last_end

    # One lexical token itself exceeded the budget. Contract until a directly
    # measured native-safe prefix is found, then recover the largest safe prefix
    # in that bounded interval. Without a tokenizer the fail-closed upper bound
    # contracts all the way to one Unicode code point.
    high = min(max(1, len(text) - 1), budget)
    while high > 1 and _estimate_xtts_tokens(text[:high]) > budget:
        high = max(1, high // 2)
    if _estimate_xtts_tokens(text[:high]) > budget:
        return 1
    safe = high
    low = high + 1
    high = min(len(text) - 1, high * 2)
    while low <= high:
        middle = (low + high) // 2
        if _estimate_xtts_tokens(text[:middle]) <= budget:
            safe = middle
            low = middle + 1
        else:
            high = middle - 1
    return safe


def _split_xtts_safe_fragments(text: str) -> list[str]:
    """Split sanitized speech text into ordered fragments below XTTS's limit."""

    remaining = text or ""
    if not remaining.strip():
        return []
    fragments: list[str] = []
    while remaining:
        cut = _xtts_safe_cut(remaining)
        if cut is None:
            fragments.append(remaining)
            break
        fragments.append(remaining[:cut])
        remaining = remaining[cut:]
    return fragments


def _is_transient_tts_busy(exc: BaseException) -> bool:
    status = next(
        (
            value
            for value in (
                getattr(exc, "code", None),
                getattr(exc, "status", None),
                getattr(exc, "status_code", None),
            )
            if value is not None
        ),
        None,
    )
    if status == 429 or str(status) == "429":
        return True
    message = str(exc).lower()
    return "429" in message and any(
        marker in message
        for marker in ("busy", "admission", "too many", "capacity", "rate limit")
    )


def _call_tts_with_busy_retry(
    call: Callable[..., dict[str, Any]],
    kwargs: dict[str, Any],
    *,
    turn_cancel: threading.Event,
    client_alive: threading.Event | None = None,
    on_retry: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Retry only explicit model-busy admission failures, preserving order."""

    retries = _voice_rest_busy_retries()
    for attempt in range(retries + 1):
        if turn_cancel.is_set() or (client_alive is not None and not client_alive.is_set()):
            raise VoiceUnavailable("REST TTS cancelled before synthesis")
        try:
            return call(**kwargs)
        except Exception as exc:  # noqa: BLE001 - exact transient classification below
            if not _is_transient_tts_busy(exc) or attempt >= retries:
                raise
            if on_retry is not None:
                on_retry()
            delay = _voice_rest_busy_retry_delay() * (attempt + 1)
            if delay > 0 and turn_cancel.wait(delay):
                raise VoiceUnavailable("REST TTS cancelled during busy backpressure") from exc
    raise AssertionError("unreachable busy retry loop")


def _voice_rest_hol_hedge_delay() -> float:
    """Delay before one later-ready chunk may hedge the blocked in-order head."""
    try:
        val = float(os.environ.get("MS4_TTS_REST_HOL_HEDGE_DELAY_S", "1.0").strip())
    except (TypeError, ValueError):
        val = 1.0
    return max(0.0, val)


def _voice_rest_hol_fatal_timeout() -> float:
    """Fatal ceiling for an unresolved in-order head once later audio is ready.

    The default must cover a legitimate slow chunk, not only the warm-path
    median.  A live isolated request for the exact 18-word chunk rejected by
    the previous 20s ceiling completed successfully in 41.187s.  Fifty-five
    seconds keeps useful margin for that deployed path while remaining below
    the separate 60s progress-stall ceiling.  An explicit operator value
    remains exact so constrained deployments and deterministic tests can fail
    sooner.
    """
    default_s = 55.0
    configured = os.environ.get("MS4_TTS_REST_HOL_FATAL_TIMEOUT_S")
    if configured is None:
        return default_s
    try:
        val = float(configured.strip())
    except (TypeError, ValueError):
        val = default_s
    if not math.isfinite(val):
        val = default_s
    return max(0.05, val)


def _accepts_keyword(callable_obj: Callable[..., Any], keyword: str) -> bool:
    try:
        parameters = inspect.signature(callable_obj).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == keyword or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _call_speaker_identify(
    identify: Callable[..., dict[str, Any] | None],
    hivemind_url: str,
    audio: bytes,
    *,
    timeout: float,
    cancel_event: threading.Event,
) -> dict[str, Any] | None:
    kwargs: dict[str, Any] = {}
    if _accepts_keyword(identify, "timeout"):
        kwargs["timeout"] = timeout
    if _accepts_keyword(identify, "cancel_event"):
        kwargs["cancel_event"] = cancel_event
    return identify(hivemind_url, audio, **kwargs)


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
    cancel_event = threading.Event()

    def _run() -> None:
        try:
            from . import voice_identity as _vi

            holder["speaker"] = _call_speaker_identify(
                _vi.identify_speaker_from_wav,
                hivemind_url,
                audio,
                timeout=budget_s,
                cancel_event=cancel_event,
            )
        except Exception as exc:  # noqa: BLE001 — fail-soft per design
            log.info("speaker identification skipped: %s", exc)

    t = threading.Thread(target=_run, name="ms4-speaker-id", daemon=True)
    t.start()
    t.join(timeout=budget_s)
    if t.is_alive():
        cancel_event.set()
        t.join(timeout=_voice_speaker_id_terminal_budget())
        if t.is_alive():
            log.warning("bounded speaker identification did not retire after cancellation")
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


def _identify_speaker_async(
    hivemind_url: str,
    audio: bytes,
    emit: Callable[[str, dict[str, Any]], bool],
    cancel_event: threading.Event | None = None,
    timeout: float | None = None,
) -> threading.Thread | None:
    """Fire speaker identification on a daemon thread and emit a late
    ``speaker`` event when (if) it resolves. Returns IMMEDIATELY — never
    on the critical path. Fail-soft: a failed/empty identify emits
    nothing (the transcript already carried ``speaker: None``)."""
    if not audio:
        return None
    turn_cancel = cancel_event or threading.Event()
    call_timeout = _voice_speaker_id_budget() if timeout is None else max(0.05, timeout)

    def _run() -> None:
        try:
            from . import voice_identity as _vi

            speaker = _call_speaker_identify(
                _vi.identify_speaker_from_wav,
                hivemind_url,
                audio,
                timeout=call_timeout,
                cancel_event=turn_cancel,
            )
            if speaker and not turn_cancel.is_set():
                emit("speaker", {"speaker": speaker})
        except Exception as exc:  # noqa: BLE001 — fail-soft per design
            log.info("async speaker identification skipped: %s", exc)

    thread = threading.Thread(target=_run, name="ms4-speaker-id-async", daemon=True)
    thread.start()
    return thread


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


def _voice_reflex_classifier_budget() -> float:
    """Optional model-classifier budget for the pre-reply acknowledgment.

    The default is zero because the deterministic intent heuristic is already
    sufficient to choose a safe canned response and must land immediately after
    ASR. Operators can opt into model refinement, accepting the added dead air.
    """
    try:
        value = float(os.environ.get("MS4_VOICE_REFLEX_CLASSIFIER_BUDGET_S", "0").strip())
    except (TypeError, ValueError):
        value = 0.0
    return max(0.0, value)


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
    *,
    emit: Callable[[str, dict[str, Any]], bool],
    hivemind_url: str,
    transcript: str,
    session_id: str | None,
    turn_started_at: float | None = None,
) -> None:
    """Classify the utterance and emit a `reflex` SSE event naming a
    context-appropriate, varied canned acknowledgment. Best-effort: any
    failure simply means no reflex this turn. Designed to run on a daemon
    thread concurrently with the (slow) Face Lobe reply so the reflex
    lands during the dead air."""
    try:
        from .canned_reflexes import pick_reflex_for_intent, INTENT_TO_CATEGORY

        classify_started_at = time.monotonic()
        intent, source = classify_voice_intent(
            hivemind_url,
            transcript,
            timeout=_voice_reflex_classifier_budget(),
        )
        reflex_id = pick_reflex_for_intent(intent, exclude=_recent_reflexes(session_id))
        if not reflex_id:
            return
        _record_reflex(session_id, reflex_id)
        payload: dict[str, Any] = {
            "id": reflex_id,
            "category": INTENT_TO_CATEGORY.get(intent, ""),
            "intent": intent,
            "source": source,
            "classification_ms": int((time.monotonic() - classify_started_at) * 1000),
        }
        if turn_started_at is not None:
            payload["turn_ms"] = int((time.monotonic() - turn_started_at) * 1000)
        emit("reflex", payload)
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


class _FaceChatAbortSignal(threading.Event):
    """Cancellation fence carrying one aggregate first-token deadline."""

    def __init__(self, *, first_token_deadline: float) -> None:
        super().__init__()
        self.first_token_deadline = first_token_deadline
        self.request_started_at: float | None = None
        self.selected_model: str | None = None
        self._request_start_lock = threading.Lock()
        self.buffered_progress = threading.Event()

    def remaining_first_token_budget(self) -> float:
        return max(0.0, self.first_token_deadline - time.monotonic())

    def mark_request_started(self, model: str | None) -> None:
        with self._request_start_lock:
            if self.request_started_at is not None:
                return
            self.selected_model = str(model).strip() if model else None
            self.request_started_at = time.monotonic()

    def mark_buffered_progress(self) -> None:
        """Stand down the stall watchdog without releasing held model text.

        Face may intentionally buffer a high-risk follow-up until its semantic
        contract passes. A real upstream token proves the model is alive even
        though that unvalidated text must not reach the voice/TTS callback.
        """
        self.buffered_progress.set()


def _run_facechat_guarded(
    *,
    chat_call: Callable[..., dict[str, Any]],
    transcript: str,
    chat_kwargs: dict[str, Any],
    on_delta: Callable[[str], bool | None],
    first_token_timeout: float,
    model: str | None,
    cancel_event: "threading.Event | None" = None,
    client_alive: "threading.Event | None" = None,
    join_timeout: float | None = None,
) -> dict[str, Any]:
    """Run the blocking, streaming Face Lobe ``chat_call`` on a worker
    thread guarded by a first-token watchdog.

    Returns the chat response when the model starts streaming (or returns
    quickly). Raises :class:`FaceLobeStalled` if NO first token arrives
    within ``first_token_timeout`` AND the call has not returned — the
    model is wedged and the turn must fail over rather than hang.

    Routing/preparation and the actual model first-token wait share one
    monotonic deadline. ``FaceLobeChat`` marks the first real model request
    through the cancellation fence for reporting, but that boundary never
    renews or increases the configured aggregate budget.
    Once any token is seen the watchdog stands down and we wait for the
    full (possibly slow) completion. Late tokens that arrive after a bail
    are dropped so they can't emit audio onto an already-closed stream.
    """
    first_token = threading.Event()
    done = threading.Event()
    bailed = threading.Event()
    first_token_deadline = time.monotonic() + max(0.0, first_token_timeout)
    abort_request = _FaceChatAbortSignal(
        first_token_deadline=first_token_deadline,
    )
    holder: dict[str, Any] = {}

    # Finding 2: the guard observes the TURN's cancellation fence (barge /
    # client disconnect), not only its private abort. A cancelled turn retires
    # the back-lobe worker and NEVER waits on an unbounded join.
    def _turn_gone() -> bool:
        return (
            (cancel_event is not None and cancel_event.is_set())
            or (client_alive is not None and not client_alive.is_set())
        )

    join_grace = join_timeout if join_timeout is not None else _voice_facechat_join_timeout()

    def _guarded_delta(delta: str) -> bool:
        first_token.set()
        if bailed.is_set() or _turn_gone():
            abort_request.set()
            return False  # turn failed over / cancelled; stop upstream work
        if on_delta(delta) is False:
            abort_request.set()
            return False
        return True

    def _worker() -> None:
        try:
            kw = dict(chat_kwargs)
            kw["stream_callback"] = _guarded_delta
            try:
                parameters = inspect.signature(chat_call).parameters.values()
                supports_cancel = any(
                    parameter.name == "cancel_event"
                    or parameter.kind == inspect.Parameter.VAR_KEYWORD
                    for parameter in parameters
                )
            except (TypeError, ValueError):
                supports_cancel = False
            if supports_cancel:
                kw["cancel_event"] = abort_request
            holder["resp"] = chat_call(transcript, **kw)
        except BaseException as exc:  # noqa: BLE001 — surfaced to the turn thread
            holder["err"] = exc
        finally:
            done.set()

    th = threading.Thread(target=_worker, name="ms4-facechat", daemon=True)
    th.start()

    while (
        not first_token.is_set()
        and not abort_request.buffered_progress.is_set()
        and not done.is_set()
    ):
        remaining = abort_request.remaining_first_token_budget()
        if remaining <= 0:
            break
        if _turn_gone():
            # Finding 2: browser aborted before first token — retire the worker
            # with a bounded join rather than waiting out the watchdog budget.
            bailed.set()
            abort_request.set()
            th.join(join_grace)
            raise VoiceUnavailable("voice Face Lobe turn cancelled before first token")
        time.sleep(min(0.05, remaining))

    if (
        not first_token.is_set()
        and not abort_request.buffered_progress.is_set()
        and not done.is_set()
    ):
        bailed.set()
        abort_request.set()
        reported_model = abort_request.selected_model or model
        log.warning(
            "voice Face Lobe model=%r produced no first token within %.0fs; "
            "failing over to error reflex (set MS4_VOICE_FIRST_TOKEN_TIMEOUT_S to tune)",
            reported_model, first_token_timeout,
        )
        # Bounded join: give the worker a chance to observe abort_request and
        # retire so it does not outlive the turn as a stranded daemon.
        th.join(join_grace)
        raise FaceLobeStalled(
            f"Face Lobe model {reported_model or '(auto)'} did not respond within "
            f"{int(first_token_timeout)}s"
        )

    # First token seen (or the call already returned). Wait for completion, but
    # honor cancellation with a bounded join so a barged/timed-out turn cannot
    # outlive the handler cleanup budget (Finding 2: no unbounded join).
    while th.is_alive():
        th.join(0.1)
        if not th.is_alive():
            break
        if _turn_gone() or bailed.is_set():
            bailed.set()
            abort_request.set()
            th.join(join_grace)
            break
    if th.is_alive():
        raise FaceLobeStalled(
            f"Face Lobe model {model or '(auto)'} did not retire within "
            f"{join_grace:.0f}s of cancellation"
        )
    if holder.get("err") is not None:
        raise holder["err"]
    response = holder.get("resp") or {}
    if response.get("completed") is not True or response.get("cancelled") is True:
        abort_request.set()
        raise VoiceUnavailable(
            "voice Face Lobe stream ended before verified completion"
        )
    return response


# ---------------------------------------------------------------------------
# Fail-closed gate (MS3 /voice/status)
# ---------------------------------------------------------------------------


def _hivemind_asr_fallback(
    hivemind_url: str | None,
    *,
    ms3_unreachable: bool = False,
    ms3_voice_status: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Confirm voice readiness directly against HiveMind ASR.

    MS3 is the identity/consciousness sidecar; the actual ASR runs on
    HiveMind. When MS3 is down OR reachable but stale/misconfigured, a
    healthy HiveMind ASR should keep the voice front door usable while
    marking MS3-owned identity/speaker features as degraded. Returns a
    synthetic ready payload if HiveMind ASR is healthy, else ``None``
    (caller fails closed)."""
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
            "MS3 /voice/status not usable; proceeding on HEALTHY HiveMind ASR "
            "(voice works; MS3 identity/speaker features degraded until MS3 is back)"
        )
        result: dict[str, Any] = {
            "voice_input_ready": True,
            "source": "hivemind-asr-fallback",
            "asr": st,
        }
        if ms3_unreachable:
            result["ms3_unreachable"] = True
        if ms3_voice_status is not None:
            result["ms3_reported_not_ready"] = True
            result["ms3_voice_status"] = ms3_voice_status
        return result
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
            fallback = _hivemind_asr_fallback(hivemind_url, ms3_unreachable=True)
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
        fallback = _hivemind_asr_fallback(hivemind_url, ms3_voice_status=payload)
        if fallback is not None:
            return _cache_ok(fallback)
        asr_detail = ""
        asr = payload.get("asr")
        if isinstance(asr, dict):
            asr_detail = f" asr.status={asr.get('status')} detail={asr.get('detail')}"
        raise VoiceUnavailable(f"voice_input_ready=false{asr_detail}")
    return _cache_ok(payload)


# ---------------------------------------------------------------------------
# HiveMind ASR / TTS bridges
# ---------------------------------------------------------------------------


_PTT_ASR_BRAND_ALIAS_RE = re.compile(
    r"(?<!\w)(?:HiveMine|Hive Mine)(?!\w)",
    re.IGNORECASE,
)


def _ptt_transcript_from_asr(asr_result: dict[str, Any]) -> tuple[str, str]:
    """Return canonical downstream text plus the untouched ASR transcript.

    This is the single normalization boundary shared by streaming and
    non-streaming PTT. The bounded aliases are ASR-only brand corrections:
    whole ``HiveMine`` / ``Hive Mine`` tokens become ``HiveMind`` while
    substrings such as ``HiveMiner`` and unrelated prose remain untouched.
    """
    raw_transcript = str(asr_result.get("text") or "")
    transcript = _PTT_ASR_BRAND_ALIAS_RE.sub("HiveMind", raw_transcript).strip()
    if transcript != raw_transcript.strip():
        log.info("normalized spoken ASR brand alias for downstream PTT routing")
    return transcript, raw_transcript


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


def _decode_chunked_http_body(data: bytes) -> bytes | None:
    decoded = bytearray()
    position = 0
    while True:
        line_end = data.find(b"\r\n", position)
        if line_end < 0:
            return None
        size_text = data[position:line_end].split(b";", 1)[0].strip()
        try:
            size = int(size_text, 16)
        except ValueError as exc:
            raise VoiceUnavailable("HiveMind /v1/audio/speech returned invalid chunk framing") from exc
        position = line_end + 2
        if size == 0:
            if len(data) < position + 2:
                return None
            return bytes(decoded)
        if len(data) < position + size + 2:
            return None
        decoded.extend(data[position:position + size])
        position += size
        if data[position:position + 2] != b"\r\n":
            raise VoiceUnavailable("HiveMind /v1/audio/speech returned invalid chunk terminator")
        position += 2


def _resolve_addresses_cancellable(
    host: str,
    port: int,
    *,
    deadline: float,
    cancel_event: threading.Event,
) -> list[tuple[Any, ...]]:
    done = threading.Event()
    holder: dict[str, Any] = {}

    def _resolve() -> None:
        try:
            holder["addresses"] = socket.getaddrinfo(
                host,
                port,
                type=socket.SOCK_STREAM,
            )
        except OSError as exc:
            holder["error"] = exc
        finally:
            done.set()

    threading.Thread(target=_resolve, name="ms4-dns-resolve", daemon=True).start()
    while not done.wait(0.02):
        if cancel_event.is_set():
            raise VoiceUnavailable("HiveMind /v1/audio/speech cancelled during DNS resolution")
        if time.monotonic() >= deadline:
            raise TimeoutError("DNS resolution timed out")
    if holder.get("error") is not None:
        raise holder["error"]
    return list(holder.get("addresses") or [])


# ---------------------------------------------------------------------------
# Non-secret HLI provenance retention + fail-closed secret exclusion.
#
# HiveMind returns provenance ALONGSIDE a /v1/audio/speech reply (server-minted
# request id, routing, timing). We surface ONLY an allowlist of non-secret
# provenance under a ``provenance`` mapping and fail closed against every
# secret-named or secret-valued header (Authorization, Cookie/Set-Cookie,
# api-key, token, credential, bearer, JWT, opaque credential material) on every
# serialized surface. The allowlist maps the R3-verified live HLI header
# spellings AND the accepted contract fixture spellings to canonical keys; no
# header name is invented (any name not in the map is dropped by omission).
# ---------------------------------------------------------------------------

# Response header (lowercased) -> canonical, non-secret provenance key.
_HLI_PROVENANCE_HEADER_MAP: dict[str, str] = {
    "x-request-id": "request_id",
    "x-hivemind-served-by": "served_by",
    "x-hivemind-location": "location",
    "x-hivemind-endpoint": "endpoint",
    "x-hivemind-source": "source",
    "x-hivemind-peer": "peer",
    "x-hivemind-gateway": "gateway",
    "x-hivemind-gateway-version": "gateway_version",
    "x-hivemind-model": "model",
    "x-hivemind-node-id": "node_id",
    "x-hivemind-timing-ms": "timing_ms",
    "x-hivemind-queue-path": "queue_path",
    "x-hivemind-queue-depth-total": "queue_depth_total",
    "x-hivemind-queue-enabled": "queue_enabled",
    "x-hivemind-queue-max-depth": "queue_max_depth",
    "x-hivemind-queue-per-endpoint-cap": "queue_per_endpoint_cap",
    "x-tts-path": "tts_path",
    "x-queue-path": "queue_path",
}
# Canonical provenance keys permitted onto a serialized surface.
_HLI_PROVENANCE_KEY_ALLOWLIST: frozenset[str] = frozenset(
    _HLI_PROVENANCE_HEADER_MAP.values()
)
# Secret-indicating NAME substrings (fail closed regardless of allowlist).
_SECRET_NAME_SUBSTRINGS: tuple[str, ...] = (
    "authorization", "cookie", "api-key", "api_key", "apikey",
    "token", "credential", "secret", "password", "passwd", "bearer",
    "jwt", "auth",
)
# Known credential/token VALUE prefixes (compared case-insensitively). Each is
# distinctive -- it carries a separator or provider marker -- so ordinary ids,
# hosts, paths, and region names never match. Deliberately EXCLUDES bare-alpha
# provider prefixes (e.g. AWS "ASIA"/"AKIA") that collide with legitimate values
# such as region names ("asia-east1"); those can be added if a live header
# capture ever shows them.
_SECRET_VALUE_PREFIXES: tuple[str, ...] = (
    "sk-", "sk_", "sk-ant-", "sk-proj-", "hf_", "xoxb-", "xoxp-", "xoxa-",
    "xoxr-", "xoxs-", "xoxe-", "xapp-", "ghp_", "gho_", "ghu_", "ghs_",
    "ghr_", "github_pat_", "glpat-", "gsk_", "shpat_", "shpss_", "ya29.",
)
# Sensitive URL query/fragment parameter-name tokens (normalized: lowercased,
# non-alphanumerics stripped). A value that embeds one of these as a query or
# fragment PARAMETER NAME carries a secret and is dropped.
_SECRET_QUERY_PARAM_TOKENS: tuple[str, ...] = (
    "apikey", "accesstoken", "refreshtoken", "idtoken", "clientsecret",
    "sessiontoken", "privatekey", "token", "secret", "signature",
    "credential", "password", "passwd",
)
# Exact-match set for URL parameter-NAME classification. A sensitive parameter name
# is matched by EXACT normalized token -- either the whole non-alphanumeric-stripped
# name (api_key -> "apikey") or the trailing separator-delimited component
# (X-Amz-Signature -> "signature") -- NOT a substring collision, so a legitimate
# near-miss name (tokenizer / signature_algorithm / secretariat) is preserved.
_SECRET_QUERY_PARAM_TOKENS_SET: frozenset[str] = frozenset(_SECRET_QUERY_PARAM_TOKENS)
# Fail-closed recursion bounds for secret inspection. Real HLI provenance is a flat
# mapping of string values (depth 0-1, a few dozen entries); these ceilings sit far
# above any legitimate provenance yet bound adversarial cyclic/deep/huge inputs so
# inspection can never raise (e.g. RecursionError) or run unbounded. A cycle, a
# depth overflow, or a node-budget overflow fails CLOSED (treated as secret ->
# omitted), never by exception.
_SECRET_SCAN_MAX_DEPTH: int = 8
_SECRET_SCAN_MAX_NODES: int = 2048


def _param_name_is_sensitive(raw_name: str) -> bool:
    """True when a URL query/fragment parameter NAME is sensitive by EXACT normalized
    token: the whole non-alphanumeric-stripped name equals a token (api_key ->
    "apikey"), OR the trailing separator-delimited component equals a token
    (X-Amz-Signature -> "signature"). A substring collision (tokenizer,
    signature_algorithm, secretariat) is NOT sensitive."""
    lowered = raw_name.lower()
    whole = re.sub(r"[^a-z0-9]", "", lowered)
    if not whole:
        return False
    if whole in _SECRET_QUERY_PARAM_TOKENS_SET:
        return True
    components = [c for c in re.split(r"[^a-z0-9]+", lowered) if c]
    return bool(components) and components[-1] in _SECRET_QUERY_PARAM_TOKENS_SET


def _scalar_str_has_secret(value: str) -> bool:
    """Known credential material in a PLAIN string (no URL-parameter recursion):
    auth-scheme material (Bearer/Basic/Digest, any case), known short credential/
    token value prefixes, JWT-shaped triples, and long opaque credential/hash blobs.
    Used both directly and to inspect a decoded URL parameter VALUE."""
    v = value.strip()
    if not v:
        return False
    low = v.lower()
    if low.startswith(("bearer ", "basic ", "digest ")) or "bearer " in low:
        return True
    if low.startswith(_SECRET_VALUE_PREFIXES):
        return True  # known credential/token value prefix
    parts = v.split(".")
    if len(parts) == 3 and all(re.fullmatch(r"[A-Za-z0-9_-]{8,}", p) for p in parts):
        return True  # JWT-shaped
    if len(v) >= 64 and re.fullmatch(r"[A-Za-z0-9+/=_-]+", v):
        return True  # long opaque credential/hash blob
    return False


def _url_value_has_secret_param(value: str) -> bool:
    """True when ``value`` embeds a secret in its query/fragment. Only inspects text
    AFTER the first ``?``/``#`` (ordinary hosts/paths are never scanned). A parameter
    is secret when its NAME is a sensitive EXACT token (see ``_param_name_is_sensitive``,
    which rejects substring collisions such as tokenizer/signature_algorithm/
    secretariat) OR its DECODED VALUE carries known credential scheme/prefix/JWT/opaque
    material (closes the mid-URL secret-VALUE escape under an ordinary parameter name).
    Never broadens logging and never serializes a suspicious value into an error."""
    cuts = [i for i in (value.find("?"), value.find("#")) if i != -1]
    if not cuts:
        return False
    tail = value[min(cuts):]
    for part in re.split(r"[?#&;]", tail):
        if "=" not in part:
            continue
        raw_name, raw_value = part.split("=", 1)
        if _param_name_is_sensitive(raw_name):
            return True  # sensitive parameter NAME (exact token)
        try:
            decoded_value = urllib.parse.unquote_plus(raw_value)
        except Exception:
            decoded_value = raw_value
        if _scalar_str_has_secret(decoded_value):
            return True  # known credential material in a parameter VALUE
    return False


def _string_looks_secret(value: str) -> bool:
    """Fail-closed secret check for a plain string: known scheme/prefix/JWT/opaque
    material, OR a URL carrying a sensitive query/fragment secret (name or value)."""
    if _scalar_str_has_secret(value):
        return True
    v = value.strip()
    if ("?" in v or "#" in v) and _url_value_has_secret_param(v):
        return True  # URL query/fragment secret parameter
    return False


def _looks_secret_value(value: Any) -> bool:
    """Conservative, EXPLICITLY BOUNDED, fail-closed value check.

    Scalars: ordinary JSON scalars remain usable -- a ``str`` is scanned for
    scheme/prefix/JWT/opaque/URL secrets; ``None``/``bool``/``int`` and finite
    ``float`` are non-secret ordinary JSON scalars, while a non-finite ``float``
    (``NaN``/``+Inf``/``-Inf``) is NOT JSON-safe and fails closed by omission.
    Containers: only JSON-native containers (list/tuple/dict, and secret-named
    nested keys) are RECURSED so a secret hidden inside an allowlisted field's
    nested value fails closed. A non-JSON-native set/frozenset is NOT recursed --
    it fails closed by omission via the unknown-object path (it cannot serialize
    under strict json.dumps / Starlette JSONResponse).

    Bounded and fail-closed: recursion is guarded by a per-path visited-id set
    (cycle detection), a maximum depth, and a global node budget. A cycle, a
    depth/size overflow, a non-JSON-native container (set/frozenset), a
    non-JSON-safe byte-like/unknown object, or ANY unexpected exception is OMITTED
    (treated as secret -> dropped) rather than raising. The
    production call sites feed string-only header mappings, but the inspector must
    fail closed by OMISSION -- never by exception -- for every caller. Deliberately
    narrow on plain strings so ordinary non-secret provenance (ids, routing URLs,
    IPs, timings, models, hostnames) is never scrubbed."""
    try:
        return _looks_secret_value_bounded(value, 0, set(), [_SECRET_SCAN_MAX_NODES])
    except Exception:
        return True  # fail closed: anything we cannot cleanly inspect is omitted


def _looks_secret_value_bounded(
    value: Any, depth: int, seen: set[int], budget: list[int]
) -> bool:
    """Bounded recursive worker for :func:`_looks_secret_value` (see its contract).
    ``depth`` overflow, ``budget`` exhaustion, and cycles all fail closed (return
    True -> omit). ``seen`` holds the ids on the CURRENT path only (added on descent,
    discarded on ascent) so a DAG is not misread as a cycle."""
    if budget[0] <= 0 or depth > _SECRET_SCAN_MAX_DEPTH:
        return True  # node-budget / depth overflow -> fail closed
    budget[0] -= 1
    if isinstance(value, (list, tuple)):
        # Only JSON-native sequences (list/tuple both serialize as JSON arrays)
        # are traversed/retained. A non-JSON-native set/frozenset is deliberately
        # NOT matched here: it falls through to the unknown-object path below and
        # fails closed by omission (strict json.dumps/JSONResponse would otherwise
        # raise TypeError on it), honoring this inspector's fail-closed contract.
        marker = id(value)
        if marker in seen:
            return True  # cycle -> fail closed
        seen.add(marker)
        try:
            return any(
                _looks_secret_value_bounded(item, depth + 1, seen, budget)
                for item in value
            )
        finally:
            seen.discard(marker)
    if isinstance(value, dict):
        marker = id(value)
        if marker in seen:
            return True  # cycle -> fail closed
        seen.add(marker)
        try:
            for key, nested in value.items():
                # Fail closed on any mapping KEY that JSON cannot serialize as an
                # object key: strict json.dumps / Starlette JSONResponse accept
                # ONLY str/int/float/bool/None keys, and reject a non-finite float
                # key under allow_nan=False. A tuple/frozenset/bytes/arbitrary
                # object key would raise TypeError, and a NaN/+-Inf key would raise
                # ValueError, so the whole mapping is OMITTED (never stringified,
                # never mutated) rather than retained into a value that cannot
                # serialize. isinstance short-circuits before math.isfinite (a huge
                # int is a valid JSON key, and math.isfinite raises OverflowError
                # on an int too large to convert to float).
                if (
                    not (key is None or isinstance(key, (str, int, float)))
                    or (isinstance(key, float) and not math.isfinite(key))
                ):
                    return True  # non-JSON-coercible mapping key -> fail closed (omit)
                name = str(key).strip().lower()
                if any(sub in name for sub in _SECRET_NAME_SUBSTRINGS):
                    return True
                if _looks_secret_value_bounded(nested, depth + 1, seen, budget):
                    return True
            return False
        finally:
            seen.discard(marker)
    if value is None or isinstance(value, bool):
        return False  # ordinary JSON scalar (bool before int; JSON-safe, non-secret)
    if isinstance(value, (int, float)):
        # Finite int/float are ordinary JSON scalars. A non-finite float
        # (NaN/+Inf/-Inf) is NOT JSON-safe -- strict json.dumps and Starlette
        # JSONResponse serialize with allow_nan=False and raise -- so it fails
        # closed by omission. int short-circuits before math.isfinite (an int is
        # always JSON-safe, and math.isfinite raises OverflowError on an int too
        # large to convert to float).
        return isinstance(value, float) and not math.isfinite(value)
    if isinstance(value, str):
        return _string_looks_secret(value)
    return True  # non-JSON-native (set/frozenset), byte-like, or unknown object -> fail closed (omit)


def _sanitize_hli_provenance(prov: Any) -> dict[str, Any]:
    """Retain ONLY allowlisted, non-secret provenance keys from a provenance
    mapping; drop every secret-named or secret-valued entry (fail closed).
    Returns ``{}`` for anything that is not a mapping."""
    if not isinstance(prov, dict):
        return {}
    clean: dict[str, Any] = {}
    for key, value in prov.items():
        k = str(key).strip().lower()
        if k not in _HLI_PROVENANCE_KEY_ALLOWLIST:
            continue
        if any(sub in k for sub in _SECRET_NAME_SUBSTRINGS):
            continue
        if _looks_secret_value(value):
            continue
        clean[k] = value.strip() if isinstance(value, str) else value
    return clean


def _extract_response_headers(headers: Any) -> list[tuple[str, str]]:
    """Enumerate response headers as (name, value) pairs when possible. Falls
    back to an empty list for header objects that do not expose ``items()`` (so
    provenance is simply absent rather than raising)."""
    items = getattr(headers, "items", None)
    if callable(items):
        try:
            return list(items())
        except Exception:
            return []
    return []


def _provenance_from_response_headers(headers: Any) -> dict[str, Any]:
    """Build a non-secret HLI provenance mapping from HTTP response headers via
    the header->canonical allowlist. Secret headers are dropped by omission,
    then values are scrubbed defensively. Accepts a mapping or an iterable of
    (name, value) pairs."""
    if isinstance(headers, dict):
        items: Any = headers.items()
    else:
        items = headers or ()
    prov: dict[str, Any] = {}
    for name, value in items:
        canonical = _HLI_PROVENANCE_HEADER_MAP.get(str(name).strip().lower())
        if not canonical:
            continue
        if _looks_secret_value(value):
            continue
        prov[canonical] = value.strip() if isinstance(value, str) else value
    return prov


def _cancellable_speech_request(
    *,
    url: str,
    body: bytes,
    headers: dict[str, str],
    timeout: float,
    cancel_event: threading.Event,
    on_request_dispatched: Callable[[], None] | None = None,
) -> tuple[bytes, str, dict[str, str]]:
    """POST speech bytes with a transport that terminal cancellation can close.

    Returns ``(audio_bytes, response_mime, response_headers)``; the lowercased
    response header map lets the caller retain non-secret HLI provenance.
    ``on_request_dispatched`` fires exactly once after the full HTTP request has
    been accepted by the connected socket, before any response read."""
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise VoiceUnavailable(f"HiveMind /v1/audio/speech has invalid URL: {url}")
    deadline = time.monotonic() + max(0.05, timeout)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    request_path = parsed.path or "/"
    if parsed.query:
        request_path = f"{request_path}?{parsed.query}"
    wire_headers = dict(headers)
    default_port = 443 if parsed.scheme == "https" else 80
    wire_headers.setdefault(
        "Host",
        parsed.hostname if port == default_port else f"{parsed.hostname}:{port}",
    )
    wire_headers["Content-Length"] = str(len(body))
    wire_headers["Connection"] = "close"
    request_head = (
        f"POST {request_path} HTTP/1.1\r\n"
        + "".join(f"{name}: {value}\r\n" for name, value in wire_headers.items())
        + "\r\n"
    ).encode("iso-8859-1")
    transport: socket.socket | ssl.SSLSocket | None = None
    try:
        if cancel_event.is_set():
            raise VoiceUnavailable("HiveMind /v1/audio/speech cancelled before dispatch")
        addresses = _resolve_addresses_cancellable(
            parsed.hostname,
            port,
            deadline=deadline,
            cancel_event=cancel_event,
        )
        connect_error: OSError | None = None
        for family, socktype, proto, _canonname, sockaddr in addresses:
            candidate = socket.socket(family, socktype, proto)
            candidate.settimeout(min(0.2, max(0.05, deadline - time.monotonic())))
            try:
                candidate.connect(sockaddr)
                transport = candidate
                break
            except OSError as exc:
                connect_error = exc
                candidate.close()
        if transport is None:
            raise connect_error or OSError("no addresses resolved for speech transport")
        transport.settimeout(0.05)
        if parsed.scheme == "https":
            context = ssl.create_default_context()
            transport = context.wrap_socket(
                transport,
                server_hostname=parsed.hostname,
                do_handshake_on_connect=False,
            )
            while True:
                if cancel_event.is_set():
                    raise VoiceUnavailable("HiveMind /v1/audio/speech cancelled")
                if time.monotonic() >= deadline:
                    raise TimeoutError("TLS handshake timed out")
                try:
                    transport.do_handshake()
                    break
                except (ssl.SSLWantReadError, ssl.SSLWantWriteError, socket.timeout):
                    continue

        outbound = memoryview(request_head + body)
        while outbound:
            if cancel_event.is_set():
                raise VoiceUnavailable("HiveMind /v1/audio/speech cancelled")
            if time.monotonic() >= deadline:
                raise TimeoutError("request send timed out")
            try:
                sent = transport.send(outbound)
            except socket.timeout:
                continue
            if sent <= 0:
                raise OSError("speech transport closed while sending request")
            outbound = outbound[sent:]
        if on_request_dispatched is not None:
            try:
                on_request_dispatched()
            except Exception as exc:
                cancel_event.set()
                raise VoiceUnavailable(
                    f"HiveMind /v1/audio/speech dispatch callback failed: {exc}"
                ) from exc

        response_bytes = bytearray()
        header_end = -1
        status = 0
        response_headers: dict[str, str] = {}
        content_length: int | None = None
        chunked = False
        while True:
            if cancel_event.is_set():
                raise VoiceUnavailable("HiveMind /v1/audio/speech cancelled")
            if time.monotonic() >= deadline:
                raise TimeoutError("response read timed out")
            try:
                chunk = transport.recv(64 * 1024)
            except socket.timeout:
                continue
            if not chunk:
                break
            response_bytes.extend(chunk)
            if header_end < 0:
                header_end = response_bytes.find(b"\r\n\r\n")
                if header_end >= 0:
                    head = bytes(response_bytes[:header_end]).decode("iso-8859-1")
                    lines = head.split("\r\n")
                    status_parts = lines[0].split(" ", 2)
                    if len(status_parts) < 2 or not status_parts[1].isdigit():
                        raise VoiceUnavailable(
                            "HiveMind /v1/audio/speech returned invalid HTTP status"
                        )
                    status = int(status_parts[1])
                    for line in lines[1:]:
                        name, separator, value = line.partition(":")
                        if separator:
                            response_headers[name.strip().lower()] = value.strip()
                    if response_headers.get("content-length", "").isdigit():
                        content_length = int(response_headers["content-length"])
                    chunked = "chunked" in response_headers.get("transfer-encoding", "").lower()
            if header_end >= 0:
                received_body = bytes(response_bytes[header_end + 4:])
                if content_length is not None and len(received_body) >= content_length:
                    break
                if chunked and _decode_chunked_http_body(received_body) is not None:
                    break

        if header_end < 0:
            raise VoiceUnavailable("HiveMind /v1/audio/speech returned no HTTP headers")
        encoded_body = bytes(response_bytes[header_end + 4:])
        if chunked:
            decoded = _decode_chunked_http_body(encoded_body)
            if decoded is None:
                raise VoiceUnavailable("HiveMind /v1/audio/speech returned incomplete chunks")
            audio_bytes = decoded
        elif content_length is not None:
            if len(encoded_body) < content_length:
                raise VoiceUnavailable("HiveMind /v1/audio/speech returned a truncated body")
            audio_bytes = encoded_body[:content_length]
        else:
            audio_bytes = encoded_body
        response_mime = response_headers.get("content-type", "audio/wav")
        if status >= 400:
            detail = audio_bytes.decode("utf-8", errors="replace")[:400]
            raise VoiceUnavailable(
                f"HiveMind /v1/audio/speech returned {status}: {detail}"
            )
        return audio_bytes, response_mime, dict(response_headers)
    except VoiceUnavailable:
        raise
    except (OSError, TimeoutError, ssl.SSLError) as exc:
        if cancel_event.is_set():
            raise VoiceUnavailable("HiveMind /v1/audio/speech cancelled") from exc
        raise VoiceUnavailable(f"HiveMind /v1/audio/speech unreachable: {exc}") from exc
    finally:
        if transport is not None:
            try:
                transport.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                transport.close()
            except OSError:
                pass


def synthesize(
    *,
    hivemind_url: str,
    text: str,
    model: str | None = None,
    voice: str | None = None,
    response_format: str | None = None,
    timeout: float = 60,
    cancel_event: threading.Event | None = None,
    on_request_dispatched: Callable[[], None] | None = None,
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
    location_policy = _tts_location_policy()
    if location_policy:
        headers["X-HiveMind-Location"] = location_policy
    headers.update(hivemind_auth_headers())
    speech_url = f"{hivemind_url.rstrip('/')}/v1/audio/speech"
    request = urllib.request.Request(
        speech_url,
        data=body,
        headers=headers,
        method="POST",
    )
    response_headers: Any
    if cancel_event is not None or on_request_dispatched is not None:
        request_cancel = cancel_event or threading.Event()
        audio_bytes, response_mime, response_headers = _cancellable_speech_request(
            url=speech_url,
            body=body,
            headers=headers,
            timeout=timeout,
            cancel_event=request_cancel,
            on_request_dispatched=on_request_dispatched,
        )
    else:
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                audio_bytes = response.read()
                response_mime = response.headers.get("Content-Type", "audio/wav")
                response_headers = _extract_response_headers(response.headers)
        except urllib.error.HTTPError as exc:
            raise VoiceUnavailable(
                f"HiveMind /v1/audio/speech returned {exc.code}: "
                f"{exc.read().decode('utf-8', errors='replace')[:400]}"
            ) from exc
        except (urllib.error.URLError, OSError) as exc:
            raise VoiceUnavailable(f"HiveMind /v1/audio/speech unreachable: {exc}") from exc
    if not audio_bytes:
        raise VoiceUnavailable("HiveMind /v1/audio/speech returned empty body")
    result: dict[str, Any] = {
        "audio_bytes": audio_bytes,
        "content_type": response_mime,
        "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
        "model": model or DEFAULT_TTS_MODEL,
        "voice": voice or DEFAULT_TTS_VOICE,
        "format": response_format or DEFAULT_TTS_FORMAT,
    }
    # Retain non-secret HLI provenance response headers (fail closed on secrets).
    provenance = _provenance_from_response_headers(response_headers)
    if provenance:
        result["provenance"] = provenance
    return result


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
    client_id: str | None = None,
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
    transcript, raw_transcript = _ptt_transcript_from_asr(asr_result)
    if not transcript:
        raise VoiceRequestError(
            "transcription returned empty text; no chat turn dispatched"
        )
    _chat_kw: dict[str, Any] = {"session_id": session_id, "model": model}
    if depth_model:
        _chat_kw["depth_model"] = depth_model
    if _voice_brevity_enabled():
        _chat_kw["voice_mode"] = True
    if client_id:
        _chat_kw["client_id"] = client_id
    chat_response = runner.chat(transcript, **_chat_kw)
    reply_text = (chat_response.get("text") or "").strip()
    if (
        chat_response.get("completed") is False
        or chat_response.get("cancelled") is True
        or not reply_text
    ):
        metrics = chat_response.get("metrics") or {}
        incomplete_reason = str(
            metrics.get("incomplete_reason")
            or "voice Face Lobe returned no complete semantic answer"
        )
        raise VoiceUnavailable(f"voice turn incomplete: {incomplete_reason}")
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
        raw_transcript=raw_transcript,
        grounding_source=chat_response.get("grounding_source"),
        router=chat_response.get("router"),
        dispatched_job=chat_response.get("dispatched_job"),
    )


# ---------------------------------------------------------------------------
# Streaming PTT turn (sentence-chunked TTS for ChatGPT-mobile parity)
# ---------------------------------------------------------------------------


_SENTENCE_END_RE = re.compile(r"([.!?])(\s+|$)")
_NEWLINE_BREAK_RE = re.compile(r"\n\n+")
# First-chunk acceleration accepts punctuation at the current buffer end.
_FIRST_CHUNK_BREAK_RE = re.compile(r"[,;:—–-]\s+|[,;:]$")
# Steady-state clause boundaries must be complete (followed by whitespace).
# Colons stay attached to continuations such as URLs and structured key/value
# speech; commas/semicolons/dashes still provide natural tool-result breaks.
_STEADY_CHUNK_BREAK_RE = re.compile(r"[,;—–-]\s+")


# ---------------------------------------------------------------------------
# Audio duration helpers (shared by the reflex QA and the runtime gate)
# ---------------------------------------------------------------------------

_AUDIO_NORM_RE = re.compile(r"[^a-z0-9' ]+")


def wav_duration_secs(audio_bytes: bytes) -> float | None:
    """Exact duration of WAV bytes, or None if not parseable. A babbling
    TTS tail makes a clip far longer than the text warrants - the single
    most reliable, model-free garble signal.

    Streaming WAV producers may leave ``0x7fffffff`` sentinel sizes in the
    RIFF and data headers even though the HTTP body is already complete. The
    declared frame count is therefore not authoritative; derive duration from
    the PCM bytes actually present in this bounded response.
    """
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as w:
            rate = w.getframerate()
            channels = w.getnchannels()
            sample_width = w.getsampwidth()
            if rate <= 0 or channels <= 0 or sample_width <= 0:
                return None
            pcm = w.readframes(w.getnframes())
            frame_width = channels * sample_width
            if not pcm or len(pcm) % frame_width:
                return None
            return (len(pcm) // frame_width) / float(rate)
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


_RUNTIME_AUDIO_GATE_META_KEY = "_ms4_runtime_audio_gate"


def _runtime_audio_gate_measurement(
    audio_bytes: bytes,
    content_type: str,
    text: str,
) -> dict[str, float | bool] | None:
    """Measure the duration gate without changing the supplied audio.

    Millisecond values retain three decimal places so even a one-frame WAV
    overrun remains visible in evidence.  ``None`` means the duration gate is
    not applicable (non-WAV or an unparseable WAV), not that the audio passed.
    """

    if "wav" not in (content_type or "").lower():
        return None
    duration_s = wav_duration_secs(audio_bytes)
    if duration_s is None:
        return None
    limit_s = expected_speech_secs(text) + 1.0
    excess_s = max(0.0, duration_s - limit_s)
    return {
        "original_duration_ms": round(duration_s * 1000.0, 3),
        "limit_ms": round(limit_s * 1000.0, 3),
        "excess_ms": round(excess_s * 1000.0, 3),
        "would_gate": duration_s > limit_s,
    }


def _gate_chunk_audio(audio_bytes: bytes, content_type: str, text: str) -> tuple[bytes, bool]:
    """If a live chunk's WAV has a clear babble tail (well beyond what the
    text warrants), trim it. Conservative headroom so legit slightly-long
    speech is never cut. Returns (audio_bytes, trimmed)."""
    measurement = _runtime_audio_gate_measurement(audio_bytes, content_type, text)
    if measurement is None or not measurement["would_gate"]:
        return audio_bytes, False
    trimmed = trim_wav(audio_bytes, float(measurement["limit_ms"]) / 1000.0)
    if trimmed:
        return trimmed, True
    return audio_bytes, False


def _call_tts_with_runtime_audio_gate_recovery(
    call: Callable[..., dict[str, Any]],
    kwargs: dict[str, Any],
    *,
    turn_cancel: threading.Event,
    client_alive: threading.Event | None = None,
    on_busy_retry: Callable[[], None] | None = None,
    on_gate_retry: Callable[[], None] | None = None,
    on_gate_recovery: Callable[[], None] | None = None,
    on_gate_retry_failure: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Run TTS and regenerate once before admitting duration-gated audio.

    A duration overrun is a useful fail-closed signal, but physically trimming
    the first render can also remove legitimate tail words.  Make one fresh
    same-route synthesis attempt while the call still owns its executor slot.
    A clean retry replaces the first render.  A retry exception or a second
    gated render retains the best original candidate for the existing trim +
    degraded-client path; it never weakens that terminal contract.
    """

    def _call_once() -> dict[str, Any]:
        return _call_tts_with_busy_retry(
            call,
            kwargs,
            turn_cancel=turn_cancel,
            client_alive=client_alive,
            on_retry=on_busy_retry,
        )

    def _require_live(stage: str) -> None:
        if turn_cancel.is_set() or (
            client_alive is not None and not client_alive.is_set()
        ):
            raise VoiceUnavailable(f"REST TTS cancelled {stage}")

    primary = _call_once()
    _require_live("after initial runtime-gate synthesis")
    if not _runtime_audio_gate_enabled() or not isinstance(primary, dict):
        return primary
    # Idempotence fence: a caller may already expose the same bounded helper.
    # Its metadata proves the one recovery decision has been made, so an outer
    # adapter must not turn one permitted regeneration into an unbounded chain.
    if _RUNTIME_AUDIO_GATE_META_KEY in primary:
        return primary
    primary_measurement = _runtime_audio_gate_measurement(
        primary.get("audio_bytes") or b"",
        primary.get("content_type") or "",
        str(kwargs.get("text") or ""),
    )
    if primary_measurement is None or not primary_measurement["would_gate"]:
        return primary

    metadata: dict[str, Any] = {
        "original_duration_ms": primary_measurement["original_duration_ms"],
        "limit_ms": primary_measurement["limit_ms"],
        "excess_ms": primary_measurement["excess_ms"],
        "retry_count": 1,
        "retry_outcome": "pending",
        "recovered": False,
        "selected_attempt": "initial",
    }
    _require_live("before runtime-gate recovery")
    if on_gate_retry is not None:
        on_gate_retry()

    try:
        retry = _call_once()
        _require_live("after runtime-gate recovery")
    except Exception:
        # Cancellation/liveness always wins; never emit the retained first render
        # after its owner has gone away.  An ordinary retry fault preserves the
        # original bounded fail-closed audio instead of converting it to silence.
        if turn_cancel.is_set() or (client_alive is not None and not client_alive.is_set()):
            raise
        metadata["retry_outcome"] = "exception"
        if on_gate_retry_failure is not None:
            on_gate_retry_failure()
        decorated = dict(primary)
        decorated[_RUNTIME_AUDIO_GATE_META_KEY] = metadata
        return decorated

    retry_measurement = None
    if isinstance(retry, dict):
        retry_measurement = _runtime_audio_gate_measurement(
            retry.get("audio_bytes") or b"",
            retry.get("content_type") or "",
            str(kwargs.get("text") or ""),
        )
    if retry_measurement is not None:
        metadata.update({
            "retry_original_duration_ms": retry_measurement["original_duration_ms"],
            "retry_limit_ms": retry_measurement["limit_ms"],
            "retry_excess_ms": retry_measurement["excess_ms"],
        })

    if (
        isinstance(retry, dict)
        and retry.get("audio_bytes")
        and retry_measurement is not None
        and not retry_measurement["would_gate"]
    ):
        metadata.update({
            "retry_outcome": "clean",
            "recovered": True,
            "selected_attempt": "retry",
        })
        if on_gate_recovery is not None:
            on_gate_recovery()
        decorated = dict(retry)
        decorated[_RUNTIME_AUDIO_GATE_META_KEY] = metadata
        return decorated

    metadata["retry_outcome"] = (
        "still_gated"
        if retry_measurement is not None and retry_measurement["would_gate"]
        else "invalid_audio"
    )
    if on_gate_retry_failure is not None:
        on_gate_retry_failure()

    # If both renders gate, keep the one with the smaller measured excess.  It
    # will still be trimmed and reported as degraded, but discards fewer bytes.
    selected = primary
    if (
        isinstance(retry, dict)
        and retry_measurement is not None
        and retry_measurement["would_gate"]
        and float(retry_measurement["excess_ms"])
        < float(primary_measurement["excess_ms"])
    ):
        selected = retry
        metadata["selected_attempt"] = "retry"
    decorated = dict(selected)
    decorated[_RUNTIME_AUDIO_GATE_META_KEY] = metadata
    return decorated


def _emit_text_as_parallel_chunks(
    *, text: str, emit: Callable[[str, dict[str, Any]], bool], hivemind_url: str,
    tts_model: str | None, tts_voice: str | None, response_format: str | None,
    pool_size: int = TTS_POOL_SIZE, start_index: int = 0,
    cancel_event: threading.Event | None = None,
    client_alive: threading.Event | None = None,
    lifecycle: dict[str, int] | None = None,
) -> int:
    """Sentence-chunk ``text`` and emit the first chunk before tail fanout.

    The first chunk is synthesized alone. Only after it has been offered to
    the client are the remaining chunks submitted in parallel and emitted in
    order. This keeps first-audio latency bounded when HiveMind routes every
    request to one serialized worker while retaining tail concurrency when
    multiple workers are available. Returns the count accepted by the client.
    """
    fallback_words = len((text or "").split())
    chunker = SentenceChunker(
        first_chunk_min_words=(
            MIN_CHUNK_WORDS if fallback_words <= MIN_CHUNK_WORDS else FIRST_CHUNK_MIN_WORDS
        )
    )
    sentences = list(chunker.add(text or ""))
    tail = chunker.flush()
    if tail:
        sentences.append(tail)
    from .spoken_text_filter import sanitize_for_speech as _san
    sanitized = [(_san(s).strip() if TTS_FILTER_ENABLED else s) for s in sentences]
    pieces = [
        fragment
        for piece in sanitized
        if piece
        for fragment in _split_xtts_safe_fragments(piece)
    ]
    if not pieces:
        return 0
    emitted = 0
    consumer_open = True
    turn_cancel = cancel_event or threading.Event()
    lifecycle = lifecycle if lifecycle is not None else {}
    lifecycle.setdefault("rest_fallback_tasks_submitted", 0)
    lifecycle.setdefault("rest_fallback_tasks_completed", 0)
    lifecycle.setdefault("rest_fallback_workers_started", 0)
    lifecycle.setdefault("rest_fallback_workers_joined", 0)
    lifecycle.setdefault("rest_fallback_worker_join_timeouts", 0)
    lifecycle.setdefault("rest_fallback_busy_retries", 0)
    lifecycle.setdefault("rest_fallback_runtime_gate_retries", 0)
    lifecycle.setdefault("rest_fallback_runtime_gate_recoveries", 0)
    lifecycle.setdefault("rest_fallback_runtime_gate_retry_failures", 0)
    lifecycle_lock = threading.Lock()
    effective_pool_size = max(1, min(pool_size, _voice_rest_capacity()))

    def _increment_lifecycle(name: str) -> None:
        with lifecycle_lock:
            lifecycle[name] += 1

    def _emit_result(i: int, res: dict[str, Any]) -> tuple[bool, bool]:
        def _fail_missing_audio(reason: str) -> tuple[bool, bool]:
            emit("audio_error", {
                "index": start_index + i,
                "error": reason,
            })
            turn_cancel.set()
            return False, False

        audio_bytes = res.get("audio_bytes") or b""
        if not audio_bytes:
            return _fail_missing_audio("TTS returned no audio bytes for this chunk")
        mime = res.get("content_type") or "audio/wav"
        runtime_gate_telemetry = res.get(_RUNTIME_AUDIO_GATE_META_KEY)
        if isinstance(runtime_gate_telemetry, dict):
            runtime_gate_telemetry = dict(runtime_gate_telemetry)
        else:
            runtime_gate_telemetry = None
        selected_original_duration_s = wav_duration_secs(audio_bytes)
        if _runtime_audio_gate_enabled():
            audio_bytes, trimmed = _gate_chunk_audio(audio_bytes, mime, pieces[i])
        else:
            trimmed = False
        if not audio_bytes:
            return _fail_missing_audio("TTS audio failed the runtime audio gate")
        if runtime_gate_telemetry is not None:
            emitted_duration_s = wav_duration_secs(audio_bytes)
            runtime_gate_telemetry.update({
                "selected_original_duration_ms": (
                    round(selected_original_duration_s * 1000.0, 3)
                    if selected_original_duration_s is not None
                    else None
                ),
                "emitted_duration_ms": (
                    round(emitted_duration_s * 1000.0, 3)
                    if emitted_duration_s is not None
                    else None
                ),
                "trim_delta_ms": (
                    round(max(
                        0.0,
                        selected_original_duration_s - emitted_duration_s,
                    ) * 1000.0, 3)
                    if (
                        selected_original_duration_s is not None
                        and emitted_duration_s is not None
                    )
                    else None
                ),
                "final_runtime_gated": trimmed,
            })
        payload = {
            "index": start_index + i,
            "text": pieces[i],
            "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
            "audio_mime": mime,
            "tts_engine": "rest_fallback",
            "runtime_gated": trimmed,
        }
        if runtime_gate_telemetry is not None:
            payload["runtime_gate"] = runtime_gate_telemetry
        client_written = emit("audio_chunk", payload)
        if client_written is False:
            turn_cancel.set()
        return client_written is not False, client_written is not False

    # Prime playback before scheduling the tail. A single serialized TTS
    # worker can otherwise finish the logical first sentence last.
    def _synthesize_piece(piece: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "hivemind_url": hivemind_url,
            "text": piece,
            "model": tts_model,
            "voice": tts_voice,
            "response_format": response_format,
        }
        if _accepts_keyword(synthesize, "cancel_event"):
            kwargs["cancel_event"] = turn_cancel
        return _call_tts_with_runtime_audio_gate_recovery(
            synthesize,
            kwargs,
            turn_cancel=turn_cancel,
            client_alive=client_alive,
            on_busy_retry=lambda: _increment_lifecycle(
                "rest_fallback_busy_retries"
            ),
            on_gate_retry=lambda: _increment_lifecycle(
                "rest_fallback_runtime_gate_retries"
            ),
            on_gate_recovery=lambda: _increment_lifecycle(
                "rest_fallback_runtime_gate_recoveries"
            ),
            on_gate_retry_failure=lambda: _increment_lifecycle(
                "rest_fallback_runtime_gate_retry_failures"
            ),
        )

    try:
        if turn_cancel.is_set() or (client_alive is not None and not client_alive.is_set()):
            return 0
        lifecycle["rest_fallback_tasks_submitted"] += 1
        first_result = _synthesize_piece(pieces[0])
        lifecycle["rest_fallback_tasks_completed"] += 1
        consumer_open, first_emitted = _emit_result(0, first_result)
        emitted += int(first_emitted)
    except Exception as exc:  # noqa: BLE001
        consumer_open = emit(
            "audio_error",
            {"index": start_index, "error": str(exc)[:200]},
        ) is not False
    if not consumer_open or len(pieces) == 1:
        return emitted

    ex = ThreadPoolExecutor(
        max_workers=max(1, min(effective_pool_size, len(pieces) - 1)),
        thread_name_prefix="ms4-tts-fb",
    )
    futures: list[tuple[int, Future]] = []
    completed_normally = False
    try:
        futures = [
            (
                i,
                ex.submit(
                    _synthesize_piece,
                    pieces[i],
                ),
            )
            for i in range(1, len(pieces))
        ]
        lifecycle["rest_fallback_tasks_submitted"] += len(futures)
        final_timeout = _voice_rest_final_timeout(
            piece_count=len(futures),
            capacity=effective_pool_size,
        )
        lifecycle["rest_fallback_final_timeout_s"] = final_timeout
        lifecycle["rest_fallback_piece_count"] = len(pieces)
        lifecycle["rest_fallback_capacity"] = effective_pool_size
        deadline = time.monotonic() + final_timeout
        for position, (i, fut) in enumerate(futures):
            while not fut.done():
                if (
                    turn_cancel.is_set()
                    or (client_alive is not None and not client_alive.is_set())
                    or time.monotonic() >= deadline
                ):
                    consumer_open = False
                    turn_cancel.set()
                    break
                time.sleep(0.01)
            if not consumer_open:
                for _, pending in futures[position:]:
                    pending.cancel()
                break
            try:
                consumer_open, did_emit = _emit_result(i, fut.result())
                lifecycle["rest_fallback_tasks_completed"] += 1
                emitted += int(did_emit)
            except Exception as exc:  # noqa: BLE001
                consumer_open = emit(
                    "audio_error",
                    {"index": start_index + i, "error": str(exc)[:200]},
                ) is not False
            if not consumer_open:
                for _, pending in futures[position + 1:]:
                    pending.cancel()
                break
        completed_normally = consumer_open and all(future.done() for _, future in futures)
    finally:
        if completed_normally:
            ex.shutdown(wait=True)
            workers = list(getattr(ex, "_threads", ()))
            lifecycle["rest_fallback_workers_started"] += len(workers)
            lifecycle["rest_fallback_workers_joined"] += len(workers)
        else:
            cleanup = _shutdown_executor_bounded(
                ex,
                [future for _, future in futures],
                cancel_event=turn_cancel,
                label="REST fallback",
            )
            lifecycle["rest_fallback_workers_started"] += cleanup["workers_started"]
            lifecycle["rest_fallback_workers_joined"] += cleanup["workers_joined"]
            lifecycle["rest_fallback_worker_join_timeouts"] += cleanup["worker_join_timeouts"]
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


# ---------------------------------------------------------------------------
# Marker-confirmation contract (2026-07-06)
#
# A physical push-to-talk turn said: "...confirm you heard the Cobalt 216
# marker through the physical microphone." ASR + the outbound chat request
# preserved that exact text, but ordinary llama3.1:8b FaceLobe returned a
# generic no-context reply and ``_on_delta`` streamed it into
# sentence-chunking / TTS with NO semantic postcondition (see
# FACE_LOBE_CONTEXT_OWNER_DIAGNOSIS.md). This is a deterministic, voice-level
# postcondition for EXPLICIT marker-confirmation requests ONLY — NOT a broad
# LLM judge and NOT a Cobalt-specific hack. Ordinary voice turns produce no
# obligation (extract returns None), so their streaming / first-audio behavior
# is completely unchanged; only an explicit "confirm you heard the <marker>
# marker" turn buffers deltas and enforces the obligation before any
# user-visible text or TTS scheduling.
# ---------------------------------------------------------------------------

# Parser for the POSITIVE marker-confirmation relation: "confirm (that) you
# heard/received the <marker> marker". Deliberately tight so it rejects the
# adversarial near-misses WITHOUT a Cobalt-specific blocklist:
#   * the connective between "confirm" and the hearing verb allows ONLY an
#     optional "that", an optional subject pronoun, and optional AFFIRMATIVE
#     auxiliaries — so "confirm the meeting, and I heard ..." cannot bridge the
#     object NP / clause boundary to the hearing clause, and "confirm you did
#     not hear ..." cannot match (the "not"/"n't" is not an allowed connective
#     token, breaking the affirmative relation);
#   * a negation governing the "confirm" imperative itself ("do not / don't /
#     never confirm ...") is rejected by a clause-local negation-cue check in
#     _extract_marker_obligation.
_MARKER_CONFIRM_RE = re.compile(
    r"\bconfirm\b\s+"
    r"(?:that\s+)?(?:you\s+|u\s+)?"
    r"(?:have\s+|had\s+|did\s+|actually\s+|really\s+|indeed\s+|clearly\s+|just\s+|already\s+)*"
    r"\b(?:heard|hear|hearing|received|receive|receiving|got|detected|detect|detecting)\b"
    r"\s+(?:the|that|my|a|an)\s+(?P<marker>[A-Za-z0-9][A-Za-z0-9 '\-]*?)\s+marker\b",
    re.IGNORECASE,
)

# Negation / prohibition cues that, when they appear ANYWHERE in the clause
# that governs the matched "confirm" imperative, flip the request away from a
# positive "do confirm you heard it". Exact tokens (not prefixes) so innocuous
# words like "banana" or "important" never trip a cue.
_MARKER_NEG_CUES = frozenset({
    # negations
    "not", "no", "none", "never", "cannot", "without", "neither", "nor",
    "dont", "cant", "wont", "didnt", "doesnt", "isnt", "wasnt", "werent",
    "havent", "hasnt", "hadnt", "wouldnt", "couldnt", "shouldnt", "mustnt",
    # prohibition / refusal verbs (+ common inflections)
    "avoid", "avoids", "avoided",
    "refrain", "refrains", "refrained",
    "refuse", "refuses", "refused",
    "forbid", "forbids", "forbidden",
    "prohibit", "prohibits", "prohibited",
    "ban", "bans", "banned",
    "disallow", "disallows", "disallowed",
    "prevent", "prevents", "prevented",
})
_MARKER_CLAUSE_SPLIT_RE = re.compile(r"[.?!;:\n]")
_MARKER_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


def _normalize_marker(text: str) -> str:
    """Lowercase, strip punctuation (keep ``a-z0-9' ``), and collapse
    whitespace — the same normalization style the audio helpers use
    (:data:`_AUDIO_NORM_RE`). Used ONLY for marker comparison, never to alter
    user-visible or spoken text."""
    return " ".join(_AUDIO_NORM_RE.sub(" ", (text or "").lower()).split())


def _extract_marker_obligation(transcript: str) -> str | None:
    """Return the requested marker phrase (ORIGINAL casing) when ``transcript``
    is an explicit POSITIVE marker-confirmation request — i.e. it affirmatively
    asks to confirm that the marker WAS heard/received ("confirm (that) you
    heard/received the <marker> marker"). Returns ``None`` otherwise.

    Rejects (returns ``None``):
      * a negation OR prohibition anywhere in the clause that governs the
        confirm imperative — including DISTAL cues ("do not ever under any
        circumstances confirm ...", "under no circumstances should you
        confirm ...") and prohibition verbs ("I forbid you to confirm ...").
        A cue in a PRIOR clause (before ';' or a sentence boundary) is out of
        scope and does not reject a later positive confirm;
      * a negation attached to the hear/receive clause
        ("confirm you did not / didn't hear ...") — the tight affirmative
        connective cannot bridge a "not"/"n't";
      * a "confirm" that governs a DIFFERENT object with the marker only in a
        separate declarative clause ("Confirm the meeting, and I heard the
        <marker> marker.") — the tight connective cannot bridge the object NP
        or a clause boundary.

    General over marker names (Cobalt 216, Alpha Seven, codeword Delta), not a
    string blocklist: rejection is a structural relation + negation-cue check.
    Conservative — a normal voice turn returns ``None`` and is never buffered or
    rewritten."""
    if not transcript:
        return None
    match = _MARKER_CONFIRM_RE.search(transcript)
    if match is None:
        return None
    # Reject when a negation OR prohibition governs THIS confirm imperative.
    # Isolate the clause that holds the matched relation by splitting the
    # pre-confirm text on clause boundaries (';', sentence terminators, ':',
    # newline) and taking the last segment — the span from the start of that
    # clause up to "confirm". Then scan the WHOLE span (not a fixed last-N
    # window) so a DISTAL cue still scopes the imperative ("do not ever under
    # any circumstances confirm ...", "under no circumstances should you
    # confirm ...", "I forbid you to confirm ..."). A cue in a PRIOR clause
    # (e.g. before a ';') is out of scope, so a positive confirm after the
    # boundary survives. Conservative: any in-clause negation/prohibition
    # returns None (ordinary model path) rather than risk a false "I heard the
    # marker" confirmation.
    clause = _MARKER_CLAUSE_SPLIT_RE.split(transcript[: match.start()])[-1]
    clause_tokens = _MARKER_TOKEN_RE.findall(clause.lower())
    if any(tok in _MARKER_NEG_CUES or tok.endswith("n't") for tok in clause_tokens):
        return None
    marker = " ".join(match.group("marker").split()).strip(" '-")
    if not marker or not _normalize_marker(marker):
        return None
    return marker


def _marker_reply_satisfies(reply: str, marker: str) -> bool:
    """True when ``reply`` contains the whole normalized ``marker`` phrase as a
    contiguous whitespace-delimited token run, so ``cobalt 216`` matches
    ``I heard the Cobalt 216 marker`` but NOT ``cobalt 2160``."""
    marker_norm = _normalize_marker(marker)
    if not marker_norm:
        return False
    return f" {marker_norm} " in f" {_normalize_marker(reply)} "


def _marker_confirmation_text(marker: str) -> str:
    """The single short deterministic confirmation emitted when the model reply
    fails the obligation, grounded in the requested marker phrase (original
    casing preserved). Satisfies its own contract by construction."""
    clean = " ".join((marker or "").split()).strip(" '-")
    return f"I heard the {clean} marker."


class _MarkerConfirmationGate:
    """Per-turn voice-level postcondition for explicit marker-confirmation
    requests.

    When ``active`` (the ASR transcript carries an explicit marker obligation),
    the caller BUFFERS model deltas via :meth:`buffer_delta` instead of
    streaming them, so an invalid reply never reaches ``text_delta`` /
    ``chunk_scheduled`` / REST or WS TTS synthesis. After chat completes,
    :meth:`resolve` returns the text to release: the model reply UNCHANGED when
    it already confirms the marker, else ONE short deterministic confirmation
    grounded in the request. When inactive (every ordinary turn) the caller
    streams deltas immediately and this gate does nothing — preserving the
    current streaming / first-audio behavior for normal turns.
    """

    def __init__(self, transcript: str) -> None:
        self.marker = _extract_marker_obligation(transcript)
        self.active = self.marker is not None
        self.correction_required = False
        self.resolved = False
        self.released_text = ""
        self._buffer: list[str] = []

    def buffer_delta(self, delta: str) -> None:
        if delta:
            self._buffer.append(delta)

    @property
    def buffered_text(self) -> str:
        return "".join(self._buffer)

    def resolve(self, model_reply: str) -> str:
        """Decide the text to release for an active contract. Idempotent: the
        first call fixes ``released_text`` / ``correction_required``."""
        if self.resolved:
            return self.released_text
        reply = (model_reply or "").strip() or self.buffered_text.strip()
        if self.marker and _marker_reply_satisfies(reply, self.marker):
            self.correction_required = False
            self.released_text = reply
        else:
            self.correction_required = True
            self.released_text = _marker_confirmation_text(self.marker or "")
        self.resolved = True
        return self.released_text

    def result(self) -> dict[str, Any]:
        """Truthful metric: did the contract apply, and was a deterministic
        correction required?"""
        return {
            "applied": self.active,
            "marker": self.marker,
            "correction_required": self.correction_required,
        }


class SentenceChunker:
    """Consume streaming text deltas and emit chunks at speech boundaries.

    Strategy:

      1. **Asymmetric startup-pair acceleration.** The first chunk uses the
         configured ``first_chunk_min_words`` threshold for fast synthesis.
         The second uses ``min_chunk_words`` as a bounded runway fragment.
         This keeps first audio responsive without repeating two tiny TTS calls
         whose combined speech can end before the next parallel result arrives.
      2. **Later chunks coalesce short sentences/lines** up to
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
        # Coalescing target for chunks after the startup pair; clamp to >=1 and
        # never above the hard ceiling.
        self.min_chunk_words = max(1, min(min_chunk_words, max_chunk_words))
        self.buffer = ""
        self.chunk_count = 0
        self.first_chunk_sanitizer: Callable[[str], str] | None = None
        self._first_chunk_weak_comma_balanced = False

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

        The startup pair is deliberately asymmetric: a small first fragment for
        onset, then one normal-sized fragment for playback runway. Later chunks
        coalesce short sentences/lines up to ``min_chunk_words`` so a chatty or
        list-heavy reply doesn't fire a flurry of tiny TTS calls.
        """
        if not self.buffer:
            return None
        chunk: str | None
        if self.chunk_count > 0 and self.min_chunk_words > 1:
            if (
                self.chunk_count == 1
                and self.first_chunk_min_words < self.min_chunk_words
            ):
                chunk = self._extract_coalesced(
                    min_words=self.min_chunk_words,
                    run_on_word_limit=(
                        self.max_chunk_words
                        if self._first_chunk_weak_comma_balanced
                        else self.min_chunk_words
                    ),
                )
            else:
                chunk = self._extract_coalesced()
        else:
            chunk = self._extract_first_chunk()
        if chunk is not None:
            return chunk
        return self._extract_xtts_safe_chunk()

    def _extract_xtts_safe_chunk(self) -> str | None:
        """Last-resort token ceiling after the normal speech boundaries.

        Word and sentence chunking stays authoritative for ordinary prose and
        first-audio latency. This path only fires when structured/no-whitespace
        text would otherwise exceed XTTS's tokenizer ceiling.
        """

        cut = _xtts_safe_cut(self.buffer)
        if cut is None:
            return None
        fragment = self.buffer[:cut]
        self.buffer = self.buffer[cut:]
        return fragment

    def _extract_coalesced(
        self,
        min_words: int | None = None,
        run_on_word_limit: int | None = None,
    ) -> str | None:
        """Emit at the earliest sentence/paragraph boundary whose head
        reaches ``min_chunk_words``, merging shorter sentences/lines. When no
        complete sentence is buffered, a clause boundary may release text after
        the same floor. If no boundary is big enough yet and we're under the
        hard ceiling, wait for more text (return None); the end-of-stream
        ``flush`` will emit any trailing short fragment."""
        target_words = max(1, min(min_words or self.min_chunk_words, self.max_chunk_words))
        run_on_words = max(
            1,
            min(run_on_word_limit or self.max_chunk_words, self.max_chunk_words),
        )
        # The startup runway fragment passes the normal coalescing floor as its
        # ``run_on_word_limit``. Compute that cut before scanning natural
        # boundaries: a complete sentence may already be buffered well beyond
        # the limit, and returning that whole sentence first would silently turn
        # chunk 1 back into a 15-20 word synthesis request.
        limit_cut = self._complete_word_cut(run_on_words)
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
            if len(self.buffer[:head_end].split()) >= target_words:
                # Whichever eligible boundary arrived first wins.  Prefer the
                # natural boundary on an exact tie so its trailing whitespace is
                # consumed; otherwise the explicit run-on/startup ceiling is
                # authoritative.
                selected_head_end = head_end
                selected_consume_end = consume_end
                if limit_cut is not None and limit_cut < head_end:
                    selected_head_end = limit_cut
                    selected_consume_end = limit_cut
                sentence = " ".join(self.buffer[:selected_head_end].split())
                self.buffer = self.buffer[selected_consume_end:].lstrip()
                return sentence if sentence else ""
            # Too short — look past this boundary to coalesce the next.
            search = consume_end
        # With no complete sentence buffered, choose whichever arrives first:
        # an eligible natural clause or the applicable startup/run-on ceiling.
        # This keeps the low-latency second startup chunk and the hard ceiling
        # authoritative even when a later weak delimiter is already buffered.
        weak_cut = None
        for weak in _STEADY_CHUNK_BREAK_RE.finditer(self.buffer):
            candidate_cut = weak.end()
            if len(self.buffer[:candidate_cut].split()) >= target_words:
                weak_cut = candidate_cut
                break
        consume_end = (
            weak_cut
            if weak_cut is not None and (limit_cut is None or weak_cut <= limit_cut)
            else limit_cut
        )
        if consume_end is not None:
            sentence = " ".join(self.buffer[:consume_end].split())
            self.buffer = self.buffer[consume_end:].lstrip()
            return sentence if sentence else ""
        return None

    def _extract_first_chunk(self) -> str | None:
        """Fast path for the first chunk (and per-sentence fallback when
        coalescing is disabled): ship at the first natural boundary."""
        first_cut = None
        if self.chunk_count == 0:
            if self.first_chunk_sanitizer is None:
                first_cut = self._complete_word_cut(self.first_chunk_min_words)
            else:
                for complete_word in re.finditer(r"\S+\s+", self.buffer):
                    candidate = self.buffer[:complete_word.end()].strip()
                    spoken_words = len(self.first_chunk_sanitizer(candidate).split())
                    if (
                        spoken_words >= self.first_chunk_min_words
                        and (
                            spoken_words > self.first_chunk_min_words
                            or _FIRST_CHUNK_BREAK_RE.search(candidate) is None
                        )
                    ):
                        first_cut = complete_word.end()
                        break

        # Pass 1: explicit sentence terminator.
        search = 0
        while True:
            match = _SENTENCE_END_RE.search(self.buffer, search)
            if match is None:
                break
            end = match.end()
            if first_cut is not None and first_cut < end:
                break
            sentence = self.buffer[:end].strip()
            if (
                self.chunk_count > 0
                or self.first_chunk_sanitizer is None
                or len(self.first_chunk_sanitizer(sentence).split())
                >= self.first_chunk_min_words
            ):
                self.buffer = self.buffer[end:]
                return sentence if sentence else ""
            search = end

        # Pass 2: paragraph break.
        search = 0
        while True:
            nm = _NEWLINE_BREAK_RE.search(self.buffer, search)
            if nm is None:
                break
            cut = nm.start()
            if first_cut is not None and first_cut < cut:
                break
            sentence = self.buffer[:cut].strip()
            if (
                self.chunk_count > 0
                or self.first_chunk_sanitizer is None
                or len(self.first_chunk_sanitizer(sentence).split())
                >= self.first_chunk_min_words
            ):
                self.buffer = self.buffer[nm.end():]
                return sentence if sentence else ""
            search = nm.end()

        # Pass 3a: first-chunk acceleration — break on natural inner-
        # punctuation (comma / colon / em-dash / semicolon) so the
        # first chunk ships ASAP. Many model replies start "Looking
        # into that, give me a sec..." or "Yeah, the cluster is
        # idle." — breaking on the first comma cuts ~6-8 words off
        # the first chunk and slashes the in-order hold time the
        # REST engine pays.
        if self.chunk_count == 0:
            search = 0
            while True:
                inner = _FIRST_CHUNK_BREAK_RE.search(self.buffer, search)
                if inner is None:
                    break
                cut = inner.end()
                if first_cut is not None and first_cut < cut:
                    break
                sentence = self.buffer[:cut].strip().rstrip(",;:—-").strip()
                spoken_words = (
                    len(self.first_chunk_sanitizer(sentence).split())
                    if self.first_chunk_sanitizer is not None
                    else len(sentence.split())
                )
                if (
                    sentence
                    and spoken_words >= self.first_chunk_min_words
                    and (
                        self.first_chunk_sanitizer is None
                        or spoken_words > self.first_chunk_min_words
                    )
                ):
                    self.buffer = self.buffer[cut:].lstrip()
                    return sentence
                search = inner.end()

        # Pass 3b: first-chunk min-words fallback. Emit early so audio
        # starts regardless of punctuation.
        if self.chunk_count == 0 and first_cut is not None:
            head_len = first_cut
            sentence = self.buffer[:head_len].strip()
            self._first_chunk_weak_comma_balanced = bool(
                re.search(r",\s+\S", sentence)
            )
            self.buffer = self.buffer[head_len:].lstrip()
            return sentence

        # Pass 4: hard ceiling — don't let a single chunk grow forever.
        max_cut = self._complete_word_cut(self.max_chunk_words)
        if max_cut is not None:
            head_len = max_cut
            sentence = self.buffer[:head_len].strip()
            if (
                self.chunk_count > 0
                or self.first_chunk_sanitizer is None
                or len(self.first_chunk_sanitizer(sentence).split())
                >= self.first_chunk_min_words
            ):
                self.buffer = self.buffer[head_len:].lstrip()
                return sentence

        return None  # not ready

    def _complete_word_cut(self, word_count: int) -> int | None:
        """Return a cut after ``word_count`` whitespace-terminated words.

        Model deltas can end mid-token. Requiring the terminating whitespace
        keeps that fragment buffered until the next delta completes it;
        punctuation paths above still ship complete clauses immediately.
        """
        if word_count <= 0:
            return None
        seen = 0
        for match in re.finditer(r"\S+\s+", self.buffer):
            seen += 1
            if seen == word_count:
                return match.end()
        return None

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

    def flush_first_complete_boundary(self) -> str | None:
        """Deadline flush for the first chunk without consuming a partial token."""
        if self.chunk_count or not self.buffer:
            return None
        candidates: list[int] = []
        candidates.extend(match.end() for match in _SENTENCE_END_RE.finditer(self.buffer))
        candidates.extend(match.end() for match in _FIRST_CHUNK_BREAK_RE.finditer(self.buffer))
        candidates.extend(match.start() for match in _NEWLINE_BREAK_RE.finditer(self.buffer))
        candidates.extend(match.end() for match in re.finditer(r"\S+\s+", self.buffer))
        if re.search(r"[.!?,;:\u2014\u2013-]$", self.buffer):
            candidates.append(len(self.buffer))
        if not candidates:
            return None
        cut = max(candidates)
        text = " ".join(self.buffer[:cut].split()).strip()
        if not text:
            return None
        if self.first_chunk_sanitizer is not None:
            spoken_words = len(self.first_chunk_sanitizer(text).split())
            if (
                spoken_words < self.first_chunk_min_words
                or (
                    spoken_words == self.first_chunk_min_words
                    and _FIRST_CHUNK_BREAK_RE.search(text) is not None
                )
            ):
                return None
        self.buffer = self.buffer[cut:].lstrip()
        self.chunk_count += 1
        return text


class _FirstChunkDeadlineController:
    """Serialize chunker access and fire one complete-boundary deadline flush."""

    def __init__(
        self,
        *,
        chunker: SentenceChunker,
        schedule: Callable[[str], bool],
        active: Callable[[], bool],
        deadline_s: float | None = None,
    ) -> None:
        self.chunker = chunker
        self.schedule = schedule
        self.active = active
        self.deadline_s = _voice_first_chunk_deadline() if deadline_s is None else max(0.0, deadline_s)
        self._lock = threading.RLock()
        self._timer: threading.Timer | None = None
        self.deadline_fired = False
        self.deadline_flushes = 0

    def _start_timer_locked(self) -> None:
        if self.deadline_s <= 0 or self._timer is not None or self.chunker.chunk_count:
            return
        self._timer = threading.Timer(self.deadline_s, self._on_deadline)
        self._timer.name = "ms4-first-chunk-deadline"
        self._timer.daemon = True
        self._timer.start()

    def _cancel_timer_locked(self) -> threading.Timer | None:
        timer = self._timer
        if timer is not None:
            timer.cancel()
        return timer

    def _on_deadline(self) -> None:
        with self._lock:
            self.deadline_fired = True
            if not self.active() or self.chunker.chunk_count:
                return
            chunk = self.chunker.flush_first_complete_boundary()
            if chunk and self.schedule(chunk):
                self.deadline_flushes += 1

    def add(self, delta: str) -> bool:
        with self._lock:
            chunks = self.chunker.add(delta)
            if chunks:
                self._cancel_timer_locked()
            elif delta and delta.strip() and self.chunker.buffer:
                self._start_timer_locked()
            for chunk in chunks:
                if not self.schedule(chunk):
                    return False
            return True

    def flush(self) -> str | None:
        with self._lock:
            self._cancel_timer_locked()
            return self.chunker.flush()

    def close(self) -> bool:
        with self._lock:
            timer = self._cancel_timer_locked()
        if timer is not None and timer is not threading.current_thread():
            timer.join(timeout=max(0.05, min(0.5, self.deadline_s + 0.05)))
        return bool(timer is not None and timer.is_alive())


def _shutdown_executor_bounded(
    executor: ThreadPoolExecutor,
    futures: list[Future],
    *,
    cancel_event: threading.Event,
    label: str,
) -> dict[str, int]:
    """Cancel and boundedly join a turn-owned executor, or fail terminally."""
    cancel_event.set()
    cancelled = 0
    for future in futures:
        cancelled += int(future.cancel())
    executor.shutdown(wait=False, cancel_futures=True)
    threads = list(getattr(executor, "_threads", ()))
    deadline = time.monotonic() + _voice_rest_cleanup_timeout()
    for thread in threads:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        thread.join(timeout=remaining)
    alive = sum(thread.is_alive() for thread in threads)
    metrics = {
        "workers_started": len(threads),
        "workers_joined": len(threads) - alive,
        "worker_join_timeouts": alive,
        "futures_cancelled": cancelled,
    }
    if alive:
        raise VoiceUnavailable(
            f"{label} cancellation failed: {alive} synthesis worker(s) still alive "
            f"after {_voice_rest_cleanup_timeout():.2f}s"
        )
    return metrics


class _RestAttemptCancel:
    """Attempt-local cancellation linked to its owning voice turn."""

    def __init__(self, turn_cancel: threading.Event):
        self._turn_cancel = turn_cancel
        self._attempt_cancel = threading.Event()

    def is_set(self) -> bool:
        return self._turn_cancel.is_set() or self._attempt_cancel.is_set()

    def set(self) -> None:
        self._attempt_cancel.set()

    def wait(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        while not self.is_set():
            if deadline is None:
                self._attempt_cancel.wait(0.02)
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return self.is_set()
            self._attempt_cancel.wait(min(0.02, remaining))
        return True


@dataclass
class _RestSynthesisAttempt:
    future: Future
    cancel_event: _RestAttemptCancel | None
    completed_at: float | None = None


@dataclass
class _PendingChunk:
    index: int
    text: str
    scheduled_at: float          # monotonic seconds since turn start
    tts_kwargs: dict[str, Any]
    attempts: list[_RestSynthesisAttempt]
    hedge_started: bool = False
    tts_completed_at: float = 0  # winning attempt completion
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
    cancel_event: threading.Event | None = None,
    client_id: str | None = None,
    turn_id: str | None = None,
) -> dict[str, Any]:
    """Streaming variant of :func:`voice_ptt_turn`.

    Calls ``emit(event_name, payload)`` for each progress event:

      * ``transcript`` once with the ASR result.
      * ``text_delta`` for every chat token (as Face Lobe streams).
      * ``audio_chunk`` per sentence, IN ORDER, as TTS completes.
      * ``status`` for phase changes (``transcribing``, ``thinking``).
      * ``audio_error`` before a terminal ordered-drain if a TTS fragment fails.

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
    turn_cancel = cancel_event or threading.Event()
    turn_correlation_id = turn_id or _new_voice_turn_id()
    lifecycle: dict[str, int] = {
        "terminal_cancel_signals": 0,
        "first_chunk_timer_join_timeouts": 0,
        "rest_tasks_submitted": 0,
        "rest_tasks_completed": 0,
        "rest_workers_started": 0,
        "rest_workers_joined": 0,
        "rest_worker_join_timeouts": 0,
        "rest_emit_coordinator_started": 0,
        "rest_emit_coordinator_joined": 0,
        "rest_emit_coordinator_join_timeouts": 0,
        "rest_hol_deadline_join_timeouts": 0,
        "rest_hol_tail_futures_cancelled": 0,
        "rest_hol_tail_requeues": 0,
        "rest_hol_tail_requeue_failures": 0,
        "speaker_threads_started": 0,
        "speaker_threads_joined": 0,
        "speaker_thread_join_timeouts": 0,
        "rest_busy_retries": 0,
        "runtime_gate_retries": 0,
        "runtime_gate_recoveries": 0,
        "runtime_gate_retry_failures": 0,
    }

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
    # Finding 2: the turn-scoped cancellation fence covers readiness/ASR too. A
    # browser abort here must retire the turn before it spends ASR/Face Lobe/TTS
    # resources — honor the emit() liveness signal and re-check cancellation
    # around the (blocking, non-interruptible) readiness and ASR calls so a
    # cancel that arrived just before/during them is observed immediately after.
    if turn_cancel.is_set() or (client_alive is not None and not client_alive.is_set()):
        raise VoiceUnavailable("voice turn cancelled before ASR")
    if emit("status", {"phase": "transcribing"}) is False:
        if client_alive is not None:
            client_alive.clear()
        turn_cancel.set()
        raise VoiceUnavailable("voice turn cancelled before ASR (client gone)")
    check_voice_ready(runner.ms3_url, hivemind_url=runner.hivemind_url)
    if turn_cancel.is_set() or (client_alive is not None and not client_alive.is_set()):
        raise VoiceUnavailable("voice turn cancelled before ASR")
    t_asr_start = time.monotonic()
    asr_result = asr_fn(
        hivemind_url=runner.hivemind_url,
        audio=audio,
        filename=filename,
        model=transcribe_model,
    )
    transcript, raw_transcript = _ptt_transcript_from_asr(asr_result)
    asr_ms = int((time.monotonic() - t_asr_start) * 1000)
    if turn_cancel.is_set() or (client_alive is not None and not client_alive.is_set()):
        raise VoiceUnavailable("voice turn cancelled after ASR, before chat")
    # Remember the Face model this turn talks to so the keep-warm loop
    # keeps exactly that model resident between turns.
    record_face_model(model)
    # Speaker identification is fully async by default. Defer starting its
    # HiveMind request until the first audio chunk has been emitted so it
    # cannot compete with the Face Lobe or TTS for first-audio latency. The late
    # `speaker` event is unchanged, and the starter is idempotent so every
    # normal turn still attempts eventual identification exactly once.
    speaker_id_start_lock = threading.Lock()
    speaker_id_started = [False]
    speaker_id_thread: list[threading.Thread | None] = [None]
    speaker_event_lock = threading.Lock()
    speaker_event_open = [True]
    speaker_id_cancel = threading.Event()
    speaker_finish_lock = threading.Lock()
    speaker_id_finished = [False]

    def _emit_speaker_event(event: str, payload: dict[str, Any]) -> bool:
        with speaker_event_lock:
            if (
                not speaker_event_open[0]
                or speaker_id_cancel.is_set()
                or turn_cancel.is_set()
                or (client_alive is not None and not client_alive.is_set())
            ):
                return False
            return emit(event, payload)

    def _start_speaker_id() -> None:
        if (
            not _voice_speaker_id_async()
            or not audio
            or turn_cancel.is_set()
            or (client_alive is not None and not client_alive.is_set())
        ):
            return
        with speaker_id_start_lock:
            if speaker_id_started[0]:
                return
            speaker_id_started[0] = True
        identify_kwargs: dict[str, Any] = {}
        if _accepts_keyword(_identify_speaker_async, "timeout"):
            identify_kwargs["timeout"] = _voice_speaker_id_budget()
        speaker_id_thread[0] = _identify_speaker_async(
            runner.hivemind_url,
            audio,
            _emit_speaker_event,
            speaker_id_cancel,
            **identify_kwargs,
        )
        if speaker_id_thread[0] is not None:
            lifecycle["speaker_threads_started"] += 1

    def _finish_speaker_id() -> None:
        """Order an eventual speaker event before terminal `done`.

        This runs only after reply audio has finalized, so it cannot delay
        first audio. Identity gets a small ordering budget, then its event gate
        closes so a stuck daemon cannot emit after terminal ``done``.
        """
        with speaker_finish_lock:
            if speaker_id_finished[0]:
                return
            speaker_id_finished[0] = True
            with speaker_event_lock:
                speaker_event_open[0] = False
            if not speaker_id_cancel.is_set():
                lifecycle["terminal_cancel_signals"] += 1
            speaker_id_cancel.set()
            thread = speaker_id_thread[0]
            if thread is None:
                return
            thread.join(timeout=_voice_speaker_id_terminal_budget())
            if thread.is_alive():
                lifecycle["speaker_thread_join_timeouts"] += 1
                raise VoiceUnavailable(
                    "speaker identification cancellation failed: thread still alive after "
                    f"{_voice_speaker_id_terminal_budget():.2f}s"
                )
            lifecycle["speaker_threads_joined"] += 1

    if _voice_speaker_id_async():
        emit("transcript", {
            "text": transcript,
            "raw_text": raw_transcript,
            "asr_ms": asr_ms,
            "model": asr_result.get("model"),
            "speaker": None,
            "turn_id": turn_correlation_id,
        })
    else:
        speaker = _identify_speaker_bounded(runner.hivemind_url, audio, _voice_speaker_id_budget())
        emit("transcript", {
            "text": transcript,
            "raw_text": raw_transcript,
            "asr_ms": asr_ms,
            "model": asr_result.get("model"),
            "speaker": speaker,
            "turn_id": turn_correlation_id,
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
            _start_speaker_id()
            _finish_speaker_id()
            _egg_result["raw_transcript"] = raw_transcript
            _egg_result["turn_id"] = turn_correlation_id
            return _egg_result

    # Context-aware "buying time" reflex: classify the utterance with a
    # tiny model and emit a `reflex` event naming a fitting, varied canned
    # acknowledgment. Runs on a daemon thread CONCURRENTLY with the (slow)
    # Face Lobe reply so it lands during the dead air. Fail-soft.
    if _voice_smart_reflex_enabled():
        threading.Thread(
            target=emit_smart_reflex,
            kwargs={"emit": emit, "hivemind_url": runner.hivemind_url,
                    "transcript": transcript, "session_id": session_id,
                    "turn_started_at": t_total},
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
            ws_result = _run_ws_super_engine(
                runner=runner,
                transcript=transcript,
                session_id=session_id,
                model=model,
                depth_model=depth_model,
                tts_voice=tts_voice,
                asr_result=asr_result,
                asr_ms=asr_ms,
                tts_model=tts_model,
                response_format=response_format,
                chat_call=chat_call,
                emit=emit,
                t_total=t_total,
                start_speaker_id=_start_speaker_id,
                finish_speaker_id=_finish_speaker_id,
                ws_engine_factory=ws_engine_factory,
                client_alive=client_alive,
                cancel_event=turn_cancel,
                client_id=client_id,
                turn_id=turn_correlation_id,
                lifecycle=lifecycle,
            )
            ws_result["raw_transcript"] = raw_transcript
            return ws_result
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

    requested_tts_pool_size = max(1, tts_pool_size)
    if synthesize_fn is None:
        observed_rest_capacity, rest_capacity_provenance = (
            _voice_rest_capacity_snapshot()
        )
        effective_tts_pool_size = min(
            requested_tts_pool_size, observed_rest_capacity
        )
    else:
        # A test/application-injected synthesizer owns its own concurrency
        # contract and is not evidence about the deployed HiveMind REST pool.
        effective_tts_pool_size = requested_tts_pool_size
        rest_capacity_provenance = "injected_synthesizer"
    lifecycle["rest_capacity_requested"] = requested_tts_pool_size
    lifecycle["rest_capacity_effective"] = effective_tts_pool_size
    lifecycle["rest_capacity_provenance"] = rest_capacity_provenance
    # The browser needs this admission fact before its first decoded REST chunk:
    # only a proven capacity-two path may trade the conservative sibling hold for
    # the four-second response-start deadline.  This is scheduling metadata, not
    # a claim that synthesis completed or that a chunk is already audible.
    emit("status", {
        "phase": "thinking",
        "engine": "rest",
        "rest_capacity_effective": effective_tts_pool_size,
        "rest_capacity_provenance": rest_capacity_provenance,
    })
    tts_executor = ThreadPoolExecutor(
        max_workers=effective_tts_pool_size,
        thread_name_prefix="ms4-tts",
    )
    pending: dict[int, _PendingChunk] = {}
    pending_lock = threading.RLock()
    emit_lock = threading.Lock()
    emit_coordinator_wake = threading.Event()
    emit_coordinator_stop = threading.Event()
    emit_coordinator_started = [False]
    next_emit_idx = [0]
    all_rest_futures: list[Future] = []
    counters = {
        "audio_generated": 0,
        "audio_gateway_enqueued": 0,
        "audio_client_written": 0,
        "audio_errors": 0,
        "first_audio_at": None,
        "gated": 0,
    }
    rest_consumer_open = threading.Event()
    rest_consumer_open.set()
    qa_samples: list[tuple[str, bytes]] = []  # (text, emitted audio) for async reply QA
    t_first_token: dict[str, float | None] = {"ms": None}
    chunk_index_counter = [0]
    # Producer-minted correlation ids (R3 GREEN): one stable turn_id for this REST
    # turn; _try_emit_ready mints one unique chunk_id per emitted chunk. Both ids
    # are surfaced on the emitted audio_chunk payload AND the per-chunk metric so a
    # chunk metric and the browser turn telemetry for the same reply join 1:1.
    # server.py relays them byte-faithfully (proven by the frozen relay test).
    chat_metrics: dict[str, Any] = {}
    # Marker-confirmation contract for this turn (inactive for ordinary turns).
    marker_gate = _MarkerConfirmationGate(transcript)

    chunk_timings: list[dict[str, Any]] = []
    # Keep one speech filter for the entire REST turn. SentenceChunker may split
    # between an opening and closing fenced-code marker, so stateless per-chunk
    # sanitization can otherwise speak code that the WS route correctly drops.
    # chunk_scheduled still reports the exact sanitized text sent to TTS.
    from .spoken_text_filter import SpokenTextFilter, sanitize_for_speech as _sanitize

    rest_tts_filter = SpokenTextFilter() if TTS_FILTER_ENABLED else None
    chunker.first_chunk_sanitizer = _sanitize if TTS_FILTER_ENABLED else None

    tail_dispatch_fully_released = threading.Event()
    tail_dispatch_lock = threading.Lock()
    lead_tail_release_seen = [False]
    # The built-in REST transport fills the proven capacity and may queue one
    # continuation behind it when the caller retains the full default pool.
    # That queued future starts as soon as c0/c1 frees a worker, before their
    # SSE writes can consume the available playback runway. Custom/injected
    # synthesizers retain the historical one-lead behavior.
    lead_tail_budget = (
        min(
            effective_tts_pool_size,
            1 + int(requested_tts_pool_size >= TTS_POOL_SIZE),
        )
        if synthesize_fn is None
        else min(1, max(0, effective_tts_pool_size - 1))
    )
    lead_tails_dispatched = [0]
    extra_lead_ack_pending = threading.Event()
    deferred_tail_dispatches: list[tuple[int, str, dict[str, Any]]] = []
    active_hol_timer: list[threading.Timer | None] = [None]
    active_hol_chunk: list[_PendingChunk | None] = [None]
    active_hol_fatal_timer: list[threading.Timer | None] = [None]
    active_hol_fatal_chunk: list[_PendingChunk | None] = [None]
    hol_timers: list[threading.Timer] = []
    rest_hol_fatal_timeout_s = _voice_rest_hol_fatal_timeout()
    fatal_rest_error: list[VoiceUnavailable | None] = [None]

    def _increment_lifecycle(name: str) -> None:
        with pending_lock:
            lifecycle[name] += 1

    def _run_synthesis_with_busy_retry(**attempt_kwargs: Any) -> dict[str, Any]:
        return _call_tts_with_runtime_audio_gate_recovery(
            tts_fn,
            attempt_kwargs,
            turn_cancel=turn_cancel,
            client_alive=client_alive,
            on_busy_retry=lambda: _increment_lifecycle("rest_busy_retries"),
            on_gate_retry=lambda: _increment_lifecycle("runtime_gate_retries"),
            on_gate_recovery=lambda: _increment_lifecycle("runtime_gate_recoveries"),
            on_gate_retry_failure=lambda: _increment_lifecycle(
                "runtime_gate_retry_failures"
            ),
        )

    def _attempt_kwargs(
        tts_kwargs: dict[str, Any],
    ) -> tuple[dict[str, Any], _RestAttemptCancel | None]:
        attempt_kwargs = dict(tts_kwargs)
        attempt_cancel: _RestAttemptCancel | None = None
        if "cancel_event" in attempt_kwargs:
            attempt_cancel = _RestAttemptCancel(turn_cancel)
            attempt_kwargs["cancel_event"] = attempt_cancel
        return attempt_kwargs, attempt_cancel

    def _later_ready_locked(chunk: _PendingChunk) -> bool:
        return any(
            later_idx > chunk.index
            and any(
                attempt.completed_at is not None
                and not attempt.future.cancelled()
                and attempt.future.exception() is None
                for attempt in later_chunk.attempts
            )
            for later_idx, later_chunk in pending.items()
        )

    def _cancel_hol_deadline_locked(
        chunk: _PendingChunk | None = None,
    ) -> None:
        timer = active_hol_timer[0]
        if timer is None or (chunk is not None and active_hol_chunk[0] is not chunk):
            return
        active_hol_timer[0] = None
        active_hol_chunk[0] = None
        timer.cancel()

    def _cancel_hol_fatal_deadline_locked(
        chunk: _PendingChunk | None = None,
    ) -> None:
        timer = active_hol_fatal_timer[0]
        if (
            timer is None
            or (chunk is not None and active_hol_fatal_chunk[0] is not chunk)
        ):
            return
        active_hol_fatal_timer[0] = None
        active_hol_fatal_chunk[0] = None
        timer.cancel()

    def _on_hol_deadline(
        timer: threading.Timer,
        idx: int,
        expected_chunk: _PendingChunk,
    ) -> None:
        start_hedge = False
        with pending_lock:
            if active_hol_timer[0] is not timer:
                return
            active_hol_timer[0] = None
            active_hol_chunk[0] = None
            chunk = pending.get(idx)
            if (
                chunk is expected_chunk
                and next_emit_idx[0] == idx
                and not chunk.hedge_started
                and not turn_cancel.is_set()
                and rest_consumer_open.is_set()
                and (client_alive is None or client_alive.is_set())
                and _later_ready_locked(chunk)
            ):
                chunk.hedge_started = True
                start_hedge = True
        if start_hedge:
            _start_hol_hedge(idx, expected_chunk)

    def _arm_hol_deadline_locked(
        chunk: _PendingChunk,
        remaining_s: float,
    ) -> None:
        if active_hol_chunk[0] is chunk and active_hol_timer[0] is not None:
            return
        _cancel_hol_deadline_locked()
        timer: threading.Timer
        timer = threading.Timer(
            remaining_s,
            lambda: _on_hol_deadline(timer, chunk.index, chunk),
        )
        timer.name = "ms4-rest-hol-hedge-deadline"
        timer.daemon = True
        active_hol_timer[0] = timer
        active_hol_chunk[0] = chunk
        hol_timers.append(timer)
        timer.start()

    def _on_hol_fatal_deadline(
        timer: threading.Timer,
        idx: int,
        expected_chunk: _PendingChunk,
    ) -> None:
        with pending_lock:
            if active_hol_fatal_timer[0] is not timer:
                return
            active_hol_fatal_timer[0] = None
            active_hol_fatal_chunk[0] = None
            chunk = pending.get(idx)
            if (
                chunk is not expected_chunk
                or next_emit_idx[0] != idx
                or turn_cancel.is_set()
                or not rest_consumer_open.is_set()
                or (client_alive is not None and not client_alive.is_set())
                or not _later_ready_locked(chunk)
            ):
                return
            resolved_successfully = any(
                attempt.future.done()
                and not attempt.future.cancelled()
                and attempt.future.exception() is None
                for attempt in chunk.attempts
            )
            if resolved_successfully or all(
                attempt.future.done() for attempt in chunk.attempts
            ):
                return
            if fatal_rest_error[0] is not None:
                return
            fatal_rest_error[0] = VoiceUnavailable(
                f"REST TTS continuation blocked at chunk index {idx}: "
                f"head-of-line deadline exceeded {rest_hol_fatal_timeout_s:.2f}s"
            )
            rest_consumer_open.clear()
            _cancel_hol_deadline_locked(chunk)
            turn_cancel.set()
            for queued in pending.values():
                for attempt in queued.attempts:
                    if attempt.cancel_event is not None:
                        attempt.cancel_event.set()
                    attempt.future.cancel()

    def _arm_hol_fatal_deadline_locked(chunk: _PendingChunk) -> None:
        if (
            active_hol_fatal_chunk[0] is chunk
            and active_hol_fatal_timer[0] is not None
        ):
            return
        _cancel_hol_fatal_deadline_locked()
        remaining_s = rest_hol_fatal_timeout_s
        timer: threading.Timer
        timer = threading.Timer(
            remaining_s,
            lambda: _on_hol_fatal_deadline(timer, chunk.index, chunk),
        )
        timer.name = "ms4-rest-hol-fatal-deadline"
        timer.daemon = True
        active_hol_fatal_timer[0] = timer
        active_hol_fatal_chunk[0] = chunk
        hol_timers.append(timer)
        timer.start()

    def _settle_hol_deadlines() -> int:
        with pending_lock:
            _cancel_hol_deadline_locked()
            _cancel_hol_fatal_deadline_locked()
            timers = list(hol_timers)
        for timer in timers:
            timer.cancel()
        for timer in timers:
            if timer is not threading.current_thread():
                timer.join(timeout=_voice_rest_cleanup_timeout())
        return sum(timer.is_alive() for timer in timers)

    def _rest_futures_snapshot() -> list[Future]:
        with pending_lock:
            return list(all_rest_futures)

    def _register_attempt(idx: int, attempt: _RestSynthesisAttempt) -> None:
        with pending_lock:
            all_rest_futures.append(attempt.future)

        def _on_done(_future: Future) -> None:
            with pending_lock:
                lifecycle["rest_tasks_completed"] += int(not _future.cancelled())
                attempt.completed_at = time.monotonic() - t_total
            emit_coordinator_wake.set()

        attempt.future.add_done_callback(_on_done)

    def _dispatch_synthesis(
        idx: int,
        speakable: str,
        tts_kwargs: dict[str, Any],
        *,
        defer_schedule_ack: bool = False,
    ) -> bool:
        if (
            turn_cancel.is_set()
            or
            not rest_consumer_open.is_set()
            or (client_alive is not None and not client_alive.is_set())
        ):
            return False
        scheduled_at = time.monotonic() - t_total

        def _write_schedule_ack() -> bool:
            nonlocal scheduled_at
            scheduled_at = time.monotonic() - t_total
            with pending_lock:
                if idx in pending:
                    pending[idx].scheduled_at = scheduled_at
            try:
                acknowledged = emit("chunk_scheduled", {
                    "index": idx,
                    "text": speakable,
                    "scheduled_ms": int(scheduled_at * 1000),
                })
            except Exception:
                rest_consumer_open.clear()
                turn_cancel.set()
                if client_alive is not None:
                    client_alive.clear()
                raise
            if acknowledged is not False:
                return True
            rest_consumer_open.clear()
            turn_cancel.set()
            if client_alive is not None:
                client_alive.clear()
            return False

        registration_ready: threading.Event | None = None
        if defer_schedule_ack:
            # Only the additional speculative lead uses this wrapper. Its ACK
            # remains mandatory before synthesis, but it runs inside the same
            # tracked executor so chunk 0's response-read callback never waits
            # on the extra client's write acknowledgement.
            registration_ready = threading.Event()
            extra_lead_ack_pending.set()

            def _ack_then_synthesize(**kwargs):
                registration_ready.wait()
                try:
                    if (
                        turn_cancel.is_set()
                        or not rest_consumer_open.is_set()
                        or (client_alive is not None and not client_alive.is_set())
                    ):
                        raise VoiceUnavailable(
                            f"REST chunk {idx} cancelled before schedule acknowledgement"
                        )
                    if not _write_schedule_ack():
                        raise VoiceUnavailable(
                            f"REST chunk {idx} schedule acknowledgement was rejected"
                        )
                finally:
                    extra_lead_ack_pending.clear()
                    if tail_dispatch_fully_released.is_set():
                        _release_all_tails()
                if (
                    turn_cancel.is_set()
                    or (client_alive is not None and not client_alive.is_set())
                ):
                    raise VoiceUnavailable(
                        f"REST chunk {idx} cancelled after schedule acknowledgement"
                    )
                return _run_synthesis_with_busy_retry(**kwargs)

            lifecycle["rest_tasks_submitted"] += 1
            attempt_kwargs, attempt_cancel = _attempt_kwargs(tts_kwargs)
            try:
                fut = tts_executor.submit(_ack_then_synthesize, **attempt_kwargs)
            except Exception:
                extra_lead_ack_pending.clear()
                raise
        else:
            if not _write_schedule_ack():
                return False
            lifecycle["rest_tasks_submitted"] += 1
            attempt_kwargs, attempt_cancel = _attempt_kwargs(tts_kwargs)
            fut = tts_executor.submit(
                _run_synthesis_with_busy_retry,
                **attempt_kwargs,
            )
        attempt = _RestSynthesisAttempt(future=fut, cancel_event=attempt_cancel)
        with pending_lock:
            pending[idx] = _PendingChunk(
                index=idx,
                text=speakable,
                scheduled_at=scheduled_at,
                tts_kwargs=dict(tts_kwargs),
                attempts=[attempt],
            )
        if registration_ready is not None:
            registration_ready.set()
        _register_attempt(idx, attempt)
        return True

    def _schedule(text: str) -> bool:
        if (
            turn_cancel.is_set()
            or
            not rest_consumer_open.is_set()
            or (client_alive is not None and not client_alive.is_set())
        ):
            return False
        if rest_tts_filter is None:
            speakable = text.strip()
        else:
            ready_parts = [rest_tts_filter.push(text)]
            # SentenceChunker has already declared this prose fragment ready.
            # Drain an ordinary non-code tail immediately while retaining state
            # across complete or unterminated fenced blocks.
            if not rest_tts_filter.in_code_block and rest_tts_filter.buffer:
                ready_parts.append(_sanitize(rest_tts_filter.buffer))
                rest_tts_filter.buffer = ""
            speakable = " ".join(
                part.strip() for part in ready_parts if part.strip()
            )
        if not speakable:
            # Sanitizer ate the whole chunk (e.g. it was just a bullet
            # marker or an emoji). Don't schedule an empty TTS call.
            return True
        for fragment in _split_xtts_safe_fragments(speakable):
            if not _schedule_speakable(fragment):
                return False
        return True

    def _schedule_speakable(speakable: str) -> bool:
        idx = chunk_index_counter[0]
        chunk_index_counter[0] += 1
        tts_kwargs: dict[str, Any] = {
            "hivemind_url": runner.hivemind_url,
            "text": speakable,
            "model": tts_model,
            "voice": tts_voice,
            "response_format": response_format,
        }
        if _accepts_keyword(tts_fn, "cancel_event"):
            tts_kwargs["cancel_event"] = turn_cancel
        if idx == 0:
            if _accepts_keyword(tts_fn, "on_request_dispatched"):
                tts_kwargs["on_request_dispatched"] = _release_lead_tail
            return _dispatch_synthesis(idx, speakable, tts_kwargs)
        with tail_dispatch_lock:
            if (
                tail_dispatch_fully_released.is_set()
                and not extra_lead_ack_pending.is_set()
            ):
                return _dispatch_synthesis(idx, speakable, tts_kwargs)
            if (
                lead_tail_release_seen[0]
                and lead_tails_dispatched[0] < lead_tail_budget
            ):
                dispatched = _dispatch_synthesis(
                    idx,
                    speakable,
                    tts_kwargs,
                    defer_schedule_ack=lead_tails_dispatched[0] > 0,
                )
                lead_tails_dispatched[0] += int(dispatched)
                return dispatched
            deferred_tail_dispatches.append((idx, speakable, tts_kwargs))
        return True

    def _release_lead_tail() -> None:
        with tail_dispatch_lock:
            if tail_dispatch_fully_released.is_set() or lead_tail_release_seen[0]:
                return
            if (
                turn_cancel.is_set()
                or not rest_consumer_open.is_set()
                or (client_alive is not None and not client_alive.is_set())
            ):
                deferred_tail_dispatches.clear()
                return
            lead_tail_release_seen[0] = True
            while (
                deferred_tail_dispatches
                and lead_tails_dispatched[0] < lead_tail_budget
            ):
                idx, speakable, tts_kwargs = deferred_tail_dispatches.pop(0)
                if not _dispatch_synthesis(
                    idx,
                    speakable,
                    tts_kwargs,
                    defer_schedule_ack=lead_tails_dispatched[0] > 0,
                ):
                    break
                lead_tails_dispatched[0] += 1

    def _release_all_tails() -> None:
        with tail_dispatch_lock:
            if (
                turn_cancel.is_set()
                or not rest_consumer_open.is_set()
                or (client_alive is not None and not client_alive.is_set())
            ):
                deferred_tail_dispatches.clear()
                return
            tail_dispatch_fully_released.set()
            if extra_lead_ack_pending.is_set():
                return
            queued = list(deferred_tail_dispatches)
            deferred_tail_dispatches.clear()
            for idx, speakable, tts_kwargs in queued:
                if not _dispatch_synthesis(idx, speakable, tts_kwargs):
                    break

    def _start_hol_hedge(
        idx: int,
        expected_chunk: _PendingChunk | None = None,
    ) -> None:
        with tail_dispatch_lock, pending_lock:
            chunk = pending.get(idx)
            if (
                chunk is None
                or (expected_chunk is not None and chunk is not expected_chunk)
                or turn_cancel.is_set()
                or not rest_consumer_open.is_set()
                or (client_alive is not None and not client_alive.is_set())
            ):
                return
            attempt_kwargs, attempt_cancel = _attempt_kwargs(chunk.tts_kwargs)
            if attempt_cancel is None:
                return

            cancelled_later_chunks: list[_PendingChunk] = []
            cancelled_later_indices: set[int] = set()
            for later_idx in sorted(pending):
                if later_idx <= idx:
                    continue
                later_chunk = pending[later_idx]
                for later_attempt in later_chunk.attempts:
                    later_future = later_attempt.future
                    if later_future.done() or later_future.running():
                        continue
                    if later_future.cancel():
                        lifecycle["rest_hol_tail_futures_cancelled"] += 1
                        if later_idx not in cancelled_later_indices:
                            cancelled_later_indices.add(later_idx)
                            cancelled_later_chunks.append(later_chunk)

            def _requeue_cancelled_later_locked() -> None:
                failed_requeues: list[int] = []
                for later_chunk in cancelled_later_chunks:
                    if (
                        turn_cancel.is_set()
                        or not rest_consumer_open.is_set()
                        or (client_alive is not None and not client_alive.is_set())
                    ):
                        return
                    replacement_kwargs, replacement_cancel = _attempt_kwargs(
                        later_chunk.tts_kwargs
                    )
                    try:
                        replacement_future = tts_executor.submit(
                            _run_synthesis_with_busy_retry,
                            **replacement_kwargs,
                        )
                    except RuntimeError:
                        lifecycle["rest_hol_tail_requeue_failures"] += 1
                        failed_requeues.append(later_chunk.index)
                        continue
                    lifecycle["rest_tasks_submitted"] += 1
                    lifecycle["rest_hol_tail_requeues"] += 1
                    replacement = _RestSynthesisAttempt(
                        future=replacement_future,
                        cancel_event=replacement_cancel,
                    )
                    later_chunk.attempts.append(replacement)
                    _register_attempt(later_chunk.index, replacement)
                if not failed_requeues:
                    return
                failed_indices = ", ".join(str(index) for index in failed_requeues)
                if fatal_rest_error[0] is None:
                    fatal_rest_error[0] = VoiceUnavailable(
                        "REST HOL hedge tail restoration failed for chunk "
                        f"index(es): {failed_indices}"
                    )
                rest_consumer_open.clear()
                turn_cancel.set()
                for queued in pending.values():
                    for queued_attempt in queued.attempts:
                        if queued_attempt.cancel_event is not None:
                            queued_attempt.cancel_event.set()
                        queued_attempt.future.cancel()

            try:
                future = tts_executor.submit(
                    _run_synthesis_with_busy_retry,
                    **attempt_kwargs,
                )
            except RuntimeError:
                if pending.get(idx) is chunk:
                    chunk.hedge_started = False
                _requeue_cancelled_later_locked()
                return
            lifecycle["rest_tasks_submitted"] += 1
            attempt = _RestSynthesisAttempt(
                future=future,
                cancel_event=attempt_cancel,
            )
            stale = (
                pending.get(idx) is not chunk
                or turn_cancel.is_set()
                or not rest_consumer_open.is_set()
                or (client_alive is not None and not client_alive.is_set())
            )
            if not stale:
                chunk.attempts.append(attempt)
            if stale:
                attempt_cancel.set()
                future.cancel()
            _register_attempt(idx, attempt)
            if not stale:
                _requeue_cancelled_later_locked()

    def _try_emit_ready() -> None:
        if (
            turn_cancel.is_set()
            or not rest_consumer_open.is_set()
            or (client_alive is not None and not client_alive.is_set())
        ):
            return
        release_all_tail = False
        hedge_idx: int | None = None
        with emit_lock:
            while True:
                with pending_lock:
                    if next_emit_idx[0] not in pending:
                        break
                    if (
                        turn_cancel.is_set()
                        or not rest_consumer_open.is_set()
                        or (client_alive is not None and not client_alive.is_set())
                    ):
                        return
                    chunk = pending[next_emit_idx[0]]
                    completed = [
                        attempt
                        for attempt in chunk.attempts
                        if attempt.completed_at is not None
                    ]
                    successful = [
                        attempt
                        for attempt in completed
                        if not attempt.future.cancelled()
                        and attempt.future.exception() is None
                    ]
                    if successful:
                        selected = min(
                            successful,
                            key=lambda attempt: attempt.completed_at,
                        )
                    elif len(completed) == len(chunk.attempts):
                        selected = min(
                            completed,
                            key=lambda attempt: attempt.completed_at,
                        )
                    else:
                        later_ready = _later_ready_locked(chunk)
                        if later_ready:
                            _arm_hol_fatal_deadline_locked(chunk)
                        if (
                            later_ready
                            and not chunk.hedge_started
                            and "cancel_event" in chunk.tts_kwargs
                        ):
                            remaining_s = (
                                _voice_rest_hol_hedge_delay()
                                - (time.monotonic() - t_total - chunk.scheduled_at)
                            )
                            if remaining_s <= 0:
                                _cancel_hol_deadline_locked(chunk)
                                chunk.hedge_started = True
                                hedge_idx = chunk.index
                            else:
                                _arm_hol_deadline_locked(chunk, remaining_s)
                        break
                    _cancel_hol_deadline_locked(chunk)
                    _cancel_hol_fatal_deadline_locked(chunk)
                    assert selected.completed_at is not None
                    chunk.tts_completed_at = selected.completed_at
                    for attempt in chunk.attempts:
                        if attempt is selected or attempt.future.done():
                            continue
                        if attempt.cancel_event is not None:
                            attempt.cancel_event.set()
                        attempt.future.cancel()
                try:
                    tts_result = selected.future.result()
                    chunk.emitted_at = time.monotonic() - t_total
                    # Per-chunk timings exposed so the operator can see
                    # (a) actual TTS latency on HiveMind's side
                    # (b) how long the in-order emit guarantee held a chunk
                    tts_ms = int((chunk.tts_completed_at - chunk.scheduled_at) * 1000)
                    held_ms = max(0, int((chunk.emitted_at - chunk.tts_completed_at) * 1000))
                    # Producer-minted per-chunk correlation id (unique across the
                    # turn via the strictly-increasing emit index); the same id is
                    # placed on the emitted audio_chunk payload below.
                    chunk_correlation_id = f"{turn_correlation_id}-c{chunk.index}"
                    timing = {
                        "index": chunk.index,
                        "text_len": len(chunk.text),
                        "scheduled_ms": int(chunk.scheduled_at * 1000),
                        "tts_completed_ms": int(chunk.tts_completed_at * 1000),
                        "emitted_ms": int(chunk.emitted_at * 1000),
                        "tts_ms": tts_ms,
                        "held_for_inorder_ms": held_ms,
                        "turn_id": turn_correlation_id,
                        "chunk_id": chunk_correlation_id,
                    }
                    # Retain non-secret HLI provenance on the per-chunk metric
                    # (fail closed on secrets); absent when the reply carried none.
                    chunk_provenance = _sanitize_hli_provenance(
                        tts_result.get("provenance")
                    )
                    if chunk_provenance:
                        timing["provenance"] = chunk_provenance
                    # Runtime garble gate: trim a babble tail off this live
                    # chunk before it plays (instant, model-free). The async
                    # verifier (opt-in) double-checks what actually played.
                    audio_bytes = tts_result["audio_bytes"]
                    if not audio_bytes:
                        raise VoiceUnavailable("HiveMind /v1/audio/speech returned empty body")
                    counters["audio_generated"] += 1
                    audio_mime = tts_result["content_type"]
                    audio_b64 = tts_result["audio_base64"]
                    runtime_gate_telemetry = tts_result.get(
                        _RUNTIME_AUDIO_GATE_META_KEY
                    )
                    if isinstance(runtime_gate_telemetry, dict):
                        runtime_gate_telemetry = dict(runtime_gate_telemetry)
                    else:
                        runtime_gate_telemetry = None
                    selected_original_duration_s = wav_duration_secs(audio_bytes)
                    if _runtime_audio_gate_enabled():
                        gated_bytes, trimmed = _gate_chunk_audio(audio_bytes, audio_mime, chunk.text)
                        if trimmed:
                            audio_bytes = gated_bytes
                            audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
                    else:
                        trimmed = False
                    if runtime_gate_telemetry is not None:
                        emitted_duration_s = wav_duration_secs(audio_bytes)
                        runtime_gate_telemetry.update({
                            "selected_original_duration_ms": (
                                round(selected_original_duration_s * 1000.0, 3)
                                if selected_original_duration_s is not None
                                else None
                            ),
                            "emitted_duration_ms": (
                                round(emitted_duration_s * 1000.0, 3)
                                if emitted_duration_s is not None
                                else None
                            ),
                            "trim_delta_ms": (
                                round(max(
                                    0.0,
                                    selected_original_duration_s - emitted_duration_s,
                                ) * 1000.0, 3)
                                if (
                                    selected_original_duration_s is not None
                                    and emitted_duration_s is not None
                                )
                                else None
                            ),
                            "final_runtime_gated": trimmed,
                        })
                        timing["runtime_gate"] = runtime_gate_telemetry
                    counters["audio_gateway_enqueued"] += 1
                    audio_emitted = emit("audio_chunk", {
                        "index": chunk.index,
                        "text": chunk.text,
                        "audio_base64": audio_b64,
                        "audio_mime": audio_mime,
                        "tts_ms": tts_ms,
                        "held_for_inorder_ms": held_ms,
                        "turn_id": turn_correlation_id,
                        "chunk_id": chunk_correlation_id,
                        # A duration-gated WAV may no longer contain the whole
                        # text carried in this same envelope. Surface that fact
                        # on the exact chunk instead of allowing metadata-only
                        # binding to imply complete audible delivery.
                        "runtime_gated": trimmed,
                    })
                    if audio_emitted is False:
                        rest_consumer_open.clear()
                        turn_cancel.set()
                        if client_alive is not None:
                            client_alive.clear()
                        with pending_lock:
                            _cancel_hol_deadline_locked()
                            _cancel_hol_fatal_deadline_locked()
                            for queued in pending.values():
                                for attempt in queued.attempts:
                                    attempt.future.cancel()
                            pending.clear()
                            next_emit_idx[0] = chunk_index_counter[0]
                        return
                    if counters["first_audio_at"] is None:
                        counters["first_audio_at"] = int((time.monotonic() - t_total) * 1000)
                    counters["audio_client_written"] += 1
                    chunk_timings.append(timing)
                    if trimmed:
                        counters["gated"] += 1
                        log.info("runtime gate trimmed a babble tail on chunk %d (%r)",
                                 chunk.index, chunk.text[:30])
                    if _runtime_qa_enabled():
                        qa_samples.append((chunk.text, audio_bytes))
                    _start_speaker_id()
                except Exception as exc:
                    if not turn_cancel.is_set():
                        try:
                            emit(
                                "audio_error",
                                {
                                    "index": chunk.index,
                                    "text": chunk.text,
                                    "error": str(exc),
                                },
                            )
                        except Exception:  # noqa: BLE001 - terminal state still wins
                            pass
                    counters["audio_errors"] += 1
                    message = (
                        f"REST TTS terminal failure at chunk index {chunk.index}: {exc}"
                    )
                    if counters["audio_client_written"] == 0:
                        message += (
                            "; voice output unavailable: speakable reply produced "
                            "zero client-written audio"
                        )
                    with pending_lock:
                        if fatal_rest_error[0] is None:
                            fatal_rest_error[0] = VoiceUnavailable(message)
                        rest_consumer_open.clear()
                        _cancel_hol_deadline_locked()
                        _cancel_hol_fatal_deadline_locked()
                        for queued in pending.values():
                            for attempt in queued.attempts:
                                if attempt.cancel_event is not None:
                                    attempt.cancel_event.set()
                                attempt.future.cancel()
                        pending.clear()
                        next_emit_idx[0] = chunk_index_counter[0]
                    turn_cancel.set()
                    return
                with pending_lock:
                    completed_idx = next_emit_idx[0]
                    if pending.get(completed_idx) is chunk:
                        del pending[completed_idx]
                    next_emit_idx[0] += 1
                release_all_tail = release_all_tail or completed_idx == 0
        if release_all_tail:
            # First successful delivery opens full
            # parallel fanout; the request-dispatched callback opens only the
            # bounded lead set above.
            _release_all_tails()
        if hedge_idx is not None:
            _start_hol_hedge(hedge_idx)

    def _run_emit_coordinator() -> None:
        while not emit_coordinator_stop.is_set():
            emit_coordinator_wake.wait()
            emit_coordinator_wake.clear()
            if emit_coordinator_stop.is_set():
                return
            try:
                _try_emit_ready()
            except Exception as exc:  # noqa: BLE001 - fail the turn closed
                with pending_lock:
                    if fatal_rest_error[0] is None:
                        fatal_rest_error[0] = VoiceUnavailable(
                            f"REST emission coordinator failed: {exc}"
                        )
                    rest_consumer_open.clear()
                turn_cancel.set()
                if client_alive is not None:
                    client_alive.clear()
                return

    emit_coordinator_thread = threading.Thread(
        target=_run_emit_coordinator,
        name="ms4-rest-emit-coordinator",
        daemon=True,
    )

    def _stop_emit_coordinator_bounded() -> None:
        if not emit_coordinator_started[0]:
            return
        emit_coordinator_stop.set()
        emit_coordinator_wake.set()
        if emit_coordinator_thread is not threading.current_thread():
            emit_coordinator_thread.join(timeout=_voice_rest_cleanup_timeout())
        alive = int(emit_coordinator_thread.is_alive())
        lifecycle["rest_emit_coordinator_joined"] += 1 - alive
        lifecycle["rest_emit_coordinator_join_timeouts"] += alive
        if alive:
            raise VoiceUnavailable(
                "REST voice stream cancellation failed: emission coordinator "
                f"still alive after {_voice_rest_cleanup_timeout():.2f}s"
            )
        emit_coordinator_started[0] = False

    chunk_controller = _FirstChunkDeadlineController(
        chunker=chunker,
        schedule=_schedule,
        active=lambda: (
            rest_consumer_open.is_set()
            and not turn_cancel.is_set()
            and (client_alive is None or client_alive.is_set())
        ),
    )
    rest_cleanup_complete = [False]

    def _cleanup_rest_turn_bounded(*, cancel_pending: bool) -> None:
        """Best-effort every independent phase of one bounded REST cleanup."""
        if rest_cleanup_complete[0]:
            return

        cleanup_failures: list[tuple[str, Exception]] = []

        def _record_cleanup_failure(phase: str, exc: Exception) -> None:
            cleanup_failures.append((phase, exc))
            log.error(
                "REST turn cleanup phase %s failed: %s: %s",
                phase,
                type(exc).__name__,
                exc,
                exc_info=True,
            )

        if cancel_pending:
            rest_consumer_open.clear()
            turn_cancel.set()
        try:
            lifecycle["first_chunk_timer_join_timeouts"] += int(
                chunk_controller.close()
            )
        except Exception as exc:  # noqa: BLE001 - continue independent cleanup
            _record_cleanup_failure("first-chunk timer close", exc)
        try:
            lifecycle["rest_hol_deadline_join_timeouts"] += _settle_hol_deadlines()
        except Exception as exc:  # noqa: BLE001 - continue independent cleanup
            _record_cleanup_failure("HOL deadline settlement", exc)
        try:
            rest_futures = _rest_futures_snapshot()
        except Exception as exc:  # noqa: BLE001 - executor shutdown must still run
            _record_cleanup_failure("future snapshot", exc)
            rest_futures = []
        try:
            if cancel_pending or not all(future.done() for future in rest_futures):
                cleanup = _shutdown_executor_bounded(
                    tts_executor,
                    rest_futures,
                    cancel_event=turn_cancel,
                    label="REST voice stream",
                )
                lifecycle["rest_workers_started"] += cleanup["workers_started"]
                lifecycle["rest_workers_joined"] += cleanup["workers_joined"]
                lifecycle["rest_worker_join_timeouts"] += cleanup[
                    "worker_join_timeouts"
                ]
            else:
                tts_executor.shutdown(wait=True)
                rest_threads = list(getattr(tts_executor, "_threads", ()))
                lifecycle["rest_workers_started"] += len(rest_threads)
                lifecycle["rest_workers_joined"] += len(rest_threads)
        except Exception as exc:  # noqa: BLE001 - coordinator must still stop
            _record_cleanup_failure("future/executor shutdown", exc)
        try:
            _stop_emit_coordinator_bounded()
        except Exception as exc:  # noqa: BLE001 - report after all phases
            _record_cleanup_failure("emit coordinator stop", exc)
        rest_cleanup_complete[0] = True
        if cleanup_failures:
            first_phase, first_failure = cleanup_failures[0]
            first_failure.add_note(f"REST turn cleanup failed in {first_phase}")
            for phase, failure in cleanup_failures[1:]:
                first_failure.add_note(
                    "Additional REST turn cleanup failure in "
                    f"{phase}: {type(failure).__name__}: {failure}"
                )
            raise first_failure

    def _stream_text(text: str) -> bool:
        """Emit user-visible ``text_delta`` for ``text`` and feed it to the
        sentence chunker (TTS scheduling). Shared by ordinary streamed deltas
        and the marker-contract release below."""
        if emit("text_delta", {"text": text}) is False:
            rest_consumer_open.clear()
            turn_cancel.set()
            if client_alive is not None:
                client_alive.clear()
            return False
        return chunk_controller.add(text)

    def _on_delta(delta: str) -> bool:
        if t_first_token["ms"] is None and delta.strip():
            t_first_token["ms"] = int((time.monotonic() - t_total) * 1000)
        # Marker-confirmation contract: on an EXPLICIT "confirm you heard the
        # <marker> marker" turn, BUFFER model deltas so an invalid reply never
        # reaches text_delta / chunk_scheduled / TTS. The release (original if
        # it confirms the marker, else a deterministic confirmation) is streamed
        # once, after the completion is checked. Ordinary turns (gate inactive)
        # stream immediately, unchanged — no first-audio-latency regression.
        if marker_gate.active:
            marker_gate.buffer_delta(delta)
            return True
        return _stream_text(delta)

    t_chat_start = time.monotonic()
    try:
        emit_coordinator_thread.start()
        emit_coordinator_started[0] = True
        lifecycle["rest_emit_coordinator_started"] += 1
        _chat_kw: dict[str, Any] = {"session_id": session_id, "model": model}
        if depth_model:
            _chat_kw["depth_model"] = depth_model
        if _voice_brevity_enabled():
            _chat_kw["voice_mode"] = True
        if client_id:
            _chat_kw["client_id"] = client_id
        if _accepts_keyword(chat_call, "turn_id"):
            _chat_kw["turn_id"] = turn_correlation_id
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
            cancel_event=turn_cancel,
            client_alive=client_alive,
        )
    except Exception as exc:
        try:
            _cleanup_rest_turn_bounded(cancel_pending=True)
        except Exception as cleanup_exc:
            exc.add_note(
                "REST turn cleanup also failed: "
                f"{type(cleanup_exc).__name__}: {cleanup_exc}"
            )
            log.error(
                "REST turn cleanup failed while preserving primary exception: %s: %s",
                type(cleanup_exc).__name__,
                cleanup_exc,
                exc_info=True,
            )
            _finish_speaker_id()
            raise exc from cleanup_exc
        _finish_speaker_id()
        if fatal_rest_error[0] is not None:
            raise fatal_rest_error[0] from exc
        raise
    chat_ms = int((time.monotonic() - t_chat_start) * 1000)
    chat_metrics = (chat_response or {}).get("metrics") or {}

    def _finish_rest_after_chat() -> bool:
        # Marker-confirmation contract release: the model reply is complete.
        if marker_gate.active:
            _model_reply = (
                (chat_response or {}).get("text")
                if isinstance(chat_response, dict)
                else ""
            )
            _release_text = marker_gate.resolve(_model_reply or "")
            if _release_text and rest_consumer_open.is_set():
                _stream_text(_release_text)

        # Flush the final unterminated fragment into the same TTS path.
        tail = chunk_controller.flush()
        if tail and rest_consumer_open.is_set():
            _schedule(tail)
        if rest_tts_filter is not None:
            filtered_tail = rest_tts_filter.flush().strip()
            if filtered_tail and rest_consumer_open.is_set():
                for fragment in _split_xtts_safe_fragments(filtered_tail):
                    if not _schedule_speakable(fragment):
                        break

        drain_started_at = time.monotonic()
        hard_timeout_s = _voice_rest_final_timeout(
            piece_count=chunk_index_counter[0],
            capacity=effective_tts_pool_size,
        )
        progress_timeout_s = _voice_rest_progress_timeout()
        hard_deadline = drain_started_at + hard_timeout_s
        last_progress_at = drain_started_at
        last_emitted_count = counters["audio_client_written"]
        cleanup_required = False
        while True:
            with pending_lock:
                rest_futures = list(all_rest_futures)
                work_remaining = (
                    next_emit_idx[0] < chunk_index_counter[0]
                    or not all(future.done() for future in rest_futures)
                )
            if not work_remaining:
                break
            if (
                turn_cancel.is_set()
                or not rest_consumer_open.is_set()
                or (client_alive is not None and not client_alive.is_set())
            ):
                cleanup_required = True
                break
            _try_emit_ready()
            now = time.monotonic()
            emitted_count = counters["audio_client_written"]
            if emitted_count > last_emitted_count:
                last_emitted_count = emitted_count
                last_progress_at = now
            with pending_lock:
                rest_futures = list(all_rest_futures)
                work_remaining = (
                    next_emit_idx[0] < chunk_index_counter[0]
                    or not all(future.done() for future in rest_futures)
                )
            if not work_remaining:
                break
            if now >= hard_deadline:
                message = (
                    "REST TTS hard deadline exceeded after "
                    f"{hard_timeout_s:.2f}s with "
                    f"{last_emitted_count}/{chunk_index_counter[0]} chunks "
                    "client-written"
                )
                log.warning(message)
                with pending_lock:
                    if fatal_rest_error[0] is None:
                        fatal_rest_error[0] = VoiceUnavailable(message)
                cleanup_required = True
                break
            idle_s = now - last_progress_at
            if idle_s >= progress_timeout_s:
                message = (
                    "REST TTS stalled waiting on chunks: no client-written "
                    f"audio progress for {progress_timeout_s:.2f}s "
                    f"({last_emitted_count}/{chunk_index_counter[0]} chunks)"
                )
                log.warning(message)
                with pending_lock:
                    if fatal_rest_error[0] is None:
                        fatal_rest_error[0] = VoiceUnavailable(message)
                cleanup_required = True
                break
            time.sleep(0.05)
        return cleanup_required

    def _raise_primary_after_speaker_cleanup(
        primary: Exception,
        *,
        cause: Exception | None = None,
    ) -> None:
        """Retire speaker ID without allowing its failure to mask ``primary``."""

        try:
            _finish_speaker_id()
        except Exception as speaker_cleanup_exc:  # noqa: BLE001 - preserve primary
            primary.add_note(
                "Speaker cleanup also failed while preserving primary voice error: "
                f"{type(speaker_cleanup_exc).__name__}: {speaker_cleanup_exc}"
            )
            log.error(
                "Speaker cleanup failed while preserving primary voice error: %s: %s",
                type(speaker_cleanup_exc).__name__,
                speaker_cleanup_exc,
                exc_info=True,
            )
            if cause is None:
                cause = speaker_cleanup_exc
        if cause is not None:
            raise primary from cause
        raise primary

    try:
        cleanup_required = _finish_rest_after_chat()
    except Exception as exc:
        cleanup_failure: Exception | None = None
        try:
            _cleanup_rest_turn_bounded(cancel_pending=True)
        except Exception as cleanup_exc:
            exc.add_note(
                "REST turn cleanup also failed: "
                f"{type(cleanup_exc).__name__}: {cleanup_exc}"
            )
            log.error(
                "REST turn cleanup failed while preserving primary exception: %s: %s",
                type(cleanup_exc).__name__,
                cleanup_exc,
                exc_info=True,
            )
            cleanup_failure = cleanup_exc
        _raise_primary_after_speaker_cleanup(exc, cause=cleanup_failure)
    cleanup_failure = None
    try:
        _cleanup_rest_turn_bounded(cancel_pending=cleanup_required)
    except Exception as cleanup_exc:
        primary = fatal_rest_error[0]
        if primary is None:
            raise
        primary.add_note(
            "REST turn cleanup also failed after terminal TTS error: "
            f"{type(cleanup_exc).__name__}: {cleanup_exc}"
        )
        log.error(
            "REST turn cleanup failed while preserving terminal TTS error: %s: %s",
            type(cleanup_exc).__name__,
            cleanup_exc,
            exc_info=True,
        )
        cleanup_failure = cleanup_exc

    if fatal_rest_error[0] is not None:
        _raise_primary_after_speaker_cleanup(
            fatal_rest_error[0],
            cause=cleanup_failure,
        )

    # Async full-QA of what actually played (opt-in): transcribe the reply
    # chunks back and judge them in the background, logging any garbles. Never
    # blocks the turn.
    if _runtime_qa_enabled() and qa_samples:
        threading.Thread(
            target=_async_verify_reply, args=(list(qa_samples), runner.hivemind_url),
            daemon=True, name="ms4-reply-qa",
        ).start()

    reply_text = (chat_response.get("text") or "").strip() if isinstance(chat_response, dict) else ""
    # Contract release replaces the returned reply so the UI/return value match
    # what was actually spoken (never the invalid model text).
    if marker_gate.active and marker_gate.resolved:
        reply_text = marker_gate.released_text.strip()
    if turn_cancel.is_set() or (client_alive is not None and not client_alive.is_set()):
        _finish_speaker_id()
        raise VoiceUnavailable("voice turn cancelled before generated audio delivery completed")
    scheduled_speakable_output = chunk_index_counter[0] > 0
    speakable_reply_expected = scheduled_speakable_output or bool(_sanitize(reply_text).strip())
    if speakable_reply_expected and counters["audio_client_written"] == 0:
        _finish_speaker_id()
        raise VoiceUnavailable(
            "voice output unavailable: speakable reply produced zero client-written audio"
        )
    _finish_speaker_id()
    total_ms = int((time.monotonic() - t_total) * 1000)
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
        "turn_id": turn_correlation_id,
        "transcript": transcript,
        "raw_transcript": raw_transcript,
        "reply_text": reply_text,
        "session_id": chat_response.get("session_id") if isinstance(chat_response, dict) else None,
        "foreground_model": chat_response.get("face_lobe_model") if isinstance(chat_response, dict) else None,
        "router": chat_response.get("router") if isinstance(chat_response, dict) else None,
        "dispatched_job": chat_response.get("dispatched_job") if isinstance(chat_response, dict) else None,
        "revision_id": chat_response.get("revision_id") if isinstance(chat_response, dict) else None,
        "grounding_source": chat_response.get("grounding_source") if isinstance(chat_response, dict) else None,
        "transcription_model": asr_result.get("model"),
        "tts_model": tts_model or DEFAULT_TTS_MODEL,
        "metrics": {
            "schema": "Ms4VoiceStreamMetrics.v1",
            "engine": "rest",
            "asr_ms": asr_ms,
            "chat_ms": chat_ms,
            "total_ms": total_ms,
            "first_text_token_ms": t_first_token["ms"],
            "first_audio_chunk_ms": counters["first_audio_at"],
            "audio_chunks": counters["audio_client_written"],
            "audio_generated_chunks": counters["audio_generated"],
            "audio_gateway_enqueued": counters["audio_gateway_enqueued"],
            "audio_client_written": counters["audio_client_written"],
            "audio_errors": counters["audio_errors"],
            "runtime_gated": counters["gated"],
            "chunks": chunk_timings,
            "tts_parallelism": parallelism,
            "chat_metrics": chat_metrics,
            "first_chunk_deadline_ms": int(chunk_controller.deadline_s * 1000),
            "first_chunk_deadline_fired": chunk_controller.deadline_fired,
            "first_chunk_deadline_flushes": chunk_controller.deadline_flushes,
            "marker_contract": marker_gate.result(),
            "lifecycle": dict(lifecycle),
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
    response_format: str | None,
    chat_call: Callable[..., dict[str, Any]],
    emit: Callable[[str, dict[str, Any]], bool],
    t_total: float,
    start_speaker_id: Callable[[], None],
    finish_speaker_id: Callable[[], None],
    ws_engine_factory: Callable[..., Any] | None = None,
    client_alive: threading.Event | None = None,
    cancel_event: threading.Event | None = None,
    client_id: str | None = None,
    turn_id: str | None = None,
    lifecycle: dict[str, int] | None = None,
) -> dict[str, Any]:
    """The ``engine="ws_super"`` branch of voice_ptt_turn_stream.

    Open one WebSocket to HiveMind's TTS_SUPER ``stream-input``, schedule
    speech-safe chunks as soon as the first clause/word threshold is met,
    then flush at end of chat and wait for ``isFinal``. The WS server emits
    audio chunks as it synthesizes; the engine emits them as SSE ``audio_chunk`` events
    so the existing browser path consumes them unchanged.

    This is the default streaming engine. Open failure and zero-audio
    failure still fall back to the sentence-chunked REST path.
    """
    emit("status", {"phase": "thinking", "engine": "ws_super"})
    turn_cancel = cancel_event or threading.Event()
    lifecycle = lifecycle if lifecycle is not None else {}
    turn_id = turn_id or _new_voice_turn_id()
    lifecycle.setdefault("ws_sender_threads_started", 0)
    lifecycle.setdefault("ws_sender_threads_joined", 0)
    lifecycle.setdefault("ws_sender_thread_join_timeouts", 0)
    using_builtin_engine = ws_engine_factory is None
    ws_audio_lock = threading.Lock()
    ws_generated_audio = [0]
    ws_gateway_enqueued_audio = [0]
    ws_client_written_audio = [0]
    ws_first_audio_ms: list[int | None] = [None]
    ws_consumer_open = threading.Event()
    ws_consumer_open.set()
    # Marker-confirmation contract for this turn (inactive for ordinary turns).
    # Same semantic boundary as the REST path.
    marker_gate = _MarkerConfirmationGate(transcript)

    def _emit_ws_engine(event: str, payload: dict[str, Any]) -> bool:
        if event == "audio_chunk":
            payload = dict(payload)
            chunk_index = payload.get("index", ws_generated_audio[0])
            payload.setdefault("turn_id", turn_id)
            payload.setdefault("chunk_id", f"{turn_id}-c{chunk_index}")
            with ws_audio_lock:
                ws_generated_audio[0] += 1
                ws_gateway_enqueued_audio[0] += 1
        emitted = emit(event, payload)
        if emitted is False:
            ws_consumer_open.clear()
            turn_cancel.set()
            if client_alive is not None:
                client_alive.clear()
            return False
        if event == "audio_chunk":
            with ws_audio_lock:
                ws_client_written_audio[0] += 1
                if ws_first_audio_ms[0] is None:
                    ws_first_audio_ms[0] = int((time.monotonic() - t_total) * 1000)
            start_speaker_id()
        return emitted

    engine = None
    cancel_pending_send: Callable[[], None] | None = None
    try:
        # Local import so the REST path never has to install websockets.
        if ws_engine_factory is None:
            from .tts_super_ws import TtsSuperWsEngine
            ws_engine_factory = TtsSuperWsEngine
        engine = ws_engine_factory(
            hivemind_url=runner.hivemind_url,
            voice=tts_voice,
            emit=_emit_ws_engine,
            t_start=t_total,
        )
        engine.open()
        if using_builtin_engine:
            def _cancel_builtin_send() -> None:
                connection = getattr(engine, "_ws", None)
                transport = getattr(connection, "socket", None)
                if transport is not None:
                    try:
                        transport.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    try:
                        transport.close()
                    except OSError:
                        pass
                else:
                    engine.close()

            cancel_pending_send = _cancel_builtin_send
        else:
            candidate = getattr(engine, "cancel_pending_send", None)
            if callable(candidate):
                cancel_pending_send = candidate
            else:
                raise RuntimeError("WS engine does not expose bounded send cancellation")
    except Exception as exc:
        # Don't emit ``error`` here — that's the SSE terminal-failure
        # event the UI shows as a red banner. The caller will fall
        # back to the REST engine and the user will still get audio.
        # Emit a status note instead so anyone watching the SSE feed
        # sees the engine switch.
        if engine is not None:
            try:
                engine.close()
            except Exception:  # noqa: BLE001
                pass
        log.warning("TTS_SUPER WS setup failed: %s", exc)
        emit("status", {
            "phase": "tts_engine_open_failed",
            "engine": "ws_super",
            "error": str(exc)[:200],
        })
        raise WsOpenFailed(f"failed to set up TTS_SUPER WS: {exc}") from exc
    assert engine is not None
    assert cancel_pending_send is not None
    chat_response: dict[str, Any] = {}
    chat_metrics: dict[str, Any] = {}
    t_first_token: dict[str, float | None] = {"ms": None}
    t_chat_start = time.monotonic()

    # Chunk before sanitizing so a long sentence cannot withhold the first
    # speakable fragment. One ordered sender keeps a blocking WS send from
    # stalling the Face Lobe's token-consumption thread.
    from .spoken_text_filter import SpokenTextFilter, sanitize_for_speech as _sanitize

    ws_chunker = SentenceChunker()
    tts_filter = SpokenTextFilter() if TTS_FILTER_ENABLED else None
    ws_chunk_schedule: list[dict[str, Any]] = []
    streamed_reply_parts: list[str] = []
    ws_queue_max_items = _voice_ws_queue_max_items()
    ws_queue_max_bytes = _voice_ws_queue_max_bytes()
    ws_queue_put_timeout = _voice_ws_queue_put_timeout()
    send_queue: queue.Queue[tuple[str, int]] = queue.Queue(maxsize=ws_queue_max_items)
    send_cancelled = threading.Event()
    send_producer_done = threading.Event()
    send_errors: list[BaseException] = []
    send_capacity = threading.Condition()
    send_outstanding_items = [0]
    send_outstanding_bytes = [0]
    send_peak_items = [0]
    send_peak_bytes = [0]
    send_backpressure_failures = [0]

    def _release_send_capacity(text_bytes: int) -> None:
        with send_capacity:
            send_outstanding_items[0] = max(0, send_outstanding_items[0] - 1)
            send_outstanding_bytes[0] = max(0, send_outstanding_bytes[0] - text_bytes)
            send_capacity.notify_all()

    def _reserve_send_capacity(text_bytes: int) -> bool:
        if text_bytes > ws_queue_max_bytes:
            send_backpressure_failures[0] += 1
            send_errors.append(VoiceUnavailable(
                "WS text queue capacity exceeded: one fragment is larger than the byte limit"
            ))
            return False
        deadline = time.monotonic() + ws_queue_put_timeout
        with send_capacity:
            while (
                send_outstanding_items[0] >= ws_queue_max_items
                or send_outstanding_bytes[0] + text_bytes > ws_queue_max_bytes
            ):
                if (
                    send_cancelled.is_set()
                    or turn_cancel.is_set()
                    or not ws_consumer_open.is_set()
                    or (client_alive is not None and not client_alive.is_set())
                ):
                    return False
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    send_backpressure_failures[0] += 1
                    send_errors.append(VoiceUnavailable(
                        "WS text queue capacity exceeded while transport was blocked"
                    ))
                    return False
                send_capacity.wait(timeout=min(0.02, remaining))
            send_outstanding_items[0] += 1
            send_outstanding_bytes[0] += text_bytes
            send_peak_items[0] = max(send_peak_items[0], send_outstanding_items[0])
            send_peak_bytes[0] = max(send_peak_bytes[0], send_outstanding_bytes[0])
            return True

    def _send_loop() -> None:
        try:
            while True:
                if send_cancelled.is_set():
                    return
                try:
                    text, text_bytes = send_queue.get(timeout=0.02)
                except queue.Empty:
                    if send_producer_done.is_set():
                        return
                    continue
                try:
                    if send_cancelled.is_set():
                        return
                    engine.push(text)
                finally:
                    _release_send_capacity(text_bytes)
                    send_queue.task_done()
        except BaseException as exc:  # surfaced on the turn thread
            send_errors.append(exc)

    sender = threading.Thread(target=_send_loop, name="ms4-tts-ws-send", daemon=True)
    sender.start()
    lifecycle["ws_sender_threads_started"] += 1
    sender_join_recorded = [False]

    def _record_sender_join() -> None:
        if not sender_join_recorded[0] and not sender.is_alive():
            sender_join_recorded[0] = True
            lifecycle["ws_sender_threads_joined"] += 1

    def _stop_sender() -> None:
        send_producer_done.set()

    transport_cancelled = [False]

    def _cancel_ws_transport(reason: str) -> None:
        if transport_cancelled[0]:
            return
        transport_cancelled[0] = True
        try:
            cancel_pending_send()
        except Exception as exc:  # noqa: BLE001
            log.warning("ws_super transport cancellation failed (%s): %s", reason, exc)

    def _cancel_and_join_sender(reason: str) -> None:
        send_cancelled.set()
        _stop_sender()
        _cancel_ws_transport(reason)
        sender.join(timeout=_voice_ws_sender_cleanup_timeout())
        if sender.is_alive():
            lifecycle["ws_sender_thread_join_timeouts"] += 1
            raise RuntimeError(
                "WS sender cancellation contract failed; sender is still alive "
                f"after {reason}"
            )
        _record_sender_join()
        while True:
            try:
                _text, text_bytes = send_queue.get_nowait()
            except queue.Empty:
                break
            _release_send_capacity(text_bytes)
            send_queue.task_done()

    def _wait_for_sender() -> bool:
        _stop_sender()
        deadline = time.monotonic() + _voice_ws_send_timeout()
        while sender.is_alive():
            if (
                not ws_consumer_open.is_set()
                or turn_cancel.is_set()
                or (client_alive is not None and not client_alive.is_set())
            ):
                _cancel_and_join_sender("client disconnect")
                return False
            if time.monotonic() >= deadline:
                _cancel_and_join_sender("send timeout")
                raise TimeoutError("timed out draining queued TTS_SUPER text")
            sender.join(timeout=0.01)
        sender.join()
        _record_sender_join()
        if send_errors:
            raise send_errors[0]
        return True

    def _schedule_ws_chunk(text: str) -> bool:
        if (
            turn_cancel.is_set()
            or
            not ws_consumer_open.is_set()
            or (client_alive is not None and not client_alive.is_set())
        ):
            return False
        if tts_filter is None:
            speakable = text.strip()
        else:
            ready_parts = [tts_filter.push(text)]
            # SentenceChunker has declared this fragment ready. Drain any
            # non-code tail now instead of waiting for a full sentence, while
            # retaining SpokenTextFilter's state across fenced code blocks.
            if not tts_filter.in_code_block and tts_filter.buffer:
                ready_parts.append(_sanitize(tts_filter.buffer))
                tts_filter.buffer = ""
            speakable = " ".join(part.strip() for part in ready_parts if part.strip())
        if not speakable:
            return True
        idx = len(ws_chunk_schedule)
        scheduled_ms = int((time.monotonic() - t_total) * 1000)
        if emit("chunk_scheduled", {
            "index": idx,
            "text": speakable,
            "scheduled_ms": scheduled_ms,
            "engine": "ws_super",
        }) is False:
            ws_consumer_open.clear()
            turn_cancel.set()
            if client_alive is not None:
                client_alive.clear()
            return False
        ws_chunk_schedule.append({
            "index": idx,
            "text_len": len(speakable),
            "scheduled_ms": scheduled_ms,
        })
        text_bytes = len(speakable.encode("utf-8"))
        if not _reserve_send_capacity(text_bytes):
            ws_consumer_open.clear()
            turn_cancel.set()
            send_cancelled.set()
            if client_alive is not None:
                client_alive.clear()
            _stop_sender()
            _cancel_ws_transport("text queue capacity")
            return False
        try:
            send_queue.put_nowait((speakable, text_bytes))
        except queue.Full:
            _release_send_capacity(text_bytes)
            send_backpressure_failures[0] += 1
            send_errors.append(VoiceUnavailable(
                "WS text queue capacity accounting diverged from queue capacity"
            ))
            ws_consumer_open.clear()
            turn_cancel.set()
            send_cancelled.set()
            if client_alive is not None:
                client_alive.clear()
            _stop_sender()
            _cancel_ws_transport("text queue full")
            return False
        return True

    ws_chunk_controller = _FirstChunkDeadlineController(
        chunker=ws_chunker,
        schedule=_schedule_ws_chunk,
        active=lambda: (
            ws_consumer_open.is_set()
            and not turn_cancel.is_set()
            and (client_alive is None or client_alive.is_set())
        ),
    )

    def _stream_text_ws(text: str) -> bool:
        """Emit user-visible ``text_delta`` for ``text`` and feed it to the WS
        sentence chunker (TTS_SUPER scheduling). Shared by ordinary streamed
        deltas and the marker-contract release below."""
        if emit("text_delta", {"text": text}) is False:
            ws_consumer_open.clear()
            turn_cancel.set()
            if client_alive is not None:
                client_alive.clear()
            return False
        return ws_chunk_controller.add(text)

    def _on_delta(delta: str) -> bool:
        if t_first_token["ms"] is None and delta.strip():
            t_first_token["ms"] = int((time.monotonic() - t_total) * 1000)
        # Marker-confirmation contract (same boundary as the REST path): buffer
        # model deltas when an explicit marker obligation is present so the
        # invalid reply never reaches text_delta / chunk_scheduled / the WS
        # TTS_SUPER stream OR the zero-audio REST fallback (which synthesizes
        # reply_text — overridden to the release text below). Buffered deltas
        # are intentionally NOT appended to streamed_reply_parts. Ordinary turns
        # stream immediately, unchanged.
        if marker_gate.active:
            marker_gate.buffer_delta(delta)
            return True
        streamed_reply_parts.append(delta)
        return _stream_text_ws(delta)

    chat_response = None
    chat_metrics: dict[str, Any] = {}
    try:
        _chat_kw: dict[str, Any] = {"session_id": session_id, "model": model}
        if depth_model:
            _chat_kw["depth_model"] = depth_model
        if _voice_brevity_enabled():
            _chat_kw["voice_mode"] = True
        if client_id:
            _chat_kw["client_id"] = client_id
        if _accepts_keyword(chat_call, "turn_id"):
            _chat_kw["turn_id"] = turn_id
        # First-token watchdog (same as the REST path): a stalled Face
        # model raises FaceLobeStalled rather than hanging the WS turn.
        chat_response = _run_facechat_guarded(
            chat_call=chat_call,
            transcript=transcript,
            chat_kwargs=_chat_kw,
            on_delta=_on_delta,
            first_token_timeout=_voice_first_token_timeout(),
            model=model,
            cancel_event=turn_cancel,
            client_alive=client_alive,
        )
        chat_metrics = (chat_response or {}).get("metrics") or {}
        chat_ms = int((time.monotonic() - t_chat_start) * 1000)
    except BaseException as chat_exc:
        # Close the WS engine and fail the turn over to the error reflex.
        # Do NOT fall back to REST — the same model would stall there too.
        try:
            _cancel_and_join_sender("chat failure")
        finally:
            lifecycle["first_chunk_timer_join_timeouts"] = (
                lifecycle.get("first_chunk_timer_join_timeouts", 0)
                + int(ws_chunk_controller.close())
            )
            try:
                engine.close()
            finally:
                finish_speaker_id()
        if send_errors and isinstance(send_errors[0], VoiceUnavailable):
            raise send_errors[0] from chat_exc
        raise
    # Marker-confirmation contract release (WS): stream the release text through
    # the same text_delta + WS chunk path so it — and ONLY it — is scheduled for
    # TTS_SUPER. Buffered invalid deltas were never pushed to the WS engine.
    if marker_gate.active:
        _model_reply = (chat_response or {}).get("text") if isinstance(chat_response, dict) else ""
        _release_text = marker_gate.resolve(_model_reply or "")
        if _release_text and ws_consumer_open.is_set():
            _stream_text_ws(_release_text)
    tail = ws_chunk_controller.flush()
    if tail and ws_consumer_open.is_set():
        _schedule_ws_chunk(tail)
    lifecycle["first_chunk_timer_join_timeouts"] = (
        lifecycle.get("first_chunk_timer_join_timeouts", 0)
        + int(ws_chunk_controller.close())
    )
    if tts_filter is not None:
        # Usually empty because each scheduled fragment is drained above.
        # An unterminated fenced code block intentionally flushes to nothing.
        filter_tail = tts_filter.flush().strip()
        if filter_tail and ws_consumer_open.is_set():
            _schedule_ws_chunk(filter_tail)
    ws_failure: BaseException | None = None
    sender_drained = False
    try:
        sender_drained = _wait_for_sender()
        if sender_drained:
            engine.flush()
    except BaseException as exc:  # recover only when no audio was accepted
        ws_failure = exc
        if sender.is_alive():
            _cancel_and_join_sender("post-chat WS failure")

    # Barge-in: if the client connection died while chat was running,
    # don't bother waiting for HiveMind to finish synthesizing audio
    # nobody is going to hear. Close the WS immediately and bail.
    consumer_gone = (
        not ws_consumer_open.is_set()
        or turn_cancel.is_set()
        or (client_alive is not None and not client_alive.is_set())
    )
    if consumer_gone or ws_failure is not None or not sender_drained:
        log.info("ws_super engine: closing without wait_for_final (consumer_gone=%s, failure=%s)",
                 consumer_gone, ws_failure)
        _cancel_ws_transport("consumer gone or WS failure")
        engine.close()
    else:
        # Wait for isFinal with TWO budgets:
        #
        #   * first-audio budget: if no audio chunk has arrived within
        #     ``MS4_TTS_WS_FIRST_AUDIO_TIMEOUT`` (default 10s) AFTER
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
        # in. With the early bail, that becomes a bounded fallback
        # because REST takes over the moment we know WS is dead.
        def _wait_for_final_safely(timeout: float) -> bool:
            nonlocal ws_failure
            try:
                return bool(engine.wait_for_final(timeout=timeout))
            except BaseException as exc:
                ws_failure = exc
                return True

        first_audio_timeout = _voice_ws_first_audio_timeout()
        overall_timeout = float(os.environ.get("MS4_TTS_WS_FINAL_TIMEOUT", "60"))
        poll_interval = 0.5
        t_wait_start = time.monotonic()
        # Absolute monotonic deadlines, fixed ONCE from t_wait_start. Each poll
        # is clamped to the time remaining, so the 0.5s cadence can neither
        # extend the deadline (a late-returning poll) nor shorten it (an early
        # poll): the bail fires AT the deadline. A true no-audio stall therefore
        # cancels exactly once and hands off to the REST fallback exactly once.
        first_audio_deadline = t_wait_start + first_audio_timeout
        overall_deadline = t_wait_start + overall_timeout
        cancel_reason: str | None = None
        while True:
            now = time.monotonic()
            # Finding 2: honor the turn cancellation fence while waiting on WS
            # synthesis. A barge/disconnect must tear down the WS transport now
            # instead of polling wait_for_final for the full budget.
            if turn_cancel.is_set() or (client_alive is not None and not client_alive.is_set()):
                cancel_reason = "turn cancelled"
                break
            with ws_audio_lock:
                chunks_so_far = ws_client_written_audio[0]
            if chunks_so_far == 0:
                # No audio yet: bound the wait by the absolute first-audio deadline.
                remaining = first_audio_deadline - now
                if remaining <= 0:
                    log.warning(
                        "ws_super engine: bailing at the first-audio deadline "
                        "(%.1fs) with 0 audio chunks; REST fallback will take over",
                        first_audio_timeout,
                    )
                    cancel_reason = "first-audio timeout"
                    break
            else:
                remaining = overall_deadline - now
                if remaining <= 0:
                    log.warning(
                        "ws_super engine: wait_for_final hard timeout after %.1fs "
                        "(%d chunks emitted)",
                        overall_timeout, chunks_so_far,
                    )
                    cancel_reason = "final-audio timeout"
                    break
            # Clamp the poll so it can never overshoot the deadline checked above.
            wait_budget = poll_interval if remaining > poll_interval else remaining
            if _wait_for_final_safely(timeout=wait_budget):
                if ws_failure is not None:
                    cancel_reason = "wait_for_final failure"
                break  # isFinal arrived — synthesis complete
            try:
                cur_metrics = engine.metrics()
            except BaseException as exc:
                ws_failure = exc
                break
            with ws_audio_lock:
                chunks_so_far = ws_client_written_audio[0]
            # Also bail if the WS reader has reported a fatal error
            # (keepalive timeout, internal error) — no point waiting
            # on a connection that's already torn down.
            if cur_metrics.get("error") and chunks_so_far == 0:
                log.warning(
                    "ws_super engine: bailing — reader reported error=%s with 0 audio chunks",
                    str(cur_metrics.get("error"))[:120],
                )
                cancel_reason = "WS reader error"
                break
        # Graceful-first teardown for NORMAL timeouts and a zero-audio reader
        # error ("first-audio timeout", "final-audio timeout", "WS reader
        # error"): route straight through engine.close()'s bounded graceful
        # RFC6455 close (Close frame, then raw socket shutdown fallback) so the
        # peer observes a normal close instead of an abrupt EOF / protocol error.
        # Only barge / client-disconnect ("turn cancelled") and a broken
        # wait_for_final are emergencies that need the raw _cancel_ws_transport
        # pre-abort first to unblock a possibly-stuck send before closing.
        if cancel_reason in ("turn cancelled", "wait_for_final failure"):
            _cancel_ws_transport(cancel_reason)
        engine.close()

    try:
        eng_metrics = engine.metrics()
    except Exception as exc:  # noqa: BLE001
        eng_metrics = {"engine": "ws_super", "error": f"metrics failed: {exc}"}
        ws_failure = ws_failure or exc
    lifecycle["ws_reader_threads_started"] = int(
        bool(eng_metrics.get("reader_started"))
    )
    lifecycle["ws_reader_threads_joined"] = int(
        bool(eng_metrics.get("reader_joined"))
    )
    lifecycle["ws_reader_thread_join_timeouts"] = int(
        bool(eng_metrics.get("reader_join_timeout"))
    )
    reply_text = (chat_response.get("text") or "").strip() if isinstance(chat_response, dict) else ""
    if not reply_text:
        reply_text = "".join(streamed_reply_parts).strip()
    # Contract release replaces the returned/fallback reply so the WS zero-audio
    # REST fallback (which synthesizes reply_text) and the return value both
    # reflect the released text, never the invalid model reply.
    if marker_gate.active and marker_gate.resolved:
        reply_text = marker_gate.released_text.strip()

    # WS-emitted-zero-audio fallback (May 26 2026): on long replies
    # the WS engine sometimes finishes with chunks_emitted=0 and a
    # populated error (observed live: 613 chat chunks streamed, 0
    # audio chunks emitted, 1 error). If that happens AND we still
    # have reply_text AND the client is alive, fall back to chunked,
    # parallel REST /v1/audio/speech calls so the user hears
    # *something* instead of silence + a red banner.
    with ws_audio_lock:
        chunks_emitted = ws_client_written_audio[0]
    terminal_ws_failure: BaseException | None = ws_failure
    if terminal_ws_failure is None and eng_metrics.get("error"):
        terminal_ws_failure = RuntimeError(str(eng_metrics["error"]))
    if terminal_ws_failure is not None and chunks_emitted > 0 and not consumer_gone:
        finish_speaker_id()
        raise terminal_ws_failure
    rest_fallback_used = False
    fallback_metrics = {
        "first_audio_ms": None,
        "audio_errors": 0,
        "audio_generated": 0,
        "audio_gateway_enqueued": 0,
        "audio_client_written": 0,
    }

    def _emit_rest_fallback(event: str, payload: dict[str, Any]) -> bool:
        if event == "audio_error":
            fallback_metrics["audio_errors"] += 1
        if event == "audio_chunk":
            payload = dict(payload)
            chunk_index = payload.get("index", fallback_metrics["audio_generated"])
            payload.setdefault("turn_id", turn_id)
            payload.setdefault("chunk_id", f"{turn_id}-fallback-c{chunk_index}")
            fallback_metrics["audio_generated"] += 1
            fallback_metrics["audio_gateway_enqueued"] += 1
        emitted = emit(event, payload)
        if emitted is False:
            ws_consumer_open.clear()
            turn_cancel.set()
            if client_alive is not None:
                client_alive.clear()
            return False
        if event == "audio_chunk" and emitted is not False:
            fallback_metrics["audio_client_written"] += 1
            if fallback_metrics["first_audio_ms"] is None:
                fallback_metrics["first_audio_ms"] = int((time.monotonic() - t_total) * 1000)
            start_speaker_id()
        return emitted

    if (
        chunks_emitted == 0
        and reply_text
        and ws_consumer_open.is_set()
        and (client_alive is None or client_alive.is_set())
    ):
        rest_fallback_used = True
        log.warning(
            "ws_super engine emitted 0 audio chunks (error=%s); "
            "falling back to chunked REST TTS on %d chars of reply text",
            eng_metrics.get("error"),
            len(reply_text),
        )
        fallback_status_accepted = emit("status", {
            "phase": "tts_engine_fallback_after_zero_audio",
            "from": "ws_super",
            "to": "rest",
            "ws_error": str(terminal_ws_failure or "0 chunks emitted")[:200],
        })
        if fallback_status_accepted is False:
            ws_consumer_open.clear()
            if client_alive is not None:
                client_alive.clear()
        # Prime the first REST sentence, then fan out the tail. This avoids
        # head-of-line blocking when every request lands on one TTS worker.
        if reply_text.strip() and ws_consumer_open.is_set():
            try:
                chunks_emitted = _emit_text_as_parallel_chunks(
                    text=reply_text, emit=_emit_rest_fallback, hivemind_url=runner.hivemind_url,
                    tts_model=tts_model, tts_voice=tts_voice,
                    response_format=response_format or DEFAULT_TTS_FORMAT,
                    cancel_event=turn_cancel,
                    client_alive=client_alive,
                    lifecycle=lifecycle,
                )
            except Exception as exc:
                log.error("REST chunked fallback failed: %s", exc)
                _emit_rest_fallback("audio_error", {"index": 0, "error": str(exc)[:200]})

    if turn_cancel.is_set() or (client_alive is not None and not client_alive.is_set()):
        finish_speaker_id()
        if send_errors and isinstance(send_errors[0], VoiceUnavailable):
            raise send_errors[0]
        raise VoiceUnavailable("voice turn cancelled before generated audio delivery completed")
    from .spoken_text_filter import sanitize_for_speech as _sanitize_reply
    if _sanitize_reply(reply_text).strip() and chunks_emitted == 0:
        finish_speaker_id()
        raise VoiceUnavailable(
            "voice output unavailable: WS and REST produced zero client-written audio"
        )
    finish_speaker_id()
    total_ms = int((time.monotonic() - t_total) * 1000)
    return {
        "turn_id": turn_id,
        "transcript": transcript,
        "reply_text": reply_text,
        "session_id": chat_response.get("session_id") if isinstance(chat_response, dict) else None,
        "foreground_model": chat_response.get("face_lobe_model") if isinstance(chat_response, dict) else None,
        "router": chat_response.get("router") if isinstance(chat_response, dict) else None,
        "dispatched_job": chat_response.get("dispatched_job") if isinstance(chat_response, dict) else None,
        "revision_id": chat_response.get("revision_id") if isinstance(chat_response, dict) else None,
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
            "first_audio_chunk_ms": (
                ws_first_audio_ms[0]
                if ws_first_audio_ms[0] is not None
                else fallback_metrics["first_audio_ms"]
            ),
            "audio_chunks": chunks_emitted,
            "audio_generated_chunks": (
                ws_generated_audio[0] + fallback_metrics["audio_generated"]
            ),
            "audio_gateway_enqueued": (
                ws_gateway_enqueued_audio[0]
                + fallback_metrics["audio_gateway_enqueued"]
            ),
            "audio_client_written": chunks_emitted,
            "audio_errors": (
                (1 if terminal_ws_failure is not None else 0)
                + fallback_metrics["audio_errors"]
            ),
            "rest_fallback_used": rest_fallback_used,
            "first_chunk_scheduled_ms": (
                ws_chunk_schedule[0]["scheduled_ms"] if ws_chunk_schedule else None
            ),
            "chunks_scheduled": len(ws_chunk_schedule),
            "chunk_schedule": ws_chunk_schedule,
            "ws_text_queue": {
                "max_items": ws_queue_max_items,
                "max_bytes": ws_queue_max_bytes,
                "put_timeout_ms": int(ws_queue_put_timeout * 1000),
                "peak_outstanding_items": send_peak_items[0],
                "peak_outstanding_bytes": send_peak_bytes[0],
                "backpressure_failures": send_backpressure_failures[0],
            },
            "chunks": [],   # WS engine doesn't expose per-chunk text grouping
            "tts_parallelism": {  # not applicable — single stream
                "engine": "ws_super",
                "speedup_ratio": None,
                "any_held_for_inorder": False,
                "max_held_for_inorder_ms": 0,
            },
            "tts_engine": eng_metrics,
            "chat_metrics": chat_metrics,
            "first_chunk_deadline_ms": int(ws_chunk_controller.deadline_s * 1000),
            "first_chunk_deadline_fired": ws_chunk_controller.deadline_fired,
            "first_chunk_deadline_flushes": ws_chunk_controller.deadline_flushes,
            "marker_contract": marker_gate.result(),
            "lifecycle": dict(lifecycle),
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
    provision_status: dict[str, Any] | None = None
    if os.environ.get("MS4_VOICE_ASR_AUTOPROVISION", "1").strip().lower() in {"1", "true", "yes", "on"}:
        try:
            from .voice_admin import get_provision_status, poll_until_healthy, request_voice_service

            provision_status = get_provision_status(hivemind_url, "ASR", timeout=min(timeout, 10))
            if not provision_status.get("healthy"):
                state = str(provision_status.get("provisioning_state") or "")
                if state != "provisioning":
                    request_voice_service(hivemind_url, "ASR", model=model, timeout=min(timeout, 30))
                try:
                    wait_secs = float(os.environ.get("MS4_VOICE_ASR_AUTOPROVISION_TIMEOUT_S", "120"))
                except (TypeError, ValueError):
                    wait_secs = 120.0
                provision_status = poll_until_healthy(
                    hivemind_url,
                    "ASR",
                    timeout_secs=max(1.0, wait_secs),
                    poll_interval_secs=1.5,
                )
        except Exception as exc:
            log.warning("ASR auto-provision failed before pre-warm (voice remains fail-closed): %s", exc)
            provision_status = {"healthy": False, "error": str(exc), "autoprovision": "failed"}
    try:
        result = transcribe(
            hivemind_url=hivemind_url,
            audio=build_silent_wav(),
            filename="prewarm.wav",
            model=model,
            timeout=timeout,
        )
        if provision_status is not None:
            result["provision_status"] = provision_status
        return result
    except Exception as exc:
        log.warning("ASR pre-warm failed (safe to ignore on cold cluster): %s", exc)
        payload = {"text": "", "error": str(exc), "warmed": False}
        if provision_status is not None:
            payload["provision_status"] = provision_status
        return payload


def prewarm_face_lobe_model(
    *,
    hivemind_url: str,
    model: str | None = None,
    timeout: int = 60,
    exact_model: bool = False,
    cancel_event: threading.Event | None = None,
    latency_budget_ms: int | None = None,
    face_chat: Any | None = None,
    lease_scope_id: str | None = None,
) -> dict[str, Any]:
    """Warm Face residency, then admit an exact selection on production TTFT.

    Automatic boot warm keeps the historical lightweight behavior.  An exact
    operator-selected model receives two distinct stages: the same residency
    warm, followed by a cache-busted production-sized request that streams one
    visible token.  Selection is ready only when both stages use the requested
    model and that token arrives inside the explicit latency budget.
    """
    admission_budget = (
        selected_face_admission_latency_budget_ms()
        if latency_budget_ms is None
        else int(latency_budget_ms)
    )
    admission_budget = max(1, admission_budget)

    def admission_fields(
        *,
        first_token_ms: int | None = None,
        admitted: bool = False,
    ) -> dict[str, Any]:
        return {
            "admission_profile": SELECTED_FACE_ADMISSION_PROFILE,
            "first_token_ms": first_token_ms,
            "latency_budget_ms": admission_budget,
            "latency_admitted": admitted,
        }

    try:
        from .face_lobe_chat import FaceLobeChat, FaceLobeChatError
        chat = face_chat or FaceLobeChat(
            hivemind_url=hivemind_url,
            http_timeout=timeout,
            allow_model_fallback=not exact_model,
        )
        from machine_spirit_4.double_agent.model_picker import choose_foreground_model
        if model is None:
            choice = choose_foreground_model(hivemind_url=hivemind_url, force_refresh=True)
            model = choice.model_id
        recommend_receipt: dict[str, Any] | None = None
        if exact_model:
            from machine_spirit_4.double_agent.model_picker import (
                ForegroundModelUnavailable,
                require_exact_face_cluster_admission,
            )
            from machine_spirit_4.double_agent.recommend_lease import public_recommend_evidence
            try:
                recommend_receipt = require_exact_face_cluster_admission(
                    hivemind_url=hivemind_url,
                    requested_model=str(model),
                )
            except ForegroundModelUnavailable as exc:
                payload = {
                    "error": str(exc),
                    "warmed": False,
                    "model": None,
                    "requested_model": model,
                    "fallback_used": False,
                    "reply_len": 0,
                    "completed": False,
                    "cancelled": bool(
                        cancel_event is not None and cancel_event.is_set()
                    ),
                    "fail_closed": True,
                }
                payload.update(admission_fields())
                return payload
            bind_receipt = getattr(chat, "bind_recommend_receipt", None)
            if callable(bind_receipt):
                bind_receipt(lease_scope_id, recommend_receipt)
        result = chat.chat(
            "Reply with the single word: ready.",
            session_id=f"prewarm-{uuid.uuid4().hex[:8]}",
            model=model,
            # The exact selected-model path is streamed so its cancellation
            # event can stop a cold request after the endpoint's wall clock.
            stream_callback=(lambda _fragment: None) if exact_model else None,
            extra_system=None,
            cancel_event=cancel_event,
            recommend_receipt=recommend_receipt,
            recommend_validate_only=bool(exact_model),
        )
        effective_model = str(result.get("model") or "").strip()
        if not effective_model:
            raise RuntimeError("Face Lobe pre-warm response omitted its effective model")
        completed = (
            result.get("completed") is True
            if exact_model
            else result.get("completed", True) is not False
        )
        cancelled = result.get("cancelled") is True
        fallback_used = bool(result.get("fallback_used")) or effective_model != model
        reply_len = len(result.get("text") or "")
        warmed = bool(
            completed
            and not cancelled
            and reply_len > 0
            and (not exact_model or (not fallback_used and effective_model == model))
        )
        payload: dict[str, Any] = {
            "model": effective_model,
            "requested_model": model,
            "fallback_used": fallback_used,
            "reply_len": reply_len,
            "completed": completed,
            "cancelled": cancelled,
            "warmed": warmed,
        }
        if recommend_receipt:
            from machine_spirit_4.double_agent.recommend_lease import public_recommend_evidence

            payload["recommend"] = public_recommend_evidence(recommend_receipt)
        if not exact_model:
            return payload
        payload.update(admission_fields())
        if not warmed:
            payload.setdefault(
                "error",
                "selected Face residency warm did not complete exactly",
            )
            return payload

        probe_message, probe_context = _selected_face_admission_probe_payload()
        try:
            probe = dict(
                chat.probe_first_token(
                    probe_message,
                    model=model,
                    extra_system=probe_context,
                    cancel_event=cancel_event,
                    first_token_timeout=admission_budget / 1000.0,
                    max_visible_tokens=1,
                    recommend_receipt=recommend_receipt,
                    recommend_validate_only=True,
                )
                or {}
            )
        except Exception as exc:
            payload["warmed"] = False
            payload["error"] = f"selected Face production admission failed: {exc}"
            return payload

        probe_model = str(probe.get("model") or "").strip()
        raw_first_token_ms = probe.get("first_token_ms")
        first_token_ms = (
            int(raw_first_token_ms)
            if isinstance(raw_first_token_ms, (int, float))
            and not isinstance(raw_first_token_ms, bool)
            and math.isfinite(float(raw_first_token_ms))
            and raw_first_token_ms >= 0
            else None
        )
        first_token_observed = bool(
            probe.get("first_token_observed") is True
            and probe.get("visible_tokens") == 1
            and str(probe.get("text") or "").strip()
        )
        latency_admitted = bool(
            first_token_observed
            and probe.get("completed") is True
            and first_token_ms is not None
            and first_token_ms <= admission_budget
            and probe_model == model
            and not (cancel_event is not None and cancel_event.is_set())
        )
        payload.update(
            admission_fields(
                first_token_ms=first_token_ms,
                admitted=latency_admitted,
            )
        )
        payload["warmed"] = bool(warmed and latency_admitted)
        if not latency_admitted:
            if probe_model and probe_model != model:
                payload["error"] = (
                    "selected Face production admission returned a different model"
                )
            elif cancel_event is not None and cancel_event.is_set():
                payload["error"] = "selected Face production admission was superseded"
            elif first_token_ms is None or not first_token_observed:
                payload["error"] = (
                    "selected Face production admission produced no visible first token"
                )
            elif probe.get("completed") is not True:
                payload["error"] = (
                    "selected Face production admission stream ended before a verified terminal"
                )
            elif first_token_ms > admission_budget:
                payload["error"] = (
                    "selected Face production first-token latency "
                    f"{first_token_ms}ms exceeded {admission_budget}ms admission budget"
                )
            else:
                payload["error"] = (
                    "selected Face production admission did not satisfy the latency contract"
                )
        return payload
    except FaceLobeChatError as exc:
        log.warning("Face Lobe pre-warm failed closed: %s", exc)
        payload = {
            "error": str(exc),
            "warmed": False,
            "model": None,
            "requested_model": model,
            "fallback_used": False,
            "reply_len": 0,
            "completed": False,
            "cancelled": bool(cancel_event is not None and cancel_event.is_set()),
            "fail_closed": True,
        }
        if exact_model:
            payload.update(admission_fields())
        return payload
    except FaceLobeChatError as exc:
        log.warning("Face Lobe pre-warm failed closed: %s", exc)
        payload = {
            "error": str(exc),
            "warmed": False,
            "model": None,
            "requested_model": model,
            "fallback_used": False,
            "reply_len": 0,
            "completed": False,
            "cancelled": bool(cancel_event is not None and cancel_event.is_set()),
            "fail_closed": True,
        }
        if exact_model:
            payload.update(admission_fields())
        return payload
    except Exception as exc:
        log.warning("Face Lobe pre-warm failed (safe to ignore on cold cluster): %s", exc)
        payload = {
            "error": str(exc),
            "warmed": False,
            "model": None,
            "requested_model": model,
            "fallback_used": False,
            "reply_len": 0,
            "completed": False,
            "cancelled": bool(cancel_event is not None and cancel_event.is_set()),
        }
        if exact_model:
            payload.update(admission_fields())
        return payload


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
        _record_voice_rest_capacity_observation(payload)
        return payload
    except urllib.error.HTTPError as exc:
        # 404 => older gateway without the scale endpoint. Not an error
        # for us; the single replica still serves TTS.
        _record_voice_rest_capacity_observation(None)
        log.info("TTS scale-out unavailable (HTTP %s); single-replica TTS in effect", exc.code)
        return {"ok": False, "status": exc.code}
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        _record_voice_rest_capacity_observation(None)
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
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Open a TTS_SUPER WebSocket, push a single "Ready." token, and verify the
    GIM actually synthesized audio so the first real voice turn doesn't pay the
    TCP/TLS handshake + WS upgrade + HiveMind-side GIM-allocation cost.

    Truthful verdict: ``warmed`` is True ONLY when the GIM produced non-empty
    audio media AND signalled ``isFinal`` within the 10s budget with no error.
    Timeout, zero media, a missing final, or any reader error all report
    ``warmed=False`` (with a reason) — the first real turn would still pay the
    cold cost, so a warm claim would be a lie. Best-effort and non-fatal: the
    caller (startup prewarm) only logs the verdict; any cluster without
    TTS_SUPER provisioned still serves audio via the REST engine fallback.
    """
    try:
        from .tts_super_ws import TtsSuperWsEngine
    except Exception as exc:
        log.warning("TTS_SUPER WS pre-warm: import failed (%s)", exc)
        return {"warmed": False, "error": str(exc)}
    t0 = time.monotonic()
    deadline = t0 + timeout
    engine = None
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
        # ONE absolute monotonic budget across connect + final/media completion:
        # the final wait gets only the time REMAINING after connect, never a
        # fresh full budget. A warm claim also requires real synthesis (media +
        # isFinal + no error), never just "the socket opened".
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            final = False
        else:
            final = bool(engine.wait_for_final(timeout=remaining))
        metrics = engine.metrics()
        completion_s = time.monotonic() - t0  # connect + final, BEFORE teardown
    except Exception as exc:
        log.warning("TTS_SUPER WS pre-warm failed (safe to ignore): %s", exc)
        if engine is not None:
            try:
                engine.close()
            except Exception:  # noqa: BLE001
                pass
        return {"warmed": False, "error": str(exc)}
    try:
        engine.close()
    except Exception:  # noqa: BLE001 — verdict already captured from metrics
        pass
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    within_budget = completion_s <= timeout
    error = metrics.get("error")
    media_bytes = int(metrics.get("bytes_received") or 0)
    is_final = bool(metrics.get("is_final_seen"))
    # Never warm past the absolute deadline, even if media + isFinal arrived late.
    warmed = final and is_final and error is None and media_bytes > 0 and within_budget
    result: dict[str, Any] = {
        "warmed": warmed,
        "elapsed_ms": elapsed_ms,
        "bytes_received": media_bytes,
        "is_final_seen": is_final,
    }
    if not warmed:
        if error is not None:
            result["error"] = str(error)
        elif not final or not is_final:
            result["error"] = "timeout: no isFinal within budget"
        elif media_bytes <= 0:
            result["error"] = "zero audio media"
        elif not within_budget:
            result["error"] = "exceeded 10s prewarm budget"
        else:
            result["error"] = "prewarm not verified"
    return result


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
