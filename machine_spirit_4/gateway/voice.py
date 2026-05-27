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
# This is the single biggest "audio starts ASAP" lever for the REST path
# because most replies don't terminate until 15-25 words in but the
# human ear is fine with sub-sentence first chunks. Lowered through:
#   May 22 2026: 4 → 2 words
#   May 26 2026: 2 → 1 word + first-chunk comma/colon boundary
# (live evidence: held_for_inorder=7264ms because the first chunk of
# a long reply blocked all 8 parallel chunks behind it. Smaller first
# chunk = audio plays sooner).
FIRST_CHUNK_MIN_WORDS = int(os.environ.get("MS4_VOICE_FIRST_CHUNK_MIN_WORDS", "1"))
MAX_CHUNK_WORDS = int(os.environ.get("MS4_VOICE_MAX_CHUNK_WORDS", "40"))
# Bumped 3 → 6 (May 26 2026): the operator has 2x RTX PRO 6000
# Blackwell — plenty of GPU headroom for concurrent TTS requests, and
# REST is the default engine so more parallelism = faster total
# synthesis on longer replies.
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


# ---------------------------------------------------------------------------
# Fail-closed gate (MS3 /voice/status)
# ---------------------------------------------------------------------------


def check_voice_ready(ms3_url: str, *, timeout: int = 5, retries: int = 2) -> dict[str, Any]:
    """Query MS3 ``/voice/status``. Raise ``VoiceUnavailable`` if ASR is not
    ready. Returns the raw status payload on success.

    Retries on transport errors (transient socket drops like
    ``[WinError 10054] An existing connection was forcibly closed``
    observed live when MS3 momentarily resets TCP listeners during
    its tick loop). Each retry has a 250 ms backoff. A
    ``voice_input_ready: false`` response from MS3 is NOT retried —
    that's MS3 telling us the state, not a transport hiccup.
    """
    url = f"{ms3_url.rstrip('/')}/voice/status"
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8", "replace"))
            break
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, json.JSONDecodeError) as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(0.25)
                continue
            raise VoiceUnavailable(f"MS3 /voice/status check failed (after {retries + 1} attempts): {exc}") from exc
    if not isinstance(payload, dict):
        raise VoiceUnavailable("MS3 /voice/status returned non-object payload")
    ready_flag = payload.get("voice_input_ready")
    if ready_flag is False:
        asr_detail = ""
        asr = payload.get("asr")
        if isinstance(asr, dict):
            asr_detail = f" asr.status={asr.get('status')} detail={asr.get('detail')}"
        raise VoiceUnavailable(f"voice_input_ready=false{asr_detail}")
    return payload


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
    transcribe_model: str | None = None,
    tts_model: str | None = None,
    tts_voice: str | None = None,
    response_format: str | None = None,
) -> VoicePttResult:
    """Chained ASR → chat → TTS turn for the push-to-talk UI.

    ``runner`` is the live :class:`Ms4HermesRunner`. We fail closed on
    ASR readiness first, then run the chat path (which already injects
    the Face Lobe context block and bumps the conversation revision),
    then synthesize the reply.
    """
    check_voice_ready(runner.ms3_url)
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
    chat_response = runner.chat(transcript, session_id=session_id, model=model)
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


class SentenceChunker:
    """Consume streaming text deltas and emit chunks at speech boundaries.

    Strategy:

      1. **First-chunk acceleration.** Until the first chunk fires we
         look for either a sentence terminator OR the configured
         ``first_chunk_min_words`` count. Whichever comes first wins.
         This guarantees the user hears audio start within a small,
         deterministic window no matter what the model decides to say.
      2. **Subsequent chunks fire on sentence boundaries** (``. ! ?``
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
    ) -> None:
        self.first_chunk_min_words = first_chunk_min_words
        self.max_chunk_words = max_chunk_words
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
        ready yet; returns the chunk text otherwise."""
        if not self.buffer:
            return None

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

    # Voice turns ALWAYS use the fast Face Lobe picker, ignoring the
    # UI's "model" dropdown. Live evidence (May 26 2026): operator had
    # qwen3-coder-next:latest selected and every voice turn paid 17-40s
    # chat latency. The dropdown is for text-chat freedom; voice is
    # fail-fast by design. Set MS4_VOICE_ALLOW_MODEL_OVERRIDE=1 to
    # honor the dropdown for voice too (e.g. for debugging).
    if model and os.environ.get("MS4_VOICE_ALLOW_MODEL_OVERRIDE", "").strip().lower() not in {"1", "true", "yes", "on"}:
        log.info(
            "voice turn ignoring per-turn model=%r (fast picker preferred); "
            "set MS4_VOICE_ALLOW_MODEL_OVERRIDE=1 to honor the dropdown for voice",
            model,
        )
        model = None

    # ---- Phase 1: fail-closed check + ASR ---------------------------------
    emit("status", {"phase": "transcribing"})
    check_voice_ready(runner.ms3_url)
    t_asr_start = time.monotonic()
    asr_result = asr_fn(
        hivemind_url=runner.hivemind_url,
        audio=audio,
        filename=filename,
        model=transcribe_model,
    )
    transcript = (asr_result.get("text") or "").strip()
    asr_ms = int((time.monotonic() - t_asr_start) * 1000)
    # Speaker identification (best-effort, capped at ~5s by the
    # voice_identity.identify_speaker_from_wav timeout). Runs in
    # parallel with the chat call ideally; for now we do it inline
    # because the chat call hasn't started yet and ASR already
    # consumed the audio. Subsequent integration could move this to
    # a background thread that joins before the final ``done`` event.
    speaker: dict[str, Any] | None = None
    try:
        from . import voice_identity as _vi

        speaker = _vi.identify_speaker_from_wav(runner.hivemind_url, audio)
    except Exception as exc:  # noqa: BLE001 — fail-soft per design
        log.info("speaker identification skipped: %s", exc)
    emit("transcript", {
        "text": transcript,
        "asr_ms": asr_ms,
        "model": asr_result.get("model"),
        "speaker": speaker,
    })
    if not transcript:
        raise VoiceRequestError("transcription returned empty text; no chat turn dispatched")

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
    counters = {"audio_chunks": 0, "audio_errors": 0, "first_audio_at": None}
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
                    emit("audio_chunk", {
                        "index": chunk.index,
                        "text": chunk.text,
                        "audio_base64": tts_result["audio_base64"],
                        "audio_mime": tts_result["content_type"],
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
        chat_response = chat_call(
            transcript,
            session_id=session_id,
            model=model,
            stream_callback=_on_delta,
        )
    except Exception as exc:
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

    try:
        chat_response = chat_call(
            transcript,
            session_id=session_id,
            model=model,
            stream_callback=_on_delta,
        )
        chat_metrics = (chat_response or {}).get("metrics") or {}
    finally:
        # Drain the filter so any tail-text the model emitted after
        # the last sentence boundary still gets spoken.
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
        first_audio_timeout = float(os.environ.get("MS4_TTS_WS_FIRST_AUDIO_TIMEOUT", "15"))
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
        # Sanitize the reply text the same way the WS path would have.
        from .spoken_text_filter import sanitize_for_speech as _sanitize_fallback
        spoken = _sanitize_fallback(reply_text) if TTS_FILTER_ENABLED else reply_text
        if spoken.strip():
            try:
                # Call the module-level synthesize directly — tts_fn is
                # in the outer voice_ptt_turn_stream scope and not
                # visible from inside _run_ws_super_engine.
                rest_result = synthesize(
                    hivemind_url=runner.hivemind_url,
                    text=spoken,
                    model=tts_model,
                    voice=tts_voice,
                    # response_format isn't plumbed through to the WS
                    # engine call — use the module default. (The REST
                    # path's existing branch handles its own override.)
                    response_format=DEFAULT_TTS_FORMAT,
                )
                audio_bytes = rest_result.get("audio_bytes") or b""
                if audio_bytes:
                    emit("audio_chunk", {
                        "index": 0,
                        "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
                        "audio_mime": rest_result.get("content_type") or "audio/wav",
                        "tts_engine": "rest_fallback",
                    })
                    chunks_emitted = 1
            except Exception as exc:
                log.error("REST TTS fallback also failed: %s", exc)
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
