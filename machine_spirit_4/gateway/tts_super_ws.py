"""HiveMind TTS_SUPER WebSocket engine for the streaming voice pipeline.

Background
----------

The default voice engine (``voice.py`` ``voice_ptt_turn_stream`` with
``engine="rest"``) chunks the model's reply by sentence and fires one
``POST /v1/audio/speech`` per chunk in parallel. That works, but each
chunk pays a per-call cost on HiveMind and the in-order emit guarantee
sometimes holds a fast chunk behind a slow one.

HiveMind exposes a much more efficient surface:

  ``WS /v1/text-to-speech/{voice_id}/stream-input``

You open one WebSocket per turn, push text fragments as they arrive,
and HiveMind emits ``{audio: <base64 PCM s16 mono @ 24 kHz>, isFinal: false}``
frames continuously as it synthesizes. When the turn ends you push
``{flush: true}`` and the server emits ``{audio: null, isFinal: true}``
exactly once.

Wins over the REST path measured live against the same cluster:

  REST per-sentence (current default) :  first audio ~10s, total ~25s
  WS TTS_SUPER stream-input (this)    :  first audio ~2.5s, total ~5.9s

This module wraps that WebSocket so it integrates with the existing
``emit(event, payload)`` callback shape that the gateway uses to drive
the SSE response — the UI sees the same ``audio_chunk`` event format,
so no UI change is needed. Each PCM frame is wrapped in a minimal WAV
header here on the server so the browser's ``decodeAudioData`` call
keeps working unchanged.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import struct
import threading
import time
import wave
from typing import Any, Callable

import websockets.sync.client as ws_sync


log = logging.getLogger("ms4.gateway.tts_super_ws")


DEFAULT_VOICE = "alloy"
# Open timeout for the TTS_SUPER WS handshake. 5s was tight for the
# cold-GIM case (observed live: handshake timed out at 5s when the
# GIM had been idle ~10 minutes between turns). Bumped to 10s so a
# cold GIM has time to wake before MS4 falls back to REST. Operators
# on warm clusters won't notice; cold paths now degrade gracefully
# rather than failing the whole turn.
DEFAULT_OPEN_TIMEOUT_SECS = float(os.environ.get("MS4_TTS_WS_OPEN_TIMEOUT", "10.0"))
DEFAULT_FINAL_TIMEOUT_SECS = float(os.environ.get("MS4_TTS_WS_FINAL_TIMEOUT", "60.0"))
# HiveMind TTS_SUPER emits raw PCM s16 mono @ 24 kHz per the live
# probe (and per docs/specs/HIVEMIND_VOICE_CONTRACT.md §4).
WS_PCM_SAMPLE_RATE = 24_000
WS_PCM_CHANNELS = 1
WS_PCM_SAMPLE_WIDTH = 2  # bytes (s16)


# ---------------------------------------------------------------------------
# PCM -> WAV wrap (so the browser keeps using decodeAudioData unchanged)
# ---------------------------------------------------------------------------


def pcm_to_wav(
    pcm_bytes: bytes,
    *,
    sample_rate: int = WS_PCM_SAMPLE_RATE,
    channels: int = WS_PCM_CHANNELS,
    sample_width: int = WS_PCM_SAMPLE_WIDTH,
) -> bytes:
    """Wrap a raw PCM blob in a minimal WAV container.

    The browser's Web Audio ``decodeAudioData`` expects a container
    format (WAV / Ogg / etc.). HiveMind's WS gives us raw PCM, so we
    add the 44-byte WAV/RIFF header per chunk. The cost is negligible
    (44 bytes per chunk; chunks are typically tens of kB)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm_bytes)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class TtsSuperWsEngine:
    """One-WebSocket-per-turn TTS engine that streams audio chunks via
    the ``emit(event, payload)`` callback exactly like the REST path.

    Lifecycle:

      engine = TtsSuperWsEngine(hivemind_url=..., voice="alloy",
                                emit=emit, t_start=t_turn_start)
      engine.open()
      try:
          for token in face_lobe_stream:
              engine.push(token)            # safe to call from any thread
          engine.flush()                    # signal end of input
          engine.wait_for_final(timeout=60) # block until isFinal arrives
      finally:
          engine.close()
          metrics = engine.metrics()

    The reader runs in a background thread and emits ``audio_chunk``
    events as PCM frames arrive. Frames are wrapped as WAV so the
    browser's existing ``decodeAudioData`` path stays the same.

    Errors during open or read surface via the ``error`` SSE event
    (the engine emits it directly so the caller doesn't have to wrap
    every push() call).
    """

    def __init__(
        self,
        *,
        hivemind_url: str,
        voice: str | None = None,
        emit: Callable[[str, dict[str, Any]], bool],
        t_start: float,
        open_timeout: float = DEFAULT_OPEN_TIMEOUT_SECS,
        connect_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.hivemind_url = hivemind_url.rstrip("/")
        self.voice = voice or DEFAULT_VOICE
        self.emit = emit
        self.t_start = t_start
        self.open_timeout = open_timeout
        self.connect_kwargs = connect_kwargs or {}
        self._ws: Any = None
        self._reader: threading.Thread | None = None
        self._lock = threading.Lock()
        self._done_event = threading.Event()
        self._closed = False
        self._chunk_index = 0
        self._first_audio_ms: int | None = None
        self._chunks_emitted = 0
        self._bytes_received = 0
        self._is_final_seen = False
        self._opened_ms: int | None = None
        self._error: str | None = None

    # ---- websocket URL ----------------------------------------------------

    def _ws_url(self) -> str:
        base = self.hivemind_url
        if base.startswith("http://"):
            base = "ws://" + base[len("http://"):]
        elif base.startswith("https://"):
            base = "wss://" + base[len("https://"):]
        return f"{base}/v1/text-to-speech/{self.voice}/stream-input"

    # ---- lifecycle --------------------------------------------------------

    def open(self) -> None:
        from .hivemind_state import hivemind_auth_headers

        url = self._ws_url()
        t0 = time.monotonic()
        # Bearer auth (when MS4_HIVEMIND_API_KEY is set) so the WS
        # handshake survives against clusters with MENTA_API_KEYS
        # configured. ``additional_headers`` is the documented kwarg
        # in websockets.sync.client.connect; merge instead of replace
        # so callers can still pass connect_kwargs of their own.
        connect_kwargs = dict(self.connect_kwargs)
        auth = hivemind_auth_headers()
        if auth:
            existing = connect_kwargs.get("additional_headers") or []
            if isinstance(existing, dict):
                merged = dict(existing)
                merged.update(auth)
                connect_kwargs["additional_headers"] = merged
            else:
                merged = list(existing)
                for header_name, header_value in auth.items():
                    merged.append((header_name, header_value))
                connect_kwargs["additional_headers"] = merged
        # Disable ping/pong keepalive. Live evidence (May 26 2026):
        # HiveMind's TTS_SUPER GIM is too busy synthesizing to pong
        # the client's 20s default keepalive ping, so the connection
        # gets killed with code 1011 (internal error) keepalive ping
        # timeout — emitting zero audio chunks. A single-turn TTS
        # synthesis is short-lived enough that we don't need pings
        # to detect dead connections; the existing wait_for_final
        # timeout (60s) bounds the wait either way.
        connect_kwargs.setdefault("ping_interval", None)
        connect_kwargs.setdefault("ping_timeout", None)
        self._ws = ws_sync.connect(
            url,
            open_timeout=self.open_timeout,
            max_size=None,
            **connect_kwargs,
        )
        self._opened_ms = int((time.monotonic() - t0) * 1000)
        self.emit("tts_engine", {
            "engine": "ws_super",
            "voice": self.voice,
            "url": url,
            "connect_ms": self._opened_ms,
        })
        self._reader = threading.Thread(
            target=self._read_loop, daemon=True, name="ms4-tts-ws-reader",
        )
        self._reader.start()

    def push(self, text: str) -> None:
        if not text or self._closed or self._ws is None:
            return
        try:
            with self._lock:
                self._ws.send(json.dumps({
                    "text": text,
                    "try_trigger_generation": True,
                }))
        except Exception as exc:
            self._error = self._error or f"push failed: {exc}"
            log.warning("TTS WS push failed: %s", exc)

    def flush(self) -> None:
        if self._closed or self._ws is None:
            return
        try:
            with self._lock:
                self._ws.send(json.dumps({"text": "", "flush": True}))
        except Exception as exc:
            self._error = self._error or f"flush failed: {exc}"
            log.warning("TTS WS flush failed: %s", exc)

    def wait_for_final(self, timeout: float = DEFAULT_FINAL_TIMEOUT_SECS) -> bool:
        return self._done_event.wait(timeout=timeout)

    def close(self) -> None:
        self._closed = True
        try:
            if self._ws is not None:
                self._ws.close()
        except Exception:
            pass
        # The reader thread exits naturally when the WS closes.

    # ---- reader -----------------------------------------------------------

    def _read_loop(self) -> None:
        try:
            for msg in self._ws:
                try:
                    evt = json.loads(msg) if isinstance(msg, (str, bytes, bytearray)) else msg
                except Exception:
                    log.warning("TTS WS reader: non-JSON frame ignored: %r", str(msg)[:80])
                    continue
                if not isinstance(evt, dict):
                    continue
                audio_b64 = evt.get("audio")
                if audio_b64:
                    try:
                        pcm = base64.b64decode(audio_b64)
                    except Exception as exc:
                        log.warning("TTS WS reader: bad base64: %s", exc)
                        continue
                    self._bytes_received += len(pcm)
                    wav = pcm_to_wav(pcm)
                    if self._first_audio_ms is None:
                        self._first_audio_ms = int((time.monotonic() - self.t_start) * 1000)
                    self.emit("audio_chunk", {
                        "index": self._chunk_index,
                        "text": "",  # WS doesn't tell us which clause this audio belongs to
                        "audio_base64": base64.b64encode(wav).decode("ascii"),
                        "audio_mime": "audio/wav",
                        "tts_ms": int((time.monotonic() - self.t_start) * 1000),
                        # WS audio is naturally ordered, so there's never a held-for-inorder cost.
                        "held_for_inorder_ms": 0,
                        "engine": "ws_super",
                    })
                    self._chunk_index += 1
                    self._chunks_emitted += 1
                if evt.get("isFinal"):
                    self._is_final_seen = True
                    self._done_event.set()
                    return
        except Exception as exc:
            self._error = self._error or f"reader error: {exc}"
            log.warning("TTS WS reader closed unexpectedly: %s", exc)
        finally:
            # Always release waiters so the parent caller doesn't hang
            # on wait_for_final even if the server closed the WS without
            # sending isFinal.
            self._done_event.set()

    # ---- summary ----------------------------------------------------------

    def metrics(self) -> dict[str, Any]:
        return {
            "engine": "ws_super",
            "voice": self.voice,
            "connect_ms": self._opened_ms,
            "first_audio_ms": self._first_audio_ms,
            "chunks_emitted": self._chunks_emitted,
            "bytes_received": self._bytes_received,
            "is_final_seen": self._is_final_seen,
            "error": self._error,
        }
