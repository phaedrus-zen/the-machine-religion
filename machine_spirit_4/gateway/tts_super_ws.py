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
DEFAULT_READER_JOIN_TIMEOUT_SECS = float(
    os.environ.get("MS4_TTS_WS_READER_JOIN_TIMEOUT", "1.0")
)
# Bounded RFC6455 closing-handshake budget. A normal cancellation attempts a
# graceful Close first (so the peer observes a normal close, not an abrupt EOF /
# protocol error); this caps how long that handshake may wait before the raw
# socket shutdown fallback runs. It is never unbounded.
DEFAULT_CLOSE_TIMEOUT_SECS = float(os.environ.get("MS4_TTS_WS_CLOSE_TIMEOUT", "1.0"))
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
        close_timeout: float = DEFAULT_CLOSE_TIMEOUT_SECS,
        connect_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.hivemind_url = hivemind_url.rstrip("/")
        self.voice = voice or DEFAULT_VOICE
        self.emit = emit
        self.t_start = t_start
        self.open_timeout = open_timeout
        self.close_timeout = close_timeout
        self.connect_kwargs = connect_kwargs or {}
        self._ws: Any = None
        self._reader: threading.Thread | None = None
        self._lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._emit_lock = threading.Lock()
        self._done_event = threading.Event()
        self._closed = False
        self._chunk_index = 0
        self._first_audio_ms: int | None = None
        self._audio_frames_received = 0
        self._events_gateway_enqueued = 0
        self._events_client_written = 0
        self._bytes_received = 0
        self._is_final_seen = False
        self._opened_ms: int | None = None
        self._error: str | None = None
        self._delivery_stopped = False
        self._reader_started = False
        self._reader_joined = False
        self._reader_join_timeout = False
        self._first_media_received_ms: int | None = None
        self._first_client_written_ms: int | None = None

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
        # Bound the RFC6455 closing handshake so _close_transport's graceful
        # close can never wait unbounded on an unresponsive peer.
        connect_kwargs.setdefault("close_timeout", self.close_timeout)
        self._ws = ws_sync.connect(
            url,
            open_timeout=self.open_timeout,
            max_size=None,
            **connect_kwargs,
        )
        self._opened_ms = int((time.monotonic() - t0) * 1000)
        accepted = self.emit("tts_engine", {
            "engine": "ws_super",
            "voice": self.voice,
            "url": url,
            "connect_ms": self._opened_ms,
        })
        if accepted is False:
            self._delivery_stopped = True
            self._error = "gateway rejected tts_engine event"
            self._closed = True
            self._close_transport()
            self._done_event.set()
            raise RuntimeError(self._error)
        self._reader = threading.Thread(
            target=self._read_loop, daemon=True, name="ms4-tts-ws-reader",
        )
        self._reader_started = True
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

    def _close_transport(self) -> None:
        ws = self._ws
        # Normal cancellation: attempt the graceful RFC6455 Close FIRST so the
        # peer observes a proper Close frame instead of an abrupt EOF / protocol
        # error. It is bounded by the connect-time close_timeout, so it can never
        # wait unbounded. The raw socket shutdown below is the hard fallback for
        # when the graceful close blocks or fails — and is also the path taken by
        # voice.py's _cancel_builtin_send raw abort for a dead client / blocked
        # send (which cannot send a Close frame and must not wait on a handshake).
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        transport = getattr(ws, "socket", None)
        if transport is not None:
            try:
                transport.shutdown(2)
            except OSError:
                pass
            try:
                transport.close()
            except OSError:
                pass

    def close(self, timeout: float = DEFAULT_READER_JOIN_TIMEOUT_SECS) -> None:
        # Bounded teardown: set the stop flag under the (fast) state lock ONLY —
        # never block on _emit_lock, which the reader holds for the full duration
        # of a blocking emit callback (a wedged consumer would otherwise hang
        # close()). Tearing down the transport (bounded graceful close + raw
        # fallback) unblocks a reader stuck on the WS read; a reader wedged
        # inside a blocking emitter is surfaced via reader_join_timeout rather
        # than hanging close().
        with self._state_lock:
            self._closed = True
        self._close_transport()
        reader = self._reader
        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=max(0.05, timeout))
            with self._state_lock:
                self._reader_joined = not reader.is_alive()
                self._reader_join_timeout = reader.is_alive()
                if reader.is_alive():
                    self._error = self._error or (
                        "WS reader cleanup timed out after "
                        f"{max(0.05, timeout):.2f}s"
                    )
        self._done_event.set()
        if reader is not None and reader.is_alive():
            raise RuntimeError(self._error or "WS reader cleanup timed out")

    # ---- reader -----------------------------------------------------------

    def _read_loop(self) -> None:
        try:
            for msg in self._ws:
                with self._state_lock:
                    if self._closed or self._delivery_stopped:
                        return
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
                    with self._state_lock:
                        self._bytes_received += len(pcm)
                        self._audio_frames_received += 1
                        if self._first_media_received_ms is None:
                            self._first_media_received_ms = int(
                                (time.monotonic() - self.t_start) * 1000
                            )
                    wav = pcm_to_wav(pcm)
                    with self._emit_lock:
                        with self._state_lock:
                            if self._closed or self._delivery_stopped:
                                return
                            self._events_gateway_enqueued += 1
                        accepted = self.emit("audio_chunk", {
                            "index": self._chunk_index,
                            "text": "",  # WS doesn't identify the source clause.
                            "audio_base64": base64.b64encode(wav).decode("ascii"),
                            "audio_mime": "audio/wav",
                            "tts_ms": int((time.monotonic() - self.t_start) * 1000),
                            "held_for_inorder_ms": 0,
                            "engine": "ws_super",
                        })
                        if accepted is False:
                            with self._state_lock:
                                self._delivery_stopped = True
                                self._closed = True
                                self._error = self._error or "gateway rejected audio delivery"
                    if accepted is False:
                        self._close_transport()
                        return
                    with self._state_lock:
                        if self._closed or self._delivery_stopped:
                            return
                        if self._first_audio_ms is None:
                            self._first_audio_ms = int(
                                (time.monotonic() - self.t_start) * 1000
                            )
                            self._first_client_written_ms = self._first_audio_ms
                        self._chunk_index += 1
                        self._events_client_written += 1
                if evt.get("isFinal"):
                    with self._state_lock:
                        self._is_final_seen = True
                        if self._bytes_received == 0:
                            self._error = self._error or "isFinal received with zero audio media"
                    self._done_event.set()
                    return
        except Exception as exc:
            self._error = self._error or f"reader error: {exc}"
            log.warning("TTS WS reader closed unexpectedly: %s", exc)
        finally:
            with self._state_lock:
                if not self._is_final_seen:
                    if self._error is None:
                        self._error = "WS reader closed before isFinal"
                    elif "before isFinal" not in self._error:
                        self._error = f"{self._error}; closed before isFinal"
            # Always release waiters so the parent caller doesn't hang
            # on wait_for_final even if the server closed the WS without
            # sending isFinal.
            self._done_event.set()

    # ---- summary ----------------------------------------------------------

    def metrics(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                "engine": "ws_super",
                "voice": self.voice,
                "connect_ms": self._opened_ms,
                "first_audio_ms": self._first_audio_ms,
                "first_media_received_ms": self._first_media_received_ms,
                "first_client_written_ms": self._first_client_written_ms,
                "chunks_emitted": self._events_client_written,
                "audio_frames_received": self._audio_frames_received,
                "events_gateway_enqueued": self._events_gateway_enqueued,
                "events_client_written": self._events_client_written,
                "bytes_received": self._bytes_received,
                "is_final_seen": self._is_final_seen,
                "zero_media": self._bytes_received == 0,
                "delivery_stopped": self._delivery_stopped,
                "reader_started": self._reader_started,
                "reader_joined": self._reader_joined,
                "reader_join_timeout": self._reader_join_timeout,
                "reader_alive": bool(self._reader is not None and self._reader.is_alive()),
                "error": self._error,
            }
