"""Tests for the HiveMind TTS_SUPER WebSocket engine.

We exercise two layers:

1. **pcm_to_wav** — pure function. Wraps a raw PCM blob in a minimal
   WAV header and produces something the browser's ``decodeAudioData``
   can decode. We verify the standard WAV structural fields directly.

2. **TtsSuperWsEngine** — end-to-end against a tiny in-process
   WebSocket server that mimics HiveMind's ``stream-input`` protocol.
   We assert:
     - the engine emits ``tts_engine`` once at open with engine="ws_super"
     - the engine emits ``audio_chunk`` events as PCM frames arrive,
       wrapped as WAV, in order
     - the engine resolves ``wait_for_final`` when the fake server
       sends ``{audio: null, isFinal: true}``
     - the engine's ``metrics()`` reports chunks_emitted, first_audio_ms,
       and is_final_seen=True
"""

from __future__ import annotations

import asyncio
import base64
import json
import struct
import threading
import time
import wave
from io import BytesIO

import pytest
import websockets

from machine_spirit_4.gateway.tts_super_ws import (
    TtsSuperWsEngine,
    WS_PCM_CHANNELS,
    WS_PCM_SAMPLE_RATE,
    WS_PCM_SAMPLE_WIDTH,
    pcm_to_wav,
)


# ---------------------------------------------------------------------------
# pcm_to_wav
# ---------------------------------------------------------------------------


def test_pcm_to_wav_produces_valid_riff_header():
    pcm = b"\x00\x00" * 100  # 100 silent s16 samples = 200 bytes
    wav = pcm_to_wav(pcm)
    assert wav.startswith(b"RIFF")
    assert wav[8:12] == b"WAVE"
    # The standard 44-byte header + the data.
    assert len(wav) == 44 + len(pcm)
    # Parse via stdlib wave to double-check we wrote it correctly.
    with wave.open(BytesIO(wav), "rb") as r:
        assert r.getnchannels() == WS_PCM_CHANNELS
        assert r.getsampwidth() == WS_PCM_SAMPLE_WIDTH
        assert r.getframerate() == WS_PCM_SAMPLE_RATE
        assert r.readframes(r.getnframes()) == pcm


def test_pcm_to_wav_supports_overrides():
    pcm = b"\x00\x00" * 20
    wav = pcm_to_wav(pcm, sample_rate=16000, channels=2, sample_width=2)
    with wave.open(BytesIO(wav), "rb") as r:
        assert r.getframerate() == 16000
        assert r.getnchannels() == 2


# ---------------------------------------------------------------------------
# TtsSuperWsEngine — integration vs fake WS server
# ---------------------------------------------------------------------------


# Build a small reusable fake WS server harness. The fake speaks the
# subset of the HiveMind TTS_SUPER stream-input protocol that the
# engine uses: it reads pushed {text, ...} frames; after the first
# ~3 frames or on flush:true it emits two PCM audio chunks; finally
# {audio: null, isFinal: true}.


class _FakeTtsSuperServer:
    def __init__(self):
        self.received_texts: list[str] = []
        self.flushed = False
        self._stop = threading.Event()
        self._server_thread: threading.Thread | None = None
        self.port = 0

    def start(self):
        loop_ready = threading.Event()

        def _run():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            async def handler(ws):
                try:
                    push_count = 0
                    async for raw in ws:
                        try:
                            msg = json.loads(raw)
                        except Exception:
                            continue
                        if msg.get("flush"):
                            self.flushed = True
                            await self._emit_two_audio_chunks(ws)
                            await ws.send(json.dumps({"audio": None, "isFinal": True}))
                            return
                        text = msg.get("text") or ""
                        if text:
                            self.received_texts.append(text)
                            push_count += 1
                            if push_count == 2:
                                # emit the first chunk immediately so the
                                # engine's first_audio_ms is recorded
                                # before the flush path
                                await ws.send(json.dumps({
                                    "audio": base64.b64encode(b"\x10\x00" * 50).decode("ascii"),
                                    "isFinal": False,
                                }))
                except websockets.exceptions.ConnectionClosed:
                    return

            async def _serve():
                async with websockets.serve(handler, "127.0.0.1", 0) as s:
                    sock = s.sockets[0]
                    self.port = sock.getsockname()[1]
                    loop_ready.set()
                    while not self._stop.is_set():
                        await asyncio.sleep(0.05)

            try:
                self._loop.run_until_complete(_serve())
            finally:
                self._loop.close()

        self._server_thread = threading.Thread(target=_run, daemon=True, name="fake-tts-ws")
        self._server_thread.start()
        loop_ready.wait(timeout=5)

    async def _emit_two_audio_chunks(self, ws):
        # Two chunks with different sizes so the test can detect both arrived.
        await ws.send(json.dumps({
            "audio": base64.b64encode(b"\x20\x00" * 25).decode("ascii"),
            "isFinal": False,
        }))
        await ws.send(json.dumps({
            "audio": base64.b64encode(b"\x30\x00" * 75).decode("ascii"),
            "isFinal": False,
        }))

    def stop(self):
        self._stop.set()
        if self._server_thread:
            self._server_thread.join(timeout=3)


@pytest.fixture
def fake_tts_server():
    s = _FakeTtsSuperServer()
    s.start()
    try:
        yield s
    finally:
        s.stop()


def test_engine_opens_emits_chunks_and_completes_on_final(fake_tts_server):
    events: list[tuple[str, dict]] = []
    lock = threading.Lock()

    def emit(name, payload):
        with lock:
            events.append((name, dict(payload)))
        return True

    engine = TtsSuperWsEngine(
        hivemind_url=f"http://127.0.0.1:{fake_tts_server.port}",
        voice="alloy",
        emit=emit,
        t_start=time.monotonic(),
    )
    engine.open()
    try:
        engine.push("Hello ")
        engine.push("there. ")
        engine.push("Friend.")
        engine.flush()
        completed = engine.wait_for_final(timeout=5.0)
        assert completed, "engine should resolve wait_for_final when fake sends isFinal"
    finally:
        engine.close()

    types = [e for e, _ in events]
    assert "tts_engine" in types, f"expected tts_engine open event, got {types}"
    audio_payloads = [p for n, p in events if n == "audio_chunk"]
    assert len(audio_payloads) >= 2, f"expected at least 2 audio_chunks, got {len(audio_payloads)}"
    # All audio_chunks should declare the WS engine + be valid WAV.
    for p in audio_payloads:
        assert p.get("engine") == "ws_super"
        assert p.get("audio_mime") == "audio/wav"
        wav = base64.b64decode(p["audio_base64"])
        assert wav.startswith(b"RIFF") and wav[8:12] == b"WAVE"
    # Indices are monotonic.
    indices = [p["index"] for p in audio_payloads]
    assert indices == sorted(indices)
    # Held-for-inorder is always 0 for the WS engine.
    assert all(p.get("held_for_inorder_ms") == 0 for p in audio_payloads)
    # Metrics report finals.
    m = engine.metrics()
    assert m["engine"] == "ws_super"
    assert m["chunks_emitted"] == len(audio_payloads)
    assert m["is_final_seen"] is True
    assert m["first_audio_ms"] is not None and m["first_audio_ms"] >= 0
    # Fake server should have received the pushed text.
    assert "".join(fake_tts_server.received_texts) == "Hello there. Friend."
    assert fake_tts_server.flushed is True


def test_engine_close_releases_waiters_even_without_isfinal(fake_tts_server):
    """If the server drops the WS without sending isFinal, the engine
    MUST NOT block forever — wait_for_final should resolve from the
    finally-clause in the reader thread, and the metrics should
    report is_final_seen=False so the caller can detect it."""
    events: list[tuple[str, dict]] = []

    def emit(name, payload):
        events.append((name, dict(payload)))
        return True

    engine = TtsSuperWsEngine(
        hivemind_url=f"http://127.0.0.1:{fake_tts_server.port}",
        voice="alloy",
        emit=emit,
        t_start=time.monotonic(),
    )
    engine.open()
    # Don't push anything. Don't flush. Close the client side so the
    # server's async-for loop exits naturally without sending isFinal.
    engine.close()
    waited = engine.wait_for_final(timeout=3.0)
    assert waited, "wait_for_final should resolve when WS closes (no isFinal)"
    assert engine.metrics()["is_final_seen"] is False
