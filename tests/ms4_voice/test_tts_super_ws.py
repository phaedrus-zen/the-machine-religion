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
    assert m["audio_frames_received"] == len(audio_payloads)
    assert m["events_gateway_enqueued"] == len(audio_payloads)
    assert m["events_client_written"] == len(audio_payloads)
    assert m["is_final_seen"] is True
    assert m["reader_joined"] is True
    assert m["reader_alive"] is False
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
    metrics = engine.metrics()
    assert metrics["is_final_seen"] is False
    assert metrics["reader_joined"] is True
    assert metrics["reader_alive"] is False
    assert "before isFinal" in metrics["error"]


def _engine_with_frames(frames, emit):
    engine = TtsSuperWsEngine(
        hivemind_url="http://127.0.0.1:1",
        voice="alloy",
        emit=emit,
        t_start=time.monotonic(),
    )
    engine._ws = iter(frames)
    return engine


def test_rejected_emitter_stops_delivery_and_is_not_counted():
    pcm = base64.b64encode(b"\x10\x00" * 20).decode("ascii")
    callbacks: list[str] = []

    def reject(name, _payload):
        callbacks.append(name)
        return False

    engine = _engine_with_frames(
        [
            json.dumps({"audio": pcm, "isFinal": False}),
            json.dumps({"audio": pcm, "isFinal": False}),
            json.dumps({"audio": None, "isFinal": True}),
        ],
        reject,
    )
    engine._read_loop()
    metrics = engine.metrics()

    assert callbacks == ["audio_chunk"]
    assert metrics["audio_frames_received"] == 1
    assert metrics["events_gateway_enqueued"] == 1
    assert metrics["events_client_written"] == 0
    assert metrics["chunks_emitted"] == 0
    assert metrics["delivery_stopped"] is True
    assert "rejected" in metrics["error"]


def test_final_without_media_is_terminal_failure():
    engine = _engine_with_frames(
        [json.dumps({"audio": None, "isFinal": True})],
        lambda _name, _payload: True,
    )
    engine._read_loop()
    metrics = engine.metrics()

    assert metrics["is_final_seen"] is True
    assert metrics["zero_media"] is True
    assert metrics["events_client_written"] == 0
    assert "zero audio media" in metrics["error"]


def test_reader_end_without_final_is_terminal_failure():
    engine = _engine_with_frames([], lambda _name, _payload: True)
    engine._read_loop()
    metrics = engine.metrics()

    assert metrics["is_final_seen"] is False
    assert metrics["zero_media"] is True
    assert "before isFinal" in metrics["error"]


def test_close_boundedly_joins_reader():
    release = threading.Event()

    class _BlockingWs:
        def __iter__(self):
            return self

        def __next__(self):
            release.wait(timeout=2.0)
            raise StopIteration

        def close(self):
            release.set()

    engine = _engine_with_frames([], lambda _name, _payload: True)
    engine._ws = _BlockingWs()
    engine._reader = threading.Thread(target=engine._read_loop, name="ms4-tts-ws-reader")
    engine._reader_started = True
    engine._reader.start()
    engine.close(timeout=0.5)
    metrics = engine.metrics()

    assert metrics["reader_joined"] is True
    assert metrics["reader_join_timeout"] is False
    assert metrics["reader_alive"] is False


def test_closed_engine_never_invokes_post_close_callback():
    pcm = base64.b64encode(b"\x10\x00" * 20).decode("ascii")
    callbacks: list[str] = []
    engine = _engine_with_frames(
        [json.dumps({"audio": pcm, "isFinal": False})],
        lambda name, _payload: callbacks.append(name) or True,
    )
    engine._closed = True
    engine._read_loop()

    assert callbacks == []
    assert engine.metrics()["events_gateway_enqueued"] == 0


# ---------------------------------------------------------------------------
# 2026-07-05 graceful RFC6455 close before raw shutdown (+ bounded fallback)
#
# A normal timeout/cancellation must send a graceful Close frame FIRST so the
# peer observes a normal close instead of an abrupt EOF / protocol error. The
# raw socket shutdown is only the hard fallback (and stays bounded). The raw
# abort for dead-client / blocked-send emergencies (voice.py _cancel_builtin_send)
# is preserved separately and is not exercised here.
# ---------------------------------------------------------------------------


class _RawRecordingTransport:
    def __init__(self, calls):
        self._calls = calls

    def shutdown(self, _how):
        self._calls.append("raw_shutdown")

    def close(self):
        self._calls.append("raw_close")


class _CloseRecordingWs:
    """Fake ws exposing the two teardown surfaces ``_close_transport`` uses:
    a graceful ``close()`` (RFC6455 Close handshake) and a raw ``socket``
    (shutdown/close). Records call order."""

    def __init__(self, calls, *, close_raises=False, close_sleep=0.0):
        self._calls = calls
        self.socket = _RawRecordingTransport(calls)
        self._close_raises = close_raises
        self._close_sleep = close_sleep

    def close(self):
        if self._close_sleep:
            time.sleep(self._close_sleep)
        self._calls.append("graceful_close")
        if self._close_raises:
            raise RuntimeError("graceful close failed")


def _bare_engine():
    return TtsSuperWsEngine(
        hivemind_url="http://127.0.0.1:1",
        voice="alloy",
        emit=lambda _n, _p: True,
        t_start=time.monotonic(),
    )


def test_close_transport_attempts_graceful_close_before_raw_shutdown():
    calls: list[str] = []
    engine = _bare_engine()
    engine._ws = _CloseRecordingWs(calls)
    engine._close_transport()
    assert "graceful_close" in calls and "raw_shutdown" in calls
    assert calls.index("graceful_close") < calls.index("raw_shutdown"), (
        "a normal cancellation must send the RFC6455 Close first; the raw socket "
        f"shutdown is only the hard fallback. order={calls}"
    )


def test_close_transport_raw_fallback_runs_and_is_bounded_when_graceful_fails():
    calls: list[str] = []
    engine = _bare_engine()
    # Graceful close briefly blocks AND raises: the raw shutdown MUST still run
    # as the hard fallback, and the whole teardown must stay bounded.
    engine._ws = _CloseRecordingWs(calls, close_raises=True, close_sleep=0.1)
    t0 = time.monotonic()
    engine._close_transport()
    elapsed = time.monotonic() - t0
    assert calls and calls[0] == "graceful_close", (
        f"graceful close must be attempted first, got order={calls}"
    )
    assert "raw_shutdown" in calls, "raw shutdown must still run when graceful close fails"
    assert elapsed < 1.0, "teardown must be bounded, never an unbounded close wait"


class _CloseObservingServer:
    """Minimal WS server that records how the client closed: a graceful
    RFC6455 close yields a normal close code; a raw socket shutdown yields a
    1006 abnormal close (the abrupt EOF the peer logs as a protocol error)."""

    def __init__(self):
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.port = 0
        self.close_seen = threading.Event()
        self.close_code = None
        self.abnormal = None

    def start(self):
        ready = threading.Event()

        def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def handler(ws):
                try:
                    async for _raw in ws:
                        pass
                except websockets.exceptions.ConnectionClosed:
                    pass
                finally:
                    self.close_code = getattr(ws, "close_code", None)
                    self.abnormal = self.close_code == 1006
                    self.close_seen.set()

            async def _serve():
                async with websockets.serve(handler, "127.0.0.1", 0) as s:
                    self.port = s.sockets[0].getsockname()[1]
                    ready.set()
                    while not self._stop.is_set():
                        await asyncio.sleep(0.05)

            try:
                loop.run_until_complete(_serve())
            finally:
                loop.close()

        self._thread = threading.Thread(target=_run, daemon=True, name="close-observing-ws")
        self._thread.start()
        ready.wait(timeout=5)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)


@pytest.fixture
def close_observing_server():
    s = _CloseObservingServer()
    s.start()
    try:
        yield s
    finally:
        s.stop()


def test_graceful_close_lets_peer_observe_normal_close_not_abrupt_eof(close_observing_server):
    engine = TtsSuperWsEngine(
        hivemind_url=f"http://127.0.0.1:{close_observing_server.port}",
        voice="alloy",
        emit=lambda _n, _p: True,
        t_start=time.monotonic(),
    )
    engine.open()
    engine.close()
    assert close_observing_server.close_seen.wait(timeout=3.0), "server never observed the client close"
    assert close_observing_server.abnormal is False, (
        "a raw socket shutdown makes the peer see a 1006 abnormal/EOF close; a "
        f"graceful RFC6455 close must be a normal close. code={close_observing_server.close_code}"
    )


# ---------------------------------------------------------------------------
# 2026-07-05 truthful TTS_SUPER prewarm verdict: warmed=True ONLY after real
# audio media + isFinal + no error, within a 10s budget.
# ---------------------------------------------------------------------------


def _run_prewarm_with_fake_engine(monkeypatch, *, bytes_received, is_final_seen, error, wait_returns):
    import machine_spirit_4.gateway.tts_super_ws as ttsmod
    from machine_spirit_4.gateway.voice import prewarm_tts_super_ws

    seen: dict[str, float] = {}

    class _FakePrewarmEngine:
        def __init__(self, *, hivemind_url, voice, emit, t_start, open_timeout=5.0, **_):
            seen["open_timeout"] = open_timeout

        def open(self):
            pass

        def push(self, _t):
            pass

        def flush(self):
            pass

        def wait_for_final(self, timeout=60):
            seen["wait_timeout"] = timeout
            return wait_returns

        def close(self):
            pass

        def metrics(self):
            return {
                "engine": "ws_super",
                "bytes_received": bytes_received,
                "is_final_seen": is_final_seen,
                "error": error,
            }

    monkeypatch.setattr(ttsmod, "TtsSuperWsEngine", _FakePrewarmEngine)
    return prewarm_tts_super_ws(hivemind_url="http://hive:0"), seen


def test_prewarm_true_only_with_media_final_and_no_error(monkeypatch):
    result, seen = _run_prewarm_with_fake_engine(
        monkeypatch, bytes_received=200, is_final_seen=True, error=None, wait_returns=True
    )
    assert result["warmed"] is True
    # One 10s budget: connect gets the full budget, the final wait gets the
    # REMAINING budget (~10s here because the fake connect is instant).
    assert seen["open_timeout"] == 10.0
    assert 9.0 <= seen["wait_timeout"] <= 10.0


def test_prewarm_false_on_zero_media(monkeypatch):
    result, _ = _run_prewarm_with_fake_engine(
        monkeypatch, bytes_received=0, is_final_seen=True, error=None, wait_returns=True
    )
    assert result["warmed"] is False


def test_prewarm_false_on_missing_final(monkeypatch):
    result, _ = _run_prewarm_with_fake_engine(
        monkeypatch, bytes_received=200, is_final_seen=False, error=None, wait_returns=False
    )
    assert result["warmed"] is False


def test_prewarm_false_on_error(monkeypatch):
    result, _ = _run_prewarm_with_fake_engine(
        monkeypatch, bytes_received=200, is_final_seen=True, error="boom", wait_returns=True
    )
    assert result["warmed"] is False


def test_prewarm_false_on_timeout(monkeypatch):
    result, _ = _run_prewarm_with_fake_engine(
        monkeypatch, bytes_received=0, is_final_seen=False, error=None, wait_returns=False
    )
    assert result["warmed"] is False


# ---------------------------------------------------------------------------
# 2026-07-05 (verifier P1): prewarm must enforce ONE absolute monotonic 10s
# budget across connect + final/media completion — the later wait gets only the
# remaining time, and warmed is never True past the absolute deadline. Simulated
# time is injected via a monotonic offset (original captured to avoid recursion).
# ---------------------------------------------------------------------------


def test_prewarm_passes_only_remaining_budget_to_final_wait(monkeypatch):
    import machine_spirit_4.gateway.voice as v
    import machine_spirit_4.gateway.tts_super_ws as ttsmod

    orig = v.time.monotonic
    offset = {"v": 0.0}
    monkeypatch.setattr(v.time, "monotonic", lambda: orig() + offset["v"])
    seen: dict[str, float] = {}

    class _SlowConnectEngine:
        def __init__(self, *, open_timeout=5.0, **_):
            seen["open_timeout"] = open_timeout

        def open(self):
            offset["v"] += 8.0  # connect consumes 8s of the 10s budget

        def push(self, _t):
            pass

        def flush(self):
            pass

        def wait_for_final(self, timeout=60):
            seen["wait_timeout"] = timeout
            offset["v"] += max(0.0, timeout)  # the final wait consumes its whole budget
            return True

        def close(self):
            pass

        def metrics(self):
            return {"engine": "ws_super", "bytes_received": 200, "is_final_seen": True, "error": None}

    monkeypatch.setattr(ttsmod, "TtsSuperWsEngine", _SlowConnectEngine)
    result = v.prewarm_tts_super_ws(hivemind_url="http://hive:0")

    assert seen["wait_timeout"] <= 2.01, (
        f"the final wait must get only the ~2s remaining budget, not a fresh 10s; got {seen['wait_timeout']}"
    )
    assert result["elapsed_ms"] <= 10_050, (
        f"one absolute 10s budget must cap connect+final; got {result['elapsed_ms']}ms"
    )


def test_prewarm_not_warmed_after_absolute_deadline(monkeypatch):
    import machine_spirit_4.gateway.voice as v
    import machine_spirit_4.gateway.tts_super_ws as ttsmod

    orig = v.time.monotonic
    offset = {"v": 0.0}
    monkeypatch.setattr(v.time, "monotonic", lambda: orig() + offset["v"])
    seen: dict[str, float] = {}

    class _OverBudgetEngine:
        def __init__(self, *, open_timeout=5.0, **_):
            seen["open_timeout"] = open_timeout

        def open(self):
            offset["v"] += 11.0  # connect alone blows the 10s budget

        def push(self, _t):
            pass

        def flush(self):
            pass

        def wait_for_final(self, timeout=60):
            seen["wait_timeout"] = timeout
            offset["v"] += max(0.0, timeout)
            return True

        def close(self):
            pass

        def metrics(self):
            return {"engine": "ws_super", "bytes_received": 200, "is_final_seen": True, "error": None}

    monkeypatch.setattr(ttsmod, "TtsSuperWsEngine", _OverBudgetEngine)
    result = v.prewarm_tts_super_ws(hivemind_url="http://hive:0")

    assert result["warmed"] is False, "must never report warmed after the absolute 10s deadline"
    assert result["elapsed_ms"] >= 11_000
    # No fresh positive budget may be handed to a wait started past the deadline.
    assert seen.get("wait_timeout", 0.0) <= 0.01


# ---------------------------------------------------------------------------
# 2026-07-05 (verifier P1): public close() must stay bounded even when the
# reader is wedged inside a blocking emit callback holding _emit_lock. It must
# attempt the bounded graceful RFC6455 close then the raw shutdown fallback and
# bounded-join, never blocking on _emit_lock.
# ---------------------------------------------------------------------------


def test_public_close_is_bounded_even_when_emitter_holds_emit_lock():
    emit_entered = threading.Event()
    graceful: list[bool] = []
    raw_shutdown: list[bool] = []

    class _StuckTransport:
        def shutdown(self, _how):
            raw_shutdown.append(True)

        def close(self):
            pass

    class _StuckWs:
        def __init__(self):
            self._frames = iter([
                json.dumps({"audio": base64.b64encode(b"\x10\x00" * 20).decode("ascii"), "isFinal": False}),
            ])
            self.socket = _StuckTransport()

        def __iter__(self):
            return self

        def __next__(self):
            return next(self._frames)  # one frame, then StopIteration

        def close(self):
            graceful.append(True)

    def blocking_emit(_name, _payload):
        emit_entered.set()
        time.sleep(1.0)  # wedged consumer: reader holds _emit_lock this whole time
        return True

    engine = TtsSuperWsEngine(
        hivemind_url="http://127.0.0.1:1", voice="alloy", emit=blocking_emit,
        t_start=time.monotonic(), close_timeout=0.05,
    )
    engine._ws = _StuckWs()
    engine._reader = threading.Thread(target=engine._read_loop, name="ms4-tts-ws-reader")
    engine._reader_started = True
    engine._reader.start()
    assert emit_entered.wait(timeout=2.0), "reader must be inside the blocking emit (holding _emit_lock)"

    t0 = time.monotonic()
    try:
        engine.close(timeout=0.05)
    except RuntimeError:
        pass  # a wedged reader is surfaced as a bounded cleanup timeout
    elapsed = time.monotonic() - t0

    engine._reader.join(timeout=3.0)  # let the wedged emit finish and the reader exit

    assert elapsed < 0.5, f"public close() must stay bounded with a stuck emitter; took {elapsed:.3f}s"
    assert graceful, "close() must attempt the graceful RFC6455 ws.close()"
    assert raw_shutdown, "close() must run the raw socket shutdown fallback"
