"""Streaming voice PTT tests.

Two surfaces:

1. ``SentenceChunker`` — pure-function chunking strategy. Covers
   first-chunk acceleration, sentence-boundary splitting, paragraph
   breaks, hard-ceiling chunking, and end-of-stream flush.

2. ``voice_ptt_turn_stream`` — the orchestrator. Uses fake ASR / chat /
   TTS callables so we can exercise the event ordering and parallel TTS
   scheduling without booting HiveMind. The contract we lock in:
     - exactly one ``transcript`` event before any chat tokens.
     - ``text_delta`` events for every token.
     - ``chunk_scheduled`` BEFORE the matching ``audio_chunk`` for each chunk.
     - ``audio_chunk`` events emitted in strict ``index`` order even when
       individual TTS calls complete out of order.
     - the final metrics dict has positive durations, ``audio_chunks ==
       chunk_scheduled count``, and reports ``first_audio_chunk_ms``.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from machine_spirit_4.gateway.voice import (
    SentenceChunker,
    VoiceRequestError,
    voice_ptt_turn_stream,
)


# ---------------------------------------------------------------------------
# SentenceChunker
# ---------------------------------------------------------------------------


def test_chunker_first_chunk_fires_at_min_words_without_period():
    """No sentence terminator yet, but we've crossed the first-chunk
    word threshold — emit the partial chunk now so audio can start
    playing within ~1s of the first token."""
    c = SentenceChunker(first_chunk_min_words=4, max_chunk_words=100)
    out = c.add("Hello there friend, how are you today")
    assert len(out) == 1
    assert out[0].startswith("Hello there friend, how")  # 4 words
    # the buffer still holds the tail
    assert "how" in (out[0] + c.buffer) and "today" in (out[0] + c.buffer)


def test_chunker_subsequent_chunks_fire_on_sentence_boundary():
    c = SentenceChunker(first_chunk_min_words=2)
    c.add("Hi there.")  # first chunk fires either way; min_words=2
    c.add(" Second sentence here. Third one!")
    # After the second/third add we should have at least two new chunks.
    # The chunker emits per call so let's accumulate.
    out_total = []
    out_total += c.add("")  # noop
    # Use a fresh chunker to make ordering easier.
    c2 = SentenceChunker(first_chunk_min_words=100)  # disable first-chunk acceleration
    out = c2.add("First sentence. Second sentence! Third? And more")
    assert out[0].rstrip() == "First sentence."
    assert out[1].rstrip() == "Second sentence!"
    assert out[2].rstrip() == "Third?"
    # The trailing "And more" has no terminator and is below the
    # word-count ceiling, so it stays in the buffer until flush().
    assert "And more" in c2.buffer
    tail = c2.flush()
    assert tail == "And more"


def test_chunker_hard_ceiling_chunks_runaway_unpunctuated_text():
    c = SentenceChunker(first_chunk_min_words=100, max_chunk_words=5)
    out = c.add("one two three four five six seven eight nine ten")
    assert len(out) >= 1
    assert len(out[0].split()) == 5


def test_chunker_paragraph_break_is_a_chunk_boundary():
    c = SentenceChunker(first_chunk_min_words=100)
    out = c.add("A line of text\n\nanother paragraph follows here that has many words to keep going")
    # First emit is the paragraph-1 chunk, then nothing more (no period).
    assert out[0].strip() == "A line of text"


def test_chunker_flush_returns_only_remaining_text():
    c = SentenceChunker(first_chunk_min_words=100)
    c.add("Short one. ")
    # First sentence already emitted; nothing left to flush.
    assert c.flush() is None


# ---------------------------------------------------------------------------
# voice_ptt_turn_stream orchestrator
# ---------------------------------------------------------------------------


class _FakeRunner:
    def __init__(self):
        self.hivemind_url = "http://hive:0"
        self.ms3_url = "http://ms3:0"


def _ms3_voice_ready(monkeypatch):
    """Bypass the MS3 /voice/status fail-closed check."""
    monkeypatch.setattr(
        "machine_spirit_4.gateway.voice.check_voice_ready",
        lambda *a, **k: {"voice_input_ready": True},
    )


def test_stream_emits_events_in_order_with_parallel_tts(monkeypatch):
    _ms3_voice_ready(monkeypatch)

    def fake_transcribe(*, hivemind_url, audio, filename, model):
        time.sleep(0.01)
        return {"text": "What's the date?", "model": "whisper-1"}

    chat_text = "Today is Thursday. May 21st 2026. Anything else?"

    def fake_chat(message, *, session_id=None, model=None, stream_callback=None):
        # Stream out one word at a time with tiny delays so the chunker
        # gets exercised, but stay fast so the test runs sub-second.
        for word in chat_text.split(" "):
            stream_callback(word + " ")
            time.sleep(0.005)
        return {"text": chat_text, "session_id": "sess-1", "metrics": {"duration_ms": 50}}

    # Simulate variable TTS latency: chunks 0 & 1 take longer than chunk 2.
    # This proves the in-order emit guarantee — chunk 2's audio must NOT
    # show up before chunks 0 and 1.
    call_counter = {"n": 0}

    def fake_synthesize(*, hivemind_url, text, model, voice, response_format):
        idx = call_counter["n"]
        call_counter["n"] += 1
        delay = 0.18 if idx == 0 else (0.12 if idx == 1 else 0.04)
        time.sleep(delay)
        return {
            "audio_bytes": b"\x00" * 8,
            "content_type": "audio/wav",
            "audio_base64": f"AUDIO-FOR-{text[:20]!r}",
            "model": "tts-1",
            "voice": "alloy",
            "format": "wav",
        }

    events: list[tuple[str, dict[str, Any]]] = []
    lock = threading.Lock()

    def emit(event, payload):
        with lock:
            events.append((event, dict(payload)))
        return True

    runner = _FakeRunner()
    result = voice_ptt_turn_stream(
        runner=runner,
        audio=b"FAKE_WAV",
        chat_fn=fake_chat,
        transcribe_fn=fake_transcribe,
        synthesize_fn=fake_synthesize,
        emit=emit,
    )

    event_types = [e for e, _ in events]
    # Phase signals.
    assert event_types.count("status") >= 2  # transcribing + thinking
    assert event_types.index("transcript") < event_types.index("text_delta")
    # We saw chat tokens.
    assert "text_delta" in event_types
    # We scheduled and received at least one chunk.
    sched = [p for e, p in events if e == "chunk_scheduled"]
    audio = [p for e, p in events if e == "audio_chunk"]
    assert len(sched) >= 1
    assert len(audio) >= 1
    # In-order emission: indices must be 0, 1, 2, ... contiguous.
    indices = [p["index"] for p in audio]
    assert indices == sorted(indices)
    assert indices == list(range(len(indices)))
    # The metrics dict reports both first-text and first-audio timings.
    assert result["metrics"]["first_text_token_ms"] is not None
    assert result["metrics"]["first_audio_chunk_ms"] is not None
    assert result["metrics"]["audio_chunks"] == len(audio)
    assert result["transcript"] == "What's the date?"


def test_stream_raises_on_empty_transcript(monkeypatch):
    _ms3_voice_ready(monkeypatch)

    def fake_transcribe(*, hivemind_url, audio, filename, model):
        return {"text": "", "model": "whisper-1"}

    events: list[tuple[str, dict[str, Any]]] = []

    def emit(event, payload):
        events.append((event, dict(payload)))
        return True

    runner = _FakeRunner()
    with pytest.raises(VoiceRequestError):
        voice_ptt_turn_stream(
            runner=runner,
            audio=b"FAKE_WAV",
            chat_fn=lambda *a, **k: {"text": "", "session_id": "s"},
            transcribe_fn=fake_transcribe,
            synthesize_fn=lambda **k: {"audio_base64": "", "content_type": "audio/wav"},
            emit=emit,
        )
    # The transcript event was still emitted with the empty text so the
    # client knows ASR finished.
    assert any(e == "transcript" for e, _ in events)


def test_metrics_report_per_chunk_timings_and_parallelism(monkeypatch):
    """The metrics block should let the operator see:
      - per-chunk TTS latency (scheduled_ms / tts_completed_ms / tts_ms)
      - whether the in-order emit guarantee held a chunk (held_for_inorder_ms > 0)
      - aggregate speedup from running HiveMind TTS in parallel
    """
    _ms3_voice_ready(monkeypatch)

    def fake_transcribe(**_):
        return {"text": "Tell me three things.", "model": "whisper-1"}

    def fake_chat(message, *, session_id=None, model=None, stream_callback=None):
        for s in ("First short. ", "Second medium length sentence here. ", "Third one. "):
            for w in s.split():
                stream_callback(w + " ")
        return {"text": "First short. Second medium length sentence here. Third one.", "session_id": "s"}

    # First TTS call deliberately slow, so chunk 0 holds back chunks 1 and 2
    # that complete much earlier — this is the exact pattern the user
    # observed live where chunks 1+2 emit at the same timestamp.
    call_counter = {"n": 0}

    def fake_synthesize(**kw):
        n = call_counter["n"]
        call_counter["n"] += 1
        time.sleep(0.20 if n == 0 else 0.03)
        return {
            "audio_base64": f"AUDIO-{n}",
            "content_type": "audio/wav",
            "audio_bytes": b"\x00",
            "model": "tts-1",
            "voice": "alloy",
            "format": "wav",
        }

    runner = _FakeRunner()
    result = voice_ptt_turn_stream(
        runner=runner,
        audio=b"FAKE_WAV",
        chat_fn=fake_chat,
        transcribe_fn=fake_transcribe,
        synthesize_fn=fake_synthesize,
        emit=lambda e, p: True,
    )

    m = result["metrics"]
    chunks = m.get("chunks") or []
    assert len(chunks) == 3, f"expected 3 chunks, got {len(chunks)}"
    # Every per-chunk record has the fields we promised.
    for c in chunks:
        assert {"index", "text_len", "scheduled_ms", "tts_completed_ms", "emitted_ms", "tts_ms", "held_for_inorder_ms"} <= set(c.keys())
        assert c["tts_ms"] >= 0
        assert c["held_for_inorder_ms"] >= 0
    # The slow chunk 0 has near-zero hold; later chunks completed well
    # before chunk 0 emitted, so they were held.
    held = [c for c in chunks if c["index"] > 0]
    assert any(c["held_for_inorder_ms"] > 50 for c in held), \
        "later chunks should have been held by in-order emit; got " + str(held)
    # Parallelism summary present and indicates concurrency (sum of TTS
    # times noticeably > wall-clock TTS window).
    tp = m.get("tts_parallelism") or {}
    assert tp.get("speedup_ratio", 0) > 1.2, f"expected >1.2x speedup, got {tp}"
    assert tp.get("any_held_for_inorder") is True


def test_voice_stream_engine_ws_super_routes_through_ws_engine(monkeypatch):
    """When engine='ws_super' is selected, voice_ptt_turn_stream MUST
    delegate TTS to a TtsSuperWsEngine instance instead of running the
    REST sentence-chunked path. We assert via a fake engine factory
    that the SSE event shape stays identical (the UI must work
    unchanged) and that the final metrics declare engine='ws_super'."""
    _ms3_voice_ready(monkeypatch)

    def fake_transcribe(**_):
        return {"text": "Tell me three things.", "model": "whisper-1"}

    def fake_chat(message, *, session_id=None, model=None, stream_callback=None):
        for word in "First sentence. Second sentence. Third one.".split():
            stream_callback(word + " ")
        return {"text": "First sentence. Second sentence. Third one.", "session_id": "s"}

    class _FakeWsEngine:
        def __init__(self, *, hivemind_url, voice, emit, t_start, **_):
            self.emit = emit
            self.t_start = t_start
            self.voice = voice
            self.pushed: list[str] = []
            self.flushed = False
            self.closed = False
            self._chunks_emitted = 0
            self._first_audio_ms = None

        def open(self):
            self.emit("tts_engine", {"engine": "ws_super", "voice": self.voice, "connect_ms": 5})

        def push(self, text):
            self.pushed.append(text)
            # emit a fake audio chunk every 3 pushes so the test sees
            # multiple ordered audio_chunk events arriving while chat
            # is still streaming
            if len(self.pushed) % 3 == 0:
                if self._first_audio_ms is None:
                    self._first_audio_ms = int((time.time() - self.t_start) * 1000)
                self.emit("audio_chunk", {
                    "index": self._chunks_emitted,
                    "text": "",
                    "audio_base64": "AAA",
                    "audio_mime": "audio/wav",
                    "tts_ms": int((time.time() - self.t_start) * 1000),
                    "held_for_inorder_ms": 0,
                    "engine": "ws_super",
                })
                self._chunks_emitted += 1

        def flush(self):
            self.flushed = True

        def wait_for_final(self, timeout=60):
            return True

        def close(self):
            self.closed = True

        def metrics(self):
            return {
                "engine": "ws_super",
                "voice": self.voice,
                "connect_ms": 5,
                "first_audio_ms": self._first_audio_ms,
                "chunks_emitted": self._chunks_emitted,
                "bytes_received": 100 * max(1, self._chunks_emitted),
                "is_final_seen": True,
                "error": None,
            }

    events: list[tuple[str, dict]] = []

    def emit(name, payload):
        events.append((name, dict(payload)))
        return True

    runner = _FakeRunner()
    result = voice_ptt_turn_stream(
        runner=runner,
        audio=b"FAKE_WAV",
        chat_fn=fake_chat,
        transcribe_fn=fake_transcribe,
        synthesize_fn=lambda **_: {"audio_base64": "", "content_type": "audio/wav", "model": "tts-1", "voice": "alloy", "format": "wav"},
        emit=emit,
        engine="ws_super",
        ws_engine_factory=_FakeWsEngine,
    )

    event_types = [e for e, _ in events]
    assert "tts_engine" in event_types, "ws_super engine must emit tts_engine on open"
    audio_chunks = [p for e, p in events if e == "audio_chunk"]
    assert len(audio_chunks) >= 1, "expected at least one audio_chunk from the fake ws engine"
    # Same payload shape as the REST engine — the UI consumes both unchanged.
    for p in audio_chunks:
        assert {"index", "text", "audio_base64", "audio_mime", "tts_ms", "held_for_inorder_ms", "engine"} <= set(p.keys())
        assert p["engine"] == "ws_super"
        assert p["held_for_inorder_ms"] == 0  # WS path is naturally ordered
    # Final metrics block declares the engine and zero parallel-related fields.
    m = result["metrics"]
    assert m["engine"] == "ws_super"
    assert m["tts_engine"]["engine"] == "ws_super"
    assert m["audio_chunks"] == len(audio_chunks)
    assert m["tts_parallelism"]["any_held_for_inorder"] is False


def test_ws_engine_skips_wait_for_final_when_client_disconnects(monkeypatch):
    """Barge-in plumbing: when the client aborts the SSE fetch
    mid-turn, the gateway clears ``client_alive`` and
    voice_ptt_turn_stream MUST stop waiting for HiveMind TTS to
    finish. Without this, every barged turn would burn the full 60s
    wait_for_final budget on audio nobody is listening to."""
    _ms3_voice_ready(monkeypatch)

    def fake_transcribe(**_):
        return {"text": "tell me a story", "model": "whisper-1"}

    def fake_chat(message, *, session_id=None, model=None, stream_callback=None):
        for w in "Once upon a time there was a long reply.".split():
            stream_callback(w + " ")
        return {"text": "Once upon a time there was a long reply.", "session_id": "s"}

    wait_calls = {"n": 0, "close_at_n_wait_calls": None}

    class _SlowWsEngine:
        def __init__(self, *, hivemind_url, voice, emit, t_start, **_):
            self.emit = emit
            self.voice = voice
            self.t_start = t_start
            self._closed = False

        def open(self):
            self.emit("tts_engine", {"engine": "ws_super", "voice": self.voice, "connect_ms": 1})

        def push(self, _text):
            pass

        def flush(self):
            pass

        def wait_for_final(self, timeout=60):
            wait_calls["n"] += 1
            return True

        def close(self):
            self._closed = True

        def metrics(self):
            return {"engine": "ws_super", "voice": self.voice, "connect_ms": 1,
                    "first_audio_ms": None, "chunks_emitted": 0, "bytes_received": 0,
                    "is_final_seen": False, "error": None}

    client_alive = threading.Event()
    client_alive.set()
    client_alive.clear()  # simulate the gateway clearing it BEFORE the engine path runs wait_for_final

    runner = _FakeRunner()
    voice_ptt_turn_stream(
        runner=runner,
        audio=b"FAKE",
        chat_fn=fake_chat,
        transcribe_fn=fake_transcribe,
        synthesize_fn=lambda **_: {"audio_base64": "", "content_type": "audio/wav"},
        emit=lambda _e, _p: True,
        engine="ws_super",
        ws_engine_factory=_SlowWsEngine,
        client_alive=client_alive,
    )

    assert wait_calls["n"] == 0, (
        "wait_for_final must be skipped when client_alive is cleared "
        "(barge-in); otherwise every barged turn burns the full TTS wait"
    )


def test_ws_engine_calls_wait_for_final_when_client_still_alive(monkeypatch):
    """Inverse of the previous test: when client_alive stays set, the
    engine path runs wait_for_final exactly once."""
    _ms3_voice_ready(monkeypatch)

    def fake_transcribe(**_):
        return {"text": "hi", "model": "whisper-1"}

    def fake_chat(message, *, session_id=None, model=None, stream_callback=None):
        stream_callback("ok")
        return {"text": "ok", "session_id": "s"}

    wait_calls = {"n": 0}

    class _CountingWsEngine:
        def __init__(self, *, hivemind_url, voice, emit, t_start, **_):
            self.emit = emit
            self.voice = voice

        def open(self):
            self.emit("tts_engine", {"engine": "ws_super", "voice": self.voice, "connect_ms": 1})

        def push(self, _text):
            pass

        def flush(self):
            pass

        def wait_for_final(self, timeout=60):
            wait_calls["n"] += 1
            return True

        def close(self):
            pass

        def metrics(self):
            return {"engine": "ws_super", "voice": self.voice, "connect_ms": 1,
                    "first_audio_ms": None, "chunks_emitted": 0, "bytes_received": 0,
                    "is_final_seen": True, "error": None}

    client_alive = threading.Event()
    client_alive.set()

    voice_ptt_turn_stream(
        runner=_FakeRunner(),
        audio=b"FAKE",
        chat_fn=fake_chat,
        transcribe_fn=fake_transcribe,
        synthesize_fn=lambda **_: {"audio_base64": "", "content_type": "audio/wav"},
        emit=lambda _e, _p: True,
        engine="ws_super",
        ws_engine_factory=_CountingWsEngine,
        client_alive=client_alive,
    )
    assert wait_calls["n"] == 1


def test_voice_stream_invalid_engine_falls_back_to_rest(monkeypatch):
    _ms3_voice_ready(monkeypatch)

    def fake_transcribe(**_):
        return {"text": "Hi.", "model": "whisper-1"}

    def fake_chat(message, *, session_id=None, model=None, stream_callback=None):
        stream_callback("ok")
        return {"text": "ok", "session_id": "s"}

    events: list[tuple[str, dict]] = []
    result = voice_ptt_turn_stream(
        runner=_FakeRunner(),
        audio=b"FAKE_WAV",
        chat_fn=fake_chat,
        transcribe_fn=fake_transcribe,
        synthesize_fn=lambda **_: {"audio_base64": "AAA", "content_type": "audio/wav"},
        emit=lambda n, p: events.append((n, dict(p))) or True,
        engine="garbage_engine_name",
    )
    # Falls back to the REST path: no tts_engine open event, no engine key in
    # top-level metrics (the REST schema doesn't set it).
    assert "tts_engine" not in [e for e, _ in events]
    assert result["metrics"].get("engine") != "ws_super"


def test_stream_first_chunk_arrives_before_chat_finishes(monkeypatch):
    """The whole point of streaming TTS: the FIRST audio chunk should
    show up while the model is still generating later tokens, not after
    the whole reply is complete."""
    _ms3_voice_ready(monkeypatch)

    transcribe_done_at = []
    first_audio_emitted_at: list[float] = []
    chat_done_at: list[float] = []

    def fake_transcribe(**_):
        transcribe_done_at.append(time.monotonic())
        return {"text": "Tell me a long story.", "model": "whisper-1"}

    def fake_chat(message, *, session_id=None, model=None, stream_callback=None):
        # Reply has 3 sentences with a meaningful delay between them.
        sentences = [
            "First sentence happens fast.",
            "Second sentence happens after a longer pause.",
            "Third sentence wraps it up.",
        ]
        for i, s in enumerate(sentences):
            for w in s.split():
                stream_callback(w + " ")
                time.sleep(0.002)
            stream_callback(" ")
            if i < len(sentences) - 1:
                time.sleep(0.20)  # the chat takes a real-world-ish pause
        chat_done_at.append(time.monotonic())
        return {"text": " ".join(sentences), "session_id": "s"}

    def fake_synthesize(**kw):
        time.sleep(0.05)  # short TTS
        return {
            "audio_base64": "AAA",
            "content_type": "audio/wav",
            "audio_bytes": b"\x00",
            "model": "tts-1",
            "voice": "alloy",
            "format": "wav",
        }

    def emit(event, payload):
        if event == "audio_chunk" and not first_audio_emitted_at:
            first_audio_emitted_at.append(time.monotonic())
        return True

    runner = _FakeRunner()
    voice_ptt_turn_stream(
        runner=runner,
        audio=b"FAKE_WAV",
        chat_fn=fake_chat,
        transcribe_fn=fake_transcribe,
        synthesize_fn=fake_synthesize,
        emit=emit,
    )
    assert first_audio_emitted_at, "expected at least one audio_chunk before chat finished"
    assert chat_done_at, "fake chat should have ended"
    # First audio chunk MUST come strictly before chat finishes —
    # that's the latency win the whole architecture is for.
    assert first_audio_emitted_at[0] < chat_done_at[0], (
        f"first audio at {first_audio_emitted_at[0]}, chat ended at {chat_done_at[0]}"
    )
