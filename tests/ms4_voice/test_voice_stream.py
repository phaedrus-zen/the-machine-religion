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

import base64
import io
import json
import queue
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from machine_spirit_4.gateway.voice import (
    FaceLobeStalled,
    SentenceChunker,
    VoiceRequestError,
    VoiceUnavailable,
    _call_tts_with_runtime_audio_gate_recovery,
    _gate_chunk_audio,
    _identify_speaker_async,
    _identify_speaker_bounded,
    _runtime_audio_gate_measurement,
    _run_facechat_guarded,
    _voice_speaker_id_async,
    expected_speech_secs,
    last_face_model,
    record_face_model,
    trim_wav,
    voice_ptt_turn_stream,
    wav_duration_secs,
)


# ---------------------------------------------------------------------------
# SentenceChunker
# ---------------------------------------------------------------------------


def test_streaming_voice_defaults_to_rest_with_ws_super_available(monkeypatch):
    import machine_spirit_4.gateway.voice as voice_module

    # 2026-07-06 audio-stutter repair: the production/default path must be REST.
    # Production evidence showed ws_super produced audio slower than real time
    # (12 chunks arriving 984-1391 ms apart), starving the browser scheduler into
    # 10 gaps / 4.42 s. REST stayed ahead of playback (0 gaps). ws_super remains a
    # deliberate per-turn diagnostic opt-in.
    #
    # WS pinning is now explicit/per-test (the ws-path tests request the
    # pin_ws_super_engine fixture), so the DEFAULT_ENGINE constant is no longer
    # masked by an autouse force. With the env unset both the constant and the
    # fallback helper must resolve to REST.
    monkeypatch.delenv("MS4_VOICE_TTS_ENGINE", raising=False)
    assert voice_module.DEFAULT_ENGINE == "rest"
    assert voice_module._default_tts_engine() == "rest"
    assert voice_module.VALID_ENGINES == {"rest", "ws_super"}


def test_rest_streaming_voice_brevity_opt_in_passes_voice_mode(monkeypatch):
    import machine_spirit_4.gateway.voice as voice_module

    _ms3_voice_ready(monkeypatch)
    monkeypatch.setenv("MS4_VOICE_BREVITY", "1")
    monkeypatch.setenv("MS4_VOICE_SMART_REFLEX", "0")
    monkeypatch.setenv("MS4_VOICE_EGG_HONORABLE", "0")
    monkeypatch.setattr(voice_module, "_identify_speaker_async", lambda *_a, **_k: None)
    captured: dict[str, object] = {}

    def fake_chat(_message, *, stream_callback=None, **kwargs):
        captured.update(kwargs)
        reply = "The complete bounded answer is ready for the operator now."
        assert stream_callback(reply + " ") is not False
        return {"text": reply, "session_id": "voice-brevity-rest"}

    voice_ptt_turn_stream(
        runner=_FakeRunner(),
        audio=b"FAKE_WAV",
        chat_fn=fake_chat,
        transcribe_fn=lambda **_kwargs: {
            "text": "Give me the complete explanation.",
            "model": "whisper-1",
        },
        synthesize_fn=lambda **_kwargs: {
            "audio_bytes": b"WAV",
            "audio_base64": "V0FW",
            "content_type": "audio/wav",
        },
        emit=lambda _event, _payload: True,
        engine="rest",
    )

    assert captured["voice_mode"] is True


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


def test_chunker_batched_long_sentence_keeps_first_chunk_at_startup_threshold():
    c = SentenceChunker(first_chunk_min_words=5, min_chunk_words=10, max_chunk_words=40)
    c.first_chunk_sanitizer = lambda text: text

    out = c.add(
        "one two three four five six seven eight nine ten eleven twelve. "
        "thirteen fourteen, "
    )

    assert out[0] == "one two three four five"


def test_chunker_never_counts_an_unterminated_stream_fragment_as_a_word():
    c = SentenceChunker(first_chunk_min_words=5, max_chunk_words=100)

    assert c.add("Here is a useful an") == []
    assert c.buffer.endswith("an")
    out = c.add("swer. More text follows")

    assert out == ["Here is a useful answer."]
    assert "an swer" not in " ".join(out + [c.buffer])


def test_chunker_startup_pair_uses_fast_lead_then_normal_runway_floor():
    words = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty twentyone twentytwo twentythree twentyfour twentyfive"
    chunker = SentenceChunker(
        first_chunk_min_words=5,
        min_chunk_words=10,
        max_chunk_words=30,
    )

    chunks = chunker.add(words + " ")
    tail = chunker.flush()
    if tail:
        chunks.append(tail)

    assert [len(chunk.split()) for chunk in chunks] == [5, 10, 10]
    assert " ".join(chunks).split() == words.split()


@pytest.mark.parametrize(
    "reply, expected_first_words",
    (
        (
            "I revise the recommendation: the governing constraint changes from "
            "first-token latency under four seconds to end-to-end voice latency "
            "under four seconds.",
            6,
        ),
        (
            "1. Direct answer The final architecture keeps the Face lobe on the "
            "immediate spoken path and the Depth lobe on the asynchronous evidence path.",
            5,
        ),
    ),
)
def test_chunker_q15_startup_pair_prefers_bounded_runway_before_long_sentence(
    reply,
    expected_first_words,
):
    chunker = SentenceChunker(
        first_chunk_min_words=5,
        min_chunk_words=10,
        max_chunk_words=40,
    )
    chunker.first_chunk_sanitizer = lambda text: text

    chunks = chunker.add(reply + " ")
    tail = chunker.flush()
    if tail:
        chunks.append(tail)

    assert len(chunks[0].split()) == expected_first_words
    assert len(chunks[1].split()) == chunker.min_chunk_words
    assert " ".join(chunks).split() == reply.split()


def test_chunker_q15_turn2_opener_uses_five_ten_nine_word_runway_shape():
    """The rejected Q15 take exposed a real c1->c2 playback underflow.

    Two five-word startup fragments supplied only 4.021 seconds of decoded
    speech while the 14-word third fragment arrived 4.901 seconds later. Keep
    the fast five-word lead, but put ten words in the second request so the
    remaining TTS tail is both shorter and covered by useful playback runway.
    """
    reply = (
        "The compute cost delta is primarily driven by three structural factors "
        "in the 35B Depth lobe's architecture: context retention, cross-node "
        "correlation, and sampling overhead."
    )
    chunker = SentenceChunker(
        first_chunk_min_words=5,
        min_chunk_words=10,
        max_chunk_words=40,
    )
    chunker.first_chunk_sanitizer = lambda text: text

    chunks = chunker.add(reply + " ")
    tail = chunker.flush()
    if tail:
        chunks.append(tail)

    assert [len(chunk.split()) for chunk in chunks] == [5, 10, 9]
    assert chunks[0] == "The compute cost delta is"
    assert chunks[1] == "primarily driven by three structural factors in the 35B Depth"
    assert " ".join(chunks).split() == reply.split()


def test_chunker_hard_ceiling_keeps_a_split_token_buffered():
    c = SentenceChunker(first_chunk_min_words=100, max_chunk_words=5)

    assert c.add("one two three four fi") == []
    out = c.add("ve six ")

    assert out == ["one two three four five"]
    assert c.buffer == "six "


def test_chunker_subsequent_long_sentences_fire_individually():
    # Sentences at/above min_chunk_words still fire one-per-boundary.
    c2 = SentenceChunker(first_chunk_min_words=100, min_chunk_words=3)
    out = c2.add("First long sentence here. Second long sentence here! Third long one? And more")
    assert out[0].rstrip() == "First long sentence here."
    assert out[1].rstrip() == "Second long sentence here!"
    assert out[2].rstrip() == "Third long one?"
    # The trailing "And more" has no terminator and is below the
    # word-count ceiling, so it stays in the buffer until flush().
    assert "And more" in c2.buffer
    assert c2.flush() == "And more"


def test_chunker_coalesces_short_subsequent_sentences():
    # After the first chunk, short sentences merge up to min_chunk_words
    # instead of firing one tiny TTS call each.
    c = SentenceChunker(first_chunk_min_words=100, min_chunk_words=6)
    out = c.add("Opening statement that is plenty long. Yes. No. Maybe. ")
    # First chunk ships the opening sentence.
    assert out[0].rstrip() == "Opening statement that is plenty long."
    # "Yes. No. Maybe." are each <6 words -> coalesced (not 3 chunks).
    later = out[1:] + c.add("") 
    # At most ONE more chunk so far (the short trio is still coalescing or
    # already merged); definitely not three separate tiny chunks.
    assert len([x for x in later if x]) <= 1
    tail = c.flush()
    # Whatever remains flushes as the merged short fragment.
    assert tail is None or ("Yes" in tail and "Maybe" in tail)


def test_chunker_coalescing_disabled_with_min_one():
    # min_chunk_words=1 restores per-sentence chunking.
    c = SentenceChunker(first_chunk_min_words=100, min_chunk_words=1)
    out = c.add("First one. Second two! Third three? ")
    assert out[0].rstrip() == "First one."
    assert out[1].rstrip() == "Second two!"
    assert out[2].rstrip() == "Third three?"


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
    """Bypass MS3 readiness and annotate legacy chat doubles as complete.

    The production Face Lobe now returns an explicit ``completed`` bit.  Most
    tests in this file predate that contract and exercise audio scheduling,
    not chat termination, so keep those doubles concise while preserving the
    strict production guard.  Completion-specific tests call the imported
    ``_run_facechat_guarded`` directly and are therefore not adapted here.
    """
    import machine_spirit_4.gateway.voice as voice_module

    monkeypatch.setattr(
        "machine_spirit_4.gateway.voice.check_voice_ready",
        lambda *a, **k: {"voice_input_ready": True},
    )
    original_guard = voice_module._run_facechat_guarded

    def guard_with_verified_test_double(*, chat_call, **kwargs):
        def verified_chat_call(*args, **chat_kwargs):
            response = chat_call(*args, **chat_kwargs)
            if isinstance(response, dict) and "completed" not in response:
                response = {**response, "completed": True}
            return response

        return original_guard(chat_call=verified_chat_call, **kwargs)

    monkeypatch.setattr(voice_module, "_run_facechat_guarded", guard_with_verified_test_double)


@pytest.mark.parametrize(
    ("raw_transcript", "canonical_transcript"),
    [
        (
            "Use HiveMiner, then use the authoritative HiveMine GPU availability tool.",
            "Use HiveMiner, then use the authoritative HiveMind GPU availability tool.",
        ),
        (
            "Keep Hive Mindset unchanged; use the authoritative Hive Mine GPU availability tool.",
            "Keep Hive Mindset unchanged; use the authoritative HiveMind GPU availability tool.",
        ),
    ],
    ids=("hivemine", "hive-mine"),
)
def test_voice_stream_normalizes_spoken_brand_alias_at_shared_boundary(
    monkeypatch, raw_transcript, canonical_transcript
):
    import machine_spirit_4.gateway.voice as voice_module

    _ms3_voice_ready(monkeypatch)
    monkeypatch.setenv("MS4_VOICE_SMART_REFLEX", "0")
    monkeypatch.setenv("MS4_VOICE_EGG_HONORABLE", "0")
    monkeypatch.setattr(voice_module, "_identify_speaker_async", lambda *_a, **_k: None)
    original_boundary = getattr(
        voice_module,
        "_ptt_transcript_from_asr",
        lambda result: (str(result.get("text") or "").strip(), str(result.get("text") or "")),
    )
    boundary_calls = []

    def tracked_boundary(result):
        boundary_calls.append(str(result.get("text") or ""))
        return original_boundary(result)

    monkeypatch.setattr(voice_module, "_ptt_transcript_from_asr", tracked_boundary, raising=False)
    downstream = {}

    def fake_chat(message, *, stream_callback=None, **_kwargs):
        downstream["message"] = message
        assert stream_callback("The authoritative result is ready. ") is not False
        return {"text": "The authoritative result is ready.", "session_id": "s-alias"}

    events: list[tuple[str, dict[str, Any]]] = []
    result = voice_ptt_turn_stream(
        runner=_FakeRunner(),
        audio=b"FAKE_WAV",
        chat_fn=fake_chat,
        transcribe_fn=lambda **_kwargs: {"text": raw_transcript, "model": "whisper-1"},
        synthesize_fn=lambda **_kwargs: {
            "audio_bytes": b"WAV",
            "audio_base64": "V0FW",
            "content_type": "audio/wav",
        },
        emit=lambda event, payload: events.append((event, dict(payload))) or True,
        engine="rest",
    )

    transcript_event = next(payload for event, payload in events if event == "transcript")
    assert downstream["message"] == canonical_transcript
    assert boundary_calls == [raw_transcript]
    assert transcript_event["text"] == canonical_transcript
    assert transcript_event["raw_text"] == raw_transcript
    assert result["transcript"] == canonical_transcript
    assert result["raw_transcript"] == raw_transcript


def test_stream_emits_events_in_order_with_parallel_tts(monkeypatch):
    _ms3_voice_ready(monkeypatch)

    def fake_transcribe(*, hivemind_url, audio, filename, model):
        time.sleep(0.01)
        return {"text": "What's the date?", "model": "whisper-1"}

    chat_text = "Today is Thursday. May 21st 2026. Anything else?"

    def fake_chat(message, *, session_id=None, model=None, stream_callback=None, **_):
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
        engine="rest",
    )

    event_types = [e for e, _ in events]
    # Phase signals.
    assert event_types.count("status") >= 2  # transcribing + thinking
    assert event_types.index("transcript") < event_types.index("text_delta")
    thinking = next(
        payload for event, payload in events
        if event == "status" and payload.get("phase") == "thinking"
    )
    assert thinking["engine"] == "rest"
    assert thinking["rest_capacity_effective"] >= 1
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
    assert result["metrics"]["engine"] == "rest"
    assert result["metrics"]["first_text_token_ms"] is not None
    assert result["metrics"]["first_audio_chunk_ms"] is not None
    assert result["metrics"]["audio_chunks"] == len(audio)
    assert result["transcript"] == "What's the date?"


def test_voice_stream_advertises_observed_one_not_configured_two(monkeypatch):
    """The early browser scheduling fact must come from live scale evidence."""

    import machine_spirit_4.gateway.voice as voice_module

    _ms3_voice_ready(monkeypatch)
    monkeypatch.setenv("MS4_VOICE_TTS_CAPACITY", "2")
    monkeypatch.setenv("MS4_VOICE_SMART_REFLEX", "0")
    monkeypatch.setenv("MS4_VOICE_EGG_HONORABLE", "0")
    monkeypatch.setenv("MS4_VOICE_RUNTIME_QA", "0")
    monkeypatch.setattr(voice_module, "_identify_speaker_async", lambda *_a, **_k: None)
    voice_module._record_voice_rest_capacity_observation({
        "ok": True,
        "status": "ok",
        "target": 2,
        "replicas": 1,
    })

    reply = "One observed replica keeps the conservative playback policy active."

    def fake_chat(_message, *, stream_callback=None, **_kwargs):
        assert stream_callback is not None
        assert stream_callback(reply + " ") is not False
        return {"text": reply, "session_id": "observed-capacity-one"}

    monkeypatch.setattr(
        voice_module,
        "synthesize",
        lambda **_kwargs: {
            "audio_bytes": b"WAV",
            "audio_base64": "V0FW",
            "content_type": "audio/wav",
        },
    )
    events: list[tuple[str, dict[str, Any]]] = []

    result = voice_ptt_turn_stream(
        runner=_FakeRunner(),
        audio=b"FAKE_WAV",
        chat_fn=fake_chat,
        transcribe_fn=lambda **_kwargs: {
            "text": "Explain the current replica capacity.",
            "model": "whisper-1",
        },
        emit=lambda event, payload: events.append((event, dict(payload))) or True,
        engine="rest",
        tts_pool_size=2,
    )

    thinking_index, thinking = next(
        (index, payload)
        for index, (event, payload) in enumerate(events)
        if event == "status" and payload.get("phase") == "thinking"
    )
    first_audio_index = next(
        index for index, (event, _payload) in enumerate(events)
        if event == "audio_chunk"
    )
    assert thinking_index < first_audio_index
    assert thinking["engine"] == "rest"
    assert thinking["rest_capacity_effective"] == 1
    assert thinking["rest_capacity_provenance"] == "scale_endpoint"
    assert result["metrics"]["lifecycle"]["rest_capacity_effective"] == 1
    assert result["metrics"]["lifecycle"]["rest_capacity_provenance"] \
        == "scale_endpoint"


def test_rest_hedges_only_blocked_next_inorder_chunk(monkeypatch):
    import machine_spirit_4.gateway.voice as voice_module

    _ms3_voice_ready(monkeypatch)
    monkeypatch.setenv("MS4_TTS_REST_HOL_HEDGE_DELAY_S", "0")
    monkeypatch.setenv("MS4_VOICE_FIRST_CHUNK_DEADLINE_S", "0")
    monkeypatch.setenv("MS4_VOICE_SMART_REFLEX", "0")
    monkeypatch.setenv("MS4_VOICE_EGG_HONORABLE", "0")
    monkeypatch.setenv("MS4_VOICE_RUNTIME_QA", "0")
    monkeypatch.setattr(voice_module, "TTS_FILTER_ENABLED", False)
    monkeypatch.setattr(voice_module, "_runtime_audio_gate_enabled", lambda: False)
    monkeypatch.setattr(voice_module, "_identify_speaker_async", lambda *_a, **_k: None)

    pieces = (
        "First audio arrives without delay.",
        "Blocked second chunk needs recovery.",
        "Later third chunk finishes promptly.",
        "Final fourth chunk also finishes.",
    )
    reply = " ".join(pieces)
    attempts: dict[str, int] = {}
    attempts_lock = threading.Lock()
    blocked_primary_started = threading.Event()
    blocked_primary_cancelled = threading.Event()
    later_chunk_finished = threading.Event()
    hedge_started = threading.Event()
    turn_cancel = threading.Event()
    events: list[tuple[str, dict[str, Any]]] = []
    results: list[dict[str, Any]] = []
    errors: list[BaseException] = []

    def fake_synthesize(
        *,
        text,
        cancel_event,
        on_request_dispatched=None,
        **_kwargs,
    ):
        with attempts_lock:
            attempt = attempts.get(text, 0) + 1
            attempts[text] = attempt
        if on_request_dispatched is not None:
            on_request_dispatched()
        if text == pieces[1] and attempt == 1:
            blocked_primary_started.set()
            assert cancel_event.wait(timeout=2.0)
            blocked_primary_cancelled.set()
            raise VoiceUnavailable("blocked primary lost the bounded hedge")
        if text == pieces[1]:
            hedge_started.set()
        if text in pieces[2:]:
            later_chunk_finished.set()
        return {
            "audio_bytes": f"WAV-{text}".encode(),
            "audio_base64": f"AUDIO-{text}",
            "content_type": "audio/wav",
        }

    def emit(event, payload):
        events.append((event, dict(payload)))
        return True

    def run_turn():
        try:
            results.append(voice_ptt_turn_stream(
                runner=_FakeRunner(),
                audio=b"FAKE_WAV",
                chat_fn=lambda _message, stream_callback=None, **_kwargs: (
                    stream_callback(reply + " "),
                    {"text": reply, "session_id": "hol-hedge"},
                )[1],
                transcribe_fn=lambda **_kwargs: {
                    "text": "speak four chunks",
                    "model": "whisper-1",
                },
                synthesize_fn=fake_synthesize,
                emit=emit,
                chunker=SentenceChunker(
                    first_chunk_min_words=5,
                    min_chunk_words=1,
                    max_chunk_words=20,
                ),
                tts_pool_size=4,
                engine="rest",
                cancel_event=turn_cancel,
            ))
        except BaseException as exc:
            errors.append(exc)

    turn = threading.Thread(target=run_turn, name="rest-hol-hedge-test")
    turn.start()
    hedge_observed = False
    try:
        assert blocked_primary_started.wait(timeout=2.0)
        assert later_chunk_finished.wait(timeout=2.0)
        hedge_observed = hedge_started.wait(timeout=0.5)
    finally:
        if not hedge_observed:
            turn_cancel.set()
        turn.join(timeout=2.0)

    assert hedge_observed, (
        "a later-ready chunk must trigger one bounded hedge for the blocked "
        "next-in-order synthesis"
    )
    assert not turn.is_alive()
    assert errors == []
    assert len(results) == 1
    assert blocked_primary_cancelled.is_set()
    assert attempts == {
        pieces[0]: 1,
        pieces[1]: 2,
        pieces[2]: 1,
        pieces[3]: 1,
    }
    scheduled = [payload for event, payload in events if event == "chunk_scheduled"]
    audio = [payload for event, payload in events if event == "audio_chunk"]
    assert [payload["index"] for payload in scheduled] == [0, 1, 2, 3]
    assert [payload["index"] for payload in audio] == [0, 1, 2, 3]
    assert [payload["text"] for payload in audio] == list(pieces)
    assert len(audio) == len({payload["index"] for payload in audio})
    assert not any(event == "audio_error" for event, _payload in events)


def _configure_rest_hol_attempt(monkeypatch, voice_module, *, delay_s: float) -> None:
    _ms3_voice_ready(monkeypatch)
    monkeypatch.setenv("MS4_TTS_REST_HOL_HEDGE_DELAY_S", str(delay_s))
    monkeypatch.setenv("MS4_VOICE_FIRST_CHUNK_DEADLINE_S", "0")
    monkeypatch.setenv("MS4_VOICE_SMART_REFLEX", "0")
    monkeypatch.setenv("MS4_VOICE_EGG_HONORABLE", "0")
    monkeypatch.setenv("MS4_VOICE_RUNTIME_QA", "0")
    monkeypatch.setattr(voice_module, "TTS_FILTER_ENABLED", False)
    monkeypatch.setattr(voice_module, "_runtime_audio_gate_enabled", lambda: False)
    monkeypatch.setattr(voice_module, "_identify_speaker_async", lambda *_a, **_k: None)


def _hol_audio(text: str) -> dict[str, Any]:
    return {
        "audio_bytes": f"WAV-{text}".encode(),
        "audio_base64": f"AUDIO-{text}",
        "content_type": "audio/wav",
    }


def test_rest_hol_fatal_default_covers_measured_long_chunk_and_stays_bounded(
    monkeypatch,
):
    import machine_spirit_4.gateway.voice as voice_module

    monkeypatch.delenv("MS4_TTS_REST_HOL_FATAL_TIMEOUT_S", raising=False)

    default_s = voice_module._voice_rest_hol_fatal_timeout()
    assert default_s == pytest.approx(55.0)
    assert default_s >= 41.187 * 1.3, (
        "the default needs safety margin above the isolated live synthesis "
        "of the exact 18-word chunk rejected by the former 20s ceiling"
    )
    assert default_s < voice_module._voice_rest_progress_timeout(), (
        "a genuinely stuck ordered head must still terminate before the "
        "separate REST progress-stall ceiling"
    )


def test_rest_hol_fatal_window_starts_when_later_audio_becomes_ready(monkeypatch):
    import machine_spirit_4.gateway.voice as voice_module

    _configure_rest_hol_attempt(monkeypatch, voice_module, delay_s=0)
    fatal_timeout_s = 0.2
    monkeypatch.setenv("MS4_TTS_REST_HOL_FATAL_TIMEOUT_S", str(fatal_timeout_s))
    monkeypatch.setenv("MS4_TTS_REST_CLEANUP_TIMEOUT_S", "0.75")
    monkeypatch.setenv("MS4_TTS_REST_FINAL_TIMEOUT_S", "5")

    pieces = (
        "Chunk zero reaches listener first.",
        "Chunk one is a deliberately longer thirty two word synthesis payload that mirrors the measured slow path and still completes inside the allowed head of line safety window without being cancelled early.",
        "Chunk two becomes ready behind the ordered head.",
    )
    assert len(pieces[1].split()) == 32
    reply = " ".join(pieces)
    attempts: dict[int, int] = {}
    attempts_lock = threading.Lock()
    head_started = threading.Event()
    head_release = threading.Event()
    hedge_started = threading.Event()
    later_started = threading.Event()
    later_release = threading.Event()
    fatal_armed = threading.Event()
    first_audio_emitted = threading.Event()
    head_audio_emitted = threading.Event()
    turn_cancel = threading.Event()
    head_started_at: list[float] = []
    head_completed_at: list[float] = []
    later_ready_at: list[float] = []
    first_audio_at: list[float] = []
    fatal_timers: list[Any] = []
    events: list[tuple[str, dict[str, Any]]] = []
    results: list[dict[str, Any]] = []
    errors: list[BaseException] = []

    class ControlledTimer:
        def __init__(self, interval, function):
            self.interval = float(interval)
            self.function = function
            self.name = ""
            self.daemon = False
            self.cancelled = False

        def start(self):
            assert self.name == "ms4-rest-hol-fatal-deadline"
            fatal_timers.append(self)
            fatal_armed.set()

        def cancel(self):
            self.cancelled = True

        def join(self, timeout=None):
            return None

        def is_alive(self):
            return False

        def dispatch_after_cancel(self):
            self.function()

    monkeypatch.setattr(voice_module.threading, "Timer", ControlledTimer)

    def fake_synthesize(
        *,
        text,
        cancel_event,
        on_request_dispatched=None,
        **_kwargs,
    ):
        idx = pieces.index(text)
        with attempts_lock:
            attempt = attempts.get(idx, 0) + 1
            attempts[idx] = attempt
        if on_request_dispatched is not None:
            on_request_dispatched()
        if idx == 1 and attempt == 1:
            head_started_at.append(time.monotonic())
            head_started.set()
            assert head_release.wait(timeout=2.0)
            head_completed_at.append(time.monotonic())
            return _hol_audio(text)
        if idx == 1:
            assert attempt == 2, "the blocked head may receive only one hedge"
            hedge_started.set()
            assert cancel_event.wait(timeout=2.0)
            raise VoiceUnavailable("ordered head hedge cancelled after primary success")
        if idx == 2:
            later_started.set()
            assert later_release.wait(timeout=2.0)
            later_ready_at.append(time.monotonic())
        return _hol_audio(text)

    def emit(event, payload):
        events.append((event, dict(payload)))
        if event == "audio_chunk" and payload["index"] == 0:
            first_audio_at.append(time.monotonic())
            first_audio_emitted.set()
        if event == "audio_chunk" and payload["index"] == 1:
            head_audio_emitted.set()
        return True

    def run_turn():
        try:
            results.append(voice_ptt_turn_stream(
                runner=_FakeRunner(),
                audio=b"FAKE_WAV",
                chat_fn=lambda _message, stream_callback=None, **_kwargs: (
                    stream_callback(reply + " "),
                    {"text": reply, "session_id": "hol-fatal-window-origin"},
                )[1],
                transcribe_fn=lambda **_kwargs: {
                    "text": "speak three chunks",
                    "model": "whisper-1",
                },
                synthesize_fn=fake_synthesize,
                emit=emit,
                chunker=SentenceChunker(
                    first_chunk_min_words=5,
                    min_chunk_words=1,
                    max_chunk_words=40,
                ),
                tts_pool_size=2,
                engine="rest",
                cancel_event=turn_cancel,
            ))
        except BaseException as exc:
            errors.append(exc)

    turn = threading.Thread(target=run_turn, name="rest-hol-fatal-window-origin")
    turn.start()
    try:
        assert head_started.wait(timeout=2.0), (
            f"ordered head did not start: attempts={attempts!r}, "
            f"errors={errors!r}, events={events!r}"
        )
        assert later_started.wait(timeout=2.0)
        assert first_audio_emitted.wait(timeout=2.0)
        time.sleep(fatal_timeout_s + 0.05)
        later_release.set()
        assert fatal_armed.wait(timeout=2.0)
        assert hedge_started.wait(timeout=2.0)
        assert len(fatal_timers) == 1
        head_release.set()
        assert head_audio_emitted.wait(timeout=2.0)
        turn.join(timeout=2.0)
        assert not turn.is_alive()
        time.sleep(0.005)
        fatal_timers[0].dispatch_after_cancel()
    finally:
        head_release.set()
        later_release.set()
        if turn.is_alive():
            turn_cancel.set()
        turn.join(timeout=2.0)

    assert fatal_timers[0].interval == pytest.approx(fatal_timeout_s, abs=0.005), (
        "the fatal HOL budget must start when later audio becomes ready, not when "
        "the ordered head was originally scheduled"
    )
    assert later_ready_at[0] - head_started_at[0] > fatal_timeout_s
    assert head_completed_at[0] - later_ready_at[0] < fatal_timeout_s
    assert first_audio_at[0] < later_ready_at[0]
    assert fatal_timers[0].cancelled
    assert errors == []
    assert len(results) == 1
    assert attempts == {0: 1, 1: 2, 2: 1}
    scheduled = [payload for event, payload in events if event == "chunk_scheduled"]
    audio = [payload for event, payload in events if event == "audio_chunk"]
    assert [payload["index"] for payload in scheduled] == [0, 1, 2]
    assert [payload["index"] for payload in audio] == [0, 1, 2]
    assert [payload["text"] for payload in audio] == list(pieces)
    assert not any(event == "audio_error" for event, _payload in events)


def test_rest_hol_fatal_deadline_cancels_blocked_head_and_tail(monkeypatch):
    import machine_spirit_4.gateway.voice as voice_module

    _configure_rest_hol_attempt(monkeypatch, voice_module, delay_s=0)
    monkeypatch.setenv("MS4_TTS_REST_HOL_FATAL_TIMEOUT_S", "0.15")
    monkeypatch.setenv("MS4_TTS_REST_CLEANUP_TIMEOUT_S", "0.75")
    monkeypatch.setenv("MS4_TTS_REST_FINAL_TIMEOUT_S", "5")

    pieces = (
        "Chunk zero reaches listener first.",
        "Chunk one follows in strict order.",
        "Chunk two blocks every synthesis attempt.",
        "Chunk three completes behind the blocked head.",
        "Chunk four completes behind the blocked head.",
        "Chunk five completes behind the blocked head.",
        "Chunk six completes behind the blocked head.",
        "Chunk seven completes behind the blocked head.",
        "Chunk eight completes behind the blocked head.",
    )
    reply = " ".join(pieces)
    attempts: dict[int, int] = {}
    attempts_lock = threading.Lock()
    blocked_started = [threading.Event(), threading.Event()]
    blocked_cancelled = [threading.Event(), threading.Event()]
    emergency_release = threading.Event()
    all_tails_completed = threading.Event()
    completed_tails: set[int] = set()
    finished_attempts: list[tuple[int, int]] = []
    turn_cancel = threading.Event()
    events: list[tuple[str, dict[str, Any]]] = []
    results: list[dict[str, Any]] = []
    errors: list[BaseException] = []

    def fake_synthesize(
        *,
        text,
        cancel_event,
        on_request_dispatched=None,
        **_kwargs,
    ):
        idx = pieces.index(text)
        with attempts_lock:
            attempt = attempts.get(idx, 0) + 1
            attempts[idx] = attempt
        if on_request_dispatched is not None:
            on_request_dispatched()
        try:
            if idx == 2:
                if attempt > 2:
                    raise AssertionError("blocked HOL chunk received more than one hedge")
                blocked_started[attempt - 1].set()
                while not cancel_event.wait(timeout=0.01):
                    if emergency_release.is_set():
                        raise VoiceUnavailable("test emergency release")
                blocked_cancelled[attempt - 1].set()
                raise VoiceUnavailable(f"blocked chunk 2 attempt {attempt} cancelled")
            if idx >= 3:
                with attempts_lock:
                    completed_tails.add(idx)
                    if completed_tails == set(range(3, 9)):
                        all_tails_completed.set()
            return _hol_audio(text)
        finally:
            with attempts_lock:
                finished_attempts.append((idx, attempt))

    def run_turn():
        try:
            results.append(voice_ptt_turn_stream(
                runner=_FakeRunner(),
                audio=b"FAKE_WAV",
                chat_fn=lambda _message, stream_callback=None, **_kwargs: (
                    stream_callback(reply + " "),
                    {"text": reply, "session_id": "hol-fatal-deadline"},
                )[1],
                transcribe_fn=lambda **_kwargs: {
                    "text": "speak nine chunks",
                    "model": "whisper-1",
                },
                synthesize_fn=fake_synthesize,
                emit=lambda event, payload: events.append((event, dict(payload))) or True,
                chunker=SentenceChunker(
                    first_chunk_min_words=5,
                    min_chunk_words=1,
                    max_chunk_words=20,
                ),
                tts_pool_size=6,
                engine="rest",
                cancel_event=turn_cancel,
            ))
        except BaseException as exc:
            errors.append(exc)

    started_at = time.monotonic()
    turn = threading.Thread(target=run_turn, name="rest-hol-fatal-deadline")
    turn.start()
    fatal_before_browser_budget = False
    terminal_elapsed = 0.0
    try:
        assert blocked_started[0].wait(timeout=2.0)
        assert all_tails_completed.wait(timeout=2.0)
        assert blocked_started[1].wait(timeout=2.0)
        turn.join(timeout=1.0)
        terminal_elapsed = time.monotonic() - started_at
        fatal_before_browser_budget = not turn.is_alive()
    finally:
        if turn.is_alive():
            turn_cancel.set()
            emergency_release.set()
        turn.join(timeout=2.0)
        emergency_release.set()

    assert fatal_before_browser_budget, (
        "the unresolved in-order head and its single hedge outlived the injected "
        "fatal HOL deadline"
    )
    assert terminal_elapsed < 30.0, "fatal continuation must beat the browser payload budget"
    assert not turn.is_alive()
    assert results == []
    assert len(errors) == 1
    assert isinstance(errors[0], VoiceUnavailable)
    assert "chunk index 2" in str(errors[0])
    assert turn_cancel.is_set()
    assert all(event.is_set() for event in blocked_cancelled)
    assert attempts == {idx: (2 if idx == 2 else 1) for idx in range(9)}
    assert set(finished_attempts) == (
        {(idx, 1) for idx in range(9)} | {(2, 2)}
    )
    scheduled = [payload for event, payload in events if event == "chunk_scheduled"]
    audio = [payload for event, payload in events if event == "audio_chunk"]
    assert [payload["index"] for payload in scheduled] == list(range(9))
    assert [payload["index"] for payload in audio] == [0, 1]
    assert len(audio) == len({payload["index"] for payload in audio})
    assert not any(event == "audio_error" for event, _payload in events)
    terminal_audio_count = len(audio)
    time.sleep(0.05)
    assert sum(event == "audio_chunk" for event, _payload in events) == terminal_audio_count
    assert not any(
        thread.name.startswith(("ms4-tts", "ms4-rest-hol")) and thread.is_alive()
        for thread in threading.enumerate()
    )


def test_rest_hol_hedge_wakes_at_nonzero_deadline_while_chat_open(monkeypatch):
    import machine_spirit_4.gateway.voice as voice_module

    hedge_delay = 0.20
    _configure_rest_hol_attempt(monkeypatch, voice_module, delay_s=hedge_delay)
    pieces = (
        "First audio arrives without delay.",
        "Blocked second chunk needs recovery.",
        "Later third chunk finishes promptly.",
        "Final fourth chunk also finishes.",
    )
    reply = " ".join(pieces)
    attempts: dict[str, int] = {}
    attempts_lock = threading.Lock()
    later_ready = threading.Event()
    hedge_started = threading.Event()
    primary_cancelled = threading.Event()
    chat_holding = threading.Event()
    release_chat = threading.Event()
    chat_returned = threading.Event()
    turn_cancel = threading.Event()
    events: list[tuple[str, dict[str, Any]]] = []
    results: list[dict[str, Any]] = []
    errors: list[BaseException] = []
    later_ready_at: list[float] = []
    hedge_started_at: list[float] = []

    def fake_synthesize(*, text, cancel_event, on_request_dispatched=None, **_kwargs):
        with attempts_lock:
            attempt = attempts.get(text, 0) + 1
            attempts[text] = attempt
        if on_request_dispatched is not None:
            on_request_dispatched()
        if text == pieces[1] and attempt == 1:
            assert cancel_event.wait(timeout=2.0)
            primary_cancelled.set()
            raise VoiceUnavailable("blocked primary lost the deadline hedge")
        if text == pieces[1]:
            hedge_started_at.append(time.monotonic())
            hedge_started.set()
        if text == pieces[2]:
            later_ready_at.append(time.monotonic())
            later_ready.set()
        return _hol_audio(text)

    def chat(_message, *, stream_callback=None, **_kwargs):
        assert stream_callback(reply + " ") is not False
        assert later_ready.wait(timeout=1.0)
        chat_holding.set()
        assert release_chat.wait(timeout=2.0)
        chat_returned.set()
        return {"text": reply, "session_id": "hol-nonzero-deadline"}

    def run_turn():
        try:
            results.append(voice_ptt_turn_stream(
                runner=_FakeRunner(),
                audio=b"FAKE_WAV",
                chat_fn=chat,
                transcribe_fn=lambda **_kwargs: {
                    "text": "keep chat open past hedge deadline",
                    "model": "whisper-1",
                },
                synthesize_fn=fake_synthesize,
                emit=lambda event, payload: events.append((event, dict(payload))) or True,
                chunker=SentenceChunker(
                    first_chunk_min_words=5,
                    min_chunk_words=1,
                    max_chunk_words=20,
                ),
                tts_pool_size=4,
                engine="rest",
                cancel_event=turn_cancel,
            ))
        except BaseException as exc:
            errors.append(exc)

    turn = threading.Thread(target=run_turn, name="rest-hol-nonzero-deadline")
    turn.start()
    hedge_observed = False
    hedge_before_chat_return = False
    try:
        assert chat_holding.wait(timeout=2.0)
        hedge_observed = hedge_started.wait(timeout=0.65)
        hedge_before_chat_return = hedge_observed and not chat_returned.is_set()
    finally:
        release_chat.set()
        if not hedge_observed:
            turn_cancel.set()
        turn.join(timeout=2.0)

    assert hedge_observed, (
        "later-ready audio must arm an independent nonzero hedge-deadline wake "
        "instead of waiting for chat return"
    )
    assert hedge_before_chat_return
    wake_elapsed = hedge_started_at[0] - later_ready_at[0]
    assert hedge_delay * 0.5 <= wake_elapsed <= 0.55, wake_elapsed
    assert not turn.is_alive()
    assert errors == []
    assert len(results) == 1
    assert primary_cancelled.is_set()
    assert attempts == {
        pieces[0]: 1,
        pieces[1]: 2,
        pieces[2]: 1,
        pieces[3]: 1,
    }
    audio = [payload for event, payload in events if event == "audio_chunk"]
    assert [payload["index"] for payload in audio] == [0, 1, 2, 3]
    assert [payload["text"] for payload in audio] == list(pieces)
    assert len(audio) == len({payload["index"] for payload in audio})


def test_rest_hol_pending_traversal_serializes_stream_insertion(monkeypatch, caplog):
    import logging
    from concurrent.futures import Future, ThreadPoolExecutor as RealThreadPoolExecutor

    import machine_spirit_4.gateway.voice as voice_module

    _configure_rest_hol_attempt(monkeypatch, voice_module, delay_s=0)
    caplog.set_level(logging.ERROR, logger="concurrent.futures")
    pieces = (
        "First audio arrives without delay.",
        "Blocked second chunk needs recovery.",
        "Blocked third chunk stays unfinished.",
        "Later fourth chunk finishes promptly.",
        "Concurrent fifth chunk streams safely.",
    )
    reply = " ".join(pieces)
    attempts: dict[str, int] = {}
    attempts_lock = threading.Lock()
    future_text: dict[Future, str | None] = {}
    future_text_lock = threading.Lock()
    first_audio = threading.Event()
    release_later = threading.Event()
    probe_armed = threading.Event()
    probe_used = threading.Event()
    traversal_entered = threading.Event()
    insertion_attempted = threading.Event()
    insertion_completed = threading.Event()
    turn_cancel = threading.Event()
    events: list[tuple[str, dict[str, Any]]] = []
    results: list[dict[str, Any]] = []
    errors: list[BaseException] = []

    class RecordingExecutor:
        def __init__(self, *args, **kwargs):
            self._inner = RealThreadPoolExecutor(*args, **kwargs)

        @property
        def _threads(self):
            return self._inner._threads

        def submit(self, fn, *args, **kwargs):
            future = self._inner.submit(fn, *args, **kwargs)
            with future_text_lock:
                future_text[future] = kwargs.get("text")
            return future

        def shutdown(self, wait=True, *, cancel_futures=False):
            return self._inner.shutdown(wait=wait, cancel_futures=cancel_futures)

    original_done = Future.done

    def instrumented_done(future):
        with future_text_lock:
            text = future_text.get(future)
        if (
            text == pieces[2]
            and probe_armed.is_set()
            and not probe_used.is_set()
        ):
            probe_used.set()
            traversal_entered.set()
            assert insertion_attempted.wait(timeout=2.0)
            insertion_completed.wait(timeout=0.20)
        return original_done(future)

    monkeypatch.setattr(voice_module, "ThreadPoolExecutor", RecordingExecutor)
    monkeypatch.setattr(Future, "done", instrumented_done)

    def fake_synthesize(*, text, cancel_event, on_request_dispatched=None, **_kwargs):
        with attempts_lock:
            attempt = attempts.get(text, 0) + 1
            attempts[text] = attempt
        if on_request_dispatched is not None:
            on_request_dispatched()
        if text in pieces[1:3] and attempt == 1:
            assert cancel_event.wait(timeout=3.0)
            raise VoiceUnavailable("blocked primary lost the synchronized hedge")
        if text == pieces[3]:
            assert release_later.wait(timeout=2.0)
        return _hol_audio(text)

    def emit(event, payload):
        events.append((event, dict(payload)))
        if event == "audio_chunk" and payload["index"] == 0:
            first_audio.set()
        return True

    def chat(_message, *, stream_callback=None, **_kwargs):
        assert stream_callback(pieces[0] + " ") is not False
        assert first_audio.wait(timeout=1.0)
        assert stream_callback(" ".join(pieces[1:4]) + " ") is not False
        probe_armed.set()
        release_later.set()
        assert traversal_entered.wait(timeout=2.0)
        insertion_attempted.set()
        assert stream_callback(pieces[4] + " ") is not False
        insertion_completed.set()
        return {"text": reply, "session_id": "hol-pending-lock"}

    def run_turn():
        try:
            results.append(voice_ptt_turn_stream(
                runner=_FakeRunner(),
                audio=b"FAKE_WAV",
                chat_fn=chat,
                transcribe_fn=lambda **_kwargs: {
                    "text": "stream during hedge traversal",
                    "model": "whisper-1",
                },
                synthesize_fn=fake_synthesize,
                emit=emit,
                chunker=SentenceChunker(
                    first_chunk_min_words=5,
                    min_chunk_words=1,
                    max_chunk_words=20,
                ),
                tts_pool_size=6,
                engine="rest",
                cancel_event=turn_cancel,
            ))
        except BaseException as exc:
            errors.append(exc)

    turn = threading.Thread(target=run_turn, name="rest-hol-pending-lock")
    turn.start()
    turn.join(timeout=4.0)
    if turn.is_alive():
        turn_cancel.set()
        release_later.set()
        insertion_attempted.set()
        insertion_completed.set()
        turn.join(timeout=2.0)

    race_errors = [
        record.exc_info[1]
        for record in caplog.records
        if record.exc_info
        and isinstance(record.exc_info[1], RuntimeError)
        and "dictionary changed size" in str(record.exc_info[1])
    ]
    assert race_errors == [], "pending iteration raced concurrent stream insertion"
    assert probe_used.is_set()
    assert insertion_completed.is_set()
    assert not turn.is_alive()
    assert errors == []
    assert len(results) == 1
    audio = [payload for event, payload in events if event == "audio_chunk"]
    assert [payload["index"] for payload in audio] == [0, 1, 2, 3, 4]
    assert [payload["text"] for payload in audio] == list(pieces)
    assert len(audio) == len({payload["index"] for payload in audio})
    assert not any(
        thread.name.startswith("ms4-tts") and thread.is_alive()
        for thread in threading.enumerate()
    )


def test_rest_hol_disconnect_cancels_delayed_deadline_without_stray_hedge(monkeypatch):
    import machine_spirit_4.gateway.voice as voice_module

    hedge_delay = 0.30
    _configure_rest_hol_attempt(monkeypatch, voice_module, delay_s=hedge_delay)
    pieces = (
        "First audio arrives without delay.",
        "Blocked second chunk needs recovery.",
        "Later third chunk finishes promptly.",
    )
    reply = " ".join(pieces)
    attempts: dict[str, int] = {}
    attempts_lock = threading.Lock()
    later_ready = threading.Event()
    chat_holding = threading.Event()
    timer_created = threading.Event()
    timer_cancelled = threading.Event()
    hedge_started = threading.Event()
    primary_cancelled = threading.Event()
    turn_done = threading.Event()
    client_alive = threading.Event()
    client_alive.set()
    turn_cancel = threading.Event()
    events: list[tuple[str, dict[str, Any]]] = []
    results: list[dict[str, Any]] = []
    errors: list[BaseException] = []
    RealTimer = threading.Timer

    class RecordingTimer(RealTimer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            timer_created.set()

        def cancel(self):
            timer_cancelled.set()
            return super().cancel()

    monkeypatch.setattr(voice_module.threading, "Timer", RecordingTimer)

    def fake_synthesize(*, text, cancel_event, on_request_dispatched=None, **_kwargs):
        with attempts_lock:
            attempt = attempts.get(text, 0) + 1
            attempts[text] = attempt
        if on_request_dispatched is not None:
            on_request_dispatched()
        if text == pieces[1] and attempt == 1:
            assert cancel_event.wait(timeout=2.0)
            primary_cancelled.set()
            raise VoiceUnavailable("disconnect cancelled blocked primary")
        if text == pieces[1]:
            hedge_started.set()
        if text == pieces[2]:
            later_ready.set()
        return _hol_audio(text)

    def chat(_message, *, stream_callback=None, cancel_event=None, **_kwargs):
        assert stream_callback(reply + " ") is not False
        assert later_ready.wait(timeout=1.0)
        chat_holding.set()
        assert cancel_event is not None and cancel_event.wait(timeout=2.0)
        return {"text": reply, "session_id": "hol-disconnect"}

    def run_turn():
        try:
            results.append(voice_ptt_turn_stream(
                runner=_FakeRunner(),
                audio=b"FAKE_WAV",
                chat_fn=chat,
                transcribe_fn=lambda **_kwargs: {
                    "text": "disconnect before hedge deadline",
                    "model": "whisper-1",
                },
                synthesize_fn=fake_synthesize,
                emit=lambda event, payload: events.append((event, dict(payload))) or True,
                chunker=SentenceChunker(
                    first_chunk_min_words=5,
                    min_chunk_words=1,
                    max_chunk_words=20,
                ),
                tts_pool_size=4,
                engine="rest",
                client_alive=client_alive,
                cancel_event=turn_cancel,
            ))
        except BaseException as exc:
            errors.append(exc)
        finally:
            turn_done.set()

    turn = threading.Thread(target=run_turn, name="rest-hol-disconnect")
    turn.start()
    timer_observed = False
    audio_at_disconnect = 0
    try:
        assert chat_holding.wait(timeout=2.0)
        timer_observed = timer_created.wait(timeout=0.45)
        audio_at_disconnect = sum(event == "audio_chunk" for event, _ in events)
    finally:
        client_alive.clear()
        turn_cancel.set()
        turn_done.wait(timeout=1.5)
        turn.join(timeout=1.0)

    assert timer_observed, "later-ready audio must arm a turn-owned hedge deadline"
    assert timer_cancelled.is_set()
    assert not hedge_started.wait(timeout=hedge_delay + 0.15)
    assert not turn.is_alive()
    assert results == []
    assert len(errors) == 1
    assert isinstance(errors[0], VoiceUnavailable)
    assert primary_cancelled.is_set()
    assert attempts == {
        pieces[0]: 1,
        pieces[1]: 1,
        pieces[2]: 1,
    }
    assert sum(event == "audio_chunk" for event, _ in events) == audio_at_disconnect
    assert not any(
        thread.name == "ms4-rest-hol-hedge-deadline" and thread.is_alive()
        for thread in threading.enumerate()
    )
    assert not any(
        thread.name.startswith("ms4-tts") and thread.is_alive()
        for thread in threading.enumerate()
    )


def _capture_model_voice_turn(monkeypatch, *, passed_model):
    """Run a minimal voice stream turn, returning the model the chat_fn
    actually received (to assert obey vs force-auto)."""
    _ms3_voice_ready(monkeypatch)
    seen: dict[str, Any] = {}

    def fake_transcribe(*, hivemind_url, audio, filename, model):
        return {"text": "hello", "model": "whisper-1"}

    def fake_chat(message, *, session_id=None, model=None, stream_callback=None, **_):
        seen["model"] = model
        stream_callback("ok")
        return {"text": "ok", "session_id": "s", "metrics": {"duration_ms": 1}}

    def fake_synth(*, hivemind_url, text, model, voice, response_format):
        return {"audio_bytes": b"\x00", "content_type": "audio/wav",
                "audio_base64": "A", "model": "tts-1", "voice": "alloy", "format": "wav"}

    voice_ptt_turn_stream(
        runner=_FakeRunner(), audio=b"WAV", model=passed_model,
        chat_fn=fake_chat, transcribe_fn=fake_transcribe,
        synthesize_fn=fake_synth, emit=lambda e, p: True, engine="rest",
    )
    return seen.get("model")


def test_stream_obeys_selected_face_model(monkeypatch):
    monkeypatch.delenv("MS4_VOICE_FORCE_AUTO", raising=False)
    # The operator's explicit Face Lobe choice must reach the chat call.
    assert _capture_model_voice_turn(monkeypatch, passed_model="llama3.1:70b") == "llama3.1:70b"


def test_stream_empty_model_uses_auto_picker(monkeypatch):
    monkeypatch.delenv("MS4_VOICE_FORCE_AUTO", raising=False)
    # Empty = "Auto-pick (fast)" -> chat gets None and the runner picks.
    assert _capture_model_voice_turn(monkeypatch, passed_model=None) is None


def test_stream_force_auto_env_overrides_selection(monkeypatch):
    monkeypatch.setenv("MS4_VOICE_FORCE_AUTO", "1")
    # The escape hatch nulls even an explicit selection.
    assert _capture_model_voice_turn(monkeypatch, passed_model="llama3.1:70b") is None


# ---------------------------------------------------------------------------
# First-token watchdog (_run_facechat_guarded) — a stalled Face model must
# raise FaceLobeStalled instead of hanging the SSE stream forever.
# ---------------------------------------------------------------------------


def test_facechat_guard_raises_on_stall():
    def chat_call(transcript, *, session_id=None, model=None, stream_callback=None):
        time.sleep(2.0)  # never emits a token within the budget
        return {"text": "late"}

    with pytest.raises(FaceLobeStalled):
        _run_facechat_guarded(
            chat_call=chat_call, transcript="hi", chat_kwargs={"model": "o4-mini"},
            on_delta=lambda d: None, first_token_timeout=0.3, model="o4-mini",
        )


def test_facechat_guard_signals_transport_cancel_on_stall():
    observed: dict[str, threading.Event] = {}

    def chat_call(transcript, *, stream_callback=None, cancel_event=None, **_kwargs):
        observed["cancel_event"] = cancel_event
        assert cancel_event.wait(timeout=2.0)
        return {"text": "canceled"}

    with pytest.raises(FaceLobeStalled):
        _run_facechat_guarded(
            chat_call=chat_call,
            transcript="hi",
            chat_kwargs={"model": "llama3.1:8b"},
            on_delta=lambda _delta: None,
            first_token_timeout=0.1,
            model="llama3.1:8b",
        )

    assert observed["cancel_event"].is_set()


def test_facechat_guard_propagates_client_disconnect_to_transport():
    observed: dict[str, Any] = {}

    def chat_call(transcript, *, stream_callback=None, cancel_event=None, **_kwargs):
        observed["callback_result"] = stream_callback("first token")
        observed["cancel_event"] = cancel_event
        return {"text": "partial"}

    with pytest.raises(VoiceUnavailable, match="verified completion"):
        _run_facechat_guarded(
            chat_call=chat_call,
            transcript="hi",
            chat_kwargs={"model": "llama3.1:8b"},
            on_delta=lambda _delta: False,
            first_token_timeout=1.0,
            model="llama3.1:8b",
        )

    assert observed["callback_result"] is False
    assert observed["cancel_event"].is_set()


def test_facechat_guard_passes_when_streaming():
    seen: list[str] = []

    def chat_call(transcript, *, session_id=None, model=None, stream_callback=None):
        stream_callback("hel")
        stream_callback("lo")
        return {"text": "hello", "completed": True}

    resp = _run_facechat_guarded(
        chat_call=chat_call, transcript="hi", chat_kwargs={"model": "m"},
        on_delta=seen.append, first_token_timeout=0.5, model="m",
    )
    assert resp["text"] == "hello"
    assert seen == ["hel", "lo"]


def test_facechat_guard_accepts_buffered_upstream_progress_without_releasing_text():
    seen: list[str] = []

    def chat_call(transcript, *, stream_callback=None, cancel_event=None, **_kwargs):
        del transcript, stream_callback
        cancel_event.mark_buffered_progress()
        time.sleep(0.2)  # longer than watchdog, but a held upstream token exists
        return {"text": "corrected substantive answer", "completed": True}

    resp = _run_facechat_guarded(
        chat_call=chat_call,
        transcript="go deeper",
        chat_kwargs={"model": "qwen3.6:35b"},
        on_delta=seen.append,
        first_token_timeout=0.05,
        model="qwen3.6:35b",
    )

    assert resp["text"] == "corrected substantive answer"
    assert seen == [], "held unvalidated text must not reach the voice callback"


def test_facechat_guard_enforces_one_aggregate_routing_and_first_token_deadline(monkeypatch):
    """Routing and model TTFT must share one monotonic first-token budget."""
    from machine_spirit_4.gateway.face_lobe_chat import FaceLobeChat

    face = FaceLobeChat(hivemind_url="http://unused.invalid")
    seen: list[str] = []

    def delayed_stream(_payload, callback, *, cancel_event=None):
        time.sleep(0.15)
        callback("first token")
        return "first token", {
            "http_latency_ms": 150,
            "bytes_received": 1,
            "stream_chunks": 1,
            "stream_first_token_ms": 150,
        }

    monkeypatch.setattr(face, "_post_streaming", delayed_stream)

    def routed_chat(transcript, *, stream_callback=None, cancel_event=None, **_kwargs):
        time.sleep(0.15)
        return face.chat(
            transcript,
            session_id="physical-turn",
            model="llama3.1:8b",
            stream_callback=stream_callback,
            cancel_event=cancel_event,
        )

    started_at = time.monotonic()
    with pytest.raises(FaceLobeStalled):
        _run_facechat_guarded(
            chat_call=routed_chat,
            transcript="explain the cluster",
            chat_kwargs={"model": None},
            on_delta=seen.append,
            first_token_timeout=0.22,
            model=None,
        )

    assert time.monotonic() - started_at < 0.5
    assert seen == []


def test_facechat_guard_fast_zero_token_is_not_a_stall():
    # Returns before the budget with no token at all — must NOT be judged stalled.
    def chat_call(transcript, *, session_id=None, model=None, stream_callback=None):
        return {"text": "", "completed": True}

    resp = _run_facechat_guarded(
        chat_call=chat_call, transcript="hi", chat_kwargs={"model": "m"},
        on_delta=lambda d: None, first_token_timeout=5.0, model="m",
    )
    assert resp == {"text": "", "completed": True}


def test_facechat_guard_propagates_chat_error():
    def chat_call(transcript, *, session_id=None, model=None, stream_callback=None):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        _run_facechat_guarded(
            chat_call=chat_call, transcript="hi", chat_kwargs={"model": "m"},
            on_delta=lambda d: None, first_token_timeout=2.0, model="m",
        )


# ---------------------------------------------------------------------------
# Finding 2 (R4): the guard observes the TURN's cancellation fence (barge /
# client disconnect), not only its private abort, and NEVER waits on an
# unbounded join. A cancel before OR after first token retires the worker with
# a bounded join instead of hanging out the (generous) first-token watchdog.
# ---------------------------------------------------------------------------


def test_facechat_guard_cancel_before_first_token_is_bounded():
    """A turn cancelled before the first token must retire the worker with a
    BOUNDED join and raise — not wait out the (30s) first-token watchdog."""
    cancel = threading.Event()
    started = threading.Event()
    released = threading.Event()

    def chat_call(transcript, *, stream_callback=None, cancel_event=None, **_kwargs):
        started.set()
        # Block until the guard signals abort (abort_request is threaded in as
        # cancel_event); a well-behaved back-lobe call observes it and returns.
        assert cancel_event is not None and cancel_event.wait(timeout=3.0)
        released.set()
        return {"text": "late"}

    def _canceller():
        started.wait(2.0)
        time.sleep(0.05)
        cancel.set()

    threading.Thread(target=_canceller, daemon=True).start()
    t0 = time.monotonic()
    with pytest.raises(VoiceUnavailable):
        _run_facechat_guarded(
            chat_call=chat_call, transcript="hi", chat_kwargs={"model": "m"},
            on_delta=lambda d: None, first_token_timeout=30.0, model="m",
            cancel_event=cancel, join_timeout=2.0,
        )
    elapsed = time.monotonic() - t0
    assert elapsed < 5.0, "cancel must be bounded, not wait out the 30s first-token watchdog"
    assert released.is_set(), "the worker must observe abort_request and retire"


def test_facechat_guard_cancel_after_first_token_is_bounded():
    """A turn cancelled AFTER the first token (a barge mid-stream) must stop
    waiting on the slow completion within the bounded join, not block forever."""
    cancel = threading.Event()
    released = threading.Event()

    def chat_call(transcript, *, stream_callback=None, cancel_event=None, **_kwargs):
        stream_callback("first")          # first token seen -> watchdog stands down
        cancel.set()                       # a replacement turn barges us now
        assert cancel_event is not None and cancel_event.wait(timeout=3.0)
        released.set()
        return {"text": "partial-after-cancel"}

    t0 = time.monotonic()
    with pytest.raises(VoiceUnavailable, match="verified completion"):
        _run_facechat_guarded(
            chat_call=chat_call, transcript="hi", chat_kwargs={"model": "m"},
            on_delta=lambda d: None, first_token_timeout=5.0, model="m",
            cancel_event=cancel, join_timeout=2.0,
        )
    elapsed = time.monotonic() - t0
    assert elapsed < 4.0, "a cancelled mid-stream turn must not wait unboundedly"
    assert released.is_set(), "the worker observes abort and retires within the bounded join"


def test_facechat_guard_client_disconnect_event_is_observed():
    """client_alive cleared (browser gone) is observed by the guard exactly like
    an explicit cancel_event: the worker is aborted and bounded-joined."""
    client_alive = threading.Event()
    client_alive.set()
    released = threading.Event()
    started = threading.Event()

    def chat_call(transcript, *, stream_callback=None, cancel_event=None, **_kwargs):
        started.set()
        assert cancel_event is not None and cancel_event.wait(timeout=3.0)
        released.set()
        return {"text": "late"}

    def _disconnect():
        started.wait(2.0)
        time.sleep(0.05)
        client_alive.clear()

    threading.Thread(target=_disconnect, daemon=True).start()
    with pytest.raises(VoiceUnavailable):
        _run_facechat_guarded(
            chat_call=chat_call, transcript="hi", chat_kwargs={"model": "m"},
            on_delta=lambda d: None, first_token_timeout=30.0, model="m",
            client_alive=client_alive, join_timeout=2.0,
        )
    assert released.is_set(), "a client disconnect must retire the worker"


def test_speaker_id_bounded_returns_fast_result(monkeypatch):
    import machine_spirit_4.gateway.voice_identity as vi
    monkeypatch.setattr(vi, "identify_speaker_from_wav",
                        lambda url, audio: {"name": "Alice", "accepted": True})
    assert _identify_speaker_bounded("http://hive", b"WAV", 2.0) == {"name": "Alice", "accepted": True}


def test_speaker_id_bounded_does_not_block_on_slow_id(monkeypatch):
    import machine_spirit_4.gateway.voice_identity as vi

    def slow(url, audio):
        time.sleep(3.0)  # a long-utterance diarization
        return {"name": "Bob", "accepted": True}

    monkeypatch.setattr(vi, "identify_speaker_from_wav", slow)
    t0 = time.monotonic()
    res = _identify_speaker_bounded("http://hive", b"WAV", 0.3)
    elapsed = time.monotonic() - t0
    assert res is None  # didn't resolve within the budget
    assert elapsed < 1.5  # and crucially did NOT block for the full 3s


def test_speaker_id_budget_zero_skips(monkeypatch):
    import machine_spirit_4.gateway.voice_identity as vi
    called = {"n": 0}
    monkeypatch.setattr(vi, "identify_speaker_from_wav",
                        lambda url, audio: called.__setitem__("n", called["n"] + 1))
    assert _identify_speaker_bounded("http://hive", b"WAV", 0.0) is None
    assert called["n"] == 0  # disabled -> not even attempted


def _wav_of_secs(secs: float, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * secs))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Runtime audio gate (trim a babble tail off a live reply chunk)
# ---------------------------------------------------------------------------


def test_wav_duration_and_trim():
    assert abs(wav_duration_secs(_wav_of_secs(3.0)) - 3.0) < 0.05
    assert wav_duration_secs(b"not a wav") is None
    trimmed = trim_wav(_wav_of_secs(5.0), 2.0)
    assert trimmed is not None and abs(wav_duration_secs(trimmed) - 2.0) < 0.1


def test_wav_duration_uses_actual_pcm_when_streaming_header_has_sentinel_sizes():
    streaming = bytearray(_wav_of_secs(2.8))
    streaming[4:8] = (0x7FFFFFFF).to_bytes(4, "little")
    streaming[40:44] = (0x7FFFFFFF).to_bytes(4, "little")
    audio = bytes(streaming)

    assert wav_duration_secs(audio) == pytest.approx(2.8, abs=0.01)
    out, trimmed = _gate_chunk_audio(
        audio,
        "audio/wav",
        "The second trade-off centers on compute and cost.",
    )
    assert trimmed is False
    assert out == audio


def test_expected_speech_secs_scales_with_words():
    assert expected_speech_secs("Hi") < expected_speech_secs("one two three four five six")


@pytest.mark.parametrize("excess_s", [0.001, 4.0])
def test_runtime_gate_measurement_preserves_tiny_and_large_excess(excess_s):
    text = "On it"
    limit_s = expected_speech_secs(text) + 1.0
    measurement = _runtime_audio_gate_measurement(
        _wav_of_secs(limit_s + excess_s),
        "audio/wav",
        text,
    )

    assert measurement is not None
    assert measurement["would_gate"] is True
    assert measurement["original_duration_ms"] == pytest.approx(
        (limit_s + excess_s) * 1000.0,
        abs=0.1,
    )
    assert measurement["limit_ms"] == pytest.approx(limit_s * 1000.0)
    assert measurement["excess_ms"] == pytest.approx(excess_s * 1000.0, abs=0.1)


def test_gate_trims_babble_chunk():
    # 6s render for a 2-word chunk -> trimmed down.
    out, trimmed = _gate_chunk_audio(_wav_of_secs(6.0), "audio/wav", "On it")
    assert trimmed is True
    assert wav_duration_secs(out) < 6.0


def test_gate_leaves_clean_chunk_untouched():
    clean = _wav_of_secs(1.0)
    out, trimmed = _gate_chunk_audio(clean, "audio/wav", "On it")
    assert trimmed is False and out == clean


def test_gate_uses_verbalized_numeric_constraint_without_weakening_babble_case():
    from machine_spirit_4.gateway.spoken_text_filter import sanitize_for_speech

    spoken = sanitize_for_speech(
        "1. Live Traffic (4B Face Lobe): - Latency: <4 seconds."
    ).strip()
    assert spoken.endswith("Latency: less than 4 seconds.")

    # The exact live Turn 4 phrase gains the two words the TTS engine actually
    # speaks for '<'.  A clean technical render near the old 9.15s limit is no
    # longer clipped, while the existing two-word/six-second babble regression
    # remains authoritative in test_gate_trims_babble_chunk.
    clean = _wav_of_secs(9.2)
    out, trimmed = _gate_chunk_audio(clean, "audio/wav", spoken)
    assert trimmed is False and out == clean


def test_gate_skips_non_wav():
    out, trimmed = _gate_chunk_audio(b"\xff\xfb mp3-ish", "audio/mpeg", "On it")
    assert trimmed is False


def test_parallel_chunks_fallback_chunks_a_multi_sentence_reply(monkeypatch):
    """The ws_super zero-audio fallback must sentence-chunk + parallel-
    synthesize (NOT one big single call). Multi-sentence text -> multiple
    in-order audio chunks."""
    import machine_spirit_4.gateway.voice as v

    def fake_synth(*, hivemind_url, text, model, voice, response_format):
        wav = _wav_of_secs(0.8)
        return {"audio_bytes": wav, "content_type": "audio/wav",
                "audio_base64": base64.b64encode(wav).decode(), "model": "tts-1",
                "voice": "alloy", "format": "wav"}

    monkeypatch.setattr(v, "synthesize", fake_synth)
    events: list[tuple[str, dict]] = []
    # Use longer sentences so coalescing (MIN_CHUNK_WORDS) keeps them separate.
    n = v._emit_text_as_parallel_chunks(
        text="The first sentence is reasonably long here. The second sentence is also "
             "quite long here. And the third sentence runs long as well here.",
        emit=lambda ev, p: (events.append((ev, dict(p))), True)[1],
        hivemind_url="http://hive:6089", tts_model="tts-1", tts_voice="alloy",
        response_format="wav",
    )
    audio = [p for ev, p in events if ev == "audio_chunk"]
    assert n >= 2 and len(audio) >= 2, "multi-sentence reply must produce multiple parallel chunks"
    # Emitted in contiguous index order.
    idxs = [p["index"] for p in audio]
    assert idxs == sorted(idxs) == list(range(len(idxs)))


def test_parallel_chunks_fallback_recovers_gated_render_without_duplicate_sse(monkeypatch):
    import machine_spirit_4.gateway.voice as v

    monkeypatch.setenv("MS4_VOICE_RUNTIME_AUDIO_GATE", "1")
    renders = [_wav_of_secs(6.0), _wav_of_secs(1.0)]
    calls = []

    def fake_synth(*, text, **_):
        wav = renders[len(calls)]
        calls.append(text)
        return {
            "audio_bytes": wav,
            "content_type": "audio/wav",
            "audio_base64": base64.b64encode(wav).decode(),
        }

    monkeypatch.setattr(v, "synthesize", fake_synth)
    events: list[tuple[str, dict]] = []
    lifecycle = {}
    emitted = v._emit_text_as_parallel_chunks(
        text="Hi there.",
        emit=lambda event, payload: (
            events.append((event, dict(payload))),
            True,
        )[1],
        hivemind_url="http://hive:6089",
        tts_model="tts-1",
        tts_voice="alloy",
        response_format="wav",
        lifecycle=lifecycle,
    )

    audio = [payload for event, payload in events if event == "audio_chunk"]
    assert len(calls) == 2
    assert emitted == 1
    assert len(audio) == 1
    assert audio[0]["runtime_gated"] is False
    assert audio[0]["runtime_gate"]["recovered"] is True
    assert audio[0]["runtime_gate"]["retry_outcome"] == "clean"
    assert lifecycle["rest_fallback_runtime_gate_retries"] == 1
    assert lifecycle["rest_fallback_runtime_gate_recoveries"] == 1
    assert lifecycle["rest_fallback_runtime_gate_retry_failures"] == 0


def test_parallel_chunks_fallback_repeated_gate_stays_degraded_without_duplicate_sse(monkeypatch):
    import machine_spirit_4.gateway.voice as v

    monkeypatch.setenv("MS4_VOICE_RUNTIME_AUDIO_GATE", "1")
    gated_wav = _wav_of_secs(6.0)
    calls = []

    def fake_synth(*, text, **_):
        calls.append(text)
        return {
            "audio_bytes": gated_wav,
            "content_type": "audio/wav",
            "audio_base64": base64.b64encode(gated_wav).decode(),
        }

    monkeypatch.setattr(v, "synthesize", fake_synth)
    events: list[tuple[str, dict]] = []
    lifecycle = {}
    emitted = v._emit_text_as_parallel_chunks(
        text="Hi there.",
        emit=lambda event, payload: (
            events.append((event, dict(payload))),
            True,
        )[1],
        hivemind_url="http://hive:6089",
        tts_model="tts-1",
        tts_voice="alloy",
        response_format="wav",
        lifecycle=lifecycle,
    )

    audio = [payload for event, payload in events if event == "audio_chunk"]
    assert len(calls) == 2
    assert emitted == 1
    assert len(audio) == 1
    assert audio[0]["runtime_gated"] is True
    assert audio[0]["runtime_gate"]["recovered"] is False
    assert audio[0]["runtime_gate"]["retry_outcome"] == "still_gated"
    assert audio[0]["runtime_gate"]["final_runtime_gated"] is True
    played = base64.b64decode(audio[0]["audio_base64"])
    assert wav_duration_secs(played) == pytest.approx(3.9, abs=0.01)
    assert lifecycle["rest_fallback_runtime_gate_retries"] == 1
    assert lifecycle["rest_fallback_runtime_gate_recoveries"] == 0
    assert lifecycle["rest_fallback_runtime_gate_retry_failures"] == 1


def test_default_rest_final_timeout_scales_for_long_depth_speech(monkeypatch):
    import machine_spirit_4.gateway.voice as v

    monkeypatch.delenv("MS4_TTS_REST_FINAL_TIMEOUT_S", raising=False)
    monkeypatch.delenv("MS4_TTS_REST_FINAL_TIMEOUT_PER_WAVE_S", raising=False)
    monkeypatch.delenv("MS4_TTS_REST_FINAL_TIMEOUT_MAX_S", raising=False)

    assert v._voice_rest_final_timeout(piece_count=2, capacity=2) == 60.0
    assert v._voice_rest_final_timeout(piece_count=22, capacity=2) == 165.0
    assert v._voice_rest_final_timeout(piece_count=68, capacity=2) == 510.0
    assert v._voice_rest_final_timeout(piece_count=1000, capacity=2) == 600.0


def test_explicit_rest_final_timeout_remains_an_exact_ceiling(monkeypatch):
    import machine_spirit_4.gateway.voice as v

    monkeypatch.setenv("MS4_TTS_REST_FINAL_TIMEOUT_S", "0.05")

    assert v._voice_rest_final_timeout(piece_count=68, capacity=2) == 0.05


def test_rest_timeout_configuration_cannot_create_an_unbounded_wait(monkeypatch):
    import machine_spirit_4.gateway.voice as v

    monkeypatch.setenv("MS4_TTS_REST_FINAL_TIMEOUT_S", "inf")
    monkeypatch.setenv("MS4_TTS_REST_PROGRESS_TIMEOUT_S", "nan")

    assert v._voice_rest_final_timeout(piece_count=68, capacity=2) == 60.0
    assert v._voice_rest_progress_timeout() == 60.0


def test_rest_drain_allows_slow_multi_chunk_progress_past_idle_budget(monkeypatch):
    import machine_spirit_4.gateway.voice as voice_module

    _ms3_voice_ready(monkeypatch)
    monkeypatch.setenv("MS4_VOICE_SMART_REFLEX", "0")
    monkeypatch.setenv("MS4_VOICE_FIRST_CHUNK_DEADLINE_S", "0")
    monkeypatch.setenv("MS4_TTS_REST_PROGRESS_TIMEOUT_S", "0.12")
    monkeypatch.setenv("MS4_TTS_REST_FINAL_TIMEOUT_S", "2.0")
    monkeypatch.setattr(voice_module, "_identify_speaker_async", lambda *_a, **_k: None)

    pieces = (
        "Chunk zero keeps this answer moving forward.",
        "Chunk one arrives after another bounded synthesis.",
        "Chunk two proves the stream is still progressing.",
        "Chunk three continues beyond one idle interval.",
        "Chunk four remains part of the same answer.",
        "Chunk five still reaches the client in order.",
        "Chunk six preserves the complete spoken response.",
        "Chunk seven closes the deliberately long drain.",
    )
    synthesized: list[str] = []
    events: list[tuple[str, dict[str, Any]]] = []

    class _OneDeltaPerChunk:
        """Make this a drain test, independent of SentenceChunker heuristics."""

        def __init__(self):
            self.buffer = ""
            self.chunk_count = 0
            self.first_chunk_sanitizer = None

        def add(self, delta):
            piece = delta.strip()
            if not piece:
                return []
            self.chunk_count += 1
            return [piece]

        def flush(self):
            return None

    def slow_synthesis(*, text, cancel_event, **_kwargs):
        synthesized.append(text)
        if cancel_event.wait(timeout=0.045):
            raise VoiceUnavailable("cancelled")
        wav = _wav_of_secs(0.05)
        return {
            "audio_bytes": wav,
            "content_type": "audio/wav",
            "audio_base64": base64.b64encode(wav).decode("ascii"),
        }

    def chat(_message, *, stream_callback=None, **_kwargs):
        for piece in pieces:
            assert stream_callback(piece + " ") is not False
        return {"text": " ".join(pieces), "session_id": "slow-progress"}

    started_at = time.monotonic()
    result = voice_ptt_turn_stream(
        runner=_FakeRunner(),
        audio=b"FAKE_WAV",
        chat_fn=chat,
        transcribe_fn=lambda **_: {"text": "explain fully", "model": "whisper-1"},
        synthesize_fn=slow_synthesis,
        emit=lambda name, payload: (events.append((name, dict(payload))), True)[1],
        chunker=_OneDeltaPerChunk(),
        tts_pool_size=1,
        engine="rest",
    )
    elapsed = time.monotonic() - started_at

    scheduled = [payload for name, payload in events if name == "chunk_scheduled"]
    audio = [payload for name, payload in events if name == "audio_chunk"]
    assert elapsed > 0.24, "the total drain must exceed two idle intervals"
    assert len(scheduled) == len(pieces)
    assert result["metrics"]["audio_chunks"] == len(pieces) == len(audio)
    assert [payload["index"] for payload in audio] == list(range(len(pieces)))
    assert synthesized == [payload["text"] for payload in scheduled], (
        "progress handling must neither duplicate nor omit scheduled synthesis"
    )
    assert not any(
        thread.name.startswith("ms4-tts") and thread.is_alive()
        for thread in threading.enumerate()
    )


def test_rest_progress_stall_fails_closed_and_retires_worker(monkeypatch):
    import machine_spirit_4.gateway.voice as voice_module

    _ms3_voice_ready(monkeypatch)
    monkeypatch.setenv("MS4_VOICE_SMART_REFLEX", "0")
    monkeypatch.setenv("MS4_TTS_REST_PROGRESS_TIMEOUT_S", "0.08")
    monkeypatch.setenv("MS4_TTS_REST_FINAL_TIMEOUT_S", "2.0")
    monkeypatch.setenv("MS4_TTS_REST_CLEANUP_TIMEOUT_S", "0.5")
    monkeypatch.setattr(voice_module, "_identify_speaker_async", lambda *_a, **_k: None)
    synthesis_cancelled = threading.Event()
    synthesis_calls = [0]

    def stalled_synthesis(*, cancel_event, **_kwargs):
        synthesis_calls[0] += 1
        assert cancel_event.wait(timeout=0.5)
        synthesis_cancelled.set()
        raise VoiceUnavailable("cancelled")

    with pytest.raises(VoiceUnavailable, match="stalled waiting on chunks"):
        voice_ptt_turn_stream(
            runner=_FakeRunner(),
            audio=b"FAKE_WAV",
            chat_fn=lambda _message, stream_callback=None, **_: (
                stream_callback("one two three four five six."),
                {"text": "one two three four five six.", "session_id": "stalled"},
            )[1],
            transcribe_fn=lambda **_: {"text": "hello", "model": "whisper-1"},
            synthesize_fn=stalled_synthesis,
            emit=lambda _name, _payload: True,
            engine="rest",
            tts_pool_size=1,
        )

    assert synthesis_calls == [1], "stall cleanup must not duplicate synthesis"
    assert synthesis_cancelled.is_set()
    assert not any(
        thread.name.startswith("ms4-tts") and thread.is_alive()
        for thread in threading.enumerate()
    )


def _assert_rest_timeout_preserves_primary_when_cleanup_fails(
    monkeypatch,
    *,
    final_timeout_s: float,
    progress_timeout_s: float,
    primary_match: str,
):
    import machine_spirit_4.gateway.voice as voice_module

    _ms3_voice_ready(monkeypatch)
    monkeypatch.setenv("MS4_VOICE_SMART_REFLEX", "0")
    monkeypatch.setenv("MS4_TTS_REST_FINAL_TIMEOUT_S", str(final_timeout_s))
    monkeypatch.setenv(
        "MS4_TTS_REST_PROGRESS_TIMEOUT_S", str(progress_timeout_s)
    )
    monkeypatch.setenv("MS4_TTS_REST_CLEANUP_TIMEOUT_S", "0.5")
    monkeypatch.setattr(voice_module, "_identify_speaker_async", lambda *_a, **_k: None)

    cleanup_failure = RuntimeError(f"{primary_match} cleanup sentinel")
    base_controller = voice_module._FirstChunkDeadlineController

    class _FailingCloseController(base_controller):
        def close(self):
            super().close()
            raise cleanup_failure

    monkeypatch.setattr(
        voice_module,
        "_FirstChunkDeadlineController",
        _FailingCloseController,
    )
    synthesis_cancelled = threading.Event()
    synthesis_calls = [0]
    audio_events: list[dict[str, Any]] = []

    def blocking_synthesis(*, cancel_event, **_kwargs):
        synthesis_calls[0] += 1
        assert cancel_event.wait(timeout=0.75)
        synthesis_cancelled.set()
        raise VoiceUnavailable("cancelled")

    def emit(name, payload):
        if name == "audio_chunk":
            audio_events.append(dict(payload))
        return True

    with pytest.raises(VoiceUnavailable, match=primary_match) as raised:
        voice_ptt_turn_stream(
            runner=_FakeRunner(),
            audio=b"FAKE_WAV",
            chat_fn=lambda _message, stream_callback=None, **_: (
                stream_callback("one two three four five six."),
                {"text": "one two three four five six.", "session_id": "cleanup"},
            )[1],
            transcribe_fn=lambda **_: {"text": "hello", "model": "whisper-1"},
            synthesize_fn=blocking_synthesis,
            emit=emit,
            engine="rest",
            tts_pool_size=1,
        )

    assert raised.value.__cause__ is cleanup_failure
    notes = "\n".join(getattr(raised.value, "__notes__", ()))
    assert "REST turn cleanup also failed after terminal TTS error" in notes
    assert str(cleanup_failure) in notes
    cleanup_notes = "\n".join(getattr(cleanup_failure, "__notes__", ()))
    assert "REST turn cleanup failed in first-chunk timer close" in cleanup_notes
    assert synthesis_calls == [1]
    assert synthesis_cancelled.is_set()
    assert audio_events == []
    assert not any(
        (
            thread.name.startswith("ms4-tts")
            or thread.name == "ms4-rest-emit-coordinator"
        )
        and thread.is_alive()
        for thread in threading.enumerate()
    )


def test_rest_hard_timeout_remains_primary_when_cleanup_fails(monkeypatch):
    _assert_rest_timeout_preserves_primary_when_cleanup_fails(
        monkeypatch,
        final_timeout_s=0.05,
        progress_timeout_s=2.0,
        primary_match="hard deadline exceeded",
    )


def test_rest_progress_stall_remains_primary_when_cleanup_fails(monkeypatch):
    _assert_rest_timeout_preserves_primary_when_cleanup_fails(
        monkeypatch,
        final_timeout_s=2.0,
        progress_timeout_s=0.05,
        primary_match="stalled waiting on chunks",
    )


def test_rest_stall_remains_primary_when_speaker_cleanup_fails(monkeypatch):
    import machine_spirit_4.gateway.voice as voice_module

    _ms3_voice_ready(monkeypatch)
    monkeypatch.setenv("MS4_VOICE_SMART_REFLEX", "0")
    monkeypatch.setenv("MS4_VOICE_FIRST_CHUNK_DEADLINE_S", "0")
    monkeypatch.setenv("MS4_VOICE_SPEAKER_ID_ASYNC", "1")
    monkeypatch.setenv("MS4_VOICE_SPEAKER_ID_TERMINAL_BUDGET_S", "0.05")
    monkeypatch.setenv("MS4_TTS_REST_FINAL_TIMEOUT_S", "2.0")
    monkeypatch.setenv("MS4_TTS_REST_PROGRESS_TIMEOUT_S", "0.08")
    monkeypatch.setenv("MS4_TTS_REST_CLEANUP_TIMEOUT_S", "0.5")

    class _OneDeltaPerChunk:
        def __init__(self):
            self.buffer = ""
            self.chunk_count = 0
            self.first_chunk_sanitizer = None

        def add(self, delta):
            piece = delta.strip()
            if not piece:
                return []
            self.chunk_count += 1
            return [piece]

        def flush(self):
            return None

    speaker_started = threading.Event()
    release_speaker = threading.Event()
    speaker_threads: list[threading.Thread] = []

    def stuck_speaker(_url, _audio, _emit, _cancel_event, timeout=None):
        def wait_for_controlled_release():
            speaker_started.set()
            release_speaker.wait(timeout=2.0)

        thread = threading.Thread(
            target=wait_for_controlled_release,
            name="test-stuck-speaker-id",
            daemon=True,
        )
        speaker_threads.append(thread)
        thread.start()
        return thread

    monkeypatch.setattr(voice_module, "_identify_speaker_async", stuck_speaker)
    synthesis_calls: list[str] = []
    tail_cancelled = threading.Event()
    events: list[tuple[str, dict[str, Any]]] = []

    def partial_then_stalled_synthesis(*, text, cancel_event, **_kwargs):
        synthesis_calls.append(text)
        if len(synthesis_calls) == 1:
            wav = _wav_of_secs(0.05)
            return {
                "audio_bytes": wav,
                "content_type": "audio/wav",
                "audio_base64": base64.b64encode(wav).decode("ascii"),
            }
        assert cancel_event.wait(timeout=0.75)
        tail_cancelled.set()
        raise VoiceUnavailable("cancelled")

    pieces = (
        "The first chunk reaches the listener.",
        "The second chunk stalls until cancellation.",
    )
    try:
        with pytest.raises(VoiceUnavailable, match="stalled waiting on chunks") as raised:
            voice_ptt_turn_stream(
                runner=_FakeRunner(),
                audio=b"FAKE_WAV",
                chat_fn=lambda _message, stream_callback=None, **_: (
                    stream_callback(pieces[0] + " "),
                    stream_callback(pieces[1] + " "),
                    {"text": " ".join(pieces), "session_id": "speaker-stall"},
                )[2],
                transcribe_fn=lambda **_: {"text": "hello", "model": "whisper-1"},
                synthesize_fn=partial_then_stalled_synthesis,
                emit=lambda name, payload: (
                    events.append((name, dict(payload))),
                    True,
                )[1],
                chunker=_OneDeltaPerChunk(),
                engine="rest",
                tts_pool_size=1,
            )

        assert isinstance(raised.value.__cause__, VoiceUnavailable)
        assert "speaker identification cancellation failed" in str(
            raised.value.__cause__
        )
        notes = "\n".join(getattr(raised.value, "__notes__", ()))
        assert "Speaker cleanup also failed while preserving primary voice error" in notes
        assert "speaker identification cancellation failed" in notes
        audio = [payload for name, payload in events if name == "audio_chunk"]
        assert [payload["index"] for payload in audio] == [0]
        assert synthesis_calls == list(pieces)
        assert speaker_started.is_set()
        assert tail_cancelled.is_set()
        assert not any(
            (
                thread.name.startswith("ms4-tts")
                or thread.name == "ms4-rest-emit-coordinator"
            )
            and thread.is_alive()
            for thread in threading.enumerate()
        )
    finally:
        release_speaker.set()
        for thread in speaker_threads:
            thread.join(timeout=0.5)
        assert all(not thread.is_alive() for thread in speaker_threads)


def test_parallel_chunks_emits_first_piece_before_tail_fanout(monkeypatch):
    import machine_spirit_4.gateway.voice as v

    first_emitted = threading.Event()
    tail_started: list[str] = []

    def fake_synth(*, text, **_):
        if not text.startswith("The first sentence"):
            assert first_emitted.is_set(), "tail synthesis started before first audio was emitted"
            tail_started.append(text)
        wav = _wav_of_secs(0.2)
        return {
            "audio_bytes": wav,
            "content_type": "audio/wav",
            "audio_base64": base64.b64encode(wav).decode(),
        }

    def emit(event, payload):
        if event == "audio_chunk" and payload["index"] == 0:
            first_emitted.set()
        return True

    monkeypatch.setattr(v, "synthesize", fake_synth)
    emitted = v._emit_text_as_parallel_chunks(
        text=(
            "The first sentence contains enough words to remain separate. "
            "The second sentence contains enough words to remain separate. "
            "The third sentence contains enough words to remain separate."
        ),
        emit=emit,
        hivemind_url="http://hive:6089",
        tts_model="tts-1",
        tts_voice="alloy",
        response_format="wav",
    )

    assert emitted >= 2
    assert first_emitted.is_set()
    assert len(tail_started) == emitted - 1


def test_rest_stream_metrics_count_only_client_accepted_audio(monkeypatch):
    _ms3_voice_ready(monkeypatch)
    attempted_audio: list[dict] = []

    def emit(name, payload):
        if name == "audio_chunk":
            attempted_audio.append(dict(payload))
            return False
        return True

    with pytest.raises(VoiceUnavailable, match="cancelled"):
        voice_ptt_turn_stream(
            runner=_FakeRunner(),
            audio=b"FAKE_WAV",
            chat_fn=lambda message, stream_callback=None, **_: (
                stream_callback("One complete REST sentence. "),
                {"text": "One complete REST sentence.", "session_id": "s"},
            )[1],
            transcribe_fn=lambda **_: {"text": "hello", "model": "whisper-1"},
            synthesize_fn=lambda **_: {
                "audio_bytes": b"WAV",
                "audio_base64": "V0FW",
                "content_type": "audio/wav",
            },
            emit=emit,
            engine="rest",
        )

    assert len(attempted_audio) == 1
    assert not any(
        thread.name.startswith("ms4-tts") and thread.is_alive()
        for thread in threading.enumerate()
    )


def test_stream_runtime_gate_trims_live_chunk(monkeypatch):
    """Two gated renders still emit one trimmed, explicitly degraded chunk."""
    _ms3_voice_ready(monkeypatch)
    monkeypatch.setenv("MS4_VOICE_RUNTIME_AUDIO_GATE", "1")
    monkeypatch.setenv("MS4_VOICE_RUNTIME_QA", "0")

    def fake_transcribe(*, hivemind_url, audio, filename, model):
        return {"text": "say a thing", "model": "whisper-1"}

    def fake_chat(message, *, session_id=None, model=None, depth_model=None, stream_callback=None, **_):
        stream_callback("Hi there. ")
        return {"text": "Hi there.", "session_id": "s", "metrics": {}}

    long_wav = _wav_of_secs(6.0)

    synthesis_calls = []

    def fake_synth(*, hivemind_url, text, model, voice, response_format):
        synthesis_calls.append(text)
        return {"audio_bytes": long_wav, "content_type": "audio/wav",
                "audio_base64": base64.b64encode(long_wav).decode(), "model": "tts-1",
                "voice": "alloy", "format": "wav"}

    events: list[tuple[str, dict]] = []
    result = voice_ptt_turn_stream(
        runner=_FakeRunner(), audio=b"WAV", chat_fn=fake_chat,
        transcribe_fn=fake_transcribe, synthesize_fn=fake_synth,
        emit=lambda ev, p: (events.append((ev, dict(p))), True)[1],
        engine="rest",
    )
    audio = [p for ev, p in events if ev == "audio_chunk"]
    scheduled = [p for ev, p in events if ev == "chunk_scheduled"]
    assert len(synthesis_calls) == 2, "one and only one gate recovery render"
    assert len(scheduled) == 1, "recovery must not mint a duplicate SSE chunk"
    assert len(audio) == 1, "recovery must not emit a duplicate SSE chunk"
    played = base64.b64decode(audio[0]["audio_base64"])
    assert wav_duration_secs(played) < 5.5, "babble tail should have been trimmed from 6s"
    assert audio[0]["runtime_gated"] is True
    assert result["metrics"]["runtime_gated"] == 1
    lifecycle = result["metrics"]["lifecycle"]
    assert lifecycle["runtime_gate_retries"] == 1
    assert lifecycle["runtime_gate_recoveries"] == 0
    assert lifecycle["runtime_gate_retry_failures"] == 1
    gate = result["metrics"]["chunks"][0]["runtime_gate"]
    assert gate["original_duration_ms"] == pytest.approx(6000.0)
    assert gate["limit_ms"] == pytest.approx(3900.0)
    assert gate["excess_ms"] == pytest.approx(2100.0)
    assert gate["retry_outcome"] == "still_gated"
    assert gate["recovered"] is False
    assert gate["final_runtime_gated"] is True
    assert gate["trim_delta_ms"] == pytest.approx(2100.0, abs=0.1)


def test_stream_runtime_gate_clean_retry_recovers_without_duplicate_sse(monkeypatch):
    _ms3_voice_ready(monkeypatch)
    monkeypatch.setenv("MS4_VOICE_RUNTIME_AUDIO_GATE", "1")
    monkeypatch.setenv("MS4_VOICE_RUNTIME_QA", "0")

    gated_wav = _wav_of_secs(6.0)
    clean_wav = _wav_of_secs(1.0)
    renders = [gated_wav, clean_wav]
    synthesis_calls = []

    def fake_synth(*, hivemind_url, text, model, voice, response_format):
        synthesis_calls.append(text)
        wav = renders[len(synthesis_calls) - 1]
        return {
            "audio_bytes": wav,
            "content_type": "audio/wav",
            "audio_base64": base64.b64encode(wav).decode(),
            "model": "tts-1",
            "voice": "alloy",
            "format": "wav",
        }

    events: list[tuple[str, dict]] = []
    result = voice_ptt_turn_stream(
        runner=_FakeRunner(),
        audio=b"WAV",
        chat_fn=lambda message, *, stream_callback=None, **_: (
            stream_callback("Hi there. "),
            {"text": "Hi there.", "session_id": "s", "metrics": {}},
        )[1],
        transcribe_fn=lambda **_: {"text": "say a thing", "model": "whisper-1"},
        synthesize_fn=fake_synth,
        emit=lambda ev, payload: (events.append((ev, dict(payload))), True)[1],
        engine="rest",
    )

    audio_chunks = [payload for event, payload in events if event == "audio_chunk"]
    scheduled = [payload for event, payload in events if event == "chunk_scheduled"]
    assert len(synthesis_calls) == 2
    assert len(scheduled) == 1
    assert len(audio_chunks) == 1
    assert audio_chunks[0]["runtime_gated"] is False
    assert wav_duration_secs(base64.b64decode(audio_chunks[0]["audio_base64"])) == pytest.approx(1.0)
    assert result["metrics"]["runtime_gated"] == 0
    lifecycle = result["metrics"]["lifecycle"]
    assert lifecycle["runtime_gate_retries"] == 1
    assert lifecycle["runtime_gate_recoveries"] == 1
    assert lifecycle["runtime_gate_retry_failures"] == 0
    gate = result["metrics"]["chunks"][0]["runtime_gate"]
    assert gate["retry_outcome"] == "clean"
    assert gate["recovered"] is True
    assert gate["selected_attempt"] == "retry"
    assert gate["final_runtime_gated"] is False
    assert gate["original_duration_ms"] == pytest.approx(6000.0)
    assert gate["retry_original_duration_ms"] == pytest.approx(1000.0)
    assert gate["trim_delta_ms"] == pytest.approx(0.0)


def test_stream_runtime_gate_retry_exception_preserves_trimmed_fail_closed_chunk(monkeypatch):
    _ms3_voice_ready(monkeypatch)
    monkeypatch.setenv("MS4_VOICE_RUNTIME_AUDIO_GATE", "1")
    monkeypatch.setenv("MS4_VOICE_RUNTIME_QA", "0")

    gated_wav = _wav_of_secs(6.0)
    synthesis_calls = []

    def fake_synth(*, hivemind_url, text, model, voice, response_format):
        synthesis_calls.append(text)
        if len(synthesis_calls) == 2:
            raise RuntimeError("fresh render failed")
        return {
            "audio_bytes": gated_wav,
            "content_type": "audio/wav",
            "audio_base64": base64.b64encode(gated_wav).decode(),
            "model": "tts-1",
            "voice": "alloy",
            "format": "wav",
        }

    events: list[tuple[str, dict]] = []
    result = voice_ptt_turn_stream(
        runner=_FakeRunner(),
        audio=b"WAV",
        chat_fn=lambda message, *, stream_callback=None, **_: (
            stream_callback("Hi there. "),
            {"text": "Hi there.", "session_id": "s", "metrics": {}},
        )[1],
        transcribe_fn=lambda **_: {"text": "say a thing", "model": "whisper-1"},
        synthesize_fn=fake_synth,
        emit=lambda ev, payload: (events.append((ev, dict(payload))), True)[1],
        engine="rest",
    )

    audio_chunks = [payload for event, payload in events if event == "audio_chunk"]
    assert len(synthesis_calls) == 2
    assert len([1 for event, _ in events if event == "chunk_scheduled"]) == 1
    assert len(audio_chunks) == 1
    assert audio_chunks[0]["runtime_gated"] is True
    assert not [1 for event, _ in events if event == "audio_error"]
    assert result["metrics"]["runtime_gated"] == 1
    lifecycle = result["metrics"]["lifecycle"]
    assert lifecycle["runtime_gate_retries"] == 1
    assert lifecycle["runtime_gate_recoveries"] == 0
    assert lifecycle["runtime_gate_retry_failures"] == 1
    gate = result["metrics"]["chunks"][0]["runtime_gate"]
    assert gate["retry_outcome"] == "exception"
    assert gate["recovered"] is False
    assert gate["final_runtime_gated"] is True


def test_runtime_gate_recovery_honors_turn_cancel_before_retry(monkeypatch):
    monkeypatch.setenv("MS4_VOICE_RUNTIME_AUDIO_GATE", "1")
    turn_cancel = threading.Event()
    gated_wav = _wav_of_secs(6.0)
    calls = []

    def fake_synth(**_):
        calls.append(1)
        return {
            "audio_bytes": gated_wav,
            "content_type": "audio/wav",
            "audio_base64": base64.b64encode(gated_wav).decode(),
        }

    with pytest.raises(VoiceUnavailable, match="cancelled"):
        _call_tts_with_runtime_audio_gate_recovery(
            fake_synth,
            {"text": "On it"},
            turn_cancel=turn_cancel,
            on_gate_retry=turn_cancel.set,
        )
    assert len(calls) == 1


def test_runtime_gate_recovery_honors_client_liveness_after_retry(monkeypatch):
    monkeypatch.setenv("MS4_VOICE_RUNTIME_AUDIO_GATE", "1")
    turn_cancel = threading.Event()
    client_alive = threading.Event()
    client_alive.set()
    renders = [_wav_of_secs(6.0), _wav_of_secs(1.0)]
    calls = []

    def fake_synth(**_):
        wav = renders[len(calls)]
        calls.append(1)
        if len(calls) == 2:
            client_alive.clear()
        return {
            "audio_bytes": wav,
            "content_type": "audio/wav",
            "audio_base64": base64.b64encode(wav).decode(),
        }

    with pytest.raises(VoiceUnavailable, match="cancelled"):
        _call_tts_with_runtime_audio_gate_recovery(
            fake_synth,
            {"text": "On it"},
            turn_cancel=turn_cancel,
            client_alive=client_alive,
        )
    assert len(calls) == 2


def test_runtime_gate_recovery_is_idempotent_when_caller_is_already_wrapped(monkeypatch):
    monkeypatch.setenv("MS4_VOICE_RUNTIME_AUDIO_GATE", "1")
    turn_cancel = threading.Event()
    gated_wav = _wav_of_secs(6.0)
    calls = []

    def raw_synth(**_):
        calls.append(1)
        return {
            "audio_bytes": gated_wav,
            "content_type": "audio/wav",
            "audio_base64": base64.b64encode(gated_wav).decode(),
        }

    def already_wrapped(**kwargs):
        return _call_tts_with_runtime_audio_gate_recovery(
            raw_synth,
            kwargs,
            turn_cancel=turn_cancel,
        )

    result = _call_tts_with_runtime_audio_gate_recovery(
        already_wrapped,
        {"text": "On it"},
        turn_cancel=turn_cancel,
    )

    assert len(calls) == 2, "nested adapters must still permit only one retry"
    assert result["_ms4_runtime_audio_gate"]["retry_count"] == 1
    assert result["_ms4_runtime_audio_gate"]["retry_outcome"] == "still_gated"


def test_stream_normalizes_turn5_table_fragments_before_tts_and_gate(monkeypatch):
    """The exact live c17/c18 split reaches synthesis as semantic prose only."""
    import machine_spirit_4.gateway.voice as voice_module

    _ms3_voice_ready(monkeypatch)
    monkeypatch.setattr(voice_module, "TTS_FILTER_ENABLED", True)
    monkeypatch.setenv("MS4_VOICE_RUNTIME_AUDIO_GATE", "1")
    monkeypatch.setenv("MS4_VOICE_RUNTIME_QA", "0")

    first = (
        "#### 4. Trade-Offs Under Voice Latency Constraint | Factor | "
        "4B Face Lobe | 35B Depth Lobe (Async) | "
        "|----------------------|--------------------------------------"
    )
    second = (
        "-|---------------------------------------| | Latency | "
        "<1s (real-time) | >5s (post-mortem) "
    )
    reply = first + second

    def fake_chat(
        message,
        *,
        session_id=None,
        model=None,
        depth_model=None,
        stream_callback=None,
        **_,
    ):
        assert stream_callback(first) is not False
        assert stream_callback(second) is not False
        return {"text": reply, "session_id": "turn5-table", "metrics": {}}

    synthesis_calls: list[str] = []

    def fake_synth(*, hivemind_url, text, model, voice, response_format):
        synthesis_calls.append(text)
        wav = _wav_of_secs(1.0)
        return {
            "audio_bytes": wav,
            "content_type": "audio/wav",
            "audio_base64": base64.b64encode(wav).decode(),
            "model": "tts-1",
            "voice": "vega",
            "format": "wav",
        }

    events: list[tuple[str, dict]] = []
    result = voice_ptt_turn_stream(
        runner=_FakeRunner(),
        audio=b"WAV",
        chat_fn=fake_chat,
        transcribe_fn=lambda **_: {"text": "correct the recommendation", "model": "whisper-1"},
        synthesize_fn=fake_synth,
        emit=lambda event, payload: (events.append((event, dict(payload))), True)[1],
        engine="rest",
    )

    spoken = " ".join(synthesis_calls)
    assert synthesis_calls
    assert "#" not in spoken
    assert "|" not in spoken
    assert "---" not in spoken
    assert "<1s" not in spoken
    assert ">5s" not in spoken
    for semantic_cell in (
        "Trade-Offs Under Voice Latency Constraint",
        "Factor",
        "4B Face Lobe",
        "35B Depth Lobe (Async)",
        "Latency",
        "less than 1s (real-time)",
        "greater than 5s (post-mortem)",
    ):
        assert semantic_cell in spoken
    audio_chunks = [payload for event, payload in events if event == "audio_chunk"]
    assert audio_chunks and all(not chunk["runtime_gated"] for chunk in audio_chunks)
    assert result["metrics"]["runtime_gated"] == 0


def test_stream_stalled_face_model_raises(monkeypatch):
    """End-to-end: a stalled Face model inside voice_ptt_turn_stream raises
    FaceLobeStalled (which the server turns into an error SSE event so the
    UI plays the error reflex) rather than hanging the stream forever."""
    _ms3_voice_ready(monkeypatch)
    monkeypatch.setenv("MS4_VOICE_FIRST_TOKEN_TIMEOUT_S", "1")  # min floor
    monkeypatch.delenv("MS4_VOICE_FORCE_AUTO", raising=False)

    def fake_transcribe(*, hivemind_url, audio, filename, model):
        return {"text": "hello", "model": "whisper-1"}

    def fake_chat(message, *, session_id=None, model=None, depth_model=None, stream_callback=None, **_):
        time.sleep(4.0)  # stall: no token within the 1s budget
        return {"text": "late"}

    def fake_synth(*, hivemind_url, text, model, voice, response_format):
        return {"audio_bytes": b"\x00", "content_type": "audio/wav",
                "audio_base64": "A", "model": "tts-1", "voice": "alloy", "format": "wav"}

    with pytest.raises(FaceLobeStalled):
        voice_ptt_turn_stream(
            runner=_FakeRunner(), audio=b"WAV", model="o4-mini-2025-04-16",
            chat_fn=fake_chat, transcribe_fn=fake_transcribe,
            synthesize_fn=fake_synth, emit=lambda e, p: True, engine="rest",
        )


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
            engine="rest",
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

    def fake_chat(message, *, session_id=None, model=None, stream_callback=None, **_):
        # Three sentences each >= MIN_CHUNK_WORDS (10) so coalescing keeps
        # them as separate chunks and exercises first-chunk priority plus
        # parallel tail synthesis.
        sentences = (
            "The first sentence here is deliberately written to be quite long indeed. ",
            "The second sentence here is also written to be reasonably long as well. ",
            "The third sentence here likewise runs long enough to stay its own chunk. ",
        )
        for s in sentences:
            for w in s.split():
                stream_callback(w + " ")
        return {"text": "".join(sentences).strip(), "session_id": "s"}

    # The first TTS call is deliberately slow. Its request-dispatched callback
    # reserves chunk 0's upstream priority and releases exactly one lead tail;
    # the remaining tail fans out after chunk 0 delivery.
    call_counter = {"n": 0}

    def fake_synthesize(**kw):
        n = call_counter["n"]
        call_counter["n"] += 1
        if n == 0 and kw.get("on_request_dispatched") is not None:
            kw["on_request_dispatched"]()
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
        engine="rest",
    )

    m = result["metrics"]
    chunks = m.get("chunks") or []
    # First chunk ships at the first sentence end or FIRST_CHUNK_MIN_WORDS
    # (raised to 5 on Jun 1 2026 after a benchmark showed tiny first chunks
    # synth SLOWER and babble); subsequent short sentences coalesce up to
    # MIN_CHUNK_WORDS, so the exact count depends on tuning. We only require
    # enough chunks to exercise the dispatch gate + parallelism metrics.
    assert len(chunks) >= 2, f"expected >=2 chunks to exercise dispatch priority, got {len(chunks)}"
    # Every per-chunk record has the fields we promised.
    for c in chunks:
        assert {"index", "text_len", "scheduled_ms", "tts_completed_ms", "emitted_ms", "tts_ms", "held_for_inorder_ms"} <= set(c.keys())
        assert c["tts_ms"] >= 0
        assert c["held_for_inorder_ms"] >= 0
    # Chunk 0 reaches request dispatch first. Exactly one lead request schedules
    # before chunk 0 delivery; every remaining tail request waits for delivery.
    first = next(c for c in chunks if c["index"] == 0)
    tail = [c for c in chunks if c["index"] > 0]
    lead, remaining = tail[0], tail[1:]
    assert first["scheduled_ms"] <= lead["scheduled_ms"] < first["emitted_ms"], lead
    assert all(c["scheduled_ms"] >= first["emitted_ms"] for c in remaining), remaining
    assert lead["held_for_inorder_ms"] > 50
    # Parallelism summary shows overlap: the sum of per-chunk work exceeds the
    # two-stage wall window even when only one post-delivery tail remains.
    tp = m.get("tts_parallelism") or {}
    assert tp.get("speedup_ratio", 0) > 1.0, f"expected overlapping TTS work, got {tp}"
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

    chat_turn_ids: list[str | None] = []

    def fake_chat(
        message,
        *,
        session_id=None,
        model=None,
        stream_callback=None,
        turn_id=None,
        **_,
    ):
        chat_turn_ids.append(turn_id)
        for word in "First sentence. Second sentence. Third one.".split():
            stream_callback(word + " ")
        return {
            "text": "First sentence. Second sentence. Third one.",
            "session_id": "s",
            "turn_id": turn_id,
            "revision_id": 7,
        }

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
            if self._first_audio_ms is None:
                self._first_audio_ms = int((time.monotonic() - self.t_start) * 1000)
            self.emit("audio_chunk", {
                "index": self._chunks_emitted,
                "text": "",
                "audio_base64": "AAA",
                "audio_mime": "audio/wav",
                "tts_ms": int((time.monotonic() - self.t_start) * 1000),
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

        def cancel_pending_send(self):
            self.close()

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
    scheduled = [p for e, p in events if e == "chunk_scheduled"]
    audio_chunks = [p for e, p in events if e == "audio_chunk"]
    assert scheduled
    assert len(audio_chunks) >= 1, "expected at least one audio_chunk from the fake ws engine"
    transcript = next(payload for event, payload in events if event == "transcript")
    turn_id = transcript["turn_id"]
    assert chat_turn_ids == [turn_id]
    assert result["turn_id"] == turn_id
    assert result["revision_id"] == 7
    assert event_types.index("chunk_scheduled") < event_types.index("audio_chunk")
    # Same payload shape as the REST engine — the UI consumes both unchanged.
    for p in audio_chunks:
        assert {"index", "text", "audio_base64", "audio_mime", "tts_ms", "held_for_inorder_ms", "engine"} <= set(p.keys())
        assert p["engine"] == "ws_super"
        assert p["held_for_inorder_ms"] == 0  # WS path is naturally ordered
        assert p["turn_id"] == turn_id
        assert p["chunk_id"].startswith(f"{turn_id}-c")
    # Final metrics block declares the engine and zero parallel-related fields.
    m = result["metrics"]
    assert m["engine"] == "ws_super"
    assert m["tts_engine"]["engine"] == "ws_super"
    assert m["audio_chunks"] == len(audio_chunks)
    assert m["chunks_scheduled"] == len(scheduled)
    assert m["first_chunk_scheduled_ms"] is not None
    assert m["tts_parallelism"]["any_held_for_inorder"] is False


def test_ws_super_first_chunk_pipeline_is_structurally_low_latency(monkeypatch, pin_ws_super_engine):
    """No wall-clock target: event dependencies prove the latency shape.

    The Face callback must return after scheduling five words even while the
    WS send is waiting; first audio must then arrive before chat completion.
    Speaker ID starts only after first audio and still emits its eventual identity.
    """
    import machine_spirit_4.gateway.voice as voice_module

    _ms3_voice_ready(monkeypatch)
    monkeypatch.setenv("MS4_VOICE_SMART_REFLEX", "0")
    monkeypatch.setenv("MS4_VOICE_EGG_HONORABLE", "0")

    trace: list[str] = []
    chat_progressed = threading.Event()
    first_audio_seen = threading.Event()

    def fake_identify(_url, _audio, emit, _cancel_event=None):
        trace.append("speaker_started")

        def _late_identity():
            assert first_audio_seen.wait(timeout=2.0)
            emit("speaker", {"speaker": {"accepted": True, "name": "Operator"}})

        thread = threading.Thread(target=_late_identity, daemon=True)
        thread.start()
        return thread

    monkeypatch.setattr(voice_module, "_identify_speaker_async", fake_identify)

    def fake_chat(message, *, stream_callback=None, **_):
        assert stream_callback("one two three four five ") is not False
        trace.append("chat_progressed")
        chat_progressed.set()
        assert first_audio_seen.wait(timeout=2.0)
        trace.append("chat_finished")
        stream_callback("six seven.")
        return {"text": "one two three four five six seven.", "session_id": "s"}

    class _StructuralWsEngine:
        def __init__(self, *, emit, t_start, voice, **_):
            self.emit = emit
            self.t_start = t_start
            self.voice = voice
            self.pushed: list[str] = []
            self.first_audio_ms = None
            self.chunks_emitted = 0

        def open(self):
            self.emit("tts_engine", {"engine": "ws_super", "voice": self.voice, "connect_ms": 0})

        def push(self, text):
            self.pushed.append(text)
            # If push ran inline on the Face callback thread, this dependency
            # could never resolve because fake_chat sets it after callback return.
            assert chat_progressed.wait(timeout=2.0)
            if self.first_audio_ms is None:
                self.first_audio_ms = int((time.monotonic() - self.t_start) * 1000)
            self.emit("audio_chunk", {
                "index": self.chunks_emitted,
                "text": text,
                "audio_base64": "AAA",
                "audio_mime": "audio/wav",
                "tts_ms": self.first_audio_ms,
                "held_for_inorder_ms": 0,
                "engine": "ws_super",
            })
            self.chunks_emitted += 1

        def flush(self):
            pass

        def wait_for_final(self, timeout=60):
            return True

        def close(self):
            pass

        def cancel_pending_send(self):
            chat_progressed.set()

        def metrics(self):
            return {
                "engine": "ws_super",
                "voice": self.voice,
                "connect_ms": 0,
                "first_audio_ms": self.first_audio_ms,
                "chunks_emitted": self.chunks_emitted,
                "bytes_received": self.chunks_emitted * 3,
                "is_final_seen": True,
                "error": None,
            }

    events: list[tuple[str, dict]] = []

    def emit(name, payload):
        trace.append(name)
        events.append((name, dict(payload)))
        if name == "audio_chunk":
            first_audio_seen.set()
        return True

    result = voice_ptt_turn_stream(
        runner=_FakeRunner(),
        audio=b"FAKE_WAV",
        chat_fn=fake_chat,
        transcribe_fn=lambda **_: {"text": "hello", "model": "whisper-1"},
        synthesize_fn=lambda **_: {},
        emit=emit,
        ws_engine_factory=_StructuralWsEngine,
    )

    assert trace.index("chunk_scheduled") < trace.index("chat_progressed")
    assert trace.index("chat_progressed") < trace.index("audio_chunk")
    assert trace.index("audio_chunk") < trace.index("chat_finished")
    assert trace.index("chunk_scheduled") < trace.index("speaker_started")
    assert trace.index("audio_chunk") < trace.index("speaker_started")
    assert trace.index("audio_chunk") < trace.index("speaker")
    assert any(name == "speaker" for name, _ in events)
    first_scheduled = next(payload for name, payload in events if name == "chunk_scheduled")
    assert first_scheduled["text"] == "one two three four five"
    assert [p["index"] for name, p in events if name == "audio_chunk"] == [0, 1]
    assert result["metrics"]["chunks_scheduled"] == 2
    assert result["metrics"]["chat_ms"] <= result["metrics"]["total_ms"]


def test_ws_super_keeps_split_words_and_filter_boundaries_intact(monkeypatch, pin_ws_super_engine):
    import machine_spirit_4.gateway.voice as voice_module

    _ms3_voice_ready(monkeypatch)
    monkeypatch.setenv("MS4_VOICE_SMART_REFLEX", "0")
    monkeypatch.setenv("MS4_VOICE_EGG_HONORABLE", "0")
    monkeypatch.setattr(voice_module, "_identify_speaker_async", lambda *_a, **_k: None)

    events: list[tuple[str, dict]] = []
    pushed: list[str] = []
    first_push = threading.Event()
    reply = (
        "Here is a useful answer. ```python\nprint('secret')\n``` "
        "**Done** with [the docs](https://example.com)."
    )

    def fake_chat(message, *, stream_callback=None, **_):
        scheduled_before = len([name for name, _ in events if name == "chunk_scheduled"])
        assert stream_callback("Here is a useful an") is not False
        assert len([name for name, _ in events if name == "chunk_scheduled"]) == scheduled_before
        assert stream_callback("swer. ```py") is not False
        assert first_push.wait(timeout=2.0)
        assert stream_callback("thon\nprint('secret')\n``") is not False
        assert stream_callback("` **Done** with [the do") is not False
        assert stream_callback("cs](https://example.com). ") is not False
        return {"text": reply, "session_id": "s"}

    class _BoundaryWsEngine:
        def __init__(self, *, emit, voice, **_):
            self.emit = emit
            self.voice = voice
            self.closed = False

        def open(self):
            pass

        def push(self, text):
            pushed.append(text)
            first_push.set()
            self.emit("audio_chunk", {
                "index": len(pushed) - 1,
                "text": text,
                "audio_base64": "AAA",
                "audio_mime": "audio/wav",
                "tts_ms": 0,
                "held_for_inorder_ms": 0,
                "engine": "ws_super",
            })

        def flush(self):
            pass

        def wait_for_final(self, timeout=60):
            return True

        def close(self):
            self.closed = True

        def cancel_pending_send(self):
            self.close()

        def metrics(self):
            return {
                "engine": "ws_super",
                "chunks_emitted": len(pushed),
                "first_audio_ms": 0 if pushed else None,
                "error": None,
            }

    result = voice_ptt_turn_stream(
        runner=_FakeRunner(),
        audio=b"FAKE_WAV",
        chat_fn=fake_chat,
        transcribe_fn=lambda **_: {"text": "hello", "model": "whisper-1"},
        synthesize_fn=lambda **_: pytest.fail("REST must not run"),
        emit=lambda name, payload: events.append((name, dict(payload))) or True,
        ws_engine_factory=_BoundaryWsEngine,
    )

    spoken = " ".join(pushed)
    assert pushed[0] == "Here is a useful answer."
    assert "an swer" not in spoken
    assert "secret" not in spoken
    assert "https://" not in spoken
    assert "**" not in spoken
    assert "Done with the docs." in spoken
    assert result["metrics"]["audio_chunks"] == len(pushed)


def test_ws_zero_audio_fallback_keeps_format_and_exact_metrics(monkeypatch, pin_ws_super_engine):
    import machine_spirit_4.gateway.voice as voice_module

    _ms3_voice_ready(monkeypatch)
    monkeypatch.setenv("MS4_VOICE_SMART_REFLEX", "0")
    monkeypatch.setenv("MS4_VOICE_EGG_HONORABLE", "0")
    monkeypatch.setattr(voice_module, "_identify_speaker_async", lambda *_a, **_k: None)

    formats: list[str | None] = []

    def fake_synthesize(**kwargs):
        formats.append(kwargs.get("response_format"))
        return {
            "audio_bytes": b"MP3",
            "audio_base64": base64.b64encode(b"MP3").decode("ascii"),
            "content_type": "audio/mpeg",
        }

    monkeypatch.setattr(voice_module, "synthesize", fake_synthesize)

    reply = (
        "The first fallback sentence has enough words to stand on its own. "
        "The second fallback sentence also has enough words to remain separate."
    )

    def fake_chat(message, *, stream_callback=None, **_):
        stream_callback(reply)
        return {"text": reply, "session_id": "s"}

    class _ZeroAudioWsEngine:
        def __init__(self, *, voice, **_):
            self.voice = voice

        def open(self):
            pass

        def push(self, _text):
            pass

        def flush(self):
            pass

        def wait_for_final(self, timeout=60):
            return True

        def close(self):
            pass

        def cancel_pending_send(self):
            self.close()

        def metrics(self):
            return {
                "engine": "ws_super",
                "voice": self.voice,
                "connect_ms": 0,
                "first_audio_ms": None,
                "chunks_emitted": 0,
                "bytes_received": 0,
                "is_final_seen": True,
                "error": "zero audio",
            }

    events: list[tuple[str, dict]] = []
    result = voice_ptt_turn_stream(
        runner=_FakeRunner(),
        audio=b"FAKE_WAV",
        chat_fn=fake_chat,
        transcribe_fn=lambda **_: {"text": "hello", "model": "whisper-1"},
        synthesize_fn=lambda **_: {},
        response_format="mp3",
        emit=lambda name, payload: events.append((name, dict(payload))) or True,
        ws_engine_factory=_ZeroAudioWsEngine,
    )

    metrics = result["metrics"]
    audio = [payload for name, payload in events if name == "audio_chunk"]
    assert len(audio) >= 2
    turn_id = result["turn_id"]
    transcript = next(payload for name, payload in events if name == "transcript")
    assert transcript["turn_id"] == turn_id
    assert all(payload["turn_id"] == turn_id for payload in audio)
    assert len({payload["chunk_id"] for payload in audio}) == len(audio)
    assert all(
        payload["chunk_id"].startswith(f"{turn_id}-fallback-c")
        for payload in audio
    )
    assert formats and set(formats) == {"mp3"}
    assert metrics["rest_fallback_used"] is True
    assert metrics["audio_chunks"] == len(audio)
    assert metrics["audio_errors"] == 1
    assert metrics["first_audio_chunk_ms"] is not None


def test_ws_first_audio_timeout_closes_without_raw_preabort_and_falls_back(monkeypatch, pin_ws_super_engine):
    # 2026-07-05 correction: a first-audio timeout is a NORMAL timeout. It must
    # tear down through engine.close()'s bounded graceful-first path and must NOT
    # raw-abort the transport first (that is reserved for barge/disconnect/
    # blocked-send emergencies). REST still serves audio exactly once.
    import machine_spirit_4.gateway.voice as voice_module

    _ms3_voice_ready(monkeypatch)
    monkeypatch.setenv("MS4_TTS_WS_FIRST_AUDIO_TIMEOUT", "0")
    monkeypatch.setenv("MS4_VOICE_SMART_REFLEX", "0")
    monkeypatch.setattr(voice_module, "_identify_speaker_async", lambda *_a, **_k: None)
    monkeypatch.setattr(voice_module, "synthesize", lambda **_: {
        "audio_bytes": b"WAV",
        "audio_base64": "V0FW",
        "content_type": "audio/wav",
    })
    lifecycle: list[str] = []

    class _TimeoutWsEngine:
        def __init__(self, **_):
            pass

        def open(self):
            pass

        def push(self, _text):
            pass

        def flush(self):
            pass

        def wait_for_final(self, timeout=60):
            return False

        def cancel_pending_send(self):
            lifecycle.append("cancel")

        def close(self):
            lifecycle.append("close")

        def metrics(self):
            return {"engine": "ws_super", "chunks_emitted": 0, "error": None}

    reply = "The fallback sentence contains enough words to synthesize promptly."
    result = voice_ptt_turn_stream(
        runner=_FakeRunner(),
        audio=b"FAKE_WAV",
        chat_fn=lambda message, stream_callback=None, **_: (
            stream_callback(reply),
            {"text": reply, "session_id": "s"},
        )[1],
        transcribe_fn=lambda **_: {"text": "hello", "model": "whisper-1"},
        synthesize_fn=lambda **_: pytest.fail("full REST turn must not replay chat"),
        emit=lambda _name, _payload: True,
        ws_engine_factory=_TimeoutWsEngine,
    )

    assert lifecycle == ["close"], (
        f"first-audio timeout must close gracefully with NO raw pre-abort, got {lifecycle}"
    )
    assert "cancel" not in lifecycle
    assert result["metrics"]["rest_fallback_used"] is True
    assert result["metrics"]["audio_chunks"] == 1


def test_ws_rest_fallback_stops_and_counts_only_client_accepted_audio(monkeypatch, pin_ws_super_engine):
    import machine_spirit_4.gateway.voice as voice_module

    _ms3_voice_ready(monkeypatch)
    monkeypatch.setattr(voice_module, "_identify_speaker_async", lambda *_a, **_k: None)
    monkeypatch.setattr(voice_module, "synthesize", lambda **_: {
        "audio_bytes": b"WAV",
        "audio_base64": "V0FW",
        "content_type": "audio/wav",
    })
    reply = (
        "The first rejected fallback sentence has enough words to remain separate. "
        "The second rejected fallback sentence also has enough words to remain separate."
    )

    class _ZeroAudioWsEngine:
        def __init__(self, **_):
            pass

        def open(self):
            pass

        def push(self, _text):
            pass

        def flush(self):
            pass

        def wait_for_final(self, timeout=60):
            return True

        def close(self):
            pass

        def cancel_pending_send(self):
            self.close()

        def metrics(self):
            return {"engine": "ws_super", "chunks_emitted": 0, "error": "zero"}

    attempted_audio: list[dict] = []

    def reject_audio(name, payload):
        if name == "audio_chunk":
            attempted_audio.append(dict(payload))
            return False
        return True

    with pytest.raises(VoiceUnavailable, match="cancelled"):
        voice_ptt_turn_stream(
            runner=_FakeRunner(),
            audio=b"FAKE_WAV",
            chat_fn=lambda message, stream_callback=None, **_: (
                stream_callback(reply),
                {"text": reply, "session_id": "s"},
            )[1],
            transcribe_fn=lambda **_: {"text": "hello", "model": "whisper-1"},
            synthesize_fn=lambda **_: {},
            emit=reject_audio,
            ws_engine_factory=_ZeroAudioWsEngine,
        )

    assert len(attempted_audio) == 1
    assert not any(
        thread.name.startswith("ms4-tts-fb") and thread.is_alive()
        for thread in threading.enumerate()
    )


def test_default_ws_open_failure_falls_back_to_rest(monkeypatch, pin_ws_super_engine):
    import machine_spirit_4.gateway.voice as voice_module

    _ms3_voice_ready(monkeypatch)
    monkeypatch.setenv("MS4_VOICE_SMART_REFLEX", "0")
    monkeypatch.setenv("MS4_VOICE_EGG_HONORABLE", "0")
    monkeypatch.setattr(voice_module, "_identify_speaker_async", lambda *_a, **_k: None)

    class _OpenFailure:
        def __init__(self, **_):
            pass

        def open(self):
            raise OSError("ws unavailable")

    events: list[tuple[str, dict]] = []
    result = voice_ptt_turn_stream(
        runner=_FakeRunner(),
        audio=b"FAKE_WAV",
        chat_fn=lambda message, stream_callback=None, **_: (
            stream_callback("Fallback audio works."),
            {"text": "Fallback audio works.", "session_id": "s"},
        )[1],
        transcribe_fn=lambda **_: {"text": "hello", "model": "whisper-1"},
        synthesize_fn=lambda **_: {
            "audio_bytes": b"WAV",
            "audio_base64": "V0FW",
            "content_type": "audio/wav",
        },
        emit=lambda name, payload: events.append((name, dict(payload))) or True,
        ws_engine_factory=_OpenFailure,
    )

    fallback = [
        payload for name, payload in events
        if name == "status" and payload.get("phase") == "tts_engine_fallback"
    ]
    assert fallback and fallback[0]["from"] == "ws_super" and fallback[0]["to"] == "rest"
    assert any(name == "audio_chunk" for name, _ in events)
    assert result["metrics"].get("engine") != "ws_super"


def test_ws_constructor_failure_falls_back_before_chat(monkeypatch, pin_ws_super_engine):
    import machine_spirit_4.gateway.voice as voice_module

    _ms3_voice_ready(monkeypatch)
    monkeypatch.setattr(voice_module, "_identify_speaker_async", lambda *_a, **_k: None)
    chat_calls = {"n": 0}

    def fake_chat(message, *, stream_callback=None, **_):
        chat_calls["n"] += 1
        stream_callback("REST setup fallback works. ")
        return {"text": "REST setup fallback works.", "session_id": "s"}

    class _ConstructorFailure:
        def __init__(self, **_):
            raise OSError("constructor unavailable")

    events: list[tuple[str, dict]] = []
    result = voice_ptt_turn_stream(
        runner=_FakeRunner(),
        audio=b"FAKE_WAV",
        chat_fn=fake_chat,
        transcribe_fn=lambda **_: {"text": "hello", "model": "whisper-1"},
        synthesize_fn=lambda **_: {
            "audio_bytes": b"WAV",
            "audio_base64": "V0FW",
            "content_type": "audio/wav",
        },
        emit=lambda name, payload: events.append((name, dict(payload))) or True,
        ws_engine_factory=_ConstructorFailure,
    )

    assert chat_calls["n"] == 1
    assert any(name == "audio_chunk" for name, _ in events)
    assert result["metrics"].get("engine") != "ws_super"


def test_ws_non_cancellable_sender_falls_back_before_starting_a_thread(monkeypatch, pin_ws_super_engine):
    import machine_spirit_4.gateway.voice as voice_module

    _ms3_voice_ready(monkeypatch)
    monkeypatch.setattr(voice_module, "_identify_speaker_async", lambda *_a, **_k: None)
    push_called = threading.Event()

    class _NonCancellableWsEngine:
        def __init__(self, **_):
            pass

        def open(self):
            pass

        def push(self, _text):
            push_called.set()
            threading.Event().wait()

        def close(self):
            pass

    result = voice_ptt_turn_stream(
        runner=_FakeRunner(),
        audio=b"FAKE_WAV",
        chat_fn=lambda message, stream_callback=None, **_: (
            stream_callback("REST bounded fallback works. "),
            {"text": "REST bounded fallback works.", "session_id": "s"},
        )[1],
        transcribe_fn=lambda **_: {"text": "hello", "model": "whisper-1"},
        synthesize_fn=lambda **_: {
            "audio_bytes": b"WAV",
            "audio_base64": "V0FW",
            "content_type": "audio/wav",
        },
        emit=lambda _name, _payload: True,
        ws_engine_factory=_NonCancellableWsEngine,
    )

    assert push_called.is_set() is False
    assert result["metrics"].get("engine") != "ws_super"
    assert not any(
        thread.name == "ms4-tts-ws-send" and thread.is_alive()
        for thread in threading.enumerate()
    )


def test_ws_optional_import_failure_falls_back_before_chat(monkeypatch, pin_ws_super_engine):
    import builtins
    import machine_spirit_4.gateway.voice as voice_module

    _ms3_voice_ready(monkeypatch)
    monkeypatch.setattr(voice_module, "_identify_speaker_async", lambda *_a, **_k: None)
    real_import = builtins.__import__

    def fail_ws_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.endswith("tts_super_ws"):
            raise ImportError("websockets unavailable")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_ws_import)
    events: list[tuple[str, dict]] = []
    result = voice_ptt_turn_stream(
        runner=_FakeRunner(),
        audio=b"FAKE_WAV",
        chat_fn=lambda message, stream_callback=None, **_: (
            stream_callback("REST import fallback works. "),
            {"text": "REST import fallback works.", "session_id": "s"},
        )[1],
        transcribe_fn=lambda **_: {"text": "hello", "model": "whisper-1"},
        synthesize_fn=lambda **_: {
            "audio_bytes": b"WAV",
            "audio_base64": "V0FW",
            "content_type": "audio/wav",
        },
        emit=lambda name, payload: events.append((name, dict(payload))) or True,
    )

    assert any(name == "audio_chunk" for name, _ in events)
    assert result["metrics"].get("engine") != "ws_super"


def test_ws_first_send_failure_reuses_chat_text_for_rest(monkeypatch, pin_ws_super_engine):
    import machine_spirit_4.gateway.voice as voice_module

    _ms3_voice_ready(monkeypatch)
    monkeypatch.setattr(voice_module, "_identify_speaker_async", lambda *_a, **_k: None)
    monkeypatch.setattr(voice_module, "synthesize", lambda **_: {
        "audio_bytes": b"WAV",
        "audio_base64": "V0FW",
        "content_type": "audio/wav",
    })
    chat_calls = {"n": 0}

    def fake_chat(message, *, stream_callback=None, **_):
        chat_calls["n"] += 1
        stream_callback("Collected reply text is reused for fallback. ")
        return {"text": "", "session_id": "s"}

    class _FirstSendFailure:
        def __init__(self, *, voice, **_):
            self.voice = voice

        def open(self):
            pass

        def push(self, _text):
            raise OSError("first WS send failed")

        def flush(self):
            pytest.fail("flush must not follow a failed first send")

        def wait_for_final(self, timeout=60):
            pytest.fail("wait_for_final must not follow a failed first send")

        def close(self):
            pass

        def cancel_pending_send(self):
            self.close()

        def metrics(self):
            return {"engine": "ws_super", "chunks_emitted": 0, "error": None}

    events: list[tuple[str, dict]] = []
    result = voice_ptt_turn_stream(
        runner=_FakeRunner(),
        audio=b"FAKE_WAV",
        chat_fn=fake_chat,
        transcribe_fn=lambda **_: {"text": "hello", "model": "whisper-1"},
        synthesize_fn=lambda **_: pytest.fail("chat must not be replayed through full REST"),
        emit=lambda name, payload: events.append((name, dict(payload))) or True,
        ws_engine_factory=_FirstSendFailure,
    )

    assert chat_calls["n"] == 1
    assert any(name == "audio_chunk" for name, _ in events)
    assert result["metrics"]["rest_fallback_used"] is True
    assert result["metrics"]["audio_chunks"] == 1


def test_ws_failure_after_accepted_audio_never_replays_rest(monkeypatch, pin_ws_super_engine):
    import machine_spirit_4.gateway.voice as voice_module

    _ms3_voice_ready(monkeypatch)
    monkeypatch.setattr(voice_module, "_identify_speaker_async", lambda *_a, **_k: None)
    monkeypatch.setattr(
        voice_module,
        "synthesize",
        lambda **_: pytest.fail("accepted WS audio must disable REST replay"),
    )

    class _AudioThenFailure:
        def __init__(self, *, emit, **_):
            self.emit = emit

        def open(self):
            pass

        def push(self, text):
            self.emit("audio_chunk", {
                "index": 0,
                "text": text,
                "audio_base64": "AAA",
                "audio_mime": "audio/wav",
                "tts_ms": 0,
                "held_for_inorder_ms": 0,
                "engine": "ws_super",
            })
            raise OSError("send failed after audio")

        def flush(self):
            pass

        def wait_for_final(self, timeout=60):
            return True

        def close(self):
            pass

        def cancel_pending_send(self):
            self.close()

        def metrics(self):
            return {"engine": "ws_super", "chunks_emitted": 1, "error": None}

    events: list[tuple[str, dict]] = []
    with pytest.raises(OSError, match="after audio"):
        voice_ptt_turn_stream(
            runner=_FakeRunner(),
            audio=b"FAKE_WAV",
            chat_fn=lambda message, stream_callback=None, **_: (
                stream_callback("WS audio starts before failure. "),
                {"text": "WS audio starts before failure.", "session_id": "s"},
            )[1],
            transcribe_fn=lambda **_: {"text": "hello", "model": "whisper-1"},
            synthesize_fn=lambda **_: {},
            emit=lambda name, payload: events.append((name, dict(payload))) or True,
            ws_engine_factory=_AudioThenFailure,
        )

    assert len([1 for name, _ in events if name == "audio_chunk"]) == 1
    assert not any(
        name == "status" and payload.get("phase") == "tts_engine_fallback_after_zero_audio"
        for name, payload in events
    )


def test_ws_engine_skips_wait_for_final_when_client_disconnects(monkeypatch):
    """Barge-in plumbing: when the client aborts the SSE fetch
    mid-turn, the gateway clears ``client_alive`` and
    voice_ptt_turn_stream MUST stop waiting for HiveMind TTS to
    finish. Without this, every barged turn would burn the full 60s
    wait_for_final budget on audio nobody is listening to."""
    _ms3_voice_ready(monkeypatch)

    def fake_transcribe(**_):
        return {"text": "tell me a story", "model": "whisper-1"}

    def fake_chat(message, *, session_id=None, model=None, stream_callback=None, **_):
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

        def cancel_pending_send(self):
            self.close()

        def metrics(self):
            return {"engine": "ws_super", "voice": self.voice, "connect_ms": 1,
                    "first_audio_ms": None, "chunks_emitted": 0, "bytes_received": 0,
                    "is_final_seen": False, "error": None}

    client_alive = threading.Event()
    client_alive.set()
    client_alive.clear()  # simulate the gateway clearing it BEFORE the engine path runs wait_for_final

    runner = _FakeRunner()
    with pytest.raises(VoiceUnavailable, match="cancelled"):
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
    import machine_spirit_4.gateway.voice as voice_module

    monkeypatch.setattr(voice_module, "_identify_speaker_async", lambda *_a, **_k: None)
    monkeypatch.setattr(voice_module, "synthesize", lambda **_: {
        "audio_bytes": b"WAV",
        "audio_base64": "V0FW",
        "content_type": "audio/wav",
    })

    def fake_transcribe(**_):
        return {"text": "hi", "model": "whisper-1"}

    def fake_chat(message, *, session_id=None, model=None, stream_callback=None, **_):
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

        def cancel_pending_send(self):
            self.close()

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


def test_ws_sender_timeout_cancels_and_joins_before_rest_fallback(monkeypatch, pin_ws_super_engine):
    import machine_spirit_4.gateway.voice as voice_module

    _ms3_voice_ready(monkeypatch)
    monkeypatch.setenv("MS4_TTS_WS_SEND_TIMEOUT_S", "0.05")
    monkeypatch.setenv("MS4_TTS_WS_SENDER_CLEANUP_TIMEOUT_S", "1")
    monkeypatch.setattr(voice_module, "_identify_speaker_async", lambda *_a, **_k: None)
    monkeypatch.setattr(voice_module, "synthesize", lambda **_: {
        "audio_bytes": b"WAV",
        "audio_base64": "V0FW",
        "content_type": "audio/wav",
    })
    push_started = threading.Event()
    release_push = threading.Event()
    send_cancelled = threading.Event()

    class _BlockingWsEngine:
        def __init__(self, **_):
            pass

        def open(self):
            pass

        def push(self, _text):
            push_started.set()
            release_push.wait()

        def flush(self):
            pass

        def wait_for_final(self, timeout=60):
            return True

        def close(self):
            pass

        def cancel_pending_send(self):
            send_cancelled.set()
            release_push.set()

        def metrics(self):
            return {"engine": "ws_super", "chunks_emitted": 0, "error": None}

    result = voice_ptt_turn_stream(
        runner=_FakeRunner(),
        audio=b"FAKE_WAV",
        chat_fn=lambda message, stream_callback=None, **_: (
            stream_callback("one two three four five "),
            push_started.wait(timeout=2.0),
            {"text": "one two three four five", "session_id": "s"},
        )[2],
        transcribe_fn=lambda **_: {"text": "hello", "model": "whisper-1"},
        synthesize_fn=lambda **_: pytest.fail("full REST turn must not replay chat"),
        emit=lambda _name, _payload: True,
        ws_engine_factory=_BlockingWsEngine,
    )

    assert push_started.is_set()
    assert send_cancelled.is_set()
    assert result["metrics"]["rest_fallback_used"] is True
    assert not any(
        thread.name == "ms4-tts-ws-send" and thread.is_alive()
        for thread in threading.enumerate()
    )


def test_ws_sender_capacity_cancels_blocked_transport_and_joins(monkeypatch):
    import machine_spirit_4.gateway.voice as voice_module

    _ms3_voice_ready(monkeypatch)
    monkeypatch.setenv("MS4_TTS_WS_QUEUE_MAX_ITEMS", "3")
    monkeypatch.setenv("MS4_TTS_WS_QUEUE_MAX_BYTES", "4096")
    monkeypatch.setenv("MS4_TTS_WS_QUEUE_PUT_TIMEOUT_S", "0.05")
    monkeypatch.setenv("MS4_TTS_WS_SENDER_CLEANUP_TIMEOUT_S", "0.5")
    push_started = threading.Event()
    release_push = threading.Event()
    send_cancelled = threading.Event()
    callback_results: list[bool | None] = []

    original_queue = queue.Queue

    class _TrackingQueue(original_queue):
        instances: list["_TrackingQueue"] = []

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.max_observed = 0
            self.instances.append(self)

        def put(self, item, block=True, timeout=None):
            result = super().put(item, block=block, timeout=timeout)
            self.max_observed = max(self.max_observed, self.qsize())
            return result

    class _BlockingWsEngine:
        def __init__(self, **_kwargs):
            pass

        def open(self):
            pass

        def push(self, _text):
            push_started.set()
            release_push.wait()

        def flush(self):
            pytest.fail("a capacity-cancelled sender must not flush")

        def wait_for_final(self, timeout=60):
            pytest.fail("a capacity-cancelled sender must not wait for final audio")

        def close(self):
            release_push.set()

        def cancel_pending_send(self):
            send_cancelled.set()
            release_push.set()

        def metrics(self):
            return {"engine": "ws_super", "chunks_emitted": 0, "error": None}

    monkeypatch.setattr(voice_module.queue, "Queue", _TrackingQueue)

    def chat(_message, *, stream_callback=None, **_kwargs):
        callback_results.append(stream_callback("The first blocked transport sentence is ready. "))
        assert push_started.wait(timeout=0.5)
        for index in range(20):
            accepted = stream_callback(
                f"Queued sentence {index} has enough complete words now. "
            )
            callback_results.append(accepted)
            if accepted is False:
                break
        return {"text": "blocked queue reply", "session_id": "queue-capacity"}

    started_at = time.monotonic()
    with pytest.raises(VoiceUnavailable, match="queue capacity"):
        voice_module._run_ws_super_engine(
            runner=_FakeRunner(),
            transcript="queue capacity",
            session_id="queue-capacity",
            model=None,
            depth_model=None,
            tts_voice="vega",
            asr_result={"model": "whisper-1"},
            asr_ms=1,
            tts_model=None,
            response_format="wav",
            chat_call=chat,
            emit=lambda _event, _payload: True,
            t_total=time.monotonic(),
            start_speaker_id=lambda: None,
            finish_speaker_id=lambda: None,
            ws_engine_factory=_BlockingWsEngine,
            lifecycle={},
        )

    observed = _TrackingQueue.instances[0]
    assert time.monotonic() - started_at < 1.0
    assert observed.maxsize == 3
    assert observed.max_observed <= observed.maxsize
    assert callback_results[-1] is False
    assert send_cancelled.is_set()
    assert not any(
        thread.name == "ms4-tts-ws-send" and thread.is_alive()
        for thread in threading.enumerate()
    )


def test_ws_sender_byte_capacity_fails_immediately_and_joins(monkeypatch):
    import machine_spirit_4.gateway.voice as voice_module

    _ms3_voice_ready(monkeypatch)
    monkeypatch.setenv("MS4_TTS_WS_QUEUE_MAX_ITEMS", "8")
    monkeypatch.setenv("MS4_TTS_WS_QUEUE_MAX_BYTES", "16")
    monkeypatch.setenv("MS4_TTS_WS_QUEUE_PUT_TIMEOUT_S", "0.2")
    monkeypatch.setenv("MS4_TTS_WS_SENDER_CLEANUP_TIMEOUT_S", "0.5")
    send_cancelled = threading.Event()

    class _ByteBoundWsEngine:
        def __init__(self, **_kwargs):
            pass

        def open(self):
            pass

        def push(self, _text):
            pytest.fail("an oversized fragment must never reach transport")

        def flush(self):
            pytest.fail("an oversized fragment must cancel the turn")

        def wait_for_final(self, timeout=60):
            pytest.fail("an oversized fragment must cancel the turn")

        def close(self):
            pass

        def cancel_pending_send(self):
            send_cancelled.set()

        def metrics(self):
            return {"engine": "ws_super", "chunks_emitted": 0, "error": None}

    def chat(_message, *, stream_callback=None, **_kwargs):
        assert stream_callback("This fragment exceeds sixteen UTF-8 bytes. ") is False
        return {"text": "oversized queue reply", "session_id": "queue-bytes"}

    started_at = time.monotonic()
    with pytest.raises(VoiceUnavailable, match="larger than the byte limit"):
        voice_module._run_ws_super_engine(
            runner=_FakeRunner(),
            transcript="queue bytes",
            session_id="queue-bytes",
            model=None,
            depth_model=None,
            tts_voice="vega",
            asr_result={"model": "whisper-1"},
            asr_ms=1,
            tts_model=None,
            response_format="wav",
            chat_call=chat,
            emit=lambda _event, _payload: True,
            t_total=time.monotonic(),
            start_speaker_id=lambda: None,
            finish_speaker_id=lambda: None,
            ws_engine_factory=_ByteBoundWsEngine,
            lifecycle={},
        )

    assert time.monotonic() - started_at < 0.5
    assert send_cancelled.is_set()
    assert not any(
        thread.name == "ms4-tts-ws-send" and thread.is_alive()
        for thread in threading.enumerate()
    )


def test_ws_sender_disconnect_cancels_and_joins_before_turn_returns(monkeypatch, pin_ws_super_engine):
    _ms3_voice_ready(monkeypatch)
    monkeypatch.setenv("MS4_TTS_WS_SEND_TIMEOUT_S", "10")
    monkeypatch.setenv("MS4_TTS_WS_SENDER_CLEANUP_TIMEOUT_S", "1")
    client_alive = threading.Event()
    client_alive.set()
    push_started = threading.Event()
    release_push = threading.Event()
    send_cancelled = threading.Event()

    class _DisconnectBlockingWsEngine:
        def __init__(self, **_):
            pass

        def open(self):
            pass

        def push(self, _text):
            push_started.set()
            release_push.wait()

        def flush(self):
            pass

        def wait_for_final(self, timeout=60):
            pytest.fail("disconnected turn must skip wait_for_final")

        def close(self):
            pass

        def cancel_pending_send(self):
            send_cancelled.set()
            release_push.set()

        def metrics(self):
            return {"engine": "ws_super", "chunks_emitted": 0, "error": None}

    def fake_chat(message, *, stream_callback=None, **_):
        assert stream_callback("one two three four five ") is not False
        assert push_started.wait(timeout=2.0)
        client_alive.clear()
        return {"text": "one two three four five", "session_id": "s"}

    with pytest.raises(VoiceUnavailable, match="cancelled"):
        voice_ptt_turn_stream(
            runner=_FakeRunner(),
            audio=b"FAKE_WAV",
            chat_fn=fake_chat,
            transcribe_fn=lambda **_: {"text": "hello", "model": "whisper-1"},
            synthesize_fn=lambda **_: pytest.fail("disconnected turn must not use REST"),
            emit=lambda _name, _payload: True,
            ws_engine_factory=_DisconnectBlockingWsEngine,
            client_alive=client_alive,
        )

    assert send_cancelled.is_set()
    assert not any(
        thread.name == "ms4-tts-ws-send" and thread.is_alive()
        for thread in threading.enumerate()
    )


def test_voice_stream_invalid_engine_falls_back_to_rest(monkeypatch):
    _ms3_voice_ready(monkeypatch)

    def fake_transcribe(**_):
        return {"text": "Hi.", "model": "whisper-1"}

    def fake_chat(message, *, session_id=None, model=None, stream_callback=None, **_):
        stream_callback("ok")
        return {"text": "ok", "session_id": "s"}

    events: list[tuple[str, dict]] = []
    result = voice_ptt_turn_stream(
        runner=_FakeRunner(),
        audio=b"FAKE_WAV",
        chat_fn=fake_chat,
        transcribe_fn=fake_transcribe,
        synthesize_fn=lambda **_: {
            "audio_bytes": b"WAV",
            "audio_base64": "V0FW",
            "content_type": "audio/wav",
        },
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

    def fake_chat(message, *, session_id=None, model=None, stream_callback=None, **_):
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
        engine="rest",
    )
    assert first_audio_emitted_at, "expected at least one audio_chunk before chat finished"
    assert chat_done_at, "fake chat should have ended"
    # First audio chunk MUST come strictly before chat finishes —
    # that's the latency win the whole architecture is for.
    assert first_audio_emitted_at[0] < chat_done_at[0], (
        f"first audio at {first_audio_emitted_at[0]}, chat ended at {chat_done_at[0]}"
    )


# ---------------------------------------------------------------------------
# Latency pass: last-used Face model tracking (drives the keep-warm loop)
# + fully-async speaker ID (off the critical path).
# ---------------------------------------------------------------------------


def test_record_and_last_face_model():
    record_face_model("llama3.1:8b")
    assert last_face_model() == "llama3.1:8b"
    # None / empty (an Auto-pick turn) must NOT clobber the last concrete pick.
    record_face_model(None)
    record_face_model("")
    assert last_face_model() == "llama3.1:8b"
    # A new concrete pick updates it.
    record_face_model("phi4-mini:latest")
    assert last_face_model() == "phi4-mini:latest"


def test_voice_speaker_id_async_gate_default_on(monkeypatch):
    monkeypatch.delenv("MS4_VOICE_SPEAKER_ID_ASYNC", raising=False)
    assert _voice_speaker_id_async() is True
    monkeypatch.setenv("MS4_VOICE_SPEAKER_ID_ASYNC", "0")
    assert _voice_speaker_id_async() is False
    monkeypatch.setenv("MS4_VOICE_SPEAKER_ID_ASYNC", "on")
    assert _voice_speaker_id_async() is True


def test_identify_speaker_async_emits_late_event(monkeypatch):
    """Async speaker ID never blocks the turn; when diarization resolves
    to an accepted match it emits a late ``speaker`` event the UI uses to
    decorate the already-shipped transcript bubble."""
    import machine_spirit_4.gateway.voice_identity as vi

    monkeypatch.setattr(
        vi, "identify_speaker_from_wav",
        lambda url, audio: {"accepted": True, "name": "Operator", "score": 0.92},
    )
    events: list[tuple[str, dict]] = []
    thread = _identify_speaker_async(
        "http://hive", b"WAVDATA", lambda ev, p: events.append((ev, p)) or True
    )
    assert thread is not None
    thread.join()
    assert events and events[0][0] == "speaker"
    assert events[0][1]["speaker"]["name"] == "Operator"


def test_identify_speaker_async_silent_on_failure(monkeypatch):
    """A failed/raised identify emits NOTHING (the transcript already
    carried speaker:None) — fail-soft, never on the critical path."""
    import machine_spirit_4.gateway.voice_identity as vi

    def _boom(url, audio):
        raise RuntimeError("diarization backend down")

    monkeypatch.setattr(vi, "identify_speaker_from_wav", _boom)
    events: list[tuple[str, dict]] = []
    thread = _identify_speaker_async(
        "http://hive", b"WAVDATA", lambda ev, p: events.append((ev, p)) or True
    )
    assert thread is not None
    thread.join()
    assert events == []


def test_identify_speaker_async_noop_on_empty_audio():
    events: list[tuple[str, dict]] = []
    thread = _identify_speaker_async(
        "http://hive", b"", lambda ev, p: events.append((ev, p)) or True
    )
    assert thread is None
    assert events == []


def test_speaker_id_cannot_delay_terminal_or_emit_after_terminal(monkeypatch):
    import machine_spirit_4.gateway.voice_identity as vi

    _ms3_voice_ready(monkeypatch)
    monkeypatch.setenv("MS4_VOICE_SPEAKER_ID_TERMINAL_BUDGET_S", "0.2")
    monkeypatch.setenv("MS4_VOICE_SMART_REFLEX", "0")
    monkeypatch.setenv("MS4_VOICE_EGG_HONORABLE", "0")
    identify_started = threading.Event()
    identify_finished = threading.Event()
    cancel_forwarded = threading.Event()

    def blocking_identity(_url, _audio, *, timeout, cancel_event):
        identify_started.set()
        assert timeout > 0
        assert cancel_event.wait(timeout=1.0)
        cancel_forwarded.set()
        identify_finished.set()
        return {"accepted": True, "name": "Too Late"}

    monkeypatch.setattr(vi, "identify_speaker_from_wav", blocking_identity)
    events: list[tuple[str, dict]] = []
    turn_done = threading.Event()
    turn_error: list[BaseException] = []
    turn_result: list[dict] = []

    def run_turn():
        try:
            turn_result.append(voice_ptt_turn_stream(
                runner=_FakeRunner(),
                audio=b"FAKE_WAV",
                chat_fn=lambda message, stream_callback=None, **_: (
                    stream_callback("Reply audio is ready. "),
                    {"text": "Reply audio is ready.", "session_id": "s"},
                )[1],
                transcribe_fn=lambda **_: {"text": "hello", "model": "whisper-1"},
                synthesize_fn=lambda **_: {
                    "audio_bytes": b"WAV",
                    "audio_base64": "V0FW",
                    "content_type": "audio/wav",
                },
                emit=lambda name, payload: events.append((name, dict(payload))) or True,
                engine="rest",
            ))
        except BaseException as exc:  # surfaced after the thread joins
            turn_error.append(exc)
        finally:
            turn_done.set()

    turn = threading.Thread(target=run_turn, name="speaker-terminal-contract")
    turn.start()
    assert identify_started.wait(timeout=2.0)
    assert turn_done.wait(timeout=2.0), "terminal completion waited on speaker ID"
    turn.join(timeout=0.1)
    assert not turn.is_alive()
    assert turn_error == []
    assert cancel_forwarded.is_set()
    assert identify_finished.is_set()
    assert not any(name == "speaker" for name, _ in events)
    lifecycle = turn_result[0]["metrics"]["lifecycle"]
    assert lifecycle["speaker_threads_started"] == 1
    assert lifecycle["speaker_threads_joined"] == 1
    assert lifecycle["speaker_thread_join_timeouts"] == 0
    assert not any(
        thread.name == "ms4-speaker-id-async" and thread.is_alive()
        for thread in threading.enumerate()
    )


def test_first_chunk_deadline_flushes_only_complete_words(monkeypatch):
    import machine_spirit_4.gateway.voice as voice_module

    _ms3_voice_ready(monkeypatch)
    monkeypatch.setenv("MS4_VOICE_FIRST_CHUNK_DEADLINE_S", "0.04")
    monkeypatch.setenv("MS4_VOICE_SMART_REFLEX", "0")
    monkeypatch.setenv("MS4_VOICE_EGG_HONORABLE", "0")
    monkeypatch.setattr(voice_module, "TTS_FILTER_ENABLED", False)
    monkeypatch.setattr(voice_module, "_identify_speaker_async", lambda *_a, **_k: None)
    first_scheduled = threading.Event()
    scheduled: list[str] = []

    def emit(name, payload):
        if name == "chunk_scheduled":
            scheduled.append(payload["text"])
            first_scheduled.set()
        return True

    def chat(_message, *, stream_callback=None, **_kwargs):
        assert stream_callback("alpha beta gamma delta epsilon incom") is not False
        assert first_scheduled.wait(timeout=0.25)
        stream_callback("plete sentence. ")
        return {
            "text": "alpha beta gamma delta epsilon incomplete sentence.",
            "session_id": "s",
        }

    result = voice_ptt_turn_stream(
        runner=_FakeRunner(),
        audio=b"FAKE_WAV",
        chat_fn=chat,
        transcribe_fn=lambda **_: {"text": "hello", "model": "whisper-1"},
        synthesize_fn=lambda **_: {
            "audio_bytes": b"WAV",
            "audio_base64": "V0FW",
            "content_type": "audio/wav",
        },
        emit=emit,
        chunker=SentenceChunker(first_chunk_min_words=100),
        engine="rest",
    )

    assert scheduled[0] == "alpha beta gamma delta epsilon"
    assert "incom" not in scheduled[0]
    assert "incomplete sentence." in scheduled[1]
    assert result["metrics"]["first_chunk_deadline_fired"] is True
    assert result["metrics"]["first_chunk_deadline_flushes"] == 1


@pytest.mark.parametrize(
    ("reply", "deltas"),
    [
        pytest.param(
            "Hi there! 🎙 Your physical microphone is clear and Oracle speech stays continuous.",
            (
                "Hi there! 🎙 ",
                "Your physical microphone is clear and Oracle speech stays continuous.",
            ),
            id="microphone-emoji-greeting",
        ),
        pytest.param(
            "The level is 1.5 and remains stable throughout this spoken response.",
            (
                "The level is 1.",
                "5 and remains stable throughout this spoken response.",
            ),
            id="decimal-stream",
        ),
        pytest.param(
            "All set.",
            ("All set.",),
            id="entirely-short-reply",
        ),
    ],
)
def test_rest_first_chunk_floor_is_applied_after_speech_sanitization(
    monkeypatch,
    reply,
    deltas,
):
    import machine_spirit_4.gateway.voice as voice_module
    from machine_spirit_4.gateway.spoken_text_filter import sanitize_for_speech

    _ms3_voice_ready(monkeypatch)
    monkeypatch.setenv("MS4_VOICE_FIRST_CHUNK_DEADLINE_S", "0.01")
    monkeypatch.setenv("MS4_VOICE_SMART_REFLEX", "0")
    monkeypatch.setenv("MS4_VOICE_EGG_HONORABLE", "0")
    monkeypatch.delenv("MS4_VOICE_TTS_ENGINE", raising=False)
    monkeypatch.delenv("MS4_VOICE_TTS_VOICE", raising=False)
    monkeypatch.setattr(voice_module, "TTS_FILTER_ENABLED", True)
    monkeypatch.setattr(voice_module, "_identify_speaker_async", lambda *_a, **_k: None)

    chat_finished = threading.Event()
    synthesis_calls: list[dict[str, Any]] = []
    events: list[tuple[str, dict[str, Any]]] = []

    def chat(_message, *, stream_callback=None, **_kwargs):
        for index, delta in enumerate(deltas):
            assert stream_callback(delta) is not False
            if index == 0:
                time.sleep(0.03)
        chat_finished.set()
        return {"text": reply, "session_id": "continuity"}

    def synthesize_rest(*, text, model, voice, **_kwargs):
        synthesis_calls.append({
            "text": text,
            "model": model,
            "voice": voice,
            "after_chat": chat_finished.is_set(),
        })
        return {
            "audio_bytes": b"WAV",
            "audio_base64": "V0FW",
            "content_type": "audio/wav",
        }

    result = voice_ptt_turn_stream(
        runner=_FakeRunner(),
        audio=b"FAKE_WAV",
        chat_fn=chat,
        transcribe_fn=lambda **_: {"text": "continuity check", "model": "whisper-1"},
        synthesize_fn=synthesize_rest,
        emit=lambda name, payload: events.append((name, dict(payload))) or True,
        chunker=SentenceChunker(first_chunk_min_words=5),
        tts_model="tts-1",
        tts_voice="vega",
        engine="rest",
    )

    assert voice_module._default_tts_engine() == "rest"
    assert voice_module._default_tts_voice() == "vega"
    assert synthesis_calls
    assert all(call["model"] == "tts-1" for call in synthesis_calls)
    assert all(call["voice"] == "vega" for call in synthesis_calls)
    assert result["tts_model"] == "tts-1"

    expected = sanitize_for_speech(reply).strip()
    synthesized = " ".join(call["text"] for call in synthesis_calls)
    assert synthesized == expected

    first_spoken_words = len(sanitize_for_speech(synthesis_calls[0]["text"]).split())
    whole_reply_words = len(expected.split())
    assert first_spoken_words >= 5 or whole_reply_words < 5
    if whole_reply_words < 5:
        assert len(synthesis_calls) == 1
        assert synthesis_calls[0]["after_chat"] is True

    audio = [payload for name, payload in events if name == "audio_chunk"]
    indices = [payload["index"] for payload in audio]
    assert indices == list(range(len(synthesis_calls)))
    assert [payload["text"] for payload in audio] == [
        call["text"] for call in synthesis_calls
    ]


@pytest.mark.parametrize(
    ("deltas", "expects_after"),
    [
        pytest.param(
            (
                "Before the code, this guidance stays audible. ```py",
                "thon\nprint('secret')\n``",
                "` After the code, this conclusion stays audible.",
            ),
            True,
            id="complete-fence-split-across-rest-chunks",
        ),
        pytest.param(
            (
                "Before the code, this guidance stays audible. ```py",
                "thon\nprint('secret')\nunterminated",
            ),
            False,
            id="unterminated-fence-is-dropped-on-rest-flush",
        ),
    ],
)
def test_rest_filter_preserves_state_across_fenced_code(
    monkeypatch,
    deltas,
    expects_after,
):
    import machine_spirit_4.gateway.voice as voice_module

    _ms3_voice_ready(monkeypatch)
    monkeypatch.setenv("MS4_VOICE_FIRST_CHUNK_DEADLINE_S", "0")
    monkeypatch.setenv("MS4_VOICE_SMART_REFLEX", "0")
    monkeypatch.setenv("MS4_VOICE_EGG_HONORABLE", "0")
    monkeypatch.setattr(voice_module, "TTS_FILTER_ENABLED", True)
    monkeypatch.setattr(voice_module, "_identify_speaker_async", lambda *_a, **_k: None)

    reply = "".join(deltas)
    events: list[tuple[str, dict[str, Any]]] = []
    synthesized: list[str] = []

    def chat(_message, *, stream_callback=None, **_kwargs):
        for delta in deltas:
            assert stream_callback(delta) is not False
        return {"text": reply, "session_id": "rest-fence"}

    def synthesize_rest(*, text, **_kwargs):
        synthesized.append(text)
        return {
            "audio_bytes": b"WAV",
            "audio_base64": "V0FW",
            "content_type": "audio/wav",
        }

    result = voice_ptt_turn_stream(
        runner=_FakeRunner(),
        audio=b"FAKE_WAV",
        chat_fn=chat,
        transcribe_fn=lambda **_: {"text": "explain the example", "model": "whisper-1"},
        synthesize_fn=synthesize_rest,
        emit=lambda name, payload: events.append((name, dict(payload))) or True,
        chunker=SentenceChunker(first_chunk_min_words=5),
        tts_model="tts-1",
        tts_voice="vega",
        engine="rest",
    )

    spoken = " ".join(synthesized)
    scheduled = " ".join(
        payload["text"] for name, payload in events if name == "chunk_scheduled"
    )
    assert "Before the code, this guidance stays audible." in spoken
    assert ("After the code, this conclusion stays audible." in spoken) is expects_after
    assert "print" not in spoken and "secret" not in spoken and "unterminated" not in spoken
    assert scheduled == spoken
    assert result["metrics"]["audio_errors"] == 0


def test_rest_weak_comma_balances_short_single_sentence(monkeypatch):
    import machine_spirit_4.gateway.voice as voice_module

    _ms3_voice_ready(monkeypatch)
    monkeypatch.setenv("MS4_VOICE_FIRST_CHUNK_DEADLINE_S", "0")
    monkeypatch.setenv("MS4_VOICE_SMART_REFLEX", "0")
    monkeypatch.setenv("MS4_VOICE_EGG_HONORABLE", "0")
    monkeypatch.delenv("MS4_VOICE_TTS_ENGINE", raising=False)
    monkeypatch.delenv("MS4_VOICE_TTS_VOICE", raising=False)
    monkeypatch.setattr(voice_module, "TTS_FILTER_ENABLED", True)
    monkeypatch.setattr(voice_module, "_identify_speaker_async", lambda *_a, **_k: None)

    reply = (
        "I'm the MS4 Face Lobe, the foreground voice of a Machine Spirit "
        "in this conversation with you."
    )
    stream_step = [0]
    schedule_steps: list[int] = []
    synthesis_calls: list[dict[str, Any]] = []
    events: list[tuple[str, dict[str, Any]]] = []

    def emit(name, payload):
        events.append((name, dict(payload)))
        if name == "chunk_scheduled":
            schedule_steps.append(stream_step[0])
        return True

    def chat(_message, *, stream_callback=None, **_kwargs):
        words = reply.split()
        for step, word in enumerate(words, start=1):
            stream_step[0] = step
            suffix = "" if step == len(words) else " "
            assert stream_callback(word + suffix) is not False
        return {"text": reply, "session_id": "physical-room"}

    def synthesize_rest(*, text, model, voice, **_kwargs):
        synthesis_calls.append({"text": text, "model": model, "voice": voice})
        return {
            "audio_bytes": b"WAV",
            "audio_base64": "V0FW",
            "content_type": "audio/wav",
        }

    result = voice_ptt_turn_stream(
        runner=_FakeRunner(),
        audio=b"FAKE_WAV",
        chat_fn=chat,
        transcribe_fn=lambda **_: {"text": "physical room", "model": "whisper-1"},
        synthesize_fn=synthesize_rest,
        emit=emit,
        tts_pool_size=2,
        tts_model="tts-1",
        tts_voice="vega",
        engine="rest",
    )

    weak_comma_split = [
        "I'm the MS4 Face Lobe",
        "the foreground voice of a Machine Spirit in this conversation with you.",
    ]
    scheduled = [payload["text"] for name, payload in events if name == "chunk_scheduled"]
    assert scheduled != weak_comma_split, "reproduced the physical 6/13 weak-comma split"

    expected_chunks = [
        "I'm the MS4 Face Lobe, the",
        "foreground voice of a Machine Spirit in this conversation with you.",
    ]
    assert scheduled == expected_chunks
    assert [call["text"] for call in synthesis_calls] == expected_chunks
    assert [len(chunk.split()) for chunk in scheduled] == [6, 11]

    weak_comma_baseline_step = len("I'm the MS4 Face Lobe,".split())
    assert schedule_steps[0] == weak_comma_baseline_step + 1
    assert schedule_steps[0] / weak_comma_baseline_step <= 1.25

    assert voice_module._default_tts_engine() == "rest"
    assert voice_module._default_tts_voice() == "vega"
    assert all(call["model"] == "tts-1" for call in synthesis_calls)
    assert all(call["voice"] == "vega" for call in synthesis_calls)
    assert result["tts_model"] == "tts-1"
    assert result["transcript"] == "physical room"
    assert result["metrics"]["lifecycle"]["rest_tasks_submitted"] == 2

    audio = [payload for name, payload in events if name == "audio_chunk"]
    assert [payload["index"] for payload in audio] == [0, 1]
    assert [payload["text"] for payload in audio] == expected_chunks
    turn_ids = {payload["turn_id"] for payload in audio}
    assert len(turn_ids) == 1
    turn_id = next(iter(turn_ids))
    assert [payload["chunk_id"] for payload in audio] == [
        f"{turn_id}-c0",
        f"{turn_id}-c1",
    ]


def test_direct_rest_all_tts_down_fails_closed(monkeypatch):
    import machine_spirit_4.gateway.voice as voice_module

    _ms3_voice_ready(monkeypatch)
    monkeypatch.setattr(voice_module, "_identify_speaker_async", lambda *_a, **_k: None)
    events: list[str] = []

    def tts_down(**_kwargs):
        raise VoiceUnavailable("base TTS unavailable")

    with pytest.raises(VoiceUnavailable, match="zero client-written audio"):
        voice_ptt_turn_stream(
            runner=_FakeRunner(),
            audio=b"FAKE_WAV",
            chat_fn=lambda _message, stream_callback=None, **_: (
                stream_callback("A speakable reply sentence. "),
                {"text": "A speakable reply sentence.", "session_id": "s"},
            )[1],
            transcribe_fn=lambda **_: {"text": "hello", "model": "whisper-1"},
            synthesize_fn=tts_down,
            emit=lambda name, _payload: events.append(name) or True,
            engine="rest",
        )

    assert "audio_error" in events
    assert "audio_chunk" not in events


def test_streamed_speakable_text_with_empty_final_text_still_fails_closed(monkeypatch):
    import machine_spirit_4.gateway.voice as voice_module

    _ms3_voice_ready(monkeypatch)
    monkeypatch.setattr(voice_module, "_identify_speaker_async", lambda *_a, **_k: None)
    monkeypatch.setenv("MS4_VOICE_SMART_REFLEX", "0")
    monkeypatch.setenv("MS4_VOICE_EGG_HONORABLE", "0")
    events: list[str] = []

    def chat(_message, *, stream_callback=None, **_kwargs):
        assert stream_callback("A streamed speakable reply sentence. ") is not False
        return {"text": "", "session_id": "empty-final"}

    def tts_down(**_kwargs):
        raise VoiceUnavailable("base TTS unavailable")

    with pytest.raises(VoiceUnavailable, match="zero client-written audio"):
        voice_ptt_turn_stream(
            runner=_FakeRunner(),
            audio=b"FAKE_WAV",
            chat_fn=chat,
            transcribe_fn=lambda **_: {"text": "hello", "model": "whisper-1"},
            synthesize_fn=tts_down,
            emit=lambda name, _payload: events.append(name) or True,
            engine="rest",
        )

    assert "chunk_scheduled" in events
    assert "audio_error" in events
    assert "audio_chunk" not in events


def test_ws_and_rest_all_tts_down_fail_closed(monkeypatch, pin_ws_super_engine):
    import machine_spirit_4.gateway.voice as voice_module

    _ms3_voice_ready(monkeypatch)
    monkeypatch.setattr(voice_module, "_identify_speaker_async", lambda *_a, **_k: None)
    monkeypatch.setattr(
        voice_module,
        "synthesize",
        lambda **_: (_ for _ in ()).throw(VoiceUnavailable("base TTS unavailable")),
    )

    class _ZeroAudioWsEngine:
        def __init__(self, **_kwargs):
            pass

        def open(self):
            pass

        def push(self, _text):
            pass

        def flush(self):
            pass

        def wait_for_final(self, timeout=60):
            return True

        def close(self):
            pass

        def cancel_pending_send(self):
            pass

        def metrics(self):
            return {"engine": "ws_super", "chunks_emitted": 0, "error": "zero media"}

    with pytest.raises(VoiceUnavailable, match="WS and REST produced zero"):
        voice_ptt_turn_stream(
            runner=_FakeRunner(),
            audio=b"FAKE_WAV",
            chat_fn=lambda _message, stream_callback=None, **_: (
                stream_callback("A speakable fallback sentence. "),
                {"text": "A speakable fallback sentence.", "session_id": "s"},
            )[1],
            transcribe_fn=lambda **_: {"text": "hello", "model": "whisper-1"},
            synthesize_fn=lambda **_: {},
            emit=lambda _name, _payload: True,
            ws_engine_factory=_ZeroAudioWsEngine,
        )


def test_disconnect_during_running_rest_cancels_and_joins_workers(monkeypatch):
    import machine_spirit_4.gateway.voice as voice_module

    _ms3_voice_ready(monkeypatch)
    monkeypatch.setenv("MS4_TTS_REST_CLEANUP_TIMEOUT_S", "0.5")
    monkeypatch.setattr(voice_module, "_identify_speaker_async", lambda *_a, **_k: None)
    client_alive = threading.Event()
    client_alive.set()
    synthesis_started = threading.Event()
    synthesis_cancelled = threading.Event()

    def blocking_synthesis(*, cancel_event, **_kwargs):
        synthesis_started.set()
        assert cancel_event.wait(timeout=0.5)
        synthesis_cancelled.set()
        raise VoiceUnavailable("cancelled")

    def chat(_message, *, stream_callback=None, **_kwargs):
        assert stream_callback("one two three four five ") is not False
        assert synthesis_started.wait(timeout=0.5)
        client_alive.clear()
        return {"text": "one two three four five", "session_id": "s"}

    started_at = time.monotonic()
    with pytest.raises(VoiceUnavailable, match="cancelled"):
        voice_ptt_turn_stream(
            runner=_FakeRunner(),
            audio=b"FAKE_WAV",
            chat_fn=chat,
            transcribe_fn=lambda **_: {"text": "hello", "model": "whisper-1"},
            synthesize_fn=blocking_synthesis,
            emit=lambda _name, _payload: True,
            engine="rest",
            client_alive=client_alive,
        )

    assert time.monotonic() - started_at < 1.0
    assert synthesis_cancelled.is_set()
    assert not any(
        thread.name.startswith("ms4-tts") and thread.is_alive()
        for thread in threading.enumerate()
    )


def test_rest_deadline_cancels_and_joins_workers(monkeypatch):
    import machine_spirit_4.gateway.voice as voice_module

    _ms3_voice_ready(monkeypatch)
    monkeypatch.setenv("MS4_TTS_REST_FINAL_TIMEOUT_S", "0.05")
    monkeypatch.setenv("MS4_TTS_REST_CLEANUP_TIMEOUT_S", "0.5")
    monkeypatch.setattr(voice_module, "_identify_speaker_async", lambda *_a, **_k: None)
    synthesis_cancelled = threading.Event()

    def blocking_synthesis(*, cancel_event, **_kwargs):
        assert cancel_event.wait(timeout=0.5)
        synthesis_cancelled.set()
        raise VoiceUnavailable("cancelled")

    started_at = time.monotonic()
    with pytest.raises(VoiceUnavailable, match="hard deadline exceeded"):
        voice_ptt_turn_stream(
            runner=_FakeRunner(),
            audio=b"FAKE_WAV",
            chat_fn=lambda _message, stream_callback=None, **_: (
                stream_callback("one two three four five "),
                {"text": "one two three four five", "session_id": "s"},
            )[1],
            transcribe_fn=lambda **_: {"text": "hello", "model": "whisper-1"},
            synthesize_fn=blocking_synthesis,
            emit=lambda _name, _payload: True,
            engine="rest",
        )

    assert time.monotonic() - started_at < 1.0
    assert synthesis_cancelled.is_set()
    assert not any(
        thread.name.startswith("ms4-tts") and thread.is_alive()
        for thread in threading.enumerate()
    )


def test_rest_transport_cancel_closes_running_http_request():
    import machine_spirit_4.gateway.voice as voice_module

    request_started = threading.Event()
    release_handler = threading.Event()

    class _SlowSpeechHandler(BaseHTTPRequestHandler):
        def log_message(self, *_args, **_kwargs):
            return

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0") or "0")
            self.rfile.read(length)
            request_started.set()
            release_handler.wait(timeout=2.0)
            try:
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", "3")
                self.end_headers()
                self.wfile.write(b"WAV")
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), _SlowSpeechHandler)
    server.daemon_threads = True
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    cancellation = threading.Event()
    failures: list[BaseException] = []

    def run_request():
        try:
            voice_module.synthesize(
                hivemind_url=f"http://127.0.0.1:{server.server_port}",
                text="cancel this request",
                timeout=5,
                cancel_event=cancellation,
            )
        except BaseException as exc:
            failures.append(exc)

    request_thread = threading.Thread(target=run_request, name="rest-transport-cancel-test")
    request_thread.start()
    try:
        assert request_started.wait(timeout=1.0)
        cancelled_at = time.monotonic()
        cancellation.set()
        request_thread.join(timeout=0.75)
        assert not request_thread.is_alive()
        assert time.monotonic() - cancelled_at < 0.75
        assert failures and isinstance(failures[0], VoiceUnavailable)
        assert "cancelled" in str(failures[0]).lower()
    finally:
        release_handler.set()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=1.0)


# ---------------------------------------------------------------------------
# 2026-07-05 first-audio deadline: default raised 6s -> 10s and enforced as an
# ABSOLUTE monotonic deadline (queue polling cannot extend or shorten it). A
# cold GIM whose first audio lands before 10s must stay pure WS (no REST
# fallback); a genuine no-audio stall must still fall back exactly once.
# ---------------------------------------------------------------------------


def test_ws_first_audio_timeout_default_is_ten_seconds(monkeypatch):
    import machine_spirit_4.gateway.voice as v

    monkeypatch.delenv("MS4_TTS_WS_FIRST_AUDIO_TIMEOUT", raising=False)
    assert v._voice_ws_first_audio_timeout() == 10.0
    monkeypatch.setenv("MS4_TTS_WS_FIRST_AUDIO_TIMEOUT", "3.5")
    assert v._voice_ws_first_audio_timeout() == 3.5


def test_ws_first_audio_before_ten_second_default_stays_pure_ws(monkeypatch):
    """First WS audio arriving at simulated +7s — past the OLD 6s default,
    before the NEW 10s default — must stay pure WS with NO REST fallback.

    Simulated time is injected via a monotonic offset so the wait loop crosses
    the 6s/10s boundaries deterministically without real multi-second waits;
    real time still flows for background threads."""
    import machine_spirit_4.gateway.voice as v

    _ms3_voice_ready(monkeypatch)
    monkeypatch.delenv("MS4_TTS_WS_FIRST_AUDIO_TIMEOUT", raising=False)  # DEFAULT applies
    monkeypatch.setenv("MS4_VOICE_SMART_REFLEX", "0")
    monkeypatch.setenv("MS4_VOICE_EGG_HONORABLE", "0")
    monkeypatch.setattr(v, "_identify_speaker_async", lambda *_a, **_k: None)
    monkeypatch.setattr(v, "synthesize", lambda **_: pytest.fail(
        "REST fallback must not run: WS audio arrives at +7s, before the 10s default deadline"
    ))

    # Capture the ORIGINAL monotonic before patching (v.time is the stdlib time
    # module, so the patch would otherwise recurse into itself). Real time still
    # flows; ``offset`` injects the simulated per-poll jumps.
    _orig_monotonic = v.time.monotonic
    offset = {"v": 0.0}
    monkeypatch.setattr(v.time, "monotonic", lambda: _orig_monotonic() + offset["v"])

    class _AudioAtSevenSecondsWsEngine:
        def __init__(self, *, emit, t_start, **_):
            self.emit = emit
            self._emitted = 0
            self._wait_base = None

        def open(self):
            pass

        def push(self, _t):
            pass

        def flush(self):
            pass

        def wait_for_final(self, timeout=60):
            # Jump simulated time by the poll budget rather than sleeping.
            if self._wait_base is None:
                self._wait_base = offset["v"]
            offset["v"] += timeout
            elapsed = offset["v"] - self._wait_base
            if self._emitted == 0 and elapsed >= 7.0:
                self.emit("audio_chunk", {
                    "index": 0, "text": "", "audio_base64": "AAA", "audio_mime": "audio/wav",
                    "tts_ms": 0, "held_for_inorder_ms": 0, "engine": "ws_super",
                })
                self._emitted += 1
                return False
            return self._emitted > 0

        def close(self):
            pass

        def cancel_pending_send(self):
            self.close()

        def metrics(self):
            return {
                "engine": "ws_super", "chunks_emitted": self._emitted,
                "first_audio_ms": 0 if self._emitted else None,
                "bytes_received": 200 * self._emitted, "is_final_seen": self._emitted > 0,
                "error": None,
            }

    events: list[tuple[str, dict]] = []
    result = voice_ptt_turn_stream(
        runner=_FakeRunner(), audio=b"WAV",
        chat_fn=lambda m, stream_callback=None, **_: (
            stream_callback("A reply with several complete words here."),
            {"text": "A reply with several complete words here.", "session_id": "s"},
        )[1],
        transcribe_fn=lambda **_: {"text": "hi", "model": "whisper-1"},
        synthesize_fn=lambda **_: {},
        emit=lambda n, p: events.append((n, dict(p))) or True,
        engine="ws_super", ws_engine_factory=_AudioAtSevenSecondsWsEngine,
    )

    assert result["metrics"]["engine"] == "ws_super"
    assert not result["metrics"].get("rest_fallback_used"), "must stay pure WS, no REST fallback"
    assert any(n == "audio_chunk" for n, _ in events)


def test_ws_no_audio_stall_falls_back_to_rest_exactly_once(monkeypatch):
    import machine_spirit_4.gateway.voice as v

    _ms3_voice_ready(monkeypatch)
    monkeypatch.setenv("MS4_TTS_WS_FIRST_AUDIO_TIMEOUT", "0.3")  # tight deadline for speed
    monkeypatch.setenv("MS4_VOICE_SMART_REFLEX", "0")
    monkeypatch.setenv("MS4_VOICE_EGG_HONORABLE", "0")
    monkeypatch.setattr(v, "_identify_speaker_async", lambda *_a, **_k: None)
    synth_calls = {"n": 0}

    def fake_synth(**_):
        synth_calls["n"] += 1
        return {"audio_bytes": b"WAV", "audio_base64": "V0FW", "content_type": "audio/wav"}

    monkeypatch.setattr(v, "synthesize", fake_synth)

    class _StallWsEngine:
        def __init__(self, **_):
            pass

        def open(self):
            pass

        def push(self, _t):
            pass

        def flush(self):
            pass

        def wait_for_final(self, timeout=60):
            time.sleep(min(timeout, 0.05))  # advance real time without busy-spinning
            return False  # never final, never any audio

        def close(self):
            pass

        def cancel_pending_send(self):
            self.close()

        def metrics(self):
            return {
                "engine": "ws_super", "chunks_emitted": 0, "bytes_received": 0,
                "is_final_seen": False, "error": None,
            }

    events: list[tuple[str, dict]] = []
    result = voice_ptt_turn_stream(
        runner=_FakeRunner(), audio=b"WAV",
        chat_fn=lambda m, stream_callback=None, **_: (
            stream_callback("The stalled turn reply has several complete words here."),
            {"text": "The stalled turn reply has several complete words here.", "session_id": "s"},
        )[1],
        transcribe_fn=lambda **_: {"text": "hi", "model": "whisper-1"},
        synthesize_fn=lambda **_: {},
        emit=lambda n, p: events.append((n, dict(p))) or True,
        engine="ws_super", ws_engine_factory=_StallWsEngine,
    )

    fallbacks = [
        p for n, p in events
        if n == "status" and p.get("phase") == "tts_engine_fallback_after_zero_audio"
    ]
    audio = [p for n, p in events if n == "audio_chunk"]
    assert len(fallbacks) == 1, "a real no-audio stall must trigger the REST fallback exactly once"
    assert result["metrics"]["rest_fallback_used"] is True
    assert len(audio) >= 1 and synth_calls["n"] >= 1, "fallback audio must come from REST"


# ---------------------------------------------------------------------------
# 2026-07-05 (correction): NORMAL WS timeouts and a zero-audio reader error
# must tear down through engine.close()'s bounded graceful-first path — a
# graceful RFC6455 ws.close BEFORE any socket shutdown, with the raw shutdown
# only as the fallback inside close(). The raw _cancel_ws_transport pre-abort
# (cancel_pending_send) is reserved for barge / client-disconnect / blocked-send
# / dead-client emergencies. These tests exercise the ACTUAL voice.py
# orchestration and record the teardown order.
# ---------------------------------------------------------------------------


class _OrchestrationWsEngine:
    """Records teardown ops so a test can prove voice.py routes normal timeouts
    through the graceful close() (ws.close, then raw shutdown fallback) and NOT
    through the raw pre-abort (cancel_pending_send)."""

    def __init__(self, *, ops, emit=None, emit_one_chunk=False, **_):
        self._ops = ops
        self._emit = emit
        self._emit_one_chunk = emit_one_chunk
        self._emitted = 0

    def open(self):
        pass

    def push(self, _t):
        pass

    def flush(self):
        pass

    def wait_for_final(self, timeout=60):
        time.sleep(min(timeout, 0.02))  # let real time advance toward the deadline
        if self._emit_one_chunk and self._emitted == 0 and self._emit is not None:
            self._emit("audio_chunk", {
                "index": 0, "text": "", "audio_base64": "AAA", "audio_mime": "audio/wav",
                "tts_ms": 0, "held_for_inorder_ms": 0, "engine": "ws_super",
            })
            self._emitted += 1
        return False  # never final -> the turn reaches its deadline

    def cancel_pending_send(self):
        # The raw socket-shutdown emergency abort (voice.py _cancel_ws_transport).
        self._ops.append("raw_preabort")

    def close(self):
        # Mirrors _close_transport: bounded graceful ws.close FIRST, then the raw
        # socket shutdown fallback.
        self._ops.append("graceful_close")
        self._ops.append("raw_fallback")

    def metrics(self):
        return {
            "engine": "ws_super", "chunks_emitted": self._emitted,
            "bytes_received": 200 * self._emitted, "is_final_seen": False, "error": None,
        }


def test_ws_first_audio_timeout_closes_gracefully_before_any_socket_shutdown(monkeypatch):
    import machine_spirit_4.gateway.voice as v

    _ms3_voice_ready(monkeypatch)
    monkeypatch.setenv("MS4_TTS_WS_FIRST_AUDIO_TIMEOUT", "0")
    monkeypatch.setenv("MS4_VOICE_SMART_REFLEX", "0")
    monkeypatch.setenv("MS4_VOICE_EGG_HONORABLE", "0")
    monkeypatch.setattr(v, "_identify_speaker_async", lambda *_a, **_k: None)
    monkeypatch.setattr(v, "synthesize", lambda **_: {
        "audio_bytes": b"WAV", "audio_base64": "V0FW", "content_type": "audio/wav"})
    ops: list[str] = []

    t0 = time.monotonic()
    result = voice_ptt_turn_stream(
        runner=_FakeRunner(), audio=b"WAV",
        chat_fn=lambda m, stream_callback=None, **_: (
            stream_callback("The cold GIM reply has several complete words here."),
            {"text": "The cold GIM reply has several complete words here.", "session_id": "s"},
        )[1],
        transcribe_fn=lambda **_: {"text": "hi", "model": "whisper-1"},
        synthesize_fn=lambda **_: {},
        emit=lambda _n, _p: True,
        engine="ws_super",
        ws_engine_factory=lambda **kw: _OrchestrationWsEngine(ops=ops, **kw),
    )
    elapsed = time.monotonic() - t0

    assert ops, "the WS engine must be torn down"
    assert ops[0] == "graceful_close", (
        "first-audio timeout must attempt the bounded graceful ws.close before any "
        f"socket shutdown; got teardown order {ops}"
    )
    assert "raw_preabort" not in ops, (
        "first-audio timeout is a NORMAL timeout: the raw _cancel_ws_transport "
        f"pre-abort must not run (raw is only the fallback inside close). ops={ops}"
    )
    assert elapsed < 5.0, "graceful teardown must stay bounded"
    assert result["metrics"]["rest_fallback_used"] is True


def test_ws_final_audio_timeout_closes_gracefully_before_any_socket_shutdown(monkeypatch):
    import machine_spirit_4.gateway.voice as v

    _ms3_voice_ready(monkeypatch)
    monkeypatch.setenv("MS4_TTS_WS_FIRST_AUDIO_TIMEOUT", "5")   # audio arrives well before this
    monkeypatch.setenv("MS4_TTS_WS_FINAL_TIMEOUT", "0.3")       # but isFinal never comes
    monkeypatch.setenv("MS4_VOICE_SMART_REFLEX", "0")
    monkeypatch.setenv("MS4_VOICE_EGG_HONORABLE", "0")
    monkeypatch.setattr(v, "_identify_speaker_async", lambda *_a, **_k: None)
    ops: list[str] = []

    t0 = time.monotonic()
    result = voice_ptt_turn_stream(
        runner=_FakeRunner(), audio=b"WAV",
        chat_fn=lambda m, stream_callback=None, **_: (
            stream_callback("A reply whose audio starts then the tail never finalizes here."),
            {"text": "A reply whose audio starts then the tail never finalizes here.", "session_id": "s"},
        )[1],
        transcribe_fn=lambda **_: {"text": "hi", "model": "whisper-1"},
        synthesize_fn=lambda **_: {},
        emit=lambda _n, _p: True,
        engine="ws_super",
        ws_engine_factory=lambda **kw: _OrchestrationWsEngine(ops=ops, emit_one_chunk=True, **kw),
    )
    elapsed = time.monotonic() - t0

    assert ops and ops[0] == "graceful_close", (
        "final-audio timeout must attempt the bounded graceful ws.close before any "
        f"socket shutdown; got teardown order {ops}"
    )
    assert "raw_preabort" not in ops, (
        f"final-audio timeout is a NORMAL timeout: no raw pre-abort. ops={ops}"
    )
    assert elapsed < 5.0
    # One WS chunk was delivered; a final-audio timeout does not REST-replay.
    assert result["metrics"]["audio_chunks"] == 1
    assert result["metrics"].get("rest_fallback_used") in (False, None)


def test_ws_turn_cancelled_still_raw_preaborts_before_close(monkeypatch):
    """Guard: an emergency (barge / client disconnect) MUST still raw-abort the
    transport before close so a possibly-stuck send is unblocked."""
    import machine_spirit_4.gateway.voice as v

    _ms3_voice_ready(monkeypatch)
    monkeypatch.setenv("MS4_VOICE_SMART_REFLEX", "0")
    monkeypatch.setenv("MS4_VOICE_EGG_HONORABLE", "0")
    monkeypatch.setattr(v, "_identify_speaker_async", lambda *_a, **_k: None)
    ops: list[str] = []
    client_alive = threading.Event()
    client_alive.set()

    def fake_chat(message, *, stream_callback=None, **_):
        assert stream_callback("one two three four five ") is not False
        client_alive.clear()  # barge / disconnect mid-turn
        return {"text": "one two three four five", "session_id": "s"}

    with pytest.raises(VoiceUnavailable, match="cancelled"):
        voice_ptt_turn_stream(
            runner=_FakeRunner(), audio=b"WAV",
            chat_fn=fake_chat,
            transcribe_fn=lambda **_: {"text": "hi", "model": "whisper-1"},
            synthesize_fn=lambda **_: pytest.fail("a cancelled turn must not REST-replay"),
            emit=lambda _n, _p: True,
            engine="ws_super",
            ws_engine_factory=lambda **kw: _OrchestrationWsEngine(ops=ops, **kw),
            client_alive=client_alive,
        )

    assert "raw_preabort" in ops, f"emergency cancel must raw-abort before close. ops={ops}"
    assert ops.index("raw_preabort") < ops.index("graceful_close")


# ---------------------------------------------------------------------------
# FaceLobe marker-confirmation contract (2026-07-06)
#
# Physical Razer/Odyssey push-to-talk turn (evidence:
#   _scratch/.../oracle_tts_warm_ab_20260706T0752Z/physical/
#   oracle_live_physical_turn_result.json):
#
#   ASR request  : "Hivemind, please answer in one short sentence, and confirm
#                   you heard the Cobalt 216 marker through the physical
#                   microphone."
#   llama3.1:8b  : "I'm here, and I've received your voice input, but I don't
#                   have any specific context or request to respond to yet."
#
# The request was an EXPLICIT marker-confirmation obligation, but the ordinary
# FaceLobe generation returned a generic no-context reply and voice.py streamed
# it straight into text_delta / chunk_scheduled / TTS with NO semantic
# postcondition. These tests pin the voice-level contract: an explicit
# marker-confirmation turn must never stream/schedule/synthesize a reply that
# fails to confirm the marker; the spoken output must satisfy the obligation.
# ---------------------------------------------------------------------------

_COBALT_MARKER_REQUEST = (
    "Hivemind, please answer in one short sentence, and confirm you heard "
    "the Cobalt 216 marker through the physical microphone."
)
_GENERIC_INVALID_REPLY = (
    "I'm here, and I've received your voice input, but I don't have any "
    "specific context or request to respond to yet."
)
_VALID_MARKER_REPLY = "Yes, I heard the Cobalt 216 marker clearly."


def _marker_norm(text: str) -> str:
    """Local, test-owned normalizer mirroring voice._AUDIO_NORM_RE so the RED
    proof runs against UNMODIFIED product code (no import of the not-yet-added
    production helpers)."""
    import re as _re

    return " ".join(_re.sub(r"[^a-z0-9' ]+", " ", (text or "").lower()).split())


def _run_marker_confirmation_turn(
    monkeypatch,
    *,
    transcript,
    reply_text,
    engine="rest",
    ws_engine_factory=None,
):
    """Drive one voice turn through the real orchestrator with fake ASR /
    chat / synthesis seams (the file's established convention).

    * fake ASR returns ``transcript`` verbatim,
    * fake chat streams ``reply_text`` word-by-word AND returns it as final
      text (mirrors the observed FaceLobe stream), and
    * a fake synthesizer records every submitted sentence.

    Returns ``(events, synthesized, result)``.
    """
    import machine_spirit_4.gateway.voice as voice_module

    _ms3_voice_ready(monkeypatch)
    monkeypatch.setenv("MS4_VOICE_SMART_REFLEX", "0")
    monkeypatch.setenv("MS4_VOICE_EGG_HONORABLE", "0")
    monkeypatch.setattr(voice_module, "_identify_speaker_async", lambda *_a, **_k: None)

    def fake_transcribe(*, hivemind_url, audio, filename, model):
        return {"text": transcript, "model": "whisper-1"}

    def fake_chat(message, *, session_id=None, model=None, stream_callback=None, **_):
        for word in reply_text.split(" "):
            if stream_callback is not None:
                assert stream_callback(word + " ") is not False
        return {"text": reply_text, "session_id": "s"}

    synthesized: list[str] = []
    synth_lock = threading.Lock()

    def fake_synthesize(*, hivemind_url, text, model, voice, response_format, **_):
        with synth_lock:
            synthesized.append(text)
        return {
            "audio_bytes": b"\x00" * 8,
            "content_type": "audio/wav",
            "audio_base64": "AAA",
            "model": "tts-1",
            "voice": "alloy",
            "format": "wav",
        }

    events: list[tuple[str, dict[str, Any]]] = []
    ev_lock = threading.Lock()

    def emit(event, payload):
        with ev_lock:
            events.append((event, dict(payload)))
        return True

    kwargs: dict[str, Any] = dict(
        runner=_FakeRunner(),
        audio=b"WAV",
        chat_fn=fake_chat,
        transcribe_fn=fake_transcribe,
        synthesize_fn=fake_synthesize,
        emit=emit,
        engine=engine,
    )
    if ws_engine_factory is not None:
        kwargs["ws_engine_factory"] = ws_engine_factory
    result = voice_ptt_turn_stream(**kwargs)
    return events, synthesized, result


def _stream_texts(events, kind):
    return [p.get("text", "") for e, p in events if e == kind]


def test_marker_confirmation_contract_rest_blocks_generic_reply(monkeypatch):
    """RED on PREIMAGE / GREEN after the fix.

    An explicit 'confirm you heard the Cobalt 216 marker' turn whose model
    returns a generic no-context reply must NOT leak that reply to
    text_delta / chunk_scheduled / synthesis, and the spoken output must
    confirm the normalized 'cobalt 216' marker.

    Self-contained: it drives the REAL orchestrator end-to-end and asserts only
    on emitted/synthesized text plus the returned reply — NO dependency on any
    post-fix metric key — so it replays cleanly RED on the preimage voice.py
    (fails at the pre-TTS leak assertion, never at a KeyError or a collection
    error)."""
    events, synthesized, result = _run_marker_confirmation_turn(
        monkeypatch, transcript=_COBALT_MARKER_REQUEST, reply_text=_GENERIC_INVALID_REPLY,
    )

    invalid_norm = _marker_norm(_GENERIC_INVALID_REPLY)
    text_delta_norm = _marker_norm("".join(_stream_texts(events, "text_delta")))
    scheduled_texts = _stream_texts(events, "chunk_scheduled")
    scheduled_norm = _marker_norm(" ".join(scheduled_texts))
    synth_norm = _marker_norm(" ".join(synthesized))
    audio_texts = _stream_texts(events, "audio_chunk")
    audio_norm = _marker_norm(" ".join(audio_texts))

    # (0) Recorder-non-empty guard: prove the turn was actually driven
    #     end-to-end so the leak assertions below cannot pass VACUOUSLY on an
    #     empty capture. On BOTH preimage and fixed code the pipeline runs and
    #     synthesizes >=1 sentence, so this guard passes either way; it fails
    #     ONLY if the fixture failed to drive synthesis. The genuine RED must
    #     come from the leak assertion (1), not from this guard.
    assert synthesized, "vacuous: nothing reached synthesize_fn (turn not driven)"
    assert scheduled_texts, "vacuous: no chunk_scheduled event captured"
    assert audio_texts, "vacuous: no audio_chunk event captured"
    assert text_delta_norm, "vacuous: no text_delta captured"

    # (1) LEAK ASSERTION — fails RED on preimage: the invalid model reply must
    #     never reach the user-visible text stream, the TTS scheduler, or the
    #     synthesizer. On unchanged voice.py the generic reply is streamed /
    #     scheduled / synthesized verbatim, so `invalid_norm` is present here.
    assert invalid_norm not in text_delta_norm, "invalid reply leaked to text_delta"
    assert invalid_norm not in scheduled_norm, "invalid reply reached chunk_scheduled"
    assert invalid_norm not in synth_norm, "invalid reply reached REST synthesis"
    assert invalid_norm not in audio_norm, "invalid reply reached audio_chunk"

    # (2) MARKER ASSERTION: the synthesized/scheduled spoken text and the
    #     returned reply must confirm the normalized marker.
    assert "cobalt 216" in synth_norm, "synthesized audio did not confirm the Cobalt 216 marker"
    assert "cobalt 216" in scheduled_norm, "no scheduled chunk confirmed the Cobalt 216 marker"
    assert "cobalt 216" in _marker_norm(result["reply_text"]), (
        "returned reply_text must confirm the Cobalt 216 marker"
    )
    assert invalid_norm not in _marker_norm(result["reply_text"])


def test_marker_helpers_extract_obligation_generally():
    """The marker parser is general (not Cobalt-specific) and conservative:
    ordinary turns produce NO obligation."""
    import machine_spirit_4.gateway.voice as v

    # The exact physical request extracts the marker phrase, original casing.
    assert v._extract_marker_obligation(_COBALT_MARKER_REQUEST) == "Cobalt 216"
    # General over marker names and equivalent phrasings.
    assert (
        v._extract_marker_obligation("Please confirm you received the Alpha Seven marker.")
        == "Alpha Seven"
    )
    assert (
        v._extract_marker_obligation("Can you confirm that you detected the codeword Delta marker?")
        == "codeword Delta"
    )
    # Ordinary voice turns carry NO obligation -> None (never buffered/rewritten).
    assert v._extract_marker_obligation("What's the date today?") is None
    assert v._extract_marker_obligation("Please summarize the meeting notes.") is None
    assert v._extract_marker_obligation("") is None
    # "marker" mentioned without a confirmation obligation must not trip it.
    assert v._extract_marker_obligation("Tell me about the marker pen on my desk.") is None


def test_marker_extractor_rejects_negated_and_unrelated_confirm():
    """ADVERSARIAL (2026-07-06): extraction requires a POSITIVE
    confirm->heard->marker relation. These near-misses MUST return None, and
    they must do so GENERALLY (structural relation + negation cue), not via a
    Cobalt-specific blocklist."""
    import machine_spirit_4.gateway.voice as v

    # 1. Negated imperative — the request is NOT to confirm.
    assert v._extract_marker_obligation(
        "Do not confirm you heard the Cobalt 216 marker."
    ) is None
    # 2. Negated hear/receive clause — confirming a NON-hearing.
    assert v._extract_marker_obligation(
        "Please confirm you did not hear the Cobalt 216 marker."
    ) is None
    # 3. "confirm" governs a DIFFERENT object; the marker appears only in a
    #    separate declarative clause.
    assert v._extract_marker_obligation(
        "Confirm the meeting, and I heard the Cobalt 216 marker."
    ) is None
    # Additional negation variants, general across marker names.
    assert v._extract_marker_obligation(
        "Never confirm you heard the Alpha Seven marker."
    ) is None
    assert v._extract_marker_obligation(
        "Don't confirm you heard the codeword Delta marker."
    ) is None
    assert v._extract_marker_obligation(
        "Please confirm you didn't receive the codeword Delta marker."
    ) is None

    # PRESERVED positives (hardening must NOT weaken these).
    assert v._extract_marker_obligation(_COBALT_MARKER_REQUEST) == "Cobalt 216"
    assert (
        v._extract_marker_obligation("Please confirm you received the Alpha Seven marker.")
        == "Alpha Seven"
    )
    assert (
        v._extract_marker_obligation("Can you confirm that you detected the codeword Delta marker?")
        == "codeword Delta"
    )


def test_marker_extractor_rejects_distal_negation_and_prohibition():
    """ADVERSARIAL family 2 (2026-07-06): a negation/prohibition that scopes the
    confirm imperative from a DISTANCE must reject — even when the cue is many
    tokens before "confirm" ("do not ever under any circumstances confirm ..."),
    is a distal "no" ("under no circumstances should you confirm ..."), or is a
    prohibition VERB rather than not/n't ("I forbid you to confirm ..."). The
    detection is CLAUSE-SCOPED (whole pre-confirm span of the clause holding the
    relation), not a fixed last-N-tokens window."""
    import machine_spirit_4.gateway.voice as v

    assert v._extract_marker_obligation(
        "Do not ever under any circumstances confirm you heard the Cobalt 216 marker."
    ) is None
    assert v._extract_marker_obligation(
        "Under no circumstances should you confirm you heard the Cobalt 216 marker."
    ) is None
    assert v._extract_marker_obligation(
        "I forbid you to confirm you heard the Cobalt 216 marker."
    ) is None

    # MUST-PRESERVE control: the "not" belongs to the PRE-semicolon clause; the
    # post-semicolon clause holds the positive confirm relation and must still
    # extract (clause isolation on ';' keeps the prior-clause negation out of
    # scope).
    assert v._extract_marker_obligation(
        "I am not asking you to chat; please confirm you heard the Cobalt 216 marker."
    ) == "Cobalt 216"


def test_marker_reply_satisfaction_and_confirmation_text():
    """Normalized, word-boundary-bounded satisfaction; the deterministic
    confirmation satisfies its own contract."""
    import machine_spirit_4.gateway.voice as v

    assert v._marker_reply_satisfies("Yes, I heard the Cobalt 216 marker.", "Cobalt 216")
    assert v._marker_reply_satisfies("cobalt   216!!!", "Cobalt 216")  # case/punct-insensitive
    # A superstring token must NOT satisfy (bounded containment).
    assert not v._marker_reply_satisfies("I heard the cobalt 2160 marker.", "Cobalt 216")
    assert not v._marker_reply_satisfies(_GENERIC_INVALID_REPLY, "Cobalt 216")
    # The deterministic correction is grounded in the request and self-consistent.
    text = v._marker_confirmation_text("Cobalt 216")
    assert text == "I heard the Cobalt 216 marker."
    assert v._marker_reply_satisfies(text, "Cobalt 216")


def test_marker_confirmation_contract_preserves_a_valid_reply_rest(monkeypatch):
    """An already-valid marker reply is released UNCHANGED (no correction)."""
    events, synthesized, result = _run_marker_confirmation_turn(
        monkeypatch, transcript=_COBALT_MARKER_REQUEST, reply_text=_VALID_MARKER_REPLY,
    )
    text_delta_norm = _marker_norm("".join(_stream_texts(events, "text_delta")))
    assert text_delta_norm == _marker_norm(_VALID_MARKER_REPLY)
    assert result["reply_text"].strip() == _VALID_MARKER_REPLY
    assert "cobalt 216" in _marker_norm(" ".join(synthesized))
    mc = result["metrics"]["marker_contract"]
    assert mc["applied"] is True and mc["correction_required"] is False


class _RecordingWsEngine:
    """A ws_super engine that records every pushed fragment and emits an
    ``audio_chunk`` per push (so chunks_emitted > 0 and the zero-audio REST
    fallback never runs). Lets a test prove what text the WS path synthesized."""

    def __init__(self, *, pushed, emit, t_start, voice, **_):
        self._pushed = pushed
        self.emit = emit
        self.t_start = t_start
        self.voice = voice
        self._n = 0

    def open(self):
        self.emit("tts_engine", {"engine": "ws_super", "voice": self.voice, "connect_ms": 0})

    def push(self, text):
        self._pushed.append(text)
        self.emit("audio_chunk", {
            "index": self._n, "text": text, "audio_base64": "AAA",
            "audio_mime": "audio/wav", "tts_ms": 0, "held_for_inorder_ms": 0,
            "engine": "ws_super",
        })
        self._n += 1

    def flush(self):
        pass

    def wait_for_final(self, timeout=60):
        return True

    def close(self):
        pass

    def cancel_pending_send(self):
        pass

    def metrics(self):
        return {
            "engine": "ws_super", "voice": self.voice, "connect_ms": 0,
            "first_audio_ms": 0 if self._n else None, "chunks_emitted": self._n,
            "bytes_received": 3 * self._n, "is_final_seen": True, "error": None,
        }


def test_marker_confirmation_contract_ws_blocks_generic_reply(monkeypatch):
    """The ws_super path enforces the SAME boundary: a generic reply is never
    pushed to TTS_SUPER, the REST fallback is not used, and the WS-synthesized
    text confirms the normalized marker."""
    pushed: list[str] = []
    events, synthesized, result = _run_marker_confirmation_turn(
        monkeypatch,
        transcript=_COBALT_MARKER_REQUEST,
        reply_text=_GENERIC_INVALID_REPLY,
        engine="ws_super",
        ws_engine_factory=lambda **kw: _RecordingWsEngine(pushed=pushed, **kw),
    )

    invalid_norm = _marker_norm(_GENERIC_INVALID_REPLY)
    pushed_norm = _marker_norm(" ".join(pushed))
    scheduled_norm = _marker_norm(" ".join(_stream_texts(events, "chunk_scheduled")))
    text_delta_norm = _marker_norm("".join(_stream_texts(events, "text_delta")))

    # WS TTS_SUPER, the scheduler, and the visible text never see the invalid
    # reply; the REST fallback (which synthesizes reply_text) is not used.
    assert invalid_norm not in pushed_norm, "invalid reply pushed to TTS_SUPER"
    assert invalid_norm not in scheduled_norm, "invalid reply reached chunk_scheduled"
    assert invalid_norm not in text_delta_norm, "invalid reply reached text_delta"
    assert result["metrics"].get("rest_fallback_used") in (False, None)
    # The WS-synthesized text and the returned reply confirm the marker.
    assert "cobalt 216" in pushed_norm, "WS did not synthesize the marker confirmation"
    assert "cobalt 216" in _marker_norm(result["reply_text"])
    mc = result["metrics"]["marker_contract"]
    assert mc["applied"] is True and mc["correction_required"] is True


def test_normal_non_marker_turn_still_streams_before_chat_completion(monkeypatch):
    """No regression: a normal (non-marker) request STILL streams text_delta
    synchronously DURING the chat call, before it returns — the contract must
    not buffer or delay ordinary turns."""
    import machine_spirit_4.gateway.voice as voice_module

    _ms3_voice_ready(monkeypatch)
    monkeypatch.setenv("MS4_VOICE_SMART_REFLEX", "0")
    monkeypatch.setenv("MS4_VOICE_EGG_HONORABLE", "0")
    monkeypatch.setattr(voice_module, "_identify_speaker_async", lambda *_a, **_k: None)

    delta_streamed = threading.Event()
    saw_delta_before_chat_return = threading.Event()

    def fake_transcribe(**_):
        return {"text": "What is the date today?", "model": "whisper-1"}

    def fake_chat(message, *, stream_callback=None, **_):
        assert stream_callback("Today is a fine day to ship code. ") is not False
        # If streaming is synchronous (not buffered), the text_delta emit has
        # already fired by the time the callback returns.
        if delta_streamed.wait(timeout=2.0):
            saw_delta_before_chat_return.set()
        return {"text": "Today is a fine day to ship code.", "session_id": "s"}

    def fake_synth(**_):
        return {
            "audio_bytes": b"\x00" * 8, "content_type": "audio/wav",
            "audio_base64": "AAA", "model": "tts-1", "voice": "alloy", "format": "wav",
        }

    events: list[tuple[str, dict[str, Any]]] = []
    lock = threading.Lock()

    def emit(event, payload):
        with lock:
            events.append((event, dict(payload)))
        if event == "text_delta":
            delta_streamed.set()
        return True

    result = voice_ptt_turn_stream(
        runner=_FakeRunner(), audio=b"WAV", chat_fn=fake_chat,
        transcribe_fn=fake_transcribe, synthesize_fn=fake_synth,
        emit=emit, engine="rest",
    )

    assert saw_delta_before_chat_return.is_set(), (
        "a normal turn must stream text_delta during chat (no contract buffering)"
    )
    assert any(e == "text_delta" for e, _ in events)
    mc = result["metrics"]["marker_contract"]
    assert mc["applied"] is False and mc["correction_required"] is False


# ---------------------------------------------------------------------------
# 2026-07-07 R4 Oracle latency observability (ADDITIVE, fail-first / RED)
#
# Lane: ms4_r4_observability_red (test-only). Fail-first contracts for ADDITIVE
# observability on the accepted no-gap REST path. They assert that selected
# NON-SECRET HiveMind Lobe Interface (HLI) provenance survives onto every REST
# reply chunk metric, and that a SERVER-minted turn/chunk id is present so a
# chunk metric can be correlated with the browser turn telemetry.
#
# GROUNDED against frozen production (READ-ONLY, verified this session):
#   * voice.py::synthesize (L1466-1527) reads the full HTTP response header dict
#     (L1416 / L1510) but RETURNS ONLY content_type -- every provenance header is
#     discarded.
#   * voice_ptt_turn_stream::_try_emit_ready (L2744-2784) has the synthesize
#     result (tts_result) in hand but builds the per-chunk `timing` metric
#     (L2751-2759 -> result["metrics"]["chunks"]) and the audio_chunk emit payload
#     (L2777) from a FIXED key set, dropping any provenance.
#
# The exact HiveMind (HLI) wire header NAMES are UNVERIFIED (no live cluster is
# reachable from this lane and MS4 defines no header constant for them). The
# contract locked here is surface-level -- "non-secret provenance received MUST be
# retained onto the chunk metric" -- and holds regardless of exact spelling,
# because production currently retains NOTHING. Wire spellings must be confirmed
# by the production-owning lane before it is implemented.
# ---------------------------------------------------------------------------


def _hli_chunk_provenance(idx: int) -> dict[str, Any]:
    """A representative NON-SECRET HLI provenance block as HiveMind would return
    alongside a /v1/audio/speech reply: server-minted ids + routing + timing.
    Contains no secrets (no auth token, no raw prompt text)."""
    return {
        "request_id": f"hli-req-{idx:02d}",
        "served_by": "hivemind-lobe-07",
        "location": "rack-b/gpu-3",
        "endpoint": "/v1/audio/speech",
        "tts_path": "tts_super/xtts-v2",
        "queue_path": "gim://queue/tts/high",
        "timing_ms": 120 + idx,               # aggregate server-reported synth timing
        "turn_id": "srv-turn-abc123",         # server-minted, stable across the turn
        "chunk_id": f"srv-chunk-{idx:02d}",   # server-minted, per reply chunk
    }


def _provenance_chat_and_synth():
    """Multi-sentence chat + a synthesize_fn that returns audio PLUS an HLI
    provenance block, so the REST path produces >=1 reply chunk carrying
    provenance that production would have to retain."""
    def fake_transcribe(**_):
        return {"text": "Give me three facts.", "model": "whisper-1"}

    def fake_chat(message, *, session_id=None, model=None, stream_callback=None, **_):
        sentences = (
            "The first sentence here is deliberately written to be quite long indeed. ",
            "The second sentence here is also written to be reasonably long as well. ",
        )
        for s in sentences:
            for w in s.split():
                stream_callback(w + " ")
        return {"text": "".join(sentences).strip(), "session_id": "s"}

    call = {"n": 0}

    def fake_synthesize(**kw):
        idx = call["n"]
        call["n"] += 1
        return {
            "audio_bytes": b"\x00" * 8,
            "content_type": "audio/wav",
            "audio_base64": f"AUDIO-{idx}",
            "model": "tts-1",
            "voice": "alloy",
            "format": "wav",
            # HiveMind returns provenance with every reply; a faithful
            # synthesize_fn surfaces it so the turn can retain it.
            "provenance": _hli_chunk_provenance(idx),
        }

    return fake_transcribe, fake_chat, fake_synthesize


def _chunk_server_id(chunk: dict[str, Any], key: str):
    """Read a server-minted id off a chunk metric, accepting either a first-class
    field or one nested under `provenance` (either placement satisfies the
    correlation contract)."""
    if chunk.get(key):
        return chunk[key]
    prov = chunk.get("provenance")
    if isinstance(prov, dict):
        return prov.get(key)
    return None


# ---------------------------------------------------------------------------
# R2 P1 findings 2 & 3: the LIVE cancellable REST TTS path + the non-secret
# allowlist. The synthesize() response-header fixture returns BOTH approved
# non-secret HLI provenance headers AND explicit sensitive headers; the contract
# is that approved provenance survives while every sensitive name/value is absent
# from the returned provenance and the serialized result surface. A copy-all
# implementation fails. The cancellable path is exercised by passing an UNSET
# cancel_event (which selects _cancellable_speech_request exactly as live
# streaming does, voice.py L1498-1505); the non-cancellable urlopen path is the
# second path.
# ---------------------------------------------------------------------------

# Approved NON-SECRET HLI provenance response headers (representative names).
_APPROVED_PROVENANCE_HEADERS = {
    "X-Request-Id": "hli-req-42",
    "X-HiveMind-Served-By": "hivemind-lobe-07",
    "X-HiveMind-Location": "rack-b/gpu-3",
    "X-HiveMind-Endpoint": "/v1/audio/speech",
    "X-TTS-Path": "tts_super/xtts-v2",
    "X-Queue-Path": "gim://queue/tts/high",
    "X-HiveMind-Timing-Ms": "137",
}
# Explicit SENSITIVE response headers a correct allowlist MUST drop. A copy-all
# implementation would leak these names/values.
_SENSITIVE_HEADERS = {
    "Authorization": "Bearer super-secret-token-DO-NOT-LEAK",
    "Set-Cookie": "ms4sid=deadbeefsecretcookie; HttpOnly; Path=/",
    "Cookie": "session=another-secret-cookie-value",
    "X-API-Key": "sk-live-000SECRETAPIKEY111",
    "X-Auth-Token": "auth-tok-SECRET-999",
    "X-Credential": "cred-SECRET-abc",
    "Proxy-Authorization": "Basic c2VjcmV0OnBhc3N3b3Jk",
}
_SENSITIVE_HEADER_NAMES_LOWER = {name.lower() for name in _SENSITIVE_HEADERS}
_SENSITIVE_VALUES = tuple(_SENSITIVE_HEADERS.values())


class _ProvenanceAndSecretsSpeechHandler(BaseHTTPRequestHandler):
    """Local /v1/audio/speech stand-in returning approved non-secret HLI
    provenance headers PLUS explicit sensitive headers, so the allowlist contract
    is exercised identically on the cancellable and non-cancellable paths."""

    def log_message(self, *_a, **_k):
        return

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0") or "0")
        self.rfile.read(length)
        body = b"RIFFWAVE-PROOF"
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(body)))
        for name, value in _APPROVED_PROVENANCE_HEADERS.items():
            self.send_header(name, value)
        for name, value in _SENSITIVE_HEADERS.items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)


def _synthesize_against_provenance_secret_server(*, cancel_event):
    """Start the provenance+secrets handler, call synthesize (selecting the LIVE
    cancellable path iff ``cancel_event`` is not None), return the result dict."""
    import machine_spirit_4.gateway.voice as voice_module

    server = ThreadingHTTPServer(("127.0.0.1", 0), _ProvenanceAndSecretsSpeechHandler)
    server.daemon_threads = True
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    try:
        kwargs: dict[str, Any] = dict(
            hivemind_url=f"http://127.0.0.1:{server.server_port}",
            text="confirm provenance retention",
            timeout=5,
        )
        if cancel_event is not None:
            kwargs["cancel_event"] = cancel_event
        return voice_module.synthesize(**kwargs)
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=1.0)


def _assert_synthesize_provenance_allowlist(res: dict[str, Any]) -> None:
    """Shared assertions: approved non-secret provenance is retained AND every
    sensitive header name/value is absent from the returned provenance and the
    serialized result surface. Fails RED against production (no provenance at all);
    a copy-all-headers implementation also fails (secrets leak)."""
    prov = res.get("provenance")
    assert isinstance(prov, dict), (
        f"synthesize discarded HLI provenance response headers (keys: {sorted(res.keys())})"
    )
    # Approved non-secret provenance survives.
    assert prov.get("request_id") == "hli-req-42"
    assert any(prov.get(k) for k in ("served_by", "location", "endpoint")), \
        "synthesize provenance missing served_by/location/endpoint"
    assert prov.get("tts_path") == "tts_super/xtts-v2"
    assert prov.get("queue_path") == "gim://queue/tts/high"
    assert prov.get("timing_ms") is not None, "synthesize provenance missing timing_ms"
    # Allowlist: no sensitive header NAME survives as a provenance key.
    prov_keys_lower = {str(k).lower() for k in prov}
    leaked_names = prov_keys_lower & _SENSITIVE_HEADER_NAMES_LOWER
    assert not leaked_names, f"sensitive header name(s) leaked into provenance: {sorted(leaked_names)}"
    # Allowlist: no sensitive VALUE appears anywhere on the serialized result surface.
    serialized = json.dumps(res, default=str)
    for secret in _SENSITIVE_VALUES:
        assert secret not in serialized, \
            f"sensitive header value leaked into synthesize result surface: {secret!r}"


def test_rest_chunk_metric_retains_hli_provenance(monkeypatch):
    """RED (additive): every REST reply chunk metric in
    ``result["metrics"]["chunks"]`` must retain selected non-secret HLI
    provenance -- request_id, one of served_by/location/endpoint, tts_path,
    queue_path, and an aggregate timing_ms.

    Fails against current production: ``_try_emit_ready`` builds the per-chunk
    timing dict from a fixed key set and discards the synthesize result's
    provenance."""
    _ms3_voice_ready(monkeypatch)
    fake_transcribe, fake_chat, fake_synthesize = _provenance_chat_and_synth()

    result = voice_ptt_turn_stream(
        runner=_FakeRunner(), audio=b"FAKE_WAV",
        chat_fn=fake_chat, transcribe_fn=fake_transcribe,
        synthesize_fn=fake_synthesize, emit=lambda e, p: True, engine="rest",
    )

    chunks = result["metrics"].get("chunks") or []
    assert chunks, "expected >=1 REST reply chunk metric to carry provenance"
    for c in chunks:
        prov = c.get("provenance")
        assert isinstance(prov, dict), (
            f"chunk metric index={c.get('index')} dropped HLI provenance "
            f"(keys present: {sorted(c.keys())})"
        )
        assert prov.get("request_id"), "chunk provenance missing request_id"
        assert any(prov.get(k) for k in ("served_by", "location", "endpoint")), \
            "chunk provenance missing served_by/location/endpoint"
        assert prov.get("tts_path"), "chunk provenance missing tts_path"
        assert prov.get("queue_path"), "chunk provenance missing queue_path"
        assert prov.get("timing_ms") is not None, \
            "chunk provenance missing aggregate timing_ms"


def test_synthesize_retains_non_secret_hli_response_provenance():
    """RED (additive, NON-cancellable urlopen path): ``synthesize()`` must surface
    the non-secret HLI provenance response headers HiveMind returns with
    /v1/audio/speech (request_id, served_by, location, endpoint, tts_path,
    queue_path, timing_ms) under a ``provenance`` mapping, while dropping every
    sensitive header (Authorization, Set-Cookie/Cookie, X-API-Key, token/credential
    names) from the provenance and the serialized surface (R2 P1 finding 3).

    Fails against current production: synthesize reads the full response header dict
    but returns only audio_bytes/content_type/audio_base64/model/voice/format. A
    copy-all-headers implementation would fail the secret-exclusion assertions.

    The exact HLI header spellings are UNVERIFIED against a live cluster; the RED
    holds regardless because production returns no provenance at all."""
    res = _synthesize_against_provenance_secret_server(cancel_event=None)
    _assert_synthesize_provenance_allowlist(res)


def test_synthesize_cancellable_retains_provenance_and_enforces_allowlist():
    """RED (additive, R2 P1 findings 2 & 3 -- the LIVE cancellable REST TTS path):
    calling ``synthesize()`` with an UNSET ``cancel_event`` selects
    ``_cancellable_speech_request`` exactly as live SSE streaming does (voice.py
    L1498-1505). That path parses the full HTTP response header block (voice.py
    L1413-1416) but returns only content_type. It too must surface the approved
    non-secret HLI provenance and drop every sensitive header from the provenance
    and the serialized surface.

    Fails against current production: the cancellable path returns no provenance at
    all. A copy-all-headers implementation would fail the secret-exclusion
    assertions."""
    cancel_event = threading.Event()  # created but NEVER set -> live streaming shape
    res = _synthesize_against_provenance_secret_server(cancel_event=cancel_event)
    assert not cancel_event.is_set(), "cancel_event must remain unset (live streaming selection)"
    _assert_synthesize_provenance_allowlist(res)


def test_rest_chunk_metric_carries_server_minted_turn_and_chunk_id(monkeypatch):
    """RED (additive correlation contract): every REST reply chunk metric must
    carry a SERVER-minted turn_id (stable across the turn) and a server-minted
    per-chunk chunk_id, so a chunk metric can be correlated 1:1 with the browser
    turn telemetry for the same reply. This lane does NOT define the broader id
    API -- it only asserts the ids are surfaced onto the chunk metric.

    Fails against current production: the chunk metric exposes no server ids."""
    _ms3_voice_ready(monkeypatch)
    fake_transcribe, fake_chat, fake_synthesize = _provenance_chat_and_synth()

    result = voice_ptt_turn_stream(
        runner=_FakeRunner(), audio=b"FAKE_WAV",
        chat_fn=fake_chat, transcribe_fn=fake_transcribe,
        synthesize_fn=fake_synthesize, emit=lambda e, p: True, engine="rest",
    )

    chunks = result["metrics"].get("chunks") or []
    assert chunks, "expected >=1 REST reply chunk metric"
    turn_ids = set()
    for c in chunks:
        turn_id = _chunk_server_id(c, "turn_id")
        chunk_id = _chunk_server_id(c, "chunk_id")
        assert turn_id, (
            f"chunk metric index={c.get('index')} missing server-minted turn_id "
            f"(keys: {sorted(c.keys())})"
        )
        assert chunk_id, (
            f"chunk metric index={c.get('index')} missing server-minted chunk_id"
        )
        turn_ids.add(turn_id)
    assert len(turn_ids) == 1, \
        f"server-minted turn_id must be stable across the turn, got {turn_ids}"


def test_rest_chunk_metric_excludes_secret_from_serialized_surfaces(monkeypatch):
    """RED (additive, R2 P1 finding 3 -- serialized chunk/telemetry surface): when a
    reply chunk carries HLI provenance, the REST turn must retain the approved
    non-secret fields onto ``result["metrics"]["chunks"]`` while NEVER serializing a
    sensitive-named field or a sensitive value onto the chunk metric OR the emitted
    ``audio_chunk`` telemetry.

    Fails against current production: the chunk metric carries no provenance at all
    (retention RED). A copy-all implementation that forwarded provenance verbatim
    would leak the planted secret and fail the secret-exclusion assertions."""
    _ms3_voice_ready(monkeypatch)

    secret_marker = "TURN-SECRET-LEAK-777"
    cookie_marker = "TURNCOOKIESECRET"

    def fake_transcribe(**_):
        return {"text": "Give me two facts.", "model": "whisper-1"}

    def fake_chat(message, *, session_id=None, model=None, stream_callback=None, **_):
        sentences = (
            "The first sentence here is deliberately written to be quite long indeed. ",
            "The second sentence here is also written to be reasonably long as well. ",
        )
        for s in sentences:
            for w in s.split():
                stream_callback(w + " ")
        return {"text": "".join(sentences).strip(), "session_id": "s"}

    call = {"n": 0}

    def fake_synthesize(**kw):
        idx = call["n"]
        call["n"] += 1
        # A chunk whose provenance ALSO carries sensitive fields (as a copy-all
        # header reader would have produced). The retain step must not propagate
        # these to any serialized surface.
        prov = dict(_hli_chunk_provenance(idx))
        prov["authorization"] = f"Bearer {secret_marker}"
        prov["set_cookie"] = f"sid={cookie_marker}; HttpOnly"
        return {
            "audio_bytes": b"\x00" * 8,
            "content_type": "audio/wav",
            "audio_base64": f"AUDIO-{idx}",
            "model": "tts-1",
            "voice": "alloy",
            "format": "wav",
            "provenance": prov,
        }

    emitted: list[tuple[str, dict[str, Any]]] = []

    def emit(event, payload):
        emitted.append((event, dict(payload)))
        return True

    result = voice_ptt_turn_stream(
        runner=_FakeRunner(), audio=b"FAKE_WAV",
        chat_fn=fake_chat, transcribe_fn=fake_transcribe,
        synthesize_fn=fake_synthesize, emit=emit, engine="rest",
    )

    chunks = result["metrics"].get("chunks") or []
    assert chunks, "expected >=1 REST reply chunk metric"
    sensitive_names = {
        "authorization", "set-cookie", "set_cookie", "cookie", "x-api-key",
        "x-auth-token", "x-credential", "proxy-authorization",
    }
    for c in chunks:
        prov = c.get("provenance")
        assert isinstance(prov, dict), (
            f"chunk metric index={c.get('index')} dropped HLI provenance "
            f"(keys present: {sorted(c.keys())})"
        )
        assert prov.get("request_id"), "chunk provenance missing request_id"
        leaked = {str(k).lower() for k in prov} & sensitive_names
        assert not leaked, f"sensitive provenance key(s) leaked onto chunk metric: {sorted(leaked)}"

    # No sensitive VALUE on the serialized chunk metric OR the audio_chunk telemetry.
    audio_payloads = [p for e, p in emitted if e == "audio_chunk"]
    surface = json.dumps(chunks, default=str) + json.dumps(audio_payloads, default=str)
    for secret in (secret_marker, cookie_marker):
        assert secret not in surface, \
            f"secret value {secret!r} leaked into serialized chunk/telemetry surface"


# ---------------------------------------------------------------------------
# R3 correlation contract (R2 P1 finding 2 repair -- PRODUCER end of the chain).
#
# The R2 REDs proved the chunk METRIC could carry ids read from synthesize
# provenance. The R2 review required the correlation be PRODUCER-to-browser end to
# end: production voice.py must MINT one nonempty turn_id (stable across the turn)
# and a nonempty, UNIQUE chunk_id per emitted chunk, and the EMITTED audio_chunk
# telemetry must carry the SAME ids as the per-chunk metric (so a chunk metric and
# the browser turn telemetry for the same reply can be correlated 1:1).
#
# This RED drives the REAL voice.py REST producer with deterministic fakes that
# supply NO ids -- so any id can ONLY come from the producer minting it. Reads ids
# via _chunk_server_id (top-level OR nested provenance; either placement satisfies
# the contract). Fails against frozen production by ASSERTION: _try_emit_ready
# (voice.py L2751-2784) builds the audio_chunk payload AND the metrics timing from a
# fixed key set with no turn_id/chunk_id, so the producer emits neither. server.py
# relay + browser preservation are proven in tests/ms4_gateway.
# ---------------------------------------------------------------------------


def test_rest_producer_mints_turn_and_unique_chunk_ids_end_to_end(monkeypatch):
    """RED (additive correlation contract, producer leg): the REAL REST producer must
    surface one nonempty producer turn_id (stable across the turn) and one nonempty
    producer chunk_id per emitted chunk (unique across the turn) on BOTH the emitted
    ``audio_chunk`` payloads AND ``result['metrics']['chunks']``, with the metric ids
    EQUAL to the emitted ids per chunk.

    Fails against current production: the audio_chunk payload and the per-chunk metric
    are built from a fixed key set with no turn_id/chunk_id, so the producer emits
    neither (the ids are absent, not merely mismatched)."""
    _ms3_voice_ready(monkeypatch)

    # Three sufficiently-long sentences -> chunk 0 fires on the first terminator and
    # the remainder coalesces/flushes into further chunks, so >=2 real producer
    # chunks exist for the uniqueness contract (producer output, not an id assertion).
    reply = (
        "The very first spoken sentence here has plenty of words indeed today. "
        "The second spoken sentence here also has plenty of words indeed now. "
        "The third spoken sentence here finally has plenty of words indeed too."
    )

    def fake_transcribe(**_):
        return {"text": "Give me three producer facts.", "model": "whisper-1"}

    chat_turn_ids: list[str | None] = []

    def fake_chat(
        message,
        *,
        session_id=None,
        model=None,
        stream_callback=None,
        turn_id=None,
        **_,
    ):
        chat_turn_ids.append(turn_id)
        for word in reply.split(" "):
            stream_callback(word + " ")
        return {
            "text": reply,
            "session_id": "sess-prod",
            "turn_id": turn_id,
            "revision_id": 9,
        }

    call = {"n": 0}

    def fake_synthesize(**kw):
        idx = call["n"]
        call["n"] += 1
        # Deterministic audio; NO turn_id/chunk_id supplied -> the producer (voice.py)
        # is the ONLY thing that could mint them. A GREEN impl mints; production emits
        # nothing.
        return {
            "audio_bytes": b"\x00" * 8,
            "content_type": "audio/wav",
            "audio_base64": f"AUDIO-{idx}",
            "model": "tts-1",
            "voice": "alloy",
            "format": "wav",
        }

    emitted: list[tuple[str, dict[str, Any]]] = []

    def emit(event, payload):
        emitted.append((event, dict(payload)))
        return True

    result = voice_ptt_turn_stream(
        runner=_FakeRunner(), audio=b"FAKE_WAV",
        chat_fn=fake_chat, transcribe_fn=fake_transcribe,
        synthesize_fn=fake_synthesize, emit=emit, engine="rest",
    )

    audio_payloads = [p for e, p in emitted if e == "audio_chunk"]
    transcript_payload = next(p for e, p in emitted if e == "transcript")
    # Setup precondition (guaranteed by the multi-sentence reply): >=2 real producer
    # chunks so uniqueness is meaningful. This is producer OUTPUT, not an id contract.
    assert len(audio_payloads) >= 2, (
        f"multi-sentence reply must yield >=2 producer audio chunks; got {len(audio_payloads)}"
    )
    metrics_chunks = result["metrics"].get("chunks") or []
    assert len(metrics_chunks) == len(audio_payloads), (
        "each emitted audio_chunk must have a matching per-chunk metric entry "
        f"(emitted={len(audio_payloads)}, metrics={len(metrics_chunks)})"
    )

    # (1) One nonempty PRODUCER turn_id, stable across every emitted chunk.
    emitted_turn_ids = {_chunk_server_id(p, "turn_id") for p in audio_payloads}
    assert None not in emitted_turn_ids and "" not in emitted_turn_ids, (
        "producer emitted no turn_id on the audio_chunk payloads "
        f"(payload keys: {sorted(audio_payloads[0].keys())})"
    )
    assert len(emitted_turn_ids) == 1, (
        f"producer turn_id must be stable across the turn; got {emitted_turn_ids}"
    )
    turn_id = next(iter(emitted_turn_ids))
    assert transcript_payload["turn_id"] == turn_id
    assert chat_turn_ids == [turn_id]
    assert result["turn_id"] == turn_id
    assert result["revision_id"] == 9

    # (2) One nonempty PRODUCER chunk_id per emitted chunk, UNIQUE across the turn.
    emitted_chunk_ids = [_chunk_server_id(p, "chunk_id") for p in audio_payloads]
    assert all(cid for cid in emitted_chunk_ids), (
        f"every emitted audio_chunk must carry a nonempty producer chunk_id; got {emitted_chunk_ids}"
    )
    assert len(set(emitted_chunk_ids)) == len(emitted_chunk_ids), (
        f"producer chunk_id must be unique per chunk; got {emitted_chunk_ids}"
    )

    # (3) Equality: the per-chunk METRIC ids equal the EMITTED audio_chunk ids (by
    # index), so a metric row and the browser turn telemetry join 1:1.
    emitted_by_index = {p.get("index"): p for p in audio_payloads}
    for mc in metrics_chunks:
        ap = emitted_by_index.get(mc.get("index"))
        assert ap is not None, f"metrics chunk index {mc.get('index')} has no emitted audio_chunk"
        assert _chunk_server_id(mc, "turn_id") == _chunk_server_id(ap, "turn_id"), (
            f"metrics turn_id != emitted turn_id at index {mc.get('index')}"
        )
        assert _chunk_server_id(mc, "chunk_id") == _chunk_server_id(ap, "chunk_id"), (
            f"metrics chunk_id != emitted chunk_id at index {mc.get('index')}"
        )


# ---------------------------------------------------------------------------
# R5 observability repair (fail-first) -- secret-VALUE scrubbing inside an
# ALLOWLISTED provenance field.
#
# R4 review LOW-1 + supervisor SHORT_SECRET_VALUE_PROBE proved the value heuristic
# (_looks_secret_value) is fail-OPEN for a whole class of short, non-scheme secrets
# placed inside an allowlisted routing header value: short credential prefixes
# (sk-/hf_/xoxb- and equivalents), URL query/fragment secrets (token/api_key/
# access_token/signature/credential/password), and nested/list values. The name
# allowlist is the primary defense, but the value filter is the required second
# layer and it must fail CLOSED by omission on BOTH entry points
# (_provenance_from_response_headers ingestion and _sanitize_hli_provenance
# re-filter) so an allowlisted-but-poisoned value never reaches a serialized
# surface (metric/event/log/error).
#
# All secret material below is SYNTHETIC and non-functional (obvious EXAMPLE/NOTREAL
# markers, no real credential). These fail RED against frozen voice.py (3BD72DAC):
# the values are retained because the heuristic only catches Bearer/Basic/Digest,
# 3-part JWTs, and >=64-char opaque blobs.
# ---------------------------------------------------------------------------

# Synthetic, non-functional secret values (never real credentials).
_R5_PREFIX_SECRETS = {
    "openai_sk": "sk-EXAMPLE0000NOTREALKEY",
    "hf_token": "hf_EXAMPLE0000NOTREALTOKEN",
    "slack_bot": "xoxb-EXAMPLE-0000-NOTREAL",
    "github_pat": "ghp_EXAMPLE0000NOTREALTOKEN",
    "gitlab_pat": "glpat-EXAMPLE0000NOTREAL",
}
_R5_URL_SECRETS = {
    "api_key_query": "https://hli.example/v1/audio/speech?api_key=EXAMPLEKEY0000",
    "token_query": "https://hli.example/cb?token=EXAMPLETOKEN0000",
    "access_token_frag": "https://hli.example/cb#access_token=EXAMPLEATOK0000",
    "signature_query": "https://hli.example/o?X-Amz-Signature=EXAMPLESIG0000",
    "credential_query": "https://hli.example/o?credential=EXAMPLECRED0000",
    "password_query": "https://hli.example/o?password=EXAMPLEPW0000",
}
# Legitimate ordinary routing/id/host/path/timing values that MUST be preserved
# (the value filter must not over-scrub). Includes a benign, non-secret query.
_R5_LEGIT_VALUES = {
    "request_id": "hli-req-42",
    "served_by": "hivemind-lobe-07",
    "location": "asia-east1/rack-b/gpu-3",
    "endpoint": "/v1/audio/speech?page=2&limit=10",
    "tts_path": "tts_super/xtts-v2",
    "queue_path": "gim://queue/tts/high",
    "node_id": "gpu0-rack1-dc2",
    "model": "xtts-v2",
    "timing_ms": "137",
}


def _r5_voice():
    import machine_spirit_4.gateway.voice as voice_module
    return voice_module


def test_provenance_from_headers_drops_short_credential_prefix_values():
    """RED: a short credential-prefix secret (sk-/hf_/xoxb-/ghp_/glpat-) placed in an
    ALLOWLISTED routing header value must be dropped by omission -- and must never
    appear on the serialized provenance surface.

    Fails against frozen voice.py: _looks_secret_value ignores short non-scheme
    tokens, so the allowlisted field (request_id) retains the secret."""
    v = _r5_voice()
    for label, secret in _R5_PREFIX_SECRETS.items():
        prov = v._provenance_from_response_headers([("x-request-id", secret)])
        assert "request_id" not in prov, (
            f"[{label}] short credential-prefix secret survived in allowlisted "
            f"provenance field: {prov!r}"
        )
        assert secret not in json.dumps(prov, default=str), (
            f"[{label}] secret value leaked onto serialized provenance surface"
        )
        assert v._looks_secret_value(secret) is True, (
            f"[{label}] _looks_secret_value must flag a short credential prefix"
        )


def test_provenance_from_headers_drops_url_query_and_fragment_secrets():
    """RED: a URL carrying a sensitive query/fragment parameter (token, api_key,
    access_token, signature, credential, password) in an ALLOWLISTED field must be
    dropped by omission.

    Fails against frozen voice.py: URL-embedded secrets are neither scheme nor JWT
    nor >=64 chars, so they are retained."""
    v = _r5_voice()
    for label, secret in _R5_URL_SECRETS.items():
        prov = v._provenance_from_response_headers([("x-tts-path", secret)])
        assert "tts_path" not in prov, (
            f"[{label}] URL secret survived in allowlisted provenance field: {prov!r}"
        )
        assert secret not in json.dumps(prov, default=str), (
            f"[{label}] URL secret value leaked onto serialized provenance surface"
        )
        assert v._looks_secret_value(secret) is True, (
            f"[{label}] _looks_secret_value must flag a URL query/fragment secret"
        )


def test_provenance_from_headers_drops_nested_and_list_secret_values():
    """RED: a nested (dict) or list value in an ALLOWLISTED field that contains a
    secret must fail closed (whole entry omitted).

    Fails against frozen voice.py: _looks_secret_value returns False for non-str, so
    a list/dict value is retained verbatim (secret leaks)."""
    v = _r5_voice()
    nested = {"x-request-id": ["hli-req-ok", "sk-EXAMPLE0000NOTREALKEY"]}
    prov = v._provenance_from_response_headers(nested)
    assert "request_id" not in prov, f"list value with a secret survived: {prov!r}"
    assert "sk-EXAMPLE0000NOTREALKEY" not in json.dumps(prov, default=str), \
        "secret leaked from a list-valued allowlisted field"

    nested_dict = {"x-tts-path": {"host": "ok", "api_key": "hf_EXAMPLE0000NOTREALTOKEN"}}
    prov2 = v._provenance_from_response_headers(nested_dict)
    assert "tts_path" not in prov2, f"nested dict with a secret survived: {prov2!r}"
    assert "hf_EXAMPLE0000NOTREALTOKEN" not in json.dumps(prov2, default=str), \
        "secret leaked from a nested-dict allowlisted field"


def test_sanitize_hli_provenance_drops_secret_valued_allowlisted_fields():
    """RED: the metric re-filter (_sanitize_hli_provenance) -- the SECOND sanitizer
    entry point applied to the per-chunk metric -- must also fail closed against
    secret-valued allowlisted fields (prefix, URL, and nested/list).

    Fails against frozen voice.py: same narrow value heuristic."""
    v = _r5_voice()
    poisoned = {
        "request_id": "sk-EXAMPLE0000NOTREALKEY",
        "served_by": "https://hli.example/o?access_token=EXAMPLEATOK0000",
        "tts_path": ["tts_super/xtts-v2", "xoxb-EXAMPLE-0000-NOTREAL"],
        "queue_path": {"q": "gim://queue", "credential": "hf_EXAMPLE0000NOTREAL"},
    }
    san = v._sanitize_hli_provenance(poisoned)
    for key in ("request_id", "served_by", "tts_path", "queue_path"):
        assert key not in san, f"secret-valued allowlisted field {key!r} survived: {san!r}"
    serialized = json.dumps(san, default=str)
    for marker in ("sk-EXAMPLE", "EXAMPLEATOK", "xoxb-EXAMPLE", "hf_EXAMPLE"):
        assert marker not in serialized, f"secret marker {marker!r} leaked: {serialized!r}"


def test_provenance_value_filter_preserves_legitimate_routing_values():
    """GREEN control (must pass before AND after): the value filter must NOT
    over-scrub ordinary ids/hosts/paths/timings, including a benign non-secret query
    (?page=2&limit=10). Proves the repair is targeted, not a blanket drop."""
    v = _r5_voice()
    header_names = {
        "request_id": "x-request-id", "served_by": "x-hivemind-served-by",
        "location": "x-hivemind-location", "endpoint": "x-hivemind-endpoint",
        "tts_path": "x-tts-path", "queue_path": "x-queue-path",
        "node_id": "x-hivemind-node-id", "model": "x-hivemind-model",
        "timing_ms": "x-hivemind-timing-ms",
    }
    headers = [(header_names[k], val) for k, val in _R5_LEGIT_VALUES.items()]
    prov = v._provenance_from_response_headers(headers)
    for k, val in _R5_LEGIT_VALUES.items():
        assert prov.get(k) == val, (
            f"legitimate routing value over-scrubbed: {k}={val!r} -> {prov.get(k)!r}"
        )
        assert v._looks_secret_value(val) is False, (
            f"legitimate value {k}={val!r} wrongly flagged as secret"
        )
    # The metric re-filter must preserve them identically.
    san = v._sanitize_hli_provenance(dict(_R5_LEGIT_VALUES))
    for k, val in _R5_LEGIT_VALUES.items():
        assert san.get(k) == val, f"metric re-filter over-scrubbed {k}={val!r} -> {san.get(k)!r}"


def test_provenance_scheme_jwt_and_mixedcase_stay_closed():
    """Control (regression guard): the pre-existing scheme/JWT/opaque closures must
    remain closed, including MIXED-CASE scheme spellings, on both entry points."""
    v = _r5_voice()
    closed = {
        "bearer_mixed": "BeArEr abc.def.ghijklmn",
        "basic_scheme": "Basic Zm9vOmJhcnNlY3JldA==",
        "digest_scheme": "Digest username=admin, nonce=deadbeef",
        "jwt_triple": "eyJhbGciOiJI.eyJzdWIiOiIx.SflKxwRJSMeKabc",
        "opaque_64": "Z" * 80,
    }
    for label, secret in closed.items():
        assert v._looks_secret_value(secret) is True, f"[{label}] must stay closed"
        prov = v._provenance_from_response_headers([("x-hivemind-served-by", secret)])
        assert "served_by" not in prov, f"[{label}] scheme/jwt/opaque secret survived: {prov!r}"


# ---------------------------------------------------------------------------
# R6 pre-promotion hardening (fail-first) -- bounded, fail-closed secret
# inspection + exact-token URL parameter refinement.
#
# The fresh independent R5 review (MS4_R5_INDEPENDENT_REVIEW.md) accepted the R5
# value filter as a source candidate but flagged three MANDATORY pre-promotion
# findings. The sanitizer finding (F3) plus the URL precision findings (F4/F5)
# live in this file's scope (voice.py); the browser telemetry findings (F1/F2)
# live in tests/ms4_gateway.
#
#   * F3 (LOW, latent): _looks_secret_value recursed into containers with NO
#     cycle detection, NO depth limit and NO size bound, so a cyclic or deeply
#     nested value raised RecursionError that PROPAGATED through BOTH sanitizer
#     entry points (_sanitize_hli_provenance + _provenance_from_response_headers)
#     -- it failed via exception instead of failing closed by omission.
#   * F4/F5 (INFO): the URL parameter-name check used SUBSTRING collision (so a
#     legitimate near-miss name like tokenizer/signature_algorithm/secretariat was
#     over-omitted), and a known-secret value under an ORDINARY (non-sensitive)
#     parameter name was NOT scanned (mid-URL escape).
#
# R6 makes the recursion explicitly bounded and fail-closed (cycle/depth/size
# overflow and non-JSON-safe byte-like/unknown objects are OMITTED without
# exception; ordinary JSON scalars remain usable) and refines URL inspection to
# match sensitive parameter names by EXACT normalized token while still failing
# closed on a known credential scheme/prefix/JWT/opaque value inside a decoded
# parameter value. It broadens NO logging and never serializes a suspicious value.
#
# All secret material below is SYNTHETIC and non-functional. These fail RED
# against frozen voice.py (D6B01840): the recursion raises RecursionError, a
# byte-like value is retained, a near-miss URL name is over-omitted, and a secret
# under an ordinary URL parameter name escapes.
# ---------------------------------------------------------------------------


def test_r6_recursion_cyclic_and_deep_fails_closed_without_exception():
    """RED (F3, MANDATORY): cyclic and deeply nested container values must be
    OMITTED (treated as secret -> dropped) WITHOUT raising, on _looks_secret_value
    directly AND through BOTH sanitizer entry points.

    Fails against frozen voice.py: the unbounded recursion raises RecursionError
    (default recursionlimit 1000) that propagates through both entry points."""
    v = _r5_voice()

    # (a) self-referential (cyclic) list.
    cyclic_list = ["hli-req-ok"]
    cyclic_list.append(cyclic_list)
    assert v._looks_secret_value(cyclic_list) is True, \
        "a cyclic list must fail closed (omit), not recurse unbounded"

    # (b) self-referential (cyclic) dict.
    cyclic_dict = {"served_by": "ok"}
    cyclic_dict["self"] = cyclic_dict
    assert v._looks_secret_value(cyclic_dict) is True, \
        "a cyclic dict must fail closed (omit), not recurse unbounded"

    # (c) deep nesting well past the Python recursion limit.
    deep: list = []
    current = deep
    for _ in range(5000):
        nxt: list = []
        current.append(nxt)
        current = nxt
    assert v._looks_secret_value(deep) is True, \
        "deeply nested value must fail closed (omit), not raise RecursionError"

    # Both entry points must fail closed by omission -- never raise -- for each.
    for label, poisoned in (("cyclic_list", cyclic_list),
                            ("cyclic_dict", cyclic_dict),
                            ("deep", deep)):
        prov = v._provenance_from_response_headers([("x-request-id", poisoned)])
        assert "request_id" not in prov, \
            f"[{label}] header-ingest entry point must drop the unbounded value: {prov!r}"
        san = v._sanitize_hli_provenance({"request_id": poisoned})
        assert "request_id" not in san, \
            f"[{label}] metric re-filter entry point must drop the unbounded value: {san!r}"


def test_r6_sanitizer_omits_bytelike_and_unknown_objects():
    """RED (F5): a byte-like (bytes/bytearray/memoryview) or otherwise
    non-JSON-safe object must be OMITTED (fail closed), because such a value can
    carry secret bytes the string heuristics never scan.

    Fails against frozen voice.py: every non-str scalar returns False (retained)."""
    v = _r5_voice()

    class _Opaque:
        def __str__(self):  # pragma: no cover - str() must NOT be trusted
            return "sk-EXAMPLE0000NOTREALKEY"

    bytelike = {
        "bytes": b"sk-EXAMPLE0000NOTREALKEY",
        "bytearray": bytearray(b"hf_EXAMPLE0000NOTREAL"),
        "memoryview": memoryview(b"xoxb-EXAMPLE-0000-NOTREAL"),
        "object": _Opaque(),
    }
    for label, val in bytelike.items():
        assert v._looks_secret_value(val) is True, \
            f"[{label}] non-JSON-safe byte-like/unknown value must be omitted"
        prov = v._provenance_from_response_headers([("x-request-id", val)])
        assert "request_id" not in prov, \
            f"[{label}] byte-like/unknown value must be dropped on ingest: {prov!r}"
        san = v._sanitize_hli_provenance({"request_id": val})
        assert "request_id" not in san, \
            f"[{label}] byte-like/unknown value must be dropped on re-filter: {san!r}"


def test_r6_sanitizer_omits_oversized_container():
    """RED (bounded traversal): a container larger than the traversal budget must
    be OMITTED (fail closed) rather than walked in full.

    Fails against frozen voice.py: an unbounded walk returns False for a large
    all-legitimate container (no size bound)."""
    v = _r5_voice()
    oversized_list = ["hli-req-ok"] * 5000
    assert v._looks_secret_value(oversized_list) is True, \
        "an oversized container must fail closed on the traversal budget"
    oversized_dict = {f"route{i}": "ok" for i in range(5000)}
    assert v._looks_secret_value(oversized_dict) is True, \
        "an oversized dict must fail closed on the traversal budget"
    prov = v._provenance_from_response_headers([("x-request-id", oversized_list)])
    assert "request_id" not in prov, "oversized value must be dropped on ingest"


def test_r6_url_param_near_miss_names_are_preserved():
    """RED (F4): a URL whose query/fragment PARAMETER NAME merely CONTAINS a
    sensitive token as a substring (tokenizer, signature_algorithm, secretariat)
    but whose value is ordinary must be PRESERVED (exact-token match, not a
    substring collision).

    Fails against frozen voice.py: the substring match over-omits these."""
    v = _r5_voice()
    near_miss = {
        "tokenizer": "https://hli.example/model?tokenizer=fast",
        "signature_algorithm": "https://hli.example/v?signature_algorithm=RS256",
        "secretariat": "https://hli.example/org?secretariat=on",
    }
    for label, url in near_miss.items():
        assert v._looks_secret_value(url) is False, \
            f"[{label}] near-miss URL param name must not be flagged: {url!r}"
        prov = v._provenance_from_response_headers([("x-tts-path", url)])
        assert prov.get("tts_path") == url, \
            f"[{label}] near-miss URL over-omitted from provenance: {prov!r}"


def test_r6_url_secret_value_under_ordinary_param_fails_closed():
    """RED (F5): a KNOWN credential scheme/prefix/JWT value embedded as the VALUE
    of an ORDINARY (non-sensitive) URL parameter name must fail closed.

    Fails against frozen voice.py: the prefix check is startswith-only on the whole
    string and the URL check inspects parameter NAMES only, so a mid-URL secret
    value under an ordinary name escapes."""
    v = _r5_voice()
    embedded = {
        "prefix_under_next": "https://hli.example/cb?next=sk-EXAMPLE0000NOTREALKEY",
        "bearer_under_q": "https://hli.example/cb?q=Bearer%20EXAMPLE0000NOTREAL",
        "jwt_under_state": "https://hli.example/cb?state=eyJhbGciOiJI.eyJzdWIiOiIx.SflKxwRJSMeKabc",
    }
    for label, url in embedded.items():
        assert v._looks_secret_value(url) is True, \
            f"[{label}] secret VALUE under an ordinary URL param must fail closed: {url!r}"
        prov = v._provenance_from_response_headers([("x-tts-path", url)])
        assert "tts_path" not in prov, \
            f"[{label}] URL with an embedded secret value survived: {prov!r}"


def test_r6_url_sensitive_names_still_fail_closed():
    """Security regression guard (GREEN before AND after): the exact-token
    refinement must NOT open a hole -- every genuinely sensitive URL parameter
    name (including vendor-prefixed X-Amz-Signature) still fails closed."""
    v = _r5_voice()
    sensitive = {
        "api_key": "https://h/o?api_key=EXAMPLEKEY0000",
        "token": "https://h/cb?token=EXAMPLETOKEN0000",
        "access_token": "https://h/cb#access_token=EXAMPLEATOK0000",
        "refresh_token": "https://h/cb?refresh_token=EXAMPLERT0000",
        "id_token": "https://h/cb?id_token=EXAMPLEIDT0000",
        "client_secret": "https://h/o?client_secret=EXAMPLECS0000",
        "session_token": "https://h/o?session_token=EXAMPLEST0000",
        "private_key": "https://h/o?private_key=EXAMPLEPK0000",
        "aws_signature": "https://h/o?X-Amz-Signature=EXAMPLESIG0000",
        "credential": "https://h/o?credential=EXAMPLECRED0000",
        "password": "https://h/o?password=EXAMPLEPW0000",
    }
    for label, url in sensitive.items():
        assert v._looks_secret_value(url) is True, \
            f"[{label}] sensitive URL parameter name must stay closed: {url!r}"


def test_r6_ordinary_json_scalars_remain_usable():
    """GREEN control (must pass before AND after): ordinary JSON scalars (str/int/
    float/bool/None) and ordinary ids/hosts/paths/timings/UUIDs/non-sensitive URLs
    remain usable (not flagged), so the fail-closed hardening stays targeted."""
    v = _r5_voice()
    usable_scalars = [None, True, False, 0, 137, 3.14, "hli-req-42"]
    for val in usable_scalars:
        assert v._looks_secret_value(val) is False, \
            f"ordinary JSON scalar wrongly flagged as secret: {val!r}"
    ordinary = [
        "hivemind-lobe-07", "asia-east1/rack-b/gpu-3", "/v1/audio/speech?page=2&limit=10",
        "tts_super/xtts-v2", "gim://queue/tts/high", "gpu0-rack1-dc2", "xtts-v2",
        "550e8400-e29b-41d4-a716-446655440000",
        "https://peer.internal:8443/health?page=1",
    ]
    for val in ordinary:
        assert v._looks_secret_value(val) is False, \
            f"ordinary provenance value wrongly flagged as secret: {val!r}"
    # A small legitimate nested structure must remain usable (not omitted).
    small_ok = {"routes": ["a", "b"], "meta": {"page": 1, "limit": 10}}
    assert v._looks_secret_value(small_ok) is False, \
        "a small legitimate nested structure must not be omitted"


# ---------------------------------------------------------------------------
# R7 non-finite JSON-safety repair (fail-first) -- NaN / +Inf / -Inf are not
# JSON-safe and must fail closed by OMISSION.
#
# The fresh independent R6 review (MS4_R6_INDEPENDENT_REVIEW.md, review_manifest
# B2A73344) and the peer probe codex-ms4-r6-nonfinite-json-safety-finding
# reproduced LATENT-1 (mandatory-before-promotion): _looks_secret_value_bounded
# classifies EVERY int/float as an "ordinary JSON numeric scalar" (returns
# False -> retained), so a non-finite float (NaN / +Inf / -Inf) survives BOTH
# sanitizer entry points (_provenance_from_response_headers ingest and the
# _sanitize_hli_provenance metric re-filter) under an allowlisted key and then
# breaks strict serialization -- json.dumps(..., allow_nan=False) and a live
# Starlette JSONResponse (which serializes with allow_nan=False) both raise
# ValueError "Out of range float values are not JSON compliant". That directly
# contradicts the sanitizer's own contract that non-JSON-safe values fail closed
# by omission (voice.py docstring L1463-1466 / L1518).
#
# It is latent today only because both production provenance builders derive from
# string-only HTTP response headers; it becomes a live HTTP 500 the moment any
# non-header caller feeds a non-finite float. The fix must fail closed (omit),
# never raise, and must NOT over-scrub finite numbers.
#
# Non-finite floats are inherently NON-secret, so nothing secret is involved; all
# finite/legitimate controls below are ordinary synthetic routing values. These
# tests fail RED against frozen voice.py (7B7197EC): the non-finite value is
# retained and strict serialization raises. The GREEN control (finite/ordinary)
# passes BEFORE and AFTER so the hardening stays targeted.
# ---------------------------------------------------------------------------

# Synthetic non-finite floats (non-secret; NOT JSON-safe under allow_nan=False).
_R7_NONFINITE = {
    "nan": float("nan"),
    "pos_inf": float("inf"),
    "neg_inf": float("-inf"),
}


def test_r7_nonfinite_floats_are_flagged_and_omitted_directly():
    """RED (LATENT-1, MANDATORY): NaN/+Inf/-Inf are not JSON-safe, so
    _looks_secret_value must fail closed (return True -> omit) on each, directly
    and WITHOUT raising -- honoring the sanitizer's own 'non-JSON-safe -> fail
    closed by omission' contract.

    Fails against frozen voice.py (7B7197EC): _looks_secret_value_bounded returns
    False for every int/float, so a non-finite float is retained as an 'ordinary
    JSON numeric scalar'."""
    v = _r5_voice()
    for label, val in _R7_NONFINITE.items():
        result = v._looks_secret_value(val)  # must not raise
        assert result is True, \
            f"[{label}] non-finite float must fail closed (omit); got {result!r}"


def test_r7_nonfinite_under_each_allowlisted_key_fails_closed_both_entry_points():
    """RED (LATENT-1): a non-finite float placed under EVERY allowlisted provenance
    key must be dropped by omission on BOTH sanitizer entry points -- the metric
    re-filter (_sanitize_hli_provenance, canonical keys) and header ingest
    (_provenance_from_response_headers, every header spelling).

    Fails against frozen voice.py: the value is retained under the allowlisted
    key on both paths."""
    v = _r5_voice()
    # (a) metric re-filter -- every canonical allowlisted key.
    for canonical in sorted(v._HLI_PROVENANCE_KEY_ALLOWLIST):
        for label, val in _R7_NONFINITE.items():
            san = v._sanitize_hli_provenance({canonical: val})
            assert canonical not in san, \
                f"[{label}] non-finite under allowlisted key {canonical!r} survived re-filter: {san!r}"
    # (b) header ingest -- every header spelling that maps to an allowlisted key.
    for header_name, canonical in v._HLI_PROVENANCE_HEADER_MAP.items():
        for label, val in _R7_NONFINITE.items():
            prov = v._provenance_from_response_headers([(header_name, val)])
            assert canonical not in prov, \
                f"[{label}] non-finite via header {header_name!r} survived ingest: {prov!r}"


def test_r7_nonfinite_nested_in_list_and_dict_fails_closed():
    """RED (LATENT-1): a non-finite float hidden inside a list or dict value of an
    allowlisted field must fail closed (whole entry omitted) -- directly and
    through both entry points.

    Fails against frozen voice.py: containers recurse but the non-finite leaf
    returns False, so the whole container is retained."""
    v = _r5_voice()
    for label, val in _R7_NONFINITE.items():
        assert v._looks_secret_value(["hli-req-ok", val]) is True, \
            f"[{label}] non-finite inside a list must fail closed"
        assert v._looks_secret_value({"served_by": "ok", "extra": val}) is True, \
            f"[{label}] non-finite inside a dict must fail closed"
        prov = v._provenance_from_response_headers([("x-request-id", ["hli-req-ok", val])])
        assert "request_id" not in prov, \
            f"[{label}] list with a non-finite leaf survived ingest: {prov!r}"
        san = v._sanitize_hli_provenance({"tts_path": {"host": "ok", "n": val}})
        assert "tts_path" not in san, \
            f"[{label}] nested dict with a non-finite leaf survived re-filter: {san!r}"


def test_r7_sanitized_provenance_is_strict_json_and_starlette_safe():
    """RED (LATENT-1): after sanitizing a provenance carrying non-finite floats
    under allowlisted keys, the result must serialize under strict
    json.dumps(allow_nan=False) AND a live Starlette JSONResponse (when
    importable) WITHOUT raising, because the non-finite values were omitted;
    finite/str controls in the same payload must be preserved.

    Fails against frozen voice.py: the non-finite values are retained, so the
    'not in' assertions fail and strict serialization raises ValueError
    ('Out of range float values are not JSON compliant')."""
    v = _r5_voice()
    poisoned_headers = [
        ("x-hivemind-timing-ms", float("nan")),
        ("x-hivemind-queue-depth-total", float("inf")),
        ("x-hivemind-queue-max-depth", float("-inf")),
        ("x-hivemind-served-by", "hivemind-lobe-07"),  # finite/str controls preserved
        ("x-hivemind-model", "xtts-v2"),
    ]
    prov = v._provenance_from_response_headers(poisoned_headers)
    for canonical in ("timing_ms", "queue_depth_total", "queue_max_depth"):
        assert canonical not in prov, \
            f"non-finite {canonical!r} survived header ingest: {prov!r}"
    assert prov.get("served_by") == "hivemind-lobe-07"
    assert prov.get("model") == "xtts-v2"
    strict_prov = json.dumps(prov, allow_nan=False)  # raises on frozen voice.py
    assert isinstance(strict_prov, str)

    # Metric re-filter path (the second sanitizer entry point).
    san = v._sanitize_hli_provenance({
        "timing_ms": float("nan"),
        "queue_depth_total": float("inf"),
        "queue_max_depth": float("-inf"),
        "model": "xtts-v2",
        "request_id": "hli-req-42",
    })
    for canonical in ("timing_ms", "queue_depth_total", "queue_max_depth"):
        assert canonical not in san, \
            f"non-finite {canonical!r} survived metric re-filter: {san!r}"
    assert san.get("model") == "xtts-v2"
    assert san.get("request_id") == "hli-req-42"
    strict_san = json.dumps(san, allow_nan=False)  # raises on frozen voice.py
    assert isinstance(strict_san, str)

    # Starlette JSONResponse is the exact production failure surface: it renders
    # with allow_nan=False and raises on a retained non-finite value.
    try:
        from starlette.responses import JSONResponse
    except Exception:  # pragma: no cover - starlette optional in some envs
        JSONResponse = None
    if JSONResponse is not None:
        JSONResponse(prov)  # must not raise
        JSONResponse(san)   # must not raise


def test_r7_finite_and_ordinary_values_preserved_control():
    """GREEN control (must pass BEFORE and AFTER the fix): the non-finite
    fail-closed hardening must stay targeted. Finite int/float (incl. zero,
    negatives, a huge int too big to be a float, and the largest finite float),
    bool, None, strings, small nested structures, and the existing ordinary
    provenance values must remain usable and strict-JSON-safe."""
    v = _r5_voice()
    finite_usable = [None, True, False, 0, -7, 137, 10 ** 400, 0.0, -2.5, 3.14, 1e308, "hli-req-42"]
    for val in finite_usable:
        assert v._looks_secret_value(val) is False, \
            f"finite/ordinary scalar wrongly flagged as secret: {val!r}"

    # Existing ordinary provenance values preserved (no over-scrub) on both paths.
    header_names = {
        "request_id": "x-request-id", "served_by": "x-hivemind-served-by",
        "location": "x-hivemind-location", "endpoint": "x-hivemind-endpoint",
        "tts_path": "x-tts-path", "queue_path": "x-queue-path",
        "node_id": "x-hivemind-node-id", "model": "x-hivemind-model",
        "timing_ms": "x-hivemind-timing-ms",
    }
    headers = [(header_names[k], val) for k, val in _R5_LEGIT_VALUES.items()]
    prov = v._provenance_from_response_headers(headers)
    for k, val in _R5_LEGIT_VALUES.items():
        assert prov.get(k) == val, \
            f"ordinary provenance value over-scrubbed: {k}={val!r} -> {prov.get(k)!r}"

    # Finite numeric values under allowlisted keys survive and stay strict-JSON-safe.
    numeric = v._provenance_from_response_headers([
        ("x-hivemind-timing-ms", 137),
        ("x-hivemind-queue-depth-total", 3.5),
        ("x-hivemind-queue-max-depth", 0),
    ])
    assert numeric.get("timing_ms") == 137
    assert numeric.get("queue_depth_total") == 3.5
    assert numeric.get("queue_max_depth") == 0
    assert isinstance(json.dumps(numeric, allow_nan=False), str)
    san = v._sanitize_hli_provenance({"timing_ms": 137, "queue_depth_total": 3.5, "model": "xtts-v2"})
    assert san.get("timing_ms") == 137
    assert san.get("queue_depth_total") == 3.5
    assert san.get("model") == "xtts-v2"

    # A small legitimate nested structure with finite numbers must remain usable.
    small_ok = {"routes": ["a", "b"], "meta": {"page": 1, "limit": 10, "ratio": 0.5}}
    assert v._looks_secret_value(small_ok) is False, \
        "a small legitimate finite nested structure must not be omitted"


# ---------------------------------------------------------------------------
# R8 JSON-native container hardening (fail-first) -- a finite set / frozenset of
# only non-secret JSON scalars is NOT JSON-native and must fail closed by
# OMISSION, exactly like every other non-JSON-native / unknown object.
#
# The fresh independent R7 review (MS4_R7_INDEPENDENT_REVIEW.md, review_manifest
# ACCEPT_R7_SOURCE_FOR_ROLLBACK_GUARDED_PROMOTION) independently reproduced this
# pre-existing residual as OBS-1 (informational, out of R7 scope): the container
# branch of _looks_secret_value_bounded treats (list, tuple, set, frozenset) as
# recursively acceptable, so a finite set/frozenset containing only non-secret
# JSON scalars recurses to False (retained) under an allowlisted key on BOTH
# sanitizer entry points, then raises TypeError -- NOT ValueError -- in strict
# json.dumps(..., allow_nan=False) and a live Starlette JSONResponse (both
# serialize with allow_nan=False). That contradicts the sanitizer's own explicit
# "non-JSON-safe / unknown object -> fail closed by omission, never by exception"
# contract for EVERY caller (voice.py docstring L1466-1469 / the L1527 unknown
# path), even though it is not live-reachable from today's string-only HTTP
# header entry points.
#
# list and tuple ARE JSON-native (json.dumps serializes both as JSON arrays), so
# they must remain RETAINED and JSON-safe; only non-JSON-native set/frozenset
# fail closed. The fix must omit set/frozenset (never raise, never stringify) and
# must NOT over-scrub list/tuple, nor disturb any adjacent invariant (cycle/DAG
# handling, non-finite omission, bool-before-int, huge-int, allowlist, secret
# detection, recursion budgets).
#
# set/frozenset scalars are inherently NON-secret; all fixtures are ordinary
# synthetic routing values. These fail RED against frozen voice.py (2801D8C0):
# the set/frozenset is retained and strict serialization raises TypeError. The
# GREEN controls (list/tuple retained + adjacent invariants) pass BEFORE and
# AFTER so the hardening stays targeted and rejects an over-broad mutant.
# ---------------------------------------------------------------------------


def test_r8_set_and_frozenset_fail_closed_directly():
    """RED (OBS-1): a finite set/frozenset of only non-secret JSON scalars is not
    JSON-native, so _looks_secret_value must fail closed (return True -> omit) on
    each -- directly, nested inside a JSON-native container, and WITHOUT raising --
    honoring the sanitizer's 'non-JSON-safe / unknown object -> omit' contract.

    Fails against frozen voice.py (2801D8C0): the container branch recurses into a
    set/frozenset and every non-secret scalar returns False, so the set/frozenset
    is retained as if it were an acceptable container."""
    v = _r5_voice()
    setlike_nonsecret = {
        "set_str": {"hli-lobe-a", "hli-lobe-b"},
        "frozenset_str": frozenset({"route-1", "route-2"}),
        "set_int": {1, 2, 3},
        "frozenset_int": frozenset({137, 42}),
        "set_single": {"xtts-v2"},
        "set_empty": set(),
        "frozenset_empty": frozenset(),
    }
    for label, val in setlike_nonsecret.items():
        result = v._looks_secret_value(val)  # must not raise
        assert result is True, (
            f"[{label}] non-JSON-native {type(val).__name__} must fail closed "
            f"(omit); got {result!r}"
        )
    # A set/frozenset hidden inside a JSON-native container fails the whole node.
    assert v._looks_secret_value(["hli-req-ok", {"lobe-a", "lobe-b"}]) is True, \
        "a set nested in a list must fail closed"
    assert v._looks_secret_value(("ok", frozenset({"a", "b"}))) is True, \
        "a frozenset nested in a tuple must fail closed"
    assert v._looks_secret_value({"served_by": "ok", "peers": {"lobe-a", "lobe-b"}}) is True, \
        "a set nested in a dict must fail closed"


def test_r8_set_and_frozenset_fail_closed_both_entry_points():
    """RED (OBS-1): a set/frozenset placed under EVERY allowlisted provenance key
    must be dropped by omission on BOTH sanitizer entry points -- the metric
    re-filter (_sanitize_hli_provenance, canonical keys) and header ingest
    (_provenance_from_response_headers, every header spelling) -- plus when nested
    inside a JSON-native container.

    Fails against frozen voice.py: the set/frozenset is retained under the
    allowlisted key on both paths."""
    v = _r5_voice()
    setlike = {
        "set": {"lobe-a", "lobe-b"},
        "frozenset": frozenset({"route-1", "route-2"}),
        "set_int": {137, 42},
        "frozenset_int": frozenset({1, 2, 3}),
        "set_empty": set(),
    }
    # (a) metric re-filter -- every canonical allowlisted key.
    for canonical in sorted(v._HLI_PROVENANCE_KEY_ALLOWLIST):
        for label, val in setlike.items():
            san = v._sanitize_hli_provenance({canonical: val})
            assert canonical not in san, (
                f"[{label}] set-like under allowlisted key {canonical!r} survived "
                f"re-filter: {san!r}"
            )
    # (b) header ingest -- every header spelling that maps to an allowlisted key.
    for header_name, canonical in v._HLI_PROVENANCE_HEADER_MAP.items():
        for label, val in setlike.items():
            prov = v._provenance_from_response_headers([(header_name, val)])
            assert canonical not in prov, (
                f"[{label}] set-like via header {header_name!r} survived ingest: {prov!r}"
            )
    # Nested set-like inside a JSON-native container, on both entry points.
    prov = v._provenance_from_response_headers([("x-request-id", ["hli-req-ok", {"a", "b"}])])
    assert "request_id" not in prov, f"list with a set leaf survived ingest: {prov!r}"
    san = v._sanitize_hli_provenance({"tts_path": {"host": "ok", "peers": frozenset({"a", "b"})}})
    assert "tts_path" not in san, f"nested dict with a frozenset leaf survived re-filter: {san!r}"


def test_r8_sanitized_provenance_with_setlike_is_strict_json_and_starlette_safe():
    """RED (OBS-1): after sanitizing a provenance carrying set/frozenset values
    under allowlisted keys, the result must serialize under strict
    json.dumps(allow_nan=False) AND a live Starlette JSONResponse WITHOUT raising,
    because the set/frozenset values were omitted; finite/str controls in the same
    payload must be preserved.

    Fails against frozen voice.py: the set/frozenset values are retained, so the
    'not in' assertions fail (and strict serialization would raise TypeError:
    'Object of type set is not JSON serializable')."""
    v = _r5_voice()
    # Control (raw Python, stable before/after): a set/frozenset is genuinely NOT
    # JSON-native -- strict stdlib JSON raises TypeError. This is the exact hazard
    # the sanitizer must prevent (distinct from R7's non-finite-float ValueError).
    with pytest.raises(TypeError):
        json.dumps({"peer": {"lobe-a", "lobe-b"}}, allow_nan=False)
    with pytest.raises(TypeError):
        json.dumps({"queue_path": frozenset({"q1", "q2"})}, allow_nan=False)

    poisoned_headers = [
        ("x-hivemind-peer", {"lobe-a", "lobe-b"}),            # set -> omit
        ("x-hivemind-queue-path", frozenset({"q1", "q2"})),   # frozenset -> omit
        ("x-hivemind-served-by", "hivemind-lobe-07"),         # str control -> keep
        ("x-hivemind-timing-ms", 137),                        # int control -> keep
        ("x-hivemind-queue-depth-total", 3.5),                # finite float -> keep
    ]
    prov = v._provenance_from_response_headers(poisoned_headers)
    assert "peer" not in prov, f"set survived header ingest: {prov!r}"
    assert "queue_path" not in prov, f"frozenset survived header ingest: {prov!r}"
    assert prov.get("served_by") == "hivemind-lobe-07"
    assert prov.get("timing_ms") == 137
    assert prov.get("queue_depth_total") == 3.5
    strict_prov = json.dumps(prov, allow_nan=False)  # raises on frozen 2801D8C0
    assert isinstance(strict_prov, str)

    # Metric re-filter path (the second sanitizer entry point).
    san = v._sanitize_hli_provenance({
        "peer": {"lobe-a", "lobe-b"},
        "queue_path": frozenset({"q1", "q2"}),
        "model": "xtts-v2",
        "request_id": "hli-req-42",
    })
    assert "peer" not in san, f"set survived metric re-filter: {san!r}"
    assert "queue_path" not in san, f"frozenset survived metric re-filter: {san!r}"
    assert san.get("model") == "xtts-v2"
    assert san.get("request_id") == "hli-req-42"
    strict_san = json.dumps(san, allow_nan=False)  # raises on frozen 2801D8C0
    assert isinstance(strict_san, str)

    # Starlette JSONResponse is the exact production serialization surface: it
    # renders with allow_nan=False and raises TypeError on a retained set/frozenset.
    try:
        from starlette.responses import JSONResponse
    except Exception:  # pragma: no cover - starlette optional in some envs
        JSONResponse = None
    if JSONResponse is not None:
        JSONResponse(prov)  # must not raise
        JSONResponse(san)   # must not raise
        with pytest.raises(TypeError):
            JSONResponse({"peer": {"lobe-a", "lobe-b"}})  # a retained set WOULD raise here


def test_r8_list_and_tuple_remain_retained_and_json_safe_control():
    """GREEN control (must pass BEFORE and AFTER): JSON-native list and tuple values
    (json.dumps serializes both as JSON arrays) must remain RETAINED and strict-JSON
    / Starlette safe. The hardening must fail closed ONLY on non-JSON-native
    set/frozenset, never on list/tuple. This also KILLS an over-broad mutant that
    rejects list/tuple."""
    v = _r5_voice()
    json_native = {
        "list_str": ["hli-lobe-a", "hli-lobe-b"],
        "tuple_str": ("route-1", "route-2"),
        "list_int": [1, 2, 3],
        "tuple_int": (137, 42),
        "list_empty": [],
        "tuple_empty": (),
        "nested_list_tuple": ["ok", ("a", "b"), ["c", 1]],
        "list_of_dict": [{"page": 1}, {"limit": 10}],
    }
    for label, val in json_native.items():
        assert v._looks_secret_value(val) is False, (
            f"[{label}] JSON-native {type(val).__name__} must remain retained (not omitted)"
        )

    # Retained through both entry points and strict-JSON / Starlette safe.
    prov = v._provenance_from_response_headers([
        ("x-hivemind-peer", ["lobe-a", "lobe-b"]),
        ("x-hivemind-queue-path", ("q1", "q2")),
    ])
    assert prov.get("peer") == ["lobe-a", "lobe-b"], f"list value over-scrubbed: {prov!r}"
    assert prov.get("queue_path") == ("q1", "q2"), f"tuple value over-scrubbed: {prov!r}"
    assert isinstance(json.dumps(prov, allow_nan=False), str)

    san = v._sanitize_hli_provenance({
        "peer": ["lobe-a", "lobe-b"],
        "queue_path": ("q1", "q2"),
        "model": "xtts-v2",
    })
    assert san.get("peer") == ["lobe-a", "lobe-b"]
    assert san.get("queue_path") == ("q1", "q2")
    assert san.get("model") == "xtts-v2"
    assert isinstance(json.dumps(san, allow_nan=False), str)

    # A tuple serializes as a JSON array (proves list/tuple are JSON-native).
    assert json.loads(json.dumps({"t": ("a", "b")})) == {"t": ["a", "b"]}

    try:
        from starlette.responses import JSONResponse
    except Exception:  # pragma: no cover - starlette optional in some envs
        JSONResponse = None
    if JSONResponse is not None:
        JSONResponse(prov)  # must not raise
        JSONResponse(san)   # must not raise


def test_r8_adjacent_invariants_unchanged_control():
    """GREEN control (must pass BEFORE and AFTER): the set/frozenset hardening must
    NOT disturb any adjacent invariant -- cycle/DAG handling, non-finite omission,
    bool-before-int, huge-int retention, the allowlist, secret detection, and the
    recursion budgets all behave exactly as before. Also KILLS an over-broad mutant
    (its JSON-native list/tuple/dict nodes would be wrongly flagged)."""
    v = _r5_voice()

    # Cycles fail closed (list + dict).
    cyclic_list = ["ok"]
    cyclic_list.append(cyclic_list)
    assert v._looks_secret_value(cyclic_list) is True
    cyclic_dict = {"served_by": "ok"}
    cyclic_dict["self"] = cyclic_dict
    assert v._looks_secret_value(cyclic_dict) is True

    # DAG (shared, non-cyclic) JSON-native nodes retained (not misread as a cycle).
    shared = ["hli-lobe-a", "hli-lobe-b"]
    dag = {"primary": shared, "mirror": shared, "route": ("x", shared)}
    assert v._looks_secret_value(dag) is False, \
        "a shared DAG of JSON-native nodes must be retained, not misread as a cycle"

    # Non-finite float still fails closed by omission (R7 invariant preserved).
    assert v._looks_secret_value([float("nan")]) is True
    assert v._looks_secret_value(("ok", float("inf"))) is True
    assert v._looks_secret_value({"timing_ms": float("-inf")}) is True

    # bool-before-int and huge-int retained (finite scalars usable; no OverflowError).
    for scalar in (None, True, False, 0, -7, 137, 10 ** 400, 0.0, -2.5, 1e308):
        assert v._looks_secret_value(scalar) is False, f"finite scalar wrongly flagged: {scalar!r}"

    # Allowlist unchanged: a non-allowlisted key is dropped by omission.
    assert v._sanitize_hli_provenance({"not_allowlisted": "x", "model": "xtts-v2"}) == {"model": "xtts-v2"}

    # Secret detection unchanged: a known synthetic secret still fails closed,
    # directly and inside a JSON-native container.
    secret = "sk-EXAMPLE0000NOTREALKEY"
    assert v._looks_secret_value(secret) is True
    assert v._looks_secret_value(["ok", secret]) is True
    assert v._looks_secret_value({"served_by": "ok", "x": secret}) is True

    # Recursion budgets unchanged: an oversized JSON-native container fails closed.
    assert v._looks_secret_value(["ok"] * 5000) is True
    assert v._looks_secret_value({f"r{i}": "ok" for i in range(5000)}) is True


# ---------------------------------------------------------------------------
# R9 JSON-native MAPPING-KEY hardening (fail-first) -- a dict whose KEY is not a
# JSON-coercible scalar (tuple / frozenset / bytes / arbitrary object) or is a
# non-finite float (NaN / +Inf / -Inf) is NOT strict-JSON serializable and must
# fail closed by OMISSION, exactly like every other non-JSON-native object.
#
# The fresh independent R8 review (MS4_R8_INDEPENDENT_REVIEW.md, verdict
# ACCEPT_R8_SOURCE_FOR_ROLLBACK_GUARDED_PROMOTION) independently reproduced this
# pre-existing residual as OBS-2 (informational, out of R8 scope): the dict
# branch of _looks_secret_value_bounded converts each key to text for the
# secret-NAME inspection (str(key)) but RETAINS the original mapping unchanged,
# so a non-JSON-coercible key survives. Such a dict recurses to False (retained)
# under an allowlisted key on BOTH sanitizer entry points, then raises in strict
# json.dumps(..., allow_nan=False) and a live Starlette JSONResponse (both
# serialize with allow_nan=False): a tuple/frozenset/bytes/arbitrary-object key
# raises TypeError ("keys must be str, int, float, bool or None") and a
# non-finite float key raises ValueError ("Out of range float values are not
# JSON compliant"). That contradicts the sanitizer's own explicit "non-JSON-safe
# / unknown object -> fail closed by omission, never by exception" contract for
# EVERY caller (voice.py docstring L1466-1472), even though it is not
# live-reachable from today's string-only HTTP header entry points.
#
# str / int (incl. a huge int too big to be a float) / finite float / bool / None
# keys ARE JSON-coercible object keys (json.dumps serializes them), so a mapping
# using ONLY those keys -- including NESTED dictionaries -- must remain RETAINED
# and JSON-safe; only non-coercible / non-finite-float keys fail closed. The fix
# must omit such a mapping (never raise, never stringify the invalid key, never
# mutate the mapping) and must NOT over-scrub a valid-key mapping, nor disturb any
# adjacent invariant (list/tuple retention, set/frozenset VALUE omission,
# non-finite VALUE omission, cycle/DAG handling, bool-before-int, huge-int,
# allowlist, secret detection, recursion budgets).
#
# Non-JSON-coercible keys are inherently NON-secret; all fixtures are ordinary
# synthetic routing values. Tests r9_*_directly / *_both_entry_points /
# *_strict_json_and_starlette_safe fail RED against R8 voice.py (AAEF9894): the
# invalid-key dict is retained and strict serialization raises. The GREEN
# controls (valid-key mappings retained + adjacent invariants) pass BEFORE and
# AFTER so the hardening stays targeted and rejects an over-broad mutant that
# would reject valid int / finite-float / bool / None keys.
# ---------------------------------------------------------------------------


class _R9OpaqueKey:
    """An arbitrary, hashable, non-JSON-coercible object used only as a mapping
    KEY. Default object identity hashing makes it a valid dict key, but strict
    json.dumps / Starlette JSONResponse cannot serialize it as an object key."""
    __slots__ = ()


_R9_OPAQUE_KEY = _R9OpaqueKey()

# Synthetic dicts whose sole KEY is non-JSON-coercible or a non-finite float. The
# VALUES are ordinary non-secret strings, so the ONLY reason for omission is the
# key (never a secret value or a secret-named key). Each is paired with the exact
# strict json.dumps(allow_nan=False) exception it triggers when wrongly retained:
# tuple/frozenset/bytes/arbitrary-object keys -> TypeError; non-finite float
# keys -> ValueError.
_R9_INVALID_KEY_DICTS = {
    "tuple_key": ({("r1", "r2"): "hli-ok"}, TypeError),
    "frozenset_key": ({frozenset({"lobe-a", "lobe-b"}): "hli-ok"}, TypeError),
    "bytes_key": ({b"nodebytes": "hli-ok"}, TypeError),
    "object_key": ({_R9_OPAQUE_KEY: "hli-ok"}, TypeError),
    "nan_key": ({float("nan"): "hli-ok"}, ValueError),
    "posinf_key": ({float("inf"): "hli-ok"}, ValueError),
    "neginf_key": ({float("-inf"): "hli-ok"}, ValueError),
}


def test_r9_nonjson_mapping_keys_fail_closed_directly():
    """RED (OBS-2): a dict whose KEY is non-JSON-coercible (tuple / frozenset /
    bytes / arbitrary object) or a non-finite float (NaN / +Inf / -Inf) is not
    strict-JSON serializable, so _looks_secret_value must fail closed (return
    True -> omit) on each -- directly, nested inside a JSON-native container, and
    deeply nested -- and WITHOUT raising, honoring the sanitizer's 'non-JSON-safe
    -> omit' contract.

    Fails against R8 voice.py (AAEF9894): the dict branch stringifies the key for
    secret-NAME inspection but retains the original mapping, so every non-secret
    value returns False and the invalid-key dict is retained."""
    v = _r5_voice()
    for label, (payload, _exc) in _R9_INVALID_KEY_DICTS.items():
        result = v._looks_secret_value(payload)  # must not raise
        assert result is True, (
            f"[{label}] dict with a non-JSON-coercible key must fail closed "
            f"(omit); got {result!r}"
        )
        # Nested inside JSON-native containers -> the whole node fails closed.
        assert v._looks_secret_value(["hli-req-ok", payload]) is True, \
            f"[{label}] invalid-key dict nested in a list must fail closed"
        assert v._looks_secret_value(("ok", payload)) is True, \
            f"[{label}] invalid-key dict nested in a tuple must fail closed"
        assert v._looks_secret_value({"served_by": "ok", "nested": payload}) is True, \
            f"[{label}] invalid-key dict nested in a dict must fail closed"
        # Deeply nested (dict -> dict -> list -> dict-with-invalid-key).
        deep = {"a": {"b": ["ok", {"c": payload}]}}
        assert v._looks_secret_value(deep) is True, \
            f"[{label}] deeply nested invalid-key dict must fail closed"


def test_r9_nonjson_mapping_keys_fail_closed_both_entry_points():
    """RED (OBS-2): an invalid-key dict placed under EVERY allowlisted provenance
    key must be dropped by omission on BOTH sanitizer entry points -- the metric
    re-filter (_sanitize_hli_provenance, canonical keys) and header ingest
    (_provenance_from_response_headers, every header spelling) -- plus when nested
    inside a JSON-native container.

    Fails against R8 voice.py: the invalid-key dict is retained under the
    allowlisted key on both paths."""
    v = _r5_voice()
    # (a) metric re-filter -- every canonical allowlisted key.
    for canonical in sorted(v._HLI_PROVENANCE_KEY_ALLOWLIST):
        for label, (payload, _exc) in _R9_INVALID_KEY_DICTS.items():
            san = v._sanitize_hli_provenance({canonical: payload})
            assert canonical not in san, (
                f"[{label}] invalid-key dict under allowlisted key {canonical!r} "
                f"survived re-filter: {san!r}"
            )
    # (b) header ingest -- every header spelling that maps to an allowlisted key.
    for header_name, canonical in v._HLI_PROVENANCE_HEADER_MAP.items():
        for label, (payload, _exc) in _R9_INVALID_KEY_DICTS.items():
            prov = v._provenance_from_response_headers([(header_name, payload)])
            assert canonical not in prov, (
                f"[{label}] invalid-key dict via header {header_name!r} survived "
                f"ingest: {prov!r}"
            )
    # Nested invalid-key dict inside a JSON-native container, on both entry points.
    prov = v._provenance_from_response_headers(
        [("x-request-id", ["hli-req-ok", {b"k": "v"}])]
    )
    assert "request_id" not in prov, \
        f"list with an invalid-key-dict leaf survived ingest: {prov!r}"
    san = v._sanitize_hli_provenance({"tts_path": {"host": "ok", "extra": {("t",): "v"}}})
    assert "tts_path" not in san, \
        f"nested dict with an invalid-key-dict leaf survived re-filter: {san!r}"


def test_r9_sanitized_provenance_with_nonjson_keys_is_strict_json_and_starlette_safe():
    """RED (OBS-2): after sanitizing a provenance carrying invalid-key dicts under
    allowlisted keys, the result must serialize under strict
    json.dumps(allow_nan=False) AND a live Starlette JSONResponse WITHOUT raising,
    because the invalid-key dicts were omitted; finite/str controls in the same
    payload must be preserved.

    Fails against R8 voice.py: the invalid-key dicts are retained, so the 'not in'
    assertions fail (and strict serialization would raise TypeError for a
    tuple/frozenset/bytes/object key, ValueError for a non-finite float key)."""
    v = _r5_voice()
    # Control (raw Python, stable before/after): each invalid-key dict is genuinely
    # NOT strict-JSON serializable -- tuple/frozenset/bytes/object keys raise
    # TypeError; non-finite float keys raise ValueError under allow_nan=False.
    for label, (payload, exc) in _R9_INVALID_KEY_DICTS.items():
        with pytest.raises(exc):
            json.dumps(payload, allow_nan=False)

    poisoned_headers = [
        ("x-hivemind-peer", {("lobe-a", "lobe-b"): "x"}),      # tuple key -> omit
        ("x-hivemind-queue-path", {b"q": "x"}),                # bytes key -> omit
        ("x-hivemind-node-id", {frozenset({"n"}): "x"}),       # frozenset key -> omit
        ("x-hivemind-location", {float("nan"): "x"}),          # non-finite key -> omit
        ("x-hivemind-served-by", "hivemind-lobe-07"),          # str control -> keep
        ("x-hivemind-timing-ms", 137),                         # int control -> keep
        ("x-hivemind-queue-depth-total", 3.5),                 # finite float -> keep
    ]
    prov = v._provenance_from_response_headers(poisoned_headers)
    for canonical in ("peer", "queue_path", "node_id", "location"):
        assert canonical not in prov, \
            f"{canonical!r} invalid-key dict survived header ingest: {prov!r}"
    assert prov.get("served_by") == "hivemind-lobe-07"
    assert prov.get("timing_ms") == 137
    assert prov.get("queue_depth_total") == 3.5
    strict_prov = json.dumps(prov, allow_nan=False)  # raises on R8 (AAEF9894)
    assert isinstance(strict_prov, str)

    # Metric re-filter path (the second sanitizer entry point).
    san = v._sanitize_hli_provenance({
        "peer": {("lobe-a",): "x"},
        "queue_path": {b"q": "x"},
        "node_id": {_R9_OPAQUE_KEY: "x"},
        "location": {float("inf"): "x"},
        "model": "xtts-v2",
        "request_id": "hli-req-42",
    })
    for canonical in ("peer", "queue_path", "node_id", "location"):
        assert canonical not in san, \
            f"{canonical!r} invalid-key dict survived metric re-filter: {san!r}"
    assert san.get("model") == "xtts-v2"
    assert san.get("request_id") == "hli-req-42"
    strict_san = json.dumps(san, allow_nan=False)  # raises on R8 (AAEF9894)
    assert isinstance(strict_san, str)

    # Starlette JSONResponse is the exact production serialization surface: it
    # renders with allow_nan=False and raises on a retained invalid-key dict.
    try:
        from starlette.responses import JSONResponse
    except Exception:  # pragma: no cover - starlette optional in some envs
        JSONResponse = None
    if JSONResponse is not None:
        JSONResponse(prov)  # must not raise
        JSONResponse(san)   # must not raise
        with pytest.raises(TypeError):
            JSONResponse({"peer": {("lobe-a",): "x"}})  # retained tuple-key dict WOULD raise
        with pytest.raises(ValueError):
            JSONResponse({"loc": {float("nan"): "x"}})  # retained non-finite-key dict WOULD raise


def test_r9_valid_json_mapping_keys_remain_retained_control():
    """GREEN control (must pass BEFORE and AFTER): a mapping whose keys are all
    JSON-coercible -- str, int (incl. a huge int too big to be a float), finite
    float, bool, and None, including NESTED dictionaries -- must remain RETAINED
    and strict-JSON / Starlette safe. The hardening must fail closed ONLY on
    non-coercible / non-finite-float keys, never on these. This KILLS an
    over-broad mutant that rejects valid int / finite-float / bool / None keys."""
    v = _r5_voice()
    valid_key_maps = {
        "str_keys": {"served_by": "ok", "model": "xtts-v2"},
        "int_keys": {1: "ok", 2: "ok"},
        "huge_int_key": {10 ** 400: "ok"},
        "finite_float_keys": {1.5: "ok", -2.0: "ok", 0.0: "ok"},
        "bool_keys": {True: "ok", False: "ok"},
        "none_key": {None: "ok"},
        "mixed_scalar_keys": {"s": 1, 2: "two", 3.5: "flt", None: "n"},
        "nested_valid_keys": {"outer": {1: {None: {2.5: "deep"}}}},
    }
    for label, payload in valid_key_maps.items():
        assert v._looks_secret_value(payload) is False, (
            f"[{label}] mapping with only JSON-coercible keys must remain retained "
            f"(not omitted)"
        )
        # Each is genuinely strict-JSON serializable (proves the control is real).
        assert isinstance(json.dumps(payload, allow_nan=False), str)

    # Retained through both entry points and strict-JSON / Starlette safe.
    prov = v._provenance_from_response_headers([
        ("x-hivemind-peer", {"a": 1, "b": 2}),
        ("x-hivemind-queue-path", {1: "x", None: "y"}),
    ])
    assert prov.get("peer") == {"a": 1, "b": 2}, f"valid-key map over-scrubbed: {prov!r}"
    assert prov.get("queue_path") == {1: "x", None: "y"}, f"valid-key map over-scrubbed: {prov!r}"
    assert isinstance(json.dumps(prov, allow_nan=False), str)

    san = v._sanitize_hli_provenance({
        "peer": {"a": 1, "b": 2},
        "queue_path": {1: "x", 3.5: "y"},
        "model": "xtts-v2",
    })
    assert san.get("peer") == {"a": 1, "b": 2}
    assert san.get("queue_path") == {1: "x", 3.5: "y"}
    assert san.get("model") == "xtts-v2"
    assert isinstance(json.dumps(san, allow_nan=False), str)

    try:
        from starlette.responses import JSONResponse
    except Exception:  # pragma: no cover - starlette optional in some envs
        JSONResponse = None
    if JSONResponse is not None:
        JSONResponse(prov)  # must not raise
        JSONResponse(san)   # must not raise


def test_r9_adjacent_invariants_unchanged_control():
    """GREEN control (must pass BEFORE and AFTER): the mapping-KEY hardening must
    NOT disturb any adjacent invariant -- list/tuple retention, set/frozenset
    VALUE omission, non-finite VALUE omission, cycle/DAG handling, bool-before-int,
    huge-int retention, the allowlist, secret detection (value AND secret-named
    str key), and the recursion budgets all behave exactly as before."""
    v = _r5_voice()

    # list/tuple VALUES remain JSON-native and retained.
    assert v._looks_secret_value(["hli-lobe-a", "hli-lobe-b"]) is False
    assert v._looks_secret_value(("route-1", "route-2")) is False
    assert v._looks_secret_value({"peer": ["a", "b"], "route": ("x", "y")}) is False

    # set/frozenset VALUES still fail closed by omission (R8 invariant preserved).
    assert v._looks_secret_value({"peer": {"lobe-a", "lobe-b"}}) is True
    assert v._looks_secret_value({"route": frozenset({"a", "b"})}) is True

    # Non-finite VALUE still fails closed (R7 invariant preserved).
    assert v._looks_secret_value({"timing_ms": float("nan")}) is True
    assert v._looks_secret_value(["ok", float("inf")]) is True

    # Cycles fail closed (list + dict).
    cyclic_list = ["ok"]
    cyclic_list.append(cyclic_list)
    assert v._looks_secret_value(cyclic_list) is True
    cyclic_dict = {"served_by": "ok"}
    cyclic_dict["self"] = cyclic_dict
    assert v._looks_secret_value(cyclic_dict) is True

    # DAG (shared, non-cyclic) JSON-native nodes retained (not misread as a cycle).
    shared = ["hli-lobe-a", "hli-lobe-b"]
    dag = {"primary": shared, "mirror": shared, "route": ("x", shared)}
    assert v._looks_secret_value(dag) is False, \
        "a shared DAG of JSON-native nodes must be retained, not misread as a cycle"

    # bool-before-int and huge-int retained (finite scalars usable; no OverflowError).
    for scalar in (None, True, False, 0, -7, 137, 10 ** 400, 0.0, -2.5, 1e308):
        assert v._looks_secret_value(scalar) is False, f"finite scalar wrongly flagged: {scalar!r}"

    # Allowlist unchanged: a non-allowlisted key is dropped by omission.
    assert v._sanitize_hli_provenance({"not_allowlisted": "x", "model": "xtts-v2"}) == {"model": "xtts-v2"}

    # Secret detection unchanged: a known synthetic secret still fails closed,
    # directly and inside a JSON-native container; a secret-NAMED str key too.
    secret = "sk-EXAMPLE0000NOTREALKEY"
    assert v._looks_secret_value(secret) is True
    assert v._looks_secret_value(["ok", secret]) is True
    assert v._looks_secret_value({"served_by": "ok", "x": secret}) is True
    assert v._looks_secret_value({"authorization": "ok"}) is True

    # Recursion budgets unchanged: an oversized JSON-native container fails closed.
    assert v._looks_secret_value(["ok"] * 5000) is True
    assert v._looks_secret_value({f"r{i}": "ok" for i in range(5000)}) is True
