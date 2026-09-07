"""Fail-first contracts for the physical Oracle REST-TTS failure.

These tests keep the production invariants narrow: two single-flight TTS
replicas, XTTS-safe fragments, ordered delivery, bounded busy retry, and
turn-scoped cancellation.
"""

from __future__ import annotations

import re
import random
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

import pytest

import machine_spirit_4.gateway.voice as voice
from tests.ms4_voice._test_doubles import completed_chat_response


class _Runner:
    hivemind_url = "http://hive.invalid"
    ms3_url = "http://ms3.invalid"


def _configure_rest(monkeypatch) -> None:
    monkeypatch.setattr(
        voice,
        "check_voice_ready",
        lambda *_args, **_kwargs: {"voice_input_ready": True},
    )
    monkeypatch.setattr(voice, "TTS_FILTER_ENABLED", False)
    monkeypatch.setattr(voice, "_runtime_audio_gate_enabled", lambda: False)
    monkeypatch.setattr(voice, "_identify_speaker_async", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("MS4_VOICE_SMART_REFLEX", "0")
    monkeypatch.setenv("MS4_VOICE_EGG_HONORABLE", "0")
    monkeypatch.setenv("MS4_VOICE_RUNTIME_QA", "0")
    monkeypatch.setenv("MS4_VOICE_FIRST_CHUNK_DEADLINE_S", "0")
    monkeypatch.setenv("MS4_TTS_REST_BUSY_RETRY_DELAY_S", "0")


def _audio(text: str) -> dict[str, Any]:
    return {
        "audio_bytes": f"WAV-{text}".encode(),
        "audio_base64": f"AUDIO-{text}",
        "content_type": "audio/wav",
    }


def _run_rest_turn(
    monkeypatch,
    *,
    pieces: tuple[str, ...],
    synthesize: Callable[..., dict[str, Any]],
    production_transport: bool = False,
    tts_pool_size: int = 6,
    cancel_event: threading.Event | None = None,
    events: list[tuple[str, dict[str, Any]]] | None = None,
) -> tuple[dict[str, Any], list[tuple[str, dict[str, Any]]]]:
    _configure_rest(monkeypatch)
    events = [] if events is None else events
    reply = " ".join(pieces)

    def chat(_message, *, stream_callback=None, **_kwargs):
        assert stream_callback is not None
        accepted = stream_callback(reply + " ")
        assert accepted is not False or (
            cancel_event is not None and cancel_event.is_set()
        )
        return completed_chat_response(reply, "capacity-token-contract")

    kwargs: dict[str, Any] = {}
    if production_transport:
        monkeypatch.setattr(voice, "synthesize", synthesize)
    else:
        kwargs["synthesize_fn"] = synthesize

    result = voice.voice_ptt_turn_stream(
        runner=_Runner(),
        audio=b"FAKE_WAV",
        chat_fn=chat,
        transcribe_fn=lambda **_kwargs: {
            "text": "HiveMind Oracle say ready",
            "model": "whisper-1",
        },
        emit=lambda event, payload: events.append((event, dict(payload))) or True,
        chunker=voice.SentenceChunker(
            first_chunk_min_words=5,
            min_chunk_words=1,
            max_chunk_words=20,
        ),
        tts_pool_size=tts_pool_size,
        engine="rest",
        cancel_event=cancel_event,
        **kwargs,
    )
    return result, events


def test_six_way_request_is_bounded_to_two_single_lock_replicas(monkeypatch):
    """Causal RED: requested pool=6 must never produce six in-flight calls."""

    pieces = (
        "Chunk zero has five words.",
        "Chunk one has five words.",
        "Chunk two has five words.",
        "Chunk three has five words.",
        "Chunk four has five words.",
        "Chunk five has five words.",
        "Chunk six has five words.",
    )
    lock = threading.Lock()
    active = 0
    max_active = 0

    def synthesize(*, text, on_request_dispatched=None, **_kwargs):
        nonlocal active, max_active
        if on_request_dispatched is not None:
            on_request_dispatched()
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            if text != pieces[0]:
                time.sleep(0.08)
            return _audio(text)
        finally:
            with lock:
                active -= 1

    result, events = _run_rest_turn(
        monkeypatch,
        pieces=pieces,
        synthesize=synthesize,
        production_transport=True,
        tts_pool_size=6,
    )

    assert max_active <= 2, f"observed {max_active}-way fanout against two replicas"
    assert result["metrics"]["audio_chunks"] == len(pieces)
    assert [payload["index"] for event, payload in events if event == "audio_chunk"] == list(
        range(len(pieces))
    )


def _tokenish_count(text: str) -> int:
    """Test-side lower bound: words/numbers and each punctuation mark."""

    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def _native_xtts_token_count(text: str) -> int:
    """Count exactly with the deployed XTTS-v2 tokenizer vocabulary.

    Mirror the production English preparation before delegating to the same
    Hugging Face vocabulary: normalized whitespace, lowercase, ``[en]`` prefix,
    and XTTS's explicit ``[SPACE]`` marker.
    """

    tokenizers = pytest.importorskip("tokenizers")
    candidates = [
        Path(os.environ["MS4_VOICE_XTTS_VOCAB"])
        for _ in (0,)
        if os.environ.get("MS4_VOICE_XTTS_VOCAB")
    ]
    if model_repo := os.environ.get("HIVEMIND_MODEL_REPO"):
        candidates.append(
            Path(model_repo)
            / "tts"
            / "tts_models--multilingual--multi-dataset--xtts_v2"
            / "vocab.json"
        )
    candidates.append(
        Path(r"C:\HiveMind\model_repo\tts")
        / "tts_models--multilingual--multi-dataset--xtts_v2"
        / "vocab.json"
    )
    vocab = next((candidate for candidate in candidates if candidate.is_file()), None)
    if vocab is None:
        pytest.skip("deployed XTTS-v2 vocab.json is unavailable")
    tokenizer = tokenizers.Tokenizer.from_file(str(vocab))
    cleaned = " ".join((text or "").split()).lower()
    prepared = f"[en]{cleaned}".replace(" ", "[SPACE]")
    return len(tokenizer.encode(prepared).ids)


@pytest.mark.parametrize(
    "adversarial",
    [
        "qzxjkv" * 240,
        "".join(
            random.Random(9917).choices(
                "{}[](),:;=/\\|_+-!?@#$%^&*abcdefghijklmnopqrstuvwxyz0123456789",
                k=1200,
            )
        ),
        (
            "HiveMind Oracle keeps each spoken fragment ordered while bounded "
            "capacity and cancellation protect the live listener. "
            * 50
        )[:5000],
    ],
    ids=("long_identifier", "structured_ascii", "representative_prose"),
)
def test_every_fallback_fragment_is_below_native_xtts_token_limit(
    adversarial,
    monkeypatch,
):
    """Host proof: no-dependency fallback stays below deployed XTTS BPE."""

    monkeypatch.setattr(voice, "_native_xtts_tokenizer", lambda: None)
    pieces = voice._split_xtts_safe_fragments(adversarial)

    assert "".join(pieces) == adversarial
    native_counts = [_native_xtts_token_count(piece) for piece in pieces]
    assert native_counts
    assert max(native_counts) < voice.XTTS_HARD_TOKEN_LIMIT, native_counts


def test_missing_native_tokenizer_uses_bounded_noncatastrophic_fallback(monkeypatch):
    """Runtime RED: no-tokenizers fallback must be safe without char fan-out."""

    monkeypatch.setattr(voice, "_native_xtts_tokenizer", lambda: None)
    fixtures = {
        "structured719": ('{"a":1},' * 90)[:719],
        "identifier1200": "qzxjkv" * 200,
        "prose5000": (
            "HiveMind Oracle keeps each spoken fragment ordered while bounded "
            "capacity and cancellation protect the live listener. "
            * 50
        )[:5000],
    }

    started = time.perf_counter()
    split = {
        name: voice._split_xtts_safe_fragments(text)
        for name, text in fixtures.items()
    }
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    for name, text in fixtures.items():
        assert "".join(split[name]) == text
        assert split[name]
        assert max(map(len, split[name])) >= 32, (
            name,
            [len(piece) for piece in split[name]][:20],
        )
    assert 2 <= len(split["structured719"]) <= 12
    assert 2 <= len(split["identifier1200"]) <= 16
    assert 2 <= len(split["prose5000"]) <= 96
    assert sum(map(len, split.values())) <= 124
    assert elapsed_ms < 250.0, elapsed_ms


def test_719_character_structured_tail_is_split_below_xtts_token_limit():
    """Causal RED: flush must not send one >400-token structured fragment."""

    structured = ('{"a":1},' * 90)[:719]
    assert len(structured) == 719
    assert _tokenish_count(structured) > 400

    chunker = voice.SentenceChunker(
        first_chunk_min_words=10_000,
        min_chunk_words=10_000,
        max_chunk_words=10_000,
    )
    pieces = chunker.add(structured)
    tail = chunker.flush()
    if tail:
        pieces.append(tail)

    assert len(pieces) > 1
    assert "".join(pieces) == structured
    assert all(_tokenish_count(piece) < 400 for piece in pieces)


def test_transient_429_retries_once_and_preserves_strict_audio_order(monkeypatch):
    """Causal RED: a busy head is retried, never skipped past by later audio."""

    pieces = (
        "Chunk zero reaches the listener.",
        "Chunk one is transiently busy.",
        "Chunk two waits behind one.",
        "Chunk three remains strictly ordered.",
    )
    attempts: dict[str, int] = {}

    def synthesize(*, text, on_request_dispatched=None, **_kwargs):
        if on_request_dispatched is not None:
            on_request_dispatched()
        attempts[text] = attempts.get(text, 0) + 1
        if text == pieces[1] and attempts[text] == 1:
            raise voice.VoiceUnavailable(
                "HiveMind /v1/audio/speech returned 429: Model busy"
            )
        return _audio(text)

    result, events = _run_rest_turn(
        monkeypatch,
        pieces=pieces,
        synthesize=synthesize,
        tts_pool_size=2,
    )

    assert attempts[pieces[1]] == 2
    assert not any(event == "audio_error" for event, _payload in events)
    assert [payload["index"] for event, payload in events if event == "audio_chunk"] == [
        0,
        1,
        2,
        3,
    ]
    assert result["metrics"]["audio_chunks"] == 4


def test_terminal_earlier_failure_cancels_tail_instead_of_emitting_later_audio(monkeypatch):
    """Causal RED: terminal c1 failure must not emit decoded c2/c3 out of order."""

    pieces = (
        "Chunk zero reaches the listener.",
        "Chunk one fails terminal synthesis.",
        "Chunk two must be drained.",
        "Chunk three must be drained.",
    )
    turn_cancel = threading.Event()
    events: list[tuple[str, dict[str, Any]]] = []

    def synthesize(*, text, on_request_dispatched=None, **_kwargs):
        if on_request_dispatched is not None:
            on_request_dispatched()
        if text == pieces[1]:
            raise voice.VoiceUnavailable(
                "HiveMind /v1/audio/speech returned 500: XTTS token limit"
            )
        if text in pieces[2:]:
            time.sleep(0.02)
        return _audio(text)

    with pytest.raises(voice.VoiceUnavailable, match="chunk index 1"):
        _run_rest_turn(
            monkeypatch,
            pieces=pieces,
            synthesize=synthesize,
            tts_pool_size=2,
            cancel_event=turn_cancel,
            events=events,
        )

    assert turn_cancel.is_set()
    assert [payload["index"] for event, payload in events if event == "audio_chunk"] == [0]


def test_barge_in_cancels_running_synthesis_without_tail_audio(monkeypatch):
    """Preservation guard: the capacity gate remains turn-cancel aware."""

    pieces = (
        "Chunk zero blocks for cancellation.",
        "Chunk one must never play.",
    )
    started = threading.Event()
    turn_cancel = threading.Event()
    events: list[tuple[str, dict[str, Any]]] = []
    failures: list[BaseException] = []

    def synthesize(*, cancel_event, on_request_dispatched=None, **_kwargs):
        if on_request_dispatched is not None:
            on_request_dispatched()
        started.set()
        assert cancel_event.wait(timeout=2.0)
        raise voice.VoiceUnavailable("cancelled by barge-in")

    def run() -> None:
        nonlocal events
        try:
            _result, events = _run_rest_turn(
                monkeypatch,
                pieces=pieces,
                synthesize=synthesize,
                production_transport=True,
                tts_pool_size=6,
                cancel_event=turn_cancel,
            )
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=run, name="rest-capacity-barge-guard")
    thread.start()
    assert started.wait(timeout=2.0)
    turn_cancel.set()
    thread.join(timeout=3.0)

    assert not thread.is_alive()
    assert len(failures) == 1
    assert isinstance(failures[0], voice.VoiceUnavailable)
    assert "cancel" in str(failures[0]).lower()
    assert not any(event == "audio_chunk" for event, _payload in events)


def test_first_audio_is_emitted_before_slow_tail_finishes(monkeypatch):
    """Preservation guard: capacity/backpressure must not delay the lead chunk."""

    pieces = (
        "Chunk zero starts audio immediately.",
        "Chunk one is deliberately slower.",
        "Chunk two is deliberately slower.",
    )
    first_emitted_at: list[float] = []
    tail_finished_at: list[float] = []

    def synthesize(*, text, on_request_dispatched=None, **_kwargs):
        if on_request_dispatched is not None:
            on_request_dispatched()
        if text != pieces[0]:
            time.sleep(0.08)
            tail_finished_at.append(time.monotonic())
        return _audio(text)

    _configure_rest(monkeypatch)
    reply = " ".join(pieces)
    events: list[tuple[str, dict[str, Any]]] = []

    def emit(event, payload):
        events.append((event, dict(payload)))
        if event == "audio_chunk" and payload["index"] == 0:
            first_emitted_at.append(time.monotonic())
        return True

    monkeypatch.setattr(voice, "synthesize", synthesize)
    voice.voice_ptt_turn_stream(
        runner=_Runner(),
        audio=b"FAKE_WAV",
        chat_fn=lambda _message, stream_callback=None, **_kwargs: (
            stream_callback(reply + " "),
            completed_chat_response(reply, "first-audio-guard"),
        )[1],
        transcribe_fn=lambda **_kwargs: {"text": "say ready", "model": "whisper-1"},
        emit=emit,
        chunker=voice.SentenceChunker(
            first_chunk_min_words=5,
            min_chunk_words=1,
            max_chunk_words=20,
        ),
        tts_pool_size=6,
        engine="rest",
    )

    assert first_emitted_at
    assert tail_finished_at
    assert first_emitted_at[0] < min(tail_finished_at)
