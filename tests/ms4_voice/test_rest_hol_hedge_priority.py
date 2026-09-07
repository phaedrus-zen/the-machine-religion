"""Repair66 REST HOL hedge queue-priority regression."""

from __future__ import annotations

import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor as RealThreadPoolExecutor
from typing import Any

from machine_spirit_4.gateway import voice
from tests.ms4_voice._test_doubles import completed_chat_response


class _Runner:
    hivemind_url = "http://hive.invalid"
    ms3_url = "http://ms3.invalid"


def _audio(text: str) -> dict[str, Any]:
    return {
        "audio_bytes": f"WAV-{text}".encode(),
        "audio_base64": f"AUDIO-{text}",
        "content_type": "audio/wav",
    }


def _configure_rest(monkeypatch, *, capacity: int) -> None:
    monkeypatch.setenv("MS4_VOICE_TTS_CAPACITY", str(capacity))
    if capacity >= 2:
        proof = voice._voice_rest_concurrency_verdict(
            single_ms=2000,
            concurrent_wall_ms=2200,
            concurrent_requests=2,
            valid_audio_results=2,
            warmup_audio_valid=True,
            single_audio_valid=True,
            provenance_sample_count=4,
            provenance_non_cloud_count=4,
            location_policy="peer_only",
        )
        assert proof["passed"] is True
        assert voice._record_voice_rest_concurrency_observation(proof) == 2
    else:
        voice._record_voice_rest_capacity_observation({
            "ok": True,
            "replicas": capacity,
        })
    monkeypatch.setenv("MS4_TTS_REST_HOL_HEDGE_DELAY_S", "0")
    monkeypatch.setenv("MS4_TTS_REST_HOL_FATAL_TIMEOUT_S", "5")
    monkeypatch.setenv("MS4_VOICE_FIRST_CHUNK_DEADLINE_S", "0")
    monkeypatch.setenv("MS4_VOICE_SMART_REFLEX", "0")
    monkeypatch.setenv("MS4_VOICE_EGG_HONORABLE", "0")
    monkeypatch.setenv("MS4_VOICE_RUNTIME_QA", "0")
    monkeypatch.setattr(voice, "TTS_FILTER_ENABLED", False)
    monkeypatch.setattr(voice, "_runtime_audio_gate_enabled", lambda: False)
    monkeypatch.setattr(
        voice,
        "check_voice_ready",
        lambda *_args, **_kwargs: {"voice_input_ready": True},
    )
    monkeypatch.setattr(
        voice,
        "_identify_speaker_async",
        lambda *_args, **_kwargs: None,
    )


def _live_rest_threads(baseline: set[threading.Thread]) -> list[str]:
    return [
        thread.name
        for thread in threading.enumerate()
        if thread not in baseline
        and thread.is_alive()
        and (
            thread.name.startswith("ms4-tts")
            or thread.name.startswith("ms4-rest-hol")
            or thread.name == "ms4-rest-emit-coordinator"
        )
    ]


def test_rest_hol_hedge_preempts_queued_later_tail_futures(monkeypatch):
    """A c1 hedge must run ahead of queued c4+ work without adding capacity."""

    _configure_rest(monkeypatch, capacity=2)
    pieces = (
        "Chunk zero opens the ordered stream.",
        "Chunk one blocks on remote synthesis.",
        "Chunk two completes behind the head.",
        "Chunk three occupies the newly freed worker.",
        "Chunk four waits in the executor queue.",
        "Chunk five waits in the executor queue.",
        "Chunk six waits in the executor queue.",
        "Chunk seven waits in the executor queue.",
        "Chunk eight closes the ordered stream.",
    )
    reply = " ".join(pieces)
    baseline_threads = set(threading.enumerate())
    state_lock = threading.Lock()
    release_c0 = threading.Event()
    release_c2 = threading.Event()
    release_c3 = threading.Event()
    hedge_submitted = threading.Event()
    priority_started = threading.Event()
    chat_finished = threading.Event()
    all_tails_scheduled = threading.Event()
    turn_cancel = threading.Event()
    started = [threading.Event() for _ in pieces]
    attempts: dict[int, int] = defaultdict(int)
    start_order: list[tuple[int, int]] = []
    priority_order: list[tuple[int, int]] = []
    events: list[tuple[str, dict[str, Any]]] = []
    results: list[dict[str, Any]] = []
    failures: list[BaseException] = []
    active = 0
    max_active = 0
    c1_submissions = 0
    cancelled_before_tts: list[int] = []

    class _RecordingExecutor(RealThreadPoolExecutor):
        def submit(self, fn, /, *args, **kwargs):
            nonlocal c1_submissions
            future = super().submit(fn, *args, **kwargs)
            text = kwargs.get("text")
            if text in pieces[4:]:
                index = pieces.index(text)
                original_cancel = future.cancel

                def tracked_cancel():
                    cancelled = original_cancel()
                    if cancelled:
                        with state_lock:
                            cancelled_before_tts.append(index)
                    return cancelled

                future.cancel = tracked_cancel
            if kwargs.get("text") == pieces[1]:
                with state_lock:
                    c1_submissions += 1
                    if c1_submissions == 2:
                        hedge_submitted.set()
            return future

    monkeypatch.setattr(voice, "ThreadPoolExecutor", _RecordingExecutor)

    def synthesize(
        *,
        text,
        cancel_event=None,
        on_request_dispatched=None,
        **_kwargs,
    ):
        nonlocal active, max_active
        index = pieces.index(text)
        with state_lock:
            attempts[index] += 1
            attempt = attempts[index]
            token = (index, attempt)
            start_order.append(token)
            active += 1
            max_active = max(max_active, active)
            if token == (1, 2) or index >= 4:
                priority_order.append(token)
                priority_started.set()
        started[index].set()
        try:
            if index == 0:
                assert on_request_dispatched is not None
                on_request_dispatched()
                assert release_c0.wait(timeout=5), "c0 release timed out"
            elif index == 1 and attempt == 1:
                assert cancel_event is not None
                assert cancel_event.wait(timeout=5), "blocked c1 primary was not cancelled"
                raise voice.VoiceUnavailable("controlled blocked c1 primary cancelled")
            elif index == 2:
                assert release_c2.wait(timeout=5), "c2 release timed out"
            elif index == 3:
                assert release_c3.wait(timeout=5), "c3 release timed out"
            return _audio(text)
        finally:
            with state_lock:
                active -= 1

    monkeypatch.setattr(voice, "synthesize", synthesize)

    def emit(event, payload):
        with state_lock:
            events.append((event, dict(payload)))
        if event == "chunk_scheduled" and payload["index"] == len(pieces) - 1:
            all_tails_scheduled.set()
        return True

    def chat(_message, *, stream_callback=None, **_kwargs):
        assert stream_callback is not None
        assert stream_callback(reply)
        chat_finished.set()
        return completed_chat_response(reply, "repair66-hol-priority")

    def run_turn() -> None:
        try:
            results.append(
                voice.voice_ptt_turn_stream(
                    runner=_Runner(),
                    audio=b"FAKE_WAV",
                    chat_fn=chat,
                    transcribe_fn=lambda **_kwargs: {
                        "text": "reproduce REST HOL queue inversion",
                        "model": "whisper-1",
                    },
                    emit=emit,
                    chunker=voice.SentenceChunker(
                        first_chunk_min_words=6,
                        min_chunk_words=1,
                        max_chunk_words=20,
                    ),
                    tts_pool_size=6,
                    engine="rest",
                    cancel_event=turn_cancel,
                )
            )
        except BaseException as exc:
            failures.append(exc)

    turn = threading.Thread(target=run_turn, name="repair66-hol-priority-turn")
    turn.start()
    try:
        assert chat_finished.wait(timeout=5)
        assert started[0].wait(timeout=5)
        assert started[1].wait(timeout=5)
        assert not started[2].is_set(), "c2 must initially queue at capacity two"

        release_c0.set()
        assert started[2].wait(timeout=5)
        assert all_tails_scheduled.wait(timeout=5)
        assert not started[3].is_set(), "c3 must remain queued while c1/c2 run"

        release_c2.set()
        assert started[3].wait(timeout=5)
        assert hedge_submitted.wait(timeout=5)
        assert not any(event.is_set() for event in started[4:])

        release_c3.set()
        assert priority_started.wait(timeout=5)
        turn.join(timeout=5)
        assert not turn.is_alive()
    finally:
        release_c0.set()
        release_c2.set()
        release_c3.set()
        if turn.is_alive():
            turn_cancel.set()
            turn.join(timeout=5)

    assert failures == []
    assert len(results) == 1
    lifecycle = results[0]["metrics"]["lifecycle"]
    assert priority_order[0] == (1, 2), (
        "the c1 HOL hedge remained behind queued c4+ tails: "
        f"starts={start_order}"
    )
    assert max_active <= 2
    assert attempts[1] == 2
    assert all(attempts[index] == 1 for index in range(2, len(pieces)))
    assert cancelled_before_tts == list(range(4, len(pieces)))
    assert lifecycle["rest_hol_tail_futures_cancelled"] == len(pieces) - 4
    assert lifecycle["rest_hol_tail_requeues"] == len(pieces) - 4
    assert lifecycle["rest_hol_tail_requeue_failures"] == 0
    assert [
        payload["index"]
        for event, payload in events
        if event == "audio_chunk"
    ][:3] == [0, 1, 2]
    assert _live_rest_threads(baseline_threads) == []


def test_rest_capacity_one_cancellation_is_bounded_and_leak_free(monkeypatch):
    """Capacity one must cancel its queued tail without running or leaking it."""

    _configure_rest(monkeypatch, capacity=1)
    pieces = (
        "Chunk zero blocks until turn cancellation.",
        "Chunk one stays queued without synthesis.",
    )
    reply = " ".join(pieces)
    baseline_threads = set(threading.enumerate())
    executor_widths: list[int] = []
    started = [threading.Event() for _ in pieces]
    attempts: list[int] = []
    events: list[tuple[str, dict[str, Any]]] = []
    failures: list[BaseException] = []
    turn_cancel = threading.Event()
    chat_finished = threading.Event()
    state_lock = threading.Lock()
    active = 0
    max_active = 0

    class _CapacityExecutor(RealThreadPoolExecutor):
        def __init__(self, *, max_workers, **kwargs):
            executor_widths.append(max_workers)
            super().__init__(max_workers=max_workers, **kwargs)

    monkeypatch.setattr(voice, "ThreadPoolExecutor", _CapacityExecutor)

    def synthesize(
        *,
        text,
        cancel_event=None,
        on_request_dispatched=None,
        **_kwargs,
    ):
        nonlocal active, max_active
        index = pieces.index(text)
        with state_lock:
            attempts.append(index)
            active += 1
            max_active = max(max_active, active)
        started[index].set()
        try:
            if index == 0:
                assert on_request_dispatched is not None
                on_request_dispatched()
            assert cancel_event is not None
            assert cancel_event.wait(timeout=5)
            raise voice.VoiceUnavailable(f"controlled c{index} cancellation")
        finally:
            with state_lock:
                active -= 1

    monkeypatch.setattr(voice, "synthesize", synthesize)

    def chat(_message, *, stream_callback=None, **_kwargs):
        assert stream_callback is not None
        assert stream_callback(reply)
        chat_finished.set()
        return completed_chat_response(reply, "repair66-capacity-one")

    def run_turn() -> None:
        try:
            voice.voice_ptt_turn_stream(
                runner=_Runner(),
                audio=b"FAKE_WAV",
                chat_fn=chat,
                transcribe_fn=lambda **_kwargs: {
                    "text": "cancel a capacity one REST turn",
                    "model": "whisper-1",
                },
                emit=lambda event, payload: (
                    events.append((event, dict(payload))),
                    True,
                )[1],
                chunker=voice.SentenceChunker(
                    first_chunk_min_words=6,
                    min_chunk_words=1,
                    max_chunk_words=20,
                ),
                tts_pool_size=6,
                engine="rest",
                cancel_event=turn_cancel,
            )
        except BaseException as exc:
            failures.append(exc)

    turn = threading.Thread(target=run_turn, name="repair66-capacity-one-turn")
    turn.start()
    try:
        assert chat_finished.wait(timeout=5)
        assert started[0].wait(timeout=5)
        assert not started[1].is_set()
        turn_cancel.set()
        turn.join(timeout=5)
        assert not turn.is_alive()
    finally:
        turn_cancel.set()
        if turn.is_alive():
            turn.join(timeout=5)

    assert executor_widths == [1]
    assert attempts == [0]
    assert max_active <= 1
    assert len(failures) == 1
    assert isinstance(failures[0], voice.VoiceUnavailable)
    assert not any(event == "audio_chunk" for event, _payload in events)
    assert _live_rest_threads(baseline_threads) == []


def test_rest_hol_tail_requeue_rejection_fails_closed_after_remaining_attempts(
    monkeypatch,
):
    """A replacement submit failure is accounted and cannot strand tails."""

    _configure_rest(monkeypatch, capacity=2)
    pieces = (
        "Chunk zero opens the ordered stream.",
        "Chunk one blocks on remote synthesis.",
        "Chunk two completes behind the head.",
        "Chunk three holds the second worker.",
        "Chunk four needs restoration after cancellation.",
        "Chunk five needs restoration after cancellation.",
        "Chunk six needs restoration after cancellation.",
    )
    reply = " ".join(pieces)
    baseline_threads = set(threading.enumerate())
    state_lock = threading.Lock()
    release_c0 = threading.Event()
    release_c2 = threading.Event()
    requeue_rejected = threading.Event()
    chat_finished = threading.Event()
    all_tails_scheduled = threading.Event()
    turn_cancel = threading.Event()
    started = [threading.Event() for _ in pieces]
    submission_counts: dict[int, int] = defaultdict(int)
    attempts: dict[int, int] = defaultdict(int)
    events: list[tuple[str, dict[str, Any]]] = []
    failures: list[BaseException] = []
    active = 0
    max_active = 0

    class _RejectFirstRequeueExecutor(RealThreadPoolExecutor):
        def submit(self, fn, /, *args, **kwargs):
            text = kwargs.get("text")
            index = pieces.index(text)
            with state_lock:
                submission_counts[index] += 1
                submission = submission_counts[index]
            if index == 4 and submission == 2:
                requeue_rejected.set()
                raise RuntimeError("controlled Repair66 tail requeue rejection")
            return super().submit(fn, *args, **kwargs)

    monkeypatch.setattr(voice, "ThreadPoolExecutor", _RejectFirstRequeueExecutor)

    def synthesize(
        *,
        text,
        cancel_event=None,
        on_request_dispatched=None,
        **_kwargs,
    ):
        nonlocal active, max_active
        index = pieces.index(text)
        with state_lock:
            attempts[index] += 1
            attempt = attempts[index]
            active += 1
            max_active = max(max_active, active)
        started[index].set()
        try:
            if index == 0:
                assert on_request_dispatched is not None
                on_request_dispatched()
                assert release_c0.wait(timeout=5)
            elif index == 1 and attempt == 1:
                assert cancel_event is not None
                assert cancel_event.wait(timeout=5)
                raise voice.VoiceUnavailable("controlled blocked c1 cancellation")
            elif index == 2:
                assert release_c2.wait(timeout=5)
            elif index == 3:
                assert cancel_event is not None
                assert cancel_event.wait(timeout=5)
                raise voice.VoiceUnavailable("controlled running c3 cancellation")
            return _audio(text)
        finally:
            with state_lock:
                active -= 1

    monkeypatch.setattr(voice, "synthesize", synthesize)

    def emit(event, payload):
        with state_lock:
            events.append((event, dict(payload)))
        if event == "chunk_scheduled" and payload["index"] == len(pieces) - 1:
            all_tails_scheduled.set()
        return True

    def chat(_message, *, stream_callback=None, **_kwargs):
        assert stream_callback is not None
        assert stream_callback(reply)
        chat_finished.set()
        return completed_chat_response(reply, "repair66-requeue-rejection")

    def run_turn() -> None:
        try:
            voice.voice_ptt_turn_stream(
                runner=_Runner(),
                audio=b"FAKE_WAV",
                chat_fn=chat,
                transcribe_fn=lambda **_kwargs: {
                    "text": "reject one REST tail restoration",
                    "model": "whisper-1",
                },
                emit=emit,
                chunker=voice.SentenceChunker(
                    first_chunk_min_words=6,
                    min_chunk_words=1,
                    max_chunk_words=20,
                ),
                tts_pool_size=6,
                engine="rest",
                cancel_event=turn_cancel,
            )
        except BaseException as exc:
            failures.append(exc)

    turn = threading.Thread(target=run_turn, name="repair66-requeue-rejection-turn")
    turn.start()
    try:
        assert chat_finished.wait(timeout=5)
        assert started[0].wait(timeout=5)
        assert started[1].wait(timeout=5)
        release_c0.set()
        assert started[2].wait(timeout=5)
        assert all_tails_scheduled.wait(timeout=5)
        release_c2.set()
        assert started[3].wait(timeout=5)
        assert requeue_rejected.wait(timeout=5)
        turn.join(timeout=5)
        assert not turn.is_alive()
    finally:
        release_c0.set()
        release_c2.set()
        if turn.is_alive():
            turn_cancel.set()
            turn.join(timeout=5)

    assert turn_cancel.is_set()
    assert len(failures) == 1
    assert isinstance(failures[0], voice.VoiceUnavailable)
    assert "tail restoration failed" in str(failures[0])
    assert "4" in str(failures[0])
    assert [submission_counts[index] for index in range(4, 7)] == [2, 2, 2]
    assert all(attempts[index] == 0 for index in range(4, 7))
    assert attempts[1] == 1
    assert max_active <= 2
    assert not any(
        event == "audio_chunk" and payload["index"] >= 1
        for event, payload in events
    )
    assert _live_rest_threads(baseline_threads) == []
