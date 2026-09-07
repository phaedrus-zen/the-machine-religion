"""CAP50 capacity-two inter-fragment continuation regression."""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import Future
from typing import Any

from machine_spirit_4.gateway import voice
from tests.ms4_voice._test_doubles import completed_chat_response


class _Runner:
    hivemind_url = "http://hive.invalid"
    ms3_url = "http://ms3.invalid"


def _admit_measured_capacity_two() -> None:
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


def test_capacity_two_queues_c2_before_first_audio_delivery(monkeypatch):
    """Queue one continuation before c0/c1 SSE writes can consume its runway."""

    monkeypatch.setenv("MS4_VOICE_TTS_CAPACITY", "2")
    _admit_measured_capacity_two()
    monkeypatch.setenv("MS4_VOICE_FIRST_CHUNK_DEADLINE_S", "0")
    monkeypatch.setenv("MS4_VOICE_SMART_REFLEX", "0")
    monkeypatch.setenv("MS4_VOICE_EGG_HONORABLE", "0")
    monkeypatch.setenv("MS4_VOICE_RUNTIME_QA", "0")
    monkeypatch.setattr(voice, "TTS_FILTER_ENABLED", False)
    monkeypatch.setattr(
        voice,
        "check_voice_ready",
        lambda *_args, **_kwargs: {"voice_input_ready": True},
    )
    monkeypatch.setattr(voice, "_identify_speaker_async", lambda *_args, **_kwargs: None)

    pieces = (
        "Chunk zero carries the opening.",
        "Chunk one extends playback runway.",
        "Chunk two must already continue.",
    )
    reply = " ".join(pieces)
    lock = threading.Lock()
    chat_finished = threading.Event()
    dispatch_callback_returned = threading.Event()
    cancel_event = threading.Event()
    submissions: list[tuple[int, Any, tuple[Any, ...], dict[str, Any], Future]] = []
    executor_widths: list[int] = []
    events: list[tuple[str, dict[str, Any]]] = []
    turn_errors: list[BaseException] = []

    class _ControlledExecutor:
        def __init__(self, *, max_workers, **_kwargs):
            executor_widths.append(max_workers)
            self._threads: set[threading.Thread] = set()

        def submit(self, fn, *args, **kwargs):
            index = pieces.index(kwargs["text"])
            future = Future()
            with lock:
                submissions.append((index, fn, args, kwargs, future))
            return future

        def shutdown(self, wait=True, *, cancel_futures=False):
            if cancel_futures:
                with lock:
                    futures = [entry[-1] for entry in submissions]
                for future in futures:
                    future.cancel()

    monkeypatch.setattr(voice, "ThreadPoolExecutor", _ControlledExecutor)

    def speech_request(*, body, cancel_event: threading.Event, on_request_dispatched=None, **_kwargs):
        text = json.loads(body.decode("utf-8"))["input"]
        index = pieces.index(text)
        assert index == 0, "queued continuation must not execute in admission capture"
        assert on_request_dispatched is not None
        on_request_dispatched()
        dispatch_callback_returned.set()
        assert cancel_event.wait(timeout=5)
        raise voice.VoiceUnavailable("controlled CAP50 admission capture cancelled")

    monkeypatch.setattr(voice, "_cancellable_speech_request", speech_request)

    def run_turn() -> None:
        try:
            voice.voice_ptt_turn_stream(
                runner=_Runner(),
                audio=b"FAKE_WAV",
                chat_fn=lambda _message, stream_callback=None, **_kwargs: (
                    stream_callback(reply),
                    chat_finished.set(),
                    completed_chat_response(reply, "cap50-continuation"),
                )[2],
                transcribe_fn=lambda **_kwargs: {
                    "text": "reproduce CAP50 continuation",
                    "model": "whisper-1",
                },
                emit=lambda event, payload: (
                    events.append((event, dict(payload))),
                    True,
                )[1],
                chunker=voice.SentenceChunker(
                    first_chunk_min_words=5,
                    min_chunk_words=1,
                    max_chunk_words=20,
                ),
                tts_pool_size=6,
                engine="rest",
                cancel_event=cancel_event,
            )
        except BaseException as exc:
            turn_errors.append(exc)

    def complete_first(entry) -> None:
        _index, fn, args, kwargs, future = entry
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:
            future.set_exception(exc)

    turn = threading.Thread(target=run_turn, name="cap50-continuation-turn")
    turn.start()
    assert chat_finished.wait(timeout=5)
    with lock:
        initial = list(submissions)
    assert [entry[0] for entry in initial] == [0]

    first_worker = threading.Thread(
        target=complete_first,
        args=(initial[0],),
        name="cap50-continuation-c0",
    )
    first_worker.start()
    assert dispatch_callback_returned.wait(timeout=5)
    with lock:
        before_first_delivery = [entry[0] for entry in submissions]

    cancel_event.set()
    first_worker.join(timeout=5)
    turn.join(timeout=5)
    assert not first_worker.is_alive()
    assert not turn.is_alive()

    # CAP50 decoded 1.755125s + 1.429333s of runway, then observed one
    # 1.057542s gap because c2 was admitted only after those SSE writes. The
    # executor remains capacity two; one additional future is queued, not run.
    assert round(sum((1.755125, 1.429333)), 6) == 3.184458
    assert executor_widths == [2]
    assert before_first_delivery == [0, 1, 2], (
        "capacity two must admit one queued c2 continuation before c0 delivery; "
        f"CAP50 late-c2 underflow path admitted {before_first_delivery}"
    )
    assert [payload["index"] for event, payload in events if event == "chunk_scheduled"] == [0, 1]
    assert not any(event == "audio_chunk" for event, _payload in events)
    assert len(turn_errors) == 1
    assert isinstance(turn_errors[0], voice.VoiceUnavailable)


def _audio(text: str) -> dict[str, Any]:
    return {
        "audio_bytes": f"WAV-{text}".encode(),
        "audio_base64": f"AUDIO-{text}",
        "content_type": "audio/wav",
    }


def _configure_real_executor(monkeypatch) -> None:
    monkeypatch.setenv("MS4_VOICE_TTS_CAPACITY", "2")
    _admit_measured_capacity_two()
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
    monkeypatch.setattr(voice, "_identify_speaker_async", lambda *_args, **_kwargs: None)


def test_queued_c2_starts_when_tts_slot_frees_before_first_audio_write_returns(
    monkeypatch,
):
    """A freed backend slot must not remain idle behind c0's SSE write."""

    _configure_real_executor(monkeypatch)
    pieces = (
        "Chunk zero carries the opening.",
        "Chunk one extends playback runway.",
        "Chunk two must already continue.",
    )
    reply = " ".join(pieces)
    started = [threading.Event() for _ in pieces]
    completed = [threading.Event() for _ in pieces]
    release = [threading.Event() for _ in pieces]
    first_audio_write_entered = threading.Event()
    release_first_audio_write = threading.Event()
    lock = threading.Lock()
    active = 0
    max_active = 0
    events: list[tuple[str, dict[str, Any]]] = []
    results: list[dict[str, Any]] = []
    failures: list[BaseException] = []

    def synthesize(*, text, on_request_dispatched=None, **_kwargs):
        nonlocal active, max_active
        index = pieces.index(text)
        if index == 0:
            assert on_request_dispatched is not None
            on_request_dispatched()
        with lock:
            active += 1
            max_active = max(max_active, active)
        started[index].set()
        try:
            assert release[index].wait(timeout=5), f"release timeout for c{index}"
            return _audio(text)
        finally:
            with lock:
                active -= 1
            completed[index].set()

    monkeypatch.setattr(voice, "synthesize", synthesize)

    def emit(event, payload):
        events.append((event, dict(payload)))
        if event == "audio_chunk" and payload["index"] == 0:
            first_audio_write_entered.set()
            assert release_first_audio_write.wait(timeout=5)
        return True

    def run_turn() -> None:
        try:
            results.append(
                voice.voice_ptt_turn_stream(
                    runner=_Runner(),
                    audio=b"FAKE_WAV",
                    chat_fn=lambda _message, stream_callback=None, **_kwargs: (
                        stream_callback(reply),
                        completed_chat_response(reply, "repair56-real-executor"),
                    )[1],
                    transcribe_fn=lambda **_kwargs: {
                        "text": "verify CAP50 continuation",
                        "model": "whisper-1",
                    },
                    emit=emit,
                    chunker=voice.SentenceChunker(
                        first_chunk_min_words=5,
                        min_chunk_words=1,
                        max_chunk_words=20,
                    ),
                    tts_pool_size=6,
                    engine="rest",
                )
            )
        except BaseException as exc:
            failures.append(exc)

    turn = threading.Thread(target=run_turn, name="repair56-real-executor-turn")
    turn.start()
    assert started[0].wait(timeout=5)
    assert started[1].wait(timeout=5)
    assert not started[2].is_set(), "c2 must initially be queued behind capacity two"

    release[0].set()
    assert first_audio_write_entered.wait(timeout=5)
    assert completed[0].is_set(), "c0 synthesis must have returned before its audio write"
    c2_started_before_first_audio_returned = started[2].wait(timeout=0.5)

    release_first_audio_write.set()
    release[1].set()
    assert started[2].wait(timeout=5)
    release[2].set()
    turn.join(timeout=5)

    assert not turn.is_alive()
    assert failures == []
    assert len(results) == 1
    assert max_active <= 2
    assert c2_started_before_first_audio_returned, (
        "c2 stayed queued after c0 synthesis returned because the executor worker "
        "ran the c0 done-callback/audio write before dequeuing c2"
    )
    assert [payload["index"] for event, payload in events if event == "audio_chunk"] == [
        0,
        1,
        2,
    ]


def test_capacity_one_cancellation_cancels_queued_tail_without_audio(monkeypatch):
    """At capacity one, a queued continuation must remain cancel-safe."""

    _configure_real_executor(monkeypatch)
    monkeypatch.setenv("MS4_VOICE_TTS_CAPACITY", "1")
    voice._record_voice_rest_capacity_observation({"ok": True, "replicas": 1})
    pieces = (
        "Chunk zero blocks until cancellation.",
        "Chunk one must remain queued.",
    )
    reply = " ".join(pieces)
    cancel = threading.Event()
    started = [threading.Event() for _ in pieces]
    events: list[tuple[str, dict[str, Any]]] = []
    failures: list[BaseException] = []

    def synthesize(*, text, on_request_dispatched=None, cancel_event=None, **_kwargs):
        index = pieces.index(text)
        if index == 0:
            assert on_request_dispatched is not None
            on_request_dispatched()
        started[index].set()
        assert cancel_event is not None
        assert cancel_event.wait(timeout=5)
        raise voice.VoiceUnavailable(f"c{index} cancelled")

    monkeypatch.setattr(voice, "synthesize", synthesize)

    def run_turn() -> None:
        try:
            voice.voice_ptt_turn_stream(
                runner=_Runner(),
                audio=b"FAKE_WAV",
                chat_fn=lambda _message, stream_callback=None, **_kwargs: (
                    stream_callback(reply),
                    completed_chat_response(reply, "repair56-capacity-one"),
                )[1],
                transcribe_fn=lambda **_kwargs: {
                    "text": "cancel queued continuation",
                    "model": "whisper-1",
                },
                emit=lambda event, payload: events.append((event, dict(payload))) or True,
                chunker=voice.SentenceChunker(
                    first_chunk_min_words=5,
                    min_chunk_words=1,
                    max_chunk_words=20,
                ),
                tts_pool_size=6,
                engine="rest",
                cancel_event=cancel,
            )
        except BaseException as exc:
            failures.append(exc)

    turn = threading.Thread(target=run_turn, name="repair56-capacity-one-turn")
    turn.start()
    assert started[0].wait(timeout=5)
    time.sleep(0.1)
    assert not started[1].is_set()
    cancel.set()
    turn.join(timeout=5)

    assert not turn.is_alive()
    assert len(failures) == 1
    assert isinstance(failures[0], voice.VoiceUnavailable)
    assert not started[1].is_set()
    assert not any(event == "audio_chunk" for event, _payload in events)


def test_post_chat_tail_schedule_exception_releases_coordinator_and_tts_workers(
    monkeypatch,
):
    """A post-chat client exception must retire every turn-owned thread."""

    _configure_real_executor(monkeypatch)
    opening = "Chunk zero starts the synthesis worker."
    tail = "Tail scheduling fails after chat returns"
    reply = f"{opening} {tail}"
    worker_started = threading.Event()
    worker_released = threading.Event()
    chat_completed = threading.Event()
    cancel = threading.Event()
    sentinel = RuntimeError("repair59 post-chat schedule sentinel")
    failure_after_chat: list[bool] = []
    events: list[tuple[str, dict[str, Any]]] = []
    executors: list[Any] = []
    baseline_threads = set(threading.enumerate())
    real_executor = voice.ThreadPoolExecutor

    def tracking_executor(*args, **kwargs):
        executor = real_executor(*args, **kwargs)
        executors.append(executor)
        return executor

    monkeypatch.setattr(voice, "ThreadPoolExecutor", tracking_executor)

    def synthesize(*, text, on_request_dispatched=None, cancel_event=None, **_kwargs):
        assert text == opening
        assert on_request_dispatched is not None
        assert cancel_event is not None
        on_request_dispatched()
        worker_started.set()
        try:
            assert cancel_event.wait(timeout=5)
        finally:
            worker_released.set()
        raise voice.VoiceUnavailable("controlled Repair59 worker cancellation")

    monkeypatch.setattr(voice, "synthesize", synthesize)

    def chat(_message, *, stream_callback=None, **_kwargs):
        assert stream_callback is not None
        assert stream_callback(f"{opening} ")
        assert worker_started.wait(timeout=5)
        assert stream_callback(tail)
        chat_completed.set()
        return completed_chat_response(reply, "repair59-post-chat-cleanup")

    def emit(event, payload):
        events.append((event, dict(payload)))
        if event == "chunk_scheduled" and payload["index"] == 1:
            failure_after_chat.append(chat_completed.is_set())
            raise sentinel
        return True

    def live_turn_threads() -> list[threading.Thread]:
        return [
            thread
            for thread in threading.enumerate()
            if thread not in baseline_threads
            and thread.is_alive()
            and (
                thread.name.startswith("ms4-tts")
                or thread.name == "ms4-rest-emit-coordinator"
            )
        ]

    raised: BaseException | None = None
    try:
        try:
            voice.voice_ptt_turn_stream(
                runner=_Runner(),
                audio=b"FAKE_WAV",
                chat_fn=chat,
                transcribe_fn=lambda **_kwargs: {
                    "text": "force one post-chat tail scheduling failure",
                    "model": "whisper-1",
                },
                emit=emit,
                chunker=voice.SentenceChunker(
                    first_chunk_min_words=10,
                    min_chunk_words=1,
                    max_chunk_words=20,
                ),
                tts_pool_size=6,
                engine="rest",
                cancel_event=cancel,
            )
        except BaseException as exc:
            raised = exc

        assert raised is sentinel, "cleanup must preserve the post-chat client exception"
        assert failure_after_chat == [True]
        assert cancel.is_set(), "the post-chat scheduling exception must fail the turn closed"
        assert worker_released.wait(timeout=5)
        assert not any(event == "audio_chunk" for event, _payload in events)
        assert live_turn_threads() == [], (
            "post-chat exception leaked turn-owned threads: "
            f"{[thread.name for thread in live_turn_threads()]}"
        )
    finally:
        # RED teardown only: a broken product cannot be allowed to strand pytest.
        coordinator_threads = [
            thread
            for thread in live_turn_threads()
            if thread.name == "ms4-rest-emit-coordinator"
        ]
        for thread in coordinator_threads:
            target = getattr(thread, "_target", None)
            if target is None or target.__closure__ is None:
                continue
            closed = {
                name: cell.cell_contents
                for name, cell in zip(target.__code__.co_freevars, target.__closure__)
            }
            closed["emit_coordinator_stop"].set()
            closed["emit_coordinator_wake"].set()
        for executor in executors:
            executor.shutdown(wait=False, cancel_futures=True)
        for thread in coordinator_threads:
            thread.join(timeout=5)
        for executor in executors:
            executor.shutdown(wait=True, cancel_futures=True)


def test_cleanup_close_failure_preserves_primary_and_runs_later_cleanup(
    monkeypatch,
    caplog,
):
    """A cleanup failure must not mask the active post-chat failure or skip phases."""

    _configure_real_executor(monkeypatch)
    opening = "Chunk zero starts the real synthesis worker."
    tail = "Tail scheduling raises the primary sentinel"
    reply = f"{opening} {tail}"
    primary = RuntimeError("repair63 post-chat primary sentinel")
    cleanup_failure = RuntimeError("repair63 timer close cleanup sentinel")
    worker_started = threading.Event()
    close_entered = threading.Event()
    worker_released = threading.Event()
    chat_completed = threading.Event()
    cancel = threading.Event()
    executors: list[Any] = []
    futures: list[Future] = []
    future_cancel_calls: list[Future] = []
    shutdown_calls: list[tuple[bool, bool]] = []
    coordinator_join_calls: list[threading.Thread] = []
    close_live_names: list[str] = []
    baseline_threads = set(threading.enumerate())
    real_executor = voice.ThreadPoolExecutor
    real_controller = voice._FirstChunkDeadlineController
    real_join = threading.Thread.join

    class TrackingExecutor(real_executor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            executors.append(self)

        def submit(self, *args, **kwargs):
            future = super().submit(*args, **kwargs)
            futures.append(future)
            original_cancel = future.cancel

            def tracked_cancel():
                future_cancel_calls.append(future)
                return original_cancel()

            future.cancel = tracked_cancel
            return future

        def shutdown(self, wait=True, *, cancel_futures=False):
            shutdown_calls.append((wait, cancel_futures))
            return super().shutdown(wait=wait, cancel_futures=cancel_futures)

    class FailingCloseController(real_controller):
        def close(self):
            close_live_names.extend(
                thread.name for thread in threading.enumerate() if thread.is_alive()
            )
            close_entered.set()
            raise cleanup_failure

    def tracking_join(thread, *args, **kwargs):
        if thread.name == "ms4-rest-emit-coordinator":
            coordinator_join_calls.append(thread)
        return real_join(thread, *args, **kwargs)

    monkeypatch.setattr(voice, "ThreadPoolExecutor", TrackingExecutor)
    monkeypatch.setattr(voice, "_FirstChunkDeadlineController", FailingCloseController)
    monkeypatch.setattr(threading.Thread, "join", tracking_join)

    def synthesize(*, text, on_request_dispatched=None, cancel_event=None, **_kwargs):
        assert text == opening
        assert on_request_dispatched is not None
        assert cancel_event is not None
        on_request_dispatched()
        worker_started.set()
        assert cancel_event.wait(timeout=5)
        assert close_entered.wait(timeout=5)
        worker_released.set()
        raise voice.VoiceUnavailable("controlled Repair63 worker cancellation")

    monkeypatch.setattr(voice, "synthesize", synthesize)

    def chat(_message, *, stream_callback=None, **_kwargs):
        assert stream_callback is not None
        assert stream_callback(f"{opening} ")
        assert worker_started.wait(timeout=5)
        assert stream_callback(tail)
        chat_completed.set()
        return completed_chat_response(reply, "repair63-cleanup-causality")

    def emit(event, payload):
        if event == "chunk_scheduled" and payload["index"] == 1:
            assert chat_completed.is_set()
            raise primary
        return True

    def live_turn_threads() -> list[threading.Thread]:
        return [
            thread
            for thread in threading.enumerate()
            if thread not in baseline_threads
            and thread.is_alive()
            and (
                thread.name.startswith("ms4-tts")
                or thread.name == "ms4-rest-emit-coordinator"
            )
        ]

    raised: BaseException | None = None
    try:
        try:
            voice.voice_ptt_turn_stream(
                runner=_Runner(),
                audio=b"FAKE_WAV",
                chat_fn=chat,
                transcribe_fn=lambda **_kwargs: {
                    "text": "exercise cleanup failure causality",
                    "model": "whisper-1",
                },
                emit=emit,
                chunker=voice.SentenceChunker(
                    first_chunk_min_words=10,
                    min_chunk_words=1,
                    max_chunk_words=20,
                ),
                tts_pool_size=6,
                engine="rest",
                cancel_event=cancel,
            )
        except BaseException as exc:
            raised = exc

        assert raised is primary, (
            "cleanup must retain the original post-chat exception as top-level; "
            f"got {raised!r}"
        )
        assert raised.__cause__ is cleanup_failure
        assert any(
            "repair63 timer close cleanup sentinel" in note
            for note in getattr(raised, "__notes__", ())
        )
        assert any(
            "REST turn cleanup failed while preserving primary exception"
            in record.getMessage()
            and "repair63 timer close cleanup sentinel" in record.getMessage()
            for record in caplog.records
        )
        assert futures and future_cancel_calls == futures
        assert (False, True) in shutdown_calls
        assert coordinator_join_calls
        assert "ms4-rest-emit-coordinator" in close_live_names
        assert any(name.startswith("ms4-tts") for name in close_live_names)
        assert worker_released.wait(timeout=5)
        assert live_turn_threads() == []
    finally:
        close_entered.set()
        cancel.set()
        for executor in executors:
            executor.shutdown(wait=False, cancel_futures=True)
        for executor in executors:
            executor.shutdown(wait=True, cancel_futures=True)
