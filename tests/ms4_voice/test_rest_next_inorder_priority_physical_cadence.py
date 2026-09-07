from __future__ import annotations

import json
import threading
from concurrent.futures import Future, ThreadPoolExecutor as RealThreadPoolExecutor
from typing import Any

import pytest

from machine_spirit_4.gateway import voice
from tests.ms4_voice._test_doubles import completed_chat_response


class _Runner:
    hivemind_url = "http://hive:0"
    ms3_url = "http://ms3:0"


def test_rest_starts_next_inorder_chunk_before_first_audio_delivery(monkeypatch):
    """Default six-worker REST keeps chunk 2 on the pre-delivery priority path."""
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
        "The first chunk is ready.",
        "The second chunk is ready.",
        "The third chunk is ready.",
        "The fourth chunk is ready.",
    )
    reply = " ".join(pieces)
    state_lock = threading.Lock()
    request_dispatched = threading.Event()
    dispatch_callback_returned = threading.Event()
    allow_first_response = threading.Event()
    first_audio_delivered = threading.Event()
    chat_finished = threading.Event()
    turn_cancel = threading.Event()
    chunk_two_started = threading.Event()
    allow_chunk_two_response = threading.Event()
    submissions: list[tuple[int, Any, tuple[Any, ...], dict[str, Any], Future]] = []
    started: list[int] = []
    events: list[tuple[str, dict[str, Any]]] = []
    worker_errors: list[BaseException] = []
    turn_errors: list[BaseException] = []
    turn_results: list[dict[str, Any]] = []

    class _ControlledExecutor:
        def __init__(self, **_kwargs):
            self._threads: set[threading.Thread] = set()

        def submit(self, fn, *args, **kwargs):
            index = pieces.index(kwargs["text"])
            future = Future()
            with state_lock:
                submissions.append((index, fn, args, kwargs, future))
            return future

        def shutdown(self, wait=True, *, cancel_futures=False):
            if cancel_futures:
                with state_lock:
                    futures = [entry[-1] for entry in submissions]
                for future in futures:
                    future.cancel()

    monkeypatch.setattr(voice, "ThreadPoolExecutor", _ControlledExecutor)

    def speech_request(*, body, cancel_event, on_request_dispatched=None, **_kwargs):
        text = json.loads(body.decode("utf-8"))["input"]
        index = pieces.index(text)
        with state_lock:
            started.append(index)
        if index == 0:
            request_dispatched.set()
            assert on_request_dispatched is not None
            on_request_dispatched()
            dispatch_callback_returned.set()
            assert allow_first_response.wait(timeout=5)
        elif index == 2:
            chunk_two_started.set()
            assert allow_chunk_two_response.wait(timeout=5)
        return f"WAV-{index}".encode(), "audio/wav", {}

    monkeypatch.setattr(voice, "_cancellable_speech_request", speech_request)

    def emit(event, payload):
        with state_lock:
            events.append((event, dict(payload)))
        if event == "audio_chunk" and payload["index"] == 0:
            first_audio_delivered.set()
        return True

    def chat(_message, *, stream_callback=None, **_kwargs):
        assert stream_callback(reply) is not False
        chat_finished.set()
        return completed_chat_response(reply, "next-inorder-priority")

    def run_turn():
        try:
            turn_results.append(voice.voice_ptt_turn_stream(
                runner=_Runner(),
                audio=b"FAKE_WAV",
                chat_fn=chat,
                transcribe_fn=lambda **_kwargs: {
                    "text": "speak four priority chunks",
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
                cancel_event=turn_cancel,
            ))
        except BaseException as exc:
            turn_errors.append(exc)

    def complete_submission(
        submission: tuple[int, Any, tuple[Any, ...], dict[str, Any], Future]
    ) -> None:
        _index, fn, args, kwargs, future = submission
        try:
            value = fn(*args, **kwargs)
        except BaseException as exc:
            with state_lock:
                worker_errors.append(exc)
            future.set_exception(exc)
        else:
            future.set_result(value)

    turn = threading.Thread(target=run_turn, name="rest-next-inorder-priority-turn")
    turn.start()
    assert chat_finished.wait(timeout=5)
    with state_lock:
        initial = list(submissions)
    assert [entry[0] for entry in initial] == [0]

    first_worker = threading.Thread(
        target=complete_submission,
        args=(initial[0],),
        name="rest-next-inorder-priority-first",
    )
    first_worker.start()
    assert request_dispatched.wait(timeout=5)
    assert dispatch_callback_returned.wait(timeout=5)

    with state_lock:
        before_first_delivery = list(submissions)
    early_tail_workers = [
        (entry[0], threading.Thread(target=complete_submission, args=(entry,)))
        for entry in before_first_delivery
        if entry[0] in {1, 2}
    ]
    for _index, worker in early_tail_workers:
        worker.start()
    assert chunk_two_started.wait(timeout=5)
    for index, worker in early_tail_workers:
        if index == 1:
            worker.join(timeout=5)
            assert not worker.is_alive()
    assert first_audio_delivered.is_set() is False

    allow_first_response.set()
    first_worker.join(timeout=5)
    assert not first_worker.is_alive()
    assert first_audio_delivered.wait(timeout=5), (
        "chunk 0 synthesis completed but its audio was not accepted by the emitter"
    )
    with state_lock:
        after_first_delivery = list(submissions)
    after_first_indices = [entry[0] for entry in after_first_delivery]
    if 3 not in after_first_indices:
        # Cleanup for the RED path: the turn would otherwise wait until its
        # production final timeout because the deferred tail was never released.
        turn_cancel.set()
    allow_chunk_two_response.set()
    for index, worker in early_tail_workers:
        if index == 2:
            worker.join(timeout=5)
            assert not worker.is_alive()
    with state_lock:
        remaining = [entry for entry in submissions if not entry[-1].done()]
    for entry in remaining:
        complete_submission(entry)
    turn.join(timeout=5)
    assert not turn.is_alive()

    early_indices = [entry[0] for entry in before_first_delivery]
    assert early_indices == [0, 1, 2], (
        "chunk 2 must start with the bounded lead set instead of waiting for "
        f"chunk 0 delivery; pre-delivery submissions={early_indices}"
    )
    assert after_first_indices == [0, 1, 2, 3], (
        "successful chunk-0 delivery must durably release every remaining tail "
        f"even while chunk 2 is unfinished; submissions={after_first_indices}"
    )
    assert worker_errors == []
    assert turn_errors == []
    assert len(turn_results) == 1
    assert first_audio_delivered.is_set()
    scheduled = [payload["index"] for event, payload in events if event == "chunk_scheduled"]
    emitted = [payload["index"] for event, payload in events if event == "audio_chunk"]
    assert scheduled == [0, 1, 2, 3]
    assert emitted == [0, 1, 2, 3]
    first_audio_position = next(
        i for i, (event, payload) in enumerate(events)
        if event == "audio_chunk" and payload["index"] == 0
    )
    final_tail_schedule_position = next(
        i for i, (event, payload) in enumerate(events)
        if event == "chunk_scheduled" and payload["index"] == 3
    )
    assert first_audio_position < final_tail_schedule_position


def _configure_voice_test(monkeypatch) -> None:
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


def _capture_pre_delivery_admissions(
    monkeypatch,
    *,
    pool_size: int,
    duplicate_callback: bool = False,
) -> tuple[list[int], list[int]]:
    _configure_voice_test(monkeypatch)
    pieces = (
        "The first chunk is ready.",
        "The second chunk is ready.",
        "The third chunk is ready.",
        "The fourth chunk is ready.",
    )
    reply = " ".join(pieces)
    state_lock = threading.Lock()
    chat_finished = threading.Event()
    callback_returned = threading.Event()
    cancel_event = threading.Event()
    submissions: list[tuple[int, Any, tuple[Any, ...], dict[str, Any], Future]] = []
    events: list[tuple[str, dict[str, Any]]] = []
    turn_errors: list[BaseException] = []

    class _ControlledExecutor:
        def __init__(self, **_kwargs):
            self._threads: set[threading.Thread] = set()

        def submit(self, fn, *args, **kwargs):
            index = pieces.index(kwargs["text"])
            future = Future()
            with state_lock:
                submissions.append((index, fn, args, kwargs, future))
            return future

        def shutdown(self, wait=True, *, cancel_futures=False):
            if cancel_futures:
                with state_lock:
                    futures = [entry[-1] for entry in submissions]
                for future in futures:
                    future.cancel()

    monkeypatch.setattr(voice, "ThreadPoolExecutor", _ControlledExecutor)

    def speech_request(*, body, on_request_dispatched=None, **_kwargs):
        text = json.loads(body.decode("utf-8"))["input"]
        index = pieces.index(text)
        if index == 0:
            assert on_request_dispatched is not None
            on_request_dispatched()
            if duplicate_callback:
                on_request_dispatched()
            callback_returned.set()
            assert cancel_event.wait(timeout=5)
            raise voice.VoiceUnavailable("controlled admission capture cancelled")
        raise AssertionError("tail work must not execute in admission-only capture")

    monkeypatch.setattr(voice, "_cancellable_speech_request", speech_request)

    def run_turn():
        try:
            voice.voice_ptt_turn_stream(
                runner=_Runner(),
                audio=b"FAKE_WAV",
                chat_fn=lambda _message, stream_callback=None, **_kwargs: (
                    stream_callback(reply),
                    chat_finished.set(),
                    completed_chat_response(reply, "admission-budget"),
                )[2],
                transcribe_fn=lambda **_kwargs: {
                    "text": "capture lead admissions",
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
                tts_pool_size=pool_size,
                engine="rest",
                cancel_event=cancel_event,
            )
        except BaseException as exc:
            turn_errors.append(exc)

    def complete_first(
        submission: tuple[int, Any, tuple[Any, ...], dict[str, Any], Future]
    ) -> None:
        _index, fn, args, kwargs, future = submission
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:
            future.set_exception(exc)

    turn = threading.Thread(target=run_turn, name=f"rest-admission-pool-{pool_size}")
    turn.start()
    assert chat_finished.wait(timeout=5)
    with state_lock:
        initial = list(submissions)
    assert [entry[0] for entry in initial] == [0]
    first_worker = threading.Thread(target=complete_first, args=(initial[0],))
    first_worker.start()
    assert callback_returned.wait(timeout=5)
    with state_lock:
        admitted = [entry[0] for entry in submissions]
    cancel_event.set()
    first_worker.join(timeout=5)
    assert not first_worker.is_alive()
    turn.join(timeout=5)
    assert not turn.is_alive()
    assert len(turn_errors) == 1
    assert not any(event == "audio_chunk" for event, _payload in events)
    scheduled = [payload["index"] for event, payload in events if event == "chunk_scheduled"]
    return admitted, scheduled


@pytest.mark.parametrize(
    ("pool_size", "expected_admitted", "expected_scheduled"),
    [
        (1, [0, 1], [0, 1]),
        (2, [0, 1], [0, 1]),
        (3, [0, 1], [0, 1]),
        (4, [0, 1], [0, 1]),
        (5, [0, 1], [0, 1]),
        (6, [0, 1, 2], [0, 1]),
        (10, [0, 1, 2], [0, 1]),
    ],
)
def test_builtin_rest_lead_budget_is_bounded(
    monkeypatch,
    pool_size,
    expected_admitted,
    expected_scheduled,
):
    admitted, scheduled = _capture_pre_delivery_admissions(
        monkeypatch, pool_size=pool_size
    )
    assert admitted == expected_admitted
    assert scheduled == expected_scheduled


def test_duplicate_request_dispatched_callback_is_idempotent(monkeypatch):
    admitted, scheduled = _capture_pre_delivery_admissions(
        monkeypatch,
        pool_size=6,
        duplicate_callback=True,
    )
    assert admitted == [0, 1, 2]
    assert scheduled == [0, 1]


def test_extra_lead_ack_does_not_block_chunk_zero_response(monkeypatch):
    _configure_voice_test(monkeypatch)
    pieces = (
        "The first chunk is ready.",
        "The second chunk is ready.",
        "The third chunk is ready.",
        "The fourth chunk is ready.",
    )
    reply = " ".join(pieces)
    extra_ack_entered = threading.Event()
    release_extra_ack = threading.Event()
    callback_returned = threading.Event()
    response_read_started = threading.Event()
    first_audio_delivered = threading.Event()
    extra_synthesis_started = threading.Event()
    chat_finished = threading.Event()
    cancel_event = threading.Event()
    events: list[tuple[str, dict[str, Any]]] = []
    state_lock = threading.Lock()
    submissions: list[tuple[int, Any, tuple[Any, ...], dict[str, Any], Future]] = []
    turn_errors: list[BaseException] = []

    class _ControlledExecutor:
        def __init__(self, **_kwargs):
            self._threads: set[threading.Thread] = set()

        def submit(self, fn, *args, **kwargs):
            index = pieces.index(kwargs["text"])
            future = Future()
            entry = (index, fn, args, kwargs, future)
            with state_lock:
                submissions.append(entry)
            if index == 2:
                worker = threading.Thread(
                    target=self._complete,
                    args=(entry,),
                    name="rest-extra-lead-ack-worker",
                )
                self._threads.add(worker)
                worker.start()
            return future

        @staticmethod
        def _complete(entry):
            _index, fn, args, kwargs, future = entry
            try:
                future.set_result(fn(*args, **kwargs))
            except BaseException as exc:
                future.set_exception(exc)

        def shutdown(self, wait=True, *, cancel_futures=False):
            if cancel_futures:
                with state_lock:
                    futures = [entry[-1] for entry in submissions]
                for future in futures:
                    future.cancel()
            if wait:
                for worker in list(self._threads):
                    worker.join(timeout=5)

    monkeypatch.setattr(voice, "ThreadPoolExecutor", _ControlledExecutor)

    def speech_request(*, body, on_request_dispatched=None, **_kwargs):
        text = json.loads(body.decode("utf-8"))["input"]
        index = pieces.index(text)
        if index == 0:
            assert on_request_dispatched is not None
            on_request_dispatched()
            callback_returned.set()
            response_read_started.set()
        if index == 2:
            extra_synthesis_started.set()
        return f"WAV-{index}".encode(), "audio/wav", {}

    monkeypatch.setattr(voice, "_cancellable_speech_request", speech_request)

    def emit(event, payload):
        with state_lock:
            events.append((event, dict(payload)))
        if event == "chunk_scheduled" and payload["index"] == 2:
            extra_ack_entered.set()
            assert release_extra_ack.wait(timeout=5)
        if event == "audio_chunk" and payload["index"] == 0:
            first_audio_delivered.set()
        return True

    def run_turn():
        try:
            voice.voice_ptt_turn_stream(
                runner=_Runner(),
                audio=b"FAKE_WAV",
                chat_fn=lambda _message, stream_callback=None, **_kwargs: (
                    stream_callback(reply),
                    chat_finished.set(),
                    completed_chat_response(reply, "blocked-extra-ack"),
                )[2],
                transcribe_fn=lambda **_kwargs: {
                    "text": "block only the extra lead acknowledgement",
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
                cancel_event=cancel_event,
            )
        except BaseException as exc:
            turn_errors.append(exc)

    def complete_submission(
        submission: tuple[int, Any, tuple[Any, ...], dict[str, Any], Future]
    ) -> None:
        _ControlledExecutor._complete(submission)

    turn = threading.Thread(target=run_turn, name="rest-extra-lead-ack")
    turn.start()
    assert chat_finished.wait(timeout=5)
    with state_lock:
        initial = list(submissions)
    assert [entry[0] for entry in initial] == [0]
    first_worker = threading.Thread(
        target=complete_submission,
        args=(initial[0],),
        name="rest-extra-lead-ack-first",
    )
    first_worker.start()
    assert extra_ack_entered.wait(timeout=5)
    callback_before_ack = callback_returned.wait(timeout=1)
    response_before_ack = response_read_started.is_set()
    first_audio_before_ack = (
        first_audio_delivered.wait(timeout=1)
        if callback_before_ack
        else first_audio_delivered.is_set()
    )
    extra_started_before_ack = extra_synthesis_started.is_set()
    release_extra_ack.set()
    extra_started_after_ack = extra_synthesis_started.wait(timeout=5)
    first_worker.join(timeout=5)
    assert not first_worker.is_alive()
    with state_lock:
        after_first_delivery = list(submissions)
    if not any(entry[0] == 3 for entry in after_first_delivery):
        cancel_event.set()
    with state_lock:
        remaining = [
            entry for entry in submissions
            if entry[0] != 2 and not entry[-1].done()
        ]
    for entry in remaining:
        complete_submission(entry)
    turn.join(timeout=5)
    assert not turn.is_alive()

    assert callback_before_ack is True
    assert response_before_ack is True
    assert first_audio_before_ack is True
    assert extra_started_before_ack is False
    assert extra_started_after_ack is True
    assert turn_errors == []
    assert [payload["index"] for event, payload in events if event == "audio_chunk"] == [
        0, 1, 2, 3,
    ]


def test_cancellation_after_three_leads_start_emits_no_audio(monkeypatch):
    """Two running leads plus one queued lead cancel without tail effects.

    The original node name is retained for stable CI/Repair66 lineage. The
    production REST contract now caps built-in synthesis at two workers, so
    the third admitted lead is intentionally queued rather than started.
    """
    _configure_voice_test(monkeypatch)
    pieces = (
        "The first chunk is ready.",
        "The second chunk is ready.",
        "The third chunk is ready.",
        "The fourth chunk is ready.",
    )
    reply = " ".join(pieces)
    cancel_event = threading.Event()
    started_events = {index: threading.Event() for index in range(4)}
    started: list[int] = []
    events: list[tuple[str, dict[str, Any]]] = []
    turn_errors: list[BaseException] = []
    state_lock = threading.Lock()

    def speech_request(*, body, on_request_dispatched=None, **_kwargs):
        text = json.loads(body.decode("utf-8"))["input"]
        index = pieces.index(text)
        with state_lock:
            started.append(index)
        started_events[index].set()
        if index == 0:
            assert on_request_dispatched is not None
            on_request_dispatched()
        assert cancel_event.wait(timeout=5)
        raise voice.VoiceUnavailable(f"chunk {index} cancelled")

    monkeypatch.setattr(voice, "_cancellable_speech_request", speech_request)

    def run_turn():
        try:
            voice.voice_ptt_turn_stream(
                runner=_Runner(),
                audio=b"FAKE_WAV",
                chat_fn=lambda _message, stream_callback=None, **_kwargs: (
                    stream_callback(reply),
                    completed_chat_response(reply, "cancel-capacity-two-leads"),
                )[1],
                transcribe_fn=lambda **_kwargs: {
                    "text": "cancel running and queued speculative leads",
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

    turn = threading.Thread(target=run_turn, name="rest-cancel-capacity-two-leads")
    turn.start()
    for index in (0, 1):
        assert started_events[index].wait(timeout=5)
    assert started_events[2].is_set() is False
    assert started_events[3].is_set() is False
    cancel_event.set()
    turn.join(timeout=5)
    assert not turn.is_alive()
    assert sorted(started) == [0, 1]
    assert [payload["index"] for event, payload in events if event == "chunk_scheduled"] == [
        0, 1,
    ]
    assert not any(event == "audio_chunk" for event, _payload in events)
    assert len(turn_errors) == 1


def test_failed_chunk_zero_write_does_not_release_or_emit_tail(monkeypatch):
    _configure_voice_test(monkeypatch)
    pieces = (
        "The first chunk is ready.",
        "The second chunk is ready.",
        "The third chunk is ready.",
        "The fourth chunk is ready.",
    )
    reply = " ".join(pieces)
    events: list[tuple[str, dict[str, Any]]] = []
    turn_errors: list[BaseException] = []

    def speech_request(*, body, on_request_dispatched=None, **_kwargs):
        text = json.loads(body.decode("utf-8"))["input"]
        index = pieces.index(text)
        if index == 0:
            assert on_request_dispatched is not None
            on_request_dispatched()
        return f"WAV-{index}".encode(), "audio/wav", {}

    monkeypatch.setattr(voice, "_cancellable_speech_request", speech_request)

    def emit(event, payload):
        events.append((event, dict(payload)))
        return not (event == "audio_chunk" and payload["index"] == 0)

    try:
        voice.voice_ptt_turn_stream(
            runner=_Runner(),
            audio=b"FAKE_WAV",
            chat_fn=lambda _message, stream_callback=None, **_kwargs: (
                stream_callback(reply),
                completed_chat_response(reply, "failed-first-write"),
            )[1],
            transcribe_fn=lambda **_kwargs: {
                "text": "fail the first client write",
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
    except BaseException as exc:
        turn_errors.append(exc)

    scheduled = [payload["index"] for event, payload in events if event == "chunk_scheduled"]
    attempted_audio = [payload["index"] for event, payload in events if event == "audio_chunk"]
    assert set(scheduled) <= {0, 1, 2}
    assert 3 not in scheduled
    assert attempted_audio == [0]
    assert len(turn_errors) == 1


def _run_extra_lead_schedule_failure(monkeypatch, *, raise_error: bool) -> dict[str, Any]:
    _configure_voice_test(monkeypatch)
    pieces = (
        "The first chunk is ready.",
        "The second chunk is ready.",
        "The third chunk is ready.",
        "The fourth chunk is ready.",
    )
    reply = " ".join(pieces)
    chat_finished = threading.Event()
    failure_observed = threading.Event()
    cancel_event = threading.Event()
    client_alive = threading.Event()
    client_alive.set()
    cancellation_seen_before_response: list[bool] = []
    events: list[tuple[str, dict[str, Any]]] = []
    started: list[int] = []
    turn_errors: list[BaseException] = []
    turn_results: list[dict[str, Any]] = []
    executor_instances: list[Any] = []

    class _RecordingExecutor:
        def __init__(self, *args, **kwargs):
            self._inner = RealThreadPoolExecutor(*args, **kwargs)
            self.futures: list[tuple[str | None, Future]] = []
            executor_instances.append(self)

        @property
        def _threads(self):
            return self._inner._threads

        def submit(self, fn, *args, **kwargs):
            future = self._inner.submit(fn, *args, **kwargs)
            self.futures.append((kwargs.get("text"), future))
            return future

        def shutdown(self, wait=True, *, cancel_futures=False):
            return self._inner.shutdown(wait=wait, cancel_futures=cancel_futures)

    monkeypatch.setattr(voice, "ThreadPoolExecutor", _RecordingExecutor)

    def speech_request(
        *,
        body,
        cancel_event: threading.Event,
        on_request_dispatched=None,
        **_kwargs,
    ):
        text = json.loads(body.decode("utf-8"))["input"]
        index = pieces.index(text)
        started.append(index)
        if index == 0:
            assert chat_finished.wait(timeout=5)
            assert on_request_dispatched is not None
            on_request_dispatched()
            assert failure_observed.wait(timeout=5)
            cancellation_seen_before_response.append(cancel_event.wait(timeout=1))
        return f"WAV-{index}".encode(), "audio/wav", {}

    monkeypatch.setattr(voice, "_cancellable_speech_request", speech_request)

    def emit(event, payload):
        events.append((event, dict(payload)))
        if event == "chunk_scheduled" and payload["index"] == 2:
            failure_observed.set()
            if raise_error:
                raise RuntimeError("extra lead schedule ACK exploded")
            return False
        return True

    def run_turn():
        try:
            turn_results.append(voice.voice_ptt_turn_stream(
                runner=_Runner(),
                audio=b"FAKE_WAV",
                chat_fn=lambda _message, stream_callback=None, **_kwargs: (
                    stream_callback(reply),
                    chat_finished.set(),
                    completed_chat_response(reply, "extra-lead-schedule-failure"),
                )[2],
                transcribe_fn=lambda **_kwargs: {
                    "text": "fail only the extra lead schedule acknowledgement",
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
                cancel_event=cancel_event,
                client_alive=client_alive,
            ))
        except BaseException as exc:
            turn_errors.append(exc)

    turn = threading.Thread(target=run_turn, name="rest-extra-lead-schedule-failure")
    turn.start()
    turn.join(timeout=5)
    assert not turn.is_alive()
    assert len(executor_instances) == 1
    executor = executor_instances[0]
    chunk_two_futures = [
        future for text, future in executor.futures if text == pieces[2]
    ]
    assert len(chunk_two_futures) == 1
    chunk_two_future = chunk_two_futures[0]
    assert chunk_two_future.done()
    return {
        "cancelled": cancel_event.is_set(),
        "client_alive": client_alive.is_set(),
        "cancellation_seen_before_response": cancellation_seen_before_response,
        "events": events,
        "started": started,
        "turn_errors": turn_errors,
        "turn_results": turn_results,
        "future_exception": chunk_two_future.exception(),
        "workers_alive": [
            worker.name for worker in executor._threads if worker.is_alive()
        ],
    }


def _assert_extra_lead_schedule_failure_closed(result: dict[str, Any]) -> None:
    assert result["cancelled"] is True
    assert result["client_alive"] is False
    assert result["cancellation_seen_before_response"] == [True]
    assert set(result["started"]) == {0, 1}
    assert result["turn_results"] == []
    assert len(result["turn_errors"]) == 1
    assert result["workers_alive"] == []
    scheduled = [
        payload["index"]
        for event, payload in result["events"]
        if event == "chunk_scheduled"
    ]
    emitted = [
        payload["index"]
        for event, payload in result["events"]
        if event == "audio_chunk"
    ]
    assert scheduled == [0, 1, 2]
    assert emitted == []


def test_extra_lead_schedule_ack_exception_fails_closed(monkeypatch):
    result = _run_extra_lead_schedule_failure(monkeypatch, raise_error=True)
    _assert_extra_lead_schedule_failure_closed(result)
    failure = result["future_exception"]
    assert isinstance(failure, RuntimeError)
    assert str(failure) == "extra lead schedule ACK exploded"


def test_extra_lead_schedule_ack_false_still_fails_closed(monkeypatch):
    result = _run_extra_lead_schedule_failure(monkeypatch, raise_error=False)
    _assert_extra_lead_schedule_failure_closed(result)
    failure = result["future_exception"]
    assert isinstance(failure, voice.VoiceUnavailable)
    assert "acknowledgement was rejected" in str(failure)
