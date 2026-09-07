from __future__ import annotations

import json
import threading
from concurrent.futures import Future
from typing import Any

from machine_spirit_4.gateway import voice
from tests.ms4_voice._test_doubles import completed_chat_response


class _Runner:
    hivemind_url = "http://hive:0"
    ms3_url = "http://ms3:0"


def test_rest_stream_uses_one_lead_then_fans_remaining_tail(monkeypatch):
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
    first_delivery_accepted = threading.Event()
    chat_finished = threading.Event()
    request_dispatched = threading.Event()
    dispatch_callback_returned = threading.Event()
    allow_first_response = threading.Event()
    tail_pair = threading.Barrier(2)
    later_tail_completed = threading.Event()
    tail_fanout_submitted = threading.Event()
    state_lock = threading.Lock()
    tail_started_before_request_dispatch: list[int] = []
    lead_started_after_first_delivery: list[int] = []
    later_tail_started_before_first_delivery: list[int] = []
    tail_completion_order: list[int] = []
    later_tail_thread_ids: set[int] = set()
    events: list[tuple[str, dict[str, Any]]] = []
    submissions: list[tuple[int, Any, tuple[Any, ...], dict[str, Any], Future]] = []
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
                if index == len(pieces) - 1:
                    tail_fanout_submitted.set()
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
        if index == 0:
            request_dispatched.set()
            if on_request_dispatched is not None:
                on_request_dispatched()
            dispatch_callback_returned.set()
            assert allow_first_response.wait(timeout=5)
        elif index == 1:
            with state_lock:
                if not request_dispatched.is_set():
                    tail_started_before_request_dispatch.append(index)
                if first_delivery_accepted.is_set():
                    lead_started_after_first_delivery.append(index)
                tail_completion_order.append(index)
        else:
            with state_lock:
                if not request_dispatched.is_set():
                    tail_started_before_request_dispatch.append(index)
                if not first_delivery_accepted.is_set():
                    later_tail_started_before_first_delivery.append(index)
                later_tail_thread_ids.add(threading.get_ident())
            tail_pair.wait(timeout=5)
            if index == 2:
                assert later_tail_completed.wait(timeout=5)
            else:
                with state_lock:
                    tail_completion_order.append(index)
                later_tail_completed.set()
            if index == 2:
                with state_lock:
                    tail_completion_order.append(index)
        return f"WAV-{index}".encode(), "audio/wav", {}

    monkeypatch.setattr(voice, "_cancellable_speech_request", speech_request)

    def emit(event, payload):
        with state_lock:
            events.append((event, dict(payload)))
        if event == "audio_chunk" and payload["index"] == 0:
            # Returning True is the existing delivery acknowledgement contract.
            first_delivery_accepted.set()
        return True

    def chat(_message, *, stream_callback=None, **_kwargs):
        assert stream_callback(reply) is not False
        chat_finished.set()
        return completed_chat_response(reply, "dispatch-gate")

    def run_turn():
        try:
            turn_results.append(voice.voice_ptt_turn_stream(
                runner=_Runner(),
                audio=b"FAKE_WAV",
                chat_fn=chat,
                transcribe_fn=lambda **_kwargs: {
                    "text": "speak four chunks",
                    "model": "whisper-1",
                },
                emit=emit,
                chunker=voice.SentenceChunker(
                    first_chunk_min_words=5,
                    min_chunk_words=1,
                    max_chunk_words=20,
                ),
                tts_pool_size=3,
                engine="rest",
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

    def complete_concurrently(
        selected: list[tuple[int, Any, tuple[Any, ...], dict[str, Any], Future]]
    ) -> None:
        threads = [
            threading.Thread(target=complete_submission, args=(submission,))
            for submission in selected
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
            assert not thread.is_alive()

    turn = threading.Thread(target=run_turn, name="rest-first-chunk-gate-test")
    turn.start()
    assert chat_finished.wait(timeout=5)

    with state_lock:
        before_request_dispatch = list(submissions)
    assert [submission[0] for submission in before_request_dispatch] == [0]

    first_worker = threading.Thread(
        target=complete_submission,
        args=(before_request_dispatch[0],),
    )
    first_worker.start()
    assert request_dispatched.wait(timeout=5)
    assert dispatch_callback_returned.wait(timeout=5)

    with state_lock:
        after_request_dispatch = list(submissions)
    delivery_completed_at_release = first_delivery_accepted.is_set()
    complete_concurrently(after_request_dispatch[1:])

    allow_first_response.set()
    first_worker.join(timeout=5)
    assert not first_worker.is_alive()
    assert tail_fanout_submitted.wait(timeout=5), (
        "chunk 0 was accepted but the remaining tail was not submitted"
    )

    with state_lock:
        remaining_tail = [
            submission
            for submission in submissions
            if submission[0] > 0 and not submission[-1].done()
        ]
    complete_concurrently(remaining_tail)
    turn.join(timeout=5)
    assert not turn.is_alive()

    assert [submission[0] for submission in after_request_dispatch] == [0, 1], (
        "request dispatch released more than the single lead tail"
    )
    assert delivery_completed_at_release is False
    assert tail_started_before_request_dispatch == []
    assert lead_started_after_first_delivery == []
    assert later_tail_started_before_first_delivery == []
    assert worker_errors == []
    assert turn_errors == []
    assert len(turn_results) == 1
    assert first_delivery_accepted.is_set()
    assert len(later_tail_thread_ids) == 2
    assert tail_completion_order == [1, 3, 2]
    scheduled = [payload["text"] for event, payload in events if event == "chunk_scheduled"]
    emitted = [payload for event, payload in events if event == "audio_chunk"]
    assert scheduled == list(pieces)
    assert [payload["index"] for payload in emitted] == [0, 1, 2, 3]
    assert [payload["text"] for payload in emitted] == list(pieces)
    first_audio_position = next(
        index for index, (event, payload) in enumerate(events)
        if event == "audio_chunk" and payload["index"] == 0
    )
    lead_schedule_position = next(
        index for index, (event, payload) in enumerate(events)
        if event == "chunk_scheduled" and payload["index"] == 1
    )
    assert lead_schedule_position < first_audio_position
    assert all(
        next(
            index for index, (event, payload) in enumerate(events)
            if event == "chunk_scheduled" and payload["index"] == tail_index
        ) > first_audio_position
        for tail_index in (2, 3)
    )
    result = turn_results[0]
    lifecycle = result["metrics"]["lifecycle"]
    assert result["metrics"]["audio_client_written"] == 4
    assert lifecycle["rest_tasks_submitted"] == 4
    assert lifecycle["rest_tasks_completed"] == 4
    assert lifecycle["rest_workers_started"] == 0
    assert lifecycle["rest_workers_joined"] == 0
    assert lifecycle["rest_worker_join_timeouts"] == 0


def test_rest_stream_cancel_before_request_dispatch_never_starts_tail(monkeypatch):
    monkeypatch.setenv("MS4_VOICE_FIRST_CHUNK_DEADLINE_S", "0")
    monkeypatch.setenv("MS4_VOICE_SMART_REFLEX", "0")
    monkeypatch.setenv("MS4_VOICE_EGG_HONORABLE", "0")
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
    )
    first_started = threading.Event()
    turn_cancel = threading.Event()
    tail_started: list[str] = []
    events: list[tuple[str, dict[str, Any]]] = []
    turn_errors: list[BaseException] = []

    def synthesize(*, text, cancel_event, on_request_dispatched=None, **_kwargs):
        if text == pieces[0]:
            first_started.set()
            assert cancel_event.wait(timeout=5)
            raise voice.VoiceUnavailable("cancelled before request dispatch")
        tail_started.append(text)
        raise AssertionError("tail synthesis started before request dispatch")

    def run_turn():
        try:
            voice.voice_ptt_turn_stream(
                runner=_Runner(),
                audio=b"FAKE_WAV",
                chat_fn=lambda _message, stream_callback=None, **_kwargs: (
                    stream_callback(" ".join(pieces)),
                    completed_chat_response(" ".join(pieces), "cancel-before-dispatch"),
                )[1],
                transcribe_fn=lambda **_kwargs: {
                    "text": "cancel this turn",
                    "model": "whisper-1",
                },
                synthesize_fn=synthesize,
                emit=lambda event, payload: events.append((event, dict(payload))) or True,
                chunker=voice.SentenceChunker(
                    first_chunk_min_words=5,
                    min_chunk_words=1,
                    max_chunk_words=20,
                ),
                tts_pool_size=3,
                engine="rest",
                cancel_event=turn_cancel,
            )
        except BaseException as exc:
            turn_errors.append(exc)

    turn = threading.Thread(target=run_turn, name="rest-cancel-before-dispatch-test")
    turn.start()
    assert first_started.wait(timeout=5)
    turn_cancel.set()
    turn.join(timeout=5)

    assert not turn.is_alive()
    assert tail_started == []
    assert len(turn_errors) == 1
    assert "cancel" in str(turn_errors[0]).lower()
    assert [payload["index"] for event, payload in events if event == "chunk_scheduled"] == [0]
    assert not any(event == "audio_chunk" for event, _payload in events)


def test_request_dispatch_before_tail_formation_retains_one_lead_slot(monkeypatch):
    monkeypatch.setenv("MS4_VOICE_FIRST_CHUNK_DEADLINE_S", "0")
    monkeypatch.setenv("MS4_VOICE_SMART_REFLEX", "0")
    monkeypatch.setenv("MS4_VOICE_EGG_HONORABLE", "0")
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
    )
    request_dispatched = threading.Event()
    dispatch_callback_returned = threading.Event()
    lead_started = threading.Event()
    allow_first_response = threading.Event()
    first_delivery = threading.Event()
    started: list[int] = []
    later_started_before_delivery: list[int] = []
    events: list[tuple[str, dict[str, Any]]] = []
    turn_errors: list[BaseException] = []
    state_lock = threading.Lock()

    def synthesize(*, text, on_request_dispatched=None, **_kwargs):
        index = pieces.index(text)
        with state_lock:
            started.append(index)
            if index >= 2 and not first_delivery.is_set():
                later_started_before_delivery.append(index)
        if index == 0:
            request_dispatched.set()
            assert on_request_dispatched is not None
            on_request_dispatched()
            dispatch_callback_returned.set()
            assert allow_first_response.wait(timeout=5)
        elif index == 1:
            lead_started.set()
        return {
            "audio_bytes": f"WAV-{index}".encode(),
            "audio_base64": f"V0FW-{index}",
            "content_type": "audio/wav",
        }

    def emit(event, payload):
        events.append((event, dict(payload)))
        if event == "audio_chunk" and payload["index"] == 0:
            first_delivery.set()
        return True

    def chat(_message, *, stream_callback=None, **_kwargs):
        assert stream_callback(pieces[0] + " ") is not False
        assert dispatch_callback_returned.wait(timeout=5)
        assert stream_callback(" ".join(pieces[1:])) is not False
        return completed_chat_response(" ".join(pieces), "late-lead")

    def run_turn():
        try:
            voice.voice_ptt_turn_stream(
                runner=_Runner(),
                audio=b"FAKE_WAV",
                chat_fn=chat,
                transcribe_fn=lambda **_kwargs: {
                    "text": "form the tail later",
                    "model": "whisper-1",
                },
                synthesize_fn=synthesize,
                emit=emit,
                chunker=voice.SentenceChunker(
                    first_chunk_min_words=5,
                    min_chunk_words=1,
                    max_chunk_words=20,
                ),
                tts_pool_size=3,
                engine="rest",
            )
        except BaseException as exc:
            turn_errors.append(exc)

    turn = threading.Thread(target=run_turn, name="rest-late-lead-slot-test")
    turn.start()
    assert request_dispatched.wait(timeout=5)
    assert lead_started.wait(timeout=5)
    with state_lock:
        before_first_delivery = list(started)

    allow_first_response.set()
    turn.join(timeout=5)
    assert not turn.is_alive()
    assert before_first_delivery == [0, 1]
    assert later_started_before_delivery == []
    assert turn_errors == []
    assert started == [0, 1, 2]
    assert [payload["index"] for event, payload in events if event == "audio_chunk"] == [0, 1, 2]


def test_cancel_after_lead_release_retires_started_pair_only(monkeypatch):
    monkeypatch.setenv("MS4_VOICE_FIRST_CHUNK_DEADLINE_S", "0")
    monkeypatch.setenv("MS4_VOICE_SMART_REFLEX", "0")
    monkeypatch.setenv("MS4_VOICE_EGG_HONORABLE", "0")
    monkeypatch.setenv("MS4_TTS_REST_CLEANUP_TIMEOUT_S", "1")
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
    )
    turn_cancel = threading.Event()
    chat_buffered = threading.Event()
    lead_started = threading.Event()
    state_lock = threading.Lock()
    started: list[int] = []
    cancelled: list[int] = []
    events: list[tuple[str, dict[str, Any]]] = []
    turn_errors: list[BaseException] = []

    def synthesize(
        *,
        text,
        cancel_event,
        on_request_dispatched=None,
        **_kwargs,
    ):
        index = pieces.index(text)
        with state_lock:
            started.append(index)
        if index == 0:
            assert chat_buffered.wait(timeout=5)
            assert on_request_dispatched is not None
            on_request_dispatched()
        elif index == 1:
            lead_started.set()
        assert cancel_event.wait(timeout=5)
        with state_lock:
            cancelled.append(index)
        raise voice.VoiceUnavailable(f"chunk {index} cancelled")

    def chat(_message, *, stream_callback=None, **_kwargs):
        assert stream_callback(" ".join(pieces)) is not False
        chat_buffered.set()
        return completed_chat_response(" ".join(pieces), "cancel-after-lead")

    def run_turn():
        try:
            voice.voice_ptt_turn_stream(
                runner=_Runner(),
                audio=b"FAKE_WAV",
                chat_fn=chat,
                transcribe_fn=lambda **_kwargs: {
                    "text": "cancel after lead",
                    "model": "whisper-1",
                },
                synthesize_fn=synthesize,
                emit=lambda event, payload: events.append((event, dict(payload))) or True,
                chunker=voice.SentenceChunker(
                    first_chunk_min_words=5,
                    min_chunk_words=1,
                    max_chunk_words=20,
                ),
                tts_pool_size=3,
                engine="rest",
                cancel_event=turn_cancel,
            )
        except BaseException as exc:
            turn_errors.append(exc)

    turn = threading.Thread(target=run_turn, name="rest-cancel-after-lead-test")
    turn.start()
    assert lead_started.wait(timeout=5)
    turn_cancel.set()
    turn.join(timeout=5)

    assert not turn.is_alive()
    assert sorted(started) == [0, 1]
    assert sorted(cancelled) == [0, 1]
    assert len(turn_errors) == 1
    assert "cancel" in str(turn_errors[0]).lower()
    assert [payload["index"] for event, payload in events if event == "chunk_scheduled"] == [0, 1]
    assert not any(event == "audio_chunk" for event, _payload in events)
    assert not any(
        thread.name.startswith("ms4-tts") and thread.is_alive()
        for thread in threading.enumerate()
    )


def test_cancellable_speech_request_notifies_after_send_before_response(monkeypatch):
    trace: list[str] = []
    response = (
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: audio/wav\r\n"
        b"Content-Length: 3\r\n\r\n"
        b"WAV"
    )

    class _Socket:
        def __init__(self):
            self.response_sent = False

        def settimeout(self, _timeout):
            pass

        def connect(self, _sockaddr):
            trace.append("connect")

        def send(self, outbound):
            assert "dispatched" not in trace
            trace.append("send")
            return min(7, len(outbound))

        def recv(self, _size):
            assert "dispatched" in trace
            trace.append("recv")
            if not self.response_sent:
                self.response_sent = True
                return response
            return b""

        def shutdown(self, _how):
            pass

        def close(self):
            pass

    monkeypatch.setattr(
        voice,
        "_resolve_addresses_cancellable",
        lambda *_args, **_kwargs: [
            (voice.socket.AF_INET, voice.socket.SOCK_STREAM, 0, "", ("127.0.0.1", 6089))
        ],
    )
    monkeypatch.setattr(voice.socket, "socket", lambda *_args, **_kwargs: _Socket())

    audio, mime, _headers = voice._cancellable_speech_request(
        url="http://127.0.0.1:6089/v1/audio/speech",
        body=b"request-body",
        headers={"Content-Type": "application/json"},
        timeout=1,
        cancel_event=threading.Event(),
        on_request_dispatched=lambda: trace.append("dispatched"),
    )

    assert audio == b"WAV"
    assert mime == "audio/wav"
    assert trace.count("dispatched") == 1
    assert max(index for index, item in enumerate(trace) if item == "send") < trace.index("dispatched")
    assert trace.index("dispatched") < trace.index("recv")
