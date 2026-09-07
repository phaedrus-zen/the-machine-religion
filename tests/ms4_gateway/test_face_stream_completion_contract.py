"""Regression contract for truthful Face-Lobe streaming completion.

Partial bytes are useful for diagnostics and may already have reached the
browser, but they are not a successful assistant turn.  These tests keep the
transport boundary honest: only an explicit ``[DONE]`` marker or a normal
``finish_reason=stop`` may become committed conversation history.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable
from typing import Any

import pytest

from machine_spirit_4.gateway.face_lobe_chat import FaceLobeChat


def _sse_event(
    content: str = "",
    *,
    finish_reason: str | None = None,
    warning: dict[str, Any] | None = None,
) -> bytes:
    event: dict[str, Any] = {
        "choices": [
            {
                "delta": {"content": content},
                "finish_reason": finish_reason,
            }
        ]
    }
    if warning is not None:
        event["hivemind_warning"] = warning
    return f"data: {json.dumps(event)}\n\n".encode("utf-8")


class _Socket:
    def __init__(self) -> None:
        self.timeout: float | None = None

    def settimeout(self, timeout: float) -> None:
        self.timeout = timeout


class _Raw:
    def __init__(self, sock: _Socket) -> None:
        self._sock = sock


class _File:
    def __init__(self, sock: _Socket) -> None:
        self.raw = _Raw(sock)


class _Response:
    """Small urllib-compatible response with scriptable read outcomes."""

    def __init__(self, *reads: bytes | BaseException) -> None:
        self.socket = _Socket()
        self.fp = _File(self.socket)
        self._reads = list(reads)
        self.closed = False

    def readline(self) -> bytes:
        if not self._reads:
            return b""
        outcome = self._reads.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def close(self) -> None:
        self.closed = True


def _run_stream(
    monkeypatch: pytest.MonkeyPatch,
    response: _Response,
    *,
    callback: Callable[[str], Any] | None = None,
) -> tuple[str, dict[str, Any], list[str], list[tuple[str, str]]]:
    monkeypatch.setattr(urllib.request, "urlopen", lambda *_args, **_kwargs: response)
    face = FaceLobeChat(
        hivemind_url="http://hive.test:6089",
        http_timeout=30,
        stream_stall_timeout=2,
    )
    cancel_calls: list[tuple[str, str]] = []

    def cancel(trace_id: str, *, reason: str) -> bool:
        cancel_calls.append((trace_id, reason))
        return True

    monkeypatch.setattr(face, "_cancel_upstream_trace", cancel)
    captured: list[str] = []
    cb = callback or (lambda fragment: captured.append(fragment))
    text, stats = face._post_streaming(
        {"model": "face-test", "messages": [], "stream": True},
        cb,
    )
    return text, stats, captured, cancel_calls


@pytest.mark.parametrize(
    ("terminal", "expected_text"),
    [
        (b"data: [DONE]\n\n", "complete answer"),
        (_sse_event("", finish_reason="stop"), "complete answer"),
    ],
    ids=("done-marker", "normal-stop"),
)
def test_explicit_normal_terminal_marks_stream_complete(
    monkeypatch: pytest.MonkeyPatch,
    terminal: bytes,
    expected_text: str,
) -> None:
    response = _Response(_sse_event(expected_text), terminal)

    text, stats, captured, cancel_calls = _run_stream(monkeypatch, response)

    assert text == expected_text
    assert captured == [expected_text]
    assert stats["stream_completed"] is True
    assert stats.get("incomplete_reason") in (None, "")
    assert cancel_calls == []
    assert response.closed is True


def test_eof_without_terminal_is_partial_not_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _Response(_sse_event("useful but partial"), b"")

    text, stats, captured, cancel_calls = _run_stream(monkeypatch, response)

    assert text == "useful but partial"
    assert captured == ["useful but partial"]
    assert stats["stream_completed"] is False
    assert "eof" in str(stats["incomplete_reason"]).lower()
    assert len(cancel_calls) == 1
    assert cancel_calls[0][0] == stats["trace_id"]


def test_read_timeout_is_partial_and_cancels_exact_upstream_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _Response(_sse_event("partial"), TimeoutError("read timed out"))

    text, stats, _captured, cancel_calls = _run_stream(monkeypatch, response)

    assert text == "partial"
    assert stats["stream_completed"] is False
    assert "timeout" in str(stats["incomplete_reason"]).lower()
    assert cancel_calls == [
        (stats["trace_id"], "MS4 Face Lobe stream read timeout")
    ]
    assert stats["upstream_canceled"] is True


def test_length_finish_reason_is_truncation_not_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _Response(_sse_event("cut off", finish_reason="length"))

    text, stats, captured, _cancel_calls = _run_stream(monkeypatch, response)

    assert text == "cut off"
    assert captured == ["cut off"]
    assert stats["stream_completed"] is False
    assert "length" in str(stats["incomplete_reason"]).lower()


def test_reasoning_truncation_warning_is_not_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _Response(
        _sse_event(
            "reasoning consumed the budget",
            finish_reason="stop",
            warning={"type": "reasoning_model_truncated"},
        )
    )

    text, stats, _captured, _cancel_calls = _run_stream(monkeypatch, response)

    assert text == "reasoning consumed the budget"
    assert stats["stream_completed"] is False
    assert "reasoning_model_truncated" in str(stats["incomplete_reason"])


def test_downstream_cancel_is_not_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _Response(
        _sse_event("first"),
        _sse_event(" second"),
        b"data: [DONE]\n\n",
    )
    received: list[str] = []

    def cancel_after_first(fragment: str) -> bool:
        received.append(fragment)
        return False

    text, stats, _captured, cancel_calls = _run_stream(
        monkeypatch,
        response,
        callback=cancel_after_first,
    )

    assert text == "first"
    assert received == ["first"]
    assert stats["stream_completed"] is False
    assert "downstream" in str(stats["incomplete_reason"]).lower()
    assert cancel_calls == [
        (stats["trace_id"], "MS4 Face Lobe downstream canceled stream")
    ]


def test_incomplete_stream_never_commits_a_conversation_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    face = FaceLobeChat(hivemind_url="http://hive.test:6089")
    emitted: list[str] = []

    def incomplete_transport(*_args: Any, **_kwargs: Any) -> tuple[str, dict[str, Any]]:
        emitted.append("partial answer")
        return (
            "partial answer",
            {
                "stream_completed": False,
                "incomplete_reason": "eof_before_terminal",
                "http_latency_ms": 1,
                "bytes_received": 14,
                "stream_chunks": 1,
                "stream_first_token_ms": 1,
            },
        )

    monkeypatch.setattr(face, "_post_streaming", incomplete_transport)

    result = face.chat(
        "question",
        session_id="truth-session",
        model="face-test",
        stream_callback=lambda _fragment: True,
    )

    assert result["text"] == "partial answer"
    assert result["completed"] is False
    assert result["cancelled"] is False
    assert result["metrics"]["stream_completed"] is False
    assert result["metrics"]["incomplete_reason"] == "eof_before_terminal"
    assert face._sessions["truth-session"].messages == []


def test_done_marker_with_zero_visible_content_is_not_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _Response(b"data: [DONE]\n\n")

    text, stats, captured, _cancel_calls = _run_stream(monkeypatch, response)

    assert text == ""
    assert captured == []
    assert stats["stream_completed"] is False
    assert "zero visible content" in str(stats["incomplete_reason"]).lower()


def test_malformed_frame_cannot_be_laundered_by_done_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _Response(
        b'data: {"choices":[BROKEN}\n\n',
        _sse_event("apparently complete"),
        b"data: [DONE]\n\n",
    )

    text, stats, captured, _cancel_calls = _run_stream(monkeypatch, response)

    assert text == "apparently complete"
    assert captured == ["apparently complete"]
    assert stats["stream_completed"] is False
    assert stats["malformed_frames"] == 1
    assert "malformed" in str(stats["incomplete_reason"]).lower()
