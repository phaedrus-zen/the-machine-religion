import io
import json
import threading
import types

from machine_spirit_4.gateway import server as srv_module
from machine_spirit_4.gateway import voice as voice_module


class _Runner:
    hivemind_url = "http://hive:6089"


def _handler(text: str) -> srv_module.Ms4GatewayHandler:
    body = json.dumps({"text": text}).encode("utf-8")
    handler = srv_module.Ms4GatewayHandler.__new__(srv_module.Ms4GatewayHandler)
    handler.runner = _Runner()
    handler.command = "POST"
    handler.path = "/voice/synthesize/stream"
    headers = {
        "Host": "127.0.0.1:9180",
        "Content-Length": str(len(body)),
        "Content-Type": "application/json",
    }
    handler.headers = types.SimpleNamespace(get=headers.get)
    handler.rfile = io.BytesIO(body)
    handler.wfile = io.BytesIO()
    handler.requestline = "POST /voice/synthesize/stream HTTP/1.1"
    handler.request_version = "HTTP/1.1"
    handler.client_address = ("127.0.0.1", 0)
    handler.server = types.SimpleNamespace(server_name="test", server_port=0)
    handler.protocol_version = "HTTP/1.1"
    return handler


def _capture_sse(monkeypatch):
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(srv_module, "_sse_start", lambda _handler: None)
    monkeypatch.setattr(
        srv_module,
        "_sse_event",
        lambda _handler, event, payload, **_kwargs: (
            events.append((event, dict(payload))),
            True,
        )[1],
    )
    return events


def test_text_synthesis_stream_internal_cancel_emits_one_error_and_no_done(monkeypatch):
    events = _capture_sse(monkeypatch)

    def cancel_inside_helper(*, cancel_event, **_kwargs):
        cancel_event.set()
        return 1

    monkeypatch.setattr(srv_module, "_emit_text_as_parallel_chunks", cancel_inside_helper)
    handler = _handler("A complete sentence for synthesis.")
    handler._voice_synthesize_stream()

    terminal = [(event, payload) for event, payload in events if event in {"done", "error"}]
    assert len(terminal) == 1
    assert terminal[0][0] == "error"
    assert terminal[0][1]["completed"] is False
    assert not any(event == "done" for event, _payload in events)
    assert handler.close_connection is True


def test_text_synthesis_stream_missing_chunk_fails_closed(monkeypatch):
    events = _capture_sse(monkeypatch)
    calls = 0

    def synthesize_with_missing_middle(*, text, **_kwargs):
        nonlocal calls
        calls += 1
        audio = b"" if calls == 2 else b"WAV"
        return {"audio_bytes": audio, "content_type": "audio/wav"}

    monkeypatch.setattr(voice_module, "synthesize", synthesize_with_missing_middle)
    monkeypatch.setattr(voice_module, "_runtime_audio_gate_enabled", lambda: False)
    handler = _handler(
        "The first sentence contains enough words to stay separate. "
        "The second sentence contains enough words to stay separate. "
        "The third sentence contains enough words to stay separate."
    )
    handler._voice_synthesize_stream()

    audio = [payload for event, payload in events if event == "audio_chunk"]
    errors = [payload for event, payload in events if event == "audio_error"]
    terminal = [(event, payload) for event, payload in events if event in {"done", "error"}]
    assert [payload["index"] for payload in audio] == [0]
    assert len(errors) == 1 and errors[0]["index"] == 1
    assert len(terminal) == 1 and terminal[0][0] == "error"
    assert terminal[0][1]["completed"] is False
    assert not any(event == "done" for event, _payload in events)


def test_text_synthesis_stream_write_failure_nacks_and_cancels_worker(monkeypatch):
    writes: list[str] = []
    observed: dict[str, bool] = {}

    def helper(*, emit, cancel_event, **_kwargs):
        observed["accepted"] = emit(
            "audio_chunk",
            {
                "index": 0,
                "text": "first chunk",
                "audio_base64": "V0FW",
                "audio_mime": "audio/wav",
            },
        )
        observed["cancelled"] = cancel_event.is_set()
        return 0

    def sse(_handler, event, _payload, **_kwargs):
        writes.append(event)
        return event != "audio_chunk"

    monkeypatch.setattr(srv_module, "_emit_text_as_parallel_chunks", helper)
    monkeypatch.setattr(srv_module, "_sse_start", lambda _handler: None)
    monkeypatch.setattr(srv_module, "_sse_event", sse)
    handler = _handler("A complete sentence for synthesis.")
    handler._voice_synthesize_stream()

    assert observed == {"accepted": False, "cancelled": True}
    assert writes.count("audio_chunk") == 1
    assert writes.count("done") == 0
    assert not any(
        thread.name == "ms4-text-synthesis-stream" and thread.is_alive()
        for thread in threading.enumerate()
    )
