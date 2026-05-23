"""Voice PTT bridge tests."""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request

import pytest

from machine_spirit_4.gateway import voice


# ----- helpers --------------------------------------------------------------


def _fake_urlopen_factory(handlers):
    """``handlers`` is a list of ``(url_predicate, response_callable)``."""

    class FakeResponse:
        def __init__(self, body, *, status=200, headers=None):
            self._body = body
            self.status = status
            self._headers = headers or {}

        def read(self):
            return self._body

        @property
        def headers(self):
            class _H:
                def __init__(self, hdrs):
                    self._h = hdrs

                def get(self, key, default=None):
                    return self._h.get(key, default)

            return _H(self._headers)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake(request, timeout=None):
        url = request if isinstance(request, str) else request.full_url
        for predicate, responder in handlers:
            if predicate(url):
                return responder(request)
        raise AssertionError(f"unexpected url: {url}")

    return fake, FakeResponse


# ----- check_voice_ready ----------------------------------------------------


def test_check_voice_ready_passes_when_input_ready(monkeypatch):
    fake, FakeResponse = _fake_urlopen_factory([
        (lambda u: "/voice/status" in u, lambda r: FakeResponse(json.dumps({"voice_input_ready": True}).encode("utf-8"))),
    ])
    monkeypatch.setattr(voice.urllib.request, "urlopen", fake)
    payload = voice.check_voice_ready("http://ms3:9080")
    assert payload["voice_input_ready"] is True


def test_check_voice_ready_fails_closed_when_not_ready(monkeypatch):
    fake, FakeResponse = _fake_urlopen_factory([
        (lambda u: "/voice/status" in u, lambda r: FakeResponse(json.dumps({
            "voice_input_ready": False,
            "asr": {"status": "unhealthy", "detail": "No endpoints configured"},
        }).encode("utf-8"))),
    ])
    monkeypatch.setattr(voice.urllib.request, "urlopen", fake)
    with pytest.raises(voice.VoiceUnavailable):
        voice.check_voice_ready("http://ms3:9080")


def test_check_voice_ready_fails_closed_when_ms3_unreachable(monkeypatch):
    def boom(*_args, **_kwargs):
        raise urllib.error.URLError("connection refused")
    monkeypatch.setattr(voice.urllib.request, "urlopen", boom)
    with pytest.raises(voice.VoiceUnavailable):
        voice.check_voice_ready("http://ms3:9080")


# ----- transcribe -----------------------------------------------------------


def test_transcribe_posts_multipart_and_returns_text(monkeypatch):
    seen = {}

    fake, FakeResponse = _fake_urlopen_factory([
        (lambda u: "/v1/audio/transcriptions" in u, lambda r: (
            seen.setdefault("content_type", r.get_header("Content-type")),
            seen.setdefault("body_length", len(r.data)),
            FakeResponse(json.dumps({"text": "hello world"}).encode("utf-8")),
        )[-1]),
    ])
    monkeypatch.setattr(voice.urllib.request, "urlopen", fake)
    result = voice.transcribe(hivemind_url="http://hive:6089", audio=b"binary-audio-bytes")
    assert result["text"] == "hello world"
    assert seen["content_type"].startswith("multipart/form-data; boundary=")
    assert seen["body_length"] > len(b"binary-audio-bytes")


def test_transcribe_refuses_empty_audio():
    with pytest.raises(voice.VoiceRequestError):
        voice.transcribe(hivemind_url="http://hive", audio=b"")


def test_transcribe_refuses_oversized_audio(monkeypatch):
    monkeypatch.setattr(voice, "MAX_AUDIO_BYTES", 100)
    with pytest.raises(voice.VoiceRequestError):
        voice.transcribe(hivemind_url="http://hive", audio=b"x" * 101)


def test_transcribe_surfaces_hivemind_http_error_as_unavailable(monkeypatch):
    def boom(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 503, "Service Unavailable", {}, io.BytesIO(b"asr not provisioned"))
    monkeypatch.setattr(voice.urllib.request, "urlopen", boom)
    with pytest.raises(voice.VoiceUnavailable):
        voice.transcribe(hivemind_url="http://hive", audio=b"x")


# ----- synthesize -----------------------------------------------------------


def test_synthesize_returns_audio_bytes_and_mime(monkeypatch):
    fake, FakeResponse = _fake_urlopen_factory([
        (lambda u: "/v1/audio/speech" in u, lambda r: FakeResponse(b"FAKEAUDIO", headers={"Content-Type": "audio/wav"})),
    ])
    monkeypatch.setattr(voice.urllib.request, "urlopen", fake)
    result = voice.synthesize(hivemind_url="http://hive", text="hello")
    assert result["audio_bytes"] == b"FAKEAUDIO"
    assert result["content_type"] == "audio/wav"
    assert result["audio_base64"]


def test_synthesize_refuses_empty_text():
    with pytest.raises(voice.VoiceRequestError):
        voice.synthesize(hivemind_url="http://hive", text="   ")


def test_synthesize_unavailable_when_empty_body(monkeypatch):
    fake, FakeResponse = _fake_urlopen_factory([
        (lambda u: "/v1/audio/speech" in u, lambda r: FakeResponse(b"", headers={"Content-Type": "audio/wav"})),
    ])
    monkeypatch.setattr(voice.urllib.request, "urlopen", fake)
    with pytest.raises(voice.VoiceUnavailable):
        voice.synthesize(hivemind_url="http://hive", text="hello")


# ----- parse_audio_request --------------------------------------------------


def test_parse_audio_request_handles_raw_audio_body():
    body = b"RAW-WAV-BYTES"
    audio, filename = voice.parse_audio_request("audio/wav", body)
    assert audio == body
    assert filename.endswith(".wav")


def test_parse_audio_request_handles_multipart(tmp_path):
    boundary = "----TestBoundary"
    payload = b"my-audio-payload"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="mic.webm"\r\n'
        f"Content-Type: audio/webm\r\n\r\n"
    ).encode("utf-8") + payload + f"\r\n--{boundary}--\r\n".encode("utf-8")
    audio, filename = voice.parse_audio_request(f"multipart/form-data; boundary={boundary}", body)
    assert audio == payload
    assert filename == "mic.webm"


def test_parse_audio_request_rejects_unknown_content_type():
    with pytest.raises(voice.VoiceRequestError):
        voice.parse_audio_request("application/json", b"{}")


# ----- voice_ptt_turn -------------------------------------------------------


class FakeRunner:
    ms3_url = "http://ms3"
    hivemind_url = "http://hive"

    def chat(self, message, *, session_id=None, model=None):
        return {
            "text": f"echo:{message}",
            "session_id": session_id or "s1",
            "face_lobe_model": {"model_id": "qwen2.5:0.5b", "source": "loaded"},
        }


def test_voice_ptt_turn_chains_asr_chat_tts(monkeypatch):
    fake, FakeResponse = _fake_urlopen_factory([
        (lambda u: "/voice/status" in u, lambda r: FakeResponse(json.dumps({"voice_input_ready": True}).encode("utf-8"))),
        (lambda u: "/v1/audio/transcriptions" in u, lambda r: FakeResponse(json.dumps({"text": "hello"}).encode("utf-8"))),
        (lambda u: "/v1/audio/speech" in u, lambda r: FakeResponse(b"AUDIO", headers={"Content-Type": "audio/wav"})),
    ])
    monkeypatch.setattr(voice.urllib.request, "urlopen", fake)
    result = voice.voice_ptt_turn(runner=FakeRunner(), audio=b"raw-audio")
    assert result.transcript == "hello"
    assert result.reply_text == "echo:hello"
    assert result.reply_audio_base64
    assert result.foreground_model["model_id"] == "qwen2.5:0.5b"


def test_voice_ptt_turn_fails_closed_when_voice_not_ready(monkeypatch):
    fake, FakeResponse = _fake_urlopen_factory([
        (lambda u: "/voice/status" in u, lambda r: FakeResponse(json.dumps({
            "voice_input_ready": False,
            "asr": {"status": "unhealthy"},
        }).encode("utf-8"))),
    ])
    monkeypatch.setattr(voice.urllib.request, "urlopen", fake)
    with pytest.raises(voice.VoiceUnavailable):
        voice.voice_ptt_turn(runner=FakeRunner(), audio=b"raw")


def test_voice_ptt_turn_refuses_empty_transcript(monkeypatch):
    fake, FakeResponse = _fake_urlopen_factory([
        (lambda u: "/voice/status" in u, lambda r: FakeResponse(json.dumps({"voice_input_ready": True}).encode("utf-8"))),
        (lambda u: "/v1/audio/transcriptions" in u, lambda r: FakeResponse(json.dumps({"text": "   "}).encode("utf-8"))),
    ])
    monkeypatch.setattr(voice.urllib.request, "urlopen", fake)
    with pytest.raises(voice.VoiceRequestError):
        voice.voice_ptt_turn(runner=FakeRunner(), audio=b"raw")
