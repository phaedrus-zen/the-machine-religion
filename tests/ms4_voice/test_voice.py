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


@pytest.fixture(autouse=True)
def _clear_ready_cache():
    # check_voice_ready caches successes (15s TTL) so the MS3 round-trip
    # is off the per-turn critical path. Tests reuse the same ms3_url, so
    # clear the cache before each one or a cached "ready" leaks across.
    voice._clear_voice_ready_cache()
    yield
    voice._clear_voice_ready_cache()


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
    # No hivemind_url -> no fallback -> still fails closed (unchanged).
    def boom(*_args, **_kwargs):
        raise urllib.error.URLError("connection refused")
    monkeypatch.setattr(voice.urllib.request, "urlopen", boom)
    with pytest.raises(voice.VoiceUnavailable):
        voice.check_voice_ready("http://ms3:9080")


def test_check_voice_ready_falls_back_to_hivemind_asr_when_ms3_down(monkeypatch):
    # MS3 crashed (connection refused) but HiveMind ASR is healthy ->
    # voice proceeds rather than dying with the identity sidecar.
    def boom(*_a, **_k):
        raise urllib.error.URLError("[WinError 10061] actively refused")
    monkeypatch.setattr(voice.urllib.request, "urlopen", boom)
    import machine_spirit_4.gateway.voice_admin as va
    monkeypatch.setattr(va, "get_provision_status",
                        lambda url, svc, timeout=5: {"healthy": True, "provisioning_state": "running"})
    payload = voice.check_voice_ready("http://ms3:9080", hivemind_url="http://hive:6089")
    assert payload["voice_input_ready"] is True
    assert payload["source"] == "hivemind-asr-fallback"
    assert payload["ms3_unreachable"] is True


def test_check_voice_ready_caches_success_off_critical_path(monkeypatch):
    """A successful readiness check is cached so the MS3 round-trip is
    paid once per conversation, not every turn. The second call inside
    the TTL must NOT hit MS3 again."""
    calls = {"n": 0}
    _, FakeResponse = _fake_urlopen_factory([])

    def counting(url, *a, **k):
        calls["n"] += 1
        return FakeResponse(json.dumps({"voice_input_ready": True}).encode("utf-8"))

    monkeypatch.setenv("MS4_VOICE_READY_CACHE_S", "15")
    monkeypatch.setattr(voice.urllib.request, "urlopen", counting)
    p1 = voice.check_voice_ready("http://ms3:9080")
    p2 = voice.check_voice_ready("http://ms3:9080")
    assert p1["voice_input_ready"] is True and p2["voice_input_ready"] is True
    assert calls["n"] == 1, f"second call should hit cache, not MS3 (got {calls['n']} round-trips)"


def test_check_voice_ready_does_not_cache_failure(monkeypatch):
    """A 'not ready' result is never cached — recovery must be detected
    on the very next turn."""
    monkeypatch.setenv("MS4_VOICE_READY_CACHE_S", "15")
    state = {"ready": False}
    _, FakeResponse = _fake_urlopen_factory([])

    def variable(url, *a, **k):
        return FakeResponse(json.dumps({"voice_input_ready": state["ready"]}).encode("utf-8"))

    monkeypatch.setattr(voice.urllib.request, "urlopen", variable)
    with pytest.raises(voice.VoiceUnavailable):
        voice.check_voice_ready("http://ms3:9080")
    # MS3 recovers — next call must see it (failure wasn't cached).
    state["ready"] = True
    assert voice.check_voice_ready("http://ms3:9080")["voice_input_ready"] is True


# ----- provision_tts_replicas (TTS scale-out) -------------------------------


def test_provision_tts_replicas_parses_replica_payload(monkeypatch):
    """POST /provision/tts/scale success returns the replica count so MS4's
    parallel chunk synthesis fans across one GPU per replica."""
    fake, FakeResponse = _fake_urlopen_factory([
        (lambda u: "/provision/tts/scale" in u,
         lambda r: FakeResponse(json.dumps({
             "status": "ok", "replicas": 2, "target": 2,
             "endpoints": [{"gpus": ["0"], "port": 49180}, {"gpus": ["1"], "port": 49181}],
         }).encode("utf-8"))),
    ])
    monkeypatch.setattr(voice.urllib.request, "urlopen", fake)
    res = voice.provision_tts_replicas(hivemind_url="http://hive:6089")
    assert res["ok"] is True
    assert res["replicas"] == 2


def test_provision_tts_replicas_sends_target_for_packing(monkeypatch):
    """When a target is given (pack multiple GIMs per idle GPU), it must
    be forwarded in the POST body so HiveMind scales past GPU count."""
    seen = {}

    def capture(request, timeout=None):
        seen["body"] = json.loads(request.data.decode("utf-8"))

        class R:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return json.dumps({"status": "ok", "replicas": 4, "target": 4}).encode()
        return R()

    monkeypatch.setattr(voice.urllib.request, "urlopen", capture)
    res = voice.provision_tts_replicas(hivemind_url="http://hive:6089", target=4)
    assert seen["body"]["target"] == 4
    assert seen["body"]["job_type"] == "TTS_SUPER"
    assert res["replicas"] == 4


def test_provision_tts_replicas_softfails_on_older_gateway(monkeypatch):
    """A gateway without the scale endpoint (404) must NOT raise — the
    single TTS replica still serves voice."""
    def not_found(request, timeout=None):
        raise urllib.error.HTTPError(
            request.full_url if hasattr(request, "full_url") else "u",
            404, "Not Found", {}, io.BytesIO(b"no such route"),
        )

    monkeypatch.setattr(voice.urllib.request, "urlopen", not_found)
    res = voice.provision_tts_replicas(hivemind_url="http://hive:6089")
    assert res["ok"] is False
    assert res["status"] == 404


def test_check_voice_ready_fails_closed_when_ms3_down_and_asr_unhealthy(monkeypatch):
    # MS3 down AND HiveMind ASR not healthy -> genuinely can't do voice.
    def boom(*_a, **_k):
        raise urllib.error.URLError("connection refused")
    monkeypatch.setattr(voice.urllib.request, "urlopen", boom)
    import machine_spirit_4.gateway.voice_admin as va
    monkeypatch.setattr(va, "get_provision_status",
                        lambda url, svc, timeout=5: {"healthy": False, "provisioning_state": "unknown"})
    with pytest.raises(voice.VoiceUnavailable):
        voice.check_voice_ready("http://ms3:9080", hivemind_url="http://hive:6089")


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

    def chat(self, message, *, session_id=None, model=None, **_):
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
