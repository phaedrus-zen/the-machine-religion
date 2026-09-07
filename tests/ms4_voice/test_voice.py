"""Voice PTT bridge tests."""

from __future__ import annotations

import io
import json
import struct
import threading
import time
import urllib.error
import urllib.request
import wave

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


def test_check_voice_ready_falls_back_when_ms3_reports_stale_asr(monkeypatch):
    # MS3 can be reachable but stale/misconfigured while HiveMind ASR is
    # actually healthy. Voice should use the real ASR readiness source and
    # mark MS3-owned identity/speaker features degraded.
    ms3_status = {
        "voice_input_ready": False,
        "asr": {"status": "unhealthy", "detail": "No endpoints configured"},
    }
    fake, FakeResponse = _fake_urlopen_factory([
        (lambda u: "/voice/status" in u, lambda r: FakeResponse(json.dumps(ms3_status).encode("utf-8"))),
    ])
    monkeypatch.setattr(voice.urllib.request, "urlopen", fake)
    import machine_spirit_4.gateway.voice_admin as va
    monkeypatch.setattr(va, "get_provision_status",
                        lambda url, svc, timeout=5: {"healthy": True, "provisioning_state": "running"})

    payload = voice.check_voice_ready("http://ms3:9080", hivemind_url="http://hive:6089")

    assert payload["voice_input_ready"] is True
    assert payload["source"] == "hivemind-asr-fallback"
    assert payload["ms3_reported_not_ready"] is True
    assert payload["ms3_voice_status"] == ms3_status


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
    assert voice._voice_rest_capacity_snapshot() == (1, "scale_response_unverified")


def _passing_capacity_two_proof(**overrides):
    values = {
        "single_ms": 2000.0,
        "concurrent_wall_ms": 2200.0,
        "concurrent_requests": 2,
        "valid_audio_results": 2,
        "warmup_audio_valid": True,
        "single_audio_valid": True,
        "provenance_sample_count": 4,
        "provenance_non_cloud_count": 4,
        "location_policy": "peer_only",
    }
    values.update(overrides)
    return voice._voice_rest_concurrency_verdict(**values)


def _valid_capacity_probe_wav() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16_000)
        writer.writeframes(b"\x00\x00" * 320)
    return output.getvalue()


def _capacity_probe_wav_with_raw_tail() -> bytes:
    return _valid_capacity_probe_wav() + b"EVIL"


def _capacity_probe_wav_with_rewritten_four_byte_evil_tail() -> bytes:
    payload = bytearray(_valid_capacity_probe_wav())
    payload.extend(b"EVIL")
    payload[4:8] = struct.pack("<I", len(payload) - 8)
    return bytes(payload)


def _capacity_probe_wav_with_complete_evil_chunk_after_data() -> bytes:
    payload = bytearray(_valid_capacity_probe_wav())
    payload.extend(b"EVIL" + struct.pack("<I", 0))
    payload[4:8] = struct.pack("<I", len(payload) - 8)
    return bytes(payload)


def _capacity_probe_wav_with_odd_junk_before_data(*, pad: bytes) -> bytes:
    payload = bytearray(_valid_capacity_probe_wav())
    data_offset = payload.index(b"data", 12)
    chunk = b"JUNK" + struct.pack("<I", 1) + b"X" + pad
    payload[data_offset:data_offset] = chunk
    payload[4:8] = struct.pack("<I", len(payload) - 8)
    return bytes(payload)


@pytest.mark.parametrize(
    ("audio_factory", "accepted"),
    [
        (_valid_capacity_probe_wav, True),
        (lambda: _capacity_probe_wav_with_odd_junk_before_data(pad=b"\x00"), True),
        (_capacity_probe_wav_with_raw_tail, False),
        (_capacity_probe_wav_with_rewritten_four_byte_evil_tail, False),
        (_capacity_probe_wav_with_complete_evil_chunk_after_data, False),
        (lambda: _capacity_probe_wav_with_odd_junk_before_data(pad=b"\xff"), False),
        (lambda: _capacity_probe_wav_with_odd_junk_before_data(pad=b""), False),
    ],
)
def test_capacity_probe_wav_parser_is_strict_and_requires_final_data(
    audio_factory,
    accepted,
):
    result = {
        "audio_bytes": audio_factory(),
        "content_type": "audio/wav",
    }

    assert voice._voice_rest_wav_audio_is_valid(result) is accepted


def test_live_serialized_measurement_does_not_admit_capacity_two(monkeypatch):
    """The observed 1954 ms single / 3609 ms n=2 wall path is only 1.08x."""

    monkeypatch.setenv("MS4_VOICE_TTS_CAPACITY", "2")
    proof = _passing_capacity_two_proof(
        single_ms=1954,
        concurrent_wall_ms=3609,
    )

    assert proof["observed_speedup"] == 1.083
    assert proof["minimum_speedup"] == 1.5
    assert proof["passed"] is False
    assert proof["reason"] == "speedup_below_floor"
    assert voice._record_voice_rest_concurrency_observation(proof) == 1
    assert voice._voice_rest_capacity_snapshot() == (1, "concurrency_probe_failed")


def test_measured_capacity_two_expires_fail_closed(monkeypatch):
    now = {"value": 100.0}
    monkeypatch.setenv("MS4_VOICE_TTS_CAPACITY", "2")
    monkeypatch.setenv("MS4_VOICE_TTS_CAPACITY_PROOF_TTL_S", "10")
    monkeypatch.setattr(voice.time, "monotonic", lambda: now["value"])
    proof = _passing_capacity_two_proof()

    assert proof["passed"] is True
    assert voice._record_voice_rest_concurrency_observation(proof) == 2
    assert voice._voice_rest_capacity_snapshot() == (
        2,
        "measured_concurrency_probe",
    )

    now["value"] = 111.0
    assert voice._voice_rest_capacity_snapshot() == (1, "concurrency_proof_stale")


@pytest.mark.parametrize(
    ("policy", "provenance", "accepted"),
    [
        ("peer_only", {"served_by": "hivemind-peer"}, False),
        ("peer_only", {"served_by": "hivemind-peer", "location": "cloud"}, False),
        ("peer_only", {"served_by": "hivemind-peer", "location": "rack-b/gpu-3"}, False),
        ("peer_only", {"served_by": "hivemind-peer", "location": "local"}, False),
        ("peer_only", {"served_by": "azure-tts", "location": "lan"}, False),
        ("peer_only", {"served_by": "hivemind-peer", "location": "lan"}, True),
        ("local_only", {"served_by": "hivemind-local", "location": "ondevice"}, True),
    ],
)
def test_capacity_probe_provenance_requires_explicit_non_cloud_location(
    policy,
    provenance,
    accepted,
):
    assert voice._voice_rest_provenance_is_non_cloud(
        {"provenance": provenance},
        location_policy=policy,
    ) is accepted


def test_capacity_probe_refuses_network_without_explicit_location_policy(monkeypatch):
    calls = []

    def should_not_run(**kwargs):
        calls.append(kwargs)
        raise AssertionError("probe escaped without a strict location policy")

    proof = voice.probe_voice_rest_concurrency(
        hivemind_url="http://hive:6089",
        synthesize_fn=should_not_run,
    )

    assert calls == []
    assert proof["passed"] is False
    assert proof["reason"] == "explicit_non_cloud_location_policy_required"
    assert voice._voice_rest_capacity_snapshot() == (1, "concurrency_probe_failed")


def test_capacity_probe_measures_same_pinned_route_and_admits_two(monkeypatch):
    calls = []
    lock = threading.Lock()
    wav_bytes = _valid_capacity_probe_wav()
    monkeypatch.setenv("MS4_TTS_LOCATION_POLICY", "peer_only")
    monkeypatch.setenv("MS4_VOICE_TTS_CAPACITY", "2")

    def local_synth(**kwargs):
        with lock:
            calls.append((kwargs, voice._tts_location_policy()))
        # Long enough that Windows thread-pool startup jitter cannot dominate
        # the synthetic 2x overlap this fixture is meant to prove.
        time.sleep(0.08)
        return {
            "audio_bytes": wav_bytes,
            "content_type": "audio/wav",
            "provenance": {
                "served_by": "hivemind-test-peer",
                "location": "lan",
                "endpoint": "/v1/audio/speech",
            },
        }

    proof = voice.probe_voice_rest_concurrency(
        hivemind_url="http://hive:6089",
        model="tts-1",
        timeout=1,
        synthesize_fn=local_synth,
    )

    assert len(calls) == 4
    assert all(policy == "peer_only" for _kwargs, policy in calls)
    assert all(kwargs["hivemind_url"] == "http://hive:6089" for kwargs, _ in calls)
    assert all(kwargs["model"] == "tts-1" for kwargs, _ in calls)
    assert all(kwargs["response_format"] == "wav" for kwargs, _ in calls)
    assert all(kwargs["text"].startswith("Oracle capacity probe ") for kwargs, _ in calls)
    assert len({len(kwargs["text"]) for kwargs, _ in calls}) == 1
    assert proof["route"] == "/v1/audio/speech"
    assert proof["location_policy"] == "peer_only"
    assert proof["provenance_non_cloud_count"] == 4
    assert proof["passed"] is True
    assert proof["observed_speedup"] >= 1.5
    assert voice._voice_rest_capacity_snapshot() == (
        2,
        "measured_concurrency_probe",
    )


def test_capacity_probe_rejects_one_mixed_cloud_sample(monkeypatch):
    call_number = 0
    lock = threading.Lock()
    monkeypatch.setenv("MS4_TTS_LOCATION_POLICY", "peer_only")
    monkeypatch.setenv("MS4_VOICE_TTS_CAPACITY", "2")
    wav_bytes = _valid_capacity_probe_wav()

    def mixed_synth(**_kwargs):
        nonlocal call_number
        with lock:
            sample = call_number
            call_number += 1
        time.sleep(0.01)
        return {
            "audio_bytes": wav_bytes,
            "content_type": "audio/wav",
            "provenance": {
                "served_by": "azure-tts" if sample == 2 else "hivemind-test-peer",
                "location": "lan",
            },
        }

    proof = voice.probe_voice_rest_concurrency(
        hivemind_url="http://hive:6089",
        timeout=1,
        synthesize_fn=mixed_synth,
    )

    assert proof["passed"] is False
    assert proof["reason"] == "non_cloud_hli_provenance_required_for_all_samples"
    assert proof["provenance_sample_count"] == 4
    assert proof["provenance_non_cloud_count"] == 3
    assert voice._voice_rest_capacity_snapshot() == (1, "concurrency_probe_failed")


def test_capacity_probe_warms_before_timing_and_rejects_serialized_pair(monkeypatch):
    """A cold first request must not manufacture capacity-two speedup."""

    call_number = 0
    call_lock = threading.Lock()
    serialized_origin = threading.Lock()
    monkeypatch.setenv("MS4_TTS_LOCATION_POLICY", "peer_only")
    monkeypatch.setenv("MS4_VOICE_TTS_CAPACITY", "2")
    wav_bytes = _valid_capacity_probe_wav()

    def cold_then_serialized(**_kwargs):
        nonlocal call_number
        with call_lock:
            sample = call_number
            call_number += 1
        with serialized_origin:
            time.sleep(0.06 if sample == 0 else 0.015)
        return {
            "audio_bytes": wav_bytes,
            "content_type": "audio/wav",
            "provenance": {
                "served_by": "hivemind-single-origin",
                "location": "lan",
            },
        }

    proof = voice.probe_voice_rest_concurrency(
        hivemind_url="http://hive:6089",
        timeout=1,
        synthesize_fn=cold_then_serialized,
    )

    assert call_number == 4
    assert proof["warmup_audio_valid"] is True
    assert proof["single_audio_valid"] is True
    assert proof["observed_speedup"] < 1.5
    assert proof["passed"] is False
    assert proof["reason"] == "speedup_below_floor"
    assert voice._voice_rest_capacity_snapshot() == (1, "concurrency_probe_failed")


@pytest.mark.parametrize(
    ("audio_factory", "content_type"),
    [
        (lambda: b'THIS IS NOT AUDIO', "audio/wav"),
        (lambda: b'{"error":"not audio"}', "application/json"),
        (lambda: b"<html>not audio</html>", "text/html"),
        (lambda: b"RIFF", "audio/wav"),
        (lambda: b"undecodable bytes", "audio/wav"),
        (lambda: _valid_capacity_probe_wav()[:-5], "audio/wav"),
        (lambda: _valid_capacity_probe_wav(), "application/octet-stream"),
        (_capacity_probe_wav_with_raw_tail, "audio/wav"),
        (_capacity_probe_wav_with_rewritten_four_byte_evil_tail, "audio/wav"),
    ],
)
def test_capacity_probe_rejects_corrupt_truncated_or_wrong_type_audio(
    monkeypatch,
    audio_factory,
    content_type,
):
    calls = 0
    monkeypatch.setenv("MS4_TTS_LOCATION_POLICY", "peer_only")
    monkeypatch.setenv("MS4_VOICE_TTS_CAPACITY", "2")

    def invalid_synth(**_kwargs):
        nonlocal calls
        calls += 1
        time.sleep(0.02)
        return {
            "audio_bytes": audio_factory(),
            "content_type": content_type,
            "provenance": {
                "served_by": "hivemind-test-peer",
                "location": "lan",
            },
        }

    proof = voice.probe_voice_rest_concurrency(
        hivemind_url="http://hive:6089",
        timeout=1,
        synthesize_fn=invalid_synth,
    )

    assert calls == 4
    assert proof["passed"] is False
    assert proof["reason"] == "nonempty_audio_required_for_all_samples"
    assert proof["warmup_audio_valid"] is False
    assert proof["single_audio_valid"] is False
    assert proof["valid_audio_results"] == 0
    assert proof["audio_validation"] == "strict_final_data_pcm_wav_v2"
    assert voice._voice_rest_capacity_snapshot() == (1, "concurrency_probe_failed")


def test_capacity_probe_refuses_non_wav_format_before_network(monkeypatch):
    calls = []
    monkeypatch.setenv("MS4_TTS_LOCATION_POLICY", "peer_only")

    proof = voice.probe_voice_rest_concurrency(
        hivemind_url="http://hive:6089",
        response_format="mp3",
        synthesize_fn=lambda **kwargs: calls.append(kwargs),
    )

    assert calls == []
    assert proof["passed"] is False
    assert proof["reason"] == "wav_response_format_required"
    assert voice._voice_rest_capacity_snapshot() == (1, "concurrency_probe_failed")


def test_provision_tts_replicas_records_one_observed_replica_fail_closed(monkeypatch):
    """A requested target of two is not evidence of two usable replicas."""

    def one_replica(_request, timeout=None):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps({
                    "ok": True,
                    "status": "ok",
                    "target": 2,
                    "replicas": 1,
                    "endpoints": [{"port": 49180}],
                }).encode()

        return Response()

    monkeypatch.setenv("MS4_VOICE_TTS_CAPACITY", "2")
    monkeypatch.setattr(voice.urllib.request, "urlopen", one_replica)

    result = voice.provision_tts_replicas(
        hivemind_url="http://hive:6089", target=2, job_type="TTS"
    )

    assert result["replicas"] == 1
    assert voice._voice_rest_capacity_snapshot() == (1, "scale_endpoint")


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
    assert voice._voice_rest_capacity_snapshot() == (1, "fail_closed_default")


def test_prewarm_asr_autoprovisions_before_silence_gate(monkeypatch):
    """Boot prewarm must not be fooled by HiveMind's silence short-circuit.
    If ASR is configured but not running, request the named resource first,
    then use the silent WAV only as a warmup smoke."""
    import machine_spirit_4.gateway.voice_admin as va

    calls = []
    monkeypatch.setenv("MS4_VOICE_ASR_AUTOPROVISION", "1")
    monkeypatch.setenv("MS4_VOICE_ASR_AUTOPROVISION_TIMEOUT_S", "1")
    monkeypatch.setattr(va, "get_provision_status", lambda *_a, **_k: {
        "healthy": False,
        "provisioning_state": "configured_not_provisioned",
    })

    def request(hivemind_url, service, **kwargs):
        calls.append((hivemind_url, service, kwargs))
        return {"status": "provisioning", "service": service}

    monkeypatch.setattr(va, "request_voice_service", request)
    monkeypatch.setattr(va, "poll_until_healthy", lambda *_a, **_k: {
        "healthy": True,
        "provisioning_state": "running",
    })
    monkeypatch.setattr(voice, "transcribe", lambda **_kw: {"text": "", "model": "whisper-1"})

    result = voice.prewarm_asr(hivemind_url="http://hive:6089", model="whisper-1")

    assert calls and calls[0][1] == "ASR"
    assert result["provision_status"]["healthy"] is True


def test_prewarm_asr_skips_autoprovision_when_already_healthy(monkeypatch):
    import machine_spirit_4.gateway.voice_admin as va

    monkeypatch.setenv("MS4_VOICE_ASR_AUTOPROVISION", "1")
    monkeypatch.setattr(va, "get_provision_status", lambda *_a, **_k: {
        "healthy": True,
        "provisioning_state": "running",
    })
    monkeypatch.setattr(
        va,
        "request_voice_service",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("unexpected provision request")),
    )
    monkeypatch.setattr(voice, "transcribe", lambda **_kw: {"text": "", "model": "whisper-1"})

    result = voice.prewarm_asr(hivemind_url="http://hive:6089")

    assert result["provision_status"]["healthy"] is True


def test_face_prewarm_picker_error_never_calls_default_model(monkeypatch):
    """Unexpected picker defects fail closed instead of using a raw default."""
    import machine_spirit_4.double_agent.model_picker as picker_module
    import machine_spirit_4.gateway.face_lobe_chat as face_module

    calls = []

    class FakeFaceLobeChat:
        def __init__(self, **_kwargs):
            pass

        def chat(self, *_args, **kwargs):
            calls.append(kwargs)
            return {"text": "unexpected", "model": kwargs["model"]}

    def picker_boom(**_kwargs):
        raise TypeError("unexpected picker defect")

    monkeypatch.setenv("MS4_DEFAULT_MODEL", "rejected-heavy:70b")
    monkeypatch.setattr(face_module, "FaceLobeChat", FakeFaceLobeChat)
    monkeypatch.setattr(picker_module, "choose_foreground_model", picker_boom)

    result = voice.prewarm_face_lobe_model(hivemind_url="http://hive:6089")

    assert result["warmed"] is False
    assert "unexpected picker defect" in result["error"]
    assert calls == []


def test_face_prewarm_reports_effective_fallback_model(monkeypatch):
    """Prewarm evidence names the model that actually produced the reply."""
    import machine_spirit_4.double_agent.model_picker as picker_module
    import machine_spirit_4.gateway.face_lobe_chat as face_module

    calls = []

    class FakeFaceLobeChat:
        def __init__(self, **_kwargs):
            pass

        def chat(self, *_args, **kwargs):
            calls.append(kwargs["model"])
            return {"text": "ready", "model": "safe-fallback:8b"}

    monkeypatch.setattr(face_module, "FaceLobeChat", FakeFaceLobeChat)
    monkeypatch.setattr(
        picker_module,
        "choose_foreground_model",
        lambda **_kwargs: type("Choice", (), {"model_id": "rejected-heavy:70b"})(),
    )

    result = voice.prewarm_face_lobe_model(hivemind_url="http://hive:6089")

    assert calls == ["rejected-heavy:70b"]
    assert result["warmed"] is True
    assert result["model"] == "safe-fallback:8b"
    assert result["requested_model"] == "rejected-heavy:70b"
    assert result["fallback_used"] is True


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


def test_synthesize_can_pin_only_tts_to_local_hivemind_capacity(monkeypatch):
    seen = {}
    fake, FakeResponse = _fake_urlopen_factory([
        (lambda u: "/v1/audio/speech" in u, lambda r: (
            seen.setdefault("location", r.get_header("X-hivemind-location")),
            FakeResponse(b"FAKEAUDIO", headers={"Content-Type": "audio/wav"}),
        )[-1]),
    ])
    monkeypatch.setenv("MS4_TTS_LOCATION_POLICY", "local-only")
    monkeypatch.setattr(voice.urllib.request, "urlopen", fake)

    result = voice.synthesize(hivemind_url="http://hive", text="hello")

    assert result["audio_bytes"] == b"FAKEAUDIO"
    assert seen["location"] == "local_only"


def test_synthesize_rejects_unknown_tts_location_policy(monkeypatch):
    monkeypatch.setenv("MS4_TTS_LOCATION_POLICY", "surprise_peer")
    with pytest.raises(voice.VoiceRequestError, match="MS4_TTS_LOCATION_POLICY"):
        voice.synthesize(hivemind_url="http://hive", text="hello")


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


def test_voice_brevity_is_legacy_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("MS4_VOICE_BREVITY", raising=False)
    assert voice._voice_brevity_enabled() is False

    monkeypatch.setenv("MS4_VOICE_BREVITY", "1")
    assert voice._voice_brevity_enabled() is True

    monkeypatch.setenv("MS4_VOICE_BREVITY", "off")
    assert voice._voice_brevity_enabled() is False


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


def test_voice_ptt_turn_defaults_to_complete_conversation_mode(monkeypatch):
    monkeypatch.delenv("MS4_VOICE_BREVITY", raising=False)
    monkeypatch.setattr(
        voice,
        "check_voice_ready",
        lambda *_args, **_kwargs: {"voice_input_ready": True},
    )
    monkeypatch.setattr(
        voice,
        "transcribe",
        lambda **_kwargs: {"text": "Explain the tradeoff fully", "model": "whisper-1"},
    )
    monkeypatch.setattr(
        voice,
        "synthesize",
        lambda **_kwargs: {
            "audio_base64": "QUVESU8=",
            "content_type": "audio/wav",
            "model": "tts-1",
        },
    )
    captured: dict[str, object] = {}

    class CapturingRunner(FakeRunner):
        def chat(self, message, *, session_id=None, model=None, **kwargs):
            captured.update(kwargs)
            return super().chat(message, session_id=session_id, model=model)

    voice.voice_ptt_turn(runner=CapturingRunner(), audio=b"raw")

    assert "voice_mode" not in captured


def test_voice_ptt_turn_legacy_brevity_opt_in_sets_voice_mode(monkeypatch):
    monkeypatch.setenv("MS4_VOICE_BREVITY", "1")
    monkeypatch.setattr(
        voice,
        "check_voice_ready",
        lambda *_args, **_kwargs: {"voice_input_ready": True},
    )
    monkeypatch.setattr(
        voice,
        "transcribe",
        lambda **_kwargs: {"text": "Give me the short version", "model": "whisper-1"},
    )
    monkeypatch.setattr(
        voice,
        "synthesize",
        lambda **_kwargs: {
            "audio_base64": "QUVESU8=",
            "content_type": "audio/wav",
            "model": "tts-1",
        },
    )
    captured: dict[str, object] = {}

    class CapturingRunner(FakeRunner):
        def chat(self, message, *, session_id=None, model=None, **kwargs):
            captured.update(kwargs)
            return super().chat(message, session_id=session_id, model=model)

    voice.voice_ptt_turn(runner=CapturingRunner(), audio=b"raw")

    assert captured["voice_mode"] is True


def test_voice_ptt_turn_semantic_incomplete_never_reaches_tts(monkeypatch):
    monkeypatch.setattr(
        voice,
        "check_voice_ready",
        lambda *_args, **_kwargs: {"voice_input_ready": True},
    )
    monkeypatch.setattr(
        voice,
        "transcribe",
        lambda **_kwargs: {
            "text": "Go deeper on the second tradeoff.",
            "model": "whisper-1",
        },
    )
    tts_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        voice,
        "synthesize",
        lambda **kwargs: tts_calls.append(kwargs),
    )

    class IncompleteRunner(FakeRunner):
        def chat(self, *_args, **_kwargs):
            return {
                "text": "",
                "completed": False,
                "cancelled": False,
                "metrics": {
                    "incomplete_reason": "non_substantive_referential_followup"
                },
            }

    with pytest.raises(
        voice.VoiceUnavailable,
        match="voice turn incomplete: non_substantive_referential_followup",
    ):
        voice.voice_ptt_turn(runner=IncompleteRunner(), audio=b"raw")

    assert tts_calls == []


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
def test_voice_ptt_turn_normalizes_spoken_brand_alias_at_shared_boundary(
    monkeypatch, raw_transcript, canonical_transcript
):
    """Non-streaming PTT sends canonical text downstream but retains ASR raw text.

    The tracked boundary assertion prevents a route-local regex from satisfying
    this contract: both PTT paths must call the same shared ASR-result boundary.
    """
    monkeypatch.setattr(
        voice,
        "check_voice_ready",
        lambda *_args, **_kwargs: {"voice_input_ready": True},
    )
    monkeypatch.setattr(
        voice,
        "transcribe",
        lambda **_kwargs: {"text": raw_transcript, "model": "whisper-1"},
    )
    monkeypatch.setattr(
        voice,
        "synthesize",
        lambda **_kwargs: {
            "audio_base64": "QUVESU8=",
            "content_type": "audio/wav",
            "model": "tts-1",
        },
    )

    boundary_calls = []
    original_boundary = getattr(
        voice,
        "_ptt_transcript_from_asr",
        lambda result: (str(result.get("text") or "").strip(), str(result.get("text") or "")),
    )

    def tracked_boundary(result):
        boundary_calls.append(str(result.get("text") or ""))
        return original_boundary(result)

    monkeypatch.setattr(
        voice,
        "_ptt_transcript_from_asr",
        tracked_boundary,
        raising=False,
    )

    downstream = {}

    class CapturingRunner(FakeRunner):
        def chat(self, message, *, session_id=None, model=None, **_):
            downstream["message"] = message
            return {
                "text": "The authoritative result is ready.",
                "session_id": session_id or "s-alias",
                "face_lobe_model": {"model_id": "qwen2.5:0.5b", "source": "loaded"},
            }

    result = voice.voice_ptt_turn(runner=CapturingRunner(), audio=b"raw-audio")

    assert downstream["message"] == canonical_transcript
    assert boundary_calls == [raw_transcript]
    assert result.transcript == canonical_transcript
    assert result.raw_transcript == raw_transcript


# ----- default TTS voice (Oracle regular-TTS Vega default) ------------------
#
# HiveMind/Oracle narration rides the regular REST TTS path (engine stays
# ``rest``, model stays ``tts-1``) with the ``vega`` voice by default. The
# built-in fallback moved alloy -> vega; the MS4_VOICE_TTS_VOICE env override
# must still win. The ms4_voice conftest strips MS4_VOICE_TTS_VOICE before
# every test, so the built-in default is observable without a module reload.


def test_default_tts_voice_builtin_is_vega(monkeypatch):
    """With no env override, the built-in default Oracle TTS voice is vega."""
    monkeypatch.delenv("MS4_VOICE_TTS_VOICE", raising=False)
    assert voice._default_tts_voice() == "vega"


def test_default_tts_voice_env_override_wins(monkeypatch):
    """An explicit MS4_VOICE_TTS_VOICE still overrides the built-in vega default."""
    monkeypatch.setenv("MS4_VOICE_TTS_VOICE", "shimmer")
    assert voice._default_tts_voice() == "shimmer"
