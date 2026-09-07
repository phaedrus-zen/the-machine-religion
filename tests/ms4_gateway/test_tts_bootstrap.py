from types import SimpleNamespace

from machine_spirit_4.gateway import server as srv


class _DoubleAgentRunner:
    def recover_on_startup(self) -> None:
        pass


class _NoopHttpServer:
    def __init__(self, _address, _handler) -> None:
        pass

    def serve_forever(self) -> None:
        pass


def _capture_boot_threads(
    monkeypatch,
    *,
    model: str,
    autoscale: str = "1",
    replica_target: str = "2",
    capacity_probe=None,
    location_policy: str = "peer_only",
):
    captured = []

    class CapturedThread:
        def __init__(
            self,
            *,
            target=None,
            args=(),
            kwargs=None,
            daemon=None,
            name=None,
        ) -> None:
            self.target = target
            self.args = args
            self.kwargs = kwargs or {}
            self.daemon = daemon
            self.name = name
            self.started = False
            captured.append(self)

        def start(self) -> None:
            self.started = True

        def run(self) -> None:
            assert self.started
            assert self.target is not None
            self.target(*self.args, **self.kwargs)

    monkeypatch.setattr(srv, "DEFAULT_TTS_MODEL", model, raising=False)
    monkeypatch.setattr(srv, "require_contained_runtime", lambda _service: None)
    monkeypatch.setattr(
        srv,
        "build_runner",
        lambda: SimpleNamespace(
            hivemind_url="http://hive:6089",
            ms3_url="http://ms3:9080",
        ),
    )
    monkeypatch.setattr(srv.hermes_admin, "initialize_state", lambda: None)
    monkeypatch.setattr(srv, "default_runner", _DoubleAgentRunner)
    monkeypatch.setattr(srv, "_set_hivemind_url_for_auth", lambda _url: None)
    monkeypatch.setattr(srv, "start_heartbeat_thread", lambda _url: None)
    monkeypatch.setattr(srv, "ThreadingHTTPServer", _NoopHttpServer)
    monkeypatch.setattr(srv.threading, "Thread", CapturedThread)
    monkeypatch.setattr(
        srv,
        "probe_voice_rest_concurrency",
        capacity_probe
        or (lambda **_kwargs: {
            "passed": False,
            "reason": "test_noop",
            "observed_speedup": 0,
            "minimum_speedup": 1.5,
            "location_policy": None,
            "provenance_non_cloud_count": 0,
            "provenance_sample_count": 0,
        }),
    )
    monkeypatch.setenv("MS4_DA_CONTINUATION_CLASSIFIER", "0")
    monkeypatch.setenv("MS4_VOICE_TTS_AUTOSCALE", autoscale)
    monkeypatch.setenv("MS4_VOICE_TTS_AUTOSCALE_SECS", "0")
    monkeypatch.setenv("MS4_VOICE_TTS_REPLICA_TARGET", replica_target)
    monkeypatch.setenv("MS4_TTS_LOCATION_POLICY", location_policy)
    monkeypatch.setenv("MS4_VOICE_TTS_KEEPWARM_SECS", "0")
    monkeypatch.setenv("MS4_VOICE_FACE_KEEPWARM_SECS", "0")

    srv.run(port=0)
    return captured


def _run_initial_selected_pool_tasks(threads, *, job_type: str) -> None:
    selected_names = {"ms4-tts-autoscale"}
    if job_type == "TTS":
        selected_names.add("ms4-tts-prewarm")
    else:
        selected_names.add("ms4-tts-super-prewarm")
    for thread in threads:
        if thread.name in selected_names:
            thread.run()


def test_regular_oracle_default_scales_then_warms_reported_replicas(monkeypatch):
    events = []

    def scale(*, hivemind_url, target=None, job_type="TTS_SUPER", **_kwargs):
        events.append(("scale", job_type, target))
        return {"ok": True, "status": "ok", "replicas": 2, "target": 2}

    def warm(*, model=None, **_kwargs):
        events.append(("warm", model or srv.DEFAULT_TTS_MODEL))
        return {"audio_bytes": b"RIFF"}

    monkeypatch.setattr(srv, "provision_tts_replicas", scale)
    monkeypatch.setattr(srv, "prewarm_tts", warm)
    threads = _capture_boot_threads(monkeypatch, model="tts-1")

    _run_initial_selected_pool_tasks(threads, job_type="TTS")

    assert events == [
        ("scale", "TTS", 2),
        ("warm", "tts-1"),
        ("warm", "tts-1"),
    ]


def test_capacity_probe_runs_only_after_selected_pool_is_warm(monkeypatch):
    events = []

    def scale(*, job_type="TTS_SUPER", **_kwargs):
        events.append(("scale", job_type))
        return {"ok": True, "status": "ok", "replicas": 2, "target": 2}

    def warm(*, model=None, **_kwargs):
        events.append(("warm", model))
        return {"audio_bytes": b"RIFF"}

    def probe(**kwargs):
        events.append(("probe", kwargs["model"], kwargs["response_format"]))
        return {
            "passed": True,
            "reason": "measured_capacity_two",
            "observed_speedup": 1.7,
            "minimum_speedup": 1.5,
            "location_policy": "peer_only",
            "provenance_non_cloud_count": 4,
            "provenance_sample_count": 4,
        }

    monkeypatch.setattr(srv, "provision_tts_replicas", scale)
    monkeypatch.setattr(srv, "prewarm_tts", warm)
    threads = _capture_boot_threads(
        monkeypatch,
        model="tts-1",
        capacity_probe=probe,
    )

    _run_initial_selected_pool_tasks(threads, job_type="TTS")

    assert events == [
        ("scale", "TTS"),
        ("warm", "tts-1"),
        ("warm", "tts-1"),
        ("probe", "tts-1", srv.DEFAULT_TTS_FORMAT),
    ]


def test_peer_only_measures_existing_route_when_local_scale_reports_zero(monkeypatch):
    events = []

    def scale(**_kwargs):
        events.append(("scale",))
        return {"ok": False, "status": "ok", "replicas": 0, "target": 2}

    def warm(**_kwargs):
        events.append(("warm",))
        return {"audio_bytes": b"RIFF"}

    def probe(**_kwargs):
        events.append(("probe",))
        return {
            "passed": False,
            "reason": "speedup_below_floor",
            "observed_speedup": 1.08,
            "minimum_speedup": 1.5,
            "location_policy": "peer_only",
            "provenance_non_cloud_count": 4,
            "provenance_sample_count": 4,
        }

    monkeypatch.setattr(srv, "provision_tts_replicas", scale)
    monkeypatch.setattr(srv, "prewarm_tts", warm)
    threads = _capture_boot_threads(
        monkeypatch,
        model="tts-1",
        capacity_probe=probe,
        location_policy="peer_only",
    )

    _run_initial_selected_pool_tasks(threads, job_type="TTS")

    assert events == [("scale",), ("warm",), ("probe",)]


def test_local_only_requires_two_scale_candidates_before_measurement(monkeypatch):
    events = []

    def scale(**_kwargs):
        events.append(("scale",))
        return {"ok": True, "status": "ok", "replicas": 1, "target": 2}

    def warm(**_kwargs):
        events.append(("warm",))
        return {"audio_bytes": b"RIFF"}

    def probe(**_kwargs):
        events.append(("probe",))
        raise AssertionError("local capacity-one route must not be measured as n=2")

    monkeypatch.setattr(srv, "provision_tts_replicas", scale)
    monkeypatch.setattr(srv, "prewarm_tts", warm)
    threads = _capture_boot_threads(
        monkeypatch,
        model="tts-1",
        capacity_probe=probe,
        location_policy="local_only",
    )

    _run_initial_selected_pool_tasks(threads, job_type="TTS")

    assert events == [("scale",), ("warm",)]


def test_initial_regular_prewarm_runs_after_unsuccessful_scale(monkeypatch):
    events = []

    def scale(*, job_type="TTS_SUPER", **_kwargs):
        events.append(("scale", job_type))
        return {"ok": False, "status": "unavailable"}

    def warm(**_kwargs):
        events.append(("warm",))
        return {"audio_bytes": b"RIFF"}

    monkeypatch.setattr(srv, "provision_tts_replicas", scale)
    monkeypatch.setattr(srv, "prewarm_tts", warm)
    threads = _capture_boot_threads(monkeypatch, model="tts-1")

    _run_initial_selected_pool_tasks(threads, job_type="TTS")

    assert events == [
        ("scale", "TTS"),
        ("warm",),
    ]


def test_initial_regular_prewarm_runs_after_scale_exception(monkeypatch):
    events = []

    def scale(*, job_type="TTS_SUPER", **_kwargs):
        events.append(("scale", job_type))
        raise RuntimeError("scale unavailable")

    def warm(**_kwargs):
        events.append(("warm",))
        return {"audio_bytes": b"RIFF"}

    monkeypatch.setattr(srv, "provision_tts_replicas", scale)
    monkeypatch.setattr(srv, "prewarm_tts", warm)
    threads = _capture_boot_threads(monkeypatch, model="tts-1")

    _run_initial_selected_pool_tasks(threads, job_type="TTS")

    assert events == [
        ("scale", "TTS"),
        ("warm",),
    ]


def test_selected_super_pool_prewarms_when_autoscale_disabled(monkeypatch):
    events = []

    def scale(**_kwargs):
        events.append(("scale",))
        return {"ok": True, "status": "ok", "replicas": 2, "target": 2}

    def warm_super(**_kwargs):
        events.append(("warm", "TTS_SUPER"))
        return {"warmed": True, "bytes_received": 16}

    monkeypatch.setattr(srv, "provision_tts_replicas", scale)
    monkeypatch.setattr(srv, "prewarm_tts_super_ws", warm_super)
    threads = _capture_boot_threads(
        monkeypatch,
        model="tts-1-hd",
        autoscale="0",
    )

    _run_initial_selected_pool_tasks(threads, job_type="TTS_SUPER")

    assert events == [("warm", "TTS_SUPER")]


def test_super_model_scales_tts_super_before_selected_prewarm(monkeypatch):
    events = []

    def scale(*, job_type="TTS_SUPER", **_kwargs):
        events.append(("scale", job_type))
        return {"ok": True, "status": "ok", "replicas": 2}

    def warm_super(**_kwargs):
        events.append(("warm", "TTS_SUPER"))
        return {"warmed": True, "bytes_received": 16}

    monkeypatch.setattr(srv, "provision_tts_replicas", scale)
    monkeypatch.setattr(srv, "prewarm_tts_super_ws", warm_super)
    threads = _capture_boot_threads(monkeypatch, model="tts-1-hd")

    _run_initial_selected_pool_tasks(threads, job_type="TTS_SUPER")

    assert events == [
        ("scale", "TTS_SUPER"),
        ("warm", "TTS_SUPER"),
    ]


def test_regular_prewarm_reports_warm_only_for_nonempty_audio(monkeypatch, capsys):
    monkeypatch.setattr(
        srv,
        "provision_tts_replicas",
        lambda **_kwargs: {"ok": True, "status": "ok", "replicas": 2},
    )
    monkeypatch.setattr(srv, "prewarm_tts", lambda **_kwargs: {"audio_bytes": b""})
    threads = _capture_boot_threads(monkeypatch, model="tts-1")

    _run_initial_selected_pool_tasks(threads, job_type="TTS")

    output = capsys.readouterr().out
    assert "'warmed': False" in output
    assert "'warmed': True" not in output
