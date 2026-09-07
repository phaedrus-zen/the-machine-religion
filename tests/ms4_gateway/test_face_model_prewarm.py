from __future__ import annotations

import io
import itertools
import json
import threading
import time
import types
from pathlib import Path

from machine_spirit_4.gateway import server as srv_module
from machine_spirit_4.gateway import voice as voice_module
from machine_spirit_4.double_agent.model_picker import ForegroundModelUnavailable


ROOT = Path(__file__).resolve().parents[2]
_CLIENT_COUNTER = itertools.count(1)


def _allow_exact_cluster_admission(monkeypatch, model: str = "selected:4b") -> None:
    monkeypatch.setattr(
        "machine_spirit_4.double_agent.model_picker.require_exact_face_cluster_admission",
        lambda **kwargs: {
            "schema": "hivemind.models.recommend@v1",
            "model": kwargs.get("requested_model") or model,
            "backend": "ollama",
            "owner": "node-a",
            "node": "node-a",
            "endpoint": "http://192.0.2.10:6089/v1/chat/completions",
            "ready": True,
            "snapshot_age_s": 1.0,
            "generation": 1,
            "current_generation": 1,
            "source": "admitted",
            "lease": {
                "id": "test-lease",
                "issued": 1_700_000_000.0 - 5,
                "expires": 1_700_000_000.0 + 30,
                "generation": 1,
            },
        },
    )


def _admitted_result(
    model: str,
    *,
    first_token_ms: int = 1200,
    latency_budget_ms: int = 15_000,
) -> dict:
    return {
        "model": model,
        "requested_model": model,
        "fallback_used": False,
        "reply_len": 5,
        "completed": True,
        "cancelled": False,
        "warmed": True,
        "admission_profile": voice_module.SELECTED_FACE_ADMISSION_PROFILE,
        "first_token_ms": first_token_ms,
        "latency_budget_ms": latency_budget_ms,
        "latency_admitted": first_token_ms <= latency_budget_ms,
    }


class _DummyRunner:
    hivemind_url = "http://hive:6089"


def _make_handler(
    body: object,
    *,
    client_id: str | None = None,
    generation: int = 1,
) -> srv_module.Ms4GatewayHandler:
    if isinstance(body, dict):
        body = {
            "client_id": client_id or f"test-client-{next(_CLIENT_COUNTER)}",
            "generation": generation,
            **body,
        }
    return _make_raw_handler(json.dumps(body).encode("utf-8"))


def _make_raw_handler(raw: bytes) -> srv_module.Ms4GatewayHandler:
    handler = srv_module.Ms4GatewayHandler.__new__(srv_module.Ms4GatewayHandler)
    handler.runner = _DummyRunner()
    handler.command = "POST"
    handler.path = "/voice/prewarm/face"
    headers = {
        "Host": "127.0.0.1:9180",
        "Content-Length": str(len(raw)),
        "Content-Type": "application/json",
    }
    handler.headers = types.SimpleNamespace(get=headers.get)
    handler.rfile = io.BytesIO(raw)
    handler.wfile = io.BytesIO()
    handler.requestline = "POST /voice/prewarm/face HTTP/1.1"
    handler.request_version = "HTTP/1.1"
    handler.client_address = ("127.0.0.1", 0)
    handler.server = types.SimpleNamespace(server_name="test", server_port=0)
    handler.protocol_version = "HTTP/1.1"
    return handler


def _response(handler: srv_module.Ms4GatewayHandler) -> tuple[int, dict]:
    raw = handler.wfile.getvalue()
    status_line, _, rest = raw.partition(b"\r\n")
    status = int(status_line.split(b" ")[1])
    _, _, body = rest.partition(b"\r\n\r\n")
    return status, json.loads(body.decode("utf-8"))


def test_face_prewarm_endpoint_warms_and_records_only_exact_completed_model(monkeypatch):
    calls: list[dict] = []
    recorded: list[str] = []
    events: list[tuple[str, dict]] = []

    def prewarm(**kwargs):
        calls.append(kwargs)
        return _admitted_result("ministral-3:latest_ollama")

    monkeypatch.setattr(srv_module, "prewarm_face_lobe_model", prewarm)
    monkeypatch.setattr(srv_module, "_selected_face_prewarm_timeout_s", lambda: 12.0)
    monkeypatch.setattr(srv_module, "record_face_model", recorded.append)
    monkeypatch.setattr(srv_module, "append_event", lambda name, data: events.append((name, data)))

    handler = _make_handler({"model": "ministral-3:latest_ollama"})
    handler.do_POST()
    status, body = _response(handler)

    assert status == 200
    assert body == {
        "warmed": True,
        "requested_model": "ministral-3:latest_ollama",
        "resolved_requested_model": "ministral-3:latest_ollama",
        "effective_model": "ministral-3:latest_ollama",
        "fallback_used": False,
        "reply_len": 5,
        "admission_profile": voice_module.SELECTED_FACE_ADMISSION_PROFILE,
        "first_token_ms": 1200,
        "latency_budget_ms": 15_000,
        "latency_admitted": True,
        "generation": 1,
    }
    assert recorded == ["ministral-3:latest_ollama"]
    assert events[0][0] == "face_model_prewarmed"
    assert len(calls) == 1
    assert calls[0]["model"] == "ministral-3:latest_ollama"
    assert calls[0]["timeout"] == 7
    assert calls[0]["exact_model"] is True
    assert calls[0]["latency_budget_ms"] == 15_000
    assert hasattr(calls[0]["cancel_event"], "set")
    assert hasattr(calls[0]["cancel_event"], "is_set")


def test_face_prewarm_endpoint_rejects_fallback_without_recording(monkeypatch):
    recorded: list[str] = []
    monkeypatch.setattr(
        srv_module,
        "prewarm_face_lobe_model",
        lambda **_kwargs: {
            **_admitted_result("selected:4b"),
            "model": "fallback:4b",
            "fallback_used": True,
        },
    )
    monkeypatch.setattr(srv_module, "record_face_model", recorded.append)
    monkeypatch.setattr(srv_module, "append_event", lambda *_args, **_kwargs: None)

    handler = _make_handler({"model": "selected:4b"})
    handler.do_POST()
    status, body = _response(handler)

    assert status == 503
    assert body["warmed"] is False
    assert body["fallback_used"] is True
    assert body["effective_model"] == "fallback:4b"
    assert body["fail_closed"] is True
    assert recorded == []


def test_face_prewarm_endpoint_rejects_incomplete_or_cancelled_result(monkeypatch):
    for completed, cancelled in ((False, False), (True, True)):
        monkeypatch.setattr(
            srv_module,
            "prewarm_face_lobe_model",
            lambda **_kwargs: {
                **_admitted_result("selected:4b"),
                "completed": completed,
                "cancelled": cancelled,
            },
        )
        monkeypatch.setattr(srv_module, "record_face_model", lambda _model: None)
        monkeypatch.setattr(srv_module, "append_event", lambda *_args, **_kwargs: None)
        handler = _make_handler({"model": "selected:4b"})
        handler.do_POST()
        status, body = _response(handler)
        assert status == 503
        assert body["warmed"] is False
        assert body["fail_closed"] is True


def test_face_prewarm_endpoint_rejects_missing_or_over_budget_latency(monkeypatch):
    recorded: list[str] = []
    events: list[tuple[str, dict]] = []
    results = (
        {
            key: value
            for key, value in _admitted_result("selected:4b").items()
            if key != "first_token_ms"
        },
        _admitted_result("selected:4b", first_token_ms=15_001),
        {
            **_admitted_result("selected:4b"),
            "latency_admitted": False,
        },
        {
            **_admitted_result("selected:4b"),
            "admission_profile": "stale-profile",
        },
        {**_admitted_result("selected:4b"), "first_token_ms": True},
        {**_admitted_result("selected:4b"), "first_token_ms": "1200"},
        {**_admitted_result("selected:4b"), "first_token_ms": 1200.0},
        {**_admitted_result("selected:4b"), "first_token_ms": -1},
        {**_admitted_result("selected:4b"), "latency_budget_ms": True},
        {**_admitted_result("selected:4b"), "latency_budget_ms": "15000"},
        {**_admitted_result("selected:4b"), "latency_budget_ms": 15000.0},
        {**_admitted_result("selected:4b"), "latency_budget_ms": 0},
        {**_admitted_result("selected:4b"), "latency_budget_ms": 14_999},
    )
    monkeypatch.setattr(srv_module, "record_face_model", recorded.append)
    monkeypatch.setattr(
        srv_module,
        "append_event",
        lambda name, data: events.append((name, data)),
    )

    for result in results:
        monkeypatch.setattr(
            srv_module,
            "prewarm_face_lobe_model",
            lambda **_kwargs: result,
        )
        handler = _make_handler({"model": "selected:4b"})
        handler.do_POST()
        status, body = _response(handler)

        assert status == 503
        assert body["warmed"] is False
        assert body["latency_admitted"] is False
        assert body["fail_closed"] is True

    assert recorded == []
    assert all(name == "face_model_prewarm_failed" for name, _data in events)


def test_face_prewarm_endpoint_validates_request_shape(monkeypatch):
    monkeypatch.setattr(
        srv_module,
        "prewarm_face_lobe_model",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("must not warm")),
    )
    for body in (
        {},
        {"model": None},
        {"model": ""},
        {"model": "   "},
        {"model": 7},
        {"model": "ok:4b", "unexpected": True},
        {"model": "x" * 257},
        {"operation": "prewarm"},
        {"operation": "invalidate", "model": "must-not-warm:4b"},
        {"operation": None},
        {"operation": False},
        {"operation": ""},
        {"operation": 7},
        {"model": "ok:4b", "operation": None},
        {"model": "ok:4b", "operation": False},
        {"model": "ok:4b", "operation": ""},
        {"model": "ok:4b", "operation": 7},
    ):
        handler = _make_handler(body)
        handler.do_POST()
        status, payload = _response(handler)
        assert status == 400
        assert payload["warmed"] is False
        assert payload["fail_closed"] is True

    duplicate = _make_raw_handler(
        b'{"model":"a:4b","model":"b:4b","client_id":"dup","generation":1}'
    )
    duplicate.do_POST()
    status, payload = _response(duplicate)
    assert status == 400
    assert "duplicate field: model" in payload["error"]

    duplicate_operation = _make_raw_handler(
        b'{"operation":"invalidate","operation":"invalidate",'
        b'"client_id":"dup-operation","generation":1}'
    )
    duplicate_operation.do_POST()
    status, payload = _response(duplicate_operation)
    assert status == 400
    assert "duplicate field: operation" in payload["error"]


def test_face_prewarm_endpoint_timeout_cancels_and_retires_worker(monkeypatch):
    cancel_seen = threading.Event()
    worker_retired = threading.Event()
    recorded: list[str] = []

    def blocking_prewarm(**kwargs):
        cancel_event = kwargs["cancel_event"]
        cancel_event.wait(timeout=1.0)
        if cancel_event.is_set():
            cancel_seen.set()
        worker_retired.set()
        return {"warmed": False, "cancelled": True}

    monkeypatch.setattr(srv_module, "prewarm_face_lobe_model", blocking_prewarm)
    monkeypatch.setattr(srv_module, "_selected_face_prewarm_timeout_s", lambda: 0.05)
    monkeypatch.setattr(srv_module, "record_face_model", recorded.append)
    monkeypatch.setattr(srv_module, "append_event", lambda *_args, **_kwargs: None)

    handler = _make_handler({"model": "selected:4b"})
    started = time.monotonic()
    handler.do_POST()
    elapsed = time.monotonic() - started
    status, body = _response(handler)

    assert elapsed < 1.5
    assert status == 504
    assert body["warmed"] is False
    assert body["worker_retired"] is True
    assert body["fail_closed"] is True
    assert cancel_seen.is_set()
    assert worker_retired.is_set()
    assert recorded == []


def test_new_generation_cancels_server_worker_and_stale_model_cannot_commit(monkeypatch):
    a_started = threading.Event()
    release_a = threading.Event()
    a_cancel_events: list[threading.Event] = []
    recorded: list[str] = []

    def prewarm(**kwargs):
        model = kwargs["model"]
        if model == "model-a:4b":
            a_cancel_events.append(kwargs["cancel_event"])
            a_started.set()
            assert release_a.wait(timeout=2.0)
        return _admitted_result(model)

    monkeypatch.setattr(srv_module, "prewarm_face_lobe_model", prewarm)
    monkeypatch.setattr(srv_module, "record_face_model", recorded.append)
    monkeypatch.setattr(srv_module, "append_event", lambda *_args, **_kwargs: None)

    client_id = f"server-race-{next(_CLIENT_COUNTER)}"
    handler_a = _make_handler(
        {"model": "model-a:4b"},
        client_id=client_id,
        generation=1,
    )
    a_thread = threading.Thread(target=handler_a.do_POST)
    a_thread.start()
    assert a_started.wait(timeout=1.0)

    handler_b = _make_handler(
        {"model": "model-b:4b"},
        client_id=client_id,
        generation=2,
    )
    handler_b.do_POST()
    status_b, body_b = _response(handler_b)

    assert status_b == 200
    assert body_b["effective_model"] == "model-b:4b"
    assert a_cancel_events and a_cancel_events[0].is_set()
    assert recorded == ["model-b:4b"]


def test_auto_invalidation_cancels_inflight_exact_warm_and_retains_tombstone(monkeypatch):
    a_started = threading.Event()
    release_a = threading.Event()
    a_cancel_events: list[threading.Event] = []
    prewarmed: list[str] = []
    recorded: list[str] = []

    def prewarm(**kwargs):
        model = kwargs["model"]
        prewarmed.append(model)
        if model == "model-a:4b":
            a_cancel_events.append(kwargs["cancel_event"])
            a_started.set()
            assert release_a.wait(timeout=2.0)
        return _admitted_result(model)

    monkeypatch.setattr(srv_module, "prewarm_face_lobe_model", prewarm)
    monkeypatch.setattr(srv_module, "record_face_model", recorded.append)
    monkeypatch.setattr(srv_module, "append_event", lambda *_args, **_kwargs: None)

    client_id = f"server-auto-race-{next(_CLIENT_COUNTER)}"
    handler_a = _make_handler(
        {"model": "model-a:4b"},
        client_id=client_id,
        generation=1,
    )
    a_thread = threading.Thread(target=handler_a.do_POST)
    a_thread.start()
    assert a_started.wait(timeout=1.0)

    invalidate = _make_handler(
        {"operation": "invalidate"},
        client_id=client_id,
        generation=2,
    )
    invalidate.do_POST()
    invalidate_status, invalidate_body = _response(invalidate)

    assert invalidate_status == 200
    assert invalidate_body == {
        "warmed": False,
        "invalidated": True,
        "automatic": True,
        "operation": "invalidate",
        "generation": 2,
    }
    assert a_cancel_events and a_cancel_events[0].is_set()
    assert prewarmed == ["model-a:4b"]
    assert recorded == []

    release_a.set()
    a_thread.join(timeout=2.0)
    assert not a_thread.is_alive()
    status_a, body_a = _response(handler_a)
    assert status_a == 409
    assert body_a["superseded"] is True
    assert recorded == []

    replay = _make_handler(
        {"operation": "invalidate"},
        client_id=client_id,
        generation=2,
    )
    replay.do_POST()
    replay_status, replay_body = _response(replay)
    assert replay_status == 409
    assert replay_body["operation"] == "invalidate"
    assert replay_body["invalidated"] is False
    assert replay_body["latest_generation"] == 2
    assert prewarmed == ["model-a:4b"]
    assert recorded == []

    handler_b = _make_handler(
        {"model": "model-b:4b"},
        client_id=client_id,
        generation=3,
    )
    handler_b.do_POST()
    status_b, body_b = _response(handler_b)
    assert status_b == 200
    assert body_b["effective_model"] == "model-b:4b"
    assert prewarmed == ["model-a:4b", "model-b:4b"]
    assert recorded == ["model-b:4b"]

    release_a.set()
    a_thread.join(timeout=2.0)
    assert not a_thread.is_alive()
    status_a, body_a = _response(handler_a)
    assert status_a == 409
    assert body_a["superseded"] is True
    assert recorded == ["model-b:4b"]


def test_exact_voice_prewarm_fails_closed_without_typed_cluster_admission(monkeypatch):
    import machine_spirit_4.gateway.face_lobe_chat as face_module

    class FakeFaceLobeChat:
        def chat(self, *_args, **_kwargs):
            raise AssertionError("chat must not run without typed cluster admission")

        def probe_first_token(self, *_args, **_kwargs):
            raise AssertionError("probe must not run without typed cluster admission")

        def __init__(self, **_kwargs):
            pass

    monkeypatch.setattr(face_module, "FaceLobeChat", FakeFaceLobeChat)
    monkeypatch.setattr(
        "machine_spirit_4.double_agent.model_picker.require_exact_face_cluster_admission",
        lambda **_kwargs: (_ for _ in ()).throw(
            ForegroundModelUnavailable("cluster recommendation admission absent")
        ),
    )
    result = voice_module.prewarm_face_lobe_model(
        hivemind_url="http://hive:6089",
        model="selected:4b",
        exact_model=True,
    )
    assert result["warmed"] is False
    assert "admission" in str(result.get("error") or "")
    assert result.get("latency_admitted") is False


def test_exact_voice_prewarm_disables_fallback_and_requires_completion(monkeypatch):
    import machine_spirit_4.gateway.face_lobe_chat as face_module

    _allow_exact_cluster_admission(monkeypatch)
    constructor_calls: list[dict] = []
    chat_calls: list[dict] = []
    probe_calls: list[dict] = []
    cancellation = threading.Event()

    class FakeFaceLobeChat:
        def __init__(self, **kwargs):
            constructor_calls.append(kwargs)

        def chat(self, *_args, **kwargs):
            chat_calls.append(kwargs)
            return {
                "text": "ready",
                "model": kwargs["model"],
                "completed": True,
                "cancelled": False,
                "fallback_used": False,
            }

        def probe_first_token(self, *_args, **kwargs):
            probe_calls.append(kwargs)
            return {
                "text": "r",
                "model": kwargs["model"],
                "requested_model": kwargs["model"],
                "first_token_observed": True,
                "first_token_ms": 1200,
                "visible_tokens": 1,
                "completed": True,
                "metrics": {"stream_first_token_ms": 1200},
            }

    monkeypatch.setattr(face_module, "FaceLobeChat", FakeFaceLobeChat)
    result = voice_module.prewarm_face_lobe_model(
        hivemind_url="http://hive:6089",
        model="selected:4b",
        timeout=17,
        exact_model=True,
        cancel_event=cancellation,
        latency_budget_ms=4000,
    )

    assert result["warmed"] is True
    assert result["completed"] is True
    assert result["fallback_used"] is False
    assert result["admission_profile"] == voice_module.SELECTED_FACE_ADMISSION_PROFILE
    assert result["first_token_ms"] == 1200
    assert result["latency_budget_ms"] == 4000
    assert result["latency_admitted"] is True
    assert constructor_calls == [{
        "hivemind_url": "http://hive:6089",
        "http_timeout": 17,
        "allow_model_fallback": False,
    }]
    assert callable(chat_calls[0]["stream_callback"])
    assert chat_calls[0]["cancel_event"] is cancellation
    assert len(probe_calls) == 1
    assert probe_calls[0]["model"] == "selected:4b"
    assert probe_calls[0]["cancel_event"] is cancellation
    assert probe_calls[0]["first_token_timeout"] == 4.0
    assert probe_calls[0]["max_visible_tokens"] == 1
    assert voice_module.SELECTED_FACE_ADMISSION_PROFILE in probe_calls[0]["extra_system"]
    assert len(probe_calls[0]["extra_system"].split()) >= 650


def test_exact_voice_prewarm_does_not_admit_incomplete_result(monkeypatch):
    import machine_spirit_4.gateway.face_lobe_chat as face_module

    _allow_exact_cluster_admission(monkeypatch)

    class FakeFaceLobeChat:
        def __init__(self, **_kwargs):
            pass

        def chat(self, *_args, **kwargs):
            return {
                "text": "ready",
                "model": kwargs["model"],
                "completed": False,
                "cancelled": False,
                "fallback_used": False,
            }

        def probe_first_token(self, *_args, **_kwargs):
            raise AssertionError("incomplete residency warm must not run admission probe")

    monkeypatch.setattr(face_module, "FaceLobeChat", FakeFaceLobeChat)
    result = voice_module.prewarm_face_lobe_model(
        hivemind_url="http://hive:6089",
        model="selected:4b",
        exact_model=True,
    )

    assert result["warmed"] is False
    assert result["completed"] is False


def test_exact_voice_prewarm_does_not_admit_zero_reply(monkeypatch):
    import machine_spirit_4.gateway.face_lobe_chat as face_module

    _allow_exact_cluster_admission(monkeypatch)

    class FakeFaceLobeChat:
        def __init__(self, **_kwargs):
            pass

        def chat(self, *_args, **kwargs):
            return {
                "text": "",
                "model": kwargs["model"],
                "completed": True,
                "cancelled": False,
                "fallback_used": False,
            }

        def probe_first_token(self, *_args, **_kwargs):
            raise AssertionError("zero-content residency warm must not run admission probe")

    monkeypatch.setattr(face_module, "FaceLobeChat", FakeFaceLobeChat)
    result = voice_module.prewarm_face_lobe_model(
        hivemind_url="http://hive:6089",
        model="selected:4b",
        exact_model=True,
    )

    assert result["warmed"] is False
    assert result["reply_len"] == 0


def test_exact_voice_prewarm_rejects_missing_or_over_budget_probe(monkeypatch):
    import machine_spirit_4.gateway.face_lobe_chat as face_module

    _allow_exact_cluster_admission(monkeypatch)

    probe_results = (
        {"text": "", "model": "selected:4b", "first_token_ms": None},
        {
            "text": "r",
            "model": "selected:4b",
            "first_token_observed": True,
            "visible_tokens": 1,
            "first_token_ms": 4001,
            "completed": True,
        },
        {
            "text": "r",
            "model": "wrong:4b",
            "first_token_observed": True,
            "visible_tokens": 1,
            "first_token_ms": 1000,
            "completed": True,
        },
    )

    for probe_result in probe_results:
        class FakeFaceLobeChat:
            def __init__(self, **_kwargs):
                pass

            def chat(self, *_args, **kwargs):
                return {
                    "text": "ready",
                    "model": kwargs["model"],
                    "completed": True,
                    "cancelled": False,
                    "fallback_used": False,
                }

            def probe_first_token(self, *_args, **_kwargs):
                return dict(probe_result)

        monkeypatch.setattr(face_module, "FaceLobeChat", FakeFaceLobeChat)
        result = voice_module.prewarm_face_lobe_model(
            hivemind_url="http://hive:6089",
            model="selected:4b",
            exact_model=True,
            latency_budget_ms=4000,
        )

        assert result["warmed"] is False
        assert result["latency_admitted"] is False
        assert result["latency_budget_ms"] == 4000


def test_selected_face_admission_context_is_cache_busted_and_production_sized():
    message_a, context_a = voice_module._selected_face_admission_probe_payload()
    message_b, context_b = voice_module._selected_face_admission_probe_payload()

    assert message_a == message_b
    assert context_a != context_b
    assert voice_module.SELECTED_FACE_ADMISSION_PROFILE in context_a
    assert "VOICE MODE" in context_a
    assert "MS4 Face Lobe context" in context_a
    assert len(context_a.split()) >= 650


def test_selected_face_admission_budget_is_explicit_and_below_voice_watchdog(monkeypatch):
    monkeypatch.delenv("MS4_FACE_SELECTED_ADMISSION_TTFT_MS", raising=False)
    monkeypatch.setattr(voice_module, "_voice_first_token_timeout", lambda: 20.0)
    assert voice_module.selected_face_admission_latency_budget_ms() == 15_000

    monkeypatch.setenv("MS4_FACE_SELECTED_ADMISSION_TTFT_MS", "90000")
    assert voice_module.selected_face_admission_latency_budget_ms() == 19000

    monkeypatch.setenv("MS4_FACE_SELECTED_ADMISSION_TTFT_MS", "not-a-number")
    assert voice_module.selected_face_admission_latency_budget_ms() == 15_000


def test_browser_prewarms_restored_and_new_face_selection():
    html = (ROOT / "machine_spirit_4" / "web" / "index.html").read_text(encoding="utf-8")
    for required in (
        "startSelectedFaceModelPrewarm('restored selection')",
        "startSelectedFaceModelPrewarm('model selection')",
        "fetch('/voice/prewarm/face'",
        "data.resolved_requested_model === model",
        "data.effective_model === model",
        "data.fallback_used === false",
        "data.admission_profile === FACE_MODEL_ADMISSION_PROFILE",
        "data.latency_admitted === true",
        "data.first_token_ms <= data.latency_budget_ms",
        "client_id: String(oracleVoicePageInstanceId)",
        "generation: state.generation",
        "data.generation === state.generation",
        "selectedFacePrewarmState !== state",
        "selectedFaceModelId() !== model",
        "state.controller.abort()",
        "if (!modelCatalogReady && modelCatalogReadyPromise) await modelCatalogReadyPromise",
        "frozen_model: model",
        "voiceQueryParams({faceModel: faceAdmission.frozen_model})",
        "saved selection after catalog failure",
        "(saved; catalog unavailable)",
        "operation: 'invalidate'",
        "data.invalidated === true",
        "data.automatic === true",
        "FACE_MODEL_INVALIDATION_TIMEOUT_MS = 5000",
        "if (state && state.model !== '')",
        "Automatic Face selection was not admitted within 5 seconds.",
    ):
        assert required in html


def test_all_face_turn_paths_wait_before_sending_inference():
    html = (ROOT / "machine_spirit_4" / "web" / "index.html").read_text(encoding="utf-8")
    blocking = html.split("async function sendBlockingChat(", 1)[1].split(
        "function markStreamingAssistantIncomplete", 1
    )[0]
    streaming = html.split("async function sendStreamingChat(", 1)[1].split(
        "document.getElementById('chatForm')", 1
    )[0]
    voice = html.split("async function submitWavBlobAsVoiceTurn(", 1)[1].split(
        "// SECOND parallel /voice/turn/stream", 1
    )[0]

    assert blocking.index("await awaitSelectedFaceModelPrewarm()") < blocking.index("fetch('/chat'")
    assert streaming.index("await awaitSelectedFaceModelPrewarm()") < streaming.index("fetch('/chat/stream'")
    assert voice.index("await awaitSelectedFaceModelPrewarm()") < voice.index("/voice/turn/stream")
    assert "The voice turn was stopped before inference, speech, or history commit." in voice


def test_native_capture_hook_awaits_exact_created_turn_state():
    html = (ROOT / "machine_spirit_4" / "web" / "index.html").read_text(encoding="utf-8")
    submit = html.split("async function submitWavBlobAsVoiceTurn(", 1)[1].split(
        "// Mic press = barge-in", 1
    )[0]
    capture = html.split("function runCapture(prompt, runMarker)", 1)[1].split(
        "var hooks =", 1
    )[0]

    assert "onTurnStateCreated(turnState)" in submit
    assert submit.index("activeVoiceTurnState = turnState") < submit.index(
        "onTurnStateCreated(turnState)"
    )
    assert "onTurnStateCreated: resolve" in capture
    assert "Promise.all([turnStateP, submitP])" in capture
    assert "turnState = values[0]" in capture
    assert "turnState = activeVoiceTurnState" not in capture
