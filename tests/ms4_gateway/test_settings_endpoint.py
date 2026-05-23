"""Settings endpoints: GET /settings and POST /settings/clear-grounding-cache.

These are read by the UI's ⚙ Settings dialog so the operator can:
- see what env-defaults the server is honoring (voice engine, TTS
  model/voice, grounding cache TTL, ASR readiness, Hermes version)
- clear the grounding cache without restarting the gateway

Tests use BaseHTTPRequestHandler-style direct construction with a fake
runner + fake wfile, mirroring the existing handler tests.
"""

from __future__ import annotations

import io
import json
import types

import pytest

import machine_spirit_4.gateway.context as ctx_module
import machine_spirit_4.gateway.voice as voice_module
from machine_spirit_4.gateway import server as srv_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummyRunner:
    ms3_url = "http://ms3:9080"
    hivemind_url = "http://hive:6089"
    default_model = "qwen3-coder-next:latest"
    face_lobe_chat = types.SimpleNamespace(_sessions={})


def _make_handler(method: str, path: str, body: bytes = b"") -> srv_module.Ms4GatewayHandler:
    """Construct a Ms4GatewayHandler attached to in-memory rfile/wfile
    streams, bypassing the BaseHTTPRequestHandler init that wants a
    real socket."""
    handler = srv_module.Ms4GatewayHandler.__new__(srv_module.Ms4GatewayHandler)
    handler.runner = _DummyRunner()
    handler.command = method
    handler.path = path
    headers = {"Host": "127.0.0.1:9180"}
    if body:
        headers["Content-Length"] = str(len(body))
        headers["Content-Type"] = "application/json"
    handler.headers = types.SimpleNamespace(get=headers.get)
    handler.rfile = io.BytesIO(body)
    handler.wfile = io.BytesIO()
    handler.requestline = f"{method} {path} HTTP/1.1"
    handler.request_version = "HTTP/1.1"
    handler.client_address = ("127.0.0.1", 0)
    handler.server = types.SimpleNamespace(server_name="test", server_port=0)
    handler.protocol_version = "HTTP/1.1"
    # send_response / send_header / end_headers all write to wfile.
    return handler


def _parse_handler_response(handler) -> tuple[int, dict]:
    """Pull the HTTP status + JSON body back out of the in-memory wfile."""
    raw = handler.wfile.getvalue()
    status_line, _, rest = raw.partition(b"\r\n")
    status = int(status_line.split(b" ")[1])
    head, _, body = rest.partition(b"\r\n\r\n")
    return status, json.loads(body.decode("utf-8") or "{}")


# ---------------------------------------------------------------------------
# GET /settings
# ---------------------------------------------------------------------------


def test_settings_get_returns_env_defaults_and_cache_snapshot(monkeypatch):
    # Force-known env defaults so we don't depend on the actual env.
    monkeypatch.setattr(voice_module, "DEFAULT_ENGINE", "ws_super")
    monkeypatch.setattr(voice_module, "DEFAULT_TTS_VOICE", "alloy")
    monkeypatch.setattr(voice_module, "DEFAULT_TTS_MODEL", "tts-1")
    monkeypatch.setattr(voice_module, "DEFAULT_TTS_FORMAT", "wav")
    monkeypatch.setattr(voice_module, "DEFAULT_TRANSCRIBE_MODEL", "whisper-1")
    monkeypatch.setattr(voice_module, "FIRST_CHUNK_MIN_WORDS", 4)
    monkeypatch.setattr(ctx_module, "GROUNDING_CACHE_TTL_SECS", 60)
    monkeypatch.delenv("MS4_HIVEMIND_API_KEY", raising=False)
    monkeypatch.delenv("MS4_HIVEMIND_MCP_URL", raising=False)
    # Skip the MS3 voice readiness probe to keep the test offline.
    monkeypatch.setattr(voice_module, "check_voice_ready",
                        lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("offline")))
    # Seed a fake grounding cache entry.
    ctx_module.clear_grounding_cache()
    ctx_module._cache_put("inventory::http://hive:6089", "fake-inventory")

    handler = _make_handler("GET", "/settings")
    handler._settings_get()
    status, body = _parse_handler_response(handler)

    assert status == 200
    assert body["schema"] == "Ms4Settings.v1"
    assert body["voice"]["tts_engine_default"] == "ws_super"
    assert body["voice"]["tts_voice_default"] == "alloy"
    assert body["voice"]["tts_model_default"] == "tts-1"
    assert "alloy" in body["voice"]["tts_voice_known"]
    assert "ws_super" in body["voice"]["tts_engine_choices"]
    assert body["voice"]["asr_ready"]["ready"] is False
    assert body["grounding"]["cache_ttl_secs"] == 60
    assert "inventory::http://hive:6089" in body["grounding"]["cache_entries"]
    assert body["endpoints"]["hivemind"] == "http://hive:6089"
    # New surface — direct MCP URL + auth status — must show up so
    # the UI can render the HiveMind cluster-state panel.
    assert body["endpoints"]["hivemind_mcp"].endswith(":6105")
    assert body["auth"]["hivemind_auth_configured"] is False


def test_settings_get_reports_auth_configured_when_env_set(monkeypatch):
    monkeypatch.setenv("MS4_HIVEMIND_API_KEY", "test-token")
    monkeypatch.setattr(voice_module, "check_voice_ready",
                        lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("offline")))
    handler = _make_handler("GET", "/settings")
    handler._settings_get()
    status, body = _parse_handler_response(handler)
    assert status == 200
    assert body["auth"]["hivemind_auth_configured"] is True


# ---------------------------------------------------------------------------
# POST /settings/clear-grounding-cache
# ---------------------------------------------------------------------------


def test_settings_clear_grounding_cache_empties_cache(monkeypatch):
    ctx_module.clear_grounding_cache()
    ctx_module._cache_put("inventory::http://hive:6089", "fake-inventory")
    ctx_module._cache_put("tools::http://ms3:9080::http://hive:6089", "fake-tools")
    handler = _make_handler("POST", "/settings/clear-grounding-cache")
    handler._settings_clear_grounding_cache()
    status, body = _parse_handler_response(handler)

    assert status == 200
    assert body["ok"] is True
    assert body["cleared"] == 2
    assert sorted(body["entries_before"]) == sorted([
        "inventory::http://hive:6089",
        "tools::http://ms3:9080::http://hive:6089",
    ])
    # Cache must actually be empty now.
    with ctx_module._GROUNDING_CACHE_LOCK:
        assert ctx_module._GROUNDING_CACHE == {}


# ---------------------------------------------------------------------------
# /voice/turn/stream ?engine= override is wired through
# ---------------------------------------------------------------------------


def test_voice_services_get_proxies_voice_admin(monkeypatch):
    """GET /voice/services calls voice_admin.list_voice_services and
    returns the snapshot under a stable schema the UI consumes."""
    fake_services = [
        {"schema": "Ms4VoiceServiceStatus.v1", "service": "ASR", "healthy": False,
         "detail": "No endpoints configured", "endpoint": "",
         "endpoints_configured": 1, "provisioning_state": "configured_not_provisioned",
         "http_status": 200, "raw": {}},
        {"schema": "Ms4VoiceServiceStatus.v1", "service": "TTS", "healthy": False,
         "detail": "No endpoints configured", "endpoint": "",
         "endpoints_configured": 1, "provisioning_state": "configured_not_provisioned",
         "http_status": 200, "raw": {}},
        {"schema": "Ms4VoiceServiceStatus.v1", "service": "TTS_SUPER", "healthy": True,
         "detail": "running", "endpoint": "http://hive:49153/gim/tts_super",
         "endpoints_configured": 4, "provisioning_state": "running",
         "http_status": 200, "raw": {}},
    ]
    monkeypatch.setattr(srv_module, "list_voice_services", lambda *_a, **_kw: fake_services)
    handler = _make_handler("GET", "/voice/services")
    handler._voice_services_get()
    status, body = _parse_handler_response(handler)
    assert status == 200
    assert body["schema"] == "Ms4VoiceServicesSnapshot.v1"
    assert [s["service"] for s in body["services"]] == ["ASR", "TTS", "TTS_SUPER"]
    assert body["managed_services"] == ["ASR", "TTS", "TTS_SUPER"]


def test_voice_service_provision_returns_provision_id(monkeypatch):
    """POST /voice/services/ASR/provision calls request_voice_service
    and returns 202 with the HiveMind provision_id surfaced for the UI
    to poll on."""
    monkeypatch.setattr(srv_module, "request_voice_service",
                        lambda *_a, **_kw: {"service": "ASR", "status": "provisioning",
                                            "provision_id": "abc-123", "backend": "gim", "model": "whisper-1"})
    handler = _make_handler("POST", "/voice/services/ASR/provision", body=b"{}")
    handler._voice_service_provision("ASR")
    status, body = _parse_handler_response(handler)
    assert status == 202
    assert body["schema"] == "Ms4VoiceProvisionAck.v1"
    assert body["service"] == "ASR"
    assert body["result"]["provision_id"] == "abc-123"
    assert body["poll_url"] == "/voice/services"


def test_voice_service_provision_unknown_service_returns_400(monkeypatch):
    monkeypatch.setattr(srv_module, "request_voice_service",
                        lambda *_a, **_kw: (_ for _ in ()).throw(
                            srv_module.VoiceServiceUnknown("unknown voice service 'NOPE'")))
    handler = _make_handler("POST", "/voice/services/NOPE/provision", body=b"{}")
    handler._voice_service_provision("NOPE")
    status, body = _parse_handler_response(handler)
    assert status == 400
    assert "unknown voice service" in body["error"]


def test_voice_service_release_returns_200(monkeypatch):
    monkeypatch.setattr(srv_module, "release_voice_service",
                        lambda *_a, **_kw: {"service": "ASR", "status": "released"})
    handler = _make_handler("POST", "/voice/services/ASR/release", body=b'{"provision_id": "abc"}')
    handler._voice_service_release("ASR")
    status, body = _parse_handler_response(handler)
    assert status == 200
    assert body["schema"] == "Ms4VoiceReleaseAck.v1"
    assert body["service"] == "ASR"


def test_reflexes_list_returns_catalog_snapshot(monkeypatch):
    """GET /reflexes proxies to canned_reflexes.list_reflexes and
    returns the Ms4ReflexCatalog.v1 schema the UI consumes."""
    fake_catalog = {
        "schema": "Ms4ReflexCatalog.v1", "voice": "alloy", "default_voice": "alloy",
        "total": 3, "available": 1, "total_bytes": 1234,
        "categories": ["ack", "error"],
        "reflexes": [
            {"id": "ack_mhm", "text": "Mhm?", "category": "ack", "voice": "alloy",
             "url": "/reflexes/alloy/ack_mhm.wav", "available": True, "size_bytes": 1234, "description": ""},
            {"id": "ack_yes", "text": "Yes?", "category": "ack", "voice": "alloy",
             "url": "/reflexes/alloy/ack_yes.wav", "available": False, "size_bytes": 0, "description": ""},
            {"id": "err_cluster_unreachable", "text": "Trouble.", "category": "error", "voice": "alloy",
             "url": "/reflexes/alloy/err_cluster_unreachable.wav", "available": False, "size_bytes": 0, "description": ""},
        ],
    }
    monkeypatch.setattr(srv_module, "list_reflexes", lambda **_kw: fake_catalog)
    handler = _make_handler("GET", "/reflexes")
    handler._reflexes_list()
    status, body = _parse_handler_response(handler)
    assert status == 200
    assert body["schema"] == "Ms4ReflexCatalog.v1"
    assert body["available"] == 1
    assert any(r["available"] for r in body["reflexes"])


def test_reflexes_serve_returns_audio_bytes(monkeypatch):
    """GET /reflexes/<voice>/<id>.wav returns audio/wav bytes."""
    monkeypatch.setattr(srv_module, "read_reflex", lambda **_kw: b"RIFF\x00\x00\x00\x00WAVE")
    handler = _make_handler("GET", "/reflexes/alloy/ack_mhm.wav")
    handler._reflexes_serve("alloy/ack_mhm.wav")
    raw = handler.wfile.getvalue()
    head, _, body = raw.partition(b"\r\n\r\n")
    head_str = head.decode("latin1", errors="replace")
    assert "200 OK" in head_str
    assert "audio/wav" in head_str
    assert body.startswith(b"RIFF")


def test_reflexes_serve_returns_404_when_missing(monkeypatch):
    monkeypatch.setattr(srv_module, "read_reflex", lambda **_kw: None)
    handler = _make_handler("GET", "/reflexes/alloy/ack_mhm.wav")
    handler._reflexes_serve("alloy/ack_mhm.wav")
    status, body = _parse_handler_response(handler)
    assert status == 404
    assert "not generated" in body["error"].lower()
    assert body["reflex_id"] == "ack_mhm"


def test_reflexes_serve_returns_400_for_unknown_id(monkeypatch):
    def raise_unknown(**_kw):
        raise srv_module.ReflexUnknown("unknown reflex id: 'totally_bogus'")
    monkeypatch.setattr(srv_module, "read_reflex", raise_unknown)
    handler = _make_handler("GET", "/reflexes/alloy/totally_bogus.wav")
    handler._reflexes_serve("alloy/totally_bogus.wav")
    status, body = _parse_handler_response(handler)
    assert status == 400
    assert "known_reflexes" in body


def test_reflexes_regenerate_proxies_to_generate_all(monkeypatch):
    called = {}

    def fake_generate(**kwargs):
        called.update(kwargs)
        return {
            "schema": "Ms4ReflexGeneration.v1",
            "voice": kwargs.get("voice") or "alloy",
            "model": "tts-1",
            "force": kwargs.get("force", True),
            "generated": [{"id": "ack_mhm", "voice": "alloy", "size_bytes": 100}],
            "skipped": [],
            "failed": {},
            "total": 1,
        }

    monkeypatch.setattr(srv_module, "generate_all", fake_generate)
    handler = _make_handler("POST", "/reflexes/regenerate", body=b'{"force": true}')
    handler._reflexes_regenerate()
    status, body = _parse_handler_response(handler)
    assert status == 200
    assert body["schema"] == "Ms4ReflexGeneration.v1"
    assert called.get("force") is True


def test_doctrine_meta_route_returns_schema(monkeypatch):
    fake_meta = {"schema": "Ms4DoctrineMeta.v1", "path": "/x/bible.md",
                 "chars": 196011, "section_count": 51, "mtime": 1709810000.0, "summary": "TMR is..."}
    monkeypatch.setattr(srv_module, "doctrine_meta", lambda: fake_meta)
    handler = _make_handler("GET", "/doctrine/tmr/meta")
    handler._doctrine_meta()
    status, body = _parse_handler_response(handler)
    assert status == 200
    assert body == fake_meta


def test_doctrine_sections_route_returns_list(monkeypatch):
    class _Sec:
        def __init__(self, id, title, level, start, end):
            self.id, self.title, self.level, self.start, self.end = id, title, level, start, end
        @property
        def char_count(self): return self.end - self.start

    fake_sections = [
        _Sec("part_i", "PART I", 1, 0, 1000),
        _Sec("part_ii", "PART II", 1, 1000, 3000),
    ]
    monkeypatch.setattr(srv_module, "list_sections", lambda: fake_sections)
    handler = _make_handler("GET", "/doctrine/tmr/sections")
    handler._doctrine_sections_list()
    status, body = _parse_handler_response(handler)
    assert status == 200
    assert body["schema"] == "Ms4DoctrineSections.v1"
    assert body["total_sections"] == 2
    assert body["sections"][0]["id"] == "part_i"
    assert body["sections"][0]["char_count"] == 1000


def test_doctrine_read_into_session_proxies_injection(monkeypatch):
    called = {}

    def fake_inject(face_lobe_chat, **kw):
        called.update(kw)
        return {"schema": "Ms4DoctrineInjection.v1", "session_id": kw["session_id"],
                "kind": kw["kind"], "section_id": kw.get("section_id"),
                "chars_injected": 196011, "message_count": 2, "acknowledgment": "Doctrine read."}

    monkeypatch.setattr(srv_module, "inject_into_session", fake_inject)
    handler = _make_handler("POST", "/doctrine/tmr/read-into-session",
                            body=b'{"session_id":"abc","kind":"full"}')
    handler._doctrine_read_into_session()
    status, body = _parse_handler_response(handler)
    assert status == 200
    assert body["schema"] == "Ms4DoctrineInjection.v1"
    assert called["session_id"] == "abc"
    assert called["kind"] == "full"


def test_spirit_state_route_includes_heartbeat(monkeypatch):
    monkeypatch.setattr(srv_module, "get_state_snapshot", lambda _url: {
        "schema": "Ms4SpiritState.v1", "ms3_reachable": True,
        "identity": {"chosen_name": "Sister"},
        "personality": None, "resonance": None, "state": None,
        "last_self_examination": None, "errors": [],
    })
    monkeypatch.setattr(srv_module, "heartbeat_status", lambda: {
        "running": True, "interval_secs": 60, "last_at_unix": 1_700_000_000.0,
        "last_age_secs": 5.0, "count": 12, "errors": 0, "last_result": {"ok": True},
    })
    handler = _make_handler("GET", "/spirit/state")
    handler._spirit_state_get()
    status, body = _parse_handler_response(handler)
    assert status == 200
    assert body["schema"] == "Ms4SpiritState.v1"
    assert body["identity"]["chosen_name"] == "Sister"
    assert body["heartbeat"]["count"] == 12
    assert body["heartbeat"]["running"] is True


def test_voice_turn_stream_separates_chat_model_from_tts_model(monkeypatch):
    """The bug we caught live: ?model=tts-1 from the Settings dialog was
    being routed as the FACE LOBE chat model, which made the Face Lobe
    try to do chat completions against tts-1 and fall through to the
    canned-failure reply on every voice turn. The contract MUST be:
      ?model=     -> Face Lobe chat model (e.g. phi4-mini)
      ?tts_model= -> TTS model (e.g. tts-1, tts-1-hd)
    """
    captured: dict[str, object] = {}

    def fake_voice_stream(**kwargs):
        captured.update(kwargs)
        # Pretend the turn finished so the SSE drain loop terminates.
        kwargs["emit"]("status", {"phase": "transcribing"})
        return {"transcript": "x", "reply_text": "y", "session_id": "s",
                "metrics": {"schema": "Ms4VoiceStreamMetrics.v1"}}

    monkeypatch.setattr(srv_module, "voice_ptt_turn_stream", fake_voice_stream)

    boundary = "----TESTBOUNDARY"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="t.wav"\r\n'
        f"Content-Type: audio/wav\r\n\r\nWAVDATA\r\n--{boundary}--\r\n"
    ).encode()
    handler = _make_handler(
        "POST",
        "/voice/turn/stream?session_id=abc&model=phi4-mini&tts_model=tts-1-hd&voice=vega&engine=ws_super",
        body,
    )
    handler.headers = types.SimpleNamespace(get={
        "Host": "127.0.0.1:9180",
        "Content-Length": str(len(body)),
        "Content-Type": f"multipart/form-data; boundary={boundary}",
    }.get)
    handler.rfile = io.BytesIO(body)
    handler._voice_turn_stream()

    assert captured.get("model") == "phi4-mini", "chat model must come from ?model="
    assert captured.get("tts_model") == "tts-1-hd", "tts model must come from ?tts_model= (NOT ?model=)"
    assert captured.get("tts_voice") == "vega"
    assert captured.get("engine") == "ws_super"
    assert captured.get("session_id") == "abc"


def test_voice_turn_stream_query_param_engine_reaches_orchestrator(monkeypatch):
    """The UI sends ?engine=ws_super on the /voice/turn/stream URL to
    override the env default per-turn. Make sure the server actually
    plumbs that value into voice_ptt_turn_stream(engine=...)."""
    captured: dict[str, object] = {}

    def fake_voice_stream(**kwargs):
        captured.update(kwargs)
        # Emit a minimal "done" so the SSE drain loop terminates.
        kwargs["emit"]("status", {"phase": "transcribing"})
        return {
            "transcript": "test",
            "reply_text": "ok",
            "session_id": "s",
            "metrics": {"schema": "Ms4VoiceStreamMetrics.v1", "engine": "ws_super"},
        }

    monkeypatch.setattr(srv_module, "voice_ptt_turn_stream", fake_voice_stream)

    # Stitch a minimal multipart body so _read_audio_body / parse_audio_request
    # accepts it. The orchestrator is faked anyway, so we just need to get
    # past the "is this a real audio body?" check.
    boundary = "----TESTBOUNDARY"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="t.wav"\r\n'
        f"Content-Type: audio/wav\r\n\r\n"
        f"WAVDATA\r\n--{boundary}--\r\n"
    ).encode()
    handler = _make_handler("POST", "/voice/turn/stream?engine=ws_super&voice=vega&session_id=abc", body)
    handler.headers = types.SimpleNamespace(get={
        "Host": "127.0.0.1:9180",
        "Content-Length": str(len(body)),
        "Content-Type": f"multipart/form-data; boundary={boundary}",
    }.get)
    handler.rfile = io.BytesIO(body)
    handler._voice_turn_stream()

    # We don't care about the SSE bytes; we care that the handler passed
    # the right kwargs through to the orchestrator.
    assert captured.get("engine") == "ws_super"
    assert captured.get("tts_voice") == "vega"
    assert captured.get("session_id") == "abc"


# ---------------------------------------------------------------------------
# /hivemind/active, /hivemind/load, /hivemind/state
# ---------------------------------------------------------------------------


def test_hivemind_active_route_returns_jobs_active_envelope(monkeypatch):
    """GET /hivemind/active should return whatever
    hivemind.jobs.active@v1 reports — same schema the docs promise."""
    from machine_spirit_4.gateway import hivemind_state as hm

    monkeypatch.setattr(
        srv_module, "get_active_jobs",
        lambda *_a, **_kw: {
            "total_active": 2,
            "summary": "2 tasks: 1 inference, 1 pull",
            "inference": [{"id": "j1"}], "pulls": [{"pull_id": "p1"}],
            "scatter": [], "training": [],
        },
    )
    handler = _make_handler("GET", "/hivemind/active")
    handler._hivemind_active_get()
    status, body = _parse_handler_response(handler)
    assert status == 200
    assert body["total_active"] == 2
    assert body["summary"].startswith("2 tasks")


def test_hivemind_active_route_502_on_unreachable(monkeypatch):
    from machine_spirit_4.gateway.hivemind_state import HivemindStateError

    def _explode(*_a, **_kw):
        raise HivemindStateError("hivemind unreachable: nope")

    monkeypatch.setattr(srv_module, "get_active_jobs", _explode)
    handler = _make_handler("GET", "/hivemind/active")
    handler._hivemind_active_get()
    status, body = _parse_handler_response(handler)
    assert status == 502
    assert "unreachable" in body["error"]


def test_hivemind_load_route_returns_load_stats(monkeypatch):
    monkeypatch.setattr(
        srv_module, "get_cluster_load",
        lambda *_a, **_kw: {
            "trackers": {"total": 4, "amplification_ratio": 1.0},
            "deadlines": {"early_exits_total": 0},
            "shed_total": {"deadline": 0, "budget": 0, "peer_overload": 0},
        },
    )
    handler = _make_handler("GET", "/hivemind/load")
    handler._hivemind_load_get()
    status, body = _parse_handler_response(handler)
    assert status == 200
    assert body["trackers"]["amplification_ratio"] == 1.0


def test_hivemind_state_route_returns_combined_snapshot(monkeypatch):
    """GET /hivemind/state is the fail-soft snapshot the Settings dialog
    renders. Per-tool failures land in ``errors`` instead of failing
    the whole response."""
    snapshot = {
        "schema": "Ms4HivemindState.v1",
        "hivemind_url": "http://hive:6089",
        "mcp_base_url": "http://hive:6105",
        "auth_configured": False,
        "active_jobs": {"total_active": 0, "summary": ""},
        "cluster_load": None,
        "service_health": {"menta_hli": {"healthy": True}},
        "errors": ["cluster.load: HivemindStateError"],
    }
    monkeypatch.setattr(srv_module, "get_hivemind_snapshot", lambda *_a, **_kw: snapshot)
    handler = _make_handler("GET", "/hivemind/state")
    handler._hivemind_state_get()
    status, body = _parse_handler_response(handler)
    assert status == 200
    assert body["schema"] == "Ms4HivemindState.v1"
    assert body["active_jobs"]["total_active"] == 0
    assert body["service_health"]["menta_hli"]["healthy"] is True
    assert "cluster.load" in body["errors"][0]
