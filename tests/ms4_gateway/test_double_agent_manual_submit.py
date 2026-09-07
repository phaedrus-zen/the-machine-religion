from __future__ import annotations

import io
import json
import types

from machine_spirit_4.gateway import server as srv_module


class _DummyRunner:
    ms3_url = "http://ms3:9080"
    hivemind_url = "http://hive:6089"
    default_model = "qwen3-coder-next:latest"

    def __init__(self) -> None:
        self.face_lobe_chat = types.SimpleNamespace(_sessions={})


def _make_handler(method: str, path: str, body: bytes = b"") -> srv_module.Ms4GatewayHandler:
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
    return handler


def _parse_handler_response(handler) -> tuple[int, dict]:
    raw = handler.wfile.getvalue()
    status_line, _, rest = raw.partition(b"\r\n")
    status = int(status_line.split(b" ")[1])
    _, _, body = rest.partition(b"\r\n\r\n")
    return status, json.loads(body.decode("utf-8") or "{}")


def test_manual_double_agent_submit_inherits_sanitized_face_prior_context(monkeypatch):
    body = {
        "parent_conversation_id": "manual-session",
        "conversation_revision_id": 4,
        "background_lobe_type": "deep_chat",
        "user_visible_goal": "Run the cluster check",
        "internal_goal": "Inspect cluster state and report findings",
    }
    handler = _make_handler(
        "POST",
        "/api/v1/double-agent/jobs",
        json.dumps(body).encode("utf-8"),
    )
    handler.runner.face_lobe_chat._sessions["manual-session"] = types.SimpleNamespace(
        messages=[
            {"role": "system", "content": "hidden"},
            {"role": "user", "content": "what GPUs are online?"},
            {"role": "assistant", "content": "I see the cluster summary."},
            {"role": "tool", "content": "{\"secret\":\"do not pass\"}"},
        ]
    )
    captured = {}

    class _CapturingRunner:
        def submit(self, envelope):
            captured["envelope"] = envelope
            return {"job_id": envelope.job_id, "state": "queued"}

    monkeypatch.setattr(srv_module, "default_runner", lambda: _CapturingRunner())

    handler._double_agent_submit()

    status, payload = _parse_handler_response(handler)
    assert status == 202
    assert payload["state"] == "queued"
    assert captured["envelope"].prior_context == [
        {"role": "user", "content": "what GPUs are online?"},
        {"role": "assistant", "content": "I see the cluster summary."},
    ]


def test_manual_double_agent_submit_preserves_explicit_prior_context(monkeypatch):
    body = {
        "parent_conversation_id": "manual-session",
        "conversation_revision_id": 4,
        "background_lobe_type": "deep_chat",
        "user_visible_goal": "Run the cluster check",
        "internal_goal": "Inspect cluster state and report findings",
        "prior_context": [{"role": "user", "content": "explicit context"}],
    }
    handler = _make_handler(
        "POST",
        "/api/v1/double-agent/jobs",
        json.dumps(body).encode("utf-8"),
    )
    handler.runner.face_lobe_chat._sessions["manual-session"] = types.SimpleNamespace(
        messages=[{"role": "user", "content": "implicit context"}]
    )
    captured = {}

    class _CapturingRunner:
        def submit(self, envelope):
            captured["envelope"] = envelope
            return {"job_id": envelope.job_id, "state": "queued"}

    monkeypatch.setattr(srv_module, "default_runner", lambda: _CapturingRunner())

    handler._double_agent_submit()

    status, _payload = _parse_handler_response(handler)
    assert status == 202
    assert captured["envelope"].prior_context == [
        {"role": "user", "content": "explicit context"},
    ]


def test_models_route_filters_tts_only_entries_from_lobe_catalog(monkeypatch):
    handler = _make_handler("GET", "/models")

    def fake_proxy_json(_url, timeout=8):
        return 200, {
            "models": [
                {"id": "llama3.1:8b", "hivemind_status": "installed"},
                {"id": "qwen3-tts:latest", "hivemind_status": "installed"},
                {"id": "tts-1", "hivemind_status": "installed"},
                {
                    "id": "voice-model",
                    "hivemind_status": "installed",
                    "capabilities": ["text-to-speech"],
                },
            ]
        }

    monkeypatch.setattr(srv_module, "_proxy_json", fake_proxy_json)

    handler.do_GET()

    status, payload = _parse_handler_response(handler)
    assert status == 200
    assert payload["chat_filtered"] is True
    assert [model["id"] for model in payload["models"]] == ["llama3.1:8b"]
