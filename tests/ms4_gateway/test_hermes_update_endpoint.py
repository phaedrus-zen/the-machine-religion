"""Executable REST contract for the Hermes Update control."""

from __future__ import annotations

import io
import json
import types

from machine_spirit_4.gateway import server as srv_module
from machine_spirit_4.hermes_admin import HermesUpgradeError, UpdateJobSnapshot


class _DummyRunner:
    ms3_url = "http://ms3.invalid"
    hivemind_url = "http://hive.invalid"


def _handler(method: str, path: str, payload: dict | None = None):
    body = json.dumps(payload).encode("utf-8") if payload is not None else b""
    handler = srv_module.Ms4GatewayHandler.__new__(srv_module.Ms4GatewayHandler)
    handler.runner = _DummyRunner()
    handler.command = method
    handler.path = path
    headers = {"Host": "127.0.0.1:9180", "Content-Length": str(len(body))}
    if body:
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


def _response(handler) -> tuple[int, dict]:
    raw = handler.wfile.getvalue()
    status_line, _, rest = raw.partition(b"\r\n")
    _, _, body = rest.partition(b"\r\n\r\n")
    return int(status_line.split()[1]), json.loads(body.decode("utf-8"))


def _running_snapshot() -> UpdateJobSnapshot:
    return UpdateJobSnapshot(
        job_id="job-1",
        started_at="2026-09-06T12:00:00+00:00",
        finished_at=None,
        status="running",
        phase="queued",
        from_version="0.19.0",
        to_version="0.20.0",
        install_mode="editable",
        error=None,
        progress=[],
        request_user="ms4-gateway",
    )


def test_update_post_starts_exact_signed_target(monkeypatch):
    calls = []

    def trigger_update(**kwargs):
        calls.append(kwargs)
        return _running_snapshot()

    monkeypatch.setattr(srv_module.hermes_admin, "trigger_update", trigger_update)
    handler = _handler(
        "POST",
        "/api/v1/hermes/update",
        {"target_version": "0.20.0"},
    )

    handler.do_POST()

    status, payload = _response(handler)
    assert status == 202
    assert payload["status"] == "running"
    assert payload["to_version"] == "0.20.0"
    assert calls == [
        {"target_version": "0.20.0", "request_user": "ms4-gateway"}
    ]


def test_update_post_surfaces_preflight_failure(monkeypatch):
    monkeypatch.setattr(
        srv_module.hermes_admin,
        "trigger_update",
        lambda **_kwargs: (_ for _ in ()).throw(
            HermesUpgradeError("Git preflight failed before install")
        ),
    )
    handler = _handler(
        "POST",
        "/api/v1/hermes/update",
        {"target_version": "0.20.0"},
    )

    handler.do_POST()

    status, payload = _response(handler)
    assert status == 400
    assert payload == {"error": "Git preflight failed before install"}


def test_version_get_returns_cold_filled_update_state(monkeypatch):
    expected = {
        "schema": "Ms4HermesVersion.v1",
        "current": "0.19.0",
        "latest": "0.20.0",
        "update_available": True,
        "update_in_progress": False,
        "operator_state": "update_available",
    }
    monkeypatch.setattr(srv_module.hermes_admin, "version_info", lambda: expected)
    handler = _handler("GET", "/api/v1/hermes/version")

    handler.do_GET()

    assert _response(handler) == (200, expected)
