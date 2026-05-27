"""HiveMind voice service admin tests.

What's covered
--------------

* ``get_provision_status`` / ``list_voice_services`` parse the live
  HiveMind ``/provision/status/{SERVICE}`` shape correctly into
  ``Ms4VoiceServiceStatus.v1``.
* ``request_voice_service`` / ``release_voice_service`` produce the
  exact MCP JSON-RPC envelope HiveMind expects, unwrap the
  ``result.content[0].text`` JSON correctly, and surface
  ``provision_id`` + ``poll_url`` back to the caller.
* Unknown services raise ``VoiceServiceUnknown`` (mapped to 400 by
  the gateway).
* HiveMind 502 / unreachable surfaces as ``VoiceAdminError`` (mapped
  to 502 by the gateway).
* ``poll_until_healthy`` returns on the first healthy snapshot and
  honors ``timeout_secs``.

The fake HiveMind server speaks the real wire format observed live:

    GET /provision/status/ASR
        {"detail":"running","endpoint":"http://...","service_health":{"healthy":true,"provisioning_state":"running"}}
    POST /v1/mcp tools/call hivemind.resources.request@v1
        {"jsonrpc":"2.0","id":"...","result":{"content":[{"type":"text","text":"<inner-json>"}]}}
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from machine_spirit_4.gateway.voice_admin import (
    VOICE_SERVICES,
    VoiceAdminError,
    VoiceServiceUnknown,
    get_provision_status,
    list_voice_services,
    poll_until_healthy,
    release_voice_service,
    request_voice_service,
)


# ---------------------------------------------------------------------------
# Fake HiveMind server (subset of /provision/status + /v1/mcp)
# ---------------------------------------------------------------------------


class _FakeHive:
    """In-process HTTP server scripting the subset of HiveMind that
    voice_admin talks to. Mutate ``.status`` between calls to simulate
    provisioning transitions; inspect ``.mcp_calls`` to assert the
    MCP envelope was correct."""

    def __init__(self):
        self.status: dict[str, dict] = {
            svc: {
                "backend": "gim",
                "detail": "No endpoints configured",
                "endpoint": "",
                "endpoints_configured": 1,
                "job_type": svc,
                "port": 0,
                "service_health": {
                    "endpoint": "",
                    "healthy": False,
                    "last_error": "No endpoints configured",
                    "provisioning_state": "configured_not_provisioned",
                },
            }
            for svc in VOICE_SERVICES
        }
        self.mcp_calls: list[dict] = []
        self.next_mcp_result: dict = {
            "service": "ASR",
            "capability": "asr",
            "status": "provisioning",
            "provision_id": "00000000-test-uuid",
            "poll_url": "/provision/status/ASR",
            "backend": "gim",
            "model": "whisper-1",
        }
        self._server = None
        self._thread = None

    def __enter__(self):
        outer = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a, **_kw):
                pass

            def do_GET(self):  # noqa: N802
                if self.path.startswith("/provision/status/"):
                    svc = self.path.rsplit("/", 1)[-1]
                    body = outer.status.get(svc)
                    if body is None:
                        self.send_error(404)
                        return
                    out = json.dumps(body).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(out)))
                    self.end_headers()
                    self.wfile.write(out)
                    return
                self.send_error(404)

            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length", "0") or "0")
                body = self.rfile.read(length).decode("utf-8") if length else ""
                if self.path == "/v1/mcp":
                    envelope = json.loads(body or "{}")
                    outer.mcp_calls.append(envelope)
                    inner = json.dumps(outer.next_mcp_result)
                    resp = json.dumps({
                        "jsonrpc": "2.0",
                        "id": envelope.get("id"),
                        "result": {"content": [{"type": "text", "text": inner}]},
                    }).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(resp)))
                    self.end_headers()
                    self.wfile.write(resp)
                    return
                self.send_error(404)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        return self

    def __exit__(self, *_exc):
        if self._server:
            self._server.shutdown()
        if self._thread:
            self._thread.join(timeout=3)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_get_provision_status_projects_real_hivemind_shape_to_v1_schema():
    with _FakeHive() as hive:
        hive.status["ASR"] = {
            "detail": "running",
            "endpoint": "http://127.0.0.1:49120/v1/audio/transcriptions",
            "endpoints_configured": 1,
            "service_health": {
                "endpoint": "http://127.0.0.1:49120/v1/audio/transcriptions",
                "healthy": True,
                "last_error": "",
                "provisioning_state": "running",
            },
            "job_type": "ASR",
            "port": 49120,
        }
        st = get_provision_status(hive.url, "asr")  # accepts lowercase
        assert st["schema"] == "Ms4VoiceServiceStatus.v1"
        assert st["service"] == "ASR"
        assert st["healthy"] is True
        assert st["detail"] == "running"
        assert st["endpoint"].endswith("/audio/transcriptions")
        assert st["endpoints_configured"] == 1
        assert st["provisioning_state"] == "running"


def test_get_provision_status_handles_unhealthy_and_no_endpoints():
    with _FakeHive() as hive:
        st = get_provision_status(hive.url, "ASR")
        assert st["service"] == "ASR"
        assert st["healthy"] is False
        assert st["provisioning_state"] == "configured_not_provisioned"
        assert "No endpoints configured" in st["detail"]


def test_unknown_service_raises_clearly():
    with pytest.raises(VoiceServiceUnknown):
        get_provision_status("http://hive", "DOES_NOT_EXIST")


def test_list_voice_services_returns_all_three_in_order():
    with _FakeHive() as hive:
        services = list_voice_services(hive.url)
        assert [s["service"] for s in services] == list(VOICE_SERVICES)
        assert all(s["schema"] == "Ms4VoiceServiceStatus.v1" for s in services)


def _find_resources_request(calls):
    """Helper: voice_admin first calls hivemind.service_health@v1 to
    check the maintenance window (added May 25 2026), then calls
    hivemind.resources.request@v1. Return the resources.request
    envelope so the assertions don't break when the maintenance check
    is enabled."""
    for envelope in calls:
        params = envelope.get("params") or {}
        if params.get("name") == "hivemind.resources.request@v1":
            return envelope
    raise AssertionError(f"no hivemind.resources.request@v1 call found in {calls}")


def test_request_voice_service_sends_correct_mcp_envelope():
    with _FakeHive() as hive:
        result = request_voice_service(hive.url, "ASR", tier="balanced")
    envelope = _find_resources_request(hive.mcp_calls)
    assert envelope["jsonrpc"] == "2.0"
    assert envelope["method"] == "tools/call"
    assert envelope["params"]["name"] == "hivemind.resources.request@v1"
    args = envelope["params"]["arguments"]
    assert args["capability"] == "asr"
    assert args["tier"] == "balanced"
    # The inner result.content[0].text is unwrapped to the dict the
    # gateway and UI consume directly.
    assert result["provision_id"] == "00000000-test-uuid"
    assert result["status"] == "provisioning"
    assert result["service"] == "ASR"


def test_request_voice_service_tts_super_maps_to_superskill_capability():
    with _FakeHive() as hive:
        request_voice_service(hive.url, "TTS_SUPER")
    envelope = _find_resources_request(hive.mcp_calls)
    args = envelope["params"]["arguments"]
    assert args["capability"] == "superskill:tts_super"


def test_release_voice_service_envelope():
    with _FakeHive() as hive:
        release_voice_service(hive.url, "ASR", provision_id="abc-123")
    envelope = hive.mcp_calls[0]
    assert envelope["params"]["name"] == "hivemind.resources.release@v1"
    assert envelope["params"]["arguments"]["provision_id"] == "abc-123"


def test_hivemind_unreachable_raises_voice_admin_error():
    # Bind to a port that nothing is listening on.
    with pytest.raises(VoiceAdminError):
        get_provision_status("http://127.0.0.1:1", "ASR", timeout=1)


def test_poll_until_healthy_returns_on_first_healthy_snapshot():
    with _FakeHive() as hive:
        # Flip to healthy after first poll.
        def flip_after_first_poll():
            time.sleep(0.05)
            hive.status["ASR"]["service_health"]["healthy"] = True
            hive.status["ASR"]["service_health"]["provisioning_state"] = "running"
            hive.status["ASR"]["detail"] = "running"

        threading.Thread(target=flip_after_first_poll, daemon=True).start()
        st = poll_until_healthy(hive.url, "ASR", timeout_secs=5.0, poll_interval_secs=0.1)
    assert st["healthy"] is True
    assert st["provisioning_state"] == "running"


def test_poll_until_healthy_respects_timeout():
    with _FakeHive() as hive:
        t0 = time.monotonic()
        st = poll_until_healthy(hive.url, "ASR", timeout_secs=0.3, poll_interval_secs=0.1)
    elapsed = time.monotonic() - t0
    assert st["healthy"] is False
    assert elapsed < 1.0  # didn't run forever
