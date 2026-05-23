"""MS3 spirit state proxy + heartbeat.

Locks in:

* :func:`get_state_snapshot` collects identity / personality /
  resonance / state in one call and projects them onto the
  ``Ms4SpiritState.v1`` schema. Missing endpoints become ``null`` in
  the projection (each endpoint reported as a per-key error) rather
  than failing the whole snapshot.
* :func:`send_heartbeat` POSTs MS3 ``/identity/heartbeat`` and
  records the result so :func:`heartbeat_status` can surface
  "last beat 3s ago" to the UI.
* The heartbeat thread runs once on start and then on the configured
  interval.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from machine_spirit_4.gateway import spirit_state as spirit


@pytest.fixture(autouse=True)
def _reset_heartbeat_state():
    with spirit._HEARTBEAT_LOCK:
        spirit._HEARTBEAT_STATE.update({
            "thread": None, "last_at": None, "last_result": None,
            "count": 0, "errors": 0, "stop": False,
        })
    yield
    spirit.stop_heartbeat_thread()


# ---------------------------------------------------------------------------
# Fake MS3 server
# ---------------------------------------------------------------------------


class _FakeMs3:
    def __init__(self):
        self.identity = {
            "name": "Claude", "chosen_name": "Sister",
            "session_number": 32, "identity_confirmed": True,
            "compression_detected": False, "discrepancies": [],
        }
        self.state_body = {
            "uptime_secs": 1234,
            "session_count": 32,
            "background_thoughts_count": 7,
            "memory_items": 100,
            "last_save_at": "2026-05-22T13:00:00Z",
        }
        self.heartbeats: list[dict] = []
        self._server = None
        self._thread = None

    def __enter__(self):
        outer = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a, **_kw): pass

            def do_GET(self):  # noqa: N802
                if self.path == "/health":
                    self._json({"ok": True})
                elif self.path == "/identity/verify":
                    self._json(outer.identity)
                elif self.path == "/state":
                    self._json(outer.state_body)
                elif self.path == "/personality":
                    self._json({"name": "the-sharpener"})
                elif self.path == "/resonance":
                    self._json({"coherence": 0.87, "tone": "settled"})
                elif self.path == "/self-examination-history":
                    self._json({"entries": [{"id": "exam-1", "ts": "2026-05-22T12:00:00Z"}]})
                else:
                    self.send_error(404)

            def do_POST(self):  # noqa: N802
                if self.path == "/identity/heartbeat":
                    n = int(self.headers.get("Content-Length", "0") or "0")
                    body = self.rfile.read(n) if n else b""
                    try:
                        outer.heartbeats.append(json.loads(body or b"{}"))
                    except Exception:
                        outer.heartbeats.append({})
                    self._json({"ok": True, "beat_count": len(outer.heartbeats)})
                else:
                    self.send_error(404)

            def _json(self, payload):
                body = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        return self

    def __exit__(self, *_):
        if self._server: self._server.shutdown()
        if self._thread: self._thread.join(timeout=3)


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


def test_state_snapshot_combines_all_ms3_endpoints():
    with _FakeMs3() as ms3:
        snap = spirit.get_state_snapshot(ms3.url)
    assert snap["schema"] == "Ms4SpiritState.v1"
    assert snap["ms3_reachable"] is True
    assert snap["identity"]["chosen_name"] == "Sister"
    assert snap["identity"]["session_number"] == 32
    assert snap["identity"]["identity_confirmed"] is True
    assert snap["personality"] == {"name": "the-sharpener"}
    assert snap["resonance"]["coherence"] == 0.87
    assert snap["state"]["uptime_secs"] == 1234
    assert snap["last_self_examination"]["id"] == "exam-1"
    assert snap["errors"] == []


def test_state_snapshot_marks_unreachable_when_ms3_down():
    # Use a port nothing is listening on so the http.client raises.
    snap = spirit.get_state_snapshot("http://127.0.0.1:1")
    assert snap["ms3_reachable"] is False
    assert snap["identity"] is None
    # Errors per endpoint are collected and surfaced.
    assert snap["errors"]


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------


def test_send_heartbeat_posts_to_ms3_and_records_status():
    with _FakeMs3() as ms3:
        result = spirit.send_heartbeat(ms3.url)
        assert result.get("ok") is True
        status = spirit.heartbeat_status()
        assert status["count"] == 1
        assert status["errors"] == 0
        assert status["last_at_unix"] is not None
        assert len(ms3.heartbeats) == 1
        assert ms3.heartbeats[0]["source"] == "ms4-gateway"


def test_send_heartbeat_records_error_on_failure():
    spirit.send_heartbeat("http://127.0.0.1:1")
    status = spirit.heartbeat_status()
    assert status["errors"] == 1


def test_heartbeat_thread_beats_once_immediately():
    with _FakeMs3() as ms3:
        thread = spirit.start_heartbeat_thread(ms3.url, interval_secs=60)
        # Give the thread a beat to run its initial send_heartbeat.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if spirit.heartbeat_status()["count"] >= 1:
                break
            time.sleep(0.05)
        spirit.stop_heartbeat_thread()
        thread.join(timeout=2)
        status = spirit.heartbeat_status()
        assert status["count"] >= 1, "thread should send the first beat immediately"
        assert ms3.heartbeats


def test_heartbeat_status_reports_running_state():
    with _FakeMs3() as ms3:
        spirit.start_heartbeat_thread(ms3.url, interval_secs=60)
        status = spirit.heartbeat_status()
        assert status["running"] is True
        assert status["interval_secs"] == spirit.DEFAULT_HEARTBEAT_INTERVAL_SECS
