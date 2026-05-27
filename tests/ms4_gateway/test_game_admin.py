"""Unit + route + MCP tests for the game-session admin module.

Mirrors ``test_hivemind_admin_expansion.py``'s in-process MCP fake
pattern. Covers the May 26 2026 HiveMind release shipping the Phase-1
dry-run ``hivemind.game.ensure_available@v1`` +
``hivemind.game_session.{plan,run,status,evidence,cancel}@v1`` tools
and MS4's wrappers / routes / MCP proxies for them.
"""

from __future__ import annotations

import io
import json
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from machine_spirit_4.gateway import game_admin, server as srv_module
from machine_spirit_4.gateway.game_admin import GameAdminError
from machine_spirit_4.mcp import tools as mcp_tools


# ---------------------------------------------------------------------------
# In-process fake MCP server
# ---------------------------------------------------------------------------


class _FakeMcp:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []
        self.responses: dict[str, Any] = {}
        self.errors: dict[str, str] = {}
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.port: int = 0

    def set_response(self, tool: str, payload: Any) -> None:
        self.responses[tool] = payload

    def set_error(self, tool: str, message: str) -> None:
        self.errors[tool] = message

    def start(self) -> None:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args, **_kwargs):
                return

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0") or "0")
                raw = self.rfile.read(length).decode("utf-8") if length else "{}"
                try:
                    body = json.loads(raw)
                except json.JSONDecodeError:
                    body = {}
                outer.calls.append({"body": body})
                tool_name = (body.get("params") or {}).get("name") if isinstance(body, dict) else None
                if tool_name in outer.errors:
                    envelope = {
                        "jsonrpc": "2.0",
                        "id": body.get("id", 1),
                        "result": {
                            "content": [{"type": "text", "text": outer.errors[tool_name]}],
                            "isError": True,
                        },
                    }
                else:
                    payload = outer.responses.get(tool_name or "", {})
                    envelope = {
                        "jsonrpc": "2.0",
                        "id": body.get("id", 1),
                        "result": {
                            "content": [{"type": "text", "text": json.dumps(payload)}],
                            "isError": False,
                        },
                    }
                wire = json.dumps(envelope).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(wire)))
                self.end_headers()
                self.wfile.write(wire)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]

    def stop(self) -> None:
        if self.server:
            try:
                self.server.shutdown()
            except Exception:
                pass


@pytest.fixture
def fake_mcp(monkeypatch):
    server = _FakeMcp()
    server.start()
    monkeypatch.setenv("MS4_HIVEMIND_MCP_URL", f"http://127.0.0.1:{server.port}/mcp")
    monkeypatch.delenv("MS4_HIVEMIND_API_KEY", raising=False)
    yield server
    server.stop()


def hurl(fake: _FakeMcp) -> str:
    return f"http://127.0.0.1:{fake.port}"


# ===========================================================================
# Admin module — happy + error paths
# ===========================================================================


def test_ensure_available_returns_remediation_for_unknown(fake_mcp):
    fake_mcp.set_response(
        "hivemind.game.ensure_available@v1",
        {"available": False, "remediation": {"hint": "set MS_GAME_PATH_CYBERPUNK"}},
    )
    snap = game_admin.ensure_available(hurl(fake_mcp), "cyberpunk-2077")
    assert snap["schema"] == "Ms4GameAvailability.v1"
    assert snap["game_id"] == "cyberpunk-2077"
    assert snap["available"] is False
    assert snap["remediation"]["hint"] == "set MS_GAME_PATH_CYBERPUNK"


def test_ensure_available_returns_path_when_available(fake_mcp):
    fake_mcp.set_response(
        "hivemind.game.ensure_available@v1",
        {"available": True, "path": "C:/Games/Cyberpunk 2077"},
    )
    snap = game_admin.ensure_available(hurl(fake_mcp), "cyberpunk-2077")
    assert snap["available"] is True
    assert snap["path"] == "C:/Games/Cyberpunk 2077"


def test_ensure_available_rejects_empty():
    with pytest.raises(ValueError):
        game_admin.ensure_available("http://x", "")


def test_plan_passes_quality_and_returns_job_id(fake_mcp):
    fake_mcp.set_response(
        "hivemind.game_session.plan@v1",
        {"job_id": "gs-42", "plan": {"phases": ["resolve_host", "attach_gpu"]}},
    )
    body = game_admin.plan(hurl(fake_mcp), game="cyberpunk-2077", client="ms4-ui", quality="best_available")
    assert body["job_id"] == "gs-42"
    args = fake_mcp.calls[-1]["body"]["params"]["arguments"]
    assert args["game"] == "cyberpunk-2077"
    assert args["client"] == "ms4-ui"
    assert args["quality"] == "best_available"


def test_plan_rejects_empty_game():
    with pytest.raises(ValueError):
        game_admin.plan("http://x", game="")


def test_run_status_evidence_cancel_lifecycle(fake_mcp):
    fake_mcp.set_response("hivemind.game_session.run@v1", {"state": "COMPLETE"})
    fake_mcp.set_response(
        "hivemind.game_session.status@v1",
        {"state": "COMPLETE", "transitions": [{"phase": "resolve_host"}]},
    )
    fake_mcp.set_response(
        "hivemind.game_session.evidence@v1",
        {"ledger": [{"phase": "resolve_host", "action": "would_call hivemind.hosts.list@v1"}]},
    )
    fake_mcp.set_response("hivemind.game_session.cancel@v1", {"state": "CANCELLED"})
    assert game_admin.run(hurl(fake_mcp), "gs-1")["state"] == "COMPLETE"
    assert game_admin.status(hurl(fake_mcp), "gs-1")["transitions"][0]["phase"] == "resolve_host"
    ledger = game_admin.evidence(hurl(fake_mcp), "gs-1")["ledger"]
    assert ledger[0]["action"].startswith("would_call")
    assert game_admin.cancel(hurl(fake_mcp), "gs-1")["state"] == "CANCELLED"


def test_run_rejects_empty_job_id():
    with pytest.raises(ValueError):
        game_admin.run("http://x", "")


def test_admin_wraps_mcp_error_as_admin_error(fake_mcp):
    fake_mcp.set_error("hivemind.game_session.plan@v1", "planner unreachable")
    with pytest.raises(GameAdminError) as exc:
        game_admin.plan(hurl(fake_mcp), game="cyberpunk-2077")
    assert "planner unreachable" in str(exc.value)


def test_plan_run_and_collect_happy(fake_mcp):
    fake_mcp.set_response(
        "hivemind.game_session.plan@v1", {"job_id": "gs-5", "plan": {"phases": ["a", "b"]}}
    )
    fake_mcp.set_response("hivemind.game_session.run@v1", {"state": "COMPLETE"})
    fake_mcp.set_response(
        "hivemind.game_session.evidence@v1",
        {"ledger": [{"phase": "a"}]},
    )
    snap = game_admin.plan_run_and_collect(hurl(fake_mcp), game="cyberpunk-2077", quality="best_available")
    assert snap["schema"] == "Ms4GameSession.v1"
    assert snap["game"] == "cyberpunk-2077"
    assert snap["job_id"] == "gs-5"
    assert snap["run_result"]["state"] == "COMPLETE"
    assert snap["evidence"]["ledger"][0]["phase"] == "a"
    assert snap["errors"] == []


def test_plan_run_and_collect_fail_soft_on_run_error(fake_mcp):
    """Plan succeeded but run errored — UI still gets plan + the
    error captured in errors[]."""
    fake_mcp.set_response("hivemind.game_session.plan@v1", {"job_id": "gs-9", "plan": {}})
    fake_mcp.set_error("hivemind.game_session.run@v1", "state machine wedged")
    fake_mcp.set_response("hivemind.game_session.evidence@v1", {"ledger": []})
    snap = game_admin.plan_run_and_collect(hurl(fake_mcp), game="cyberpunk-2077")
    assert snap["job_id"] == "gs-9"
    assert snap["run_result"] is None
    assert any("state machine wedged" in e for e in snap["errors"])
    assert snap["evidence"] == {"ledger": []}


# ===========================================================================
# REST routes
# ===========================================================================


class _DummyRunner:
    ms3_url = "http://ms3:9080"
    default_model = "qwen3-coder-next:latest"
    face_lobe_chat = types.SimpleNamespace(_sessions={})

    def __init__(self, hivemind_url: str):
        self.hivemind_url = hivemind_url


def _make_handler(runner, method: str, path: str, body: bytes = b"") -> srv_module.Ms4GatewayHandler:
    handler = srv_module.Ms4GatewayHandler.__new__(srv_module.Ms4GatewayHandler)
    handler.runner = runner
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


def _read_response(handler) -> tuple[int, dict[str, Any]]:
    raw = handler.wfile.getvalue()
    status_line, _, rest = raw.partition(b"\r\n")
    status = int(status_line.split(b" ")[1])
    _head, _, body = rest.partition(b"\r\n\r\n")
    return status, json.loads(body.decode("utf-8") or "{}")


def test_route_game_availability(fake_mcp):
    fake_mcp.set_response(
        "hivemind.game.ensure_available@v1",
        {"available": False, "remediation": {"hint": "install game"}},
    )
    handler = _make_handler(
        _DummyRunner(hurl(fake_mcp)), "GET", "/hivemind/games/cyberpunk-2077/availability"
    )
    handler._hivemind_game_availability_get("cyberpunk-2077")
    status, body = _read_response(handler)
    assert status == 200
    assert body["schema"] == "Ms4GameAvailability.v1"
    assert body["game_id"] == "cyberpunk-2077"
    assert body["available"] is False


def test_route_game_session_plan_validates_game(fake_mcp):
    handler = _make_handler(
        _DummyRunner(hurl(fake_mcp)), "POST", "/hivemind/game-sessions/plan", body=b"{}"
    )
    handler._hivemind_game_session_plan()
    status, body = _read_response(handler)
    assert status == 400
    assert "game" in body["error"]


def test_route_game_session_plan_happy(fake_mcp):
    fake_mcp.set_response(
        "hivemind.game_session.plan@v1",
        {"job_id": "gs-1", "plan": {"phases": ["a"]}},
    )
    handler = _make_handler(
        _DummyRunner(hurl(fake_mcp)),
        "POST",
        "/hivemind/game-sessions/plan",
        body=b'{"game": "cyberpunk-2077", "quality": "best_available"}',
    )
    handler._hivemind_game_session_plan()
    status, body = _read_response(handler)
    assert status == 200
    assert body["job_id"] == "gs-1"


def test_route_game_session_run_requires_job_id(fake_mcp):
    handler = _make_handler(
        _DummyRunner(hurl(fake_mcp)), "POST", "/hivemind/game-sessions/run", body=b"{}"
    )
    handler._hivemind_game_session_run()
    status, body = _read_response(handler)
    assert status == 400
    assert "job_id" in body["error"]


def test_route_game_session_status_get(fake_mcp):
    fake_mcp.set_response("hivemind.game_session.status@v1", {"state": "RUNNING"})
    handler = _make_handler(
        _DummyRunner(hurl(fake_mcp)), "GET", "/hivemind/game-sessions/gs-1/status"
    )
    handler._hivemind_game_session_status_get("gs-1")
    status, body = _read_response(handler)
    assert status == 200
    assert body["state"] == "RUNNING"


def test_route_game_session_evidence_get(fake_mcp):
    fake_mcp.set_response(
        "hivemind.game_session.evidence@v1",
        {"ledger": [{"phase": "resolve_host", "action": "would_call hivemind.hosts.list@v1"}]},
    )
    handler = _make_handler(
        _DummyRunner(hurl(fake_mcp)), "GET", "/hivemind/game-sessions/gs-1/evidence"
    )
    handler._hivemind_game_session_evidence_get("gs-1")
    status, body = _read_response(handler)
    assert status == 200
    assert body["ledger"][0]["action"].startswith("would_call")


def test_route_game_session_cancel(fake_mcp):
    fake_mcp.set_response("hivemind.game_session.cancel@v1", {"state": "CANCELLED"})
    handler = _make_handler(
        _DummyRunner(hurl(fake_mcp)), "POST", "/hivemind/game-sessions/gs-1/cancel"
    )
    handler._hivemind_game_session_cancel("gs-1")
    status, body = _read_response(handler)
    assert status == 200
    assert body["state"] == "CANCELLED"


def test_route_game_session_plan_run_happy(fake_mcp):
    fake_mcp.set_response("hivemind.game_session.plan@v1", {"job_id": "gs-x", "plan": {}})
    fake_mcp.set_response("hivemind.game_session.run@v1", {"state": "COMPLETE"})
    fake_mcp.set_response("hivemind.game_session.evidence@v1", {"ledger": []})
    handler = _make_handler(
        _DummyRunner(hurl(fake_mcp)),
        "POST",
        "/hivemind/game-sessions/plan-run",
        body=b'{"game": "cyberpunk-2077"}',
    )
    handler._hivemind_game_session_plan_run()
    status, body = _read_response(handler)
    assert status == 200
    assert body["schema"] == "Ms4GameSession.v1"
    assert body["job_id"] == "gs-x"
    assert body["run_result"]["state"] == "COMPLETE"


# ===========================================================================
# MCP proxies — verify registry exposes the six new tools
# ===========================================================================


def test_mcp_registry_exposes_game_proxies():
    reg = mcp_tools.build_tool_registry()
    expected = {
        "ms4.hivemind.game.ensure_available@v1",
        "ms4.hivemind.game_session.plan@v1",
        "ms4.hivemind.game_session.run@v1",
        "ms4.hivemind.game_session.status@v1",
        "ms4.hivemind.game_session.evidence@v1",
        "ms4.hivemind.game_session.cancel@v1",
    }
    assert expected <= set(reg.keys())


def test_mcp_proxy_game_ensure_available_calls_admin(monkeypatch, fake_mcp):
    fake_mcp.set_response(
        "hivemind.game.ensure_available@v1", {"available": True, "path": "/x"}
    )
    reg = mcp_tools.build_tool_registry()
    tool = reg["ms4.hivemind.game.ensure_available@v1"]
    runtime = types.SimpleNamespace(hivemind_url=hurl(fake_mcp))
    result = tool.handler(runtime, {"game_id": "cyberpunk-2077"})
    assert result["available"] is True
    assert result["schema"] == "Ms4GameAvailability.v1"
