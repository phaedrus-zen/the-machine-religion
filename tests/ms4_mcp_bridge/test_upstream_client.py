"""Phase 1 — upstream MCP client (stdio + HTTP) against fakes."""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from machine_spirit_4.mcp_bridge.upstream_client import (
    HttpUpstream,
    StdioUpstream,
    UpstreamConfig,
    UpstreamError,
    _parse_http_body,
    connect,
)


_FAKE_SERVER = str(Path(__file__).resolve().parent / "fake_stdio_server.py")


# ---------------------------------------------------------------------------
# stdio
# ---------------------------------------------------------------------------


@pytest.fixture
def stdio():
    up = StdioUpstream(UpstreamConfig(
        server_id="fake", transport="stdio",
        command=sys.executable, args=[_FAKE_SERVER],
        init_timeout=10.0, request_timeout=10.0,
    ))
    yield up
    up.stop()


def test_stdio_initialize_and_list(stdio):
    info = stdio.initialize()
    assert info["serverInfo"]["name"] == "fake-stdio"
    tools = stdio.tools_list()
    names = {t["name"] for t in tools}
    assert names == {"echo", "delete_thing"}


def test_stdio_lazy_initialize_on_list(stdio):
    # tools_list without an explicit initialize() still handshakes.
    tools = stdio.tools_list()
    assert any(t["name"] == "echo" for t in tools)
    assert stdio.health()["initialized"] is True


def test_stdio_tools_call(stdio):
    result = stdio.tools_call("echo", {"text": "hello"})
    text = result["content"][0]["text"]
    assert json.loads(text) == {"echoed": "hello"}


def test_stdio_health_and_stop(stdio):
    stdio.initialize()
    assert stdio.health()["running"] is True
    stdio.stop()
    assert stdio.health()["running"] is False


def test_stdio_unknown_tool_raises(stdio):
    stdio.initialize()
    with pytest.raises(UpstreamError):
        stdio.tools_call("does_not_exist", {})


def test_stdio_bad_command_raises():
    up = StdioUpstream(UpstreamConfig(
        server_id="bad", transport="stdio",
        command="this_binary_does_not_exist_xyz", args=[],
    ))
    with pytest.raises(UpstreamError):
        up.tools_list()


def test_stdio_concurrent_calls(stdio):
    stdio.initialize()
    results = {}
    errors = []

    def worker(i):
        try:
            r = stdio.tools_call("echo", {"text": f"n{i}"})
            results[i] = json.loads(r["content"][0]["text"])["echoed"]
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)
    assert not errors
    assert results == {i: f"n{i}" for i in range(8)}


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


class _FakeHttpMcp:
    def __init__(self, *, sse=False):
        self.sse = sse
        self.server = None
        self.thread = None
        self.port = 0
        self.session_header_seen = []

    def start(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a, **_k):
                return

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0") or "0")
                body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
                outer.session_header_seen.append(self.headers.get("Mcp-Session-Id"))
                method = body.get("method")
                mid = body.get("id")
                if mid is None:
                    self.send_response(202)
                    self.end_headers()
                    return
                if method == "initialize":
                    result = {"protocolVersion": "2025-11-25", "capabilities": {},
                              "serverInfo": {"name": "fake-http", "version": "0.0.1"}}
                elif method == "tools/list":
                    result = {"tools": [{"name": "ping", "description": "p", "inputSchema": {"type": "object"}}]}
                elif method == "tools/call":
                    result = {"content": [{"type": "text", "text": json.dumps({"pong": True})}], "isError": False}
                else:
                    result = {}
                env = {"jsonrpc": "2.0", "id": mid, "result": result}
                if outer.sse:
                    wire = f"event: message\ndata: {json.dumps(env)}\n\n".encode()
                    ctype = "text/event-stream"
                else:
                    wire = json.dumps(env).encode()
                    ctype = "application/json"
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Mcp-Session-Id", "sess-123")
                self.send_header("Content-Length", str(len(wire)))
                self.end_headers()
                self.wfile.write(wire)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]

    def stop(self):
        if self.server:
            try:
                self.server.shutdown()
            except Exception:
                pass


@pytest.fixture
def http_json():
    s = _FakeHttpMcp(sse=False)
    s.start()
    yield s
    s.stop()


@pytest.fixture
def http_sse():
    s = _FakeHttpMcp(sse=True)
    s.start()
    yield s
    s.stop()


def _http_up(server):
    return HttpUpstream(UpstreamConfig(
        server_id="fakehttp", transport="http",
        url=f"http://127.0.0.1:{server.port}/mcp",
        init_timeout=10.0, request_timeout=10.0,
    ))


def test_http_initialize_list_call(http_json):
    up = _http_up(http_json)
    info = up.initialize()
    assert info["serverInfo"]["name"] == "fake-http"
    tools = up.tools_list()
    assert tools[0]["name"] == "ping"
    result = up.tools_call("ping", {})
    assert json.loads(result["content"][0]["text"]) == {"pong": True}


def test_http_session_id_is_carried(http_json):
    up = _http_up(http_json)
    up.initialize()
    up.tools_list()
    # First request had no session header; later requests carry sess-123.
    assert http_json.session_header_seen[0] is None
    assert "sess-123" in http_json.session_header_seen[1:]


def test_http_sse_response_parsed(http_sse):
    up = _http_up(http_sse)
    tools = up.tools_list()
    assert tools[0]["name"] == "ping"


def test_http_unreachable_raises():
    up = HttpUpstream(UpstreamConfig(server_id="x", transport="http", url="http://127.0.0.1:1/mcp", request_timeout=2.0, init_timeout=2.0))
    with pytest.raises(UpstreamError):
        up.tools_list()


# ---------------------------------------------------------------------------
# helpers / factory / config validation
# ---------------------------------------------------------------------------


def test_parse_http_body_json():
    assert _parse_http_body('{"a": 1}', "application/json") == {"a": 1}


def test_parse_http_body_sse_takes_last_data():
    raw = "event: message\ndata: {\"id\": 1}\n\nevent: message\ndata: {\"id\": 2, \"result\": {}}\n\n"
    assert _parse_http_body(raw, "text/event-stream") == {"id": 2, "result": {}}


def test_parse_http_body_empty():
    assert _parse_http_body("", "application/json") == {}


def test_connect_factory_picks_transport():
    assert isinstance(connect(UpstreamConfig(server_id="a", transport="stdio", command="x")), StdioUpstream)
    assert isinstance(connect(UpstreamConfig(server_id="b", transport="http", url="http://x/mcp")), HttpUpstream)


def test_config_validation():
    with pytest.raises(UpstreamError):
        UpstreamConfig(server_id="a", transport="stdio").validate()
    with pytest.raises(UpstreamError):
        UpstreamConfig(server_id="a", transport="http").validate()
    with pytest.raises(UpstreamError):
        UpstreamConfig(server_id="a", transport="carrier-pigeon").validate()
