"""Phase 5 — imported tools in MS4's MCP server registry + the gateway
REST importer endpoints (register/list/refresh/remove/health)."""

from __future__ import annotations

import io
import json
import types

import pytest

import machine_spirit_4.gateway.server as srv_module
from machine_spirit_4.mcp_bridge import ImportedServer, UpstreamConfig, load


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("MS4_MCP_IMPORTS_PATH", str(tmp_path / "mcp_imports.json"))
    monkeypatch.setenv("MS4_AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    yield


class _FakeUpstream:
    def __init__(self, tools):
        self._tools = tools

    def tools_list(self):
        return self._tools

    def stop(self):
        pass


def _connect_fn(tools):
    def _fn(_config: UpstreamConfig):
        return _FakeUpstream(tools)
    return _fn


# A command guaranteed not to exist, so even a botched connect injection
# fails fast instead of launching a real long-lived process.
_BOGUS_CMD = "ms4_nonexistent_cmd_xyz"


def _register_demo(allow_inline=False, tools=None):
    reg = load()
    reg.register(
        ImportedServer(server_id="demo", transport="stdio", command=_BOGUS_CMD, allow_inline=allow_inline),
        connect_fn=_connect_fn(tools if tools is not None else [
            {"name": "echo", "description": "Echo text", "annotations": {"readOnlyHint": True}},
        ]),
    )
    return reg


# ---------------------------------------------------------------------------
# MS4 MCP server registry
# ---------------------------------------------------------------------------


def test_build_tool_registry_includes_imported_tools():
    _register_demo()
    from machine_spirit_4.mcp.tools import build_tool_registry

    registry = build_tool_registry()
    assert "ext.demo.echo@v1" in registry
    tool = registry["ext.demo.echo@v1"]
    assert tool.description.startswith("[ext:demo]")
    mcp = tool.as_mcp_tool()
    assert mcp["name"] == "ext.demo.echo@v1"
    assert mcp["annotations"]["openWorldHint"] is True


def test_imported_tool_handler_proxies(monkeypatch):
    _register_demo()
    from machine_spirit_4.mcp.tools import build_tool_registry
    import machine_spirit_4.mcp_bridge.proxy as proxy_mod

    captured = {}

    def fake_call(tool_id, args):
        captured["tool"] = tool_id
        captured["args"] = args
        return {"ok": True, "is_error": False, "result": {"echoed": args.get("text")}, "server_id": "demo"}

    monkeypatch.setattr(proxy_mod, "call_imported_tool", fake_call)

    registry = build_tool_registry()
    handler = registry["ext.demo.echo@v1"].handler
    out = handler(object(), {"text": "hi"})
    assert out["result"] == {"echoed": "hi"}
    assert captured["tool"] == "ext.demo.echo@v1"


def test_imported_tool_handler_raises_on_proxy_failure(monkeypatch):
    _register_demo()
    from machine_spirit_4.mcp.tools import build_tool_registry
    import machine_spirit_4.mcp_bridge.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "call_imported_tool", lambda t, a: {"ok": False, "error": "down"})

    registry = build_tool_registry()
    handler = registry["ext.demo.echo@v1"].handler
    with pytest.raises(RuntimeError):
        handler(object(), {})


def test_no_imported_tools_when_registry_empty():
    from machine_spirit_4.mcp.tools import build_tool_registry

    registry = build_tool_registry()
    assert not any(name.startswith("ext.") for name in registry)
    # native tools still present
    assert "ms4.identity.verify@v1" in registry


# ---------------------------------------------------------------------------
# Gateway REST endpoints
# ---------------------------------------------------------------------------


class _DummyRunner:
    def __init__(self):
        self.hivemind_url = "http://hive:6089"
        self.ms3_url = "http://ms3:9080"


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


def _read_response(handler) -> tuple[int, dict]:
    raw = handler.wfile.getvalue()
    status_line, _, rest = raw.partition(b"\r\n")
    status = int(status_line.split(b" ")[1])
    _head, _, body = rest.partition(b"\r\n\r\n")
    return status, json.loads(body.decode("utf-8") or "{}")


def test_rest_register_then_list(monkeypatch):
    # Register without refresh (no upstream launch) so the test is offline.
    body = json.dumps({
        "server_id": "github",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-github"],
        "env_passthrough": ["GITHUB_TOKEN"],
        "refresh": False,
    }).encode()
    h = _make_handler("POST", "/api/v1/mcp/imports", body)
    h._mcp_imports_register()
    status, payload = _read_response(h)
    assert status == 200
    assert payload["server_id"] == "github"
    assert payload["transport"] == "stdio"

    # List shows it.
    h2 = _make_handler("GET", "/api/v1/mcp/imports")
    h2._mcp_imports_list()
    status2, payload2 = _read_response(h2)
    assert status2 == 200
    assert any(s["server_id"] == "github" for s in payload2["servers"])


def test_rest_register_validation():
    body = json.dumps({"server_id": "x"}).encode()  # missing transport
    h = _make_handler("POST", "/api/v1/mcp/imports", body)
    h._mcp_imports_register()
    status, payload = _read_response(h)
    assert status == 400
    assert "transport" in payload["error"]


def test_rest_register_with_refresh_populates_tools(monkeypatch):
    # Patch connect so register(refresh=True) uses a fake upstream.
    import machine_spirit_4.mcp_bridge.registry as reg_mod

    def fake_connect(config):
        return _FakeUpstream([{"name": "echo", "description": "e", "annotations": {"readOnlyHint": True}}])

    monkeypatch.setattr(reg_mod, "connect", fake_connect)
    body = json.dumps({
        "server_id": "demo", "transport": "stdio", "command": _BOGUS_CMD,
        "allow_inline": True, "refresh": True,
    }).encode()
    h = _make_handler("POST", "/api/v1/mcp/imports", body)
    h._mcp_imports_register()
    status, payload = _read_response(h)
    assert status == 200
    assert payload["tool_count"] == 1


def test_rest_health(monkeypatch):
    _register_demo()
    h = _make_handler("GET", "/api/v1/mcp/imports/demo/health")
    h._mcp_imports_health("demo")
    status, payload = _read_response(h)
    assert status == 200
    assert payload["server_id"] == "demo"
    assert payload["tool_count"] == 1


def test_rest_health_unknown():
    h = _make_handler("GET", "/api/v1/mcp/imports/nope/health")
    h._mcp_imports_health("nope")
    status, payload = _read_response(h)
    assert status == 404


def test_rest_refresh(monkeypatch):
    _register_demo()
    import machine_spirit_4.mcp_bridge.registry as reg_mod
    monkeypatch.setattr(reg_mod, "connect",
                        lambda c: _FakeUpstream([{"name": "echo"}, {"name": "echo2"}]))
    h = _make_handler("POST", "/api/v1/mcp/imports/demo/refresh")
    h._mcp_imports_refresh("demo")
    status, payload = _read_response(h)
    assert status == 200
    assert payload["tool_count"] == 2


def test_rest_remove(monkeypatch):
    _register_demo()
    h = _make_handler("DELETE", "/api/v1/mcp/imports/demo")
    h._mcp_imports_remove("demo")
    status, payload = _read_response(h)
    assert status == 200
    assert payload["removed"] is True
    assert load().get("demo") is None


def test_rest_remove_unknown():
    h = _make_handler("DELETE", "/api/v1/mcp/imports/nope")
    h._mcp_imports_remove("nope")
    status, payload = _read_response(h)
    assert status == 404
    assert payload["removed"] is False
