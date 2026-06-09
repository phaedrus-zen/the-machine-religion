"""Phase 3 — proxy executor: resolve, call, unwrap, fail-soft, retry,
connection reuse."""

from __future__ import annotations

import json

import pytest

from machine_spirit_4.mcp_bridge import ImportedServer, Registry, UpstreamConfig
from machine_spirit_4.mcp_bridge.proxy import BridgeProxy, _unwrap


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("MS4_MCP_IMPORTS_PATH", str(tmp_path / "mcp_imports.json"))
    monkeypatch.setenv("MS4_AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    yield


class _FakeUpstream:
    def __init__(self, *, behavior):
        self.behavior = behavior
        self.calls = 0
        self.stopped = 0

    def tools_list(self):
        return [{"name": "echo", "description": "e"}]

    def tools_call(self, name, arguments):
        self.calls += 1
        if self.behavior == "ok":
            return {"content": [{"type": "text", "text": json.dumps({"echoed": arguments.get("text")})}], "isError": False}
        if self.behavior == "tool_error":
            return {"content": [{"type": "text", "text": "boom"}], "isError": True}
        if self.behavior == "raise_once":
            if self.calls == 1:
                raise RuntimeError("transient")
            return {"content": [{"type": "text", "text": json.dumps({"recovered": True})}], "isError": False}
        if self.behavior == "raise_always":
            raise RuntimeError("dead")
        raise AssertionError(self.behavior)

    def stop(self):
        self.stopped += 1


def _registry_with(server_id="demo", tools=None, *, enabled=True):
    reg = Registry([], )
    server = ImportedServer(server_id=server_id, transport="stdio", command="x", enabled=enabled)
    server.tools = tools if tools is not None else [
        {"name": f"ext.{server_id}.echo@v1", "upstream_name": "echo", "kind": "runtime_action"},
    ]
    reg._servers[server_id] = server
    return reg


def _proxy(behavior, *, connect_calls=None):
    fake = _FakeUpstream(behavior=behavior)

    def connect_fn(config: UpstreamConfig):
        if connect_calls is not None:
            connect_calls.append(config.server_id)
        return fake

    proxy = BridgeProxy(connect_fn=connect_fn, load_fn=lambda: _registry_with())
    return proxy, fake


def test_call_ok_unwraps_result():
    proxy, fake = _proxy("ok")
    out = proxy.call("ext.demo.echo@v1", {"text": "hi"})
    assert out["ok"] is True
    assert out["result"] == {"echoed": "hi"}
    assert out["is_error"] is False
    assert out["server_id"] == "demo"


def test_call_preserves_is_error():
    proxy, fake = _proxy("tool_error")
    out = proxy.call("ext.demo.echo@v1", {})
    assert out["ok"] is True  # transport succeeded
    assert out["is_error"] is True  # but the tool reported an error


def test_unknown_tool_returns_error_envelope():
    proxy, _ = _proxy("ok")
    out = proxy.call("ext.demo.nope@v1", {})
    assert out["ok"] is False
    assert "unknown imported tool" in out["error"]


def test_disabled_server_returns_error():
    fake = _FakeUpstream(behavior="ok")
    proxy = BridgeProxy(connect_fn=lambda c: fake, load_fn=lambda: _registry_with(enabled=False))
    out = proxy.call("ext.demo.echo@v1", {})
    assert out["ok"] is False
    assert "disabled" in out["error"]


def test_transient_error_reconnects_and_retries():
    connect_calls = []
    proxy, fake = _proxy("raise_once", connect_calls=connect_calls)
    out = proxy.call("ext.demo.echo@v1", {})
    assert out["ok"] is True
    assert out["result"] == {"recovered": True}
    # connected twice (initial + reconnect after the transient failure)
    assert len(connect_calls) == 2
    assert fake.stopped >= 1  # dead connection dropped


def test_persistent_error_returns_envelope_after_retry():
    proxy, fake = _proxy("raise_always")
    out = proxy.call("ext.demo.echo@v1", {})
    assert out["ok"] is False
    assert "dead" in out["error"]
    assert fake.calls == 2  # tried twice


def test_connection_is_reused_across_calls():
    connect_calls = []
    proxy, fake = _proxy("ok", connect_calls=connect_calls)
    proxy.call("ext.demo.echo@v1", {"text": "a"})
    proxy.call("ext.demo.echo@v1", {"text": "b"})
    assert len(connect_calls) == 1  # one connection reused
    assert fake.calls == 2


def test_shutdown_all_stops_connections():
    proxy, fake = _proxy("ok")
    proxy.call("ext.demo.echo@v1", {})
    proxy.shutdown_all()
    assert fake.stopped >= 1


# ---------------------------------------------------------------------------
# _unwrap
# ---------------------------------------------------------------------------


def test_unwrap_json_content():
    raw = {"content": [{"type": "text", "text": json.dumps({"a": 1})}], "isError": False}
    assert _unwrap(raw) == {"a": 1}


def test_unwrap_non_json_text():
    raw = {"content": [{"type": "text", "text": "plain words"}]}
    assert _unwrap(raw) == {"raw_text": "plain words"}


def test_unwrap_bare_result():
    assert _unwrap({"ok": True}) == {"ok": True}


def test_unwrap_none():
    assert _unwrap("nope") is None
