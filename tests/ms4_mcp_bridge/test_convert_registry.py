"""Phase 2 — conversion (namespacing, fail-closed kind) + registry
(persistence, refresh, secret hygiene)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from machine_spirit_4.mcp_bridge import (
    ImportedServer,
    Registry,
    UpstreamConfig,
    convert_tools,
    imported_tool_id,
    imported_tool_records,
    kind_for_tool,
    load,
    sanitize_segment,
    upstream_tool_to_record,
)
from machine_spirit_4.mcp_bridge.convert import KIND_READ_ONLY, KIND_RUNTIME_ACTION


@pytest.fixture(autouse=True)
def _isolate_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("MS4_MCP_IMPORTS_PATH", str(tmp_path / "mcp_imports.json"))
    monkeypatch.setenv("MS4_AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    yield


# ---------------------------------------------------------------------------
# convert
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("server,tool,expected", [
    ("github", "create_issue", "ext.github.create_issue@v1"),
    ("My-Server", "Do.Thing", "ext.my_server.do_thing@v1"),
    ("x y", "a b", "ext.x_y.a_b@v1"),
])
def test_imported_tool_id_namespacing(server, tool, expected):
    assert imported_tool_id(server, tool) == expected


def test_sanitize_segment():
    assert sanitize_segment("Hello-World.x") == "hello_world_x"
    assert sanitize_segment("") == "unknown"
    assert sanitize_segment("!!!") == "unknown"


def test_kind_fail_closed_default():
    # No annotations -> runtime_action even if allow_inline.
    assert kind_for_tool({}, allow_inline=True) == KIND_RUNTIME_ACTION


def test_kind_read_only_requires_hint_and_optin():
    ro = {"annotations": {"readOnlyHint": True, "destructiveHint": False}}
    # readOnlyHint but server NOT opted in -> still runtime_action
    assert kind_for_tool(ro, allow_inline=False) == KIND_RUNTIME_ACTION
    # readOnlyHint AND opted in -> read_only
    assert kind_for_tool(ro, allow_inline=True) == KIND_READ_ONLY


def test_kind_destructive_never_read_only():
    d = {"annotations": {"readOnlyHint": True, "destructiveHint": True}}
    assert kind_for_tool(d, allow_inline=True) == KIND_RUNTIME_ACTION


def test_upstream_tool_to_record_shape():
    rec = upstream_tool_to_record("github", {
        "name": "list_repos",
        "description": "List repositories",
        "inputSchema": {"type": "object", "properties": {"org": {"type": "string"}}},
        "annotations": {"readOnlyHint": True},
    }, allow_inline=True)
    assert rec["name"] == "ext.github.list_repos@v1"
    assert rec["server_id"] == "github"
    assert rec["upstream_name"] == "list_repos"
    assert rec["kind"] == KIND_READ_ONLY
    assert rec["input_schema"]["properties"]["org"]["type"] == "string"
    assert rec["schema"] == "Ms4ImportedTool.v1"


def test_upstream_tool_to_record_rejects_malformed():
    assert upstream_tool_to_record("s", {"description": "no name"}, allow_inline=True) is None
    assert upstream_tool_to_record("s", {"name": ""}, allow_inline=True) is None


def test_convert_tools_dedup_and_skip():
    tools = [
        {"name": "a", "description": "1"},
        {"name": "a", "description": "2"},  # dup id -> last wins
        "not a dict",
        {"no_name": True},
    ]
    out = convert_tools("srv", tools, allow_inline=False)
    assert len(out) == 1
    assert out[0]["description"] == "2"


# ---------------------------------------------------------------------------
# registry — fake upstream injection
# ---------------------------------------------------------------------------


class _FakeUpstream:
    def __init__(self, tools, *, fail=False):
        self._tools = tools
        self._fail = fail
        self.stopped = False

    def tools_list(self):
        if self._fail:
            raise RuntimeError("upstream down")
        return self._tools

    def stop(self):
        self.stopped = True


def _connect_fn(tools, *, fail=False):
    captured = {}

    def _fn(config: UpstreamConfig):
        captured["config"] = config
        return _FakeUpstream(tools, fail=fail)

    _fn.captured = captured
    return _fn


def test_register_and_refresh_populates_tools():
    reg = load()
    connect_fn = _connect_fn([
        {"name": "echo", "description": "e", "annotations": {"readOnlyHint": True}},
        {"name": "rm", "description": "d", "annotations": {"destructiveHint": True}},
    ])
    server = ImportedServer(server_id="demo", transport="stdio", command="x", allow_inline=True)
    reg.register(server, connect_fn=connect_fn)

    tools = {t["name"]: t for t in reg.all_tools()}
    assert "ext.demo.echo@v1" in tools
    assert tools["ext.demo.echo@v1"]["kind"] == KIND_READ_ONLY  # readOnly + opt-in
    assert tools["ext.demo.rm@v1"]["kind"] == KIND_RUNTIME_ACTION  # destructive


def test_register_persists_and_reloads():
    reg = load()
    reg.register(
        ImportedServer(server_id="demo", transport="stdio", command="x"),
        connect_fn=_connect_fn([{"name": "echo", "description": "e"}]),
    )
    # Reload from disk -> tools survive.
    reg2 = load()
    assert "ext.demo.echo@v1" in {t["name"] for t in reg2.all_tools()}
    assert reg2.get("demo").transport == "stdio"


def test_refresh_failure_captured_not_raised():
    reg = load()
    reg.register(
        ImportedServer(server_id="bad", transport="stdio", command="x"),
        connect_fn=_connect_fn([], fail=True),
    )
    server = reg.get("bad")
    assert server.last_error is not None
    assert "upstream down" in server.last_error
    assert server.tools == []


def test_remove():
    reg = load()
    reg.register(ImportedServer(server_id="demo", transport="stdio", command="x"),
                 connect_fn=_connect_fn([{"name": "echo"}]))
    assert reg.remove("demo") is True
    assert reg.remove("demo") is False
    assert load().get("demo") is None


def test_server_for_tool_lookup():
    reg = load()
    reg.register(ImportedServer(server_id="demo", transport="stdio", command="x"),
                 connect_fn=_connect_fn([{"name": "echo"}]))
    found = reg.server_for_tool("ext.demo.echo@v1")
    assert found is not None and found.server_id == "demo"
    assert reg.server_for_tool("ext.nope.x@v1") is None


def test_disabled_server_tools_excluded():
    reg = load()
    reg.register(ImportedServer(server_id="demo", transport="stdio", command="x", enabled=False),
                 connect_fn=_connect_fn([{"name": "echo"}]))
    assert reg.all_tools() == []


# ---------------------------------------------------------------------------
# secret hygiene
# ---------------------------------------------------------------------------


def test_env_passthrough_resolved_at_launch_not_persisted(monkeypatch):
    monkeypatch.setenv("MY_SECRET_TOKEN", "s3cr3t")
    server = ImportedServer(
        server_id="gh", transport="stdio", command="x",
        env={"PUBLIC_FLAG": "1"}, env_passthrough=["MY_SECRET_TOKEN"],
    )
    # Resolved config has the secret value...
    cfg = server.to_upstream_config()
    assert cfg.env["PUBLIC_FLAG"] == "1"
    assert cfg.env["MY_SECRET_TOKEN"] == "s3cr3t"
    # ...but the persisted form only keeps the NAME, never the value.
    d = server.to_dict()
    assert d["env"] == {"PUBLIC_FLAG": "1"}
    assert d["env_passthrough"] == ["MY_SECRET_TOKEN"]
    assert "s3cr3t" not in json.dumps(d)


def test_imported_tool_records_convenience():
    reg = load()
    reg.register(ImportedServer(server_id="demo", transport="stdio", command="x"),
                 connect_fn=_connect_fn([{"name": "echo", "description": "e"}]))
    records = imported_tool_records()
    assert any(r["name"] == "ext.demo.echo@v1" for r in records)


def test_audit_event_has_no_secret(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_SECRET_TOKEN", "s3cr3t")
    reg = load()
    reg.register(
        ImportedServer(server_id="gh", transport="stdio", command="x",
                       env_passthrough=["MY_SECRET_TOKEN"]),
        connect_fn=_connect_fn([{"name": "echo"}]),
    )
    audit = (Path(tmp_path) / "audit.jsonl").read_text(encoding="utf-8")
    assert "mcp_import_registered" in audit
    assert "s3cr3t" not in audit
