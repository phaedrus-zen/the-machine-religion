"""Phase 4 — imported 3rd-party MCP tools in the Quartermaster.

Catalog third source (external_mcp), taxonomy ext.<server>.* -> server
toolbox, fail-closed inline eligibility, cascade routing, and the
executor routing ext.* through the bridge.
"""

from __future__ import annotations

import pytest

from machine_spirit_4.gateway.quartermaster import (
    BACKEND_TFIDF,
    build_index,
    is_inline_eligible,
    merge_records,
    resolve,
)
from machine_spirit_4.gateway.quartermaster import taxonomy
from machine_spirit_4.gateway.quartermaster.catalog import (
    KIND_READ_ONLY,
    KIND_RUNTIME_ACTION,
    SOURCE_EXTERNAL_MCP,
    _ext_entry_to_tool,
)


def _imported(name, desc, kind=KIND_RUNTIME_ACTION, input_schema=None):
    return {
        "schema": "Ms4ImportedTool.v1",
        "name": name,
        "server_id": name.split(".")[1],
        "upstream_name": name.split(".")[2].split("@")[0],
        "description": desc,
        "kind": kind,
        "input_schema": input_schema or {"type": "object", "properties": {}},
        "annotations": {},
    }


# ---------------------------------------------------------------------------
# taxonomy
# ---------------------------------------------------------------------------


def test_ext_tool_domain():
    assert taxonomy.tool_domain("ext.github.create_issue@v1") == "github"
    assert taxonomy.tool_domain("ext.slack.post_message@v1") == "slack"
    assert taxonomy.tool_domain("ext@v1") == "ext"


# ---------------------------------------------------------------------------
# catalog third source
# ---------------------------------------------------------------------------


def test_imported_tools_enter_catalog_as_external_source():
    cat = merge_records(
        hivemind_records=[{"name": "hivemind.vm.list@v1", "description": "List VMs"}],
        ms4_records=[],
        imported_records=[
            _imported("ext.github.list_issues@v1", "List issues", kind=KIND_READ_ONLY),
            _imported("ext.github.create_issue@v1", "Create an issue", kind=KIND_RUNTIME_ACTION),
        ],
        hivemind_url="test",
    )
    by_name = cat.by_name()
    assert "ext.github.list_issues@v1" in by_name
    entry = by_name["ext.github.list_issues@v1"]
    assert entry.source == SOURCE_EXTERNAL_MCP
    assert entry.toolbox == "github"
    assert entry.cluster == "external_mcp"
    assert cat.sources["external_mcp"] == 2


def test_ext_entry_to_tool_defaults_runtime_action_on_missing_kind():
    entry = _ext_entry_to_tool({"name": "ext.x.y@v1", "description": "d"})
    assert entry.kind == KIND_RUNTIME_ACTION
    assert entry.cluster == "external_mcp"


# ---------------------------------------------------------------------------
# fail-closed inline eligibility
# ---------------------------------------------------------------------------


def test_imported_runtime_action_not_inline_eligible():
    entry = _ext_entry_to_tool(_imported("ext.github.create_issue@v1", "Create", kind=KIND_RUNTIME_ACTION))
    assert is_inline_eligible(entry) is False


def test_imported_read_only_is_inline_eligible():
    # kind=read_only is only set by convert when readOnlyHint + opt-in.
    entry = _ext_entry_to_tool(_imported("ext.github.list_issues@v1", "List", kind=KIND_READ_ONLY))
    assert is_inline_eligible(entry) is True


def test_imported_read_only_with_arbitrary_verb_still_eligible():
    """External read-only tools don't need to match the HiveMind verb
    allowlist (their verb 'fetch_weather' isn't on it) — kind is the gate."""
    entry = _ext_entry_to_tool(_imported("ext.weather.fetch_weather@v1", "Get weather", kind=KIND_READ_ONLY))
    assert is_inline_eligible(entry) is True


# ---------------------------------------------------------------------------
# cascade routing
# ---------------------------------------------------------------------------


def test_cascade_can_resolve_imported_tool():
    cat = merge_records(
        hivemind_records=[{"name": "hivemind.vm.list@v1", "description": "List virtual machines"}],
        ms4_records=[],
        imported_records=[
            _imported("ext.github.search_issues@v1", "Search GitHub issues and pull requests", kind=KIND_READ_ONLY),
        ],
        hivemind_url="test",
    )
    idx = build_index(cat, backend=BACKEND_TFIDF)
    res = resolve("search github issues", catalog=cat, index=idx)
    names = [t.name for t in res.tools]
    assert "ext.github.search_issues@v1" in names


# ---------------------------------------------------------------------------
# executor routes ext.* through the bridge
# ---------------------------------------------------------------------------


def test_executor_routes_ext_tool_through_bridge(monkeypatch):
    from machine_spirit_4.gateway.quartermaster import executor

    calls = {}

    def fake_call_imported_tool(tool_id, args):
        calls["tool"] = tool_id
        calls["args"] = args
        return {"ok": True, "is_error": False, "result": {"issues": [1, 2, 3]}}

    import machine_spirit_4.mcp_bridge.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "call_imported_tool", fake_call_imported_tool)

    out = executor.execute_inline_tool("http://hive:6089", "ext.github.list_issues@v1", arguments={"org": "x"})
    assert out is not None
    assert out["tool"] == "ext.github.list_issues@v1"
    assert out["result"] == {"issues": [1, 2, 3]}
    assert calls["tool"] == "ext.github.list_issues@v1"
    assert calls["args"] == {"org": "x"}


def test_executor_ext_bridge_error_returns_none(monkeypatch):
    from machine_spirit_4.gateway.quartermaster import executor
    import machine_spirit_4.mcp_bridge.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "call_imported_tool",
                        lambda t, a: {"ok": False, "error": "upstream down"})
    out = executor.execute_inline_tool("http://hive:6089", "ext.github.list_issues@v1")
    assert out is None  # fail-soft -> Depth
