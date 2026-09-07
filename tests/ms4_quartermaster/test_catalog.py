"""Catalog tests — manifest loading + live tools/list union +
version hashing + cache TTL.

In-process fake MCP for ``tools/list`` mirrors the pattern used by
``tests/ms4_gateway/test_hivemind_admin_modules.py``.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from machine_spirit_4.gateway.quartermaster import (
    KIND_HIVEMIND_NATIVE,
    KIND_READ_ONLY,
    KIND_RUNTIME_ACTION,
    Catalog,
    ToolEntry,
    build_catalog,
    get_catalog,
    is_inline_eligible,
    reset_catalog_cache_for_tests,
    tool_verb,
)


# ---------------------------------------------------------------------------
# In-process fake MCP server for tools/list
# ---------------------------------------------------------------------------


class _FakeMcp:
    """Just answers tools/list. The catalog uses tools/list (not
    tools/call), so we don't need the full content-wrap shape."""

    def __init__(self):
        self.tools: list[dict[str, Any]] = []
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.port: int = 0
        self.fail_with_status: int | None = None  # set to e.g. 500 to simulate cluster failure

    def set_tools(self, tools: list[dict[str, Any]]) -> None:
        self.tools = tools

    def start(self) -> None:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args, **_kwargs):
                return

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0") or "0")
                _ = self.rfile.read(length)
                if outer.fail_with_status is not None:
                    self.send_response(outer.fail_with_status)
                    self.end_headers()
                    return
                envelope = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {"tools": outer.tools},
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
    reset_catalog_cache_for_tests()
    yield server
    server.stop()
    reset_catalog_cache_for_tests()


def hurl(fake: _FakeMcp) -> str:
    return f"http://127.0.0.1:{fake.port}"


@pytest.fixture
def tiny_manifest(tmp_path, monkeypatch):
    """A minimal MS4 manifest the catalog can load deterministically."""
    manifest = {
        "schema": "Ms4McpManifest.v1",
        "tools": [
            {"name": "ms4.identity.verify@v1", "kind": "read_only"},
            {"name": "ms4.chat.send@v1", "kind": "runtime_action", "gated_by": ["ms4_consciousness"]},
            {"name": "ms4.hivemind.vms@v1", "kind": "read_only", "description": "VMs snapshot"},
            {"name": "ms4.hivemind.vm.start@v1", "kind": "runtime_action"},
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setenv("MS4_QM_MANIFEST_PATH", str(path))
    return path


# ---------------------------------------------------------------------------
# build_catalog
# ---------------------------------------------------------------------------


def test_build_catalog_unions_ms4_manifest_and_hivemind_tools(fake_mcp, tiny_manifest):
    fake_mcp.set_tools([
        {"name": "hivemind.vm.list@v1", "description": "List VMs in the cluster", "inputSchema": {}},
        {"name": "hivemind.gpu.availability@v1", "description": "Free GPUs", "inputSchema": {}},
        {"name": "hivemind.training.start@v1", "description": "Start a training job", "inputSchema": {}},
    ])
    catalog = build_catalog(hurl(fake_mcp))

    names = {t.name for t in catalog.tools}
    assert "ms4.identity.verify@v1" in names
    assert "ms4.chat.send@v1" in names
    assert "hivemind.vm.list@v1" in names
    assert "hivemind.gpu.availability@v1" in names

    # ms4.hivemind.* proxies are dropped from retrieval by default
    # (aliases of native hivemind.* tools).
    assert "ms4.hivemind.vms@v1" not in names
    assert "ms4.hivemind.vm.start@v1" not in names

    # Genuine MS4-native tools kept (2), proxies dropped (2); hivemind = 3.
    assert catalog.sources["ms4"] == 2
    assert catalog.sources["hivemind"] == 3

    by_name = catalog.by_name()
    assert by_name["hivemind.vm.list@v1"].toolbox == "vm"
    # The native vm tool occupies the vm toolbox bucket.
    vm_bucket = catalog.toolboxes["vm"]
    vm_names = {t.name for t in vm_bucket}
    assert "hivemind.vm.list@v1" in vm_names

    # Cluster derivation
    assert by_name["hivemind.vm.list@v1"].cluster == "compute_infra"
    assert by_name["ms4.chat.send@v1"].cluster == "ms4_local"
    assert by_name["hivemind.training.start@v1"].cluster == "ai_inference"


def test_ms4_hivemind_proxies_kept_when_drop_disabled(fake_mcp, tiny_manifest):
    """The drop is a default, not a hard rule — disabling it keeps the
    proxies (used by the eval to show the dedup matters)."""
    from machine_spirit_4.gateway.quartermaster import merge_records
    from machine_spirit_4.gateway.quartermaster.catalog import _load_ms4_manifest

    ms4_raw, _ = _load_ms4_manifest()
    catalog = merge_records(
        hivemind_records=[{"name": "hivemind.vm.list@v1", "description": "x"}],
        ms4_records=ms4_raw,
        hivemind_url="test",
        drop_ms4_hivemind_proxies=False,
    )
    names = {t.name for t in catalog.tools}
    assert "ms4.hivemind.vms@v1" in names


def test_build_catalog_kinds(fake_mcp, tiny_manifest):
    fake_mcp.set_tools([
        {"name": "hivemind.vm.list@v1", "description": "x"},
    ])
    catalog = build_catalog(hurl(fake_mcp))
    by_name = catalog.by_name()
    # MS4 manifest carries kind verbatim
    assert by_name["ms4.identity.verify@v1"].kind == KIND_READ_ONLY
    assert by_name["ms4.chat.send@v1"].kind == KIND_RUNTIME_ACTION
    assert by_name["ms4.chat.send@v1"].gated_by == ("ms4_consciousness",)
    # HiveMind native entries get the hivemind_native marker
    assert by_name["hivemind.vm.list@v1"].kind == KIND_HIVEMIND_NATIVE


def test_build_catalog_version_changes_with_content(fake_mcp, tiny_manifest):
    fake_mcp.set_tools([{"name": "hivemind.vm.list@v1", "description": "v1"}])
    c1 = build_catalog(hurl(fake_mcp))

    fake_mcp.set_tools([{"name": "hivemind.vm.list@v1", "description": "v1 updated"}])
    c2 = build_catalog(hurl(fake_mcp))

    fake_mcp.set_tools([
        {"name": "hivemind.vm.list@v1", "description": "v1"},
        {"name": "hivemind.vm.start@v1", "description": "start"},
    ])
    c3 = build_catalog(hurl(fake_mcp))

    assert c1.version != c2.version
    assert c1.version != c3.version
    # Versions are short content hashes (16 hex chars)
    assert all(len(c.version) == 16 for c in (c1, c2, c3))


def test_build_catalog_fail_soft_when_cluster_unreachable(fake_mcp, tiny_manifest):
    """tools/list 500 → no hivemind entries, errors captured, MS4
    manifest still surfaces. A degraded Quartermaster is still useful."""
    fake_mcp.fail_with_status = 500
    catalog = build_catalog(hurl(fake_mcp))
    assert catalog.sources["hivemind"] == 0
    # manifest still loaded; proxies dropped so 2 of the 4 remain
    assert catalog.sources["ms4"] == 2
    assert any("hivemind tools/list" in e for e in catalog.errors)


def test_build_catalog_fail_soft_when_manifest_missing(fake_mcp, monkeypatch):
    monkeypatch.setenv("MS4_QM_MANIFEST_PATH", "/path/that/definitely/does/not/exist.json")
    fake_mcp.set_tools([{"name": "hivemind.vm.list@v1", "description": "x"}])
    catalog = build_catalog(hurl(fake_mcp))
    assert catalog.sources["ms4"] == 0
    assert catalog.sources["hivemind"] == 1
    assert any("ms4 manifest not found" in e for e in catalog.errors)


def test_catalog_read_only_filter(fake_mcp, tiny_manifest):
    fake_mcp.set_tools([])
    catalog = build_catalog(hurl(fake_mcp))
    read_only = {t.name for t in catalog.read_only_tools()}
    assert "ms4.identity.verify@v1" in read_only
    # runtime_action is excluded
    assert "ms4.chat.send@v1" not in read_only
    # ms4.hivemind.vms proxy was dropped, so it's not here either
    assert "ms4.hivemind.vms@v1" not in read_only


# ---------------------------------------------------------------------------
# Inline eligibility — the safety-critical predicate (verb allowlist +
# non-destructive guard). This is what stops the inline fast-path from
# ever running a mutating tool.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,expected_verb", [
    ("hivemind.vm.list@v1", "list"),
    ("hivemind.gpu.availability@v1", "availability"),
    ("hivemind.service_health@v1", "service_health"),
    ("hivemind.capability.matrix@v1", "matrix"),
    ("hivemind.time.now@v1", "now"),
    ("hivemind.vm.force_stop@v1", "force_stop"),
    ("ms4.hivemind.vms@v1", "vms"),
    ("", ""),
])
def test_tool_verb(name, expected_verb):
    assert tool_verb(name) == expected_verb


def test_hivemind_native_read_only_tools_are_inline_eligible():
    """The core regression this guards: HiveMind-native read-only
    tools (kind='hivemind_native') MUST be inline-eligible via the
    verb allowlist. The prior dead-stub implementation made them all
    ineligible, defeating the inline fast-path."""
    from machine_spirit_4.gateway.quartermaster.catalog import _hivemind_entry_to_tool

    for name, desc in [
        ("hivemind.vm.list@v1", "List VMs"),
        ("hivemind.gpu.availability@v1", "Free GPUs"),
        ("hivemind.jobs.active@v1", "Active jobs"),
        ("hivemind.time.now@v1", "Cluster time"),
        ("hivemind.service_health@v1", "Service health"),
        ("hivemind.storage.volumes@v1", "Volumes"),
        ("hivemind.models.list@v1", "Models"),
    ]:
        entry = _hivemind_entry_to_tool({"name": name, "description": desc})
        assert entry is not None
        assert entry.kind == KIND_HIVEMIND_NATIVE
        assert is_inline_eligible(entry), f"{name} should be inline-eligible"


def test_destructive_hivemind_tools_are_not_inline_eligible():
    """Hard safety gate: mutating verbs never inline, even when
    HiveMind-native (no explicit kind)."""
    from machine_spirit_4.gateway.quartermaster.catalog import _hivemind_entry_to_tool

    for name in [
        "hivemind.vm.start@v1",
        "hivemind.vm.force_stop@v1",
        "hivemind.vm.delete@v1",
        "hivemind.storage.volume_create@v1",
        "hivemind.gpu_mode.set@v1",
        "hivemind.services.restart@v1",
        "hivemind.training.start@v1",
        "hivemind.jobs.cancel@v1",
    ]:
        entry = _hivemind_entry_to_tool({"name": name, "description": "mutates"})
        assert entry is not None
        assert not is_inline_eligible(entry), f"{name} must NOT be inline-eligible"


def test_ms4_tools_need_read_only_kind_and_verb(fake_mcp, tiny_manifest):
    fake_mcp.set_tools([])
    catalog = build_catalog(hurl(fake_mcp))
    by_name = catalog.by_name()
    # read_only kind + allowlisted verb (verify is not in the allowlist
    # though, so it is NOT eligible — that's fine, it's reachable via Depth)
    # ms4.hivemind.vms@v1 -> verb "vms" is NOT in the allowlist either.
    # Confirm the gate: ms4.chat.send@v1 (runtime_action) is never eligible.
    assert not is_inline_eligible(by_name["ms4.chat.send@v1"])


def test_ms4_runtime_action_with_readonly_verb_still_blocked():
    """A runtime_action MS4 tool whose verb happens to be allowlisted
    must STILL be blocked (kind gate wins)."""
    entry = ToolEntry(
        schema="Ms4QuartermasterTool.v1",
        name="ms4.something.list@v1",
        toolbox="something",
        cluster="unclassified",
        description="x",
        source="ms4",
        kind=KIND_RUNTIME_ACTION,
    )
    assert not is_inline_eligible(entry)


def test_confirm_gated_tool_never_eligible():
    entry = ToolEntry(
        schema="Ms4QuartermasterTool.v1",
        name="hivemind.something.list@v1",
        toolbox="something",
        cluster="unclassified",
        description="x",
        source="hivemind",
        kind=KIND_HIVEMIND_NATIVE,
        gated_by=("confirm:true required",),
    )
    assert not is_inline_eligible(entry)


def test_catalog_inline_eligible_tools_includes_hivemind_native(fake_mcp, tiny_manifest):
    fake_mcp.set_tools([
        {"name": "hivemind.gpu.availability@v1", "description": "Free GPUs"},
        {"name": "hivemind.vm.start@v1", "description": "Start a VM"},
    ])
    catalog = build_catalog(hurl(fake_mcp))
    eligible = {t.name for t in catalog.inline_eligible_tools()}
    assert "hivemind.gpu.availability@v1" in eligible
    assert "hivemind.vm.start@v1" not in eligible


def test_catalog_cluster_summary_counts(fake_mcp, tiny_manifest):
    fake_mcp.set_tools([
        {"name": "hivemind.vm.list@v1", "description": "x"},
        {"name": "hivemind.gpu.availability@v1", "description": "x"},
        {"name": "hivemind.training.start@v1", "description": "x"},
    ])
    catalog = build_catalog(hurl(fake_mcp))
    summary = catalog.cluster_summary()
    # 1 ms4_local (identity, chat) + 2 compute_infra (vm+vms+vm.start+vm.list+gpu) + 1 ai_inference
    assert summary["compute_infra"] >= 1
    assert summary["ai_inference"] >= 1
    assert summary["ms4_local"] >= 1


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def test_get_catalog_caches_within_ttl(fake_mcp, tiny_manifest, monkeypatch):
    monkeypatch.setenv("MS4_QM_CATALOG_TTL_SECS", "60")
    fake_mcp.set_tools([{"name": "hivemind.vm.list@v1", "description": "first"}])
    c1 = get_catalog(hurl(fake_mcp))
    fake_mcp.set_tools([{"name": "hivemind.vm.list@v1", "description": "second"}])
    c2 = get_catalog(hurl(fake_mcp))
    # Same version → cache hit
    assert c1.version == c2.version
    by_name = c2.by_name()
    assert by_name["hivemind.vm.list@v1"].description == "first"


def test_get_catalog_force_refresh(fake_mcp, tiny_manifest):
    fake_mcp.set_tools([{"name": "hivemind.vm.list@v1", "description": "first"}])
    c1 = get_catalog(hurl(fake_mcp))
    fake_mcp.set_tools([{"name": "hivemind.vm.list@v1", "description": "second"}])
    c2 = get_catalog(hurl(fake_mcp), force_refresh=True)
    assert c1.version != c2.version
    by_name = c2.by_name()
    assert by_name["hivemind.vm.list@v1"].description == "second"


def test_get_catalog_short_ttl_refreshes(fake_mcp, tiny_manifest):
    fake_mcp.set_tools([{"name": "hivemind.vm.list@v1", "description": "first"}])
    c1 = get_catalog(hurl(fake_mcp), ttl_secs=5)
    fake_mcp.set_tools([{"name": "hivemind.vm.list@v1", "description": "second"}])
    # ttl_secs=0 would be clamped by _env_int's minimum but explicit kwarg
    # bypasses the env helper. Pass a tiny positive value to keep behavior
    # under test but force the entry to look expired.
    import time
    # Manually expire by reaching into the cache:
    from machine_spirit_4.gateway.quartermaster import catalog as cat_mod
    with cat_mod._CACHE_LOCK:
        entry = cat_mod._CACHE.get(hurl(fake_mcp))
        if entry:
            entry.expires_at = time.time() - 1
    c2 = get_catalog(hurl(fake_mcp))
    assert c1.version != c2.version
