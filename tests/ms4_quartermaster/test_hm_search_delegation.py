"""Phase B (MS4 side) — cascade delegates to hivemind.tools.search@v1
when it's in the catalog, and falls back to the local engine when it's
absent or fails.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from machine_spirit_4.gateway.quartermaster import (
    BACKEND_TFIDF,
    HM_SEARCH_TOOL,
    TIER_DETERMINISTIC,
    TIER_HM_SEARCH,
    Catalog,
    ToolEntry,
    build_index,
    resolve,
)
from machine_spirit_4.gateway.quartermaster import catalog as catalog_mod
from machine_spirit_4.gateway.quartermaster import taxonomy


def _entry(name, description):
    domain = taxonomy.tool_domain(name)
    tb = taxonomy.canonical_toolbox(domain)
    return ToolEntry(
        schema="Ms4QuartermasterTool.v1", name=name, toolbox=tb,
        cluster=taxonomy.cluster_for_toolbox(tb), description=description,
        source="hivemind", kind="hivemind_native",
    )


def _catalog(entries):
    by_toolbox = {}
    for e in entries:
        by_toolbox.setdefault(e.toolbox, []).append(e)
    return Catalog(
        schema="Ms4QuartermasterCatalog.v1", version=catalog_mod._version_of(entries),
        built_at="2026-05-30T00:00:00+00:00", hivemind_url="http://test",
        tools=tuple(sorted(entries, key=lambda t: t.name)),
        toolboxes={tb: tuple(sorted(es, key=lambda t: t.name)) for tb, es in by_toolbox.items()},
        sources={"hivemind": len(entries), "ms4": 0}, errors=(),
    )


class _FakeSearchServer:
    """Fakes hivemind.tools.search@v1 via tools/call."""

    def __init__(self):
        self.response_tools: list[dict[str, Any]] = []
        self.calls = 0
        self.fail = False
        self.server = None
        self.thread = None
        self.port = 0

    def start(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a, **_k):
                return

            def do_POST(self):
                outer.calls += 1
                length = int(self.headers.get("Content-Length", "0") or "0")
                _ = self.rfile.read(length)
                if outer.fail:
                    self.send_response(500)
                    self.end_headers()
                    return
                payload = {
                    "jsonrpc": "2.0", "id": 1,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps({
                            "query": "q", "catalog_version": "x", "tools": outer.response_tools,
                        })}],
                        "isError": False,
                    },
                }
                wire = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
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
def fake_search(monkeypatch):
    s = _FakeSearchServer()
    s.start()
    monkeypatch.setenv("MS4_HIVEMIND_MCP_URL", f"http://127.0.0.1:{s.port}/mcp")
    monkeypatch.setenv("MS4_QM_DELEGATE", "1")
    monkeypatch.delenv("MS4_HIVEMIND_API_KEY", raising=False)
    yield s
    s.stop()


def _cat_with_search():
    return _catalog([
        _entry(HM_SEARCH_TOOL, "Search the tool catalog"),
        _entry("hivemind.vm.list@v1", "List the virtual machines"),
        _entry("hivemind.gpu.availability@v1", "Free GPUs"),
    ])


def test_delegates_to_hm_search_when_present(fake_search):
    cat = _cat_with_search()
    idx = build_index(cat, backend=BACKEND_TFIDF)
    fake_search.response_tools = [
        {"name": "hivemind.gpu.availability@v1", "score": 0.92, "description": "Free GPUs"},
        {"name": "hivemind.vm.list@v1", "score": 0.4, "description": "List VMs"},
    ]
    url = f"http://127.0.0.1:{fake_search.port}"
    res = resolve("anything", catalog=cat, index=idx, hivemind_url=url)
    assert res.tier == TIER_HM_SEARCH
    assert res.top_tool().name == "hivemind.gpu.availability@v1"
    assert res.confidence == pytest.approx(0.92)
    # kind/inline_eligible re-attached from the local catalog
    assert res.top_tool().inline_eligible is True
    assert fake_search.calls == 1


def test_falls_back_to_local_when_search_absent(fake_search):
    # Catalog WITHOUT the search tool -> no delegation, local engine.
    cat = _catalog([
        _entry("hivemind.vm.list@v1", "List the virtual machines running or stopped"),
        _entry("hivemind.gpu.availability@v1", "Free GPUs"),
    ])
    idx = build_index(cat, backend=BACKEND_TFIDF)
    url = f"http://127.0.0.1:{fake_search.port}"
    res = resolve("list my virtual machines", catalog=cat, index=idx, hivemind_url=url)
    assert res.tier == TIER_DETERMINISTIC
    assert fake_search.calls == 0  # never called the search server


def test_falls_back_to_local_when_search_errors(fake_search):
    cat = _cat_with_search()
    idx = build_index(cat, backend=BACKEND_TFIDF)
    fake_search.fail = True
    url = f"http://127.0.0.1:{fake_search.port}"
    res = resolve("list my virtual machines", catalog=cat, index=idx, hivemind_url=url)
    # Delegation failed (500) -> local deterministic tier answered.
    assert res.tier == TIER_DETERMINISTIC
    assert res.top_tool().name == "hivemind.vm.list@v1"


def test_falls_back_when_search_returns_empty(fake_search):
    cat = _cat_with_search()
    idx = build_index(cat, backend=BACKEND_TFIDF)
    fake_search.response_tools = []  # empty -> not usable
    url = f"http://127.0.0.1:{fake_search.port}"
    res = resolve("list my virtual machines", catalog=cat, index=idx, hivemind_url=url)
    assert res.tier == TIER_DETERMINISTIC


def test_delegation_disabled_by_env(fake_search, monkeypatch):
    monkeypatch.setenv("MS4_QM_DELEGATE", "0")
    cat = _cat_with_search()
    idx = build_index(cat, backend=BACKEND_TFIDF)
    fake_search.response_tools = [{"name": "hivemind.gpu.availability@v1", "score": 0.9}]
    url = f"http://127.0.0.1:{fake_search.port}"
    res = resolve("list my virtual machines", catalog=cat, index=idx, hivemind_url=url)
    assert res.tier == TIER_DETERMINISTIC
    assert fake_search.calls == 0


def test_delegated_tool_not_in_local_catalog_is_synthesized(fake_search):
    """A delegated result for a tool MS4 doesn't have locally still
    gets kind/inline_eligible derived so the router can gate it."""
    cat = _cat_with_search()
    idx = build_index(cat, backend=BACKEND_TFIDF)
    fake_search.response_tools = [
        {"name": "hivemind.newdomain.list@v1", "score": 0.8, "description": "A brand new read tool"},
    ]
    url = f"http://127.0.0.1:{fake_search.port}"
    res = resolve("anything", catalog=cat, index=idx, hivemind_url=url)
    assert res.tier == TIER_HM_SEARCH
    top = res.top_tool()
    assert top.name == "hivemind.newdomain.list@v1"
    assert top.toolbox == "newdomain"
    assert top.inline_eligible is True  # verb "list" is read-only
