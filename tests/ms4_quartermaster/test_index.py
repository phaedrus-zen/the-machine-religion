"""Index tests — TF-IDF backend determinism + HiveMind backend
shape + two-stage retrieval + on-disk cache.

The HiveMind backend uses a fake embeddings server reusing the same
in-process MCP harness as the catalog tests.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from machine_spirit_4.gateway.quartermaster import (
    BACKEND_HIVEMIND,
    BACKEND_TFIDF,
    Catalog,
    IndexHit,
    ToolEntry,
    ToolIndex,
    build_index,
    get_index,
    query_toolboxes,
    query_tools,
    reset_catalog_cache_for_tests,
    reset_index_cache_for_tests,
)
from machine_spirit_4.gateway.quartermaster import catalog as catalog_mod
from machine_spirit_4.gateway.quartermaster import index as index_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(name: str, description: str, *, source: str = "hivemind", kind: str = "read_only") -> ToolEntry:
    """Build a ToolEntry directly bypassing the catalog so index
    tests don't depend on the full manifest+tools/list flow."""
    from machine_spirit_4.gateway.quartermaster import taxonomy
    domain = taxonomy.tool_domain(name)
    toolbox = taxonomy.canonical_toolbox(domain)
    return ToolEntry(
        schema="Ms4QuartermasterTool.v1",
        name=name,
        toolbox=toolbox,
        cluster=taxonomy.cluster_for_toolbox(toolbox),
        description=description,
        source=source,
        kind=kind,
    )


def _make_catalog(entries: list[ToolEntry]) -> Catalog:
    by_toolbox: dict[str, list[ToolEntry]] = {}
    for e in entries:
        by_toolbox.setdefault(e.toolbox, []).append(e)
    version = catalog_mod._version_of(entries)
    return Catalog(
        schema="Ms4QuartermasterCatalog.v1",
        version=version,
        built_at="2026-05-30T00:00:00+00:00",
        hivemind_url="http://test",
        tools=tuple(sorted(entries, key=lambda t: t.name)),
        toolboxes={tb: tuple(sorted(es, key=lambda t: t.name)) for tb, es in by_toolbox.items()},
        sources={"hivemind": len(entries), "ms4": 0},
        errors=(),
    )


@pytest.fixture(autouse=True)
def _reset_caches(tmp_path, monkeypatch):
    # Put the on-disk cache somewhere that doesn't survive the test.
    monkeypatch.setenv("MS4_QM_INDEX_DIR", str(tmp_path / "qm_idx"))
    monkeypatch.setenv("MS4_QM_USE_HM_EMBEDDINGS", "0")
    reset_catalog_cache_for_tests()
    reset_index_cache_for_tests()
    yield
    reset_index_cache_for_tests()


# ---------------------------------------------------------------------------
# TF-IDF backend
# ---------------------------------------------------------------------------


def test_tfidf_build_index_shape():
    catalog = _make_catalog([
        _make_entry("hivemind.vm.list@v1", "List all virtual machines in the cluster"),
        _make_entry("hivemind.vm.start@v1", "Start a virtual machine by name"),
        _make_entry("hivemind.gpu.availability@v1", "Report free GPUs in the cluster"),
        _make_entry("hivemind.time.now@v1", "Authoritative cluster time"),
    ])
    idx = build_index(catalog, backend=BACKEND_TFIDF)
    assert idx.backend == BACKEND_TFIDF
    assert idx.version == catalog.version
    assert len(idx.tool_vectors) == 4
    # Two distinct toolboxes (vm collapsed) + gpu + time
    assert set(idx.toolbox_vectors.keys()) == {"vm", "gpu", "time"}
    # Every tool vector has the same dimensionality (dense vocab)
    dims = {len(v) for v in idx.tool_vectors.values()}
    assert len(dims) == 1


def test_tfidf_query_toolboxes_top_match():
    catalog = _make_catalog([
        _make_entry("hivemind.vm.list@v1", "List all virtual machines in the cluster"),
        _make_entry("hivemind.gpu.availability@v1", "Report free GPUs in the cluster"),
        _make_entry("hivemind.time.now@v1", "Authoritative cluster time"),
        _make_entry("hivemind.storage.volumes@v1", "List storage volumes"),
    ])
    idx = build_index(catalog, backend=BACKEND_TFIDF)
    hits = query_toolboxes(idx, "list my virtual machines", top_k=3)
    assert hits
    assert hits[0].name == "vm"
    assert hits[0].score > 0


def test_tfidf_query_tools_two_stage():
    catalog = _make_catalog([
        _make_entry("hivemind.vm.list@v1", "List virtual machines"),
        _make_entry("hivemind.vm.start@v1", "Start a virtual machine"),
        _make_entry("hivemind.gpu.availability@v1", "Free GPUs"),
    ])
    idx = build_index(catalog, backend=BACKEND_TFIDF)

    # Without filter — ranks by similarity overall
    hits_all = query_tools(idx, "list my virtual machines", top_k=3)
    assert hits_all[0].name == "hivemind.vm.list@v1"

    # With toolbox filter — only vm tools surface
    hits_vm = query_tools(idx, "start virtual machine", top_k=5, toolbox_filter={"vm"})
    assert {h.name for h in hits_vm} <= {"hivemind.vm.list@v1", "hivemind.vm.start@v1"}
    assert hits_vm[0].name == "hivemind.vm.start@v1"


def test_tfidf_empty_query_returns_empty():
    catalog = _make_catalog([_make_entry("hivemind.vm.list@v1", "List VMs")])
    idx = build_index(catalog, backend=BACKEND_TFIDF)
    assert query_tools(idx, "") == []
    assert query_toolboxes(idx, "") == []


def test_tfidf_unrecognized_query_returns_empty():
    """No vocab overlap → empty result, not error. The cascade
    treats this as 'unsure' and escalates."""
    catalog = _make_catalog([_make_entry("hivemind.vm.list@v1", "List virtual machines")])
    idx = build_index(catalog, backend=BACKEND_TFIDF)
    # Token has no overlap with the catalog vocab.
    assert query_tools(idx, "zzzzzzzzz") == []


def test_tfidf_index_is_deterministic():
    """Same catalog -> identical index. Required for the eval
    harness to be reproducible."""
    catalog = _make_catalog([
        _make_entry("hivemind.vm.list@v1", "List virtual machines"),
        _make_entry("hivemind.gpu.availability@v1", "Free GPUs"),
    ])
    idx1 = build_index(catalog, backend=BACKEND_TFIDF)
    idx2 = build_index(catalog, backend=BACKEND_TFIDF)
    assert idx1.vocab == idx2.vocab
    assert idx1.tool_vectors == idx2.tool_vectors
    assert idx1.toolbox_vectors == idx2.toolbox_vectors


# ---------------------------------------------------------------------------
# Cache behavior
# ---------------------------------------------------------------------------


def test_get_index_in_memory_cache():
    catalog = _make_catalog([_make_entry("hivemind.vm.list@v1", "List VMs")])
    idx1 = get_index(catalog, backend=BACKEND_TFIDF)
    idx2 = get_index(catalog, backend=BACKEND_TFIDF)
    assert idx1 is idx2  # same object → in-memory cache hit


def test_get_index_disk_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("MS4_QM_INDEX_DIR", str(tmp_path))
    catalog = _make_catalog([_make_entry("hivemind.vm.list@v1", "List VMs")])
    idx1 = get_index(catalog, backend=BACKEND_TFIDF)
    # File should now exist on disk:
    files = list(tmp_path.glob("*.tfidf.json"))
    assert files
    # Clear in-memory cache → next lookup should re-load from disk
    reset_index_cache_for_tests()
    idx2 = get_index(catalog, backend=BACKEND_TFIDF)
    assert idx1 is not idx2  # different object (loaded from disk)
    assert idx1.tool_vectors == idx2.tool_vectors  # same content


def test_disk_cache_invalidates_on_version_change(tmp_path, monkeypatch):
    monkeypatch.setenv("MS4_QM_INDEX_DIR", str(tmp_path))
    cat_a = _make_catalog([_make_entry("hivemind.vm.list@v1", "Old desc")])
    get_index(cat_a, backend=BACKEND_TFIDF)
    cat_b = _make_catalog([_make_entry("hivemind.vm.list@v1", "New desc")])
    assert cat_a.version != cat_b.version  # sanity: version changed
    reset_index_cache_for_tests()
    idx_b = get_index(cat_b, backend=BACKEND_TFIDF)
    assert idx_b.version == cat_b.version


# ---------------------------------------------------------------------------
# HiveMind backend (opt-in, with fake embeddings server)
# ---------------------------------------------------------------------------


class _FakeEmbeddingsServer:
    """Fakes hivemind.embeddings.create@v1. Returns deterministic
    short vectors derived from query length so we can assert shape
    + integration without depending on a real embedding model."""

    def __init__(self):
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.port: int = 0
        self.calls: int = 0
        self.fail_with: int | None = None

    def start(self) -> None:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args, **_kwargs):
                return

            def do_POST(self):
                outer.calls += 1
                length = int(self.headers.get("Content-Length", "0") or "0")
                raw = self.rfile.read(length).decode("utf-8") if length else "{}"
                body = json.loads(raw)
                if outer.fail_with is not None:
                    self.send_response(outer.fail_with)
                    self.end_headers()
                    return
                inputs = body.get("params", {}).get("arguments", {}).get("input", [])
                if not isinstance(inputs, list):
                    inputs = [inputs]
                vectors = []
                for s in inputs:
                    # Deterministic 4-dim vector keyed off token counts.
                    text = str(s).lower()
                    vectors.append({
                        "embedding": [
                            float(text.count("vm") + text.count("virtual")),
                            float(text.count("gpu") + text.count("graphics")),
                            float(text.count("time") + text.count("date") + text.count("clock")),
                            float(text.count("storage") + text.count("volume")),
                        ]
                    })
                # Wrap in MCP tools/call response shape.
                payload = {
                    "jsonrpc": "2.0",
                    "id": body.get("id"),
                    "result": {
                        "content": [{
                            "type": "text",
                            "text": json.dumps({"data": vectors}),
                        }],
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

    def stop(self) -> None:
        if self.server:
            try:
                self.server.shutdown()
            except Exception:
                pass


@pytest.fixture
def fake_embeddings(monkeypatch):
    server = _FakeEmbeddingsServer()
    server.start()
    monkeypatch.setenv("MS4_HIVEMIND_MCP_URL", f"http://127.0.0.1:{server.port}/mcp")
    monkeypatch.delenv("MS4_HIVEMIND_API_KEY", raising=False)
    yield server
    server.stop()


def test_hivemind_backend_builds_and_queries(fake_embeddings, monkeypatch):
    catalog = _make_catalog([
        _make_entry("hivemind.vm.list@v1", "List virtual machines"),
        _make_entry("hivemind.gpu.availability@v1", "Report free GPUs"),
        _make_entry("hivemind.time.now@v1", "Cluster time"),
    ])
    idx = build_index(
        catalog,
        hivemind_url=f"http://127.0.0.1:{fake_embeddings.port}",
        backend=BACKEND_HIVEMIND,
    )
    assert idx.backend == BACKEND_HIVEMIND
    # Two calls: tools + toolboxes
    assert fake_embeddings.calls == 2
    # All tool vectors have the fake's 4-dim shape
    assert all(len(v) == 4 for v in idx.tool_vectors.values())
    # Query
    hits = query_toolboxes(
        idx, "list my virtual machines", top_k=3,
        hivemind_url=f"http://127.0.0.1:{fake_embeddings.port}",
    )
    # The "vm" toolbox should rank top since the fake bumps the first
    # dim on the word "vm"/"virtual".
    assert hits and hits[0].name == "vm"


def test_hivemind_backend_falls_back_to_tfidf_on_failure(fake_embeddings):
    fake_embeddings.fail_with = 500
    catalog = _make_catalog([_make_entry("hivemind.vm.list@v1", "List VMs")])
    idx = build_index(
        catalog,
        hivemind_url=f"http://127.0.0.1:{fake_embeddings.port}",
        backend=BACKEND_HIVEMIND,
    )
    # Falls back to TF-IDF rather than raising
    assert idx.backend == BACKEND_TFIDF


def test_hivemind_backend_requires_url():
    catalog = _make_catalog([_make_entry("hivemind.vm.list@v1", "x")])
    idx = build_index(catalog, hivemind_url=None, backend=BACKEND_HIVEMIND)
    # No URL → fall back to TF-IDF (no warnings beyond the log line)
    assert idx.backend == BACKEND_TFIDF


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------


def test_tool_index_to_dict_and_from_dict():
    catalog = _make_catalog([
        _make_entry("hivemind.vm.list@v1", "List virtual machines"),
        _make_entry("hivemind.gpu.availability@v1", "Free GPUs"),
    ])
    idx = build_index(catalog, backend=BACKEND_TFIDF)
    serialized = idx.to_dict()
    rebuilt = ToolIndex.from_dict(serialized)
    assert rebuilt.version == idx.version
    assert rebuilt.backend == idx.backend
    assert rebuilt.tool_vectors == idx.tool_vectors
    assert rebuilt.toolbox_vectors == idx.toolbox_vectors
