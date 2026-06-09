"""Phase D2 — Quartermaster-curated tools answer replaces the full
75-tool dump for capability questions.

Evidence gate: the curated summary is materially smaller than the
legacy full dump, while still naming the relevant tools.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from machine_spirit_4.gateway import context as context_mod
from machine_spirit_4.gateway.context import (
    format_tools_answer,
    format_tools_answer_quartermaster,
)


class _FakeCluster:
    def __init__(self):
        self.tools: list[dict[str, Any]] = []
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.port = 0

    def start(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a, **_k):
                return

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0") or "0")
                raw = self.rfile.read(length).decode("utf-8") if length else "{}"
                body = json.loads(raw)
                if body.get("method") == "tools/list":
                    result = {"tools": outer.tools}
                else:
                    result = {"content": [{"type": "text", "text": "{}"}], "isError": False}
                env = {"jsonrpc": "2.0", "id": body.get("id"), "result": result}
                wire = json.dumps(env).encode()
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
def fake_cluster(monkeypatch):
    c = _FakeCluster()
    c.tools = [
        {"name": "hivemind.vm.list@v1", "description": "List the virtual machines in the cluster"},
        {"name": "hivemind.vm.start@v1", "description": "Start a virtual machine"},
        {"name": "hivemind.gpu.availability@v1", "description": "Report free GPUs"},
        {"name": "hivemind.storage.volumes@v1", "description": "List storage volumes"},
        {"name": "hivemind.audio.transcribe@v1", "description": "Transcribe speech to text"},
        {"name": "hivemind.time.now@v1", "description": "Cluster time"},
    ]
    c.start()
    monkeypatch.setenv("MS4_HIVEMIND_MCP_URL", f"http://127.0.0.1:{c.port}/mcp")
    monkeypatch.delenv("MS4_HIVEMIND_API_KEY", raising=False)
    from machine_spirit_4.gateway.quartermaster import (
        reset_catalog_cache_for_tests,
        reset_index_cache_for_tests,
    )
    reset_catalog_cache_for_tests()
    reset_index_cache_for_tests()
    yield c
    c.stop()
    reset_catalog_cache_for_tests()
    reset_index_cache_for_tests()


def _url(c):
    return f"http://127.0.0.1:{c.port}"


def test_specific_question_lists_relevant_toolbox(fake_cluster):
    out = format_tools_answer_quartermaster(
        "what VM tools do you have", _url(fake_cluster), "http://127.0.0.1:9180"
    )
    assert "toolbox `vm`" in out
    assert "hivemind.vm.list@v1" in out
    # Compact cluster map present
    assert "Full tool shed" in out
    assert "compute_infra:" in out


def test_broad_question_shows_cluster_map(fake_cluster):
    out = format_tools_answer_quartermaster(
        "what can you do", _url(fake_cluster), "http://127.0.0.1:9180"
    )
    assert "Full tool shed" in out
    # The toolbox the catalog has are named in the map
    assert "vm" in out
    assert "audio" in out or "voice" in out


def test_curated_summary_is_smaller_than_full_dump(fake_cluster, monkeypatch):
    """The whole point of D2: curated summary << full dump."""
    # The full dump reads the real MS4 manifest (75 tools). Point MS3
    # url somewhere harmless; format_tools_answer only needs the manifest
    # + a hivemind count (which our fake serves).
    monkeypatch.setenv("MS4_MS3_URL", "http://127.0.0.1:9080")
    full = format_tools_answer("http://127.0.0.1:9080", _url(fake_cluster), "http://127.0.0.1:9180")
    curated = format_tools_answer_quartermaster(
        "what VM tools do you have", _url(fake_cluster), "http://127.0.0.1:9180"
    )
    # Curated must be materially smaller (fewer chars, far fewer tool lines).
    assert len(curated) < len(full)
    full_tool_lines = full.count("    * ")
    curated_tool_lines = curated.count("    * ")
    assert curated_tool_lines < full_tool_lines


def test_build_grounded_user_message_uses_qm_summary_when_enabled(fake_cluster, monkeypatch):
    """When MS4_QM_TOOLS_SUMMARY=1, the tools-question grounding branch
    routes through the Quartermaster summary (source label + content)."""
    monkeypatch.setenv("MS4_QM_TOOLS_SUMMARY", "1")
    # Avoid the grounding cache returning a stale legacy entry.
    with context_mod._GROUNDING_CACHE_LOCK:
        context_mod._GROUNDING_CACHE.clear()
    grounded, source = context_mod.build_grounded_user_message(
        "what VM tools do you have", _url(fake_cluster)
    )
    assert source is not None and source.startswith("ms4-tools-grounding-qm")
    assert "Quartermaster-curated" in grounded
    assert "toolbox `vm`" in grounded


def test_build_grounded_user_message_legacy_when_disabled(fake_cluster, monkeypatch):
    monkeypatch.setenv("MS4_QM_TOOLS_SUMMARY", "0")
    monkeypatch.setenv("MS4_MS3_URL", "http://127.0.0.1:9080")
    grounded, source = context_mod.build_grounded_user_message(
        "what tools do you have", _url(fake_cluster)
    )
    assert source is not None and source.startswith("ms4-tools-grounding")
    assert "ms4-tools-grounding-qm" not in source
    assert "MS4 capability surface" in grounded


def test_falls_back_when_catalog_empty(monkeypatch):
    """No cluster + no manifest -> still returns a non-empty answer
    (the legacy dump)."""
    monkeypatch.setenv("MS4_HIVEMIND_MCP_URL", "http://127.0.0.1:0/mcp")  # unreachable
    monkeypatch.setenv("MS4_QM_MANIFEST_PATH", "/no/such/manifest.json")
    monkeypatch.setenv("MS4_MS3_URL", "http://127.0.0.1:9080")
    from machine_spirit_4.gateway.quartermaster import reset_catalog_cache_for_tests
    reset_catalog_cache_for_tests()
    out = format_tools_answer_quartermaster(
        "what tools", "http://127.0.0.1:0", "http://127.0.0.1:9180"
    )
    # Falls back to format_tools_answer which always returns a body.
    assert "MS4 capability surface" in out
    reset_catalog_cache_for_tests()
