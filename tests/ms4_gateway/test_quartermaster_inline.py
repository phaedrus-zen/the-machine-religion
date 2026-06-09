"""Phase D — Face Lobe inline fast-path wiring in Ms4HermesRunner.chat().

Two layers of test:

1. ``chat()`` integration via the patchable ``_try_quartermaster_inline``
   seam: inline-hit skips Depth dispatch and injects the block;
   depth-fallback (None) dispatches Depth exactly as before.
2. ``_try_quartermaster_inline`` end-to-end against a fake MCP server
   (catalog tools/list + tool exec) with an injected ethics evaluator,
   proving a real read-only query runs inline and surfaces a block.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from machine_spirit_4.gateway.face_lobe_chat import FaceLobeChat
from machine_spirit_4.gateway.hermes_runner import Ms4HermesRunner


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeSession:
    def __init__(self, session_id, model):
        self.session_id = session_id
        self.model = model
        self.messages: list[dict[str, Any]] = []


class FakeFaceLobeChat:
    """Records the extra_system it was handed so tests can assert the
    inline block reached the model."""

    def __init__(self, *, hivemind_url: str = "http://hive:6089"):
        self.hivemind_url = hivemind_url
        self.calls: list[dict[str, Any]] = []
        self._sessions: dict[str, _FakeSession] = {}

    def chat(self, message, *, session_id, model, stream_callback=None, extra_system=None):
        self.calls.append({"message": message, "extra_system": extra_system or ""})
        sid = session_id or "s"
        self._sessions.setdefault(sid, _FakeSession(sid, model))
        return {
            "text": f"reply:{message}",
            "session_id": sid,
            "model": model,
            "runtime": "face-lobe-direct",
            "completed": True,
            "api_calls": 1,
            "metrics": {"schema": "Ms4TurnMetrics.v1"},
        }

    def sessions(self):
        return []


class FakeHermesAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = kwargs.get("model")


def _runner(tmp_path, monkeypatch, hivemind_url="http://hive:6089"):
    monkeypatch.setenv("MS4_QM_INLINE", "1")  # opt back in for these tests
    return Ms4HermesRunner(
        hermes_dir=str(tmp_path),
        hivemind_url=hivemind_url,
        agent_cls=FakeHermesAgent,
        face_lobe_chat=FakeFaceLobeChat(hivemind_url=hivemind_url),
    )


# ---------------------------------------------------------------------------
# chat() integration via the patchable seam
# ---------------------------------------------------------------------------


def test_inline_hit_skips_depth_and_injects_block(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch)
    block = (
        "MS4 Quartermaster inline tool result (authoritative):\n"
        "- tool: hivemind.gpu.availability@v1\n- result: {\"free\": 2}"
    )
    monkeypatch.setattr(
        runner, "_try_quartermaster_inline",
        lambda message: (block, {"verdict": "inline", "inline_executed": True}),
    )

    # Force the auto-router to "deep" so the inline path is consulted.
    monkeypatch.setattr(
        "machine_spirit_4.gateway.hermes_runner.router_route",
        lambda message: type("R", (), {
            "kind": "deep", "cleaned_message": message, "goal": "g",
            "to_dict": lambda self: {"kind": "deep"},
        })(),
    )

    resp = runner.chat("what gpus are available", session_id="s1", model="m")
    # Inline fired: no Depth job dispatched, block reached the model.
    assert resp["quartermaster_inline"] is True
    assert resp["dispatched_job"] is None
    assert "hivemind.gpu.availability@v1" in runner.face_lobe_chat.calls[0]["extra_system"]


def test_depth_fallback_when_inline_returns_none(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch)
    monkeypatch.setattr(
        runner, "_try_quartermaster_inline",
        lambda message: (None, {"verdict": "depth", "reason": "needs args"}),
    )
    monkeypatch.setattr(
        "machine_spirit_4.gateway.hermes_runner.router_route",
        lambda message: type("R", (), {
            "kind": "deep", "cleaned_message": message, "goal": "g",
            "to_dict": lambda self: {"kind": "deep"},
        })(),
    )

    resp = runner.chat("refactor the whole module and run the tests", session_id="s2", model="m")
    # Depth path taken: a job was dispatched (or attempted), inline false.
    assert resp["quartermaster_inline"] is False
    assert resp["dispatched_job"] is not None
    assert resp["quartermaster"] == {"verdict": "depth", "reason": "needs args"}


def test_inline_helper_exception_is_fail_soft(tmp_path, monkeypatch):
    """If the inline helper itself raises, chat() must still complete
    via the Depth path (the helper swallows internally, but assert the
    turn survives a None return)."""
    runner = _runner(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "_try_quartermaster_inline", lambda message: (None, {"error": "boom"}))
    monkeypatch.setattr(
        "machine_spirit_4.gateway.hermes_runner.router_route",
        lambda message: type("R", (), {
            "kind": "deep", "cleaned_message": message, "goal": "g",
            "to_dict": lambda self: {"kind": "deep"},
        })(),
    )
    resp = runner.chat("do something tool-ish", session_id="s3", model="m")
    assert resp["completed"] is True
    assert resp["quartermaster_inline"] is False


def test_direct_route_never_consults_quartermaster(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch)
    calls = {"n": 0}

    def spy(message):
        calls["n"] += 1
        return None, None

    monkeypatch.setattr(runner, "_try_quartermaster_inline", spy)
    monkeypatch.setattr(
        "machine_spirit_4.gateway.hermes_runner.router_route",
        lambda message: type("R", (), {
            "kind": "direct", "cleaned_message": message, "goal": "",
            "to_dict": lambda self: {"kind": "direct"},
        })(),
    )
    runner.chat("hello there", session_id="s4", model="m")
    assert calls["n"] == 0  # direct turns skip the Quartermaster entirely


# ---------------------------------------------------------------------------
# _try_quartermaster_inline end-to-end against a fake MCP server
# ---------------------------------------------------------------------------


class _FakeCluster:
    """Answers tools/list (for the catalog) and tools/call (for inline
    execution). Lets us drive the whole inline path with no real cluster."""

    def __init__(self):
        self.tools: list[dict[str, Any]] = []
        self.call_results: dict[str, Any] = {}
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
                method = body.get("method")
                if method == "tools/list":
                    result = {"tools": outer.tools}
                else:  # tools/call
                    name = (body.get("params") or {}).get("name")
                    payload = outer.call_results.get(name, {"ok": True})
                    result = {
                        "content": [{"type": "text", "text": json.dumps(payload)}],
                        "isError": False,
                    }
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
    c.start()
    monkeypatch.setenv("MS4_HIVEMIND_MCP_URL", f"http://127.0.0.1:{c.port}/mcp")
    monkeypatch.setenv("MS4_QM_INLINE", "1")
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


def test_inline_end_to_end_readonly_query(tmp_path, monkeypatch, fake_cluster):
    fake_cluster.tools = [
        {"name": "hivemind.gpu.availability@v1", "description": "Report free GPUs available for scheduling", "inputSchema": {}},
        {"name": "hivemind.vm.start@v1", "description": "Start a virtual machine", "inputSchema": {"required": ["name"]}},
    ]
    fake_cluster.call_results["hivemind.gpu.availability@v1"] = {"free_gpus": 2, "nodes": ["a", "b"]}

    url = f"http://127.0.0.1:{fake_cluster.port}"
    runner = Ms4HermesRunner(
        hermes_dir=str(tmp_path), hivemind_url=url,
        agent_cls=FakeHermesAgent, face_lobe_chat=FakeFaceLobeChat(hivemind_url=url),
    )
    # Inject an allow-everything ethics evaluator into the default router.
    from machine_spirit_4.gateway.quartermaster import ToolRouter
    from machine_spirit_4.gateway.quartermaster import router as qm_router
    monkeypatch.setattr(qm_router, "_DEFAULT_ROUTER", ToolRouter(ethics_evaluator=lambda _i: {"allowed": True}))

    block, outcome = runner._try_quartermaster_inline("what gpus are available")
    assert block is not None
    assert "hivemind.gpu.availability@v1" in block
    assert "free_gpus" in block
    assert outcome["verdict"] == "inline"
    assert outcome["inline_executed"] is True


def test_inline_falls_back_when_top_tool_needs_args(tmp_path, monkeypatch, fake_cluster):
    fake_cluster.tools = [
        {"name": "hivemind.vm.start@v1", "description": "Start a virtual machine by name", "inputSchema": {"required": ["name"]}},
    ]
    url = f"http://127.0.0.1:{fake_cluster.port}"
    runner = Ms4HermesRunner(
        hermes_dir=str(tmp_path), hivemind_url=url,
        agent_cls=FakeHermesAgent, face_lobe_chat=FakeFaceLobeChat(hivemind_url=url),
    )
    from machine_spirit_4.gateway.quartermaster import ToolRouter
    from machine_spirit_4.gateway.quartermaster import router as qm_router
    monkeypatch.setattr(qm_router, "_DEFAULT_ROUTER", ToolRouter(ethics_evaluator=lambda _i: {"allowed": True}))

    block, outcome = runner._try_quartermaster_inline("start the virtual machine")
    assert block is None  # needs args -> depth
    assert outcome["verdict"] == "depth"


def test_inline_disabled_by_env(tmp_path, monkeypatch, fake_cluster):
    monkeypatch.setenv("MS4_QM_INLINE", "0")
    url = f"http://127.0.0.1:{fake_cluster.port}"
    runner = Ms4HermesRunner(
        hermes_dir=str(tmp_path), hivemind_url=url,
        agent_cls=FakeHermesAgent, face_lobe_chat=FakeFaceLobeChat(hivemind_url=url),
    )
    block, outcome = runner._try_quartermaster_inline("what gpus are available")
    assert block is None
    assert outcome is None
