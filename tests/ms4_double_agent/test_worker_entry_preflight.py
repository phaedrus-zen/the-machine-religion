from __future__ import annotations

import json
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from machine_spirit_4.double_agent import JobEnvelope, _worker_entry, safety


class _ReachableServer:
    def __init__(self):
        self.observed_headers: list[dict[str, str]] = []
        self.observed_paths: list[str] = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def stop(self) -> None:
        self.server.shutdown()

    def _handler(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args, **_kwargs):
                return

            def do_GET(self):
                outer.observed_paths.append(self.path)
                outer.observed_headers.append({k: v for k, v in self.headers.items()})
                body = b'{"ok":true}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                outer.observed_paths.append(self.path)
                outer.observed_headers.append({k: v for k, v in self.headers.items()})
                length = int(self.headers.get("Content-Length", "0") or "0")
                if length:
                    self.rfile.read(length)
                body = json.dumps(
                    {"jsonrpc": "2.0", "id": "preflight", "result": {"tools": []}}
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        return Handler


def test_depth_preflight_urls_respect_legacy_hli_alias_and_mcp_pin(monkeypatch):
    monkeypatch.delenv("MS4_HIVEMIND_URL", raising=False)
    monkeypatch.setenv("MS4_HIVEMIND_HLI_URL", "http://hive.local:6089/")
    monkeypatch.setenv("MS4_HIVEMIND_MCP_URL", "http://hive.local:6105")
    monkeypatch.delenv("MS4_MS3_URL", raising=False)

    urls = _worker_entry._build_depth_preflight_urls()

    assert urls["hivemind_url"] == "http://hive.local:6089"
    assert urls["hivemind_mcp_url"] == "http://hive.local:6105/mcp"
    assert urls["ms3_url"] == "http://127.0.0.1:9080"


def test_depth_worker_preflight_probes_runtime_namespace_and_redacts_secret(monkeypatch):
    server = _ReachableServer()
    try:
        monkeypatch.setenv("MS4_HIVEMIND_URL", server.url)
        monkeypatch.setenv("MS4_HIVEMIND_MCP_URL", f"{server.url}/mcp")
        monkeypatch.setenv("MS4_MS3_URL", server.url)
        monkeypatch.setenv("MS4_HIVEMIND_API_KEY", "super-secret-token")

        report = _worker_entry.run_depth_worker_preflight(timeout=1.0)

        assert report["schema"] == "Ms4DepthWorkerPreflight.v1"
        assert report["ok"] is True
        assert report["env_configured"]["MS4_HIVEMIND_API_KEY"] is True
        assert "super-secret-token" not in json.dumps(report)
        assert [probe["name"] for probe in report["probes"]] == [
            "hivemind_hli_health",
            "hivemind_mcp_tools_list",
            "ms3_health",
        ]
        assert all(probe["reachable"] for probe in report["probes"])
        assert "/mcp" in server.observed_paths
        assert any(
            headers.get("Authorization") == "Bearer super-secret-token"
            for headers in server.observed_headers
        )
    finally:
        server.stop()


def test_depth_worker_preflight_reports_namespace_connection_failure(monkeypatch):
    def refuse(_req, timeout=0):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(_worker_entry.urllib.request, "urlopen", refuse)
    monkeypatch.setenv("MS4_HIVEMIND_URL", "http://127.0.0.1:6089")
    monkeypatch.setenv("MS4_HIVEMIND_MCP_URL", "http://127.0.0.1:6105/mcp")
    monkeypatch.setenv("MS4_MS3_URL", "http://127.0.0.1:9080")

    report = _worker_entry.run_depth_worker_preflight(timeout=0.01)

    assert report["ok"] is False
    assert {probe["reachable"] for probe in report["probes"]} == {False}
    assert all(probe["error_type"] == "URLError" for probe in report["probes"])


def test_depth_worker_preflight_warns_for_posix_loopback_failure(monkeypatch):
    def refuse(_req, timeout=0):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(_worker_entry.urllib.request, "urlopen", refuse)
    monkeypatch.setenv("MS4_HIVEMIND_URL", "http://127.0.0.1:6089")
    monkeypatch.setenv("MS4_HIVEMIND_MCP_URL", "http://127.0.0.1:6105/mcp")
    monkeypatch.setenv("MS4_MS3_URL", "http://127.0.0.1:9080")

    report = _worker_entry.run_depth_worker_preflight(timeout=0.01)
    diagnostics = _worker_entry._depth_preflight_diagnostics(
        urls=report["urls"],
        probes=report["probes"],
        os_name="posix",
    )

    assert report["ok"] is False
    assert report["diagnostics"] == []
    assert diagnostics == [
        {
            "code": "posix_loopback_namespace_unreachable",
            "severity": "warning",
            "summary": (
                "This POSIX Depth worker is using loopback URLs that are "
                "unreachable from its namespace. 127.0.0.1 points at the "
                "worker namespace, not necessarily the Windows host."
            ),
            "loopback_urls": [
                "hivemind_mcp_url",
                "hivemind_url",
                "ms3_url",
            ],
            "unreachable_probes": [
                "hivemind_hli_health",
                "hivemind_mcp_tools_list",
                "ms3_health",
            ],
            "operator_action": (
                "Inject host-routable values for MS4_HIVEMIND_URL, "
                "MS4_HIVEMIND_MCP_URL, and MS4_MS3_URL, then rerun "
                "_worker_entry.py --preflight --json."
            ),
        }
    ]


def test_preflight_cli_does_not_require_worker_blackboard_args(monkeypatch, capsys):
    monkeypatch.setattr(
        _worker_entry,
        "run_depth_worker_preflight",
        lambda *, timeout: {
            "schema": "Ms4DepthWorkerPreflight.v1",
            "ok": True,
            "probes": [],
        },
    )
    monkeypatch.setattr(
        _worker_entry.sys,
        "argv",
        ["_worker_entry.py", "--preflight", "--json", "--timeout", "0.1"],
    )

    assert _worker_entry.main() == 0
    assert json.loads(capsys.readouterr().out)["schema"] == "Ms4DepthWorkerPreflight.v1"


def test_worker_runner_construction_uses_shared_depth_urls(monkeypatch):
    seen: dict[str, object] = {}

    class FakeHermesRunner:
        def __init__(self, **kwargs):
            seen["runner_kwargs"] = kwargs

    def fake_build_real_chat_runner(runner):
        seen["runner"] = runner
        return "chat-runner"

    import machine_spirit_4.gateway.hermes_runner as hermes_runner
    from machine_spirit_4.scripts import runtime_common

    monkeypatch.setattr(hermes_runner, "Ms4HermesRunner", FakeHermesRunner)
    monkeypatch.setattr(
        runtime_common,
        "hermes_dir",
        lambda: "C:/Managed-Hermes",
    )
    monkeypatch.setattr(_worker_entry, "build_real_chat_runner", fake_build_real_chat_runner)
    monkeypatch.delenv("MS4_DOUBLE_AGENT_FAKE_CHAT_RUNNER", raising=False)
    monkeypatch.setenv("MS4_HERMES_DIR", "C:/Hermes")
    monkeypatch.setenv("MS4_HIVEMIND_URL", "http://hive-primary:6089")
    monkeypatch.setenv("MS4_HIVEMIND_HLI_URL", "http://hive-legacy:6089")
    monkeypatch.setenv("MS4_MS3_URL", "http://ms3-host:9080")
    monkeypatch.setenv("MS4_DEFAULT_MODEL", "face-depth:latest")
    monkeypatch.setenv("MS4_DEPTH_FALLBACK_MODEL", "gemma4:31b")

    env = JobEnvelope(
        job_id=safety.new_job_id(),
        parent_conversation_id="conv-preflight-build",
        conversation_revision_id=1,
        background_lobe_type="deep_chat",
        user_visible_goal="Construct runner.",
        internal_goal="Construct runner.",
    )
    env.validate()

    assert _worker_entry._build_runner_or_die(env, blackboard=None) == "chat-runner"
    assert seen["runner_kwargs"] == {
        "hermes_dir": "C:/Managed-Hermes",
        "hivemind_url": "http://hive-primary:6089",
        "ms3_url": "http://ms3-host:9080",
        "default_model": "gemma4:31b",
    }


def test_gateway_and_mcp_constructors_use_marker_aware_hermes_dir(monkeypatch):
    from machine_spirit_4.gateway import server as gateway_server
    from machine_spirit_4.mcp import server as mcp_server
    from machine_spirit_4.scripts import runtime_common

    constructed: list[dict[str, object]] = []

    class FakeHermesRunner:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            constructed.append(kwargs)

    monkeypatch.setenv("MS4_HERMES_DIR", "C:/Inherited-External")
    monkeypatch.setattr(
        runtime_common,
        "hermes_dir",
        lambda: "C:/Managed-Hermes",
    )
    monkeypatch.setattr(gateway_server, "Ms4HermesRunner", FakeHermesRunner)
    monkeypatch.setattr(mcp_server, "Ms4HermesRunner", FakeHermesRunner)
    monkeypatch.setattr(mcp_server, "Ms4McpRuntime", lambda runner: runner)

    gateway_runner = gateway_server.build_runner()
    mcp_runner = mcp_server.build_runtime()

    assert gateway_runner.kwargs["hermes_dir"] == "C:/Managed-Hermes"
    assert mcp_runner.kwargs["hermes_dir"] == "C:/Managed-Hermes"
    assert [entry["hermes_dir"] for entry in constructed] == [
        "C:/Managed-Hermes",
        "C:/Managed-Hermes",
    ]
