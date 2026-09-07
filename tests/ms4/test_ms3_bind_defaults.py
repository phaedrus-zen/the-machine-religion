from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
MS3 = ROOT / "machine_spirit_3"
MS4 = ROOT / "machine_spirit_4"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_runtime_common():
    return _load_module("ms4_runtime_common_bind_test", MS4 / "scripts" / "runtime_common.py")


def test_ms4_managed_ms3_defaults_to_loopback(monkeypatch):
    monkeypatch.delenv("MS3_HOST", raising=False)

    env = _load_runtime_common().ms4_env(ms3_port=9080)

    assert env["MS3_HOST"] == "127.0.0.1"
    assert env["MS4_MS3_URL"] == "http://127.0.0.1:9080"
    assert env["MS4_MS3_SIDECAR_URL"] == "http://127.0.0.1:9080"


def test_ms4_managed_ms3_preserves_explicit_lab_lan_override(monkeypatch):
    monkeypatch.setenv("MS3_HOST", "0.0.0.0")

    env = _load_runtime_common().ms4_env(ms3_port=9080)

    assert env["MS3_HOST"] == "0.0.0.0"


def test_ms4_managed_ms3_explicit_argument_overrides_ambient_value(monkeypatch):
    monkeypatch.setenv("MS3_HOST", "0.0.0.0")

    env = _load_runtime_common().ms4_env(ms3_port=9080, ms3_host="127.0.0.1")

    assert env["MS3_HOST"] == "127.0.0.1"


def test_ms3_python_runner_defaults_to_loopback(monkeypatch):
    runner = _load_module("ms3_run_bind_default_test", MS3 / "run.py")
    captured: dict[str, object] = {}

    def fake_call(command, *, cwd, env):
        captured.update(command=command, cwd=cwd, env=env)
        return 0

    monkeypatch.delenv("MS3_HOST", raising=False)
    monkeypatch.setattr(sys, "argv", ["run.py", "--debug"])
    monkeypatch.setattr(runner.subprocess, "call", fake_call)

    assert runner.main() == 0
    assert captured["env"]["MS3_HOST"] == "127.0.0.1"


def test_ms3_python_runner_accepts_explicit_lab_lan_override(monkeypatch):
    runner = _load_module("ms3_run_bind_override_test", MS3 / "run.py")
    captured: dict[str, object] = {}

    def fake_call(command, *, cwd, env):
        captured.update(command=command, cwd=cwd, env=env)
        return 0

    monkeypatch.setattr(sys, "argv", ["run.py", "--debug", "--host", "0.0.0.0"])
    monkeypatch.setattr(runner.subprocess, "call", fake_call)

    assert runner.main() == 0
    assert captured["env"]["MS3_HOST"] == "0.0.0.0"


def test_ms3_and_ms4_service_manifests_pin_loopback():
    ms3_service = json.loads((MS3 / "warden_service.json").read_text(encoding="utf-8"))
    ms4_service = json.loads((MS4 / "warden_service.json").read_text(encoding="utf-8"))

    assert ms3_service["environment"]["MS3_HOST"] == "127.0.0.1"
    assert ms4_service["environment"]["MS3_HOST"] == "127.0.0.1"


@pytest.mark.parametrize(
    ("extra_args", "expected_bind"),
    [([], "127.0.0.1"), (["--ms3-host", "0.0.0.0"], "0.0.0.0")],
)
def test_ms4_launcher_applies_ms3_bind_contract(monkeypatch, extra_args, expected_bind):
    runtime_common = _load_runtime_common()
    monkeypatch.setitem(sys.modules, "runtime_common", runtime_common)
    launcher = _load_module("start_ms4_bind_contract_test", MS4 / "scripts" / "start_ms4.py")
    launches: list[dict[str, object]] = []
    probes: list[tuple[str, int]] = []

    def fake_is_port_listening(host, port):
        probes.append((host, port))
        return port != 9080

    def fake_launch_process(command, **kwargs):
        launches.append({"command": command, **kwargs})
        return object()

    monkeypatch.delenv("MS3_HOST", raising=False)
    monkeypatch.setattr(sys, "argv", ["start_ms4.py", "--skip-validation", *extra_args])
    monkeypatch.setattr(launcher, "require_venv_python", lambda: Path(sys.executable))
    monkeypatch.setattr(launcher, "ensure_ms3_binary", lambda: None)
    monkeypatch.setattr(launcher, "ms3_binary", lambda: Path("machine_spirit_3"))
    monkeypatch.setattr(launcher, "is_port_listening", fake_is_port_listening)
    monkeypatch.setattr(launcher, "wait_http", lambda _url, **_kwargs: True)
    monkeypatch.setattr(launcher, "launch_process", fake_launch_process)

    assert launcher.main() == 0
    assert probes[0] == ("127.0.0.1", 9080)
    assert launches[0]["env"]["MS3_HOST"] == expected_bind


MS3_URL = "http://127.0.0.1:9080/health"
GATEWAY_URL = "http://127.0.0.1:9180/api/v1/ms4_gateway/status"
MCP_URL = "http://127.0.0.1:9181/api/v1/ms4_mcp/status"
MS3_OK = {"status": "alive", "service": "Machine Spirit 3", "version": "extra"}
GATEWAY_OK = {"service": "ms4-gateway", "status": "ready", "endpoint": "/chat", "ok": True}
MCP_OK = {"service": "ms4-mcp-server", "status": "ready", "endpoint": "/mcp", "tools_loaded": 1}


def _poison(*_args, **_kwargs):
    raise AssertionError("reachable real seam invoked")


def _identity_launcher(monkeypatch, bodies: dict[str, dict[str, object]]):
    from contextlib import redirect_stdout
    from io import StringIO

    runtime = _load_module(f"runtime_common_identity_{id(bodies)}", MS4 / "scripts" / "runtime_common.py")
    monkeypatch.setitem(sys.modules, "runtime_common", runtime)
    launcher = _load_module(f"start_ms4_identity_{id(bodies)}", MS4 / "scripts" / "start_ms4.py")

    ports: list[tuple[str, int]] = []
    urls: list[str] = []
    reads: list[bytes] = []
    sleeps: list[float] = []
    clock = [0.0]

    class FakeResponse:
        status = 200

        def __init__(self, payload: dict[str, object]):
            self._body = json.dumps(payload).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def read(self):
            reads.append(self._body)
            return self._body

    def fake_urlopen(url, timeout=None, **_kwargs):
        urls.append(url)
        if url == MS3_URL or url.endswith(":9080/health"):
            payload = bodies["ms3"]
        elif "ms4_gateway" in url or ":9180/" in url:
            payload = bodies["gateway"]
        elif "ms4_mcp" in url or ":9181/" in url:
            payload = bodies["mcp"]
        else:
            raise AssertionError(url)
        return FakeResponse(payload)

    def fake_is_port_listening(host, port, timeout=0.5):
        ports.append((host, port))
        if (host, port) in {("127.0.0.1", 9080), ("127.0.0.1", 9180), ("127.0.0.1", 9181)}:
            return True
        raise AssertionError((host, port))

    def fake_sleep(seconds):
        sleeps.append(seconds)
        clock[0] += 100

    class NoopDir:
        def mkdir(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(sys, "argv", ["start_ms4.py", "--skip-validation"])
    monkeypatch.setattr(launcher, "LOG_DIR", NoopDir())
    monkeypatch.setattr(launcher, "require_venv_python", lambda: MS4 / "scripts" / "start_ms4.py")
    monkeypatch.setattr(launcher, "ms4_env", lambda **_kwargs: {"MS3_HOST": "127.0.0.1"})
    monkeypatch.setattr(launcher, "ensure_ms3_binary", lambda: None)
    monkeypatch.setattr(launcher, "is_port_listening", fake_is_port_listening)
    monkeypatch.setattr(launcher, "ms3_binary", _poison)
    monkeypatch.setattr(launcher, "launch_process", _poison)
    monkeypatch.setattr(launcher.subprocess, "check_call", _poison)
    monkeypatch.setattr(launcher.subprocess, "call", _poison)
    monkeypatch.setattr(runtime.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(runtime.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(runtime.time, "sleep", fake_sleep)

    def run_main():
        buf = StringIO()
        with redirect_stdout(buf):
            result = launcher.main()
        return result, buf.getvalue()

    return run_main, ports, urls, reads, sleeps


@pytest.mark.parametrize(
    ("target", "field", "error", "url_prefix", "port_prefix"),
    [
        ("ms3", "status", "MS3 did not become healthy on port 9080", [MS3_URL], [("127.0.0.1", 9080)]),
        ("ms3", "service", "MS3 did not become healthy on port 9080", [MS3_URL], [("127.0.0.1", 9080)]),
        ("gateway", "service", "MS4 gateway did not become healthy on port 9180", [MS3_URL, GATEWAY_URL], [("127.0.0.1", 9080), ("127.0.0.1", 9180)]),
        ("gateway", "status", "MS4 gateway did not become healthy on port 9180", [MS3_URL, GATEWAY_URL], [("127.0.0.1", 9080), ("127.0.0.1", 9180)]),
        ("gateway", "endpoint", "MS4 gateway did not become healthy on port 9180", [MS3_URL, GATEWAY_URL], [("127.0.0.1", 9080), ("127.0.0.1", 9180)]),
        ("mcp", "service", "MS4 MCP did not become healthy on port 9181", [MS3_URL, GATEWAY_URL, MCP_URL], [("127.0.0.1", 9080), ("127.0.0.1", 9180), ("127.0.0.1", 9181)]),
        ("mcp", "status", "MS4 MCP did not become healthy on port 9181", [MS3_URL, GATEWAY_URL, MCP_URL], [("127.0.0.1", 9080), ("127.0.0.1", 9180), ("127.0.0.1", 9181)]),
        ("mcp", "endpoint", "MS4 MCP did not become healthy on port 9181", [MS3_URL, GATEWAY_URL, MCP_URL], [("127.0.0.1", 9080), ("127.0.0.1", 9180), ("127.0.0.1", 9181)]),
    ],
)
def test_start_ms4_rejects_wrong_service_identity(monkeypatch, target, field, error, url_prefix, port_prefix):
    bodies = {"ms3": dict(MS3_OK), "gateway": dict(GATEWAY_OK), "mcp": dict(MCP_OK)}
    bodies[target] = dict(bodies[target])
    bodies[target][field] = "wrong"
    run_main, ports, urls, reads, sleeps = _identity_launcher(monkeypatch, bodies)

    with pytest.raises(RuntimeError, match=error):
        run_main()

    assert ports == port_prefix
    assert urls == url_prefix
    assert len(reads) == len(url_prefix)
    assert sleeps == [0.5]


def test_start_ms4_accepts_identified_ready_bodies(monkeypatch):
    run_main, ports, urls, reads, sleeps = _identity_launcher(
        monkeypatch,
        {"ms3": dict(MS3_OK), "gateway": dict(GATEWAY_OK), "mcp": dict(MCP_OK)},
    )

    result, stdout = run_main()

    assert result == 0
    assert "MS4 runtime ready" in stdout
    assert ports == [("127.0.0.1", 9080), ("127.0.0.1", 9180), ("127.0.0.1", 9181)]
    assert urls == [MS3_URL, GATEWAY_URL, MCP_URL]
    assert len(reads) == 3
    assert sleeps == []
