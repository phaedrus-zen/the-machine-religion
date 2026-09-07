from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SUPERVISE_PATH = REPO / "machine_spirit_4" / "scripts" / "supervise_ms4.py"
CHILD_CODE = 23


def _poison(*_args, **_kwargs):
    raise AssertionError("reachable real seam invoked")


def _load_supervise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "tmr"
    ms4 = root / "machine_spirit_4"
    python = ms4 / ".venv" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"")

    runtime = types.ModuleType("runtime_common")
    runtime.MS4 = ms4
    runtime.ROOT = root
    runtime.is_port_listening = _poison
    runtime.venv_python = lambda: python
    monkeypatch.setitem(sys.modules, "runtime_common", runtime)

    spec = importlib.util.spec_from_file_location("supervise_ms4_under_test", SUPERVISE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    assert module.__name__ != "__main__"

    monkeypatch.setattr(module.subprocess, "run", _poison)
    monkeypatch.setattr(module.subprocess, "Popen", _poison)
    monkeypatch.setattr(module.subprocess, "check_output", _poison)
    monkeypatch.setattr(module.urllib.request, "urlopen", _poison)
    monkeypatch.setattr(module, "open_oracle_ui_once", _poison)
    monkeypatch.setattr(module, "is_port_listening", _poison)
    monkeypatch.setattr(module, "hivemind_health_status", _poison)
    monkeypatch.setattr(module, "down_ports", _poison)
    monkeypatch.setattr(module.time, "sleep", _poison)
    monkeypatch.setattr(module, "acquire_watch_mutex", _poison)
    return module, root, ms4, python


def test_supervise_once_returns_logged_nonzero_child_code(tmp_path, monkeypatch):
    module, root, ms4, python = _load_supervise(tmp_path, monkeypatch)
    logs: list[str] = []
    child_calls: list[dict[str, object]] = []
    child_log = tmp_path / "start_ms4.child.log"

    def fake_run(command, **kwargs):
        child_calls.append({"command": command, **kwargs})
        return types.SimpleNamespace(returncode=CHILD_CODE)

    monkeypatch.setattr(module, "down_ports", lambda: [9180])
    monkeypatch.setattr(module, "hivemind_health_status", lambda: "unreachable")
    monkeypatch.setattr(module, "write_log", logs.append)
    monkeypatch.setattr(module, "log_path", lambda: child_log)
    monkeypatch.setattr(module.subprocess, "run", fake_run)

    assert module.supervise_once() == CHILD_CODE
    assert f"start_ms4 exit={CHILD_CODE}" in logs
    assert len(child_calls) == 1
    assert child_calls[0]["command"] == [str(python), str(ms4 / "scripts" / "start_ms4.py"), "--skip-validation"]
    assert child_calls[0]["cwd"] == str(root)


ALL_LISTENING = "ok: 9080/9180/9181 all listening"


def _already_listening_child(module, tmp_path, monkeypatch, ms4, python, root, child_code: int):
    logs: list[str] = []
    ui: list[str] = []
    events: list[str] = []
    child_calls: list[dict[str, object]] = []
    child_log = tmp_path / "start_ms4.child.log"

    def fake_run(command, **kwargs):
        child_calls.append({"command": command, **kwargs})
        events.append("child")
        return types.SimpleNamespace(returncode=child_code)

    def fake_log(message: str) -> None:
        logs.append(message)
        events.append(message)

    def fake_ui() -> None:
        ui.append("open_oracle_ui_once")
        events.append("open_oracle_ui_once")

    monkeypatch.setattr(module, "down_ports", lambda: [])
    monkeypatch.setattr(module, "hivemind_health_status", lambda: "unreachable")
    monkeypatch.setattr(module, "write_log", fake_log)
    monkeypatch.setattr(module, "log_path", lambda: child_log)
    monkeypatch.setattr(module, "open_oracle_ui_once", fake_ui)
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    return logs, ui, child_calls, events


def test_already_listening_nonzero_child_skips_success_and_ui(tmp_path, monkeypatch):
    module, root, ms4, python = _load_supervise(tmp_path, monkeypatch)
    logs, ui, child_calls, _events = _already_listening_child(
        module, tmp_path, monkeypatch, ms4, python, root, CHILD_CODE
    )

    assert module.supervise_once() == CHILD_CODE
    assert f"start_ms4 exit={CHILD_CODE}" in logs
    assert len(child_calls) == 1
    assert child_calls[0]["command"] == [str(python), str(ms4 / "scripts" / "start_ms4.py"), "--skip-validation"]
    assert child_calls[0]["cwd"] == str(root)
    assert ALL_LISTENING not in logs
    assert ui == []


def test_already_listening_zero_child_logs_success_then_ui(tmp_path, monkeypatch):
    module, root, ms4, python = _load_supervise(tmp_path, monkeypatch)
    logs, ui, child_calls, events = _already_listening_child(
        module, tmp_path, monkeypatch, ms4, python, root, 0
    )

    assert module.supervise_once() == 0
    assert len(child_calls) == 1
    assert child_calls[0]["command"] == [str(python), str(ms4 / "scripts" / "start_ms4.py"), "--skip-validation"]
    assert child_calls[0]["cwd"] == str(root)
    assert events.count("child") == 1
    assert events.count(ALL_LISTENING) == 1
    assert events.count("open_oracle_ui_once") == 1
    assert events.index("child") < events.index(ALL_LISTENING) < events.index("open_oracle_ui_once")
    assert ui == ["open_oracle_ui_once"]
    assert ALL_LISTENING in logs


def _run_watch(module, monkeypatch, once_result: int) -> list[str]:
    logs: list[str] = []
    sleeps: list[float] = []

    def fake_sleep(seconds):
        sleeps.append(seconds)
        raise KeyboardInterrupt

    monkeypatch.setattr(module, "acquire_watch_mutex", lambda: True)
    monkeypatch.setattr(module, "supervise_once", lambda: once_result)
    monkeypatch.setattr(module, "write_log", logs.append)
    monkeypatch.setattr(module.time, "sleep", fake_sleep)
    monkeypatch.setattr(sys, "argv", ["supervise_ms4.py", "--watch", "--interval-seconds", "30"])

    assert module.main() == 0
    assert sleeps == [30]
    assert "user watchdog stopped" in logs
    return logs


def test_watch_logs_nonzero_managed_iteration_then_stops(tmp_path, monkeypatch):
    module, *_ = _load_supervise(tmp_path, monkeypatch)
    logs = _run_watch(module, monkeypatch, CHILD_CODE)
    assert f"managed iteration failed code={CHILD_CODE}" in logs


def test_watch_zero_result_emits_no_managed_failure_log(tmp_path, monkeypatch):
    module, *_ = _load_supervise(tmp_path, monkeypatch)
    logs = _run_watch(module, monkeypatch, 0)
    assert not any("managed iteration failed" in message for message in logs)
