from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "machine_spirit_4" / "scripts" / "inspect_process_env_allowlist.py"


def _load():
    spec = importlib.util.spec_from_file_location("inspect_process_env_allowlist", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_filter_environ_keeps_only_allowlisted_voice_keys() -> None:
    module = _load()
    raw = {
        "MS4_VOICE_CANDIDATE_RECONCILE_ENABLED": "1",
        "MS4_VOICE_CANDIDATE_RECONCILE_SECS": "3600",
        "MS4_VOICE_CANDIDATE_RECONCILE_TIMEOUT_SECS": "3",
        "MS4_HIVEMIND_URL": "http://127.0.0.1:6089",
        "MS4_HIVEMIND_HLI_URL": "http://127.0.0.1:6089",
        "MS4_OPEN_UI": "0",
        "MS4_HIVEMIND_API_KEY": "should-never-appear",
        "AUTHORIZATION": "Bearer leaked",
        "PATH": "C:\\Windows\\System32",
    }
    filtered = module.filter_environ(raw)
    assert filtered == {
        "MS4_VOICE_CANDIDATE_RECONCILE_ENABLED": "1",
        "MS4_VOICE_CANDIDATE_RECONCILE_SECS": "3600",
        "MS4_VOICE_CANDIDATE_RECONCILE_TIMEOUT_SECS": "3",
        "MS4_HIVEMIND_URL": "http://127.0.0.1:6089",
        "MS4_HIVEMIND_HLI_URL": "http://127.0.0.1:6089",
        "MS4_OPEN_UI": "0",
    }
    blob = str(filtered)
    assert "should-never-appear" not in blob
    assert "Bearer" not in blob
    assert "Windows" not in blob


def test_inspect_pid_uses_injected_reader_and_marks_missing_keys() -> None:
    module = _load()
    payload = module.inspect_pid(33084, read_environ=lambda _pid: {"MS4_OPEN_UI": "0"})
    assert payload["pid"] == 33084
    assert payload["ok"] is True
    assert payload["env"]["MS4_OPEN_UI"] == "0"
    assert payload["env"]["MS4_VOICE_CANDIDATE_RECONCILE_ENABLED"] is None
    assert "MS4_HIVEMIND_API_KEY" not in payload["env"]


def test_inspect_pid_fail_closed_on_reader_error() -> None:
    module = _load()

    def boom(_pid: int):
        raise OSError("denied")

    payload = module.inspect_pid(1, read_environ=boom)
    assert payload["ok"] is False
    assert payload["env"] == {}
    assert payload["error"] == "reader_unavailable"
