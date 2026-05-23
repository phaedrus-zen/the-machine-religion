"""Installer phase + rollback tests.

We mock subprocess so the test never actually touches git or pip;
the goal is to verify the phase transitions, plugin-resync step,
and the safe_target_version refusal land in the persisted snapshot
the same way ``ollama_admin::run_update_job`` records its phases.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from machine_spirit_4.hermes_admin import installer, state, versioning


class FakeProcess:
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def _install_fake_subprocess(monkeypatch, *, fail_on: str | None = None):
    calls: list[list[str]] = []

    def fake_run(cmd, cwd=None, capture_output=True, text=True, timeout=None, check=False):
        calls.append(list(cmd))
        if fail_on and any(fail_on in str(c) for c in cmd):
            return FakeProcess(stderr=f"simulated failure: {fail_on}", returncode=1)
        if cmd[0].endswith(("python.exe", "python")) and "-c" in cmd:
            return FakeProcess(stdout="0.14.0\n")
        return FakeProcess(stdout="ok\n")

    monkeypatch.setattr(installer.subprocess, "run", fake_run)
    return calls


def _install_fake_versioning(monkeypatch, *, mode_value: str, directory: Path | None, tag_name: str = "v0.14.0"):
    versioning._clear_caches_for_test()
    monkeypatch.setattr(
        installer.versioning,
        "install_mode",
        lambda: {"mode": mode_value, "version": "0.13.0", "directory": directory},
    )
    monkeypatch.setattr(
        installer.versioning,
        "latest_version",
        lambda force_refresh=False: versioning.CachedLatest(
            version="0.14.0",
            published_at="2026-05-16T00:00:00Z",
            tag_name=tag_name,
            html_url="https://example/release",
            fetched_at_unix=0.0,
        ),
    )
    monkeypatch.setattr(installer.versioning, "recent_releases", lambda force_refresh=False: [])
    monkeypatch.setattr(installer.versioning, "resolve_git_tag", lambda target: tag_name)


def test_editable_upgrade_runs_all_phases(tmp_path, monkeypatch):
    state._reset_for_tests(tmp_path / "snap.json")
    hermes_dir = tmp_path / "hermes-agent"
    (hermes_dir / ".git").mkdir(parents=True)
    plugin_src = tmp_path / "plugin_src"
    plugin_src.mkdir()
    (plugin_src / "plugin.yaml").write_text("name: ms4_consciousness\n", encoding="utf-8")
    venv_python = tmp_path / "venv" / "python.exe"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("")

    _install_fake_versioning(monkeypatch, mode_value="editable", directory=hermes_dir, tag_name="v2026.5.16")
    calls = _install_fake_subprocess(monkeypatch)

    state.start_job(from_version="0.13.0", to_version="0.14.0", install_mode="editable")
    result = installer.run_update_job(
        target_version="0.14.0",
        plugin_src=plugin_src,
        venv_python=venv_python,
    )

    final = state.last_update()
    assert final is not None
    assert final.status == "success"
    assert result["to_version"] == "0.14.0"
    phases = [entry["phase"] for entry in final.progress]
    for required in ("preflight", "fetching_release", "fetching_remote", "checking_out", "syncing_plugin", "pip_installing", "validating", "done"):
        assert required in phases, f"missing phase {required} in {phases}"
    plugin_dst = hermes_dir / "plugins" / "ms4_consciousness"
    assert (plugin_dst / "plugin.yaml").read_text(encoding="utf-8") == "name: ms4_consciousness\n"
    flat_cmds = [" ".join(c) for c in calls]
    assert any("git" in c and "fetch" in c for c in flat_cmds)
    assert any("checkout" in c and "v2026.5.16" in c for c in flat_cmds), flat_cmds
    assert any("pip" in c and "install" in c and "-e" in c for c in flat_cmds)


def test_pypi_upgrade_invokes_wheel_install(tmp_path, monkeypatch):
    state._reset_for_tests(tmp_path / "snap.json")
    plugin_src = tmp_path / "plugin_src"
    plugin_src.mkdir()
    venv_python = tmp_path / "venv" / "python.exe"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("")

    _install_fake_versioning(monkeypatch, mode_value="pypi", directory=None)
    calls = _install_fake_subprocess(monkeypatch)

    state.start_job(from_version="0.13.0", to_version="0.14.0", install_mode="pypi")
    result = installer.run_update_job(
        target_version="0.14.0",
        plugin_src=plugin_src,
        venv_python=venv_python,
    )

    final = state.last_update()
    assert final is not None
    assert final.status == "success"
    assert result["to_version"] == "0.14.0"
    assert any("hermes-agent==0.14.0" in " ".join(c) for c in calls)
    assert not any("git" in c[0] for c in calls)


def test_refuses_unsafe_target_version(tmp_path, monkeypatch):
    state._reset_for_tests(tmp_path / "snap.json")
    plugin_src = tmp_path / "plugin_src"
    plugin_src.mkdir()
    venv_python = tmp_path / "python"
    venv_python.write_text("")

    _install_fake_versioning(monkeypatch, mode_value="pypi", directory=None)
    calls = _install_fake_subprocess(monkeypatch)

    state.start_job(from_version="0.13.0", to_version=None, install_mode="pypi")
    with pytest.raises(installer.HermesUpgradeError):
        installer.run_update_job(
            target_version="0.14.0; rm -rf /",
            plugin_src=plugin_src,
            venv_python=venv_python,
        )
    final = state.last_update()
    assert final is not None
    assert final.status == "failed"
    assert "unsafe target_version" in (final.error or "")
    assert not any("install" in " ".join(c) for c in calls if c)


def test_pip_failure_finalizes_with_error(tmp_path, monkeypatch):
    state._reset_for_tests(tmp_path / "snap.json")
    plugin_src = tmp_path / "plugin_src"
    plugin_src.mkdir()
    venv_python = tmp_path / "python"
    venv_python.write_text("")

    _install_fake_versioning(monkeypatch, mode_value="pypi", directory=None)
    _install_fake_subprocess(monkeypatch, fail_on="hermes-agent==")

    state.start_job(from_version="0.13.0", to_version=None, install_mode="pypi")
    with pytest.raises(installer.HermesUpgradeError):
        installer.run_update_job(
            target_version="0.14.0",
            plugin_src=plugin_src,
            venv_python=venv_python,
        )
    final = state.last_update()
    assert final is not None
    assert final.status == "failed"
    assert "simulated failure" in (final.error or "")


def test_trigger_update_is_idempotent_when_already_running(tmp_path, monkeypatch):
    state._reset_for_tests(tmp_path / "snap.json")
    plugin_src = tmp_path / "plugin_src"
    plugin_src.mkdir()
    venv_python = tmp_path / "python"
    venv_python.write_text("")
    _install_fake_versioning(monkeypatch, mode_value="pypi", directory=None)

    state.start_job(from_version="0.13.0", to_version="0.14.0", install_mode="pypi")
    snap = installer.trigger_update(target_version="0.14.0", plugin_src=plugin_src, venv_python=venv_python)
    assert snap.status == "running"
    assert state.update_in_progress() is True


def test_missing_install_mode_fails_fast(tmp_path, monkeypatch):
    state._reset_for_tests(tmp_path / "snap.json")
    plugin_src = tmp_path / "plugin_src"
    plugin_src.mkdir()
    venv_python = tmp_path / "python"
    venv_python.write_text("")

    monkeypatch.setattr(installer.versioning, "install_mode", lambda: {"mode": "missing", "version": None, "directory": None})

    state.start_job(from_version=None, to_version="0.14.0", install_mode="missing")
    with pytest.raises(installer.HermesUpgradeError):
        installer.run_update_job(target_version="0.14.0", plugin_src=plugin_src, venv_python=venv_python)
    final = state.last_update()
    assert final is not None
    assert final.status == "failed"
    assert "not installed" in (final.error or "")


def test_no_powershell_or_bash_only_python_subprocess(monkeypatch):
    """Cross-platform contract: installer never invokes shell scripts directly.

    Mirrors the cross-platform rule baked into the TMR repo — pip and
    git are cross-platform CLIs; PowerShell / Bash scripts are not.
    """
    seen: list[list[str]] = []

    def fake_run(cmd, cwd=None, capture_output=True, text=True, timeout=None, check=False):
        seen.append(list(cmd))
        return FakeProcess(stdout="0.14.0\n")

    monkeypatch.setattr(installer.subprocess, "run", fake_run)
    _install_fake_versioning(monkeypatch, mode_value="pypi", directory=None)
    state._reset_for_tests(None)
    state.start_job(from_version="0.13.0", to_version="0.14.0", install_mode="pypi")
    installer.run_update_job(target_version="0.14.0")

    flat = [" ".join(c).lower() for c in seen]
    forbidden = ("powershell", "pwsh.exe", "/bin/bash -c", "cmd.exe", "winget")
    assert not any(any(token in line for token in forbidden) for line in flat), flat
