"""Contained-runtime guard tests.

Verifies that MS4 service entrypoints refuse to boot from anything
other than ``machine_spirit_4/.venv`` — closes the door on the global
``Python311\\python.exe`` zombies that were re-spawning alongside the
contained gateway and racing for the port bind.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from machine_spirit_4 import contained


def test_expected_venv_points_into_machine_spirit_4():
    assert contained.EXPECTED_VENV.name == ".venv"
    assert contained.EXPECTED_VENV.parent.name == "machine_spirit_4"


def test_expected_python_path_is_platform_appropriate():
    expected = contained.expected_python()
    assert expected.parent in {contained.EXPECTED_VENV / "Scripts", contained.EXPECTED_VENV / "bin"}


def test_require_contained_runtime_passes_when_executable_in_venv(monkeypatch):
    fake_python = contained.EXPECTED_VENV / "Scripts" / "python.exe"
    monkeypatch.setattr(contained.sys, "executable", str(fake_python))
    monkeypatch.delenv(contained.ENV_OVERRIDE, raising=False)
    contained.require_contained_runtime("test")


def test_require_contained_runtime_refuses_global_python(monkeypatch, capsys):
    fake_global = Path("C:/Users/op/AppData/Local/Programs/Python/Python311/python.exe")
    monkeypatch.setattr(contained.sys, "executable", str(fake_global))
    monkeypatch.delenv(contained.ENV_OVERRIDE, raising=False)
    with pytest.raises(SystemExit) as excinfo:
        contained.require_contained_runtime("ms4-gateway")
    assert excinfo.value.code == contained.EXIT_CONTAINMENT_VIOLATION
    captured = capsys.readouterr()
    assert "containment violation" in captured.err
    assert "ms4-gateway" in captured.err
    assert "setup_ms4_runtime.py" in captured.err


def test_require_contained_runtime_honors_override(monkeypatch):
    fake_global = Path("C:/Users/op/AppData/Local/Programs/Python/Python311/python.exe")
    monkeypatch.setattr(contained.sys, "executable", str(fake_global))
    monkeypatch.setenv(contained.ENV_OVERRIDE, "1")
    contained.require_contained_runtime("ms4-mcp-server")
    monkeypatch.setenv(contained.ENV_OVERRIDE, "true")
    contained.require_contained_runtime("ms4-mcp-server")
    monkeypatch.setenv(contained.ENV_OVERRIDE, "0")
    with pytest.raises(SystemExit):
        contained.require_contained_runtime("ms4-mcp-server")


def test_is_contained_returns_false_when_executable_unrelated(monkeypatch):
    monkeypatch.setattr(contained.sys, "executable", "C:/totally/different/python.exe")
    assert contained.is_contained() is False
