from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MS4 = ROOT / "machine_spirit_4"


def test_dependency_manifests_define_core_and_optional_groups():
    core = (MS4 / "requirements.txt").read_text(encoding="utf-8")
    full = (MS4 / "requirements-full-hermes.txt").read_text(encoding="utf-8")
    lock = json.loads((MS4 / "deps.lock.json").read_text(encoding="utf-8"))

    assert "C:/Users/" not in full
    assert "websockets" in full
    assert "playwright" in full
    assert "mss" in full
    assert "pyautogui" in full
    assert "pygetwindow" in full
    assert lock["schema"] == "Ms4DependencyLock.v1"
    assert "browser" in lock["capability_groups"]
    assert "desktop" in lock["capability_groups"]
    assert "hermes" in lock["capability_groups"]
    assert lock["capability_groups"]["hermes"]["editable_env"] == "MS4_HERMES_DIR"
    assert "jsonschema" in core


def test_setup_and_check_scripts_exist_with_expected_contracts():
    setup_py = (MS4 / "scripts" / "setup_ms4_runtime.py").read_text(encoding="utf-8")
    runtime_common = (MS4 / "scripts" / "runtime_common.py").read_text(encoding="utf-8")
    check_py = (MS4 / "scripts" / "check_ms4_deps.py").read_text(encoding="utf-8")

    assert ".venv" in runtime_common
    assert "venv_python" in setup_py
    assert "requirements-full-hermes.txt" in setup_py
    assert "runtime_manifest.json" in setup_py
    assert "playwright" in check_py
    assert "pyautogui" in check_py
    assert "mss" in check_py
    assert "desktop_control" in check_py


def test_launchers_require_contained_python():
    for script_name in ("run_ms4_gateway.py", "run_ms4_mcp.py", "start_ms4.py"):
        script = (MS4 / "scripts" / script_name).read_text(encoding="utf-8")
        assert "require_venv_python" in script


def test_runtime_scripts_are_cross_platform_python_entrypoints():
    forbidden_suffixes = {".ps1", ".sh", ".bat", ".cmd"}
    offenders = [
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.suffix.lower() in forbidden_suffixes
        and ".git" not in path.parts
        and ".venv" not in path.parts
    ]

    assert offenders == []


def test_gateway_and_mcp_expose_dependency_status_assets():
    gateway = (MS4 / "gateway" / "server.py").read_text(encoding="utf-8")
    mcp_tools = (MS4 / "mcp" / "tools.py").read_text(encoding="utf-8")

    assert '"/deps/status"' in gateway
    assert "ms4.runtime.deps.status@v1" in mcp_tools
