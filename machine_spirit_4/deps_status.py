from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MS4 = ROOT / "machine_spirit_4"
VENV = MS4 / ".venv"


def _venv_python() -> Path:
    if os.name == "nt":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def _import_status(name: str) -> dict[str, Any]:
    try:
        spec = importlib.util.find_spec(name)
        return {"available": spec is not None, "error": None}
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def _file_status(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.exists()}


def dependency_status() -> dict[str, Any]:
    venv_python = _venv_python()
    playwright_cache = MS4 / ".cache" / "playwright"
    ms3_binary = ROOT / "machine_spirit_3" / "target" / "debug" / (
        "machine_spirit_3.exe" if os.name == "nt" else "machine_spirit_3"
    )
    manifest_path = MS4 / "runtime" / "runtime_manifest.json"

    hermes = _import_status("run_agent")
    plugin = _import_status("plugins.ms4_consciousness")
    browser = {
        "websockets": _import_status("websockets"),
        "playwright": _import_status("playwright"),
        "playwright_browsers": {
            "path": str(playwright_cache),
            "exists": playwright_cache.exists(),
            "has_files": playwright_cache.exists() and any(playwright_cache.iterdir()),
        },
    }
    desktop = {
        "mss": _import_status("mss"),
        "pyautogui": _import_status("pyautogui"),
        "pygetwindow": _import_status("pygetwindow"),
        "desktop_control": {"enabled": os.environ.get("MS4_DESKTOP_CONTROL", "").strip().lower() in {"1", "true", "yes", "on"}},
    }

    return {
        "schema": "Ms4DependencyStatus.v1",
        "python": {
            "executable": sys.executable,
            "contained": Path(sys.executable).resolve() == venv_python.resolve(),
        },
        "venv": _file_status(VENV),
        "venv_python": _file_status(venv_python),
        "runtime_manifest": _file_status(manifest_path),
        "hermes": hermes,
        "ms4_consciousness_plugin": plugin,
        "browser": browser,
        "desktop": desktop,
        "ms3_binary": _file_status(ms3_binary),
        "capabilities": {
            "core": True,
            "hermes": bool(hermes["available"] and plugin["available"]),
            "browser": bool(browser["websockets"]["available"] and browser["playwright"]["available"]),
            "desktop_python": bool(desktop["mss"]["available"] and desktop["pyautogui"]["available"] and desktop["pygetwindow"]["available"]),
            "desktop_control": bool(desktop["desktop_control"]["enabled"] and desktop["mss"]["available"] and desktop["pyautogui"]["available"]),
        },
    }
