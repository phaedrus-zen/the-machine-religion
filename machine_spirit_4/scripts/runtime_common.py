from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MS3 = ROOT / "machine_spirit_3"
MS4 = ROOT / "machine_spirit_4"
VENV = MS4 / ".venv"
LOG_DIR = MS4 / "logs"


def venv_python() -> Path:
    if os.name == "nt":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def require_venv_python() -> Path:
    python = venv_python()
    if not python.exists():
        raise RuntimeError("MS4 contained runtime is missing. Run: python machine_spirit_4/scripts/setup_ms4_runtime.py")
    return python


def hermes_dir() -> Path:
    return Path(os.environ.get("MS4_HERMES_DIR") or (Path.home() / "Documents" / "hermes-agent"))


def hermes_home() -> Path:
    """The Hermes DATA home (config.yaml, sessions, memories) — distinct
    from :func:`hermes_dir` (the Hermes CODE checkout).

    Pinned to the legacy ``~/.hermes`` this deployment has always used.
    The June 8 2026 Hermes upgrade moved the platform-native default on
    Windows to ``%LOCALAPPDATA%/hermes``, silently orphaning the real
    config (including ``plugins.enabled: [ms4_consciousness]``) and
    breaking every Depth Lobe job with ``HermesUnavailable: plugin not
    enabled``. Hermes' own guidance (issue #18594) is that subprocess
    spawners must propagate ``HERMES_HOME`` explicitly — this is that.
    Override with ``HERMES_HOME`` in the environment."""
    return Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))


def playwright_browsers_path() -> Path:
    return MS4 / ".cache" / "playwright"


def ms3_binary(profile: str = "debug") -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    return MS3 / "target" / profile / f"machine_spirit_3{suffix}"


def ms4_env(
    *,
    gateway_port: int | None = None,
    gateway_host: str | None = None,
    mcp_port: int | None = None,
    mcp_host: str | None = None,
    ms3_port: int | None = None,
    ms3_host: str | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("MS4_HERMES_DIR", str(hermes_dir()))
    env.setdefault("HERMES_HOME", str(hermes_home()))
    env.setdefault("MS4_HIVEMIND_URL", "http://127.0.0.1:6089")
    env.setdefault("MS4_DEFAULT_MODEL", "qwen3-coder-next:latest")
    env.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(playwright_browsers_path()))
    if gateway_port is not None:
        env["MS4_GATEWAY_PORT"] = str(gateway_port)
    if gateway_host is not None:
        env["MS4_GATEWAY_HOST"] = gateway_host
    if mcp_port is not None:
        env["MS4_MCP_PORT"] = str(mcp_port)
    if mcp_host is not None:
        env["MS4_MCP_HOST"] = mcp_host
    if ms3_port is not None:
        env["MS3_PORT"] = str(ms3_port)
        env.setdefault("MS4_MS3_URL", f"http://127.0.0.1:{ms3_port}")
        env.setdefault("MS4_MS3_SIDECAR_URL", f"http://127.0.0.1:{ms3_port}")
    else:
        env.setdefault("MS4_MS3_URL", "http://127.0.0.1:9080")
        env.setdefault("MS4_MS3_SIDECAR_URL", env["MS4_MS3_URL"])
    if ms3_host is not None:
        env["MS3_HOST"] = ms3_host
    env.setdefault("MS4_SPIRIT_ID", "sister")
    return env


def is_port_listening(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def wait_http(url: str, seconds: int = 45, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                if 200 <= response.status < 500:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def request_json(method: str, url: str, payload: dict[str, Any] | None = None, timeout: int = 8) -> tuple[int, str, Any]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            text = response.read().decode("utf-8", "replace")
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                data = None
            return response.status, text, data
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", "replace")
        return exc.code, text, None


def launch_process(
    args: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
) -> subprocess.Popen:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stdout = stdout_path.open("ab")
    stderr = stderr_path.open("ab")
    return subprocess.Popen(args, cwd=str(cwd), env=env, stdout=stdout, stderr=stderr)


def run_script(script: Path, *, env: dict[str, str] | None = None, timeout: int | None = None) -> int:
    python = require_venv_python()
    return subprocess.run([str(python), str(script)], cwd=str(ROOT), env=env or ms4_env(), timeout=timeout).returncode


def print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2), flush=True)


def exit_code(ok: bool) -> int:
    return 0 if ok else 1
