from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path, PurePath
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MS3 = ROOT / "machine_spirit_3"
MS4 = ROOT / "machine_spirit_4"
VENV = MS4 / ".venv"
LOG_DIR = MS4 / "logs"
WINDOWS_HERMES_GIT_BASH_PATH = r"C:\Program Files\Git\bin\bash.exe"
HERMES_PROVENANCE_SIGNER_PRINCIPAL = "127238744+teknium1@users.noreply.github.com"
HERMES_PROVENANCE_SIGNER_KEY = (
    "ssh-ed25519 "
    "AAAAC3NzaC1lZDI1NTE5AAAAIPpWPAE2WMbZ0fAZ8xsqiTIJqA28qDBfGru8kPrpNyUb"
)
HERMES_PROVENANCE_SIGNER_FINGERPRINT = (
    "SHA256:x9xNOpeJhoEAY2gWhmWHZROC3QF3VjOEbmNo9vQ8y2A"
)
HERMES_PROVENANCE_ALLOWED_SIGNERS = (
    f'{HERMES_PROVENANCE_SIGNER_PRINCIPAL} namespaces="git" '
    f"{HERMES_PROVENANCE_SIGNER_KEY}"
)
HERMES_PROVENANCE_ALLOWED_ORIGIN = "https://github.com/NousResearch/hermes-agent.git"
WINDOWS_HERMES_PROVENANCE_SSH_KEYGEN = (
    r"C:\Windows\System32\OpenSSH\ssh-keygen.exe"
)
POSIX_HERMES_PROVENANCE_SSH_KEYGEN = "/usr/bin/ssh-keygen"


def venv_python() -> Path:
    if os.name == "nt":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def require_venv_python() -> Path:
    python = venv_python()
    if not python.exists():
        raise RuntimeError("MS4 contained runtime is missing. Run: python machine_spirit_4/scripts/setup_ms4_runtime.py")
    return python


_ACTIVE_CHECKOUT_SCHEMA = "Ms4HermesActiveCheckout.v1"
_ACTIVE_CHECKOUT_MARKER_NAME = "hermes_active_checkout.json"
_MANAGED_CHECKOUT_NAME = "hermes-managed"
_SAFE_VERSION_RE = re.compile(r"^[0-9A-Za-z._-]{1,32}$")
_COMMIT_RE = re.compile(r"^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$")
_GIT_REPOSITORY_ENV_VARS = frozenset(
    {
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_CEILING_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_DIR",
        "GIT_DISCOVERY_ACROSS_FILESYSTEM",
        "GIT_INDEX_FILE",
        "GIT_NAMESPACE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_PREFIX",
        "GIT_WORK_TREE",
    }
)


def _active_checkout_from_marker(ms4_root: Path) -> Path | None:
    marker = ms4_root / "runtime" / _ACTIVE_CHECKOUT_MARKER_NAME
    if not marker.is_file():
        return None
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("schema") != _ACTIVE_CHECKOUT_SCHEMA:
        return None
    directory = payload.get("directory")
    version = payload.get("version")
    commit = payload.get("commit")
    if not isinstance(directory, str) or not directory.strip():
        return None
    if not isinstance(version, str) or not _SAFE_VERSION_RE.fullmatch(version):
        return None
    if not isinstance(commit, str) or not _COMMIT_RE.fullmatch(commit):
        return None

    candidate = Path(directory).expanduser()
    runtime_root = ms4_root / "runtime"
    expected = runtime_root / _MANAGED_CHECKOUT_NAME
    try:
        if os.path.normcase(os.path.abspath(str(candidate))) != os.path.normcase(
            os.path.abspath(str(expected))
        ):
            return None
        ms4_real = os.path.normcase(os.path.realpath(os.path.abspath(str(ms4_root))))
        runtime_real = os.path.normcase(
            os.path.realpath(os.path.abspath(str(runtime_root)))
        )
        expected_runtime_real = os.path.normcase(os.path.join(ms4_real, "runtime"))
        candidate_real = os.path.normcase(
            os.path.realpath(os.path.abspath(str(candidate)))
        )
        expected_candidate_real = os.path.normcase(
            os.path.join(expected_runtime_real, _MANAGED_CHECKOUT_NAME)
        )
        git_dir = candidate / ".git"
        expected_git_real = os.path.normcase(
            os.path.join(expected_candidate_real, ".git")
        )
        if (
            os.path.islink(runtime_root)
            or os.path.islink(candidate)
            or os.path.islink(git_dir)
            or runtime_real != expected_runtime_real
            or candidate_real != expected_candidate_real
            or not git_dir.is_dir()
            or os.path.normcase(os.path.realpath(str(git_dir)))
            != expected_git_real
        ):
            return None
    except (OSError, ValueError):
        return None
    git_env = os.environ.copy()
    for key in _GIT_REPOSITORY_ENV_VARS:
        git_env.pop(key, None)
    git_env["GIT_OPTIONAL_LOCKS"] = "0"
    git_env["GIT_TERMINAL_PROMPT"] = "0"
    try:
        head = subprocess.run(
            [
                "git",
                "-C",
                str(candidate),
                "rev-parse",
                "--absolute-git-dir",
                "--show-toplevel",
                "--verify",
                "HEAD^{commit}",
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
            env=git_env,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = [line.strip() for line in (head.stdout or "").splitlines() if line.strip()]
    if head.returncode != 0 or len(lines) != 3:
        return None
    reported_gitdir, reported_top, reported_head = lines
    try:
        if (
            os.path.normcase(os.path.realpath(reported_gitdir)) != expected_git_real
            or os.path.normcase(os.path.realpath(reported_top))
            != expected_candidate_real
            or reported_head.lower() != commit.lower()
        ):
            return None
    except (OSError, ValueError):
        return None
    return candidate


def _default_hermes_dir(
    platform_name: str,
    environ: Mapping[str, str],
    home: Path | PurePath,
) -> Path | PurePath:
    def configured_path(value: str) -> Path | PurePath:
        if isinstance(home, Path):
            return Path(value).expanduser()
        return type(home)(value)

    raw = environ.get("MS4_HERMES_DIR")
    if raw:
        return configured_path(raw)
    if platform_name == "win32":
        return home / "Documents" / "hermes-agent"
    if platform_name == "darwin":
        return home / "Library" / "Application Support" / "hermes-agent"
    xdg = environ.get("XDG_DATA_HOME")
    if xdg:
        return configured_path(xdg) / "hermes-agent"
    return home / ".local" / "share" / "hermes-agent"


def hermes_dir() -> Path:
    managed = _active_checkout_from_marker(MS4)
    if managed is not None:
        return managed
    return Path(_default_hermes_dir(sys.platform, os.environ, Path.home()))


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
    env["MS4_HERMES_DIR"] = str(hermes_dir())
    env.setdefault("HERMES_HOME", str(hermes_home()))
    if sys.platform == "win32":
        env["HERMES_GIT_BASH_PATH"] = WINDOWS_HERMES_GIT_BASH_PATH
    env.pop("MS4_HERMES_PROVENANCE_ALLOWED_SIGNERS", None)
    env["MS4_HERMES_PROVENANCE_ALLOWED_SIGNERS_INLINE"] = (
        HERMES_PROVENANCE_ALLOWED_SIGNERS
    )
    env["MS4_HERMES_PROVENANCE_ALLOWED_ORIGINS"] = (
        HERMES_PROVENANCE_ALLOWED_ORIGIN
    )
    env["MS4_HERMES_PROVENANCE_ALLOWED_PRINCIPALS"] = (
        HERMES_PROVENANCE_SIGNER_PRINCIPAL
    )
    env["MS4_HERMES_PROVENANCE_ALLOWED_FINGERPRINTS"] = (
        HERMES_PROVENANCE_SIGNER_FINGERPRINT
    )
    env["MS4_HERMES_PROVENANCE_SSH_KEYGEN"] = (
        WINDOWS_HERMES_PROVENANCE_SSH_KEYGEN
        if sys.platform == "win32"
        else POSIX_HERMES_PROVENANCE_SSH_KEYGEN
    )
    if "MS4_HIVEMIND_URL" not in env and env.get("MS4_HIVEMIND_HLI_URL"):
        env["MS4_HIVEMIND_URL"] = env["MS4_HIVEMIND_HLI_URL"]
    env.setdefault("MS4_HIVEMIND_URL", "http://127.0.0.1:6089")
    env.setdefault("MS4_HIVEMIND_HLI_URL", env["MS4_HIVEMIND_URL"])
    # Face and Depth are independent cluster roles. A 16 GiB node is an
    # eligible fleet tier; it is not expected to co-reside both models.
    # Managed runtime uses the GPU-aware profile: an installed/reachable
    # qwen3.6:35b is preferred only when carrier telemetry proves >=32 GiB VRAM;
    # otherwise the existing low-latency Face policy remains in force.
    env.setdefault("MS4_FACE_PROFILE", "auto")
    env.setdefault("MS4_DEFAULT_MODEL", "nemotron-3-nano:4b")
    env.setdefault("MS4_DEPTH_FALLBACK_MODEL", "nemotron-3-nano:30b")
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
    else:
        env.setdefault("MS3_HOST", "127.0.0.1")
    env.setdefault("MS4_SPIRIT_ID", "sister")
    return env


def is_port_listening(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def wait_http(
    url: str, seconds: int = 45, timeout: float = 3.0, *, expected_json: Mapping[str, Any]
) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                if 200 <= response.status < 300:
                    payload = json.loads(response.read())
                    if isinstance(payload, Mapping) and all(
                        payload.get(key) == value for key, value in expected_json.items()
                    ):
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
    try:
        return subprocess.Popen(
            args,
            cwd=str(cwd),
            env=env,
            stdout=stdout,
            stderr=stderr,
            close_fds=True,
        )
    finally:
        stdout.close()
        stderr.close()


def run_script(script: Path, *, env: dict[str, str] | None = None, timeout: int | None = None) -> int:
    python = require_venv_python()
    return subprocess.run([str(python), str(script)], cwd=str(ROOT), env=env or ms4_env(), timeout=timeout).returncode


def print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2), flush=True)


def exit_code(ok: bool) -> int:
    return 0 if ok else 1
