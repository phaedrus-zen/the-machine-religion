"""Idempotent Hermes upgrader.

Mirrors HiveMind ``ollama_admin::run_update_job``. The shape is the
same — phase transitions, persistent snapshot, blocking subprocess
launches off the request thread, recovery on failure — but the actual
install commands run through the operator's normal package manager:

  * ``editable``: ``git fetch --tags origin && git checkout v<X> && pip install -e .``
    inside MS4's contained ``.venv``. Re-stamps the
    ``ms4_consciousness`` plugin from the MS4 source-of-truth after
    checkout so a Hermes tag bump never loses it.
  * ``pypi``:     ``pip install --upgrade hermes-agent==<X>`` inside
    MS4's contained ``.venv``. No git involvement.

There is no vendored Hermes anywhere in TMR. The install commands
pull straight from GitHub (for git operations) and PyPI (for the
wheel), the same paths a fresh operator would use by hand.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable

from . import state, versioning


MS4_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLUGIN_SRC = MS4_ROOT / "plugins" / "hermes" / "ms4_consciousness"
DEFAULT_VENV_PYTHON = (
    MS4_ROOT / ".venv" / ("Scripts/python.exe" if hasattr(sys, "getwindowsversion") else "bin/python")
)


class HermesUpgradeError(RuntimeError):
    """Raised when an upgrade step fails. Surfaced to the operator."""


def _run(cmd: list[str], *, cwd: Path | None = None, timeout: int = 600) -> subprocess.CompletedProcess:
    """Blocking subprocess wrapper. Stdout/stderr are captured so the
    failure message contains the actual error text the operator needs
    to see in the UI, not a generic non-zero exit code.
    """
    state.append_progress(f"$ {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise HermesUpgradeError(f"Command timed out after {timeout}s: {' '.join(cmd)}") from exc
    except FileNotFoundError as exc:
        raise HermesUpgradeError(f"Command not found ({cmd[0]}): {exc}") from exc
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        message = stderr or stdout or f"exit {result.returncode}"
        raise HermesUpgradeError(f"{cmd[0]} exit={result.returncode}: {message[:600]}")
    if result.stdout:
        state.append_progress(result.stdout.strip().splitlines()[-1][:200])
    return result


def _git_fetch_checkout(directory: Path, target_tag: str) -> None:
    if not directory.exists():
        raise HermesUpgradeError(f"Hermes checkout not found at {directory}")
    if not (directory / ".git").exists():
        raise HermesUpgradeError(f"{directory} is not a git checkout; cannot tag-pin in editable mode")
    state.set_phase("fetching_remote")
    _run(["git", "-C", str(directory), "fetch", "--tags", "--prune", "origin"], timeout=600)
    state.set_phase("checking_out")
    _run(
        ["git", "-C", str(directory), "-c", "advice.detachedHead=false", "checkout", target_tag],
        timeout=120,
    )


def _sync_plugin(plugin_src: Path, plugin_dst: Path) -> None:
    """Re-stamp ``ms4_consciousness`` plugin from MS4's source-of-truth.

    Modeled on Ollama's "re-warm" step — after a Hermes checkout swap,
    we restore our plugin into the new tree before the editable install
    re-registers it. The source lives outside the Hermes checkout so a
    tag bump never deletes it.
    """
    if not plugin_src.exists():
        raise HermesUpgradeError(f"MS4 plugin source missing: {plugin_src}")
    state.set_phase("syncing_plugin", note=f"sync {plugin_src} -> {plugin_dst}")
    plugin_dst.parent.mkdir(parents=True, exist_ok=True)
    if plugin_dst.exists():
        shutil.rmtree(plugin_dst, ignore_errors=False)
    shutil.copytree(plugin_src, plugin_dst)


def _pip_install_editable(python: Path, directory: Path) -> None:
    state.set_phase("pip_installing")
    _run([str(python), "-m", "pip", "install", "-e", str(directory)], timeout=900)


def _pip_install_wheel(python: Path, target_version: str) -> None:
    state.set_phase("pip_installing")
    _run(
        [str(python), "-m", "pip", "install", "--upgrade", f"hermes-agent=={target_version}"],
        timeout=900,
    )


def _validate(python: Path) -> str:
    state.set_phase("validating")
    result = _run(
        [
            str(python),
            "-c",
            "import importlib.metadata as m, plugins.ms4_consciousness as p; "
            "print(m.version('hermes-agent'))",
        ],
        timeout=60,
    )
    version = (result.stdout or "").strip().splitlines()[-1] if result.stdout else ""
    if not version:
        raise HermesUpgradeError("Validation succeeded but version output was empty")
    return version


def run_update_job(
    *,
    target_version: str | None,
    plugin_src: Path = DEFAULT_PLUGIN_SRC,
    venv_python: Path = DEFAULT_VENV_PYTHON,
    runner: Callable[[list[str]], subprocess.CompletedProcess] | None = None,
) -> dict[str, str | None]:
    """Run an upgrade synchronously. Used by tests and by the async
    spawn path in :func:`trigger_update`.

    Returns ``{"to_version": "0.14.0"}`` on success. Raises
    :class:`HermesUpgradeError` on failure (and finalizes the
    persisted job snapshot with ``status="failed"``).
    """
    try:
        state.set_phase("preflight")
        if target_version is not None and not versioning.is_safe_target_version(target_version):
            raise HermesUpgradeError(f"refused unsafe target_version: {target_version!r}")
        mode = versioning.install_mode()
        if mode.get("mode") == "missing":
            raise HermesUpgradeError("Hermes is not installed in the contained venv")

        state.set_phase("fetching_release")
        latest = versioning.latest_version(force_refresh=True)
        # Refresh the recent releases too so resolve_git_tag can pair the
        # operator-picked wheel version to its actual git tag.
        versioning.recent_releases(force_refresh=True)
        if not latest and not target_version:
            raise HermesUpgradeError("Could not determine latest Hermes release and no target_version supplied")
        target = target_version or (latest.version if latest else "")
        if not target or not versioning.is_safe_target_version(target):
            raise HermesUpgradeError(f"resolved target_version is unsafe or empty: {target!r}")

        if mode["mode"] == "editable":
            tag = versioning.resolve_git_tag(target) or (target if target.startswith("v") else f"v{target}")
            if not versioning.is_safe_target_version(tag.lstrip("v")):
                raise HermesUpgradeError(f"resolved git tag is unsafe: {tag!r}")
            directory = mode.get("directory") or versioning.hermes_dir()
            _git_fetch_checkout(Path(directory), tag)
            _sync_plugin(plugin_src, Path(directory) / "plugins" / "ms4_consciousness")
            _pip_install_editable(venv_python, Path(directory))
        elif mode["mode"] == "pypi":
            _pip_install_wheel(venv_python, target)
        else:
            raise HermesUpgradeError(f"Unknown Hermes install mode: {mode['mode']!r}")

        installed_version = _validate(venv_python)
        state.finalize_job(to_version=installed_version)
        return {"to_version": installed_version}
    except HermesUpgradeError as exc:
        state.finalize_job(error=str(exc))
        raise
    except Exception as exc:
        state.finalize_job(error=f"unexpected error: {exc}")
        raise HermesUpgradeError(str(exc)) from exc


def trigger_update(
    *,
    target_version: str | None = None,
    request_user: str | None = None,
    plugin_src: Path = DEFAULT_PLUGIN_SRC,
    venv_python: Path = DEFAULT_VENV_PYTHON,
) -> state.UpdateJobSnapshot:
    """Spawn a background upgrade. Returns the running job snapshot
    immediately. Idempotent — refuses to start a second job while one
    is already running.
    """
    state.initialize_state()
    if state.update_in_progress():
        existing = state.last_update()
        assert existing is not None
        return existing

    if target_version is not None and not versioning.is_safe_target_version(target_version):
        raise HermesUpgradeError(f"refused unsafe target_version: {target_version!r}")

    mode = versioning.install_mode()
    from_version = mode.get("version")
    latest = versioning.latest_version()
    to_version = target_version or (latest.version if latest else None)
    snap = state.start_job(
        from_version=from_version,
        to_version=to_version,
        install_mode=mode.get("mode"),
        request_user=request_user,
    )

    def worker() -> None:
        try:
            run_update_job(
                target_version=target_version,
                plugin_src=plugin_src,
                venv_python=venv_python,
            )
        except HermesUpgradeError:
            pass

    thread = threading.Thread(target=worker, daemon=True, name=f"hermes-upgrade-{snap.job_id[:8]}")
    thread.start()
    return snap
