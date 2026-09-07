"""Contained-runtime guard for MS4 service entrypoints.

MS4 is designed to run from its own service-local Python environment at
``machine_spirit_4/.venv`` (mirrors HiveMind's microservice pattern).
Any service entrypoint that boots from a different interpreter — the
host's global ``python``, a stale Cursor background task, an operator
shell with the wrong activation — gets contaminated dependency state:
``importlib.metadata`` resolves against a different ``site-packages``,
the Hermes editable install record disappears, the Hermes auto-update
banner lies about ``install_mode``, and two processes can race for the
TCP bind.

The guard is a hard refusal. Service entrypoints call
``require_contained_runtime`` immediately at startup and exit with a
clear operator message rather than serving stale data.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


MS4_ROOT = Path(__file__).resolve().parent
EXPECTED_VENV = (MS4_ROOT / ".venv").resolve()
ENV_OVERRIDE = "MS4_ALLOW_UNCONTAINED_RUNTIME"
EXIT_CONTAINMENT_VIOLATION = 78  # EX_CONFIG


def _normalized(path: Path) -> Path:
    return Path(os.path.normcase(str(path.resolve())))


def is_contained() -> bool:
    """True when the current interpreter lives inside MS4's ``.venv``."""
    if not EXPECTED_VENV.exists():
        return False
    expected = _normalized(EXPECTED_VENV)
    executable = _normalized(Path(sys.executable))
    return expected == executable or expected in executable.parents


def expected_python() -> Path:
    """The interpreter path the service is supposed to run from."""
    if os.name == "nt":
        return EXPECTED_VENV / "Scripts" / "python.exe"
    return EXPECTED_VENV / "bin" / "python"


def require_contained_runtime(service_name: str) -> None:
    """Refuse to start when the active interpreter is not MS4's ``.venv``.

    Set ``MS4_ALLOW_UNCONTAINED_RUNTIME=1`` to override (intended for
    test environments that import the service module without running
    it as the active interpreter, e.g. pytest-driven asset checks).

    Uses ``os._exit`` rather than ``sys.exit`` for the violation path
    because we have seen real-world ghost processes hang inside
    ``sys.stderr.flush()`` when their stderr is captured by a
    non-draining IDE/task-runner pipe. ``os._exit`` does not run
    cleanups, atexit handlers, or buffer flushes — exactly what a hard
    refusal needs.
    """
    if os.environ.get(ENV_OVERRIDE, "").strip().lower() in {"1", "true", "yes", "on"}:
        return
    if is_contained():
        return
    message = (
        f"MS4 {service_name} refused to start: containment violation.\n"
        f"  Active interpreter: {sys.executable}\n"
        f"  Expected:           {expected_python()}\n"
        f"  Fix: python machine_spirit_4/scripts/setup_ms4_runtime.py\n"
        f"  Then launch via:    python machine_spirit_4/scripts/start_ms4.py\n"
        f"  Override (tests):   set {ENV_OVERRIDE}=1\n"
    )
    # Best-effort write to stderr and to a stable log path. If either
    # blocks we still hard-exit below; nothing should keep this
    # process alive once containment is violated.
    try:
        sys.stderr.write(message)
        sys.stderr.flush()
    except Exception:
        pass
    if _called_from_pytest():
        raise SystemExit(EXIT_CONTAINMENT_VIOLATION)
    try:
        log_dir = MS4_ROOT / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        with (log_dir / "containment_violations.log").open("a", encoding="utf-8") as handle:
            handle.write(message + "---\n")
    except Exception:
        pass
    os._exit(EXIT_CONTAINMENT_VIOLATION)


def _called_from_pytest() -> bool:
    """True when running under pytest. Lets tests assert ``SystemExit``
    instead of being torn down by ``os._exit``, which pytest cannot
    intercept (no finally / no traceback)."""
    if "pytest" in sys.modules:
        return True
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return True
    return False
