"""Subprocess entrypoint for a Double Agent worker.

Launched by ``JobRunner._spawn_worker`` as a child Python process so
the parent gateway can issue a real OS-level ``terminate()`` when an
operator cancels a job. Without this, a blocking Hermes model call
would keep running until it returned naturally, no matter how many
threading.Event flags we flipped.

The subprocess:

  1. Reads the envelope from stdin as one JSON document (avoids
     command-line length / quoting hazards on Windows).
  2. Opens the same SQLite blackboard the parent uses (SQLite WAL
     handles cross-process writes).
  3. Builds the real Hermes-backed chat runner and runs
     :class:`DoubleAgentWorker.run` in-process.
  4. Exits 0 on success, non-zero on failure. The blackboard rows are
     the source of truth; the exit code is just for parent
     book-keeping.

We deliberately don't use ``multiprocessing`` here because Hermes /
Hermes' dependencies misbehave with ``fork`` on Linux and with
``spawn`` pickling on Windows. A plain script + subprocess argument is
the path that survives every platform we ship to.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
import traceback
from pathlib import Path

# Allow the subprocess to be invoked without setting PYTHONPATH manually.
# `machine_spirit_4` is two parents up from this file.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from machine_spirit_4.double_agent import safety  # noqa: E402
from machine_spirit_4.double_agent.blackboard import Blackboard  # noqa: E402
from machine_spirit_4.double_agent.schemas import JobEnvelope, JobEvent  # noqa: E402
from machine_spirit_4.double_agent.worker import (  # noqa: E402
    DoubleAgentWorker,
    build_real_chat_runner,
)


def _read_envelope_payload() -> dict:
    raw = sys.stdin.read()
    if not raw.strip():
        raise SystemExit("Double Agent worker subprocess: no envelope on stdin")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise SystemExit("Double Agent worker subprocess: envelope must be a JSON object")
    return payload


def _install_signal_handlers(cancel_event: threading.Event) -> None:
    """Translate signals into the in-process cancel event.

    On Windows, ``terminate()`` raises ``CTRL_BREAK_EVENT``; on POSIX,
    parent code sends ``SIGTERM``. Both flip the same in-process event
    that the worker checks at every natural checkpoint.
    """

    def _handler(_signum, _frame):
        cancel_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            # Some platforms (e.g. Windows) restrict which signals a
            # subprocess may handle. Best-effort.
            pass
    if os.name == "nt":
        try:
            signal.signal(signal.SIGBREAK, _handler)  # type: ignore[attr-defined]
        except (AttributeError, ValueError, OSError):
            pass


def _build_runner_or_die(envelope: JobEnvelope, blackboard: Blackboard):
    """Construct the live Hermes runner the worker will drive.

    Failure here is recorded as a ``job.failed`` event on the
    blackboard before the process exits so the parent / UI can render
    a useful error instead of seeing the job hang in ``running``."""
    fake_spec = os.environ.get("MS4_DOUBLE_AGENT_FAKE_CHAT_RUNNER")
    if fake_spec:
        return _load_fake_chat_runner(fake_spec)
    try:
        from machine_spirit_4.gateway.hermes_runner import Ms4HermesRunner

        hermes_dir = os.environ.get(
            "MS4_HERMES_DIR", str(Path.home() / "Documents" / "hermes-agent")
        )
        hivemind_url = os.environ.get("MS4_HIVEMIND_URL", "http://127.0.0.1:6089")
        ms3_url = os.environ.get("MS4_MS3_URL", "http://127.0.0.1:9080")
        default_model = os.environ.get(
            "MS4_DEFAULT_MODEL", "qwen3-coder-next:latest"
        )
        runner = Ms4HermesRunner(
            hermes_dir=hermes_dir,
            hivemind_url=hivemind_url,
            ms3_url=ms3_url,
            default_model=default_model,
        )
        return build_real_chat_runner(runner)
    except Exception as exc:
        tb = traceback.format_exc()
        try:
            blackboard.insert_event(
                JobEvent.make(
                    job_id=envelope.job_id,
                    type="job.failed",
                    safe_user_status=f"Worker failed to construct Hermes runner: {type(exc).__name__}: {exc}",
                )
            )
            blackboard.insert_event(
                JobEvent.make(
                    job_id=envelope.job_id,
                    type="job.failed",
                    safe_user_status="operator detail attached",
                    visibility="operator_only",
                    payload={"traceback": tb[-4000:]},
                )
            )
            blackboard.update_job_state(
                envelope.job_id,
                state="failed",
                finished_at=str(time.time()),
                last_safe_user_status=f"Worker failed to start: {type(exc).__name__}",
            )
        except Exception:
            pass
        raise SystemExit(2)


def _load_fake_chat_runner(spec: str):
    """Resolve a ``module:callable`` spec to a chat runner callable.

    Used only by the test suite (set via ``MS4_DOUBLE_AGENT_FAKE_CHAT_RUNNER``)
    so we can exercise the real subprocess + blackboard wiring without
    starting Hermes/HiveMind.
    """
    if ":" not in spec:
        raise SystemExit(
            f"MS4_DOUBLE_AGENT_FAKE_CHAT_RUNNER must be 'module:callable', got {spec!r}"
        )
    module_name, _, attr = spec.partition(":")
    import importlib

    module = importlib.import_module(module_name)
    factory = getattr(module, attr)
    return factory()


def main() -> int:
    parser = argparse.ArgumentParser(description="Double Agent worker subprocess")
    parser.add_argument("--db", required=True, help="Path to the Double Agent SQLite blackboard.")
    parser.add_argument("--job-id", required=True, help="Job id; used for sanity check against envelope.")
    args = parser.parse_args()

    if not safety.is_safe_job_id(args.job_id):
        raise SystemExit(f"Double Agent worker subprocess: refused unsafe job_id={args.job_id!r}")

    payload = _read_envelope_payload()
    envelope = JobEnvelope.from_dict(payload)
    if envelope.job_id != args.job_id:
        raise SystemExit(
            f"Double Agent worker subprocess: --job-id ({args.job_id}) does not match envelope ({envelope.job_id})"
        )

    db_path = Path(args.db).resolve()
    blackboard = Blackboard(db_path)
    cancel_event = threading.Event()
    _install_signal_handlers(cancel_event)
    chat_runner = _build_runner_or_die(envelope, blackboard)

    worker = DoubleAgentWorker(
        envelope,
        blackboard=blackboard,
        cancel_event=cancel_event,
        chat_runner=chat_runner,
    )
    result = worker.run()
    return 0 if result.status == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
