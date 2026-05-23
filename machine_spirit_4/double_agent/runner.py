"""Job runner — submit, list, get, cancel, mark_stale.

Owns the in-process pool of background worker watchers plus the
per-job cancel handles. Mirrors the single-flight semantics of
``machine_spirit_4.hermes_admin.installer.trigger_update`` but
generalized to N concurrent jobs (capped at
``safety.DEFAULT_MAX_CONCURRENT_JOBS`` to avoid foreground starvation —
artifact §17).

Two execution backends:

* ``subprocess`` (default in production): each job runs in a child
  Python process launched via :mod:`machine_spirit_4.double_agent._worker_entry`.
  The blackboard is process-shared (SQLite WAL). Cancel sends
  ``SIGTERM``/``CTRL_BREAK_EVENT`` and force-kills after a grace
  period. This is the only way to interrupt a blocking Hermes model
  call before it returns.
* ``thread`` (used by the existing test suite and by callers that
  pass an explicit ``chat_runner_factory``): the worker runs on a
  ``ThreadPoolExecutor`` thread, same as before. Cancellation is
  cooperative via ``threading.Event``.

The backend is picked automatically:

* If a ``chat_runner_factory`` is set, use ``thread`` (assumed test or
  embedded mode — the caller owns the chat runner).
* Otherwise, use ``subprocess`` (production: launches the real
  Hermes-backed worker entrypoint).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from .blackboard import Blackboard, default_blackboard
from .schemas import JobEnvelope, SchemaError
from .worker import DoubleAgentWorker
from . import safety


log = logging.getLogger("ms4.double_agent.runner")


class RunnerError(RuntimeError):
    pass


ChatRunnerFactory = Callable[[], Callable[..., dict[str, Any]]]
"""Lazy factory that returns the actual chat runner callable used by the
worker. We pass a factory instead of the callable directly so the Hermes
plugin/import chain is touched only when the first job actually starts."""


class JobRunner:
    def __init__(
        self,
        *,
        blackboard: Blackboard | None = None,
        chat_runner_factory: ChatRunnerFactory | None = None,
        max_concurrent_jobs: int = safety.DEFAULT_MAX_CONCURRENT_JOBS,
        cancel_grace_seconds: float = 5.0,
    ) -> None:
        self.blackboard = blackboard or default_blackboard()
        self._chat_runner_factory = chat_runner_factory
        self._lock = threading.Lock()
        self._cancel_events: dict[str, threading.Event] = {}
        self._futures: dict[str, Future] = {}
        # subprocess.Popen per running job_id; populated in subprocess mode.
        self._procs: dict[str, subprocess.Popen[bytes]] = {}
        self._max_concurrent = max(1, int(max_concurrent_jobs))
        self._cancel_grace_seconds = max(0.5, float(cancel_grace_seconds))
        self._pool = ThreadPoolExecutor(
            max_workers=self._max_concurrent,
            thread_name_prefix="ms4-da-watcher",
        )
        self._recovered = False

    # ------- lifecycle ----------------------------------------------------

    def recover_on_startup(self) -> list[str]:
        with self._lock:
            if self._recovered:
                return []
            self._recovered = True
        affected = self.blackboard.hydrate_recover()
        if affected:
            log.info(
                "double_agent.runner: flipped %d in-flight jobs to failed during hydrate (%s)",
                len(affected),
                affected,
            )
        return affected

    # ------- submit / list / get -----------------------------------------

    def submit(self, envelope: JobEnvelope) -> dict[str, Any]:
        envelope.validate()
        envelope.state = "queued"
        self.blackboard.insert_job(envelope)
        cancel = threading.Event()
        with self._lock:
            self._cancel_events[envelope.job_id] = cancel
        if self._chat_runner_factory is not None:
            self._submit_thread_backend(envelope, cancel)
        else:
            self._submit_subprocess_backend(envelope, cancel)
        snapshot = self.blackboard.get_job_snapshot(envelope.job_id) or {}
        return snapshot

    # ---- thread backend (tests + embedded callers) ---------------------

    def _submit_thread_backend(self, envelope: JobEnvelope, cancel: threading.Event) -> None:
        runner_callable = self._chat_runner()
        worker = DoubleAgentWorker(
            envelope,
            blackboard=self.blackboard,
            cancel_event=cancel,
            chat_runner=runner_callable,
        )
        future = self._pool.submit(self._run_thread_worker, worker)
        with self._lock:
            self._futures[envelope.job_id] = future

    def _run_thread_worker(self, worker: DoubleAgentWorker) -> None:
        try:
            worker.run()
        finally:
            with self._lock:
                self._cancel_events.pop(worker.envelope.job_id, None)
                self._futures.pop(worker.envelope.job_id, None)

    # ---- subprocess backend (production: real cancellation) ------------

    def _submit_subprocess_backend(self, envelope: JobEnvelope, cancel: threading.Event) -> None:
        python = sys.executable
        entry = str(Path(__file__).resolve().parent / "_worker_entry.py")
        db_path = str(self.blackboard.path.resolve())
        cmd = [python, entry, "--db", db_path, "--job-id", envelope.job_id]
        env = os.environ.copy()
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("MS4_ALLOW_UNCONTAINED_RUNTIME", "1")
        creationflags = 0
        if os.name == "nt":
            # Put the child in its own process group so terminate() can deliver
            # CTRL_BREAK_EVENT without taking down the parent gateway.
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            creationflags=creationflags,
        )
        # Feed envelope JSON on stdin and close it so the child knows we're done.
        try:
            assert proc.stdin is not None
            proc.stdin.write(json.dumps(envelope.to_dict(), ensure_ascii=False).encode("utf-8"))
            proc.stdin.close()
        except Exception as exc:
            log.error("double_agent.runner: failed to write envelope to worker stdin: %s", exc)
            proc.terminate()
            raise
        with self._lock:
            self._procs[envelope.job_id] = proc
        future = self._pool.submit(self._watch_subprocess, envelope.job_id, proc, cancel)
        with self._lock:
            self._futures[envelope.job_id] = future

    def _watch_subprocess(
        self,
        job_id: str,
        proc: subprocess.Popen[bytes],
        cancel: threading.Event,
    ) -> None:
        """Tail the subprocess; on cancel, terminate then kill."""
        try:
            while True:
                try:
                    proc.wait(timeout=0.5)
                    break
                except subprocess.TimeoutExpired:
                    pass
                if cancel.is_set():
                    self._kill_subprocess(job_id, proc)
                    break
            # Drain stderr for the audit log (operator_only level only). Don't
            # surface stdout/stderr through events — workers communicate via
            # the blackboard.
            try:
                stderr_bytes = proc.stderr.read() if proc.stderr is not None else b""
            except Exception:
                stderr_bytes = b""
            if proc.returncode not in (0, None):
                snap = self.blackboard.get_job_snapshot(job_id)
                if snap and snap.get("state") in {"queued", "running"}:
                    # Worker died without finalizing. Mark it failed so the UI
                    # doesn't show a forever-running job.
                    self.blackboard.update_job_state(
                        job_id,
                        state="failed",
                        finished_at=str(time.time()),
                        last_safe_user_status=(
                            f"Background worker subprocess exited {proc.returncode} "
                            "without finalizing. See operator_only events."
                        ),
                    )
            if stderr_bytes:
                log.warning(
                    "double_agent.runner: worker %s stderr: %s",
                    job_id,
                    stderr_bytes.decode("utf-8", errors="replace").strip()[-2000:],
                )
        finally:
            with self._lock:
                self._cancel_events.pop(job_id, None)
                self._futures.pop(job_id, None)
                self._procs.pop(job_id, None)

    def _kill_subprocess(self, job_id: str, proc: subprocess.Popen[bytes]) -> None:
        if proc.poll() is not None:
            return
        try:
            if os.name == "nt":
                proc.send_signal(signal_break())
            else:
                proc.terminate()
        except Exception as exc:
            log.warning("double_agent.runner: terminate(%s) failed: %s", job_id, exc)
        try:
            proc.wait(timeout=self._cancel_grace_seconds)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except Exception as exc:
                log.warning("double_agent.runner: kill(%s) failed: %s", job_id, exc)
            try:
                proc.wait(timeout=self._cancel_grace_seconds)
            except subprocess.TimeoutExpired:
                pass
        snap = self.blackboard.get_job_snapshot(job_id)
        if snap and snap.get("state") in {"queued", "running"}:
            self.blackboard.update_job_state(
                job_id,
                state="canceled",
                finished_at=str(time.time()),
                last_safe_user_status="Background work was canceled (subprocess terminated).",
            )

    def list(
        self,
        *,
        conversation_id: str | None = None,
        states: tuple[str, ...] | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        return self.blackboard.list_jobs(
            conversation_id=conversation_id,
            states=states,
            limit=limit,
        )

    def get(self, job_id: str) -> dict[str, Any] | None:
        return self.blackboard.get_job_snapshot(job_id)

    # ------- cancel / mark_stale -----------------------------------------

    def cancel(self, job_id: str) -> dict[str, Any]:
        if not safety.is_safe_job_id(job_id):
            raise SchemaError(f"unsafe job_id: {job_id!r}")
        with self._lock:
            event = self._cancel_events.get(job_id)
            proc = self._procs.get(job_id)
        if event is None and proc is None:
            # No live worker handle: the job is either finished, stale,
            # or was submitted by a previous gateway process. Update the
            # blackboard directly so the caller still gets a coherent
            # response.
            snap = self.blackboard.get_job_snapshot(job_id)
            if snap is None:
                raise RunnerError(f"unknown job_id: {job_id}")
            if snap["state"] in {"queued", "running"}:
                self.blackboard.update_job_state(
                    job_id,
                    state="canceled",
                    last_safe_user_status="Background work was canceled (no live worker; updated state directly).",
                )
                return self.blackboard.get_job_snapshot(job_id) or snap
            return snap
        if event is not None:
            event.set()
        if proc is not None:
            # Tell the watcher loop to kill the child immediately rather than
            # waiting for the next poll tick.
            try:
                self._kill_subprocess(job_id, proc)
            except Exception as exc:
                log.warning("double_agent.runner: cancel kill(%s) failed: %s", job_id, exc)
        return self.blackboard.get_job_snapshot(job_id) or {}

    def mark_stale(self, job_id: str, *, reason: str = "Marked stale by operator.") -> dict[str, Any]:
        if not safety.is_safe_job_id(job_id):
            raise SchemaError(f"unsafe job_id: {job_id!r}")
        snap = self.blackboard.get_job_snapshot(job_id)
        if snap is None:
            raise RunnerError(f"unknown job_id: {job_id}")
        self.blackboard.update_job_state(
            job_id,
            state="stale",
            last_safe_user_status=safety.coerce_safe_user_status(reason),
        )
        # Best effort: also signal the worker so it stops doing useless work.
        with self._lock:
            event = self._cancel_events.get(job_id)
            proc = self._procs.get(job_id)
        if event is not None:
            event.set()
        if proc is not None:
            try:
                self._kill_subprocess(job_id, proc)
            except Exception:
                pass
        return self.blackboard.get_job_snapshot(job_id) or snap

    # ------- revision bumping --------------------------------------------

    def bump_revision(self, conversation_id: str, *, user_message_excerpt: str = "") -> dict[str, Any]:
        revision = self.blackboard.bump_revision(
            conversation_id,
            user_message_excerpt=user_message_excerpt,
        )
        stale_ids = self.blackboard.mark_stale_jobs_for_revision(
            conversation_id, revision.revision_id
        )
        with self._lock:
            for jid in stale_ids:
                event = self._cancel_events.get(jid)
                proc = self._procs.get(jid)
                if event is not None:
                    event.set()
                if proc is not None:
                    try:
                        self._kill_subprocess(jid, proc)
                    except Exception:
                        pass
        return {
            "revision": revision.to_dict(),
            "marked_stale": list(stale_ids),
        }

    # ------- internals ----------------------------------------------------

    def _chat_runner(self) -> Callable[..., dict[str, Any]]:
        if self._chat_runner_factory is None:
            raise RunnerError(
                "JobRunner is not wired to a chat runner. Either pass "
                "chat_runner_factory at construction or set it via set_chat_runner_factory()."
            )
        return self._chat_runner_factory()

    def set_chat_runner_factory(self, factory: ChatRunnerFactory) -> None:
        """Wire a thread-mode chat runner. When set, ``submit()`` runs
        the worker in-thread instead of spawning a child Python
        subprocess. Used by tests and by callers that supply their own
        chat runner (e.g. embedding MS4 in a unified process). Setting
        this back to ``None`` returns to subprocess mode."""
        self._chat_runner_factory = factory

    # ------- shutdown for tests ------------------------------------------

    def shutdown(self, *, wait: bool = True) -> None:
        with self._lock:
            for event in self._cancel_events.values():
                event.set()
            procs = list(self._procs.items())
        for jid, proc in procs:
            try:
                self._kill_subprocess(jid, proc)
            except Exception:
                pass
        self._pool.shutdown(wait=wait)


def signal_break():
    """Return the right termination signal for the host OS.

    Windows uses ``CTRL_BREAK_EVENT`` (only deliverable to a child
    process group, which is why ``_submit_subprocess_backend`` sets
    ``CREATE_NEW_PROCESS_GROUP``). POSIX uses ``SIGTERM``.
    """
    if os.name == "nt":
        return signal_const("CTRL_BREAK_EVENT")
    return signal_const("SIGTERM")


def signal_const(name: str) -> int:
    import signal as _signal

    return int(getattr(_signal, name))


_DEFAULT_RUNNER: JobRunner | None = None


def default_runner() -> JobRunner:
    global _DEFAULT_RUNNER
    if _DEFAULT_RUNNER is None:
        _DEFAULT_RUNNER = JobRunner()
    return _DEFAULT_RUNNER


def _reset_default_runner_for_tests(runner: JobRunner | None = None) -> JobRunner | None:
    global _DEFAULT_RUNNER
    if _DEFAULT_RUNNER is not None and runner is not _DEFAULT_RUNNER:
        try:
            _DEFAULT_RUNNER.shutdown(wait=False)
        except Exception:
            pass
    _DEFAULT_RUNNER = runner
    return _DEFAULT_RUNNER
