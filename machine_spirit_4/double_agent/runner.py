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
import urllib.error
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .blackboard import Blackboard, default_blackboard
from .continuation import (
    CLASSIFIER_CONSULT_MAX_LEN,
    ContinuationClassifier,
    phrase_is_continuation,
)
from .schemas import JobEnvelope, JobEvent, JobResult, SchemaError
from .worker import DoubleAgentWorker
from .plan_templates import GAMESTREAM_LOBE_TYPE
from . import safety


log = logging.getLogger("ms4.double_agent.runner")


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return max(minimum, int(raw.strip()))
    except (TypeError, ValueError):
        log.warning("ms4.double_agent: ignoring non-integer %s=%r; default %d", name, raw, default)
        return default


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return max(minimum, float(raw.strip()))
    except (TypeError, ValueError):
        log.warning("ms4.double_agent: ignoring non-float %s=%r; default %s", name, raw, default)
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _autostale_enabled() -> bool:
    """Whether a new user turn auto-stales in-flight Depth Lobe jobs from
    older revisions. Default OFF (May 30 2026): deep jobs run to
    completion and the system delivers the result when done; only an
    explicit cancel stops a job. Set ``MS4_DA_AUTOSTALE=1`` to restore the
    legacy revision-driven staling (continuation-aware)."""
    return _env_bool("MS4_DA_AUTOSTALE", False)


def _hivemind_trace_id(job_id: str) -> str | None:
    if not job_id.startswith("da-"):
        return None
    try:
        return str(uuid.UUID(job_id.removeprefix("da-")))
    except (ValueError, AttributeError):
        return None


def _cancel_hivemind_trace(job_id: str, *, reason: str) -> bool:
    """Mark the HLI request owned by a stopped Depth worker terminal.

    HLI's operator-kill contract is observability cancellation: it removes the
    exact trace from ``/jobs/active`` and makes the terminal verdict sticky.
    Closing the worker still interrupts the client side of the HTTP request;
    this call keeps HLI accounting aligned with that process-level stop.
    """
    if os.environ.get("MS4_DOUBLE_AGENT_FAKE_CHAT_RUNNER"):
        return False
    trace_id = _hivemind_trace_id(job_id)
    if trace_id is None:
        return False
    base_url = (
        os.environ.get("MS4_HIVEMIND_URL")
        or os.environ.get("MS4_HIVEMIND_HLI_URL")
        or "http://127.0.0.1:6089"
    ).rstrip("/")
    query = urllib.parse.urlencode({"reason": reason})
    request = urllib.request.Request(
        f"{base_url}/jobs/{trace_id}?{query}",
        method="DELETE",
    )
    try:
        with urllib.request.urlopen(request, timeout=5.0) as response:
            return 200 <= int(response.status) < 300
    except (urllib.error.HTTPError, urllib.error.URLError, OSError, TimeoutError) as exc:
        log.warning("double_agent.runner: HLI trace cancel failed for %s: %s", job_id, exc)
        return False


# Backwards-compatible alias. The phrase fast-path moved to
# :mod:`machine_spirit_4.double_agent.continuation`; keep the old name
# importable for any caller/test that referenced it.
_is_short_continuation = phrase_is_continuation


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
        max_concurrent_jobs: int | None = None,
        cancel_grace_seconds: float | None = None,
        post_tool_final_timeout_seconds: float | None = None,
        continuation_classifier: ContinuationClassifier | None = None,
    ) -> None:
        self.blackboard = blackboard or default_blackboard()
        self._chat_runner_factory = chat_runner_factory
        self._continuation_classifier = continuation_classifier
        self._lock = threading.Lock()
        self._cancel_events: dict[str, threading.Event] = {}
        self._futures: dict[str, Future] = {}
        # subprocess.Popen per running job_id; populated in subprocess mode.
        self._procs: dict[str, subprocess.Popen[bytes]] = {}
        self._canceling_jobs: set[str] = set()
        # Concurrency cap + cancel grace are env-tunable so an operator can
        # trade foreground headroom / shutdown latency without a code change.
        # Explicit constructor args still win (tests pass them directly).
        if max_concurrent_jobs is None:
            max_concurrent_jobs = _env_int(
                "MS4_DA_MAX_CONCURRENT_JOBS", safety.DEFAULT_MAX_CONCURRENT_JOBS, minimum=1
            )
        if cancel_grace_seconds is None:
            cancel_grace_seconds = _env_float(
                "MS4_DA_CANCEL_GRACE_SECONDS", 5.0, minimum=0.5
            )
        if post_tool_final_timeout_seconds is None:
            post_tool_final_timeout_seconds = _env_float(
                "MS4_DA_POST_TOOL_FINAL_TIMEOUT_SECONDS", 45.0, minimum=0.0
            )
        self._max_concurrent = max(1, int(max_concurrent_jobs))
        self._cancel_grace_seconds = max(0.5, float(cancel_grace_seconds))
        self._post_tool_final_timeout_seconds = max(0.0, float(post_tool_final_timeout_seconds))
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
        if self._depth_worker_preflight_enabled(envelope):
            preflight = self._run_depth_worker_preflight(entry, env=env)
            if not preflight.get("ok"):
                self._fail_depth_worker_preflight(envelope, preflight)
                with self._lock:
                    self._cancel_events.pop(envelope.job_id, None)
                return
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

    def _depth_worker_preflight_enabled(self, envelope: JobEnvelope) -> bool:
        if os.environ.get("MS4_DOUBLE_AGENT_FAKE_CHAT_RUNNER"):
            return False
        if envelope.background_lobe_type == GAMESTREAM_LOBE_TYPE:
            return False
        return _env_bool("MS4_DA_DEPTH_PREFLIGHT", True)

    def _run_depth_worker_preflight(self, entry: str, *, env: dict[str, str]) -> dict[str, Any]:
        timeout = _env_float("MS4_DA_DEPTH_PREFLIGHT_TIMEOUT_SECONDS", 3.0, minimum=0.1)
        cmd = [
            sys.executable,
            entry,
            "--preflight",
            "--json",
            "--timeout",
            str(timeout),
        ]
        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                timeout=max(5.0, timeout + 3.0),
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "ok": False,
                "reason": "depth_worker_preflight_timeout",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "stdout_excerpt": (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
                "stderr_excerpt": (exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else "",
            }
        except Exception as exc:  # noqa: BLE001 - parent gate must fail closed
            return {
                "ok": False,
                "reason": "depth_worker_preflight_exception",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        try:
            report = json.loads(stdout) if stdout.strip() else {}
        except json.JSONDecodeError as exc:
            report = {
                "parse_error": str(exc),
                "stdout_excerpt": stdout[-2000:],
            }
        if not isinstance(report, dict):
            report = {"parse_error": "preflight stdout was not a JSON object"}
        ok = completed.returncode == 0 and bool(report.get("ok"))
        return {
            "ok": ok,
            "reason": None if ok else "depth_worker_preflight_failed",
            "returncode": completed.returncode,
            "report": report,
            "stderr_excerpt": stderr[-2000:],
        }

    def _fail_depth_worker_preflight(
        self,
        envelope: JobEnvelope,
        preflight: dict[str, Any],
    ) -> None:
        summary = self._depth_preflight_user_summary(preflight)
        report = preflight.get("report") if isinstance(preflight.get("report"), dict) else {}
        payload = {
            "reason": "depth_worker_preflight_failed",
            "diagnostic_codes": self._depth_preflight_diagnostic_codes(report),
            "unreachable_probes": self._depth_preflight_unreachable_probes(report),
        }
        self.blackboard.insert_event(
            JobEvent.make(
                job_id=envelope.job_id,
                type="job.failed",
                safe_user_status=summary,
                payload=payload,
            )
        )
        self.blackboard.insert_event(
            JobEvent.make(
                job_id=envelope.job_id,
                type="job.failed",
                safe_user_status="Depth worker preflight diagnostics attached.",
                visibility="operator_only",
                payload={"preflight": preflight},
            )
        )
        self.blackboard.insert_result(
            JobResult(
                job_id=envelope.job_id,
                status="failed",
                summary=summary,
                text=(
                    f"{summary}\n\n"
                    "The Depth worker was not launched, because its own namespace "
                    "could not prove reachability to the configured services."
                ),
                actions_taken=[{"action": "depth_worker_preflight"}],
                next_steps=[
                    "Set host-routable MS4_HIVEMIND_URL, MS4_HIVEMIND_MCP_URL, "
                    "and MS4_MS3_URL values for the worker namespace, or restore "
                    "the target services, then retry."
                ],
                confidence="high",
                conversation_revision_id=envelope.conversation_revision_id,
                turn_id=envelope.turn_id,
            )
        )
        self.blackboard.update_job_state(
            envelope.job_id,
            state="failed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            last_safe_user_status=summary,
        )

    def _depth_preflight_user_summary(self, preflight: dict[str, Any]) -> str:
        report = preflight.get("report") if isinstance(preflight.get("report"), dict) else {}
        diagnostics = report.get("diagnostics") if isinstance(report, dict) else None
        if isinstance(diagnostics, list):
            for diagnostic in diagnostics:
                if (
                    isinstance(diagnostic, dict)
                    and diagnostic.get("code") == "posix_loopback_namespace_unreachable"
                ):
                    return (
                        "Depth worker preflight failed before launch: the worker namespace "
                        "cannot reach HiveMind/MS3/MCP loopback URLs. Use host-routable MS4_* URLs and retry."
                    )
        return (
            "Depth worker preflight failed before launch: the worker cannot reach the "
            "configured HiveMind/MS3/MCP services. Restore service reachability and retry."
        )

    def _depth_preflight_diagnostic_codes(self, report: dict[str, Any]) -> list[str]:
        diagnostics = report.get("diagnostics") if isinstance(report, dict) else None
        if not isinstance(diagnostics, list):
            return []
        codes: list[str] = []
        for diagnostic in diagnostics:
            if isinstance(diagnostic, dict) and diagnostic.get("code"):
                codes.append(str(diagnostic["code"]))
        return codes[:10]

    def _depth_preflight_unreachable_probes(self, report: dict[str, Any]) -> list[str]:
        probes = report.get("probes") if isinstance(report, dict) else None
        if not isinstance(probes, list):
            return []
        names: list[str] = []
        for probe in probes:
            if isinstance(probe, dict) and not bool(probe.get("reachable")):
                names.append(str(probe.get("name") or "unknown"))
        return names[:10]

    def _watch_subprocess(
        self,
        job_id: str,
        proc: subprocess.Popen[bytes],
        cancel: threading.Event,
    ) -> None:
        """Tail the subprocess; on cancel, terminate then kill."""
        checkpoint_interval = self._subprocess_checkpoint_interval(job_id)
        next_checkpoint = time.monotonic() + checkpoint_interval if checkpoint_interval else None
        try:
            while True:
                try:
                    proc.wait(timeout=0.5)
                    break
                except subprocess.TimeoutExpired:
                    pass
                if checkpoint_interval and next_checkpoint is not None:
                    now = time.monotonic()
                    if now >= next_checkpoint:
                        self._emit_subprocess_checkpoint(job_id)
                        next_checkpoint = now + checkpoint_interval
                if cancel.is_set():
                    self._kill_subprocess(job_id, proc)
                    break
                timeout_event = self._post_tool_final_timeout_event(job_id)
                if timeout_event is not None:
                    self._finalize_post_tool_timeout(job_id, proc, timeout_event)
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
                    _cancel_hivemind_trace(
                        job_id,
                        reason=f"MS4 Depth worker exited {proc.returncode}",
                    )
                    # Worker died without finalizing. Mark it failed so the UI
                    # doesn't show a forever-running job.
                    self.blackboard.update_job_state(
                        job_id,
                        state="failed",
                        finished_at=datetime.now(timezone.utc).isoformat(),
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
                self._canceling_jobs.discard(job_id)

    def _subprocess_checkpoint_interval(self, job_id: str) -> int:
        job = self.blackboard.get_job(job_id)
        if job is None or not job.status_policy.emit_progress_events:
            return 0
        return max(1, min(60, int(job.status_policy.summarize_every_seconds)))

    def _emit_subprocess_checkpoint(self, job_id: str) -> None:
        snap = self.blackboard.get_job_snapshot(job_id)
        if not snap or snap.get("state") not in {"queued", "running"}:
            return
        events = self.blackboard.list_events(job_id, limit=1000)
        last = events[-1] if events else {}
        last_type = str(last.get("type") or "")
        if last_type == "job.tool.call.completed":
            tool = ""
            payload = last.get("payload")
            if isinstance(payload, dict):
                tool = str(payload.get("tool") or "")
            suffix = f" ({tool})" if tool else ""
            status = f"Tool completed{suffix}; waiting for the Depth model final answer."
        elif (
            last_type == "job.checkpoint"
            and isinstance(last.get("payload"), dict)
            and last["payload"].get("source") == "worker_stream"
        ):
            status = "Depth model is drafting the final answer."
        else:
            status = "Background work is still running."
        self.blackboard.insert_event(
            JobEvent.make(
                job_id=job_id,
                type="job.checkpoint",
                safe_user_status=status,
                payload={"last_event_type": last_type or None, "source": "parent_watcher"},
            )
        )

    def _post_tool_final_timeout_event(self, job_id: str) -> dict[str, Any] | None:
        timeout = self._post_tool_final_timeout_seconds
        if timeout <= 0:
            return None
        snap = self.blackboard.get_job_snapshot(job_id)
        if not snap or snap.get("state") not in {"queued", "running"}:
            return None
        events = self.blackboard.list_events(job_id, limit=1000)
        tool_event: dict[str, Any] | None = None
        for event in reversed(events):
            if event.get("type") == "job.tool.call.completed":
                tool_event = event
                break
        if tool_event is None:
            return None
        tool_payload = tool_event.get("payload")
        if not isinstance(tool_payload, dict):
            return None
        completed_tool = str(tool_payload.get("tool") or "")
        if not completed_tool.startswith("hivemind_"):
            return None
        tool_ts = self._event_timestamp_seconds(tool_event)
        if tool_ts is None:
            return None
        last_progress_ts = tool_ts
        for event in events:
            event_ts = self._event_timestamp_seconds(event)
            if event_ts is None or event_ts < tool_ts:
                continue
            event_type = event.get("type")
            if event_type in {"job.completed", "job.failed", "job.canceled"}:
                return None
            payload = event.get("payload")
            if event_type == "job.tool.call.started":
                if not isinstance(payload, dict):
                    return None
                started_tool = str(payload.get("tool") or "")
                if started_tool != completed_tool:
                    return None
            if (
                event_type == "job.checkpoint"
                and isinstance(payload, dict)
                and payload.get("source") == "worker_stream"
            ):
                last_progress_ts = max(last_progress_ts, event_ts)
        if time.time() - last_progress_ts < timeout:
            return None
        return tool_event

    def _event_timestamp_seconds(self, event: dict[str, Any]) -> float | None:
        raw = str(event.get("timestamp") or "")
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None

    def _finalize_post_tool_timeout(
        self,
        job_id: str,
        proc: subprocess.Popen[bytes],
        tool_event: dict[str, Any],
    ) -> None:
        payload = tool_event.get("payload")
        tool = ""
        result_excerpt = ""
        if isinstance(payload, dict):
            tool = str(payload.get("tool") or "")
            result_excerpt = str(payload.get("result_excerpt") or "").strip()
        _cancel_hivemind_trace(
            job_id,
            reason="MS4 Depth worker final-answer timeout",
        )
        self._terminate_subprocess_process(job_id, proc)
        snap = self.blackboard.get_job_snapshot(job_id)
        if not snap or snap.get("state") not in {"queued", "running"}:
            return
        envelope = self.blackboard.get_job(job_id)
        revision = (
            envelope.conversation_revision_id
            if envelope is not None
            else int(snap.get("conversation_revision_id") or 1)
        )
        turn_id = envelope.turn_id if envelope is not None else snap.get("turn_id")
        if result_excerpt:
            summary = (
                f"{tool or 'Tool'} completed, but the Depth model final answer timed out. "
                "Showing the verified tool result excerpt."
            )
            result = JobResult(
                job_id=job_id,
                status="success",
                summary=summary,
                text=f"{summary}\n\n{result_excerpt}",
                evidence=[],
                actions_taken=[{"tool": tool or None}],
                next_steps=[
                    "Retry final synthesis with a chat-capable model if a natural-language answer is required."
                ],
                confidence="low",
                conversation_revision_id=revision,
                turn_id=turn_id,
            )
            self.blackboard.insert_result(result)
            self.blackboard.insert_event(
                JobEvent.make(
                    job_id=job_id,
                    type="job.completed",
                    safe_user_status=summary,
                    payload={
                        "reason": "post_tool_final_answer_timeout",
                        "tool": tool or None,
                        "fallback": "tool_result_excerpt",
                    },
                )
            )
            self.blackboard.update_job_state(
                job_id,
                state="completed",
                finished_at=datetime.now(timezone.utc).isoformat(),
                last_safe_user_status=summary,
            )
            return
        reason = (
            f"{tool or 'Tool'} completed, but the Depth model final answer timed out "
            "and no user-safe tool excerpt was available."
        )
        result = JobResult(
            job_id=job_id,
            status="failed",
            summary=reason,
            actions_taken=[{"tool": tool or None}],
            confidence="low",
            conversation_revision_id=revision,
            turn_id=turn_id,
        )
        self.blackboard.insert_result(result)
        self.blackboard.insert_event(
            JobEvent.make(
                job_id=job_id,
                type="job.failed",
                safe_user_status=reason,
                payload={"reason": "post_tool_final_answer_timeout", "tool": tool or None},
            )
        )
        self.blackboard.update_job_state(
            job_id,
            state="failed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            last_safe_user_status=reason,
        )

    def _kill_subprocess(self, job_id: str, proc: subprocess.Popen[bytes]) -> None:
        with self._lock:
            if job_id in self._canceling_jobs:
                return
            self._canceling_jobs.add(job_id)
        try:
            if proc.poll() is not None:
                return

            # Cancel HLI accounting before waiting for the worker to unwind.
            # The old ordering waited the full process grace period first,
            # which let short generations complete naturally and made this
            # DELETE arrive too late to set cancelled_by_operator.
            hli_canceled = _cancel_hivemind_trace(
                job_id,
                reason="MS4 Depth worker canceled",
            )
            snap = self.blackboard.get_job_snapshot(job_id)
            if snap and snap.get("state") in {"queued", "running"}:
                status = "Background work was canceled (upstream abort requested)."
                self.blackboard.insert_event(
                    JobEvent.make(
                        job_id=job_id,
                        type="job.canceled",
                        safe_user_status=status,
                        payload={
                            "source": "parent_runner",
                            "hli_trace_canceled": hli_canceled,
                        },
                    )
                )
                self.blackboard.update_job_state(
                    job_id,
                    state="canceled",
                    finished_at=datetime.now(timezone.utc).isoformat(),
                    last_safe_user_status=status,
                )

            # SIGTERM/CTRL_BREAK flips the child cancel event.  Its cancellation
            # monitor calls Hermes.interrupt(), whose transport closes the
            # worker-local streaming HTTP client.  kill() remains the bounded
            # fallback for providers that do not unwind cooperatively.
            self._terminate_subprocess_process(job_id, proc)
        finally:
            with self._lock:
                self._canceling_jobs.discard(job_id)

    def _terminate_subprocess_process(self, job_id: str, proc: subprocess.Popen[bytes]) -> None:
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

    def get_result(self, job_id: str) -> dict[str, Any] | None:
        """The persisted ``JobResult`` (text/summary/evidence) for a
        finished job, or ``None`` if it hasn't produced one yet."""
        return self.blackboard.get_result(job_id)

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
                status = "Background work was canceled (no live worker; updated state directly)."
                hli_canceled = _cancel_hivemind_trace(
                    job_id,
                    reason="MS4 Depth job canceled without a live worker",
                )
                self.blackboard.insert_event(
                    JobEvent.make(
                        job_id=job_id,
                        type="job.canceled",
                        safe_user_status=status,
                        payload={"source": "parent_runner", "hli_trace_canceled": hli_canceled},
                    )
                )
                self.blackboard.update_job_state(
                    job_id,
                    state="canceled",
                    finished_at=datetime.now(timezone.utc).isoformat(),
                    last_safe_user_status=status,
                )
                return self.blackboard.get_job_snapshot(job_id) or snap
            return snap
        if proc is not None:
            # Tell the watcher loop to kill the child immediately rather than
            # waiting for the next poll tick.
            try:
                self._kill_subprocess(job_id, proc)
            except Exception as exc:
                log.warning("double_agent.runner: cancel kill(%s) failed: %s", job_id, exc)
        if event is not None:
            event.set()
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

    def _has_staleable_jobs(self, conversation_id: str) -> bool:
        """True if the conversation has any queued/running job that a
        revision bump could stale. Used to bound continuation detection
        (no jobs in flight → no decision to make → never spend an LLM
        call)."""
        try:
            jobs = self.blackboard.list_jobs(
                conversation_id=conversation_id,
                states=("queued", "running"),
                limit=1,
            )
        except Exception:
            return False
        return bool(jobs)

    def _detect_continuation(self, message: str, *, has_jobs: bool) -> tuple[bool, str]:
        """Decide whether ``message`` is a continuation of in-flight work.

        Returns ``(is_continuation, method)`` where method is one of
        ``phrase`` / ``classifier`` / ``none``. Two-layer:

        1. Phrase fast-path (zero latency, always on).
        2. Optional injected LLM classifier — consulted ONLY when there
           are staleable jobs, the phrase path missed, and the message
           is short enough to plausibly be a continuation. Fail-safe:
           an unsure/unavailable verdict (``None``) is treated as "not a
           continuation" so behavior matches the pre-classifier runtime.
        """
        if phrase_is_continuation(message):
            return True, "phrase"
        classifier = self._continuation_classifier
        if (
            classifier is not None
            and has_jobs
            and message
            and len(message.strip()) <= CLASSIFIER_CONSULT_MAX_LEN
        ):
            try:
                verdict = classifier(message)
            except Exception as exc:  # noqa: BLE001 — classifier must never break a turn
                log.info("continuation classifier raised (treating as unsure): %s", exc)
                verdict = None
            if verdict is True:
                return True, "classifier"
        return False, "none"

    def bump_revision(
        self,
        conversation_id: str,
        *,
        user_message_excerpt: str = "",
        turn_id: str | None = None,
    ) -> dict[str, Any]:
        revision = self.blackboard.bump_revision(
            conversation_id,
            user_message_excerpt=user_message_excerpt,
            turn_id=turn_id,
        )
        # May 30 2026: auto-staling is OFF by default. Deep jobs now run
        # to completion regardless of subsequent turns, and the system
        # delivers the result when done (UI auto-announce). Only an
        # explicit cancel stops a job. This fixes the live failure mode
        # where asking "is it done?" bumped the revision and killed the
        # very job the operator was waiting on. Set MS4_DA_AUTOSTALE=1 to
        # restore the legacy continuation-aware revision staling below.
        if not _autostale_enabled():
            return {
                "revision": revision.to_dict(),
                "marked_stale": [],
                "continuation_detected": False,
                "autostale": False,
            }
        # ---- Legacy path (MS4_DA_AUTOSTALE=1) ----
        # Continuation-aware staling (May 26 2026; classifier added
        # May 28 2026): when the new user message is a continuation
        # ("do that", "yes", "ok", "any update?", or a paraphrase the
        # phrase list can't enumerate), the user is WAITING for the
        # in-flight background work — staling those jobs is exactly the
        # wrong thing. Only spend a classifier call when there's actually
        # staleable work in flight.
        has_jobs = self._has_staleable_jobs(conversation_id)
        is_continuation, method = self._detect_continuation(
            user_message_excerpt, has_jobs=has_jobs
        )
        if is_continuation:
            return {
                "revision": revision.to_dict(),
                "marked_stale": [],
                "continuation_detected": True,
                "continuation_method": method,
            }
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
            "continuation_detected": False,
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

    def set_continuation_classifier(self, classifier: ContinuationClassifier | None) -> None:
        """Wire (or clear) the optional LLM continuation classifier.

        Production wires this in ``server.run()`` so paraphrased
        continuations don't wrongly stale in-flight work. Tests leave it
        unset, so ``bump_revision`` stays deterministic and offline
        (phrase fast-path only). Setting ``None`` returns to phrase-only."""
        self._continuation_classifier = classifier

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
