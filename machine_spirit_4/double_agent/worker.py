"""Background worker for a Double Agent job.

Wraps :class:`machine_spirit_4.gateway.hermes_runner.Ms4HermesRunner.chat`
in a long-running thread, translating Hermes lifecycle callbacks into
allowlisted :class:`JobEvent`\\ s on the blackboard.

Anti-hallucination contract enforced here (artifact §13):

* The Hermes ``stream_delta_callback`` is consumed but tokens are NOT
  emitted as events. Streaming exists only so the worker can take a
  liveness pulse (last-token timestamp) for the periodic
  ``job.checkpoint`` summary.
* ``tool_start_callback`` and ``tool_complete_callback`` produce
  ``job.tool.call.started`` / ``job.tool.call.completed`` events. The
  ``safe_user_status`` for these is derived from the tool name and a
  short fact about its arguments — never from the model's reasoning.
* Cancellation is observed between tool calls (the only natural
  checkpoint inside a Hermes turn that we can interrupt without
  killing the model process). Workers also re-check between successive
  Hermes iterations.
"""

from __future__ import annotations

import json
import threading
import traceback
from datetime import datetime, timezone
from typing import Any, Callable

from .blackboard import Blackboard, default_blackboard
from .schemas import (
    JobEnvelope,
    JobEvent,
    JobResult,
    SchemaError,
)
from . import safety


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class WorkerCanceled(RuntimeError):
    """Raised inside the worker when a cancellation event is observed."""


def _short_args_repr(args: dict[str, Any] | None) -> str:
    if not args:
        return ""
    try:
        rendered = json.dumps(args, ensure_ascii=False, default=str)
    except Exception:
        rendered = str(args)
    if len(rendered) > 80:
        rendered = rendered[:77] + "..."
    return rendered


def _format_tool_status(event_kind: str, tool_name: str, args: dict[str, Any] | None) -> str:
    """Produce a safe_user_status for tool start/complete events.

    Reads only operational facts (tool name + a tiny excerpt of args).
    Never inspects the model's reasoning."""
    arg_blurb = _short_args_repr(args)
    if event_kind == "start":
        if arg_blurb:
            return f"Started tool {tool_name}: {arg_blurb}"
        return f"Started tool {tool_name}"
    if arg_blurb:
        return f"Finished tool {tool_name}: {arg_blurb}"
    return f"Finished tool {tool_name}"


class DoubleAgentWorker:
    """Runs one job to completion (or cancellation/failure).

    The worker depends on a callable that performs the actual deep
    chat. In production this is the existing
    ``Ms4HermesRunner.chat``; in tests we inject a fake to keep the
    suite hermetic and fast.
    """

    def __init__(
        self,
        envelope: JobEnvelope,
        *,
        blackboard: Blackboard,
        cancel_event: threading.Event,
        chat_runner: Callable[..., dict[str, Any]],
    ) -> None:
        self.envelope = envelope
        self.blackboard = blackboard
        self.cancel_event = cancel_event
        self._chat_runner = chat_runner
        self._tool_traces: list[dict[str, Any]] = []
        self._token_count = 0

    # ------- event helpers --------------------------------------------------

    def _emit(
        self,
        event_type: str,
        safe_user_status: str,
        *,
        visibility: str = "user_safe",
        evidence_refs: list[str] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        event = JobEvent.make(
            job_id=self.envelope.job_id,
            type=event_type,
            safe_user_status=safe_user_status,
            visibility=visibility,
            evidence_refs=evidence_refs,
            payload=payload,
        )
        self.blackboard.insert_event(event)

    def _maybe_cancel(self) -> None:
        if self.cancel_event.is_set():
            raise WorkerCanceled()

    # ------- Hermes callbacks ---------------------------------------------

    def _on_tool_start(self, tool_call_id: str, name: str, args: dict[str, Any]) -> None:
        self._maybe_cancel()
        self._tool_traces.append(
            {"event": "start", "tool_call_id": tool_call_id, "tool": name, "args": args}
        )
        if self.envelope.status_policy.emit_tool_events:
            self._emit(
                "job.tool.call.started",
                _format_tool_status("start", name, args),
                evidence_refs=[f"tool_call:{tool_call_id}"],
                payload={"tool": name, "args_excerpt": _short_args_repr(args)},
            )

    def _on_tool_complete(self, tool_call_id: str, name: str, args: dict[str, Any], result: Any) -> None:
        self._maybe_cancel()
        result_excerpt = str(result)
        if len(result_excerpt) > 2000:
            result_excerpt = result_excerpt[:1997] + "..."
        self._tool_traces.append(
            {
                "event": "complete",
                "tool_call_id": tool_call_id,
                "tool": name,
                "args": args,
                "result_excerpt": result_excerpt,
            }
        )
        if self.envelope.status_policy.emit_tool_events:
            self._emit(
                "job.tool.call.completed",
                _format_tool_status("complete", name, args),
                evidence_refs=[f"tool_call:{tool_call_id}"],
                payload={"tool": name},
            )

    def _on_stream_delta(self, _delta: str) -> None:
        # The artifact's anti-hallucination rule: we observe tokens to
        # know we're alive, but we never relay them into events. Just
        # bump a counter for the eventual checkpoint summary.
        self._token_count += 1

    # ------- main loop ----------------------------------------------------

    def run(self) -> JobResult:
        try:
            self.blackboard.update_job_state(
                self.envelope.job_id,
                state="running",
                last_safe_user_status=self.envelope.user_visible_goal,
            )
            self._emit(
                "job.started",
                self.envelope.user_visible_goal or "Background work started.",
            )
            self._maybe_cancel()

            response = self._chat_runner(
                message=self.envelope.internal_goal or self.envelope.user_visible_goal,
                session_id=f"da-worker-{self.envelope.job_id}",
                model=self.envelope.resource_request.model_override,
                stream_callback=self._on_stream_delta,
                tool_start_callback=self._on_tool_start,
                tool_complete_callback=self._on_tool_complete,
            )

            self._maybe_cancel()
            text = str(response.get("text", "")) if isinstance(response, dict) else str(response)
            summary = self._summarize(text)
            actions = [
                {"tool": t.get("tool"), "tool_call_id": t.get("tool_call_id")}
                for t in self._tool_traces
                if t.get("event") == "complete"
            ]
            result = JobResult(
                job_id=self.envelope.job_id,
                status="success",
                summary=summary,
                text=text,
                evidence=[],
                actions_taken=actions,
                next_steps=[],
                confidence="medium" if text else "low",
                conversation_revision_id=self.envelope.conversation_revision_id,
            )
            self.blackboard.insert_result(result)
            # Emit the terminal lifecycle event BEFORE updating state so
            # any consumer polling on state (test harness, UI, etc.)
            # sees the matching event already in the events table when
            # it reacts to the state transition. The failed path below
            # follows the same ordering for the same reason.
            self._emit("job.completed", summary or "Background work completed.")
            self.blackboard.update_job_state(
                self.envelope.job_id,
                state="completed",
                finished_at=_now_iso(),
                last_safe_user_status=summary,
            )
            return result
        except WorkerCanceled:
            self._emit("job.canceled", "Background work was canceled.")
            self.blackboard.update_job_state(
                self.envelope.job_id,
                state="canceled",
                finished_at=_now_iso(),
                last_safe_user_status="Background work was canceled.",
            )
            result = JobResult(
                job_id=self.envelope.job_id,
                status="failed",
                summary="canceled",
                conversation_revision_id=self.envelope.conversation_revision_id,
            )
            try:
                self.blackboard.insert_result(result)
            except SchemaError:
                pass
            return result
        except Exception as exc:
            tb = traceback.format_exc()
            # Operator-only event carries the traceback; user-safe carries the
            # short reason. This split mirrors the artifact's "stream safe
            # operational events, not hidden reasoning" rule even for failures.
            self._emit(
                "job.failed",
                f"Background work failed: {type(exc).__name__}: {exc}",
                visibility="user_safe",
                payload={"error_class": type(exc).__name__},
            )
            self._emit(
                "job.failed",
                "operator detail attached",
                visibility="operator_only",
                payload={"traceback": tb[-4000:]},
            )
            # Result first, then state transition — matches the
            # success/canceled ordering above (the user-safe / operator
            # job.failed events were already emitted before we landed
            # here). The state transition is the last operation a
            # consumer can race on.
            result = JobResult(
                job_id=self.envelope.job_id,
                status="failed",
                summary=f"{type(exc).__name__}: {exc}",
                conversation_revision_id=self.envelope.conversation_revision_id,
            )
            try:
                self.blackboard.insert_result(result)
            except SchemaError:
                pass
            self.blackboard.update_job_state(
                self.envelope.job_id,
                state="failed",
                finished_at=_now_iso(),
                last_safe_user_status=f"Background work failed: {type(exc).__name__}",
            )
            return result

    def _summarize(self, text: str) -> str:
        """Generate a safe summary string for the completed result.

        We deliberately don't ask the model to summarize itself (that
        could leak its reasoning into a user_safe event). Instead, we
        take the first declarative sentence of the final response and
        clamp it. Operators who want the full text can read
        ``JobResult.text``.
        """
        if not text:
            return "Background work completed without a textual answer."
        normalized = " ".join(text.split())
        for sep in (". ", "? ", "! ", "\n"):
            if sep in normalized:
                first = normalized.split(sep, 1)[0]
                if 12 <= len(first) <= safety.SAFE_USER_STATUS_MAX_CHARS:
                    return safety.coerce_safe_user_status(first + sep.strip())
        return safety.coerce_safe_user_status(normalized)


def build_real_chat_runner(runner: Any) -> Callable[..., dict[str, Any]]:
    """Adapter so the worker can call ``Ms4HermesRunner.chat`` without
    importing it directly (keeps the worker test-friendly).

    The default :class:`Ms4HermesRunner.chat` doesn't accept the worker's
    tool callbacks — those callbacks are owned by the session that was
    constructed at session-create time. To keep phase 1 simple and
    avoid a Hermes-runner refactor today, we build a dedicated session
    per job whose callbacks we own end-to-end."""

    def _call(*, message: str, session_id: str, model: str | None,
              stream_callback: Callable[[str], None],
              tool_start_callback: Callable[[str, str, dict[str, Any]], None],
              tool_complete_callback: Callable[[str, str, dict[str, Any], Any], None]) -> dict[str, Any]:
        # Force a fresh session whose lifecycle callbacks we control.
        runner.ensure_hermes_path()
        runner.require_plugin()
        agent = runner._new_agent.__self__._agent_class()  # type: ignore[attr-defined]
        agent_instance = agent(
            base_url=f"{runner.hivemind_url}/v1",
            api_key=runner.default_model and "local-not-needed",
            provider="custom",
            api_mode="chat_completions",
            model=model or runner.default_model,
            session_id=session_id,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            platform="ms4",
            max_iterations=12,
            tool_start_callback=tool_start_callback,
            tool_complete_callback=tool_complete_callback,
        )
        kwargs: dict[str, Any] = {"conversation_history": [], "task_id": session_id}
        if stream_callback is not None:
            kwargs["stream_callback"] = stream_callback
        result = agent_instance.run_conversation(message, **kwargs)
        return {
            "text": result.get("final_response", ""),
            "session_id": session_id,
            "model": model or runner.default_model,
            "completed": result.get("completed", True),
        }

    return _call
