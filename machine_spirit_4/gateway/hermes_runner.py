from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


log = logging.getLogger("ms4.gateway.hermes_runner")


def _try_cluster_time_now(hivemind_url: str) -> str | None:
    """Best-effort cluster-time lookup via ``hivemind.time.now@v1``.

    Returns the cluster's ISO timestamp string when reachable, ``None``
    on any failure (offline cluster, missing tool, malformed response).
    Capped at 2s so it can't delay a chat turn.
    """
    try:
        from . import hivemind_tools

        body = hivemind_tools.time_now(hivemind_url)
    except Exception:
        return None
    if isinstance(body, dict):
        for key in ("iso", "iso8601", "now", "datetime", "time"):
            value = body.get(key)
            if isinstance(value, str) and value:
                return value
    if isinstance(body, str):
        return body
    return None


def _now_local_iso() -> str:
    """Return the current local date/time as ISO-8601 with timezone.

    Used to ground the Face Lobe on every turn so questions like
    "what is the date?" don't require a Depth Lobe dispatch. Local
    time (with tzinfo) is what the operator actually wants to see.
    """
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _canned_chat_failure_text(exc: Exception) -> str:
    """Translate a FaceLobeChat exception into a short user-visible
    reply that's safe to feed into TTS. The raw exception string
    (full URL, timeout numbers, etc.) is operator detail and goes to
    metrics.error / the audit log, not the user.

    Tailored to the three failure modes seen live:
      - HiveMind unreachable
      - Streaming chat completion stalled / timed out
      - Server returned a non-2xx (5xx)
    """
    msg = str(exc).lower()
    if "stream" in msg and ("timed out" in msg or "unreachable" in msg):
        return (
            "I'm having trouble reaching the language cluster right now. "
            "Give it a moment and try again, or use the Settings dialog "
            "to switch the TTS engine to REST if HiveMind's streaming "
            "endpoint is unhappy."
        )
    if "unreachable" in msg or "connection refused" in msg or "name or service" in msg:
        return (
            "HiveMind looks unreachable from MS4 right now. Check that "
            "the cluster is running on the configured URL and try again."
        )
    if "5" in msg and ("502" in msg or "503" in msg or "500" in msg):
        return (
            "HiveMind returned a server error on the chat completion. "
            "It's usually transient — try the same prompt again in a "
            "few seconds."
        )
    # Catch-all canned text. The operator can dig the real exception out
    # of metrics.error.
    return (
        "Something went wrong reaching the language model. The operator "
        "can check the audit log for the exact error."
    )


def _combine_grounding(
    grounding_source: str | None,
    face_lobe_block: str | None,
    dispatched_job: dict[str, Any] | None,
) -> str:
    """Combine the grounding-source labels so the operator can see at a
    glance what context layers were applied to a turn."""
    parts: list[str] = []
    if grounding_source:
        parts.append(grounding_source)
    if face_lobe_block:
        parts.append("face_lobe")
    if dispatched_job and isinstance(dispatched_job, dict) and dispatched_job.get("job_id"):
        parts.append("depth_lobe_dispatched")
    return "+".join(parts) if parts else "none"

from machine_spirit_4.double_agent import (
    JobEnvelope,
    ResourceRequest,
    SchemaError,
    build_face_lobe_context_block,
    choose_depth_model,
    choose_foreground_model,
    default_runner,
    face_lobe_turn_start,
    router_route,
)
from machine_spirit_4.double_agent.safety import new_job_id

from .audit import append_event
from .context import build_grounded_user_message
from .face_lobe_chat import FaceLobeChat, FaceLobeChatError


@dataclass
class SessionState:
    session_id: str
    agent: Any
    history: list[dict[str, Any]] = field(default_factory=list)
    last_grounding_source: str | None = None
    tool_trace: list[dict[str, Any]] = field(default_factory=list)


class HermesUnavailable(RuntimeError):
    pass


class Ms4HermesRunner:
    def __init__(
        self,
        *,
        hermes_dir: str,
        hivemind_url: str = "http://127.0.0.1:6089",
        ms3_url: str = "http://127.0.0.1:9080",
        default_model: str = "qwen3-coder-next:latest",
        agent_cls: Any | None = None,
        face_lobe_chat: FaceLobeChat | None = None,
    ) -> None:
        self.hermes_dir = Path(hermes_dir)
        self.hivemind_url = hivemind_url.rstrip("/")
        self.ms3_url = ms3_url.rstrip("/")
        # Cache the cluster-time probe so we don't pay an MCP round
        # trip on every chat turn. The probe is best-effort + capped
        # at ~2s; we cache successful results for 30s and negative
        # results for 5s so a transiently-offline cluster doesn't
        # force the slow path on every turn.
        self._cluster_time_cache: tuple[float, str | None] = (0.0, None)
        self._cluster_time_lock = threading.Lock()
        self.default_model = default_model
        self._agent_cls = agent_cls
        self._sessions: dict[str, SessionState] = {}
        # Face Lobe direct-chat path: per artifact §5.1/§16.5 the foreground
        # is "status and routing focused" and should NOT pay the Hermes
        # tool-loop tax on every turn. We keep the Hermes plumbing for
        # `dispatch_hermes_tool` and Depth Lobe workers.
        self.face_lobe_chat = face_lobe_chat or FaceLobeChat(hivemind_url=self.hivemind_url)

    def ensure_hermes_path(self) -> None:
        hermes_path = str(self.hermes_dir)
        if hermes_path not in sys.path:
            sys.path.insert(0, hermes_path)

    def plugin_status(self) -> dict[str, Any]:
        self.ensure_hermes_path()
        try:
            from hermes_cli.plugins import PluginManager

            manager = PluginManager()
            manager.discover_and_load()
            loaded = manager._plugins.get("ms4_consciousness")
            hooks = sorted(manager._hooks.keys())
            return {
                "found": loaded is not None,
                "enabled": bool(loaded and loaded.enabled),
                "hooks": hooks,
            }
        except Exception as exc:
            return {"found": False, "enabled": False, "hooks": [], "error": str(exc)}

    def require_plugin(self) -> None:
        status = self.plugin_status()
        if not status.get("enabled"):
            raise HermesUnavailable(f"ms4_consciousness plugin is not enabled: {status}")

    def _agent_class(self):
        if self._agent_cls is not None:
            return self._agent_cls
        self.ensure_hermes_path()
        from run_agent import AIAgent

        return AIAgent

    def _new_agent(self, session_id: str, model: str):
        agent_cls = self._agent_class()
        state_ref = self._sessions.get(session_id)

        def on_tool_start(tool_call_id: str, name: str, args: dict[str, Any]) -> None:
            state = self._sessions.get(session_id) or state_ref
            if state is not None:
                state.tool_trace.append({"event": "start", "tool_call_id": tool_call_id, "tool": name, "args": args})

        def on_tool_complete(tool_call_id: str, name: str, args: dict[str, Any], result: str) -> None:
            state = self._sessions.get(session_id) or state_ref
            if state is not None:
                state.tool_trace.append({
                    "event": "complete",
                    "tool_call_id": tool_call_id,
                    "tool": name,
                    "args": args,
                    "result_excerpt": str(result)[:2000],
                })

        return agent_cls(
            base_url=f"{self.hivemind_url}/v1",
            api_key=os.environ.get("MS4_HIVEMIND_API_KEY", "local-not-needed"),
            provider="custom",
            api_mode="chat_completions",
            model=model,
            session_id=session_id,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            platform="ms4",
            max_iterations=12,
            tool_start_callback=on_tool_start,
            tool_complete_callback=on_tool_complete,
        )

    def get_or_create_session(self, session_id: str | None, model: str) -> SessionState:
        sid = session_id or f"ms4-{uuid.uuid4()}"
        state = self._sessions.get(sid)
        if state is None:
            state = SessionState(session_id=sid, agent=self._new_agent(sid, model))
            self._sessions[sid] = state
        elif getattr(state.agent, "model", model) != model:
            state.agent = self._new_agent(sid, model)
            state.history = []
            state.tool_trace = []
        return state

    def chat(
        self,
        message: str,
        *,
        session_id: str | None = None,
        model: str | None = None,
        stream_callback: Any | None = None,
    ) -> dict[str, Any]:
        """Foreground (Face Lobe) chat turn.

        Routes ALL foreground turns through the direct FaceLobeChat path
        (no Hermes loop, no plugin chain, no tool injection) so even
        trivial turns return in seconds instead of tens of seconds.
        Tool-requiring work is dispatched in parallel to a Depth Lobe
        background job by the auto-router; the foreground reply mentions
        the dispatch but doesn't wait on it.
        """
        os.environ["MS4_MS3_SIDECAR_URL"] = self.ms3_url
        os.environ.setdefault("MS4_SPIRIT_ID", "sister")

        # ---- Face Lobe model resolution (precedence: per-turn > session-pinned > picker > default)
        face_lobe_model: dict[str, Any]
        existing_face = self.face_lobe_chat._sessions.get(session_id) if session_id else None
        if model is not None:
            selected_model = model
            face_lobe_model = {
                "schema": "Ms4ForegroundModel.v1",
                "model_id": selected_model,
                "source": "per_turn_override",
                "detail": "request specified model",
            }
        elif existing_face is not None:
            selected_model = existing_face.model or self.default_model
            face_lobe_model = {
                "schema": "Ms4ForegroundModel.v1",
                "model_id": selected_model,
                "source": "session_pinned",
                "detail": "model pinned on session create; not re-picked mid-conversation",
            }
        else:
            try:
                choice = choose_foreground_model(hivemind_url=self.hivemind_url)
                selected_model = choice.model_id
                face_lobe_model = choice.to_dict()
            except Exception as exc:
                selected_model = self.default_model
                face_lobe_model = {
                    "schema": "Ms4ForegroundModel.v1",
                    "model_id": selected_model,
                    "source": "error",
                    "detail": str(exc),
                }

        # ---- Auto-router decision (and revision bump + stale-older-jobs)
        route_decision = router_route(message)
        message_for_chat = route_decision.cleaned_message or message
        dispatched_job: dict[str, Any] | None = None
        depth_choice_dict: dict[str, Any] | None = None

        face_lobe_outcome: dict[str, Any] = {}
        face_lobe_block: str | None = None
        # We need to know dispatched_job before we build the context
        # block (so the block can include the THIS TURN DISPATCHED /
        # DID NOT DISPATCH line), so handle revision bump first, then
        # auto-router dispatch, then build the block.
        try:
            face_lobe_outcome = face_lobe_turn_start(
                conversation_id=session_id or f"ms4-{uuid.uuid4()}",
                user_message=message_for_chat,
            )
        except Exception as exc:
            face_lobe_outcome = {"error": str(exc)}

        if route_decision.kind == "deep":
            try:
                revision_id = 1
                rev_dict = face_lobe_outcome.get("revision") if isinstance(face_lobe_outcome.get("revision"), dict) else None
                if rev_dict and isinstance(rev_dict.get("revision_id"), int):
                    revision_id = int(rev_dict["revision_id"])
                depth_choice = choose_depth_model(hivemind_url=self.hivemind_url)
                depth_choice_dict = depth_choice.to_dict()
                envelope = JobEnvelope(
                    job_id=new_job_id(),
                    parent_conversation_id=session_id or "ms4-conv",
                    conversation_revision_id=revision_id,
                    background_lobe_type="deep_chat",
                    user_visible_goal=route_decision.goal or message_for_chat[:240],
                    internal_goal=message_for_chat,
                    resource_request=ResourceRequest(model_override=depth_choice.model_id),
                )
                envelope.validate()
                dispatched_job = default_runner().submit(envelope)
            except SchemaError as exc:
                dispatched_job = {"error": f"router refused to dispatch: {exc}"}
            except Exception as exc:
                dispatched_job = {"error": f"router dispatch failed: {exc}"}

        # Current date/time as authoritative grounding so simple
        # questions like "what is the date?" don't need a dispatch.
        # Prefer the HiveMind cluster's clock when reachable so
        # operators running MS4 in a different timezone don't get
        # answers that disagree with the cluster's own logs/job
        # timestamps. Local time stays as the fail-soft fallback.
        now_local = _now_local_iso()
        cluster_time = self._try_cluster_time()
        if cluster_time:
            date_grounding_line = (
                f"current cluster date/time: {cluster_time} "
                f"(local: {now_local})"
            )
        else:
            date_grounding_line = f"current local date/time: {now_local}"
        dispatched_job_id = (
            dispatched_job["job_id"]
            if isinstance(dispatched_job, dict) and dispatched_job.get("job_id")
            else None
        )
        try:
            face_lobe_block = build_face_lobe_context_block(
                conversation_id=session_id or "",
                dispatched_this_turn_job_id=dispatched_job_id,
                extra_authoritative_lines=[date_grounding_line],
            )
        except Exception as exc:
            log.warning("face_lobe context block failed: %s", exc)
            face_lobe_block = (
                "MS4 Face Lobe context (authoritative):\n"
                f"- {date_grounding_line}\n"
                + (
                    f"- THIS TURN DISPATCHED job {dispatched_job_id} to the Depth Lobe.\n"
                    if dispatched_job_id
                    else "- THIS TURN DID NOT DISPATCH any background work.\n"
                )
            )

        # ---- Grounded user message (TMR canon, inventory, tools-list, etc.)
        grounded_message, grounding_source = build_grounded_user_message(message_for_chat, self.hivemind_url)
        extra_system_blocks: list[str] = []
        if face_lobe_block:
            extra_system_blocks.append(face_lobe_block)
        if grounded_message != message_for_chat:
            extra_system_blocks.append(grounded_message.split("\n\nUser request:", 1)[0])
        extra_system = "\n\n".join(extra_system_blocks) if extra_system_blocks else None

        # ---- Direct chat call (the actual latency-sensitive bit)
        completed = True
        api_calls: int | None = 1
        metrics: dict[str, Any] | None = None
        face_result: dict[str, Any] = {}
        effective_model = selected_model
        fallback_used = False
        wall_start = time.monotonic()
        try:
            face_result = self.face_lobe_chat.chat(
                message_for_chat,
                session_id=session_id,
                model=selected_model,
                stream_callback=stream_callback,
                extra_system=extra_system,
            )
            text = face_result["text"]
            chat_session_id = face_result["session_id"]
            effective_model = face_result.get("model") or selected_model
            fallback_used = bool(face_result.get("fallback_used"))
            api_calls = int(face_result.get("api_calls") or 1)
            metrics = face_result.get("metrics")
        except FaceLobeChatError as exc:
            # User-visible canned reply: the raw "HiveMind /v1/chat/completions
            # (stream) unreachable: timed out" is ugly and meaningless to a
            # voice user. Translate it into something the operator can act on
            # AND something the TTS pipeline can synthesize into a sensible
            # audible reply. The raw exception goes into metrics.error for
            # debug + the audit log.
            text = _canned_chat_failure_text(exc)
            log.warning("Face Lobe chat call failed: %s", exc)
            # Stream the canned text to the caller's stream_callback so the
            # WS_SUPER TTS path still gets audio to synthesize and the user
            # actually hears the apology instead of silence.
            if stream_callback is not None:
                try:
                    stream_callback(text)
                except Exception:
                    pass
            completed = False
            api_calls = 0
            chat_session_id = session_id or "ms4-unknown"
            metrics = {
                "schema": "Ms4TurnMetrics.v1",
                "duration_ms": int((time.monotonic() - wall_start) * 1000),
                "error": str(exc),
                "requested_model": selected_model,
                "streaming": stream_callback is not None,
                "canned_reply": True,
            }

        response = {
            "text": text,
            "session_id": chat_session_id,
            "hermes_session_id": chat_session_id,  # kept for backward compat with old UI fields
            "model": effective_model,
            "grounding_source": _combine_grounding(grounding_source, face_lobe_block, dispatched_job),
            "api_calls": api_calls,
            "completed": completed,
            "tool_trace": [],  # Face Lobe doesn't run tools; Depth Lobe does
            "runtime": "face-lobe-direct",
            "ms3_sidecar_url": self.ms3_url,
            "hivemind_url": self.hivemind_url,
            "face_lobe": face_lobe_outcome,
            "face_lobe_model": face_lobe_model,
            "face_lobe_context_block": face_lobe_block,
            "router": route_decision.to_dict(),
            "dispatched_job": dispatched_job,
            "depth_lobe_model": depth_choice_dict,
            "fallback_used": fallback_used,
            "requested_model": selected_model if fallback_used else None,
            "metrics": metrics,
        }
        append_event("chat_turn", {
            "session_id": chat_session_id,
            "model": effective_model,
            "requested_model": selected_model,
            "fallback_used": fallback_used,
            "grounding_source": response["grounding_source"],
            "tool_trace_count": 0,
            "completed": completed,
            "runtime": "face-lobe-direct",
            "router_kind": route_decision.kind,
            "dispatched_job_id": dispatched_job_id,
            "revision_id": face_lobe_outcome.get("revision", {}).get("revision_id") if isinstance(face_lobe_outcome.get("revision"), dict) else None,
            "marked_stale_jobs": face_lobe_outcome.get("marked_stale", []),
            "metrics": metrics,
        })
        return response

    def _try_cluster_time(self) -> str | None:
        """Cached, fail-soft wrapper around ``hivemind.time.now@v1``.

        Successful probes are cached for 30 s; failures for 5 s so a
        transiently-offline cluster forces only one slow probe per
        ~5 s window. The probe itself is capped at ~2 s by
        :func:`_try_cluster_time_now` and the underlying MCP timeout.
        """
        now = time.time()
        with self._cluster_time_lock:
            cached_at, cached_value = self._cluster_time_cache
            age = now - cached_at
            if cached_value is not None and age < 30.0:
                return cached_value
            if cached_value is None and age < 5.0:
                return None
        value = _try_cluster_time_now(self.hivemind_url)
        with self._cluster_time_lock:
            self._cluster_time_cache = (time.time(), value)
        return value

    def sessions(self) -> list[dict[str, Any]]:
        # Foreground sessions live in face_lobe_chat now; Hermes-backed
        # ad-hoc tool sessions (from dispatch_hermes_tool) live in
        # self._sessions for backward compatibility. List both with a
        # `runtime` discriminator.
        out: list[dict[str, Any]] = []
        for state in self.face_lobe_chat.sessions():
            out.append({
                "session_id": state["session_id"],
                "hermes_session_id": state["session_id"],
                "turns": state["turns"],
                "model": state["model"],
                "last_grounding_source": state["last_grounding_source"],
                "tool_trace_count": 0,
                "runtime": "face-lobe-direct",
            })
        for state in self._sessions.values():
            out.append({
                "session_id": state.session_id,
                "hermes_session_id": getattr(state.agent, "session_id", state.session_id),
                "turns": len(state.history),
                "model": getattr(state.agent, "model", self.default_model),
                "last_grounding_source": state.last_grounding_source,
                "tool_trace_count": len(state.tool_trace),
                "runtime": "hermes-tool",
            })
        return out

    def health(self) -> dict[str, Any]:
        return {
            "runtime": "ms4-fusion",
            "hermes_dir": str(self.hermes_dir),
            "hivemind_url": self.hivemind_url,
            "ms3_url": self.ms3_url,
            "plugin": self.plugin_status(),
            "sessions": len(self._sessions),
        }

    def list_hermes_tools(self, enabled_toolsets: list[str] | None = None) -> list[dict[str, Any]]:
        self.ensure_hermes_path()
        from model_tools import get_tool_definitions, get_toolset_for_tool

        definitions = get_tool_definitions(enabled_toolsets=enabled_toolsets, quiet_mode=True)
        tools = []
        for definition in definitions:
            function = definition.get("function", {})
            name = function.get("name")
            if not name:
                continue
            tools.append({
                "name": name,
                "description": function.get("description", ""),
                "toolset": get_toolset_for_tool(name),
                "schema": function,
            })
        return tools

    def dispatch_hermes_tool(
        self,
        tool_name: str,
        args: dict[str, Any] | None = None,
        *,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        self.require_plugin()
        os.environ["MS4_MS3_SIDECAR_URL"] = self.ms3_url
        os.environ.setdefault("MS4_SPIRIT_ID", "sister")
        self.ensure_hermes_path()
        from model_tools import handle_function_call, get_toolset_for_tool

        tool_args = dict(args or {})
        if tool_name == "terminal" and not tool_args.get("workdir"):
            tool_args["workdir"] = os.environ.get("MS4_HERMES_TERMINAL_WORKDIR", "/")
        sid = session_id or f"ms4-tool-{uuid.uuid4()}"
        result = handle_function_call(
            tool_name,
            tool_args,
            task_id=sid,
            session_id=sid,
            tool_call_id=f"ms4-direct-{uuid.uuid4()}",
        )
        response = {
            "tool": tool_name,
            "toolset": get_toolset_for_tool(tool_name),
            "args": tool_args,
            "result": result,
            "session_id": sid,
            "runtime": "hermes",
        }
        append_event("hermes_tool_call", {
            "session_id": sid,
            "tool": tool_name,
            "toolset": response["toolset"],
            "args": tool_args,
            "result_excerpt": str(result)[:1000],
        })
        return response
