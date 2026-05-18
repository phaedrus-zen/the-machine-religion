from __future__ import annotations

import json
import os
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .audit import append_event
from .context import build_grounded_user_message


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
    ) -> None:
        self.hermes_dir = Path(hermes_dir)
        self.hivemind_url = hivemind_url.rstrip("/")
        self.ms3_url = ms3_url.rstrip("/")
        self.default_model = default_model
        self._agent_cls = agent_cls
        self._sessions: dict[str, SessionState] = {}

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
        self.require_plugin()
        os.environ["MS4_MS3_SIDECAR_URL"] = self.ms3_url
        os.environ.setdefault("MS4_SPIRIT_ID", "sister")

        selected_model = model or self.default_model
        state = self.get_or_create_session(session_id, selected_model)
        state.tool_trace = []
        grounded_message, source = build_grounded_user_message(message, self.hivemind_url)
        state.last_grounding_source = source

        run_kwargs = {
            "conversation_history": list(state.history),
            "task_id": state.session_id,
        }
        if stream_callback is not None:
            run_kwargs["stream_callback"] = stream_callback
        result = state.agent.run_conversation(grounded_message, **run_kwargs)
        state.history = result.get("messages", state.history)
        response = {
            "text": result.get("final_response", ""),
            "session_id": state.session_id,
            "hermes_session_id": getattr(state.agent, "session_id", state.session_id),
            "model": selected_model,
            "grounding_source": source,
            "api_calls": result.get("api_calls"),
            "completed": result.get("completed", True),
            "tool_trace": list(state.tool_trace),
            "runtime": "hermes",
            "ms3_sidecar_url": self.ms3_url,
            "hivemind_url": self.hivemind_url,
        }
        append_event("chat_turn", {
            "session_id": state.session_id,
            "model": selected_model,
            "grounding_source": source,
            "tool_trace_count": len(state.tool_trace),
            "completed": response["completed"],
        })
        return response

    def sessions(self) -> list[dict[str, Any]]:
        return [
            {
                "session_id": state.session_id,
                "hermes_session_id": getattr(state.agent, "session_id", state.session_id),
                "turns": len(state.history),
                "model": getattr(state.agent, "model", self.default_model),
                "last_grounding_source": state.last_grounding_source,
                "tool_trace_count": len(state.tool_trace),
            }
            for state in self._sessions.values()
        ]

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
