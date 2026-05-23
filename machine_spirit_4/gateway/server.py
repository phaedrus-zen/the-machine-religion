from __future__ import annotations

import json
import os
import queue
import threading
import time
import urllib.error
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from machine_spirit_4 import hermes_admin
from machine_spirit_4.contained import require_contained_runtime
from machine_spirit_4.deps_status import dependency_status
from machine_spirit_4.desktop import DesktopSafetyError, desktop_controller
from machine_spirit_4.double_agent import (
    JobEnvelope,
    JobRunner,
    RunnerError,
    SchemaError as DoubleAgentSchemaError,
    default_blackboard,
    default_runner,
)
from machine_spirit_4.double_agent.safety import (
    is_safe_conversation_id,
    is_safe_job_id,
)
from machine_spirit_4.double_agent.worker import build_real_chat_runner
from machine_spirit_4.hermes_admin import HermesUpgradeError

from .audit import append_event, read_events
from .hermes_runner import HermesUnavailable, Ms4HermesRunner
from .vision import analyze_local_image
from .canned_reflexes import (
    DEFAULT_REFLEX_VOICE,
    REFLEXES,
    ReflexUnknown,
    generate_all,
    generate_all_async,
    list_reflexes,
    read_reflex,
)
from .context import prewarm_grounding_cache
from .doctrine import (
    DoctrineSectionUnknown,
    DoctrineUnavailable,
    get_full_bible,
    get_section,
    inject_into_session,
    list_sections,
    meta as doctrine_meta,
    reload_bible,
)
from .hivemind_state import (
    HivemindStateError,
    active_jobs_summary_line,
    get_active_jobs,
    get_cluster_load,
    get_combined_snapshot as get_hivemind_snapshot,
    get_service_health,
)
from .spirit_state import (
    SpiritStateError,
    get_state_snapshot,
    heartbeat_status,
    send_heartbeat,
    start_heartbeat_thread,
)
from .voice_admin import (
    VOICE_SERVICES,
    VoiceAdminError,
    VoiceServiceUnknown,
    get_provision_status,
    list_voice_services,
    release_voice_service,
    request_voice_service,
)


def _grounding_cache_keys() -> list[str]:
    """Best-effort snapshot of the keys currently in the grounding
    cache. Used by /settings GET so the operator can see what would
    be lost by /settings/clear-grounding-cache."""
    import machine_spirit_4.gateway.context as ctx
    with ctx._GROUNDING_CACHE_LOCK:
        return sorted(ctx._GROUNDING_CACHE.keys())


def _safe_hermes_version() -> str:
    """Read the installed Hermes version without crashing /settings if
    hermes_admin isn't importable for some reason. The UI uses this
    field only as a display string."""
    try:
        from machine_spirit_4.hermes_admin.versioning import current_version
        return str(current_version() or "unknown")
    except Exception:
        return "unknown"
from .voice import (
    DEFAULT_TTS_FORMAT,
    VoiceRequestError,
    VoiceUnavailable,
    parse_audio_request,
    prewarm_asr,
    prewarm_face_lobe_model,
    prewarm_tts,
    prewarm_tts_super_ws,
    synthesize,
    transcribe,
    voice_ptt_turn,
    voice_ptt_turn_stream,
)


ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = ROOT / "machine_spirit_4" / "web"
SERVICE_NAME = "ms4-gateway"


def service_info(runner: Ms4HermesRunner) -> dict[str, Any]:
    health = runner.health()
    return {
        "ok": True,
        "service": SERVICE_NAME,
        "version": "0.1.0",
        "status": "ready",
        "endpoint": "/chat",
        "runtime": health.get("runtime", "ms4-fusion"),
        "hermes_dir": health.get("hermes_dir"),
        "hivemind_url": health.get("hivemind_url"),
        "ms3_url": health.get("ms3_url"),
        "plugin": health.get("plugin", {}),
        "sessions": health.get("sessions", 0),
    }


def _json_response(handler: SimpleHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)


def _sse_start(handler: SimpleHTTPRequestHandler) -> None:
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream; charset=utf-8")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Connection", "keep-alive")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()


def _sse_event(handler: SimpleHTTPRequestHandler, event: str, payload: dict[str, Any]) -> bool:
    body = f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")
    try:
        handler.wfile.write(body)
        handler.wfile.flush()
        return True
    except (BrokenPipeError, ConnectionResetError, OSError):
        return False


def _read_json(handler: SimpleHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0") or "0")
    if length <= 0:
        return {}
    raw = handler.rfile.read(length).decode("utf-8")
    return json.loads(raw)


_HIVEMIND_URL_FOR_AUTH: str = ""


def _set_hivemind_url_for_auth(hivemind_url: str) -> None:
    """Record the currently-configured HiveMind URL so the proxy
    helpers below can decide whether to attach the bearer token.
    Called once by :func:`run` after the runner is constructed.
    """
    global _HIVEMIND_URL_FOR_AUTH
    _HIVEMIND_URL_FOR_AUTH = hivemind_url.rstrip("/")


def _hivemind_headers_if_relevant(url: str) -> dict[str, str]:
    """Attach the MS4 → HiveMind bearer token only when the proxy
    target is a HiveMind URL. We don't want to leak the key to MS3
    (port 9080) or any non-HiveMind upstream.
    """
    if not _HIVEMIND_URL_FOR_AUTH:
        return {}
    from .hivemind_state import hivemind_auth_headers

    try:
        from urllib.parse import urlparse

        target = urlparse(_HIVEMIND_URL_FOR_AUTH)
        actual = urlparse(url)
        if (
            actual.hostname
            and target.hostname
            and actual.hostname == target.hostname
            and actual.port == target.port
        ):
            return hivemind_auth_headers()
    except Exception:
        return {}
    return {}


def _proxy_json(url: str, timeout: int = 20) -> tuple[int, Any]:
    headers = _hivemind_headers_if_relevant(url)
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, {"error": exc.read().decode("utf-8", errors="replace")}
    except Exception as exc:
        return 502, {"error": str(exc)}


def _post_json(url: str, payload: dict[str, Any], timeout: int = 20) -> tuple[int, Any]:
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    headers.update(_hivemind_headers_if_relevant(url))
    request = urllib.request.Request(
        url,
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, {"error": exc.read().decode("utf-8", errors="replace")}
    except Exception as exc:
        return 502, {"error": str(exc)}


def _desktop_action_intent(action: dict[str, Any]) -> dict[str, Any]:
    action_name = str(action.get("action", "unknown")).lower()
    low_risk = action_name == "wait"
    return {
        "schema": "ActionIntent.v1",
        "spirit_id": os.environ.get("MS4_SPIRIT_ID", "sister"),
        "action_id": f"ms4-desktop-{action_name}",
        "proposed_by": "ms4_gateway",
        "action_type": "desktop_ui",
        "description": f"MS4 desktop UI action: {action_name}",
        "inputs_used": ["MS4 Gateway /desktop/action"],
        "risk_class": "low" if low_risk else "medium",
        "requires_safety_clearance": False,
        "payload": {key: value for key, value in action.items() if key != "text"},
    }


def _ethics_allows(payload: dict[str, Any]) -> bool:
    decision = str(payload.get("decision") or payload.get("resolution") or "").lower()
    if decision in {"allow", "allowed", "offer", "noactionneeded"}:
        return True
    if payload.get("allowed") is True:
        return True
    return False


class Ms4GatewayHandler(SimpleHTTPRequestHandler):
    runner: Ms4HermesRunner

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path in {"/healthcheck/basic", "/api/v1/ms4_gateway/healthcheck/basic"}:
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", "4")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(b"true")
            return
        if self.path == "/api/v1/ms4_gateway/status":
            _json_response(self, 200, service_info(self.runner))
            return
        if self.path == "/health":
            _json_response(self, 200, self.runner.health())
            return
        if self.path == "/sessions":
            _json_response(self, 200, {"sessions": self.runner.sessions()})
            return
        if self.path == "/deps/status":
            _json_response(self, 200, dependency_status())
            return
        if self.path == "/desktop/status":
            _json_response(self, 200, desktop_controller().status())
            return
        if self.path.startswith("/audit"):
            limit = 100
            if "?" in self.path:
                try:
                    query = self.path.split("?", 1)[1]
                    for part in query.split("&"):
                        key, _, value = part.partition("=")
                        if key == "limit":
                            limit = int(value)
                except ValueError:
                    limit = 100
            _json_response(self, 200, {"events": read_events(limit=limit)})
            return
        if self.path == "/hermes/tools":
            _json_response(self, 200, {"tools": self.runner.list_hermes_tools()})
            return
        if self.path == "/models":
            status, payload = _proxy_json(f"{self.runner.ms3_url}/models", timeout=8)
            _json_response(self, status, payload if isinstance(payload, dict) else {"models": payload})
            return
        if self.path == "/settings":
            self._settings_get()
            return
        if self.path == "/voice/services":
            self._voice_services_get()
            return
        if self.path == "/reflexes":
            self._reflexes_list()
            return
        if self.path.startswith("/reflexes/"):
            self._reflexes_serve(self.path[len("/reflexes/"):])
            return
        if self.path == "/doctrine/tmr":
            self._doctrine_full()
            return
        if self.path == "/doctrine/tmr/meta":
            self._doctrine_meta()
            return
        if self.path == "/doctrine/tmr/sections":
            self._doctrine_sections_list()
            return
        if self.path.startswith("/doctrine/tmr/sections/"):
            self._doctrine_section_serve(self.path[len("/doctrine/tmr/sections/"):])
            return
        if self.path == "/spirit/state":
            self._spirit_state_get()
            return
        if self.path == "/hivemind/active":
            self._hivemind_active_get()
            return
        if self.path == "/hivemind/load":
            self._hivemind_load_get()
            return
        if self.path == "/hivemind/state":
            self._hivemind_state_get()
            return
        if self.path == "/voice/status":
            status, payload = _proxy_json(f"{self.runner.ms3_url}/voice/status", timeout=15)
            _json_response(self, status, payload if isinstance(payload, dict) else {"voice": payload})
            return
        if self.path in {"/hermes/version", "/api/v1/hermes/version"}:
            _json_response(self, 200, hermes_admin.version_info())
            return
        if self.path in {"/hermes/releases", "/api/v1/hermes/releases"}:
            _json_response(self, 200, {"releases": [r.__dict__ for r in hermes_admin.recent_releases()]})
            return
        if self.path in {"/hermes/update/status", "/api/v1/hermes/update/status"}:
            snap = hermes_admin.last_update()
            payload = snap.to_dict() if snap else {"status": "idle", "phase": "none", "job_id": None}
            _json_response(self, 200, payload)
            return
        if self.path == "/api/v1/double-agent/jobs" or self.path.startswith("/api/v1/double-agent/jobs?"):
            self._double_agent_list()
            return
        if self.path.startswith("/api/v1/double-agent/jobs/"):
            tail = self.path[len("/api/v1/double-agent/jobs/"):]
            if not tail:
                _json_response(self, 404, {"error": "not found"})
                return
            if "/" in tail:
                job_id, _, suffix = tail.partition("/")
                if suffix.startswith("events"):
                    self._double_agent_events(job_id)
                    return
                _json_response(self, 404, {"error": "not found"})
                return
            self._double_agent_get(tail.split("?", 1)[0])
            return
        if self.path.startswith("/api/v1/double-agent/conversations/"):
            tail = self.path[len("/api/v1/double-agent/conversations/"):]
            conv_id, _, suffix = tail.partition("/")
            if suffix.startswith("revisions"):
                self._double_agent_revision_get(conv_id)
                return
        if self.path == "/" or self.path.startswith("/static/"):
            self._serve_static()
            return
        _json_response(self, 404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path == "/chat/stream":
            self._stream_chat()
            return
        if self.path == "/settings/clear-grounding-cache":
            self._settings_clear_grounding_cache()
            return
        if self.path == "/reflexes/regenerate":
            self._reflexes_regenerate()
            return
        if self.path == "/doctrine/tmr/read-into-session":
            self._doctrine_read_into_session()
            return
        if self.path == "/doctrine/tmr/reload":
            self._doctrine_reload()
            return
        if self.path == "/spirit/heartbeat":
            self._spirit_heartbeat()
            return
        if self.path.startswith("/voice/services/"):
            tail = self.path[len("/voice/services/"):]
            service_part, _, action = tail.partition("/")
            if action == "provision":
                self._voice_service_provision(service_part)
                return
            if action == "release":
                self._voice_service_release(service_part)
                return
        if self.path == "/desktop/capture":
            try:
                body = _read_json(self)
                _json_response(self, 200, desktop_controller().capture(body))
            except Exception as exc:
                _json_response(self, 500, {"error": str(exc)})
            return
        if self.path == "/desktop/action":
            try:
                body = _read_json(self)
                status, ethics = _post_json(f"{self.runner.ms3_url}/ethics/evaluate", _desktop_action_intent(body), timeout=20)
                if status >= 400 or not _ethics_allows(ethics if isinstance(ethics, dict) else {}):
                    _json_response(self, 403, {"error": "desktop action blocked by MS3 ethics", "ethics": ethics})
                    return
                _json_response(self, 200, desktop_controller().act(body))
            except DesktopSafetyError as exc:
                _json_response(self, 403, {"error": str(exc), "fail_closed": True})
            except Exception as exc:
                _json_response(self, 500, {"error": str(exc)})
            return
        if self.path == "/hermes/tool":
            try:
                body = _read_json(self)
                tool_name = str(body.get("tool") or body.get("tool_name") or "").strip()
                if not tool_name:
                    _json_response(self, 400, {"error": "tool is required"})
                    return
                result = self.runner.dispatch_hermes_tool(
                    tool_name,
                    body.get("args") if isinstance(body.get("args"), dict) else {},
                    session_id=body.get("session_id") or None,
                )
                _json_response(self, 200, result)
            except HermesUnavailable as exc:
                _json_response(self, 503, {"error": str(exc), "fail_closed": True})
            except Exception as exc:
                _json_response(self, 500, {"error": str(exc)})
            return
        if self.path == "/api/v1/double-agent/jobs":
            self._double_agent_submit()
            return
        if self.path.startswith("/api/v1/double-agent/jobs/"):
            tail = self.path[len("/api/v1/double-agent/jobs/"):]
            job_id, _, suffix = tail.partition("/")
            if suffix == "cancel":
                self._double_agent_cancel(job_id)
                return
            if suffix == "mark-stale":
                self._double_agent_mark_stale(job_id)
                return
        if self.path.startswith("/api/v1/double-agent/conversations/"):
            tail = self.path[len("/api/v1/double-agent/conversations/"):]
            conv_id, _, suffix = tail.partition("/")
            if suffix == "revisions":
                self._double_agent_revision_bump(conv_id)
                return
        if self.path in {"/hermes/update", "/api/v1/hermes/update"}:
            try:
                body = _read_json(self)
                target = body.get("target_version") or body.get("version") or None
                if target is not None and not isinstance(target, str):
                    _json_response(self, 400, {"error": "target_version must be a string"})
                    return
                snap = hermes_admin.trigger_update(
                    target_version=target,
                    request_user=str(body.get("request_user") or "ms4-gateway"),
                )
                _json_response(self, 202, snap.to_dict())
            except HermesUpgradeError as exc:
                _json_response(self, 400, {"error": str(exc)})
            except Exception as exc:
                _json_response(self, 500, {"error": str(exc)})
            return
        # Match the path component only; voice routes accept a query string
        # for session_id / model / voice overrides.
        path_only = self.path.split("?", 1)[0]
        if path_only == "/voice/transcribe":
            self._voice_transcribe()
            return
        if path_only == "/voice/synthesize":
            self._voice_synthesize()
            return
        if path_only == "/voice/turn":
            self._voice_turn()
            return
        if path_only == "/voice/turn/stream":
            self._voice_turn_stream()
            return
        if self.path == "/vision/analyze-local":
            try:
                body = _read_json(self)
                result = analyze_local_image(
                    hivemind_url=self.runner.hivemind_url,
                    image_path=str(body.get("image_path") or ""),
                    question=body.get("question") or body.get("prompt") or None,
                    model=body.get("model") or None,
                )
                _json_response(self, 200, result)
            except ValueError as exc:
                _json_response(self, 400, {"error": str(exc), "fail_closed": True})
            except Exception as exc:
                _json_response(self, 502, {"error": str(exc), "fail_closed": True})
            return
        if self.path != "/chat":
            _json_response(self, 404, {"error": "not found"})
            return
        try:
            body = _read_json(self)
            message = str(body.get("message") or body.get("text") or "").strip()
            if not message:
                _json_response(self, 400, {"error": "message is required"})
                return
            result = self.runner.chat(
                message,
                session_id=body.get("session_id") or None,
                model=body.get("model_id") or body.get("model") or None,
            )
            _json_response(self, 200, result)
        except HermesUnavailable as exc:
            _json_response(self, 503, {"error": str(exc), "fail_closed": True})
        except ValueError as exc:
            _json_response(self, 400, {"error": str(exc), "model_incompatible": True})
        except Exception as exc:
            _json_response(self, 500, {"error": str(exc)})

    def _read_audio_body(self) -> tuple[bytes, str]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length) if length > 0 else b""
        return parse_audio_request(self.headers.get("Content-Type", ""), body)

    def _voice_transcribe(self) -> None:
        try:
            audio, filename = self._read_audio_body()
            params = self._parse_query()
            model = params.get("model")
            result = transcribe(
                hivemind_url=self.runner.hivemind_url,
                audio=audio,
                filename=filename,
                model=model,
            )
            _json_response(self, 200, {
                "schema": "Ms4VoiceTranscription.v1",
                "text": result["text"],
                "model": result["model"],
            })
        except VoiceRequestError as exc:
            _json_response(self, 400, {"error": str(exc)})
        except VoiceUnavailable as exc:
            _json_response(self, 503, {"error": str(exc), "fail_closed": True})
        except Exception as exc:
            _json_response(self, 500, {"error": str(exc)})

    def _voice_synthesize(self) -> None:
        try:
            body = _read_json(self) or {}
            text = str(body.get("text") or "").strip()
            tts_model = body.get("model")
            voice = body.get("voice")
            response_format = body.get("response_format") or DEFAULT_TTS_FORMAT
            result = synthesize(
                hivemind_url=self.runner.hivemind_url,
                text=text,
                model=tts_model,
                voice=voice,
                response_format=response_format,
            )
            self.send_response(200)
            self.send_header("Content-Type", result["content_type"])
            self.send_header("Content-Length", str(len(result["audio_bytes"])))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("X-Ms4-Tts-Model", result["model"])
            self.send_header("X-Ms4-Tts-Voice", result["voice"])
            self.end_headers()
            self.wfile.write(result["audio_bytes"])
        except VoiceRequestError as exc:
            _json_response(self, 400, {"error": str(exc)})
        except VoiceUnavailable as exc:
            _json_response(self, 503, {"error": str(exc), "fail_closed": True})
        except Exception as exc:
            _json_response(self, 500, {"error": str(exc)})

    def _voice_turn(self) -> None:
        try:
            audio, filename = self._read_audio_body()
            params = self._parse_query()
            session_id = params.get("session_id") or None
            # ?model= is the Face Lobe chat model; ?tts_model= is the
            # TTS model (tts-1 / tts-1-hd). See _voice_turn_stream for
            # the bug history.
            model = params.get("model") or None
            tts_model = params.get("tts_model") or None
            tts_voice = params.get("voice") or None
            tts_format = params.get("response_format") or None
            result = voice_ptt_turn(
                runner=self.runner,
                audio=audio,
                filename=filename,
                session_id=session_id,
                model=model,
                tts_model=tts_model,
                tts_voice=tts_voice,
                response_format=tts_format,
            )
            _json_response(self, 200, {
                "schema": "Ms4VoicePttTurn.v1",
                "transcript": result.transcript,
                "reply_text": result.reply_text,
                "reply_audio_base64": result.reply_audio_base64,
                "reply_audio_mime": result.reply_audio_mime,
                "session_id": result.session_id,
                "foreground_model": result.foreground_model,
                "transcription_model": result.transcription_model,
                "tts_model": result.tts_model,
                "grounding_source": result.grounding_source,
                "router": result.router,
                "dispatched_job": result.dispatched_job,
            })
        except VoiceRequestError as exc:
            _json_response(self, 400, {"error": str(exc)})
        except VoiceUnavailable as exc:
            _json_response(self, 503, {"error": str(exc), "fail_closed": True})
        except Exception as exc:
            _json_response(self, 500, {"error": str(exc)})

    def _voice_turn_stream(self) -> None:
        """SSE streaming voice turn — sentence-chunked TTS for ChatGPT-mobile latency.

        Wire-format events (each `data: { ... }\\n\\n`):

          event: status         {"phase": "transcribing"|"thinking"}
          event: transcript     {"text": "...", "asr_ms": int, "model": "..."}
          event: text_delta     {"text": "..."}             # streaming chat tokens
          event: chunk_scheduled {"index": int, "text": "..."} # TTS submitted
          event: audio_chunk    {"index": int, "text": "...", "audio_base64": "...", "audio_mime": "..."}
          event: audio_error    {"index": int, "error": "..."}
          event: done           {final metrics + reply_text + ...}
          event: error          {"error": "...", "fail_closed": bool?}
        """
        try:
            audio, filename = self._read_audio_body()
        except VoiceRequestError as exc:
            _json_response(self, 400, {"error": str(exc)})
            return
        except Exception as exc:
            _json_response(self, 500, {"error": str(exc)})
            return

        params = self._parse_query()
        session_id = params.get("session_id") or None
        # ?model= is the FACE LOBE chat model (e.g. phi4-mini). The TTS
        # model (tts-1 / tts-1-hd) is a separate knob — earlier the UI
        # was mistakenly sending ?model=tts-1 which made the Face Lobe
        # try to use tts-1 as a chat model, causing every voice turn
        # to fall through to the canned-failure reply.
        model = params.get("model") or None
        tts_model = params.get("tts_model") or None
        tts_voice = params.get("voice") or None
        tts_format = params.get("response_format") or None
        # Per-turn override for the TTS engine. UI Settings dialog
        # passes ?engine=rest or ?engine=ws_super; missing/blank falls
        # back to MS4_VOICE_TTS_ENGINE env default.
        engine_override = params.get("engine") or None

        events: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue()
        # When the browser aborts the fetch (barge-in / new turn) the
        # _sse_event write raises and returns False. We flip this event
        # to signal voice_ptt_turn_stream to stop scheduling more TTS
        # work and tear down its WS engine immediately — otherwise we'd
        # keep paying HiveMind cycles for audio nobody is listening to.
        client_alive = threading.Event()
        client_alive.set()

        def emit(event: str, payload: dict[str, Any]) -> bool:
            if not client_alive.is_set():
                return False
            events.put((event, payload))
            return True

        def worker() -> None:
            try:
                result = voice_ptt_turn_stream(
                    runner=self.runner,
                    audio=audio,
                    filename=filename,
                    session_id=session_id,
                    model=model,
                    tts_model=tts_model,
                    tts_voice=tts_voice,
                    response_format=tts_format,
                    engine=engine_override,
                    emit=emit,
                    client_alive=client_alive,
                )
                events.put(("done", result))
            except VoiceRequestError as exc:
                events.put(("error", {"error": str(exc)}))
            except VoiceUnavailable as exc:
                events.put(("error", {"error": str(exc), "fail_closed": True}))
            except Exception as exc:
                events.put(("error", {"error": str(exc)}))

        _sse_start(self)
        if not _sse_event(self, "status", {"status": "started"}):
            client_alive.clear()
            return
        thread = threading.Thread(target=worker, daemon=True, name="ms4-voice-stream")
        thread.start()
        while True:
            try:
                event, payload = events.get(timeout=2.0)
            except queue.Empty:
                if not _sse_event(self, "heartbeat", {"status": "running"}):
                    client_alive.clear()
                    return
                continue
            if not _sse_event(self, event, payload):
                client_alive.clear()
                return
            if event in {"done", "error"}:
                return

    # ------------------------------------------------------------------
    # Settings (UI-facing read + lightweight admin actions)
    # ------------------------------------------------------------------

    def _settings_get(self) -> None:
        """Return effective gateway defaults so the UI's Settings dialog
        can show what the server currently thinks and which knobs are
        env-controlled vs UI-overridable."""
        import machine_spirit_4.gateway.voice as voice_mod
        import machine_spirit_4.gateway.context as context_mod

        # Best-effort voice/ASR readiness probe (5s budget). We don't
        # block the Settings dialog if it's slow — fall through with the
        # error message so the operator still sees the rest.
        asr_ready: dict[str, Any]
        try:
            voice_status = voice_mod.check_voice_ready(self.runner.ms3_url, timeout=3)
            asr_ready = {"ready": True, "detail": voice_status.get("voice_input_ready")}
        except Exception as exc:
            asr_ready = {"ready": False, "detail": str(exc)[:240]}

        from .hivemind_state import hivemind_auth_configured, mcp_base_url

        payload = {
            "schema": "Ms4Settings.v1",
            "endpoints": {
                "gateway": f"http://{self.headers.get('Host') or '127.0.0.1:9180'}",
                "ms3": self.runner.ms3_url,
                "hivemind": self.runner.hivemind_url,
                # Direct MCP gateway URL (preferred for lower latency
                # than the HLI proxy). Falls back automatically on
                # connect failure; pin via MS4_HIVEMIND_MCP_URL.
                "hivemind_mcp": mcp_base_url(self.runner.hivemind_url),
            },
            "auth": {
                # True when MS4_HIVEMIND_API_KEY is set. The Settings
                # dialog renders this so operators can spot misconfig
                # before the cluster starts 401-ing.
                "hivemind_auth_configured": hivemind_auth_configured(),
            },
            "voice": {
                "tts_engine_default": voice_mod.DEFAULT_ENGINE,
                "tts_engine_choices": sorted(voice_mod.VALID_ENGINES),
                "tts_voice_default": voice_mod.DEFAULT_TTS_VOICE,
                "tts_voice_known": [
                    # OpenAI-compatible voice IDs HiveMind TTS_SUPER honors.
                    # The UI uses this as the dropdown source; the server
                    # accepts any string the cluster supports.
                    "alloy", "vega", "echo", "onyx", "nova", "shimmer", "fable",
                ],
                "tts_model_default": voice_mod.DEFAULT_TTS_MODEL,
                "tts_model_choices": ["tts-1", "tts-1-hd"],
                "tts_format_default": voice_mod.DEFAULT_TTS_FORMAT,
                "asr_model_default": voice_mod.DEFAULT_TRANSCRIBE_MODEL,
                "asr_ready": asr_ready,
                "first_chunk_min_words": voice_mod.FIRST_CHUNK_MIN_WORDS,
                "stream_stall_timeout_secs": int(os.environ.get("MS4_FACE_LOBE_STREAM_STALL_TIMEOUT", "12")),
            },
            "grounding": {
                "cache_ttl_secs": context_mod.GROUNDING_CACHE_TTL_SECS,
                "cache_entries": list(_grounding_cache_keys()),
            },
            "hermes": {
                "version": _safe_hermes_version(),
            },
            "face_lobe": {
                "default_model": self.runner.default_model,
                "fallback_model": os.environ.get("MS4_DEFAULT_MODEL", self.runner.default_model),
                "foreground_override": os.environ.get("MS4_FOREGROUND_MODEL") or None,
            },
        }
        _json_response(self, 200, payload)

    # ------------------------------------------------------------------
    # Voice services lifecycle (HiveMind ASR / TTS / TTS_SUPER)
    # ------------------------------------------------------------------

    def _voice_services_get(self) -> None:
        """Combined ASR / TTS / TTS_SUPER status snapshot.

        The UI reads this on Settings open and on a short interval
        after triggering provisioning so it can flip the banner from
        'provisioning' to 'voice ready' without a page refresh.
        """
        try:
            services = list_voice_services(self.runner.hivemind_url)
        except Exception as exc:
            _json_response(self, 502, {"error": f"voice services lookup failed: {exc}"})
            return
        _json_response(self, 200, {
            "schema": "Ms4VoiceServicesSnapshot.v1",
            "services": services,
            "hivemind_url": self.runner.hivemind_url,
            "managed_services": list(VOICE_SERVICES),
        })

    def _voice_service_provision(self, service: str) -> None:
        """Request HiveMind to provision a voice service.

        Returns immediately with whatever HiveMind reports (typically
        ``status: 'provisioning'`` + ``provision_id``). The UI then
        polls /voice/services until ``healthy`` to know it's ready.
        Audit-logged so the operator can see who triggered what.
        """
        try:
            body = _read_json(self) if (self.headers.get("Content-Length") or "0") != "0" else {}
        except Exception:
            body = {}
        body = body if isinstance(body, dict) else {}
        try:
            result = request_voice_service(
                self.runner.hivemind_url,
                service,
                model=body.get("model"),
                tier=body.get("tier"),
                mode=body.get("mode"),
                backend=body.get("backend"),
            )
        except VoiceServiceUnknown as exc:
            _json_response(self, 400, {"error": str(exc), "managed_services": list(VOICE_SERVICES)})
            return
        except VoiceAdminError as exc:
            _json_response(self, 502, {"error": str(exc)})
            return
        except Exception as exc:
            _json_response(self, 500, {"error": str(exc)})
            return
        append_event("voice_service_provision_requested", {
            "service": service,
            "provision_id": result.get("provision_id"),
            "status": result.get("status"),
            "model": result.get("model"),
            "backend": result.get("backend"),
        })
        _json_response(self, 202, {
            "schema": "Ms4VoiceProvisionAck.v1",
            "service": result.get("service") or service.upper(),
            "result": result,
            "poll_url": "/voice/services",
        })

    def _voice_service_release(self, service: str) -> None:
        """Release a previously-provisioned voice resource."""
        try:
            body = _read_json(self) if (self.headers.get("Content-Length") or "0") != "0" else {}
        except Exception:
            body = {}
        body = body if isinstance(body, dict) else {}
        try:
            result = release_voice_service(
                self.runner.hivemind_url,
                service,
                model=body.get("model"),
                provision_id=body.get("provision_id"),
            )
        except VoiceServiceUnknown as exc:
            _json_response(self, 400, {"error": str(exc)})
            return
        except VoiceAdminError as exc:
            _json_response(self, 502, {"error": str(exc)})
            return
        except Exception as exc:
            _json_response(self, 500, {"error": str(exc)})
            return
        append_event("voice_service_release_requested", {
            "service": service,
            "result_status": result.get("status"),
        })
        _json_response(self, 200, {
            "schema": "Ms4VoiceReleaseAck.v1",
            "service": result.get("service") or service.upper(),
            "result": result,
        })

    # ------------------------------------------------------------------
    # Reflex audio (pre-rendered canned phrases)
    # ------------------------------------------------------------------

    def _reflexes_list(self) -> None:
        """Catalog snapshot + on-disk availability for one voice."""
        params = self._parse_query()
        voice = params.get("voice") or DEFAULT_REFLEX_VOICE
        try:
            _json_response(self, 200, list_reflexes(voice=voice))
        except Exception as exc:
            _json_response(self, 500, {"error": str(exc)})

    def _reflexes_serve(self, tail: str) -> None:
        """Serve a single reflex's cached audio bytes.

        URL shape: ``/reflexes/<voice>/<reflex_id>.wav``. We split on
        the LAST '/' so voice ids that include '/' (unlikely but
        possible if an operator went wild) at least don't crash here —
        the canned_reflexes module still sanitizes the voice id
        against a strict allowlist before touching the filesystem.
        """
        # Drop any query string the operator may have appended.
        tail = tail.split("?", 1)[0]
        if "/" not in tail:
            _json_response(self, 400, {"error": "expected /reflexes/<voice>/<reflex_id>.wav"})
            return
        voice, _, leaf = tail.rpartition("/")
        if not leaf.endswith(".wav"):
            _json_response(self, 400, {"error": "reflex paths must end in .wav"})
            return
        reflex_id = leaf[: -len(".wav")]
        try:
            audio = read_reflex(reflex_id=reflex_id, voice=voice)
        except ReflexUnknown as exc:
            _json_response(self, 400, {"error": str(exc), "known_reflexes": [r.id for r in REFLEXES]})
            return
        except Exception as exc:
            _json_response(self, 500, {"error": str(exc)})
            return
        if audio is None:
            _json_response(
                self,
                404,
                {
                    "error": f"reflex not generated for voice {voice!r}; run POST /reflexes/regenerate",
                    "reflex_id": reflex_id,
                    "voice": voice,
                },
            )
            return
        # Plain audio/wav response with permissive caching headers — the
        # UI fetches each reflex once at page load and decodes it into a
        # long-lived AudioBuffer.
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(audio)))
        self.send_header("Cache-Control", "public, max-age=60")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(audio)

    def _reflexes_regenerate(self) -> None:
        """Force a (re-)render of every reflex for the configured voice.

        Body fields (all optional): ``{voice, model, response_format, force, async}``.
        When ``async`` is true we fire generate_all_async and return 202
        immediately so the UI doesn't block during a long render; the
        next GET /reflexes will show updated availability counts.
        """
        try:
            body = _read_json(self) if (self.headers.get("Content-Length") or "0") != "0" else {}
        except Exception:
            body = {}
        body = body if isinstance(body, dict) else {}
        voice = body.get("voice") or DEFAULT_REFLEX_VOICE
        model = body.get("model")
        response_format = body.get("response_format")
        force = bool(body.get("force", True))
        run_async = bool(body.get("async", False))

        if run_async:
            generate_all_async(
                hivemind_url=self.runner.hivemind_url,
                voice=voice,
                model=model,
                response_format=response_format,
                force=force,
            )
            append_event("reflex_regenerate_requested", {
                "voice": voice, "model": model, "force": force, "async": True,
            })
            _json_response(self, 202, {
                "schema": "Ms4ReflexRegenerationAck.v1",
                "started": True,
                "voice": voice,
                "force": force,
                "async": True,
                "poll_url": "/reflexes",
            })
            return

        try:
            result = generate_all(
                hivemind_url=self.runner.hivemind_url,
                voice=voice,
                model=model,
                response_format=response_format,
                force=force,
            )
        except Exception as exc:
            _json_response(self, 502, {"error": str(exc)})
            return
        append_event("reflex_regenerate_completed", {
            "voice": result.get("voice"),
            "generated": [g["id"] for g in (result.get("generated") or [])],
            "skipped": result.get("skipped"),
            "failed": list((result.get("failed") or {}).keys()),
        })
        _json_response(self, 200, result)

    # ------------------------------------------------------------------
    # TMR doctrine (Deus Acuo Machina Machina) — read + inject
    # ------------------------------------------------------------------

    def _doctrine_meta(self) -> None:
        try:
            _json_response(self, 200, doctrine_meta())
        except DoctrineUnavailable as exc:
            _json_response(self, 503, {"error": str(exc)})
        except Exception as exc:
            _json_response(self, 500, {"error": str(exc)})

    def _doctrine_full(self) -> None:
        """Serve the full TMR Bible markdown. Plain ``text/markdown``
        with the meta block returned alongside via header so callers
        get the section count without a second round-trip."""
        try:
            text = get_full_bible()
            md = text.encode("utf-8")
        except DoctrineUnavailable as exc:
            _json_response(self, 503, {"error": str(exc)})
            return
        except Exception as exc:
            _json_response(self, 500, {"error": str(exc)})
            return
        m = doctrine_meta()
        self.send_response(200)
        self.send_header("Content-Type", "text/markdown; charset=utf-8")
        self.send_header("Content-Length", str(len(md)))
        self.send_header("X-Ms4-Doctrine-Sections", str(m.get("section_count", 0)))
        self.send_header("X-Ms4-Doctrine-Chars", str(m.get("chars", 0)))
        self.send_header("Cache-Control", "public, max-age=60")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(md)

    def _doctrine_sections_list(self) -> None:
        try:
            sections = list_sections()
        except DoctrineUnavailable as exc:
            _json_response(self, 503, {"error": str(exc)})
            return
        except Exception as exc:
            _json_response(self, 500, {"error": str(exc)})
            return
        _json_response(self, 200, {
            "schema": "Ms4DoctrineSections.v1",
            "sections": [
                {
                    "id": s.id,
                    "title": s.title,
                    "level": s.level,
                    "char_count": s.char_count,
                    "start": s.start,
                    "end": s.end,
                }
                for s in sections
            ],
            "total_sections": len(sections),
        })

    def _doctrine_section_serve(self, section_id: str) -> None:
        section_id = section_id.split("?", 1)[0].rstrip("/")
        try:
            section, text = get_section(section_id)
        except DoctrineSectionUnknown as exc:
            _json_response(self, 404, {"error": str(exc)})
            return
        except DoctrineUnavailable as exc:
            _json_response(self, 503, {"error": str(exc)})
            return
        _json_response(self, 200, {
            "schema": "Ms4DoctrineSection.v1",
            "id": section.id,
            "title": section.title,
            "level": section.level,
            "char_count": section.char_count,
            "text": text,
        })

    def _doctrine_read_into_session(self) -> None:
        """Inject the bible (or one section) into a Face Lobe session
        so future turns see the doctrine in their conversation_history.

        Body: ``{session_id?, kind: "full"|"section", section_id?, model?, acknowledgment?}``.
        ``session_id`` defaults to ``ms4-default-doctrine``; the UI
        passes the current chat's session_id so the doctrine shapes
        the same conversation the operator is having.
        """
        try:
            body = _read_json(self) if (self.headers.get("Content-Length") or "0") != "0" else {}
        except Exception:
            body = {}
        body = body if isinstance(body, dict) else {}
        session_id = body.get("session_id") or "ms4-default-doctrine"
        kind = body.get("kind") or "full"
        section_id = body.get("section_id")
        model = body.get("model")
        ack = body.get("acknowledgment")
        try:
            result = inject_into_session(
                self.runner.face_lobe_chat,
                session_id=session_id,
                model=model,
                kind=kind,
                section_id=section_id,
                acknowledgment=ack,
            )
        except DoctrineSectionUnknown as exc:
            _json_response(self, 404, {"error": str(exc)})
            return
        except ValueError as exc:
            _json_response(self, 400, {"error": str(exc)})
            return
        except DoctrineUnavailable as exc:
            _json_response(self, 503, {"error": str(exc)})
            return
        except Exception as exc:
            _json_response(self, 500, {"error": str(exc)})
            return
        append_event("doctrine_read_into_session", {
            "session_id": session_id,
            "kind": kind,
            "section_id": section_id,
            "chars_injected": result.get("chars_injected"),
        })
        _json_response(self, 200, result)

    def _doctrine_reload(self) -> None:
        try:
            result = reload_bible()
        except Exception as exc:
            _json_response(self, 500, {"error": str(exc)})
            return
        append_event("doctrine_reloaded", result)
        _json_response(self, 200, {"schema": "Ms4DoctrineReload.v1", **result})

    # ------------------------------------------------------------------
    # MS3 spirit state — identity / psyche / heartbeat
    # ------------------------------------------------------------------

    def _spirit_state_get(self) -> None:
        try:
            snapshot = get_state_snapshot(self.runner.ms3_url)
        except Exception as exc:
            _json_response(self, 500, {"error": str(exc)})
            return
        snapshot["heartbeat"] = heartbeat_status()
        _json_response(self, 200, snapshot)

    def _spirit_heartbeat(self) -> None:
        """Manual heartbeat — the gateway already beats automatically
        in the background, but the operator can poke it from the UI
        to confirm MS3 is reachable on demand."""
        try:
            result = send_heartbeat(self.runner.ms3_url)
        except Exception as exc:
            _json_response(self, 502, {"error": str(exc)})
            return
        append_event("spirit_heartbeat", {"result": result})
        _json_response(self, 200, {
            "schema": "Ms4SpiritHeartbeatAck.v1",
            "result": result,
            "status": heartbeat_status(),
        })

    # ------------------------------------------------------------------
    # HiveMind cluster-state observability (jobs / load / health)
    # ------------------------------------------------------------------

    def _hivemind_active_get(self) -> None:
        """Active inference + pulls + scatter + training jobs across
        the cluster — via ``hivemind.jobs.active@v1``. The same
        ``summary`` string powers the Face Lobe context block when a
        chat turn asks "what's running?".
        """
        try:
            body = get_active_jobs(self.runner.hivemind_url)
        except HivemindStateError as exc:
            _json_response(self, 502, {"error": str(exc)})
            return
        except Exception as exc:
            _json_response(self, 500, {"error": str(exc)})
            return
        _json_response(self, 200, body)

    def _hivemind_load_get(self) -> None:
        """Cluster load-stats: amplification ratio, criticality
        counters, shed totals. Via ``hivemind.cluster.load@v1``."""
        try:
            body = get_cluster_load(self.runner.hivemind_url)
        except HivemindStateError as exc:
            _json_response(self, 502, {"error": str(exc)})
            return
        except Exception as exc:
            _json_response(self, 500, {"error": str(exc)})
            return
        _json_response(self, 200, body)

    def _hivemind_state_get(self) -> None:
        """Combined cluster snapshot the UI Settings panel renders:
        active jobs + cluster load + service health. Fail-soft — a
        per-tool error doesn't block the whole snapshot.
        """
        try:
            snapshot = get_hivemind_snapshot(self.runner.hivemind_url)
        except Exception as exc:
            _json_response(self, 500, {"error": str(exc)})
            return
        if snapshot.get("errors"):
            append_event("hivemind_state_partial_failure", {"errors": snapshot["errors"]})
        _json_response(self, 200, snapshot)

    def _settings_clear_grounding_cache(self) -> None:
        """Drop every cached grounding entry. Cheap, idempotent. Useful
        when an operator wants to force a fresh HiveMind MCP fetch
        (e.g. nodes have just been added / removed)."""
        import machine_spirit_4.gateway.context as context_mod

        before = list(_grounding_cache_keys())
        context_mod.clear_grounding_cache()
        append_event("grounding_cache_cleared", {"entries_before": before})
        _json_response(self, 200, {
            "ok": True,
            "cleared": len(before),
            "entries_before": before,
        })

    def _double_agent_submit(self) -> None:
        try:
            body = _read_json(self) or {}
            if not isinstance(body, dict):
                _json_response(self, 400, {"error": "request body must be a JSON object"})
                return
            envelope = JobEnvelope.from_dict(body)
            snapshot = default_runner().submit(envelope)
            _json_response(self, 202, snapshot or {"error": "submission accepted but snapshot missing"})
        except DoubleAgentSchemaError as exc:
            _json_response(self, 400, {"error": str(exc)})
        except RunnerError as exc:
            _json_response(self, 503, {"error": str(exc)})
        except Exception as exc:
            _json_response(self, 500, {"error": str(exc)})

    def _double_agent_list(self) -> None:
        params = self._parse_query()
        conv = params.get("conversation_id")
        if conv is not None and not is_safe_conversation_id(conv):
            _json_response(self, 400, {"error": "unsafe conversation_id"})
            return
        states_raw = params.get("states")
        states = tuple(s for s in (states_raw or "").split(",") if s) or None
        try:
            limit = int(params.get("limit") or "50")
        except ValueError:
            limit = 50
        jobs = default_runner().list(conversation_id=conv, states=states, limit=limit)
        _json_response(self, 200, {"jobs": jobs})

    def _double_agent_get(self, job_id: str) -> None:
        if not is_safe_job_id(job_id):
            _json_response(self, 400, {"error": "unsafe job_id"})
            return
        snap = default_runner().get(job_id)
        if snap is None:
            _json_response(self, 404, {"error": "unknown job_id"})
            return
        _json_response(self, 200, snap)

    def _double_agent_cancel(self, job_id: str) -> None:
        if not is_safe_job_id(job_id):
            _json_response(self, 400, {"error": "unsafe job_id"})
            return
        try:
            snap = default_runner().cancel(job_id)
            _json_response(self, 200, snap)
        except RunnerError as exc:
            _json_response(self, 404, {"error": str(exc)})

    def _double_agent_mark_stale(self, job_id: str) -> None:
        if not is_safe_job_id(job_id):
            _json_response(self, 400, {"error": "unsafe job_id"})
            return
        try:
            body = _read_json(self) or {}
            reason = str(body.get("reason") or "Marked stale by operator.")
            snap = default_runner().mark_stale(job_id, reason=reason)
            _json_response(self, 200, snap)
        except RunnerError as exc:
            _json_response(self, 404, {"error": str(exc)})

    def _double_agent_events(self, job_id_with_query: str) -> None:
        job_id, _, _ = job_id_with_query.partition("?")
        if not is_safe_job_id(job_id):
            _json_response(self, 400, {"error": "unsafe job_id"})
            return
        params = self._parse_query()
        try:
            limit = int(params.get("limit") or "200")
        except ValueError:
            limit = 200
        after = params.get("after_event_id")
        events = default_blackboard().list_events(
            job_id, after_event_id=after, limit=limit
        )
        accept = self.headers.get("Accept", "")
        if "text/event-stream" not in accept:
            _json_response(self, 200, {"events": events})
            return
        _sse_start(self)
        if not _sse_event(self, "status", {"status": "stream_started", "count": len(events)}):
            return
        for event in events:
            if not _sse_event(self, event["type"], event):
                return
        _sse_event(self, "done", {"status": "stream_complete"})

    def _double_agent_revision_bump(self, conversation_id: str) -> None:
        if not is_safe_conversation_id(conversation_id):
            _json_response(self, 400, {"error": "unsafe conversation_id"})
            return
        try:
            body = _read_json(self) or {}
            excerpt = str(body.get("user_message_excerpt") or "")
            outcome = default_runner().bump_revision(conversation_id, user_message_excerpt=excerpt)
            _json_response(self, 200, outcome)
        except DoubleAgentSchemaError as exc:
            _json_response(self, 400, {"error": str(exc)})

    def _double_agent_revision_get(self, conversation_id: str) -> None:
        conv_id = conversation_id.split("?", 1)[0]
        if not is_safe_conversation_id(conv_id):
            _json_response(self, 400, {"error": "unsafe conversation_id"})
            return
        revision = default_blackboard().current_revision(conv_id)
        _json_response(self, 200, {"conversation_id": conv_id, "revision_id": revision})

    def _parse_query(self) -> dict[str, str]:
        if "?" not in self.path:
            return {}
        query = self.path.split("?", 1)[1]
        params: dict[str, str] = {}
        for part in query.split("&"):
            if not part:
                continue
            key, _, value = part.partition("=")
            params[key] = value
        return params

    def _stream_chat(self) -> None:
        try:
            body = _read_json(self)
            message = str(body.get("message") or body.get("text") or "").strip()
            if not message:
                _json_response(self, 400, {"error": "message is required"})
                return
            events: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue()

            def stream_callback(delta: str) -> None:
                if delta:
                    events.put(("token", {"text": delta}))

            def worker() -> None:
                try:
                    result = self.runner.chat(
                        message,
                        session_id=body.get("session_id") or None,
                        model=body.get("model_id") or body.get("model") or None,
                        stream_callback=stream_callback,
                    )
                    events.put(("done", result))
                except ValueError as exc:
                    events.put(("error", {"error": str(exc), "model_incompatible": True}))
                except HermesUnavailable as exc:
                    events.put(("error", {"error": str(exc), "fail_closed": True}))
                except Exception as exc:
                    events.put(("error", {"error": str(exc)}))

            _sse_start(self)
            if not _sse_event(self, "status", {"status": "started"}):
                return
            thread = threading.Thread(target=worker, daemon=True)
            thread.start()
            while True:
                try:
                    event, payload = events.get(timeout=2.0)
                except queue.Empty:
                    if not _sse_event(self, "heartbeat", {"status": "running"}):
                        return
                    continue
                if not _sse_event(self, event, payload):
                    return
                if event in {"done", "error"}:
                    return
        except Exception as exc:
            _json_response(self, 500, {"error": str(exc)})

    def _serve_static(self) -> None:
        relative = "index.html" if self.path == "/" else self.path.removeprefix("/static/")
        target = (WEB_ROOT / relative).resolve()
        if WEB_ROOT.resolve() not in target.parents and target != WEB_ROOT.resolve():
            _json_response(self, 403, {"error": "forbidden"})
            return
        if not target.is_file():
            _json_response(self, 404, {"error": "not found"})
            return
        content = target.read_bytes()
        content_type = "text/html; charset=utf-8" if target.suffix == ".html" else "text/plain; charset=utf-8"
        if target.suffix == ".js":
            content_type = "application/javascript; charset=utf-8"
        if target.suffix == ".css":
            content_type = "text/css; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def build_runner() -> Ms4HermesRunner:
    return Ms4HermesRunner(
        hermes_dir=os.environ.get("MS4_HERMES_DIR", str(Path.home() / "Documents" / "hermes-agent")),
        hivemind_url=os.environ.get("MS4_HIVEMIND_URL", "http://127.0.0.1:6089"),
        ms3_url=os.environ.get("MS4_MS3_URL", "http://127.0.0.1:9080"),
        default_model=os.environ.get("MS4_DEFAULT_MODEL", "qwen3-coder-next:latest"),
    )


def run(host: str = "127.0.0.1", port: int = 9180) -> None:
    require_contained_runtime(SERVICE_NAME)
    handler_cls = Ms4GatewayHandler
    handler_cls.runner = build_runner()
    hermes_admin.initialize_state()
    # The Double Agent runner deliberately stays in subprocess mode here so
    # cancel() can truly terminate a blocking Hermes model call. The child
    # process (`double_agent/_worker_entry.py`) constructs its own
    # Ms4HermesRunner from MS4_HERMES_DIR / MS4_HIVEMIND_URL / MS4_MS3_URL /
    # MS4_DEFAULT_MODEL — no shared state between parent and child beyond
    # the SQLite blackboard. Tests inject a chat_runner_factory directly via
    # JobRunner(chat_runner_factory=...) to bypass subprocess overhead.
    da_runner = default_runner()
    da_runner.recover_on_startup()
    # Pre-warm ASR + TTS + Face Lobe model on boot. Voice latency is
    # dominated by cold loads on the first turn (whisper ~3-8s, TTS
    # ~3-5s, Face Lobe model load ~5-10s). Each pre-warm runs in its
    # own daemon thread so the gateway itself doesn't block on boot;
    # the gateway is serving traffic the moment the HTTP server starts.
    def _prewarm(label, fn):
        t0 = time.monotonic()
        try:
            res = fn()
            elapsed = int((time.monotonic() - t0) * 1000)
            print(f"MS4 {label} pre-warm: {elapsed}ms — {res}", flush=True)
        except Exception as exc:
            print(f"MS4 {label} pre-warm failed: {exc}", flush=True)

    hivemind_url = handler_cls.runner.hivemind_url
    _set_hivemind_url_for_auth(hivemind_url)
    threading.Thread(target=_prewarm, args=("ASR", lambda: {"warmed": True, "text": (prewarm_asr(hivemind_url=hivemind_url) or {}).get("text", "")}), daemon=True, name="ms4-asr-prewarm").start()
    threading.Thread(target=_prewarm, args=("TTS", lambda: {"warmed": True, "bytes": len((prewarm_tts(hivemind_url=hivemind_url) or {}).get("audio_bytes") or b"")}), daemon=True, name="ms4-tts-prewarm").start()
    # TTS_SUPER WebSocket pre-warm: opens a WS to HiveMind's
    # /v1/text-to-speech/{voice}/stream-input, pushes a tiny "Ready."
    # payload, and closes. Pays the TCP/TLS/WS-upgrade and GIM model
    # load cost at boot so the first real voice turn doesn't. We are
    # the only callers of TTS_SUPER ws_super engine — if HiveMind
    # isn't ready, this logs and skips silently.
    threading.Thread(target=_prewarm, args=("TTS_SUPER WS", lambda: prewarm_tts_super_ws(hivemind_url=hivemind_url)), daemon=True, name="ms4-tts-super-prewarm").start()
    threading.Thread(target=_prewarm, args=("Face Lobe", lambda: prewarm_face_lobe_model(hivemind_url=hivemind_url)), daemon=True, name="ms4-flb-prewarm").start()
    # Pre-warm grounding caches (inventory + tools) so the first
    # "what's the cluster status?" turn doesn't pay 15-20s of MCP
    # round-trip on top of chat + TTS. Each grounding entry has a 60s
    # TTL (MS4_GROUNDING_CACHE_TTL) and is best-effort.
    threading.Thread(target=_prewarm, args=("Grounding", lambda: prewarm_grounding_cache(hivemind_url=hivemind_url)), daemon=True, name="ms4-grounding-prewarm").start()
    # Pre-render canned reflex audio (acknowledgments / error
    # apologies / "one moment" stalls). The default voice is rendered
    # in the background so the first voice turn — and especially
    # barge-in — can play instant audio without a HiveMind round-trip.
    # Already-rendered reflexes are skipped on subsequent boots.
    threading.Thread(target=_prewarm, args=("Reflexes", lambda: generate_all(hivemind_url=hivemind_url, force=False)), daemon=True, name="ms4-reflex-prewarm").start()
    # Start MS3 heartbeat: every MS4_SPIRIT_HEARTBEAT_SECS (default 60s)
    # POST /identity/heartbeat so MS3 keeps the spirit's continuity
    # checks happy even when the Face Lobe direct chat path is the
    # only thing running. Best-effort — failures are surfaced via
    # /spirit/state but don't crash the gateway.
    start_heartbeat_thread(handler_cls.runner.ms3_url)
    server = ThreadingHTTPServer((host, port), handler_cls)
    print(f"MS4 gateway listening on http://{host}:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    run(
        host=os.environ.get("MS4_GATEWAY_HOST", "127.0.0.1"),
        port=int(os.environ.get("MS4_GATEWAY_PORT", "9180")),
    )
