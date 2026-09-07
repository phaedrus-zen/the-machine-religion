from __future__ import annotations

import json
import logging
import os
import queue
import threading

log = logging.getLogger("ms4.gateway.server")
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from machine_spirit_4.scripts import runtime_common

# Match managed launchers before Hermes resolves its data home during import.
os.environ.setdefault("HERMES_HOME", str(runtime_common.hermes_home()))

from machine_spirit_4 import hermes_admin
from machine_spirit_4.contained import require_contained_runtime
from machine_spirit_4.deps_status import dependency_status
from machine_spirit_4.desktop import DesktopSafetyError, desktop_controller
from machine_spirit_4.double_agent import (
    DEPTH_MIN_TOTAL_PARAM_B,
    DEPTH_FALLBACK_MIN_TOTAL_PARAM_B,
    DEPTH_FAST_TOOL_FALLBACK_MODEL,
    DEPTH_PREFERRED_CLUSTER_TARGET,
    JobEnvelope,
    JobRunner,
    MS4_DEPTH_MODEL_ENV,
    RunnerError,
    SchemaError as DoubleAgentSchemaError,
    choose_depth_model,
    depth_fallback_model,
    default_blackboard,
    default_runner,
    result_matches_job_identity,
)
from machine_spirit_4.double_agent.model_picker import (
    DEFAULT_FOREGROUND_MODEL,
    _is_chat_capable,
    choose_foreground_model,
)
from machine_spirit_4.double_agent.safety import (
    is_safe_conversation_id,
    is_safe_job_id,
)
from machine_spirit_4.double_agent.worker import build_real_chat_runner
from machine_spirit_4.hermes_admin import HermesUpgradeError

from .audit import append_event, read_events, read_recent_voice_turns
from .hermes_runner import HermesUnavailable, Ms4HermesRunner, _face_lobe_prior_context
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
from . import (
    adapter_admin,
    app_admin,
    game_admin,
    gpu_mode_admin,
    gpu_passthrough,
    hivemind_tools,
    human_approval,
    loadout_admin,
    network_admin,
    oracle_admin,
    storage_admin,
    training_admin,
    vm_admin,
    voice_identity,
)
from .adapter_admin import AdapterAdminError
from .app_admin import AppAdminError
from .game_admin import GameAdminError
from .gpu_mode_admin import GpuModeAdminError
from .gpu_passthrough import GpuPassthroughError
from .hivemind_tools import HivemindToolError
from .human_approval import HumanApprovalError
from .loadout_admin import LoadoutAdminError
from .network_admin import NetworkAdminError
from .oracle_admin import OracleAdminError
from .storage_admin import StorageAdminError
from .training_admin import TrainingAdminError
from .vm_admin import VmAdminError
from .voice_identity import VoiceIdentityError
from .voice_candidate_reconcile import VoiceCandidateReconciler
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
    DEFAULT_TTS_MODEL,
    SELECTED_FACE_ADMISSION_PROFILE,
    _tts_location_policy,
    _emit_text_as_parallel_chunks,
    _new_voice_turn_id,
    VoiceRequestError,
    VoiceUnavailable,
    last_face_model,
    parse_audio_request,
    prewarm_asr,
    prewarm_face_lobe_model,
    prewarm_tts,
    prewarm_tts_super_ws,
    probe_voice_rest_concurrency,
    provision_tts_replicas,
    record_face_model,
    selected_face_admission_latency_budget_ms,
    check_voice_ready,
    synthesize,
    transcribe,
    voice_ptt_turn,
    voice_ptt_turn_stream,
)


ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = ROOT / "machine_spirit_4" / "web"
SERVICE_NAME = "ms4-gateway"
ROOT_WEB_ASSETS = {
    "/manifest.webmanifest": ("manifest.webmanifest", "application/manifest+json; charset=utf-8"),
    "/service-worker.js": ("service-worker.js", "application/javascript; charset=utf-8"),
}
_TTS_SUPER_MODELS = {
    "tts-1-hd",
    "gpt-4o-mini-tts",
    "tts_super",
    "tts-super",
}


def _tts_job_type_for_model(model: str) -> str:
    normalized = str(model or "").strip().lower()
    return "TTS_SUPER" if normalized in _TTS_SUPER_MODELS else "TTS"


def _selected_face_prewarm_timeout_s() -> float:
    """Outer selected-model cold-load budget (observed loads can exceed 80s)."""
    try:
        value = float(os.environ.get("MS4_FACE_SELECTED_PREWARM_TIMEOUT_S", "120"))
    except (TypeError, ValueError):
        value = 120.0
    # The browser grants five seconds of transport slack above this hard cap.
    return min(120.0, max(10.0, value))


def _public_recommend_audit(result: dict[str, Any] | None) -> dict[str, Any]:
    rec = (result or {}).get("recommend")
    if not isinstance(rec, dict) or not rec:
        return {}
    return {
        "recommend_owner": rec.get("owner"),
        "recommend_endpoint": rec.get("endpoint"),
        "recommend_generation": rec.get("generation"),
        "recommend_lease_id": rec.get("lease_id"),
    }


_SELECTED_FACE_PREWARM_LOCK = threading.Lock()
_SELECTED_FACE_PREWARM_CLIENTS: dict[str, dict[str, Any]] = {}
_SELECTED_FACE_PREWARM_CLIENT_LIMIT = 1024


def _begin_selected_face_prewarm(
    client_id: str,
    generation: int,
) -> tuple[threading.Event | None, int | None]:
    """Claim one browser generation and cancel its older server-side warm."""
    cancel_event = threading.Event()
    with _SELECTED_FACE_PREWARM_LOCK:
        previous = _SELECTED_FACE_PREWARM_CLIENTS.get(client_id)
        if previous is not None and generation <= int(previous["generation"]):
            return None, int(previous["generation"])
        if previous is not None:
            previous_cancel = previous.get("cancel_event")
            if isinstance(previous_cancel, threading.Event):
                previous_cancel.set()
        _SELECTED_FACE_PREWARM_CLIENTS[client_id] = {
            "generation": generation,
            "cancel_event": cancel_event,
            "updated_at": time.monotonic(),
        }
        if len(_SELECTED_FACE_PREWARM_CLIENTS) > _SELECTED_FACE_PREWARM_CLIENT_LIMIT:
            oldest = min(
                (
                    (key, value)
                    for key, value in _SELECTED_FACE_PREWARM_CLIENTS.items()
                    if key != client_id
                ),
                key=lambda item: float(item[1].get("updated_at") or 0.0),
                default=None,
            )
            if oldest is not None:
                stale = _SELECTED_FACE_PREWARM_CLIENTS.pop(oldest[0])
                stale_cancel = stale.get("cancel_event")
                if isinstance(stale_cancel, threading.Event):
                    stale_cancel.set()
    return cancel_event, None


def _selected_face_prewarm_is_current(
    client_id: str,
    generation: int,
    cancel_event: threading.Event,
) -> bool:
    with _SELECTED_FACE_PREWARM_LOCK:
        current = _SELECTED_FACE_PREWARM_CLIENTS.get(client_id)
        return bool(
            current is not None
            and int(current["generation"]) == generation
            and current.get("cancel_event") is cancel_event
            and not cancel_event.is_set()
        )


def _finish_selected_face_prewarm(
    client_id: str,
    generation: int,
    cancel_event: threading.Event,
) -> None:
    with _SELECTED_FACE_PREWARM_LOCK:
        current = _SELECTED_FACE_PREWARM_CLIENTS.get(client_id)
        if (
            current is not None
            and int(current["generation"]) == generation
            and current.get("cancel_event") is cancel_event
        ):
            current["cancel_event"] = None
            current["updated_at"] = time.monotonic()


def _commit_selected_face_prewarm(
    client_id: str,
    generation: int,
    cancel_event: threading.Event,
    effective_model: str,
) -> bool:
    """Atomically prove latest-generation ownership before keepwarm mutation."""
    with _SELECTED_FACE_PREWARM_LOCK:
        current = _SELECTED_FACE_PREWARM_CLIENTS.get(client_id)
        if not (
            current is not None
            and int(current["generation"]) == generation
            and current.get("cancel_event") is cancel_event
            and not cancel_event.is_set()
        ):
            return False
        record_face_model(effective_model)
        current["cancel_event"] = None
        current["updated_at"] = time.monotonic()
        return True


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


def _sse_event(
    handler: SimpleHTTPRequestHandler,
    event: str,
    payload: dict[str, Any],
    *,
    write_timeout: float | None = None,
) -> bool:
    body = f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")
    connection = getattr(handler, "connection", None)
    previous_timeout = None
    try:
        if write_timeout is not None and connection is not None:
            previous_timeout = connection.gettimeout()
            connection.settimeout(max(0.05, write_timeout))
        handler.wfile.write(body)
        handler.wfile.flush()
        return True
    except (BrokenPipeError, ConnectionResetError, OSError):
        return False
    finally:
        if write_timeout is not None and connection is not None:
            try:
                connection.settimeout(previous_timeout)
            except OSError:
                pass


def _voice_sse_write_ack_timeout() -> float:
    try:
        value = float(os.environ.get("MS4_VOICE_SSE_WRITE_ACK_TIMEOUT_S", "2.0"))
    except (TypeError, ValueError):
        value = 2.0
    return max(0.05, value)


def _voice_worker_cleanup_timeout() -> float:
    try:
        value = float(os.environ.get("MS4_VOICE_WORKER_CLEANUP_TIMEOUT_S", "2.0"))
    except (TypeError, ValueError):
        value = 2.0
    return max(0.05, value)


@dataclass
class _VoiceSseDelivery:
    event: str
    payload: dict[str, Any]
    terminal: bool = False
    acknowledged: threading.Event = field(default_factory=threading.Event)
    client_written: bool = False


def _voice_error_payload(exc: Exception, *, fail_closed: bool = False) -> dict[str, Any]:
    """Separate concise operator copy from raw voice-gate diagnostics."""
    diagnostics = str(exc)
    lowered = " ".join(diagnostics.split()).lower()
    if "provisioning" in lowered and "asr" in lowered:
        message = "Voice input is still provisioning. Try again shortly."
    elif (
        "voice_input_ready=false" in lowered
        or "configured_not_provisioned" in lowered
        or "no endpoints configured" in lowered
        or ("asr" in lowered and ("unhealthy" in lowered or "not ready" in lowered))
    ):
        message = "Voice input is not ready. Provision ASR and try again."
    elif any(token in lowered for token in ("unreachable", "connection refused", "timed out", "timeout")):
        message = "Voice services cannot reach HiveMind right now. Check the cluster and try again."
    else:
        message = "Voice is unavailable right now. Check the diagnostics and try again."

    payload: dict[str, Any] = {"error": message, "diagnostics": diagnostics}
    if fail_closed:
        payload["fail_closed"] = True
    return payload


def _read_json(handler: SimpleHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0") or "0")
    if length <= 0:
        return {}
    raw = handler.rfile.read(length).decode("utf-8")
    return json.loads(raw)


_VOICE_TRANSCRIPT_SCHEMA = "Ms4VoiceTranscriptTurn.v1"
_VOICE_TRANSCRIPT_MODEL = "bounded-session"
_VOICE_TRANSCRIPT_SOURCE = "voice_input_session"
_VOICE_TRANSCRIPT_MAX_CHARS = 65_536
_VOICE_TRANSCRIPT_FIELDS = frozenset({
    "schema",
    "transcript",
    "transcription_model",
    "source",
})


def _parse_voice_transcript_envelope(body: bytes) -> str:
    if not body:
        raise VoiceRequestError("transcript-ready request body is empty")
    if len(body) > (_VOICE_TRANSCRIPT_MAX_CHARS * 4) + 4096:
        raise VoiceRequestError("transcript-ready request body exceeds the bounded limit")
    try:
        text = body.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise VoiceRequestError("transcript-ready request body must be UTF-8 JSON") from exc

    def reject_duplicate_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        parsed: dict[str, Any] = {}
        for key, value in pairs:
            if key in parsed:
                raise VoiceRequestError(f"duplicate transcript-ready field: {key}")
            parsed[key] = value
        return parsed

    try:
        payload = json.loads(text, object_pairs_hook=reject_duplicate_fields)
    except VoiceRequestError:
        raise
    except (TypeError, ValueError) as exc:
        raise VoiceRequestError("transcript-ready request body must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise VoiceRequestError("transcript-ready request body must be a JSON object")

    fields = set(payload)
    if fields != _VOICE_TRANSCRIPT_FIELDS:
        unknown = sorted(fields - _VOICE_TRANSCRIPT_FIELDS)
        missing = sorted(_VOICE_TRANSCRIPT_FIELDS - fields)
        detail = []
        if unknown:
            detail.append(f"unknown fields: {', '.join(unknown)}")
        if missing:
            detail.append(f"missing fields: {', '.join(missing)}")
        raise VoiceRequestError(
            "transcript-ready envelope fields are not exact"
            + (f" ({'; '.join(detail)})" if detail else "")
        )
    if payload["schema"] != _VOICE_TRANSCRIPT_SCHEMA:
        raise VoiceRequestError("unsupported transcript-ready schema")
    if payload["transcription_model"] != _VOICE_TRANSCRIPT_MODEL:
        raise VoiceRequestError("unsupported transcript-ready transcription_model")
    if payload["source"] != _VOICE_TRANSCRIPT_SOURCE:
        raise VoiceRequestError("unsupported transcript-ready source")

    transcript = payload["transcript"]
    if not isinstance(transcript, str):
        raise VoiceRequestError("transcript-ready transcript must be a string")
    if len(transcript.encode("utf-16-le")) // 2 > _VOICE_TRANSCRIPT_MAX_CHARS:
        raise VoiceRequestError("transcript-ready transcript exceeds 65,536 characters")
    transcript = transcript.strip()
    if not transcript:
        raise VoiceRequestError("transcript-ready transcript must not be empty")
    return transcript


def _multipart_contains_transcript_fields(content_type: str, body: bytes) -> bool:
    boundary = None
    for part in (content_type or "").split(";"):
        part = part.strip()
        if part.lower().startswith("boundary="):
            boundary = part.split("=", 1)[1].strip().strip('"')
            break
    if not boundary:
        return False
    marker = ("--" + boundary).encode("utf-8")
    transcript_names = tuple(
        f'name="{field}"'.encode("utf-8") for field in _VOICE_TRANSCRIPT_FIELDS
    )
    for section in body.split(marker):
        header_end = section.find(b"\r\n\r\n")
        if header_end < 0:
            continue
        headers = section[:header_end].lower()
        if any(name in headers for name in transcript_names):
            return True
    return False


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
            body = payload if isinstance(payload, dict) else {"models": payload}
            if isinstance(body, dict):
                models = body.get("models")
                if isinstance(models, list):
                    body = dict(body)
                    body["models"] = [
                        model for model in models
                        if isinstance(model, dict) and _is_chat_capable(model)
                    ]
                    body["chat_filtered"] = True
            _json_response(self, status, body)
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
        if self.path.startswith("/easter/honorable"):
            self._easter_honorable()
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
        if self.path.startswith("/voice/recent-turns"):
            self._voice_recent_turns_get()
            return
        # ---- 3rd-party MCP importer/bridge (read) ----
        if self.path == "/api/v1/mcp/imports":
            self._mcp_imports_list()
            return
        if self.path.startswith("/api/v1/mcp/imports/") and self.path.endswith("/health"):
            self._mcp_imports_health(self.path[len("/api/v1/mcp/imports/"):-len("/health")])
            return
        # ---- Pure-MCP read routes (no HLI dependency, work even when
        # the inference gateway is down). Each route returns a stable
        # UI-friendly snapshot via the per-domain admin module.
        if self.path == "/hivemind/time":
            self._hivemind_time_get()
            return
        if self.path == "/hivemind/capability_matrix":
            self._hivemind_capability_matrix_get()
            return
        if self.path == "/hivemind/vms":
            self._hivemind_vms_get()
            return
        if self.path.startswith("/hivemind/vms/") and self.path.endswith("/gpus"):
            self._hivemind_vm_gpus_get(self.path[len("/hivemind/vms/"):-len("/gpus")])
            return
        if self.path.startswith("/hivemind/vms/") and self.path.endswith("/screenshot"):
            self._hivemind_vm_screenshot(self.path[len("/hivemind/vms/"):-len("/screenshot")])
            return
        if self.path == "/hivemind/apps":
            self._hivemind_apps_get()
            return
        if self.path.startswith("/hivemind/apps/"):
            tail = self.path[len("/hivemind/apps/"):]
            self._hivemind_app_get(tail)
            return
        if self.path == "/hivemind/storage":
            self._hivemind_storage_get()
            return
        if self.path == "/hivemind/network":
            self._hivemind_network_get()
            return
        if self.path == "/hivemind/gpu":
            self._hivemind_gpu_get()
            return
        # ---- GPU passthrough workflow (May 27 2026: GPU-P / DDA / vGPU) ----
        if self.path == "/hivemind/gpu/passthrough/snapshot":
            self._hivemind_gpu_passthrough_snapshot()
            return
        if self.path == "/hivemind/voice_identities":
            self._hivemind_voice_identities_get()
            return
        if self.path.startswith("/hivemind/approval/status/"):
            request_id = self.path[len("/hivemind/approval/status/"):]
            self._hivemind_approval_status_get(request_id)
            return
        if self.path == "/hivemind/api_keys":
            self._hivemind_api_keys_get()
            return
        if self.path == "/hivemind/ollama/tags":
            self._hivemind_ollama_tags_get()
            return
        if self.path == "/hivemind/crown":
            self._hivemind_crown_get()
            return
        # ---- New admin domains (May 26 2026 expansion) ----
        if self.path == "/hivemind/oracle/readiness":
            self._hivemind_oracle_readiness_get()
            return
        if self.path == "/hivemind/oracle/status":
            self._hivemind_oracle_status_get()
            return
        if self.path == "/hivemind/training":
            self._hivemind_training_get()
            return
        if self.path.startswith("/hivemind/training/status/"):
            job_id = self.path[len("/hivemind/training/status/"):]
            self._hivemind_training_status_get(job_id)
            return
        if self.path == "/hivemind/adapters":
            self._hivemind_adapters_get()
            return
        if self.path == "/hivemind/loadout":
            self._hivemind_loadout_get()
            return
        if self.path == "/hivemind/inference/models":
            self._hivemind_inference_models_get()
            return
        if self.path == "/hivemind/logos/prompts":
            self._hivemind_logos_prompts_get()
            return
        if self.path.startswith("/hivemind/logos/prompts/"):
            prompt_id = self.path[len("/hivemind/logos/prompts/"):]
            self._hivemind_logos_prompt_get(prompt_id)
            return
        if self.path == "/hivemind/services":
            self._hivemind_services_list_get()
            return
        # ---- Game-session orchestration (May 26 2026 Phase 1 dry-run) ----
        if self.path.startswith("/hivemind/games/") and self.path.endswith("/availability"):
            game_id = self.path[len("/hivemind/games/"):-len("/availability")]
            self._hivemind_game_availability_get(game_id)
            return
        if self.path.startswith("/hivemind/game-sessions/") and self.path.endswith("/status"):
            job_id = self.path[len("/hivemind/game-sessions/"):-len("/status")]
            self._hivemind_game_session_status_get(job_id)
            return
        if self.path.startswith("/hivemind/game-sessions/") and self.path.endswith("/evidence"):
            job_id = self.path[len("/hivemind/game-sessions/"):-len("/evidence")]
            self._hivemind_game_session_evidence_get(job_id)
            return
        if self.path == "/voice/status":
            self._voice_status_get()
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
        # Static dispatch alone uses the parsed path so cache-busting queries
        # reach reviewed files. All API routes above retain their existing raw
        # request-target matching and query parsing.
        request_path = urllib.parse.urlsplit(self.path).path
        if (
            request_path == "/"
            or request_path in ROOT_WEB_ASSETS
            or request_path.startswith("/static/")
        ):
            self._serve_static()
            return
        _json_response(self, 404, {"error": "not found"})

    def do_DELETE(self) -> None:
        if self.path.startswith("/api/v1/mcp/imports/"):
            self._mcp_imports_remove(self.path[len("/api/v1/mcp/imports/"):])
            return
        _json_response(self, 404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path == "/chat/stream":
            self._stream_chat()
            return
        if self.path == "/voice/prewarm/face":
            self._voice_face_prewarm()
            return
        if self.path == "/settings/clear-grounding-cache":
            self._settings_clear_grounding_cache()
            return
        if self.path == "/reflexes/regenerate":
            self._reflexes_regenerate()
            return
        if self.path == "/reflexes/validate":
            self._reflexes_validate()
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
        # ---- 3rd-party MCP importer/bridge (register / refresh) ----
        if self.path == "/api/v1/mcp/imports":
            self._mcp_imports_register()
            return
        if self.path.startswith("/api/v1/mcp/imports/") and self.path.endswith("/refresh"):
            self._mcp_imports_refresh(self.path[len("/api/v1/mcp/imports/"):-len("/refresh")])
            return
        # ---- HiveMind admin POSTs (mutations). Each route is audit-
        # logged; destructive ones (delete / force_stop / restore)
        # require body {"confirm": true} to guard against accidental
        # UI clicks.
        if self.path.startswith("/hivemind/vms/"):
            tail = self.path[len("/hivemind/vms/"):]
            vm_id, _, action = tail.partition("/")
            if action and self._dispatch_vm_action(vm_id, action):
                return
        if self.path.startswith("/hivemind/apps/"):
            tail = self.path[len("/hivemind/apps/"):]
            app_id, _, action = tail.partition("/")
            if action and self._dispatch_app_action(app_id, action):
                return
        if self.path == "/hivemind/vms/create_prebuilt":
            self._hivemind_vm_create_prebuilt()
            return
        if self.path == "/hivemind/storage/volumes":
            self._hivemind_storage_create_volume()
            return
        if self.path.startswith("/hivemind/storage/volumes/"):
            tail = self.path[len("/hivemind/storage/volumes/"):]
            volume_id, _, action = tail.partition("/")
            if action and self._dispatch_volume_action(volume_id, action):
                return
        if self.path.startswith("/hivemind/storage/snapshots/"):
            tail = self.path[len("/hivemind/storage/snapshots/"):]
            snapshot_id, _, action = tail.partition("/")
            if action and self._dispatch_snapshot_action(snapshot_id, action):
                return
        if self.path == "/hivemind/storage/snapshots":
            self._hivemind_storage_create_snapshot()
            return
        if self.path == "/hivemind/network":
            self._hivemind_network_create()
            return
        if self.path.startswith("/hivemind/network/"):
            tail = self.path[len("/hivemind/network/"):]
            network_id, _, action = tail.partition("/")
            if action and self._dispatch_network_action(network_id, action):
                return
        if self.path == "/hivemind/gpu/mode":
            self._hivemind_gpu_set_mode()
            return
        if self.path == "/hivemind/gpu/vgpu":
            self._hivemind_gpu_create_vgpu()
            return
        # ---- GPU passthrough workflow (May 27 2026: GPU-P / DDA / vGPU) ----
        if self.path == "/hivemind/gpu/passthrough/prepare":
            self._hivemind_gpu_passthrough_prepare()
            return
        if self.path == "/hivemind/gpu/passthrough/vgpu":
            self._hivemind_gpu_passthrough_vgpu()
            return
        if self.path == "/hivemind/gpu/passthrough/game-stream-vm":
            self._hivemind_gpu_passthrough_game_stream_vm()
            return
        if self.path == "/hivemind/voice_identities/enroll":
            self._hivemind_voice_identity_enroll()
            return
        if self.path.startswith("/hivemind/voice_identities/"):
            tail = self.path[len("/hivemind/voice_identities/"):]
            identity_id, _, action = tail.partition("/")
            if action and self._dispatch_voice_identity_action(identity_id, action):
                return
        if self.path == "/hivemind/approval/request":
            self._hivemind_approval_request()
            return
        if self.path == "/hivemind/approval/notify":
            self._hivemind_approval_notify()
            return
        if self.path.startswith("/hivemind/maintenance/"):
            tail = self.path[len("/hivemind/maintenance/"):]
            service_name, _, action = tail.partition("/")
            if action == "enter":
                self._hivemind_maintenance_enter(service_name)
                return
            if action == "clear":
                self._hivemind_maintenance_clear(service_name)
                return
        if self.path == "/hivemind/ollama/control":
            self._hivemind_ollama_control()
            return
        # ---- New admin domains POST (May 26 2026) ----
        if self.path == "/hivemind/oracle/chat":
            self._hivemind_oracle_chat()
            return
        if self.path == "/hivemind/oracle/configure":
            self._hivemind_oracle_configure()
            return
        if self.path == "/hivemind/training/start":
            self._hivemind_training_start()
            return
        if self.path == "/hivemind/adapters/deploy":
            self._hivemind_adapter_deploy()
            return
        if self.path == "/hivemind/loadout/apply":
            self._hivemind_loadout_apply()
            return
        if self.path == "/hivemind/deploy/gim":
            self._hivemind_deploy_gim()
            return
        if self.path == "/hivemind/inference/chat":
            self._hivemind_inference_chat()
            return
        if self.path == "/hivemind/logos/optimize":
            self._hivemind_logos_optimize()
            return
        if self.path.startswith("/hivemind/logos/prompts/") and self.path.endswith("/fork"):
            prompt_id = self.path[len("/hivemind/logos/prompts/"):-len("/fork")]
            self._hivemind_logos_prompt_fork(prompt_id)
            return
        if self.path.startswith("/hivemind/logos/candidates/") and self.path.endswith("/promote"):
            candidate_id = self.path[len("/hivemind/logos/candidates/"):-len("/promote")]
            self._hivemind_logos_candidate_promote(candidate_id)
            return
        if self.path.startswith("/hivemind/services/"):
            tail = self.path[len("/hivemind/services/"):]
            svc, _, action = tail.partition("/")
            if action == "enable":
                self._hivemind_service_enable(svc)
                return
            if action == "disable":
                self._hivemind_service_disable(svc)
                return
            if action == "restart":
                self._hivemind_service_restart(svc)
                return
        if self.path == "/hivemind/jobs/cancel":
            self._hivemind_jobs_cancel()
            return
        # ---- Game-session POST (May 26 2026) ----
        if self.path == "/hivemind/game-sessions/plan":
            self._hivemind_game_session_plan()
            return
        if self.path == "/hivemind/game-sessions/run":
            self._hivemind_game_session_run()
            return
        if self.path == "/hivemind/game-sessions/plan-run":
            # Convenience: plan + run + collect evidence in one call.
            self._hivemind_game_session_plan_run()
            return
        if self.path.startswith("/hivemind/game-sessions/") and self.path.endswith("/cancel"):
            job_id = self.path[len("/hivemind/game-sessions/"):-len("/cancel")]
            self._hivemind_game_session_cancel(job_id)
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
            if suffix == "deliver":
                self._double_agent_deliver(job_id)
                return
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
        if path_only == "/voice/synthesize/stream":
            self._voice_synthesize_stream()
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
            _face_model = body.get("model_id") or body.get("model") or None
            record_face_model(_face_model)
            result = self.runner.chat(
                message,
                session_id=body.get("session_id") or None,
                model=_face_model,
                depth_model=body.get("depth_model_id") or body.get("depth_model") or None,
                client_id=body.get("client_id") or None,
            )
            if result.get("fail_closed"):
                _json_response(
                    self,
                    409,
                    {
                        "error": result.get("error") or "recommend lease failed closed",
                        "fail_closed": True,
                    },
                )
                return
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

    def _voice_face_prewarm(self) -> None:
        """Warm the exact Face model selected by the operator.

        The boot prewarm uses the automatic picker and can therefore warm a
        different model from a value restored by the browser.  Voice turns
        fail closed on this route so a cold selected model cannot cross the
        first-token watchdog and become audible as a canned fallback.
        """
        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length <= 0 or length > 4096:
                raise ValueError("request body must be 1 to 4096 bytes")
            raw = self.rfile.read(length)

            def reject_duplicate_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
                parsed: dict[str, Any] = {}
                for key, value in pairs:
                    if key in parsed:
                        raise ValueError(f"duplicate field: {key}")
                    parsed[key] = value
                return parsed

            body = json.loads(
                raw.decode("utf-8", errors="strict"),
                object_pairs_hook=reject_duplicate_fields,
            )
            if not isinstance(body, dict):
                raise ValueError("request body must be a JSON object")
            unknown = sorted(
                set(body) - {"model", "client_id", "generation", "operation"}
            )
            if unknown:
                raise ValueError(f"unknown fields: {', '.join(unknown)}")
            operation_present = "operation" in body
            operation = body.get("operation")
            invalidate_only = operation_present and operation == "invalidate"
            if operation_present and not invalidate_only:
                raise ValueError("operation must be invalidate when provided")
            if invalidate_only:
                if "model" in body:
                    raise ValueError("invalidate operation must not include model")
                requested = None
            else:
                requested = body.get("model")
                if not isinstance(requested, str):
                    raise ValueError("model must be a non-empty string")
                requested = requested.strip()
                if not requested:
                    raise ValueError("model must be a non-empty string")
                if len(requested) > 256:
                    raise ValueError("model exceeds 256 characters")
            client_id = body.get("client_id")
            if not isinstance(client_id, str):
                raise ValueError("client_id must be a non-empty string")
            client_id = client_id.strip()
            if not (
                1 <= len(client_id) <= 256
                and client_id.isascii()
                and all(char.isalnum() or char in "-_.:" for char in client_id)
            ):
                raise ValueError("client_id has an invalid format")
            generation = body.get("generation")
            if (
                isinstance(generation, bool)
                or not isinstance(generation, int)
                or generation < 1
                or generation > 9_007_199_254_740_991
            ):
                raise ValueError("generation must be a positive safe integer")
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            _json_response(
                self,
                400,
                {"error": str(exc), "warmed": False, "fail_closed": True},
            )
            return

        cancel_event, latest_generation = _begin_selected_face_prewarm(
            client_id,
            generation,
        )
        if cancel_event is None:
            stale_payload = {
                "error": "selected Face prewarm generation is stale",
                "warmed": False,
                "generation": generation,
                "latest_generation": latest_generation,
                "superseded": True,
                "fail_closed": True,
            }
            if invalidate_only:
                stale_payload.update({"operation": "invalidate", "invalidated": False})
            else:
                stale_payload["requested_model"] = requested
            _json_response(
                self,
                409,
                stale_payload,
            )
            return
        if invalidate_only:
            # Auto does not warm or record a concrete model.  It only advances
            # this page's server-side generation tombstone so an older exact
            # warm cannot become the keepwarm model after the UI has moved on.
            _finish_selected_face_prewarm(client_id, generation, cancel_event)
            _json_response(
                self,
                200,
                {
                    "warmed": False,
                    "invalidated": True,
                    "automatic": True,
                    "operation": "invalidate",
                    "generation": generation,
                },
            )
            return
        result_box: dict[str, Any] = {}
        absolute_timeout = _selected_face_prewarm_timeout_s()
        request_timeout = max(5, int(absolute_timeout - 5))
        latency_budget_ms = selected_face_admission_latency_budget_ms()

        def run_exact_prewarm() -> None:
            try:
                result_box["result"] = dict(
                    prewarm_face_lobe_model(
                        hivemind_url=self.runner.hivemind_url,
                        model=requested,
                        timeout=request_timeout,
                        exact_model=True,
                        cancel_event=cancel_event,
                        latency_budget_ms=latency_budget_ms,
                        face_chat=getattr(self.runner, "face_lobe_chat", None),
                        lease_scope_id=client_id,
                    )
                    or {}
                )
            except Exception as exc:
                result_box["result"] = {"error": str(exc), "warmed": False}

        worker = threading.Thread(
            target=run_exact_prewarm,
            daemon=True,
            name="ms4-selected-face-prewarm",
        )
        worker.start()
        worker.join(timeout=absolute_timeout)
        if worker.is_alive():
            cancel_event.set()
            worker.join(timeout=1.0)
            worker_retired = not worker.is_alive()
            _finish_selected_face_prewarm(client_id, generation, cancel_event)
            append_event(
                "face_model_prewarm_failed",
                {
                    "requested_model": requested,
                    "reason": "absolute_timeout",
                    "worker_retired": worker_retired,
                },
            )
            _json_response(
                self,
                504,
                {
                    "error": (
                        "selected Face model prewarm exceeded "
                        f"{absolute_timeout:g} seconds"
                    ),
                    "warmed": False,
                    "requested_model": requested,
                    "worker_retired": worker_retired,
                    "fail_closed": True,
                },
            )
            return
        result = dict(result_box.get("result") or {})

        if not _selected_face_prewarm_is_current(
            client_id,
            generation,
            cancel_event,
        ):
            _json_response(
                self,
                409,
                {
                    "error": "selected Face prewarm was superseded",
                    "warmed": False,
                    "requested_model": requested,
                    "generation": generation,
                    "superseded": True,
                    "fail_closed": True,
                },
            )
            return

        effective = str(result.get("model") or "").strip()
        resolved_requested = str(result.get("requested_model") or "").strip()
        admission_profile = str(result.get("admission_profile") or "").strip()
        raw_first_token_ms = result.get("first_token_ms")
        first_token_valid = bool(
            isinstance(raw_first_token_ms, int)
            and not isinstance(raw_first_token_ms, bool)
            and raw_first_token_ms >= 0
        )
        first_token_ms = raw_first_token_ms if first_token_valid else None
        raw_result_budget_ms = result.get("latency_budget_ms")
        result_budget_valid = bool(
            isinstance(raw_result_budget_ms, int)
            and not isinstance(raw_result_budget_ms, bool)
            and raw_result_budget_ms > 0
            and raw_result_budget_ms == latency_budget_ms
        )
        exact = bool(
            result.get("warmed") is True
            and result.get("completed") is True
            and result.get("cancelled") is not True
            and int(result.get("reply_len") or 0) > 0
            and effective
            and resolved_requested
            and effective == resolved_requested
            and not result.get("fallback_used")
            and (requested is None or resolved_requested == requested)
            and admission_profile == SELECTED_FACE_ADMISSION_PROFILE
            and result.get("latency_admitted") is True
            and first_token_valid
            and first_token_ms is not None
            and first_token_ms <= latency_budget_ms
            and result_budget_valid
        )
        if not exact:
            _finish_selected_face_prewarm(client_id, generation, cancel_event)
            append_event(
                "face_model_prewarm_failed",
                {
                    "requested_model": requested,
                    "resolved_requested_model": resolved_requested or None,
                    "effective_model": effective or None,
                    "fallback_used": bool(result.get("fallback_used")),
                    "admission_profile": admission_profile or None,
                    "first_token_ms": first_token_ms,
                    "latency_budget_ms": latency_budget_ms,
                    "latency_admitted": False,
                    **_public_recommend_audit(result),
                },
            )
            _json_response(
                self,
                503,
                {
                    "error": str(result.get("error") or "exact Face model did not warm"),
                    "warmed": False,
                    "requested_model": requested,
                    "resolved_requested_model": resolved_requested or None,
                    "effective_model": effective or None,
                    "fallback_used": bool(result.get("fallback_used")),
                    "admission_profile": admission_profile or None,
                    "first_token_ms": first_token_ms,
                    "latency_budget_ms": latency_budget_ms,
                    "latency_admitted": False,
                    "fail_closed": True,
                },
            )
            return

        if not _commit_selected_face_prewarm(
            client_id,
            generation,
            cancel_event,
            effective,
        ):
            _json_response(
                self,
                409,
                {
                    "error": "selected Face prewarm was superseded before commit",
                    "warmed": False,
                    "requested_model": requested,
                    "generation": generation,
                    "superseded": True,
                    "fail_closed": True,
                },
            )
            return
        append_event(
            "face_model_prewarmed",
            {
                "requested_model": requested,
                "effective_model": effective,
                "reply_len": int(result.get("reply_len") or 0),
                "generation": generation,
                "admission_profile": SELECTED_FACE_ADMISSION_PROFILE,
                "first_token_ms": first_token_ms,
                "latency_budget_ms": latency_budget_ms,
                "latency_admitted": True,
                **_public_recommend_audit(result),
            },
        )
        success_payload = {
            "warmed": True,
            "requested_model": requested,
            "resolved_requested_model": resolved_requested,
            "effective_model": effective,
            "fallback_used": False,
            "reply_len": int(result.get("reply_len") or 0),
            "admission_profile": SELECTED_FACE_ADMISSION_PROFILE,
            "first_token_ms": first_token_ms,
            "latency_budget_ms": latency_budget_ms,
            "latency_admitted": True,
            "generation": generation,
        }
        recommend = result.get("recommend")
        if isinstance(recommend, dict) and recommend:
            success_payload["recommend"] = recommend
        _json_response(
            self,
            200,
            success_payload,
        )

    def _read_voice_turn_input(self) -> tuple[bytes, str, str | None]:
        content_type = self.headers.get("Content-Type", "") or ""
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length) if length > 0 else b""
        media_type = content_type.split(";", 1)[0].strip().lower()
        if media_type == "application/json":
            transcript = _parse_voice_transcript_envelope(body)
            return b"", "transcript.json", transcript
        if (
            media_type == "multipart/form-data"
            and _multipart_contains_transcript_fields(content_type, body)
        ):
            raise VoiceRequestError(
                "mixed multipart audio and transcript-ready fields are not accepted"
            )
        audio, filename = parse_audio_request(content_type, body)
        return audio, filename, None

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
            _json_response(self, 503, _voice_error_payload(exc, fail_closed=True))
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
            _json_response(self, 503, _voice_error_payload(exc, fail_closed=True))
        except Exception as exc:
            _json_response(self, 500, {"error": str(exc)})

    def _voice_synthesize_stream(self) -> None:
        """Sentence-chunked, cancel-aware TTS for already-produced text."""
        try:
            body = _read_json(self) or {}
            text = str(body.get("text") or "").strip()
            if not text:
                _json_response(self, 400, {"error": "text is required"})
                return
            if len(text) > 20_000:
                _json_response(self, 400, {"error": "text exceeds 20000 characters"})
                return
            tts_model = body.get("model")
            tts_voice = body.get("voice")
            response_format = body.get("response_format") or DEFAULT_TTS_FORMAT
            events: queue.Queue[_VoiceSseDelivery] = queue.Queue()
            client_alive = threading.Event()
            client_alive.set()
            cancel_event = threading.Event()
            worker_done = threading.Event()
            lifecycle: dict[str, int] = {}
            audio_errors = [0]

            def emit(event: str, payload: dict[str, Any]) -> bool:
                if not client_alive.is_set() or cancel_event.is_set():
                    return False
                if event == "audio_error":
                    audio_errors[0] += 1
                delivery = _VoiceSseDelivery(event=event, payload=dict(payload))
                events.put(delivery)
                if not delivery.acknowledged.wait(_voice_sse_write_ack_timeout()):
                    client_alive.clear()
                    cancel_event.set()
                    return False
                if not delivery.client_written:
                    client_alive.clear()
                    cancel_event.set()
                    return False
                if event == "audio_error":
                    cancel_event.set()
                    return False
                return True

            def worker() -> None:
                try:
                    chunks = _emit_text_as_parallel_chunks(
                        text=text,
                        emit=emit,
                        hivemind_url=self.runner.hivemind_url,
                        tts_model=tts_model,
                        tts_voice=tts_voice,
                        response_format=response_format,
                        cancel_event=cancel_event,
                        client_alive=client_alive,
                        lifecycle=lifecycle,
                    )
                    if client_alive.is_set() and (
                        cancel_event.is_set() or audio_errors[0] or chunks <= 0
                    ):
                        events.put(_VoiceSseDelivery(event="error", payload={
                            "error": "Text synthesis did not produce a complete audio stream.",
                            "fail_closed": True,
                            "completed": False,
                            "audio_chunks": chunks,
                            "audio_errors": audio_errors[0],
                            "lifecycle": lifecycle,
                        }, terminal=True))
                    elif client_alive.is_set() and not cancel_event.is_set():
                        events.put(_VoiceSseDelivery(event="done", payload={
                            "schema": "Ms4TextSynthesisStream.v1",
                            "completed": True,
                            "audio_chunks": chunks,
                            "lifecycle": lifecycle,
                        }, terminal=True))
                except Exception as exc:
                    if client_alive.is_set():
                        events.put(_VoiceSseDelivery(event="error", payload={
                            "error": str(exc),
                            "fail_closed": True,
                            "completed": False,
                        }, terminal=True))
                finally:
                    worker_done.set()

            _sse_start(self)
            if not _sse_event(
                self,
                "status",
                {"status": "synthesizing"},
                write_timeout=_voice_sse_write_ack_timeout(),
            ):
                client_alive.clear()
                cancel_event.set()
                return
            thread = threading.Thread(
                target=worker,
                daemon=True,
                name="ms4-text-synthesis-stream",
            )
            thread.start()
            while True:
                try:
                    delivery = events.get(timeout=0.25)
                except queue.Empty:
                    if worker_done.is_set():
                        if _sse_event(self, "error", {
                            "error": "Text synthesis worker ended without a terminal result.",
                            "fail_closed": True,
                            "completed": False,
                        }, write_timeout=_voice_sse_write_ack_timeout()):
                            client_alive.clear()
                            self.close_connection = True
                        return
                    if not _sse_event(
                        self,
                        "heartbeat",
                        {"status": "synthesizing"},
                        write_timeout=_voice_sse_write_ack_timeout(),
                    ):
                        client_alive.clear()
                        cancel_event.set()
                        thread.join(timeout=2.0)
                        return
                    continue
                written = _sse_event(
                    self,
                    delivery.event,
                    delivery.payload,
                    write_timeout=_voice_sse_write_ack_timeout(),
                )
                delivery.client_written = written
                delivery.acknowledged.set()
                if not written:
                    client_alive.clear()
                    cancel_event.set()
                    thread.join(timeout=2.0)
                    return
                if delivery.terminal:
                    client_alive.clear()
                    self.close_connection = True
                    return
        except VoiceRequestError as exc:
            _json_response(self, 400, {"error": str(exc)})
        except VoiceUnavailable as exc:
            _json_response(self, 503, _voice_error_payload(exc, fail_closed=True))
        except Exception as exc:
            _json_response(self, 500, {"error": str(exc)})

    def _voice_turn(self) -> None:
        try:
            audio, filename = self._read_audio_body()
            params = self._parse_query()
            session_id = params.get("session_id") or None
            client_id = params.get("client_id") or None
            # ?model= is the Face Lobe chat model; ?tts_model= is the
            # TTS model (tts-1 / tts-1-hd). See _voice_turn_stream for
            # the bug history.
            model = params.get("model") or params.get("model_id") or None
            depth_model = params.get("depth_model") or params.get("depth_model_id") or None
            tts_model = params.get("tts_model") or None
            tts_voice = params.get("voice") or None
            tts_format = params.get("response_format") or None
            result = voice_ptt_turn(
                runner=self.runner,
                audio=audio,
                filename=filename,
                session_id=session_id,
                model=model,
                depth_model=depth_model,
                tts_model=tts_model,
                tts_voice=tts_voice,
                response_format=tts_format,
                client_id=client_id,
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
            _json_response(self, 503, _voice_error_payload(exc, fail_closed=True))
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
            audio, filename, transcript_ready = self._read_voice_turn_input()
        except VoiceRequestError as exc:
            _json_response(self, 400, {"error": str(exc)})
            return
        except Exception as exc:
            _json_response(self, 500, {"error": str(exc)})
            return

        transcribe_model_override: str | None = None
        transcribe_fn_override = None
        if transcript_ready is not None:
            transcribe_model_override = _VOICE_TRANSCRIPT_MODEL

            def accepted_transcript(**_kwargs: Any) -> dict[str, str]:
                return {
                    "text": transcript_ready,
                    "model": _VOICE_TRANSCRIPT_MODEL,
                }

            transcribe_fn_override = accepted_transcript

        params = self._parse_query()
        session_id = params.get("session_id") or None
        client_id = params.get("client_id") or None
        turn_id = _new_voice_turn_id()
        # ?model= is the FACE LOBE chat model (e.g. phi4-mini). The TTS
        # model (tts-1 / tts-1-hd) is a separate knob — earlier the UI
        # was mistakenly sending ?model=tts-1 which made the Face Lobe
        # try to use tts-1 as a chat model, causing every voice turn
        # to fall through to the canned-failure reply.
        model = params.get("model") or params.get("model_id") or None
        depth_model = params.get("depth_model") or params.get("depth_model_id") or None
        tts_model = params.get("tts_model") or None
        tts_voice = params.get("voice") or None
        tts_format = params.get("response_format") or None
        # Per-turn override for the TTS engine. UI Settings dialog
        # passes ?engine=rest or ?engine=ws_super; missing/blank falls
        # back to MS4_VOICE_TTS_ENGINE env default.
        engine_override = params.get("engine") or None

        events: queue.Queue[_VoiceSseDelivery] = queue.Queue()
        # When the browser aborts the fetch (barge-in / new turn) the
        # _sse_event write raises and returns False. We flip this event
        # to signal voice_ptt_turn_stream to stop scheduling more TTS
        # work and tear down its WS engine immediately — otherwise we'd
        # keep paying HiveMind cycles for audio nobody is listening to.
        client_alive = threading.Event()
        client_alive.set()
        turn_cancel = threading.Event()
        delivery_lock = threading.Lock()
        delivery_counts = {
            "gateway_enqueued": 0,
            "client_written": 0,
            "write_failures": 0,
            "write_ack_timeouts": 0,
        }
        terminal_lock = threading.Lock()
        terminal_queued = [False]

        def cancel_turn() -> None:
            client_alive.clear()
            turn_cancel.set()

        def delivery_snapshot() -> dict[str, int]:
            with delivery_lock:
                return dict(delivery_counts)

        def queue_terminal(event: str, payload: dict[str, Any]) -> bool:
            with terminal_lock:
                if terminal_queued[0]:
                    return False
                terminal_queued[0] = True
            terminal_payload = dict(payload)
            terminal_payload.setdefault("turn_id", turn_id)
            item = _VoiceSseDelivery(
                event=event,
                payload=terminal_payload,
                terminal=True,
            )
            with delivery_lock:
                delivery_counts["gateway_enqueued"] += 1
            events.put(item)
            return True

        def emit(event: str, payload: dict[str, Any]) -> bool:
            if not client_alive.is_set() or turn_cancel.is_set():
                return False
            event_payload = dict(payload)
            event_payload.setdefault("turn_id", turn_id)
            item = _VoiceSseDelivery(event=event, payload=event_payload)
            with delivery_lock:
                delivery_counts["gateway_enqueued"] += 1
            events.put(item)
            if not item.acknowledged.wait(timeout=_voice_sse_write_ack_timeout()):
                with delivery_lock:
                    delivery_counts["write_ack_timeouts"] += 1
                cancel_turn()
                return False
            return item.client_written

        def worker() -> None:
            try:
                result = voice_ptt_turn_stream(
                    runner=self.runner,
                    audio=audio,
                    filename=filename,
                    session_id=session_id,
                    model=model,
                    depth_model=depth_model,
                    transcribe_model=transcribe_model_override,
                    tts_model=tts_model,
                    tts_voice=tts_voice,
                    response_format=tts_format,
                    engine=engine_override,
                    emit=emit,
                    client_alive=client_alive,
                    cancel_event=turn_cancel,
                    transcribe_fn=transcribe_fn_override,
                    client_id=client_id,
                    turn_id=turn_id,
                )
                if turn_cancel.is_set():
                    raise VoiceUnavailable(
                        "voice stream delivery was cancelled before terminal acknowledgement"
                    )
                # Per-turn structured log: everything an operator needs
                # to debug a slow / wrong / silent turn after the fact.
                # Written to the audit log (jsonl) so `/audit?limit=N`
                # and the Settings "Recent voice turns" panel surface
                # it. Keep the payload small — full per-chunk timings
                # live in result["metrics"] which we drop here.
                try:
                    metrics = (result or {}).get("metrics") or {}
                    append_event("voice_turn_complete", {
                        "turn_id": (result or {}).get("turn_id") or turn_id,
                        "session_id": (result or {}).get("session_id"),
                        "revision_id": (result or {}).get("revision_id"),
                        "transcript": (result or {}).get("transcript", "")[:200],
                        "reply_text_preview": (result or {}).get("reply_text", "")[:200],
                        "reply_text_chars": len((result or {}).get("reply_text", "") or ""),
                        "engine": metrics.get("engine"),
                        "asr_ms": metrics.get("asr_ms"),
                        "chat_ms": metrics.get("chat_ms"),
                        "total_ms": metrics.get("total_ms"),
                        "first_text_token_ms": metrics.get("first_text_token_ms"),
                        "first_audio_chunk_ms": metrics.get("first_audio_chunk_ms"),
                        "audio_chunks": metrics.get("audio_chunks"),
                        "audio_errors": metrics.get("audio_errors"),
                        "rest_fallback_used": metrics.get("rest_fallback_used"),
                        "tts_parallelism_speedup": (metrics.get("tts_parallelism") or {}).get("speedup_ratio"),
                        "any_held_for_inorder": (metrics.get("tts_parallelism") or {}).get("any_held_for_inorder"),
                        "max_held_for_inorder_ms": (metrics.get("tts_parallelism") or {}).get("max_held_for_inorder_ms"),
                        "foreground_model": (result or {}).get("foreground_model"),
                        "router": (result or {}).get("router"),
                        "dispatched_job": (result or {}).get("dispatched_job"),
                        "grounding_source": (result or {}).get("grounding_source"),
                        "tts_model": (result or {}).get("tts_model"),
                        "transcription_model": (result or {}).get("transcription_model"),
                        "delivery": delivery_snapshot(),
                        **_public_recommend_audit(result),
                    })
                except Exception as exc:
                    log.warning("voice_turn_complete audit log failed: %s", exc)
                metrics = (result or {}).setdefault("metrics", {})
                if isinstance(metrics, dict):
                    metrics["delivery"] = delivery_snapshot()
                queue_terminal("done", result)
            except VoiceRequestError as exc:
                append_event("voice_turn_failed", {"turn_id": turn_id, "kind": "request_error", "error": str(exc)})
                queue_terminal("error", {"error": str(exc)})
            except VoiceUnavailable as exc:
                append_event("voice_turn_failed", {"turn_id": turn_id, "kind": "fail_closed", "error": str(exc)})
                queue_terminal("error", _voice_error_payload(exc, fail_closed=True))
            except Exception as exc:
                append_event("voice_turn_failed", {"turn_id": turn_id, "kind": "exception", "error": str(exc)})
                queue_terminal("error", {"error": str(exc)})

        _sse_start(self)
        if not _sse_event(
            self,
            "status",
            {"status": "started", "turn_id": turn_id},
            write_timeout=_voice_sse_write_ack_timeout(),
        ):
            cancel_turn()
            return
        thread = threading.Thread(target=worker, daemon=True, name="ms4-voice-stream")
        thread.start()
        terminal_sent = False

        def retire_after_write_failure(failed_item: _VoiceSseDelivery | None) -> None:
            nonlocal terminal_sent
            cancel_turn()
            if failed_item is not None:
                failed_item.client_written = False
                failed_item.acknowledged.set()
            thread.join(timeout=_voice_worker_cleanup_timeout())
            worker_leaked = thread.is_alive()
            if worker_leaked:
                log.error(
                    "voice worker did not retire within %.2fs after client write failure",
                    _voice_worker_cleanup_timeout(),
                )

            terminal_item: _VoiceSseDelivery | None = None
            while True:
                try:
                    queued = events.get_nowait()
                except queue.Empty:
                    break
                if queued.terminal and terminal_item is None:
                    terminal_item = queued
                else:
                    queued.client_written = False
                    queued.acknowledged.set()

            if failed_item is not None and failed_item.terminal:
                terminal_sent = True
                return
            if terminal_item is None:
                payload = {
                    "error": "Voice stream stopped after client write failure.",
                    "fail_closed": True,
                    "worker_join_timeout": worker_leaked,
                }
                if queue_terminal("error", payload):
                    terminal_item = events.get_nowait()
            if terminal_item is not None and not terminal_sent:
                terminal_sent = True
                _sse_event(
                    self,
                    terminal_item.event,
                    terminal_item.payload,
                    write_timeout=_voice_sse_write_ack_timeout(),
                )
                terminal_item.acknowledged.set()

        while True:
            try:
                item = events.get(timeout=2.0)
            except queue.Empty:
                if not _sse_event(
                    self,
                    "heartbeat",
                    {"status": "running", "turn_id": turn_id},
                    write_timeout=_voice_sse_write_ack_timeout(),
                ):
                    with delivery_lock:
                        delivery_counts["write_failures"] += 1
                    retire_after_write_failure(None)
                    return
                continue
            written = _sse_event(
                self,
                item.event,
                item.payload,
                write_timeout=_voice_sse_write_ack_timeout(),
            )
            item.client_written = written
            with delivery_lock:
                if written:
                    delivery_counts["client_written"] += 1
                else:
                    delivery_counts["write_failures"] += 1
            item.acknowledged.set()
            if not written:
                retire_after_write_failure(item)
                return
            if item.terminal:
                terminal_sent = True
                thread.join(timeout=_voice_worker_cleanup_timeout())
                if thread.is_alive():
                    log.error(
                        "voice worker still alive %.2fs after terminal write",
                        _voice_worker_cleanup_timeout(),
                    )
                self.close_connection = True
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

        # ``runner.default_model`` belongs to the Depth worker.  Face and
        # Depth intentionally have separate defaults now, so a direct gateway
        # launch (without runtime_common seeding the environment) must not
        # report the 30B Depth fallback as Face configuration.
        configured_default_model = (
            os.environ.get("MS4_DEFAULT_MODEL") or DEFAULT_FOREGROUND_MODEL
        )
        try:
            fallback_choice = choose_foreground_model(
                hivemind_url=self.runner.hivemind_url,
                force_refresh=False,
            )
            fallback_model = fallback_choice.model_id
            fallback_source = fallback_choice.source
            fallback_detail = fallback_choice.detail
        except Exception as exc:
            fallback_model = None
            fallback_source = "error"
            fallback_detail = str(exc)[:240]

        depth_override = os.environ.get(MS4_DEPTH_MODEL_ENV, "").strip() or None
        depth_fallback = depth_fallback_model()
        try:
            depth_choice = choose_depth_model(
                hivemind_url=self.runner.hivemind_url,
                force_refresh=False,
            )
            depth_selection = {
                "model_id": depth_choice.model_id,
                "source": depth_choice.source,
                "detail": depth_choice.detail,
                "policy_tier": (
                    "quality_target"
                    if depth_choice.source == "loaded"
                    else "explicit_override"
                    if depth_choice.source in {"env_override", "envelope_override"}
                    else "degraded_fast_tool_fallback"
                    if depth_choice.model_id == DEPTH_FAST_TOOL_FALLBACK_MODEL
                    else "configured_fallback"
                ),
            }
        except Exception as exc:
            depth_selection = {
                "model_id": None,
                "source": "error",
                "detail": str(exc)[:240],
                "policy_tier": "unavailable",
            }

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
                "default_model": configured_default_model,
                "configured_default_model": configured_default_model,
                "fallback_model": fallback_model,
                "fallback_source": fallback_source,
                "fallback_detail": fallback_detail,
                "foreground_override": os.environ.get("MS4_FOREGROUND_MODEL") or None,
                "serving_scope": "hivemind_cluster",
            },
            "depth_lobe": {
                # Durable policy intent is distinct from the readiness-driven
                # selection below. A cold preferred model must not disappear
                # from the operator snapshot merely because the safe fallback
                # is serving the current turn.
                "preferred_cluster_target": DEPTH_PREFERRED_CLUSTER_TARGET,
                "quality_target_model": DEPTH_PREFERRED_CLUSTER_TARGET,
                "automatic_selection": depth_selection,
                "configured_override": depth_override,
                "fallback_model": depth_fallback,
                "fallback_policy_tier": (
                    "degraded_fast_tool_fallback"
                    if depth_fallback == DEPTH_FAST_TOOL_FALLBACK_MODEL
                    else "configured_fallback"
                ),
                "fallback_minimum_total_parameters_b": DEPTH_FALLBACK_MIN_TOTAL_PARAM_B,
                "minimum_total_parameters_b": DEPTH_MIN_TOTAL_PARAM_B,
                "minimum_target_total_parameters_b": DEPTH_MIN_TOTAL_PARAM_B,
                "serving_scope": "hivemind_cluster",
            },
        }
        _json_response(self, 200, payload)

    # ------------------------------------------------------------------
    # Voice services lifecycle (HiveMind ASR / TTS / TTS_SUPER)
    # ------------------------------------------------------------------

    def _voice_status_get(self) -> None:
        """Effective voice-readiness view for the Oracle front door.

        Voice turns already route through ``check_voice_ready`` so healthy
        HiveMind ASR can keep working when MS3's cached readiness is stale.
        The UI status route must report the same effective readiness; a raw
        MS3 proxy makes the mic look broken even when the turn path can use
        HiveMind ASR directly.
        """
        try:
            payload = check_voice_ready(
                self.runner.ms3_url,
                hivemind_url=self.runner.hivemind_url,
                timeout=5,
                retries=1,
            )
            out = dict(payload) if isinstance(payload, dict) else {"voice": payload}
            out.setdefault("schema", "VoiceReadiness.v1")
            out.setdefault("voice_input_ready", True)
            out.setdefault("effective_source", out.get("source") or "ms3")
            _json_response(self, 200, out)
            return
        except VoiceUnavailable as exc:
            status, payload = _proxy_json(f"{self.runner.ms3_url}/voice/status", timeout=10)
            out = dict(payload) if isinstance(payload, dict) else {"voice": payload}
            out.setdefault("schema", "VoiceReadiness.v1")
            out["voice_input_ready"] = False
            out.setdefault("effective_source", "ms3")
            out["effective_detail"] = str(exc)[:400]
            try:
                services = list_voice_services(self.runner.hivemind_url)
                asr = next((svc for svc in services if svc.get("service") == "ASR"), None)
                if asr is not None:
                    out["hivemind_asr"] = {
                        "healthy": bool(asr.get("healthy")),
                        "detail": asr.get("detail"),
                        "endpoint": asr.get("endpoint"),
                        "endpoints_configured": asr.get("endpoints_configured"),
                        "provisioning_state": asr.get("provisioning_state"),
                    }
            except Exception as svc_exc:  # noqa: BLE001 - status route is diagnostic
                out["hivemind_asr_error"] = str(svc_exc)[:240]
            _json_response(self, 200 if status == 200 else status, out)
            return

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

    def _easter_honorable(self) -> None:
        """Serve the "Honorable" easter-egg audio clip (BIA - WE ON GO).

        The browser can't read an arbitrary local file path, so the
        gateway streams the configured clip over HTTP. Path is fixed via
        ``MS4_VOICE_EGG_HONORABLE_CLIP`` (NOT user input -> no traversal
        risk). 404 if the file is missing, in which case the UI falls back
        to MS4 simply saying the line. Content type is inferred from the
        extension (mp3 -> audio/mpeg)."""
        default_clip = r"C:\Users\nexus-hc-win-00\Downloads\BIA - WE ON GO (Official Audio).mp3"
        clip_path = os.environ.get("MS4_VOICE_EGG_HONORABLE_CLIP", default_clip)
        try:
            with open(clip_path, "rb") as fh:
                data = fh.read()
        except FileNotFoundError:
            _json_response(self, 404, {"error": "easter-egg clip not found", "path": clip_path})
            return
        except Exception as exc:
            _json_response(self, 500, {"error": str(exc)})
            return
        ext = os.path.splitext(clip_path)[1].lower()
        mime = {
            ".mp3": "audio/mpeg", ".wav": "audio/wav", ".ogg": "audio/ogg",
            ".m4a": "audio/mp4", ".aac": "audio/aac", ".flac": "audio/flac",
        }.get(ext, "application/octet-stream")
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "public, max-age=300")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

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

    def _reflexes_validate(self) -> None:
        """Round-trip QA pass: validate existing renders (transcribe back +
        LLM judge) and re-render any that fail (with an auto-rephrased,
        TTS-stable equivalent). Does NOT force-rerender good ones.

        Body (optional): ``{voice, async}``. Async returns 202 and the QA
        runs in the background; poll GET /reflexes for `validated`/`heard`.
        """
        try:
            body = _read_json(self) if (self.headers.get("Content-Length") or "0") != "0" else {}
        except Exception:
            body = {}
        body = body if isinstance(body, dict) else {}
        voice = body.get("voice") or DEFAULT_REFLEX_VOICE
        run_async = bool(body.get("async", True))  # default async — QA is slow (ASR per reflex)

        if run_async:
            generate_all_async(hivemind_url=self.runner.hivemind_url, voice=voice, force=False, qa=True)
            append_event("reflex_validate_requested", {"voice": voice, "async": True})
            _json_response(self, 202, {
                "schema": "Ms4ReflexValidateAck.v1", "started": True,
                "voice": voice, "async": True, "poll_url": "/reflexes",
            })
            return
        try:
            result = generate_all(hivemind_url=self.runner.hivemind_url, voice=voice, force=False, qa=True)
        except Exception as exc:
            _json_response(self, 502, {"error": str(exc)})
            return
        append_event("reflex_validate_completed", {
            "voice": result.get("voice"),
            "regenerated": [g["id"] for g in (result.get("generated") or [])],
            "revalidated": result.get("revalidated"),
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

    # ------------------------------------------------------------------
    # /voice/recent-turns — per-turn diagnostics for the Settings UI
    # ------------------------------------------------------------------

    def _voice_recent_turns_get(self) -> None:
        """Return the most recent ``voice_turn_complete`` and
        ``voice_turn_failed`` audit events so the UI can render a
        per-turn diagnostics panel. Bounded reverse scan + ring; newest first."""
        limit = 25
        try:
            if "?" in self.path:
                qs = self.path.split("?", 1)[1]
                for part in qs.split("&"):
                    key, _, value = part.partition("=")
                    if key == "limit":
                        try:
                            limit = max(1, min(200, int(value)))
                        except ValueError:
                            limit = 25
            payload = read_recent_voice_turns(limit=limit)
            if "schema" not in payload:
                payload["schema"] = "Ms4VoiceRecentTurns.v1"
            _json_response(self, 200, payload)
        except Exception as exc:
            _json_response(self, 200, {
                "schema": "Ms4VoiceRecentTurns.v1",
                "turns": [],
                "limit": limit,
                "source": "error",
                "error": str(exc),
                "complete": False,
                "stale": False,
            })

    # ------------------------------------------------------------------
    # HiveMind admin shared error mapping
    # ------------------------------------------------------------------

    def _emit_admin_error(self, exc: Exception) -> None:
        """Map any of the HiveMind admin error types to a 502 (transport
        / cluster) or 500 (programmer error). The UI shows the message
        as a red banner, so keep it actionable.
        """
        if isinstance(exc, (HivemindToolError, VmAdminError, AppAdminError,
                            StorageAdminError, NetworkAdminError,
                            GpuModeAdminError, VoiceIdentityError,
                            HumanApprovalError, OracleAdminError,
                            TrainingAdminError, AdapterAdminError,
                            LoadoutAdminError, GameAdminError,
                            GpuPassthroughError)):
            _json_response(self, 502, {"error": str(exc)})
            return
        if isinstance(exc, ValueError):
            _json_response(self, 400, {"error": str(exc)})
            return
        _json_response(self, 500, {"error": str(exc)})

    # ------------------------------------------------------------------
    # /hivemind/time — authoritative cluster time
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # /api/v1/mcp/imports — 3rd-party MCP server importer/bridge
    # ------------------------------------------------------------------

    def _mcp_imports_list(self) -> None:
        from machine_spirit_4.mcp_bridge.registry import load

        reg = load()
        _json_response(self, 200, {
            "schema": "Ms4McpImports.v1",
            "servers": [s.to_dict(include_tools=False) for s in reg.list_servers()],
            "tool_count": len(reg.all_tools()),
        })

    def _mcp_imports_register(self) -> None:
        from machine_spirit_4.mcp_bridge.registry import ImportedServer, load
        from machine_spirit_4.mcp_bridge.upstream_client import UpstreamError

        try:
            body = _read_json(self) or {}
        except (ValueError, json.JSONDecodeError):
            _json_response(self, 400, {"error": "invalid JSON body"})
            return
        server_id = body.get("server_id")
        transport = body.get("transport")
        if not server_id or transport not in ("stdio", "http"):
            _json_response(self, 400, {"error": "server_id and transport ('stdio'|'http') are required"})
            return
        server = ImportedServer(
            server_id=str(server_id),
            transport=str(transport),
            command=(str(body["command"]) if body.get("command") else None),
            args=[str(a) for a in (body.get("args") or [])],
            env={str(k): str(v) for k, v in (body.get("env") or {}).items()},
            env_passthrough=[str(n) for n in (body.get("env_passthrough") or []) if isinstance(n, str)],
            cwd=(str(body["cwd"]) if body.get("cwd") else None),
            url=(str(body["url"]) if body.get("url") else None),
            headers={str(k): str(v) for k, v in (body.get("headers") or {}).items()},
            allow_inline=bool(body.get("allow_inline", False)),
            enabled=bool(body.get("enabled", True)),
        )
        refresh = bool(body.get("refresh", True))
        reg = load()
        try:
            reg.register(server, refresh=refresh)
        except UpstreamError as exc:
            _json_response(self, 400, {"error": str(exc)})
            return
        except Exception as exc:  # noqa: BLE001
            _json_response(self, 500, {"error": f"register failed: {exc}"})
            return
        result = reg.get(server.server_id)
        _json_response(self, 200, result.to_dict() if result else {"error": "register did not persist"})

    def _mcp_imports_refresh(self, server_id: str) -> None:
        from machine_spirit_4.mcp_bridge.registry import load

        reg = load()
        if reg.get(server_id) is None:
            _json_response(self, 404, {"error": f"unknown imported server: {server_id}"})
            return
        server = reg.refresh(server_id)
        _json_response(self, 200, server.to_dict())

    def _mcp_imports_remove(self, server_id: str) -> None:
        from machine_spirit_4.mcp_bridge.registry import load

        reg = load()
        removed = reg.remove(server_id)
        _json_response(self, 200 if removed else 404, {"removed": removed, "server_id": server_id})

    def _mcp_imports_health(self, server_id: str) -> None:
        from machine_spirit_4.mcp_bridge.registry import load

        reg = load()
        server = reg.get(server_id)
        if server is None:
            _json_response(self, 404, {"error": f"unknown imported server: {server_id}"})
            return
        _json_response(self, 200, {
            "server_id": server.server_id,
            "transport": server.transport,
            "enabled": server.enabled,
            "allow_inline": server.allow_inline,
            "tool_count": len(server.tools),
            "last_refresh": server.last_refresh,
            "last_error": server.last_error,
        })

    def _hivemind_time_get(self) -> None:
        try:
            body = hivemind_tools.time_now(self.runner.hivemind_url)
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        _json_response(self, 200, body if isinstance(body, dict) else {"raw": body})

    def _hivemind_capability_matrix_get(self) -> None:
        try:
            body = hivemind_tools.capability_matrix(self.runner.hivemind_url)
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        _json_response(self, 200, body if isinstance(body, dict) else {"matrix": body})

    # ------------------------------------------------------------------
    # /hivemind/vms — VM lifecycle (read + actions)
    # ------------------------------------------------------------------

    def _hivemind_vms_get(self) -> None:
        try:
            snapshot = vm_admin.list_with_gpu_assignments(self.runner.hivemind_url)
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        if snapshot.get("errors"):
            append_event("hivemind_vms_partial_failure", {"errors": snapshot["errors"]})
        _json_response(self, 200, snapshot)

    def _hivemind_vm_gpus_get(self, vm_id: str) -> None:
        """``hivemind.vm.gpus@v1`` is cluster-wide (no per-VM filter
        per May-26 2026 contract). We still expose this per-VM route
        for UI ergonomics — filter client-side after fetching the
        full list.
        """
        try:
            raw = hivemind_tools.vm_gpus(self.runner.hivemind_url)
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        items = []
        if isinstance(raw, dict):
            for key in ("gpus", "data", "items"):
                if isinstance(raw.get(key), list):
                    items = raw[key]
                    break
        elif isinstance(raw, list):
            items = raw
        filtered = [g for g in items if isinstance(g, dict) and (
            g.get("vm") == vm_id or g.get("vm_name") == vm_id or g.get("assigned_to") == vm_id
        )]
        _json_response(self, 200, {"vm_id": vm_id, "gpus": filtered, "all_gpus": items})

    def _hivemind_vm_screenshot(self, vm_id: str) -> None:
        """``GET /hivemind/vms/<id>/screenshot?width=1280&height=720`` —
        optional width/height query string (defaults to 1280x720 per
        the May-26 2026 cluster contract requirement)."""
        try:
            width = 1280
            height = 720
            qs = self.path.split("?", 1)[1] if "?" in self.path else ""
            if qs:
                import urllib.parse
                params = urllib.parse.parse_qs(qs)
                if "width" in params:
                    try:
                        width = max(64, min(7680, int(params["width"][0])))
                    except ValueError:
                        pass
                if "height" in params:
                    try:
                        height = max(64, min(4320, int(params["height"][0])))
                    except ValueError:
                        pass
            body = vm_admin.get_screenshot(
                self.runner.hivemind_url, vm_id, width=width, height=height
            )
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        _json_response(self, 200, body if isinstance(body, dict) else {"raw": body})

    def _dispatch_vm_action(self, vm_id: str, action: str) -> bool:
        """Return True if we handled the action; False if the caller
        should fall through to the 404 default. Body must be JSON
        (possibly empty); destructive actions require ``confirm: true``.
        """
        try:
            body = _read_json(self) or {}
            confirm = bool(body.get("confirm"))
            if action == "start":
                result = vm_admin.start_vm(self.runner.hivemind_url, vm_id)
                append_event("hivemind_vm_start", {"vm_id": vm_id})
            elif action == "stop":
                result = vm_admin.stop_vm(self.runner.hivemind_url, vm_id)
                append_event("hivemind_vm_stop", {"vm_id": vm_id})
            elif action == "force_stop":
                result = vm_admin.force_stop_vm(self.runner.hivemind_url, vm_id, confirm=confirm)
                append_event("hivemind_vm_force_stop", {"vm_id": vm_id})
            elif action == "delete":
                result = vm_admin.delete_vm(self.runner.hivemind_url, vm_id, confirm=confirm)
                append_event("hivemind_vm_delete", {"vm_id": vm_id})
            elif action == "deploy":
                # Live HiveMind contract only takes ``name``; ``target_node``
                # is forwarded as a forward-compat opt (cluster ignores
                # unknown fields today).
                deploy_opts = {k: v for k, v in body.items() if k != "confirm"}
                result = vm_admin.deploy_vm(self.runner.hivemind_url, vm_id, **deploy_opts)
                append_event("hivemind_vm_deploy", {"vm_id": vm_id, "opts": deploy_opts})
            elif action == "undeploy":
                result = vm_admin.undeploy_vm(self.runner.hivemind_url, vm_id)
                append_event("hivemind_vm_undeploy", {"vm_id": vm_id})
            else:
                return False
        except Exception as exc:
            self._emit_admin_error(exc)
            return True
        _json_response(self, 200, result if isinstance(result, dict) else {"result": result})
        return True

    def _hivemind_vm_create_prebuilt(self) -> None:
        """``POST /hivemind/vms/create_prebuilt`` — body must include
        ``vm_type`` (template id) and ``name``. Accepts legacy
        ``template`` key as an alias for ``vm_type`` so older
        clients keep working."""
        try:
            body = _read_json(self) or {}
            # Backwards-compat: pre-May-26 clients sent {template, ...}.
            vm_type = body.pop("vm_type", None) or body.pop("template", None)
            name = body.pop("name", None)
            if not vm_type:
                _json_response(self, 400, {"error": "vm_type is required"})
                return
            if not name:
                _json_response(self, 400, {"error": "name is required"})
                return
            result = vm_admin.create_prebuilt(
                self.runner.hivemind_url, str(vm_type), str(name), **body
            )
            append_event(
                "hivemind_vm_create_prebuilt",
                {"vm_type": vm_type, "name": name, "opts": body},
            )
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        _json_response(self, 200, result if isinstance(result, dict) else {"result": result})

    # ------------------------------------------------------------------
    # /hivemind/apps
    # ------------------------------------------------------------------

    def _hivemind_apps_get(self) -> None:
        try:
            snapshot = app_admin.list_with_status_and_metrics(self.runner.hivemind_url)
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        if snapshot.get("errors"):
            append_event("hivemind_apps_partial_failure", {"errors": snapshot["errors"]})
        _json_response(self, 200, snapshot)

    def _hivemind_app_get(self, app_id: str) -> None:
        try:
            body = app_admin.get_app(self.runner.hivemind_url, app_id)
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        _json_response(self, 200, body)

    def _dispatch_app_action(self, app_id: str, action: str) -> bool:
        try:
            if action == "start":
                result = app_admin.start_app(self.runner.hivemind_url, app_id)
                append_event("hivemind_app_start", {"app_id": app_id})
            elif action == "stop":
                result = app_admin.stop_app(self.runner.hivemind_url, app_id)
                append_event("hivemind_app_stop", {"app_id": app_id})
            elif action == "status":
                result = app_admin.get_app_status(self.runner.hivemind_url, app_id)
            elif action == "metrics":
                result = app_admin.get_app_metrics(self.runner.hivemind_url, app_id)
            else:
                return False
        except Exception as exc:
            self._emit_admin_error(exc)
            return True
        _json_response(self, 200, result if isinstance(result, dict) else {"result": result})
        return True

    # ------------------------------------------------------------------
    # /hivemind/storage
    # ------------------------------------------------------------------

    def _hivemind_storage_get(self) -> None:
        try:
            snapshot = storage_admin.combined_snapshot(self.runner.hivemind_url)
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        if snapshot.get("errors"):
            append_event("hivemind_storage_partial_failure", {"errors": snapshot["errors"]})
        _json_response(self, 200, snapshot)

    def _hivemind_storage_create_volume(self) -> None:
        try:
            body = _read_json(self) or {}
            result = storage_admin.create_volume(self.runner.hivemind_url, **body)
            append_event("hivemind_storage_volume_create", {"opts": body})
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        _json_response(self, 200, result if isinstance(result, dict) else {"result": result})

    def _dispatch_volume_action(self, volume_id: str, action: str) -> bool:
        try:
            body = _read_json(self) or {}
            confirm = bool(body.get("confirm"))
            if action == "delete":
                result = storage_admin.delete_volume(self.runner.hivemind_url, volume_id, confirm=confirm)
                append_event("hivemind_storage_volume_delete", {"volume_id": volume_id})
            elif action == "attach":
                target = body.get("target")
                if not target:
                    _json_response(self, 400, {"error": "target required"})
                    return True
                result = storage_admin.attach_volume(self.runner.hivemind_url, volume_id, str(target))
                append_event("hivemind_storage_volume_attach", {"volume_id": volume_id, "target": target})
            elif action == "detach":
                result = storage_admin.detach_volume(self.runner.hivemind_url, volume_id)
                append_event("hivemind_storage_volume_detach", {"volume_id": volume_id})
            elif action == "resize":
                new_size = int(body.get("new_size_bytes") or 0)
                result = storage_admin.resize_volume(self.runner.hivemind_url, volume_id, new_size)
                append_event("hivemind_storage_volume_resize", {"volume_id": volume_id, "new_size_bytes": new_size})
            else:
                return False
        except Exception as exc:
            self._emit_admin_error(exc)
            return True
        _json_response(self, 200, result if isinstance(result, dict) else {"result": result})
        return True

    def _hivemind_storage_create_snapshot(self) -> None:
        try:
            body = _read_json(self) or {}
            volume_id = body.get("volume_id")
            if not volume_id:
                _json_response(self, 400, {"error": "volume_id required"})
                return
            label = body.get("label")
            result = storage_admin.create_snapshot(self.runner.hivemind_url, str(volume_id), label)
            append_event("hivemind_storage_snapshot_create", {"volume_id": volume_id, "label": label})
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        _json_response(self, 200, result if isinstance(result, dict) else {"result": result})

    def _dispatch_snapshot_action(self, snapshot_id: str, action: str) -> bool:
        try:
            body = _read_json(self) or {}
            confirm = bool(body.get("confirm"))
            if action == "delete":
                result = storage_admin.delete_snapshot(self.runner.hivemind_url, snapshot_id, confirm=confirm)
                append_event("hivemind_storage_snapshot_delete", {"snapshot_id": snapshot_id})
            elif action == "restore":
                result = storage_admin.restore_snapshot(self.runner.hivemind_url, snapshot_id, confirm=confirm)
                append_event("hivemind_storage_snapshot_restore", {"snapshot_id": snapshot_id})
            else:
                return False
        except Exception as exc:
            self._emit_admin_error(exc)
            return True
        _json_response(self, 200, result if isinstance(result, dict) else {"result": result})
        return True

    # ------------------------------------------------------------------
    # /hivemind/network
    # ------------------------------------------------------------------

    def _hivemind_network_get(self) -> None:
        try:
            snapshot = network_admin.combined_snapshot(self.runner.hivemind_url)
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        if snapshot.get("errors"):
            append_event("hivemind_network_partial_failure", {"errors": snapshot["errors"]})
        _json_response(self, 200, snapshot)

    def _hivemind_network_create(self) -> None:
        try:
            body = _read_json(self) or {}
            result = network_admin.create_network(self.runner.hivemind_url, **body)
            append_event("hivemind_network_create", {"opts": body})
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        _json_response(self, 200, result if isinstance(result, dict) else {"result": result})

    def _dispatch_network_action(self, network_id: str, action: str) -> bool:
        try:
            body = _read_json(self) or {}
            confirm = bool(body.get("confirm"))
            if action == "delete":
                result = network_admin.delete_network(self.runner.hivemind_url, network_id, confirm=confirm)
                append_event("hivemind_network_delete", {"network_id": network_id})
            elif action == "attach":
                target = body.get("target")
                if not target:
                    _json_response(self, 400, {"error": "target required"})
                    return True
                result = network_admin.attach(self.runner.hivemind_url, network_id, str(target))
                append_event("hivemind_network_attach", {"network_id": network_id, "target": target})
            elif action == "detach":
                target = body.get("target")
                if not target:
                    _json_response(self, 400, {"error": "target required"})
                    return True
                result = network_admin.detach(self.runner.hivemind_url, network_id, str(target))
                append_event("hivemind_network_detach", {"network_id": network_id, "target": target})
            elif action == "isolate":
                isolated = bool(body.get("isolated", True))
                result = network_admin.set_isolation(self.runner.hivemind_url, network_id, isolated)
                append_event("hivemind_network_isolate", {"network_id": network_id, "isolated": isolated})
            else:
                return False
        except Exception as exc:
            self._emit_admin_error(exc)
            return True
        _json_response(self, 200, result if isinstance(result, dict) else {"result": result})
        return True

    # ------------------------------------------------------------------
    # /hivemind/gpu
    # ------------------------------------------------------------------

    def _hivemind_gpu_get(self) -> None:
        try:
            snapshot = gpu_mode_admin.combined_snapshot(self.runner.hivemind_url)
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        if snapshot.get("errors"):
            append_event("hivemind_gpu_partial_failure", {"errors": snapshot["errors"]})
        _json_response(self, 200, snapshot)

    def _hivemind_gpu_set_mode(self) -> None:
        """``POST /hivemind/gpu/mode`` — body
        ``{gpu_pci_id, desired_mode, vm_uuid?}``. ``desired_mode``
        must be ``'vgpu'`` or ``'passthrough'`` per the live cluster
        contract. Accepts legacy ``{node_id, gpu_id, mode}`` keys for
        backwards compat — ``gpu_id`` is mapped to ``gpu_pci_id`` and
        ``mode`` is mapped to ``desired_mode``."""
        try:
            body = _read_json(self) or {}
            gpu_pci_id = str(
                body.get("gpu_pci_id") or body.get("gpu_id") or ""
            )
            desired_mode = str(
                body.get("desired_mode") or body.get("mode") or ""
            )
            vm_uuid = body.get("vm_uuid")
            if not (gpu_pci_id and desired_mode):
                _json_response(
                    self,
                    400,
                    {"error": "gpu_pci_id and desired_mode are required"},
                )
                return
            if desired_mode not in ("vgpu", "passthrough"):
                _json_response(
                    self,
                    400,
                    {
                        "error": "desired_mode must be 'vgpu' or 'passthrough'",
                        "note": "GPU-P (Hyper-V GPU Partitioning) is NOT a gpu_mode; use POST /hivemind/gpu/passthrough/game-stream-vm instead.",
                    },
                )
                return
            result = gpu_mode_admin.set_mode(
                self.runner.hivemind_url,
                gpu_pci_id=gpu_pci_id,
                desired_mode=desired_mode,
                vm_uuid=vm_uuid,
            )
            append_event(
                "hivemind_gpu_set_mode",
                {
                    "gpu_pci_id": gpu_pci_id,
                    "desired_mode": desired_mode,
                    "vm_uuid": vm_uuid,
                },
            )
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        _json_response(self, 200, result if isinstance(result, dict) else {"result": result})

    def _hivemind_gpu_create_vgpu(self) -> None:
        """``POST /hivemind/gpu/vgpu`` — body
        ``{gpu_pci_id, profile, count?}``. Accepts legacy
        ``{node_id, gpu_id, profile}`` keys for backwards compat —
        ``gpu_id`` is mapped to ``gpu_pci_id``."""
        try:
            body = _read_json(self) or {}
            gpu_pci_id = str(
                body.get("gpu_pci_id") or body.get("gpu_id") or ""
            )
            profile = str(body.get("profile") or "")
            count = int(body.get("count") or 1)
            if not (gpu_pci_id and profile):
                _json_response(
                    self,
                    400,
                    {"error": "gpu_pci_id and profile are required"},
                )
                return
            result = gpu_mode_admin.create_vgpu(
                self.runner.hivemind_url,
                gpu_pci_id=gpu_pci_id,
                profile=profile,
                count=count,
            )
            append_event(
                "hivemind_gpu_create_vgpu",
                {"gpu_pci_id": gpu_pci_id, "profile": profile, "count": count},
            )
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        _json_response(self, 200, result if isinstance(result, dict) else {"result": result})

    # ------------------------------------------------------------------
    # /hivemind/gpu/passthrough — GPU-P / DDA / vGPU workflow
    # (May 27 2026: GPU-P is the active path on consumer hardware;
    # DDA wiring is complete so it lights up when WS license arrives)
    # ------------------------------------------------------------------

    def _hivemind_gpu_passthrough_snapshot(self) -> None:
        try:
            snap = gpu_passthrough.snapshot(self.runner.hivemind_url)
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        if snap.get("errors"):
            append_event(
                "hivemind_gpu_passthrough_partial_failure",
                {"errors": snap["errors"]},
            )
        _json_response(self, 200, snap)

    def _hivemind_gpu_passthrough_prepare(self) -> None:
        """``POST /hivemind/gpu/passthrough/prepare`` — switch a GPU
        to ``passthrough`` (DDA path) or ``vgpu`` mode. Requires
        ``{gpu_pci_id, desired_mode, confirm:true}``."""
        try:
            body = _read_json(self) or {}
            gpu_pci_id = str(body.get("gpu_pci_id") or "")
            desired_mode = str(body.get("desired_mode") or "")
            vm_uuid = body.get("vm_uuid")
            confirm = bool(body.get("confirm"))
            if not gpu_pci_id:
                _json_response(self, 400, {"error": "gpu_pci_id is required"})
                return
            if desired_mode not in ("passthrough", "vgpu"):
                _json_response(
                    self,
                    400,
                    {
                        "error": "desired_mode must be 'passthrough' or 'vgpu'",
                        "note": "GPU-P is not a gpu_mode; use POST /hivemind/gpu/passthrough/game-stream-vm",
                    },
                )
                return
            if not confirm:
                _json_response(self, 400, {"error": "confirm: true is required (driver rebind dismounts the GPU)"})
                return
            result = gpu_passthrough.prepare_mode(
                self.runner.hivemind_url,
                gpu_pci_id=gpu_pci_id,
                desired_mode=desired_mode,
                vm_uuid=vm_uuid,
                confirm=True,
            )
            append_event(
                "hivemind_gpu_passthrough_prepare",
                {
                    "gpu_pci_id": gpu_pci_id,
                    "desired_mode": desired_mode,
                    "vm_uuid": vm_uuid,
                    "intent": result.get("intent"),
                },
            )
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        _json_response(self, 200, result)

    def _hivemind_gpu_passthrough_vgpu(self) -> None:
        """``POST /hivemind/gpu/passthrough/vgpu`` — create vGPU
        mediated devices. Requires ``{gpu_pci_id, profile, confirm:true}``."""
        try:
            body = _read_json(self) or {}
            gpu_pci_id = str(body.get("gpu_pci_id") or "")
            profile = str(body.get("profile") or "")
            count = int(body.get("count") or 1)
            confirm = bool(body.get("confirm"))
            if not gpu_pci_id or not profile:
                _json_response(self, 400, {"error": "gpu_pci_id and profile are required"})
                return
            if not confirm:
                _json_response(self, 400, {"error": "confirm: true is required"})
                return
            result = gpu_passthrough.create_vgpu(
                self.runner.hivemind_url,
                gpu_pci_id=gpu_pci_id,
                profile=profile,
                count=count,
                confirm=True,
            )
            append_event(
                "hivemind_gpu_passthrough_vgpu_create",
                {"gpu_pci_id": gpu_pci_id, "profile": profile, "count": count},
            )
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        _json_response(self, 200, result)

    def _hivemind_gpu_passthrough_game_stream_vm(self) -> None:
        """``POST /hivemind/gpu/passthrough/game-stream-vm`` —
        provision a Windows 11 GPU-P game-streaming VM. Requires
        ``{name, confirm:true}``."""
        try:
            body = _read_json(self) or {}
            name = str(body.get("name") or "")
            confirm = bool(body.get("confirm"))
            if not name:
                _json_response(self, 400, {"error": "name is required (must match ^[A-Za-z0-9._-]+$)"})
                return
            if not confirm:
                _json_response(self, 400, {"error": "confirm: true is required (new VM consumes host resources)"})
                return
            extra = {k: v for k, v in body.items() if k not in ("name", "confirm")}
            result = gpu_passthrough.create_game_stream_vm(
                self.runner.hivemind_url, name=name, confirm=True, **extra
            )
            append_event(
                "hivemind_gpu_passthrough_game_stream_vm",
                {"name": name, "opts": extra, "errors": result.get("errors", [])},
            )
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        _json_response(self, 200, result)

    # ------------------------------------------------------------------
    # /hivemind/voice_identities
    # ------------------------------------------------------------------

    def _hivemind_voice_identities_get(self) -> None:
        try:
            identities = voice_identity.list_identities(self.runner.hivemind_url)
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        _json_response(self, 200, {
            "schema": "Ms4VoiceIdentitiesSnapshot.v1",
            "identities": identities,
        })

    def _hivemind_voice_identity_enroll(self) -> None:
        try:
            body = _read_json(self) or {}
            name = str(body.get("name") or "")
            audio_b64 = body.get("audio_base64") or ""
            if not name or not audio_b64:
                _json_response(self, 400, {"error": "name and audio_base64 required"})
                return
            import base64 as _b64
            audio = _b64.b64decode(audio_b64)
            metadata = body.get("metadata") if isinstance(body.get("metadata"), dict) else None
            result = voice_identity.enroll(
                self.runner.hivemind_url, name=name, audio=audio, metadata=metadata
            )
            append_event("hivemind_voice_identity_enroll", {"name": name, "audio_bytes": len(audio)})
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        _json_response(self, 200, result if isinstance(result, dict) else {"result": result})

    def _dispatch_voice_identity_action(self, identity_id: str, action: str) -> bool:
        try:
            body = _read_json(self) or {}
            if action == "delete":
                result = voice_identity.delete(self.runner.hivemind_url, identity_id)
                append_event("hivemind_voice_identity_delete", {"identity_id": identity_id})
            elif action == "refine":
                embedding = body.get("embedding")
                if not isinstance(embedding, list) or not embedding:
                    _json_response(self, 400, {
                        "error": "embedding array required; HLI refine does not accept audio_base64"
                    })
                    return True
                result = voice_identity.refine(
                    self.runner.hivemind_url,
                    name=identity_id,
                    embedding=embedding,
                    blend_alpha=body.get("blend_alpha"),
                    metadata=body.get("metadata") if isinstance(body.get("metadata"), dict) else None,
                )
                append_event("hivemind_voice_identity_refine", {
                    "name": identity_id,
                    "embedding_dims": len(embedding),
                })
            else:
                return False
        except Exception as exc:
            self._emit_admin_error(exc)
            return True
        _json_response(self, 200, result if isinstance(result, dict) else {"result": result})
        return True

    # ------------------------------------------------------------------
    # /hivemind/approval
    # ------------------------------------------------------------------

    def _hivemind_approval_request(self) -> None:
        try:
            body = _read_json(self) or {}
            action_id = str(body.get("action_id") or "")
            summary = str(body.get("summary") or "")
            if not action_id or not summary:
                _json_response(self, 400, {"error": "action_id and summary required"})
                return
            result = human_approval.request(
                self.runner.hivemind_url,
                action_id=action_id,
                summary=summary,
                details=body.get("details") if isinstance(body.get("details"), dict) else None,
                risk_level=str(body.get("risk_level") or "medium"),
                timeout_secs=int(body.get("timeout_secs") or human_approval.DEFAULT_OVERALL_TIMEOUT_SECS),
            )
            append_event("hivemind_approval_request", {"action_id": action_id, "summary": summary[:200]})
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        _json_response(self, 200, result if isinstance(result, dict) else {"result": result})

    def _hivemind_approval_status_get(self, request_id: str) -> None:
        try:
            body = human_approval.status(self.runner.hivemind_url, request_id)
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        _json_response(self, 200, body if isinstance(body, dict) else {"raw": body})

    def _hivemind_approval_notify(self) -> None:
        try:
            body = _read_json(self) or {}
            channel = str(body.get("channel") or "log")
            message = str(body.get("message") or "")
            if not message:
                _json_response(self, 400, {"error": "message required"})
                return
            result = human_approval.notify(
                self.runner.hivemind_url,
                channel=channel,
                message=message,
                severity=str(body.get("severity") or "info"),
            )
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        _json_response(self, 200, result if isinstance(result, dict) else {"result": result})

    # ------------------------------------------------------------------
    # /hivemind/maintenance
    # ------------------------------------------------------------------

    def _hivemind_maintenance_enter(self, service_name: str) -> None:
        try:
            body = _read_json(self) or {}
            result = hivemind_tools.services_maintenance_enter(
                self.runner.hivemind_url,
                service_name=service_name,
                reason=body.get("reason"),
                duration_secs=body.get("duration_secs"),
            )
            append_event("hivemind_maintenance_enter", {"service_name": service_name, "reason": body.get("reason")})
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        _json_response(self, 200, result if isinstance(result, dict) else {"result": result})

    def _hivemind_maintenance_clear(self, service_name: str) -> None:
        try:
            result = hivemind_tools.services_maintenance_clear(self.runner.hivemind_url, service_name)
            append_event("hivemind_maintenance_clear", {"service_name": service_name})
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        _json_response(self, 200, result if isinstance(result, dict) else {"result": result})

    # ------------------------------------------------------------------
    # /hivemind/api_keys, /hivemind/ollama/*, /hivemind/crown
    # ------------------------------------------------------------------

    def _hivemind_api_keys_get(self) -> None:
        try:
            body = hivemind_tools.api_keys_status(self.runner.hivemind_url)
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        _json_response(self, 200, body if isinstance(body, dict) else {"raw": body})

    def _hivemind_ollama_tags_get(self) -> None:
        try:
            body = hivemind_tools.ollama_tags(self.runner.hivemind_url)
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        _json_response(self, 200, body if isinstance(body, dict) else {"raw": body})

    def _hivemind_ollama_control(self) -> None:
        try:
            body = _read_json(self) or {}
            action = str(body.get("action") or "")
            if not action:
                _json_response(self, 400, {"error": "action required (start|stop|restart)"})
                return
            result = hivemind_tools.ollama_service_control(self.runner.hivemind_url, action)
            append_event("hivemind_ollama_control", {"action": action})
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        _json_response(self, 200, result if isinstance(result, dict) else {"result": result})

    # ------------------------------------------------------------------
    # /hivemind/oracle — planner status / configure / chat
    # ------------------------------------------------------------------

    def _hivemind_oracle_readiness_get(self) -> None:
        try:
            snap = oracle_admin.readiness(self.runner.hivemind_url)
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        if snap.get("errors"):
            append_event("hivemind_oracle_readiness_partial_failure", {"errors": snap["errors"][:5]})
        _json_response(self, 200, snap)

    def _hivemind_oracle_status_get(self) -> None:
        try:
            snap = oracle_admin.status(self.runner.hivemind_url)
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        _json_response(self, 200, snap)

    def _hivemind_oracle_chat(self) -> None:
        try:
            body = _read_json(self) or {}
            message = str(body.get("message") or "")
            if not message:
                _json_response(self, 400, {"error": "message required"})
                return
            opts = {k: v for k, v in body.items() if k != "message"}
            result = oracle_admin.chat(self.runner.hivemind_url, message, **opts)
            append_event("hivemind_oracle_chat", {"message_preview": message[:200]})
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        _json_response(self, 200, result)

    def _hivemind_oracle_configure(self) -> None:
        try:
            body = _read_json(self) or {}
            config = body.get("config") if isinstance(body.get("config"), dict) else None
            if config is None:
                _json_response(self, 400, {"error": "config object required"})
                return
            result = oracle_admin.configure(self.runner.hivemind_url, config)
            append_event("hivemind_oracle_configure", {"config_keys": list(config.keys())})
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        _json_response(self, 200, result)

    # ------------------------------------------------------------------
    # /hivemind/training — fine-tune jobs
    # ------------------------------------------------------------------

    def _hivemind_training_get(self) -> None:
        try:
            snap = training_admin.combined_snapshot(self.runner.hivemind_url)
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        if snap.get("errors"):
            append_event("hivemind_training_partial_failure", {"errors": snap["errors"]})
        _json_response(self, 200, snap)

    def _hivemind_training_status_get(self, job_id: str) -> None:
        try:
            result = training_admin.status(self.runner.hivemind_url, job_id=job_id or None)
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        _json_response(self, 200, result)

    def _hivemind_training_start(self) -> None:
        try:
            body = _read_json(self) or {}
            recipe = body.get("recipe") if isinstance(body.get("recipe"), dict) else None
            if not recipe:
                _json_response(self, 400, {"error": "recipe object required"})
                return
            result = training_admin.start_job(self.runner.hivemind_url, recipe)
            append_event("hivemind_training_start", {"backend": recipe.get("backend"), "model": recipe.get("model")})
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        _json_response(self, 200, result)

    # ------------------------------------------------------------------
    # /hivemind/adapters
    # ------------------------------------------------------------------

    def _hivemind_adapters_get(self) -> None:
        try:
            snap = adapter_admin.combined_snapshot(self.runner.hivemind_url)
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        _json_response(self, 200, snap)

    def _hivemind_adapter_deploy(self) -> None:
        try:
            body = _read_json(self) or {}
            adapter_id = str(body.get("adapter_id") or "")
            target_model = body.get("target_model")
            if not adapter_id:
                _json_response(self, 400, {"error": "adapter_id required"})
                return
            result = adapter_admin.deploy(self.runner.hivemind_url, adapter_id, target_model)
            append_event("hivemind_adapter_deploy", {"adapter_id": adapter_id, "target_model": target_model})
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        _json_response(self, 200, result)

    # ------------------------------------------------------------------
    # /hivemind/loadout
    # ------------------------------------------------------------------

    def _hivemind_loadout_get(self) -> None:
        try:
            snap = loadout_admin.combined_snapshot(self.runner.hivemind_url)
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        _json_response(self, 200, snap)

    def _hivemind_loadout_apply(self) -> None:
        try:
            body = _read_json(self) or {}
            profile_id = str(body.get("profile_id") or "")
            if not profile_id:
                _json_response(self, 400, {"error": "profile_id required"})
                return
            result = loadout_admin.apply(self.runner.hivemind_url, profile_id)
            append_event("hivemind_loadout_apply", {"profile_id": profile_id})
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        _json_response(self, 200, result)

    # ------------------------------------------------------------------
    # /hivemind/deploy/gim
    # ------------------------------------------------------------------

    def _hivemind_deploy_gim(self) -> None:
        try:
            body = _read_json(self) or {}
            gim_name = str(body.get("gim_name") or "")
            if not gim_name:
                _json_response(self, 400, {"error": "gim_name required"})
                return
            opts = {k: v for k, v in body.items() if k != "gim_name"}
            result = hivemind_tools.deploy_gim(self.runner.hivemind_url, gim_name=gim_name, **opts)
            append_event("hivemind_deploy_gim", {"gim_name": gim_name, "opts": opts})
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        _json_response(self, 200, result if isinstance(result, dict) else {"result": result})

    # ------------------------------------------------------------------
    # /hivemind/inference
    # ------------------------------------------------------------------

    def _hivemind_inference_models_get(self) -> None:
        try:
            result = hivemind_tools.inference_models(self.runner.hivemind_url)
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        _json_response(self, 200, result if isinstance(result, dict) else {"models": result})

    def _hivemind_inference_chat(self) -> None:
        try:
            body = _read_json(self) or {}
            messages = body.get("messages") if isinstance(body.get("messages"), list) else None
            model = str(body.get("model") or "")
            if not messages or not model:
                _json_response(self, 400, {"error": "messages (list) and model (string) required"})
                return
            opts = {k: v for k, v in body.items() if k not in ("messages", "model")}
            result = hivemind_tools.inference_chat(
                self.runner.hivemind_url, messages=messages, model=model, **opts
            )
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        _json_response(self, 200, result if isinstance(result, dict) else {"result": result})

    # ------------------------------------------------------------------
    # /hivemind/logos — prompt optimization
    # ------------------------------------------------------------------

    def _hivemind_logos_prompts_get(self) -> None:
        try:
            result = hivemind_tools.logos_prompts_list(self.runner.hivemind_url)
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        _json_response(self, 200, result if isinstance(result, dict) else {"prompts": result})

    def _hivemind_logos_prompt_get(self, prompt_id: str) -> None:
        try:
            result = hivemind_tools.logos_prompts_get(self.runner.hivemind_url, prompt_id=prompt_id)
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        _json_response(self, 200, result if isinstance(result, dict) else {"result": result})

    def _hivemind_logos_optimize(self) -> None:
        try:
            body = _read_json(self) or {}
            prompt_id = str(body.get("prompt_id") or "")
            if not prompt_id:
                _json_response(self, 400, {"error": "prompt_id required"})
                return
            opts = {k: v for k, v in body.items() if k != "prompt_id"}
            result = hivemind_tools.logos_optimize(self.runner.hivemind_url, prompt_id=prompt_id, **opts)
            append_event("hivemind_logos_optimize", {"prompt_id": prompt_id})
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        _json_response(self, 200, result if isinstance(result, dict) else {"result": result})

    def _hivemind_logos_prompt_fork(self, prompt_id: str) -> None:
        try:
            body = _read_json(self) or {}
            opts = body or {}
            result = hivemind_tools.logos_prompts_fork(self.runner.hivemind_url, prompt_id=prompt_id, **opts)
            append_event("hivemind_logos_fork", {"prompt_id": prompt_id})
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        _json_response(self, 200, result if isinstance(result, dict) else {"result": result})

    def _hivemind_logos_candidate_promote(self, candidate_id: str) -> None:
        try:
            result = hivemind_tools.logos_candidates_promote(
                self.runner.hivemind_url, candidate_id=candidate_id
            )
            append_event("hivemind_logos_promote", {"candidate_id": candidate_id})
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        _json_response(self, 200, result if isinstance(result, dict) else {"result": result})

    # ------------------------------------------------------------------
    # /hivemind/services lifecycle (enable / disable / restart / list)
    # ------------------------------------------------------------------

    def _hivemind_services_list_get(self) -> None:
        try:
            result = hivemind_tools.services_list(self.runner.hivemind_url)
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        _json_response(self, 200, result if isinstance(result, dict) else {"services": result})

    def _hivemind_service_enable(self, service_name: str) -> None:
        try:
            result = hivemind_tools.services_enable(self.runner.hivemind_url, service_name=service_name)
            append_event("hivemind_service_enable", {"service_name": service_name})
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        _json_response(self, 200, result if isinstance(result, dict) else {"result": result})

    def _hivemind_service_disable(self, service_name: str) -> None:
        try:
            result = hivemind_tools.services_disable(self.runner.hivemind_url, service_name=service_name)
            append_event("hivemind_service_disable", {"service_name": service_name})
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        _json_response(self, 200, result if isinstance(result, dict) else {"result": result})

    def _hivemind_service_restart(self, service_name: str) -> None:
        try:
            result = hivemind_tools.services_restart(self.runner.hivemind_url, service_name=service_name)
            append_event("hivemind_service_restart", {"service_name": service_name})
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        _json_response(self, 200, result if isinstance(result, dict) else {"result": result})

    # ------------------------------------------------------------------
    # /hivemind/jobs/cancel - destructive, UUID-scoped inference cancel
    # ------------------------------------------------------------------

    def _hivemind_jobs_cancel(self) -> None:
        try:
            body = _read_json(self) or {}
            confirm = bool(body.get("confirm"))
            if not confirm:
                _json_response(self, 400, {
                    "error": "jobs.cancel is destructive. Pass {\"confirm\": true, "
                             "\"job_id\": \"<trace-uuid>\"} to proceed."
                })
                return
            job_id = str(body.get("job_id") or "").strip()
            if not job_id:
                _json_response(self, 400, {"error": "job_id trace UUID is required"})
                return
            reason = str(body.get("reason") or "MS4 operator cancellation").strip()
            result = hivemind_tools.jobs_cancel(
                self.runner.hivemind_url,
                job_id=job_id,
                reason=reason,
            )
            append_event("hivemind_jobs_cancel", {"job_id": job_id, "reason": reason})
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        _json_response(self, 200, result if isinstance(result, dict) else {"result": result})

    # ------------------------------------------------------------------
    # /hivemind/games + /hivemind/game-sessions — Phase 1 dry-run
    # orchestration shipped upstream May 26 2026
    # ------------------------------------------------------------------

    def _hivemind_game_availability_get(self, game_id: str) -> None:
        try:
            result = game_admin.ensure_available(self.runner.hivemind_url, game_id)
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        _json_response(self, 200, result)

    def _hivemind_game_session_plan(self) -> None:
        try:
            body = _read_json(self) or {}
            game = str(body.get("game") or "")
            if not game:
                _json_response(self, 400, {"error": "game required"})
                return
            kwargs = {k: v for k, v in body.items() if k != "game"}
            result = game_admin.plan(self.runner.hivemind_url, game=game, **kwargs)
            append_event("hivemind_game_session_plan", {
                "game": game,
                "client": body.get("client"),
                "quality": body.get("quality"),
                "job_id": result.get("job_id") if isinstance(result, dict) else None,
            })
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        _json_response(self, 200, result)

    def _hivemind_game_session_run(self) -> None:
        try:
            body = _read_json(self) or {}
            job_id = str(body.get("job_id") or "")
            if not job_id:
                _json_response(self, 400, {"error": "job_id required"})
                return
            result = game_admin.run(self.runner.hivemind_url, job_id)
            append_event("hivemind_game_session_run", {"job_id": job_id})
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        _json_response(self, 200, result)

    def _hivemind_game_session_status_get(self, job_id: str) -> None:
        try:
            result = game_admin.status(self.runner.hivemind_url, job_id)
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        _json_response(self, 200, result)

    def _hivemind_game_session_evidence_get(self, job_id: str) -> None:
        try:
            result = game_admin.evidence(self.runner.hivemind_url, job_id)
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        _json_response(self, 200, result)

    def _hivemind_game_session_cancel(self, job_id: str) -> None:
        try:
            result = game_admin.cancel(self.runner.hivemind_url, job_id)
            append_event("hivemind_game_session_cancel", {"job_id": job_id})
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        _json_response(self, 200, result)

    def _hivemind_game_session_plan_run(self) -> None:
        """Convenience: plan → run → fetch evidence in one call.
        Returns ``Ms4GameSession.v1``. Fail-soft per stage."""
        try:
            body = _read_json(self) or {}
            game = str(body.get("game") or "")
            if not game:
                _json_response(self, 400, {"error": "game required"})
                return
            result = game_admin.plan_run_and_collect(
                self.runner.hivemind_url,
                game=game,
                client=body.get("client"),
                quality=body.get("quality"),
                latency=body.get("latency"),
                duration_hint=body.get("duration_hint"),
            )
            append_event("hivemind_game_session_plan_run", {
                "game": game,
                "job_id": result.get("job_id"),
                "errors": result.get("errors", []),
            })
        except Exception as exc:
            self._emit_admin_error(exc)
            return
        _json_response(self, 200, result)

    # ------------------------------------------------------------------
    # Admin error mapping for the new modules
    # ------------------------------------------------------------------

    def _hivemind_crown_get(self) -> None:
        """Combined Crown snapshot: status + latest event + signal
        quality. Useful for the Settings panel even when MS4 doesn't
        drive Crown itself."""
        snap: dict[str, Any] = {
            "schema": "Ms4CrownSnapshot.v1",
            "status": None,
            "latest": None,
            "signal_quality": None,
            "errors": [],
        }
        for key, fn in (
            ("status", hivemind_tools.crown_status),
            ("latest", hivemind_tools.crown_latest),
            ("signal_quality", hivemind_tools.crown_signal_quality),
        ):
            try:
                snap[key] = fn(self.runner.hivemind_url)
            except HivemindToolError as exc:
                snap["errors"].append(f"{key}: {exc}")
        _json_response(self, 200, snap)

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
            if not body.get("prior_context"):
                session_id = body.get("parent_conversation_id")
                prior_context = _face_lobe_prior_context(
                    getattr(self.runner, "face_lobe_chat", None),
                    str(session_id) if session_id else None,
                )
                if prior_context:
                    body["prior_context"] = prior_context
            envelope = JobEnvelope.from_dict(body)
            # Obey an explicit Depth Lobe model from the submit dialog
            # (top-level depth_model_id / model_override) when the
            # envelope didn't already carry one.
            override = body.get("depth_model_id") or body.get("model_override")
            if override and not envelope.resource_request.model_override:
                envelope.resource_request.model_override = str(override)
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
        runner = default_runner()
        snap = runner.get(job_id)
        if snap is None:
            _json_response(self, 404, {"error": "unknown job_id"})
            return
        # Attach the full JobResult (text/summary) so the UI can show the
        # complete answer in the completion announcement's "Show details".
        result = runner.get_result(job_id)
        if result is not None:
            if result_matches_job_identity(snap, result):
                snap = {**snap, "result": result}
            else:
                snap = {**snap, "result_error": "result_identity_mismatch"}
        _json_response(self, 200, snap)

    def _double_agent_deliver(self, job_id: str) -> None:
        if not is_safe_job_id(job_id):
            _json_response(self, 400, {"error": "unsafe job_id"})
            return
        try:
            body = _read_json(self) or {}
            conversation_id = str(body.get("conversation_id") or "").strip()
            if not is_safe_conversation_id(conversation_id):
                _json_response(self, 400, {"error": "safe conversation_id is required"})
                return
            delivery = self.runner.deliver_depth_result(job_id, conversation_id)
            _json_response(self, 200, delivery)
        except KeyError:
            _json_response(self, 404, {"error": "unknown job_id"})
        except ValueError as exc:
            _json_response(self, 409, {"error": str(exc), "fail_closed": True})
        except RuntimeError as exc:
            _json_response(self, 409, {"error": str(exc), "fail_closed": True})
        except Exception as exc:
            _json_response(self, 500, {"error": str(exc), "fail_closed": True})

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
        # URLSearchParams percent-encodes model ids such as
        # ``llama3.1:8b``. Keep the existing last-value-wins behavior while
        # applying the standard query-string decoding contract (including
        # ``+`` as a space) before any value reaches routing/model selection.
        return dict(urllib.parse.parse_qsl(query, keep_blank_values=True))

    def _stream_chat(self) -> None:
        try:
            body = _read_json(self)
            message = str(body.get("message") or body.get("text") or "").strip()
            if not message:
                _json_response(self, 400, {"error": "message is required"})
                return
            events: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue()
            client_alive = threading.Event()
            client_alive.set()
            turn_cancel = threading.Event()

            def stream_callback(delta: str) -> bool:
                if not client_alive.is_set():
                    turn_cancel.set()
                    return False
                if delta:
                    events.put(("token", {"text": delta}))
                return client_alive.is_set()

            def worker() -> None:
                try:
                    result = self.runner.chat(
                        message,
                        session_id=body.get("session_id") or None,
                        model=body.get("model_id") or body.get("model") or None,
                        depth_model=body.get("depth_model_id") or body.get("depth_model") or None,
                        stream_callback=stream_callback,
                        cancel_event=turn_cancel,
                        client_id=body.get("client_id") or None,
                    )
                    if client_alive.is_set():
                        if result.get("fail_closed"):
                            events.put(("error", {
                                "error": result.get("error") or "recommend lease failed closed",
                                "fail_closed": True,
                            }))
                        elif result.get("completed") is True and result.get("cancelled") is not True:
                            events.put(("done", result))
                        else:
                            events.put(("incomplete", {
                                "error": "Face response ended before verified completion.",
                                "code": "face_stream_incomplete",
                                "text": str(result.get("text") or ""),
                                 "session_id": result.get("session_id"),
                                 "model": result.get("model"),
                                 # Dispatch happens before Face inference. Keep
                                 # the safe job identity so an incomplete Face
                                 # stream cannot orphan a valid Depth result.
                                 "dispatched_job": result.get("dispatched_job"),
                                 "router": result.get("router"),
                                 "depth_lobe_model": result.get("depth_lobe_model"),
                                 "completed": False,
                                "cancelled": bool(result.get("cancelled")),
                                "metrics": result.get("metrics") or {},
                            }))
                except ValueError as exc:
                    if client_alive.is_set():
                        events.put(("error", {"error": str(exc), "model_incompatible": True}))
                except HermesUnavailable as exc:
                    if client_alive.is_set():
                        events.put(("error", {"error": str(exc), "fail_closed": True}))
                except Exception as exc:
                    if client_alive.is_set():
                        events.put(("error", {"error": str(exc)}))

            _sse_start(self)
            if not _sse_event(self, "status", {"status": "started"}):
                client_alive.clear()
                turn_cancel.set()
                return
            thread = threading.Thread(target=worker, daemon=True)
            thread.start()
            while True:
                try:
                    event, payload = events.get(timeout=2.0)
                except queue.Empty:
                    if not _sse_event(self, "heartbeat", {"status": "running"}):
                        client_alive.clear()
                        turn_cancel.set()
                        return
                    continue
                if not _sse_event(self, event, payload):
                    client_alive.clear()
                    turn_cancel.set()
                    return
                if event in {"done", "incomplete", "error"}:
                    client_alive.clear()
                    # _sse_start advertises keep-alive for the streaming
                    # phase. A terminal frame has no Content-Length, so the
                    # socket close is the only unambiguous EOF for HTTP/1.1
                    # clients that keep reading after done/error.
                    self.close_connection = True
                    return
        except Exception as exc:
            _json_response(self, 500, {"error": str(exc)})

    def _serve_static(self) -> None:
        # Route on the URL path, not the raw request target. A reviewed build can
        # therefore be requested as ``/?v=<sha256>`` without turning the query
        # string into a filesystem name and returning a false 404.
        request_path = urllib.parse.unquote(urllib.parse.urlsplit(self.path).path)
        root_asset = ROOT_WEB_ASSETS.get(request_path)
        if request_path == "/":
            relative = "index.html"
        elif root_asset:
            relative = root_asset[0]
        elif request_path.startswith("/static/"):
            relative = request_path.removeprefix("/static/")
        else:
            _json_response(self, 404, {"error": "not found"})
            return
        target = (WEB_ROOT / relative).resolve()
        if WEB_ROOT.resolve() not in target.parents and target != WEB_ROOT.resolve():
            _json_response(self, 403, {"error": "forbidden"})
            return
        if not target.is_file():
            _json_response(self, 404, {"error": "not found"})
            return
        content = target.read_bytes()
        content_type = (
            root_asset[1]
            if root_asset
            else "text/html; charset=utf-8"
            if target.suffix == ".html"
            else "text/plain; charset=utf-8"
        )
        if target.suffix == ".js":
            content_type = "application/javascript; charset=utf-8"
        if target.suffix == ".css":
            content_type = "text/css; charset=utf-8"
        if target.suffix == ".svg":
            content_type = "image/svg+xml; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("X-Content-Type-Options", "nosniff")
        if request_path == "/service-worker.js":
            self.send_header("Service-Worker-Allowed", "/")
        if target.suffix == ".html" or request_path == "/service-worker.js":
            # The Oracle UI is an executable control surface. Never let a prior
            # candidate survive a source switch in the browser HTTP cache.
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
        self.end_headers()
        self.wfile.write(content)


def build_runner() -> Ms4HermesRunner:
    return Ms4HermesRunner(
        hermes_dir=str(runtime_common.hermes_dir()),
        hivemind_url=(
            os.environ.get("MS4_HIVEMIND_URL")
            or os.environ.get("MS4_HIVEMIND_HLI_URL")
            or "http://127.0.0.1:6089"
        ),
        ms3_url=os.environ.get("MS4_MS3_URL", "http://127.0.0.1:9080"),
        default_model=depth_fallback_model(),
    )


def run(host: str = "127.0.0.1", port: int = 9180) -> None:
    require_contained_runtime(SERVICE_NAME)
    handler_cls = Ms4GatewayHandler
    handler_cls.runner = build_runner()
    hermes_admin.initialize_state()
    try:
        hermes_admin.recover_interrupted_update()
        hermes_admin.reconcile_durable_terminal_state()
    except Exception as exc:
        log.warning("Hermes terminal-state reconcile on startup failed: %s", exc)
    # The Double Agent runner deliberately stays in subprocess mode here so
    # cancel() can truly terminate a blocking Hermes model call. The child
    # process (`double_agent/_worker_entry.py`) constructs its own
    # Ms4HermesRunner from MS4_HERMES_DIR / MS4_HIVEMIND_URL / MS4_MS3_URL /
    # MS4_DEFAULT_MODEL — no shared state between parent and child beyond
    # the SQLite blackboard. Tests inject a chat_runner_factory directly via
    # JobRunner(chat_runner_factory=...) to bypass subprocess overhead.
    da_runner = default_runner()
    da_runner.recover_on_startup()
    # Wire the optional LLM continuation classifier onto the production
    # runner so paraphrased follow-ups ("go for it", "sounds good, run
    # it") don't wrongly stale in-flight Depth Lobe work that the phrase
    # fast-path can't enumerate. Fail-safe + bounded: only consulted when
    # there are staleable jobs and the phrase path missed; any error /
    # timeout degrades to the phrase-only behavior. Disable with
    # MS4_DA_CONTINUATION_CLASSIFIER=0.
    if os.environ.get("MS4_DA_CONTINUATION_CLASSIFIER", "1").strip() not in ("0", "false", "no"):
        try:
            from machine_spirit_4.double_agent.continuation import (
                make_llm_continuation_classifier,
            )

            da_runner.set_continuation_classifier(
                make_llm_continuation_classifier(handler_cls.runner.hivemind_url)
            )
            print("MS4 Double Agent continuation classifier: enabled", flush=True)
        except Exception as exc:
            print(f"MS4 Double Agent continuation classifier wiring failed: {exc}", flush=True)
    # Pre-warm ASR + Face Lobe model on boot. Voice latency is
    # dominated by cold loads on the first turn (whisper ~3-8s, TTS
    # ~3-5s, Face Lobe model load ~5-10s). The autoscale daemon below
    # attempts selected-pool scale-out when enabled, then warms that pool
    # regardless of scale availability. Gateway boot never blocks.
    def _prewarm(label, fn):
        t0 = time.monotonic()
        try:
            res = fn()
            elapsed = int((time.monotonic() - t0) * 1000)
            print(f"MS4 {label} pre-warm: {elapsed}ms — {res}", flush=True)
        except Exception as exc:
            print(f"MS4 {label} pre-warm failed: {exc}", flush=True)

    hivemind_url = handler_cls.runner.hivemind_url
    tts_model = DEFAULT_TTS_MODEL
    tts_job_type = _tts_job_type_for_model(tts_model)
    _set_hivemind_url_for_auth(hivemind_url)
    threading.Thread(target=_prewarm, args=("ASR", lambda: {"warmed": True, "text": (prewarm_asr(hivemind_url=hivemind_url) or {}).get("text", "")}), daemon=True, name="ms4-asr-prewarm").start()
    # TTS_SUPER WebSocket pre-warm: opens a WS to HiveMind's
    # /v1/text-to-speech/{voice}/stream-input, pushes a tiny "Ready."
    # payload, and closes. Pays the TCP/TLS/WS-upgrade and GIM model
    # load cost at boot so the first real voice turn doesn't. We are
    # the only callers of TTS_SUPER ws_super engine — if HiveMind
    # isn't ready, this logs and skips silently.
    # When TTS_SUPER is the selected model pool, the autoscale daemon
    # performs this pre-warm after its scale attempt instead of racing it.
    if tts_job_type != "TTS_SUPER":
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
    # Periodic TTS keep-warm: HiveMind unloads an idle TTS model, so the
    # next voice turn's FIRST audio chunk pays a 3-5s cold load (the
    # dominant TTS-specific latency). Every MS4_VOICE_TTS_KEEPWARM_SECS
    # (default 240s; 0 disables) ping TTS with a tiny phrase so it stays
    # resident. Best-effort and silent on failure (a real turn would just
    # pay the cold load, i.e. status quo).
    def _tts_keepwarm_loop() -> None:
        try:
            interval = int(os.environ.get("MS4_VOICE_TTS_KEEPWARM_SECS", "240"))
        except (TypeError, ValueError):
            interval = 240
        if interval <= 0:
            return
        while True:
            time.sleep(interval)
            try:
                prewarm_tts(hivemind_url=hivemind_url)
            except Exception:
                pass  # best-effort keep-warm; next real turn handles cold load
    threading.Thread(target=_tts_keepwarm_loop, daemon=True, name="ms4-tts-keepwarm").start()
    # TTS replica scale-out. MS4 fires per-chunk TTS in parallel, but
    # HiveMind serializes generation per GIM process (GEN_LOCK — confirmed
    # by the HiveMind team 2026-06-02), so a reply's chunks only synthesize
    # concurrently if there are ~as many TTS replicas as chunks. Their
    # POST /provision/tts/scale (shipped 2026-06-02) launches replicas and
    # round-robins /v1/audio/speech across them; throughput scales
    # near-linearly with replica count (1→2.73, 3→6.58 rps). Crucially TTS
    # is GPU-IDLE (~5-10% util, ~3GB/replica), so replicas can be PACKED
    # multiple-per-GPU and `target` may exceed the GPU count.
    #
    # We default the target to MS4_VOICE_TTS_REPLICA_TARGET (2). IMPORTANT:
    # HiveMind's TTS GIMs are CPU-bound (per the 2026-06-02 study: ~5-10%
    # GPU, scales by process until CPU saturates). Measured the hard way on
    # this box: leaving ~6 TTS GIMs running pegged the CPU at 100% and
    # STARVED the LLM (/v1/chat/completions timed out) and real-speech ASR
    # — i.e. over-provisioning TTS breaks the rest of the voice loop. So the
    # default is deliberately modest: 2 replicas give a typical reply enough
    # concurrency to stay smooth while leaving CPU for the LLM + ASR. Raise
    # MS4_VOICE_TTS_REPLICA_TARGET only if you have CPU headroom (or TTS on
    # dedicated nodes); empty/0 = let HiveMind place one-per-GPU. NOTE the
    # scale endpoint only ADDS replicas (floors), so this can't reduce an
    # already-oversized pool — that's a HiveMind-side reset. We provision on
    # boot and re-issue periodically (idempotent). MS4_VOICE_TTS_AUTOSCALE=0
    # disables scale-out, not initial pre-warm; MS4_VOICE_TTS_AUTOSCALE_SECS
    # (default 600; 0 = boot-only).
    def _tts_replica_target() -> int | None:
        raw = os.environ.get("MS4_VOICE_TTS_REPLICA_TARGET", "2").strip()
        if not raw:
            return None
        try:
            t = int(raw)
        except ValueError:
            return None
        return t if t > 0 else None

    def _prewarm_selected_tts_pool() -> dict[str, Any]:
        if tts_job_type == "TTS_SUPER":
            result = dict(prewarm_tts_super_ws(hivemind_url=hivemind_url) or {})
            result.update({"model": tts_model, "job_type": tts_job_type})
            return result
        result = prewarm_tts(hivemind_url=hivemind_url, model=tts_model) or {}
        audio = result.get("audio_bytes") if isinstance(result, dict) else b""
        audio_bytes = audio if isinstance(audio, (bytes, bytearray)) else b""
        verdict: dict[str, Any] = {
            "warmed": bool(audio_bytes),
            "bytes": len(audio_bytes),
            "model": tts_model,
            "job_type": tts_job_type,
        }
        if isinstance(result, dict) and result.get("error"):
            verdict["error"] = result["error"]
        return verdict

    def _regular_tts_warm_attempts(
        scale_result: dict[str, Any],
        requested_target: int | None,
    ) -> int:
        if not scale_result.get("ok"):
            return 1
        replicas = scale_result.get("replicas")
        response_target = scale_result.get("target")
        if type(replicas) is not int or replicas <= 0:
            return 1
        limits = []
        if requested_target is not None:
            limits.append(requested_target)
        if type(response_target) is int and response_target > 0:
            limits.append(response_target)
        if not limits:
            return 1
        return max(1, min(replicas, *limits))

    def _probe_selected_tts_capacity(
        scale_result: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Admit REST capacity two only after a same-route live measurement."""

        replicas = scale_result.get("replicas")
        local_scale_proves_two_candidates = (
            scale_result.get("ok") is True
            and type(replicas) is int
            and replicas >= 2
        )
        location_policy = _tts_location_policy()
        # peer_only routes to the LAN pool, which the local scale endpoint does
        # not enumerate. Measure that already-existing same route even when
        # local replicas=0/1. local_only must retain the >=2 local-discovery
        # precondition so this background check cannot invent a second origin.
        if location_policy == "local_only" and not local_scale_proves_two_candidates:
            return None
        if location_policy != "peer_only" and not local_scale_proves_two_candidates:
            return None
        result = probe_voice_rest_concurrency(
            hivemind_url=hivemind_url,
            model=tts_model,
            response_format=DEFAULT_TTS_FORMAT,
        )
        # The probe result is intentionally sanitized: no text, audio, endpoint,
        # token, or raw provenance is retained in the boot log.
        print(
            f"MS4 {tts_job_type} capacity proof: "
            f"passed={result.get('passed')} "
            f"speedup={result.get('observed_speedup')} "
            f"floor={result.get('minimum_speedup')} "
            f"policy={result.get('location_policy')} "
            f"provenance={result.get('provenance_non_cloud_count')}/"
            f"{result.get('provenance_sample_count')} "
            f"reason={result.get('reason')}",
            flush=True,
        )
        return result

    def _tts_autoscale_loop() -> None:
        autoscale_enabled = (
            os.environ.get("MS4_VOICE_TTS_AUTOSCALE", "1").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        target = _tts_replica_target()
        res: dict[str, Any] = {"ok": False}
        if autoscale_enabled:
            try:
                res = provision_tts_replicas(
                    hivemind_url=hivemind_url,
                    target=target,
                    job_type=tts_job_type,
                )
            except Exception as exc:
                print(f"MS4 {tts_job_type} scale-out failed: {exc}", flush=True)
            if res.get("ok"):
                print(f"MS4 {tts_job_type} scale-out: {res.get('replicas')} replica(s) "
                      f"(target={target if target else 'per-GPU'}) — {res.get('status')}", flush=True)

        warm_attempts = (
            _regular_tts_warm_attempts(res, target)
            if tts_job_type == "TTS"
            else 1
        )
        for attempt in range(warm_attempts):
            label = (
                tts_job_type
                if warm_attempts == 1
                else f"{tts_job_type} {attempt + 1}/{warm_attempts}"
            )
            _prewarm(label, _prewarm_selected_tts_pool)
        _probe_selected_tts_capacity(res)

        if not autoscale_enabled:
            return
        try:
            interval = int(os.environ.get("MS4_VOICE_TTS_AUTOSCALE_SECS", "600"))
        except (TypeError, ValueError):
            interval = 600
        if interval <= 0:
            return
        while True:
            time.sleep(interval)
            try:
                refreshed = provision_tts_replicas(
                    hivemind_url=hivemind_url,
                    target=_tts_replica_target(),
                    job_type=tts_job_type,
                )
                refreshed_target = _tts_replica_target()
                refreshed_warm_attempts = (
                    _regular_tts_warm_attempts(refreshed, refreshed_target)
                    if tts_job_type == "TTS"
                    else 1
                )
                for attempt in range(refreshed_warm_attempts):
                    label = (
                        f"{tts_job_type} refresh"
                        if refreshed_warm_attempts == 1
                        else (
                            f"{tts_job_type} refresh "
                            f"{attempt + 1}/{refreshed_warm_attempts}"
                        )
                    )
                    _prewarm(label, _prewarm_selected_tts_pool)
                _probe_selected_tts_capacity(refreshed)
            except Exception:
                pass  # best-effort; single replica still serves TTS
    threading.Thread(target=_tts_autoscale_loop, daemon=True, name="ms4-tts-autoscale").start()
    # Periodic Face Lobe keep-warm: HiveMind evicts an idle chat model
    # after its residency window (~12 min — client keep_alive is capped
    # there, verified live), so a conversational gap longer than that
    # cold-loads the Face model on the next turn (5-10s ON the path to
    # first token — the single biggest idle-cold-start hit). Every
    # MS4_VOICE_FACE_KEEPWARM_SECS (default 180s, comfortably under the
    # ~12 min cap; 0 disables) ping the model the operator is ACTUALLY
    # talking to (last_face_model(), tracked per turn) with a 1-token
    # completion so it stays resident. Falls back to the Face Lobe
    # auto-picker when every turn so far was Auto-pick. Best-effort and
    # silent on failure (a real turn would just pay the cold load).
    def _face_keepwarm_loop() -> None:
        try:
            interval = int(os.environ.get("MS4_VOICE_FACE_KEEPWARM_SECS", "180"))
        except (TypeError, ValueError):
            interval = 180
        if interval <= 0:
            return
        while True:
            time.sleep(interval)
            try:
                prewarm_face_lobe_model(hivemind_url=hivemind_url, model=last_face_model())
            except Exception:
                pass  # best-effort keep-warm; next real turn handles cold load
    threading.Thread(target=_face_keepwarm_loop, daemon=True, name="ms4-flb-keepwarm").start()
    # Start MS3 heartbeat: every MS4_SPIRIT_HEARTBEAT_SECS (default 60s)
    # POST /identity/heartbeat so MS3 keeps the spirit's continuity
    # checks happy even when the Face Lobe direct chat path is the
    # only thing running. Best-effort — failures are surfaced via
    # /spirit/state but don't crash the gateway.
    start_heartbeat_thread(handler_cls.runner.ms3_url)
    server = ThreadingHTTPServer((host, port), handler_cls)
    # MS4-owned dream-cadence/background review consumer (not MS3 dreaming).
    voice_candidate_reconciler = VoiceCandidateReconciler(
        hivemind_url=handler_cls.runner.hivemind_url,
    )
    # Same-instance hook for a future manifest-owned, gated admission handler.
    server.voice_candidate_reconciler = voice_candidate_reconciler
    voice_candidate_reconciler.start()
    print(f"MS4 gateway listening on http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    finally:
        voice_candidate_reconciler.stop()
        closer = getattr(server, "server_close", None)
        if closer is not None:
            closer()


if __name__ == "__main__":
    run(
        host=os.environ.get("MS4_GATEWAY_HOST", "127.0.0.1"),
        port=int(os.environ.get("MS4_GATEWAY_PORT", "9180")),
    )
