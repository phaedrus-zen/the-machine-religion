from __future__ import annotations

import json
import logging
import os
import queue
import threading

log = logging.getLogger("ms4.gateway.server")
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
        if self.path.startswith("/voice/recent-turns"):
            self._voice_recent_turns_get()
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
                # Per-turn structured log: everything an operator needs
                # to debug a slow / wrong / silent turn after the fact.
                # Written to the audit log (jsonl) so `/audit?limit=N`
                # and the Settings "Recent voice turns" panel surface
                # it. Keep the payload small — full per-chunk timings
                # live in result["metrics"] which we drop here.
                try:
                    metrics = (result or {}).get("metrics") or {}
                    append_event("voice_turn_complete", {
                        "session_id": (result or {}).get("session_id"),
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
                    })
                except Exception as exc:
                    log.warning("voice_turn_complete audit log failed: %s", exc)
                events.put(("done", result))
            except VoiceRequestError as exc:
                append_event("voice_turn_failed", {"kind": "request_error", "error": str(exc)})
                events.put(("error", {"error": str(exc)}))
            except VoiceUnavailable as exc:
                append_event("voice_turn_failed", {"kind": "fail_closed", "error": str(exc)})
                events.put(("error", {"error": str(exc), "fail_closed": True}))
            except Exception as exc:
                append_event("voice_turn_failed", {"kind": "exception", "error": str(exc)})
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

    # ------------------------------------------------------------------
    # /voice/recent-turns — per-turn diagnostics for the Settings UI
    # ------------------------------------------------------------------

    def _voice_recent_turns_get(self) -> None:
        """Return the most recent ``voice_turn_complete`` and
        ``voice_turn_failed`` audit events so the UI can render a
        per-turn diagnostics panel. Tail-only, newest first."""
        try:
            limit = 25
            if "?" in self.path:
                qs = self.path.split("?", 1)[1]
                for part in qs.split("&"):
                    key, _, value = part.partition("=")
                    if key == "limit":
                        try:
                            limit = max(1, min(200, int(value)))
                        except ValueError:
                            limit = 25
            # Pull a generous slice from the audit log (events are
            # heterogeneous), then filter to voice events.
            events = read_events(limit=limit * 8)
            voice_events = [
                e for e in events
                if isinstance(e, dict)
                and e.get("event") in ("voice_turn_complete", "voice_turn_failed")
            ]
            voice_events.reverse()  # newest first
            _json_response(self, 200, {
                "schema": "Ms4VoiceRecentTurns.v1",
                "turns": voice_events[:limit],
                "limit": limit,
            })
        except Exception as exc:
            _json_response(self, 500, {"error": str(exc)})

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
                audio_b64 = body.get("audio_base64") or ""
                if not audio_b64:
                    _json_response(self, 400, {"error": "audio_base64 required"})
                    return True
                import base64 as _b64
                audio = _b64.b64decode(audio_b64)
                result = voice_identity.refine(self.runner.hivemind_url, identity_id=identity_id, audio=audio)
                append_event("hivemind_voice_identity_refine", {"identity_id": identity_id, "audio_bytes": len(audio)})
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
    # /hivemind/jobs/cancel — destructive (resets all active jobs per spec)
    # ------------------------------------------------------------------

    def _hivemind_jobs_cancel(self) -> None:
        try:
            body = _read_json(self) or {}
            confirm = bool(body.get("confirm"))
            if not confirm:
                _json_response(self, 400, {
                    "error": "jobs.cancel currently resets ALL active inference jobs "
                             "(per HiveMind spec — per-job cancel is not yet implemented). "
                             "Pass {\"confirm\": true} to proceed."
                })
                return
            job_id = str(body.get("job_id") or "all")
            result = hivemind_tools.jobs_cancel(self.runner.hivemind_url, job_id=job_id)
            append_event("hivemind_jobs_cancel_all", {"job_id_requested": job_id})
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
