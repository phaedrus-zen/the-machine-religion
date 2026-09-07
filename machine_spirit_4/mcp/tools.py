from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from machine_spirit_4 import hermes_admin
from machine_spirit_4.deps_status import dependency_status
from machine_spirit_4.desktop import DesktopSafetyError, desktop_controller
from machine_spirit_4.double_agent import (
    JobEnvelope,
    RunnerError,
    SchemaError as DoubleAgentSchemaError,
    default_blackboard,
    default_runner,
)
from machine_spirit_4.gateway.context import TMR_GROUNDING, format_inventory_answer
from machine_spirit_4.gateway.vision import analyze_local_image
from machine_spirit_4.hermes_admin import HermesUpgradeError
from .schemas import ToolInputError, optional_string, require_object, require_string


MCP_TEXT = "text"
_RUNTIME_ACTION_ANNOTATION = "_ms4RuntimeAction"


@dataclass(frozen=True)
class ToolDef:
    name: str
    description: str
    input_schema: dict[str, Any]
    annotations: dict[str, Any]
    handler: Callable[[Any, dict[str, Any]], Any]

    @property
    def is_runtime_action(self) -> bool:
        return self.annotations.get(_RUNTIME_ACTION_ANNOTATION) is True

    def as_mcp_tool(self) -> dict[str, Any]:
        annotations = dict(self.annotations)
        annotations.pop(_RUNTIME_ACTION_ANNOTATION, None)
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
            "annotations": annotations,
        }


def read_only_annotations(title: str) -> dict[str, Any]:
    return {
        "title": title,
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }


def dry_run_annotations(title: str) -> dict[str, Any]:
    return {
        "title": title,
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }


def chat_annotations(title: str) -> dict[str, Any]:
    return {
        "title": title,
        "readOnlyHint": False,
        "idempotentHint": False,
        # Internal registry truth used to prevent a mutable manifest from
        # reclassifying or bypassing runtime-action dispatch policy.
        _RUNTIME_ACTION_ANNOTATION: True,
    }


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def identity_verify(runtime: Any, arguments: dict[str, Any]) -> Any:
    status, payload = runtime.post_json(
        f"{runtime.ms3_url}/identity/verify",
        {
            "spirit_id": optional_string(arguments, "spirit_id") or "sister",
            # Read-only contract (manifest kind=read_only, readOnlyHint=True):
            # this tool never initializes an anchor. Anchor initialization is
            # the boot path's job, so allow_initialize is not exposed here.
            "allow_initialize": False,
        },
        timeout=20,
    )
    if status >= 400:
        raise RuntimeError(payload)
    return payload


def identity_state(runtime: Any, arguments: dict[str, Any]) -> Any:
    status, payload = runtime.get_json(f"{runtime.ms3_url}/state", timeout=20)
    if status >= 400:
        raise RuntimeError(payload)
    return payload


def ethics_evaluate(runtime: Any, arguments: dict[str, Any]) -> Any:
    intent = require_object(arguments, "intent")
    status, payload = runtime.post_json(f"{runtime.ms3_url}/ethics/evaluate", intent, timeout=20)
    if status >= 400:
        raise RuntimeError(payload)
    return payload


def voice_status(runtime: Any, arguments: dict[str, Any]) -> Any:
    status, payload = runtime.get_json(f"{runtime.ms3_url}/voice/status", timeout=15)
    if status >= 400:
        raise RuntimeError(payload)
    return payload


def sessions_list(runtime: Any, arguments: dict[str, Any]) -> Any:
    return {"sessions": runtime.runner.sessions()}


def chat_send(runtime: Any, arguments: dict[str, Any]) -> Any:
    message = require_string(arguments, "message")
    return runtime.runner.chat(
        message,
        session_id=optional_string(arguments, "session_id"),
        model=optional_string(arguments, "model"),
    )


def models_list(runtime: Any, arguments: dict[str, Any]) -> Any:
    status, payload = runtime.get_json(f"{runtime.ms3_url}/models", timeout=30)
    if status >= 400:
        raise RuntimeError(payload)
    return payload


def vision_analyze_local(runtime: Any, arguments: dict[str, Any]) -> Any:
    return analyze_local_image(
        hivemind_url=runtime.hivemind_url,
        image_path=require_string(arguments, "image_path"),
        question=optional_string(arguments, "question"),
        model=optional_string(arguments, "model"),
    )


def doctrine_explain(runtime: Any, arguments: dict[str, Any]) -> Any:
    return {
        "schema": "Ms4DoctrineExplanation.v1",
        "topic": optional_string(arguments, "topic") or "The Machine Religion",
        "text": TMR_GROUNDING,
    }


def hivemind_inventory(runtime: Any, arguments: dict[str, Any]) -> Any:
    summary = runtime.mcp_call(
        "hivemind.cluster.summary@v1",
        {"include_gpu_details": bool(arguments.get("include_gpu_details", True))},
    )
    hosts = runtime.mcp_call(
        "hivemind.hosts.list@v1",
        {"status_filter": optional_string(arguments, "status_filter") or "all"},
    )
    return {
        "schema": "Ms4HiveMindInventory.v1",
        "sources": ["hivemind.cluster.summary@v1", "hivemind.hosts.list@v1"],
        "text": format_inventory_answer(summary, hosts),
    }


def nibbles_dry_run(runtime: Any, arguments: dict[str, Any]) -> Any:
    intent = require_object(arguments, "intent")
    payload = intent.get("payload") if isinstance(intent.get("payload"), dict) else {}
    if payload.get("dry_run_only") is not True:
        raise ToolInputError("Nibbles dry run requires payload.dry_run_only=true")
    if payload.get("physical_output_enabled") is not False:
        raise ToolInputError("Nibbles dry run requires payload.physical_output_enabled=false")
    if intent.get("action_type") not in {"scare", "physical"}:
        raise ToolInputError("Nibbles dry run only accepts scare or physical action intents")
    status, decision = runtime.post_json(f"{runtime.ms3_url}/ethics/evaluate", intent, timeout=20)
    if status >= 400:
        raise RuntimeError(decision)
    return {
        "schema": "NibblesDryRunResult.v1",
        "dry_run_only": True,
        "intent": intent,
        "ethics_decision": decision,
    }


def hermes_tools_list(runtime: Any, arguments: dict[str, Any]) -> Any:
    toolsets = arguments.get("toolsets")
    if toolsets is not None and not isinstance(toolsets, list):
        raise ToolInputError("toolsets must be an array of strings")
    enabled_toolsets = [str(toolset) for toolset in toolsets] if toolsets else None
    return {"tools": runtime.runner.list_hermes_tools(enabled_toolsets=enabled_toolsets)}


def hermes_tool_call(runtime: Any, arguments: dict[str, Any]) -> Any:
    tool_name = require_string(arguments, "tool")
    args = arguments.get("args")
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise ToolInputError("args must be an object")
    return runtime.runner.dispatch_hermes_tool(
        tool_name,
        args,
        session_id=optional_string(arguments, "session_id"),
    )


def runtime_deps_status(runtime: Any, arguments: dict[str, Any]) -> Any:
    return dependency_status()


def hermes_version(runtime: Any, arguments: dict[str, Any]) -> Any:
    refresh = bool(arguments.get("refresh"))
    return hermes_admin.version_info(force_refresh_latest=refresh)


def hermes_releases(runtime: Any, arguments: dict[str, Any]) -> Any:
    return {"releases": [r.__dict__ for r in hermes_admin.recent_releases(force_refresh=bool(arguments.get("refresh")))]}


def hermes_update(runtime: Any, arguments: dict[str, Any]) -> Any:
    target = optional_string(arguments, "target_version") or optional_string(arguments, "version")
    requester = optional_string(arguments, "request_user") or "ms4-mcp"
    try:
        snap = hermes_admin.trigger_update(target_version=target, request_user=requester)
    except HermesUpgradeError as exc:
        raise ToolInputError(str(exc)) from exc
    return snap.to_dict()


def hermes_update_status(runtime: Any, arguments: dict[str, Any]) -> Any:
    snap = hermes_admin.last_update()
    return snap.to_dict() if snap else {"status": "idle", "phase": "none", "job_id": None}


def double_agent_submit(runtime: Any, arguments: dict[str, Any]) -> Any:
    try:
        envelope = JobEnvelope.from_dict(dict(arguments))
        return default_runner().submit(envelope)
    except DoubleAgentSchemaError as exc:
        raise ToolInputError(str(exc)) from exc
    except RunnerError as exc:
        raise RuntimeError(str(exc)) from exc


def double_agent_status(runtime: Any, arguments: dict[str, Any]) -> Any:
    job_id = require_string(arguments, "job_id")
    snap = default_runner().get(job_id)
    if snap is None:
        raise ToolInputError(f"unknown job_id: {job_id}")
    return snap


def double_agent_list(runtime: Any, arguments: dict[str, Any]) -> Any:
    conv = optional_string(arguments, "conversation_id")
    states_arg = arguments.get("states")
    if states_arg is not None and not isinstance(states_arg, list):
        raise ToolInputError("states must be an array of strings")
    states = tuple(str(s) for s in (states_arg or [])) or None
    limit = arguments.get("limit")
    if limit is not None and not isinstance(limit, int):
        raise ToolInputError("limit must be an integer")
    return {"jobs": default_runner().list(
        conversation_id=conv, states=states, limit=int(limit or 50),
    )}


def double_agent_events(runtime: Any, arguments: dict[str, Any]) -> Any:
    job_id = require_string(arguments, "job_id")
    after = optional_string(arguments, "after_event_id")
    limit = arguments.get("limit")
    if limit is not None and not isinstance(limit, int):
        raise ToolInputError("limit must be an integer")
    return {
        "job_id": job_id,
        "events": default_blackboard().list_events(
            job_id, after_event_id=after, limit=int(limit or 200),
        ),
    }


def double_agent_cancel(runtime: Any, arguments: dict[str, Any]) -> Any:
    job_id = require_string(arguments, "job_id")
    try:
        return default_runner().cancel(job_id)
    except RunnerError as exc:
        raise ToolInputError(str(exc)) from exc


def double_agent_mark_stale(runtime: Any, arguments: dict[str, Any]) -> Any:
    job_id = require_string(arguments, "job_id")
    reason = optional_string(arguments, "reason") or "Marked stale via MCP."
    try:
        return default_runner().mark_stale(job_id, reason=reason)
    except RunnerError as exc:
        raise ToolInputError(str(exc)) from exc


def _desktop_action_intent(arguments: dict[str, Any]) -> dict[str, Any]:
    action_name = str(arguments.get("action", "unknown")).lower()
    low_risk = action_name == "wait"
    return {
        "schema": "ActionIntent.v1",
        "spirit_id": optional_string(arguments, "spirit_id") or "sister",
        "action_id": f"ms4-desktop-{action_name}",
        "proposed_by": "ms4_mcp",
        "action_type": "desktop_ui",
        "description": f"MS4 desktop UI action: {action_name}",
        "inputs_used": ["ms4.desktop.action@v1"],
        "risk_class": "low" if low_risk else "medium",
        "requires_safety_clearance": False,
        "payload": {key: value for key, value in arguments.items() if key not in {"text", "spirit_id"}},
    }


def _ethics_allows(payload: dict[str, Any]) -> bool:
    decision = str(payload.get("decision") or payload.get("resolution") or "").lower()
    return decision in {"allow", "allowed", "offer", "noactionneeded"} or payload.get("allowed") is True


def desktop_status(runtime: Any, arguments: dict[str, Any]) -> Any:
    return desktop_controller().status()


def desktop_capture(runtime: Any, arguments: dict[str, Any]) -> Any:
    return desktop_controller().capture(arguments)


def desktop_action(runtime: Any, arguments: dict[str, Any]) -> Any:
    action = require_string(arguments, "action")
    payload = dict(arguments)
    payload["action"] = action
    status, decision = runtime.post_json(f"{runtime.ms3_url}/ethics/evaluate", _desktop_action_intent(payload), timeout=20)
    if status >= 400 or not _ethics_allows(decision if isinstance(decision, dict) else {}):
        raise DesktopSafetyError(f"desktop action blocked by MS3 ethics: {decision}")
    return desktop_controller().act(payload)


# ---------------------------------------------------------------------------
# HiveMind proxy handlers (May 25 2026)
#
# Thin pass-through over the gateway admin modules. Errors are mapped
# to RuntimeError so the MCP framework wraps them as isError replies
# without leaking transport-level details.
# ---------------------------------------------------------------------------


def hivemind_time(runtime: Any, arguments: dict[str, Any]) -> Any:
    from machine_spirit_4.gateway import hivemind_tools as _tools
    return _tools.time_now(runtime.hivemind_url)


def hivemind_capability_matrix(runtime: Any, arguments: dict[str, Any]) -> Any:
    from machine_spirit_4.gateway import hivemind_tools as _tools
    return _tools.capability_matrix(runtime.hivemind_url)


def hivemind_vms_list(runtime: Any, arguments: dict[str, Any]) -> Any:
    from machine_spirit_4.gateway import vm_admin
    return vm_admin.list_with_gpu_assignments(runtime.hivemind_url)


def hivemind_vm_start(runtime: Any, arguments: dict[str, Any]) -> Any:
    from machine_spirit_4.gateway import vm_admin
    return vm_admin.start_vm(runtime.hivemind_url, require_string(arguments, "vm_id"))


def hivemind_vm_stop(runtime: Any, arguments: dict[str, Any]) -> Any:
    from machine_spirit_4.gateway import vm_admin
    return vm_admin.stop_vm(runtime.hivemind_url, require_string(arguments, "vm_id"))


def hivemind_vm_screenshot(runtime: Any, arguments: dict[str, Any]) -> Any:
    from machine_spirit_4.gateway import vm_admin
    return vm_admin.get_screenshot(runtime.hivemind_url, require_string(arguments, "vm_id"))


def hivemind_apps_list(runtime: Any, arguments: dict[str, Any]) -> Any:
    from machine_spirit_4.gateway import app_admin
    return app_admin.list_with_status_and_metrics(runtime.hivemind_url)


def hivemind_app_start(runtime: Any, arguments: dict[str, Any]) -> Any:
    from machine_spirit_4.gateway import app_admin
    return app_admin.start_app(runtime.hivemind_url, require_string(arguments, "app_id"))


def hivemind_app_stop(runtime: Any, arguments: dict[str, Any]) -> Any:
    from machine_spirit_4.gateway import app_admin
    return app_admin.stop_app(runtime.hivemind_url, require_string(arguments, "app_id"))


def hivemind_storage_snapshot(runtime: Any, arguments: dict[str, Any]) -> Any:
    from machine_spirit_4.gateway import storage_admin
    return storage_admin.combined_snapshot(runtime.hivemind_url)


def hivemind_network_snapshot(runtime: Any, arguments: dict[str, Any]) -> Any:
    from machine_spirit_4.gateway import network_admin
    return network_admin.combined_snapshot(runtime.hivemind_url)


def hivemind_gpu_snapshot(runtime: Any, arguments: dict[str, Any]) -> Any:
    from machine_spirit_4.gateway import gpu_mode_admin
    return gpu_mode_admin.combined_snapshot(runtime.hivemind_url)


def hivemind_voice_identities_list(runtime: Any, arguments: dict[str, Any]) -> Any:
    from machine_spirit_4.gateway import voice_identity
    return {
        "schema": "Ms4VoiceIdentitiesSnapshot.v1",
        "identities": voice_identity.list_identities(runtime.hivemind_url),
    }


def hivemind_approval_request(runtime: Any, arguments: dict[str, Any]) -> Any:
    from machine_spirit_4.gateway import human_approval
    return human_approval.request(
        runtime.hivemind_url,
        action_id=require_string(arguments, "action_id"),
        summary=require_string(arguments, "summary"),
        details=arguments.get("details") if isinstance(arguments.get("details"), dict) else None,
        risk_level=optional_string(arguments, "risk_level") or "medium",
        timeout_secs=int(arguments.get("timeout_secs") or human_approval.DEFAULT_OVERALL_TIMEOUT_SECS),
    )


def hivemind_approval_status(runtime: Any, arguments: dict[str, Any]) -> Any:
    from machine_spirit_4.gateway import human_approval
    return human_approval.status(runtime.hivemind_url, require_string(arguments, "request_id"))


def hivemind_notify(runtime: Any, arguments: dict[str, Any]) -> Any:
    from machine_spirit_4.gateway import human_approval
    return human_approval.notify(
        runtime.hivemind_url,
        channel=optional_string(arguments, "channel") or "log",
        message=require_string(arguments, "message"),
        severity=optional_string(arguments, "severity") or "info",
    )


# ---------------------------------------------------------------------------
# PsyKyo proxy handlers (May 26 2026)
#
# Thin pass-through over the gateway hivemind_tools.psykyo_* helpers, which
# in turn call HiveMind's hivemind.psykyo.* MCP tools. Those tools proxy to
# the loopback-only PsyKyo MCP gateway (PSYKYO_MCP_BASE, default
# http://127.0.0.1:6765). Loopback validation lives in the HiveMind MCP
# gateway dispatcher, so no extra host check is needed here. Errors surface
# as RuntimeError -> MCP isError replies.
# ---------------------------------------------------------------------------


def hivemind_psykyo_benchmark_run(runtime: Any, arguments: dict[str, Any]) -> Any:
    from machine_spirit_4.gateway import hivemind_tools as _tools
    return _tools.psykyo_benchmark_run(runtime.hivemind_url, **arguments)


def hivemind_psykyo_benchmark_gap(runtime: Any, arguments: dict[str, Any]) -> Any:
    from machine_spirit_4.gateway import hivemind_tools as _tools
    return _tools.psykyo_benchmark_gap(runtime.hivemind_url)


def hivemind_psykyo_benchmark_workqueue(runtime: Any, arguments: dict[str, Any]) -> Any:
    from machine_spirit_4.gateway import hivemind_tools as _tools
    return _tools.psykyo_benchmark_workqueue(runtime.hivemind_url)


def hivemind_psykyo_evidence_latest(runtime: Any, arguments: dict[str, Any]) -> Any:
    from machine_spirit_4.gateway import hivemind_tools as _tools
    return _tools.psykyo_evidence_latest(runtime.hivemind_url)


def hivemind_psykyo_vlm_consensus(runtime: Any, arguments: dict[str, Any]) -> Any:
    from machine_spirit_4.gateway import hivemind_tools as _tools
    evidence_ref = require_string(arguments, "evidence_ref")
    models = arguments.get("models")
    if models is not None and not isinstance(models, list):
        raise ToolInputError("models must be an array of strings")
    return _tools.psykyo_vlm_consensus(
        runtime.hivemind_url,
        evidence_ref=evidence_ref,
        models=[str(m) for m in models] if models is not None else None,
    )


# ---------------------------------------------------------------------------
# May 26 2026 — new HiveMind admin proxies (Oracle, training, adapters,
# loadout, deploy, inference, logos, services lifecycle, jobs.cancel)
#
# Each proxy lets OTHER agents drive HiveMind through MS4 so the
# ethics + audit pipeline applies. Errors propagate as RuntimeError →
# the MCP framework wraps them as isError replies.
# ---------------------------------------------------------------------------


def hivemind_oracle_status(runtime: Any, arguments: dict[str, Any]) -> Any:
    from machine_spirit_4.gateway import oracle_admin
    return oracle_admin.status(runtime.hivemind_url)


def hivemind_oracle_chat(runtime: Any, arguments: dict[str, Any]) -> Any:
    from machine_spirit_4.gateway import oracle_admin
    opts = {k: v for k, v in arguments.items() if k != "message"}
    return oracle_admin.chat(runtime.hivemind_url, require_string(arguments, "message"), **opts)


def hivemind_training_backends(runtime: Any, arguments: dict[str, Any]) -> Any:
    from machine_spirit_4.gateway import training_admin
    return {"backends": training_admin.list_backends(runtime.hivemind_url)}


def hivemind_training_start(runtime: Any, arguments: dict[str, Any]) -> Any:
    from machine_spirit_4.gateway import training_admin
    return training_admin.start_job(runtime.hivemind_url, require_object(arguments, "recipe"))


def hivemind_training_status(runtime: Any, arguments: dict[str, Any]) -> Any:
    from machine_spirit_4.gateway import training_admin
    return training_admin.status(runtime.hivemind_url, optional_string(arguments, "job_id"))


def hivemind_adapters_list(runtime: Any, arguments: dict[str, Any]) -> Any:
    from machine_spirit_4.gateway import adapter_admin
    return {"schema": "Ms4AdapterSnapshot.v1", "adapters": adapter_admin.list_adapters(runtime.hivemind_url)}


def hivemind_adapters_deploy(runtime: Any, arguments: dict[str, Any]) -> Any:
    from machine_spirit_4.gateway import adapter_admin
    return adapter_admin.deploy(
        runtime.hivemind_url,
        require_string(arguments, "adapter_id"),
        optional_string(arguments, "target_model"),
    )


def hivemind_loadout_profiles(runtime: Any, arguments: dict[str, Any]) -> Any:
    from machine_spirit_4.gateway import loadout_admin
    return {"schema": "Ms4LoadoutSnapshot.v1", "profiles": loadout_admin.list_profiles(runtime.hivemind_url)}


def hivemind_loadout_apply(runtime: Any, arguments: dict[str, Any]) -> Any:
    from machine_spirit_4.gateway import loadout_admin
    return loadout_admin.apply(runtime.hivemind_url, require_string(arguments, "profile_id"))


def hivemind_deploy_gim(runtime: Any, arguments: dict[str, Any]) -> Any:
    from machine_spirit_4.gateway import hivemind_tools as _tools
    gim_name = require_string(arguments, "gim_name")
    opts = {k: v for k, v in arguments.items() if k != "gim_name"}
    return _tools.deploy_gim(runtime.hivemind_url, gim_name=gim_name, **opts)


def hivemind_inference_models(runtime: Any, arguments: dict[str, Any]) -> Any:
    from machine_spirit_4.gateway import hivemind_tools as _tools
    return _tools.inference_models(runtime.hivemind_url)


def hivemind_inference_chat(runtime: Any, arguments: dict[str, Any]) -> Any:
    from machine_spirit_4.gateway import hivemind_tools as _tools
    messages = arguments.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ToolInputError("messages (list) required")
    model = require_string(arguments, "model")
    opts = {
        key: arguments[key]
        for key in ("temperature", "max_tokens")
        if key in arguments
    }
    return _tools.inference_chat(runtime.hivemind_url, messages=messages, model=model, **opts)


def hivemind_logos_optimize(runtime: Any, arguments: dict[str, Any]) -> Any:
    """Run the Logos Machina optimizer; manifest gate = 'confirm:true required'."""
    if not bool(arguments.get("confirm")):
        raise ToolInputError("logos.optimize mutates a managed prompt. Pass confirm:true to proceed.")
    from machine_spirit_4.gateway import hivemind_tools as _tools
    prompt_id = require_string(arguments, "prompt_id")
    opts = {k: v for k, v in arguments.items() if k != "prompt_id"}
    return _tools.logos_optimize(runtime.hivemind_url, prompt_id=prompt_id, **opts)


def hivemind_services_enable(runtime: Any, arguments: dict[str, Any]) -> Any:
    from machine_spirit_4.gateway import hivemind_tools as _tools
    return _tools.services_enable(runtime.hivemind_url, service_name=require_string(arguments, "service_name"))


def hivemind_services_disable(runtime: Any, arguments: dict[str, Any]) -> Any:
    from machine_spirit_4.gateway import hivemind_tools as _tools
    return _tools.services_disable(runtime.hivemind_url, service_name=require_string(arguments, "service_name"))


def hivemind_services_restart(runtime: Any, arguments: dict[str, Any]) -> Any:
    from machine_spirit_4.gateway import hivemind_tools as _tools
    return _tools.services_restart(runtime.hivemind_url, service_name=require_string(arguments, "service_name"))


def hivemind_gpu_passthrough_snapshot(runtime: Any, arguments: dict[str, Any]) -> Any:
    from machine_spirit_4.gateway import gpu_passthrough
    return gpu_passthrough.snapshot(runtime.hivemind_url)


def hivemind_gpu_passthrough_prepare(runtime: Any, arguments: dict[str, Any]) -> Any:
    from machine_spirit_4.gateway import gpu_passthrough
    gpu_pci_id = require_string(arguments, "gpu_pci_id")
    desired_mode = require_string(arguments, "desired_mode")
    if not arguments.get("confirm"):
        raise ValueError("confirm: true is required")
    return gpu_passthrough.prepare_mode(
        runtime.hivemind_url,
        gpu_pci_id=gpu_pci_id,
        desired_mode=desired_mode,
        vm_uuid=arguments.get("vm_uuid"),
        confirm=True,
    )


def hivemind_gpu_passthrough_vgpu(runtime: Any, arguments: dict[str, Any]) -> Any:
    from machine_spirit_4.gateway import gpu_passthrough
    gpu_pci_id = require_string(arguments, "gpu_pci_id")
    profile = require_string(arguments, "profile")
    if not arguments.get("confirm"):
        raise ValueError("confirm: true is required")
    return gpu_passthrough.create_vgpu(
        runtime.hivemind_url,
        gpu_pci_id=gpu_pci_id,
        profile=profile,
        count=int(arguments.get("count") or 1),
        confirm=True,
    )


def hivemind_gpu_passthrough_game_stream_vm(runtime: Any, arguments: dict[str, Any]) -> Any:
    from machine_spirit_4.gateway import gpu_passthrough
    name = require_string(arguments, "name")
    if not arguments.get("confirm"):
        raise ValueError("confirm: true is required")
    extra = {k: v for k, v in arguments.items() if k not in ("name", "confirm")}
    return gpu_passthrough.create_game_stream_vm(
        runtime.hivemind_url, name=name, confirm=True, **extra
    )


def hivemind_game_ensure_available(runtime: Any, arguments: dict[str, Any]) -> Any:
    from machine_spirit_4.gateway import game_admin
    return game_admin.ensure_available(runtime.hivemind_url, require_string(arguments, "game_id"))


def hivemind_game_session_plan(runtime: Any, arguments: dict[str, Any]) -> Any:
    from machine_spirit_4.gateway import game_admin
    game = require_string(arguments, "game")
    kwargs = {k: v for k, v in arguments.items() if k != "game"}
    return game_admin.plan(runtime.hivemind_url, game=game, **kwargs)


def hivemind_game_session_run(runtime: Any, arguments: dict[str, Any]) -> Any:
    from machine_spirit_4.gateway import game_admin
    return game_admin.run(runtime.hivemind_url, require_string(arguments, "job_id"))


def hivemind_game_session_status(runtime: Any, arguments: dict[str, Any]) -> Any:
    from machine_spirit_4.gateway import game_admin
    return game_admin.status(runtime.hivemind_url, require_string(arguments, "job_id"))


def hivemind_game_session_evidence(runtime: Any, arguments: dict[str, Any]) -> Any:
    from machine_spirit_4.gateway import game_admin
    return game_admin.evidence(runtime.hivemind_url, require_string(arguments, "job_id"))


def hivemind_game_session_cancel(runtime: Any, arguments: dict[str, Any]) -> Any:
    from machine_spirit_4.gateway import game_admin
    return game_admin.cancel(runtime.hivemind_url, require_string(arguments, "job_id"))


def hivemind_jobs_cancel(runtime: Any, arguments: dict[str, Any]) -> Any:
    """Cancel one inference trace through HiveMind's UUID-scoped contract."""
    if not bool(arguments.get("confirm")):
        raise ToolInputError("jobs.cancel is destructive. Pass confirm:true to proceed.")
    job_id = require_string(arguments, "job_id")
    from machine_spirit_4.gateway import hivemind_tools as _tools
    return _tools.jobs_cancel(
        runtime.hivemind_url,
        job_id=job_id,
        reason=optional_string(arguments, "reason"),
    )


def build_tool_registry() -> dict[str, ToolDef]:
    tools = [
        ToolDef(
            "ms4.identity.verify@v1",
            "Verify active MS4 spirit identity through the MS3 sidecar.",
            {"type": "object", "properties": {"spirit_id": {"type": "string"}}},
            read_only_annotations("MS4 identity verify"),
            identity_verify,
        ),
        ToolDef(
            "ms4.identity.state@v1",
            "Return compact MS3 identity and runtime state.",
            {"type": "object", "properties": {}},
            read_only_annotations("MS4 identity state"),
            identity_state,
        ),
        ToolDef(
            "ms4.ethics.evaluate@v1",
            "Evaluate an ActionIntent.v1 through MS3 Great Lense sidecar.",
            {"type": "object", "required": ["intent"], "properties": {"intent": {"type": "object"}}},
            read_only_annotations("MS4 ethics evaluate"),
            ethics_evaluate,
        ),
        ToolDef(
            "ms4.voice.status@v1",
            "Return MS4/MS3 voice readiness and ASR fail-closed state.",
            {"type": "object", "properties": {}},
            read_only_annotations("MS4 voice status"),
            voice_status,
        ),
        ToolDef(
            "ms4.sessions.list@v1",
            "List fused MS4/Hermes sessions.",
            {"type": "object", "properties": {}},
            read_only_annotations("MS4 sessions list"),
            sessions_list,
        ),
        ToolDef(
            "ms4.chat.send@v1",
            "Send a chat turn through the fused Hermes/MS4 runtime.",
            {"type": "object", "required": ["message"], "properties": {"message": {"type": "string"}, "session_id": {"type": "string"}, "model": {"type": "string"}}},
            chat_annotations("MS4 chat send"),
            chat_send,
        ),
        ToolDef(
            "ms4.models.list@v1",
            "Return the MS4/HiveMind model catalog view.",
            {"type": "object", "properties": {}},
            read_only_annotations("MS4 models list"),
            models_list,
        ),
        ToolDef(
            "ms4.vision.analyze_local@v1",
            "Analyze a local image file through a HiveMind VLM. Read-only; validates file type and sends a base64 image payload to HiveMind.",
            {"type": "object", "required": ["image_path"], "properties": {"image_path": {"type": "string"}, "question": {"type": "string"}, "model": {"type": "string"}}},
            read_only_annotations("MS4 local image VLM analysis"),
            vision_analyze_local,
        ),
        ToolDef(
            "ms4.doctrine.explain@v1",
            "Return deterministic TMR doctrine grounding.",
            {"type": "object", "properties": {"topic": {"type": "string"}}},
            read_only_annotations("MS4 doctrine explain"),
            doctrine_explain,
        ),
        ToolDef(
            "ms4.hivemind.inventory@v1",
            "Return deterministic live HiveMind node/GPU inventory through read-only HiveMind MCP tools.",
            {"type": "object", "properties": {"include_gpu_details": {"type": "boolean"}, "status_filter": {"type": "string"}}},
            read_only_annotations("MS4 HiveMind inventory"),
            hivemind_inventory,
        ),
        ToolDef(
            "ms4.nibbles.dry_run@v1",
            "Validate Nibbles scare/physical dry-run intents only; never triggers hardware.",
            {"type": "object", "required": ["intent"], "properties": {"intent": {"type": "object"}}},
            dry_run_annotations("MS4 Nibbles dry run"),
            nibbles_dry_run,
        ),
        ToolDef(
            "ms4.hermes.tools.list@v1",
            "List Hermes tools exposed through the full MS4 body.",
            {"type": "object", "properties": {"toolsets": {"type": "array", "items": {"type": "string"}}}},
            read_only_annotations("MS4 Hermes tools list"),
            hermes_tools_list,
        ),
        ToolDef(
            "ms4.hermes.tool.call@v1",
            "Dispatch a Hermes tool through MS4. Effectful tools are mediated by ms4_consciousness and MS3 ethics.",
            {"type": "object", "required": ["tool"], "properties": {"tool": {"type": "string"}, "args": {"type": "object"}, "session_id": {"type": "string"}}},
            chat_annotations("MS4 Hermes tool call"),
            hermes_tool_call,
        ),
        ToolDef(
            "ms4.runtime.deps.status@v1",
            "Report contained MS4 runtime dependency and capability status.",
            {"type": "object", "properties": {}},
            read_only_annotations("MS4 runtime dependency status"),
            runtime_deps_status,
        ),
        ToolDef(
            "ms4.hermes.version@v1",
            "Report current vs latest Hermes Agent version (public GitHub releases). Read-only.",
            {"type": "object", "properties": {"refresh": {"type": "boolean"}}},
            read_only_annotations("MS4 Hermes version"),
            hermes_version,
        ),
        ToolDef(
            "ms4.hermes.releases@v1",
            "List recent Hermes Agent releases for the pin-a-version selector. Cached ~1h.",
            {"type": "object", "properties": {"refresh": {"type": "boolean"}}},
            read_only_annotations("MS4 Hermes releases"),
            hermes_releases,
        ),
        ToolDef(
            "ms4.hermes.update@v1",
            "Trigger a Hermes Agent upgrade to the latest release or a pinned target_version. Spawns a background job; poll ms4.hermes.update.status@v1.",
            {"type": "object", "properties": {"target_version": {"type": "string"}, "version": {"type": "string"}, "request_user": {"type": "string"}}},
            chat_annotations("MS4 Hermes update"),
            hermes_update,
        ),
        ToolDef(
            "ms4.hermes.update.status@v1",
            "Report current/last Hermes upgrade job phase and progress trail.",
            {"type": "object", "properties": {}},
            read_only_annotations("MS4 Hermes update status"),
            hermes_update_status,
        ),
        ToolDef(
            "ms4.double_agent.submit@v1",
            "Submit a Double Agent background job. Enforces phase-1 authority (can_mutate_world must be false) and safe-id allowlists. Spawns a contained worker; poll ms4.double_agent.status@v1 for progress.",
            {
                "type": "object",
                "required": ["parent_conversation_id", "conversation_revision_id", "user_visible_goal"],
                "properties": {
                    "job_id": {"type": "string"},
                    "parent_conversation_id": {"type": "string"},
                    "conversation_revision_id": {"type": "integer"},
                    "background_lobe_type": {"type": "string"},
                    "user_visible_goal": {"type": "string"},
                    "internal_goal": {"type": "string"},
                    "authority": {"type": "object"},
                    "resource_request": {"type": "object"},
                    "status_policy": {"type": "object"},
                    "priority": {"type": "string"},
                    "latency_class": {"type": "string"},
                    "request_user": {"type": "string"},
                },
            },
            chat_annotations("MS4 Double Agent submit"),
            double_agent_submit,
        ),
        ToolDef(
            "ms4.double_agent.status@v1",
            "Current snapshot for one Double Agent job: state, last safe_user_status, is_stale, last event type/timestamp.",
            {"type": "object", "required": ["job_id"], "properties": {"job_id": {"type": "string"}}},
            read_only_annotations("MS4 Double Agent status"),
            double_agent_status,
        ),
        ToolDef(
            "ms4.double_agent.list@v1",
            "List Double Agent jobs, optionally filtered by conversation_id and state(s).",
            {
                "type": "object",
                "properties": {
                    "conversation_id": {"type": "string"},
                    "states": {"type": "array", "items": {"type": "string"}},
                    "limit": {"type": "integer"},
                },
            },
            read_only_annotations("MS4 Double Agent list"),
            double_agent_list,
        ),
        ToolDef(
            "ms4.double_agent.events@v1",
            "Recent allowlisted lifecycle events for a Double Agent job. Anti-hallucination contract: raw model tokens are never returned.",
            {
                "type": "object",
                "required": ["job_id"],
                "properties": {
                    "job_id": {"type": "string"},
                    "after_event_id": {"type": "string"},
                    "limit": {"type": "integer"},
                },
            },
            read_only_annotations("MS4 Double Agent events"),
            double_agent_events,
        ),
        ToolDef(
            "ms4.double_agent.cancel@v1",
            "Request cancellation of a running Double Agent job. Idempotent on already-finished jobs.",
            {"type": "object", "required": ["job_id"], "properties": {"job_id": {"type": "string"}}},
            chat_annotations("MS4 Double Agent cancel"),
            double_agent_cancel,
        ),
        ToolDef(
            "ms4.double_agent.mark_stale@v1",
            "Mark a Double Agent job stale (e.g. when the user changes direction). The worker also gets a cancel signal so it stops doing useless work.",
            {
                "type": "object",
                "required": ["job_id"],
                "properties": {
                    "job_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
            },
            chat_annotations("MS4 Double Agent mark stale"),
            double_agent_mark_stale,
        ),
        ToolDef(
            "ms4.desktop.status@v1",
            "Report Windows desktop-control readiness, screen state, and active window context.",
            {"type": "object", "properties": {}},
            read_only_annotations("MS4 desktop status"),
            desktop_status,
        ),
        ToolDef(
            "ms4.desktop.capture@v1",
            "Capture desktop screenshot metadata and optionally a base64 image.",
            {"type": "object", "properties": {"include_image": {"type": "boolean"}, "monitor": {"type": "integer"}}},
            read_only_annotations("MS4 desktop capture"),
            desktop_capture,
        ),
        ToolDef(
            "ms4.desktop.action@v1",
            "Perform a local desktop UI action through MS4 after MS3 ethics mediation and hard safety checks.",
            {"type": "object", "required": ["action"], "properties": {"action": {"type": "string"}, "x": {"type": "integer"}, "y": {"type": "integer"}, "text": {"type": "string"}, "keys": {"type": "array", "items": {"type": "string"}}, "seconds": {"type": "number"}}},
            chat_annotations("MS4 desktop action"),
            desktop_action,
        ),
        # ----- HiveMind admin proxies (May 25 2026) -----
        # These let other agents drive HiveMind through MS4 so MS4's
        # ethics + audit pipeline applies. They are intentionally a
        # subset — the most operationally useful tools from the
        # May-2026 HiveMind catalog expansion (143 tools total). All
        # are thin proxies over the matching gateway module; failures
        # surface as MCP isError replies via the tool registry.
        ToolDef(
            "ms4.hivemind.time@v1",
            "Authoritative cluster time via hivemind.time.now@v1.",
            {"type": "object", "properties": {}},
            read_only_annotations("MS4 HiveMind time"),
            hivemind_time,
        ),
        ToolDef(
            "ms4.hivemind.capability_matrix@v1",
            "Per-node capability matrix (GIMs, models, vGPU) via hivemind.capability.matrix@v1.",
            {"type": "object", "properties": {}},
            read_only_annotations("MS4 HiveMind capability matrix"),
            hivemind_capability_matrix,
        ),
        ToolDef(
            "ms4.hivemind.vms@v1",
            "VM inventory + GPU assignments snapshot (joins hivemind.vm.list@v1 + hivemind.vm.gpus@v1).",
            {"type": "object", "properties": {}},
            read_only_annotations("MS4 HiveMind VMs"),
            hivemind_vms_list,
        ),
        ToolDef(
            "ms4.hivemind.vm.start@v1",
            "Start a HiveMind VM by id.",
            {"type": "object", "required": ["vm_id"], "properties": {"vm_id": {"type": "string"}}},
            chat_annotations("MS4 HiveMind VM start"),
            hivemind_vm_start,
        ),
        ToolDef(
            "ms4.hivemind.vm.stop@v1",
            "Graceful stop of a HiveMind VM (ACPI shutdown).",
            {"type": "object", "required": ["vm_id"], "properties": {"vm_id": {"type": "string"}}},
            chat_annotations("MS4 HiveMind VM stop"),
            hivemind_vm_stop,
        ),
        ToolDef(
            "ms4.hivemind.vm.screenshot@v1",
            "Capture a HiveMind VM's display via hivemind.vm.screenshot@v1.",
            {"type": "object", "required": ["vm_id"], "properties": {"vm_id": {"type": "string"}}},
            read_only_annotations("MS4 HiveMind VM screenshot"),
            hivemind_vm_screenshot,
        ),
        ToolDef(
            "ms4.hivemind.apps@v1",
            "Cluster app inventory with status + metrics joined per-app.",
            {"type": "object", "properties": {}},
            read_only_annotations("MS4 HiveMind apps"),
            hivemind_apps_list,
        ),
        ToolDef(
            "ms4.hivemind.app.start@v1",
            "Start a HiveMind-registered app by id.",
            {"type": "object", "required": ["app_id"], "properties": {"app_id": {"type": "string"}}},
            chat_annotations("MS4 HiveMind app start"),
            hivemind_app_start,
        ),
        ToolDef(
            "ms4.hivemind.app.stop@v1",
            "Stop a HiveMind-registered app by id.",
            {"type": "object", "required": ["app_id"], "properties": {"app_id": {"type": "string"}}},
            chat_annotations("MS4 HiveMind app stop"),
            hivemind_app_stop,
        ),
        ToolDef(
            "ms4.hivemind.storage@v1",
            "Storage snapshot: pools + volumes + snapshots + status.",
            {"type": "object", "properties": {}},
            read_only_annotations("MS4 HiveMind storage"),
            hivemind_storage_snapshot,
        ),
        ToolDef(
            "ms4.hivemind.network@v1",
            "Network snapshot: networks + bridges + interfaces + attachments.",
            {"type": "object", "properties": {}},
            read_only_annotations("MS4 HiveMind network"),
            hivemind_network_snapshot,
        ),
        ToolDef(
            "ms4.hivemind.gpu@v1",
            "GPU mode capabilities + vGPU status + current availability.",
            {"type": "object", "properties": {}},
            read_only_annotations("MS4 HiveMind GPU"),
            hivemind_gpu_snapshot,
        ),
        ToolDef(
            "ms4.hivemind.voice_identities@v1",
            "List enrolled voice identities.",
            {"type": "object", "properties": {}},
            read_only_annotations("MS4 HiveMind voice identities"),
            hivemind_voice_identities_list,
        ),
        ToolDef(
            "ms4.hivemind.approval.request@v1",
            "Request human approval for an action via hivemind.human.approval.request@v1.",
            {
                "type": "object",
                "required": ["action_id", "summary"],
                "properties": {
                    "action_id": {"type": "string"},
                    "summary": {"type": "string"},
                    "details": {"type": "object"},
                    "risk_level": {"type": "string"},
                    "timeout_secs": {"type": "integer"},
                },
            },
            chat_annotations("MS4 HiveMind approval request"),
            hivemind_approval_request,
        ),
        ToolDef(
            "ms4.hivemind.approval.status@v1",
            "Poll a previously-issued human approval request.",
            {"type": "object", "required": ["request_id"], "properties": {"request_id": {"type": "string"}}},
            read_only_annotations("MS4 HiveMind approval status"),
            hivemind_approval_status,
        ),
        ToolDef(
            "ms4.hivemind.notify@v1",
            "Fire-and-forget operator notification (Telegram / log).",
            {
                "type": "object",
                "required": ["message"],
                "properties": {
                    "channel": {"type": "string"},
                    "message": {"type": "string"},
                    "severity": {"type": "string"},
                },
            },
            chat_annotations("MS4 HiveMind notify"),
            hivemind_notify,
        ),
        # ----- PsyKyo proxies via HiveMind (May 26 2026) -----
        # Five wrappers over hivemind.psykyo.*; HiveMind's MCP gateway
        # enforces loopback-only PSYKYO_MCP_BASE, so MS4 just relays.
        # benchmark.run drives Cyberpunk 2077 and is gated by PsyKyo's
        # confirm_actuation_token + vlm_consensus_gate promotion.
        ToolDef(
            "ms4.hivemind.psykyo.benchmark.run@v1",
            "Run the PsyKyo Cyberpunk 2077 benchmark operator via hivemind.psykyo.benchmark.run@v1. Launches the game, drives the in-game settings + Run Benchmark UI, waits ~64s, and returns FPS/variance/thermals through PsyKyo's vlm_consensus_gate promotion.",
            {
                "type": "object",
                "properties": {
                    "resolution": {"type": "string"},
                    "preset": {"type": "string"},
                    "ray_tracing": {"type": "string", "enum": ["off", "on", "psycho", "overdrive"]},
                    "upscaler": {"type": "string", "enum": ["off", "dlss", "fsr", "xess"]},
                    "upscaler_quality": {"type": "string", "enum": ["performance", "balanced", "quality", "auto"]},
                    "runs": {"type": "integer"},
                    "allow_launch": {"type": "boolean"},
                    "allow_foreground": {"type": "boolean"},
                    "allow_capture": {"type": "boolean"},
                    "confirm_actuation_token": {"type": "string"},
                },
            },
            chat_annotations("MS4 PsyKyo benchmark run"),
            hivemind_psykyo_benchmark_run,
        ),
        ToolDef(
            "ms4.hivemind.psykyo.benchmark.gap@v1",
            "PsyKyo benchmark pre-flight gap-gate: confirm the operator is reachable and visual boxes still match the live game UI. Idempotent and side-effect-free; does NOT launch the benchmark.",
            {"type": "object", "properties": {}},
            read_only_annotations("MS4 PsyKyo benchmark gap"),
            hivemind_psykyo_benchmark_gap,
        ),
        ToolDef(
            "ms4.hivemind.psykyo.benchmark.workqueue@v1",
            "List queued PsyKyo gap-gate work items (boxes that need review or replay before benchmark.run can be trusted). Read-only.",
            {"type": "object", "properties": {}},
            read_only_annotations("MS4 PsyKyo benchmark workqueue"),
            hivemind_psykyo_benchmark_workqueue,
        ),
        ToolDef(
            "ms4.hivemind.psykyo.evidence.latest@v1",
            "Stable PsyKyo validation pointers and latest evidence summary. Read-only; safe without confirm_actuation_token.",
            {"type": "object", "properties": {}},
            read_only_annotations("MS4 PsyKyo evidence latest"),
            hivemind_psykyo_evidence_latest,
        ),
        ToolDef(
            "ms4.hivemind.psykyo.vlm_consensus@v1",
            "Reconcile saved HiveMind VLM/OCR sidecar artifacts across models (llama3.2-vision + qwen3-vl MUST agree) before semantic promotion. Read-only consensus check on existing evidence; does not call VLM/OCR live and does not mutate profiles.",
            {
                "type": "object",
                "required": ["evidence_ref"],
                "properties": {
                    "evidence_ref": {"type": "string"},
                    "models": {"type": "array", "items": {"type": "string"}},
                },
            },
            chat_annotations("MS4 PsyKyo VLM consensus"),
            hivemind_psykyo_vlm_consensus,
        ),
        # ----- May 26 2026 — new HiveMind admin proxies -----
        ToolDef(
            "ms4.hivemind.oracle.status@v1",
            "HiveMind Oracle planner status snapshot (hivemind.oracle.status).",
            {"type": "object", "properties": {}},
            read_only_annotations("MS4 HiveMind Oracle status"),
            hivemind_oracle_status,
        ),
        ToolDef(
            "ms4.hivemind.oracle.chat@v1",
            "Ask the HiveMind Oracle planner to plan/reason about a request (hivemind.oracle.chat). Suitable for 'what should I do next?' / capacity-aware coordination workflows.",
            {"type": "object", "required": ["message"], "properties": {"message": {"type": "string"}}},
            chat_annotations("MS4 HiveMind Oracle chat"),
            hivemind_oracle_chat,
        ),
        ToolDef(
            "ms4.hivemind.training.backends@v1",
            "List available HiveMind training backends (hivemind.training.backends@v1).",
            {"type": "object", "properties": {}},
            read_only_annotations("MS4 HiveMind training backends"),
            hivemind_training_backends,
        ),
        ToolDef(
            "ms4.hivemind.training.start@v1",
            "Start a HiveMind training job (LoRA / PEFT / forge). Recipe is backend-specific.",
            {
                "type": "object",
                "required": ["recipe"],
                "properties": {"recipe": {"type": "object"}},
            },
            chat_annotations("MS4 HiveMind training start"),
            hivemind_training_start,
        ),
        ToolDef(
            "ms4.hivemind.training.status@v1",
            "Poll the status of a HiveMind training job (all or by job_id).",
            {"type": "object", "properties": {"job_id": {"type": "string"}}},
            read_only_annotations("MS4 HiveMind training status"),
            hivemind_training_status,
        ),
        ToolDef(
            "ms4.hivemind.adapters.list@v1",
            "List HiveMind-known adapters (LoRA outputs from training jobs).",
            {"type": "object", "properties": {}},
            read_only_annotations("MS4 HiveMind adapters list"),
            hivemind_adapters_list,
        ),
        ToolDef(
            "ms4.hivemind.adapters.deploy@v1",
            "Deploy an adapter to a model runtime so it can be loaded for inference.",
            {
                "type": "object",
                "required": ["adapter_id"],
                "properties": {"adapter_id": {"type": "string"}, "target_model": {"type": "string"}},
            },
            chat_annotations("MS4 HiveMind adapters deploy"),
            hivemind_adapters_deploy,
        ),
        ToolDef(
            "ms4.hivemind.loadout.profiles@v1",
            "List configured HiveMind loadout profiles (which models loaded together).",
            {"type": "object", "properties": {}},
            read_only_annotations("MS4 HiveMind loadout profiles"),
            hivemind_loadout_profiles,
        ),
        ToolDef(
            "ms4.hivemind.loadout.apply@v1",
            "Apply a HiveMind loadout profile (load/unload models to match).",
            {"type": "object", "required": ["profile_id"], "properties": {"profile_id": {"type": "string"}}},
            chat_annotations("MS4 HiveMind loadout apply"),
            hivemind_loadout_apply,
        ),
        ToolDef(
            "ms4.hivemind.deploy.gim@v1",
            "Deploy / start a HiveMind GIM by name (hivemind.deploy.gim@v1).",
            {"type": "object", "required": ["gim_name"], "properties": {"gim_name": {"type": "string"}}},
            chat_annotations("MS4 HiveMind deploy GIM"),
            hivemind_deploy_gim,
        ),
        ToolDef(
            "ms4.hivemind.inference.models@v1",
            "Model catalog via the MCP native resilient handler (hivemind.inference.models@v1).",
            {"type": "object", "properties": {}},
            read_only_annotations("MS4 HiveMind inference models"),
            hivemind_inference_models,
        ),
        ToolDef(
            "ms4.hivemind.inference.chat@v1",
            "Direct chat completion through the HiveMind MCP gateway (hivemind.inference.chat@v1). Messages follow OpenAI shape.",
            {
                "type": "object",
                "required": ["messages", "model"],
                "properties": {
                    "messages": {"type": "array"},
                    "model": {"type": "string"},
                    "temperature": {"type": "number"},
                    "max_tokens": {"type": "integer"},
                },
            },
            chat_annotations("MS4 HiveMind inference chat"),
            hivemind_inference_chat,
        ),
        ToolDef(
            "ms4.hivemind.logos.optimize@v1",
            "Run the Logos Machina prompt optimizer on a managed prompt (hivemind.logos.optimize@v1).",
            {"type": "object", "required": ["prompt_id"], "properties": {"prompt_id": {"type": "string"}}},
            chat_annotations("MS4 HiveMind Logos optimize"),
            hivemind_logos_optimize,
        ),
        ToolDef(
            "ms4.hivemind.services.enable@v1",
            "Enable a disabled Warden-managed service by name.",
            {"type": "object", "required": ["service_name"], "properties": {"service_name": {"type": "string"}}},
            chat_annotations("MS4 HiveMind service enable"),
            hivemind_services_enable,
        ),
        ToolDef(
            "ms4.hivemind.services.disable@v1",
            "Disable a running Warden-managed service by name.",
            {"type": "object", "required": ["service_name"], "properties": {"service_name": {"type": "string"}}},
            chat_annotations("MS4 HiveMind service disable"),
            hivemind_services_disable,
        ),
        ToolDef(
            "ms4.hivemind.services.restart@v1",
            "Restart a Warden-managed service by name (Warden's restart route).",
            {"type": "object", "required": ["service_name"], "properties": {"service_name": {"type": "string"}}},
            chat_annotations("MS4 HiveMind service restart"),
            hivemind_services_restart,
        ),
        ToolDef(
            "ms4.hivemind.jobs.cancel@v1",
            "Cancel one inference job by its HiveMind trace UUID. Requires confirm:true.",
            {
                "type": "object",
                "required": ["confirm", "job_id"],
                "properties": {
                    "confirm": {"type": "boolean"},
                    "job_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
            },
            chat_annotations("MS4 HiveMind jobs cancel"),
            hivemind_jobs_cancel,
        ),
        # ----- Game session orchestration (Phase 1 dry-run) -----
        ToolDef(
            "ms4.hivemind.game.ensure_available@v1",
            "Read-only availability check for a game id (e.g. 'cyberpunk-2077'). Resolves to env override path / default install / golden VHDX. Returns 'available + path' or 'unavailable + remediation'. Never installs.",
            {"type": "object", "required": ["game_id"], "properties": {"game_id": {"type": "string"}}},
            read_only_annotations("MS4 HiveMind game availability"),
            hivemind_game_ensure_available,
        ),
        ToolDef(
            "ms4.hivemind.game_session.plan@v1",
            "Produce a dry-run Plan for a game-streaming workload intent. Probes hosts/GPU/VMs when reachable, falls back to a synthetic demo plan. Never reserves resources. Returns {job_id, plan}.",
            {
                "type": "object",
                "required": ["game"],
                "properties": {
                    "game": {"type": "string"},
                    "client": {"type": "string"},
                    "duration_hint": {"type": "string"},
                    "latency": {"type": "string"},
                    "quality": {"type": "string"},
                },
            },
            read_only_annotations("MS4 HiveMind game_session plan"),
            hivemind_game_session_plan,
        ),
        ToolDef(
            "ms4.hivemind.game_session.run@v1",
            "Walk the simulated state machine for a planned game_session job. Pure dry-run in Phase 1: every phase records 'would_call <hivemind.vm.X@v1>' evidence but never mutates. Returns when terminal (COMPLETE/FAILED_*/CANCELLED).",
            {"type": "object", "required": ["job_id"], "properties": {"job_id": {"type": "string"}}},
            chat_annotations("MS4 HiveMind game_session run"),
            hivemind_game_session_run,
        ),
        ToolDef(
            "ms4.hivemind.game_session.status@v1",
            "Current state-machine position for a game_session job_id + per-phase transitions + dry_run flag.",
            {"type": "object", "required": ["job_id"], "properties": {"job_id": {"type": "string"}}},
            read_only_annotations("MS4 HiveMind game_session status"),
            hivemind_game_session_status,
        ),
        ToolDef(
            "ms4.hivemind.game_session.evidence@v1",
            "Full per-phase evidence ledger for a game_session job_id (transitions, simulated actions, planner inputs, last_error). In-memory only in Phase 1.",
            {"type": "object", "required": ["job_id"], "properties": {"job_id": {"type": "string"}}},
            read_only_annotations("MS4 HiveMind game_session evidence"),
            hivemind_game_session_evidence,
        ),
        ToolDef(
            "ms4.hivemind.game_session.cancel@v1",
            "Request cancellation of one game_session job.",
            {"type": "object", "required": ["job_id"], "properties": {"job_id": {"type": "string"}}},
            chat_annotations("MS4 HiveMind game_session cancel"),
            hivemind_game_session_cancel,
        ),
        # ----- GPU passthrough workflow (GPU-P / DDA / vGPU) -----
        ToolDef(
            "ms4.hivemind.gpu.passthrough.snapshot@v1",
            "Combined Ms4GpuPassthroughSnapshot.v1: gpu_mode.capabilities + vm.gpus + gpu_mode.vgpu_status + gpu.availability + per-mode (GPU-P / DDA / vGPU) availability/licensing notes. Read-only.",
            {"type": "object", "properties": {}},
            read_only_annotations("MS4 HiveMind GPU passthrough snapshot"),
            hivemind_gpu_passthrough_snapshot,
        ),
        ToolDef(
            "ms4.hivemind.gpu.passthrough.prepare@v1",
            "Switch a GPU to 'passthrough' (Hyper-V DDA path; works on consumer SKUs for rebind, requires Windows Server license for actual VM attach) or 'vgpu' (NVIDIA vGPU stack). Driver rebind is destructive — dismounts the GPU from the host. Requires confirm:true.",
            {
                "type": "object",
                "required": ["gpu_pci_id", "desired_mode", "confirm"],
                "properties": {
                    "gpu_pci_id": {"type": "string", "description": "PCI address, e.g. '0000:01:00.0'"},
                    "desired_mode": {"type": "string", "enum": ["passthrough", "vgpu"]},
                    "vm_uuid": {"type": "string"},
                    "confirm": {"type": "boolean"},
                },
            },
            chat_annotations("MS4 HiveMind GPU passthrough prepare"),
            hivemind_gpu_passthrough_prepare,
        ),
        ToolDef(
            "ms4.hivemind.gpu.passthrough.vgpu@v1",
            "Create one or more vGPU mediated device (mdev) instances on a GPU. Requires the GPU to already be in 'vgpu' mode (call passthrough.prepare with desired_mode='vgpu' first) and an NVIDIA vGPU license. Requires confirm:true.",
            {
                "type": "object",
                "required": ["gpu_pci_id", "profile", "confirm"],
                "properties": {
                    "gpu_pci_id": {"type": "string"},
                    "profile": {"type": "string", "description": "Profile name under /sys/.../mdev_supported_types/"},
                    "count": {"type": "integer", "minimum": 1, "maximum": 64, "default": 1},
                    "confirm": {"type": "boolean"},
                },
            },
            chat_annotations("MS4 HiveMind GPU vGPU create"),
            hivemind_gpu_passthrough_vgpu,
        ),
        ToolDef(
            "ms4.hivemind.gpu.passthrough.game_stream_vm@v1",
            "Provision a Windows 11 GPU-P (Hyper-V GPU Partitioning) game-streaming VM via hivemind.vm.create_prebuilt(vm_type='windows_game_stream_prebuilt') + vm.deploy. This is the consumer-licensed path — works on Windows 11 + any modern NVIDIA GPU. Requires confirm:true.",
            {
                "type": "object",
                "required": ["name", "confirm"],
                "properties": {
                    "name": {"type": "string", "description": "VM name; must match ^[A-Za-z0-9._-]+$"},
                    "confirm": {"type": "boolean"},
                },
            },
            chat_annotations("MS4 HiveMind GPU-P game stream VM"),
            hivemind_gpu_passthrough_game_stream_vm,
        ),
    ]
    registry = {tool.name: tool for tool in tools}
    _append_imported_tools(registry)
    return registry


def _imported_annotations(record: dict[str, Any]) -> dict[str, Any]:
    """Derive MCP annotations for an imported tool from its bridge
    record (fail-closed: unless classified read_only, treat as
    non-read-only/open-world)."""
    ann = record.get("annotations") if isinstance(record.get("annotations"), dict) else {}
    read_only = record.get("kind") == "read_only" or bool(ann.get("readOnlyHint"))
    return {
        "title": f"[ext:{record.get('server_id', '?')}] {record.get('upstream_name', record.get('name', ''))}",
        "readOnlyHint": bool(read_only),
        "destructiveHint": bool(ann.get("destructiveHint", False)),
        "idempotentHint": bool(ann.get("idempotentHint", False)),
        "openWorldHint": True,  # 3rd-party tool — assume it touches the outside world
    }


def _make_imported_handler(tool_name: str) -> Callable[[Any, dict[str, Any]], Any]:
    """Build a handler that proxies an imported tool call to its
    upstream MCP server via the mcp_bridge proxy."""

    def _handler(_runtime: Any, arguments: dict[str, Any]) -> Any:
        from machine_spirit_4.mcp_bridge.proxy import call_imported_tool

        out = call_imported_tool(tool_name, arguments or {})
        if not isinstance(out, dict) or not out.get("ok"):
            raise RuntimeError(out.get("error") if isinstance(out, dict) else "imported tool call failed")
        return {
            "tool": tool_name,
            "server_id": out.get("server_id"),
            "is_error": bool(out.get("is_error")),
            "result": out.get("result"),
        }

    return _handler


def _append_imported_tools(registry: dict[str, ToolDef]) -> None:
    """Append imported 3rd-party MCP tools (from the mcp_bridge
    registry) to the MS4 MCP tool registry. Fail-soft: a missing or
    broken registry simply yields no imported tools."""
    try:
        from machine_spirit_4.mcp_bridge.registry import imported_tool_records

        records = imported_tool_records()
    except Exception:  # noqa: BLE001 — never let the bridge break the core registry
        return
    for record in records:
        name = record.get("name")
        if not isinstance(name, str) or not name or name in registry:
            continue
        input_schema = record.get("input_schema")
        if not isinstance(input_schema, dict):
            input_schema = {"type": "object", "properties": {}}
        description = str(record.get("description") or "").strip()
        registry[name] = ToolDef(
            name=name,
            description=f"[ext:{record.get('server_id', '?')}] {description}".strip(),
            input_schema=input_schema,
            annotations=_imported_annotations(record),
            handler=_make_imported_handler(name),
        )


def result_content(data: Any) -> dict[str, Any]:
    return {"content": [{"type": MCP_TEXT, "text": _json(data)}], "isError": False}


def error_content(message: str) -> dict[str, Any]:
    return {"content": [{"type": MCP_TEXT, "text": message}], "isError": True}
