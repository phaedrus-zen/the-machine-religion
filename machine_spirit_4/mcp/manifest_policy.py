from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Collection


MANIFEST_PATH = Path(__file__).with_name("manifest.json")
MAX_ARGUMENT_BYTES = 1024 * 1024
RUNTIME_ACTION_KIND = "runtime_action"
RUNTIME_POLICY_FIELDS = ("gated_by", "idempotency", "cancellation", "audit")
GATE_MODES = frozenset({"none", "confirm", "deny_until_implemented"})
IDEMPOTENCY_MODES = frozenset({"none", "idempotent"})
CANCELLATION_MODES = frozenset({"none"})
NONE_GATED_RUNTIME_ACTIONS = frozenset(
    {
        "ms4.chat.send@v1",
        "ms4.double_agent.cancel@v1",
        "ms4.double_agent.mark_stale@v1",
        "ms4.hivemind.oracle.chat@v1",
        "ms4.hivemind.inference.chat@v1",
        "ms4.hivemind.game_session.cancel@v1",
    }
)
CONFIRM_GATED_RUNTIME_ACTIONS = frozenset(
    {
        "ms4.hermes.tool.call@v1",
        "ms4.hermes.update@v1",
        "ms4.desktop.action@v1",
        "ms4.hivemind.approval.request@v1",
        "ms4.hivemind.notify@v1",
        "ms4.hivemind.logos.optimize@v1",
        "ms4.hivemind.jobs.cancel@v1",
        "ms4.hivemind.gpu.passthrough.prepare@v1",
        "ms4.hivemind.gpu.passthrough.vgpu@v1",
        "ms4.hivemind.gpu.passthrough.game_stream_vm@v1",
    }
)
DENIED_RUNTIME_ACTIONS = frozenset(
    {
        "ms4.double_agent.submit@v1",
        "ms4.hivemind.vm.start@v1",
        "ms4.hivemind.vm.stop@v1",
        "ms4.hivemind.app.start@v1",
        "ms4.hivemind.app.stop@v1",
        "ms4.hivemind.psykyo.benchmark.run@v1",
        "ms4.hivemind.psykyo.vlm_consensus@v1",
        "ms4.hivemind.training.start@v1",
        "ms4.hivemind.adapters.deploy@v1",
        "ms4.hivemind.loadout.apply@v1",
        "ms4.hivemind.deploy.gim@v1",
        "ms4.hivemind.services.enable@v1",
        "ms4.hivemind.services.disable@v1",
        "ms4.hivemind.services.restart@v1",
        "ms4.hivemind.game_session.run@v1",
    }
)
GATE_BY_RUNTIME_ACTION = {
    **{name: "none" for name in NONE_GATED_RUNTIME_ACTIONS},
    **{name: "confirm" for name in CONFIRM_GATED_RUNTIME_ACTIONS},
    **{name: "deny_until_implemented" for name in DENIED_RUNTIME_ACTIONS},
}
IDEMPOTENT_RUNTIME_ACTIONS = frozenset(
    {
        "ms4.hermes.update@v1",
        "ms4.double_agent.cancel@v1",
    }
)
ACTION_STATES = frozenset(
    {
        "approval_required",
        "denied",
        "running",
        "committed",
        "cancelled",
        "reconciliation_required",
    }
)


class ManifestPolicyError(ValueError):
    pass


def load_manifest(path: Path | str | None = None) -> dict[str, Any]:
    source = Path(path) if path is not None else MANIFEST_PATH
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestPolicyError(f"cannot load MCP manifest {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ManifestPolicyError("MCP manifest root must be an object")
    return payload


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_manifest(
    manifest: dict[str, Any],
    *,
    registry_names: Collection[str] | None = None,
    registry_runtime_actions: Collection[str] | None = None,
) -> dict[str, dict[str, Any]]:
    errors: list[str] = []
    tools = manifest.get("tools")
    if not isinstance(tools, list):
        raise ManifestPolicyError("manifest.tools must be an array")

    by_name: dict[str, dict[str, Any]] = {}
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict) or not _nonempty_string(tool.get("name")):
            errors.append(f"tools[{index}]: name must be a non-empty string")
            continue
        name = tool["name"]
        if name in by_name:
            errors.append(f"{name}: duplicate tool declaration")
            continue
        by_name[name] = tool

    runtime_names = {
        name for name, tool in by_name.items() if tool.get("kind") == RUNTIME_ACTION_KIND
    }
    safety = manifest.get("safety")
    effectful = safety.get("effectful_tools_in_v1") if isinstance(safety, dict) else None
    if not isinstance(effectful, list) or not all(_nonempty_string(name) for name in effectful):
        errors.append("safety.effectful_tools_in_v1 must be an array of non-empty strings")
        effectful_names: set[str] = set()
    else:
        effectful_names = set(effectful)
        missing = sorted(runtime_names - effectful_names)
        extra = sorted(effectful_names - runtime_names)
        if missing:
            errors.append(f"safety.effectful_tools_in_v1 missing runtime actions: {missing}")
        if extra:
            errors.append(f"safety.effectful_tools_in_v1 has non-runtime actions: {extra}")

    for name in sorted(runtime_names):
        policy = by_name[name]
        for field in RUNTIME_POLICY_FIELDS:
            if field not in policy:
                errors.append(f"{name}: missing {field}")

        gates = policy.get("gated_by")
        if not (
            isinstance(gates, list)
            and len(gates) == 1
            and gates[0] in GATE_MODES
        ):
            errors.append(f"{name}: gated_by must contain exactly one of {sorted(GATE_MODES)}")
        else:
            expected_gate = GATE_BY_RUNTIME_ACTION.get(name)
            if expected_gate is None:
                errors.append(f"{name}: no server-enforced gate policy is registered")
            elif gates != [expected_gate]:
                errors.append(f"{name}: gated_by must be [{expected_gate!r}]")
        if not _nonempty_string(policy.get("gate_rationale")):
            errors.append(f"{name}: gate_rationale must be a non-empty string")

        bounds = policy.get("bounds")
        max_bytes = bounds.get("max_arguments_bytes") if isinstance(bounds, dict) else None
        if (
            not isinstance(max_bytes, int)
            or isinstance(max_bytes, bool)
            or not 0 < max_bytes <= MAX_ARGUMENT_BYTES
        ):
            errors.append(
                f"{name}: bounds.max_arguments_bytes must be between 1 and {MAX_ARGUMENT_BYTES}"
            )

        idempotency = policy.get("idempotency")
        if not isinstance(idempotency, dict):
            errors.append(f"{name}: idempotency must be an object")
        else:
            if idempotency.get("mode") not in IDEMPOTENCY_MODES:
                errors.append(
                    f"{name}: idempotency.mode must be one of {sorted(IDEMPOTENCY_MODES)}"
                )
            else:
                expected_idempotency = (
                    "idempotent" if name in IDEMPOTENT_RUNTIME_ACTIONS else "none"
                )
                if idempotency.get("mode") != expected_idempotency:
                    errors.append(
                        f"{name}: idempotency.mode must be {expected_idempotency}"
                    )
            if not _nonempty_string(idempotency.get("rationale")):
                errors.append(f"{name}: idempotency.rationale must be a non-empty string")

        cancellation = policy.get("cancellation")
        if not isinstance(cancellation, dict):
            errors.append(f"{name}: cancellation must be an object")
        else:
            mode = cancellation.get("mode")
            if mode not in CANCELLATION_MODES:
                errors.append(
                    f"{name}: cancellation.mode must be one of {sorted(CANCELLATION_MODES)}"
                )
            if not _nonempty_string(cancellation.get("rationale")):
                errors.append(f"{name}: cancellation.rationale must be a non-empty string")
        audit_policy = policy.get("audit")
        if not isinstance(audit_policy, dict):
            errors.append(f"{name}: audit must be an object")
        else:
            if audit_policy.get("mode") != "required":
                errors.append(f"{name}: audit.mode must be required")
            if audit_policy.get("event") != "ms4_mcp_runtime_action":
                errors.append(f"{name}: audit.event must be ms4_mcp_runtime_action")

    if registry_names is not None:
        registry_name_set = set(registry_names)
        missing_registry = sorted(set(by_name) - registry_name_set)
        if missing_registry:
            errors.append(f"manifest tools missing from registry: {missing_registry}")
        unmanaged_native = sorted(
            name
            for name in registry_name_set - set(by_name)
            if isinstance(name, str) and name.startswith("ms4.")
        )
        if unmanaged_native:
            errors.append(f"native registry tools missing from manifest: {unmanaged_native}")

    if registry_runtime_actions is not None:
        registered_runtime_names = {
            name
            for name in registry_runtime_actions
            if isinstance(name, str) and name.startswith("ms4.")
        }
        omitted = sorted(registered_runtime_names - runtime_names)
        misclassified = sorted(runtime_names - registered_runtime_names)
        if omitted:
            errors.append(f"native runtime actions missing policy: {omitted}")
        if misclassified:
            errors.append(f"manifest runtime actions not runtime actions in registry: {misclassified}")

    if errors:
        raise ManifestPolicyError("\n".join(errors))
    return by_name


def load_and_validate_manifest(
    *,
    path: Path | str | None = None,
    registry_names: Collection[str] | None = None,
    registry_runtime_actions: Collection[str] | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = load_manifest(path)
    return manifest, validate_manifest(
        manifest,
        registry_names=registry_names,
        registry_runtime_actions=registry_runtime_actions,
    )


def admission_state(policy: dict[str, Any], arguments: dict[str, Any]) -> tuple[str, str | None]:
    max_bytes = policy["bounds"]["max_arguments_bytes"]
    try:
        argument_bytes = len(
            json.dumps(
                arguments,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
    except (TypeError, ValueError, UnicodeError):
        return "denied", "arguments_not_utf8_json"
    if argument_bytes > max_bytes:
        return "denied", "arguments_too_large"

    gate = policy["gated_by"][0]
    if gate == "none":
        return "running", None
    if gate == "confirm":
        if arguments.get("confirm") is True:
            return "running", None
        return "approval_required", "confirm_true_required"
    if gate == "deny_until_implemented":
        return "denied", "gate_not_implemented"
    raise ManifestPolicyError(f"unsupported gate mode after validation: {gate}")


def decorate_mcp_tool(tool: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    decorated = copy.deepcopy(tool)
    metadata = {field: copy.deepcopy(policy[field]) for field in RUNTIME_POLICY_FIELDS}
    metadata.update(
        {
            "kind": RUNTIME_ACTION_KIND,
            "gate_rationale": policy["gate_rationale"],
            "bounds": copy.deepcopy(policy["bounds"]),
            "action_states": sorted(ACTION_STATES),
        }
    )
    decorated.setdefault("_meta", {})["ms4/runtimeActionPolicy"] = metadata
    decorated.setdefault("annotations", {})["idempotentHint"] = (
        policy["idempotency"]["mode"] == "idempotent"
    )

    if policy["gated_by"] == ["confirm"]:
        schema = decorated.setdefault("inputSchema", {"type": "object"})
        properties = schema.setdefault("properties", {})
        properties["confirm"] = {
            "type": "boolean",
            "const": True,
            "description": "Explicit approval for this runtime action.",
        }
        required = schema.setdefault("required", [])
        if "confirm" not in required:
            required.append("confirm")
    return decorated
