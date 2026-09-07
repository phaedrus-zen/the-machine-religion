import json
import subprocess
import sys
from pathlib import Path

import pytest

from machine_spirit_4.mcp.manifest_policy import ManifestPolicyError, validate_manifest
from machine_spirit_4.mcp.tools import build_tool_registry


ROOT = Path(__file__).resolve().parents[2]
MCP = ROOT / "machine_spirit_4" / "mcp"
RUNTIME_ACTION_NAMES = tuple(
    tool["name"]
    for tool in json.loads((MCP / "manifest.json").read_text(encoding="utf-8"))["tools"]
    if tool.get("kind") == "runtime_action"
)


def test_manifest_lists_all_v1_tools_and_safety_posture():
    manifest = json.loads((MCP / "manifest.json").read_text(encoding="utf-8"))
    tool_names = {tool["name"] for tool in manifest["tools"]}

    assert manifest["schema"] == "Ms4McpManifest.v1"
    assert manifest["endpoint"] == "http://127.0.0.1:9181/mcp"
    assert manifest["safety"]["default_posture"] == "fail_closed"
    # Count of tools in the manifest. Original v1 had 27; the May-25
    # 2026 HiveMind catalog expansion added 16 hivemind.* proxy tools,
    # and the May-26 PsyKyo bridge added 5 more typed wrappers, bringing
    # the total to 48. Bump this when adding/removing tools so the
    # manifest + registry never drift.
    # 27 original v1 + 16 hivemind.* proxies (May 25) + 5 hivemind.psykyo.*
    # (May 26 PsyKyo round) + 17 hivemind.* admin proxies (May 26 fill-
    # all-gaps round: oracle, training, adapters, loadout, deploy.gim,
    # inference, logos.optimize, services.{enable,disable,restart},
    # jobs.cancel) + 6 hivemind.game.* / hivemind.game_session.* proxies
    # (May 26 game-session round: ensure_available + plan/run/status/
    # evidence/cancel) + 4 hivemind.gpu.passthrough.* proxies
    # (May 27 GPU-P / DDA / vGPU round: snapshot + prepare + vgpu +
    # game_stream_vm). Bump when adding / removing tools.
    assert len(tool_names) == 75
    assert "ms4.chat.send@v1" in tool_names
    assert "ms4.nibbles.dry_run@v1" in tool_names
    assert "ms4.hermes.tools.list@v1" in tool_names
    assert "ms4.hermes.tool.call@v1" in tool_names
    assert "ms4.runtime.deps.status@v1" in tool_names
    assert "ms4.vision.analyze_local@v1" in tool_names
    assert "ms4.desktop.status@v1" in tool_names
    assert "ms4.desktop.capture@v1" in tool_names
    assert "ms4.desktop.action@v1" in tool_names
    assert "ms4.hermes.version@v1" in tool_names
    assert "ms4.hermes.releases@v1" in tool_names
    assert "ms4.hermes.update@v1" in tool_names
    assert "ms4.hermes.update.status@v1" in tool_names
    assert "ms4.hermes.update@v1" in manifest["safety"]["effectful_tools_in_v1"]
    for da_tool in (
        "ms4.double_agent.submit@v1",
        "ms4.double_agent.status@v1",
        "ms4.double_agent.list@v1",
        "ms4.double_agent.events@v1",
        "ms4.double_agent.cancel@v1",
        "ms4.double_agent.mark_stale@v1",
    ):
        assert da_tool in tool_names
    for effectful_da in (
        "ms4.double_agent.submit@v1",
        "ms4.double_agent.cancel@v1",
        "ms4.double_agent.mark_stale@v1",
    ):
        assert effectful_da in manifest["safety"]["effectful_tools_in_v1"]
    for psykyo_tool in (
        "ms4.hivemind.psykyo.benchmark.run@v1",
        "ms4.hivemind.psykyo.benchmark.gap@v1",
        "ms4.hivemind.psykyo.benchmark.workqueue@v1",
        "ms4.hivemind.psykyo.evidence.latest@v1",
        "ms4.hivemind.psykyo.vlm_consensus@v1",
    ):
        assert psykyo_tool in tool_names
    for effectful_psykyo in (
        "ms4.hivemind.psykyo.benchmark.run@v1",
        "ms4.hivemind.psykyo.vlm_consensus@v1",
    ):
        assert effectful_psykyo in manifest["safety"]["effectful_tools_in_v1"]


def test_every_runtime_action_declares_complete_validated_policy():
    manifest = json.loads((MCP / "manifest.json").read_text(encoding="utf-8"))
    registry = build_tool_registry()
    by_name = validate_manifest(
        manifest,
        registry_names=registry,
        registry_runtime_actions=(
            name for name, tool in registry.items() if tool.is_runtime_action
        ),
    )
    runtime_names = {
        name for name, tool in by_name.items() if tool.get("kind") == "runtime_action"
    }

    assert len(runtime_names) == 31
    assert runtime_names == set(manifest["safety"]["effectful_tools_in_v1"])
    assert by_name["ms4.hermes.update@v1"]["idempotency"]["mode"] == "idempotent"
    assert by_name["ms4.double_agent.cancel@v1"]["idempotency"]["mode"] == "idempotent"
    assert by_name["ms4.hivemind.game_session.cancel@v1"]["idempotency"]["mode"] == "none"


@pytest.mark.parametrize("tool_name", RUNTIME_ACTION_NAMES)
@pytest.mark.parametrize("field", ("gated_by", "idempotency", "cancellation", "audit"))
def test_manifest_validator_rejects_each_missing_runtime_policy_field(tool_name, field):
    manifest = json.loads((MCP / "manifest.json").read_text(encoding="utf-8"))
    action = next(tool for tool in manifest["tools"] if tool["name"] == tool_name)
    del action[field]

    with pytest.raises(ManifestPolicyError, match=rf"{tool_name}: missing {field}"):
        validate_manifest(manifest)


def test_manifest_validator_rejects_unbounded_or_unjustified_none_gate():
    manifest = json.loads((MCP / "manifest.json").read_text(encoding="utf-8"))
    action = next(tool for tool in manifest["tools"] if tool.get("gated_by") == ["none"])
    action["gate_rationale"] = ""
    action.pop("bounds", None)

    with pytest.raises(ManifestPolicyError) as exc_info:
        validate_manifest(manifest)

    message = str(exc_info.value)
    assert "gate_rationale" in message
    assert "bounds.max_arguments_bytes" in message


def test_manifest_validator_rejects_unenforceable_prose_gate():
    manifest = json.loads((MCP / "manifest.json").read_text(encoding="utf-8"))
    action = next(tool for tool in manifest["tools"] if tool.get("kind") == "runtime_action")
    action["gated_by"] = ["MS3 ethics future"]

    with pytest.raises(ManifestPolicyError, match="gated_by"):
        validate_manifest(manifest)


def test_manifest_validator_rejects_gate_policy_drift():
    manifest = json.loads((MCP / "manifest.json").read_text(encoding="utf-8"))
    action = next(
        tool for tool in manifest["tools"] if tool["name"] == "ms4.double_agent.submit@v1"
    )
    action["gated_by"] = ["confirm"]

    with pytest.raises(ManifestPolicyError, match="gated_by must be"):
        validate_manifest(manifest)


def test_manifest_validator_rejects_idempotency_policy_drift():
    manifest = json.loads((MCP / "manifest.json").read_text(encoding="utf-8"))
    action = next(
        tool for tool in manifest["tools"] if tool["name"] == "ms4.hermes.update@v1"
    )
    action["idempotency"]["mode"] = "none"

    with pytest.raises(ManifestPolicyError, match="idempotency.mode must be idempotent"):
        validate_manifest(manifest)


def test_manifest_validator_rejects_unimplemented_cancellation_mode():
    manifest = json.loads((MCP / "manifest.json").read_text(encoding="utf-8"))
    action = next(tool for tool in manifest["tools"] if tool.get("kind") == "runtime_action")
    action["cancellation"] = {
        "mode": "tool",
        "tool": "ms4.double_agent.cancel@v1",
        "identifier": "job_id",
        "rationale": "Not implemented by dispatch.",
    }

    with pytest.raises(ManifestPolicyError, match="cancellation.mode"):
        validate_manifest(manifest)


def test_offline_manifest_validator_script_passes_canonical_and_rejects_mutant(tmp_path):
    script = ROOT / "machine_spirit_4" / "scripts" / "validate_ms4_mcp_manifest.py"
    canonical = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert canonical.returncode == 0, canonical.stderr
    assert json.loads(canonical.stdout)["runtime_actions"] == 31

    manifest = json.loads((MCP / "manifest.json").read_text(encoding="utf-8"))
    action = next(tool for tool in manifest["tools"] if tool.get("kind") == "runtime_action")
    del action["audit"]
    mutant = tmp_path / "manifest.json"
    with mutant.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle)
    rejected = subprocess.run(
        [sys.executable, str(script), "--manifest", str(mutant)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert rejected.returncode == 1
    assert f"{action['name']}: missing audit" in rejected.stderr

    manifest = json.loads((MCP / "manifest.json").read_text(encoding="utf-8"))
    action = next(tool for tool in manifest["tools"] if tool["name"] == "ms4.chat.send@v1")
    action["kind"] = "read_only"
    manifest["safety"]["effectful_tools_in_v1"].remove(action["name"])
    with mutant.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle)
    rejected = subprocess.run(
        [sys.executable, str(script), "--manifest", str(mutant)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert rejected.returncode == 1
    assert f"native runtime actions missing policy: ['{action['name']}']" in rejected.stderr


def test_confirm_gated_manifest_tools_keep_handler_defense_in_depth():
    """Central dispatch gates these tools; their local guards remain a backstop.

    Scoped to the two tools that share the ToolInputError('Pass confirm:true')
    contract (jobs.cancel, logos.optimize); the gpu.passthrough.* handlers
    enforce the same gate with a different message/ordering.
    """
    from machine_spirit_4.mcp import tools as tools_module

    manifest = json.loads((MCP / "manifest.json").read_text(encoding="utf-8"))
    by_name = {tool["name"]: tool for tool in manifest["tools"]}
    registry = tools_module.build_tool_registry()

    class _Runtime:
        hivemind_url = "http://127.0.0.1:9"
        ms3_url = "http://127.0.0.1:9"

    for name, args in (
        ("ms4.hivemind.logos.optimize@v1", {"prompt_id": "x"}),
        ("ms4.hivemind.jobs.cancel@v1", {"job_id": "x"}),
    ):
        assert "confirm" in (by_name[name].get("gated_by") or []), name
        handler = registry[name].handler
        try:
            handler(_Runtime(), dict(args))
        except tools_module.ToolInputError as exc:
            assert "Pass confirm:true" in str(exc), f"{name}: {exc}"
        else:
            raise AssertionError(f"{name} ran without confirm:true")


def test_client_config_examples_include_cursor_and_jsonrpc_shapes():
    examples = json.loads((MCP / "client_config_examples.json").read_text(encoding="utf-8"))

    assert examples["cursor_mcp"]["mcpServers"]["ms4"]["url"] == "http://127.0.0.1:9181/mcp"
    assert examples["generic_jsonrpc"]["initialize"]["method"] == "initialize"
    assert examples["generic_jsonrpc"]["tools_list"]["method"] == "tools/list"
