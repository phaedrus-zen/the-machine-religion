import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MCP = ROOT / "machine_spirit_4" / "mcp"


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


def test_every_runtime_action_declares_gate_policy():
    """Gate metadata is the contract the Quartermaster / MCP layer enforce.

    Every kind=runtime_action tool must carry a non-empty gated_by list of
    non-empty strings: either a real gate ('confirm:true required',
    'operator-level', 'safe-id guard', ...) or an explicit 'none: <rationale>'
    declaration. A missing list is an undeclared policy, not 'no gate'.
    """
    manifest = json.loads((MCP / "manifest.json").read_text(encoding="utf-8"))
    undeclared = []
    malformed = []
    for tool in manifest["tools"]:
        if tool.get("kind") != "runtime_action":
            continue
        gated_by = tool.get("gated_by")
        if not gated_by:
            undeclared.append(tool["name"])
            continue
        if not isinstance(gated_by, list) or not all(
            isinstance(gate, str) and gate.strip() for gate in gated_by
        ):
            malformed.append(tool["name"])
    assert undeclared == [], f"runtime_action tools without gated_by: {undeclared}"
    assert malformed == [], f"runtime_action tools with malformed gated_by: {malformed}"


def test_confirm_gated_manifest_tools_enforce_confirm_in_registry():
    """A 'confirm:true required' manifest gate must be enforced by the handler.

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
        assert "confirm:true required" in (by_name[name].get("gated_by") or []), name
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
