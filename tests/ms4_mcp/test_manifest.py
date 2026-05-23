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
    assert len(tool_names) == 27
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


def test_client_config_examples_include_cursor_and_jsonrpc_shapes():
    examples = json.loads((MCP / "client_config_examples.json").read_text(encoding="utf-8"))

    assert examples["cursor_mcp"]["mcpServers"]["ms4"]["url"] == "http://127.0.0.1:9181/mcp"
    assert examples["generic_jsonrpc"]["initialize"]["method"] == "initialize"
    assert examples["generic_jsonrpc"]["tools_list"]["method"] == "tools/list"
