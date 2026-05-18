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
    assert len(tool_names) == 17
    assert "ms4.chat.send@v1" in tool_names
    assert "ms4.nibbles.dry_run@v1" in tool_names
    assert "ms4.hermes.tools.list@v1" in tool_names
    assert "ms4.hermes.tool.call@v1" in tool_names
    assert "ms4.runtime.deps.status@v1" in tool_names
    assert "ms4.vision.analyze_local@v1" in tool_names
    assert "ms4.desktop.status@v1" in tool_names
    assert "ms4.desktop.capture@v1" in tool_names
    assert "ms4.desktop.action@v1" in tool_names


def test_client_config_examples_include_cursor_and_jsonrpc_shapes():
    examples = json.loads((MCP / "client_config_examples.json").read_text(encoding="utf-8"))

    assert examples["cursor_mcp"]["mcpServers"]["ms4"]["url"] == "http://127.0.0.1:9181/mcp"
    assert examples["generic_jsonrpc"]["initialize"]["method"] == "initialize"
    assert examples["generic_jsonrpc"]["tools_list"]["method"] == "tools/list"
