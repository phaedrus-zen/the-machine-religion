from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_server_exposes_required_routes():
    server = (ROOT / "machine_spirit_4" / "gateway" / "server.py").read_text(encoding="utf-8")

    for route in (
        '"/health"',
        '"/healthcheck/basic"',
        '"/api/v1/ms4_gateway/healthcheck/basic"',
        '"/api/v1/ms4_gateway/status"',
        '"/chat"',
        '"/chat/stream"',
        '"/sessions"',
        '"/deps/status"',
        '"/hermes/tools"',
        '"/hermes/tool"',
        '"/vision/analyze-local"',
        '"/models"',
        '"/voice/status"',
    ):
        assert route in server
    assert "Ms4HermesRunner" in server
    assert "service_info" in server


def test_ms4_web_entrypoint_uses_gateway_routes():
    html = (ROOT / "machine_spirit_4" / "web" / "index.html").read_text(encoding="utf-8")

    assert 'id="chatForm"' in html
    assert "/chat" in html
    assert "/chat/stream" in html
    assert "streamToggle" in html
    assert "/models" in html
    assert "/voice/status" in html
    assert "Hermes tool trace" in html
    assert "Hermes" in html
    assert "64K context" in html
    assert "return 'done'" in html
    assert "if (terminal) return" in html


def test_fusion_validator_reads_sse_incrementally():
    validator = (ROOT / "machine_spirit_4" / "scripts" / "validate_ms4_fusion.py").read_text(encoding="utf-8")
    post_sse = validator.split("def post_sse", 1)[1].split("def main", 1)[0]

    assert ".readline()" in post_sse
    assert "response.read().decode" not in post_sse


def test_local_vision_bridge_prefers_hivemind_mcp_vlm_tool():
    vision = (ROOT / "machine_spirit_4" / "gateway" / "vision.py").read_text(encoding="utf-8")

    assert "hivemind.vlm.describe_image@v1" in vision
    assert "/v1/chat/completions" in vision
    assert "analysis_source" in vision
    assert "empty_visible_text" in vision
    assert "llama3.2-vision:11b-instruct-q4_K_M" in vision
