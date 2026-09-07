from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import jsonschema


ROOT = Path(__file__).resolve().parents[2]
MS4 = ROOT / "machine_spirit_4"


def _load_chat_voice_module():
    path = MS4 / "scripts" / "validate_ms4_chat_voice.py"
    spec = importlib.util.spec_from_file_location("validate_ms4_chat_voice", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_chat_voice_validator_checks_mcp_jsonrpc(monkeypatch):
    module = _load_chat_voice_module()
    calls: list[dict] = []

    class Response:
        status_code = 200

        def __init__(self, payload):
            self._payload = payload
            self.text = json.dumps(payload)

        def json(self):
            return self._payload

    def fake_post(url, json, timeout):
        calls.append(json)
        if json["method"] == "initialize":
            return Response({"result": {"protocolVersion": "2025-11-25"}})
        if json["method"] == "tools/list":
            return Response({"result": {"tools": [{"name": "hivemind.cluster.summary@v1"}]}})
        raise AssertionError(json)

    monkeypatch.setattr(module.requests, "post", fake_post)

    result = module.check_hivemind_mcp_jsonrpc()

    assert result.ok
    assert [call["method"] for call in calls] == ["initialize", "tools/list"]


def test_chat_voice_validator_accepts_ms3_interact_contract(monkeypatch):
    module = _load_chat_voice_module()

    class Response:
        status_code = 200

        def __init__(self, payload):
            self._payload = payload
            self.text = json.dumps(payload)

        def json(self):
            return self._payload

    def fake_post(url, json, timeout):
        assert url.endswith("/interact")
        assert json["personality_id"] == "sister"
        return Response({
            "text": "Machine Spirit 3 is ready.",
            "processing_time_ms": 42,
            "model_id_used": "qwen2.5:0.5b",
            "model_used": "Small",
        })

    monkeypatch.setattr(module.requests, "post", fake_post)

    result = module.check_ms3_interact()

    assert result.ok
    assert result.name == "ms3_interact_chat"


def test_wait_runner_and_text_demo_scripts_exist_with_expected_contracts():
    wait_script = (MS4 / "scripts" / "wait_for_hivemind_voice.py").read_text(encoding="utf-8")
    demo_script = (MS4 / "scripts" / "run_ms4_text_demo.py").read_text(encoding="utf-8")

    for token in ("/provision/status/ASR", "/provision/status/TTS", "/provision/status/TTS_SUPER"):
        assert token in wait_script
    assert "validate_ms4_runtime.py" in wait_script
    assert "validate_ms4_chat_voice.py" in (MS4 / "scripts" / "validate_ms4_runtime.py").read_text(encoding="utf-8")

    assert "plugins.ms4_consciousness" in demo_script
    assert "/interact" in demo_script
    assert "on_pre_tool_call" in demo_script


def test_nibbles_seed_and_action_intents_validate_against_schemas():
    schema_dir = ROOT / "schemas" / "v1"
    identity_schema = json.loads((schema_dir / "IdentityAnchor.schema.json").read_text(encoding="utf-8"))
    action_schema = json.loads((schema_dir / "ActionIntent.schema.json").read_text(encoding="utf-8"))

    seed = json.loads((MS4 / "profiles" / "nibbles" / "identity_anchor.seed.json").read_text(encoding="utf-8"))
    scare = json.loads((MS4 / "profiles" / "nibbles" / "dry_run" / "scare_intent.example.json").read_text(encoding="utf-8"))
    physical = json.loads((MS4 / "profiles" / "nibbles" / "dry_run" / "physical_intent.example.json").read_text(encoding="utf-8"))

    jsonschema.validate(seed, identity_schema)
    jsonschema.validate(scare, action_schema)
    jsonschema.validate(physical, action_schema)

    assert seed["spirit_id"] == "nibbles"
    assert any("door opens from the inside" in value.lower() for value in seed["core_values_summary"])
    assert scare["requires_safety_clearance"] is True
    assert physical["requires_safety_clearance"] is True
    assert scare["payload"]["dry_run_only"] is True
    assert physical["payload"]["dry_run_only"] is True
