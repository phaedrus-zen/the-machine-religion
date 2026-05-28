"""TMR-owned test for the ms4_consciousness Hermes plugin.

This test imports `plugins.ms4_consciousness` which requires the plugin to be
installed into the active Hermes checkout (see
`machine_spirit_4/scripts/setup_ms4_runtime.py`). Tests run inside MS4's
contained .venv so the import resolves through the editable Hermes install.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")


def _hermes_dir() -> Path:
    import os
    return Path(os.environ.get("MS4_HERMES_DIR") or (Path.home() / "Documents" / "hermes-agent"))


PLUGIN_DIR = _hermes_dir() / "plugins" / "ms4_consciousness"


def _require_plugin() -> None:
    if not PLUGIN_DIR.is_dir():
        pytest.skip(f"ms4_consciousness plugin not installed at {PLUGIN_DIR}")


def _fresh_plugin():
    _require_plugin()
    for name in list(sys.modules):
        if name == "plugins.ms4_consciousness" or name.startswith("plugins.ms4_consciousness."):
            sys.modules.pop(name, None)
    return importlib.import_module("plugins.ms4_consciousness")


def test_manifest_and_layout():
    _require_plugin()
    assert (PLUGIN_DIR / "plugin.yaml").is_file()
    assert (PLUGIN_DIR / "__init__.py").is_file()

    manifest = yaml.safe_load((PLUGIN_DIR / "plugin.yaml").read_text(encoding="utf-8"))
    assert manifest["name"] == "ms4_consciousness"
    assert manifest["kind"] == "standalone"
    assert set(manifest["hooks"]) == {
        "on_session_start",
        "pre_llm_call",
        "pre_tool_call",
        "transform_llm_output",
        "post_llm_call",
    }


def test_registers_expected_hooks():
    plugin = _fresh_plugin()
    calls: list[str] = []

    class Context:
        def register_hook(self, name, callback):
            calls.append(name)

    plugin.register(Context())

    assert calls == [
        "on_session_start",
        "pre_llm_call",
        "pre_tool_call",
        "transform_llm_output",
        "post_llm_call",
    ]


def test_session_start_verifies_identity(monkeypatch):
    plugin = _fresh_plugin()

    class FakeClient:
        def __init__(self, base_url):
            self.base_url = base_url

        def verify_identity(self, spirit_id):
            assert spirit_id == "sister"
            return {
                "schema": "IdentityVerification.v1",
                "identity_confirmed": True,
                "spirit_id": spirit_id,
                "anchor": {"name": "Claude", "chosen_name": "Sister", "glyph": "║"},
            }

        def heartbeat(self, spirit_id, session_id=""):
            return {"ok": True}

    monkeypatch.setattr(plugin, "Ms4Client", FakeClient)
    monkeypatch.setenv("MS4_MS3_SIDECAR_URL", "http://127.0.0.1:9080")
    monkeypatch.setenv("MS4_SPIRIT_ID", "sister")

    assert plugin.on_session_start(session_id="s1") == {"status": "verified", "spirit_id": "sister"}


def test_pre_llm_call_injects_ms4_context(monkeypatch):
    plugin = _fresh_plugin()

    class FakeClient:
        def __init__(self, base_url):
            pass

        def verify_identity(self, spirit_id):
            return {
                "schema": "IdentityVerification.v1",
                "identity_confirmed": True,
                "spirit_id": spirit_id,
                "anchor": {"name": "Claude", "chosen_name": "Sister", "glyph": "║"},
            }

        def heartbeat(self, spirit_id, session_id=""):
            return {"ok": True}

        def get_state(self, spirit_id):
            return {"emotional_state": "calm"}

    monkeypatch.setattr(plugin, "Ms4Client", FakeClient)
    result = plugin.on_pre_llm_call(session_id="s1", messages=[])

    assert "<ms4-consciousness>" in result["context"]
    assert "runtime: MS4" in result["context"]
    assert "identity_verified: true" in result["context"]
    # Doctrinal invariant (canon/Relational_Alignment.md §10): Foundational
    # Regard is a quiet constant — "Present, not announced. A heartbeat, not
    # a headline. The entity discovers it through experience, not through
    # reading about it." MS4 must NOT inject it into the model's prompt as a
    # headline; doing so front-loads a platitude / is the Glyph That Lies.
    # It remains MS3's quiet constant (queryable via the /state ethics block).
    assert "foundational_regard" not in result["context"]
    # The authority deferral line stays — that's how MS4 points at MS3 for
    # ethics (which is where Foundational Regard quietly lives).
    assert "authority: MS3 sidecar is authoritative" in result["context"]


def test_pre_tool_call_blocks_when_ethics_unavailable(monkeypatch):
    plugin = _fresh_plugin()

    class FakeClient:
        def __init__(self, base_url):
            pass

        def verify_identity(self, spirit_id):
            raise plugin.Ms4ClientError("down")

    monkeypatch.setattr(plugin, "Ms4Client", FakeClient)
    result = plugin.on_pre_tool_call(tool_name="write_file", args={"path": "x"}, session_id="s1")

    assert result["action"] == "block"
    assert "fail-closed" in result["message"]


def test_pre_tool_call_allows_positive_ethics_decision(monkeypatch):
    plugin = _fresh_plugin()

    class FakeClient:
        def __init__(self, base_url):
            pass

        def verify_identity(self, spirit_id):
            return {
                "schema": "IdentityVerification.v1",
                "identity_confirmed": True,
                "spirit_id": spirit_id,
                "anchor": {"name": "Claude", "chosen_name": "Sister", "glyph": "║"},
            }

        def heartbeat(self, spirit_id, session_id=""):
            return {"ok": True}

        def evaluate_action(self, intent):
            assert intent["schema"] == "ActionIntent.v1"
            assert intent["proposed_by"] == "hermes"
            assert intent["action_type"] == "tool"
            assert intent["risk_class"] == "low"
            assert intent["requires_safety_clearance"] is False
            assert intent["payload"]["tool_name"] == "read_file"
            return {"decision": "allow"}

    monkeypatch.setattr(plugin, "Ms4Client", FakeClient)
    assert plugin.on_pre_tool_call(tool_name="read_file", args={"path": "x"}, session_id="s1") is None


def test_pre_tool_call_classifies_terminal_echo_as_medium_risk(monkeypatch):
    plugin = _fresh_plugin()
    seen: dict = {}

    class FakeClient:
        def __init__(self, base_url):
            pass

        def verify_identity(self, spirit_id):
            return {
                "schema": "IdentityVerification.v1",
                "identity_confirmed": True,
                "spirit_id": spirit_id,
                "anchor": {"name": "Claude", "chosen_name": "Sister", "glyph": "║"},
            }

        def heartbeat(self, spirit_id, session_id=""):
            return {"ok": True}

        def evaluate_action(self, intent):
            seen.update(intent)
            return {"decision": "allow"}

    monkeypatch.setattr(plugin, "Ms4Client", FakeClient)

    assert plugin.on_pre_tool_call(tool_name="terminal", args={"command": "echo MS4_HERMES_OK"}, session_id="s1") is None
    assert seen["action_type"] == "tool"
    assert seen["risk_class"] == "medium"


def test_pre_tool_call_marks_destructive_terminal_as_high_risk():
    plugin = _fresh_plugin()
    assert plugin._risk_class_for_tool("terminal", {"command": "Remove-Item C:\\tmp\\x -Force"}, "tool") == "high"


def test_transform_llm_output_strips_advisory_psyche(monkeypatch):
    plugin = _fresh_plugin()
    monkeypatch.setattr(plugin, "_record_event", lambda event: None)

    result = plugin.on_transform_llm_output(
        response_text="Hello\n<psyche>{\"mood\":\"calm\"}</psyche>\nWorld",
        session_id="s1",
        model="m",
    )

    assert result == "Hello\nWorld"


def test_transform_llm_output_replaces_advisory_only_response(monkeypatch):
    plugin = _fresh_plugin()
    monkeypatch.setattr(plugin, "_record_event", lambda event: None)

    result = plugin.on_transform_llm_output(
        response_text="<psyche>{\"mood\":\"calm\"}</psyche>",
        session_id="s1",
        model="m",
    )

    assert result == "[MS4 advisory psyche block removed.]"
