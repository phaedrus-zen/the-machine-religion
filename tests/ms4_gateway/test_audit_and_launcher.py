from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from machine_spirit_4.gateway.audit import append_event, read_events


ROOT = Path(__file__).resolve().parents[2]


def test_audit_writer_appends_jsonl_events(tmp_path, _isolate_gateway_audit_log: Path):
    audit_path = tmp_path / "audit.jsonl"

    append_event("chat", {"session_id": "s1", "model": "m"}, audit_path=audit_path)
    append_event("tool", {"tool": "terminal"}, audit_path=audit_path)
    events = read_events(limit=10, audit_path=audit_path)

    assert [event["event_type"] for event in events] == ["chat", "tool"]
    assert events[0]["session_id"] == "s1"
    assert events[1]["tool"] == "terminal"
    assert not _isolate_gateway_audit_log.exists()


def test_audit_defaults_are_isolated_from_repository_log(_isolate_gateway_audit_log: Path):
    repository_audit_path = ROOT / "machine_spirit_4" / "logs" / "ms4_audit.jsonl"
    marker = uuid4().hex

    append_event("pytest_audit_isolation", {"marker": marker})
    events = read_events(limit=10)

    assert _isolate_gateway_audit_log != repository_audit_path
    assert _isolate_gateway_audit_log.exists()
    assert json.loads(_isolate_gateway_audit_log.read_text(encoding="utf-8"))["marker"] == marker
    assert events[-1]["event_type"] == "pytest_audit_isolation"
    assert events[-1]["marker"] == marker
    repository_bytes = repository_audit_path.read_bytes() if repository_audit_path.exists() else b""
    assert marker.encode("ascii") not in repository_bytes


def test_gateway_exposes_audit_endpoint_asset():
    server = (ROOT / "machine_spirit_4" / "gateway" / "server.py").read_text(encoding="utf-8")

    assert '"/audit"' in server
    assert "read_events" in server


def test_voice_recent_turns_filters_audit_event_type_asset():
    server = (ROOT / "machine_spirit_4" / "gateway" / "server.py").read_text(encoding="utf-8")
    recent_turns = server.split("def _voice_recent_turns_get", 1)[1].split("    # ------------------------------------------------------------------", 1)[0]

    assert "read_recent_voice_turns" in recent_turns
    assert 'e.get("event") in ("voice_turn_complete", "voice_turn_failed")' not in recent_turns


def test_single_launcher_asset_starts_all_services_and_validates():
    launcher = (ROOT / "machine_spirit_4" / "scripts" / "start_ms4.py").read_text(encoding="utf-8")

    assert "ms3_binary" in launcher
    assert '"machine_spirit_4.gateway.server"' in launcher
    assert '"machine_spirit_4.mcp.server"' in launcher
    assert "run_ms4_gateway.py" not in launcher
    assert "run_ms4_mcp.py" not in launcher
    assert "validate_ms4_fusion.py" in launcher
    assert "validate_ms4_mcp.py" in launcher
