from __future__ import annotations

import json
from pathlib import Path

from machine_spirit_4.gateway.audit import append_event, read_events


ROOT = Path(__file__).resolve().parents[2]


def test_audit_writer_appends_jsonl_events(tmp_path):
    audit_path = tmp_path / "audit.jsonl"

    append_event("chat", {"session_id": "s1", "model": "m"}, audit_path=audit_path)
    append_event("tool", {"tool": "terminal"}, audit_path=audit_path)
    events = read_events(limit=10, audit_path=audit_path)

    assert [event["event_type"] for event in events] == ["chat", "tool"]
    assert events[0]["session_id"] == "s1"
    assert events[1]["tool"] == "terminal"


def test_gateway_exposes_audit_endpoint_asset():
    server = (ROOT / "machine_spirit_4" / "gateway" / "server.py").read_text(encoding="utf-8")

    assert '"/audit"' in server
    assert "read_events" in server


def test_single_launcher_asset_starts_all_services_and_validates():
    launcher = (ROOT / "machine_spirit_4" / "scripts" / "start_ms4.py").read_text(encoding="utf-8")

    assert "ms3_binary" in launcher
    assert "run_ms4_gateway.py" in launcher
    assert "run_ms4_mcp.py" in launcher
    assert "validate_ms4_fusion.py" in launcher
    assert "validate_ms4_mcp.py" in launcher
