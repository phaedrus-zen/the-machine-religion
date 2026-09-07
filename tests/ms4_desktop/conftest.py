from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_ms4_audit_log(monkeypatch, tmp_path):
    monkeypatch.setenv("MS4_AUDIT_LOG", str(tmp_path / "ms4_audit.jsonl"))
