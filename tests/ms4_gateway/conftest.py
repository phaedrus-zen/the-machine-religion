"""Isolation for ms4_gateway tests.

The gateway holds a few process-global caches: the grounding cache
(``machine_spirit_4.gateway.context``), the foreground model picker
cache (``machine_spirit_4.double_agent.model_picker``), etc. Without
explicit cleanup, tests that touch ``build_grounded_user_message`` or
``choose_foreground_model`` leak state into the next test in the file
or in a sibling file — observed live as ``test_tools_question_routes_
through_tools_grounding`` failing only when run after the grounding
cache suite because the suite leaves the tools entry warm.

Clear those caches autouse before every test so behavior is hermetic.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_gateway_audit_log(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Path:
    audit_path = tmp_path / "ms4_audit.jsonl"
    monkeypatch.setenv("MS4_AUDIT_LOG", str(audit_path))
    return audit_path


def _clear_voice_turn_last_good(audit_module) -> None:
    with audit_module._LAST_GOOD_LOCK:
        if hasattr(audit_module, "_LAST_GOOD_VOICE_TURNS"):
            audit_module._LAST_GOOD_VOICE_TURNS.clear()
        scoped = getattr(audit_module, "_LAST_GOOD_BY_SCOPE", None)
        if scoped is not None:
            scoped.clear()
    if hasattr(audit_module, "_SCAN_CHUNK_HOOK"):
        audit_module._SCAN_CHUNK_HOOK = None


@pytest.fixture(autouse=True)
def _isolate_voice_turn_last_good():
    from machine_spirit_4.gateway import audit as audit_module

    _clear_voice_turn_last_good(audit_module)
    yield
    _clear_voice_turn_last_good(audit_module)


@pytest.fixture(autouse=True)
def _isolate_gateway_caches(_isolate_gateway_audit_log: Path):
    from machine_spirit_4.gateway import context as ctx_module
    from machine_spirit_4.double_agent import model_picker as picker_module

    ctx_module.clear_grounding_cache()
    if hasattr(picker_module, "_clear_cache_for_tests"):
        picker_module._clear_cache_for_tests()
    yield
    ctx_module.clear_grounding_cache()
    if hasattr(picker_module, "_clear_cache_for_tests"):
        picker_module._clear_cache_for_tests()
