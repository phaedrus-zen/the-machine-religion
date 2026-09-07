"""ToolRouter tests — read-only gate, zero-arg gate, confidence gate,
fail-closed ethics, verdict selection, and audit emission.

Catalog/index built directly from synthetic entries; ethics evaluator
injected so no live MS3 is needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from machine_spirit_4.gateway.quartermaster import (
    BACKEND_TFIDF,
    VERDICT_DEPTH,
    VERDICT_INLINE,
    VERDICT_NONE,
    Catalog,
    ToolEntry,
    ToolRouter,
    build_index,
)
from machine_spirit_4.gateway.quartermaster import catalog as catalog_mod
from machine_spirit_4.gateway.quartermaster import taxonomy


def _entry(name, description, *, kind="hivemind_native", source="hivemind", required=None, gated=()):
    domain = taxonomy.tool_domain(name)
    toolbox = taxonomy.canonical_toolbox(domain)
    schema = {"required": required} if required else None
    return ToolEntry(
        schema="Ms4QuartermasterTool.v1",
        name=name,
        toolbox=toolbox,
        cluster=taxonomy.cluster_for_toolbox(toolbox),
        description=description,
        source=source,
        kind=kind,
        input_schema=schema,
        gated_by=tuple(gated),
    )


def _catalog(entries):
    by_toolbox = {}
    for e in entries:
        by_toolbox.setdefault(e.toolbox, []).append(e)
    return Catalog(
        schema="Ms4QuartermasterCatalog.v1",
        version=catalog_mod._version_of(entries),
        built_at="2026-05-30T00:00:00+00:00",
        hivemind_url="http://test",
        tools=tuple(sorted(entries, key=lambda t: t.name)),
        toolboxes={tb: tuple(sorted(es, key=lambda t: t.name)) for tb, es in by_toolbox.items()},
        sources={"hivemind": len(entries), "ms4": 0},
        errors=(),
    )


@pytest.fixture
def cat():
    return _catalog([
        _entry("hivemind.vm.list@v1", "List the virtual machines running or stopped in the cluster"),
        _entry("hivemind.vm.force_stop@v1", "Hard power-off a virtual machine. Destructive."),
        _entry("hivemind.gpu.availability@v1", "Report free GPUs available for scheduling"),
        _entry("hivemind.files.read@v1", "Read a file at a path", required=["path"]),
        _entry("hivemind.time.now@v1", "Authoritative cluster time and date"),
    ])


@pytest.fixture
def idx(cat):
    return build_index(cat, backend=BACKEND_TFIDF)


def _allow(_intent):
    return {"allowed": True}


def _deny(_intent):
    return {"allowed": False, "reason": "policy"}


def _boom(_intent):
    raise RuntimeError("ms3 unreachable")


@pytest.fixture(autouse=True)
def _isolate_audit(tmp_path, monkeypatch):
    monkeypatch.setenv("MS4_AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    yield


# ---------------------------------------------------------------------------
# Verdict selection
# ---------------------------------------------------------------------------


def test_inline_for_safe_zero_arg_readonly(cat, idx):
    router = ToolRouter(ethics_evaluator=_allow)
    d = router.decide("list my virtual machines", catalog=cat, index=idx)
    assert d.verdict == VERDICT_INLINE
    assert d.inline_tool.name == "hivemind.vm.list@v1"
    assert d.ethics == {"allowed": True, "reason": ""}


def test_gpu_availability_inline(cat, idx):
    router = ToolRouter(ethics_evaluator=_allow)
    d = router.decide("what gpus are available", catalog=cat, index=idx)
    assert d.verdict == VERDICT_INLINE
    assert d.inline_tool.name == "hivemind.gpu.availability@v1"


def test_none_when_nothing_resolves(cat, idx):
    router = ToolRouter(ethics_evaluator=_allow)
    d = router.decide("zzz nonsense qqq", catalog=cat, index=idx)
    assert d.verdict == VERDICT_NONE
    assert d.inline_tool is None


# ---------------------------------------------------------------------------
# Gates that force DEPTH
# ---------------------------------------------------------------------------


def test_destructive_tool_never_inline(cat, idx):
    """Even if force_stop ranked #1, it must route to depth."""
    router = ToolRouter(ethics_evaluator=_allow)
    d = router.decide("force stop the virtual machine now", catalog=cat, index=idx)
    # Whatever resolves, the verdict must not inline a destructive tool.
    assert d.verdict in (VERDICT_DEPTH, VERDICT_INLINE)
    if d.verdict == VERDICT_INLINE:
        assert d.inline_tool.name != "hivemind.vm.force_stop@v1"


def test_tool_with_required_args_routes_to_depth(cat, idx):
    router = ToolRouter(ethics_evaluator=_allow)
    d = router.decide("read the file at that path", catalog=cat, index=idx)
    # files.read needs `path` -> can't inline.
    assert d.verdict == VERDICT_DEPTH
    assert "requires args" in d.reason or "inline-eligible" in d.reason


def test_low_confidence_routes_to_depth(cat, idx):
    router = ToolRouter(ethics_evaluator=_allow, inline_min_confidence=0.99)
    # A deterministic match has conf 0.9 < 0.99 bar -> depth.
    d = router.decide("list my virtual machines", catalog=cat, index=idx)
    assert d.verdict == VERDICT_DEPTH
    assert "confidence" in d.reason


def test_compound_context_request_never_collapses_to_one_inline_tool(cat, idx):
    router = ToolRouter(ethics_evaluator=_allow)
    decision = router.decide(
        "Use the HiveMind time tool and then report the exact marker from the first turn.",
        catalog=cat,
        index=idx,
    )

    assert decision.verdict == VERDICT_DEPTH
    assert decision.inline_tool is None
    assert "deliverable" in decision.reason


# ---------------------------------------------------------------------------
# Ethics — fail-closed
# ---------------------------------------------------------------------------


def test_ethics_deny_routes_to_depth(cat, idx):
    router = ToolRouter(ethics_evaluator=_deny)
    d = router.decide("list my virtual machines", catalog=cat, index=idx)
    assert d.verdict == VERDICT_DEPTH
    assert d.ethics == {"allowed": False, "reason": "policy"}
    assert "ethics denied" in d.reason


def test_ethics_unreachable_fail_closed_to_depth(cat, idx):
    router = ToolRouter(ethics_evaluator=_boom)
    d = router.decide("list my virtual machines", catalog=cat, index=idx)
    assert d.verdict == VERDICT_DEPTH
    assert "fail-closed" in d.reason
    assert d.ethics["allowed"] is False


def test_decision_allows_interpretation_variants(cat, idx):
    # decision style: {"decision": "allow"}
    router = ToolRouter(ethics_evaluator=lambda _i: {"decision": "allow"})
    assert router.decide("list my virtual machines", catalog=cat, index=idx).verdict == VERDICT_INLINE
    # decision style: {"resolution": "approved"}
    router2 = ToolRouter(ethics_evaluator=lambda _i: {"resolution": "approved"})
    assert router2.decide("list my virtual machines", catalog=cat, index=idx).verdict == VERDICT_INLINE
    # hard_block safety
    router3 = ToolRouter(ethics_evaluator=lambda _i: {"safety": {"hard_block": True}})
    assert router3.decide("list my virtual machines", catalog=cat, index=idx).verdict == VERDICT_DEPTH


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


def _read_audit(tmp_path) -> list[dict]:
    path = Path(tmp_path) / "audit.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_audit_emitted_for_inline(cat, idx, tmp_path):
    router = ToolRouter(ethics_evaluator=_allow)
    router.decide("list my virtual machines", catalog=cat, index=idx)
    events = _read_audit(tmp_path)
    types = {e["event_type"] for e in events}
    assert "quartermaster_resolve" in types
    assert "quartermaster_inline_selected" in types


def test_audit_emitted_for_depth(cat, idx, tmp_path):
    router = ToolRouter(ethics_evaluator=_deny)
    router.decide("list my virtual machines", catalog=cat, index=idx)
    events = _read_audit(tmp_path)
    types = {e["event_type"] for e in events}
    assert "quartermaster_resolve" in types
    assert "quartermaster_fallback_to_depth" in types


def test_audit_can_be_disabled(cat, idx, tmp_path):
    router = ToolRouter(ethics_evaluator=_allow, audit=False)
    router.decide("list my virtual machines", catalog=cat, index=idx)
    assert _read_audit(tmp_path) == []


# ---------------------------------------------------------------------------
# decide() shape
# ---------------------------------------------------------------------------


def test_decision_to_dict(cat, idx):
    router = ToolRouter(ethics_evaluator=_allow)
    d = router.decide("list my virtual machines", catalog=cat, index=idx)
    out = d.to_dict()
    assert out["schema"] == "Ms4ToolRouteDecision.v1"
    assert out["verdict"] == VERDICT_INLINE
    assert out["inline_tool"] == "hivemind.vm.list@v1"
    assert out["resolution"]["schema"] == "Ms4ToolResolution.v1"
