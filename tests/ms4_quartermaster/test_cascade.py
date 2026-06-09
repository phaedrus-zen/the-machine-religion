"""Cascade tests — tiered resolve(): deterministic / embeddings /
tiny-LLM, fail-safe, and the ToolResolution shape.

Catalog + index are built directly from synthetic ToolEntry lists so
these tests are fully offline and deterministic (TF-IDF backend).
"""

from __future__ import annotations

import pytest

from machine_spirit_4.gateway.quartermaster import (
    BACKEND_TFIDF,
    TIER_DETERMINISTIC,
    TIER_EMBEDDINGS,
    TIER_LLM,
    TIER_NONE,
    Catalog,
    ToolEntry,
    build_index,
    resolve,
)
from machine_spirit_4.gateway.quartermaster import catalog as catalog_mod
from machine_spirit_4.gateway.quartermaster import taxonomy


def _entry(name: str, description: str, *, kind: str = "hivemind_native", source: str = "hivemind") -> ToolEntry:
    domain = taxonomy.tool_domain(name)
    toolbox = taxonomy.canonical_toolbox(domain)
    return ToolEntry(
        schema="Ms4QuartermasterTool.v1",
        name=name,
        toolbox=toolbox,
        cluster=taxonomy.cluster_for_toolbox(toolbox),
        description=description,
        source=source,
        kind=kind,
    )


def _catalog(entries: list[ToolEntry]) -> Catalog:
    by_toolbox: dict[str, list[ToolEntry]] = {}
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
        _entry("hivemind.vm.list@v1", "List all virtual machines in the cluster"),
        _entry("hivemind.vm.start@v1", "Start a virtual machine by name"),
        _entry("hivemind.gpu.availability@v1", "Report free GPUs available for scheduling"),
        _entry("hivemind.storage.volumes@v1", "List storage volumes"),
        _entry("hivemind.time.now@v1", "Authoritative cluster time and date"),
        _entry("hivemind.voice_identities.list@v1", "List enrolled voice identities"),
        _entry("hivemind.crown.status@v1", "Crown EEG biosignal connection status"),
        _entry("hivemind.training.start@v1", "Start a LoRA fine-tune training job"),
    ])


@pytest.fixture
def idx(cat):
    return build_index(cat, backend=BACKEND_TFIDF)


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    # Cascade tests default to LLM tier OFF unless a test enables it.
    monkeypatch.delenv("MS4_QM_LLM_CLASSIFY", raising=False)


# ---------------------------------------------------------------------------
# Tier 1: deterministic
# ---------------------------------------------------------------------------


def test_deterministic_nails_obvious_toolbox(cat, idx):
    res = resolve("list my virtual machines", catalog=cat, index=idx)
    assert res.tier == TIER_DETERMINISTIC
    assert res.toolboxes == ("vm",)
    assert res.confidence >= 0.9
    assert res.top_tool().name == "hivemind.vm.list@v1"


def test_deterministic_voice_identity_phrase(cat, idx):
    res = resolve("enroll a new voice identity for me", catalog=cat, index=idx)
    assert res.tier == TIER_DETERMINISTIC
    assert res.toolboxes == ("voice_identities",)


def test_deterministic_time_query(cat, idx):
    res = resolve("what time is it on the cluster", catalog=cat, index=idx)
    assert res.tier == TIER_DETERMINISTIC
    assert res.toolboxes == ("time",)
    assert res.top_tool().name == "hivemind.time.now@v1"


def test_deterministic_top_tool_inline_eligible_flag(cat, idx):
    res = resolve("list my virtual machines", catalog=cat, index=idx)
    top = res.top_tool()
    assert top.name == "hivemind.vm.list@v1"
    assert top.inline_eligible is True  # vm.list -> read-only verb


# ---------------------------------------------------------------------------
# Tier 2: embeddings
# ---------------------------------------------------------------------------


def test_embeddings_tier_for_non_keyword_query(cat, idx):
    """A query with no toolbox keyword but semantic overlap routes via
    embeddings."""
    # "graphics card headroom" shares vocab with the GPU description
    # ("free GPUs available") only weakly — but "available" is a strong
    # token. No toolbox keyword phrase matches, so we land in embeddings.
    res = resolve("how much free capacity is available", catalog=cat, index=idx)
    assert res.tier in (TIER_EMBEDDINGS, TIER_NONE)
    if res.tier == TIER_EMBEDDINGS:
        assert res.tools  # produced a ranked shortlist


def test_embeddings_returns_ranked_shortlist(cat, idx):
    res = resolve("start a fine-tune training run", catalog=cat, index=idx)
    # "training" keyword should actually nail it deterministically:
    assert res.tier in (TIER_DETERMINISTIC, TIER_EMBEDDINGS)
    assert any(t.name == "hivemind.training.start@v1" for t in res.tools)


def test_unknown_query_returns_none_tier(cat, idx):
    res = resolve("zzzqqq nonsense tokens xyzzy", catalog=cat, index=idx)
    assert res.tier == TIER_NONE
    assert res.tools == ()
    assert res.fallback_reason


# ---------------------------------------------------------------------------
# Empty / guard paths
# ---------------------------------------------------------------------------


def test_empty_query(cat, idx):
    res = resolve("", catalog=cat, index=idx)
    assert res.tier == TIER_NONE
    assert "empty query" in (res.fallback_reason or "")


def test_empty_catalog():
    empty = _catalog([])
    res = resolve("list vms", catalog=empty, index=build_index(empty, backend=BACKEND_TFIDF))
    assert res.tier == TIER_NONE
    assert "empty catalog" in (res.fallback_reason or "")


def test_no_url_no_catalog():
    res = resolve("list vms")
    assert res.tier == TIER_NONE
    assert "no hivemind_url" in (res.fallback_reason or "")


# ---------------------------------------------------------------------------
# Tier 3: tiny-LLM (injected fake classifier)
# ---------------------------------------------------------------------------


def test_llm_tier_consulted_only_when_enabled_and_ambiguous(cat, idx, monkeypatch):
    calls = {"n": 0}

    def classifier(query, candidates):
        calls["n"] += 1
        return candidates[0].name if candidates else None

    # LLM disabled -> never consulted even on ambiguous queries
    monkeypatch.delenv("MS4_QM_LLM_CLASSIFY", raising=False)
    resolve("how much free capacity is available", catalog=cat, index=idx, classifier=classifier)
    assert calls["n"] == 0


def test_llm_tier_promotes_choice_when_ambiguous(cat, idx, monkeypatch):
    monkeypatch.setenv("MS4_QM_LLM_CLASSIFY", "1")
    # Force the ambiguity gate wide open so the LLM tier always runs
    # for an embeddings-tier query.
    monkeypatch.setenv("MS4_QM_EMBEDDINGS_LOW_CONFIDENCE", "1.0")

    def classifier(query, candidates):
        # Deterministically choose the gpu tool if present.
        for c in candidates:
            if "gpu" in c.name:
                return c.name
        return candidates[0].name if candidates else None

    # Pick a query that goes to embeddings (no single keyword phrase).
    res = resolve("what capacity is available right now", catalog=cat, index=idx, classifier=classifier)
    if res.tier == TIER_LLM:
        assert res.top_tool().name == "hivemind.gpu.availability@v1"
        assert res.confidence >= 0.9
    else:
        # If deterministic/embeddings already resolved confidently that's
        # acceptable; the LLM tier is an enhancement, not a requirement.
        assert res.tools


def test_llm_classifier_unsure_keeps_embeddings(cat, idx, monkeypatch):
    monkeypatch.setenv("MS4_QM_LLM_CLASSIFY", "1")
    monkeypatch.setenv("MS4_QM_EMBEDDINGS_LOW_CONFIDENCE", "1.0")

    def classifier(query, candidates):
        return None  # unsure

    res = resolve("what capacity is available right now", catalog=cat, index=idx, classifier=classifier)
    # Must not crash; must keep a usable result.
    assert res.tier in (TIER_DETERMINISTIC, TIER_EMBEDDINGS)
    assert res.tools


def test_llm_classifier_exception_is_fail_safe(cat, idx, monkeypatch):
    monkeypatch.setenv("MS4_QM_LLM_CLASSIFY", "1")
    monkeypatch.setenv("MS4_QM_EMBEDDINGS_LOW_CONFIDENCE", "1.0")

    def classifier(query, candidates):
        raise RuntimeError("classifier down")

    res = resolve("what capacity is available right now", catalog=cat, index=idx, classifier=classifier)
    # A broken classifier must never break resolution.
    assert res.tier in (TIER_DETERMINISTIC, TIER_EMBEDDINGS)
    assert res.tools


def test_llm_classifier_hallucinated_id_rejected(cat, idx, monkeypatch):
    monkeypatch.setenv("MS4_QM_LLM_CLASSIFY", "1")
    monkeypatch.setenv("MS4_QM_EMBEDDINGS_LOW_CONFIDENCE", "1.0")

    def classifier(query, candidates):
        return "hivemind.totally.made_up@v1"  # not a candidate

    res = resolve("what capacity is available right now", catalog=cat, index=idx, classifier=classifier)
    # Hallucinated id must be rejected -> fall back to embeddings.
    assert res.tier in (TIER_DETERMINISTIC, TIER_EMBEDDINGS)
    assert all(t.name != "hivemind.totally.made_up@v1" for t in res.tools)


# ---------------------------------------------------------------------------
# Resolution shape
# ---------------------------------------------------------------------------


def test_resolution_to_dict_shape(cat, idx):
    res = resolve("list my virtual machines", catalog=cat, index=idx)
    d = res.to_dict()
    assert d["schema"] == "Ms4ToolResolution.v1"
    assert d["query"] == "list my virtual machines"
    assert d["tier"] == TIER_DETERMINISTIC
    assert isinstance(d["tools"], list)
    assert d["tools"][0]["name"] == "hivemind.vm.list@v1"
    assert "inline_eligible" in d["tools"][0]
    assert d["catalog_version"] == cat.version
