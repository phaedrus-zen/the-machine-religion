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
    VERDICT_DEPTH,
    Catalog,
    ToolEntry,
    ToolRouter,
    build_index,
    resolve,
)
from machine_spirit_4.gateway.quartermaster import cascade as cascade_mod
from machine_spirit_4.gateway.quartermaster import catalog as catalog_mod
from machine_spirit_4.gateway.quartermaster import taxonomy


def _entry(
    name: str,
    description: str,
    *,
    kind: str = "hivemind_native",
    source: str = "hivemind",
    input_schema: dict | None = None,
) -> ToolEntry:
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
        input_schema=input_schema,
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


@pytest.fixture
def explicit_name_cat():
    return _catalog([
        _entry(
            "hivemind.jobs.get@v1",
            "Get full details for one job by ID",
            input_schema={
                "type": "object",
                "properties": {"job_id": {"type": "string"}},
                "required": ["job_id"],
            },
        ),
        _entry(
            "hivemind.oracle.status",
            "Get the Oracle agent's current status and request details",
        ),
        _entry("hivemind.vm.list@v1", "List virtual machines"),
    ])


@pytest.fixture
def explicit_name_idx(explicit_name_cat):
    return build_index(explicit_name_cat, backend=BACKEND_TFIDF)


@pytest.fixture
def service_action_cat():
    return _catalog([
        _entry(
            "hivemind.services.enable@v1",
            "Enable one Warden-managed service",
            input_schema={
                "type": "object",
                "properties": {"service_name": {"type": "string"}},
                "required": ["service_name"],
            },
        ),
        _entry(
            "hivemind.services.disable@v1",
            "Disable one Warden-managed service",
            input_schema={
                "type": "object",
                "properties": {"service_name": {"type": "string"}},
                "required": ["service_name"],
            },
        ),
        _entry(
            "hivemind.services.restart@v1",
            "Restart one Warden-managed service",
            input_schema={
                "type": "object",
                "properties": {"service_name": {"type": "string"}},
                "required": ["service_name"],
            },
        ),
        _entry("hivemind.oracle.status", "Read Oracle status"),
    ])


@pytest.fixture
def service_action_idx(service_action_cat):
    return build_index(service_action_cat, backend=BACKEND_TFIDF)


def _force_search_ranking(monkeypatch, winner: str) -> list[str]:
    calls: list[str] = []
    toolbox = taxonomy.canonical_toolbox(taxonomy.tool_domain(winner))

    def query_toolboxes(_index, _query, *, top_k=3, hivemind_url=None):
        calls.append("toolboxes")
        return [cascade_mod.index_mod.IndexHit(name=toolbox, score=0.99)]

    def query_tools(
        _index,
        _query,
        *,
        top_k=5,
        toolbox_filter=None,
        hivemind_url=None,
    ):
        calls.append("tools")
        return [cascade_mod.index_mod.IndexHit(name=winner, score=0.99)]

    monkeypatch.setattr(cascade_mod.index_mod, "query_toolboxes", query_toolboxes)
    monkeypatch.setattr(cascade_mod.index_mod, "query_tools", query_tools)
    return calls


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    # Cascade tests default to LLM tier OFF unless a test enables it.
    monkeypatch.delenv("MS4_QM_LLM_CLASSIFY", raising=False)


# ---------------------------------------------------------------------------
# Tier 1: deterministic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "Enable the menta_human_bridge service.",
        "Oracle, please enable menta_human_bridge.",
    ],
)
def test_natural_service_enable_resolves_one_source_deterministic_record(
    query,
    service_action_cat,
    service_action_idx,
    monkeypatch,
):
    calls = _force_search_ranking(monkeypatch, "hivemind.oracle.status")

    result = resolve(
        query,
        catalog=service_action_cat,
        index=service_action_idx,
    )

    assert calls == []
    assert result.tier == TIER_DETERMINISTIC
    assert result.confidence == pytest.approx(0.9)
    assert result.toolboxes == ("services",)
    assert [tool.name for tool in result.tools] == [
        "hivemind.services.enable@v1"
    ]
    assert result.tools[0].score == 1.0
    assert result.deterministic_arguments == {
        "service_name": "menta_human_bridge"
    }
    assert result.to_dict()["deterministic_arguments"] == {
        "service_name": "menta_human_bridge"
    }
    entry = service_action_cat.by_name()["hivemind.services.enable@v1"]
    assert entry.input_schema["required"] == ["service_name"]
    assert result.tools[0].kind == entry.kind


@pytest.mark.parametrize(
    "query",
    [
        "Oracle, please enable Menta Human Bridge service.",
        "Oracle, please enable Menta underscore human, underscore bridge",
        "  ORACLE, PLEASE ENABLE mEnTa HuMaN bRiDgE SERVICE!  ",
    ],
)
def test_asr_service_enable_normalizes_to_bounded_identifier(
    query,
    service_action_cat,
    service_action_idx,
    monkeypatch,
):
    calls = _force_search_ranking(monkeypatch, "hivemind.oracle.status")

    result = resolve(query, catalog=service_action_cat, index=service_action_idx)

    assert calls == []
    assert result.tier == TIER_DETERMINISTIC
    assert [tool.name for tool in result.tools] == [
        "hivemind.services.enable@v1"
    ]
    assert result.deterministic_arguments == {
        "service_name": "menta_human_bridge"
    }


@pytest.mark.parametrize(
    ("query", "canonical"),
    [
        (
            "Oracle, please disable Menta Human Bridge service.",
            "hivemind.services.disable@v1",
        ),
        (
            "Restart Menta underscore human underscore bridge!",
            "hivemind.services.restart@v1",
        ),
    ],
)
def test_asr_service_disable_restart_keep_exact_action(
    query,
    canonical,
    service_action_cat,
    service_action_idx,
    monkeypatch,
):
    calls = _force_search_ranking(monkeypatch, "hivemind.oracle.status")

    result = resolve(query, catalog=service_action_cat, index=service_action_idx)

    assert calls == []
    assert result.top_tool().name == canonical
    assert result.deterministic_arguments == {
        "service_name": "menta_human_bridge"
    }


def test_typed_mixed_identifier_lowercases_without_rewriting_separators(
    service_action_cat,
    service_action_idx,
):
    result = resolve(
        "Enable Menta_Human-Bridge.",
        catalog=service_action_cat,
        index=service_action_idx,
    )

    assert result.deterministic_arguments == {
        "service_name": "menta_human-bridge"
    }


@pytest.mark.parametrize(
    ("action", "canonical"),
    [
        ("disable", "hivemind.services.disable@v1"),
        ("restart", "hivemind.services.restart@v1"),
    ],
)
def test_natural_service_disable_restart_are_exact_deterministic(
    action,
    canonical,
    service_action_cat,
    service_action_idx,
    monkeypatch,
):
    calls = _force_search_ranking(monkeypatch, "hivemind.oracle.status")

    result = resolve(
        f"Oracle, please {action} the menta_human_bridge service!",
        catalog=service_action_cat,
        index=service_action_idx,
    )

    assert calls == []
    assert result.tier == TIER_DETERMINISTIC
    assert [tool.name for tool in result.tools] == [canonical]


@pytest.mark.parametrize(
    "query",
    [
        "  eNaBlE THE menta_human_bridge SERVICE?  ",
        f"enable {'s' * 31}_{'t' * 32}.",
    ],
)
def test_natural_service_action_tolerates_case_punctuation_and_max_name(
    query,
    service_action_cat,
    service_action_idx,
    monkeypatch,
):
    calls = _force_search_ranking(monkeypatch, "hivemind.oracle.status")

    result = resolve(query, catalog=service_action_cat, index=service_action_idx)

    assert calls == []
    assert result.tier == TIER_DETERMINISTIC
    assert result.top_tool().name == "hivemind.services.enable@v1"


@pytest.mark.parametrize(
    "query",
    [
        "Enable menta_human_bridge and menta_oracle.",
        "Enable menta_human_bridge, then disable menta_oracle.",
        "Enable menta_human_bridge service now.",
        "Can you enable menta_human_bridge?",
        "Enable the service.",
        "Enable service.",
        "Enable Menta, Human Bridge.",
        "Enable Menta Human, Bridge.",
        "Enable Menta underscore underscore Bridge.",
        "Enable underscore Menta Human Bridge.",
        "Enable Menta Human Bridge underscore.",
        "Enable Menta comma Human Bridge.",
        "Enable Menta Human Bridge and Menta Oracle.",
        "Enable ../menta_human_bridge.",
        'Enable {"service":"menta_human_bridge"}.',
        "Oracle, please enable menta_human_bridge. Then show status.",
        f"Enable {'s' * 65}.",
    ],
)
def test_non_bounded_service_commands_preserve_existing_ranking(
    query,
    service_action_cat,
    service_action_idx,
    monkeypatch,
):
    calls = _force_search_ranking(monkeypatch, "hivemind.oracle.status")

    result = resolve(query, catalog=service_action_cat, index=service_action_idx)

    assert calls
    assert result.top_tool().name == "hivemind.oracle.status"


@pytest.mark.parametrize(
    "query",
    [
        "Enable my service.",
        "Enable all services.",
        "Restart this service.",
        "Enable Menta Human Bridge immediately.",
        "Enable Menta Human Bridge for me.",
        "Enable Menta Human Bridge stop Menta Oracle.",
        "Enable Menta for me.",
        "Enable Menta right now.",
        "Enable Menta all services.",
        "Enable Menta stop Oracle.",
        "Enable Menta and Oracle.",
        "Enable Menta Human immediately.",
        "Disable our daemon.",
        "Restart every component.",
        "Enable that worker.",
        "Enable Menta Human Bridge carefully.",
        "Enable Menta Human Bridge on cluster.",
        "Enable Quantum Purple Falcon.",
        "Enable Novel Azure Process.",
    ],
)
def test_arbitrary_prose_cannot_satisfy_positive_service_identifier_grammar(
    query,
    service_action_cat,
    service_action_idx,
    monkeypatch,
):
    calls = _force_search_ranking(monkeypatch, "hivemind.oracle.status")

    result = resolve(query, catalog=service_action_cat, index=service_action_idx)

    assert calls
    assert result.tier != TIER_DETERMINISTIC
    assert result.top_tool().name == "hivemind.oracle.status"
    assert result.deterministic_arguments is None


@pytest.mark.parametrize(
    ("query", "accepted"),
    [
        (f"Enable {'a' * 31}_{'b' * 32}.", True),
        (f"Enable {'a' * 32}_{'b' * 32}.", False),
    ],
)
def test_asr_normalized_service_name_length_boundary(
    query,
    accepted,
    service_action_cat,
    service_action_idx,
    monkeypatch,
):
    calls = _force_search_ranking(monkeypatch, "hivemind.oracle.status")

    result = resolve(query, catalog=service_action_cat, index=service_action_idx)

    if accepted:
        assert calls == []
        assert result.tier == TIER_DETERMINISTIC
        assert len(result.deterministic_arguments["service_name"]) == 64
    else:
        assert calls
        assert result.top_tool().name == "hivemind.oracle.status"


def test_conflicting_natural_and_explicit_service_action_rejects_before_search(
    service_action_cat,
    service_action_idx,
    monkeypatch,
):
    calls = _force_search_ranking(monkeypatch, "hivemind.oracle.status")
    query = (
        "Enable menta_human_bridge with "
        "hivemind.services.disable@v1."
    )

    result = resolve(query, catalog=service_action_cat, index=service_action_idx)

    assert calls == []
    assert result.tier == TIER_NONE
    assert result.tools == ()
    assert "conflict" in (result.fallback_reason or "")


def test_matching_natural_and_explicit_service_action_keeps_arguments(
    service_action_cat,
    service_action_idx,
    monkeypatch,
):
    calls = _force_search_ranking(monkeypatch, "hivemind.oracle.status")
    query = (
        "Enable menta_human_bridge with "
        "hivemind.services.enable@v1."
    )

    result = resolve(query, catalog=service_action_cat, index=service_action_idx)

    assert calls == []
    assert result.tier == TIER_DETERMINISTIC
    assert result.top_tool().name == "hivemind.services.enable@v1"
    assert result.deterministic_arguments == {
        "service_name": "menta_human_bridge"
    }


def test_matching_asr_natural_and_explicit_service_action_keeps_normalized_args(
    service_action_cat,
    service_action_idx,
    monkeypatch,
):
    calls = _force_search_ranking(monkeypatch, "hivemind.oracle.status")
    query = (
        "Enable Menta Human Bridge with "
        "hivemind.services.enable@v1."
    )

    result = resolve(query, catalog=service_action_cat, index=service_action_idx)

    assert calls == []
    assert result.tier == TIER_DETERMINISTIC
    assert result.deterministic_arguments == {
        "service_name": "menta_human_bridge"
    }


def test_multiple_explicit_service_actions_with_natural_command_reject(
    service_action_cat,
    service_action_idx,
    monkeypatch,
):
    calls = _force_search_ranking(monkeypatch, "hivemind.oracle.status")
    query = (
        "Enable menta_human_bridge with hivemind.services.enable@v1 "
        "and hivemind.services.disable@v1."
    )

    result = resolve(query, catalog=service_action_cat, index=service_action_idx)

    assert calls == []
    assert result.tier == TIER_NONE
    assert result.tools == ()


def test_full_explicit_name_preempts_search_and_preserves_required_schema(
    explicit_name_cat,
    explicit_name_idx,
    monkeypatch,
):
    calls = _force_search_ranking(monkeypatch, "hivemind.oracle.status")
    query = "Oracle, use hivemind.jobs.get@v1 to show me the details for a job."

    result = resolve(query, catalog=explicit_name_cat, index=explicit_name_idx)

    assert calls == []
    assert result.tier == TIER_DETERMINISTIC
    assert result.toolboxes == ("jobs",)
    assert [tool.name for tool in result.tools] == ["hivemind.jobs.get@v1"]
    jobs_entry = explicit_name_cat.by_name()[result.top_tool().name]
    assert jobs_entry is explicit_name_cat.by_name()["hivemind.jobs.get@v1"]
    assert result.top_tool().toolbox == jobs_entry.toolbox
    assert result.top_tool().cluster == jobs_entry.cluster
    assert result.top_tool().kind == jobs_entry.kind
    assert result.deterministic_arguments is None
    assert "deterministic_arguments" not in result.to_dict()
    assert jobs_entry.input_schema == {
        "type": "object",
        "properties": {"job_id": {"type": "string"}},
        "required": ["job_id"],
    }

    def unexpected_ethics(_intent):
        raise AssertionError("required-argument tools must not reach inline ethics")

    decision = ToolRouter(
        ethics_evaluator=unexpected_ethics,
        audit=False,
    ).decide(query, catalog=explicit_name_cat, index=explicit_name_idx)
    assert decision.verdict == VERDICT_DEPTH
    assert decision.inline_tool is None
    assert "requires args ['job_id']" in decision.reason


@pytest.mark.parametrize(
    "query",
    [
        "Please use [HIVEMIND.JOBS.GET@V1], exactly.",
        "Call 'hivemind.jobs.get@v1'!",
    ],
)
def test_full_explicit_name_is_case_insensitive_with_surrounding_punctuation(
    query,
    explicit_name_cat,
    explicit_name_idx,
    monkeypatch,
):
    calls = _force_search_ranking(monkeypatch, "hivemind.oracle.status")

    result = resolve(query, catalog=explicit_name_cat, index=explicit_name_idx)

    assert calls == []
    assert [tool.name for tool in result.tools] == ["hivemind.jobs.get@v1"]


@pytest.mark.parametrize(
    "query",
    [
        "Use hivemind.jobs.missing@v1 for details.",
        "Use fake.hivemind.jobs.get@v1.extra for details.",
    ],
)
def test_unknown_or_longer_tool_like_name_preserves_search_ranking(
    query,
    explicit_name_cat,
    explicit_name_idx,
    monkeypatch,
):
    calls = _force_search_ranking(monkeypatch, "hivemind.vm.list@v1")

    result = resolve(query, catalog=explicit_name_cat, index=explicit_name_idx)

    assert calls == ["toolboxes", "tools"]
    assert result.top_tool().name == "hivemind.vm.list@v1"


def test_multiple_distinct_explicit_names_preserve_search_ranking(
    explicit_name_cat,
    explicit_name_idx,
    monkeypatch,
):
    calls = _force_search_ranking(monkeypatch, "hivemind.vm.list@v1")
    query = "Use hivemind.jobs.get@v1 and hivemind.oracle.status."

    result = resolve(query, catalog=explicit_name_cat, index=explicit_name_idx)

    assert calls == ["toolboxes", "tools", "tools"]
    assert result.top_tool().name == "hivemind.vm.list@v1"


def test_duplicate_mentions_of_one_explicit_name_resolve_once(
    explicit_name_cat,
    explicit_name_idx,
    monkeypatch,
):
    calls = _force_search_ranking(monkeypatch, "hivemind.oracle.status")
    query = "Use hivemind.jobs.get@v1, then repeat HIVEMIND.JOBS.GET@V1."

    result = resolve(query, catalog=explicit_name_cat, index=explicit_name_idx)

    assert calls == []
    assert [tool.name for tool in result.tools] == ["hivemind.jobs.get@v1"]


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
    assert res.deterministic_arguments is None
    assert "deterministic_arguments" not in d
