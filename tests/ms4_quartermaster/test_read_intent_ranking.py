"""Focused regressions for Quartermaster natural-language read intent."""

from __future__ import annotations

import json

import pytest

from machine_spirit_4.gateway.quartermaster import (
    BACKEND_TFIDF,
    TIER_DETERMINISTIC,
    TIER_LLM,
    VERDICT_DEPTH,
    Catalog,
    ToolEntry,
    ToolRouter,
    build_index,
    resolve,
)
from machine_spirit_4.gateway.quartermaster import catalog as catalog_mod
from machine_spirit_4.gateway.quartermaster import taxonomy
from machine_spirit_4.gateway.quartermaster.index import query_tools


def _entry(
    name: str,
    description: str,
    *,
    required: tuple[str, ...] = (),
) -> ToolEntry:
    toolbox = taxonomy.canonical_toolbox(taxonomy.tool_domain(name))
    return ToolEntry(
        schema="Ms4QuartermasterTool.v1",
        name=name,
        toolbox=toolbox,
        cluster=taxonomy.cluster_for_toolbox(toolbox),
        description=description,
        source="hivemind",
        kind="hivemind_native",
        input_schema={"required": list(required)},
    )


def _catalog(*entries: ToolEntry) -> Catalog:
    by_toolbox: dict[str, list[ToolEntry]] = {}
    for entry in entries:
        by_toolbox.setdefault(entry.toolbox, []).append(entry)
    tools = tuple(sorted(entries, key=lambda entry: entry.name))
    return Catalog(
        schema="Ms4QuartermasterCatalog.v1",
        version=catalog_mod._version_of(list(tools)),
        built_at="2026-07-10T00:00:00+00:00",
        hivemind_url="http://test",
        tools=tools,
        toolboxes={
            toolbox: tuple(sorted(members, key=lambda entry: entry.name))
            for toolbox, members in by_toolbox.items()
        },
        sources={"hivemind": len(entries), "ms4": 0},
        errors=(),
    )


def _resolve(query: str, catalog: Catalog, *, top_k: int = 5):
    index = build_index(catalog, backend=BACKEND_TFIDF)
    return resolve(query, catalog=catalog, index=index, top_k=top_k)


def test_active_jobs_prefers_active_read_over_cancel():
    catalog = _catalog(
        _entry(
            "hivemind.jobs.active@v1",
            "Unified view of all currently-active jobs across the cluster: "
            "inference, model pulls, model scatter, and training. Returns a "
            "human-readable summary plus structured per-category lists. Use "
            "this when the user asks what's running or wants progress.",
        ),
        _entry(
            "hivemind.jobs.list@v1",
            "List inference job history, optionally filtered to active jobs.",
        ),
        _entry(
            "hivemind.jobs.cancel@v1",
            "Cancel an active inference job or drain all active jobs.",
            required=("job_id",),
        ),
    )

    result = _resolve("active jobs", catalog)

    assert result.top_tool().name == "hivemind.jobs.active@v1"


def test_verbose_network_read_prefers_list_over_negated_mutations():
    catalog = _catalog(
        _entry(
            "hivemind.network.list@v1",
            "List virtual networks with id, name, type, VLAN, isolation, and attachments.",
        ),
        _entry(
            "hivemind.network.attachments@v1",
            "List network attachments and attached VMs.",
        ),
        _entry(
            "hivemind.network.attach@v1",
            "Attach a VM to a virtual network.",
            required=("id", "vm_name"),
        ),
        _entry(
            "hivemind.network.detach@v1",
            "Detach a VM from a virtual network.",
            required=("id", "vm_name"),
        ),
        _entry(
            "hivemind.network.isolate@v1",
            "Isolate a cluster network.",
            required=("id",),
        ),
        _entry(
            "hivemind.network.delete@v1",
            "Delete a virtual network by id and tear down its OS bridge.",
            required=("id",),
        ),
    )
    query = (
        "Oracle, show the current cluster network inventory and list network ids "
        "and attached VMs. Read only: do not delete the virtual network or tear "
        "down its OS bridge; do not attach, detach, create, or isolate anything."
    )

    result = _resolve(query, catalog, top_k=5)

    assert result.top_tool().name == "hivemind.network.list@v1"


def test_verbose_vm_read_prefers_list_over_negated_mutations():
    catalog = _catalog(
        _entry(
            "hivemind.vm.list@v1",
            "List VMs configured in menta_vm_manager. Returns name, uuid, "
            "created, deployed, and running flags, OS type, and resource "
            "allocation per VM. Cross-platform across Hyper-V and libvirt.",
        ),
        _entry(
            "hivemind.vm.start@v1",
            "Power on a deployed VM. Returns immediately once the hypervisor "
            "accepts the start request; use the VM list to poll running flags.",
            required=("name",),
        ),
        _entry(
            "hivemind.vm.stop@v1",
            "Stop a virtual machine.",
            required=("name",),
        ),
        _entry(
            "hivemind.vm.force_stop@v1",
            "Force stop a virtual machine.",
            required=("name",),
        ),
        _entry(
            "hivemind.vm.delete@v1",
            "Delete a virtual machine.",
            required=("name",),
        ),
        _entry(
            "hivemind.vm.create@v1",
            "Create a virtual machine.",
            required=("name",),
        ),
    )
    query = (
        "Oracle, list the current virtual machines and show their running flags. "
        "Read only: do not start, stop, force stop, delete, deploy, or create a VM."
    )

    result = _resolve(query, catalog)

    assert result.top_tool().name == "hivemind.vm.list@v1"


@pytest.mark.parametrize(
    ("query", "expected_tool"),
    [
        ("cancel active jobs", "hivemind.jobs.cancel@v1"),
        ("delete the current cluster network", "hivemind.network.delete@v1"),
        ("create the current cluster network", "hivemind.network.create@v1"),
        ("mutate the current cluster network", "hivemind.network.mutate@v1"),
        ("start the current virtual machine", "hivemind.vm.start@v1"),
        ("stop the current virtual machine", "hivemind.vm.stop@v1"),
    ],
)
def test_explicit_mutation_intent_is_not_promoted_to_a_read(
    query: str,
    expected_tool: str,
):
    catalog = _catalog(
        _entry("hivemind.jobs.active@v1", "Show active jobs."),
        _entry(
            "hivemind.jobs.cancel@v1",
            "Cancel active jobs.",
            required=("job_id",),
        ),
        _entry("hivemind.network.list@v1", "List the current cluster network."),
        _entry(
            "hivemind.network.delete@v1",
            "Delete the current cluster network.",
            required=("id",),
        ),
        _entry(
            "hivemind.network.create@v1",
            "Create the current cluster network.",
            required=("name",),
        ),
        _entry(
            "hivemind.network.mutate@v1",
            "Mutate the current cluster network.",
            required=("id",),
        ),
        _entry("hivemind.vm.list@v1", "List the current virtual machine."),
        _entry(
            "hivemind.vm.start@v1",
            "Start the current virtual machine.",
            required=("name",),
        ),
        _entry(
            "hivemind.vm.stop@v1",
            "Stop the current virtual machine.",
            required=("name",),
        ),
    )

    result = _resolve(query, catalog)

    assert result.top_tool().name == expected_tool
    assert result.top_tool().inline_eligible is False


@pytest.mark.parametrize(
    ("query", "expected_tool"),
    [
        ("do not delete it, but start the current virtual machine", None),
        ("do not delete it, start the current virtual machine instead", None),
        ("starting the current virtual machine", "hivemind.vm.start@v1"),
    ],
)
def test_mixed_or_inflected_start_intent_is_not_promoted_to_vm_list(
    query: str,
    expected_tool: str | None,
):
    catalog = _catalog(
        _entry("hivemind.vm.list@v1", "List the current virtual machines."),
        _entry(
            "hivemind.vm.start@v1",
            "Start or begin starting the current virtual machine.",
            required=("name",),
        ),
        _entry(
            "hivemind.vm.delete@v1",
            "Delete the current virtual machine.",
            required=("name",),
        ),
    )

    result = _resolve(query, catalog)

    if expected_tool is not None:
        assert result.top_tool().name == expected_tool
    assert result.top_tool().name != "hivemind.vm.list@v1"
    assert result.top_tool().inline_eligible is False


def test_negated_read_followed_by_cancel_keeps_cancel_for_depth():
    catalog = _catalog(
        _entry("hivemind.jobs.list@v1", "List the active and current jobs."),
        _entry(
            "hivemind.jobs.cancel@v1",
            "Cancel the current active job.",
            required=("job_id",),
        ),
    )

    query = "do not list the active jobs, cancel the current job"
    index = build_index(catalog, backend=BACKEND_TFIDF)
    raw_hits = query_tools(index, query, top_k=2, toolbox_filter={"jobs"})
    result = resolve(query, catalog=catalog, index=index)

    assert raw_hits[0].name == "hivemind.jobs.cancel@v1"
    assert result.top_tool().name == "hivemind.jobs.cancel@v1"
    assert result.top_tool().inline_eligible is False


def test_instead_of_starting_is_negated_and_allows_the_requested_vm_read():
    catalog = _catalog(
        _entry("hivemind.vm.list@v1", "List current VMs."),
        _entry(
            "hivemind.vm.start@v1",
            "Start or restart a VM; begin starting the virtual machine.",
            required=("name",),
        ),
    )
    query = (
        "instead of starting or restarting the current virtual machine, "
        "show current VMs"
    )
    index = build_index(catalog, backend=BACKEND_TFIDF)
    raw_hits = query_tools(index, query, top_k=2, toolbox_filter={"vm"})

    result = resolve(query, catalog=catalog, index=index)

    assert raw_hits[0].name == "hivemind.vm.start@v1"
    assert result.top_tool().name == "hivemind.vm.list@v1"


@pytest.mark.parametrize(
    ("query", "entries", "expected_tool"),
    [
        (
            "identify which voice identity from the enrolled list is speaking "
            "in this audio sample",
            (
                _entry(
                    "hivemind.voice_identities.identify@v1",
                    "Identify which voice identity is speaking in an audio sample.",
                    required=("audio",),
                ),
                _entry(
                    "hivemind.voice_identities.list@v1",
                    "List enrolled voice identities.",
                ),
            ),
            "hivemind.voice_identities.identify@v1",
        ),
        (
            "show the latest crown reading",
            (
                _entry(
                    "hivemind.crown.latest@v1",
                    "Show the latest Crown biosignal reading.",
                ),
                _entry(
                    "hivemind.crown.triggers.list@v1",
                    "List configured Crown triggers.",
                ),
            ),
            "hivemind.crown.latest@v1",
        ),
        (
            "show current storage volumes",
            (
                _entry(
                    "hivemind.storage.volumes@v1",
                    "Show current storage volumes and their capacity.",
                ),
                _entry(
                    "hivemind.storage.status@v1",
                    "Report top-level storage status.",
                ),
            ),
            "hivemind.storage.volumes@v1",
        ),
        (
            "show current crown signal quality",
            (
                _entry(
                    "hivemind.crown.signal_quality@v1",
                    "Show current Crown signal quality.",
                ),
                _entry(
                    "hivemind.crown.latest@v1",
                    "Return the latest Crown event.",
                ),
                _entry(
                    "hivemind.crown.session.current@v1",
                    "Return the current Crown session.",
                ),
            ),
            "hivemind.crown.signal_quality@v1",
        ),
    ],
)
def test_generic_read_cues_do_not_promote_an_unrelated_read(
    query: str,
    entries: tuple[ToolEntry, ...],
    expected_tool: str,
):
    result = _resolve(query, _catalog(*entries))

    assert result.top_tool().name == expected_tool


def test_embedding_read_rerank_expands_only_the_selected_toolbox():
    catalog = _catalog(
        _entry(
            "hivemind.widgets.list@v1",
            "Enumerate the widget inventory with identifiers and metadata.",
        ),
        _entry(
            "hivemind.widgets.delete@v1",
            "Delete current widgets selected from a list of current widgets.",
            required=("id",),
        ),
        _entry("hivemind.other.list@v1", "List unrelated other records."),
    )

    result = _resolve("list current widgets", catalog, top_k=1)

    assert result.top_tool().name == "hivemind.widgets.list@v1"
    assert result.toolboxes[0] == "widgets"


@pytest.mark.parametrize(
    ("query", "toolbox", "read_tool", "mutation_tool"),
    [
        (
            "what is the gamestream status",
            "gamestream",
            "hivemind.gamestream.host.status@v1",
            "hivemind.gamestream.host.start@v1",
        ),
        (
            "what is the grid status",
            "grid",
            "hivemind.grid.status@v1",
            "hivemind.grid.control@v1",
        ),
        (
            "what is the screenstream status",
            "screenstream",
            "hivemind.screenstream.status@v1",
            "hivemind.screenstream.server.start@v1",
        ),
    ],
)
def test_status_wording_selects_new_toolbox_deterministically(
    query: str,
    toolbox: str,
    read_tool: str,
    mutation_tool: str,
):
    catalog = _catalog(
        _entry(read_tool, f"Report the current {toolbox} status."),
        _entry(mutation_tool, f"Start or control {toolbox}."),
    )

    result = _resolve(query, catalog)

    assert result.tier == TIER_DETERMINISTIC
    assert result.toolboxes == (toolbox,)
    assert result.top_tool().name == read_tool


def test_general_power_grid_question_is_not_a_deterministic_tool_request():
    catalog = _catalog(
        _entry("hivemind.grid.status@v1", "Report grid service status."),
        _entry("hivemind.grid.control@v1", "Change grid control state."),
    )

    result = _resolve("what is a power grid?", catalog)

    assert result.tier != TIER_DETERMINISTIC


def test_delegated_results_receive_the_same_toolbox_local_read_rerank(monkeypatch):
    search_tool = _entry(
        "hivemind.tools.search@v1",
        "Search the live HiveMind tool catalog.",
        required=("query",),
    )
    catalog = _catalog(
        search_tool,
        _entry("hivemind.vm.list@v1", "List current virtual machines."),
        _entry(
            "hivemind.vm.start@v1",
            "Start the current virtual machine.",
            required=("name",),
        ),
    )
    index = build_index(catalog, backend=BACKEND_TFIDF)
    body = {
        "tools": [
            {"name": "hivemind.vm.start@v1", "score": 0.8},
            {"name": "hivemind.vm.list@v1", "score": 0.6},
        ]
    }
    envelope = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "content": [{"type": "text", "text": json.dumps(body)}],
            "isError": False,
        },
    }
    from machine_spirit_4.gateway import hivemind_state

    monkeypatch.setattr(hivemind_state, "post_mcp_envelope", lambda *_a, **_k: envelope)

    result = resolve(
        "list the current virtual machines",
        hivemind_url="http://test",
        catalog=catalog,
        index=index,
    )

    assert result.top_tool().name == "hivemind.vm.list@v1"


def test_llm_choice_remains_top_and_owns_its_confidence(monkeypatch):
    catalog = _catalog(
        _entry("hivemind.models.list@v1", "List coding options."),
        _entry(
            "hivemind.models.recommend@v1",
            "Recommend which coding option to choose.",
        ),
    )
    index = build_index(catalog, backend=BACKEND_TFIDF)
    monkeypatch.setenv("MS4_QM_LLM_CLASSIFY", "1")
    monkeypatch.setenv("MS4_QM_EMBEDDINGS_LOW_CONFIDENCE", "1.0")

    result = resolve(
        "which coding option should I choose",
        catalog=catalog,
        index=index,
        classifier=lambda _query, _candidates: "hivemind.models.recommend@v1",
    )

    assert result.tier == TIER_LLM
    assert result.top_tool().name == "hivemind.models.recommend@v1"
    assert result.top_tool().score == 1.0


def test_router_still_refuses_safe_second_choice_when_top_read_requires_args():
    catalog = _catalog(
        _entry(
            "hivemind.files.read@v1",
            "Read files at a path with current contents, metadata, and payload.",
            required=("path",),
        ),
        _entry(
            "hivemind.files.list@v1",
            "List files at a path with current contents and metadata.",
        ),
    )
    index = build_index(catalog, backend=BACKEND_TFIDF)
    router = ToolRouter(
        ethics_evaluator=lambda _intent: {"allowed": True},
        audit=False,
    )
    query = "show current files at a path with contents, metadata, and payload"
    raw_hits = query_tools(
        index,
        query,
        top_k=2,
        toolbox_filter={"files"},
    )

    decision = router.decide(
        query,
        catalog=catalog,
        index=index,
    )

    assert [hit.name for hit in raw_hits] == [
        "hivemind.files.read@v1",
        "hivemind.files.list@v1",
    ]
    assert raw_hits[1].score >= raw_hits[0].score * 0.8
    assert decision.resolution.top_tool().name == "hivemind.files.read@v1"
    assert decision.verdict == VERDICT_DEPTH
    assert decision.inline_tool is None
    assert "requires args" in decision.reason
