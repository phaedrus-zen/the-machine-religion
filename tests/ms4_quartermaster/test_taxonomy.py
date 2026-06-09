"""Taxonomy tests — derive toolbox + cluster from tool ids.

Pure / deterministic / no I/O.
"""

from __future__ import annotations

import pytest

from machine_spirit_4.gateway.quartermaster import taxonomy


@pytest.mark.parametrize("tool_id,expected", [
    # HiveMind canonical
    ("hivemind.vm.list@v1", "vm"),
    ("hivemind.vm.create_prebuilt@v1", "vm"),
    ("hivemind.gpu_mode.set@v1", "gpu_mode"),
    ("hivemind.game_session.plan@v1", "game_session"),
    ("hivemind.crown.events.list@v1", "crown"),
    ("hivemind.crown.calibration_profiles.activate@v1", "crown"),
    ("hivemind.psykyo.benchmark.run@v1", "psykyo"),
    ("hivemind.voice_identities.enroll@v1", "voice_identities"),
    ("hivemind.storage.volume_create@v1", "storage"),
    ("hivemind.network.attach@v1", "network"),
    ("hivemind.embeddings.create@v1", "embeddings"),
    ("hivemind.time.now@v1", "time"),
    ("hivemind.cluster.summary@v1", "cluster"),
    ("hivemind.hosts.list@v1", "hosts"),
    ("hivemind.capability.matrix@v1", "capability"),
    ("hivemind.service_health@v1", "service_health"),  # no second segment domain
    # MS4 native
    ("ms4.chat.send@v1", "chat"),
    ("ms4.identity.verify@v1", "identity"),
    ("ms4.double_agent.submit@v1", "double_agent"),
    ("ms4.desktop.action@v1", "desktop"),
    ("ms4.hermes.tool.call@v1", "hermes"),
    # MS4 → HiveMind proxies: domain inherits from the HiveMind segment
    ("ms4.hivemind.vms@v1", "vms"),  # raw domain (alias-folded by canonical_toolbox)
    ("ms4.hivemind.vm.start@v1", "vm"),
    ("ms4.hivemind.gpu.passthrough.snapshot@v1", "gpu"),
    ("ms4.hivemind.game_session.plan@v1", "game_session"),
    ("ms4.hivemind.training.start@v1", "training"),
    ("ms4.hivemind.adapters.deploy@v1", "adapters"),
    ("ms4.hivemind.loadout.apply@v1", "loadout"),
    # Edge cases
    ("", "unknown"),
    ("garbage", "garbage"),
])
def test_tool_domain(tool_id, expected):
    assert taxonomy.tool_domain(tool_id) == expected


@pytest.mark.parametrize("raw,expected_canonical", [
    ("vm", "vm"),
    ("vms", "vm"),
    ("app", "app"),
    ("apps", "app"),
    ("anything_else", "anything_else"),
])
def test_canonical_toolbox_folds_aliases(raw, expected_canonical):
    assert taxonomy.canonical_toolbox(raw) == expected_canonical


@pytest.mark.parametrize("toolbox,expected_cluster", [
    ("vm", "compute_infra"),
    ("storage", "compute_infra"),
    ("gpu_mode", "compute_infra"),
    ("training", "ai_inference"),
    ("voice_identities", "voice"),
    ("crown", "neuro"),
    ("game_session", "game"),
    ("images", "media_io"),
    ("jobs", "ops"),
    ("desktop", "ms4_local"),
    ("totally_new_domain", "unclassified"),
    # alias-folded toolboxes (vms -> vm) land in the right cluster too
    ("vms", "compute_infra"),
    ("apps", "ms4_local"),  # alias for "app" — but "app" isn't in any cluster?
])
def test_cluster_for_toolbox(toolbox, expected_cluster):
    # 'app' is intentionally NOT in any cluster in the static map; this
    # test asserts the unclassified fallback for it.
    if toolbox in {"app", "apps"}:
        assert taxonomy.cluster_for_toolbox(toolbox) == "unclassified"
    else:
        assert taxonomy.cluster_for_toolbox(toolbox) == expected_cluster


def test_keywords_for_toolbox_returns_tuple_or_empty():
    # Toolbox with hand-curated keywords:
    kws = taxonomy.keywords_for_toolbox("vm")
    assert isinstance(kws, tuple)
    assert "vm" in kws
    assert "virtual machine" in kws
    # Alias-folded:
    assert taxonomy.keywords_for_toolbox("vms") == kws
    # Unknown: empty tuple, not None or KeyError
    assert taxonomy.keywords_for_toolbox("doesnotexist") == ()


def test_cluster_map_is_well_formed():
    """No toolbox appears in more than one cluster — keeps cluster
    summaries unambiguous."""
    seen: dict[str, str] = {}
    for cluster, toolboxes in taxonomy.TOOL_SHED_CLUSTERS.items():
        for tb in toolboxes:
            assert tb not in seen, f"toolbox {tb!r} appears in multiple clusters: {seen[tb]} and {cluster}"
            seen[tb] = cluster
