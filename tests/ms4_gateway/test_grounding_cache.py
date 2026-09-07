"""TTL cache + stale-fallback for HiveMind grounding fetches.

The MCP calls that back the ``hivemind-mcp-live-context`` (inventory)
and ``ms4-tools-grounding`` (tools list) groundings can take 15-20s
under cluster load, and they used to fire on EVERY chat turn that
matched. That made ``what's the cluster status?`` a 40s round-trip.

These tests lock in the new behavior:

* First call fetches and caches.
* Subsequent calls within ``MS4_GROUNDING_CACHE_TTL`` (default 60s)
  return the cached value and the grounding_source label carries
  ``+cached(age=Ns)`` so the operator can see the hit.
* Once the TTL expires the next call refetches.
* If a refetch fails AND a stale cached value exists, we serve the
  stale value (better than a "lookup failed" prompt) and the label
  carries ``+stale(age=Ns)``.
* If a refetch fails AND no cached value exists, the original error
  branch fires (``hivemind-mcp-live-context-error``).
* Cache keys are per-(hivemind_url) so swapping clusters doesn't
  mix state.
"""

from __future__ import annotations

import time

import pytest

from machine_spirit_4.gateway import context as ctx_module
from machine_spirit_4.gateway.context import build_grounded_user_message


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_grounding_cache(monkeypatch):
    ctx_module.clear_grounding_cache()
    monkeypatch.setattr(ctx_module, "GROUNDING_CACHE_TTL_SECS", 60)
    yield
    ctx_module.clear_grounding_cache()


_FAKE_SUMMARY_JSON = (
    '{"cluster_statistics": {"total_nodes": 8, "active_nodes": 8, "total_gpus": 16}}'
)
_FAKE_HOSTS_JSON = (
    '{"nodes": ['
    '{"name": "DESKTOP-OH048LC", "status": "active", "ip_addresses": ["192.168.0.196"],'
    ' "hardware": {"devices": {"g0": {"compute_device_type": "GPU", "manufacturer": "NVIDIA", "device_name": "RTX PRO 6000"}}}},'
    '{"name": "nexus-server", "status": "active", "ip_addresses": ["192.168.0.21"], "hardware": {"devices": {}}}'
    ']}'
)


@pytest.fixture
def fake_mcp(monkeypatch):
    """Replace context.mcp_call with a counter so each test can assert
    exactly how many times we hit HiveMind. Default returns canned
    cluster.summary / hosts.list payloads. mcp_call's contract is
    'returns a JSON string that the formatter can json.loads()'."""
    calls = {"n": 0, "by_name": []}

    def fake(hivemind_url, name, arguments, timeout=30):
        calls["n"] += 1
        calls["by_name"].append(name)
        if name == "hivemind.cluster.summary@v1":
            return _FAKE_SUMMARY_JSON
        if name == "hivemind.hosts.list@v1":
            return _FAKE_HOSTS_JSON
        raise RuntimeError(f"unexpected MCP call {name!r}")

    monkeypatch.setattr(ctx_module, "mcp_call", fake)
    return calls


# ---------------------------------------------------------------------------
# Inventory grounding
# ---------------------------------------------------------------------------


def test_inventory_first_call_fetches_and_caches(fake_mcp):
    msg = "What's the current status of the cluster GPUs?"
    text, source = build_grounded_user_message(msg, "http://hive:6089")
    assert source == "hivemind-mcp-live-context"
    assert "8 total node" in text
    # Exactly one inventory turn = 2 MCP calls (summary + hosts).
    assert fake_mcp["n"] == 2


def test_inventory_second_call_hits_cache(fake_mcp):
    msg = "What's the current status of the cluster GPUs?"
    build_grounded_user_message(msg, "http://hive:6089")  # primes cache
    text, source = build_grounded_user_message(msg, "http://hive:6089")
    assert source.startswith("hivemind-mcp-live-context+cached(age=")
    assert "8 total node" in text
    assert fake_mcp["n"] == 2, "second call must NOT refetch within TTL"


def test_inventory_cache_expires_and_refetches(fake_mcp, monkeypatch):
    monkeypatch.setattr(ctx_module, "GROUNDING_CACHE_TTL_SECS", 0)
    msg = "List the GPUs in the cluster."
    build_grounded_user_message(msg, "http://hive:6089")
    # TTL=0 -> any non-zero age expires the entry.
    time.sleep(0.02)
    build_grounded_user_message(msg, "http://hive:6089")
    assert fake_mcp["n"] == 4, "TTL expiry must trigger a refetch (4 calls = 2 turns × 2)"


def test_inventory_failure_falls_back_to_stale_cache(monkeypatch):
    """If the fresh fetch raises AND we have a stale cached value, we
    MUST serve the stale value with a +stale(age=...) label instead
    of returning the bare error message in the prompt."""
    calls = {"n": 0}

    def fake_mcp(hivemind_url, name, arguments, timeout=30):
        calls["n"] += 1
        if calls["n"] <= 2:
            # First turn: succeed, populate cache.
            return _FAKE_SUMMARY_JSON if "summary" in name else _FAKE_HOSTS_JSON
        # Subsequent calls: fail.
        raise RuntimeError("MCP storm")

    monkeypatch.setattr(ctx_module, "mcp_call", fake_mcp)
    monkeypatch.setattr(ctx_module, "GROUNDING_CACHE_TTL_SECS", 0)
    msg = "How many GPUs?"

    # Turn 1: succeed, cache populated.
    _, source1 = build_grounded_user_message(msg, "http://hive:6089")
    assert source1 == "hivemind-mcp-live-context"

    # Turn 2: TTL expired, refetch raises; we should fall back to stale.
    time.sleep(0.02)
    text2, source2 = build_grounded_user_message(msg, "http://hive:6089")
    assert source2.startswith("hivemind-mcp-live-context+stale(age=")
    assert "8 total node" in text2, "stale fallback must reuse the previously cached text"


def test_inventory_failure_no_cache_returns_error_branch(monkeypatch):
    """If the very first turn fails with no cache to fall back on,
    keep today's existing error branch behavior so operators still
    get a clear signal in the prompt."""

    def fake_mcp(hivemind_url, name, arguments, timeout=30):
        raise RuntimeError("MCP storm")

    monkeypatch.setattr(ctx_module, "mcp_call", fake_mcp)
    text, source = build_grounded_user_message("how many GPUs?", "http://hive:6089")
    assert source == "hivemind-mcp-live-context-error"
    assert "MCP storm" in text


def test_inventory_cache_key_isolates_per_cluster_url(fake_mcp):
    """Switching the hivemind_url MUST NOT serve a cached value from a
    different cluster — operator may legitimately be talking to two
    HiveMind deployments and the inventories don't share state."""
    msg = "list the GPUs in the cluster"
    build_grounded_user_message(msg, "http://hive-a:6089")
    build_grounded_user_message(msg, "http://hive-b:6089")
    assert fake_mcp["n"] == 4, "different URLs must trigger separate fetches"


# ---------------------------------------------------------------------------
# Wall-clock fetch budget (a slow cold cluster must not stall the turn)
# ---------------------------------------------------------------------------


def test_grounding_budget_bounds_slow_fetch_and_serves_stale(monkeypatch):
    key = "inv::budget-stale"
    # Seed an OLD entry (age > 0) so TTL=0 treats it as stale, not a fresh hit.
    with ctx_module._GROUNDING_CACHE_LOCK:
        ctx_module._GROUNDING_CACHE[key] = (time.time() - 120, "OLD INVENTORY")
    monkeypatch.setattr(ctx_module, "GROUNDING_CACHE_TTL_SECS", 0)  # force a refetch

    def slow_fetch():
        time.sleep(2.0)
        return "NEW INVENTORY"

    t0 = time.monotonic()
    text, source = ctx_module._grounding_with_cache(
        cache_key=key, fetch=slow_fetch, fresh_source="src", budget_s=0.3,
    )
    elapsed = time.monotonic() - t0
    assert elapsed < 1.5, "must not wait out the full 2s fetch"
    assert text == "OLD INVENTORY", "served the stale cached value"
    assert source.startswith("src+stale(age=")


def test_grounding_budget_no_cache_raises_within_budget():
    def slow_fetch():
        time.sleep(2.0)
        return "X"

    t0 = time.monotonic()
    with pytest.raises(Exception):
        ctx_module._grounding_with_cache(
            cache_key="inv::budget-nocache", fetch=slow_fetch, fresh_source="src", budget_s=0.3,
        )
    assert time.monotonic() - t0 < 1.5, "budget must bound the wait even with no cache"


def test_grounding_late_fetch_warms_cache_for_next_turn(monkeypatch):
    monkeypatch.setattr(ctx_module, "GROUNDING_CACHE_TTL_SECS", 60)
    key = "inv::late-warm"

    def slowish():
        time.sleep(0.4)
        return "WARMED"

    # First turn gives up at 0.2s (no cache) and raises.
    with pytest.raises(Exception):
        ctx_module._grounding_with_cache(cache_key=key, fetch=slowish, fresh_source="src", budget_s=0.2)
    # The background thread finishes ~0.4s later and warms the cache.
    time.sleep(0.6)
    text, source = ctx_module._grounding_with_cache(
        cache_key=key, fetch=lambda: "SHOULD_NOT_RUN", fresh_source="src", budget_s=5.0,
    )
    assert text == "WARMED", "late-completing fetch should have warmed the cache"
    assert source.startswith("src+cached(age=")


# ---------------------------------------------------------------------------
# Tools grounding
# ---------------------------------------------------------------------------


def test_tools_grounding_is_cached(monkeypatch):
    calls = {"n": 0}

    def fake_format(ms3, hive, gw):
        calls["n"] += 1
        return "MS4 tools: ms4.identity.verify@v1, ms4.ethics.evaluate@v1, ..."

    monkeypatch.setattr(ctx_module, "format_tools_answer", fake_format)
    msg = "what tools do you have?"
    build_grounded_user_message(msg, "http://hive:6089")
    text2, source2 = build_grounded_user_message(msg, "http://hive:6089")
    assert source2.startswith("ms4-tools-grounding+cached(age=")
    assert "ms4.identity.verify@v1" in text2
    assert calls["n"] == 1, "tools grounding must hit cache on second turn"


# ---------------------------------------------------------------------------
# Pre-warm
# ---------------------------------------------------------------------------


def test_prewarm_grounding_cache_seeds_both_entries(monkeypatch):
    calls = {"mcp": 0, "tools": 0}

    def fake_mcp(hivemind_url, name, arguments, timeout=30):
        calls["mcp"] += 1
        return _FAKE_SUMMARY_JSON if "summary" in name else _FAKE_HOSTS_JSON

    def fake_format(ms3, hive, gw):
        calls["tools"] += 1
        return "tools text"

    monkeypatch.setattr(ctx_module, "mcp_call", fake_mcp)
    monkeypatch.setattr(ctx_module, "format_tools_answer", fake_format)

    status = ctx_module.prewarm_grounding_cache(hivemind_url="http://hive:6089")
    assert status["inventory"]["ok"] is True
    assert status["tools"]["ok"] is True
    assert calls["mcp"] == 2 and calls["tools"] == 1

    # A subsequent live request should hit the cache (no extra fetches).
    text, source = build_grounded_user_message("list the GPUs in the cluster", "http://hive:6089")
    assert source.startswith("hivemind-mcp-live-context+cached(age=")
    assert calls["mcp"] == 2, "pre-warmed cache must satisfy the first live turn"
