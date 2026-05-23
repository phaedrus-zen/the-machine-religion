"""Face Lobe model picker tests."""

from __future__ import annotations

import pytest

from machine_spirit_4.double_agent import (
    DEFAULT_FOREGROUND_MODEL,
    MS4_FOREGROUND_MODEL_ENV,
    choose_foreground_model,
)
from machine_spirit_4.double_agent import model_picker


@pytest.fixture(autouse=True)
def _clear_cache():
    model_picker._clear_cache_for_tests()
    yield
    model_picker._clear_cache_for_tests()


def _patch_catalog(monkeypatch, catalog):
    def _fake_get(_url, timeout=5):
        return catalog
    monkeypatch.setattr(model_picker, "_http_get_models", _fake_get)


def test_env_override_beats_catalog(monkeypatch):
    monkeypatch.setenv(MS4_FOREGROUND_MODEL_ENV, "custom-tiny:0.5b")
    _patch_catalog(monkeypatch, [{"id": "qwen2.5:0.5b", "loaded": True, "available": True}])
    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)
    assert choice.model_id == "custom-tiny:0.5b"
    assert choice.source == "override"


def test_picker_prefers_coherent_4b_face_lobe_model(monkeypatch):
    """The Face Lobe bypasses Hermes (direct /v1/chat/completions) so
    short-context models are fine, but sub-2B models (qwen2.5:0.5b,
    tinyllama) don't reliably follow the Face Lobe system prompt — we
    observed live refusals on benign messages. The picker prefers a
    4–8B instruct model when one is loaded, even if a tiny one is
    also loaded."""
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    _patch_catalog(monkeypatch, [
        {"id": "qwen2.5:0.5b", "loaded": True, "available": True},
        {"id": "llama3.1:8b", "loaded": True, "available": True},
        {"id": "qwen3-coder-next:latest", "loaded": True, "available": True},
    ])
    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)
    assert choice.model_id == "llama3.1:8b"
    assert choice.source == "loaded"


def test_picker_prefers_phi4mini_when_loaded(monkeypatch):
    """phi4-mini sits at the top of the priority list — it's a 3.8B
    instruct model with strong system-prompt adherence, fast enough on
    consumer hardware via direct chat-completion."""
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    _patch_catalog(monkeypatch, [
        {"id": "phi4-mini", "loaded": True, "available": True},
        {"id": "llama3.1:8b", "loaded": True, "available": True},
        {"id": "gemma4", "loaded": True, "available": True},
    ])
    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)
    assert choice.model_id == "phi4-mini"
    assert choice.source == "loaded"


def test_picker_never_picks_dumb_tiny_models_for_foreground(monkeypatch):
    """qwen2.5:0.5b and tinyllama:1.1b are deliberately NOT in the
    priority list. If they're the only `loaded` entries available, the
    picker should fall all the way through to a higher-quality model
    (loaded OR available) rather than picking them."""
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    _patch_catalog(monkeypatch, [
        {"id": "qwen2.5:0.5b", "loaded": True, "available": True},
        {"id": "tinyllama:1.1b", "loaded": True, "available": True},
        {"id": "phi3.5", "loaded": False, "available": True},
    ])
    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)
    assert "phi" in choice.model_id.lower(), \
        f"picker shouldn't choose tiny models for the foreground; got {choice.model_id}"


def test_picker_falls_through_priority_when_top_unavailable(monkeypatch):
    """phi4-mini absent, phi3.5 also absent. Catalog has llama3.1:8b
    (priority entry, loaded). Picker should land on llama3.1:8b."""
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    _patch_catalog(monkeypatch, [
        {"id": "llama3.1:8b", "loaded": True, "available": True},
        {"id": "qwen3-coder-next:latest", "loaded": True, "available": True},
    ])
    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)
    assert choice.model_id == "llama3.1:8b"
    assert choice.source == "loaded"


def test_picker_prefers_loaded_lower_priority_over_available_higher_priority(monkeypatch):
    """THE bug that caused the live '(no reply text)' turns: phi4-mini
    (priority pattern #1) was in the catalog as ``available:true /
    loaded:false`` while llama3.1:8b (priority #6) was ``loaded:true``.
    The old picker would return phi4-mini (cold-load risk, observed
    empty content), but the right answer is llama3.1:8b — a loaded
    model anywhere in the priority list beats an available higher one,
    because warm beats waiting-for-cold-load in every interactive
    context."""
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    _patch_catalog(monkeypatch, [
        {"id": "phi4-mini", "loaded": False, "available": True},
        {"id": "llama3.1:8b", "loaded": True, "available": True},
        {"id": "qwen3-coder-next:latest", "loaded": True, "available": True},
    ])
    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)
    assert choice.model_id == "llama3.1:8b"
    assert choice.source == "loaded"


def test_picker_only_uses_available_when_nothing_priority_is_loaded(monkeypatch):
    """If NO priority pattern has a loaded entry, the picker falls back
    to the highest-priority *available* entry (still better than the
    static fallback)."""
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    _patch_catalog(monkeypatch, [
        {"id": "phi4-mini", "loaded": False, "available": True},
        {"id": "gemma3", "loaded": False, "available": True},
        {"id": "qwen3-coder-next:latest", "loaded": False, "available": True},
    ])
    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)
    assert choice.model_id == "phi4-mini"  # priority #1 in priority list
    assert choice.source == "available"


def test_picker_understands_real_hivemind_status_field(monkeypatch):
    """HiveMind's live ``/v1/models`` doesn't use the OpenAI-style
    ``loaded``/``available`` booleans; it uses ``hivemind_status`` with
    values like ``installed`` (warm-ready), ``available`` (cold), etc.
    The picker MUST treat ``installed + reachable`` as loaded —
    otherwise (the live bug) every model gets tagged "available", the
    loaded-pass never matches anything, and we pick the wrong
    higher-priority cold model."""
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    _patch_catalog(monkeypatch, [
        # phi4-mini priority #1 — but only available, not installed.
        {"id": "phi4-mini:latest", "hivemind_status": "available", "hivemind_reachable": True},
        # llama3.1:8b priority #6 — installed and reachable (warm).
        {"id": "llama3.1:8b", "hivemind_status": "installed", "hivemind_reachable": True},
        # An alias-suffixed variant of the same warm model: must NOT win.
        {"id": "llama3.1:8b_ollama", "hivemind_status": "installed", "hivemind_reachable": True},
        # qwen3-coder-next:latest — installed too, but lower priority than llama3.1:8b.
        {"id": "qwen3-coder-next:latest", "hivemind_status": "installed", "hivemind_reachable": True},
    ])
    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)
    assert choice.model_id == "llama3.1:8b"
    assert choice.source == "loaded"
    assert "installed" in choice.detail


def test_picker_skips_unreachable_models(monkeypatch):
    """Unreachable catalog entries (other nodes have them but our
    cluster can't route there) must not be picked even if they're
    nominally installed."""
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    _patch_catalog(monkeypatch, [
        {"id": "phi4-mini:latest", "hivemind_status": "installed", "hivemind_reachable": False},
        {"id": "llama3.1:8b", "hivemind_status": "installed", "hivemind_reachable": True},
    ])
    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)
    assert choice.model_id == "llama3.1:8b"
    assert choice.source == "loaded"


def test_picker_prefers_canonical_id_over_alias_suffix(monkeypatch):
    """``llama3.1:8b`` beats ``llama3.1:8b_ollama`` when both match
    the same priority pattern with equal readiness — the canonical
    id is what operators expect to see in audit logs."""
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    _patch_catalog(monkeypatch, [
        {"id": "llama3.1:8b_ollama", "hivemind_status": "installed", "hivemind_reachable": True},
        {"id": "llama3.1:8b", "hivemind_status": "installed", "hivemind_reachable": True},
    ])
    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)
    assert choice.model_id == "llama3.1:8b"


def test_picker_fallback_when_catalog_has_no_priority_match(monkeypatch):
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    monkeypatch.delenv("MS4_DEFAULT_MODEL", raising=False)
    _patch_catalog(monkeypatch, [
        {"id": "exotic-model:42b", "loaded": True, "available": True},
    ])
    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)
    assert choice.model_id == DEFAULT_FOREGROUND_MODEL
    assert choice.source == "fallback"


def test_picker_fallback_when_catalog_endpoint_errors(monkeypatch):
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    monkeypatch.delenv("MS4_DEFAULT_MODEL", raising=False)

    def _boom(_url, timeout=5):
        raise RuntimeError("network down")
    monkeypatch.setattr(model_picker, "_http_get_models", _boom)
    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)
    assert choice.model_id == DEFAULT_FOREGROUND_MODEL
    assert choice.source == "error"
    assert "network down" in choice.detail


def test_picker_fallback_honors_ms4_default_model_env(monkeypatch):
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    monkeypatch.setenv("MS4_DEFAULT_MODEL", "operator-pinned:latest")
    _patch_catalog(monkeypatch, [
        {"id": "exotic-model:42b", "loaded": True, "available": True},
    ])
    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)
    assert choice.model_id == "operator-pinned:latest"
    assert choice.source == "fallback"


def test_picker_caches_within_ttl(monkeypatch):
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    calls = {"n": 0}

    def _fake(_url, timeout=5):
        calls["n"] += 1
        return [{"id": "qwen2.5:0.5b", "loaded": True, "available": True}]
    monkeypatch.setattr(model_picker, "_http_get_models", _fake)

    choose_foreground_model(hivemind_url="http://hive")
    choose_foreground_model(hivemind_url="http://hive")
    choose_foreground_model(hivemind_url="http://hive")
    assert calls["n"] == 1
    choose_foreground_model(hivemind_url="http://hive", force_refresh=True)
    assert calls["n"] == 2
