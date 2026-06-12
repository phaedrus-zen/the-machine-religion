"""Depth Lobe model picker tests."""

from __future__ import annotations

import pytest

from machine_spirit_4.double_agent import (
    MS4_DEPTH_MODEL_ENV,
    choose_depth_model,
)
from machine_spirit_4.double_agent import depth_picker


@pytest.fixture(autouse=True)
def _clear_cache():
    depth_picker._clear_cache_for_tests()
    yield
    depth_picker._clear_cache_for_tests()


def _patch_catalog(monkeypatch, catalog):
    def _fake(_url, timeout=5):
        return catalog
    monkeypatch.setattr(depth_picker, "_http_get_models", _fake)


def test_envelope_override_beats_everything(monkeypatch):
    monkeypatch.setenv(MS4_DEPTH_MODEL_ENV, "env-only-default")
    _patch_catalog(monkeypatch, [{"id": "qwen3-coder-next:latest", "loaded": True, "available": True}])
    choice = choose_depth_model(
        hivemind_url="http://hive",
        envelope_override="job-pin-model",
        force_refresh=True,
    )
    assert choice.model_id == "job-pin-model"
    assert choice.source == "envelope_override"


def test_env_override_beats_catalog(monkeypatch):
    monkeypatch.setenv(MS4_DEPTH_MODEL_ENV, "env-pinned-big-model")
    _patch_catalog(monkeypatch, [{"id": "qwen3-coder-next:latest", "loaded": True, "available": True}])
    choice = choose_depth_model(hivemind_url="http://hive", force_refresh=True)
    assert choice.model_id == "env-pinned-big-model"
    assert choice.source == "env_override"


def test_picker_prefers_loaded_qwen3_coder_next(monkeypatch):
    monkeypatch.delenv(MS4_DEPTH_MODEL_ENV, raising=False)
    _patch_catalog(monkeypatch, [
        {"id": "qwen3-coder-next:latest", "loaded": True, "available": True},
        {"id": "qwen2.5-coder:32b", "loaded": True, "available": True},
        {"id": "qwen2.5:0.5b", "loaded": True, "available": True},
    ])
    choice = choose_depth_model(hivemind_url="http://hive", force_refresh=True)
    assert "qwen3-coder-next" in choice.model_id
    assert choice.source == "loaded"


def test_picker_falls_back_to_default_when_no_match(monkeypatch):
    monkeypatch.delenv(MS4_DEPTH_MODEL_ENV, raising=False)
    monkeypatch.delenv("MS4_DEFAULT_MODEL", raising=False)
    _patch_catalog(monkeypatch, [{"id": "exotic-toy:42b", "loaded": True, "available": True}])
    choice = choose_depth_model(hivemind_url="http://hive", force_refresh=True)
    assert choice.model_id == "qwen3-coder-next:latest"
    assert choice.source == "fallback"


def test_picker_fallback_when_catalog_errors(monkeypatch):
    monkeypatch.delenv(MS4_DEPTH_MODEL_ENV, raising=False)

    def boom(_url, timeout=5):
        raise RuntimeError("net down")
    monkeypatch.setattr(depth_picker, "_http_get_models", boom)
    choice = choose_depth_model(hivemind_url="http://hive", force_refresh=True)
    assert choice.source == "error"
    assert "net down" in choice.detail


# ---------------------------------------------------------------------------
# model_class-driven pick (resource_request.model_class wiring)
# ---------------------------------------------------------------------------

# Catalog where the class choice is observable: no qwen3-coder* present,
# so the default/deep_reasoning ordering hits "qwen3-next" first while
# deep_coder's ordering hits "deepseek-coder-v2" first.
_CLASS_SENSITIVE_CATALOG = [
    {"id": "qwen3-next:80b", "loaded": True, "available": True},
    {"id": "deepseek-coder-v2:16b", "loaded": True, "available": True},
]


def test_model_class_deep_coder_prefers_coder_families(monkeypatch):
    monkeypatch.delenv(MS4_DEPTH_MODEL_ENV, raising=False)
    _patch_catalog(monkeypatch, _CLASS_SENSITIVE_CATALOG)
    choice = choose_depth_model(
        hivemind_url="http://hive", model_class="deep_coder", force_refresh=True
    )
    assert choice.model_id == "deepseek-coder-v2:16b"
    assert choice.source == "loaded"


def test_model_class_deep_reasoning_matches_legacy_ordering(monkeypatch):
    monkeypatch.delenv(MS4_DEPTH_MODEL_ENV, raising=False)
    _patch_catalog(monkeypatch, _CLASS_SENSITIVE_CATALOG)
    default_choice = choose_depth_model(hivemind_url="http://hive", force_refresh=True)
    reasoning_choice = choose_depth_model(
        hivemind_url="http://hive", model_class="deep_reasoning", force_refresh=True
    )
    assert default_choice.model_id == "qwen3-next:80b"
    assert reasoning_choice.model_id == default_choice.model_id


def test_model_class_unknown_falls_back_to_default_ordering(monkeypatch):
    monkeypatch.delenv(MS4_DEPTH_MODEL_ENV, raising=False)
    _patch_catalog(monkeypatch, _CLASS_SENSITIVE_CATALOG)
    choice = choose_depth_model(
        hivemind_url="http://hive", model_class="quantum_vibes", force_refresh=True
    )
    assert choice.model_id == "qwen3-next:80b"


def test_model_class_is_part_of_the_cache_key(monkeypatch):
    monkeypatch.delenv(MS4_DEPTH_MODEL_ENV, raising=False)
    _patch_catalog(monkeypatch, _CLASS_SENSITIVE_CATALOG)
    coder = choose_depth_model(
        hivemind_url="http://hive", model_class="deep_coder", force_refresh=True
    )
    # No force_refresh: a different class must not be served the
    # coder-class cached choice.
    reasoning = choose_depth_model(
        hivemind_url="http://hive", model_class="deep_reasoning"
    )
    assert coder.model_id == "deepseek-coder-v2:16b"
    assert reasoning.model_id == "qwen3-next:80b"


def test_model_class_envelope_override_still_wins(monkeypatch):
    monkeypatch.delenv(MS4_DEPTH_MODEL_ENV, raising=False)
    _patch_catalog(monkeypatch, _CLASS_SENSITIVE_CATALOG)
    choice = choose_depth_model(
        hivemind_url="http://hive",
        envelope_override="pinned-model:1b",
        model_class="deep_coder",
        force_refresh=True,
    )
    assert choice.model_id == "pinned-model:1b"
    assert choice.source == "envelope_override"
