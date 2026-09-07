"""Depth Lobe model picker tests."""

from __future__ import annotations

import pytest

from machine_spirit_4.double_agent import (
    MS4_DEPTH_MODEL_ENV,
    choose_depth_model,
)
from machine_spirit_4.double_agent import depth_picker


@pytest.fixture(autouse=True)
def _clear_cache(monkeypatch):
    monkeypatch.delenv(MS4_DEPTH_MODEL_ENV, raising=False)
    monkeypatch.delenv(depth_picker.DEFAULT_DEPTH_MODEL_ENV, raising=False)
    monkeypatch.delenv("MS4_DEFAULT_MODEL", raising=False)
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
    monkeypatch.setenv(MS4_DEPTH_MODEL_ENV, "gemma4:31b")
    _patch_catalog(monkeypatch, [{"id": "nemotron-3-nano:30b", "loaded": True, "available": True}])
    choice = choose_depth_model(hivemind_url="http://hive", force_refresh=True)
    assert choice.model_id == "gemma4:31b"
    assert choice.source == "env_override"


def test_picker_prefers_quality_lead_35b_then_verified_nemotron(monkeypatch):
    _patch_catalog(monkeypatch, [
        {"id": "nemotron-3-nano:30b", "loaded": True, "available": True},
        {"id": "gemma4:31b", "loaded": True, "available": True},
        {"id": "qwen3.6:35b", "loaded": True, "available": True},
        {"id": "qwen2.5-coder:32b", "loaded": True, "available": True},
        {"id": "qwen2.5:0.5b", "loaded": True, "available": True},
    ])
    choice = choose_depth_model(hivemind_url="http://hive", force_refresh=True)
    assert choice.model_id == "qwen3.6:35b"
    assert choice.source == "loaded"


def test_below_quality_models_use_degraded_nemotron_fallback(monkeypatch):
    _patch_catalog(monkeypatch, [
        {"id": "nemotron-3-nano:30b", "loaded": True, "available": True},
        {"id": "gemma4:31b", "loaded": True, "available": True},
    ])

    choice = choose_depth_model(hivemind_url="http://hive", force_refresh=True)

    assert choice.model_id == "nemotron-3-nano:30b"
    assert choice.source == "fallback"
    assert ">=35B quality floor" in choice.detail
    assert "degraded fast/tool fallback" in choice.detail


@pytest.mark.parametrize("model_class", [None, "deep_reasoning", "deep_coder"])
def test_picker_does_not_auto_select_manual_only_gemma(monkeypatch, model_class):
    _patch_catalog(monkeypatch, [
        {"id": "gemma4:31b", "loaded": True, "available": True},
    ])

    choice = choose_depth_model(
        hivemind_url="http://hive",
        model_class=model_class,
        force_refresh=True,
    )

    assert choice.model_id == "nemotron-3-nano:30b"
    assert choice.source == "fallback"


def test_picker_uses_degraded_fallback_when_quality_target_is_only_available(monkeypatch):
    _patch_catalog(monkeypatch, [
        {"id": "qwen3.6:35b", "loaded": False, "available": True},
        {"id": "nemotron-3-nano:30b", "loaded": True, "available": True},
    ])
    choice = choose_depth_model(hivemind_url="http://hive", force_refresh=True)
    assert choice.model_id == "nemotron-3-nano:30b"
    assert choice.source == "fallback"


def test_preferred_qwen_installed_reachable_survives_idle_provider_unload(monkeypatch):
    """HLI cluster readiness, not provider-local residency, governs Depth."""
    _patch_catalog(monkeypatch, [
        {
            "id": "qwen3.6:35b",
            "hivemind_status": "installed",
            "hivemind_reachable": True,
        },
        {"id": "nemotron-3-nano:30b", "loaded": True, "available": True},
    ])

    choice = choose_depth_model(hivemind_url="http://hive", force_refresh=True)

    assert choice.model_id == depth_picker.DEPTH_PREFERRED_CLUSTER_TARGET
    assert choice.source == "loaded"
    assert "hivemind_status='installed'" in choice.detail
    assert "reachable=True" in choice.detail


def test_unreachable_installed_qwen_falls_through_to_ready_nemotron(monkeypatch):
    _patch_catalog(monkeypatch, [
        {
            "id": "qwen3.6:35b",
            "hivemind_status": "installed",
            "hivemind_reachable": False,
        },
        {"id": "nemotron-3-nano:30b", "loaded": True, "available": True},
    ])

    choice = choose_depth_model(hivemind_url="http://hive", force_refresh=True)

    assert choice.model_id == "nemotron-3-nano:30b"
    assert choice.source == "fallback"


def test_picker_falls_back_to_default_when_no_match(monkeypatch):
    _patch_catalog(monkeypatch, [{"id": "exotic-toy:42b", "loaded": True, "available": True}])
    choice = choose_depth_model(hivemind_url="http://hive", force_refresh=True)
    assert choice.model_id == "nemotron-3-nano:30b"
    assert choice.source == "fallback"


def test_picker_fallback_when_catalog_errors(monkeypatch):
    def boom(_url, timeout=5):
        raise RuntimeError("net down")
    monkeypatch.setattr(depth_picker, "_http_get_models", boom)
    choice = choose_depth_model(hivemind_url="http://hive", force_refresh=True)
    assert choice.model_id == "nemotron-3-nano:30b"
    assert choice.source == "error"
    assert "net down" in choice.detail


# ---------------------------------------------------------------------------
# model_class-driven pick (resource_request.model_class wiring)
# ---------------------------------------------------------------------------

# Catalog where the class choice is observable while every automatic candidate
# still clears the 35B Depth quality floor.
_CLASS_SENSITIVE_CATALOG = [
    {"id": "qwen3.6:35b", "loaded": True, "available": True},
    {"id": "qwen2.5-coder:72b", "loaded": True, "available": True},
]


def test_model_class_deep_coder_prefers_coder_families(monkeypatch):
    _patch_catalog(monkeypatch, _CLASS_SENSITIVE_CATALOG)
    choice = choose_depth_model(
        hivemind_url="http://hive", model_class="deep_coder", force_refresh=True
    )
    assert choice.model_id == "qwen2.5-coder:72b"
    assert choice.source == "loaded"


def test_model_class_deep_reasoning_matches_legacy_ordering(monkeypatch):
    _patch_catalog(monkeypatch, _CLASS_SENSITIVE_CATALOG)
    default_choice = choose_depth_model(hivemind_url="http://hive", force_refresh=True)
    reasoning_choice = choose_depth_model(
        hivemind_url="http://hive", model_class="deep_reasoning", force_refresh=True
    )
    assert default_choice.model_id == "qwen3.6:35b"
    assert reasoning_choice.model_id == default_choice.model_id


def test_model_class_unknown_falls_back_to_default_ordering(monkeypatch):
    _patch_catalog(monkeypatch, _CLASS_SENSITIVE_CATALOG)
    choice = choose_depth_model(
        hivemind_url="http://hive", model_class="quantum_vibes", force_refresh=True
    )
    assert choice.model_id == "qwen3.6:35b"


def test_model_class_is_part_of_the_cache_key(monkeypatch):
    _patch_catalog(monkeypatch, _CLASS_SENSITIVE_CATALOG)
    coder = choose_depth_model(
        hivemind_url="http://hive", model_class="deep_coder", force_refresh=True
    )
    # No force_refresh: a different class must not be served the
    # coder-class cached choice.
    reasoning = choose_depth_model(
        hivemind_url="http://hive", model_class="deep_reasoning"
    )
    assert coder.model_id == "qwen2.5-coder:72b"
    assert reasoning.model_id == "qwen3.6:35b"


def test_model_class_envelope_override_still_wins(monkeypatch):
    _patch_catalog(monkeypatch, _CLASS_SENSITIVE_CATALOG)
    choice = choose_depth_model(
        hivemind_url="http://hive",
        envelope_override="pinned-model:1b",
        model_class="deep_coder",
        force_refresh=True,
    )
    assert choice.model_id == "pinned-model:1b"
    assert choice.source == "envelope_override"


@pytest.mark.parametrize(
    "metadata",
    [
        {"hivemind_category": "tts"},
        {"category": "embedding"},
        {"hivemind_capabilities": {"image-generation": True}},
    ],
    ids=("tts", "embedding", "image"),
)
def test_depth_picker_rejects_opaque_authoritative_non_chat_metadata(metadata):
    entry = {
        "id": "nemotron-3-nano:30b",
        "hivemind_status": "installed",
        **metadata,
    }

    choice = depth_picker._pick_from_catalog(
        [entry], depth_picker.DEPTH_PRIORITY_PATTERNS
    )

    assert choice is None


def test_depth_picker_tts_only_catalog_falls_back(monkeypatch):
    _patch_catalog(
        monkeypatch,
        [
            {
                "id": "nemotron-3-nano:30b",
                "hivemind_category": "tts",
                "hivemind_status": "installed",
            },
            {
                "id": "deepseek-r1-private",
                "category": "text-to-speech",
                "hivemind_status": "installed",
            },
        ],
    )

    choice = choose_depth_model(hivemind_url="http://hive", force_refresh=True)

    assert choice.model_id == "nemotron-3-nano:30b"
    assert choice.source == "fallback"


def test_depth_catalog_filters_before_normalization(monkeypatch):
    catalog = [
        {
            "id": "qwen3-coder-next:tts",
            "hivemind_status": "installed",
            "category": "tts",
        },
        {
            "id": "deepseek-r1-embed",
            "hivemind_status": "installed",
            "hivemind_category": "embedding",
        },
        {
            "id": "qwen2.5-coder-image",
            "hivemind_status": "installed",
            "hivemind_capabilities": {"image-generation": True},
        },
        {
            "id": "qwen2.5-coder:72b",
            "hivemind_status": "installed",
            "category": "llm",
            "capabilities": ["chat"],
        },
    ]
    seen = []
    original = depth_picker._normalize_model_entry

    def track(entry):
        seen.append(entry["id"])
        return original(entry)

    monkeypatch.setattr(depth_picker, "_normalize_model_entry", track)

    choice = depth_picker._pick_from_catalog(catalog)

    assert seen == ["qwen2.5-coder:72b"]
    assert choice is not None
    assert choice.model_id == "qwen2.5-coder:72b"


def test_automatic_picker_rejects_loaded_models_below_35b(monkeypatch):
    _patch_catalog(monkeypatch, [
        {"id": "qwen2.5-coder:32b", "loaded": True, "available": True},
        {"id": "nemotron-3-nano:30b", "loaded": True, "available": True},
        {"id": "qwen3.6:27b", "loaded": True, "available": True},
        {"id": "deepseek-coder-v2:16b", "loaded": True, "available": True},
        {"id": "qwen3:8b", "loaded": True, "available": True},
    ])

    choice = choose_depth_model(hivemind_url="http://hive", force_refresh=True)

    assert choice.model_id == "nemotron-3-nano:30b"
    assert choice.source == "fallback"
    assert ">=35B quality floor" in choice.detail
    assert "degraded fast/tool fallback" in choice.detail


def test_depth_selection_is_cluster_scoped_not_local_vram_gated(monkeypatch):
    _patch_catalog(monkeypatch, [
        {
            "id": "qwen3.6:35b",
            "hivemind_status": "installed",
            "hivemind_reachable": True,
            "hivemind_host": "remote-5090-node",
        }
    ])

    choice = choose_depth_model(hivemind_url="http://coordinator", force_refresh=True)

    assert choice.model_id == "qwen3.6:35b"
    assert choice.source == "loaded"


def test_below_floor_depth_fallback_env_and_shared_face_default_are_ignored(monkeypatch):
    monkeypatch.setenv(depth_picker.DEFAULT_DEPTH_MODEL_ENV, "qwen3:8b")
    monkeypatch.setenv("MS4_DEFAULT_MODEL", "llama3.1:8b")
    _patch_catalog(monkeypatch, [])

    choice = choose_depth_model(hivemind_url="http://hive", force_refresh=True)

    assert choice.model_id == "nemotron-3-nano:30b"
    assert choice.source == "fallback"


def test_valid_depth_fallback_env_is_gated_and_part_of_cache_key(monkeypatch):
    _patch_catalog(monkeypatch, [])
    monkeypatch.setenv(depth_picker.DEFAULT_DEPTH_MODEL_ENV, "gemma4:31b")
    first = choose_depth_model(hivemind_url="http://hive")

    monkeypatch.setenv(depth_picker.DEFAULT_DEPTH_MODEL_ENV, "qwen3.6:35b")
    second = choose_depth_model(hivemind_url="http://hive")

    assert first.model_id == "gemma4:31b"
    assert second.model_id == "qwen3.6:35b"


def test_depth_size_parser_uses_total_weights_for_sparse_model():
    assert depth_picker._declared_total_params_b("nemotron-3-nano:30b-a3b") == 30
    assert depth_picker._declared_total_params_b("qwen3.6:35b_ollama") == 35
    assert depth_picker._meets_depth_quality_floor("qwen3.6:35b") is True
    assert depth_picker._meets_depth_quality_floor("nemotron-3-nano:30b") is False
    assert depth_picker._meets_depth_fallback_floor("nemotron-3-nano:30b") is True
    assert depth_picker._meets_depth_quality_floor("qwen3.6:27b") is False
    assert depth_picker._meets_depth_quality_floor("qwen3-coder-next:latest") is False
