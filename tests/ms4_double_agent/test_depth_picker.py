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
