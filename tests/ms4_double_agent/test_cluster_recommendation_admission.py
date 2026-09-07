"""Exact Face warm→probe admission is additive; ordinary picker is unchanged."""

from __future__ import annotations

import math

import pytest

from machine_spirit_4.double_agent import model_picker
from machine_spirit_4.double_agent.model_picker import (
    EXACT_FACE_CLUSTER_ADMISSION_SCHEMA,
    EXACT_FACE_CLUSTER_ADMISSION_TTL_S,
    ForegroundModelUnavailable,
    admit_cluster_recommendation,
    choose_foreground_model,
    require_exact_face_cluster_admission,
)
from machine_spirit_4.gateway import hivemind_tools

NOW_S = 1_700_000_000.0
CATALOG_IDS = {"nemotron-3-nano:4b", "llama3.1:8b"}


def _openai_row(model_id: str) -> dict:
    return {
        "id": model_id,
        "object": "model",
        "created": 0,
        "owned_by": "organization",
    }


def _admitted(**overrides: object) -> dict:
    body: dict = {
        "schema": EXACT_FACE_CLUSTER_ADMISSION_SCHEMA,
        "capability": "chat",
        "recommended_model": "nemotron-3-nano:4b",
        "backend": "ollama",
        "ready": True,
        "owner": "node-a",
        "endpoint": "http://node-a:11434/v1/chat/completions",
        "snapshot_age_s": 1.25,
        "current_generation": 7,
        "lease": {
            "id": "lease-nemotron-node-a",
            "affinity": "node-a:ollama:nemotron-3-nano:4b",
            "issued": NOW_S - 5,
            "expires": NOW_S + 30,
            "generation": 7,
        },
    }
    lease_override = overrides.pop("lease", None)
    body.update(overrides)
    if isinstance(lease_override, dict):
        merged = dict(body["lease"])
        merged.update(lease_override)
        body["lease"] = merged
    elif lease_override is not None:
        body["lease"] = lease_override
    return body


def _admit(payload: dict, **kwargs: object):
    kwargs.setdefault("catalog_ids", CATALOG_IDS)
    kwargs.setdefault("requested_model", "nemotron-3-nano:4b")
    kwargs.setdefault("now_s", NOW_S)
    return admit_cluster_recommendation(payload, **kwargs)


@pytest.fixture(autouse=True)
def _clear(monkeypatch):
    monkeypatch.delenv("MS4_FOREGROUND_MODEL", raising=False)
    monkeypatch.delenv("MS4_DEFAULT_MODEL", raising=False)
    monkeypatch.setenv("MS4_FACE_PROFILE", "latency")
    monkeypatch.setattr(hivemind_tools, "models_recommend", lambda *_a, **_k: {})
    model_picker._clear_cache_for_tests()
    yield
    model_picker._clear_cache_for_tests()


def _patch_catalog(monkeypatch, catalog):
    monkeypatch.setattr(model_picker, "_http_get_models", lambda *_a, **_k: catalog)


def _patch_recommend(monkeypatch, payload):
    monkeypatch.setattr(
        hivemind_tools, "models_recommend", lambda *_a, **_k: payload
    )
    monkeypatch.setattr(
        "machine_spirit_4.double_agent.recommend_lease.fetch_typed_recommend_document",
        lambda *_a, **_k: payload,
    )


def test_openai_strict_catalog_row_is_non_ready_for_exact_admission():
    entry = model_picker._normalize_model_entry(_openai_row("nemotron-3-nano:4b"))
    assert entry["id"] == "nemotron-3-nano:4b"
    assert entry["loaded"] is False
    with pytest.raises(ForegroundModelUnavailable, match="admission"):
        _admit({})


def test_absent_private_extensions_are_not_exact_warm_authority():
    entry = model_picker._normalize_model_entry({"id": "llama3.1:8b"})
    assert entry["loaded"] is False
    installed = model_picker._normalize_model_entry(
        {
            "id": "llama3.1:8b",
            "hivemind_status": "installed",
            "hivemind_reachable": True,
        }
    )
    assert installed["loaded"] is True
    with pytest.raises(ForegroundModelUnavailable, match="admission"):
        _admit(
            {
                "capability": "chat",
                "recommended_model": "llama3.1:8b",
                "backend": "ollama",
            },
            requested_model="llama3.1:8b",
        )


def test_strict_bare_models_without_recommendation_fail_closed_exact_gate(monkeypatch):
    catalog = [_openai_row("nemotron-3-nano:4b"), _openai_row("llama3.1:8b")]
    _patch_catalog(monkeypatch, catalog)
    _patch_recommend(monkeypatch, {})
    with pytest.raises(ForegroundModelUnavailable, match="admission"):
        require_exact_face_cluster_admission(
            hivemind_url="http://hive",
            requested_model="nemotron-3-nano:4b",
            now_s=NOW_S,
        )


def test_strict_bare_models_plus_admitted_recommendation(monkeypatch):
    catalog = [_openai_row("nemotron-3-nano:4b"), _openai_row("llama3.1:8b")]
    admitted = _admit(_admitted())
    assert admitted["model"] == "nemotron-3-nano:4b"
    assert admitted["source"] == "admitted"
    assert admitted["backend"] == "ollama"
    assert admitted["node"] == "node-a"
    assert admitted["owner"] == "node-a"
    assert admitted["schema"] == EXACT_FACE_CLUSTER_ADMISSION_SCHEMA
    _patch_catalog(monkeypatch, catalog)
    _patch_recommend(monkeypatch, _admitted())
    live = require_exact_face_cluster_admission(
        hivemind_url="http://hive",
        requested_model="nemotron-3-nano:4b",
        now_s=NOW_S,
    )
    assert live["source"] == "admitted"
    assert "ollama" in live["backend"]
    assert live["node"] == "node-a"


def test_stale_recommendation_is_rejected():
    with pytest.raises(ForegroundModelUnavailable, match="stale"):
        _admit(_admitted(snapshot_age_s=EXACT_FACE_CLUSTER_ADMISSION_TTL_S + 1))


def test_unready_recommendation_is_rejected():
    with pytest.raises(ForegroundModelUnavailable, match="unready"):
        _admit(_admitted(ready=False))


def test_mismatched_recommendation_is_rejected():
    with pytest.raises(ForegroundModelUnavailable, match="mismatch"):
        _admit(
            _admitted(recommended_model="llama3.1:8b"),
            catalog_ids={"nemotron-3-nano:4b"},
        )


def test_incomplete_scalar_recommend_is_not_exact_admission(monkeypatch):
    scalar = {
        "capability": "chat",
        "recommended_model": "nemotron-3-nano:4b",
        "backend": "ollama",
    }
    with pytest.raises(ForegroundModelUnavailable, match="admission"):
        _admit(scalar)
    _patch_catalog(
        monkeypatch,
        [
            {
                "id": "nemotron-3-nano:4b",
                "hivemind_status": "installed",
                "hivemind_reachable": True,
            }
        ],
    )
    _patch_recommend(monkeypatch, scalar)
    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)
    assert choice.model_id == "nemotron-3-nano:4b"
    assert choice.source == "loaded"


def test_wrong_schema_is_rejected():
    with pytest.raises(ForegroundModelUnavailable, match="schema"):
        _admit(_admitted(schema="Wrong.v9"))


def test_negative_snapshot_age_is_rejected():
    with pytest.raises(ForegroundModelUnavailable, match="freshness"):
        _admit(_admitted(snapshot_age_s=-1))


def test_nan_snapshot_age_is_rejected():
    with pytest.raises(ForegroundModelUnavailable, match="freshness"):
        _admit(_admitted(snapshot_age_s=float("nan")))


def test_bare_generation_without_current_generation_is_rejected():
    payload = _admitted()
    payload.pop("snapshot_age_s")
    payload.pop("lease")
    payload.pop("current_generation")
    payload["generation"] = 1
    with pytest.raises(ForegroundModelUnavailable, match="admission"):
        _admit(payload)


def test_expired_lease_is_rejected():
    with pytest.raises(ForegroundModelUnavailable, match="expired"):
        _admit(_admitted(lease={"expires": NOW_S - 1}))


def test_wrong_owner_is_rejected():
    with pytest.raises(ForegroundModelUnavailable, match="mismatch"):
        _admit(_admitted(owner="node-b"), expected_owner="node-a")


def test_wrong_endpoint_is_rejected():
    with pytest.raises(ForegroundModelUnavailable, match="mismatch"):
        _admit(
            _admitted(endpoint="http://node-b:11434/v1/chat/completions"),
            expected_endpoint="http://node-a:11434/v1/chat/completions",
        )


def test_ttl_age_boundaries_are_exact():
    ttl = EXACT_FACE_CLUSTER_ADMISSION_TTL_S
    assert _admit(_admitted(snapshot_age_s=0))["source"] == "admitted"
    assert _admit(_admitted(snapshot_age_s=ttl))["source"] == "admitted"
    with pytest.raises(ForegroundModelUnavailable, match="stale"):
        _admit(_admitted(snapshot_age_s=ttl + 1))
    inf_age = _admitted(snapshot_age_s=math.inf)
    with pytest.raises(ForegroundModelUnavailable, match="freshness"):
        _admit(inf_age)
