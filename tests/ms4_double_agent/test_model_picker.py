"""Face Lobe model picker tests."""

from __future__ import annotations

import pytest
import decimal
import random
import re
from decimal import Decimal, Context, Overflow, Subnormal
from fractions import Fraction

from machine_spirit_4.double_agent import (
    DEFAULT_FOREGROUND_MODEL,
    MS4_FOREGROUND_MODEL_ENV,
    choose_foreground_model,
)
from machine_spirit_4.double_agent import model_picker
from machine_spirit_4.gateway import hivemind_tools


@pytest.fixture(autouse=True)
def _clear_cache(monkeypatch):
    # Keep picker tests hermetic by default. Recommendation-specific tests
    # explicitly replace this local no-result seam with their intended payload.
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    monkeypatch.delenv("MS4_DEFAULT_MODEL", raising=False)
    # Legacy picker cases exercise the explicit latency profile. Dedicated
    # high-memory tests below delete this override to prove product-default auto.
    monkeypatch.setenv("MS4_FACE_PROFILE", "latency")
    monkeypatch.setattr(hivemind_tools, "models_recommend", lambda *_a, **_k: {})
    model_picker._clear_cache_for_tests()
    yield
    model_picker._clear_cache_for_tests()


def _patch_catalog(monkeypatch, catalog):
    def _fake_get(_url, timeout=5):
        return catalog
    monkeypatch.setattr(model_picker, "_http_get_models", _fake_get)


def test_high_memory_quality_profile_selects_installed_qwen36_without_pull(monkeypatch):
    monkeypatch.delenv("MS4_FACE_PROFILE", raising=False)
    catalog = [
        {
            "id": "qwen3.6:35b",
            "hivemind_status": "installed",
            "hivemind_reachable": True,
            "hivemind_node_id": "node-high-memory",
            "hivemind_reachable_nodes": ["192.0.2.35"],
        },
        {
            "id": "nemotron-3-nano:4b",
            "hivemind_status": "installed",
            "hivemind_reachable": True,
        },
    ]
    _patch_catalog(monkeypatch, catalog)
    monkeypatch.setattr(
        model_picker,
        "_http_get_carrier_nodes",
        lambda *_a, **_k: [
            {
                "node_id": "node-high-memory",
                "lan_ip": "192.0.2.35",
                "gpus": [
                    {
                        "name": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
                        "vram_total_mb": 97_887,
                    }
                ],
            }
        ],
    )

    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)

    assert choice.model_id == "qwen3.6:35b"
    assert choice.source == "loaded"
    assert "max_vram_mib=97887" in choice.detail


def test_high_memory_quality_profile_keeps_latency_model_on_16gib_tier(monkeypatch):
    monkeypatch.delenv("MS4_FACE_PROFILE", raising=False)
    catalog = [
        {
            "id": "qwen3.6:35b",
            "hivemind_status": "installed",
            "hivemind_reachable": True,
            "hivemind_node_id": "node-16g",
        },
        {
            "id": "nemotron-3-nano:4b",
            "hivemind_status": "installed",
            "hivemind_reachable": True,
        },
    ]
    _patch_catalog(monkeypatch, catalog)
    monkeypatch.setattr(
        model_picker,
        "_http_get_carrier_nodes",
        lambda *_a, **_k: [
            {
                "node_id": "node-16g",
                "gpus": [{"name": "16 GiB tier", "vram_total_mb": 16_384}],
            }
        ],
    )

    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)

    assert choice.model_id == "nemotron-3-nano:4b"


def test_high_memory_auto_profile_fails_soft_when_capacity_is_unknown(monkeypatch):
    monkeypatch.delenv("MS4_FACE_PROFILE", raising=False)
    catalog = [
        {
            "id": "qwen3.6:35b",
            "hivemind_status": "installed",
            "hivemind_reachable": True,
            "hivemind_node_id": "node-unknown",
        },
        {
            "id": "nemotron-3-nano:4b",
            "hivemind_status": "installed",
            "hivemind_reachable": True,
        },
    ]
    _patch_catalog(monkeypatch, catalog)

    def capacity_unavailable(*_args, **_kwargs):
        raise RuntimeError("carrier telemetry unavailable")

    monkeypatch.setattr(model_picker, "_http_get_carrier_nodes", capacity_unavailable)

    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)

    assert choice.model_id == "nemotron-3-nano:4b"


def _patch_recommendation(monkeypatch, payload):
    monkeypatch.setattr(
        hivemind_tools,
        "models_recommend",
        lambda *_args, **_kwargs: payload,
    )


def test_env_override_beats_catalog(monkeypatch):
    monkeypatch.setenv(MS4_FOREGROUND_MODEL_ENV, "custom-tiny:0.5b")
    _patch_catalog(monkeypatch, [{"id": "qwen2.5:0.5b", "loaded": True, "available": True}])
    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)
    assert choice.model_id == "custom-tiny:0.5b"
    assert choice.source == "override"


def test_picker_prefers_cluster_served_nemotron_4b_for_face(monkeypatch):
    _patch_catalog(monkeypatch, [
        {
            "id": "nemotron-3-nano:4b",
            "hivemind_status": "installed",
            "hivemind_reachable": True,
            "hivemind_host": "remote-5090-node",
        },
        {"id": "qwen3:8b", "loaded": True, "available": True},
        {"id": "llama3.1:8b", "loaded": True, "available": True},
    ])

    choice = choose_foreground_model(hivemind_url="http://coordinator", force_refresh=True)

    assert DEFAULT_FOREGROUND_MODEL == "nemotron-3-nano:4b"
    assert choice.model_id == "nemotron-3-nano:4b"
    assert choice.source == "loaded"
    assert model_picker._satisfies_foreground_contract(choice.model_id) is True


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


def test_picker_prefers_8b_instruct_when_loaded(monkeypatch):
    """7B-8B instruct models are now #1 priority (May 26 2026): live
    evidence showed phi4-mini can't reliably follow the Face Lobe
    anti-hallucination contract (denied a dispatched job, fabricated
    fake `hivemind` Python APIs). The picker now prefers a coherent
    8B instruct first; phi4-mini stays in the fallback chain."""
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    _patch_catalog(monkeypatch, [
        {"id": "phi4-mini", "loaded": True, "available": True},
        {"id": "llama3.1:8b", "loaded": True, "available": True},
        {"id": "qwen3:8b", "loaded": True, "available": True},
        {"id": "gemma4", "loaded": True, "available": True},
    ])
    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)
    # qwen3:8b is now priority #1 in FOREGROUND_PRIORITY_PATTERNS
    assert choice.model_id == "qwen3:8b"
    assert choice.source == "loaded"


def test_picker_never_picks_dumb_tiny_models_for_foreground(monkeypatch):
    """qwen2.5:0.5b and tinyllama:1.1b are deliberately NOT in the
    priority list. If they're the only `loaded` entries available, the
    picker must fail fast rather than cold-select a higher-quality model."""
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    _patch_catalog(monkeypatch, [
        {"id": "qwen2.5:0.5b", "loaded": True, "available": True},
        {"id": "tinyllama:1.1b", "loaded": True, "available": True},
        {"id": "phi3.5", "loaded": False, "available": True},
    ])
    with pytest.raises(RuntimeError, match="no loaded, reachable, eligible"):
        choose_foreground_model(hivemind_url="http://hive", force_refresh=True)


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


def test_picker_rejects_available_only_foreground_candidates(monkeypatch):
    """Available-only candidates require a cold load and must fail fast."""
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    # May 26 2026 picker reorder: 7-8B instruct models come first,
    # phi4-mini moved down because it can't follow the Face Lobe
    # anti-hallucination contract reliably. Pin the same set of
    # candidates the picker would actually see plus include qwen3:8b
    # (the new #1).
    _patch_catalog(monkeypatch, [
        {"id": "qwen3:8b", "loaded": False, "available": True},
        {"id": "phi4-mini", "loaded": False, "available": True},
        {"id": "gemma3", "loaded": False, "available": True},
        {"id": "qwen3-coder-next:latest", "loaded": False, "available": True},
    ])
    with pytest.raises(RuntimeError, match="no loaded, reachable, eligible"):
        choose_foreground_model(hivemind_url="http://hive", force_refresh=True)


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
    # R2.4: MS4_DEFAULT_MODEL is a GATED fallback. A gate-VALID default is
    # honored; off-gate defaults fall back to the built-in llama3.1:8b (see
    # test_p4_ms4_default_model_offgate_*).
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    monkeypatch.setenv("MS4_DEFAULT_MODEL", "qwen3:8b")
    _patch_catalog(monkeypatch, [
        {"id": "exotic-model:42b", "loaded": True, "available": True},
    ])
    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)
    assert choice.model_id == "qwen3:8b"
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


@pytest.mark.parametrize(
    "payload",
    [
        {
            "recommended": [
                {"model_id": "opaque-recommendation-a", "hivemind_category": "tts"}
            ]
        },
        {
            "recommended": [
                {"model_id": "opaque-recommendation-b", "category": "embedding"}
            ]
        },
        {
            "recommended": [
                {
                    "model_id": "opaque-recommendation-c",
                    "hivemind_capabilities": {"image-generation": True},
                }
            ]
        },
        {
            "recommended": [
                {
                    "model_id": "opaque-recommendation-d",
                    "category": "tts",
                    "capabilities": ["chat"],
                }
            ]
        },
        {"recommended": [None, {"score": 1.0}]},
        {"recommended_model": "opaque-current-chat", "backend": "ollama"},
        {"capability": "chat", "recommended_model": "opaque-current-chat"},
        {
            "capability": "chat",
            "recommended_model": {"id": "opaque-current-chat"},
            "backend": "ollama",
        },
    ],
    ids=(
        "tts",
        "embedding",
        "image",
        "conflicting-category",
        "malformed-legacy",
        "malformed-current-capability",
        "malformed-current-backend",
        "malformed-current-model",
    ),
)
def test_choose_foreground_rejects_unsafe_structured_recommendations(
    monkeypatch, payload
):
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    _patch_recommendation(monkeypatch, payload)
    _patch_catalog(
        monkeypatch,
        [
                {
                    # R2.4 reconcile: the positive-allow gate requires a CLEAN
                    # whole-id, so the warm catalog stand-in is a real exact-safe
                    # id now (the prior "qwen3:8b-valid-catalog" synthetic id was
                    # only a P3 substring match and no longer admits).
                    "id": "qwen3:8b",
                    "hivemind_status": "installed",
                    "category": "llm",
                    "capabilities": ["chat"],
                }
        ],
    )

    choice = choose_foreground_model(
        hivemind_url="http://hive",
        force_refresh=True,
    )

    assert choice.model_id == "qwen3:8b"
    assert choice.source == "loaded"


def test_choose_foreground_reconciles_legacy_recommendation_through_ready_policy(
    monkeypatch,
):
    """Legacy recommendations are advisory until the Face policy reconciles
    them against the live readiness catalog."""
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    _patch_recommendation(
        monkeypatch,
        {
            "recommended": [
                {
                    "model_id": "qwen3:8b",
                    "category": "llm",
                    "capabilities": ["chat"],
                    "score": 1.0,
                }
            ]
        },
    )
    _patch_catalog(
        monkeypatch,
        [{"id": "qwen3:8b", "hivemind_status": "installed", "hivemind_reachable": True}],
    )

    choice = choose_foreground_model(
        hivemind_url="http://hive",
        force_refresh=True,
    )

    assert choice.model_id == "qwen3:8b"
    assert choice.source == "loaded"


def test_ready_nemotron_face_policy_beats_generic_llama_recommendation(monkeypatch):
    """Regression: recommend@v1 currently has only a generic chat contract.

    A ready lower-priority Llama recommendation must not override MS4's
    Face-role ordering when the preferred Nemotron 4B candidate is also ready
    somewhere in the HiveMind cluster.
    """
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    _patch_recommendation(
        monkeypatch,
        {
            "capability": "chat",
            "recommended_model": "llama3.1:8b",
            "backend": "ollama",
        },
    )
    _patch_catalog(
        monkeypatch,
        [
            {
                "id": "llama3.1:8b",
                "hivemind_status": "installed",
                "hivemind_reachable": True,
            },
            {
                "id": "nemotron-3-nano:4b",
                "hivemind_status": "installed",
                "hivemind_reachable": True,
                "hivemind_host": "remote-5090-node",
            },
        ],
    )

    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)

    assert choice.model_id == "nemotron-3-nano:4b"
    assert choice.source == "loaded"


def test_choose_foreground_reconciles_offcontract_scalar_recommendation(monkeypatch):
    """The current scalar recommendation form (top-level ``recommended_model``
    + ``backend``) is still parsed and the recommend seam is still called with
    ``capability="chat"`` (recommend-call contract preserved). But when the
    recommended model is OFF-contract (not in ``FOREGROUND_PRIORITY_PATTERNS``)
    and a suitable small/warm candidate is available in the catalog, the Face
    Lobe defers to the local picker (catalog IS consulted) instead of consuming
    the recommendation verbatim."""
    observed = {}

    def recommend(url, **kwargs):
        observed.update({"url": url, "kwargs": kwargs})
        return {
            "capability": "chat",
            "quality": "balanced",
            "recommended_model": "opaque-current-chat",
            "backend": "ollama",
        }

    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    monkeypatch.setattr(hivemind_tools, "models_recommend", recommend)
    _patch_catalog(
        monkeypatch,
        [
            {
                "id": "llama3.1:8b",
                "hivemind_status": "installed",
                "hivemind_reachable": True,
            },
        ],
    )

    choice = choose_foreground_model(
        hivemind_url="http://hive",
        force_refresh=True,
    )

    assert observed == {"url": "http://hive", "kwargs": {"capability": "chat"}}
    assert choice.model_id == "llama3.1:8b"
    assert choice.source == "loaded"


def test_choose_foreground_iterates_to_contract_satisfying_legacy_recommendation(
    monkeypatch,
):
    """``_try_hivemind_recommend`` iterates the legacy list, skipping the
    non-chat first entry, and returns the second (chat) entry. When that
    winning recommendation satisfies the Face-Lobe safety gate it is still
    reconciled through the ready policy catalog; non-chat entries are skipped."""
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    _patch_recommendation(
        monkeypatch,
        {
            "recommended": [
                {"model_id": "opaque-non-chat", "type": "tts"},
                {
                    "model_id": "llama3.1:8b",
                    "category": "llm",
                    "capabilities": ["chat"],
                },
            ]
        },
    )
    _patch_catalog(
        monkeypatch,
        [{"id": "llama3.1:8b", "hivemind_status": "installed", "hivemind_reachable": True}],
    )

    choice = choose_foreground_model(
        hivemind_url="http://hive",
        force_refresh=True,
    )

    assert choice.model_id == "llama3.1:8b"
    assert choice.source == "loaded"


def test_choose_foreground_reconciles_offcontract_heavy_recommendation(monkeypatch):
    """R1 Face-Lobe small/responsive contract (artifact §16).

    With NO explicit pin and NO ``MS4_FOREGROUND_MODEL``, when the
    HiveMind recommend seam returns an unbounded HEAVY general model
    (``nemotron:latest`` -- deliberately NOT in
    ``FOREGROUND_PRIORITY_PATTERNS``) *while a small/warm foreground
    candidate is present in the catalog* (``llama3.1:8b``, installed),
    the Face Lobe must NOT consume the heavy recommendation verbatim.
    It must honor the documented small/responsive contract and resolve
    to the small/warm catalog candidate instead.

    The reconcile has since landed: ``choose_foreground_model`` calls the
    recommend seam first but defers an off-contract result to the local
    fast/small/warm picker. This test guards that behavior -- it must
    resolve to the small/warm ``llama3.1:8b`` rather than the heavy
    ``nemotron:latest``.
    """
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    monkeypatch.delenv("MS4_DEFAULT_MODEL", raising=False)
    _patch_recommendation(
        monkeypatch,
        {
            "capability": "chat",
            "recommended_model": "nemotron:latest",
            "backend": "ollama",
        },
    )
    _patch_catalog(
        monkeypatch,
        [
            {
                "id": "nemotron:latest",
                "hivemind_status": "installed",
                "hivemind_reachable": True,
            },
            {
                "id": "llama3.1:8b",
                "hivemind_status": "installed",
                "hivemind_reachable": True,
            },
        ],
    )

    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)

    assert choice.model_id != "nemotron:latest", (
        "Face Lobe must not consume the heavy/unbounded cluster "
        f"recommendation verbatim; got {choice.model_id!r}"
    )
    assert choice.model_id == "llama3.1:8b", (
        "Face Lobe must honor the small/responsive contract and resolve "
        f"to the small/warm catalog candidate; got {choice.model_id!r}"
    )


# ---------------------------------------------------------------------------
# R2 hardening: explicit heavy-size guard (Gap A) + fail-closed reconcile
# (Gap B), plus reconcile coverage (Gap C).
# ---------------------------------------------------------------------------


def test_satisfies_foreground_contract_rejects_heavy_size_variants():
    """Gap A: the broad family-substring contract must NOT admit explicit
    HEAVY size variants. Small family members stay valid; 27B/70B/235B (and
    the clean total-param MoE form) are rejected by the parameter-size guard.
    Expert-product MoE notation like ``8x7b`` is a documented non-goal."""
    # Small members (<= foreground size ceiling, or no explicit size) satisfy.
    assert model_picker._satisfies_foreground_contract("llama3.1:8b") is True
    assert model_picker._satisfies_foreground_contract("qwen3:8b") is True
    assert model_picker._satisfies_foreground_contract("qwen2.5:7b") is True
    assert model_picker._satisfies_foreground_contract("phi4-mini") is True
    # R2.4 supersedes P3: a bare approved-family with NO explicit size now fails
    # closed (positive-allow); an explicit small size on a grammar family admits.
    assert model_picker._satisfies_foreground_contract("gemma3:8b") is True
    assert model_picker._satisfies_foreground_contract("gemma3") is False
    # Explicit heavy variants of an otherwise-small family are rejected.
    assert model_picker._satisfies_foreground_contract("llama3.1:70b") is False
    assert model_picker._satisfies_foreground_contract("qwen3:235b") is False
    assert model_picker._satisfies_foreground_contract("gemma3:27b") is False
    # Clean total-param MoE form (total billions) is rejected on total size.
    assert model_picker._satisfies_foreground_contract("qwen3:30b-a3b") is False


@pytest.mark.parametrize(
    "heavy_id",
    ["llama3.1:70b", "qwen3:235b", "gemma3:27b", "qwen3:30b-a3b"],
    ids=["llama-70b", "qwen-235b", "gemma-27b", "qwen-moe-30b-a3b"],
)
def test_choose_foreground_reconciles_heavy_family_variant_to_small(monkeypatch, heavy_id):
    """Gap A: a recommend result that is an explicit HEAVY family variant
    (matches a FOREGROUND_PRIORITY_PATTERNS family but exceeds the foreground
    size ceiling) must reconcile to the warm small candidate, not be consumed
    verbatim."""
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    monkeypatch.delenv("MS4_DEFAULT_MODEL", raising=False)
    _patch_recommendation(
        monkeypatch,
        {"capability": "chat", "recommended_model": heavy_id, "backend": "ollama"},
    )
    _patch_catalog(
        monkeypatch,
        [
            {"id": heavy_id, "hivemind_status": "installed", "hivemind_reachable": True},
            {"id": "llama3.1:8b", "hivemind_status": "installed", "hivemind_reachable": True},
        ],
    )

    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)

    assert choice.model_id != heavy_id, (
        f"heavy family variant {heavy_id!r} must not be consumed verbatim"
    )
    assert choice.model_id == "llama3.1:8b"


def test_choose_foreground_offcontract_rec_fails_closed_on_catalog_error(monkeypatch):
    """Gap B: an off-contract recommendation must fail CLOSED. If the catalog
    lookup RAISES, the safe fallback/default is used -- never the off-contract
    recommendation."""
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    monkeypatch.delenv("MS4_DEFAULT_MODEL", raising=False)
    _patch_recommendation(
        monkeypatch,
        {"capability": "chat", "recommended_model": "nemotron:latest", "backend": "ollama"},
    )

    def _boom(_url, timeout=5):
        raise RuntimeError("catalog down")

    monkeypatch.setattr(model_picker, "_http_get_models", _boom)

    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)

    assert choice.model_id != "nemotron:latest", (
        "off-contract recommendation must not survive a catalog error"
    )
    assert choice.model_id == DEFAULT_FOREGROUND_MODEL


def test_choose_foreground_offcontract_rec_fails_closed_on_no_catalog_match(monkeypatch):
    """Gap B: an off-contract recommendation must fail CLOSED. If the catalog
    has NO foreground-priority candidate, the safe fallback/default is used --
    never the off-contract recommendation."""
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    monkeypatch.delenv("MS4_DEFAULT_MODEL", raising=False)
    _patch_recommendation(
        monkeypatch,
        {"capability": "chat", "recommended_model": "nemotron:latest", "backend": "ollama"},
    )
    _patch_catalog(
        monkeypatch,
        [
            {
                "id": "some-other-heavy:400b",
                "hivemind_status": "installed",
                "hivemind_reachable": True,
            }
        ],
    )

    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)

    assert choice.model_id != "nemotron:latest", (
        "off-contract recommendation must not survive an empty foreground catalog"
    )
    assert choice.model_id == DEFAULT_FOREGROUND_MODEL


def test_choose_foreground_reconciles_offcontract_legacy_recommendation(monkeypatch):
    """Gap C coverage: the legacy list recommendation form is also reconciled
    -- an off-contract chat entry with a warm small candidate available
    resolves to the candidate (catalog consulted)."""
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    _patch_recommendation(
        monkeypatch,
        {
            "recommended": [
                {"model_id": "opaque-legacy-chat", "category": "llm", "capabilities": ["chat"]}
            ]
        },
    )
    _patch_catalog(
        monkeypatch,
        [{"id": "llama3.1:8b", "hivemind_status": "installed", "hivemind_reachable": True}],
    )

    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)

    assert choice.model_id == "llama3.1:8b"
    assert choice.source == "loaded"


def test_choose_foreground_reconciles_unreachable_scalar_recommendation(monkeypatch):
    """A scalar recommendation requires exact loaded/reachable catalog proof."""
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    _patch_recommendation(
        monkeypatch,
        {"capability": "chat", "recommended_model": "qwen3:8b", "backend": "ollama"},
    )
    observed = {"catalog_calls": 0}

    def catalog(*_args, **_kwargs):
        observed["catalog_calls"] += 1
        return [
            {"id": "qwen3:8b", "hivemind_status": "installed", "hivemind_reachable": False},
            {"id": "llama3.1:8b", "hivemind_status": "installed", "hivemind_reachable": True},
        ]

    monkeypatch.setattr(model_picker, "_http_get_models", catalog)

    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)

    assert observed["catalog_calls"] == 1
    assert choice.model_id == "llama3.1:8b"
    assert choice.source == "loaded"


def test_choose_foreground_reconciles_legacy_loaded_unreachable_scalar_recommendation(
    monkeypatch,
):
    """Legacy loaded/available booleans cannot override unreachable=false."""
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    _patch_recommendation(
        monkeypatch,
        {"capability": "chat", "recommended_model": "qwen3:8b", "backend": "ollama"},
    )
    observed = {"catalog_calls": 0}

    def catalog(*_args, **_kwargs):
        observed["catalog_calls"] += 1
        return [
            {
                "id": "qwen3:8b",
                "loaded": True,
                "available": True,
                "hivemind_reachable": False,
            },
            {
                "id": "llama3.1:8b",
                "loaded": True,
                "available": True,
                "hivemind_reachable": True,
            },
        ]

    monkeypatch.setattr(model_picker, "_http_get_models", catalog)

    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)

    assert observed["catalog_calls"] == 1
    assert choice.model_id == "llama3.1:8b"
    assert choice.source == "loaded"


def test_explicit_override_short_circuits_recommend_seam(monkeypatch):
    """Gap C coverage: an explicit MS4_FOREGROUND_MODEL override wins outright
    and short-circuits BOTH the recommend seam and the catalog."""
    monkeypatch.setenv(MS4_FOREGROUND_MODEL_ENV, "operator-choice:latest")
    monkeypatch.setattr(
        hivemind_tools,
        "models_recommend",
        lambda *_a, **_k: pytest.fail("override must short-circuit the recommend seam"),
    )
    monkeypatch.setattr(
        model_picker,
        "_http_get_models",
        lambda *_a, **_k: pytest.fail("override must not consult the catalog"),
    )

    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)

    assert choice.model_id == "operator-choice:latest"
    assert choice.source == "override"


# ---------------------------------------------------------------------------
# R2.1 hardening: size-gate the CATALOG picker (an over-ceiling family variant
# that is the only match must not be selected on EITHER the off-contract
# reconcile path or the ordinary no-recommend path) + close the expert-product
# MoE (`<experts>x<size>b`) bypass.
# ---------------------------------------------------------------------------


def test_choose_foreground_offcontract_rec_fails_closed_when_only_heavy_catalog_match(monkeypatch):
    """R2.1: catalog contains ONLY an over-ceiling family variant
    (``qwen3:235b``). An off-contract recommendation must NOT be salvaged into
    that heavy variant (nor kept verbatim) -- it must fall to the safe default."""
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    monkeypatch.delenv("MS4_DEFAULT_MODEL", raising=False)
    _patch_recommendation(
        monkeypatch,
        {"capability": "chat", "recommended_model": "nemotron:latest", "backend": "ollama"},
    )
    _patch_catalog(
        monkeypatch,
        [{"id": "qwen3:235b", "hivemind_status": "installed", "hivemind_reachable": True}],
    )

    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)

    assert choice.model_id != "qwen3:235b", (
        "the catalog picker must not return an over-ceiling family variant"
    )
    assert choice.model_id != "nemotron:latest", "off-contract rec must not survive"
    assert choice.model_id == DEFAULT_FOREGROUND_MODEL


def test_choose_foreground_no_rec_fails_closed_when_only_heavy_catalog_match(monkeypatch):
    """R2.1: with no usable recommendation, the ordinary catalog path must not
    select an over-ceiling family variant (``qwen3:235b``); it falls to the
    safe default."""
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    monkeypatch.delenv("MS4_DEFAULT_MODEL", raising=False)
    _patch_recommendation(monkeypatch, {})  # no usable recommendation -> None
    _patch_catalog(
        monkeypatch,
        [{"id": "qwen3:235b", "hivemind_status": "installed", "hivemind_reachable": True}],
    )

    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)

    assert choice.model_id != "qwen3:235b", (
        "the no-recommend catalog path must not select an over-ceiling variant"
    )
    assert choice.model_id == DEFAULT_FOREGROUND_MODEL


def test_moe_expert_product_size_is_total_and_rejected():
    """R2.1: expert-product MoE notation ``<experts>x<size>b`` parses to TOTAL
    billions (``8x7b`` -> 56) and is rejected by the size ceiling. Plain
    version fragments stay UNPARSED so small explicit sizes remain accepted,
    and the total-param MoE form is unchanged."""
    assert model_picker._parse_param_size_b("qwen3:8x7b") == 56.0
    assert model_picker._parse_param_size_b("mixtral:8x7b") == 56.0
    assert model_picker._satisfies_foreground_contract("qwen3:8x7b") is False
    assert model_picker._satisfies_foreground_contract("mixtral:8x7b") is False
    # Version fragments must NOT be misread as sizes; small sizes stay accepted.
    assert model_picker._parse_param_size_b("llama3.1:8b") == 8.0
    assert model_picker._parse_param_size_b("qwen2.5:7b") == 7.0
    assert model_picker._satisfies_foreground_contract("llama3.1:8b") is True
    assert model_picker._satisfies_foreground_contract("qwen2.5:7b") is True
    # R2.3-P3: "30b-a3b" is now INVALID (dangling "a3b" island, not a full-match)
    # -> parse None; the CONTRACT OUTCOME (reject) is preserved.
    assert model_picker._parse_param_size_b("qwen3:30b-a3b") is None
    assert model_picker._satisfies_foreground_contract("qwen3:30b-a3b") is False


def test_choose_foreground_rejects_moe_expert_product_e2e(monkeypatch):
    """R2.1: an expert-product MoE recommendation (``qwen3:8x7b`` = 56B) with
    only itself in the catalog resolves to the safe default, never ``8x7b``."""
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    monkeypatch.delenv("MS4_DEFAULT_MODEL", raising=False)
    _patch_recommendation(
        monkeypatch,
        {"capability": "chat", "recommended_model": "qwen3:8x7b", "backend": "ollama"},
    )
    _patch_catalog(
        monkeypatch,
        [{"id": "qwen3:8x7b", "hivemind_status": "installed", "hivemind_reachable": True}],
    )

    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)

    assert choice.model_id != "qwen3:8x7b", (
        "expert-product MoE variant must not be consumed verbatim or picked"
    )
    assert choice.model_id == DEFAULT_FOREGROUND_MODEL


# ---------------------------------------------------------------------------
# R2.1-P2: the size-token RIGHT boundary was too strict -- a "<n>b" tag
# followed by anything other than end/':'/'-'/'_' (e.g. ".gguf", "@sha256:",
# "/it", an appended quant like "q4_K_M") parsed to None and was falsely
# admitted as small. Recognize "<n>b" when followed by end OR any non-digit.
# ---------------------------------------------------------------------------


def test_size_token_right_boundary_catches_suffixed_variants():
    """R2.1-P2 unit: heavy ``<n>b`` tokens followed by a non-delimiter suffix
    must still parse to their real billions and be rejected; small sizes and
    version fragments are unaffected."""
    # R2.3-P3: glued suffixes make "70b.gguf" / "235bq4" non-full-match islands
    # -> INVALID -> parse None; separator-delimited "235b@..." / "27b/it" stay SIZE.
    # The CONTRACT OUTCOME (reject) is preserved for all four (asserted below).
    assert model_picker._parse_param_size_b("llama3.1:70b.gguf") is None
    assert model_picker._parse_param_size_b("qwen3:235b@sha256:deadbeef") == 235.0
    assert model_picker._parse_param_size_b("gemma3:27b/it") == 27.0
    assert model_picker._parse_param_size_b("qwen3:235bq4_K_M") is None
    assert model_picker._satisfies_foreground_contract("llama3.1:70b.gguf") is False
    assert model_picker._satisfies_foreground_contract("qwen3:235b@sha256:deadbeef") is False
    assert model_picker._satisfies_foreground_contract("gemma3:27b/it") is False
    assert model_picker._satisfies_foreground_contract("qwen3:235bq4_K_M") is False
    # Small sizes / version fragments must remain correctly parsed + accepted.
    assert model_picker._parse_param_size_b("llama3.1:8b") == 8.0
    assert model_picker._parse_param_size_b("qwen2.5:7b") == 7.0
    # R2.3-P3: "30b-a3b" now INVALID (dangling a3b island) -> None; reject preserved.
    assert model_picker._parse_param_size_b("qwen3:30b-a3b") is None
    assert model_picker._satisfies_foreground_contract("qwen3:30b-a3b") is False
    assert model_picker._parse_param_size_b("qwen3:8x7b") == 56.0
    assert model_picker._satisfies_foreground_contract("llama3.1:8b") is True
    assert model_picker._satisfies_foreground_contract("qwen2.5:7b") is True


def test_choose_foreground_rejects_suffixed_heavy_recommendation_e2e(monkeypatch):
    """R2.1-P2 e2e: a recommendation carrying a suffixed heavy tag
    (``llama3.1:70b.gguf``) must be rejected by size and reconcile to the warm
    small candidate, not consumed verbatim."""
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    monkeypatch.delenv("MS4_DEFAULT_MODEL", raising=False)
    _patch_recommendation(
        monkeypatch,
        {"capability": "chat", "recommended_model": "llama3.1:70b.gguf", "backend": "ollama"},
    )
    _patch_catalog(
        monkeypatch,
        [
            {"id": "llama3.1:70b.gguf", "hivemind_status": "installed", "hivemind_reachable": True},
            {"id": "llama3.1:8b", "hivemind_status": "installed", "hivemind_reachable": True},
        ],
    )

    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)

    assert choice.model_id != "llama3.1:70b.gguf", (
        "suffixed heavy recommendation must not be consumed verbatim"
    )
    assert choice.model_id == "llama3.1:8b"


def test_choose_foreground_catalog_only_suffixed_heavy_fails_closed(monkeypatch):
    """R2.1-P2 catalog: catalog contains ONLY a suffixed heavy variant
    (``qwen3:235b@sha256:...``) and there is no usable recommendation. The
    catalog picker must reject it on size and fall to the safe default."""
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    monkeypatch.delenv("MS4_DEFAULT_MODEL", raising=False)
    _patch_recommendation(monkeypatch, {})  # no usable recommendation -> None
    _patch_catalog(
        monkeypatch,
        [
            {
                "id": "qwen3:235b@sha256:deadbeef",
                "hivemind_status": "installed",
                "hivemind_reachable": True,
            }
        ],
    )

    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)

    assert choice.model_id != "qwen3:235b@sha256:deadbeef", (
        "suffixed heavy catalog-only variant must not be selected"
    )
    assert choice.model_id == DEFAULT_FOREGROUND_MODEL


# ---------------------------------------------------------------------------
# R2.2: the size parser must NOT depend on a delimiter whitelist on either
# side. Heavy variants whose size token is bounded by a non-[:\-_] LEFT
# separator ("/", ".", "@") or a trailing DIGIT (quant like "4bit") were still
# parsed to None and falsely admitted. Six representative forms, each rejected.
# ---------------------------------------------------------------------------

_R2_2_HEAVY_FORMS = [
    ("llama3.1/70b", 70.0),    # LEFT separator "/"
    ("llama3.1.70b", 70.0),    # LEFT separator "."
    ("llama3.1@70b", 70.0),    # LEFT separator "@"
    ("gemma3/27b", 27.0),      # LEFT separator "/"
    ("qwen3:235b4bit", 235.0), # RIGHT side is a digit (quant)
    ("qwen3:8x7b4bit", 56.0),  # MoE + RIGHT digit quant
]
_R2_2_IDS = ["llama-slash-70b", "llama-dot-70b", "llama-at-70b",
             "gemma-slash-27b", "qwen-235b4bit", "qwen-8x7b4bit"]


def test_anchor_free_size_parse_rejects_separator_and_quant_variants():
    """R2.2 unit: size tokens are found regardless of the surrounding separator
    (LEFT not in [:\\-_]) or a trailing DIGIT (quant). All six forms parse to
    their real billions and fail the contract; prior accepted vectors and
    unsized aliases are unaffected."""
    for form, _expected in _R2_2_HEAVY_FORMS:
        # R2.3-P3: separator-delimited forms stay SIZE while glued forms
        # ("llama3.1.70b", "235b4bit", "8x7b4bit") are now INVALID; the CONTRACT
        # OUTCOME (reject) is preserved for all six regardless of internal value.
        assert model_picker._satisfies_foreground_contract(form) is False, form
    # Prior accepted vectors unchanged.
    assert model_picker._parse_param_size_b("llama3.1:8b") == 8.0
    assert model_picker._parse_param_size_b("qwen2.5:7b") == 7.0
    # R2.3-P3: "30b-a3b" now INVALID (dangling a3b island) -> None; reject preserved.
    assert model_picker._parse_param_size_b("qwen3:30b-a3b") is None
    assert model_picker._satisfies_foreground_contract("qwen3:30b-a3b") is False
    assert model_picker._parse_param_size_b("qwen3:8x7b") == 56.0
    assert model_picker._satisfies_foreground_contract("llama3.1:8b") is True
    assert model_picker._satisfies_foreground_contract("qwen2.5:7b") is True
    # R2.4 supersedes P3: the classifier still leaves the unsized alias unparsed
    # (None), but the positive-allow gate now FAILS CLOSED on a mutable ":latest"
    # alias (not exact-safe, no explicit size) instead of admitting it.
    assert model_picker._parse_param_size_b("qwen3-coder-next:latest") is None
    assert model_picker._satisfies_foreground_contract("qwen3-coder-next:latest") is False


@pytest.mark.parametrize("heavy_id,_size", _R2_2_HEAVY_FORMS, ids=_R2_2_IDS)
def test_offcontract_rec_separator_quant_variant_fails_closed(monkeypatch, heavy_id, _size):
    """R2.2 e2e (reconcile path): an off-contract recommendation in a
    separator/quant form, with only itself in the catalog, resolves to the safe
    default -- never the heavy id."""
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    monkeypatch.delenv("MS4_DEFAULT_MODEL", raising=False)
    _patch_recommendation(
        monkeypatch,
        {"capability": "chat", "recommended_model": heavy_id, "backend": "ollama"},
    )
    _patch_catalog(
        monkeypatch,
        [{"id": heavy_id, "hivemind_status": "installed", "hivemind_reachable": True}],
    )

    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)

    assert choice.model_id != heavy_id, f"{heavy_id!r} must not be consumed verbatim"
    assert choice.model_id == DEFAULT_FOREGROUND_MODEL


@pytest.mark.parametrize("heavy_id,_size", _R2_2_HEAVY_FORMS, ids=_R2_2_IDS)
def test_no_rec_catalog_only_separator_quant_variant_fails_closed(monkeypatch, heavy_id, _size):
    """R2.2 e2e (no-recommend catalog path): with only a separator/quant heavy
    variant in the catalog and no usable recommendation, the catalog picker
    rejects it on size and falls to the safe default."""
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    monkeypatch.delenv("MS4_DEFAULT_MODEL", raising=False)
    _patch_recommendation(monkeypatch, {})  # no usable recommendation -> None
    _patch_catalog(
        monkeypatch,
        [{"id": heavy_id, "hivemind_status": "installed", "hivemind_reachable": True}],
    )

    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)

    assert choice.model_id != heavy_id, f"{heavy_id!r} must not be selected by the catalog"
    assert choice.model_id == DEFAULT_FOREGROUND_MODEL


# ---------------------------------------------------------------------------
# R2.3: exact-decimal, non-overlapping size parsing.
#   Defect 1 (precision): float() rounds "16.0000000000000001" -> 16.0, so the
#     boundary heavy was admitted. Must reject via exact Decimal compare.
#   Defect 2 (overlapping): the lookahead parsed "0.5b"->5 and "1.17b"->17,
#     false-rejecting ~1B models. Must tokenize non-overlapping.
# ---------------------------------------------------------------------------

_PRECISION_HEAVY = "qwen3:16.0000000000000001b"  # > 16 only under exact decimal


def test_size_parse_is_exact_decimal_and_nonoverlapping():
    """R2.3 unit: sizes are exact Decimals from a non-overlapping tokenizer --
    decimals are single tokens (never split), and all prior vectors hold."""
    P = model_picker._parse_param_size_b
    # non-overlapping: "0.5"/"1.17" stay whole (were 5 / 17 under overlapping)
    assert P("0.5b") == Decimal("0.5")
    assert P("1.17b") == Decimal("1.17")
    assert P("15.9999999999999999b") == Decimal("15.9999999999999999")
    assert P("16b") == Decimal("16")
    assert P("16.0000000000000000b") == Decimal("16.0000000000000000")
    # exact decimal: this stays strictly above 16 (float would round to 16.0)
    assert P("16.0000000000000001b") == Decimal("16.0000000000000001")
    assert P("16.1b") == Decimal("16.1")
    # MoE decimal products
    assert P("2x8b") == Decimal("16")
    assert P("2x8.1b") == Decimal("16.2")
    # preserved R2.0-R2.2 vectors: separator-delimited heavy forms stay SIZE
    assert P("llama3.1/70b") == Decimal("70")
    assert P("llama3.1@70b") == Decimal("70")
    assert P("gemma3/27b") == Decimal("27")
    assert P("qwen3:8x7b") == Decimal("56")
    assert P("llama3.1:8b") == Decimal("8")
    assert P("qwen2.5:7b") == Decimal("7")
    # R2.3-P3: glued/dangling forms are now INVALID (non-full-match island) ->
    # parse None; their CONTRACT OUTCOME (reject) is preserved.
    for _inv in ("llama3.1.70b", "qwen3:235b4bit", "qwen3:8x7b4bit", "qwen3:30b-a3b"):
        assert P(_inv) is None, _inv
        assert model_picker._satisfies_foreground_contract(_inv) is False, _inv
    # unsized alias -> no "b" marker -> None (admitted on family; residual)
    assert P("qwen3-coder-next:latest") is None


def test_contract_rejects_above_ceiling_exactly():
    """R2.3 unit: the ceiling compare is exact Decimal -- 16 and
    16.0000000000000000 admit; 16.0000000000000001 and 16.1 reject; small
    decimals are admitted (not false-rejected)."""
    C = model_picker._satisfies_foreground_contract
    assert C("qwen3:15.9999999999999999b") is True
    assert C("qwen3:16b") is True
    assert C("qwen3:16.0000000000000000b") is True
    assert C("qwen3:16.0000000000000001b") is False   # precision boundary heavy
    assert C("qwen3:16.1b") is False
    assert C("qwen3:2x8b") is True                      # 16 exactly
    assert C("qwen3:2x8.1b") is False                   # 16.2
    assert C("qwen3:0.5b") is True                      # was false-rejected (5) — no
    assert C("qwen3:1.17b") is True                     # was false-rejected (17) — no


def test_precision_boundary_contract_and_direct_catalog():
    """R2.3: the precision-boundary heavy is rejected by the contract helper and
    by the direct catalog picker."""
    assert model_picker._satisfies_foreground_contract(_PRECISION_HEAVY) is False
    entry = {"id": _PRECISION_HEAVY, "hivemind_status": "installed", "hivemind_reachable": True}
    assert model_picker._pick_from_catalog([entry]) is None


def test_precision_boundary_suitable_recommendation_reconciles(monkeypatch):
    """R2.3: a 'suitable' recommendation of the precision-boundary heavy must NOT
    be consumed verbatim; it reconciles to the warm small candidate."""
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    monkeypatch.delenv("MS4_DEFAULT_MODEL", raising=False)
    _patch_recommendation(
        monkeypatch,
        {"capability": "chat", "recommended_model": _PRECISION_HEAVY, "backend": "ollama"},
    )
    _patch_catalog(
        monkeypatch,
        [
            {"id": _PRECISION_HEAVY, "hivemind_status": "installed", "hivemind_reachable": True},
            {"id": "llama3.1:8b", "hivemind_status": "installed", "hivemind_reachable": True},
        ],
    )

    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)

    assert choice.model_id != _PRECISION_HEAVY
    assert choice.model_id == "llama3.1:8b"


def test_precision_boundary_offcontract_rec_heavy_only_catalog_fails_closed(monkeypatch):
    """R2.3: off-contract precision-boundary rec + heavy-only catalog -> safe default."""
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    monkeypatch.delenv("MS4_DEFAULT_MODEL", raising=False)
    _patch_recommendation(
        monkeypatch,
        {"capability": "chat", "recommended_model": _PRECISION_HEAVY, "backend": "ollama"},
    )
    _patch_catalog(
        monkeypatch,
        [{"id": _PRECISION_HEAVY, "hivemind_status": "installed", "hivemind_reachable": True}],
    )

    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)

    assert choice.model_id != _PRECISION_HEAVY
    assert choice.model_id == DEFAULT_FOREGROUND_MODEL


def test_precision_boundary_no_rec_heavy_only_catalog_fails_closed(monkeypatch):
    """R2.3: no recommendation + heavy-only catalog of the precision-boundary
    variant -> safe default (catalog picker rejects it exactly)."""
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    monkeypatch.delenv("MS4_DEFAULT_MODEL", raising=False)
    _patch_recommendation(monkeypatch, {})  # no usable recommendation -> None
    _patch_catalog(
        monkeypatch,
        [{"id": _PRECISION_HEAVY, "hivemind_status": "installed", "hivemind_reachable": True}],
    )

    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)

    assert choice.model_id != _PRECISION_HEAVY
    assert choice.model_id == DEFAULT_FOREGROUND_MODEL


def test_small_decimal_model_admitted_not_false_rejected(monkeypatch):
    """R2.3: a ~1B decimal model ('qwen3:1.17b') satisfies the contract and is
    consumed once exact loaded/reachable catalog readiness is confirmed."""
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    _patch_recommendation(
        monkeypatch,
        {"capability": "chat", "recommended_model": "qwen3:1.17b", "backend": "ollama"},
    )
    _patch_catalog(
        monkeypatch,
        [{"id": "qwen3:1.17b", "hivemind_status": "installed", "hivemind_reachable": True}],
    )

    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)

    assert choice.model_id == "qwen3:1.17b"
    assert choice.source == "loaded"


# ---------------------------------------------------------------------------
# R2.3-P2: explicit SIZE / NO_SIZE / INVALID parser contract.
#   INVALID (malformed or family/version-ambiguous size-like expr) must FAIL
#   CLOSED (contract=False; route to safe default) -- never salvaged to a
#   smaller token, never treated as NO_SIZE. All complete valid spans
#   contribute to the max. MoE products are exact (no ambient rounding).
# ---------------------------------------------------------------------------

_R2_3_P2_INVALID = [
    "gemma3.17b",        # family digit 3 glued to .17 (must NOT under-read to 3.17)
    "qwen3:17..0b",      # double dot
    "qwen3:17.0.0b",     # multiple dots
    "qwen3:1e2b",        # scientific notation (must NOT salvage 2b)
    "qwen3:9xx2b",       # double x
    "qwen3:9x2.b",       # trailing dot in MoE size
    "qwen3:.5b",         # leading dot (no integer part)
    "qwen3:16.b",        # trailing dot before b
    "gemma3.9x2b",       # family 3 glued into 3.9x2
]

# subset used for the heavier e2e matrices (INVALID + the exact-precision MoE)
_R2_3_P2_FAILCLOSED = [
    "gemma3.17b",
    "qwen3:17..0b",
    "qwen3:1e2b",
    "gemma3.9x2b",
    "qwen3:2x8.00000000000000000000000000001b",
]


@pytest.mark.parametrize("mid", _R2_3_P2_INVALID)
def test_invalid_size_expressions_fail_closed_contract(mid):
    """R2.3-P2: a malformed / family-ambiguous size-like expression is INVALID
    and must fail the contract (never salvaged to a smaller token, never
    NO_SIZE-admitted)."""
    assert model_picker._satisfies_foreground_contract(mid) is False, mid


def test_multi_span_size_takes_max_reject():
    """R2.3-P2: every complete size span contributes; max wins -> reject."""
    assert model_picker._satisfies_foreground_contract("qwen3:8b-meta-70b") is False


def test_multi_span_size_all_small_allow():
    """R2.4 supersedes R2.3-P2: the diagnostic classifier still reads the max
    span (16), but the positive-allow gate requires a SINGLE clean whole-ID
    ``^family:sizeb$`` tag, so a hyphenated multi-island tag fails closed."""
    assert model_picker._classify_param_size("qwen3:0.5b-1.17b-16b") == ("size", Decimal("16"))
    assert model_picker._satisfies_foreground_contract("qwen3:0.5b-1.17b-16b") is False


def test_moe_exact_precision_reject():
    """R2.3-P2: 2 x 8.00000000000000000000000000001 = 16.00000000000000000000000000002
    (exact, no ambient rounding to 16.0) -> reject."""
    assert model_picker._satisfies_foreground_contract(
        "qwen3:2x8.00000000000000000000000000001b"
    ) is False


def test_moe_exact_precision_allow():
    """R2.3-P2: 2 x 7.99999999999999999999999999999 = 15.99999999999999999999999999998
    -> allow (exact)."""
    assert model_picker._satisfies_foreground_contract(
        "qwen3:2x7.99999999999999999999999999999b"
    ) is True


@pytest.mark.parametrize("mid", _R2_3_P2_FAILCLOSED)
def test_failclosed_rec_with_small_fallback_reconciles(monkeypatch, mid):
    """R2.3-P2 e2e: an INVALID/off-contract recommendation with a warm small
    candidate in the catalog reconciles to the small candidate, never itself."""
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    monkeypatch.delenv("MS4_DEFAULT_MODEL", raising=False)
    _patch_recommendation(
        monkeypatch,
        {"capability": "chat", "recommended_model": mid, "backend": "ollama"},
    )
    _patch_catalog(
        monkeypatch,
        [
            {"id": mid, "hivemind_status": "installed", "hivemind_reachable": True},
            {"id": "llama3.1:8b", "hivemind_status": "installed", "hivemind_reachable": True},
        ],
    )

    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)

    assert choice.model_id != mid, mid
    assert choice.model_id == "llama3.1:8b"


@pytest.mark.parametrize("mid", _R2_3_P2_FAILCLOSED)
def test_failclosed_rec_heavy_only_catalog_safe_default(monkeypatch, mid):
    """R2.3-P2 e2e: an INVALID/off-contract recommendation + heavy-only catalog
    (only itself) -> safe default."""
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    monkeypatch.delenv("MS4_DEFAULT_MODEL", raising=False)
    _patch_recommendation(
        monkeypatch,
        {"capability": "chat", "recommended_model": mid, "backend": "ollama"},
    )
    _patch_catalog(
        monkeypatch,
        [{"id": mid, "hivemind_status": "installed", "hivemind_reachable": True}],
    )

    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)

    assert choice.model_id != mid, mid
    assert choice.model_id == DEFAULT_FOREGROUND_MODEL


@pytest.mark.parametrize("mid", _R2_3_P2_FAILCLOSED)
def test_failclosed_no_rec_heavy_only_catalog_safe_default(monkeypatch, mid):
    """R2.3-P2 e2e: no usable recommendation + heavy-only catalog of an
    INVALID/off-contract variant -> safe default (catalog gate rejects it)."""
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    monkeypatch.delenv("MS4_DEFAULT_MODEL", raising=False)
    _patch_recommendation(monkeypatch, {})  # no usable recommendation -> None
    _patch_catalog(
        monkeypatch,
        [{"id": mid, "hivemind_status": "installed", "hivemind_reachable": True}],
    )

    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)

    assert choice.model_id != mid, mid
    assert choice.model_id == DEFAULT_FOREGROUND_MODEL


# ---------------------------------------------------------------------------
# R2.3-P3 (reviewer Peirce): lexical island grammar. Two new P1 bypass classes
# the P2 region-scan missed must FAIL CLOSED:
#   * sign/scientific salvage: "1e+2b"->2, "2x+8.1b"->8.1 (punctuation outside
#     the [0-9.xe] region let the scan under-read a trailing number).
#   * glued-integer NO_SIZE admission: "gemma327b", "phi4-mini70b",
#     "qwen3:foo70b" (a letter/family-glued <n>b fell through to NO_SIZE).
# Islands are bounded ONLY by model separators (: / @ - _ whitespace); a
# size-relevant island must FULL-MATCH plain/MoE grammar or it is INVALID.
# ---------------------------------------------------------------------------

_P3_BYPASS_INVALID = [
    "qwen3:1e+2b",
    "qwen3:1e-2b",
    "qwen3:2x+8.1b",
    "qwen3:2x-8.1b",
    "gemma327b",
    "phi4-mini70b",
    "qwen3:foo70b",
]


@pytest.mark.parametrize("mid", _P3_BYPASS_INVALID)
def test_p3_bypass_forms_fail_closed_contract(mid):
    """R2.3-P3: sign/scientific and glued-integer size-like ids are INVALID and
    fail the contract (never salvaged to a trailing number, never NO_SIZE)."""
    assert model_picker._satisfies_foreground_contract(mid) is False, mid


@pytest.mark.parametrize("mid", _P3_BYPASS_INVALID)
def test_p3_bypass_rec_with_small_fallback_reconciles(monkeypatch, mid):
    """R2.3-P3 e2e (suitable/off-contract recommendation + small fallback): a
    bypass-form recommendation with a warm small candidate reconciles to the
    small candidate, never itself."""
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    monkeypatch.delenv("MS4_DEFAULT_MODEL", raising=False)
    _patch_recommendation(
        monkeypatch,
        {"capability": "chat", "recommended_model": mid, "backend": "ollama"},
    )
    _patch_catalog(
        monkeypatch,
        [
            {"id": mid, "hivemind_status": "installed", "hivemind_reachable": True},
            {"id": "llama3.1:8b", "hivemind_status": "installed", "hivemind_reachable": True},
        ],
    )

    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)

    assert choice.model_id != mid, mid
    assert choice.model_id == "llama3.1:8b"


@pytest.mark.parametrize("mid", _P3_BYPASS_INVALID)
def test_p3_bypass_rec_heavy_only_catalog_safe_default(monkeypatch, mid):
    """R2.3-P3 e2e (recommendation + heavy-only catalog): -> safe default."""
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    monkeypatch.delenv("MS4_DEFAULT_MODEL", raising=False)
    _patch_recommendation(
        monkeypatch,
        {"capability": "chat", "recommended_model": mid, "backend": "ollama"},
    )
    _patch_catalog(
        monkeypatch,
        [{"id": mid, "hivemind_status": "installed", "hivemind_reachable": True}],
    )

    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)

    assert choice.model_id != mid, mid
    assert choice.model_id == DEFAULT_FOREGROUND_MODEL


@pytest.mark.parametrize("mid", _P3_BYPASS_INVALID)
def test_p3_bypass_no_rec_heavy_only_catalog_safe_default(monkeypatch, mid):
    """R2.3-P3 e2e (no recommendation + heavy-only catalog): the catalog gate
    rejects the bypass form -> safe default."""
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    monkeypatch.delenv("MS4_DEFAULT_MODEL", raising=False)
    _patch_recommendation(monkeypatch, {})  # no usable recommendation -> None
    _patch_catalog(
        monkeypatch,
        [{"id": mid, "hivemind_status": "installed", "hivemind_reachable": True}],
    )

    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)

    assert choice.model_id != mid, mid
    assert choice.model_id == DEFAULT_FOREGROUND_MODEL


def test_p3_bypass_legacy_recommendation_reconciles(monkeypatch):
    """R2.3-P3 e2e (legacy list recommendation shape): a glued bypass form in
    the legacy ``{"recommended":[...]}`` shape reconciles to the small candidate."""
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    monkeypatch.delenv("MS4_DEFAULT_MODEL", raising=False)
    _patch_recommendation(
        monkeypatch,
        {"recommended": [{"model_id": "gemma327b", "category": "llm", "capabilities": ["chat"]}]},
    )
    _patch_catalog(
        monkeypatch,
        [{"id": "llama3.1:8b", "hivemind_status": "installed", "hivemind_reachable": True}],
    )

    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)

    assert choice.model_id != "gemma327b"
    assert choice.model_id == "llama3.1:8b"


def test_p3_preserve_valid_admit_guard():
    """R2.3-P3 guard: preserve-valid vectors still admit (SIZE<=ceiling), and
    unsized ids classify NO_SIZE. Family gating is a SEPARATE concern -- bare
    "qwen2.5" is not itself a FOREGROUND_PRIORITY_PATTERNS entry (only
    "qwen2.5:7b" is), so it is asserted at the classifier level, not the
    contract level."""
    C = model_picker._satisfies_foreground_contract
    K = model_picker._classify_param_size
    assert C("qwen3:0.5b") is True
    assert C("qwen3:1.17b") is True
    assert C("qwen3:16b") is True
    assert C("qwen3:2x8b") is True
    # R2.4 supersedes P3: a multi-island tag, a bare family, and a ":latest"
    # alias all fail closed under the whole-ID positive-allow gate.
    assert C("qwen3:0.5b-1.17b-16b") is False
    assert C("llama3.1") is False
    assert C("qwen3-coder-next:latest") is False
    # NO_SIZE classification (no size-gate rejection); family match is separate.
    assert K("qwen2.5") == ("no_size", None)
    assert K("llama3.1") == ("no_size", None)
    assert K("family:latest") == ("no_size", None)
    # qwen2.5 is version-only: "2.5" must NOT be misparsed as a size (NO_SIZE),
    # but qwen2.5 is NOT a foreground family (only "qwen2.5:7b" is), so its PRIOR
    # contract outcome (False) is preserved -- the grammar does not widen support.
    assert model_picker._parse_param_size_b("qwen2.5") is None
    assert C("qwen2.5") is False


def test_p3_preserve_moe_precision_guard():
    """R2.3-P3 guard: exact MoE precision preserved."""
    C = model_picker._satisfies_foreground_contract
    assert C("qwen3:2x7.99999999999999999999999999999b") is True
    assert C("qwen3:2x8.00000000000000000000000000001b") is False


# ---------------------------------------------------------------------------
# R2.4 (reviewers Mencius + Descartes): bounded positive-ALLOW gate. App
# Registry gives only an ID and /v1/models has NO trusted param-count metadata,
# so auto-selection admits ONLY:
#   (a) the exact-safe immutable ID set (EXACTLY eight ids), OR
#   (b) a whole-ID anchored ``^<grammar-family>:<size|MoE>b$`` where the family
#       segment equals ONE grammar family EXACTLY (anchored, whole
#       family-before-colon) and 0 < size <= 16 (exact Decimal).
# Grammar families are LIMITED to the intentionally family-like priority
# patterns {qwen3, llama3.1, gemma4, gemma3, gemma2, gemma}. The exact-priority
# ids qwen2.5:7b / llama3.2:8b are NOT widened into families: qwen2.5 / llama3.2
# are admitted ONLY as their exact-safe ids, so qwen2.5:14b, llama3.2:2x8b and
# qwen3-coder-next:8b FAIL CLOSED. Preconditions: input is a non-empty str,
# len <= 256, ASCII. Case is normalized to lowercase (documented + asserted).
# Multi-island tags (e.g. qwen3:0.5b-1.17b-16b) are NOT a single clean
# ^family:sizeb$ and fail closed. `_p4_oracle` is an INDEPENDENT reference impl.
# ---------------------------------------------------------------------------

_P4_EXACT_SAFE = frozenset({
    "nemotron-3-nano:4b",
    "qwen3:8b", "llama3.1:8b", "llama3.2:8b", "qwen2.5:7b",
    "dolphin-llama3", "phi4-mini", "phi3.5",
})
# Grammar families = ONLY the intentionally family-like priority patterns
# (bare-family / "qwen3:" prefix). qwen2.5, llama3.2, qwen3-coder-next,
# dolphin-llama3, phi4-mini and phi3.5 are NOT grammar families (exact-safe
# only) -- so the family segment must equal one of these SIX exactly.
_P4_GRAMMAR_FAMILIES = frozenset({
    "qwen3", "llama3.1", "gemma4", "gemma3", "gemma2", "gemma",
})
_P4_PLAIN_RE = re.compile(r"(\d+(?:\.\d+)?)b")
_P4_MOE_RE = re.compile(r"(\d+)x(\d+(?:\.\d+)?)b")


def _oracle_int_or_none(s):
    return int(s) if (s != "" and s.isdigit()) else None


def _oracle_dec_rational(token):
    """Parse '<digits>[.<digits>]b' as an integer (num, den) rational, else None.
    Manual character scan -- NO production regex, NO Fraction."""
    if not token.endswith("b"):
        return None
    body = token[:-1]
    if body.count(".") > 1:
        return None
    if "." in body:
        ip, _, fp = body.partition(".")
        if ip == "" or fp == "" or not ip.isdigit() or not fp.isdigit():
            return None
        return (int(ip) * (10 ** len(fp)) + int(fp), 10 ** len(fp))
    if body == "" or not body.isdigit():
        return None
    return (int(body), 1)


def _oracle_size_rational(tag):
    """Whole-tag size as an integer (num, den) rational: plain '<n>b' or MoE
    '<e>x<n>b'. Manual scan -- structurally independent of the production
    regex/Fraction path."""
    if "x" in tag:
        left, _, right = tag.partition("x")
        e = _oracle_int_or_none(left)
        s = _oracle_dec_rational(right)
        if e is None or s is None:
            return None
        return (e * s[0], s[1])
    return _oracle_dec_rational(tag)


def _p4_oracle(mid):
    """R2.4-P4 (Sagan P2-3): STRUCTURALLY INDEPENDENT reference implementation of
    the corrected R2.4 policy. It shares only the POLICY (allowlist tuples), not
    the production PARSING/ARITHMETIC structure: a manual character scan +
    integer (num, den) rational compare (``num/den <= 16`` via ``num <= 16*den``)
    -- NO production regex, NO Fraction, NO ``.isascii()``. A shared regex or
    Fraction structural bug therefore cannot hide behind a mirrored oracle.
    Case-insensitive; non-str / empty / >256 / non-ASCII fail closed."""
    if not isinstance(mid, str) or not mid:
        return False
    if len(mid) > 256:
        return False
    if any(ord(ch) > 127 for ch in mid):
        return False
    low = mid.lower()
    if low in _P4_EXACT_SAFE:
        return True
    if ":" not in low:
        return False
    fam, _, tag = low.partition(":")
    if fam not in _P4_GRAMMAR_FAMILIES:
        return False
    r = _oracle_size_rational(tag)
    if r is None:
        return False
    num, den = r
    return num > 0 and num <= 16 * den


_P4_ADMIT = [
    # exact-safe immutable set (all eight)
    "nemotron-3-nano:4b",
    "qwen3:8b", "llama3.1:8b", "llama3.2:8b", "qwen2.5:7b",
    "dolphin-llama3", "phi4-mini", "phi3.5",
    # grammar-family + explicit small size (NOT exact-safe): the ONLY widening
    "qwen3:0.5b", "qwen3:16b", "qwen3:2x8b", "qwen3:1.17b", "qwen3:14b",
    "gemma3:8b", "gemma:16b", "gemma2:2x8b",
    # case-insensitive normalization (pinned): mixed-case forms admit
    "Qwen3:8B", "QWEN3:8B",
]
_P4_REJECT = [
    # unknown family / no family / bare size
    "mixtral:8x7b", "exotic-model:42b", "some-other-heavy:400b", "16b", "0.5b", "2x8b",
    # NON-grammar approved-priority families sized: exact-safe ONLY, so these
    # fail closed (Mencius: do NOT widen exact priorities into families).
    "qwen2.5:14b", "llama3.2:2x8b", "qwen3-coder-next:8b", "qwen2.5:0.5b", "llama3.2:16b",
    # unsized / :latest / bare / mutable alias (P3 fail-open)
    "llama3.1", "gemma3", "qwen2.5", "qwen3:latest", "qwen3-coder-next:latest",
    "phi4-mini:latest", "qwen3:instruct",
    # multi-island whole-ID (not a single clean ^family:sizeb$) (Descartes)
    "qwen3:0.5b-1.17b-16b", "qwen3:8b-meta-70b",
    # over-ceiling
    "qwen3:70b", "qwen3:235b", "qwen3:8x7b", "qwen3:16.1b",
    "qwen3:16.0000000000000001b", "qwen3:2x8.00000000000000000000000000001b",
    # zero-operand MoE / zero (P3 fail-open: size 0 admitted)
    "qwen3:0x235b", "qwen3:235x0b", "qwen3:0b",
    # signed / scientific / glued / punctuation / suffix
    "qwen3:1e+2b", "qwen3:1e-2b", "qwen3:2x+8.1b", "qwen3:2x-8.1b",
    "gemma327b", "phi4-mini70b", "qwen3:foo70b", "gemma3.17b", "llama3.1.70b",
    "qwen3:235b4bit", "llama3.1/70b", "gemma3/27b", "llama3.1:8b_ollama",
    # whitespace / non-ascii-letter / over-length
    "qwen3:8b\u3000", "qwen\u039e3:8b", "qwen3:" + "8" * 300 + "b",
]


@pytest.mark.parametrize("mid", _P4_ADMIT, ids=[f"admit{i}" for i in range(len(_P4_ADMIT))])
def test_p4_admit(mid):
    """R2.4: exact-safe + approved-family:size(<=16) admit (and match the oracle)."""
    assert model_picker._satisfies_foreground_contract(mid) is True, ascii(mid)
    assert _p4_oracle(mid) is True, ascii(mid)


@pytest.mark.parametrize("mid", _P4_REJECT, ids=[f"reject{i}" for i in range(len(_P4_REJECT))])
def test_p4_reject(mid):
    """R2.4: unknown / unsized / :latest / glued / punctuation / zero-MoE /
    non-ascii / over-length / over-ceiling all fail closed (and match oracle)."""
    assert model_picker._satisfies_foreground_contract(mid) is False, ascii(mid)
    assert _p4_oracle(mid) is False, ascii(mid)


# Process-deterministic generation inputs: FIXED sorted tuples (NEVER
# list(frozenset(...)), whose order varies with hash randomization across
# processes). All inputs are ASCII so the corpus never trips a Decimal parse on
# a Unicode digit. The generated corpus is byte-identical across runs/processes.
_P4_GEN_FAMILIES = (
    "bar", "codellama", "dolphin-llama3", "exotic", "foo", "gemma", "gemma2",
    "gemma3", "gemma4", "gemmax", "llama3.1", "llama3.2", "llama4", "mixtral",
    "nemotron", "phi3.5", "phi4-mini", "qwen2.5", "qwen3", "qwen3-coder-next",
    "qwen9",
)
_P4_GEN_SEPS = (":", ":", ":", "-", "/", "@", ".", "", "_")
_P4_GEN_WORD_TAGS = ("latest", "instruct", "chat", "q4_0", "8b_ollama", "8b4bit", "")
_P4_GEN_BAD_TAGS = ("1e+2b", "2x+8.1b", "0x8b", "8x0b", "16.b", ".5b",
                    "8..0b", "1e2b", "70b.gguf", "27b/it")
_P4_GEN_SUFFIX = ("-instruct", "/x", "..0b", ".gguf", "xx2b", "+q")


def _p4_gen(rnd):
    fam = rnd.choice(_P4_GEN_FAMILIES)
    r = rnd.random()
    if r < 0.12:
        return fam  # bare family
    sep = rnd.choice(_P4_GEN_SEPS)
    t = rnd.random()
    if t < 0.4:
        tag = f"{rnd.randint(0, 300)}b"
    elif t < 0.55:
        tag = f"{rnd.randint(0, 20)}.{rnd.randint(0, 99)}b"
    elif t < 0.72:
        tag = f"{rnd.randint(0, 16)}x{rnd.randint(0, 50)}b"
    elif t < 0.84:
        tag = rnd.choice(_P4_GEN_WORD_TAGS)
    else:
        tag = rnd.choice(_P4_GEN_BAD_TAGS)
    mid = f"{fam}{sep}{tag}"
    if rnd.random() < 0.03:
        mid = mid + rnd.choice(_P4_GEN_SUFFIX)
    return mid


def test_p4_corpus_matches_oracle():
    """R2.4: a deterministic ~50k generated corpus must satisfy
    `_satisfies_foreground_contract(id) == _p4_oracle(id)` for every id.
    On the P3 inference parser this fails open on family+no-size / :latest /
    non-ascii ids; the positive-allow gate makes it agree with the oracle."""
    rnd = random.Random(20260706)
    corpus = [_p4_gen(rnd) for _ in range(50000)]
    C = model_picker._satisfies_foreground_contract
    mismatches = []
    for mid in corpus:
        if C(mid) != _p4_oracle(mid):
            mismatches.append(mid)
            if len(mismatches) >= 25:
                break
    assert not mismatches, (
        f"{len(mismatches)}+ corpus mismatches (fail-open/arch), e.g. "
        f"{[ascii(m) for m in mismatches[:12]]}"
    )


_P4_E2E_FAILCLOSED = ["qwen3:latest", "llama3.1", "qwen3-coder-next:latest", "qwen3:0x235b", "qwen2.5:14b"]


@pytest.mark.parametrize("mid", _P4_E2E_FAILCLOSED, ids=[f"fc{i}" for i in range(len(_P4_E2E_FAILCLOSED))])
def test_p4_failclosed_rec_reconciles_to_small(monkeypatch, mid):
    """R2.4 e2e (suitable / off-contract rec + small fallback): a non-admitted
    recommendation with a warm exact-safe candidate reconciles to it."""
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    monkeypatch.delenv("MS4_DEFAULT_MODEL", raising=False)
    _patch_recommendation(
        monkeypatch,
        {"capability": "chat", "recommended_model": mid, "backend": "ollama"},
    )
    _patch_catalog(
        monkeypatch,
        [
            {"id": mid, "hivemind_status": "installed", "hivemind_reachable": True},
            {"id": "llama3.1:8b", "hivemind_status": "installed", "hivemind_reachable": True},
        ],
    )
    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)
    assert choice.model_id != mid, ascii(mid)
    assert choice.model_id == "llama3.1:8b"


@pytest.mark.parametrize("mid", _P4_E2E_FAILCLOSED, ids=[f"fc{i}" for i in range(len(_P4_E2E_FAILCLOSED))])
def test_p4_failclosed_no_rec_heavy_only_catalog(monkeypatch, mid):
    """R2.4 e2e (no recommendation + non-admitted-only catalog): safe default."""
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    monkeypatch.delenv("MS4_DEFAULT_MODEL", raising=False)
    _patch_recommendation(monkeypatch, {})
    _patch_catalog(
        monkeypatch,
        [{"id": mid, "hivemind_status": "installed", "hivemind_reachable": True}],
    )
    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)
    assert choice.model_id != mid, ascii(mid)
    assert choice.model_id == DEFAULT_FOREGROUND_MODEL


def test_p4_genuine_admit_rec_after_ready_catalog_confirmation(monkeypatch):
    """R2.4 guard: an exact-safe recommendation still requires ready catalog proof."""
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    _patch_recommendation(
        monkeypatch,
        {"capability": "chat", "recommended_model": "qwen3:8b", "backend": "ollama"},
    )
    _patch_catalog(
        monkeypatch,
        [{"id": "qwen3:8b", "hivemind_status": "installed", "hivemind_reachable": True}],
    )
    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)
    assert choice.model_id == "qwen3:8b"
    assert choice.source == "loaded"


def test_p4_direct_loaded_catalog_admits(monkeypatch):
    """R2.4 guard: no recommendation + a loaded exact-safe catalog entry is picked."""
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    _patch_recommendation(monkeypatch, {})
    _patch_catalog(
        monkeypatch,
        [{"id": "qwen3:8b", "loaded": True, "available": True}],
    )
    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)
    assert choice.model_id == "qwen3:8b"
    assert choice.source == "loaded"


def test_p4_override_still_wins(monkeypatch):
    """R2.4 guard: explicit MS4_FOREGROUND_MODEL override bypasses the gate."""
    monkeypatch.setenv(MS4_FOREGROUND_MODEL_ENV, "operator-choice:custom")
    monkeypatch.setattr(
        hivemind_tools, "models_recommend",
        lambda *_a, **_k: pytest.fail("override must short-circuit the seam"),
    )
    monkeypatch.setattr(
        model_picker, "_http_get_models",
        lambda *_a, **_k: pytest.fail("override must not consult the catalog"),
    )
    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)
    assert choice.model_id == "operator-choice:custom"
    assert choice.source == "override"


# ===========================================================================
# R2.4-P4 (second-wave reviewer): the RETAINED DIAGNOSTIC helpers must ALSO be
# ambient-decimal-context independent. `_moe_product` used decimal.localcontext()
# (which COPIES the ambient context), so `_classify_param_size` /
# `_parse_param_size_b` RAISE (Overflow/Subnormal) or FLIP (clamp->Infinity)
# under a hostile ambient context even though they are diagnostic-only. Assert
# the DESIRED behavior (correct value, no raise) so the current code fails by
# raising/flipping.
# ===========================================================================

_P4P4_HOSTILE = [
    ("overflow", Context(prec=28, Emax=0, Emin=-999999, traps=[Overflow])),
    ("clamp", Context(prec=28, Emax=0, Emin=-999999, clamp=1, traps=[])),
    ("subnormal", Context(prec=28, Emax=999999, Emin=0, traps=[Subnormal])),
]


@pytest.mark.parametrize("label,ctx", _P4P4_HOSTILE, ids=[c[0] for c in _P4P4_HOSTILE])
def test_p4p4_classify_diagnostic_ambient_independent(label, ctx):
    """R2.4-P4: the retained diagnostic classifier + parser must be ambient
    decimal-context independent -- correct SIZE classification, no raise/flip,
    under a hostile ambient context."""
    with decimal.localcontext(ctx):
        assert model_picker._classify_param_size("qwen3:2x8b") == ("size", 16)
        assert model_picker._parse_param_size_b("qwen3:2x8b") == 16
        assert model_picker._classify_param_size("qwen3:1x0.5b") == ("size", Fraction(1, 2))
        assert model_picker._parse_param_size_b("qwen3:1x0.5b") == Fraction(1, 2)


@pytest.mark.parametrize("label,ctx", _P4P4_HOSTILE, ids=[c[0] for c in _P4P4_HOSTILE])
def test_p4p4_moe_product_ambient_independent(label, ctx):
    """R2.4-P4: _moe_product itself must be exact and ambient decimal-context
    independent (no Overflow/Subnormal raise, no clamp -> Infinity flip)."""
    with decimal.localcontext(ctx):
        assert model_picker._moe_product(Decimal("2"), Decimal("8")) == 16
        assert model_picker._moe_product(Decimal("1"), Decimal("0.5")) == Fraction(1, 2)


def test_p4p4_diagnostic_semantics_preserved():
    """R2.4-P4 guard: the diagnostic classifications/sizes are unchanged under a
    NORMAL ambient context (the ambient-independence fix must not weaken the
    retained diagnostic semantics)."""
    K = model_picker._classify_param_size
    P = model_picker._parse_param_size_b
    assert K("qwen3:8x7b") == ("size", 56)
    assert P("qwen3:8x7b") == 56
    assert K("qwen3:0.5b-1.17b-16b") == ("size", Decimal("16"))
    assert K("qwen2.5") == ("no_size", None)
    assert K("gemma327b") == ("invalid", None)
    assert P("qwen2.5") is None


def test_p4p4_cache_result_not_poisonable(monkeypatch):
    """R2.4-P4 (Sagan P2-1): a cached ForegroundChoice must not be poisonable by a
    caller mutating the RETURNED object. Each call must hand back a defensive
    copy (or an immutable result), so a same-key call cannot return a poisoned
    off-contract object without a recompute."""
    import dataclasses as _dc
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    monkeypatch.delenv("MS4_DEFAULT_MODEL", raising=False)
    _patch_recommendation(monkeypatch, {})
    _patch_catalog(monkeypatch, [{"id": "qwen3:8b", "loaded": True, "available": True}])
    c1 = choose_foreground_model(hivemind_url="http://hive")   # cache MISS -> cached
    assert c1.model_id == "qwen3:8b"
    try:
        c1.model_id = "qwen3:235b"   # caller poisons the returned object
    except _dc.FrozenInstanceError:
        pass  # an immutable (frozen) result is also acceptable
    c2 = choose_foreground_model(hivemind_url="http://hive")   # same key -> cache HIT
    assert c2.model_id == "qwen3:8b", f"cache poisoned by caller mutation: {c2.model_id}"


def test_p4p4_stale_inflight_does_not_overwrite_newer(monkeypatch):
    """R2.4-P4 (Sagan P2-2): an older in-flight selection must NOT blind-overwrite
    a newer force_refresh commit. Deterministic (event barrier, NO sleeps): OLD
    (generation N) pauses before commit; NEW (generation N+1) commits qwen3:14b;
    OLD resumes and must NOT replace it with the stale qwen3:8b."""
    import threading
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    monkeypatch.delenv("MS4_DEFAULT_MODEL", raising=False)
    lock = threading.Lock()
    order = {"n": 0}
    selection = threading.local()
    old_in_recommend = threading.Event()
    new_committed = threading.Event()

    def fake_recommend(*_a, **_k):
        with lock:
            order["n"] += 1
            mine = order["n"]
        if mine == 1:
            old_in_recommend.set()
            assert new_committed.wait(timeout=10)
            selection.model_id = "qwen3:8b"
            return {"capability": "chat", "recommended_model": "qwen3:8b", "backend": "ollama"}
        selection.model_id = "qwen3:14b"
        return {"capability": "chat", "recommended_model": "qwen3:14b", "backend": "ollama"}

    monkeypatch.setattr(hivemind_tools, "models_recommend", fake_recommend)
    monkeypatch.setattr(
        model_picker,
        "_http_get_models",
        lambda *_a, **_k: [
            {
                "id": selection.model_id,
                "hivemind_status": "installed",
                "hivemind_reachable": True,
            }
        ],
    )
    results = {}

    def run_old():
        results["old"] = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)

    t_old = threading.Thread(target=run_old)
    t_old.start()
    assert old_in_recommend.wait(timeout=10)   # OLD captured its generation and is paused
    results["new"] = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)
    new_committed.set()
    t_old.join(timeout=10)
    final = choose_foreground_model(hivemind_url="http://hive")   # cached
    assert results["new"].model_id == "qwen3:14b"
    assert final.model_id == "qwen3:14b", f"stale in-flight overwrote newer commit: {final.model_id}"


def test_p4p5_no_cross_key_leak_on_skipped_commit(monkeypatch):
    """R2.4-P5: when an older in-flight commit is skipped because a newer
    generation committed first UNDER A DIFFERENT KEY, the older caller must
    return ITS OWN computed result for ITS OWN key -- never adopt a value
    computed under a different url / override / MS4_DEFAULT_MODEL. The newer
    cache entry is preserved (only the older commit is skipped). Deterministic
    (event barrier, no sleeps).

    The generation-ordering fix (Sagan P2-2) introduced an adopt-newer RETURN in
    its skipped-commit else branch that leaked a cross-key result to the older
    caller; this pins the required per-key isolation."""
    import threading
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    monkeypatch.delenv("MS4_DEFAULT_MODEL", raising=False)
    lock = threading.Lock()
    order = {"n": 0}
    selection = threading.local()
    old_in_recommend = threading.Event()
    new_committed = threading.Event()

    def fake_recommend(*_a, **_k):
        with lock:
            order["n"] += 1
            mine = order["n"]
        if mine == 1:
            old_in_recommend.set()
            assert new_committed.wait(timeout=10)
            selection.model_id = "qwen3:8b"
            return {"capability": "chat", "recommended_model": "qwen3:8b", "backend": "ollama"}
        selection.model_id = "qwen3:14b"
        return {"capability": "chat", "recommended_model": "qwen3:14b", "backend": "ollama"}

    monkeypatch.setattr(hivemind_tools, "models_recommend", fake_recommend)
    monkeypatch.setattr(
        model_picker,
        "_http_get_models",
        lambda *_a, **_k: [
            {
                "id": selection.model_id,
                "hivemind_status": "installed",
                "hivemind_reachable": True,
            }
        ],
    )
    results = {}

    def run_old():
        results["old"] = choose_foreground_model(hivemind_url="http://old", force_refresh=True)

    t_old = threading.Thread(target=run_old)
    t_old.start()
    assert old_in_recommend.wait(timeout=10)   # OLD (url=http://old) paused post gen-capture
    results["new"] = choose_foreground_model(hivemind_url="http://new", force_refresh=True)
    new_committed.set()
    t_old.join(timeout=10)
    new_cache = choose_foreground_model(hivemind_url="http://new")   # newer cache preserved
    assert results["new"].model_id == "qwen3:14b"
    assert new_cache.model_id == "qwen3:14b", f"newer cache not preserved: {new_cache.model_id}"
    # the older caller must receive ITS OWN result (http://old -> qwen3:8b), NOT
    # the cross-key newer value computed for http://new (qwen3:14b):
    assert results["old"].model_id == "qwen3:8b", f"cross-key leak: {results['old'].model_id}"


# --- R2.4 isolated guards (Descartes) -------------------------------------


def test_p4_nonstring_and_empty_fail_closed():
    """R2.4: non-string input and the empty string fail closed (positive-allow
    precondition). P3 crashes on non-string ids (no isinstance guard)."""
    C = model_picker._satisfies_foreground_contract
    assert C("") is False
    assert _p4_oracle("") is False
    for bad in (None, 123, 4.5, ["qwen3:8b"], {"id": "qwen3:8b"}, object()):
        assert C(bad) is False, ascii(bad)
        assert _p4_oracle(bad) is False, ascii(bad)


def test_p4_length_boundary():
    """R2.4: an id of EXACTLY 256 chars (leading-zero size 8) admits; the SAME
    shape at 257 chars fails on the length gate alone (P3 has no length gate)."""
    C = model_picker._satisfies_foreground_contract
    ok256 = "qwen3:" + "0" * 248 + "8b"     # 6 + 248 + 2 = 256, size == 8 <= 16
    assert len(ok256) == 256
    assert C(ok256) is True
    assert _p4_oracle(ok256) is True
    over257 = "qwen3:" + "0" * 249 + "8b"    # 257 -> length gate rejects
    assert len(over257) == 257
    assert C(over257) is False
    assert _p4_oracle(over257) is False


def test_p4_unicode_digit_fails_closed():
    """R2.4: an approved family with a Unicode (Arabic-Indic) digit fails closed
    on the ASCII precondition even though the char is a 'digit'. P3 has no ASCII
    gate and mis-handles it (admits or raises)."""
    assert _p4_oracle("qwen3:\u0668b") is False
    assert model_picker._satisfies_foreground_contract("qwen3:\u0668b") is False


def test_p4_case_is_lowercase_normalized():
    """R2.4 pin: matching is case-insensitive (lowercase-normalized), so
    ``Qwen3:8B`` / ``QWEN3:8B`` map to the exact-safe id and admit."""
    C = model_picker._satisfies_foreground_contract
    assert C("Qwen3:8B") is True
    assert C("QWEN3:8B") is True
    assert _p4_oracle("Qwen3:8B") is True
    assert _p4_oracle("QWEN3:8B") is True


def test_p4_corpus_is_process_deterministic():
    """R2.4 (Descartes): generation inputs are FIXED sorted tuples, so the
    corpus is byte-identical across runs/processes (no frozenset ordering)."""
    def build():
        r = random.Random(20260706)
        return [_p4_gen(r) for _ in range(2000)]
    assert build() == build()
    assert _P4_GEN_FAMILIES == tuple(sorted(_P4_GEN_FAMILIES))


# --- R2.4 MS4_DEFAULT_MODEL is a GATED fallback, not an override (Mencius) --


def test_p4_ms4_default_model_offgate_no_match_uses_builtin(monkeypatch):
    """R2.4: an off-gate MS4_DEFAULT_MODEL with no catalog match falls back to
    the BUILT-IN exact-safe default, NOT the invalid env value."""
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    monkeypatch.setenv("MS4_DEFAULT_MODEL", "operator-pinned:latest")
    _patch_recommendation(monkeypatch, {})
    _patch_catalog(monkeypatch, [{"id": "exotic-model:42b", "loaded": True, "available": True}])
    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)
    assert choice.model_id != "operator-pinned:latest"
    assert choice.model_id == DEFAULT_FOREGROUND_MODEL


def test_p4_ms4_default_model_offgate_catalog_error_uses_builtin(monkeypatch):
    """R2.4: an off-gate MS4_DEFAULT_MODEL on a catalog error falls back to the
    BUILT-IN exact-safe default, NOT the invalid env value."""
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    monkeypatch.setenv("MS4_DEFAULT_MODEL", "operator-pinned:latest")
    _patch_recommendation(monkeypatch, {})

    def _boom(_url, timeout=5):
        raise RuntimeError("catalog down")

    monkeypatch.setattr(model_picker, "_http_get_models", _boom)
    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)
    assert choice.model_id != "operator-pinned:latest"
    assert choice.model_id == DEFAULT_FOREGROUND_MODEL


# --- R2.4 e2e: both recommendation shapes are gated ------------------------


def test_p4_failclosed_legacy_rec_shape_reconciles(monkeypatch):
    """R2.4 e2e: a P4-rejected alias via the LEGACY list recommendation shape
    also reconciles to the warm exact-safe candidate (both rec shapes gated)."""
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    monkeypatch.delenv("MS4_DEFAULT_MODEL", raising=False)
    _patch_recommendation(
        monkeypatch,
        {"recommended": [{"model_id": "qwen3-coder-next:latest",
                          "category": "llm", "capabilities": ["chat"]}]},
    )
    _patch_catalog(
        monkeypatch,
        [{"id": "llama3.1:8b", "hivemind_status": "installed", "hivemind_reachable": True}],
    )
    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)
    assert choice.model_id != "qwen3-coder-next:latest"
    assert choice.model_id == "llama3.1:8b"


@pytest.mark.parametrize("state", ["loaded", "available"])
def test_p4_family_grammar_catalog_requires_loaded(monkeypatch, state):
    """R2.4: a genuine NON-exact-safe grammar-family id (qwen3:14b, admitted via
    the family grammar, not the exact-safe set) is picked only when loaded."""
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    _patch_recommendation(monkeypatch, {})
    entry = {"id": "qwen3:14b", "loaded": state == "loaded", "available": True}
    _patch_catalog(monkeypatch, [entry])
    if state == "available":
        with pytest.raises(RuntimeError, match="no loaded, reachable, eligible"):
            choose_foreground_model(hivemind_url="http://hive", force_refresh=True)
        return
    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)
    assert choice.model_id == "qwen3:14b"
    assert choice.source == "loaded"


# ===========================================================================
# R2.4-P3 (reviewer Leibniz)
# P1: the size arithmetic (esp. the MoE product) MUST be independent of the
#     ambient decimal context. decimal.localcontext() COPIES the ambient
#     context (Emax/Emin/traps/clamp), so a hostile ambient context makes the
#     gate raise (Overflow/Subnormal trap) or flip (clamp overflow -> Infinity).
#     A caller anywhere in-process can set such a context. The gate must instead
#     use context-free exact arithmetic (Fraction / fresh fully-specified
#     Context). The explicit MS4_FOREGROUND_MODEL override MUST short-circuit
#     BEFORE any automatic-fallback (MS4_DEFAULT_MODEL) validation, so a hostile
#     default is never evaluated when an override is set.
# P2: choose_foreground_model's cache keys only (url, override) and reads
#     MS4_DEFAULT_MODEL AFTER the cache lookup, so a fallback change within the
#     TTL returns a stale result. The effective fallback must be part of the
#     cache key / invalidation.
# ===========================================================================

_HOSTILE_CTXS = [
    ("overflow_trap", Context(prec=28, Emax=0, Emin=-999999, traps=[Overflow])),
    ("clamp_no_trap", Context(prec=28, Emax=0, Emin=-999999, clamp=1, traps=[])),
]


@pytest.mark.parametrize("label,ctx", _HOSTILE_CTXS, ids=[c[0] for c in _HOSTILE_CTXS])
def test_p4p3_moe_gate_hostile_ambient_direct(label, ctx):
    """P1 direct gate: qwen3:2x8b (product 16 <= 16) must admit under a hostile
    ambient decimal context (Emax=0 + Overflow trap, or Emax=0 + clamp), NOT
    raise and NOT flip to False."""
    with decimal.localcontext(ctx):
        assert model_picker._satisfies_foreground_contract("qwen3:2x8b") is True


def test_p4p3_moe_gate_hostile_subnormal_direct():
    """P1 direct gate: a small MoE product (1x0.5 = 0.5) must admit under an
    Emin=0 + Subnormal-trap ambient context (must not raise Subnormal)."""
    with decimal.localcontext(Context(prec=28, Emax=999999, Emin=0, traps=[Subnormal])):
        assert model_picker._satisfies_foreground_contract("qwen3:1x0.5b") is True


def test_p4p3_moe_unavailable_scalar_recommendation_uses_ready_catalog(monkeypatch):
    """A contract-safe but unavailable scalar recommendation cannot win."""
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    _patch_recommendation(
        monkeypatch,
        {"capability": "chat", "recommended_model": "qwen3:2x8b", "backend": "ollama"},
    )
    observed = {"catalog_calls": 0}

    def catalog(*_args, **_kwargs):
        observed["catalog_calls"] += 1
        return [
            {"id": "qwen3:2x8b", "hivemind_status": "available", "hivemind_reachable": True},
            {"id": "llama3.1:8b", "hivemind_status": "installed", "hivemind_reachable": True},
        ]

    monkeypatch.setattr(model_picker, "_http_get_models", catalog)
    with decimal.localcontext(Context(prec=28, Emax=0, Emin=-999999, traps=[Overflow])):
        choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)
    assert observed["catalog_calls"] == 1
    assert choice.model_id == "llama3.1:8b"
    assert choice.source == "loaded"


def test_p4p3_moe_hostile_catalog(monkeypatch):
    """P1 catalog: a warm grammar-family MoE catalog entry is picked under a
    hostile ambient context (the contract filter must not raise)."""
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    _patch_recommendation(monkeypatch, {})
    _patch_catalog(monkeypatch, [{"id": "qwen3:2x8b", "loaded": True, "available": True}])
    with decimal.localcontext(Context(prec=28, Emax=0, Emin=-999999, traps=[Overflow])):
        choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)
    assert choice.model_id == "qwen3:2x8b"
    assert choice.source == "loaded"


def test_p4p3_moe_hostile_offgate_ms4_default(monkeypatch):
    """P1 MS4_DEFAULT_MODEL: an off-gate heavy-MoE default (2x20 = 40 > 16) under
    a hostile ambient context must cleanly fail the gate and fall back to the
    built-in Face default -- not raise Overflow while validating the default."""
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    monkeypatch.setenv("MS4_DEFAULT_MODEL", "qwen3:2x20b")
    _patch_recommendation(monkeypatch, {})
    _patch_catalog(monkeypatch, [{"id": "exotic-model:42b", "loaded": True, "available": True}])
    with decimal.localcontext(Context(prec=28, Emax=0, Emin=-999999, traps=[Overflow])):
        choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)
    assert choice.model_id == DEFAULT_FOREGROUND_MODEL


def test_p4p3_override_shortcircuits_before_hostile_default(monkeypatch):
    """P1 ordering: an explicit MS4_FOREGROUND_MODEL override must short-circuit
    BEFORE any MS4_DEFAULT_MODEL validation, so a hostile MoE default is never
    evaluated (no raise) when an override is present."""
    monkeypatch.setenv(MS4_FOREGROUND_MODEL_ENV, "operator-choice:custom")
    monkeypatch.setenv("MS4_DEFAULT_MODEL", "qwen3:2x8b")
    monkeypatch.setattr(
        hivemind_tools, "models_recommend",
        lambda *_a, **_k: pytest.fail("override must short-circuit the seam"),
    )
    monkeypatch.setattr(
        model_picker, "_http_get_models",
        lambda *_a, **_k: pytest.fail("override must not consult the catalog"),
    )
    with decimal.localcontext(Context(prec=28, Emax=0, Emin=-999999, traps=[Overflow])):
        choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)
    assert choice.model_id == "operator-choice:custom"
    assert choice.source == "override"


def test_p4p3_cache_reflects_ms4_default_change(monkeypatch):
    """P2 cache key: within a single TTL window (no force_refresh), a change to
    MS4_DEFAULT_MODEL must be reflected in the effective fallback -- the cache
    must not return a stale result keyed only on (url, override)."""
    monkeypatch.delenv(MS4_FOREGROUND_MODEL_ENV, raising=False)
    _patch_recommendation(monkeypatch, {})
    _patch_catalog(monkeypatch, [{"id": "exotic-model:42b", "loaded": True, "available": True}])

    monkeypatch.setenv("MS4_DEFAULT_MODEL", "qwen3:16b")
    c1 = choose_foreground_model(hivemind_url="http://hive")
    assert c1.model_id == "qwen3:16b"

    monkeypatch.setenv("MS4_DEFAULT_MODEL", "qwen3:14b")
    c2 = choose_foreground_model(hivemind_url="http://hive")
    assert c2.model_id == "qwen3:14b"      # NOT the cached qwen3:16b

    monkeypatch.delenv("MS4_DEFAULT_MODEL", raising=False)
    c3 = choose_foreground_model(hivemind_url="http://hive")
    assert c3.model_id == DEFAULT_FOREGROUND_MODEL    # built-in when unset

    monkeypatch.setenv("MS4_DEFAULT_MODEL", "qwen3:70b")
    c4 = choose_foreground_model(hivemind_url="http://hive")
    assert c4.model_id == DEFAULT_FOREGROUND_MODEL    # off-gate -> built-in


def test_p4p3_override_no_default_eval(monkeypatch):
    """P1 ordering (call-order spy): when an override is set, the
    automatic-fallback (MS4_DEFAULT_MODEL) gate MUST NOT be evaluated at all. A
    spy on _satisfies_foreground_contract fails if called; the override must win
    without touching it. This asserts the ordering directly (independent of the
    Fraction fix), so a pure ordering regression is caught even though the gate
    no longer raises under a hostile context."""
    monkeypatch.setenv(MS4_FOREGROUND_MODEL_ENV, "operator-choice:custom")
    monkeypatch.setenv("MS4_DEFAULT_MODEL", "qwen3:2x8b")

    def _spy(*_a, **_k):
        pytest.fail("override must short-circuit before any default-gate evaluation")

    monkeypatch.setattr(model_picker, "_satisfies_foreground_contract", _spy)
    monkeypatch.setattr(
        hivemind_tools, "models_recommend",
        lambda *_a, **_k: pytest.fail("override must short-circuit the seam"),
    )
    monkeypatch.setattr(
        model_picker, "_http_get_models",
        lambda *_a, **_k: pytest.fail("override must not consult the catalog"),
    )
    choice = choose_foreground_model(hivemind_url="http://hive", force_refresh=True)
    assert choice.model_id == "operator-choice:custom"
    assert choice.source == "override"


def test_p6_monotonic_clock_survives_wall_clock_rollback(monkeypatch):
    """TTL expiry must use monotonic time even if wall time moves backward."""
    monotonic_values = iter((100.0, 161.0))
    wall_values = iter((1000.0, 900.0))
    monkeypatch.setattr(model_picker.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(model_picker.time, "time", lambda: next(wall_values))

    calls = []

    def fake_get(_url, timeout=20):
        calls.append(1)
        model_id = "qwen3:8b" if len(calls) == 1 else "qwen3:14b"
        return [{"id": model_id, "loaded": True, "available": True}]

    monkeypatch.setattr(model_picker, "_http_get_models", fake_get)
    first = choose_foreground_model(hivemind_url="http://clock")
    after_expiry = choose_foreground_model(hivemind_url="http://clock")

    assert first.model_id == "qwen3:8b"
    assert after_expiry.model_id == "qwen3:14b"
    assert len(calls) == 2


def test_p6_cache_key_includes_hivemind_url(monkeypatch):
    """A cached choice from one HiveMind endpoint cannot satisfy another."""
    monkeypatch.setattr(model_picker.time, "monotonic", lambda: 100.0)
    calls = []
    by_url = {"http://old": "qwen3:8b", "http://new": "qwen3:14b"}

    def fake_get(url, timeout=20):
        calls.append(url)
        return [{"id": by_url[url], "loaded": True, "available": True}]

    monkeypatch.setattr(model_picker, "_http_get_models", fake_get)
    old = choose_foreground_model(hivemind_url="http://old")
    new = choose_foreground_model(hivemind_url="http://new")

    assert old.model_id == "qwen3:8b"
    assert new.model_id == "qwen3:14b"
    assert calls == ["http://old", "http://new"]


def test_p6_cache_key_includes_explicit_override(monkeypatch):
    """Changing the explicit override inside the TTL invalidates the cache."""
    monkeypatch.setattr(model_picker.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(
        model_picker,
        "_http_get_models",
        lambda *_a, **_k: [{"id": "qwen3:8b", "loaded": True, "available": True}],
    )

    automatic = choose_foreground_model(hivemind_url="http://same")
    monkeypatch.setenv(MS4_FOREGROUND_MODEL_ENV, "operator:new")
    overridden = choose_foreground_model(hivemind_url="http://same")

    assert automatic.model_id == "qwen3:8b"
    assert overridden.model_id == "operator:new"
    assert overridden.source == "override"
