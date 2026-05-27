"""Hardware-aware Face Lobe model selection.

The artifact §16 says the Face Lobe should run a small, fast model
(typically 0.5-7B) so it can stay responsive while a heavier Depth
Lobe runs on the same hardware. MS4 picks the foreground model by:

  1. Honoring the explicit operator override
     ``MS4_FOREGROUND_MODEL`` env var if set.
  2. Otherwise consulting the live HiveMind ``/v1/models`` catalog
     and selecting the highest-priority small instruct model that is
     either already ``loaded`` (warm) or at least ``available``.
  3. Otherwise falling back to a small-by-default model id
     (``qwen2.5:0.5b``).

The choice is cached for a short TTL so we don't beat up HiveMind's
``/v1/models`` endpoint on every chat turn. The cache is invalidated
when ``MS4_FOREGROUND_MODEL`` changes or when ``force_refresh=True``.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request
import json
from dataclasses import dataclass
from typing import Any, Iterable


log = logging.getLogger("ms4.double_agent.model_picker")

# NOTE: The Face Lobe no longer goes through the Hermes loop. It calls
# HiveMind's OpenAI-compatible /v1/chat/completions directly, so the
# previous 64K context minimum (a Hermes-specific guard) does not
# apply. We can prefer small/fast models again. The constant is kept
# for historical reference but the priority list below is no longer
# constrained by it.
HERMES_MIN_CONTEXT_TOKENS = 64_000

# Safe-fallback env var. We fall back to MS4_DEFAULT_MODEL (typically
# ``qwen3-coder-next:latest``) when the catalog has nothing that
# matches the priority list. The default is known-warm on this
# cluster.
DEFAULT_FALLBACK_ENV = "MS4_DEFAULT_MODEL"
DEFAULT_FOREGROUND_MODEL = "qwen3-coder-next:latest"
ENV_OVERRIDE = "MS4_FOREGROUND_MODEL"
MODELS_TTL_SECS = 60.0

# Priority order. Models are matched substring-style against the catalog
# entries returned by ``/v1/models`` (case-insensitive). The first priority
# pattern that has at least one matching catalog entry wins; among those
# matches, ``loaded`` beats ``available`` beats unknown.
#
# Sub-2B models (qwen2.5:0.5b, tinyllama:1.1b) are deliberately NOT in
# this list anymore. We observed live that they don't reliably follow
# the Face Lobe system prompt — qwen2.5:0.5b's first turn produced
# "I'm sorry, but I can't assist with that request" for a perfectly
# benign greeting. The Face Lobe is "status and routing focused" per
# artifact §16.5, but it still has to be coherent. A 4B–8B instruct
# model is the right floor; on common modern GPUs it still returns in
# <1s for short turns via direct chat-completion.
#
# Order is "balanced 3–4B instruct" → "8B instruct" → "verified large
# coder" so the auto-picker prefers fast-and-coherent over fastest.
# Operators who specifically want a tiny model can set
# ``MS4_FOREGROUND_MODEL=qwen2.5:0.5b`` to override.
FOREGROUND_PRIORITY_PATTERNS: tuple[str, ...] = (
    # 7B–8B instruct models first. phi4-mini (3.8B) was promoted to #1
    # in May 22's "fast picker" round, but live evidence on May 26 2026
    # showed it can't reliably follow the Face Lobe anti-hallucination
    # contract — the operator hit a turn where the router correctly
    # dispatched a Depth Lobe job but phi4-mini still said "I don't
    # have direct access" and then fabricated fake `hivemind` Python
    # API calls (tool_manager.run_async, ServerSideToolManager).
    # Reordered so a coherent 7-8B model wins by default; phi4-mini
    # stays in the fallback chain for hardware that can't run 8B fast.
    "qwen3:8b",
    "llama3.1:8b",
    "llama3.2:8b",
    "qwen2.5:7b",
    "dolphin-llama3",
    "llama3.1",
    "qwen3",
    "phi4-mini",
    "phi3.5",
    "gemma4",
    "gemma3",
    "gemma2",
    "gemma",
    "qwen3-coder-next:latest",
)


@dataclass
class ForegroundChoice:
    model_id: str
    source: str  # "override" | "loaded" | "available" | "fallback" | "error"
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "Ms4ForegroundModel.v1",
            "model_id": self.model_id,
            "source": self.source,
            "detail": self.detail,
        }


_CACHE_LOCK = threading.Lock()
_CACHE: dict[str, Any] = {"choice": None, "fetched_at": 0.0, "hivemind_url": None, "override": None}


def _http_get_models(hivemind_url: str, timeout: int = 20) -> list[dict[str, Any]]:
    """Fetch HiveMind's full ``/v1/models`` catalog.

    The catalog can have hundreds of entries on a multi-node cluster
    (598 observed live on the operator's setup). The default timeout
    needs to be generous enough that the picker doesn't fall back to
    ``qwen3-coder-next:latest`` (the static fallback) on every cold
    cache miss — otherwise the auto-pick is useless. The result is
    cached for ``MODELS_TTL_SECS`` so this is paid at most once per
    minute per gateway.
    """
    try:
        from ..gateway.hivemind_state import hivemind_auth_headers
    except Exception:
        def hivemind_auth_headers() -> dict[str, str]:  # type: ignore[no-redef]
            return {}

    url = f"{hivemind_url.rstrip('/')}/v1/models"
    headers = {"Accept": "application/json"}
    headers.update(hivemind_auth_headers())
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"models endpoint failed: {exc}") from exc
    data = payload.get("data") if isinstance(payload, dict) else payload
    if isinstance(data, list):
        return [e for e in data if isinstance(e, dict)]
    # Some MS4-proxied responses look like {"models": [...]}
    models = payload.get("models") if isinstance(payload, dict) else None
    if isinstance(models, list):
        return [e for e in models if isinstance(e, dict)]
    return []


# HiveMind's /v1/models response uses ``hivemind_status`` to describe
# readiness, not the OpenAI-style ``loaded`` / ``available`` booleans.
# Observed live values on this cluster:
#   installed      — model files are on a HiveMind node and ready to
#                    serve immediately (this is "warm/loaded" for us).
#   running        — model is actively serving inference right now
#                    (strictly warmer than installed).
#   cloud_ready    — model is a cloud provider (OpenAI/Anthropic/etc.)
#                    with a working key; ready to serve.
#   available      — known to the catalog but NOT installed on any
#                    node; would require cold provisioning before any
#                    call could succeed.
#   cloud_needs_key — cloud provider but missing API key (unusable).
_STATUS_READY = frozenset({"installed", "running", "cloud_ready"})
_STATUS_COLD = frozenset({"available"})


def _normalize_model_entry(entry: dict[str, Any]) -> dict[str, Any]:
    model_id = entry.get("id") or entry.get("model") or entry.get("name") or ""
    status = str(entry.get("hivemind_status") or "").strip().lower()
    reachable = bool(entry.get("hivemind_reachable", True))
    # ``installed/running/cloud_ready`` + reachable = warm-loaded.
    # ``available`` + reachable = cold-but-can-be-loaded.
    # Anything unreachable or with cloud_needs_key is treated as unusable.
    ready = status in _STATUS_READY and reachable
    cold = status in _STATUS_COLD and reachable
    # Backwards compat: if the upstream IS the legacy shape (has
    # ``loaded`` / ``available`` booleans), honour them.
    if "loaded" in entry or status == "":
        ready = bool(entry.get("loaded")) or ready
        cold = (bool(entry.get("available", True)) and not ready) or cold
    return {
        "id": str(model_id),
        "loaded": ready,
        "available": ready or cold,
        "hivemind_status": status,
        "hivemind_reachable": reachable,
    }


def _score(entry: dict[str, Any]) -> tuple[int, int]:
    """Higher tuple wins. loaded > available > unknown."""
    return (1 if entry["loaded"] else 0, 1 if entry["available"] else 0)


def _pick_from_catalog(catalog: list[dict[str, Any]]) -> ForegroundChoice | None:
    """Pick the foreground model from HiveMind's /v1/models catalog.

    Two passes:

      1. **Loaded pass**: walk the priority list and return the FIRST
         priority pattern that has at least one entry with ``loaded:
         true``. A loaded entry is already warm on the GPU — calls
         return in milliseconds and there's no cold-load risk (the
         observed live failure mode was a model in ``available:true /
         loaded:false`` state cold-loading on first call and silently
         returning empty content).
      2. **Available pass**: only if no priority pattern has any loaded
         entry, walk the list again and return the FIRST priority
         pattern with an ``available`` entry. We accept the cold-load
         risk in that case because there's nothing else to pick.

    Without this two-pass structure the picker would happily choose
    ``phi4-mini (available)`` over ``llama3.1:8b (loaded)`` just because
    phi4-mini is higher on the priority list — exactly the bug that
    caused the user-visible "(no reply text)" turns.
    """
    if not catalog:
        return None
    normalized = [_normalize_model_entry(e) for e in catalog]
    normalized = [e for e in normalized if e["id"]]
    if not normalized:
        return None

    def _alias_key(entry: dict[str, Any]) -> tuple[int, int, str]:
        """Sort key that prefers canonical ids over alias-suffixed ones.
        HiveMind exposes ``llama3.1:8b`` and ``llama3.1:8b_ollama`` as
        separate catalog entries with identical readiness; we always
        want the canonical one. (Lower tuple wins under ``sorted``.)"""
        eid = entry["id"]
        elower = eid.lower()
        alias_penalty = 1 if (elower.endswith("_ollama") or elower.endswith("_gim")) else 0
        return (alias_penalty, len(eid), eid)

    # Loaded pass: highest-priority pattern with at least one loaded
    # match wins; among multiple loaded matches the canonical id wins.
    for pattern in FOREGROUND_PRIORITY_PATTERNS:
        plower = pattern.lower()
        loaded_matches = [
            e for e in normalized
            if plower in e["id"].lower() and e["loaded"]
        ]
        if loaded_matches:
            loaded_matches.sort(key=_alias_key)
            winner = loaded_matches[0]
            return ForegroundChoice(
                model_id=winner["id"],
                source="loaded",
                detail=(
                    f"matched priority pattern {pattern!r} "
                    f"(hivemind_status={winner.get('hivemind_status')!r}) "
                    "in HiveMind catalog"
                ),
            )
    # Available pass: no priority pattern has a loaded entry, so we
    # accept a cold-load candidate.
    for pattern in FOREGROUND_PRIORITY_PATTERNS:
        plower = pattern.lower()
        available_matches = [
            e for e in normalized
            if plower in e["id"].lower() and e["available"]
        ]
        if available_matches:
            available_matches.sort(key=_alias_key)
            winner = available_matches[0]
            return ForegroundChoice(
                model_id=winner["id"],
                source="available",
                detail=(
                    f"matched priority pattern {pattern!r} "
                    f"(hivemind_status={winner.get('hivemind_status')!r}, "
                    "not loaded — no loaded priority match) in HiveMind catalog"
                ),
            )
    return None


def _try_hivemind_recommend(hivemind_url: str) -> ForegroundChoice | None:
    """Attempt to resolve the foreground model via
    ``hivemind.models.recommend@v1``.

    Returns a ForegroundChoice with ``source='hivemind.models.recommend'``
    when HiveMind returned a sensible recommendation, ``None`` on any
    failure (missing tool, transport error, empty result). The caller
    falls back to the hand-rolled catalog pick in either case so the
    integration is strictly additive — turning HiveMind's recommend
    tool off cannot break MS4.
    """
    try:
        from machine_spirit_4.gateway import hivemind_tools

        body = hivemind_tools.models_recommend(
            hivemind_url,
            workload="foreground_chat_small",
            constraints={
                # Mirror what FaceLobeChat actually does: a 4-8B
                # instruct-tuned model with tool-calling capable enough
                # to handle our anti-hallucination system prompt.
                "max_size_b": 8,
                "min_size_b": 2,
                "instruct": True,
                "tool_use": True,
            },
        )
    except Exception:
        return None
    if not isinstance(body, dict):
        return None
    candidates = body.get("recommended") or body.get("models") or []
    if not isinstance(candidates, list) or not candidates:
        return None
    top = candidates[0] if isinstance(candidates[0], dict) else None
    if not top:
        return None
    model_id = str(top.get("model_id") or top.get("id") or top.get("model") or "")
    if not model_id:
        return None
    reason = str(top.get("reason") or top.get("score") or "")
    return ForegroundChoice(
        model_id=model_id,
        source="hivemind.models.recommend",
        detail=f"recommend@v1 picked {model_id} ({reason})" if reason else f"recommend@v1 picked {model_id}",
    )


def choose_foreground_model(
    *,
    hivemind_url: str,
    force_refresh: bool = False,
) -> ForegroundChoice:
    """Return the Face Lobe model to use for the next chat turn.

    Cached for ``MODELS_TTL_SECS`` keyed on (hivemind_url, override).
    """
    override = os.environ.get(ENV_OVERRIDE, "").strip() or None
    now = time.time()
    with _CACHE_LOCK:
        cached = _CACHE.get("choice")
        if (
            cached is not None
            and not force_refresh
            and _CACHE.get("hivemind_url") == hivemind_url
            and _CACHE.get("override") == override
            and (now - _CACHE.get("fetched_at", 0.0)) < MODELS_TTL_SECS
        ):
            return cached

    fallback_model = os.environ.get(DEFAULT_FALLBACK_ENV, "").strip() or DEFAULT_FOREGROUND_MODEL
    if override:
        choice = ForegroundChoice(
            model_id=override,
            source="override",
            detail=f"MS4_FOREGROUND_MODEL env var set",
        )
    else:
        # Pass 1: ask HiveMind directly via hivemind.models.recommend@v1.
        # Cluster-aware recommendation beats our hand-rolled priority
        # list when HiveMind has visibility we don't (cold/hot model
        # state, current load, VRAM headroom). Best-effort — any failure
        # falls through to the local pick below.
        choice = _try_hivemind_recommend(hivemind_url)
        if choice is None:
            try:
                catalog = _http_get_models(hivemind_url)
                choice = _pick_from_catalog(catalog)
                if choice is None:
                    choice = ForegroundChoice(
                        model_id=fallback_model,
                        source="fallback",
                        detail="no matching priority pattern in HiveMind catalog",
                    )
            except RuntimeError as exc:
                choice = ForegroundChoice(
                    model_id=fallback_model,
                    source="error",
                    detail=str(exc),
                )

    with _CACHE_LOCK:
        _CACHE["choice"] = choice
        _CACHE["fetched_at"] = now
        _CACHE["hivemind_url"] = hivemind_url
        _CACHE["override"] = override
    return choice


def _clear_cache_for_tests() -> None:
    with _CACHE_LOCK:
        _CACHE["choice"] = None
        _CACHE["fetched_at"] = 0.0
        _CACHE["hivemind_url"] = None
        _CACHE["override"] = None
