"""Depth Lobe model picker.

Mirrors :mod:`machine_spirit_4.double_agent.model_picker` (foreground)
but biased toward bigger, smarter coder/reasoning models. Used when
the auto-router dispatches a Depth Lobe job and the envelope's
``resource_request.model_override`` is unset.

Precedence (highest first):

1. ``resource_request.model_override`` on the envelope (per-job pin).
2. ``MS4_DEPTH_MODEL`` env override.
3. Auto-pick the highest-priority large coder/reasoning model that is
   ``loaded`` (warm) in HiveMind's ``/v1/models`` catalog.
4. ``MS4_DEFAULT_MODEL`` (currently ``qwen3-coder-next:latest``).

Same caching shape as the foreground picker so we don't hammer
``/v1/models``.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Any

from .model_picker import _http_get_models, _normalize_model_entry, _score


ENV_OVERRIDE = "MS4_DEPTH_MODEL"
DEFAULT_DEPTH_MODEL_ENV = "MS4_DEFAULT_MODEL"
HARDCODED_DEFAULT = "qwen3-coder-next:latest"
MODELS_TTL_SECS = 60.0

# Priority order. Matched substring-style against HiveMind catalog entries
# (case-insensitive). The first priority pattern that has at least one
# matching catalog entry wins; among those matches, ``loaded`` beats
# ``available`` beats unknown.
DEPTH_PRIORITY_PATTERNS: tuple[str, ...] = (
    "qwen3-coder-next:latest",
    "qwen3-coder",
    "qwen2.5-coder:32b",
    "qwen2.5-coder",
    "qwen3-next",
    "qwen3.6:27b",
    "qwen3.5",
    "deepseek-v3",
    "deepseek-r1",
    "deepseek-coder-v2",
    "deepseek-coder",
    "codestral",
    "phi4-reasoning",
    "phi4",
    "qwen3-8b",
    "qwen3",
    "llama-3.3-70b",
    "llama-3.1-70b",
    "mistral",
)

# Coder-class ordering: strictly code-specialist families first, then
# the general deep set as fallback. Used when the job envelope carries
# ``resource_request.model_class == 'deep_coder'``.
DEPTH_CODER_PRIORITY_PATTERNS: tuple[str, ...] = (
    "qwen3-coder-next:latest",
    "qwen3-coder",
    "qwen2.5-coder:32b",
    "qwen2.5-coder",
    "deepseek-coder-v2",
    "deepseek-coder",
    "codestral",
    "deepseek-v3",
    "deepseek-r1",
    "qwen3-next",
    "qwen3.6:27b",
    "qwen3.5",
    "phi4-reasoning",
    "phi4",
    "qwen3-8b",
    "qwen3",
    "llama-3.3-70b",
    "llama-3.1-70b",
    "mistral",
)

# Reasoning-class ordering: reasoning/general families ahead of the
# coder specialists. Used for ``model_class == 'deep_reasoning'`` —
# which is also the schema default, so this MUST stay behaviorally
# compatible with the legacy DEPTH_PRIORITY_PATTERNS pick for the
# catalogs this cluster actually serves (the coder-next family stays
# first because it IS the cluster's strongest reasoner today).
DEPTH_REASONING_PRIORITY_PATTERNS: tuple[str, ...] = DEPTH_PRIORITY_PATTERNS

# ``resource_request.model_class`` -> priority patterns. Unknown / None
# classes use the default depth ordering (legacy behavior). The phase-4
# HiveMind capability-lease routing will eventually replace this table;
# until then it is the honest, local interpretation of model_class.
CLASS_PRIORITY_PATTERNS: dict[str, tuple[str, ...]] = {
    "deep_reasoning": DEPTH_REASONING_PRIORITY_PATTERNS,
    "deep_coder": DEPTH_CODER_PRIORITY_PATTERNS,
}


@dataclass
class DepthChoice:
    model_id: str
    source: str  # "envelope_override" | "env_override" | "loaded" | "available" | "fallback" | "error"
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "Ms4DepthModel.v1",
            "model_id": self.model_id,
            "source": self.source,
            "detail": self.detail,
        }


_CACHE_LOCK = threading.Lock()
_CACHE: dict[str, Any] = {
    "choice": None,
    "fetched_at": 0.0,
    "hivemind_url": None,
    "override": None,
    "model_class": None,
}


def _hardcoded_default() -> str:
    return os.environ.get(DEFAULT_DEPTH_MODEL_ENV, HARDCODED_DEFAULT)


def _patterns_for_class(model_class: str | None) -> tuple[str, ...]:
    if not model_class:
        return DEPTH_PRIORITY_PATTERNS
    return CLASS_PRIORITY_PATTERNS.get(
        str(model_class).strip().lower(), DEPTH_PRIORITY_PATTERNS
    )


def _pick_from_catalog(
    catalog: list[dict[str, Any]],
    patterns: tuple[str, ...] = DEPTH_PRIORITY_PATTERNS,
) -> DepthChoice | None:
    if not catalog:
        return None
    normalized = [_normalize_model_entry(e) for e in catalog]
    normalized = [e for e in normalized if e["id"]]
    if not normalized:
        return None
    for pattern in patterns:
        plower = pattern.lower()
        matches = [e for e in normalized if plower in e["id"].lower()]
        if not matches:
            continue
        matches.sort(key=_score, reverse=True)
        winner = matches[0]
        source = "loaded" if winner["loaded"] else ("available" if winner["available"] else "fallback")
        return DepthChoice(
            model_id=winner["id"],
            source=source,
            detail=f"matched depth priority pattern {pattern!r}",
        )
    return None


def choose_depth_model(
    *,
    hivemind_url: str,
    envelope_override: str | None = None,
    model_class: str | None = None,
    force_refresh: bool = False,
) -> DepthChoice:
    """Pick the Depth Lobe model for the next dispatched job.

    ``model_class`` (from ``resource_request.model_class``) selects the
    priority-pattern table: ``deep_coder`` biases code-specialist
    families to the front, ``deep_reasoning`` (the schema default) and
    unknown/None classes keep the legacy depth ordering. The catalog
    fetch is cached for ``MODELS_TTL_SECS`` keyed on
    (hivemind_url, env_override, model_class). The envelope_override
    path is never cached (it's per-job).
    """
    if envelope_override and envelope_override.strip():
        return DepthChoice(
            model_id=envelope_override.strip(),
            source="envelope_override",
            detail="job envelope resource_request.model_override",
        )

    env_override = os.environ.get(ENV_OVERRIDE, "").strip() or None
    normalized_class = str(model_class).strip().lower() if model_class else None
    now = time.time()
    with _CACHE_LOCK:
        cached = _CACHE.get("choice")
        if (
            cached is not None
            and not force_refresh
            and _CACHE.get("hivemind_url") == hivemind_url
            and _CACHE.get("override") == env_override
            and _CACHE.get("model_class") == normalized_class
            and (now - _CACHE.get("fetched_at", 0.0)) < MODELS_TTL_SECS
        ):
            return cached

    if env_override:
        choice = DepthChoice(
            model_id=env_override,
            source="env_override",
            detail=f"{ENV_OVERRIDE} env var set",
        )
    else:
        try:
            catalog = _http_get_models(hivemind_url)
            choice = _pick_from_catalog(catalog, _patterns_for_class(normalized_class))
            if choice is None:
                choice = DepthChoice(
                    model_id=_hardcoded_default(),
                    source="fallback",
                    detail="no matching depth priority pattern in HiveMind catalog",
                )
        except RuntimeError as exc:
            choice = DepthChoice(
                model_id=_hardcoded_default(),
                source="error",
                detail=str(exc),
            )

    with _CACHE_LOCK:
        _CACHE["choice"] = choice
        _CACHE["fetched_at"] = now
        _CACHE["hivemind_url"] = hivemind_url
        _CACHE["override"] = env_override
        _CACHE["model_class"] = normalized_class
    return choice


def _clear_cache_for_tests() -> None:
    with _CACHE_LOCK:
        _CACHE["choice"] = None
        _CACHE["fetched_at"] = 0.0
        _CACHE["hivemind_url"] = None
        _CACHE["override"] = None
        _CACHE["model_class"] = None
