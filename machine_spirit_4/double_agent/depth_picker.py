"""Depth Lobe model picker.

Mirrors :mod:`machine_spirit_4.double_agent.model_picker` (foreground)
but biased toward bigger, smarter coder/reasoning models. Used when
the auto-router dispatches a Depth Lobe job and the envelope's
``resource_request.model_override`` is unset.

Precedence (highest first):

1. ``resource_request.model_override`` on the envelope (per-job pin).
2. ``MS4_DEPTH_MODEL`` env override.
3. Auto-pick the highest-priority quality-floor model that is reachable from
   HiveMind's cluster-wide ``/v1/models`` catalog.
4. The separately gated ``MS4_DEPTH_FALLBACK_MODEL`` (built-in:
   ``nemotron-3-nano:30b``).

Same caching shape as the foreground picker so we don't hammer
``/v1/models``.
"""

from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from .model_picker import _http_get_models, _is_chat_capable, _normalize_model_entry


ENV_OVERRIDE = "MS4_DEPTH_MODEL"
DEFAULT_DEPTH_MODEL_ENV = "MS4_DEPTH_FALLBACK_MODEL"
DEPTH_FAST_TOOL_FALLBACK_MODEL = "nemotron-3-nano:30b"
HARDCODED_DEFAULT = DEPTH_FAST_TOOL_FALLBACK_MODEL
DEPTH_PREFERRED_CLUSTER_TARGET = "qwen3.6:35b"
DEPTH_MIN_TOTAL_PARAM_B = 35
DEPTH_FALLBACK_MIN_TOTAL_PARAM_B = 30
MODELS_TTL_SECS = 60.0

# Priority order. Matched substring-style against HiveMind catalog entries
# (case-insensitive), then constrained by ``DEPTH_MIN_TOTAL_PARAM_B``. Current
# cluster checks justify Qwen 35B as the preferred installed/reachable target
# and tool-capable Nemotron 30B as its automatic fallback. Gemma 31B exceeded
# the bounded latency gates without reaching a tool call, so it remains
# available only through an explicit operator/job override. The readiness gate
# below keeps a truly absent/unreachable higher-priority model from displacing
# a model HiveMind can actually serve now.
#
# The catalog is cluster-wide: no local-GPU or co-residency assumption belongs
# in this picker. A 16 GiB node is a minimum fleet tier, not a requirement that
# Face and Depth weights fit together on the MS4 host.
DEPTH_PRIORITY_PATTERNS: tuple[str, ...] = (
    DEPTH_PREFERRED_CLUSTER_TARGET,
    "qwen3-coder",
    "qwen2.5-coder",
    "qwen3-next",
    "qwen3.6",
    "qwen3.5",
    "deepseek-v3",
    "deepseek-r1",
    "deepseek-coder-v2",
    "deepseek-coder",
    "codestral",
    "llama-3.3-70b",
    "llama-3.1-70b",
    "mistral",
)

# Coder-class ordering: strictly code-specialist families first, then
# the general deep set as fallback. Used when the job envelope carries
# ``resource_request.model_class == 'deep_coder'``.
DEPTH_CODER_PRIORITY_PATTERNS: tuple[str, ...] = (
    "qwen3-coder",
    "qwen2.5-coder",
    DEPTH_PREFERRED_CLUSTER_TARGET,
    "deepseek-coder-v2",
    "deepseek-coder",
    "codestral",
    "deepseek-v3",
    "deepseek-r1",
    "qwen3-next",
    "qwen3.6",
    "qwen3.5",
    "llama-3.3-70b",
    "llama-3.1-70b",
    "mistral",
)

# Reasoning-class ordering: reasoning/general families ahead of the
# coder specialists. Used for ``model_class == 'deep_reasoning'`` —
# which is also the schema default, so this MUST stay behaviorally
# compatible with the legacy DEPTH_PRIORITY_PATTERNS pick for the
# catalogs this cluster actually serves.
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
    source: str  # "envelope_override" | "env_override" | "loaded" | "fallback" | "error"
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
    "fallback": None,
}


_DEPTH_SIZE_TAG_RE = re.compile(
    r"(?:^|[:/@_-])(\d+(?:\.\d+)?)b(?=$|[-_])",
    re.IGNORECASE,
)


def _declared_total_params_b(model_id: str) -> Fraction | None:
    """Read an explicit total-parameter size from a catalog model id.

    This intentionally does not infer size from mutable aliases such as
    ``:latest``. Sparse models such as ``30B-A3B`` are classified by total
    resident weights (30B), because active parameters describe per-token
    compute rather than the Depth quality/loadout tier requested here.
    """
    if not isinstance(model_id, str) or not model_id or not model_id.isascii():
        return None
    normalized = model_id.strip().lower()
    for suffix in ("_ollama", "_gim"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    values = [Fraction(match) for match in _DEPTH_SIZE_TAG_RE.findall(normalized)]
    return max(values) if values else None


def _meets_depth_quality_floor(model_id: str) -> bool:
    size_b = _declared_total_params_b(model_id)
    return size_b is not None and size_b >= DEPTH_MIN_TOTAL_PARAM_B


def _meets_depth_fallback_floor(model_id: str) -> bool:
    size_b = _declared_total_params_b(model_id)
    return size_b is not None and size_b >= DEPTH_FALLBACK_MIN_TOTAL_PARAM_B


def depth_fallback_model() -> str:
    """Return a quality-floor-safe automatic Depth fallback.

    ``MS4_DEPTH_MODEL`` remains the explicit operator override and may name any
    model. The fallback is deliberately a degraded tier below the 35B quality
    target, but a configured fallback must still clear the separate 30B safety
    floor instead of silently turning the Depth Lobe into another Face Lobe.
    """
    configured = os.environ.get(DEFAULT_DEPTH_MODEL_ENV, "").strip()
    if configured and _meets_depth_fallback_floor(configured):
        return configured
    return HARDCODED_DEFAULT


def _hardcoded_default() -> str:
    """Backward-compatible private name for the effective Depth fallback."""
    return depth_fallback_model()


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
    normalized = [
        _normalize_model_entry(entry)
        for entry in catalog
        if _is_chat_capable(entry)
    ]
    normalized = [
        entry
        for entry in normalized
        if entry["id"] and _meets_depth_quality_floor(entry["id"])
    ]
    if not normalized:
        return None

    def _alias_key(entry: dict[str, Any]) -> tuple[int, int, str]:
        eid = entry["id"]
        elower = eid.lower()
        alias_penalty = 1 if (elower.endswith("_ollama") or elower.endswith("_gim")) else 0
        return (alias_penalty, len(eid), eid)

    for pattern in patterns:
        plower = pattern.lower()
        # ``loaded`` is the normalized cluster-readiness bit. In particular,
        # HLI ``hivemind_status=installed`` + ``hivemind_reachable=true`` is
        # eligible even when a provider-local /api/tags snapshot omits an
        # idle-unloaded model; HiveMind can warm and route that installed peer.
        loaded_matches = [
            e for e in normalized
            if plower in e["id"].lower() and e["loaded"]
        ]
        if loaded_matches:
            loaded_matches.sort(key=_alias_key)
            winner = loaded_matches[0]
            return DepthChoice(
                model_id=winner["id"],
                source="loaded",
                detail=(
                    f"matched cluster-ready depth priority pattern {pattern!r}; "
                    f"hivemind_status={winner.get('hivemind_status')!r}, "
                    f"reachable={winner.get('hivemind_reachable')!r}; "
                    f"declared total parameters >= {DEPTH_MIN_TOTAL_PARAM_B}B"
                ),
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
    (hivemind_url, env_override, model_class, depth_fallback). The envelope_override
    path is never cached (it's per-job).
    """
    if envelope_override and envelope_override.strip():
        return DepthChoice(
            model_id=envelope_override.strip(),
            source="envelope_override",
            detail="job envelope resource_request.model_override",
        )

    env_override = os.environ.get(ENV_OVERRIDE, "").strip() or None
    fallback_raw = os.environ.get(DEFAULT_DEPTH_MODEL_ENV, "").strip()
    normalized_class = str(model_class).strip().lower() if model_class else None
    now = time.monotonic()
    with _CACHE_LOCK:
        cached = _CACHE.get("choice")
        if (
            cached is not None
            and not force_refresh
            and _CACHE.get("hivemind_url") == hivemind_url
            and _CACHE.get("override") == env_override
            and _CACHE.get("model_class") == normalized_class
            and _CACHE.get("fallback") == fallback_raw
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
                    detail=(
                        "no loaded, reachable Depth model met the automatic "
                        f">={DEPTH_MIN_TOTAL_PARAM_B}B quality floor; using "
                        "the separately gated degraded fast/tool fallback"
                    ),
                )
        except RuntimeError as exc:
            choice = DepthChoice(
                model_id=_hardcoded_default(),
                source="error",
                detail=(
                    f"{exc}; using the separately gated degraded fast/tool "
                    "fallback"
                ),
            )

    with _CACHE_LOCK:
        _CACHE["choice"] = choice
        _CACHE["fetched_at"] = now
        _CACHE["hivemind_url"] = hivemind_url
        _CACHE["override"] = env_override
        _CACHE["model_class"] = normalized_class
        _CACHE["fallback"] = fallback_raw
    return choice


def _clear_cache_for_tests() -> None:
    with _CACHE_LOCK:
        _CACHE["choice"] = None
        _CACHE["fetched_at"] = 0.0
        _CACHE["hivemind_url"] = None
        _CACHE["override"] = None
        _CACHE["model_class"] = None
        _CACHE["fallback"] = None
