"""Cluster-aware Face Lobe model selection.

The artifact §16 says the Face Lobe should run a small, fast model
so it can stay responsive while a heavier Depth Lobe is served elsewhere
in the HiveMind cluster. MS4 picks the foreground model by:

  1. Honoring the explicit operator override
     ``MS4_FOREGROUND_MODEL`` env var if set.
  2. Otherwise consulting the live HiveMind ``/v1/models`` catalog
     and selecting the highest-priority small instruct model that is
     already ``loaded`` (warm), reachable, and eligible.
  3. Otherwise failing fast rather than cold-loading an ``available``
     model during the foreground turn.

The choice is cached for a short (monotonic-clock) TTL so we don't beat
up HiveMind's ``/v1/models`` endpoint on every chat turn. The cache key
includes ``MS4_FOREGROUND_MODEL`` (override) AND ``MS4_DEFAULT_MODEL``
(the gated fallback), and a monotonic generation guards commit-ordering,
so the cache is re-derived when either env var changes or when
``force_refresh=True`` -- and an older in-flight call can never overwrite
a newer committed result.
"""

from __future__ import annotations

import logging
import math
import os
import re
import threading
import time
import urllib.error
import urllib.request
import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal
from fractions import Fraction
from typing import Any, Iterable


log = logging.getLogger("ms4.double_agent.model_picker")

# NOTE: The Face Lobe no longer goes through the Hermes loop. It calls
# HiveMind's OpenAI-compatible /v1/chat/completions directly, so the
# previous 64K context minimum (a Hermes-specific guard) does not
# apply. We can prefer small/fast models again. The constant is kept
# for historical reference but the priority list below is no longer
# constrained by it.
HERMES_MIN_CONTEXT_TOKENS = 64_000

# Safe-fallback env var. We fall back to MS4_DEFAULT_MODEL when the
# catalog has nothing that matches the priority list. The built-in
# default is a coherent fast-chat Face Lobe model; heavy coder models
# belong in the Depth Lobe.
DEFAULT_FALLBACK_ENV = "MS4_DEFAULT_MODEL"
DEFAULT_FOREGROUND_MODEL = "nemotron-3-nano:4b"
ENV_OVERRIDE = "MS4_FOREGROUND_MODEL"
FACE_PROFILE_ENV = "MS4_FACE_PROFILE"
HIGH_MEMORY_QUALITY_MODEL = "qwen3.6:35b"
HIGH_MEMORY_MIN_VRAM_MIB = 32 * 1024
MODELS_TTL_SECS = 60.0
# Exact Face warm→probe consumes the live HLI recommend tool identity as its
# schema/version discriminator. Legacy scalar recommend@v1 bodies without this
# typed envelope remain ordinary picker input and cannot satisfy exact admission.
EXACT_FACE_CLUSTER_ADMISSION_SCHEMA = "hivemind.models.recommend@v1"
EXACT_FACE_CLUSTER_ADMISSION_TTL_S = MODELS_TTL_SECS

# Priority order. Models are matched substring-style against the catalog
# entries returned by ``/v1/models`` (case-insensitive). The first priority
# pattern that has at least one matching catalog entry wins; among those
# matches, only a loaded/reachable entry is eligible.
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
    # Live Face-contract validation on the 5090 node established this as the
    # preferred fast/coherent cluster-served model. Placement is HiveMind's
    # concern: the MS4 host does not need to have the model installed locally.
    "nemotron-3-nano:4b",
    # 7B–8B instruct models follow the verified Nemotron default. phi4-mini
    # (3.8B) was promoted to #1
    # in May 22's "fast picker" round, but live evidence on May 26 2026
    # showed it can't reliably follow the Face Lobe anti-hallucination
    # contract — the operator hit a turn where the router correctly
    # dispatched a Depth Lobe job but phi4-mini still said "I don't
    # have direct access" and then fabricated fake `hivemind` Python
    # API calls (tool_manager.run_async, ServerSideToolManager).
    # Reordered so a coherent 7-8B model remains the next choice; phi4-mini
    # stays in the fallback chain for clusters without the preferred model.
    "qwen3:8b",
    "llama3.1:8b",
    "llama3.2:8b",
    "qwen2.5:7b",
    "dolphin-llama3",
    "llama3.1",
    "qwen3:",
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
    catalog_confirmation_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "Ms4ForegroundModel.v1",
            "model_id": self.model_id,
            "source": self.source,
            "detail": self.detail,
        }


class ForegroundModelUnavailable(RuntimeError):
    """No loaded, reachable, eligible automatic foreground model exists."""


_CACHE_LOCK = threading.Lock()
_CACHE: dict[str, Any] = {
    "choice": None,
    "fetched_at": 0.0,
    "hivemind_url": None,
    "override": None,
    "env_default": None,
    "face_profile": None,
    "gen": 0,
}
# R2.4-P4 (Sagan P2-2): a monotonically increasing generation assigned to each
# call under _CACHE_LOCK. A commit only wins if its generation is >= the last
# committed one, so a slow older computation cannot clobber a newer result.
_NEXT_GEN: list[int] = [0]


def _http_get_models(hivemind_url: str, timeout: int = 20) -> list[dict[str, Any]]:
    """Fetch HiveMind's full ``/v1/models`` catalog.

    The catalog can have hundreds of entries on a multi-node cluster
    (598 observed live on the operator's setup). The default timeout
    needs to be generous enough that the picker doesn't fall back to
    the static fallback on every cold
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


def _http_get_carrier_nodes(
    hivemind_url: str,
    timeout: int = 10,
) -> list[dict[str, Any]]:
    """Fetch the read-only carrier hardware roster used by quality profile.

    Model readiness and hardware capacity are separate facts: ``/v1/models``
    proves Qwen is installed/reachable, while carrier-sync proves the serving
    node has enough VRAM. Failure is fail-soft; callers retain the latency
    profile instead of guessing capacity or provisioning anything.
    """
    try:
        from ..gateway.hivemind_state import hivemind_auth_headers
    except Exception:
        def hivemind_auth_headers() -> dict[str, str]:  # type: ignore[no-redef]
            return {}

    url = f"{hivemind_url.rstrip('/')}/api/v1/carrier_sync/nodes"
    headers = {"Accept": "application/json"}
    headers.update(hivemind_auth_headers())
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"carrier hardware endpoint failed: {exc}") from exc
    nodes = payload.get("nodes") if isinstance(payload, dict) else payload
    return [node for node in nodes if isinstance(node, dict)] if isinstance(nodes, list) else []


def _face_profile() -> str:
    value = os.environ.get(FACE_PROFILE_ENV, "auto").strip().lower()
    return value if value in {"latency", "auto", "quality"} else "latency"


def _pick_high_memory_quality_model(
    catalog: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
) -> ForegroundChoice | None:
    """Select exact Qwen 3.6 35B only with model + >=32 GiB node proof."""
    candidates = []
    for entry in catalog:
        if not _is_chat_capable(entry):
            continue
        normalized = _normalize_model_entry(entry)
        canonical_id = normalized["id"].lower().removesuffix("_ollama")
        if canonical_id != HIGH_MEMORY_QUALITY_MODEL or not normalized["loaded"]:
            continue
        candidates.append(entry)
    if not candidates:
        return None

    for entry in candidates:
        node_id = str(entry.get("hivemind_node_id") or "").strip()
        reachable = {
            str(value).strip()
            for value in (entry.get("hivemind_reachable_nodes") or [])
            if str(value).strip()
        }
        for node in nodes:
            node_matches = bool(node_id and str(node.get("node_id") or "") == node_id)
            address_matches = str(node.get("lan_ip") or "").strip() in reachable
            if not (node_matches or address_matches):
                continue
            max_vram_mib = max(
                (
                    int(gpu.get("vram_total_mb") or 0)
                    for gpu in (node.get("gpus") or [])
                    if isinstance(gpu, dict)
                ),
                default=0,
            )
            if max_vram_mib >= HIGH_MEMORY_MIN_VRAM_MIB:
                return ForegroundChoice(
                    model_id=HIGH_MEMORY_QUALITY_MODEL,
                    source="loaded",
                    detail=(
                        "high-memory Face quality profile selected installed/reachable "
                        f"{HIGH_MEMORY_QUALITY_MODEL} on node {node_id or node.get('node_id')} "
                        f"with max_vram_mib={max_vram_mib}"
                    ),
                )
    return None


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

_NON_CHAT_METADATA_TOKENS = frozenset({
    "asr",
    "audio",
    "audio-generation",
    "classification",
    "diarization",
    "embedding",
    "embeddings",
    "image",
    "image-gen",
    "image-generation",
    "image-to-image",
    "moderation",
    "nmt",
    "object-detection",
    "ocr",
    "od",
    "osd",
    "rerank",
    "sentence-embedding",
    "speech",
    "speech-to-text",
    "stt",
    "text-embedding",
    "text-embeddings",
    "text-to-image",
    "text-to-speech",
    "transcribe",
    "transcription",
    "translation",
    "tts",
    "vad",
    "video",
    "vision",
    "vlm",
    "whisper",
})
_CHAT_METADATA_TOKENS = frozenset({
    "chat",
    "chat-completion",
    "chat-completions",
    "completion",
    "completions",
    "conversation",
    "conversational",
    "text-generation",
})
_CATEGORY_METADATA_KEYS = (
    "hivemind_category",
    "category",
)
_CAPABILITY_METADATA_KEYS = (
    "hivemind_capability",
    "hivemind_capabilities",
    "capability",
    "capabilities",
)
_OTHER_METADATA_KEYS = (
    "task",
    "tasks",
    "modality",
    "modalities",
    "input_modality",
    "input_modalities",
    "output_modality",
    "output_modalities",
    "type",
    "types",
)
_RECOGNIZED_METADATA_KEYS = (
    *_CATEGORY_METADATA_KEYS,
    *_CAPABILITY_METADATA_KEYS,
    *_OTHER_METADATA_KEYS,
)
_NON_CHAT_ID_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"(^|[-_:])asr($|[-_:])",
        r"(^|[-_:])bark($|[-_:])",
        r"(^|[-_:])diar",
        r"(^|[-_:])embed",
        r"(^|[-_:])flux($|[-_:])",
        r"(^|[-_:])image($|[-_:])",
        r"(^|[-_:])kokoro($|[-_:])",
        r"(^|[-_:])moderation($|[-_:])",
        r"(^|[-_:])nmt($|[-_:])",
        r"(^|[-_:])ocr($|[-_:])",
        r"(^|[-_:])osd($|[-_:])",
        r"(^|[-_:])parler($|[-_:])",
        r"(^|[-_:])piper($|[-_:])",
        r"(^|[-_:])rerank($|[-_:])",
        r"(^|[-_:])sd($|[-_:])",
        r"(^|[-_:])speech($|[-_:])",
        r"(^|[-_:])stt($|[-_:])",
        r"(^|[-_:])tts($|[-_:])",
        r"(^|[-_:])vad($|[-_:])",
        r"(^|[-_:])vision($|[-_:])",
        r"(^|[-_:])vl($|[-_:])",
        r"(^|[-_:])vlm($|[-_:])",
        r"(^|[-_:])whisper($|[-_:])",
        r"bge[-_:]",
        r"nomic[-_:]embed",
        r"stable[-_:]?diffusion",
        r"xtts",
    )
)


def _flatten_metadata_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        out: list[str] = []
        for key, item in value.items():
            if item is True:
                out.append(str(key))
            elif item not in (False, None, ""):
                out.append(str(key))
                out.extend(_flatten_metadata_value(item))
        return out
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        out: list[str] = []
        for item in value:
            out.extend(_flatten_metadata_value(item))
        return out
    return [str(value)]


def _metadata_values(entry: dict[str, Any], keys: Iterable[str]) -> list[str]:
    values: list[str] = []
    for key in keys:
        values.extend(_flatten_metadata_value(entry.get(key)))
    details = entry.get("details")
    if isinstance(details, dict):
        for key in keys:
            values.extend(_flatten_metadata_value(details.get(key)))
    return values


def _metadata_tokens(entry: dict[str, Any], keys: Iterable[str]) -> set[str]:
    tokens: set[str] = set()
    for raw in _metadata_values(entry, keys):
        normalized = str(raw).strip().lower().replace("_", "-")
        if not normalized:
            continue
        tokens.add(normalized)
        tokens.update(part for part in re.split(r"[\s,/|]+", normalized) if part)
    return tokens


def _is_chat_capable(entry: dict[str, Any]) -> bool:
    """Return False for catalog entries that are clearly not text-chat models."""
    model_id = str(
        entry.get("recommended_model")
        or entry.get("model_id")
        or entry.get("id")
        or entry.get("model")
        or entry.get("name")
        or ""
    ).strip().lower()
    if not model_id:
        return False
    category_tokens = _metadata_tokens(entry, _CATEGORY_METADATA_KEYS)
    if category_tokens & _NON_CHAT_METADATA_TOKENS:
        return False
    capability_tokens = _metadata_tokens(entry, _CAPABILITY_METADATA_KEYS)
    if capability_tokens & _CHAT_METADATA_TOKENS:
        return True
    if capability_tokens & _NON_CHAT_METADATA_TOKENS:
        return False
    other_tokens = _metadata_tokens(entry, _OTHER_METADATA_KEYS)
    if other_tokens & _NON_CHAT_METADATA_TOKENS:
        return False
    return not any(pattern.search(model_id) for pattern in _NON_CHAT_ID_PATTERNS)


def _normalize_model_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Ordinary catalog policy. This is not exact warm→probe admission.

    HiveMind-shaped rows use ``hivemind_status`` ∈ {installed, running,
    cloud_ready} plus reachable as loaded. Legacy OpenAI booleans still
    work. A strict OpenAI name row (id/object/created/owned_by only) is
    catalog-cold: it may drive non-exact Face policy / fallback, but it
    cannot satisfy typed exact-admission.
    """
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
        ready = (bool(entry.get("loaded")) and reachable) or ready
        cold = (bool(entry.get("available", True)) and reachable and not ready) or cold
    normalized = {
        "id": str(model_id),
        "loaded": ready,
        "available": ready or cold,
        "hivemind_status": status,
        "hivemind_reachable": reachable,
    }
    for key in _RECOGNIZED_METADATA_KEYS:
        if key in entry:
            normalized[key] = entry[key]
    if "details" in entry:
        normalized["details"] = entry["details"]
    return normalized


def _score(entry: dict[str, Any]) -> tuple[int, int]:
    """Higher tuple wins. loaded > available > unknown."""
    return (1 if entry["loaded"] else 0, 1 if entry["available"] else 0)


def _pick_from_catalog(catalog: list[dict[str, Any]]) -> ForegroundChoice | None:
    """Pick the foreground model from HiveMind's /v1/models catalog.

    Walk the priority list and return the first loaded/reachable eligible
    match. An ``available`` entry is cold and must never be selected during
    an interactive foreground turn.
    """
    if not catalog:
        return None
    normalized = [
        _normalize_model_entry(entry)
        for entry in catalog
        if _is_chat_capable(entry)
    ]
    normalized = [e for e in normalized if e["id"]]
    if not normalized:
        return None

    # R2.1: enforce the foreground contract (family match AND parsed size <=
    # MAX_FOREGROUND_PARAM_B) on catalog candidates too, so this picker can
    # never return an explicit over-ceiling variant (e.g. qwen3:235b, an 8x7b
    # MoE, or a suffixed llama3.1:70b.gguf) even when it is the only match.
    # When nothing contract-compliant remains, both the off-contract reconcile
    # path and the ordinary no-recommend path fall to the safe fallback/default.
    normalized = [e for e in normalized if _satisfies_foreground_contract(e["id"])]
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
    raise ForegroundModelUnavailable(
        "no loaded, reachable, eligible foreground model in HiveMind catalog"
    )


def _nonempty_str(value: Any) -> str:
    return value.strip() if isinstance(value, str) and value.strip() else ""


def _finite_nonnegative_s(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        return None
    return number


def _recommendation_snapshot_age_s(payload: dict[str, Any]) -> float | None:
    if "snapshot_age_s" in payload:
        return _finite_nonnegative_s(payload.get("snapshot_age_s"))
    if "snapshot_age_ms" in payload:
        value = _finite_nonnegative_s(payload.get("snapshot_age_ms"))
        return None if value is None else value / 1000.0
    if "snapshot_age" in payload:
        return _finite_nonnegative_s(payload.get("snapshot_age"))
    return None


def _as_epoch_s(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    text = _nonempty_str(value)
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _positive_generation(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value


def _recommendation_generation(payload: dict[str, Any]) -> int | None:
    for key in ("generation", "snapshot_generation"):
        parsed = _positive_generation(payload.get(key))
        if parsed is not None:
            return parsed
    return None


def _admission_lease(payload: dict[str, Any]) -> dict[str, Any] | None:
    lease = payload.get("lease")
    if not isinstance(lease, dict) or not lease:
        return None
    identity = (
        _nonempty_str(lease.get("id"))
        or _nonempty_str(lease.get("lease_id"))
        or _nonempty_str(lease.get("affinity"))
        or _nonempty_str(lease.get("affinity_id"))
    )
    issued = _as_epoch_s(lease.get("issued") or lease.get("issued_at"))
    expires = _as_epoch_s(lease.get("expires") or lease.get("expires_at"))
    generation = _recommendation_generation(lease)
    if not identity or issued is None or expires is None or generation is None:
        return None
    return {
        "id": identity,
        "issued": issued,
        "expires": expires,
        "generation": generation,
    }


def admit_cluster_recommendation(
    payload: Any,
    *,
    catalog_ids: set[str] | None = None,
    requested_model: str | None = None,
    expected_owner: str | None = None,
    expected_endpoint: str | None = None,
    current_generation: int | None = None,
    now_s: float | None = None,
    ttl_s: float | None = None,
) -> dict[str, Any]:
    """Exact Face warm→probe gate. Not a replacement for ordinary picking.

    Required typed evidence: the live HLI recommend schema/version discriminator,
    finite nonnegative snapshot age within the explicit TTL, exact
    model/backend/ready/owner/endpoint, and a revalidatable lease/affinity
    identity (issued/expires/current-generation). Catalog name rows never
    confer this receipt.
    """
    if not isinstance(payload, dict) or not payload:
        raise ForegroundModelUnavailable("cluster recommendation admission absent")
    schema = _nonempty_str(payload.get("schema"))
    if not schema:
        raise ForegroundModelUnavailable("cluster recommendation admission absent")
    if schema != EXACT_FACE_CLUSTER_ADMISSION_SCHEMA:
        raise ForegroundModelUnavailable("cluster recommendation schema mismatch")
    model = _nonempty_str(payload.get("recommended_model")) or _nonempty_str(
        payload.get("model")
    ) or _nonempty_str(payload.get("model_id"))
    backend = _nonempty_str(payload.get("backend"))
    owner = _nonempty_str(payload.get("owner")) or _nonempty_str(payload.get("node"))
    endpoint = _nonempty_str(payload.get("endpoint"))
    ready = payload.get("ready")
    if ready is False:
        raise ForegroundModelUnavailable("cluster recommendation unready")
    if not model or not backend or not owner or not endpoint or ready is not True:
        raise ForegroundModelUnavailable("cluster recommendation admission absent")

    ttl = EXACT_FACE_CLUSTER_ADMISSION_TTL_S if ttl_s is None else float(ttl_s)
    if not math.isfinite(ttl) or ttl < 0.0:
        raise ForegroundModelUnavailable("cluster recommendation admission absent")
    if "snapshot_age_s" in payload:
        raw_age = payload.get("snapshot_age_s")
        if isinstance(raw_age, bool) or not isinstance(raw_age, (int, float)) or not math.isfinite(float(raw_age)):
            raise ForegroundModelUnavailable("cluster recommendation freshness invalid")
        if float(raw_age) < 0.0:
            raise ForegroundModelUnavailable("cluster recommendation freshness invalid")
    age_s = _recommendation_snapshot_age_s(payload)
    if age_s is None:
        raise ForegroundModelUnavailable("cluster recommendation admission absent")
    if age_s > ttl:
        raise ForegroundModelUnavailable("cluster recommendation stale")

    lease = _admission_lease(payload)
    if lease is None:
        raise ForegroundModelUnavailable("cluster recommendation admission absent")
    clock = time.time() if now_s is None else float(now_s)
    if not math.isfinite(clock):
        raise ForegroundModelUnavailable("cluster recommendation admission absent")
    if lease["expires"] < clock:
        raise ForegroundModelUnavailable("cluster recommendation lease expired")
    if lease["issued"] > clock:
        raise ForegroundModelUnavailable("cluster recommendation admission absent")

    authoritative = _positive_generation(current_generation)
    if authoritative is None:
        authoritative = _positive_generation(payload.get("current_generation"))
    if authoritative is None:
        lease_obj = payload.get("lease")
        if isinstance(lease_obj, dict):
            authoritative = _positive_generation(lease_obj.get("current_generation"))
    if authoritative is None or lease["generation"] != authoritative:
        raise ForegroundModelUnavailable("cluster recommendation admission absent")

    if not _is_chat_capable(
        {"id": model, "capability": payload.get("capability") or "chat"}
    ):
        raise ForegroundModelUnavailable("cluster recommendation mismatch")
    capability = payload.get("capability")
    if capability is not None and _nonempty_str(capability).lower() not in {"", "chat"}:
        raise ForegroundModelUnavailable("cluster recommendation mismatch")
    if catalog_ids is not None and model not in catalog_ids:
        raise ForegroundModelUnavailable("cluster recommendation mismatch")
    requested = _nonempty_str(requested_model)
    if requested and model != requested:
        raise ForegroundModelUnavailable("cluster recommendation mismatch")
    expected_owner_id = _nonempty_str(expected_owner)
    if expected_owner_id and owner != expected_owner_id:
        raise ForegroundModelUnavailable("cluster recommendation mismatch")
    expected_endpoint_id = _nonempty_str(expected_endpoint)
    if expected_endpoint_id and endpoint != expected_endpoint_id:
        raise ForegroundModelUnavailable("cluster recommendation mismatch")
    return {
        "schema": schema,
        "model": model,
        "backend": backend,
        "owner": owner,
        "node": owner,
        "endpoint": endpoint,
        "ready": True,
        "snapshot_age_s": age_s,
        "generation": lease["generation"],
        "current_generation": authoritative,
        "lease": lease,
        "source": "admitted",
    }


def require_exact_face_cluster_admission(
    *,
    hivemind_url: str,
    requested_model: str,
    recommend_payload: Any | None = None,
    catalog: list[dict[str, Any]] | None = None,
    expected_owner: str | None = None,
    expected_endpoint: str | None = None,
    current_generation: int | None = None,
    now_s: float | None = None,
    ttl_s: float | None = None,
) -> dict[str, Any]:
    """Fetch-or-use a typed recommendation and admit it for exact Face prewarm.

    Ordinary ``choose_foreground_model`` is unchanged. This seam is the only
    place catalog-only / legacy scalar recommend is refused as warm authority.
    """
    rec_body = recommend_payload
    if rec_body is None:
        try:
            from machine_spirit_4.double_agent.recommend_lease import (
                fetch_typed_recommend_document,
            )

            rec_body = fetch_typed_recommend_document(
                hivemind_url,
                capability="chat",
                requested_model=requested_model,
            )
        except ForegroundModelUnavailable:
            raise
        except Exception:
            rec_body = None
    if catalog is None:
        try:
            catalog = _http_get_models(hivemind_url)
        except RuntimeError as exc:
            raise ForegroundModelUnavailable(
                f"cluster recommendation admission unavailable; catalog failed: {exc}"
            ) from exc
    return admit_cluster_recommendation(
        rec_body,
        catalog_ids=_catalog_model_ids(catalog),
        requested_model=requested_model,
        expected_owner=expected_owner,
        expected_endpoint=expected_endpoint,
        current_generation=current_generation,
        now_s=now_s,
        ttl_s=ttl_s,
    )


def _catalog_model_ids(catalog: list[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for entry in catalog:
        if not isinstance(entry, dict):
            continue
        model_id = _nonempty_str(
            entry.get("id")
        ) or _nonempty_str(entry.get("model")) or _nonempty_str(entry.get("name"))
        if model_id:
            ids.add(model_id)
    return ids


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
            capability="chat",
        )
    except Exception:
        return None
    if not isinstance(body, dict):
        return None
    scalar_recommendation = "recommended_model" in body
    if scalar_recommendation:
        required_values = (
            body.get("capability"),
            body.get("recommended_model"),
            body.get("backend"),
        )
        if not all(isinstance(value, str) and value.strip() for value in required_values):
            return None
        candidates: list[Any] = [body]
    else:
        candidates = []
        for legacy_key in ("recommended", "models"):
            legacy_candidates = body.get(legacy_key)
            if isinstance(legacy_candidates, list) and legacy_candidates:
                candidates = legacy_candidates
                break

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        model_id = str(
            candidate.get("recommended_model")
            or candidate.get("model_id")
            or candidate.get("id")
            or candidate.get("model")
            or ""
        ).strip()
        if not model_id or not _is_chat_capable(candidate):
            continue
        reason = str(
            candidate.get("reason")
            or candidate.get("score")
            or candidate.get("backend")
            or ""
        )
        return ForegroundChoice(
            model_id=model_id,
            source="hivemind.models.recommend",
            detail=(
                f"recommend@v1 picked {model_id} ({reason})"
                if reason
                else f"recommend@v1 picked {model_id}"
            ),
            catalog_confirmation_required=scalar_recommendation,
        )
    return None


# Foreground parameter-size ceiling (billions). FOREGROUND_PRIORITY_PATTERNS
# are substring matches, so a heavy variant of an otherwise-small family
# (llama3.1:70b matches "llama3.1", qwen3:235b matches "qwen3:", gemma3:27b
# matches "gemma3") would otherwise pass the family check and be treated as
# "small/responsive". This ceiling admits the existing foreground models
# (0.5b-8b, with headroom for mid ~13-14b) and rejects the 27b/70b/235b heavy
# variants the review flagged. Chosen as 2x the 8b ceiling of the priority
# list; it is a generic size gate, not a model-id blacklist.
MAX_FOREGROUND_PARAM_B = 16.0

# DIAGNOSTIC ONLY (R2.4): the safety gate is now the bounded positive-allow
# ``_satisfies_foreground_contract`` below. This island classifier is retained
# as a NON-authoritative diagnostic (still exercised by unit tests documenting
# how a size claim would be read); it no longer decides admission.
#
# Parameter-size classifier (value in billions). Explicit SIZE / NO_SIZE /
# INVALID contract via a LEXICAL ISLAND GRAMMAR (R2.3-P3):
#   * NO_SIZE : no size-relevant island -> admitted on family match (e.g.
#     "llama3.1", "qwen2.5", "family:latest"). No size is invented.
#   * SIZE(n) : one or more islands FULL-MATCH the size grammar; effective size
#     = max of all such islands; admit iff n <= MAX_FOREGROUND_PARAM_B (exact),
#     else reject.
#   * INVALID : any size-relevant island that does NOT full-match the grammar
#     -> FAIL CLOSED (contract False). NEVER salvaged to a smaller token and
#     NEVER treated as NO_SIZE.
# The id is split into ISLANDS bounded ONLY by explicit model separators
# (":" "/" "@" "-" "_" whitespace). Dots, "x", "+", "e"/"E", letters and digits
# are INTRA-island, so punctuation cannot hide a malformed number ("1e+2b",
# "2x+8.1b" are single islands that fail to full-match -> INVALID) and a
# glued/letter-prefixed size cannot fall through to NO_SIZE ("gemma327b",
# "phi4-mini70b", "foo70b" are size-relevant islands that fail to full-match ->
# INVALID). An island is size-relevant if it is digit-led OR contains a
# digit-led run ending in "b". A size-relevant island is SIZE only if the WHOLE
# island matches plain "^<int>(.<frac>)?b$" or MoE "^<int>x<int|dec>b$" (MoE
# product via EXACT ambient-independent Fraction arithmetic); otherwise it is
# INVALID. Ambiguous glued alphanumeric islands fail closed: availability loss
# (e.g. a bare numeric island like "3" in a hypothetical "qwen-3-8b", a glued
# quant like "235b4bit", or "30b-a3b") is preferred over unsafe admission.
_ISLAND_SEP_RE = re.compile(r"[:/@\-_\s]+")
_SIZE_CLAIM_RE = re.compile(r"\d[0-9.xe+]*b")
_PLAIN_FULL_RE = re.compile(r"^(\d+(?:\.\d+)?)b$")
_MOE_FULL_RE = re.compile(r"^(\d+)x(\d+(?:\.\d+)?)b$")


def _moe_product(experts: Decimal, size: Decimal) -> Fraction:
    """Exact experts x size product via ``fractions.Fraction`` (pure-integer,
    FULLY ambient-decimal-context independent).

    SECURITY (R2.4-P4, scan 2026-07-06): this is the SAME ambient-independent
    primitive the gate uses. The earlier flow multiplied under
    ``decimal.localcontext()``, which COPIES the caller's ambient context, so a
    hostile ambient context (Emax=0 + Overflow trap, clamp=1, or Emin=0 +
    Subnormal trap) made this raise or round even for the retained diagnostics.
    Fraction has no context and cannot be perturbed by any in-process caller."""
    return Fraction(experts) * Fraction(size)


def _classify_param_size(model_id: str) -> tuple[str, Fraction | None]:
    """Classify ``model_id`` as ("no_size", None), ("size", max_billions), or
    ("invalid", None) using the island grammar above. INVALID (any
    size-relevant island that is not a clean plain/MoE size token) fails closed
    and is never salvaged to a smaller number.

    R2.4-P4: sizes are exact, ambient-decimal-context independent
    ``fractions.Fraction`` values (numerically equal to the prior Decimal
    results); this diagnostic no longer raises under a hostile ambient context."""
    mid = (model_id or "").lower()
    invalid = False
    sizes: list[Fraction] = []
    for island in _ISLAND_SEP_RE.split(mid):
        if not island:
            continue
        size_relevant = island[0].isdigit() or bool(_SIZE_CLAIM_RE.search(island))
        if not size_relevant:
            continue  # NO_SIZE contribution (family / word / version island)
        plain = _PLAIN_FULL_RE.match(island)
        if plain:
            sizes.append(Fraction(plain.group(1)))
            continue
        moe = _MOE_FULL_RE.match(island)
        if moe:
            sizes.append(_moe_product(Decimal(moe.group(1)), Decimal(moe.group(2))))
            continue
        invalid = True  # size-relevant but not a clean size token -> fail closed
    if invalid:
        return ("invalid", None)
    if sizes:
        return ("size", max(sizes))
    return ("no_size", None)


def _parse_param_size_b(model_id: str) -> Fraction | None:
    """Return the effective SIZE in billions (exact, ambient-decimal-context
    independent ``fractions.Fraction``) when ``model_id`` carries a well-formed
    size expression, else None (for BOTH NO_SIZE and INVALID). Callers that must
    distinguish INVALID (fail closed) use ``_classify_param_size`` /
    ``_satisfies_foreground_contract``."""
    cls, val = _classify_param_size(model_id)
    return val if cls == "size" else None


# R2.4 bounded positive-ALLOW gate (reviewers Mencius + Descartes). App Registry
# hands MS4 only an opaque model id and /v1/models carries NO trusted
# param-count metadata, so an inference-over-arbitrary-ids parser fails OPEN on
# fuzzed inputs (observed: 800/10k fuzz cases admitted). The Face-Lobe safety
# gate therefore ADMITS ONLY a bounded positive set and fails closed on
# everything else:
#   (a) the exact-safe immutable id set below, OR
#   (b) a whole-id ``<grammar-family>:<size|MoE>b`` whose family segment equals
#       ONE grammar family EXACTLY and whose size satisfies
#       0 < size <= MAX_FOREGROUND_PARAM_B (exact, ambient-independent
#       Fraction arithmetic via _foreground_tag_size_b).
# Grammar families are intentionally LIMITED to the family-like priority
# patterns; the exact-priority ids qwen2.5:7b / llama3.2:8b are NOT widened into
# families (so qwen2.5:14b, llama3.2:2x8b, qwen3-coder-next:8b fail closed).
# Preconditions: non-empty str, len <= 256, ASCII; matching is
# lowercase-normalized. The _classify_param_size island parser above is retained
# ONLY as a non-authoritative diagnostic and NO LONGER gates safety.
_FOREGROUND_EXACT_SAFE: frozenset[str] = frozenset({
    "nemotron-3-nano:4b",
    "qwen3:8b",
    "llama3.1:8b",
    "llama3.2:8b",
    "qwen2.5:7b",
    "dolphin-llama3",
    "phi4-mini",
    "phi3.5",
})
_FOREGROUND_GRAMMAR_FAMILIES: frozenset[str] = frozenset({
    "qwen3",
    "llama3.1",
    "gemma4",
    "gemma3",
    "gemma2",
    "gemma",
})
_FOREGROUND_MAX_ID_LEN = 256
_FG_PLAIN_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)b")
_FG_MOE_SIZE_RE = re.compile(r"(\d+)x(\d+(?:\.\d+)?)b")


def _foreground_tag_size_b(tag: str) -> Fraction | None:
    """Exact size in billions for a whole size tag: plain ``<n>b`` or MoE
    ``<e>x<n>b`` (experts x size). Returns None when the tag is not a single
    clean size token (letters, ``:latest``, signed, scientific, punctuation,
    hyphenated multi-island, empty). The tag must FULL-match -- a substring size
    never leaks through.

    SECURITY (R2.4-P3, scan 2026-07-06): size arithmetic uses
    ``fractions.Fraction`` (exact, pure-integer, FULLY ambient-decimal-context
    independent). The earlier flow computed the MoE product via
    ``decimal.localcontext()``, which COPIES the caller's ambient context; a
    hostile ambient context (Emax=0 + Overflow trap, clamp=1, or Emin=0 +
    Subnormal trap) set anywhere in-process could make the gate RAISE or FLIP.
    Fraction has no context and cannot be perturbed by any caller."""
    plain = _FG_PLAIN_SIZE_RE.fullmatch(tag)
    if plain:
        return Fraction(plain.group(1))
    moe = _FG_MOE_SIZE_RE.fullmatch(tag)
    if moe:
        return Fraction(int(moe.group(1))) * Fraction(moe.group(2))
    return None


def _satisfies_foreground_contract(model_id: str) -> bool:
    """R2.4 bounded positive-ALLOW gate: True IFF ``model_id`` is a non-empty
    ASCII string of length <= 256 that is either (a) in the exact-safe immutable
    set, or (b) a whole id ``<grammar-family>:<size|MoE>b`` whose family segment
    equals one ``_FOREGROUND_GRAMMAR_FAMILIES`` entry EXACTLY and whose explicit
    size satisfies ``0 < size <= MAX_FOREGROUND_PARAM_B`` (exact, ambient-decimal
    -context-independent ``Fraction`` arithmetic).

    Everything else fails closed: unknown / non-grammar families, unsized or
    mutable aliases (``:latest``), hyphenated multi-island tags, zero and
    zero-operand MoE, arbitrary suffix / digest / quant / punctuation,
    non-ASCII, over-length, and over-ceiling sizes.

    This is the SOLE safety gate for auto-selection: ``choose_foreground_model``
    (recommendation reconcile + the gated ``MS4_DEFAULT_MODEL`` fallback) and
    ``_pick_from_catalog`` both route through it. The explicit
    ``MS4_FOREGROUND_MODEL`` operator override bypasses this gate in
    ``choose_foreground_model`` and is unaffected. Matching is case-insensitive
    (lowercase-normalized)."""
    if not isinstance(model_id, str) or not model_id:
        return False
    if len(model_id) > _FOREGROUND_MAX_ID_LEN or not model_id.isascii():
        return False
    mid = model_id.lower()
    if mid in _FOREGROUND_EXACT_SAFE:
        return True
    if ":" not in mid:
        return False
    family, _, tag = mid.partition(":")
    if family not in _FOREGROUND_GRAMMAR_FAMILIES:
        return False
    size_b = _foreground_tag_size_b(tag)
    if size_b is None:
        return False
    # R2.4-P3: exact Fraction comparison, independent of the ambient decimal ctx.
    return Fraction(0) < size_b <= Fraction(str(MAX_FOREGROUND_PARAM_B))


def _catalog_confirms_ready_model(
    catalog: list[dict[str, Any]],
    model_id: str,
) -> bool:
    """Require an exact loaded/reachable/eligible catalog match."""
    if not _satisfies_foreground_contract(model_id):
        return False
    for entry in catalog:
        if not _is_chat_capable(entry):
            continue
        normalized = _normalize_model_entry(entry)
        if normalized["id"] != model_id:
            continue
        if normalized["loaded"] and normalized["hivemind_reachable"]:
            return True
    return False


def choose_foreground_model(
    *,
    hivemind_url: str,
    force_refresh: bool = False,
) -> ForegroundChoice:
    """Return the Face Lobe model to use for the next chat turn.

    Cached for ``MODELS_TTL_SECS`` (monotonic clock) keyed on
    (hivemind_url, ``MS4_FOREGROUND_MODEL`` override, ``MS4_DEFAULT_MODEL``).
    A monotonic generation guards commit-ordering so an older in-flight call
    cannot overwrite a newer committed result, and every return is a defensive
    copy so a caller mutating the result cannot poison the cache.
    """
    override = os.environ.get(ENV_OVERRIDE, "").strip() or None
    # R2.4-P3 (Leibniz): read the raw MS4_DEFAULT_MODEL BEFORE the cache lookup
    # and key the cache on it too, so a fallback change within the TTL is
    # re-derived rather than served stale (the cache previously keyed only on
    # (url, override) and read MS4_DEFAULT_MODEL after the lookup).
    env_default_raw = os.environ.get(DEFAULT_FALLBACK_ENV, "").strip()
    face_profile = _face_profile()
    # R2.4-P4 (Sagan P2-2): monotonic clock for the TTL (time.time() can roll
    # backwards) + a per-call monotonic generation for commit-ordering.
    now = time.monotonic()
    with _CACHE_LOCK:
        _NEXT_GEN[0] += 1
        my_gen = _NEXT_GEN[0]
        cached = _CACHE.get("choice")
        if (
            cached is not None
            and not force_refresh
            and _CACHE.get("hivemind_url") == hivemind_url
            and _CACHE.get("override") == override
            and _CACHE.get("env_default") == env_default_raw
            and _CACHE.get("face_profile") == face_profile
            and (now - _CACHE.get("fetched_at", 0.0)) < MODELS_TTL_SECS
        ):
            # R2.4-P4 (Sagan P2-1): defensive copy so a caller mutating the
            # returned object cannot poison the cached result.
            return replace(cached)

    if override:
        # R2.4-P3 (Leibniz): the explicit override short-circuits BEFORE any
        # automatic-fallback (MS4_DEFAULT_MODEL) validation, so a hostile or
        # off-gate default is never evaluated when an override is set.
        choice = ForegroundChoice(
            model_id=override,
            source="override",
            detail=f"MS4_FOREGROUND_MODEL env var set",
        )
    else:
        # R2.4: MS4_DEFAULT_MODEL is a GATED automatic fallback, NOT an override.
        # Honor it only when it passes the positive-allow gate; otherwise
        # (off-gate, or unset) fall back to the built-in exact-safe default. Only
        # MS4_FOREGROUND_MODEL (the override branch above) is the explicit
        # operator bypass of the gate.
        if env_default_raw and _satisfies_foreground_contract(env_default_raw):
            fallback_model = env_default_raw
        else:
            fallback_model = DEFAULT_FOREGROUND_MODEL
        choice: ForegroundChoice | None = None
        quality_choice_selected = False
        if face_profile in {"auto", "quality"}:
            try:
                quality_catalog = _http_get_models(hivemind_url)
                quality_nodes = _http_get_carrier_nodes(hivemind_url)
                choice = _pick_high_memory_quality_model(
                    quality_catalog,
                    quality_nodes,
                )
                quality_choice_selected = choice is not None
            except RuntimeError as exc:
                log.info(
                    "high-memory Face quality profile unavailable; retaining latency policy: %s",
                    exc,
                )

        # HiveMind's generic ``capability=chat`` recommendation has useful
        # cluster information, but it does not carry MS4's Face-role ordering
        # contract. Reconcile every recommendation against the same live
        # catalog used by the policy picker. A loaded/reachable policy candidate
        # therefore wins first (Nemotron 4B when ready); only when no policy
        # candidate exists may an exact, foreground-safe, catalog-ready
        # recommendation survive. This also removes the old legacy-list bypass
        # that trusted recommendation metadata without readiness proof.
        # Typed exact-admission (schema/owner/freshness/lease) is a separate
        # gate used by selected Face prewarm/TTFT, not this ordinary path.
        if choice is None:
            choice = _try_hivemind_recommend(hivemind_url)
        if choice is not None and not quality_choice_selected:
            recommended_id = choice.model_id
            try:
                catalog = _http_get_models(hivemind_url)
                policy_choice = _pick_from_catalog(catalog)
                if policy_choice is not None:
                    local_choice = policy_choice
                elif _catalog_confirms_ready_model(catalog, recommended_id):
                    local_choice = replace(
                        choice,
                        catalog_confirmation_required=False,
                    )
                else:
                    local_choice = None
            except ForegroundModelUnavailable:
                raise
            except RuntimeError as exc:
                local_choice = ForegroundChoice(
                    model_id=fallback_model,
                    source="error",
                    detail=(
                        f"recommendation {recommended_id!r} not accepted; "
                        f"catalog unavailable: {exc}"
                    ),
                )
            if local_choice is None:
                local_choice = ForegroundChoice(
                    model_id=fallback_model,
                    source="fallback",
                    detail=(
                        f"recommendation {recommended_id!r} not accepted; "
                        "no ready foreground candidate in catalog"
                    ),
                )
            choice = local_choice
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
            except ForegroundModelUnavailable:
                raise
            except RuntimeError as exc:
                choice = ForegroundChoice(
                    model_id=fallback_model,
                    source="error",
                    detail=str(exc),
                )

    with _CACHE_LOCK:
        # R2.4-P4 (Sagan P2-2): commit only if this generation is at least as new
        # as the last committed one, so a slow older computation cannot replace a
        # newer force_refresh result.
        # R2.4-P5: when our commit is skipped we DO NOT adopt the cached value --
        # it may have been computed under a DIFFERENT key (url / override /
        # MS4_DEFAULT_MODEL). The older caller returns ITS OWN computed result
        # for ITS OWN key; only the COMMIT is skipped, and the newer cache entry
        # is preserved. Cross-key results never leak between callers.
        if my_gen >= _CACHE.get("gen", 0):
            _CACHE["choice"] = choice
            _CACHE["fetched_at"] = now
            _CACHE["hivemind_url"] = hivemind_url
            _CACHE["override"] = override
            _CACHE["env_default"] = env_default_raw
            _CACHE["face_profile"] = face_profile
            _CACHE["gen"] = my_gen
    # R2.4-P4 (Sagan P2-1): hand back a defensive copy, never the cached object.
    return replace(choice)


def _clear_cache_for_tests() -> None:
    with _CACHE_LOCK:
        _CACHE["choice"] = None
        _CACHE["fetched_at"] = 0.0
        _CACHE["hivemind_url"] = None
        _CACHE["override"] = None
        _CACHE["env_default"] = None
        _CACHE["face_profile"] = None
        _CACHE["gen"] = 0
        _NEXT_GEN[0] = 0
