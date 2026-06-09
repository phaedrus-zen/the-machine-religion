"""Quartermaster cascade — tiered, confidence-gated tool retrieval.

Given a natural-language query, return a ranked shortlist of the
tools most likely to serve it, using the cheapest tier that is
confident:

1. **Deterministic** (instant, offline): the toolbox keyword lexicon
   (:mod:`taxonomy`) nails the toolbox when the query unambiguously
   names one ("list my VMs" -> ``vm``). Tool-level ranking within the
   matched toolbox uses the offline TF-IDF index. High confidence.
2. **Embeddings** (~ms, offline TF-IDF by default; HiveMind vectors
   when ``MS4_QM_USE_HM_EMBEDDINGS=1``): two-stage retrieval — toolbox
   shortlist, then tools within those toolboxes. Confidence = the top
   tool's cosine score.
3. **Tiny-LLM** (only on ambiguity): an injectable, fail-safe
   classifier picks one tool from the embeddings shortlist. Consulted
   ONLY when (a) a classifier is supplied, (b) ``MS4_QM_LLM_CLASSIFY``
   is enabled, and (c) the embeddings confidence is below threshold.
   Any error / unsure verdict degrades to the embeddings result — the
   LLM can only ever *sharpen* a low-confidence answer, never hang a
   turn or invent a new failure mode. Mirrors the proven pattern in
   :mod:`machine_spirit_4.double_agent.router` and
   :mod:`machine_spirit_4.double_agent.continuation`.

The cascade returns *which tools* (ranked) plus a confidence and the
tier that produced it. It does NOT execute, fill arguments, or gate
— that is the :mod:`router`'s job (Phase C).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from . import catalog as catalog_mod
from . import index as index_mod
from . import taxonomy
from .catalog import Catalog, ToolEntry, is_inline_eligible
from .index import ToolIndex


log = logging.getLogger("ms4.gateway.quartermaster.cascade")


TIER_DETERMINISTIC = "deterministic"
TIER_EMBEDDINGS = "embeddings"
TIER_LLM = "llm"
TIER_HM_SEARCH = "hm_search"
TIER_NONE = "none"


# The HiveMind substrate primitive (Phase B). When present in the live
# catalog AND ``MS4_QM_DELEGATE`` is on, the cascade delegates retrieval
# to it (one auto-updating index shared by every MCP client) and falls
# back to the local engine on any failure.
HM_SEARCH_TOOL = "hivemind.tools.search@v1"


def _delegate_enabled() -> bool:
    return os.environ.get("MS4_QM_DELEGATE", "1").strip().lower() not in {"0", "false", "no"}


# A tool classifier takes (query, candidate_entries) and returns the
# chosen tool id (must be one of the candidate names) or None (unsure).
ToolClassifier = Callable[[str, list[ToolEntry]], Optional[str]]


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------


def _env_float(name: str, default: float, *, minimum: float = 0.0, maximum: float = 1.0) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return max(minimum, min(maximum, float(raw.strip())))
    except (TypeError, ValueError):
        log.warning("quartermaster.cascade: ignoring non-float %s=%r; default %s", name, raw, default)
        return default


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return max(minimum, int(raw.strip()))
    except (TypeError, ValueError):
        return default


def _deterministic_confidence() -> float:
    return _env_float("MS4_QM_DETERMINISTIC_CONFIDENCE", 0.9)


def _embeddings_low_confidence() -> float:
    """Embeddings top-score at/below which the result is "ambiguous"
    and the tiny-LLM tier is consulted (when available)."""
    return _env_float("MS4_QM_EMBEDDINGS_LOW_CONFIDENCE", 0.18)


def _llm_classify_enabled() -> bool:
    return os.environ.get("MS4_QM_LLM_CLASSIFY", "").strip().lower() in {"1", "true", "yes", "on"}


def _default_top_k() -> int:
    return _env_int("MS4_QM_TOP_K", 5, minimum=1)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedTool:
    name: str
    toolbox: str
    cluster: str
    score: float
    kind: str
    inline_eligible: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "toolbox": self.toolbox,
            "cluster": self.cluster,
            "score": round(self.score, 6),
            "kind": self.kind,
            "inline_eligible": self.inline_eligible,
        }


@dataclass(frozen=True)
class ToolResolution:
    """``Ms4ToolResolution.v1`` — the cascade's output."""

    schema: str
    query: str
    tier: str
    confidence: float
    toolboxes: tuple[str, ...]
    tools: tuple[ResolvedTool, ...]
    catalog_version: str
    fallback_reason: str | None = None

    def top_tool(self) -> ResolvedTool | None:
        return self.tools[0] if self.tools else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "query": self.query,
            "tier": self.tier,
            "confidence": round(self.confidence, 6),
            "toolboxes": list(self.toolboxes),
            "tools": [t.to_dict() for t in self.tools],
            "catalog_version": self.catalog_version,
            "fallback_reason": self.fallback_reason,
        }


def _empty_resolution(query: str, catalog_version: str, reason: str) -> ToolResolution:
    return ToolResolution(
        schema="Ms4ToolResolution.v1",
        query=query,
        tier=TIER_NONE,
        confidence=0.0,
        toolboxes=(),
        tools=(),
        catalog_version=catalog_version,
        fallback_reason=reason,
    )


def _resolved(entry: ToolEntry, score: float) -> ResolvedTool:
    return ResolvedTool(
        name=entry.name,
        toolbox=entry.toolbox,
        cluster=entry.cluster,
        score=score,
        kind=entry.kind,
        inline_eligible=is_inline_eligible(entry),
    )


# ---------------------------------------------------------------------------
# Tier 1: deterministic keyword → toolbox
# ---------------------------------------------------------------------------


def _deterministic_toolboxes(query: str, catalog: Catalog) -> list[str]:
    """Return toolboxes whose keyword lexicon matches the query, with
    the strongest match first. Only toolboxes that actually exist in
    the catalog are returned.

    A "match" requires the keyword/phrase to appear as a word-ish
    substring. We score by (number of distinct keywords matched,
    longest keyword matched) so a specific multi-word phrase
    ("voice identity") beats a generic single token.
    """
    lower = f" {query.lower().strip()} "
    present = set(catalog.toolboxes.keys())
    scored: list[tuple[float, int, str]] = []
    for toolbox in present:
        keywords = taxonomy.keywords_for_toolbox(toolbox)
        if not keywords:
            continue
        matched = [kw for kw in keywords if _keyword_in(lower, kw)]
        if not matched:
            continue
        longest = max(len(kw) for kw in matched)
        scored.append((float(len(matched)), longest, toolbox))
    # Sort by (#matches desc, longest-keyword desc, name asc) for stable order.
    scored.sort(key=lambda t: (-t[0], -t[1], t[2]))
    return [tb for _, _, tb in scored]


def _keyword_in(padded_lower_query: str, keyword: str) -> bool:
    """Substring match with light word-boundary safety. ``padded_lower_query``
    is the query lower-cased and wrapped in spaces. Multi-word phrases
    match as substrings; single short tokens require space boundaries so
    "vm" doesn't fire inside "vmware-something" unintentionally."""
    kw = keyword.lower()
    if " " in kw or "-" in kw or "?" in kw:
        return kw in padded_lower_query
    return f" {kw} " in padded_lower_query


def _is_deterministic_confident(toolboxes: list[str], query: str, catalog: Catalog) -> bool:
    """A deterministic match is "confident" when exactly one toolbox
    matched, OR the top toolbox clearly dominates (matched on a
    multi-word phrase while runners-up matched only single tokens).
    Conservative: when two toolboxes tie, defer to embeddings."""
    if not toolboxes:
        return False
    if len(toolboxes) == 1:
        return True
    # Multiple matched: confident only if the top one matched a
    # multi-word phrase that the second did not.
    lower = f" {query.lower().strip()} "
    top_multi = any(
        (" " in kw) and _keyword_in(lower, kw)
        for kw in taxonomy.keywords_for_toolbox(toolboxes[0])
    )
    second_multi = any(
        (" " in kw) and _keyword_in(lower, kw)
        for kw in taxonomy.keywords_for_toolbox(toolboxes[1])
    )
    return top_multi and not second_multi


# ---------------------------------------------------------------------------
# Tier 3: tiny-LLM classifier (optional, injectable, fail-safe)
# ---------------------------------------------------------------------------


_LLM_SYSTEM_PROMPT = (
    "You select the single best tool for a user's request from a short "
    "candidate list. Each candidate is 'TOOL_ID: description'. Answer "
    "with EXACTLY one tool id copied verbatim from the list, or the "
    "single word NONE if none fit. No punctuation, no explanation."
)


def _extract_text(resp: Any) -> str:
    if isinstance(resp, str):
        return resp
    if not isinstance(resp, dict):
        return ""
    choices = resp.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            msg = first.get("message")
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                return msg["content"]
            if isinstance(first.get("text"), str):
                return first["text"]
    for key in ("text", "content", "output"):
        if isinstance(resp.get(key), str):
            return resp[key]
    return ""


def make_llm_tool_classifier(
    hivemind_url: str,
    *,
    model: str | None = None,
    timeout: float | None = None,
) -> ToolClassifier:
    """Build a fail-safe LLM tool classifier.

    ``model`` defaults to ``MS4_QM_MODEL``; when unset we ask the Face
    Lobe small-model picker (:func:`choose_foreground_model`) for a
    cheap loaded model. Any failure path returns ``None`` ("unsure")
    so the cascade keeps the embeddings result.
    """
    resolved_model = model or os.environ.get("MS4_QM_MODEL") or ""
    if timeout is None:
        try:
            timeout = float(os.environ.get("MS4_QM_LLM_TIMEOUT", "3.0"))
        except (TypeError, ValueError):
            timeout = 3.0

    def _classify(query: str, candidates: list[ToolEntry]) -> Optional[str]:
        if not query or not candidates:
            return None
        try:
            from .. import hivemind_tools
        except Exception as exc:  # pragma: no cover — import guard
            log.debug("quartermaster classifier import failed: %s", exc)
            return None

        chat_model = resolved_model
        if not chat_model:
            try:
                from ...double_agent.model_picker import choose_foreground_model
                chat_model = choose_foreground_model(hivemind_url=hivemind_url).model_id
            except Exception as exc:
                log.debug("quartermaster classifier model pick failed: %s", exc)
                return None
        if not chat_model:
            return None

        candidate_lines = "\n".join(
            f"{c.name}: {c.description[:140]}" for c in candidates[:12]
        )
        user_msg = f"Request: {query.strip()[:240]}\n\nCandidates:\n{candidate_lines}"
        try:
            resp = hivemind_tools.inference_chat(
                hivemind_url,
                messages=[
                    {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                model=chat_model,
                max_tokens=24,
                temperature=0.0,
                timeout=timeout,
            )
        except Exception as exc:
            log.info("quartermaster classifier call failed (unsure): %s", exc)
            return None

        text = _extract_text(resp).strip()
        if not text or text.lower().startswith("none"):
            return None
        # Accept only a verbatim candidate id (defends against the model
        # hallucinating a tool name). Tolerate surrounding quotes/space.
        cleaned = text.strip().strip("`'\"").split()[0] if text.split() else ""
        names = {c.name for c in candidates}
        if cleaned in names:
            return cleaned
        # Sometimes the model returns the id mid-sentence; scan for an
        # exact candidate substring.
        for c in candidates:
            if c.name in text:
                return c.name
        return None

    return _classify


# ---------------------------------------------------------------------------
# Phase B: delegation to the HiveMind tools.search primitive
# ---------------------------------------------------------------------------


def _resolve_via_hm_search(
    query: str,
    catalog: Catalog,
    *,
    hivemind_url: str,
    top_k: int,
) -> ToolResolution | None:
    """Delegate retrieval to ``hivemind.tools.search@v1`` when present.

    Returns a :class:`ToolResolution` (tier ``hm_search``) on success,
    or ``None`` on any failure / empty result so :func:`resolve` falls
    back to the local engine. The HiveMind tool returns ranked tool
    ids; we re-attach ``kind``/``inline_eligible`` from the local
    catalog so the router's safety gates still apply to delegated
    results."""
    try:
        from ..hivemind_state import post_mcp_envelope
    except Exception as exc:  # pragma: no cover — import guard
        log.debug("quartermaster delegate import failed: %s", exc)
        return None

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": HM_SEARCH_TOOL, "arguments": {"query": query, "top_k": top_k}},
    }
    try:
        envelope = post_mcp_envelope(hivemind_url, payload, timeout=5)
    except Exception as exc:
        log.info("quartermaster hm_search delegate failed (-> local): %s", exc)
        return None

    body = _unwrap_search_result(envelope)
    if not isinstance(body, dict):
        return None
    raw_tools = body.get("tools")
    if not isinstance(raw_tools, list) or not raw_tools:
        return None

    by_name = catalog.by_name()
    tools: list[ResolvedTool] = []
    toolboxes: list[str] = []
    for item in raw_tools[:top_k]:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name:
            continue
        score = float(item.get("score") or 0.0)
        entry = by_name.get(name)
        if entry is not None:
            tools.append(_resolved(entry, score))
            if entry.toolbox not in toolboxes:
                toolboxes.append(entry.toolbox)
        else:
            # Delegated tool not in MS4's local catalog view — derive a
            # minimal entry so the router can still gate it.
            from . import taxonomy
            from .catalog import KIND_HIVEMIND_NATIVE, ToolEntry, is_inline_eligible
            tb = taxonomy.canonical_toolbox(taxonomy.tool_domain(name))
            synth = ToolEntry(
                schema="Ms4QuartermasterTool.v1",
                name=name,
                toolbox=tb,
                cluster=taxonomy.cluster_for_toolbox(tb),
                description=str(item.get("description") or ""),
                source="hivemind",
                kind=KIND_HIVEMIND_NATIVE,
            )
            tools.append(
                ResolvedTool(
                    name=name, toolbox=tb, cluster=synth.cluster, score=score,
                    kind=synth.kind, inline_eligible=is_inline_eligible(synth),
                )
            )
            if tb not in toolboxes:
                toolboxes.append(tb)

    if not tools:
        return None
    return ToolResolution(
        schema="Ms4ToolResolution.v1",
        query=query,
        tier=TIER_HM_SEARCH,
        confidence=tools[0].score,
        toolboxes=tuple(toolboxes[:3]),
        tools=tuple(tools),
        catalog_version=catalog.version,
    )


def _unwrap_search_result(envelope: Any) -> Any:
    if not isinstance(envelope, dict) or envelope.get("error"):
        return None
    result = envelope.get("result")
    if not isinstance(result, dict) or result.get("isError"):
        return None
    content = result.get("content")
    if isinstance(content, list):
        for chunk in content:
            if isinstance(chunk, dict) and chunk.get("type") == "text":
                import json as _json
                try:
                    return _json.loads(chunk.get("text") or "")
                except _json.JSONDecodeError:
                    return None
    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def resolve(
    query: str,
    *,
    hivemind_url: str | None = None,
    catalog: Catalog | None = None,
    index: ToolIndex | None = None,
    classifier: ToolClassifier | None = None,
    top_k: int | None = None,
) -> ToolResolution:
    """Resolve ``query`` to a ranked tool shortlist via the tiered
    cascade. Always returns a :class:`ToolResolution` — failures
    surface as ``tier="none"`` with a ``fallback_reason``.

    ``catalog`` / ``index`` may be supplied directly (tests, or a
    caller that already built them); otherwise they're fetched for
    ``hivemind_url``.
    """
    k = top_k if top_k is not None else _default_top_k()

    if catalog is None:
        if not hivemind_url:
            return _empty_resolution(query, "", "no hivemind_url and no catalog supplied")
        catalog = catalog_mod.get_catalog(hivemind_url)
    if not catalog.tools:
        return _empty_resolution(query, catalog.version, "empty catalog")
    if not query or not query.strip():
        return _empty_resolution(query, catalog.version, "empty query")

    if index is None:
        index = index_mod.get_index(catalog, hivemind_url=hivemind_url)

    by_name = catalog.by_name()

    # ---- Phase B: delegate to the HiveMind substrate primitive when
    # it's in the live catalog (one shared, auto-updating index). Any
    # failure / empty result falls through to the local tiers below.
    if hivemind_url and _delegate_enabled() and HM_SEARCH_TOOL in by_name:
        delegated = _resolve_via_hm_search(query, catalog, hivemind_url=hivemind_url, top_k=k)
        if delegated is not None and delegated.tools:
            return delegated

    # ---- Tier 1: deterministic ----
    det_toolboxes = _deterministic_toolboxes(query, catalog)
    if det_toolboxes and _is_deterministic_confident(det_toolboxes, query, catalog):
        winner = det_toolboxes[0]
        # Rank tools within the matched toolbox via the index; fall
        # back to catalog order when the index can't score them.
        hits = index_mod.query_tools(
            index, query, top_k=k, toolbox_filter={winner}, hivemind_url=hivemind_url
        )
        tools = _hits_to_tools(hits, by_name)
        if not tools:
            # Index produced nothing (e.g. query shares no vocab) — just
            # return the toolbox's tools in catalog order; the toolbox
            # itself was a confident keyword match.
            tools = tuple(
                _resolved(e, 0.0) for e in catalog.toolboxes.get(winner, ())[:k]
            )
        return ToolResolution(
            schema="Ms4ToolResolution.v1",
            query=query,
            tier=TIER_DETERMINISTIC,
            confidence=_deterministic_confidence(),
            toolboxes=(winner,),
            tools=tools,
            catalog_version=catalog.version,
        )

    # ---- Tier 2: embeddings ----
    toolbox_hits = index_mod.query_toolboxes(index, query, top_k=3, hivemind_url=hivemind_url)
    # Union the keyword-matched toolboxes (if any) with the embedding
    # toolbox shortlist so a partial keyword signal still narrows.
    shortlist_boxes: list[str] = []
    for tb in det_toolboxes[:3]:
        if tb not in shortlist_boxes:
            shortlist_boxes.append(tb)
    for hit in toolbox_hits:
        if hit.name not in shortlist_boxes:
            shortlist_boxes.append(hit.name)

    tool_hits = index_mod.query_tools(
        index,
        query,
        top_k=k,
        toolbox_filter=set(shortlist_boxes) or None,
        hivemind_url=hivemind_url,
    )
    if not tool_hits:
        # Embeddings found nothing comparable. If keywords matched a
        # toolbox, hand those tools back at low confidence; else "none".
        if shortlist_boxes:
            box = shortlist_boxes[0]
            tools = tuple(_resolved(e, 0.0) for e in catalog.toolboxes.get(box, ())[:k])
            return ToolResolution(
                schema="Ms4ToolResolution.v1",
                query=query,
                tier=TIER_EMBEDDINGS,
                confidence=0.0,
                toolboxes=tuple(shortlist_boxes[:3]),
                tools=tools,
                catalog_version=catalog.version,
                fallback_reason="embeddings produced no comparable vector; using keyword toolbox",
            )
        return _empty_resolution(query, catalog.version, "no deterministic or embedding match")

    tools = _hits_to_tools(tool_hits, by_name)
    top_score = tools[0].score if tools else 0.0
    resolution = ToolResolution(
        schema="Ms4ToolResolution.v1",
        query=query,
        tier=TIER_EMBEDDINGS,
        confidence=top_score,
        toolboxes=tuple(shortlist_boxes[:3]),
        tools=tools,
        catalog_version=catalog.version,
    )

    # ---- Tier 3: tiny-LLM (only when ambiguous + available) ----
    if (
        classifier is not None
        and _llm_classify_enabled()
        and top_score <= _embeddings_low_confidence()
    ):
        candidate_entries = [by_name[t.name] for t in tools if t.name in by_name]
        try:
            chosen = classifier(query, candidate_entries)
        except Exception as exc:  # noqa: BLE001 — classifier must never break resolution
            log.info("quartermaster classifier raised (keeping embeddings result): %s", exc)
            chosen = None
        if chosen and chosen in by_name:
            chosen_entry = by_name[chosen]
            # Promote the LLM's pick to the front; keep the rest as
            # context. Confidence becomes deterministic-ish because a
            # capable model affirmatively chose it.
            promoted = [_resolved(chosen_entry, 1.0)]
            promoted.extend(t for t in tools if t.name != chosen)
            return ToolResolution(
                schema="Ms4ToolResolution.v1",
                query=query,
                tier=TIER_LLM,
                confidence=max(top_score, _deterministic_confidence()),
                toolboxes=(chosen_entry.toolbox,) + tuple(b for b in shortlist_boxes[:3] if b != chosen_entry.toolbox),
                tools=tuple(promoted[:k]),
                catalog_version=catalog.version,
            )
        # Classifier unsure / unavailable → keep embeddings result with a note.
        return ToolResolution(
            schema="Ms4ToolResolution.v1",
            query=query,
            tier=TIER_EMBEDDINGS,
            confidence=top_score,
            toolboxes=resolution.toolboxes,
            tools=resolution.tools,
            catalog_version=catalog.version,
            fallback_reason="classifier unsure; kept embeddings ranking",
        )

    return resolution


def _hits_to_tools(hits, by_name: dict[str, ToolEntry]) -> tuple[ResolvedTool, ...]:
    out: list[ResolvedTool] = []
    for hit in hits:
        entry = by_name.get(hit.name)
        if entry is None:
            continue
        out.append(_resolved(entry, hit.score))
    return tuple(out)
