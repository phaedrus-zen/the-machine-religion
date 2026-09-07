"""Quartermaster — MS4 tool router.

Confidence-gated cascade (deterministic → embeddings → tiny-LLM,
fail-safe) that retrieves only relevant tools per request instead of
dumping the full 75-tool MS4 / 154-tool HiveMind catalog into the
model's context. See
``docs/superpowers/specs/2026-05-30-quartermaster-tool-router-design.md``
for full design + code evidence.

Public API (re-exported here):
  * :func:`get_catalog` — current Quartermaster catalog snapshot.
  * :class:`ToolEntry`, :class:`Catalog` — schema types.
  * :func:`tool_domain` — derive toolbox from a tool id.
  * :data:`TOOL_SHED_CLUSTERS` — static cluster map.
  * :func:`resolve` — tiered retrieval (Phase A3).
  * :func:`decide`, :class:`ToolRouter` — policy layer (Phase C).
"""

from __future__ import annotations

from .catalog import (
    KIND_DRY_RUN_ONLY,
    KIND_HIVEMIND_NATIVE,
    KIND_READ_ONLY,
    KIND_READ_ONLY_EVAL,
    KIND_RUNTIME_ACTION,
    READ_ONLY_VERBS,
    Catalog,
    ToolEntry,
    build_catalog,
    catalog_for_hivemind_url,
    get_catalog,
    is_destructive_kind,
    is_inline_eligible,
    merge_records,
    reset_catalog_cache_for_tests,
    tool_verb,
)
from .cascade import (
    HM_SEARCH_TOOL,
    TIER_DETERMINISTIC,
    TIER_EMBEDDINGS,
    TIER_HM_SEARCH,
    TIER_LLM,
    TIER_NONE,
    ResolvedTool,
    ToolClassifier,
    ToolResolution,
    make_llm_tool_classifier,
    resolve,
)
from .executor import execute_inline_tool, format_inline_block
from .index import (
    BACKEND_HIVEMIND,
    BACKEND_TFIDF,
    IndexHit,
    ToolIndex,
    build_index,
    get_index,
    query_toolboxes,
    query_tools,
    reset_index_cache_for_tests,
)
from .router import (
    VERDICT_DEPTH,
    VERDICT_INLINE,
    VERDICT_NONE,
    EthicsEvaluator,
    ToolRouteDecision,
    ToolRouter,
    decide,
    default_router,
)
from .taxonomy import (
    TOOL_SHED_CLUSTERS,
    canonical_toolbox,
    cluster_for_toolbox,
    hermes_toolsets_for_query,
    keywords_for_toolbox,
    tool_domain,
)


__all__ = [
    "BACKEND_HIVEMIND",
    "BACKEND_TFIDF",
    "Catalog",
    "IndexHit",
    "KIND_DRY_RUN_ONLY",
    "KIND_HIVEMIND_NATIVE",
    "KIND_READ_ONLY",
    "KIND_READ_ONLY_EVAL",
    "KIND_RUNTIME_ACTION",
    "READ_ONLY_VERBS",
    "ResolvedTool",
    "HM_SEARCH_TOOL",
    "TIER_DETERMINISTIC",
    "TIER_EMBEDDINGS",
    "TIER_HM_SEARCH",
    "TIER_LLM",
    "TIER_NONE",
    "TOOL_SHED_CLUSTERS",
    "ToolClassifier",
    "ToolEntry",
    "ToolIndex",
    "ToolResolution",
    "ToolRouteDecision",
    "ToolRouter",
    "EthicsEvaluator",
    "VERDICT_DEPTH",
    "VERDICT_INLINE",
    "VERDICT_NONE",
    "build_catalog",
    "build_index",
    "canonical_toolbox",
    "catalog_for_hivemind_url",
    "cluster_for_toolbox",
    "decide",
    "default_router",
    "execute_inline_tool",
    "format_inline_block",
    "get_catalog",
    "get_index",
    "hermes_toolsets_for_query",
    "is_destructive_kind",
    "is_inline_eligible",
    "keywords_for_toolbox",
    "make_llm_tool_classifier",
    "merge_records",
    "query_toolboxes",
    "query_tools",
    "reset_catalog_cache_for_tests",
    "reset_index_cache_for_tests",
    "resolve",
    "tool_domain",
    "tool_verb",
]
