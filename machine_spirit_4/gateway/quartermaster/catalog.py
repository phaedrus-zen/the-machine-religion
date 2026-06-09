"""Quartermaster catalog — union of live HiveMind ``tools/list`` and
the MS4 MCP manifest, organised by toolbox.

The catalog is the single source of truth the cascade reasons over.
Build flow:

  1. Query HiveMind ``tools/list`` via the shared
     :func:`machine_spirit_4.gateway.hivemind_state.post_mcp_envelope`
     transport (direct MCP 6105 → HLI proxy fallback + bearer auth).
     Failures are captured in ``Catalog.errors`` rather than raised —
     a Quartermaster that can answer some questions from the MS4
     manifest is more useful than one that can't answer any when the
     cluster is unreachable.
  2. Load the MS4 MCP manifest (``mcp/manifest.json``). It carries
     the authoritative ``kind`` field (``read_only`` |
     ``runtime_action`` | ``read_only_evaluation`` | ``dry_run_only``)
     that the read-only inline gate keys off.
  3. Merge: HiveMind entries inherit ``kind="hivemind_native"``
     unless the MS4 manifest has a corresponding ``ms4.hivemind.*``
     proxy whose ``kind`` we can lift. Every entry gets a derived
     ``toolbox`` and ``cluster`` via :mod:`taxonomy`.
  4. Hash the sorted ids + descriptions → ``version``. Downstream
     callers (embeddings index, evaluation harness, audit logs) key
     their state on this version.

In-memory caching: a single :class:`Catalog` is held per
``hivemind_url`` for ``MS4_QM_CATALOG_TTL_SECS`` seconds (default
300). Tests can clear it via :func:`reset_catalog_cache_for_tests`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import taxonomy


log = logging.getLogger("ms4.gateway.quartermaster.catalog")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return max(minimum, int(raw.strip()))
    except (TypeError, ValueError):
        log.warning(
            "quartermaster.catalog: ignoring non-integer %s=%r; default %d",
            name, raw, default,
        )
        return default


def _catalog_ttl_secs() -> int:
    return _env_int("MS4_QM_CATALOG_TTL_SECS", 300, minimum=5)


def _catalog_fetch_timeout() -> int:
    return _env_int("MS4_QM_CATALOG_FETCH_TIMEOUT", 10, minimum=2)


# ``ms4.*`` tool prefix vs ``hivemind.*`` — used to tag the entry's
# source and to align MS4 proxies with the HiveMind tools they wrap.
_HIVEMIND_PREFIX = "hivemind."
_MS4_PREFIX = "ms4."


# Lower-case ``kind`` values the manifest may carry. ``read_only`` is
# the only MS4 ``kind`` eligible for inline execution; everything else
# routes to the Depth Lobe.
KIND_READ_ONLY = "read_only"
KIND_READ_ONLY_EVAL = "read_only_evaluation"
KIND_RUNTIME_ACTION = "runtime_action"
KIND_DRY_RUN_ONLY = "dry_run_only"
KIND_HIVEMIND_NATIVE = "hivemind_native"

# Tool sources.
SOURCE_HIVEMIND = "hivemind"
SOURCE_MS4 = "ms4"
SOURCE_EXTERNAL_MCP = "external_mcp"


# ---------------------------------------------------------------------------
# Inline read-only eligibility
# ---------------------------------------------------------------------------
#
# The Face Lobe inline fast-path (Phase D) may only execute tools that
# cannot mutate cluster state. There is no single source of truth for
# "is this read-only" across HiveMind-native tools (which carry no MS4
# ``kind``) and MS4 proxies (which do), so we combine two signals:
#
#   1. A conservative *verb allowlist* — the final dotted segment of a
#      tool id must be a verb that NEVER mutates. This is the only
#      signal available for HiveMind-native tools.
#   2. For MS4 proxies (``source == "ms4"``) we ALSO require the
#      manifest ``kind`` to be ``read_only``/``read_only_evaluation``
#      (belt-and-suspenders: the manifest already classified it).
#
# Anything explicitly destructive is excluded regardless of verb:
#   * ``kind`` of ``runtime_action`` / ``dry_run_only``
#   * any ``gated_by`` entry mentioning ``confirm``
#
# The verb allowlist is deliberately conservative. A verb only belongs
# here if NO tool in the catalog that ends in it mutates state. When in
# doubt, leave it out — the tool simply routes to the Depth Lobe (the
# full-catalog safety net) instead of running inline. The hard eval
# gate (zero inline executions of runtime_action/confirm tools) keeps
# this honest.

READ_ONLY_VERBS: frozenset[str] = frozenset({
    # Enumeration / inspection
    "list", "get", "status", "summary", "load", "state", "current",
    "latest", "inventory", "discover", "tags", "catalog", "recommend",
    "matrix", "capabilities", "availability", "active", "health",
    "service_health", "now", "convert",
    # Storage / network read views
    "pools", "volumes", "snapshots", "bridges", "interfaces",
    "attachments",
    # AI / model read views
    "models", "backends", "profiles",
    # Crown read views
    "signal_quality",
    # vGPU read view
    "vgpu_status",
    # PsyKyo / game read views
    "evidence", "gap", "workqueue",
    # File read (no mutation)
    "read",
})


def tool_verb(name: str) -> str:
    """Return the final dotted segment of a tool id (minus the
    ``@vN`` suffix), lower-cased. ``hivemind.gpu.availability@v1`` ->
    ``availability``; ``hivemind.service_health@v1`` -> ``service_health``."""
    if not name:
        return ""
    base = name.split("@", 1)[0]
    parts = [p for p in base.split(".") if p]
    return parts[-1].lower() if parts else ""


def is_destructive_kind(kind: str) -> bool:
    return kind in (KIND_RUNTIME_ACTION, KIND_DRY_RUN_ONLY)


def is_inline_eligible(entry: "ToolEntry") -> bool:
    """True when ``entry`` is safe to execute on the Face Lobe inline
    fast-path. Shared by the cascade (labelling) and the router
    (gating) so the definition lives in exactly one place."""
    if is_destructive_kind(entry.kind):
        return False
    if any("confirm" in g.lower() for g in entry.gated_by):
        return False
    # Imported 3rd-party MCP tools have arbitrary verbs we can't put on
    # a HiveMind-style allowlist, so their trust anchor is the
    # conversion-time classification: ``kind == read_only`` is set ONLY
    # when the upstream declared readOnlyHint AND an operator opted the
    # server in (``allow_inline``). That, plus the router's ethics gate,
    # is the gate. We do NOT apply the verb allowlist to them.
    if entry.source == SOURCE_EXTERNAL_MCP:
        return entry.kind == KIND_READ_ONLY
    if tool_verb(entry.name) not in READ_ONLY_VERBS:
        return False
    # MS4 proxies must ALSO be manifest-classified read-only.
    if entry.source == SOURCE_MS4 and entry.kind not in (KIND_READ_ONLY, KIND_READ_ONLY_EVAL):
        return False
    return True


# ---------------------------------------------------------------------------
# Schema types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolEntry:
    """One tool in the Quartermaster catalog.

    The schema is intentionally narrow: the embeddings tier needs
    ``description``, the deterministic tier needs ``toolbox``, the
    read-only gate needs ``kind``, and inline execution needs the
    ``name`` to dispatch via the typed ``hivemind_tools`` wrappers
    or the MS4 MCP proxy.
    """

    schema: str
    name: str
    toolbox: str
    cluster: str
    description: str
    source: str  # "hivemind" | "ms4" | "external_mcp"
    kind: str  # see KIND_* constants above
    input_schema: dict[str, Any] | None = None
    gated_by: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "name": self.name,
            "toolbox": self.toolbox,
            "cluster": self.cluster,
            "description": self.description,
            "source": self.source,
            "kind": self.kind,
            "input_schema": self.input_schema,
            "gated_by": list(self.gated_by),
        }


@dataclass(frozen=True)
class Catalog:
    """Snapshot of the full tool universe at a point in time."""

    schema: str
    version: str
    built_at: str
    hivemind_url: str
    tools: tuple[ToolEntry, ...]
    toolboxes: dict[str, tuple[ToolEntry, ...]]
    sources: dict[str, int]
    errors: tuple[str, ...]

    def by_name(self) -> dict[str, ToolEntry]:
        return {t.name: t for t in self.tools}

    def read_only_tools(self) -> tuple[ToolEntry, ...]:
        """MS4 tools the manifest explicitly classified ``read_only``.
        Note this does NOT include HiveMind-native read-only tools (which
        carry ``kind='hivemind_native'``) — use :meth:`inline_eligible_tools`
        for the full set the inline fast-path may execute."""
        return tuple(t for t in self.tools if t.kind == KIND_READ_ONLY)

    def inline_eligible_tools(self) -> tuple[ToolEntry, ...]:
        """All tools the Face Lobe inline fast-path may execute —
        HiveMind-native or MS4, verb-allowlisted, non-destructive.
        See :func:`is_inline_eligible`."""
        return tuple(t for t in self.tools if is_inline_eligible(t))

    def cluster_summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for t in self.tools:
            counts[t.cluster] = counts.get(t.cluster, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "version": self.version,
            "built_at": self.built_at,
            "hivemind_url": self.hivemind_url,
            "tool_count": len(self.tools),
            "toolboxes": {tb: [t.name for t in entries] for tb, entries in self.toolboxes.items()},
            "sources": dict(self.sources),
            "cluster_summary": self.cluster_summary(),
            "errors": list(self.errors),
        }


# ---------------------------------------------------------------------------
# Catalog construction
# ---------------------------------------------------------------------------


def _ms4_manifest_path() -> Path:
    """Resolve the canonical MS4 manifest path.

    ``machine_spirit_4/gateway/quartermaster/catalog.py`` → ascend two
    levels (out of ``quartermaster``, then ``gateway``) and read
    ``mcp/manifest.json``. Override with ``MS4_QM_MANIFEST_PATH`` for
    tests.
    """
    pin = os.environ.get("MS4_QM_MANIFEST_PATH", "").strip()
    if pin:
        return Path(pin)
    return Path(__file__).resolve().parents[2] / "mcp" / "manifest.json"


def _load_ms4_manifest() -> tuple[list[dict[str, Any]], list[str]]:
    """Return (tool_entries, errors). Errors are stringified so the
    catalog can surface them in ``Catalog.errors`` instead of raising
    — a missing manifest in a stripped-down install shouldn't kill
    the Quartermaster; the cluster catalog alone is still useful."""
    errors: list[str] = []
    path = _ms4_manifest_path()
    if not path.exists():
        errors.append(f"ms4 manifest not found at {path}")
        return [], errors
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"ms4 manifest unparseable: {exc}")
        return [], errors
    tools = data.get("tools") if isinstance(data, dict) else None
    if not isinstance(tools, list):
        errors.append("ms4 manifest 'tools' missing or not a list")
        return [], errors
    return [t for t in tools if isinstance(t, dict)], errors


def _fetch_hivemind_tools(hivemind_url: str, *, timeout: int) -> tuple[list[dict[str, Any]], list[str]]:
    """Return (tool_entries, errors). Live ``tools/list`` via the
    shared MCP transport. Fail-soft."""
    # Local import keeps the catalog module light at import time —
    # ``hivemind_state`` pulls urllib + uuid + json.
    try:
        from ..hivemind_state import HivemindStateError, post_mcp_envelope
    except Exception as exc:  # pragma: no cover — import guard
        return [], [f"hivemind_state import failed: {exc}"]

    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    try:
        envelope = post_mcp_envelope(hivemind_url, payload, timeout=timeout)
    except HivemindStateError as exc:
        return [], [f"hivemind tools/list unreachable: {exc}"]
    except Exception as exc:  # noqa: BLE001 — defensive: never let catalog build crash a turn
        return [], [f"hivemind tools/list raised: {type(exc).__name__}: {exc}"]

    if not isinstance(envelope, dict):
        return [], ["hivemind tools/list returned non-dict envelope"]
    if envelope.get("error"):
        return [], [f"hivemind tools/list JSON-RPC error: {envelope['error']}"]
    result = envelope.get("result") or {}
    tools = result.get("tools") if isinstance(result, dict) else None
    if not isinstance(tools, list):
        return [], ["hivemind tools/list 'tools' missing or not a list"]
    return [t for t in tools if isinstance(t, dict)], []


def _ms4_entry_to_tool(entry: dict[str, Any]) -> ToolEntry | None:
    name = entry.get("name")
    if not isinstance(name, str) or not name:
        return None
    domain = taxonomy.tool_domain(name)
    canonical = taxonomy.canonical_toolbox(domain)
    cluster = taxonomy.cluster_for_toolbox(canonical)
    kind = str(entry.get("kind") or KIND_RUNTIME_ACTION)
    gated_raw = entry.get("gated_by") or ()
    if isinstance(gated_raw, str):
        gated_tuple: tuple[str, ...] = (gated_raw,)
    elif isinstance(gated_raw, (list, tuple)):
        gated_tuple = tuple(str(g) for g in gated_raw)
    else:
        gated_tuple = ()
    description = str(entry.get("description") or "")
    return ToolEntry(
        schema="Ms4QuartermasterTool.v1",
        name=name,
        toolbox=canonical,
        cluster=cluster,
        description=description,
        source="ms4",
        kind=kind,
        input_schema=None,  # MS4 manifest doesn't ship per-tool schemas here
        gated_by=gated_tuple,
    )


def _hivemind_entry_to_tool(
    entry: dict[str, Any],
    *,
    kind_override: str | None = None,
) -> ToolEntry | None:
    name = entry.get("name")
    if not isinstance(name, str) or not name:
        return None
    domain = taxonomy.tool_domain(name)
    canonical = taxonomy.canonical_toolbox(domain)
    cluster = taxonomy.cluster_for_toolbox(canonical)
    description = str(entry.get("description") or "")
    input_schema = entry.get("inputSchema") if isinstance(entry.get("inputSchema"), dict) else None
    kind = kind_override or KIND_HIVEMIND_NATIVE
    return ToolEntry(
        schema="Ms4QuartermasterTool.v1",
        name=name,
        toolbox=canonical,
        cluster=cluster,
        description=description,
        source="hivemind",
        kind=kind,
        input_schema=input_schema,
        gated_by=(),
    )


def _ext_entry_to_tool(record: dict[str, Any]) -> ToolEntry | None:
    """Convert an imported-MCP-tool record (from the mcp_bridge
    registry) into a catalog :class:`ToolEntry`. Its toolbox is the
    source server; its cluster is always ``external_mcp``. ``kind`` is
    taken verbatim from the record (the bridge already classified it
    fail-closed)."""
    name = record.get("name")
    if not isinstance(name, str) or not name:
        return None
    domain = taxonomy.tool_domain(name)
    canonical = taxonomy.canonical_toolbox(domain)
    kind = str(record.get("kind") or KIND_RUNTIME_ACTION)
    input_schema = record.get("input_schema") if isinstance(record.get("input_schema"), dict) else None
    return ToolEntry(
        schema="Ms4QuartermasterTool.v1",
        name=name,
        toolbox=canonical,
        cluster=taxonomy.EXTERNAL_CLUSTER,
        description=str(record.get("description") or ""),
        source=SOURCE_EXTERNAL_MCP,
        kind=kind,
        input_schema=input_schema,
        gated_by=("external_mcp",),
    )


def _load_imported_mcp_tools() -> tuple[list[dict[str, Any]], list[str]]:
    """Return (records, errors) for imported 3rd-party MCP tools from
    the mcp_bridge registry. Fail-soft — a missing/broken registry
    just means no imported tools, never a crashed catalog."""
    try:
        from ...mcp_bridge.registry import imported_tool_records

        return imported_tool_records(), []
    except Exception as exc:  # noqa: BLE001 — defensive
        return [], [f"mcp_bridge registry load failed: {type(exc).__name__}: {exc}"]


def _version_of(tools: list[ToolEntry]) -> str:
    """Content hash over sorted ids + descriptions + kinds.

    Embeddings + eval-harness state can key on this so a catalog
    change auto-invalidates downstream caches."""
    h = hashlib.sha256()
    for t in sorted(tools, key=lambda t: t.name):
        h.update(t.name.encode("utf-8"))
        h.update(b"\x1f")
        h.update(t.description.encode("utf-8"))
        h.update(b"\x1f")
        h.update(t.kind.encode("utf-8"))
        h.update(b"\x1e")
    return h.hexdigest()[:16]


def build_catalog(
    hivemind_url: str,
    *,
    timeout: int | None = None,
) -> Catalog:
    """Build a fresh :class:`Catalog` by unioning the live HiveMind
    ``tools/list`` with the MS4 manifest. Always returns a catalog —
    failures are captured in :attr:`Catalog.errors`."""
    fetch_timeout = timeout if timeout is not None else _catalog_fetch_timeout()
    ms4_raw, ms4_errs = _load_ms4_manifest()
    hm_raw, hm_errs = _fetch_hivemind_tools(hivemind_url, timeout=fetch_timeout)
    ext_raw, ext_errs = _load_imported_mcp_tools()
    return merge_records(
        hivemind_records=hm_raw,
        ms4_records=ms4_raw,
        imported_records=ext_raw,
        hivemind_url=hivemind_url,
        extra_errors=ms4_errs + hm_errs + ext_errs,
    )


def merge_records(
    *,
    hivemind_records: list[dict[str, Any]],
    ms4_records: list[dict[str, Any]],
    imported_records: list[dict[str, Any]] | None = None,
    hivemind_url: str,
    extra_errors: list[str] | None = None,
    drop_ms4_hivemind_proxies: bool = True,
) -> Catalog:
    """Pure merge core: build a :class:`Catalog` from already-fetched
    raw records (MS4 manifest entries + HiveMind ``tools/list`` entries).

    Separated from :func:`build_catalog` so the eval harness can build
    a reproducible catalog from a frozen on-disk snapshot without
    touching the network. ``build_catalog`` fetches then delegates here.

    ``drop_ms4_hivemind_proxies`` (default True): the ``ms4.hivemind.*``
    MCP tools are thin re-exports of native ``hivemind.*`` tools, present
    so *external* MCP clients can drive HiveMind through MS4's server.
    The Quartermaster's inline executor uses the native ``hivemind_tools``
    path, so the native id is the canonical answer. Including both the
    proxy and the native in retrieval just splits the ranking between two
    ids for the same capability. We drop the proxies from retrieval (no
    capability is lost — the native tools cover them) to keep ranking
    clean. Genuinely-MS4-native tools (``ms4.chat``, ``ms4.identity``,
    ``ms4.desktop``, ``ms4.hermes``, ``ms4.double_agent``, ...) are kept.
    """
    errors: list[str] = list(extra_errors or [])
    tools: list[ToolEntry] = []
    sources_count: dict[str, int] = {"hivemind": 0, "ms4": 0, "external_mcp": 0}

    # ---- MS4 manifest records ----
    for raw in ms4_records:
        if not isinstance(raw, dict):
            continue
        name = raw.get("name")
        if (
            drop_ms4_hivemind_proxies
            and isinstance(name, str)
            and name.startswith("ms4.hivemind.")
        ):
            continue
        entry = _ms4_entry_to_tool(raw)
        if entry is not None:
            tools.append(entry)
            sources_count["ms4"] += 1

    # ---- HiveMind tools/list records ----
    # HiveMind-native tools keep ``kind='hivemind_native'``; their
    # inline read-only eligibility is decided by the shared verb
    # allowlist (:func:`is_inline_eligible`), not by a per-tool kind.
    seen_names = {t.name for t in tools}
    for raw in hivemind_records:
        name = raw.get("name") if isinstance(raw, dict) else None
        if not isinstance(name, str) or name in seen_names:
            continue
        entry = _hivemind_entry_to_tool(raw)
        if entry is not None:
            tools.append(entry)
            seen_names.add(name)
            sources_count["hivemind"] += 1

    # ---- Imported 3rd-party MCP tools (mcp_bridge registry) ----
    for raw in imported_records or []:
        if not isinstance(raw, dict):
            continue
        name = raw.get("name")
        if not isinstance(name, str) or name in seen_names:
            continue
        entry = _ext_entry_to_tool(raw)
        if entry is not None:
            tools.append(entry)
            seen_names.add(name)
            sources_count["external_mcp"] += 1

    return _finalize_catalog(tools, sources_count, errors, hivemind_url)


def _finalize_catalog(
    tools: list[ToolEntry],
    sources_count: dict[str, int],
    errors: list[str],
    hivemind_url: str,
) -> Catalog:
    # ---- Group by toolbox + version ----
    by_toolbox: dict[str, list[ToolEntry]] = {}
    for entry in tools:
        by_toolbox.setdefault(entry.toolbox, []).append(entry)

    toolboxes_frozen: dict[str, tuple[ToolEntry, ...]] = {
        tb: tuple(sorted(entries, key=lambda t: t.name))
        for tb, entries in sorted(by_toolbox.items())
    }
    tools_frozen = tuple(sorted(tools, key=lambda t: t.name))
    version = _version_of(list(tools_frozen))
    built_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    return Catalog(
        schema="Ms4QuartermasterCatalog.v1",
        version=version,
        built_at=built_at,
        hivemind_url=hivemind_url,
        tools=tools_frozen,
        toolboxes=toolboxes_frozen,
        sources=sources_count,
        errors=tuple(errors),
    )


# ---------------------------------------------------------------------------
# In-memory cache (per hivemind_url)
# ---------------------------------------------------------------------------


@dataclass
class _CacheEntry:
    catalog: Catalog
    expires_at: float


_CACHE: dict[str, _CacheEntry] = {}
_CACHE_LOCK = threading.Lock()


def get_catalog(
    hivemind_url: str,
    *,
    ttl_secs: int | None = None,
    force_refresh: bool = False,
) -> Catalog:
    """Return the cached catalog for ``hivemind_url``, rebuilding when
    expired or when ``force_refresh=True``. Thread-safe (refresh
    behind a single lock so concurrent callers don't dogpile)."""
    ttl = ttl_secs if ttl_secs is not None else _catalog_ttl_secs()
    now = time.time()
    with _CACHE_LOCK:
        entry = _CACHE.get(hivemind_url)
        if entry is not None and not force_refresh and entry.expires_at > now:
            return entry.catalog
        catalog = build_catalog(hivemind_url)
        _CACHE[hivemind_url] = _CacheEntry(catalog=catalog, expires_at=now + ttl)
        return catalog


def catalog_for_hivemind_url(hivemind_url: str) -> Catalog:
    """Alias for :func:`get_catalog` with default TTL.

    Provided so callers reading the cached snapshot can name the
    intent without passing keyword args every time."""
    return get_catalog(hivemind_url)


def reset_catalog_cache_for_tests() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()
