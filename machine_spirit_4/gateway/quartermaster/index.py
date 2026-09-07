"""Tool index — vectorised retrieval for the embeddings tier.

Two backends, both producing comparable cosine-comparable vectors:

* **TF-IDF** (deterministic, offline, no deps) — the default. Builds
  a vocab over the catalog at index-build time and scores queries
  via cosine similarity over IDF-weighted term-frequency vectors.
  Used by the eval harness for reproducibility and by the cascade
  whenever the HiveMind cluster is unreachable.
* **HiveMind embeddings** — wraps
  ``hivemind.embeddings.create@v1`` via the
  :mod:`hivemind_tools` typed wrapper. Higher-quality vectors when
  the cluster is reachable. Opt-in via ``MS4_QM_USE_HM_EMBEDDINGS=1``
  (default off so first-boot doesn't require an embeddings GIM to
  be loaded).

Both backends index two granularities:

1. **Per-tool**: each tool's name + description.
2. **Per-toolbox**: the concatenation of every tool inside it.

So a query can shortlist the toolbox first (cheap retrieval over
~25 vectors) and the tool second (~5-15 vectors per shortlisted
toolbox). This is the two-stage retrieval the spec calls for and
keeps the per-query work small.

On-disk cache: ``machine_spirit_4/runtime/quartermaster_index/<version>.json``.
Version is the catalog's content hash, so a catalog change auto-
invalidates the cache.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from . import catalog as catalog_mod
from .catalog import Catalog


log = logging.getLogger("ms4.gateway.quartermaster.index")


# ---------------------------------------------------------------------------
# Constants + helpers
# ---------------------------------------------------------------------------


BACKEND_TFIDF = "tfidf"
BACKEND_HIVEMIND = "hivemind"


def _runtime_dir() -> Path:
    pin = os.environ.get("MS4_QM_INDEX_DIR", "").strip()
    if pin:
        return Path(pin)
    # machine_spirit_4/gateway/quartermaster/index.py  → ascend 2 → machine_spirit_4
    return Path(__file__).resolve().parents[2] / "runtime" / "quartermaster_index"


def _selected_backend() -> str:
    """Resolve the embeddings backend. HiveMind only when explicitly
    opted in — keeps first-boot working without an embeddings GIM
    loaded, keeps tests + the eval harness deterministic by default."""
    raw = os.environ.get("MS4_QM_USE_HM_EMBEDDINGS", "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return BACKEND_HIVEMIND
    return BACKEND_TFIDF


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization. Drops 1-character tokens
    and the version suffix tokens (``v1``) so they don't dominate the
    vocab."""
    if not text:
        return []
    out: list[str] = []
    for raw in re.findall(r"[A-Za-z0-9_]+", text.lower()):
        if len(raw) <= 1:
            continue
        if raw in {"v1", "v2", "v3"}:
            continue
        out.append(raw)
    return out


# ---------------------------------------------------------------------------
# Schema types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolIndex:
    """Vector index of catalog entries. Cosine-comparable across both
    backends."""

    schema: str
    version: str  # mirrors Catalog.version
    backend: str  # BACKEND_TFIDF | BACKEND_HIVEMIND
    built_at: str
    # For TF-IDF backend only: stable vocab ordering (token -> idx).
    vocab: tuple[str, ...]
    idf: tuple[float, ...]
    # Vectors: sparse for TF-IDF (dict[idx]->float), dense for HM
    # (list[float]). We store both as plain dict[str, list[float]]
    # over the public name; for TF-IDF we expand to a dense list at
    # build time (vocab is small — a few hundred tokens at most).
    tool_vectors: dict[str, list[float]]
    toolbox_vectors: dict[str, list[float]]
    tool_to_toolbox: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "version": self.version,
            "backend": self.backend,
            "built_at": self.built_at,
            "vocab": list(self.vocab),
            "idf": list(self.idf),
            "tool_vectors": self.tool_vectors,
            "toolbox_vectors": self.toolbox_vectors,
            "tool_to_toolbox": self.tool_to_toolbox,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolIndex":
        return cls(
            schema=data["schema"],
            version=data["version"],
            backend=data["backend"],
            built_at=data["built_at"],
            vocab=tuple(data.get("vocab") or ()),
            idf=tuple(data.get("idf") or ()),
            tool_vectors={k: list(v) for k, v in (data.get("tool_vectors") or {}).items()},
            toolbox_vectors={k: list(v) for k, v in (data.get("toolbox_vectors") or {}).items()},
            tool_to_toolbox=dict(data.get("tool_to_toolbox") or {}),
        )


@dataclass(frozen=True)
class IndexHit:
    """One scored retrieval result."""

    name: str  # tool name or toolbox name
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "score": round(self.score, 6)}


# ---------------------------------------------------------------------------
# Vector math
# ---------------------------------------------------------------------------


def _dot(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        # Vectors from different vocabularies: zero similarity.
        return 0.0
    return sum(x * y for x, y in zip(a, b))


def _norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v)) or 0.0


def _cosine(a: list[float], b: list[float]) -> float:
    na = _norm(a)
    nb = _norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return _dot(a, b) / (na * nb)


# ---------------------------------------------------------------------------
# TF-IDF backend
# ---------------------------------------------------------------------------


def _tool_document(entry) -> str:
    """Text representation a tool gets indexed under. Uses the name
    (so domain/verb tokens count), the toolbox (so a single keyword
    can pull the whole bucket), and the description verbatim."""
    parts = [entry.name, entry.toolbox, entry.cluster, entry.description]
    return " ".join(p for p in parts if p)


def _toolbox_document(entries: Iterable) -> str:
    return " ".join(_tool_document(e) for e in entries)


def _build_tfidf(catalog: Catalog) -> ToolIndex:
    # Build vocab + per-tool term frequencies.
    per_tool_tf: dict[str, dict[str, int]] = {}
    df: dict[str, int] = {}
    for entry in catalog.tools:
        tokens = _tokenize(_tool_document(entry))
        if not tokens:
            per_tool_tf[entry.name] = {}
            continue
        tf: dict[str, int] = {}
        for tok in tokens:
            tf[tok] = tf.get(tok, 0) + 1
        per_tool_tf[entry.name] = tf
        for tok in set(tokens):
            df[tok] = df.get(tok, 0) + 1

    # Stable vocab ordering.
    vocab = tuple(sorted(df.keys()))
    vocab_idx = {tok: i for i, tok in enumerate(vocab)}
    n_docs = max(1, len(catalog.tools))
    idf = tuple(math.log((1 + n_docs) / (1 + df[tok])) + 1.0 for tok in vocab)

    # Per-tool dense vectors (vocab is bounded by tool descriptions —
    # typically a few hundred tokens, so a dense vector is cheap).
    tool_vectors: dict[str, list[float]] = {}
    for entry in catalog.tools:
        vec = [0.0] * len(vocab)
        tf = per_tool_tf.get(entry.name, {})
        for tok, count in tf.items():
            j = vocab_idx.get(tok)
            if j is None:
                continue
            vec[j] = float(count) * idf[j]
        tool_vectors[entry.name] = vec

    # Per-toolbox vectors = sum of member-tool TFs, IDF-weighted.
    toolbox_vectors: dict[str, list[float]] = {}
    for toolbox, entries in catalog.toolboxes.items():
        vec = [0.0] * len(vocab)
        for entry in entries:
            tf = per_tool_tf.get(entry.name, {})
            for tok, count in tf.items():
                j = vocab_idx.get(tok)
                if j is None:
                    continue
                vec[j] += float(count) * idf[j]
        toolbox_vectors[toolbox] = vec

    tool_to_toolbox = {t.name: t.toolbox for t in catalog.tools}
    return ToolIndex(
        schema="Ms4QuartermasterIndex.v1",
        version=catalog.version,
        backend=BACKEND_TFIDF,
        built_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        vocab=vocab,
        idf=idf,
        tool_vectors=tool_vectors,
        toolbox_vectors=toolbox_vectors,
        tool_to_toolbox=tool_to_toolbox,
    )


def _embed_query_tfidf(index: ToolIndex, query: str) -> list[float]:
    tokens = _tokenize(query)
    vec = [0.0] * len(index.vocab)
    if not tokens:
        return vec
    counts: dict[str, int] = {}
    for tok in tokens:
        counts[tok] = counts.get(tok, 0) + 1
    vocab_idx = {tok: i for i, tok in enumerate(index.vocab)}
    for tok, count in counts.items():
        j = vocab_idx.get(tok)
        if j is None:
            continue
        vec[j] = float(count) * index.idf[j]
    return vec


# ---------------------------------------------------------------------------
# HiveMind backend (opt-in)
# ---------------------------------------------------------------------------


def _build_hivemind(catalog: Catalog, *, hivemind_url: str) -> ToolIndex:
    """Build the index by asking HiveMind to embed every tool +
    toolbox document. Falls through to TF-IDF (logged warning) on
    any failure so first-boot can still answer queries."""
    try:
        from .. import hivemind_tools
    except Exception as exc:  # pragma: no cover — import guard
        log.warning("quartermaster.index: hivemind_tools import failed (%s); falling back to TF-IDF", exc)
        return _build_tfidf(catalog)

    # Build the list of documents in a stable order. The HiveMind
    # call shape accepts a list of strings; we ship tools first, then
    # toolboxes, so a partial failure can still salvage the per-tool
    # vectors.
    tool_names = [t.name for t in catalog.tools]
    tool_docs = [_tool_document(t) for t in catalog.tools]
    toolbox_names = list(catalog.toolboxes.keys())
    toolbox_docs = [_toolbox_document(catalog.toolboxes[tb]) for tb in toolbox_names]

    try:
        tool_resp = hivemind_tools.embeddings_create(hivemind_url, input_text=tool_docs)
        toolbox_resp = hivemind_tools.embeddings_create(hivemind_url, input_text=toolbox_docs)
    except Exception as exc:
        log.warning("quartermaster.index: HiveMind embeddings_create failed (%s); falling back to TF-IDF", exc)
        return _build_tfidf(catalog)

    tool_vectors = _extract_vectors(tool_resp, expected=len(tool_names))
    toolbox_vectors = _extract_vectors(toolbox_resp, expected=len(toolbox_names))
    if not tool_vectors or not toolbox_vectors:
        log.warning("quartermaster.index: empty embeddings response; falling back to TF-IDF")
        return _build_tfidf(catalog)

    tool_vec_map = dict(zip(tool_names, tool_vectors))
    toolbox_vec_map = dict(zip(toolbox_names, toolbox_vectors))
    tool_to_toolbox = {t.name: t.toolbox for t in catalog.tools}
    return ToolIndex(
        schema="Ms4QuartermasterIndex.v1",
        version=catalog.version,
        backend=BACKEND_HIVEMIND,
        built_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        vocab=(),  # not used by the HiveMind backend
        idf=(),
        tool_vectors=tool_vec_map,
        toolbox_vectors=toolbox_vec_map,
        tool_to_toolbox=tool_to_toolbox,
    )


def _extract_vectors(resp: Any, *, expected: int) -> list[list[float]] | None:
    """Unwrap an OpenAI-shaped embeddings response into a flat list
    of dense vectors. Returns None on any shape mismatch."""
    if not isinstance(resp, dict):
        return None
    data = resp.get("data")
    if not isinstance(data, list):
        return None
    vectors: list[list[float]] = []
    for entry in data:
        if not isinstance(entry, dict):
            return None
        vec = entry.get("embedding")
        if not isinstance(vec, list):
            return None
        try:
            vectors.append([float(x) for x in vec])
        except (TypeError, ValueError):
            return None
    if len(vectors) != expected:
        return None
    return vectors


def _embed_query_hivemind(index: ToolIndex, query: str, *, hivemind_url: str) -> list[float] | None:
    """Embed a query with the HiveMind backend. Returns None on
    failure (caller falls back to TF-IDF or treats as 'unsure')."""
    try:
        from .. import hivemind_tools
        resp = hivemind_tools.embeddings_create(hivemind_url, input_text=[query])
    except Exception as exc:
        log.info("quartermaster.index: HiveMind query embedding failed (%s)", exc)
        return None
    vecs = _extract_vectors(resp, expected=1)
    if not vecs:
        return None
    return vecs[0]


# ---------------------------------------------------------------------------
# Build + cache
# ---------------------------------------------------------------------------


_INDEX_CACHE: dict[str, ToolIndex] = {}
_INDEX_LOCK = threading.Lock()


def build_index(
    catalog: Catalog,
    *,
    hivemind_url: str | None = None,
    backend: str | None = None,
) -> ToolIndex:
    """Build a fresh index over ``catalog``. Backend defaults to the
    env-resolved selection; pass ``backend`` explicitly to pin (the
    eval harness pins TF-IDF for reproducibility)."""
    chosen = backend or _selected_backend()
    if chosen == BACKEND_HIVEMIND:
        if not hivemind_url:
            log.warning("quartermaster.index: HiveMind backend requested but no hivemind_url given; falling back to TF-IDF")
            return _build_tfidf(catalog)
        return _build_hivemind(catalog, hivemind_url=hivemind_url)
    return _build_tfidf(catalog)


def get_index(
    catalog: Catalog,
    *,
    hivemind_url: str | None = None,
    backend: str | None = None,
    use_disk_cache: bool = True,
) -> ToolIndex:
    """Return the index for ``catalog``, building it if needed.

    Lookup order:
      1. In-memory cache keyed by ``catalog.version`` + backend.
      2. On-disk cache (JSON) under :func:`_runtime_dir`.
      3. Fresh build.

    Thread-safe."""
    chosen = backend or _selected_backend()
    cache_key = f"{catalog.version}::{chosen}"
    with _INDEX_LOCK:
        cached = _INDEX_CACHE.get(cache_key)
        if cached is not None:
            return cached
        if use_disk_cache:
            disk = _load_from_disk(catalog.version, chosen)
            if disk is not None:
                _INDEX_CACHE[cache_key] = disk
                return disk
        idx = build_index(catalog, hivemind_url=hivemind_url, backend=chosen)
        _INDEX_CACHE[cache_key] = idx
        if use_disk_cache:
            _save_to_disk(idx)
        return idx


def _load_from_disk(version: str, backend: str) -> ToolIndex | None:
    path = _runtime_dir() / f"{version}.{backend}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return ToolIndex.from_dict(data)
    except Exception as exc:
        log.warning("quartermaster.index: failed to load cached index %s (%s); rebuilding", path, exc)
        return None


def _save_to_disk(index: ToolIndex) -> None:
    try:
        runtime = _runtime_dir()
        runtime.mkdir(parents=True, exist_ok=True)
        path = runtime / f"{index.version}.{index.backend}.json"
        path.write_text(json.dumps(index.to_dict(), ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        log.warning("quartermaster.index: failed to persist index to disk (%s)", exc)


def reset_index_cache_for_tests() -> None:
    with _INDEX_LOCK:
        _INDEX_CACHE.clear()


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


def _embed_query(index: ToolIndex, query: str, *, hivemind_url: str | None) -> list[float] | None:
    if index.backend == BACKEND_HIVEMIND:
        if not hivemind_url:
            return None
        return _embed_query_hivemind(index, query, hivemind_url=hivemind_url)
    return _embed_query_tfidf(index, query)


def query_toolboxes(
    index: ToolIndex,
    query: str,
    *,
    top_k: int = 3,
    hivemind_url: str | None = None,
) -> list[IndexHit]:
    """Top-k toolboxes for the query, sorted by descending cosine.

    Returns an empty list (not an error) when query embedding fails
    or the catalog has no tools — the caller treats that as "unsure"
    and falls back to a wider tier."""
    qvec = _embed_query(index, query, hivemind_url=hivemind_url)
    if qvec is None or not any(qvec):
        return []
    scored: list[IndexHit] = []
    for name, vec in index.toolbox_vectors.items():
        s = _cosine(qvec, vec)
        if s > 0.0:
            scored.append(IndexHit(name=name, score=s))
    scored.sort(key=lambda h: h.score, reverse=True)
    return scored[: max(1, top_k)]


def query_tools(
    index: ToolIndex,
    query: str,
    *,
    top_k: int = 5,
    toolbox_filter: Iterable[str] | None = None,
    hivemind_url: str | None = None,
) -> list[IndexHit]:
    """Top-k tools for the query, optionally restricted to a
    ``toolbox_filter`` (e.g. the toolboxes from :func:`query_toolboxes`)."""
    qvec = _embed_query(index, query, hivemind_url=hivemind_url)
    if qvec is None or not any(qvec):
        return []
    allowed: set[str] | None = None
    if toolbox_filter is not None:
        allowed = set(toolbox_filter)
    scored: list[IndexHit] = []
    for name, vec in index.tool_vectors.items():
        if allowed is not None:
            tb = index.tool_to_toolbox.get(name)
            if tb not in allowed:
                continue
        s = _cosine(qvec, vec)
        if s > 0.0:
            scored.append(IndexHit(name=name, score=s))
    scored.sort(key=lambda h: h.score, reverse=True)
    return scored[: max(1, top_k)]
