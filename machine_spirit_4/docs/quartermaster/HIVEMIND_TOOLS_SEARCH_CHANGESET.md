# HiveMind change-set: `hivemind.tools.search@v1` + `hivemind.tools.toolboxes@v1`

**Status:** PROPOSED — approval-gated. This document is the exact change set for the
HiveMind coding agent / operator. **MS4 must not modify the HiveMind repo or restart
Warden without explicit approval** (workspace rule + plan Phase B gate). MS4's side
(delegation + local fallback) is already implemented and ships independently; these
two tools are an *enhancement* that lets every MCP client share one auto-updating tool
search instead of each rebuilding its own index.

## Why this belongs in HiveMind (substrate), not MS4

The tool catalog lives in HiveMind (`menta_mcp_gateway/deps/tools.json`, 150+ tools as of
2026-05-30). Co-locating the search index with the catalog means it auto-updates the
moment HiveMind ships a tool, and every client (MS4 Face Lobe, the Depth Lobe / Hermes,
Oracle, future agents) gets the same retrieval without re-deriving the taxonomy. This
mirrors why model-routing and the resource broker live in HiveMind, not in each caller.

MS4's `quartermaster` package already implements the full engine locally (catalog union,
toolbox taxonomy, TF-IDF/embeddings index, confidence-gated cascade). When these HiveMind
tools exist, MS4's `cascade.py` delegates to them and falls back to the local engine when
they're absent — so this change set is purely additive and non-breaking.

## Backend

The MCP gateway (`menta_mcp_gateway/deps/tools.json`) is a thin proxy: each tool maps to a
backend service endpoint. The natural home for tool search is the gateway's own catalog
host. Two options:

- **(A, recommended)** A small read-only handler co-located with the MCP gateway that
  indexes `tools.json` (it already has the authoritative tool list + descriptions) and
  answers search queries. Lowest latency, always in sync, no extra service.
- **(B)** Extend `menta_hli` (`:6089`) with `/tools/search` + `/tools/toolboxes` endpoints
  that read the gateway's `tools.json`.

Either way the **MCP tool contract below is identical**; only the `backend` block differs.

## tools.json entries to add (matching existing conventions)

```json
{
  "name": "hivemind.tools.search@v1",
  "stability": "experimental",
  "description": "Search the HiveMind MCP tool catalog and return the tools most relevant to a natural-language query, ranked by relevance. Use this to find the right tool without loading every tool schema into context. Returns ranked tool ids + descriptions + their toolbox.",
  "phase": 1,
  "role": "viewer",
  "timeout_ms": 5000,
  "backend": {
    "service": "menta_mcp_gateway",
    "base_url": "http://127.0.0.1:6105",
    "endpoint": "/tools/search",
    "method": "POST"
  },
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": { "type": "string", "description": "Natural-language description of what you want to do." },
      "top_k": { "type": "integer", "description": "Max tools to return (default 5).", "default": 5 },
      "toolbox": { "type": "string", "description": "Optional: restrict results to one toolbox (domain) e.g. 'vm'." }
    },
    "required": ["query"]
  },
  "outputSchema": {
    "type": "object",
    "properties": {
      "query": { "type": "string" },
      "catalog_version": { "type": "string", "description": "Content hash of the catalog the index was built from." },
      "tools": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "name": { "type": "string", "description": "Tool id, e.g. hivemind.vm.list@v1" },
            "toolbox": { "type": "string", "description": "Derived domain, e.g. vm" },
            "score": { "type": "number", "description": "Relevance score 0..1" },
            "description": { "type": "string" }
          }
        }
      }
    }
  }
},
{
  "name": "hivemind.tools.toolboxes@v1",
  "stability": "experimental",
  "description": "List the HiveMind tool catalog organised into toolboxes (tool domains derived from the hivemind.<domain>.<verb> naming) and capability clusters. Use to get an overview of what categories of tools exist before searching within one.",
  "phase": 1,
  "role": "viewer",
  "timeout_ms": 5000,
  "backend": {
    "service": "menta_mcp_gateway",
    "base_url": "http://127.0.0.1:6105",
    "endpoint": "/tools/toolboxes",
    "method": "GET"
  },
  "inputSchema": { "type": "object", "properties": {} },
  "outputSchema": {
    "type": "object",
    "properties": {
      "catalog_version": { "type": "string" },
      "tool_count": { "type": "integer" },
      "toolboxes": {
        "type": "object",
        "description": "Map of toolbox name -> list of tool ids.",
        "additionalProperties": { "type": "array", "items": { "type": "string" } }
      }
    }
  }
}
```

## Backend handler contract (matches MS4's local engine so results are interchangeable)

- **Toolbox derivation:** for a tool id `hivemind.<domain>.<verb>@v1`, the toolbox is
  `<domain>` (collapse multi-segment domains to the top-level segment; e.g.
  `hivemind.crown.triggers.list@v1` -> `crown`). This is exactly
  `machine_spirit_4/gateway/quartermaster/taxonomy.py:tool_domain`.
- **Index:** TF-IDF over each tool's `name + description` is sufficient and dependency-free
  (MS4's `index.py` is a reference implementation). Embeddings are optional.
- **`catalog_version`:** content hash over sorted `(name, description)` so clients can cache
  on it (MS4 uses a 16-hex sha256 prefix; any stable hash is fine).
- **Determinism:** identical catalog -> identical ranking (so the contract test is stable).

## Validation (before deploy)

- Contract test: `POST /tools/search {"query":"list my vms"}` returns `hivemind.vm.list@v1`
  in the top 3; `GET /tools/toolboxes` returns a `vm` toolbox containing the vm tools.
- Reuse MS4's golden set (`machine_spirit_4/gateway/quartermaster/eval/golden_set.jsonl`)
  as a cross-check: the HiveMind handler should hit comparable recall@3.

## Deploy (the gated part)

1. Edit `menta_mcp_gateway/deps/tools.json` (add the two entries above).
2. Implement the backend handler (option A or B).
3. Restart the MCP gateway / Warden-managed service so the new tools register.

Steps 1-3 are an **approval-gated runtime-registry + service change**. Do not perform them
from MS4. When approved, MS4 picks the tools up automatically (see Phase B MS4 delegation).

## MS4 side (already implemented, no HiveMind dependency)

`machine_spirit_4/gateway/quartermaster/cascade.py` checks for `hivemind.tools.search@v1`
in the live catalog; when present it delegates retrieval to it, else it uses the local
TF-IDF/embeddings engine. Tests: `tests/ms4_quartermaster/test_hm_search_delegation.py`.
