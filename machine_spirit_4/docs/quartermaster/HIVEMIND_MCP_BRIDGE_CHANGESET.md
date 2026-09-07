# HiveMind change-set: cluster-wide 3rd-party MCP importer/bridge

**Status:** PROPOSED — approval-gated. This is the exact change set for the
HiveMind coding agent / operator to host the MCP importer at the substrate
level. **MS4 must not modify the HiveMind repo or restart Warden without
explicit approval.** The MS4-side importer (this repo's
`machine_spirit_4/mcp_bridge/`) already works standalone and ships
independently; this change set is the optional "make it cluster-wide" step.

## Why host it in HiveMind too

The MS4 bridge imports 3rd-party MCP servers into *one MS4 instance*. Hosting
the same capability in HiveMind's `menta_mcp_gateway` means every MCP client
on the cluster (all MS4 spirits, Hermes workers, Oracle, future agents) shares
one imported-tool registry — register the GitHub MCP server once, and
`hivemind.ext.github.*` tools appear for everyone, auto-updating. Same
rationale as `hivemind.tools.search@v1` (see
`HIVEMIND_TOOLS_SEARCH_CHANGESET.md`).

## Design

HiveMind's MCP gateway (`menta_mcp_gateway/deps/tools.json`) is a proxy: each
tool maps to a backend service endpoint. To host imported tools cluster-wide:

1. A small **bridge service** (or an extension of the gateway) that:
   - Holds a registry of upstream MCP servers (stdio/http) — same config shape
     as MS4's `ImportedServer` (`server_id`, `transport`, `command`/`args`/`env`
     + `env_passthrough`, or `url`/`headers`, `allow_inline`, `enabled`).
   - On register/refresh: connects, `tools/list`, converts each tool to
     `hivemind.ext.<server>.<tool>@v1`, and **dynamically registers** it in the
     gateway's tool table (the gateway already supports a tool list; this adds
     dynamic entries with `backend` pointing at the bridge's `/ext/call`).
   - On `tools/call` for an `ext.*` tool: proxies to the owning upstream and
     returns the result.
2. Admin tools to manage it (mirroring MS4's REST surface):

```json
{
  "name": "hivemind.mcp_imports.register@v1",
  "stability": "experimental",
  "role": "admin",
  "description": "Register a 3rd-party MCP server (stdio or http); imports its tools cluster-wide as hivemind.ext.<server>.<tool>@v1.",
  "inputSchema": {
    "type": "object",
    "required": ["server_id", "transport"],
    "properties": {
      "server_id": {"type": "string"},
      "transport": {"type": "string", "enum": ["stdio", "http"]},
      "command": {"type": "string"}, "args": {"type": "array", "items": {"type": "string"}},
      "env": {"type": "object"}, "env_passthrough": {"type": "array", "items": {"type": "string"}},
      "url": {"type": "string"}, "headers": {"type": "object"},
      "allow_inline": {"type": "boolean", "default": false}
    }
  }
},
{ "name": "hivemind.mcp_imports.list@v1",    "role": "viewer", "description": "List imported MCP servers and their tool counts/health." },
{ "name": "hivemind.mcp_imports.refresh@v1", "role": "operator", "description": "Re-fetch an imported server's tools." },
{ "name": "hivemind.mcp_imports.remove@v1",  "role": "operator", "description": "Remove an imported MCP server and its tools." }
```

## Safety (must port from the MS4 design)

- Imported tools are **fail-closed**: `read_only` (inline-eligible) ONLY when
  the upstream declares `readOnlyHint`, is not `destructiveHint`, AND the
  operator set `allow_inline`. Everything else routes through normal
  per-call authority. Preserve the upstream `annotations` so downstream
  ethics/role gating can see them.
- Launching stdio subprocesses cluster-wide is powerful: gate `register`
  behind the `admin` role, run upstreams under the same isolation Warden
  uses for managed services, and pass secrets only via `env_passthrough`
  (names; values resolved from the host env at launch, never persisted).
- Namespacing `hivemind.ext.<server>.<tool>@v1` prevents collisions with
  native `hivemind.*` tools.

## Validation (before deploy)

- Register a known reference server (e.g. `@modelcontextprotocol/server-everything`),
  confirm its tools appear as `hivemind.ext.everything.*` and a `tools/call`
  round-trips.
- Confirm a destructive imported tool is NOT marked read-only.

## Deploy (the gated part)

1. Add the bridge service (or gateway extension) + the four admin tools above.
2. Wire dynamic tool registration into the gateway tool table.
3. Restart the MCP gateway / Warden-managed service.

Steps 1-3 are an approval-gated runtime-registry + service change. Do not
perform them from MS4. Once live, MS4's Quartermaster catalog picks up the new
`hivemind.ext.*` tools automatically via the normal `tools/list` union.

## MS4 side (already implemented, no HiveMind dependency)

`machine_spirit_4/mcp_bridge/` (upstream client, convert, registry, proxy) +
the gateway REST endpoints + the Quartermaster catalog third source. MS4
imports work today against any stdio/http MCP server without this change set.
