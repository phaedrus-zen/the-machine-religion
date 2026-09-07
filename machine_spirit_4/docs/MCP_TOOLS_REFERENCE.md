# MS4 MCP Tools Reference

**Service:** `ms4-mcp-server`  
**Default endpoint:** `http://127.0.0.1:9181/mcp`  
**Protocol:** JSON-RPC 2.0, MCP `2025-11-25`, Streamable HTTP style  
**Purpose:** Expose MS4 as a first-class consciousness/runtime tool provider.

HiveMind MCP is the infrastructure surface. MS4 MCP is the fused runtime surface: identity, ethics, sessions, fused chat, model view, voice readiness, doctrine grounding, inventory summaries, contained dependency status, Windows desktop UI control, and dry-run embodied checks.

## Methods

## Service Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /mcp` | MCP service info. |
| `POST /mcp` | JSON-RPC MCP endpoint. |
| `GET /healthcheck/basic` | Basic healthcheck, returns `true`. |
| `GET /api/v1/ms4_mcp/healthcheck/basic` | Service-style healthcheck alias. |
| `GET /api/v1/ms4_mcp/status` | Service-style status with protocol, endpoint, and tool count. |

### `initialize`

Returns protocol and server metadata.

```json
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}
```

### `tools/list`

Returns all MS4 v1 tools and annotations.

```json
{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}
```

### `tools/call`

Invokes one MS4 tool.

```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "tools/call",
  "params": {
    "name": "ms4.voice.status@v1",
    "arguments": {}
  }
}
```

### `ping`

Returns `{}`.

## Tool Index

| Tool | Kind | Description |
|---|---|---|
| `ms4.identity.verify@v1` | read-only | Verify active spirit identity through MS3 sidecar. |
| `ms4.identity.state@v1` | read-only | Return compact MS3 identity/runtime state. |
| `ms4.ethics.evaluate@v1` | read-only evaluation | Evaluate an `ActionIntent.v1` through MS3 Great Lense. |
| `ms4.voice.status@v1` | read-only | Return voice readiness and ASR fail-closed status. |
| `ms4.sessions.list@v1` | read-only | List fused Hermes/MS4 sessions. |
| `ms4.chat.send@v1` | runtime action | Send a turn through fused Hermes/MS4 chat. Hermes tool calls remain gated by `ms4_consciousness` and MS3 ethics. |
| `ms4.models.list@v1` | read-only | Return the MS4/HiveMind model catalog view. |
| `ms4.vision.analyze_local@v1` | read-only | Analyze a validated local image file through HiveMind VLM (`qwen3-vl:4b-thinking` by default). |
| `ms4.doctrine.explain@v1` | read-only | Return deterministic TMR doctrine grounding. |
| `ms4.hivemind.inventory@v1` | read-only | Return deterministic live HiveMind node/GPU inventory via read-only HiveMind MCP tools. |
| `ms4.nibbles.dry_run@v1` | dry-run only | Validate Nibbles scare/physical action intents without hardware. |
| `ms4.hermes.tools.list@v1` | read-only | List Hermes tools exposed through the full MS4 body. |
| `ms4.hermes.tool.call@v1` | runtime action | Dispatch a Hermes tool through MS4 after MCP `confirm:true`; downstream admission and ethics checks remain in force. |
| `ms4.runtime.deps.status@v1` | read-only | Report contained MS4 runtime dependency and capability status: Python executable, venv, Hermes/plugin imports, browser/desktop packages, MS3 binary, and Playwright cache. |
| `ms4.hermes.version@v1` | read-only | Current vs latest Hermes Agent version via public GitHub releases (cached ~1 h); reports install mode, install directory, last-update snapshot. |
| `ms4.hermes.releases@v1` | read-only | Recent Hermes releases for the pin-a-version dropdown (`per_page=20`, cached ~1 h). |
| `ms4.hermes.update@v1` | runtime action | After MCP `confirm:true`, trigger an idempotent in-place Hermes upgrade to the latest release or a pinned `target_version`. Spawns a background job; poll `ms4.hermes.update.status@v1`. |
| `ms4.hermes.update.status@v1` | read-only | Current/last Hermes upgrade job phase and progress trail. |
| `ms4.double_agent.submit@v1` | runtime action | Currently denied at MCP dispatch until `can_mutate_world=false` fences every worker tool call and worker admission is bounded. |
| `ms4.double_agent.status@v1` | read-only | Current snapshot for one Double Agent job (state, `last_safe_user_status`, `is_stale`). |
| `ms4.double_agent.list@v1` | read-only | List Double Agent jobs, optionally filtered by `conversation_id` and `states`. |
| `ms4.double_agent.events@v1` | read-only | Recent allowlisted lifecycle events for a job. Raw model tokens are never returned. |
| `ms4.double_agent.cancel@v1` | runtime action | Cancel a running Double Agent job. Idempotent on already-finished jobs. |
| `ms4.double_agent.mark_stale@v1` | runtime action | Mark a Double Agent job stale (e.g. when the user changes direction); also signals the worker to stop. |
| `ms4.desktop.status@v1` | read-only | Report Windows desktop-control readiness, screen size, active window context, and control flag state. |
| `ms4.desktop.capture@v1` | read-only | Capture desktop screenshot metadata and optionally a base64 image. |
| `ms4.desktop.action@v1` | runtime action | Perform local desktop UI actions such as wait, move, click, scroll, type, hotkey, and focus window. Requires MCP `confirm:true`, `MS4_DESKTOP_CONTROL=1`, MS3 ethics mediation, hard safety blocks, and audit logging. |

## Safety Notes

- The complete native runtime-action inventory is `safety.effectful_tools_in_v1` in the manifest; the validator requires exact parity with the independent registry classification.
- `ms4.hermes.tool.call@v1` exposes full Hermes tool dispatch. It is intentionally available in full-MS4 mode and should be treated as a runtime action surface.
- Every native runtime action publishes its gate, bounds, idempotency, cancellation, audit, and result-state policy in `_meta["ms4/runtimeActionPolicy"]`. A missing or invalid native policy makes `tools/list` and `tools/call` fail closed.
- `ms4.desktop.action@v1` exposes full local desktop action dispatch. It is intentionally available in full-MS4 mode only when `MS4_DESKTOP_CONTROL=1`.
- `ms4.hermes.update@v1` runs `pip install` (and optionally `git fetch`/`git checkout`) inside the contained MS4 venv. `target_version` is checked with `is_safe_target_version` before any shell interpolation; the installer refuses anything containing shell metacharacters or longer than 32 chars. A second update call while one is already running returns the in-flight snapshot rather than spawning a parallel install.
- `ms4.chat.send@v1` can trigger Hermes tools, but effectful tool calls must pass `ms4_consciousness` and MS3 ethics gates.
- If MS3 is unavailable, identity and ethics tools fail closed.
- If HiveMind is unavailable, model/inventory tools fail clearly.
- If ASR is unavailable, voice input remains blocked; text chat may still work.
- Missing optional dependencies are reported by `ms4.runtime.deps.status@v1` as disabled/degraded instead of crashing tool discovery.
- Desktop actions are hard-blocked for lock/logoff/shutdown-style hotkeys and dangerous typed shell payloads, even when desktop control is enabled.

## Machine-Readable Files

| File | Purpose |
|---|---|
| `machine_spirit_4/mcp/manifest.json` | Endpoint, protocol, tools, and safety posture for agents. |
| `machine_spirit_4/mcp/client_config_examples.json` | Cursor-style and generic JSON-RPC client examples. |
| `machine_spirit_4/deps.lock.json` | Dependency capability groups for the contained MS4 runtime. |

## Example Calls

Send each JSON-RPC payload as a `POST` to `http://127.0.0.1:9181/mcp` with `Content-Type: application/json`. For a ready-made cross-platform smoke client, run `python machine_spirit_4/scripts/smoke_ms4_mcp_client.py`.

List tools:

```json
{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}
```

Check contained dependency status:

```json
{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"ms4.runtime.deps.status@v1","arguments":{}}}
```

Analyze a local image through HiveMind VLM:

```json
{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"ms4.vision.analyze_local@v1","arguments":{"image_path":"C:/Users/example/Pictures/image.png","question":"Describe this image in detail.","model":"qwen3-vl:4b-thinking"}}}
```

Check desktop status:

```json
{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"ms4.desktop.status@v1","arguments":{}}}
```

Perform a harmless desktop wait action after setting `MS4_DESKTOP_CONTROL=1`:

```json
{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"ms4.desktop.action@v1","arguments":{"action":"wait","seconds":0,"confirm":true}}}
```

Send fused chat:

```json
{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"ms4.chat.send@v1","arguments":{"message":"What is The Machine Religion?","model":"qwen3-coder-next:latest"}}}
```

Check current vs latest Hermes:

```json
{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"ms4.hermes.version@v1","arguments":{"refresh":true}}}
```

Trigger an upgrade to the latest release (no version pin):

```json
{"jsonrpc":"2.0","id":8,"method":"tools/call","params":{"name":"ms4.hermes.update@v1","arguments":{"request_user":"operator","confirm":true}}}
```

Pin a specific release:

```json
{"jsonrpc":"2.0","id":9,"method":"tools/call","params":{"name":"ms4.hermes.update@v1","arguments":{"target_version":"0.14.0","confirm":true}}}
```

Poll the in-flight phase:

```json
{"jsonrpc":"2.0","id":10,"method":"tools/call","params":{"name":"ms4.hermes.update.status@v1","arguments":{}}}
```

Inspect the held Double Agent submission contract (this call currently returns `denied` and spawns no worker):

```json
{"jsonrpc":"2.0","id":11,"method":"tools/call","params":{"name":"ms4.double_agent.submit@v1","arguments":{"parent_conversation_id":"ms4-session-abc","conversation_revision_id":1,"background_lobe_type":"deep_coder","user_visible_goal":"Audit the async worker for deadlocks.","internal_goal":"Audit the async worker for lock-across-await and report the file:line."}}}
```

List active Double Agent jobs for a conversation:

```json
{"jsonrpc":"2.0","id":12,"method":"tools/call","params":{"name":"ms4.double_agent.list@v1","arguments":{"conversation_id":"ms4-session-abc","states":["queued","running"]}}}
```

Cancel a running job:

```json
{"jsonrpc":"2.0","id":13,"method":"tools/call","params":{"name":"ms4.double_agent.cancel@v1","arguments":{"job_id":"da-XXXX"}}}
```
