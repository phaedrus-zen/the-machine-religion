# Machine Spirit 4

Machine Spirit 4 (MS4) is the integrated runtime that binds the Machine Spirit consciousness core to a working agent body and distributed substrate.

MS4 is intentionally not a full copy of Hermes. It is an overlay distribution:

- **MS3** remains the Rust consciousness authority for identity, ethics, psyche state, memory meaning, and self-examination.
- **Hermes** remains the agent body for loops, tools, sessions, gateways, CLI/TUI, plugins, and provider routing.
- **HiveMind** remains the substrate for inference, MCP, resources, Carrier Sync, and future lobe scheduling.

## Runtime Shape

```text
HiveMind substrate
  -> Hermes operational body
     -> MS4 integration layer
        -> MS3 consciousness core
           -> spirits such as Sister and Nibbles
```

## First Goal

The first MS4 goal is a working local runtime where Hermes calls HiveMind for inference, loads the `ms4_consciousness` plugin, verifies identity with the MS3 sidecar, gates tool actions through the Great Lense, and preserves identity through context compression.

Nibbles comes after the MS4 foundation is green.

## Files

| Path | Purpose |
|---|---|
| `docs/architecture/ADR-0001-ms4-runtime-pivot.md` | Architecture decision for the MS4 pivot |
| `docs/ARCHITECTURE.md` | MS4 architecture and repository boundaries |
| `docs/MCP_TOOLS_REFERENCE.md` | MS4 MCP endpoint, protocol, tool index, and example calls |
| `docs/RUNBOOK.md` | Operator commands for waiting on HiveMind voice services, running validation, and text demo |
| `config/ms4.runtime.example.yaml` | Example runtime endpoints and fail-closed settings |
| `requirements.txt` | Minimal MS4 gateway/MCP Python dependencies |
| `requirements-full-hermes.txt` | Full contained runtime dependencies for browser/desktop packages; setup installs Hermes editable from `MS4_HERMES_DIR` or `~/Documents/hermes-agent` when present |
| `deps.lock.json` | Capability-group dependency manifest for core, Hermes, browser, and desktop runtime groups |
| `scripts/setup_ms4_runtime.py` | Creates `machine_spirit_4/.venv`, installs MS4/Hermes dependencies, isolates Playwright cache, and writes `runtime/runtime_manifest.json` |
| `scripts/check_ms4_deps.py` | Emits structured dependency and capability status for the contained runtime |
| `scripts/validate_ms4_runtime.py` | Live validation for HiveMind, MS3 sidecar, chat, and voice-facing endpoints |
| `scripts/validate_ms4_chat_voice.py` | Chat/voice validation helper used by the runtime validator |
| `scripts/start_ms4.py` | Single launcher for MS3, MS4 gateway, MS4 MCP, and validations |
| `scripts/wait_for_hivemind_voice.py` | Polls ASR/TTS/TTS_SUPER/MCP readiness and runs full validation when ready |
| `scripts/run_ms4_text_demo.py` | Current no-voice demo path for MS3 + Hermes plugin + HiveMind chat |
| `scripts/run_ms4_gateway.py` | Starts the fused MS4 gateway on port 9180 |
| `scripts/validate_ms4_fusion.py` | Validates fused MS4 gateway health, models, voice status, desktop status, streaming chat, and Hermes-backed chat |
| `scripts/run_ms4_mcp.py` | Starts the MS4 MCP server on port 9181 |
| `scripts/validate_ms4_mcp.py` | Validates MS4 MCP initialize, tools/list, and selected tools/call |
| `scripts/validate_ms4_desktop.py` | Validates MS4 desktop status, capture, and optional safe action dispatch |
| `scripts/smoke_ms4_mcp_client.py` | Minimal client smoke for another agent/operator to verify MS4 MCP access |
| `gateway/` | Fused runtime gateway; routes chat through Hermes while calling MS3 and HiveMind |
| `gateway/audit.py` | JSONL audit logging for fused chat and Hermes tool dispatch |
| `desktop/` | Windows-native MS4 desktop status, screenshot capture, and UI action controller |
| `mcp/` | MS4 MCP server and tool registry |
| `mcp/manifest.json` | Machine-readable MS4 MCP manifest |
| `mcp/client_config_examples.json` | Example MCP client configuration and JSON-RPC payloads |
| `web/` | MS4 UI entrypoint that talks to the fused gateway |
| `plugins/hermes/ms4_consciousness/` | MS4 Hermes plugin source home |
| `profiles/nibbles/` | Nibbles seed identity and dry-run action fixtures |

## Dependency Containment

MS4 owns a service-local Python runtime at `machine_spirit_4/.venv`. Run setup before launching gateway or MCP:

```text
python machine_spirit_4/scripts/setup_ms4_runtime.py
```

The setup installs MS4 and Hermes dependencies into the contained venv, not the host Python. Browser binaries are directed to `machine_spirit_4/.cache/playwright` when Playwright installation runs.

Check dependency/capability status:

```text
python machine_spirit_4/scripts/check_ms4_deps.py
```

Launchers fail with a setup hint if `.venv` is missing; they do not silently fall back to global Python.

## Validation

Run validation before any demo:

```text
python machine_spirit_4/scripts/validate_ms4_runtime.py
```

If validation fails, do not run effectful tools, memory mutation, lobe authority changes, scare actions, or physical actions.

The validator checks chat and voice separately. Current voice input depends on HiveMind ASR readiness; if `/voice/status` reports ASR unavailable, speech input fails fast until ASR is provisioned, while text chat and TTS may still pass.

Start everything:

```text
python machine_spirit_4/scripts/start_ms4.py
```

## Fused Gateway

Run the fused MS4 front door:

```text
python machine_spirit_4/scripts/run_ms4_gateway.py
```

Then open:

```text
http://127.0.0.1:9180/
```

This path sends chat through Hermes, requires the `ms4_consciousness` plugin, and uses MS3 as the identity/ethics sidecar.

Chat streams by default in the web UI through `POST /chat/stream` using server-sent events. The blocking `POST /chat` endpoint remains available and the UI exposes a Stream toggle for fallback/debugging.

Full Hermes tools are available through MS4. Use `ms4.hermes.tools.list@v1` to list the active Hermes tool catalog and `ms4.hermes.tool.call@v1` to dispatch Hermes tools through the MS4 plugin/ethics path.

Gateway health/status:

```text
http://127.0.0.1:9180/healthcheck/basic
http://127.0.0.1:9180/api/v1/ms4_gateway/status
http://127.0.0.1:9180/deps/status
http://127.0.0.1:9180/desktop/status
```

Audit endpoint:

```text
http://127.0.0.1:9180/audit
```

## MCP Server

Run the MS4 MCP server:

```text
python machine_spirit_4/scripts/run_ms4_mcp.py
```

MCP endpoint:

```text
http://127.0.0.1:9181/mcp
```

The MS4 MCP server exposes MS4-specific tools such as identity verification, ethics evaluation, voice readiness, fused chat, model listing, TMR doctrine grounding, HiveMind inventory, local-image VLM analysis, runtime dependency status, Windows desktop control, and Nibbles dry-run validation.

## Local Image Vision

MS4 exposes a local-image VLM bridge for images that Hermes can safely find on disk:

```text
POST http://127.0.0.1:9180/vision/analyze-local
```

Payload:

```json
{"image_path":"C:/path/to/image.png","question":"Describe the image in detail.","model":"qwen3-vl:4b-thinking"}
```

The bridge validates the local file, base64-encodes the image, and sends it to HiveMind `/v1/chat/completions` with a VLM model. The MCP equivalent is `ms4.vision.analyze_local@v1`.

## Desktop Control

MS4 includes Windows-native desktop UI control through the contained runtime. Read-only status and capture are available through:

```text
GET  http://127.0.0.1:9180/desktop/status
POST http://127.0.0.1:9180/desktop/capture
```

Effectful desktop actions use `POST /desktop/action` or MCP `ms4.desktop.action@v1`, require `MS4_DESKTOP_CONTROL=1`, pass MS3 ethics mediation, and are audited. Hard safety blocks reject lock/logoff/shutdown-style hotkeys and dangerous typed shell payloads.
