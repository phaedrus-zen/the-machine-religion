# MS4 Operator Runbook

This runbook covers the local MS4 runtime while HiveMind voice services are being rebuilt or restarted.

## Current Safety Rule

MS4 can run text chat when HiveMind chat and the MS3 sidecar are healthy.

Voice input requires HiveMind ASR readiness. If `GET /voice/status` reports ASR unavailable, speech input must fail fast. Do not bypass that check.

Physical, scare, lobe authority, memory mutation, and identity mutation remain blocked unless validation passes and the relevant safety/consent path is green.

## Start MS4

First create the contained runtime:

```text
python machine_spirit_4/scripts/setup_ms4_runtime.py
```

That creates `machine_spirit_4/.venv`, installs MS4 plus Hermes dependencies into it, writes `machine_spirit_4/runtime/runtime_manifest.json`, and directs Playwright browser installs to `machine_spirit_4/.cache/playwright` when browser dependencies are installed.

Use the single launcher:

```text
python machine_spirit_4/scripts/start_ms4.py
```

It starts or verifies:

- MS3 sidecar on `9080`
- MS4 gateway on `9180`, using `machine_spirit_4/.venv`
- MS4 MCP on `9181`, using `machine_spirit_4/.venv`
- fusion and MCP validators unless `--skip-validation` is set

Audit log:

```text
machine_spirit_4/logs/ms4_audit.jsonl
```

Audit endpoint:

```text
http://127.0.0.1:9180/audit
```

Dependency/capability status:

```text
python machine_spirit_4/scripts/check_ms4_deps.py
```

```text
http://127.0.0.1:9180/deps/status
```

Desktop control is disabled for effectful actions unless explicitly enabled:

Set `MS4_DESKTOP_CONTROL=1` in the environment of the process that starts MS4.

Read-only desktop status and capture do not require the flag. Effectful actions through `/desktop/action` or `ms4.desktop.action@v1` still pass MS3 ethics and hard safety blocks.

## Wait For Voice Stack

Use this while HiveMind is rebuilding ASR/TTS:

```text
python machine_spirit_4/scripts/wait_for_hivemind_voice.py
```

The wait runner polls:

- `GET /v1/health`
- `GET /v1/mcp/status`
- `GET /provision/status/ASR`
- `GET /provision/status/TTS`
- `GET /provision/status/TTS_SUPER`

When all are healthy, it runs `validate_ms4_runtime.py`.

## Full Validation

Run:

```text
python machine_spirit_4/scripts/validate_ms4_runtime.py
```

This checks HiveMind health, models, MCP status, resources, App Registry, Carrier Sync, MS3 health, identity, ethics, chat, TTS, ASR silence-gate, MS3 `/interact`, MS3 `/voice/status`, and `/voice-interact` fast-fail behavior.

## Text Demo

The current no-voice demo path is:

```text
python machine_spirit_4/scripts/run_ms4_text_demo.py
```

It starts MS3 if needed, loads Hermes `ms4_consciousness`, verifies identity, checks an ethics gate for a read-only tool, and calls MS3 `/interact`.

## Fused MS4 Gateway

Start the fused front door:

```text
python machine_spirit_4/scripts/run_ms4_gateway.py
```

Open:

```text
http://127.0.0.1:9180/
```

Validate:

```text
python machine_spirit_4/scripts/validate_ms4_fusion.py
```

The fused gateway sends chat through Hermes. It is the path that exercises Hermes sessions, tool loop, provider routing, plugin hooks, and MS3 sidecar authority together.

The web UI streams chat by default through `POST /chat/stream`, with periodic heartbeat events while Hermes is still working. Disable the Stream checkbox to use the blocking `POST /chat` fallback.

Local image VLM analysis is available through:

```text
POST http://127.0.0.1:9180/vision/analyze-local
```

Use payload fields `image_path`, optional `question`, and optional `model`. The default VLM model is `qwen3-vl:4b-thinking`.

Full Hermes posture:

- Hermes tools are exposed through MS4 MCP via `ms4.hermes.tools.list@v1` and `ms4.hermes.tool.call@v1`.
- Effectful Hermes actions are not hidden in full-MS4 mode.
- `ms4_consciousness` and MS3 ethics remain native runtime mediation.
- Validation uses harmless commands such as `echo MS4_HERMES_OK`.

Health/status:

```text
http://127.0.0.1:9180/healthcheck/basic
http://127.0.0.1:9180/api/v1/ms4_gateway/status
http://127.0.0.1:9180/desktop/status
```

Desktop validation:

```text
python machine_spirit_4/scripts/validate_ms4_desktop.py
```

With `MS4_DESKTOP_CONTROL=1`, the validator also performs harmless `wait` actions through Gateway and MCP. It does not type into apps unless an operator runs a separate directed test.

## MS4 MCP Server

Start:

```text
python machine_spirit_4/scripts/run_ms4_mcp.py
```

Validate:

```text
python machine_spirit_4/scripts/validate_ms4_mcp.py
```

Minimal client smoke:

```text
python machine_spirit_4/scripts/smoke_ms4_mcp_client.py
```

Machine-readable client files:

```text
machine_spirit_4/mcp/manifest.json
machine_spirit_4/mcp/client_config_examples.json
```

Endpoint:

```text
http://127.0.0.1:9181/mcp
```

Health/status:

```text
http://127.0.0.1:9181/healthcheck/basic
http://127.0.0.1:9181/api/v1/ms4_mcp/status
```

Use this when another agent needs to discover MS4-specific tools instead of talking directly to the MS3 diagnostic UI or HiveMind infrastructure MCP.

## Nibbles Dry Run

Nibbles seed files live at:

```text
machine_spirit_4/profiles/nibbles/
```

The dry-run action examples are schema-valid `ActionIntent.v1` documents. They deliberately set `dry_run_only: true`, `physical_output_enabled: false`, and `requires_safety_clearance: true`.

Do not connect these examples to hardware. They are contract fixtures for consent, safety, and ethics validation.

## Expected Failure During Rebuilds

If HiveMind reports:

```text
ASR status=unhealthy, detail=No endpoints configured
```

then MS3 `/voice/status` should report `voice_input_ready: false`, and `/voice-interact` should return a clear ASR-unavailable error quickly. That is the correct fail-closed behavior.
