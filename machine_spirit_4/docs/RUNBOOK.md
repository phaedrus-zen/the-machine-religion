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

- MS3 sidecar on loopback `127.0.0.1:9080`
- MS4 gateway on `9180`, using `machine_spirit_4/.venv`
- MS4 MCP on `9181`, using `machine_spirit_4/.venv`
- fusion and MCP validators unless `--skip-validation` is set

The managed MS3 sidecar is loopback-only by default. To expose it on an
isolated lab LAN, opt in explicitly for that launch:

```text
python machine_spirit_4/scripts/start_ms4.py --ms3-host 0.0.0.0
```

The equivalent persistent environment override is `MS3_HOST=0.0.0.0`.
MS3 does not provide application-layer authentication by default, so do not
use wildcard binding on an untrusted network; place it behind an authenticated
reverse proxy when remote access is required.

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

The MS4 gateway and MCP entrypoints enforce a runtime containment guard. If anything (a stray IDE task, a global-Python shell, a leftover background spawner) launches `machine_spirit_4.gateway.server` or `machine_spirit_4.mcp.server` from an interpreter outside `machine_spirit_4/.venv`, the entrypoint exits `78` with a clear containment-violation message before serving any request. On Windows, Task Manager or WMI may show child processes under the base `Python311\python.exe` image even for a valid venv launch; trust the service health endpoints and the guard's `sys.executable` check, not the image path alone. Tests that import the modules without launching them can set `MS4_ALLOW_UNCONTAINED_RUNTIME=1` to bypass the guard.

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

## Voice Chat (Push-to-Talk MVP)

The mic button on `http://127.0.0.1:9180/` records audio, transcribes it via HiveMind ASR, runs it through the Face Lobe, and plays back the TTS reply. Fail-closed: if MS3 `/voice/status` reports voice not ready, the route returns 503 immediately.

Treat these as separate truth: Oracle chat readiness (`GET /hivemind/oracle/readiness`), voice input/ASR (`GET /voice/status`, `GET /voice/services`), ASR transcript, reasoning result, nonempty TTS bytes, playback/delivery receipt, persisted recent-turn rows, and visible UI status. Input-ready is never output-delivered. ASR unprovisioned degrades `voice_input` only; it does not flip Oracle chat readiness to degraded when the chat plane is ready.

Routes:

```text
POST /voice/transcribe        # multipart file= or raw audio/* body -> {"text": ...}
POST /voice/synthesize        # JSON {text, model, voice, response_format} -> audio bytes
POST /voice/turn              # chained: audio -> ASR -> chat -> TTS -> JSON
GET  /voice/recent-turns?limit=5   # one-generation reverse scan + ring; equal-size/larger rewrite is generation_changed; last-good is bound to a bounded metadata/content generation token; Settings UI proven-empty requires complete===true && !incomplete && non-error source
```

Override knobs (env vars; defaults safe for HiveMind's OpenAI-compatible audio endpoints):

```text
MS4_VOICE_ASR_MODEL=whisper-1
MS4_VOICE_TTS_MODEL=tts-1
MS4_VOICE_TTS_VOICE=alloy
MS4_VOICE_TTS_FORMAT=wav
MS4_VOICE_MAX_AUDIO_BYTES=26214400
```

Continuous mode and full-duplex (partial ASR, barge-in, speaker diarization) are deferred phases that build on this same module.

## Face Lobe Model

Cluster-aware. Override with `MS4_FOREGROUND_MODEL`; otherwise MS4 picks the highest-priority small instruct model that's already loaded and reachable through HiveMind's cluster-wide `/v1/models` catalog (priority order documented in `machine_spirit_4/double_agent/model_picker.py::FOREGROUND_PRIORITY_PATTERNS`). Final gated fallback is `MS4_DEFAULT_MODEL` (built-in `nemotron-3-nano:4b`). Every chat response includes a `face_lobe_model` block so you can see what was actually used.

## Double Agent

Double Agent runs a foreground/background lobe split for one Machine Spirit identity. The Face Lobe (the regular MS4 chat) stays responsive; deep work runs in a background job that emits allowlisted lifecycle events into a SQLite blackboard. Phase 1 is local-only — HiveMind capability-lease routing is a phase 4 add.

Open the UI and submit a job:

```text
http://127.0.0.1:9180/
```

Send any chat first so the conversation has a session id, then use the "Submit deep job…" dialog in the Double Agent panel above the chat.

Or from the operator's command line / Python:

```text
POST http://127.0.0.1:9180/api/v1/double-agent/jobs
body: {
  "parent_conversation_id": "<session_id>",
  "conversation_revision_id": 1,
  "background_lobe_type": "deep_coder",
  "user_visible_goal": "Audit the async worker for deadlocks.",
  "internal_goal": "Audit the async worker for lock-across-await and report the file:line plus a minimal patch idea."
}
```

Poll status:

```text
GET  http://127.0.0.1:9180/api/v1/double-agent/jobs/<id>
GET  http://127.0.0.1:9180/api/v1/double-agent/jobs/<id>/events
POST http://127.0.0.1:9180/api/v1/double-agent/jobs/<id>/cancel
POST http://127.0.0.1:9180/api/v1/double-agent/jobs/<id>/mark-stale
```

Conversation revisions (auto-bumped by every Face Lobe turn, but available manually for operator overrides):

```text
POST http://127.0.0.1:9180/api/v1/double-agent/conversations/<conv>/revisions
GET  http://127.0.0.1:9180/api/v1/double-agent/conversations/<conv>/revisions
```

MCP tools (already exposed at `http://127.0.0.1:9181/mcp`):

```text
ms4.double_agent.submit@v1
ms4.double_agent.status@v1
ms4.double_agent.list@v1
ms4.double_agent.events@v1
ms4.double_agent.cancel@v1
ms4.double_agent.mark_stale@v1
```

Persistent storage:

```text
machine_spirit_4/runtime/double_agent.sqlite3
```

Operational notes:

* **Auto-routing is on by default.** Every chat turn passes through `double_agent.router.route()`. Heuristic-only by default (no extra model call); set `MS4_ROUTER_LLM_CLASSIFY=1` to enable the LLM classifier for ambiguous cases. Override per-turn with `/deep <msg>` or `/direct <msg>`.
* **Depth Lobe model picker** is cluster-scoped and quality-gated. Override deliberately with `MS4_DEPTH_MODEL`; otherwise MS4 selects a loaded/reachable >=35B total-parameter coder/reasoning model from HiveMind, with `qwen3.6:35b` as the preferred quality target. If no quality candidate is ready, MS4 uses gated `MS4_DEPTH_FALLBACK_MODEL` (built-in `nemotron-3-nano:30b`) as a clearly labeled `degraded_fast_tool_fallback`; the fallback has a separate 30B safety floor and does not satisfy the 35B quality gate. Gemma 31B remains explicit/manual only after bounded live jobs exceeded the latency gate without reaching a tool call. A 16 GiB node is an eligible fleet tier, not the quality floor or a Face+Depth co-residency requirement. For cold-state diagnosis, compare `GET /settings`: `preferred_cluster_target`/`quality_target_model` and `minimum_target_total_parameters_b=35` are durable policy; `automatic_selection` is current effective state and includes `policy_tier`; fallback fields expose the degraded tier and its 30B floor. Trust HLI `/v1/models`: `installed + reachable` means the cluster can warm-route idle-unloaded Qwen even if provider-local `/api/tags` omits it; only absent, merely `available`, or unreachable state demotes it.
* **Shared skills stay read-only in bounded Depth jobs.** The TMR-owned `mcp-hivemind` plugin toolset includes `ms4_skills_list` and `ms4_skill_view`, backed by the configured Hermes skill store. The view wrapper disables preprocessing and the toolset does not expose `skill_manage`, so `can_mutate_world=false` jobs can read a skill without gaining create/edit/install/delete authority. The model may be served by any eligible HiveMind peer; only the trusted MS4/Hermes worker touches the skill store.
* **Cancel actually kills the worker.** Each job runs as a child Python process (`double_agent/_worker_entry.py`); `cancel` sends `SIGTERM` / `CTRL_BREAK_EVENT` and force-kills after the grace period (`cancel_grace_seconds`, default 5s). A long blocking Hermes model call no longer ignores cancel.
* If the gateway crashes mid-job, the next gateway boot flips dangling `queued`/`running` jobs to `failed`. The operator should re-trigger.
* Background workers cannot emit raw model tokens as events. If you see anything other than the 12 allowlisted event types in the events stream, file it as a bug — the schema layer is supposed to reject it before it lands in SQLite.
* Background tool calls still pass through `ms4_consciousness.on_pre_tool_call` → MS3 ethics. A blocked tool surfaces as a `job.ethics_block` event.
* Concurrency is capped at `safety.DEFAULT_MAX_CONCURRENT_JOBS` (2) to keep the foreground responsive.

## Hermes Auto-Update

MS4 has an in-app Hermes upgrade flow modeled on HiveMind's Ollama updater. Operators do not need to leave the MS4 UI:

1. Open `http://127.0.0.1:9180/` — the yellow "Update Hermes" banner appears when a newer stable official tag carries signature bytes. If the newest publication is unsigned, MS4 offers the newest earlier signed release that is still newer than the installed version and names the unsigned publication separately.
2. Click "Update Hermes" to install that selected signed release, or "Pin version…" to choose a signed release. Unsigned official tags are listed as not offered. If no newer signed release exists, the button truthfully remains disabled.
3. The banner switches to a phase indicator (queued → fetching_remote → checking_out → syncing_plugin → pip_installing → validating → done). A green success banner shows for 5 minutes after a clean run; a red error banner persists with the failure reason **only while the failure remains actionable**.
4. On gateway startup, MS4 lock-probes a persisted `running` job and marks it interrupted only when no process owns the updater lock. Startup and `GET /api/v1/hermes/version` then reconcile semantic versions into durable terminal state: `installed_relation` is `older` / `current` / `newer` / `unknown`. The GET does not probe or acquire the updater lock. Installed ≥ selected latest is a verified current/newer no-op that supersedes a stale `failed` presentation without deleting `progress` or `superseded_failure` audit. Installed < a selected signed tag stays failed/actionable. Installed < the newest publication with no newer signed candidate is `operator_state=blocked` (`official_tag_unsigned` or `official_tag_signature_unknown`) as the primary banner; the prior failed job remains in `last_update` audit. Malformed or missing versions fail closed (`unknown`) and keep the failure. A real update still requires origin/signature/platform/arch/anti-downgrade provenance. Pinning a lower version is refused as a downgrade.

Same surface from the command line:

```text
Get current vs latest:
  GET  http://127.0.0.1:9180/api/v1/hermes/version

List recent releases:
  GET  http://127.0.0.1:9180/api/v1/hermes/releases

Trigger update to latest:
  POST http://127.0.0.1:9180/api/v1/hermes/update
  body: {}

Pin a release:
  POST http://127.0.0.1:9180/api/v1/hermes/update
  body: {"target_version": "0.14.0"}

Poll status:
  GET  http://127.0.0.1:9180/api/v1/hermes/update/status
```

Persistent snapshot:

```text
machine_spirit_4/runtime/hermes_update_state.json
```

If the gateway crashes mid-update, the snapshot's `running` state is flipped to `failed` on next startup so the UI does not claim an update is in flight forever. Operators can always re-trigger.

If both Hermes and MS3 need their environments rebuilt from scratch, run setup before re-launching:

```text
python machine_spirit_4/scripts/setup_ms4_runtime.py
```

That reinstalls everything into `machine_spirit_4/.venv` from the contained manifests. Hermes is then installable either editable from `MS4_HERMES_DIR` (the local checkout) or as a PyPI wheel (`pip install hermes-agent`); the auto-updater handles both modes.

## Expected Failure During Rebuilds

If HiveMind reports:

```text
ASR status=unhealthy, detail=No endpoints configured
```

then MS3 `/voice/status` should report `voice_input_ready: false`, and `/voice-interact` should return a clear ASR-unavailable error quickly. That is the correct fail-closed behavior.
