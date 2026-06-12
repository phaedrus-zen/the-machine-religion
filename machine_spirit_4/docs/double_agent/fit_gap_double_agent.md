# Double Agent — Fit / Gap Analysis (Phase 0)

**Status:** Authoritative for this MVP build (phases 0+1+2).
**Reference artifact:** `machine_spirit_4/docs/double_agent/research_artifact_v2.md`.
**Scope:** Local-only MVP. HiveMind distributed scheduling deferred to phase 4.

The research artifact is advisory. This document records the codebase evidence and pins which Double Agent primitives map onto existing TMR / Hermes / HiveMind code and which are real gaps that must be filled.

## 1. Architectural Decision Anchor

```
MS4 owns the Face Lobe experience, Double Agent job lifecycle, conversation
revisioning, anti-hallucination status contract, and result reconciliation.

HiveMind continues to own compute substrate (models, resources, MCP tool
catalog, provisioning). Phase 1 calls HiveMind through the same /v1/* routes
already used by the chat path. No new HiveMind primitives are required for
the MVP; the artifact's "capability lease" primitive maps onto the existing
HiveMind resources/provisioning API for phase 4.

Hermes owns the worker loop. `AIAgent.run_conversation` already exposes the
lifecycle callbacks Double Agent needs (`tool_start_callback`,
`tool_complete_callback`, `stream_delta_callback`, session/task id). MS4's
existing `Ms4HermesRunner` already wires three of those.
```

## 2. Primitive Mapping — What Exists vs Gap

| Double Agent primitive (artifact §) | Existing TMR/HiveMind/Hermes primitive | Decision |
|---|---|---|
| Job envelope (§9) | `machine_spirit_4.hermes_admin.state.UpdateJobSnapshot` — single-job, single-flight, persistent JSON, phase transitions, on-disk recovery, lock-protected | **Reuse the shape**, generalize to N concurrent jobs and SQLite-backed storage. New module `machine_spirit_4.double_agent`. |
| Persistent durable state (§21 phase 1) | `runtime/hermes_update_state.json` (single record), `gateway/audit.py` (append-only JSONL) | **Gap**: need queryable per-job + per-event store. **Add** SQLite at `machine_spirit_4/runtime/double_agent.sqlite3`. Keep JSONL audit as secondary write for parity with existing audit consumers. |
| Conversation revision id (§12) | None. `core/src/types.rs` has `SessionId` but no revision concept. Gateway `_sessions` map is per-session only. | **Gap**: net new. Introduce a `conversation_revisions` table keyed by `conversation_id`, monotonically incremented on every Face Lobe user message. |
| Foreground (Face) Lobe (§5.1) | `gateway/hermes_runner.Ms4HermesRunner.chat()` — already the user-facing turn driver; already uses `ms4_consciousness` plugin for identity/ethics/grounding | **Reuse**. Extend the chat path to (a) bump revision id, (b) classify intent, (c) submit Double Agent jobs when deep work is required, (d) read job state on follow-up turns. |
| Background (Depth) Lobe (§5.2) | `Ms4HermesRunner.chat()` itself, plus `run_agent.AIAgent.run_conversation()` | **Reuse**. The "background worker" is the same `run_conversation` loop, executed on a worker thread with a dedicated session id and a status-event sink instead of the streaming HTTP response. |
| MS4 Governor (§5.3) | `plugins/hermes/ms4_consciousness/__init__.py` — already does identity gating, ethics evaluation, fail-closed authority | **Reuse**. Add Face Lobe context block (revision id + active jobs) into `on_pre_llm_call`. |
| Capability lease (§5.4) | HiveMind `/v1/resources/request`, `/provision/status/*`, `/v1/chat/completions`, model catalog. MS4 `mcp_call`. | **Reuse**. Phase 1 leases the model the same way today's `Ms4HermesRunner` does (via HiveMind `/v1/chat/completions`). Phase 4 may route by `resource_request.model_class` through the existing resources endpoint. |
| Blackboard / evidence ledger (§5.5) | `gateway/audit.py` (JSONL append-only) | **Augment**: add SQLite store with `jobs`, `events`, `results`, `conversation_revisions` tables. Mirror critical events to the existing audit log so a single audit consumer still sees everything. |
| Event stream (§10.3) | Gateway already has SSE (`_sse_start`/`_sse_event` in `gateway/server.py` for `/chat/stream`) | **Reuse the SSE helpers** for a new `/api/v1/double-agent/jobs/{id}/events` route. Polling endpoint also provided for non-SSE clients. |
| Status anti-hallucination contract (§13) | None. `Ms4HermesRunner.chat()` returns whatever the model says. | **Gap**: net new. Implement as: (a) workers emit only `safe_user_status` strings derived from a controlled vocabulary; (b) Face Lobe context block enumerates active jobs and forbids invented status; (c) MCP tools refuse to read raw reasoning tokens. |
| Multi-job / fanout (§14) | None (`hermes_admin` is single-flight). | **Partial**: SQLite store supports N jobs natively. Phase 1 supports submit/track/cancel multiple jobs per conversation. Reducer/reconciler job deferred to phase 5. |
| Cancellation (§22.4) | `subprocess` timeout only (`hermes_admin.installer`). No worker-level cancel signal. | **Gap**: net new. Use `threading.Event` per running job. Worker polls the event between tool calls and between Hermes iterations. |
| Stale-on-revision-change (§12.1) | None. | **Gap**: net new. `blackboard.mark_stale_jobs_for_revision(conversation_id, current_revision)` flips any job whose `conversation_revision_id < current_revision` to `stale`, and the Face Lobe reports accordingly. |
| Authority envelope (§27) | `ms4_consciousness._risk_class_for_tool` + `Ms4Client.evaluate_action` | **Reuse** for any tool call the background worker tries to make. Job envelope stores the operator-supplied authority profile, but per-tool evaluation continues to go through MS3 ethics. |
| Conversation history per session | `Ms4HermesRunner.SessionState.history` | **Reuse**: the background worker uses a forked snapshot of the session history at submission time, not the live session, so the foreground can keep mutating it. |
| MCP tool catalog (§22) | `mcp/tools.py` — `ToolDef` + `build_tool_registry()` | **Reuse**. Add 6 new tools under `ms4.double_agent.*`. |
| Resource isolation (§17.2) | None in MS4. HiveMind has per-resource provisioning. | **Phase 4** concern. For phase 1, cap concurrent background jobs (`max_concurrent_jobs`) and run workers on Python threads; foreground stays on the gateway request thread. |

## 3. What I Will NOT Build in Phase 1

* HiveMind distributed scheduling (§21 phase 4) — defer.
* Fanout reconciler (§14, §21 phase 5) — defer; submit/list/cancel only.
* Token-level event stream from the model — the existing Hermes `stream_delta_callback` is wrapped instead into `step_started/step_completed` summary events, per the artifact's "stream safe operational events, not hidden reasoning" rule (§10.3).
* Cloud fallback (§9.1 `fallback_allowed`) — phase 1 honors `fallback_allowed=false` strictly; HiveMind-only.
* Persistent multi-process queue — phase 1 uses an in-process job runner; if the gateway restarts, all in-flight jobs are flipped to `failed` on hydration (same recovery semantics as `hermes_admin.state`).

## 4. New Module Layout

```
machine_spirit_4/double_agent/
├── __init__.py            # public API surface
├── schemas.py             # JobEnvelope, JobEvent, JobResult, ConversationRevision dataclasses + validators
├── safety.py              # is_safe_job_id, is_safe_conversation_id, allowlists for event types and phases
├── blackboard.py          # SQLite-backed store (jobs, events, results, conversation_revisions)
├── runner.py              # JobRunner: submit, list, get, cancel, mark_stale; single-flight per job_id
└── worker.py              # HermesDoubleAgentWorker: wraps Ms4HermesRunner.chat for background execution; emits safe events; honors cancellation
```

```
machine_spirit_4/docs/double_agent/
├── research_artifact_v2.md            # canonical input (stash; do not edit)
├── fit_gap_double_agent.md            # this doc
├── existing_primitive_mapping.md      # short reference table
└── (later) test_plan_double_agent.md
```

```
machine_spirit_4/runtime/
├── hermes_update_state.json           # existing
└── double_agent.sqlite3               # new; created on first job submission
```

## 5. New Schemas (Phase 1 minimum)

```
DoubleAgentJobEnvelope.v1
  job_id                          uuid; safe id guard
  parent_conversation_id          string; safe id guard
  conversation_revision_id        integer; >= 1
  owner_identity                  "MachineSpirit4" only in phase 1
  foreground_lobe                 "ms4_face"
  background_lobe_type            enum: deep_chat | deep_coder | diagnostic | research | verifier |
                                      oracle_deep_chat | oracle_deep_coder |
                                      oracle_diagnostic | oracle_planner |
                                      oracle_research | oracle_verifier
  user_visible_goal               string; <= 280 chars
  internal_goal                   string; <= 1024 chars
  priority                        enum: interactive | batch
  latency_class                   enum: foreground | interactive_background | batch_background
  authority                       AuthorityEnvelope.v1 (see below)
  resource_request                ResourceRequest.v1 (see below)
  status_policy                   StatusPolicy.v1
  state                           enum: queued | running | completed | failed | canceled | stale
  created_at / updated_at         iso8601
  finished_at                     iso8601 | null

AuthorityEnvelope.v1
  can_read_state: bool
  can_call_tools: bool
  can_mutate_world: bool         # phase 1: must be false
  allowed_toolsets: [string]
  requires_approval_for: [string]

ResourceRequest.v1
  model_class                    free string; default "deep_reasoning"
  preferred_runtime              "local" | "hivemind"
  preferred_hardware             "best_available" | "cpu" | "gpu"
  fallback_allowed               bool; phase 1 default false
  model_override                 optional HiveMind model id

StatusPolicy.v1
  emit_progress_events: bool     # always true in phase 1
  emit_raw_tokens: bool          # always false in phase 1 (anti-hallucination rule)
  emit_tool_events: bool         # default true
  summarize_every_seconds: int   # default 10

DoubleAgentJobEvent.v1
  event_id                       uuid
  job_id                         uuid
  type                           one of EVENT_TYPES (allowlist; see safety.py)
  timestamp                      iso8601
  safe_user_status               string; <= 280 chars; required for user-visible events
  evidence_refs                  [string]                           # optional
  visibility                     "user_safe" | "operator_only"

DoubleAgentJobResult.v1
  job_id                         uuid
  status                         success | partial | failed | stale | needs_user
  summary                        string
  text                           string                              # the Hermes final_response
  evidence                       [object]
  actions_taken                  [object]
  next_steps                     [string]
  confidence                     low | medium | high
  finished_at                    iso8601
  conversation_revision_id       integer (the revision the result was generated for)

ConversationRevision.v1
  conversation_id                string
  revision_id                    integer
  user_message_excerpt           string                              # <= 200 chars; for diagnosis
  created_at                     iso8601
```

Allowlisted event types (mirrors §10.3 with one MS4 addition for ethics blocks):

```
job.queued
job.started
job.tool.call.started
job.tool.call.completed
job.checkpoint
job.partial_finding
job.needs_input
job.ethics_block             # MS4 addition: MS3 refused a tool call inside the worker
job.completed
job.failed
job.canceled
job.stale
```

## 6. New HTTP Routes (Gateway)

| Route | Method | Purpose |
|---|---|---|
| `/api/v1/double-agent/jobs` | POST | Submit a job envelope. Validates schema + safe-id guard + authority. Returns job snapshot. |
| `/api/v1/double-agent/jobs` | GET | List active and recently completed jobs. Optional `?conversation_id=` filter. |
| `/api/v1/double-agent/jobs/{id}` | GET | Current job snapshot incl. safe_user_status, last event, is_stale. |
| `/api/v1/double-agent/jobs/{id}/events` | GET | SSE stream (reuses `_sse_start`/`_sse_event`); falls back to JSON page if `Accept: application/json`. |
| `/api/v1/double-agent/jobs/{id}/cancel` | POST | Sets cancellation event; worker observes and finalizes as `canceled`. |
| `/api/v1/double-agent/jobs/{id}/mark-stale` | POST | Manually mark a job stale (e.g. operator override). |
| `/api/v1/double-agent/conversations/{conv_id}/revisions` | POST | Bump revision id. Auto-marks any older-revision running jobs as stale per `status_policy`. |
| `/api/v1/double-agent/conversations/{conv_id}/revisions` | GET | Current revision id + recent excerpt. |

## 7. New MCP Tools

| Tool | Kind | Purpose |
|---|---|---|
| `ms4.double_agent.submit@v1` | runtime_action | Submit a Double Agent job. Gated by `is_safe_*` guards; `can_mutate_world` must be false in phase 1. |
| `ms4.double_agent.status@v1` | read-only | Current snapshot for a job. |
| `ms4.double_agent.list@v1` | read-only | Active + recent jobs, optionally filtered by conversation. |
| `ms4.double_agent.events@v1` | read-only | Most recent N events for a job (page through the SSE stream synchronously). |
| `ms4.double_agent.cancel@v1` | runtime_action | Request cancellation. |
| `ms4.double_agent.mark_stale@v1` | runtime_action | Mark stale. |

Total tool count bump: 21 → 27. `submit/cancel/mark_stale` go in `safety.effectful_tools_in_v1`.

## 8. Face Lobe Behavior (phase 2)

Face Lobe is the existing chat path with three changes:

1. **Revision bump on every user message**. The gateway `chat`/`chat/stream` handlers call `blackboard.bump_revision(conversation_id)` first.
2. **Anti-hallucination context block**. `ms4_consciousness.on_pre_llm_call` is extended (via the existing context-injection point) to include:
   * Current `conversation_revision_id`.
   * Active Double Agent jobs (id, type, state, last `safe_user_status`).
   * The artifact §13 forbidden-claims list (rendered as constraints).
3. **No automatic submission in phase 1**. The Face Lobe will *accept* operator/tool-driven job submission and *report* on it, but routing decisions (artifact §15) stay manual through the MCP tool surface. Automatic intent classification is a phase 2.5 follow-up; doing it now risks the kind of "guess and dispatch" behavior we're explicitly trying to avoid.

## 9. Worker (phase 1) — How Background Execution Actually Works

```
HermesDoubleAgentWorker.run(envelope, blackboard, cancel_event):
  1. emit job.started with safe_user_status = envelope.user_visible_goal
  2. Build a fresh Ms4HermesRunner session keyed by envelope.job_id
  3. Wire callbacks that translate Hermes events into Double Agent events:
       tool_start_callback   -> job.tool.call.started
       tool_complete_callback-> job.tool.call.completed
       stream_delta_callback -> swallow tokens silently (anti-hallucination)
       plugin ethics block   -> job.ethics_block + abort
  4. Periodically (every status_policy.summarize_every_seconds) emit
     job.checkpoint with a safe_user_status derived from the most recent
     tool action (never from model reasoning).
  5. Cancellation: between iterations + before every tool call, check
     cancel_event.is_set(); if so emit job.canceled and stop.
  6. On completion, persist DoubleAgentJobResult.v1 + emit job.completed.
  7. On unhandled exception, emit job.failed with the exception summary.
```

The worker NEVER emits the model's raw reasoning tokens. The `summary` and `safe_user_status` strings are either operator-supplied templates ("checking smart bridge reachability") or operational facts derived from tool events ("read worker.rs", "command sent to bridge"). This satisfies §13.

## 10. Tests (phase 1)

```
tests/ms4_double_agent/
  test_schemas.py            # validators, allowlists, safe-id guards
  test_blackboard.py         # SQLite CRUD, revision bumping, stale-on-revision-change, hydration recovery
  test_runner.py             # single-flight submit, cancellation, list filtering
  test_worker.py             # event lifecycle order, anti-hallucination (raw_tokens never escape), MS3-ethics-block → job.ethics_block
  test_gateway_routes.py     # /api/v1/double-agent/* routes exposed + correct shapes
  test_mcp_tools.py          # registry count == 27, tools dispatch, manifest updated
  test_safety_contract.py    # forbidden event types refused at the schema layer; safe_user_status <= 280 chars enforced
```

## 11. Out-of-Scope Risks (acknowledged from §17)

* Foreground starvation — phase 1 caps `max_concurrent_jobs=2` and runs workers on a `ThreadPoolExecutor`, so even if a worker monopolizes a model, the foreground request thread is independent.
* Hallucinated status — addressed by §9 + §13 (workers cannot emit raw tokens; Face Lobe context block is constrained).
* Stale results — addressed by revision bumping + `mark_stale_jobs_for_revision`.
* Blocking model calls — accepted in phase 1. The Hermes worker still does a blocking model call, but the *foreground* is no longer blocked because the worker runs in another thread.
* Background tool authority — every tool call inside the worker still goes through `ms4_consciousness.on_pre_tool_call` → `Ms4Client.evaluate_action`, so existing MS3 ethics enforcement applies.
* Missing cancellation — addressed by `threading.Event` polled at all natural checkpoints.
* Fanout result conflicts — out of scope until phase 5 reconciler.
* Event stream leakage of unsafe/raw reasoning — addressed by `safety.py` allowlists at insert time.

## 12. Acceptance (mirrors §31 MVP)

Phase 1 acceptance:

1. POST `/api/v1/double-agent/jobs` returns 202 with a snapshot.
2. GET `/api/v1/double-agent/jobs/{id}` reports `queued -> running -> completed`.
3. Events stream (SSE or polled) shows lifecycle events in order with `safe_user_status`.
4. POST cancel transitions running job to `canceled` within `status_policy.summarize_every_seconds` (default 10s).
5. Bumping revision to N+1 marks any running job at revision N as `stale`.
6. Restarting the gateway hydrates and flips any dangling `running` jobs to `failed` (same recovery contract as `hermes_admin`).
7. The MCP surface exposes the 6 new tools; `tools_loaded == 27`.
8. The web UI shows an active-jobs panel pulled from the GET endpoint; clicking cancel posts the cancel route.
9. No event emitted has `type == "raw_reasoning_tokens"` (or any other non-allowlisted type) — schema layer rejects it.

Phase 2 acceptance (Face Lobe wiring):

10. Every user message bumps the revision.
11. The `on_pre_llm_call` context block lists active jobs and forbids invented status (verified via prompt assertion in `test_face_lobe_contract.py`).
12. No regression on the existing 96-test suite.
