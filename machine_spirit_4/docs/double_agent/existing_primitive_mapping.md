# Double Agent — Existing Primitive Mapping (Quick Reference)

Shortlist for engineers. For rationale, see `fit_gap_double_agent.md`.

## Reuse As-Is

| Need | File | Notes |
|---|---|---|
| SSE helpers | `machine_spirit_4/gateway/server.py` (`_sse_start`, `_sse_event`) | Already drains heartbeats cleanly; perfect for `/jobs/{id}/events`. |
| Audit JSONL | `machine_spirit_4/gateway/audit.py` | Mirror Double Agent events here in addition to SQLite. |
| Hermes worker loop | `Documents/hermes-agent/run_agent.py` `AIAgent.run_conversation` | Already accepts `stream_callback`, `tool_start_callback`, `tool_complete_callback`, `conversation_history`, `task_id`. |
| Per-session bookkeeping | `machine_spirit_4/gateway/hermes_runner.py` `Ms4HermesRunner` | Forked into a worker session keyed by `job_id`. |
| Identity / ethics gate inside the worker | `machine_spirit_4/plugins/hermes/ms4_consciousness/__init__.py` | `on_pre_tool_call` still fires for the background worker; no new ethics path required. |
| MCP tool registration | `machine_spirit_4/mcp/tools.py` `ToolDef`, `build_tool_registry` | Add 6 entries, bump manifest count. |
| Runtime containment | `machine_spirit_4/contained.py` | The Double Agent runner is imported by the same gateway; the guard already protects it. |

## Pattern Templates (Copy/Adapt, Don't Reinvent)

| Need | Template |
|---|---|
| Persistent job state machine with phase transitions, single-flight lock, on-disk recovery | `machine_spirit_4/hermes_admin/state.py` (`UpdateJobSnapshot`, `_StateStore.hydrate`). Generalize from one record to N records (SQLite). |
| Background thread runner with phase/status events | `machine_spirit_4/hermes_admin/installer.py` (`trigger_update`, daemon thread, `state.set_phase`). |
| Safe-target / allowlist string guard | `machine_spirit_4/hermes_admin/versioning.py` (`is_safe_target_version`). |

## HiveMind Surfaces Already Available (No New HiveMind Code Required for MVP)

| Need | HiveMind surface |
|---|---|
| Run a model | `POST /v1/chat/completions` (already used by `Ms4HermesRunner` via Hermes) |
| Long-running job analog | `/provision/status/{job_type}` — pattern to imitate, not call directly |
| Cluster inventory | `hivemind.cluster.summary@v1`, `hivemind.hosts.list@v1` |
| Resource lease (phase 4) | `POST /v1/resources/request` |

## Real Gaps (Net-New Code in MS4)

| Gap | Lives In |
|---|---|
| `DoubleAgentJobEnvelope.v1` + sibling schemas | `machine_spirit_4/double_agent/schemas.py` |
| Safe id/event/phase allowlists | `machine_spirit_4/double_agent/safety.py` |
| Multi-job SQLite blackboard + revision table | `machine_spirit_4/double_agent/blackboard.py` |
| Job runner (submit/list/get/cancel/mark_stale) | `machine_spirit_4/double_agent/runner.py` |
| Hermes-backed background worker w/ event translation | `machine_spirit_4/double_agent/worker.py` |
| 6 MCP tools | `machine_spirit_4/mcp/tools.py` |
| 7 gateway routes | `machine_spirit_4/gateway/server.py` |
| Face Lobe context block (revision id + active jobs + no-invented-status rules) | `machine_spirit_4/plugins/hermes/ms4_consciousness/psyche.py` extension or new `face_lobe.py` |
| UI active-jobs panel | `machine_spirit_4/web/index.html` |

## Phase Boundary

| Phase | This MVP includes | Deferred |
|---|---|---|
| 0 | This doc + research artifact stash | — |
| 1 | schemas, safety, blackboard, runner, worker, 7 gateway routes, 6 MCP tools, tests | — |
| 2 | Face Lobe context block + UI panel | Automatic intent classification / auto-submission |
| 4 | — | HiveMind capability-lease routing per `resource_request` |
| 5 | — | Fanout reducer / cross-job reconciler |
| 6 | — | Approval gates beyond MS3 ethics, replay tests, distributed cancellation |
