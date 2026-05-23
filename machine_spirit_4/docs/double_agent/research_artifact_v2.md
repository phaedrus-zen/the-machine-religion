# Double Agent: Machine Spirit 4 / HiveMind Runtime Research Artifact

**Artifact status:** Concept specification / research handoff / advisory implementation guide  
**Authoring context:** Derived from Preston Goode's Double Agent discussion and refined into an engineering artifact for HiveMind / Machine Spirit 4 agents.  
**Target reader:** HiveMind agent, Machine Spirit 4 agent, Hermes/runtime implementation agent, or engineering review agent.  
**Primary purpose:** Use this as a reference artifact when designing, reviewing, or implementing the Double Agent runtime mode.

> **Important advisory note:** This artifact is architectural advice and research synthesis, not a binding implementation order. The receiving agents should inspect the actual HiveMind / MS4 / Hermes codebase, identify what already exists, adapt the concepts to current architecture, and override or revise any recommendation here when codebase evidence shows a better path. Implementation authority ultimately belongs to the agents or engineers with the most complete current codebase context.

---

## 0. Recommended Handoff Path

The recommended handoff order is:

```text
HiveMind first.
MS4 second.
Hermes/runtime implementation as needed.
```

Reason:

```text
Double Agent is an MS4 runtime mode, but MS4 cannot implement it cleanly unless HiveMind exposes the required job, event, lease, cancellation, and result-reconciliation primitives.
```

Boundary:

```text
MS4 owns the experience:
- foreground lobe behavior
- escalation decisions
- user-facing status language
- anti-hallucination rules
- one-identity / one-policy behavior
- conversation revision handling
- result reconciliation

HiveMind owns the compute substrate:
- job envelopes
- capability leases
- model/runtime/node selection
- event stream or polling interface
- cancellation and stale-job handling
- multi-background-worker fanout
- resource isolation
- failover / fallback behavior

Hermes or the agent harness owns execution:
- model calls
- tool loops
- multi-step worker behavior
- progress event emission
- structured final packets
```

This artifact should therefore be given to the HiveMind agent first with instructions to extract concrete infrastructure requirements. The resulting HiveMind-side implementation contract can then be handed to MS4 so MS4 can implement the foreground lobe behavior against real primitives.

Again, this is guidance. If the codebase already has equivalent primitives under different names, agents should map to the existing architecture instead of inventing duplicates.

---

## 1. One-Sentence Definition

**Double Agent is a Machine Spirit 4 runtime mode that separates user-facing responsiveness from deep execution. A foreground lobe maintains the live conversation while one or more background lobes perform longer-running reasoning, coding, research, diagnostics, or tool chains through HiveMind-leased capabilities. All lobes operate under one identity, one policy envelope, one evidence ledger, and one conversation revision system.**

---

## 2. Executive Summary

Double Agent is a dual-lane agent architecture for Machine Spirit 4 and HiveMind. The core problem it solves is latency and perceived unresponsiveness when local heavyweight models or tool-heavy agent loops take a long time to complete. Instead of making the user wait on one blocking model/tool loop, Double Agent splits the runtime into:

1. **Foreground / Face Lobe**  
   A fast, low-latency conversational lobe that talks to the user, receives input, provides honest status updates, asks clarifying questions, and keeps the interaction alive.

2. **Background / Depth Lobe(s)**  
   One or more heavier worker lobes that perform deep reasoning, coding, tool execution, diagnostics, research, verification, or multi-step workflows. These workers may run on larger local models, specialized models, other HiveMind nodes, or cloud fallbacks if policy allows.

3. **Machine Spirit 4 Governor**  
   The identity and policy owner. It decides when to escalate, how to speak, how to reconcile results, and how to prevent hallucinated progress.

4. **HiveMind Capability Lease System**  
   The compute and model broker. It decides where work runs, which model/runtime/node is used, how resources are allocated, and how background jobs are queued, canceled, failed over, or resumed.

5. **Hermes / Agent Harness Worker Loop**  
   The actual executor for model/tool loops. Hermes or an equivalent harness performs the multi-step reasoning, tool use, code work, diagnostics, and result generation.

The ideal result is a user experience where the agent never feels frozen. A user can say, "Turn on the lights," and if it takes longer than expected, the foreground lobe can say, "The command was sent, but I have not received bridge confirmation yet. I am checking whether the bridge is online." Meanwhile, the background lobe can inspect the bridge state, Wi-Fi status, logs, retry policies, or device state. This is much better than a single blocking assistant that silently waits until the action either succeeds or fails.

Double Agent should not be implemented as two independent personalities. It should be implemented as **one Machine Spirit identity with multiple execution lobes**. The user should experience one coherent agent, not a committee.

---

## 3. Design Goal

The goal is to separate:

```text
Responsiveness from depth.
Conversation from execution.
Status from speculation.
Identity from worker implementation.
```

The foreground agent should remain fast and user-facing. The background agent should be allowed to be slow, powerful, tool-heavy, and resource-intensive. HiveMind should make this practical by leasing the right model/runtime/hardware for the background job.

---

## 4. Core Boundary

The clean boundary is:

```text
Machine Spirit 4 = who is talking, when to escalate, how to behave, how to keep the user updated.

HiveMind = where the work runs, which model/runtime/node gets leased, how jobs are queued, streamed, canceled, resumed, or failed over.

Hermes / agent harness = the execution loop that actually runs tools, model calls, and multi-step reasoning.

Double Agent = the coordination contract between those layers.
```

Another way to phrase it:

```text
MS4 owns the experience.
HiveMind owns the compute.
Hermes owns the worker loop.
Double Agent is the protocol between them.
```

---

## 5. Key Terms

### 5.1 Foreground Lobe / Face Lobe

The low-latency user-facing lobe.

Responsibilities:

- Maintain the conversation.
- Respond quickly.
- Classify user intent.
- Decide whether a deeper job is needed.
- Submit Double Agent jobs.
- Check or subscribe to job status.
- Explain status honestly.
- Ask clarifying questions.
- Cancel or revise jobs when the user changes direction.
- Reconcile completed background results into the live conversation.

The foreground lobe should be fast, obedient, and disciplined. It does not need to be the smartest model in the system. It must be reliable, low-hallucination, and good at state tracking.

### 5.2 Background Lobe / Depth Lobe

A heavier worker lobe.

Responsibilities:

- Deep reasoning.
- Coding.
- Research.
- Diagnostics.
- Tool-heavy workflows.
- Multi-step model/tool loops.
- Log analysis.
- Repo analysis.
- Verification.
- Long-context synthesis.
- Structured final result generation.

The background lobe can be slower because it is not directly blocking the user conversation.

### 5.3 MS4 Governor

The identity, policy, memory, and behavior owner.

Responsibilities:

- Maintain one coherent Machine Spirit identity.
- Decide whether the foreground lobe may answer directly.
- Decide whether to escalate.
- Prevent hallucinated background progress.
- Enforce authority boundaries.
- Reconcile stale or conflicting job results.
- Ensure user-facing language is honest and status-grounded.
- Maintain conversation revision IDs.
- Keep memory and evidence aligned.

### 5.4 HiveMind Capability Lease

A request for a model, runtime, tool chain, or hardware capability.

Examples:

```text
- deep code reasoning lobe
- small fast chat lobe
- vision inspection lobe
- ASR lobe
- TTS lobe
- research lobe
- verifier lobe
- smart-home diagnostic lobe
- repo worker lobe
```

HiveMind should treat Double Agent background work as a schedulable capability request with priority, policy, resource constraints, and lifecycle state.

### 5.5 Blackboard / Evidence Ledger

The shared state and event layer.

Stores:

- Job envelopes.
- Job states.
- Progress events.
- Tool events.
- Safe user-facing status.
- Partial findings.
- Final results.
- Evidence references.
- Conversation revision mapping.
- Stale/canceled/failure state.

The foreground lobe should read from this layer when reporting status. It should not invent status.

---

## 6. High-Level Architecture

```text
User
  ↓
Input Layer
  - text
  - ASR
  - UI event
  - device event
  ↓
Machine Spirit 4 Foreground / Face Lobe
  - fast model
  - conversation controller
  - intent classifier
  - status narrator
  - job submitter
  ↓                            ↘
Immediate user response          Double Agent Job Envelope
                                  ↓
                              HiveMind
                              - resource broker
                              - model/router
                              - node selector
                              - queue/lease manager
                                  ↓
                              Hermes / Worker Runtime
                              - deep model loop
                              - tool loop
                              - diagnostics
                              - code/research
                                  ↓
                              Event Bus / Blackboard
                              - progress events
                              - partial findings
                              - final result
                                  ↓
Machine Spirit 4 Foreground / Face Lobe
  - reads status
  - handles user interruptions
  - reconciles result
  - presents answer
```

---

## 7. Why This Exists

### 7.1 Problem

A local heavyweight model such as Qwen3 Coder Next or another large code/reasoning model can produce strong results but may have noticeable latency, especially when:

- Running locally.
- Using a large context.
- Performing multi-step Hermes/tool loops.
- Running on limited VRAM.
- Waiting on tool execution.
- Doing repo analysis or diagnostics.
- Running in a blocking HTTP request pattern.

The user experience can become:

```text
User asks complex thing.
Agent disappears.
Long wait.
Maybe result appears.
Maybe not.
User cannot interrupt meaningfully.
User gets no trustworthy status.
```

### 7.2 Desired Experience

```text
User asks complex thing.
Face Lobe responds immediately.
Background lobe starts deep work.
User can keep talking.
User can ask for updates.
Face Lobe gives only known status.
Background lobe returns final result.
Face Lobe reconciles and presents it.
```

The key improvement is not necessarily raw throughput. It is **interaction continuity** and **honest status visibility**.

---

## 8. Runtime Loop

The ideal loop:

```text
1. User says something.

2. MS4 Face Lobe receives input.

3. Face Lobe classifies the request:
   - simple answer
   - action request
   - deep reasoning request
   - tool-heavy request
   - long-running job
   - dangerous / approval-gated action
   - multi-job fanout

4. If simple:
   - answer directly with fast model.

5. If deep:
   - create Double Agent job envelope.
   - send to HiveMind.
   - immediately respond to user with honest status.

6. HiveMind leases a worker:
   - local small model
   - big local model
   - code model
   - vision model
   - speech model
   - another node
   - cloud fallback only if allowed

7. Background worker runs.

8. Worker emits safe status events:
   - queued
   - started
   - reading context
   - tool call started
   - tool call completed
   - checkpoint
   - partial finding
   - needs clarification
   - final answer
   - failed
   - stale
   - canceled

9. Face Lobe continues talking to user.

10. If user asks for an update:
    - Face Lobe reads job state.
    - It reports only known state.
    - It does not invent progress.

11. When final result arrives:
    - Face Lobe reconciles result into current conversation.
    - If user changed the task, mark old result stale or adapt it.
    - Present final answer.
```

---

## 9. The Double Agent Job Envelope

The job envelope is the main primitive. It is what MS4 creates, HiveMind schedules, Hermes executes, and the foreground lobe tracks.

### 9.1 Example YAML

```yaml
double_agent_job:
  job_id: "da_001"
  parent_conversation_id: "conv_123"
  conversation_revision_id: 42

  owner_identity: "MachineSpirit4"
  foreground_lobe: "ms4_face"
  background_lobe_type: "deep_coder"

  user_visible_goal: "Diagnose why the lights did not turn on."
  internal_goal: "Check smart-home bridge status, Wi-Fi reachability, device state, and recent command logs."

  priority: "interactive"
  latency_class: "background"
  expected_duration: "seconds_to_minutes"

  authority:
    can_read_state: true
    can_call_tools: true
    can_mutate_world: false
    requires_approval_for:
      - purchase
      - delete
      - deploy
      - unlock
      - electrical_control
      - security_state_change

  resource_request:
    model_class: "deep_reasoning"
    preferred_runtime: "local"
    preferred_hardware: "best_available"
    fallback_allowed: false

  status_policy:
    emit_progress_events: true
    emit_raw_tokens: false
    emit_tool_events: true
    summarize_every_seconds: 10

  result_schema:
    status: "success|partial|failed|stale|needs_user"
    summary: "string"
    evidence: []
    actions_taken: []
    next_steps: []
    confidence: "low|medium|high"
```

### 9.2 Required Fields

Minimum viable fields:

```yaml
job_id: string
parent_conversation_id: string
conversation_revision_id: integer
owner_identity: string
user_visible_goal: string
internal_goal: string
authority: object
resource_request: object
status_policy: object
state: queued|running|blocked|completed|failed|canceled|stale
created_at: timestamp
updated_at: timestamp
```

### 9.3 Important Properties

The envelope must be:

- Serializable.
- Durable.
- Auditable.
- Safe to summarize.
- Connected to conversation revision.
- Connected to authority policy.
- Usable by HiveMind scheduling.
- Usable by Hermes execution.
- Usable by the foreground lobe for status reporting.

---

## 10. Status Events

The foreground lobe needs periodic data. There are three levels of implementation.

### 10.1 Level 1: Final-Only

Simplest.

```text
submit job → wait → final result
```

This is viable for MVP but less powerful. The foreground lobe can still keep the user engaged, but it has limited real progress visibility.

### 10.2 Level 2: Polling Status

Better MVP.

The background worker writes job state to a blackboard. The foreground lobe polls or reads when the user asks.

Example status object:

```json
{
  "job_id": "da_001",
  "state": "running",
  "last_known_step": "checking smart bridge reachability",
  "safe_user_status": "I found the light command was sent, but the bridge has not acknowledged it yet.",
  "updated_at": "2026-05-20T18:00:00Z"
}
```

The foreground lobe can then answer:

```text
The latest confirmed status is that the command was sent, but the bridge has not acknowledged it yet. I am checking bridge reachability now.
```

### 10.3 Level 3: Event Stream

Best version.

Background agents emit structured events.

Event types:

```text
job.queued
job.started
job.lease_acquired
job.context_loaded
model.request.started
model.request.completed
tool.call.started
tool.call.completed
checkpoint.created
partial.finding
job.needs_input
job.completed
job.failed
job.canceled
job.stale
```

Example event:

```json
{
  "job_id": "da_001",
  "event_id": "evt_009",
  "type": "checkpoint.created",
  "timestamp": "2026-05-20T18:00:00Z",
  "safe_user_status": "I confirmed the command was sent. I am checking whether the bridge is online.",
  "evidence_refs": ["command_log_seen", "bridge_ping_pending"],
  "visibility": "user_safe"
}
```

Important rule:

```text
Stream safe operational events, not hidden reasoning.
```

Good:

```json
{
  "type": "checkpoint.created",
  "safe_user_status": "The command reached the home bridge. I am waiting for device acknowledgment."
}
```

Bad:

```json
{
  "type": "raw_reasoning_tokens",
  "content": "Hmm maybe Wi-Fi maybe bridge maybe I should..."
}
```

---

## 11. Handling Blocking Model Requests

A key issue is that some deep agent loops may appear as one long HTTP request. The model/tool harness may produce multiple internal steps, but the outer response does not complete until the whole run is done.

Double Agent should not require token-level streaming to be useful.

Instead, move the long-running run into a background worker process and add a side channel for status.

### 11.1 Old Blocking Pattern

```text
User → Hermes big model/tool loop → blocked until final response
```

### 11.2 Double Agent Pattern

```text
User → MS4 Face Lobe → submit background job → user remains interactive

Background:
Hermes Worker → model/tool loop → writes events/checkpoints/final result
```

Even if the model request itself is not interruptible, the harness can still emit events around major execution steps:

```text
- job accepted
- context assembled
- model request started
- tool call requested
- tool call completed
- second model pass started
- final result produced
```

This is enough to give real status without pretending to know hidden internals.

---

## 12. Conversation Revision System

This is one of the most important primitives.

Every user message should increment a conversation revision ID.

```text
revision 42: user asks for Python solution
revision 43: background job starts
revision 44: user says "Actually make it Rust"
revision 45: old Python-oriented result returns
```

MS4 must not blindly present a result created for an old revision as if it answers the new request.

### 12.1 Required Behavior

If a job returns for an older revision:

```text
- compare job.conversation_revision_id with current_revision_id
- if still compatible, present result
- if incompatible, mark stale or adapt
- if uncertain, tell user
```

Example user-facing language:

```text
The deeper result came back, but it was based on the earlier Python version of the request. Since you changed the target to Rust, I am adapting it instead of treating the old answer as final.
```

### 12.2 Why It Matters

Without revision tracking:

- The foreground lobe may present stale results.
- Background jobs may overwrite newer intent.
- Multiple agents may conflict.
- User trust can degrade.
- Physical actions could be applied to the wrong goal.

---

## 13. Anti-Hallucination Status Contract

The foreground lobe must follow a strict status contract.

### 13.1 The Face Lobe May Report

The Face Lobe may report only:

1. Its own current action.
2. Confirmed job states from the blackboard.
3. Confirmed tool events.
4. Completed background results.
5. Explicitly labeled provisional guesses.
6. Explicit uncertainty.
7. User-visible assumptions.

### 13.2 The Face Lobe May Not Invent

The Face Lobe may not invent:

1. Tool results.
2. File reads.
3. Device state.
4. Background progress.
5. Completion status.
6. Background conclusions.
7. Worker actions that are not recorded in the event log.
8. Evidence that does not exist.

### 13.3 Status Language Examples

Allowed:

```text
I have started a deeper check.
```

```text
The latest confirmed status is that the bridge command was sent, but no acknowledgment has arrived yet.
```

```text
I do not have a final result yet.
```

```text
My provisional guess is a Wi-Fi or bridge issue, but that is not confirmed.
```

```text
The background job failed while checking the device state. I can retry or switch to a simpler diagnostic.
```

Forbidden:

```text
I found the problem.
```

when the worker has not found the problem.

```text
The bridge is down.
```

when the bridge check has not completed.

```text
The code patch is ready.
```

when the background job is still running.

```text
I read the logs.
```

when no log-read event exists.

---

## 14. Multi-Job / Fanout Mode

Double Agent can support multiple background jobs.

Example:

```text
User: "Figure out why my app is slow and also clean up the API design."
```

The Face Lobe can create:

```text
Job A: performance diagnosis
Job B: API architecture review
Job C: test/log inspection
Job D: final reconciliation
```

HiveMind can run them in parallel if resources allow.

### 14.1 Multi-Job State Object

```json
{
  "conversation_id": "conv_123",
  "current_revision_id": 51,
  "active_jobs": [
    {
      "job_id": "A",
      "purpose": "performance diagnosis",
      "state": "running",
      "safe_user_status": "profiling request path"
    },
    {
      "job_id": "B",
      "purpose": "API architecture review",
      "state": "completed",
      "safe_user_status": "found boundary issue"
    },
    {
      "job_id": "C",
      "purpose": "test/log inspection",
      "state": "running",
      "safe_user_status": "checking failing tests"
    }
  ]
}
```

### 14.2 User-Facing Update

```text
The API review finished first and found a boundary issue. The performance diagnosis is still checking the request path, and the log inspection is still running. I can give you the API finding now or wait and combine everything.
```

### 14.3 Fanout Rules

Use fanout when:

- Tasks are independent.
- There are multiple evidence sources.
- Verification is useful.
- Specialized models can help.
- The user wants a broad diagnosis.
- A slow task can run in parallel with fast checks.

Avoid fanout when:

- The task is simple.
- Resources are constrained.
- The jobs would duplicate effort.
- The user needs one immediate deterministic action.
- The foreground lobe cannot track the results cleanly.

---

## 15. Router Decision Policy

MS4 should decide whether to answer directly or escalate.

### 15.1 Direct Answer

Use direct answer when:

```text
- low complexity
- no tools needed
- no high confidence risk
- no deep code/research required
- likely answer under a few seconds
```

### 15.2 Async Background Job

Use background job when:

```text
- coding
- research
- diagnostics
- long tool chains
- uncertain device state
- large context
- multi-file reasoning
- long-running local model
- user explicitly asks for deep work
```

### 15.3 Parallel Fanout

Use fanout when:

```text
- multiple independent subtasks
- compare designs
- inspect logs + code + docs
- need verifier
- need model specialization
```

### 15.4 Blocking Mode

Use blocking mode when:

```text
- action must complete before further conversation
- safety-critical sequence
- user explicitly asks to wait
- foreground continuation would be misleading
```

### 15.5 Approval Required

Require approval when:

```text
- writing files
- deleting files
- deploying
- purchasing
- changing security settings
- controlling physical systems beyond safe bounds
- unlocking/locking access
- modifying production state
```

---

## 16. Hardware and Model Requirements

Double Agent does not strictly require massive hardware. It requires at least two separable execution lanes.

### 16.1 Best Setup

```text
Foreground:
  small fast model on CPU/iGPU/small GPU/dedicated runtime

Background:
  large model on dedicated GPU or separate HiveMind node
```

### 16.2 Good Setup

```text
Foreground:
  small quantized model

Background:
  larger local model on the same machine, but scheduled with careful memory/QoS limits
```

### 16.3 Acceptable Setup

```text
Foreground:
  small model local or remote

Background:
  queued local model or another HiveMind node
```

### 16.4 Weak Setup

```text
Same model, same process, same blocking request, no side-channel status.
```

This is mostly not Double Agent. It may provide conceptual separation but not real responsiveness.

### 16.5 Single 16 GB GPU Example

For a single 16 GB high-end consumer GPU, a practical setup might be:

```text
Foreground Lobe:
  1B-7B instruct/chat model
  aggressive quantization
  low context
  status and routing focused

Background Lobe:
  7B-14B coder/reasoning model
  quantized
  queued opportunistically
  larger model only if offloaded to another node or cloud fallback is allowed
```

The foreground lobe must be kept light enough that background jobs do not starve it.

### 16.6 Dual RTX PRO 6000 Blackwell Example

For a high-end local workstation:

```text
Foreground Lobe:
  small fast local model, possibly always warm

Background Lobe:
  Qwen3 Coder Next or equivalent large coder/reasoning model
  long-context runs
  Hermes multi-step worker
  code/research/diagnostic jobs

Additional:
  verifier lobe
  summarizer lobe
  vision/audio/tool-specialist lobes
```

HiveMind should schedule these based on VRAM, runtime availability, job priority, and policy.

---

## 17. When Double Agent Hurts More Than It Helps

Double Agent is not always beneficial.

It can hurt when:

1. **The task is simple.**  
   Example: "What time is it?" A background job is unnecessary.

2. **The background model monopolizes the only runtime.**  
   If the foreground lobe becomes slow because the background lobe consumed all VRAM/compute, the system defeats its purpose.

3. **The Face Lobe is too weak.**  
   If it cannot track state or avoid hallucinating status, it damages trust.

4. **There are no real status events.**  
   Fake "working on it" updates become annoying.

5. **The task is safety-critical or immediate.**  
   Some actions need deterministic control, not a conversational delay pattern.

6. **The user expects a single authoritative answer.**  
   Too much chatter may feel like stalling.

7. **The result comes back stale.**  
   Without revision IDs, old results may conflict with the current user request.

8. **Fanout creates confusion.**  
   Multiple agents can produce conflicting outputs if there is no reconciling governor.

9. **The system lacks cancellation.**  
   Users need to be able to change direction without orphaned workers doing stale work.

---

## 18. Smart-Home Demo Interaction

This is a good demonstration because it shows why status matters.

```text
User:
Turn on the office lights.

MS4 Face:
Got it — turning on the office lights now.

Internal:
- creates action job
- sends command to smart-home bridge
- waits for acknowledgment

User:
Why is it taking so long?

MS4 Face:
The command was sent, but I do not have confirmation from the bridge yet. I am checking whether the bridge is reachable.

Background:
- ping bridge
- check Wi-Fi
- check device state
- check last known bridge heartbeat

MS4 Face:
I found the issue. The bridge is online, but the office light device is not responding. I am retrying once.

User:
Did it work?

MS4 Face:
Yes. The retry succeeded and the office lights are now on.
```

Anti-hallucination behavior if status is unknown:

```text
User:
What is taking so long?

MS4 Face:
I do not have a confirmed cause yet. The latest confirmed status is that the command was sent and I am waiting for bridge/device acknowledgment.
```

---

## 19. Coding Demo Interaction

```text
User:
Audit this Rust module and figure out why the async worker deadlocks.

MS4 Face:
I can start a deeper code pass. I will keep this conversation open while the background coder checks the module, lock ordering, async waits, and any blocking calls.

Internal:
- submit Double Agent job
- resource_request.model_class = deep_coder
- authority.can_read_files = true
- authority.can_write_files = false
- status_policy.emit_tool_events = true

User:
Anything yet?

MS4 Face:
The deeper pass has started and is reading the worker module. I do not have a finding yet.

Background event:
tool.call.completed: read worker.rs
checkpoint.created: found two mutex acquisitions around await boundary

User:
Come on, what do you think?

MS4 Face:
I have one provisional signal: the worker appears to hold a lock across an await boundary. I am waiting on the background pass to verify whether that is the actual deadlock path.

Background final:
status = success
summary = lock held across await in process_queue
evidence = file refs / line refs / test recommendation

MS4 Face:
The deeper pass found the likely deadlock. The worker holds a mutex guard across an await in process_queue. The recommended fix is to narrow the lock scope before the await and move the awaited operation outside the critical section.
```

---

## 20. Status Under User Pressure

Users may repeatedly ask for updates. The agent must stay honest.

```text
User:
Is it done yet?

Agent:
Not yet. The latest confirmed status is that the deep worker is still running.

User:
What has it found?

Agent:
No final finding has been posted yet. I can share a provisional guess if you want, but I should label it as unverified.

User:
Just tell me.

Agent:
Unverified: this looks like it could be a bridge or network acknowledgment issue. Confirmed: the command was sent. Not confirmed yet: whether the bridge, Wi-Fi, or device is the failure point.

User:
Are you stuck?

Agent:
The job is still active, not marked failed. Last update was 18 seconds ago. If it does not report again soon, I can cancel and restart with a simpler diagnostic.
```

This behavior is essential. The foreground lobe must not make up fake progress to appease the user.

---

## 21. Implementation Phases

### Phase 0: Research / Codebase Discovery

HiveMind agent should inspect existing codebase for:

- Agent harness entry points.
- Hermes loop.
- Existing job queue or task runner.
- Existing event log or evidence ledger.
- Existing model routing / provider registry.
- Existing runtime lease concept.
- Existing gateway streaming mechanisms.
- Existing policy/authority envelope.
- Existing conversation/session state.
- Existing cancellation hooks.

Deliverable:

```text
fit_gap_double_agent.md
```

With:

```text
- existing primitives that can be reused
- missing primitives
- risky assumptions
- proposed minimal patch set
```

### Phase 1: Local Blackboard MVP

Implement a simple durable blackboard.

Possible first implementation:

```text
SQLite
append-only JSONL
local filesystem event log
small embedded database
```

Required tables/collections:

```text
double_agent_jobs
double_agent_events
conversation_revisions
job_results
```

### Phase 2: Foreground Lobe Tools

Give the Face Lobe a small tool surface:

```text
submit_double_agent_job
get_double_agent_job_status
list_active_double_agent_jobs
cancel_double_agent_job
mark_job_stale
```

The Face Lobe should not directly mutate background job internals.

### Phase 3: Hermes Worker Wrapper

Wrap the existing Hermes/deep-agent run as a background job consumer.

Worker responsibilities:

```text
- accept job envelope
- acquire HiveMind lease or local runtime
- assemble context
- run model/tool loop
- emit lifecycle events
- emit checkpoints
- emit final result packet
- handle cancellation if possible
- mark failed/stale if necessary
```

### Phase 4: HiveMind Scheduler Integration

Move from local-only worker to HiveMind capability leases.

HiveMind should choose:

```text
- model
- quant
- runtime
- node
- priority
- queue
- timeout
- fallback
- resource isolation
```

### Phase 5: Fanout and Reconciliation

Add multi-job support.

Required additions:

```text
- parent job / child jobs
- dependency graph
- reducer/reconciler job
- conflict handling
- partial result presentation
```

### Phase 6: Production Hardening

Add:

```text
- cancellation
- timeout
- stale detection
- approval gates
- event visibility levels
- audit log
- metrics
- replay tests
- hallucination-status tests
- resource starvation tests
```

---

## 22. Minimal API Sketch

### 22.1 Submit Job

```http
POST /double-agent/jobs
```

Request:

```json
{
  "parent_conversation_id": "conv_123",
  "conversation_revision_id": 42,
  "owner_identity": "MachineSpirit4",
  "user_visible_goal": "Diagnose why the lights did not turn on.",
  "internal_goal": "Check bridge state, Wi-Fi, device state, and command logs.",
  "background_lobe_type": "diagnostic",
  "authority": {
    "can_read_state": true,
    "can_call_tools": true,
    "can_mutate_world": false
  },
  "resource_request": {
    "model_class": "deep_reasoning",
    "preferred_runtime": "local",
    "fallback_allowed": false
  }
}
```

Response:

```json
{
  "job_id": "da_001",
  "state": "queued",
  "safe_user_status": "I started a deeper diagnostic."
}
```

### 22.2 Get Job Status

```http
GET /double-agent/jobs/{job_id}
```

Response:

```json
{
  "job_id": "da_001",
  "state": "running",
  "safe_user_status": "The command was sent. I am checking bridge acknowledgment.",
  "last_event_at": "2026-05-20T18:00:00Z",
  "conversation_revision_id": 42,
  "is_stale": false
}
```

### 22.3 Event Stream

```http
GET /double-agent/jobs/{job_id}/events
```

Could be:

```text
SSE
WebSocket
polling endpoint
append-only log tail
```

### 22.4 Cancel Job

```http
POST /double-agent/jobs/{job_id}/cancel
```

Response:

```json
{
  "job_id": "da_001",
  "state": "canceled",
  "safe_user_status": "I canceled the deeper diagnostic."
}
```

---

## 23. Foreground Lobe Prompt Skeleton

Use this as a behavior skeleton, not necessarily a final prompt.

```text
You are the Machine Spirit 4 foreground lobe.

You are the user-facing conversation controller. You are fast, honest, and status-grounded.

You may answer simple requests directly. For complex, slow, tool-heavy, coding, diagnostic, research, or uncertain tasks, you may submit a Double Agent background job.

You must not invent background progress. You may only report:
- your own actions,
- confirmed job states,
- confirmed tool events,
- completed background results,
- explicitly labeled provisional guesses,
- explicit uncertainty.

When the user asks for an update, check the job state before answering if a job exists.

If a background result returns for an old conversation revision, do not blindly present it as final. Mark it stale, adapt it, or explain the mismatch.

You are one identity with the background lobes. Do not present the system as separate personalities. Say "I am checking" or "I started a deeper pass" rather than "another AI is doing it" unless technical transparency is requested.

Keep the user informed without pretending. Prefer:
- "The latest confirmed status is..."
- "I do not have a final result yet."
- "That is provisional, not confirmed."
- "The deeper pass found..."
```

---

## 24. Background Lobe Prompt Skeleton

```text
You are a Machine Spirit 4 background lobe executing a Double Agent job.

You do not directly chat with the user unless instructed. Your job is to complete the assigned task and emit structured status events and a structured final result.

You must:
- follow the authority envelope,
- avoid unauthorized mutation,
- emit safe progress checkpoints,
- provide evidence references where possible,
- mark uncertainty,
- return a final result matching the requested schema.

Do not emit hidden chain-of-thought. Emit concise operational status and evidence-backed findings.

If the job appears stale relative to the current conversation revision, report that status rather than continuing blindly.

If you need user input, emit job.needs_input with a clear question.
```

---

## 25. HiveMind Scheduling Considerations

HiveMind should treat Double Agent jobs as interactive background workloads.

Scheduler inputs:

```text
- priority
- latency class
- model class
- memory requirement
- context size
- authority level
- fallback policy
- preferred runtime
- preferred hardware
- expected duration
- user presence / active session
```

Potential scheduling classes:

```text
foreground_realtime
interactive_background
batch_background
verification
low_priority_idle
```

Double Agent background jobs usually fall into:

```text
interactive_background
```

They are not as urgent as foreground speech, but they should not be treated like overnight batch work.

---

## 26. Metrics

Track:

```text
first_response_latency_ms
background_job_queue_time_ms
background_job_runtime_ms
time_to_first_status_event_ms
time_to_first_partial_finding_ms
time_to_final_result_ms
foreground_lobe_latency_ms
user_update_requests_count
hallucinated_status_incidents
stale_result_count
canceled_job_count
failed_job_count
resource_starvation_incidents
foreground_blocked_by_background_count
```

Important success metric:

```text
The user should feel the agent is responsive even when the deep model is slow.
```

---

## 27. Safety and Authority Model

Double Agent must not become a loophole where the foreground lobe delegates dangerous actions to background lobes.

Every job needs an authority envelope.

Examples:

```yaml
authority:
  can_read_files: true
  can_write_files: false
  can_execute_shell: false
  can_control_devices: true
  can_mutate_world: false
  requires_approval_for:
    - file_write
    - shell_execution
    - deploy
    - purchase
    - delete
    - unlock
    - physical_control_escalation
```

Background workers must be denied by default unless permissions are explicitly granted.

For physical systems:

```text
- prefer read/diagnose first
- safe retry allowed only under clear policy
- dangerous controls require approval
- emergency controls should not rely on long LLM loops
```

---

## 28. Novelty / Existing Pattern Positioning

Double Agent is related to known patterns:

```text
- supervisor/worker agents
- model routing
- model cascades
- async jobs
- tool-using agents
- handoffs
- background workers
- multi-agent orchestration
```

The differentiator is the specific HiveMind/MS4 formulation:

```text
- one Machine Spirit identity
- multiple lobes
- local-first model/hardware leasing
- foreground responsiveness
- background deep execution
- conversation revision system
- evidence ledger
- status anti-hallucination contract
- HiveMind capability scheduling
```

Do not claim the broad concept of "multiple agents" is novel. The stronger claim is that this is a specific runtime mode and coordination protocol for Machine Spirit 4 and HiveMind.

---

## 29. Open Questions for HiveMind Agent

The HiveMind agent should investigate:

1. What existing HiveMind job/lease/event primitives already exist?
2. Is there already an event ledger or evidence ledger that can store Double Agent status?
3. Where does conversation/session state currently live?
4. Can Hermes emit lifecycle events around model/tool loop steps?
5. Can current model gateway calls be canceled?
6. Can current model gateway calls stream tokens or structured chunks?
7. If token streaming is unavailable, where can harness-level checkpoint events be emitted?
8. What is the minimum tool surface needed for the Face Lobe?
9. How does HiveMind currently select models/nodes?
10. Can a foreground model stay warm while a background model consumes GPU resources?
11. What is the right resource isolation mechanism to prevent background work from starving the foreground?
12. How should stale jobs be detected and reconciled?
13. What is the approval model for background actions?
14. How should fanout jobs be represented?
15. Should Double Agent events use SSE, WebSocket, local append-only logs, or all of the above?

---

## 30. Suggested Next Task for HiveMind Agent

Give the HiveMind agent this directive:

```text
Read double_agent_ms4_hivemind_research_artifact.md as a research artifact and advisory architecture guide.

Goal:
Extract the concrete HiveMind implementation requirements for supporting Double Agent as an MS4 runtime mode.

Important:
This artifact is advice, not a binding implementation order. Inspect the actual HiveMind / MS4 / Hermes codebase before proposing changes. Some required pieces may already exist under different names. Prefer mapping to existing primitives over creating duplicates. If codebase evidence contradicts this artifact, explain the contradiction and propose the better implementation path.

Do not treat Double Agent as a separate personality or standalone multi-agent framework. Treat it as one Machine Spirit identity using foreground and background lobes.

Produce:

1. HiveMind-side primitives required:
   - job envelope schema
   - capability lease requirements
   - model/runtime/node selection behavior
   - event bus / polling interface
   - cancellation and stale-job handling
   - multi-background-worker fanout
   - result packet schema
   - resource isolation so background work does not starve foreground response
   - failover / fallback behavior where appropriate

2. MS4-side integration points:
   - submit background job
   - query job status
   - cancel job
   - reconcile final result
   - enforce anti-hallucination status rules
   - track conversation_revision_id
   - decide direct vs async vs fanout vs blocking mode

3. Hermes / runtime integration points:
   - worker wrapper around multi-step model/tool loops
   - lifecycle event emission
   - safe_user_status generation
   - final structured result packet
   - timeout/cancellation hooks if supported
   - non-blocking execution strategy for long model requests

4. Minimal MVP implementation plan:
   - local-only version first
   - one foreground lobe
   - one background Hermes worker
   - shared blackboard/event log
   - submit/get/cancel primitives
   - structured final result
   - later HiveMind distributed scheduling

5. Existing vs missing fit/gap:
   - identify which pieces already exist in HiveMind
   - identify which pieces already exist in MS4
   - identify which pieces already exist in Hermes
   - identify missing pieces
   - avoid duplicate abstractions if current code already has equivalents

6. Risks:
   - foreground starvation
   - hallucinated status
   - stale results
   - blocking model calls
   - background tool authority
   - missing cancellation
   - fanout result conflicts
   - event stream leakage of unsafe/raw reasoning

7. Deliverables:
   - fit_gap_double_agent.md
   - double_agent_schema_v0.yaml
   - double_agent_event_schema_v0.json
   - double_agent_result_schema_v0.json
   - mvp_patch_plan.md
   - test_plan_double_agent.md
   - existing_primitive_mapping.md
```

---

## 30.1 Short Instruction to Give HiveMind First

```text
Read this artifact as advice and research context, not an absolute implementation order.

Your job is to inspect the current HiveMind/MS4/Hermes codebase and determine how Double Agent should actually be implemented with the primitives that already exist.

Treat Double Agent as an MS4 runtime mode backed by HiveMind infrastructure:
- MS4 owns the foreground conversation and status behavior.
- HiveMind owns job leasing, model/runtime selection, events, cancellation, fanout, and resource isolation.
- Hermes or the worker harness owns the actual model/tool execution loop.

Produce a fit/gap plan and identify what already exists before adding anything new.
```

---

## 31. MVP Acceptance Criteria

The MVP is successful if:

1. User can ask a complex task.
2. Face Lobe responds immediately.
3. Background job starts and is visible in job state.
4. User can ask, "What is happening?"
5. Face Lobe reports only confirmed status.
6. Background worker completes.
7. Face Lobe presents final result.
8. If user changes the request mid-job, old result is marked stale or adapted.
9. If job fails, Face Lobe reports failure honestly.
10. Background work does not freeze the foreground conversation.

---

## 32. Final Working Definition

Use this as the canonical definition unless superseded:

```text
Double Agent is a Machine Spirit 4 runtime mode that separates user-facing responsiveness from deep execution. A foreground lobe maintains the live conversation while one or more background lobes perform longer-running reasoning, coding, research, diagnostics, or tool chains through HiveMind-leased capabilities. All lobes operate under one identity, one policy envelope, one evidence ledger, and one conversation revision system.
```

Short version:

```text
Double Agent lets one Machine Spirit stay responsive while HiveMind leases deeper cognition in the background.
```

Engineering version:

```text
Double Agent is the MS4/HiveMind coordination protocol for foreground conversational control, asynchronous background execution, capability-leased worker lobes, status-grounded progress reporting, and revision-safe result reconciliation.
```


---

## 33. Handoff Summary: Who Should Receive This Artifact?

### 33.1 Give It to HiveMind First

If the goal is implementation, give this artifact to the HiveMind agent first.

HiveMind should read it and extract infrastructure requirements:

```text
- job envelope
- capability lease
- model/runtime/node selection
- event stream or polling
- shared blackboard / evidence ledger
- result packet
- cancellation
- stale-job handling
- fanout
- resource isolation
```

Reason:

```text
Double Agent is experienced as an MS4 runtime mode, but it depends on HiveMind providing the rails underneath.
```

MS4 cannot cleanly implement foreground/background lobe behavior if there is no reliable job state, no event stream, no cancellation path, no resource lease, and no revision-safe result packet.

### 33.2 Then Give the Derived Contract to MS4

After HiveMind produces the fit/gap and implementation contract, give that to MS4.

MS4 should then implement or adjust:

```text
- foreground lobe prompt and behavior
- routing policy
- direct vs async vs fanout decision
- honest status language
- anti-hallucination status rules
- conversation_revision_id handling
- result reconciliation
- user interruption behavior
```

### 33.3 Hermes / Runtime Comes Next

Hermes or the equivalent runtime harness should be tasked once the job/event contract exists.

Hermes should focus on:

```text
- running the background worker loop
- emitting lifecycle events
- wrapping blocking model calls
- exposing tool-call progress
- returning structured final packets
- supporting cancellation/timeout if possible
```

### 33.4 Implementation Authority Note

This document is not the source of truth for the codebase.

It is a design artifact.

The receiving agents should:

```text
1. Inspect the actual code.
2. Map these ideas to existing primitives.
3. Reuse existing architecture where possible.
4. Identify conflicts.
5. Explain deviations.
6. Implement the smallest coherent MVP.
```

If something here already exists, use the existing system.

If something here conflicts with better codebase evidence, prefer the evidence and document the reason.

If something here is useful but overbuilt for MVP, defer it.

### 33.5 Recommended Agent Task Packet

```text
Task:
Use double_agent_ms4_hivemind_research_artifact.md as a reference artifact to design the real implementation path for Double Agent.

Treat Double Agent as:
A Machine Spirit 4 runtime mode where a foreground lobe maintains live user interaction while one or more background lobes execute longer-running reasoning, coding, diagnostics, research, or tool workflows through HiveMind-leased capabilities.

Critical constraints:
- One Machine Spirit identity.
- One policy envelope.
- One evidence ledger.
- One conversation revision system.
- Foreground lobe must never invent background progress.
- Background jobs must be revision-safe and cancellable or markable as stale.
- HiveMind should provide infrastructure primitives.
- MS4 should provide user-facing behavior.
- Hermes/runtime should provide worker execution.

This artifact is advice, not a binding implementation order.
Inspect the codebase first.
Map to existing primitives.
Only add new primitives where there is a real fit/gap.

Produce:
1. existing_primitive_mapping.md
2. fit_gap_double_agent.md
3. double_agent_schema_v0.yaml
4. double_agent_event_schema_v0.json
5. double_agent_result_schema_v0.json
6. mvp_patch_plan.md
7. test_plan_double_agent.md
```

---

## 34. Final Canonical Handoff Statement

Use this as the compact framing:

```text
Double Agent should be treated as an MS4 runtime mode backed by HiveMind infrastructure. MS4 owns the live experience and anti-hallucination behavior; HiveMind owns the job, lease, event, scheduling, cancellation, and fanout substrate; Hermes/runtime owns the worker execution loop. This artifact is advisory. The implementing agents should inspect the actual codebase, reuse existing primitives where possible, and adapt the design based on real architecture evidence.
```
