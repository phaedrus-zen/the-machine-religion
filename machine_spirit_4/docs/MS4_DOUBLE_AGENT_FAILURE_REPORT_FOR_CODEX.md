# MS4 Double-Agent — Failure Report & Fix Plan (for Codex)

**Date:** 2026-06-13 · **Author:** Claude (read-only investigation; no MS4 files edited, nothing run)
**Scope:** `C:\Users\nexus-hc-win-00\Downloads\TMR\machine_spirit_4` (MS4 gateway) + the external Hermes
agent at `C:\Users\nexus-hc-win-00\Documents\hermes-agent`.
**Evidence appendix (verbatim logs):** `docs/MS4_FAILURE_EVIDENCE_20260613.md` (same dir).
All `file:line` below were verified against the source tree.

> NOTE ON A COMMON MISCONCEPTION (read this first): the FACE lobe does **not** call `hivemind.*` tools.
> `gateway/face_lobe_chat.py` is a **tool-less** direct call to HiveMind `/v1/chat/completions` (docstring
> 1-22; `chat()` 269-433 sends only system+history+user, never a `tools` array). The "hivemind-mcp-live-context"
> you see is a **grounding LABEL** (set at `gateway/context.py:624`): MS4's gateway synchronously pre-fetches
> `hivemind.cluster.summary@v1` + `hivemind.hosts.list@v1` (`context.py:617-618`) and **injects the result as
> text** into the Face lobe's system prompt (`hermes_runner.chat` builds `extra_system` at 673-690). The DEPTH
> lobe is the only real Hermes tool-loop agent — and it has **no** `hivemind.*` tools registered. Do **not**
> "fix" the Face side to call tools; that path works as designed.

---

## Shared root cause (the one design flaw behind 5 of 6 bugs)
MS4's two-lobe design **free-generates over tool results that are absent, unreliable, or context-less, with no
integrity contract at the seams.** Each layer trusts the layer beneath without verifying it:
- the FACE lobe trusts the model not to invent tool output (no output-vs-grounding check) → **#1 fabrication**;
- the router punts tool-shaped turns to DEPTH on weak signals (**#5**) carrying **no conversation context** (**#4**);
- the DEPTH lobe it lands on has **none of the 143 HiveMind tools registered** (**#2 / toolbox**);
- and the model picker may hand either lobe a **non-chat model** (**#3 Qwen3-TTS**) that returns empty content.
The fix theme is to **add a verification/contract at each seam**, not to coax the model with prose.

---

## Ranked fixes

### RANK 1 — TRUST BREAK: Face lobe presents invented / mis-attributed JSON as real tool output (P0)
**Symptom (logs):** user "Results?" → Face lobe emitted `{"total_nodes":4,"active_nodes":4}` and
`[{"node_id":"node-123","gpu_count":2},...]` as the deep job's output (job `da-7bbf1888` had produced nothing);
"weather" → invented `{"current_weather":{"temperature":22°C,...}}` attributed to a GPU job; "can you list the
GPUs?" → "The Depth Lobe job da-7bbf1888 already listed the GPUs" (it didn't). grounding=`face_lobe` (no payload).
**Code / mechanism:** `gateway/face_lobe_chat.py:404` returns model text verbatim (`cleaned = text.strip()`)
with **zero** validation that emitted JSON/tool-results match the grounding MS4 injected. The only guard is prose
in the system prompt (`face_lobe.py:35-60` / `face_lobe_chat.py:69-131`), which small models ignore.
**Exact fix:** add an output guard between the fallback chain (~`face_lobe_chat.py:402`) and the verbatim return
(`:404`). Thread the injected grounding text (the `extra_system` built at `hermes_runner.py:673-690`) into
`chat()` so the guard has an allowlist. Detect emitted tool-result/JSON blocks (fenced ```json, bare `{...}` with
`tool`/`result`/`id` keys, fabricated symbols like `tool_manager`, `ServerSideToolManager`, `hivemind.*@v1`
called as code). For any block **not supported by the grounding**: strip it and replace with an honest
disclaimer; if a stored Depth-Lobe job result exists for the session, substitute the **real** result. Fail
honest — when in doubt, strip rather than pass through.
**Test:** stub the model to emit invented JSON while `extra_system` lacks it → assert stripped; seed a completed
job result → assert "Results?" echoes the real result; a grounded answer (JSON present in `extra_system`) passes
unchanged.
**Self-contained:** No (needs grounding threaded into `chat()` + a strip-vs-substitute policy decision).

### RANK 2 — Model picker selects non-chat models (Qwen3-TTS) as the chat/depth model (P0, quick win)
**Symptom (logs):** Face + Depth pickers show `● Qwen3-TTS` selected; many deep jobs returned "the model returned
empty content after retries and any fallback providers" (a TTS model has no chat head). `face_lobe.default_model
= qwen3-coder-next:latest` yet turns ran `llama3.1:8b` / `qwen2.5:0.5b`.
**Code / mechanism:** `double_agent/model_picker.py:87` and `depth_picker.py:57` carry a **bare `qwen3`** priority
token; matched by substring (`model_picker.py:243` / `depth_picker.py:163`) it collides with `qwen3-tts`.
`_normalize_model_entry` (`model_picker.py:170-190`) reads only id/status/loaded — **never task/modality/
capability** — so a TTS entry is indistinguishable from chat. The UI `/models` proxy is unfiltered
(`server.py:356`) → `web/index.html` `loadModels`/`_fillModelDropdown`/`_modelOption` (3478/3449/3442) render
every id. The "empty content" string is **HiveMind's upstream error body**, surfaced by MS4 as the chat content;
MS4's own empty-content fallback chain is `face_lobe_chat.py:338-411` and re-runs the **same unfiltered picker**.
**Exact fix (the single highest-value change is step 3):**
1. Extend `_normalize_model_entry` to also capture `task`/`modality`/`capabilities` when present.
2. Add `_is_chat_capable(entry)`: False when metadata indicates tts/speech/audio/embedding/transcribe/asr/stt/
   whisper/image/moderation/rerank; with no metadata, fall back to an id-substring denylist (`-tts`,`tts-`,
   `whisper`,`embed`,`nomic-embed`,`bge-`,`rerank`,`moderation`,`parler`,`bark`,`xtts`,`piper`,`kokoro`);
   **fail-open** for unknown ids so genuine chat models are never dropped.
3. In **both** `_pick_from_catalog` bodies, after the `if e["id"]` filter add
   `normalized = [e for e in normalized if _is_chat_capable(e)]`.
4. Change bare `qwen3` → tag-scoped `qwen3:` at `model_picker.py:87` + `depth_picker.py:57`.
5. UI: filter `loadModels`/`_fillModelDropdown` by the same denylist; clear stale `localStorage`
   `ms4_model_id`/`ms4_depth_model_id` when no longer chat-capable. (Optionally filter server-side at
   `server.py:356` so every consumer benefits.)
6. Skip non-chat candidates in the `face_lobe_chat.py:338-411` fallback chain.
**Test:** `tests/ms4_gateway/test_model_picker.py` + `test_depth_picker.py`: catalog where the only loaded entry
is `Qwen3-TTS` + an installed `llama3.1:8b` → picker returns `llama3.1:8b`; Qwen3-TTS as ONLY loaded → falls
through, never selects it; bare `qwen3` no longer matches `Qwen3-TTS`; `_is_chat_capable("mystery-chat:13b")`
(no metadata) → True.
**Self-contained:** Yes.
**Caveat:** the exact HiveMind catalog field names for task/modality live in HiveMind, not this repo — write the
filter defensively (prefer metadata, fall back to the id denylist, which alone fixes the observed case).

### RANK 3 — TOOLBOX: the DEPTH lobe has zero `hivemind.*` tools registered (P0)
> This is the architectural item you flagged: MS4 is meant to have an efficient **toolbox** (curated tool
> selection), not "shotgun all defs into context." That system EXISTS and is good — see the toolbox section
> below — but it is **not wired into the depth lobe's Hermes catalog**.

**Symptom (logs):** depth jobs (qwen3-coder-next) said "a tool called 'run' that doesn't exist" and listed only
generic Hermes tools. The gpt-5.5 depth job confirmed it: its real toolset is `browser_*`, file, terminal/
process/execute_code, skills/memory/session, todo/clarify/delegate_task, media — **and ZERO `hivemind.*` / zero
quartermaster toolboxes**; it could only list hivemind tools "from the skill DOCS … could not verify live."
**Code / mechanism:** the DEPTH lobe is a real Hermes `AIAgent` (`hermes_runner._construct_agent` 319-360 →
`new_background_agent` 389-422 → `double_agent/worker.py:350-384` → `_worker_entry.py:106-123`). Its tool catalog
comes 100% from Hermes `model_tools.get_tool_definitions(enabled_toolsets=...)` (`hermes-agent/model_tools.py:337`),
which resolves only **registered Hermes toolsets**. The depth job runs with `enabled_toolsets=None` (full Hermes
catalog = generic built-ins). **Nothing ever registers the HiveMind MCP tools into Hermes' registry:** grep for
`register_tool`/`tools.registry` across `machine_spirit_4` = **zero**. The `ms4_consciousness` Hermes plugin
(`plugins/hermes/ms4_consciousness/__init__.py:17-22`) registers only **hooks**, never `ctx.register_tool()`.
The 1841-line typed wrappers in `gateway/hivemind_tools.py` are consumed only MS4-side, never surfaced to Hermes.
**Exact fix:** register the HiveMind tools as a Hermes `mcp-hivemind` toolset (Hermes treats `mcp-`-prefixed
toolsets as first-class — `hermes-agent/tools/registry.py:262-269`; binding via `PluginContext.register_tool`
`hermes-agent/hermes_cli/plugins.py:319`). In `ms4_consciousness/register(ctx)` (after the hook registrations),
loop over the HiveMind tools the depth lobe should reach (start **read-only**: `cluster_summary`, `hosts_list`,
`vm_list`, `capability_matrix`) and `ctx.register_tool(name="hivemind_cluster_summary", toolset="mcp-hivemind",
schema={...}, handler=lambda **a: hivemind_tools.cluster_summary(HIVEMIND_URL, **a))`, backed by the existing
`gateway/hivemind_tools.py` wrappers (they already do JSON-RPC + content unwrap + auth + PsyKyo signing). Use
identifier-safe names + a bidirectional map (`hivemind_cluster_summary ↔ hivemind.cluster.summary@v1`).
`HIVEMIND_URL` from `os.environ["MS4_HIVEMIND_URL"]`. The depth worker re-discovers/loads the plugin in-subprocess,
so registration runs before `get_tool_definitions` is read; ethics gating is automatic via the existing
`pre_tool_call` hook (`__init__.py:47`). To route through the toolbox, ensure
`quartermaster.hermes_toolsets_for_query` (`hermes_runner.py:611`) can include `mcp-hivemind` when
`MS4_QM_TRIM_DEPTH=1`.
**Test:** after `PluginManager.discover_and_load` loads `ms4_consciousness`, assert
`model_tools.get_tool_definitions(enabled_toolsets=None)` includes `hivemind_cluster_summary` and
`get_toolset_for_tool("hivemind_cluster_summary") == "mcp-hivemind"`; build via `new_background_agent` and assert
≥1 tool name starts with `hivemind_` (RED before, GREEN after); monkeypatch `hivemind_tools._call_tool` → assert
the handler routes to tool id `hivemind.cluster.summary@v1`.
**Self-contained:** No (spans MS4 plugin + the external hermes-agent registry contract; decide which tools to
expose, read-only first).
**⚠ ENVIRONMENT BLOCKER (must verify — see gpt-5.5 evidence):** job `da-6ee24adc` showed the depth worker's
shell could not resolve the Windows cwd (`C:\...\TMR` "does not exist from this shell, so I reran from /") and
got **Connection refused** on MS3 `:9080`, HiveMind MCP `:6089/v1/mcp`, MS4 MCP `:9181/mcp`. So the depth worker
runs in a **different filesystem + network namespace** (WSL/container/posix shell) where Windows-host localhost
services are unreachable. Registering the tools is necessary but **not sufficient** unless the in-process
handler (running `hivemind_tools.*` against `MS4_HIVEMIND_URL`) can actually reach HiveMind from the worker.
Verify the worker process's network reach to the HiveMind URL; if the worker is WSL/remote, point
`MS4_HIVEMIND_URL` at a host-reachable address (not `127.0.0.1`) or run the worker in the host namespace.
Also note the **wrong MCP endpoints** the depth model improvised against (`:6089/v1/mcp`, `:9181/mcp`) vs the real
`mcp_base_url = :6105` — once tools are registered, handlers use the typed wrappers (correct), but any docs/skill
hints pointing at `:6089/v1/mcp` should be corrected to `:6105`.

### RANK 4 — Conversational context dropped at dispatch + stale-context bleed (P1)
**Symptom (logs):** "run them both and tell me the output" dispatched goal `"run them both"` (the prior turn that
named the two tools was not attached) → depth: "I don't see which programs"; later a bare follow-up replayed an
**earlier** job's "missing 'run' tool" lecture (job `da-62888384`) — stale `_pending_deep_goal`.
**Code / mechanism:** the `JobEnvelope` is built from one bare utterance (`hermes_runner.py:616-629`,
`internal_goal=dispatch_intent`); `worker.py:373` hardcodes `conversation_history=[]`; `_pending_deep_goal`
stores bare `message_for_chat[:240]` (`hermes_runner.py:749`); `schemas.py` `JobEnvelope` has no context field.
**Exact fix:** (A) add optional `prior_context: list[dict]` to `JobEnvelope` (`schemas.py`, sanitized role+content,
clamped last N turns, round-trip + validate). (B) in `hermes_runner.chat` before building the envelope (~616),
pull the FACE lobe's recent turns for the session and set `prior_context` for **both** the deep-route and confirm
paths; optionally prepend a deterministic resolved-referent hint (scan last assistant turn for tool ids/quoted
names — no model call). (C) in `worker.py:373` replace `conversation_history=[]` with the supplied history.
(D) store the **resolved** goal (+context) in `_pending_deep_goal` at `:749`, not the bare string. (Per-job
isolation is already correct — unique `da-worker-{job_id}` session, fresh subprocess.)
**Test:** stub a prior assistant turn naming two tools → assert `envelope.prior_context` carries it (RED today);
fake chat runner records `conversation_history` kwarg → assert it's passed (empty today); topic-change then bare
"yes" → dispatched goal corresponds to the resolved context, not a referent-less verb.
**Self-contained:** No (cross-process schema change).

### RANK 5 — Over-eager deep dispatch on weak keyword matches (P1, quick win)
**Symptom:** trivial verbs route to DEPTH. **Code / mechanism:** `double_agent/router.py:309` matches
`_TOOL_INTENT_VERBS` (`:183`, bare `run`/`open`/`execute`) by **bare substring**, and `:316` returns deep on ANY
single hit — so "running back", "markets are open" trip it. The LLM safety net (`router.py:398-425`) is dead code
because `hermes_runner.py:542` passes no classifier.
**Exact fix:** (a) compile word-boundary patterns like the existing `_DIRECT_HINT_RES` (`router.py:230`,
`re.compile(r"\b"+re.escape(v)+r"\b")`) and match those at `:309`; (b) replace the single-hit short-circuit at
`:316` with a weighted score requiring ≥2 signals or verb+deep-noun to cross the 0.5 threshold (`:327`); (c) pass
a `classifier=` into `router_route` at `hermes_runner.py:542` (or remove the dead branch). **Sequence after
ranks 1 & 4** (it changes routing-to-deep, which interacts with the trust guard and context fix).
**Test:** "running back to the store" / "the markets are open" → direct; "run the cluster summary tool" / "open a
vm and start it" → deep; single incidental verb < 0.5, verb+deep-noun ≥ 0.5.
**Self-contained:** Yes.

### RANK 6 — Mis-attributed failure message: "switch TTS to REST" on a CHAT stall (P2, quick win)
**Symptom:** "I'm having trouble reaching the language cluster… switch the TTS engine to REST…" on turns whose
**chat** latency hit 12s (`chat:12093ms`/`14842ms`). **Code / mechanism:** `face_lobe_chat.py:522` (`_post_streaming`
urlopen timeout 12s) raises stream-unreachable; `hermes_runner.py:112-118` `_canned_chat_failure_text` returns the
switch-TTS-to-REST text — but REST is already `DEFAULT_ENGINE` (`voice.py:107`) and this is an **LLM stream** failure.
`voice.stream_stall_secs = 12` is the trip. **Exact fix:** in `hermes_runner.py:112-118` drop the TTS/REST clause;
say the language cluster's streaming chat endpoint is slow/unreachable and suggest retry or a lighter Face-lobe
model. (Optional: raise the 12s stall timeout — separate tuning call.) **Test:** `_canned_chat_failure_text` with a
`stream`+`timed out` error → asserts no TTS/REST mention, mentions LLM/cluster slowness + retry/lighter model.
**Self-contained:** Yes.

---

## The toolbox (`quartermaster/`) — what it is, and why it's bypassed
Your "efficient toolbox" intent is **real and implemented** in `gateway/quartermaster/` (3,137 LOC). Its own
docstring (`quartermaster/__init__.py:3-6`): *"Confidence-gated cascade (deterministic → embeddings → tiny-LLM,
fail-safe) that retrieves only relevant tools per request instead of dumping the full 75-tool MS4 / 154-tool
HiveMind catalog into the model's context."* Hierarchy = **tool shed → toolbox → tool** (`taxonomy.py`):
- `taxonomy.py` — `tool_domain`, `canonical_toolbox`, `TOOL_SHED_CLUSTERS`, `keywords_for_toolbox`, and
  `hermes_toolsets_for_query(query)` (the bridge that maps a query → the Hermes toolsets to enable).
- `index.py` — tiered retrieval (`resolve`, `query_tools`, `query_toolboxes`; TF-IDF + HiveMind backends).
- `cascade.py` — the deterministic→embeddings→tiny-LLM resolve cascade (`resolve`, `ResolvedTool`, tiers).
- `router.py` — policy layer: verdict `INLINE` / `DEPTH` / `NONE` (`decide`, `ToolRouter`).
- `catalog.py` — the tool catalog snapshot; `executor.py` — inline tool execution.
- Design doc: `docs/superpowers/specs/2026-05-30-quartermaster-tool-router-design.md`.

**Why it doesn't help the depth lobe today:** the toolbox can only select among tools that are *registered in
Hermes' tool registry* — and (RANK 3) the `hivemind.*` tools were never registered there. So
`hermes_toolsets_for_query` has no `mcp-hivemind` toolset to return, and the depth job (which usually runs
`enabled_toolsets=None` anyway — `hermes_runner.py:608`, only trimmed when `MS4_QM_TRIM_DEPTH=1`) falls back to the
full generic Hermes catalog. **Fix order:** RANK 3 first (register `mcp-hivemind`), then have
`hermes_toolsets_for_query` include it and enable `MS4_QM_TRIM_DEPTH` so the depth lobe gets the *curated* hivemind
subset per query — which is exactly your "no shotgunning" design.

---

## Suggested sequencing for Codex
1. **RANK 2** (model-picker chat-capability filter) — self-contained, unblocks "empty content" immediately.
2. **RANK 6** + **RANK 5** (canned-message wording; router word-boundary) — self-contained quick wins.
3. **RANK 1** (Face-lobe output-vs-grounding guard) — the trust fix; needs grounding threaded into `chat()`.
4. **RANK 3** (register `mcp-hivemind` toolset) — **verify the depth-worker network/namespace blocker first**.
5. **RANK 4** (JobEnvelope `prior_context`) — cross-process schema change.
Re-evaluate RANK 5 after 1 & 4 land (routing interacts with the trust guard + context fix).

## Two low-priority cleanups flagged in passing
- `README` priority list (~lines 924-925) is stale vs `FOREGROUND_PRIORITY_PATTERNS` (ordering + final-fallback model).
- `double_agent/depth_picker.py:28` imports `_score` but never uses it.

## Verification artifacts
- Verbatim logs + the gpt-5.5 ground-truth jobs: `docs/MS4_FAILURE_EVIDENCE_20260613.md`.
- All `file:line` verified against `C:\Users\nexus-hc-win-00\Downloads\TMR\machine_spirit_4` and
  `C:\Users\nexus-hc-win-00\Documents\hermes-agent`. No files were edited and nothing was executed during this
  investigation (read-only).
