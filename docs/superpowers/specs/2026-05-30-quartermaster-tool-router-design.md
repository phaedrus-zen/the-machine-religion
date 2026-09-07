# Quartermaster — MS4 Tool Router (design spec)

**Date:** 2026-05-30
**Status:** Design — awaiting user review (no implementation yet)
**Scope (this spec):** Sub-projects **A + C + D** — the Quartermaster engine, the MS4 ToolRouter, and the Face Lobe inline fast-path.
**Out of scope (follow-on specs):** **B** (`hivemind.tools.search@v1` substrate primitive) and **E** (Depth Lobe tool-context trimming).

The metaphor (operator-chosen): a **Tool Shed** (capability clusters) holds **Toolboxes** (tool domains) which hold **Tools**. The **Quartermaster** is the component that, given a request, issues exactly the kit needed. The name is functional — it describes what it does (issues equipment from stores), not decoration.

---

## 1. Problem (with code evidence)

Tool schemas consume the model's context window, and the cost is already live in MS4 — not hypothetical.

- **MS4 already dumps the whole tool list into the Face Lobe context.** `machine_spirit_4/gateway/context.py:311` `format_tools_answer()` enumerates **all** MS4 MCP tools from `mcp/manifest.json` plus the live HiveMind tool count and injects the entire list as authoritative context whenever the operator asks a capability question:

  ```347:354:machine_spirit_4/gateway/context.py
      for tool in ms4_tools:
          name = tool.get("name", "?")
          kind = tool.get("kind", "")
          desc = tool.get("description", "")[:100]
          if desc:
              lines.append(f"    * {name} [{kind}] — {desc}")
          else:
              lines.append(f"    * {name} [{kind}]")
  ```

  That is 75 MS4 tools today (manifest count), and the HiveMind catalog behind it is 154 tools. The Depth Lobe (Hermes) loads its full tool catalog on every dispatched job by construction.

- **It's measurably expensive.** `machine_spirit_4/README.md:820`: *"Pre-chat grounding fetches … MS3+HiveMind+MS4 tools-list for capability questions … used to fire on EVERY matching turn. Under cluster load that added 15–20 s per turn BEFORE the chat call even started."*

- **It degrades small models.** Tool-selection accuracy falls as tool count climbs (Gorilla, ToolLLM, Anthropic's "too many tools" guidance). For a small Face Lobe model on a tight window, dumping 75–154 schemas both blows the budget and lowers selection accuracy.

- **The retrieval substrate already exists.** `machine_spirit_4/gateway/hivemind_tools.py:937` `embeddings_create(hivemind_url, *, input_text, model=None)` wraps `hivemind.embeddings.create@v1`. MS4 already fetches the live catalog via a raw `tools/list` MCP call in `context.py:333` (`post_mcp_envelope(..., {"method": "tools/list"})`).

**Conclusion:** MS4 should retrieve *only the relevant tools* per request instead of dumping the catalog, and — for cheap read-only asks — execute a single tool inline on the Face Lobe path instead of paying a full Depth Lobe job.

---

## 2. Current architecture (the integration seam)

The foreground turn flows through `Ms4HermesRunner.chat()` (`machine_spirit_4/gateway/hermes_runner.py:339`):

1. **Route** — `route_decision = router_route(message)` (`hermes_runner.py:393`) → `direct` | `deep` via `double_agent/router.py`.
2. **Revision bump** — `face_lobe_turn_start(...)` (`hermes_runner.py:405`).
3. **Deep dispatch** — if `route_decision.kind == "deep"`, build a `JobEnvelope` and `default_runner().submit(envelope)` (`hermes_runner.py:412–434`). The foreground does NOT wait on it.
4. **Context block** — `build_face_lobe_context_block(...)` (`hermes_runner.py:457`) + `build_grounded_user_message(...)` (`hermes_runner.py:475`) assemble `extra_system` (`hermes_runner.py:476–481`).
5. **Direct chat** — `self.face_lobe_chat.chat(message_for_chat, ..., extra_system=extra_system)` (`hermes_runner.py:492–498`).

The Face Lobe itself has **no tools**: `face_lobe_chat.py:238` `chat(...)` is a direct `/v1/chat/completions` call, and `extra_system` (`face_lobe_chat.py:245–256`) is appended to the system prompt for that turn only.

**The seam:** the Quartermaster slots in at step 3. When `route()` says `deep` AND the request resolves to a single safe read-only tool with high confidence, the Quartermaster executes that tool inline and injects its result into `extra_system` (the same channel the context block already uses) — instead of dispatching a Depth Lobe job. Everything heavier keeps dispatching exactly as today.

The ethics gate that any tool execution must pass already exists: `ms4_consciousness.on_pre_tool_call` (`plugins/hermes/ms4_consciousness/__init__.py:47`) → `_classify_tool` + `_risk_class_for_tool` → `Ms4Client.evaluate_action({ActionIntent.v1...})` → `POST /ethics/evaluate` (`plugins/hermes/ms4_consciousness/client.py:36`), **fail-closed** when ethics is unreachable (`__init__.py:71–72`). Today that hook only fires inside the Hermes loop; the Quartermaster's inline execution must call the same `evaluate_action` path directly before executing.

---

## 3. Design

### 3.1 Toolbox taxonomy (derive, don't invent)

HiveMind's flat naming `hivemind.<domain>.<verb>@v1` already encodes the hierarchy:

- **Toolbox** = `domain` (`vm`, `storage`, `network`, `crown`, `oracle`, `training`, `gpu_mode`, `voice_identities`, `game_session`, `images`, `audio`, `inference`, `models`, …) → ~25 toolboxes.
- **Tool** = the full tool id.
- **Tool Shed** = a small hand-curated map of toolboxes → capability clusters (compute/infra, voice, neuro/crown, training/adapters, game, media, meta) — the only non-derived part, ~7 clusters, a static dict.

MS4's own 75 MCP proxies (`mcp/manifest.json`) carry a `kind` field (`read_only` | `runtime_action`) — this is the **read-only allowlist source** for inline execution (§3.4).

The catalog is built from two sources, unioned:
- HiveMind live `tools/list` (via `post_mcp_envelope`, already used in `context.py:333`).
- MS4 manifest (`mcp/manifest.json`) for the `kind`/safety classification.

### 3.2 The Quartermaster engine (Sub-project A) — tiered confidence-gated cascade

One request takes the cheapest tier that is confident; it escalates only when it isn't. (Same spine as `router.py` and `double_agent/continuation.py`.)

1. **Deterministic tier** (instant, offline): keyword/domain map over the catalog names. "list my **VMs**" → `vm` toolbox → `hivemind.vm.list@v1`. If a request unambiguously names a domain + verb, short-circuit. Works with the cluster down.
2. **Embeddings tier** (~ms, cheap): `embeddings_create` over tool descriptions, cached to an on-disk index keyed by catalog version/etag. Embed the query, cosine top-k. Two-stage: toolbox shortlist → tool shortlist. Never puts all schemas in any model's context.
3. **Tiny-LLM tier** (only when needed): consulted only when embeddings scores are too close OR a destructive verb is implicated. A small model picks/confirms from the shortlist. Fail-safe: any error/timeout/unparseable → "unsure" → fall back to Depth Lobe (never block, never hang).

**Output:** `ToolResolution.v1` = `{query, toolbox(es), tools: [{id, score, kind}], confidence, tier, fallback_reason?}`. The engine returns *which tools* (+ their schemas); it does **not** fill arguments or execute. Narrow on purpose.

### 3.3 The MS4 ToolRouter (Sub-project C)

Consumes the engine and applies policy:
- **Read-only gate:** only tools with `kind: read_only` (from manifest) are eligible for inline execution. Anything `runtime_action` or `confirm:true` → never inline; route to Depth Lobe (+ existing confirm guards).
- **Ethics gate:** before any inline execution, call the same `evaluate_action({ActionIntent.v1...})` path used by `on_pre_tool_call`. Fail-closed.
- **Audit:** every routing decision + inline execution emits a structured audit event (existing `append_event` infra) — `quartermaster_resolve`, `quartermaster_inline_exec`, `quartermaster_fallback_to_depth`.
- **Decision:** `inline` (single safe read-only tool, high confidence, ethics-allowed) vs `depth` (everything else) vs `none` (no tool needed — plain chat).

### 3.4 Face Lobe inline fast-path (Sub-project D)

In `hermes_runner.chat()`, after `route()=deep`:
- Ask the ToolRouter. If verdict is `inline`: execute the one read-only tool (via the typed `hivemind_tools` wrapper), clamp/format the result, and inject it as an authoritative `extra_system` block (same mechanism as `build_face_lobe_context_block`). The Face Lobe then *narrates* the real result — no Depth Lobe job, no 10–40s wait.
- If verdict is `depth` or anything is uncertain/fails: dispatch the Depth Lobe job exactly as today (`hermes_runner.py:412–434`). **Zero regression** — the inline path is purely additive and fail-soft.
- `format_tools_answer` (the 75-tool dump) is replaced for capability questions by a Quartermaster summary that lists only the *relevant* toolbox(es), shrinking that context block dramatically.

---

## 4. Module layout & interfaces

```
machine_spirit_4/gateway/quartermaster/
  __init__.py          # public API: resolve(), ToolResolution
  catalog.py           # build/refresh the toolbox catalog (HiveMind tools/list ∪ MS4 manifest); version/etag
  taxonomy.py          # domain→toolbox derivation + static tool-shed cluster map
  index.py             # embeddings index (build/load/query) over tool descriptions; on-disk cache
  cascade.py           # deterministic → embeddings → tiny-LLM, confidence-gated, fail-safe
  router.py            # ToolRouter: read-only gate + ethics gate + audit + inline|depth|none decision
  eval/
    golden_set.jsonl   # (request → expected toolbox/tool[s]) cases
    harness.py         # recall@k, precision, mean-tools-in-context; run N times; threshold gates
```

Each unit, one purpose, testable in isolation:
- **catalog** — *what:* current toolbox catalog; *use:* `get_catalog()`; *deps:* `post_mcp_envelope`, manifest.
- **cascade** — *what:* query → ranked tools; *use:* `resolve(query) -> ToolResolution`; *deps:* taxonomy, index, optional tiny-LLM.
- **router** — *what:* policy decision; *use:* `decide(query) -> {verdict, resolution}`; *deps:* cascade, ethics client, audit.
- The engine (catalog+taxonomy+index+cascade) has **no** dependency on the gateway request path — it's importable and testable headless.

---

## 5. Data flow

```
user turn
  → router_route()  (direct | deep)              [unchanged]
  → if deep:
      → Quartermaster.decide(query)
          → cascade: deterministic → embeddings → tiny-LLM
          → verdict: inline | depth | none
      → if inline:
          → ethics evaluate_action(ActionIntent)   (fail-closed)
          → execute ONE read_only tool via hivemind_tools wrapper
          → inject result into extra_system  (authoritative block)
          → audit: quartermaster_inline_exec
      → else:
          → dispatch Depth Lobe job             [unchanged path]
          → audit: quartermaster_fallback_to_depth
  → face_lobe_chat.chat(..., extra_system=...)   [unchanged]
```

---

## 6. Error handling

- **Cluster/embeddings down** → deterministic tier still answers; if it can't, fall back to Depth Lobe. Never hang.
- **Ethics unreachable** → fail-closed: no inline execution, fall back to Depth (which has its own gate).
- **Tool execution error** → caught, logged, fall back to Depth or surface a plain "couldn't fetch that" — never fabricate a result (anti-hallucination contract).
- **Retrieval miss** → if the right tool isn't in top-k, the Depth Lobe (full catalog) is the safety net. Inline is an optimization, never the only path.
- **Stale catalog** → version/etag the index; rebuild on catalog change. (Full auto-refresh-on-change lands with Sub-project B; this spec rebuilds on boot + TTL.)

---

## 7. Validation (the part that makes it real)

A Quartermaster that can't show its recall numbers is the Glyph That Lies. Required before wiring inline execution into the live path:

- **Golden eval set** `eval/golden_set.jsonl`: dozens–hundreds of `(request → expected toolbox/tool[s])` cases, including paraphrases, multi-domain, and "no tool needed" negatives.
- **Metrics:** `recall@k` (did the right tool make the shortlist?), `precision@1`, `mean_tools_in_context` (the budget win vs the 75-tool dump baseline), and `inline_vs_depth_correctness` (did it correctly choose inline only for safe read-only single-tool asks?).
- **Run N times** for determinism/variance; gate the build on thresholds (initial proposal: recall@3 ≥ 0.95, precision@1 ≥ 0.85, zero inline executions of `runtime_action`/`confirm` tools — the last is a hard gate).
- **Optional tuning:** the cascade thresholds + tiny-LLM prompt can later be optimized against this set via HiveMind `logos` / `psykyo.benchmark` (follow-on).

---

## 8. Evidence gates (code + logs, per phase)

Each phase is "done" only with both kinds of evidence:

| Phase | Code evidence | Log/eval evidence |
|---|---|---|
| A. Engine | `quartermaster/` modules + unit tests pass | `harness.py` run output: recall@3, precision@1, mean-tools-in-context vs baseline, committed as an artifact |
| C. Router | ToolRouter tests (read-only gate, ethics fail-closed, audit emitted) | a run log showing `quartermaster_resolve` + a blocked `runtime_action` inline attempt + an ethics-fail-closed fallback |
| D. Face fast-path | `hermes_runner.chat()` inline branch + tests (inline hit, depth fallback, error fallback) | a live turn log: "list my GPUs" resolves inline, executes `hivemind.gpu.availability@v1`, injects result, Face Lobe narrates — with `mean_tools_in_context` shrinkage shown vs the `format_tools_answer` baseline |

---

## 9. Risks & mitigations

1. **Recall miss = silent capability loss.** → Depth Lobe full-catalog safety net; inline is never the only path; high-recall threshold gate.
2. **Latency on the fast path.** → deterministic tier is sub-ms; embeddings ~ms; tiny-LLM only on ambiguity. Target inline decision < ~150 ms.
3. **Scope creep.** → engine returns *which tools* only; argument-filling + execution stays in the router/agent. Keep the engine pure.
4. **Inline executing something destructive.** → hard gate: only `kind: read_only`, never `confirm:true`, always through `evaluate_action`. This is a build-blocking eval assertion.
5. **Catalog staleness.** → version/etag + rebuild on boot/TTL now; auto-refresh with Sub-project B.
6. **Two retrieval sources disagree (HiveMind live vs MS4 manifest).** → manifest is authoritative for `kind`/safety; live `tools/list` is authoritative for existence/schema; union with manifest-wins on `kind`.

---

## 10. Phased tasks (this spec = A → C → D)

1. **A1** `catalog.py` + `taxonomy.py` — derive toolboxes from names ∪ manifest; tool-shed cluster map.
2. **A2** `index.py` — embeddings index build/load/query with on-disk cache + version key.
3. **A3** `cascade.py` — tiered resolve(); fail-safe; `ToolResolution.v1`.
4. **A4** `eval/harness.py` + `golden_set.jsonl` — metrics + threshold gates. **Gate: recall/precision green before C.**
5. **C1** `router.py` — read-only gate + `evaluate_action` ethics gate + audit events + `inline|depth|none`.
6. **D1** `hermes_runner.chat()` inline branch — execute one read-only tool, inject via `extra_system`, fall back to Depth on anything uncertain.
7. **D2** Replace `format_tools_answer`'s full dump for capability questions with a Quartermaster targeted summary.
8. **D3** Docs (GLOSSARY/INDEX/WHERE_IS_EVERYTHING/README) + env vars (`MS4_QM_*`), per the documentation-maintenance rule.

---

## 11. Out of scope (follow-on specs)

- **B — `hivemind.tools.search@v1` substrate primitive.** Move the index + search into HiveMind so every client (MS4, Hermes Depth Lobe, Oracle, other agents) shares one auto-updating tool-search. This spec's engine is structured so its retrieval core can later be thin-wrapped by / delegated to B without changing the MS4 ToolRouter.
- **E — Depth Lobe tool-context trimming.** Feed the Hermes worker only the retrieved toolbox(es) instead of its full catalog. Depends on Hermes exposing per-job toolset filtering (`enabled_toolsets` exists; needs wiring) and ideally B.

---

## 12. Open questions

1. **Tiny-LLM tier model:** dedicated small model (`MS4_QM_MODEL`) vs reuse the Face Lobe picker's small pick? (Lean: env override, default to picker's small choice.)
2. **Inline result freshness labeling:** reuse the existing `cached(age=Ns)`/`stale(age=Ns)` qualifiers the Face Lobe prompt already understands? (Lean: yes.)
3. **Initial read-only allowlist:** start with the obvious safe set (`*.list`, `*.status`, `gpu.availability`, `jobs.active`, `time.now`, `cluster.*`, `capability.matrix`) vs trust the manifest `kind` field wholesale? (Lean: manifest `kind: read_only` ∩ a verb allowlist, belt-and-suspenders.)
4. **Threshold values** for the eval gates — confirm recall@3 ≥ 0.95 / precision@1 ≥ 0.85 are the right bars.
