# MS4 Double-Agent — Raw Evidence Appendix (2026-06-13)

Verbatim logs/observations captured for the Codex failure report. Code `file:line` root causes are
added by the `ms4-lobe-failures-rootcause` investigation; this file is the evidence appendix.

## A. Server defaults (MS4 Settings → "Server defaults (read-only)")
```
schema: Ms4Settings.v1
gateway:  http://127.0.0.1:9180
ms3:      http://127.0.0.1:9080
hivemind: http://127.0.0.1:6089
mcp_base_url (HiveMind cluster state): http://127.0.0.1:6105

voice.tts_engine_default = rest
voice.tts_voice_default  = alloy
voice.tts_model_default  = tts-1
voice.asr_ready          = {"ready":true,"detail":true}
voice.first_chunk_words  = 5
voice.stream_stall_secs  = 12
grounding.cache_ttl_secs = 60
grounding.cache_entries  = inventory::http://127.0.0.1:6089, tools::http://127.0.0.1:9080::http://127.0.0.1:6089
hermes.version           = 0.16.0
face_lobe.default_model  = qwen3-coder-next:latest   <-- CODER model as Face Lobe default
face_lobe.fallback_model = qwen3-coder-next:latest
face_lobe.override       = (none)
```
KEY: the Face Lobe default is a CODER model, the picker shows **Qwen3-TTS** (a TTS model) selected for
BOTH Face and Depth lobes, yet actual turns ran `llama3.1:8b` and `qwen2.5:0.5b`. The model-selection
layer is incoherent and admits non-chat models (TTS/embedding/transcribe/image) as chat/depth models.

## B. Owner design intent (must inform the fix)
"MS4 is supposed to have a very efficient **toolbox** system (NOT just shotgunning all tool defs directly
into context)." => the `quartermaster/` tool-router should select a CURATED, relevant tool subset per job.
Bug #2 (depth lobe can't reach `hivemind.*`) is a toolbox/quartermaster mis-routing: the depth lobe gets
generic Hermes tools (browser_*, read_file, terminal, execute_code, delegate_task) and ZERO `hivemind.*`
MCP tools, while the Face lobe has `hivemind-mcp-live-context`. Manual overrides exist: `/deep <msg>`
forces a Depth dispatch, `/direct` keeps foreground-only.

## C. Voice diagnostics (per-turn structured log; newest first)
```
2026-06-13 07:01:28  face=qwen2.5:0.5b  total 4046ms  router:direct  grounding:face_lobe
  user: "Please say, HD04, OK."
  reply: "Looking into that — I'll surface the result on the next turn." (61c)   <-- non-answer canned filler

2026-06-13 06:56:32  face=qwen2.5:0.5b  total 4250ms  router:direct  grounding:face_lobe
  reply: "Looking into that — I'll surface the result on the next turn."

2026-06-13 06:42:53 / 06:48:39 / 06:49:27  ERROR
  voice_input_ready=false  asr.status=unhealthy  detail=No endpoints configured   <-- ASR unprovisioned

2026-06-13 00:01:20  face=llama3.1:8b  total 20219ms  chat:12093ms  router:direct  grounding:face_lobe
  user: "Okay, what about now?"
  reply: "I'm having trouble reaching the language cluster right now..." (196c)   <-- stall (stream_stall_secs=12)

2026-06-13 00:00:38  face=llama3.1:8b  total 25375ms  chat:14842ms  router:deep  dispatched da-b852e98e
  grounding: hivemind-mcp-live-context+face_lobe+depth_lobe_dispatched
  user: "Yeah, give me the full list, please. Also, are there any AMD GPUs?"
  reply: "I'm having trouble reaching the language cluster right now..."

2026-06-12 23:59:43  face=llama3.1:8b  total 21671ms  chat:5405ms  router:deep  dispatched da-04d99896
  grounding: face_lobe+depth_lobe_dispatched
  reply: "According to hivemind.hosts.list@v1, there are 7 GPU devices..."   <-- says 7 GPUs
```
NOTE inconsistency: this turn says **7 GPU devices**; later face-lobe turns say **6 GPUs**; and
`hivemind.cluster.summary@v1` says 6 GPUs / 4 nodes while `hivemind.hosts.list@v1` returns 3 node records.
The two MCP tools disagree and the count drifts across turns.

## D. Deep-job result evidence (from the chat transcript, revisions 11→30)
- Jobs `da-61db5857`, `da-04d99896`, `da-b852e98e`, `da-553555b7`: result =
  "⚠️ No reply: the model returned empty content after retries and any fallback providers."
- Job `da-1599d387` (List all tools): "a tool called 'run' that doesn't exist... available tools include
  browser_*, read_file, write_file, terminal, todo, delegate_task, vision_analyze, ... " (generic Hermes
  tools; NO hivemind.* tools).
- Job `da-7bbf1888` (run them both): "I don't see which programs or commands you'd like me to execute."
  (the antecedent "them" = the two hivemind.* tools was NOT passed into the job.)
- Job `da-62888384` (verify GPUs via deep lobe): "I understand you're seeing an error about a missing
  'run' tool... use execute_code / terminal / browser tools / read_file." (STALE context from the earlier
  'run' job bled into a new job; still no hivemind.* tools.)
- FABRICATION (most serious): user "Results?" → Face Lobe (grounding=face_lobe, NO tool payload) emitted
  invented JSON: `{"total_nodes":4,"active_nodes":4}` and `[{"node_id":"node-123","gpu_count":2},
  {"node_id":"node-456","gpu_count":3}]` — presented as the deep job's output. The real job `da-7bbf1888`
  had produced nothing. On "can you list the GPUs?" it claimed "The Depth Lobe job da-7bbf1888 already
  listed the GPUs" though that job produced no such list (mis-attribution).

## E. Cluster / backend state at report time
```
hivemind.jobs.active@v1   total_active=4 (4 scatter: functiongemma:latest 1/4, qwen2.5:0.5b 1/4)
hivemind.cluster.load@v1  all zeros (amplification 0, shed 0)
auth_configured           = no
Voice services            ASR :49170 healthy, TTS :49169 healthy, TTS_SUPER :49166 healthy (now)
storage (6103)            Backend unavailable
training/forge (6115)     menta_forge not running (503)
oracle                    readiness=degraded, provision=diagnostic_required
```

## E2. DEPTH-LOBE GROUND TRUTH (gpt-5.5 jobs, rev 45 + 47) — most authoritative evidence
When the depth lobe ran on a capable model (gpt-5.5) instead of a tiny one, it reported the REAL state:

Job da-ed9b9b05 (goal "can you list tools? Toolboxes? MCP tools?"), depth=gpt-5.5:
- Attempted MCP `tools/list` against HiveMind MCP `http://127.0.0.1:6089/v1/mcp` AND MS4 MCP
  `http://127.0.0.1:9181/mcp` -> BOTH "Connection refused". (NOTE: real MCP base per settings is :6105,
  not :6089/v1/mcp -> WRONG ENDPOINT + unreachable.)
- Enumerated the toolsets it ACTUALLY has (all generic Hermes, ZERO hivemind.* / ZERO quartermaster
  toolboxes): browser_*, file (read_file/write_file/patch/search_files), terminal/process/execute_code,
  skills/memory/session, todo/clarify/delegate_task, media (vision_analyze/video_analyze/image_generate/
  text_to_speech). Delegated-subagent toolsets: browser, computer_use, cronjob, discord, file,
  homeassistant, image_gen, kanban, search, terminal, todo, tts, video, web, x_search, yuanbao.
- Listed hivemind.* tools ONLY as "Known MCP tools from the HiveMind skill DOCS ... could not verify live
  because the endpoint is refusing connections": hivemind.cluster.summary@v1, hosts.list, services.list,
  models.list, service_health, jobs.list, jobs.cancel, services.restart/enable/disable, time.now.

Job da-6ee24adc (goal "run the tool"), depth=gpt-5.5 -- THE KEY EVIDENCE:
```
I attempted to run the MCP tool under the MS4/Sister identity context.
Result: blocked by unavailable local services. Real execution output showed:
- MS3 identity verify at 127.0.0.1:9080/identity/verify : Connection refused
- MS3 ethics evaluate  at 127.0.0.1:9080/ethics/evaluate : Connection refused
- HiveMind MCP tool call at 127.0.0.1:6089/v1/mcp for hivemind.time.now@v1 : Connection refused
There was also an environment issue: the configured working directory
C:\Users\nexus-hc-win-00\Downloads\TMR / /c/Users/nexus-hc-win-00/Downloads/TMR does NOT EXIST from
this shell, so I reran from / ; the network services were still unreachable.
```
=> ROOT INSIGHT: the depth-lobe Hermes worker executes in a DIFFERENT filesystem + network namespace
   (the Windows TMR path doesn't exist for it; Windows-host localhost services :9080/:6105/:6089/:9181
   are not reachable from it). So even with a great model + correct endpoints it cannot reach MS3 or the
   HiveMind/MS4 MCP. This is the deepest cause of "depth lobe can't use the tools": a sandbox/namespace
   isolation + endpoint-config problem, on top of the toolbox-not-feeding-hivemind-tools problem.

## E3. FABRICATION IS RAMPANT (rev 47, additional instances)
- "weather" -> Face lobe invented `{"current_weather":{"temperature":22°C,"humidity":60%,...}}` and
  attributed it to job da-62888384 (which was actually about GPUs / a 'run' tool error). Pure invention +
  mis-attribution. grounding=face_lobe (no tool payload).
- "more" -> Face lobe invented a JSON tool list attributed to da-7bbf1888 (which produced no such list).
- "list all tools" -> invented a 6-item tool list as if from a completed job.
- da-76dcd6fa "list tools/toolboxes/MCP" -> depth (qwen3-coder-next) returned "I've reviewed your task
  list. All items are completed: 1. List all available tools — Completed" (it answered with the TODO tool,
  not a tool listing -- wrong-toolset dispatch).
- Model routing chaos: depth jobs ran on qwen3-coder-next AND gpt-5.5 (cloud) across the session; face on
  llama3.1:8b; picker still shows Qwen3-TTS selected. Only gpt-5.5 produced truthful tool output.

## F. MS3 identity (continuity context)
```
identity.name=Claude  chosen=Sister  session#=76  confirmed=true
heartbeat: running, 60s interval, count=450, errors=0
```
