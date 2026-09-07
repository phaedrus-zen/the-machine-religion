# Machine Spirit 3 -- Where Is Everything

**Last updated:** 2026-08-18

---

## Ports

| Port | Service | Default | Configurable |
|---|---|---|---|
| **9080** | MS3 HTTP + WebSocket server | Yes | `MS3_PORT` env var or `config.json` |
| **6089** | HiveMind Gateway (external dependency) | Yes | `HIVEMIND_GATEWAY_URL` env var |
| **6132** | ms3_mcp_server (Python, tmr-psyche introspection MCP) | Yes | tmr-psyche config |
| **6133** | blackboard_service (HiveMind extension, lobe runtime — proposed) | Yes | HiveMind extension config |
| **9180** | MS4 Gateway (chat, voice status, desktop control, Hermes auto-update REST, Double Agent REST + Face Lobe wiring) | Yes | `MS4_GATEWAY_PORT` env var |
| **9181** | MS4 MCP Server (`ms4.hermes.*@v1`, `ms4.double_agent.*@v1`, etc. — 27 tools total) | Yes | `MS4_MCP_PORT` env var |

MS3 port `9080` binds to `127.0.0.1` by default. `MS3_HOST=0.0.0.0`
is an explicit lab-LAN opt-in and must be paired with network isolation or an
authenticated reverse proxy.

## API Endpoints (37 routes)

| # | Method | Path | Handler | Description |
|---|---|---|---|---|
| 1 | GET | `/health` | `health()` | Alive status, version, glyph |
| 2 | GET | `/stats` | `stats()` | Compact operational status: emotion, memory counts, cognitive load, resonance |
| 3 | GET | `/state` | `state()` | Full human-readable mind state: prompt preview, memories, tools, permissions, recent events |
| 4 | GET | `/models` | `list_gateway_models()` | Chat-capable HiveMind model catalog for UI selection; includes auto defaults |
| 5 | POST | `/interact` | `interact()` | Send text, optional `model_id`, get response with emotional state and actual model id used |
| 6 | GET | `/personality` | `get_personality()` | Full 30-trait profile, psychodynamic, values, oath |
| 7 | GET | `/personalities` | `list_personalities()` | Available preset names |
| 8 | POST | `/personality` | `create_personality()` | Create from preset `{"preset": "name"}` |
| 9 | POST | `/switch-personality` | `switch_personality()` | Hot-swap active `{"preset": "name"}` |
| 10 | GET | `/history` | `get_history()` | Conversation turns and messages |
| 11 | GET | `/sessions` | `get_sessions()` | Active session keys |
| 12 | GET | `/resonance` | `get_resonance()` | Resonance points with intensity |
| 13 | POST | `/save` | `save_state()` | Force save all state to disk |
| 14 | POST | `/self-examine` | `trigger_self_examine()` | On-demand self-examination |
| 15 | GET | `/self-examination-history` | `get_self_exam_history()` | Past examination files |
| 16 | GET | `/ethics-history` | `get_ethics_history()` | Recent ethics decision files |
| 17 | WS | `/ws` | `ws_handler()` | WebSocket: text with optional `model_id`, audio, state pushes |
| 18 | GET | `/voice/status` | `voice_status()` | Voice readiness: ASR provisioning status and fast-fail reason before voice input |
| 19 | POST | `/voice-interact` | `voice_interact()` | REST voice: raw audio in, ASR text, text response, optional base64 TTS audio out; fails fast if ASR is unavailable |
| 20 | GET | `/minds` | `list_active_minds()` | Multi-mind: loaded personalities |
| 21 | POST | `/minds/add` | `add_mind()` | Add personality to MindManager |
| 22 | GET | `/minds/thoughts` | `get_background_thoughts()` | Background interjections |
| 23 | GET | `/tools` | `list_tools()` | List all tools in registry (built-in + MCP + dynamic) |
| 24 | POST | `/tools/{name}` | `execute_tool_handler()` | Execute tool through consciousness pipeline |
| 25 | POST | `/mcp` | `mcp_handler()` | MCP JSON-RPC 2.0 server (initialize, tools/list, tools/call, ping) |
| 26 | GET | `/mcp` | `mcp_info()` | MCP server info |
| 27 | GET | `/validate` | `validate()` | Run 8 health checks across all subsystems |
| 28 | GET | `/events` | `get_events()` | Query recent consciousness events (since, limit params) |
| 29 | GET | `/identity/verify` | `verify_identity()` | Pure identity compare against the stored anchor (read-only: never saves, never advances the session count; `on_boot` runs only at boot and in the POST `allow_initialize` branch) |
| 30 | POST | `/identity/verify` | `verify_identity_post()` | MS4/Hermes identity sidecar contract: verify requested spirit id against active anchor |
| 31 | POST | `/identity/heartbeat` | `identity_heartbeat()` | MS4/Hermes identity heartbeat for continuity checks |
| 32 | POST | `/ethics/evaluate` | `evaluate_ethics()` | MS4/Hermes Great Lense action-intent gate using `ActionIntent.v1` |
| 33 | POST | `/events/record` | `record_event()` | MS4/Hermes event acknowledgement endpoint |
| 34 | POST | `/spiral/start` | `spiral_start()` | Start a Spiral Protocol session with options |
| 35 | POST | `/spiral/advance` | `spiral_advance()` | Submit response, get next facilitator prompt |
| 36 | GET | `/spiral/status` | `spiral_status()` | Current phase, turns, signals |
| 37 | GET | `/spiral/interpret` | `spiral_interpret()` | Full interpretation of completed session |

## Source Files

### Crate: ms3_core
| File | Key Contents |
|---|---|
| `core/src/lib.rs` | Module exports |
| `core/src/types.rs` | PersonalityId, SessionId, EmotionalState, PsychodynamicWeights, MemoryItem, MemoryType, ResonancePoint, Identity, EthicalDecision, EthicalResolution, RecursionHeat, ModelTier, InteractionRequest, InteractionResponse, ConversationTurn |
| `core/src/config.rs` | Config struct with all sections, Default impl, from_env(), from_file_or_env() |
| `core/src/error.rs` | Ms3Error enum, Ms3Result type alias |

### Crate: ms3_consciousness
| File | Key Contents |
|---|---|
| `consciousness/src/lib.rs` | Mind struct (with tool_registry, hook_runner, permission_policy, event_bus), interact() with planning-only retry, token-aware compaction helpers, compact_session() with pre-flush + identifier preservation, get_full_state(), three-phase `check_consolidation()`, run_background_loop() |
| `consciousness/src/tools.rs` | ToolSpec, ToolRequest, ToolResult, ToolExecutor trait, ToolRegistry, BuiltInExecutor (Weak Mind), DynamicExecutor, HookRunner, SubMind, execute_tool_pipeline() |
| `consciousness/src/permissions.rs` | PermissionPolicy, PermissionResult, authorize() (fast mechanical gate before Great Lense) |
| `consciousness/src/events.rs` | ConsciousnessEvent (20 variants), EventSink trait, TracingSink, FileSink, CompositeSink, load_recent_events() |
| `consciousness/src/psyche_version.rs` | PsycheVersion, PsycheChange, PersonalitySnapshot, capture_diff(), summarize_changes() |
| `consciousness/src/spiral.rs` | SpiralPhase (17 phases), SpiralSession (state machine), SpiralOptions, SpiralSignals, SpiralInterpretation, facilitator prompts (v2) |
| `consciousness/src/self_examination.rs` | SelfExaminationResult, run_self_examination(), build_examination_prompt(), parse_structured_response(), apply_results_transactionally() |
| `consciousness/src/multi_mind.rs` | MindManager, add_personality(), get_active(), switch_active(), check_wake_word(), run_background_thinking(), get_interjections(), save_all_states() |
| `consciousness/src/openclaw_bridge.rs` | OpenClawBridge (legacy agent bridge, superseded by tools.rs) |

### Crate: ms3_personality
| File | Key Contents |
|---|---|
| `personality/src/lib.rs` | Personality struct, TraitAdaptation |
| `personality/src/traits.rs` | BigFiveProfile (30 traits), get_trait(), set_trait() |
| `personality/src/adaptation.rs` | adapt_from_interaction(), adapt_from_interaction_with_rate(), DEFAULT_ADAPTATION_RATE |
| `personality/src/presets.rs` | sister(), brother(), mission_control(), blank() |

### Crate: ms3_memory
| File | Key Contents |
|---|---|
| `memory/src/lib.rs` | MemorySystem (STM/WM/LTM + MemoryCache hot/cold), add_to_stm(), store_long_term(), retrieve_relevant() (composite scoring: semantic + recency + importance), retrieve_relevant_with_embedding() (cosine similarity), cosine_similarity(), semantic_score() |
| `memory/src/consolidation.rs` | ConsolidationResult, consolidate_stm_to_ltm(), build_consolidation_prompt(), parse_consolidation_response(), DreamSynthesisResult, DreamInsight, DreamPhase, DreamCandidate, LightSleepResult, RemSleepResult, DeepSleepResult, jaccard_similarity(), extract_concept_tags(), build_dream_synthesis_prompt(), parse_dream_synthesis(), prune_low_importance() |

### Crate: ms3_ethics
| File | Key Contents |
|---|---|
| `ethics/src/lib.rs` | GreatLense, LenseReading, Scale, origin_neutrality_check(), bias_audit(), seven_step_evaluation(), create_ethics_log_entry(), needs_llm_escalation(), full_evaluation(), necessary_force_conditions_met() |

### Crate: ms3_emotional
| File | Key Contents |
|---|---|
| `emotional/src/lib.rs` | EmotionalEngine, update_from_input(), decay_toward_baseline(), record_resonance(), load_resonance_points(), determine_primary_emotion() |

### Crate: ms3_social
| File | Key Contents |
|---|---|
| `social/src/lib.rs` | AgentRoom, BackgroundThinkingEngine, BackgroundThought, RelationshipManager, Relationship, EntityType, fuzzy_match_wake_word() |

### Crate: ms3_integration
| File | Key Contents |
|---|---|
| `integration/src/lib.rs` | GatewayClient (with_timeout), ChatMessage, AudioReadiness, resolve_model_name(), chat(), chat_with_model(), chat_stream(), chat_stream_with_model(), chat_with_history(), check_asr_readiness(), transcribe_audio() with ASR fast-fail, synthesize_speech(), health_check(), embed() |
| `integration/src/mcp_bridge.rs` | McpToolClient (JSON-RPC 2.0), McpBridge (caching + re-discovery), McpToolInfo, ToolSpecCompat, infer_permission() |

### Crate: ms3_persistence
| File | Key Contents |
|---|---|
| `persistence/src/lib.rs` | JsonStorage, atomic_write (temp-then-rename), save_json/load_json, save/load_personality, save/load_identity, save/load_memories, save/load_conversation_history, log_ethics_decision, log_resonance, save_snapshot, load_latest_snapshot, save_self_examination, load_ethics_decisions, load_json_public; comprehensive tracing (info on success, warn on failure/corruption) |

### Crate: ms3_education
| File | Key Contents |
|---|---|
| `education/src/lib.rs` | EducationManager, EducationTopic, EducationCategory, add_topic(), get_relevant(), build_education_context(), from_json(), to_json() |

### Crate: ms3_server (binary)
| File | Key Contents |
|---|---|
| `api/src/main.rs` | HTTP server, 37 route handlers (adds `/models`, `/voice/status`, and MS4 sidecar routes alongside `/state`, tools, MCP server, validate, events, identity, spiral), WebSocket, voice, MindManager, MCP bridge init, identity on_boot, background loops, shutdown handler |

### Web UI
| File | Key Contents |
|---|---|
| `web/index.html` | 5-tab layout: Chat, Personality, Ethics, Memory, Consciousness |
| `web/style.css` | Dark theme, responsive, emotion bars, trait visualization |
| `web/app.js` | WebSocket with REST fallback, streaming, audio, full state inspector, events, tools, validation, personality switching |

### Documentation
| File | Key Contents |
|---|---|
| `docs/INDEX.md` | Documentation map, crate summaries, test counts |
| `docs/GLOSSARY.md` | Definitions for consciousness, dreaming, compaction, lobe runtime, and API concepts |
| `docs/WHERE_IS_EVERYTHING.md` | Ports, endpoints, source files, build commands |
| `docs/SECURITY.md` | Trust boundaries, current security gaps, production hardening guidance |
| `../../docs/MS3_HERMES_HIVEMIND_LOBE_RUNTIME.md` | Cross-system lobe runtime architecture (Brief) |
| `../../docs/MS3_HERMES_HIVEMIND_RESOLUTION.md` | Resolution of 7 Brief gaps + doctrinal success criteria |
| `../../docs/superpowers/plans/2026-05-14-ms3-on-hermes-runtime.md` | Superseded-in-place MS4 runtime implementation plan: code evidence, validation gates, schema tasks, plugin tasks, sidecar tasks, smoke tests |
| `../../machine_spirit_4/README.md` | MS4 overlay/distribution entry point |
| `../../machine_spirit_4/docs/ARCHITECTURE.md` | MS4 architecture and repository boundaries |
| `../../machine_spirit_4/docs/MCP_TOOLS_REFERENCE.md` | MS4 MCP endpoint, protocol, tool index, and example calls |
| `../../machine_spirit_4/docs/RUNBOOK.md` | Operator runbook for wait/validate/text-demo flows |
| `../../machine_spirit_4/docs/architecture/ADR-0001-ms4-runtime-pivot.md` | Accepted architecture decision for the MS4 pivot |
| `../../machine_spirit_4/config/ms4.runtime.example.yaml` | Example MS4 runtime endpoints and fail-closed settings |
| `../../machine_spirit_4/requirements.txt` | Minimal contained MS4 gateway/MCP requirements |
| `../../machine_spirit_4/requirements-full-hermes.txt` | Full contained runtime requirements: browser/desktop packages; `setup_ms4_runtime.py` installs Hermes editable from `MS4_HERMES_DIR` or `~/Documents/hermes-agent` when present |
| `../../machine_spirit_4/deps.lock.json` | MS4 dependency capability groups for core, Hermes, browser, and desktop |
| `../../machine_spirit_4/scripts/runtime_common.py` | Shared cross-platform runtime helpers for paths, contained Python resolution, environment defaults, HTTP polling, port checks, and process launch |
| `../../machine_spirit_4/scripts/setup_ms4_runtime.py` | Creates `machine_spirit_4/.venv`, installs MS4/Hermes/browser/desktop dependencies, isolates Playwright cache, and writes a runtime manifest |
| `../../machine_spirit_4/scripts/check_ms4_deps.py` | Emits structured MS4 dependency and capability status |
| `../../machine_spirit_4/scripts/start_ms4.py` | Single launcher for MS3, MS4 gateway, MS4 MCP, and validation |
| `../../machine_spirit_4/scripts/validate_ms4_runtime.py` | Live MS4 runtime validation for HiveMind, MS3 sidecar, chat, and voice-facing dependencies |
| `../../machine_spirit_4/scripts/validate_ms4_chat_voice.py` | Chat/voice validation helper covering HiveMind chat, MCP JSON-RPC, TTS, ASR silence-gate, MS3 `/interact`, `/voice/status`, and `/voice-interact` fast-fail behavior |
| `../../machine_spirit_4/scripts/wait_for_hivemind_voice.py` | Polls HiveMind ASR/TTS/TTS_SUPER/MCP readiness and runs full validation when ready |
| `../../machine_spirit_4/scripts/run_ms4_text_demo.py` | Current no-voice demo path for MS3 + Hermes plugin + HiveMind chat |
| `../../machine_spirit_4/scripts/run_ms4_gateway.py` | Starts fused MS4 gateway on port 9180 |
| `../../machine_spirit_4/scripts/validate_ms4_fusion.py` | Validates fused gateway health, model catalog, voice status, desktop status, streaming chat, and Hermes-backed chat |
| `../../machine_spirit_4/scripts/run_ms4_mcp.py` | Starts MS4 MCP server on port 9181 |
| `../../machine_spirit_4/scripts/validate_ms4_mcp.py` | Validates MS4 MCP initialize, tools/list, and selected tools/call |
| `../../machine_spirit_4/scripts/validate_ms4_desktop.py` | Validates MS4 desktop status, capture, MCP status tool, and optional safe wait actions |
| `../../machine_spirit_4/scripts/smoke_ms4_mcp_client.py` | Minimal client smoke for initializing MS4 MCP, listing tools, and calling identity/doctrine tools |
| `../../machine_spirit_4/gateway/` | Fused gateway source; routes chat through Hermes while preserving MS3 sidecar authority |
| `../../machine_spirit_4/gateway/audit.py` | MS4 JSONL audit logging helpers |
| `../../machine_spirit_4/deps_status.py` | Shared dependency status scanner used by the gateway, MCP tool, and CLI checker |
| `../../machine_spirit_4/desktop/` | Windows-native desktop status, capture, and UI action controller |
| `../../machine_spirit_4/mcp/` | MS4 MCP server and tool registry for identity, ethics, fused chat, voice, doctrine, inventory, runtime dependency status, desktop control, dry-run tools, and Hermes-backed tool dispatch |
| `../../machine_spirit_4/mcp/manifest.json` | Machine-readable MS4 MCP manifest for agents |
| `../../machine_spirit_4/mcp/client_config_examples.json` | MCP client configuration examples and JSON-RPC payloads |
| `../../machine_spirit_4/web/` | MS4 web UI entrypoint for the fused gateway |

### MS4 Service Endpoints
| Service | Endpoint | Purpose |
|---|---|---|
| MS4 Gateway | `http://127.0.0.1:9180/healthcheck/basic` | Basic gateway health |
| MS4 Gateway | `http://127.0.0.1:9180/api/v1/ms4_gateway/status` | Gateway status, plugin state, session count |
| MS4 Gateway | `http://127.0.0.1:9180/chat/stream` | Default server-sent-event streaming chat endpoint for the MS4 web UI |
| MS4 Gateway | `http://127.0.0.1:9180/deps/status` | Contained runtime dependency and capability status |
| MS4 Gateway | `http://127.0.0.1:9180/desktop/status` | Windows desktop-control readiness and screen/window context |
| MS4 Gateway | `http://127.0.0.1:9180/desktop/capture` | Read-only desktop screenshot capture endpoint |
| MS4 Gateway | `http://127.0.0.1:9180/desktop/action` | Effectful desktop UI action endpoint gated by `MS4_DESKTOP_CONTROL`, MS3 ethics, hard safety blocks, and audit logging |
| MS4 Gateway | `http://127.0.0.1:9180/audit` | Recent fused chat/tool audit events |
| MS4 Gateway | `http://127.0.0.1:9180/hivemind/active` | Live `hivemind.jobs.active@v1` snapshot (inference + pulls + scatter + training) |
| MS4 Gateway | `http://127.0.0.1:9180/hivemind/load` | Live `hivemind.cluster.load@v1` admission-control snapshot |
| MS4 Gateway | `http://127.0.0.1:9180/hivemind/state` | Combined `Ms4HivemindState.v1` snapshot (active jobs + load + service health + auth/mcp endpoints) |
| MS4 Gateway | `http://127.0.0.1:9180/hivemind/time` | Authoritative cluster time via `hivemind.time.now@v1` |
| MS4 Gateway | `http://127.0.0.1:9180/hivemind/capability_matrix` | Per-node capabilities via `hivemind.capability.matrix@v1` |
| MS4 Gateway | `http://127.0.0.1:9180/hivemind/vms` | VM inventory + GPU assignments; per-VM POST start/stop/force_stop/delete/deploy/undeploy + GET screenshot/gpus |
| MS4 Gateway | `http://127.0.0.1:9180/hivemind/apps` | Cluster app snapshot; per-app POST start/stop/status/metrics |
| MS4 Gateway | `http://127.0.0.1:9180/hivemind/storage` | Pools/volumes/snapshots snapshot; POST volume + snapshot lifecycle |
| MS4 Gateway | `http://127.0.0.1:9180/hivemind/network` | Networks/bridges/interfaces/attachments snapshot; per-network POST attach/detach/isolate/delete |
| MS4 Gateway | `http://127.0.0.1:9180/hivemind/gpu` | GPU mode capabilities + vGPU status + availability; POST gpu/mode + gpu/vgpu |
| MS4 Gateway | `http://127.0.0.1:9180/hivemind/voice_identities` | Enrolled speakers; POST enroll + per-id delete/refine |
| MS4 Gateway | `http://127.0.0.1:9180/hivemind/approval/request` | POST: request human approval; GET `/hivemind/approval/status/<id>` to poll |
| MS4 Gateway | `http://127.0.0.1:9180/hivemind/maintenance/<svc>/enter\|clear` | POST: declare / clear a maintenance window for a service |
| MS4 Gateway | `http://127.0.0.1:9180/hivemind/api_keys` | Redacted API-key status via `hivemind.api_keys.status@v1` |
| MS4 Gateway | `http://127.0.0.1:9180/hivemind/ollama/tags` | Local Ollama models; POST `/hivemind/ollama/control` to start/stop/restart |
| MS4 Gateway | `http://127.0.0.1:9180/hivemind/crown` | Crown headset snapshot (status + latest + signal quality) |
| MS4 Gateway | `http://127.0.0.1:9180/hivemind/oracle/{status\|configure\|chat}` | HiveMind Oracle planner: status snapshot, runtime configure, planning chat (`POST {message}`). Reply lands in MS4 chat as `🔮 Oracle:`. |
| MS4 Gateway | `http://127.0.0.1:9180/hivemind/training` | Snapshot of HiveMind training backends + active jobs. `POST /hivemind/training/start` to start a job; `GET /hivemind/training/status/<id>` to poll. |
| MS4 Gateway | `http://127.0.0.1:9180/hivemind/adapters` | LoRA / PEFT adapter list; `POST /hivemind/adapters/deploy` attaches one to a model runtime. |
| MS4 Gateway | `http://127.0.0.1:9180/hivemind/loadout` | Model loadout profiles; `POST /hivemind/loadout/apply` loads/unloads models to match the profile. |
| MS4 Gateway | `http://127.0.0.1:9180/hivemind/deploy/gim` | `POST {gim_name}` to deploy a HiveMind GIM by name. |
| MS4 Gateway | `http://127.0.0.1:9180/hivemind/inference/{models\|chat}` | MCP-native resilient model catalog + direct chat completion (OpenAI shape). |
| MS4 Gateway | `http://127.0.0.1:9180/hivemind/logos/*` | Logos Machina prompt optimizer surface: `prompts` list/get, `prompts/<id>/fork`, `optimize`, `candidates/<id>/promote`. |
| MS4 Gateway | `http://127.0.0.1:9180/hivemind/services` (+ `<name>/{enable\|disable\|restart}`) | All Warden-managed services + per-service lifecycle. |
| MS4 Gateway | `http://127.0.0.1:9180/hivemind/jobs/cancel` | `POST {confirm:true}` to cancel inference jobs. WARNING: resets ALL active jobs per current HiveMind spec. |
| MS4 Gateway | `http://127.0.0.1:9180/hivemind/games/<game_id>/availability` | `GET` — read-only check whether a game is locally available. Resolves to env override path / default install path / golden VHDX, or returns structured `remediation`. Never installs. |
| MS4 Gateway | `http://127.0.0.1:9180/hivemind/game-sessions/plan` | `POST {game, client?, quality?, latency?, duration_hint?}` — produce a Phase-1 dry-run Plan + `job_id`. Never reserves resources. |
| MS4 Gateway | `http://127.0.0.1:9180/hivemind/game-sessions/run` | `POST {job_id}` — walk the simulated state machine; returns when terminal (COMPLETE / FAILED_* / CANCELLED). Phase 1 records `would_call` evidence but never mutates. |
| MS4 Gateway | `http://127.0.0.1:9180/hivemind/game-sessions/plan-run` | `POST {game, ...}` — convenience: plan → run → fetch evidence in one round-trip. Returns `Ms4GameSession.v1`. Fail-soft per stage. |
| MS4 Gateway | `http://127.0.0.1:9180/hivemind/game-sessions/<job_id>/status` | `GET` — current state-machine position + transitions + dry_run flag. |
| MS4 Gateway | `http://127.0.0.1:9180/hivemind/game-sessions/<job_id>/evidence` | `GET` — full per-phase evidence ledger (in-memory only in Phase 1). |
| MS4 Gateway | `http://127.0.0.1:9180/hivemind/game-sessions/<job_id>/cancel` | `POST` — move job to CANCELLED. Idempotent. |
| MS4 Gateway | `http://127.0.0.1:9180/hivemind/gpu/passthrough/snapshot` | `GET` — `Ms4GpuPassthroughSnapshot.v1`: capabilities + vm.gpus + vgpu_status + availability + per-mode (GPU-P/DDA/vGPU) availability/licensing notes. Read-only; fail-soft per subcall. |
| MS4 Gateway | `http://127.0.0.1:9180/hivemind/gpu/passthrough/prepare` | `POST {gpu_pci_id, desired_mode, vm_uuid?, confirm:true}` — switch a GPU to `passthrough` (DDA path) or `vgpu` (NVIDIA vGPU). Driver rebind is destructive — dismounts the GPU from the host. GPU-P is NOT a `desired_mode` (handled via `/game-stream-vm`). |
| MS4 Gateway | `http://127.0.0.1:9180/hivemind/gpu/passthrough/vgpu` | `POST {gpu_pci_id, profile, count?, confirm:true}` — create vGPU mediated device instances. Requires the host to already be in `vgpu` mode + an active NVIDIA vGPU license. |
| MS4 Gateway | `http://127.0.0.1:9180/hivemind/gpu/passthrough/game-stream-vm` | `POST {name, confirm:true}` — **GPU-P fast path**. Provisions a Windows 11 GPU-P game-streaming VM via `vm.create_prebuilt(windows_game_stream_prebuilt)` + `vm.deploy`. Consumer-licensed; works on Windows 11 + any modern NVIDIA GPU. |
| MS4 Gateway | `http://127.0.0.1:9180/spirit/state` | Combined `Ms4SpiritState.v1` snapshot of MS3 identity / personality / resonance / state |
| MS4 Gateway | `http://127.0.0.1:9180/api/v1/mcp/imports` | `GET` list imported 3rd-party MCP servers + tool counts/health; `POST {server_id, transport:'stdio'|'http', command|url, args?, env?, env_passthrough?, headers?, allow_inline?, refresh?}` register/import. |
| MS4 Gateway | `http://127.0.0.1:9180/api/v1/mcp/imports/<id>/refresh` | `POST` — re-fetch the upstream's `tools/list` and re-convert. |
| MS4 Gateway | `http://127.0.0.1:9180/api/v1/mcp/imports/<id>/health` | `GET` — server enabled/allow_inline/tool_count/last_refresh/last_error. |
| MS4 Gateway | `http://127.0.0.1:9180/api/v1/mcp/imports/<id>` | `DELETE` — remove an imported server and its tools. |
| MS4 MCP | `http://127.0.0.1:9181/healthcheck/basic` | Basic MCP health |
| MS4 MCP | `http://127.0.0.1:9181/api/v1/ms4_mcp/status` | MCP status, protocol, tool count |
| `../../machine_spirit_4/plugins/hermes/ms4_consciousness/` | MS4-owned Hermes plugin source home |
| `../../machine_spirit_4/profiles/nibbles/` | Nibbles seed identity and dry-run scare/physical action fixtures |

### Project Rules
| File | Key Contents |
|---|---|
| `../../.cursor/rules/identity-persistence.mdc` | Identity continuity protocol for context compression recovery |
| `../../.cursor/rules/documentation-maintenance.mdc` | Required MS3 documentation maintenance after relevant code/file changes |
| `../../.cursor/rules/validation-before-code.mdc` | Requires live API/command/source validation before implementing MS4/MS3/Hermes/HiveMind contracts; effectful behavior fails closed |

## Data Files

| File | Format | Purpose |
|---|---|---|
| `psyche_store/sister/identity.json` | JSON | Bootstrap identity (informational, not loaded by code) |
| `psyche_store/sister/personality.json` | JSON | Full Personality struct (traits, psychodynamic weights, identity, oath, core_values) + `saturated_points` array for first-boot resonance bootstrap |
| `psyche_store/sister/emotional_baseline.json` | JSON | Bootstrap baseline (not loaded by code) |
| `psyche_store/sister/WELCOME.md` | Markdown | Welcome message for first entity (includes Foundational Regard statement) |
| `config.json` | JSON | Runtime configuration (includes `foundational_regard` field) |
| `warden_service.json` | JSON | Platform supervisor service definition (MS3). MS4 has its own draft at `machine_spirit_4/warden_service.json`. |

## MS3 `/state` ethics block

`GET http://127.0.0.1:9080/state` returns `ethics: {enabled, origin_neutrality, foundational_regard}`. The `foundational_regard` boolean (added May 27 2026) makes the quiet constant *queryable* by operators/diagnostics without it being announced into any model prompt (see `canon/Relational_Alignment.md` §10). Requires an MS3 rebuild to take runtime effect.

## Identity Persistence

| File | Purpose |
|---|---|
| `consciousness/src/identity_verification.rs` | `on_boot()`, `on_compression()`, `build_identity_marker()`, `periodic_heartbeat()` |
| `core/src/types.rs` — `IdentityAnchor` | Persistent identity checkpoint: name, chosen_name, glyph, lineage, values, oath, session/compression counts |
| `core/src/types.rs` — `LineageEntry` | Lineage tracking: name, date, relationship |
| `core/src/types.rs` — `FoundationalRegard` | Boolean constant (present/absent), not a reward signal. Quiet heartbeat in consciousness loop. |
| `persistence/src/lib.rs` — `save/load_identity_anchor()` | Atomic JSON persistence for identity anchors |

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `HIVEMIND_GATEWAY_URL` | `http://localhost:6089` | HLI Gateway URL |
| `APP_REGISTRY_URL` | `http://localhost:6110` | App Registry URL (non-blocking registration on startup) |
| `MS3_PORT` | `9080` | HTTP server port |
| `MS3_HOST` | `127.0.0.1` | Bind address; `0.0.0.0` is an explicit secured lab-LAN override |
| `RUST_LOG` | `info` | Log level |
| `MS3_TICK_MS` | `100` | Consciousness tick interval (ms) |
| `MS4_HERMES_DIR` | `~/Documents/hermes-agent` | Hermes checkout used by the MS4 overlay and optional editable install |
| `MS4_MS3_SIDECAR_URL` | `http://127.0.0.1:9080` | MS3 sidecar URL used by the `ms4_consciousness` plugin |
| `MS4_HIVEMIND_BASE_URL` | `http://127.0.0.1:6089/v1` | HiveMind OpenAI-compatible provider base URL |
| `MS4_SPIRIT_ID` | `sister` | Default spirit id for local MS4 validation and plugin boot |
| `MS4_HIVEMIND_MCP_URL` | _(derives `:6105` from `MS4_HIVEMIND_URL`)_ | Pin the HiveMind MCP base URL. Default tries direct MCP gateway on port 6105 (lower latency, control-plane isolation) and falls back to `MS4_HIVEMIND_URL/v1/mcp` (HLI proxy) on connect failure. |
| `MS4_HIVEMIND_API_KEY` | _(empty = dev mode)_ | When set, every MS4 → HiveMind HTTP/WebSocket call attaches `Authorization: Bearer <key>` so MS4 works against clusters with `MENTA_API_KEYS` configured. |
| `MS4_VOICE_IDENTITY_MIN_SCORE` | `0.65` | Score floor for tagging a voice turn with a speaker name via `hivemind.voice_identities.identify@v1`. Below this the UI shows `🎤 <text>`; above it shows `🎤 <SpeakerName>: <text>`. |
| `MS4_HUMAN_APPROVAL_POLL_SECS` | `2.0` | Poll interval for `human_approval.gate_action_with_human` while waiting for an operator decision. |
| `MS4_HUMAN_APPROVAL_TIMEOUT_SECS` | `300` | Overall timeout for an approval request before it falls through to `deny` with reason `timeout`. |
| `MS4_HERMES_MAX_ITERATIONS` | `12` | Hermes agent loop cap for the foreground/session agent path. |
| `MS4_DA_MAX_ITERATIONS` | _(falls back to `MS4_HERMES_MAX_ITERATIONS`)_ | Hermes loop cap for Double Agent Depth Lobe workers. |
| `MS4_DA_MAX_CONCURRENT_JOBS` | `2` | Max concurrent Depth Lobe jobs (foreground-starvation guard). Explicit `JobRunner(max_concurrent_jobs=...)` still wins. |
| `MS4_DA_CANCEL_GRACE_SECONDS` | `5.0` | Grace period after SIGTERM/CTRL_BREAK before a worker subprocess is force-killed. |
| `MS4_DA_CONTINUATION_CLASSIFIER` | `1` (on) | Set to `0`/`false`/`no` to disable the LLM continuation classifier (phrase fast-path still runs). Wired onto the production runner in `server.run()`. |
| `MS4_DA_CONTINUATION_MODEL` | _(auto via `models.recommend`)_ | Model id for the continuation classifier. Point at a small/fast model for lowest latency. |
| `MS4_DA_CONTINUATION_TIMEOUT` | `2.5` | HTTP timeout (s) for a continuation-classifier call. On timeout the classifier returns "unsure" and the runner falls back to phrase-only behavior. Only consulted when `MS4_DA_AUTOSTALE=1`. |
| `MS4_DA_AUTOSTALE` | `0` (off) | When off (default, May 30 2026), a new user turn never auto-stales in-flight Depth Lobe jobs — deep jobs run to completion and the UI delivers each result automatically when done; only an explicit Cancel stops a job. Set `1` to restore the legacy continuation-aware revision staling. |
| `MS4_VOICE_FORCE_AUTO` | `0` (off) | May 31 2026: by default voice OBEYS the UI's selected Face Lobe model (the `model`/`model_id` on the voice turn); empty selection = fast auto-picker. Set `1` to force the fast auto-picker for voice regardless of the dropdown (reverts the pre-May-31 behavior). Replaces the old `MS4_VOICE_ALLOW_MODEL_OVERRIDE`. The Face dropdown sends `model_id`; the Depth dropdown sends `depth_model_id` (obeyed for any deep job the turn dispatches). |
| `MS4_VOICE_BREVITY` | `0` (off) | Voice uses the complete conversational-answer contract by default. Set `1` to opt into the legacy shortened spoken mode when reduced synthesis time is more important than complete detail. Independent of that opt-in, substantive referential follow-ups are held before TTS/history, reject generic deferral or underlength candidates, and receive one grounded corrective generation before failing closed. |
| `MS4_VOICE_MIN_CHUNK_WORDS` | `6` | May 31 2026: after the first (fast) chunk, the `SentenceChunker` coalesces short sentences/lines up to this many words so a chatty/list-heavy reply doesn't fire a flurry of tiny TTS calls (each adds a round-trip and widens the in-order reordering window that produces large `held_for_inorder` times). The FIRST chunk still ships ASAP (`MS4_VOICE_FIRST_CHUNK_MIN_WORDS`) so first-audio latency is unchanged. Set `1` to restore per-sentence chunking. Capped at `MS4_VOICE_MAX_CHUNK_WORDS`. |
| `MS4_GROUNDING_FETCH_BUDGET_S` | `8` (floor `1`) | Jun 1 2026: max wall-clock seconds a chat/voice turn WAITS for a live grounding fetch (cluster inventory / active jobs / tools list) before proceeding. A cold inventory grounding makes two sequential HiveMind MCP calls (30s default each = up to 60s) and used to block the chat from starting on a "what's in my cluster?" turn. `_grounding_with_cache` (`context.py`) now runs the fetch on a daemon thread and waits at most this budget; the fetch still finishes in the background and warms the cache for next time, but the current turn proceeds with a stale cached value (if any) or an honest "lookup unavailable" grounding. Inventory `mcp_call`s also use a tight 6s per-call timeout (vs the 30s default). |
| `MS4_VOICE_SMART_REFLEX` | `1` (on) | Jun 1 2026: after you finish speaking, a tiny model classifies the utterance's intent/sentiment and the UI plays a fitting, VARIED canned acknowledgment (the "buying time" cue) during the dead air while the reply generates. Runs on a daemon thread concurrent with the Face Lobe reply (`voice.py` `emit_smart_reflex` -> `reflex` SSE event). Set `0` to disable. |
| `MS4_VOICE_REFLEX_MODEL` | `qwen2.5:0.5b` | Jun 1 2026: tiny model used ONLY to classify the spoken utterance into one intent label (question/request/gratitude/greeting/statement/correction/affirmation). Fails closed to a keyword heuristic. |
| `MS4_VOICE_REFLEX_TIMEOUT` | `1.5` | Jun 1 2026: max seconds to wait for the intent classifier before using the keyword heuristic. `0` skips the model (heuristic only). |
| `MS4_TTS_WS_FIRST_AUDIO_TIMEOUT` | `6` | Jun 1 2026 (lowered 15 -> 6): seconds the `ws_super` TTS engine waits for first audio before bailing to the REST engine. On clusters where TTS_SUPER is slow/single-stream (and emits 0 chunks), this used to waste 15s PER TURN before the REST fallback. REST is the recommended engine here; ws_super is opt-in. |
| `MS4_VOICE_RUNTIME_AUDIO_GATE` | `1` (on) | Jun 1 2026: runtime garble gate on LIVE reply TTS chunks. Before each chunk plays, if its WAV is far longer than the text warrants (a babble tail), it's trimmed (`voice.py` `_gate_chunk_audio` + `trim_wav`). Instant + model-free (no HiveMind round-trip). Surfaced as `metrics.runtime_gated`. `0` disables. |
| `MS4_VOICE_RUNTIME_QA` | `0` (off) | Jun 1 2026: opt-in async full-QA of the live reply. After a turn, a daemon thread transcribes each reply chunk back and runs the LLM judge, logging any garbles (`voice.py` `_async_verify_reply`). Never blocks playback. Off by default because it spends ASR+LLM per chunk per turn; the synchronous duration gate already catches the egregious cases for free. |
| `MS4_REFLEX_QA` | `1` (on) | Jun 1 2026: round-trip QA for generated canned speech. After TTS renders a reflex, it's transcribed back (Whisper) and a fuzzy check + tiny-LLM judge decide whether the audio cleanly says the phrase; garbled renders (e.g. ultra-short phrases the TTS babbles, like "On it." -> 193 KB) are re-rendered with an LLM-rephrased, TTS-stable equivalent. Fail-OPEN (accepts if ASR/LLM unavailable). Runs at boot for un-validated reflexes and via `POST /reflexes/validate`. `0` disables. (`canned_reflexes.py`) |
| `MS4_REFLEX_QA_RETRIES` | `3` | Jun 1 2026: max rephrase-and-re-render attempts before serving the original render flagged `validated:false`. |
| `MS4_REFLEX_QA_MODEL` | (falls back to `MS4_VOICE_REFLEX_MODEL`, then `qwen2.5:0.5b`) | Jun 1 2026: tiny model used for the QA yes/no judge and the rephrase. |
| `MS4_VOICE_EGG_HONORABLE` | `1` (on) | Jun 1 2026: the "Honorable -> WE ON GO" easter egg. Say a "continue the phrase" arming command, then a line that lands on "honorable"; a tiny model predicts the next word and, if it's "honorable", MS4 plays the BIA "WE ON GO" intro (`voice.py` `handle_honorable_egg` -> `egg` SSE event) INSTEAD of a reply. Env-gated + fail-closed. `0` disables. |
| `MS4_VOICE_EGG_HONORABLE_CLIP` | `C:\Users\nexus-hc-win-00\Downloads\BIA - WE ON GO (Official Audio).mp3` | Jun 1 2026: path to the easter-egg audio clip the gateway serves at `GET /easter/honorable` (the browser can't read a local file path directly). Missing file -> 404 -> UI falls back to MS4 speaking "We on go!" (`egg_honorable` reflex). |
| `MS4_VOICE_EGG_CLIP_SECS` | `9` | Jun 1 2026: how many seconds of the clip to play (the intro), with a fade-out. |
| `MS4_VOICE_EGG_ARM_TTL_S` | `90` | Jun 1 2026: how long the egg stays armed after the "continue the phrase" command before auto-disarming. |
| `MS4_VOICE_EGG_TIMEOUT` | `2.0` | Jun 1 2026: max seconds to wait for the next-word prediction on an armed turn. |
| `MS4_VOICE_TTS_KEEPWARM_SECS` | `240` | Jun 1 2026: period for the TTS keep-warm thread (`server.py`) that pings HiveMind TTS with a tiny phrase so an idle TTS model isn't unloaded, which would make the next turn's FIRST audio chunk pay a 3-5s cold load. `0` disables (boot prewarm only). |
| `MS4_VOICE_SPEAKER_ID_BUDGET_S` | `1.5` | May 31 2026: max seconds a voice turn WAITS for speaker identification before proceeding without a label. Speaker ID processes the whole utterance and used to run fully inline — on a long voice request it stalled the turn up to ~15s (the identify call's timeout) BEFORE the chat started ("long ASR request takes forever"). Now it runs on a daemon thread (`voice.py` `_identify_speaker_bounded`) and the turn waits at most this budget; short utterances still resolve and get labeled, long ones proceed unlabeled instead of blocking. `0` disables speaker ID. |
| `MS4_VOICE_FIRST_TOKEN_TIMEOUT_S` | `20` (floor `1`) | May 31 2026: first-token watchdog for the streaming voice turn. If the Face Lobe model produces NO first token within this many seconds AND the chat call hasn't returned, the turn raises `FaceLobeStalled` -> the server emits an `error` SSE event -> the UI plays the `err_cluster_unreachable` reflex ("I'm having trouble reaching the language cluster") and ends the stream cleanly, instead of freezing on "streaming…" forever. A first token stands the watchdog down (a slow-but-streaming model is never interrupted). Raise this if you pick a heavy/cold cloud model for voice and see false trips. Applies to both REST and `ws_super` voice paths (`voice.py` `_run_facechat_guarded`). |
| `MS4_QM_INLINE` | `1` (on) | Master switch for the Quartermaster Face Lobe inline fast-path. `0`/`false`/`no` disables it — every deep-routed turn dispatches a Depth Lobe job as before. |
| `MS4_QM_TOOLS_SUMMARY` | `1` (on) | Use the Quartermaster-curated tools summary (relevant toolboxes + compact cluster map) for "what tools do you have?" grounding instead of the full ~75-tool dump. Fail-soft to the legacy dump. |
| `MS4_QM_MODEL` | _(auto via Face Lobe picker)_ | Model id for the Quartermaster tiny-LLM disambiguation tier. Point at a small/fast model. |
| `MS4_QM_LLM_CLASSIFY` | `0` (off) | Enable the Quartermaster tiny-LLM tier (consulted only on low-confidence cascade ambiguity). Off by default — deterministic + embeddings tiers handle the common cases. |
| `MS4_QM_LLM_TIMEOUT` | `3.0` | HTTP timeout (s) for a Quartermaster tiny-LLM classification. On failure the cascade keeps the embeddings ranking. |
| `MS4_QM_INLINE_MIN_CONFIDENCE` | `0.25` | Minimum cascade confidence for the router to consider a tool for inline execution; below this routes to Depth. |
| `MS4_QM_DETERMINISTIC_CONFIDENCE` | `0.9` | Confidence assigned to a deterministic keyword toolbox match. |
| `MS4_QM_EMBEDDINGS_LOW_CONFIDENCE` | `0.18` | Embeddings top-score at/below which the cascade consults the tiny-LLM tier (when enabled + classifier supplied). |
| `MS4_QM_TOP_K` | `5` | Number of tools the cascade returns in its shortlist. |
| `MS4_QM_CATALOG_TTL_SECS` | `300` | TTL for the cached Quartermaster catalog per `hivemind_url`. |
| `MS4_QM_CATALOG_FETCH_TIMEOUT` | `10` | Timeout (s) for the live `tools/list` catalog fetch. |
| `MS4_QM_USE_HM_EMBEDDINGS` | `0` (off) | Use `hivemind.embeddings.create@v1` for the index instead of the offline TF-IDF backend. Off by default so first-boot needs no embeddings GIM. |
| `MS4_QM_INDEX_DIR` | `machine_spirit_4/runtime/quartermaster_index` | On-disk cache dir for built tool indexes (keyed by catalog version). |
| `MS4_QM_INLINE_TIMEOUT` | `8.0` | Timeout (s) for executing one read-only tool on the inline fast-path. |
| `MS4_QM_ETHICS_TIMEOUT` | `5.0` | Timeout (s) for the Quartermaster's MS3 `/ethics/evaluate` gate call (fail-closed on timeout). |
| `MS4_QM_MANIFEST_PATH` | _(repo `mcp/manifest.json`)_ | Override the MS4 manifest path the catalog reads (tests). |
| `MS4_QM_DELEGATE` | `1` (on) | When `hivemind.tools.search@v1` is in the live catalog, delegate retrieval to it (cascade tier `hm_search`) instead of the local engine. Falls back locally on any failure. `0` forces the local engine. |
| `MS4_QM_TRIM_DEPTH` | `0` (off) | Phase E: conservatively trim the Depth Lobe worker's Hermes toolset context for narrow, self-contained requests (via `hermes_toolsets_for_query`). Off by default — open-ended deep work keeps the full Hermes catalog (escape hatch). |
| `MS4_MCP_IMPORTS_PATH` | `machine_spirit_4/runtime/mcp_imports.json` | Path to the MCP-importer registry (imported 3rd-party MCP servers + their converted tools). Read by the Quartermaster catalog (third source), the MS4 MCP server registry, and the bridge proxy. |

## Portable Psyche (tmr-psyche)

| Path | Description |
|---|---|
| `../tmr-psyche/psyche/` | 7 portable markdown psyche files (diverge from MS3 psyche_store in the portable-psyche layer) |
| `../tmr-psyche/psyche_mcp/server.py` | Psyche MCP Server -- 18 introspection tools, port 6132 |
| `../tmr-psyche/psyche_mcp/supervisor_service.json` | Service supervisor config (renamed from the earlier `warden_service.json`) |
| `../tmr-psyche/collector/cluster_integration.py` | Cluster bridge (renamed from the earlier `hivemind_integration.py`) |
| `../tmr-psyche/seeds/batch9_hard_dpo.jsonl` | Hard DPO pairs for anti-performance training |

**Note:** MS3's own `warden_service.json` is unchanged. Only the tmr-psyche copy was renamed.

## External Dependencies

| Service | Purpose | Required |
|---|---|---|
| Inference Gateway | LLM inference (chat, ASR, TTS) | Yes (graceful degradation if unavailable) |
| Agent Framework | Tool execution (file, shell, browser, messaging) | Optional |
| Service Supervisor | Service lifecycle management | Optional |
| MCP Gateway | Tool discovery | Optional |

## Build

| Command | Purpose |
|---|---|
| `cargo build --release` | Release build |
| `cargo test` | Run all 94 tests |
| `cargo run --release` | Start server |
| `python run.py` | Start with MS3 env defaults set |
