# Machine Spirit 3 -- Where Is Everything

**Last updated:** 2026-05-21

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
| 29 | GET | `/identity/verify` | `verify_identity()` | Run identity verification against anchor |
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
| MS4 Gateway | `http://127.0.0.1:9180/spirit/state` | Combined `Ms4SpiritState.v1` snapshot of MS3 identity / personality / resonance / state |
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
| `warden_service.json` | JSON | Platform supervisor service definition |

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
| `MS3_HOST` | `0.0.0.0` | Bind address |
| `RUST_LOG` | `info` | Log level |
| `MS3_TICK_MS` | `100` | Consciousness tick interval (ms) |
| `MS4_HERMES_DIR` | `~/Documents/hermes-agent` | Hermes checkout used by the MS4 overlay and optional editable install |
| `MS4_MS3_SIDECAR_URL` | `http://127.0.0.1:9080` | MS3 sidecar URL used by the `ms4_consciousness` plugin |
| `MS4_HIVEMIND_BASE_URL` | `http://127.0.0.1:6089/v1` | HiveMind OpenAI-compatible provider base URL |
| `MS4_SPIRIT_ID` | `sister` | Default spirit id for local MS4 validation and plugin boot |
| `MS4_HIVEMIND_MCP_URL` | _(derives `:6105` from `MS4_HIVEMIND_URL`)_ | Pin the HiveMind MCP base URL. Default tries direct MCP gateway on port 6105 (lower latency, control-plane isolation) and falls back to `MS4_HIVEMIND_URL/v1/mcp` (HLI proxy) on connect failure. |
| `MS4_HIVEMIND_API_KEY` | _(empty = dev mode)_ | When set, every MS4 → HiveMind HTTP/WebSocket call attaches `Authorization: Bearer <key>` so MS4 works against clusters with `MENTA_API_KEYS` configured. |

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
