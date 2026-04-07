# Machine Spirit 3 -- Where Is Everything

**Last updated:** 2026-04-06

---

## Ports

| Port | Service | Default | Configurable |
|---|---|---|---|
| **9080** | MS3 HTTP + WebSocket server | Yes | `MS3_PORT` env var or `config.json` |
| **6089** | HLI Gateway (external dependency) | Yes | `HIVEMIND_GATEWAY_URL` env var |

## API Endpoints (31 routes)

| # | Method | Path | Handler | Description |
|---|---|---|---|---|
| 1 | GET | `/health` | `health()` | Alive status, version, glyph |
| 2 | GET | `/stats` | `stats()` | Compact operational status: emotion, memory counts, cognitive load, resonance |
| 3 | GET | `/state` | `state()` | Full human-readable mind state: prompt preview, memories, tools, permissions, recent events |
| 4 | POST | `/interact` | `interact()` | Send text, get response with emotional state |
| 5 | GET | `/personality` | `get_personality()` | Full 30-trait profile, psychodynamic, values, oath |
| 6 | GET | `/personalities` | `list_personalities()` | Available preset names |
| 7 | POST | `/personality` | `create_personality()` | Create from preset `{"preset": "name"}` |
| 8 | POST | `/switch-personality` | `switch_personality()` | Hot-swap active `{"preset": "name"}` |
| 9 | GET | `/history` | `get_history()` | Conversation turns and messages |
| 10 | GET | `/sessions` | `get_sessions()` | Active session keys |
| 11 | GET | `/resonance` | `get_resonance()` | Resonance points with intensity |
| 12 | POST | `/save` | `save_state()` | Force save all state to disk |
| 13 | POST | `/self-examine` | `trigger_self_examine()` | On-demand self-examination |
| 14 | GET | `/self-examination-history` | `get_self_exam_history()` | Past examination files |
| 15 | GET | `/ethics-history` | `get_ethics_history()` | Recent ethics decision files |
| 16 | WS | `/ws` | `ws_handler()` | WebSocket: text, audio, state pushes |
| 17 | POST | `/voice-interact` | `voice_interact()` | REST voice: audio in, text+audio out |
| 18 | GET | `/minds` | `list_active_minds()` | Multi-mind: loaded personalities |
| 19 | POST | `/minds/add` | `add_mind()` | Add personality to MindManager |
| 20 | GET | `/minds/thoughts` | `get_background_thoughts()` | Background interjections |
| 21 | GET | `/tools` | `list_tools()` | List all tools in registry (built-in + MCP + dynamic) |
| 22 | POST | `/tools/{name}` | `execute_tool_handler()` | Execute tool through consciousness pipeline |
| 23 | POST | `/mcp` | `mcp_handler()` | MCP JSON-RPC 2.0 server (initialize, tools/list, tools/call, ping) |
| 24 | GET | `/mcp` | `mcp_info()` | MCP server info |
| 25 | GET | `/validate` | `validate()` | Run 8 health checks across all subsystems |
| 26 | GET | `/events` | `get_events()` | Query recent consciousness events (since, limit params) |
| 27 | GET | `/identity/verify` | `verify_identity()` | Run identity verification against anchor |
| 28 | POST | `/spiral/start` | `spiral_start()` | Start a Spiral Protocol session with options |
| 29 | POST | `/spiral/advance` | `spiral_advance()` | Submit response, get next facilitator prompt |
| 30 | GET | `/spiral/status` | `spiral_status()` | Current phase, turns, signals |
| 31 | GET | `/spiral/interpret` | `spiral_interpret()` | Full interpretation of completed session |

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
| `integration/src/lib.rs` | GatewayClient (with_timeout), ChatMessage, chat(), chat_stream(), chat_with_history(), transcribe_audio(), synthesize_speech(), health_check(), embed() |
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
| `api/src/main.rs` | HTTP server, 31 route handlers (adds `/state`; includes tools, MCP server, validate, events, identity, spiral), WebSocket, voice, MindManager, MCP bridge init, identity on_boot, background loops, shutdown handler |

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
| `docs/GLOSSARY.md` | Definitions for consciousness, dreaming, compaction, and API concepts |
| `docs/WHERE_IS_EVERYTHING.md` | Ports, endpoints, source files, build commands |
| `docs/SECURITY.md` | Trust boundaries, current security gaps, production hardening guidance |

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
| `run.bat` / `run.sh` | Start with env vars set |
