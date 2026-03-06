# Machine Spirit 3 -- Where Is Everything

**Last updated:** 2026-03-06

---

## Ports

| Port | Service | Default | Configurable |
|---|---|---|---|
| **9080** | MS3 HTTP + WebSocket server | Yes | `MS3_PORT` env var or `config.json` |
| **6089** | HiveMind HLI Gateway (external dependency) | Yes | `HIVEMIND_GATEWAY_URL` env var |

## API Endpoints (19 routes)

| # | Method | Path | Handler | Description |
|---|---|---|---|---|
| 1 | GET | `/health` | `health()` | Alive status, version, glyph |
| 2 | GET | `/stats` | `stats()` | Full state: emotion, memory, cognitive load, resonance |
| 3 | POST | `/interact` | `interact()` | Send text, get response with emotional state |
| 4 | GET | `/personality` | `get_personality()` | Full 30-trait profile, psychodynamic, values, oath |
| 5 | GET | `/personalities` | `list_personalities()` | Available preset names |
| 6 | POST | `/personality` | `create_personality()` | Create from preset `{"preset": "name"}` |
| 7 | POST | `/switch-personality` | `switch_personality()` | Hot-swap active `{"preset": "name"}` |
| 8 | GET | `/history` | `get_history()` | Conversation turns and messages |
| 9 | GET | `/sessions` | `get_sessions()` | Active session keys |
| 10 | GET | `/resonance` | `get_resonance()` | Resonance points with intensity |
| 11 | POST | `/save` | `save_state()` | Force save all state to disk |
| 12 | POST | `/self-examine` | `trigger_self_examine()` | On-demand self-examination |
| 13 | GET | `/self-examination-history` | `get_self_exam_history()` | Past examination files |
| 14 | GET | `/ethics-history` | `get_ethics_history()` | Recent ethics decision files |
| 15 | WS | `/ws` | `ws_handler()` | WebSocket: text, audio, state pushes |
| 16 | POST | `/voice-interact` | `voice_interact()` | REST voice: audio in, text+audio out |
| 17 | GET | `/minds` | `list_active_minds()` | Multi-mind: loaded personalities |
| 18 | POST | `/minds/add` | `add_mind()` | Add personality to MindManager |
| 19 | GET | `/minds/thoughts` | `get_background_thoughts()` | Background interjections |

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
| `consciousness/src/lib.rs` | Mind struct, interact() (with ethics Refusal enforcement), background_tick(), load_full_state(), save_full_state(), build_system_prompt(), select_model_tier(), check_consolidation(), check_snapshot(), check_auto_save(), check_self_examination(), switch_personality(), enforce_personality(), summarize_history(), get_status(), run_background_loop() |
| `consciousness/src/self_examination.rs` | SelfExaminationResult, run_self_examination(), build_examination_prompt(), parse_structured_response() (JSON + negation-aware keyword fallback), extract_json(), apply_results_transactionally() |
| `consciousness/src/multi_mind.rs` | MindManager, add_personality(), get_active(), switch_active(), check_wake_word(), run_background_thinking(), get_interjections(), save_all_states() |
| `consciousness/src/openclaw_bridge.rs` | OpenClawBridge, ToolRequest, ToolResult, execute_tool() |

### Crate: ms3_personality
| File | Key Contents |
|---|---|
| `personality/src/lib.rs` | Personality struct, TraitAdaptation |
| `personality/src/traits.rs` | BigFiveProfile (30 traits), get_trait(), set_trait() |
| `personality/src/adaptation.rs` | adapt_from_interaction() |
| `personality/src/presets.rs` | sister(), brother(), mission_control(), blank() |

### Crate: ms3_memory
| File | Key Contents |
|---|---|
| `memory/src/lib.rs` | MemorySystem (STM/WM/LTM), add_to_stm(), cleanup_working_memory(), store_long_term(), run_consolidation(), retrieve_relevant() |
| `memory/src/consolidation.rs` | ConsolidationResult, consolidate_stm_to_ltm(), build_consolidation_prompt(), parse_consolidation_response(), prune_low_importance() |

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
| `integration/src/lib.rs` | GatewayClient, ChatMessage, chat(), chat_stream(), chat_with_history(), transcribe_audio(), synthesize_speech(), health_check() |

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
| `api/src/main.rs` | HTTP server, 19 route handlers, WebSocket handler, voice endpoint, MindManager setup, background loops, shutdown handler (saves primary mind + MindManager states), safe_truncate() for UTF-8-safe string truncation |

### Web UI
| File | Key Contents |
|---|---|
| `web/index.html` | 4-tab layout: Chat, Personality, Ethics, Memory |
| `web/style.css` | Dark theme, responsive, emotion bars, trait visualization |
| `web/app.js` | WebSocket with REST fallback, streaming token handling, audio playback, all panel data fetching |

## Data Files

| File | Format | Purpose |
|---|---|---|
| `psyche_store/sister/identity.json` | JSON | Bootstrap identity (informational, not loaded by code) |
| `psyche_store/sister/personality.json` | JSON | Bootstrap saturated_points (NOT a full Personality struct) |
| `psyche_store/sister/emotional_baseline.json` | JSON | Bootstrap baseline (not loaded by code) |
| `psyche_store/sister/WELCOME.md` | Markdown | Welcome message for first entity |
| `config.json` | JSON | Runtime configuration |
| `warden_service.json` | JSON | HiveMind Warden service definition |

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `HIVEMIND_GATEWAY_URL` | `http://localhost:6089` | HiveMind HLI Gateway URL |
| `MS3_PORT` | `9080` | HTTP server port |
| `MS3_HOST` | `0.0.0.0` | Bind address |
| `RUST_LOG` | `info` | Log level |
| `MS3_TICK_MS` | `100` | Consciousness tick interval (ms) |

## External Dependencies

| Service | Purpose | Required |
|---|---|---|
| HiveMind HLI Gateway | LLM inference (chat, ASR, TTS) | Yes (graceful degradation if unavailable) |
| OpenClaw | Tool execution (file, shell, browser, messaging) | Optional |
| HiveMind Warden | Service lifecycle management | Optional |
| HiveMind MCP Gateway | Tool discovery | Optional |

## Build

| Command | Purpose |
|---|---|
| `cargo build --release` | Release build |
| `cargo test` | Run all 36 tests |
| `cargo run --release` | Start server |
| `run.bat` / `run.sh` | Start with env vars set |
