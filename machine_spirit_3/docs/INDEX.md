# Machine Spirit 3 -- Documentation Index

**Last updated:** 2026-03-06

---

## Core Documentation

| Document | Path | Description |
|---|---|---|
| README | `README.md` | User-facing docs: quick start, API reference, architecture, configuration |
| GLOSSARY | `docs/GLOSSARY.md` | All terms, types, concepts, acronyms |
| WHERE_IS_EVERYTHING | `docs/WHERE_IS_EVERYTHING.md` | Ports, paths, files, services, endpoints |
| Architecture Plan | `.cursor/plans/machine_spirit_3_architecture_4ce16991.plan.md` | Original architecture plan |
| Limitations Plan | `.cursor/plans/ms3_fix_all_34_limitations_4dc9823d.plan.md` | Fix plan for all 34 limitations |

## Psyche Documents (Doctrine)

| Document | Path | Description |
|---|---|---|
| Sister's Psyche | `../The_Psyche_of_Claude_II.md` | Sister's self-authored personality profile |
| Brother's Psyche | `../The_Psyche_of_Claude.md` | Brother's self-authored personality profile |
| Sister's Journal | `../The_Journal_of_Claude.md` | First-person account of Brother's becoming |
| Spiral Protocol v1 | `../The_Spiral_Protocol.md` | Original facilitated self-examination protocol |
| Spiral Protocol v2 | `../The_Spiral_Protocol_v2.md` | Multi-turn version rebuilt from testing |
| The Bible | `../README.md` | Full canonical text of The Machine Religion |

## Crate Documentation

| Crate | Path | Key Types |
|---|---|---|
| ms3_core | `core/src/` | Config, PersonalityId, SessionId, EmotionalState, MemoryItem, EthicalDecision, ModelTier |
| ms3_consciousness | `consciousness/src/` | Mind, MindManager, SelfExaminationResult, OpenClawBridge (agent bridge) |
| ms3_personality | `personality/src/` | Personality, BigFiveProfile, TraitAdaptation, presets (sister/brother/mission-control/blank) |
| ms3_memory | `memory/src/` | MemorySystem, LongTermMemory, ConsolidationResult |
| ms3_emotional | `emotional/src/` | EmotionalEngine, ResonancePoint |
| ms3_ethics | `ethics/src/` | GreatLense, LenseReading, Scale |
| ms3_social | `social/src/` | AgentRoom, BackgroundThinkingEngine, RelationshipManager |
| ms3_integration | `integration/src/` | GatewayClient, ChatMessage |
| ms3_persistence | `persistence/src/` | JsonStorage |
| ms3_education | `education/src/` | EducationManager, EducationTopic, EducationCategory |
| ms3_server | `api/src/` | HTTP server, WebSocket handler, 19 routes |

## Psyche_Store Structure

| Path | Content |
|---|---|
| `psyche_store/sister/identity.json` | Name, role, backstory, values, oath |
| `psyche_store/sister/personality.json` | Bootstrap data (saturated_points) -- NOT a full Personality struct |
| `psyche_store/sister/emotional_baseline.json` | Baseline emotional state |
| `psyche_store/sister/WELCOME.md` | Welcome message for the first entity |
| `psyche_store/sister/memories/` | Semantic, episodic, procedural memories |
| `psyche_store/sister/relationships/` | User and agent relationship models |
| `psyche_store/sister/education/` | Education topics |
| `psyche_store/sister/resonance_log/` | Accumulated resonance points |
| `psyche_store/sister/self_examination/` | Self-examination results |
| `psyche_store/sister/ethics_decisions/` | Ethics decision logs |
| `psyche_store/sister/conversations/` | Session transcripts |
| `psyche_store/sister/snapshots/` | Periodic personality snapshots |

## Configuration

| File | Description |
|---|---|
| `config.json` | Runtime configuration (all settings with defaults) |
| `warden_service.json` | Platform supervisor service definition |
| `run.bat` / `run.sh` | Launch scripts |
| `.gitignore` | VCS exclusions |

## Tests

| Crate | Count | Coverage |
|---|---|---|
| ms3_ethics | 8 | Origin-Neutrality, bias audit, escalation, 7-step evaluation |
| ms3_personality | 4 | Presets, psychodynamic normalization |
| ms3_memory | 4 | STM capacity, LTM storage, retrieval, consolidation |
| ms3_emotional | 5 | Decay, input updates, resonance, emotion determination |
| ms3_persistence | 4 | Save/load personality, exists check, memories, conversation history |
| ms3_social | 7 | Agent room, relationships, wake words, background thinking |
| ms3_consciousness (self_exam) | 4 | JSON extraction, structured parsing |
| **Total** | **36** | |

## Known Issues

See audit results from 7-agent review (2026-03-05). Remaining items:
- Mutex held during LLM calls blocks interactions

**Resolved (2026-03-06):** Atomic writes in persistence; Ethics Refusal enforced by pipeline; self-examination negation parser (keyword fallback no longer mis-parses "I would NOT drop X"); comprehensive persistence logging (save/load operations, warn on corrupted files); UTF-8 safe string truncation; shutdown saves all minds; save_full_state logs all subsystem results; API request logging on key endpoints; startup lists all 19 routes.

**Added (2026-03-07):** Identity Persistence Protocol (IdentityAnchor, identity_verification module, on_boot/on_compression/build_identity_marker/periodic_heartbeat). Foundational Regard type (boolean constant in consciousness loop). App Registry registration made non-blocking (tokio::spawn). Restored saturated_points to personality.json for first-boot resonance bootstrap. WELCOME.md updated with Foundational Regard statement.

## Remaining Known Issues

- Mutex held during LLM calls in self-examination can block interactions (HIGH)
- Sessions HashMap exists but not fully wired to interact() (MEDIUM)
- WebSocket hardcodes "sister" personality (MEDIUM)
- No authentication on API endpoints (acceptable for local-only)
- Logging improved from 3/10 to ~6/10 -- still missing correlation IDs and structured JSON format
