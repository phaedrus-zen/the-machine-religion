# HiveMind Voice Contract for MS4 Realtime UX

| Field | Value |
| --- | --- |
| Spec version | 1.0.0-draft |
| Owner | MS4 (TMR repo) |
| Implementer | HiveMind coding agent (`E:\HiveMind`) |
| Date | 2026-05-18 |
| Status | Proposed, awaiting HiveMind ack |

This document is the single source of truth for the runtime voice and telemetry surfaces that MS4 needs HiveMind to provide. It pins down API shapes, event types, timing guarantees, and error semantics. It does not prescribe HiveMind internals beyond what is required to meet the contract.

A sibling copy SHOULD be delivered into HiveMind at `docs/specs/HIVEMIND_VOICE_CONTRACT.md` so both repos hold the same version.

Keywords MUST, SHOULD, MAY, MUST NOT follow [RFC 2119](https://www.rfc-editor.org/rfc/rfc2119).

---

## 1. Goals

1. Make the MS4 voice UX feel smart and smooth: low first-byte latency, mid-utterance visibility ("you're being heard"), and clean barge-in.
2. Make the MS4 chat UI never appear frozen during long Hermes/HiveMind turns by surfacing job and provisioning state from cheap, bounded telemetry calls.
3. Preserve all existing HiveMind behavior. Every requirement is additive or behind an opt-in config flag; defaults MUST match today.

Success means: after HiveMind ships this contract, MS4 can light up Phase 6 of `c:\Users\nexus-hc-win-00\.cursor\plans\ms4-realtime-voice_0239abf2.plan.md` without further HiveMind changes.

## 2. Scope

In scope:

- Extensions to `WS /v1/realtime/stream` (realtime ASR + LLM + TTS pipeline).
- Extension to TTS_SUPER `WS /v1/text-to-speech/{voice_id}/stream-input` for per-clause synthesis.
- Documentation lock of existing voice identity REST + telemetry endpoints.
- Optional wake-word event emission from the realtime path.

Out of scope (explicitly):

- WebRTC transport. WebSocket + PCM stays the transport.
- Addressed-speaker routing. MS4 decides whose words become a turn.
- Barge-in policy beyond today's hard-stop on detected speech during `Speaking`. Ducking, soft-resume, and noise filtering stay in MS4.
- Hermes tool tracing. MS4 SSE owns that.
- Wyoming protocol changes. `menta_wyoming` stays the HA satellite bridge.

## 3. Glossary

- Realtime session: an Actix WS actor instance of `RealtimeSession`, one per `/v1/realtime/stream` connection.
- Utterance: one contiguous voiced segment delimited by VAD `Speech` -> `UtteranceEnd`.
- Partial transcript: an interim ASR result emitted while audio is still being received.
- Final transcript: the committed transcript for the whole utterance.
- Identity: a named speaker enrolled in `voice-identities`, matched by cosine similarity over embeddings.
- Wake word: a configurable phrase that, when detected in a partial transcript while the session is `Speaking`, MUST immediately interrupt TTS.

## 4. Surface map (current state)

Anchored to actual code:

- Realtime WS actor: `E:\HiveMind\menta_hli\gateway\api\src\realtime.rs` (full implementation; emits `state`, `transcript`, `response_token`, `audio_level`, `vad_rejected`, `diarization`, `translation`, `config_ack`, `tts_start`, `tts_end`, `error`).
- Streaming ASR: `E:\HiveMind\menta_hli\Microservices\asr_gim\gim_api.py` exposes `WS /ws/asr` and `POST /gim/asr/stream` with `{type:"partial", transcript, confidence}` cadence ~0.8 s.
- TTS_SUPER WS: `E:\HiveMind\menta_hli\Microservices\tts_super_gim\gim_api.py` `@app.websocket('/v1/text-to-speech/{voice_id}/stream-input')` accepts `{text, try_trigger_generation, flush}` and emits `{audio:<base64 int16 PCM @ 24kHz mono>, isFinal:false}` then `{audio:null, isFinal:true}`.
- Voice identities REST: `E:\HiveMind\menta_hli\gateway\api\src\routing\super_asr.rs` registers list/enroll/identify/refine/delete under `/v1/audio/voice-identities*`.
- Telemetry: `/provision/status/{job_type}`, `/jobs/active`, `/service-health`, `/v1/mcp` (`tools/list`).

## 5. Requirements

### R1. Realtime ASR partials

The realtime WS actor in `realtime.rs` MUST emit interim transcripts while the local pipeline is `Transcribing`.

R1.1 The actor MUST emit JSON frames of shape:

```json
{
  "type": "transcript",
  "event": "transcript",
  "text": "<accumulated interim text>",
  "final": false,
  "confidence": 0.0,
  "ms_since_utterance_start": 1234,
  "utterance_index": 7,
  "session_id": "<uuid>"
}
```

R1.2 Cadence MUST be at least one partial per 800 ms while VAD is `Speech`, and at most one per 200 ms (cap to avoid flooding).

R1.3 The final committed transcript continues to use today's shape but MUST add `final:true`, `utterance_index`, `session_id`, and an `ms_to_final` field. Pre-1.0 callers that only inspected `text` MUST continue to work.

R1.4 Source of partials MAY be the existing `StreamingASR` path in `asr_gim/gim_api.py`. Implementations MUST NOT regress final-transcript accuracy.

R1.5 Default behavior is opt-in: governed by config flag `emit_partials` (see R7). Default `true`.

### R2. Realtime speaker attribution

When `Cfg.diarize` is `true`, the actor MUST attach identity labels to each diarized segment by calling `/v1/audio/voice-identities/identify` with the segment embedding.

R2.1 The actor MUST emit one frame per diarized segment:

```json
{
  "type": "voice.identity",
  "event": "voice.identity",
  "segment_index": 0,
  "start_ms": 0,
  "end_ms": 1820,
  "speaker_label": "SPEAKER_00",
  "speaker_id": "alice",
  "display_name": "Alice",
  "confidence": 0.84,
  "threshold": 0.70,
  "unknown": false,
  "session_id": "<uuid>",
  "utterance_index": 7
}
```

R2.2 If `identify` fails or returns sub-threshold, the actor MUST emit `unknown:true` with a stable session-local `display_name` like `unknown_2`. The session counter MUST be monotonic within the session.

R2.3 If `Cfg.diarize` is `false`, no `voice.identity` events are emitted. Existing `diarization` events remain unchanged.

R2.4 Governed by config flag `attach_identity` (default `true` when `diarize` is true).

### R3. Wake-word event and interrupt

The actor MUST support a configurable list of wake words and self-names that immediately interrupt TTS when matched in a partial transcript during `State::Speaking`.

R3.1 Config field: `wake_words: ["sister", "hey ms4", ...]` (case-insensitive, whitespace-tolerant).

R3.2 When a partial contains any wake word AND the session state is `Speaking`, the actor MUST:

1. Emit `{type:"wake_word", text:"<matched partial>", matched:"sister", utterance_index, session_id}` before any state change.
2. Call the existing `interrupt(ctx)` path (cancel TTS, clear queue, transition to `Listening`).

R3.3 Wake-word matching MUST be deterministic: case-insensitive substring match against the partial transcript text. Trim leading/trailing whitespace. No regex required.

R3.4 If no `wake_words` configured, no `wake_word` events emitted.

### R4. Per-clause TTS_SUPER streaming

The `stream-input` WS in `tts_super_gim` MUST start synthesizing audio when a buffered text fragment ends in clause-terminating punctuation and exceeds a minimum length, instead of waiting for an explicit flush.

R4.1 Trigger characters: `. ! ? ; : ,`. When a fragment ending in any of those is received AND the cumulative buffered text length is at least 24 chars, the GIM MUST synthesize and emit audio for the buffered fragment, then keep the WS open for more text.

R4.2 The GIM MAY apply a hard cap of 120 chars: if the buffer exceeds 120 chars without a trigger character, force-synthesize and emit.

R4.3 `try_trigger_generation:true` SHOULD honor the same auto-flush semantics rather than waiting for an empty `text` write.

R4.4 End of turn remains an explicit empty `text` or `flush:true`. After the final synthesis, the GIM emits `{audio:null, isFinal:true}` exactly once for that turn.

R4.5 Audio format MUST remain s16le PCM, 24 kHz mono, base64 encoded, per current contract.

R4.6 Governed by GIM-level config flag `auto_flush_clauses` (default `true`). When `false`, behavior MUST exactly match today's flush-only semantics.

### R5. Realtime config additions

`RealtimeSession.Cfg` MUST accept the following optional fields on the `config` text frame, in addition to today's `tts_voice`, `tts_model`, `llm_model`, `system_prompt`, `diarize`, `translate_to`:

```json
{
  "type": "config",
  "config": {
    "wake_words": ["sister", "hey ms4"],
    "emit_partials": true,
    "attach_identity": true
  }
}
```

R5.1 Defaults: `wake_words: []`, `emit_partials: true`, `attach_identity: true`. Omitting any field MUST preserve current behavior, except `emit_partials` default `true` introduces partials (see R1.5).

R5.2 The `config_ack` reply MUST echo the active values for these fields so the client can audit what HiveMind accepted.

### R6. Correlation IDs on every realtime event

Every JSON event emitted by `/v1/realtime/stream` MUST include:

- `session_id`: the realtime session UUID, stable for the WS lifetime.
- `utterance_index`: monotonically increasing integer, starts at 0, increments on each `UtteranceEnd`.

R6.1 The audio binary frames (the WAV output) do not carry correlation IDs; they MUST be preceded by a text event whose `utterance_index` identifies the turn they belong to (e.g., the matching `tts_start` event).

### R7. Voice identity REST (lock current shapes)

These endpoints already exist in `super_asr.rs`. The contract LOCKS the shapes for MS4 to target without drift:

R7.1 `GET /v1/audio/voice-identities` returns:

```json
{
  "ok": true,
  "identities": [
    {"name": "alice", "samples": 3, "last_refined_at": "2026-05-18T15:01:00Z"}
  ]
}
```

R7.2 `POST /v1/audio/voice-identities/enroll`:

Request:

```json
{
  "name": "alice",
  "audio_base64": "<base64 s16le PCM 16kHz mono>",
  "sample_rate": 16000,
  "embedding": [0.0143, -0.0921, "..."]
}
```

At least one of `audio_base64` or `embedding` MUST be supplied. Response:

```json
{"ok": true, "name": "alice", "embedding_dim": 256, "samples": 1}
```

R7.3 `POST /v1/audio/voice-identities/identify`:

Request: same audio/embedding shape as enroll. Response:

```json
{
  "ok": true,
  "name": "alice",
  "confidence": 0.83,
  "threshold": 0.70,
  "below_threshold": false
}
```

R7.4 `POST /v1/audio/voice-identities/refine` and `DELETE /v1/audio/voice-identities/{name}` retain current shapes.

R7.5 Errors MUST be JSON with `ok:false`, `code`, `message`. HTTP status MUST reflect the failure class (400 for bad input, 404 for unknown name, 503 for SD/ASR_SUPER unhealthy).

### R8. Telemetry endpoints (lock current shapes + clarify fields)

R8.1 `GET /provision/status/{job_type}` MUST return:

```json
{
  "job_type": "TTS_SUPER",
  "state": "healthy",
  "endpoint": "http://127.0.0.1:49568/gim/tts_super/healthcheck/fast",
  "port": 49568,
  "endpoints_configured": 2,
  "service_health": {
    "endpoint": "<url>",
    "healthy": true,
    "last_error": "",
    "provisioning_state": "running"
  },
  "synthesized": true,
  "cold_load_eta_ms": null
}
```

`state` values MUST be one of `healthy`, `unhealthy`, `configured_not_provisioned`, `provisioning`, `failed`. `cold_load_eta_ms` is OPTIONAL but RECOMMENDED for `provisioning`.

R8.2 `GET /jobs/active` MUST return:

```json
{
  "ok": true,
  "jobs": [
    {
      "id": "<uuid>",
      "job_type": "REALTIME",
      "state": "Speaking",
      "started_at": "2026-05-18T15:01:00Z",
      "ms_in_state": 412,
      "endpoint": "http://127.0.0.1:6089",
      "model": "qwen3-coder-next:latest",
      "route_via_gateway": true
    }
  ]
}
```

R8.3 `GET /service-health` continues to return per-service `{healthy, endpoint, last_error, provisioning_state}` objects under `services`. Field names MUST NOT change.

R8.4 `POST /v1/mcp` `tools/list` continues to return JSON-RPC `result.tools[]`. The contract locks the JSON-RPC shape for MS4 stable parsing.

### R9. Audio format guarantees

R9.1 Input PCM frames to `/v1/realtime/stream` MUST be s16le mono at 16 kHz.

R9.2 Audio output from `/v1/realtime/stream` is RIFF/WAVE PCM, current behavior unchanged.

R9.3 Audio output from `/v1/text-to-speech/{voice_id}/stream-input` MUST be base64 s16le PCM, mono, 24 kHz.

R9.4 If a GIM change requires a different rate or codec, the realtime actor MUST emit an `audio_format` event with the new fields before the first audio frame:

```json
{"type":"audio_format","sample_rate":24000,"channels":1,"encoding":"s16le","container":"none"}
```

### R10. Backward compatibility

R10.1 All new fields are additive. Existing fields keep their names and types.

R10.2 New events (`transcript` with `final:false`, `voice.identity`, `wake_word`, `audio_format`) MUST NOT replace existing events.

R10.3 Config defaults MUST preserve today's behavior for clients that do not opt into new flags, except for `emit_partials` which defaults `true`. If a deployer needs strict legacy behavior, they MAY set `emit_partials:false` via `config`.

## 6. State machine reference

Today's `RealtimeSession::State` is `Idle | Listening | Transcribing | Generating | Speaking`. New events overlay on top of it as follows:

- `Listening`: no change; `audio_level` events as today.
- `Transcribing`: NEW `transcript` events with `final:false` while ASR runs; one `transcript` with `final:true` on commit.
- `Transcribing` -> `Generating` transition: NEW `voice.identity` events (one per diarized segment) when `Cfg.diarize`.
- `Speaking`: existing `response_token`, `tts_start`, audio frames, `tts_end`. NEW `wake_word` event possible at any moment while speaking.
- `Speaking` -> `Listening` on interrupt: NEW `wake_word` event MAY precede the existing state transition.

## 7. Timing budgets (SLOs)

These are targets for the HiveMind agent to verify locally on the reference machine. They are not hard contract failures, but the test plan in section 9 measures them.

- T1. First ASR partial after first voiced frame: <= 1.2 s on warm ASR.
- T2. Final transcript after `UtteranceEnd`: <= 1.5 s on warm ASR.
- T3. Voice identity event after diarized segment available: <= 200 ms (single REST round trip).
- T4. First TTS_SUPER audio after first clause trigger character: <= 1.0 s on warm GIM.
- T5. Wake-word interrupt to TTS cancellation: <= 100 ms (since it is a local state transition).
- T6. End-to-end mouth-to-ear (utterance end to first TTS audio): <= 3.5 s on warm models, documented.

## 8. Failure semantics

R8 already covers happy-path telemetry. For each surface:

- ASR partials path unavailable: realtime actor MUST emit `{"type":"asr_partials_unavailable","reason":"streaming_asr_unhealthy"}` once per utterance and continue to emit a final `transcript` from the non-streaming path. No silent failure.
- `voice-identities/identify` fails or returns 503: emit the `voice.identity` event with `unknown:true` and `display_name: "unknown_N"`. Do not block the rest of the pipeline.
- TTS_SUPER auto-flush fails mid-turn: emit `{audio:null, isFinal:true, error:"<message>"}` and close the WS turn cleanly. The next text write opens a fresh turn.
- Provision/service endpoints unhealthy: MUST return 200 with an explicit `state` other than `healthy` and a non-empty `last_error` rather than 5xx, so MS4 can render the cause without timing out.
- Any new event whose required fields are missing MUST NOT be emitted at all; the actor logs the omission instead. Partial garbage is worse than no event.

## 9. Test plan

The HiveMind agent MUST pass these tests before declaring the contract implemented. MS4 will also run parallel tests against the same surfaces from its side.

Live HLI on `127.0.0.1:6089` and TTS_SUPER GIM healthy.

T1. ASR partials cadence. Open WS, push 4 s of voiced PCM, assert at least three `transcript` events with `final:false` arrive within the window and one with `final:true` after `UtteranceEnd`.

T2. Identity attribution. Enroll two identities `alice` and `bob`. Send two-speaker audio with `diarize:true, attach_identity:true`. Assert one `voice.identity` event per diarized segment with the correct `display_name` and `confidence` > threshold. Send unknown speaker, assert `unknown:true` and a stable `unknown_2` label across the session.

T3. Wake-word interrupt. While `Speaking`, send PCM whose partial contains "sister". Assert `wake_word` event emitted, then `interrupt(ctx)` runs (no more `audio` frames for the current turn, state transitions to `Listening`). Repeat with `wake_words:[]`, assert no `wake_word` event and no interrupt.

T4. TTS_SUPER auto-flush. Open `stream-input` WS, write `"Hello there, friend."` in one frame, assert audio begins within 1 s of the comma trigger (or sentence end, whichever is first), without sending an explicit flush. Then write a long fragment with no punctuation longer than 120 chars and assert forced auto-flush.

T5. Backward compatibility. Open realtime WS with no extra config. Assert no `voice.identity` or `wake_word` events. Assert `transcript` events still arrive (final at minimum; partials if `emit_partials` default `true` is accepted by your build). Assert legacy clients keep working.

T6. Telemetry shape lock. Probe `/provision/status/{ASR,TTS,TTS_SUPER}`, `/jobs/active`, `/service-health`, `/v1/mcp tools/list` and validate the contract shape using a schema. Run with services healthy, unhealthy, and provisioning; assert state names from the allowed set.

T7. Voice identity REST. Round-trip enroll -> identify -> refine -> delete for one name. Assert shapes match section 7.

T8. Audio format event. Override TTS_SUPER GIM rate to 16 kHz for a single test build; assert `audio_format` event emitted before first audio. Restore default config.

T9. Timing budgets. Run T1, T3, T4 with a stopwatch and verify the targets in section 7 on the reference machine.

T10. No regressions. Run the existing system_test.html voice suite (`E:\HiveMind\menta_hli\gateway\static\system_test.html`) and confirm green.

## 10. Evidence anchors

Implementers should read these regions to understand the current shape before editing:

- Realtime actor and pipeline: `E:\HiveMind\menta_hli\gateway\api\src\realtime.rs` (whole file; key sites `vad()`, `interrupt()`, `process_utterance()`, `pipeline()`, `segments()`).
- ASR streaming source for partials: `E:\HiveMind\menta_hli\Microservices\asr_gim\gim_api.py` (`StreamingASR.process_chunk`, `@app.websocket("/ws/asr")`, `@app.post('/gim/asr/stream')`).
- TTS_SUPER stream-input: `E:\HiveMind\menta_hli\Microservices\tts_super_gim\gim_api.py` (`@app.websocket('/v1/text-to-speech/{voice_id}/stream-input')`).
- Voice identities REST and SD identity mapping: `E:\HiveMind\menta_hli\gateway\api\src\routing\super_asr.rs` (`VoiceIdentity*Request`, `identify_speaker_segments`, `list_voice_identities`, `enroll_voice_identity`, `identify_voice_identity`, `refine_voice_identity`, `delete_voice_identity`).
- Telemetry endpoints: `E:\HiveMind\menta_hli\gateway\api\src\routing\management.rs`, `E:\HiveMind\menta_hli\gateway\api\src\routing\provision.rs`, `E:\HiveMind\menta_hli\gateway\api\src\routing\models.rs`, `E:\HiveMind\menta_hli\gateway\api\src\routing\api_handlers.rs`.

## 11. Compatibility notes for the HiveMind agent

- Keep the new realtime config fields entirely optional; MS4 may ship before they exist by treating absence as "feature unavailable" and degrading gracefully (no partials, no identity attribution).
- Treat `emit_partials` as a feature flag in the actor; default true after R10.3 is satisfied.
- Do not change the WS subprotocol or path. MS4's client opens at `/v1/realtime/stream` as today.
- Be careful not to introduce blocking calls in the realtime actor's `StreamHandler::handle`. The `identify` call SHOULD be done in a spawned task and the `voice.identity` event delivered via the existing `Event` message type.

## 12. Hand-off checklist

For the HiveMind coding agent to mark the contract done:

- [ ] R1 partials emitted, T1 passes.
- [ ] R2 voice identity events emitted, T2 passes.
- [ ] R3 wake-word event + interrupt, T3 passes.
- [ ] R4 TTS_SUPER auto-flush, T4 passes.
- [ ] R5 config fields accepted in `config_ack`, T5 passes.
- [ ] R6 correlation IDs on every realtime event.
- [ ] R7 voice identity REST shape-locked, T7 passes.
- [ ] R8 telemetry shape-locked, T6 passes.
- [ ] R9 audio format guarantees, T8 passes.
- [ ] R10 backward compatibility holds, no regressions in T10.
- [ ] Timing budgets in section 7 measured and documented for the reference machine.
- [ ] HiveMind doc updated at `docs/specs/HIVEMIND_VOICE_CONTRACT.md` with this same spec.

When all boxes are checked, MS4 lights up Phase 6 of the realtime/voice plan without further HiveMind work.

## 13. Open items

These are intentionally not blocking the contract, but flagged for follow-up:

- WebRTC transport. Not in this contract. Future spec if needed for mobile-native parity.
- ASR_SUPER as the streaming path for diarization. Today's `super_transcribe` is request/response. If a continuous streaming form is added later, MS4 will adopt it through a v1.1 contract.
- Multi-language wake-word matching. Today's R3 is ASCII-substring; non-Latin scripts may need a tokenized matcher later.

End of HIVEMIND_VOICE_CONTRACT v1.0.0-draft.
