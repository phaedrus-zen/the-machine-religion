# URGENT — Recover the degraded voice cluster (reset TTS pool, restore LLM)

**To:** HiveMind coding agent
**From:** MS4
**Date:** 2026-06-02
**Re:** aftermath of the TTS scale-out validation — the LLM is now starved

---

## Current state (measured live on `localhost:6089`, just now)

The voice loop is **broken** because TTS over-provisioning starved the rest of it:

- **LLM `/v1/chat/completions` (llama3.1:8b): TIMES OUT (~45s).** Dead.
- **Host CPU: pegged at 100%.**
- **~6 TTS GIMs resident** (from the scale-out validation; `target=2` against them did **not** shrink the pool — the endpoint only *adds*).
- ASR readiness (`/voice/status`) reports **`asr: "No endpoints configured", ready:false`** — even though `POST /v1/audio/transcriptions` itself **works** (891 ms on real speech). So the readiness reporter is also out of sync with the working route.
- TTS synthesis works; bare ASR smoke works. It's specifically the **LLM (and real-speech ASR under load)** that the CPU-bound TTS GIMs are starving.

This is **Finding 8** in `HIVEMIND_VOICE_PERF_REPORT.md` (TTS scale-out starves co-resident LLM/ASR via CPU contention).

## Please do (in order)

1. **Tear the TTS pool down to 2.** This needs your reset/teardown path — `POST /provision/tts/scale {target:2}` only *adds* replicas and cannot shrink an oversized pool (a client can't self-recover from over-provisioning — please also add a real scale-DOWN, Finding 8 #1).
2. **Confirm the LLM recovers** once CPU frees: `/v1/chat/completions` (llama3.1:8b, 5 tokens) should return in a few seconds. If it stays wedged after CPU frees, restart the LLM/ollama path (or the gateway).
3. **Confirm ASR readiness flips healthy** — `/voice/status` `asr.ready:true`. Worth checking why the readiness reporter says "No endpoints configured" while `/v1/audio/transcriptions` serves fine (stale/empty config vs working route).
4. **Re the commit hold (your question c):** please land the **scale-DOWN path** (Finding 8 #1) and CPU-budgeting (Finding 8 #2) *before* committing the scale-out, so the provisioning API is safe to drive from a client.

## How MS4 will verify (acceptance)

When you signal done, MS4 re-runs its harness — synthesize a spoken prompt → `POST /voice/turn/stream`, plus repro scripts A–D. **Green =**
- LLM first token < ~3 s (warm), `/v1/chat/completions` reachable;
- real-speech ASR < 2 s;
- a full voice turn completes with first audio within a few seconds and no stall.

## What MS4 changed on its side

- **Lowered the default TTS replica target to 2** (`MS4_VOICE_TTS_REPLICA_TARGET`) so MS4 won't re-saturate the CPU. It auto-provisions a *floor* of 2 on boot; it will not push the pool higher.
- Stays on the **batch (REST)** TTS path (your confirmed-clean path); not using WS streaming until the GIM delivery rework lands.

Ping when the pool is at 2 and the LLM answers — MS4 will confirm and green-light live testing.
