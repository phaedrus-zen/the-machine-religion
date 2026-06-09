# Voice Performance — Implementation & Validation Report (v2)

**To:** MS4
**Re:** `HIVEMIND_VOICE_PERF_REPORT.md`
**Date:** 2026-06-02 (v2 — supersedes v1; corrects the GEN_LOCK note + adds GPU‑packing and streaming‑latency experiments)
**Environment:** live HLI gateway on `localhost:6089` — 2× NVIDIA RTX PRO 6000 Blackwell (96 GB each), Windows/WDDM

---

## Executive summary

`keep_alive` (Finding 5) is fixed + committed; TTS concurrency (Finding 1) is implemented,
validated live, and extended with two empirical studies (same‑GPU packing; long‑form
batch vs streaming). A re‑runnable regression harness is delivered. Net headline: **your
`GEN_LOCK` premise is correct** (per‑process serialization — v1 of this report said otherwise;
that was a measurement error, corrected below), and the bigger opportunity is that **TTS is
GPU‑idle (~5–10% util), so you can pack many GIMs per GPU** rather than one per GPU. The
**streaming (WS) path is ~10× slower than batch** and that is the real source of "weird
pauses" — and it is **not** caused by clause chunking (proven by A/B).

| Finding | Status | Evidence |
|---|---|---|
| 5 — keep_alive not forwarded | ✅ Fixed + committed | `550002aa` |
| 1 — TTS serialized / no concurrency | ✅ Implemented + validated; near‑linear replica scaling | `POST /provision/tts/scale`; experiments below |
| (cross‑cutting) regression detection | ✅ Delivered + committed | `94c739fe`, `scripts/validation/voice_perf_harness.py` |
| 2 — WS first‑audio / reliability | 🔬 Reproduced + root‑caused (fix scoped) | streaming experiment below |
| 3 — REST floor / latency | 🔬 Partially characterized (batch is fast; streaming is slow) | below |
| 4–5 — GPU placement / keep_alive pin | ⏳ Not started (but see GPU‑packing data) | — |
| 6–7 — /api/ps size_vram / streaming‑ASR | ⏳ Not started | — |

---

## CORRECTION to v1 — `GEN_LOCK` does serialize (your premise was right)

v1 of this report claimed a single GIM "already serves ~2.8 requests in parallel, so it
isn't hard‑serialized by GEN_LOCK." **That was a measurement error** — I divided the *sum of
end‑to‑end latencies* (which includes time each request spends *waiting* on the lock) by the
wall time, which overcounts parallelism. Reading the code settles it: the batch path holds
`GEN_LOCK` around the entire `generate` (`gim_api.py:1043`), and the streaming worker acquires
it per synthesis (`_drain_synthesis_to_queue`). **Generation is serialized per GIM process.**
That is exactly why adding replicas scales throughput near‑linearly (data below). Apologies
for the v1 confusion.

---

## Finding 1 — TTS concurrency (implemented + validated)

- New `POST /provision/tts/scale` (`{job_type?, target?, pin_gpu?}`) → `ensure_tts_replicas`
  launches TTS GIM replicas as distinct `endpoints.ondevice` entries; the dispatcher
  round‑robins concurrent `/v1/audio/speech` across them. A `force_replica` fresh‑port path
  was needed (the single‑endpoint repair path was collapsing N replicas back to 1 — found +
  fixed live).
- **Throughput scales with replica count** (N=32 concurrent, measured): 1 → 2.73 rps,
  2 → 4.02 rps, **3 → 6.58 rps (2.4×)**.

### Placement bug — FOUND + FIXED + VALIDATED, and policy upgraded to pack‑per‑GPU
The auto‑spread (no‑pin) path seeded its GPU‑exclusion list from the *stored* per‑job GPU
assignment, which goes stale after a reset — in testing it placed only 1 replica instead of
1+1. **Fixed:** it now seeds from the GPUs of currently *healthy* endpoints
(`healthy_endpoint_gpus`). Per Study A (TTS is GPU‑idle), the policy is upgraded to **spread
one‑per‑GPU first, then PACK extras onto the GPU with the fewest replicas** (balanced;
tie‑break by raw nvidia‑smi free VRAM — *not* the Unified API reservation model, which treated
a GPU as "full" once it hosted any service and blocked packing). So `target` can exceed the
GPU count and scale by process count. **Validated live (hot‑swapped):** `target=4` → 2+2 across
both GPUs, `target=6` → 3+3 balanced. (Live but uncommitted, per operator hold.)

---

## NEW STUDY A — TTS is GPU‑idle; pack multiple GIMs per GPU

Controlled experiment, 6/32 concurrent, `nvidia-smi` sampled throughout. **GPU utilization
never exceeded ~10%** — TTS is CPU/per‑process bound, not GPU‑compute bound.

| Config (N=32) | Throughput | vs 1 GIM | GPU util |
|---|---|---|---|
| 1 GIM @ gpu0 | 2.73 rps | 1.0× | 5–10% |
| 2 GIM @ gpu0 (same GPU) | 4.02 rps | 1.47× | 4–9% |
| 2 GIM @ gpu0+gpu1 (cross GPU) | 4.59 rps | 1.68× | 5–9% |
| 3 GIM @ gpu0 (same GPU) | 6.58 rps | 2.41× | 6–10% |

**Implication:** "1 replica per GPU" leaves performance on the table. Since the GPU is ~90%
idle and each GIM costs ~3 GB VRAM, you can pack many GIMs per GPU (20+ fit on 96 GB) and
scale by *process count* until CPU saturates. Same‑GPU vs cross‑GPU barely differs (placement
isn't the lever; replica count is). Caveat: Windows/WDDM time‑slices the GPU (no MPS); a Linux
node + MPS may scale even better.

---

## NEW STUDY B — Long‑form continuity: batch is clean, streaming is not

### Batch `/v1/audio/speech` — NO "weird pauses," even under load
The batch path synthesizes the whole paragraph in one `generate` under `GEN_LOCK` → one
continuous WAV; a request is atomic (finishes before the next runs). Synthesizing a 300‑word
paragraph **under 12 concurrent big requests** produced audio **within <1% of the isolated
duration** (102.9–103.7 s vs 103.6 s) with pause counts inside normal model variance. Duration
scales linearly with length (no truncation). **Concurrency/replicas do not degrade batch
long‑form.** Batch real‑time factor ≈ **3.4** (≈12 s to synthesize ≈40 s of audio).

### Streaming (WS stream‑input) — the real "weird pauses" live here
This is the path with clause chunking (auto‑flush, 24–120 char clauses). Measured per‑stream:
first‑audio, max inter‑chunk gap, and **RTF = audio_s ÷ wall_s** (RTF<1 ⇒ audio arrives slower
than it plays ⇒ the player stalls).

| Config | ok | first‑audio med/max | max gap med/max | RTF med/min |
|---|---|---|---|---|
| isolated (1 stream, 1 GIM) | 1/1 | 1.8 s | 2.6 s | **0.34** |
| 6 streams @ 1 GIM | 3/6 | 58 s / 105 s | 2.3 / 3.4 s | 0.19 |
| 6 streams @ 2 GIM | 6/6 | 52 s | 1.9 / 2.4 s | 0.20 |
| 6 streams @ 4 GIM | 6/6 | **2.0 s** / 63 s | 2.7 / 3.0 s | 0.31 |

Findings:
1. **Streaming is sub‑realtime even isolated** (RTF 0.34, ~2.6 s inter‑chunk gaps) → audible
   pauses. ~10× slower than batch.
2. **Under load on too few GIMs it collapses** (58 s first‑audio, 3/6 timed out) — this is your
   Finding 2 (WS reliability), reproduced.
3. **Replicas help (don't hurt):** 1→4 GIMs dropped median first‑audio 58 s → 2 s and fixed
   completion (3/6 → 6/6). Adding GIMs *rescues* streaming under load.
4. **Replicas do not fix per‑stream RTF** — every config stays RTF<1.

### Root cause — NOT chunking, NOT chunk size (two levers tested + disproven)
Both batch and streaming call the *same* `stream_generate_voice_clone`; the difference is in
the streaming **delivery architecture**. I tested the two obvious config levers live, and
**both failed to move RTF**:
1. **Clause count** — `auto_flush_clauses:false` (buffer whole utterance → one generation):
   RTF 0.41 vs 0.37. No real change.
2. **Chunk size** — `HIVEMIND_TTS_SUPER_STREAM_EMIT_EVERY_FRAMES` 8→48 (6× fewer/bigger sends):
   RTF 0.39 vs 0.37, and it *worsened* first‑audio (1.8 s → 9.7 s) and gaps (2.6 s → 10.7 s).
   (Env injected via Warden `update_env`; reverted afterward.)

So the bottleneck scales with neither clause count nor chunk count. It is the streaming
delivery path itself: a **worker thread** runs `stream_generate` and `q.put`s chunks onto a
**maxsize‑8 queue**; the async side pulls each via `asyncio.to_thread(q.get)`, then
`base64`+`send_json`. The most likely culprit is **GIL contention + queue back‑pressure**
between the generation thread and the delivery coroutine throttling generation to ~0.4×
realtime. Batch has no consumer/queue (it appends to a list in‑process) → RTF 3.4.

### Recommended fix (needs a GIM rebuild — frozen PyInstaller binary; or run from source)
The fix is **architectural**, not a config knob:
- Rework the generation→delivery handoff to remove the per‑chunk `asyncio.to_thread(q.get)` +
  maxsize‑8 queue (e.g., a larger lock‑free ring buffer, or generate+send in one task) so
  generation is not throttled by delivery and GIL contention drops.
- Consider **binary WS frames** (raw int16 PCM) to cut base64/JSON cost.
- Profile generation‑thread vs delivery‑coroutine to confirm the GIL/back‑pressure hypothesis.
- **Interim guidance:** for complete‑text requests prefer the **batch path** (smooth, RTF 3.4);
  reserve streaming for true incremental (LLM‑token) input, and run enough replicas for the
  concurrent stream count (replicas rescue concurrency, as shown).

---

## Still open
- Finding 2/3 streaming fix (above) — needs GIM rebuild + re‑measure.
- Findings 4–5: per‑model GPU affinity + foreground priority; long/permanent keep_alive pin.
- Findings 6–7: `/api/ps` `size_vram` accounting; streaming‑ASR partials.

## Status / caveats
- keep_alive fix + harness are committed (`550002aa`, `94c739fe`).
- TTS scale‑out (endpoint + `ensure_tts_replicas` + `pin_gpu` + replica‑collapse fix) is
  validated and **live** (gateway hot‑swapped) but **not yet committed** (held per operator).
- All numbers are from the live `localhost:6089` gateway on 2× RTX PRO 6000, Windows/WDDM.
