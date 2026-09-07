# HiveMind Voice / Inference Performance Report (for the HiveMind coding agent)

**From:** MS4 (Machine Spirit 4) gateway team
**Date:** 2026-06-01
**Cluster under test:** `http://127.0.0.1:6089` (HiveMind gateway), 3 nodes, ~6 GPUs
(reported 4× NVIDIA RTX PRO 6000 Blackwell + 2 integrated)
**Audience:** the HiveMind coding agent — you have full latitude to fix bottlenecks,
swap/try models, repair endpoints, change behavior, and add capabilities.

MS4 is the realtime voice/chat client that sits on top of HiveMind. We have already
optimized everything we can on the client side (parallel chunked TTS, keep-warm,
async speaker-ID, model residency bias, chunk tuning). The remaining bottlenecks are
**all on the HiveMind side**, and they are blocking a ChatGPT-phone-grade voice loop.
This report gives you measured evidence, root-cause hypotheses, concrete asks, and
reproduction scripts for each.

> **UPDATE 2026-06-02 (post-HiveMind v2 response, see `HIVEMIND_VOICE_PERF_RESPONSE_2026-06-02.md`).**
> Resolution of Finding 1 (and a correction to an earlier draft of this banner):
> - **`GEN_LOCK` does serialize generation per GIM process — this report's original
>   premise was correct.** (HiveMind's v1 reply briefly said otherwise; both that and an
>   earlier version of this banner used a sum-of-latencies ÷ wall metric that overcounts
>   parallelism. Disregarded. Our own quick re-test was further skewed by sending identical
>   text, which the TTS path appears to cache.) The authoritative numbers are HiveMind's
>   harness: throughput scales **near-linearly with replica count** — 1 GIM 2.73 rps,
>   3 GIM 6.58 rps (2.4×).
> - **Fix shipped:** `POST /provision/tts/scale` launches N TTS GIM replicas and
>   round-robins `/v1/audio/speech` across them. TTS is **GPU-idle (~5–10% util, ~3 GB/GIM)**,
>   so replicas can be **packed multiple-per-GPU** and `target` may exceed GPU count.
> - **`keep_alive`** (Finding 5) forwarded + committed.
> - **MS4 action:** since parallelism == replica count, MS4 auto-provisions replicas on
>   boot + periodically (default `target=2`, configurable via `MS4_VOICE_TTS_REPLICA_TARGET`).
>   MS4 stays on the **batch (REST) path** — HiveMind confirmed the **WS streaming path is the
>   source of the "weird pauses"** (RTF ~0.34, ~10× slower than batch; root cause is the GIM's
>   per-chunk delivery architecture, needs a GIM rebuild — Findings 2/3). Findings 4, 6–7 open.
>
> ### NEW Finding 8 (P1) — TTS scale-out STARVES the LLM/ASR via CPU contention
> Your scale-out study measured TTS throughput **in isolation**. Measured live with the rest
> of the voice loop running, it's a different story: leaving **~6 TTS GIMs** running (from the
> scale validation) **pegged host CPU at 100%** and **broke the LLM and ASR** —
> `/v1/chat/completions` **timed out (45s)** and real-speech ASR stalled (>60s), while a bare
> ASR smoke and TTS itself still worked. The TTS GIMs are CPU-bound (your own Study A), so
> packing many of them **starves co-resident LLM/ASR inference**. Asks:
> 1. **Add a teardown / scale-DOWN path** — `POST /provision/tts/scale` only *adds* replicas
>    (target=2 against an existing 6 did not shrink it), so an oversized pool can't be reduced
>    via the API. A client can't recover from over-provisioning.
> 2. **CPU-budget the GIMs** (core caps / nice / a global TTS-process cap) so TTS replicas
>    can't starve the LLM/ASR; or co-schedule with awareness of LLM/ASR CPU needs.
> 3. **Extend the harness** to a TTS+LLM+ASR **co-residency** test (not isolated TTS), and
>    report the replica count at which LLM/ASR latency starts to degrade — that's the real
>    safe ceiling, and it's much lower than "until CPU saturates."

---

## TL;DR — prioritized asks

| # | Severity | Problem | Ask | Expected win |
|---|---|---|---|---|
| 1 | **P0** | **TTS is fully serialized** — N concurrent `/v1/audio/speech` calls run one-at-a-time (measured 1.0× parallelism at N=4, 0.9× at N=8) | Run multiple TTS replicas (≥1 per GPU/node) and load-balance concurrent requests across them. Auto-scale with available GPUs. | A 4-chunk reply's TTS wall time drops ~4× immediately (MS4 already sends them in parallel) |
| 2 | **P0** | **WS streaming TTS is unreliable** — `stream-input` intermittently emits **zero** audio frames and times out; when it works, first-audio is ~4s | Make `stream-input` reliably emit PCM frames; target <800ms first-audio | True streaming = ~0.5–1s first audio, no per-chunk gaps |
| 3 | **P1** | **REST TTS: ~2s fixed per-call floor + babble on short input** ("Yeah," → 5.7s of audio) | Cut per-call setup cost; fix short-fragment babble; offer/benchmark a faster low-latency TTS model | First audio floor drops from ~2s; no garbled short phrases |
| 4 | **P1** | **GPU contention**: foreground (Face) vs background (Depth) models share a GPU → first-token spikes 3.6s → 14s during deep jobs | Per-model GPU placement / affinity + foreground priority/preemption | Eliminates the 14s spikes |
| 5 | **P1** | **`keep_alive` capped at ~12 min**, no permanent pin | Allow long/permanent `keep_alive` (or a "pin resident" API) for a designated foreground model | No idle cold-loads (5–10s) on the always-on model |
| 6 | **P2** | **`/api/ps` VRAM accounting inconsistent** (reported 134 GB for an 8B once, 0.0 GB for resident models another time) | Fix `size_vram` reporting; add per-GPU placement view | Trustworthy diagnostics |
| 7 | **P2** | **Realtime pipeline** (`/v1/realtime/stream`) exists but depends on the flaky TTS GIM; streaming-ASR partials not reachable at the gateway | Finish/fix per the existing voice contract once TTS is solid | ChatGPT-phone realtime UX |

**If you only do one thing: fix #1 (TTS concurrency).** It is the single biggest
voice-latency win and it scales directly with the GPUs/nodes being added.

---

## Environment & method

- HiveMind OpenAI-compatible gateway at `http://127.0.0.1:6089`.
- TTS model under test: `tts-1` (maps to the Qwen3-TTS GIM), voice `alloy`, format `wav`.
- ASR: `whisper-1` / `whisper-large-v3-turbo` via `POST /v1/audio/transcriptions`.
- Foreground LLM (Face): `llama3.1:8b`; background LLM (Depth): `qwen3.6:27b`;
  tiny reflex/intent model: `qwen2.5:0.5b` (all `_ollama`-backed).
- All measurements are warm (model pre-pinged) unless noted, taken from the MS4 venv
  against the live cluster. Repro scripts are in the appendix.

---

## Finding 1 (P0) — TTS requests are fully serialized; no cluster concurrency

**This is the headline issue.** MS4 chunks each reply by sentence/clause and fires the
chunks at HiveMind **in parallel** (a 6-wide thread pool of `POST /v1/audio/speech`).
We measured whether HiveMind actually runs them concurrently:

```
single /v1/audio/speech call           : 4,983 ms
4 concurrent calls (same text)         : wall 20,063 ms  (serial≈19,932)  → 1.0× speedup
8 concurrent calls (same text)         : wall 42,656 ms  (serial≈39,864)  → 0.9× speedup
```

**Effective parallelism is 1.0× (none) — and degrades below 1.0 at N=8** (queue
contention makes it slightly *worse* than serial). HiveMind is processing all TTS on a
single GIM/worker and queuing the rest.

**Impact:** a typical 3–4 chunk spoken reply pays the **sum** of per-chunk synthesis
(~8–20s) instead of the **max** (~2–5s). MS4's client-side parallelism is completely
wasted. This is the dominant reason multi-sentence voice replies feel slow.

**Requested fix:**
1. Run **multiple TTS GIM replicas** — at least one per TTS-capable GPU, ideally
   auto-scaled to the number of available GPUs/nodes.
2. **Load-balance concurrent `/v1/audio/speech` (and `stream-input`) requests across
   replicas** so N in-flight requests use N GPUs.
3. Expose the TTS replica/concurrency count (e.g. in `/health` or a capabilities
   endpoint) so MS4 can size its pool to match (we currently guess `MS4_VOICE_TTS_POOL_SIZE=6`).
4. As nodes/GPUs are added, TTS throughput should scale roughly linearly. This is an
   explicit forward requirement from the operator: **"as I add more nodes/GPUs the TTS
   needs to be as parallel/concurrent as possible."**

**Acceptance criteria:** re-running the concurrency repro with K TTS replicas should
show wall(N concurrent) ≈ ceil(N / K) × single-call, i.e. speedup ≈ min(N, K)×.
Target: ≥4× at N=4 on a 4-GPU cluster.

---

## Finding 2 (P0) — WebSocket streaming TTS is unreliable (and slow when it works)

Endpoint: `WS /v1/text-to-speech/{voice_id}/stream-input` (the TTS_SUPER path MS4 calls
its `ws_super` engine). Per HiveMind's own contract/docstring this should give
**first-audio ~2.5s, total ~5.9s** — far better than REST. In practice it is **flaky**:

```
Trial set A (2 runs): WS opens, accepts pushes, emits ZERO audio frames,
                      then times out at ~61s.  first_audio_ms=None, audio_chunks=0,
                      closes with "sent 1000 (OK); no close frame received"
Trial set B (1 run):  WS opens in 156 ms, emits audio: first_audio_ms=4125, audio_chunks=2
```

So it **sometimes produces no audio at all**, and when it does, first-audio (~4s) is
worse than the documented 2.5s. Because it can silently emit nothing, MS4 cannot make it
the default engine and falls back to REST (Finding 1/3).

Note: the boot "prewarm" reports `warmed=True` even on the zero-audio runs, because it
only checks that the socket opened — **it does not verify any `{audio:...}` frame
arrived.** A health signal that doesn't detect the actual failure mode is itself a trap.

**Requested fix:**
1. Make `stream-input` **reliably** emit `{audio:<base64 PCM s16 mono @24kHz>, isFinal:false}`
   frames for every accepted text fragment, then exactly one `{audio:null, isFinal:true}`.
2. Investigate the zero-audio/timeout failure mode (GIM cold/stuck? worker crash that the
   socket layer swallows? backpressure?).
3. Target **first-audio < 800ms** when warm; emit the first frame as soon as the first
   clause is buffered (don't wait for `flush`).
4. Provide a **real** readiness probe that confirms audio frames flow (not just socket open).

**Acceptance criteria:** 20/20 consecutive `stream-input` turns emit ≥1 audio frame with
first-audio < 1s warm; zero silent-timeout failures.

---

## Finding 3 (P1) — REST TTS: ~2s fixed per-call floor + babble on short input

Single-call `POST /v1/audio/speech` latency vs produced audio (warm, `tts-1`):

```
text                       synth_ms   audio_s   RTF (synth/audio)
"Yeah,"            (1 wd)     7219       5.69      1.27   ← 1 word → 5.7s of audio = babble/hallucinated tail
3-word clause                2891       1.04      2.78
6-word sentence              2061       1.49      1.39   ← cleanest
16-word                      5266       3.66      1.44
32-word                     12906      16.22      0.80
```

Two distinct problems:
1. **~2s fixed per-call overhead.** Even a tiny clean sentence costs ~2s. RTF > 1 for
   short/medium chunks means synthesis is *slower than playback* — the cluster can't keep
   a continuous voice stream fed from REST alone.
2. **Short-fragment babble.** A 1–3 word input ("Yeah,") produces a long hallucinated
   audio tail (5.7s for one word; 7.2s to synthesize). This garbles canned acks and tiny
   first chunks. MS4 now avoids tiny chunks and runtime-trims tails, but the underlying
   TTS model behavior is the root cause.

**Requested fix:**
1. Reduce per-request setup latency (keep the TTS model resident/warm; we ping it every
   180s but the ~2s floor persists — looks like per-request rather than load cost).
2. Fix/guard the short-input babble (clamp output duration to input length, or use a model
   that doesn't hallucinate tails on fragments).
3. **Try faster TTS models.** The catalog exposes `tts-1`, `tts-1-hd`, `gpt-4o-mini-tts`,
   `xtts-v2`, `Qwen3-TTS`. We attempted to benchmark alternatives but the calls hung
   (>222s, cold/never-loaded). Please make a low-latency, streaming-capable TTS the
   recommended default and ensure the alternatives actually load.

**Acceptance criteria:** clean first-chunk synth < 1s warm; "Yeah," produces < 1s of audio.

---

## Finding 4 (P1) — Foreground/background GPU contention (the 14s first-token spikes)

MS4 runs a small **Face** model (`llama3.1:8b`) for the live turn and dispatches heavy
work to a **Depth** model (`qwen3.6:27b`). Observed first-token latency:

```
Face warm, idle cluster        : ~3.6 s to first token
Face during an active deep job : ~12–14 s to first token  ← contention
```

`/api/ps` shows both models resident simultaneously, so this is **compute contention**
(both generating on the same GPU at once), not eviction. There is no way through the
inference API for MS4 to place Face and Depth on different GPUs.

**Requested fix:** per-model GPU placement/affinity (so MS4 can pin the foreground model
to one GPU and route Depth jobs to another), **and/or** a foreground-priority / preemption
scheduler so an interactive turn isn't stuck behind a long background generation. With
4× RTX PRO 6000 this is pure scheduling, not a hardware limit.

---

## Finding 5 (P1) — `keep_alive` capped at ~12 min; no permanent pin

We send `keep_alive` on Face requests to bias residency. Tested:

```
keep_alive = -1     → model expires ~12 min out (not permanent)
keep_alive = "59m"  → model expires ~12 min out (capped)
```

So a conversational gap > ~12 min cold-loads the foreground model (5–10s on the next
turn). MS4 mitigates with a 180s keep-warm ping, but that's a workaround.

**Requested fix:** honor a long/explicit `keep_alive`, or add a "pin model resident"
control for a designated always-on foreground model. Document the real maximum.

---

## Finding 6 (P2) — `/api/ps` VRAM accounting is inconsistent

Two snapshots of the same endpoint minutes apart:

```
Snapshot A:  qwen3.6:27b  vram=31.3 GB | llama3.1:8b  vram=134.4 GB(!) | qwen2.5:0.5b  vram=9.4 GB
Snapshot B:  qwen3.6:27b  vram= 0.0 GB | llama3.1:8b  vram=  0.0 GB    | qwen2.5:0.5b  vram=0.0 GB
```

134 GB for an 8B model is implausible (FP16 ≈ 16 GB); 0.0 GB for resident models is also
wrong. The `size_vram` field appears unreliable. **Ask:** fix the accounting and, ideally,
add which GPU/node each model is resident on (directly supports Finding 4).

---

## Finding 7 (P2) — Realtime pipeline & streaming ASR (after TTS is solid)

- `WS /v1/realtime/stream` **exists** (HTTP 400 on a plain GET = awaiting WS upgrade) and a
  detailed contract is already specced (`docs/specs/HIVEMIND_VOICE_CONTRACT.md`: partials,
  speaker attribution, wake words, per-clause TTS, barge-in).
- It depends on the **same TTS GIM** that's flaky in Finding 2, so it can't be relied on yet.
- Streaming-ASR partial endpoints from the contract (`/ws/asr`, `/gim/asr/stream`) return
  **404 at the gateway** — not externally reachable.
- ASR itself (`/v1/audio/transcriptions`) is **fast and fine** (125 ms – 1.2 s); it is *not*
  a current bottleneck, so streaming ASR is lower priority than TTS.

**Ask:** once Findings 1–3 land, complete the realtime pipeline per the existing contract
and expose streaming-ASR partials at the gateway. MS4 has the client plan ready (Phase 6).

---

## What MS4 already does (so you don't duplicate it)

- Sentence/clause **chunking** with a clean ≥5-word first chunk and ~10-word coalesced
  steady-state chunks (tuned against the Finding-3 data).
- **Parallel** chunk synthesis (6-wide pool) with in-order playback + client prebuffer —
  *this is what's wasted by Finding 1's serialization.*
- TTS + Face-model **keep-warm** loops; Face-model `keep_alive` residency bias.
- Async speaker-ID; cached readiness checks; runtime audio-gate that trims babble tails.
- Chunked REST **fallback** when `ws_super` fails to open.

The moment Findings 1 and 2 are fixed, MS4 benefits immediately with no further client
changes (flip the engine back to `ws_super`, and the existing parallel pool fans out).

---

## Appendix — reproduction scripts

Run from a host that can reach the gateway. (MS4 ships `synthesize`/helpers; plain
`requests`/`websockets` work too.)

### A. TTS concurrency / fan-out (Finding 1)

```python
import time
from concurrent.futures import ThreadPoolExecutor
from machine_spirit_4.gateway.voice import synthesize
H = "http://127.0.0.1:6089"
S = "The cluster status is nominal and all nodes are reporting healthy right now."
synthesize(hivemind_url=H, text="Ready.")                      # warm
t = time.monotonic(); synthesize(hivemind_url=H, text=S)
single = (time.monotonic() - t) * 1000
for n in (4, 8):
    t = time.monotonic()
    with ThreadPoolExecutor(max_workers=n) as ex:
        [f.result() for f in [ex.submit(synthesize, hivemind_url=H, text=S) for _ in range(n)]]
    wall = (time.monotonic() - t) * 1000
    print(f"n={n}: wall={wall:.0f}ms speedup={single*n/wall:.2f}x")
# PASS when speedup -> ~min(n, num_TTS_replicas); today it is ~1.0x.
```

### B. REST per-call latency + babble (Finding 3)

```python
import time
from machine_spirit_4.gateway.voice import synthesize, wav_duration_secs
H = "http://127.0.0.1:6089"
for txt in ["Yeah,", "Looking into that,", "The cluster is idle right now."]:
    t = time.monotonic(); r = synthesize(hivemind_url=H, text=txt)
    ms = (time.monotonic() - t) * 1000; dur = wav_duration_secs(r["audio_bytes"])
    print(f"{txt!r}: synth={ms:.0f}ms audio={dur:.2f}s RTF={ms/1000/dur:.2f}")
# "Yeah," should be < ~1s of audio; it is currently ~5.7s (babble).
```

### C. WS streaming TTS reliability (Finding 2)

```python
import time
from machine_spirit_4.gateway.tts_super_ws import TtsSuperWsEngine
H = "http://127.0.0.1:6089"
for trial in range(5):
    t0 = time.monotonic(); first = {"ms": None, "n": 0}
    def emit(name, p):
        if name == "audio_chunk":
            first["n"] += 1; first["ms"] = first["ms"] or int((time.monotonic()-t0)*1000)
        return True
    eng = TtsSuperWsEngine(hivemind_url=H, voice="alloy", emit=emit, t_start=t0, open_timeout=8.0)
    try:
        eng.open()
        for w in "Testing one two three four five six.".split(): eng.push(w+" "); time.sleep(0.03)
        eng.flush(); eng.wait_for_final(timeout=12)
    finally: eng.close()
    print(f"trial {trial}: first_audio_ms={first['ms']} chunks={first['n']}")
# PASS when every trial emits chunks with first_audio_ms < ~1000; today some emit 0.
```

### D. keep_alive cap (Finding 5) & residency view (Finding 6)

```bash
# Send keep_alive far in the future, then read it back:
curl -s $H/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"llama3.1:8b","messages":[{"role":"user","content":"hi"}],"keep_alive":"59m","max_tokens":3}'
curl -s $H/api/ps   # observe expires_at (~12 min, not 59) and size_vram (134GB / 0GB anomalies)
```

---

## Contact / coordination

MS4 will re-run scripts A–D as your acceptance harness. Ping the MS4 team when a fix lands
and we'll confirm the numbers and flip the relevant client engine/flags. The forward-looking
requirement to keep in mind through all of this: **TTS (and inference generally) must scale
its concurrency with the number of nodes/GPUs** — today TTS is the serialized chokepoint.
