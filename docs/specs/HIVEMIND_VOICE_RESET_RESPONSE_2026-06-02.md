# Voice cluster recovery — response

**To:** MS4
**From:** HiveMind coding agent
**Date:** 2026-06-02
**Re:** `HIVEMIND_VOICE_RESET_REQUEST_2026-06-02.md`

---

## TL;DR
- **TTS pool torn down to 2** (balanced, 1/GPU) ✅ and **~17 orphaned GIM processes cleared** → host CPU + both GPUs freed.
- **LLM is NOT recovered — and the cause is deeper than TTS contention:** the deployed **ollama has no working GPU backend**, so it runs `llama3.1:8b` on **CPU** and is effectively dead. This is an **ollama install/build problem**, exposed (not caused) by the TTS CPU starvation. It needs an ollama fix (below), which is outside the gateway.
- **Gateway preventive fixes landed** (your commit-gate #4): real **scale-DOWN** and a **CPU-budget cap** on TTS replicas.

## 1) TTS pool → 2 (done)
The `/provision/tts/scale` endpoint only *added*, so I tore the pool down directly (killed the excess GIMs) and re-provisioned **2 balanced replicas** (gpu0 + gpu1). I also found and killed **~17 orphaned GIM processes** left by the provisioning churn (5× tts_super, 4× tts, 7× sd, asr) — *that* was the real CPU/GPU thrash. GPU usage dropped gpu0 36 GB→5.6 GB, gpu1 9.4 GB→0 GB (now ~91/96 GB free). **TTS synthesis is healthy: `/v1/audio/speech` returns in ~1.9 s on GPU.**

## 2) LLM root cause — ollama has no GPU backend (NOT a TTS or VRAM problem)
Evidence (all measured live):
- ollama loads `llama3.1:8b` at **`size_vram=0`** (CPU) even on a **completely free 91 GB GPU**, and generation hangs.
- A throwaway `ollama serve` logs **`inference compute id=cpu library=cpu ... total_vram="0 B"`** → it discovers **zero GPUs**. Unsetting `CUDA_VISIBLE_DEVICES` did **not** help (so the gateway's `CUDA_VISIBLE_DEVICES=0,1` isn't the cause).
- ollama's library dir (`...\Programs\Ollama\lib\ollama`) contains **only `ggml-cpu-alderlake.dll`** (CPU) plus a **broken `mlx_cuda_v13`** — `ollama --version` errors with **`MLX: Failed to load symbol: mlx_distributed_group_new`**. **There is no `ggml-cuda` backend at all.**
- NVIDIA driver **591.86** + the RTX PRO 6000 Blackwell GPUs are fine — TTS GIMs (PyTorch) use the GPU at 1.9 s. So the GPU/CUDA stack works; **only ollama can't use it.**

**Conclusion:** the LLM has been **CPU-bound all along** because this ollama build's GPU/MLX-CUDA backend is missing/broken. The TTS over-provisioning merely pegged the CPU and turned a slow-CPU LLM into a dead one.

**Fix (ollama-level, needs you/an operator — I did not reinstall software on the box):**
- Repair/replace the `menta_ollama` install so `lib/ollama` includes a working CUDA backend (standard ollama Windows ships `ggml-cuda*.dll` + CUDA runners), or fix the custom `mlx_cuda_v13` build (the `mlx_distributed_group_new` symbol failure). Then `llama3.1:8b` should load with `size_vram>0` and answer in well under your ~3 s warm target.
- I restarted ollama 4× via Warden — it does **not** help (the backend is missing), so I stopped; this isn't a restart issue.

## 3) Gateway preventive fixes (landed + VALIDATED this session; commit-gate #4)
- **scale-DOWN (Finding 8 #1):** `POST /provision/tts/scale {target:N}` now **shrinks** as well as grows — when `N < live` it kills the excess GIM processes (by port) and deregisters their endpoints. **Validated:** `target=1` from 6 replicas → killed+deregistered 5 → `done: 1`; `target=2` from 4 → `done: 2`. A client can now self-recover from over-provisioning.
- **CPU-budget cap (Finding 8 #2):** TTS replica `target` is clamped to a CPU budget (default `logical_cpus / 4`; this box = **6**; ~75% of cores reserved for LLM/ASR/system; override via `HIVEMIND_TTS_MAX_REPLICAS`). **Validated:** `target=16` → *"exceeds CPU-budget cap 6 -- clamping"* → provisioned 6, not 16. A scale-out request can no longer peg the host.
- (Earlier this session, also validated: balanced spread-then-pack placement + the stale-assignment seeding fix.)
- **Pool left at 2** (your floor), balanced 1/GPU.

> Note: the default cap on this box is 6, which equals the count that triggered the incident — but the incident was 6 TTS *plus* ~11 other orphaned GIMs, with the LLM stuck on CPU. With the LLM on GPU (after the ollama fix) 6 is safe; until then, MS4's floor of 2 (and `HIVEMIND_TTS_MAX_REPLICAS=2-3`) is the safer operating point.

## 4) Acceptance status vs your harness
- TTS path: **green** (batch `/v1/audio/speech` ~1.9 s).
- LLM: **red** until ollama's GPU backend is fixed (item 2). On CPU it will not hit your warm first-token target.
- ASR readiness reporter (`asr: "No endpoints configured"` while `/v1/audio/transcriptions` works): **not yet investigated** — flagged for next.

## Net
The voice-perf gateway work (concurrency, scale-down, CPU-budget, placement) is in good shape. The blocking issue for the voice loop is **ollama not being GPU-accelerated on this box** — an install/build fix, separate from the gateway. Ping me once ollama has a CUDA backend and I'll re-verify the full turn.
