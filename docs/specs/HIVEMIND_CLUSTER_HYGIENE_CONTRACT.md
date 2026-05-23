# HiveMind Cluster Hygiene Contract for MS4 Realtime UX

| Field | Value |
| --- | --- |
| Spec version | 1.0.0-draft |
| Owner | MS4 (TMR repo) |
| Implementer | HiveMind coding agent (`E:\HiveMind`) |
| Date | 2026-05-22 |
| Status | Proposed, awaiting HiveMind ack |
| Sibling | `c:\Users\nexus-hc-win-00\Downloads\TMR\docs\specs\HIVEMIND_VOICE_CONTRACT.md` |

This document is the single source of truth for the cluster-hygiene and routing fixes MS4 needs HiveMind to provide so the streaming voice UX (sentence-chunked TTS, `/voice/turn/stream`) reaches consistent ChatGPT-mobile latency.

A sibling copy SHOULD be delivered into HiveMind at `docs/specs/HIVEMIND_CLUSTER_HYGIENE_CONTRACT.md` so both repos hold the same version.

Keywords MUST, SHOULD, MAY, MUST NOT follow [RFC 2119](https://www.rfc-editor.org/rfc/rfc2119).

---

## 1. Goals

1. Eliminate the background scatter/probe storm that consumes cluster cycles without producing a successful provisioning decision (~80% of current menta_hli log volume).
2. Stop probing nodes that have been confirmed dead so resources go to real work.
3. Give MS4 a way to **pin per-request audio routing to the local ondevice TTS_SUPER** so per-call latency variance stays bounded, instead of being subject to LAN router decisions made under scatter pressure.

Success means: after HiveMind ships this contract, MS4's `/voice/turn/stream` measures `tts_parallelism.speedup_ratio` ≥ 2.0 on 3-chunk turns reliably, with `max_held_for_inorder_ms` < 2000ms, instead of today's 1.19× and 28906ms (live observed on 2026-05-22).

## 2. Scope

In scope:
- Storage-quota probe behavior (`GET /storage/quota`) and the scatter cooldown logic in `menta_hli/.../scatter`.
- Dead-node eviction from cluster topology after sustained probe failures.
- An opt-in routing hint on `/v1/audio/speech` and `/v1/audio/transcriptions` to prefer the local ondevice instance.

Out of scope (explicitly):
- Changes to the model catalog shape (`/v1/models` `hivemind_status`, `hivemind_reachable`) — MS4 is already handling those correctly.
- Changes to MCP server or `/v1/realtime/stream` (covered by `HIVEMIND_VOICE_CONTRACT.md`).
- New telemetry endpoints.
- Cluster autoscaling or capacity policy changes beyond what's needed to land R1.

## 3. Evidence (observed live 2026-05-22 04:20:57Z–04:21:25Z)

These four log patterns sustained throughout the capture; not isolated spikes. Source: `c:\HiveMind\menta_warden\logs\menta_hli.log` lines 126618–126917 (~700 lines, ~28 seconds wall-clock).

**Pattern A — broken storage quota probe causing every scatter to skip:**

```
[scatter] Skipping qwen3-coder-next:latest on 192.168.0.14: storage quota probe failed:
  GET http://192.168.0.14:6089/storage/quota cooldown (21s remaining after recent failure)
[scatter] Skipping tinyllama:1.1b on 192.168.0.14: storage quota probe failed:
  GET http://192.168.0.14:6089/storage/quota cooldown (20s remaining after recent failure)
[scatter] Skipping llama3.1:latest on 192.168.0.14: storage quota probe failed:
  GET http://192.168.0.14:6089/storage/quota cooldown (20s remaining after recent failure)
... (same shape, same .14, same cooldown countdown, ~80 lines/min)
```

Same pattern on `.196`, `.199`, `.23`. Net: scatter cycles fire every 2-3s but make no decisions because the probe never recovers.

**Pattern B — dead nodes still iterated every scatter cycle:**

```
[scatter] Skipping dead node node-adfd3a7e2b72 (192.168.0.199)
[scatter] Skipping dead node node-77ee7c29-577 (192.168.0.23)
```

Logged ~20 times per scatter cycle (once per model × 2 dead nodes). No timestamp shows them ever rejoining or being removed from topology.

**Pattern C — provisioning 500 on phi4-mini and phi3.5:**

```
[scatter] Provisioning 'phi4-mini:latest' on 192.168.0.14 (GPUs=0, parallel=2)
[scatter] phi4-mini:latest provision failed on 192.168.0.14: 500 Internal Server Error
[scatter] Retrying phi4-mini:latest on 192.168.0.14 (attempt 2)
... same 500 after retry, no fallback to a different node
```

phi4-mini is MS4's Face Lobe pick. When the local Ollama on `.14` 500s and there's no alternative, downstream `/v1/chat/completions` either lands on a stressed instance or fails.

**Pattern D — TTS_SUPER local path works fine:**

```
TTS_SUPER ondevice native /v1/audio/speech success via http://127.0.0.1:49168/v1/audio/speech
Streaming job ca07c63d-... finalized: success=true, location=Device, endpoint=http://127.0.0.1:49168/v1/audio/speech
```

The local ondevice path is healthy. MS4's TTS latency variance is therefore not a TTS-engine problem; it's a routing problem (calls occasionally go LAN when they could go local).

## 4. Requirements

### R1. Storage-quota probe — fix or back off

The `/storage/quota` endpoint on `menta_hli` peer nodes MUST either succeed or fail in a bounded, low-frequency way. The current cooldown loop produces ~80 WARN log lines per minute while never advancing scatter state.

**R1.1** The scatter cooldown after a `/storage/quota` failure MUST use exponential backoff with a configurable cap:
- Initial cooldown: 30s (today: ~20s)
- Multiplier: 2.0 on each consecutive failure
- Cap: 300s (5 minutes)
- Reset to initial on first successful probe.

**R1.2** When `/storage/quota` returns the same error class (connection refused, 5xx, timeout) `N` consecutive times (default `N=10`), the peer MUST be flagged as `storage_unknown` for that error window and scatter MUST proceed with a heuristic budget (e.g. "assume node has half its last-known free VRAM") instead of blocking the whole scatter cycle for that node. A single WARN log line per backoff-window-entry is enough; do not log per-probe-attempt.

**R1.3** Per-node, per-window log throttling: emit at most 1 WARN per peer per backoff window for the storage-quota failure, not 1 per skipped model. Today's per-model logging amplifies a 4-node failure into ~80 lines per cycle.

**R1.4** If the `/storage/quota` endpoint is genuinely missing on a peer (404), the peer MUST be marked `storage_capability_missing` and excluded from scatter that requires the probe for the entire process lifetime (or until restart). One INFO line per peer per restart, not WARN-per-cycle.

**R1.5** Telemetry: add a `storage_quota_state` field per peer to `/cluster/summary` (or a new `/cluster/probe-health` endpoint) so MS4 and operators can observe peer probe health without grepping logs.

### R2. Dead-node eviction

A node marked `dead` MUST be evicted from the active scatter topology after a bounded grace period so it stops generating per-cycle "Skipping dead node" log entries and stops consuming probe budget.

**R2.1** When a node has been continuously `dead` (i.e. `routing::discovery` has not seen a mDNS heartbeat from it) for ≥ 300s, the node MUST be removed from:
- the scatter target list,
- any LAN endpoint registrations that point at it (already partially done via `[CarrierSync] pruned N stale on_demand_* LAN endpoint(s)` — extend that pruner to the topology layer too),
- the `Skipping dead node` log path (no per-cycle re-logging).

**R2.2** Rediscovery: when the discovery layer sees the node's mDNS again, it MUST be re-added to the topology in the standard "new peer joined" path (today's `[discovery] Added peer` flow), with one INFO line per join.

**R2.3** Configuration: `dead_node_eviction_grace_secs` MUST be operator-tunable, default 300, minimum 60.

**R2.4** Observability: `/cluster/summary` MUST report `evicted_nodes` (list of node_ids that were once active but are currently evicted) so MS4's UI/health pills can show "3 nodes evicted (grace window elapsed)" instead of guessing.

### R3. Per-call routing hint on audio endpoints

MS4 needs a way to request that a given `/v1/audio/speech` or `/v1/audio/transcriptions` call be served by the **local ondevice instance** rather than letting the router pick. Today MS4 has no way to express this preference and observed a 33-second TTS outlier on a 13-character sentence (one call out of three in a parallel batch) — almost certainly LAN routing under scatter pressure.

**R3.1** `/v1/audio/speech` MUST accept an optional query parameter or JSON body field:

```http
POST /v1/audio/speech?route=local_first
Content-Type: application/json

{
  "model": "tts-1",
  "voice": "alloy",
  "input": "Hello.",
  "response_format": "wav",
  "route_hint": "local_first"     // alternative to query param; both MUST be accepted
}
```

Allowed `route` / `route_hint` values:
- `local_first` (default behavior with hint set): try the ondevice instance first; if it returns non-2xx or is unavailable within a 500ms attempt budget, fall back to LAN.
- `local_only`: ondevice only; return `503` immediately if no ondevice TTS_SUPER is reachable. Useful for MS4 to detect "cluster only" deployments.
- `lan_balanced`: today's default behavior, kept as the fallback for callers that omit the hint.
- `lan_only`: skip ondevice; force LAN. Useful for testing the LAN path.

**R3.2** `/v1/audio/transcriptions` MUST accept the same `route` query parameter / `route_hint` field with the same allowed values.

**R3.3** The response MUST include a header (or, for streaming, a final-metadata event) reporting which route actually served the call:

```http
X-HiveMind-Route: local
X-HiveMind-Audio-Endpoint: http://127.0.0.1:49168/v1/audio/speech
X-HiveMind-Route-Decision: local_first_hit_local
```

`X-HiveMind-Route-Decision` values: `local_first_hit_local`, `local_first_fell_back_to_lan`, `local_only_succeeded`, `local_only_unavailable`, `lan_balanced_chose_<peer_id>`, `lan_only_chose_<peer_id>`.

**R3.4** Default behavior when no hint is sent MUST be unchanged from today (`lan_balanced` semantics). All MS4 deployments that currently work MUST keep working without code changes.

**R3.5** The hint MAY be combined with the existing model selection. If `model="tts-1-hd"` is requested with `route=local_only` and the local TTS_SUPER does not support that model, return `503` with body `{"error":"local_only requested but model not available locally","model":"tts-1-hd"}`. Do not silently fall back when `local_only` was explicit.

## 5. Acceptance tests

These are the minimum tests HiveMind SHOULD pass before MS4 closes this contract:

**T1** Storage-quota probe error reduction: cause `/storage/quota` to fail on one peer for 5 minutes. WARN log volume MUST drop to ≤ 6 lines for that peer over the window (one per backoff entry), not 80+ per minute.

**T2** Dead-node eviction: stop a known peer's `menta_hli`. After 300s, `/cluster/summary` MUST list it under `evicted_nodes` and the scatter log MUST stop emitting `Skipping dead node` entries for it.

**T3** Local-first routing hit:
```bash
curl -sS -X POST "http://127.0.0.1:6089/v1/audio/speech?route=local_first" \
  -H "Content-Type: application/json" -i \
  -d '{"model":"tts-1","voice":"alloy","input":"Hello.","response_format":"wav"}'
```
MUST return `X-HiveMind-Route: local` and `X-HiveMind-Route-Decision: local_first_hit_local` within p95 < 2s on a warm cluster.

**T4** Local-only fail-fast:
```bash
# With ondevice TTS_SUPER stopped
curl -sS -X POST "http://127.0.0.1:6089/v1/audio/speech?route=local_only" ...
```
MUST return `503` within 200ms, NOT block on a LAN attempt.

**T5** Default behavior preserved: calls without `route` query param or `route_hint` body field MUST behave bit-for-bit identically to today (same default routing decisions, same response headers, same latencies).

## 6. Non-requirements / explicit non-asks

- HiveMind is NOT being asked to add load-aware routing for TTS. Just expose the hint; MS4 will use `local_first` for foreground voice and `lan_balanced` (default) for background Depth Lobe work.
- HiveMind is NOT being asked to remove the scatter system or change its goals. R1 only asks for bounded retries and bounded log volume during sustained probe failures.
- HiveMind is NOT being asked to fix the underlying `/storage/quota` 500 root cause on `.14`/`.196`/`.199`/`.23` (that's a separate menta_hli bug); R1 only asks for graceful degradation when it fails.

## 7. Open questions

1. Should the `route` hint also apply to `/v1/chat/completions`? MS4 has the same per-call latency variance concern for chat (see TMR `Ms4TurnMetrics.v1.http_latency_ms` outliers). Out of scope for v1.0.0 of this contract — would be v1.1.0.
2. Should there be a global config knob to set the default hint for an entire MS4 deployment? Out of scope; MS4 can pass the hint per call.
3. Should the WARN throttling in R1.3 apply to other probe types (e.g. `/health`, `/capacity`)? Out of scope; same pattern can be applied if observed but R1 is specifically for `/storage/quota`.

## 8. Implementation notes (non-binding)

- The cooldown state in `menta_hli/.../scatter` is per-(node, probe). Extending it to track consecutive-failure-count and per-window-warn-throttle is mostly additive bookkeeping.
- The "evict dead node" logic can hook into the existing `routing::cluster` `[CarrierSync] pruned N stale on_demand_* LAN endpoint(s)` cleaner — same pattern, extended to the topology layer.
- The `route_hint` on audio endpoints can be implemented in `routing::aiaas` where the current `local_first` fallback for TTS_SUPER already happens (line 126774 of the observed log proves the ondevice-first code path exists internally; this requirement just promotes it to a caller-visible knob).
- All three changes SHOULD ship behind separate feature flags so they can be rolled back independently if a regression is found in one without affecting the others.

## 9. Change log

- v1.0.0-draft (2026-05-22): initial draft.
