# Oracle private-remote PWA runbook

Status: isolated Stage-A candidate only. This runbook does not authorize or
perform a live TMR deploy, Tailscale change, microphone capture, or camera use.

## Why this path

Tailscale Serve is the narrow first deployment path because it can terminate
HTTPS at the existing tailnet boundary and proxy only to the Oracle process on
loopback. That gives the browser the secure context required by service workers,
microphone APIs, and Screen Wake Lock without adding a public phone provider or
placing the Oracle on the public internet. Do not use Tailscale Funnel for this
stage.

The PWA caches only its static shell. Health, chat, voice, job, and conversation
responses are deliberately never cached; an offline shell must say the private
link is offline rather than replaying stale Oracle state.

## Current read-only gate (2026-08-11)

- Tailscale client: installed and running; the local node reports online.
- MagicDNS: enabled.
- Tailnet HTTPS: externally confirmed disabled; the local read-only status also
  reported no usable certificate domain.
- Existing Serve state: a live private HTTP-only listener on port 18080 proxies
  to `http://127.0.0.1:9180`.
- Backend lifecycle: the active local owner is the per-user
  `HiveMind Oracle.lnk` Startup watchdog. It runs
  `supervise_ms4.py --watch --interval-seconds 300` in the operator's
  interactive identity and starts only missing MS3 `:9080`, MS4 gateway
  `:9180`, and MS4 MCP `:9181` listeners. The
  `machine_spirit_4/warden_service.json` definition remains reference-only;
  MS4 is not currently Warden-managed. This proves a configured local restart
  path, not a secure browser origin, remote microphone path, or continuous
  300-second availability.
- Mutation performed by this candidate: none.

Therefore the current disposition is
`BLOCKED_TAILNET_HTTPS_DISABLED`. Plain tailnet HTTP is encrypted by Tailscale,
but browsers do not treat it as a secure web origin, so PWA installation,
service workers, wake lock, and remote microphone use cannot be accepted there.

## Gate 1: human enables tailnet HTTPS

Stop here if tailnet HTTPS is disabled or unproven. A tailnet administrator must:

1. Open the Tailscale admin console DNS page.
2. Confirm MagicDNS is enabled.
3. Enable HTTPS Certificates and acknowledge the certificate-transparency name
   disclosure described by Tailscale.
4. Confirm the intended Oracle machine name is suitable for that public
   certificate ledger.

Do not let an unattended agent click the consent page or change the tailnet.
Re-run the read-only status check and require a usable `*.ts.net` certificate
domain before continuing.

## Gate 2: deploy the reviewed candidate locally

1. Reconcile the live TMR preimages against the hashes in the candidate author
   receipt. Stop on any mismatch; do not overwrite newer v5 work.
2. Review and apply the sealed packet's `candidate.patch` to a new canary source
   tree.
3. Run the focused Node/Python checks and the selected real-browser synthetic
   regressions from the receipt.
4. Build and start only an Oracle canary bound to loopback, for example
   `127.0.0.1:9481`. Keep production `127.0.0.1:9180` unchanged.
5. Verify `/api/v1/ms4_gateway/status`, `/manifest.webmanifest`, the exact-build
   `/service-worker.js?build=...`, and all three exact-build `/static/` scripts
   locally. The status body must report `ok=true`, `service=ms4-gateway`,
   `status=ready`, and `runtime=ms4-fusion`. The service worker response must include
   `Service-Worker-Allowed: /` and `Cache-Control: no-store`.
6. Run the predecessor-to-successor browser upgrade gate. New HTML must never
   execute an older lifecycle module; the successor controller must perform
   exactly one reconciliation reload and leave only the successor static cache.

Physical microphone and camera validation remain a separate human gate. Use
synthetic WAV fixtures until the human explicitly says `ready`.

## Gate 3: private HTTPS canary

First save the current read-only Serve status. Do not reset Serve because this
machine already has an unrelated HTTP listener.

After human authorization and Gate 1 passes, add only the HTTPS canary mount:

```powershell
tailscale serve --bg --https=443 http://127.0.0.1:9481
tailscale serve status --json
```

Expected URL shape:

```text
https://<machine>.<tailnet>.ts.net/
```

From a second authenticated tailnet device, prove:

1. `/api/v1/ms4_gateway/status` returns the exact MS4 canary identity. An HTTP
   200 from a wrong or malformed same-origin backend must render the link as
   misrouted, not online.
2. `/manifest.webmanifest` has `application/manifest+json`.
3. `/service-worker.js` has JavaScript MIME, no-store, and root scope.
4. The header visibly reports `private link online`, a finite latency, `PWA
   ready`, and `oracle-pwa-stage-a/2026-08-11.v5`.
5. `window.isSecureContext === true`, the registered service-worker scope is
   the origin root, and both the active worker and current controller use the
   exact visible build query. `PWA ready` is invalid while merely installing,
   activating, uncontrolled, or generation-mismatched.
6. Reconnect uses one in-flight probe and the bounded 1, 2, 4, 8, 15 second
   backoff; success returns to a 15 second health cadence.
7. Install the PWA, close/reopen it, and confirm the shell loads while Oracle API
   state still fails truthful/offline when the canary is stopped.
8. With synthetic audio only, confirm a background Depth dispatch plays exactly
   one immediate optional pending cue when first tracked.

Only after those checks pass should a separately approved promotion point HTTPS
443 at `http://127.0.0.1:9180`.

## Wake lock and voice behavior

The candidate requests Screen Wake Lock only after full-duplex successfully
arms. It releases the lock when full-duplex disables, reacquires after a
visibility return, and requests exactly one replacement when the browser or OS
revokes the current lock while hands-free intent remains active and visible.
An explicit disable or hidden document cannot trigger that replacement.
Unsupported or denied wake lock is non-fatal; it never changes microphone
selection, VAD, barge-in, cancellation, or exact device binding.

Remote mobile browsers may suspend a PWA despite wake lock or OS power policy.
The connection pill and bounded reconnect make that condition visible; they do
not claim a browser tab is a 24/7 server. On this host, the per-user Startup
watchdog is the current lifecycle owner for `:9080`, `:9180`, and `:9181` on a
300-second cadence. The Warden definition is only a future/reference contract,
so this Stage-A candidate does not claim Warden-managed 24/7 availability.

## Narrow rollback

If the HTTPS canary fails, remove only its exact mount and leave the pre-existing
HTTP 18080 handler untouched:

```powershell
tailscale serve --https=443 --set-path=/ off
tailscale serve status --json
```

Then stop the canary and restore the source preimages. Do not use
`tailscale serve reset` because it would erase unrelated Serve configuration.

## Acceptance boundary

This Stage-A candidate proves installability plumbing, same-origin remote URL
construction, exact MS4 connection/latency identity, bounded reconnect,
wake-lock lifecycle, coherent service-worker upgrades, immediate synthetic
Depth waiting feedback, and visible build identity. It does
not prove live tailnet HTTPS, production deployment, physical voice quality,
background mobile push, PSTN calling, video, or 24/7 end-to-end availability.

## Primary references

- [Tailscale Serve](https://tailscale.com/docs/features/tailscale-serve)
- [Tailscale Serve CLI](https://tailscale.com/docs/reference/tailscale-cli/serve)
- [Tailscale HTTPS certificates](https://tailscale.com/docs/how-to/set-up-https-certificates)
- [MDN secure-context restricted features](https://developer.mozilla.org/en-US/docs/Web/Security/Defenses/Secure_Contexts/features_restricted_to_secure_contexts)
- [MDN Screen Wake Lock](https://developer.mozilla.org/en-US/docs/Web/API/WakeLock)
