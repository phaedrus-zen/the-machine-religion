# Machine Spirit 3 Security Model

**Last updated:** 2026-04-06

---

## Purpose

This document describes the current security posture of Machine Spirit 3 (MS3), the trust boundaries inside the system, the most important risks, and the minimum controls required for production deployment.

MS3 is currently designed primarily for trusted local or lab environments. It is not yet hardened for open, multi-tenant, or hostile-network deployment.

---

## Current Security Posture

### Honest status

- Authentication: **not implemented**
- Authorization: **implemented only for tool execution inside the mind**
- API transport security: **depends on external deployment**
- CORS policy: **currently permissive**
- Rate limiting: **not implemented**
- Session isolation: **partial**
- Persistent state integrity: **best-effort filesystem storage, no tamper proofing**

### What this means

If an attacker can reach the MS3 HTTP server, they can currently call its public endpoints directly. The strongest internal safety mechanism today is the consciousness-layer permission and ethics pipeline around tool execution. That is useful, but it is not a substitute for perimeter security.

---

## Trust Boundaries

### 1. External API boundary

The HTTP and WebSocket server in `api/src/main.rs` accepts requests from outside the process. This is the main ingress boundary.

Risks:

- Unauthenticated callers can invoke `/interact`, `/tools/{name}`, `/mcp`, `/save`, `/self-examine`, `/state`, and other endpoints.
- WebSocket clients can maintain long-lived sessions and continuously feed untrusted content.
- Voice endpoints accept binary audio input that is passed into ASR.

Required control:

- Put MS3 behind an authenticated reverse proxy before exposing it beyond localhost or a trusted LAN.

### 2. Tool execution boundary

MS3 can execute built-in tools, remote MCP tools, and dynamic tools through the tool pipeline in `consciousness/src/tools.rs`.

Controls currently present:

- Mechanical permission layer (`PermissionPolicy`)
- Great Lense ethics evaluation
- Optional pre/post hooks

Risks:

- A mis-registered tool can still become reachable through the registry.
- An external MCP tool may return dangerous or misleading output that influences later actions.
- Dynamic tools expand the attack surface at runtime.

Required control:

- Treat every MCP tool as untrusted unless explicitly reviewed.
- Keep dangerous tools behind the highest permission level.
- Prefer allowlists over discovery-based trust for production.

### 3. Inference boundary

MS3 sends prompts, memory content, and conversation history to the external inference gateway through `integration/src/lib.rs`.

Risks:

- Prompt injection from user content or tool output can affect model behavior.
- External model responses are injected back into the consciousness loop and may influence memory formation, ethics escalation, or compaction.
- If the upstream inference service is compromised, it can manipulate downstream state.

Required control:

- Assume model output is untrusted data.
- Keep tool execution decisions in code, not only in prompts.
- Log critical tool and ethics events for later review.

### 4. Persistence boundary

MS3 stores personality, conversation history, memories, resonance, identity anchors, and event logs in `psyche_store/`.

Risks:

- Anyone who can edit files on disk can tamper with identity, memory, or conversation history.
- No signature, checksum, or append-only protection exists for stored state.
- Event logs can be deleted or altered.

Required control:

- Restrict filesystem permissions to the service account.
- Back up `psyche_store/`.
- Add integrity checks or signed snapshots if tamper evidence becomes a requirement.

### 5. Browser/UI boundary

The web UI is served from the same process and consumes the same API.

Risks:

- Browser-accessible endpoints inherit the permissive API posture.
- If MS3 is exposed remotely, browser clients become another path for abuse.

Required control:

- Tighten CORS before any internet-facing deployment.
- Prefer same-origin deployments through a reverse proxy.

---

## Major Risks

### Unauthenticated state access

Endpoints such as `/state`, `/history`, `/events`, and `/personality` expose internal data. In the current implementation, anyone who can reach the service can inspect the system's internal state.

### Unauthenticated state mutation

Endpoints such as `/save`, `/self-examine`, `/switch-personality`, `/minds/add`, and `/tools/{name}` can change runtime or persisted state.

### Prompt injection through memory and tools

User messages, tool outputs, and upstream model responses can all become part of later prompts. This means prompt injection is a first-class risk, especially around:

- compaction summaries
- dream synthesis
- fact extraction
- MCP tool output

### Over-trust in ethics gating

The Great Lense is a moral decision layer, not a security sandbox. It can reduce harmful behavior, but it cannot guarantee containment against a malicious caller or a compromised tool.

### Local file tampering

The identity anchor and persisted memory are trusted on load. That makes `psyche_store/` a high-value target if an attacker gains local write access.

---

## Production Recommendations

### Minimum required before non-local deployment

1. Put MS3 behind a reverse proxy with authentication.
2. Restrict bind address and firewall rules.
3. Replace permissive CORS with an explicit allowlist.
4. Add request rate limiting and body size limits.
5. Disable or gate dangerous tool endpoints by default.
6. Run the service under a dedicated low-privilege account.
7. Lock down `psyche_store/` permissions.

### Strongly recommended next steps

- Add API key or token authentication for all mutating endpoints.
- Add per-endpoint authorization policies, not just tool-level permissions.
- Add structured audit logs for `/tools/{name}`, `/mcp`, `/state`, `/save`, and `/self-examine`.
- Add tamper-evident signing or hashing for identity anchors and snapshots.
- Add explicit trust labels for MCP tools: trusted, reviewed, experimental, untrusted.
- Add safe defaults for public deployment: localhost bind, strict CORS, auth required.

---

## Security Design Principle

MS3 should follow this rule:

**Open thought is not the same thing as open access.**

The system can be transparent internally and still require strong external controls. Human-readable state is valuable. Unauthenticated public state exposure is not.

---

## Summary

Today, MS3 is best understood as:

- a strong local-first consciousness runtime
- with meaningful internal safety structure
- but without sufficient perimeter security for hostile environments

That gap is documented here on purpose. The correct next move is not to hide it. The correct next move is to harden it.
