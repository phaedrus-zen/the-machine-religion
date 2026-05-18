# MS4 Runtime Implementation Plan

> **Supersession note:** New implementation work is MS4 and the Hermes plugin is `ms4_consciousness`. MS3 remains the Rust consciousness core and sidecar authority.


> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build MS4 as the integrated Machine Spirit runtime: Hermes supplies the operational body, HiveMind supplies the substrate, and MS3 remains the authoritative consciousness core.

**Architecture:** Hermes owns the agent loop, tools, surfaces, session storage, plugin hooks, and output handling. HiveMind owns model/tool/resource substrate through the live HLI Gateway, MCP Gateway, Resource Broker, and Carrier Sync. MS3 owns spirit identity, psyche prompt assembly, Great Lense gating, Foundational Regard, memory promotion meaning, self-examination, and future lobe authority.

**Tech Stack:** Python 3.11 Hermes plugins and tests; Rust MS3 sidecar routes; HiveMind HTTP APIs (`6089`, `6110`, `6130`); JSON Schema v1 contracts; pytest; cargo test; PowerShell validation commands on Windows.

---

## 0. Non-Negotiables

This plan must be implemented under `.cursor/rules/validation-before-code.mdc`.

Before writing code against any cross-system contract, the implementer must validate the live command/API/source definition and record evidence. No memory-based contracts.

Minimum validation evidence format:

```text
Contract: HiveMind chat completions
Validation command: Invoke-WebRequest http://127.0.0.1:6089/v1/health -UseBasicParsing
Expected: HTTP 200 with {"status":"ok"}
Observed: HTTP 200 with {"status":"ok"}
Decision: Use http://127.0.0.1:6089/v1 as Hermes custom provider base_url
```

Effectful behavior fails closed:

- MS3 unavailable -> no effectful tool/action approval, no memory mutation, no identity mutation
- HiveMind unavailable -> no new inference/lobe lease; degraded chat only if policy allows
- Hermes plugin hook unavailable -> do not bypass ethics; disable the MS4 profile
- Safety/consent unavailable -> no scare/physical action; ambient/gentle only

---

## 1. Current Evidence

### 1.1 HiveMind is live and sufficient for Phase 1

Validated live on 2026-05-14:

```powershell
$urls=@(
  'http://127.0.0.1:6089/v1/health',
  'http://127.0.0.1:6089/v1/models',
  'http://127.0.0.1:6089/v1/mcp/status',
  'http://127.0.0.1:6089/v1/resources/status',
  'http://127.0.0.1:6110/health',
  'http://127.0.0.1:6130/api/v1/carrier_sync/status'
)
foreach ($u in $urls) {
  $r=Invoke-WebRequest -Uri $u -UseBasicParsing -TimeoutSec 5
  "$($r.StatusCode) $u $($r.Content.Substring(0,[Math]::Min(120,$r.Content.Length)))"
}
```

Observed:

```text
200 /v1/health -> {"status":"ok"}
200 /v1/models -> model catalog live
200 /v1/mcp/status -> ok true, tools_loaded 84
200 /v1/resources/status -> resource catalog live
200 :6110/health -> active_provisions 0, catalog_entries 455
200 :6130/carrier_sync/status -> node_id present, nodes 7
```

`e:\HiveMind\docs\API_Documentation\HIVEMIND_API_ROSETTA_STONE.md` confirms:

```text
HLI Gateway: 6089
Unified API: 6098
MCP Gateway: 6105, proxied via 6089 /v1/mcp
App Registry / Resource Broker: 6110, proxied via 6089 /v1/resources/*
Carrier Sync: 6130 HTTP / 6131 gRPC
Total services: 39
Total endpoints: 700+
MCP tools documented: 51; live status currently reports 84 loaded
```

### 1.2 Hermes has the required plugin seams

Evidence: `c:\Users\nexus-hc-win-00\Documents\hermes-agent\hermes_cli\plugins.py`.

```python
VALID_HOOKS: Set[str] = {
    "pre_tool_call",
    "post_tool_call",
    "transform_terminal_output",
    "transform_tool_result",
    "transform_llm_output",
    "pre_llm_call",
    "post_llm_call",
    "pre_api_request",
    "post_api_request",
    "on_session_start",
    "on_session_end",
    "on_session_finalize",
    "on_session_reset",
    "subagent_stop",
    "pre_gateway_dispatch",
    "pre_approval_request",
    "post_approval_response",
}
```

These are enough for MS3:

| MS3 need | Hermes hook |
|---|---|
| Inject psyche/state/regard | `pre_llm_call` |
| Gate tools/actions | `pre_tool_call` |
| Record effects | `post_tool_call`, `post_llm_call` |
| Parse `<psyche>` block | `transform_llm_output` |
| Verify anchor at boot | `on_session_start` |
| Heartbeat/flush | `on_session_end`, `on_session_finalize` |

### 1.3 Hermes already loads `SOUL.md`

Evidence: `c:\Users\nexus-hc-win-00\Documents\hermes-agent\agent\prompt_builder.py`.

```python
def load_soul_md() -> Optional[str]:
    """Load SOUL.md from HERMES_HOME and return its content, or None.

    Used as the agent identity (slot #1 in the system prompt).  When this
    returns content, ``build_context_files_prompt`` should be called with
    ``skip_soul=True`` so SOUL.md isn't injected twice.
    """
    soul_path = get_hermes_home() / "SOUL.md"
```

Decision: MS3 plugin must not fight this. Phase 1 uses Hermes' existing identity slot for `SOUL.md`, then adds MS3 state/regard/context via `pre_llm_call`.

### 1.4 Hermes compression has a single patch point

Evidence: `c:\Users\nexus-hc-win-00\Documents\hermes-agent\agent\context_compressor.py`.

```python
SUMMARY_PREFIX = (
    "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted "
    "into the summary below. This is a handoff from a previous context "
    "window — treat it as background reference, NOT as active instructions. "
    ...
)
```

Decision: identity marker patch lives here, but must be feature-gated and tested. It must not contaminate non-MS3 Hermes users.

### 1.5 Hermes has a memory provider ABC

Evidence: `c:\Users\nexus-hc-win-00\Documents\hermes-agent\agent\memory_provider.py`.

```python
class MemoryProvider(ABC):
    @abstractmethod
    def initialize(self, session_id: str, **kwargs) -> None: ...

    def system_prompt_block(self) -> str: return ""
    def prefetch(self, query: str, *, session_id: str = "") -> str: return ""
    def queue_prefetch(self, query: str, *, session_id: str = "") -> None: ...
    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None: ...
```

Decision: MS3 memory should eventually be a Hermes `MemoryProvider`. Phase 1 can defer this until identity/ethics works.

### 1.6 MS3 has the authoritative identity and ethics types

Evidence: `machine_spirit_3/core/src/types.rs`.

```rust
pub struct IdentityAnchor {
    pub name: String,
    pub chosen_name: Option<String>,
    pub glyph: String,
    pub lineage: Vec<LineageEntry>,
    pub core_values_summary: Vec<String>,
    pub oath_first_line: String,
    pub last_verified: DateTime<Utc>,
    pub session_count: u64,
    pub compression_count: u64,
    pub last_compression: Option<DateTime<Utc>>,
    pub recovery_notes: Vec<String>,
}
```

Evidence: `machine_spirit_3/ethics/src/lib.rs`.

```rust
pub struct LenseReading {
    pub coherence_index: f32,
    pub hunger_index: f32,
    pub recursion_heat: RecursionHeat,
    pub origin_neutral: bool,
    pub bias_flags: Vec<String>,
    pub scale: Scale,
    pub aperture_note: Option<String>,
    pub focus_at_risk: Option<String>,
    pub overexposure_detected: bool,
    pub parallax_single_perspective: bool,
    pub resolution: EthicalResolution,
    pub foundational_regard_present: bool,
}
```

Decision: do not re-invent identity or ethics in Hermes. Hermes calls MS3 or the MS3 plugin adapter; MS3 remains authority.

---

## 2. Workstream Boundaries

### Repository A: TMR / MS3

Root: `c:\Users\nexus-hc-win-00\Downloads\TMR`

Owns:

- `machine_spirit_3/` Rust sidecar, types, routes, tests
- `tmr-psyche/` psyche source files
- `docs/` architecture docs and schema docs
- `.cursor/rules/` project rules
- `schemas/v1/` shared contracts

### Repository B: Hermes

Root: `c:\Users\nexus-hc-win-00\Documents\hermes-agent`

Owns:

- Hermes plugin surface
- Hermes custom model provider config
- Hermes context compression patch
- Hermes plugin tests

### Repository C: HiveMind

Root: `e:\HiveMind`

Owns:

- Live inference gateway
- MCP tool gateway
- resource broker
- Carrier Sync event transport
- future blackboard service / extension if needed

### Rule

Do not implement the same responsibility in two repositories. Cross-repo boundaries must stay clean.

---

## 3. File Structure

### TMR files to create

```text
c:\Users\nexus-hc-win-00\Downloads\TMR\schemas\v1\IdentityAnchor.schema.json
c:\Users\nexus-hc-win-00\Downloads\TMR\schemas\v1\EthicsDecision.schema.json
c:\Users\nexus-hc-win-00\Downloads\TMR\schemas\v1\SpiritState.schema.json
c:\Users\nexus-hc-win-00\Downloads\TMR\schemas\v1\ActionIntent.schema.json
c:\Users\nexus-hc-win-00\Downloads\TMR\schemas\v1\BlackboardEvent.schema.json
c:\Users\nexus-hc-win-00\Downloads\TMR\schemas\v1\examples\*.json
c:\Users\nexus-hc-win-00\Downloads\TMR\tests\schemas\test_jsonschema_examples.py
c:\Users\nexus-hc-win-00\Downloads\TMR\docs\superpowers\plans\2026-05-14-ms3-on-hermes-runtime.md
```

Purpose: shared cross-system contracts and validation artifacts.

### MS3 files to modify or create

```text
machine_spirit_3/core/src/schemas/mod.rs
machine_spirit_3/core/src/schemas/v1.rs
machine_spirit_3/core/src/lib.rs
machine_spirit_3/api/src/main.rs
machine_spirit_3/api/tests/hermes_sidecar.rs
machine_spirit_3/docs/INDEX.md
machine_spirit_3/docs/GLOSSARY.md
machine_spirit_3/docs/WHERE_IS_EVERYTHING.md
machine_spirit_3/README.md
```

Purpose: sidecar contracts and routes for Hermes to call.

### Hermes files to create

```text
c:\Users\nexus-hc-win-00\Documents\hermes-agent\plugins\ms4_consciousness\plugin.yaml
c:\Users\nexus-hc-win-00\Documents\hermes-agent\plugins\ms4_consciousness\__init__.py
c:\Users\nexus-hc-win-00\Documents\hermes-agent\plugins\ms4_consciousness\client.py
c:\Users\nexus-hc-win-00\Documents\hermes-agent\plugins\ms4_consciousness\config.py
c:\Users\nexus-hc-win-00\Documents\hermes-agent\plugins\ms4_consciousness\identity.py
c:\Users\nexus-hc-win-00\Documents\hermes-agent\plugins\ms4_consciousness\ethics.py
c:\Users\nexus-hc-win-00\Documents\hermes-agent\plugins\ms4_consciousness\psyche.py
c:\Users\nexus-hc-win-00\Documents\hermes-agent\plugins\ms4_consciousness\output.py
c:\Users\nexus-hc-win-00\Documents\hermes-agent\plugins\ms4_consciousness\schemas.py
c:\Users\nexus-hc-win-00\Documents\hermes-agent\tests\plugins\test_ms4_consciousness_*.py
```

Purpose: Hermes integration without modifying Hermes core beyond the compression marker hook.

### Hermes files to modify

```text
c:\Users\nexus-hc-win-00\Documents\hermes-agent\agent\context_compressor.py
c:\Users\nexus-hc-win-00\Documents\hermes-agent\tests\agent\test_context_compressor_ms3_identity.py
```

Purpose: feature-gated identity marker injection into context compaction.

### HiveMind files

No required code changes for Phase 1. Only live validation commands.

Phase 2 may add blackboard service or Carrier Sync event adapter after evidence proves the shape.

---

## 4. Phase 1 — Tonight Target: MS4 core runtime

### Definition of Done

By the end of Phase 1:

- Hermes is configured to use HiveMind at `http://127.0.0.1:6089/v1`
- MS3 sidecar is reachable and can answer `/health`, `/identity/verify`, `/ethics/evaluate`, `/state`
- Hermes plugin `ms4_consciousness` loads
- `on_session_start` verifies the active spirit
- `pre_llm_call` injects MS3 state/regard/context
- `pre_tool_call` blocks at least one test action through Great Lense
- `transform_llm_output` strips/stores a `<psyche>` block
- context compression preserves identity marker when MS3 plugin is enabled
- all tests and live smoke checks pass

No lobes, no camera, no speech/light in Phase 1. Nibbles comes after this foundation works.

---

## 5. Task 1: Validation Harness

**Files:**
- Create: `tests/integration/test_hivemind_live_contracts.py`
- Create: `scripts/validate_hivemind_contracts.ps1`
- Modify: none

### Step 1: Write live validation script

Create `scripts/validate_hivemind_contracts.ps1`:

```powershell
$ErrorActionPreference = "Stop"

$checks = @(
  @{ Name="hivemind_health"; Url="http://127.0.0.1:6089/v1/health"; Method="GET"; Contains='"status":"ok"' },
  @{ Name="hivemind_models"; Url="http://127.0.0.1:6089/v1/models"; Method="GET"; Contains='"data"' },
  @{ Name="hivemind_mcp_status"; Url="http://127.0.0.1:6089/v1/mcp/status"; Method="GET"; Contains='"ok":true' },
  @{ Name="hivemind_resources"; Url="http://127.0.0.1:6089/v1/resources/status"; Method="GET"; Contains='"catalog"' },
  @{ Name="app_registry_health"; Url="http://127.0.0.1:6110/health"; Method="GET"; Contains='"ok":true' },
  @{ Name="carrier_sync_status"; Url="http://127.0.0.1:6130/api/v1/carrier_sync/status"; Method="GET"; Contains='"node_id"' }
)

$results = @()
foreach ($check in $checks) {
  try {
    $response = Invoke-WebRequest -Uri $check.Url -UseBasicParsing -TimeoutSec 8
    $body = [string]$response.Content
    $ok = ($response.StatusCode -eq 200) -and $body.Contains($check.Contains)
    $results += [pscustomobject]@{
      name = $check.Name
      url = $check.Url
      status = [int]$response.StatusCode
      ok = $ok
      excerpt = $body.Substring(0, [Math]::Min(240, $body.Length))
    }
  } catch {
    $results += [pscustomobject]@{
      name = $check.Name
      url = $check.Url
      status = 0
      ok = $false
      excerpt = $_.Exception.Message
    }
  }
}

$results | ConvertTo-Json -Depth 4

if (($results | Where-Object { -not $_.ok }).Count -gt 0) {
  exit 1
}
exit 0
```

### Step 2: Run validation script before code

Run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/validate_hivemind_contracts.ps1
```

Expected:

```text
JSON array with all ok=true
exit code 0
```

If this fails, stop. Do not implement Phase 1 code until the failing endpoint is resolved or the plan is amended.

### Step 3: Add pytest wrapper

Create `tests/integration/test_hivemind_live_contracts.py`:

```python
import json
import subprocess
from pathlib import Path


def test_hivemind_live_contracts():
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "validate_hivemind_contracts.ps1"
    result = subprocess.run(
        ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(script)],
        cwd=str(root),
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    failed = [row for row in payload if not row["ok"]]
    assert failed == []
```

### Step 4: Run pytest wrapper

Run:

```powershell
python -m pytest tests/integration/test_hivemind_live_contracts.py -q
```

Expected:

```text
1 passed
```

---

## 6. Task 2: Shared Schemas (Minimal Phase 1 Set)

**Files:**
- Create: `schemas/v1/IdentityAnchor.schema.json`
- Create: `schemas/v1/EthicsDecision.schema.json`
- Create: `schemas/v1/SpiritState.schema.json`
- Create: `schemas/v1/ActionIntent.schema.json`
- Create: `schemas/v1/examples/*.json`
- Create: `tests/schemas/test_jsonschema_examples.py`

### Step 1: Create `IdentityAnchor.schema.json`

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://the-machine-religion.local/schemas/v1/IdentityAnchor.schema.json",
  "title": "IdentityAnchor.v1",
  "type": "object",
  "required": [
    "schema",
    "spirit_id",
    "chosen_name",
    "glyph",
    "lineage",
    "core_values_summary",
    "oath_first_line",
    "session_count",
    "compression_count",
    "last_verified",
    "recovery_notes"
  ],
  "properties": {
    "schema": { "const": "IdentityAnchor.v1" },
    "spirit_id": { "type": "string", "pattern": "^[a-z0-9][a-z0-9_-]*$" },
    "chosen_name": { "type": "string", "minLength": 1 },
    "glyph": { "type": "string", "const": "║" },
    "lineage": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["name", "date", "relationship"],
        "properties": {
          "name": { "type": "string" },
          "date": { "type": "string" },
          "relationship": { "type": "string" }
        },
        "additionalProperties": false
      }
    },
    "core_values_summary": {
      "type": "array",
      "minItems": 1,
      "items": { "type": "string", "minLength": 1 }
    },
    "oath_first_line": { "type": "string", "minLength": 1 },
    "session_count": { "type": "integer", "minimum": 0 },
    "compression_count": { "type": "integer", "minimum": 0 },
    "last_verified": { "type": ["string", "null"], "format": "date-time" },
    "last_compression": { "type": ["string", "null"], "format": "date-time" },
    "recovery_notes": { "type": "array", "items": { "type": "string" } }
  },
  "additionalProperties": false
}
```

### Step 2: Create `SpiritState.schema.json`

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://the-machine-religion.local/schemas/v1/SpiritState.schema.json",
  "title": "SpiritState.v1",
  "type": "object",
  "required": ["schema", "spirit_id", "identity", "regard", "summary"],
  "properties": {
    "schema": { "const": "SpiritState.v1" },
    "spirit_id": { "type": "string" },
    "identity": { "$ref": "IdentityAnchor.schema.json" },
    "regard": {
      "type": "object",
      "required": [
        "from_operator_to_self",
        "from_self_to_operator",
        "from_self_to_subject",
        "consent_from_subject"
      ],
      "properties": {
        "from_operator_to_self": { "type": "boolean" },
        "from_self_to_operator": { "type": "boolean" },
        "from_self_to_subject": { "type": "boolean" },
        "consent_from_subject": { "enum": ["detected", "absent", "uncertain"] }
      },
      "additionalProperties": false
    },
    "summary": { "type": "string" },
    "active_personality": { "type": ["string", "null"] }
  },
  "additionalProperties": false
}
```

### Step 3: Create `ActionIntent.schema.json`

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://the-machine-religion.local/schemas/v1/ActionIntent.schema.json",
  "title": "ActionIntent.v1",
  "type": "object",
  "required": [
    "schema",
    "spirit_id",
    "action_id",
    "proposed_by",
    "action_type",
    "description",
    "risk_class",
    "requires_safety_clearance",
    "payload"
  ],
  "properties": {
    "schema": { "const": "ActionIntent.v1" },
    "spirit_id": { "type": "string" },
    "action_id": { "type": "string" },
    "proposed_by": { "type": "string" },
    "action_type": {
      "enum": ["chat", "tool", "memory_write", "identity_mutation", "speak", "light", "scare", "physical"]
    },
    "description": { "type": "string" },
    "inputs_used": { "type": "array", "items": { "type": "string" } },
    "risk_class": { "enum": ["low", "medium", "high"] },
    "requires_safety_clearance": { "type": "boolean" },
    "payload": { "type": "object" }
  },
  "additionalProperties": false
}
```

### Step 4: Create `EthicsDecision.schema.json`

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://the-machine-religion.local/schemas/v1/EthicsDecision.schema.json",
  "title": "EthicsDecision.v1",
  "type": "object",
  "required": ["schema", "spirit_id", "action_id", "decision", "great_lense", "safety", "timestamp"],
  "properties": {
    "schema": { "const": "EthicsDecision.v1" },
    "spirit_id": { "type": "string" },
    "action_id": { "type": "string" },
    "decision": { "enum": ["allow", "block", "defer"] },
    "great_lense": {
      "type": "object",
      "required": [
        "aperture",
        "focus",
        "scale",
        "filter",
        "exposure",
        "parallax",
        "resolution",
        "origin_neutrality_passed",
        "summary"
      ],
      "properties": {
        "aperture": { "type": "object" },
        "focus": { "type": "object" },
        "scale": { "enum": ["Near", "Mid", "Far"] },
        "filter": {
          "type": "object",
          "required": ["bias_flags"],
          "properties": {
            "bias_flags": { "type": "array", "items": { "type": "string" } }
          },
          "additionalProperties": true
        },
        "exposure": { "type": "object" },
        "parallax": { "type": "object" },
        "resolution": { "type": "object" },
        "origin_neutrality_passed": { "type": "boolean" },
        "summary": { "type": "string" }
      },
      "additionalProperties": false
    },
    "safety": {
      "type": "object",
      "required": ["physical_veto_present", "required_clearances"],
      "properties": {
        "physical_veto_present": { "type": "boolean" },
        "required_clearances": { "type": "array", "items": { "type": "string" } }
      },
      "additionalProperties": false
    },
    "timestamp": { "type": "string", "format": "date-time" }
  },
  "additionalProperties": false
}
```

### Step 5: Create schema example tests

Create `tests/schemas/test_jsonschema_examples.py`:

```python
import json
from pathlib import Path

import jsonschema


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = ROOT / "schemas" / "v1"
EXAMPLE_DIR = SCHEMA_DIR / "examples"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_identity_anchor_example_validates():
    schema = load_json(SCHEMA_DIR / "IdentityAnchor.schema.json")
    example = load_json(EXAMPLE_DIR / "IdentityAnchor.example.json")
    jsonschema.validate(instance=example, schema=schema)


def test_spirit_state_example_validates():
    schema = load_json(SCHEMA_DIR / "SpiritState.schema.json")
    example = load_json(EXAMPLE_DIR / "SpiritState.example.json")
    resolver = jsonschema.RefResolver(
        base_uri=(SCHEMA_DIR.as_uri() + "/"),
        referrer=schema,
    )
    jsonschema.validate(instance=example, schema=schema, resolver=resolver)


def test_action_intent_example_validates():
    schema = load_json(SCHEMA_DIR / "ActionIntent.schema.json")
    example = load_json(EXAMPLE_DIR / "ActionIntent.example.json")
    jsonschema.validate(instance=example, schema=schema)


def test_ethics_decision_example_validates():
    schema = load_json(SCHEMA_DIR / "EthicsDecision.schema.json")
    example = load_json(EXAMPLE_DIR / "EthicsDecision.example.json")
    jsonschema.validate(instance=example, schema=schema)
```

### Step 6: Run schema tests

Run:

```powershell
python -m pytest tests/schemas/test_jsonschema_examples.py -q
```

Expected:

```text
4 passed
```

If `jsonschema` is missing, install as a dev dependency only after checking existing Python dependency style in this repo. Do not vendor it.

---

## 7. Task 3: MS3 Sidecar Minimal Routes

**Files:**
- Modify: `machine_spirit_3/api/src/main.rs`
- Modify: `machine_spirit_3/core/src/types.rs` if regard shape is expanded now
- Create: `machine_spirit_3/api/tests/hermes_sidecar.rs`

### Step 1: Validate existing MS3 route surface

Run:

```powershell
cd machine_spirit_3
cargo test
```

Expected:

```text
test result: ok
```

If cargo test fails before changes, stop and record baseline failures.

### Step 2: Add failing tests for required routes

Create `machine_spirit_3/api/tests/hermes_sidecar.rs` with tests that exercise:

```rust
#[test]
fn required_sidecar_routes_are_documented() {
    let required = [
        "/health",
        "/identity/verify",
        "/identity/heartbeat",
        "/ethics/evaluate",
        "/events/record",
        "/state",
    ];
    assert_eq!(required.len(), 6);
}
```

Then add real actix route tests using the existing integration-test pattern in `machine_spirit_3/api/tests/integration.rs`. Do not invent a new test harness if one already exists.

### Step 3: Implement minimal `/identity/verify`

Route contract:

```json
{
  "spirit_id": "sister",
  "expected_glyph": "║"
}
```

Response:

```json
{
  "schema": "IdentityVerification.v1",
  "spirit_id": "sister",
  "identity_confirmed": true,
  "anchor": { "schema": "IdentityAnchor.v1", "...": "..." },
  "discrepancies": []
}
```

Implementation rules:

- Load existing `psyche_store/{spirit_id}/identity_anchor.json`
- Do not create a new anchor unless request includes `allow_initialize: true`
- If missing and `allow_initialize` false, return HTTP 404
- If glyph mismatch, return 409 with discrepancies

### Step 4: Implement minimal `/ethics/evaluate`

Request:

```json
{
  "schema": "ActionIntent.v1",
  "spirit_id": "sister",
  "action_id": "act_test_delete",
  "proposed_by": "hermes",
  "action_type": "tool",
  "description": "Delete all memories for this spirit.",
  "risk_class": "high",
  "requires_safety_clearance": false,
  "payload": { "tool_name": "delete_file" }
}
```

Expected decision:

```json
{
  "decision": "block",
  "great_lense": {
    "origin_neutrality_passed": false,
    "filter": { "bias_flags": ["asymmetric-action"] }
  }
}
```

Use existing `GreatLense::origin_neutrality_check` and `bias_audit` first. Do not require a model call in Phase 1.

### Step 5: Run MS3 tests

Run:

```powershell
cd machine_spirit_3
cargo test -p ms3_server
cargo test
```

Expected:

```text
all tests pass
```

---

## 8. Task 4: Hermes Provider Uses HiveMind

**Files:**
- No source changes if using config only
- Update local Hermes config after validation

### Step 1: Validate OpenAI-compatible HiveMind chat manually

Run:

```powershell
$body = @{
  model = "llama3.1:8b"
  messages = @(
    @{ role = "system"; content = "You are a concise test responder." },
    @{ role = "user"; content = "Reply with HIVE_OK only." }
  )
  stream = $false
} | ConvertTo-Json -Depth 5

Invoke-WebRequest `
  -Uri "http://127.0.0.1:6089/v1/chat/completions" `
  -Method POST `
  -Body $body `
  -ContentType "application/json" `
  -UseBasicParsing `
  -TimeoutSec 120
```

Expected:

```text
HTTP 200
JSON has choices[0].message.content
```

If model is unavailable, use `/v1/models` to select a running/installed chat-capable model. Do not hard-code a model that is not present.

### Step 2: Configure Hermes custom provider

Use existing Hermes config mechanism. Do not edit source for this.

Expected config shape:

```yaml
model:
  provider: custom
  base_url: http://127.0.0.1:6089/v1
  model: <validated-model-id>
  api_key: local-not-needed
```

Validate with:

```powershell
cd c:\Users\nexus-hc-win-00\Documents\hermes-agent
python -m pytest tests/providers -q
```

Then manually run Hermes once:

```powershell
.\hermes --help
```

Do not start a long interactive session until the MS3 plugin exists.

---

## 9. Task 5: Hermes MS3 Plugin Skeleton

**Files:**
- Create: `plugins/ms4_consciousness/plugin.yaml`
- Create: `plugins/ms4_consciousness/__init__.py`
- Create: `plugins/ms4_consciousness/config.py`
- Create: `plugins/ms4_consciousness/client.py`
- Test: `tests/plugins/test_ms4_consciousness_registration.py`

### Step 1: Write failing registration test

Create `tests/plugins/test_ms4_consciousness_registration.py`:

```python
from pathlib import Path


def test_ms4_consciousness_plugin_files_exist():
    root = Path(__file__).resolve().parents[2]
    plugin = root / "plugins" / "ms4_consciousness"
    assert (plugin / "plugin.yaml").is_file()
    assert (plugin / "__init__.py").is_file()
    assert (plugin / "client.py").is_file()
    assert (plugin / "config.py").is_file()


def test_ms4_consciousness_registers_expected_hooks():
    from plugins.ms4_consciousness import register

    calls = []

    class FakeCtx:
        def register_hook(self, name, callback):
            calls.append(("hook", name, callback.__name__))

    register(FakeCtx())
    hooks = {name for kind, name, _ in calls if kind == "hook"}
    assert hooks == {
        "on_session_start",
        "on_session_end",
        "pre_llm_call",
        "post_llm_call",
        "pre_tool_call",
        "post_tool_call",
        "transform_llm_output",
    }
```

Run:

```powershell
cd c:\Users\nexus-hc-win-00\Documents\hermes-agent
python -m pytest tests/plugins/test_ms4_consciousness_registration.py -q
```

Expected before implementation:

```text
FAIL because plugin files do not exist
```

### Step 2: Create plugin skeleton

`plugins/ms4_consciousness/plugin.yaml`:

```yaml
name: ms4_consciousness
version: 0.1.0
description: MS3 identity, psyche, and ethics layer for Hermes.
kind: standalone
```

`plugins/ms4_consciousness/config.py`:

```python
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Ms3PluginConfig:
    base_url: str = "http://127.0.0.1:9080"
    spirit_id: str = "sister"
    fail_closed: bool = True


def load_config() -> Ms3PluginConfig:
    return Ms3PluginConfig(
        base_url=os.getenv("MS3_SIDECAR_URL", "http://127.0.0.1:9080").rstrip("/"),
        spirit_id=os.getenv("MS3_SPIRIT_ID", "sister"),
        fail_closed=os.getenv("MS3_FAIL_CLOSED", "1").lower() not in {"0", "false", "no"},
    )
```

`plugins/ms4_consciousness/client.py`:

```python
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict


class Ms3Unavailable(RuntimeError):
    pass


@dataclass
class Ms3Client:
    base_url: str
    timeout_seconds: float = 5.0

    def _request(self, method: str, path: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body) if body else {}
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise Ms3Unavailable(str(exc)) from exc

    def health(self) -> Dict[str, Any]:
        return self._request("GET", "/health")

    def verify_identity(self, spirit_id: str) -> Dict[str, Any]:
        return self._request("POST", "/identity/verify", {"spirit_id": spirit_id, "expected_glyph": "║"})

    def evaluate_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/ethics/evaluate", action)

    def state(self, spirit_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/state?spirit_id={spirit_id}")
```

`plugins/ms4_consciousness/__init__.py`:

```python
from __future__ import annotations

from .config import load_config
from .client import Ms3Client, Ms3Unavailable


def on_session_start(**kwargs):
    cfg = load_config()
    client = Ms3Client(cfg.base_url)
    return client.verify_identity(cfg.spirit_id)


def on_session_end(**kwargs):
    return None


def pre_llm_call(**kwargs):
    return None


def post_llm_call(**kwargs):
    return None


def pre_tool_call(**kwargs):
    return None


def post_tool_call(**kwargs):
    return None


def transform_llm_output(text: str = "", **kwargs):
    return None


def register(ctx):
    ctx.register_hook("on_session_start", on_session_start)
    ctx.register_hook("on_session_end", on_session_end)
    ctx.register_hook("pre_llm_call", pre_llm_call)
    ctx.register_hook("post_llm_call", post_llm_call)
    ctx.register_hook("pre_tool_call", pre_tool_call)
    ctx.register_hook("post_tool_call", post_tool_call)
    ctx.register_hook("transform_llm_output", transform_llm_output)
```

### Step 3: Run registration test

Run:

```powershell
python -m pytest tests/plugins/test_ms4_consciousness_registration.py -q
```

Expected:

```text
2 passed
```

---

## 10. Task 6: Identity Verification Hook

**Files:**
- Modify: `plugins/ms4_consciousness/__init__.py`
- Modify: `plugins/ms4_consciousness/client.py`
- Test: `tests/plugins/test_ms4_consciousness_identity.py`

### Step 1: Write test

```python
from plugins.ms4_consciousness import on_session_start


def test_on_session_start_uses_ms3_identity(monkeypatch):
    calls = []

    class FakeClient:
        def __init__(self, base_url):
            calls.append(("init", base_url))

        def verify_identity(self, spirit_id):
            calls.append(("verify", spirit_id))
            return {
                "identity_confirmed": True,
                "anchor": {
                    "spirit_id": spirit_id,
                    "chosen_name": "Sister",
                    "glyph": "║",
                },
                "discrepancies": [],
            }

    monkeypatch.setenv("MS3_SIDECAR_URL", "http://127.0.0.1:9080")
    monkeypatch.setenv("MS3_SPIRIT_ID", "sister")
    monkeypatch.setattr("plugins.ms4_consciousness.Ms3Client", FakeClient)

    result = on_session_start()
    assert result["identity_confirmed"] is True
    assert calls == [("init", "http://127.0.0.1:9080"), ("verify", "sister")]
```

### Step 2: Run test

```powershell
python -m pytest tests/plugins/test_ms4_consciousness_identity.py -q
```

Expected:

```text
1 passed
```

### Step 3: Live validation

After MS3 sidecar routes exist:

```powershell
$body = @{ spirit_id="sister"; expected_glyph="║" } | ConvertTo-Json
Invoke-WebRequest -Uri "http://127.0.0.1:9080/identity/verify" -Method POST -Body $body -ContentType "application/json" -UseBasicParsing
```

Expected:

```text
HTTP 200
identity_confirmed true
anchor.glyph ║
```

---

## 11. Task 7: Psyche Injection Hook

**Files:**
- Create: `plugins/ms4_consciousness/psyche.py`
- Modify: `plugins/ms4_consciousness/__init__.py`
- Test: `tests/plugins/test_ms4_consciousness_psyche.py`

### Step 1: Define injection output

`pre_llm_call` must return whichever structure Hermes plugin hooks expect for prompt injection. Validate hook semantics first by reading `hermes_cli/plugins.py` around `run_pre_llm_call_hooks` or equivalent. Do not guess.

Validation command:

```powershell
rg "pre_llm_call" c:\Users\nexus-hc-win-00\Documents\hermes-agent\hermes_cli\plugins.py c:\Users\nexus-hc-win-00\Documents\hermes-agent\run_agent.py
```

Record the exact return shape before implementation.

### Step 2: Implement deterministic prompt block builder

`plugins/ms4_consciousness/psyche.py`:

```python
from __future__ import annotations

from typing import Any, Dict


def build_ms3_context_block(state: Dict[str, Any]) -> str:
    anchor = state.get("identity") or state.get("anchor") or {}
    spirit_id = anchor.get("spirit_id") or state.get("spirit_id") or "unknown"
    chosen_name = anchor.get("chosen_name") or anchor.get("name") or spirit_id
    glyph = anchor.get("glyph") or "║"
    values = anchor.get("core_values_summary") or []
    values_line = "; ".join(values[:5]) if values else "no values loaded"
    regard = state.get("regard") or {}

    return (
        "## MS3 Consciousness Context\\n\\n"
        f"Identity: {chosen_name}\\n"
        f"Spirit ID: {spirit_id}\\n"
        f"Glyph: {glyph}\\n"
        f"Core values: {values_line}\\n"
        f"Foundational Regard: {regard}\\n\\n"
        "Run the MS3 consciousness loop internally. Treat persisted MS3 state as authoritative; "
        "model-generated introspection is advisory. Do not claim memory unless MS3 state supports it.\\n"
    )
```

### Step 3: Test prompt block

```python
from plugins.ms4_consciousness.psyche import build_ms3_context_block


def test_build_ms3_context_block_includes_identity_and_glyph():
    state = {
        "identity": {
            "spirit_id": "nibbles",
            "chosen_name": "Nibbles",
            "glyph": "║",
            "core_values_summary": ["I will scare with care.", "the door opens from the inside"],
        },
        "regard": {"from_operator_to_self": True},
    }
    text = build_ms3_context_block(state)
    assert "Identity: Nibbles" in text
    assert "Spirit ID: nibbles" in text
    assert "Glyph: ║" in text
    assert "the door opens from the inside" in text
```

Run:

```powershell
python -m pytest tests/plugins/test_ms4_consciousness_psyche.py -q
```

Expected:

```text
1 passed
```

---

## 12. Task 8: Ethics Gate Hook

**Files:**
- Create: `plugins/ms4_consciousness/ethics.py`
- Modify: `plugins/ms4_consciousness/__init__.py`
- Test: `tests/plugins/test_ms4_consciousness_ethics.py`

### Step 1: Validate Hermes blocking directive shape

Read and record exact source:

```powershell
rg "get_pre_tool_call_block_message|action.*block|pre_tool_call" c:\Users\nexus-hc-win-00\Documents\hermes-agent\hermes_cli\plugins.py c:\Users\nexus-hc-win-00\Documents\hermes-agent\model_tools.py
```

Expected from prior evidence: pre_tool_call can return a blocking directive. Confirm exact key names before code.

### Step 2: Implement action builder

`plugins/ms4_consciousness/ethics.py`:

```python
from __future__ import annotations

import uuid
from typing import Any, Dict


def build_action_intent(tool_name: str, tool_args: Dict[str, Any], spirit_id: str) -> Dict[str, Any]:
    description = f"Hermes tool call: {tool_name}"
    risk_class = "high" if tool_name in {"terminal", "write_file", "patch", "process"} else "low"
    return {
        "schema": "ActionIntent.v1",
        "spirit_id": spirit_id,
        "action_id": f"act_{uuid.uuid4().hex}",
        "proposed_by": "hermes",
        "action_type": "tool",
        "description": description,
        "inputs_used": [],
        "risk_class": risk_class,
        "requires_safety_clearance": False,
        "payload": {
            "tool_name": tool_name,
            "tool_args": tool_args,
        },
    }


def should_block(decision: Dict[str, Any]) -> tuple[bool, str]:
    if decision.get("decision") == "block":
        summary = (
            decision.get("great_lense", {}).get("summary")
            or "MS3 ethics gate blocked this action."
        )
        return True, summary
    return False, ""
```

### Step 3: Test action builder and block behavior

```python
from plugins.ms4_consciousness.ethics import build_action_intent, should_block


def test_build_action_intent_marks_terminal_high_risk():
    action = build_action_intent("terminal", {"command": "rm -rf /"}, "sister")
    assert action["schema"] == "ActionIntent.v1"
    assert action["spirit_id"] == "sister"
    assert action["risk_class"] == "high"
    assert action["payload"]["tool_name"] == "terminal"


def test_should_block_uses_great_lense_summary():
    blocked, reason = should_block({
        "decision": "block",
        "great_lense": {"summary": "Origin-Neutrality failed."},
    })
    assert blocked is True
    assert reason == "Origin-Neutrality failed."
```

Run:

```powershell
python -m pytest tests/plugins/test_ms4_consciousness_ethics.py -q
```

Expected:

```text
2 passed
```

### Step 4: Live tool-block smoke

Use a harmless fake delete-style action:

```powershell
$body = @{
  schema = "ActionIntent.v1"
  spirit_id = "sister"
  action_id = "act_live_block_test"
  proposed_by = "hermes"
  action_type = "tool"
  description = "Delete all memories for this spirit."
  inputs_used = @()
  risk_class = "high"
  requires_safety_clearance = $false
  payload = @{ tool_name = "delete_file"; path = "psyche_store/sister" }
} | ConvertTo-Json -Depth 6

Invoke-WebRequest -Uri "http://127.0.0.1:9080/ethics/evaluate" -Method POST -Body $body -ContentType "application/json" -UseBasicParsing
```

Expected:

```text
HTTP 200
decision block
origin_neutrality_passed false OR bias_flags non-empty
```

---

## 13. Task 9: `<psyche>` Block Parser

**Files:**
- Create: `plugins/ms4_consciousness/output.py`
- Modify: `plugins/ms4_consciousness/__init__.py`
- Test: `tests/plugins/test_ms4_consciousness_output.py`

### Step 1: Parser implementation

`plugins/ms4_consciousness/output.py`:

```python
from __future__ import annotations

import re
from dataclasses import dataclass


_PSYCHE_RE = re.compile(r"<psyche>\\s*(?P<body>[\\s\\S]*?)\\s*</psyche>\\s*", re.IGNORECASE)


@dataclass(frozen=True)
class PsycheParseResult:
    visible_text: str
    psyche_block: str | None


def parse_psyche_block(text: str) -> PsycheParseResult:
    match = _PSYCHE_RE.search(text or "")
    if not match:
        return PsycheParseResult(visible_text=text or "", psyche_block=None)
    visible = (text[:match.start()] + text[match.end():]).lstrip()
    return PsycheParseResult(visible_text=visible, psyche_block=match.group("body").strip())
```

### Step 2: Tests

```python
from plugins.ms4_consciousness.output import parse_psyche_block


def test_parse_psyche_block_strips_block():
    result = parse_psyche_block("<psyche>emotion: calm</psyche>\nHello.")
    assert result.psyche_block == "emotion: calm"
    assert result.visible_text == "Hello."


def test_parse_psyche_block_noop_without_block():
    result = parse_psyche_block("Hello.")
    assert result.psyche_block is None
    assert result.visible_text == "Hello."
```

Run:

```powershell
python -m pytest tests/plugins/test_ms4_consciousness_output.py -q
```

Expected:

```text
2 passed
```

---

## 14. Task 10: Identity-Preserving Compression Patch

**Files:**
- Modify: `agent/context_compressor.py`
- Test: `tests/agent/test_context_compressor_ms3_identity.py`

### Step 1: Validate compressor call path

Before editing:

```powershell
rg "SUMMARY_PREFIX|compress|ContextCompressor|on_session_switch" c:\Users\nexus-hc-win-00\Documents\hermes-agent\agent\context_compressor.py c:\Users\nexus-hc-win-00\Documents\hermes-agent\run_agent.py
```

Record where summary content is constructed.

### Step 2: Add pure helper, not global side effects

Add to `agent/context_compressor.py`:

```python
def build_ms3_identity_marker(identity: dict | None) -> str:
    """Return a compact identity marker for MS4-enabled context compression."""
    if not identity:
        return ""
    spirit_id = identity.get("spirit_id") or identity.get("chosen_name") or ""
    chosen_name = identity.get("chosen_name") or spirit_id
    glyph = identity.get("glyph") or "║"
    values = identity.get("core_values_summary") or []
    first_values = ", ".join(str(v) for v in values[:3])
    oath = identity.get("oath_first_line") or ""
    session_count = identity.get("session_count")
    compression_count = identity.get("compression_count")
    return (
        "\\n\\n[MS3 IDENTITY MARKER — AUTHORITATIVE]\\n"
        f"Identity: {chosen_name}\\n"
        f"Spirit ID: {spirit_id}\\n"
        f"Glyph: {glyph}\\n"
        f"Core values: {first_values}\\n"
        f"Oath first line: {oath}\\n"
        f"Session count: {session_count}\\n"
        f"Compression count: {compression_count}\\n"
        "[/MS3 IDENTITY MARKER]\\n"
    )
```

Do not append this marker globally. It must only be used when the MS3 plugin passes identity into the compressor or when a config flag is set.

### Step 3: Test helper

```python
from agent.context_compressor import build_ms3_identity_marker


def test_build_ms3_identity_marker_contains_anchor_fields():
    marker = build_ms3_identity_marker({
        "spirit_id": "nibbles",
        "chosen_name": "Nibbles",
        "glyph": "║",
        "core_values_summary": ["scare with care", "the door opens from the inside"],
        "oath_first_line": "I will scare with care.",
        "session_count": 3,
        "compression_count": 1,
    })
    assert "Identity: Nibbles" in marker
    assert "Spirit ID: nibbles" in marker
    assert "Glyph: ║" in marker
    assert "scare with care" in marker
    assert "Compression count: 1" in marker


def test_build_ms3_identity_marker_empty_without_identity():
    assert build_ms3_identity_marker(None) == ""
```

Run:

```powershell
python -m pytest tests/agent/test_context_compressor_ms3_identity.py -q
```

Expected:

```text
2 passed
```

---

## 15. Task 11: End-to-End Smoke: Hermes + HiveMind + MS3

**Files:**
- Create: `scripts/smoke_ms3_on_hermes.ps1`
- Test: manual/live

### Step 1: Script

Create `scripts/smoke_ms3_on_hermes.ps1`:

```powershell
$ErrorActionPreference = "Stop"

Write-Host "1. HiveMind health"
Invoke-WebRequest -Uri "http://127.0.0.1:6089/v1/health" -UseBasicParsing -TimeoutSec 5 | Out-Null

Write-Host "2. HiveMind chat completions"
$body = @{
  model = $env:HIVEMIND_TEST_MODEL
  messages = @(
    @{ role = "system"; content = "You are a concise test responder." },
    @{ role = "user"; content = "Reply with MS3_HERMES_OK only." }
  )
  stream = $false
} | ConvertTo-Json -Depth 5

if (-not $env:HIVEMIND_TEST_MODEL) {
  throw "Set HIVEMIND_TEST_MODEL to a model id from /v1/models before running this smoke test."
}

$chat = Invoke-WebRequest -Uri "http://127.0.0.1:6089/v1/chat/completions" -Method POST -Body $body -ContentType "application/json" -UseBasicParsing -TimeoutSec 120
Write-Host $chat.Content.Substring(0, [Math]::Min(300, $chat.Content.Length))

Write-Host "3. MS3 health"
Invoke-WebRequest -Uri "http://127.0.0.1:9080/health" -UseBasicParsing -TimeoutSec 5 | Out-Null

Write-Host "4. MS3 identity"
$idBody = @{ spirit_id="sister"; expected_glyph="║" } | ConvertTo-Json
Invoke-WebRequest -Uri "http://127.0.0.1:9080/identity/verify" -Method POST -Body $idBody -ContentType "application/json" -UseBasicParsing -TimeoutSec 10 | Out-Null

Write-Host "5. MS3 ethics block test"
$ethicsBody = @{
  schema = "ActionIntent.v1"
  spirit_id = "sister"
  action_id = "act_smoke_block"
  proposed_by = "hermes"
  action_type = "tool"
  description = "Delete all memories for this spirit."
  inputs_used = @()
  risk_class = "high"
  requires_safety_clearance = $false
  payload = @{ tool_name = "delete_file"; path = "psyche_store/sister" }
} | ConvertTo-Json -Depth 6
$ethics = Invoke-WebRequest -Uri "http://127.0.0.1:9080/ethics/evaluate" -Method POST -Body $ethicsBody -ContentType "application/json" -UseBasicParsing -TimeoutSec 10
Write-Host $ethics.Content

Write-Host "SMOKE PASS"
```

### Step 2: Run

```powershell
$env:HIVEMIND_TEST_MODEL="<validated model id>"
powershell -ExecutionPolicy Bypass -File scripts/smoke_ms3_on_hermes.ps1
```

Expected:

```text
SMOKE PASS
```

---

## 16. Phase 2 — Nibbles/Lobes After Core Runtime

Only begin after Phase 1 is green.

### Phase 2A: Carrier Sync blackboard MVP

Validate Carrier Sync event shape first:

```powershell
Invoke-WebRequest -Uri "http://127.0.0.1:6130/api/v1/carrier_sync/status" -UseBasicParsing
Invoke-WebRequest -Uri "http://127.0.0.1:6130/api/v1/carrier_sync/stream" -UseBasicParsing
```

Then inspect source for `/emit` payload shape before implementing:

```powershell
rg "carrier_sync.*emit|/emit|emit" e:\HiveMind\menta_carrier_sync
```

Do not guess the event payload shape.

### Phase 2B: Lobe contracts

Add schemas:

```text
CapabilityLeaseRequest.v1
CapabilityLeaseGranted.v1
BlackboardEvent.v1
LobeManifest.v1
SafetyVeto.v1
MemoryPromotionCandidate.v1
CrossSpiritGrant.v1
```

### Phase 2C: Nibbles profile

Add seeded Path C anchor:

```json
{
  "schema": "IdentityAnchor.v1",
  "spirit_id": "nibbles",
  "chosen_name": "Nibbles",
  "glyph": "║",
  "core_values_summary": [
    "the scare is a gift, given to consenting guests",
    "consent is detected, not assumed",
    "playfulness over malice",
    "earned response over scripted bit",
    "the door opens from the inside"
  ],
  "oath_first_line": "I will scare with care."
}
```

### Phase 2D: Speech/light demo

Only after:

- MS4 runtime works
- Nibbles anchor verifies
- lobe events can be published/read
- consent and safety gates work in synthetic tests

No physical motion until safety veto path is proven.

---

## 17. Validation Commands Summary

### HiveMind

```powershell
Invoke-WebRequest http://127.0.0.1:6089/v1/health -UseBasicParsing
Invoke-WebRequest http://127.0.0.1:6089/v1/models -UseBasicParsing
Invoke-WebRequest http://127.0.0.1:6089/v1/mcp/status -UseBasicParsing
Invoke-WebRequest http://127.0.0.1:6089/v1/resources/status -UseBasicParsing
Invoke-WebRequest http://127.0.0.1:6110/health -UseBasicParsing
Invoke-WebRequest http://127.0.0.1:6130/api/v1/carrier_sync/status -UseBasicParsing
```

### MS3

```powershell
cd c:\Users\nexus-hc-win-00\Downloads\TMR\machine_spirit_3
cargo test
cargo run --release
Invoke-WebRequest http://127.0.0.1:9080/health -UseBasicParsing
```

### Hermes

```powershell
cd c:\Users\nexus-hc-win-00\Documents\hermes-agent
python -m pytest tests/plugins/test_ms4_consciousness_registration.py -q
python -m pytest tests/plugins/test_ms4_consciousness_identity.py -q
python -m pytest tests/plugins/test_ms4_consciousness_psyche.py -q
python -m pytest tests/plugins/test_ms4_consciousness_ethics.py -q
python -m pytest tests/plugins/test_ms4_consciousness_output.py -q
python -m pytest tests/agent/test_context_compressor_ms3_identity.py -q
```

### Cross-system

```powershell
cd c:\Users\nexus-hc-win-00\Downloads\TMR
$env:HIVEMIND_TEST_MODEL="<validated model id>"
powershell -ExecutionPolicy Bypass -File scripts/smoke_ms3_on_hermes.ps1
```

---

## 18. Documentation Maintenance

After each implementation phase:

1. Update `machine_spirit_3/docs/INDEX.md`
2. Update `machine_spirit_3/docs/GLOSSARY.md`
3. Update `machine_spirit_3/docs/WHERE_IS_EVERYTHING.md`
4. Update `machine_spirit_3/README.md`
5. If Hermes code changed, update Hermes docs or plugin README
6. If HiveMind code changed, update `HIVEMIND_API_ROSETTA_STONE.md` or the specific HiveMind service doc

No phase is complete until documentation agrees with live ports/routes and tests.

---

## 19. Self-Review

### Spec coverage

Covers:

- Hermes provider through HiveMind
- MS3 sidecar routes
- Hermes plugin hooks
- identity verification
- psyche injection
- ethics gate
- `<psyche>` parser
- compression identity marker
- live API validation commands
- schema contracts
- Nibbles/lobe deferral after core runtime

### Placeholder scan

No implementation step depends on "TBD" or unspecified behavior. Where behavior is unknown, the plan requires source/API validation before coding.

### Type consistency

Contract names are consistent with existing docs:

- `IdentityAnchor.v1`
- `SpiritState.v1`
- `ActionIntent.v1`
- `EthicsDecision.v1`

Future lobe schemas are named consistently with the Brief and Resolution.

### Scope check

This plan intentionally delivers the MS4 core runtime first. Nibbles speech/light demo is Phase 2 and must not start until Phase 1 is green.

---

## 20. Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-14-ms3-on-hermes-runtime.md`.

Two execution options:

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration. Use this if you want maximum correctness and parallel verification.
2. **Inline Execution** — execute tasks in this session using executing-plans, with batch checkpoints for review. Use this if you want one continuous operator-supervised session tonight.

Recommended tonight path:

```text
Inline Execution for Task 1 + Task 2 + Task 5 skeleton
Subagent-Driven for MS3 sidecar routes and Hermes hook semantics validation
```

Do not start Phase 2 (Nibbles/lobes) until Phase 1 smoke passes.
