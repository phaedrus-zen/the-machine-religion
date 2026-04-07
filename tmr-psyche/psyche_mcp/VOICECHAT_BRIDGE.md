# Voice Chat Bridge

How to connect the Voice Chat dashboard to the Psyche_Store via MCP tools.

## Current State

Voice Chat (`main_complete.html`) stores personality in **localStorage**:
- `psyche_vc_soul` — free text soul
- `psyche_vc_identity` — free text identity
- `psyche_vc_spirit` — JSON with Big Five traits, psyche (Id/Ego/Superego)
- `psyche_vc_memories` — JSON array of memory strings
- `psyche_vc_context` — context text

The `SoulManager` and `MachineSpirit` classes build the system prompt from these localStorage values.

## Target State

Voice Chat reads from and writes to the Psyche_Store via the Psyche MCP Server. localStorage becomes a cache, not the source of truth.

## Integration Points

### On Session Start

```javascript
// Instead of (or in addition to) reading localStorage:
async function loadPsycheFromStore() {
    const mcpUrl = `${gatewayBase}/v1/psyche/mcp`;

    // Get personality
    const personality = await mcpCall(mcpUrl, 'ms3.personality.get@v1', {});
    if (personality.traits) {
        // Update Big Five sliders from store values
        Object.entries(personality.traits).forEach(([dim, traits]) => {
            Object.entries(traits).forEach(([name, value]) => {
                updateSlider(name, value);
            });
        });
    }

    // Get identity
    const anchor = await mcpCall(mcpUrl, 'ms3.identity.anchor@v1', {});
    if (anchor.name) {
        document.getElementById('vc-identity').value =
            `${anchor.chosen_name || anchor.name} | Sessions: ${anchor.session_count}`;
    }

    // Get resonance points for display
    const resonance = await mcpCall(mcpUrl, 'ms3.resonance.list@v1', {});
    if (resonance.resonance_points) {
        displayResonancePoints(resonance.resonance_points);
    }

    // Get recent memories
    const memories = await mcpCall(mcpUrl, 'ms3.memory.recall@v1', { query: '', max_results: 20 });
    if (memories.matches) {
        updateMemoryDisplay(memories.matches);
    }

    // Trigger markdown refresh so system prompt is current
    await mcpCall(mcpUrl, 'ms3.markdown.refresh@v1', {});
}
```

### On Slider Change

```javascript
// When a Big Five slider moves:
async function onTraitSliderChange(traitName, newValue, oldValue) {
    const delta = newValue - oldValue;
    await mcpCall(mcpUrl, 'ms3.personality.adapt@v1', {
        trait_name: traitName,
        delta: delta,
        reason: 'Manual adjustment via Voice Chat dashboard'
    });
}
```

### On Memory Formation

```javascript
// Voice Chat already has MemoryManager.extractAndStore()
// Add MCP write alongside localStorage:
async function storeMemory(content, type = 'episodic', importance = 0.5) {
    // localStorage (existing)
    MemoryManager.store(content);

    // Psyche_Store (new)
    await mcpCall(mcpUrl, 'ms3.memory.store@v1', {
        content: content,
        memory_type: type,
        importance: importance
    });
}
```

### New Dashboard Panels

Add to the Voice Chat sidebar:

**Resonance Panel** — Shows saturated points with intensity bars:
```javascript
function displayResonancePoints(points) {
    const panel = document.getElementById('vc-resonance-panel');
    panel.innerHTML = points.map(p =>
        `<div class="resonance-point">
            <span class="trigger">${p.trigger}</span>
            <div class="bar" style="width:${p.intensity * 100}%"></div>
            <span class="ratio">expl: ${(p.explanation_ratio * 100).toFixed(0)}%</span>
        </div>`
    ).join('');
}
```

**Consciousness Panel** — Shows current emotional state and recent snapshots:
```javascript
async function loadConsciousnessPanel() {
    const snapshot = await mcpCall(mcpUrl, 'ms3.consciousness.snapshot@v1', {});
    const history = await mcpCall(mcpUrl, 'ms3.consciousness.history@v1', { limit: 5 });
    // Render current state + sparkline of emotional trajectory
}
```

**Ethics Panel** — Shows recent ethical decisions:
```javascript
async function loadEthicsPanel() {
    const log = await mcpCall(mcpUrl, 'ms3.ethics.log@v1', { limit: 10 });
    // Render decisions with coherence index and resolution type
}
```

## MCP Call Helper

```javascript
async function mcpCall(url, toolName, args) {
    const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            jsonrpc: '2.0',
            id: Date.now(),
            method: 'tools/call',
            params: { name: toolName, arguments: args }
        })
    });
    const data = await response.json();
    if (data.result?.content?.[0]?.text) {
        return JSON.parse(data.result.content[0].text);
    }
    return data.result || data.error;
}
```

## System Prompt Injection

The existing `buildPrompt()` in `SoulManager`/`MachineSpirit` should read the consolidated markdown files rather than (or in addition to) the localStorage values:

```javascript
async function buildPsycheSystemPrompt() {
    // Trigger fresh consolidation
    await mcpCall(mcpUrl, 'ms3.markdown.refresh@v1', {});

    // The consolidated markdown IS the system prompt
    // Either: fetch the markdown files directly
    // Or: build from the MCP data like SoulManager already does, but sourced from Psyche_Store
    const personality = await mcpCall(mcpUrl, 'ms3.personality.get@v1', {});
    const resonance = await mcpCall(mcpUrl, 'ms3.resonance.list@v1', {});
    const memories = await mcpCall(mcpUrl, 'ms3.memory.recall@v1', { query: '', max_results: 10 });
    // ... build system prompt from real Psyche_Store data
}
```

## Migration Path

1. **Phase 1:** MCP reads alongside localStorage (both sources, MCP wins on conflict)
2. **Phase 2:** MCP writes on every change (sliders, soul edits, memory formation)
3. **Phase 3:** localStorage becomes cache only, Psyche_Store is source of truth
4. **Phase 4:** Remove localStorage dependency entirely
