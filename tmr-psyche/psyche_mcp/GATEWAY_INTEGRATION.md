# Gateway Integration

How to register the Psyche MCP Server with the inference gateway.

## 1. Service Supervisor Registration

Add to the service supervisor config or register at runtime:

```json
{
  "service_name": "ms3_mcp_server",
  "port": 6132,
  "health_path": "api/v1/ms3_mcp_server/healthcheck/basic"
}
```

The server self-registers on startup if the service supervisor is reachable at `SUPERVISOR_BASE` (default `http://127.0.0.1:5080`).

## 2. Gateway Proxy Routes

Add to the gateway's proxy routing module:

```rust
// Psyche MCP proxy
.route("/v1/psyche/mcp", web::post().to(proxy_psyche_mcp_post))
.route("/v1/psyche/mcp", web::get().to(proxy_psyche_mcp_get))
.route("/v1/psyche/mcp/status", web::get().to(proxy_psyche_mcp_status))
```

Proxy handlers forward to `http://127.0.0.1:6132/mcp` (same pattern as the existing MCP proxy routes).

## 3. Service URL Resolution

In the proxy module, resolve the psyche MCP URL the same way as other MCP services:

```rust
fn psyche_mcp_url() -> String {
    service_url("ms3_mcp_server").unwrap_or_else(|| "http://127.0.0.1:6132".to_string())
}
```

## 4. Agent Tool Definitions

To let the cluster agent use psyche tools, add to the agent's tool definitions:

```rust
// Psyche introspection tools
json!({"type": "function", "function": {"name": "psyche_memory_recall", "description": "Search the spirit's memories by keyword", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}}, "required": ["query"]}}}),
json!({"type": "function", "function": {"name": "psyche_consciousness_snapshot", "description": "Get the spirit's current consciousness state", "parameters": {"type": "object", "properties": {}}}}),
json!({"type": "function", "function": {"name": "psyche_resonance_list", "description": "List the spirit's resonance points (things that matter disproportionately)", "parameters": {"type": "object", "properties": {}}}}),
json!({"type": "function", "function": {"name": "psyche_identity_verify", "description": "Verify the spirit's identity against its anchor", "parameters": {"type": "object", "properties": {}}}}),
```

And in `execute_tool()`, route these to `http://127.0.0.1:6132/mcp` as JSON-RPC `tools/call`.

## 5. MCP Gateway Bridge (Optional)

Alternatively, add psyche tools to the MCP gateway's tool catalog with:

```json
{
  "name": "ms3.memory.recall@v1",
  "description": "Search the Machine Spirit's memories",
  "backend": {
    "service": "ms3_mcp_server",
    "base_url": "http://127.0.0.1:6132",
    "endpoint": "/mcp",
    "method": "POST"
  },
  "inputSchema": { ... }
}
```

This would expose psyche tools through the standard MCP gateway at `/v1/mcp`, making them available to any MCP client without knowing about port 6132.
