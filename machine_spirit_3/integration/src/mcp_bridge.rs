use ms3_core::{Ms3Error, Ms3Result};
use ms3_core::config::PermissionLevel;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Thin MCP client that calls HiveMind's /v1/mcp endpoint.
/// Discovers tools via JSON-RPC `tools/list` and executes via `tools/call`.
pub struct McpToolClient {
    client: reqwest::Client,
    mcp_url: String,
}

#[derive(Debug, Deserialize)]
struct JsonRpcResponse {
    result: Option<Value>,
    error: Option<JsonRpcError>,
}

#[derive(Debug, Deserialize)]
struct JsonRpcError {
    code: i64,
    message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolInfo {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default, rename = "inputSchema")]
    pub input_schema: Value,
}

impl McpToolClient {
    pub fn new(gateway_base_url: &str) -> Self {
        let mcp_url = format!("{}/v1/mcp", gateway_base_url.trim_end_matches('/'));
        Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(60))
                .build()
                .unwrap_or_default(),
            mcp_url,
        }
    }

    pub async fn discover_tools(&self) -> Ms3Result<Vec<McpToolInfo>> {
        let payload = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {}
        });

        let resp = self.client.post(&self.mcp_url)
            .json(&payload)
            .send().await
            .map_err(|e| Ms3Error::Gateway(format!("MCP discovery failed: {}", e)))?;

        let body: JsonRpcResponse = resp.json().await
            .map_err(|e| Ms3Error::Gateway(format!("MCP discovery parse error: {}", e)))?;

        if let Some(err) = body.error {
            return Err(Ms3Error::Gateway(format!("MCP error {}: {}", err.code, err.message)));
        }

        let tools_value = body.result
            .and_then(|r| r.get("tools").cloned())
            .unwrap_or(Value::Array(Vec::new()));

        let tools: Vec<McpToolInfo> = serde_json::from_value(tools_value)
            .unwrap_or_default();

        Ok(tools)
    }

    pub async fn execute_tool(&self, tool_name: &str, input: &Value) -> Ms3Result<String> {
        let payload = serde_json::json!({
            "jsonrpc": "2.0",
            "id": chrono::Utc::now().timestamp_millis(),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": input
            }
        });

        let resp = self.client.post(&self.mcp_url)
            .json(&payload)
            .send().await
            .map_err(|e| Ms3Error::Gateway(format!("MCP call failed: {}", e)))?;

        let body: JsonRpcResponse = resp.json().await
            .map_err(|e| Ms3Error::Gateway(format!("MCP response parse error: {}", e)))?;

        if let Some(err) = body.error {
            return Err(Ms3Error::Gateway(format!("MCP tool error {}: {}", err.code, err.message)));
        }

        let result = body.result.unwrap_or(Value::Null);
        if let Some(content) = result.get("content") {
            if let Some(arr) = content.as_array() {
                if let Some(first) = arr.first() {
                    if let Some(text) = first.get("text").and_then(|t| t.as_str()) {
                        return Ok(text.to_string());
                    }
                }
            }
        }

        Ok(serde_json::to_string_pretty(&result).unwrap_or_default())
    }

    pub async fn health_check(&self) -> bool {
        let payload = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "ping",
            "params": {}
        });
        self.client.post(&self.mcp_url)
            .json(&payload)
            .send().await
            .is_ok()
    }
}

/// Bridge that wraps McpToolClient with tool caching and periodic re-discovery.
pub struct McpBridge {
    client: McpToolClient,
    cached_tools: Arc<RwLock<Vec<McpToolInfo>>>,
    enabled: bool,
}

impl McpBridge {
    pub fn new(gateway_base_url: &str, enabled: bool) -> Self {
        Self {
            client: McpToolClient::new(gateway_base_url),
            cached_tools: Arc::new(RwLock::new(Vec::new())),
            enabled,
        }
    }

    pub async fn discover(&self) -> Ms3Result<Vec<McpToolInfo>> {
        if !self.enabled {
            return Ok(Vec::new());
        }
        match self.client.discover_tools().await {
            Ok(tools) => {
                tracing::info!("MCP bridge discovered {} tools", tools.len());
                let mut cache = self.cached_tools.write().await;
                *cache = tools.clone();
                Ok(tools)
            }
            Err(e) => {
                tracing::warn!("MCP discovery failed (non-fatal): {}", e);
                Ok(Vec::new())
            }
        }
    }

    pub async fn cached_tools(&self) -> Vec<McpToolInfo> {
        self.cached_tools.read().await.clone()
    }

    pub async fn execute(&self, tool_name: &str, input: &Value) -> Ms3Result<String> {
        if !self.enabled {
            return Err(Ms3Error::Config("MCP bridge is not enabled".into()));
        }
        self.client.execute_tool(tool_name, input).await
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Convert discovered MCP tools into ToolSpec format for the registry.
    pub fn to_tool_specs(tools: &[McpToolInfo]) -> Vec<crate::mcp_bridge::ToolSpecCompat> {
        tools.iter().map(|t| ToolSpecCompat {
            name: t.name.clone(),
            description: t.description.clone(),
            input_schema: t.input_schema.clone(),
            required_permission: infer_permission(&t.name),
        }).collect()
    }
}

/// Infer permission level from tool name patterns.
fn infer_permission(tool_name: &str) -> PermissionLevel {
    if tool_name.contains("commit") || tool_name.contains("push")
        || tool_name.contains("delete") || tool_name.contains("deploy") {
        PermissionLevel::DangerFullAccess
    } else if tool_name.contains("store") || tool_name.contains("adapt")
        || tool_name.contains("update") || tool_name.contains("record")
        || tool_name.contains("branch") || tool_name.contains("stash") {
        PermissionLevel::Modify
    } else {
        PermissionLevel::ReadOnly
    }
}

/// Lightweight struct for bridging MCP discovery to the consciousness tool registry.
#[derive(Debug, Clone)]
pub struct ToolSpecCompat {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
    pub required_permission: PermissionLevel,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mcp_client_url_construction() {
        let client = McpToolClient::new("http://localhost:6089");
        assert_eq!(client.mcp_url, "http://localhost:6089/v1/mcp");
    }

    #[test]
    fn test_mcp_client_trailing_slash() {
        let client = McpToolClient::new("http://localhost:6089/");
        assert_eq!(client.mcp_url, "http://localhost:6089/v1/mcp");
    }

    #[test]
    fn test_infer_permission_commit_is_danger() {
        assert_eq!(infer_permission("git.commit@v1"), PermissionLevel::DangerFullAccess);
        assert_eq!(infer_permission("git.push@v1"), PermissionLevel::DangerFullAccess);
    }

    #[test]
    fn test_infer_permission_read_is_readonly() {
        assert_eq!(infer_permission("ms3.memory.recall@v1"), PermissionLevel::ReadOnly);
        assert_eq!(infer_permission("code.lsp.definition@v1"), PermissionLevel::ReadOnly);
    }

    #[test]
    fn test_infer_permission_store_is_modify() {
        assert_eq!(infer_permission("ms3.memory.store@v1"), PermissionLevel::Modify);
        assert_eq!(infer_permission("ms3.personality.adapt@v1"), PermissionLevel::Modify);
    }

    #[test]
    fn test_bridge_disabled_returns_empty() {
        let bridge = McpBridge::new("http://localhost:6089", false);
        assert!(!bridge.is_enabled());
        let rt = tokio::runtime::Runtime::new().unwrap();
        let tools = rt.block_on(bridge.discover()).unwrap();
        assert!(tools.is_empty());
    }
}
