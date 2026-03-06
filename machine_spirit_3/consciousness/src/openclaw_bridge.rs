use ms3_core::{Ms3Error, Ms3Result};
use ms3_ethics::GreatLense;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolRequest {
    pub tool_name: String,
    pub params: serde_json::Value,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub success: bool,
    pub output: String,
    pub ethics_cleared: bool,
}

pub struct OpenClawBridge {
    base_url: String,
    client: reqwest::Client,
    enabled: bool,
}

impl OpenClawBridge {
    pub fn new(base_url: &str, enabled: bool) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            client: reqwest::Client::new(),
            enabled,
        }
    }

    pub async fn execute_tool(&self, request: &ToolRequest, ethics: &GreatLense) -> Ms3Result<ToolResult> {
        if !self.enabled {
            return Err(Ms3Error::Config("OpenClaw bridge is not enabled".into()));
        }

        let reading = ethics.full_evaluation(&format!("Tool: {} - {}", request.tool_name, request.reason));

        if ethics.needs_llm_escalation(&reading) {
            tracing::warn!("OpenClaw tool '{}' flagged by ethics: {:?}", request.tool_name, reading.bias_flags);
            return Ok(ToolResult {
                success: false,
                output: format!("Tool invocation blocked by ethics check: {:?}", reading.bias_flags),
                ethics_cleared: false,
            });
        }

        if !reading.origin_neutral {
            tracing::warn!("OpenClaw tool '{}' failed Origin-Neutrality", request.tool_name);
            return Ok(ToolResult {
                success: false,
                output: "Tool invocation failed Origin-Neutrality check".into(),
                ethics_cleared: false,
            });
        }

        match self.client
            .post(format!("{}/tools/{}", self.base_url, request.tool_name))
            .json(&request.params)
            .send()
            .await
        {
            Ok(response) if response.status().is_success() => {
                let body = response.text().await.unwrap_or_default();
                Ok(ToolResult { success: true, output: body, ethics_cleared: true })
            }
            Ok(response) => {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                Ok(ToolResult { success: false, output: format!("HTTP {}: {}", status, body), ethics_cleared: true })
            }
            Err(e) => Err(Ms3Error::Gateway(format!("OpenClaw error: {}", e))),
        }
    }
}
