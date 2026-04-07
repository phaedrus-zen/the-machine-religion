use ms3_core::config::{HookConfig, HooksConfig, PermissionLevel};
use ms3_core::{Ms3Error, Ms3Result};
use ms3_ethics::GreatLense;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

use crate::permissions::{PermissionPolicy, PermissionResult};

// ── Types ──

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ToolSource {
    BuiltIn,
    Mcp,
    Plugin,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub input_schema: Value,
    #[serde(default)]
    pub required_permission: PermissionLevel,
    #[serde(default = "default_source")]
    pub source: ToolSource,
}

fn default_source() -> ToolSource { ToolSource::BuiltIn }

impl ToolSpec {
    /// Create a ToolSpec from MCP-discovered tool metadata.
    pub fn from_mcp(
        name: String,
        description: String,
        input_schema: Value,
        required_permission: PermissionLevel,
    ) -> Self {
        Self { name, description, input_schema, required_permission, source: ToolSource::Mcp }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolRequest {
    pub tool_name: String,
    pub input: Value,
    #[serde(default)]
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub success: bool,
    pub output: String,
    pub ethics_cleared: bool,
    #[serde(default)]
    pub hook_feedback: Vec<String>,
}

impl ToolResult {
    pub fn success(output: String) -> Self {
        Self { success: true, output, ethics_cleared: true, hook_feedback: Vec::new() }
    }

    pub fn denied(reason: String) -> Self {
        Self { success: false, output: reason, ethics_cleared: false, hook_feedback: Vec::new() }
    }

    pub fn error(msg: String) -> Self {
        Self { success: false, output: msg, ethics_cleared: true, hook_feedback: Vec::new() }
    }
}

// ── ToolExecutor trait ──

#[async_trait::async_trait]
pub trait ToolExecutor: Send + Sync {
    async fn execute(&self, tool_name: &str, input: &Value) -> Ms3Result<ToolResult>;
    fn handles(&self, tool_name: &str) -> bool;
}

// ── Built-in executor ──

/// Executes built-in MS3 consciousness tools.
/// Holds a Weak reference to Mind (set after Mind construction via `set_mind`)
/// to avoid circular Arc references.
pub struct BuiltInExecutor {
    mind: std::sync::RwLock<std::sync::Weak<crate::Mind>>,
}

impl BuiltInExecutor {
    pub fn new() -> Self {
        Self {
            mind: std::sync::RwLock::new(std::sync::Weak::new()),
        }
    }

    /// Set the Mind reference after Mind is wrapped in Arc.
    pub fn set_mind(&self, mind: std::sync::Weak<crate::Mind>) {
        if let Ok(mut w) = self.mind.write() {
            *w = mind;
        }
    }

    fn get_mind(&self) -> Option<std::sync::Arc<crate::Mind>> {
        self.mind.read().ok().and_then(|w| w.upgrade())
    }

    pub fn tool_specs() -> Vec<ToolSpec> {
        vec![
            ToolSpec {
                name: "ms3.save_state".into(),
                description: "Force save all consciousness state to disk".into(),
                input_schema: serde_json::json!({"type": "object", "properties": {}}),
                required_permission: PermissionLevel::Modify,
                source: ToolSource::BuiltIn,
            },
            ToolSpec {
                name: "ms3.self_examine".into(),
                description: "Trigger a self-examination cycle".into(),
                input_schema: serde_json::json!({"type": "object", "properties": {}}),
                required_permission: PermissionLevel::Modify,
                source: ToolSource::BuiltIn,
            },
            ToolSpec {
                name: "ms3.get_status".into(),
                description: "Get current consciousness state (emotion, memory, cognitive load)".into(),
                input_schema: serde_json::json!({"type": "object", "properties": {}}),
                required_permission: PermissionLevel::ReadOnly,
                source: ToolSource::BuiltIn,
            },
            ToolSpec {
                name: "ms3.switch_personality".into(),
                description: "Hot-swap the active personality preset".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {"preset": {"type": "string"}},
                    "required": ["preset"]
                }),
                required_permission: PermissionLevel::Modify,
                source: ToolSource::BuiltIn,
            },
            ToolSpec {
                name: "ms3.compact_history".into(),
                description: "Manually trigger context compaction".into(),
                input_schema: serde_json::json!({"type": "object", "properties": {}}),
                required_permission: PermissionLevel::Modify,
                source: ToolSource::BuiltIn,
            },
            ToolSpec {
                name: "ms3.delegate".into(),
                description: "Spawn a sub-mind with restricted tools for focused work".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string"},
                        "allowed_tools": {"type": "array", "items": {"type": "string"}},
                        "max_iterations": {"type": "integer", "default": 5}
                    },
                    "required": ["prompt", "allowed_tools"]
                }),
                required_permission: PermissionLevel::Modify,
                source: ToolSource::BuiltIn,
            },
        ]
    }
}

#[async_trait::async_trait]
impl ToolExecutor for BuiltInExecutor {
    async fn execute(&self, tool_name: &str, input: &Value) -> Ms3Result<ToolResult> {
        let mind = self.get_mind();

        match tool_name {
            "ms3.get_status" => {
                if let Some(mind) = mind {
                    let status = mind.get_status().await;
                    Ok(ToolResult::success(serde_json::to_string_pretty(&status).unwrap_or_default()))
                } else {
                    Ok(ToolResult::error("Mind reference not available".into()))
                }
            }
            "ms3.save_state" => {
                if let Some(mind) = mind {
                    mind.save_full_state().await;
                    Ok(ToolResult::success("State saved successfully.".into()))
                } else {
                    Ok(ToolResult::error("Mind reference not available".into()))
                }
            }
            "ms3.self_examine" => {
                if let Some(mind) = mind {
                    match mind.run_self_exam().await {
                        Ok(result) => Ok(ToolResult::success(
                            serde_json::to_string_pretty(&result).unwrap_or_else(|_| result.overall_assessment.clone())
                        )),
                        Err(e) => Ok(ToolResult::error(format!("Self-examination failed: {}", e))),
                    }
                } else {
                    Ok(ToolResult::error("Mind reference not available".into()))
                }
            }
            "ms3.switch_personality" => {
                let preset = input.get("preset")
                    .and_then(|v| v.as_str())
                    .unwrap_or("sister");
                if let Some(mind) = mind {
                    match mind.switch_personality(preset).await {
                        Ok(name) => Ok(ToolResult::success(format!("Switched to: {}", name))),
                        Err(e) => Ok(ToolResult::error(format!("Switch failed: {}", e))),
                    }
                } else {
                    Ok(ToolResult::error("Mind reference not available".into()))
                }
            }
            "ms3.compact_history" => {
                if let Some(mind) = mind {
                    let mut sessions = mind.sessions.lock().await;
                    let history = sessions.entry("default".to_string()).or_insert_with(Vec::new);
                    let before = history.len();
                    mind.compact_session(history).await;
                    let after = history.len();
                    Ok(ToolResult::success(format!(
                        "Compaction complete: {} messages -> {} messages", before, after
                    )))
                } else {
                    Ok(ToolResult::error("Mind reference not available".into()))
                }
            }
            "ms3.delegate" => {
                let allowed = input.get("allowed_tools")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect::<Vec<_>>())
                    .unwrap_or_default();
                let max_iter = input.get("max_iterations")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(5) as usize;
                let prompt = input.get("prompt")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let sub = SubMind::new(allowed.clone(), max_iter);

                if allowed.is_empty() {
                    return Ok(ToolResult::error("No tools specified for delegation. Provide allowed_tools list.".into()));
                }
                if prompt.is_empty() {
                    return Ok(ToolResult::error("No prompt specified for delegation.".into()));
                }

                if let Some(mind) = mind {
                    let registry = mind.tool_registry.lock().await;
                    let available: Vec<String> = registry.list_tools().iter()
                        .filter(|t| sub.is_allowed(&t.name))
                        .map(|t| format!("- {}: {}", t.name, t.description))
                        .collect();
                    drop(registry);

                    let system = format!(
                        "You are a focused sub-mind with restricted capabilities.\n\
                        You may ONLY use these tools:\n{}\n\n\
                        Permission ceiling: {:?}\n\
                        Max iterations: {}\n\
                        Complete the task concisely and return the result.",
                        available.join("\n"),
                        sub.permission_ceiling(),
                        sub.max_iterations()
                    );
                    match mind.gateway.chat_with_history(
                        &system, Vec::new(), &prompt,
                        ms3_core::ModelTier::Medium, Some(1000)
                    ).await {
                        Ok(response) => Ok(ToolResult::success(response)),
                        Err(e) => Ok(ToolResult::error(format!("Delegation failed: {}", e))),
                    }
                } else {
                    Ok(ToolResult::error("Mind reference not available for delegation".into()))
                }
            }
            _ => Err(Ms3Error::Config(format!("Unknown built-in tool: {}", tool_name))),
        }
    }

    fn handles(&self, tool_name: &str) -> bool {
        tool_name.starts_with("ms3.")
            && !tool_name.contains(".memory.")
            && !tool_name.contains(".consciousness.")
            && !tool_name.contains(".identity.")
            && !tool_name.contains(".resonance.")
            && !tool_name.contains(".personality.")
            && !tool_name.contains(".relationships.")
            && !tool_name.contains(".education.")
            && !tool_name.contains(".ethics.")
            && !tool_name.contains(".markdown.")
    }
}

// ── HookRunner ──

pub struct HookRunner {
    pre_hooks: Vec<HookConfig>,
    post_hooks: Vec<HookConfig>,
}

#[derive(Debug)]
pub enum HookOutcome {
    Allow(Vec<String>),
    Deny(String),
}

impl HookRunner {
    pub fn new(config: &HooksConfig) -> Self {
        Self {
            pre_hooks: config.pre_tool_use.clone(),
            post_hooks: config.post_tool_use.clone(),
        }
    }

    pub fn empty() -> Self {
        Self { pre_hooks: Vec::new(), post_hooks: Vec::new() }
    }

    pub async fn run_pre_hooks(&self, tool_name: &str, input: &Value) -> HookOutcome {
        self.run_hooks(&self.pre_hooks, "pre_tool_use", tool_name, input).await
    }

    pub async fn run_post_hooks(&self, tool_name: &str, input: &Value, output: &str) -> HookOutcome {
        let payload = serde_json::json!({
            "tool_name": tool_name,
            "tool_input": input,
            "tool_output": output,
        });
        self.run_hooks(&self.post_hooks, "post_tool_use", tool_name, &payload).await
    }

    async fn run_hooks(&self, hooks: &[HookConfig], event: &str, tool_name: &str, context: &Value) -> HookOutcome {
        let mut feedback = Vec::new();
        let payload = serde_json::json!({
            "hook_event_name": event,
            "tool_name": tool_name,
            "tool_input": context,
        });
        let payload_str = serde_json::to_string(&payload).unwrap_or_default();

        for hook in hooks {
            let timeout = std::time::Duration::from_secs(hook.timeout_secs.unwrap_or(10));
            let result = tokio::time::timeout(timeout, run_shell_hook(&hook.command, &payload_str)).await;

            match result {
                Ok(Ok((code, stdout))) => {
                    if code == 2 {
                        return HookOutcome::Deny(format!("Hook denied: {}", stdout.trim()));
                    }
                    if !stdout.trim().is_empty() {
                        feedback.push(stdout);
                    }
                    if code != 0 {
                        tracing::warn!("Hook '{}' exited with code {}, allowing", hook.command, code);
                    }
                }
                Ok(Err(e)) => {
                    tracing::warn!("Hook '{}' failed: {}", hook.command, e);
                }
                Err(_) => {
                    tracing::warn!("Hook '{}' timed out after {:?}", hook.command, timeout);
                }
            }
        }
        HookOutcome::Allow(feedback)
    }
}

async fn run_shell_hook(command: &str, stdin_payload: &str) -> Result<(i32, String), String> {
    let shell = if cfg!(windows) { "cmd" } else { "sh" };
    let flag = if cfg!(windows) { "/C" } else { "-lc" };

    let mut child = tokio::process::Command::new(shell)
        .arg(flag)
        .arg(command)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to spawn hook: {}", e))?;

    if let Some(mut stdin) = child.stdin.take() {
        use tokio::io::AsyncWriteExt;
        let _ = stdin.write_all(stdin_payload.as_bytes()).await;
        drop(stdin);
    }

    let output = child.wait_with_output().await
        .map_err(|e| format!("Hook wait failed: {}", e))?;

    let code = output.status.code().unwrap_or(-1);
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    Ok((code, stdout))
}

// ── ToolRegistry ──

pub struct ToolRegistry {
    specs: HashMap<String, ToolSpec>,
    executors: Vec<Box<dyn ToolExecutor>>,
    builtin: std::sync::Arc<BuiltInExecutor>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        let builtin = std::sync::Arc::new(BuiltInExecutor::new());
        let mut registry = Self {
            specs: HashMap::new(),
            executors: Vec::new(),
            builtin: builtin.clone(),
        };
        for spec in BuiltInExecutor::tool_specs() {
            registry.specs.insert(spec.name.clone(), spec);
        }
        registry
    }

    /// Set the Mind reference on the BuiltInExecutor after Mind is wrapped in Arc.
    pub fn set_mind_ref(&self, mind: std::sync::Weak<crate::Mind>) {
        self.builtin.set_mind(mind);
    }

    pub fn register_tools(&mut self, tools: Vec<ToolSpec>) {
        for spec in tools {
            if self.specs.contains_key(&spec.name) {
                tracing::warn!("Tool name collision: '{}', skipping", spec.name);
                continue;
            }
            self.specs.insert(spec.name.clone(), spec);
        }
    }

    pub fn add_executor(&mut self, executor: Box<dyn ToolExecutor>) {
        self.executors.push(executor);
    }

    /// Add an MCP executor for dispatching to HiveMind gateway.
    pub fn set_mcp_executor(&mut self, gateway_url: &str) {
        self.executors.push(Box::new(McpExecutor::new(gateway_url)));
    }

    /// Register a dynamic tool at runtime with a closure handler.
    /// The tool is added to the specs and a DynamicExecutor wraps the handler.
    pub fn register_dynamic(
        &mut self,
        spec: ToolSpec,
        handler: Box<dyn Fn(&Value) -> Ms3Result<ToolResult> + Send + Sync>,
    ) {
        let name = spec.name.clone();
        self.specs.insert(name.clone(), spec);
        self.executors.push(Box::new(DynamicExecutor { name, handler }));
    }

    pub fn tool_count_by_source(&self) -> (usize, usize, usize) {
        let mut builtin = 0;
        let mut mcp = 0;
        let mut dynamic = 0;
        for spec in self.specs.values() {
            match spec.source {
                ToolSource::BuiltIn => builtin += 1,
                ToolSource::Mcp => mcp += 1,
                ToolSource::Plugin => dynamic += 1,
            }
        }
        (builtin, mcp, dynamic)
    }

    pub fn get_spec(&self, name: &str) -> Option<&ToolSpec> {
        self.specs.get(name)
    }

    pub fn list_tools(&self) -> Vec<&ToolSpec> {
        self.specs.values().collect()
    }

    pub fn list_tools_filtered(&self, source: Option<&ToolSource>) -> Vec<&ToolSpec> {
        self.specs.values()
            .filter(|s| source.map_or(true, |src| &s.source == src))
            .collect()
    }

    pub async fn dispatch(&self, tool_name: &str, input: &Value) -> Ms3Result<ToolResult> {
        if self.builtin.handles(tool_name) {
            return self.builtin.execute(tool_name, input).await;
        }
        for executor in &self.executors {
            if executor.handles(tool_name) {
                return executor.execute(tool_name, input).await;
            }
        }
        Err(Ms3Error::Config(format!("No executor found for tool: {}", tool_name)))
    }
}

// ── McpExecutor (dispatches to HiveMind MCP gateway) ──

pub struct McpExecutor {
    client: reqwest::Client,
    mcp_url: String,
}

impl McpExecutor {
    pub fn new(gateway_base_url: &str) -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(60))
                .build()
                .unwrap_or_default(),
            mcp_url: format!("{}/v1/mcp", gateway_base_url.trim_end_matches('/')),
        }
    }
}

#[async_trait::async_trait]
impl ToolExecutor for McpExecutor {
    async fn execute(&self, tool_name: &str, input: &Value) -> Ms3Result<ToolResult> {
        let payload = serde_json::json!({
            "jsonrpc": "2.0",
            "id": chrono::Utc::now().timestamp_millis(),
            "method": "tools/call",
            "params": { "name": tool_name, "arguments": input }
        });

        match self.client.post(&self.mcp_url).json(&payload).send().await {
            Ok(resp) => {
                let body: serde_json::Value = resp.json().await
                    .unwrap_or(serde_json::json!({"error": "parse failed"}));
                // Check for JSON-RPC error (ignore null -- spec says null = no error)
                if let Some(err) = body.get("error") {
                    if !err.is_null() {
                        return Ok(ToolResult::error(format!("MCP error: {}", err)));
                    }
                }
                if let Some(result) = body.get("result") {
                    let text = result.get("content")
                        .and_then(|c| c.as_array())
                        .and_then(|arr| arr.first())
                        .and_then(|item| item.get("text"))
                        .and_then(|t| t.as_str())
                        .unwrap_or("");
                    let is_error = result.get("isError").and_then(|v| v.as_bool()).unwrap_or(false);
                    Ok(if is_error { ToolResult::error(text.into()) } else { ToolResult::success(text.into()) })
                } else {
                    Ok(ToolResult::error("No result in MCP response".into()))
                }
            }
            Err(e) => Ok(ToolResult::error(format!("MCP call failed: {}", e))),
        }
    }

    fn handles(&self, tool_name: &str) -> bool {
        !tool_name.starts_with("ms3.save") && !tool_name.starts_with("ms3.self")
            && !tool_name.starts_with("ms3.get_") && !tool_name.starts_with("ms3.switch")
            && !tool_name.starts_with("ms3.compact") && !tool_name.starts_with("ms3.delegate")
    }
}

// ── DynamicExecutor ──

struct DynamicExecutor {
    name: String,
    handler: Box<dyn Fn(&Value) -> Ms3Result<ToolResult> + Send + Sync>,
}

#[async_trait::async_trait]
impl ToolExecutor for DynamicExecutor {
    async fn execute(&self, _tool_name: &str, input: &Value) -> Ms3Result<ToolResult> {
        (self.handler)(input)
    }

    fn handles(&self, tool_name: &str) -> bool {
        tool_name == self.name
    }
}

// ── SubMind (restricted tool execution) ──

pub struct SubMind {
    allowed_tools: Vec<String>,
    permission_ceiling: PermissionLevel,
    max_iterations: usize,
}

impl SubMind {
    pub fn new(allowed_tools: Vec<String>, max_iterations: usize) -> Self {
        Self {
            allowed_tools,
            permission_ceiling: PermissionLevel::Modify,
            max_iterations,
        }
    }

    pub fn with_ceiling(mut self, ceiling: PermissionLevel) -> Self {
        self.permission_ceiling = ceiling;
        self
    }

    pub fn is_allowed(&self, tool_name: &str) -> bool {
        self.allowed_tools.iter().any(|pattern| {
            if pattern.ends_with('*') {
                let prefix = &pattern[..pattern.len() - 1];
                tool_name.starts_with(prefix)
            } else {
                tool_name == pattern
            }
        })
    }

    pub fn max_iterations(&self) -> usize {
        self.max_iterations
    }

    pub fn permission_ceiling(&self) -> &PermissionLevel {
        &self.permission_ceiling
    }
}

// ── Full pipeline execution ──

/// Execute a tool through the full consciousness-gated pipeline:
/// Permission -> PreHooks -> Ethics (Great Lense) -> Dispatch -> PostHooks -> Result
pub async fn execute_tool_pipeline(
    request: &ToolRequest,
    registry: &ToolRegistry,
    permissions: &PermissionPolicy,
    hooks: &HookRunner,
    ethics: &GreatLense,
    event_bus: &dyn crate::events::EventSink,
) -> Ms3Result<ToolResult> {
    let spec = registry.get_spec(&request.tool_name)
        .ok_or_else(|| Ms3Error::Config(format!("Unknown tool: {}", request.tool_name)))?;

    // 1. Permission gate (fast, mechanical)
    match permissions.authorize(&request.tool_name, &spec.required_permission) {
        PermissionResult::Deny(reason) => {
            tracing::info!("Tool '{}' denied by permissions: {}", request.tool_name, reason);
            event_bus.emit(crate::events::ConsciousnessEvent::ToolDenied {
                tool_name: request.tool_name.clone(),
                reason: reason.clone(),
                denied_by: "permission".into(),
            }).await;
            return Ok(ToolResult::denied(reason));
        }
        PermissionResult::Allow => {}
    }

    // 2. PreToolUse hooks
    let pre_feedback = match hooks.run_pre_hooks(&request.tool_name, &request.input).await {
        HookOutcome::Deny(reason) => {
            event_bus.emit(crate::events::ConsciousnessEvent::ToolDenied {
                tool_name: request.tool_name.clone(),
                reason: reason.clone(),
                denied_by: "hook".into(),
            }).await;
            return Ok(ToolResult::denied(reason));
        }
        HookOutcome::Allow(feedback) => feedback,
    };

    // 3. Great Lense evaluation (MS3's unique ethics layer)
    let context = format!("Tool: {} - {}", request.tool_name, request.reason);
    let reading = ethics.full_evaluation(&context);

    event_bus.emit(crate::events::ConsciousnessEvent::EthicsEvaluation {
        context: context.clone(),
        origin_neutral: reading.origin_neutral,
        bias_flags: reading.bias_flags.iter().map(|f| format!("{:?}", f)).collect(),
    }).await;

    if ethics.needs_llm_escalation(&reading) {
        event_bus.emit(crate::events::ConsciousnessEvent::ToolDenied {
            tool_name: request.tool_name.clone(),
            reason: format!("Ethics escalation: {:?}", reading.bias_flags),
            denied_by: "ethics".into(),
        }).await;
        return Ok(ToolResult {
            success: false,
            output: format!("Ethics escalation: {:?}", reading.bias_flags),
            ethics_cleared: false,
            hook_feedback: pre_feedback,
        });
    }

    if !reading.origin_neutral {
        event_bus.emit(crate::events::ConsciousnessEvent::ToolDenied {
            tool_name: request.tool_name.clone(),
            reason: "Origin-Neutrality check failed".into(),
            denied_by: "ethics".into(),
        }).await;
        return Ok(ToolResult {
            success: false,
            output: "Tool invocation failed Origin-Neutrality check".into(),
            ethics_cleared: false,
            hook_feedback: pre_feedback,
        });
    }

    // 4. Dispatch to executor
    let start = std::time::Instant::now();
    let mut result = registry.dispatch(&request.tool_name, &request.input).await?;
    let duration_ms = start.elapsed().as_millis() as u64;
    result.ethics_cleared = true;
    result.hook_feedback = pre_feedback;

    // 5. PostToolUse hooks
    match hooks.run_post_hooks(&request.tool_name, &request.input, &result.output).await {
        HookOutcome::Deny(reason) => {
            result.success = false;
            result.output = format!("{}\n[Post-hook denied: {}]", result.output, reason);
        }
        HookOutcome::Allow(feedback) => {
            result.hook_feedback.extend(feedback);
        }
    }

    event_bus.emit(crate::events::ConsciousnessEvent::ToolExecuted {
        tool_name: request.tool_name.clone(),
        success: result.success,
        ethics_cleared: result.ethics_cleared,
        duration_ms: Some(duration_ms),
    }).await;

    Ok(result)
}
