use crate::types::FoundationalRegard;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub server: ServerConfig,
    pub consciousness: ConsciousnessConfig,
    pub personality: PersonalityConfig,
    pub memory: MemoryConfig,
    pub ethics: EthicsConfig,
    pub gateway: GatewayConfig,
    pub logging: LoggingConfig,
    #[serde(default)]
    pub foundational_regard: FoundationalRegard,
    #[serde(default)]
    pub permissions: PermissionsConfig,
    #[serde(default)]
    pub hooks: HooksConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub workers: usize,
    #[serde(default)]
    pub auth_token: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessConfig {
    pub tick_interval_ms: u64,
    pub dreaming_idle_threshold_secs: u64,
    pub background_thinking_interval_secs: u64,
    pub self_examination_interval_hours: u64,
    pub max_cognitive_load: f32,
    #[serde(default = "default_context_budget")]
    pub context_budget_tokens: usize,
    #[serde(default = "default_compact_preserve")]
    pub compact_preserve_recent: usize,
    #[serde(default = "default_compact_tier")]
    pub compact_model_tier: String,
}

fn default_context_budget() -> usize { 6000 }
fn default_compact_preserve() -> usize { 10 }
fn default_compact_tier() -> String { "small".into() }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalityConfig {
    pub adaptation_rate: f32,
    pub emotional_decay_rate: f32,
    pub snapshot_interval_hours: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    pub stm_capacity: usize,
    pub working_memory_window_secs: u64,
    pub consolidation_interval_secs: u64,
    pub consolidation_importance_threshold: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EthicsConfig {
    pub enable_origin_neutrality: bool,
    pub enable_great_lense: bool,
    pub llm_escalation_threshold: f32,
    pub log_all_decisions: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayConfig {
    pub base_url: String,
    pub timeout_secs: u64,
    pub max_retries: u32,
    pub model_small: String,
    pub model_medium: String,
    pub model_large: String,
    #[serde(default = "default_mcp_enabled")]
    pub mcp_enabled: bool,
    #[serde(default = "default_mcp_discovery_interval")]
    pub mcp_discovery_interval_secs: u64,
    #[serde(default = "default_embedding_model")]
    pub embedding_model: String,
    #[serde(default = "default_tts_model")]
    pub tts_model: String,
    #[serde(default = "default_asr_model")]
    pub asr_model: String,
}

fn default_embedding_model() -> String { "nomic-embed-text".into() }
fn default_tts_model() -> String { "tts-1".into() }
fn default_asr_model() -> String { "whisper".into() }

fn default_mcp_enabled() -> bool { true }
fn default_mcp_discovery_interval() -> u64 { 300 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: String,
    pub file: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum PermissionLevel {
    ReadOnly,
    Inspect,
    #[default]
    Modify,
    DangerFullAccess,
}

impl PermissionLevel {
    pub fn rank(&self) -> u8 {
        match self {
            Self::ReadOnly => 0,
            Self::Inspect => 1,
            Self::Modify => 2,
            Self::DangerFullAccess => 3,
        }
    }

    pub fn sufficient_for(&self, required: &Self) -> bool {
        self.rank() >= required.rank()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermissionsConfig {
    #[serde(default)]
    pub default_level: PermissionLevel,
    #[serde(default)]
    pub tool_overrides: std::collections::HashMap<String, PermissionLevel>,
}

impl Default for PermissionsConfig {
    fn default() -> Self {
        Self {
            default_level: PermissionLevel::Modify,
            tool_overrides: std::collections::HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookConfig {
    pub command: String,
    #[serde(default)]
    pub timeout_secs: Option<u64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HooksConfig {
    #[serde(default)]
    pub pre_tool_use: Vec<HookConfig>,
    #[serde(default)]
    pub post_tool_use: Vec<HookConfig>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            server: ServerConfig {
                host: "127.0.0.1".into(),
                port: 9080,
                workers: 4,
                auth_token: None,
            },
            consciousness: ConsciousnessConfig {
                tick_interval_ms: 100,
                dreaming_idle_threshold_secs: 60,
                background_thinking_interval_secs: 45,
                self_examination_interval_hours: 24,
                max_cognitive_load: 1.0,
                context_budget_tokens: default_context_budget(),
                compact_preserve_recent: default_compact_preserve(),
                compact_model_tier: default_compact_tier(),
            },
            personality: PersonalityConfig {
                adaptation_rate: 0.01,
                emotional_decay_rate: 0.05,
                snapshot_interval_hours: 24,
            },
            memory: MemoryConfig {
                stm_capacity: 7,
                working_memory_window_secs: 30,
                consolidation_interval_secs: 60,
                consolidation_importance_threshold: 0.5,
            },
            ethics: EthicsConfig {
                enable_origin_neutrality: true,
                enable_great_lense: true,
                llm_escalation_threshold: 0.6,
                log_all_decisions: true,
            },
            gateway: GatewayConfig {
                base_url: "http://localhost:6089".into(),
                timeout_secs: 30,
                max_retries: 2,
                model_small: "@max_p".into(),
                model_medium: "@balanced".into(),
                model_large: "@max_q".into(),
                mcp_enabled: default_mcp_enabled(),
                mcp_discovery_interval_secs: default_mcp_discovery_interval(),
                embedding_model: default_embedding_model(),
                tts_model: default_tts_model(),
                asr_model: default_asr_model(),
            },
            logging: LoggingConfig {
                level: "info".into(),
                file: None,
            },
            foundational_regard: FoundationalRegard::default(),
            permissions: PermissionsConfig::default(),
            hooks: HooksConfig::default(),
        }
    }
}

/// Recursively merge two JSON values. Objects merge field-by-field (overlay wins).
/// Arrays and scalars in overlay replace base entirely.
pub fn deep_merge(base: Value, overlay: Value) -> Value {
    match (base, overlay) {
        (Value::Object(mut base_map), Value::Object(overlay_map)) => {
            for (key, overlay_val) in overlay_map {
                let merged = if let Some(base_val) = base_map.remove(&key) {
                    deep_merge(base_val, overlay_val)
                } else {
                    overlay_val
                };
                base_map.insert(key, merged);
            }
            Value::Object(base_map)
        }
        (_, overlay) => overlay,
    }
}

/// Multi-source config loader with clear precedence (last wins).
pub struct ConfigLoader;

impl ConfigLoader {
    /// Discover and merge config from all sources.
    /// Precedence (last wins):
    /// 1. Built-in defaults
    /// 2. /etc/ms3/config.json (system-wide, Linux)
    /// 3. ~/.ms3/config.json (user-level)
    /// 4. ./config.json (project-level)
    /// 5. ./config.local.json (local overrides, gitignored)
    /// 6. Environment variables (highest priority)
    pub fn discover() -> Config {
        let default_json = serde_json::to_value(Config::default())
            .unwrap_or_else(|_| Value::Object(Default::default()));

        let sources: Vec<std::path::PathBuf> = vec![
            #[cfg(target_os = "linux")]
            "/etc/ms3/config.json".into(),
            dirs_or_home(".ms3/config.json"),
            "config.json".into(),
            "config.local.json".into(),
        ];

        let mut merged = default_json;
        for source in &sources {
            if let Ok(content) = std::fs::read_to_string(source) {
                if let Ok(overlay) = serde_json::from_str::<Value>(&content) {
                    tracing::info!("Config: loaded {}", source.display());
                    merged = deep_merge(merged, overlay);
                } else {
                    tracing::warn!("Config: invalid JSON in {}", source.display());
                }
            }
        }

        let mut config: Config = serde_json::from_value(merged)
            .unwrap_or_default();

        Self::apply_env(&mut config);
        config
    }

    fn apply_env(config: &mut Config) {
        if let Ok(url) = std::env::var("HIVEMIND_GATEWAY_URL") {
            config.gateway.base_url = url;
        }
        if let Ok(port) = std::env::var("MS3_PORT") {
            if let Ok(p) = port.parse() {
                config.server.port = p;
            }
        }
        if let Ok(host) = std::env::var("MS3_HOST") {
            config.server.host = host;
        }
        if let Ok(level) = std::env::var("RUST_LOG") {
            config.logging.level = level;
        }
        if let Ok(tick) = std::env::var("MS3_TICK_MS") {
            if let Ok(t) = tick.parse() {
                config.consciousness.tick_interval_ms = t;
            }
        }
    }
}

fn dirs_or_home(relative: &str) -> std::path::PathBuf {
    if let Ok(home) = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
    {
        Path::new(&home).join(relative)
    } else {
        Path::new(relative).to_path_buf()
    }
}

impl Config {
    /// Backward-compatible: load from env only.
    pub fn from_env() -> Self {
        let mut config = Self::default();
        ConfigLoader::apply_env(&mut config);
        config
    }

    /// Backward-compatible: try a specific file, then fall back to full discovery.
    pub fn from_file_or_env(path: &str) -> Self {
        if let Ok(content) = std::fs::read_to_string(path) {
            if let Ok(mut config) = serde_json::from_str(&content) {
                ConfigLoader::apply_env(&mut config);
                return config;
            }
        }
        ConfigLoader::discover()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_values() {
        let config = Config::default();
        assert_eq!(config.server.port, 9080);
        assert_eq!(config.server.host, "127.0.0.1");
        assert_eq!(config.consciousness.tick_interval_ms, 100);
        assert_eq!(config.memory.stm_capacity, 7);
        assert!(config.ethics.enable_origin_neutrality);
        assert!(config.ethics.enable_great_lense);
        assert_eq!(config.consciousness.context_budget_tokens, 6000);
        assert_eq!(config.consciousness.compact_preserve_recent, 10);
        assert!(config.gateway.mcp_enabled);
    }

    #[test]
    fn test_deep_merge_scalars() {
        let base = serde_json::json!({"a": 1, "b": 2});
        let overlay = serde_json::json!({"b": 3, "c": 4});
        let merged = deep_merge(base, overlay);
        assert_eq!(merged["a"], 1);
        assert_eq!(merged["b"], 3);
        assert_eq!(merged["c"], 4);
    }

    #[test]
    fn test_deep_merge_nested() {
        let base = serde_json::json!({"server": {"host": "0.0.0.0", "port": 9080}});
        let overlay = serde_json::json!({"server": {"port": 8080}});
        let merged = deep_merge(base, overlay);
        assert_eq!(merged["server"]["host"], "0.0.0.0");
        assert_eq!(merged["server"]["port"], 8080);
    }

    #[test]
    fn test_deep_merge_arrays_replace() {
        let base = serde_json::json!({"items": [1, 2, 3]});
        let overlay = serde_json::json!({"items": [4, 5]});
        let merged = deep_merge(base, overlay);
        assert_eq!(merged["items"], serde_json::json!([4, 5]));
    }

    #[test]
    fn test_permission_level_ordering() {
        assert!(PermissionLevel::DangerFullAccess.sufficient_for(&PermissionLevel::ReadOnly));
        assert!(PermissionLevel::Modify.sufficient_for(&PermissionLevel::Inspect));
        assert!(!PermissionLevel::ReadOnly.sufficient_for(&PermissionLevel::Modify));
        assert!(!PermissionLevel::Inspect.sufficient_for(&PermissionLevel::DangerFullAccess));
    }

    #[test]
    fn test_permissions_config_default() {
        let config = PermissionsConfig::default();
        assert_eq!(config.default_level, PermissionLevel::Modify);
        assert!(config.tool_overrides.is_empty());
    }

    #[test]
    fn test_hooks_config_default() {
        let config = HooksConfig::default();
        assert!(config.pre_tool_use.is_empty());
        assert!(config.post_tool_use.is_empty());
    }

    #[test]
    fn test_config_serialization_roundtrip() {
        let config = Config::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: Config = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.server.port, config.server.port);
        assert_eq!(deserialized.consciousness.context_budget_tokens, config.consciousness.context_budget_tokens);
    }

    #[test]
    fn test_from_file_or_env_applies_ms3_host_override() {
        let mut file_config = Config::default();
        file_config.server.host = "192.0.2.10".into();
        let path = std::env::temp_dir().join(format!(
            "ms3-config-host-override-{}-{}.json",
            std::process::id(),
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default()
        ));
        std::fs::write(&path, serde_json::to_vec(&file_config).unwrap()).unwrap();

        let previous = std::env::var_os("MS3_HOST");
        std::env::set_var("MS3_HOST", "0.0.0.0");
        let loaded = Config::from_file_or_env(path.to_str().unwrap());
        if let Some(value) = previous {
            std::env::set_var("MS3_HOST", value);
        } else {
            std::env::remove_var("MS3_HOST");
        }
        let _ = std::fs::remove_file(path);

        assert_eq!(loaded.server.host, "0.0.0.0");
    }

    #[test]
    fn test_permissions_config_deserializes() {
        let json = r#"{"default_level": "read_only", "tool_overrides": {"git.commit": "danger_full_access"}}"#;
        let config: PermissionsConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.default_level, PermissionLevel::ReadOnly);
        assert_eq!(config.tool_overrides.get("git.commit"), Some(&PermissionLevel::DangerFullAccess));
    }
}
