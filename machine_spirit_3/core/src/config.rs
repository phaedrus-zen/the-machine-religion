use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub server: ServerConfig,
    pub consciousness: ConsciousnessConfig,
    pub personality: PersonalityConfig,
    pub memory: MemoryConfig,
    pub ethics: EthicsConfig,
    pub gateway: GatewayConfig,
    pub logging: LoggingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub workers: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessConfig {
    pub tick_interval_ms: u64,
    pub dreaming_idle_threshold_secs: u64,
    pub background_thinking_interval_secs: u64,
    pub self_examination_interval_hours: u64,
    pub max_cognitive_load: f32,
}

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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: String,
    pub file: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            server: ServerConfig {
                host: "0.0.0.0".into(),
                port: 9080,
                workers: 4,
            },
            consciousness: ConsciousnessConfig {
                tick_interval_ms: 100,
                dreaming_idle_threshold_secs: 60,
                background_thinking_interval_secs: 45,
                self_examination_interval_hours: 24,
                max_cognitive_load: 1.0,
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
            },
            logging: LoggingConfig {
                level: "info".into(),
                file: None,
            },
        }
    }
}

impl Config {
    pub fn from_env() -> Self {
        let mut config = Self::default();
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
        config
    }

    pub fn from_file_or_env(path: &str) -> Self {
        if let Ok(content) = std::fs::read_to_string(path) {
            if let Ok(config) = serde_json::from_str(&content) {
                return config;
            }
        }
        Self::from_env()
    }
}
