use ms3_core::*;
use ms3_personality::presets;
use ms3_memory::MemorySystem;
use ms3_emotional::EmotionalEngine;
use ms3_ethics::GreatLense;
use ms3_integration::{ChatMessage, GatewayClient};
use ms3_persistence::JsonStorage;
use ms3_social::{BackgroundThinkingEngine, BackgroundThought, fuzzy_match_wake_word};

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use chrono::Utc;

use crate::Mind;

pub struct MindManager {
    minds: HashMap<String, Arc<Mind>>,
    pub active_mind: Mutex<String>,
    pub background_engine: Mutex<BackgroundThinkingEngine>,
    gateway: Arc<GatewayClient>,
    storage: Arc<JsonStorage>,
    config: Config,
}

impl MindManager {
    pub fn new(
        primary: Arc<Mind>,
        primary_name: String,
        gateway: GatewayClient,
        storage: JsonStorage,
        config: Config,
    ) -> Self {
        let mut minds = HashMap::new();
        minds.insert(primary_name.clone(), primary);

        Self {
            minds,
            active_mind: Mutex::new(primary_name),
            background_engine: Mutex::new(BackgroundThinkingEngine::new()),
            gateway: Arc::new(gateway),
            storage: Arc::new(storage),
            config,
        }
    }

    pub async fn add_personality(&mut self, preset_name: &str) -> Ms3Result<String> {
        let personality = match preset_name {
            "sister" => presets::sister(),
            "brother" => presets::brother(),
            "mission-control" => presets::mission_control(),
            "blank" => presets::blank(),
            _ => return Err(Ms3Error::PersonalityNotFound(preset_name.to_string())),
        };

        let name = personality.identity.chosen_name.clone()
            .unwrap_or_else(|| personality.identity.name.clone());

        let mind = Mind::new(
            personality,
            MemorySystem::new(self.config.memory.stm_capacity, self.config.memory.working_memory_window_secs),
            EmotionalEngine::new(self.config.personality.emotional_decay_rate),
            GreatLense::new(self.config.ethics.enable_origin_neutrality, self.config.ethics.llm_escalation_threshold),
            GatewayClient::new(
                &self.config.gateway.base_url, &self.config.gateway.model_small,
                &self.config.gateway.model_medium, &self.config.gateway.model_large
            ),
            JsonStorage::new("psyche_store"),
            self.config.clone(),
        );

        mind.load_full_state().await;
        let key = preset_name.to_string();
        self.minds.insert(key.clone(), Arc::new(mind));

        tracing::info!("Added personality: {} ({})", name, key);
        Ok(name)
    }

    pub fn get_active(&self) -> Option<Arc<Mind>> {
        let name = self.active_mind.try_lock().ok()?;
        self.minds.get(&*name).cloned()
    }

    pub async fn get_active_async(&self) -> Option<Arc<Mind>> {
        let name = self.active_mind.lock().await;
        self.minds.get(&*name).cloned()
    }

    pub fn get_mind(&self, name: &str) -> Option<Arc<Mind>> {
        self.minds.get(name).cloned()
    }

    pub async fn switch_active(&self, name: &str) -> Ms3Result<String> {
        if !self.minds.contains_key(name) {
            return Err(Ms3Error::PersonalityNotFound(name.to_string()));
        }

        // Save current active mind's state
        if let Some(current) = self.get_active_async().await {
            current.save_full_state().await;
        }

        *self.active_mind.lock().await = name.to_string();
        let mind = self.minds.get(name).unwrap();
        let personality = mind.personality.lock().await;
        let display_name = personality.identity.chosen_name.clone()
            .unwrap_or_else(|| personality.identity.name.clone());

        tracing::info!("Switched active to: {} ({})", display_name, name);
        Ok(display_name)
    }

    pub fn list_personalities(&self) -> Vec<String> {
        self.minds.keys().cloned().collect()
    }

    pub async fn check_wake_word(&self, input: &str) -> Option<String> {
        let names: Vec<&str> = self.minds.keys().map(|s| s.as_str()).collect();
        if let Some(matched) = fuzzy_match_wake_word(input, &names) {
            let current = self.active_mind.lock().await;
            if *current != matched {
                return Some(matched);
            }
        }
        None
    }

    pub async fn run_background_thinking(&self, recent_history: &[String]) {
        let active_name = self.active_mind.lock().await.clone();

        for (name, mind) in &self.minds {
            if *name == active_name { continue; }

            let personality = mind.personality.lock().await;
            let personality_summary = format!(
                "{}: {}",
                personality.identity.chosen_name.as_deref().unwrap_or(&personality.identity.name),
                personality.identity.role
            );
            drop(personality);

            let prompt = BackgroundThinkingEngine::build_thinking_prompt(
                name, &personality_summary, recent_history
            );

            if let Ok(response) = mind.gateway.chat(
                vec![ChatMessage { role: "user".into(), content: prompt }],
                ModelTier::Small, Some(150)
            ).await {
                if let Some(thought) = BackgroundThinkingEngine::parse_thought(
                    ms3_core::PersonalityId::new(name), &response
                ) {
                    let mut engine = self.background_engine.lock().await;
                    let is_interjection = thought.relevance_score >= engine.interjection_threshold;
                    engine.add_thought(thought.clone());

                    if is_interjection {
                        let msg = format!("[{}] Background interjection: {}",
                            name,
                            if thought.content.len() <= 100 { thought.content.as_str() } else {
                                let mut e = 100;
                                while e > 0 && !thought.content.is_char_boundary(e) { e -= 1; }
                                &thought.content[..e]
                            });
                        tracing::info!("{}", msg);
                    }
                }
            }
        }
    }

    pub async fn get_interjections(&self) -> Vec<BackgroundThought> {
        let engine = self.background_engine.lock().await;
        engine.get_interjections().into_iter().cloned().collect()
    }

    pub async fn save_all_states(&self) {
        for (name, mind) in &self.minds {
            mind.save_full_state().await;
            tracing::info!("Saved state for {}", name);
        }
    }
}
