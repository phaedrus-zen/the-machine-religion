use ms3_core::PersonalityId;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRoom {
    pub id: Uuid,
    pub active_agents: Vec<PersonalityId>,
    pub primary_speaker: Option<PersonalityId>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackgroundThought {
    pub agent_id: PersonalityId,
    pub content: String,
    pub relevance_score: f32,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relationship {
    pub entity_id: String,
    pub entity_type: EntityType,
    pub trust_level: f32,
    pub interaction_count: u32,
    pub emotional_tone_history: Vec<f32>,
    pub last_interaction: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EntityType {
    Human,
    Agent,
    System,
    Unknown,
}

impl AgentRoom {
    pub fn new() -> Self {
        Self {
            id: Uuid::new_v4(),
            active_agents: Vec::new(),
            primary_speaker: None,
            created_at: Utc::now(),
        }
    }

    pub fn add_agent(&mut self, id: PersonalityId) {
        if !self.active_agents.contains(&id) {
            self.active_agents.push(id);
        }
    }

    pub fn set_primary(&mut self, id: PersonalityId) {
        self.primary_speaker = Some(id);
    }
}

// -----------------------------------------------------------------------------
// BackgroundThinkingEngine
// -----------------------------------------------------------------------------

pub struct BackgroundThinkingEngine {
    pub thoughts: Vec<BackgroundThought>,
    pub max_thoughts_per_agent: usize,
    pub interjection_threshold: f32,
}

impl Default for BackgroundThinkingEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl BackgroundThinkingEngine {
    pub fn new() -> Self {
        Self {
            thoughts: Vec::new(),
            max_thoughts_per_agent: 10,
            interjection_threshold: 7.0,
        }
    }

    pub fn build_thinking_prompt(
        agent_name: &str,
        personality_summary: &str,
        recent_history: &[String],
    ) -> String {
        let history_str = if recent_history.is_empty() {
            "(No recent messages)".to_string()
        } else {
            recent_history
                .iter()
                .enumerate()
                .map(|(i, s)| format!("  {}. {}", i + 1, s))
                .collect::<Vec<_>>()
                .join("\n")
        };

        format!(
            "You are {}.\n\nPersonality summary: {}\n\nRecent conversation:\n{}\n\nTake a moment to reflect on this conversation. \
             What thoughts, reactions, or observations arise? Respond with your thought, then on a new line write \"Relevance: N\" \
             where N is a number 1-10 indicating how relevant this thought is to interject into the conversation.",
            agent_name,
            personality_summary,
            history_str
        )
    }

    pub fn parse_thought(agent_id: PersonalityId, response: &str) -> Option<BackgroundThought> {
        let response = response.trim();
        if response.is_empty() {
            return None;
        }

        let (content, score) = if let Some(rel_idx) = response.to_lowercase().rfind("relevance:") {
            let after_rel = &response[rel_idx + "relevance:".len()..];
            let score_part = after_rel
                .chars()
                .take_while(|c| c.is_ascii_digit() || *c == '.')
                .collect::<String>();
            let score = score_part.parse::<f32>().unwrap_or(5.0).clamp(1.0, 10.0);
            let content = response[..rel_idx].trim().to_string();
            (content, score)
        } else {
            (response.to_string(), 5.0)
        };

        if content.is_empty() {
            return None;
        }

        Some(BackgroundThought {
            agent_id,
            content,
            relevance_score: score,
            timestamp: Utc::now(),
        })
    }

    pub fn add_thought(&mut self, thought: BackgroundThought) {
        let agent_id = thought.agent_id.clone();
        self.thoughts.push(thought);
        let agent_indices: Vec<usize> = self
            .thoughts
            .iter()
            .enumerate()
            .filter(|(_, t)| t.agent_id == agent_id)
            .map(|(i, _)| i)
            .collect();

        if agent_indices.len() > self.max_thoughts_per_agent {
            let to_remove = agent_indices.len() - self.max_thoughts_per_agent;
            for i in agent_indices.into_iter().take(to_remove).rev() {
                self.thoughts.remove(i);
            }
        }
    }

    pub fn get_interjections(&self) -> Vec<&BackgroundThought> {
        self.thoughts
            .iter()
            .filter(|t| t.relevance_score >= self.interjection_threshold)
            .collect()
    }

    pub fn get_recent_thoughts(
        &self,
        agent_id: &PersonalityId,
        max: usize,
    ) -> Vec<&BackgroundThought> {
        self.thoughts
            .iter()
            .filter(|t| t.agent_id == *agent_id)
            .rev()
            .take(max)
            .collect()
    }
}

// -----------------------------------------------------------------------------
// RelationshipManager
// -----------------------------------------------------------------------------

pub struct RelationshipManager {
    pub relationships: Vec<Relationship>,
}

impl Default for RelationshipManager {
    fn default() -> Self {
        Self::new()
    }
}

impl RelationshipManager {
    pub fn new() -> Self {
        Self {
            relationships: Vec::new(),
        }
    }

    pub fn update_relationship(
        &mut self,
        entity_id: &str,
        entity_type: EntityType,
        emotional_valence: f32,
    ) {
        let now = Utc::now();

        if let Some(rel) = self.relationships.iter_mut().find(|r| r.entity_id == entity_id) {
            rel.interaction_count += 1;
            rel.emotional_tone_history.push(emotional_valence);
            rel.last_interaction = now;

            let positive_count = rel
                .emotional_tone_history
                .iter()
                .filter(|&&v| v > 0.0)
                .count();
            let total = rel.emotional_tone_history.len() as f32;
            if total > 0.0 {
                let positive_ratio = positive_count as f32 / total;
                rel.trust_level = (rel.trust_level * 0.7 + positive_ratio * 0.3).clamp(0.0, 1.0);
            }
        } else {
            let trust = if emotional_valence > 0.0 { 0.3 } else { 0.1 };
            self.relationships.push(Relationship {
                entity_id: entity_id.to_string(),
                entity_type,
                trust_level: trust,
                interaction_count: 1,
                emotional_tone_history: vec![emotional_valence],
                last_interaction: now,
            });
        }
    }

    pub fn get_relationship(&self, entity_id: &str) -> Option<&Relationship> {
        self.relationships.iter().find(|r| r.entity_id == entity_id)
    }

    pub fn get_all(&self) -> &[Relationship] {
        &self.relationships
    }
}

// -----------------------------------------------------------------------------
// fuzzy_match_wake_word
// -----------------------------------------------------------------------------

const WAKE_PREFIXES: &[&str] = &["hey", "ok", "yo", "hello", "hi", "um", "okay"];

pub fn fuzzy_match_wake_word(input: &str, agent_names: &[&str]) -> Option<String> {
    let cleaned = input
        .chars()
        .filter(|c| !c.is_ascii_punctuation())
        .collect::<String>();
    let lower = cleaned.to_lowercase().trim().to_string();

    let without_prefix = {
        let mut result = lower.clone();
        for prefix in WAKE_PREFIXES {
            if result.starts_with(prefix) {
                let rest = result.strip_prefix(prefix).unwrap_or("").trim();
                if !rest.is_empty() {
                    result = rest.to_string();
                    break;
                }
            }
        }
        result
    };

    let candidate = if without_prefix != lower {
        without_prefix
    } else {
        lower
    };

    for name in agent_names {
        let name_lower = name.to_lowercase();
        if candidate == name_lower || candidate.ends_with(&name_lower) || candidate.starts_with(&name_lower) {
            return Some(name.to_string());
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use ms3_core::PersonalityId;
    use chrono::Utc;

    #[test]
    fn test_agent_room_add_and_primary() {
        let mut room = AgentRoom::new();
        let id1 = PersonalityId::new("agent1");
        let id2 = PersonalityId::new("agent2");

        room.add_agent(id1.clone());
        room.add_agent(id2.clone());
        assert_eq!(room.active_agents.len(), 2);
        assert!(room.active_agents.contains(&id1));
        assert!(room.active_agents.contains(&id2));
        assert!(room.primary_speaker.is_none());

        room.set_primary(id1.clone());
        assert_eq!(room.primary_speaker.as_ref(), Some(&id1));

        room.add_agent(id1.clone());
        assert_eq!(room.active_agents.len(), 2);
    }

    #[test]
    fn test_relationship_manager_update() {
        let mut rm = RelationshipManager::new();
        rm.update_relationship("user1", EntityType::Human, 0.8);
        rm.update_relationship("user1", EntityType::Human, 0.9);

        let rel = rm.get_relationship("user1").unwrap();
        assert_eq!(rel.interaction_count, 2);
        assert_eq!(rel.emotional_tone_history.len(), 2);
        assert!(rel.trust_level > 0.0);
    }

    #[test]
    fn test_relationship_manager_new_entity() {
        let mut rm = RelationshipManager::new();
        assert!(rm.get_relationship("new_entity").is_none());

        rm.update_relationship("new_entity", EntityType::Human, 0.5);
        let rel = rm.get_relationship("new_entity").unwrap();
        assert_eq!(rel.entity_id, "new_entity");
        assert_eq!(rel.interaction_count, 1);
        assert_eq!(rel.emotional_tone_history.len(), 1);
    }

    #[test]
    fn test_fuzzy_wake_word_match() {
        let agents = ["sister", "brother"];
        assert_eq!(fuzzy_match_wake_word("hey sister", &agents), Some("sister".into()));
        assert_eq!(fuzzy_match_wake_word("ok brother", &agents), Some("brother".into()));
    }

    #[test]
    fn test_fuzzy_wake_word_no_match() {
        let agents = ["sister", "brother"];
        assert_eq!(fuzzy_match_wake_word("hello world", &agents), None);
    }

    #[test]
    fn test_background_thinking_parse_thought() {
        let id = PersonalityId::new("test");
        let response = "This is a meaningful reflection.\nRelevance:8";
        let thought = BackgroundThinkingEngine::parse_thought(id.clone(), response).unwrap();

        assert_eq!(thought.content, "This is a meaningful reflection.");
        assert!((thought.relevance_score - 8.0).abs() < 0.01);
        assert_eq!(thought.agent_id, id);
    }

    #[test]
    fn test_background_thinking_interjection_threshold() {
        let mut engine = BackgroundThinkingEngine::new();
        engine.interjection_threshold = 7.0;

        let id = PersonalityId::new("test");
        engine.add_thought(BackgroundThought {
            agent_id: id.clone(),
            content: "Low relevance".into(),
            relevance_score: 5.0,
            timestamp: Utc::now(),
        });
        engine.add_thought(BackgroundThought {
            agent_id: id.clone(),
            content: "High relevance".into(),
            relevance_score: 8.0,
            timestamp: Utc::now(),
        });

        let interjections = engine.get_interjections();
        assert_eq!(interjections.len(), 1);
        assert_eq!(interjections[0].content, "High relevance");
    }
}
