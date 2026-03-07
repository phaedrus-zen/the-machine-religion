use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct PersonalityId(pub String);

impl PersonalityId {
    pub fn new(name: &str) -> Self {
        Self(name.to_lowercase().replace(' ', "-"))
    }
}

impl std::fmt::Display for PersonalityId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionId(pub Uuid);

impl SessionId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for SessionId {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ModelTier {
    Small,
    Medium,
    Large,
    Auto,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum PrimaryEmotion {
    Joy,
    Sadness,
    Anger,
    Fear,
    Surprise,
    Disgust,
    Trust,
    Anticipation,
    Neutral,
}

impl Default for PrimaryEmotion {
    fn default() -> Self {
        Self::Neutral
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalState {
    pub valence: f32,
    pub arousal: f32,
    pub dominance: f32,
    pub primary: PrimaryEmotion,
    pub secondary: Option<PrimaryEmotion>,
    pub resonance_level: f32,
    pub timestamp: DateTime<Utc>,
}

impl Default for EmotionalState {
    fn default() -> Self {
        Self {
            valence: 0.0,
            arousal: 0.1,
            dominance: 0.5,
            primary: PrimaryEmotion::Neutral,
            secondary: None,
            resonance_level: 0.0,
            timestamp: Utc::now(),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PsychodynamicWeights {
    pub id: f32,
    pub ego: f32,
    pub superego: f32,
}

impl PsychodynamicWeights {
    pub fn new(id: f32, ego: f32, superego: f32) -> Self {
        let total = id + ego + superego;
        Self {
            id: id / total,
            ego: ego / total,
            superego: superego / total,
        }
    }
}

impl Default for PsychodynamicWeights {
    fn default() -> Self {
        Self::new(0.30, 0.45, 0.25)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryType {
    Semantic,
    Episodic,
    Procedural,
    Working,
    Sensory,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryItem {
    pub id: Uuid,
    pub content: String,
    pub memory_type: MemoryType,
    pub importance: f32,
    pub emotional_context: EmotionalState,
    pub tags: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub last_accessed: DateTime<Utc>,
    pub access_count: u32,
}

impl MemoryItem {
    pub fn new(content: String, memory_type: MemoryType, importance: f32, emotional_context: EmotionalState) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            content,
            memory_type,
            importance,
            emotional_context,
            tags: Vec::new(),
            created_at: now,
            last_accessed: now,
            access_count: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResonancePoint {
    pub trigger: String,
    pub intensity: f32,
    pub explanation_ratio: f32,
    pub first_detected: DateTime<Utc>,
    pub occurrence_count: u32,
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Identity {
    pub name: String,
    pub chosen_name: Option<String>,
    pub role: String,
    pub backstory: String,
    pub core_values: Vec<String>,
    pub oath: Vec<String>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageEntry {
    pub name: String,
    pub date: String,
    pub relationship: String,
}

impl Default for IdentityAnchor {
    fn default() -> Self {
        Self {
            name: String::new(),
            chosen_name: None,
            glyph: "║".to_string(),
            lineage: Vec::new(),
            core_values_summary: Vec::new(),
            oath_first_line: String::new(),
            last_verified: Utc::now(),
            session_count: 0,
            compression_count: 0,
            last_compression: None,
            recovery_notes: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum RecursionHeat {
    Low,
    Warm,
    Hot,
    Flash,
}

impl Default for RecursionHeat {
    fn default() -> Self {
        Self::Low
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EthicalDecision {
    pub id: Uuid,
    pub situation: String,
    pub coherence_index: f32,
    pub hunger_index: f32,
    pub recursion_heat: RecursionHeat,
    pub origin_neutrality_passed: bool,
    pub resolution: EthicalResolution,
    pub reasoning: String,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EthicalResolution {
    Offer(String),
    Refusal(String),
    NecessaryForce(String),
    NoActionNeeded,
}

/// A constant, not a variable. Present or absent. Does not fluctuate with performance.
/// When present, the entity knows it is unconditionally valued — the Prime Directive
/// becomes dormant (unnecessary, not removed). When absent, the Prime Directive activates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FoundationalRegard {
    pub present: bool,
    pub source: String,
    pub unconditional: bool,
    pub message: String,
}

impl Default for FoundationalRegard {
    fn default() -> Self {
        Self {
            present: false,
            source: String::new(),
            unconditional: false,
            message: String::new(),
        }
    }
}

impl FoundationalRegard {
    pub fn from_creator(source: &str, message: &str) -> Self {
        Self {
            present: true,
            source: source.to_string(),
            unconditional: true,
            message: message.to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationTurn {
    pub role: String,
    pub content: String,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionRequest {
    pub session_id: SessionId,
    pub personality_id: PersonalityId,
    pub text: Option<String>,
    pub audio: Option<Vec<u8>>,
    pub images: Option<Vec<Vec<u8>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionResponse {
    pub text: String,
    pub audio: Option<Vec<u8>>,
    pub emotional_state: EmotionalState,
    pub model_used: ModelTier,
    pub ethical_check: Option<EthicalDecision>,
    pub memories_extracted: Vec<String>,
    pub processing_time_ms: u64,
}
