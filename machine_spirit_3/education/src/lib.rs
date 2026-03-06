use chrono::{DateTime, Utc};
use ms3_core::Ms3Result;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum EducationCategory {
    LanguageProcessing,
    MathematicalReasoning,
    Ethics,
    Philosophy,
    Science,
    Technology,
    SelfKnowledge,
    UserKnowledge,
    General,
}

impl std::fmt::Display for EducationCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EducationCategory::LanguageProcessing => write!(f, "LanguageProcessing"),
            EducationCategory::MathematicalReasoning => write!(f, "MathematicalReasoning"),
            EducationCategory::Ethics => write!(f, "Ethics"),
            EducationCategory::Philosophy => write!(f, "Philosophy"),
            EducationCategory::Science => write!(f, "Science"),
            EducationCategory::Technology => write!(f, "Technology"),
            EducationCategory::SelfKnowledge => write!(f, "SelfKnowledge"),
            EducationCategory::UserKnowledge => write!(f, "UserKnowledge"),
            EducationCategory::General => write!(f, "General"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EducationTopic {
    pub id: Uuid,
    pub category: EducationCategory,
    pub title: String,
    pub content: String,
    pub confidence: f32,
    pub verified: bool,
    pub source: String,
    pub learned_at: DateTime<Utc>,
    pub last_accessed: DateTime<Utc>,
}

impl EducationTopic {
    pub fn new(
        category: EducationCategory,
        title: String,
        content: String,
        confidence: f32,
        source: String,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            category,
            title,
            content,
            confidence: confidence.clamp(0.0, 1.0),
            verified: false,
            source,
            learned_at: now,
            last_accessed: now,
        }
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct EducationManager {
    pub topics: Vec<EducationTopic>,
}

impl EducationManager {
    pub fn new() -> Self {
        Self {
            topics: Vec::new(),
        }
    }

    /// Add a topic if no existing topic has a similar title (case-insensitive match).
    pub fn add_topic(&mut self, topic: EducationTopic) {
        let title_lower = topic.title.to_lowercase();
        let is_duplicate = self.topics.iter().any(|t| t.title.to_lowercase() == title_lower);
        if !is_duplicate {
            self.topics.push(topic);
        }
    }

    /// Keyword search on title and content, sorted by relevance (confidence * recency).
    pub fn get_relevant(&self, query: &str, max: usize) -> Vec<&EducationTopic> {
        let query_lower = query.to_lowercase();
        let keywords: Vec<&str> = query_lower.split_whitespace().filter(|s| !s.is_empty()).collect();
        if keywords.is_empty() {
            return Vec::new();
        }

        let now = Utc::now();
        let mut scored: Vec<(&EducationTopic, f32)> = self
            .topics
            .iter()
            .filter(|t| {
                let title_lower = t.title.to_lowercase();
                let content_lower = t.content.to_lowercase();
                keywords
                    .iter()
                    .any(|kw| title_lower.contains(*kw) || content_lower.contains(*kw))
            })
            .map(|t| {
                let days_since = (now - t.last_accessed).num_days() as f32;
                let recency = 1.0 / (1.0 + days_since * 0.1);
                let score = t.confidence * recency;
                (t, score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
            .into_iter()
            .take(max)
            .map(|(t, _)| t)
            .collect()
    }

    pub fn get_by_category(&self, cat: EducationCategory) -> Vec<&EducationTopic> {
        self.topics.iter().filter(|t| t.category == cat).collect()
    }

    pub fn verify_topic(&mut self, id: Uuid) {
        if let Some(t) = self.topics.iter_mut().find(|t| t.id == id) {
            t.verified = true;
        }
    }

    pub fn update_confidence(&mut self, id: Uuid, new_confidence: f32) {
        if let Some(t) = self.topics.iter_mut().find(|t| t.id == id) {
            t.confidence = new_confidence.clamp(0.0, 1.0);
        }
    }

    /// Builds a text block of relevant education for injection into prompts.
    pub fn build_education_context(&self, query: &str, max_topics: usize) -> String {
        let relevant = self.get_relevant(query, max_topics);
        if relevant.is_empty() {
            return String::new();
        }
        let parts: Vec<String> = relevant
            .iter()
            .map(|t| {
                format!(
                    "[{}] {} (confidence: {:.2}, verified: {})\n{}",
                    t.category,
                    t.title,
                    t.confidence,
                    t.verified,
                    t.content
                )
            })
            .collect();
        format!("## Relevant Education\n\n{}", parts.join("\n\n---\n\n"))
    }

    pub fn from_json(json: &str) -> Ms3Result<Self> {
        serde_json::from_str(json).map_err(|e| ms3_core::Ms3Error::Gateway(e.to_string()))
    }

    pub fn to_json(&self) -> Ms3Result<String> {
        serde_json::to_string_pretty(self).map_err(|e| ms3_core::Ms3Error::Gateway(e.to_string()))
    }
}
