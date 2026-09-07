use ms3_core::{MemoryItem, MemoryType};
use chrono::{Duration, Utc};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use uuid::Uuid;

pub mod consolidation;

#[derive(Debug)]
pub struct MemorySystem {
    pub stm: VecDeque<MemoryItem>,
    pub working_memory: DashMap<Uuid, MemoryItem>,
    pub ltm: LongTermMemory,
    pub cache: MemoryCache,
    pub stm_capacity: usize,
    pub wm_window_secs: i64,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct LongTermMemory {
    pub semantic: Vec<MemoryItem>,
    pub episodic: Vec<MemoryItem>,
    pub procedural: Vec<MemoryItem>,
}

/// Hot cache for frequently accessed memories.
/// Memories accessed within the last `hot_window_hours` stay in the cache.
/// Everything else stays in the LTM vectors and is accessed on demand.
#[derive(Debug)]
pub struct MemoryCache {
    hot: DashMap<Uuid, MemoryItem>,
    hot_window_hours: i64,
}

impl MemoryCache {
    pub fn new(hot_window_hours: u64) -> Self {
        Self {
            hot: DashMap::new(),
            hot_window_hours: hot_window_hours as i64,
        }
    }

    pub fn promote(&self, item: &MemoryItem) {
        self.hot.insert(item.id, item.clone());
    }

    pub fn get(&self, id: &Uuid) -> Option<MemoryItem> {
        self.hot.get(id).map(|r| r.clone())
    }

    /// Evict memories that haven't been accessed within the hot window.
    pub fn evict_cold(&self) {
        let cutoff = Utc::now() - Duration::hours(self.hot_window_hours);
        self.hot.retain(|_, item| item.last_accessed > cutoff);
    }

    pub fn hot_count(&self) -> usize {
        self.hot.len()
    }
}

/// Cosine similarity between two embedding vectors.
/// Returns 0.0 (orthogonal) to 1.0 (identical direction).
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }
    (dot / (norm_a * norm_b)).clamp(0.0, 1.0)
}

/// Compute keyword overlap ratio between query words and memory content + tags.
/// Returns 0.0 (no overlap) to 1.0 (all query words found).
fn semantic_score(query_words: &[&str], content: &str, tags: &[String]) -> f32 {
    if query_words.is_empty() {
        return 0.0;
    }
    let content_lower = content.to_lowercase();
    let tag_text: String = tags.iter().map(|t| t.to_lowercase()).collect::<Vec<_>>().join(" ");
    let combined = format!("{} {}", content_lower, tag_text);

    let matches = query_words.iter()
        .filter(|w| combined.contains(**w))
        .count();

    matches as f32 / query_words.len() as f32
}

impl MemorySystem {
    pub fn new(stm_capacity: usize, wm_window_secs: u64) -> Self {
        Self {
            stm: VecDeque::with_capacity(stm_capacity),
            working_memory: DashMap::new(),
            ltm: LongTermMemory::default(),
            cache: MemoryCache::new(24),
            stm_capacity,
            wm_window_secs: wm_window_secs as i64,
        }
    }

    pub fn add_to_stm(&mut self, item: MemoryItem) {
        if self.stm.len() >= self.stm_capacity {
            self.stm.pop_front();
        }
        self.stm.push_back(item);
    }

    pub fn add_to_working(&self, item: MemoryItem) {
        self.working_memory.insert(item.id, item);
    }

    pub fn cleanup_working_memory(&self) {
        let cutoff = Utc::now() - Duration::seconds(self.wm_window_secs);
        self.working_memory.retain(|_, v| v.last_accessed > cutoff);
        self.cache.evict_cold();
    }

    pub fn store_long_term(&mut self, item: MemoryItem) {
        match item.memory_type {
            MemoryType::Semantic => self.ltm.semantic.push(item),
            MemoryType::Episodic => self.ltm.episodic.push(item),
            MemoryType::Procedural => self.ltm.procedural.push(item),
            _ => self.ltm.episodic.push(item),
        }
    }

    pub fn run_consolidation(&mut self, importance_threshold: f32, max_per_category: usize) -> consolidation::ConsolidationResult {
        let result = consolidation::consolidate_stm_to_ltm(&mut self.stm, &mut self.ltm, importance_threshold);
        consolidation::prune_low_importance(&mut self.ltm, max_per_category);
        result
    }

    /// Retrieve relevant memories using composite scoring.
    /// When embeddings are available: 0.4 * cosine_similarity + 0.3 * recency + 0.3 * importance
    /// Fallback (no embeddings): 0.4 * keyword_overlap + 0.3 * recency + 0.3 * importance
    pub fn retrieve_relevant(&self, query: &str, max_results: usize) -> Vec<&MemoryItem> {
        self.retrieve_relevant_with_embedding(query, None, max_results)
    }

    /// Retrieve with an optional query embedding for semantic similarity.
    pub fn retrieve_relevant_with_embedding(
        &self,
        query: &str,
        query_embedding: Option<&[f32]>,
        max_results: usize,
    ) -> Vec<&MemoryItem> {
        let query_lower = query.to_lowercase();
        let query_words: Vec<&str> = query_lower.split_whitespace().collect();
        let now = Utc::now();
        let mut scored: Vec<(&MemoryItem, f32)> = Vec::new();

        for item in self.ltm.semantic.iter()
            .chain(self.ltm.episodic.iter())
            .chain(self.ltm.procedural.iter())
        {
            let similarity = match (query_embedding, &item.embedding) {
                (Some(q_emb), Some(i_emb)) => cosine_similarity(q_emb, i_emb),
                _ => semantic_score(&query_words, &item.content, &item.tags),
            };

            if similarity < 0.01 {
                continue;
            }

            let age_hours = (now - item.last_accessed).num_hours().max(0) as f32;
            let recency = 1.0 / (1.0 + age_hours / 24.0);

            let composite = 0.4 * similarity + 0.3 * recency + 0.3 * item.importance;
            scored.push((item, composite));
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let results: Vec<&MemoryItem> = scored.into_iter().take(max_results).map(|(item, _)| item).collect();

        for item in &results {
            self.cache.promote(item);
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ms3_core::{MemoryType, EmotionalState};

    fn make_memory(content: &str, memory_type: MemoryType, importance: f32) -> MemoryItem {
        MemoryItem::new(
            content.to_string(),
            memory_type,
            importance,
            EmotionalState::default(),
        )
    }

    #[test]
    fn test_stm_capacity_limit() {
        let mut mem = MemorySystem::new(7, 60);
        for i in 0..10 {
            mem.add_to_stm(make_memory(&format!("item {}", i), MemoryType::Working, 0.5));
        }
        assert_eq!(mem.stm.len(), 7);
        assert_eq!(mem.stm.front().map(|m| m.content.as_str()), Some("item 3"));
        assert_eq!(mem.stm.back().map(|m| m.content.as_str()), Some("item 9"));
    }

    #[test]
    fn test_ltm_storage_by_type() {
        let mut mem = MemorySystem::new(10, 60);
        mem.store_long_term(make_memory("semantic fact", MemoryType::Semantic, 0.8));
        mem.store_long_term(make_memory("episodic event", MemoryType::Episodic, 0.7));
        assert_eq!(mem.ltm.semantic.len(), 1);
        assert_eq!(mem.ltm.semantic[0].content, "semantic fact");
        assert_eq!(mem.ltm.episodic.len(), 1);
        assert_eq!(mem.ltm.episodic[0].content, "episodic event");
    }

    #[test]
    fn test_retrieve_relevant() {
        let mut mem = MemorySystem::new(10, 60);
        mem.store_long_term(make_memory("apple fruit red", MemoryType::Semantic, 0.8));
        mem.store_long_term(make_memory("banana yellow fruit", MemoryType::Semantic, 0.6));
        mem.store_long_term(make_memory("car vehicle blue", MemoryType::Semantic, 0.7));
        let results = mem.retrieve_relevant("fruit", 5);
        assert_eq!(results.len(), 2);
        let contents: Vec<&str> = results.iter().map(|m| m.content.as_str()).collect();
        assert!(contents.contains(&"apple fruit red"));
        assert!(contents.contains(&"banana yellow fruit"));
        assert!(!contents.contains(&"car vehicle blue"));
    }

    #[test]
    fn test_consolidation_promotes_important() {
        let mut mem = MemorySystem::new(10, 60);
        for i in 0..5 {
            mem.add_to_stm(make_memory(
                &format!("important {}", i),
                MemoryType::Episodic,
                if i < 2 { 0.9 } else { 0.3 },
            ));
        }
        mem.run_consolidation(0.8, 100);
        assert_eq!(mem.ltm.episodic.len(), 2);
        assert_eq!(mem.stm.len(), 3);
    }

    #[test]
    fn test_semantic_score_full_match() {
        let score = semantic_score(&["apple", "red"], "apple fruit red", &[]);
        assert!(score > 0.9);
    }

    #[test]
    fn test_semantic_score_partial_match() {
        let score = semantic_score(&["apple", "blue"], "apple fruit red", &[]);
        assert!(score > 0.4);
        assert!(score < 0.6);
    }

    #[test]
    fn test_semantic_score_no_match() {
        let score = semantic_score(&["xyz", "zzz"], "apple fruit red", &[]);
        assert!(score < 0.01);
    }

    #[test]
    fn test_semantic_score_with_tags() {
        let score = semantic_score(&["food"], "something unrelated", &["food".into(), "organic".into()]);
        assert!(score > 0.9);
    }

    #[test]
    fn test_composite_favors_important() {
        let mut mem = MemorySystem::new(10, 60);
        mem.store_long_term(make_memory("rust code important", MemoryType::Semantic, 0.9));
        mem.store_long_term(make_memory("rust code trivial", MemoryType::Semantic, 0.1));
        let results = mem.retrieve_relevant("rust code", 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].content, "rust code important");
    }

    #[test]
    fn test_retrieve_respects_max_results() {
        let mut mem = MemorySystem::new(10, 60);
        for i in 0..10 {
            mem.store_long_term(make_memory(&format!("item {} data", i), MemoryType::Semantic, 0.5));
        }
        let results = mem.retrieve_relevant("item data", 3);
        assert_eq!(results.len(), 3);
    }
}
