use ms3_core::{EmotionalState, MemoryItem, MemoryType};
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
    pub stm_capacity: usize,
    pub wm_window_secs: i64,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct LongTermMemory {
    pub semantic: Vec<MemoryItem>,
    pub episodic: Vec<MemoryItem>,
    pub procedural: Vec<MemoryItem>,
}

impl MemorySystem {
    pub fn new(stm_capacity: usize, wm_window_secs: u64) -> Self {
        Self {
            stm: VecDeque::with_capacity(stm_capacity),
            working_memory: DashMap::new(),
            ltm: LongTermMemory::default(),
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

    pub fn retrieve_relevant(&self, query: &str, max_results: usize) -> Vec<&MemoryItem> {
        let query_lower = query.to_lowercase();
        let mut results: Vec<(&MemoryItem, f32)> = Vec::new();

        for item in self.ltm.semantic.iter()
            .chain(self.ltm.episodic.iter())
            .chain(self.ltm.procedural.iter())
        {
            let content_lower = item.content.to_lowercase();
            if content_lower.contains(&query_lower) || item.tags.iter().any(|t| query_lower.contains(&t.to_lowercase())) {
                results.push((item, item.importance));
            }
        }

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.into_iter().take(max_results).map(|(item, _)| item).collect()
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
}
