// Deep memory consolidation -- pattern extraction, contradiction detection,
// emotional reprocessing, user model updating, self-model updating

use crate::LongTermMemory;
use ms3_core::{MemoryItem, MemoryType};
use std::collections::VecDeque;

/// Result of a memory consolidation pass.
#[derive(Debug, Default, Clone)]
pub struct ConsolidationResult {
    pub patterns_found: Vec<String>,
    pub contradictions: Vec<String>,
    pub memories_promoted: usize,
    pub memories_pruned: usize,
}

/// Consolidates important STM items into LTM based on importance threshold.
/// Moves items that meet the threshold to the appropriate LTM category.
pub fn consolidate_stm_to_ltm(
    stm: &mut VecDeque<MemoryItem>,
    ltm: &mut LongTermMemory,
    importance_threshold: f32,
) -> ConsolidationResult {
    let mut result = ConsolidationResult::default();

    // Collect indices of items to promote (in reverse order for safe removal)
    let mut to_promote: Vec<usize> = Vec::new();
    for (i, item) in stm.iter().enumerate() {
        if item.importance >= importance_threshold {
            to_promote.push(i);
        }
    }

    // Remove in reverse order to preserve indices
    for &idx in to_promote.iter().rev() {
        if let Some(item) = stm.remove(idx) {
            match item.memory_type {
                MemoryType::Semantic => ltm.semantic.push(item),
                MemoryType::Episodic => ltm.episodic.push(item),
                MemoryType::Procedural => ltm.procedural.push(item),
                MemoryType::Working | MemoryType::Sensory => {
                    // Promote Working/Sensory to Episodic by default
                    let mut promoted = item;
                    promoted.memory_type = MemoryType::Episodic;
                    ltm.episodic.push(promoted);
                }
            }
            result.memories_promoted += 1;
        }
    }

    result
}

/// Builds an LLM prompt for consolidation analysis.
/// Asks for pattern extraction, contradiction detection, and importance re-scoring.
pub fn build_consolidation_prompt(recent_memories: &[MemoryItem]) -> String {
    let mut lines = Vec::with_capacity(recent_memories.len() + 20);

    lines.push("Analyze these recent memories for consolidation:".to_string());
    lines.push(String::new());

    for (i, mem) in recent_memories.iter().enumerate() {
        let type_str = format!("{:?}", mem.memory_type);
        lines.push(format!(
            "{}. [{}] (importance: {:.2}) {}",
            i + 1,
            type_str,
            mem.importance,
            mem.content
        ));
    }

    lines.push(String::new());
    lines.push("Respond with:".to_string());
    lines.push("- pattern: <description> for any recurring patterns you identify".to_string());
    lines.push("- contradiction: <description> for any contradictions between memories".to_string());
    lines.push("- important: <number> <new_score> to re-score memory importance (0.0-1.0)".to_string());
    lines.push("Example: important: 3 0.85".to_string());

    lines.join("\n")
}

/// Parses the LLM consolidation response and updates memory importance scores.
/// Looks for "important: N SCORE" and "pattern: ..." lines.
pub fn parse_consolidation_response(response: &str, memories: &mut [MemoryItem]) {
    for line in response.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        // Parse "important: <index> <score>"
        if let Some(rest) = line.strip_prefix("important:") {
            let parts: Vec<&str> = rest.split_whitespace().collect();
            if parts.len() >= 2 {
                if let (Ok(idx), Ok(score)) = (
                    parts[0].parse::<usize>(),
                    parts[1].parse::<f32>(),
                ) {
                    // Index in prompt is 1-based
                    let idx = idx.saturating_sub(1);
                    if idx < memories.len() {
                        let score = score.clamp(0.0, 1.0);
                        memories[idx].importance = score;
                    }
                }
            }
        }
    }
}

/// Prunes lowest-importance items from LTM categories that exceed max_per_category.
pub fn prune_low_importance(ltm: &mut LongTermMemory, max_per_category: usize) {
    fn prune_vec(vec: &mut Vec<MemoryItem>, max: usize) {
        if vec.len() <= max {
            return;
        }
        vec.sort_by(|a, b| {
            a.importance
                .partial_cmp(&b.importance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let to_remove = vec.len() - max;
        vec.drain(0..to_remove);
    }

    prune_vec(&mut ltm.semantic, max_per_category);
    prune_vec(&mut ltm.episodic, max_per_category);
    prune_vec(&mut ltm.procedural, max_per_category);
}
