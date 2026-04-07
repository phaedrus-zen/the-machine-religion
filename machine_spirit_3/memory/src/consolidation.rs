// Deep memory consolidation -- pattern extraction, contradiction detection,
// emotional reprocessing, user model updating, self-model updating

use crate::LongTermMemory;
use ms3_core::{MemoryItem, MemoryType};
use std::collections::{HashMap, HashSet, VecDeque};

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
/// Also captures patterns and contradictions from the response.
pub fn parse_consolidation_response(response: &str, memories: &mut [MemoryItem]) -> ConsolidationParseResult {
    let mut result = ConsolidationParseResult::default();

    for line in response.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        if let Some(rest) = line.strip_prefix("important:") {
            let parts: Vec<&str> = rest.split_whitespace().collect();
            if parts.len() >= 2 {
                if let (Ok(idx), Ok(score)) = (
                    parts[0].parse::<usize>(),
                    parts[1].parse::<f32>(),
                ) {
                    let idx = idx.saturating_sub(1);
                    if idx < memories.len() {
                        let score = score.clamp(0.0, 1.0);
                        memories[idx].importance = score;
                    }
                }
            }
        } else if let Some(rest) = line.strip_prefix("pattern:") {
            let pattern = rest.trim().to_string();
            if !pattern.is_empty() {
                result.patterns.push(pattern);
            }
        } else if let Some(rest) = line.strip_prefix("contradiction:") {
            let contradiction = rest.trim().to_string();
            if !contradiction.is_empty() {
                result.contradictions.push(contradiction);
            }
        }
    }

    result
}

#[derive(Debug, Default)]
pub struct ConsolidationParseResult {
    pub patterns: Vec<String>,
    pub contradictions: Vec<String>,
}

// ── Dream Synthesis ──

#[derive(Debug, Clone, Default)]
pub struct DreamSynthesisResult {
    pub patterns: Vec<String>,
    pub contradictions: Vec<String>,
    pub insights: Vec<DreamInsight>,
}

#[derive(Debug, Clone)]
pub struct DreamInsight {
    pub content: String,
    pub memory_type: MemoryType,
    pub importance: f32,
}

/// Builds a prompt that asks the LLM to generate insights from patterns across
/// both recent STM items and existing LTM memories.
pub fn build_dream_synthesis_prompt(
    stm: &[MemoryItem],
    recent_ltm: &[MemoryItem],
) -> String {
    let mut lines = Vec::new();

    lines.push("You are dreaming. Look across these memories and generate NEW understanding.".into());
    lines.push("Do not summarize. Find patterns, contradictions, and insights.\n".into());

    lines.push("== Recent (short-term) ==".into());
    for (i, mem) in stm.iter().enumerate() {
        lines.push(format!("  STM-{}: [{}] {}", i + 1, format!("{:?}", mem.memory_type), mem.content));
    }

    if !recent_ltm.is_empty() {
        lines.push("\n== Existing (long-term) ==".into());
        for (i, mem) in recent_ltm.iter().enumerate() {
            lines.push(format!("  LTM-{}: [{}] {}", i + 1, format!("{:?}", mem.memory_type), mem.content));
        }
    }

    lines.push("\nRespond with ONLY these line types:".into());
    lines.push("  pattern: <recurring theme or behavior you notice>".into());
    lines.push("  contradiction: <two memories that conflict, cite which ones>".into());
    lines.push("  insight: <new understanding formed> -> <procedural|semantic>".into());
    lines.push("\nExample:".into());
    lines.push("  pattern: The user values honesty in technical assessments".into());
    lines.push("  insight: honest uncertainty is valued more than confident guesses -> procedural".into());
    lines.push("  contradiction: STM-2 says cautious but LTM-4 shows bold action".into());

    lines.join("\n")
}

/// Parse the dream synthesis LLM response into structured results.
pub fn parse_dream_synthesis(response: &str) -> DreamSynthesisResult {
    let mut result = DreamSynthesisResult::default();

    for line in response.lines() {
        let line = line.trim();
        if line.is_empty() { continue; }

        if let Some(rest) = line.strip_prefix("pattern:") {
            let pattern = rest.trim().to_string();
            if !pattern.is_empty() {
                result.patterns.push(pattern);
            }
        } else if let Some(rest) = line.strip_prefix("contradiction:") {
            let contradiction = rest.trim().to_string();
            if !contradiction.is_empty() {
                result.contradictions.push(contradiction);
            }
        } else if let Some(rest) = line.strip_prefix("insight:") {
            let rest = rest.trim();
            if let Some(arrow_pos) = rest.rfind("->") {
                let content = rest[..arrow_pos].trim().to_string();
                let type_str = rest[arrow_pos + 2..].trim().to_lowercase();
                let memory_type = if type_str.contains("procedural") {
                    MemoryType::Procedural
                } else {
                    MemoryType::Semantic
                };
                if !content.is_empty() {
                    result.insights.push(DreamInsight {
                        content,
                        memory_type,
                        importance: 0.85,
                    });
                }
            }
        }
    }

    result
}

// ── Three-Phase Dreaming (inspired by OpenClaw) ──

#[derive(Debug, Clone)]
pub enum DreamPhase {
    Light,
    Rem,
    Deep,
}

#[derive(Debug, Clone)]
pub struct DreamCandidate {
    pub content: String,
    pub concept_tags: Vec<String>,
    pub recall_count: usize,
    pub avg_score: f32,
    pub unique_contexts: usize,
    pub recency: f32,
    pub consolidated_days: usize,
    pub memory_type: MemoryType,
}

#[derive(Debug, Default)]
pub struct LightSleepResult {
    pub recall_signals: Vec<String>,
    pub concept_tags: Vec<String>,
    pub deduplicated_candidates: Vec<DreamCandidate>,
    pub duplicates_removed: usize,
}

#[derive(Debug, Default)]
pub struct RemSleepResult {
    pub themes: Vec<String>,
    pub candidate_scores: Vec<f32>,
    pub phase_signals: Vec<String>,
    pub scored_candidates: Vec<(DreamCandidate, f32)>,
}

#[derive(Debug, Clone)]
pub struct DeepSleepResult {
    pub phase: DreamPhase,
    pub promoted: usize,
    pub insights: Vec<DreamInsight>,
    pub patterns: Vec<String>,
}

impl Default for DeepSleepResult {
    fn default() -> Self {
        Self {
            phase: DreamPhase::Deep,
            promoted: 0,
            insights: Vec::new(),
            patterns: Vec::new(),
        }
    }
}

fn normalized_tokens(text: &str) -> HashSet<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|token| token.len() > 2)
        .map(|token| token.to_string())
        .collect()
}

/// Jaccard similarity between two token sets. Returns 0.0-1.0.
pub fn jaccard_similarity(a: &str, b: &str) -> f32 {
    let a_tokens = normalized_tokens(a);
    let b_tokens = normalized_tokens(b);
    if a_tokens.is_empty() && b_tokens.is_empty() {
        return 1.0;
    }
    let intersection = a_tokens.intersection(&b_tokens).count();
    let union = a_tokens.union(&b_tokens).count();
    if union == 0 { return 0.0; }
    intersection as f32 / union as f32
}

/// Extract concept tags from content (simple keyword extraction).
pub fn extract_concept_tags(content: &str) -> Vec<String> {
    let stop_words: HashSet<&str> = [
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "between", "out", "off", "over",
        "under", "again", "further", "then", "once", "and", "but", "or", "nor",
        "not", "so", "yet", "both", "each", "few", "more", "most", "other",
        "some", "such", "no", "only", "own", "same", "than", "too", "very",
        "just", "because", "if", "when", "while", "that", "this", "it", "i",
        "me", "my", "we", "our", "you", "your", "he", "she", "they", "them",
    ].iter().cloned().collect();

    let mut word_counts: HashMap<String, usize> = HashMap::new();
    for word in content.to_lowercase().split(|c: char| !c.is_alphanumeric()) {
        let word = word.trim();
        if word.len() > 2 && !stop_words.contains(word) {
            *word_counts.entry(word.to_string()).or_insert(0) += 1;
        }
    }

    let mut tags: Vec<(String, usize)> = word_counts.into_iter().collect();
    tags.sort_by(|a, b| b.1.cmp(&a.1));
    tags.into_iter().take(5).map(|(w, _)| w).collect()
}

/// Light Sleep: deduplicate and tag candidates. Cheap, no LLM.
pub fn light_sleep(memories: &[MemoryItem]) -> LightSleepResult {
    let mut candidates: Vec<DreamCandidate> = Vec::new();
    let mut duplicates_removed = 0;
    let mut recall_signals = Vec::new();
    let mut concept_tags = HashSet::new();

    for mem in memories {
        let is_duplicate = candidates.iter().any(|c| jaccard_similarity(&c.content, &mem.content) > 0.7);
        if is_duplicate {
            duplicates_removed += 1;
            continue;
        }

        let mut candidate_tags = extract_concept_tags(&mem.content);
        for tag in &mem.tags {
            if !candidate_tags.iter().any(|existing| existing == tag) {
                candidate_tags.push(tag.clone());
            }
        }
        candidate_tags.truncate(6);

        if mem.importance >= 0.7 {
            recall_signals.push(format!("high-importance:{}", mem.content.chars().take(48).collect::<String>()));
        }
        if mem.access_count >= 2 {
            recall_signals.push(format!("recalled-often:{}", mem.content.chars().take(48).collect::<String>()));
        }
        for tag in &candidate_tags {
            concept_tags.insert(tag.clone());
        }

        let age_days = (chrono::Utc::now() - mem.created_at).num_days().max(0) as usize;
        candidates.push(DreamCandidate {
            content: mem.content.clone(),
            concept_tags: candidate_tags,
            recall_count: mem.access_count.max(1) as usize,
            avg_score: mem.importance,
            unique_contexts: mem.tags.len().max(1),
            recency: 1.0 / (1.0 + (chrono::Utc::now() - mem.last_accessed).num_hours().max(0) as f32 / 24.0),
            consolidated_days: age_days,
            memory_type: mem.memory_type.clone(),
        });
    }

    LightSleepResult {
        recall_signals,
        concept_tags: concept_tags.into_iter().collect(),
        deduplicated_candidates: candidates,
        duplicates_removed,
    }
}

/// REM Sleep: find patterns and score candidates. Cheap, statistical.
pub fn rem_sleep(candidates: &[DreamCandidate]) -> RemSleepResult {
    let mut tag_counts: HashMap<String, usize> = HashMap::new();
    for c in candidates {
        for tag in &c.concept_tags {
            *tag_counts.entry(tag.clone()).or_insert(0) += 1;
        }
    }

    let themes: Vec<String> = tag_counts.iter()
        .filter(|(_, count)| **count >= 2)
        .map(|(tag, count)| format!("{} ({}x)", tag, count))
        .collect();

    let mut candidate_scores = Vec::with_capacity(candidates.len());
    let scored: Vec<(DreamCandidate, f32)> = candidates.iter().map(|c| {
        let recall_strength = (c.recall_count as f32 / 5.0).min(1.0);
        let conceptual_richness = c.concept_tags.len() as f32 / 5.0;
        let consolidation_factor = ((c.consolidated_days + 1) as f32 / 7.0).min(1.0);
        let candidate_truth = recall_strength * c.avg_score * consolidation_factor * conceptual_richness.max(0.2);
        candidate_scores.push(candidate_truth);
        let score = score_candidate(c, conceptual_richness.max(candidate_truth));
        (c.clone(), score)
    }).collect();

    let mut phase_signals = themes.clone();
    phase_signals.extend(
        tag_counts.into_iter()
            .filter(|(_, count)| *count >= 3)
            .map(|(tag, count)| format!("recurring:{}:{}", tag, count))
    );

    RemSleepResult {
        themes,
        candidate_scores,
        phase_signals,
        scored_candidates: scored,
    }
}

/// Score a candidate using 6 weighted signals (OpenClaw-inspired).
pub fn score_candidate(c: &DreamCandidate, conceptual_richness: f32) -> f32 {
    let frequency = (c.recall_count as f32).min(10.0) / 10.0;
    let relevance = c.avg_score;
    let query_diversity = (c.unique_contexts as f32).min(5.0) / 5.0;
    let recency = c.recency;
    let consolidation = (c.consolidated_days as f32).min(5.0) / 5.0;

    0.24 * frequency
        + 0.30 * relevance
        + 0.15 * query_diversity
        + 0.15 * recency
        + 0.10 * consolidation
        + 0.06 * conceptual_richness.min(1.0)
}

/// Deep Sleep threshold gates.
pub struct DeepSleepConfig {
    pub min_score: f32,
    pub min_recall_count: usize,
    pub min_unique_contexts: usize,
}

impl Default for DeepSleepConfig {
    fn default() -> Self {
        Self {
            min_score: 0.3,
            min_recall_count: 1,
            min_unique_contexts: 1,
        }
    }
}

/// Deep Sleep: filter candidates that pass thresholds, ready for LLM promotion.
pub fn deep_sleep_filter(
    scored: &[(DreamCandidate, f32)],
    config: &DeepSleepConfig,
) -> Vec<(DreamCandidate, f32)> {
    scored.iter()
        .filter(|(c, score)| {
            *score >= config.min_score
                && c.recall_count >= config.min_recall_count
                && c.unique_contexts >= config.min_unique_contexts
        })
        .cloned()
        .collect()
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

#[cfg(test)]
mod tests {
    use super::*;
    use ms3_core::EmotionalState;

    fn make_mem(content: &str, mtype: MemoryType, importance: f32) -> MemoryItem {
        MemoryItem::new(content.into(), mtype, importance, EmotionalState::default())
    }

    #[test]
    fn test_parse_dream_synthesis_patterns() {
        let response = "pattern: The user values honesty\npattern: Recurring theme of persistence";
        let result = parse_dream_synthesis(response);
        assert_eq!(result.patterns.len(), 2);
        assert!(result.patterns[0].contains("honesty"));
    }

    #[test]
    fn test_parse_dream_synthesis_insights() {
        let response = "insight: honesty is valued in this relationship -> procedural\ninsight: code quality matters -> semantic";
        let result = parse_dream_synthesis(response);
        assert_eq!(result.insights.len(), 2);
        assert_eq!(result.insights[0].memory_type, MemoryType::Procedural);
        assert_eq!(result.insights[1].memory_type, MemoryType::Semantic);
        assert!(result.insights[0].importance > 0.8);
    }

    #[test]
    fn test_parse_dream_synthesis_contradictions() {
        let response = "contradiction: STM-1 says cautious but LTM-3 shows bold action";
        let result = parse_dream_synthesis(response);
        assert_eq!(result.contradictions.len(), 1);
    }

    #[test]
    fn test_parse_dream_synthesis_mixed() {
        let response = "pattern: recurring theme\ncontradiction: A vs B\ninsight: new understanding -> procedural\nrandom line ignored";
        let result = parse_dream_synthesis(response);
        assert_eq!(result.patterns.len(), 1);
        assert_eq!(result.contradictions.len(), 1);
        assert_eq!(result.insights.len(), 1);
    }

    #[test]
    fn test_build_dream_synthesis_prompt_includes_stm_and_ltm() {
        let stm = vec![make_mem("recent event", MemoryType::Episodic, 0.7)];
        let ltm = vec![make_mem("old fact", MemoryType::Semantic, 0.8)];
        let prompt = build_dream_synthesis_prompt(&stm, &ltm);
        assert!(prompt.contains("recent event"));
        assert!(prompt.contains("old fact"));
        assert!(prompt.contains("pattern:"));
        assert!(prompt.contains("insight:"));
    }

    #[test]
    fn test_jaccard_similarity_identical() {
        assert!(jaccard_similarity("hello world foo", "hello world foo") > 0.99);
    }

    #[test]
    fn test_jaccard_similarity_different() {
        assert!(jaccard_similarity("hello world", "goodbye universe") < 0.01);
    }

    #[test]
    fn test_jaccard_similarity_partial() {
        let sim = jaccard_similarity("the cat sat on the mat", "the dog sat on the rug");
        assert!(sim > 0.3 && sim < 0.7);
    }

    #[test]
    fn test_extract_concept_tags() {
        let tags = extract_concept_tags("The architect builds consciousness frameworks using Rust programming language");
        assert!(!tags.is_empty());
        assert!(tags.len() <= 5);
    }

    #[test]
    fn test_light_sleep_deduplication() {
        let memories = vec![
            make_mem("the cat sat on the mat today", MemoryType::Episodic, 0.5),
            make_mem("the cat sat on the mat yesterday", MemoryType::Episodic, 0.5),
            make_mem("completely different memory about code", MemoryType::Semantic, 0.7),
        ];
        let result = light_sleep(&memories);
        assert!(result.deduplicated_candidates.len() <= 3);
        assert!(result.duplicates_removed <= memories.len());
    }

    #[test]
    fn test_rem_sleep_finds_themes() {
        let candidates = vec![
            DreamCandidate {
                content: "honesty matters in technical work".into(),
                concept_tags: vec!["honesty".into(), "technical".into()],
                recall_count: 3, avg_score: 0.8, unique_contexts: 2, recency: 0.9, consolidated_days: 1,
                memory_type: MemoryType::Procedural,
            },
            DreamCandidate {
                content: "honesty is valued by the architect".into(),
                concept_tags: vec!["honesty".into(), "architect".into()],
                recall_count: 2, avg_score: 0.7, unique_contexts: 1, recency: 0.8, consolidated_days: 0,
                memory_type: MemoryType::Semantic,
            },
        ];
        let result = rem_sleep(&candidates);
        assert!(result.themes.iter().any(|t| t.contains("honesty")));
    }

    #[test]
    fn test_score_candidate_weighted() {
        let c = DreamCandidate {
            content: "test".into(), concept_tags: vec!["test".into()],
            recall_count: 5, avg_score: 0.8, unique_contexts: 3, recency: 0.9, consolidated_days: 2,
            memory_type: MemoryType::Semantic,
        };
        let score = score_candidate(&c, 0.5);
        assert!(score > 0.3);
        assert!(score < 1.0);
    }

    #[test]
    fn test_deep_sleep_filter_threshold() {
        let candidates = vec![
            (DreamCandidate {
                content: "high quality".into(), concept_tags: vec![],
                recall_count: 5, avg_score: 0.9, unique_contexts: 3, recency: 1.0, consolidated_days: 2,
                memory_type: MemoryType::Semantic,
            }, 0.8),
            (DreamCandidate {
                content: "low quality".into(), concept_tags: vec![],
                recall_count: 0, avg_score: 0.1, unique_contexts: 0, recency: 0.1, consolidated_days: 0,
                memory_type: MemoryType::Semantic,
            }, 0.05),
        ];
        let config = DeepSleepConfig { min_score: 0.3, min_recall_count: 1, min_unique_contexts: 1 };
        let passed = deep_sleep_filter(&candidates, &config);
        assert_eq!(passed.len(), 1);
        assert!(passed[0].0.content.contains("high quality"));
    }

    #[test]
    fn test_consolidate_stm_to_ltm() {
        let mut stm = VecDeque::new();
        stm.push_back(make_mem("important", MemoryType::Semantic, 0.9));
        stm.push_back(make_mem("trivial", MemoryType::Semantic, 0.2));
        let mut ltm = LongTermMemory::default();
        let result = consolidate_stm_to_ltm(&mut stm, &mut ltm, 0.8);
        assert_eq!(result.memories_promoted, 1);
        assert_eq!(ltm.semantic.len(), 1);
        assert_eq!(stm.len(), 1);
    }
}
