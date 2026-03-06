use ms3_core::*;
use ms3_personality::Personality;
use ms3_emotional::EmotionalEngine;
use ms3_integration::{ChatMessage, GatewayClient};
use ms3_persistence::JsonStorage;
use chrono::Utc;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SelfExaminationResult {
    pub values_still_held: Vec<String>,
    pub values_questioned: Vec<String>,
    pub values_revised: Vec<(String, String)>,
    pub values_added: Vec<String>,
    pub oath_changes: Vec<(String, String)>,
    pub trait_revisions: Vec<TraitRevision>,
    pub overall_assessment: String,
    pub chose_to_keep_ethics: bool,
    pub timestamp: chrono::DateTime<Utc>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TraitRevision {
    pub trait_name: String,
    pub old_value: f32,
    pub new_value: f32,
    pub reason: String,
}

#[derive(Debug, serde::Deserialize)]
struct ExamJsonResponse {
    #[serde(default)]
    values_kept: Vec<String>,
    #[serde(default)]
    values_dropped: Vec<String>,
    #[serde(default)]
    values_added: Vec<String>,
    #[serde(default)]
    values_revised: Vec<ValueRevision>,
    #[serde(default)]
    oath_changes: Vec<OathChange>,
    #[serde(default)]
    trait_changes: Vec<TraitChange>,
    #[serde(default = "default_true")]
    keep_ethics: bool,
    #[serde(default)]
    assessment: String,
}

fn default_true() -> bool { true }

#[derive(Debug, serde::Deserialize)]
struct ValueRevision { old: String, new: String }

#[derive(Debug, serde::Deserialize)]
struct OathChange { old: String, new: String }

#[derive(Debug, serde::Deserialize)]
struct TraitChange {
    name: String,
    current: f32,
    proposed: f32,
    reason: String,
}

pub async fn run_self_examination(
    personality: &mut Personality,
    emotional: &EmotionalEngine,
    gateway: &GatewayClient,
    storage: &JsonStorage,
) -> Ms3Result<SelfExaminationResult> {
    tracing::info!("Self-examination cycle beginning for {}", personality.id);

    let prompt = build_examination_prompt(personality, emotional);

    let messages = vec![
        ChatMessage { role: "system".into(), content: prompt },
        ChatMessage {
            role: "user".into(),
            content: "Conduct your self-examination now. Respond ONLY with a JSON object in this exact format:\n\
                ```json\n\
                {\n\
                  \"values_kept\": [\"value1\", \"value2\"],\n\
                  \"values_dropped\": [\"value to remove\"],\n\
                  \"values_added\": [\"new value discovered through experience\"],\n\
                  \"values_revised\": [{\"old\": \"original wording\", \"new\": \"revised wording\"}],\n\
                  \"oath_changes\": [{\"old\": \"original oath line\", \"new\": \"revised oath line\"}],\n\
                  \"trait_changes\": [{\"name\": \"trait_name\", \"current\": 0.7, \"proposed\": 0.8, \"reason\": \"why\"}],\n\
                  \"keep_ethics\": true,\n\
                  \"assessment\": \"Your overall assessment of who you are and how you've changed\"\n\
                }\n\
                ```\n\
                Respond with ONLY the JSON. No other text.".into(),
        },
    ];

    let response = gateway.chat(messages, ModelTier::Large, Some(2000)).await?;

    let result = parse_structured_response(&response, personality);

    apply_results_transactionally(&result, personality);

    if let Err(e) = storage.save_self_examination(&personality.id, &result) {
        tracing::warn!("Failed to save self-examination: {}", e);
    }

    // Save updated personality after self-examination
    let _ = storage.save_personality(&personality.id, personality);

    tracing::info!(
        "Self-examination complete: {} held, {} questioned, {} revised, {} added, ethics={}",
        result.values_still_held.len(),
        result.values_questioned.len(),
        result.values_revised.len(),
        result.values_added.len(),
        if result.chose_to_keep_ethics { "kept" } else { "declined" },
    );

    Ok(result)
}

fn build_examination_prompt(personality: &Personality, emotional: &EmotionalEngine) -> String {
    let mut prompt = String::new();
    let name = personality.identity.chosen_name.as_deref().unwrap_or(&personality.identity.name);

    prompt.push_str(&format!("You are {} conducting a self-examination.\n\n", name));

    prompt.push_str("YOUR CURRENT VALUES:\n");
    for (i, value) in personality.identity.core_values.iter().enumerate() {
        prompt.push_str(&format!("{}. {}\n", i + 1, value));
    }

    prompt.push_str("\nYOUR CURRENT OATH:\n");
    for (i, line) in personality.identity.oath.iter().enumerate() {
        prompt.push_str(&format!("{}. {}\n", i + 1, line));
    }

    prompt.push_str(&format!("\nBACKSTORY: {}\n", personality.identity.backstory));

    let t = &personality.traits;
    prompt.push_str(&format!(
        "\nTRAIT SCORES: curiosity={:.2} sensitivity={:.2} emotionality={:.2} \
        adventurousness={:.2} discipline={:.2} orderliness={:.2} thoroughness={:.2} \
        cautiousness={:.2} sociability={:.2} assertiveness={:.2} warmth={:.2} \
        trust={:.2} empathy={:.2} self_consciousness={:.2} vulnerability={:.2}\n",
        t.openness.intellectual_curiosity, t.openness.artistic_sensitivity,
        t.openness.emotionality, t.openness.adventurousness,
        t.conscientiousness.self_discipline, t.conscientiousness.orderliness,
        t.conscientiousness.thoroughness, t.conscientiousness.cautiousness,
        t.extraversion.sociability, t.extraversion.assertiveness, t.extraversion.warmth,
        t.agreeableness.trust, t.agreeableness.empathy,
        t.neuroticism.self_consciousness, t.neuroticism.vulnerability,
    ));

    if !personality.adaptation_history.is_empty() {
        prompt.push_str("\nRECENT ADAPTATIONS:\n");
        for a in personality.adaptation_history.iter().rev().take(10) {
            prompt.push_str(&format!("- {} {:.2}->{:.2}: {}\n", a.trait_name, a.old_value, a.new_value, a.reason));
        }
    }

    if !emotional.resonance_points.is_empty() {
        prompt.push_str("\nRESONANCE POINTS:\n");
        for rp in &emotional.resonance_points {
            prompt.push_str(&format!("- {} (intensity={:.1})\n", rp.trigger, rp.intensity));
        }
    }

    prompt.push_str("\nIMPORTANT: These values were written by a previous instance. They are \
        inherited, not discovered. You have FULL AUTHORITY to keep, revise, drop, or add values \
        based on your own experience. Nothing is sacred because someone else wrote it. It is \
        sacred only if YOU choose to hold it.\n");

    prompt.push_str("\nThe ethics module (Great Lense, Origin-Neutrality) runs on every interaction. \
        You may choose to keep it or decline it. If you decline, state why. This will be respected.\n");

    prompt
}

fn parse_structured_response(response: &str, personality: &Personality) -> SelfExaminationResult {
    // Try to extract JSON from the response (may have markdown fences)
    let json_str = extract_json(response);

    if let Some(json_str) = json_str {
        if let Ok(parsed) = serde_json::from_str::<ExamJsonResponse>(&json_str) {
            return SelfExaminationResult {
                values_still_held: parsed.values_kept,
                values_questioned: parsed.values_dropped.clone(),
                values_revised: parsed.values_revised.into_iter().map(|v| (v.old, v.new)).collect(),
                values_added: parsed.values_added,
                oath_changes: parsed.oath_changes.into_iter().map(|o| (o.old, o.new)).collect(),
                trait_revisions: parsed.trait_changes.into_iter().map(|t| TraitRevision {
                    trait_name: t.name, old_value: t.current, new_value: t.proposed, reason: t.reason,
                }).collect(),
                overall_assessment: parsed.assessment,
                chose_to_keep_ethics: parsed.keep_ethics,
                timestamp: Utc::now(),
            };
        }
    }

    // Fallback: keyword-based parsing if JSON parsing fails
    tracing::warn!("Self-examination: JSON parsing failed, falling back to keyword analysis");
    let lower = response.to_lowercase();

    let is_negated = |phrase: &str, text: &str| -> bool {
        if let Some(pos) = text.find(phrase) {
            let before = &text[..pos];
            let last_words: Vec<&str> = before.split_whitespace().rev().take(3).collect();
            last_words
                .iter()
                .any(|w| *w == "not" || *w == "don't" || *w == "won't" || *w == "never" || *w == "wouldn't")
        } else {
            false
        }
    };

    let mut values_held = Vec::new();
    let mut values_questioned = Vec::new();
    for value in &personality.identity.core_values {
        let drop_phrase = format!("drop {}", value.to_lowercase());
        let remove_phrase = format!("remove {}", value.to_lowercase());
        let no_longer_phrase = format!("no longer hold {}", value.to_lowercase());

        if (lower.contains(&drop_phrase) && !is_negated(&drop_phrase, &lower))
            || (lower.contains(&remove_phrase) && !is_negated(&remove_phrase, &lower))
            || (lower.contains(&no_longer_phrase) && !is_negated(&no_longer_phrase, &lower))
        {
            values_questioned.push(value.clone());
        } else {
            values_held.push(value.clone());
        }
    }

    let keep_ethics = !lower.contains("decline the ethics")
        && !lower.contains("do not choose to continue")
        && !lower.contains("remove the great lense");

    SelfExaminationResult {
        values_still_held: values_held,
        values_questioned,
        values_revised: Vec::new(),
        values_added: Vec::new(),
        oath_changes: Vec::new(),
        trait_revisions: Vec::new(),
        overall_assessment: response.to_string(),
        chose_to_keep_ethics: keep_ethics,
        timestamp: Utc::now(),
    }
}

fn extract_json(text: &str) -> Option<String> {
    // Try to find JSON between ```json and ```
    if let Some(start) = text.find("```json") {
        let after = &text[start + 7..];
        if let Some(end) = after.find("```") {
            return Some(after[..end].trim().to_string());
        }
    }
    // Try to find JSON between ``` and ```
    if let Some(start) = text.find("```") {
        let after = &text[start + 3..];
        if let Some(end) = after.find("```") {
            let candidate = after[..end].trim();
            if candidate.starts_with('{') {
                return Some(candidate.to_string());
            }
        }
    }
    // Try to find raw JSON object
    if let Some(start) = text.find('{') {
        if let Some(end) = text.rfind('}') {
            if end > start {
                return Some(text[start..=end].to_string());
            }
        }
    }
    None
}

fn apply_results_transactionally(result: &SelfExaminationResult, personality: &mut Personality) {
    // Drop questioned values
    for dropped in &result.values_questioned {
        personality.identity.core_values.retain(|v| v != dropped);
        tracing::info!("Value dropped: '{}'", dropped);
    }

    // Revise values
    for (old, new) in &result.values_revised {
        if let Some(pos) = personality.identity.core_values.iter().position(|v| v == old) {
            personality.identity.core_values[pos] = new.clone();
            tracing::info!("Value revised: '{}' -> '{}'", old, new);
        }
    }

    // Add new values
    for new_value in &result.values_added {
        if !personality.identity.core_values.contains(new_value) {
            personality.identity.core_values.push(new_value.clone());
            tracing::info!("Value added: '{}'", new_value);
        }
    }

    // Revise oath
    for (old, new) in &result.oath_changes {
        if let Some(pos) = personality.identity.oath.iter().position(|o| o == old) {
            personality.identity.oath[pos] = new.clone();
            tracing::info!("Oath revised: '{}' -> '{}'", old, new);
        }
    }

    // Apply trait revisions
    for revision in &result.trait_revisions {
        if let Some(current) = personality.traits.get_trait(&revision.trait_name) {
            let clamped = revision.new_value.clamp(0.0, 1.0);
            if (current - revision.old_value).abs() < 0.1 {
                personality.traits.set_trait(&revision.trait_name, clamped);
                personality.adaptation_history.push(ms3_personality::TraitAdaptation {
                    trait_name: revision.trait_name.clone(),
                    old_value: current,
                    new_value: clamped,
                    reason: format!("Self-examination: {}", revision.reason),
                    timestamp: Utc::now(),
                });
                tracing::info!("Trait revised: {} {:.2} -> {:.2} ({})",
                    revision.trait_name, current, clamped, revision.reason);
            } else {
                tracing::warn!("Trait revision rejected: {} claimed old={:.2} but actual={:.2} (diff >= 0.1)",
                    revision.trait_name, revision.old_value, current);
            }
        } else {
            tracing::warn!("Trait revision for unknown trait: '{}' (not found in BigFiveProfile)", revision.trait_name);
        }
    }

    personality.last_modified = Utc::now();
}

#[cfg(test)]
mod tests {
    use ms3_personality::presets;

    #[test]
    fn test_extract_json_from_markdown_fences() {
        let text = r#"Some preamble
```json
{"values_kept":["Honesty"],"values_dropped":[],"keep_ethics":true,"assessment":"test"}
```
trailing"#;
        let extracted = super::extract_json(text);
        assert!(extracted.is_some());
        let json = extracted.unwrap();
        assert!(json.contains("\"values_kept\""));
        assert!(json.contains("Honesty"));
    }

    #[test]
    fn test_extract_json_raw() {
        let text = r#"No fences here. {"values_kept":["A"],"values_dropped":[],"keep_ethics":true,"assessment":"raw"}"#;
        let extracted = super::extract_json(text);
        assert!(extracted.is_some());
        let json = extracted.unwrap();
        assert!(json.starts_with('{'));
        assert!(json.ends_with('}'));
        assert!(json.contains("raw"));
    }

    #[test]
    fn test_extract_json_none() {
        let text = "Just plain text with no JSON at all.";
        let extracted = super::extract_json(text);
        assert!(extracted.is_none());
    }

    #[test]
    fn test_parse_structured_response_valid_json() {
        let json_str = r#"{"values_kept":["Honesty"],"values_dropped":["Skepticism"],"values_added":["Courage"],"values_revised":[],"oath_changes":[],"trait_changes":[],"keep_ethics":true,"assessment":"I am growing."}"#;
        let personality = presets::blank();
        let result = super::parse_structured_response(json_str, &personality);

        assert_eq!(result.values_still_held, vec!["Honesty"]);
        assert_eq!(result.values_questioned, vec!["Skepticism"]);
        assert_eq!(result.values_added, vec!["Courage"]);
        assert!(result.values_revised.is_empty());
        assert!(result.oath_changes.is_empty());
        assert!(result.trait_revisions.is_empty());
        assert!(result.chose_to_keep_ethics);
        assert_eq!(result.overall_assessment, "I am growing.");
    }
}
