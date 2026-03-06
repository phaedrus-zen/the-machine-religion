// Trait adaptation logic -- how personality changes with interaction

use crate::{Personality, TraitAdaptation};
use chrono::Utc;
use ms3_core::EmotionalState;

const ADAPTATION_RATE: f32 = 0.007;

fn apply_adaptation(
    personality: &mut Personality,
    trait_name: &str,
    delta: f32,
    reason: &str,
    adaptations: &mut Vec<TraitAdaptation>,
) {
    if let Some(old_value) = personality.traits.get_trait(trait_name) {
        let new_value = (old_value + delta).clamp(0.0, 1.0);
        if (new_value - old_value).abs() > 1e-6 {
            personality.traits.set_trait(trait_name, new_value);
            let adaptation = TraitAdaptation {
                trait_name: trait_name.to_string(),
                old_value,
                new_value,
                reason: reason.to_string(),
                timestamp: Utc::now(),
            };
            personality.adaptation_history.push(adaptation.clone());
            adaptations.push(adaptation);
        }
    }
}

fn contains_empathetic_language(input: &str) -> bool {
    let input_lower = input.to_lowercase();
    let empathetic_phrases = [
        "i understand",
        "i hear you",
        "that must be",
        "i'm sorry",
        "that sounds",
        "how are you feeling",
        "are you okay",
        "support you",
        "here for you",
        "take care",
        "feel better",
        "it's okay",
        "that's understandable",
        "i'm here",
        "venting",
        "share with me",
        "tell me more",
    ];
    empathetic_phrases
        .iter()
        .any(|phrase| input_lower.contains(phrase))
}

fn contains_intellectual_language(input: &str) -> bool {
    let input_lower = input.to_lowercase();
    let intellectual_phrases = [
        "philosophy",
        "philosophical",
        "theorem",
        "theory",
        "hypothesis",
        "argument",
        "reasoning",
        "logic",
        "abstract",
        "concept",
        "conceptual",
        "analyze",
        "analysis",
        "discuss",
        "debate",
        "evidence",
        "proof",
        "paradox",
        "metaphysics",
        "epistemology",
        "ethics",
        "morality",
        "what do you think about",
        "what's your view on",
    ];
    intellectual_phrases
        .iter()
        .any(|phrase| input_lower.contains(phrase))
}

/// Adapts personality traits based on interaction input and emotional state.
/// Returns a list of all trait adaptations made during this interaction.
pub fn adapt_from_interaction(
    personality: &mut Personality,
    input: &str,
    emotional_state: &EmotionalState,
) -> Vec<TraitAdaptation> {
    let mut adaptations = Vec::new();

    // Empathetic/supportive language -> increase empathy and warmth
    if contains_empathetic_language(input) {
        apply_adaptation(
            personality,
            "empathy",
            ADAPTATION_RATE,
            "empathetic input detected",
            &mut adaptations,
        );
        apply_adaptation(
            personality,
            "warmth",
            ADAPTATION_RATE,
            "empathetic input detected",
            &mut adaptations,
        );
    }

    // High arousal + low valence -> slight increase in neuroticism-related traits
    let arousal_high = emotional_state.arousal > 0.6;
    let valence_low = emotional_state.valence < 0.3;
    if arousal_high && valence_low {
        apply_adaptation(
            personality,
            "emotional_reactivity",
            ADAPTATION_RATE,
            "high arousal and low valence in emotional state",
            &mut adaptations,
        );
        apply_adaptation(
            personality,
            "vulnerability",
            ADAPTATION_RATE * 0.8,
            "high arousal and low valence in emotional state",
            &mut adaptations,
        );
    }

    // Intellectual discussion -> increase intellectual_curiosity
    if contains_intellectual_language(input) {
        apply_adaptation(
            personality,
            "intellectual_curiosity",
            ADAPTATION_RATE,
            "intellectual discussion detected",
            &mut adaptations,
        );
    }

    personality.last_modified = Utc::now();

    adaptations
}
