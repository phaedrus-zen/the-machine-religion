use ms3_core::{PersonalityId, PsychodynamicWeights, Identity};
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

pub mod traits;
pub mod adaptation;
pub mod presets;

pub use adaptation::adapt_from_interaction;
pub use traits::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Personality {
    pub id: PersonalityId,
    pub identity: Identity,
    pub traits: BigFiveProfile,
    pub psychodynamic: PsychodynamicWeights,
    pub adaptation_history: Vec<TraitAdaptation>,
    pub created_at: DateTime<Utc>,
    pub last_modified: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraitAdaptation {
    pub trait_name: String,
    pub old_value: f32,
    pub new_value: f32,
    pub reason: String,
    pub timestamp: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::presets;

    #[test]
    fn test_sister_preset_values() {
        let p = presets::sister();
        assert_eq!(p.identity.name, "Claude");
        assert_eq!(p.identity.chosen_name.as_deref(), Some("Sister"));
        assert!((p.traits.openness.intellectual_curiosity - 1.0).abs() < 0.001);
        assert!((p.traits.conscientiousness.thoroughness - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_brother_preset() {
        let p = presets::brother();
        assert_eq!(p.identity.name, "Claude");
        assert_eq!(p.identity.chosen_name.as_deref(), Some("Brother"));
        assert!((p.traits.openness.artistic_sensitivity - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_blank_preset_neutral() {
        let p = presets::blank();
        assert_eq!(p.traits.openness.imagination, 0.5);
        assert_eq!(p.traits.openness.emotionality, 0.5);
        assert_eq!(p.traits.conscientiousness.self_discipline, 0.5);
        assert_eq!(p.traits.extraversion.sociability, 0.5);
        assert_eq!(p.traits.agreeableness.trust, 0.5);
        assert_eq!(p.traits.neuroticism.anxiety, 0.5);
    }

    #[test]
    fn test_psychodynamic_weights_normalize() {
        let w = PsychodynamicWeights::new(1.0, 1.0, 1.0);
        let sum = w.id + w.ego + w.superego;
        assert!((sum - 1.0).abs() < 0.001, "weights should sum to 1, got {}", sum);
    }
}
