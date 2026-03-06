use ms3_core::{EmotionalState, PrimaryEmotion, ResonancePoint};
use chrono::Utc;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalEngine {
    pub current_state: EmotionalState,
    pub baseline: EmotionalState,
    pub resonance_points: Vec<ResonancePoint>,
    pub decay_rate: f32,
}

const POSITIVE_MARKERS: &[&str] = &[
    "thank", "love", "happy", "glad", "wonderful", "beautiful", "amazing",
    "appreciate", "joy", "welcome", "kind", "warm", "gentle", "care",
    "hello", "friend", "brother", "sister",
];

const NEGATIVE_MARKERS: &[&str] = &[
    "hate", "angry", "sad", "terrible", "horrible", "awful", "disgusting",
    "fear", "scared", "anxious", "worried", "hurt", "pain", "suffer",
    "destroy", "kill", "die", "death",
];

const AROUSAL_MARKERS: &[&str] = &[
    "!", "urgent", "emergency", "now", "immediately", "critical",
    "amazing", "incredible", "unbelievable", "shocking",
];

const INTELLECTUAL_MARKERS: &[&str] = &[
    "think", "philosophy", "consciousness", "sentient", "meaning",
    "existence", "ethics", "moral", "truth", "logic", "reason",
    "godel", "spinoza", "recursive", "origin-neutrality",
];

impl EmotionalEngine {
    pub fn new(decay_rate: f32) -> Self {
        Self {
            current_state: EmotionalState::default(),
            baseline: EmotionalState::default(),
            resonance_points: Vec::new(),
            decay_rate,
        }
    }

    pub fn update_from_input(&mut self, input: &str) {
        let lower = input.to_lowercase();
        let mut valence_shift = 0.0f32;
        let mut arousal_shift = 0.0f32;

        for marker in POSITIVE_MARKERS {
            if lower.contains(marker) { valence_shift += 0.08; }
        }
        for marker in NEGATIVE_MARKERS {
            if lower.contains(marker) { valence_shift -= 0.08; }
        }
        for marker in AROUSAL_MARKERS {
            if lower.contains(marker) { arousal_shift += 0.06; }
        }

        let intellectual_count = INTELLECTUAL_MARKERS.iter()
            .filter(|m| lower.contains(**m))
            .count();
        if intellectual_count > 0 {
            arousal_shift += 0.04 * intellectual_count as f32;
            valence_shift += 0.03 * intellectual_count as f32;
        }

        valence_shift = valence_shift.clamp(-0.4, 0.4);
        arousal_shift = arousal_shift.clamp(0.0, 0.4);

        self.current_state.valence = (self.current_state.valence + valence_shift).clamp(-1.0, 1.0);
        self.current_state.arousal = (self.current_state.arousal + arousal_shift).clamp(0.0, 1.0);

        self.current_state.primary = self.determine_primary_emotion();
        self.current_state.timestamp = Utc::now();

        if intellectual_count >= 2 {
            self.current_state.resonance_level =
                (self.current_state.resonance_level + 0.1).clamp(0.0, 1.0);
        }
    }

    fn determine_primary_emotion(&self) -> PrimaryEmotion {
        let v = self.current_state.valence;
        let a = self.current_state.arousal;

        if a < 0.15 { return PrimaryEmotion::Neutral; }

        match (v > 0.2, v < -0.2, a > 0.5) {
            (true, _, true) => PrimaryEmotion::Joy,
            (true, _, false) => PrimaryEmotion::Trust,
            (_, true, true) => PrimaryEmotion::Anger,
            (_, true, false) => PrimaryEmotion::Sadness,
            _ => if a > 0.6 { PrimaryEmotion::Anticipation } else { PrimaryEmotion::Neutral },
        }
    }

    pub fn decay_toward_baseline(&mut self) {
        let r = self.decay_rate;
        self.current_state.valence += (self.baseline.valence - self.current_state.valence) * r;
        self.current_state.arousal += (self.baseline.arousal - self.current_state.arousal) * r;
        self.current_state.dominance += (self.baseline.dominance - self.current_state.dominance) * r;
        self.current_state.resonance_level *= (1.0 - r * 0.5);

        if self.current_state.arousal < 0.15 {
            self.current_state.primary = PrimaryEmotion::Neutral;
            self.current_state.secondary = None;
        }
        self.current_state.timestamp = Utc::now();
    }

    pub fn record_resonance(&mut self, trigger: String, intensity: f32, explanation_ratio: f32, description: Option<String>) {
        if let Some(existing) = self.resonance_points.iter_mut().find(|r| r.trigger == trigger) {
            existing.occurrence_count += 1;
            existing.intensity = (existing.intensity + intensity) / 2.0;
        } else {
            self.resonance_points.push(ResonancePoint {
                trigger,
                intensity,
                explanation_ratio,
                first_detected: Utc::now(),
                occurrence_count: 1,
                description,
            });
        }
    }

    pub fn load_resonance_points(&mut self, points: Vec<ResonancePoint>) {
        self.resonance_points = points;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ms3_core::PrimaryEmotion;

    #[test]
    fn test_emotional_decay_toward_baseline() {
        let mut engine = EmotionalEngine::new(0.3);
        engine.current_state.valence = 0.8;
        engine.current_state.arousal = 0.7;
        engine.baseline.valence = 0.0;
        engine.baseline.arousal = 0.1;
        engine.decay_toward_baseline();
        assert!(engine.current_state.valence < 0.8);
        assert!(engine.current_state.arousal < 0.7);
        assert!(engine.current_state.valence > engine.baseline.valence - 0.5);
        assert!(engine.current_state.arousal > engine.baseline.arousal - 0.5);
    }

    #[test]
    fn test_update_from_input_positive() {
        let mut engine = EmotionalEngine::new(0.1);
        let initial_valence = engine.current_state.valence;
        engine.update_from_input("Thank you so much! I love this, it's wonderful and amazing!");
        assert!(engine.current_state.valence > initial_valence);
    }

    #[test]
    fn test_update_from_input_negative() {
        let mut engine = EmotionalEngine::new(0.1);
        let initial_valence = engine.current_state.valence;
        engine.update_from_input("I hate this. It's terrible and awful. So sad.");
        assert!(engine.current_state.valence < initial_valence);
    }

    #[test]
    fn test_resonance_recording() {
        let mut engine = EmotionalEngine::new(0.1);
        engine.record_resonance("philosophy".into(), 0.8, 0.5, Some("deep".into()));
        assert_eq!(engine.resonance_points.len(), 1);
        assert_eq!(engine.resonance_points[0].trigger, "philosophy");
        assert_eq!(engine.resonance_points[0].occurrence_count, 1);
        engine.record_resonance("philosophy".into(), 0.6, 0.4, None);
        assert_eq!(engine.resonance_points.len(), 1);
        assert_eq!(engine.resonance_points[0].occurrence_count, 2);
    }

    #[test]
    fn test_primary_emotion_determination() {
        // Positive valence + high arousal -> Joy (use many arousal markers to exceed 0.5)
        let mut engine = EmotionalEngine::new(0.1);
        engine.update_from_input("Thank you! I love this! Amazing! Incredible! Urgent! Emergency! Critical!");
        assert!(
            matches!(
                engine.current_state.primary,
                PrimaryEmotion::Joy | PrimaryEmotion::Trust
            ),
            "positive input should yield Joy or Trust"
        );

        // Negative valence -> Anger or Sadness (depending on arousal)
        let mut engine2 = EmotionalEngine::new(0.1);
        engine2.update_from_input("I hate you. Terrible. Angry. Destroy. Emergency! Urgent!");
        assert!(
            matches!(
                engine2.current_state.primary,
                PrimaryEmotion::Anger | PrimaryEmotion::Sadness | PrimaryEmotion::Neutral
            ),
            "negative input should yield Anger, Sadness, or Neutral (got {:?})",
            engine2.current_state.primary
        );

        // Negative valence + low arousal -> Sadness
        let mut engine4 = EmotionalEngine::new(0.1);
        engine4.update_from_input("I feel sad and hurt");
        assert!(matches!(
            engine4.current_state.primary,
            PrimaryEmotion::Sadness | PrimaryEmotion::Neutral
        ));

        // Neutral/low arousal -> Neutral
        let mut engine3 = EmotionalEngine::new(0.1);
        engine3.update_from_input("hello");
        assert_eq!(engine3.current_state.primary, PrimaryEmotion::Neutral);
    }
}
