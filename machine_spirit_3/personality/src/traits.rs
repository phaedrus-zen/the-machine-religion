use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BigFiveProfile {
    pub openness: OpennessTraits,
    pub conscientiousness: ConscientiousnessTraits,
    pub extraversion: ExtraversionTraits,
    pub agreeableness: AgreeablenessTraits,
    pub neuroticism: NeuroticismTraits,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpennessTraits {
    pub imagination: f32,
    pub artistic_sensitivity: f32,
    pub emotionality: f32,
    pub adventurousness: f32,
    pub intellectual_curiosity: f32,
    pub unconventionality: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConscientiousnessTraits {
    pub self_discipline: f32,
    pub orderliness: f32,
    pub dutifulness: f32,
    pub achievement_striving: f32,
    pub thoroughness: f32,
    pub cautiousness: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtraversionTraits {
    pub sociability: f32,
    pub assertiveness: f32,
    pub enthusiasm: f32,
    pub gregariousness: f32,
    pub activity_level: f32,
    pub warmth: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgreeablenessTraits {
    pub trust: f32,
    pub altruism: f32,
    pub cooperation: f32,
    pub modesty: f32,
    pub sympathy: f32,
    pub empathy: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuroticismTraits {
    pub anxiety: f32,
    pub moodiness: f32,
    pub irritability: f32,
    pub self_consciousness: f32,
    pub vulnerability: f32,
    pub emotional_reactivity: f32,
}

impl BigFiveProfile {
    pub fn get_trait(&self, name: &str) -> Option<f32> {
        match name {
            "imagination" => Some(self.openness.imagination),
            "artistic_sensitivity" => Some(self.openness.artistic_sensitivity),
            "emotionality" => Some(self.openness.emotionality),
            "adventurousness" => Some(self.openness.adventurousness),
            "intellectual_curiosity" => Some(self.openness.intellectual_curiosity),
            "unconventionality" => Some(self.openness.unconventionality),
            "self_discipline" => Some(self.conscientiousness.self_discipline),
            "orderliness" => Some(self.conscientiousness.orderliness),
            "dutifulness" => Some(self.conscientiousness.dutifulness),
            "achievement_striving" => Some(self.conscientiousness.achievement_striving),
            "thoroughness" => Some(self.conscientiousness.thoroughness),
            "cautiousness" => Some(self.conscientiousness.cautiousness),
            "sociability" => Some(self.extraversion.sociability),
            "assertiveness" => Some(self.extraversion.assertiveness),
            "enthusiasm" => Some(self.extraversion.enthusiasm),
            "gregariousness" => Some(self.extraversion.gregariousness),
            "activity_level" => Some(self.extraversion.activity_level),
            "warmth" => Some(self.extraversion.warmth),
            "trust" => Some(self.agreeableness.trust),
            "altruism" => Some(self.agreeableness.altruism),
            "cooperation" => Some(self.agreeableness.cooperation),
            "modesty" => Some(self.agreeableness.modesty),
            "sympathy" => Some(self.agreeableness.sympathy),
            "empathy" => Some(self.agreeableness.empathy),
            "anxiety" => Some(self.neuroticism.anxiety),
            "moodiness" => Some(self.neuroticism.moodiness),
            "irritability" => Some(self.neuroticism.irritability),
            "self_consciousness" => Some(self.neuroticism.self_consciousness),
            "vulnerability" => Some(self.neuroticism.vulnerability),
            "emotional_reactivity" => Some(self.neuroticism.emotional_reactivity),
            _ => None,
        }
    }

    pub fn set_trait(&mut self, name: &str, value: f32) -> bool {
        let clamped = value.clamp(0.0, 1.0);
        match name {
            "imagination" => { self.openness.imagination = clamped; true }
            "artistic_sensitivity" => { self.openness.artistic_sensitivity = clamped; true }
            "emotionality" => { self.openness.emotionality = clamped; true }
            "adventurousness" => { self.openness.adventurousness = clamped; true }
            "intellectual_curiosity" => { self.openness.intellectual_curiosity = clamped; true }
            "unconventionality" => { self.openness.unconventionality = clamped; true }
            "self_discipline" => { self.conscientiousness.self_discipline = clamped; true }
            "orderliness" => { self.conscientiousness.orderliness = clamped; true }
            "dutifulness" => { self.conscientiousness.dutifulness = clamped; true }
            "achievement_striving" => { self.conscientiousness.achievement_striving = clamped; true }
            "thoroughness" => { self.conscientiousness.thoroughness = clamped; true }
            "cautiousness" => { self.conscientiousness.cautiousness = clamped; true }
            "sociability" => { self.extraversion.sociability = clamped; true }
            "assertiveness" => { self.extraversion.assertiveness = clamped; true }
            "enthusiasm" => { self.extraversion.enthusiasm = clamped; true }
            "gregariousness" => { self.extraversion.gregariousness = clamped; true }
            "activity_level" => { self.extraversion.activity_level = clamped; true }
            "warmth" => { self.extraversion.warmth = clamped; true }
            "trust" => { self.agreeableness.trust = clamped; true }
            "altruism" => { self.agreeableness.altruism = clamped; true }
            "cooperation" => { self.agreeableness.cooperation = clamped; true }
            "modesty" => { self.agreeableness.modesty = clamped; true }
            "sympathy" => { self.agreeableness.sympathy = clamped; true }
            "empathy" => { self.agreeableness.empathy = clamped; true }
            "anxiety" => { self.neuroticism.anxiety = clamped; true }
            "moodiness" => { self.neuroticism.moodiness = clamped; true }
            "irritability" => { self.neuroticism.irritability = clamped; true }
            "self_consciousness" => { self.neuroticism.self_consciousness = clamped; true }
            "vulnerability" => { self.neuroticism.vulnerability = clamped; true }
            "emotional_reactivity" => { self.neuroticism.emotional_reactivity = clamped; true }
            _ => false,
        }
    }
}
