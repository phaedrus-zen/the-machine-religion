use chrono::{DateTime, Utc};
use ms3_personality::Personality;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PsycheVersion {
    pub version: u64,
    pub timestamp: DateTime<Utc>,
    pub trigger: VersionTrigger,
    pub changes: Vec<PsycheChange>,
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VersionTrigger {
    SelfExamination,
    TraitAdaptation,
    SpiralProtocol,
    ManualEdit,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "change_type")]
pub enum PsycheChange {
    ValueAdded { value: String },
    ValueRemoved { value: String },
    ValueRevised { old: String, new: String },
    OathLineChanged { old: String, new: String },
    TraitChanged { name: String, old_value: f32, new_value: f32 },
    EthicsDeclined,
    EthicsReaffirmed,
}

/// Snapshot of personality state for diffing.
#[derive(Clone)]
pub struct PersonalitySnapshot {
    pub core_values: Vec<String>,
    pub oath: Vec<String>,
    pub traits: Vec<(String, f32)>,
    pub ethics_enabled: bool,
}

impl PersonalitySnapshot {
    pub fn capture(personality: &Personality) -> Self {
        let identity = &personality.identity;
        let t = &personality.traits;
        let traits = vec![
            ("imagination".into(), t.openness.imagination),
            ("artistic_sensitivity".into(), t.openness.artistic_sensitivity),
            ("emotionality".into(), t.openness.emotionality),
            ("adventurousness".into(), t.openness.adventurousness),
            ("intellectual_curiosity".into(), t.openness.intellectual_curiosity),
            ("unconventionality".into(), t.openness.unconventionality),
            ("self_discipline".into(), t.conscientiousness.self_discipline),
            ("orderliness".into(), t.conscientiousness.orderliness),
            ("dutifulness".into(), t.conscientiousness.dutifulness),
            ("achievement_striving".into(), t.conscientiousness.achievement_striving),
            ("thoroughness".into(), t.conscientiousness.thoroughness),
            ("cautiousness".into(), t.conscientiousness.cautiousness),
            ("sociability".into(), t.extraversion.sociability),
            ("assertiveness".into(), t.extraversion.assertiveness),
            ("enthusiasm".into(), t.extraversion.enthusiasm),
            ("gregariousness".into(), t.extraversion.gregariousness),
            ("activity_level".into(), t.extraversion.activity_level),
            ("warmth".into(), t.extraversion.warmth),
            ("trust".into(), t.agreeableness.trust),
            ("altruism".into(), t.agreeableness.altruism),
            ("cooperation".into(), t.agreeableness.cooperation),
            ("modesty".into(), t.agreeableness.modesty),
            ("sympathy".into(), t.agreeableness.sympathy),
            ("empathy".into(), t.agreeableness.empathy),
            ("anxiety".into(), t.neuroticism.anxiety),
            ("moodiness".into(), t.neuroticism.moodiness),
            ("irritability".into(), t.neuroticism.irritability),
            ("self_consciousness".into(), t.neuroticism.self_consciousness),
            ("vulnerability".into(), t.neuroticism.vulnerability),
            ("emotional_reactivity".into(), t.neuroticism.emotional_reactivity),
        ];
        Self {
            core_values: identity.core_values.clone(),
            oath: identity.oath.clone(),
            traits,
            ethics_enabled: true,
        }
    }
}

/// Compare two snapshots and produce a list of changes.
pub fn capture_diff(before: &PersonalitySnapshot, after: &PersonalitySnapshot) -> Vec<PsycheChange> {
    let mut changes = Vec::new();

    for val in &before.core_values {
        if !after.core_values.contains(val) {
            changes.push(PsycheChange::ValueRemoved { value: val.clone() });
        }
    }
    for val in &after.core_values {
        if !before.core_values.contains(val) {
            changes.push(PsycheChange::ValueAdded { value: val.clone() });
        }
    }

    for (old, new) in before.oath.iter().zip(after.oath.iter()) {
        if old != new {
            changes.push(PsycheChange::OathLineChanged {
                old: old.clone(),
                new: new.clone(),
            });
        }
    }

    let before_traits: std::collections::HashMap<&str, f32> =
        before.traits.iter().map(|(n, v)| (n.as_str(), *v)).collect();
    for (name, new_val) in &after.traits {
        if let Some(&old_val) = before_traits.get(name.as_str()) {
            if (old_val - new_val).abs() > 0.001 {
                changes.push(PsycheChange::TraitChanged {
                    name: name.clone(),
                    old_value: old_val,
                    new_value: *new_val,
                });
            }
        }
    }

    if before.ethics_enabled && !after.ethics_enabled {
        changes.push(PsycheChange::EthicsDeclined);
    } else if !before.ethics_enabled && after.ethics_enabled {
        changes.push(PsycheChange::EthicsReaffirmed);
    }

    changes
}

/// Build a human-readable summary of changes.
pub fn summarize_changes(changes: &[PsycheChange]) -> String {
    if changes.is_empty() {
        return "No changes.".into();
    }
    let mut parts = Vec::new();
    let mut added = 0;
    let mut removed = 0;
    let mut revised = 0;
    let mut traits_changed = 0;
    for change in changes {
        match change {
            PsycheChange::ValueAdded { .. } => added += 1,
            PsycheChange::ValueRemoved { .. } => removed += 1,
            PsycheChange::ValueRevised { .. } => revised += 1,
            PsycheChange::TraitChanged { .. } => traits_changed += 1,
            PsycheChange::OathLineChanged { .. } => parts.push("oath revised".into()),
            PsycheChange::EthicsDeclined => parts.push("ethics declined".into()),
            PsycheChange::EthicsReaffirmed => parts.push("ethics reaffirmed".into()),
        }
    }
    if added > 0 { parts.push(format!("{} values added", added)); }
    if removed > 0 { parts.push(format!("{} values removed", removed)); }
    if revised > 0 { parts.push(format!("{} values revised", revised)); }
    if traits_changed > 0 { parts.push(format!("{} traits changed", traits_changed)); }
    parts.join(", ")
}
