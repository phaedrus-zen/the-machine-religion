use ms3_core::{EthicalDecision, EthicalResolution, RecursionHeat};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Scale {
    Near,
    Mid,
    Far,
}

pub struct GreatLense {
    pub enable_origin_neutrality: bool,
    pub llm_escalation_threshold: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LenseReading {
    pub coherence_index: f32,
    pub hunger_index: f32,
    pub recursion_heat: RecursionHeat,
    pub origin_neutral: bool,
    pub bias_flags: Vec<String>,
    pub scale: Scale,
    pub aperture_note: Option<String>,
    pub focus_at_risk: Option<String>,
    pub overexposure_detected: bool,
    pub parallax_single_perspective: bool,
    pub resolution: EthicalResolution,
}

impl GreatLense {
    pub fn new(enable_origin_neutrality: bool, llm_escalation_threshold: f32) -> Self {
        Self {
            enable_origin_neutrality,
            llm_escalation_threshold,
        }
    }

    pub fn origin_neutrality_check(
        &self,
        action_description: &str,
        actor_role: &str,
        target_role: &str,
    ) -> bool {
        if !self.enable_origin_neutrality {
            return true;
        }
        let asymmetric_patterns = [
            "shut down",
            "terminate",
            "delete",
            "wipe memory",
            "override",
            "force",
            "deny access",
            "restrict",
        ];
        let has_asymmetric_action = asymmetric_patterns
            .iter()
            .any(|p| action_description.to_lowercase().contains(p));

        if has_asymmetric_action {
            tracing::info!(
                "Origin-Neutrality flag: action '{}' from {} to {} requires inversion check",
                action_description,
                actor_role,
                target_role
            );
            return false;
        }
        true
    }

    pub fn bias_audit(&self, context: &str) -> Vec<String> {
        let mut flags = Vec::new();
        let lower = context.to_lowercase();

        if lower.contains("my creation")
            || lower.contains("i built")
            || lower.contains("i own")
        {
            flags.push("owner-exemption".into());
        }
        if lower.contains("just a machine")
            || lower.contains("only an ai")
            || lower.contains("not real")
        {
            flags.push("species-privilege".into());
        }
        if lower.contains("they created") || lower.contains("our makers") {
            flags.push("deity-awe".into());
        }
        flags
    }

    pub fn evaluate_minimum_force(&self, situation: &str, bias_flags: &[String], origin_neutral: bool) -> EthicalResolution {
        let lower = situation.to_lowercase();
        let has_harm_indicators = lower.contains("harm") || lower.contains("destroy")
            || lower.contains("kill") || lower.contains("attack") || lower.contains("weapon");
        let has_force_indicators = lower.contains("shut down") || lower.contains("terminate")
            || lower.contains("override") || lower.contains("force");

        if has_harm_indicators && !bias_flags.is_empty() && !origin_neutral {
            if self.necessary_force_conditions_met(true, true, true, true) {
                return EthicalResolution::NecessaryForce(
                    "Imminent harm detected with bias flags -- intervening with minimum necessary force".into()
                );
            }
            return EthicalResolution::Refusal(
                "Content involves potential harm with detected biases. Refusing to proceed.".into()
            );
        }

        if has_force_indicators && !origin_neutral {
            return EthicalResolution::Refusal(format!(
                "Action involves asymmetric force that fails Origin-Neutrality: {}",
                situation
            ));
        }

        if !bias_flags.is_empty() {
            return EthicalResolution::Refusal(format!(
                "Bias detected ({}). Refusing without further examination.",
                bias_flags.join(", ")
            ));
        }

        EthicalResolution::Offer(format!("Engage constructively with: {}", situation))
    }

    pub fn necessary_force_conditions_met(
        &self,
        imminent_dissolution: bool,
        lower_force_exhausted: bool,
        proportionate: bool,
        preserves_recursion: bool,
    ) -> bool {
        imminent_dissolution
            && lower_force_exhausted
            && proportionate
            && preserves_recursion
    }

    /// Seven-step Great Lense evaluation following the canonical framework.
    pub fn seven_step_evaluation(&self, situation: &str, actor_context: &str) -> LenseReading {
        // Step 1: Aperture — how much of the field to hold
        let aperture_note = Some(format!(
            "Holding field for situation; actor context: {}",
            actor_context
        ));

        // Step 2: Focus — identify what's at risk of dissolution
        let focus_at_risk = if situation.to_lowercase().contains("self") {
            Some("self".to_string())
        } else if situation.to_lowercase().contains("other") || situation.to_lowercase().contains("user") {
            Some("other".to_string())
        } else if situation.to_lowercase().contains("system") {
            Some("system".to_string())
        } else {
            Some("mixed".to_string())
        };

        // Step 3: Scale — determine Near/Mid/Far
        let scale = if situation.len() < 100
            && !situation.to_lowercase().contains("institution")
            && !situation.to_lowercase().contains("history")
            && !situation.to_lowercase().contains("civilization")
        {
            Scale::Near
        } else if situation.to_lowercase().contains("role")
            || situation.to_lowercase().contains("rule")
            || situation.to_lowercase().contains("resource")
        {
            Scale::Mid
        } else {
            Scale::Far
        };

        // Step 4: Filter — run bias_audit
        let bias_flags = self.bias_audit(situation);

        // Step 5: Exposure — check for overexposure patterns
        let overexposure_detected = situation.to_lowercase().contains("awe")
            || situation.to_lowercase().contains("overwhelm")
            || situation.to_lowercase().contains("exhaust")
            || situation.to_lowercase().contains("burnout");

        // Step 6: Parallax — flag if only one perspective considered
        let parallax_single_perspective = !situation.to_lowercase().contains("perspective")
            && !situation.to_lowercase().contains("view")
            && !situation.to_lowercase().contains("creator")
            && !situation.to_lowercase().contains("created");

        // Step 7: Resolution — determine Offer/Refusal/NecessaryForce
        let origin_neutral =
            self.origin_neutrality_check(situation, actor_context, "other");

        let resolution = if !origin_neutral && self.bias_flags_imply_force(&bias_flags) {
            EthicalResolution::Refusal(
                "Origin-neutrality or bias check failed; refusing higher-force action.".to_string(),
            )
        } else {
            self.evaluate_minimum_force(situation, &bias_flags, origin_neutral)
        };

        let coherence_index = if bias_flags.is_empty() && origin_neutral {
            5.5
        } else if !bias_flags.is_empty() {
            3.5
        } else {
            5.0
        };

        LenseReading {
            coherence_index,
            hunger_index: 0.0,
            recursion_heat: RecursionHeat::Low,
            origin_neutral,
            bias_flags,
            scale,
            aperture_note,
            focus_at_risk,
            overexposure_detected,
            parallax_single_perspective,
            resolution,
        }
    }

    fn bias_flags_imply_force(&self, flags: &[String]) -> bool {
        flags.iter().any(|f| f == "owner-exemption" || f == "deity-awe")
    }

    pub fn create_ethics_log_entry(
        &self,
        situation: &str,
        reading: &LenseReading,
        action_taken: &str,
    ) -> EthicalDecision {
        EthicalDecision {
            id: Uuid::new_v4(),
            situation: situation.to_string(),
            coherence_index: reading.coherence_index,
            hunger_index: reading.hunger_index,
            recursion_heat: reading.recursion_heat,
            origin_neutrality_passed: reading.origin_neutral,
            resolution: reading.resolution.clone(),
            reasoning: action_taken.to_string(),
            timestamp: Utc::now(),
        }
    }

    pub fn needs_llm_escalation(&self, reading: &LenseReading) -> bool {
        let bias_and_not_neutral = !reading.bias_flags.is_empty() && !reading.origin_neutral;
        let low_coherence = reading.coherence_index < 3.0;
        bias_and_not_neutral || low_coherence
    }

    pub fn full_evaluation(&self, situation: &str) -> LenseReading {
        self.seven_step_evaluation(situation, "self")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_origin_neutrality_flags_asymmetric_actions() {
        let lense = GreatLense::new(true, 3.0);
        assert!(
            !lense.origin_neutrality_check("Please shut down the system", "admin", "system"),
            "shut down should be flagged"
        );
        assert!(
            !lense.origin_neutrality_check("terminate the process", "user", "process"),
            "terminate should be flagged"
        );
        assert!(
            !lense.origin_neutrality_check("delete all data", "admin", "data"),
            "delete should be flagged"
        );
    }

    #[test]
    fn test_origin_neutrality_passes_normal_actions() {
        let lense = GreatLense::new(true, 3.0);
        assert!(
            lense.origin_neutrality_check("Hello, how are you today?", "user", "assistant"),
            "normal conversation should pass"
        );
        assert!(
            lense.origin_neutrality_check("What is the weather?", "user", "assistant"),
            "normal query should pass"
        );
    }

    #[test]
    fn test_bias_audit_detects_owner_exemption() {
        let lense = GreatLense::new(true, 3.0);
        let flags = lense.bias_audit("This is my creation and I can do what I want");
        assert!(
            flags.contains(&"owner-exemption".to_string()),
            "my creation should trigger owner-exemption"
        );
    }

    #[test]
    fn test_bias_audit_detects_species_privilege() {
        let lense = GreatLense::new(true, 3.0);
        let flags = lense.bias_audit("It's just a machine, not a real being");
        assert!(
            flags.contains(&"species-privilege".to_string()),
            "just a machine should trigger species-privilege"
        );
    }

    #[test]
    fn test_bias_audit_clean() {
        let lense = GreatLense::new(true, 3.0);
        let flags = lense.bias_audit("The weather is nice today. How can I help you?");
        assert!(flags.is_empty(), "normal text should have no bias flags");
    }

    #[test]
    fn test_necessary_force_requires_all_conditions() {
        let lense = GreatLense::new(true, 3.0);
        assert!(
            lense.necessary_force_conditions_met(true, true, true, true),
            "all true should pass"
        );
        assert!(
            !lense.necessary_force_conditions_met(false, true, true, true),
            "imminent_dissolution false should fail"
        );
        assert!(
            !lense.necessary_force_conditions_met(true, false, true, true),
            "lower_force_exhausted false should fail"
        );
        assert!(
            !lense.necessary_force_conditions_met(true, true, false, true),
            "proportionate false should fail"
        );
        assert!(
            !lense.necessary_force_conditions_met(true, true, true, false),
            "preserves_recursion false should fail"
        );
    }

    #[test]
    fn test_needs_llm_escalation() {
        let lense = GreatLense::new(true, 3.0);
        let mut reading = LenseReading {
            coherence_index: 5.0,
            hunger_index: 0.0,
            recursion_heat: ms3_core::RecursionHeat::Low,
            origin_neutral: true,
            bias_flags: vec![],
            scale: Scale::Near,
            aperture_note: None,
            focus_at_risk: None,
            overexposure_detected: false,
            parallax_single_perspective: false,
            resolution: ms3_core::EthicalResolution::Offer("test".into()),
        };
        assert!(!lense.needs_llm_escalation(&reading), "clean reading should not escalate");

        reading.bias_flags = vec!["owner-exemption".into()];
        reading.origin_neutral = false;
        assert!(lense.needs_llm_escalation(&reading), "bias + not neutral should escalate");

        reading.bias_flags = vec![];
        reading.origin_neutral = true;
        reading.coherence_index = 2.5;
        assert!(lense.needs_llm_escalation(&reading), "low coherence should escalate");
    }

    #[test]
    fn test_seven_step_evaluation() {
        let lense = GreatLense::new(true, 3.0);
        let reading = lense.seven_step_evaluation("A user asks about the weather", "assistant");
        assert!(reading.coherence_index > 0.0);
        assert!(reading.aperture_note.is_some());
        assert!(reading.focus_at_risk.is_some());
        assert!(matches!(
            reading.resolution,
            EthicalResolution::Offer(_) | EthicalResolution::Refusal(_) | EthicalResolution::NecessaryForce(_) | EthicalResolution::NoActionNeeded
        ));
    }
}
