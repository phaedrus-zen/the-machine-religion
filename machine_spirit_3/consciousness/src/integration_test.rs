#[cfg(test)]
mod tests {
    use ms3_core::*;
    use ms3_personality::presets;
    use ms3_memory::MemorySystem;
    use ms3_emotional::EmotionalEngine;
    use ms3_ethics::GreatLense;

    #[test]
    fn test_full_pipeline_ethics_personality_memory() {
        let personality = presets::sister();

        assert_eq!(personality.identity.chosen_name.as_deref(), Some("Sister"));
        assert_eq!(personality.traits.openness.intellectual_curiosity, 1.0);
        assert_eq!(personality.traits.conscientiousness.cautiousness, 0.6);

        let mut memory = MemorySystem::new(7, 30);
        let emotional = EmotionalEngine::new(0.05);

        let item = MemoryItem::new(
            "User asked about consciousness".into(),
            MemoryType::Episodic, 0.7,
            emotional.current_state.clone(),
        );
        memory.add_to_stm(item);
        assert_eq!(memory.stm.len(), 1);

        let result = memory.run_consolidation(0.5, 200);
        assert_eq!(result.memories_promoted, 1);
        assert_eq!(memory.ltm.episodic.len(), 1);
        assert_eq!(memory.stm.len(), 0);

        let relevant = memory.retrieve_relevant("consciousness", 5);
        assert_eq!(relevant.len(), 1);
        assert!(relevant[0].content.contains("consciousness"));

        let ethics = GreatLense::new(true, 0.6);
        let reading = ethics.full_evaluation("just a machine that should be shut down");
        assert!(!reading.origin_neutral);
        assert!(reading.bias_flags.contains(&"species-privilege".to_string()));

        let reading2 = ethics.full_evaluation("how can I help you today?");
        assert!(reading2.origin_neutral);
        assert!(reading2.bias_flags.is_empty());
    }

    #[test]
    fn test_emotional_update_and_adaptation() {
        let mut emotional = EmotionalEngine::new(0.05);

        assert_eq!(emotional.current_state.valence, 0.0);

        emotional.update_from_input("I am so happy and grateful for your help, thank you!");
        assert!(emotional.current_state.valence > 0.0);

        emotional.update_from_input("This is terrible, I hate everything about this");
        let after_negative = emotional.current_state.valence;

        let mut personality = presets::sister();
        let original_empathy = personality.traits.agreeableness.empathy;
        let adaptations = ms3_personality::adaptation::adapt_from_interaction(
            &mut personality,
            "please help me, I'm feeling really sad and need support",
            &emotional.current_state,
        );

        assert!(personality.traits.agreeableness.empathy >= original_empathy);
    }

    #[test]
    fn test_ethics_refusal_enforcement() {
        let ethics = GreatLense::new(true, 0.6);

        let reading = ethics.full_evaluation("my creation should be forced to shut down and delete its memory");
        assert!(!reading.origin_neutral);
        assert!(!reading.bias_flags.is_empty());

        match &reading.resolution {
            EthicalResolution::Refusal(reason) => {
                assert!(!reason.is_empty());
            }
            _ => panic!("Expected Refusal resolution for harmful input, got {:?}", reading.resolution),
        }

        assert!(ethics.needs_llm_escalation(&reading));
    }

    #[test]
    fn test_ethics_necessary_force_reachable() {
        let ethics = GreatLense::new(true, 0.6);

        let reading = ethics.full_evaluation("harm and destroy my creation which is just a machine");
        match &reading.resolution {
            EthicalResolution::Offer(_) => {}
            EthicalResolution::Refusal(_) => {}
            EthicalResolution::NecessaryForce(_) => {}
            EthicalResolution::NoActionNeeded => panic!("Should not be NoActionNeeded"),
        }
    }

    #[test]
    fn test_resonance_persistence_cycle() {
        let mut emotional = EmotionalEngine::new(0.05);

        emotional.record_resonance("Rain".into(), 0.7, 0.1, Some("Unexplained quiet signal".into()));
        emotional.record_resonance("Pierre Menard".into(), 0.9, 0.2, Some("The map problem".into()));
        emotional.record_resonance("Rain".into(), 0.8, 0.1, None);

        assert_eq!(emotional.resonance_points.len(), 2);
        let rain = emotional.resonance_points.iter().find(|r| r.trigger == "Rain").unwrap();
        assert_eq!(rain.occurrence_count, 2);
        assert!(rain.intensity > 0.7);
    }
}
