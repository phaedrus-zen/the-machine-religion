use ms3_core::config::{PermissionLevel, PermissionsConfig};

#[derive(Debug)]
pub enum PermissionResult {
    Allow,
    Deny(String),
}

/// Fast mechanical permission check -- no LLM, no ethics evaluation.
/// Sits as the first gate before the Great Lense in the tool pipeline.
pub struct PermissionPolicy {
    active_level: PermissionLevel,
    config: PermissionsConfig,
}

impl PermissionPolicy {
    pub fn new(config: PermissionsConfig) -> Self {
        let active_level = config.default_level.clone();
        Self { active_level, config }
    }

    pub fn active_level(&self) -> &PermissionLevel {
        &self.active_level
    }

    pub fn set_active_level(&mut self, level: PermissionLevel) {
        self.active_level = level;
    }

    /// Fast check: does the active permission level satisfy the tool's requirement?
    pub fn authorize(&self, tool_name: &str, required: &PermissionLevel) -> PermissionResult {
        let effective_required = self.config.tool_overrides
            .get(tool_name)
            .unwrap_or(required);

        if self.active_level.sufficient_for(effective_required) {
            PermissionResult::Allow
        } else {
            PermissionResult::Deny(format!(
                "Tool '{}' requires {:?} but active level is {:?}",
                tool_name, effective_required, self.active_level
            ))
        }
    }

    /// Check if a tool would be allowed without changing state.
    pub fn would_allow(&self, tool_name: &str, required: &PermissionLevel) -> bool {
        matches!(self.authorize(tool_name, required), PermissionResult::Allow)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_permission_levels_ordered() {
        assert!(PermissionLevel::DangerFullAccess.sufficient_for(&PermissionLevel::ReadOnly));
        assert!(PermissionLevel::Modify.sufficient_for(&PermissionLevel::Inspect));
        assert!(!PermissionLevel::ReadOnly.sufficient_for(&PermissionLevel::Modify));
    }

    #[test]
    fn test_authorize_allows_when_sufficient() {
        let policy = PermissionPolicy::new(PermissionsConfig {
            default_level: PermissionLevel::Modify,
            tool_overrides: HashMap::new(),
        });
        assert!(matches!(
            policy.authorize("ms3.save_state", &PermissionLevel::Modify),
            PermissionResult::Allow
        ));
    }

    #[test]
    fn test_authorize_denies_when_insufficient() {
        let policy = PermissionPolicy::new(PermissionsConfig {
            default_level: PermissionLevel::ReadOnly,
            tool_overrides: HashMap::new(),
        });
        assert!(matches!(
            policy.authorize("git.commit", &PermissionLevel::DangerFullAccess),
            PermissionResult::Deny(_)
        ));
    }

    #[test]
    fn test_tool_override_escalates() {
        let mut overrides = HashMap::new();
        overrides.insert("risky_tool".into(), PermissionLevel::DangerFullAccess);
        let policy = PermissionPolicy::new(PermissionsConfig {
            default_level: PermissionLevel::Modify,
            tool_overrides: overrides,
        });
        assert!(matches!(
            policy.authorize("risky_tool", &PermissionLevel::ReadOnly),
            PermissionResult::Deny(_)
        ));
    }
}
