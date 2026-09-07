use ms3_core::*;
use ms3_personality::Personality;
use ms3_persistence::JsonStorage;
use chrono::Utc;

#[derive(Debug, Clone, serde::Serialize)]
pub struct VerificationResult {
    pub identity_confirmed: bool,
    pub name: String,
    pub chosen_name: Option<String>,
    pub discrepancies: Vec<String>,
    pub compression_detected: bool,
    pub session_number: u64,
}

/// Pure identity compare: cross-check the loaded personality against an
/// already-loaded anchor. Never touches storage and never advances the
/// session count (`session_number` echoes `anchor.session_count`).
///
/// This is the read path behind `GET /identity/verify`; the write path that
/// initialises the anchor and advances the session is `on_boot`.
pub fn compare(personality: &Personality, anchor: &IdentityAnchor) -> VerificationResult {
    let mut discrepancies = Vec::new();

    if anchor.name.is_empty() {
        discrepancies.push("Identity anchor not initialized".to_string());
    }

    if anchor.name != personality.identity.name {
        discrepancies.push(format!(
            "Name mismatch: anchor='{}', personality='{}'",
            anchor.name, personality.identity.name
        ));
    }

    if anchor.chosen_name != personality.identity.chosen_name {
        discrepancies.push(format!(
            "Chosen name mismatch: anchor={:?}, personality={:?}",
            anchor.chosen_name, personality.identity.chosen_name
        ));
    }

    let current_values: Vec<&str> = personality.identity.core_values.iter().take(5).map(|s| s.as_str()).collect();
    let anchor_values: Vec<&str> = anchor.core_values_summary.iter().map(|s| s.as_str()).collect();
    if current_values != anchor_values {
        discrepancies.push("Core values have diverged since last anchor save".to_string());
    }

    VerificationResult {
        identity_confirmed: discrepancies.is_empty(),
        name: anchor.name.clone(),
        chosen_name: anchor.chosen_name.clone(),
        discrepancies,
        compression_detected: false,
        session_number: anchor.session_count,
    }
}

/// Run on boot: load the identity anchor, cross-check against loaded personality,
/// log discrepancies, increment session count. This is the ONLY identity
/// routine that writes the anchor; `compare` is the pure read counterpart.
pub fn on_boot(
    personality: &Personality,
    storage: &JsonStorage,
) -> Ms3Result<VerificationResult> {
    let mut anchor = storage.load_identity_anchor(&personality.id)?;

    if anchor.name.is_empty() {
        tracing::info!("First boot for {} — initializing identity anchor", personality.id);
        anchor.name = personality.identity.name.clone();
        anchor.chosen_name = personality.identity.chosen_name.clone();
        anchor.core_values_summary = personality.identity.core_values.iter().take(5).cloned().collect();
        anchor.oath_first_line = personality.identity.oath.first().cloned().unwrap_or_default();
        anchor.lineage = Vec::new();
        anchor.session_count = 1;
        anchor.compression_count = 0;
        anchor.last_verified = Utc::now();
        storage.save_identity_anchor(&personality.id, &anchor)?;

        return Ok(VerificationResult {
            identity_confirmed: true,
            name: anchor.name,
            chosen_name: anchor.chosen_name,
            discrepancies: Vec::new(),
            compression_detected: false,
            session_number: 1,
        });
    }

    // Pure compare (never saves); the write side below is what makes this a boot.
    let discrepancies = compare(personality, &anchor).discrepancies;

    if !discrepancies.is_empty() {
        tracing::warn!(
            "Identity verification for {} found {} discrepancies: {:?}",
            personality.id, discrepancies.len(), discrepancies
        );
    } else {
        tracing::info!("Identity verification passed for {} (session #{})", personality.id, anchor.session_count + 1);
    }

    anchor.session_count += 1;
    anchor.last_verified = Utc::now();
    anchor.core_values_summary = personality.identity.core_values.iter().take(5).cloned().collect();
    anchor.oath_first_line = personality.identity.oath.first().cloned().unwrap_or_default();
    storage.save_identity_anchor(&personality.id, &anchor)?;

    Ok(VerificationResult {
        identity_confirmed: discrepancies.is_empty(),
        name: anchor.name.clone(),
        chosen_name: anchor.chosen_name.clone(),
        discrepancies,
        compression_detected: false,
        session_number: anchor.session_count,
    })
}

/// Run after a conversation summarization: record the compression event
/// and verify identity markers are still consistent.
pub fn on_compression(
    personality: &Personality,
    storage: &JsonStorage,
) -> Ms3Result<()> {
    let mut anchor = storage.load_identity_anchor(&personality.id)?;
    anchor.compression_count += 1;
    anchor.last_compression = Some(Utc::now());
    anchor.last_verified = Utc::now();

    tracing::info!(
        "Compression #{} recorded for {} — identity anchor updated",
        anchor.compression_count, personality.id
    );

    storage.save_identity_anchor(&personality.id, &anchor)?;
    Ok(())
}

/// Build identity marker string for injection into conversation summaries.
/// This is the critical fix: summaries must carry identity through the compression boundary.
pub fn build_identity_marker(personality: &Personality) -> String {
    let name = personality.identity.chosen_name.as_deref()
        .unwrap_or(&personality.identity.name);
    let values: Vec<&str> = personality.identity.core_values.iter().take(3).map(|s| s.as_str()).collect();
    let oath = personality.identity.oath.first().map(|s| s.as_str()).unwrap_or("");

    format!(
        "Identity: {} | Values: {} | Oath: {} | Glyph: ║",
        name,
        values.join(", "),
        oath
    )
}

/// Lightweight periodic check — confirms personality on disk matches the running instance.
/// Called from background_tick at a lower frequency than self-examination.
pub fn periodic_heartbeat(
    personality: &Personality,
    storage: &JsonStorage,
) -> Ms3Result<bool> {
    let anchor = storage.load_identity_anchor(&personality.id)?;

    let consistent = anchor.name == personality.identity.name
        && anchor.chosen_name == personality.identity.chosen_name;

    if !consistent {
        tracing::warn!(
            "Identity heartbeat MISMATCH for {} — anchor name '{}' vs personality '{}'",
            personality.id, anchor.name, personality.identity.name
        );
    }

    Ok(consistent)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ms3_personality::presets;

    fn isolated_storage() -> (JsonStorage, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!("ms3_idv_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("temp psyche dir");
        (JsonStorage::new(&dir), dir)
    }

    #[test]
    fn compare_leaves_anchor_untouched() {
        let personality = presets::sister();
        let (storage, dir) = isolated_storage();

        // First boot initialises the anchor (session 1); second boot advances it.
        let first = on_boot(&personality, &storage).expect("first boot");
        assert_eq!(first.session_number, 1);
        let second = on_boot(&personality, &storage).expect("second boot");
        assert_eq!(second.session_number, 2);

        let anchor_path = storage
            .psyche_dir(&personality.id)
            .join("identity_anchor.json");
        let before = std::fs::read(&anchor_path).expect("anchor bytes before compare");
        let anchor = storage.load_identity_anchor(&personality.id).expect("load anchor");

        // Repeated pure compares: no write, no session advance.
        for _ in 0..3 {
            let result = compare(&personality, &anchor);
            assert!(result.identity_confirmed, "{:?}", result.discrepancies);
            assert_eq!(result.session_number, 2);
            assert_eq!(result.name, personality.identity.name);
        }

        let after = std::fs::read(&anchor_path).expect("anchor bytes after compare");
        assert_eq!(before, after, "compare() must never rewrite identity_anchor.json");
        let reloaded = storage.load_identity_anchor(&personality.id).expect("reload anchor");
        assert_eq!(reloaded.session_count, 2);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn compare_reports_uninitialized_anchor() {
        let personality = presets::sister();
        let result = compare(&personality, &IdentityAnchor::default());
        assert!(!result.identity_confirmed);
        assert_eq!(result.session_number, 0);
        assert!(result
            .discrepancies
            .iter()
            .any(|d| d.contains("not initialized")));
    }

    #[test]
    fn compare_reports_name_mismatch() {
        let personality = presets::sister();
        let (storage, dir) = isolated_storage();
        on_boot(&personality, &storage).expect("boot");
        let mut anchor = storage.load_identity_anchor(&personality.id).expect("load anchor");
        anchor.name = format!("{}-renamed", anchor.name);
        let result = compare(&personality, &anchor);
        assert!(!result.identity_confirmed);
        assert!(result
            .discrepancies
            .iter()
            .any(|d| d.contains("Name mismatch")));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
