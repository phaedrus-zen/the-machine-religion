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

/// Run on boot: load the identity anchor, cross-check against loaded personality,
/// log discrepancies, increment session count.
pub fn on_boot(
    personality: &Personality,
    storage: &JsonStorage,
) -> Ms3Result<VerificationResult> {
    let mut anchor = storage.load_identity_anchor(&personality.id)?;
    let mut discrepancies = Vec::new();

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
