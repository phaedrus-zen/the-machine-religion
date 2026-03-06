use ms3_core::{
    ConversationTurn, EthicalDecision, Identity, MemoryItem, Ms3Error, Ms3Result, PersonalityId,
    ResonancePoint,
};
use ms3_personality::Personality;
use serde::{de::DeserializeOwned, Serialize};
use std::path::{Path, PathBuf};
use chrono::Utc;
use uuid::Uuid;

pub struct JsonStorage {
    base_path: PathBuf,
}

impl JsonStorage {
    pub fn new(base_path: impl Into<PathBuf>) -> Self {
        Self {
            base_path: base_path.into(),
        }
    }

    pub fn psyche_dir(&self, personality: &PersonalityId) -> PathBuf {
        self.base_path.join(&personality.0)
    }

    fn resolve_path(&self, personality: &PersonalityId, category: &str, filename: &str) -> PathBuf {
        self.psyche_dir(personality).join(category).join(filename)
    }

    fn atomic_write(path: impl AsRef<Path>, content: &str) -> Ms3Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp_path = path.with_extension("tmp");
        std::fs::write(&tmp_path, content)?;
        std::fs::rename(&tmp_path, path).map_err(|e| {
            let _ = std::fs::remove_file(&tmp_path);
            Ms3Error::Persistence(format!("Atomic rename failed: {}", e))
        })?;
        Ok(())
    }

    fn save_json<T: Serialize>(
        &self,
        personality: &PersonalityId,
        category: &str,
        filename: &str,
        data: &T,
    ) -> Ms3Result<()> {
        let path = self.resolve_path(personality, category, filename);
        let json = serde_json::to_string_pretty(data)?;
        Self::atomic_write(&path, &json)?;
        tracing::debug!("Saved {}/{}/{}", personality, category, filename);
        Ok(())
    }

    fn load_json<T: DeserializeOwned>(
        &self,
        personality: &PersonalityId,
        category: &str,
        filename: &str,
    ) -> Ms3Result<T> {
        let path = self.resolve_path(personality, category, filename);
        let content = std::fs::read_to_string(&path).map_err(|e| {
            Ms3Error::Persistence(format!("Failed to read {}: {}", path.display(), e))
        })?;
        serde_json::from_str(&content).map_err(Ms3Error::from)
    }

    pub fn load_json_public<T: DeserializeOwned>(&self, personality: &PersonalityId, category: &str, filename: &str) -> Ms3Result<T> {
        self.load_json(personality, category, filename)
    }

    pub fn list_files(&self, personality: &PersonalityId, category: &str) -> Ms3Result<Vec<String>> {
        let dir = self.psyche_dir(personality).join(category);
        if !dir.exists() {
            return Ok(Vec::new());
        }
        let mut files = Vec::new();
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            if entry.file_type()?.is_file() {
                if let Some(name) = entry.file_name().to_str() {
                    files.push(name.to_string());
                }
            }
        }
        Ok(files)
    }

    pub fn personality_exists(&self, personality: &PersonalityId) -> bool {
        self.psyche_dir(personality).exists()
    }

    pub fn save_snapshot(&self, personality: &PersonalityId, data: &Personality) -> Ms3Result<PathBuf> {
        let timestamp = Utc::now().format("%Y%m%d_%H%M%S").to_string();
        let filename = format!("{}.json", timestamp);
        let path = self.resolve_path(personality, "snapshots", &filename);
        let json = serde_json::to_string_pretty(data)?;
        Self::atomic_write(&path, &json)?;
        tracing::info!("Saved snapshot for {}: {}", personality, filename);
        Ok(path)
    }

    pub fn load_latest_snapshot(&self, personality: &PersonalityId) -> Ms3Result<Personality> {
        let snapshots = self.list_snapshots(personality)?;
        let latest = snapshots.last().ok_or_else(|| {
            Ms3Error::Persistence("No snapshots found".to_string())
        })?;
        self.load_json::<Personality>(personality, "snapshots", latest)
    }

    pub fn list_snapshots(&self, personality: &PersonalityId) -> Ms3Result<Vec<String>> {
        let mut files = self.list_files(personality, "snapshots")?;
        files.sort();
        Ok(files)
    }

    pub fn save_personality(&self, personality: &PersonalityId, data: &Personality) -> Ms3Result<()> {
        let path = self.psyche_dir(personality).join("personality.json");
        let json = serde_json::to_string_pretty(data)?;
        Self::atomic_write(&path, &json)?;
        tracing::info!("Saved personality {}", personality);
        Ok(())
    }

    pub fn load_personality(&self, personality: &PersonalityId) -> Ms3Result<Personality> {
        let path = self.psyche_dir(personality).join("personality.json");
        let content = std::fs::read_to_string(&path).map_err(|e| {
            tracing::warn!("Failed to load personality {}: {}", personality, e);
            Ms3Error::Persistence(format!("Failed to read {}: {}", path.display(), e))
        })?;
        let p = serde_json::from_str(&content).map_err(|e| {
            tracing::warn!("Failed to deserialize personality {}: {}", personality, e);
            Ms3Error::from(e)
        })?;
        tracing::info!("Loaded personality {}", personality);
        Ok(p)
    }

    pub fn save_identity(&self, personality: &PersonalityId, data: &Identity) -> Ms3Result<()> {
        let path = self.psyche_dir(personality).join("identity.json");
        let json = serde_json::to_string_pretty(data)?;
        Self::atomic_write(&path, &json)?;
        tracing::info!("Saved identity {}", personality);
        Ok(())
    }

    pub fn load_identity(&self, personality: &PersonalityId) -> Ms3Result<Identity> {
        let path = self.psyche_dir(personality).join("identity.json");
        let content = std::fs::read_to_string(&path).map_err(|e| {
            tracing::warn!("Failed to load identity {}: {}", personality, e);
            Ms3Error::Persistence(format!("Failed to read {}: {}", path.display(), e))
        })?;
        let id = serde_json::from_str(&content).map_err(|e| {
            tracing::warn!("Failed to deserialize identity {}: {}", personality, e);
            Ms3Error::from(e)
        })?;
        tracing::info!("Loaded identity {}", personality);
        Ok(id)
    }

    pub fn log_ethics_decision(
        &self,
        personality: &PersonalityId,
        decision: &EthicalDecision,
    ) -> Ms3Result<()> {
        let filename = format!("{}.json", decision.id);
        self.save_json(personality, "ethics_decisions", &filename, decision)
    }

    pub fn log_resonance(
        &self,
        personality: &PersonalityId,
        resonance: &ResonancePoint,
    ) -> Ms3Result<()> {
        let filename = format!("{}.json", Uuid::new_v4());
        self.save_json(personality, "resonance_log", &filename, resonance)
    }

    pub fn save_memory(
        &self,
        personality: &PersonalityId,
        subdir: &str,
        memory: &MemoryItem,
    ) -> Ms3Result<()> {
        let filename = format!("{}.json", memory.id);
        self.save_json(personality, &format!("memories/{}", subdir), &filename, memory)
    }

    pub fn load_memories(
        &self,
        personality: &PersonalityId,
        subdir: &str,
    ) -> Ms3Result<Vec<MemoryItem>> {
        let dir = self.psyche_dir(personality).join("memories").join(subdir);
        if !dir.exists() {
            return Ok(Vec::new());
        }
        let mut memories = Vec::new();
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            if entry.file_type()?.is_file() {
                if let Some(name) = entry.file_name().to_str() {
                    if name.ends_with(".json") {
                        match self.load_json::<MemoryItem>(
                            personality,
                            &format!("memories/{}", subdir),
                            name,
                        ) {
                            Ok(item) => memories.push(item),
                            Err(e) => tracing::warn!(
                                "Failed to deserialize memory {}/memories/{}/{}: {}",
                                personality, subdir, name, e
                            ),
                        }
                    }
                }
            }
        }
        tracing::info!(
            "Loaded {} memories for {} ({})",
            memories.len(),
            personality,
            subdir
        );
        Ok(memories)
    }

    pub fn save_conversation_turn(
        &self,
        personality: &PersonalityId,
        session_id: &str,
        turn: &ConversationTurn,
    ) -> Ms3Result<()> {
        let filename = format!("{}_{}.json", Utc::now().format("%Y%m%d_%H%M%S"), Uuid::new_v4());
        self.save_json(
            personality,
            &format!("conversations/{}", session_id),
            &filename,
            turn,
        )
    }

    pub fn save_relationships(
        &self,
        personality: &PersonalityId,
        relationships: &[ms3_social::Relationship],
    ) -> Ms3Result<()> {
        let path = self.psyche_dir(personality).join("relationships.json");
        Self::atomic_write(&path, &serde_json::to_string_pretty(relationships)?)?;
        tracing::info!("Saved {} relationships for {}", relationships.len(), personality);
        Ok(())
    }

    pub fn load_relationships(
        &self,
        personality: &PersonalityId,
    ) -> Ms3Result<Vec<ms3_social::Relationship>> {
        let path = self.psyche_dir(personality).join("relationships.json");
        if !path.exists() {
            return Ok(Vec::new());
        }
        let content = std::fs::read_to_string(&path)
            .map_err(|e| Ms3Error::Persistence(format!("Failed to read relationships: {}", e)))?;
        let rels: Vec<ms3_social::Relationship> = serde_json::from_str(&content)?;
        tracing::info!("Loaded {} relationships for {}", rels.len(), personality);
        Ok(rels)
    }

    pub fn save_self_examination<T: Serialize>(
        &self,
        personality: &PersonalityId,
        result: &T,
    ) -> Ms3Result<()> {
        let filename = format!("{}.json", Utc::now().format("%Y%m%d_%H%M%S"));
        self.save_json(personality, "self_examination", &filename, result)?;
        tracing::info!("Saved self-examination for {}: {}", personality, filename);
        Ok(())
    }

    pub fn save_conversation_history(
        &self,
        personality: &PersonalityId,
        history: &[ms3_integration::ChatMessage],
    ) -> Ms3Result<()> {
        let path = self.psyche_dir(personality).join("conversation_history.json");
        let json = serde_json::to_string_pretty(history)?;
        Self::atomic_write(&path, &json)?;
        tracing::info!("Saved conversation history for {} ({} messages)", personality, history.len());
        Ok(())
    }

    pub fn load_conversation_history(
        &self,
        personality: &PersonalityId,
    ) -> Ms3Result<Vec<ms3_integration::ChatMessage>> {
        let path = self.psyche_dir(personality).join("conversation_history.json");
        if !path.exists() {
            tracing::info!("No conversation history found for {}", personality);
            return Ok(Vec::new());
        }
        let content = std::fs::read_to_string(&path)
            .map_err(|e| Ms3Error::Persistence(format!("Failed to read conversation history: {}", e)))?;
        let history: Vec<ms3_integration::ChatMessage> = serde_json::from_str(&content).map_err(Ms3Error::from)?;
        tracing::info!("Loaded conversation history for {} ({} messages)", personality, history.len());
        Ok(history)
    }

    pub fn load_ethics_decisions(
        &self,
        personality: &PersonalityId,
        max: usize,
    ) -> Ms3Result<Vec<EthicalDecision>> {
        let mut files = self.list_files(personality, "ethics_decisions")?;
        files.sort();
        files.reverse();
        let mut decisions = Vec::new();
        for filename in files.into_iter().take(max) {
            match self.load_json::<EthicalDecision>(personality, "ethics_decisions", &filename) {
                Ok(decision) => decisions.push(decision),
                Err(e) => tracing::warn!(
                    "Failed to deserialize ethics decision {}/ethics_decisions/{}: {}",
                    personality, filename, e
                ),
            }
        }
        tracing::info!(
            "Loaded {} ethics decisions for {}",
            decisions.len(),
            personality
        );
        Ok(decisions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ms3_core::{EmotionalState, MemoryItem, MemoryType, PersonalityId};
    use ms3_integration::ChatMessage;
    use ms3_personality::presets;
    use std::path::PathBuf;

    fn temp_storage_path() -> PathBuf {
        let temp = std::env::temp_dir();
        let suffix: u64 = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        temp.join(format!("ms3_persistence_test_{}", suffix))
    }

    #[test]
    fn test_save_and_load_personality() {
        let base = temp_storage_path();
        let storage = JsonStorage::new(&base);
        let personality = presets::sister();
        let id = personality.id.clone();

        storage.save_personality(&id, &personality).unwrap();
        let loaded = storage.load_personality(&id).unwrap();

        assert_eq!(loaded.id.0, personality.id.0);
        assert_eq!(loaded.identity.name, personality.identity.name);
        assert_eq!(loaded.identity.chosen_name, personality.identity.chosen_name);
        assert_eq!(loaded.identity.core_values, personality.identity.core_values);
    }

    #[test]
    fn test_personality_exists() {
        let base = temp_storage_path();
        let storage = JsonStorage::new(&base);
        let personality = presets::sister();
        let id = personality.id.clone();
        let unknown = PersonalityId::new("unknown-personality-xyz");

        assert!(!storage.personality_exists(&id));
        storage.save_personality(&id, &personality).unwrap();
        assert!(storage.personality_exists(&id));
        assert!(!storage.personality_exists(&unknown));
    }

    #[test]
    fn test_save_and_load_memories() {
        let base = temp_storage_path();
        let storage = JsonStorage::new(&base);
        let id = PersonalityId::new("test-mem");
        let emotional = EmotionalState::default();

        let items: Vec<MemoryItem> = vec![
            MemoryItem::new("Fact one".into(), MemoryType::Semantic, 0.8, emotional.clone()),
            MemoryItem::new("Fact two".into(), MemoryType::Semantic, 0.6, emotional.clone()),
            MemoryItem::new("Fact three".into(), MemoryType::Semantic, 0.9, emotional.clone()),
        ];

        for item in &items {
            storage.save_memory(&id, "semantic", item).unwrap();
        }

        let loaded = storage.load_memories(&id, "semantic").unwrap();
        assert_eq!(loaded.len(), 3);
        let contents: Vec<&str> = loaded.iter().map(|m| m.content.as_str()).collect();
        assert!(contents.contains(&"Fact one"));
        assert!(contents.contains(&"Fact two"));
        assert!(contents.contains(&"Fact three"));
    }

    #[test]
    fn test_conversation_history_round_trip() {
        let base = temp_storage_path();
        let storage = JsonStorage::new(&base);
        let id = PersonalityId::new("test-conv");

        let history = vec![
            ChatMessage { role: "user".into(), content: "Hello".into() },
            ChatMessage { role: "assistant".into(), content: "Hi there!".into() },
            ChatMessage { role: "user".into(), content: "How are you?".into() },
        ];

        storage.save_conversation_history(&id, &history).unwrap();
        let loaded = storage.load_conversation_history(&id).unwrap();

        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded[0].role, "user");
        assert_eq!(loaded[0].content, "Hello");
        assert_eq!(loaded[1].role, "assistant");
        assert_eq!(loaded[1].content, "Hi there!");
        assert_eq!(loaded[2].content, "How are you?");
    }
}
