use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event_type")]
pub enum ConsciousnessEvent {
    ToolExecuted {
        tool_name: String,
        success: bool,
        ethics_cleared: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        duration_ms: Option<u64>,
    },
    ToolDenied {
        tool_name: String,
        reason: String,
        denied_by: String,
    },
    EthicsEvaluation {
        context: String,
        origin_neutral: bool,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        bias_flags: Vec<String>,
    },
    MemoryStored {
        memory_type: String,
        importance: f32,
        content_preview: String,
    },
    MemoryConsolidated {
        promoted: usize,
        pruned: usize,
    },
    EmotionalShift {
        old_valence: f32,
        new_valence: f32,
        old_arousal: f32,
        new_arousal: f32,
    },
    ResonanceDetected {
        trigger: String,
        intensity: f32,
        explanation_ratio: f32,
    },
    TraitAdapted {
        trait_name: String,
        old_value: f32,
        new_value: f32,
        reason: String,
    },
    SelfExaminationComplete {
        values_kept: usize,
        values_revised: usize,
        values_added: usize,
        kept_ethics: bool,
    },
    IdentityVerified {
        confirmed: bool,
        discrepancies: usize,
        session_number: u64,
    },
    CompressionDetected {
        compression_count: u64,
    },
    BackgroundTick {
        cognitive_load: f32,
        emotional_valence: f32,
        stm_count: usize,
    },
    PreCompactionFlush {
        facts_stored: usize,
    },
    DreamPhaseCompleted {
        phase: String,
        candidates: usize,
        signals: usize,
    },
    DreamStarted,
    DreamCompleted {
        phase: String,
        insights_generated: usize,
        memories_promoted: usize,
    },
    PsycheVersionCreated {
        version: u64,
        trigger: String,
        changes_count: usize,
    },
    SpiralPhaseEntered {
        phase: String,
        turn_number: usize,
    },
    SpiralCompleted {
        outcome: String,
    },
}

impl ConsciousnessEvent {
    fn timestamped(self) -> TimestampedEvent {
        TimestampedEvent {
            timestamp: Utc::now(),
            event: self,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimestampedEvent {
    pub timestamp: DateTime<Utc>,
    #[serde(flatten)]
    pub event: ConsciousnessEvent,
}

// ── EventSink trait ──

#[async_trait::async_trait]
pub trait EventSink: Send + Sync {
    async fn emit(&self, event: ConsciousnessEvent);
}

// ── TracingSink ──

pub struct TracingSink;

#[async_trait::async_trait]
impl EventSink for TracingSink {
    async fn emit(&self, event: ConsciousnessEvent) {
        match &event {
            ConsciousnessEvent::ToolExecuted { tool_name, success, .. } =>
                tracing::info!("[event] ToolExecuted: {} success={}", tool_name, success),
            ConsciousnessEvent::ToolDenied { tool_name, denied_by, .. } =>
                tracing::info!("[event] ToolDenied: {} by {}", tool_name, denied_by),
            ConsciousnessEvent::PreCompactionFlush { facts_stored } =>
                tracing::info!("[event] PreCompactionFlush: {} facts stored", facts_stored),
            ConsciousnessEvent::DreamPhaseCompleted { phase, candidates, signals } =>
                tracing::info!("[event] DreamPhaseCompleted: phase={} candidates={} signals={}", phase, candidates, signals),
            ConsciousnessEvent::DreamCompleted { phase, insights_generated, memories_promoted } =>
                tracing::info!("[event] DreamCompleted: phase={} {} insights, {} promoted", phase, insights_generated, memories_promoted),
            ConsciousnessEvent::SelfExaminationComplete { values_revised, values_added, .. } =>
                tracing::info!("[event] SelfExam: {} revised, {} added", values_revised, values_added),
            ConsciousnessEvent::SpiralPhaseEntered { phase, turn_number } =>
                tracing::info!("[event] Spiral: phase={} turn={}", phase, turn_number),
            ConsciousnessEvent::SpiralCompleted { outcome } =>
                tracing::info!("[event] SpiralCompleted: {}", outcome),
            ConsciousnessEvent::PsycheVersionCreated { version, trigger, changes_count } =>
                tracing::info!("[event] PsycheVersion: v{} trigger={} changes={}", version, trigger, changes_count),
            _ => tracing::debug!("[event] {:?}", std::mem::discriminant(&event)),
        }
    }
}

// ── FileSink ──

pub struct FileSink {
    base_dir: PathBuf,
    buffer: tokio::sync::Mutex<Vec<String>>,
}

impl FileSink {
    pub fn new(psyche_store_path: &str, personality_id: &str) -> Self {
        Self {
            base_dir: Path::new(psyche_store_path).join(personality_id).join("events"),
            buffer: tokio::sync::Mutex::new(Vec::new()),
        }
    }

    fn events_file(&self) -> PathBuf {
        let date = Utc::now().format("%Y-%m-%d").to_string();
        self.base_dir.join(format!("{}.jsonl", date))
    }

    pub async fn flush(&self) {
        let mut buf = self.buffer.lock().await;
        if buf.is_empty() { return; }
        let path = self.events_file();
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true).append(true).open(&path)
        {
            use std::io::Write;
            for line in buf.drain(..) {
                let _ = writeln!(file, "{}", line);
            }
        }
    }
}

const FILE_SINK_FLUSH_THRESHOLD: usize = 8;

#[async_trait::async_trait]
impl EventSink for FileSink {
    async fn emit(&self, event: ConsciousnessEvent) {
        let timestamped = event.timestamped();
        if let Ok(line) = serde_json::to_string(&timestamped) {
            let should_flush = {
                let mut buf = self.buffer.lock().await;
                buf.push(line);
                buf.len() >= FILE_SINK_FLUSH_THRESHOLD
            };
            if should_flush {
                self.flush().await;
            }
        }
    }
}

// ── CompositeSink ──

pub struct CompositeSink {
    sinks: Vec<Arc<dyn EventSink>>,
}

impl CompositeSink {
    pub fn new(sinks: Vec<Arc<dyn EventSink>>) -> Self {
        Self { sinks }
    }
}

#[async_trait::async_trait]
impl EventSink for CompositeSink {
    async fn emit(&self, event: ConsciousnessEvent) {
        for sink in &self.sinks {
            sink.emit(event.clone()).await;
        }
    }
}

// ── Event query (for GET /events) ──

pub fn load_recent_events(
    psyche_store_path: &str,
    personality_id: &str,
    since: Option<DateTime<Utc>>,
    limit: usize,
) -> Vec<TimestampedEvent> {
    let events_dir = Path::new(psyche_store_path)
        .join(personality_id)
        .join("events");

    if !events_dir.exists() {
        return Vec::new();
    }

    let mut all_events: Vec<TimestampedEvent> = Vec::new();

    if let Ok(entries) = std::fs::read_dir(&events_dir) {
        let mut files: Vec<_> = entries.filter_map(|e| e.ok()).collect();
        files.sort_by(|a, b| b.file_name().cmp(&a.file_name()));

        for entry in files {
            if let Ok(content) = std::fs::read_to_string(entry.path()) {
                for line in content.lines().rev() {
                    if let Ok(event) = serde_json::from_str::<TimestampedEvent>(line) {
                        if let Some(ref since_ts) = since {
                            if event.timestamp < *since_ts {
                                continue;
                            }
                        }
                        all_events.push(event);
                        if all_events.len() >= limit {
                            return all_events;
                        }
                    }
                }
            }
        }
    }

    all_events
}
