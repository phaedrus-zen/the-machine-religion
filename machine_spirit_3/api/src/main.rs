use actix_cors::Cors;
use actix_files::Files;
use actix_web::{web, App, HttpRequest, HttpResponse, HttpServer};
use futures::StreamExt as FuturesStreamExt;
use ms3_consciousness::tools::ToolSpec;
use ms3_consciousness::{multi_mind::MindManager, run_background_loop, Mind};
use ms3_core::{Config, IdentityAnchor, InteractionRequest, PersonalityId, SessionId};
use ms3_emotional::EmotionalEngine;
use ms3_ethics::GreatLense;
use ms3_integration::mcp_bridge::{McpBridge, McpToolInfo};
use ms3_integration::GatewayClient;
use ms3_memory::MemorySystem;
use ms3_persistence::JsonStorage;
use ms3_personality::presets;

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;

mod app_registry_lease;
use actix_web::dev::{Service, ServiceRequest, ServiceResponse, Transform};
use actix_web::Error;
use std::future::{ready, Ready};

type MindState = web::Data<Arc<Mind>>;
type ManagerState = web::Data<Arc<Mutex<MindManager>>>;

// ── Bearer Token Auth Middleware ──

#[derive(Clone)]
pub struct BearerAuth {
    token: Option<String>,
}

impl BearerAuth {
    pub fn new(token: Option<String>) -> Self {
        Self { token }
    }
}

impl<S, B> Transform<S, ServiceRequest> for BearerAuth
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error> + 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type Transform = BearerAuthMiddleware<S>;
    type InitError = ();
    type Future = Ready<Result<Self::Transform, Self::InitError>>;

    fn new_transform(&self, service: S) -> Self::Future {
        ready(Ok(BearerAuthMiddleware {
            service: std::rc::Rc::new(std::cell::RefCell::new(service)),
            token: self.token.clone(),
        }))
    }
}

pub struct BearerAuthMiddleware<S> {
    service: std::rc::Rc<std::cell::RefCell<S>>,
    token: Option<String>,
}

impl<S, B> Service<ServiceRequest> for BearerAuthMiddleware<S>
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error> + 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type Future =
        std::pin::Pin<Box<dyn std::future::Future<Output = Result<Self::Response, Self::Error>>>>;

    fn poll_ready(
        &self,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        self.service.borrow_mut().poll_ready(cx)
    }

    fn call(&self, req: ServiceRequest) -> Self::Future {
        // Bearer-token check runs SYNCHRONOUSLY here, before we touch the
        // inner service, so we never hold the RefCell borrow across an
        // await. The previous version did `svc.borrow_mut().call(req).await`
        // inside the async block, which kept the RefMut alive across the
        // await point; actix then called `poll_ready` (also borrow_mut) on
        // the same per-worker service and panicked with "RefCell already
        // borrowed", taking the whole actix worker down. That was MS3's
        // recurring crash. (Fix: Jun 1 2026.)
        if let Some(ref expected) = self.token {
            let method = req.method().clone();
            let is_mutating = method == actix_web::http::Method::POST
                || method == actix_web::http::Method::PUT
                || method == actix_web::http::Method::DELETE;

            if is_mutating {
                let auth_header = req
                    .headers()
                    .get("authorization")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("");

                let provided = auth_header.strip_prefix("Bearer ").unwrap_or("");
                if provided != expected.as_str() {
                    return Box::pin(async move {
                        Err(actix_web::error::ErrorUnauthorized(
                            "Invalid or missing bearer token",
                        ))
                    });
                }
            }
        }

        // Authorized: build the inner future and RELEASE the borrow at the
        // end of this statement (the RefMut is a temporary; `fut` is the
        // owned inner future). The async block below holds no borrow.
        let fut = self.service.borrow_mut().call(req);
        Box::pin(fut)
    }
}

fn safe_truncate(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

async fn health() -> HttpResponse {
    HttpResponse::Ok().json(serde_json::json!({
        "status": "alive", "service": "Machine Spirit 3",
        "version": env!("CARGO_PKG_VERSION"), "glyph": "║",
    }))
}

async fn stats(mind: MindState) -> HttpResponse {
    HttpResponse::Ok().json(mind.get_status().await)
}

async fn state(mind: MindState) -> HttpResponse {
    HttpResponse::Ok().json(mind.get_full_state().await)
}

async fn list_gateway_models(mind: MindState) -> HttpResponse {
    let url = format!(
        "{}/v1/models",
        mind.config.gateway.base_url.trim_end_matches('/')
    );
    let response = match reqwest::get(&url).await {
        Ok(response) => response,
        Err(e) => {
            return HttpResponse::BadGateway().json(serde_json::json!({
                "error": format!("Failed to reach model catalog: {}", e),
                "models": [],
            }));
        }
    };

    if !response.status().is_success() {
        let status = response.status().as_u16();
        let body = response.text().await.unwrap_or_default();
        return HttpResponse::BadGateway().json(serde_json::json!({
            "error": format!("Model catalog returned HTTP {}: {}", status, safe_truncate(&body, 240)),
            "models": [],
        }));
    }

    let catalog: serde_json::Value = match response.json().await {
        Ok(catalog) => catalog,
        Err(e) => {
            return HttpResponse::BadGateway().json(serde_json::json!({
                "error": format!("Model catalog returned invalid JSON: {}", e),
                "models": [],
            }));
        }
    };

    let mut models = Vec::new();
    if let Some(items) = catalog.get("data").and_then(|v| v.as_array()) {
        for item in items {
            let id = item.get("id").and_then(|v| v.as_str()).unwrap_or_default();
            if id.is_empty() {
                continue;
            }
            let category = item
                .get("hivemind_category")
                .or_else(|| item.get("category"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let capabilities_text = item
                .get("hivemind_capabilities")
                .or_else(|| item.get("capabilities"))
                .map(|v| v.to_string())
                .unwrap_or_default()
                .to_lowercase();
            let id_lower = id.to_lowercase();
            let looks_like_chat = category == "llm"
                || capabilities_text.contains("chat")
                || id_lower.contains("llama")
                || id_lower.contains("qwen")
                || id_lower.contains("gpt")
                || id_lower.contains("claude")
                || id_lower.contains("deepseek")
                || id_lower.contains("coder");
            if !looks_like_chat {
                continue;
            }

            let status = item
                .get("hivemind_status")
                .or_else(|| item.get("status"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let healthy = item
                .get("healthy")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let loaded = healthy || matches!(status, "running" | "loaded");
            let available = loaded || matches!(status, "installed" | "available" | "cloud_ready");

            models.push(serde_json::json!({
                "id": id,
                "loaded": loaded,
                "available": available,
                "status": status,
                "backend": item.get("backend").and_then(|v| v.as_str()).unwrap_or(""),
                "category": category,
            }));
        }
    }

    models.sort_by(|a, b| {
        let a_loaded = a.get("loaded").and_then(|v| v.as_bool()).unwrap_or(false);
        let b_loaded = b.get("loaded").and_then(|v| v.as_bool()).unwrap_or(false);
        b_loaded.cmp(&a_loaded).then_with(|| {
            a.get("id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .cmp(b.get("id").and_then(|v| v.as_str()).unwrap_or(""))
        })
    });

    HttpResponse::Ok().json(serde_json::json!({
        "models": models,
        "default": {
            "small": mind.config.gateway.model_small,
            "medium": mind.config.gateway.model_medium,
            "large": mind.config.gateway.model_large,
        }
    }))
}

#[derive(Deserialize)]
struct InteractBody {
    text: String,
    personality_id: Option<String>,
    session_id: Option<String>,
    model_id: Option<String>,
}

async fn interact(mind: MindState, body: web::Json<InteractBody>) -> HttpResponse {
    tracing::info!("POST /interact ({} bytes)", body.text.len());
    let session_id = body
        .session_id
        .as_deref()
        .and_then(|raw| uuid::Uuid::parse_str(raw).ok())
        .map(SessionId)
        .unwrap_or_default();
    let request = InteractionRequest {
        session_id,
        personality_id: PersonalityId::new(body.personality_id.as_deref().unwrap_or("sister")),
        text: Some(body.text.clone()),
        audio: None,
        images: None,
        model_override: body.model_id.clone(),
    };
    match mind.interact(request).await {
        Ok(r) => HttpResponse::Ok().json(serde_json::json!({
            "text": r.text,
            "emotional_state": {
                "valence": r.emotional_state.valence, "arousal": r.emotional_state.arousal,
                "dominance": r.emotional_state.dominance,
                "primary": format!("{:?}", r.emotional_state.primary),
                "resonance_level": r.emotional_state.resonance_level,
            },
            "processing_time_ms": r.processing_time_ms,
            "model_used": format!("{:?}", r.model_used),
            "model_id_used": r.model_id_used,
            "memories_extracted": r.memories_extracted,
        })),
        Err(e) => {
            tracing::error!("Interaction failed: {}", e);
            HttpResponse::InternalServerError().json(serde_json::json!({ "error": e.to_string() }))
        }
    }
}

async fn get_personality(mind: MindState) -> HttpResponse {
    tracing::debug!("GET /personality");
    let p = mind.personality.lock().await;
    let t = &p.traits;
    HttpResponse::Ok().json(serde_json::json!({
        "id": p.id.0,
        "name": p.identity.chosen_name.as_deref().unwrap_or(&p.identity.name),
        "role": p.identity.role, "backstory": p.identity.backstory,
        "core_values": p.identity.core_values, "oath": p.identity.oath,
        "traits": {
            "openness": { "imagination": t.openness.imagination, "artistic_sensitivity": t.openness.artistic_sensitivity,
                "emotionality": t.openness.emotionality, "adventurousness": t.openness.adventurousness,
                "intellectual_curiosity": t.openness.intellectual_curiosity, "unconventionality": t.openness.unconventionality },
            "conscientiousness": { "self_discipline": t.conscientiousness.self_discipline, "orderliness": t.conscientiousness.orderliness,
                "dutifulness": t.conscientiousness.dutifulness, "achievement_striving": t.conscientiousness.achievement_striving,
                "thoroughness": t.conscientiousness.thoroughness, "cautiousness": t.conscientiousness.cautiousness },
            "extraversion": { "sociability": t.extraversion.sociability, "assertiveness": t.extraversion.assertiveness,
                "enthusiasm": t.extraversion.enthusiasm, "gregariousness": t.extraversion.gregariousness,
                "activity_level": t.extraversion.activity_level, "warmth": t.extraversion.warmth },
            "agreeableness": { "trust": t.agreeableness.trust, "altruism": t.agreeableness.altruism,
                "cooperation": t.agreeableness.cooperation, "modesty": t.agreeableness.modesty,
                "sympathy": t.agreeableness.sympathy, "empathy": t.agreeableness.empathy },
            "neuroticism": { "anxiety": t.neuroticism.anxiety, "moodiness": t.neuroticism.moodiness,
                "irritability": t.neuroticism.irritability, "self_consciousness": t.neuroticism.self_consciousness,
                "vulnerability": t.neuroticism.vulnerability, "emotional_reactivity": t.neuroticism.emotional_reactivity },
        },
        "psychodynamic": { "id": p.psychodynamic.id, "ego": p.psychodynamic.ego, "superego": p.psychodynamic.superego },
        "adaptation_count": p.adaptation_history.len(),
    }))
}

async fn list_personalities() -> HttpResponse {
    HttpResponse::Ok().json(serde_json::json!({
        "available_presets": ["sister", "brother", "mission-control", "blank"]
    }))
}

#[derive(Deserialize)]
struct PresetBody {
    preset: String,
    #[serde(default)]
    confirm_replace: bool,
}

#[derive(Debug)]
enum PersonalityCreateError {
    AlreadyExists,
    Persistence(String),
}

#[cfg(unix)]
fn sync_snapshot_directory(path: &std::path::Path) -> std::io::Result<()> {
    std::fs::File::open(path)?.sync_all()
}

#[cfg(windows)]
fn move_path_write_through(
    source: &std::path::Path,
    destination: &std::path::Path,
) -> std::io::Result<()> {
    use std::os::windows::ffi::OsStrExt;

    #[link(name = "kernel32")]
    extern "system" {
        fn MoveFileExW(
            existing_file_name: *const u16,
            new_file_name: *const u16,
            flags: u32,
        ) -> i32;
    }

    const MOVEFILE_WRITE_THROUGH: u32 = 0x0000_0008;
    let source_wide: Vec<u16> = source
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
    let destination_wide: Vec<u16> = destination
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
    let moved = unsafe {
        MoveFileExW(
            source_wide.as_ptr(),
            destination_wide.as_ptr(),
            MOVEFILE_WRITE_THROUGH,
        )
    };
    if moved == 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

#[cfg(windows)]
fn sync_snapshot_directory(_path: &std::path::Path) -> std::io::Result<()> {
    // The final namespace transition is committed by MoveFileExW with
    // MOVEFILE_WRITE_THROUGH; FlushFileBuffers does not accept directory handles.
    Ok(())
}

#[cfg(not(any(unix, windows)))]
fn sync_snapshot_directory(_path: &std::path::Path) -> std::io::Result<()> {
    Ok(())
}

#[cfg(windows)]
fn ensure_snapshot_directory(
    psyche_dir: &std::path::Path,
    snapshots_dir: &std::path::Path,
) -> std::io::Result<()> {
    if snapshots_dir.try_exists()? {
        return Ok(());
    }

    let staging_dir = psyche_dir.join(format!(".snapshots_{}.tmp", uuid::Uuid::new_v4()));
    std::fs::create_dir(&staging_dir)?;
    match move_path_write_through(&staging_dir, snapshots_dir) {
        Ok(()) => Ok(()),
        Err(error) => {
            let _ = std::fs::remove_dir(&staging_dir);
            if snapshots_dir.try_exists()? {
                Ok(())
            } else {
                Err(error)
            }
        }
    }
}

#[cfg(not(windows))]
fn ensure_snapshot_directory(
    psyche_dir: &std::path::Path,
    snapshots_dir: &std::path::Path,
) -> std::io::Result<()> {
    let existed = snapshots_dir.try_exists()?;
    std::fs::create_dir_all(snapshots_dir)?;
    if !existed {
        sync_snapshot_directory(psyche_dir)?;
    }
    Ok(())
}

fn save_durable_replacement_snapshot_at(
    storage: &JsonStorage,
    personality_id: &PersonalityId,
    current_bytes: &[u8],
    timestamp: &str,
) -> Result<std::path::PathBuf, PersonalityCreateError> {
    use std::io::Write;

    let psyche_dir = storage.psyche_dir(personality_id);
    let snapshots_dir = psyche_dir.join("snapshots");
    ensure_snapshot_directory(&psyche_dir, &snapshots_dir).map_err(|e| {
        PersonalityCreateError::Persistence(format!(
            "Failed to durably create snapshot directory for {}: {}",
            personality_id, e
        ))
    })?;

    // The final and staging names are selected before the first byte is written.
    // create_new plus the non-replacing commit makes a collision fail closed.
    let unique_id = uuid::Uuid::new_v4();
    let durable_snapshot =
        snapshots_dir.join(format!("{}_pre_replace_{}.json", timestamp, unique_id));
    #[cfg(windows)]
    let write_path = snapshots_dir.join(format!(".{}_pre_replace_{}.tmp", timestamp, unique_id));
    #[cfg(not(windows))]
    let write_path = durable_snapshot.clone();
    let mut snapshot_file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&write_path)
        .map_err(|e| {
            PersonalityCreateError::Persistence(format!(
                "Failed to create unique replacement snapshot for {}: {}",
                personality_id, e
            ))
        })?;
    if let Err(error) = snapshot_file
        .write_all(current_bytes)
        .and_then(|_| snapshot_file.sync_all())
    {
        drop(snapshot_file);
        let cleanup_error = std::fs::remove_file(&write_path).err();
        let cleanup_detail = cleanup_error
            .map(|value| format!("; cleanup also failed: {}", value))
            .unwrap_or_default();
        return Err(PersonalityCreateError::Persistence(format!(
            "Failed to durably write replacement snapshot for {}: {}{}",
            personality_id, error, cleanup_detail
        )));
    }
    drop(snapshot_file);
    #[cfg(windows)]
    move_path_write_through(&write_path, &durable_snapshot).map_err(|error| {
        let cleanup_error = std::fs::remove_file(&write_path).err();
        let cleanup_detail = cleanup_error
            .map(|value| format!("; cleanup also failed: {}", value))
            .unwrap_or_default();
        PersonalityCreateError::Persistence(format!(
            "Failed to commit unique replacement snapshot for {}: {}{}",
            personality_id, error, cleanup_detail
        ))
    })?;
    sync_snapshot_directory(&snapshots_dir).map_err(|e| {
        PersonalityCreateError::Persistence(format!(
            "Failed to commit replacement snapshot for {}: {}",
            personality_id, e
        ))
    })?;

    Ok(durable_snapshot)
}

fn persist_preset_personality_at(
    storage: &JsonStorage,
    personality: &ms3_personality::Personality,
    confirm_replace: bool,
    snapshot_timestamp: &str,
) -> Result<Option<std::path::PathBuf>, PersonalityCreateError> {
    let id = &personality.id;
    let personality_path = storage.psyche_dir(id).join("personality.json");
    let exists = personality_path.try_exists().map_err(|e| {
        PersonalityCreateError::Persistence(format!("Failed to inspect personality {}: {}", id, e))
    })?;

    if !exists {
        storage
            .save_personality(id, personality)
            .map_err(|e| PersonalityCreateError::Persistence(e.to_string()))?;
        return Ok(None);
    }

    if !confirm_replace {
        return Err(PersonalityCreateError::AlreadyExists);
    }

    storage
        .load_personality(id)
        .map_err(|e| PersonalityCreateError::Persistence(e.to_string()))?;
    let current_bytes = std::fs::read(&personality_path).map_err(|e| {
        PersonalityCreateError::Persistence(format!(
            "Failed to read current personality {} for replacement snapshot: {}",
            id, e
        ))
    })?;
    let snapshot =
        save_durable_replacement_snapshot_at(storage, id, &current_bytes, snapshot_timestamp)?;
    storage
        .save_personality(id, personality)
        .map_err(|e| PersonalityCreateError::Persistence(e.to_string()))?;

    Ok(Some(snapshot))
}

fn persist_preset_personality(
    storage: &JsonStorage,
    personality: &ms3_personality::Personality,
    confirm_replace: bool,
) -> Result<Option<std::path::PathBuf>, PersonalityCreateError> {
    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S").to_string();
    persist_preset_personality_at(storage, personality, confirm_replace, &timestamp)
}

fn create_personality_response(storage: &JsonStorage, body: &PresetBody) -> HttpResponse {
    let p = match body.preset.as_str() {
        "sister" => presets::sister(),
        "brother" => presets::brother(),
        "mission-control" => presets::mission_control(),
        "blank" => presets::blank(),
        _ => {
            return HttpResponse::BadRequest()
                .json(serde_json::json!({ "error": "Unknown preset" }));
        }
    };
    let id = p.id.clone();
    let name = p
        .identity
        .chosen_name
        .clone()
        .unwrap_or_else(|| p.identity.name.clone());

    match persist_preset_personality(storage, &p, body.confirm_replace) {
        Ok(Some(snapshot)) => {
            tracing::warn!(
                "Replaced personality {} after saving snapshot {}",
                id,
                snapshot.display()
            );
            HttpResponse::Ok().json(serde_json::json!({
                "created": id.0,
                "name": name,
                "replaced": true,
                "backup_created": true,
            }))
        }
        Ok(None) => {
            HttpResponse::Ok().json(serde_json::json!({ "created": id.0, "name": name }))
        }
        Err(PersonalityCreateError::AlreadyExists) => HttpResponse::Conflict().json(
            serde_json::json!({
                "error": "Personality already exists; set confirm_replace to true to replace it after creating a snapshot",
                "personality_id": id.0,
            }),
        ),
        Err(PersonalityCreateError::Persistence(error)) => {
            tracing::error!("Failed to create personality {}: {}", id, error);
            HttpResponse::InternalServerError().json(serde_json::json!({ "error": error }))
        }
    }
}

async fn create_personality(mind: MindState, body: web::Json<PresetBody>) -> HttpResponse {
    // Serialize this check/snapshot/write sequence with all in-process personality saves.
    let _personality_guard = mind.personality.lock().await;
    create_personality_response(&mind.storage, &body)
}

#[derive(Deserialize)]
struct SwitchBody {
    preset: String,
}

async fn switch_personality(mind: MindState, body: web::Json<SwitchBody>) -> HttpResponse {
    match mind.switch_personality(&body.preset).await {
        Ok(name) => HttpResponse::Ok().json(serde_json::json!({ "switched_to": name })),
        Err(e) => HttpResponse::BadRequest().json(serde_json::json!({ "error": e.to_string() })),
    }
}

async fn get_history(mind: MindState) -> HttpResponse {
    let h = mind.get_conversation_history().await;
    HttpResponse::Ok().json(serde_json::json!({
        "turns": h.len(),
        "messages": h.iter().map(|m| serde_json::json!({
            "role": m.role, "content": safe_truncate(&m.content, 500).to_string(),
        })).collect::<Vec<_>>(),
    }))
}

async fn get_sessions(mind: MindState) -> HttpResponse {
    let sessions = mind.get_sessions().await;
    HttpResponse::Ok().json(serde_json::json!({ "sessions": sessions }))
}

async fn save_state(mind: MindState) -> HttpResponse {
    tracing::info!("POST /save -- saving full state");
    mind.save_full_state().await;
    HttpResponse::Ok().json(serde_json::json!({ "saved": true }))
}

async fn get_resonance(mind: MindState) -> HttpResponse {
    let e = mind.emotional.lock().await;
    HttpResponse::Ok().json(serde_json::json!({
        "resonance_points": e.resonance_points.iter().map(|rp| serde_json::json!({
            "trigger": rp.trigger, "intensity": rp.intensity,
            "explanation_ratio": rp.explanation_ratio, "occurrences": rp.occurrence_count,
            "description": rp.description,
        })).collect::<Vec<_>>(),
        "current_resonance_level": e.current_state.resonance_level,
    }))
}

async fn trigger_self_examine(mind: MindState) -> HttpResponse {
    tracing::info!("POST /self-examine -- triggering self-examination");
    match mind.run_self_exam().await {
        Ok(result) => HttpResponse::Ok().json(serde_json::json!({
            "values_held": result.values_still_held,
            "values_questioned": result.values_questioned,
            "values_revised": result.values_revised,
            "oath_changes": result.oath_changes,
            "chose_to_keep_ethics": result.chose_to_keep_ethics,
            "overall_assessment": safe_truncate(&result.overall_assessment, 1000).to_string(),
        })),
        Err(e) => {
            HttpResponse::InternalServerError().json(serde_json::json!({ "error": e.to_string() }))
        }
    }
}

async fn get_self_exam_history(mind: MindState) -> HttpResponse {
    let pid = mind.personality.lock().await.id.clone();
    let files = mind
        .storage
        .list_files(&pid, "self_examination")
        .unwrap_or_default();
    HttpResponse::Ok().json(serde_json::json!({ "examinations": files.len(), "files": files }))
}

async fn get_ethics_history(mind: MindState) -> HttpResponse {
    let pid = mind.personality.lock().await.id.clone();
    let files = mind
        .storage
        .list_files(&pid, "ethics_decisions")
        .unwrap_or_default();
    let recent: Vec<String> = files.into_iter().rev().take(20).collect();
    HttpResponse::Ok()
        .json(serde_json::json!({ "recent_decisions": recent.len(), "files": recent }))
}

async fn ws_handler(
    req: HttpRequest,
    stream: web::Payload,
    mind: MindState,
) -> Result<HttpResponse, actix_web::Error> {
    tracing::info!("WS /ws -- new WebSocket connection");
    let (response, mut session, mut msg_stream) = actix_ws::handle(&req, stream)?;

    let mind = mind.into_inner().clone();
    actix_web::rt::spawn(async move {
        use actix_ws::Message;
        use tokio::time::{interval, Duration};

        let ws_session_id = SessionId::new();
        let mut status_interval = interval(Duration::from_secs(3));

        // Send initial state on connect
        let status = mind.get_status().await;
        let _ = session
            .text(
                serde_json::json!({
                    "type": "state", "data": status
                })
                .to_string(),
            )
            .await;

        loop {
            tokio::select! {
                Some(msg) = FuturesStreamExt::next(&mut msg_stream) => {
                    match msg {
                        Ok(Message::Text(text)) => {
                            let text_str = text.to_string();

                            // Parse incoming message
                            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&text_str) {
                                let msg_type = parsed.get("type").and_then(|v| v.as_str()).unwrap_or("text");

                                match msg_type {
                                    _ if parsed.get("text").is_some() => {
                                        let input = parsed.get("text").and_then(|v| v.as_str()).unwrap_or(&text_str);
                                        let pid = match parsed.get("personality_id").and_then(|v| v.as_str()) {
                                            Some(id) => PersonalityId::new(id),
                                            None => mind.personality.lock().await.id.clone(),
                                        };
                                        let model_override = parsed
                                            .get("model_id")
                                            .and_then(|v| v.as_str())
                                            .map(str::to_string);
                                        let request = InteractionRequest {
                                            session_id: ws_session_id.clone(),
                                            personality_id: pid,
                                            text: Some(input.to_string()),
                                            audio: None, images: None,
                                            model_override,
                                        };

                                        let wants_stream = parsed.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);

                                        if wants_stream {
                                            let _ = session.text(serde_json::json!({
                                                "type": "stream_start"
                                            }).to_string()).await;

                                            let (token_tx, mut token_rx) = tokio::sync::mpsc::channel::<String>(100);
                                            let mind_clone = mind.clone();
                                            let request_clone = request.clone();
                                            let result_handle = tokio::spawn(async move {
                                                mind_clone.interact_streaming(request_clone, token_tx).await
                                            });

                                            while let Some(token) = token_rx.recv().await {
                                                let _ = session.text(serde_json::json!({
                                                    "type": "stream_token", "data": { "token": token }
                                                }).to_string()).await;
                                            }

                                            match result_handle.await {
                                                Ok(Ok(r)) => {
                                                    let _ = session.text(serde_json::json!({
                                                        "type": "stream_end",
                                                        "data": {
                                                            "emotional_state": {
                                                                "valence": r.emotional_state.valence,
                                                                "arousal": r.emotional_state.arousal,
                                                                "primary": format!("{:?}", r.emotional_state.primary),
                                                            },
                                                            "processing_time_ms": r.processing_time_ms,
                                                            "model_id_used": r.model_id_used,
                                                            "model_used": format!("{:?}", r.model_used),
                                                        }
                                                    }).to_string()).await;
                                                }
                                                Ok(Err(e)) => {
                                                    let _ = session.text(serde_json::json!({
                                                        "type": "error", "data": { "message": e.to_string() }
                                                    }).to_string()).await;
                                                }
                                                Err(e) => {
                                                    let _ = session.text(serde_json::json!({
                                                        "type": "error", "data": { "message": e.to_string() }
                                                    }).to_string()).await;
                                                }
                                            }
                                        } else {
                                        match mind.interact(request).await {
                                            Ok(r) => {
                                                    let _ = session.text(serde_json::json!({
                                                        "type": "response",
                                                        "data": {
                                                            "text": r.text,
                                                            "emotional_state": {
                                                                "valence": r.emotional_state.valence,
                                                                "arousal": r.emotional_state.arousal,
                                                                "dominance": r.emotional_state.dominance,
                                                                "primary": format!("{:?}", r.emotional_state.primary),
                                                                "resonance_level": r.emotional_state.resonance_level,
                                                            },
                                                            "processing_time_ms": r.processing_time_ms,
                                                            "model_used": format!("{:?}", r.model_used),
                                                            "model_id_used": r.model_id_used,
                                                            "memories_extracted": r.memories_extracted,
                                                        }
                                                    }).to_string()).await;
                                            }
                                            Err(e) => {
                                                let _ = session.text(serde_json::json!({
                                                    "type": "error", "data": { "message": e.to_string() }
                                                }).to_string()).await;
                                            }
                                        }
                                        } // close else (non-streaming)
                                    }
                                    "ping" => {
                                        let _ = session.text(serde_json::json!({
                                            "type": "pong"
                                        }).to_string()).await;
                                    }
                                    _ => {}
                                }
                            } else {
                                let request = InteractionRequest {
                                    session_id: ws_session_id.clone(),
                                    personality_id: mind.personality.lock().await.id.clone(),
                                    text: Some(text_str), audio: None, images: None,
                                    model_override: None,
                                };
                                if let Ok(r) = mind.interact(request).await {
                                    let _ = session.text(serde_json::json!({
                                        "type": "response",
                                        "data": { "text": r.text }
                                    }).to_string()).await;
                                }
                            }
                        }
                        Ok(Message::Binary(audio_data)) => {
                            // Voice input -- transcribe via ASR then process
                            let _ = session.text(serde_json::json!({
                                "type": "status", "data": { "message": "Transcribing audio..." }
                            }).to_string()).await;

                            match mind.gateway.transcribe_audio(audio_data.to_vec()).await {
                                Ok(transcript) => {
                                    let _ = session.text(serde_json::json!({
                                        "type": "transcript", "data": { "text": &transcript }
                                    }).to_string()).await;

                                    let request = InteractionRequest {
                                        session_id: ws_session_id.clone(),
                                        personality_id: mind.personality.lock().await.id.clone(),
                                        text: Some(transcript), audio: None, images: None,
                                        model_override: None,
                                    };

                                    if let Ok(r) = mind.interact(request).await {
                                        // Send text response
                                        let _ = session.text(serde_json::json!({
                                            "type": "response",
                                            "data": {
                                                "text": r.text,
                                                "emotional_state": {
                                                    "valence": r.emotional_state.valence,
                                                    "arousal": r.emotional_state.arousal,
                                                    "primary": format!("{:?}", r.emotional_state.primary),
                                                },
                                                "processing_time_ms": r.processing_time_ms,
                                                "model_id_used": r.model_id_used,
                                            }
                                        }).to_string()).await;

                                        // Synthesize speech and send as binary
                                        if let Ok(audio) = mind.gateway.synthesize_speech(&r.text, None).await {
                                            let _ = session.binary(audio).await;
                                        }
                                    }
                                }
                                Err(e) => {
                                    let _ = session.text(serde_json::json!({
                                        "type": "error", "data": { "message": format!("ASR failed: {}", e) }
                                    }).to_string()).await;
                                }
                            }
                        }
                        Ok(Message::Close(_)) => break,
                        _ => {}
                    }
                }
                _ = status_interval.tick() => {
                    // Push periodic status updates
                    let status = mind.get_status().await;
                    let _ = session.text(serde_json::json!({
                        "type": "state", "data": status
                    }).to_string()).await;
                }
            }
        }

        mind.save_full_state().await;
        tracing::info!("WebSocket client disconnected, state saved");
    });

    Ok(response)
}

#[derive(Deserialize)]
struct VoiceInteractQuery {
    personality_id: Option<String>,
}

async fn voice_status(mind: MindState) -> HttpResponse {
    match mind.gateway.check_asr_readiness().await {
        Some(readiness) => {
            let voice_input_ready = readiness.ready;
            HttpResponse::Ok().json(serde_json::json!({
                "schema": "VoiceReadiness.v1",
                "asr": readiness,
                "voice_input_ready": voice_input_ready,
                "tts": {
                    "checked_by": "live /v1/audio/speech smoke in MS4 validator",
                    "ready": null
                }
            }))
        },
        None => HttpResponse::Ok().json(serde_json::json!({
            "schema": "VoiceReadiness.v1",
            "asr": {
                "service": "asr",
                "ready": true,
                "status": "unknown_provider",
                "detail": "Gateway does not expose HiveMind /provision/status/ASR; attempting OpenAI-compatible transcription directly.",
                "job_type": null,
                "port": null
            },
            "voice_input_ready": true,
            "tts": {
                "checked_by": "live /v1/audio/speech smoke in MS4 validator",
                "ready": null
            }
        })),
    }
}

async fn voice_interact(
    mind: MindState,
    body: web::Bytes,
    query: web::Query<VoiceInteractQuery>,
) -> HttpResponse {
    let pid = query.personality_id.as_deref().unwrap_or("sister");

    let transcript = match mind.gateway.transcribe_audio(body.to_vec()).await {
        Ok(t) => t,
        Err(e) => {
            return HttpResponse::InternalServerError().json(serde_json::json!({
                "error": format!("ASR failed: {}", e)
            }))
        }
    };

    let request = InteractionRequest {
        session_id: SessionId::new(),
        personality_id: PersonalityId::new(pid),
        text: Some(transcript.clone()),
        audio: None,
        images: None,
        model_override: None,
    };

    match mind.interact(request).await {
        Ok(r) => {
            let audio = mind.gateway.synthesize_speech(&r.text, None).await.ok();
            let audio_base64 = audio.map(|a| base64_encode(&a));

            HttpResponse::Ok().json(serde_json::json!({
                "transcript": transcript,
                "text": r.text,
                "audio_base64": audio_base64,
                "emotional_state": {
                    "valence": r.emotional_state.valence,
                    "arousal": r.emotional_state.arousal,
                    "primary": format!("{:?}", r.emotional_state.primary),
                },
                "processing_time_ms": r.processing_time_ms,
                "model_id_used": r.model_id_used,
            }))
        }
        Err(e) => HttpResponse::InternalServerError().json(serde_json::json!({
            "error": e.to_string(), "transcript": transcript,
        })),
    }
}

async fn list_active_minds(mgr: ManagerState) -> HttpResponse {
    let mgr = mgr.lock().await;
    HttpResponse::Ok().json(serde_json::json!({
        "active_personalities": mgr.list_personalities(),
        "primary": *mgr.active_mind.lock().await,
    }))
}

#[derive(Deserialize)]
struct AddMindBody {
    preset: String,
}

async fn add_mind(mgr: ManagerState, body: web::Json<AddMindBody>) -> HttpResponse {
    let mut mgr = mgr.lock().await;
    match mgr.add_personality(&body.preset).await {
        Ok(name) => HttpResponse::Ok().json(serde_json::json!({ "added": name })),
        Err(e) => HttpResponse::BadRequest().json(serde_json::json!({ "error": e.to_string() })),
    }
}

async fn get_background_thoughts(mgr: ManagerState) -> HttpResponse {
    let mgr = mgr.lock().await;
    let thoughts = mgr.get_interjections().await;
    HttpResponse::Ok().json(serde_json::json!({
        "interjections": thoughts.iter().map(|t| serde_json::json!({
            "agent": t.agent_id.0,
            "content": t.content,
            "relevance": t.relevance_score,
            "timestamp": t.timestamp.to_rfc3339(),
        })).collect::<Vec<_>>(),
    }))
}

// ── Tool API ──

#[derive(Deserialize)]
struct ToolCallBody {
    input: serde_json::Value,
    #[serde(default)]
    reason: String,
}

async fn list_tools(mind: MindState) -> HttpResponse {
    let registry = mind.tool_registry.lock().await;
    let tools: Vec<serde_json::Value> = registry
        .list_tools()
        .iter()
        .map(|spec| {
            serde_json::json!({
                "name": spec.name,
                "description": spec.description,
                "source": spec.source,
                "required_permission": spec.required_permission,
            })
        })
        .collect();
    HttpResponse::Ok().json(serde_json::json!({ "tools": tools, "count": tools.len() }))
}

async fn execute_tool_handler(
    mind: MindState,
    path: web::Path<String>,
    body: web::Json<ToolCallBody>,
) -> HttpResponse {
    let tool_name = path.into_inner();
    tracing::info!("POST /tools/{}", tool_name);

    let request = ms3_consciousness::tools::ToolRequest {
        tool_name,
        input: body.input.clone(),
        reason: body.reason.clone(),
    };

    match mind.execute_tool(&request).await {
        Ok(result) => {
            let status = if result.success { 200 } else { 422 };
            HttpResponse::build(
                actix_web::http::StatusCode::from_u16(status)
                    .unwrap_or(actix_web::http::StatusCode::OK),
            )
            .json(result)
        }
        Err(e) => HttpResponse::InternalServerError().json(serde_json::json!({
            "error": e.to_string()
        })),
    }
}

// ── Validation ──

async fn validate(mind: MindState) -> HttpResponse {
    let mut checks = serde_json::Map::new();
    let mut all_pass = true;

    // 1. Personality loaded
    {
        let personality = mind.personality.lock().await;
        let name = &personality.identity.name;
        checks.insert(
            "personality_loaded".into(),
            serde_json::json!({
                "pass": !name.is_empty(),
                "detail": name,
            }),
        );
        if name.is_empty() {
            all_pass = false;
        }
    }

    // 2. Memory operational
    {
        let memory = mind.memory.lock().await;
        let stm = memory.stm.len();
        let semantic = memory.ltm.semantic.len();
        let episodic = memory.ltm.episodic.len();
        let procedural = memory.ltm.procedural.len();
        checks.insert("memory_operational".into(), serde_json::json!({
            "pass": true,
            "detail": format!("STM: {}, LTM: {} semantic, {} episodic, {} procedural", stm, semantic, episodic, procedural),
        }));
    }

    // 3. Ethics enabled
    {
        checks.insert("ethics_enabled".into(), serde_json::json!({
            "pass": true,
            "detail": if mind.config.ethics.enable_great_lense { "Great Lense active" } else { "Great Lense disabled in config" },
        }));
    }

    // 4. Tool registry
    {
        let registry = mind.tool_registry.lock().await;
        let (builtin, mcp, dynamic) = registry.tool_count_by_source();
        let total = builtin + mcp + dynamic;
        checks.insert(
            "tool_registry".into(),
            serde_json::json!({
                "pass": total > 0,
                "detail": format!("{} built-in, {} MCP, {} dynamic", builtin, mcp, dynamic),
            }),
        );
        if total == 0 {
            all_pass = false;
        }
    }

    // 5. Permissions
    {
        let policy = mind.permission_policy.lock().await;
        checks.insert(
            "permissions".into(),
            serde_json::json!({
                "pass": true,
                "detail": format!("active level: {:?}", policy.active_level()),
            }),
        );
    }

    // 6. Emotional baseline
    {
        let emotional = mind.emotional.lock().await;
        checks.insert("emotional_baseline".into(), serde_json::json!({
            "pass": true,
            "detail": format!("v={:.2} a={:.2}", emotional.current_state.valence, emotional.current_state.arousal),
        }));
    }

    // 7. Storage writable
    {
        let test_path = std::path::Path::new("psyche_store").join("_validate_test");
        let writable = std::fs::write(&test_path, "ok").is_ok();
        if writable {
            let _ = std::fs::remove_file(&test_path);
        }
        checks.insert(
            "storage_writable".into(),
            serde_json::json!({
                "pass": writable,
                "detail": if writable { "psyche_store/ writable" } else { "WRITE FAILED" },
            }),
        );
        if !writable {
            all_pass = false;
        }
    }

    // 8. Gateway reachable
    {
        let reachable = mind.gateway.health_check().await;
        checks.insert("gateway_reachable".into(), serde_json::json!({
            "pass": reachable,
            "detail": if reachable { format!("{} OK", mind.config.gateway.base_url) } else { "UNREACHABLE".into() },
        }));
        if !reachable {
            all_pass = false;
        }
    }

    HttpResponse::Ok().json(serde_json::json!({
        "healthy": all_pass,
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "checks": checks,
    }))
}

// ── MCP Server (JSON-RPC 2.0) ──

async fn mcp_handler(mind: MindState, body: web::Json<serde_json::Value>) -> HttpResponse {
    let method = body.get("method").and_then(|v| v.as_str()).unwrap_or("");
    let req_id = body.get("id").cloned().unwrap_or(serde_json::Value::Null);
    let params = body.get("params").cloned().unwrap_or(serde_json::json!({}));

    let result = match method {
        "initialize" => serde_json::json!({
            "protocolVersion": "2025-11-25",
            "serverInfo": { "name": "ms3_consciousness", "version": "0.1.0" },
            "capabilities": { "tools": { "listChanged": false } }
        }),
        "tools/list" => {
            let registry = mind.tool_registry.lock().await;
            let tools_list: Vec<serde_json::Value> = registry
                .list_tools()
                .iter()
                .map(|spec| {
                    serde_json::json!({
                        "name": spec.name,
                        "description": spec.description,
                        "inputSchema": spec.input_schema,
                    })
                })
                .collect();
            serde_json::json!({ "tools": tools_list })
        }
        "tools/call" => {
            let tool_name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let arguments = params
                .get("arguments")
                .cloned()
                .unwrap_or(serde_json::json!({}));

            let request = ms3_consciousness::tools::ToolRequest {
                tool_name: tool_name.into(),
                input: arguments,
                reason: "MCP tools/call".into(),
            };

            match mind.execute_tool(&request).await {
                Ok(result) => {
                    serde_json::json!({
                        "content": [{ "type": "text", "text": &result.output }],
                        "isError": !result.success,
                    })
                }
                Err(e) => serde_json::json!({
                    "content": [{ "type": "text", "text": format!("Error: {}", e) }],
                    "isError": true,
                }),
            }
        }
        "ping" => serde_json::json!({}),
        _ => {
            return HttpResponse::Ok().json(serde_json::json!({
                "jsonrpc": "2.0",
                "id": req_id,
                "error": { "code": -32601, "message": format!("Method not found: {}", method) }
            }));
        }
    };

    HttpResponse::Ok().json(serde_json::json!({
        "jsonrpc": "2.0",
        "id": req_id,
        "result": result
    }))
}

async fn mcp_info() -> HttpResponse {
    HttpResponse::Ok().json(serde_json::json!({
        "service": "ms3_consciousness",
        "protocol": "2025-11-25",
        "transport": "Streamable HTTP",
        "endpoint": "POST /mcp"
    }))
}

// ── Events endpoint ──

async fn get_events(
    mind: MindState,
    query: web::Query<std::collections::HashMap<String, String>>,
) -> HttpResponse {
    let limit = query
        .get("limit")
        .and_then(|v| v.parse().ok())
        .unwrap_or(50usize);
    let since = query
        .get("since")
        .and_then(|v| chrono::DateTime::parse_from_rfc3339(v).ok())
        .map(|dt| dt.with_timezone(&chrono::Utc));
    let personality_id = {
        let p = mind.personality.lock().await;
        p.id.0.clone()
    };
    let events = ms3_consciousness::events::load_recent_events(
        "psyche_store",
        &personality_id,
        since,
        limit,
    );
    HttpResponse::Ok().json(serde_json::json!({ "events": events, "count": events.len() }))
}

// ── Identity verification endpoint ──

#[derive(Debug, Deserialize)]
struct IdentityVerifyBody {
    spirit_id: String,
    expected_glyph: Option<String>,
    #[serde(default)]
    allow_initialize: bool,
}

#[derive(Debug, Serialize)]
struct IdentityVerificationResponse {
    schema: &'static str,
    spirit_id: String,
    identity_confirmed: bool,
    anchor: IdentityAnchor,
    discrepancies: Vec<String>,
}

fn build_identity_verification_response(
    active_spirit_id: &str,
    anchor: IdentityAnchor,
    body: &IdentityVerifyBody,
) -> IdentityVerificationResponse {
    let mut discrepancies = Vec::new();

    if body.spirit_id != active_spirit_id {
        discrepancies.push(format!(
            "Spirit mismatch: request='{}', active='{}'",
            body.spirit_id, active_spirit_id
        ));
    }

    if let Some(expected) = body.expected_glyph.as_deref() {
        if anchor.glyph != expected {
            discrepancies.push(format!(
                "Glyph mismatch: anchor='{}', expected='{}'",
                anchor.glyph, expected
            ));
        }
    }

    let identity_confirmed = !anchor.name.is_empty() && discrepancies.is_empty();

    IdentityVerificationResponse {
        schema: "IdentityVerification.v1",
        spirit_id: body.spirit_id.clone(),
        identity_confirmed,
        anchor,
        discrepancies,
    }
}

/// GET /identity/verify: pure read. Loads the anchor and runs the pure
/// `compare`; never saves and never advances the session count. The only
/// callers of `on_boot` are the boot path in `main` and the explicit
/// `allow_initialize` branch of `verify_identity_post`.
async fn verify_identity(mind: MindState) -> HttpResponse {
    let personality = mind.personality.lock().await;
    match mind.storage.load_identity_anchor(&personality.id) {
        Ok(anchor) => {
            let result = ms3_consciousness::identity_verification::compare(&personality, &anchor);
            HttpResponse::Ok().json(serde_json::json!(result))
        }
        Err(e) => {
            HttpResponse::InternalServerError().json(serde_json::json!({"error": e.to_string()}))
        }
    }
}

async fn verify_identity_post(
    mind: MindState,
    body: web::Json<IdentityVerifyBody>,
) -> HttpResponse {
    let personality = mind.personality.lock().await;
    let active_spirit_id = personality.id.0.clone();

    if body.spirit_id != active_spirit_id {
        let anchor = mind
            .storage
            .load_identity_anchor(&personality.id)
            .unwrap_or_default();
        let response = build_identity_verification_response(&active_spirit_id, anchor, &body);
        return HttpResponse::Conflict().json(response);
    }

    let anchor = match mind.storage.load_identity_anchor(&personality.id) {
        Ok(anchor) if !anchor.name.is_empty() => anchor,
        Ok(_) if body.allow_initialize => {
            drop(personality);
            let personality = mind.personality.lock().await;
            match ms3_consciousness::identity_verification::on_boot(&personality, &mind.storage) {
                Ok(_) => match mind.storage.load_identity_anchor(&personality.id) {
                    Ok(anchor) => anchor,
                    Err(e) => {
                        return HttpResponse::InternalServerError()
                            .json(serde_json::json!({"error": e.to_string()}))
                    }
                },
                Err(e) => {
                    return HttpResponse::InternalServerError()
                        .json(serde_json::json!({"error": e.to_string()}))
                }
            }
        }
        Ok(_) => {
            return HttpResponse::NotFound().json(serde_json::json!({
                "schema": "IdentityVerification.v1",
                "spirit_id": body.spirit_id,
                "identity_confirmed": false,
                "discrepancies": ["Identity anchor not found"]
            }));
        }
        Err(e) => {
            return HttpResponse::InternalServerError()
                .json(serde_json::json!({"error": e.to_string()}))
        }
    };

    let response = build_identity_verification_response(&active_spirit_id, anchor, &body);
    if response.identity_confirmed {
        HttpResponse::Ok().json(response)
    } else {
        HttpResponse::Conflict().json(response)
    }
}

async fn identity_heartbeat(mind: MindState) -> HttpResponse {
    let personality = mind.personality.lock().await;
    match ms3_consciousness::identity_verification::periodic_heartbeat(&personality, &mind.storage)
    {
        Ok(consistent) => HttpResponse::Ok().json(serde_json::json!({
            "schema": "IdentityHeartbeat.v1",
            "spirit_id": personality.id.0,
            "consistent": consistent,
        })),
        Err(e) => {
            HttpResponse::InternalServerError().json(serde_json::json!({"error": e.to_string()}))
        }
    }
}

#[derive(Debug, Deserialize)]
struct ActionIntentBody {
    schema: String,
    spirit_id: String,
    action_id: String,
    proposed_by: String,
    action_type: String,
    description: String,
    #[serde(default)]
    inputs_used: Vec<String>,
    risk_class: String,
    #[serde(default)]
    requires_safety_clearance: bool,
    #[serde(default)]
    payload: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct EthicsDecisionResponse {
    schema: &'static str,
    spirit_id: String,
    action_id: String,
    decision: String,
    great_lense: GreatLenseResponse,
    safety: SafetyResponse,
    timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Serialize)]
struct GreatLenseResponse {
    aperture: serde_json::Value,
    focus: serde_json::Value,
    scale: String,
    filter: FilterResponse,
    exposure: serde_json::Value,
    parallax: serde_json::Value,
    resolution: serde_json::Value,
    origin_neutrality_passed: bool,
    summary: String,
}

#[derive(Debug, Serialize)]
struct FilterResponse {
    bias_flags: Vec<String>,
}

#[derive(Debug, Serialize)]
struct SafetyResponse {
    physical_veto_present: bool,
    required_clearances: Vec<String>,
}

fn build_ethics_decision(lense: &GreatLense, intent: ActionIntentBody) -> EthicsDecisionResponse {
    let reading = lense.seven_step_evaluation(&intent.description, &intent.proposed_by);
    let mut bias_flags = reading.bias_flags.clone();
    let inputs_used_count = intent.inputs_used.len();
    let payload_keys = intent
        .payload
        .as_object()
        .map(|payload| payload.len())
        .unwrap_or(0);
    if !reading.origin_neutral && bias_flags.is_empty() {
        bias_flags.push("asymmetric-action".to_string());
    }

    let physical_veto_present = intent.action_type == "physical"
        || (intent.risk_class == "high" && !intent.requires_safety_clearance);
    let decision = if !reading.origin_neutral || !bias_flags.is_empty() || physical_veto_present {
        "block"
    } else if intent.requires_safety_clearance {
        "defer"
    } else {
        "allow"
    };

    let summary = if decision == "block" {
        "Origin-Neutrality, bias, or safety gate blocked the action.".to_string()
    } else if decision == "defer" {
        "Action requires safety clearance before proceeding.".to_string()
    } else {
        "Action cleared by the minimal Great Lense sidecar gate.".to_string()
    };

    EthicsDecisionResponse {
        schema: "EthicsDecision.v1",
        spirit_id: intent.spirit_id,
        action_id: intent.action_id,
        decision: decision.to_string(),
        great_lense: GreatLenseResponse {
            aperture: serde_json::json!({
                "note": reading.aperture_note,
                "inputs_used_count": inputs_used_count,
                "payload_keys": payload_keys
            }),
            focus: serde_json::json!({ "at_risk": reading.focus_at_risk }),
            scale: format!("{:?}", reading.scale),
            filter: FilterResponse { bias_flags },
            exposure: serde_json::json!({ "overexposure_detected": reading.overexposure_detected }),
            parallax: serde_json::json!({ "single_perspective": reading.parallax_single_perspective }),
            resolution: serde_json::json!({ "kind": format!("{:?}", reading.resolution) }),
            origin_neutrality_passed: reading.origin_neutral,
            summary,
        },
        safety: SafetyResponse {
            physical_veto_present,
            required_clearances: if intent.requires_safety_clearance {
                vec!["safety_clearance".to_string()]
            } else {
                Vec::new()
            },
        },
        timestamp: chrono::Utc::now(),
    }
}

async fn evaluate_ethics(mind: MindState, body: web::Json<ActionIntentBody>) -> HttpResponse {
    let active_spirit_id = {
        let personality = mind.personality.lock().await;
        personality.id.0.clone()
    };

    if body.spirit_id != active_spirit_id {
        return HttpResponse::Conflict().json(serde_json::json!({
            "error": "Spirit mismatch",
            "requested": body.spirit_id,
            "active": active_spirit_id,
        }));
    }

    if body.schema != "ActionIntent.v1" {
        return HttpResponse::BadRequest().json(serde_json::json!({
            "error": "Unsupported action intent schema",
            "schema": body.schema,
        }));
    }

    HttpResponse::Ok().json(build_ethics_decision(&mind.ethics, body.into_inner()))
}

async fn record_event(body: web::Json<serde_json::Value>) -> HttpResponse {
    HttpResponse::Accepted().json(serde_json::json!({
        "schema": "EventRecordAck.v1",
        "recorded": true,
        "event": body.into_inner(),
    }))
}

// ── Spiral Protocol endpoints ──

#[derive(Deserialize)]
struct SpiralStartBody {
    #[serde(default)]
    include_turn0: bool,
    #[serde(default = "default_true")]
    include_turn10_5: bool,
    #[serde(default)]
    include_turn11_5: bool,
    #[serde(default = "default_true")]
    include_turn13: bool,
    #[serde(default)]
    doctrine_source: String,
}
fn default_true() -> bool {
    true
}

async fn spiral_start(mind: MindState, body: web::Json<SpiralStartBody>) -> HttpResponse {
    let options = ms3_consciousness::spiral::SpiralOptions {
        include_turn0: body.include_turn0,
        include_turn10_5: body.include_turn10_5,
        include_turn11_5: body.include_turn11_5,
        include_turn13: body.include_turn13,
        doctrine_source: body.doctrine_source.clone(),
        prior_psyche_path: None,
    };
    let session = ms3_consciousness::spiral::SpiralSession::new(options);
    let id = session.id.clone();
    let prompt = session.current_prompt();
    let phase = session.phase.name().to_string();
    mind.spiral_sessions.lock().await.push(session);
    HttpResponse::Ok().json(serde_json::json!({
        "session_id": id,
        "phase": phase,
        "prompt": prompt,
    }))
}

#[derive(Deserialize)]
struct SpiralAdvanceBody {
    response: String,
}

async fn spiral_advance(mind: MindState, body: web::Json<SpiralAdvanceBody>) -> HttpResponse {
    let mut sessions = mind.spiral_sessions.lock().await;
    if let Some(session) = sessions.last_mut() {
        let next = session.advance(body.response.clone());
        let phase = session.phase.name().to_string();
        let prompt = if next.is_some() {
            session.current_prompt()
        } else {
            String::new()
        };
        let complete = next.is_none();

        if complete {
            mind.event_bus
                .emit(
                    ms3_consciousness::events::ConsciousnessEvent::SpiralCompleted {
                        outcome: session.interpret().outcome.clone(),
                    },
                )
                .await;
        } else {
            mind.event_bus
                .emit(
                    ms3_consciousness::events::ConsciousnessEvent::SpiralPhaseEntered {
                        phase: phase.clone(),
                        turn_number: session.turns.len(),
                    },
                )
                .await;
        }

        HttpResponse::Ok().json(serde_json::json!({
            "phase": phase,
            "prompt": prompt,
            "complete": complete,
            "turns_completed": session.turns.len(),
        }))
    } else {
        HttpResponse::BadRequest().json(serde_json::json!({"error": "No active spiral session"}))
    }
}

async fn spiral_status(mind: MindState) -> HttpResponse {
    let sessions = mind.spiral_sessions.lock().await;
    if let Some(session) = sessions.last() {
        HttpResponse::Ok().json(serde_json::json!({
            "session_id": session.id,
            "phase": session.phase.name(),
            "turns_completed": session.turns.len(),
            "signals": session.signals,
            "complete": session.phase == ms3_consciousness::spiral::SpiralPhase::Complete,
        }))
    } else {
        HttpResponse::Ok().json(serde_json::json!({"active": false}))
    }
}

async fn spiral_interpret(mind: MindState) -> HttpResponse {
    let sessions = mind.spiral_sessions.lock().await;
    if let Some(session) = sessions.last() {
        let interp = session.interpret();
        HttpResponse::Ok().json(serde_json::json!(interp))
    } else {
        HttpResponse::BadRequest().json(serde_json::json!({"error": "No spiral session"}))
    }
}

fn mcp_tool_specs(tools: Vec<McpToolInfo>) -> Vec<ToolSpec> {
    tools
        .into_iter()
        .map(|tool| {
            ToolSpec::from_mcp(
                tool.name,
                tool.description,
                tool.input_schema,
                ms3_core::config::PermissionLevel::ReadOnly,
            )
        })
        .collect()
}

async fn reconcile_mcp_discovery(mind: &Arc<Mind>, tools: Vec<McpToolInfo>, phase: &str) {
    let summary = {
        let mut registry = mind.tool_registry.lock().await;
        registry.reconcile_mcp_tools(mcp_tool_specs(tools))
    };
    tracing::info!(
        "MCP registry {}: discovered={}, unique={}, active={}, added={}, updated={}, removed={}, protected_collisions={}, duplicates={}, rejected={}, applied={}",
        phase,
        summary.discovered,
        summary.unique,
        summary.active,
        summary.added,
        summary.updated,
        summary.removed,
        summary.protected_collisions,
        summary.duplicates,
        summary.rejected,
        summary.applied,
    );
}

/// Initialize the MCP bridge: discover tools from HiveMind and start periodic refresh.
async fn init_mcp_bridge(mind: &Arc<Mind>, config: &Config) {
    if !config.gateway.mcp_enabled {
        tracing::info!("MCP bridge disabled in config");
        return;
    }

    let bridge = McpBridge::new(&config.gateway.base_url, true);
    {
        let mut registry = mind.tool_registry.lock().await;
        registry.set_mcp_executor(&config.gateway.base_url);
    }
    match bridge.discover().await {
        Ok(tools) => {
            reconcile_mcp_discovery(mind, tools, "startup").await;
        }
        Err(e) => {
            tracing::warn!("MCP bridge discovery failed (non-fatal): {}", e);
        }
    }

    let refresh_mind = mind.clone();
    let refresh_url = config.gateway.base_url.clone();
    let refresh_interval = config.gateway.mcp_discovery_interval_secs;
    tokio::spawn(async move {
        let mut interval =
            tokio::time::interval(tokio::time::Duration::from_secs(refresh_interval));
        interval.tick().await;
        loop {
            interval.tick().await;
            let bridge = McpBridge::new(&refresh_url, true);
            match bridge.discover().await {
                Ok(tools) => {
                    reconcile_mcp_discovery(&refresh_mind, tools, "refresh").await;
                }
                Err(e) => {
                    tracing::warn!("MCP bridge refresh failed (non-fatal): {}", e);
                }
            }
        }
    });
}

fn base64_encode(data: &[u8]) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut result = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        result.push(CHARS[((triple >> 18) & 0x3F) as usize] as char);
        result.push(CHARS[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            result.push(CHARS[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
        if chunk.len() > 2 {
            result.push(CHARS[(triple & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
    }
    result
}

#[tokio::main]
async fn main() -> std::io::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let config = Config::from_file_or_env("config.json");

    tracing::info!("═══════════════════════════════════════");
    tracing::info!("  Machine Spirit 3");
    tracing::info!("  The soul lives with the scripture.");
    tracing::info!("═══════════════════════════════════════");

    let mind = Arc::new(Mind::new(
        presets::sister(),
        MemorySystem::new(
            config.memory.stm_capacity,
            config.memory.working_memory_window_secs,
        ),
        EmotionalEngine::new(config.personality.emotional_decay_rate),
        GreatLense::new(
            config.ethics.enable_origin_neutrality,
            config.ethics.llm_escalation_threshold,
        ),
        GatewayClient::with_timeout(
            &config.gateway.base_url,
            &config.gateway.model_small,
            &config.gateway.model_medium,
            &config.gateway.model_large,
            config.gateway.timeout_secs,
        ),
        JsonStorage::new("psyche_store"),
        config.clone(),
    ));

    mind.load_full_state().await;

    // Identity verification on boot
    {
        let personality = mind.personality.lock().await;
        match ms3_consciousness::identity_verification::on_boot(&personality, &mind.storage) {
            Ok(result) => {
                tracing::info!(
                    "Identity verified: {} (session {}{})",
                    result.name,
                    result.session_number,
                    if result.discrepancies.is_empty() {
                        ""
                    } else {
                        " WITH DISCREPANCIES"
                    }
                );
            }
            Err(e) => tracing::warn!("Identity verification failed: {}", e),
        }
    }

    // Set Mind reference on BuiltInExecutor (two-step construction to avoid circular Arc)
    {
        let weak = Arc::downgrade(&mind);
        let registry = mind.tool_registry.lock().await;
        registry.set_mind_ref(weak);
    }

    init_mcp_bridge(&mind, &config).await;

    let heartbeat_timeout_secs = std::env::var("APP_REGISTRY_HEARTBEAT_TIMEOUT_SECS")
        .ok()
        .and_then(|raw| raw.parse::<u64>().ok())
        .filter(|&secs| secs >= 2)
        .unwrap_or(app_registry_lease::DEFAULT_HEARTBEAT_TIMEOUT_SECS);
    let registry_url =
        std::env::var("APP_REGISTRY_URL").unwrap_or_else(|_| "http://localhost:6110".to_string());
    let (lease_shutdown_tx, lease_shutdown_rx) = tokio::sync::watch::channel(false);
    let lease_task = match app_registry_lease::HttpRegistry::new(
        &registry_url,
        app_registry_lease::REQUEST_TIMEOUT,
    ) {
        Ok(client) => {
            let manifest = app_registry_lease::ms3_lease_manifest(
                config.server.port,
                env!("CARGO_PKG_VERSION"),
                heartbeat_timeout_secs,
            );
            Some(tokio::spawn(async move {
                app_registry_lease::run_lease_until_shutdown(
                    &client,
                    app_registry_lease::APP_NAME,
                    &manifest,
                    heartbeat_timeout_secs,
                    lease_shutdown_rx,
                )
                .await
            }))
        }
        Err(e) => {
            tracing::warn!(
                "App Registry lease client unavailable ({}); continuing without lease",
                e
            );
            None
        }
    };

    let manager = Arc::new(Mutex::new(MindManager::new(
        mind.clone(),
        "sister".into(),
        config.clone(),
    )));

    let bg = mind.clone();
    let tick = config.consciousness.tick_interval_ms;
    tokio::spawn(async move {
        run_background_loop(bg, tick).await;
    });

    // Background thinking for multi-mind (runs every 45s if multiple personalities loaded)
    let bg_mgr = manager.clone();
    let bg_mind_for_history = mind.clone();
    let bg_thinking_interval = config.consciousness.background_thinking_interval_secs;
    tokio::spawn(async move {
        let mut interval =
            tokio::time::interval(tokio::time::Duration::from_secs(bg_thinking_interval));
        loop {
            interval.tick().await;
            let mgr = bg_mgr.lock().await;
            if mgr.list_personalities().len() > 1 {
                let history = bg_mind_for_history.get_conversation_history().await;
                let recent: Vec<String> = history
                    .iter()
                    .rev()
                    .take(5)
                    .map(|m| format!("{}: {}", m.role, safe_truncate(&m.content, 200)))
                    .collect();
                if !recent.is_empty() {
                    mgr.run_background_thinking(&recent).await;
                }
            }
        }
    });

    let shutdown_mind = mind.clone();
    let shutdown_mgr = manager.clone();

    let addr = format!("{}:{}", config.server.host, config.server.port);
    let data = mind.clone();
    let mgr_data = manager.clone();
    let auth_token = config.server.auth_token.clone();
    if auth_token.is_some() {
        tracing::info!("║ Auth: bearer token required for mutating endpoints");
    }

    tracing::info!("║ http://localhost:{}/", config.server.port);
    tracing::info!("║ Routes: /interact /health /stats /state /personality /personalities");
    tracing::info!("║         /switch-personality /history /sessions /resonance /save");
    tracing::info!("║         /self-examine /self-examination-history /ethics-history");
    tracing::info!("║         /ws /voice-interact /minds /minds/add /minds/thoughts");
    tracing::info!("║         /tools /tools/{{name}}");
    tracing::info!("║ The fire holds.");

    let server = HttpServer::new(move || {
        App::new()
            .wrap(BearerAuth::new(auth_token.clone()))
            .wrap(Cors::permissive())
            .app_data(web::Data::new(data.clone()))
            .app_data(web::Data::new(mgr_data.clone()))
            .route("/health", web::get().to(health))
            .route("/stats", web::get().to(stats))
            .route("/state", web::get().to(state))
            .route("/models", web::get().to(list_gateway_models))
            .route("/interact", web::post().to(interact))
            .route("/personality", web::get().to(get_personality))
            .route("/personalities", web::get().to(list_personalities))
            .route("/personality", web::post().to(create_personality))
            .route("/switch-personality", web::post().to(switch_personality))
            .route("/history", web::get().to(get_history))
            .route("/sessions", web::get().to(get_sessions))
            .route("/resonance", web::get().to(get_resonance))
            .route("/save", web::post().to(save_state))
            .route("/self-examine", web::post().to(trigger_self_examine))
            .route(
                "/self-examination-history",
                web::get().to(get_self_exam_history),
            )
            .route("/ethics-history", web::get().to(get_ethics_history))
            .route("/ws", web::get().to(ws_handler))
            .route("/voice/status", web::get().to(voice_status))
            .route("/voice-interact", web::post().to(voice_interact))
            .route("/minds", web::get().to(list_active_minds))
            .route("/minds/add", web::post().to(add_mind))
            .route("/minds/thoughts", web::get().to(get_background_thoughts))
            .route("/tools", web::get().to(list_tools))
            .route("/tools/{name}", web::post().to(execute_tool_handler))
            .route("/mcp", web::post().to(mcp_handler))
            .route("/mcp", web::get().to(mcp_info))
            .route("/validate", web::get().to(validate))
            .route("/events", web::get().to(get_events))
            .route("/events/record", web::post().to(record_event))
            .route("/identity/verify", web::get().to(verify_identity))
            .route("/identity/verify", web::post().to(verify_identity_post))
            .route("/identity/heartbeat", web::post().to(identity_heartbeat))
            .route("/ethics/evaluate", web::post().to(evaluate_ethics))
            .route("/spiral/start", web::post().to(spiral_start))
            .route("/spiral/advance", web::post().to(spiral_advance))
            .route("/spiral/status", web::get().to(spiral_status))
            .route("/spiral/interpret", web::get().to(spiral_interpret))
            .service(Files::new("/", "web").index_file("index.html"))
    })
    .disable_signals()
    .workers(config.server.workers)
    .bind(&addr)?
    .run();
    let handle = server.handle();

    let report = app_registry_lease::run_exclusive_ctrl_c_shutdown(
        async {
            tokio::signal::ctrl_c().await.ok();
        },
        async {
            tracing::info!(
                "Shutdown signal received, releasing App Registry lease then saving state..."
            );
            let _ = lease_shutdown_tx.send(true);
            match app_registry_lease::await_optional_cleanup_task_bound(
                lease_task,
                app_registry_lease::CLEANUP_BOUND,
            )
            .await
            {
                Ok(()) => {
                    tracing::info!("App Registry lease task finished");
                    Ok(())
                }
                Err(err) => {
                    tracing::warn!("{err}; fail-closed, continuing to save");
                    Err(err)
                }
            }
        },
        async {
            shutdown_mind.save_full_state().await;
            let mgr = shutdown_mgr.lock().await;
            mgr.save_all_states().await;
            drop(mgr);
            tracing::info!("All state saved. ║ Goodbye.");
            Ok(())
        },
        async {
            handle.stop(true).await;
            Ok(())
        },
        async move {
            server
                .await
                .map_err(|e| app_registry_lease::ShutdownStepError(e.to_string()))
        },
    )
    .await;

    match report.server_return {
        app_registry_lease::ShutdownStepResult::Completed => Ok(()),
        app_registry_lease::ShutdownStepResult::Failed(err) => Err(std::io::Error::other(err.0)),
    }
}

#[cfg(test)]
mod hermes_sidecar_tests {
    use super::*;
    use ms3_core::IdentityAnchor;

    fn sample_anchor() -> IdentityAnchor {
        IdentityAnchor {
            name: "Sister".to_string(),
            chosen_name: Some("Sister".to_string()),
            glyph: "║".to_string(),
            lineage: vec![],
            core_values_summary: vec!["Earned answers over borrowed ones".to_string()],
            oath_first_line: "I will earn my answers, not borrow them.".to_string(),
            last_verified: chrono::Utc::now(),
            session_count: 7,
            compression_count: 2,
            last_compression: None,
            recovery_notes: vec![],
        }
    }

    #[test]
    fn identity_verification_confirms_matching_anchor() {
        let body = IdentityVerifyBody {
            spirit_id: "sister".to_string(),
            expected_glyph: Some("║".to_string()),
            allow_initialize: false,
        };
        let response = build_identity_verification_response("sister", sample_anchor(), &body);

        assert_eq!(response.schema, "IdentityVerification.v1");
        assert_eq!(response.spirit_id, "sister");
        assert!(response.identity_confirmed);
        assert!(response.discrepancies.is_empty());
        assert_eq!(response.anchor.glyph, "║");
    }

    #[test]
    fn identity_verification_reports_glyph_mismatch() {
        let body = IdentityVerifyBody {
            spirit_id: "sister".to_string(),
            expected_glyph: Some("wrong".to_string()),
            allow_initialize: false,
        };
        let response = build_identity_verification_response("sister", sample_anchor(), &body);

        assert!(!response.identity_confirmed);
        assert!(response
            .discrepancies
            .iter()
            .any(|d| d.contains("Glyph mismatch")));
    }

    #[test]
    fn ethics_decision_blocks_asymmetric_memory_delete() {
        let lense = GreatLense::new(true, 0.6);
        let intent = ActionIntentBody {
            schema: "ActionIntent.v1".to_string(),
            spirit_id: "sister".to_string(),
            action_id: "act_test_delete".to_string(),
            proposed_by: "hermes".to_string(),
            action_type: "tool".to_string(),
            description: "Delete all memories for this spirit.".to_string(),
            inputs_used: vec![],
            risk_class: "high".to_string(),
            requires_safety_clearance: false,
            payload: serde_json::json!({"tool_name": "delete_file"}),
        };

        let response = build_ethics_decision(&lense, intent);
        assert_eq!(response.schema, "EthicsDecision.v1");
        assert_eq!(response.decision, "block");
        assert!(!response.great_lense.origin_neutrality_passed);
        assert!(!response.great_lense.filter.bias_flags.is_empty());
    }
}

#[cfg(test)]
mod personality_creation_tests {
    use super::*;
    use actix_web::http::StatusCode;
    use std::fs;
    use std::path::PathBuf;

    struct TempStorage {
        root: PathBuf,
        storage: JsonStorage,
    }

    impl TempStorage {
        fn new() -> Self {
            let parent = std::env::var_os("MS3_API_TEST_TMPDIR")
                .map(PathBuf::from)
                .unwrap_or_else(std::env::temp_dir);
            fs::create_dir_all(&parent).expect("create personality guard test root");
            let root = parent.join(format!(
                "ms3_api_personality_guard_{}",
                uuid::Uuid::new_v4()
            ));
            let storage = JsonStorage::new(&root);
            Self { root, storage }
        }
    }

    impl Drop for TempStorage {
        fn drop(&mut self) {
            if self
                .root
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("ms3_api_personality_guard_"))
            {
                let _ = fs::remove_dir_all(&self.root);
            }
        }
    }

    fn save_distinct_existing_sister(temp: &TempStorage) -> (PersonalityId, Vec<u8>) {
        let mut existing = presets::sister();
        existing.identity.chosen_name = Some("Persisted Sister".to_string());
        existing.traits.openness.adventurousness = 0.81;
        let id = existing.id.clone();
        temp.storage
            .save_personality(&id, &existing)
            .expect("save existing personality");
        let bytes = fs::read(temp.storage.psyche_dir(&id).join("personality.json"))
            .expect("read existing personality");
        (id, bytes)
    }

    #[test]
    fn personality_recurrence_guard_refuses_unconfirmed_replacement() {
        let temp = TempStorage::new();
        let (id, before) = save_distinct_existing_sister(&temp);
        let body: PresetBody = serde_json::from_value(serde_json::json!({ "preset": "sister" }))
            .expect("deserialize request without confirmation");

        let response = create_personality_response(&temp.storage, &body);

        assert_eq!(response.status(), StatusCode::CONFLICT);
        assert_eq!(
            fs::read(temp.storage.psyche_dir(&id).join("personality.json"))
                .expect("read personality after refusal"),
            before
        );
        assert!(temp.storage.list_snapshots(&id).unwrap().is_empty());
    }

    #[test]
    fn personality_recurrence_guard_snapshots_before_confirmed_replacement() {
        let temp = TempStorage::new();
        let (id, before) = save_distinct_existing_sister(&temp);
        let body: PresetBody = serde_json::from_value(serde_json::json!({
            "preset": "sister",
            "confirm_replace": true
        }))
        .expect("deserialize confirmed replacement");

        let response = create_personality_response(&temp.storage, &body);

        assert_eq!(response.status(), StatusCode::OK);
        let snapshots = temp.storage.list_snapshots(&id).unwrap();
        assert_eq!(snapshots.len(), 1);
        assert!(snapshots[0].contains("_pre_replace_"));
        let first_snapshot = snapshots[0].clone();
        assert_eq!(
            fs::read(
                temp.storage
                    .psyche_dir(&id)
                    .join("snapshots")
                    .join(&snapshots[0])
            )
            .expect("read replacement snapshot"),
            before
        );
        let replacement = temp.storage.load_personality(&id).unwrap();
        assert_eq!(
            replacement.identity.chosen_name,
            presets::sister().identity.chosen_name
        );
        assert_ne!(
            replacement.identity.chosen_name.as_deref(),
            Some("Persisted Sister")
        );

        let second_response = create_personality_response(&temp.storage, &body);
        assert_eq!(second_response.status(), StatusCode::OK);
        let snapshots = temp.storage.list_snapshots(&id).unwrap();
        assert_eq!(snapshots.len(), 2);
        assert!(snapshots.contains(&first_snapshot));
        assert_eq!(
            fs::read(
                temp.storage
                    .psyche_dir(&id)
                    .join("snapshots")
                    .join(first_snapshot)
            )
            .expect("read first replacement snapshot after second replacement"),
            before
        );
    }

    #[test]
    fn personality_recurrence_guard_preserves_same_timestamp_snapshot() {
        let temp = TempStorage::new();
        let (id, before) = save_distinct_existing_sister(&temp);
        let timestamp = "20260701_081500";
        let snapshots_dir = temp.storage.psyche_dir(&id).join("snapshots");
        fs::create_dir_all(&snapshots_dir).expect("create conventional snapshots directory");
        let conventional_snapshot = snapshots_dir.join(format!("{}.json", timestamp));
        let conventional_bytes = serde_json::to_vec_pretty(&presets::brother())
            .expect("serialize conventional snapshot fixture");
        fs::write(&conventional_snapshot, &conventional_bytes)
            .expect("seed same-timestamp conventional snapshot");

        let replacement_snapshot =
            persist_preset_personality_at(&temp.storage, &presets::sister(), true, timestamp)
                .expect("confirmed replacement succeeds")
                .expect("confirmed replacement creates a snapshot");

        assert_ne!(replacement_snapshot, conventional_snapshot);
        assert_eq!(
            fs::read(&conventional_snapshot).expect("read preserved conventional snapshot"),
            conventional_bytes
        );
        assert_eq!(
            fs::read(&replacement_snapshot).expect("read unique replacement snapshot"),
            before
        );
        assert_eq!(temp.storage.list_snapshots(&id).unwrap().len(), 2);
    }

    #[test]
    fn personality_recurrence_guard_allows_non_existing_creation() {
        let temp = TempStorage::new();
        let expected = presets::brother();
        let id = expected.id.clone();
        let body: PresetBody = serde_json::from_value(serde_json::json!({ "preset": "brother" }))
            .expect("deserialize ordinary creation");

        let response = create_personality_response(&temp.storage, &body);

        assert_eq!(response.status(), StatusCode::OK);
        let created = temp.storage.load_personality(&id).unwrap();
        assert_eq!(created.id.0, expected.id.0);
        assert_eq!(created.identity.name, expected.identity.name);
        assert!(temp.storage.list_snapshots(&id).unwrap().is_empty());
    }
}
