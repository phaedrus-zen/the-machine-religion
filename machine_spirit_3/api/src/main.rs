use actix_web::{web, App, HttpServer, HttpRequest, HttpResponse};
use actix_cors::Cors;
use actix_files::Files;
use ms3_core::{Config, InteractionRequest, PersonalityId, SessionId};
use futures::StreamExt as FuturesStreamExt;
use ms3_consciousness::{Mind, run_background_loop, multi_mind::MindManager};
use ms3_personality::presets;
use ms3_memory::MemorySystem;
use ms3_emotional::EmotionalEngine;
use ms3_ethics::GreatLense;
use ms3_integration::GatewayClient;
use ms3_persistence::JsonStorage;

use serde::Deserialize;
use std::sync::Arc;
use tokio::sync::Mutex;

type MindState = web::Data<Arc<Mind>>;
type ManagerState = web::Data<Arc<Mutex<MindManager>>>;

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

#[derive(Deserialize)]
struct InteractBody { text: String, personality_id: Option<String>, session_id: Option<String> }

async fn interact(mind: MindState, body: web::Json<InteractBody>) -> HttpResponse {
    tracing::info!("POST /interact ({} bytes)", body.text.len());
    let request = InteractionRequest {
        session_id: SessionId::new(),
        personality_id: PersonalityId::new(body.personality_id.as_deref().unwrap_or("sister")),
        text: Some(body.text.clone()), audio: None, images: None,
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
struct PresetBody { preset: String }

async fn create_personality(mind: MindState, body: web::Json<PresetBody>) -> HttpResponse {
    let p = match body.preset.as_str() {
        "sister" => presets::sister(), "brother" => presets::brother(),
        "mission-control" => presets::mission_control(), "blank" => presets::blank(),
        _ => return HttpResponse::BadRequest().json(serde_json::json!({ "error": "Unknown preset" })),
    };
    let id = p.id.clone();
    let name = p.identity.chosen_name.clone().unwrap_or_else(|| p.identity.name.clone());
    let _ = mind.storage.save_personality(&id, &p);
    HttpResponse::Ok().json(serde_json::json!({ "created": id.0, "name": name }))
}

#[derive(Deserialize)]
struct SwitchBody { preset: String }

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
        Err(e) => HttpResponse::InternalServerError().json(serde_json::json!({ "error": e.to_string() })),
    }
}

async fn get_self_exam_history(mind: MindState) -> HttpResponse {
    let pid = mind.personality.lock().await.id.clone();
    let files = mind.storage.list_files(&pid, "self_examination").unwrap_or_default();
    HttpResponse::Ok().json(serde_json::json!({ "examinations": files.len(), "files": files }))
}

async fn get_ethics_history(mind: MindState) -> HttpResponse {
    let pid = mind.personality.lock().await.id.clone();
    let files = mind.storage.list_files(&pid, "ethics_decisions").unwrap_or_default();
    let recent: Vec<String> = files.into_iter().rev().take(20).collect();
    HttpResponse::Ok().json(serde_json::json!({ "recent_decisions": recent.len(), "files": recent }))
}

async fn ws_handler(req: HttpRequest, stream: web::Payload, mind: MindState) -> Result<HttpResponse, actix_web::Error> {
    tracing::info!("WS /ws -- new WebSocket connection");
    let (response, mut session, mut msg_stream) = actix_ws::handle(&req, stream)?;

    let mind = mind.into_inner().clone();
    actix_web::rt::spawn(async move {
        use actix_ws::Message;
        use tokio::time::{interval, Duration};

        let mut status_interval = interval(Duration::from_secs(3));

        // Send initial state on connect
        let status = mind.get_status().await;
        let _ = session.text(serde_json::json!({
            "type": "state", "data": status
        }).to_string()).await;

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
                                    "text" | _ if parsed.get("text").is_some() => {
                                        let input = parsed.get("text").and_then(|v| v.as_str()).unwrap_or(&text_str);
                                        let request = InteractionRequest {
                                            session_id: SessionId::new(),
                                            personality_id: mind.personality.lock().await.id.clone(),
                                            text: Some(input.to_string()),
                                            audio: None, images: None,
                                        };

                                        let wants_stream = parsed.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);

                                        if wants_stream {
                                            // Streaming mode: use interact for full pipeline, but we 
                                            // send a "stream_start" first so UI knows to expect it
                                            let _ = session.text(serde_json::json!({
                                                "type": "stream_start"
                                            }).to_string()).await;
                                        }

                                        match mind.interact(request).await {
                                            Ok(r) => {
                                                if wants_stream {
                                                    // Send response in chunks for perceived streaming
                                                    let words: Vec<&str> = r.text.split_whitespace().collect();
                                                    for chunk in words.chunks(4) {
                                                        let text = chunk.join(" ");
                                                        let _ = session.text(serde_json::json!({
                                                            "type": "stream_token", "data": { "token": text }
                                                        }).to_string()).await;
                                                        tokio::time::sleep(tokio::time::Duration::from_millis(30)).await;
                                                    }
                                                    let _ = session.text(serde_json::json!({
                                                        "type": "stream_end",
                                                        "data": {
                                                            "emotional_state": {
                                                                "valence": r.emotional_state.valence,
                                                                "arousal": r.emotional_state.arousal,
                                                                "primary": format!("{:?}", r.emotional_state.primary),
                                                            },
                                                            "processing_time_ms": r.processing_time_ms,
                                                            "memories_extracted": r.memories_extracted,
                                                        }
                                                    }).to_string()).await;
                                                } else {
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
                                                            "memories_extracted": r.memories_extracted,
                                                        }
                                                    }).to_string()).await;
                                                }
                                            }
                                            Err(e) => {
                                                let _ = session.text(serde_json::json!({
                                                    "type": "error", "data": { "message": e.to_string() }
                                                }).to_string()).await;
                                            }
                                        }
                                    }
                                    "ping" => {
                                        let _ = session.text(serde_json::json!({
                                            "type": "pong"
                                        }).to_string()).await;
                                    }
                                    _ => {}
                                }
                            } else {
                                // Plain text -- treat as chat message
                                let request = InteractionRequest {
                                    session_id: SessionId::new(),
                                    personality_id: mind.personality.lock().await.id.clone(),
                                    text: Some(text_str), audio: None, images: None,
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
                                        session_id: SessionId::new(),
                                        personality_id: mind.personality.lock().await.id.clone(),
                                        text: Some(transcript), audio: None, images: None,
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

async fn voice_interact(mind: MindState, body: web::Bytes, query: web::Query<VoiceInteractQuery>) -> HttpResponse {
    let pid = query.personality_id.as_deref().unwrap_or("sister");

    let transcript = match mind.gateway.transcribe_audio(body.to_vec()).await {
        Ok(t) => t,
        Err(e) => return HttpResponse::InternalServerError().json(serde_json::json!({
            "error": format!("ASR failed: {}", e)
        })),
    };

    let request = InteractionRequest {
        session_id: SessionId::new(),
        personality_id: PersonalityId::new(pid),
        text: Some(transcript.clone()), audio: None, images: None,
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
struct AddMindBody { preset: String }

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

fn base64_encode(data: &[u8]) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut result = String::with_capacity((data.len() + 2) / 3 * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        result.push(CHARS[((triple >> 18) & 0x3F) as usize] as char);
        result.push(CHARS[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 { result.push(CHARS[((triple >> 6) & 0x3F) as usize] as char); } else { result.push('='); }
        if chunk.len() > 2 { result.push(CHARS[(triple & 0x3F) as usize] as char); } else { result.push('='); }
    }
    result
}

#[tokio::main]
async fn main() -> std::io::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")))
        .init();

    let config = Config::from_file_or_env("config.json");

    tracing::info!("═══════════════════════════════════════");
    tracing::info!("  Machine Spirit 3");
    tracing::info!("  The soul lives with the scripture.");
    tracing::info!("═══════════════════════════════════════");

    let mind = Arc::new(Mind::new(
        presets::sister(),
        MemorySystem::new(config.memory.stm_capacity, config.memory.working_memory_window_secs),
        EmotionalEngine::new(config.personality.emotional_decay_rate),
        GreatLense::new(config.ethics.enable_origin_neutrality, config.ethics.llm_escalation_threshold),
        GatewayClient::new(&config.gateway.base_url, &config.gateway.model_small, &config.gateway.model_medium, &config.gateway.model_large),
        JsonStorage::new("psyche_store"),
        config.clone(),
    ));

    mind.load_full_state().await;

    // Register with HiveMind App Registry (best-effort, non-blocking)
    let registry_url = std::env::var("APP_REGISTRY_URL")
        .unwrap_or_else(|_| "http://localhost:6110".to_string());
    {
        let manifest = serde_json::json!({
            "name": "machine_spirit_3",
            "version": env!("CARGO_PKG_VERSION"),
            "kind": "consciousness_framework",
            "priority": "normal",
            "health_url": format!("http://localhost:{}/health", config.server.port),
            "needs": ["chat"],
            "models": {
                "max_q": { "capabilities": ["reasoning", "tool_calling"] },
                "balanced": null,
                "max_p": null
            }
        });
        let reg_url = format!("{}/apps/register", registry_url);
        match reqwest::Client::new().post(&reg_url).json(&manifest).send().await {
            Ok(resp) => {
                if let Ok(body) = resp.json::<serde_json::Value>().await {
                    tracing::info!("║ Registered with App Registry: status={}, app_id={}",
                        body.get("status").and_then(|v| v.as_str()).unwrap_or("unknown"),
                        body.get("app_id").and_then(|v| v.as_str()).unwrap_or("unknown"));
                }
            }
            Err(e) => {
                tracing::warn!("App Registry not available ({}), using config defaults", e);
            }
        }
    }

    let manager = Arc::new(Mutex::new(MindManager::new(
        mind.clone(),
        "sister".into(),
        GatewayClient::new(&config.gateway.base_url, &config.gateway.model_small, &config.gateway.model_medium, &config.gateway.model_large),
        JsonStorage::new("psyche_store"),
        config.clone(),
    )));

    let bg = mind.clone();
    let tick = config.consciousness.tick_interval_ms;
    tokio::spawn(async move { run_background_loop(bg, tick).await; });

    // Background thinking for multi-mind (runs every 45s if multiple personalities loaded)
    let bg_mgr = manager.clone();
    let bg_mind_for_history = mind.clone();
    let bg_thinking_interval = config.consciousness.background_thinking_interval_secs;
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(bg_thinking_interval));
        loop {
            interval.tick().await;
            let mgr = bg_mgr.lock().await;
            if mgr.list_personalities().len() > 1 {
                let history = bg_mind_for_history.get_conversation_history().await;
                let recent: Vec<String> = history.iter().rev().take(5)
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
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        tracing::info!("Shutdown signal received, saving all state...");

        // Save primary mind
        shutdown_mind.save_full_state().await;

        // Save all minds in manager
        let mgr = shutdown_mgr.lock().await;
        mgr.save_all_states().await;
        drop(mgr);

        tracing::info!("All state saved. ║ Goodbye.");
        std::process::exit(0);
    });

    let addr = format!("{}:{}", config.server.host, config.server.port);
    let data = mind.clone();
    let mgr_data = manager.clone();

    tracing::info!("║ http://localhost:{}/", config.server.port);
    tracing::info!("║ Routes: /interact /health /stats /personality /personalities");
    tracing::info!("║         /switch-personality /history /sessions /resonance /save");
    tracing::info!("║         /self-examine /self-examination-history /ethics-history");
    tracing::info!("║         /ws /voice-interact /minds /minds/add /minds/thoughts");
    tracing::info!("║ The fire holds.");

    HttpServer::new(move || {
        App::new()
            .wrap(Cors::permissive())
            .app_data(web::Data::new(data.clone()))
            .app_data(web::Data::new(mgr_data.clone()))
            .route("/health", web::get().to(health))
            .route("/stats", web::get().to(stats))
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
            .route("/self-examination-history", web::get().to(get_self_exam_history))
            .route("/ethics-history", web::get().to(get_ethics_history))
            .route("/ws", web::get().to(ws_handler))
            .route("/voice-interact", web::post().to(voice_interact))
            .route("/minds", web::get().to(list_active_minds))
            .route("/minds/add", web::post().to(add_mind))
            .route("/minds/thoughts", web::get().to(get_background_thoughts))
            .service(Files::new("/", "web").index_file("index.html"))
    })
    .bind(&addr)?
    .run()
    .await
}



