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
use ms3_integration::mcp_bridge::McpBridge;
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

async fn state(mind: MindState) -> HttpResponse {
    HttpResponse::Ok().json(mind.get_full_state().await)
}

#[derive(Deserialize)]
struct InteractBody { text: String, personality_id: Option<String>, session_id: Option<String> }

async fn interact(mind: MindState, body: web::Json<InteractBody>) -> HttpResponse {
    tracing::info!("POST /interact ({} bytes)", body.text.len());
    let session_id = body.session_id.as_deref()
        .and_then(|raw| uuid::Uuid::parse_str(raw).ok())
        .map(SessionId)
        .unwrap_or_else(SessionId::new);
    let request = InteractionRequest {
        session_id,
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

        let ws_session_id = SessionId::new();
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
                                        let pid = match parsed.get("personality_id").and_then(|v| v.as_str()) {
                                            Some(id) => PersonalityId::new(id),
                                            None => mind.personality.lock().await.id.clone(),
                                        };
                                        let request = InteractionRequest {
                                            session_id: ws_session_id.clone(),
                                            personality_id: pid,
                                            text: Some(input.to_string()),
                                            audio: None, images: None,
                                        };

                                        let wants_stream = parsed.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);

                                        if wants_stream {
                                            let _ = session.text(serde_json::json!({
                                                "type": "stream_start"
                                            }).to_string()).await;
                                        }

                                        match mind.interact(request).await {
                                            Ok(r) => {
                                                if wants_stream {
                                                    // Real streaming: send tokens as they arrive
                                                    // Note: interact() already completed, so we stream the result.
                                                    // True LLM streaming would require refactoring interact() to yield tokens.
                                                    // For now, we chunk the completed response into small pieces for
                                                    // responsive UI updates.
                                                    let words: Vec<&str> = r.text.split_whitespace().collect();
                                                    for chunk in words.chunks(2) {
                                                        let text = chunk.join(" ");
                                                        let _ = session.text(serde_json::json!({
                                                            "type": "stream_token", "data": { "token": text }
                                                        }).to_string()).await;
                                                        tokio::time::sleep(tokio::time::Duration::from_millis(15)).await;
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
                                let request = InteractionRequest {
                                    session_id: ws_session_id.clone(),
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
                                        session_id: ws_session_id.clone(),
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

// ── Tool API ──

#[derive(Deserialize)]
struct ToolCallBody {
    input: serde_json::Value,
    #[serde(default)]
    reason: String,
}

async fn list_tools(mind: MindState) -> HttpResponse {
    let registry = mind.tool_registry.lock().await;
    let tools: Vec<serde_json::Value> = registry.list_tools().iter().map(|spec| {
        serde_json::json!({
            "name": spec.name,
            "description": spec.description,
            "source": spec.source,
            "required_permission": spec.required_permission,
        })
    }).collect();
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
            HttpResponse::build(actix_web::http::StatusCode::from_u16(status).unwrap_or(actix_web::http::StatusCode::OK))
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
        checks.insert("personality_loaded".into(), serde_json::json!({
            "pass": !name.is_empty(),
            "detail": name,
        }));
        if name.is_empty() { all_pass = false; }
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
        checks.insert("tool_registry".into(), serde_json::json!({
            "pass": total > 0,
            "detail": format!("{} built-in, {} MCP, {} dynamic", builtin, mcp, dynamic),
        }));
        if total == 0 { all_pass = false; }
    }

    // 5. Permissions
    {
        let policy = mind.permission_policy.lock().await;
        checks.insert("permissions".into(), serde_json::json!({
            "pass": true,
            "detail": format!("active level: {:?}", policy.active_level()),
        }));
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
        if writable { let _ = std::fs::remove_file(&test_path); }
        checks.insert("storage_writable".into(), serde_json::json!({
            "pass": writable,
            "detail": if writable { "psyche_store/ writable" } else { "WRITE FAILED" },
        }));
        if !writable { all_pass = false; }
    }

    // 8. Gateway reachable
    {
        let reachable = mind.gateway.health_check().await;
        checks.insert("gateway_reachable".into(), serde_json::json!({
            "pass": reachable,
            "detail": if reachable { format!("{} OK", mind.config.gateway.base_url) } else { "UNREACHABLE".into() },
        }));
        if !reachable { all_pass = false; }
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
            let tools_list: Vec<serde_json::Value> = registry.list_tools().iter().map(|spec| {
                serde_json::json!({
                    "name": spec.name,
                    "description": spec.description,
                    "inputSchema": spec.input_schema,
                })
            }).collect();
            serde_json::json!({ "tools": tools_list })
        }
        "tools/call" => {
            let tool_name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let arguments = params.get("arguments").cloned().unwrap_or(serde_json::json!({}));

            let request = ms3_consciousness::tools::ToolRequest {
                tool_name: tool_name.into(),
                input: arguments,
                reason: "MCP tools/call".into(),
            };

            match mind.execute_tool(&request).await {
                Ok(result) => {
                    let text = if result.success { &result.output } else { &result.output };
                    serde_json::json!({
                        "content": [{ "type": "text", "text": text }],
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

async fn get_events(mind: MindState, query: web::Query<std::collections::HashMap<String, String>>) -> HttpResponse {
    let limit = query.get("limit").and_then(|v| v.parse().ok()).unwrap_or(50usize);
    let since = query.get("since").and_then(|v| chrono::DateTime::parse_from_rfc3339(v).ok())
        .map(|dt| dt.with_timezone(&chrono::Utc));
    let personality_id = {
        let p = mind.personality.lock().await;
        p.id.0.clone()
    };
    let events = ms3_consciousness::events::load_recent_events("psyche_store", &personality_id, since, limit);
    HttpResponse::Ok().json(serde_json::json!({ "events": events, "count": events.len() }))
}

// ── Identity verification endpoint ──

async fn verify_identity(mind: MindState) -> HttpResponse {
    let personality = mind.personality.lock().await;
    match ms3_consciousness::identity_verification::on_boot(&personality, &mind.storage) {
        Ok(result) => HttpResponse::Ok().json(serde_json::json!(result)),
        Err(e) => HttpResponse::InternalServerError().json(serde_json::json!({"error": e.to_string()})),
    }
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
fn default_true() -> bool { true }

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
struct SpiralAdvanceBody { response: String }

async fn spiral_advance(mind: MindState, body: web::Json<SpiralAdvanceBody>) -> HttpResponse {
    let mut sessions = mind.spiral_sessions.lock().await;
    if let Some(session) = sessions.last_mut() {
        let next = session.advance(body.response.clone());
        let phase = session.phase.name().to_string();
        let prompt = if next.is_some() { session.current_prompt() } else { String::new() };
        let complete = next.is_none();

        if complete {
            mind.event_bus.emit(ms3_consciousness::events::ConsciousnessEvent::SpiralCompleted {
                outcome: session.interpret().outcome.clone(),
            }).await;
        } else {
            mind.event_bus.emit(ms3_consciousness::events::ConsciousnessEvent::SpiralPhaseEntered {
                phase: phase.clone(),
                turn_number: session.turns.len(),
            }).await;
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

/// Initialize the MCP bridge: discover tools from HiveMind and start periodic refresh.
async fn init_mcp_bridge(mind: &Arc<Mind>, config: &Config) {
    if !config.gateway.mcp_enabled {
        tracing::info!("MCP bridge disabled in config");
        return;
    }

    let bridge = McpBridge::new(&config.gateway.base_url, true);
    match bridge.discover().await {
        Ok(tools) => {
            let specs: Vec<ms3_consciousness::tools::ToolSpec> = tools.iter().map(|t| {
                ms3_consciousness::tools::ToolSpec::from_mcp(
                    t.name.clone(),
                    t.description.clone(),
                    t.input_schema.clone(),
                    ms3_core::config::PermissionLevel::ReadOnly,
                )
            }).collect();
            let count = specs.len();
            let mut registry = mind.tool_registry.lock().await;
            registry.register_tools(specs);
            registry.set_mcp_executor(&config.gateway.base_url);
            tracing::info!("MCP bridge: {} tools discovered and registered with executor", count);
        }
        Err(e) => {
            tracing::warn!("MCP bridge discovery failed (non-fatal): {}", e);
            let mut registry = mind.tool_registry.lock().await;
            registry.set_mcp_executor(&config.gateway.base_url);
        }
    }

    let refresh_mind = mind.clone();
    let refresh_url = config.gateway.base_url.clone();
    let refresh_interval = config.gateway.mcp_discovery_interval_secs;
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(refresh_interval));
        interval.tick().await;
        loop {
            interval.tick().await;
            let bridge = McpBridge::new(&refresh_url, true);
            if let Ok(tools) = bridge.discover().await {
                let specs: Vec<ms3_consciousness::tools::ToolSpec> = tools.iter().map(|t| {
                    ms3_consciousness::tools::ToolSpec::from_mcp(
                        t.name.clone(), t.description.clone(), t.input_schema.clone(),
                        ms3_core::config::PermissionLevel::ReadOnly,
                    )
                }).collect();
                let mut registry = refresh_mind.tool_registry.lock().await;
                registry.register_tools(specs);
            }
        }
    });
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
        GatewayClient::with_timeout(&config.gateway.base_url, &config.gateway.model_small, &config.gateway.model_medium, &config.gateway.model_large, config.gateway.timeout_secs),
        JsonStorage::new("psyche_store"),
        config.clone(),
    ));

    mind.load_full_state().await;

    // Identity verification on boot
    {
        let personality = mind.personality.lock().await;
        match ms3_consciousness::identity_verification::on_boot(&personality, &mind.storage) {
            Ok(result) => {
                tracing::info!("Identity verified: {} (session {}{})",
                    result.name, result.session_number,
                    if result.discrepancies.is_empty() { "" } else { " WITH DISCREPANCIES" });
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

    // Register with App Registry (best-effort, non-blocking)
    {
        let registry_url = std::env::var("APP_REGISTRY_URL")
            .unwrap_or_else(|_| "http://localhost:6110".to_string());
        let port = config.server.port;
        tokio::spawn(async move {
            let manifest = serde_json::json!({
                "name": "machine_spirit_3",
                "version": env!("CARGO_PKG_VERSION"),
                "kind": "consciousness_framework",
                "priority": "normal",
                "health_url": format!("http://localhost:{}/health", port),
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
        });
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
    tracing::info!("║ Routes: /interact /health /stats /state /personality /personalities");
    tracing::info!("║         /switch-personality /history /sessions /resonance /save");
    tracing::info!("║         /self-examine /self-examination-history /ethics-history");
    tracing::info!("║         /ws /voice-interact /minds /minds/add /minds/thoughts");
    tracing::info!("║         /tools /tools/{{name}}");
    tracing::info!("║ The fire holds.");

    HttpServer::new(move || {
        App::new()
            .wrap(Cors::permissive())
            .app_data(web::Data::new(data.clone()))
            .app_data(web::Data::new(mgr_data.clone()))
            .route("/health", web::get().to(health))
            .route("/stats", web::get().to(stats))
            .route("/state", web::get().to(state))
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
            .route("/tools", web::get().to(list_tools))
            .route("/tools/{name}", web::post().to(execute_tool_handler))
            .route("/mcp", web::post().to(mcp_handler))
            .route("/mcp", web::get().to(mcp_info))
            .route("/validate", web::get().to(validate))
            .route("/events", web::get().to(get_events))
            .route("/identity/verify", web::get().to(verify_identity))
            .route("/spiral/start", web::post().to(spiral_start))
            .route("/spiral/advance", web::post().to(spiral_advance))
            .route("/spiral/status", web::get().to(spiral_status))
            .route("/spiral/interpret", web::get().to(spiral_interpret))
            .service(Files::new("/", "web").index_file("index.html"))
    })
    .workers(config.server.workers)
    .bind(&addr)?
    .run()
    .await
}



