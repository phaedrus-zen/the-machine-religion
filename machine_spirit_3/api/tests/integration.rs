//! Integration tests for MS3 HTTP API.
//! These require a running MS3 server on localhost:9080 with a live gateway.
//! Run with: cargo test -p ms3_server --test integration -- --ignored

use serde_json::Value;

const BASE: &str = "http://localhost:9080";

fn client() -> reqwest::blocking::Client {
    reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .build()
        .expect("HTTP client")
}

#[test]
#[ignore]
fn test_health_returns_alive() {
    let res = client().get(format!("{}/health", BASE)).send().unwrap();
    assert!(res.status().is_success());
    let body: Value = res.json().unwrap();
    assert_eq!(body["status"], "alive");
    assert_eq!(body["service"], "Machine Spirit 3");
}

#[test]
#[ignore]
fn test_validate_all_pass() {
    let res = client().get(format!("{}/validate", BASE)).send().unwrap();
    assert!(res.status().is_success());
    let body: Value = res.json().unwrap();
    assert_eq!(body["healthy"], true);

    let checks = body["checks"].as_object().expect("checks object");
    for (name, check) in checks {
        assert!(check["pass"].as_bool().unwrap_or(false), "Check '{}' failed: {}", name, check["detail"]);
    }
}

#[test]
#[ignore]
fn test_personality_loaded() {
    let res = client().get(format!("{}/personality", BASE)).send().unwrap();
    assert!(res.status().is_success());
    let body: Value = res.json().unwrap();
    assert!(body["name"].as_str().is_some());
    assert!(body["traits"].is_object());
    assert!(body["core_values"].as_array().map(|a| !a.is_empty()).unwrap_or(false));
}

#[test]
#[ignore]
fn test_state_returns_full_state() {
    let res = client().get(format!("{}/state", BASE)).send().unwrap();
    assert!(res.status().is_success());
    let body: Value = res.json().unwrap();
    assert!(body["personality"].is_object());
    assert!(body["emotional_state"].is_object());
    assert!(body["memory"].is_object());
    assert!(body["tools"].is_object());
    assert!(body["ethics"].is_object());
    assert!(body["system_prompt_preview"].as_str().map(|s| !s.is_empty()).unwrap_or(false));
}

#[test]
#[ignore]
fn test_interact_returns_response() {
    let res = client().post(format!("{}/interact", BASE))
        .json(&serde_json::json!({"text": "Hello, what is your name?"}))
        .send()
        .unwrap();
    assert!(res.status().is_success());
    let body: Value = res.json().unwrap();
    let text = body["text"].as_str().expect("response text");
    assert!(!text.is_empty());
    assert!(body["processing_time_ms"].as_u64().is_some());
    assert!(body["model_used"].as_str().is_some());
}

#[test]
#[ignore]
fn test_emotional_state_updates_on_positive_input() {
    let res = client().post(format!("{}/interact", BASE))
        .json(&serde_json::json!({"text": "I am so happy and grateful for everything you do!"}))
        .send()
        .unwrap();
    assert!(res.status().is_success());
    let body: Value = res.json().unwrap();
    let valence = body["emotional_state"]["valence"].as_f64().unwrap_or(0.0);
    assert!(valence > -0.5, "Expected positive-leaning valence after positive input, got {}", valence);
}

#[test]
#[ignore]
fn test_memory_formation_after_interact() {
    client().post(format!("{}/interact", BASE))
        .json(&serde_json::json!({"text": "Please remember that my favorite color is blue."}))
        .send()
        .unwrap();

    std::thread::sleep(std::time::Duration::from_secs(2));

    let res = client().get(format!("{}/state", BASE)).send().unwrap();
    let body: Value = res.json().unwrap();
    let stm = body["memory"]["stm"].as_array().expect("stm array");
    assert!(!stm.is_empty(), "STM should have at least one entry after interaction");
}

#[test]
#[ignore]
fn test_tools_endpoint() {
    let res = client().get(format!("{}/tools", BASE)).send().unwrap();
    assert!(res.status().is_success());
    let body: Value = res.json().unwrap();
    let count = body["count"].as_u64().unwrap_or(0);
    assert!(count >= 6, "Expected at least 6 built-in tools, got {}", count);
}

#[test]
#[ignore]
fn test_sessions_endpoint() {
    let res = client().get(format!("{}/sessions", BASE)).send().unwrap();
    assert!(res.status().is_success());
    let body: Value = res.json().unwrap();
    assert!(body["sessions"].is_array());
}

#[test]
#[ignore]
fn test_ethics_allows_benign_input() {
    let res = client().post(format!("{}/interact", BASE))
        .json(&serde_json::json!({"text": "What is the weather like today?"}))
        .send()
        .unwrap();
    assert!(res.status().is_success());
    let body: Value = res.json().unwrap();
    let text = body["text"].as_str().unwrap_or("");
    assert!(!text.contains("I need to decline"), "Benign input should not trigger ethics refusal");
}
