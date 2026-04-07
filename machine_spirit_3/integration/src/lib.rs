pub mod mcp_bridge;

/// Detect audio format from magic bytes.
fn detect_audio_format(data: &[u8]) -> (&'static str, &'static str) {
    if data.len() >= 4 {
        if data[0] == 0x1A && data[1] == 0x45 && data[2] == 0xDF && data[3] == 0xA3 {
            return ("audio.webm", "audio/webm");
        }
        if &data[0..4] == b"RIFF" {
            return ("audio.wav", "audio/wav");
        }
        if &data[0..3] == b"ID3" || (data[0] == 0xFF && (data[1] & 0xE0) == 0xE0) {
            return ("audio.mp3", "audio/mpeg");
        }
        if &data[0..4] == b"OggS" {
            return ("audio.ogg", "audio/ogg");
        }
        if &data[0..4] == b"fLaC" {
            return ("audio.flac", "audio/flac");
        }
    }
    ("audio.mp3", "audio/mpeg")
}

use futures::StreamExt;
use ms3_core::{ModelTier, Ms3Error, Ms3Result};
use reqwest::multipart;
use serde::{Deserialize, Serialize};

pub struct GatewayClient {
    client: reqwest::Client,
    base_url: String,
    model_small: String,
    model_medium: String,
    model_large: String,
}

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    stream: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatMessage,
}

#[derive(Debug, Deserialize)]
struct StreamChunk {
    choices: Option<Vec<StreamChoice>>,
}

#[derive(Debug, Deserialize)]
struct StreamChoice {
    delta: StreamDelta,
}

#[derive(Debug, Deserialize)]
struct StreamDelta {
    content: Option<String>,
}

impl GatewayClient {
    pub fn new(base_url: &str, model_small: &str, model_medium: &str, model_large: &str) -> Self {
        Self::with_timeout(base_url, model_small, model_medium, model_large, 30)
    }

    pub fn with_timeout(base_url: &str, model_small: &str, model_medium: &str, model_large: &str, timeout_secs: u64) -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(timeout_secs))
                .build()
                .unwrap_or_default(),
            base_url: base_url.trim_end_matches('/').to_string(),
            model_small: model_small.to_string(),
            model_medium: model_medium.to_string(),
            model_large: model_large.to_string(),
        }
    }

    fn model_name(&self, tier: ModelTier) -> &str {
        match tier {
            ModelTier::Small => &self.model_small,
            ModelTier::Medium => &self.model_medium,
            ModelTier::Large => &self.model_large,
            ModelTier::Auto => &self.model_medium,
        }
    }

    pub async fn chat(&self, messages: Vec<ChatMessage>, tier: ModelTier, max_tokens: Option<u32>) -> Ms3Result<String> {
        let request = ChatRequest {
            model: self.model_name(tier).to_string(),
            messages,
            max_tokens,
            temperature: Some(0.7),
            stream: Some(false),
        };

        let response = self.client
            .post(format!("{}/v1/chat/completions", self.base_url))
            .json(&request)
            .send()
            .await
            .map_err(|e| Ms3Error::Gateway(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(Ms3Error::Gateway(format!("HTTP {}: {}", status, body)));
        }

        let chat_response: ChatResponse = response
            .json()
            .await
            .map_err(|e| Ms3Error::Gateway(e.to_string()))?;

        chat_response.choices.first()
            .map(|c| c.message.content.clone())
            .ok_or_else(|| Ms3Error::Gateway("Empty response from gateway".into()))
    }

    /// Streams chat completion tokens via SSE. Returns a receiver that yields content tokens as they arrive.
    pub async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tier: ModelTier,
        max_tokens: Option<u32>,
    ) -> Ms3Result<tokio::sync::mpsc::Receiver<String>> {
        let request = ChatRequest {
            model: self.model_name(tier).to_string(),
            messages,
            max_tokens,
            temperature: Some(0.7),
            stream: Some(true),
        };

        let response = self.client
            .post(format!("{}/v1/chat/completions", self.base_url))
            .json(&request)
            .send()
            .await
            .map_err(|e| Ms3Error::Gateway(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(Ms3Error::Gateway(format!("HTTP {}: {}", status, body)));
        }

        let (tx, rx) = tokio::sync::mpsc::channel(100);
        let mut stream = response.bytes_stream();
        let mut buffer = Vec::new();

        tokio::spawn(async move {
            'stream: while let Some(chunk_result) = stream.next().await {
                let chunk = match chunk_result {
                    Ok(b) => b,
                    Err(e) => {
                        tracing::error!("Stream error: {}", e);
                        break;
                    }
                };
                buffer.extend_from_slice(&chunk);

                // Process complete lines (SSE: "data: {...}\n" or "data: [DONE]\n")
                while let Some(newline_pos) = buffer.iter().position(|&b| b == b'\n') {
                    let line = buffer.drain(..=newline_pos).collect::<Vec<_>>();
                    let line_str = String::from_utf8_lossy(&line).trim().to_string();
                    drop(line);

                    if line_str.starts_with("data: ") {
                        let payload = line_str.trim_start_matches("data: ").trim();
                        if payload == "[DONE]" {
                            break 'stream;
                        }
                        if let Ok(chunk) = serde_json::from_str::<StreamChunk>(payload) {
                            if let Some(choices) = chunk.choices {
                                if let Some(choice) = choices.first() {
                                    if let Some(content) = choice.delta.content.as_ref() {
                                        if !content.is_empty() && tx.send(content.clone()).await.is_err() {
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });

        Ok(rx)
    }

    pub async fn health_check(&self) -> bool {
        self.client
            .get(format!("{}/health", self.base_url))
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }

    /// Transcribes audio to text via POST to /v1/audio/transcriptions.
    /// Sends the audio data as multipart form data.
    /// Always sends as WAV because HiveMind's local ASR path requires WAV format.
    /// If the input is already WAV, sends as-is. Otherwise, sends with WAV MIME
    /// and trusts the gateway's cloud fallback for non-WAV formats.
    pub async fn transcribe_audio(&self, audio_data: Vec<u8>) -> Ms3Result<String> {
        let (detected_name, _detected_mime) = detect_audio_format(&audio_data);
        let is_wav = detected_name == "audio.wav";

        // HiveMind's local ASR requires WAV. For non-WAV formats,
        // we still send the original bytes -- the gateway will fall through
        // to cloud Whisper which accepts all formats.
        let (filename, mime) = if is_wav {
            ("audio.wav", "audio/wav")
        } else {
            // Send with original format info so the gateway can route correctly
            (detected_name, _detected_mime)
        };

        let part = multipart::Part::bytes(audio_data)
            .file_name(filename.to_string())
            .mime_str(mime)
            .map_err(|e| Ms3Error::Gateway(e.to_string()))?;

        let form = multipart::Form::new().part("file", part);

        let response = self.client
            .post(format!("{}/v1/audio/transcriptions", self.base_url))
            .multipart(form)
            .send()
            .await
            .map_err(|e| Ms3Error::Gateway(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(Ms3Error::Gateway(format!("HTTP {}: {}", status, body)));
        }

        // OpenAI-style transcription returns JSON with "text" field
        let json: serde_json::Value = response
            .json()
            .await
            .map_err(|e| Ms3Error::Gateway(e.to_string()))?;

        json.get("text")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| Ms3Error::Gateway("No text in transcription response".into()))
    }

    /// Synthesizes speech from text via POST to /v1/audio/speech.
    /// Model name comes from the tts_model parameter (configurable, defaults to "tts").
    pub async fn synthesize_speech(&self, text: &str, voice: Option<&str>) -> Ms3Result<Vec<u8>> {
        self.synthesize_speech_with_model(text, voice, "tts-1").await
    }

    pub async fn synthesize_speech_with_model(&self, text: &str, voice: Option<&str>, model: &str) -> Ms3Result<Vec<u8>> {
        let request = serde_json::json!({
            "model": model,
            "input": text,
            "voice": voice.unwrap_or("alloy"),
            "response_format": "wav",
        });

        let response = self.client
            .post(format!("{}/v1/audio/speech", self.base_url))
            .json(&request)
            .send()
            .await
            .map_err(|e| Ms3Error::Gateway(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(Ms3Error::Gateway(format!("HTTP {}: {}", status, body)));
        }

        let bytes = response
            .bytes()
            .await
            .map_err(|e| Ms3Error::Gateway(e.to_string()))?;

        Ok(bytes.to_vec())
    }

    /// Convenience wrapper that prepends a system message and appends the user message
    /// to the history, then calls chat.
    pub async fn chat_with_history(
        &self,
        system: &str,
        history: Vec<ChatMessage>,
        user_input: &str,
        tier: ModelTier,
        max_tokens: Option<u32>,
    ) -> Ms3Result<String> {
        let mut messages = Vec::with_capacity(history.len() + 2);
        messages.push(ChatMessage {
            role: "system".to_string(),
            content: system.to_string(),
        });
        messages.extend(history);
        messages.push(ChatMessage {
            role: "user".to_string(),
            content: user_input.to_string(),
        });
        self.chat(messages, tier, max_tokens).await
    }

    /// Generate a vector embedding for the given text via /v1/embeddings.
    /// Returns None if the endpoint is unavailable (graceful degradation).
    pub async fn embed(&self, text: &str) -> Option<Vec<f32>> {
        self.embed_with_model(text, "nomic-embed-text").await
    }

    pub async fn embed_with_model(&self, text: &str, model: &str) -> Option<Vec<f32>> {
        let body = serde_json::json!({
            "model": model,
            "input": text,
        });

        let response = self.client
            .post(format!("{}/v1/embeddings", self.base_url))
            .json(&body)
            .send()
            .await
            .ok()?;

        if !response.status().is_success() {
            return None;
        }

        let json: serde_json::Value = response.json().await.ok()?;
        json.get("data")
            .and_then(|d| d.as_array())
            .and_then(|arr| arr.first())
            .and_then(|item| item.get("embedding"))
            .and_then(|emb| emb.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect())
    }
}
