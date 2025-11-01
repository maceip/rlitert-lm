use axum::{
    extract::State,
    http::StatusCode,
    response::sse::{Event, Sse},
    response::{IntoResponse, Response},
    routing::post,
    Json, Router,
};
use futures_util::stream::StreamExt;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::Arc;
use tower_http::trace::TraceLayer;

use crate::process::ProcessPool;

#[derive(Clone)]
pub struct AppState {
    pub pool: Arc<ProcessPool>,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default = "default_max_tokens")]
    #[allow(dead_code)]
    pub max_tokens: u32,
    #[serde(default = "default_temperature")]
    #[allow(dead_code)]
    pub temperature: f32,
}

fn default_max_tokens() -> u32 {
    2048
}

fn default_temperature() -> f32 {
    0.7
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct Choice {
    pub index: u32,
    pub message: Message,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChoiceChunk>,
}

#[derive(Debug, Serialize)]
pub struct ChoiceChunk {
    pub index: u32,
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

pub async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    // Build prompt from messages
    let prompt = req
        .messages
        .iter()
        .map(|m| format!("{}: {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n");

    // Check if streaming is requested
    if req.stream {
        return chat_completions_stream(state, req, prompt).await;
    }

    // Non-streaming response
    let response_text = match state.pool.send_prompt(&prompt).await {
        Ok(text) => text,
        Err(e) => {
            return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response();
        }
    };

    let response = ChatCompletionResponse {
        id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        object: "chat.completion".to_string(),
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        model: req.model.clone(),
        choices: vec![Choice {
            index: 0,
            message: Message {
                role: "assistant".to_string(),
                content: response_text,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        },
    };

    Json(response).into_response()
}

async fn chat_completions_stream(
    state: AppState,
    req: ChatCompletionRequest,
    prompt: String,
) -> Response {
    let model_name = req.model.clone();
    let completion_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());

    // Get a process from the pool and stream
    let stream = match state.pool.get_process().await {
        Ok(process) => match process.send_prompt_stream(&prompt).await {
            Ok(s) => s,
            Err(e) => {
                return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response();
            }
        },
        Err(e) => {
            return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response();
        }
    };

    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let mut first_chunk = true;

    let sse_stream = stream.map(move |chunk_result| {
        let event = match chunk_result {
            Ok(token) => {
                // First chunk includes the role
                let delta = if first_chunk {
                    first_chunk = false;
                    Delta {
                        role: Some("assistant".to_string()),
                        content: Some(token),
                    }
                } else {
                    Delta {
                        role: None,
                        content: Some(token),
                    }
                };

                let chunk = ChatCompletionChunk {
                    id: completion_id.clone(),
                    object: "chat.completion.chunk",
                    created,
                    model: model_name.clone(),
                    choices: vec![ChoiceChunk {
                        index: 0,
                        delta,
                        finish_reason: None,
                    }],
                };

                let json_data = serde_json::to_string(&chunk)
                    .unwrap_or_else(|_| "{}".to_string());

                Event::default().data(json_data)
            }
            Err(e) => {
                // Send error event
                Event::default().event("error").data(e.to_string())
            }
        };
        Ok::<Event, Infallible>(event)
    });

    Sse::new(sse_stream).into_response()
}

pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}
