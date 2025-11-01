use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::sse::{Event, Sse},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use futures_util::stream::StreamExt;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::Arc;
use tower_http::trace::TraceLayer;

use crate::process::ProcessPool;

use crate::manager::LitManager;

#[derive(Clone)]
pub struct AppState {
    pub pool: Arc<ProcessPool>,
    pub manager: Arc<LitManager>,
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
#[serde(untagged)]
pub enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: serde_json::Value },
}

#[derive(Debug, Serialize, Clone)]
pub struct Message {
    pub role: String,
    #[serde(serialize_with = "serialize_content")]
    pub content: MessageContent,
}

#[derive(Debug, Clone)]
pub enum MessageContent {
    String(String),
    Parts(Vec<ContentPart>),
}

fn serialize_content<S>(content: &MessageContent, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    match content {
        MessageContent::String(s) => serializer.serialize_str(s),
        MessageContent::Parts(parts) => parts.serialize(serializer),
    }
}

impl<'de> Deserialize<'de> for Message {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct MessageHelper {
            role: String,
            content: serde_json::Value,
        }

        let helper = MessageHelper::deserialize(deserializer)?;
        let content = match helper.content {
            serde_json::Value::String(s) => MessageContent::String(s),
            serde_json::Value::Array(arr) => {
                let parts: Vec<ContentPart> = serde_json::from_value(serde_json::Value::Array(arr))
                    .map_err(serde::de::Error::custom)?;
                MessageContent::Parts(parts)
            }
            _ => return Err(serde::de::Error::custom("content must be string or array")),
        };

        Ok(Message {
            role: helper.role,
            content,
        })
    }
}

impl Message {
    pub fn content_as_string(&self) -> String {
        match &self.content {
            MessageContent::String(s) => s.clone(),
            MessageContent::Parts(parts) => {
                parts
                    .iter()
                    .filter_map(|part| match part {
                        ContentPart::Text { text } => Some(text.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            }
        }
    }
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
        .map(|m| format!("{}: {}", m.role, m.content_as_string()))
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
                content: MessageContent::String(response_text),
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

// Models endpoint structures
#[derive(Debug, Serialize)]
pub struct ModelObject {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
}

#[derive(Debug, Serialize)]
pub struct ModelsListResponse {
    pub object: &'static str,
    pub data: Vec<ModelObject>,
}

// List all locally downloaded models
pub async fn list_models(State(state): State<AppState>) -> Response {
    // Get list of locally downloaded models
    let models_output = match state.manager.list_models(false).await {
        Ok(output) => output,
        Err(e) => {
            return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response();
        }
    };

    // Parse the output to extract model names
    let model_names: Vec<String> = models_output
        .lines()
        .filter(|line| {
            let trimmed = line.trim();
            !trimmed.is_empty()
                && !trimmed.starts_with("Available")
                && !trimmed.starts_with("Downloaded")
                && !trimmed.starts_with("ALIAS")
        })
        .filter_map(|line| line.split_whitespace().next())
        .map(|s| s.to_string())
        .collect();

    // Create model objects
    let models: Vec<ModelObject> = model_names
        .into_iter()
        .map(|id| ModelObject {
            id,
            object: "model",
            created: 1700000000, // Static timestamp
            owned_by: "litert-lm",
        })
        .collect();

    let response = ModelsListResponse {
        object: "list",
        data: models,
    };

    Json(response).into_response()
}

// Get a specific model by ID
pub async fn get_model(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
) -> Response {
    // Get list of locally downloaded models
    let models_output = match state.manager.list_models(false).await {
        Ok(output) => output,
        Err(e) => {
            return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response();
        }
    };

    // Check if the requested model exists
    let model_exists = models_output
        .lines()
        .filter(|line| {
            let trimmed = line.trim();
            !trimmed.is_empty()
                && !trimmed.starts_with("Available")
                && !trimmed.starts_with("Downloaded")
                && !trimmed.starts_with("ALIAS")
        })
        .any(|line| line.trim() == model_id);

    if !model_exists {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": {
                    "message": format!("Model '{}' not found", model_id),
                    "type": "invalid_request_error",
                    "code": "model_not_found"
                }
            })),
        )
            .into_response();
    }

    let model = ModelObject {
        id: model_id,
        object: "model",
        created: 1700000000,
        owned_by: "litert-lm",
    };

    Json(model).into_response()
}

pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        .route("/v1/models/:model", get(get_model))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}
