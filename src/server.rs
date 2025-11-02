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

/// Check if this is a DSpy-rs formatted prompt by looking for multiple specific patterns
fn is_dspy_request(prompt: &str) -> bool {
    // DSpy-rs has very specific patterns - we need at least 3 of these to be confident:
    // 1. "Your input fields are:" or "Your output fields are:"
    // 2. Field markers like "[[ ## field_name ## ]]"
    // 3. "All interactions will be structured"
    // 4. "Given the fields" instruction pattern

    let has_field_declaration = prompt.contains("Your input fields are:")
        || prompt.contains("Your output fields are:");
    let has_field_markers = prompt.contains("[[ ## ") && prompt.contains(" ## ]]");
    let has_structure_instruction = prompt.contains("All interactions will be structured");
    let has_completion_marker = prompt.contains("[[ ## completed ## ]]")
        || prompt.contains("ending with the marker for `completed`");

    // Require at least 3 of these patterns to be present
    let pattern_count = [
        has_field_declaration,
        has_field_markers,
        has_structure_instruction,
        has_completion_marker,
    ].iter().filter(|&&x| x).count();

    pattern_count >= 3
}

/// Extract output field names from DSpy-rs formatted prompt
fn extract_dspy_output_fields(prompt: &str) -> Vec<String> {
    let mut fields = Vec::new();

    // Look for "Your output fields are:" section
    if let Some(output_section) = prompt.split("Your output fields are:").nth(1) {
        // Extract field names from lines like "1. `field_name` (String)"
        for line in output_section.lines() {
            if let Some(field_start) = line.find('`') {
                if let Some(field_end) = line[field_start + 1..].find('`') {
                    let field_name = &line[field_start + 1..field_start + 1 + field_end];
                    fields.push(field_name.to_string());
                }
            }
            // Stop at the next section
            if line.contains("All interactions will be structured") {
                break;
            }
        }
    }

    fields
}

/// Extract the actual user question from DSpy-rs formatted prompt
fn extract_dspy_question(prompt: &str) -> Option<String> {
    // Find the user's actual question after the format template
    // Look for pattern: user: [[ ## <field> ## ]]\n<actual_question>
    if let Some(user_section) = prompt.split("user: [[ ## ").nth(1) {
        if let Some(question_start) = user_section.find("## ]]\n") {
            let question = &user_section[question_start + 6..];
            return Some(question.trim().to_string());
        }
    }
    None
}

/// Format LLM response with DSpy-rs field markers
fn format_dspy_response(llm_output: &str, output_fields: &[String]) -> String {
    let cleaned_output = llm_output.trim();

    // For now, put the entire response in the first output field
    // This is a simple heuristic - could be improved with better parsing
    let mut formatted = String::new();

    if let Some(first_field) = output_fields.first() {
        formatted.push_str(&format!("[[ ## {} ## ]]\n", first_field));
        formatted.push_str(cleaned_output);
        formatted.push_str("\n\n");
    }

    // Add completion marker
    formatted.push_str("[[ ## completed ## ]]\n");

    formatted
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
    tracing::info!(
        model = %req.model,
        message_count = req.messages.len(),
        stream = req.stream,
        "Received chat completion request"
    );

    // Build prompt from messages
    let mut prompt = req
        .messages
        .iter()
        .map(|m| format!("{}: {}", m.role, m.content_as_string()))
        .collect::<Vec<_>>()
        .join("\n");

    tracing::debug!(
        model = %req.model,
        prompt_length = prompt.len(),
        "Built prompt from messages"
    );
    tracing::trace!(prompt = %prompt, "Full prompt text");

    // Check if streaming is requested
    if req.stream {
        tracing::debug!("Routing to streaming handler");
        return chat_completions_stream(state, req, prompt).await;
    }

    // Detect if this is a DSpy-rs structured output request
    let is_dspy = is_dspy_request(&prompt);
    let output_fields = if is_dspy {
        tracing::debug!("Detected DSpy-rs structured output request");
        // Extract output field names from the system message
        let fields = extract_dspy_output_fields(&prompt);
        tracing::debug!(fields = ?fields, "Extracted DSpy-rs output fields");

        // For small models, simplify by extracting just the actual question
        if let Some(question) = extract_dspy_question(&prompt) {
            tracing::debug!(original_length = prompt.len(), simplified_length = question.len(), "Simplified DSpy prompt for small model");
            prompt = question;
            tracing::trace!(simplified_prompt = %prompt, "Using simplified question");
        } else {
            tracing::warn!("Failed to extract question from DSpy prompt, using original");
        }

        fields
    } else {
        vec![]
    };

    // Non-streaming response
    tracing::debug!("Sending prompt to process pool");
    let mut response_text = match state.pool.send_prompt(&prompt).await {
        Ok(text) => {
            tracing::info!(
                response_length = text.len(),
                "Received completion from LLM"
            );
            tracing::trace!(response = %text, "LLM response text");
            text
        }
        Err(e) => {
            tracing::error!(error = %e, "Failed to get completion from process pool");
            return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response();
        }
    };

    // If DSpy-rs request, format the response with field markers
    if is_dspy && !output_fields.is_empty() {
        tracing::debug!(field_count = output_fields.len(), "Formatting response for DSpy-rs");
        response_text = format_dspy_response(&response_text, &output_fields);
        tracing::trace!(formatted_response = %response_text, "DSpy-rs formatted response");
    }

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
    mut prompt: String,
) -> Response {
    let model_name = req.model.clone();
    let completion_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());

    // Detect if this is a DSpy-rs structured output request and simplify for streaming
    let is_dspy = is_dspy_request(&prompt);
    let output_fields = if is_dspy {
        tracing::debug!("Detected DSpy-rs structured output request in streaming mode");
        let fields = extract_dspy_output_fields(&prompt);
        tracing::debug!(fields = ?fields, "Extracted DSpy-rs output fields");

        // Simplify by extracting just the actual question
        if let Some(question) = extract_dspy_question(&prompt) {
            tracing::debug!(original_length = prompt.len(), simplified_length = question.len(), "Simplified DSpy prompt for streaming");
            prompt = question;
            tracing::trace!(simplified_prompt = %prompt, "Using simplified question for streaming");
        } else {
            tracing::warn!("Failed to extract question from DSpy prompt in streaming mode, using original");
        }

        fields
    } else {
        vec![]
    };

    tracing::info!(
        completion_id = %completion_id,
        model = %model_name,
        is_dspy = is_dspy,
        "Starting streaming completion"
    );

    // Get a process from the pool and stream
    let stream = match state.pool.get_process().await {
        Ok(process) => {
            tracing::debug!("Acquired process from pool for streaming");
            match process.send_prompt_stream(&prompt).await {
                Ok(s) => {
                    tracing::debug!("Stream initialized successfully");
                    s
                }
                Err(e) => {
                    tracing::error!(error = %e, "Failed to initialize prompt stream");
                    return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response();
                }
            }
        }
        Err(e) => {
            tracing::error!(error = %e, "Failed to acquire process from pool");
            return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response();
        }
    };

    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    // Create state for the stream transformation
    struct StreamState {
        dspy_header_sent: bool,
        is_dspy: bool,
        first_field: Option<String>,
        completion_sent: bool,
    }

    let state = StreamState {
        dspy_header_sent: false,
        is_dspy: is_dspy,
        first_field: output_fields.first().cloned(),
        completion_sent: false,
    };

    use futures_util::stream;

    // Transform the stream to add DSpy markers if needed
    let transformed_stream = stream::unfold((stream, state), move |(mut s, mut state)| async move {
        match s.next().await {
            Some(Ok(mut token)) => {
                // For DSpy requests, wrap the first chunk with field marker
                if state.is_dspy && !state.dspy_header_sent {
                    if let Some(ref first_field) = state.first_field {
                        token = format!("[[ ## {} ## ]]\n{}", first_field, token);
                        state.dspy_header_sent = true;
                    }
                }

                Some((Ok(token), (s, state)))
            }
            Some(Err(e)) => Some((Err(e), (s, state))),
            None => {
                // Stream ended - if DSpy and haven't sent completion, send it now
                if state.is_dspy && !state.completion_sent {
                    state.completion_sent = true;
                    Some((Ok("\n\n[[ ## completed ## ]]\n".to_string()), (s, state)))
                } else {
                    None
                }
            }
        }
    });

    let mut first_chunk = true;
    let mut chunk_sent_completion = false;
    let sse_stream = transformed_stream.map(move |chunk_result| {
        let event = match chunk_result {
            Ok(token) => {
                // Check if this is a completion marker chunk (before moving token)
                let is_completion = token.contains("[[ ## completed ## ]]");
                let finish_reason = if is_completion && !chunk_sent_completion {
                    chunk_sent_completion = true;
                    Some("stop".to_string())
                } else {
                    None
                };

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
                        finish_reason,
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
    tracing::debug!("Listing locally downloaded models");

    // Get list of locally downloaded models
    let models_output = match state.manager.list_models(false).await {
        Ok(output) => {
            tracing::debug!("Successfully retrieved model list");
            output
        }
        Err(e) => {
            tracing::error!(error = %e, "Failed to list models");
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
    tracing::debug!(model_id = %model_id, "Looking up specific model");

    // Get list of locally downloaded models
    let models_output = match state.manager.list_models(false).await {
        Ok(output) => output,
        Err(e) => {
            tracing::error!(error = %e, model_id = %model_id, "Failed to list models");
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
        tracing::warn!(model_id = %model_id, "Model not found");
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

    tracing::debug!(model_id = %model_id, "Model found");
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
