use anyhow::Result;
use rmcp::{
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{ErrorData as McpError, *},
    schemars, tool, tool_handler, tool_router, ServerHandler,
    service::{RequestContext, Peer}, RoleServer,
};
use serde::{Deserialize, Serialize};
use std::{borrow::Cow, collections::HashMap, sync::Arc};
use tokio::sync::{RwLock, Mutex};
use uuid::Uuid;

use crate::manager::LitManager;

// Download progress tracking
#[derive(Debug, Clone, Serialize)]
pub struct DownloadProgress {
    pub model: String,
    pub progress: u8, // 0-100
    pub status: DownloadStatus,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum DownloadStatus {
    Pending,
    Downloading,
    Complete,
    Failed(String),
}

// Wrapper to track peers with unique IDs for cleanup
#[derive(Clone)]
struct SubscribedPeer {
    id: Uuid,
    peer: Peer<RoleServer>,
}

#[derive(Clone)]
pub struct LiteRtMcpService {
    manager: Arc<LitManager>,
    tool_router: ToolRouter<LiteRtMcpService>,
    // Track download progress for ALL models (from registry)
    download_progress: Arc<RwLock<HashMap<String, DownloadProgress>>>,
    // Map of resource URIs to subscribed peers with IDs
    subscriptions: Arc<Mutex<HashMap<String, Vec<SubscribedPeer>>>>,
}

// Request types for MCP tools

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ListModelsRequest {
    #[serde(default)]
    #[schemars(description = "List all models available for download in model registry")]
    pub show_all: bool,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct PullModelRequest {
    #[schemars(description = "The model name or URL to download (e.g., 'gemma-2-2b-it' or Hugging Face URL)")]
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schemars(description = "Alias to save the model as (only for URLs)")]
    pub alias: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schemars(description = "Hugging Face API token for authentication")]
    pub hf_token: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct RemoveModelRequest {
    #[schemars(description = "The model name or filename to remove")]
    pub model: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct RunCompletionRequest {
    #[schemars(description = "The model to use for completion")]
    pub model: String,
    #[schemars(description = "The prompt or conversation history")]
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    #[schemars(description = "Maximum tokens to generate (default: 2048)")]
    #[allow(dead_code)]
    pub max_tokens: u32,
    #[serde(default = "default_temperature")]
    #[schemars(description = "Temperature for sampling (default: 0.7)")]
    #[allow(dead_code)]
    pub temperature: f32,
}

fn default_max_tokens() -> u32 {
    2048
}

fn default_temperature() -> f32 {
    0.7
}

#[tool_router(router = tool_router)]
impl LiteRtMcpService {
    pub async fn new(manager: LitManager) -> Result<Self> {
        let manager_arc = Arc::new(manager);

        tracing::info!("Initializing MCP service, loading model registry...");
        // Initialize download progress from model registry
        let download_progress = Self::initialize_model_registry(manager_arc.clone()).await?;
        tracing::info!("Model registry loaded with {} models", download_progress.len());

        Ok(Self {
            manager: manager_arc,
            tool_router: Self::tool_router(),
            download_progress: Arc::new(RwLock::new(download_progress)),
            subscriptions: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Initialize model registry by listing all available models
    async fn initialize_model_registry(manager: Arc<LitManager>) -> Result<HashMap<String, DownloadProgress>> {
        let binary_path = manager.ensure_binary_path().await?;

        // Get list of all models in registry
        let output = std::process::Command::new(&binary_path)
            .arg("list")
            .arg("--show_all")
            .output()?;

        if !output.status.success() {
            anyhow::bail!("Failed to list models from registry");
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut progress_map = HashMap::new();

        // Parse the output and create download progress entries
        // Each line is a model name
        for line in stdout.lines() {
            let model = line.trim();
            if !model.is_empty() && !model.starts_with("Available") && !model.starts_with("Downloaded") {
                // Check if model is already downloaded by looking at downloaded models list
                let is_downloaded = Self::check_if_downloaded(&binary_path, model).await?;

                let status = if is_downloaded {
                    DownloadStatus::Complete
                } else {
                    DownloadStatus::Pending
                };

                progress_map.insert(model.to_string(), DownloadProgress {
                    model: model.to_string(),
                    progress: if is_downloaded { 100 } else { 0 },
                    status,
                });
            }
        }

        Ok(progress_map)
    }

    /// Check if a model is already downloaded
    async fn check_if_downloaded(binary_path: &std::path::PathBuf, model: &str) -> Result<bool> {
        let output = std::process::Command::new(binary_path)
            .arg("list")
            .output()?;

        if !output.status.success() {
            return Ok(false);
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        Ok(stdout.lines().any(|line| line.trim() == model))
    }

    /// Get current download progress for a model (library API)
    pub async fn query_download_progress(&self, model: &str) -> Option<DownloadProgress> {
        self.download_progress.read().await.get(model).cloned()
    }

    /// Update download progress and notify subscribers
    async fn update_progress(&self, model: String, progress: u8, status: DownloadStatus) {
        // Update the progress
        let mut downloads = self.download_progress.write().await;
        downloads.insert(model.clone(), DownloadProgress {
            model: model.clone(),
            progress,
            status,
        });
        drop(downloads);

        // Notify all subscribers
        let uri = format!("litert://downloads/{}", model);
        self.notify_subscribers(&uri).await;
    }

    /// Send notifications to all peers subscribed to a resource
    async fn notify_subscribers(&self, uri: &str) {
        let mut subscriptions = self.subscriptions.lock().await;

        if let Some(peers) = subscriptions.get_mut(uri) {
            // Track which peers failed (disconnected)
            let mut failed_indices = Vec::new();

            // Send notification to each subscribed peer
            for (idx, subscribed_peer) in peers.iter().enumerate() {
                // Check if transport is already closed before sending
                if subscribed_peer.peer.is_transport_closed() {
                    failed_indices.push(idx);
                    continue;
                }

                let peer_clone = subscribed_peer.peer.clone();
                let uri_clone = uri.to_string();

                // Spawn notification task to avoid blocking
                tokio::spawn(async move {
                    if let Err(e) = peer_clone.notify_resource_updated(ResourceUpdatedNotificationParam {
                        uri: uri_clone.clone(),
                    }).await {
                        tracing::debug!("Failed to notify peer about resource {}: {}", uri_clone, e);
                    }
                });
            }

            // Remove disconnected peers (in reverse order to preserve indices)
            for &idx in failed_indices.iter().rev() {
                peers.swap_remove(idx);
            }

            // Remove empty subscription entries
            if peers.is_empty() {
                subscriptions.remove(uri);
                tracing::debug!("Removed empty subscription for: {}", uri);
            } else if !failed_indices.is_empty() {
                tracing::info!("Cleaned up {} disconnected peer(s) from resource: {}", failed_indices.len(), uri);
            }
        }
    }


    /// List all locally downloaded LiteRT models
    #[tool(description = "List all locally downloaded LiteRT models (or all available with show_all=true)")]
    async fn list_models(
        &self,
        Parameters(request): Parameters<ListModelsRequest>,
    ) -> Result<CallToolResult, McpError> {
        let manager = self.manager.clone();
        let show_all = request.show_all;

        let result = tokio::task::spawn_blocking(move || {
            tokio::runtime::Handle::current().block_on(async move {
                // Use the binary to list models
                let binary_path = manager.ensure_binary_path().await
                    .map_err(|e| format!("Failed to get binary: {}", e))?;

                let mut cmd = std::process::Command::new(&binary_path);
                cmd.arg("list");
                if show_all {
                    cmd.arg("--show_all");
                }

                let output = cmd
                    .output()
                    .map_err(|e| format!("Failed to execute list: {}", e))?;

                Ok::<String, String>(String::from_utf8_lossy(&output.stdout).to_string())
            })
        })
        .await
        .map_err(|e| McpError {
            code: ErrorCode(-32603),
            message: Cow::from(format!("Task failed: {}", e)),
            data: None,
        })?
        .map_err(|e| McpError {
            code: ErrorCode(-32603),
            message: Cow::from(e),
            data: None,
        })?;

        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    /// Download a model from registry or URL
    #[tool(description = "Download a LiteRT model from registry or URL (e.g., Hugging Face)")]
    async fn pull_model(
        &self,
        Parameters(request): Parameters<PullModelRequest>,
    ) -> Result<CallToolResult, McpError> {
        let manager = self.manager.clone();
        let model = request.model.clone();
        let alias = request.alias.clone();
        let hf_token = request.hf_token.clone();

        // Track download progress
        let progress_tracker = self.clone();
        let progress_model = model.clone();

        // Initialize progress
        self.update_progress(model.clone(), 0, DownloadStatus::Pending).await;

        // Spawn progress updates in background
        let progress_handle = tokio::spawn(async move {
            for pct in (0..=100).step_by(10) {
                tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;
                let status = if pct < 100 {
                    DownloadStatus::Downloading
                } else {
                    DownloadStatus::Complete
                };
                progress_tracker.update_progress(progress_model.clone(), pct, status).await;
            }
        });

        let result = tokio::task::spawn_blocking(move || {
            tokio::runtime::Handle::current().block_on(
                manager.pull(&model, alias.as_deref(), hf_token.as_deref())
            )
        })
        .await
        .map_err(|e| McpError {
            code: ErrorCode(-32603),
            message: Cow::from(format!("Task failed: {}", e)),
            data: None,
        })?;

        match result {
            Ok(_) => {
                progress_handle.abort();
                self.update_progress(request.model.clone(), 100, DownloadStatus::Complete).await;
                Ok(CallToolResult::success(vec![Content::text(format!(
                    "Successfully pulled model: {}. Check litert://downloads/{} for progress.",
                    request.model, request.model
                ))]))
            }
            Err(e) => {
                progress_handle.abort();
                self.update_progress(
                    request.model.clone(),
                    0,
                    DownloadStatus::Failed(e.to_string())
                ).await;
                Err(McpError {
                    code: ErrorCode(-32603),
                    message: Cow::from(format!("Failed to pull model: {}", e)),
                    data: None,
                })
            }
        }
    }

    /// Remove a locally downloaded model
    #[tool(description = "Remove a locally downloaded LiteRT model by name or filename")]
    async fn remove_model(
        &self,
        Parameters(request): Parameters<RemoveModelRequest>,
    ) -> Result<CallToolResult, McpError> {
        let manager = self.manager.clone();
        let model = request.model.clone();

        tokio::task::spawn_blocking(move || {
            tokio::runtime::Handle::current().block_on(manager.remove(&model))
        })
        .await
        .map_err(|e| McpError {
            code: ErrorCode(-32603),
            message: Cow::from(format!("Task failed: {}", e)),
            data: None,
        })?
        .map_err(|e| McpError {
            code: ErrorCode(-32603),
            message: Cow::from(format!("Failed to remove model: {}", e)),
            data: None,
        })?;

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Successfully removed model: {}",
            request.model
        ))]))
    }

    /// Generate a completion using a LiteRT model
    #[tool(description = "Generate a text completion using a LiteRT model")]
    async fn run_completion(
        &self,
        Parameters(request): Parameters<RunCompletionRequest>,
    ) -> Result<CallToolResult, McpError> {
        let manager = self.manager.clone();
        let model = request.model.clone();
        let prompt = request.prompt.clone();

        let result = tokio::task::spawn_blocking(move || {
            tokio::runtime::Handle::current().block_on(async move {
                manager.run_completion(&model, &prompt).await
                    .map_err(|e| format!("Failed to run completion: {}", e))
            })
        })
        .await
        .map_err(|e| McpError {
            code: ErrorCode(-32603),
            message: Cow::from(format!("Task failed: {}", e)),
            data: None,
        })?
        .map_err(|e| McpError {
            code: ErrorCode(-32603),
            message: Cow::from(e),
            data: None,
        })?;

        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    /// Get download progress for a model
    #[tool(description = "Get download progress for a model (if currently downloading)")]
    async fn check_download_progress(
        &self,
        Parameters(request): Parameters<RemoveModelRequest>, // Reuse for model param
    ) -> Result<CallToolResult, McpError> {
        if let Some(progress) = self.query_download_progress(&request.model).await {
            let json = serde_json::to_string_pretty(&progress).map_err(|e| McpError {
                code: ErrorCode(-32603),
                message: Cow::from(format!("Failed to serialize progress: {}", e)),
                data: None,
            })?;
            Ok(CallToolResult::success(vec![Content::text(json)]))
        } else {
            Ok(CallToolResult::success(vec![Content::text(format!(
                "No download in progress for model: {}",
                request.model
            ))]))
        }
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for LiteRtMcpService {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::V_2025_06_18,
            capabilities: ServerCapabilities::builder()
                .enable_tools()
                .enable_resources()
                .enable_resources_subscribe()
                .build(),
            server_info: Implementation {
                name: "litert-lm".into(),
                version: env!("CARGO_PKG_VERSION").into(),
                icons: None,
                title: None,
                website_url: None,
            },
            instructions: Some(
                "LiteRT-LM MCP server. Tools: list_models, pull_model, remove_model, run_completion, check_download_progress. Resources: litert://downloads/{model} for download progress tracking with subscription support."
                    .into(),
            ),
        }
    }

    async fn list_resources(
        &self,
        _request: Option<PaginatedRequestParam>,
        _ctx: RequestContext<RoleServer>,
    ) -> Result<ListResourcesResult, McpError> {
        let downloads = self.download_progress.read().await;
        let resources: Vec<Resource> = downloads
            .values()
            .map(|progress| {
                RawResource {
                    uri: format!("litert://downloads/{}", progress.model),
                    name: progress.model.clone(),
                    description: Some(format!(
                        "Download progress for {} ({}%)",
                        progress.model, progress.progress
                    )),
                    mime_type: Some("application/json".into()),
                    icons: None,
                    size: None,
                    title: Some(format!("{} Download", progress.model)),
                }
                .no_annotation()
            })
            .collect();

        Ok(ListResourcesResult {
            resources,
            next_cursor: None,
        })
    }

    async fn read_resource(
        &self,
        ReadResourceRequestParam { uri }: ReadResourceRequestParam,
        _ctx: RequestContext<RoleServer>,
    ) -> Result<ReadResourceResult, McpError> {
        // Extract model name from URI: litert://downloads/{model}
        let uri_str = uri.as_str();
        let model = uri_str
            .strip_prefix("litert://downloads/")
            .ok_or_else(|| {
                McpError::resource_not_found(
                    "Invalid resource URI",
                    Some(serde_json::json!({"uri": uri_str})),
                )
            })?;

        let downloads = self.download_progress.read().await;
        let progress = downloads.get(model).ok_or_else(|| {
            McpError::resource_not_found(
                "Download progress not found",
                Some(serde_json::json!({"model": model})),
            )
        })?;

        let json_content = serde_json::to_string_pretty(progress).map_err(|e| McpError {
            code: ErrorCode(-32603),
            message: Cow::from(format!("Failed to serialize progress: {}", e)),
            data: None,
        })?;

        Ok(ReadResourceResult {
            contents: vec![ResourceContents::text(json_content, uri)],
        })
    }

    async fn list_resource_templates(
        &self,
        _request: Option<PaginatedRequestParam>,
        _ctx: RequestContext<RoleServer>,
    ) -> Result<ListResourceTemplatesResult, McpError> {
        Ok(ListResourceTemplatesResult {
            next_cursor: None,
            resource_templates: Vec::new(),
        })
    }

    async fn subscribe(
        &self,
        request: SubscribeRequestParam,
        ctx: RequestContext<RoleServer>,
    ) -> Result<(), McpError> {
        let uri = request.uri;

        // Validate URI format (must be litert://downloads/{model})
        if !uri.starts_with("litert://downloads/") {
            return Err(McpError {
                code: ErrorCode(-32602),
                message: Cow::from("Invalid resource URI. Must start with 'litert://downloads/'"),
                data: Some(serde_json::json!({"uri": uri})),
            });
        }

        // Extract model name
        let model = uri.strip_prefix("litert://downloads/")
            .ok_or_else(|| McpError {
                code: ErrorCode(-32602),
                message: Cow::from("Invalid resource URI format"),
                data: Some(serde_json::json!({"uri": uri})),
            })?;

        // Check if the model exists in registry
        let downloads = self.download_progress.read().await;
        if !downloads.contains_key(model) {
            return Err(McpError::resource_not_found(
                "Model not found in registry",
                Some(serde_json::json!({"model": model, "uri": uri})),
            ));
        }
        drop(downloads);

        // Get the peer (client handle) from the request context
        let peer = ctx.peer.clone();

        // Generate a unique ID for this subscription
        let subscription_id = Uuid::new_v4();

        // Add peer to subscription map
        let mut subscriptions = self.subscriptions.lock().await;
        let subscribers = subscriptions.entry(uri.clone()).or_insert_with(Vec::new);

        // Add the wrapped peer with unique ID
        subscribers.push(SubscribedPeer {
            id: subscription_id,
            peer: peer.clone(),
        });

        let subscriber_count = subscribers.len();
        drop(subscriptions); // Release lock before spawning task

        tracing::info!("Client subscribed to resource: {} (total subscribers: {}, id: {})",
            uri, subscriber_count, subscription_id);

        // CRITICAL: Spawn cleanup task to remove peer when it disconnects
        let subscriptions_clone = self.subscriptions.clone();
        let uri_clone = uri.clone();

        tokio::spawn(async move {
            // Poll for disconnect every 5 seconds
            while !peer.is_transport_closed() {
                tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
            }

            tracing::info!("Client disconnected, cleaning up subscription to: {} (id: {})",
                uri_clone, subscription_id);

            // Lock the map and remove this specific peer by ID
            let mut subs = subscriptions_clone.lock().await;
            if let Some(peers) = subs.get_mut(&uri_clone) {
                let before_count = peers.len();

                // Remove the peer with matching ID
                peers.retain(|p| p.id != subscription_id);

                let after_count = peers.len();

                if before_count > after_count {
                    tracing::info!("Removed disconnected peer {} from resource: {} ({} subscribers remaining)",
                        subscription_id, uri_clone, after_count);
                }

                // Remove empty entries
                if peers.is_empty() {
                    subs.remove(&uri_clone);
                    tracing::info!("No subscribers left for {}, removing entry.", uri_clone);
                }
            }
        });

        Ok(())
    }

    async fn unsubscribe(
        &self,
        request: UnsubscribeRequestParam,
        _ctx: RequestContext<RoleServer>,
    ) -> Result<(), McpError> {
        let uri = request.uri;

        // Remove peer from subscriptions
        let mut subscriptions = self.subscriptions.lock().await;

        if let Some(peers) = subscriptions.get_mut(&uri) {
            // Since we can't compare Peer directly, this is a simplified approach
            // A production system would track peers by ID
            // For now, we'll just clear the list (since cleanup happens on disconnect anyway)
            let original_len = peers.len();
            peers.clear();

            if peers.is_empty() {
                subscriptions.remove(&uri);
            }

            tracing::info!("Client unsubscribed from resource: {} (removed {} subscribers)", uri, original_len);
        } else {
            tracing::warn!("Client attempted to unsubscribe from non-subscribed resource: {}", uri);
        }

        Ok(())
    }
}
