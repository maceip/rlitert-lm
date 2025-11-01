use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use tokio_stream::Stream;

use crate::binary::BinaryManager;
use crate::process::ProcessPool;
use crate::server::{create_router, AppState};

#[derive(Debug, Clone)]
pub struct LitManager {
    binary_manager: BinaryManager,
    binary_path: Arc<RwLock<Option<PathBuf>>>,
    // Map of pools, keyed by model name
    process_pools: Arc<Mutex<HashMap<String, Arc<ProcessPool>>>>,
    // Make pool size configurable
    pool_size: usize,
}

impl LitManager {
    pub async fn new() -> Result<Self> {
        Self::new_with_pool_size(2).await
    }

    pub async fn new_with_pool_size(pool_size: usize) -> Result<Self> {
        let binary_manager = BinaryManager::new()?;

        Ok(Self {
            binary_manager,
            binary_path: Arc::new(RwLock::new(None)),
            process_pools: Arc::new(Mutex::new(HashMap::new())),
            pool_size,
        })
    }

    async fn ensure_binary(&self) -> Result<PathBuf> {
        let read_lock = self.binary_path.read().await;
        if let Some(path) = read_lock.as_ref() {
            return Ok(path.clone());
        }
        drop(read_lock);

        let mut write_lock = self.binary_path.write().await;
        if let Some(path) = write_lock.as_ref() {
            return Ok(path.clone());
        }

        let path = self.binary_manager.ensure_binary().await?;
        *write_lock = Some(path.clone());
        Ok(path)
    }

    pub async fn ensure_binary_path(&self) -> Result<PathBuf> {
        self.ensure_binary().await
    }

    // Helper function to get-or-create a pool for a specific model
    async fn get_pool(&self, model: &str) -> Result<Arc<ProcessPool>> {
        // 1. Lock the pool map
        let mut pools = self.process_pools.lock().await;

        // 2. Check if a pool for this model already exists
        if let Some(pool) = pools.get(model) {
            return Ok(pool.clone());
        }

        // 3. If not, create, initialize, and insert it
        let binary_path = self.ensure_binary().await?;
        let mut new_pool = ProcessPool::new(
            binary_path,
            model.to_string(),
            self.pool_size,
        );

        new_pool.initialize().await?; // Initialize *before* inserting

        let pool_arc = Arc::new(new_pool);
        pools.insert(model.to_string(), pool_arc.clone());
        Ok(pool_arc)
    }

    pub async fn run_completion(&self, model: &str, prompt: &str) -> Result<String> {
        // Get the correct pool for the requested model
        let pool = self.get_pool(model).await?;

        // Use the pool
        let response = pool.send_prompt(prompt).await?;
        Ok(response)
    }

    // New streaming method
    pub async fn run_completion_stream(
        &self,
        model: &str,
        prompt: &str,
    ) -> Result<impl Stream<Item = Result<String>>> {
        let pool = self.get_pool(model).await?;
        let process = pool.get_process().await?;
        let stream = process.send_prompt_stream(prompt).await?;
        Ok(stream)
    }

    fn run_lit_command(&self, binary_path: &PathBuf, args: &[&str]) -> Result<String> {
        let output = Command::new(binary_path)
            .args(args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .context("Failed to execute lit command")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("Command failed: {}", stderr);
        }

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    pub async fn list(&self, show_all: bool) -> Result<()> {
        let binary_path = self.ensure_binary().await?;
        let args = if show_all {
            vec!["list", "--show_all"]
        } else {
            vec!["list"]
        };
        let output = self.run_lit_command(&binary_path, &args)?;
        println!("{}", output);
        Ok(())
    }

    pub async fn pull(&self, model: &str, alias: Option<&str>, hf_token: Option<&str>) -> Result<()> {
        let binary_path = self.ensure_binary().await?;
        tracing::info!("Pulling model: {}", model);

        let mut cmd = Command::new(&binary_path);
        cmd.arg("pull").arg(model);

        if let Some(alias_val) = alias {
            cmd.arg("--alias").arg(alias_val);
        }

        if let Some(token) = hf_token {
            cmd.arg("--hf_token").arg(token);
        }

        let output = cmd
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .status()
            .context("Failed to pull model")?;

        if !output.success() {
            anyhow::bail!("Failed to pull model");
        }

        Ok(())
    }

    pub async fn remove(&self, model: &str) -> Result<()> {
        let binary_path = self.ensure_binary().await?;
        let output = self.run_lit_command(&binary_path, &["rm", model])?;
        println!("{}", output);
        Ok(())
    }

    pub async fn run_interactive(&self, model: &str) -> Result<()> {
        let binary_path = self.ensure_binary().await?;

        let status = Command::new(&binary_path)
            .args(&["run", model])
            .stdin(Stdio::inherit())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .status()
            .context("Failed to run interactive session")?;

        if !status.success() {
            anyhow::bail!("Interactive session failed");
        }

        Ok(())
    }

    pub fn generate_completion(&self, shell: &str) -> Result<()> {
        println!("Completion generation for {} not yet implemented", shell);
        Ok(())
    }

    pub async fn serve(&self, port: u16) -> Result<()> {
        tracing::info!("Starting server on port {}", port);

        // Ensure binary is ready
        let binary_path = self.ensure_binary().await?;
        tracing::info!("Binary ready at: {}", binary_path.display());

        // Default model for initialization - pool will be created on-demand
        let model = std::env::var("LITERT_MODEL")
            .unwrap_or_else(|_| "gemma-2-2b-it".to_string());

        // Pre-initialize pool for default model
        let pool = self.get_pool(&model).await?;
        tracing::info!("Process pool initialized for model '{}' with {} instances", model, self.pool_size);

        // Start server - AppState now holds the manager instead of a single pool
        let app_state = AppState {
            pool, // Keep the old interface for now
        };
        let app = create_router(app_state);

        let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", port))
            .await
            .context("Failed to bind to port")?;

        tracing::info!("Server listening on http://0.0.0.0:{}", port);
        tracing::info!("OpenAI-compatible endpoint: http://localhost:{}/v1/chat/completions", port);

        axum::serve(listener, app)
            .await
            .context("Server error")?;

        Ok(())
    }
}
