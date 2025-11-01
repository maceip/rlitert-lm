//! LiteRT-LM: A Rust library for LiteRT-LM model inference
//!
//! This library provides:
//! - Model download and management
//! - Process pool management for efficient inference
//! - Streaming completions
//! - MCP (Model Context Protocol) service
//! - OpenAI-compatible API server
//!
//! # Example
//!
//! ```no_run
//! use litert_lm::{LitManager, Result};
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     let manager = LitManager::new().await?;
//!
//!     // Pull a model
//!     manager.pull("gemma-3n-E4B", None, None).await?;
//!
//!     // Run completion
//!     let response = manager.run_completion("gemma-3n-E4B", "Hello!").await?;
//!     println!("{}", response);
//!
//!     Ok(())
//! }
//! ```

pub mod binary;
pub mod manager;
pub mod mcp;
pub mod process;
pub mod server;

// Re-export main types for library users
pub use manager::LitManager;
pub use mcp::LiteRtMcpService;
pub use process::{LitProcess, ProcessPool};
pub use server::{AppState, ChatCompletionRequest, create_router};

// Re-export common types
pub type Result<T> = std::result::Result<T, anyhow::Error>;
