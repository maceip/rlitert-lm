//! Example MCP server with stdio transport
//!
//! This demonstrates how to create an MCP server that can be used with
//! Claude Desktop, Cursor, or other MCP clients.
//!
//! Run with: cargo run --example mcp_server

use litert_lm::{LitManager, LiteRtMcpService, Result};
use rmcp::ServiceExt;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    println!("Starting MCP server with stdio transport...");
    println!("This server exposes 4 tools:");
    println!("  - list_models: List locally downloaded models");
    println!("  - pull_model: Download a model");
    println!("  - remove_model: Remove a model");
    println!("  - run_completion: Generate text completions");
    println!();

    // Create the manager
    let manager = LitManager::new().await?;

    // Create MCP service
    let service = LiteRtMcpService::new(manager).await?;

    // Serve over stdin/stdout
    let (stdin, stdout) = (tokio::io::stdin(), tokio::io::stdout());
    service.serve((stdin, stdout)).await?;

    Ok(())
}
