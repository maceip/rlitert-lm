//! Basic example showing how to use litert-lm as a library
//!
//! Run with: cargo run --example basic_usage

use litert_lm::{LitManager, Result};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing for logs
    tracing_subscriber::fmt::init();

    // Create the manager - this will download the binary if needed
    let manager = LitManager::new().await?;

    // List available models
    println!("=== Listing Models ===");
    manager.list().await?;

    // Pull a model (you can skip this if already downloaded)
    println!("\n=== Pulling Model ===");
    match manager.pull("gemma-2-2b-it").await {
        Ok(_) => println!("Model pulled successfully"),
        Err(e) => println!("Note: {}", e),
    }

    // Run a completion
    println!("\n=== Running Completion ===");
    let response = manager
        .run_completion("gemma-2-2b-it", "What is the capital of France?")
        .await?;

    println!("Response: {}", response);

    Ok(())
}
