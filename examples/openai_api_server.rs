//! Example OpenAI-compatible API server
//!
//! This creates an HTTP server with OpenAI-compatible endpoints
//! that can be used as a drop-in replacement.
//!
//! Run with: cargo run --example openai_api_server
//! Test with: curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"gemma-3n-E4B","messages":[{"role":"user","content":"Hello!"}]}'

use litert_lm::{LitManager, Result};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    println!("Starting OpenAI-compatible API server on port 8080...");
    println!("Endpoint: http://localhost:8080/v1/chat/completions");
    println!();
    println!("Example curl command:");
    println!(r#"  curl http://localhost:8080/v1/chat/completions \"#);
    println!(r#"    -H "Content-Type: application/json" \"#);
    println!(r#"    -d '{{"model":"gemma-3n-E4B","messages":[{{"role":"user","content":"Hello!"}}]}}'"#);
    println!();

    // Set the model to use (optional, defaults to gemma-3n-E4B)
    std::env::set_var("LITERT_MODEL", "gemma-3n-E4B");

    let manager = LitManager::new().await?;
    manager.serve(8080).await?;

    Ok(())
}
