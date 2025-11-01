//! Example demonstrating concurrent requests with the process pool
//!
//! This shows how the process pool handles multiple requests concurrently
//! by maintaining 2 isolated LiteRT processes.
//!
//! Run with: cargo run --example process_pool

use litert_lm::{LitManager, Result};
use tokio::task::JoinSet;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    println!("Demonstrating concurrent completions with process pool...");
    println!("The pool maintains 2 isolated LiteRT processes\n");

    let manager = LitManager::new().await?;

    // Create multiple concurrent requests
    let prompts = vec![
        "What is 2+2?",
        "Name a color",
        "What is the capital of Japan?",
        "Count to 3",
    ];

    let mut tasks = JoinSet::new();

    for (i, prompt) in prompts.iter().enumerate() {
        let mgr = manager.clone();
        let p = prompt.to_string();

        tasks.spawn(async move {
            println!("[Request {}] Sending: {}", i + 1, p);
            let start = std::time::Instant::now();

            match mgr.run_completion("gemma-2-2b-it", &p).await {
                Ok(response) => {
                    println!(
                        "[Request {}] Completed in {:?}: {}",
                        i + 1,
                        start.elapsed(),
                        response.lines().next().unwrap_or(&response)
                    );
                }
                Err(e) => {
                    println!("[Request {}] Error: {}", i + 1, e);
                }
            }
        });
    }

    // Wait for all requests to complete
    while let Some(result) = tasks.join_next().await {
        result?;
    }

    println!("\nAll requests completed!");

    Ok(())
}
