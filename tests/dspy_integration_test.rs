/// Integration test for DSpy-rs compatibility
///
/// This test verifies that DSpy-rs can use our litert-lm server as a custom
/// OpenAI-compatible endpoint using LM::builder() with base_url.
///
/// Based on the example from:
/// https://github.com/krypticmouse/DSRs/blob/main/crates/dspy-rs/examples/10-gepa-llm-judge.rs

use anyhow::Result;
use dspy_rs::*;
use litert_lm::LitManager;
use secrecy::SecretString;
use tokio::time::{sleep, Duration};

#[tokio::test]
async fn test_dspy_custom_endpoint() -> Result<()> {
    tracing_subscriber::fmt::init();

    // Start our OpenAI-compatible server
    let manager = LitManager::new().await?;
    let port = 18082;

    // Spawn server in background
    let server_handle = tokio::spawn(async move {
        manager.serve(port).await
    });

    // Give server time to start
    sleep(Duration::from_secs(3)).await;

    // Test DSpy-rs LM::builder() with custom endpoint
    println!("Testing DSpy-rs LM::builder() with custom OpenAI-compatible endpoint...");

    let lm = LM::builder()
        .base_url(format!("http://localhost:{}/v1", port))
        .api_key(SecretString::new("dummy-key".to_string()))
        .model("gemma-3n-E4B".to_string())
        .build()
        .await?;

    println!("✓ LM builder successfully connected to custom endpoint");

    // Test a simple completion through DSpy-rs
    println!("\nTesting simple completion through DSpy-rs...");

    let prompt = example! {
        "question": "input" => "What is 2+2?"
    };

    // Create a simple predict module
    let predictor = Predict::new("question -> answer");

    match predictor.forward_with_config(prompt, std::sync::Arc::new(lm)).await {
        Ok(prediction) => {
            println!("✓ DSpy-rs successfully got response from our server");
            println!("Response: {:?}", prediction);
        }
        Err(e) => {
            eprintln!("✗ DSpy-rs failed to get response: {}", e);
            server_handle.abort();
            return Err(e);
        }
    }

    // Cleanup
    server_handle.abort();
    println!("\n✓ All DSpy-rs integration tests passed!");

    Ok(())
}

#[tokio::test]
async fn test_dspy_math_judge_pattern() -> Result<()> {
    tracing_subscriber::fmt::init();

    // Start our OpenAI-compatible server
    let manager = LitManager::new().await?;
    let port = 18083;

    let server_handle = tokio::spawn(async move {
        manager.serve(port).await
    });

    sleep(Duration::from_secs(3)).await;

    println!("Testing DSpy-rs math problem solver pattern (like 10-gepa-llm-judge.rs)...");

    // Create LM for the solver
    let solver_lm = LM::builder()
        .base_url(format!("http://localhost:{}/v1", port))
        .api_key(SecretString::new("dummy-key".to_string()))
        .model("gemma-3n-E4B".to_string())
        .build()
        .await?;

    println!("✓ Created solver LM with custom endpoint");

    // Create a math problem example
    let math_problem = example! {
        "problem": "input" => "If Sarah has 3 apples and buys 2 more, how many apples does she have?",
        "expected_answer": "output" => "5"
    };

    // Create a predictor for solving math problems
    let solver = Predict::new("problem -> reasoning, answer");

    println!("\nSending math problem to solver...");
    match solver.forward_with_config(math_problem, std::sync::Arc::new(solver_lm)).await {
        Ok(prediction) => {
            println!("✓ Solver responded successfully");
            println!("Answer: {:?}", prediction.get("answer", None));
            println!("Reasoning: {:?}", prediction.get("reasoning", None));
        }
        Err(e) => {
            eprintln!("✗ Solver failed: {}", e);
            server_handle.abort();
            return Err(e);
        }
    }

    server_handle.abort();
    println!("\n✓ Math judge pattern test passed!");

    Ok(())
}
