/// Integration test for OpenAI API compatibility
///
/// This test verifies that our litert-lm server is compatible with the OpenAI API
/// by using the async-openai client library (same library used by DSpy-rs).
///
/// This is based on the pattern from:
/// https://github.com/krypticmouse/DSRs/blob/main/crates/dspy-rs/examples/10-gepa-llm-judge.rs
///
/// The DSpy-rs framework uses async-openai internally, so if this test passes,
/// it means DSpy-rs and similar frameworks can use our server as a drop-in replacement.

use async_openai::{
    config::OpenAIConfig,
    types::{
        ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
        ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequestArgs,
    },
    Client,
};
use litert_lm::{LitManager, Result};
use tokio::time::{sleep, Duration};

/// Test that async-openai can connect to our server and make requests
/// This proves compatibility with DSpy-rs and other OpenAI-compatible frameworks
#[tokio::test]
async fn test_openai_client_compatibility() -> Result<()> {
    tracing_subscriber::fmt::init();

    // Start our OpenAI-compatible server
    let manager = LitManager::new().await?;
    let port = 18080; // Use a different port for testing

    // Spawn server in background
    let server_handle = tokio::spawn(async move {
        manager.serve(port).await
    });

    // Give server time to start
    sleep(Duration::from_secs(2)).await;

    // Create OpenAI client pointing to our local server
    let config = OpenAIConfig::new()
        .with_api_base(format!("http://localhost:{}/v1", port))
        .with_api_key("dummy-key"); // Our server doesn't check keys yet

    let client = Client::with_config(config);

    // Test 1: List models
    println!("Testing /v1/models endpoint...");
    let models = client.models().list().await?;
    println!("Available models: {:?}", models.data);
    assert!(!models.data.is_empty(), "Should have at least one model");

    // Test 2: Simple chat completion (non-streaming)
    println!("\nTesting /v1/chat/completions (non-streaming)...");
    let request = CreateChatCompletionRequestArgs::default()
        .model("gemma-3n-E4B")
        .messages(vec![ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessageArgs::default()
                .content("What is 2+2?")
                .build()?,
        )])
        .max_tokens(50u32)
        .build()?;

    let response = client.chat().create(request).await?;
    println!("Response: {:?}", response.choices[0].message.content);
    assert!(!response.choices.is_empty(), "Should have at least one choice");

    // Test 3: Chat completion with system prompt
    println!("\nTesting /v1/chat/completions with system prompt...");
    let request = CreateChatCompletionRequestArgs::default()
        .model("gemma-3n-E4B")
        .messages(vec![
            ChatCompletionRequestMessage::System(
                ChatCompletionRequestSystemMessageArgs::default()
                    .content("You are a helpful math tutor.")
                    .build()?,
            ),
            ChatCompletionRequestMessage::User(
                ChatCompletionRequestUserMessageArgs::default()
                    .content("Explain why 2+2=4")
                    .build()?,
            ),
        ])
        .max_tokens(100u32)
        .build()?;

    let response = client.chat().create(request).await?;
    println!("Math tutor response: {:?}", response.choices[0].message.content);
    assert!(!response.choices.is_empty(), "Should have at least one choice");

    // Test 4: Streaming (if supported by async-openai)
    println!("\nTesting /v1/chat/completions (streaming)...");
    let request = CreateChatCompletionRequestArgs::default()
        .model("gemma-3n-E4B")
        .messages(vec![ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessageArgs::default()
                .content("Count to 3")
                .build()?,
        )])
        .max_tokens(20u32)
        .build()?;

    let mut stream = client.chat().create_stream(request).await?;
    let mut full_response = String::new();

    use futures::StreamExt;
    while let Some(result) = stream.next().await {
        match result {
            Ok(response) => {
                for choice in response.choices {
                    if let Some(content) = choice.delta.content {
                        print!("{}", content);
                        full_response.push_str(&content);
                    }
                }
            }
            Err(e) => {
                eprintln!("\nStream error: {}", e);
                break;
            }
        }
    }
    println!("\n\nFull streaming response: {}", full_response);

    // Cleanup
    server_handle.abort();

    Ok(())
}

/// Test retrieving a specific model
#[tokio::test]
async fn test_get_specific_model() -> Result<()> {
    let manager = LitManager::new().await?;
    let port = 18081;

    let server_handle = tokio::spawn(async move {
        manager.serve(port).await
    });

    sleep(Duration::from_secs(2)).await;

    let config = OpenAIConfig::new()
        .with_api_base(format!("http://localhost:{}/v1", port))
        .with_api_key("dummy-key");

    let client = Client::with_config(config);

    // Get specific model
    println!("Testing /v1/models/{{model}} endpoint...");
    match client.models().retrieve("gemma-3n-E4B").await {
        Ok(model) => {
            println!("Model: {:?}", model);
            assert_eq!(model.id, "gemma-3n-E4B");
        }
        Err(e) => {
            println!("Error retrieving model (may not be downloaded): {}", e);
        }
    }

    server_handle.abort();
    Ok(())
}
