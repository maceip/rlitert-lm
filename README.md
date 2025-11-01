# litert-lm

Rust wrapper around [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) providing MCP and OpenAI-compatible interfaces. Auto-downloads platform binary, manages process pools, exposes model operations as tools.

## Install

### As a Binary (MCP Server)

```bash
cargo install litert-lm
```

### As a Library

```toml
[dependencies]
litert-lm = "0.2"
```

## Usage

### A) MCP Server (Binary)

Run as an MCP server to expose LiteRT-LM as tools for AI assistants like Claude Desktop.

#### Stdio Transport (Claude Desktop, Cursor)

```bash
litert-lm mcp --transport stdio
```

Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "litert-lm": {
      "command": "/path/to/litert-lm",
      "args": ["mcp", "--transport", "stdio"]
    }
  }
}
```

#### SSE Transport (HTTP)

```bash
litert-lm mcp --transport sse --port 3000
```

#### MCP Features

**Tools:**
- `list_models` - List downloaded or available models
- `pull_model` - Download a model with real-time progress
- `remove_model` - Delete a downloaded model
- `run_completion` - Generate text completions
- `check_download_progress` - Query download status

**Resources:**
`litert://downloads/{model}` - Subscribe to live download progress for any model in the registry

### B) Rust Library

Use litert-lm directly in your Rust code for model inference.

```rust
use litert_lm::{LitManager, Result};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize manager (auto-downloads lit binary if needed)
    let manager = LitManager::new().await?;

    // Download a model
    println!("Downloading model...");
    manager.pull("gemma-3n-E4B", None, None).await?;

    // Run inference
    let response = manager
        .run_completion("gemma-3n-E4B", "What is the capital of France?")
        .await?;

    println!("Response: {}", response);

    Ok(())
}
```

#### Advanced: Progress Tracking

```rust
use litert_lm::{LitManager, Result};

#[tokio::main]
async fn main() -> Result<()> {
    let manager = LitManager::new().await?;

    // Download with real-time progress callback
    manager.pull_with_progress(
        "gemma-3n-E4B",
        None,
        None,
        |progress| {
            println!("Download progress: {:.1}%", progress);
        }
    ).await?;

    // Run completion
    let response = manager
        .run_completion("gemma-3n-E4B", "Hello!")
        .await?;

    println!("{}", response);

    Ok(())
}
```

#### Streaming Responses

```rust
use litert_lm::{LitManager, Result};
use tokio_stream::StreamExt;

#[tokio::main]
async fn main() -> Result<()> {
    let manager = LitManager::new().await?;

    let mut stream = manager
        .run_completion_stream("gemma-3n-E4B", "Tell me a story")
        .await?;

    while let Some(chunk) = stream.next().await {
        print!("{}", chunk?);
    }

    Ok(())
}
```

## OpenAI-Compatible API

Run an OpenAI-compatible server:

```bash
litert-lm serve --port 8080
```

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-3n-E4B",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

## Architecture

- **Auto-download**: Fetches platform-specific lit binary on first run
- **Process pools**: Multi-model support with per-model process isolation
- **Streaming**: Character-level streaming with GPU/CPU fallback
- **MCP**: Resource subscriptions with real download progress tracking
- **OpenAI API**: SSE streaming for compatibility

## Available Models

```bash
# List all available models in registry
litert-lm list --show-all

# List downloaded models only
litert-lm list
```

Some models require a Hugging Face token. Set via environment variable or flag:

```bash
export HUGGING_FACE_HUB_TOKEN=hf_your_token
litert-lm pull gemma3-1b
```

## Testing

See `tests/mcp-tests/` for comprehensive MCP integration tests:

```bash
cd tests/mcp-tests
uv run test_mcp_client.py           # Basic stdio test
uv run test_mcp_sse.py               # SSE transport test
uv run test_mcp_download_quick.py   # Download progress test
```

## License

MIT
