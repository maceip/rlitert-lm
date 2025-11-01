# litert-lm

Rust wrapper around [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) providing MCP and OpenAI-compatible interfaces. Auto-downloads platform binary, manages process pools, exposes model operations as tools.

## Install

```bash
cargo build --release
```

## MCP Server

Exposes LiteRT as tools for AI assistants. Provides model download tracking via resources with subscription support.

```bash
# Stdio (Claude Desktop, Cursor)
litert-lm mcp --transport stdio

# SSE over HTTP
litert-lm mcp --transport sse --port 3000
```

Claude Desktop config:

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

Tools: list_models, pull_model, remove_model, run_completion, check_download_progress

Resources: `litert://downloads/{model}` for each model in registry with download state and progress tracking

## OpenAI API

Drop-in replacement for OpenAI endpoints.

```bash
litert-lm serve --port 8080
```

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma-2-2b-it", "messages": [{"role": "user", "content": "Hello"}], "stream": true}'
```

## CLI

```bash
litert-lm list                    # Show downloaded models
litert-lm list --show_all         # Show all available models
litert-lm pull gemma-2-2b-it      # Download model
litert-lm rm gemma-2-2b-it        # Remove model
litert-lm run gemma-2-2b-it       # Interactive session
```

## Library Usage

```rust
use litert_lm::{LitManager, LiteRtMcpService};

let manager = LitManager::new().await?;
manager.pull("gemma-2-2b-it", None, None).await?;
let response = manager.run_completion("gemma-2-2b-it", "Hello").await?;

// MCP service with resource subscriptions
let service = LiteRtMcpService::new(manager).await?;
```

See LIBRARY_USAGE.md for full API documentation.

## Architecture

Binary auto-download for current platform. Multi-model process pools with per-model isolation. Character-level streaming with GPU fallback to CPU. MCP resource subscriptions with automatic peer cleanup on disconnect. SSE streaming for OpenAI API.

## License

MIT
