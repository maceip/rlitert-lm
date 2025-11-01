# MCP Tests

Python test suite for litert-lm MCP server using uv for dependency management.

## Setup

```bash
# From repository root, build the binary first
cargo build --release

# Navigate to test directory
cd tests/mcp-tests

# Dependencies are already configured in pyproject.toml
# uv will handle the virtual environment automatically
```

## Run Tests

### Basic Tests (No external requirements)

```bash
# Test basic MCP completion via stdio
uv run test_mcp_client.py

# Test MCP resource subscriptions
uv run test_mcp_resources.py
```

### SSE Transport Test

```bash
# Start the MCP server in SSE mode (in one terminal):
../../target/release/litert-lm mcp --transport sse --port 3000

# Run the SSE test (in another terminal):
uv run test_mcp_sse.py
```

### Download with Progress Test

Requires Hugging Face token for model downloads:

```bash
# Get a token from: https://huggingface.co/settings/tokens
export HF_TOKEN=hf_your_token_here

# Run the download test
HF_TOKEN=$HF_TOKEN uv run test_mcp_download.py
```

