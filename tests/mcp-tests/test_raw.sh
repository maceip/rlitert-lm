#!/bin/bash
# Raw JSON-RPC test for MCP server

(
  # Initialize
  cat <<'EOF'
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}
EOF
  sleep 0.2

  # Initialized notification
  cat <<'EOF'
{"jsonrpc":"2.0","method":"notifications/initialized"}
EOF
  sleep 0.2

  # List tools
  cat <<'EOF'
{"jsonrpc":"2.0","id":2,"method":"tools/list"}
EOF
  sleep 1

  # Call list_models
  cat <<'EOF'
{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"list_models","arguments":{"show_all":false}}}
EOF
  sleep 2

) | ../../target/release/litert-lm mcp --transport stdio 2>&1 | grep -v "INFO\|ERROR\|DEBUG" | jq -s '.'
