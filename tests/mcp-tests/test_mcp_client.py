#!/usr/bin/env python3
"""
Test MCP client that connects to litert-lm MCP server and runs a completion.
Demonstrates listing models, pulling if needed, and running completion.

Usage:
    python test_mcp_client.py
"""

import asyncio
import json
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    # Server parameters - launches litert-lm MCP server via stdio
    # Path relative to tests/mcp-tests directory
    server_params = StdioServerParameters(
        command="../../target/release/litert-lm",
        args=["mcp", "--transport", "stdio"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()
            print("✓ Connected to litert-lm MCP server")

            # List available tools
            tools = await session.list_tools()
            print(f"\n✓ Available tools: {[t.name for t in tools.tools]}")

            # Check what models we have
            print("\n→ Checking downloaded models...")
            result = await session.call_tool("list_models", {"show_all": False})
            models_output = result.content[0].text
            print(f"Downloaded models:\n{models_output}")

            # Use the first available model (prefer already downloaded)
            model_name = "gemma-3n-E4B"
            print(f"\n→ Using model: {model_name}")

            # Run completion asking "what is MCP?"
            print("\n→ Running completion: 'What is MCP?'")
            result = await session.call_tool(
                "run_completion",
                {
                    "model": model_name,
                    "prompt": "What is MCP?",
                    "max_tokens": 2048,
                    "temperature": 0.7
                }
            )

            # Display the response
            response = result.content[0].text
            print(f"\n{'='*60}")
            print("Response:")
            print(f"{'='*60}")
            print(response)
            print(f"{'='*60}")

            print("\n✓ Test completed successfully!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)
