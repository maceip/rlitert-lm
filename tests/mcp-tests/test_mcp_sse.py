#!/usr/bin/env python3
"""
Test MCP SSE (Server-Sent Events) transport.
This demonstrates using MCP over HTTP with SSE for server-to-client communication.

Usage:
    # Start the MCP server in SSE mode (in another terminal):
    # ../../target/release/litert-lm mcp --transport sse --port 3000

    # Then run this test:
    python test_mcp_sse.py
"""

import asyncio
import sys
from mcp import ClientSession
from mcp.client.sse import sse_client


async def main():
    """Test MCP server over SSE transport."""
    # SSE endpoint configuration
    sse_url = "http://localhost:3000/sse"

    print(f"→ Connecting to MCP server at {sse_url}")

    try:
        async with sse_client(sse_url) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the session
                await session.initialize()
                print("✓ Connected to MCP server via SSE")

                # List available tools
                print("\n→ Listing tools...")
                tools = await session.list_tools()
                print(f"✓ Available tools: {[t.name for t in tools.tools]}")

                # List downloaded models
                print("\n→ Listing downloaded models...")
                result = await session.call_tool("list_models", {"show_all": False})
                print(f"✓ Downloaded models:\n{result.content[0].text}")

                # List all available models
                print("\n→ Listing all available models...")
                result = await session.call_tool("list_models", {"show_all": True})
                print(f"✓ All models:\n{result.content[0].text}")

                # List resources (download progress tracking)
                print("\n→ Listing resources...")
                resources = await session.list_resources()
                print(f"✓ Found {len(resources.resources)} resources")
                for resource in resources.resources[:3]:  # Show first 3
                    print(f"  - {resource.name}")

                # Run a completion with an already-downloaded model
                model_name = "gemma-3n-E4B"
                print(f"\n→ Running completion with {model_name}...")
                result = await session.call_tool(
                    "run_completion",
                    {
                        "model": model_name,
                        "prompt": "What is SSE?",
                        "max_tokens": 150,
                        "temperature": 0.7
                    }
                )

                print("\n" + "="*60)
                print("Response:")
                print("="*60)
                print(result.content[0].text)
                print("="*60)

                print("\n✓ SSE test completed successfully!")

    except ConnectionRefusedError:
        print("\n✗ Error: Could not connect to MCP server")
        print("\nPlease start the server first:")
        print("  cd /Users/rpm/litert-lm")
        print("  ./target/release/litert-lm mcp --transport sse --port 3000")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
