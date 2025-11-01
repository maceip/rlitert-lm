#!/usr/bin/env python3
"""
Test MCP resource subscriptions - demonstrates subscribing to download progress.

Usage:
    python test_mcp_resources.py
"""

import asyncio
import json
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    # Path relative to tests/mcp-tests directory
    server_params = StdioServerParameters(
        command="../../target/release/litert-lm",
        args=["mcp", "--transport", "stdio"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("✓ Connected to litert-lm MCP server")

            # List available resources
            print("\n→ Listing resources...")
            resources = await session.list_resources()
            print(f"✓ Found {len(resources.resources)} resources")

            for resource in resources.resources[:5]:  # Show first 5
                print(f"  - {resource.uri}: {resource.name}")
                if resource.description:
                    print(f"    {resource.description}")

            # Subscribe to a model's download progress
            model_uri = "litert://downloads/gemma3-1b"
            print(f"\n→ Subscribing to {model_uri}")

            try:
                await session.subscribe_resource(model_uri)
                print("✓ Subscribed successfully")
            except Exception as e:
                print(f"Note: {e}")

            # Read the resource to see current state
            print(f"\n→ Reading resource {model_uri}")
            try:
                resource_content = await session.read_resource(model_uri)
                content = resource_content.contents[0].text
                progress = json.loads(content)
                print(f"✓ Current state:")
                print(f"  Model: {progress['model']}")
                print(f"  Progress: {progress['progress']}%")
                print(f"  Status: {progress['status']}")
            except Exception as e:
                print(f"Note: {e}")

            # List all models in registry
            print("\n→ Listing all available models...")
            result = await session.call_tool("list_models", {"show_all": True})
            print(result.content[0].text)

            print("\n✓ Resource test completed!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)
