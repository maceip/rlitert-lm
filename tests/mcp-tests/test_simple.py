#!/usr/bin/env python3
"""Simple test to debug MCP connection issues."""

import asyncio
import sys
import traceback
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    server_params = StdioServerParameters(
        command="../../target/release/litert-lm",
        args=["mcp", "--transport", "stdio"],
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                print("✓ Connected and initialized")

                # Try to list tools
                print("\n→ Listing tools...")
                tools = await session.list_tools()
                print(f"✓ Tools: {[t.name for t in tools.tools]}")

                # Keep connection alive for a moment
                await asyncio.sleep(1)

                print("\n✓ Test completed")
    except Exception as e:
        print(f"\n✗ Exception: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Top-level error: {e}")
        traceback.print_exc()
        sys.exit(1)
