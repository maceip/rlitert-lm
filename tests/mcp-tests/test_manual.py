#!/usr/bin/env python3
"""Manual test to check MCP server stays alive."""

import asyncio
import sys
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    server_params = StdioServerParameters(
        command="../../target/release/litert-lm",
        args=["mcp", "--transport", "stdio"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            print("→ Initializing session...")
            await session.initialize()
            print("✓ Session initialized")

            # Add a small delay to ensure init is complete
            await asyncio.sleep(0.5)

            print("\n→ Listing tools...")
            try:
                tools_result = await session.list_tools()
                print(f"✓ Found {len(tools_result.tools)} tools")
                for tool in tools_result.tools:
                    print(f"  - {tool.name}: {tool.description}")
            except Exception as e:
                print(f"✗ Failed to list tools: {e}")
                raise

            print("\n→ Calling list_models...")
            try:
                result = await session.call_tool("list_models", {"show_all": False})
                print("✓ Result:")
                print(result.content[0].text)
            except Exception as e:
                print(f"✗ Failed to call tool: {e}")
                raise

            print("\n✓ All tests passed!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
