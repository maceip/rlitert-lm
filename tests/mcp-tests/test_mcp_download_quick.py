#!/usr/bin/env python3
"""
Quick test for MCP model download progress tracking.
Starts a download and verifies we get progress updates, then cancels.

Requirements:
    Set HF_TOKEN environment variable with your Hugging Face token:
    export HF_TOKEN=hf_your_token_here

Usage:
    HF_TOKEN=hf_your_token_here python test_mcp_download_quick.py
"""

import asyncio
import sys
import json
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    # Check for HF token
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("✗ Error: HF_TOKEN environment variable not set")
        print("\nPlease set your Hugging Face token:")
        print("  export HF_TOKEN=hf_your_token_here")
        sys.exit(1)

    server_params = StdioServerParameters(
        command="../../target/release/litert-lm",
        args=["mcp", "--transport", "stdio"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("✓ Connected to litert-lm MCP server\n")

            # Get list of available models
            print("→ Listing available models...")
            result = await session.call_tool("list_models", {"show_all": True})
            all_models = result.content[0].text

            # Choose a small model
            model_to_download = "gemma3-1b"
            if model_to_download not in all_models:
                print(f"✗ Model {model_to_download} not in registry")
                sys.exit(1)

            print(f"✓ Will test download of {model_to_download}\n")
            resource_uri = f"litert://downloads/{model_to_download}"

            # Subscribe to progress BEFORE starting download
            print(f"→ Subscribing to {resource_uri}")
            try:
                await session.subscribe_resource(resource_uri)
                print("✓ Subscribed to download progress\n")
            except Exception as e:
                print(f"✗ Subscription failed: {e}")
                sys.exit(1)

            # Start download in background
            print(f"→ Starting download of {model_to_download}...")
            download_task = asyncio.create_task(
                session.call_tool(
                    "pull_model",
                    {
                        "model": model_to_download,
                        "hf_token": hf_token
                    }
                )
            )

            # Poll for progress updates - wait for actual percentage change
            print("→ Waiting for progress updates...\n")
            got_update = False
            got_percentage = False

            for i in range(15):  # Try for 15 seconds
                await asyncio.sleep(1)

                try:
                    resource_content = await session.read_resource(resource_uri)
                    progress_data = json.loads(resource_content.contents[0].text)

                    print(f"  Progress: {progress_data['progress']}% - Status: {progress_data['status']}")

                    # Check for status change
                    if progress_data['status'] != 'pending':
                        got_update = True

                    # Check for percentage change
                    if progress_data['progress'] > 0:
                        got_percentage = True
                        print("\n✓ Got percentage update! Progress tracking is working.")
                        break

                except Exception as e:
                    print(f"  Waiting... ({i+1}s)")

            if not got_percentage and got_update:
                print("\n✓ Got status update (pending → downloading). Progress tracking is working.")

            # Cancel the download
            download_task.cancel()
            try:
                await download_task
            except asyncio.CancelledError:
                print("✓ Download cancelled\n")

            if got_percentage or got_update:
                print("✓ Download progress tracking test PASSED!")
                if got_percentage:
                    print("  - Received percentage updates")
                if got_update:
                    print("  - Received status updates")
            else:
                print("✗ No progress updates received")
                sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
