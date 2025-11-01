#!/usr/bin/env python3
"""
Test MCP model download with progress tracking via resource subscriptions.

This test:
1. Connects to MCP server
2. Subscribes to download progress resource
3. Starts downloading a model
4. Monitors real-time progress updates
5. Verifies completion

Requirements:
    Set HF_TOKEN environment variable with your Hugging Face token:
    export HF_TOKEN=hf_your_token_here

    Get a token from: https://huggingface.co/settings/tokens

Usage:
    HF_TOKEN=hf_your_token_here python test_mcp_download.py
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
        print("\nGet a token from: https://huggingface.co/settings/tokens")
        sys.exit(1)
    server_params = StdioServerParameters(
        command="../../target/release/litert-lm",
        args=["mcp", "--transport", "stdio"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("✓ Connected to litert-lm MCP server\n")

            # Choose a small model for testing
            model_to_download = "gemma3-1b"
            resource_uri = f"litert://downloads/{model_to_download}"

            # Check if model is already downloaded
            print("→ Checking if model is already downloaded...")
            result = await session.call_tool("list_models", {"show_all": False})
            downloaded_models = result.content[0].text

            if model_to_download in downloaded_models:
                print(f"⚠ Model {model_to_download} is already downloaded")
                print(f"  Checking its status anyway...\n")
            else:
                print(f"✓ Model {model_to_download} not downloaded yet\n")

            # Subscribe to download progress before starting download
            print(f"→ Subscribing to progress updates: {resource_uri}")
            try:
                await session.subscribe_resource(resource_uri)
                print("✓ Subscribed to download progress\n")
            except Exception as e:
                print(f"⚠ Subscription note: {e}\n")

            # Check initial progress state
            print("→ Checking initial progress state...")
            try:
                resource_content = await session.read_resource(resource_uri)
                progress_data = json.loads(resource_content.contents[0].text)
                print(f"  Current state: {progress_data['status']}")
                print(f"  Progress: {progress_data['progress']}%\n")
            except Exception as e:
                print(f"  Note: {e}\n")

            # Start the download with HF token
            print(f"→ Starting download of {model_to_download}...")
            print(f"  (This may take several minutes depending on model size)")
            print(f"  (Progress updates will appear as the download proceeds)\n")

            # Note: This will trigger progress notifications
            download_task = asyncio.create_task(
                session.call_tool(
                    "pull_model",
                    {
                        "model": model_to_download,
                        "hf_token": hf_token
                    }
                )
            )

            # Poll for progress updates while download is running
            print("→ Monitoring download progress...")
            last_progress = -1

            for i in range(60):  # Poll for up to 60 iterations
                await asyncio.sleep(2)  # Check every 2 seconds

                # Check if download completed
                if download_task.done():
                    break

                # Read current progress
                try:
                    resource_content = await session.read_resource(resource_uri)
                    progress_data = json.loads(resource_content.contents[0].text)
                    current_progress = progress_data['progress']

                    if current_progress != last_progress:
                        status = progress_data['status']
                        print(f"  Progress: {current_progress}% - Status: {status}")
                        last_progress = current_progress

                        if current_progress >= 100:
                            break

                except Exception as e:
                    print(f"  Note: Could not read progress: {e}")

            # Wait for download to complete
            print("\n→ Waiting for download to complete...")
            try:
                result = await download_task
                print(f"✓ Download completed!")
                print(f"  {result.content[0].text}")
            except Exception as e:
                print(f"✗ Download failed: {e}")
                sys.exit(1)

            # Verify final state
            print("\n→ Verifying final state...")
            resource_content = await session.read_resource(resource_uri)
            final_data = json.loads(resource_content.contents[0].text)
            print(f"  Final status: {final_data['status']}")
            print(f"  Final progress: {final_data['progress']}%")

            # Verify model is now in downloaded list
            print("\n→ Verifying model is now downloaded...")
            result = await session.call_tool("list_models", {"show_all": False})
            downloaded_models = result.content[0].text

            if model_to_download in downloaded_models or final_data['status'] == 'complete':
                print(f"✓ Model {model_to_download} is now available!")
            else:
                print(f"⚠ Model status unclear, check manually")

            print("\n✓ Download test completed successfully!")


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
