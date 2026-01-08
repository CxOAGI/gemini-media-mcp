"""Integration tests for MCP server features.

These tests require GOOGLE_API_KEY environment variable to be set.
Run with: uv run pytest tests/test_mcp_integration.py -v -s
"""

import asyncio
import os
import pytest
import pytest_asyncio
from pathlib import Path

# Skip all tests if no API key
pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("GOOGLE_API_KEY"),
        reason="GOOGLE_API_KEY not set"
    ),
    pytest.mark.asyncio,
]


@pytest.fixture
def api_key():
    return os.environ.get("GOOGLE_API_KEY")


@pytest.fixture
def temp_data_folder(tmp_path):
    """Create temp data folder structure."""
    images_dir = tmp_path / "images"
    videos_dir = tmp_path / "videos"
    images_dir.mkdir()
    videos_dir.mkdir()
    return tmp_path


class TestMCPIntegration:
    """Test MCP server via stdio client."""

    @pytest_asyncio.fixture
    async def mcp_client(self, temp_data_folder):
        """Create MCP client connected to server."""
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        env = os.environ.copy()
        env["DATA_FOLDER"] = str(temp_data_folder)

        server_params = StdioServerParameters(
            command="uv",
            args=["run", "gemini-media-mcp", "stdio"],
            env=env,
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session

    @pytest.mark.asyncio
    async def test_list_tools(self, mcp_client):
        """Test that all expected tools are available."""
        result = await mcp_client.list_tools()
        tool_names = {tool.name for tool in result.tools}

        expected_tools = {
            "generate_image",
            "edit_image",
            "generate_video",
            "check_video_status",
        }
        assert expected_tools.issubset(tool_names), f"Missing tools: {expected_tools - tool_names}"
        print(f"✓ Found {len(tool_names)} tools: {tool_names}")

    @pytest.mark.asyncio
    async def test_generate_image_basic(self, mcp_client):
        """Test basic image generation."""
        result = await mcp_client.call_tool(
            "generate_image",
            {"prompt": "A simple red circle on white background"}
        )

        assert result.content, "Expected content in response"
        text_content = next((c.text for c in result.content if hasattr(c, 'text')), None)
        assert text_content, "Expected text response"
        assert "error" not in text_content.lower() or "saved" in text_content.lower()
        print(f"✓ Image generation response: {text_content[:200]}")

    @pytest.mark.asyncio
    async def test_generate_image_with_size(self, mcp_client):
        """Test image generation with size parameter (Gemini 3 Pro feature)."""
        result = await mcp_client.call_tool(
            "generate_image",
            {
                "prompt": "A blue square",
                "model": "gemini-3-pro-image-preview",
                "image_size": "1K",
            }
        )

        assert result.content, "Expected content in response"
        text_content = next((c.text for c in result.content if hasattr(c, 'text')), None)
        print(f"✓ Image with size response: {text_content[:200] if text_content else 'No text'}")

    @pytest.mark.asyncio
    async def test_generate_image_with_thinking(self, mcp_client):
        """Test image generation with thinking level (Gemini 3 Pro feature)."""
        result = await mcp_client.call_tool(
            "generate_image",
            {
                "prompt": "A complex scene: a cat wearing a hat sitting on a chair",
                "model": "gemini-3-pro-image-preview",
                "thinking_level": "high",
            }
        )

        assert result.content, "Expected content in response"
        text_content = next((c.text for c in result.content if hasattr(c, 'text')), None)
        print(f"✓ Image with thinking response: {text_content[:200] if text_content else 'No text'}")

        # Check if thought_signature is returned
        if text_content and "thought_signature" in text_content:
            print("✓ Thought signature returned for multi-turn editing")

    @pytest.mark.asyncio
    async def test_generate_video_basic(self, mcp_client):
        """Test basic video generation."""
        result = await mcp_client.call_tool(
            "generate_video",
            {
                "prompt": "A spinning red cube",
                "model": "veo-2.0-generate-001",
            }
        )

        assert result.content, "Expected content in response"
        text_content = next((c.text for c in result.content if hasattr(c, 'text')), None)
        assert text_content, "Expected text response"
        print(f"✓ Video generation response: {text_content[:300]}")

    @pytest.mark.asyncio
    async def test_generate_video_veo31_duration(self, mcp_client):
        """Test VEO 3.1 with duration parameter."""
        result = await mcp_client.call_tool(
            "generate_video",
            {
                "prompt": "Ocean waves on a beach",
                "model": "veo-3.1-generate-preview",
                "duration_seconds": 6,
            }
        )

        assert result.content, "Expected content in response"
        text_content = next((c.text for c in result.content if hasattr(c, 'text')), None)
        print(f"✓ VEO 3.1 response: {text_content[:300] if text_content else 'No text'}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
