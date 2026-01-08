"""Integration tests for MCP server features.

These tests require GEMINI_API_KEY environment variable to be set.
Run with: GEMINI_API_KEY=your_key uv run pytest tests/test_mcp_integration.py -v -s

Note: Gemini 3 Pro Image and VEO 3.1 advanced features require Vertex AI credentials.
"""

import asyncio
import os
import pytest
import pytest_asyncio
from pathlib import Path


def get_api_key():
    """Get API key from either GEMINI_API_KEY or GOOGLE_API_KEY."""
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


# Skip all tests if no API key
pytestmark = [
    pytest.mark.skipif(
        not get_api_key(),
        reason="GEMINI_API_KEY or GOOGLE_API_KEY not set"
    ),
    pytest.mark.asyncio,
]


@pytest.fixture
def api_key():
    return get_api_key()


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
        # Ensure GEMINI_API_KEY is set (server expects this name)
        if "GEMINI_API_KEY" not in env and "GOOGLE_API_KEY" in env:
            env["GEMINI_API_KEY"] = env["GOOGLE_API_KEY"]

        server_params = StdioServerParameters(
            command="uv",
            args=["run", "gemini-media-mcp", "stdio"],
            env=env,
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session

    async def test_list_tools(self, mcp_client):
        """Test that all expected tools are available."""
        result = await mcp_client.list_tools()
        tool_names = {tool.name for tool in result.tools}

        # These are the actual tools exposed by the MCP server
        expected_tools = {"generate_image", "generate_video"}
        assert expected_tools.issubset(tool_names), f"Missing tools: {expected_tools - tool_names}"
        print(f"✓ Found {len(tool_names)} tools: {tool_names}")

    async def test_generate_image_basic(self, mcp_client):
        """Test basic image generation with gemini-2.0-flash."""
        result = await mcp_client.call_tool(
            "generate_image",
            {
                "prompt": "A simple red circle on white background",
                "model": "gemini-2.5-flash-image",
            }
        )

        assert result.content, "Expected content in response"
        text_content = next((c.text for c in result.content if hasattr(c, 'text')), None)
        assert text_content, "Expected text response"
        print(f"✓ Image generation response: {text_content[:300]}")

        # Check for success (saved image) or expected API error
        is_success = "saved" in text_content.lower() or "image_url" in text_content.lower()
        is_expected_error = "safety" in text_content.lower() or "blocked" in text_content.lower()
        assert is_success or is_expected_error, f"Unexpected response: {text_content}"

    async def test_generate_video_basic(self, mcp_client):
        """Test basic video generation with VEO 2.0."""
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

        # Video generation is async - check for operation started or success
        is_success = "video_url" in text_content.lower() or "generated" in text_content.lower()
        is_expected_error = "quota" in text_content.lower() or "permission" in text_content.lower()
        assert is_success or is_expected_error, f"Unexpected response: {text_content}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
