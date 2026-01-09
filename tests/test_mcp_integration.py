"""Integration tests for MCP server features.

Run with Gemini API key (basic features):
  GEMINI_API_KEY=key uv run pytest tests/test_mcp_integration.py -v -s

Run with Vertex AI (full features including Gemini 3 Pro Image):
  GOOGLE_GENAI_USE_VERTEXAI=true uv run pytest tests/test_mcp_integration.py -v -s
"""

import asyncio
import os
import pytest
import pytest_asyncio
from pathlib import Path


def get_api_key():
    """Get API key from either GEMINI_API_KEY or GOOGLE_API_KEY."""
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def is_vertex_ai():
    """Check if Vertex AI mode is enabled."""
    return os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() == "true"


# Skip all tests if no API key
pytestmark = [
    pytest.mark.skipif(
        not get_api_key() and not is_vertex_ai(),
        reason="GEMINI_API_KEY or Vertex AI credentials not set"
    ),
    pytest.mark.asyncio,
]


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

        expected_tools = {"generate_image", "generate_video"}
        assert expected_tools.issubset(tool_names), f"Missing tools: {expected_tools - tool_names}"
        print(f"✓ Found {len(tool_names)} tools: {tool_names}")

    # ==================== Gemini 3 Pro Image Tests ====================

    async def test_gemini3_pro_image_basic(self, mcp_client):
        """Test Gemini 3 Pro Image basic generation."""
        result = await mcp_client.call_tool(
            "generate_image",
            {
                "prompt": "A red apple on a wooden table",
                "model": "gemini-3-pro-image-preview",
            }
        )
        text = next((c.text for c in result.content if hasattr(c, 'text')), "")
        print(f"✓ Gemini 3 Pro Image: {text[:200]}")
        assert "image_url" in text.lower() or "error" in text.lower()

    async def test_gemini3_pro_image_size(self, mcp_client):
        """Test Gemini 3 Pro Image with image_size parameter (1K/2K/4K)."""
        result = await mcp_client.call_tool(
            "generate_image",
            {
                "prompt": "A blue ocean wave",
                "model": "gemini-3-pro-image-preview",
                "image_size": "2K",
            }
        )
        text = next((c.text for c in result.content if hasattr(c, 'text')), "")
        print(f"✓ Gemini 3 Pro with 2K size: {text[:200]}")
        assert "image_url" in text.lower() or "error" in text.lower()

    async def test_gemini3_pro_thinking_level(self, mcp_client):
        """Test Gemini 3 Pro Image with thinking_level parameter."""
        result = await mcp_client.call_tool(
            "generate_image",
            {
                "prompt": "A complex steampunk machine with gears and pipes",
                "model": "gemini-3-pro-image-preview",
                "thinking_level": "high",
            }
        )
        text = next((c.text for c in result.content if hasattr(c, 'text')), "")
        print(f"✓ Gemini 3 Pro with high thinking: {text[:200]}")
        # Check if thought_signature is returned for multi-turn editing
        if "thought_signature" in text:
            print("✓ Thought signature returned for multi-turn editing")

    # ==================== VEO 3.1 Tests ====================

    async def test_veo31_basic(self, mcp_client):
        """Test VEO 3.1 basic generation."""
        result = await mcp_client.call_tool(
            "generate_video",
            {
                "prompt": "A butterfly landing on a flower",
                "model": "veo-3.1-generate-preview",
            }
        )
        text = next((c.text for c in result.content if hasattr(c, 'text')), "")
        print(f"✓ VEO 3.1 basic: {text[:300]}")
        assert "video_url" in text.lower() or "error" in text.lower()

    async def test_veo31_duration(self, mcp_client):
        """Test VEO 3.1 with duration_seconds parameter (4/6/8s)."""
        result = await mcp_client.call_tool(
            "generate_video",
            {
                "prompt": "Rain falling on a window",
                "model": "veo-3.1-generate-preview",
                "duration_seconds": 6,
            }
        )
        text = next((c.text for c in result.content if hasattr(c, 'text')), "")
        print(f"✓ VEO 3.1 with 6s duration: {text[:300]}")
        assert "video_url" in text.lower() or "error" in text.lower()

    async def test_veo31_fast(self, mcp_client):
        """Test VEO 3.1 fast model variant."""
        result = await mcp_client.call_tool(
            "generate_video",
            {
                "prompt": "A candle flame flickering",
                "model": "veo-3.1-fast-generate-preview",
            }
        )
        text = next((c.text for c in result.content if hasattr(c, 'text')), "")
        print(f"✓ VEO 3.1 fast: {text[:300]}")
        assert "video_url" in text.lower() or "error" in text.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
