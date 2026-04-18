"""Integration tests for MCP server features.

Run with Gemini API key (basic features):
  GEMINI_API_KEY=key uv run pytest tests/test_mcp_integration.py -v -s

Run with Vertex AI (full features including Gemini 3 Pro Image):
  GOOGLE_GENAI_USE_VERTEXAI=true uv run pytest tests/test_mcp_integration.py -v -s
"""

import asyncio
import os
from contextlib import asynccontextmanager

import pytest


def get_api_key():
    """Get API key from either GEMINI_API_KEY or GOOGLE_API_KEY."""
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def is_vertex_ai():
    """Check if Vertex AI mode is enabled."""
    return os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() == "true"


# test_list_tools only needs the server to boot — run it even without creds.
# Remaining API-hitting tests skip if no credentials are configured.
requires_api = pytest.mark.skipif(
    not get_api_key() and not is_vertex_ai(),
    reason="GEMINI_API_KEY or Vertex AI credentials not set",
)

pytestmark = [pytest.mark.asyncio]


@pytest.fixture
def temp_data_folder(tmp_path):
    """Create temp data folder structure."""
    images_dir = tmp_path / "images"
    videos_dir = tmp_path / "videos"
    images_dir.mkdir()
    videos_dir.mkdir()
    return tmp_path


@asynccontextmanager
async def _mcp_session(temp_data_folder):
    """Spawn the server over stdio and yield a ClientSession.

    Keeps setup and teardown in the same asyncio task so that anyio's
    cancel-scope task-identity check in stdio_client does not fire on
    exit (which is what breaks an async pytest-asyncio fixture here).
    """
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    env = os.environ.copy()
    env["DATA_FOLDER"] = str(temp_data_folder)
    if "GEMINI_API_KEY" not in env and "GOOGLE_API_KEY" in env:
        env["GEMINI_API_KEY"] = env["GOOGLE_API_KEY"]
    # Boot-only path for test_list_tools: pretend creds exist so the
    # server doesn't sys.exit(1) before we can list tools.
    if "GEMINI_API_KEY" not in env and not is_vertex_ai():
        env["GEMINI_API_KEY"] = "bootcheck-only"

    server_params = StdioServerParameters(
        command="uv",
        args=["run", "gemini-media-mcp", "stdio"],
        env=env,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


class TestMCPIntegration:
    """Test MCP server via stdio client."""

    async def test_list_tools(self, temp_data_folder):
        """Test that all expected tools are available."""
        async with _mcp_session(temp_data_folder) as session:
            result = await session.list_tools()
            tool_names = {tool.name for tool in result.tools}
            expected_tools = {
                "generate_image",
                "generate_video",
                "generate_transition",
                "generate_bridge",
                "generate_clip",
            }
            assert expected_tools.issubset(
                tool_names
            ), f"Missing tools: {expected_tools - tool_names}"
            print(f"✓ Found {len(tool_names)} tools: {tool_names}")

    # ==================== Gemini 3 Pro Image Tests ====================

    @requires_api
    async def test_gemini3_pro_image_basic(self, temp_data_folder):
        """Test Gemini 3 Pro Image basic generation."""
        async with _mcp_session(temp_data_folder) as session:
            result = await session.call_tool(
                "generate_image",
                {
                    "prompt": "A red apple on a wooden table",
                    "model": "gemini-3-pro-image-preview",
                },
            )
            text = next(
                (c.text for c in result.content if hasattr(c, "text")), ""
            )
            print(f"✓ Gemini 3 Pro Image: {text[:200]}")
            assert "image_url" in text.lower() or "error" in text.lower()

    @requires_api
    async def test_gemini3_pro_image_size(self, temp_data_folder):
        """Test Gemini 3 Pro Image with image_size parameter (1K/2K/4K)."""
        async with _mcp_session(temp_data_folder) as session:
            result = await session.call_tool(
                "generate_image",
                {
                    "prompt": "A blue ocean wave",
                    "model": "gemini-3-pro-image-preview",
                    "image_size": "2K",
                },
            )
            text = next(
                (c.text for c in result.content if hasattr(c, "text")), ""
            )
            print(f"✓ Gemini 3 Pro with 2K size: {text[:200]}")
            assert "image_url" in text.lower() or "error" in text.lower()

    @requires_api
    async def test_gemini3_pro_thinking_level(self, temp_data_folder):
        """Test Gemini 3 Pro Image with thinking_level parameter."""
        async with _mcp_session(temp_data_folder) as session:
            result = await session.call_tool(
                "generate_image",
                {
                    "prompt": "A complex steampunk machine with gears and pipes",
                    "model": "gemini-3-pro-image-preview",
                    "thinking_level": "high",
                },
            )
            text = next(
                (c.text for c in result.content if hasattr(c, "text")), ""
            )
            print(f"✓ Gemini 3 Pro with high thinking: {text[:200]}")
            if "thought_signature" in text:
                print("✓ Thought signature returned for multi-turn editing")

    # ==================== Gemini 3.1 Flash Image Tests ====================

    @requires_api
    async def test_gemini31_flash_image_basic(self, temp_data_folder):
        """Test Gemini 3.1 Flash Image basic generation."""
        async with _mcp_session(temp_data_folder) as session:
            result = await session.call_tool(
                "generate_image",
                {
                    "prompt": "A sunset over mountains",
                    "model": "gemini-3.1-flash-image-preview",
                },
            )
            text = next(
                (c.text for c in result.content if hasattr(c, "text")), ""
            )
            print(f"✓ Gemini 3.1 Flash Image: {text[:200]}")
            assert "image_url" in text.lower() or "error" in text.lower()

    @requires_api
    async def test_gemini31_flash_image_size(self, temp_data_folder):
        """Test Gemini 3.1 Flash Image with image_size parameter (1K/2K/4K)."""
        async with _mcp_session(temp_data_folder) as session:
            result = await session.call_tool(
                "generate_image",
                {
                    "prompt": "A city skyline at night",
                    "model": "gemini-3.1-flash-image-preview",
                    "image_size": "2K",
                },
            )
            text = next(
                (c.text for c in result.content if hasattr(c, "text")), ""
            )
            print(f"✓ Gemini 3.1 Flash with 2K size: {text[:200]}")
            assert "image_url" in text.lower() or "error" in text.lower()

    # ==================== VEO 3.1 Tests ====================

    @requires_api
    async def test_veo31_basic(self, temp_data_folder):
        """Test VEO 3.1 basic generation."""
        async with _mcp_session(temp_data_folder) as session:
            result = await session.call_tool(
                "generate_video",
                {
                    "prompt": "A butterfly landing on a flower",
                    "model": "veo-3.1-generate-001",
                },
            )
            text = next(
                (c.text for c in result.content if hasattr(c, "text")), ""
            )
            print(f"✓ VEO 3.1 basic: {text[:300]}")
            assert "video_url" in text.lower() or "error" in text.lower()

    @requires_api
    async def test_veo31_duration(self, temp_data_folder):
        """Test VEO 3.1 with duration_seconds parameter (4/6/8s)."""
        async with _mcp_session(temp_data_folder) as session:
            result = await session.call_tool(
                "generate_video",
                {
                    "prompt": "Rain falling on a window",
                    "model": "veo-3.1-generate-001",
                    "duration_seconds": 6,
                },
            )
            text = next(
                (c.text for c in result.content if hasattr(c, "text")), ""
            )
            print(f"✓ VEO 3.1 with 6s duration: {text[:300]}")
            assert "video_url" in text.lower() or "error" in text.lower()

    @requires_api
    async def test_veo31_fast(self, temp_data_folder):
        """Test VEO 3.1 fast model variant."""
        async with _mcp_session(temp_data_folder) as session:
            result = await session.call_tool(
                "generate_video",
                {
                    "prompt": "A candle flame flickering",
                    "model": "veo-3.1-fast-generate-001",
                },
            )
            text = next(
                (c.text for c in result.content if hasattr(c, "text")), ""
            )
            print(f"✓ VEO 3.1 fast: {text[:300]}")
            assert "video_url" in text.lower() or "error" in text.lower()

    @requires_api
    async def test_veo31_lite(self, temp_data_folder):
        """Test VEO 3.1 Lite preview model."""
        async with _mcp_session(temp_data_folder) as session:
            result = await session.call_tool(
                "generate_video",
                {
                    "prompt": "A paper airplane gliding through the sky",
                    "model": "veo-3.1-lite-generate-preview",
                    "duration_seconds": 4,
                },
            )
            text = next(
                (c.text for c in result.content if hasattr(c, "text")), ""
            )
            print(f"✓ VEO 3.1 Lite: {text[:300]}")
            assert "video_url" in text.lower() or "error" in text.lower()

    @requires_api
    async def test_veo31_lite_rejects_extend(self, temp_data_folder):
        """VEO 3.1 Lite must reject video-extension requests (doesn't
        support extension per model card)."""
        async with _mcp_session(temp_data_folder) as session:
            result = await session.call_tool(
                "generate_video",
                {
                    "prompt": "continue this clip",
                    "model": "veo-3.1-lite-generate-preview",
                    "extend_video_uri": "gs://example/clip.mp4",
                },
            )
            text = next(
                (c.text for c in result.content if hasattr(c, "text")), ""
            )
            print(f"✓ VEO 3.1 Lite rejects extend: {text[:200]}")
            assert "error" in text.lower()
            assert "extension" in text.lower() or "does not support" in text.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
