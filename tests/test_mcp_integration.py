"""Integration tests for MCP server features.

Run with Gemini API key (basic features):
  GEMINI_API_KEY=key uv run pytest tests/test_mcp_integration.py -v -s

Run with Vertex AI (full features including Gemini 3 Pro Image):
  GOOGLE_GENAI_USE_VERTEXAI=true uv run pytest tests/test_mcp_integration.py -v -s
"""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

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


def _payload(result: Any) -> dict[str, Any]:
    """Parse a tool result's JSON text part.

    Fails the test on a non-JSON body rather than falling back to substring
    matching, which is what let the old assertions accept anything.
    """
    text = next((c.text for c in result.content if hasattr(c, "text")), "")
    assert text, "tool returned no text content"
    return json.loads(text)


def _assert_generated(result: Any, url_key: str) -> dict[str, Any]:
    """Assert the call actually produced media, and show the API error if not.

    These live tests previously asserted `"image_url" in text or "error" in
    text`, which is true of EVERY possible response — success and failure
    alike — so a total API outage passed. This demands the media URL and, on
    failure, surfaces the server's own error message.
    """
    payload = _payload(result)
    assert "error" not in payload, f"tool returned an error: {payload['error']}"
    assert payload.get(url_key), f"no {url_key} in response: {payload}"
    assert str(payload[url_key]).startswith(("file://", "gs://")), payload[url_key]
    return payload


def _assert_refused(result: Any, expected_fragment: str) -> None:
    """Assert the call was refused for the stated reason.

    A bare `"error" in text` passes on ANY failure — including a network
    blip or a bad key — so the reason has to be checked too.
    """
    payload = _payload(result)
    assert "error" in payload, f"expected a refusal, got: {payload}"
    assert expected_fragment.lower() in payload["error"].lower(), payload["error"]


class TestMCPIntegration:
    """Test MCP server via stdio client."""

    async def test_list_tools(self, temp_data_folder):
        """Test that all expected tools are available."""
        async with _mcp_session(temp_data_folder) as session:
            result = await session.list_tools()
            tool_names = {tool.name for tool in result.tools}
            # Exact roster, not a subset: unit tests import the tool functions
            # directly, so a deleted @mcp.tool() registration fails nothing
            # else in the suite. This subset check silently missed five tools
            # as they were added.
            expected_tools = {
                "plan_generation",
                "generate_image",
                "generate_storyboard",
                "generate_video",
                "generate_transition",
                "generate_bridge",
                "generate_clip",
                "generate_video_omni",
                "edit_video",
                "loop_extend",
            }
            assert tool_names == expected_tools, (
                f"Missing: {expected_tools - tool_names} | "
                f"Unexpected: {tool_names - expected_tools}"
            )
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
                    "model": "gemini-3-pro-image",
                },
            )
            payload = _assert_generated(result, "image_url")
            assert payload["model"] == "gemini-3-pro-image"

    @requires_api
    async def test_gemini3_pro_image_size(self, temp_data_folder):
        """Test Gemini 3 Pro Image with image_size parameter (1K/2K/4K)."""
        async with _mcp_session(temp_data_folder) as session:
            result = await session.call_tool(
                "generate_image",
                {
                    "prompt": "A blue ocean wave",
                    "model": "gemini-3-pro-image",
                    "image_size": "2K",
                },
            )
            payload = _assert_generated(result, "image_url")
            # The manifest records the size actually used, so a silently
            # dropped 2K would surface here.
            sidecar = json.loads(Path(payload["sidecar_url"][7:]).read_text())
            assert sidecar["image_size"] == "2K"

    @requires_api
    async def test_gemini3_pro_returns_a_thought_signature(self, temp_data_folder):
        """Multi-turn editing depends on the signature being returned.

        This test used to pass `thinking_level: "high"` — not a parameter of
        generate_image — and assert nothing. MCP leaves additionalProperties
        unset, so the unknown key was silently dropped and the test could
        never fail. It now checks the capability that actually exists.
        """
        async with _mcp_session(temp_data_folder) as session:
            result = await session.call_tool(
                "generate_image",
                {
                    "prompt": "A complex steampunk machine with gears and pipes",
                    "model": "gemini-3-pro-image",
                },
            )
            payload = _assert_generated(result, "image_url")
            signature_url = payload.get("thought_signature_url")
            assert signature_url, f"no thought_signature_url: {payload}"
            assert Path(signature_url[7:]).stat().st_size > 0

    # ==================== Gemini 3.1 Flash Image Tests ====================

    @requires_api
    async def test_gemini31_flash_image_basic(self, temp_data_folder):
        """Test Gemini 3.1 Flash Image basic generation."""
        async with _mcp_session(temp_data_folder) as session:
            result = await session.call_tool(
                "generate_image",
                {
                    "prompt": "A sunset over mountains",
                    "model": "gemini-3.1-flash-image",
                },
            )
            payload = _assert_generated(result, "image_url")
            assert payload["model"] == "gemini-3.1-flash-image"

    @requires_api
    async def test_gemini31_flash_image_size(self, temp_data_folder):
        """Test Gemini 3.1 Flash Image with image_size parameter (1K/2K/4K)."""
        async with _mcp_session(temp_data_folder) as session:
            result = await session.call_tool(
                "generate_image",
                {
                    "prompt": "A city skyline at night",
                    "model": "gemini-3.1-flash-image",
                    "image_size": "2K",
                },
            )
            payload = _assert_generated(result, "image_url")
            assert payload.get("cost") is not None, "a real run must report cost"

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
            payload = _assert_generated(result, "video_url")
            assert payload.get("cost") is not None, "a real run must report cost"

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
            payload = _assert_generated(result, "video_url")
            assert payload.get("cost") is not None, "a real run must report cost"

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
            payload = _assert_generated(result, "video_url")
            assert payload.get("cost") is not None, "a real run must report cost"

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
            payload = _assert_generated(result, "video_url")
            assert payload.get("cost") is not None, "a real run must report cost"

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
            text = next((c.text for c in result.content if hasattr(c, "text")), "")
            print(f"✓ VEO 3.1 Lite rejects extend: {text[:200]}")
            assert "error" in text.lower()
            assert "extension" in text.lower() or "does not support" in text.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
