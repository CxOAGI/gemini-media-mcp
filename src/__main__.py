"""MCP server for Gemini media generation."""

import asyncio
import base64
import json
import logging
import os
import sys
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import aiohttp
from google import genai
from google.cloud import storage
from mcp.server.fastmcp import Context, FastMCP, Image
from mcp.server.session import ServerSession
from mcp.types import TextContent

from .image import ImageModel, ImageSize, MediaResolution
from .image import generate_image as generate_image_impl
from .video import VideoModel
from .video import generate_video as generate_video_impl

logger = logging.getLogger(__name__)


@dataclass
class AppContext:
    """Application context with resources and configuration."""

    images_dir: Path
    videos_dir: Path
    client: genai.Client
    temp_creds_path: Path | None = None
    video_gcs_bucket: str | None = None  # Default GCS bucket for video output


def setup_vertex_credentials() -> Path | None:
    """Setup Vertex AI credentials from service account JSON or environment."""
    if os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() != "true":
        return None

    sa_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    gac = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")

    if not sa_json and gac.strip().startswith("{"):
        sa_json = gac

    if sa_json:
        try:
            data = json.loads(sa_json)
            fd, path_str = tempfile.mkstemp(suffix=".json", prefix="gcp_sa_")
            path = Path(path_str)
            with open(fd, "w") as f:
                json.dump(data, f)
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(path)
            logger.info("Created temp credentials file: %s", path)
            return path
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Failed to setup credentials: %s", e)
            return None

    return None


def cleanup_credentials(path: Path | None) -> None:
    """Clean up temporary credentials file."""
    if path and path.exists():
        try:
            path.unlink()
            logger.info("Cleaned up credentials: %s", path)
        except OSError:
            pass


def check_credentials() -> bool:
    """Check if credentials are configured."""
    if os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() == "true":
        return True
    if os.environ.get("GEMINI_API_KEY"):
        return True
    return False


def create_client() -> genai.Client:
    """Create a Google GenAI client."""
    if os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() == "true":
        return genai.Client(vertexai=True)
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        return genai.Client(api_key=api_key)
    raise RuntimeError("No credentials configured")


def is_running_in_container() -> bool:
    """Check if running inside a container."""
    if os.environ.get("RUNNING_IN_CONTAINER", "").lower() == "true":
        return True
    return Path("/.dockerenv").exists()


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle - setup directories, credentials, and client."""
    if is_running_in_container() and not os.environ.get("DATA_FOLDER"):
        raise ValueError(
            "DATA_FOLDER must be set when running in a container. "
            "Set it to the host path and mount with matching paths, e.g.: "
            "-e DATA_FOLDER=/Users/you/data -v /Users/you/data:/Users/you/data"
        )

    data_folder = Path(os.environ.get("DATA_FOLDER", "data"))
    images_dir = data_folder / "images"
    videos_dir = data_folder / "videos"

    images_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    temp_creds_path = setup_vertex_credentials()
    client = create_client()
    video_gcs_bucket = os.environ.get("VIDEO_GCS_BUCKET")

    try:
        yield AppContext(
            images_dir=images_dir,
            videos_dir=videos_dir,
            client=client,
            temp_creds_path=temp_creds_path,
            video_gcs_bucket=video_gcs_bucket,
        )
    finally:
        cleanup_credentials(temp_creds_path)


async def fetch(uri: str) -> bytes | None:
    """Fetch bytes from URI (gs://, http://, https://, file://)."""
    try:
        if uri.startswith("gs://"):
            parts = uri[5:].split("/", 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid GCS URI: {uri}")
            bucket_name, object_path = parts
            client = storage.Client()
            blob = client.bucket(bucket_name).blob(object_path)
            return await asyncio.to_thread(blob.download_as_bytes)

        if uri.startswith(("http://", "https://")):
            async with aiohttp.ClientSession() as session:
                async with session.get(uri) as resp:
                    if resp.status == 200:
                        return await resp.read()
                    raise ValueError(f"HTTP {resp.status}")

        if uri.startswith("file://"):
            path = Path(uri[7:])
            if path.is_file():
                return path.read_bytes()
            raise ValueError(f"File not found: {path}")

        path = Path(uri)
        if path.is_file():
            return path.read_bytes()

        raise ValueError(f"Unsupported URI: {uri}")
    except Exception as e:
        logger.error("Failed to fetch %s: %s", uri, e)
        return None


# Create MCP server with lifespan
mcp = FastMCP(
    "gemini-media-mcp",
    instructions="MCP server for generating images and videos using Google Gemini and VEO models.",
    lifespan=app_lifespan,
)


@mcp.tool()
async def generate_image(
    ctx: Context[ServerSession, AppContext],
    prompt: str,
    model: ImageModel,
    image_uri: str | None = None,
    image_base64: str | None = None,
    reference_image_uris: list[str] | None = None,
    image_size: ImageSize | None = None,
    media_resolution: MediaResolution | None = None,
    thought_signature_url: str | None = None,
):
    """Generate an image using Google Gemini or Imagen models.

    Args:
        ctx: MCP context with application state
        prompt: Text description of the image to generate
        model: Model to use - options include:
               - "gemini-2.5-flash-image": Gemini 2.5 Flash (fast, creative editing)
               - "gemini-3-pro-image-preview": Gemini 3 Pro (highest quality, 4K, multi-reference)
               - "imagen-3.0-generate-002": Imagen 3 (high quality, text-only)
               - "imagen-4.0-generate-001": Imagen 4 Standard (balanced)
               - "imagen-4.0-ultra-generate-001": Imagen 4 Ultra (highest quality)
               - "imagen-4.0-fast-generate-001": Imagen 4 Fast (fastest)
        image_uri: Input image URI (gs://, http://, file://) for image-to-image
        image_base64: Base64 encoded input image (prefer image_uri)
        reference_image_uris: List of reference image URIs (up to 14 for Gemini 3 Pro).
            Use up to 6 object images for high-fidelity inclusion,
            up to 5 human images for character consistency across scenes.
        image_size: Output image size for Gemini 3 Pro (must use uppercase K):
            - "1K": 1024px
            - "2K": 2048px
            - "4K": 4096px
        media_resolution: Input image processing resolution:
            - "MEDIA_RESOLUTION_LOW": Faster, lower token usage
            - "MEDIA_RESOLUTION_MEDIUM": Balanced
            - "MEDIA_RESOLUTION_HIGH": Best quality, higher token usage
        thought_signature_url: For multi-turn image editing. Pass the thought_signature_url
            from a previous response to continue editing. Example workflow:
            1. First call: generate_image(prompt="Draw a cat") → returns thought_signature_url
            2. Second call: generate_image(prompt="Make it orange", thought_signature_url=<from step 1>)

    Returns:
        JSON with image_url, image_preview, and model info. For Gemini 3 Pro,
        includes thought_signature_url pointing to a file with editing context.
    """
    try:
        app_ctx = ctx.request_context.lifespan_context

        image_bytes = None
        if image_uri:
            image_bytes = await fetch(image_uri)
        elif image_base64:
            image_bytes = base64.b64decode(image_base64)

        # Fetch reference images
        reference_images: list[bytes] = []
        if reference_image_uris:
            for ref_uri in reference_image_uris[:14]:  # Max 14 for Gemini 3 Pro
                ref_bytes = await fetch(ref_uri)
                if ref_bytes:
                    reference_images.append(ref_bytes)

        # Read thought signature from file if URL provided
        thought_signature = None
        if thought_signature_url and thought_signature_url.startswith("file://"):
            sig_path = Path(thought_signature_url[7:])
            if sig_path.exists():
                thought_signature = sig_path.read_text()

        await ctx.info(f"Generating image with model={model}")
        result = await generate_image_impl(
            client=app_ctx.client,
            prompt=prompt,
            images_dir=app_ctx.images_dir,
            model=model,
            image_bytes=image_bytes,
            reference_images=reference_images if reference_images else None,
            image_size=image_size,
            media_resolution=media_resolution,
            thought_signature=thought_signature,
        )
        await ctx.info("Image generated successfully")

        # Build response dict
        response_data: dict[str, Any] = {
            "message": result["message"],
            "image_url": result["image_url"],
            "prompt": result["prompt"],
            "model": result["model"],
        }

        # Include thought_signature_url for multi-turn editing
        if "thought_signature_url" in result:
            response_data["thought_signature_url"] = result["thought_signature_url"]

        # Return image preview and structured JSON response
        preview_b64 = result["image_preview"].split(",")[1]
        preview_bytes = base64.b64decode(preview_b64)
        return [
            Image(data=preview_bytes, format="jpeg"),
            TextContent(
                type="text",
                text=json.dumps(response_data, indent=2),
            ),
        ]
    except Exception as e:
        await ctx.error(f"Image generation failed: {e}")
        logger.exception("Tool error")
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


@mcp.tool()
async def generate_video(
    ctx: Context[ServerSession, AppContext],
    prompt: str,
    model: VideoModel,
    aspect_ratio: str = "16:9",
    duration_seconds: float = 5.0,
    include_audio: bool = False,
    audio_prompt: str | None = None,
    negative_prompt: str | None = None,
    seed: int | None = None,
    image_uri: str | None = None,
    image_base64: str | None = None,
    last_frame_uri: str | None = None,
    last_frame_base64: str | None = None,
    reference_image_uris: list[str] | None = None,
    extend_video_uri: str | None = None,
    output_gcs_uri: str | None = None,
) -> str:
    """Generate a video using Google VEO models.

    Args:
        ctx: MCP context with application state
        prompt: Text description of the video to generate
        model: Model to use - options include:
               - "veo-2.0-generate-001": VEO 2.0 (stable, 5-8s duration, no audio)
               - "veo-3.1-generate-preview": VEO 3.1 (highest quality, 4/6/8s, audio)
               - "veo-3.1-fast-generate-preview": VEO 3.1 Fast (faster, 4/6/8s, audio)
        aspect_ratio: 16:9 (default) or 9:16
        duration_seconds: Video duration (VEO2: 5-8s, VEO3: 4/6/8s)
        include_audio: Enable audio generation (VEO3 only)
        audio_prompt: Audio description (VEO3 only)
        negative_prompt: Things to avoid in the video
        seed: Random seed for reproducibility
        image_uri: First frame image URI for image-to-video
        image_base64: Base64 encoded first frame image (prefer image_uri)
        last_frame_uri: Last frame image URI for first+last frame control (VEO3.1 only).
            When provided with image_uri, generates smooth transition between frames.
        last_frame_base64: Base64 encoded last frame image (prefer last_frame_uri)
        reference_image_uris: List of up to 3 reference image URIs (VEO3.1 only).
            Preserves appearance of a person, character, or product in the video.
            Note: Automatically uses 8-second duration. Cannot combine with first/last frame.
        extend_video_uri: URI of existing VEO-generated video to extend (VEO3.1 only).
            Extends the final second of the video and continues the action.
            Note: Cannot be used together with other image inputs.
            IMPORTANT: Video extension ALWAYS requires output_gcs_uri - extensions produce
            larger combined videos that exceed inline response limits.
        output_gcs_uri: GCS bucket URI for large video output (e.g. gs://bucket/path/).
            Required for video extensions and longer duration videos.

    Returns:
        JSON with video_url and generation details including generation_mode
    """
    try:
        app_ctx = ctx.request_context.lifespan_context

        # Fetch first frame image
        image_bytes = None
        if image_uri:
            image_bytes = await fetch(image_uri)
        elif image_base64:
            image_bytes = base64.b64decode(image_base64)

        # Fetch last frame image (VEO 3.1 first+last frame mode)
        last_frame_bytes = None
        if last_frame_uri:
            last_frame_bytes = await fetch(last_frame_uri)
        elif last_frame_base64:
            last_frame_bytes = base64.b64decode(last_frame_base64)

        # Fetch reference images (VEO 3.1 reference mode)
        reference_images: list[bytes] = []
        if reference_image_uris:
            for ref_uri in reference_image_uris[:3]:  # Max 3 for VEO 3.1
                ref_bytes = await fetch(ref_uri)
                if ref_bytes:
                    reference_images.append(ref_bytes)

        # Use default GCS bucket from env if not provided
        gcs_uri = output_gcs_uri or app_ctx.video_gcs_bucket

        await ctx.info(f"Generating video with model={model}")
        result = await generate_video_impl(
            client=app_ctx.client,
            prompt=prompt,
            videos_dir=app_ctx.videos_dir,
            model=model,
            image_bytes=image_bytes,
            aspect_ratio=aspect_ratio,
            duration_seconds=duration_seconds,
            include_audio=include_audio,
            audio_prompt=audio_prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            log_callback=ctx.info,
            last_frame_bytes=last_frame_bytes,
            reference_images=reference_images if reference_images else None,
            extend_video_uri=extend_video_uri,
            output_gcs_uri=gcs_uri,
        )
        await ctx.info("Video generated successfully")
        return json.dumps(result, indent=2)
    except Exception as e:
        await ctx.error(f"Video generation failed: {e}")
        logger.exception("Tool error")
        return json.dumps({"error": str(e)})


def main() -> None:
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Gemini Media MCP Server")
    parser.add_argument(
        "transport",
        nargs="?",
        default="stdio",
        choices=["stdio", "sse", "streamable-http"],
        help="Transport mode (default: stdio)",
    )
    parser.add_argument(
        "--mount-path",
        default=None,
        help="Mount path for SSE/HTTP transport (e.g., /mcp)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s:%(name)s:%(message)s",
        stream=sys.stderr,
    )

    if not check_credentials():
        logger.error(
            "No credentials configured. Set GEMINI_API_KEY or enable "
            "GOOGLE_GENAI_USE_VERTEXAI=true with appropriate credentials."
        )
        sys.exit(1)

    mcp.run(transport=args.transport, mount_path=args.mount_path)


if __name__ == "__main__":
    main()
