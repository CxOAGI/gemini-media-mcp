"""MCP server for Gemini media generation."""

import asyncio
import base64
import ipaddress
import json
import logging
import os
import socket
import sys
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import aiohttp
from google import genai
from google.cloud import storage
from mcp.server.fastmcp import Context, FastMCP, Image
from mcp.server.session import ServerSession
from mcp.types import TextContent
from PIL import Image as PILImage

from .image import ImageModel, ImageSize, MediaResolution
from .image import generate_image as generate_image_impl
from .video import VideoModel
from .video import generate_video as generate_video_impl
from .video_utils import extract_frame_png

logger = logging.getLogger(__name__)

# 50 MB cap on any single fetch to prevent memory/disk exhaustion.
MAX_FETCH_BYTES = 50 * 1024 * 1024

# Cap decoded pixel count to prevent decompression-bomb DoS on input images.
# 50 megapixels fits up to ~7000x7000; well above any realistic user input.
PILImage.MAX_IMAGE_PIXELS = 50_000_000


@dataclass
class AppContext:
    """Application context with resources and configuration."""

    data_folder: Path
    images_dir: Path
    videos_dir: Path
    client: genai.Client
    temp_creds_path: Path | None = None
    video_gcs_bucket: str | None = None  # Default GCS bucket for video output
    allowed_gcs_buckets: frozenset[str] = frozenset()  # Allowlist for gs:// URIs


def _parse_gcs_bucket(uri: str) -> str | None:
    """Return the bucket name from a gs:// URI, or None if invalid."""
    if not uri.startswith("gs://"):
        return None
    remainder = uri[5:].split("/", 1)[0]
    return remainder or None


def _write_sidecar(media_url: str, metadata: dict[str, Any]) -> str | None:
    """Write <stem>.json next to a file:// media URL and return its file:// URL.

    For remote media (gs://), no sidecar is written locally. Returns None in
    that case so callers can still include the manifest inline in the
    response.
    """
    if not media_url.startswith("file://"):
        return None
    media_path = Path(media_url[7:])
    sidecar = media_path.with_suffix(".json")
    try:
        sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    except OSError as e:
        logger.warning("Failed to write sidecar %s: %s", sidecar, e)
        return None
    return f"file://{sidecar}"


def _compute_allowed_gcs_buckets() -> frozenset[str]:
    """Build the allowlist from GCS_ALLOWED_BUCKETS and VIDEO_GCS_BUCKET."""
    raw = os.environ.get("GCS_ALLOWED_BUCKETS", "")
    buckets = {b.strip() for b in raw.split(",") if b.strip()}
    video_bucket = os.environ.get("VIDEO_GCS_BUCKET", "").strip()
    if video_bucket:
        parsed = _parse_gcs_bucket(video_bucket) or video_bucket
        buckets.add(parsed)
    return frozenset(buckets)


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
            # mkstemp creates 0600 on POSIX; make it explicit for clarity.
            try:
                os.chmod(path, 0o600)
            except OSError:
                pass
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
    allowed_gcs_buckets = _compute_allowed_gcs_buckets()

    try:
        yield AppContext(
            data_folder=data_folder,
            images_dir=images_dir,
            videos_dir=videos_dir,
            client=client,
            temp_creds_path=temp_creds_path,
            video_gcs_bucket=video_gcs_bucket,
            allowed_gcs_buckets=allowed_gcs_buckets,
        )
    finally:
        cleanup_credentials(temp_creds_path)


def _validate_local_path(path: Path, allowed_dir: Path) -> Path:
    """Validate that a local file path is within the allowed directory.

    Prevents arbitrary file read (LFI) by resolving symlinks and ensuring
    the path is inside the data folder.
    """
    resolved = path.resolve()
    allowed = allowed_dir.resolve()
    if not str(resolved).startswith(str(allowed) + os.sep) and resolved != allowed:
        raise ValueError(
            f"Access denied: '{path}' is outside the allowed directory '{allowed_dir}'. "
            f"Only files within DATA_FOLDER are accessible."
        )
    if not resolved.is_file():
        raise ValueError(f"File not found: {path}")
    return resolved


def _assert_http_host_public(url: str) -> None:
    """Reject http(s) URLs whose host resolves to a private or loopback IP.

    Mitigates SSRF against cloud metadata (169.254.169.254), localhost, or
    internal networks. Does not protect against DNS rebinding between this
    check and the actual request — acceptable for single-shot fetches.
    """
    parsed = urlparse(url)
    host = parsed.hostname
    if not host:
        raise ValueError(f"URL missing host: {url}")
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise ValueError(f"Could not resolve host '{host}': {e}") from e
    for info in infos:
        addr = info[4][0]
        ip = ipaddress.ip_address(addr)
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            raise ValueError(
                f"Refusing to fetch non-public address: {host} -> {addr}"
            )


def _assert_gcs_bucket_allowed(bucket: str, allowed: frozenset[str]) -> None:
    """Reject gs:// buckets not in the allowlist, when one is configured.

    If no allowlist is configured (empty set), defers to ambient credentials
    and logs a warning. This preserves backward compatibility but is noisy
    so operators notice.
    """
    if not allowed:
        logger.warning(
            "gs:// fetch with no allowlist configured; set GCS_ALLOWED_BUCKETS "
            "or VIDEO_GCS_BUCKET to restrict access. Bucket: %s",
            bucket,
        )
        return
    if bucket not in allowed:
        raise ValueError(
            f"GCS bucket '{bucket}' is not in the allowlist. "
            f"Configured buckets: {sorted(allowed)}"
        )


async def _read_capped_http(resp: aiohttp.ClientResponse, limit: int) -> bytes:
    """Read an HTTP response body with a hard size cap."""
    if resp.content_length is not None and resp.content_length > limit:
        raise ValueError(
            f"Response Content-Length {resp.content_length} exceeds cap {limit}"
        )
    chunks: list[bytes] = []
    total = 0
    async for chunk in resp.content.iter_chunked(64 * 1024):
        total += len(chunk)
        if total > limit:
            raise ValueError(f"Response body exceeded size cap of {limit} bytes")
        chunks.append(chunk)
    return b"".join(chunks)


async def fetch(
    uri: str,
    allowed_dir: Path | None = None,
    allowed_gcs_buckets: frozenset[str] = frozenset(),
    max_bytes: int = MAX_FETCH_BYTES,
) -> bytes | None:
    """Fetch bytes from URI (gs://, http://, https://, file://).

    - Local file access (file:// and bare paths) is restricted to allowed_dir.
    - http(s):// hosts must resolve to public IPs (SSRF guard).
    - gs:// buckets must be in allowed_gcs_buckets when configured.
    - All responses are capped at max_bytes.
    """
    try:
        if uri.startswith("gs://"):
            parts = uri[5:].split("/", 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid GCS URI: {uri}")
            bucket_name, object_path = parts
            _assert_gcs_bucket_allowed(bucket_name, allowed_gcs_buckets)
            client = storage.Client()
            blob = client.bucket(bucket_name).blob(object_path)

            def _download_capped() -> bytes:
                blob.reload()
                if blob.size is not None and blob.size > max_bytes:
                    raise ValueError(
                        f"GCS object size {blob.size} exceeds cap {max_bytes}"
                    )
                return blob.download_as_bytes(start=0, end=max_bytes)

            return await asyncio.to_thread(_download_capped)

        if uri.startswith(("http://", "https://")):
            _assert_http_host_public(uri)
            timeout = aiohttp.ClientTimeout(total=60)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(uri) as resp:
                    if resp.status == 200:
                        return await _read_capped_http(resp, max_bytes)
                    raise ValueError(f"HTTP {resp.status}")

        # Local file access — require allowed_dir
        if allowed_dir is None:
            raise ValueError(
                f"Local file access is not allowed without a configured DATA_FOLDER: {uri}"
            )

        if uri.startswith("file://"):
            path = _validate_local_path(Path(uri[7:]), allowed_dir)
            if path.stat().st_size > max_bytes:
                raise ValueError(f"File exceeds size cap {max_bytes}: {path}")
            return path.read_bytes()

        path = Path(uri)
        if path.exists():
            validated = _validate_local_path(path, allowed_dir)
            if validated.stat().st_size > max_bytes:
                raise ValueError(f"File exceeds size cap {max_bytes}: {validated}")
            return validated.read_bytes()

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
               - "gemini-3.1-flash-image-preview": Gemini 3.1 Flash (fast, 4K, multi-reference)
               - "imagen-3.0-generate-002": Imagen 3 (high quality, text-only)
               - "imagen-4.0-generate-001": Imagen 4 Standard (balanced)
               - "imagen-4.0-ultra-generate-001": Imagen 4 Ultra (highest quality)
               - "imagen-4.0-fast-generate-001": Imagen 4 Fast (fastest)
        image_uri: Input image URI (gs://, http://, file://) for image-to-image
        image_base64: Base64 encoded input image (prefer image_uri)
        reference_image_uris: List of reference image URIs (up to 14 for Gemini 3.x image models).
            Use up to 6 object images for high-fidelity inclusion,
            up to 5 human images for character consistency across scenes.
        image_size: Output image size for Gemini 3.x image models (must use uppercase K):
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
        JSON with image_url, image_preview, and model info. For Gemini 3.x image models,
        includes thought_signature_url pointing to a file with editing context.
    """
    try:
        app_ctx = ctx.request_context.lifespan_context
        data_dir = app_ctx.data_folder

        image_bytes = None
        if image_uri:
            image_bytes = await fetch(
                image_uri,
                allowed_dir=data_dir,
                allowed_gcs_buckets=app_ctx.allowed_gcs_buckets,
            )
        elif image_base64:
            image_bytes = base64.b64decode(image_base64)

        # Fetch reference images
        reference_images: list[bytes] = []
        if reference_image_uris:
            for ref_uri in reference_image_uris[:14]:  # Max 14 for Gemini 3.x
                ref_bytes = await fetch(
                    ref_uri,
                    allowed_dir=data_dir,
                    allowed_gcs_buckets=app_ctx.allowed_gcs_buckets,
                )
                if ref_bytes:
                    reference_images.append(ref_bytes)

        # Read thought signature from file if URL provided
        thought_signature = None
        if thought_signature_url and thought_signature_url.startswith("file://"):
            sig_path = Path(thought_signature_url[7:])
            validated_sig = _validate_local_path(sig_path, data_dir)
            thought_signature = validated_sig.read_text()

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

        # Write sidecar manifest so downstream tools (e.g. vfx-mcp) can read
        # generation parameters without parsing response JSON.
        manifest: dict[str, Any] = {
            "kind": "image",
            "prompt": prompt,
            "model": model,
            "image_url": result["image_url"],
            "image_size": image_size,
            "media_resolution": media_resolution,
            "reference_image_uris": reference_image_uris,
            "source_image_uri": image_uri,
            "thought_signature_url": result.get("thought_signature_url"),
        }
        sidecar_url = _write_sidecar(result["image_url"], manifest)
        if sidecar_url:
            response_data["sidecar_url"] = sidecar_url

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
               - "veo-3.1-generate-001": VEO 3.1 (highest quality, 4/6/8s, audio)
               - "veo-3.1-fast-generate-001": VEO 3.1 Fast (faster, 4/6/8s, audio)
               - "veo-3.1-lite-generate-preview": VEO 3.1 Lite (most cost-effective,
                 4/6/8s, audio; does NOT support video extension or 4K).
                 Availability note: as of launch, Lite is served via the Gemini
                 API / AI Studio; Vertex AI projects may return 404 until the
                 model is published there.
        aspect_ratio: 16:9 (default) or 9:16
        duration_seconds: Video duration (4/6/8s)
        include_audio: Enable audio generation
        audio_prompt: Audio description
        negative_prompt: Things to avoid in the video
        seed: Random seed for reproducibility
        image_uri: First frame image URI for image-to-video
        image_base64: Base64 encoded first frame image (prefer image_uri)
        last_frame_uri: Last frame image URI for first+last frame control.
            When provided with image_uri, generates smooth transition between frames.
        last_frame_base64: Base64 encoded last frame image (prefer last_frame_uri)
        reference_image_uris: List of up to 3 reference image URIs.
            Preserves appearance of a person, character, or product in the video.
            Note: Automatically uses 8-second duration. Cannot combine with first/last frame.
        extend_video_uri: URI of existing VEO-generated video to extend.
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
        data_dir = app_ctx.data_folder

        # Fetch first frame image
        image_bytes = None
        if image_uri:
            image_bytes = await fetch(
                image_uri,
                allowed_dir=data_dir,
                allowed_gcs_buckets=app_ctx.allowed_gcs_buckets,
            )
        elif image_base64:
            image_bytes = base64.b64decode(image_base64)

        # Fetch last frame image (VEO 3.1 first+last frame mode)
        last_frame_bytes = None
        if last_frame_uri:
            last_frame_bytes = await fetch(
                last_frame_uri,
                allowed_dir=data_dir,
                allowed_gcs_buckets=app_ctx.allowed_gcs_buckets,
            )
        elif last_frame_base64:
            last_frame_bytes = base64.b64decode(last_frame_base64)

        # Fetch reference images (VEO 3.1 reference mode)
        reference_images: list[bytes] = []
        if reference_image_uris:
            for ref_uri in reference_image_uris[:3]:  # Max 3 for VEO 3.1
                ref_bytes = await fetch(
                    ref_uri,
                    allowed_dir=data_dir,
                    allowed_gcs_buckets=app_ctx.allowed_gcs_buckets,
                )
                if ref_bytes:
                    reference_images.append(ref_bytes)

        # Use default GCS bucket from env if not provided; validate against allowlist.
        gcs_uri = output_gcs_uri or app_ctx.video_gcs_bucket
        if gcs_uri:
            bucket = _parse_gcs_bucket(gcs_uri)
            if bucket is None:
                raise ValueError(f"output_gcs_uri must start with gs://: {gcs_uri}")
            if app_ctx.allowed_gcs_buckets and bucket not in app_ctx.allowed_gcs_buckets:
                raise ValueError(
                    f"output_gcs_uri bucket '{bucket}' is not in the allowlist. "
                    f"Configured: {sorted(app_ctx.allowed_gcs_buckets)}"
                )

        await ctx.info(f"Generating video with model={model}")
        result = await generate_video_impl(
            client=app_ctx.client,
            prompt=prompt,
            videos_dir=app_ctx.videos_dir,
            model=model,
            image_bytes=image_bytes,
            allowed_dir=data_dir,
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

        # Write sidecar manifest alongside local video output.
        manifest: dict[str, Any] = {
            "kind": "video",
            "prompt": prompt,
            "audio_prompt": audio_prompt,
            "negative_prompt": negative_prompt,
            "model": model,
            "aspect_ratio": aspect_ratio,
            "duration_seconds": result.get("duration_seconds", duration_seconds),
            "audio_enabled": result.get("audio_enabled", include_audio),
            "generation_mode": result.get("generation_mode"),
            "seed": seed,
            "video_url": result.get("video_url"),
            "source_image_uri": image_uri,
            "last_frame_uri": last_frame_uri,
            "reference_image_uris": reference_image_uris,
            "extend_video_uri": extend_video_uri,
        }
        sidecar_url = _write_sidecar(result.get("video_url", ""), manifest)
        if sidecar_url:
            result["sidecar_url"] = sidecar_url

        return json.dumps(result, indent=2)
    except Exception as e:
        await ctx.error(f"Video generation failed: {e}")
        logger.exception("Tool error")
        return json.dumps({"error": str(e)})


@mcp.tool()
async def generate_transition(
    ctx: Context[ServerSession, AppContext],
    first_frame_uri: str,
    last_frame_uri: str,
    prompt: str = "smooth cinematic transition between the two frames",
    model: VideoModel = "veo-3.1-fast-generate-001",
    duration_seconds: float = 4.0,
    aspect_ratio: str = "16:9",
    include_audio: bool = False,
    audio_prompt: str | None = None,
    negative_prompt: str | None = None,
    seed: int | None = None,
    output_gcs_uri: str | None = None,
) -> str:
    """Generate a transition video between two still frames using VEO 3.1.

    Intended for agent workflows that combine this MCP with a cutting MCP
    (e.g. vfx-mcp). The cutting MCP extracts the last frame of clip A and the
    first frame of clip B; this tool generates the in-between video using
    VEO 3.1's first+last frame mode.

    Args:
        first_frame_uri: URI of the starting still (gs://, https://, file://)
        last_frame_uri: URI of the ending still (gs://, https://, file://)
        prompt: Description of the transition motion and style
        model: VEO model (defaults to fast; lite is supported too)
        duration_seconds: Transition length (4/6/8s; snapped to nearest)
        aspect_ratio: 16:9 or 9:16 (must match clip aspect for clean cuts)
        include_audio: Generate transitional audio
        audio_prompt: Audio description
        negative_prompt: Things to avoid in the transition
        seed: Random seed for reproducibility
        output_gcs_uri: GCS output URI for large videos

    Returns:
        JSON with video_url, sidecar_url, and generation metadata.
    """
    try:
        app_ctx = ctx.request_context.lifespan_context
        data_dir = app_ctx.data_folder

        first_bytes = await fetch(
            first_frame_uri,
            allowed_dir=data_dir,
            allowed_gcs_buckets=app_ctx.allowed_gcs_buckets,
        )
        last_bytes = await fetch(
            last_frame_uri,
            allowed_dir=data_dir,
            allowed_gcs_buckets=app_ctx.allowed_gcs_buckets,
        )
        if first_bytes is None or last_bytes is None:
            raise ValueError(
                "Could not fetch one or both transition frames. "
                f"first_frame_uri={first_frame_uri}, last_frame_uri={last_frame_uri}"
            )

        gcs_uri = output_gcs_uri or app_ctx.video_gcs_bucket
        if gcs_uri:
            bucket = _parse_gcs_bucket(gcs_uri)
            if bucket is None:
                raise ValueError(f"output_gcs_uri must start with gs://: {gcs_uri}")
            if app_ctx.allowed_gcs_buckets and bucket not in app_ctx.allowed_gcs_buckets:
                raise ValueError(
                    f"output_gcs_uri bucket '{bucket}' is not in the allowlist. "
                    f"Configured: {sorted(app_ctx.allowed_gcs_buckets)}"
                )

        await ctx.info(f"Generating transition with model={model}")
        result = await generate_video_impl(
            client=app_ctx.client,
            prompt=prompt,
            videos_dir=app_ctx.videos_dir,
            model=model,
            image_bytes=first_bytes,
            last_frame_bytes=last_bytes,
            allowed_dir=data_dir,
            aspect_ratio=aspect_ratio,
            duration_seconds=duration_seconds,
            include_audio=include_audio,
            audio_prompt=audio_prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            log_callback=ctx.info,
            output_gcs_uri=gcs_uri,
        )
        await ctx.info("Transition generated successfully")

        manifest: dict[str, Any] = {
            "kind": "transition",
            "prompt": prompt,
            "audio_prompt": audio_prompt,
            "negative_prompt": negative_prompt,
            "model": model,
            "aspect_ratio": aspect_ratio,
            "duration_seconds": result.get("duration_seconds", duration_seconds),
            "audio_enabled": result.get("audio_enabled", include_audio),
            "generation_mode": result.get("generation_mode"),
            "seed": seed,
            "video_url": result.get("video_url"),
            "first_frame_uri": first_frame_uri,
            "last_frame_uri": last_frame_uri,
        }
        sidecar_url = _write_sidecar(result.get("video_url", ""), manifest)
        if sidecar_url:
            result["sidecar_url"] = sidecar_url
        result["first_frame_uri"] = first_frame_uri
        result["last_frame_uri"] = last_frame_uri

        return json.dumps(result, indent=2)
    except Exception as e:
        await ctx.error(f"Transition generation failed: {e}")
        logger.exception("Tool error")
        return json.dumps({"error": str(e)})


@mcp.tool()
async def generate_bridge(
    ctx: Context[ServerSession, AppContext],
    from_clip_uri: str,
    to_clip_uri: str,
    prompt: str = "smooth cinematic cut between the two clips",
    model: VideoModel = "veo-3.1-fast-generate-001",
    duration_seconds: float = 4.0,
    aspect_ratio: str = "16:9",
    include_audio: bool = False,
    audio_prompt: str | None = None,
    negative_prompt: str | None = None,
    seed: int | None = None,
    output_gcs_uri: str | None = None,
) -> str:
    """Generate a short transition video that bridges two existing clips.

    Decodes the last frame of `from_clip_uri` and the first frame of
    `to_clip_uri`, then calls VEO 3.1 first+last frame mode to produce
    an in-between clip. Pair with a cutting MCP (e.g. vfx-mcp) to splice
    the output between the two originals.

    Args:
        from_clip_uri: URI of the clip whose last frame is the start of
            the bridge. gs://, https://, or file:// within DATA_FOLDER.
        to_clip_uri: URI of the clip whose first frame ends the bridge.
        prompt: Description of the transition motion and style.
        model: VEO model (fast by default).
        duration_seconds: Bridge length (4/6/8s, snapped).
        aspect_ratio: Must match source clips for a clean cut.
        include_audio: Generate transitional audio.
        audio_prompt: Audio description.
        negative_prompt: Things to avoid in the bridge.
        seed: Random seed for reproducibility.
        output_gcs_uri: GCS URI for large video output.

    Returns:
        JSON with video_url, sidecar_url, and the source clip URIs.
    """
    try:
        app_ctx = ctx.request_context.lifespan_context
        data_dir = app_ctx.data_folder

        from_bytes = await fetch(
            from_clip_uri,
            allowed_dir=data_dir,
            allowed_gcs_buckets=app_ctx.allowed_gcs_buckets,
        )
        to_bytes = await fetch(
            to_clip_uri,
            allowed_dir=data_dir,
            allowed_gcs_buckets=app_ctx.allowed_gcs_buckets,
        )
        if from_bytes is None or to_bytes is None:
            raise ValueError(
                "Could not fetch one or both bridge clips. "
                f"from_clip_uri={from_clip_uri}, to_clip_uri={to_clip_uri}"
            )

        await ctx.info("Extracting bridge frames")
        first_frame_png = await asyncio.to_thread(
            extract_frame_png, from_bytes, "end"
        )
        last_frame_png = await asyncio.to_thread(
            extract_frame_png, to_bytes, "start"
        )

        gcs_uri = output_gcs_uri or app_ctx.video_gcs_bucket
        if gcs_uri:
            bucket = _parse_gcs_bucket(gcs_uri)
            if bucket is None:
                raise ValueError(f"output_gcs_uri must start with gs://: {gcs_uri}")
            if app_ctx.allowed_gcs_buckets and bucket not in app_ctx.allowed_gcs_buckets:
                raise ValueError(
                    f"output_gcs_uri bucket '{bucket}' is not in the allowlist. "
                    f"Configured: {sorted(app_ctx.allowed_gcs_buckets)}"
                )

        await ctx.info(f"Generating bridge with model={model}")
        result = await generate_video_impl(
            client=app_ctx.client,
            prompt=prompt,
            videos_dir=app_ctx.videos_dir,
            model=model,
            image_bytes=first_frame_png,
            last_frame_bytes=last_frame_png,
            allowed_dir=data_dir,
            aspect_ratio=aspect_ratio,
            duration_seconds=duration_seconds,
            include_audio=include_audio,
            audio_prompt=audio_prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            log_callback=ctx.info,
            output_gcs_uri=gcs_uri,
        )
        await ctx.info("Bridge generated successfully")

        manifest: dict[str, Any] = {
            "kind": "bridge",
            "prompt": prompt,
            "audio_prompt": audio_prompt,
            "negative_prompt": negative_prompt,
            "model": model,
            "aspect_ratio": aspect_ratio,
            "duration_seconds": result.get("duration_seconds", duration_seconds),
            "audio_enabled": result.get("audio_enabled", include_audio),
            "generation_mode": result.get("generation_mode"),
            "seed": seed,
            "video_url": result.get("video_url"),
            "from_clip_uri": from_clip_uri,
            "to_clip_uri": to_clip_uri,
        }
        sidecar_url = _write_sidecar(result.get("video_url", ""), manifest)
        if sidecar_url:
            result["sidecar_url"] = sidecar_url
        result["from_clip_uri"] = from_clip_uri
        result["to_clip_uri"] = to_clip_uri

        return json.dumps(result, indent=2)
    except Exception as e:
        await ctx.error(f"Bridge generation failed: {e}")
        logger.exception("Tool error")
        return json.dumps({"error": str(e)})


@mcp.tool()
async def generate_clip(
    ctx: Context[ServerSession, AppContext],
    beats: list[dict[str, Any]],
    aspect_ratio: str = "9:16",
    model: VideoModel = "veo-3.1-fast-generate-001",
    include_audio: bool = True,
    add_bridges: bool = False,
    output_gcs_uri: str | None = None,
) -> str:
    """Generate a multi-beat short clip — the building block for a reel / short.

    Runs each `beat` through `generate_video` sequentially. When
    `add_bridges=True`, between each pair of beats a transition is
    generated using the last frame of beat N and the first frame of beat
    N+1 (same primitive as generate_bridge, just chained).

    The returned manifest is an ordered list of segments (beats and
    bridges, in playback order) that a downstream cutting MCP can splice
    into a final clip.

    Args:
        beats: Ordered list of beat specs. Each item accepts:
            {prompt: str, duration_seconds?: float, seed?: int,
             first_frame_uri?: str, negative_prompt?: str,
             audio_prompt?: str}
        aspect_ratio: Default 9:16 for vertical social clips.
        model: VEO model applied to every beat.
        include_audio: Enable audio on each beat (only effective on Vertex).
        add_bridges: Generate a bridge clip between consecutive beats.
        output_gcs_uri: GCS URI for all outputs (optional).

    Returns:
        JSON clip manifest:
        {
          "kind": "clip",
          "aspect_ratio": "9:16",
          "segments": [
            {"kind": "beat", "video_url": ..., "sidecar_url": ..., "prompt": ...},
            {"kind": "bridge", "video_url": ..., "sidecar_url": ...},
            ...
          ],
          "total_duration_seconds": <sum>
        }
    """
    try:
        app_ctx = ctx.request_context.lifespan_context
        data_dir = app_ctx.data_folder

        if not beats:
            raise ValueError("beats list must not be empty")

        gcs_uri = output_gcs_uri or app_ctx.video_gcs_bucket
        if gcs_uri:
            bucket = _parse_gcs_bucket(gcs_uri)
            if bucket is None:
                raise ValueError(f"output_gcs_uri must start with gs://: {gcs_uri}")
            if app_ctx.allowed_gcs_buckets and bucket not in app_ctx.allowed_gcs_buckets:
                raise ValueError(
                    f"output_gcs_uri bucket '{bucket}' is not in the allowlist. "
                    f"Configured: {sorted(app_ctx.allowed_gcs_buckets)}"
                )

        segments: list[dict[str, Any]] = []
        total_duration = 0.0
        prev_video_bytes: bytes | None = None

        for idx, beat in enumerate(beats):
            prompt = beat.get("prompt")
            if not prompt:
                raise ValueError(f"beat {idx} missing required 'prompt'")
            duration = float(beat.get("duration_seconds", 4.0))
            seed = beat.get("seed")

            # Optional per-beat first frame (image_to_video).
            first_frame_uri = beat.get("first_frame_uri")
            image_bytes = None
            if first_frame_uri:
                image_bytes = await fetch(
                    first_frame_uri,
                    allowed_dir=data_dir,
                    allowed_gcs_buckets=app_ctx.allowed_gcs_buckets,
                )

            await ctx.info(f"Generating beat {idx + 1}/{len(beats)}")
            beat_result = await generate_video_impl(
                client=app_ctx.client,
                prompt=prompt,
                videos_dir=app_ctx.videos_dir,
                model=model,
                image_bytes=image_bytes,
                allowed_dir=data_dir,
                aspect_ratio=aspect_ratio,
                duration_seconds=duration,
                include_audio=include_audio,
                audio_prompt=beat.get("audio_prompt"),
                negative_prompt=beat.get("negative_prompt"),
                seed=seed,
                log_callback=ctx.info,
                output_gcs_uri=gcs_uri,
            )

            beat_manifest = {
                "kind": "beat",
                "index": idx,
                "prompt": prompt,
                "model": model,
                "aspect_ratio": aspect_ratio,
                "duration_seconds": beat_result.get("duration_seconds", duration),
                "seed": seed,
                "video_url": beat_result.get("video_url"),
                "generation_mode": beat_result.get("generation_mode"),
            }
            sidecar_url = _write_sidecar(
                beat_result.get("video_url", ""), beat_manifest
            )
            if sidecar_url:
                beat_manifest["sidecar_url"] = sidecar_url

            # If bridging, generate the bridge between the previous beat and
            # this one now that we have both endpoints. Insert before the
            # current beat in the segments list.
            if add_bridges and idx > 0 and prev_video_bytes is not None:
                # Read this beat's bytes to extract its first frame.
                beat_url = beat_result.get("video_url", "")
                cur_bytes: bytes | None = None
                if beat_url.startswith("file://"):
                    try:
                        cur_bytes = Path(beat_url[7:]).read_bytes()
                    except OSError as e:
                        logger.warning(
                            "Skipping bridge before beat %d: %s", idx, e
                        )

                if cur_bytes is not None:
                    end_frame = await asyncio.to_thread(
                        extract_frame_png, prev_video_bytes, "end"
                    )
                    start_frame = await asyncio.to_thread(
                        extract_frame_png, cur_bytes, "start"
                    )
                    await ctx.info(f"Generating bridge before beat {idx + 1}")
                    bridge_result = await generate_video_impl(
                        client=app_ctx.client,
                        prompt="smooth cinematic cut between the two beats",
                        videos_dir=app_ctx.videos_dir,
                        model=model,
                        image_bytes=end_frame,
                        last_frame_bytes=start_frame,
                        allowed_dir=data_dir,
                        aspect_ratio=aspect_ratio,
                        duration_seconds=4.0,
                        include_audio=False,
                        log_callback=ctx.info,
                        output_gcs_uri=gcs_uri,
                    )
                    bridge_manifest = {
                        "kind": "bridge",
                        "between_beats": [idx - 1, idx],
                        "model": model,
                        "aspect_ratio": aspect_ratio,
                        "duration_seconds": bridge_result.get(
                            "duration_seconds", 4.0
                        ),
                        "video_url": bridge_result.get("video_url"),
                    }
                    b_sidecar = _write_sidecar(
                        bridge_result.get("video_url", ""), bridge_manifest
                    )
                    if b_sidecar:
                        bridge_manifest["sidecar_url"] = b_sidecar
                    segments.append(bridge_manifest)
                    total_duration += float(
                        bridge_manifest["duration_seconds"]
                    )

            segments.append(beat_manifest)
            total_duration += float(beat_manifest["duration_seconds"])

            # Cache bytes for the next iteration's bridge extraction.
            if add_bridges:
                beat_url = beat_result.get("video_url", "")
                if beat_url.startswith("file://"):
                    try:
                        prev_video_bytes = Path(beat_url[7:]).read_bytes()
                    except OSError:
                        prev_video_bytes = None
                else:
                    prev_video_bytes = None

        clip_manifest = {
            "kind": "clip",
            "aspect_ratio": aspect_ratio,
            "model": model,
            "segments": segments,
            "total_duration_seconds": total_duration,
            "beat_count": len(beats),
        }
        return json.dumps(clip_manifest, indent=2)
    except Exception as e:
        await ctx.error(f"Clip generation failed: {e}")
        logger.exception("Tool error")
        return json.dumps({"error": str(e)})


def main() -> None:
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Gemini Media MCP Server")
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

    subparsers = parser.add_subparsers(dest="command")

    # Transport subcommands preserve the existing positional behavior
    # (gemini-media-mcp [stdio|sse|streamable-http]).
    for transport in ("stdio", "sse", "streamable-http"):
        subparsers.add_parser(
            transport, help=f"Run the MCP server with {transport} transport"
        )

    setup_parser = subparsers.add_parser(
        "setup", help="Interactive setup wizard for credentials and Claude Desktop"
    )
    setup_parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Do not prompt; all values must be provided via flags.",
    )
    setup_parser.add_argument(
        "--mode",
        choices=["gemini", "vertex"],
        help="Credential mode to configure.",
    )
    setup_parser.add_argument("--api-key", help="Gemini API key (mode=gemini).")
    setup_parser.add_argument(
        "--project-id", help="Google Cloud project ID (mode=vertex)."
    )
    setup_parser.add_argument(
        "--location", help="Google Cloud location (mode=vertex, default us-central1)."
    )
    setup_parser.add_argument(
        "--sa-path", help="Path to service account JSON file (mode=vertex)."
    )
    setup_parser.add_argument(
        "--sa-json", help="Inline service account JSON string (mode=vertex)."
    )
    setup_parser.add_argument(
        "--data-folder", help="Output folder for generated media."
    )
    setup_parser.add_argument(
        "--video-gcs-bucket", help="Optional gs:// URI for large video output."
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s:%(name)s:%(message)s",
        stream=sys.stderr,
    )

    if args.command == "setup":
        from .setup_wizard import run_wizard

        overrides: dict[str, Any] = {}
        if args.mode:
            overrides["mode"] = args.mode
        if args.api_key:
            overrides["api_key"] = args.api_key
        if args.project_id:
            overrides["project_id"] = args.project_id
        if args.location:
            overrides["location"] = args.location
        if args.sa_path:
            overrides["sa_path"] = args.sa_path
        if args.sa_json:
            overrides["sa_json"] = args.sa_json
        if args.data_folder:
            overrides["data_folder"] = args.data_folder
        if args.video_gcs_bucket:
            overrides["video_gcs_bucket"] = args.video_gcs_bucket

        run_wizard(interactive=not args.non_interactive, **overrides)
        return

    transport = args.command or "stdio"

    if not check_credentials():
        logger.error(
            "No credentials configured. Set GEMINI_API_KEY or enable "
            "GOOGLE_GENAI_USE_VERTEXAI=true with appropriate credentials. "
            "Run 'gemini-media-mcp setup' for an interactive setup wizard."
        )
        sys.exit(1)

    mcp.run(transport=transport, mount_path=args.mount_path)


if __name__ == "__main__":
    main()
