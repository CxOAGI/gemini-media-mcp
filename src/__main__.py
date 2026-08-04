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
from urllib.parse import urljoin, urlparse

import aiohttp
from google import genai
from google.cloud import storage
from mcp.server.fastmcp import Context, FastMCP, Image
from mcp.server.session import ServerSession
from mcp.types import TextContent
from PIL import Image as PILImage

from .image import ImageModel, ImageSize, MediaResolution, RetiredImageModel
from .image import generate_image as generate_image_impl
from .omni import OMNI_MODEL
from .omni import generate_video_omni as generate_video_omni_impl
from .video import _VEO_LITE_MODELS, VideoModel
from .video import generate_video as generate_video_impl
from .video_utils import extract_frame_png

logger = logging.getLogger(__name__)

# 50 MB cap on any single fetch to prevent memory/disk exhaustion.
MAX_FETCH_BYTES = 50 * 1024 * 1024

# Maximum number of HTTP redirects to follow during a fetch. Each hop's
# target is re-validated against the SSRF guard before it is requested.
MAX_HTTP_REDIRECTS = 5


def _decode_base64_capped(data: str, max_bytes: int | None = None) -> bytes:
    """Base64-decode input and enforce the same size cap as URI fetches.

    Prevents an attacker from bypassing MAX_FETCH_BYTES by supplying a huge
    inline base64 payload instead of a URI.
    """
    if max_bytes is None:
        max_bytes = MAX_FETCH_BYTES
    # Strip ASCII whitespace first: MIME base64 wraps at 76 columns and
    # base64.b64decode ignores whitespace anyway, so removing it lets us reason
    # about the true payload length. Reject BEFORE decoding only when a LOWER
    # BOUND on the decoded size exceeds the cap, so wrapped/padded input whose
    # real decoded size is within the cap is not falsely rejected. For cleaned
    # length L, decoded size lies in [(L // 4) * 3 - 2, (L // 4) * 3] (trailing
    # padding removes 1-2 bytes). The exact post-decode check below is the hard
    # enforcement; this pre-check only guards against multi-hundred-MB
    # allocations from a clearly oversize payload.
    cleaned = "".join(data.split())
    if (len(cleaned) // 4) * 3 - 2 > max_bytes:
        raise ValueError(
            f"Base64 input ({len(cleaned)} chars) exceeds decoded cap {max_bytes}"
        )
    decoded = base64.b64decode(cleaned)
    if len(decoded) > max_bytes:
        raise ValueError(
            f"Decoded base64 input ({len(decoded)} bytes) exceeds cap {max_bytes}"
        )
    return decoded


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
    # Dedicated Gemini API (AI Studio) client. Used to route models that
    # are not yet published on Vertex (e.g. veo-3.1-lite-generate-preview)
    # when the primary client runs in Vertex mode. None when no
    # GEMINI_API_KEY is configured.
    gemini_api_client: genai.Client | None = None
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


def create_gemini_api_client() -> genai.Client | None:
    """Create a Gemini API (AI Studio) client if GEMINI_API_KEY is set.

    Force `vertexai=False` so this client stays on the Gemini API even when
    `GOOGLE_GENAI_USE_VERTEXAI=true` is set in the environment — otherwise
    the SDK treats the key as a Vertex API key and routes to
    aiplatform.googleapis.com.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key, vertexai=False)


def _client_for_video_model(app_ctx: AppContext, model: str) -> genai.Client:
    """Pick the right genai.Client for a video model.

    Veo 3.1 Lite is served via the Gemini API / AI Studio only; calling
    it on Vertex hits aiplatform.googleapis.com and fails. When the
    primary client is in Vertex mode, route Lite through the Gemini API
    client instead.
    """
    if model in _VEO_LITE_MODELS and getattr(
        app_ctx.client._api_client, "vertexai", False
    ):
        if app_ctx.gemini_api_client is None:
            raise RuntimeError(
                f"Model {model} is only available via the Gemini API. "
                "Set GEMINI_API_KEY so the lite model can be served by "
                "AI Studio (Vertex AI has not published it yet)."
            )
        return app_ctx.gemini_api_client
    return app_ctx.client


# Memoized global-location Vertex client for omni. On Vertex the omni
# interactions collection lives at .../locations/global/interactions, so a
# regional primary client won't reach it.
_omni_vertex_global_client: genai.Client | None = None


def _get_omni_vertex_global_client() -> genai.Client:
    """Return a memoized Vertex client pinned to the global location."""
    global _omni_vertex_global_client
    if _omni_vertex_global_client is None:
        _omni_vertex_global_client = genai.Client(vertexai=True, location="global")
    return _omni_vertex_global_client


def _client_for_omni(app_ctx: AppContext, *, need_gcs: bool = False) -> genai.Client:
    """Pick the genai.Client for gemini-omni-flash (Interactions API).

    Omni + the Interactions API are documented on BOTH backends: the Gemini
    Developer API (where Interactions is GA) and Vertex AI / Gemini Enterprise
    Agent Platform (preview; may require allowlisting).

    GCS output delivery only works on Vertex, so when the caller needs it
    (`need_gcs`) and the primary client is Vertex-capable, prefer the
    global-location Vertex client even if a Gemini API key is also configured
    — otherwise the explicit output_gcs_uri would be silently dropped. When
    GCS is not needed, prefer the dedicated Gemini API client (Interactions is
    GA there, the safest path). Falls back to a global-location Vertex client
    for a Vertex primary (omni's interactions collection is location `global`),
    or the primary client as-is on the Gemini API.
    """
    primary_is_vertex = getattr(app_ctx.client._api_client, "vertexai", False)
    if need_gcs and primary_is_vertex:
        return _get_omni_vertex_global_client()
    if app_ctx.gemini_api_client is not None:
        return app_ctx.gemini_api_client
    if primary_is_vertex:
        return _get_omni_vertex_global_client()
    return app_ctx.client


async def _omni_generate_and_manifest(
    app_ctx: AppContext,
    ctx: Context[ServerSession, AppContext],
    *,
    prompt: str,
    image_bytes_list: list[bytes] | None = None,
    input_video_bytes: bytes | None = None,
    previous_interaction_id: str | None = None,
    aspect_ratio: str = "16:9",
    duration_seconds: float = 6.0,
    output_gcs_uri: str | None = None,
    timeout_seconds: int = 600,
    manifest_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the omni impl and attach a sidecar manifest, returning the result.

    Shared by generate_video_omni, edit_video, the generate_video draft path,
    and generate_clip's animatic mode so they all produce consistent output.
    """
    # Route to the Vertex client when GCS output is requested (delivery only
    # works there), otherwise prefer the Gemini API client.
    client = _client_for_omni(app_ctx, need_gcs=bool(output_gcs_uri))

    # GCS delivery only works on Vertex; on the Gemini API omni returns bytes
    # inline. Drop an explicit output_gcs_uri on a non-Vertex omni client with
    # a warning rather than sending an unsupported field.
    effective_gcs = output_gcs_uri
    gcs_warning: str | None = None
    if output_gcs_uri and not getattr(client._api_client, "vertexai", False):
        effective_gcs = None
        gcs_warning = (
            "output_gcs_uri is ignored: omni on the Gemini API returns video "
            "inline (GCS delivery requires Vertex AI mode)."
        )

    result = await generate_video_omni_impl(
        client=client,
        prompt=prompt,
        videos_dir=app_ctx.videos_dir,
        image_bytes_list=image_bytes_list or None,
        input_video_bytes=input_video_bytes,
        previous_interaction_id=previous_interaction_id,
        aspect_ratio=aspect_ratio,
        duration_seconds=duration_seconds,
        output_gcs_uri=effective_gcs,
        timeout_seconds=timeout_seconds,
        log_callback=ctx.info,
    )
    if gcs_warning:
        result.setdefault("warnings", []).append(gcs_warning)
    manifest: dict[str, Any] = {
        "kind": "omni_video",
        "prompt": prompt,
        "model": result.get("model"),
        "aspect_ratio": aspect_ratio,
        "duration_seconds": result.get("duration_seconds", duration_seconds),
        "interaction_id": result.get("interaction_id"),
        "previous_interaction_id": previous_interaction_id,
        "output_gcs_uri": effective_gcs,
        "video_url": result.get("video_url"),
    }
    if manifest_extra:
        manifest.update(manifest_extra)
    warnings = result.get("warnings")
    if warnings:
        manifest["warnings"] = warnings
    sidecar_url = _write_sidecar(result.get("video_url", ""), manifest)
    if sidecar_url:
        result["sidecar_url"] = sidecar_url
    else:
        # No sidecar written (remote URL or write failure) — include the
        # manifest inline so interaction lineage (previous_interaction_id,
        # ignored params) isn't lost. Matches the Veo tools' fallback.
        result["manifest"] = manifest
    return result


def _validate_aspect_ratio(aspect_ratio: str) -> None:
    """Raise ValueError for aspect ratios the VEO video models don't support.

    Mirrors the impl-side backstop so the error surfaces up front (before any
    input image / frame is fetched) rather than after wasted fetch work.
    """
    if aspect_ratio not in ("16:9", "9:16"):
        raise ValueError(
            f"Unsupported aspect_ratio '{aspect_ratio}'. "
            "Supported values are '16:9' and '9:16'."
        )


def _resolve_video_gcs(
    output_gcs_uri: str | None,
    default_bucket: str | None,
    allowed_buckets: frozenset[str],
    is_vertex_client: bool,
) -> str | None:
    """Resolve the effective GCS output URI for a video call.

    Combines explicit output_gcs_uri with the env default_bucket, validates
    gs:// scheme + allowlist. On a non-Vertex client: raises ValueError if
    output_gcs_uri was explicit (GCS unsupported on the Gemini API), else
    drops the env default and returns None.
    """
    gcs_uri = output_gcs_uri or default_bucket
    # GCS output only works on Vertex AI. On the Gemini API path, reject an
    # explicit output_gcs_uri (the caller asked for the impossible) but
    # silently drop a VIDEO_GCS_BUCKET env default so Lite / text-to-video
    # still succeed inline. This drop happens BEFORE format validation so a
    # malformed env default (e.g. a bare bucket name) can't fail calls on a
    # path where it would never be used anyway.
    if gcs_uri and not is_vertex_client:
        if output_gcs_uri:
            raise ValueError(
                "output_gcs_uri requires Vertex AI mode. This model is "
                "served via the Gemini API, which does not support GCS "
                "output. Omit output_gcs_uri to receive the video inline."
            )
        return None
    if gcs_uri:
        bucket = _parse_gcs_bucket(gcs_uri)
        if bucket is None:
            raise ValueError(f"output_gcs_uri must start with gs://: {gcs_uri}")
        if allowed_buckets and bucket not in allowed_buckets:
            raise ValueError(
                f"output_gcs_uri bucket '{bucket}' is not in the allowlist. "
                f"Configured: {sorted(allowed_buckets)}"
            )
    return gcs_uri


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
    gemini_api_client = create_gemini_api_client()
    video_gcs_bucket = os.environ.get("VIDEO_GCS_BUCKET")
    allowed_gcs_buckets = _compute_allowed_gcs_buckets()

    try:
        yield AppContext(
            data_folder=data_folder,
            images_dir=images_dir,
            videos_dir=videos_dir,
            client=client,
            gemini_api_client=gemini_api_client,
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
    if resolved != allowed and not resolved.is_relative_to(allowed):
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
    Synchronous; use `_assert_http_host_public_async` from async code so
    the DNS lookup does not block the event loop.
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
            raise ValueError(f"Refusing to fetch non-public address: {host} -> {addr}")


async def _assert_http_host_public_async(url: str) -> None:
    """Async wrapper that runs the DNS lookup off the event loop."""
    await asyncio.to_thread(_assert_http_host_public, url)


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
            # Disable automatic redirects and follow them manually so the
            # SSRF guard re-runs on every hop. A public URL that 30x-redirects
            # to 169.254.169.254/ or an internal host is otherwise fetched
            # with only the original host validated.
            timeout = aiohttp.ClientTimeout(total=60)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                current_url = uri
                for _hop in range(MAX_HTTP_REDIRECTS + 1):
                    await _assert_http_host_public_async(current_url)
                    async with session.get(current_url, allow_redirects=False) as resp:
                        if resp.status in (301, 302, 303, 307, 308):
                            location = resp.headers.get("Location")
                            if not location:
                                raise ValueError(
                                    f"Redirect from {current_url} missing "
                                    "Location header"
                                )
                            current_url = urljoin(current_url, location)
                            continue
                        if resp.status == 200:
                            return await _read_capped_http(resp, max_bytes)
                        raise ValueError(f"HTTP {resp.status}")
                raise ValueError(
                    f"Exceeded redirect limit ({MAX_HTTP_REDIRECTS}) fetching {uri}"
                )

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

        # Reject unsupported schemes (ftp://, s3://, data:, ...) with a clear
        # message instead of letting them fall into local-path validation,
        # which would misreport them as "outside the allowed directory".
        if "://" in uri or uri.startswith("data:"):
            raise ValueError(f"Unsupported URI scheme: {uri}")

        # Bare path: validate the location BEFORE any filesystem probe so a
        # path outside allowed_dir never gets an exists()/stat() call that
        # could leak file existence via timing or differing error paths.
        validated = _validate_local_path(Path(uri), allowed_dir)
        if validated.stat().st_size > max_bytes:
            raise ValueError(f"File exceeds size cap {max_bytes}: {validated}")
        return validated.read_bytes()
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
    model: ImageModel | RetiredImageModel,
    image_uri: str | None = None,
    image_base64: str | None = None,
    reference_image_uris: list[str] | None = None,
    image_size: ImageSize | None = None,
    media_resolution: MediaResolution | None = None,
    aspect_ratio: str | None = None,
    person_generation: str | None = None,
    thought_signature_url: str | None = None,
):
    """Generate an image using Google Gemini image models.

    Args:
        ctx: MCP context with application state
        prompt: Text description of the image to generate
        model: Which model to call. Pick by use case:
               - Default / conversational edits / balanced, up to 4K:
                 "gemini-3.1-flash-image" (GA, Nano Banana 2)
               - Cheapest / fast iteration, 1K output only:
                 "gemini-3.1-flash-lite-image" (GA) — it cannot produce 2K or
                 4K, so use flash or pro when image_size is 2K/4K
               - Most capable: reasoning + precise text rendering, 4K,
                 up to 14 reference images:
                 "gemini-3-pro-image" (GA, Nano Banana Pro)
               Pick one of the three above and nothing else. The remaining IDs
               in the schema are superseded and exist only so pinned
               configurations keep working: every "imagen-*" image endpoint is
               discontinued 2026-08-17, the "-preview" image aliases were
               retired 2026-06-25, and "gemini-2.5-flash-image" is scheduled
               for shutdown 2026-10-02. Requesting one of those is rerouted to
               the GA replacement and reported under "warnings".
        image_uri: Input image URI (gs://, http://, file://) for image-to-image
        image_base64: Base64 encoded input image (prefer image_uri)
        reference_image_uris: List of reference image URIs (up to 14 for Gemini 3.x image models).
            Use up to 6 object images for high-fidelity inclusion,
            up to 5 human images for character consistency across scenes.
        image_size: Output image size for Gemini 3.x image models (must use uppercase K):
            - "1K": 1024px
            - "2K": 2048px — not supported by gemini-3.1-flash-lite-image
            - "4K": 4096px — not supported by gemini-3.1-flash-lite-image
        media_resolution: Input image processing resolution:
            - "MEDIA_RESOLUTION_LOW": Faster, lower token usage
            - "MEDIA_RESOLUTION_MEDIUM": Balanced
            - "MEDIA_RESOLUTION_HIGH": Best quality, higher token usage
        aspect_ratio: Desired output aspect ratio, e.g. "1:1", "16:9", "9:16",
            "4:3", "3:4".
        person_generation: Policy for generating people:
            - "dont_allow": Do not generate people
            - "allow_adult": Allow generating adults (default behavior)
            - "allow_all": Allow generating all ages
            (Some regions restrict these values.)
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
            # Fail loudly: a provided URI that can't be fetched (bad URL,
            # SSRF rejection, size cap) must not silently downgrade to a
            # text-to-image generation.
            if image_bytes is None:
                raise ValueError(f"Could not fetch image_uri: {image_uri}")
        elif image_base64:
            image_bytes = _decode_base64_capped(image_base64)

        # Fetch reference images
        reference_images: list[bytes] = []
        if reference_image_uris:
            for ref_uri in reference_image_uris[:14]:  # Max 14 for Gemini 3.x
                ref_bytes = await fetch(
                    ref_uri,
                    allowed_dir=data_dir,
                    allowed_gcs_buckets=app_ctx.allowed_gcs_buckets,
                )
                # Don't silently drop a reference the caller explicitly asked
                # for — that would quietly change the generation result.
                if ref_bytes is None:
                    raise ValueError(f"Could not fetch reference image: {ref_uri}")
                reference_images.append(ref_bytes)

        # Read thought signature from file if URL provided. A malformed value
        # must not be silently ignored — that would quietly turn a multi-turn
        # edit into an unrelated fresh generation.
        thought_signature = None
        if thought_signature_url:
            if not thought_signature_url.startswith("file://"):
                raise ValueError(
                    "thought_signature_url must be a file:// URL returned by a "
                    f"previous generate_image call, got: {thought_signature_url}"
                )
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
            aspect_ratio=aspect_ratio,
            person_generation=person_generation,
            thought_signature=thought_signature,
        )
        # The impl returns a dict without image_url/image_preview when the
        # model responds with text only (e.g. a safety refusal or a clarifying
        # question). Surface that text instead of crashing on a missing key.
        if "image_url" not in result:
            for warning in result.get("warnings", []):
                await ctx.warning(warning)
            await ctx.info("Model returned text only")
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

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

        # Surface impl warnings (e.g. an Imagen ID rerouted to its GA target).
        # These go out on the MCP logging channel as well as in the payload, so
        # a client that only reads the image sees them too.
        impl_warnings = result.get("warnings")
        if impl_warnings:
            response_data["warnings"] = impl_warnings
            for warning in impl_warnings:
                await ctx.warning(warning)

        # Write sidecar manifest so downstream tools (e.g. vfx-mcp) can read
        # generation parameters without parsing response JSON.
        manifest: dict[str, Any] = {
            "kind": "image",
            "prompt": prompt,
            # The model actually served, which differs from `model` when a
            # legacy Imagen ID was rerouted to its Gemini GA replacement.
            "model": result["model"],
            "image_url": result["image_url"],
            "image_size": image_size,
            "media_resolution": media_resolution,
            "aspect_ratio": aspect_ratio,
            "person_generation": person_generation,
            "reference_image_uris": reference_image_uris,
            "source_image_uri": image_uri,
            "thought_signature_url": result.get("thought_signature_url"),
        }
        if impl_warnings:
            manifest["warnings"] = impl_warnings
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
    resolution: str | None = None,
    person_generation: str | None = None,
    image_uri: str | None = None,
    image_base64: str | None = None,
    last_frame_uri: str | None = None,
    last_frame_base64: str | None = None,
    reference_image_uris: list[str] | None = None,
    extend_video_uri: str | None = None,
    output_gcs_uri: str | None = None,
    draft: bool = False,
) -> str:
    """Generate a video using Google VEO models.

    Args:
        ctx: MCP context with application state
        prompt: Text description of the video to generate
        model: Model to use - options include:
               - "veo-3.1-generate-001": VEO 3.1 (highest quality, 4/6/8s, audio)
               - "veo-3.1-fast-generate-001": VEO 3.1 Fast (faster, 4/6/8s, audio)
               - "veo-3.1-lite-generate-preview": VEO 3.1 Lite (most cost-effective,
                 4/6/8s, audio; supports ONLY text-to-video and
                 image-to-video — no extension, reference images,
                 first/last-frame, or 4K).
                 Availability note: Lite is served via the Gemini API / AI
                 Studio only; Vertex AI has not published it. When the server
                 runs in Vertex mode, Lite calls are automatically routed
                 through the Gemini API client, so GEMINI_API_KEY must also
                 be set to use Lite.
        aspect_ratio: 16:9 (default) or 9:16
        duration_seconds: Video duration (4/6/8s)
        include_audio: Enable audio generation
        audio_prompt: Audio description
        negative_prompt: Things to avoid in the video
        seed: Random seed for reproducibility
        resolution: Output resolution, "720p" or "1080p". "4K" is only
            available on the non-Lite VEO models (not veo-3.1-lite).
        person_generation: Person generation policy, "allow_adult" or
            "allow_all". (Some regions restrict these values.)
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
            On Vertex AI, extension requires output_gcs_uri (the larger combined
            video exceeds inline limits). On the Gemini API the extended clip is
            returned inline, so no GCS target is needed.
        output_gcs_uri: GCS bucket URI for large video output (e.g. gs://bucket/path/).
            Vertex AI only — on the Gemini API, output is always returned inline
            and an explicit output_gcs_uri is rejected.
        draft: When True, route to gemini-omni-flash for a fast 720p draft
            instead of Veo. Iterate cheaply, then re-run with draft=False to
            finalize on Veo. Omni ignores Veo-only controls (seed,
            negative_prompt, resolution, last frame, reference images,
            extension); any that were passed are noted in the response.

    Returns:
        JSON with video_url and generation details including generation_mode
    """
    try:
        app_ctx = ctx.request_context.lifespan_context
        data_dir = app_ctx.data_folder

        # Fail fast on an unsupported aspect ratio before any fetch work.
        _validate_aspect_ratio(aspect_ratio)

        # Draft mode: hand off to the fast omni path. Omni supports only a
        # text prompt + optional input image(s), so Veo-only controls are
        # ignored — surface which ones so the caller isn't surprised.
        if draft:
            # `is not None` (not truthiness): seed=0 is a valid Veo seed and
            # must be reported as ignored too. include_audio is only notable
            # when explicitly enabled (omni has no audio control).
            ignored = [
                name
                for name, val in (
                    ("seed", seed),
                    ("negative_prompt", negative_prompt),
                    ("resolution", resolution),
                    ("person_generation", person_generation),
                    ("last_frame_uri", last_frame_uri or last_frame_base64),
                    ("reference_image_uris", reference_image_uris),
                    ("extend_video_uri", extend_video_uri),
                    ("output_gcs_uri", output_gcs_uri),
                )
                if val is not None and val != []
            ]
            if include_audio:
                ignored.append("include_audio")
            # audio_prompt CAN be honored best-effort: inline it into the
            # prompt text, exactly like the Veo path does.
            draft_prompt = prompt
            if audio_prompt:
                draft_prompt = f"{prompt}\nAudio: {audio_prompt}"
            draft_image_bytes: list[bytes] | None = None
            if image_uri:
                b = await fetch(
                    image_uri,
                    allowed_dir=data_dir,
                    allowed_gcs_buckets=app_ctx.allowed_gcs_buckets,
                )
                if b is None:
                    raise ValueError(f"Could not fetch image_uri: {image_uri}")
                draft_image_bytes = [b]
            elif image_base64:
                draft_image_bytes = [_decode_base64_capped(image_base64)]

            extra: dict[str, Any] = {"draft": True}
            if ignored:
                extra["ignored_veo_params"] = ignored
            await ctx.info("Generating draft with gemini-omni-flash")
            result = await _omni_generate_and_manifest(
                app_ctx,
                ctx,
                prompt=draft_prompt,
                image_bytes_list=draft_image_bytes,
                aspect_ratio=aspect_ratio,
                duration_seconds=duration_seconds,
                manifest_extra=extra,
            )
            if ignored:
                result.setdefault("warnings", []).append(
                    "draft mode (gemini-omni-flash) ignored Veo-only params: "
                    + ", ".join(ignored)
                )
            return json.dumps(result, indent=2)

        # Fetch first frame image
        image_bytes = None
        if image_uri:
            image_bytes = await fetch(
                image_uri,
                allowed_dir=data_dir,
                allowed_gcs_buckets=app_ctx.allowed_gcs_buckets,
            )
            # Fail loudly rather than silently degrading image-to-video to
            # text-to-video when the provided URI can't be fetched.
            if image_bytes is None:
                raise ValueError(f"Could not fetch image_uri: {image_uri}")
        elif image_base64:
            image_bytes = _decode_base64_capped(image_base64)

        # Fetch last frame image (VEO 3.1 first+last frame mode)
        last_frame_bytes = None
        if last_frame_uri:
            last_frame_bytes = await fetch(
                last_frame_uri,
                allowed_dir=data_dir,
                allowed_gcs_buckets=app_ctx.allowed_gcs_buckets,
            )
            # A provided last frame that can't be fetched must not silently
            # drop first+last mode back to plain image-to-video.
            if last_frame_bytes is None:
                raise ValueError(f"Could not fetch last_frame_uri: {last_frame_uri}")
        elif last_frame_base64:
            last_frame_bytes = _decode_base64_capped(last_frame_base64)

        # Fetch reference images (VEO 3.1 reference mode)
        reference_images: list[bytes] = []
        if reference_image_uris:
            for ref_uri in reference_image_uris[:3]:  # Max 3 for VEO 3.1
                ref_bytes = await fetch(
                    ref_uri,
                    allowed_dir=data_dir,
                    allowed_gcs_buckets=app_ctx.allowed_gcs_buckets,
                )
                if ref_bytes is None:
                    raise ValueError(f"Could not fetch reference image: {ref_uri}")
                reference_images.append(ref_bytes)

        # Resolve the client up front: Veo Lite (and pure Gemini-API
        # deployments) route to the Gemini API, which does not support GCS
        # output. GCS behavior below depends on which backend is used.
        video_client = _client_for_video_model(app_ctx, model)
        is_vertex_client = getattr(video_client._api_client, "vertexai", False)

        # Combine explicit output_gcs_uri with the env default, validate the
        # allowlist, and apply the non-Vertex drop/raise gating.
        gcs_uri = _resolve_video_gcs(
            output_gcs_uri,
            app_ctx.video_gcs_bucket,
            app_ctx.allowed_gcs_buckets,
            is_vertex_client,
        )

        if extend_video_uri:
            # extend_video_uri is handed straight to the API as a video
            # source; a gs:// value must pass the same allowlist as every
            # other gs:// input so it can't reach arbitrary buckets.
            if extend_video_uri.startswith("gs://"):
                extend_bucket = _parse_gcs_bucket(extend_video_uri)
                if extend_bucket is None:
                    raise ValueError(f"Invalid extend_video_uri: {extend_video_uri}")
                _assert_gcs_bucket_allowed(extend_bucket, app_ctx.allowed_gcs_buckets)
            # On Vertex, extensions produce a larger combined video that must be
            # written to GCS. On the Gemini API there is no GCS output; the
            # extended clip is downloaded inline, so no GCS target is required.
            if is_vertex_client and not gcs_uri:
                raise ValueError(
                    "Video extension on Vertex AI requires output_gcs_uri (or a "
                    "configured VIDEO_GCS_BUCKET). Extensions produce larger "
                    "combined videos that exceed inline response limits."
                )

        await ctx.info(f"Generating video with model={model}")
        result = await generate_video_impl(
            client=video_client,
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
            resolution=resolution,
            person_generation=person_generation,
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
            "resolution": resolution,
            "person_generation": person_generation,
            "audio_enabled": result.get("audio_enabled", include_audio),
            "generation_mode": result.get("generation_mode"),
            "seed": seed,
            "video_url": result.get("video_url"),
            "source_image_uri": image_uri,
            "last_frame_uri": last_frame_uri,
            "reference_image_uris": reference_image_uris,
            "extend_video_uri": extend_video_uri,
        }
        # Surface any warnings the impl emitted for silently-dropped caller
        # intent (e.g. include_audio ignored on the Gemini API path).
        warnings = result.get("warnings")
        if warnings:
            manifest["warnings"] = warnings
        sidecar_url = _write_sidecar(result.get("video_url", ""), manifest)
        if sidecar_url:
            result["sidecar_url"] = sidecar_url
        else:
            # No sidecar for remote (gs://) outputs — include the manifest
            # inline so generation parameters aren't lost for exactly the
            # large-video path GCS output is meant for.
            result["manifest"] = manifest

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
        model: VEO model (defaults to fast; Lite does NOT support
            first/last-frame mode and cannot be used here)
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

        # Fail fast on an unsupported aspect ratio before any fetch work.
        _validate_aspect_ratio(aspect_ratio)

        # Resolve the client up front so GCS gating can see which backend is
        # in play (the Gemini API does not support GCS output).
        video_client = _client_for_video_model(app_ctx, model)
        is_vertex_client = getattr(video_client._api_client, "vertexai", False)

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

        gcs_uri = _resolve_video_gcs(
            output_gcs_uri,
            app_ctx.video_gcs_bucket,
            app_ctx.allowed_gcs_buckets,
            is_vertex_client,
        )

        await ctx.info(f"Generating transition with model={model}")
        result = await generate_video_impl(
            client=video_client,
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
        warnings = result.get("warnings")
        if warnings:
            manifest["warnings"] = warnings
        sidecar_url = _write_sidecar(result.get("video_url", ""), manifest)
        if sidecar_url:
            result["sidecar_url"] = sidecar_url
        else:
            # No sidecar for remote (gs://) outputs — include the manifest
            # inline so generation parameters aren't lost for exactly the
            # large-video path GCS output is meant for.
            result["manifest"] = manifest
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

        # Fail fast on an unsupported aspect ratio before any fetch work.
        _validate_aspect_ratio(aspect_ratio)

        # Resolve the client up front so GCS gating can see which backend is
        # in play (the Gemini API does not support GCS output).
        video_client = _client_for_video_model(app_ctx, model)
        is_vertex_client = getattr(video_client._api_client, "vertexai", False)

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
        first_frame_png = await asyncio.to_thread(extract_frame_png, from_bytes, "end")
        last_frame_png = await asyncio.to_thread(extract_frame_png, to_bytes, "start")

        gcs_uri = _resolve_video_gcs(
            output_gcs_uri,
            app_ctx.video_gcs_bucket,
            app_ctx.allowed_gcs_buckets,
            is_vertex_client,
        )

        await ctx.info(f"Generating bridge with model={model}")
        result = await generate_video_impl(
            client=video_client,
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
        warnings = result.get("warnings")
        if warnings:
            manifest["warnings"] = warnings
        sidecar_url = _write_sidecar(result.get("video_url", ""), manifest)
        if sidecar_url:
            result["sidecar_url"] = sidecar_url
        else:
            # No sidecar for remote (gs://) outputs — include the manifest
            # inline so generation parameters aren't lost for exactly the
            # large-video path GCS output is meant for.
            result["manifest"] = manifest
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
    animatic: bool = False,
) -> str:
    """Generate a multi-beat short clip — the building block for a reel / short.

    Runs each `beat` through `generate_video` sequentially. When
    `add_bridges=True`, between each pair of beats a transition is
    generated using the last frame of beat N and the first frame of beat
    N+1 (same primitive as generate_bridge, just chained).

    The returned manifest is an ordered list of segments (beats and
    bridges, in playback order) that a downstream cutting MCP can splice
    into a final clip.

    Individual beat failures do not abort the whole clip: the failed
    beat's error is recorded in the manifest's `errors` list and the
    tool continues with the next beat. Subsequent bridges that would
    have used the failed beat are skipped.

    Args:
        beats: Ordered list of beat specs. Each item accepts:
            {prompt: str, duration_seconds?: float, seed?: int,
             first_frame_uri?: str, negative_prompt?: str,
             audio_prompt?: str}. If `first_frame_uri` is supplied and
            cannot be fetched, the beat fails (rather than silently
            falling back to text-to-video).
        aspect_ratio: Default 9:16 for vertical social clips.
        model: VEO model applied to every beat.
        include_audio: Enable audio on each beat (only effective on Vertex).
        add_bridges: Generate a bridge clip between consecutive beats.
            Bridges require local (file://) beat outputs; skipped when
            beats land in GCS.
        output_gcs_uri: GCS URI for all outputs (optional).
        animatic: When True, render every beat with gemini-omni-flash (fast,
            cheap 720p) instead of Veo, for a quick storyboard preview of the
            whole reel before committing to full Veo renders. Bridges are not
            available in animatic mode (add_bridges is ignored), and Veo-only
            per-beat controls (seed, negative_prompt) are ignored.

    Returns:
        JSON clip manifest:
        {
          "kind": "clip",
          "aspect_ratio": "9:16",
          "segments": [ ... ordered beat / bridge segments ... ],
          "total_duration_seconds": <sum>,
          "errors": [ {"beat_index": N, "error": "..."} ],  // only on failure
        }
    """
    try:
        app_ctx = ctx.request_context.lifespan_context
        data_dir = app_ctx.data_folder

        if not beats:
            raise ValueError("beats list must not be empty")

        # Validate the clip-level aspect ratio once, up front. Otherwise the
        # impl's per-value ValueError fires inside every beat's error handler,
        # producing a success-shaped manifest with zero segments instead of a
        # clear top-level failure.
        _validate_aspect_ratio(aspect_ratio)

        # Resolve the client ONCE, before the beat loop. If the model can't be
        # routed (e.g. a Lite model on Vertex with no GEMINI_API_KEY, or omni
        # with no Gemini API access), let the RuntimeError propagate to the
        # outer handler so the tool returns a top-level {"error": ...} rather
        # than failing every beat individually and returning a success-shaped
        # clip with empty segments.
        clip_warnings: list[str] = []
        if animatic:
            # Fast omni preview path: no Veo GCS, no bridges. video_client is a
            # harmless placeholder (never used — beats route to omni below).
            _client_for_omni(app_ctx)  # fail fast if omni is unavailable
            video_client = app_ctx.client
            gcs_uri = None
            if add_bridges:
                add_bridges = False
                clip_warnings.append(
                    "add_bridges is ignored in animatic mode "
                    "(bridges are a Veo first/last-frame feature)."
                )
            if output_gcs_uri:
                clip_warnings.append(
                    "output_gcs_uri is ignored in animatic mode; omni previews "
                    "are always written locally."
                )
            if include_audio:
                clip_warnings.append(
                    "include_audio is ignored in animatic mode "
                    "(gemini-omni-flash previews carry no controllable audio)."
                )
        else:
            video_client = _client_for_video_model(app_ctx, model)
            is_vertex_client = getattr(video_client._api_client, "vertexai", False)
            gcs_uri = _resolve_video_gcs(
                output_gcs_uri,
                app_ctx.video_gcs_bucket,
                app_ctx.allowed_gcs_buckets,
                is_vertex_client,
            )

        segments: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        total_duration = 0.0
        prev_video_bytes: bytes | None = None
        beat_model = OMNI_MODEL if animatic else model

        for idx, beat in enumerate(beats):
            try:
                prompt = beat.get("prompt")
                if not prompt:
                    raise ValueError(f"beat {idx} missing required 'prompt'")
                duration = float(beat.get("duration_seconds", 4.0))
                seed = beat.get("seed")

                # Optional per-beat first frame (image_to_video). Fail loudly
                # if the user supplied one and it can't be fetched.
                first_frame_uri = beat.get("first_frame_uri")
                image_bytes = None
                if first_frame_uri:
                    image_bytes = await fetch(
                        first_frame_uri,
                        allowed_dir=data_dir,
                        allowed_gcs_buckets=app_ctx.allowed_gcs_buckets,
                    )
                    if image_bytes is None:
                        raise ValueError(
                            f"Could not fetch first_frame_uri: {first_frame_uri}"
                        )

                await ctx.info(f"Generating beat {idx + 1}/{len(beats)}")
                if animatic:
                    # Fast omni preview: text prompt + optional first frame.
                    # audio_prompt is inlined into the prompt (best-effort,
                    # same as the Veo path); Veo-only per-beat controls are
                    # reported as dropped rather than silently discarded.
                    beat_prompt = prompt
                    if beat.get("audio_prompt"):
                        beat_prompt = f"{prompt}\nAudio: {beat['audio_prompt']}"
                    dropped = [
                        name
                        for name in ("negative_prompt", "seed")
                        if beat.get(name) is not None
                    ]
                    if dropped:
                        clip_warnings.append(
                            f"animatic mode ignored Veo-only beat params "
                            f"({', '.join(dropped)}) on beat {idx}."
                        )
                    beat_result = await generate_video_omni_impl(
                        client=_client_for_omni(app_ctx),
                        prompt=beat_prompt,
                        videos_dir=app_ctx.videos_dir,
                        image_bytes_list=[image_bytes] if image_bytes else None,
                        aspect_ratio=aspect_ratio,
                        duration_seconds=duration,
                        log_callback=ctx.info,
                    )
                else:
                    beat_result = await generate_video_impl(
                        client=video_client,
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
            except Exception as beat_err:
                await ctx.error(f"Beat {idx} failed: {beat_err}")
                logger.exception("Beat %d failed", idx)
                errors.append({"beat_index": idx, "error": str(beat_err)})
                # Invalidate the rolling bridge source so the next beat
                # doesn't try to bridge from a beat that never rendered.
                prev_video_bytes = None
                continue

            beat_manifest = {
                "kind": "beat",
                "index": idx,
                "prompt": prompt,
                "model": beat_model,
                "aspect_ratio": aspect_ratio,
                "duration_seconds": beat_result.get("duration_seconds", duration),
                "seed": None if animatic else seed,
                "video_url": beat_result.get("video_url"),
                "generation_mode": "animatic"
                if animatic
                else beat_result.get("generation_mode"),
                "interaction_id": beat_result.get("interaction_id"),
            }
            # Surface any warnings the impl emitted for this beat (e.g.
            # include_audio ignored on the Gemini API path) in the beat
            # manifest and aggregate them into the clip-level warnings.
            beat_warnings = beat_result.get("warnings")
            if beat_warnings:
                beat_manifest["warnings"] = beat_warnings
                clip_warnings.extend(beat_warnings)
            sidecar_url = _write_sidecar(
                beat_result.get("video_url", ""), beat_manifest
            )
            if sidecar_url:
                beat_manifest["sidecar_url"] = sidecar_url

            # If bridging, generate the bridge between the previous beat and
            # this one now that we have both endpoints. Insert before the
            # current beat in the segments list.
            if add_bridges and idx > 0 and prev_video_bytes is not None:
                beat_url = beat_result.get("video_url", "")
                cur_bytes: bytes | None = None
                if beat_url.startswith("file://"):
                    try:
                        cur_bytes = Path(beat_url[7:]).read_bytes()
                    except OSError as e:
                        logger.warning("Skipping bridge before beat %d: %s", idx, e)

                if cur_bytes is not None:
                    try:
                        end_frame = await asyncio.to_thread(
                            extract_frame_png, prev_video_bytes, "end"
                        )
                        start_frame = await asyncio.to_thread(
                            extract_frame_png, cur_bytes, "start"
                        )
                        await ctx.info(f"Generating bridge before beat {idx + 1}")
                        bridge_result = await generate_video_impl(
                            client=video_client,
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
                    except Exception as bridge_err:
                        await ctx.error(
                            f"Bridge before beat {idx} failed: {bridge_err}"
                        )
                        logger.exception("Bridge before beat %d failed", idx)
                        errors.append(
                            {
                                "bridge_between_beats": [idx - 1, idx],
                                "error": str(bridge_err),
                            }
                        )
                    else:
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
                        bridge_warnings = bridge_result.get("warnings")
                        if bridge_warnings:
                            bridge_manifest["warnings"] = bridge_warnings
                            clip_warnings.extend(bridge_warnings)
                        b_sidecar = _write_sidecar(
                            bridge_result.get("video_url", ""), bridge_manifest
                        )
                        if b_sidecar:
                            bridge_manifest["sidecar_url"] = b_sidecar
                        segments.append(bridge_manifest)
                        total_duration += float(bridge_manifest["duration_seconds"])

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

        clip_manifest: dict[str, Any] = {
            "kind": "clip",
            "aspect_ratio": aspect_ratio,
            "model": beat_model,
            "animatic": animatic,
            "segments": segments,
            "total_duration_seconds": total_duration,
            "beat_count": len(beats),
        }
        if errors:
            clip_manifest["errors"] = errors
        if clip_warnings:
            # The same warning (e.g. "audio not honored on the Gemini API")
            # is emitted per beat and per bridge; dedupe at the clip level,
            # preserving first-seen order, so the manifest carries each
            # distinct warning once instead of one copy per segment.
            clip_manifest["warnings"] = list(dict.fromkeys(clip_warnings))
        return json.dumps(clip_manifest, indent=2)
    except Exception as e:
        await ctx.error(f"Clip generation failed: {e}")
        logger.exception("Tool error")
        return json.dumps({"error": str(e)})


@mcp.tool()
async def generate_video_omni(
    ctx: Context[ServerSession, AppContext],
    prompt: str,
    image_uris: list[str] | None = None,
    input_video_uri: str | None = None,
    aspect_ratio: str = "16:9",
    duration_seconds: float = 6.0,
    previous_interaction_id: str | None = None,
    output_gcs_uri: str | None = None,
    timeout_seconds: int = 600,
) -> str:
    """Generate a video fast with gemini-omni-flash (Interactions API).

    Omni is the fast/cheap path (720p, 24fps): good for drafts and quick
    iteration. The Veo tools remain the high-fidelity path (1080p/4K, seeds,
    first/last-frame control). Omni does not support seeds or negative prompts.

    Args:
        ctx: MCP context with application state
        prompt: Text description of the video to generate
        image_uris: Optional input image URIs to condition on (gs://, http(s)://,
            file:// within DATA_FOLDER)
        input_video_uri: Optional video to edit (uploaded via the Files API)
        aspect_ratio: "16:9" (default) or "9:16". Output is always 720p.
        duration_seconds: Desired duration, clamped to 3-10s (default 6)
        previous_interaction_id: Continue a prior omni interaction to edit its
            video conversationally (see also the edit_video tool)
        output_gcs_uri: Optional gs:// destination for the video (Vertex only;
            ignored with a warning on the Gemini API, which returns bytes
            inline). Must be in the configured bucket allowlist.
        timeout_seconds: Overall deadline for the render (create + polling).
            Generation typically takes over a minute; raise for long queues.

    Returns:
        JSON with video_url, interaction_id (pass to edit_video / this tool to
        keep editing), and generation details.
    """
    try:
        app_ctx = ctx.request_context.lifespan_context
        data_dir = app_ctx.data_folder

        if output_gcs_uri:
            bucket = _parse_gcs_bucket(output_gcs_uri)
            if bucket is None:
                raise ValueError(
                    f"output_gcs_uri must start with gs://: {output_gcs_uri}"
                )
            _assert_gcs_bucket_allowed(bucket, app_ctx.allowed_gcs_buckets)

        image_bytes_list: list[bytes] = []
        if image_uris:
            # Cap the list: every image is buffered in memory (each up to
            # MAX_FETCH_BYTES), and omni conditions on a small set of images.
            if len(image_uris) > 8:
                raise ValueError(
                    f"Too many image_uris ({len(image_uris)}); "
                    "gemini-omni-flash accepts at most 8 input images here."
                )
            for uri in image_uris:
                b = await fetch(
                    uri,
                    allowed_dir=data_dir,
                    allowed_gcs_buckets=app_ctx.allowed_gcs_buckets,
                )
                if b is None:
                    raise ValueError(f"Could not fetch image_uri: {uri}")
                image_bytes_list.append(b)

        input_video_bytes = None
        if input_video_uri:
            input_video_bytes = await fetch(
                input_video_uri,
                allowed_dir=data_dir,
                allowed_gcs_buckets=app_ctx.allowed_gcs_buckets,
            )
            if input_video_bytes is None:
                raise ValueError(f"Could not fetch input_video_uri: {input_video_uri}")

        await ctx.info("Generating video with gemini-omni-flash")
        result = await _omni_generate_and_manifest(
            app_ctx,
            ctx,
            prompt=prompt,
            image_bytes_list=image_bytes_list or None,
            input_video_bytes=input_video_bytes,
            previous_interaction_id=previous_interaction_id,
            aspect_ratio=aspect_ratio,
            duration_seconds=duration_seconds,
            output_gcs_uri=output_gcs_uri,
            timeout_seconds=timeout_seconds,
            manifest_extra={
                "image_uris": image_uris,
                "input_video_uri": input_video_uri,
            },
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        await ctx.error(f"Omni video generation failed: {e}")
        logger.exception("Tool error")
        return json.dumps({"error": str(e)})


@mcp.tool()
async def edit_video(
    ctx: Context[ServerSession, AppContext],
    previous_interaction_id: str,
    prompt: str,
    aspect_ratio: str = "16:9",
    duration_seconds: float = 6.0,
    timeout_seconds: int = 600,
) -> str:
    """Conversationally edit a video generated by gemini-omni-flash.

    Pass the `interaction_id` returned by a prior generate_video_omni (or
    edit_video) call plus an instruction describing only the change (e.g.
    "make the sky stormy", "remove the person on the left"). Omni holds the
    video context server-side, so unmentioned elements are preserved.
    Retention of the server-side context is limited (background interactions
    are kept ~14 days on Vertex; longer on the paid Gemini API) — do not
    assume an interaction_id stays editable indefinitely.

    Args:
        ctx: MCP context with application state
        previous_interaction_id: interaction_id from a prior omni result
        prompt: The edit instruction (describe only what should change)
        aspect_ratio: "16:9" (default) or "9:16"
        duration_seconds: Desired duration, clamped to 3-10s (default 6)
        timeout_seconds: Overall deadline for the edit render (default 600)

    Returns:
        JSON with the edited video_url and a new interaction_id for further edits.
    """
    try:
        app_ctx = ctx.request_context.lifespan_context

        await ctx.info(f"Editing video (interaction {previous_interaction_id})")
        result = await _omni_generate_and_manifest(
            app_ctx,
            ctx,
            prompt=prompt,
            previous_interaction_id=previous_interaction_id,
            aspect_ratio=aspect_ratio,
            duration_seconds=duration_seconds,
            timeout_seconds=timeout_seconds,
            manifest_extra={"kind": "omni_edit"},
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        await ctx.error(f"Video edit failed: {e}")
        logger.exception("Tool error")
        return json.dumps({"error": str(e)})


@mcp.tool()
async def loop_extend(
    ctx: Context[ServerSession, AppContext],
    video_uri: str,
    prompt: str = "continue the action",
    times: int = 1,
    model: VideoModel = "veo-3.1-generate-001",
    aspect_ratio: str = "16:9",
    include_audio: bool = True,
    output_gcs_uri: str | None = None,
) -> str:
    """Extend a Veo-generated video multiple times in one call.

    Each Veo extension continues the video by ~7 seconds; this chains them,
    feeding each extended result back in as the source for the next. Veo
    supports up to 20 extensions. Output is 720p. Not supported on Veo 3.1
    Lite. On Vertex AI, extension requires a GCS output target.

    Args:
        ctx: MCP context with application state
        video_uri: URI of the existing Veo video to extend (gs://, file://)
        prompt: What the continuation should depict
        times: Number of ~7s extensions to chain (1-20)
        model: Veo model (not the Lite model)
        aspect_ratio: Must match the source video (16:9 or 9:16)
        include_audio: Generate audio on the extended sections (default True,
            so extending an audio video doesn't go silent; Vertex only)
        output_gcs_uri: GCS output URI (required on Vertex for extensions)

    Returns:
        JSON with the final video_url and the ordered list of intermediate
        extension URLs.
    """
    try:
        app_ctx = ctx.request_context.lifespan_context
        data_dir = app_ctx.data_folder

        if times < 1 or times > 20:
            raise ValueError("times must be between 1 and 20.")
        if model in _VEO_LITE_MODELS:
            raise ValueError("Veo 3.1 Lite does not support video extension.")
        _validate_aspect_ratio(aspect_ratio)

        video_client = _client_for_video_model(app_ctx, model)
        is_vertex_client = getattr(video_client._api_client, "vertexai", False)
        gcs_uri = _resolve_video_gcs(
            output_gcs_uri,
            app_ctx.video_gcs_bucket,
            app_ctx.allowed_gcs_buckets,
            is_vertex_client,
        )
        if is_vertex_client and not gcs_uri:
            raise ValueError(
                "Video extension on Vertex AI requires output_gcs_uri (or a "
                "configured VIDEO_GCS_BUCKET)."
            )

        # Validate the initial gs:// source against the allowlist (intermediate
        # outputs land in the already-validated gcs_uri bucket).
        if video_uri.startswith("gs://"):
            src_bucket = _parse_gcs_bucket(video_uri)
            if src_bucket is None:
                raise ValueError(f"Invalid video_uri: {video_uri}")
            _assert_gcs_bucket_allowed(src_bucket, app_ctx.allowed_gcs_buckets)

        current = video_uri
        steps: list[str] = []
        step_warnings: list[str] = []
        for i in range(times):
            await ctx.info(f"Extension {i + 1}/{times}")
            ext_result = await generate_video_impl(
                client=video_client,
                prompt=prompt,
                videos_dir=app_ctx.videos_dir,
                model=model,
                extend_video_uri=current,
                allowed_dir=data_dir,
                aspect_ratio=aspect_ratio,
                include_audio=include_audio,
                log_callback=ctx.info,
                output_gcs_uri=gcs_uri,
            )
            current = ext_result.get("video_url", "")
            if not current:
                raise ValueError(f"Extension {i + 1} produced no video URL.")
            steps.append(current)
            step_warnings.extend(ext_result.get("warnings") or [])

        manifest: dict[str, Any] = {
            "kind": "loop_extend",
            "source_video_uri": video_uri,
            "prompt": prompt,
            "model": model,
            "aspect_ratio": aspect_ratio,
            "times": times,
            "final_video_url": current,
            "extension_steps": steps,
        }
        result: dict[str, Any] = {
            "message": f"Extended video {times} time(s)",
            "video_url": current,
            "model": model,
            "times": times,
            "extension_steps": steps,
        }
        if step_warnings:
            # Same warning repeats per step; dedupe, preserving order.
            deduped = list(dict.fromkeys(step_warnings))
            result["warnings"] = deduped
            manifest["warnings"] = deduped
        sidecar_url = _write_sidecar(current, manifest)
        if sidecar_url:
            result["sidecar_url"] = sidecar_url
        else:
            result["manifest"] = manifest
        return json.dumps(result, indent=2)
    except Exception as e:
        await ctx.error(f"Loop extend failed: {e}")
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
