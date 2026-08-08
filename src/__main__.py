"""MCP server for Gemini media generation."""

import argparse
import asyncio
import base64
import ipaddress
import json
import logging
import math
import os
import socket
import sys
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from stat import S_ISREG
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
from .omni import OMNI_MAX_DURATION_SECONDS, OMNI_MODEL
from .routing import BudgetPreference, MediaKind
from .storyboard import Theme
from .omni import generate_video_omni as generate_video_omni_impl
from .video import _VEO_LITE_MODELS, VideoModel
from .video import generate_video as generate_video_impl
from .video_utils import (
    assert_frame_decoding_available,
    extract_frame_png,
    measure_video_duration,
)

logger = logging.getLogger(__name__)

# 50 MB cap on any single fetch to prevent memory/disk exhaustion.
MAX_FETCH_BYTES = 50 * 1024 * 1024

# Maximum number of HTTP redirects to follow during a fetch. Each hop's
# target is re-validated against the SSRF guard before it is requested.
MAX_HTTP_REDIRECTS = 5

# Upper bound on shots in one storyboard. Every shot is a billed image
# generation, so an unbounded list is an unbounded bill; exceeding this is an
# error rather than a silent truncation.
MAX_STORYBOARD_SHOTS = 24

# Upper bound on beats in one clip. Same reasoning as the storyboard cap, but
# it matters far more here: a beat is a Veo render costing roughly a hundred
# times an image and taking minutes, and add_bridges nearly doubles the count.
# Matches loop_extend's existing ceiling of 20 chained renders.
MAX_CLIP_BEATS = 20


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
    # Cost from the duration the interaction actually rendered (clamped by
    # the impl), not the request — covers omni, edit_video and draft mode.
    # Prefer the rendered artifact over any inference. Everything else here is
    # either the caller's request or a figure the server wrote down earlier,
    # and an edit seeds the next edit's estimate — so a wrong assumption
    # propagates down the whole chain. Measuring settles what actually
    # rendered; an unmeasurable edit bills the service maximum as a labelled
    # upper bound, never a guessed figure presented as fact.
    raw_duration: Any = result.get("duration_seconds")
    effective_duration: float | None = (
        float(raw_duration) if isinstance(raw_duration, (int, float)) else None
    )
    duration_source: str | None = None

    video_url = result.get("video_url") or ""
    if isinstance(video_url, str) and video_url.startswith("file://"):
        measured = await asyncio.to_thread(measure_video_duration, Path(video_url[7:]))
        if measured is not None:
            effective_duration = measured
            duration_source = "measured from the rendered video"

    billed_upper_bound = False
    if effective_duration is None and previous_interaction_id:
        # An edit whose render cannot be measured (e.g. gs:// delivery). Two
        # measurements put the render at Omni's maximum regardless of the
        # source (3.00s and 3.01s sources both rendered 10.01s), so billing
        # the source's length here would under-bill ~3.3x — the same falsified
        # inherit model, resurrected on the one branch measurement cannot
        # reach. Bill the maximum as an upper bound and label it an estimate.
        effective_duration = float(OMNI_MAX_DURATION_SECONDS)
        billed_upper_bound = True
        duration_source = (
            "upper bound: the render could not be measured, and an edit's "
            f"length is chosen by the service, so this bills Omni's "
            f"{OMNI_MAX_DURATION_SECONDS}s maximum rather than a figure "
            "inherited from the request or the source"
        )
    elif duration_source is None:
        # Delivered somewhere this process cannot open (a gs:// URI), so the
        # render was never inspected. A FRESH omni request is honoured at its
        # clamped length, so that figure is the right basis — but it is what
        # was asked for, not what came back, and omni renders land marginally
        # over it. Price it as an estimate carrying the encoder allowance,
        # and say which it is rather than leaving the number unattributed.
        duration_source = (
            "the clamped request: the rendered file could not be opened to "
            "measure it (delivered to a remote URI), so this is the length "
            "asked for, not the length measured"
        )

    duration_is_measured = duration_source == "measured from the rendered video"
    if effective_duration is not None:
        result["duration_seconds"] = effective_duration
    if duration_source:
        result["duration_source"] = duration_source

    cost = _video_cost(
        result.get("model") or OMNI_MODEL,
        effective_duration
        if effective_duration is not None
        else float(duration_seconds),
        resolution="720p",
        include_audio=False,
        # Only a measured length is a metered cost; anything else is an
        # estimate and must say so rather than presenting a bound as fact.
        actual=duration_is_measured,
        presnapped=not duration_is_measured,
    )
    if cost:
        if not duration_is_measured:
            # Two different unmeasured cases, and saying "the service maximum"
            # for the one that prices the request would overstate the bill by
            # 3x on the response an operator reconciles against.
            cost = dict(cost)
            basis = (
                "bills the service maximum as an upper bound, not the metered length"
                if billed_upper_bound
                else "prices the length requested, not a metered length"
            )
            cost["detail"] = (
                f"{cost['detail']} — the render could not be measured, so this {basis}"
            )
        result["cost"] = cost
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
    if duration_source:
        manifest["duration_source"] = duration_source
    manifest_cost = result.get("cost")
    if manifest_cost:
        # Same as the Veo tools and loop_extend: the sidecar is what an
        # operator reconciles against later, so it carries the figure and the
        # provenance, not the figure alone.
        manifest["cost"] = manifest_cost
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


def _validate_duration_seconds(
    duration_seconds: float | None,
    field: str = "duration_seconds",
) -> None:
    """Raise ValueError for a duration no model could render.

    The impls snap a requested duration to a length the model supports, and
    the snap is nearest-match — so a negative value quietly became the 4s
    minimum and was generated and billed. Pricing declines negatives, so
    without this the two layers disagreed: a dry run reported the request as
    unpriceable while the real call charged for it.

    Zero is deliberately allowed: it snaps to the model's minimum in both
    layers, so they still agree.
    """
    if duration_seconds is None:
        return
    try:
        value = float(duration_seconds)
    except (TypeError, ValueError):
        raise ValueError(
            f"{field} must be a number, got {duration_seconds!r}"
        ) from None
    # NaN slips through a plain < 0 check (all NaN comparisons are False),
    # would snap to the model minimum, generate and bill — and then render as
    # bare NaN in the response JSON, which strict parsers reject. Python's own
    # json.loads accepts NaN/Infinity, so these values genuinely arrive.
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite, got {value!r}")
    if value < 0:
        raise ValueError(f"{field} must not be negative, got {value:g}")


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


def _assert_gcs_bucket_allowed(bucket: str, allowed: frozenset[str]) -> str | None:
    """Reject gs:// buckets not in the allowlist, when one is configured.

    If no allowlist is configured (empty set), defers to ambient credentials
    and warns. This preserves backward compatibility but is noisy so operators
    notice.

    Returns:
        The warning text when no allowlist is configured, so a caller can put
        it in the response as well as the log — a server-side log is invisible
        to the MCP client, and the docstrings promise a warning. None when a
        bucket passed a configured allowlist.
    """
    if not allowed:
        message = (
            f"gs:// access to bucket '{bucket}' with no allowlist configured; "
            "set GCS_ALLOWED_BUCKETS or VIDEO_GCS_BUCKET to restrict access."
        )
        logger.warning("%s", message)
        return message
    if bucket not in allowed:
        raise ValueError(
            f"GCS bucket '{bucket}' is not in the allowlist. "
            f"Configured buckets: {sorted(allowed)}"
        )
    return None


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


def _cost_payload(estimate: Any) -> dict[str, Any] | None:
    """Shape a pricing CostEstimate for the MCP response, or None.

    Pricing is imported lazily and defensively: an unpriced model or a missing
    pricing table must degrade to "no cost reported" rather than fail a
    generation the caller already paid for.
    """
    if estimate is None:
        return None
    try:
        from .pricing import cost_to_dict

        return cost_to_dict(estimate)
    except Exception:  # pragma: no cover - defensive
        logger.debug("Could not serialize cost estimate", exc_info=True)
        return None


def _image_cost_estimate(
    model: str,
    image_size: str | None,
    *,
    usage: dict[str, Any] | None = None,
    n: int = 1,
) -> Any:
    """Raw pricing estimate for an image call, or None if unpriceable.

    Metered from reported usage when available, otherwise a unit estimate.
    Returns the pricing object (not a dict) so several calls can be summed.
    """
    try:
        from .pricing import actual_image_cost, estimate_image_cost

        size = image_size or "1K"
        if usage is not None:
            return actual_image_cost(model, usage, size, n)
        return estimate_image_cost(model, size, n)
    except Exception:  # pragma: no cover - defensive
        logger.debug("Could not compute image cost", exc_info=True)
        return None


def _image_cost(
    model: str,
    image_size: str | None,
    *,
    usage: dict[str, Any] | None = None,
    n: int = 1,
) -> dict[str, Any] | None:
    """Cost for an image call, shaped for the MCP response."""
    return _cost_payload(_image_cost_estimate(model, image_size, usage=usage, n=n))


# Bounds on the sidecar scan behind an edit_video quote. A media directory is
# user-controlled and may sit on a network mount, so the lookup is capped in
# every dimension rather than trusted to finish.
_SIDECAR_SCAN_LIMIT = 200
_SIDECAR_MAX_BYTES = 256 * 1024
_SIDECAR_SCAN_TIMEOUT_SECONDS = 5.0


def _source_duration_for_interaction(
    videos_dir: Path, interaction_id: str
) -> float | None:
    """Duration of a prior omni interaction, from the sidecars we wrote.

    Context and fallback only, never a prediction: an edit's rendered length
    is chosen by the service and matches neither the request nor the source
    (a measured 3s source rendered 10.01s). The omni manifest records
    ``interaction_id`` alongside ``duration_seconds``, so a prior render's own
    length is discoverable locally — no API call, which keeps a dry run free,
    instant and offline.

    Hardened because the media directory is caller-controlled and may live on
    a network mount: a named pipe among the sidecars made ``read_text`` block
    forever, hanging every edit_video quote. Only regular files are opened,
    reads are size-capped, and the scan stops after a fixed number of
    candidates. Anything unreadable is skipped, never fatal — the caller's
    fallback already reports an unknown source honestly.

    Args:
        videos_dir: Directory the video sidecars are written to.
        interaction_id: The interaction whose duration is wanted.

    Returns:
        The recorded duration in seconds, or None if it cannot be determined.
    """
    candidates: list[tuple[float, Path]] = []
    try:
        for entry in videos_dir.glob("*.json"):
            try:
                stat = entry.stat()
            except OSError:
                continue
            # Regular files only: a FIFO, socket or device node would block
            # the read indefinitely, and a directory would raise.
            if not S_ISREG(stat.st_mode) or stat.st_size > _SIDECAR_MAX_BYTES:
                continue
            candidates.append((stat.st_mtime, entry))
    except OSError:  # pragma: no cover - unreadable media dir
        return None

    # Newest first: an interaction id is chained forward by edits, so the most
    # recent manifest naming it is the closest ancestor of this edit.
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    for _mtime, sidecar in candidates[:_SIDECAR_SCAN_LIMIT]:
        try:
            manifest = json.loads(sidecar.read_text())
        except (OSError, ValueError):
            continue
        if manifest.get("interaction_id") != interaction_id:
            continue
        if str(manifest.get("duration_source", "")).startswith("upper bound"):
            # That render's length was never established — the manifest holds
            # the billing ceiling. Reporting it as the source's real length
            # would launder a bound into a fact one hop later.
            continue
        duration = manifest.get("duration_seconds")
        if isinstance(duration, (int, float)) and duration > 0:
            return float(duration)
    return None


async def _source_duration_or_none(
    videos_dir: Path, interaction_id: str
) -> float | None:
    """Run the sidecar lookup off the event loop, with a deadline.

    Even bounded, this is filesystem work: on a slow or stale mount it would
    stall the whole server, since one blocked coroutine blocks every request.
    A timeout degrades to the honest "source unknown" quote rather than
    hanging a call the caller was told is instant.

    Note the worker thread cannot be cancelled — if the underlying syscall is
    genuinely wedged the thread stays parked until the OS releases it. The
    request returns regardless, which is the property that matters here; the
    scan bounds keep the worst case to one thread per stuck edit quote.
    """
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(
                _source_duration_for_interaction, videos_dir, interaction_id
            ),
            timeout=_SIDECAR_SCAN_TIMEOUT_SECONDS,
        )
    except Exception:
        # Deliberately broad: the fallback quote is honest about not knowing
        # the source length, so nothing this lookup can do should fail a call
        # the caller was promised is free and instant.
        logger.warning(
            "Sidecar lookup for interaction %s failed or exceeded %.0fs; "
            "quoting the requested duration instead",
            interaction_id,
            _SIDECAR_SCAN_TIMEOUT_SECONDS,
            exc_info=True,
        )
        return None


def _video_cost_estimate(
    model: str,
    duration_seconds: float,
    *,
    resolution: str | None = None,
    include_audio: bool = True,
    actual: bool = False,
    presnapped: bool = False,
) -> Any:
    """Raw pricing estimate for a video call, or None if unpriceable.

    Two independent knobs, because "already the effective duration" and
    "was actually metered" are different facts:
      - ``actual=True``: a real run's effective duration — no re-snap, and
        is_estimate=False because the render happened.
      - ``presnapped=True``: a pre-flight quote for a duration this caller
        already clamped (loop_extend's 7s steps, omni's [3,10] clamp) — no
        re-snap, but still is_estimate=True: nothing ran.
    The default snaps the way the impl will, so a quote matches the bill.
    """
    try:
        import dataclasses

        from .pricing import actual_video_cost, estimate_video_cost

        res = resolution or "720p"
        if actual or presnapped:
            seconds = duration_seconds
            if presnapped and not actual:
                # A pre-clamped QUOTE still needs the encoder allowance; a
                # metered cost does not, since it prices a measured length.
                from .pricing import quote_duration_for

                seconds = quote_duration_for(model, seconds)
            cost = actual_video_cost(
                model, seconds, res, include_audio, snap_duration=False
            )
            if cost is not None and presnapped and not actual:
                cost = dataclasses.replace(cost, is_estimate=True)
            return cost
        return estimate_video_cost(model, duration_seconds, res, include_audio)
    except Exception:  # pragma: no cover - defensive
        logger.debug("Could not compute video cost", exc_info=True)
        return None


def _segment_is_metered(segment: dict[str, Any]) -> bool:
    """Whether a clip segment's recorded length is what actually rendered.

    Veo renders exactly the snapped length the impl sent it, so its segments
    are metered without a probe. Omni renders run marginally over the clamped
    request, so an omni segment is metered only when the file itself was
    measured — otherwise it is an estimate and carries the encoder allowance,
    the same rule every other omni surface follows.
    """
    if segment.get("duration_source") == "measured from the rendered video":
        return True
    return str(segment.get("model") or "") != OMNI_MODEL


def _video_cost(
    model: str,
    duration_seconds: float,
    *,
    resolution: str | None = None,
    include_audio: bool = True,
    actual: bool = False,
    presnapped: bool = False,
) -> dict[str, Any] | None:
    """Cost for a video call, shaped for the MCP response."""
    return _cost_payload(
        _video_cost_estimate(
            model,
            duration_seconds,
            resolution=resolution,
            include_audio=include_audio,
            actual=actual,
            presnapped=presnapped,
        )
    )


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
    dry_run: bool = False,
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
        dry_run: When True, generate nothing and return only the cost estimate
            and the resolved model/parameters. Free and instant — use it to
            price a call before committing to it. A real run reports the
            actual cost, derived from the token counts the API metered.
            Note: a quote covers output tokens only. Input tokens are not
            knowable before the call, so a multi-turn edit (which resends the
            conversation via thought_signature_url) costs slightly more than
            quoted — measured at ~1% on a real edit, 1527 input tokens against
            14 for a fresh call. The real run reports the metered figure.

    Returns:
        JSON with image_url, image_preview, and model info. For Gemini 3.x image models,
        includes thought_signature_url pointing to a file with editing context.
    """
    try:
        app_ctx = ctx.request_context.lifespan_context
        data_dir = app_ctx.data_folder

        if dry_run:
            # Resolve the model the same way the impl would, so the estimate
            # prices what would actually run rather than the requested alias.
            from .image import resolve_image_model

            resolved, plan_warnings, effective_size = resolve_image_model(
                model, image_size
            )
            payload: dict[str, Any] = {
                "dry_run": True,
                "message": "Estimate only — nothing was generated",
                "requested_model": model,
                "model": resolved,
                "prompt": prompt,
                "image_size": effective_size,
                "estimated_cost": _image_cost(resolved, effective_size),
            }
            if plan_warnings:
                payload["warnings"] = plan_warnings
                for warning in plan_warnings:
                    await ctx.warning(warning)
            return [TextContent(type="text", text=json.dumps(payload, indent=2))]

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

        # What this call actually cost, from the token counts the API metered.
        # Falls back to unit pricing when the response carried no usage.
        usage = result.get("usage")
        cost = _image_cost(result["model"], image_size, usage=usage)
        if cost:
            response_data["cost"] = cost
        if usage:
            response_data["usage"] = usage

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
            # The size actually used, which differs from the request when the
            # resolved model cannot produce it (see warnings).
            "image_size": result.get("image_size", image_size),
            "media_resolution": media_resolution,
            "aspect_ratio": aspect_ratio,
            "person_generation": person_generation,
            "reference_image_uris": reference_image_uris,
            "source_image_uri": image_uri,
            "thought_signature_url": result.get("thought_signature_url"),
        }
        if impl_warnings:
            manifest["warnings"] = impl_warnings
        if cost:
            manifest["cost"] = cost
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
async def plan_generation(
    ctx: Context[ServerSession, AppContext],
    intent: str,
    budget: BudgetPreference | None = None,
    media_kind: MediaKind | None = None,
    aspect_ratio: str | None = None,
    image_size: ImageSize | None = None,
    duration_seconds: float | None = None,
    num_beats: int | None = None,
    needs_text_rendering: bool | None = None,
    needs_4k: bool | None = None,
    needs_audio: bool | None = None,
    needs_extension: bool | None = None,
    num_reference_images: int | None = None,
    wants_gcs_output: bool | None = None,
    is_draft: bool | None = None,
    pinned_model: str | None = None,
):
    """Decide HOW to generate something before spending anything on it.

    Describe what you want in plain language and get back ranked, ready-to-call
    plans: which tool, which model, which parameters, why that model won, what
    each option costs, and which models were ruled out and for what reason.

    Call this first when you are unsure which of the generate_* tools to use or
    which model fits. It generates nothing, costs nothing, and is instant — it
    is pure rule-based routing over this server's capability tables, not a
    model call. It never replaces the explicit tools; it tells you how to
    drive them.

    It also catches requests that cannot work before you pay for the failure —
    4K on a 1K-only model, extending or first/last-frame on Veo Lite, GCS
    output on the Gemini API — and reports them as conflicts with a fix.

    Args:
        ctx: MCP context with application state
        intent: Plain-language description of what you want to make, e.g.
            "a 3-beat vertical reel about coffee" or "a poster with the words
            GRAND OPENING". Signals are inferred from this text.
        budget: "cheap", "balanced", or "best". Overrides anything inferred.
        media_kind: Force "image" or "video" instead of inferring it.
        aspect_ratio: Target aspect ratio, e.g. "16:9", "9:16".
        image_size: Target output size for images ("1K", "2K", "4K").
        duration_seconds: Target video runtime.
        num_beats: Number of shots, for multi-beat clip planning.
        needs_text_rendering: True when legible text must appear in the image.
        needs_4k: True when 4K output is required.
        needs_audio: True when generated audio is required.
        needs_extension: True when an existing video must be lengthened.
        num_reference_images: How many reference images you intend to supply.
        wants_gcs_output: True when output must land in GCS (Vertex only).
        is_draft: True when this is a rough pass, not a final render.
        pinned_model: A model you must use. Reported as a conflict if it
            cannot satisfy the request.

    Returns:
        JSON plan: ranked `routes` (each with tool, model, ready-to-use
        `params`, score, rationale, caveats, cost), `rejected` models with
        reasons, `conflicts`, a suggested multi-step `workflow`, and `notes`.
    """
    try:
        from dataclasses import asdict

        from .routing import RoutingConstraints, plan_generation as plan_impl

        # Tell the planner what this deployment can actually reach. Without
        # it, it recommended Veo Lite — Gemini-API-only — on a Vertex server
        # with no key, a route the real call refuses with exactly that
        # message. The server knows; the router is pure and cannot find out.
        app_ctx = ctx.request_context.lifespan_context
        primary_is_vertex = bool(
            getattr(getattr(app_ctx.client, "_api_client", None), "vertexai", False)
        )
        constraints = RoutingConstraints(
            backend="vertex" if primary_is_vertex else "gemini_api",
            gemini_api_key_available=(
                not primary_is_vertex or app_ctx.gemini_api_client is not None
            ),
            budget=budget,
            media_kind=media_kind,
            aspect_ratio=aspect_ratio,
            image_size=image_size,
            duration_seconds=duration_seconds,
            num_beats=num_beats,
            needs_text_rendering=needs_text_rendering,
            needs_4k=needs_4k,
            needs_audio=needs_audio,
            needs_extension=needs_extension,
            num_reference_images=num_reference_images,
            wants_gcs_output=wants_gcs_output,
            is_draft=is_draft,
            pinned_model=pinned_model,
        )
        plan = plan_impl(intent, constraints)

        def _route(route: Any) -> dict[str, Any]:
            return {
                "tool": route.tool,
                "model": route.model,
                "params": route.params,
                "score": route.score,
                "rationale": route.rationale,
                "caveats": list(route.caveats),
                "cost": _cost_payload(route.cost),
            }

        payload: dict[str, Any] = {
            "intent": plan.intent,
            "media_kind": plan.media_kind,
            "is_satisfiable": plan.is_satisfiable,
            "routes": [_route(r) for r in plan.routes],
            "rejected": [asdict(r) for r in plan.rejected],
            "conflicts": [asdict(c) for c in plan.conflicts],
            "workflow": [asdict(w) for w in plan.workflow],
            "notes": list(plan.notes),
        }
        await ctx.info(
            f"Planned {plan.media_kind}: {len(plan.routes)} route(s), "
            f"{len(plan.conflicts)} conflict(s)"
        )
        return [TextContent(type="text", text=json.dumps(payload, indent=2))]
    except Exception as e:
        await ctx.error(f"Planning failed: {e}")
        logger.exception("Tool error")
        return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]


@mcp.tool()
async def generate_storyboard(
    ctx: Context[ServerSession, AppContext],
    shots: list[dict[str, Any]],
    title: str = "Storyboard",
    subtitle: str | None = None,
    model: ImageModel | RetiredImageModel = "gemini-3.1-flash-image",
    aspect_ratio: str = "16:9",
    image_size: ImageSize | None = None,
    theme: Theme = "dark",
    dry_run: bool = False,
):
    """Generate a keyframe per shot and return a real, reviewable storyboard.

    The missing step between an idea and `generate_clip`: render one keyframe
    for every shot, then compose them into an actual storyboard you can read —
    numbered panels with slug lines, prompts, camera notes and duration
    badges — instead of a bare list of image URLs.

    Two artifacts come back, because MCP clients render inline images but do
    not execute HTML:
      1. A composited contact-sheet PNG, returned inline. This is the thing
         you look at.
      2. A self-contained HTML page written to disk (open the file:// URL in a
         browser) with full-size frames, complete prompt text and cumulative
         timecode.

    A shot whose generation fails does not abort the board: it renders as a
    clearly marked panel showing the error, so a partial storyboard stays
    reviewable. The shot list is designed to be fed straight into
    `generate_clip` as `beats` once the board reads well.

    Args:
        ctx: MCP context with application state
        shots: Ordered shot specs. Each accepts:
            {prompt: str, caption?: str, duration_seconds?: float,
             notes?: str}. `prompt` is required; `caption` is a short slug
            line, `notes` are camera/lighting notes.
        title: Board title drawn on the sheet.
        subtitle: Optional second line under the title.
        model: Image model for the keyframes.
        aspect_ratio: Frame aspect, e.g. "16:9" for landscape, "9:16" for a
            vertical reel. Drives the panel shape on the sheet.
        image_size: Keyframe resolution. Leave unset for the model default —
            storyboard frames rarely need to be large.
        theme: "dark" (default) or "light" board styling.
        dry_run: Estimate the cost of the whole board and generate nothing.

    Returns:
        The contact sheet inline, plus JSON with storyboard_url (HTML),
        sheet_url (PNG), per-shot results, total cost and total runtime.
    """
    try:
        app_ctx = ctx.request_context.lifespan_context

        if not shots:
            raise ValueError("shots list must not be empty")
        # Each shot is a paid image generation, so an oversized board is a real
        # bill. Refuse loudly rather than truncating: silently dropping shots
        # would render a board that looks complete but is not.
        if len(shots) > MAX_STORYBOARD_SHOTS:
            raise ValueError(
                f"shots has {len(shots)} entries; the limit is "
                f"{MAX_STORYBOARD_SHOTS} because every shot is a billed image "
                "generation. Split the sequence into several storyboards, or "
                "call generate_storyboard with dry_run=True first to price it."
            )
        # Validate every field BEFORE generating anything. A bad duration or a
        # non-string caption would otherwise surface while assembling the board
        # — after every keyframe had already been generated and billed.
        for i, shot in enumerate(shots):
            if not isinstance(shot, dict):
                raise ValueError(
                    f"shots[{i}] must be an object like "
                    '{"prompt": "...", "caption": "...", "duration_seconds": 4}, '
                    f"got {type(shot).__name__}"
                )
            if not str(shot.get("prompt", "")).strip():
                raise ValueError(f"shots[{i}] is missing a non-empty 'prompt'")
            for field in ("prompt", "caption", "notes"):
                value = shot.get(field)
                if value is not None and not isinstance(value, str):
                    raise ValueError(
                        f"shots[{i}].{field} must be a string, "
                        f"got {type(value).__name__}"
                    )
            _validate_duration_seconds(
                shot.get("duration_seconds"), f"shots[{i}].duration_seconds"
            )

        from .image import resolve_image_model

        resolved, plan_warnings, effective_size = resolve_image_model(model, image_size)

        if dry_run:
            # Price the whole board in one call. Multiplying a single-frame
            # estimate would leave `breakdown` describing one image while the
            # total described N — an inconsistency a caller could act on.
            total = _image_cost(resolved, effective_size, n=len(shots))
            payload: dict[str, Any] = {
                "dry_run": True,
                "message": "Estimate only — nothing was generated",
                "shots": len(shots),
                "requested_model": model,
                "model": resolved,
                "image_size": effective_size,
                "estimated_cost": total,
            }
            if plan_warnings:
                payload["warnings"] = plan_warnings
            return [TextContent(type="text", text=json.dumps(payload, indent=2))]

        from .storyboard import StoryboardFrame, render_contact_sheet, write_storyboard

        frames: list[StoryboardFrame] = []
        shot_results: list[dict[str, Any]] = []
        costs: list[Any] = []
        warnings_seen: list[str] = list(plan_warnings)

        # Log any substitution once for the whole board. The per-shot calls
        # below are handed the already-resolved model, so the impl has nothing
        # left to reroute — otherwise a 24-shot board pinned to a retired ID
        # would emit 24 identical warnings.
        for warning in plan_warnings:
            logger.warning("%s", warning)

        for i, shot in enumerate(shots, start=1):
            prompt = str(shot["prompt"])
            duration = shot.get("duration_seconds")
            await ctx.info(f"Storyboard shot {i}/{len(shots)}")
            try:
                result = await generate_image_impl(
                    client=app_ctx.client,
                    prompt=prompt,
                    images_dir=app_ctx.images_dir,
                    model=resolved,
                    image_size=effective_size,
                    aspect_ratio=aspect_ratio,
                )
                image_url = result.get("image_url")
                frame_bytes: bytes | None = None
                if image_url:
                    frame_bytes = Path(image_url[7:]).read_bytes()
                raw_cost = _image_cost_estimate(
                    result.get("model", resolved),
                    effective_size,
                    usage=result.get("usage"),
                )
                cost = _cost_payload(raw_cost)
                if raw_cost is not None:
                    costs.append(raw_cost)
                for warning in result.get("warnings", []):
                    if warning not in warnings_seen:
                        warnings_seen.append(warning)
                if frame_bytes is None:
                    raise ValueError(
                        result.get("generated_text")
                        or "model returned no image for this shot"
                    )
                frames.append(
                    StoryboardFrame(
                        index=i,
                        image_bytes=frame_bytes,
                        prompt=prompt,
                        caption=shot.get("caption"),
                        duration_seconds=duration,
                        notes=shot.get("notes"),
                        image_url=image_url,
                    )
                )
                shot_results.append({"shot": i, "image_url": image_url, "cost": cost})
            except Exception as shot_error:  # keep the board reviewable
                logger.warning("Storyboard shot %d failed: %s", i, shot_error)
                frames.append(
                    StoryboardFrame(
                        index=i,
                        image_bytes=None,
                        prompt=prompt,
                        caption=shot.get("caption"),
                        duration_seconds=duration,
                        notes=shot.get("notes"),
                        error=str(shot_error),
                    )
                )
                shot_results.append({"shot": i, "error": str(shot_error)})

        # Rendering a large board is a few hundred ms of CPU; keep the event
        # loop free while it runs.
        artifacts = await asyncio.to_thread(
            write_storyboard,
            frames,
            app_ctx.images_dir,
            title=title,
            subtitle=subtitle,
            theme=theme,
        )
        # A narrower sheet for the inline copy: the full-size board is on disk.
        inline_png = await asyncio.to_thread(
            render_contact_sheet,
            frames,
            title=title,
            subtitle=subtitle,
            theme=theme,
            max_sheet_width=1200,
        )

        failed = [r for r in shot_results if "error" in r]
        total_runtime = sum(float(s.get("duration_seconds") or 0) for s in shots)
        response_data: dict[str, Any] = {
            "message": f"Storyboard rendered: {len(shots) - len(failed)}/{len(shots)} shots",
            "storyboard_url": artifacts["html_url"],
            "sheet_url": artifacts["sheet_url"],
            "model": resolved,
            "aspect_ratio": aspect_ratio,
            "shots": shot_results,
            "total_duration_seconds": total_runtime,
        }
        if failed:
            response_data["errors"] = failed
        if warnings_seen:
            response_data["warnings"] = warnings_seen
            for warning in warnings_seen:
                await ctx.warning(warning)
        if costs:
            try:
                from .pricing import sum_costs

                response_data["cost"] = _cost_payload(
                    sum_costs(costs, label="storyboard")
                )
            except Exception:  # pragma: no cover - defensive
                logger.debug("Could not total storyboard cost", exc_info=True)

        await ctx.info("Storyboard complete")
        return [
            Image(data=inline_png, format="png"),
            TextContent(type="text", text=json.dumps(response_data, indent=2)),
        ]
    except Exception as e:
        await ctx.error(f"Storyboard failed: {e}")
        logger.exception("Tool error")
        return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]


@mcp.tool()
async def generate_video(
    ctx: Context[ServerSession, AppContext],
    prompt: str,
    model: VideoModel,
    aspect_ratio: str = "16:9",
    duration_seconds: float = 8.0,
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
    dry_run: bool = False,
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
        dry_run: When True, generate nothing and return only the cost
            estimate for the call that would run (the omni draft price when
            draft=True). Free and instant. A real run reports the actual
            cost, derived from the effective duration the API rendered.
        draft: When True, route to gemini-omni-flash for a fast 720p draft
            instead of Veo, then re-run with draft=False to finalize.
            Faster, but NOT cheaper than veo-3.1-fast-generate-001: omni is
            $0.10136/s against Fast's $0.10/s. The saving is real only against
            veo-3.1-generate-001 ($0.40/s), so use draft for speed and to
            avoid burning a full-fidelity render on a bad idea — not to spend
            less than Fast. Omni ignores Veo-only controls (seed,
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
        _validate_duration_seconds(duration_seconds)

        if dry_run:
            if not draft:
                # Same checks the impl applies, so a quote can never succeed
                # for a render the real call would refuse. The mode is derived
                # the way the impl derives it — a Lite quote used to price
                # extension, reference and first/last-frame calls it cannot
                # serve, because only the resolution rule was shared.
                from .video import validate_render_options

                if extend_video_uri:
                    quoted_mode = "extend_video"
                elif reference_image_uris:
                    quoted_mode = "reference_to_video"
                elif (image_uri or image_base64) and (
                    last_frame_uri or last_frame_base64
                ):
                    quoted_mode = "first_last_frame"
                elif image_uri or image_base64:
                    quoted_mode = "image_to_video"
                else:
                    quoted_mode = "text_to_video"
                validate_render_options(model, resolution, quoted_mode)
            est_model = OMNI_MODEL if draft else model
            est_res = "720p" if draft else (resolution or "720p")
            # Report the duration that will actually render. Returning the
            # request beside a price for the snapped value (5s quoted as "4s
            # of video") made the payload contradict itself.
            try:
                from .pricing import snap_video_duration

                quoted_duration: float = snap_video_duration(
                    est_model, duration_seconds
                )
            except Exception:  # pragma: no cover - defensive
                quoted_duration = duration_seconds
            payload: dict[str, Any] = {
                "dry_run": True,
                "message": "Estimate only — nothing was generated",
                "model": est_model,
                "resolution": est_res,
                "requested_duration_seconds": duration_seconds,
                "duration_seconds": quoted_duration,
                "estimated_cost": _video_cost(
                    est_model,
                    duration_seconds,
                    resolution=est_res,
                    include_audio=include_audio,
                ),
            }
            return json.dumps(payload, indent=2)

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
        # Cost from what actually rendered where that can be measured, else
        # the snapped duration the impl sent to Veo. Measuring is the only
        # figure that cannot drift from the request or a stale record.
        rendered_url = result.get("video_url") or ""
        if isinstance(rendered_url, str) and rendered_url.startswith("file://"):
            measured = await asyncio.to_thread(
                measure_video_duration, Path(rendered_url[7:])
            )
            if measured is not None:
                result["duration_seconds"] = round(measured, 3)
                result["duration_source"] = "measured from the rendered video"
        cost = _video_cost(
            result.get("model", model),
            float(result.get("duration_seconds", duration_seconds)),
            resolution=resolution or "720p",
            include_audio=result.get("audio_enabled", include_audio),
            actual=True,
        )
        if cost:
            result["cost"] = cost
        manifest: dict[str, Any] = {
            "kind": "video",
            "prompt": prompt,
            "audio_prompt": audio_prompt,
            "negative_prompt": negative_prompt,
            # The model that actually ran: the impl translates Veo IDs
            # per backend (the Gemini API serves -preview spellings).
            "model": result.get("model", model),
            "aspect_ratio": aspect_ratio,
            "duration_seconds": result.get("duration_seconds", duration_seconds),
            "resolution": resolution,
            "person_generation": person_generation,
            "audio_enabled": result.get("audio_enabled", include_audio),
            "generation_mode": result.get("generation_mode"),
            "seed": seed,
            "video_url": result.get("video_url"),
            "cost": cost,
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
    dry_run: bool = False,
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
        duration_seconds: Transition length. Veo renders 4, 6 or 8s only;
            others snap to the nearest, ties down (5 -> 4, 7 -> 6).
        aspect_ratio: 16:9 or 9:16 (must match clip aspect for clean cuts)
        include_audio: Generate transitional audio
        audio_prompt: Audio description
        negative_prompt: Things to avoid in the transition
        seed: Random seed for reproducibility
        output_gcs_uri: GCS output URI for large videos

        dry_run: When True, return only the cost estimate for the Veo render
            that would run (first/last frame are not fetched). A real run reports
            the metered cost of the snapped duration.

    Returns:
        JSON with video_url, sidecar_url, and generation metadata.
    """
    try:
        app_ctx = ctx.request_context.lifespan_context
        data_dir = app_ctx.data_folder

        # Fail fast on an unsupported aspect ratio before any fetch work.
        _validate_aspect_ratio(aspect_ratio)
        _validate_duration_seconds(duration_seconds)

        if dry_run:
            # Both tools are first/last-frame renders by definition, which
            # Veo Lite cannot serve — the docstrings said so but nothing
            # enforced it on the quote path.
            from .video import validate_render_options

            validate_render_options(model, generation_mode="first_last_frame")
            return json.dumps(
                {
                    "dry_run": True,
                    "message": "Estimate only — nothing was generated",
                    "model": model,
                    "duration_seconds": duration_seconds,
                    "estimated_cost": _video_cost(
                        model,
                        duration_seconds,
                        resolution="720p",
                        include_audio=include_audio,
                    ),
                },
                indent=2,
            )

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

        # Cost from the snapped duration the impl actually sent to Veo.
        cost = _video_cost(
            result.get("model", model),
            float(result.get("duration_seconds", duration_seconds)),
            resolution="720p",
            include_audio=result.get("audio_enabled", include_audio),
            actual=True,
        )
        if cost:
            result["cost"] = cost
        manifest: dict[str, Any] = {
            "kind": "transition",
            "prompt": prompt,
            "audio_prompt": audio_prompt,
            "negative_prompt": negative_prompt,
            # The model that actually ran: the impl translates Veo IDs
            # per backend (the Gemini API serves -preview spellings).
            "model": result.get("model", model),
            "aspect_ratio": aspect_ratio,
            "duration_seconds": result.get("duration_seconds", duration_seconds),
            "audio_enabled": result.get("audio_enabled", include_audio),
            "generation_mode": result.get("generation_mode"),
            "seed": seed,
            "video_url": result.get("video_url"),
            "cost": cost,
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
    dry_run: bool = False,
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
        duration_seconds: Bridge length. Veo renders 4, 6 or 8s only;
            others snap to the nearest, ties down (5 -> 4, 7 -> 6).
        aspect_ratio: Must match source clips for a clean cut.
        include_audio: Generate transitional audio.
        audio_prompt: Audio description.
        negative_prompt: Things to avoid in the bridge.
        seed: Random seed for reproducibility.
        output_gcs_uri: GCS URI for large video output.

        dry_run: When True, return only the cost estimate for the Veo render
            that would run (source clips are not fetched). A real run reports
            the metered cost of the snapped duration.

    Returns:
        JSON with video_url, sidecar_url, and the source clip URIs.
    """
    try:
        app_ctx = ctx.request_context.lifespan_context
        data_dir = app_ctx.data_folder

        # Fail fast on an unsupported aspect ratio before any fetch work.
        _validate_aspect_ratio(aspect_ratio)
        _validate_duration_seconds(duration_seconds)

        # Bridging decodes frames out of two videos, so it needs ffmpeg. This
        # is an environment fact rather than a model capability: without the
        # check the tool quoted a price and then died on an opaque decoder
        # error. Runs before the quote so both agree.
        assert_frame_decoding_available()

        if dry_run:
            # Both tools are first/last-frame renders by definition, which
            # Veo Lite cannot serve — the docstrings said so but nothing
            # enforced it on the quote path.
            from .video import validate_render_options

            validate_render_options(model, generation_mode="first_last_frame")

            # Report the environmental check positively. It runs above (and
            # refuses when ffmpeg is missing), but a passing check looks
            # identical to no check at all — which is exactly how a reviewer
            # on an ffmpeg-equipped host read it.
            preflight = ["ffmpeg available for frame decoding"]
            return json.dumps(
                {
                    "dry_run": True,
                    "message": "Estimate only — nothing was generated",
                    "model": model,
                    "duration_seconds": duration_seconds,
                    "preflight_checks": preflight,
                    "estimated_cost": _video_cost(
                        model,
                        duration_seconds,
                        resolution="720p",
                        include_audio=include_audio,
                    ),
                },
                indent=2,
            )

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

        # Cost from the snapped duration the impl actually sent to Veo.
        cost = _video_cost(
            result.get("model", model),
            float(result.get("duration_seconds", duration_seconds)),
            resolution="720p",
            include_audio=result.get("audio_enabled", include_audio),
            actual=True,
        )
        if cost:
            result["cost"] = cost
        manifest: dict[str, Any] = {
            "kind": "bridge",
            "prompt": prompt,
            "audio_prompt": audio_prompt,
            "negative_prompt": negative_prompt,
            # The model that actually ran: the impl translates Veo IDs
            # per backend (the Gemini API serves -preview spellings).
            "model": result.get("model", model),
            "aspect_ratio": aspect_ratio,
            "duration_seconds": result.get("duration_seconds", duration_seconds),
            "audio_enabled": result.get("audio_enabled", include_audio),
            "generation_mode": result.get("generation_mode"),
            "seed": seed,
            "video_url": result.get("video_url"),
            "cost": cost,
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
    dry_run: bool = False,
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
        dry_run: When True, price the whole reel — every beat, plus every
            bridge when add_bridges is set — and generate nothing. The single
            most useful pre-flight in the server: a clip is the most expensive
            call it can make.
        animatic: When True, render every beat with gemini-omni-flash (fast,
            720p) instead of Veo, for a quick storyboard preview of the
            whole reel before committing to full Veo renders. Bridges are not
            available in animatic mode (add_bridges is ignored), and Veo-only
            per-beat controls (seed, negative_prompt) are ignored. Each beat
            is measured from the rendered file, so an animatic's reported cost
            is metered and lands marginally above the quote's nominal seconds
            (omni overshoots the clamped request by about a frame per beat);
            beats delivered to GCS cannot be measured, and the total then says
            so by reporting itself as an estimate.

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
        if len(beats) > MAX_CLIP_BEATS:
            raise ValueError(
                f"beats has {len(beats)} entries; the limit is {MAX_CLIP_BEATS} "
                "because every beat is a billed Veo render, and add_bridges "
                "nearly doubles that. Split the sequence into several clips."
            )

        # Validate every beat before rendering any of them. A bad duration in
        # beat 5 would otherwise only surface after beats 1-4 had been
        # generated and billed.
        for beat_index, beat_spec in enumerate(beats):
            if not isinstance(beat_spec, dict):
                raise ValueError(
                    f"beats[{beat_index}] must be an object with a 'prompt', "
                    f"got {type(beat_spec).__name__}"
                )
            if not str(beat_spec.get("prompt", "") or "").strip():
                raise ValueError(f"beats[{beat_index}] is missing a non-empty 'prompt'")
            _validate_duration_seconds(
                beat_spec.get("duration_seconds"),
                f"beats[{beat_index}].duration_seconds",
            )

        # Validate the clip-level aspect ratio once, up front. Otherwise the
        # impl's per-value ValueError fires inside every beat's error handler,
        # producing a success-shaped manifest with zero segments instead of a
        # clear top-level failure.
        _validate_aspect_ratio(aspect_ratio)

        preflight: list[str] = []
        if add_bridges and not animatic:
            # Bridges decode frames out of the rendered beats; fail before any
            # beat is billed rather than losing every bridge mid-run. This is
            # the composite where a missing binary hurts most — it prices
            # bridges into the estimate — so the check is reported positively
            # too, exactly as generate_bridge does.
            assert_frame_decoding_available()
            preflight.append("ffmpeg available for frame decoding (bridges)")

        if dry_run:
            est_model = OMNI_MODEL if animatic else model
            beat_costs = [
                _video_cost_estimate(
                    est_model,
                    float(b.get("duration_seconds", 4.0)),
                    resolution="720p",
                    include_audio=include_audio and not animatic,
                )
                for b in beats
            ]
            bridge_count = len(beats) - 1 if (add_bridges and not animatic) else 0
            bridge_costs = [
                _video_cost_estimate(model, 4.0, resolution="720p", include_audio=False)
                for _ in range(bridge_count)
            ]
            try:
                from .pricing import sum_costs

                total = _cost_payload(
                    sum_costs(beat_costs + bridge_costs, label="clip")
                )
            except Exception:  # pragma: no cover - defensive
                total = None
            return json.dumps(
                {
                    "dry_run": True,
                    "message": "Estimate only — nothing was generated",
                    "model": est_model,
                    "beat_count": len(beats),
                    "bridge_count": bridge_count,
                    "preflight_checks": preflight,
                    "estimated_cost": total,
                },
                indent=2,
            )

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

            # Omni renders land marginally over the clamped request (3s
            # requested measured 3.01s), so an animatic beat's length is only
            # known by opening the file. Veo renders exactly the snapped
            # length it was sent (4s requested measured 4.0s) and needs no
            # probe. Without this, a 20-beat animatic reported a cost nobody
            # measured as though it were metered.
            beat_duration = float(beat_result.get("duration_seconds") or duration)
            beat_duration_source: str | None = None
            if animatic:
                beat_url_now = beat_result.get("video_url") or ""
                if isinstance(beat_url_now, str) and beat_url_now.startswith("file://"):
                    measured_beat = await asyncio.to_thread(
                        measure_video_duration, Path(beat_url_now[7:])
                    )
                    if measured_beat is not None:
                        beat_duration = measured_beat
                        beat_duration_source = "measured from the rendered video"

            beat_manifest = {
                "kind": "beat",
                "index": idx,
                "prompt": prompt,
                "model": beat_model,
                "aspect_ratio": aspect_ratio,
                "duration_seconds": beat_duration,
                "seed": None if animatic else seed,
                "video_url": beat_result.get("video_url"),
                "generation_mode": "animatic"
                if animatic
                else beat_result.get("generation_mode"),
                "interaction_id": beat_result.get("interaction_id"),
            }
            if beat_duration_source:
                beat_manifest["duration_source"] = beat_duration_source
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
                            "model": bridge_result.get("model", model),
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
        # Total what the segments actually rendered, segment by segment —
        # summing the runtime into one estimate would snap it to 8s.
        try:
            from .pricing import sum_costs

            segment_costs = [
                _video_cost_estimate(
                    seg.get("model", beat_model),
                    float(seg.get("duration_seconds") or 0),
                    resolution="720p",
                    include_audio=bool(seg.get("audio_enabled", include_audio)),
                    actual=_segment_is_metered(seg),
                    presnapped=not _segment_is_metered(seg),
                )
                for seg in segments
                if seg.get("video_url")
            ]
            if segment_costs:
                clip_manifest["cost"] = _cost_payload(
                    sum_costs(segment_costs, label="clip")
                )
        except Exception:  # pragma: no cover - defensive
            logger.debug("Could not total clip cost", exc_info=True)
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
    dry_run: bool = False,
) -> str:
    """Generate a video fast with gemini-omni-flash (Interactions API).

    Omni is the fast path (720p, 24fps): good for drafts and quick
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
            inline). Checked against the bucket allowlist when one is
            configured (GCS_ALLOWED_BUCKETS / VIDEO_GCS_BUCKET); with no
            allowlist set the server warns and defers to ambient
            credentials, so a dry run quotes rather than refusing.
        timeout_seconds: Overall deadline for the render (create + polling).
            Generation typically takes over a minute; raise for long queues.

        dry_run: When True, return only the cost estimate for the clamped
            duration and generate nothing.

    Returns:
        JSON with video_url, interaction_id (pass to edit_video / this tool to
        keep editing), and generation details.
    """
    try:
        app_ctx = ctx.request_context.lifespan_context
        # Fail fast before any fetch or interaction work; the omni impl
        # clamps to [3, 10]s, which would turn a negative or NaN duration
        # into a billed 3s render instead of an error.
        _validate_duration_seconds(duration_seconds)

        gcs_warnings: list[str] = []
        if output_gcs_uri:
            bucket = _parse_gcs_bucket(output_gcs_uri)
            if bucket is None:
                raise ValueError(
                    f"output_gcs_uri must start with gs://: {output_gcs_uri}"
                )
            allowlist_warning = _assert_gcs_bucket_allowed(
                bucket, app_ctx.allowed_gcs_buckets
            )
            if allowlist_warning:
                gcs_warnings.append(allowlist_warning)

        if dry_run:
            # Mirror the impl's clamp so the quote matches the render.
            clamped = min(10, max(3, round(duration_seconds)))
            return json.dumps(
                {
                    "dry_run": True,
                    "message": "Estimate only — nothing was generated",
                    "model": OMNI_MODEL,
                    "duration_seconds": clamped,
                    "estimated_cost": _video_cost(
                        OMNI_MODEL,
                        float(clamped),
                        resolution="720p",
                        include_audio=False,
                        presnapped=True,
                    ),
                    **({"warnings": gcs_warnings} if gcs_warnings else {}),
                },
                indent=2,
            )
        data_dir = app_ctx.data_folder

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
    dry_run: bool = False,
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
        aspect_ratio: NOT SENT for edits — the API rejects it on an edit task
            (verified on the wire), so it does not change the output.
        duration_seconds: NOT SENT for edits, and it does not determine the
            output length. The API rejects duration on an edit task, and the
            rendered length is chosen by the service: it is predictable from
            neither this value nor the source video's length. A measured 3s
            source edited with duration_seconds=4 rendered 10.01s. Google's
            Omni documentation does not state the rule.
        timeout_seconds: Overall deadline for the edit render (default 600)
        dry_run: When True, return only the cost estimate and generate
            nothing. Because the rendered length is unpredictable, the quote
            is Omni's 10s maximum as an upper bound — a pre-flight may
            over-state but must never under-state. The source's own length is
            reported separately as `source_duration_seconds` for context, and
            `duration_source` explains the basis. The real run measures the
            rendered file and bills that; when the render is delivered
            somewhere it cannot be opened (a gs:// URI) it bills the same 10s
            upper bound, labelled as an estimate rather than as metered.

    Returns:
        JSON with the edited video_url and a new interaction_id for further edits.
    """
    try:
        app_ctx = ctx.request_context.lifespan_context
        # Fail fast before any fetch or interaction work; the omni impl
        # clamps to [3, 10]s, which would turn a negative or NaN duration
        # into a billed 3s render instead of an error.
        _validate_duration_seconds(duration_seconds)

        if dry_run:
            # An edit inherits its duration from the source video, so the
            # caller's duration_seconds is not what gets billed — quoting it
            # overstated a 3s edit by 3.3x. Recover the real length from the
            # sidecar of the interaction being edited; fall back to the
            # request only when the source is unknown, and say so.
            # An edit's output length is decided by the service: it is not
            # the request (never sent) and not the source's length either — a
            # measured 3s source edited with duration_seconds=4 rendered
            # 10.01s. Google's Omni documentation does not state the rule, so
            # quote the worst case rather than a guess. Quoting the source's
            # length under-quoted that render 3.3x, and under-quoting defeats
            # the point of a pre-flight.
            source_duration = await _source_duration_or_none(
                app_ctx.videos_dir, previous_interaction_id
            )
            quoted = float(OMNI_MAX_DURATION_SECONDS)
            payload: dict[str, Any] = {
                "dry_run": True,
                "message": "Estimate only — nothing was generated",
                "model": OMNI_MODEL,
                "previous_interaction_id": previous_interaction_id,
                "duration_seconds": quoted,
                "duration_source": (
                    f"upper bound: an edit's rendered length is chosen by the "
                    f"service and is not predictable from the request or the "
                    f"source, so this quotes Omni's {OMNI_MAX_DURATION_SECONDS}s "
                    "maximum. The real run reports the measured duration."
                ),
            }
            if source_duration is not None:
                payload["source_duration_seconds"] = source_duration
            payload["estimated_cost"] = _video_cost(
                OMNI_MODEL,
                quoted,
                resolution="720p",
                include_audio=False,
                presnapped=True,
            )
            return json.dumps(payload, indent=2)

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
    dry_run: bool = False,
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
        dry_run: When True, return only the cost of the extension chain
            (times x ~7s at the model's rate) and generate nothing.

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

        if video_uri.startswith("gs://"):
            src_bucket = _parse_gcs_bucket(video_uri)
            if src_bucket is None:
                raise ValueError(f"Invalid video_uri: {video_uri}")
            _assert_gcs_bucket_allowed(src_bucket, app_ctx.allowed_gcs_buckets)

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

        if dry_run:
            # Each extension is a ~7s Veo render billed like any other.
            return json.dumps(
                {
                    "dry_run": True,
                    "message": "Estimate only — nothing was generated",
                    "model": model,
                    "times": times,
                    "added_seconds": times * 7,
                    "estimated_cost": _video_cost(
                        model,
                        float(times * 7),
                        resolution="720p",
                        include_audio=include_audio,
                        presnapped=True,  # 7s steps must not re-snap to 8
                    ),
                },
                indent=2,
            )

        # Validate the initial gs:// source against the allowlist (intermediate
        # outputs land in the already-validated gcs_uri bucket).

        current = video_uri
        steps: list[str] = []
        step_warnings: list[str] = []
        # The model that actually ran. Seeded with the request so it is bound
        # even if the loop body never executes, then overwritten with the
        # backend-translated ID the impl reports.
        served_model = str(model)
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
            served_model = ext_result.get("model", model)

        manifest: dict[str, Any] = {
            "kind": "loop_extend",
            "source_video_uri": video_uri,
            "prompt": prompt,
            # The model that actually ran: the impl translates Veo IDs
            # per backend (the Gemini API serves -preview spellings).
            "model": served_model,
            "aspect_ratio": aspect_ratio,
            "times": times,
            "final_video_url": current,
            "extension_steps": steps,
        }
        result: dict[str, Any] = {
            "message": f"Extended video {times} time(s)",
            "video_url": current,
            "model": served_model,
            "times": times,
            "extension_steps": steps,
        }
        cost = _video_cost(
            served_model,
            float(times * 7),
            resolution="720p",
            include_audio=include_audio,
            actual=True,
        )
        if cost:
            result["cost"] = cost
            manifest["cost"] = cost
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


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser. Split from main() so tests parse the real thing."""
    parser = argparse.ArgumentParser(description="Gemini Media MCP Server")
    parser.add_argument(
        "--mount-path",
        default=None,
        help="Mount path for SSE/HTTP transport (e.g., /mcp)",
    )
    parser.add_argument(
        "--host",
        default=None,
        help=(
            "Bind address for the sse/streamable-http transports. Defaults to "
            "0.0.0.0 in a container (so a published port is reachable) and "
            "127.0.0.1 otherwise."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Bind port for the sse/streamable-http transports (default 8000).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )

    subparsers = parser.add_subparsers(dest="command")

    # Transport subcommands preserve the existing positional behavior
    # (gemini-media-mcp [stdio|sse|streamable-http]). The network flags are
    # registered on the subparsers as well as the top level because argparse
    # only accepts top-level flags BEFORE the subcommand — and the Docker
    # ENTRYPOINT appends arguments after it, so `docker run <image> sse
    # --host 0.0.0.0` is the only form a container user can produce.
    # default=SUPPRESS so the subparser only writes these when actually
    # given — a plain default of None would clobber a value supplied before
    # the subcommand (`--host X sse` would silently lose X).
    network_flags = argparse.ArgumentParser(add_help=False)
    network_flags.add_argument("--host", default=argparse.SUPPRESS)
    network_flags.add_argument("--port", type=int, default=argparse.SUPPRESS)
    network_flags.add_argument("--mount-path", default=argparse.SUPPRESS)
    for transport in ("stdio", "sse", "streamable-http"):
        subparsers.add_parser(
            transport,
            help=f"Run the MCP server with {transport} transport",
            parents=[network_flags],
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

    return parser


def _resolve_http_host(cli_host: str | None) -> str:
    """Bind address for the sse/streamable-http transports.

    Explicit wins (CLI flag, then FASTMCP_HOST). Otherwise bind all
    interfaces in a container — 127.0.0.1 there is the container's own
    loopback, so a published port would reach nothing — and keep loopback
    on a direct run so it is not exposed to the network by surprise.
    """
    if cli_host:
        return cli_host
    env_host = os.environ.get("FASTMCP_HOST")
    if env_host:
        return env_host
    return "0.0.0.0" if is_running_in_container() else "127.0.0.1"  # noqa: S104


def main() -> None:
    """Entry point."""
    parser = _build_arg_parser()
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

    if transport in ("sse", "streamable-http"):
        # FastMCP binds 127.0.0.1 by default. Inside a container that is the
        # container's own loopback, so a published port reaches nothing —
        # which made the Dockerfile's own documented `-p 8000:8000` usage
        # impossible. Bind all interfaces there, but keep loopback elsewhere
        # so a local run is not exposed to the network by surprise.
        mcp.settings.host = _resolve_http_host(getattr(args, "host", None))
        if getattr(args, "port", None):
            mcp.settings.port = args.port  # type: ignore[union-attr]
        logger.info(
            "Serving %s on %s:%s", transport, mcp.settings.host, mcp.settings.port
        )

    mcp.run(transport=transport, mount_path=args.mount_path)


if __name__ == "__main__":
    main()
