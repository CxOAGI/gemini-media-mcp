"""MCP server for Gemini media generation."""

import argparse
import asyncio
import base64
import ipaddress
import hashlib
import json
import logging
import math
import os
import re
import socket
import sys
import time
import threading
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from stat import S_ISREG
from typing import Any, NamedTuple
from urllib.parse import urljoin, urlparse

import aiohttp
from aiohttp.abc import AbstractResolver, ResolveResult
from google import genai
from google.genai import types

from . import credentials
from .video import VEO_DEFAULT_TIMEOUT_SECONDS, VEO_MEASURED_4K_SECONDS
from google.cloud import storage
from mcp.server.fastmcp import Context, FastMCP, Image
from mcp.server.session import ServerSession
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import TextContent
from PIL import Image as PILImage

from .image import ImageModel, ImageSize, MediaResolution, RetiredImageModel
from .image import generate_image as generate_image_impl
from .omni import (
    OMNI_CONVERSATIONAL_EXTENSION_SENDS_DURATION,
    OmniSubmission,
    collect_omni_interaction,
    omni_extension_refusal,
    omni_model_is_retired,
    omni_model_reroute_warning,
    omni_model_refusal,
    OMNI_1_2_MODEL,
    OMNI_DEFAULT_TIMEOUT_SECONDS,
    OMNI_MAX_DURATION_SECONDS,
    OMNI_VERTEX_INLINE_VIDEO_BYTES,
    DEFAULT_OMNI_MODEL,
    OMNI_MODELS,
    is_omni_model,
    normalize_omni_resolution,
    omni_extension_appended_seconds,
    omni_extension_output_lengths,
    omni_extension_priced_lengths,
    omni_extension_turn_seconds,
    omni_duration_clamp_warning,
    omni_continuation_upper_bound,
    omni_duration_is_sent,
    omni_extension_duration_refusal,
    omni_inline_delivery_policy,
    omni_source_limit_seconds,
    omni_slow_tier_warning,
    omni_spec,
    validate_omni_request,
)
from .routing import BudgetPreference, MediaKind
from .storyboard import Theme
from .director import DEFAULT_MAX_RETAKES
from .omni import generate_video_omni as generate_video_omni_impl
from .video import (
    _MAX_REFERENCE_IMAGES,
    VEO_EXTENSION_STEP_SECONDS,
    VEO_GEMINI_API_REFUSAL,
    TranslatedVideoModel,
    VideoModel,
    http_status,
    retry_after_seconds,
    validate_video_request_inputs,
    veo_extension_output_lengths,
)
from .video import generate_video as generate_video_impl
from .video_utils import (
    assert_frame_decoding_available,
    classify_video_resolution,
    extract_frame_png,
    measure_video_dimensions,
    measure_video_duration,
    measure_video_duration_bytes,
    measure_video_metadata_bytes,
)

logger = logging.getLogger(__name__)

# 50 MB cap on any single fetch to prevent memory/disk exhaustion.
MAX_FETCH_BYTES = 50 * 1024 * 1024

# Maximum number of HTTP redirects to follow during a fetch. Each hop's
# target is re-validated against the SSRF guard before it is requested.
MAX_HTTP_REDIRECTS = 5

# Cap on a thought-signature file read. The path is caller-supplied and the
# strongest target is a large file the server itself wrote into DATA_FOLDER —
# an uncapped read_text() of a 157 MB render added ~150 MB RSS in one call.
#
# "At most a few kilobytes" was the original premise and it is wrong. MEASURED:
# gemini-3-pro-image writes 2,828,040 bytes and gemini-3.1-flash-image
# 1,634,388, so a 256 KB cap refused every signature the server itself had just
# written and made multi-turn editing unreachable on both models — the
# docstring's own two-step example could not execute. Sized to the largest
# observed signature with generous headroom, still two orders of magnitude
# below the render that motivated the cap.
MAX_THOUGHT_SIGNATURE_BYTES = 16 * 1024 * 1024

# Upper bound on shots in one storyboard. Every shot is a billed image
# generation, so an unbounded list is an unbounded bill; exceeding this is an
# error rather than a silent truncation.
MAX_STORYBOARD_SHOTS = 24

# Per-field ceiling on the text a storyboard draws. Deliberately far above any
# real prompt: the panel shows about three wrapped lines, but `prompt` is also
# what the image model receives, and a detailed prompt is legitimately long, so
# this refuses only input that cannot be meant seriously. The layout cost is
# linear now (`_split_overlong` and `_ellipsize` binary-search their cuts), so
# this is a bound on memory and wasted work rather than the fix for the
# quadratic blow-up it used to be.
MAX_STORYBOARD_TEXT_CHARS = 20_000

# Upper bound on beats in one clip. Same reasoning as the storyboard cap, but
# it matters far more here: a beat is a Veo render costing roughly a hundred
# times an image and taking minutes, and add_bridges nearly doubles the count.
# Matches loop_extend's existing ceiling of 20 chained renders.
MAX_CLIP_BEATS = 20

# How long a bridge between two beats runs: one of Veo's lengths, and inside
# Omni 1.2's 3-10s first/last-frame range, so a full clip and its animatic
# bridge for the same time.
_CLIP_BRIDGE_SECONDS = 4.0

# Reference images a Gemini 3.x image model accepts; extras are truncated with
# a warning rather than silently dropped (the docstring's "up to 14").
_MAX_IMAGE_REFERENCE_IMAGES = 14


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
    # Dedicated Gemini API (AI Studio) client, used to reach omni on the
    # Gemini Developer API when the primary client runs in Vertex mode (see
    # _omni_backend_decision). None when no GEMINI_API_KEY is configured.
    gemini_api_client: genai.Client | None = None
    temp_creds_path: Path | None = None
    video_gcs_bucket: str | None = None  # Default GCS bucket for video output
    allowed_gcs_buckets: frozenset[str] = frozenset()  # Allowlist for gs:// URIs
    # Server-wide default for which backend serves omni: "auto", "vertex" or
    # "gemini_api" (env OMNI_BACKEND). A tool's omni_backend argument wins over
    # it. See _omni_backend_decision for what "auto" does and why this exists.
    omni_backend_default: str = "auto"


def _parse_gcs_bucket(uri: str) -> str | None:
    """Return the bucket name from a gs:// URI, or None if invalid."""
    if not uri.startswith("gs://"):
        return None
    remainder = uri[5:].split("/", 1)[0]
    return remainder or None


def _omni_gcs_source(
    app_ctx: AppContext, uri: str | None, backend: str, *, model: str
) -> str | None:
    """The gs:// input_video_uri to send as a URI part, or None to fetch it.

    Validates any gs:// source bound for Vertex (object path, bucket
    allowlist) so a dry run refuses what the render would. Only a model whose
    spec reads a gs:// source by URI (gemini-omni-1.2-flash) gets one; the
    preview model keeps the inline-bytes wire shape it was verified with, and
    with it the fetch. A URI source is never inlined, so a caller that needs
    its length downloads it with _fetch_omni_gcs_source_to_measure.

    Call once per tool call and reuse the result: each call logs the
    no-allowlist warning.
    """
    if not uri or backend != "vertex" or not uri.startswith("gs://"):
        return None
    bucket = _parse_gcs_bucket(uri)
    if not bucket or not uri[5:].partition("/")[2]:
        raise ValueError(
            "input_video_uri must name a Cloud Storage object: gs://bucket/video.mp4."
        )
    _assert_gcs_bucket_allowed(bucket, app_ctx.allowed_gcs_buckets)
    return uri if omni_spec(model).reads_gcs_source_by_uri else None


def _gcs_allowlist_warnings(app_ctx: AppContext, *uris: str | None) -> list[str]:
    """The no-allowlist warning for each gs:// bucket a call reads or writes.

    _assert_gcs_bucket_allowed returns this warning so callers can put it in
    the response, since its log line never reaches the MCP client. The omni
    tools surfaced it only for output_gcs_uri, so a gs:// input read with
    ambient credentials went unreported. One entry per bucket, in argument
    order. Never raises: refusing a bucket stays with the checks that already
    do it (_omni_gcs_source, the output check, fetch()).
    """
    if app_ctx.allowed_gcs_buckets:
        return []
    warnings: list[str] = []
    for uri in uris:
        bucket = _parse_gcs_bucket(uri) if isinstance(uri, str) else None
        if bucket:
            message = _gcs_no_allowlist_message(bucket)
            if message not in warnings:
                warnings.append(message)
    return warnings


_FILE_NAME_PATTERN = re.compile(r"/[^/]+\.[A-Za-z0-9]{2,4}$")


def _gcs_prefix_warning(output_gcs_uri: str | None) -> list[str]:
    """A warning for an output_gcs_uri that ends in a file name, or none.

    Vertex AI treats output_gcs_uri as a prefix: gs://bucket/omni/mug.mp4
    became a folder, holding gs://bucket/omni/mug.mp4/<id>.mp4. The response's
    video_url names the object either way, so this warns rather than refuses.
    """
    if not output_gcs_uri or not _FILE_NAME_PATTERN.search(output_gcs_uri):
        return []
    return [
        f"output_gcs_uri {output_gcs_uri} ends in a file name, but Vertex AI "
        "treats it as a prefix and writes the video to a new object under it "
        f"({output_gcs_uri}/<id>.mp4). Pass a prefix ending in /, such as "
        "gs://bucket/renders/; video_url names the object written."
    ]


# Where the interaction-id index lives, relative to the media directory. A
# subdirectory rather than a name prefix so the `*.json` sidecar glob -- which
# is not recursive -- cannot see these entries and mistake one for a manifest.
_INTERACTION_INDEX_DIRNAME = ".interactions"


def _interaction_index_path(media_dir: Path, interaction_id: str) -> Path:
    """Path of the index entry for ``interaction_id``.

    Hashed, not used verbatim: an interaction id is minted by the service and
    is not guaranteed to be a safe filename (a "/" in one would silently write
    outside the index, or fail).
    """
    digest = hashlib.sha256(interaction_id.encode("utf-8")).hexdigest()[:32]
    return media_dir / _INTERACTION_INDEX_DIRNAME / f"{digest}.json"


def _write_interaction_index(sidecar: Path, metadata: dict[str, Any]) -> None:
    """Record which sidecar holds ``interaction_id``, if there is one.

    Makes the by-id lookup a single read instead of a directory scan. Stores
    the sidecar's NAME rather than a copy of the manifest, so the manifest has
    exactly one home.

    Best-effort: a failure here costs the fast path and falls back to the
    scan, so it must never fail a render that already succeeded.
    """
    interaction_id = metadata.get("interaction_id")
    if not isinstance(interaction_id, str) or not interaction_id:
        return
    entry = _interaction_index_path(sidecar.parent, interaction_id)
    try:
        entry.parent.mkdir(parents=True, exist_ok=True)
        entry.write_text(
            json.dumps(
                {"interaction_id": interaction_id, "sidecar": sidecar.name},
                sort_keys=True,
            )
        )
    except OSError as e:  # pragma: no cover - index is an optimisation
        logger.warning("Failed to write interaction index %s: %s", entry, e)


def _write_remote_interaction_record(media_dir: Path, manifest: dict[str, Any]) -> None:
    """Record an interaction whose media did not land locally.

    _write_sidecar writes nothing for a gs:// delivery, so a Vertex omni render
    sent to a bucket left NO local trace of its interaction_id -- and
    PriorInteraction, which the index exists to serve, came back empty for
    exactly the Vertex-minted ids that must never be sent to the Gemini
    Developer API. A real edit_video(previous_interaction_id=<vertex id>) went
    out on the Gemini client. The manifest itself goes into the index entry
    here, since there is no sidecar to point at.
    """
    interaction_id = manifest.get("interaction_id")
    if not isinstance(interaction_id, str) or not interaction_id:
        return
    # Only the fields the by-id readers consume -- not the whole manifest,
    # whose size scales with the caller's prompt. A ~260 KB prompt pushed the
    # entry past _SIDECAR_MAX_BYTES, the reader skipped it, and with no
    # sidecar to fall back to the interaction was unfindable: exactly the
    # cross-backend hazard this record exists to close.
    record = {
        key: manifest[key]
        for key in (
            "interaction_id",
            "backend",
            "model",
            "duration_seconds",
            "duration_source",
            "duration_upper_bound_seconds",
            # Without these an upscale of a gs:// take refused it as
            # unrecorded while taking its recorded duration.
            "resolution",
            "resolution_source",
            "video_url",
            # A render this process stopped waiting for is recorded as
            # pending, with what get_omni_interaction needs to price it.
            "status",
            "task",
            "upscale_source_resolution",
        )
        if key in manifest
    }
    entry = _interaction_index_path(media_dir, interaction_id)
    try:
        entry.parent.mkdir(parents=True, exist_ok=True)
        entry.write_text(
            json.dumps(
                {"interaction_id": interaction_id, "manifest": record}, sort_keys=True
            )
        )
    except (OSError, TypeError, ValueError) as e:  # pragma: no cover
        logger.warning("Failed to record remote interaction %s: %s", entry, e)


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
    _write_interaction_index(sidecar, metadata)
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


# Credentials live in one shared module so image.py's Vertex client can be
# built the same way as every client here; see src/credentials.py.
_parse_service_account_json = credentials.parse_service_account_json
_service_account_credentials = credentials.service_account_credentials
_vertex_client_kwargs = credentials.vertex_client_kwargs


def setup_vertex_credentials() -> Path | None:
    """Validate the Vertex credentials configuration at startup.

    Kept for its name and its two callers; it no longer writes anything.
    Parsing the inline JSON here means a broken credential fails the lifespan
    before any client is built, as before. Always returns None: there is no
    temp file any more (see _service_account_credentials for why).
    """
    if os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() != "true":
        return None
    _parse_service_account_json()
    # Key text out of the path variable before anything can reach ADC.
    credentials.normalize_environment()
    return None


def cleanup_credentials(path: Path | None) -> None:
    """Remove a temporary credentials file, if one was ever written.

    None is what setup_vertex_credentials returns now; this stays as a no-op
    for that case so the lifespan's finally reads as it always did.
    """
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
        return genai.Client(**_vertex_client_kwargs())
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
    """Pick the genai.Client a Veo model runs on: the primary, on Vertex AI.

    Veo has no endpoint on the Gemini Developer API (VEO_GEMINI_API_SHUTDOWN),
    so a deployment whose primary client is not Vertex is refused here. Every
    Veo tool asks this before it quotes or fetches anything, so the refusal is
    free and a dry run cannot price a call that would 404.

    Raises:
        RuntimeError: When the primary client is not a Vertex AI client.
            ``_backend_of`` relies on this type to fall back to the primary
            client's mode.
    """
    if not getattr(app_ctx.client._api_client, "vertexai", False):
        raise RuntimeError(VEO_GEMINI_API_REFUSAL)
    return app_ctx.client


# Memoized global-location Vertex client for omni. On Vertex the omni
# interactions collection lives at .../locations/global/interactions, so a
# regional primary client won't reach it.
_omni_vertex_global_client: genai.Client | None = None


def _get_omni_vertex_global_client() -> genai.Client:
    """Return a memoized Vertex client pinned to the global location."""
    global _omni_vertex_global_client
    if _omni_vertex_global_client is None:
        _omni_vertex_global_client = genai.Client(
            **_vertex_client_kwargs(location="global")
        )
    return _omni_vertex_global_client


class _RenderedFacts(NamedTuple):
    """What a probe could establish about a file that actually rendered."""

    duration_seconds: float | None
    duration_source: str | None
    resolution_source: str | None


def _aspect_ratio_of(size: tuple[int, int]) -> str | None:
    """A frame's aspect ratio as this server spells it (16:9), or None."""
    width, height = size
    if width <= 0 or height <= 0:
        return None
    for name, ratio in (("16:9", 16 / 9), ("9:16", 9 / 16), ("1:1", 1.0)):
        if abs(width / height - ratio) < 0.02:
            return name
    divisor = math.gcd(width, height)
    return f"{width // divisor}:{height // divisor}"


async def _measure_rendered_video(
    result: dict[str, Any],
    *,
    resolution_key: str,
    requested_resolution: str | None,
    remote_app_ctx: "AppContext | None" = None,
) -> _RenderedFacts:
    """Replace echoes of the request with measurements of the artifact.

    Both families bill by resolution, and both used to report the resolution
    the caller ASKED for — a figure indistinguishable from a report of what
    rendered. Omni was given a probe; Veo was not, so generate_video billed at
    a resolution its response never named. Keeping one copy is the point:
    the divergence is what produced the gap.

    Mutates `result` with the measured dimensions, the classified resolution
    and a warning when the render disagrees with the request. Returns the
    facts the caller needs for its own cost line.

    With ``remote_app_ctx``, a render delivered to Cloud Storage is
    downloaded once (up to the fetch cap) and measured the same way. Left
    unmeasured, a gs:// edit reported the 15s bound as its length and its
    sidecar recorded it, while every edit measured locally kept its source's
    3.03s.
    """
    duration_source: str | None = None
    resolution_source: str | None = None
    raw = result.get("duration_seconds")
    duration = float(raw) if isinstance(raw, (int, float)) else None

    video_url = result.get("video_url") or ""
    measured: float | None
    size: tuple[int, int] | None
    if (
        remote_app_ctx is not None
        and isinstance(video_url, str)
        and video_url.startswith("gs://")
    ):
        try:
            data = await fetch(
                video_url,
                allowed_dir=remote_app_ctx.data_folder,
                allowed_gcs_buckets=remote_app_ctx.allowed_gcs_buckets,
            )
        except Exception:  # pragma: no cover - fetch() already returns None
            logger.debug(
                "Could not download %s to measure it", video_url, exc_info=True
            )
            data = None
        if data is None:
            # Over the fetch cap, or unreadable here: the caller labels what
            # it reports instead.
            return _RenderedFacts(duration, None, None)
        metadata = await _probe_media(measure_video_metadata_bytes, data)
        measured, size = metadata if metadata else (None, None)
    elif not isinstance(video_url, str) or not video_url.startswith("file://"):
        # Any other remote render (an https URL, or a gs:// one when the caller
        # passes no app context, as the Veo tools do) is not opened here: a
        # probe would have to download it, and silence is why an unmeasured
        # number needs a stated source.
        return _RenderedFacts(duration, None, None)
    else:
        rendered_path = Path(video_url[7:])
        measured = await _probe_media(measure_video_duration, rendered_path)
        size = await _probe_media(measure_video_dimensions, rendered_path)
    if measured is not None:
        duration = measured
        duration_source = "measured from the rendered video"
    if size is not None:
        result["rendered_dimensions"] = list(size)
        if result.get("aspect_ratio") is None:
            # A continuation sends no ratio, so the impl reports none; the
            # file's own dimensions settle it (segments read null beside a
            # measured 640x360).
            result["aspect_ratio"] = _aspect_ratio_of(size)
        classified = classify_video_resolution(size)
        if classified is not None:
            result[resolution_key] = classified
            resolution_source = "measured from the rendered video"
            if requested_resolution and requested_resolution != classified:
                result.setdefault("warnings", []).append(
                    f"Requested {requested_resolution} but the render measured "
                    f"{classified} ({size[0]}x{size[1]}). The cost below "
                    "prices what rendered, not what was asked for."
                )
    return _RenderedFacts(duration, duration_source, resolution_source)


def _stamp_provenance(
    payload: dict[str, Any],
    *,
    backend: str | None = None,
    duration_source: str | None = None,
    resolution_source: str | None = None,
) -> dict[str, Any]:
    """Fill in the provenance fields a response is required to carry.

    Three fields are meant to be uniform across every media response —
    `backend`, `duration_source`, `resolution_source` — because a number
    without a source cannot be told apart from a measured one, and the backend
    is the question a whole debugging session went to answer. They were being
    added tool by tool, which is how `backend` ended up on exactly one of
    eight. Setting them in one place means a new tool gets the contract for
    free instead of having to remember it.

    Existing values always win: a caller that has MEASURED something has said
    something stronger than this default can.
    """
    if backend is not None:
        payload.setdefault("backend", backend)
    if duration_source is not None and "duration_seconds" in payload:
        payload.setdefault("duration_source", duration_source)
    if resolution_source is not None and "resolution" in payload:
        payload.setdefault("resolution_source", resolution_source)
    return payload


# What a dry run's numbers are, in one phrase: nothing has rendered, so every
# figure in the payload is a projection of the call that would be made. These
# are the weakest true statement about a quoted number; any tool that can say
# something more specific sets its own and wins, because _stamp_provenance
# never overwrites.
QUOTE_DURATION_PROVENANCE = (
    "the request, snapped to a length the model accepts: a dry run renders "
    "nothing to measure"
)
QUOTE_RESOLUTION_PROVENANCE = (
    "the request: a dry run renders nothing whose resolution could be measured"
)


def _backend_of(app_ctx: AppContext, model: str | None) -> str:
    """Which backend a model would run on, without building a client.

    Covers both families, because `backend` is meant to be on every media
    response and a caller should not have to know which family a model belongs
    to in order to read it. Never raises: a model this cannot place is
    reported as the primary client's own mode rather than failing a response
    that is otherwise complete.
    """
    name = str(model or "")
    if is_omni_model(name):
        pin = app_ctx.omni_backend_default
        return _omni_backend_choice(app_ctx, requested=None if pin == "auto" else pin)
    try:
        # Defer to the real routing rule rather than restating it here; a
        # second copy of "which client serves Veo" is a copy that drifts.
        client = _client_for_video_model(app_ctx, name)
    except RuntimeError:
        # No client serves this model on this deployment. The call itself
        # refuses with the actionable message; report where it would have run.
        client = app_ctx.client
    return "vertex" if getattr(client._api_client, "vertexai", False) else "gemini_api"


def _respond(app_ctx: AppContext, payload: dict[str, Any]) -> str:
    """Serialize a tool response, stamping the provenance it must carry.

    `backend` belongs on every media response — it is the question a whole
    debugging session went to answer — and adding it tool by tool is how it
    ended up on one of eight. Routed through here it cannot be forgotten.
    """
    quote = bool(payload.get("dry_run"))
    # One response, one model name. A payload can carry a spelling the price
    # book folds onto another ID (a Veo `-preview` id a caller fed back), and
    # reporting that beside a cost line priced on the canonical name had two
    # fields naming different models. The quote was fixed for this once and
    # the rendered path grew it back, so it is settled here instead of per
    # tool. The reported spelling is kept, not dropped, as served_model.
    raw_model = payload.get("model")
    if isinstance(raw_model, str) and raw_model:
        from .pricing import resolve_model_id

        canonical = resolve_model_id(raw_model)
        if canonical != raw_model:
            payload["model"] = canonical
            payload.setdefault("served_model", raw_model)
    # "We are on Vertex" is not a whole-server statement, and reading it as one
    # cost several rounds of confusion: omni prefers the Gemini API whenever a
    # key is configured, while Veo stays on the primary client. Say so on the
    # response where the two disagree, rather than leaving a reader to infer a
    # bug.
    model_name = str(payload.get("model") or "")
    if is_omni_model(model_name) and "backend_note" not in payload:
        primary = (
            "vertex"
            if getattr(app_ctx.client._api_client, "vertexai", False)
            else "gemini_api"
        )
        # The backend the CALL decided, when the tool recorded one; the
        # server-wide answer otherwise. A per-call omni_backend must not be
        # described by a note computed from the default.
        served = payload.get("backend") or _backend_of(app_ctx, model_name)
        reason = payload.get("backend_reason")
        # The note explains one decision: the auto default taking omni off
        # a Vertex server because a key is set. It was stamped on any split,
        # so omni_backend="gemini_api" passed explicitly was attributed to
        # the auto default, and omni_backend="vertex" on a Gemini API server
        # was reported as running on the Gemini API.
        auto_split = (
            served == "gemini_api"
            and primary == "vertex"
            and (
                reason == _OMNI_AUTO_KEY_REASON
                if reason
                else app_ctx.omni_backend_default == "auto"
            )
        )
        if auto_split:
            # Emitted only when the backends disagree -- a call that runs on
            # the primary needs no explanation, and a note on every response
            # is a note nobody reads.
            #
            # It says what the split COSTS, not whether the split is fine: the
            # earlier wording ended "deliberate, not a misconfiguration",
            # which presented a choice the caller had no control over as
            # settled, and said nothing about the project the traffic had
            # left.
            payload["backend_note"] = (
                f"Omni {'would run' if quote else 'ran'} on the Gemini Developer "
                f"API even though this server's primary client is {primary}: "
                "with a GEMINI_API_KEY set and "
                "OMNI_BACKEND=auto, omni defaults there, which means omni "
                "traffic leaves the Vertex project's billing, quota and "
                "credentials. Veo tools in "
                f"the same session report {primary}. To keep omni on Vertex AI, "
                'pass omni_backend="vertex" or set OMNI_BACKEND=vertex '
                "server-wide. See backend_reason for this call's decision."
            )
    return json.dumps(
        _stamp_provenance(
            payload,
            backend=_backend_of(app_ctx, payload.get("model")),
            # A quoted number needs a source as much as a billed one does —
            # more so, since it is the figure a caller decides on. Defaulting
            # here is what stops the next tool from shipping a bare integer.
            duration_source=QUOTE_DURATION_PROVENANCE if quote else None,
            resolution_source=QUOTE_RESOLUTION_PROVENANCE if quote else None,
        ),
        indent=2,
    )


OMNI_BACKENDS = ("auto", "vertex", "gemini_api")


def _parse_omni_backend_env(
    *, primary_is_vertex: bool, gemini_api_available: bool
) -> str:
    """Read and validate OMNI_BACKEND at startup.

    A pin the deployment cannot honour fails here, once, with the variable
    named -- not on the first omni call, and not by quietly serving the other
    backend.
    """
    raw = os.environ.get("OMNI_BACKEND", "auto").strip().lower() or "auto"
    if raw not in OMNI_BACKENDS:
        raise ValueError(
            f"OMNI_BACKEND={raw!r} is not one of {', '.join(OMNI_BACKENDS)}."
        )
    if raw == "vertex" and not primary_is_vertex:
        raise ValueError(
            "OMNI_BACKEND=vertex, but this server's primary client is not "
            "Vertex AI (GOOGLE_GENAI_USE_VERTEXAI is not true)."
        )
    if raw == "gemini_api" and primary_is_vertex and not gemini_api_available:
        raise ValueError(
            "OMNI_BACKEND=gemini_api, but this Vertex server has no "
            "GEMINI_API_KEY to reach the Gemini Developer API with."
        )
    return raw


# The one decision backend_note explains: the auto default sending omni to the
# Gemini Developer API from a Vertex server because a key is set.
_OMNI_AUTO_KEY_REASON = (
    "GEMINI_API_KEY is set, so omni uses the Gemini Developer API "
    "rather than this server's Vertex project; pass "
    'omni_backend="vertex", or set OMNI_BACKEND=vertex server-wide, '
    "to keep omni traffic on Vertex AI"
)


def _omni_backend_decision(
    app_ctx: AppContext,
    *,
    need_gcs: bool = False,
    prefer_backend: str | None = None,
    requested: str | None = None,
) -> tuple[str, str]:
    """Which backend an omni call will use, and WHY: ``(backend, reason)``.

    The DECISION, split out from the construction it used to be entangled
    with. Pre-flights need the answer to apply the right backend's documented
    limits, and asking for it by building a client meant a yes/no question
    could construct a Vertex client on the event loop — credential resolution
    and all, which can reach for the metadata server, and which is not
    memoized when it fails, so every subsequent call retries and blocks again.
    A blocked event loop hangs every request in the process, not just this one.

    The reason is part of the answer because the default is not obvious from
    the outside. On a Vertex-primary server that also holds a GEMINI_API_KEY,
    omni went to the Gemini Developer API on every call, with nothing an
    operator or caller could set to keep it in the Vertex project, and the
    response's ``backend``
    field said where the call went but not that it could have gone elsewhere.
    For an enterprise deployment that is a different billing path, quota pool
    and credential model, chosen silently. ``requested`` is the explicit
    choice (a tool's ``omni_backend`` or the OMNI_BACKEND default) and is
    HARD: an unsatisfiable request raises rather than falling through.
    ``prefer_backend`` is the SOFT continuity hint from a prior interaction's
    recorded backend, and outranks ``need_gcs`` because an interaction lives
    on one backend: an id minted by the Developer API does not resolve on
    Vertex, whereas an unhonoured output_gcs_uri is dropped with a warning and
    the render still lands.
    """
    primary_is_vertex = bool(getattr(app_ctx.client._api_client, "vertexai", False))
    has_gemini = app_ctx.gemini_api_client is not None
    if requested == "vertex":
        if not primary_is_vertex:
            raise ValueError(
                "omni_backend=vertex was requested, but this server's primary "
                "client is not Vertex AI (GOOGLE_GENAI_USE_VERTEXAI is not true)."
            )
        return "vertex", "omni_backend=vertex was requested"
    if requested == "gemini_api":
        if primary_is_vertex and not has_gemini:
            raise ValueError(
                "omni_backend=gemini_api was requested, but this Vertex server "
                "has no GEMINI_API_KEY to reach the Gemini Developer API with."
            )
        return "gemini_api", "omni_backend=gemini_api was requested"
    if prefer_backend == "vertex" and primary_is_vertex:
        return "vertex", "continuing an interaction minted on Vertex AI"
    if prefer_backend == "gemini_api" and (has_gemini or not primary_is_vertex):
        return (
            "gemini_api",
            "continuing an interaction minted on the Gemini Developer API",
        )
    if need_gcs and primary_is_vertex:
        return "vertex", "output_gcs_uri was given, and GCS delivery is Vertex-only"
    if has_gemini:
        if primary_is_vertex:
            return "gemini_api", _OMNI_AUTO_KEY_REASON
        return "gemini_api", "this server's primary client is the Gemini Developer API"
    if primary_is_vertex:
        return "vertex", "no GEMINI_API_KEY is set, and the primary client is Vertex AI"
    return "gemini_api", "this server's primary client is the Gemini Developer API"


def _omni_backend_choice(
    app_ctx: AppContext,
    *,
    need_gcs: bool = False,
    prefer_backend: str | None = None,
    requested: str | None = None,
) -> str:
    """The backend half of _omni_backend_decision."""
    return _omni_backend_decision(
        app_ctx, need_gcs=need_gcs, prefer_backend=prefer_backend, requested=requested
    )[0]


async def _resolve_omni_backend(
    app_ctx: AppContext,
    omni_backend: str | None,
    *,
    need_gcs: bool,
    previous_interaction_id: str | None,
    task: str | None = None,
    prior: "PriorInteraction | None" = None,
) -> tuple[str, str]:
    """A tool's omni_backend argument, resolved and validated: ``(backend, reason)``.

    ``None`` means the server default (OMNI_BACKEND); ``"auto"`` means the
    documented preference order. A request that contradicts the backend a
    prior interaction was minted on is refused here, before anything is
    billed, because that continuation cannot succeed.

    An interaction recorded on a retired model is refused first, with
    _retired_interaction_refusal's advice (``task`` picks the tool it names).
    No backend can continue it, so the backend conflict was the less useful
    of two true refusals -- and when it came from the server's OMNI_BACKEND
    default, it told a caller who had passed no omni_backend to drop it,
    and never named the saved video that is the way forward.
    """
    if omni_backend is not None and omni_backend not in OMNI_BACKENDS:
        raise ValueError(
            f"omni_backend={omni_backend!r} is not one of {', '.join(OMNI_BACKENDS)}."
        )
    effective = omni_backend or app_ctx.omni_backend_default
    requested = None if effective == "auto" else effective
    if prior is None:
        prior = await _prior_interaction_or_empty(
            app_ctx.videos_dir, previous_interaction_id
        )
    retired = _retired_interaction_refusal(prior, previous_interaction_id, task)
    if retired:
        raise ValueError(retired)
    if requested is not None and prior.backend and prior.backend != requested:
        minted_on = (
            "Vertex AI" if prior.backend == "vertex" else "the Gemini Developer API"
        )
        # The caller can only drop an argument they passed, and dropping it
        # helps only when the server's default does not conflict as well;
        # otherwise the way through is to name the interaction's backend.
        default = app_ctx.omni_backend_default
        if omni_backend is None:
            fix = (
                f"This server's OMNI_BACKEND default is {requested}: pass "
                f'omni_backend="{prior.backend}" to continue it on {minted_on}, '
                "or start a new generation."
            )
        elif default in ("auto", prior.backend):
            fix = (
                f'Drop omni_backend (or pass omni_backend="{prior.backend}") '
                f"to continue it on {minted_on}, or start a new generation."
            )
        else:
            fix = (
                f'Pass omni_backend="{prior.backend}" to continue it on '
                f"{minted_on}: dropping omni_backend falls back to this "
                f"server's OMNI_BACKEND default, {default}, which cannot "
                "continue it either. Or start a new generation."
            )
        raise ValueError(
            f"previous_interaction_id was minted on {minted_on}, and an interaction "
            f"lives on one backend, so omni_backend={requested} cannot continue it. "
            f"{fix}"
        )
    backend, reason = _omni_backend_decision(
        app_ctx, need_gcs=need_gcs, prefer_backend=prior.backend, requested=requested
    )
    if omni_backend is None and requested is not None:
        reason = f"{reason} (OMNI_BACKEND={requested} is this server's default)"
    return backend, reason


def _omni_server_backend(app_ctx: AppContext) -> str | None:
    """OMNI_BACKEND as a hard request, or None when it is "auto".

    For the omni renders that take no omni_backend argument of their own
    (generate_video's draft, generate_clip's animatic): they asked for a
    backend with no request at all, so OMNI_BACKEND=vertex was ignored there
    and a draft on a Vertex-pinned server ran on the Gemini API, under a note
    telling the operator to set the variable they had set.
    """
    default = app_ctx.omni_backend_default
    return None if default == "auto" else default


def _omni_backend_is_vertex(
    app_ctx: AppContext,
    need_gcs: bool = False,
    prefer_backend: str | None = None,
    requested: str | None = None,
) -> bool:
    """Whether the call this pre-flight is checking will run on Vertex AI.

    Reads the same decision the client picker reads, without building
    anything — see _omni_backend_choice.
    """
    return (
        _omni_backend_choice(
            app_ctx,
            need_gcs=need_gcs,
            prefer_backend=prefer_backend,
            requested=requested,
        )
        == "vertex"
    )


def _client_for_omni(
    app_ctx: AppContext,
    *,
    need_gcs: bool = False,
    prefer_backend: str | None = None,
    requested: str | None = None,
) -> genai.Client:
    """Pick the genai.Client for an omni call (Interactions API).

    Omni + the Interactions API are documented on BOTH backends: the Gemini
    Developer API and Vertex AI / Gemini Enterprise Agent Platform, whose
    Interactions API reference calls gemini-omni-1.2-flash generally
    available.

    GCS output delivery only works on Vertex, so when the caller needs it
    (`need_gcs`) and the primary client is Vertex-capable, prefer the
    global-location Vertex client even if a Gemini API key is also configured
    — otherwise the explicit output_gcs_uri would be silently dropped. When
    GCS is not needed, prefer the dedicated Gemini API client. Falls back to a global-location Vertex client
    for a Vertex primary (omni's interactions collection is location `global`),
    or the primary client as-is on the Gemini API.

    ``prefer_backend`` overrides all of that, and outranks ``need_gcs``,
    because an interaction lives on ONE backend: an id minted by the Gemini
    Developer API does not resolve on Vertex AI. A continuation sent to the
    other backend fails outright, whereas an unhonored output_gcs_uri is
    dropped with a warning and the render still lands — so when the two
    conflict, the one that can still succeed wins. It is honored only where
    the backend is actually reachable; otherwise the usual choice applies and
    the call is left to fail with the service's own error rather than one this
    function invents.
    """
    if _omni_backend_is_vertex(
        app_ctx, need_gcs=need_gcs, prefer_backend=prefer_backend, requested=requested
    ):
        return _get_omni_vertex_global_client()
    if app_ctx.gemini_api_client is not None:
        return app_ctx.gemini_api_client
    return app_ctx.client


async def _omni_generate_and_manifest(
    app_ctx: AppContext,
    ctx: Context[ServerSession, AppContext],
    *,
    prompt: str,
    model: str = DEFAULT_OMNI_MODEL,
    image_bytes_list: list[bytes] | None = None,
    first_frame_bytes: bytes | None = None,
    last_frame_bytes: bytes | None = None,
    reference_image_bytes_list: list[bytes] | None = None,
    input_video_bytes: bytes | None = None,
    input_video_uri: str | None = None,
    upscale_source_resolution: str | None = None,
    source_duration_seconds: float | None = None,
    source_duration_measured: bool = False,
    resolution_origin: str | None = None,
    reference_video_bytes_list: list[bytes] | None = None,
    previous_interaction_id: str | None = None,
    task: str | None = None,
    aspect_ratio: str = "16:9",
    duration_seconds: float | None = 6.0,
    resolution: str | None = None,
    output_gcs_uri: str | None = None,
    prefer_backend: str | None = None,
    prior: "PriorInteraction | None" = None,
    backend_reason: str | None = None,
    timeout_seconds: int = OMNI_DEFAULT_TIMEOUT_SECONDS,
    manifest_extra: dict[str, Any] | None = None,
    submission: OmniSubmission | None = None,
) -> dict[str, Any]:
    """Run the omni impl and attach a sidecar manifest, returning the result.

    Shared by generate_video_omni, edit_video, extend_video_omni,
    upscale_video_omni and the generate_video draft path so they all produce
    consistent output.

    It does not judge a large inline output on Vertex AI: every caller runs
    omni_inline_delivery_policy in its own pre-flight, where its dry run runs
    it too, and carries the warning itself. Judging it again here, per render,
    would contradict a chain's own warning on every turn but its last.
    """
    # A continuation has to reach the backend that holds its context, so an
    # interaction id resolves to a backend before anything else is decided.
    # Without this the client was chosen from the CURRENT call's needs alone,
    # and a chain whose last turn wanted GCS output moved to Vertex for that
    # turn, carrying a previous_interaction_id the Gemini Developer API had
    # minted and Vertex cannot resolve — every earlier turn billed, the last
    # one dead. Callers that already know the backend (a chain pins it once)
    # pass it in rather than re-scanning per turn. Likewise the record itself:
    # each tool has already read it for the same interaction.
    if prior is None:
        prior = await _prior_interaction_or_empty(
            app_ctx.videos_dir, previous_interaction_id
        )
    spec = omni_spec(model)
    _validate_omni_continuation_source(spec, prior, previous_interaction_id, task)
    if prefer_backend is None:
        prefer_backend = prior.backend
    # Every omni tool resolves its backend and passes it in; only a render
    # with no omni_backend argument (a draft) arrives with none, and then the
    # server's OMNI_BACKEND is the request.
    requested = _omni_server_backend(app_ctx) if prefer_backend is None else None
    # Route to the Vertex client when GCS output is requested (delivery only
    # works there), otherwise prefer the Gemini API client.
    client = _client_for_omni(
        app_ctx,
        need_gcs=bool(output_gcs_uri),
        prefer_backend=prefer_backend,
        requested=requested,
    )
    client_is_vertex = bool(getattr(client._api_client, "vertexai", False))
    if backend_reason is None:
        backend_reason = _omni_backend_decision(
            app_ctx,
            need_gcs=bool(output_gcs_uri),
            prefer_backend=prefer_backend,
            requested=requested,
        )[1]

    # GCS delivery only works on Vertex; on the Gemini API omni returns bytes
    # inline. Drop an explicit output_gcs_uri on a non-Vertex omni client with
    # a warning rather than sending an unsupported field.
    effective_gcs = output_gcs_uri
    gcs_warning: str | None = None
    if output_gcs_uri and not client_is_vertex:
        effective_gcs = None
        gcs_warning = (
            "output_gcs_uri is ignored: omni on the Gemini API returns video "
            "inline (GCS delivery requires Vertex AI mode)."
        )

    # Tracked even when the caller passes no submission of its own, so a
    # render this call stops waiting for is recorded below.
    tracked = submission if submission is not None else OmniSubmission()
    try:
        result = await generate_video_omni_impl(
            client=client,
            prompt=prompt,
            videos_dir=app_ctx.videos_dir,
            model=model,
            image_bytes_list=image_bytes_list or None,
            first_frame_bytes=first_frame_bytes,
            last_frame_bytes=last_frame_bytes,
            reference_image_bytes_list=reference_image_bytes_list or None,
            input_video_bytes=input_video_bytes,
            input_video_uri=input_video_uri,
            reference_video_bytes_list=reference_video_bytes_list or None,
            previous_interaction_id=previous_interaction_id,
            task=task,
            aspect_ratio=aspect_ratio,
            duration_seconds=duration_seconds,
            resolution=resolution,
            output_gcs_uri=effective_gcs,
            # A bare delivery='uri' is a Gemini-Developer-API shape: Vertex
            # requires a gcs_uri alongside it, which is the effective_gcs
            # branch. Only this layer knows which backend the call landed on.
            allow_uri_delivery=not client_is_vertex,
            timeout_seconds=timeout_seconds,
            log_callback=ctx.info,
            submission=tracked,
            # Only to size an edit's or upscale's output if no video comes back.
            source_duration_seconds=source_duration_seconds or prior.duration_seconds,
        )
    except Exception:
        if tracked.interaction_id:
            # The interaction exists, renders and bills whatever happened
            # here. Recording it as pending tells get_omni_interaction which
            # backend holds it and what it was, so it can be collected and
            # priced; and tells a continuation of the id where to go.
            _write_remote_interaction_record(
                app_ctx.videos_dir,
                {
                    "interaction_id": tracked.interaction_id,
                    "status": "pending",
                    "backend": "vertex" if client_is_vertex else "gemini_api",
                    "model": spec.model,
                    "task": task,
                    "upscale_source_resolution": upscale_source_resolution,
                },
            )
        raise
    # A Gemini API render delivered as a Google-hosted URI carries its own
    # note from the impl, which is where that choice is made and the only
    # place that knows the output length it was judged by: re-deciding it
    # here from duration_seconds, which an edit never sends, disagreed with
    # the request. (Vertex's large-inline case is omni_inline_delivery_policy's,
    # decided before the render.)
    if gcs_warning:
        result.setdefault("warnings", []).append(gcs_warning)
    # Which backend minted this interaction_id. Read by the caller chaining
    # onto it, and written into the manifest below. Returned on the result too
    # because a sidecar is not always written (a remote render, an unwritable
    # media dir) and a chain must not fall back to re-deciding the backend
    # from the next turn's needs.
    result["backend"] = "vertex" if client_is_vertex else "gemini_api"
    # Not just where the call went, but why -- and how to change it.
    result["backend_reason"] = backend_reason
    # Cost from the duration the interaction actually rendered (clamped by
    # the impl), not the request — covers omni, edit_video and draft mode.
    # Prefer the rendered artifact over any inference. Everything else here is
    # either the caller's request or a figure the server wrote down earlier,
    # and an edit seeds the next edit's estimate — so a wrong assumption
    # propagates down the whole chain. Measuring settles what actually
    # rendered; an unmeasurable edit bills the service maximum as a labelled
    # upper bound, never a guessed figure presented as fact.
    billed_model = result.get("model") or model
    # The resolution needs the same treatment the duration got, and for a
    # sharper reason: `rendered_resolution` was an echo of the REQUEST,
    # indistinguishable from a report of what rendered, while omni's
    # per-second rate differs threefold across its tiers. One extra probe on a
    # file already open turns the billing basis into evidence.
    facts = await _measure_rendered_video(
        result,
        resolution_key="rendered_resolution",
        # What was sent, not the impl's echo of the default: an edit that
        # sent none warned "Requested 720p but the render measured 360p".
        requested_resolution=resolution,
        remote_app_ctx=app_ctx,
    )
    effective_duration = facts.duration_seconds
    duration_source = facts.duration_source
    resolution_source = facts.resolution_source

    # An edit is any turn that continues existing footage — a prior
    # interaction OR an input video — mirroring omni's own _select_task_type.
    # Both send no duration on the wire and come back with duration_seconds
    # None, so keying the upper-bound rule on previous_interaction_id alone
    # missed the input-video edit: it fell through to the fresh-render branch,
    # which billed the raw request (3.3x low) and labelled it "the length
    # asked for" — the falsified inherit model, resurrected on the second
    # edit form.
    # An extension is a continuation like an edit: neither sends a duration,
    # and the service picks the rendered length.
    is_edit = (
        bool(previous_interaction_id)
        or input_video_bytes is not None
        or bool(input_video_uri)
        or task == "extend"
    )
    billed_upper_bound = False
    upscale_source_measured = False
    if effective_duration is None and task == "upscale":
        # The longest source upscaling takes, read off the spec as the tool's
        # own pre-flight reads it, so the bill and the quote use one bound.
        longest_source = omni_source_limit_seconds(spec, task="upscale")
        effective_duration = (
            source_duration_seconds or prior.duration_seconds or longest_source
        )
        billed_upper_bound = not (source_duration_seconds or prior.duration_seconds)
        upscale_source_measured = (
            source_duration_measured
            if source_duration_seconds
            else bool(prior.duration_seconds) and prior.duration_measured
        )
        duration_source = (
            "upper bound: source and output could not be measured "
            f"({longest_source:g}s maximum)"
            if billed_upper_bound
            else "source metadata: upscaling preserves duration"
        )
    elif effective_duration is None and is_edit:
        # A continuation whose render cannot be measured (e.g. gs://
        # delivery). Two measurements put an EDIT at Omni's per-render maximum
        # regardless of the source (3.00s and 3.01s sources both rendered
        # 10.01s), so billing the source's length would under-bill ~3.3x. An
        # EXTENSION is the other way round: it returns the whole growing clip,
        # so that same per-render maximum under-bills a chain by up to 3x —
        # turn four of a chain renders 40s, not 10s. One rule, keyed on which
        # kind of continuation this was and on the source when it is known.
        effective_duration = omni_continuation_upper_bound(
            omni_spec(billed_model),
            result.get("task"),
            source_duration_seconds or prior.duration_seconds,
            # The backend this render actually ran on, whose extension
            # ceiling and retained window are its own (see omni_spec). The
            # parameter was once added and then passed by no caller, so a
            # Vertex render was billed at the Developer API's bound.
            vertexai=client_is_vertex,
            per_turn_seconds=duration_seconds if task == "extend" else None,
        )
        billed_upper_bound = True
        duration_source = (
            "upper bound: the render could not be measured, and a "
            "continuation's length is chosen by the service, so this bills "
            f"the longest clip it could have produced ({effective_duration:g}s) "
            "rather than a figure inherited from the request"
        )
    elif duration_source is None:
        # Delivered somewhere this process could not read (a gs:// render over
        # the fetch cap or outside its credentials, or another remote URI), so
        # the render was never inspected. A FRESH omni request is honoured at its
        # clamped length, so that figure is the right basis — but it is what
        # was asked for, not what came back, and omni renders land marginally
        # over it. Price it as an estimate carrying the encoder allowance,
        # and say which it is rather than leaving the number unattributed.
        duration_source = (
            "the clamped request: the rendered file could not be opened to "
            "measure it (delivered to a remote URI), so this is the length "
            "asked for, not the length measured"
        )

    if resolution_source is None:
        resolution_source = _omni_resolution_provenance(
            resolution_origin or ("request" if resolution else "default"),
            "the rendered file could not be opened to measure it",
        )
    result["resolution_source"] = resolution_source
    duration_is_measured = duration_source == "measured from the rendered video"
    if effective_duration is not None:
        if billed_upper_bound:
            # A bound prices the bill; it is not the clip's length. Reported
            # as duration_seconds, and so recorded in the sidecar, a later
            # upscale or extension of the take quoted and capped against 15s.
            result["duration_seconds"] = None
            result["duration_upper_bound_seconds"] = effective_duration
        else:
            result["duration_seconds"] = effective_duration
    if duration_source:
        result["duration_source"] = duration_source

    if task == "upscale" and upscale_source_resolution:
        # Upscaling has its own per-pair token rates, so the generation rate
        # _video_cost would apply is the wrong table entirely. Same two rules
        # as generation, though: a length measured from the rendered file is
        # metered (no encoder allowance, not an estimate), and the pair
        # priced is the one that rendered, not the one requested.
        cost = _omni_upscale_cost(
            # The upscale branch above always sets it, to the measured
            # output, the source's length, or the 15s bound.
            float(effective_duration or _omni_max_render_seconds(billed_model)),
            upscale_source_resolution,
            rendered=result.get("rendered_resolution"),
            requested=resolution,
            actual=duration_is_measured,
            # A source measured (or recorded as measured) is the output's
            # length too, since upscaling preserves it; no allowance.
            measured=duration_is_measured or upscale_source_measured,
        )
    else:
        cost = _video_cost(
            billed_model,
            effective_duration
            if effective_duration is not None
            else float(
                duration_seconds
                if duration_seconds is not None
                else _omni_max_render_seconds(billed_model)
            ),
            # What the render actually came back as, not what was asked: the
            # preview model renders 720p whatever it is sent, and 1.2 reports
            # the resolution it used. Quoting the request would price a 4K
            # render that the preview model never produced.
            resolution=str(result.get("rendered_resolution") or "720p"),
            include_audio=False,
            # Only a measured length is a metered cost; anything else is an
            # estimate and must say so rather than presenting a bound as fact.
            actual=duration_is_measured,
            presnapped=not duration_is_measured,
        )
    if cost:
        if not duration_is_measured:
            # Three different unmeasured cases, and saying "the service
            # maximum" for one that prices the request (or, for an upscale,
            # the source) would overstate the bill by 3x on the response an
            # operator reconciles against.
            cost = dict(cost)
            if billed_upper_bound:
                basis = (
                    "bills the service maximum as an upper bound, not the "
                    "metered length"
                )
            elif task == "upscale":
                basis = (
                    "prices the source's length, which upscaling preserves, "
                    "not a metered length"
                )
            else:
                basis = "prices the length requested, not a metered length"
            cost["detail"] = (
                f"{cost['detail']} — the render could not be measured, so this {basis}"
            )
        result["cost"] = cost
    manifest: dict[str, Any] = {
        "kind": "omni_video",
        "prompt": prompt,
        "model": result.get("model"),
        "task": result.get("task"),
        # Which backend minted this interaction_id. Read back by the next
        # continuation so a chain cannot drift onto a backend that has never
        # heard of it.
        "backend": "vertex" if client_is_vertex else "gemini_api",
        "backend_reason": backend_reason,
        # What rendered when measured, else what was sent (None on an edit).
        "aspect_ratio": result.get("aspect_ratio", aspect_ratio),
        "resolution": result.get("rendered_resolution"),
        "resolution_source": result.get("resolution_source"),
        "duration_seconds": result.get("duration_seconds", duration_seconds),
        "interaction_id": result.get("interaction_id"),
        "previous_interaction_id": previous_interaction_id,
        "output_gcs_uri": effective_gcs,
        "video_url": result.get("video_url"),
    }
    if duration_source:
        manifest["duration_source"] = duration_source
    if result.get("duration_upper_bound_seconds") is not None:
        manifest["duration_upper_bound_seconds"] = result[
            "duration_upper_bound_seconds"
        ]
    if result.get("effective_prompt"):
        # The role declarations the server generated changed what the model
        # was asked; the sidecar is what an operator reads back later, so the
        # text that actually rendered belongs in it.
        manifest["effective_prompt"] = result["effective_prompt"]
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
        # And keep it findable by id, or the next continuation cannot learn
        # which backend minted it.
        _write_remote_interaction_record(app_ctx.videos_dir, manifest)
    return result


def _omni_upscale_cost(
    seconds: float,
    source_resolution: str,
    *,
    rendered: str | None,
    requested: str | None,
    actual: bool,
    measured: bool | None = None,
) -> dict[str, Any] | None:
    """Cost of a standalone upscale, priced at the resolution pair that rendered.

    A render whose measured tier is not a published pair (it came back at
    the source's own tier, say) has no rate of its own, so it is priced at
    the requested pair and the detail says which was used. Never raises: the
    render has already been billed, and a pricing gap must not turn its
    result into an error.
    """
    from .pricing import estimate_omni_upscale_cost

    for target in dict.fromkeys(t for t in (rendered, requested) if t):
        try:
            estimate = estimate_omni_upscale_cost(
                seconds, source_resolution, target, actual=actual, measured=measured
            )
        except ValueError:
            continue
        cost = _cost_payload(estimate)
        if cost and rendered and target != rendered:
            cost = dict(cost)
            cost["detail"] = (
                f"{cost['detail']} — the render measured {rendered}, which has "
                f"no published upscale rate from {source_resolution}, so this "
                f"prices the {target} that was requested"
            )
        return cost
    return None


async def _emit_warnings(
    ctx: Context[ServerSession, AppContext], warnings: Any
) -> None:
    """Mirror body/manifest warnings onto the MCP notification channel.

    The video tools only ever recorded warnings in the JSON body or the
    sidecar manifest, so an agent watching the notification stream never saw
    "include_audio not honored", "add_bridges ignored in animatic mode", a
    GCS-allowlist warning, or a model reroute. generate_image already mirrors
    its warnings via ctx.warning; this brings the video tools in line.
    Dedupes so a distinct warning is emitted at most once per call.
    """
    seen: set[str] = set()
    for warning in warnings or []:
        if warning in seen:
            continue
        seen.add(warning)
        await ctx.warning(warning)


def _veo_request_mode(
    *,
    extend_video_uri: str | None,
    reference_image_uris: list[str] | None,
    has_first_frame: bool,
    has_last_frame: bool,
) -> str:
    """The Veo generation mode a request will run in, from the inputs given."""
    if extend_video_uri:
        return "extend_video"
    if reference_image_uris:
        return "reference_to_video"
    if has_first_frame and has_last_frame:
        return "first_last_frame"
    if has_first_frame:
        return "image_to_video"
    return "text_to_video"


def _veo_slow_tier_warning(
    resolution: str | None, timeout_seconds: float
) -> str | None:
    """Warn when the deadline cannot cover the tier being asked for.

    A 4-second 4K render MEASURED 335s end to end. The default deadline is
    210s, and the MCP host ceiling that default exists to stay under is about
    240s -- so a 4K render does not fit in one tool call on a typical host at
    any timeout the host will allow, and the failure arrives as a cancelled
    call AFTER the render has been billed. Said before the spend, not after.
    """
    if str(resolution or "").upper() != "4K":
        return None
    if timeout_seconds >= VEO_MEASURED_4K_SECONDS * 1.2:
        return None
    return (
        "4K is the slowest tier: a 4-second 4K render MEASURED "
        f"{VEO_MEASURED_4K_SECONDS:.0f}s end to end. This call's deadline is "
        f"{timeout_seconds:g}s, so it will very likely be cut short after the "
        "render has been billed. Raise timeout_seconds past "
        f"{VEO_MEASURED_4K_SECONDS * 1.2:.0f}s -- and note that many MCP hosts "
        "cap a single tool call near 240s, in which case no timeout_seconds "
        "value will help and 4K cannot be rendered through this tool on that "
        "host. Render at 1080p, or drive the tool from a client without that "
        "ceiling."
    )


def _veo_access_advice(exc: BaseException) -> str | None:
    """Turn a Vertex IAM refusal into the grant that fixes it, or None.

    A 403 from Veo arrives as a wall of SDK traceback ending in a
    troubleshooter URL, and says nothing about WHICH identity was refused --
    which is the one fact the operator needs, because the server may be
    running as a service account they did not expect. omni already does this
    for its own allowlist refusals (see omni._access_refusal); Veo did not,
    and a denial on the most expensive tier is the worst place to leave a raw
    error.

    Deliberately narrow: a 403 that is not an IAM permission denial, and any
    other status, are left exactly as the service reported them.
    """
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    text = str(exc)
    if code != 403 and "PERMISSION_DENIED" not in text:
        return None
    if "aiplatform" not in text:
        return None
    identity = ""
    try:
        found = credentials.service_account_credentials()
        email = getattr(found[0], "service_account_email", None) if found else None
        if email:
            identity = f" The server is authenticating as {email}."
    except Exception:  # noqa: BLE001 - advice must never mask the real error
        pass
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    grant = (
        f"gcloud projects add-iam-policy-binding {project or '<project>'} \\\n"
        f"    --member=serviceAccount:{'<service-account-email>' if not identity else identity.split()[-1].rstrip('.')} \\\n"
        "    --role=roles/aiplatform.user"
    )
    model_hint = ""
    match = re.search(r"models/([A-Za-z0-9._-]+)", text)
    if match:
        model_hint = f" for {match.group(1)}"
    return (
        f"Vertex AI refused this call{model_hint} with "
        f"aiplatform.endpoints.predict denied.{identity} That message covers two "
        "different problems, and they have different fixes.\n"
        "1. The identity lacks the permission. Then EVERY Vertex model fails "
        "the same way, and the grant is:\n"
        f"{grant}\n"
        "2. The identity has it, but this MODEL OR TIER is not enabled for the "
        "project. Then other models and other resolutions of the same model "
        "still render, and no IAM change helps -- Veo 4K in particular needs "
        "per-project access, separate from any role. Request it for the "
        "project, or render at a tier it already serves.\n"
        "Tell them apart by rendering the same model at 720p: if that works, "
        "it is (2). Check too that the Vertex AI API is enabled and that "
        "GOOGLE_CLOUD_LOCATION names a region serving the model. Nothing was "
        "rendered or billed."
    )


def _thought_signature_path(url: str) -> Path:
    """The file a thought_signature_url names, or raise for anything else."""
    if not url.startswith("file://"):
        raise ValueError(
            "thought_signature_url must be a file:// URL returned by a "
            f"previous generate_image call, got: {url}"
        )
    return Path(url[7:])


def _submission_note(
    state: str,
    *,
    handle: str | None,
    completed: bool = False,
    final_status: str | None = None,
) -> str:
    """What a failed render's caller needs to know about its bill.

    ``completed`` is for an interaction that finished and then failed to
    deliver (no video in its response, a download that broke): it has
    already rendered and billed, so "may still complete" would mislead a
    caller into waiting for a render that is over.
    """
    if state == "submitted" and final_status:
        # Ended without completing: "may still complete" would send a caller
        # to collect a render that will never arrive.
        return (
            f"{handle or 'The render'} ended with status '{final_status}' and "
            "will not complete; rendering again starts a new render."
        )
    if state == "submitted" and completed:
        return (
            f"{handle or 'The render'} completed, so it has rendered and been "
            "billed, but its video did not reach this server; rendering again "
            "bills again."
        )
    if state == "submitted":
        if handle and handle.startswith("Interaction "):
            return (
                f"{handle} was submitted and may still complete and bill; "
                "collect it with get_omni_interaction rather than rendering again."
            )
        return (
            f"{handle} was submitted and may still complete and bill; reconcile "
            "it before rendering again."
            if handle
            else "The render was submitted and may still complete and bill."
        )
    if state == "unknown":
        return (
            "The request may have reached Google before the deadline; if it "
            "did, it may still complete and bill."
        )
    return "The request had not been submitted, so nothing was billed."


def _veo_attempt_facts(
    *,
    model: str,
    draft: bool,
    draft_resolution: str | None,
    mode: str,
    duration_seconds: float,
    resolution: str | None,
    include_audio: bool,
) -> dict[str, Any]:
    """Name what a failed render was trying to do, and what it would have cost.

    Losing the error text on the most expensive tier is the worst place to
    lose it: a 4K render that outlived the host's ceiling came back as a bare
    "Tool execution failed" -- no mode, no cost, no operation -- and billed
    $1.20 anyway. Every failure on this path now names the mode, the model,
    the resolution and the attempted (quoted) cost, so the spend can be
    reconciled. Best effort, and never raises: this runs inside an error
    handler.
    """
    facts: dict[str, Any] = {
        "generation_mode": "draft (omni)" if draft else mode,
        "model": model,
        "resolution": resolution or "720p",
    }
    try:
        if draft:
            est_model, draft_res = _omni_preview_model(draft_resolution)
            facts["model"] = est_model
            facts["resolution"] = draft_res or "720p"
            facts["attempted_cost"] = _video_cost(
                est_model,
                float(duration_seconds),
                resolution=draft_res or "720p",
                include_audio=False,
            )
        else:
            # The modes with a forced length are priced at it; the others snap.
            forced = {"reference_to_video": 8.0, "extend_video": 7.0}.get(mode)
            facts["attempted_cost"] = _video_cost(
                model,
                forced if forced is not None else float(duration_seconds),
                resolution=resolution or "720p",
                include_audio=include_audio,
                presnapped=forced is not None,
            )
    except Exception:  # noqa: BLE001 - a failure to price must not mask the error
        facts["attempted_cost"] = None
    return facts


def _draft_ignored_veo_params(
    *,
    seed: int | None,
    negative_prompt: str | None,
    resolution: str | None,
    person_generation: str | None,
    last_frame: str | None,
    reference_image_uris: list[str] | None,
    extend_video_uri: str | None,
    output_gcs_uri: str | None,
    include_audio: bool,
) -> list[str]:
    """Veo-only params the omni draft model drops on a draft render.

    Shared by generate_video's dry_run and real draft paths so a quote
    discloses exactly the ignored paid parameters the real run reports —
    a pre-flight that hid them let a caller spend on controls omni never
    honored.
    """
    # `is not None` (not truthiness): seed=0 is a valid Veo seed and must be
    # reported as ignored too.
    ignored = [
        name
        for name, val in (
            ("seed", seed),
            ("negative_prompt", negative_prompt),
            ("resolution", resolution),
            ("person_generation", person_generation),
            ("last_frame_uri", last_frame),
            ("reference_image_uris", reference_image_uris),
            ("extend_video_uri", extend_video_uri),
            ("output_gcs_uri", output_gcs_uri),
        )
        if val is not None and val != []
    ]
    # include_audio is only notable when explicitly enabled (omni has no
    # audio control).
    if include_audio:
        ignored.append("include_audio")
    return ignored


def _draft_ignored_warning(ignored: list[str]) -> str:
    """The single warning line naming the Veo-only params a draft dropped."""
    return "draft mode (omni) ignored Veo-only params: " + ", ".join(ignored)


def _animatic_mode_warnings(
    *, add_bridges: bool, output_gcs_uri: str | None, include_audio_passed: bool
) -> list[str]:
    """Clip-level params the omni animatic model drops in animatic mode.

    Shared by generate_clip's dry_run and real run so a quote discloses the
    ignored inputs (output_gcs_uri, include_audio) the render silently drops
    — the pre-flight used to return an empty warnings list. add_bridges is
    honoured: an animatic's bridges render on Omni 1.2's first/last-frame
    interpolation. The parameter stays so every caller keeps one signature.
    """
    warnings: list[str] = []
    if output_gcs_uri:
        warnings.append(
            "output_gcs_uri is ignored in animatic mode; omni previews "
            "are always written locally."
        )
    if include_audio_passed:
        warnings.append(
            "include_audio is ignored in animatic mode "
            "(omni previews carry no controllable audio)."
        )
    return warnings


def _animatic_beat_dropped_warning(beat: dict[str, Any], idx: int) -> str | None:
    """Warning naming the Veo-only per-beat params animatic mode drops, if any.

    Shared by the beat loop and the dry_run disclosure so the quote lists the
    same ignored per-beat controls (seed, negative_prompt) the render reports.
    """
    dropped = [
        name for name in ("negative_prompt", "seed") if beat.get(name) is not None
    ]
    if not dropped:
        return None
    return (
        f"animatic mode ignored Veo-only beat params "
        f"({', '.join(dropped)}) on beat {idx}."
    )


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


# Cap on the images one omni request may carry. Every one is buffered whole
# (each up to MAX_FETCH_BYTES) and base64-inflated into the request body, and
# the reference's largest worked example uses six.
MAX_OMNI_INPUT_IMAGES = 8


def _omni_preview_model(resolution: str | None) -> tuple[str, str | None]:
    """The (model, resolution) a draft or animatic pass renders at.

    The model is always DEFAULT_OMNI_MODEL; only the resolution is a choice,
    validated against that model (None keeps its own default). It used to
    pick the model too -- naming a resolution selected the model that had
    one, leaving it unset took the preview -- until the preview's endpoint
    was switched off on OMNI_PREVIEW_SUNSET. The model stays in the return
    value so every draft and animatic path names what it bills from one
    place.

    360p is the point of the knob: it is a third of the 720p price, which is
    what turns a preview pass from "costs about what the delivery render
    costs" into a real saving on a 20-beat reel.
    """
    return DEFAULT_OMNI_MODEL, _validate_omni_resolution(
        _validate_omni_model(DEFAULT_OMNI_MODEL), resolution
    )


async def _omni_local_reference_video_warnings(
    ctx: Context[ServerSession, AppContext],
    spec: Any,
    uris: list[str],
) -> list[str]:
    """The reference-clip ceiling check, run at the PRE-FLIGHT on local files.

    The check itself only ran on the real render, so a 23.02s clip against a
    documented 3s ceiling drew a clean quote and said nothing — the caller
    learned about the violation after paying for it. "A dry run is offline" is
    no defence here: the clip is a local file, and loop_extend's own dry run
    measures exactly this way. Remote clips are still left to the real run,
    which is where they are fetched.
    """
    warnings: list[str] = []
    for index, uri in enumerate(uris):
        measured = await _probe_local_video_seconds(ctx, uri)
        if measured is not None and measured > (
            spec.max_reference_video_seconds + _encoder_allowance_seconds()
        ):
            warnings.append(_reference_video_over_ceiling(spec, index, measured))
    return warnings


def _reference_video_over_ceiling(spec: Any, index: int, measured: float) -> str:
    """One wording for the ceiling breach, whichever path found it.

    Shared so a quote and a render cannot describe the same violation
    differently — the divergence this codebase keeps producing.
    """
    return (
        f"reference_video_uris[{index}] is {measured:.2f}s, over the "
        f"documented {spec.max_reference_video_seconds:g}s ceiling for "
        "a video reference; the service may truncate it or refuse the "
        "request."
    )


async def _omni_reference_video_warnings(spec: Any, clips: list[bytes]) -> list[str]:
    """Warn about reference clips longer than the documented ceiling.

    "Video references support a maximum of 3 clips, up to 3 seconds each." The
    count is refused outright (a fourth clip has nowhere to go); an over-long
    clip is a warning instead, because the reference does not say whether the
    service truncates it or rejects the request, and guessing wrong would
    refuse a call that works.

    Each probe spawns an ffmpeg subprocess, so it goes off the loop like every
    other probe in this server. Run inline, three reference clips on a slow
    mount would block every other request in flight — the exact hazard
    _source_duration_or_none is built around.
    """
    warnings: list[str] = []
    for index, clip in enumerate(clips):
        measured = await _probe_media(measure_video_duration_bytes, clip)
        if measured is not None and measured > (
            spec.max_reference_video_seconds + _encoder_allowance_seconds()
        ):
            warnings.append(_reference_video_over_ceiling(spec, index, measured))
    return warnings


def _encoder_allowance_seconds() -> float:
    """One frame, for ceilings compared against a measured file (see #14)."""
    from .pricing import OMNI_ENCODER_ALLOWANCE_SECONDS

    return float(OMNI_ENCODER_ALLOWANCE_SECONDS)


def _check_omni_inline_size(
    spec: Any, data: bytes, *, vertexai: bool, inline: bool = True
) -> None:
    """Refuse a source too large for Vertex AI to take inline.

    ``inline=False`` is for bytes downloaded only to measure a source that is
    sent by gs:// URI (see _omni_gcs_source): Vertex's inline size cap does
    not apply to a request that carries no bytes.
    """
    if not (inline and vertexai and len(data) > OMNI_VERTEX_INLINE_VIDEO_BYTES):
        return
    # Only a model that reads a gs:// source by URI can be helped by "pass
    # its gs:// URI"; any other model fetches and inlines that URI too.
    if spec.reads_gcs_source_by_uri:
        remedy = "Save the video in Cloud Storage and pass its gs:// URI."
    else:
        remedy = (
            f"{spec.model} sends its source inline on Vertex AI, even one "
            "given as a gs:// URI: trim or re-encode it under 20 MB, or "
            f"use omni_model='{DEFAULT_OMNI_MODEL}', which reads a gs:// "
            "input_video_uri directly."
        )
    raise ValueError(f"Vertex inline video inputs must be at most 20 MB. {remedy}")


def _omni_source_length_refusal(
    spec: Any,
    measured: float | None,
    *,
    task: str | None = None,
    uploaded: bool = True,
) -> str | None:
    """Why a source is too long for this operation, or None.

    The one home of the source-length rule, for every path that knows a
    source's length: an uploaded clip measured from its bytes, a multi-turn
    clip's recorded length, and an upscale's source. The limit is
    omni_source_limit_seconds; the wording names the operation that is
    refused. Three copies of the 1.2 rule used to word it three ways, and the
    shared one told an upscale caller to "extend a clip this server generated
    instead", which an upscale cannot do.

    One frame of tolerance: this server's own renders measure a frame over
    their nominal length (pricing.OMNI_ENCODER_ALLOWANCE_SECONDS), so a
    strict comparison refused a clip this server had just produced.

    ``uploaded`` only changes the advice: the limit binds a source continued
    through previous_interaction_id exactly as it binds an upload.
    """
    limit = omni_source_limit_seconds(spec, task=task)
    if (
        not limit
        or measured is None
        or measured <= limit + _encoder_allowance_seconds()
    ):
        return None
    if task == "upscale":
        return (
            f"Standalone upscaling supports videos up to {limit:g} seconds, and "
            f"the source is {measured:.2f}s. Trim it to {limit:g}s or less and "
            "upscale that."
        )
    if not uploaded:
        return (
            f"{spec.model} edits source videos up to {limit:g} seconds, and "
            f"this interaction's clip is {measured:.2f}s. Trim the saved "
            f"video to {limit:g}s or less and edit that by passing it as "
            "input_video_uri."
        )
    # An extension has no source limit, so what is refused here is an edit.
    return (
        f"The source video is {measured:.2f}s. An uploaded video to edit must "
        f"be {limit:g}s or shorter on {spec.model}. Trim the source first."
    )


async def _check_omni_source_video(
    spec: Any,
    data: bytes,
    *,
    vertexai: bool = False,
    task: str | None = None,
    inline: bool = True,
) -> list[str]:
    """Refuse an uploaded source longer than omni will accept; else warn-free.

    The model's documented source ceiling (omni_source_limit_seconds) is a
    measurable property of the bytes in hand, so it is checked before
    anything is uploaded or billed rather than left to a 400 after the
    transfer. A probe that cannot read the clip returns None and the call
    proceeds — an unreadable container is not evidence of an over-long one.

    A model that documents no such ceiling gets no check. Reading the absent
    value as a limit of zero seconds refused EVERY input video on
    gemini-omni-flash-preview — which was the default model at the time, and
    whose input_video_uri path had always worked — with "must be 0s or
    shorter".

    ``vertexai`` decides only the inline size check: the length ceiling is
    the same on both backends.

    ``inline=False`` is for bytes downloaded only to measure a source that is
    sent by gs:// URI; see _check_omni_inline_size.

    Returns an empty list; it exists to raise, and returns a warning list so
    the call sites read like the other pre-flights.
    """
    _check_omni_inline_size(spec, data, vertexai=vertexai, inline=inline)
    if not omni_source_limit_seconds(spec, task=task):
        return []
    measured = await _probe_media(measure_video_duration_bytes, data)
    refusal = _omni_source_length_refusal(spec, measured, task=task)
    if refusal:
        raise ValueError(refusal)
    return []


def _omni_cumulative_cap_warning(
    spec: Any,
    source_seconds: float,
    times: int,
    planned_turns: int,
    per_turn_seconds: float | None = None,
) -> list[str]:
    """Warn when omni's cumulative ceiling cuts a chain of extensions short.

    The ceiling is the spec's max_extended_seconds: 30s on the Gemini API,
    40s on Vertex AI (which keeps the last 30s and appends to it).

    Two distinct things go wrong at the ceiling, and they need different
    warnings:

    * turns that cannot run at all, because the clip is already at the
      ceiling when they come up -- a count the caller did not choose;
    * a chain where every turn runs but the last cannot append a full step,
      so the finished clip is shorter than "source + turns x 10s" -- a length
      the caller did not choose.

    A warning rather than a refusal in both cases: the turns that fit still
    render, so there is no failure to report.

    This used to hedge ("later turns may be refused or truncated by the
    service"), because the quote clamped the chain at the ceiling while the
    render looped the full ``times`` regardless. That disagreement was the
    defect: a 10s source with ``times=4`` was quoted for 3 turns (~$9.14) and
    billed for 4 (~$13.19), and because ``max_cost_usd`` was projected off the
    clamped list, a $10 cap passed and then billed ~$14. Both paths stop at
    ``planned_turns`` now, so this states what happens instead of guessing.

    Args:
        planned_turns: Turns that fit under the ceiling -- what both the quote
            and the render will actually do.
    """
    ceiling = spec.max_extended_seconds
    if spec.extension_window_seconds:
        lengths = omni_extension_priced_lengths(
            spec, source_seconds, times, per_turn_seconds=per_turn_seconds
        )
        previous = source_seconds
        discarded = 0.0
        for output in lengths:
            discarded += max(0.0, previous - spec.extension_window_seconds)
            previous = output
        if discarded:
            return [
                f"Vertex retains the final {spec.extension_window_seconds}s of "
                f"the input on each turn. This chain discards {discarded:g}s "
                "from the starts of its inputs; all requested turns run."
            ]
        return []
    lengths = omni_extension_output_lengths(
        spec, source_seconds, times, per_turn_seconds
    )
    finish = lengths[-1] if lengths else source_seconds
    step = float(per_turn_seconds or spec.extension_step_seconds)
    if planned_turns < times:
        # Turns that cannot run at all: fewer than 3s, the shortest turn, are
        # left under the ceiling before the chain is done.
        dropped = times - planned_turns
        return [
            f"The source is {source_seconds:g}s, so only {planned_turns} of the "
            f"{times} requested turn(s) fit under {spec.model}'s documented "
            f"{ceiling}s ceiling for an extended video; the clip finishes at "
            f"{finish:g}s. The remaining {dropped} turn(s) would have less than "
            "3s, the shortest turn, to append, so they are not rendered and not "
            "billed."
        ]
    projected = round(source_seconds + step * times, 6)
    if projected > ceiling and lengths:
        # Every turn runs, but the last one cannot append a full step: it
        # asks for the whole seconds that fit, since a turn's duration is a
        # whole second. This said the clip "stops at 30s" while the chain
        # appended 6s to 29.04s.
        before = lengths[-2] if len(lengths) > 1 else source_seconds
        last = round(finish - before, 6)
        return [
            f"The source is {source_seconds:g}s and {times} turn(s) of "
            f"{step:g}s would reach {projected:g}s against {spec.model}'s "
            f"documented {ceiling}s ceiling, so the last turn asks for {last:g}s, "
            f"the whole seconds that fit, and the clip finishes at {finish:g}s."
        ]
    return []


def _omni_extension_quote(
    model: str, resolution: str, turn_lengths: list[float]
) -> dict[str, Any] | None:
    """Quote one omni extension turn per entry in ``turn_lengths``, or None.

    One estimate per TURN, then summed, for two separate reasons. Quoting the
    combined runtime as a single render loses a per-render encoder allowance
    for every turn but the first, which put this quote a hair under the
    planner's figure for the same plan. And the turns are not the same length:
    an extension returns the whole growing clip, so turn 2 renders (and bills)
    longer than turn 1 — see omni_extension_output_lengths.
    """
    try:
        from .pricing import sum_costs

        per_turn = [
            _video_cost_estimate(
                model,
                length,
                resolution=resolution,
                include_audio=False,
                presnapped=True,
            )
            for length in turn_lengths
        ]
        if not per_turn or any(estimate is None for estimate in per_turn):
            return None
        return _cost_payload(sum_costs(per_turn, label="extension"))
    except Exception:  # pragma: no cover - defensive
        logger.debug("Could not quote omni extension cost", exc_info=True)
        return None


def _omni_max_render_seconds(model: str) -> float:
    """The longest single render ``model`` produces, for an unmeasured fallback.

    Per model, not the flat OMNI_MAX_DURATION_SECONDS: that is 1.2's 15s, and
    pricing a preview render (10s maximum) at it over-states the fallback by
    50%.
    An unrecognised model falls back to the flat figure, the largest any
    omni model renders, so the bound stays an upper bound.
    """
    try:
        return float(omni_spec(model).max_duration_seconds)
    except ValueError:
        return float(OMNI_MAX_DURATION_SECONDS)


def _sum_omni_segment_costs(
    segments: list[dict[str, Any]], *, model: str, resolution: str
) -> dict[str, Any] | None:
    """Total the per-turn costs of a chained omni run, or None.

    Priced turn by turn rather than by summing the runtime: each turn is its
    own render with its own metered-or-bounded length, and a measured turn
    must not be re-priced as an estimate just because the turn beside it was
    unmeasurable. Mirrors the per-beat totalling in _assemble_clip_manifest.
    """
    try:
        from .pricing import sum_costs

        estimates = [
            _video_cost_estimate(
                str(seg.get("model") or model),
                float(
                    seg.get("duration_seconds")
                    or seg.get("duration_upper_bound_seconds")
                    or _omni_max_render_seconds(str(seg.get("model") or model))
                ),
                resolution=str(seg.get("rendered_resolution") or resolution),
                include_audio=False,
                actual=_segment_is_metered(seg),
                presnapped=not _segment_is_metered(seg),
            )
            for seg in segments
            if seg.get("video_url")
        ]
        if not estimates:
            return None
        return _cost_payload(sum_costs(estimates, label="extension"))
    except Exception:  # pragma: no cover - defensive
        logger.debug("Could not total omni extension cost", exc_info=True)
        return None


def _validate_omni_model(omni_model: str) -> Any:
    """Return the spec for ``omni_model``, or raise naming the alternatives.

    Front-loaded like every other video pre-flight: a typo'd model must fail
    before any fetch, and before a dry_run quotes a model that cannot run.
    """
    if not is_omni_model(omni_model):
        raise ValueError(
            f"Unsupported omni_model '{omni_model}'. "
            f"Supported values are {', '.join(OMNI_MODELS)}."
        )
    refusal = omni_model_refusal(omni_model)
    if refusal:
        # A dead endpoint must fail here, where it is free, not after a fetch.
        raise ValueError(refusal)
    return omni_spec(omni_model)


def _omni_requested_model(requested: str, spec: Any) -> dict[str, str]:
    """`requested_model` for a response, or nothing when no ID was folded."""
    return {"requested_model": str(requested)} if str(requested) != spec.model else {}


def _validate_omni_resolution(spec: Any, resolution: str | None) -> str | None:
    """Normalize a requested omni resolution, or raise.

    Refusing beats dropping: the preview model renders 720p whatever it is
    asked for, so silently accepting `resolution='4K'` there would quote and
    bill a render nobody can produce.
    """
    if resolution is None:
        return None
    if not spec.supports_resolution:
        raise ValueError(
            f"{spec.model} has no resolution parameter — it renders "
            f"{spec.rendered_resolution} only. Pass "
            f"omni_model='{DEFAULT_OMNI_MODEL}' to choose a resolution."
        )
    normalized = normalize_omni_resolution(resolution)
    if normalized is None or normalized not in spec.resolutions:
        raise ValueError(
            f"Unsupported resolution '{resolution}' for {spec.model}. "
            f"Supported values are {', '.join(spec.resolutions)}."
        )
    return normalized


def _omni_billed_resolution(spec: Any, resolution: str | None) -> str:
    """The resolution an omni quote should price, given what was asked.

    The preview model bills at what it renders (720p) no matter what the
    request said; 1.2 bills at the resolution it is given, defaulting to the
    documented 720p when none is.
    """
    if not spec.supports_resolution:
        return spec.rendered_resolution
    return resolution or spec.rendered_resolution


def _omni_sent_resolution(
    spec: Any,
    requested: str | None,
    prior: "PriorInteraction",
    previous_interaction_id: str | None,
) -> tuple[str | None, str]:
    """The resolution an omni render sends, and where it came from.

    The caller's, when given. Otherwise a continuation of a recorded
    interaction sends the resolution that clip rendered at, so an edit or an
    extension of a 360p draft is asked for, and priced at, 360p: sending none
    left the service to choose while the quote priced 720p, three times the
    360p rate. A fresh render, or a source with no record, sends none and
    gets the documented default.

    Returns:
        (resolution or None, "request" | "source" | "default").
    """
    if requested is not None:
        return requested, "request"
    if (
        previous_interaction_id
        and spec.supports_resolution
        and prior.resolution in spec.resolutions
    ):
        return prior.resolution, "source"
    return None, "default"


def _omni_resolution_provenance(origin: str, why: str) -> str:
    """resolution_source for a resolution that was not measured."""
    lead = {
        "request": "the request",
        "source": "the source's recorded resolution, kept because none was requested",
        "default": "the default, since none was requested",
    }.get(origin, "the request")
    return f"{lead}: {why}"


async def _probe_local_video_seconds(
    ctx: Context[ServerSession, AppContext],
    uri: str | None,
) -> float | None:
    """Duration of a LOCAL video for a pre-flight or a projection, or None.

    Path-based: ffprobe reads the container header, so there is no reason to
    load the whole file — up to the 50 MB fetch cap — into memory first, which
    is what routing this through the bytes fetch did. Containment is enforced
    the same way every other caller-supplied path is. Never raises: a source
    that cannot be probed falls back to a labelled floor, not a failure.
    """
    if not uri or not uri.startswith("file://"):
        return None
    try:
        app_ctx = ctx.request_context.lifespan_context
        path = _validate_local_path(Path(uri[7:]), app_ctx.data_folder)
    except Exception:
        logger.debug("Could not resolve %s for a probe", uri, exc_info=True)
        return None
    return await _probe_media(measure_video_duration, path)


async def _extension_source_seconds(
    ctx: Context[ServerSession, AppContext],
    uri: str | None,
    *,
    allow_remote: bool,
) -> float | None:
    """Length of an extension source, or None.

    Local always. Remote only on a REAL run: a dry run is documented free,
    instant and offline, so it does not download — but a real run is already
    committing to a render, and a capped read of the source is the difference
    between a labelled floor and a projection. On Vertex this is the common
    case, not an edge one: extension REQUIRES a GCS destination there, so the
    source is usually gs:// too, and quoting the appended 7s against a 4s
    source under-states the bill by 57%.

    Never raises: a source that cannot be read falls back to the floor.
    """
    if not uri:
        return None
    local = await _probe_local_video_seconds(ctx, uri)
    if local is not None or not allow_remote or uri.startswith("file://"):
        return local
    app_ctx = ctx.request_context.lifespan_context
    try:
        data = await fetch(
            uri,
            allowed_dir=app_ctx.data_folder,
            allowed_gcs_buckets=app_ctx.allowed_gcs_buckets,
        )
    except Exception:
        logger.debug(
            "Could not read %s to measure an extension source", uri, exc_info=True
        )
        return None
    if data is None:
        return None
    return await _probe_media(measure_video_duration_bytes, data)


async def _fetch_local_source(
    ctx: Context[ServerSession, AppContext],
    uri: str,
) -> bytes | None:
    """Read a LOCAL source for a pre-flight, or None.

    Used by both families: a dry run is documented as free, instant and
    offline, so it reads a file:// source the caller already has and leaves
    gs:// and http(s):// to the real run. Never raises: a quote that cannot open the file falls back
    to the documented maximum and says so, rather than failing a call that
    generates nothing.
    """
    if not uri.startswith("file://"):
        return None
    try:
        app_ctx = ctx.request_context.lifespan_context
        return await fetch(
            uri,
            allowed_dir=app_ctx.data_folder,
            allowed_gcs_buckets=app_ctx.allowed_gcs_buckets,
        )
    except Exception:
        logger.debug("Could not read %s for a pre-flight", uri, exc_info=True)
        return None


async def _fetch_omni_gcs_source_to_measure(
    ctx: Context[ServerSession, AppContext],
    uri: str,
    warnings: list[str],
) -> bytes | None:
    """Download a gs:// source that is SENT by URI, only to measure it.

    _omni_gcs_source passes a 1.2 source to Vertex as a URI, and skipping the
    download with it left the source unmeasured on a real run: an extension
    chain priced every turn at the 40s ceiling, skipped its cumulative-cap
    warning, and max_cost_usd refused chains that fit; an edit lost its 15s
    source check; an upscale billed the 15s bound. The bytes are probed and
    dropped, never sent. Real runs only: a dry run does not download a remote
    source.

    Never raises. A source this server cannot read (over the fetch cap, or
    outside its credentials) may still be readable by the service, so the call
    proceeds on the conservative bounds an unmeasured source gets, and
    ``warnings`` says so.
    """
    app_ctx = ctx.request_context.lifespan_context
    try:
        data = await fetch(
            uri,
            allowed_dir=app_ctx.data_folder,
            allowed_gcs_buckets=app_ctx.allowed_gcs_buckets,
        )
    except Exception:  # pragma: no cover - fetch() already returns None
        logger.debug("Could not download %s to measure it", uri, exc_info=True)
        data = None
    if data is None:
        warnings.append(
            f"input_video_uri {uri} could not be downloaded to measure it "
            f"(over the {MAX_FETCH_BYTES // (1024 * 1024)} MB fetch cap, or "
            "unreadable with this server's credentials). The service reads it "
            "directly, but its length was not checked here and the cost uses "
            "the conservative bound for an unmeasured source."
        )
    return data


async def _fetch_omni_media(
    ctx: Context[ServerSession, AppContext],
    uri: str,
    param: str,
) -> bytes:
    """Fetch one omni input, or raise a message naming the parameter."""
    app_ctx = ctx.request_context.lifespan_context
    data = await fetch(
        uri,
        allowed_dir=app_ctx.data_folder,
        allowed_gcs_buckets=app_ctx.allowed_gcs_buckets,
    )
    if data is None:
        raise ValueError(_fetch_failure(param, uri, app_ctx.data_folder))
    return data


async def _fetch_optional_omni_media(
    ctx: Context[ServerSession, AppContext],
    uri: str | None,
    param: str,
) -> bytes | None:
    """Same, for an input the caller may simply not have supplied."""
    if uri is None:
        return None
    return await _fetch_omni_media(ctx, uri, param)


def _fetch_failure(param: str, uri: str, data_dir: Path) -> str:
    """Message for a failed fetch that names WHY, for local paths.

    fetch() returns None on any failure, so a confinement refusal and a
    missing file both surfaced as the same "Could not fetch" — a caller cannot
    tell whether to move the file into DATA_FOLDER or fix a typo'd path. The
    two cases need different actions, so for a local path we re-run the same
    containment/existence check the fetch used and name the reason. Remote
    (gs://, http) failures stay generic: the reason is not knowable offline.
    """
    base = f"Could not fetch {param}: {uri}"
    if not uri or uri.startswith(("gs://", "http://", "https://")):
        return base
    raw = uri[7:] if uri.startswith("file://") else uri
    try:
        _validate_local_path(Path(raw), data_dir)
    except ValueError as reason:
        if "outside the allowed directory" in str(reason):
            return (
                f"{base} — outside the permitted data folder (DATA_FOLDER at "
                f"{data_dir}); copy the file under that folder, or pass a "
                "gs:// or https:// URI instead"
            )
        return f"{base} — {reason}"
    return base  # validation passes; the failure was a read/size error


def _assert_local_source(uri: str | None, data_dir: Path, param: str) -> None:
    """Refuse, on a dry run, a local source the real run would reject.

    The project invariant is that a quote refuses everything the real run
    refuses. A local file source that is missing or outside DATA_FOLDER is
    exactly what fetch() rejects on the real run, yet dry_run priced it — so a
    quote succeeded for a call guaranteed to fail. Remote sources (gs://, http,
    https) are not checkable offline, so they are left for the real fetch and
    still price in a quote. Reuses the same containment/existence check and the
    same message the real run emits (via _fetch_failure), so the dry-run
    refusal reads identically to the render's.
    """
    if not uri or uri.startswith(("gs://", "http://", "https://")):
        return
    raw = uri[7:] if uri.startswith("file://") else uri
    try:
        _validate_local_path(Path(raw), data_dir)
    except ValueError:
        raise ValueError(_fetch_failure(param, uri, data_dir)) from None


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

    The allowlist check runs through `_assert_gcs_bucket_allowed`, the same
    gate every gs:// *input* passes, so an output destination cannot be the
    one gs:// entry point that is silently unchecked — a render was previously
    written into an attacker-named bucket without so much as a log line.
    Default-deny was considered and rejected: the README documents an unset
    allowlist as "gs:// fetches log a warning", so denying would break every
    existing deployment that relies on ambient credentials. Warning uniformly
    is the change that makes behaviour consistent without breaking the
    documented contract.
    """
    gcs_uri = output_gcs_uri or default_bucket
    # GCS output only works on Vertex AI. On the Gemini API path, reject an
    # explicit output_gcs_uri (the caller asked for the impossible) but
    # silently drop a VIDEO_GCS_BUCKET env default so the call can still
    # succeed inline. This drop happens BEFORE format validation so a
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
        # No allowlist configured: warn, exactly as every gs:// input does.
        # Silence here is what let an unchecked output destination through.
        _assert_gcs_bucket_allowed(bucket, allowed_buckets)
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

    # Validates the inline service-account JSON before any client is built,
    # so a broken credential fails here and names its variable. Nothing is
    # written: credentials are a process-wide object now (see
    # _service_account_credentials), because on an HTTP transport this
    # lifespan runs once PER SESSION and a per-session temp file was deleted
    # from under every other live session on teardown.
    temp_creds_path = setup_vertex_credentials()
    try:
        client = create_client()
        gemini_api_client = create_gemini_api_client()
        video_gcs_bucket = os.environ.get("VIDEO_GCS_BUCKET")
        allowed_gcs_buckets = _compute_allowed_gcs_buckets()
        omni_backend_default = _parse_omni_backend_env(
            primary_is_vertex=bool(
                getattr(getattr(client, "_api_client", None), "vertexai", False)
            ),
            gemini_api_available=gemini_api_client is not None,
        )

        yield AppContext(
            data_folder=data_folder,
            images_dir=images_dir,
            videos_dir=videos_dir,
            client=client,
            gemini_api_client=gemini_api_client,
            temp_creds_path=temp_creds_path,
            video_gcs_bucket=video_gcs_bucket,
            allowed_gcs_buckets=allowed_gcs_buckets,
            omni_backend_default=omni_backend_default,
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


def _read_thought_signature(path: Path, allowed_dir: Path) -> str:
    """Read a thought-signature file with containment and size enforced.

    The path is caller-supplied, so `_validate_local_path` alone was not
    enough: containment says the file is inside DATA_FOLDER, not that it is
    small, and every rendered video and image the server writes is inside
    DATA_FOLDER. An uncapped read_text() of a 157 MB render cost ~150 MB RSS
    per call — the one fetch path that ignored MAX_FETCH_BYTES entirely.

    Blocking; call it off the event loop.
    """
    validated = _validate_local_path(path, allowed_dir)
    size = validated.stat().st_size
    if size > MAX_THOUGHT_SIGNATURE_BYTES:
        raise ValueError(
            f"thought_signature_url file is {size} bytes, over the "
            f"{MAX_THOUGHT_SIGNATURE_BYTES}-byte cap: {path}. Signatures the "
            "server writes are a few megabytes; a file this size is more "
            "likely a rendered image or video than a signature."
        )
    return validated.read_text()


async def _read_thought_signature_async(path: Path, allowed_dir: Path) -> str:
    """Async wrapper so the stat/read never runs on the event loop."""
    return await asyncio.to_thread(_read_thought_signature, path, allowed_dir)


def _is_public_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Whether an address is routable on the public internet.

    Enumerating the ranges to refuse (private / loopback / link-local /
    multicast / reserved / unspecified) left holes that matter in practice:
    RFC 6598 carrier-grade NAT space (100.64.0.0/10) matches none of those
    flags, and it is the standard pod and internal-load-balancer range on EKS
    and GKE — so `http://100.64.1.1/` was fetched and its content returned to
    the caller. `is_global` is the allowlist form of the same question and
    closes the whole class, including ranges IANA has not allocated yet.

    `is_global` alone is NOT sufficient, though, and the enumeration it
    replaced was right about one flag: `is_reserved`. CPython reports
    `is_global=True` for the IPv6 ranges that embed an IPv4 destination, so
    dropping `is_reserved` reopened the exact hole this guard exists to
    close. `64:ff9b::a9fe:a9fe` is NAT64-mapped 169.254.169.254 -- the cloud
    metadata service -- and on any host with a NAT64/DNS64 route (the
    IPv6-only pod case) it reaches it and returns the body. `::7f00:1` and
    `::a00:1` (IPv4-compatible, deprecated but still routed by some stacks)
    and `5f00::/16` are the same shape. None is `is_private`, `is_loopback`
    or `is_link_local`; every one is `is_reserved`. Checked the other way
    too: no real public address is `is_reserved`, so this costs nothing.
    """
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        # ::ffff:a.b.c.d reaches the same host as a.b.c.d, so it must be
        # judged as the IPv4 address it wraps rather than as an IPv6 one.
        ip = mapped
    if ip.is_reserved:
        # See the docstring: the IPv4-embedding IPv6 ranges are is_global=True,
        # so this is the only check standing between them and a fetch.
        return False
    if getattr(ip, "is_site_local", False):
        # Deprecated IPv6 site-local (fec0::/10) is absent from CPython's
        # private-networks table, so `is_global` reports it as public even on
        # 3.13 — yet it is exactly the internal addressing this guard exists
        # to refuse.
        return False
    if ip.is_multicast:
        # Some multicast scopes are is_global=True (ff02::1 among them), and
        # a multicast destination is never a legitimate fetch target.
        return False
    return ip.is_global


def _assert_http_host_public(url: str) -> list[str]:
    """Resolve an http(s) URL's host and reject any non-public answer.

    Mitigates SSRF against cloud metadata (169.254.169.254), localhost, and
    internal networks. Every address the name resolves to must pass, so a
    record mixing a public and an internal answer is refused outright.

    Returns the vetted addresses so the caller can connect to exactly what was
    checked. Discarding them and letting aiohttp resolve the name again left a
    DNS-rebinding window: a TTL-0 or round-robin record can hand this check a
    public address and the connector an internal one. See
    `_VettedAddressResolver` for how the answer is pinned.

    Synchronous; use `_assert_http_host_public_async` from async code so the
    DNS lookup does not block the event loop.
    """
    parsed = urlparse(url)
    host = parsed.hostname
    if not host:
        raise ValueError(f"URL missing host: {url}")
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise ValueError(f"Could not resolve host '{host}': {e}") from e
    vetted: list[str] = []
    for info in infos:
        addr = info[4][0]
        ip = ipaddress.ip_address(addr)
        if not _is_public_ip(ip):
            raise ValueError(f"Refusing to fetch non-public address: {host} -> {addr}")
        if addr not in vetted:
            vetted.append(addr)
    if not vetted:
        raise ValueError(f"Could not resolve host '{host}': no addresses returned")
    return vetted


async def _assert_http_host_public_async(url: str) -> list[str]:
    """Async wrapper that runs the DNS lookup off the event loop."""
    return await asyncio.to_thread(_assert_http_host_public, url)


def _host_pin_keys(host: str) -> list[str]:
    """Every spelling of a hostname aiohttp might hand the resolver.

    urlparse lowercases but leaves a unicode host as-is, while yarl gives the
    connector the IDNA/punycode form. Pinning under both spellings keeps an
    internationalized domain from missing its pin and failing closed.
    """
    keys = [host]
    try:
        encoded = host.encode("idna").decode("ascii")
    except (UnicodeError, ValueError):
        return keys
    if encoded not in keys:
        keys.append(encoded)
    return keys


class _VettedAddressResolver(AbstractResolver):
    """Resolver that hands aiohttp only addresses the SSRF guard vetted.

    aiohttp's default ThreadedResolver performs its own lookup, so the guard's
    answers were thrown away and the connection could land on a different
    address than the one that was checked (DNS rebinding). Pinning closes that
    window: the socket goes to the vetted IP, while the request URL still
    carries the hostname, so the Host header and TLS SNI — and therefore
    certificate validation — are unchanged.

    Pins are replaced before every redirect hop, so each hop connects only to
    what was vetted for that hop.

    Residual gap, stated honestly: when a name legitimately resolves to
    several public addresses all of them are pinned and aiohttp may use any,
    and the guard requires every answer to pass, so this narrows the window to
    "one of the addresses this process itself resolved and approved" rather
    than eliminating DNS variance entirely.
    """

    def __init__(self) -> None:
        self._pins: dict[str, list[str]] = {}

    def pin(self, host: str, addresses: list[str]) -> None:
        """Replace the vetted addresses for a host (called once per hop)."""
        for key in _host_pin_keys(host):
            self._pins[key] = list(addresses)

    async def resolve(
        self, host: str, port: int = 0, family: socket.AddressFamily = socket.AF_INET
    ) -> list[ResolveResult]:
        addresses = self._pins.get(host)
        if not addresses:
            # Fail closed: an unpinned name is one the guard never approved,
            # and falling back to a fresh lookup here would reopen the very
            # rebinding window this resolver exists to close.
            raise OSError(f"No vetted address for host '{host}'")
        return [
            ResolveResult(
                hostname=host,
                host=addr,
                port=port,
                family=socket.AF_INET6 if ":" in addr else socket.AF_INET,
                proto=socket.IPPROTO_TCP,
                flags=0,
            )
            for addr in addresses
        ]

    async def close(self) -> None:
        self._pins.clear()


def _gcs_no_allowlist_message(bucket: str) -> str:
    """The warning for a gs:// bucket reached with no allowlist configured."""
    return (
        f"gs:// access to bucket '{bucket}' with no allowlist configured; "
        "set GCS_ALLOWED_BUCKETS or VIDEO_GCS_BUCKET to restrict access."
    )


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
        message = _gcs_no_allowlist_message(bucket)
        logger.warning("%s", message)
        return message
    if bucket not in allowed:
        raise ValueError(
            f"GCS bucket '{bucket}' is not in the allowlist. "
            f"Configured buckets: {sorted(allowed)}"
        )
    return None


_storage_client: storage.Client | None = None
_storage_client_lock = threading.Lock()


def _get_storage_client() -> storage.Client:
    """Return the process-wide GCS client, building it once on first use.

    `storage.Client()` discovers credentials, and with no ADC present that
    means probing the GCE metadata server: 8.88 seconds, measured. It was
    being constructed per fetch and directly on the event loop, so a two-input
    tool (generate_transition, generate_bridge) froze the whole server —
    including every in-flight Veo poll and the MCP read loop — for ~18s. Call
    this only from a worker thread.

    A failed construction is deliberately not cached: credentials can appear
    after startup (a metadata server that was slow to come up, a mounted
    service-account file), and caching the failure would make that
    unrecoverable without a restart.
    """
    global _storage_client
    with _storage_client_lock:
        if _storage_client is None:
            found = _service_account_credentials()
            if found is not None:
                creds, project = found
                _storage_client = storage.Client(credentials=creds, project=project)
            else:
                _storage_client = storage.Client()
        return _storage_client


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

            def _download_capped() -> bytes:
                # Client construction happens here, inside the worker thread:
                # credential discovery is blocking network work and must never
                # run on the event loop.
                blob = _get_storage_client().bucket(bucket_name).blob(object_path)
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
            resolver = _VettedAddressResolver()
            # use_dns_cache=False so the pin set for THIS hop is the one the
            # connector uses; a cached entry from an earlier hop would defeat
            # the per-hop re-validation.
            connector = aiohttp.TCPConnector(resolver=resolver, use_dns_cache=False)
            try:
                async with aiohttp.ClientSession(
                    timeout=timeout, connector=connector
                ) as session:
                    current_url = uri
                    for _hop in range(MAX_HTTP_REDIRECTS + 1):
                        vetted = await _assert_http_host_public_async(current_url)
                        if vetted:
                            # Connect to the address that was just checked, not
                            # to whatever a second lookup would return.
                            host = urlparse(current_url).hostname or ""
                            resolver.pin(host, list(vetted))
                        async with session.get(
                            current_url, allow_redirects=False
                        ) as resp:
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
            finally:
                # The session normally owns and closes the connector, but a
                # test double for ClientSession does not — close it here so a
                # connector is never left dangling.
                await connector.close()

        # Local file access — require allowed_dir
        if allowed_dir is None:
            raise ValueError(
                f"Local file access is not allowed without a configured DATA_FOLDER: {uri}"
            )

        # Reject unsupported schemes (ftp://, s3://, data:, ...) with a clear
        # message instead of letting them fall into local-path validation,
        # which would misreport them as "outside the allowed directory".
        if not uri.startswith("file://") and ("://" in uri or uri.startswith("data:")):
            raise ValueError(f"Unsupported URI scheme: {uri}")

        # Resolving, stat and reading all touch the filesystem, and a slow
        # disk or a wedged mount blocks whatever thread does it. On the event
        # loop that froze every request, and no deadline a caller set around
        # this await could fire. Off the loop, on a pool of its own, a stall
        # parks one of its threads and the caller's deadline still holds.
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _get_local_read_executor(), _read_local_file, uri, allowed_dir, max_bytes
        )
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


def _read_local_file(uri: str, allowed_dir: Path, max_bytes: int) -> bytes:
    """Read a file:// URI or bare path inside ``allowed_dir``. Blocking.

    The location is validated before any probe, so a path outside
    ``allowed_dir`` never gets a stat() call that could leak whether a file
    exists through timing or differing errors.
    """
    raw = uri[len("file://") :] if uri.startswith("file://") else uri
    path = _validate_local_path(Path(raw), allowed_dir)
    if path.stat().st_size > max_bytes:
        raise ValueError(f"File exceeds size cap {max_bytes}: {path}")
    return path.read_bytes()


# Local file reads get their own pool, for the reason the probe pools below
# do: a thread blocked on a slow or wedged mount cannot be cancelled, so it
# must not be one the rest of the server draws on. Reads are per input and
# can run to MAX_FETCH_BYTES, so they are kept apart from the small probes.
_LOCAL_READ_MAX_WORKERS = 4
_local_read_executor: ThreadPoolExecutor | None = None
_local_read_lock = threading.Lock()


def _get_local_read_executor() -> ThreadPoolExecutor:
    """Return the process-wide local-file read pool, created on first use."""
    global _local_read_executor
    with _local_read_lock:
        if _local_read_executor is None:
            _local_read_executor = ThreadPoolExecutor(
                max_workers=_LOCAL_READ_MAX_WORKERS,
                thread_name_prefix="local-read",
            )
        return _local_read_executor


_SIDECAR_MAX_BYTES = 256 * 1024
_SIDECAR_SCAN_TIMEOUT_SECONDS = 5.0

# Dedicated pool for caller-triggered filesystem probes.
#
# asyncio.to_thread runs on the loop's SINGLE SHARED default executor
# (max_workers = min(32, cpu_count + 4) — 12 here, 6 on a two-core container),
# and a thread blocked in a syscall on a stale mount cannot be cancelled: the
# wait_for deadline returns the request while the thread stays parked forever.
# So N concurrent edit_video dry runs against a wedged mount permanently
# exhaust the pool that every generation path also uses (image, video, omni,
# storyboard, frame extraction) — the server stops rendering entirely while
# each individual request still looks like it handled its timeout correctly.
# Giving probes their own small pool means a wedged mount can consume only
# these threads. Queued work behind them still waits, but render work is
# never behind it.
_PROBE_EXECUTOR_MAX_WORKERS = 4
_probe_executor: ThreadPoolExecutor | None = None
_probe_executor_lock = threading.Lock()

# Media probes get their own pool for the same reason, one step further out.
# Each one shells out to ffmpeg through imageio; a malformed or half-written
# file, a stalled mount or a container under memory pressure can leave that
# subprocess unreaped and the worker parked, and a parked worker cannot be
# cancelled. Run on asyncio.to_thread — which is what these did — they draw on
# the loop's single shared default executor, the same twelve threads every
# fetch, every image render and every frame extraction also use, so a burst of
# probes can starve work that has nothing to do with media at all.
#
# Sized above the probe pool because probes are per-CALL (one duration, one
# dimension, one per reference clip) rather than per-request, and a chained
# extension issues several in a row.
_MEDIA_PROBE_MAX_WORKERS = 6
_MEDIA_PROBE_TIMEOUT_SECONDS = 30.0
_media_probe_executor: ThreadPoolExecutor | None = None
_media_probe_lock = threading.Lock()


def _get_media_probe_executor() -> ThreadPoolExecutor:
    """Return the process-wide media-probe pool, created on first use."""
    global _media_probe_executor
    with _media_probe_lock:
        if _media_probe_executor is None:
            _media_probe_executor = ThreadPoolExecutor(
                max_workers=_MEDIA_PROBE_MAX_WORKERS,
                thread_name_prefix="media-probe",
            )
        return _media_probe_executor


async def _decode_media(func: Any, /, *args: Any) -> Any:
    """Run one ffmpeg-backed DECODE off the loop, bounded and deadlined.

    Same pool as _probe_media, opposite error contract: frame extraction has
    no honest "not measured" answer — a bridge without its endpoints is not a
    degraded bridge, it is a different render — so a failure propagates to the
    per-beat handler that already knows what to do with it.
    """
    loop = asyncio.get_running_loop()
    return await asyncio.wait_for(
        loop.run_in_executor(_get_media_probe_executor(), func, *args),
        timeout=_MEDIA_PROBE_TIMEOUT_SECONDS,
    )


async def _probe_media(func: Any, /, *args: Any) -> Any:
    """Run one ffmpeg-backed probe off the loop, bounded and deadlined.

    Returns None on timeout rather than raising: every caller of these probes
    already treats "not measured" as a legitimate answer and reports it as
    such, so a slow probe degrades the provenance of a figure instead of
    failing a render that has already been paid for.
    """
    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(_get_media_probe_executor(), func, *args),
            timeout=_MEDIA_PROBE_TIMEOUT_SECONDS,
        )
    except Exception:
        logger.warning(
            "Media probe %s timed out or failed after %.0fs; continuing without "
            "a measurement",
            getattr(func, "__name__", func),
            _MEDIA_PROBE_TIMEOUT_SECONDS,
            exc_info=True,
        )
        return None


def _get_probe_executor() -> ThreadPoolExecutor:
    """Return the process-wide filesystem-probe pool, created on first use."""
    global _probe_executor
    with _probe_executor_lock:
        if _probe_executor is None:
            _probe_executor = ThreadPoolExecutor(
                max_workers=_PROBE_EXECUTOR_MAX_WORKERS,
                thread_name_prefix="fs-probe",
            )
        return _probe_executor


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
    reads are size-capped, and the index is consulted before any scan.
    Anything unreadable is skipped, never fatal — the caller's fallback
    already reports an unknown source honestly.

    Args:
        videos_dir: Directory the video sidecars are written to.
        interaction_id: The interaction whose duration is wanted.

    Returns:
        The recorded duration in seconds, or None if it cannot be determined.
    """
    manifest = _manifest_for_interaction(videos_dir, interaction_id)
    if manifest is None:
        return None
    if str(manifest.get("duration_source", "")).startswith("upper bound"):
        # That render's length was never established — the manifest holds
        # the billing ceiling. Reporting it as the source's real length
        # would launder a bound into a fact one hop later.
        return None
    duration = manifest.get("duration_seconds")
    if isinstance(duration, (int, float)) and duration > 0:
        return float(duration)
    return None


def _manifest_for_interaction(
    videos_dir: Path, interaction_id: str
) -> dict[str, Any] | None:
    """The newest sidecar manifest naming ``interaction_id``, or None.

    The hardened half of the lookup above, split out because a second caller
    needs a different field off the same record: which backend the interaction
    was created on. The scan stays paranoid — the media directory is
    caller-controlled and may live on a network mount, so only regular files
    are opened, reads are size-capped, and anything unreadable is skipped
    rather than raised. The scan reads every candidate -- an earlier cap of
    200 on the reads made it lose any record older than that, and a lookup
    that answers None for an id the directory holds is not bounded, it is
    wrong. A hit backfills the index, so the full walk is paid once per id.

    The index is tried first. Before it existed, an interaction past the
    newest 200 sidecars returned None from a directory that plainly contained
    it -- and every fact `PriorInteraction` exists to carry went with it:
    `prefer_backend` fell back to None, so the last turn of a chain carrying
    output_gcs_uri could be routed to Vertex holding a Gemini-API-minted id
    (the exact failure PriorInteraction was added to fix), and
    extend_video_omni's `prior.model != spec.model` refusal was disabled too.
    """
    entry_path = _interaction_index_path(videos_dir, interaction_id)
    try:
        entry_stat = entry_path.stat()
        if S_ISREG(entry_stat.st_mode) and entry_stat.st_size <= _SIDECAR_MAX_BYTES:
            entry = json.loads(entry_path.read_text())
            # The id is re-checked against the manifest below, so a truncated
            # digest that collided would fall through to the scan rather than
            # return the wrong interaction.
            if isinstance(entry, dict) and isinstance(entry.get("manifest"), dict):
                # A remote delivery: the manifest lives in the entry, there
                # being no local sidecar to point at.
                inline = entry["manifest"]
                if inline.get("interaction_id") == interaction_id:
                    return inline
            if isinstance(entry, dict) and isinstance(entry.get("sidecar"), str):
                # Name only: joining a stored path would let a tampered index
                # read outside the media directory.
                named = videos_dir / Path(str(entry["sidecar"])).name
                named_stat = named.stat()
                if S_ISREG(named_stat.st_mode) and (
                    named_stat.st_size <= _SIDECAR_MAX_BYTES
                ):
                    manifest = json.loads(named.read_text())
                    if (
                        isinstance(manifest, dict)
                        and manifest.get("interaction_id") == interaction_id
                    ):
                        return manifest
    except (OSError, ValueError):
        # No index entry, an unreadable one, or a sidecar that has since been
        # deleted: fall through to the scan.
        pass

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
    # Every candidate, not the newest 200. That cap made the scan bounded and
    # wrong: a directory with more than 200 renders written
    # before the index existed could not find any older one, and the first
    # version of the index left that as it was while claiming the scan was
    # "still correct for everything written before the index existed". Reads
    # stay size-capped and regular-file-only; a hit writes the index entry the
    # record never had, so this full walk happens at most once per id.
    for _mtime, sidecar in candidates:
        try:
            manifest = json.loads(sidecar.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(manifest, dict):
            continue
        if manifest.get("interaction_id") == interaction_id:
            _write_interaction_index(sidecar, manifest)
            return manifest
    return None


@dataclass(frozen=True)
class PriorInteraction:
    """What this server recorded about an interaction it is asked to continue.

    Everything a continuation needs, from one sidecar scan, because the scan
    is the expensive part:

    * ``backend`` — an interaction lives on ONE backend: an id minted by the
      Gemini Developer API does not resolve on Vertex AI and vice versa. But
      _client_for_omni picks from the CURRENT call's needs, chiefly whether it
      wants GCS output, so without this a continuation could be sent somewhere
      the interaction it names has never existed.
    * ``model`` — an interaction cannot change models. Continuing a
      preview-model interaction under a 1.2 request would go out as a bare
      conversational edit (the task cannot ride alongside a
      previous_interaction_id) and be billed as an extension. A retired ID is
      kept as recorded, not folded onto its replacement, so a continuation of
      it is refused (_retired_interaction_refusal) rather than sent to a model
      that does not hold it.
    * ``duration_seconds`` — the source's real length, which bounds what a
      continuation delivered somewhere unmeasurable could have rendered.
    * ``resolution`` — what the clip rendered at, which an upscale starts from.
    * ``video_url`` — where the clip was saved (file:// or gs://), which is
      what a caller can still pass as input_video_uri when the interaction
      itself cannot be continued.

    Every field is None when nothing was recorded (an older sidecar, a remote
    render that never got one), in which case the caller falls back rather
    than refusing.
    """

    backend: str | None = None
    model: str | None = None
    duration_seconds: float | None = None
    resolution: str | None = None
    video_url: str | None = None
    # Whether duration_seconds was measured from the rendered file, rather
    # than the length requested; only a measured length is priced as is.
    duration_measured: bool = False
    # The task the interaction ran ("upscale", say), when recorded. An
    # upscale interaction cannot be continued by id; see
    # _upscale_interaction_clip.
    task: str | None = None


def _continuation_omni_model(omni_model: str | None, prior: PriorInteraction) -> str:
    """The omni model a call runs on: the caller's choice, else the recorded one.

    An interaction cannot change models, so a continuation left unpinned has
    exactly one model that can serve it -- the one recorded for it. Resolving
    an unset omni_model to DEFAULT_OMNI_MODEL instead would refuse every
    default-args continuation of an interaction recorded on another model.
    Nothing recorded (a fresh render, an unknown id) takes the default.
    An explicit choice is returned as given, so a mismatch with the record is
    refused by _validate_omni_continuation_source rather than overridden.
    Every omni tool resolves through here before its dry_run branch, so a
    quote and a render cannot pick different models.
    """
    if omni_model is not None:
        return omni_model
    return prior.model or DEFAULT_OMNI_MODEL


def _retired_interaction_refusal(
    prior: PriorInteraction, interaction_id: str | None, task: str | None = None
) -> str | None:
    """Why an interaction recorded on a retired omni model cannot continue, or None.

    A request naming the retired model is redirected to its replacement, but
    an interaction cannot change models: the replacement does not hold the
    retired model's context, so the turn cannot be redirected with it. What
    can move is the video the interaction produced, so the refusal names the
    saved copy and the tool that takes it as input_video_uri. Every omni tool
    checks this before anything is fetched or quoted, so a dry run refuses it
    word for word.

    Args:
        prior: What was recorded about the interaction.
        interaction_id: The interaction the call continues, if any.
        task: ``extend`` or ``upscale`` for those tools; anything else is an
            edit, which generate_video_omni takes from an input_video_uri
            (edit_video has no such argument).
    """
    if not interaction_id or not omni_model_is_retired(prior.model):
        return None
    tool, verb = {
        "extend": ("extend_video_omni", "extend"),
        "upscale": ("upscale_video_omni", "upscale"),
    }.get(task or "", ("generate_video_omni", "edit"))
    replacement = omni_spec(prior.model).model
    source = _saved_clip_source(prior)
    return (
        f"Interaction {interaction_id} was created on {prior.model}, which this "
        f"server has retired, and an interaction cannot change models, so it "
        f"cannot be continued on {replacement}. To {verb} that clip on "
        f"{replacement}, call {tool} with {source} in place of "
        "previous_interaction_id. Nothing was rendered or billed."
    )


def _saved_clip_source(prior: PriorInteraction) -> str:
    """The input_video_uri a refused continuation can pass instead, as words.

    Names the clip this server saved for the interaction, so the way out is
    one copy-paste; says so when nothing was saved here rather than inventing
    a path.
    """
    if prior.video_url:
        return f"input_video_uri='{prior.video_url}' (the clip it saved)"
    return "the clip it produced as input_video_uri (no saved copy is recorded here)"


# Live, 2026-11-10: an upscale sent with the previous_interaction_id of an
# upscale interaction was refused after submission with this, while the clip
# that interaction saved, sent as input_video_uri, upscaled. Its context holds
# both the clip it was given and the one it made, so it is no single source.
_UPSCALE_BY_ID_REFUSAL = "Exactly one input video is required for the upscale task"


def _upscale_interaction_clip(
    prior: PriorInteraction, interaction_id: str | None, tool: str
) -> str | None:
    """The saved clip to send in place of an upscale interaction's id, or None.

    None when the interaction is not a recorded upscale. For one that is, the
    clip it saved, which ``tool`` sends as the source instead; refused before
    anything is sent when there is no such clip to send.
    """
    if not interaction_id or prior.task != "upscale":
        return None
    saved = prior.video_url
    if saved and not (saved.startswith("file://") and not Path(saved[7:]).is_file()):
        return saved
    where = (
        "this server recorded no saved copy of it"
        if not saved
        else f"the clip it saved, {saved}, is no longer there"
    )
    raise ValueError(
        f"Interaction {interaction_id} is an upscale, which the service will not "
        f'continue by id ("{_UPSCALE_BY_ID_REFUSAL}"); its saved clip is sent '
        f"as the source instead, and {where}. Pass the upscaled clip as "
        f"input_video_uri to {tool}. Nothing was rendered or billed."
    )


def _upscale_interaction_note(interaction_id: str, clip: str, dry_run: bool) -> str:
    """The warning a call carries when it sent an upscale's clip for its id."""
    return (
        f"Interaction {interaction_id} is an upscale, which the service will not "
        f'continue by id ("{_UPSCALE_BY_ID_REFUSAL}"), so the clip it saved, '
        f"{clip}, {'would be' if dry_run else 'was'} sent as input_video_uri "
        "instead. The result is a new interaction, not a turn of that one."
    )


def _validate_omni_continuation_source(
    spec: Any,
    prior: PriorInteraction,
    interaction_id: str | None,
    task: str | None = None,
) -> None:
    """Keep quotes and renders on the recorded model and within its limits.

    A model reaches here either because the caller named it or because
    _continuation_omni_model took the recorded one, so a mismatch is always a
    model the caller chose explicitly -- except an interaction recorded on a
    retired model, which no choice can continue and which is refused first.
    """
    retired = _retired_interaction_refusal(prior, interaction_id, task)
    if retired:
        raise ValueError(retired)
    if interaction_id and prior.model and prior.model != spec.model:
        raise ValueError(
            f"Interaction {interaction_id} was created on {prior.model}, and an "
            f"interaction cannot change models, so omni_model='{spec.model}' "
            f"cannot continue it. Leave omni_model unset (or pass "
            f"'{prior.model}') to continue it on {prior.model}, or start a new "
            f"{spec.model} interaction from the saved video by passing it as "
            "input_video_uri to generate_video_omni or extend_video_omni."
        )
    # An extension or an upscale of an interaction checks the length where it
    # measures the source; an edit's only evidence is the recorded length.
    if task not in ("extend", "upscale"):
        refusal = _omni_source_length_refusal(
            spec, prior.duration_seconds, task=task, uploaded=False
        )
        if refusal:
            raise ValueError(refusal)


def _prior_interaction(videos_dir: Path, interaction_id: str) -> PriorInteraction:
    """Read back what was recorded about ``interaction_id``."""
    manifest = _manifest_for_interaction(videos_dir, interaction_id)
    if manifest is None:
        return PriorInteraction()
    backend = manifest.get("backend")
    model = manifest.get("model")
    video_url = manifest.get("video_url")
    duration = manifest.get("duration_seconds")
    if str(manifest.get("duration_source", "")).startswith("upper bound"):
        # That render's length was never established; the manifest holds a
        # billing ceiling. Passing it on as a measured length would launder a
        # bound into a fact one hop later.
        duration = None
    return PriorInteraction(
        backend=backend if backend in ("vertex", "gemini_api") else None,
        # A retired ID stays as recorded: folding it onto its replacement
        # would claim the interaction lives on a model that never saw it.
        model=(
            str(model)
            if omni_model_is_retired(model)
            else omni_spec(model).model
            if is_omni_model(model)
            else None
        ),
        duration_seconds=(
            float(duration)
            if isinstance(duration, (int, float)) and duration > 0
            else None
        ),
        resolution=normalize_omni_resolution(manifest.get("resolution")),
        video_url=video_url if isinstance(video_url, str) and video_url else None,
        duration_measured=str(manifest.get("duration_source", "")).startswith(
            "measured"
        ),
        task=str(manifest["task"]) if manifest.get("task") else None,
    )


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
    request returns regardless, which is the property that matters here.

    That is why this runs on `_get_probe_executor()` and not `asyncio.to_thread`.
    The scan bounds do NOT keep the worst case to one thread per stuck quote,
    as this docstring previously claimed: to_thread shares the loop's single
    default executor with every generation path, so enough parked probe
    threads stop the server rendering anything at all. The dedicated pool caps
    the blast radius at the probes themselves.
    """
    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(
                _get_probe_executor(),
                _source_duration_for_interaction,
                videos_dir,
                interaction_id,
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


async def _prior_interaction_or_empty(
    videos_dir: Path, interaction_id: str | None
) -> PriorInteraction:
    """Run the sidecar lookup off the event loop, with a deadline.

    Same reasoning as _source_duration_or_none: bounded or not, this is
    filesystem work on a caller-controlled directory that may live on a slow
    mount, and one blocked coroutine blocks every request. A timeout degrades
    to "nothing recorded" — the fallbacks each caller already has — rather
    than hanging a call.
    """
    if not interaction_id:
        return PriorInteraction()
    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(
                _get_probe_executor(),
                _prior_interaction,
                videos_dir,
                interaction_id,
            ),
            timeout=_SIDECAR_SCAN_TIMEOUT_SECONDS,
        )
    except Exception:
        # Deliberately broad, for the same reason the duration lookup is: the
        # fallbacks are what this server would have done anyway.
        logger.warning(
            "Sidecar lookup for interaction %s failed or exceeded %.0fs; "
            "continuing without what it recorded",
            interaction_id,
            _SIDECAR_SCAN_TIMEOUT_SECONDS,
            exc_info=True,
        )
        return PriorInteraction()


def _video_cost_estimate(
    model: str,
    duration_seconds: float,
    *,
    resolution: str | None = None,
    include_audio: bool = True,
    actual: bool = False,
    presnapped: bool = False,
    generation_mode: str = "text_to_video",
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

    ``generation_mode`` is what makes that snap correct on the modes Veo
    overrides: reference-to-video always renders 8s and an extension always
    renders 7s, so dropping the mode quoted a 4s reference render at half the
    price the caller is actually charged. It only applies to the snapping
    path — an ``actual``/``presnapped`` figure is already the effective
    length.
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
                allowance = seconds - duration_seconds
                cost = dataclasses.replace(
                    cost,
                    is_estimate=True,
                    breakdown={
                        **cost.breakdown,
                        **(
                            {"encoder_allowance_seconds": allowance}
                            if allowance
                            else {}
                        ),
                    },
                )
            return cost
        return estimate_video_cost(
            model,
            duration_seconds,
            res,
            include_audio,
            generation_mode=generation_mode,
        )
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
    return not is_omni_model(str(segment.get("model") or ""))


def _video_cost(
    model: str,
    duration_seconds: float,
    *,
    resolution: str | None = None,
    include_audio: bool = True,
    actual: bool = False,
    presnapped: bool = False,
    generation_mode: str = "text_to_video",
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
            generation_mode=generation_mode,
        )
    )


def _snapped_duration_for_quote(
    model: str,
    duration_seconds: float,
    generation_mode: str = "text_to_video",
) -> float:
    """The duration a dry run should report: what will really be rendered.

    A quote that prints the request beside a price for the snapped value
    contradicts itself — 5s was reported as "5" next to a cost detailing "4s
    of video". Falls back to the request if pricing cannot be imported, since
    a quote is worth more than a failure.
    """
    try:
        from .pricing import snap_video_duration

        return float(snap_video_duration(model, duration_seconds, generation_mode))
    except Exception:  # pragma: no cover - defensive
        logger.debug("Could not snap quoted duration", exc_info=True)
        return duration_seconds


# Keys a generate_clip beat may carry. Anything else is refused rather than
# dropped: the beat dict is the one place in this server where a plausible
# wrong key costs full price and produces no diagnostic at all. An agent that
# chains generate_storyboard (which returns image_url) or copies
# generate_video's image_uri into a beat got a text-to-video render with no
# image conditioning and no way to tell from the response that anything was
# ignored.
_BEAT_KEYS = frozenset(
    {
        "prompt",
        "duration_seconds",
        "seed",
        "first_frame_uri",
        "negative_prompt",
        "audio_prompt",
    }
)

# A beat's length when it names none.
_DEFAULT_BEAT_SECONDS = 4.0


def _animatic_beat_seconds(beat: dict[str, Any], model: str, spec: Any) -> int:
    """How long an animatic renders a beat: as long as the Veo delivery will.

    A 3s beat previewed at 3s finalized at 4s on Veo, so the animatic and the
    clip it previews ran different lengths, unlike generate_video(draft=true),
    which already snaps. Clamped to what the animatic's Omni model renders.
    """
    veo_seconds = _snapped_duration_for_quote(model, _beat_seconds(beat))
    return min(spec.max_duration_seconds, max(3, round(veo_seconds)))


def _beat_seconds(beat: dict[str, Any]) -> float:
    """A validated beat's length: its duration_seconds, or the 4s default.

    An explicit null reads as "not given", as _validate_duration_seconds
    already reads it. Read with ``.get(key, default)``, a null passed
    validation and then failed in float(): the quote and the animatic's
    delivery pre-flight crashed the whole call on a raw Python error, while
    the render failed that beat alone.
    """
    value = beat.get("duration_seconds")
    return _DEFAULT_BEAT_SECONDS if value is None else float(value)


# Storyboard shot fields, accepted on a beat and ignored. generate_storyboard
# documents its shot list as feedable straight into generate_clip as `beats`,
# so refusing the two keys that workflow carries would break the chain the
# docstring advertises.
_BEAT_IGNORED_KEYS = frozenset({"caption", "notes"})

_SHOT_KEYS = frozenset({"prompt", "caption", "duration_seconds", "notes"})

# Wrong names seen (or trivially reachable) in agent-authored specs, mapped to
# what the caller meant. Keys are compared lowercased.
_BEAT_KEY_SUGGESTIONS: dict[str, str] = {
    "image": "first_frame_uri",
    "image_url": "first_frame_uri",
    "image_uri": "first_frame_uri",
    "first_frame": "first_frame_uri",
    "frame_uri": "first_frame_uri",
    "start_frame_uri": "first_frame_uri",
    "duration": "duration_seconds",
    "length": "duration_seconds",
    "seconds": "duration_seconds",
    "random_seed": "seed",
    "text": "prompt",
    "description": "prompt",
    "negative": "negative_prompt",
    "audio": "audio_prompt",
}

_SHOT_KEY_SUGGESTIONS: dict[str, str] = {
    "duration": "duration_seconds",
    "length": "duration_seconds",
    "seconds": "duration_seconds",
    "text": "prompt",
    "description": "prompt",
    "title": "caption",
    "slug": "caption",
    "slug_line": "caption",
    "label": "caption",
    "note": "notes",
    "camera": "notes",
    "camera_notes": "notes",
}


def _clip_total_source(segments: list[dict[str, Any]]) -> str:
    """Where a clip's total_duration_seconds comes from.

    The total was a bare sum of the snapped requests while every segment
    beside it carried a duration_source; a number without a source cannot be
    told apart from a measured one.
    """
    sources = [str(seg.get("duration_source") or "") for seg in segments]
    if sources and all(src.startswith("measured") for src in sources):
        return "sum of the measured segment durations"
    if not any(src.startswith("measured") for src in sources):
        return "sum of the snapped requested beat lengths and fixed 4s bridges, not measured"
    return "sum of measured and unmeasured segments; see each segment's duration_source"


# What a bridge between two beats asks for, on either model.
_CLIP_BRIDGE_PROMPT = "smooth cinematic cut between the two beats"


async def _measured_omni_segment(
    result: dict[str, Any],
    *,
    model: str,
    requested_seconds: float,
    requested_resolution: str | None,
) -> dict[str, Any]:
    """Length and resolution of an Omni clip segment, measured when it is local.

    The fields a segment carries for _assemble_clip_manifest to price it:
    measured from a file:// render, else the request, labelled as such.
    """
    fields: dict[str, Any] = {
        # The clip's model, as its beats carry, so one clip prices one way.
        "model": model,
        "duration_seconds": float(requested_seconds),
        "duration_source": (
            "the request: the rendered file could not be opened to measure it"
        ),
        "resolution": str(
            result.get("rendered_resolution") or requested_resolution or "720p"
        ),
        "resolution_source": (
            "the request: the rendered file could not be opened to measure it"
        ),
        "interaction_id": result.get("interaction_id"),
        "generation_mode": "animatic",
    }
    url = result.get("video_url") or ""
    if not (isinstance(url, str) and url.startswith("file://")):
        return fields
    path = Path(url[7:])
    seconds = await _probe_media(measure_video_duration, path)
    if seconds is not None:
        fields["duration_seconds"] = seconds
        fields["duration_source"] = "measured from the rendered video"
    size = await _probe_media(measure_video_dimensions, path)
    if size is not None:
        fields["rendered_dimensions"] = list(size)
        classified = classify_video_resolution(size)
        if classified is not None:
            fields["resolution"] = classified
            fields["resolution_source"] = "measured from the rendered video"
    return fields


def _assemble_clip_manifest(
    *,
    aspect_ratio: str,
    beat_model: str,
    animatic: bool,
    segments: list[dict[str, Any]],
    total_duration: float,
    beat_count: int,
    include_audio: bool,
    errors: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    """Build generate_clip's manifest, including the cost of what rendered.

    Split out of the tool so the cancellation path can produce the same
    document from a partial run: the beats already rendered were already
    billed, and their total is the one figure that exists nowhere else.
    """
    manifest: dict[str, Any] = {
        "kind": "clip",
        "aspect_ratio": aspect_ratio,
        "model": beat_model,
        "animatic": animatic,
        "segments": segments,
        "total_duration_seconds": total_duration,
        "total_duration_source": _clip_total_source(segments),
        "beat_count": beat_count,
    }
    # Total what the segments actually rendered, segment by segment —
    # summing the runtime into one estimate would snap it to 8s.
    try:
        from .pricing import sum_costs

        segment_costs = [
            _video_cost_estimate(
                seg.get("model", beat_model),
                float(seg.get("duration_seconds") or 0),
                resolution=str(seg.get("resolution") or "720p"),
                include_audio=bool(seg.get("audio_enabled", include_audio)),
                actual=_segment_is_metered(seg),
                presnapped=not _segment_is_metered(seg),
            )
            for seg in segments
            if seg.get("video_url")
        ]
        if segment_costs:
            manifest["cost"] = _cost_payload(sum_costs(segment_costs, label="clip"))
    except Exception:  # pragma: no cover - defensive
        logger.debug("Could not total clip cost", exc_info=True)
    if errors:
        manifest["errors"] = errors
    if warnings:
        # The same warning (e.g. "seed was not sent") is
        # emitted per beat and per bridge; dedupe at the clip level, preserving
        # first-seen order, so the manifest carries each distinct warning once
        # instead of one copy per segment.
        manifest["warnings"] = list(dict.fromkeys(warnings))
    return manifest


class _ExtensionBilling(NamedTuple):
    """How a Veo extension should be billed, and on what evidence."""

    billed_seconds: float
    turn_output_seconds: list[float]
    appended_seconds: float
    duration_source: str
    is_metered: bool
    warning: str | None


def veo_extension_billing(
    *,
    turns: int,
    measured_turns: list[float] | None = None,
    source_seconds: float | None = None,
) -> _ExtensionBilling:
    """Bill a Veo extension for what it renders, on the best evidence to hand.

    MEASURED: a Veo turn renders and bills the ASSEMBLED clip, so turn i is
    the source plus 7s per turn and a chain costs the SUM of those lengths.
    Billing the appended footage alone under-billed a single 4s-source
    extension by 57%.

    Three tiers, best evidence first: every turn measured, else a projection
    from a measured source, else the appended footage as a labelled floor. The
    projection matters because only the OUTPUT of a GCS-delivered render is out
    of reach — a local source still pins every turn's length, and skipping
    straight to the floor made a real run report less than its own pre-flight.

    Shared, because this is the second path to need it. generate_video's
    extend_video_uri mode had none of it: it billed the appended footage,
    asserted is_estimate=False for a file it never opened, and labelled the
    figure with the TEXT-TO-VIDEO rule ("Veo renders exactly the length it is
    sent") in the one mode where that rule is known to be false. Two
    implementations of one service's billing is what produced that.
    """
    measured = [t for t in (measured_turns or []) if t is not None]
    everything_measured = bool(turns) and len(measured) == turns
    projected = veo_extension_output_lengths(source_seconds, turns)
    appended = turns * VEO_EXTENSION_STEP_SECONDS

    if everything_measured:
        return _ExtensionBilling(
            billed_seconds=round(sum(measured), 3),
            turn_output_seconds=[round(t, 3) for t in measured],
            appended_seconds=appended,
            duration_source="measured from each rendered turn",
            is_metered=True,
            warning=None,
        )
    if projected:
        return _ExtensionBilling(
            billed_seconds=round(sum(projected), 3),
            turn_output_seconds=projected,
            appended_seconds=appended,
            duration_source=(
                "PROJECTED from a measured "
                f"{source_seconds:g}s source: the render was delivered "
                "remotely so it could not be opened, but a Veo turn renders "
                "the assembled clip, so turn i is the source plus "
                f"{VEO_EXTENSION_STEP_SECONDS:g}s per turn. This is the same "
                "figure the pre-flight quoted."
            ),
            is_metered=False,
            warning=None,
        )
    return _ExtensionBilling(
        billed_seconds=appended,
        turn_output_seconds=[],
        appended_seconds=appended,
        duration_source=(
            "FLOOR, not a metered figure: the render was delivered remotely "
            "and not opened, its source is remote too, and a Veo turn bills "
            "the assembled clip rather than the footage it appends. The true "
            "bill is higher by the source's length charged once per turn."
        ),
        is_metered=False,
        warning=(
            "This cost is a FLOOR. Neither the render nor its source could be "
            "measured, and Veo bills each turn's assembled length rather than "
            f"the {VEO_EXTENSION_STEP_SECONDS:g}s it appends."
        ),
    )


def _usd_of(cost: Any) -> float:
    """The dollar figure of a cost payload, whatever shape it came in."""
    if cost is None:
        return 0.0
    value = cost.get("usd") if isinstance(cost, dict) else getattr(cost, "usd", None)
    return float(value or 0.0)


def _over_budget_message(
    projected_usd: float,
    max_cost_usd: float,
    times: int,
    *,
    floor: bool = False,
    unmeasured: bool = False,
    measured_on_real_run: bool = False,
) -> str:
    """Why a chain was (or would be) refused against its caller's cap.

    ``measured_on_real_run`` is for a dry run that left a remote source
    unread: the real run downloads and measures it before the cap is applied,
    so the bound is not what decides the real run, and saying the source
    "could not be measured" misdescribed a quote that never tried.

    Extension chains are the one place a cap earns its keep: the bill grows
    with the SQUARE of `times`, because every turn re-renders the footage
    before it, and `times` accepts 20. Twenty turns off a 10s source bills
    ~1670s — several hundred dollars from one call with only a warning
    between the caller and the invoice.
    """
    # `times` here is the number of turns PRICED, which the caller takes from
    # the priced list -- not the number requested. The two differ when the
    # ceiling cuts a measured chain short, and quoting the requested count
    # against the clamped sum ("for 4 turn(s) is $9.14") named a number the
    # adjacent warning had just said would not be billed.
    # Two callers, two different honest labels. loop_extend (Veo) passes
    # `floor` when its projection genuinely under-states; extend_video_omni
    # passes `unmeasured`, where every unmeasured turn is priced at the
    # ceiling and the figure is therefore an upper bound, not a floor.
    if unmeasured and measured_on_real_run:
        basis = (
            "an UPPER BOUND (a dry run does not download a remote source, so "
            "every turn is priced at the ceiling; the real run measures the "
            "source first and refuses only if the measured chain is over the "
            "cap)"
        )
    elif unmeasured:
        basis = (
            "an UPPER BOUND (the source could not be measured, so every turn is "
            "priced at the ceiling; the true bill is at most this)"
        )
    elif floor:
        basis = "a FLOOR (the source could not be measured, so the true bill is higher)"
    else:
        basis = "the projected bill"
    return (
        f"Refused before rendering: {basis} for {times} extension turn(s) is "
        f"${projected_usd:.2f}, over the max_cost_usd of ${max_cost_usd:.2f} you "
        "set. Fewer turns cost far less than proportionally less — each turn "
        "re-bills everything before it — so reduce `times` first, or raise the "
        "cap deliberately."
    )


def _assemble_loop_extend_manifest(
    *,
    source_video_uri: str,
    prompt: str,
    served_model: str,
    aspect_ratio: str,
    steps: list[str],
    final_video_url: str,
    include_audio: bool,
    warnings: list[str],
    resolution: str = "720p",
    turn_seconds: list[float] | None = None,
    source_seconds: float | None = None,
) -> dict[str, Any]:
    """Build loop_extend's manifest for the extensions that actually ran.

    ``times`` is taken from the steps that completed rather than the count
    requested, so the same function describes a finished chain and one cut
    short by a disconnect — and never bills for a render that did not happen.
    """
    manifest: dict[str, Any] = {
        "kind": "loop_extend",
        "source_video_uri": source_video_uri,
        "prompt": prompt,
        # The model that actually ran: the impl normalizes an older
        # Gemini-API `-preview` spelling to its `-001` name.
        "model": served_model,
        "aspect_ratio": aspect_ratio,
        "times": len(steps),
        "final_video_url": final_video_url,
        "extension_steps": list(steps),
    }
    manifest["resolution"] = resolution
    billing = veo_extension_billing(
        turns=len(steps),
        measured_turns=turn_seconds,
        source_seconds=source_seconds,
    )
    everything_measured = billing.is_metered
    billed = billing.billed_seconds
    manifest["billed_seconds"] = billed
    manifest["appended_seconds"] = billing.appended_seconds
    if billing.turn_output_seconds:
        manifest["turn_output_seconds"] = billing.turn_output_seconds
    manifest["billed_seconds_source"] = billing.duration_source
    if billing.warning:
        warnings.append(billing.warning)
    cost = _video_cost(
        served_model,
        float(billed),
        # Priced at what the assembled clip measured, not at an assumption.
        # The tool cannot request a resolution, which says nothing about what
        # Veo handed back.
        resolution=resolution,
        include_audio=include_audio,
        # Only a fully measured chain is a metered figure; anything else stays
        # an estimate, which is what is_estimate exists to say.
        actual=everything_measured,
        presnapped=not everything_measured,
    )
    if cost:
        manifest["cost"] = cost
    if warnings:
        # Same warning repeats per step; dedupe, preserving order.
        manifest["warnings"] = list(dict.fromkeys(warnings))
    return manifest


def _write_clip_manifest(manifest: dict[str, Any]) -> str | None:
    """Write a clip manifest next to the last locally-rendered segment.

    Used when the response itself will never reach anyone (the caller
    disconnected mid-clip): every beat has its own sidecar, but the clip-level
    total lives only in the return value, so without this the record of what
    the run cost dies with the request. Written to ``<stem>_clip.json`` rather
    than ``<stem>.json`` so it cannot overwrite that segment's own sidecar.
    """
    for segment in reversed(manifest.get("segments") or []):
        url = segment.get("video_url") or ""
        if not isinstance(url, str) or not url.startswith("file://"):
            continue
        media_path = Path(url[7:])
        target = media_path.with_name(f"{media_path.stem}_clip.json")
        try:
            target.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        except OSError as e:
            logger.warning("Failed to write clip manifest %s: %s", target, e)
            return None
        return f"file://{target}"
    return None


def _normalize_spec_key(key: str) -> str:
    """Fold a spec key to letters and digits, so casing/spelling variants match."""
    return "".join(ch for ch in key.lower() if ch.isalnum())


def _validate_spec_keys(
    spec: dict[str, Any],
    field: str,
    allowed: frozenset[str],
    suggestions: dict[str, str],
    ignored: frozenset[str] = frozenset(),
) -> None:
    """Reject unknown keys in a beat/shot dict, naming what was probably meant.

    Silently dropping them is the expensive failure: the render still runs and
    still bills, just without the conditioning the caller paid for.
    """
    unknown = [
        str(key) for key in spec if str(key) not in allowed and str(key) not in ignored
    ]
    if not unknown:
        return

    problems: list[str] = []
    for key in sorted(unknown):
        target = suggestions.get(key.lower())
        if target is None:
            # Catches camelCase and punctuation variants ("durationSeconds",
            # "first-frame-uri") without an entry per spelling.
            normalized = _normalize_spec_key(key)
            target = next(
                (
                    name
                    for name in sorted(allowed)
                    if _normalize_spec_key(name) == normalized
                ),
                None,
            )
        problems.append(f"'{key}'" + (f" (did you mean '{target}'?)" if target else ""))

    raise ValueError(
        f"{field} has unknown key(s): {', '.join(problems)}. "
        f"Accepted keys: {', '.join(sorted(allowed))}."
    )


def _validate_seed(seed: Any, field: str) -> None:
    """Raise ValueError for a seed the render would choke on.

    Statically checkable and cheap here; left to the impl it surfaces as a
    TypeError from the seed comparison in src/video.py — after every earlier
    beat has already rendered and billed. Negative seeds are deliberately
    allowed: the impl drops them with a warning rather than failing.
    """
    if seed is None:
        return
    if isinstance(seed, bool) or not isinstance(seed, (int, float)):
        raise ValueError(f"{field} must be an integer, got {type(seed).__name__}")
    if isinstance(seed, float) and (not math.isfinite(seed) or seed != int(seed)):
        raise ValueError(f"{field} must be a whole number, got {seed!r}")


# Create MCP server with lifespan
mcp = FastMCP(
    "gemini-media-mcp",
    instructions=(
        "Generate and edit images and video on Google Gemini and Veo 3.1. "
        "plan_generation ranks tool and model options for an intent with costs "
        "and generates nothing; generate_storyboard and generate_clip build "
        "reviewable boards and multi-beat reels; generate_transition and "
        "generate_bridge join shots; edit_video and loop_extend revise and "
        "extend footage. Every generation tool takes dry_run to return a cost "
        "estimate before spending. generate_image, generate_video and "
        "generate_video_omni take director_notes: Gemini 4 Argon turns them "
        "into checks, reviews each take against them, and retakes on a fail."
    ),
    lifespan=app_lifespan,
)


# ============================================================================
# Virtual director
# ============================================================================
#
# Opt-in through director_notes on generate_image, generate_video and
# generate_video_omni. The tool calls itself once per take with the notes
# stripped, so a take is exactly the render the tool would have made without
# a director: same validation, same quote, same manifest. The rules of the
# review itself live in src/director.py.

# A request that timed out or hit a server error reported no usage, but Google
# may still have processed and billed it; the director cannot see that, so it
# says so.
_DIRECTOR_UNREPORTED_WARNING = (
    "A director request timed out or hit a server error. Google may still have "
    "processed and billed it; its cost is not included because it was never "
    "reported."
)

# Below this many seconds left on the caller's deadline a review is skipped
# rather than sent: a review that cannot finish still bills, and a call the
# host cancels has already billed. A MEDIUM-thinking review of a clip takes
# longer than one of an image.
_DIRECTOR_MIN_REVIEW_SECONDS: dict[str, float] = {"image": 30.0, "video": 60.0}

# The most time spent reading reference images for review, or less when the
# caller's deadline is nearer.
_DIRECTOR_REFERENCE_READ_SECONDS = 30.0

# Images the reviewer is shown from each list a render was given (image_uris,
# reference_image_uris). Each one costs up to 1,120 input tokens on every
# review, and fidelity to the first few is what a director checks. The image
# being edited and the first and last frames are always shown.
_DIRECTOR_MAX_REFERENCES = 3


def _uri_or_base64(
    kind: str, param: str, uri: str | None, b64: str | None
) -> tuple[str, str, str | None]:
    """An image a tool reads from ``{param}_uri``, or else ``{param}_base64``."""
    return (kind, f"{param}_uri", uri) if uri else (kind, f"{param}_base64", b64)


def _review_references(
    *inputs: tuple[str, str, str | None], omni: bool = False
) -> list[tuple[Any, str, str]]:
    """The images a render was given, as the reviewer will be shown them.

    Each input is (kind, parameter, URI or base64), in the order the tool
    reads them: what the render was edited from or anchored on, then the
    lists. Returns (ReviewReference, "uri" or "base64", value) for each one
    given. A list item keeps its place in the caller's list, so the second
    of reference_image_uris is reference 2 even when the first is empty.
    ``omni`` names omni's prompt tag for each.
    """
    from .director import LISTED_KINDS, ReviewReference

    counts: dict[str, int] = {}
    references: list[tuple[Any, str, str]] = []
    for kind, given_as, value in inputs:
        counts[kind] = counts.get(kind, 0) + 1 if kind in LISTED_KINDS else 1
        if value:
            references.append(
                (
                    ReviewReference(kind, counts[kind], given_as, omni),
                    "base64" if given_as.endswith("_base64") else "uri",
                    value,
                )
            )
    return references


def _named(references: list[Any]) -> str:
    """References as a response lists them: "reference 4 (...), ..."."""
    return ", ".join(ref.label for ref in references)


def _director_client(app_ctx: AppContext) -> genai.Client:
    """The client the director's understanding calls go through.

    Argon is served only from the global, us and eu locations on Vertex, so a
    Vertex deployment uses the global-location client omni already keeps; on
    the Gemini API the primary client serves it as-is.
    """
    if getattr(app_ctx.client._api_client, "vertexai", False):
        return _get_omni_vertex_global_client()
    return app_ctx.client


def _tool_payload(raw: Any) -> tuple[dict[str, Any], list[Any]]:
    """Split a tool's return into its JSON payload and any other content.

    generate_image returns a content list (a preview image beside the JSON);
    the video tools return the JSON string itself.
    """
    if isinstance(raw, str):
        return json.loads(raw), []
    text = next(item for item in raw if isinstance(item, TextContent))
    return json.loads(text.text), [item for item in raw if item is not text]


def _cost_estimate_from_dict(cost: dict[str, Any] | None) -> Any:
    """Rebuild a pricing CostEstimate from a response's cost dict, or None."""
    if not cost:
        return None
    from .pricing import CostEstimate

    return CostEstimate(
        usd=float(cost["usd"]),
        is_estimate=bool(cost.get("is_estimate", True)),
        unit=str(cost.get("unit", "mixed")),
        detail=str(cost.get("detail", "")),
        breakdown={k: float(v) for k, v in (cost.get("breakdown") or {}).items()},
        source=cost.get("pricing_source"),
        source_note=cost.get("pricing_source_note"),
        # Carried back, or a total built from rebuilt costs dated every
        # component by the full re-read: a directed 1.2 render whose director
        # calls report no usage was dated before the 1.2 table was verified.
        as_of=cost.get("pricing_as_of"),
    )


def _check_payload(check: Any) -> dict[str, Any]:
    """One derived check as it appears in a response and a manifest."""
    payload: dict[str, Any] = {
        "id": check.id,
        "check": check.check,
        "fix": check.fix,
        "from_notes": check.from_notes,
        "kind": check.kind,
    }
    if check.literal is not None:
        payload["literal"] = check.literal
    if check.target is not None:
        payload["target"] = check.target
    return payload


async def _wait(seconds: float) -> None:
    """Sleep before a render is tried again. Its own name, so a test can
    record the wait instead of taking it."""
    await asyncio.sleep(seconds)


def _seconds_value(seconds: float) -> float | int:
    return int(seconds) if seconds == int(seconds) else round(seconds, 1)


def _render_retry_reason(payload: dict[str, Any]) -> str | None:
    """Why a failed render is worth one more request, or None.

    A 429 is refused before any work, so a second request risks only the
    wait, unless the tool already sent it again (``quota_retries``). A server error, or a timeout that may or may not have reached
    Google ("unknown"), is tried again too, with the first attempt kept as a
    possible charge. A render that was submitted is never rendered again: it
    exists or will, and the take keeps the handle to it.
    """
    if "error" not in payload or payload.get("submission") == "submitted":
        return None
    status = payload.get("status_code")
    if status == 429:
        # An Omni tool resends a 429 itself and reports how often; its error
        # is the service still refusing after those waits, so another round
        # here would only stack more waits on them.
        return None if payload.get("quota_retries") else "quota"
    if (isinstance(status, int) and status >= 500) or payload.get(
        "submission"
    ) == "unknown":
        return "transient"
    return None


def _check_result_payload(result: Any) -> dict[str, Any]:
    """One check's result as it appears in a response."""
    payload: dict[str, Any] = {"id": result.id, "state": result.state}
    if result.observation:
        payload["observation"] = result.observation
    evidence = []
    for item in result.evidence:
        entry: dict[str, Any] = {}
        if item.where:
            entry["where"] = item.where
        if item.start_s is not None:
            entry["start_seconds"] = round(item.start_s, 3)
            entry["end_seconds"] = round(item.end_s or item.start_s, 3)
        if entry:
            evidence.append(entry)
    if evidence:
        payload["evidence"] = evidence
    if not result.reported:
        payload["reported"] = False
    if result.note:
        payload["note"] = result.note
    return payload


async def _review_take(
    app_ctx: AppContext,
    *,
    client: genai.Client,
    model: str,
    checks: list[Any],
    payload: dict[str, Any],
    take_prompt: str,
    media_kind: MediaKind,
    references: list[tuple[bytes, str, Any]],
    audio_requested: bool | None,
    deadline: float | None = None,
) -> tuple[Any, Any, list[Any]]:
    """Review one take. Returns the TakeReview, its cost, and the references
    left out to fit the inline request.

    ``references`` are (bytes, MIME type, ReviewReference) already prepared
    for review. The TakeReview lists the ones a sent request carried.
    ``deadline`` is the directed call's end on the monotonic clock. The review's
    timeout is taken from it after the media is read and prepared, so that
    time is spent from the same budget, and a review with too little left is
    not sent.

    Never raises for a review that cannot run: an unreadable render, a
    provider refusal or a malformed answer is an abstention with a reason, and
    the take is still delivered.
    """
    from . import director
    from .pricing import actual_understanding_cost

    media_url = str(payload.get("image_url") or payload.get("video_url") or "")
    duration = payload.get("duration_seconds")
    duration_s = float(duration) if isinstance(duration, (int, float)) else None
    is_vertex = getattr(client._api_client, "vertexai", False)

    async def read_subject() -> tuple[Any, bytes] | str:
        """The render as a request part and its bytes, or why it can't be."""
        nonlocal duration_s
        if media_kind == "video" and media_url.startswith("gs://") and is_vertex:
            part = director.video_part(
                "video/mp4", uri=media_url, duration_s=duration_s
            )
            return part, media_url.encode()
        # A clip goes inline as it is, so it is capped here; an image is
        # scaled down below, so it is fetched under the ordinary cap.
        data = await fetch(
            media_url,
            allowed_dir=app_ctx.data_folder,
            allowed_gcs_buckets=app_ctx.allowed_gcs_buckets,
            max_bytes=(
                director.MAX_INLINE_MEDIA_BYTES
                if media_kind == "video"
                else MAX_FETCH_BYTES
            ),
        )
        if data is None:
            return (
                "The render could not be read inline: it is larger than "
                f"{director.MAX_INLINE_MEDIA_BYTES // (1024 * 1024)} MB or "
                "not readable from here."
            )
        if media_kind == "video":
            if duration_s is None:
                # An omni edit leaves the length to the service and reports
                # none. Without it a per-second answer shorter than the clip
                # could pass, and evidence past its end would go unchecked.
                duration_s = await _probe_media(measure_video_duration_bytes, data)
            return director.video_part(
                "video/mp4", data=data, duration_s=duration_s
            ), data
        try:
            data, mime = await asyncio.to_thread(director.prepare_review_image, data)
        except Exception as exc:  # noqa: BLE001 - an unreadable render abstains
            return str(exc)
        return director.image_part(data, mime), data

    def skipped(
        reason: str, detail: str, attempts: int = 1
    ) -> tuple[Any, Any, list[Any]]:
        return (
            director.TakeReview(
                status="abstained", reason=reason, detail=detail, attempts=attempts
            ),
            None,
            [],
        )

    # Reading the render is bounded by the same deadline as the review: it
    # must leave a review's worth of time, or the review could not run and
    # the host could cancel the call before the paid render is delivered.
    read_budget = (
        None
        if deadline is None
        else deadline - time.monotonic() - _DIRECTOR_MIN_REVIEW_SECONDS[media_kind]
    )
    if read_budget is not None and read_budget <= 0:
        return skipped(
            "deadline",
            "Too little of timeout_seconds remained to read the render and review it.",
            attempts=0,
        )
    try:
        read = (
            await read_subject()
            if read_budget is None
            else await asyncio.wait_for(read_subject(), read_budget)
        )
    except TimeoutError:
        return skipped(
            "deadline",
            "Reading the render did not finish in the time timeout_seconds left "
            "for a review.",
            attempts=0,
        )
    if isinstance(read, str):
        return skipped("media_unavailable", read)
    subject, subject_bytes = read

    # One inline request carries the render and the references together, so
    # the references are trimmed to what fits beside the render.
    budget = director.MAX_INLINE_MEDIA_BYTES - len(subject_bytes)
    kept: list[tuple[bytes, str, Any]] = []
    for ref in references:
        if len(ref[0]) > budget:
            break
        kept.append(ref)
        budget -= len(ref[0])
    shown = tuple(ref for _, _, ref in kept)
    trimmed = [ref for _, _, ref in references[len(kept) :]]
    parts: list[Any] = []
    for ref_bytes, ref_mime, ref in kept:
        parts += [
            director.media_label(ref.key),
            director.image_part(ref_bytes, ref_mime),
        ]
    parts += [director.media_label(director.SUBJECT_KEY), subject]
    system, user = director.build_review_request(
        checks, take_prompt, media_kind, shown, audio_requested, duration_s
    )
    review_timeout = (
        director.IMAGE_REVIEW_TIMEOUT_SECONDS
        if media_kind == "image"
        else director.VIDEO_REVIEW_TIMEOUT_SECONDS
    )
    if deadline is not None:
        left = deadline - time.monotonic()
        if left < _DIRECTOR_MIN_REVIEW_SECONDS[media_kind]:
            return (
                director.TakeReview(
                    status="abstained",
                    reason="deadline",
                    detail=(
                        f"{max(left, 0):.0f}s of timeout_seconds remained once the "
                        "render was read, too little for a review."
                    ),
                    attempts=0,
                ),
                None,
                [],
            )
        review_timeout = min(review_timeout, left - 5.0)
    parts.append(types.Part(text=user))
    outcome = await director.call_model(
        client,
        model,
        system=system,
        parts=parts,
        schema=director.REVIEW_SCHEMA,
        thinking_level=director.REVIEW_THINKING_LEVEL,
        max_output_tokens=director.review_max_output_tokens(media_kind),
        # Seeded from the render's address and size, not its bytes: hashing
        # 20 MB on the event loop buys nothing a seed needs.
        seed=director.request_seed(
            model, system, user, media_url, str(len(subject_bytes))
        ),
        timeout_s=review_timeout,
        media_resolution=(
            types.MediaResolution.MEDIA_RESOLUTION_HIGH
            if media_kind == "image"
            else types.MediaResolution.MEDIA_RESOLUTION_MEDIUM
        ),
    )
    cost = actual_understanding_cost(model, outcome.usage, label="review call")
    if outcome.status != "ok":
        return (
            director.TakeReview(
                status="abstained",
                reason=outcome.reason,
                detail=outcome.detail,
                attempts=outcome.attempts,
                unreported=outcome.unreported,
                retry_skipped=outcome.retry_skipped,
                retried=outcome.retried,
                shown=shown,
            ),
            cost,
            trimmed,
        )
    try:
        results = director.parse_review(
            outcome.text or "", checks, media_kind, duration_s
        )
    except Exception as exc:  # noqa: BLE001 - any unusable answer abstains
        return (
            director.TakeReview(
                status="abstained",
                reason="parse_failed",
                detail=str(exc),
                attempts=outcome.attempts,
                unreported=outcome.unreported,
                retry_skipped=outcome.retry_skipped,
                retried=outcome.retried,
                shown=shown,
            ),
            cost,
            trimmed,
        )
    return (
        director.TakeReview(
            status="reviewed",
            results=results,
            attempts=outcome.attempts,
            unreported=outcome.unreported,
            retry_skipped=outcome.retry_skipped,
            retried=outcome.retried,
            shown=shown,
        ),
        cost,
        trimmed,
    )


def _record_take_review(
    payload: dict[str, Any], take_summary: dict[str, Any], context: dict[str, Any]
) -> bool:
    """Add a take's review to its manifest, so the sidecar carries it too.

    ``context`` is what the review is judged against: the director model, the
    caller's notes, the checks derived from them and any notes skipped. The
    response states it once for every take; a manifest is opened on its own,
    long after the call, so each one carries it, or "n1: pass" means nothing.
    """
    # The manifest already holds the prompt this take rendered; a second copy
    # doubles the file for a long prompt.
    take_summary = {
        **context,
        **{k: v for k, v in take_summary.items() if k != "prompt"},
    }
    sidecar_url = str(payload.get("sidecar_url") or "")
    if sidecar_url.startswith("file://"):
        sidecar = Path(sidecar_url[len("file://") :])
        try:
            manifest = json.loads(sidecar.read_text())
            manifest["director_review"] = take_summary
            text = json.dumps(manifest, indent=2, sort_keys=True)
            # The interaction lookup reads sidecars up to _SIDECAR_MAX_BYTES and
            # skips larger ones, which would lose the backend a continuation
            # must run on. Routing outranks the review.
            if len(text.encode()) > _SIDECAR_MAX_BYTES:
                return False
            sidecar.write_text(text)
        except (OSError, ValueError) as exc:
            logger.warning("Could not add the review to %s: %s", sidecar, exc)
            return False
    elif isinstance(payload.get("manifest"), dict):
        payload["manifest"]["director_review"] = take_summary
    return True


async def _directed(ctx: Context[ServerSession, AppContext], **kwargs: Any) -> Any:
    """Run a generation tool under the virtual director; see _directed_unguarded.

    The director hook sits before each tool's own try/except, so this is the
    boundary that turns any failure inside the director into the
    ``{"error": ...}`` body every tool otherwise returns, instead of an
    unhandled exception the host reports as a bare tool failure.
    """
    try:
        return await _directed_unguarded(ctx, **kwargs)
    except Exception as exc:  # noqa: BLE001 - the tool boundary
        await ctx.error(f"Directed render failed: {exc}")
        logger.exception("Director error")
        body = json.dumps({"error": str(exc)})
        return (
            [TextContent(type="text", text=body)]
            if kwargs.get("returns_content")
            else body
        )


async def _directed_unguarded(
    ctx: Context[ServerSession, AppContext],
    *,
    render: Any,
    returns_content: bool,
    media_kind: MediaKind,
    prompt: str,
    notes: str,
    max_retakes: int,
    dry_run: bool,
    references: list[tuple[Any, str, str]] | None = None,
    unshown_inputs: list[str] | None = None,
    duration_s: float | None = None,
    audio_requested: bool | None = None,
    deadline_s: float | None = None,
) -> Any:
    """Run a generation tool under the virtual director.

    ``render(prompt, dry_run, timeout)`` calls the tool itself without notes
    and returns what the tool returns; ``timeout`` is the time left on
    ``deadline_s``, or None. A dry run quotes the worst case: every take
    rendered and reviewed, at the output cap, with no cache hits. A real run
    derives the checks, then renders and reviews take by take until a take
    passes, a review cannot decide, or the retakes run out, and returns the
    best take with every take's result and cost.

    ``references`` are the images the render was given, from
    _review_references. The reviewer sees every one the render is anchored
    on and the first _DIRECTOR_MAX_REFERENCES of each list.
    ``unshown_inputs`` names inputs the reviewer cannot see at all, such as
    reference clips. Every image left out is named in a warning.

    ``deadline_s`` bounds the whole call, renders included. MCP hosts cancel a
    tool call after a few minutes, and a cancelled call has still billed its
    renders, so the director skips a review or a retake that could not finish
    in the time left, and says so, rather than letting the host cut it off.
    """
    from . import director
    from .pricing import (
        actual_understanding_cost,
        cost_to_dict,
        estimate_understanding_cost,
        sum_costs,
    )

    app_ctx = ctx.request_context.lifespan_context
    warnings: list[str] = []
    started = time.monotonic()

    def _remaining() -> float | None:
        if deadline_s is None:
            return None
        return deadline_s - (time.monotonic() - started)

    def _capped(limit: float) -> float:
        """``limit``, or less when the caller's deadline is nearer."""
        remaining = _remaining()
        return limit if remaining is None else max(1.0, min(limit, remaining))

    def _error(message: str) -> Any:
        body = json.dumps({"error": message})
        return [TextContent(type="text", text=body)] if returns_content else body

    async def _finish(payload: dict[str, Any], others: list[Any]) -> Any:
        if warnings:
            payload["warnings"] = [*(payload.get("warnings") or []), *warnings]
            await _emit_warnings(ctx, warnings)
        text = _respond(app_ctx, payload)
        return (
            [*others, TextContent(type="text", text=text)] if returns_content else text
        )

    try:
        notes = director.validate_director_request(notes, max_retakes)
        model = director.director_model()
    except ValueError as exc:
        return _error(str(exc))

    supplied = references or []
    over_limit = [
        ref
        for ref, _, _ in supplied
        if ref.kind in director.LISTED_KINDS and ref.index > _DIRECTOR_MAX_REFERENCES
    ]
    references_s = [item for item in supplied if item[0] not in over_limit]
    if over_limit:
        warnings.append(
            f"The reviewer is shown at most {_DIRECTOR_MAX_REFERENCES} images from "
            f"each list, so it was not shown {_named(over_limit)}; a check that "
            "needs one is answered unsure."
        )
    for unshown in unshown_inputs or []:
        warnings.append(f"The reviewer is not shown {unshown}.")
    takes_max = 1 + max_retakes

    if dry_run:
        raw = await render(prompt, True)
        payload, others = _tool_payload(raw)
        if "error" in payload:
            return raw
        per_take = _cost_estimate_from_dict(payload.get("estimated_cost"))
        # The tool's own quote resolves the length that will render (snapped,
        # forced to 8 s for references, extended, or an edit's upper bound),
        # and the review watches that clip, not the one requested.
        quoted_length = payload.get("duration_seconds")
        review_seconds = (
            float(quoted_length)
            if isinstance(quoted_length, (int, float))
            and not isinstance(quoted_length, bool)
            else duration_s
        )
        # A retake prompt carries up to MAX_CHECKS fix notes on top of the
        # caller's, so the review is quoted with the longest prompt it can see.
        longest_prompt = prompt + "x" * (
            director.MAX_CHECKS * (director.MAX_CHECK_CHARS + 20)
        )
        derive_cost = estimate_understanding_cost(
            model,
            director.derive_input_tokens(notes, prompt),
            director.DERIVE_MAX_OUTPUT_TOKENS,
            label="derive call",
        )
        review_cost = estimate_understanding_cost(
            model,
            director.review_input_tokens(
                longest_prompt,
                media_kind,
                duration_s=review_seconds,
                reference_count=len(references_s),
            ),
            director.review_max_output_tokens(media_kind),
            calls=takes_max,
            label="review call",
        )
        director_cost = sum_costs([derive_cost, review_cost], label="director")
        payload["estimated_cost"] = cost_to_dict(
            sum_costs([per_take] * takes_max + [director_cost], label="directed render")
        )
        payload["director"] = {
            "model": model,
            "max_takes": takes_max,
            "render_estimated_cost_per_take": cost_to_dict(per_take),
            "director_estimated_cost": cost_to_dict(director_cost),
            "how_to_read": (
                "estimated_cost prices the worst case: every take rendered and "
                "reviewed, each director call at its output cap, no cache hits. "
                "The call stops early as soon as a take passes. Three charges "
                "cannot be priced in advance and are not included: an image "
                "render's input tokens (generate_image quotes output only), a "
                "director request that fails without reporting its usage, and "
                "a render that fails with a server error or a timeout and is "
                "tried again, which may bill for the failed attempt. A warning "
                "names either of the last two if it happens."
            ),
        }
        return await _finish(payload, others)

    # Validate the render before paying for any director call: the tool's
    # own dry run applies every check the render would, and costs nothing.
    preflight = await render(prompt, True)
    if "error" in _tool_payload(preflight)[0]:
        return preflight

    client = _director_client(app_ctx)
    backend = (
        "vertex" if getattr(client._api_client, "vertexai", False) else "gemini_api"
    )
    director_costs: list[Any] = []

    prepared_references: list[tuple[bytes, str, Any]] = []
    unreadable: list[Any] = []

    async def read_references() -> None:
        for ref, kind, value in references_s:
            data = None
            try:
                data = (
                    _decode_base64_capped(value)
                    if kind == "base64"
                    else await fetch(
                        value,
                        allowed_dir=app_ctx.data_folder,
                        allowed_gcs_buckets=app_ctx.allowed_gcs_buckets,
                    )
                )
                if data is not None:
                    ref_bytes, ref_mime = await asyncio.to_thread(
                        director.prepare_review_image, data
                    )
                    prepared_references.append((ref_bytes, ref_mime, ref))
                    continue
            except Exception:  # noqa: BLE001 - an unreadable reference is left out
                pass
            unreadable.append(ref)

    # Read before the first render, so with a deadline it is bounded too: a
    # slow reference must not spend the time the renders need.
    try:
        await asyncio.wait_for(
            read_references(), _capped(_DIRECTOR_REFERENCE_READ_SECONDS)
        )
    except TimeoutError:
        done = {*unreadable, *(ref for _, _, ref in prepared_references)}
        warnings.append(
            "Reading the reference images did not finish in time, so the "
            "reviewer was not shown "
            + _named([ref for ref, _, _ in references_s if ref not in done])
            + "."
        )
    if unreadable:
        warnings.append(
            f"The reviewer was not shown {_named(unreadable)}: it could not be "
            "read as an image."
        )

    system, user = director.build_derive_request(notes, prompt, media_kind)
    checks: list[Any] = []
    skipped: list[Any] = []
    abstained: tuple[str | None, str | None] | None = None
    # The same notes, prompt and model give the same checks, so a request
    # this process has already paid for is not sent again.
    cache_key = director.derive_cache_key(model, system, user)
    cached = director.cached_derivation(cache_key)
    derive_attempts = 0
    derive_retried: list[str] = []
    derive_retry_skipped: str | None = None

    async def derive(timeout_s: float) -> Any:
        outcome = await director.call_model(
            client,
            model,
            system=system,
            parts=[types.Part(text=user)],
            schema=director.DERIVE_SCHEMA,
            thinking_level=director.DERIVE_THINKING_LEVEL,
            max_output_tokens=director.DERIVE_MAX_OUTPUT_TOKENS,
            seed=director.request_seed(model, system, user),
            timeout_s=timeout_s,
        )
        director_costs.append(
            actual_understanding_cost(model, outcome.usage, label="derive call")
        )
        if outcome.unreported and _DIRECTOR_UNREPORTED_WARNING not in warnings:
            warnings.append(_DIRECTOR_UNREPORTED_WARNING)
        return outcome

    if cached is not None:
        await ctx.info("Director: reusing the checks derived earlier for these notes")
        checks, skipped = cached
    else:
        await ctx.info(f"Director: turning the notes into checks with {model}")
        derived = await derive(_capped(director.DERIVE_TIMEOUT_SECONDS))
        derive_attempts = derived.attempts
        derive_retried = list(derived.retried)
        derive_retry_skipped = derived.retry_skipped
        if (
            derived.status == "error"
            and derived.reason == "timeout"
            and derived.attempts == 1
        ):
            # A timed-out request has spent its own budget, so the retry is a
            # second call with a fresh one. It is the cheapest director call
            # and comes before any render, so it is worth one more try; on the
            # video tools it may not take so long that less than half of
            # timeout_seconds is left for the render.
            delay = director.RETRY_DELAY_SECONDS["transient"]
            retry_timeout = director.DERIVE_TIMEOUT_SECONDS
            remaining = _remaining()
            if remaining is not None and deadline_s is not None:
                retry_timeout = min(retry_timeout, remaining - delay - deadline_s / 2)
            if retry_timeout < director.MIN_RETRY_SECONDS:
                derive_retry_skipped = (
                    f"{max(remaining or 0.0, 0.0):.0f}s of timeout_seconds remained; "
                    "a retry would have left less than half of it for the render."
                )
            else:
                await ctx.info(
                    "Director: turning the notes into checks timed out; trying "
                    f"once more after {delay:g}s"
                )
                await asyncio.sleep(delay)
                derive_retried.append("timeout")
                derived = await derive(retry_timeout)
                derive_attempts += derived.attempts
                derive_retried += list(derived.retried)
                derive_retry_skipped = derived.retry_skipped
        if derived.status != "ok":
            abstained = (derived.reason, derived.detail)
        else:
            try:
                checks, skipped = director.parse_derived(
                    derived.text or "", notes, media_kind
                )
                director.remember_derivation(cache_key, checks, skipped)
            except ValueError as exc:
                abstained = ("parse_failed", str(exc))

    block: dict[str, Any] = {
        "model": model,
        "backend": backend,
        "checks": [_check_payload(c) for c in checks],
    }
    # "compiled" only for a derive call that answered, even with no checks
    # (every note skipped); a call that failed compiled nothing.
    block["checks_source"] = (
        "cache" if cached is not None else "failed" if abstained else "compiled"
    )
    # Always present, as review_attempts is: 0 when the checks came from the
    # cache, 2 when the derive call was retried.
    block["derive_attempts"] = derive_attempts
    if derive_retried:
        block["derive_retried_after"] = derive_retried
    if derive_retry_skipped:
        block["derive_retry_skipped"] = derive_retry_skipped
    # Always a list, so a reader need not tell "none" from "not reported".
    block["skipped_notes"] = [{"text": s.text, "reason": s.reason} for s in skipped]

    if not checks:
        # Nothing to check against: render once, as the tool would have, and
        # say why the director stood down. The render is never withheld.
        raw = await render(prompt, False, _remaining())
        payload, others = _tool_payload(raw)
        if abstained:
            block.update(status="abstained", reason=abstained[0])
            if abstained[1]:
                block["detail"] = abstained[1]
        else:
            block["status"] = "no_checks"
        block["cost"] = cost_to_dict(_director_subtotal(director_costs))
        payload["director"] = block
        _add_total_cost(payload, [payload.get("cost")], director_costs)
        return await _finish(payload, others)

    call_deadline = None if deadline_s is None else started + deadline_s
    review_context: dict[str, Any] = {
        "model": model,
        "director_notes": notes,
        "checks": block["checks"],
        "checks_source": block["checks_source"],
        "skipped_notes": block["skipped_notes"],
    }
    takes: list[dict[str, Any]] = []
    reviews: list[Any] = []
    render_costs: list[Any] = []
    # One entry per attempted take, rendered or not, in order.
    attempted: list[dict[str, Any]] = []
    take_prompt = prompt
    # How long the last render took: what a retried render is assumed to need.
    last_render_seconds = 0.0
    for attempt in range(takes_max):
        await ctx.info(f"Director: rendering take {attempt + 1} of at most {takes_max}")
        take_started = time.monotonic()
        render_started = take_started
        raw = await render(take_prompt, False, _remaining())
        payload, others = _tool_payload(raw)
        render_attempts = 1
        render_retried: list[str] = []
        render_waits: list[float] = []
        render_retry_skipped: str | None = None
        while (retry_reason := _render_retry_reason(payload)) and len(
            render_retried
        ) < len(director.RENDER_RETRY_WAITS):
            # Further requests, each only when the time left fits the wait,
            # a render like the last one and a review. Image quotas recover
            # over tens of seconds, so the waits grow, and a 429 that says
            # how long to wait is believed, within that budget.
            after_wait = last_render_seconds + _DIRECTOR_MIN_REVIEW_SECONDS[media_kind]
            remaining = _remaining()
            wait = director.RENDER_RETRY_WAITS[len(render_retried)]
            asked = payload.get("retry_after_seconds")
            if isinstance(asked, (int, float)) and not isinstance(asked, bool):
                wait = (
                    float(asked)
                    if remaining is None
                    else min(float(asked), max(0.0, remaining - after_wait))
                )
            if remaining is not None and remaining < wait + after_wait:
                render_retry_skipped = (
                    f"{max(remaining, 0):.0f}s of timeout_seconds remained; a "
                    f"retry needs {wait + after_wait:.0f}s for the wait, a render "
                    "and a review."
                )
                break
            if retry_reason == "transient":
                # Google may have taken the failed request; nothing says
                # whether it billed, so the total is a lower bound.
                render_costs.append(None)
                if not render_retried or "transient" not in render_retried:
                    warnings.append(
                        f"Take {attempt + 1}'s render failed with a server error "
                        "or a timeout and was tried again. If a failed request "
                        "reached Google it may still bill, and that charge is "
                        "not in cost."
                    )
            await ctx.info(
                f"Director: take {attempt + 1} was not rendered ({retry_reason}); "
                f"trying again after {wait:g}s"
            )
            await _wait(wait)
            render_waits.append(wait)
            render_retried.append(retry_reason)
            render_attempts += 1
            render_started = time.monotonic()
            raw = await render(take_prompt, False, _remaining())
            payload, others = _tool_payload(raw)
        media_url = payload.get("image_url") or payload.get("video_url")
        if "error" in payload or not media_url:
            # Listed like any take, so its prompt (a retake's fix notes
            # included) can be read even though nothing rendered.
            not_rendered: dict[str, Any] = {
                "take": attempt + 1,
                "prompt": take_prompt,
                "verdict": "not_rendered",
                "error": payload.get("error") or "The model returned no media.",
                "render_attempts": render_attempts,
            }
            if render_retried:
                not_rendered["render_retried_after"] = render_retried
                not_rendered["render_retry_waits"] = [
                    _seconds_value(w) for w in render_waits
                ]
            if render_retry_skipped:
                not_rendered["reason"] = "budget"
                not_rendered["render_retry_skipped"] = render_retry_skipped
            # What the failed take may still bill, by how far it got. The tool
            # says so explicitly in "submission"; an absent operation name is
            # not enough to tell a local failure from a submitted render.
            submission = payload.get("submission")
            billed: Any = None
            if submission:
                not_rendered["submission"] = submission
            # The handle a caller needs to find, resume or reconcile the
            # render this take may still be billed for.
            for handle in ("operation_name", "interaction_id"):
                if payload.get(handle):
                    not_rendered[handle] = payload[handle]
            if submission == "submitted":
                # Veo quotes what it may bill as attempted_cost; omni does not,
                # and a missing figure makes the total a lower bound.
                billed = payload.get("attempted_cost")
                render_costs.append(billed)
            elif submission == "unknown":
                render_costs.append(None)
            elif "error" not in payload and payload.get("cost"):
                # A billed answer with no media, such as an image model's
                # text-only refusal.
                billed = payload["cost"]
                render_costs.append(billed)
            if billed:
                not_rendered["render_cost"] = billed
            attempted.append(not_rendered)
            if not takes:
                # The first take produced nothing to review: hand back exactly
                # what the tool returned, with what the director spent.
                block.update(status="abstained", reason="no_media")
                block["retakes_used"] = 0
                block["takes"] = attempted
                block["cost"] = cost_to_dict(_director_subtotal(director_costs))
                payload["director"] = block
                if "error" not in payload:
                    _add_total_cost(payload, [payload.get("cost")], director_costs)
                return await _finish(payload, others)
            warnings.append(
                f"Take {attempt + 1} did not render, so the director kept the best "
                f"earlier take: {not_rendered['error']}"
            )
            if submission == "submitted":
                warnings.append(
                    f"Take {attempt + 1} was submitted before it failed and may "
                    "still complete and bill; "
                    + (
                        "its quoted cost is included in cost."
                        if billed
                        else "its cost is unknown, so cost is a lower bound."
                    )
                )
            elif submission == "unknown":
                warnings.append(
                    f"Take {attempt + 1} may have reached Google before it failed; "
                    "if it did, it may still bill, so cost is a lower bound."
                )
            break
        last_render_seconds = time.monotonic() - render_started
        render_costs.append(payload.get("cost"))
        remaining = _remaining()
        if (
            remaining is not None
            and remaining < _DIRECTOR_MIN_REVIEW_SECONDS[media_kind]
        ):
            review, review_cost, trimmed = (
                director.TakeReview(
                    status="abstained",
                    reason="deadline",
                    detail=(
                        f"{max(remaining, 0):.0f}s of timeout_seconds remained "
                        "after the render, too little for a review."
                    ),
                    attempts=0,
                ),
                None,
                [],
            )
        else:
            await ctx.info(f"Director: reviewing take {attempt + 1}")
            try:
                review, review_cost, trimmed = await _review_take(
                    app_ctx,
                    client=client,
                    model=model,
                    checks=checks,
                    payload=payload,
                    take_prompt=take_prompt,
                    media_kind=media_kind,
                    references=prepared_references,
                    audio_requested=audio_requested,
                    deadline=call_deadline,
                )
            except Exception as exc:  # noqa: BLE001 - the take is paid for; keep it
                logger.exception("Director review failed")
                review, review_cost, trimmed = (
                    director.TakeReview(
                        status="abstained", reason="review_error", detail=str(exc)
                    ),
                    None,
                    [],
                )
        director_costs.append(review_cost)
        if review.unreported and _DIRECTOR_UNREPORTED_WARNING not in warnings:
            warnings.append(_DIRECTOR_UNREPORTED_WARNING)
        if trimmed:
            warnings.append(
                f"Take {attempt + 1}: the reviewer was not shown {_named(trimmed)}, "
                "so the render fit the inline request."
            )
        summary: dict[str, Any] = {
            "take": attempt + 1,
            "prompt": take_prompt,
            "media_url": media_url,
            "render_cost": payload.get("cost"),
            "review_cost": cost_to_dict(review_cost),
        }
        # The handle a caller needs to continue or reconcile this take, which
        # the top level carries only for the take returned.
        for handle in ("operation_name", "interaction_id"):
            if payload.get(handle):
                summary[handle] = payload[handle]
        summary["render_attempts"] = render_attempts
        if render_retried:
            summary["render_retried_after"] = render_retried
            summary["render_retry_waits"] = [_seconds_value(w) for w in render_waits]
        if review.status == "reviewed":
            summary["verdict"] = review.verdict
            summary["results"] = [_check_result_payload(r) for r in review.results]
        else:
            summary["verdict"] = "abstained"
            summary["reason"] = review.reason
            if review.detail:
                summary["detail"] = review.detail
        # Always present, so a single attempt reads as one rather than as
        # missing: 0 when no review was sent, 2 when one was retried.
        summary["review_attempts"] = review.attempts
        if review.retried:
            # Why each earlier request failed: quota (a 429, not billed) or
            # transient (a 5xx, which may have billed; see warnings).
            summary["retried_after"] = list(review.retried)
        if review.retry_skipped:
            summary["retry_skipped"] = review.retry_skipped
        if review.shown:
            summary["references_shown"] = [ref.label for ref in review.shown]
        if not _record_take_review(payload, summary, review_context):
            warnings.append(
                f"Take {attempt + 1}'s review was not added to its manifest: the "
                "file would have outgrown the size the interaction lookup reads. "
                "It is in this response."
            )
        takes.append({"payload": payload, "others": others, "summary": summary})
        attempted.append(summary)
        reviews.append(review)
        if not director.should_retake(review, takes_max - attempt - 1):
            break
        # A retake takes about as long as the take before it.
        take_seconds = time.monotonic() - take_started
        remaining = _remaining()
        if remaining is not None and remaining < take_seconds:
            warnings.append(
                f"Stopped before take {attempt + 2}: take {attempt + 1} took "
                f"{take_seconds:.0f}s and {max(remaining, 0):.0f}s of "
                "timeout_seconds remained. Raise timeout_seconds (and your MCP "
                "host's limit) to leave room for retakes."
            )
            break
        # Every correction so far, not only the latest: a retake restarts from
        # the caller's prompt, so a note dropped is a correction undone.
        take_prompt = director.retake_prompt(
            prompt,
            director.retake_notes(
                checks, [r.results for r in reviews if r.status == "reviewed"]
            ),
        )

    chosen = director.choose_take(reviews)
    final = takes[chosen]
    payload, others = final["payload"], final["others"]
    chosen_review = reviews[chosen]
    block["status"] = (
        chosen_review.verdict if chosen_review.status == "reviewed" else "abstained"
    )
    if chosen_review.status != "reviewed":
        block["reason"] = chosen_review.reason
    block["chosen_take"] = final["summary"]["take"]
    # Every retake the director started, rendered or not: a retake whose
    # render failed was still spent.
    block["retakes_used"] = len(attempted) - 1
    block["takes"] = attempted
    block["cost"] = cost_to_dict(_director_subtotal(director_costs))
    block["how_to_read"] = (
        "Each check is pass, fail, unsure or not_applicable from one review of "
        "that take. No confidence number is reported: a model's own certainty "
        "is not calibrated, and these states have not yet been compared with "
        "human labels."
    )
    payload["director"] = block
    if block["status"] == "failed":
        # "Video generated successfully" sat beside director.status "failed",
        # which a caller reading the message alone took for a pass.
        subject = str(payload.get("message") or "Media generated").split(" generated")[
            0
        ]
        payload["message"] = (
            f"{subject} generated, but the returned take failed the director's "
            "review; director.takes has each check's result."
        )
    _report_prompts(payload, requested=prompt, rendered=final["summary"]["prompt"])
    if len(takes) > 1 and chosen != len(takes) - 1:
        warnings.append(
            f"Returned take {chosen + 1}, which the review rated best; the later "
            "takes were also rendered and billed, and are listed under "
            "director.takes."
        )
    _add_total_cost(payload, render_costs, director_costs)
    return await _finish(payload, others)


def _report_prompts(payload: dict[str, Any], *, requested: str, rendered: str) -> None:
    """Keep ``prompt`` the caller's, and say what rendered when it differs.

    The returned take may be a retake, rendered from the caller's prompt plus
    fix notes. ``prompt`` stays what the caller sent and ``effective_prompt``
    carries the text that rendered, the convention generate_video_omni already
    follows. Omni may report an effective_prompt of its own, built from this
    take's prompt plus its generated role declarations; that is the fuller
    record of what was sent, so it is kept.
    """
    payload["prompt"] = requested
    if rendered != requested and "effective_prompt" not in payload:
        payload["effective_prompt"] = rendered


def _director_subtotal(director_costs: list[Any]) -> Any:
    """What the director's own calls billed, leaving out calls with no usage."""
    from .pricing import sum_costs

    return sum_costs([c for c in director_costs if c is not None], label="director")


def _add_total_cost(
    payload: dict[str, Any], render_costs: list[Any], director_costs: list[Any]
) -> None:
    """Make ``cost`` what the whole directed call billed.

    Every take's render plus every director call. Each take's own render cost
    stays on its entry in director.takes and in its manifest.
    """
    from .pricing import cost_to_dict, sum_costs

    # A director call with no reported usage (it failed before Google
    # answered) is left out rather than counted as unpriced, which would mark
    # the whole total as a lower bound.
    estimates = [_cost_estimate_from_dict(c) for c in render_costs] + [
        c for c in director_costs if c is not None
    ]
    if all(e is None for e in estimates):
        return
    payload["cost"] = cost_to_dict(sum_costs(estimates, label="directed render"))


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
    director_notes: str | None = None,
    max_retakes: int = DEFAULT_MAX_RETAKES,
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
        director_notes: Optional. What the image must get right, in plain
            words ("the label reads GOLDEN HOUR in serif capitals; no people in the shot"). Gemini 4 Argon turns the notes into checks,
            reviews the image against them, and when a check fails
            renders again with fix notes, up to max_retakes times. The response
            then carries a `director` block with each check's state on every
            take, and `cost` covers every take and every director call. Omit it
            and the tool behaves exactly as it does without a director: no
            extra call and no extra cost.
            In the notes, call the image being edited "the source image" and
            reference_image_uris[0] "reference 1", [1] "reference 2" and so on;
            the reviewer is shown the source and the first three references.
            Put text that must appear in quotes, exactly as it should be set
            ('the headline reads "Slow mornings, good coffee."'): it is
            compared character for character, case and punctuation included.
        max_retakes: Re-renders the director may spend when a check fails, 0
            to 2 (default 1). A dry run with director_notes quotes the worst
            case: every take rendered and reviewed. A retake the service turns
            away with a 429 (image quotas can allow only a few renders a
            minute) or a server error is tried again after 10 s, then 30 s,
            or the wait the 429 asked for, while the call's time allows.
        dry_run: When True, generate nothing and return only the cost estimate
            and the resolved model/parameters. Free and instant — use it to
            price a call before committing to it. A real run reports the
            actual cost, derived from the token counts the API metered.
            Note: a quote covers output tokens only. Input tokens are not
            knowable before the call, so a multi-turn edit (which resends the
            conversation via thought_signature_url) costs more than quoted.
            How much more is not stated here: an earlier note put it near
            one percent, and that cannot be squared with what the signatures
            were later measured to weigh (1.6-2.8 MB), so the figure is
            withdrawn rather than repeated. The real run reports the metered
            input and output counts.

    Returns:
        Two content blocks: a JPEG preview of the image, then JSON with
        message, image_url, prompt and model. The preview is that separate
        image block — there is no image_preview key in the JSON.

        The JSON also carries, when they apply: thought_signature_url (Gemini
        3.x, pass it back to keep editing), cost (metered from the token
        counts the API reported), usage (those raw token counts), warnings
        (e.g. a retired model ID rerouted to its GA replacement, or a size the
        resolved model cannot produce), and sidecar_url pointing at the
        manifest written next to the image.

        When the model answers with text instead of an image — a safety
        refusal or a clarifying question — there is no preview block and the
        JSON reports message, generated_text, model, and the usage/cost of
        that billed exchange.

        A dry run reports dry_run, requested_model, model, prompt, image_size,
        estimated_cost and any warnings.
    """
    if director_notes is not None:
        return await _directed(
            ctx,
            render=lambda take_prompt, quote, _time_left=None: generate_image(
                ctx=ctx,
                prompt=take_prompt,
                model=model,
                image_uri=image_uri,
                image_base64=image_base64,
                reference_image_uris=reference_image_uris,
                image_size=image_size,
                media_resolution=media_resolution,
                aspect_ratio=aspect_ratio,
                person_generation=person_generation,
                thought_signature_url=thought_signature_url,
                dry_run=quote,
            ),
            returns_content=True,
            media_kind="image",
            prompt=prompt,
            notes=director_notes,
            max_retakes=max_retakes,
            dry_run=dry_run,
            references=_review_references(
                _uri_or_base64("source", "image", image_uri, image_base64),
                *(
                    ("reference", f"reference_image_uris[{i}]", uri)
                    for i, uri in enumerate(reference_image_uris or [])
                ),
            ),
        )
    try:
        app_ctx = ctx.request_context.lifespan_context
        data_dir = app_ctx.data_folder

        # Refuse an empty prompt on both the quote and the render: the
        # composites (generate_clip / generate_storyboard) already reject a
        # blank beat/shot prompt, but the primitive priced and billed one.
        if not prompt or not prompt.strip():
            raise ValueError("prompt must be a non-empty description of the image.")

        # An over-count reference list is truncated to what the model accepts;
        # say so on both surfaces rather than silently dropping a reference the
        # caller supplied (the render was doing this with no signal at all).
        # Tense follows the surface: a dry_run has sent nothing yet, so past
        # tense there ("were not sent") reads as if the call already ran.
        ref_count = len(reference_image_uris) if reference_image_uris else 0

        def _image_ref_warning(*, future: bool) -> list[str]:
            if ref_count <= _MAX_IMAGE_REFERENCE_IMAGES:
                return []
            dropped = ref_count - _MAX_IMAGE_REFERENCE_IMAGES
            tail = (
                "will not be sent and will not influence the image"
                if future
                else "were not sent and did not influence this image"
            )
            return [
                f"{ref_count} reference images were supplied but Gemini image "
                f"models accept {_MAX_IMAGE_REFERENCE_IMAGES}; the last "
                f"{dropped} {tail}."
            ]

        if dry_run:
            # Resolve the model the same way the impl would, so the estimate
            # prices what would actually run rather than the requested alias.
            from .image import resolve_image_model

            resolved, plan_warnings, effective_size = resolve_image_model(
                model, image_size
            )

            # Refuse a local source the real run would reject, so a quote does
            # not price a render that cannot fetch its inputs. After the model
            # resolution so a model error still takes precedence. base64 inputs
            # are inline bytes, not paths, so they are never passed here.
            # References are sliced to reference_image_uris[
            # :_MAX_IMAGE_REFERENCE_IMAGES] — the render fetches only that many,
            # so validating a bad 15th reference would refuse one the render
            # silently drops (and warns about, below).
            _assert_local_source(image_uri, data_dir, "image_uri")
            for ref_uri in (reference_image_uris or [])[:_MAX_IMAGE_REFERENCE_IMAGES]:
                _assert_local_source(ref_uri, data_dir, "reference image")
            # The inputs the render decodes and reads locally, checked the same
            # way, so a quote is refused wherever the render would be.
            if image_base64 and not image_uri:
                _decode_base64_capped(image_base64)
            if thought_signature_url:
                await _read_thought_signature_async(
                    _thought_signature_path(thought_signature_url), data_dir
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
            warnings = list(plan_warnings) + _image_ref_warning(future=True)
            if warnings:
                payload["warnings"] = warnings
                await _emit_warnings(ctx, warnings)
            return [TextContent(type="text", text=_respond(app_ctx, payload))]

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
                raise ValueError(_fetch_failure("image_uri", image_uri, data_dir))
        elif image_base64:
            image_bytes = _decode_base64_capped(image_base64)

        # Fetch reference images
        reference_images: list[bytes] = []
        if reference_image_uris:
            for ref_uri in reference_image_uris[:_MAX_IMAGE_REFERENCE_IMAGES]:
                ref_bytes = await fetch(
                    ref_uri,
                    allowed_dir=data_dir,
                    allowed_gcs_buckets=app_ctx.allowed_gcs_buckets,
                )
                # Don't silently drop a reference the caller explicitly asked
                # for — that would quietly change the generation result.
                if ref_bytes is None:
                    raise ValueError(
                        _fetch_failure("reference image", ref_uri, data_dir)
                    )
                reference_images.append(ref_bytes)

        # Read thought signature from file if URL provided. A malformed value
        # must not be silently ignored — that would quietly turn a multi-turn
        # edit into an unrelated fresh generation.
        thought_signature = None
        if thought_signature_url:
            sig_path = _thought_signature_path(thought_signature_url)
            thought_signature = await _read_thought_signature_async(sig_path, data_dir)

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
            # A refusal or a clarifying question is still a billed call: the
            # tokens were metered and the account was charged. Returning usage
            # without a cost made every refusal invisible to an agent totalling
            # spend. Priced at n=0 images, so the reported output tokens are
            # billed as text rather than at the 20x image rate.
            text_usage = result.get("usage")
            if text_usage:
                text_cost = _image_cost(
                    result.get("model", model), image_size, usage=text_usage, n=0
                )
                if text_cost:
                    result["cost"] = text_cost
            for warning in result.get("warnings", []):
                await ctx.warning(warning)
            await ctx.info("Model returned text only")
            return [TextContent(type="text", text=_respond(app_ctx, result))]

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

        # Surface impl warnings (e.g. an Imagen ID rerouted to its GA target),
        # plus the reference-truncation warning computed up front so the render
        # discloses the dropped references the quote already flagged. These go
        # out on the MCP logging channel as well as in the payload, so a client
        # that only reads the image sees them too.
        impl_warnings = (result.get("warnings") or []) + _image_ref_warning(
            future=False
        )
        if impl_warnings:
            response_data["warnings"] = impl_warnings
            await _emit_warnings(ctx, impl_warnings)

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
            TextContent(type="text", text=_respond(app_ctx, response_data)),
        ]
    except Exception as e:
        await ctx.error(f"Image generation failed: {e}")
        logger.exception("Tool error")
        body: dict[str, Any] = {"error": str(e)}
        if (status := http_status(e)) is not None:
            body["status_code"] = status
        if (wait := retry_after_seconds(e)) is not None:
            body["retry_after_seconds"] = wait
        return [TextContent(type="text", text=json.dumps(body))]


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
    resolution: str | None = None,
    source_video_uri: str | None = None,
    previous_interaction_id: str | None = None,
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
    4K on a 1K-only model, Veo on the Gemini Developer API, GCS output on the
    Gemini API — and reports them as conflicts with a fix.

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
        resolution: Exact video resolution: "360p", "720p", "1080p" or "4K".
            For an upscale, the target.
        source_video_uri: An existing clip the plan works on (to upscale,
            extend or bridge from); carried into the routes' params.
        previous_interaction_id: An Omni interaction the plan continues (to
            edit, extend or upscale); carried into the routes' params. The
            plan continues it on the model and backend this server recorded
            for it; one recorded on a retired model, or on a backend this
            server cannot reach, comes back as a conflict naming the clip it
            saved, to pass as input_video_uri instead.

    Returns:
        JSON plan: ranked `routes` (each with tool, model, ready-to-use
        `params`, score, rationale, caveats, cost), `rejected` models with
        reasons, `conflicts`, a suggested multi-step `workflow`, and `notes`.
    """
    try:
        from dataclasses import asdict

        from .routing import RoutingConstraints, plan_generation as plan_impl

        # Tell the planner what this deployment can actually reach: Veo on a
        # Gemini-API server, or omni on the Gemini API from a Vertex server
        # with no key, are routes the real call refuses. The server knows; the
        # router is pure and cannot find out.
        app_ctx = ctx.request_context.lifespan_context
        primary_is_vertex = bool(
            getattr(getattr(app_ctx.client, "_api_client", None), "vertexai", False)
        )
        # What was recorded about the interaction, from the same sidecar
        # lookup the tools make. An interaction cannot change models or
        # backends, so the planner needs both to plan a continuation the tool
        # will accept -- and to plan none at all for one recorded on a retired
        # model, which the tools refuse, pointing at the clip it saved. An
        # upscale starts from the recorded resolution.
        prior = await _prior_interaction_or_empty(
            app_ctx.videos_dir, previous_interaction_id
        )
        constraints = RoutingConstraints(
            backend="vertex" if primary_is_vertex else "gemini_api",
            gemini_api_key_available=(
                not primary_is_vertex or app_ctx.gemini_api_client is not None
            ),
            omni_backend=(
                None
                if app_ctx.omni_backend_default == "auto"
                else app_ctx.omni_backend_default
            ),
            # A Veo extension on Vertex AI is refused with neither an
            # output_gcs_uri nor this bucket to write to.
            video_gcs_bucket_configured=bool(app_ctx.video_gcs_bucket),
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
            # Without these an upscale planned over MCP could never carry its
            # source or an explicit target, and always asked for them.
            resolution=resolution,
            source_video_uri=source_video_uri,
            previous_interaction_id=previous_interaction_id,
            previous_interaction_model=prior.model,
            previous_interaction_backend=prior.backend,
            previous_interaction_resolution=prior.resolution,
            previous_interaction_video_url=prior.video_url,
            previous_interaction_seconds=prior.duration_seconds,
        )
        # Off the loop: the planner is pure string work, but a long intent is
        # not free (see routing.MAX_INTENT_CHARS), and one blocked coroutine
        # blocks every request.
        plan = await asyncio.to_thread(plan_impl, intent, constraints)

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
      1. A composited contact-sheet PNG, written full-resolution to disk
         (sheet_url) and returned inline as a downscaled preview. The preview
         is the thing you look at; open sheet_url when a panel needs a closer
         read than it survives.
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
            line, `notes` are camera/lighting notes. Any other key is an
            error naming what you probably meant, rather than a silently
            dropped field on a board that still bills per frame.
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
        A preview of the contact sheet inline, plus JSON with storyboard_url
        (HTML), sheet_url (the full-resolution PNG), per-shot results, total
        cost and total runtime.
    """
    # Accumulators live outside the try so the cancellation handler below can
    # still see what was rendered -- and billed -- before the caller vanished.
    frames: list[Any] = []  # StoryboardFrame; the module is imported lazily below
    shot_results: list[dict[str, Any]] = []
    try:
        app_ctx = ctx.request_context.lifespan_context

        if not shots:
            raise ValueError("shots list must not be empty")
        for name, value in (("title", title), ("subtitle", subtitle)):
            if isinstance(value, str) and len(value) > MAX_STORYBOARD_TEXT_CHARS:
                raise ValueError(
                    f"{name} is {len(value)} characters; the limit is "
                    f"{MAX_STORYBOARD_TEXT_CHARS}. It is drawn as a single "
                    "ellipsized line."
                )
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
            _validate_spec_keys(shot, f"shots[{i}]", _SHOT_KEYS, _SHOT_KEY_SUGGESTIONS)
            if not str(shot.get("prompt", "")).strip():
                raise ValueError(f"shots[{i}] is missing a non-empty 'prompt'")
            for field in ("prompt", "caption", "notes"):
                value = shot.get(field)
                if value is not None and not isinstance(value, str):
                    raise ValueError(
                        f"shots[{i}].{field} must be a string, "
                        f"got {type(value).__name__}"
                    )
                if isinstance(value, str) and len(value) > MAX_STORYBOARD_TEXT_CHARS:
                    raise ValueError(
                        f"shots[{i}].{field} is {len(value)} characters; the "
                        f"limit is {MAX_STORYBOARD_TEXT_CHARS}. A panel draws "
                        "about three wrapped lines, so text this long cannot "
                        "be shown and is more likely a mistake than a prompt."
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
                # Mirror the plan warnings onto the notification channel, as
                # generate_image's dry_run does — a quote that hid a retired
                # model reroute understated what the real run discloses.
                for warning in plan_warnings:
                    await ctx.warning(warning)
            return [TextContent(type="text", text=_respond(app_ctx, payload))]

        from .storyboard import StoryboardFrame, render_sheet_preview, write_storyboard

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

        # Reuse the single sheet write_storyboard already composited (and wrote
        # to disk) for the inline copy, reading it back rather than compositing
        # the board a second time — the earlier separate render re-decoded every
        # frame and re-ran a LANCZOS pass per panel, measured in seconds on a
        # 24-shot board, to reproduce the same storyboard. What goes inline is a
        # bounded preview of it: a full-size sheet is over a megabyte from about
        # a dozen shots up, which an MCP client refuses outright, and the sheet
        # that matters is the one on disk at sheet_url.
        def _sheet_preview() -> bytes:
            return render_sheet_preview(Path(artifacts["sheet_path"]).read_bytes())

        inline_preview = await asyncio.to_thread(_sheet_preview)

        # A frame whose bytes exist but do not decode renders as "SHOT NOT
        # GENERATED" on the sheet, so it must not be counted as a rendered
        # shot here — this message said "3/3 shots" for a board with a hole
        # in it. write_storyboard reports them because it is the one place
        # that has tried to decode them.
        undecodable = {
            int(i) for i in (artifacts.get("undecodable_shots") or "").split(",") if i
        }
        for result in shot_results:
            if result["shot"] in undecodable and "error" not in result:
                result["error"] = "Image could not be decoded"
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
        # A sheet the preview could not decode still has both artifacts on disk,
        # so the board is delivered text-only rather than failing outright.
        blocks: list[Any] = []
        if inline_preview:
            blocks.append(Image(data=inline_preview, format="jpeg"))
        blocks.append(TextContent(type="text", text=_respond(app_ctx, response_data)))
        return blocks
    except asyncio.CancelledError:
        # A cancellation is a BaseException, so the handler below never sees
        # it -- and unlike loop_extend, generate_clip and extend_video_omni,
        # this tool had no handler of its own. Cancel after 2 of 4 paid shots
        # left two orphan PNGs on disk with no sidecar and no cost record.
        # Say what was rendered, then let the cancellation continue.
        rendered = [r for r in shot_results if "image_url" in r]
        if rendered:
            logger.warning(
                "generate_storyboard cancelled after %d of %d shot(s) rendered "
                "and billed; frames: %s",
                len(rendered),
                len(shots),
                ", ".join(str(r["image_url"]) for r in rendered),
            )
        raise
    except Exception as e:
        await ctx.error(f"Storyboard failed: {e}")
        logger.exception("Tool error")
        return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]


@mcp.tool()
async def generate_video(
    ctx: Context[ServerSession, AppContext],
    prompt: str,
    model: VideoModel | TranslatedVideoModel,
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
    draft_resolution: str | None = None,
    timeout_seconds: float = VEO_DEFAULT_TIMEOUT_SECONDS,
    dry_run: bool = False,
    director_notes: str | None = None,
    max_retakes: int = DEFAULT_MAX_RETAKES,
) -> str:
    """Generate a video using Google VEO models.

    Args:
        ctx: MCP context with application state
        prompt: Text description of the video to generate
        model: Model to use - options include:
               - "veo-3.1-generate-001": VEO 3.1 (highest quality, 4/6/8s, audio)
               - "veo-3.1-fast-generate-001": VEO 3.1 Fast (faster, 4/6/8s, audio)
                 Veo runs on Vertex AI only: Google switched off its Veo 3.1
                 models on the Gemini Developer API on 2026-10-22, so a server
                 without Vertex credentials refuses this tool and points to
                 generate_video_omni. Prefer these two names. The schema also
                 accepts "veo-3.1-generate-preview" and
                 "veo-3.1-fast-generate-preview" — the spellings older
                 Gemini-API renders reported — so a model from such a manifest
                 round-trips; they resolve to the -001 models above.
        aspect_ratio: 16:9 (default) or 9:16
        duration_seconds: Video duration (4/6/8s)
        include_audio: Enable audio generation
        audio_prompt: Audio description
        negative_prompt: Things to avoid in the video
        seed: Random seed for reproducibility
        resolution: Output resolution, "720p", "1080p" or "4K".
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
            Requires output_gcs_uri (the larger combined video exceeds inline
            limits). Billed on the ASSEMBLED clip, not the ~7s appended: a 4s
            source extended once renders and bills 11s.

            A REMOTE source (gs://, http://) makes the dry-run estimate a
            FLOOR rather than a ceiling. This is the one place a pre-flight in
            this server can under-state, and it is deliberate: the bill
            depends on the source's length, a dry run is documented offline so
            it does not download the source to measure it, and Veo publishes
            no maximum source length to assume in its place — so there is no
            honest ceiling to quote. MEASURED: an 11.01s remote source quoted
            $0.70 and billed $1.80, 61% under. billed_seconds_source says
            "FLOOR, not an estimate" when it applies, and the real run
            measures the source and reports the true figure. For a ceiling
            before you spend, quote against a local copy, or compute it:
            billed = source + ~7s per turn. (Omni's extension quote has no
            such exception — an unmeasurable source is quoted at the model's
            cumulative ceiling for an extended clip, which over-states.)
        output_gcs_uri: GCS bucket URI for large video output (e.g. gs://bucket/path/).
        timeout_seconds: Overall deadline for the render (submit + polling +
            download), default 210s -- the same figure the omni tools use, for
            the same reason: common MCP hosts cap one tool call at roughly four
            minutes, and a render that outlives the HOST's limit is cancelled
            by the host. The server never answers, the caller sees a bare
            "Tool execution failed" with no message, cost or operation, and
            the Veo operation completes and bills regardless -- which is what
            a 4K render did, twice. Under the host ceiling the server answers
            first, with the operation name, the mode and the attempted cost.
            Raise it when your host allows a longer call; 4K is the slowest
            tier.
        director_notes: Optional. What the clip must get right, in plain
            words ("the mug turns left to right; the logo stays readable; no hands enter the frame"). Gemini 4 Argon turns the notes into checks,
            reviews the clip against them, sampling it at 4 frames a second, and when a check fails
            renders again with fix notes, up to max_retakes times. The response
            then carries a `director` block with each check's state on every
            take, and `cost` covers every take and every director call. Omit it
            and the tool behaves exactly as it does without a director: no
            extra call and no extra cost.
            In the notes, call image_uri "the first frame", last_frame_uri
            "the last frame" and reference_image_uris[0] "reference 1" and so
            on; the reviewer is shown the frames and the first three
            references. A draft is reviewed against image_uri alone, as
            "input image 1", since that is all it renders from.
            Put text that must appear in quotes, exactly as it should be set:
            it is compared character for character. Say "throughout" or "the
            whole clip" for what must hold at every moment ("steam rises
            throughout"): it is judged one second at a time.
        max_retakes: Re-renders the director may spend when a check fails, 0
            to 2 (default 1). A dry run with director_notes quotes the worst
            case: every take rendered and reviewed. A retake the service turns
            away with a 429 (image quotas can allow only a few renders a
            minute) or a server error is tried again after 10 s, then 30 s,
            or the wait the 429 asked for, while the call's time allows.
            With director_notes, timeout_seconds bounds the whole call: each
            take renders in the time left, and a review or retake that could
            not finish in time is skipped and reported.
        dry_run: When True, generate nothing and return only the cost
            estimate for the call that would run (the omni draft price when
            draft=True). Free and instant. A real run reports the actual
            cost, derived from the effective duration the API rendered.
            With extend_video_uri and a REMOTE source the estimate is a
            FLOOR, not a ceiling — see the note under extend_video_uri.
        draft_resolution: Resolution for a draft pass on Omni 1.2: 360p,
            720p (default), 1080p or 4K. 360p costs about a third of 720p.
        draft: Render a fast draft on Omni 1.2, then re-run with draft=False
            to finalize on Veo. Omni 1.2 costs $0.069504/s at 720p and
            $0.023172/s at 360p. Draft mode ignores Veo-only controls (seed,
            negative_prompt, resolution, last frame, reference images,
            extension); any that were passed are noted in the response.

    Returns:
        JSON. A Veo render reports message, video_url, prompt, model,
        audio_enabled, duration_seconds, generation_mode, cost (metered from
        the rendered duration) and sidecar_url — or `manifest` inline instead
        of sidecar_url when the output landed in GCS; `warnings` and
        `extended_from` appear only when they apply.

        A draft (draft=True) is an omni render and reports the same
        message/video_url/prompt/model/duration_seconds/generation_mode/cost
        plus interaction_id, duration_source and requested_* fields.
        generation_mode is "draft". There is no audio_enabled: omni exposes no
        audio control, so the server cannot state what the render carries —
        include_audio is reported under `warnings` as ignored instead.

        A dry run reports dry_run, model, resolution, generation_mode,
        requested_duration_seconds, the snapped duration_seconds that will
        actually render, and estimated_cost.
    """
    if director_notes is not None:
        return await _directed(
            ctx,
            render=lambda take_prompt, quote, time_left=None: generate_video(
                ctx=ctx,
                prompt=take_prompt,
                model=model,
                aspect_ratio=aspect_ratio,
                duration_seconds=duration_seconds,
                include_audio=include_audio,
                audio_prompt=audio_prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                resolution=resolution,
                person_generation=person_generation,
                image_uri=image_uri,
                image_base64=image_base64,
                last_frame_uri=last_frame_uri,
                last_frame_base64=last_frame_base64,
                reference_image_uris=reference_image_uris,
                extend_video_uri=extend_video_uri,
                output_gcs_uri=output_gcs_uri,
                draft=draft,
                draft_resolution=draft_resolution,
                timeout_seconds=(
                    timeout_seconds if time_left is None else max(1.0, time_left)
                ),
                dry_run=quote,
            ),
            returns_content=False,
            media_kind="video",
            prompt=prompt,
            notes=director_notes,
            max_retakes=max_retakes,
            dry_run=dry_run,
            duration_s=duration_seconds,
            # A draft renders on omni from image_uri alone, as an input image
            # whose use omni infers; it ignores the last frame and references,
            # so the reviewer is not shown them either.
            references=(
                _review_references(
                    _uri_or_base64("input", "image", image_uri, image_base64)
                )
                if draft
                else _review_references(
                    _uri_or_base64("first_frame", "image", image_uri, image_base64),
                    _uri_or_base64(
                        "last_frame", "last_frame", last_frame_uri, last_frame_base64
                    ),
                    *(
                        ("reference", f"reference_image_uris[{i}]", uri)
                        for i, uri in enumerate(reference_image_uris or [])
                    ),
                )
            ),
            unshown_inputs=(["the video being extended"] if extend_video_uri else None),
            # A draft renders on omni, which ignores include_audio.
            audio_requested=None if draft else include_audio,
            deadline_s=timeout_seconds,
        )
    # How far a draft's omni render got, for the error body if it fails.
    draft_submission = OmniSubmission()
    try:
        app_ctx = ctx.request_context.lifespan_context
        data_dir = app_ctx.data_folder

        # Fail fast, before any fetch work AND on the dry_run path, so a quote
        # refuses exactly what the render refuses: empty prompt and mutually
        # exclusive inputs (the same check the impl runs). Draft routes to
        # omni, which takes only a prompt + optional images, so the frame /
        # extend conflicts do not apply there.
        _validate_aspect_ratio(aspect_ratio)
        _validate_duration_seconds(duration_seconds)
        if draft:
            if not prompt or not prompt.strip():
                raise ValueError("prompt must be a non-empty description of the video.")
        else:
            validate_video_request_inputs(
                prompt=prompt,
                has_first_frame=bool(image_uri or image_base64),
                has_last_frame=bool(last_frame_uri or last_frame_base64),
                reference_count=len(reference_image_uris)
                if reference_image_uris
                else 0,
                has_extend=bool(extend_video_uri),
            )

        # Resolved BEFORE the dry_run branch, so a quote applies every
        # rejection the render applies. GCS output is Vertex-only and an
        # explicit output_gcs_uri on the Gemini API is documented as rejected —
        # but the check lived past the dry_run return, so a quote priced a call
        # the real path refuses, with no warning and no warnings field at all.
        # Same defect class as the over-length extension source.
        #
        # Skipped for a draft, which is not a Veo call at all: it routes to
        # omni. Hoisting these above the draft branch made draft=True fail on
        # inputs it documents as IGNORED — `_draft_ignored_veo_params` lists
        # output_gcs_uri, yet _resolve_video_gcs raised on a Gemini-API
        # deployment before the warning naming it could ever be built, and
        # _client_for_video_model refuses Veo on a Gemini-API deployment, a
        # client the draft never uses. The backend a draft reports is omni's,
        # which is the one that will serve it.
        video_client: genai.Client | None = None
        draft_delivery: str | None = None
        draft_slow_tier: str | None = None
        draft_length: str | None = None
        draft_seconds = duration_seconds
        if draft:
            is_vertex_client = _omni_backend_is_vertex(
                app_ctx, requested=_omni_server_backend(app_ctx)
            )
            gcs_uri = None
            # A draft renders on omni, so omni's inline-delivery policy
            # applies, judged here so the quote and the render agree. It has
            # no Cloud Storage destination (output_gcs_uri is ignored).
            draft_render = _omni_preview_model(draft_resolution)
            draft_spec = omni_spec(draft_render[0])
            # A draft previews the Veo render it stands in for, so it runs as
            # long as that render will: Veo renders 4, 6 or 8 seconds, and a
            # 3s draft of a clip that finalizes at 4s judged the wrong timing.
            draft_seconds = min(
                draft_spec.max_duration_seconds,
                max(3, round(_snapped_duration_for_quote(model, duration_seconds))),
            )
            if draft_seconds != duration_seconds:
                draft_length = (
                    f"The draft renders {draft_seconds}s, the length {model} "
                    "renders this request at with draft=false, so the draft and "
                    "the final run the same length."
                )
            draft_delivery = omni_inline_delivery_policy(
                draft_spec,
                vertexai=is_vertex_client,
                output_gcs_uri=None,
                resolution=draft_render[1],
                task=None,
                output_seconds=draft_seconds,
                takes_output_gcs_uri=False,
            )
            # The draft runs under this call's deadline (see the render below).
            draft_slow_tier = omni_slow_tier_warning(
                resolution=draft_render[1],
                kind="render",
                output_seconds=draft_seconds,
                timeout_seconds=timeout_seconds,
            )
        else:
            video_client = _client_for_video_model(app_ctx, model)
            is_vertex_client = bool(
                getattr(video_client._api_client, "vertexai", False)
            )
            gcs_uri = _resolve_video_gcs(
                output_gcs_uri,
                app_ctx.video_gcs_bucket,
                app_ctx.allowed_gcs_buckets,
                is_vertex_client,
            )

        if dry_run:
            # Inline images are decoded by the render; decode them here too,
            # so a quote is refused wherever the render would be.
            if image_base64 and not image_uri:
                _decode_base64_capped(image_base64)
            # A draft ignores the last frame, so it is not checked for one.
            if last_frame_base64 and not last_frame_uri and not draft:
                _decode_base64_capped(last_frame_base64)
            # Draft mode routes to omni, which serves a plain text/image
            # request and snaps on its own [3, 15]s clamp — the Veo modes
            # below do not apply to it.
            quoted_mode = "text_to_video"
            if not draft:
                # Same checks the impl applies, so a quote can never succeed
                # for a render the real call would refuse. The mode is derived
                # the way the impl derives it and reported on the quote.
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
                # The backend is as much a constraint as the model: Veo has
                # no endpoint on the Gemini Developer API at all.
                validate_render_options(
                    model,
                    resolution,
                    backend=_backend_of(app_ctx, str(model)),
                )

                # Refuse a local source the real run would reject, so a quote
                # does not price a render that cannot fetch its inputs. AFTER
                # validate_render_options so a model or backend refusal takes
                # precedence (mirrors generate_bridge). base64 frames are inline
                # bytes, not paths, so they are never passed here. References
                # are sliced to what the render fetches (reference_image_uris[:
                # _MAX_REFERENCE_IMAGES]) — validating beyond that would refuse
                # an over-count reference the render silently truncates (and
                # warns about below), refusing what the real run accepts.
                _assert_local_source(image_uri, data_dir, "image_uri")
                _assert_local_source(last_frame_uri, data_dir, "last_frame_uri")
                _assert_local_source(extend_video_uri, data_dir, "extend_video_uri")
                for ref_uri in (reference_image_uris or [])[:_MAX_REFERENCE_IMAGES]:
                    _assert_local_source(ref_uri, data_dir, "reference image")
            else:
                # The draft takes image_uri too (omni accepts a prompt plus
                # images) and its real run refuses a missing or unconfined
                # one -- but every source check above sat under `if not
                # draft`, so generate_video(draft=True, dry_run=True,
                # image_uri=<missing file, or /etc/passwd>) quoted $0.815 for
                # a render that cannot fetch its input. Same invariant.
                _assert_local_source(image_uri, data_dir, "image_uri")
            # Report the model the real run will report, not the raw input: a
            # caller who pinned (or fed back) a `-preview` id saw the quote say
            # `-preview` while the render reported the resolved `-001` name.
            # Resolve it the way the impl does for the primary backend.
            draft_res: str | None = None
            if draft:
                est_model, draft_res = _omni_preview_model(draft_resolution)
            else:
                # The id that will actually be BILLED, which is the same
                # resolver cost.detail uses — so the two can never name
                # different models in one response. Reporting the served
                # spelling here did exactly that: "model" said
                # veo-3.1-fast-generate-preview while the detail beside it
                # said -001. It also folds a -preview id the caller fed back
                # from a previous response onto its canonical name.
                from .pricing import resolve_model_id

                est_model = resolve_model_id(str(model))
            est_res = (draft_res or "720p") if draft else (resolution or "720p")
            # Report the duration that will actually render. Returning the
            # request beside a price for the snapped value (5s quoted as "4s
            # of video") made the payload contradict itself.
            #
            # The mode has to travel with the duration: Veo forces 8s for a
            # reference render and 7s for an extension, so a 4s reference
            # request was quoted at half what the render bills — a pre-flight
            # may over-state but must never under-state.
            quoted_duration = (
                float(draft_seconds)
                if draft
                else _snapped_duration_for_quote(
                    est_model, duration_seconds, quoted_mode
                )
            )
            payload: dict[str, Any] = {
                "dry_run": True,
                "message": "Estimate only — nothing was generated",
                "model": est_model,
                "resolution": est_res,
                "generation_mode": "draft" if draft else quoted_mode,
                # Which backend this will actually run on. Omni's responses
                # carry it; Veo's did not, and a session spent three calls
                # working out a backend confusion that this field answers in
                # one.
                "backend": "vertex" if is_vertex_client else "gemini_api",
                "requested_duration_seconds": duration_seconds,
                "duration_seconds": quoted_duration,
                "estimated_cost": _video_cost(
                    est_model,
                    draft_seconds if draft else duration_seconds,
                    resolution=est_res,
                    include_audio=include_audio,
                    generation_mode=quoted_mode,
                ),
            }
            if quoted_mode == "extend_video":
                # The real run was corrected to bill the ASSEMBLED clip and the
                # quote beside it was not: 7s / $0.70 for a 4s source whose
                # extension renders and bills 11s — a quote BELOW its own real
                # run's figure. Same rule, same shared function, both ends.
                quoted_source_seconds = await _probe_local_video_seconds(
                    ctx, extend_video_uri
                )
                ext_billing = veo_extension_billing(
                    turns=1, source_seconds=quoted_source_seconds
                )
                payload["billed_seconds"] = ext_billing.billed_seconds
                payload["appended_seconds"] = ext_billing.appended_seconds
                if ext_billing.turn_output_seconds:
                    payload["turn_output_seconds"] = ext_billing.turn_output_seconds
                    payload["duration_seconds"] = ext_billing.turn_output_seconds[-1]
                payload["billed_seconds_source"] = ext_billing.duration_source
                if ext_billing.turn_output_seconds:
                    payload["duration_source"] = (
                        "projected from a measured "
                        f"{quoted_source_seconds:g}s source: an extension "
                        "renders the assembled clip (source + ~7s)"
                        if quoted_source_seconds is not None
                        else "projected: an extension renders the assembled "
                        "clip (source + ~7s)"
                    )
                payload["estimated_cost"] = _video_cost(
                    est_model,
                    float(ext_billing.billed_seconds),
                    resolution=est_res,
                    include_audio=include_audio,
                    presnapped=True,  # a measured/projected length must not re-snap
                )
                if ext_billing.warning:
                    payload.setdefault("warnings", []).append(ext_billing.warning)
            slow_tier = _veo_slow_tier_warning(resolution, timeout_seconds)
            if slow_tier:
                payload.setdefault("warnings", []).append(slow_tier)
            # Disclose the Veo-only params a draft will drop, exactly as the
            # real draft run does — a quote that hid them let a caller price a
            # render believing paid controls (seed, negative_prompt, ...) were
            # honored when omni ignores them.
            if draft:
                ignored = _draft_ignored_veo_params(
                    seed=seed,
                    negative_prompt=None,  # folded inline, see below
                    resolution=resolution,
                    person_generation=person_generation,
                    last_frame=last_frame_uri or last_frame_base64,
                    reference_image_uris=reference_image_uris,
                    extend_video_uri=extend_video_uri,
                    output_gcs_uri=output_gcs_uri,
                    include_audio=include_audio,
                )
                if ignored:
                    payload["ignored_veo_params"] = ignored
                    payload["warnings"] = [_draft_ignored_warning(ignored)]
                if draft_length:
                    payload.setdefault("warnings", []).append(draft_length)
                if draft_delivery:
                    payload.setdefault("warnings", []).append(draft_delivery)
                if draft_slow_tier:
                    payload.setdefault("warnings", []).append(draft_slow_tier)
                await _emit_warnings(ctx, payload.get("warnings"))
            elif (
                quoted_mode == "reference_to_video"
                and reference_image_uris
                and len(reference_image_uris) > _MAX_REFERENCE_IMAGES
            ):
                # The render warns and truncates over-count references; the
                # quote must say so too rather than pricing them all in silence.
                n = len(reference_image_uris)
                warning = (
                    f"{n} reference images were supplied but Veo 3.1 accepts "
                    f"{_MAX_REFERENCE_IMAGES}; the last {n - _MAX_REFERENCE_IMAGES} "
                    "will not be sent and will not influence the render."
                )
                payload["warnings"] = [warning]
                await _emit_warnings(ctx, [warning])
            return _respond(app_ctx, payload)

        # Draft mode: hand off to the fast omni path. Omni supports only a
        # text prompt + optional input image(s), so Veo-only controls are
        # ignored — surface which ones so the caller isn't surprised.
        if draft:
            ignored = _draft_ignored_veo_params(
                seed=seed,
                negative_prompt=None,  # folded inline, see below
                resolution=resolution,
                person_generation=person_generation,
                last_frame=last_frame_uri or last_frame_base64,
                reference_image_uris=reference_image_uris,
                extend_video_uri=extend_video_uri,
                output_gcs_uri=output_gcs_uri,
                include_audio=include_audio,
            )
            # audio_prompt CAN be honored best-effort: inline it into the
            # prompt text, exactly like the Veo path does.
            draft_prompt = prompt
            if audio_prompt:
                draft_prompt = f"{prompt}\nAudio: {audio_prompt}"
            if negative_prompt:
                # Omni has no negative_prompt field, and its docs say to state
                # negatives inline ("No dialogue"). Folded rather than dropped:
                # a draft that ignores the caller's exclusions previews the
                # wrong shot.
                draft_prompt = f"{draft_prompt}\nNo {negative_prompt}."
            draft_image_bytes: list[bytes] | None = None
            if image_uri:
                b = await fetch(
                    image_uri,
                    allowed_dir=data_dir,
                    allowed_gcs_buckets=app_ctx.allowed_gcs_buckets,
                )
                if b is None:
                    raise ValueError(_fetch_failure("image_uri", image_uri, data_dir))
                draft_image_bytes = [b]
            elif image_base64:
                draft_image_bytes = [_decode_base64_capped(image_base64)]

            extra: dict[str, Any] = {"draft": True}
            if ignored:
                extra["ignored_veo_params"] = ignored
            if negative_prompt:
                extra["negative_prompt_inlined"] = negative_prompt
            draft_model, draft_res = _omni_preview_model(draft_resolution)
            await ctx.info(f"Generating draft with {draft_model}")
            result = await _omni_generate_and_manifest(
                app_ctx,
                ctx,
                prompt=draft_prompt,
                model=draft_model,
                image_bytes_list=draft_image_bytes,
                aspect_ratio=aspect_ratio,
                duration_seconds=draft_seconds,
                resolution=draft_res,
                manifest_extra=extra,
                # The caller's deadline holds for a draft too; the default
                # matches omni's own, so an unset one changes nothing.
                timeout_seconds=max(1, int(timeout_seconds)),
                submission=draft_submission,
            )
            if ignored:
                result.setdefault("warnings", []).append(
                    _draft_ignored_warning(ignored)
                )
            if draft_length:
                result.setdefault("warnings", []).append(draft_length)
            if draft_delivery:
                result.setdefault("warnings", []).append(draft_delivery)
            if draft_slow_tier:
                result.setdefault("warnings", []).append(draft_slow_tier)
            if negative_prompt:
                # On the response too, not only in the manifest: the caller
                # reading the result should not need the sidecar to learn
                # their exclusion was honoured.
                result["negative_prompt_inlined"] = negative_prompt
                result.setdefault("warnings", []).append(
                    f"negative_prompt was folded into the prompt as 'No "
                    f"{negative_prompt}.' — omni has no negative_prompt field; "
                    "its documentation recommends inline negatives."
                )
            # The omni impl reports neither of these, so a draft response was
            # missing the two keys this tool's own Returns block names and
            # every non-draft response carries. `prompt` is the text actually
            # sent (audio_prompt inlined), matching what the Veo path reports.
            result.setdefault("generation_mode", "draft")
            result.setdefault("prompt", draft_prompt)
            # Mirror warnings (ignored Veo params, any GCS notice) onto the
            # notification channel, not just the JSON body.
            await _emit_warnings(ctx, result.get("warnings"))
            return _respond(app_ctx, result)

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
                raise ValueError(_fetch_failure("image_uri", image_uri, data_dir))
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
                raise ValueError(
                    _fetch_failure("last_frame_uri", last_frame_uri, data_dir)
                )
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
                    raise ValueError(
                        _fetch_failure("reference image", ref_uri, data_dir)
                    )
                reference_images.append(ref_bytes)

        # video_client / is_vertex_client / gcs_uri are resolved in the
        # pre-flight above, so the quote and the render cannot disagree about
        # the destination.

        # Combine explicit output_gcs_uri with the env default, validate the
        # allowlist, and apply the non-Vertex drop/raise gating.
        if extend_video_uri:
            # extend_video_uri is handed straight to the API as a video
            # source; a gs:// value must pass the same allowlist as every
            # other gs:// input so it can't reach arbitrary buckets.
            if extend_video_uri.startswith("gs://"):
                extend_bucket = _parse_gcs_bucket(extend_video_uri)
                if extend_bucket is None:
                    raise ValueError(f"Invalid extend_video_uri: {extend_video_uri}")
                _assert_gcs_bucket_allowed(extend_bucket, app_ctx.allowed_gcs_buckets)
            # Extensions produce a larger combined video that must be written
            # to GCS.
            if is_vertex_client and not gcs_uri:
                raise ValueError(
                    "Video extension on Vertex AI requires output_gcs_uri (or a "
                    "configured VIDEO_GCS_BUCKET). Extensions produce larger "
                    "combined videos that exceed inline response limits."
                )

        slow_tier_notice = _veo_slow_tier_warning(resolution, timeout_seconds)
        if slow_tier_notice:
            await _emit_warnings(ctx, [slow_tier_notice])
        await ctx.info(f"Generating video with model={model}")
        # Only the Veo path reaches this: the draft branch returned above.
        assert video_client is not None
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
            timeout_seconds=timeout_seconds,
        )
        await ctx.info("Video generated successfully")

        # Which backend actually ran this. Omni's responses carry it; Veo's did
        # not, and a session spent three calls working out a backend confusion
        # that this field answers in one.
        result["backend"] = "vertex" if is_vertex_client else "gemini_api"
        if slow_tier_notice:
            result.setdefault("warnings", []).append(slow_tier_notice)
        # Veo bills by resolution too, and this response named one nowhere:
        # the manifest recorded the raw request — null whenever the caller
        # defaulted — while the cost line below priced 720p. Measured, the
        # two cannot disagree silently.
        veo_facts = await _measure_rendered_video(
            result, resolution_key="resolution", requested_resolution=resolution
        )
        if veo_facts.duration_source:
            result["duration_seconds"] = round(veo_facts.duration_seconds or 0.0, 3)
            result["duration_source"] = veo_facts.duration_source
        effective_resolution = str(result.get("resolution") or resolution or "720p")
        result["resolution"] = effective_resolution
        result["resolution_source"] = veo_facts.resolution_source or (
            "the request"
            if resolution
            else "the default: generate_video renders 720p unless asked otherwise"
        )
        # A non-extension render is metered when it was measured; only an
        # extension can be a projection, and the billing helper says which.
        extension_metered = True
        # Keyed on the REQUEST, not on the impl echoing generation_mode back:
        # a caller who passed extend_video_uri is extending, and the billing
        # rule must not hinge on a field the response might omit.
        if extend_video_uri:
            # An extension is billed on the assembled clip, not on the footage
            # it appends — the same rule loop_extend applies, which this path
            # did not. It billed 7s, asserted is_estimate=False for a file it
            # never opened, and labelled the figure with the TEXT-TO-VIDEO
            # rule in the one mode where that rule is known to be false.
            ext_source_seconds = await _extension_source_seconds(
                ctx, extend_video_uri, allow_remote=True
            )
            billing = veo_extension_billing(
                turns=1,
                measured_turns=(
                    [veo_facts.duration_seconds]
                    if veo_facts.duration_source and veo_facts.duration_seconds
                    else None
                ),
                source_seconds=ext_source_seconds,
            )
            extension_metered = billing.is_metered
            result["billed_seconds"] = billing.billed_seconds
            result["appended_seconds"] = billing.appended_seconds
            if billing.turn_output_seconds:
                result["turn_output_seconds"] = billing.turn_output_seconds
                if not veo_facts.duration_source:
                    # The delivered clip is the ASSEMBLED one. The impl reports
                    # the 7s it was asked to append, which is neither what was
                    # delivered nor what is billed, and labelling that number
                    # with the projection described a length it did not hold.
                    result["duration_seconds"] = billing.turn_output_seconds[-1]
                    # Names the measurement, as billed_seconds_source does.
                    # Two fields agreeing on a number while only one says
                    # where it came from reads like they disagree.
                    result["duration_source"] = (
                        "projected from a measured "
                        f"{ext_source_seconds:g}s source: an extension delivers "
                        "the assembled clip (source + ~7s), and this one was "
                        "delivered remotely so it could not be measured"
                        if ext_source_seconds is not None
                        else "projected: an extension delivers the assembled "
                        "clip (source + ~7s), and this one was delivered "
                        "remotely so it could not be measured"
                    )
            result["billed_seconds_source"] = billing.duration_source
            if billing.warning:
                result.setdefault("warnings", []).append(billing.warning)
        result.setdefault(
            "duration_source",
            # The snapped-request rule is TEXT-TO-VIDEO's. Asserting it over an
            # extension is the one place it is known to be false: an extension
            # delivers the assembled clip, not the length it was sent. When
            # nothing could be measured, say that rather than borrow a rule.
            "not measured: the render was delivered remotely, and an extension "
            "delivers the assembled clip (source + ~7s) rather than the length "
            "it was sent, so its length is not known here"
            if extend_video_uri
            else "the snapped request: Veo renders exactly the length it is sent",
        )
        is_extension = bool(extend_video_uri)
        raw_billed = (
            result.get("billed_seconds")
            if is_extension
            else result.get("duration_seconds", duration_seconds)
        )
        # An extension always sets billed_seconds just above, but falling back
        # to the request rather than asserting keeps a missing key from
        # becoming a 500 on a render the caller has already paid for.
        billed_seconds = (
            float(raw_billed)
            if isinstance(raw_billed, (int, float))
            else float(duration_seconds)
        )
        # Metered only where the figure was measured. An extension projected
        # from its source is a good number, but it is not a meter reading —
        # and that is a fact the billing helper states, not one to re-derive
        # from the wording of a string.
        metered = extension_metered
        cost = _video_cost(
            result.get("model", model),
            billed_seconds,
            resolution=effective_resolution,
            include_audio=result.get("audio_enabled", include_audio),
            actual=metered,
            presnapped=not metered,
        )
        if cost:
            result["cost"] = cost
        manifest: dict[str, Any] = {
            "kind": "video",
            "prompt": prompt,
            "audio_prompt": audio_prompt,
            "negative_prompt": negative_prompt,
            # The model that actually ran: the impl normalizes an older
            # Gemini-API `-preview` spelling to its `-001` name.
            "model": result.get("model", model),
            "aspect_ratio": aspect_ratio,
            "duration_seconds": result.get("duration_seconds", duration_seconds),
            "duration_source": result.get("duration_source"),
            "resolution": result.get("resolution", resolution),
            "resolution_source": result.get("resolution_source"),
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
        # intent (e.g. a negative seed that was not sent).
        warnings = result.get("warnings")
        if warnings:
            manifest["warnings"] = warnings
            await _emit_warnings(ctx, warnings)
        sidecar_url = _write_sidecar(result.get("video_url", ""), manifest)
        if sidecar_url:
            result["sidecar_url"] = sidecar_url
        else:
            # No sidecar for remote (gs://) outputs — include the manifest
            # inline so generation parameters aren't lost for exactly the
            # large-video path GCS output is meant for.
            result["manifest"] = manifest

        return _respond(app_ctx, result)
    except asyncio.CancelledError:
        # The HOST's ceiling, not ours. A cancellation is a BaseException, so the
        # handler below never ran and the caller saw a bare "Tool execution
        # failed" -- no message, no cost, no operation name. That is exactly
        # what a 4K render on Vertex produced, twice, while billing $1.20 each
        # time. Record what was attempted, then let the cancellation continue.
        facts = _veo_attempt_facts(
            model=model,
            draft=draft,
            draft_resolution=draft_resolution,
            mode=_veo_request_mode(
                extend_video_uri=extend_video_uri,
                reference_image_uris=reference_image_uris,
                has_first_frame=bool(image_uri or image_base64),
                has_last_frame=bool(last_frame_uri or last_frame_base64),
            ),
            duration_seconds=duration_seconds,
            resolution=resolution,
            include_audio=include_audio,
        )
        logger.warning(
            "generate_video cancelled by the host (timeout_seconds=%s): %s -- the "
            "render may still complete and bill",
            timeout_seconds,
            json.dumps(facts, default=str),
        )
        raise
    except Exception as e:
        await ctx.error(f"Video generation failed: {e}")
        logger.exception("Tool error")
        # Structured, naming the mode, the attempted cost and -- for a timeout
        # -- the operation that may still complete and bill.
        body: dict[str, Any] = {"error": str(e)}
        if (status := http_status(e)) is not None:
            body["status_code"] = status
        if (wait := retry_after_seconds(e)) is not None:
            body["retry_after_seconds"] = wait
        body.update(
            _veo_attempt_facts(
                model=model,
                draft=draft,
                draft_resolution=draft_resolution,
                mode=_veo_request_mode(
                    extend_video_uri=extend_video_uri,
                    reference_image_uris=reference_image_uris,
                    has_first_frame=bool(image_uri or image_base64),
                    has_last_frame=bool(last_frame_uri or last_frame_base64),
                ),
                duration_seconds=duration_seconds,
                resolution=resolution,
                include_audio=include_audio,
            )
        )
        advice = _veo_access_advice(e)
        if advice:
            body["advice"] = advice
        # How far the render got, said explicitly: an absent operation name
        # cannot tell a local failure from a request that may have reached
        # Google. A draft runs on omni, which tracks its own.
        if draft and draft_submission.quota_retries:
            body["quota_retries"] = draft_submission.quota_retries
        submission = (
            draft_submission.state(e) if draft else getattr(e, "submission", "none")
        )
        operation_name = getattr(e, "operation_name", None)
        interaction_id = draft_submission.interaction_id if draft else None
        if isinstance(e, TimeoutError) or submission != "none":
            body["submission"] = submission
            if operation_name:
                body["operation_name"] = operation_name
            if interaction_id:
                body["interaction_id"] = interaction_id
            handle = (
                f"Operation {operation_name}"
                if operation_name
                else (f"Interaction {interaction_id}" if interaction_id else None)
            )
            body["note"] = _submission_note(
                submission,
                handle=handle,
                completed=draft and draft_submission.completed,
                final_status=draft_submission.final_status if draft else None,
            )
        if isinstance(e, TimeoutError):
            body["timed_out"] = True
            body["timeout_seconds"] = timeout_seconds
            body["operation_name"] = operation_name
            body["note"] = (
                "The deadline passed before the render finished. "
                + body["note"]
                + " Raise timeout_seconds if your host allows a longer call."
            )
        return _respond(ctx.request_context.lifespan_context, body)


@mcp.tool()
async def generate_transition(
    ctx: Context[ServerSession, AppContext],
    first_frame_uri: str,
    last_frame_uri: str,
    prompt: str = "smooth cinematic transition between the two frames",
    model: VideoModel | TranslatedVideoModel = "veo-3.1-fast-generate-001",
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
        model: VEO model (defaults to fast)
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

        # Resolve the client up front so GCS gating can see which backend is
        # in play (the Gemini API does not support GCS output).
        video_client = _client_for_video_model(app_ctx, model)
        is_vertex_client = getattr(video_client._api_client, "vertexai", False)
        # Resolved BEFORE the quote, as generate_video's already is. Left after
        # it, the dry run skipped the allowlist and backend checks the real
        # call applies — a quote for a destination the render would refuse.
        gcs_uri = _resolve_video_gcs(
            output_gcs_uri,
            app_ctx.video_gcs_bucket,
            app_ctx.allowed_gcs_buckets,
            is_vertex_client,
        )

        if dry_run:
            # Same rule the render applies, so the quote cannot price a call
            # the backend refuses.
            from .video import validate_render_options

            validate_render_options(
                model,
                backend=_backend_of(app_ctx, model),
            )

            # Refuse a local frame the real run would reject, so a quote does
            # not price a transition that cannot fetch its frames. After
            # validate_render_options so a backend refusal still takes
            # precedence (mirrors generate_bridge); remote frames (gs://) are
            # uncheckable offline and still price.
            _assert_local_source(first_frame_uri, data_dir, "first_frame_uri")
            _assert_local_source(last_frame_uri, data_dir, "last_frame_uri")
            return _respond(
                app_ctx,
                {
                    "dry_run": True,
                    "message": "Estimate only — nothing was generated",
                    "model": model,
                    "requested_duration_seconds": duration_seconds,
                    "duration_seconds": _snapped_duration_for_quote(
                        model, duration_seconds, "first_last_frame"
                    ),
                    "resolution": "720p",
                    "resolution_source": "fixed: this tool takes no resolution parameter",
                    "estimated_cost": _video_cost(
                        model,
                        duration_seconds,
                        resolution="720p",
                        include_audio=include_audio,
                        generation_mode="first_last_frame",
                    ),
                },
            )

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
                + _fetch_failure("first_frame_uri", first_frame_uri, data_dir)
                + "; "
                + _fetch_failure("last_frame_uri", last_frame_uri, data_dir)
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

        # Measured, like its sibling generate_bridge. Transition was left out
        # of the round that fixed bridge, and the comment written there — "the
        # one rendered path with neither a duration_source nor a
        # resolution_source" — was false as it was written: this path had the
        # same hole. Two tools with the same shape, one updated.
        # generate_transition takes no resolution parameter, but "the tool cannot
        # ask for anything else" is a fact about the tool, not about the file
        # that came back — and the cost line below prices whatever this says.
        facts = await _measure_rendered_video(
            result, resolution_key="resolution", requested_resolution=None
        )
        if facts.duration_source:
            result["duration_seconds"] = round(facts.duration_seconds or 0.0, 3)
            result["duration_source"] = facts.duration_source
        result.setdefault(
            "duration_source",
            "the snapped request: Veo renders exactly the length it is sent",
        )
        result.setdefault("resolution", "720p")
        result.setdefault(
            "resolution_source",
            facts.resolution_source
            or "assumed: generate_transition takes no resolution parameter and the "
            "render could not be measured",
        )

        # Cost from the measured length where there is one, else the snapped
        # duration the impl actually sent to Veo.
        cost = _video_cost(
            result.get("model", model),
            float(result.get("duration_seconds", duration_seconds)),
            resolution=str(result.get("resolution") or "720p"),
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
            # The model that actually ran: the impl normalizes an older
            # Gemini-API `-preview` spelling to its `-001` name.
            "model": result.get("model", model),
            "aspect_ratio": aspect_ratio,
            "duration_seconds": result.get("duration_seconds", duration_seconds),
            "duration_source": result.get("duration_source"),
            "resolution": result.get("resolution"),
            "resolution_source": result.get("resolution_source"),
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
            await _emit_warnings(ctx, warnings)
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

        return _respond(app_ctx, result)
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
    model: VideoModel | TranslatedVideoModel = "veo-3.1-fast-generate-001",
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

        # Resolve the client up front so GCS gating can see which backend is
        # in play (the Gemini API does not support GCS output).
        video_client = _client_for_video_model(app_ctx, model)
        is_vertex_client = getattr(video_client._api_client, "vertexai", False)
        # Resolved BEFORE the quote, as generate_video's already is. Left after
        # it, the dry run skipped the allowlist and backend checks the real
        # call applies — a quote for a destination the render would refuse.
        gcs_uri = _resolve_video_gcs(
            output_gcs_uri,
            app_ctx.video_gcs_bucket,
            app_ctx.allowed_gcs_buckets,
            is_vertex_client,
        )

        if dry_run:
            # Same rule the render applies, so the quote cannot price a call
            # the backend refuses.
            from .video import validate_render_options

            validate_render_options(
                model,
                backend=_backend_of(app_ctx, model),
            )

            # Refuse a local clip the real run would reject, so a quote does not
            # price a bridge that cannot render. After validate_render_options so
            # a backend refusal still takes precedence; remote clips
            # (gs://) are uncheckable offline and still price.
            _assert_local_source(from_clip_uri, data_dir, "from_clip_uri")
            _assert_local_source(to_clip_uri, data_dir, "to_clip_uri")

            # Report the environmental check positively. It runs above (and
            # refuses when ffmpeg is missing), but a passing check looks
            # identical to no check at all — which is exactly how a reviewer
            # on an ffmpeg-equipped host read it.
            preflight = ["ffmpeg available for frame decoding"]
            return _respond(
                app_ctx,
                {
                    "dry_run": True,
                    "message": "Estimate only — nothing was generated",
                    "model": model,
                    "requested_duration_seconds": duration_seconds,
                    "duration_seconds": _snapped_duration_for_quote(
                        model, duration_seconds, "first_last_frame"
                    ),
                    "resolution": "720p",
                    "resolution_source": "fixed: this tool takes no resolution parameter",
                    "preflight_checks": preflight,
                    "estimated_cost": _video_cost(
                        model,
                        duration_seconds,
                        resolution="720p",
                        include_audio=include_audio,
                        generation_mode="first_last_frame",
                    ),
                },
            )

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
                + _fetch_failure("from_clip_uri", from_clip_uri, data_dir)
                + "; "
                + _fetch_failure("to_clip_uri", to_clip_uri, data_dir)
            )

        await ctx.info("Extracting bridge frames")
        first_frame_png = await _decode_media(extract_frame_png, from_bytes, "end")
        last_frame_png = await _decode_media(extract_frame_png, to_bytes, "start")

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

        # Measured like every other local render. This comment used to call
        # bridge "the one rendered path" without a duration_source, which was
        # false as written — generate_transition beside it had the same hole
        # and kept it for two more rounds. Both now share one helper.
        # generate_bridge takes no resolution parameter, but "the tool cannot
        # ask for anything else" is a fact about the tool, not about the file
        # that came back — and the cost line below prices whatever this says.
        facts = await _measure_rendered_video(
            result, resolution_key="resolution", requested_resolution=None
        )
        if facts.duration_source:
            result["duration_seconds"] = round(facts.duration_seconds or 0.0, 3)
            result["duration_source"] = facts.duration_source
        result.setdefault(
            "duration_source",
            "the snapped request: Veo renders exactly the length it is sent",
        )
        result.setdefault("resolution", "720p")
        result.setdefault(
            "resolution_source",
            facts.resolution_source
            or "assumed: generate_bridge takes no resolution parameter and the "
            "render could not be measured",
        )
        result["backend"] = "vertex" if is_vertex_client else "gemini_api"

        # Cost from the measured length where there is one, else the snapped
        # duration the impl actually sent to Veo.
        cost = _video_cost(
            result.get("model", model),
            float(result.get("duration_seconds", duration_seconds)),
            resolution=str(result.get("resolution") or "720p"),
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
            # The model that actually ran: the impl normalizes an older
            # Gemini-API `-preview` spelling to its `-001` name.
            "model": result.get("model", model),
            "aspect_ratio": aspect_ratio,
            "duration_seconds": result.get("duration_seconds", duration_seconds),
            "duration_source": result.get("duration_source"),
            "resolution": result.get("resolution"),
            "resolution_source": result.get("resolution_source"),
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
            await _emit_warnings(ctx, warnings)
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

        return _respond(app_ctx, result)
    except Exception as e:
        await ctx.error(f"Bridge generation failed: {e}")
        logger.exception("Tool error")
        return json.dumps({"error": str(e)})


@mcp.tool()
async def generate_clip(
    ctx: Context[ServerSession, AppContext],
    beats: list[dict[str, Any]],
    aspect_ratio: str = "9:16",
    model: VideoModel | TranslatedVideoModel = "veo-3.1-fast-generate-001",
    include_audio: bool | None = None,
    add_bridges: bool = False,
    output_gcs_uri: str | None = None,
    animatic: bool = False,
    animatic_resolution: str | None = None,
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
            falling back to text-to-video). Any other key is an error naming
            what you probably meant — the still to condition on is
            `first_frame_uri`, not `image_url` (what generate_storyboard
            returns) or `image_uri` (what generate_video calls it), and a
            dropped key would have cost a full render. `caption` and `notes`
            are the exception: they are accepted and ignored, so a
            generate_storyboard shot list can be passed straight through.
        aspect_ratio: Default 9:16 for vertical social clips.
        model: VEO model applied to every beat.
        include_audio: Enable audio on each Veo beat (default true; only
            effective on Vertex). An animatic's Omni audio has no switch, so
            passing it with animatic=True is ignored with a warning.
        add_bridges: Generate a 4s bridge clip between consecutive beats,
            on Veo, or on Omni 1.2 in an animatic. Bridges require local
            (file://) beat outputs; skipped when beats land in GCS.
        output_gcs_uri: GCS URI for all outputs (optional).
        dry_run: When True, price the whole reel — every beat, plus every
            bridge when add_bridges is set — and generate nothing. The single
            most useful pre-flight in the server: a clip is the most expensive
            call it can make.
        animatic: When True, render every beat on the omni draft model
            instead of Veo, for a quick storyboard preview of the whole reel
            before committing to full Veo renders. 720p unless
            animatic_resolution says otherwise.
        animatic_resolution: Resolution for the animatic pass — "360p",
            "720p" (the default), "1080p" or "4K". Naming one renders the
            animatic on gemini-omni-1.2-flash, which has a resolution
            parameter. "360p" is about a THIRD of the 720p price, which is
            what turns the preview from roughly the cost of the delivery
            render into a real saving: a 5-beat 8s reel is ~$2.79 at 720p and
            ~$0.93 at 360p. Ignored unless animatic is True. With add_bridges,
            each bridge renders 4s on Omni 1.2 between the beats' frames, at
            the animatic's resolution. Veo-only per-beat controls (seed,
            negative_prompt) are ignored. Each beat
            is measured from the rendered file, so an animatic's reported cost
            is metered: it lands about a frame per beat above the *nominal*
            request (omni overshoots the clamped duration), but the dry-run
            quote already includes that per-beat allowance, so the metered
            total comes in at or under the quote. Beats delivered to GCS
            cannot be measured, and the total then says so by reporting itself
            as an estimate.

    Returns:
        JSON clip manifest:
        {
          "kind": "clip",
          "aspect_ratio": "9:16",
          "model": "<the model every beat rendered on>",
          "animatic": false,
          "segments": [ ... ordered beat / bridge segments ... ],
          "total_duration_seconds": <sum>,
          "beat_count": <len(beats)>,
          "cost": { ... total of the segments that rendered ... },
          "errors": [ {"beat_index": N, "error": "..."} ],  // only on failure
          "warnings": [ ... ],                              // only when non-empty
        }

        A dry run instead returns {"dry_run": true, "message", "model",
        "beat_count", "bridge_count", "preflight_checks", "estimated_cost"} —
        the price of every beat plus every bridge, and nothing is generated.
    """
    # Unset means audio on for Veo. Kept apart from the value so an animatic
    # warns only about an include_audio the caller passed: the True default
    # drew "include_audio is ignored" on every animatic.
    include_audio_passed = include_audio is not None
    include_audio = True if include_audio is None else include_audio
    # Accumulators live outside the try so the cancellation handler below can
    # still see what was rendered — and billed — before the caller vanished.
    segments: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    clip_warnings: list[str] = []
    total_duration = 0.0
    try:
        # Inside the try, and only when it applies. This sat above the try and
        # ran unconditionally, so generate_clip(animatic=False,
        # animatic_resolution="9000p") raised ValueError straight out of the
        # tool instead of returning the {"error": ...} body every other failure
        # here returns — on a parameter the docstring calls "Ignored unless
        # animatic is True".
        if animatic:
            animatic_model, animatic_res = _omni_preview_model(animatic_resolution)
            beat_model = animatic_model
        else:
            # Unused off the animatic path; bound so every use is.
            animatic_model = DEFAULT_OMNI_MODEL
            animatic_res = None
            beat_model = model
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

        # Validate every statically checkable field of every beat before
        # rendering any of them. Anything left to the impl surfaces mid-run —
        # a bad seed in beat 3 failed only after beats 1-2 had rendered and
        # billed — and nothing here needs the network to decide.
        for beat_index, beat_spec in enumerate(beats):
            if not isinstance(beat_spec, dict):
                raise ValueError(
                    f"beats[{beat_index}] must be an object with a 'prompt', "
                    f"got {type(beat_spec).__name__}"
                )
            _validate_spec_keys(
                beat_spec,
                f"beats[{beat_index}]",
                _BEAT_KEYS,
                _BEAT_KEY_SUGGESTIONS,
                ignored=_BEAT_IGNORED_KEYS,
            )
            if not str(beat_spec.get("prompt", "") or "").strip():
                raise ValueError(f"beats[{beat_index}] is missing a non-empty 'prompt'")
            _validate_duration_seconds(
                beat_spec.get("duration_seconds"),
                f"beats[{beat_index}].duration_seconds",
            )
            _validate_seed(beat_spec.get("seed"), f"beats[{beat_index}].seed")
            for text_field in ("prompt", "negative_prompt", "audio_prompt"):
                value = beat_spec.get(text_field)
                if value is not None and not isinstance(value, str):
                    raise ValueError(
                        f"beats[{beat_index}].{text_field} must be a string, "
                        f"got {type(value).__name__}"
                    )
            first_frame = beat_spec.get("first_frame_uri")
            if first_frame is not None:
                if not isinstance(first_frame, str):
                    raise ValueError(
                        f"beats[{beat_index}].first_frame_uri must be a string "
                        f"URI (gs://, http(s)://, file://), got "
                        f"{type(first_frame).__name__}"
                    )
                # An empty string is falsy, so the beat would silently render
                # text-to-video at full price instead of conditioning on the
                # frame the caller believes they supplied.
                if not first_frame.strip():
                    raise ValueError(
                        f"beats[{beat_index}].first_frame_uri is empty; omit the "
                        "key entirely for a text-to-video beat"
                    )
                # On a dry run, refuse a local frame the real run's per-beat
                # fetch would reject, so a quote does not price a beat that
                # cannot render. Guarded to dry_run: a real run keeps its
                # per-beat failure (recorded in errors, clip continues) for a
                # source only discoverable at fetch time, while a local path is
                # decidable now and a quote must refuse it up front.
                if dry_run:
                    _assert_local_source(
                        first_frame, data_dir, f"beats[{beat_index}].first_frame_uri"
                    )

        # Validate the clip-level aspect ratio once, up front. Otherwise the
        # impl's per-value ValueError fires inside every beat's error handler,
        # producing a success-shaped manifest with zero segments instead of a
        # clear top-level failure.
        _validate_aspect_ratio(aspect_ratio)

        preflight: list[str] = []
        if add_bridges:
            # Bridges decode frames out of the rendered beats; fail before any
            # beat is billed rather than losing every bridge mid-run. This is
            # the composite where a missing binary hurts most — it prices
            # bridges into the estimate — so the check is reported positively
            # too, exactly as generate_bridge does.
            assert_frame_decoding_available()
            if not animatic:
                # A bridge IS a first/last-frame render, so this composite
                # inherits the backend restriction its standalone sibling
                # already enforces. Without this the quote came back clean on
                # the Gemini API, the beats rendered and billed ~$0.80, and
                # only then did the first bridge fail — a partial-spend trap
                # made worse by a pre-flight that reassured the caller first.
                # Checked here, before any beat. An animatic's bridges render
                # on Omni 1.2, which interpolates on both backends.
                from .video import validate_render_options

                validate_render_options(
                    model,
                    backend=_backend_of(app_ctx, str(model)),
                )
            preflight.append("ffmpeg available for frame decoding (bridges)")

        if not animatic:
            # The routing rule the render applies below, asked before the
            # quote too, so a dry run cannot price beats no client can serve
            # (Veo on a Gemini-API deployment).
            _client_for_video_model(app_ctx, model)

        # An animatic renders every beat on omni, so omni's inline-delivery
        # policy applies to each beat -- judged here, before any beat renders
        # and on the quote alike, so a refusal cannot land after earlier
        # beats have billed. An animatic has no Cloud Storage destination.
        animatic_delivery: list[str] = []
        if animatic:
            animatic_spec = omni_spec(_omni_preview_model(animatic_resolution)[0])
            animatic_on_vertex = _omni_backend_is_vertex(
                app_ctx, requested=_omni_server_backend(app_ctx)
            )
            for beat_spec in beats:
                beat_delivery = omni_inline_delivery_policy(
                    animatic_spec,
                    vertexai=animatic_on_vertex,
                    output_gcs_uri=None,
                    resolution=animatic_res,
                    task=None,
                    output_seconds=_animatic_beat_seconds(
                        beat_spec, model, animatic_spec
                    ),
                    takes_output_gcs_uri=False,
                )
                if beat_delivery and beat_delivery not in animatic_delivery:
                    animatic_delivery.append(beat_delivery)
            changed = [
                f"beat {idx + 1} {_beat_seconds(b):g}s -> "
                f"{_animatic_beat_seconds(b, model, animatic_spec)}s"
                for idx, b in enumerate(beats)
                if _animatic_beat_seconds(b, model, animatic_spec) != _beat_seconds(b)
            ]
            if changed:
                animatic_delivery.append(
                    "Animatic beats render as long as the Veo delivery will "
                    f"render them ({model} renders 4, 6 or 8 seconds): "
                    + ", ".join(changed)
                    + "."
                )

        if dry_run:
            est_model = animatic_model if animatic else model
            beat_costs = [
                _video_cost_estimate(
                    est_model,
                    float(_animatic_beat_seconds(b, model, animatic_spec))
                    if animatic
                    else _beat_seconds(b),
                    resolution=(animatic_res or "720p") if animatic else "720p",
                    include_audio=include_audio and not animatic,
                )
                for b in beats
            ]
            bridge_count = len(beats) - 1 if add_bridges else 0
            bridge_costs = [
                _video_cost_estimate(
                    est_model,
                    _CLIP_BRIDGE_SECONDS,
                    resolution=(animatic_res or "720p") if animatic else "720p",
                    include_audio=False,
                )
                for _ in range(bridge_count)
            ]
            try:
                from .pricing import sum_costs

                total = _cost_payload(
                    sum_costs(beat_costs + bridge_costs, label="clip")
                )
            except Exception:  # pragma: no cover - defensive
                total = None
            dry_payload: dict[str, Any] = {
                "dry_run": True,
                "message": "Estimate only — nothing was generated",
                "model": est_model,
                "beat_count": len(beats),
                "bridge_count": bridge_count,
                "preflight_checks": preflight,
                "estimated_cost": total,
            }
            # Disclose the same warnings a real animatic run would emit for
            # these inputs, so a quote reveals the ignored inputs (add_bridges,
            # output_gcs_uri, include_audio, Veo-only per-beat params) before
            # anything is spent — the quote used to return no warnings at all.
            if animatic:
                dry_warnings = _animatic_mode_warnings(
                    add_bridges=add_bridges,
                    output_gcs_uri=output_gcs_uri,
                    include_audio_passed=include_audio_passed,
                )
                for beat_index, beat_spec in enumerate(beats):
                    beat_warning = _animatic_beat_dropped_warning(beat_spec, beat_index)
                    if beat_warning:
                        dry_warnings.append(beat_warning)
                dry_warnings.extend(animatic_delivery)
                if dry_warnings:
                    dry_payload["warnings"] = dry_warnings
                    await _emit_warnings(ctx, dry_warnings)
            return _respond(app_ctx, dry_payload)

        # Resolve the client ONCE, before the beat loop. If the model can't be
        # routed (e.g. Veo on a Gemini-API deployment, or omni with no Gemini
        # API access), let the RuntimeError propagate to the
        # outer handler so the tool returns a top-level {"error": ...} rather
        # than failing every beat individually and returning a success-shaped
        # clip with empty segments.
        if animatic:
            # Fast omni preview path: no Veo GCS, no bridges. video_client is a
            # harmless placeholder (never used — beats route to omni below).
            # Fail fast if omni is unavailable.
            _client_for_omni(app_ctx, requested=_omni_server_backend(app_ctx))
            video_client = app_ctx.client
            gcs_uri = None
            clip_warnings.extend(
                _animatic_mode_warnings(
                    add_bridges=add_bridges,
                    output_gcs_uri=output_gcs_uri,
                    include_audio_passed=include_audio_passed,
                )
            )
            clip_warnings.extend(animatic_delivery)
        else:
            video_client = _client_for_video_model(app_ctx, model)
            is_vertex_client = getattr(video_client._api_client, "vertexai", False)
            gcs_uri = _resolve_video_gcs(
                output_gcs_uri,
                app_ctx.video_gcs_bucket,
                app_ctx.allowed_gcs_buckets,
                is_vertex_client,
            )

        prev_video_bytes: bytes | None = None

        for idx, beat in enumerate(beats):
            try:
                prompt = beat.get("prompt")
                if not prompt:
                    raise ValueError(f"beat {idx} missing required 'prompt'")
                duration = (
                    float(_animatic_beat_seconds(beat, model, animatic_spec))
                    if animatic
                    else _beat_seconds(beat)
                )
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
                            _fetch_failure("first_frame_uri", first_frame_uri, data_dir)
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
                    dropped_warning = _animatic_beat_dropped_warning(beat, idx)
                    if dropped_warning:
                        clip_warnings.append(dropped_warning)
                    omni_client = _client_for_omni(
                        app_ctx, requested=_omni_server_backend(app_ctx)
                    )
                    beat_result = await generate_video_omni_impl(
                        client=omni_client,
                        prompt=beat_prompt,
                        videos_dir=app_ctx.videos_dir,
                        model=animatic_model,
                        image_bytes_list=[image_bytes] if image_bytes else None,
                        aspect_ratio=aspect_ratio,
                        duration_seconds=duration,
                        resolution=animatic_res,
                        allow_uri_delivery=not getattr(
                            omni_client._api_client, "vertexai", False
                        ),
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
            # Veo renders EXACTLY the length it is sent — the same fact
            # _segment_is_metered relies on to price a Veo segment without a
            # probe — so its duration has a provenance, it just is not a
            # measurement. Leaving the field off made a Veo segment the one
            # place in the response where a number carried no source at all.
            beat_duration_source: str | None = (
                None
                if animatic
                else "the snapped request: Veo renders exactly the length it is sent"
            )
            beat_resolution_source: str | None = None
            beat_dimensions: list[int] | None = None
            beat_resolution = (
                str(beat_result.get("rendered_resolution") or "720p")
                if animatic
                else "720p"
            )
            if animatic:
                beat_url_now = beat_result.get("video_url") or ""
                if isinstance(beat_url_now, str) and beat_url_now.startswith("file://"):
                    beat_path = Path(beat_url_now[7:])
                    measured_beat = await _probe_media(
                        measure_video_duration, beat_path
                    )
                    if measured_beat is not None:
                        beat_duration = measured_beat
                        beat_duration_source = "measured from the rendered video"
                    # The resolution needs measuring for the same reason the
                    # duration does, and more so here: a clip renders many
                    # files in one call, so an unverified echo is the cheapest
                    # place for a silent resolution regression to hide — and
                    # the per-second rate differs threefold across omni's
                    # tiers, so every beat would be mispriced together.
                    size = await _probe_media(measure_video_dimensions, beat_path)
                    if size is not None:
                        beat_dimensions = list(size)
                        classified = classify_video_resolution(size)
                        if classified is not None:
                            if classified != beat_resolution:
                                clip_warnings.append(
                                    f"Beat {idx} requested {beat_resolution} but "
                                    f"rendered {classified} ({size[0]}x{size[1]}); "
                                    "the cost prices what rendered."
                                )
                            beat_resolution = classified
                            beat_resolution_source = "measured from the rendered video"

            beat_manifest = {
                "kind": "beat",
                "index": idx,
                "prompt": prompt,
                "model": beat_model,
                # generate_clip takes no resolution parameter, so a Veo beat
                # is always 720p; an animatic beat is whatever omni MEASURED.
                "resolution": beat_resolution,
                "resolution_source": beat_resolution_source
                or (
                    "the request: the rendered file could not be opened to measure it"
                    if animatic
                    else "fixed: generate_clip takes no resolution parameter"
                ),
                **(
                    {"rendered_dimensions": beat_dimensions}
                    if beat_dimensions is not None
                    else {}
                ),
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
            # a negative seed that was not sent) in the beat
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
                        end_frame = await _decode_media(
                            extract_frame_png, prev_video_bytes, "end"
                        )
                        start_frame = await _decode_media(
                            extract_frame_png, cur_bytes, "start"
                        )
                        await ctx.info(f"Generating bridge before beat {idx + 1}")
                        if animatic:
                            # Omni 1.2 interpolates between a first and a last
                            # frame, so the preview bridges like the clip will.
                            omni_client = _client_for_omni(
                                app_ctx, requested=_omni_server_backend(app_ctx)
                            )
                            bridge_result = await generate_video_omni_impl(
                                client=omni_client,
                                prompt=_CLIP_BRIDGE_PROMPT,
                                videos_dir=app_ctx.videos_dir,
                                model=animatic_model,
                                first_frame_bytes=end_frame,
                                last_frame_bytes=start_frame,
                                aspect_ratio=aspect_ratio,
                                duration_seconds=_CLIP_BRIDGE_SECONDS,
                                resolution=animatic_res,
                                allow_uri_delivery=not getattr(
                                    omni_client._api_client, "vertexai", False
                                ),
                                log_callback=ctx.info,
                            )
                        else:
                            bridge_result = await generate_video_impl(
                                client=video_client,
                                prompt=_CLIP_BRIDGE_PROMPT,
                                videos_dir=app_ctx.videos_dir,
                                model=model,
                                image_bytes=end_frame,
                                last_frame_bytes=start_frame,
                                allowed_dir=data_dir,
                                aspect_ratio=aspect_ratio,
                                duration_seconds=_CLIP_BRIDGE_SECONDS,
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
                        # The same provenance a Veo BEAT carries, under the same
                        # policy: Veo renders exactly the length it is sent, so
                        # the request is the fact and nothing is measured (only
                        # omni beats are, because there the length is the
                        # service's choice). In-clip bridges had a bare
                        # `duration_seconds: 4` and none of resolution,
                        # resolution_source or duration_source -- flagged in
                        # live testing more than once -- while beats beside
                        # them carried all three.
                        bridge_manifest = {
                            "kind": "bridge",
                            "between_beats": [idx - 1, idx],
                            "model": bridge_result.get("model", model),
                            "aspect_ratio": aspect_ratio,
                            "duration_seconds": float(
                                bridge_result.get("duration_seconds")
                                or _CLIP_BRIDGE_SECONDS
                            ),
                            "duration_source": (
                                "the fixed 4s request: Veo renders exactly the "
                                "length it is sent"
                            ),
                            "resolution": "720p",
                            "resolution_source": (
                                "fixed: generate_clip takes no resolution parameter"
                            ),
                            "video_url": bridge_result.get("video_url"),
                        }
                        if animatic:
                            # An Omni render runs marginally over its request,
                            # so, like an animatic beat, the file is measured
                            # and priced as measured.
                            bridge_manifest.update(
                                await _measured_omni_segment(
                                    bridge_result,
                                    model=animatic_model,
                                    requested_seconds=_CLIP_BRIDGE_SECONDS,
                                    requested_resolution=animatic_res,
                                )
                            )
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

        clip_manifest = _assemble_clip_manifest(
            aspect_ratio=aspect_ratio,
            beat_model=beat_model,
            animatic=animatic,
            segments=segments,
            total_duration=total_duration,
            beat_count=len(beats),
            include_audio=include_audio,
            errors=errors,
            warnings=clip_warnings,
        )
        # Mirror the aggregated warnings (animatic drops, per-beat ignored Veo
        # params, GCS notices) onto the notification channel, not just the JSON
        # manifest — an agent watching the stream saw none of them before.
        await _emit_warnings(ctx, clip_warnings)
        return _respond(app_ctx, clip_manifest)
    except asyncio.CancelledError:
        # A cancellation is a BaseException, so the handler below never saw
        # it: a client that disconnected after five beats had rendered took
        # the clip-level total — the only record of what the run cost — with
        # it. Each beat's own sidecar survives; this persists the clip
        # manifest so the money already spent stays accounted for. The
        # cancellation is re-raised, never swallowed.
        if segments:
            partial = _assemble_clip_manifest(
                aspect_ratio=aspect_ratio,
                beat_model=beat_model,
                animatic=animatic,
                segments=segments,
                total_duration=total_duration,
                beat_count=len(beats),
                include_audio=include_audio,
                errors=errors,
                warnings=clip_warnings,
            )
            partial["cancelled"] = True
            manifest_url = _write_clip_manifest(partial)
            logger.warning(
                "generate_clip cancelled after %d segment(s); manifest written to %s",
                len(segments),
                manifest_url or "nowhere (no local segment to write beside)",
            )
        raise
    except Exception as e:
        await ctx.error(f"Clip generation failed: {e}")
        logger.exception("Tool error")
        return json.dumps({"error": str(e)})


@mcp.tool()
async def generate_video_omni(
    ctx: Context[ServerSession, AppContext],
    prompt: str,
    omni_model: str | None = None,
    omni_backend: str | None = None,
    image_uris: list[str] | None = None,
    first_frame_uri: str | None = None,
    last_frame_uri: str | None = None,
    reference_image_uris: list[str] | None = None,
    reference_video_uris: list[str] | None = None,
    input_video_uri: str | None = None,
    aspect_ratio: str = "16:9",
    duration_seconds: float = 6.0,
    resolution: str | None = None,
    previous_interaction_id: str | None = None,
    output_gcs_uri: str | None = None,
    timeout_seconds: int = OMNI_DEFAULT_TIMEOUT_SECONDS,
    dry_run: bool = False,
    director_notes: str | None = None,
    max_retakes: int = DEFAULT_MAX_RETAKES,
) -> str:
    """Generate or edit video with Gemini Omni 1.2 Flash (default).

    New video supports whole-second durations from 3 to 15, 16:9/9:16,
    native dialogue/music/effects and 360p/720p/1080p/4K output. Describe camera
    moves, speaker assignments, timing and audio in the prompt. Audio uploads
    and audio references are unsupported by the API.

    image_uris lets the model infer image roles. Alternatively use
    first_frame_uri, last_frame_uri and reference_image_uris for explicit
    roles. Last frame requires first frame; interpolation stays within 3-10s.
    Up to 6 reference images and up to 3 video references (3s each) are
    supported. An 11-15s generation accepts at most 1 video reference.

    input_video_uri edits an uploaded video (up to 15s on 1.2).
    previous_interaction_id continues an existing conversation on the same
    model/backend. Edits preserve the source context and accept reference
    images; video references work only for new generation. On Vertex AI an
    edit carries at most 2 images in all (image_uris, first/last frames and
    reference_image_uris together). Put aspect-ratio changes in the edit
    prompt. duration_seconds and aspect_ratio are omitted on edits, so the
    duration ranges do not apply there; output duration is measured and
    dry_run quotes an upper bound.

    resolution defaults to 720p (a continuation of a recorded interaction
    keeps that clip's resolution); 1080p/4K include upsampling. For a separate
    upscale of an approved draft use upscale_video_omni on Vertex AI.
    omni_backend is auto, vertex or gemini_api; auto prefers GEMINI_API_KEY.
    output_gcs_uri needs Vertex and an allowed bucket. On Vertex, an output
    that can exceed the 4 MB inline limit (1080p/4K, over 10s, or an edit,
    which can run 15s) is returned inline when output_gcs_uri is not given, with a warning
    on the dry run and the render alike; pass output_gcs_uri to be safe.
    timeout_seconds covers the entire interaction; on timeout, retain
    interaction_id to recover it.
    dry_run validates local sources and returns an estimate without rendering.

    omni_model left unset continues a previous_interaction_id on the model
    that created it and otherwise uses gemini-omni-1.2-flash; naming a model
    other than the one an interaction was created on is refused.
    gemini-omni-1.1-flash is retired here: a call naming it runs on 1.2 with a
    warning, and an interaction created on it cannot be continued -- pass its
    saved video as input_video_uri instead.
    The original gemini-omni-flash-preview endpoint has been switched off.

    Args:
        ctx: MCP context with application state.
        prompt: What to render, or on an edit only what should change. Role
            tags you write yourself (`<FIRST_FRAME>`, `<IMAGE_REF_0>`,
            `[# Sources ...]`) pass through untouched; otherwise they are
            generated from the media arguments and echoed back as
            `effective_prompt`. Timecodes ("[0-3s] a person walks") and
            audio direction work in plain language.
        omni_model: Unset continues a previous_interaction_id on the model
            that created it, and otherwise uses gemini-omni-1.2-flash.
        omni_backend: "auto", "vertex" or "gemini_api". Unset, the server's
            OMNI_BACKEND applies. "auto" uses the Gemini API whenever
            GEMINI_API_KEY is set, even on a Vertex AI server -- a different
            project, billing path and quota pool -- so a deployment that must
            keep Omni traffic on Vertex AI passes "vertex" or sets
            OMNI_BACKEND=vertex. The response reports `backend` and
            `backend_reason`. A choice that contradicts the backend a
            previous_interaction_id was created on is refused.
        image_uris: Images whose role the model infers: one opens the clip,
            several are references (gs://, http(s)://, or file:// inside
            DATA_FOLDER). Cannot be combined with the explicit-role
            arguments below.
        first_frame_uri: The image the clip opens on.
        last_frame_uri: The image the clip ends on. Needs first_frame_uri,
            and the pair keeps a new render within 3-10s; the same image for
            both makes a loop.
        reference_image_uris: Up to 6 subject or style references, bound to
            `<IMAGE_REF_0>`, `<IMAGE_REF_1>`, ... in order.
        reference_video_uris: Up to 3 likeness clips of up to 3s each, bound
            to `<VIDEO_REF_0>`, ...; at most 1 on an 11-15s render, and none
            on an edit. Audio in a reference clip is ignored.
        input_video_uri: A clip of your own to edit, up to 15s. Any images
            given with it guide the edit; they do not make it a new render.
            Editing an uploaded video is not available in the European
            Economic Area (Google's restriction): the service refuses, and
            this server cannot check it first.
        aspect_ratio: "16:9" (default) or "9:16". Not sent on an edit.
        duration_seconds: Whole seconds from 3 to 15 (default 6). Not sent
            on an edit, whose length the service picks.
        resolution: "360p", "720p" (the default), "1080p" or "4K"; 1080p
            and 4K are upsampled from the base render. Unset with
            previous_interaction_id, the resolution recorded for that
            interaction.
        previous_interaction_id: Continue an interaction this server
            created, to edit its video (see also edit_video).
        output_gcs_uri: A gs:// prefix ending in / (Vertex AI writes a new
            object under it), Vertex AI only, checked against the bucket
            allowlist (GCS_ALLOWED_BUCKETS / VIDEO_GCS_BUCKET).
        timeout_seconds: Deadline for the whole interaction, default 210s.
            It sits under the roughly 4-minute ceiling many MCP hosts put on
            one tool call, because a render that outlives the host is billed
            with no result and no interaction_id; under it, a timeout names
            the interaction_id, and get_omni_interaction collects the render.
            A 4K output likely to outlast it is warned about before it
            renders. A request refused with a 429 is sent again, at most
            twice, while the deadline leaves room.
        dry_run: Return the cost estimate and warnings, and render nothing.
        director_notes: Optional. What the clip must get right, in plain
            words ("the mug turns left to right; the logo stays readable; no
            hands enter the frame"). Gemini 4 Argon turns the notes into
            checks, reviews the clip against them, sampling it at 4 frames a
            second, and when a check fails renders again with fix notes, up
            to max_retakes times. The response then carries a `director`
            block with each check's state on every take, and `cost` covers
            every take and every director call. Omit it and the tool
            behaves exactly as it does without a director: no extra call and
            no extra cost.
            In the notes, call first_frame_uri "the first frame",
            last_frame_uri "the last frame", image_uris[0] "input image 1"
            and reference_image_uris[0] "reference 1" (or <IMAGE_REF_0>) and
            so on; the reviewer is shown the frames and the first three of
            each list. Put text that must appear in quotes, exactly as it
            should be set: it is compared character for character. Say
            "throughout" or "the whole clip" for what must hold at every
            moment ("steam rises throughout"): it is judged one second at a
            time.
        max_retakes: Re-renders the director may spend when a check fails, 0
            to 2 (default 1). A dry run with director_notes quotes the worst
            case: every take rendered and reviewed. A retake the service
            turns away with a 429 or a server error is tried again after
            10 s, then 30 s, or the wait the 429 asked for, while the call's
            time allows. With director_notes, timeout_seconds bounds the
            whole call: each take renders in the time left, and a review or
            retake that could not finish in time is skipped and reported.

    Returns:
        JSON with video_url, interaction_id (pass it to edit_video,
        extend_video_omni, upscale_video_omni or this tool to keep going),
        cost, backend, and any warnings.
    """
    if director_notes is not None:
        return await _directed(
            ctx,
            render=lambda take_prompt, quote, time_left=None: generate_video_omni(
                ctx=ctx,
                prompt=take_prompt,
                omni_model=omni_model,
                omni_backend=omni_backend,
                image_uris=image_uris,
                first_frame_uri=first_frame_uri,
                last_frame_uri=last_frame_uri,
                reference_image_uris=reference_image_uris,
                reference_video_uris=reference_video_uris,
                input_video_uri=input_video_uri,
                aspect_ratio=aspect_ratio,
                duration_seconds=duration_seconds,
                resolution=resolution,
                previous_interaction_id=previous_interaction_id,
                output_gcs_uri=output_gcs_uri,
                timeout_seconds=(
                    timeout_seconds if time_left is None else max(1, int(time_left))
                ),
                dry_run=quote,
            ),
            returns_content=False,
            media_kind="video",
            prompt=prompt,
            notes=director_notes,
            max_retakes=max_retakes,
            dry_run=dry_run,
            duration_s=duration_seconds,
            references=_review_references(
                ("first_frame", "first_frame_uri", first_frame_uri),
                ("last_frame", "last_frame_uri", last_frame_uri),
                *(
                    ("input", f"image_uris[{i}]", uri)
                    for i, uri in enumerate(image_uris or [])
                ),
                *(
                    ("reference", f"reference_image_uris[{i}]", uri)
                    for i, uri in enumerate(reference_image_uris or [])
                ),
                omni=True,
            ),
            unshown_inputs=[
                what
                for what, given in (
                    ("the reference clips", reference_video_uris),
                    ("the input video", input_video_uri),
                )
                if given
            ],
            deadline_s=timeout_seconds,
        )
    # How far the render got, for the error body if it fails.
    omni_submission = OmniSubmission()
    try:
        app_ctx = ctx.request_context.lifespan_context
        # Which backend serves this call, decided once and reported; see
        # _omni_backend_decision.
        # Read once: the backend, the model, the source length and the
        # render all come from this one record.
        prior = await _prior_interaction_or_empty(
            app_ctx.videos_dir, previous_interaction_id
        )
        omni_backend_resolved, omni_backend_reason = await _resolve_omni_backend(
            app_ctx,
            omni_backend,
            need_gcs=bool(output_gcs_uri),
            previous_interaction_id=previous_interaction_id,
            prior=prior,
        )
        # Every pre-flight runs before any fetch or interaction work: the omni
        # impl clamps to the model's range ([3, 15]s on 1.2), which would turn a
        # negative or NaN duration
        # into a billed 3s render instead of an error, and a capability the
        # chosen model lacks must fail here rather than after the spend.
        _validate_duration_seconds(duration_seconds)
        # src/omni.py refuses anything but 16:9 / 9:16, so quoting one was
        # the quote accepting what the run refuses — every other video tool
        # front-loads this check and these two did not.
        _validate_aspect_ratio(aspect_ratio)
        requested_model = _continuation_omni_model(omni_model, prior)
        spec = _validate_omni_model(requested_model)
        _validate_omni_continuation_source(spec, prior, previous_interaction_id)
        normalized_resolution, resolution_origin = _omni_sent_resolution(
            spec,
            _validate_omni_resolution(spec, resolution),
            prior,
            previous_interaction_id,
        )
        # Evaluated once: it validates (and logs) the source, and its answer
        # decides both the dry run's checks and what the render sends.
        source_uri = _omni_gcs_source(
            app_ctx, input_video_uri, omni_backend_resolved, model=spec.model
        )
        reference_images = list(reference_image_uris or [])
        reference_videos = list(reference_video_uris or [])
        inferred_images = list(image_uris or [])
        # The same rules the impl enforces on bytes, run here on counts, so a
        # dry_run never quotes a request the real call would refuse -- and
        # never refuses one it would run. A duration the request does not
        # send (an edit's) is not checked, exactly as in the impl.
        sends_duration = omni_duration_is_sent(
            spec.model,
            previous_interaction_id=previous_interaction_id,
            has_input_video=bool(input_video_uri),
            has_first_frame=first_frame_uri is not None,
            has_last_frame=last_frame_uri is not None,
            reference_image_count=len(reference_images),
            reference_video_count=len(reference_videos),
            inferred_image_count=len(inferred_images),
        )
        validate_omni_request(
            spec.model,
            resolution=resolution,
            has_first_frame=first_frame_uri is not None,
            has_last_frame=last_frame_uri is not None,
            reference_image_count=len(reference_images),
            reference_video_count=len(reference_videos),
            inferred_image_count=len(inferred_images),
            duration_seconds=duration_seconds if sends_duration else None,
            is_continuation=bool(previous_interaction_id or input_video_uri),
            vertexai=omni_backend_resolved == "vertex",
        )
        # Cap the images: every one is buffered in memory (each up to
        # MAX_FETCH_BYTES), and omni conditions on a small set.
        total_images = (
            len(inferred_images)
            + len(reference_images)
            + (first_frame_uri is not None)
            + (last_frame_uri is not None)
        )
        if total_images > MAX_OMNI_INPUT_IMAGES:
            raise ValueError(
                f"Too many input images ({total_images}); {spec.model} accepts "
                f"at most {MAX_OMNI_INPUT_IMAGES} here."
            )

        # Every gs:// URI this call reads or writes, not only the output: an
        # input read with ambient credentials is the same exposure.
        gcs_warnings: list[str] = _gcs_allowlist_warnings(
            app_ctx,
            input_video_uri,
            *inferred_images,
            first_frame_uri,
            last_frame_uri,
            *reference_images,
            *reference_videos,
            output_gcs_uri,
        )
        gcs_warnings.extend(_gcs_prefix_warning(output_gcs_uri))
        if output_gcs_uri:
            bucket = _parse_gcs_bucket(output_gcs_uri)
            if bucket is None:
                raise ValueError(
                    f"output_gcs_uri must start with gs://: {output_gcs_uri}"
                )
            # Raises for a bucket outside a configured allowlist; the
            # no-allowlist warning it returns is already in gcs_warnings.
            _assert_gcs_bucket_allowed(bucket, app_ctx.allowed_gcs_buckets)
            if omni_backend_resolved == "gemini_api":
                # The docstring promised this notice and only the allowlist
                # warning ever fired, so a caller's destination was dropped in
                # silence. Omni prefers the Gemini API whenever a key is
                # configured, which is exactly when this bites.
                gcs_warnings.append(
                    "output_gcs_uri is ignored: this call runs on the Gemini "
                    "Developer API, which has no Cloud Storage destination; a "
                    "small output comes back inline and a large one as a "
                    "Google-hosted file this server downloads. Pass "
                    'omni_backend="vertex" (or set '
                    "OMNI_BACKEND=vertex) on a Vertex server to deliver to "
                    "Cloud Storage."
                )
        # A retired model the caller named runs on its replacement. Said on
        # the quote and the render alike: the impl is handed the replacement,
        # so it has nothing left to disclose.
        reroute = omni_model_reroute_warning(requested_model)
        if reroute:
            gcs_warnings.append(reroute)
        # Large output is judged on the length the quote below prices: an
        # edit's is the service's choice, so its upper bound; a fresh render
        # honours the clamped request, mirrored from the impl's clamp so the
        # quote matches what it will bill. Ahead of the dry_run branch, so a
        # quote and a render warn -- or refuse -- alike.
        is_edit = bool(previous_interaction_id or input_video_uri)
        clamped = min(spec.max_duration_seconds, max(3, round(duration_seconds)))
        judged_seconds = (
            omni_continuation_upper_bound(
                spec,
                "edit",
                prior.duration_seconds,
                vertexai=omni_backend_resolved == "vertex",
            )
            if is_edit
            else clamped
        )
        delivery = omni_inline_delivery_policy(
            spec,
            vertexai=omni_backend_resolved == "vertex",
            output_gcs_uri=output_gcs_uri,
            resolution=normalized_resolution,
            task=None,
            output_seconds=judged_seconds,
            seconds_are_bound=is_edit,
        )
        if delivery:
            gcs_warnings.append(delivery)
        slow_tier = omni_slow_tier_warning(
            resolution=normalized_resolution,
            kind="edit" if is_edit else "render",
            output_seconds=judged_seconds,
            timeout_seconds=timeout_seconds,
            seconds_are_bound=is_edit,
        )
        if slow_tier:
            gcs_warnings.append(slow_tier)

        billed_resolution = _omni_billed_resolution(spec, normalized_resolution)

        if dry_run:
            # Disclose the GCS-allowlist warning on the notification channel,
            # not only in the quote body.
            await _emit_warnings(ctx, gcs_warnings)
            # Refuse local sources the real run would reject, so a quote does
            # not price a call that cannot run. Remote sources still price.
            for uri in (
                *inferred_images,
                *reference_images,
                *reference_videos,
                first_frame_uri,
                last_frame_uri,
            ):
                _assert_local_source(uri, app_ctx.data_folder, "image_uri")
            _assert_local_source(
                input_video_uri, app_ctx.data_folder, "input_video_uri"
            )
            # Measured here, not only on the render: pre-flighting must buy
            # the caller something for a violation the server can already see.
            gcs_warnings.extend(
                await _omni_local_reference_video_warnings(ctx, spec, reference_videos)
            )
            # An input video (or a prior interaction) makes this an edit —
            # omni._select_task_type keys on either — and an edit sends no
            # duration and renders a service-chosen length that measured 10s
            # regardless of the request. Quoting the clamped request here
            # under-quoted an input-video edit 3.3x, the same defect edit_video
            # already fixed; match it and quote the maximum as an upper bound.
            if previous_interaction_id or input_video_uri:
                if previous_interaction_id:
                    prior_duration = prior.duration_seconds
                else:
                    # Measured, like extend_video_omni's dry_run already does
                    # for a local source. Passing None here skipped the source
                    # ceiling entirely and let a quote succeed for a source the
                    # real run refuses. Remote sources stay unmeasured -- a
                    # quote does not download one.
                    prior_duration = None
                    source_bytes = await _fetch_local_source(ctx, input_video_uri)
                    if source_bytes is not None:
                        prior_duration = await _probe_media(
                            measure_video_duration_bytes, source_bytes
                        )
                        _ = await _check_omni_source_video(
                            spec,
                            source_bytes,
                            vertexai=(omni_backend_resolved == "vertex"),
                        )
                quoted = omni_continuation_upper_bound(
                    spec,
                    "edit",
                    prior_duration,
                    vertexai=(omni_backend_resolved == "vertex"),
                )
                return _respond(
                    app_ctx,
                    {
                        "dry_run": True,
                        "message": "Estimate only — nothing was generated",
                        "model": spec.model,
                        "backend": omni_backend_resolved,
                        "backend_reason": omni_backend_reason,
                        **_omni_requested_model(requested_model, spec),
                        "resolution": billed_resolution,
                        "resolution_source": _omni_resolution_provenance(
                            resolution_origin,
                            "a dry run renders nothing whose resolution could be measured",
                        ),
                        "duration_seconds": quoted,
                        "duration_source": (
                            "upper bound: an edit's rendered length is chosen "
                            "by the service and is not predictable from the "
                            f"request or the source, so this quotes the "
                            f"longest clip it could produce ({quoted:g}s). The "
                            "real run reports the measured duration."
                        ),
                        "estimated_cost": _video_cost(
                            spec.model,
                            quoted,
                            resolution=billed_resolution,
                            include_audio=False,
                            presnapped=True,
                        ),
                        **({"warnings": gcs_warnings} if gcs_warnings else {}),
                    },
                )
            # The impl says this on the real run; the quote must too.
            clamp = omni_duration_clamp_warning(spec, duration_seconds)
            quote_warnings = [*([clamp] if clamp else []), *gcs_warnings]
            return _respond(
                app_ctx,
                {
                    "dry_run": True,
                    "message": "Estimate only — nothing was generated",
                    "model": spec.model,
                    "backend": omni_backend_resolved,
                    "backend_reason": omni_backend_reason,
                    **_omni_requested_model(requested_model, spec),
                    "resolution": billed_resolution,
                    "resolution_source": _omni_resolution_provenance(
                        resolution_origin,
                        "a dry run renders nothing whose resolution could be measured",
                    ),
                    # Report both, like generate_video and edit_video: a caller
                    # who asked for 2s must see it snapped to 3 here, not a bare
                    # 3 with no sign their request was clamped.
                    "requested_duration_seconds": duration_seconds,
                    "duration_seconds": clamped,
                    "estimated_cost": _video_cost(
                        spec.model,
                        float(clamped),
                        resolution=billed_resolution,
                        include_audio=False,
                        presnapped=True,
                    ),
                    **({"warnings": quote_warnings} if quote_warnings else {}),
                },
            )

        image_bytes_list = [
            await _fetch_omni_media(ctx, uri, "image_uri") for uri in inferred_images
        ]
        reference_image_bytes = [
            await _fetch_omni_media(ctx, uri, "reference_image_uri")
            for uri in reference_images
        ]
        reference_video_bytes = [
            await _fetch_omni_media(ctx, uri, "reference_video_uri")
            for uri in reference_videos
        ]
        first_frame_bytes = await _fetch_optional_omni_media(
            ctx, first_frame_uri, "first_frame_uri"
        )
        # A loop names one image twice; it is fetched once and sent once.
        last_frame_bytes = (
            first_frame_bytes
            if last_frame_uri is not None and last_frame_uri == first_frame_uri
            else await _fetch_optional_omni_media(ctx, last_frame_uri, "last_frame_uri")
        )
        extra_warnings = await _omni_reference_video_warnings(
            spec, reference_video_bytes
        )
        if source_uri:
            # Sent by URI, so nothing is inlined; downloaded only to measure.
            input_video_bytes = None
            source_to_check = await _fetch_omni_gcs_source_to_measure(
                ctx, source_uri, extra_warnings
            )
        else:
            input_video_bytes = await _fetch_optional_omni_media(
                ctx, input_video_uri, "input_video_uri"
            )
            source_to_check = input_video_bytes
        if source_to_check is not None:
            # The model's documented edit-source ceiling. Left to the
            # service, an over-long clip was fetched, uploaded through the
            # Files API and rejected for a limit that was measurable from the
            # bytes before any of that.
            extra_warnings.extend(
                await _check_omni_source_video(
                    spec,
                    source_to_check,
                    vertexai=(omni_backend_resolved == "vertex"),
                    inline=input_video_bytes is not None,
                )
            )

        await ctx.info(f"Generating video with {spec.model}")
        result = await _omni_generate_and_manifest(
            app_ctx,
            ctx,
            prior=prior,
            prefer_backend=omni_backend_resolved,
            backend_reason=omni_backend_reason,
            prompt=prompt,
            model=spec.model,
            image_bytes_list=image_bytes_list or None,
            first_frame_bytes=first_frame_bytes,
            last_frame_bytes=last_frame_bytes,
            reference_image_bytes_list=reference_image_bytes or None,
            input_video_bytes=input_video_bytes,
            input_video_uri=source_uri,
            reference_video_bytes_list=reference_video_bytes or None,
            previous_interaction_id=previous_interaction_id,
            aspect_ratio=aspect_ratio,
            duration_seconds=duration_seconds,
            resolution=normalized_resolution,
            resolution_origin=resolution_origin,
            output_gcs_uri=output_gcs_uri,
            timeout_seconds=timeout_seconds,
            submission=omni_submission,
            manifest_extra={
                "image_uris": image_uris,
                "first_frame_uri": first_frame_uri,
                "last_frame_uri": last_frame_uri,
                "reference_image_uris": reference_image_uris,
                "reference_video_uris": reference_video_uris,
                "input_video_uri": input_video_uri,
            },
        )
        result.update(_omni_requested_model(requested_model, spec))
        # The allowlist warning computed above never reached the real-run body
        # (it was only wired into the dry_run payload). Merge it so the body
        # and the notification channel agree, then mirror every warning.
        if gcs_warnings or extra_warnings:
            body_warnings = result.setdefault("warnings", [])
            for warning in (*gcs_warnings, *extra_warnings):
                if warning not in body_warnings:
                    body_warnings.append(warning)
        await _emit_warnings(ctx, result.get("warnings"))
        return _respond(app_ctx, result)
    except Exception as e:
        await ctx.error(f"Omni video generation failed: {e}")
        logger.exception("Tool error")
        omni_body: dict[str, Any] = {"error": str(e)}
        if (status := http_status(e)) is not None:
            omni_body["status_code"] = status
        if (wait := retry_after_seconds(e)) is not None:
            omni_body["retry_after_seconds"] = wait
        if omni_submission.quota_retries:
            omni_body["quota_retries"] = omni_submission.quota_retries
        state = omni_submission.state(e)
        if isinstance(e, TimeoutError) or state != "none":
            omni_body["submission"] = state
            omni_body["timed_out"] = isinstance(e, TimeoutError)
            if omni_submission.interaction_id:
                omni_body["interaction_id"] = omni_submission.interaction_id
            handle = (
                f"Interaction {omni_submission.interaction_id}"
                if omni_submission.interaction_id
                else None
            )
            omni_body["note"] = _submission_note(
                state,
                handle=handle,
                completed=omni_submission.completed,
                final_status=omni_submission.final_status,
            )
        return json.dumps(omni_body)


@mcp.tool()
async def upscale_video_omni(
    ctx: Context[ServerSession, AppContext],
    resolution: str = "1080p",
    input_video_uri: str | None = None,
    previous_interaction_id: str | None = None,
    source_resolution: str | None = None,
    output_gcs_uri: str | None = None,
    timeout_seconds: int = OMNI_DEFAULT_TIMEOUT_SECONDS,
    max_cost_usd: float | None = None,
    dry_run: bool = False,
) -> str:
    """Upscale an Omni video up to 15s long using Omni 1.2 on Vertex AI.

    Give exactly one input_video_uri or previous_interaction_id. Supported
    transitions: 360p to 720p/1080p; 720p to 1080p/4K. A previous interaction
    must live on Vertex and on 1.2: one created on the retired
    gemini-omni-1.1-flash is refused, so pass its saved video as
    input_video_uri. Generated video is the recommended source. For a source
    without recorded metadata, supply source_resolution. A dry run reads
    local metadata and quotes a 15s upper bound for unmeasured sources.
    Upscaling preserves frames, audio and timing and bills zero input tokens.

    A 1080p or 4K target, or a 720p one longer than 10s (an unmeasured
    source counts as 15s; a dry run does not download a remote one), can
    exceed the 4 MB inline response limit. Without output_gcs_uri it is returned
    inline, which Vertex AI documents for upscaling, with a warning before
    it renders; pass output_gcs_uri to have it written to Cloud Storage, the
    safer delivery for a large output.

    Args:
        ctx: MCP context with application state.
        resolution: The target (default "1080p"): 720p or 1080p from a 360p
            source, 1080p or 4K from a 720p one.
        input_video_uri: A clip to upscale, up to 15s (gs://, http(s)://,
            or file:// inside DATA_FOLDER). Give exactly one of this and
            previous_interaction_id.
        previous_interaction_id: A 1.2 interaction created on Vertex AI. One
            that was itself an upscale is sent the clip it saved instead: by
            id, the service refuses to upscale it again.
        source_resolution: The source's resolution, needed only when it has
            no recorded metadata and cannot be measured.
        output_gcs_uri: A gs:// prefix ending in / (Vertex AI writes a new
            object under it), checked against the bucket allowlist.
        timeout_seconds: Deadline for the whole upscale, default 210s. A 4K
            upscale of more than a few seconds is likely to outlast it, and
            is warned about before it renders.
        max_cost_usd: Refuse before rendering if the quote is over this; a
            dry run returns the quote with would_be_refused instead.
        dry_run: Quote and render nothing.

    Returns:
        JSON with video_url, interaction_id, the source and target
        resolutions, cost, and any warnings.
    """
    submission = OmniSubmission()
    try:
        from .pricing import estimate_omni_upscale_cost

        app_ctx = ctx.request_context.lifespan_context
        if (input_video_uri is None) == (previous_interaction_id is None):
            raise ValueError(
                "Give exactly one input_video_uri or previous_interaction_id."
            )
        prior = await _prior_interaction_or_empty(
            app_ctx.videos_dir, previous_interaction_id
        )
        # An upscale interaction is upscaled again from the clip it saved:
        # by id, the service refused it after submission. Its record still
        # gives the source's resolution and length.
        upscaled_from = previous_interaction_id
        saved_clip = _upscale_interaction_clip(
            prior, previous_interaction_id, "upscale_video_omni"
        )
        if saved_clip is not None:
            input_video_uri, previous_interaction_id = saved_clip, None
        # Standalone upscaling exists on Vertex AI only. Said here in this
        # tool's own terms: the shared resolver below words both refusals as
        # an omni_backend request, a parameter this tool does not have.
        if not bool(getattr(app_ctx.client._api_client, "vertexai", False)):
            raise ValueError(
                "upscale_video_omni runs only on Vertex AI, and this server's "
                "primary client is the Gemini Developer API "
                "(GOOGLE_GENAI_USE_VERTEXAI is not true). Configure Vertex AI "
                "to upscale, or render at the resolution you need with "
                "generate_video_omni(resolution=...)."
            )
        retired = _retired_interaction_refusal(
            prior, previous_interaction_id, "upscale"
        )
        if retired:
            raise ValueError(retired)
        if prior.backend == "gemini_api":
            raise ValueError(
                f"Interaction {previous_interaction_id} was created on the "
                "Gemini Developer API, and standalone upscaling runs only on "
                "Vertex AI, where that interaction does not exist. Call "
                f"upscale_video_omni with {_saved_clip_source(prior)} in place "
                "of previous_interaction_id. Nothing was rendered or billed."
            )
        backend, _ = await _resolve_omni_backend(
            app_ctx,
            "vertex",
            need_gcs=bool(output_gcs_uri),
            previous_interaction_id=previous_interaction_id,
            task="upscale",
            prior=prior,
        )
        # The resolver is handed "vertex" for its checks, and its reason then
        # read "omni_backend=vertex was requested" for a call that passed
        # none; this tool has no such parameter.
        reason = "standalone upscaling runs only on Vertex AI"
        if prior.model and prior.model != OMNI_1_2_MODEL:
            raise ValueError(
                f"Interaction {previous_interaction_id} was created on "
                f"{prior.model}; standalone upscaling runs on {OMNI_1_2_MODEL}, "
                "and an interaction cannot change models. Call "
                f"upscale_video_omni with {_saved_clip_source(prior)} in place "
                "of previous_interaction_id. Nothing was rendered or billed."
            )
        spec = omni_spec(OMNI_1_2_MODEL)
        target = _validate_omni_resolution(spec, resolution)
        source = prior.resolution or normalize_omni_resolution(source_resolution)
        if (
            prior.resolution
            and source_resolution
            and normalize_omni_resolution(source_resolution) != prior.resolution
        ):
            raise ValueError(
                "source_resolution disagrees with the recorded video resolution."
            )
        _assert_local_source(input_video_uri, app_ctx.data_folder, "input_video_uri")
        # Evaluated once: it validates (and logs) the source, and its answer
        # decides what the render sends.
        source_uri = _omni_gcs_source(
            app_ctx, input_video_uri, backend, model=OMNI_1_2_MODEL
        )
        warnings = _gcs_allowlist_warnings(app_ctx, input_video_uri, output_gcs_uri)
        if saved_clip is not None and upscaled_from:
            warnings.insert(
                0, _upscale_interaction_note(upscaled_from, saved_clip, dry_run)
            )
        warnings.extend(_gcs_prefix_warning(output_gcs_uri))
        if output_gcs_uri:
            bucket = _parse_gcs_bucket(output_gcs_uri)
            if bucket is None:
                raise ValueError("output_gcs_uri must start with gs://.")
            _assert_gcs_bucket_allowed(bucket, app_ctx.allowed_gcs_buckets)
        seconds = prior.duration_seconds
        seconds_measured = bool(seconds) and prior.duration_measured

        # The longest source upscaling takes, which is also the bound an
        # unmeasured source is quoted at.
        longest = omni_source_limit_seconds(spec, task="upscale")
        if source:
            # A transition with no published rate cannot run at all; refused
            # before the source is downloaded.
            estimate_omni_upscale_cost(seconds or longest, source, target or resolution)
        local_bytes = (
            await _fetch_local_source(ctx, input_video_uri) if input_video_uri else None
        )
        if not dry_run and input_video_uri and local_bytes is None:
            if source_uri:
                # Sent by URI; downloaded only to measure its length and
                # dimensions, which price the render and verify the source.
                local_bytes = await _fetch_omni_gcs_source_to_measure(
                    ctx, source_uri, warnings
                )
            else:
                local_bytes = await _fetch_omni_media(
                    ctx, input_video_uri, "input_video_uri"
                )
        if local_bytes is not None:
            _check_omni_inline_size(
                spec, local_bytes, vertexai=True, inline=source_uri is None
            )
            # One ffmpeg probe for both figures; this used to run three on
            # the same bytes (length check, duration, dimensions).
            measured = await _probe_media(measure_video_metadata_bytes, local_bytes)
            seconds, dimensions = measured if measured else (None, None)
            seconds_measured = seconds is not None
        else:
            dimensions = None
        saved_unread = False
        if (
            not source
            and previous_interaction_id
            and prior.video_url
            and local_bytes is None
        ):
            # A take recorded before the remote record kept its resolution
            # (a gs:// delivery) still names its saved clip, which settles
            # it; asking for source_resolution there refused an upscale this
            # server could answer itself. A dry run reads only a local clip.
            saved = prior.video_url
            saved_bytes: bytes | None = None
            if saved.startswith("file://"):
                saved_bytes = await _fetch_local_source(ctx, saved)
            elif saved.startswith("gs://") and not dry_run:
                saved_bytes = await fetch(
                    saved,
                    allowed_dir=app_ctx.data_folder,
                    allowed_gcs_buckets=app_ctx.allowed_gcs_buckets,
                )
            saved_unread = saved_bytes is None
            if saved_bytes is not None:
                saved_meta = await _probe_media(
                    measure_video_metadata_bytes, saved_bytes
                )
                if saved_meta:
                    saved_seconds, dimensions = saved_meta
                    if saved_seconds is not None:
                        seconds, seconds_measured = saved_seconds, True
        too_long = _omni_source_length_refusal(
            spec,
            seconds,
            task="upscale",
            uploaded=previous_interaction_id is None,
        )
        if too_long:
            raise ValueError(too_long)
        if dimensions:
            measured_resolution = classify_video_resolution(dimensions)
            if source and measured_resolution and source != measured_resolution:
                raise ValueError(
                    "source_resolution disagrees with the source video dimensions."
                )
            source = measured_resolution or source
        if not source:
            raise ValueError(
                "Supply source_resolution (360p or 720p) for an unrecorded source."
                + (
                    " A dry run does not download the saved clip; a real run "
                    "measures it."
                    if dry_run
                    and saved_unread
                    and str(prior.video_url).startswith("gs://")
                    else ""
                )
            )
        estimate = estimate_omni_upscale_cost(
            seconds or longest,
            source,
            target or resolution,
            # A measured source is the output's length (upscaling preserves
            # it), so it is priced as measured; a recorded request or the
            # bound is padded like any other unmeasured length.
            measured=seconds_measured,
        )
        # Upscaling preserves the source's length, so the output runs as
        # long as the source measured (or was recorded), and as long as the
        # bound the estimate prices when neither is known -- which a dry run
        # of a remote source, not downloaded, always is. Same policy, and
        # same place relative to the dry_run branch, as the other omni tools.
        delivery = omni_inline_delivery_policy(
            spec,
            vertexai=True,
            output_gcs_uri=output_gcs_uri,
            resolution=target,
            task="upscale",
            output_seconds=seconds or longest,
            seconds_are_bound=not seconds,
        )
        if delivery:
            warnings.append(delivery)
        # A 10s 720p-to-4K upscale measured 210s, the default deadline itself.
        slow_tier = omni_slow_tier_warning(
            resolution=target,
            kind="upscale",
            output_seconds=seconds or longest,
            timeout_seconds=timeout_seconds,
            seconds_are_bound=not seconds,
        )
        if slow_tier:
            warnings.append(slow_tier)
        over_budget = None
        if max_cost_usd is not None:
            if not math.isfinite(max_cost_usd) or max_cost_usd <= 0:
                raise ValueError("max_cost_usd must be finite and greater than zero.")
            if estimate.usd > max_cost_usd:
                over_budget = (
                    f"Estimated upscale cost ${estimate.usd:.4f} exceeds "
                    f"max_cost_usd=${max_cost_usd:g}."
                )
                # A real run refuses before anything renders. A dry run
                # returns its quote with would_be_refused, as
                # extend_video_omni's does: the quote is what a caller needs
                # to set a cap that passes.
                if not dry_run:
                    raise ValueError(over_budget)
        if dry_run:
            if over_budget:
                warnings.append(over_budget)
            await _emit_warnings(ctx, warnings)
            return _respond(
                app_ctx,
                {
                    "dry_run": True,
                    "model": OMNI_1_2_MODEL,
                    "task": "upscale",
                    "backend": backend,
                    "backend_reason": reason,
                    "source_resolution": source,
                    "resolution": target,
                    "duration_seconds": seconds or longest,
                    "duration_source": (
                        ("measured source" if seconds_measured else "source metadata")
                        if seconds
                        else f"{longest:g}s upper bound"
                    ),
                    "estimated_cost": _cost_payload(estimate),
                    **({"would_be_refused": True} if over_budget else {}),
                    **({"warnings": warnings} if warnings else {}),
                },
            )
        result = await _omni_generate_and_manifest(
            app_ctx,
            ctx,
            prior=prior,
            prompt="Upsample video.",
            model=OMNI_1_2_MODEL,
            # A URI source's bytes were downloaded only to measure it; sending
            # them too would add a second video part.
            input_video_bytes=None if source_uri else local_bytes,
            input_video_uri=source_uri,
            previous_interaction_id=previous_interaction_id,
            task="upscale",
            duration_seconds=None,
            resolution=target,
            output_gcs_uri=output_gcs_uri,
            upscale_source_resolution=source,
            source_duration_seconds=seconds,
            source_duration_measured=seconds_measured,
            prefer_backend=backend,
            backend_reason=reason,
            timeout_seconds=timeout_seconds,
            submission=submission,
            manifest_extra={
                "kind": "omni_upscale",
                "source_resolution": source,
                **(
                    {"source_interaction_id": upscaled_from}
                    if saved_clip is not None
                    else {}
                ),
            },
        )
        if warnings:
            body_warnings = result.setdefault("warnings", [])
            for warning in warnings:
                if warning not in body_warnings:
                    body_warnings.append(warning)
        await _emit_warnings(ctx, result.get("warnings"))
        return _respond(app_ctx, result)
    except Exception as exc:
        await ctx.error(f"Omni upscaling failed: {exc}")
        return json.dumps(
            {
                "error": str(exc),
                "submission": submission.state(exc),
                **(
                    {"interaction_id": submission.interaction_id}
                    if submission.interaction_id
                    else {}
                ),
                **(
                    {"quota_retries": submission.quota_retries}
                    if submission.quota_retries
                    else {}
                ),
            }
        )


async def _interaction_manifest_or_none(
    videos_dir: Path, interaction_id: str
) -> dict[str, Any] | None:
    """_manifest_for_interaction off the event loop, with the lookups' deadline."""
    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(
                _get_probe_executor(),
                _manifest_for_interaction,
                videos_dir,
                interaction_id,
            ),
            timeout=_SIDECAR_SCAN_TIMEOUT_SECONDS,
        )
    except Exception:
        logger.warning(
            "Sidecar lookup for interaction %s failed or exceeded %.0fs",
            interaction_id,
            _SIDECAR_SCAN_TIMEOUT_SECONDS,
            exc_info=True,
        )
        return None


# Fields of a saved interaction's record that get_omni_interaction returns
# without reading the service again.
_COLLECTED_RECORD_FIELDS = (
    "model",
    "task",
    "backend",
    "video_url",
    "duration_seconds",
    "duration_source",
    "duration_upper_bound_seconds",
    "resolution",
    "resolution_source",
    "aspect_ratio",
    "previous_interaction_id",
    "cost",
)

# How long get_omni_interaction will wait for a render still in progress.
_COLLECT_MAX_WAIT_SECONDS = 1800


@mcp.tool()
async def get_omni_interaction(
    ctx: Context[ServerSession, AppContext],
    interaction_id: str,
    omni_backend: str | None = None,
    wait_seconds: int = 0,
) -> str:
    """Check on an Omni render by interaction_id, and save it once it has finished.

    For a render whose tool call ended first: a timeout, whose error names the
    interaction_id, or an MCP host that cut the call off. The service keeps
    rendering and bills it either way. This tool only reads, so it never
    bills.

    `status` is the service's. While the render is in progress nothing is
    saved: call again, or pass wait_seconds to wait here. Once it has
    completed, the video is saved (or, delivered to Cloud Storage, left there)
    and measured, and the response carries `video_url`, `duration_seconds`,
    `resolution` and `cost`. The interaction is recorded like any render, so
    edit_video, extend_video_omni and upscale_video_omni can continue from
    its id. One this server has already saved is returned from its record,
    without reading the service.

    Args:
        interaction_id: The id the timed-out Omni call reported.
        omni_backend: "gemini_api" or "vertex", where the interaction was
            created; an id resolves only there. Defaults to the backend this
            server recorded for it, else the server's OMNI_BACKEND.
        wait_seconds: How long to wait for a render still in progress, up to
            1800. 0 reads its status once. Many MCP hosts end a tool call
            near 240s.
    """
    app_ctx = ctx.request_context.lifespan_context
    try:
        interaction_id = (interaction_id or "").strip()
        if not interaction_id:
            raise ValueError("interaction_id is required.")
        if omni_backend is not None and omni_backend not in ("gemini_api", "vertex"):
            raise ValueError(
                f'omni_backend={omni_backend!r} is not "gemini_api" or "vertex".'
            )
        manifest = (
            await _interaction_manifest_or_none(app_ctx.videos_dir, interaction_id)
            or {}
        )
        if manifest.get("video_url"):
            saved = {
                key: manifest[key]
                for key in _COLLECTED_RECORD_FIELDS
                if key in manifest
            }
            return _respond(
                app_ctx,
                {
                    "message": "Already saved by this server; the service was not read.",
                    "interaction_id": interaction_id,
                    "status": "completed",
                    **saved,
                },
            )
        recorded = manifest.get("backend")
        recorded = recorded if recorded in ("gemini_api", "vertex") else None
        if omni_backend and recorded and omni_backend != recorded:
            minted_on = (
                "Vertex AI" if recorded == "vertex" else "the Gemini Developer API"
            )
            raise ValueError(
                f"Interaction {interaction_id} was created on {minted_on}, as this "
                "server recorded it, and an interaction can be read only on the "
                f'backend that created it. Drop omni_backend, or pass "{recorded}".'
            )
        requested = omni_backend or (
            None if recorded else _omni_server_backend(app_ctx)
        )
        backend, reason = _omni_backend_decision(
            app_ctx, prefer_backend=recorded, requested=requested
        )
        if recorded and omni_backend is None:
            reason = "the backend this server recorded the interaction on"
        client = _client_for_omni(app_ctx, prefer_backend=recorded, requested=requested)
        try:
            collected = await collect_omni_interaction(
                client=client,
                interaction_id=interaction_id,
                videos_dir=app_ctx.videos_dir,
                wait_seconds=max(0, min(int(wait_seconds), _COLLECT_MAX_WAIT_SECONDS)),
                log_callback=ctx.info,
            )
        except Exception as exc:
            if http_status(exc) not in (400, 403, 404):
                raise
            where = "Vertex AI" if backend == "vertex" else "the Gemini Developer API"
            other = "gemini_api" if backend == "vertex" else "vertex"
            hint = (
                ""
                if recorded or omni_backend
                else f' If it was created on the other backend, pass omni_backend="{other}".'
            )
            raise ValueError(
                f"{exc} Interaction {interaction_id} could not be read on {where} "
                f"({reason}); an interaction can be read only on the backend that "
                f"created it.{hint}"
            ) from exc

        status = collected.get("status")
        model = collected.get("model") or manifest.get("model")
        head: dict[str, Any] = {
            "interaction_id": interaction_id,
            "status": status,
            "model": model,
            "backend": backend,
            "backend_reason": reason,
        }
        if collected.get("previous_interaction_id"):
            head["previous_interaction_id"] = collected["previous_interaction_id"]
        if "error" in collected:
            return json.dumps({"error": collected["error"], **head})
        if not collected.get("video_url"):
            return _respond(
                app_ctx,
                {
                    **head,
                    "message": (
                        f"Still {status}; nothing was saved. Call "
                        "get_omni_interaction again, or pass wait_seconds to "
                        "wait here. It bills when it finishes, collected or not."
                    ),
                },
            )

        warnings: list[str] = []
        if model is None:
            model = DEFAULT_OMNI_MODEL
            head["model"] = model
            warnings.append(
                "The service did not name the model and this server has no "
                f"record of it, so it is reported and priced as {model}."
            )
        result: dict[str, Any] = {
            "message": "Video collected",
            **head,
            "video_url": collected["video_url"],
        }
        facts = await _measure_rendered_video(
            result,
            resolution_key="resolution",
            requested_resolution=None,
            remote_app_ctx=app_ctx,
        )
        task = manifest.get("task")
        if task:
            result["task"] = task
        cost: dict[str, Any] | None = None
        if facts.duration_seconds is not None and facts.duration_source:
            result["duration_seconds"] = facts.duration_seconds
            result["duration_source"] = facts.duration_source
        else:
            result["duration_seconds"] = None
            result["duration_source"] = (
                "not measured: the render was delivered where this server "
                "could not read it"
            )
        if facts.resolution_source:
            result["resolution_source"] = facts.resolution_source
        measured = result["duration_seconds"]
        rendered = result.get("resolution")
        source_resolution = manifest.get("upscale_source_resolution")
        if measured is None or not rendered:
            warnings.append(
                "Not priced: the render's length or resolution could not be "
                "measured, and nothing else here says what it billed."
            )
        elif task == "upscale" and not source_resolution:
            warnings.append(
                "Not priced: an upscale bills by its source's resolution, which "
                "this server did not record."
            )
        elif task == "upscale":
            cost = _omni_upscale_cost(
                float(measured),
                str(source_resolution),
                rendered=str(rendered),
                requested=None,
                actual=True,
                measured=True,
            )
        else:
            cost = _video_cost(
                model,
                float(measured),
                resolution=str(rendered),
                include_audio=False,
                actual=True,
            )
            # A recorded render with no task is a new render whose task was
            # inferred; an upscale always records its task. Only with no
            # record at all is "a render" a guess.
            if cost and not manifest:
                cost = dict(cost)
                cost["detail"] = (
                    f"{cost['detail']} — this server has no record of the "
                    "request, so it is priced as a render; an upscale bills less"
                )
        if cost:
            result["cost"] = cost
        if warnings:
            result["warnings"] = warnings
        record: dict[str, Any] = {
            "kind": "omni_video",
            "collected_by": "get_omni_interaction",
            **{
                key: result.get(key)
                for key in (
                    "interaction_id",
                    "model",
                    "task",
                    "backend",
                    "backend_reason",
                    "aspect_ratio",
                    "resolution",
                    "resolution_source",
                    "duration_seconds",
                    "duration_source",
                    "previous_interaction_id",
                    "video_url",
                    "cost",
                )
                if result.get(key) is not None
            },
        }
        sidecar_url = _write_sidecar(str(result["video_url"]), record)
        if sidecar_url:
            result["sidecar_url"] = sidecar_url
        else:
            result["manifest"] = record
            _write_remote_interaction_record(app_ctx.videos_dir, record)
        await _emit_warnings(ctx, result.get("warnings"))
        return _respond(app_ctx, result)
    except Exception as exc:
        await ctx.error(f"Reading the Omni interaction failed: {exc}")
        body: dict[str, Any] = {"error": str(exc), "interaction_id": interaction_id}
        if (status_code := http_status(exc)) is not None:
            body["status_code"] = status_code
        return json.dumps(body)


async def _edit_saved_clip(
    ctx: Context[ServerSession, AppContext],
    *,
    interaction_id: str,
    clip: str,
    recorded_resolution: str | None,
    prompt: str,
    omni_model: str | None,
    omni_backend: str | None,
    resolution: str | None,
    timeout_seconds: int,
    dry_run: bool,
    reference_image_uris: list[str] | None,
    output_gcs_uri: str | None,
) -> str:
    """edit_video of an upscale interaction: generate_video_omni on its clip.

    The response is generate_video_omni's, with a warning saying why. With
    resolution unset it sends the recorded one, as an edit by id would, and
    says so where a dry run would otherwise call it the caller's request.
    """
    body = json.loads(
        await generate_video_omni(
            ctx,
            prompt=prompt,
            omni_model=omni_model,
            omni_backend=omni_backend,
            input_video_uri=clip,
            resolution=resolution or recorded_resolution,
            output_gcs_uri=output_gcs_uri,
            timeout_seconds=timeout_seconds,
            dry_run=dry_run,
            reference_image_uris=reference_image_uris,
        )
    )
    if (
        resolution is None
        and recorded_resolution
        and str(body.get("resolution_source", "")).startswith("the request")
    ):
        body["resolution_source"] = _omni_resolution_provenance(
            "source", "a dry run renders nothing whose resolution could be measured"
        )
    note = _upscale_interaction_note(interaction_id, clip, dry_run)
    body["warnings"] = [note, *[w for w in body.get("warnings", []) if w != note]]
    await _emit_warnings(ctx, [note])
    return json.dumps(body)


@mcp.tool()
async def edit_video(
    ctx: Context[ServerSession, AppContext],
    previous_interaction_id: str,
    prompt: str,
    omni_model: str | None = None,
    omni_backend: str | None = None,
    aspect_ratio: str = "16:9",
    duration_seconds: float = 6.0,
    resolution: str | None = None,
    timeout_seconds: int = OMNI_DEFAULT_TIMEOUT_SECONDS,
    dry_run: bool = False,
    reference_image_uris: list[str] | None = None,
    output_gcs_uri: str | None = None,
) -> str:
    """Refine a generated Omni video through a multi-turn conversation.

    Pass previous_interaction_id and describe the change in prompt: swap a
    product/background, adjust lighting/weather, or request a different aspect
    ratio. The interaction stays on the model and backend that created it:
    omni_model left unset continues on that model, and naming a different one
    is refused. gemini-omni-1.1-flash is retired here: naming it runs on 1.2
    with a warning, and an interaction created on it cannot be edited -- pass
    its saved video as input_video_uri to generate_video_omni instead.
    Omni 1.2 edits clips up to 15s. reference_image_uris can guide the edit
    (up to 6 on the Gemini API, 2 on Vertex). Audio uploads and voice editing
    are unsupported. Context expires; Vertex retains interactions up to 7 days.
    An upscale's interaction cannot be continued by id, so its saved clip is
    edited instead, through generate_video_omni, at its recorded resolution.

    resolution selects 360p/720p/1080p/4K; unset, the edit keeps the
    resolution recorded for the interaction. output_gcs_uri requires Vertex.
    An edit can run 15s and exceed the 4 MB inline limit, so on Vertex without
    output_gcs_uri it is returned inline with a warning, dry run included.
    duration_seconds and aspect_ratio remain accepted for compatibility and
    are omitted from the edit request; describe framing changes in prompt.
    dry_run quotes the model's maximum output duration as an upper bound;
    real runs measure output when available. timeout_seconds is the complete
    interaction deadline; keep the returned interaction_id for further edits.

    Args:
        ctx: MCP context with application state.
        previous_interaction_id: interaction_id from a prior Omni result.
        prompt: Only the change ("make the sky stormy", "remove the person
            on the left"); unmentioned elements are preserved, and
            over-describing the scene changes parts you meant to keep.
        omni_model: Unset runs on the model that created the interaction;
            naming another is refused.
        omni_backend: "auto", "vertex" or "gemini_api". Unset, the server's
            OMNI_BACKEND applies. "auto" uses the Gemini API whenever
            GEMINI_API_KEY is set, even on a Vertex AI server -- a different
            project, billing path and quota pool -- so a deployment that must
            keep Omni traffic on Vertex AI passes "vertex" or sets
            OMNI_BACKEND=vertex. The response reports `backend` and
            `backend_reason`. A choice that contradicts the backend a
            previous_interaction_id was created on is refused.
        aspect_ratio: Not sent: an edit rejects it. Ask for a framing change
            in prompt.
        duration_seconds: Not sent: an edit rejects it, and the service
            picks the rendered length, up to 15s.
        resolution: "360p", "720p", "1080p" or "4K"; unset keeps the
            resolution recorded for the interaction, else the service
            default.
        timeout_seconds: Deadline for the whole edit, default 210s (see
            generate_video_omni for why).
        dry_run: Quote the edit at its 15s upper bound and render nothing.
            The real run measures the rendered file and bills that.
        reference_image_uris: Images that guide the edit, bound to
            `<IMAGE_REF_0>`, ...: up to 6 on the Gemini API, 2 on Vertex AI.
        output_gcs_uri: A gs:// prefix ending in / (Vertex AI writes a new
            object under it), Vertex AI only.

    Returns:
        JSON with the edited video_url and a new interaction_id for further
        edits.
    """
    try:
        app_ctx = ctx.request_context.lifespan_context
        # Which backend serves this call, decided once and reported; see
        # _omni_backend_decision.
        # Read once: the backend, the model, the source length and the
        # render all come from this one record.
        prior = await _prior_interaction_or_empty(
            app_ctx.videos_dir, previous_interaction_id
        )
        saved_clip = _upscale_interaction_clip(
            prior, previous_interaction_id, "generate_video_omni"
        )
        if saved_clip is not None:
            # An upscale interaction is no single clip to continue: the
            # service refused an upscale of one by id. Its saved clip is
            # edited instead, as an uploaded source, at the resolution it
            # was recorded at rather than the 720p default an upload gets.
            return await _edit_saved_clip(
                ctx,
                interaction_id=previous_interaction_id,
                clip=saved_clip,
                recorded_resolution=prior.resolution,
                prompt=prompt,
                omni_model=omni_model,
                omni_backend=omni_backend,
                resolution=resolution,
                timeout_seconds=timeout_seconds,
                dry_run=dry_run,
                reference_image_uris=reference_image_uris,
                output_gcs_uri=output_gcs_uri,
            )
        omni_backend_resolved, omni_backend_reason = await _resolve_omni_backend(
            app_ctx,
            omni_backend,
            need_gcs=bool(output_gcs_uri),
            previous_interaction_id=previous_interaction_id,
            prior=prior,
        )
        # Fail fast before any fetch or interaction work; the omni impl
        # clamps to the model's range ([3, 15]s on 1.2), which would turn a
        # negative or NaN duration
        # into a billed 3s render instead of an error.
        _validate_duration_seconds(duration_seconds)
        # An edit does not send the ratio on the wire, but src/omni.py still
        # rejects an unsupported one before it gets that far, so a quote that
        # accepted '4:3' promised a render the tool refuses. Validating here
        # keeps the two answers identical instead of deciding the edit case
        # differently from the impl that owns it.
        _validate_aspect_ratio(aspect_ratio)
        requested_model = _continuation_omni_model(omni_model, prior)
        spec = _validate_omni_model(requested_model)
        _validate_omni_continuation_source(spec, prior, previous_interaction_id)
        normalized_resolution, resolution_origin = _omni_sent_resolution(
            spec,
            _validate_omni_resolution(spec, resolution),
            prior,
            previous_interaction_id,
        )
        billed_resolution = _omni_billed_resolution(spec, normalized_resolution)
        references = list(reference_image_uris or [])
        validate_omni_request(
            spec.model,
            resolution=resolution,
            task="edit",
            reference_image_count=len(references),
            is_continuation=True,
            vertexai=omni_backend_resolved == "vertex",
        )
        if len(references) > MAX_OMNI_INPUT_IMAGES:
            # The same cap generate_video_omni and extend_video_omni apply,
            # for the same reason: every image is buffered whole (each up to
            # MAX_FETCH_BYTES) and base64-inflated into one request body, and
            # only 1.2 has a model limit of its own (checked just above).
            raise ValueError(
                f"Too many reference images ({len(references)}); "
                f"{spec.model} accepts at most {MAX_OMNI_INPUT_IMAGES} here."
            )
        for uri in references:
            _assert_local_source(uri, app_ctx.data_folder, "reference_image_uri")
        # Every gs:// URI this call reads or writes; both are reached with
        # ambient credentials when no allowlist is configured.
        warnings = _gcs_allowlist_warnings(app_ctx, *references, output_gcs_uri)
        warnings.extend(_gcs_prefix_warning(output_gcs_uri))
        if output_gcs_uri:
            bucket = _parse_gcs_bucket(output_gcs_uri)
            if bucket is None:
                raise ValueError("output_gcs_uri must start with gs://.")
            _assert_gcs_bucket_allowed(bucket, app_ctx.allowed_gcs_buckets)
        # On the quote and the render alike; see generate_video_omni.
        reroute = omni_model_reroute_warning(requested_model)
        if reroute:
            warnings.append(reroute)
        # On the length the quote below prices, the edit's upper bound, and
        # likewise ahead of the dry_run branch.
        edit_bound = omni_continuation_upper_bound(
            spec,
            "edit",
            prior.duration_seconds,
            vertexai=omni_backend_resolved == "vertex",
        )
        delivery = omni_inline_delivery_policy(
            spec,
            vertexai=omni_backend_resolved == "vertex",
            output_gcs_uri=output_gcs_uri,
            resolution=normalized_resolution,
            task="edit",
            output_seconds=edit_bound,
            seconds_are_bound=True,
        )
        if delivery:
            warnings.append(delivery)
        slow_tier = omni_slow_tier_warning(
            resolution=normalized_resolution,
            kind="edit",
            output_seconds=edit_bound,
            timeout_seconds=timeout_seconds,
            seconds_are_bound=True,
        )
        if slow_tier:
            warnings.append(slow_tier)

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
            # Its length was already held to the model's edit-source cap by
            # _validate_omni_continuation_source, from the same sidecar.
            source_duration = prior.duration_seconds
            # An unrecorded source is bounded by the model's edit-source cap,
            # which is the same on both backends.
            quoted = omni_continuation_upper_bound(
                spec,
                "edit",
                source_duration,
                vertexai=(omni_backend_resolved == "vertex"),
            )
            payload: dict[str, Any] = {
                "dry_run": True,
                "message": "Estimate only — nothing was generated",
                "model": spec.model,
                "backend": omni_backend_resolved,
                "backend_reason": omni_backend_reason,
                "resolution": billed_resolution,
                "resolution_source": _omni_resolution_provenance(
                    resolution_origin,
                    "a dry run renders nothing whose resolution could be measured",
                ),
                "previous_interaction_id": previous_interaction_id,
                "duration_seconds": quoted,
                "duration_source": (
                    "upper bound: an edit's rendered length is chosen by the "
                    "service and is not predictable from the request or the "
                    f"source, so this quotes the longest clip it could produce "
                    f"({quoted:g}s). The real run reports the measured duration."
                ),
            }
            if source_duration is not None:
                payload["source_duration_seconds"] = source_duration
            payload.update(_omni_requested_model(requested_model, spec))
            if warnings:
                payload["warnings"] = warnings
                await _emit_warnings(ctx, warnings)
            payload["estimated_cost"] = _video_cost(
                spec.model,
                quoted,
                resolution=billed_resolution,
                include_audio=False,
                presnapped=True,
            )
            return _respond(app_ctx, payload)

        await ctx.info(f"Editing video (interaction {previous_interaction_id})")
        result = await _omni_generate_and_manifest(
            app_ctx,
            ctx,
            prior=prior,
            prefer_backend=omni_backend_resolved,
            backend_reason=omni_backend_reason,
            prompt=prompt,
            model=spec.model,
            reference_image_bytes_list=[
                await _fetch_omni_media(ctx, uri, "reference_image_uri")
                for uri in references
            ],
            output_gcs_uri=output_gcs_uri,
            previous_interaction_id=previous_interaction_id,
            aspect_ratio=aspect_ratio,
            duration_seconds=duration_seconds,
            resolution=normalized_resolution,
            resolution_origin=resolution_origin,
            timeout_seconds=timeout_seconds,
            manifest_extra={"kind": "omni_edit"},
        )
        result.update(_omni_requested_model(requested_model, spec))
        if warnings:
            body_warnings = result.setdefault("warnings", [])
            for warning in warnings:
                if warning not in body_warnings:
                    body_warnings.append(warning)
        await _emit_warnings(ctx, result.get("warnings"))
        return _respond(app_ctx, result)
    except Exception as e:
        await ctx.error(f"Video edit failed: {e}")
        logger.exception("Tool error")
        return json.dumps({"error": str(e)})


@mcp.tool()
async def extend_video_omni(
    ctx: Context[ServerSession, AppContext],
    prompt: str,
    previous_interaction_id: str | None = None,
    input_video_uri: str | None = None,
    omni_model: str | None = None,
    omni_backend: str | None = None,
    times: int = 1,
    resolution: str | None = None,
    reference_image_uris: list[str] | None = None,
    output_gcs_uri: str | None = None,
    timeout_seconds: int = OMNI_DEFAULT_TIMEOUT_SECONDS,
    max_cost_usd: float | None = None,
    dry_run: bool = False,
    duration_seconds: float | None = None,
) -> str:
    """Append a seamless continuation with Omni 1.2 Flash (default).

    Give exactly one previous_interaction_id or input_video_uri. The model
    uses the final 10s as context and preserves prior footage. Describe the
    continuation in prompt; for a conversation the prompt must explicitly ask
    to extend (the API infers the task from context).

    duration_seconds controls each appended segment on 1.2: a whole second
    from 3 to 10, default 10. times chains turns up to the model's cumulative
    ceiling (30s on the Gemini API), and a call takes at most 30 divided by
    the turn length (3 turns at 10s, 10 at 3s); continue from the returned
    interaction_id for more. Near the ceiling a turn appends the whole
    seconds that still fit, and a turn with less than 3s left is not run.
    Vertex retains up to the last 30s of each input and appends the new
    segment, producing up to 40s per turn.
    Each turn renders and bills the assembled clip.
    dry_run estimates all turns; max_cost_usd refuses a chain over that cost.
    The source is measured locally or recovered from interaction metadata;
    otherwise the estimate uses a conservative bound.

    reference_image_uris supplies up to 6 images on 1.2. Video references are
    unsupported for 1.2 editing/extension. resolution selects 360p/720p/1080p/4K.
    output_gcs_uri requires Vertex and is used on the final turn. Every turn
    can exceed the 4 MB inline limit, so on Vertex a turn without output_gcs_uri
    (every turn but the last when it is given) is returned inline, with a
    warning on the dry run and the render alike. omni_backend selects
    auto/vertex/gemini_api; an existing conversation retains its backend.
    timeout_seconds applies per turn. Results include every completed segment
    and an interaction_id for further continuation.

    omni_model left unset continues a previous_interaction_id on the model
    that created it and extends an input_video_uri on gemini-omni-1.2-flash;
    naming a model other than the one an interaction was created on is
    refused. gemini-omni-1.1-flash is retired here: a call naming it runs on
    1.2 with a warning, and an interaction created on it cannot be extended
    -- pass its saved video as input_video_uri instead.

    Args:
        ctx: MCP context with application state.
        prompt: How the scene continues ("Extend this video: the camera
            pans across the mountains"). On a conversation it must ask to
            extend, because the service infers the task. In a timecode, 0s
            is the start of the new footage, not of the source.
        previous_interaction_id: interaction_id of a clip this server
            rendered. Give exactly one of this and input_video_uri. An
            upscale's interaction is extended from the clip it saved.
        input_video_uri: A clip of your own to extend (gs://, http(s)://,
            or file:// inside DATA_FOLDER). Extending an uploaded video is
            not available in the European Economic Area (Google's
            restriction): the service refuses, and this server cannot check
            it first.
        omni_model: Unset continues an interaction on the model that created
            it, and extends an upload on gemini-omni-1.2-flash.
        omni_backend: "auto", "vertex" or "gemini_api". Unset, the server's
            OMNI_BACKEND applies. "auto" uses the Gemini API whenever
            GEMINI_API_KEY is set, even on a Vertex AI server -- a different
            project, billing path and quota pool -- so a deployment that must
            keep Omni traffic on Vertex AI passes "vertex" or sets
            OMNI_BACKEND=vertex. The response reports `backend` and
            `backend_reason`. A choice that contradicts the backend a
            previous_interaction_id was created on is refused.
        times: How many turns to chain (default 1), each threaded into the
            next, up to the cumulative ceiling: 30s on the Gemini API; Vertex
            AI keeps the last 30s of each input and returns up to 40s. At
            most 30 divided by the turn length per call (3 at the default
            10s).
        resolution: "360p", "720p", "1080p" or "4K"; unset keeps the
            resolution recorded for previous_interaction_id, else the
            service default.
        reference_image_uris: Up to 6 images bound to `<IMAGE_REF_0>`, ...,
            for bringing a character or object into the continuation. Sent
            on the first turn; later turns continue from the interaction,
            which already holds them.
        output_gcs_uri: A gs:// prefix ending in / (Vertex AI writes a new
            object under it), Vertex AI only, used for the final turn.
        timeout_seconds: Deadline for each turn, default 210s. A chain can
            outlast a host's ceiling in total; completed turns are returned
            as they finish, so a cut-off chain resumes from the last
            interaction_id.
        max_cost_usd: Refuse before the first turn renders if the projected
            bill is over this. A dry run reports `would_be_refused` instead.
        dry_run: Quote every turn and render nothing. Each turn bills the
            assembled clip, so the quote needs the source's length: measured
            for an upload, read from the sidecar for an interaction, and
            otherwise taken at a conservative bound.
        duration_seconds: Length of each appended segment, whole seconds
            from 3 to 10 (default 10).

    Returns:
        JSON with the final video_url, the interaction_id to keep extending
        from, and one segment record per turn.
    """
    try:
        app_ctx = ctx.request_context.lifespan_context
        # Which backend serves this call, decided once and reported; see
        # _omni_backend_decision.
        # Read once: the backend, the model, the source length and the
        # render all come from this one record.
        prior = await _prior_interaction_or_empty(
            app_ctx.videos_dir, previous_interaction_id
        )
        # An upscale interaction is extended from the clip it saved: it is no
        # single clip to continue by id (the service refused an upscale of
        # one). Only when it is the one source; both is refused below.
        upscaled_from = previous_interaction_id
        saved_clip = (
            _upscale_interaction_clip(
                prior, previous_interaction_id, "extend_video_omni"
            )
            if input_video_uri is None
            else None
        )
        if saved_clip is not None:
            input_video_uri, previous_interaction_id = saved_clip, None
        omni_backend_resolved, omni_backend_reason = await _resolve_omni_backend(
            app_ctx,
            omni_backend,
            need_gcs=bool(output_gcs_uri),
            previous_interaction_id=previous_interaction_id,
            task="extend",
            prior=prior,
        )
        # One lookup, ahead of everything that depends on the model and ahead
        # of the dry_run branch, so a quote cannot price a call the real run
        # would refuse: it settles the model (the recorded one when
        # omni_model is unset; a hard refusal when it contradicts an explicit
        # one), the source's length (the projection below) and the backend
        # the whole chain pins to.
        requested_model = _continuation_omni_model(omni_model, prior)
        _validate_omni_model(requested_model)
        spec = omni_spec(requested_model, vertexai=omni_backend_resolved == "vertex")
        _validate_omni_continuation_source(
            spec, prior, previous_interaction_id, task="extend"
        )
        # Evaluated once: it validates (and logs) the source, and its answer
        # decides what every turn of the chain sends.
        source_uri = _omni_gcs_source(
            app_ctx, input_video_uri, omni_backend_resolved, model=spec.model
        )
        normalized_resolution, resolution_origin = _omni_sent_resolution(
            spec,
            _validate_omni_resolution(spec, resolution),
            prior,
            # The upscale's id: its recorded resolution is kept, where an
            # upload would get the 720p default.
            previous_interaction_id or (upscaled_from if saved_clip else None),
        )
        reference_images = list(reference_image_uris or [])
        if len(reference_images) > MAX_OMNI_INPUT_IMAGES:
            # The same cap generate_video_omni front-loads, and for the same
            # reason: every image is buffered whole (each up to
            # MAX_FETCH_BYTES) and base64-inflated into one request body.
            raise ValueError(
                f"Too many reference images ({len(reference_images)}); "
                f"{spec.model} accepts at most {MAX_OMNI_INPUT_IMAGES} here."
            )
        validate_omni_request(
            spec.model,
            resolution=resolution,
            task="extend",
            reference_image_count=len(reference_images),
            duration_seconds=duration_seconds,
        )
        if duration_seconds is not None:
            _validate_duration_seconds(duration_seconds)
            longest = spec.extension_step_seconds
            if not 3 <= duration_seconds <= longest or duration_seconds != round(
                duration_seconds
            ):
                raise ValueError(
                    "Extension duration_seconds must be a whole second from 3 "
                    f"to {longest}."
                )

        if (previous_interaction_id is None) == (input_video_uri is None):
            raise ValueError(
                "Give exactly one source to extend: previous_interaction_id "
                "(a clip this server generated) or input_video_uri (a clip to "
                "upload)."
            )

        max_turns = spec.max_extended_seconds // int(
            duration_seconds or spec.extension_step_seconds
        )
        if not isinstance(times, int) or isinstance(times, bool) or times < 1:
            raise ValueError(f"times must be an integer >= 1, got {times!r}.")
        if times > max_turns:
            raise ValueError(
                f"times={times}: at most {max_turns} turns are allowed per call. "
                "Continue from the last interaction_id in a subsequent call."
            )
        # Whether a turn that continues an interaction can carry a duration
        # is a live question with one switch; when it cannot, a duration the
        # chain would drop is refused here rather than mid-chain.
        cannot_honour = omni_extension_duration_refusal(
            spec,
            duration_seconds=duration_seconds,
            continues_interaction=previous_interaction_id is not None,
            times=times,
        )
        if cannot_honour:
            raise ValueError(cannot_honour)

        # Every gs:// URI this call reads or writes, not only the output: an
        # input read with ambient credentials is the same exposure.
        warnings: list[str] = _gcs_allowlist_warnings(
            app_ctx,
            input_video_uri,
            *reference_images,
            output_gcs_uri,
        )
        if saved_clip is not None and upscaled_from:
            warnings.insert(
                0, _upscale_interaction_note(upscaled_from, saved_clip, dry_run)
            )
        warnings.extend(_gcs_prefix_warning(output_gcs_uri))
        if output_gcs_uri:
            bucket = _parse_gcs_bucket(output_gcs_uri)
            if bucket is None:
                raise ValueError(
                    f"output_gcs_uri must start with gs://: {output_gcs_uri}"
                )
            # Raises for a bucket outside a configured allowlist; the
            # no-allowlist warning it returns is already in `warnings`.
            _assert_gcs_bucket_allowed(bucket, app_ctx.allowed_gcs_buckets)
        # On the quote and the render alike; see generate_video_omni.
        reroute = omni_model_reroute_warning(requested_model)
        if reroute:
            warnings.append(reroute)
        # Every extension turn returns the assembled clip, so every turn is
        # large whatever its length; and output_gcs_uri rides only on the
        # final turn, so a chain's earlier turns come back inline regardless.
        delivery = omni_inline_delivery_policy(
            spec,
            vertexai=omni_backend_resolved == "vertex",
            output_gcs_uri=output_gcs_uri,
            resolution=normalized_resolution,
            task="extend",
            output_seconds=None,
            chain_turns=times,
        )
        if delivery:
            warnings.append(delivery)
        # The deadline applies per turn, and the shortest turn renders the
        # source plus one step; an unrecorded source counts as nothing, so this
        # judges the turn at its least.
        slow_tier = omni_slow_tier_warning(
            resolution=normalized_resolution,
            kind="extend",
            output_seconds=(prior.duration_seconds or 0)
            + (duration_seconds or spec.extension_step_seconds),
            timeout_seconds=timeout_seconds,
        )
        if slow_tier:
            warnings.append(slow_tier)

        billed_resolution = _omni_billed_resolution(spec, normalized_resolution)

        if dry_run:
            for uri in reference_images:
                _assert_local_source(uri, app_ctx.data_folder, "reference_uri")
            _assert_local_source(
                input_video_uri, app_ctx.data_folder, "input_video_uri"
            )
            source_duration = prior.duration_seconds
            if source_duration is None and input_video_uri:
                # An uploaded source's length is the base every turn is billed
                # on, and it is sitting in a local file the quote can open.
                # Assuming the documented maximum instead quoted $0.6771 for a
                # render that billed $0.4396 — over-stating, so the invariant
                # held, but by 54% on a figure that was free to measure. Only
                # local sources: a quote does not download a remote one.
                source_bytes = await _fetch_local_source(ctx, input_video_uri)
                if source_bytes is not None:
                    source_duration = await _probe_media(
                        measure_video_duration_bytes, source_bytes
                    )
                    if source_duration is not None:
                        # And a source the real run would refuse must not be
                        # quoted as if it would run.
                        _ = await _check_omni_source_video(
                            spec,
                            source_bytes,
                            vertexai=(omni_backend_resolved == "vertex"),
                            task="extend",
                        )
            # Each turn renders the ASSEMBLED clip, so the source is the base
            # every turn is billed on top of and the chain stops when it
            # reaches the documented ceiling.
            at_ceiling = omni_extension_refusal(spec, source_duration)
            if at_ceiling:
                # A quote for a call that cannot run refuses, as every other
                # pre-flight here does; planned_turns: 0 beside a null cost
                # said the same thing in a way nothing acts on.
                raise ValueError(at_ceiling)
            # Priced, not merely planned: with no measured source the loop
            # below runs every requested turn, so every one is priced, at the
            # backend's cumulative ceiling (40s on Vertex, 30s on the Gemini
            # Developer API).
            chain_is_vertex = omni_backend_resolved == "vertex"
            turn_lengths = omni_extension_priced_lengths(
                spec,
                source_duration,
                times,
                vertexai=chain_is_vertex,
                per_turn_seconds=duration_seconds,
            )
            appended = omni_extension_appended_seconds(
                source_duration, turn_lengths, spec=spec
            )
            payload: dict[str, Any] = {
                "dry_run": True,
                "message": "Estimate only — nothing was generated",
                "model": spec.model,
                "backend": omni_backend_resolved,
                "backend_reason": omni_backend_reason,
                **_omni_requested_model(requested_model, spec),
                "resolution": billed_resolution,
                "resolution_source": _omni_resolution_provenance(
                    resolution_origin,
                    "a dry run renders nothing whose resolution could be measured",
                ),
                "times": times,
                # Each turn renders the assembled clip, so these three are
                # different numbers and conflating them was the under-quote:
                # what gets BILLED is the sum of the assembled lengths, what
                # the caller ENDS UP WITH is the last one, and what is NEW is
                # only the growth over the source.
                "billed_seconds": round(sum(turn_lengths), 6),
                "assembled_seconds": turn_lengths[-1] if turn_lengths else 0.0,
                "turn_output_seconds": turn_lengths,
                "planned_turns": len(turn_lengths),
                **({"appended_seconds": appended} if appended is not None else {}),
                "duration_source": (
                    "projection: a turn renders the ASSEMBLED clip, not only "
                    "the seconds it appends (measured: a 3.01s source came "
                    "back 13.01s), so every turn re-bills the footage before it"
                    + (
                        f" (source measured at {source_duration:g}s, leaving "
                        f"room for {len(turn_lengths)} turn(s) under omni's "
                        f"{spec.max_extended_seconds}s total)."
                        if source_duration is not None
                        else f" (source length is unknown, so every requested turn "
                        f"is priced at the {spec.max_extended_seconds}s cumulative "
                        "ceiling as an upper bound until the source can be measured)."
                    )
                    + " The real run reports the measured length of each turn."
                ),
                "cumulative_cap_seconds": spec.max_extended_seconds,
            }
            if source_duration is not None:
                payload["source_duration_seconds"] = source_duration
                warnings.extend(
                    _omni_cumulative_cap_warning(
                        spec,
                        source_duration,
                        times,
                        len(turn_lengths),
                        duration_seconds,
                    )
                )
            if spec.extension_window_seconds:
                payload["duration_source"] = (
                    "projection: Vertex retains up to the final "
                    f"{spec.extension_window_seconds}s of each input and bills "
                    "the assembled output on every turn. "
                    + (
                        f"Source measured at {source_duration:g}s."
                        if source_duration is not None
                        else f"Source length unknown; each turn uses the {spec.max_extended_seconds}s output bound."
                    )
                )
            payload["estimated_cost"] = _omni_extension_quote(
                spec.model, billed_resolution, turn_lengths
            )
            if warnings:
                payload["warnings"] = warnings
            # Emitted last, so the cap warning added above reaches the
            # notification channel and not just the quote body.
            await _emit_warnings(ctx, warnings)
            if max_cost_usd is not None:
                projected_usd = _usd_of(payload.get("estimated_cost"))
                if projected_usd > max_cost_usd:
                    payload["would_be_refused"] = True
                    payload.setdefault("warnings", []).append(
                        _over_budget_message(
                            projected_usd,
                            max_cost_usd,
                            len(turn_lengths),
                            unmeasured=source_duration is None,
                            # A remote upload the quote did not read; the
                            # real run fetches it (or, for a 1.2 gs:// source,
                            # downloads it to measure) before this cap.
                            measured_on_real_run=bool(input_video_uri)
                            and not input_video_uri.startswith("file://"),
                        )
                    )
            return _respond(app_ctx, payload)

        reference_image_bytes = [
            await _fetch_omni_media(ctx, uri, "reference_image_uri")
            for uri in reference_images
        ]
        if source_uri:
            # Sent by URI, so nothing is inlined; downloaded only to measure,
            # because the measured length is what trims the chain, prices each
            # turn and drives the max_cost_usd projection below.
            input_video_bytes = None
            source_to_measure = await _fetch_omni_gcs_source_to_measure(
                ctx, source_uri, warnings
            )
        else:
            input_video_bytes = await _fetch_optional_omni_media(
                ctx, input_video_uri, "input_video_uri"
            )
            source_to_measure = input_video_bytes
        if source_to_measure is not None:
            warnings.extend(
                await _check_omni_source_video(
                    spec,
                    source_to_measure,
                    vertexai=(omni_backend_resolved == "vertex"),
                    task="extend",
                    inline=input_video_bytes is not None,
                )
            )
            source_duration = await _probe_media(
                measure_video_duration_bytes, source_to_measure
            )
        elif input_video_uri:
            # An uploaded source that could not be measured; there is no
            # interaction record to fall back on.
            source_duration = None
        else:
            source_duration = prior.duration_seconds
        at_ceiling = omni_extension_refusal(spec, source_duration)
        if at_ceiling:
            raise ValueError(at_ceiling)

        # The turns that fit under the ceiling -- the same list the quote
        # prices, computed once here so the loop below, this warning and the
        # max_cost_usd projection cannot disagree about how many turns run.
        chain_is_vertex = omni_backend_resolved == "vertex"
        turn_lengths = omni_extension_priced_lengths(
            spec,
            source_duration,
            times,
            vertexai=chain_is_vertex,
            per_turn_seconds=duration_seconds,
        )
        effective_times = times
        if source_duration is not None:
            # Only a MEASURED source may shorten the chain. With no
            # measurement, the source is taken to be at the cumulative
            # ceiling, so every turn is priced at it -- deliberately
            # pessimistic for pricing -- and clamping the loop on that would
            # render fewer turns than a short real source can actually take.
            effective_times = len(turn_lengths)
            warnings.extend(
                _omni_cumulative_cap_warning(
                    spec, source_duration, times, effective_times, duration_seconds
                )
            )
        if max_cost_usd is not None:
            # Refuse BEFORE the first turn bills — same quadratic hazard as
            # loop_extend, same cap. Projected off the turns that will really
            # run: the clamped list when the source was measured, and every
            # requested turn priced at the ceiling when it was not -- the
            # first cut of this fix clamped the projection but not the loop on
            # that path, so a $10 cap passed a 3-turn $9.14 and billed 4.
            projected_usd = _usd_of(
                _omni_extension_quote(
                    spec.model,
                    billed_resolution,
                    turn_lengths,
                )
            )
            if projected_usd > max_cost_usd:
                raise ValueError(
                    _over_budget_message(
                        projected_usd,
                        max_cost_usd,
                        len(turn_lengths),
                        unmeasured=source_duration is None,
                    )
                )

        # One backend for the whole chain. Resolved once, before the first
        # turn, because every turn after it carries a previous_interaction_id
        # that only the minting backend can resolve — and the last turn's
        # output_gcs_uri would otherwise have pulled it onto Vertex on its own.
        chain_backend = omni_backend_resolved

        segments: list[dict[str, Any]] = []
        chain_id = previous_interaction_id
        pending_video: bytes | None = input_video_bytes
        pending_uri = source_uri

        def _payload(cancelled: bool = False) -> dict[str, Any]:
            chain_appended = omni_extension_appended_seconds(
                source_duration,
                [
                    float(seg["duration_seconds"])
                    for seg in segments
                    if isinstance(seg.get("duration_seconds"), (int, float))
                ],
                spec=spec,
            )
            """Assemble the response from whatever turns have completed.

            Built as a closure so a cancellation or a mid-chain failure can
            return the SAME shape as a clean run. Every completed turn was
            billed; dropping them on the floor would leave the caller unable
            even to resume from the last interaction_id they paid for — the
            hole loop_extend and generate_clip each carry an explicit handler
            for, and which this loop reintroduced.
            """
            done = len(segments)
            final = segments[-1] if segments else {}
            body: dict[str, Any] = {
                "message": (
                    f"Video extended over {done} of {effective_times} turn(s)"
                    if done != effective_times
                    else f"Video extended over {effective_times} turn(s)"
                ),
                "video_url": final.get("video_url"),
                "interaction_id": final.get("interaction_id"),
                # The backend every turn ran on, which is the one that can
                # resolve interaction_id. Left off, _respond stamped the
                # server's default instead: a live Vertex chain (2026-11-09)
                # reported gemini_api, sending a caller who continued from it
                # to a backend that cannot see the interaction.
                "backend": final.get("backend") or chain_backend,
                "backend_reason": omni_backend_reason,
                "model": spec.model,
                **_omni_requested_model(requested_model, spec),
                "resolution": billed_resolution,
                "times": times,
                # What the ceiling allowed, which is what ran. Equal to
                # `times` unless the source was long enough to cut the chain
                # short; the warning above says so when they differ.
                "planned_turns": effective_times,
                "completed_turns": done,
                # Three different numbers, because a turn renders the
                # assembled clip: billed is the sum of what each turn
                # rendered, assembled is the last one (what the caller has),
                # and appended is only the growth over the source — which is
                # unknowable without the source, so it is omitted rather than
                # guessed. Reporting the sum as "appended" overstated a 2-turn
                # chain off a 3s source as 36s of new footage when it made 20s.
                # A turn that could not be measured bills its bound.
                "billed_seconds": round(
                    sum(
                        float(billed)
                        for seg in segments
                        if isinstance(
                            billed := seg.get("duration_seconds")
                            or seg.get("duration_upper_bound_seconds"),
                            (int, float),
                        )
                    ),
                    6,
                ),
                "assembled_seconds": final.get("duration_seconds"),
                **(
                    {"appended_seconds": chain_appended}
                    if chain_appended is not None
                    else {}
                ),
                "segments": segments,
                "cost": _sum_omni_segment_costs(
                    segments, model=spec.model, resolution=billed_resolution
                ),
            }
            if cancelled:
                body["cancelled"] = True
            if warnings:
                body["warnings"] = warnings
            return body

        try:
            for turn in range(1, effective_times + 1):
                await ctx.info(
                    f"Extending video with {spec.model} (turn {turn}/{effective_times})"
                )
                # This turn's source is the last turn's output: its measured
                # length when it was measured, the projection otherwise. The
                # projection alone billed an unmeasurable second turn at a 40s
                # bound after the first measured 16s (16 + 10 = 26s).
                previous = segments[-1] if segments else {}
                measured_previous = (
                    float(previous["duration_seconds"])
                    if str(previous.get("duration_source", "")).startswith("measured")
                    and isinstance(previous.get("duration_seconds"), (int, float))
                    else None
                )
                # A turn appends whole seconds, and near the ceiling fewer than
                # it asked for: 23.03s plus 10 would pass 30s, so that turn
                # asks for 6. Judged only from a length this chain knows (the
                # measured source, the last turn's measured output, or the
                # projection from a measured source); otherwise the turn goes
                # out as asked and the service decides.
                if turn == 1:
                    known_seconds = source_duration
                elif measured_previous is not None:
                    known_seconds = measured_previous
                elif source_duration is not None and turn - 2 < len(turn_lengths):
                    known_seconds = turn_lengths[turn - 2]
                else:
                    known_seconds = None
                turn_duration = duration_seconds
                if known_seconds is not None:
                    fits = omni_extension_turn_seconds(
                        spec, known_seconds, duration_seconds, vertexai=chain_is_vertex
                    )
                    asked = duration_seconds or spec.extension_step_seconds
                    cannot_send = (
                        fits is not None
                        and fits < asked
                        and chain_id is not None
                        and not OMNI_CONVERSATIONAL_EXTENSION_SENDS_DURATION
                    )
                    if fits is None or cannot_send:
                        warnings.append(
                            f"Stopped after {turn - 1} turn(s): the clip is "
                            f"{known_seconds:g}s, and "
                            + (
                                "less than 3s, the shortest turn, fits "
                                if fits is None
                                else f"only {fits}s, which this turn cannot ask for, fits "
                            )
                            + f"under the {spec.max_extended_seconds}s ceiling."
                        )
                        break
                    if fits < asked:
                        turn_duration = fits
                result = await _omni_generate_and_manifest(
                    app_ctx,
                    ctx,
                    prompt=prompt,
                    model=spec.model,
                    # References ride along on the first turn, which is where a
                    # new character is introduced; later turns continue from the
                    # interaction, which already holds them.
                    reference_image_bytes_list=(
                        reference_image_bytes if turn == 1 else None
                    ),
                    input_video_bytes=pending_video,
                    input_video_uri=pending_uri,
                    previous_interaction_id=chain_id,
                    # The record read above is chain_id's on the first turn only.
                    prior=prior if turn == 1 else None,
                    # "A prompt plus a video" is otherwise an edit, so the task
                    # is what makes this an extension. It is passed on EVERY
                    # turn and SENT on some: _build_create_kwargs drops it
                    # whenever a previous_interaction_id is present, which the
                    # API rejects it alongside, so a multi-turn turn relies on
                    # the prompt exactly as the reference's own example does.
                    # Passing None there instead made _select_task_type fall
                    # through to "edit", and every turn after the first went
                    # into its own sidecar — and its warning text — labelled as
                    # an edit of a chain the caller had asked to extend.
                    task="extend",
                    duration_seconds=turn_duration,
                    source_duration_seconds=(
                        source_duration
                        if turn == 1
                        else measured_previous or turn_lengths[turn - 2]
                    ),
                    resolution=normalized_resolution,
                    resolution_origin=resolution_origin,
                    # effective_times, not times: on a chain the ceiling cut
                    # short no turn ever equalled the requested count, so the
                    # caller's GCS destination was silently never sent and the
                    # final clip landed locally.
                    output_gcs_uri=output_gcs_uri if turn == effective_times else None,
                    prefer_backend=chain_backend,
                    backend_reason=omni_backend_reason,
                    timeout_seconds=timeout_seconds,
                    manifest_extra={
                        "kind": "omni_extension",
                        "turn": turn,
                        "of_turns": effective_times,
                        "input_video_uri": input_video_uri if turn == 1 else None,
                    },
                )
                segments.append(result)
                chain_id = result.get("interaction_id")
                # Turn 1 decides the backend for every turn after it.
                chain_backend = chain_backend or result.get("backend")
                # Every later turn continues the interaction, so the bytes are
                # sent once and only once.
                pending_video = None
                pending_uri = None
                for warning in result.get("warnings") or []:
                    if warning not in warnings:
                        warnings.append(warning)
        except asyncio.CancelledError:
            # A cancellation is a BaseException, so the tool's own handler
            # below never sees it. Record what was paid for, then let the
            # cancellation continue.
            if segments:
                logger.warning(
                    "extend_video_omni cancelled after %d of %d turn(s); "
                    "resume from interaction %s",
                    len(segments),
                    effective_times,
                    chain_id,
                )
            raise
        except Exception as exc:
            if not segments:
                raise
            # Some turns rendered and were billed. Returning a bare error
            # would strand them: the caller could not resume, and could not
            # reconcile the spend either.
            logger.exception("extend_video_omni failed mid-chain")
            partial = _payload()
            # The failing turn's own error is kept: it is the only place the
            # caller learns WHY the chain stopped, and some of those errors
            # carry the fix (a conversational turn the service read as an
            # edit says how to re-prompt it). Replacing it with a generic line
            # dropped that advice in exactly the multi-turn case it was
            # written for.
            partial["error"] = (
                f"Extension stopped after {len(segments)} of {effective_times} turn(s). "
                f"Turn {len(segments) + 1} failed: {exc} "
                "The turns listed in `segments` rendered and were billed; "
                "resume by passing the `interaction_id` above back to this "
                "tool."
            )
            await _emit_warnings(ctx, warnings)
            return _respond(app_ctx, partial)

        payload = _payload()
        await _emit_warnings(ctx, warnings)
        return _respond(app_ctx, payload)
    except Exception as e:
        await ctx.error(f"Video extension failed: {e}")
        logger.exception("Tool error")
        return json.dumps({"error": str(e)})


@mcp.tool()
async def loop_extend(
    ctx: Context[ServerSession, AppContext],
    video_uri: str,
    prompt: str = "continue the action",
    times: int = 1,
    model: VideoModel | TranslatedVideoModel = "veo-3.1-generate-001",
    aspect_ratio: str = "16:9",
    include_audio: bool = True,
    output_gcs_uri: str | None = None,
    max_cost_usd: float | None = None,
    dry_run: bool = False,
) -> str:
    """Extend a Veo-generated video multiple times in one call.

    max_cost_usd: refuse BEFORE the first turn renders if the projected bill
    for the whole chain exceeds this. Unset by default. Worth setting: each
    turn re-bills every second before it, so the cost grows with times²
    — 20 turns off a 10s source is ~1670 billed seconds. A dry run reports
    `would_be_refused: true` instead of refusing, so the number is visible.

    Each Veo extension continues the video by ~7 seconds; this chains them,
    feeding each extended result back in as the source for the next. Veo
    supports up to 20 extensions. Output is 720p. Extension requires a GCS
    output target.

    Args:
        ctx: MCP context with application state
        video_uri: URI of the existing Veo video to extend (gs://, file://)
        prompt: What the continuation should depict
        times: Number of ~7s extensions to chain (1-20). Each turn APPENDS
            ~7s but BILLS the whole assembled clip, so cost grows with the
            square of this — see max_cost_usd.
        model: Veo model
        aspect_ratio: 16:9 or 9:16. An extension continues the source, so the
            rendered aspect ratio is the source's; this value must match it.
            The server does not verify the match — the source is typically a
            remote gs:// clip it cannot probe — so a value that differs from
            the source may be ignored or distort the render rather than being
            refused here.
        include_audio: Generate audio on the extended sections (default True,
            so extending an audio video doesn't go silent)
        output_gcs_uri: GCS output URI (required for extensions)
        dry_run: When True, return only the cost of the extension chain and
            generate nothing. The basis is NOT times x ~7s: MEASURED, a Veo
            turn renders and bills the ASSEMBLED clip (a 4.0s source came
            back 11s billed at 11s), so the chain costs the SUM of the turn
            outputs — source + 7i for turn i — and grows with times². The
            quote reports billed_seconds, assembled_seconds and
            appended_seconds separately for that reason.

            A REMOTE source makes this quote a FLOOR rather than a ceiling —
            the one place a pre-flight in this server can under-state, and
            deliberately so: a dry run is documented offline, and Veo
            publishes no maximum source length to assume instead. MEASURED:
            an 11.01s remote source quoted $0.70 and billed $1.80, 61% under.
            billed_seconds_source says "FLOOR, not an estimate" when it
            applies; the real run measures the source. For a ceiling before
            you spend, quote against a local copy, or compute it: billed =
            sum over turns i of (source + 7i).

    Returns:
        JSON with the final video_url and the ordered list of intermediate
        extension URLs.
    """
    # Outside the try so the cancellation handler can still record the
    # extensions that rendered — and billed — before the caller vanished.
    steps: list[str] = []
    step_warnings: list[str] = []
    current = video_uri
    # The model that actually ran. Seeded with the request so it is bound
    # even if the loop body never executes, then overwritten with the
    # backend-translated ID the impl reports.
    served_model = str(model)
    try:
        app_ctx = ctx.request_context.lifespan_context
        data_dir = app_ctx.data_folder

        if times < 1 or times > 20:
            raise ValueError("times must be between 1 and 20.")

        # Veo has no endpoint on the Gemini Developer API. Checked here,
        # before the quote, so a dry run cannot price a chain that cannot run.
        from .video import validate_render_options

        validate_render_options(
            model,
            backend=_backend_of(app_ctx, str(model)),
        )
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
            # Refuse a local source the real run would reject, so a quote does
            # not price an extension that cannot run. A gs:// source was already
            # allowlist-checked above and still prices.
            _assert_local_source(video_uri, data_dir, "video_uri")
            # The source is the base the chain grows from, and it is sitting
            # in a local file the quote can open. extend_video_omni measures
            # its source here; this one reported added_seconds as a bare
            # number with no base, so nothing in the quote could be
            # reconciled against the file that produced it.
            source_seconds = await _probe_local_video_seconds(ctx, video_uri)
            # MEASURED: a 4.0s source extended once returned 11.0s and billed
            # 11s. Each turn re-bills the assembly, so the chain costs the SUM
            # of the turn outputs, not the footage it added.
            turn_lengths = veo_extension_output_lengths(source_seconds, times)
            appended = times * VEO_EXTENSION_STEP_SECONDS
            quote: dict[str, Any] = {
                "dry_run": True,
                "message": "Estimate only — nothing was generated",
                "model": model,
                "resolution": "720p",
                "resolution_source": "fixed: loop_extend takes no resolution parameter",
                "times": times,
                # Three different numbers, and conflating them was the bug:
                # what is BILLED is the sum of the assembled lengths, what the
                # caller ENDS UP WITH is the last one, and what is NEW is only
                # the footage the chain appended.
                "appended_seconds": appended,
                "added_seconds": appended,  # retained: prior name for appended
            }
            if turn_lengths:
                quote["source_duration_seconds"] = round(source_seconds or 0.0, 3)
                quote["source_duration_source"] = "measured from the local source"
                quote["billed_seconds"] = round(sum(turn_lengths), 3)
                quote["assembled_seconds"] = turn_lengths[-1]
                quote["final_duration_seconds"] = turn_lengths[-1]
                quote["turn_output_seconds"] = turn_lengths
                quote["estimated_cost"] = _video_cost(
                    model,
                    float(sum(turn_lengths)),
                    resolution="720p",
                    include_audio=include_audio,
                    presnapped=True,  # measured lengths must not re-snap
                )
                quote["billed_seconds_source"] = (
                    f"projection from a measured {source_seconds:g}s source: a "
                    "Veo turn renders the ASSEMBLED clip, not the "
                    f"{VEO_EXTENSION_STEP_SECONDS:g}s it appends (measured: a "
                    "4.0s source came back 11.0s billed at 11s), so every turn "
                    "re-bills the footage before it and the cost grows "
                    "quadratically in `times`."
                )
                if times > 1:
                    step_warnings.append(
                        f"This chain bills {sum(turn_lengths):g}s to append "
                        f"{appended:g}s: each turn re-renders everything before "
                        f"it. Cost grows with times squared — {times} turns is "
                        f"{sum(turn_lengths) / appended:.1f}x the appended "
                        "footage, and 20 turns off this source would bill "
                        f"{sum(veo_extension_output_lengths(source_seconds, 20)):g}s."
                    )
            else:
                # No honest point estimate exists without the source length,
                # and quoting the floor as if it were the price is the bug
                # this whole path just had. Say so instead.
                quote["source_duration_seconds"] = None
                quote["source_duration_source"] = (
                    "not measured: the source is remote, so a dry run — which "
                    "is documented as offline — does not download it"
                )
                quote["billed_seconds"] = None
                quote["minimum_billed_seconds"] = appended
                quote["estimated_cost_is_floor"] = True
                quote["estimated_cost"] = _video_cost(
                    model,
                    float(appended),
                    resolution="720p",
                    include_audio=include_audio,
                    presnapped=True,
                )
                quote["billed_seconds_source"] = (
                    "FLOOR, not an estimate: a Veo turn re-bills the assembled "
                    "clip, so the true cost is this plus the source's length "
                    f"charged once per turn ({times}x). The source is remote "
                    "and a dry run does not download it, so its length is "
                    "unknown here. Re-quote against a local copy for a real "
                    "number."
                )
                step_warnings.append(
                    "loop_extend cost for a remote source is a FLOOR, not an "
                    f"estimate: add {times}x the source's length to it. A "
                    "local source is measured and quoted exactly."
                )
            if max_cost_usd is not None:
                projected_usd = _usd_of(quote.get("estimated_cost"))
                if projected_usd > max_cost_usd:
                    quote["would_be_refused"] = True
                    step_warnings.append(
                        _over_budget_message(
                            projected_usd, max_cost_usd, times, floor=not turn_lengths
                        )
                    )
            if step_warnings:
                quote["warnings"] = list(step_warnings)
                await _emit_warnings(ctx, step_warnings)
            return _respond(app_ctx, quote)

        # Validate the initial gs:// source against the allowlist (intermediate
        # outputs land in the already-validated gcs_uri bucket).

        # Measured up front: the projection below needs it, and the manifest
        # after the loop reuses it rather than probing twice.
        source_seconds = await _extension_source_seconds(
            ctx, video_uri, allow_remote=True
        )
        if max_cost_usd is not None:
            # Refuse BEFORE the first turn bills. This is the tool where a cap
            # earns its keep: the bill grows with the square of `times`.
            projection = veo_extension_billing(
                turns=times, source_seconds=source_seconds
            )
            projected_usd = _usd_of(
                _video_cost(
                    model,
                    float(projection.billed_seconds),
                    resolution="720p",
                    include_audio=include_audio,
                    presnapped=True,
                )
            )
            if projected_usd > max_cost_usd:
                raise ValueError(
                    _over_budget_message(
                        projected_usd,
                        max_cost_usd,
                        times,
                        floor=not projection.turn_output_seconds,
                    )
                )

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

        # What the caller ends up with, measured — before the manifest is
        # assembled, because the manifest's cost line prices this. The chain
        # reported how many times it ran and what it cost, but never how long
        # the result was, which is the one number that says whether the
        # extensions actually landed.
        result: dict[str, Any] = {
            "message": f"Extended video {times} time(s)",
            "video_url": current,
            "model": served_model,
            "times": times,
            "extension_steps": steps,
        }
        facts = await _measure_rendered_video(
            result, resolution_key="resolution", requested_resolution=None
        )
        if facts.duration_source:
            result["duration_seconds"] = round(facts.duration_seconds or 0.0, 3)
            result["duration_source"] = "measured from the assembled video"
        # The pre-flight measured this source and projected 11.0s; the real
        # run then delivered to GCS, could not open the turns, and fell all
        # the way back to appended-only — reporting a cost BELOW its own quote
        # for the identical call. Only the OUTPUT went remote. The source is
        # still local and still measurable, so the floor is the projection,
        # not the appended footage.
        # Every turn is billed at its OWN assembled length, so the chain's cost
        # needs all of them, not just the one the caller keeps.
        turn_seconds: list[float] = []
        for step_url in steps:
            if not (isinstance(step_url, str) and step_url.startswith("file://")):
                turn_seconds = []
                break
            step_measured = await _probe_media(
                measure_video_duration, Path(step_url[7:])
            )
            if step_measured is None:
                turn_seconds = []
                break
            turn_seconds.append(step_measured)
        result.setdefault(
            "duration_source",
            "not measured: the assembled video was delivered to GCS, which a "
            "local probe does not open",
        )
        result.setdefault("resolution", "720p")
        result.setdefault(
            "resolution_source",
            facts.resolution_source
            or "assumed: loop_extend takes no resolution parameter and the "
            "assembled video could not be measured",
        )
        manifest = _assemble_loop_extend_manifest(
            source_video_uri=video_uri,
            prompt=prompt,
            served_model=served_model,
            aspect_ratio=aspect_ratio,
            steps=steps,
            final_video_url=current,
            include_audio=include_audio,
            warnings=step_warnings,
            resolution=str(result.get("resolution") or "720p"),
            turn_seconds=turn_seconds,
            source_seconds=source_seconds,
        )
        for key in (
            "billed_seconds",
            "billed_seconds_source",
            "appended_seconds",
            "turn_output_seconds",
        ):
            if key in manifest:
                result[key] = manifest[key]
        if manifest.get("cost"):
            result["cost"] = manifest["cost"]
        if manifest.get("warnings"):
            result["warnings"] = manifest["warnings"]
            # Mirror the extension-chain warnings (e.g. a negative seed that
            # was not sent) onto the notification channel.
            await _emit_warnings(ctx, manifest["warnings"])
        sidecar_url = _write_sidecar(current, manifest)
        if sidecar_url:
            result["sidecar_url"] = sidecar_url
        else:
            result["manifest"] = manifest
        return _respond(app_ctx, result)
    except asyncio.CancelledError:
        # Same hole generate_clip had: a cancellation is a BaseException, so
        # the handler below never ran and a caller that disconnected after
        # three of five extensions lost the entire chain's record. Each
        # extension was billed; write the partial manifest, then let the
        # cancellation continue.
        if steps:
            partial = _assemble_loop_extend_manifest(
                source_video_uri=video_uri,
                prompt=prompt,
                served_model=served_model,
                aspect_ratio=aspect_ratio,
                steps=steps,
                final_video_url=current,
                include_audio=include_audio,
                warnings=step_warnings,
            )
            partial["cancelled"] = True
            sidecar_url = _write_sidecar(current, partial)
            logger.warning(
                "loop_extend cancelled after %d extension(s); manifest written to %s",
                len(steps),
                sidecar_url or "nowhere (the last extension is not a local file)",
            )
        raise
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
        help="Mount path for the sse transport (e.g., /custom). Ignored by "
        "streamable-http, which always serves /mcp.",
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


# Loopback spellings the SDK treats as "protect with a localhost Host
# allowlist"; anything else is an interface an operator deliberately exposed.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def _transport_security_for(host: str) -> TransportSecuritySettings:
    """DNS-rebinding settings matching the host, re-derived at run time.

    Mirrors the SDK's constructor rule (mcp.server.fastmcp): a loopback bind
    keeps the localhost Host/Origin allowlist that stops a browser page from
    rebinding to a local server, while a non-loopback bind — which only
    happens when an operator asked for it — disables the check, because the
    Host header that reaches the server is the container/LAN name, which no
    allowlist assembled here could enumerate.
    """
    if host in _LOOPBACK_HOSTS:
        return TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=["127.0.0.1:*", "localhost:*", "[::1]:*"],
            allowed_origins=[
                "http://127.0.0.1:*",
                "http://localhost:*",
                "http://[::1]:*",
            ],
        )
    return TransportSecuritySettings(enable_dns_rebinding_protection=False)


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
        resolved_host = _resolve_http_host(getattr(args, "host", None))
        mcp.settings.host = resolved_host
        # The SDK derives DNS-rebinding protection from the host AT
        # CONSTRUCTION, and `mcp` is built at import with the default
        # 127.0.0.1 — so its Host allowlist was frozen to localhost before
        # this line ever ran. Left alone, a container bind to 0.0.0.0 then
        # rejected every non-localhost Host header with 421, breaking the
        # very container HTTP use case the bind change above exists for.
        # Re-derive it from the host actually chosen, mirroring the SDK's own
        # rule: lock to localhost when bound to loopback, and when an operator
        # has explicitly bound a public interface, do not second-guess it with
        # a Host allowlist we cannot possibly populate (the container's
        # hostnames are unknown here).
        mcp.settings.transport_security = _transport_security_for(resolved_host)
        if getattr(args, "port", None):
            mcp.settings.port = args.port  # type: ignore[union-attr]
        logger.info(
            "Serving %s on %s:%s", transport, mcp.settings.host, mcp.settings.port
        )

    mcp.run(transport=transport, mount_path=args.mount_path)


if __name__ == "__main__":
    main()
