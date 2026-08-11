"""Omni video generation helpers (Interactions API).

Support for Google's ``gemini-omni-flash-preview`` video model. Unlike the VEO
models in ``video.py`` (which use the long-running ``generate_videos``
operation), the Omni model is driven through the **Interactions API**:
``client.interactions.create(..., background=True)`` starts a server-side
render, ``client.interactions.get(...)`` is polled until the interaction
completes, and multi-turn conversational editing is done by threading a
``previous_interaction_id`` (the server holds the prior video context).

Request/response shapes follow the Vertex AI "Use Gemini Omni Flash …to
generate videos" REST reference and the Interactions API docs:
  * media rides INSIDE ``input`` as flattened parts
    ({type: 'text'|'image'|'video', text|data|uri, mime_type}) — there are no
    separate image/video kwargs;
  * ``response_format`` is a LIST of one object
    ``[{'type': 'video', 'aspect_ratio': ..., 'duration': 'Ns',
    'delivery': 'uri', 'gcs_uri': ...}]``. aspect_ratio ("16:9"/"9:16") and
    duration ("3s".."10s") live here; ``delivery='uri'`` + ``gcs_uri`` sends
    output to Cloud Storage, otherwise the video bytes come back inline;
  * ``generation_config={'video_config': {'task': ...}}`` carries the task
    type (text_to_video / image_to_video / reference_to_video / edit);
  * ``background=True`` runs the (minute-plus) render asynchronously; the
    interaction id is polled until ``status == 'completed'``;
  * a finished interaction carries the clip in a ``model_output`` step's
    ``content[]`` video part (inline base64 ``data`` or a hosted ``uri``);
    ``thought`` and ``user_input`` steps are skipped; newer SDKs also expose
    a convenience ``output_video``.

On Vertex the interactions collection is location ``global``
(…/locations/global/interactions) — the caller must supply a global-location
client. NOTE: the Vertex REST example places ``background`` inside the first
``input`` part; that reads as a doc artifact (it is a request-level flag), so
it is sent top-level here, matching the Interactions API convention. If a
live call rejects it, move it into the input part.
"""

import asyncio
import base64
import functools
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from google import genai

from .video import run_off_loop

# Type for async log callback from MCP context
LogCallback = Callable[[str], Awaitable[None]]

OMNI_MODEL = "gemini-omni-flash-preview"

# Output spec limits documented for gemini-omni-flash-preview.
_SUPPORTED_ASPECT_RATIOS = ("16:9", "9:16")

# Documented output duration bounds (sent as "Ns" in response_format).
_MIN_DURATION = 3
_MAX_DURATION = 10

# Public alias: the tool layer quotes this as the worst case for an edit,
# whose rendered length the service chooses and does not document.
OMNI_MAX_DURATION_SECONDS = _MAX_DURATION

# Cap on the delivered-video download. Mirrors MAX_FETCH_BYTES (50 MB) in
# src/__main__.py — defined here rather than imported because this module does
# not own that file. files.download buffers the whole response body, so without
# this the one delivered-file path would be the only fetch in the server that
# reads an untrusted body uncapped; a 720p/<=10s clip is far under it in
# practice, so this is defence-in-depth matching the rest of the codebase.
_MAX_DELIVERED_VIDEO_BYTES = 50 * 1024 * 1024

# Interval (seconds) between polls of an in-flight background interaction.
_POLL_INTERVAL = 5

# How often an unchanged status is worth repeating. Logging every poll turned
# one routine render into 15 notifications, 13 of them the identical
# "in_progress" line, and an animatic multiplies that by its beat count.
# src/video.py polls for up to 1800s and logs twice; this keeps the state
# changes plus a sparse heartbeat so a long render still shows liveness.
_POLL_LOG_INTERVAL_SECONDS = 60.0

# Interaction statuses that mean "still rendering — keep polling". Everything
# else (failed / cancelled / budget_exceeded / incomplete / requires_action /
# unknown) is terminal: fail fast instead of polling to the full timeout.
_IN_FLIGHT_STATUSES = ("in_progress", "queued")


def _ftyp_brand(data: bytes) -> str | None:
    """Return the ISO-BMFF major brand (the token after the `ftyp` box) or None.

    PNG/JPEG/WebP don't use ftyp; MP4/MOV/HEIC/HEIF all do, distinguished by
    this brand.
    """
    if len(data) >= 12 and data[4:8] == b"ftyp":
        return data[8:12].decode("ascii", "ignore").strip().lower()
    return None


# HEIF-family ftyp brands → still images (HEIC/HEIF).
_HEIF_BRANDS = {"heic", "heix", "heim", "heis", "hevc", "hevx"}
_HEIF_SEQUENCE_BRANDS = {"mif1", "msf1"}


def _detect_image_mime(data: bytes) -> str:
    """Return the exact image MIME type from magic bytes, or raise.

    Detects the formats omni accepts as image input. Rejects unknown bytes
    rather than mislabeling them (e.g. defaulting HEIC to image/png) — an
    accurate label lets the API accept supported formats and reject the rest
    with a clear error, instead of us silently sending the wrong type.
    """
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    brand = _ftyp_brand(data)
    if brand in _HEIF_BRANDS:
        return "image/heic"
    if brand in _HEIF_SEQUENCE_BRANDS:
        return "image/heif"
    raise ValueError(
        "Unrecognized image input format; could not detect a supported image "
        "MIME type from the data. Supported: PNG, JPEG, WebP, HEIC/HEIF, GIF."
    )


def _detect_video_mime(data: bytes) -> str:
    """Return the exact video MIME type from magic bytes, or raise.

    Distinguishes MP4 vs QuickTime/MOV via the ftyp brand and recognizes
    WebM/Matroska, MPEG program/video streams, and AVI, rather than labeling
    every input video as MP4.
    """
    brand = _ftyp_brand(data)
    if brand is not None:
        if brand.startswith("qt"):
            # The SDK's VideoContentMimeType literal set spells MOV "video/mov"
            # (google/genai/_gaos/types/interactions/videocontent.py); the
            # RFC "video/quicktime" is not in it and rides as UnrecognizedStr.
            return "video/mov"
        if brand in _HEIF_BRANDS or brand in _HEIF_SEQUENCE_BRANDS:
            raise ValueError(
                "Input labeled as video but the data is a HEIC/HEIF image."
            )
        return "video/mp4"
    if data[:4] == b"\x1a\x45\xdf\xa3":
        return "video/webm"
    if data[:3] == b"\x00\x00\x01" and data[3:4] in (b"\xba", b"\xb3"):
        return "video/mpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"AVI ":
        # SDK spells AVI "video/avi", not the RFC "video/x-msvideo" — the
        # latter is absent from VideoContentMimeType and rides as
        # UnrecognizedStr. See the MOV note above.
        return "video/avi"
    raise ValueError(
        "Unrecognized video input format; could not detect a supported video "
        "MIME type from the data. Supported: MP4, QuickTime/MOV, WebM, MPEG, AVI."
    )


def _build_input_parts(
    prompt: str,
    image_bytes_list: list[bytes] | None,
    input_video_bytes: bytes | None,
) -> list[dict[str, Any]]:
    """Assemble the flattened ``input`` parts for interactions.create.

    The Interactions API takes flattened media parts ({type, data, mime_type})
    rather than generateContent's inlineData/fileData nesting. Each part's
    mime_type is detected from its bytes and unknown formats are rejected.
    """
    parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for img in image_bytes_list or []:
        parts.append(
            {
                "type": "image",
                "data": base64.b64encode(img).decode("ascii"),
                "mime_type": _detect_image_mime(img),
            }
        )
    if input_video_bytes is not None:
        parts.append(
            {
                "type": "video",
                "data": base64.b64encode(input_video_bytes).decode("ascii"),
                "mime_type": _detect_video_mime(input_video_bytes),
            }
        )
    return parts


def _select_task_type(
    *,
    previous_interaction_id: str | None,
    input_video_bytes: bytes | None,
    image_count: int,
) -> str:
    """Deterministic task-type selection, mirroring the documented semantics.

    Editing (a prior interaction or an input video) wins; multiple images are
    treated as references; a single image is a first frame; else pure text.
    """
    if previous_interaction_id or input_video_bytes is not None:
        return "edit"
    if image_count > 1:
        return "reference_to_video"
    if image_count == 1:
        return "image_to_video"
    return "text_to_video"


def _build_create_kwargs(
    *,
    prompt: str,
    image_bytes_list: list[bytes] | None,
    input_video_bytes: bytes | None,
    previous_interaction_id: str | None,
    aspect_ratio: str,
    duration_seconds_int: int,
    output_gcs_uri: str | None,
) -> dict[str, Any]:
    """Assemble the ``interactions.create`` request body.

    Pure and side-effect-free so it can be validated against the SDK's own
    request normalizer in tests. ``response_format`` is a LIST of one object;
    ``background`` is top-level (the Vertex REST example nests it in input[0],
    which is a doc artifact — the SDK's create schema has it top-level).

    Edit-type requests carry FEWER fields (live-verified against the API,
    which 400s otherwise):
      * ``previous_interaction_id`` conflicts with ``video_config.task``
        ("previous_interaction_id is not allowed when video task is set"),
        so conversational-edit turns send NO generation_config;
      * edit tasks reject ``duration`` in response_format ("Duration cannot
        be set in response format for edit task") — duration and aspect
        ratio cannot be sent for an edit-type request, so neither is. What
        the service then renders is undocumented and is NOT the source's
        length — a measured 3s source came back at 10.01s.
    """
    task_type = _select_task_type(
        previous_interaction_id=previous_interaction_id,
        input_video_bytes=input_video_bytes,
        image_count=len(image_bytes_list or []),
    )
    is_edit = task_type == "edit"

    response_format_item: dict[str, Any] = {"type": "video"}
    if not is_edit:
        response_format_item["aspect_ratio"] = aspect_ratio
        response_format_item["duration"] = f"{duration_seconds_int}s"
    if output_gcs_uri:
        response_format_item["delivery"] = "uri"
        response_format_item["gcs_uri"] = output_gcs_uri

    create_kwargs: dict[str, Any] = {
        "model": OMNI_MODEL,
        "input": _build_input_parts(prompt, image_bytes_list, input_video_bytes),
        "background": True,
        "response_format": [response_format_item],
    }
    if previous_interaction_id is not None:
        # Conversational edit: the server holds the video context; sending a
        # task alongside previous_interaction_id is rejected.
        create_kwargs["previous_interaction_id"] = previous_interaction_id
    else:
        create_kwargs["generation_config"] = {"video_config": {"task": task_type}}
    return create_kwargs


def _field(obj: Any, name: str) -> Any:
    """Read a field from an SDK object or a plain dict interchangeably."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _extract_video_payload(interaction: Any) -> tuple[str | bytes | None, str | None]:
    """Return (inline_data, uri) for the interaction's video output.

    Prefers the SDK's convenience ``output_video`` when present, then scans
    ``steps[].content[]`` for a video part (the documented REST shape).
    """
    output_video = _field(interaction, "output_video")
    if output_video is not None:
        data = _field(output_video, "data")
        uri = _field(output_video, "uri") or _field(output_video, "name")
        if data or uri:
            return data, uri

    for step in _field(interaction, "steps") or []:
        for part in _field(step, "content") or []:
            mime = _field(part, "mime_type") or _field(part, "mimeType") or ""
            is_video = _field(part, "type") == "video" or mime.startswith("video/")
            if not is_video:
                continue
            data = _field(part, "data")
            uri = (
                _field(part, "uri")
                or _field(part, "file_uri")
                or _field(part, "fileUri")
            )
            if data or uri:
                return data, uri
    return None, None


async def _resolve_video_bytes(
    client: genai.Client,
    inline_data: str | bytes | None,
    uri: str | None,
    log_callback: LogCallback | None,
    run_within_deadline: Callable[..., Awaitable[Any]],
) -> bytes:
    """Materialize mp4 bytes from an inline payload or a Files API uri.

    The download runs through the caller's deadline runner like every other
    call in this module: an interaction that completes at t=590s of a 600s
    budget would otherwise hang here forever on an untimed transfer.
    """
    if inline_data:
        if isinstance(inline_data, str):
            return base64.b64decode(inline_data)
        return bytes(inline_data)

    if not uri:
        raise ValueError("Interaction returned no inline video data and no file URI.")

    if log_callback:
        await log_callback(f"Downloading delivered video: {uri}")

    # ASSUMED SDK SHAPE: files.get resolves the uri/name to a file resource and
    # files.download returns its raw bytes.
    file_obj = await run_within_deadline(client.files.get, name=uri)

    # Reject an oversize clip before buffering when the resource advertises its
    # size, so the cap can hold without first allocating the whole body.
    declared_size = _field(file_obj, "size_bytes")
    if declared_size is not None and declared_size > _MAX_DELIVERED_VIDEO_BYTES:
        raise ValueError(
            f"Delivered video size {declared_size} exceeds cap "
            f"{_MAX_DELIVERED_VIDEO_BYTES}: {uri}"
        )

    data = await run_within_deadline(client.files.download, file=file_obj)
    # files.download returns bytes, so a zero-length body is b"" not None: the
    # None-only guard let an empty download write a 0-byte .mp4 and report
    # success. `not data` catches both None and b"".
    if not data:
        raise ValueError(f"Downloaded video file was empty: {uri}")
    # Hard enforcement of the cap even when no size was advertised (or it lied);
    # every other fetch in the server bounds its body the same way.
    if len(data) > _MAX_DELIVERED_VIDEO_BYTES:
        raise ValueError(
            f"Downloaded video ({len(data)} bytes) exceeds cap "
            f"{_MAX_DELIVERED_VIDEO_BYTES}: {uri}"
        )
    return bytes(data)


async def generate_video_omni(
    client: genai.Client,
    prompt: str,
    videos_dir: Path,
    *,
    image_bytes_list: list[bytes] | None = None,
    input_video_bytes: bytes | None = None,
    previous_interaction_id: str | None = None,
    aspect_ratio: str = "16:9",
    duration_seconds: float = 6.0,
    output_gcs_uri: str | None = None,
    timeout_seconds: int = 600,
    log_callback: LogCallback | None = None,
) -> dict[str, Any]:
    """Generate (or conversationally edit) a video with ``gemini-omni-flash-preview``.

    Args:
        client: Google GenAI client (for Vertex, a global-location client).
        prompt: Text description of the video to generate or the edit to apply.
        videos_dir: Directory to save the generated video.
        image_bytes_list: Optional input images (raw bytes). One image is
            treated as a first frame (image_to_video); several are treated as
            references (reference_to_video).
        input_video_bytes: Optional input video (raw bytes) to edit; inlined
            as a video part of the interaction input.
        previous_interaction_id: Optional id of a prior interaction to continue
            a multi-turn conversational edit (server holds the video context).
        aspect_ratio: "16:9" (default) or "9:16". Output is always 720p, 24fps.
        duration_seconds: Desired clip length; clamped to the supported [3, 10]
            seconds and sent as "Ns" in response_format.
        output_gcs_uri: Optional gs:// destination. When set, the video is
            delivered to Cloud Storage (delivery='uri') and video_url is the
            gs:// URI; otherwise the bytes come back inline and are written
            locally as a file:// URL.
        timeout_seconds: Overall deadline covering the create call and the
            background polling loop.
        log_callback: Async callback for progress logging.

    Returns:
        Dict with message, video_url (file:// or gs://), interaction_id, model,
        duration_seconds (clamped int), aspect_ratio, the requested_* originals,
        and warnings (only when non-empty). duration_seconds and aspect_ratio
        are both None for an edit — neither was sent, so neither can be
        reported as fact.
    """
    # Non-fatal warnings surfaced back to the caller.
    warnings: list[str] = []

    # Aspect ratio is a hard error rather than a silent coercion, matching
    # the style in src/video.py.
    if aspect_ratio not in _SUPPORTED_ASPECT_RATIOS:
        raise ValueError(
            f"Unsupported aspect_ratio '{aspect_ratio}'. "
            "Supported values are '16:9' and '9:16'."
        )

    # Clamp duration into the documented [3, 10]s range and send it as "Ns".
    clamped_duration = round(duration_seconds)
    if clamped_duration < _MIN_DURATION:
        clamped_duration = _MIN_DURATION
        warnings.append(
            f"duration_seconds={duration_seconds} below the {_MIN_DURATION}s "
            f"minimum; clamped to {_MIN_DURATION}s."
        )
    elif clamped_duration > _MAX_DURATION:
        clamped_duration = _MAX_DURATION
        warnings.append(
            f"duration_seconds={duration_seconds} above the {_MAX_DURATION}s "
            f"maximum; clamped to {_MAX_DURATION}s."
        )

    create_kwargs = _build_create_kwargs(
        prompt=prompt,
        image_bytes_list=image_bytes_list,
        input_video_bytes=input_video_bytes,
        previous_interaction_id=previous_interaction_id,
        aspect_ratio=aspect_ratio,
        duration_seconds_int=clamped_duration,
        output_gcs_uri=output_gcs_uri,
    )
    task_type = _select_task_type(
        previous_interaction_id=previous_interaction_id,
        input_video_bytes=input_video_bytes,
        image_count=len(image_bytes_list or []),
    )
    if task_type == "edit":
        # The API rejects duration (and task alongside previous_interaction_id)
        # on an edit, so neither duration nor aspect ratio is sent. What the
        # service then renders is NOT the source's length — a measured 3s
        # source came back at 10.01s — and is undocumented, so the warning
        # promises nothing and points at the measured figure instead.
        warnings.append(
            "Edit requests do not send duration_seconds or aspect_ratio — the "
            "API rejects them on an edit task — so the rendered length is "
            "chosen by the service and is NOT predictable from the request or "
            "from the source video's length. A measured 3s source edited with "
            "duration_seconds=4 rendered 10.01s. The response reports the "
            "duration measured from the rendered file, or that same 10s "
            "maximum as a labelled upper bound when the render is delivered "
            "somewhere it cannot be opened to measure (a gs:// URI)."
        )

    if log_callback:
        mode = "editing" if previous_interaction_id else "generating"
        await log_callback(
            f"Starting {mode} interaction with {OMNI_MODEL} (task={task_type})"
        )

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds

    expired = f"Omni video interaction timed out after {timeout_seconds}s."

    async def _run_within_deadline(func: Any, /, **kwargs: Any) -> Any:
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise TimeoutError(expired)
        # run_off_loop, not asyncio.to_thread: a timed-out call cannot be
        # cancelled, and abandoning it on the loop's shared default executor
        # burns a worker that every other request also draws from.
        return await run_off_loop(
            functools.partial(func, **kwargs),
            timeout=remaining,
            message=expired,
        )

    interaction = await _run_within_deadline(
        client.interactions.create, **create_kwargs
    )

    interaction_id = _field(interaction, "id") or _field(interaction, "name")
    if not interaction_id:
        raise ValueError("Interaction create response carried no interaction id.")

    # Poll the background interaction until it leaves the in-flight statuses.
    # `completed` proceeds to extraction; any other terminal status fails fast
    # with the raw status and any error message.
    status = _field(interaction, "status") or _field(interaction, "state")
    # Report state CHANGES and a sparse heartbeat, never one line per poll.
    logged_status: Any = object()
    last_logged_at = loop.time()
    while status in _IN_FLIGHT_STATUSES:
        if loop.time() >= deadline:
            raise TimeoutError(expired)
        if log_callback and (
            status != logged_status
            or loop.time() - last_logged_at >= _POLL_LOG_INTERVAL_SECONDS
        ):
            await log_callback(f"Interaction {interaction_id}: {status}")
            logged_status = status
            last_logged_at = loop.time()
        await asyncio.sleep(_POLL_INTERVAL)
        # google-genai's interactions.get takes the id as `id` (positional-or-
        # keyword), NOT `interaction_id`.
        interaction = await _run_within_deadline(
            client.interactions.get, id=interaction_id
        )
        status = _field(interaction, "status") or _field(interaction, "state")

    if status is not None and status != "completed":
        error = _field(interaction, "error")
        detail = _field(error, "message") if error is not None else None
        raise ValueError(
            f"Omni interaction {interaction_id} ended with status "
            f"'{status}': {detail or 'terminal or unrecognized status'}"
        )

    if log_callback:
        await log_callback("Interaction complete; resolving video output")

    inline_data, uri = _extract_video_payload(interaction)

    if uri and uri.startswith("gs://"):
        # GCS-delivered output: pass the gs:// URI through unchanged (no
        # download, no local write), mirroring the Veo gs:// output path.
        video_url = uri
    else:
        video_bytes = await _resolve_video_bytes(
            client, inline_data, uri, log_callback, _run_within_deadline
        )
        filename = f"{uuid.uuid4()}.mp4"
        filepath = videos_dir / filename
        filepath.write_bytes(video_bytes)
        video_url = f"file://{filepath}"

    result: dict[str, Any] = {
        "message": "Video generated successfully",
        "video_url": video_url,
        "interaction_id": interaction_id,
        "model": OMNI_MODEL,
        # For an edit the duration was never sent, so reporting the request
        # here would describe a render that did not happen — and the caller
        # bills from this field. None means "unknown here, resolve upstream";
        # the request is kept separately so nothing is lost.
        "duration_seconds": None if task_type == "edit" else clamped_duration,
        "requested_duration_seconds": clamped_duration,
        # Same property as duration above: _build_create_kwargs omits
        # aspect_ratio on an edit, so reporting the request here would state a
        # ratio the service never received — editing a 9:16 source under the
        # 16:9 default renders at the source's ratio, not the request's.
        "aspect_ratio": None if task_type == "edit" else aspect_ratio,
        "requested_aspect_ratio": aspect_ratio,
    }

    # Include warnings only when non-empty, matching src/video.py.
    if warnings:
        result["warnings"] = warnings

    return result
