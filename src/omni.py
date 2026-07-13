"""Omni video generation helpers (Interactions API).

Support for Google's ``gemini-omni-flash-preview`` video model. Unlike the VEO
models in ``video.py`` (which use the long-running ``generate_videos``
operation), the Omni model is driven through the **Interactions API**:
``client.interactions.create(...)`` is a single blocking call that returns the
finished result, and multi-turn conversational editing is done by threading a
``previous_interaction_id`` (the server holds the prior video context).
"""

import asyncio
import base64
import io
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from google import genai

# Type for async log callback from MCP context
LogCallback = Callable[[str], Awaitable[None]]

OMNI_MODEL = "gemini-omni-flash-preview"

# Output spec limits documented for gemini-omni-flash-preview.
_MIN_DURATION = 3
_MAX_DURATION = 10
_SUPPORTED_ASPECT_RATIOS = ("16:9", "9:16")

# Interval (seconds) between Files API polls when a video is delivered by URI.
_FILE_POLL_INTERVAL = 2


def _build_create_kwargs(
    *,
    prompt: str,
    previous_interaction_id: str | None,
    image_bytes_list: list[bytes] | None,
    input_video_file: Any | None,
    aspect_ratio: str,
    duration_seconds: int,
) -> dict[str, Any]:
    """Assemble the kwargs for ``client.interactions.create``.

    ASSUMED SDK SHAPE (verify against the live google-genai SDK; if any kwarg
    name differs this is the single place to change):
      * ``input`` carries the text prompt.
      * ``previous_interaction_id`` threads a prior interaction for multi-turn
        conversational editing (the server holds the video context).
      * Input images are attached as ``images=[<raw bytes>, ...]``.
      * An input video for editing is attached as ``video=<uploaded file>``
        (already uploaded via ``client.files.upload``).
      * Output controls (aspect ratio, duration) go under ``config``.
    The Omni model does NOT support negative prompts, seed, or system
    instructions, so those are intentionally never sent.
    """
    kwargs: dict[str, Any] = {
        "model": OMNI_MODEL,
        "input": prompt,
        "config": {
            "aspect_ratio": aspect_ratio,
            "duration_seconds": duration_seconds,
        },
    }
    if previous_interaction_id is not None:
        kwargs["previous_interaction_id"] = previous_interaction_id
    if image_bytes_list:
        kwargs["images"] = list(image_bytes_list)
    if input_video_file is not None:
        kwargs["video"] = input_video_file
    return kwargs


async def _resolve_video_bytes(
    client: genai.Client,
    output_video: Any,
    log_callback: LogCallback | None,
) -> bytes:
    """Return the mp4 bytes from an interaction's ``output_video``.

    Handles both documented delivery modes:
      * inline base64 at ``output_video.data`` (default), and
      * URI delivery, where the video is a Files API resource that must be
        polled until ``state == "ACTIVE"`` and then downloaded.
    """
    inline_data = getattr(output_video, "data", None)
    if inline_data:
        # Inline base64 payload. Accept either str or already-bytes.
        if isinstance(inline_data, str):
            return base64.b64decode(inline_data)
        return bytes(inline_data)

    uri = getattr(output_video, "uri", None) or getattr(output_video, "name", None)
    if not uri:
        raise ValueError("Interaction returned no inline video data and no file URI.")

    if log_callback:
        await log_callback(f"Polling Files API for delivered video: {uri}")

    # Poll the Files API until the resource is ACTIVE, then download its bytes.
    file_obj = await asyncio.to_thread(client.files.get, name=uri)
    while getattr(file_obj, "state", None) != "ACTIVE":
        state = getattr(file_obj, "state", None)
        if state == "FAILED":
            raise ValueError(f"Delivered video file entered FAILED state: {uri}")
        await asyncio.sleep(_FILE_POLL_INTERVAL)
        file_obj = await asyncio.to_thread(client.files.get, name=uri)

    # ASSUMED SDK SHAPE: files.download returns the raw bytes of the file.
    data = await asyncio.to_thread(client.files.download, file=file_obj)
    if data is None:
        raise ValueError(f"Downloaded video file was empty: {uri}")
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
    timeout_seconds: int = 600,
    log_callback: LogCallback | None = None,
) -> dict[str, Any]:
    """Generate (or conversationally edit) a video with ``gemini-omni-flash-preview``.

    Args:
        client: Google GenAI client.
        prompt: Text description of the video to generate or the edit to apply.
        videos_dir: Directory to save the generated video.
        image_bytes_list: Optional input images (raw bytes) to condition on.
        input_video_bytes: Optional input video (raw bytes) to edit; it is
            uploaded via the Files API before the interaction is created.
        previous_interaction_id: Optional id of a prior interaction to continue
            a multi-turn conversational edit (server holds the video context).
        aspect_ratio: "16:9" (default) or "9:16". Output is always 720p, 24fps.
        duration_seconds: Desired duration; clamped to [3, 10] and rounded to an
            int. Values outside that range are clamped with a warning.
        timeout_seconds: Hard timeout for the blocking interactions.create call.
        log_callback: Async callback for progress logging.

    Returns:
        Dict with message, video_url (file://), interaction_id, model,
        duration_seconds (clamped int), aspect_ratio, and warnings (only when
        non-empty).
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

    # Clamp duration into the supported [3, 10] range, rounding to an int.
    clamped_duration = round(duration_seconds)
    if duration_seconds < _MIN_DURATION:
        clamped_duration = _MIN_DURATION
        warnings.append(
            f"duration_seconds={duration_seconds} is below the minimum "
            f"{_MIN_DURATION}s; clamped to {_MIN_DURATION}s."
        )
    elif duration_seconds > _MAX_DURATION:
        clamped_duration = _MAX_DURATION
        warnings.append(
            f"duration_seconds={duration_seconds} exceeds the maximum "
            f"{_MAX_DURATION}s; clamped to {_MAX_DURATION}s."
        )

    # Upload an input video (for editing) via the Files API if provided.
    input_video_file: Any | None = None
    if input_video_bytes is not None:
        if log_callback:
            await log_callback("Uploading input video for editing")
        # ASSUMED SDK SHAPE: files.upload accepts a file-like object and a
        # config carrying the mime type. Change here if the SDK differs.
        input_video_file = await asyncio.to_thread(
            client.files.upload,
            file=io.BytesIO(input_video_bytes),
            config={"mime_type": "video/mp4"},
        )

    create_kwargs = _build_create_kwargs(
        prompt=prompt,
        previous_interaction_id=previous_interaction_id,
        image_bytes_list=image_bytes_list,
        input_video_file=input_video_file,
        aspect_ratio=aspect_ratio,
        duration_seconds=clamped_duration,
    )

    if log_callback:
        mode = "editing" if previous_interaction_id else "generating"
        await log_callback(f"Starting {mode} interaction with {OMNI_MODEL}")

    # interactions.create is a BLOCKING call with unbounded latency, so run it
    # off the event loop and enforce an explicit timeout.
    try:
        interaction = await asyncio.wait_for(
            asyncio.to_thread(client.interactions.create, **create_kwargs),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
        raise TimeoutError(
            f"Omni video interaction timed out after {timeout_seconds}s."
        ) from exc

    output_video = getattr(interaction, "output_video", None)
    if output_video is None:
        raise ValueError("Interaction returned no output_video.")

    if log_callback:
        await log_callback("Interaction complete; resolving video output")

    video_bytes = await _resolve_video_bytes(client, output_video, log_callback)

    filename = f"{uuid.uuid4()}.mp4"
    filepath = videos_dir / filename
    filepath.write_bytes(video_bytes)
    video_url = f"file://{filepath}"

    result: dict[str, Any] = {
        "message": "Video generated successfully",
        "video_url": video_url,
        "interaction_id": getattr(interaction, "id", None),
        "model": OMNI_MODEL,
        "duration_seconds": clamped_duration,
        "aspect_ratio": aspect_ratio,
    }

    # Include warnings only when non-empty, matching src/video.py.
    if warnings:
        result["warnings"] = warnings

    return result
