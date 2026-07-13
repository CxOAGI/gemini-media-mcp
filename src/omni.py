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
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from google import genai

# Type for async log callback from MCP context
LogCallback = Callable[[str], Awaitable[None]]

OMNI_MODEL = "gemini-omni-flash-preview"

# Output spec limits documented for gemini-omni-flash-preview.
_SUPPORTED_ASPECT_RATIOS = ("16:9", "9:16")

# Documented output duration bounds (sent as "Ns" in response_format).
_MIN_DURATION = 3
_MAX_DURATION = 10

# Interval (seconds) between polls of an in-flight background interaction.
_POLL_INTERVAL = 5

# Interaction statuses that mean "still rendering — keep polling". Everything
# else (failed / cancelled / budget_exceeded / incomplete / requires_action /
# unknown) is terminal: fail fast instead of polling to the full timeout.
_IN_FLIGHT_STATUSES = ("in_progress", "queued")


def _sniff_image_mime(data: bytes) -> str:
    """Best-effort mime detection for input images (PNG/JPEG/WebP/GIF)."""
    if data.startswith(b"\x89PNG"):
        return "image/png"
    if data.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    return "image/png"


def _build_input_parts(
    prompt: str,
    image_bytes_list: list[bytes] | None,
    input_video_bytes: bytes | None,
) -> list[dict[str, Any]]:
    """Assemble the flattened ``input`` parts for interactions.create.

    The Interactions API takes flattened media parts ({type, data, mime_type})
    rather than generateContent's inlineData/fileData nesting.
    """
    parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for img in image_bytes_list or []:
        parts.append(
            {
                "type": "image",
                "data": base64.b64encode(img).decode("ascii"),
                "mime_type": _sniff_image_mime(img),
            }
        )
    if input_video_bytes is not None:
        parts.append(
            {
                "type": "video",
                "data": base64.b64encode(input_video_bytes).decode("ascii"),
                "mime_type": "video/mp4",
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
) -> bytes:
    """Materialize mp4 bytes from an inline payload or a Files API uri."""
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
    file_obj = await asyncio.to_thread(client.files.get, name=uri)
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

    input_parts = _build_input_parts(prompt, image_bytes_list, input_video_bytes)
    task_type = _select_task_type(
        previous_interaction_id=previous_interaction_id,
        input_video_bytes=input_video_bytes,
        image_count=len(image_bytes_list or []),
    )

    # response_format is a LIST of one object carrying aspect ratio, duration,
    # and (optionally) GCS delivery. See the module docstring for the full
    # request shape and the note on `background` placement.
    response_format_item: dict[str, Any] = {
        "type": "video",
        "aspect_ratio": aspect_ratio,
        "duration": f"{clamped_duration}s",
    }
    if output_gcs_uri:
        response_format_item["delivery"] = "uri"
        response_format_item["gcs_uri"] = output_gcs_uri

    create_kwargs: dict[str, Any] = {
        "model": OMNI_MODEL,
        "input": input_parts,
        "background": True,
        "response_format": [response_format_item],
        "generation_config": {"video_config": {"task": task_type}},
    }
    if previous_interaction_id is not None:
        create_kwargs["previous_interaction_id"] = previous_interaction_id

    if log_callback:
        mode = "editing" if previous_interaction_id else "generating"
        await log_callback(
            f"Starting {mode} interaction with {OMNI_MODEL} (task={task_type})"
        )

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds

    async def _run_within_deadline(func: Any, /, **kwargs: Any) -> Any:
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise TimeoutError(
                f"Omni video interaction timed out after {timeout_seconds}s."
            )
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(func, **kwargs), timeout=remaining
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"Omni video interaction timed out after {timeout_seconds}s."
            ) from exc

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
    while status in _IN_FLIGHT_STATUSES:
        if loop.time() >= deadline:
            raise TimeoutError(
                f"Omni video interaction timed out after {timeout_seconds}s."
            )
        if log_callback:
            await log_callback(f"Interaction {interaction_id}: {status}")
        await asyncio.sleep(_POLL_INTERVAL)
        # ASSUMED SDK SHAPE: interactions.get retrieves an interaction by id.
        interaction = await _run_within_deadline(
            client.interactions.get, interaction_id=interaction_id
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
        video_bytes = await _resolve_video_bytes(client, inline_data, uri, log_callback)
        filename = f"{uuid.uuid4()}.mp4"
        filepath = videos_dir / filename
        filepath.write_bytes(video_bytes)
        video_url = f"file://{filepath}"

    result: dict[str, Any] = {
        "message": "Video generated successfully",
        "video_url": video_url,
        "interaction_id": interaction_id,
        "model": OMNI_MODEL,
        "duration_seconds": clamped_duration,
        "aspect_ratio": aspect_ratio,
    }

    # Include warnings only when non-empty, matching src/video.py.
    if warnings:
        result["warnings"] = warnings

    return result
