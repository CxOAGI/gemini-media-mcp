"""Video generation helpers."""

import asyncio
import os
import uuid
from collections.abc import Awaitable, Callable
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

from google import genai
from google.genai import types
from PIL import Image

# Type for async log callback from MCP context
LogCallback = Callable[[str], Awaitable[None]]

VideoModel = Literal[
    "veo-3.1-generate-001",
    "veo-3.1-fast-generate-001",
    "veo-3.1-lite-generate-preview",
]

# Veo 3.1 Lite does not support 4K output or video extension.
_VEO_LITE_MODELS = {"veo-3.1-lite-generate-preview"}

# The Gemini Developer API serves Veo 3.1 under `-preview` IDs, while Vertex
# AI uses the `-001` IDs (live-verified: a `-001` call on the Gemini API 404s
# with "not found for API version v1beta"). The public VideoModel values stay
# the `-001` names; they are translated per backend at call time.
_GEMINI_API_MODEL_IDS = {
    "veo-3.1-generate-001": "veo-3.1-generate-preview",
    "veo-3.1-fast-generate-001": "veo-3.1-fast-generate-preview",
}

# Generation mode for VEO 3.1
GenerationMode = Literal[
    "text_to_video",  # Text-only generation
    "image_to_video",  # First frame image input
    "first_last_frame",  # First and last frame control
    "reference_to_video",  # Reference images for style/character
    "extend_video",  # Extend existing video
]


def _prepare_image_input(image_bytes: bytes) -> types.Image:
    """Convert image bytes to types.Image for API input."""
    pil_img = Image.open(BytesIO(image_bytes))
    fmt = "PNG" if pil_img.mode in ("RGB", "RGBA") else "JPEG"
    if fmt == "JPEG" and pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    buf = BytesIO()
    pil_img.save(buf, format=fmt)
    pil_img.close()
    return types.Image(image_bytes=buf.getvalue(), mime_type=f"image/{fmt.lower()}")


def validate_render_options(model: str, resolution: str | None) -> None:
    """Raise for a model/resolution pairing Veo cannot render.

    Shared by the generation path and the tools' dry_run quotes, so a quote
    can never succeed for a call that would be refused — the same
    single-source rule as resolve_image_model on the image side.
    """
    if resolution is None:
        return
    valid_resolutions = ("720p", "1080p", "4K")
    if resolution not in valid_resolutions:
        raise ValueError(
            f"Unsupported resolution '{resolution}'. "
            f"Supported values are {', '.join(valid_resolutions)}."
        )
    if resolution == "4K" and model in _VEO_LITE_MODELS:
        raise ValueError(
            f"Model {model} does not support 4K resolution. "
            "Use veo-3.1-generate-001 or veo-3.1-fast-generate-001 instead."
        )


async def generate_video(
    client: genai.Client,
    prompt: str,
    videos_dir: Path,
    model: VideoModel = "veo-3.1-generate-001",
    image_bytes: bytes | None = None,
    allowed_dir: Path | None = None,
    aspect_ratio: str = "16:9",
    duration_seconds: float = 5.0,
    include_audio: bool = False,
    audio_prompt: str | None = None,
    negative_prompt: str | None = None,
    seed: int | None = None,
    log_callback: LogCallback | None = None,
    last_frame_bytes: bytes | None = None,
    reference_images: list[bytes] | None = None,
    extend_video_uri: str | None = None,
    resolution: str | None = None,
    person_generation: str | None = None,
    output_gcs_uri: str | None = None,
) -> dict[str, Any]:
    """Generate a video using VEO models.

    Args:
        client: Google GenAI client
        prompt: Text description of the video to generate
        videos_dir: Directory to save generated videos
        model: VEO model to use
        image_bytes: First frame image bytes for image-to-video
        aspect_ratio: Video aspect ratio (16:9 or 9:16)
        duration_seconds: Video duration (4/6/8s)
        include_audio: Enable audio generation
        audio_prompt: Audio description
        negative_prompt: Things to avoid in the video
        seed: Random seed for reproducibility
        log_callback: Async callback for progress logging
        last_frame_bytes: Last frame image bytes for first+last frame control
        reference_images: List of reference image bytes (up to 3) for style/character
        allowed_dir: Directory that file:// / bare-path extend sources must
            resolve inside. Security boundary, not a convenience: without it
            any local path readable by the server could be uploaded to the
            API as "video to extend".
        extend_video_uri: URI of existing VEO video to extend. On Vertex AI
            this requires output_gcs_uri; on the Gemini API the extended
            clip is returned inline and GCS output is not supported.
        resolution: Output resolution ("720p" or "1080p"; "4K" only for non-Lite
            models). When None, the API default is used.
        person_generation: Person generation policy ("allow_adult" or "allow_all").
            Passed through to the API for validation. When None, the API default
            is used.
        output_gcs_uri: GCS URI for output (required for extensions and large
            videos). Only supported in Vertex AI mode.

    Returns:
        Dictionary with video_url and generation metadata
    """
    model_id = str(model)

    # Translate model IDs per backend: the Gemini Developer API serves Veo
    # under `-preview` IDs and 404s on the Vertex `-001` names.
    is_vertexai = getattr(client._api_client, "vertexai", False)
    if not is_vertexai:
        model_id = _GEMINI_API_MODEL_IDS.get(model_id, model_id)

    # Non-fatal warnings surfaced back to the caller (e.g. a request that could
    # not be honored but should not abort the whole generation).
    warnings: list[str] = []

    # A last frame without a first frame would silently classify as
    # text_to_video and the fetched frame would be discarded — reject it.
    if last_frame_bytes and not image_bytes:
        raise ValueError(
            "A last frame was provided without a first frame. First+last "
            "frame mode requires both; provide image_uri/image_base64 too."
        )

    # Determine generation mode based on inputs
    generation_mode: str = "text_to_video"
    if extend_video_uri:
        generation_mode = "extend_video"
    elif reference_images:
        generation_mode = "reference_to_video"
    elif image_bytes and last_frame_bytes:
        generation_mode = "first_last_frame"
    elif image_bytes:
        generation_mode = "image_to_video"

    # Veo 3.1 Lite (served via the Gemini API) does not support video
    # extension, reference images, or first/last-frame control — fail fast
    # with a clear message instead of an opaque API error.
    if model in _VEO_LITE_MODELS and generation_mode in (
        "extend_video",
        "reference_to_video",
        "first_last_frame",
    ):
        raise ValueError(
            f"Model {model_id} does not support {generation_mode}. "
            "Veo 3.1 Lite supports only text-to-video and image-to-video; "
            "use veo-3.1-generate-001 or veo-3.1-fast-generate-001 instead."
        )

    # Prepare image inputs
    first_frame_input: types.Image | None = None
    last_frame_input: types.Image | None = None
    reference_image_inputs: list[types.VideoGenerationReferenceImage] = []

    if generation_mode == "image_to_video" and image_bytes:
        first_frame_input = _prepare_image_input(image_bytes)
    elif generation_mode == "first_last_frame":
        if image_bytes:
            first_frame_input = _prepare_image_input(image_bytes)
        if last_frame_bytes:
            last_frame_input = _prepare_image_input(last_frame_bytes)
    elif generation_mode == "reference_to_video" and reference_images:
        # VEO 3.1 supports up to 3 reference images (asset type)
        # Must wrap in VideoGenerationReferenceImage with reference_type="asset"
        for ref_bytes in reference_images[:3]:
            ref_image = _prepare_image_input(ref_bytes)
            reference_image_inputs.append(
                types.VideoGenerationReferenceImage(
                    image=ref_image,
                    reference_type="asset",  # asset for subject preservation
                )
            )

    # Aspect ratio must match source clips for transitions/bridges, so an
    # unsupported value is a hard error rather than a silent coercion.
    if aspect_ratio not in ("16:9", "9:16"):
        raise ValueError(
            f"Unsupported aspect_ratio '{aspect_ratio}'. "
            "Supported values are '16:9' and '9:16'."
        )

    config_kwargs: dict[str, Any] = {
        "number_of_videos": 1,
        "aspect_ratio": aspect_ratio,
    }

    # Compute the effective (snapped/forced) duration so it can be both sent to
    # the API and reported back to callers.
    if generation_mode == "reference_to_video":
        # Reference-to-video only supports 8 seconds.
        effective_duration = 8
    elif generation_mode == "extend_video":
        # Extend video requires exactly 7 seconds output.
        effective_duration = 7
    else:
        allowed = [4, 6, 8]
        effective_duration = min(allowed, key=lambda x: abs(x - duration_seconds))
    config_kwargs["duration_seconds"] = effective_duration
    # generate_audio is only supported on Vertex AI. Veo 3.1 already applies prompt
    # rewriting automatically, so `enhance_prompt` is Veo-2-only and must not be sent.
    # (is_vertexai computed at the top alongside model-ID translation.)
    if is_vertexai:
        # Send the flag explicitly BOTH ways: omitting it lets the API apply
        # its own default (audio on for Veo 3.1), which would silently
        # contradict include_audio=False and the reported audio_enabled.
        config_kwargs["generate_audio"] = include_audio

    # On the Gemini API path, generate_audio is never sent and Veo 3.1 always
    # generates audio natively. A caller who explicitly asked for NO audio
    # (include_audio=False, the default) cannot have that honored here, so warn
    # rather than silently returning a clip with baked-in audio. (include_audio=True
    # is satisfied since audio is produced, so no warning is needed there.)
    if not is_vertexai and not include_audio:
        warnings.append(
            "include_audio=False was not honored: Veo 3.1 on the Gemini API "
            "always generates audio. Use Vertex AI mode to control audio."
        )

    # Add last frame to config for first+last frame mode
    if last_frame_input:
        config_kwargs["last_frame"] = last_frame_input

    # Add reference images to config for VEO 3.1
    if reference_image_inputs:
        config_kwargs["reference_images"] = reference_image_inputs

    if negative_prompt:
        config_kwargs["negative_prompt"] = negative_prompt
    if seed is not None and seed >= 0:
        config_kwargs["seed"] = seed

    # output_gcs_uri is a Vertex-only config field. The Gemini API (e.g. for
    # Veo Lite) does not support GCS output, so only forward it on Vertex.
    # Callers may pass a default bucket (VIDEO_GCS_BUCKET) that applies to
    # Vertex runs; silently ignoring it on the Gemini API path keeps Lite and
    # plain text-to-video working. The generate_video tool separately rejects
    # an *explicit* output_gcs_uri on a Gemini-API-routed call.
    if output_gcs_uri and is_vertexai:
        config_kwargs["output_gcs_uri"] = output_gcs_uri

    if resolution is not None:
        validate_render_options(model, resolution)
        config_kwargs["resolution"] = resolution

    if person_generation is not None:
        # Pass through as-is and let the API validate the value.
        config_kwargs["person_generation"] = person_generation

    prompt_for_api = prompt
    if audio_prompt:
        prompt_for_api = f"{prompt}\nAudio: {audio_prompt}"

    video_config = types.GenerateVideosConfig(**config_kwargs)

    if log_callback:
        mode_desc = generation_mode.replace("_", " ")
        await log_callback(f"Starting {mode_desc} with {model_id}")

    # Build API call based on generation mode
    api_kwargs: dict[str, Any] = {
        "model": model_id,
        "prompt": prompt_for_api,
        "config": video_config,
    }

    if generation_mode == "image_to_video" and first_frame_input:
        api_kwargs["image"] = first_frame_input
    elif generation_mode == "first_last_frame" and first_frame_input:
        # First frame as image param, last frame in config (already added above)
        api_kwargs["image"] = first_frame_input
    elif generation_mode == "extend_video" and extend_video_uri:
        # Video extension for VEO 3.1
        # For file:// URIs, load from local file to get proper mime type
        if extend_video_uri.startswith("file://"):
            local_path = Path(extend_video_uri[7:])
            # Validate path is within allowed directory (prevents LFI)
            if allowed_dir is not None:
                resolved = local_path.resolve()
                allowed = allowed_dir.resolve()
                if (
                    not str(resolved).startswith(str(allowed) + os.sep)
                    and resolved != allowed
                ):
                    raise ValueError(
                        f"Access denied: '{local_path}' is outside the allowed directory."
                    )
            api_kwargs["video"] = types.Video.from_file(
                location=str(local_path), mime_type="video/mp4"
            )
        else:
            # Remote URI - pass with mime type
            api_kwargs["video"] = types.Video(
                uri=extend_video_uri, mimeType="video/mp4"
            )

    operation = await asyncio.to_thread(
        client.models.generate_videos,
        **api_kwargs,
    )

    if log_callback:
        await log_callback(f"Polling operation: {operation.name}")
    timeout = 1800
    elapsed = 0
    while not operation.done:
        if elapsed >= timeout:
            raise TimeoutError("Video generation timed out")
        await asyncio.sleep(10)
        elapsed += 10
        operation = await asyncio.to_thread(client.operations.get, operation)

    if operation.error:
        raise ValueError(f"VEO error: {operation.error}")

    result = getattr(operation, "response", None) or getattr(operation, "result", None)
    if not result or not getattr(result, "generated_videos", None):
        raise ValueError("No videos returned")

    video = result.generated_videos[0].video

    if hasattr(video, "uri") and video.uri and video.uri.startswith("gs://"):
        video_url = video.uri
    elif hasattr(video, "video_bytes") and video.video_bytes:
        filename = f"{uuid.uuid4()}.mp4"
        filepath = videos_dir / filename
        filepath.write_bytes(video.video_bytes)
        video_url = f"file://{filepath}"
    else:
        await asyncio.to_thread(client.files.download, file=video)
        filename = f"{uuid.uuid4()}.mp4"
        filepath = videos_dir / filename
        await asyncio.to_thread(video.save, str(filepath))
        video_url = f"file://{filepath}"

    # Report audio truthfully: on Vertex AI it depends on the generate_audio flag
    # (== include_audio), but on the Gemini API path Veo 3.1 always generates
    # audio natively regardless of the include_audio input.
    audio_enabled = include_audio if is_vertexai else True

    result = {
        "message": "Video generated successfully",
        "video_url": video_url,
        "prompt": prompt_for_api,
        "model": model_id,
        "audio_enabled": audio_enabled,
        "duration_seconds": effective_duration,
        "generation_mode": generation_mode,
    }

    # For extend_video mode, also return the extended video URI
    if generation_mode == "extend_video" and extend_video_uri:
        result["extended_from"] = extend_video_uri

    # Include warnings only when non-empty, so clean runs keep tidy manifests.
    if warnings:
        result["warnings"] = warnings

    return result
