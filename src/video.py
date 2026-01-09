"""Video generation helpers."""

import asyncio
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
    "veo-2.0-generate-001",
    "veo-3.1-generate-preview",
    "veo-3.1-fast-generate-preview",
]

# Generation mode for VEO 3.1
GenerationMode = Literal[
    "text_to_video",           # Text-only generation
    "image_to_video",          # First frame image input
    "first_last_frame",        # First and last frame control
    "reference_to_video",      # Reference images for style/character
    "extend_video",            # Extend existing video
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


async def generate_video(
    client: genai.Client,
    prompt: str,
    videos_dir: Path,
    model: VideoModel = "veo-2.0-generate-001",
    image_bytes: bytes | None = None,
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
        duration_seconds: Video duration (VEO2: 5-8s, VEO3: 4/6/8s)
        include_audio: Enable audio generation (VEO3 only)
        audio_prompt: Audio description (VEO3 only)
        negative_prompt: Things to avoid in the video
        seed: Random seed for reproducibility
        log_callback: Async callback for progress logging
        last_frame_bytes: Last frame image bytes for first+last frame control (VEO3.1)
        reference_images: List of reference image bytes (up to 3) for style/character (VEO3.1)
        extend_video_uri: URI of existing VEO video to extend (VEO3.1). REQUIRES output_gcs_uri.
        output_gcs_uri: GCS URI for output (required for extensions and large videos)

    Returns:
        Dictionary with video_url and generation metadata
    """
    model_id = str(model)
    is_veo3 = model_id.startswith("veo-3")

    # Determine generation mode based on inputs
    generation_mode: str = "text_to_video"
    if extend_video_uri and is_veo3:
        generation_mode = "extend_video"
    elif reference_images and is_veo3:
        generation_mode = "reference_to_video"
    elif image_bytes and last_frame_bytes and is_veo3:
        generation_mode = "first_last_frame"
    elif image_bytes:
        generation_mode = "image_to_video"

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

    config_kwargs: dict[str, Any] = {
        "number_of_videos": 1,
        "aspect_ratio": aspect_ratio if aspect_ratio in ("16:9", "9:16") else "16:9",
    }

    if is_veo3:
        # Reference-to-video only supports 8 seconds
        if generation_mode == "reference_to_video":
            config_kwargs["duration_seconds"] = 8
        # Extend video requires exactly 7 seconds output
        elif generation_mode == "extend_video":
            config_kwargs["duration_seconds"] = 7
        else:
            allowed = [4, 6, 8]
            config_kwargs["duration_seconds"] = min(
                allowed, key=lambda x: abs(x - duration_seconds)
            )
        config_kwargs["enhance_prompt"] = True
        # generate_audio only supported in Vertex AI, not Gemini API
        if include_audio and getattr(client._api_client, 'vertexai', False):
            config_kwargs["generate_audio"] = include_audio

        # Add last frame to config for first+last frame mode
        if last_frame_input:
            config_kwargs["last_frame"] = last_frame_input

        # Add reference images to config for VEO 3.1
        if reference_image_inputs:
            config_kwargs["reference_images"] = reference_image_inputs
    else:
        config_kwargs["duration_seconds"] = max(5, min(8, int(duration_seconds)))

    if negative_prompt:
        config_kwargs["negative_prompt"] = negative_prompt
    if seed is not None and seed >= 0:
        config_kwargs["seed"] = seed
    if output_gcs_uri:
        config_kwargs["output_gcs_uri"] = output_gcs_uri

    prompt_for_api = prompt
    if is_veo3 and audio_prompt:
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
            local_path = extend_video_uri[7:]  # Remove file:// prefix
            api_kwargs["video"] = types.Video.from_file(
                location=local_path, mime_type="video/mp4"
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

    result = {
        "message": "Video generated successfully",
        "video_url": video_url,
        "prompt": prompt_for_api,
        "model": model_id,
        "audio_enabled": include_audio and is_veo3,
        "generation_mode": generation_mode,
    }

    # For extend_video mode, also return the extended video URI
    if generation_mode == "extend_video" and extend_video_uri:
        result["extended_from"] = extend_video_uri

    return result
