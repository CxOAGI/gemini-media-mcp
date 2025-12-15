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
) -> dict[str, Any]:
    """Generate a video using VEO models."""
    model_id = str(model)

    image_input: types.Image | None = None
    if image_bytes:
        pil_img = Image.open(BytesIO(image_bytes))
        fmt = "PNG" if pil_img.mode in ("RGB", "RGBA") else "JPEG"
        if fmt == "JPEG" and pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        buf = BytesIO()
        pil_img.save(buf, format=fmt)
        image_input = types.Image(
            image_bytes=buf.getvalue(), mime_type=f"image/{fmt.lower()}"
        )
        pil_img.close()

    config_kwargs: dict[str, Any] = {
        "number_of_videos": 1,
        "aspect_ratio": aspect_ratio if aspect_ratio in ("16:9", "9:16") else "16:9",
    }

    if model_id.startswith("veo-3"):
        allowed = [4, 6, 8]
        config_kwargs["duration_seconds"] = min(
            allowed, key=lambda x: abs(x - duration_seconds)
        )
        config_kwargs["enhance_prompt"] = True
        config_kwargs["generate_audio"] = include_audio
    else:
        config_kwargs["duration_seconds"] = max(5, min(8, int(duration_seconds)))

    if negative_prompt:
        config_kwargs["negative_prompt"] = negative_prompt
    if seed is not None and seed >= 0:
        config_kwargs["seed"] = seed

    prompt_for_api = prompt
    if model_id.startswith("veo-3") and audio_prompt:
        prompt_for_api = f"{prompt}\nAudio: {audio_prompt}"

    video_config = types.GenerateVideosConfig(**config_kwargs)

    if log_callback:
        await log_callback(f"Starting video generation with {model_id}")
    if image_input:
        operation = await asyncio.to_thread(
            client.models.generate_videos,
            model=model_id,
            prompt=prompt_for_api,
            image=image_input,
            config=video_config,
        )
    else:
        operation = await asyncio.to_thread(
            client.models.generate_videos,
            model=model_id,
            prompt=prompt_for_api,
            config=video_config,
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

    return {
        "message": "Video generated successfully",
        "video_url": video_url,
        "prompt": prompt_for_api,
        "model": model_id,
        "audio_enabled": include_audio and model_id.startswith("veo-3"),
    }
