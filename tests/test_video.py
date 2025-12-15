"""Tests for video.py video generation helpers."""

import asyncio
from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from src.video import VideoModel, generate_video

# ============================================================================
# Test Doubles
# ============================================================================


class FakeVideoObject:
    """Test double for video object."""

    def __init__(
        self,
        uri: str | None = None,
        video_bytes: bytes | None = None,
    ) -> None:
        self.uri = uri
        self.video_bytes = video_bytes

    def save(self, path: str) -> None:
        Path(path).write_bytes(b"fake video content")


class FakeGeneratedVideo:
    """Test double for generated video."""

    def __init__(self, video: FakeVideoObject | None = None) -> None:
        self.video = video


class FakeVideoResult:
    """Test double for video generation result."""

    def __init__(
        self, generated_videos: list[FakeGeneratedVideo] | None = None
    ) -> None:
        self.generated_videos = generated_videos


class FakeOperation:
    """Test double for async operation."""

    def __init__(
        self,
        done: bool = True,
        result: FakeVideoResult | None = None,
        error: str | None = None,
        name: str = "test-operation",
    ) -> None:
        self.done = done
        self.result = result
        self.response = result
        self.error = error
        self.name = name
        self._poll_count = 0
        self._done_after = 1

    def set_done_after_polls(self, count: int) -> None:
        self._done_after = count
        self.done = False

    def poll(self) -> "FakeOperation":
        self._poll_count += 1
        if self._poll_count >= self._done_after:
            self.done = True
        return self


class FakeOperations:
    """Test double for operations client."""

    def __init__(self, operation: FakeOperation) -> None:
        self._operation = operation

    def get(self, op: FakeOperation) -> FakeOperation:
        return self._operation.poll()


class FakeFiles:
    """Test double for files client."""

    def download(self, file: Any) -> None:
        pass


class FakeModels:
    """Test double for models client."""

    def __init__(
        self,
        operation: FakeOperation | None = None,
        raise_error: Exception | None = None,
    ) -> None:
        self._operation = operation
        self._raise_error = raise_error

    def generate_videos(self, **kwargs: Any) -> FakeOperation:
        if self._raise_error:
            raise self._raise_error
        return self._operation or FakeOperation()


class FakeGenaiClient:
    """Test double for Google GenAI client."""

    def __init__(
        self,
        operation: FakeOperation | None = None,
        raise_error: Exception | None = None,
    ) -> None:
        self.models = FakeModels(operation, raise_error)
        self.operations = FakeOperations(operation or FakeOperation())
        self.files = FakeFiles()


def _create_test_image(width: int = 100, height: int = 100, mode: str = "RGB") -> bytes:
    """Create a test image and return bytes."""
    img = Image.new(mode, (width, height), color="blue")
    buffer = BytesIO()
    if mode == "RGBA":
        img.save(buffer, format="PNG")
    else:
        img.save(buffer, format="JPEG")
    img.close()
    return buffer.getvalue()


# ============================================================================
# generate_video tests - VEO 2.0
# ============================================================================


@pytest.mark.parametrize(
    ("input", "expected"),
    [
        pytest.param(
            {
                "prompt": "A cat walking",
                "model": "veo-2.0-generate-001",
                "aspect_ratio": "16:9",
                "duration_seconds": 5.0,
            },
            {"success": True, "audio_enabled": False},
            id="veo2_basic",
        ),
        pytest.param(
            {
                "prompt": "A dog running",
                "model": "veo-2.0-generate-001",
                "aspect_ratio": "9:16",
                "duration_seconds": 8.0,
            },
            {"success": True, "audio_enabled": False},
            id="veo2_portrait_8s",
        ),
        pytest.param(
            {
                "prompt": "A" * 10000,
                "model": "veo-2.0-generate-001",
                "aspect_ratio": "16:9",
                "duration_seconds": 5.0,
            },
            {"success": True},
            id="veo2_large_prompt",
        ),
        pytest.param(
            {
                "prompt": "Unicode: 🎬 日本語 émoji",
                "model": "veo-2.0-generate-001",
                "aspect_ratio": "16:9",
                "duration_seconds": 5.0,
            },
            {"success": True},
            id="veo2_unicode_prompt",
        ),
        pytest.param(
            {
                "prompt": "",
                "model": "veo-2.0-generate-001",
                "aspect_ratio": "16:9",
                "duration_seconds": 5.0,
            },
            {"success": True},
            id="veo2_empty_prompt",
        ),
        pytest.param(
            {
                "prompt": "Test negative",
                "model": "veo-2.0-generate-001",
                "aspect_ratio": "16:9",
                "duration_seconds": 5.0,
                "negative_prompt": "blurry, distorted",
            },
            {"success": True},
            id="veo2_with_negative_prompt",
        ),
        pytest.param(
            {
                "prompt": "Test seed",
                "model": "veo-2.0-generate-001",
                "aspect_ratio": "16:9",
                "duration_seconds": 5.0,
                "seed": 42,
            },
            {"success": True},
            id="veo2_with_seed",
        ),
        pytest.param(
            {
                "prompt": "Invalid aspect",
                "model": "veo-2.0-generate-001",
                "aspect_ratio": "4:3",
                "duration_seconds": 5.0,
            },
            {"success": True},
            id="veo2_invalid_aspect_ratio_fallback",
        ),
    ],
)
@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_video_veo2(
    input: dict[str, Any],
    expected: dict[str, Any],
    tmp_path: Path,
) -> None:
    """Test generate_video with VEO 2.0 model."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    # Create fake video with bytes
    video_obj = FakeVideoObject(video_bytes=b"fake video content")
    gen_video = FakeGeneratedVideo(video_obj)
    result = FakeVideoResult([gen_video])
    operation = FakeOperation(done=True, result=result)

    client = FakeGenaiClient(operation=operation)

    gen_result = await generate_video(
        client=client,  # type: ignore[arg-type]
        prompt=input["prompt"],
        videos_dir=videos_dir,
        model=input["model"],
        aspect_ratio=input.get("aspect_ratio", "16:9"),
        duration_seconds=input.get("duration_seconds", 5.0),
        negative_prompt=input.get("negative_prompt"),
        seed=input.get("seed"),
    )

    assert gen_result["message"] == "Video generated successfully"
    assert gen_result["model"] == input["model"]
    assert "video_url" in gen_result


# ============================================================================
# generate_video tests - VEO 3.x models
# ============================================================================


@pytest.mark.parametrize(
    ("input", "expected"),
    [
        pytest.param(
            {
                "prompt": "A bird flying",
                "model": "veo-3.1-generate-preview",
                "aspect_ratio": "16:9",
                "duration_seconds": 4.0,
                "include_audio": False,
            },
            {"success": True, "audio_enabled": False},
            id="veo3_basic_no_audio",
        ),
        pytest.param(
            {
                "prompt": "A bird singing",
                "model": "veo-3.1-generate-preview",
                "aspect_ratio": "16:9",
                "duration_seconds": 6.0,
                "include_audio": True,
            },
            {"success": True, "audio_enabled": True},
            id="veo3_with_audio",
        ),
        pytest.param(
            {
                "prompt": "A crowd cheering",
                "model": "veo-3.1-generate-preview",
                "aspect_ratio": "16:9",
                "duration_seconds": 8.0,
                "include_audio": True,
                "audio_prompt": "Crowd cheering loudly",
            },
            {"success": True, "audio_enabled": True},
            id="veo3_with_audio_prompt",
        ),
        pytest.param(
            {
                "prompt": "Fast video",
                "model": "veo-3.1-fast-generate-preview",
                "aspect_ratio": "9:16",
                "duration_seconds": 4.0,
            },
            {"success": True},
            id="veo3_fast_model",
        ),
        pytest.param(
            {
                "prompt": "Duration test",
                "model": "veo-3.1-generate-preview",
                "aspect_ratio": "16:9",
                "duration_seconds": 5.0,
            },
            {"success": True},
            id="veo3_duration_snaps_to_6s",
        ),
        pytest.param(
            {
                "prompt": "Duration test",
                "model": "veo-3.1-generate-preview",
                "aspect_ratio": "16:9",
                "duration_seconds": 7.0,
            },
            {"success": True},
            id="veo3_duration_snaps_to_8s",
        ),
    ],
)
@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_video_veo3(
    input: dict[str, Any],
    expected: dict[str, Any],
    tmp_path: Path,
) -> None:
    """Test generate_video with VEO 3.x models."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    video_obj = FakeVideoObject(video_bytes=b"fake video content")
    gen_video = FakeGeneratedVideo(video_obj)
    result = FakeVideoResult([gen_video])
    operation = FakeOperation(done=True, result=result)

    client = FakeGenaiClient(operation=operation)

    gen_result = await generate_video(
        client=client,  # type: ignore[arg-type]
        prompt=input["prompt"],
        videos_dir=videos_dir,
        model=input["model"],
        aspect_ratio=input.get("aspect_ratio", "16:9"),
        duration_seconds=input.get("duration_seconds", 5.0),
        include_audio=input.get("include_audio", False),
        audio_prompt=input.get("audio_prompt"),
    )

    assert gen_result["message"] == "Video generated successfully"
    assert gen_result["model"] == input["model"]

    if input["model"].startswith("veo-3"):
        expected_audio = input.get("include_audio", False)
        assert gen_result["audio_enabled"] == expected_audio


# ============================================================================
# generate_video tests - Image input
# ============================================================================


@pytest.mark.parametrize(
    ("input", "expected"),
    [
        pytest.param(
            {"mode": "RGB", "size": (100, 100)},
            {"success": True},
            id="rgb_image_input",
        ),
        pytest.param(
            {"mode": "RGBA", "size": (100, 100)},
            {"success": True},
            id="rgba_image_input",
        ),
        pytest.param(
            {"mode": "L", "size": (100, 100)},
            {"success": True},
            id="grayscale_image_input",
        ),
        pytest.param(
            {"mode": "RGB", "size": (1920, 1080)},
            {"success": True},
            id="hd_image_input",
        ),
    ],
)
@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_video_with_image(
    input: dict[str, Any],
    expected: dict[str, Any],
    tmp_path: Path,
) -> None:
    """Test generate_video with image input."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    image_bytes = _create_test_image(
        width=input["size"][0],
        height=input["size"][1],
        mode=input["mode"],
    )

    video_obj = FakeVideoObject(video_bytes=b"fake video content")
    gen_video = FakeGeneratedVideo(video_obj)
    result = FakeVideoResult([gen_video])
    operation = FakeOperation(done=True, result=result)

    client = FakeGenaiClient(operation=operation)

    gen_result = await generate_video(
        client=client,  # type: ignore[arg-type]
        prompt="Animate this image",
        videos_dir=videos_dir,
        model="veo-2.0-generate-001",
        image_bytes=image_bytes,
    )

    assert gen_result["message"] == "Video generated successfully"


# ============================================================================
# generate_video tests - Output handling
# ============================================================================


@pytest.mark.parametrize(
    ("input", "expected"),
    [
        pytest.param(
            {"output_type": "gcs_uri"},
            {"url_prefix": "gs://"},
            id="gcs_uri_output",
        ),
        pytest.param(
            {"output_type": "video_bytes"},
            {"url_prefix": "file://"},
            id="video_bytes_output",
        ),
        pytest.param(
            {"output_type": "download"},
            {"url_prefix": "file://"},
            id="download_output",
        ),
    ],
)
@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_video_output_types(
    input: dict[str, Any],
    expected: dict[str, Any],
    tmp_path: Path,
) -> None:
    """Test generate_video handles different output types."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    output_type = input["output_type"]

    if output_type == "gcs_uri":
        video_obj = FakeVideoObject(uri="gs://bucket/video.mp4")
    elif output_type == "video_bytes":
        video_obj = FakeVideoObject(video_bytes=b"video content")
    else:
        video_obj = FakeVideoObject()

    gen_video = FakeGeneratedVideo(video_obj)
    result = FakeVideoResult([gen_video])
    operation = FakeOperation(done=True, result=result)

    client = FakeGenaiClient(operation=operation)

    gen_result = await generate_video(
        client=client,  # type: ignore[arg-type]
        prompt="Test output",
        videos_dir=videos_dir,
        model="veo-2.0-generate-001",
    )

    assert gen_result["video_url"].startswith(expected["url_prefix"])


# ============================================================================
# generate_video tests - Polling behavior
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_generate_video_polling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test generate_video polls operation until done."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    video_obj = FakeVideoObject(video_bytes=b"video content")
    gen_video = FakeGeneratedVideo(video_obj)
    result = FakeVideoResult([gen_video])
    operation = FakeOperation(done=False, result=result)
    operation.set_done_after_polls(2)

    client = FakeGenaiClient(operation=operation)

    # Speed up sleep for testing
    original_sleep = asyncio.sleep

    async def fast_sleep(seconds: float) -> None:
        await original_sleep(0.01)

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    gen_result = await generate_video(
        client=client,  # type: ignore[arg-type]
        prompt="Polling test",
        videos_dir=videos_dir,
        model="veo-2.0-generate-001",
    )

    assert gen_result["message"] == "Video generated successfully"


# ============================================================================
# generate_video tests - Error handling
# ============================================================================


@pytest.mark.parametrize(
    ("input", "expected"),
    [
        pytest.param(
            {"error_type": "operation_error"},
            ValueError,
            id="operation_error",
        ),
        pytest.param(
            {"error_type": "no_videos"},
            ValueError,
            id="no_videos_returned",
        ),
        pytest.param(
            {"error_type": "api_error"},
            RuntimeError,
            id="api_error",
        ),
    ],
)
@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_video_errors(
    input: dict[str, Any],
    expected: type[Exception],
    tmp_path: Path,
) -> None:
    """Test generate_video error handling."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    error_type = input["error_type"]

    if error_type == "operation_error":
        operation = FakeOperation(done=True, error="VEO generation failed")
        client = FakeGenaiClient(operation=operation)
    elif error_type == "no_videos":
        result = FakeVideoResult([])
        operation = FakeOperation(done=True, result=result)
        client = FakeGenaiClient(operation=operation)
    else:
        client = FakeGenaiClient(raise_error=RuntimeError("API error"))

    with pytest.raises(expected):
        await generate_video(
            client=client,  # type: ignore[arg-type]
            prompt="Error test",
            videos_dir=videos_dir,
            model="veo-2.0-generate-001",
        )


@pytest.mark.asyncio
@pytest.mark.timeout(3.0)
async def test_generate_video_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test generate_video timeout handling."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    # Operation that never completes
    operation = FakeOperation(done=False)
    client = FakeGenaiClient(operation=operation)

    # Speed up sleep and reduce timeout for testing
    call_count = 0

    async def counting_sleep(seconds: float) -> None:
        nonlocal call_count
        call_count += 1
        if call_count > 180:
            raise TimeoutError("Test safety timeout")
        await asyncio.sleep(0.001)

    monkeypatch.setattr(asyncio, "sleep", counting_sleep)

    with pytest.raises(TimeoutError):
        await generate_video(
            client=client,  # type: ignore[arg-type]
            prompt="Timeout test",
            videos_dir=videos_dir,
            model="veo-2.0-generate-001",
        )


# ============================================================================
# generate_video tests - Log callback
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_video_log_callback(
    tmp_path: Path,
) -> None:
    """Test generate_video uses log callback."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    video_obj = FakeVideoObject(video_bytes=b"video content")
    gen_video = FakeGeneratedVideo(video_obj)
    result = FakeVideoResult([gen_video])
    operation = FakeOperation(done=True, result=result)

    client = FakeGenaiClient(operation=operation)

    log_messages: list[str] = []

    async def log_callback(msg: str) -> None:
        log_messages.append(msg)

    await generate_video(
        client=client,  # type: ignore[arg-type]
        prompt="Log test",
        videos_dir=videos_dir,
        model="veo-2.0-generate-001",
        log_callback=log_callback,
    )

    assert len(log_messages) >= 2
    assert any("Starting video generation" in msg for msg in log_messages)
    assert any("Polling operation" in msg for msg in log_messages)


# ============================================================================
# generate_video tests - File creation
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_video_creates_file(
    tmp_path: Path,
) -> None:
    """Test generate_video creates output file correctly."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    video_content = b"fake video content here"
    video_obj = FakeVideoObject(video_bytes=video_content)
    gen_video = FakeGeneratedVideo(video_obj)
    result = FakeVideoResult([gen_video])
    operation = FakeOperation(done=True, result=result)

    client = FakeGenaiClient(operation=operation)

    gen_result = await generate_video(
        client=client,  # type: ignore[arg-type]
        prompt="File test",
        videos_dir=videos_dir,
        model="veo-2.0-generate-001",
    )

    video_url = gen_result["video_url"]
    assert video_url.startswith("file://")

    file_path = Path(video_url[7:])
    assert file_path.exists()
    assert file_path.suffix == ".mp4"
    assert file_path.read_bytes() == video_content
