"""Tests for video.py video generation helpers."""

import asyncio
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from google.genai import errors as genai_errors
from PIL import Image

from src.video import VEO_GEMINI_API_REFUSAL, generate_video

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


class FakeApiClient:
    """Test double for internal API client.

    Vertex by default: Veo runs on Vertex AI only, and the impl refuses any
    other client before it sends anything.
    """

    def __init__(self, vertexai: bool = True) -> None:
        self.vertexai = vertexai


class FakeGenaiClient:
    """Test double for Google GenAI client."""

    def __init__(
        self,
        operation: FakeOperation | None = None,
        raise_error: Exception | None = None,
        vertexai: bool = True,
    ) -> None:
        self.models = FakeModels(operation, raise_error)
        self.operations = FakeOperations(operation or FakeOperation())
        self.files = FakeFiles()
        self._api_client = FakeApiClient(vertexai=vertexai)


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
# generate_video tests - Basic parameters
# ============================================================================


@pytest.mark.parametrize(
    ("input", "expected"),
    [
        pytest.param(
            {
                "prompt": "A cat walking",
                "model": "veo-3.1-generate-001",
                "aspect_ratio": "16:9",
                "duration_seconds": 5.0,
            },
            {"success": True, "audio_enabled": False},
            id="veo2_basic",
        ),
        pytest.param(
            {
                "prompt": "A dog running",
                "model": "veo-3.1-generate-001",
                "aspect_ratio": "9:16",
                "duration_seconds": 8.0,
            },
            {"success": True, "audio_enabled": False},
            id="veo2_portrait_8s",
        ),
        pytest.param(
            {
                "prompt": "A" * 10000,
                "model": "veo-3.1-generate-001",
                "aspect_ratio": "16:9",
                "duration_seconds": 5.0,
            },
            {"success": True},
            id="veo2_large_prompt",
        ),
        pytest.param(
            {
                "prompt": "Unicode: 🎬 日本語 émoji",
                "model": "veo-3.1-generate-001",
                "aspect_ratio": "16:9",
                "duration_seconds": 5.0,
            },
            {"success": True},
            id="veo2_unicode_prompt",
        ),
        pytest.param(
            {
                "prompt": "Test negative",
                "model": "veo-3.1-generate-001",
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
                "model": "veo-3.1-generate-001",
                "aspect_ratio": "16:9",
                "duration_seconds": 5.0,
                "seed": 42,
            },
            {"success": True},
            id="veo2_with_seed",
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
    """Test generate_video with basic parameters."""
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


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_video_refuses_an_empty_prompt(tmp_path: Path) -> None:
    """An empty prompt is refused before any render — the impl used to send it
    to the API, which either errors or bills a garbage full-price render."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    operation = FakeOperation(
        done=True,
        result=FakeVideoResult([FakeGeneratedVideo(FakeVideoObject(b"v"))]),
    )
    client = FakeGenaiClient(operation=operation)

    with pytest.raises(ValueError, match="non-empty"):
        await generate_video(
            client=client,  # type: ignore[arg-type]
            prompt="   ",
            videos_dir=videos_dir,
            model="veo-3.1-generate-001",
        )


# ============================================================================
# generate_video tests - VEO 3.x models
# ============================================================================


@pytest.mark.parametrize(
    ("input", "expected"),
    [
        pytest.param(
            {
                "prompt": "A bird flying",
                "model": "veo-3.1-generate-001",
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
                "model": "veo-3.1-generate-001",
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
                "model": "veo-3.1-generate-001",
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
                "model": "veo-3.1-fast-generate-001",
                "aspect_ratio": "9:16",
                "duration_seconds": 4.0,
            },
            {"success": True},
            id="veo3_fast_model",
        ),
        pytest.param(
            {
                "prompt": "Duration test",
                "model": "veo-3.1-generate-001",
                "aspect_ratio": "16:9",
                "duration_seconds": 5.0,
            },
            {"success": True},
            id="veo3_duration_snaps_to_6s",
        ),
        pytest.param(
            {
                "prompt": "Duration test",
                "model": "veo-3.1-generate-001",
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
    # generate_audio is sent as include_audio, so that is what the clip carries.
    assert gen_result["audio_enabled"] == input.get("include_audio", False)


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
        model="veo-3.1-generate-001",
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
        model="veo-3.1-generate-001",
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
        model="veo-3.1-generate-001",
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
            model="veo-3.1-generate-001",
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
            model="veo-3.1-generate-001",
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
        model="veo-3.1-generate-001",
        log_callback=log_callback,
    )

    assert len(log_messages) >= 2
    assert any("Starting" in msg and "video" in msg for msg in log_messages)
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
        model="veo-3.1-generate-001",
    )

    video_url = gen_result["video_url"]
    assert video_url.startswith("file://")

    file_path = Path(video_url[7:])
    assert file_path.exists()
    assert file_path.suffix == ".mp4"
    assert file_path.read_bytes() == video_content


# ============================================================================
# generate_video tests - VEO 3.1 First and Last Frame Control
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_video_first_last_frame(
    tmp_path: Path,
) -> None:
    """Test generate_video with first and last frame control for VEO 3.1."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    first_frame = _create_test_image(width=100, height=100, mode="RGB")
    last_frame = _create_test_image(width=100, height=100, mode="RGB")

    video_obj = FakeVideoObject(video_bytes=b"fake video content")
    gen_video = FakeGeneratedVideo(video_obj)
    result = FakeVideoResult([gen_video])
    operation = FakeOperation(done=True, result=result)

    client = FakeGenaiClient(vertexai=True, operation=operation)

    gen_result = await generate_video(
        client=client,  # type: ignore[arg-type]
        prompt="Transition from first to last frame",
        videos_dir=videos_dir,
        model="veo-3.1-generate-001",
        image_bytes=first_frame,
        last_frame_bytes=last_frame,
    )

    assert gen_result["message"] == "Video generated successfully"
    assert gen_result["generation_mode"] == "first_last_frame"


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_video_first_frame_only_is_image_to_video(
    tmp_path: Path,
) -> None:
    """Test that first frame only falls back to image_to_video mode."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    first_frame = _create_test_image(width=100, height=100, mode="RGB")

    video_obj = FakeVideoObject(video_bytes=b"fake video content")
    gen_video = FakeGeneratedVideo(video_obj)
    result = FakeVideoResult([gen_video])
    operation = FakeOperation(done=True, result=result)

    client = FakeGenaiClient(operation=operation)

    gen_result = await generate_video(
        client=client,  # type: ignore[arg-type]
        prompt="Animate this image",
        videos_dir=videos_dir,
        model="veo-3.1-generate-001",
        image_bytes=first_frame,
    )

    assert gen_result["message"] == "Video generated successfully"
    assert gen_result["generation_mode"] == "image_to_video"


# ============================================================================
# generate_video tests - VEO 3.1 Reference Images
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_video_reference_images(
    tmp_path: Path,
) -> None:
    """Test generate_video with reference images for VEO 3.1."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    reference_images = [
        _create_test_image(width=100, height=100, mode="RGB"),
        _create_test_image(width=100, height=100, mode="RGB"),
        _create_test_image(width=100, height=100, mode="RGB"),
    ]

    video_obj = FakeVideoObject(video_bytes=b"fake video content")
    gen_video = FakeGeneratedVideo(video_obj)
    result = FakeVideoResult([gen_video])
    operation = FakeOperation(done=True, result=result)

    client = FakeGenaiClient(vertexai=True, operation=operation)

    gen_result = await generate_video(
        client=client,  # type: ignore[arg-type]
        prompt="Video featuring the character from references",
        videos_dir=videos_dir,
        model="veo-3.1-generate-001",
        reference_images=reference_images,
    )

    assert gen_result["message"] == "Video generated successfully"
    assert gen_result["generation_mode"] == "reference_to_video"


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_video_reference_images_limited_to_3(
    tmp_path: Path,
) -> None:
    """Test that reference images are limited to 3 for VEO 3.1."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    # Create 5 reference images (should be limited to 3)
    reference_images = [
        _create_test_image(width=100, height=100, mode="RGB") for _ in range(5)
    ]

    video_obj = FakeVideoObject(video_bytes=b"fake video content")
    gen_video = FakeGeneratedVideo(video_obj)
    result = FakeVideoResult([gen_video])
    operation = FakeOperation(done=True, result=result)

    client = FakeGenaiClient(vertexai=True, operation=operation)

    gen_result = await generate_video(
        client=client,  # type: ignore[arg-type]
        prompt="Video with references",
        videos_dir=videos_dir,
        model="veo-3.1-generate-001",
        reference_images=reference_images,
    )

    assert gen_result["message"] == "Video generated successfully"
    assert gen_result["generation_mode"] == "reference_to_video"


# ============================================================================
# generate_video tests - VEO 3.1 Video Extension
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_video_extend(
    tmp_path: Path,
) -> None:
    """Test generate_video with video extension for VEO 3.1."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    video_obj = FakeVideoObject(video_bytes=b"extended video content")
    gen_video = FakeGeneratedVideo(video_obj)
    result = FakeVideoResult([gen_video])
    operation = FakeOperation(done=True, result=result)

    client = FakeGenaiClient(vertexai=True, operation=operation)

    gen_result = await generate_video(
        client=client,  # type: ignore[arg-type]
        prompt="Continue the action",
        videos_dir=videos_dir,
        model="veo-3.1-generate-001",
        extend_video_uri="gs://bucket/original_video.mp4",
    )

    assert gen_result["message"] == "Video generated successfully"
    assert gen_result["generation_mode"] == "extend_video"
    assert gen_result["extended_from"] == "gs://bucket/original_video.mp4"


# ============================================================================
# generate_video tests - Generation mode priority
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_video_refuses_extend_combined_with_image_inputs(
    tmp_path: Path,
) -> None:
    """Conflicting inputs are refused, not silently resolved by a ladder.

    This test previously asserted the ladder: pass every input at once and
    extend_video quietly wins. That is what made the defect invisible — the
    caller paid to fetch a first frame, a last frame and a reference set, all
    three were dropped without a word, and the response reported success.
    """
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    video_obj = FakeVideoObject(video_bytes=b"video content")
    gen_video = FakeGeneratedVideo(video_obj)
    result = FakeVideoResult([gen_video])
    operation = FakeOperation(done=True, result=result)

    client = FakeGenaiClient(operation=operation)

    with pytest.raises(ValueError) as excinfo:
        await generate_video(
            client=client,  # type: ignore[arg-type]
            prompt="Test priority",
            videos_dir=videos_dir,
            model="veo-3.1-generate-001",
            image_bytes=_create_test_image(),
            last_frame_bytes=_create_test_image(),
            reference_images=[_create_test_image()],
            extend_video_uri="gs://bucket/video.mp4",
        )

    # The message has to name what was dropped, or the caller cannot tell
    # which of the three inputs to remove.
    # The message names every input that would have been dropped, in the
    # caller's vocabulary (the tool parameters) rather than the impl's.
    message = str(excinfo.value)
    assert "extend_video_uri" in message
    assert "image_uri/image_base64" in message
    assert "last_frame_uri" in message
    assert "reference_image_uris" in message


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_video_refuses_references_combined_with_frames(
    tmp_path: Path,
) -> None:
    """Reference-to-video silently discarded a paid-for first frame.

    Worse than a dropped parameter: reference mode also forces the render to
    8 seconds, so a 4-second request was billed double for a render that
    ignored the frame the caller supplied.
    """
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    video_obj = FakeVideoObject(video_bytes=b"video content")
    gen_video = FakeGeneratedVideo(video_obj)
    result = FakeVideoResult([gen_video])
    operation = FakeOperation(done=True, result=result)

    client = FakeGenaiClient(operation=operation)

    with pytest.raises(ValueError) as excinfo:
        await generate_video(
            client=client,  # type: ignore[arg-type]
            prompt="Test priority",
            videos_dir=videos_dir,
            model="veo-3.1-generate-001",
            image_bytes=_create_test_image(),
            reference_images=[_create_test_image()],
        )

    message = str(excinfo.value)
    assert "reference images" in message
    assert "image_uri/image_base64" in message
    # The forced 8s render is why this is a refusal and not a warning.
    assert "8 seconds" in message


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_video_still_picks_a_mode_when_inputs_do_not_conflict(
    tmp_path: Path,
) -> None:
    """Refusing conflicts must not break the ordinary single-input paths."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    for kwargs, expected in (
        ({"extend_video_uri": "gs://bucket/video.mp4"}, "extend_video"),
        ({"reference_images": [_create_test_image()]}, "reference_to_video"),
        ({"image_bytes": _create_test_image()}, "image_to_video"),
        (
            {
                "image_bytes": _create_test_image(),
                "last_frame_bytes": _create_test_image(),
            },
            "first_last_frame",
        ),
        ({}, "text_to_video"),
    ):
        client = FakeGenaiClient(
            vertexai=True,
            operation=FakeOperation(
                done=True,
                result=FakeVideoResult(
                    [FakeGeneratedVideo(FakeVideoObject(video_bytes=b"video content"))]
                ),
            ),
        )
        gen_result = await generate_video(
            client=client,  # type: ignore[arg-type]
            prompt="Test priority",
            videos_dir=videos_dir,
            model="veo-3.1-generate-001",
            **kwargs,
        )
        assert gen_result["generation_mode"] == expected


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_video_text_only_mode(
    tmp_path: Path,
) -> None:
    """Test generate_video with text only (no images or video input)."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    video_obj = FakeVideoObject(video_bytes=b"video content")
    gen_video = FakeGeneratedVideo(video_obj)
    result = FakeVideoResult([gen_video])
    operation = FakeOperation(done=True, result=result)

    client = FakeGenaiClient(operation=operation)

    gen_result = await generate_video(
        client=client,  # type: ignore[arg-type]
        prompt="A bird flying",
        videos_dir=videos_dir,
        model="veo-3.1-generate-001",
    )

    assert gen_result["generation_mode"] == "text_to_video"


# ============================================================================
# generate_video tests - VEO 3.1 Fast model
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_video_veo3_fast_with_features(
    tmp_path: Path,
) -> None:
    """Test VEO 3.1 Fast model supports all VEO 3.1 features."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    first_frame = _create_test_image(width=100, height=100, mode="RGB")
    last_frame = _create_test_image(width=100, height=100, mode="RGB")

    video_obj = FakeVideoObject(video_bytes=b"video content")
    gen_video = FakeGeneratedVideo(video_obj)
    result = FakeVideoResult([gen_video])
    operation = FakeOperation(done=True, result=result)

    # Set vertexai=True because we're testing audio (only supported in Vertex AI)
    client = FakeGenaiClient(operation=operation, vertexai=True)

    gen_result = await generate_video(
        client=client,  # type: ignore[arg-type]
        prompt="Fast transition",
        videos_dir=videos_dir,
        model="veo-3.1-fast-generate-001",
        image_bytes=first_frame,
        last_frame_bytes=last_frame,
        include_audio=True,
    )

    assert gen_result["message"] == "Video generated successfully"
    assert gen_result["generation_mode"] == "first_last_frame"
    assert gen_result["audio_enabled"] is True


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_video_extend_rejects_path_traversal(tmp_path: Path) -> None:
    """extend_video_uri with file:// must be validated against allowed_dir."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    data_dir = tmp_path  # allowed root
    outside = tmp_path.parent / "escape.mp4"

    video_obj = FakeVideoObject(video_bytes=b"fake video")
    gen_video = FakeGeneratedVideo(video_obj)
    result = FakeVideoResult([gen_video])
    operation = FakeOperation(done=True, result=result)
    client = FakeGenaiClient(vertexai=True, operation=operation)

    with pytest.raises(ValueError, match="outside the permitted data folder"):
        await generate_video(
            client=client,  # type: ignore[arg-type]
            prompt="Extend",
            videos_dir=videos_dir,
            model="veo-3.1-generate-001",
            extend_video_uri=f"file://{outside}",
            allowed_dir=data_dir,
        )


# ============================================================================
# generate_video tests - Duration snapping and reporting
# ============================================================================


def _basic_client(vertexai: bool = True) -> "FakeGenaiClient":
    """Build a client whose fake operation returns a single video with bytes."""
    video_obj = FakeVideoObject(video_bytes=b"fake video content")
    gen_video = FakeGeneratedVideo(video_obj)
    result = FakeVideoResult([gen_video])
    operation = FakeOperation(done=True, result=result)
    return FakeGenaiClient(operation=operation, vertexai=vertexai)


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        (5.0, 4),
        (5.5, 6),
        (7.5, 8),
        (4.0, 4),
        (100.0, 8),
    ],
)
@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_video_duration_snapped_and_returned(
    requested: float,
    expected: int,
    tmp_path: Path,
) -> None:
    """Duration is snapped to [4,6,8] and the final value is returned."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    gen_result = await generate_video(
        client=_basic_client(),  # type: ignore[arg-type]
        prompt="Duration test",
        videos_dir=videos_dir,
        model="veo-3.1-generate-001",
        duration_seconds=requested,
    )

    assert gen_result["duration_seconds"] == expected


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_video_reference_duration_forced_to_8(tmp_path: Path) -> None:
    """Reference-to-video forces duration to 8 and reports it."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    gen_result = await generate_video(
        client=_basic_client(vertexai=True),  # type: ignore[arg-type]
        prompt="Reference",
        videos_dir=videos_dir,
        model="veo-3.1-generate-001",
        reference_images=[_create_test_image()],
        duration_seconds=4.0,
    )

    assert gen_result["duration_seconds"] == 8


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_video_extend_duration_forced_to_7(tmp_path: Path) -> None:
    """Extend video forces duration to 7 and reports it."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    gen_result = await generate_video(
        client=_basic_client(vertexai=True),  # type: ignore[arg-type]
        prompt="Extend",
        videos_dir=videos_dir,
        model="veo-3.1-generate-001",
        extend_video_uri="gs://bucket/video.mp4",
    )

    assert gen_result["duration_seconds"] == 7


# ============================================================================
# generate_video tests - Aspect ratio validation
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_video_invalid_aspect_ratio_raises(tmp_path: Path) -> None:
    """Unsupported aspect ratios raise instead of being silently coerced."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    with pytest.raises(ValueError, match="Unsupported aspect_ratio"):
        await generate_video(
            client=_basic_client(),  # type: ignore[arg-type]
            prompt="Bad aspect",
            videos_dir=videos_dir,
            model="veo-3.1-generate-001",
            aspect_ratio="4:3",
        )


# ============================================================================
# generate_video tests - GCS output requires Vertex AI
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_video_refuses_a_gemini_api_client(tmp_path: Path) -> None:
    """Veo has no endpoint on the Gemini Developer API since 2026-10-22.

    The impl refuses a non-Vertex client before anything is sent, so nothing
    that calls it directly can reach a switched-off model ID.
    """
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    sent: list[dict[str, Any]] = []
    client = FakeGenaiClient(vertexai=False)

    def recording_generate_videos(**kwargs: Any) -> FakeOperation:
        sent.append(kwargs)
        return FakeOperation()

    client.models.generate_videos = recording_generate_videos  # type: ignore[assignment]

    with pytest.raises(ValueError) as refused:
        await generate_video(
            client=client,  # type: ignore[arg-type]
            prompt="A cat walking",
            videos_dir=videos_dir,
            model="veo-3.1-fast-generate-001",
        )

    assert str(refused.value) == VEO_GEMINI_API_REFUSAL
    assert "2026-10-22" in str(refused.value)
    assert "generate_video_omni" in str(refused.value)
    assert "gemini-omni-1.1-flash" in str(refused.value)
    assert "Vertex AI credentials" in str(refused.value)
    assert sent == []


def test_the_shared_validator_refuses_the_gemini_api_backend() -> None:
    """The validator the quotes and the render share refuses the backend before
    it reads the request, so a caller that checks options before resolving a
    client (loop_extend does) gets the same refusal as everything else."""
    from src.video import validate_render_options

    with pytest.raises(ValueError) as refused:
        validate_render_options("veo-3.1-generate-001", "720p", backend="gemini_api")
    assert str(refused.value) == VEO_GEMINI_API_REFUSAL
    # Vertex is unaffected.
    validate_render_options("veo-3.1-generate-001", "4K", backend="vertex")


@pytest.mark.parametrize(
    ("older_spelling", "canonical"),
    [
        ("veo-3.1-generate-preview", "veo-3.1-generate-001"),
        ("veo-3.1-fast-generate-preview", "veo-3.1-fast-generate-001"),
    ],
)
@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_an_older_preview_spelling_renders_as_its_001_model_on_vertex(
    older_spelling: str, canonical: str, tmp_path: Path
) -> None:
    """A manifest from an older Gemini-API render names the `-preview` ID.
    Fed back on Vertex, it must reach the `-001` model Vertex serves, since a
    raw `-preview` 404s there."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    sent: list[dict[str, Any]] = []
    video_obj = FakeVideoObject(video_bytes=b"fake video content")
    operation = FakeOperation(
        done=True, result=FakeVideoResult([FakeGeneratedVideo(video_obj)])
    )
    client = FakeGenaiClient(operation=operation)

    def recording_generate_videos(**kwargs: Any) -> FakeOperation:
        sent.append(kwargs)
        return operation

    client.models.generate_videos = recording_generate_videos  # type: ignore[assignment]

    result = await generate_video(
        client=client,  # type: ignore[arg-type]
        prompt="rain",
        videos_dir=videos_dir,
        model=older_spelling,  # type: ignore[arg-type]
    )

    assert sent[0]["model"] == canonical
    assert result["model"] == canonical


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_video_gcs_output_allowed_on_vertex(tmp_path: Path) -> None:
    """output_gcs_uri is accepted when the client is in Vertex AI mode."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    video_obj = FakeVideoObject(video_bytes=b"fake video content")
    gen_video = FakeGeneratedVideo(video_obj)
    result = FakeVideoResult([gen_video])
    operation = FakeOperation(done=True, result=result)
    client = FakeGenaiClient(operation=operation, vertexai=True)

    gen_result = await generate_video(
        client=client,  # type: ignore[arg-type]
        prompt="GCS output",
        videos_dir=videos_dir,
        model="veo-3.1-generate-001",
        output_gcs_uri="gs://bucket/out.mp4",
    )

    assert gen_result["message"] == "Video generated successfully"


# ============================================================================
# generate_video tests - resolution and person_generation
# ============================================================================


@pytest.mark.parametrize("resolution", ["720p", "1080p", "4K"])
@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_video_resolution_passed_through(
    resolution: str,
    tmp_path: Path,
) -> None:
    """Valid resolution values are passed through to the config."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    captured: dict[str, Any] = {}

    video_obj = FakeVideoObject(video_bytes=b"fake video content")
    gen_video = FakeGeneratedVideo(video_obj)
    result = FakeVideoResult([gen_video])
    operation = FakeOperation(done=True, result=result)
    client = FakeGenaiClient(vertexai=True, operation=operation)

    def capturing_generate_videos(**kwargs: Any) -> FakeOperation:
        captured["config"] = kwargs.get("config")
        return operation

    client.models.generate_videos = capturing_generate_videos  # type: ignore[assignment]

    gen_result = await generate_video(
        client=client,  # type: ignore[arg-type]
        prompt="Resolution test",
        videos_dir=videos_dir,
        model="veo-3.1-generate-001",
        resolution=resolution,
    )

    assert gen_result["message"] == "Video generated successfully"
    assert captured["config"].resolution == resolution


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_video_resolution_invalid_raises(tmp_path: Path) -> None:
    """An unsupported resolution value raises."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    with pytest.raises(ValueError, match="Unsupported resolution"):
        await generate_video(
            client=_basic_client(),  # type: ignore[arg-type]
            prompt="Bad resolution",
            videos_dir=videos_dir,
            model="veo-3.1-generate-001",
            resolution="8K",
        )


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_video_person_generation_passed_through(tmp_path: Path) -> None:
    """person_generation is passed through to the config as-is."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    captured: dict[str, Any] = {}

    video_obj = FakeVideoObject(video_bytes=b"fake video content")
    gen_video = FakeGeneratedVideo(video_obj)
    result = FakeVideoResult([gen_video])
    operation = FakeOperation(done=True, result=result)
    client = FakeGenaiClient(operation=operation)

    def capturing_generate_videos(**kwargs: Any) -> FakeOperation:
        captured["config"] = kwargs.get("config")
        return operation

    client.models.generate_videos = capturing_generate_videos  # type: ignore[assignment]

    gen_result = await generate_video(
        client=client,  # type: ignore[arg-type]
        prompt="Person generation test",
        videos_dir=videos_dir,
        model="veo-3.1-generate-001",
        person_generation="allow_adult",
    )

    assert gen_result["message"] == "Video generated successfully"
    assert captured["config"].person_generation == "allow_adult"


# ============================================================================
# generate_video tests - Audio intent
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_video_no_audio_warning_on_vertex(tmp_path: Path) -> None:
    """On Vertex AI, include_audio=False is honored, so no warning is added."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    video_obj = FakeVideoObject(video_bytes=b"fake video content")
    gen_video = FakeGeneratedVideo(video_obj)
    result = FakeVideoResult([gen_video])
    operation = FakeOperation(done=True, result=result)
    client = FakeGenaiClient(operation=operation, vertexai=True)

    gen_result = await generate_video(
        client=client,  # type: ignore[arg-type]
        prompt="Silent video on Vertex",
        videos_dir=videos_dir,
        model="veo-3.1-generate-001",
        include_audio=False,
    )

    assert "warnings" not in gen_result
    assert gen_result["audio_enabled"] is False


# ============================================================================
# Round-4 regression tests
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_video_last_frame_without_first_raises(
    tmp_path: Path,
) -> None:
    """A last frame without a first frame raises instead of silently
    degrading to text-to-video and discarding the frame."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    with pytest.raises(ValueError, match="last frame.*without a first frame"):
        await generate_video(
            client=FakeGenaiClient(),  # type: ignore[arg-type]
            prompt="end on this",
            videos_dir=videos_dir,
            model="veo-3.1-generate-001",
            last_frame_bytes=_create_test_image(),
        )


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_video_vertex_sends_generate_audio_false(
    tmp_path: Path,
) -> None:
    """On Vertex, include_audio=False is sent explicitly so the API default
    (audio on) cannot silently contradict the request."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    captured: dict[str, Any] = {}

    video_obj = FakeVideoObject(video_bytes=b"fake video content")
    operation = FakeOperation(
        done=True, result=FakeVideoResult([FakeGeneratedVideo(video_obj)])
    )
    client = FakeGenaiClient(operation=operation, vertexai=True)

    def capturing_generate_videos(**kwargs: Any) -> FakeOperation:
        captured["config"] = kwargs.get("config")
        return operation

    client.models.generate_videos = capturing_generate_videos  # type: ignore[assignment]

    result = await generate_video(
        client=client,  # type: ignore[arg-type]
        prompt="silent clip",
        videos_dir=videos_dir,
        model="veo-3.1-generate-001",
        include_audio=False,
    )

    assert captured["config"].generate_audio is False
    assert result["audio_enabled"] is False


# ============================================================================
# Image inputs: header-checked, passed through, off the loop
# ============================================================================


def _encoded(fmt: str, size: tuple[int, int] = (64, 64), mode: str = "RGB") -> bytes:
    from io import BytesIO

    from PIL import Image

    color = {"RGB": (1, 2, 3), "RGBA": (1, 2, 3, 4)}.get(mode, 5)
    buf = BytesIO()
    Image.new(mode, size, color).save(buf, fmt)  # pyright: ignore[reportArgumentType]
    return buf.getvalue()


def test_png_and_jpeg_inputs_are_sent_as_the_bytes_they_arrived_as() -> None:
    """Every input used to be decoded and re-encoded as PNG whenever the mode
    was RGB -- a 5 MB 4000x4000 JPEG went onto the wire as a 28.5 MB PNG. The
    caller's encoding is already the right one."""
    from src.video import _prepare_image_input

    png = _encoded("PNG")
    jpeg = _encoded("JPEG")
    out_png = _prepare_image_input(png)
    out_jpeg = _prepare_image_input(jpeg)
    assert out_png.image_bytes is png and out_png.mime_type == "image/png"
    assert out_jpeg.image_bytes is jpeg and out_jpeg.mime_type == "image/jpeg"


def test_other_formats_are_normalised_to_something_the_service_accepts() -> None:
    from io import BytesIO

    from PIL import Image

    from src.video import _prepare_image_input

    # Lossless stays lossless: an RGB WEBP/BMP/TIFF reference is converted to
    # PNG, never to JPEG q75 (the first cut of the pass-through did that).
    webp = _prepare_image_input(_encoded("WEBP"))
    assert webp.mime_type == "image/png"
    assert Image.open(BytesIO(webp.image_bytes)).format == "PNG"
    bmp = _prepare_image_input(_encoded("BMP"))
    assert bmp.mime_type == "image/png"
    palette = _prepare_image_input(_encoded("PNG", mode="P"))
    assert palette.mime_type == "image/png"
    assert Image.open(BytesIO(palette.image_bytes)).mode == "RGBA"
    # Only a JPEG source is re-encoded as JPEG (e.g. a CMYK JPEG).
    cmyk = _prepare_image_input(_encoded("JPEG", mode="CMYK"))
    assert cmyk.mime_type == "image/jpeg"


def test_an_oversized_input_is_refused_from_its_header_without_decoding() -> None:
    """A 308 KB 9999x9999 solid PNG is under the fetch cap and under Pillow's
    hard bomb threshold, and cost +382 MB RSS in 1.6s on the loop. The size
    is in the header; nothing is materialised."""
    import time

    from src.video import _prepare_image_input

    from PIL import Image

    huge = _encoded("PNG", size=(9999, 9999))
    start = time.perf_counter()
    # Pillow flags the header size on open; that signal is expected, and our
    # own 40 MP refusal fires before anything is decoded.
    with pytest.warns(Image.DecompressionBombWarning):
        with pytest.raises(ValueError, match="above the 40.0MP limit"):
            _prepare_image_input(huge)
    assert time.perf_counter() - start < 1.0


def test_reference_inputs_are_capped_with_a_warning() -> None:
    from src.video import _MAX_REFERENCE_IMAGES, _prepare_frame_inputs

    first, last, refs, warnings = _prepare_frame_inputs(
        "reference_to_video", None, None, [_encoded("JPEG")] * 5
    )
    assert first is None and last is None
    assert len(refs) == _MAX_REFERENCE_IMAGES
    assert warnings and "were not" in warnings[0]


def test_image_inputs_are_prepared_off_the_event_loop() -> None:
    """Source-level, like the other executor guards: five near-limit inputs
    measured ~8s of frozen loop when prepared inline in async generate_video.
    Scanned across line breaks -- ruff wraps the call."""
    import re
    from pathlib import Path

    source = Path("src/video.py").read_text()
    assert re.search(
        r"await\s+run_off_loop\(\s*functools\.partial\(\s*_prepare_frame_inputs", source
    ), "_prepare_frame_inputs must run through run_off_loop"
    # And no bare call is left in the coroutine body.
    body = source[source.index("async def generate_video(") :]
    assert not re.search(r"^\s+\w+ = _prepare_image_input\(", body, re.M)


@pytest.mark.parametrize("fmt", ["PNG", "JPEG"])
def test_a_truncated_input_is_refused_locally_not_on_the_wire(fmt: str) -> None:
    """The old full decode refused a truncated file with "image file is
    truncated". The first cut of the pass-through skipped the decode, so the
    same file went onto the wire to fail as an opaque API 400 after the
    upload. The bytes are decoded once (off the loop, under the pixel cap) to
    prove they are whole, and still sent as they arrived."""
    from src.video import _prepare_image_input

    whole = _encoded(fmt, size=(400, 300))
    with pytest.raises(OSError):
        _prepare_image_input(whole[: len(whole) // 2])
    # A whole file is still passed through untouched.
    assert _prepare_image_input(whole).image_bytes is whole


# --------------------------------------------------------------------------
# How far a timed-out render got
# --------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_a_submit_that_times_out_may_have_reached_google(tmp_path: Path) -> None:
    """The submit call itself ran out of time: the request may have arrived,
    so the render may still bill. Neither "submitted" nor "nothing billed"."""
    from src.video import VeoTimeoutError

    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    client = FakeGenaiClient(raise_error=TimeoutError("submit stalled"), vertexai=True)
    with pytest.raises(VeoTimeoutError) as caught:
        await generate_video(
            client=client,  # type: ignore[arg-type]
            prompt="rain",
            videos_dir=videos_dir,
            model="veo-3.1-generate-001",
        )
    assert caught.value.submission == "unknown"
    assert caught.value.operation_name is None


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_a_poll_that_times_out_was_submitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Polling only happens once Google accepted the request. A plain timeout
    from a stalled poll used to carry no operation name, so the tool told the
    caller nothing had been billed."""
    import src.video as video_mod
    from src.video import VeoTimeoutError

    monkeypatch.setattr(video_mod, "_VEO_POLL_INTERVAL_SECONDS", 0.0)
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    operation = FakeOperation(name="operations/abc")
    operation.set_done_after_polls(2)
    client = FakeGenaiClient(operation=operation, vertexai=True)

    def stalled(op: Any) -> Any:
        raise TimeoutError("poll stalled")

    monkeypatch.setattr(client.operations, "get", stalled)
    with pytest.raises(VeoTimeoutError) as caught:
        await generate_video(
            client=client,  # type: ignore[arg-type]
            prompt="rain",
            videos_dir=videos_dir,
            model="veo-3.1-generate-001",
        )
    assert caught.value.submission == "submitted"
    assert caught.value.operation_name == "operations/abc"


@pytest.mark.parametrize(
    ("submission", "operation_name", "phrase"),
    [
        ("submitted", "operations/abc", "Operation operations/abc was submitted"),
        ("unknown", None, "may have reached Google"),
        ("none", None, "nothing was billed"),
    ],
)
@pytest.mark.asyncio
@pytest.mark.timeout(10.0)
async def test_the_tool_says_how_far_a_timed_out_render_got(
    submission: str,
    operation_name: str | None,
    phrase: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json
    from unittest.mock import AsyncMock, MagicMock

    import src.__main__ as server
    from src.__main__ import AppContext
    from src.__main__ import generate_video as generate_video_tool
    from src.video import VeoTimeoutError

    async def timed_out(**kwargs: Any) -> Any:
        raise VeoTimeoutError(
            "deadline",
            operation_name=operation_name,
            submission=submission,  # type: ignore[arg-type]
        )

    monkeypatch.setattr(server, "generate_video_impl", timed_out)
    (tmp_path / "videos").mkdir()
    client = MagicMock()
    client._api_client.vertexai = True
    ctx = MagicMock()
    ctx.info = AsyncMock()
    ctx.error = AsyncMock()
    ctx.warning = AsyncMock()
    ctx.request_context.lifespan_context = AppContext(
        data_folder=tmp_path,
        images_dir=tmp_path / "images",
        videos_dir=tmp_path / "videos",
        client=client,
    )
    body = json.loads(
        await generate_video_tool(
            ctx=ctx, prompt="rain", model="veo-3.1-generate-001", duration_seconds=4.0
        )
    )
    assert body["submission"] == submission
    assert phrase in body["note"]


def _httpx_request() -> Any:
    import httpx

    return httpx.Request("POST", "https://example.invalid/v1/interactions")


def _httpx_response(status: int) -> Any:
    import httpx

    return httpx.Response(status, request=_httpx_request())


@pytest.mark.parametrize(
    ("exc", "refused"),
    [
        # Either SDK's timeout, a dropped connection, a 5xx: it may have
        # reached Google.
        (TimeoutError("stalled"), False),
        ("interactions_timeout", False),
        ("interactions_504", False),
        (genai_errors.ServerError(503, {"error": {"message": "UNAVAILABLE"}}), False),
        # Turned down by the service, or never sent.
        (genai_errors.ClientError(400, {"error": {"message": "bad"}}), True),
        (genai_errors.ClientError(429, {"error": {"message": "quota"}}), True),
        ("interactions_400", True),
        (ValueError("bad argument"), True),
    ],
)
def test_only_a_clear_refusal_means_nothing_was_started(
    exc: Any, refused: bool
) -> None:
    """The Interactions client raises its own types, none of them a Python
    TimeoutError; tracking that only knew TimeoutError called a timed-out
    create "none"."""
    from google.genai._gaos.lib import compat_errors as interactions_errors

    from src.video import refused_before_work

    if exc == "interactions_timeout":
        exc = interactions_errors.APITimeoutError(request=_httpx_request())
    elif exc == "interactions_504":
        exc = interactions_errors.InternalServerError(
            "gateway timeout", response=_httpx_response(504), body=None
        )
    elif exc == "interactions_400":
        exc = interactions_errors.BadRequestError(
            "bad", response=_httpx_response(400), body=None
        )
    assert refused_before_work(exc) is refused


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_a_submit_that_fails_ambiguously_may_have_reached_google(
    tmp_path: Path,
) -> None:
    from src.video import VeoSubmissionError

    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    client = FakeGenaiClient(
        raise_error=genai_errors.ServerError(
            503, {"error": {"message": "UNAVAILABLE"}}
        ),
        vertexai=True,
    )
    with pytest.raises(VeoSubmissionError) as caught:
        await generate_video(
            client=client,  # type: ignore[arg-type]
            prompt="rain",
            videos_dir=videos_dir,
            model="veo-3.1-generate-001",
        )
    assert caught.value.submission == "unknown"


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_a_refused_submit_is_raised_as_it_was(tmp_path: Path) -> None:
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    refusal = genai_errors.ClientError(403, {"error": {"message": "PERMISSION_DENIED"}})
    client = FakeGenaiClient(raise_error=refusal, vertexai=True)
    with pytest.raises(genai_errors.ClientError):
        await generate_video(
            client=client,  # type: ignore[arg-type]
            prompt="rain",
            videos_dir=videos_dir,
            model="veo-3.1-generate-001",
        )


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_a_completed_render_that_fails_to_save_was_submitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The render finished and bills; a full disk afterwards must not make it
    look like nothing was sent."""
    from src.video import VeoSubmissionError

    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    video = FakeVideoObject()  # no bytes and no gs:// URI: it is downloaded
    operation = FakeOperation(
        result=FakeVideoResult([FakeGeneratedVideo(video)]), name="operations/done"
    )
    client = FakeGenaiClient(operation=operation, vertexai=True)

    def disk_full(path: str) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(video, "save", disk_full)
    with pytest.raises(VeoSubmissionError) as caught:
        await generate_video(
            client=client,  # type: ignore[arg-type]
            prompt="rain",
            videos_dir=videos_dir,
            model="veo-3.1-generate-001",
        )
    assert (caught.value.submission, caught.value.operation_name) == (
        "submitted",
        "operations/done",
    )


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_an_operation_google_reports_as_failed_is_not_a_submission(
    tmp_path: Path,
) -> None:
    """Google bills Veo only for a generated video, so an operation it reports
    as failed is not one to reconcile."""
    from src.video import VeoOperationFailed

    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    client = FakeGenaiClient(operation=FakeOperation(error="boom"), vertexai=True)
    with pytest.raises(VeoOperationFailed):
        await generate_video(
            client=client,  # type: ignore[arg-type]
            prompt="rain",
            videos_dir=videos_dir,
            model="veo-3.1-generate-001",
        )


# --------------------------------------------------------------------------
# The wait a refusal asks for
# --------------------------------------------------------------------------


def test_a_retry_after_header_is_read_as_seconds() -> None:
    from types import SimpleNamespace

    from src.video import retry_after_seconds

    exc = SimpleNamespace(response=SimpleNamespace(headers={"retry-after": "25"}))
    assert retry_after_seconds(exc) == 25.0  # type: ignore[arg-type]
    dated = SimpleNamespace(
        response=SimpleNamespace(
            headers={"Retry-After": "Thu, 01 Jan 2026 00:00:00 GMT"}
        )
    )
    assert retry_after_seconds(dated) == 0.0  # type: ignore[arg-type]
    assert retry_after_seconds(SimpleNamespace(response=None)) is None  # type: ignore[arg-type]


@pytest.mark.parametrize("delay", ["25s", "25.5s", {"seconds": 25, "nanos": 500000000}])
def test_retry_info_in_the_error_details_is_read(delay: object) -> None:
    from types import SimpleNamespace

    from src.video import retry_after_seconds

    exc = SimpleNamespace(
        response=None,
        details={
            "error": {
                "status": "RESOURCE_EXHAUSTED",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.RetryInfo",
                        "retryDelay": delay,
                    }
                ],
            }
        },
    )
    assert retry_after_seconds(exc) == pytest.approx(  # type: ignore[arg-type]
        25.0 if delay == "25s" else 25.5
    )
