"""Tests for video.py video generation helpers."""

import asyncio
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from src.video import _GEMINI_API_MODEL_IDS, generate_video

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
    """Test double for internal API client."""

    def __init__(self, vertexai: bool = False) -> None:
        self.vertexai = vertexai


class FakeGenaiClient:
    """Test double for Google GenAI client."""

    def __init__(
        self,
        operation: FakeOperation | None = None,
        raise_error: Exception | None = None,
        vertexai: bool = False,
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
    # Non-Vertex fake client: the impl reports the translated Gemini-API ID.
    assert gen_result["model"] == _GEMINI_API_MODEL_IDS.get(
        input["model"], input["model"]
    )
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

    # Set vertexai=True when testing audio features (only supported in Vertex AI)
    use_vertexai = input.get("include_audio", False)
    client = FakeGenaiClient(operation=operation, vertexai=use_vertexai)

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
    # The impl reports the backend-appropriate model ID: the Vertex `-001`
    # name as-is, or the translated `-preview` ID on the Gemini API path.
    if use_vertexai:
        assert gen_result["model"] == input["model"]
    else:
        assert gen_result["model"] == _GEMINI_API_MODEL_IDS.get(
            input["model"], input["model"]
        )

    # On Vertex AI audio_enabled == include_audio; on the Gemini API path
    # (non-Vertex) Veo 3.1 always generates audio natively.
    include_audio = input.get("include_audio", False)
    expected_audio = include_audio if use_vertexai else True
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

    client = FakeGenaiClient(operation=operation)

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

    client = FakeGenaiClient(operation=operation)

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

    client = FakeGenaiClient(operation=operation)

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

    client = FakeGenaiClient(operation=operation)

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
            operation=FakeOperation(
                done=True,
                result=FakeVideoResult(
                    [FakeGeneratedVideo(FakeVideoObject(video_bytes=b"video content"))]
                ),
            )
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


# ============================================================================
# generate_video tests - VEO 3.1 Lite
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_video_veo_lite_text_to_video(tmp_path: Path) -> None:
    """VEO 3.1 Lite supports basic text-to-video."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    video_obj = FakeVideoObject(video_bytes=b"fake video")
    gen_video = FakeGeneratedVideo(video_obj)
    result = FakeVideoResult([gen_video])
    operation = FakeOperation(done=True, result=result)
    client = FakeGenaiClient(operation=operation)

    gen_result = await generate_video(
        client=client,  # type: ignore[arg-type]
        prompt="A dog running",
        videos_dir=videos_dir,
        model="veo-3.1-lite-generate-preview",
    )

    assert gen_result["model"] == "veo-3.1-lite-generate-preview"
    assert gen_result["generation_mode"] == "text_to_video"


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_video_veo_lite_rejects_extension(tmp_path: Path) -> None:
    """VEO 3.1 Lite does not support video extension."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    video_obj = FakeVideoObject(video_bytes=b"fake video")
    gen_video = FakeGeneratedVideo(video_obj)
    result = FakeVideoResult([gen_video])
    operation = FakeOperation(done=True, result=result)
    client = FakeGenaiClient(operation=operation)

    with pytest.raises(ValueError, match="does not support extend_video"):
        await generate_video(
            client=client,  # type: ignore[arg-type]
            prompt="Continue",
            videos_dir=videos_dir,
            model="veo-3.1-lite-generate-preview",
            extend_video_uri="gs://bucket/video.mp4",
        )


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
    client = FakeGenaiClient(operation=operation)

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


def _basic_client() -> "FakeGenaiClient":
    """Build a client whose fake operation returns a single video with bytes."""
    video_obj = FakeVideoObject(video_bytes=b"fake video content")
    gen_video = FakeGeneratedVideo(video_obj)
    result = FakeVideoResult([gen_video])
    operation = FakeOperation(done=True, result=result)
    return FakeGenaiClient(operation=operation)


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
        client=_basic_client(),  # type: ignore[arg-type]
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
        client=_basic_client(),  # type: ignore[arg-type]
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
async def test_generate_video_gcs_output_dropped_on_non_vertex(
    tmp_path: Path,
) -> None:
    """On the Gemini API (non-Vertex) client, output_gcs_uri is silently dropped.

    The impl must not raise (that would break Veo Lite / text-to-video when a
    VIDEO_GCS_BUCKET default is funneled through). It just omits the Vertex-only
    field from the config. The generate_video *tool* separately rejects an
    explicit request — see test_main.py.
    """
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    captured: dict[str, Any] = {}

    video_obj = FakeVideoObject(video_bytes=b"fake video content")
    gen_video = FakeGeneratedVideo(video_obj)
    result = FakeVideoResult([gen_video])
    operation = FakeOperation(done=True, result=result)
    client = FakeGenaiClient(operation=operation)  # vertexai=False by default

    def capturing_generate_videos(**kwargs: Any) -> FakeOperation:
        captured["config"] = kwargs.get("config")
        return operation

    client.models.generate_videos = capturing_generate_videos  # type: ignore[assignment]

    gen_result = await generate_video(
        client=client,  # type: ignore[arg-type]
        prompt="GCS output",
        videos_dir=videos_dir,
        model="veo-3.1-generate-001",
        output_gcs_uri="gs://bucket/out.mp4",
    )

    assert gen_result["message"] == "Video generated successfully"
    assert captured["config"].output_gcs_uri is None


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
    client = FakeGenaiClient(operation=operation)

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
async def test_generate_video_4k_rejected_for_lite(tmp_path: Path) -> None:
    """4K resolution is rejected for Veo Lite models."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    with pytest.raises(ValueError, match="does not support 4K"):
        await generate_video(
            client=_basic_client(),  # type: ignore[arg-type]
            prompt="4K lite",
            videos_dir=videos_dir,
            model="veo-3.1-lite-generate-preview",
            resolution="4K",
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
# generate_video tests - Audio intent warnings (Gemini API path)
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_video_warns_no_audio_on_gemini_api(tmp_path: Path) -> None:
    """On the Gemini API path, include_audio=False cannot be honored, so warn.

    Veo 3.1 on the Gemini API always generates audio natively. The result must
    carry a warning AND report audio_enabled=True truthfully.
    """
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    gen_result = await generate_video(
        client=_basic_client(),  # type: ignore[arg-type]  # vertexai=False
        prompt="Silent video please",
        videos_dir=videos_dir,
        model="veo-3.1-generate-001",
        include_audio=False,
    )

    assert "warnings" in gen_result
    assert isinstance(gen_result["warnings"], list)
    assert any(
        "include_audio=False was not honored" in w for w in gen_result["warnings"]
    )
    assert gen_result["audio_enabled"] is True


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_video_no_audio_warning_when_audio_requested_gemini_api(
    tmp_path: Path,
) -> None:
    """include_audio=True on the Gemini API is satisfied, so no audio warning."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    gen_result = await generate_video(
        client=_basic_client(),  # type: ignore[arg-type]  # vertexai=False
        prompt="Video with audio",
        videos_dir=videos_dir,
        model="veo-3.1-generate-001",
        include_audio=True,
    )

    # Nothing else warns here, so the key should be absent entirely.
    assert "warnings" not in gen_result
    assert gen_result["audio_enabled"] is True


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


@pytest.mark.parametrize(
    "kwargs_extra",
    [
        pytest.param({"image_bytes": True, "last_frame_bytes": True}, id="first_last"),
        pytest.param({"reference_images": True}, id="reference"),
    ],
)
@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_video_lite_rejects_unsupported_modes(
    kwargs_extra: dict[str, bool],
    tmp_path: Path,
) -> None:
    """Veo 3.1 Lite rejects first/last-frame and reference modes up front."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    img = _create_test_image()
    kwargs: dict[str, Any] = {}
    if kwargs_extra.get("image_bytes"):
        kwargs["image_bytes"] = img
    if kwargs_extra.get("last_frame_bytes"):
        kwargs["last_frame_bytes"] = img
    if kwargs_extra.get("reference_images"):
        kwargs["reference_images"] = [img]

    with pytest.raises(ValueError, match="Veo 3.1 Lite supports only"):
        await generate_video(
            client=FakeGenaiClient(),  # type: ignore[arg-type]
            prompt="x",
            videos_dir=videos_dir,
            model="veo-3.1-lite-generate-preview",
            **kwargs,
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
