"""Tests for src/omni.py (gemini-omni-flash via the Interactions API)."""

import asyncio
import base64
from pathlib import Path
from typing import Any

import pytest

from src.omni import OMNI_MODEL, generate_video_omni

# ============================================================================
# Test Doubles
# ============================================================================


class FakeOutputVideo:
    def __init__(
        self,
        data: str | None = None,
        uri: str | None = None,
    ) -> None:
        self.data = data
        self.uri = uri


class FakePart:
    def __init__(
        self,
        type: str | None = None,
        data: str | None = None,
        uri: str | None = None,
        mime_type: str | None = None,
        text: str | None = None,
    ) -> None:
        self.type = type
        self.data = data
        self.uri = uri
        self.mime_type = mime_type
        self.text = text


class FakeStep:
    def __init__(self, content: list[FakePart] | None = None) -> None:
        self.content = content or []


class FakeInteraction:
    def __init__(
        self,
        id: str = "int-1",
        status: str = "completed",
        output_video: FakeOutputVideo | None = None,
        steps: list[FakeStep] | None = None,
        error: Any = None,
    ) -> None:
        self.id = id
        self.status = status
        self.output_video = output_video
        self.steps = steps or []
        self.error = error


class FakeInteractions:
    """Test double for client.interactions."""

    def __init__(
        self,
        create_result: FakeInteraction | None = None,
        get_results: list[FakeInteraction] | None = None,
    ) -> None:
        self._create_result = create_result or FakeInteraction()
        self._get_results = list(get_results or [])
        self.create_kwargs: dict[str, Any] | None = None
        self.get_calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> FakeInteraction:
        self.create_kwargs = kwargs
        return self._create_result

    def get(self, id: str, **kwargs: Any) -> FakeInteraction:
        # Mirror google-genai's real signature: the interaction id is the
        # `id` parameter (NOT `interaction_id`), so a wrong kwarg name raises
        # TypeError here instead of being silently swallowed.
        self.get_calls.append({"id": id, **kwargs})
        if self._get_results:
            return self._get_results.pop(0)
        raise AssertionError("unexpected interactions.get call")


class FakeFiles:
    def __init__(self, download_bytes: bytes = b"mp4-bytes") -> None:
        self._download_bytes = download_bytes
        self.get_calls: list[dict[str, Any]] = []

    def get(self, **kwargs: Any) -> Any:
        self.get_calls.append(kwargs)
        return object()

    def download(self, **kwargs: Any) -> bytes:
        return self._download_bytes


class FakeGenaiClient:
    def __init__(
        self,
        interactions: FakeInteractions | None = None,
        files: FakeFiles | None = None,
    ) -> None:
        self.interactions = interactions or FakeInteractions()
        self.files = files or FakeFiles()


def _inline_video_interaction(
    id: str = "int-1", payload: bytes = b"mp4-bytes"
) -> FakeInteraction:
    return FakeInteraction(
        id=id,
        status="completed",
        output_video=FakeOutputVideo(data=base64.b64encode(payload).decode()),
    )


# ============================================================================
# Happy path + request shape
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_inline_video_written_and_request_shape(tmp_path: Path) -> None:
    """Happy path: inline base64 output is written; the create request uses
    the documented Interactions API shapes."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    interactions = FakeInteractions(
        create_result=_inline_video_interaction(payload=b"VIDEO")
    )
    client = FakeGenaiClient(interactions=interactions)

    result = await generate_video_omni(
        client=client,  # type: ignore[arg-type]
        prompt="a marble rolling",
        videos_dir=videos_dir,
        aspect_ratio="9:16",
    )

    assert result["interaction_id"] == "int-1"
    assert result["model"] == OMNI_MODEL
    assert Path(result["video_url"][7:]).read_bytes() == b"VIDEO"

    kwargs = interactions.create_kwargs
    assert kwargs is not None
    assert kwargs["model"] == OMNI_MODEL
    assert kwargs["background"] is True
    # response_format is a LIST of one object carrying aspect ratio + duration.
    assert kwargs["response_format"] == [
        {"type": "video", "aspect_ratio": "9:16", "duration": "6s"}
    ]
    # generation_config carries only the task.
    assert kwargs["generation_config"] == {"video_config": {"task": "text_to_video"}}
    # input is flattened parts, prompt first.
    assert kwargs["input"][0] == {"type": "text", "text": "a marble rolling"}


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_images_become_input_parts_and_task_types(tmp_path: Path) -> None:
    """Images ride inside `input` as flattened parts; 1 image => image_to_video,
    several => reference_to_video."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    png = b"\x89PNG\r\n\x1a\nrest"
    jpg = b"\xff\xd8\xffrest"

    interactions = FakeInteractions(create_result=_inline_video_interaction())
    client = FakeGenaiClient(interactions=interactions)
    await generate_video_omni(
        client=client,  # type: ignore[arg-type]
        prompt="p",
        videos_dir=videos_dir,
        image_bytes_list=[png],
    )
    kwargs = interactions.create_kwargs
    assert kwargs is not None
    assert kwargs["generation_config"]["video_config"]["task"] == "image_to_video"
    img_part = kwargs["input"][1]
    assert img_part["type"] == "image"
    assert img_part["mime_type"] == "image/png"
    assert base64.b64decode(img_part["data"]) == png

    interactions2 = FakeInteractions(create_result=_inline_video_interaction())
    client2 = FakeGenaiClient(interactions=interactions2)
    await generate_video_omni(
        client=client2,  # type: ignore[arg-type]
        prompt="p",
        videos_dir=videos_dir,
        image_bytes_list=[png, jpg],
    )
    kwargs2 = interactions2.create_kwargs
    assert kwargs2 is not None
    assert kwargs2["generation_config"]["video_config"]["task"] == "reference_to_video"
    assert kwargs2["input"][2]["mime_type"] == "image/jpeg"


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_input_video_inlined_and_edit_task(tmp_path: Path) -> None:
    """An input video is inlined as a video part (no files.upload) and forces
    the edit task type."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    interactions = FakeInteractions(create_result=_inline_video_interaction())
    client = FakeGenaiClient(interactions=interactions)

    mp4 = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"
    await generate_video_omni(
        client=client,  # type: ignore[arg-type]
        prompt="edit this",
        videos_dir=videos_dir,
        input_video_bytes=mp4,
    )
    kwargs = interactions.create_kwargs
    assert kwargs is not None
    assert kwargs["generation_config"]["video_config"]["task"] == "edit"
    vid_part = kwargs["input"][1]
    assert vid_part["type"] == "video"
    assert vid_part["mime_type"] == "video/mp4"
    assert base64.b64decode(vid_part["data"]) == mp4


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_previous_interaction_id_forwarded(tmp_path: Path) -> None:
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    interactions = FakeInteractions(create_result=_inline_video_interaction(id="int-2"))
    client = FakeGenaiClient(interactions=interactions)

    result = await generate_video_omni(
        client=client,  # type: ignore[arg-type]
        prompt="make it stormy",
        videos_dir=videos_dir,
        previous_interaction_id="int-1",
    )
    kwargs = interactions.create_kwargs
    assert kwargs is not None
    assert kwargs["previous_interaction_id"] == "int-1"
    assert kwargs["generation_config"]["video_config"]["task"] == "edit"
    assert result["interaction_id"] == "int-2"


# ============================================================================
# Background polling & terminal statuses
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(3.0)
async def test_background_polling_until_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """create returns in_progress; the loop polls interactions.get until
    completed, then extracts the video."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    pending = FakeInteraction(id="int-9", status="in_progress")
    done = _inline_video_interaction(id="int-9", payload=b"DONE")
    interactions = FakeInteractions(
        create_result=pending,
        get_results=[FakeInteraction(id="int-9", status="queued"), done],
    )
    client = FakeGenaiClient(interactions=interactions)

    real_sleep = asyncio.sleep

    async def instant_sleep(_secs: float) -> None:
        await real_sleep(0)

    monkeypatch.setattr("src.omni.asyncio.sleep", instant_sleep)

    result = await generate_video_omni(
        client=client,  # type: ignore[arg-type]
        prompt="p",
        videos_dir=videos_dir,
    )
    assert Path(result["video_url"][7:]).read_bytes() == b"DONE"
    assert len(interactions.get_calls) == 2
    assert interactions.get_calls[0] == {"id": "int-9"}


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_terminal_status_raises(tmp_path: Path) -> None:
    """Any non-completed terminal status fails fast with the raw status."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    interactions = FakeInteractions(
        create_result=FakeInteraction(id="int-3", status="failed")
    )
    client = FakeGenaiClient(interactions=interactions)

    with pytest.raises(ValueError, match="status 'failed'"):
        await generate_video_omni(
            client=client,  # type: ignore[arg-type]
            prompt="p",
            videos_dir=videos_dir,
        )


@pytest.mark.asyncio
@pytest.mark.timeout(3.0)
async def test_polling_timeout_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interaction that never completes hits the overall deadline."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    forever = [FakeInteraction(status="in_progress") for _ in range(1000)]
    interactions = FakeInteractions(
        create_result=FakeInteraction(status="in_progress"),
        get_results=forever,
    )
    client = FakeGenaiClient(interactions=interactions)

    real_sleep = asyncio.sleep

    async def instant_sleep(_secs: float) -> None:
        await real_sleep(0)

    monkeypatch.setattr("src.omni.asyncio.sleep", instant_sleep)

    with pytest.raises(TimeoutError, match="timed out"):
        await generate_video_omni(
            client=client,  # type: ignore[arg-type]
            prompt="p",
            videos_dir=videos_dir,
            timeout_seconds=0,
        )


# ============================================================================
# Output extraction fallbacks
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_steps_fallback_when_no_output_video(tmp_path: Path) -> None:
    """Video is found in steps[].content[] when output_video is absent."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    step = FakeStep(
        content=[
            FakePart(type="text", text="rendering notes"),
            FakePart(
                type="video",
                data=base64.b64encode(b"STEPVID").decode(),
                mime_type="video/mp4",
            ),
        ]
    )
    interaction = FakeInteraction(id="int-4", status="completed", steps=[step])
    client = FakeGenaiClient(FakeInteractions(create_result=interaction))

    result = await generate_video_omni(
        client=client,  # type: ignore[arg-type]
        prompt="p",
        videos_dir=videos_dir,
    )
    assert Path(result["video_url"][7:]).read_bytes() == b"STEPVID"


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_uri_delivery_downloads_via_files(tmp_path: Path) -> None:
    """URI delivery resolves through files.get + files.download."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    interaction = FakeInteraction(
        id="int-5",
        status="completed",
        output_video=FakeOutputVideo(uri="files/abc123"),
    )
    files = FakeFiles(download_bytes=b"URIVID")
    client = FakeGenaiClient(FakeInteractions(create_result=interaction), files)

    result = await generate_video_omni(
        client=client,  # type: ignore[arg-type]
        prompt="p",
        videos_dir=videos_dir,
    )
    assert Path(result["video_url"][7:]).read_bytes() == b"URIVID"
    assert files.get_calls == [{"name": "files/abc123"}]


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_no_video_anywhere_raises(tmp_path: Path) -> None:
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    interaction = FakeInteraction(id="int-6", status="completed")
    client = FakeGenaiClient(FakeInteractions(create_result=interaction))

    with pytest.raises(ValueError, match="no inline video data and no file URI"):
        await generate_video_omni(
            client=client,  # type: ignore[arg-type]
            prompt="p",
            videos_dir=videos_dir,
        )


# ============================================================================
# Validation & advisory duration
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_invalid_aspect_ratio_raises(tmp_path: Path) -> None:
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    client = FakeGenaiClient()

    with pytest.raises(ValueError, match="Unsupported aspect_ratio"):
        await generate_video_omni(
            client=client,  # type: ignore[arg-type]
            prompt="p",
            videos_dir=videos_dir,
            aspect_ratio="1:1",
        )


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_duration_sent_as_seconds_string(tmp_path: Path) -> None:
    """Duration is sent as 'Ns' inside response_format (Vertex supports it)."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    interactions = FakeInteractions(create_result=_inline_video_interaction())
    client = FakeGenaiClient(interactions=interactions)

    result = await generate_video_omni(
        client=client,  # type: ignore[arg-type]
        prompt="p",
        videos_dir=videos_dir,
        duration_seconds=8.0,
    )
    assert result["duration_seconds"] == 8
    kwargs = interactions.create_kwargs
    assert kwargs is not None
    assert kwargs["response_format"][0]["duration"] == "8s"


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_duration_clamped_with_warning(tmp_path: Path) -> None:
    """Out-of-range durations clamp to [3, 10] with a warning."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    interactions = FakeInteractions(create_result=_inline_video_interaction())
    client = FakeGenaiClient(interactions=interactions)

    result = await generate_video_omni(
        client=client,  # type: ignore[arg-type]
        prompt="p",
        videos_dir=videos_dir,
        duration_seconds=15.0,
    )
    assert result["duration_seconds"] == 10
    assert interactions.create_kwargs["response_format"][0]["duration"] == "10s"
    assert any("clamped to 10s" in w for w in result["warnings"])


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_gcs_delivery_sets_format_and_passes_uri_through(tmp_path: Path) -> None:
    """output_gcs_uri sets delivery='uri'+gcs_uri; a gs:// result is passed
    through as video_url without a local write."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    interaction = FakeInteraction(
        id="int-7",
        status="completed",
        steps=[
            FakeStep(
                content=[
                    FakePart(
                        type="video", uri="gs://out/clip.mp4", mime_type="video/mp4"
                    )
                ]
            )
        ],
    )
    interactions = FakeInteractions(create_result=interaction)
    client = FakeGenaiClient(interactions=interactions)

    result = await generate_video_omni(
        client=client,  # type: ignore[arg-type]
        prompt="p",
        videos_dir=videos_dir,
        output_gcs_uri="gs://out/",
    )
    fmt = interactions.create_kwargs["response_format"][0]
    assert fmt["delivery"] == "uri"
    assert fmt["gcs_uri"] == "gs://out/"
    # gs:// output is passed through; nothing written locally.
    assert result["video_url"] == "gs://out/clip.mp4"
    assert not list(videos_dir.iterdir())


# ============================================================================
# MIME detection (accurate labels; reject unknown)
# ============================================================================


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        pytest.param(b"\x89PNG\r\n\x1a\nrest", "image/png", id="png"),
        pytest.param(b"\xff\xd8\xff\xe0rest", "image/jpeg", id="jpeg"),
        pytest.param(b"RIFF\x00\x00\x00\x00WEBPVP8 ", "image/webp", id="webp"),
        pytest.param(b"GIF89a....", "image/gif", id="gif"),
        pytest.param(
            b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00", "image/heic", id="heic"
        ),
        pytest.param(
            b"\x00\x00\x00\x18ftypmif1\x00\x00\x00\x00", "image/heif", id="heif"
        ),
    ],
)
def test_detect_image_mime(data: bytes, expected: str) -> None:
    from src.omni import _detect_image_mime

    assert _detect_image_mime(data) == expected


def test_detect_image_mime_rejects_unknown_and_video() -> None:
    from src.omni import _detect_image_mime

    with pytest.raises(ValueError, match="Unrecognized image"):
        _detect_image_mime(b"not-an-image-at-all")
    # An MP4 (ftyp mp42) is not an image → rejected, not mislabeled as PNG.
    with pytest.raises(ValueError, match="Unrecognized image"):
        _detect_image_mime(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00")


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        pytest.param(
            b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00", "video/mp4", id="mp4"
        ),
        pytest.param(
            b"\x00\x00\x00\x18ftypqt  \x00\x00\x00\x00", "video/quicktime", id="mov"
        ),
        pytest.param(b"\x1a\x45\xdf\xa3rest", "video/webm", id="webm"),
        pytest.param(b"\x00\x00\x01\xbarest", "video/mpeg", id="mpeg"),
        pytest.param(b"RIFF\x00\x00\x00\x00AVI LIST", "video/x-msvideo", id="avi"),
    ],
)
def test_detect_video_mime(data: bytes, expected: str) -> None:
    from src.omni import _detect_video_mime

    assert _detect_video_mime(data) == expected


def test_detect_video_mime_rejects_unknown() -> None:
    from src.omni import _detect_video_mime

    with pytest.raises(ValueError, match="Unrecognized video"):
        _detect_video_mime(b"SRC")


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_unknown_image_input_raises_in_generate(tmp_path: Path) -> None:
    """An undetectable image input surfaces a clear error rather than being
    sent with a wrong (default) MIME type."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    client = FakeGenaiClient(
        FakeInteractions(create_result=_inline_video_interaction())
    )

    with pytest.raises(ValueError, match="Unrecognized image"):
        await generate_video_omni(
            client=client,  # type: ignore[arg-type]
            prompt="p",
            videos_dir=videos_dir,
            image_bytes_list=[b"bogus-bytes"],
        )
