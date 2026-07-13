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

    def get(self, **kwargs: Any) -> FakeInteraction:
        self.get_calls.append(kwargs)
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
    # response_format is TOP-LEVEL and carries the aspect ratio.
    assert kwargs["response_format"] == {"type": "video", "aspect_ratio": "9:16"}
    # generation_config carries ONLY the task; no duration anywhere.
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

    await generate_video_omni(
        client=client,  # type: ignore[arg-type]
        prompt="edit this",
        videos_dir=videos_dir,
        input_video_bytes=b"SRC",
    )
    kwargs = interactions.create_kwargs
    assert kwargs is not None
    assert kwargs["generation_config"]["video_config"]["task"] == "edit"
    vid_part = kwargs["input"][1]
    assert vid_part["type"] == "video"
    assert vid_part["mime_type"] == "video/mp4"
    assert base64.b64decode(vid_part["data"]) == b"SRC"


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
    assert interactions.get_calls[0] == {"interaction_id": "int-9"}


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
async def test_duration_is_advisory_and_never_sent(tmp_path: Path) -> None:
    """Duration is echoed for planning with a warning, and never appears in
    the request (no documented field exists on any Omni surface)."""
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
    assert any("not controllable" in w for w in result["warnings"])
    kwargs = interactions.create_kwargs
    assert kwargs is not None
    assert "duration" not in str(kwargs["generation_config"])
    assert "duration" not in str(kwargs["response_format"])
