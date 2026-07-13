"""Tests for omni.py (gemini-omni-flash-preview / Interactions API)."""

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
    """Test double for interaction.output_video.

    Exposes ``data`` for inline base64 delivery, or ``uri`` for Files API
    (URI) delivery.
    """

    def __init__(
        self,
        data: str | None = None,
        uri: str | None = None,
    ) -> None:
        self.data = data
        self.uri = uri


class FakeInteraction:
    """Test double for a finished interaction."""

    def __init__(
        self,
        interaction_id: str = "interaction-123",
        output_video: FakeOutputVideo | None = None,
    ) -> None:
        self.id = interaction_id
        self.output_video = output_video


class FakeInteractions:
    """Test double for the interactions client."""

    def __init__(
        self,
        interaction: FakeInteraction | None = None,
        raise_error: Exception | None = None,
    ) -> None:
        self._interaction = interaction
        self._raise_error = raise_error
        self.create_kwargs: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> FakeInteraction:
        self.create_kwargs = kwargs
        if self._raise_error:
            raise self._raise_error
        return self._interaction or FakeInteraction()


class FakeFileResource:
    """Test double for a Files API resource."""

    def __init__(self, state: str = "ACTIVE") -> None:
        self.state = state


class FakeFiles:
    """Test double for the files client."""

    def __init__(
        self,
        download_bytes: bytes = b"fake video content",
        file_resource: FakeFileResource | None = None,
    ) -> None:
        self._download_bytes = download_bytes
        self._file_resource = file_resource or FakeFileResource()
        self.uploaded: Any = None

    def upload(self, **kwargs: Any) -> str:
        self.uploaded = kwargs
        return "uploaded-file-handle"

    def get(self, name: str) -> FakeFileResource:
        return self._file_resource

    def download(self, file: Any) -> bytes:
        return self._download_bytes


class FakeGenaiClient:
    """Test double for the Google GenAI client."""

    def __init__(
        self,
        interaction: FakeInteraction | None = None,
        raise_error: Exception | None = None,
        download_bytes: bytes = b"fake video content",
        file_resource: FakeFileResource | None = None,
    ) -> None:
        self.interactions = FakeInteractions(interaction, raise_error)
        self.files = FakeFiles(download_bytes, file_resource)


def _inline_interaction(
    content: bytes = b"fake video content",
    interaction_id: str = "interaction-123",
) -> FakeInteraction:
    """Build an interaction whose video is delivered inline as base64."""
    encoded = base64.b64encode(content).decode("ascii")
    return FakeInteraction(
        interaction_id=interaction_id,
        output_video=FakeOutputVideo(data=encoded),
    )


# ============================================================================
# Inline base64 happy path
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_video_omni_inline_happy_path(tmp_path: Path) -> None:
    """Inline base64 delivery writes a file and returns interaction metadata."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    content = b"real mp4 bytes here"
    client = FakeGenaiClient(interaction=_inline_interaction(content, "abc-1"))

    result = await generate_video_omni(
        client=client,  # type: ignore[arg-type]
        prompt="A cat surfing",
        videos_dir=videos_dir,
    )

    assert result["message"] == "Video generated successfully"
    assert result["model"] == OMNI_MODEL
    assert result["interaction_id"] == "abc-1"
    assert result["aspect_ratio"] == "16:9"
    assert result["duration_seconds"] == 6
    assert "warnings" not in result

    video_url = result["video_url"]
    assert video_url.startswith("file://")
    file_path = Path(video_url[7:])
    assert file_path.exists()
    assert file_path.suffix == ".mp4"
    assert file_path.read_bytes() == content


# ============================================================================
# previous_interaction_id forwarding (multi-turn edit)
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_video_omni_forwards_previous_interaction_id(
    tmp_path: Path,
) -> None:
    """previous_interaction_id is threaded into interactions.create()."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    client = FakeGenaiClient(interaction=_inline_interaction())

    await generate_video_omni(
        client=client,  # type: ignore[arg-type]
        prompt="Make it night time",
        videos_dir=videos_dir,
        previous_interaction_id="prev-999",
    )

    create_kwargs = client.interactions.create_kwargs
    assert create_kwargs is not None
    assert create_kwargs["previous_interaction_id"] == "prev-999"
    assert create_kwargs["input"] == "Make it night time"
    assert create_kwargs["model"] == OMNI_MODEL


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_video_omni_omits_previous_interaction_id_when_absent(
    tmp_path: Path,
) -> None:
    """When not editing, previous_interaction_id is not sent at all."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    client = FakeGenaiClient(interaction=_inline_interaction())

    await generate_video_omni(
        client=client,  # type: ignore[arg-type]
        prompt="Fresh generation",
        videos_dir=videos_dir,
    )

    create_kwargs = client.interactions.create_kwargs
    assert create_kwargs is not None
    assert "previous_interaction_id" not in create_kwargs


# ============================================================================
# Input attachments (images + input video)
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_video_omni_attaches_images(tmp_path: Path) -> None:
    """Input images are forwarded to create()."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    client = FakeGenaiClient(interaction=_inline_interaction())
    images = [b"img-one", b"img-two"]

    await generate_video_omni(
        client=client,  # type: ignore[arg-type]
        prompt="Use these",
        videos_dir=videos_dir,
        image_bytes_list=images,
    )

    create_kwargs = client.interactions.create_kwargs
    assert create_kwargs is not None
    assert create_kwargs["images"] == images


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_video_omni_uploads_input_video(tmp_path: Path) -> None:
    """An input video is uploaded via files.upload and attached to create()."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    client = FakeGenaiClient(interaction=_inline_interaction())

    await generate_video_omni(
        client=client,  # type: ignore[arg-type]
        prompt="Edit this clip",
        videos_dir=videos_dir,
        input_video_bytes=b"input-video-bytes",
    )

    assert client.files.uploaded is not None
    create_kwargs = client.interactions.create_kwargs
    assert create_kwargs is not None
    assert create_kwargs["video"] == "uploaded-file-handle"


# ============================================================================
# URI (Files API) delivery
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_video_omni_uri_delivery(tmp_path: Path) -> None:
    """URI delivery polls the Files API (already ACTIVE) and downloads bytes."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    downloaded = b"downloaded mp4 content"
    interaction = FakeInteraction(
        interaction_id="uri-1",
        output_video=FakeOutputVideo(uri="files/generated-video"),
    )
    client = FakeGenaiClient(
        interaction=interaction,
        download_bytes=downloaded,
        file_resource=FakeFileResource(state="ACTIVE"),
    )

    result = await generate_video_omni(
        client=client,  # type: ignore[arg-type]
        prompt="Big output",
        videos_dir=videos_dir,
    )

    file_path = Path(result["video_url"][7:])
    assert file_path.read_bytes() == downloaded
    assert result["interaction_id"] == "uri-1"


# ============================================================================
# Validation
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_video_omni_invalid_aspect_ratio_raises(
    tmp_path: Path,
) -> None:
    """Unsupported aspect ratios raise instead of being coerced."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    client = FakeGenaiClient(interaction=_inline_interaction())

    with pytest.raises(ValueError, match="Unsupported aspect_ratio"):
        await generate_video_omni(
            client=client,  # type: ignore[arg-type]
            prompt="Bad aspect",
            videos_dir=videos_dir,
            aspect_ratio="4:3",
        )


# ============================================================================
# Duration clamping
# ============================================================================


@pytest.mark.parametrize(
    ("requested", "expected", "expect_warning"),
    [
        (15.0, 10, True),
        (1.0, 3, True),
        (6.0, 6, False),
        (3.0, 3, False),
        (10.0, 10, False),
        (6.7, 7, False),
    ],
)
@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_video_omni_duration_clamping(
    requested: float,
    expected: int,
    expect_warning: bool,
    tmp_path: Path,
) -> None:
    """Duration is clamped to [3, 10] (rounded), with a warning when clamped."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    client = FakeGenaiClient(interaction=_inline_interaction())

    result = await generate_video_omni(
        client=client,  # type: ignore[arg-type]
        prompt="Duration test",
        videos_dir=videos_dir,
        duration_seconds=requested,
    )

    assert result["duration_seconds"] == expected

    # The clamped duration is also forwarded to the API config.
    create_kwargs = client.interactions.create_kwargs
    assert create_kwargs is not None
    assert create_kwargs["config"]["duration_seconds"] == expected

    if expect_warning:
        assert "warnings" in result
        assert any("clamped" in w for w in result["warnings"])
    else:
        assert "warnings" not in result


# ============================================================================
# Timeout
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_video_omni_timeout_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A create() call that exceeds the timeout raises TimeoutError."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    client = FakeGenaiClient(interaction=_inline_interaction())

    async def fake_wait_for(awaitable: Any, timeout: float) -> Any:
        # Close the underlying coroutine to avoid "never awaited" warnings.
        if hasattr(awaitable, "close"):
            awaitable.close()
        raise asyncio.TimeoutError()

    monkeypatch.setattr(asyncio, "wait_for", fake_wait_for)

    with pytest.raises(TimeoutError, match="timed out"):
        await generate_video_omni(
            client=client,  # type: ignore[arg-type]
            prompt="Slow",
            videos_dir=videos_dir,
            timeout_seconds=1,
        )


# ============================================================================
# Missing output handling
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_video_omni_no_output_video_raises(tmp_path: Path) -> None:
    """An interaction with no output_video raises a clear error."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    interaction = FakeInteraction(output_video=None)
    client = FakeGenaiClient(interaction=interaction)

    with pytest.raises(ValueError, match="no output_video"):
        await generate_video_omni(
            client=client,  # type: ignore[arg-type]
            prompt="No output",
            videos_dir=videos_dir,
        )
