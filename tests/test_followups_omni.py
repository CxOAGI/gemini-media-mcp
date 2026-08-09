"""Follow-up regression tests for src/omni.py.

Covers three delivered-video / MIME defects: the uncapped download of the
delivered file, the MOV/AVI MIME labels that deviated from the SDK's wire
contract, and the empty-download guard that checked None but not b"".
"""

from pathlib import Path
from typing import Any

import pytest

from src.omni import _MAX_DELIVERED_VIDEO_BYTES, generate_video_omni


# ============================================================================
# Test doubles (a delivered-via-uri interaction that downloads through files)
# ============================================================================


class FakeOutputVideo:
    def __init__(self, uri: str | None = None) -> None:
        self.data = None
        self.uri = uri


class FakeInteraction:
    def __init__(self, id: str, output_video: FakeOutputVideo) -> None:
        self.id = id
        self.status = "completed"
        self.output_video = output_video
        self.steps: list[Any] = []
        self.error = None


class FakeInteractions:
    def __init__(self, create_result: FakeInteraction) -> None:
        self._create_result = create_result

    def create(self, **kwargs: Any) -> FakeInteraction:
        return self._create_result

    def get(self, id: str, **kwargs: Any) -> FakeInteraction:  # noqa: A002
        raise AssertionError("create already returned completed; no poll expected")


class FakeFile:
    """A resolved file resource; ``size_bytes`` is what files.get advertises."""

    def __init__(self, size_bytes: int | None = None) -> None:
        self.size_bytes = size_bytes


class FakeFiles:
    def __init__(self, download_bytes: bytes, size_bytes: int | None = None) -> None:
        self._download_bytes = download_bytes
        self._size_bytes = size_bytes

    def get(self, **kwargs: Any) -> FakeFile:
        return FakeFile(size_bytes=self._size_bytes)

    def download(self, **kwargs: Any) -> bytes:
        return self._download_bytes


class FakeGenaiClient:
    def __init__(self, interactions: FakeInteractions, files: FakeFiles) -> None:
        self.interactions = interactions
        self.files = files


def _uri_delivery_client(download_bytes: bytes, size_bytes: int | None = None) -> Any:
    interaction = FakeInteraction(
        id="int-1", output_video=FakeOutputVideo(uri="files/abc123")
    )
    return FakeGenaiClient(
        FakeInteractions(create_result=interaction),
        FakeFiles(download_bytes=download_bytes, size_bytes=size_bytes),
    )


# ============================================================================
# (1) Size cap on the delivered-video download
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_oversize_downloaded_body_is_rejected(tmp_path: Path) -> None:
    """A downloaded body over the cap raises instead of being written.

    Before the fix the delivered-file path buffered files.download's body with
    no cap — unlike every other fetch in the server — so an oversize clip was
    written and reported as success.
    """
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    oversize = b"\x00" * (_MAX_DELIVERED_VIDEO_BYTES + 1)
    client = _uri_delivery_client(download_bytes=oversize)

    with pytest.raises(ValueError, match="exceeds cap"):
        await generate_video_omni(
            client=client,  # type: ignore[arg-type]
            prompt="p",
            videos_dir=videos_dir,
        )
    # Nothing partial left on disk.
    assert not list(videos_dir.iterdir())


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_oversize_advertised_size_rejected_before_download(
    tmp_path: Path,
) -> None:
    """When files.get advertises a size over the cap, the download is refused
    before the body is fetched (fail fast, not after buffering)."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    # Tiny body, but the resource claims it is huge — the pre-check must fire.
    client = _uri_delivery_client(
        download_bytes=b"tiny", size_bytes=_MAX_DELIVERED_VIDEO_BYTES + 1
    )

    with pytest.raises(ValueError, match="exceeds cap"):
        await generate_video_omni(
            client=client,  # type: ignore[arg-type]
            prompt="p",
            videos_dir=videos_dir,
        )
    assert not list(videos_dir.iterdir())


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_download_at_cap_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A body of EXACTLY the cap is accepted — the bound is inclusive (len >
    cap, not >=), so the cap must not reject legitimate output.

    The cap is shrunk here so the boundary is exercised for real rather than
    with a token payload well under it (which proves nothing about the edge):
    cap bytes must pass, cap+1 must raise.
    """
    monkeypatch.setattr("src.omni._MAX_DELIVERED_VIDEO_BYTES", 64)
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    at_cap = b"\x01" * 64
    result = await generate_video_omni(
        client=_uri_delivery_client(download_bytes=at_cap),  # type: ignore[arg-type]
        prompt="p",
        videos_dir=videos_dir,
    )
    assert Path(result["video_url"][7:]).read_bytes() == at_cap

    # One byte over the (shrunk) cap must be refused, proving the boundary is
    # where the docstring says it is.
    with pytest.raises(ValueError, match="exceeds cap"):
        await generate_video_omni(
            client=_uri_delivery_client(download_bytes=b"\x01" * 65),  # type: ignore[arg-type]
            prompt="p",
            videos_dir=tmp_path / "over",
        )


# ============================================================================
# (2) MOV / AVI MIME labels match the SDK wire contract
# ============================================================================


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        pytest.param(
            b"\x00\x00\x00\x18ftypqt  \x00\x00\x00\x00", "video/mov", id="mov"
        ),
        pytest.param(b"RIFF\x00\x00\x00\x00AVI LIST", "video/avi", id="avi"),
    ],
)
def test_mov_avi_use_sdk_spellings(data: bytes, expected: str) -> None:
    """MOV/AVI use the SDK's VideoContentMimeType spellings ('video/mov',
    'video/avi'), not the RFC labels ('video/quicktime', 'video/x-msvideo')
    which are absent from the literal set and ride as UnrecognizedStr."""
    from src.omni import _detect_video_mime

    assert _detect_video_mime(data) == expected


def test_mov_avi_labels_are_in_the_sdk_literal_set() -> None:
    """Pin the labels against the installed SDK's actual literal set, so a
    future divergence in either the sniffer or the SDK is caught here."""
    from typing import get_args

    from google.genai._gaos.types.interactions.videocontent import (
        VideoContentMimeType,
    )

    from src.omni import _detect_video_mime

    # get_args(Union[Literal[...], UnrecognizedStr]) → (Literal[...], str);
    # the literal's own args are the enumerated wire strings.
    literal = get_args(VideoContentMimeType)[0]
    allowed = set(get_args(literal))

    mov = _detect_video_mime(b"\x00\x00\x00\x18ftypqt  \x00\x00\x00\x00")
    avi = _detect_video_mime(b"RIFF\x00\x00\x00\x00AVI LIST")
    assert mov in allowed
    assert avi in allowed


# ============================================================================
# (3) Empty-download guard catches b"", not just None
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_empty_downloaded_body_raises(tmp_path: Path) -> None:
    """A zero-length download raises instead of writing a 0-byte .mp4.

    files.download returns bytes, so an empty body is b"" not None; the old
    `if data is None` guard let it through, wrote an empty file, and reported
    "Video generated successfully".
    """
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    client = _uri_delivery_client(download_bytes=b"")

    with pytest.raises(ValueError, match="empty"):
        await generate_video_omni(
            client=client,  # type: ignore[arg-type]
            prompt="p",
            videos_dir=videos_dir,
        )
    assert not list(videos_dir.iterdir())


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_empty_inline_data_falls_through_to_uri(tmp_path: Path) -> None:
    """The inline branch is already truthiness-based: an empty inline payload
    must not be treated as the video — it falls through to the uri path, which
    the tightened download guard then resolves normally."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    class EmptyInlineOutput:
        def __init__(self) -> None:
            self.data = ""  # empty inline payload
            self.uri = "files/abc123"

    interaction = FakeInteraction(id="int-1", output_video=EmptyInlineOutput())  # type: ignore[arg-type]
    client = FakeGenaiClient(
        FakeInteractions(create_result=interaction),
        FakeFiles(download_bytes=b"REAL"),
    )

    result = await generate_video_omni(
        client=client,  # type: ignore[arg-type]
        prompt="p",
        videos_dir=videos_dir,
    )
    assert Path(result["video_url"][7:]).read_bytes() == b"REAL"
