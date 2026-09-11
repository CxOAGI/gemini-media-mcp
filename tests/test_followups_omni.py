"""Follow-up regression tests for src/omni.py.

Covers three delivered-video / MIME defects: the uncapped download of the
delivered file, the MOV/AVI MIME labels that deviated from the SDK's wire
contract, and the empty-download guard that checked None but not b"".
"""

from pathlib import Path
from typing import Any

import pytest

from src.omni import (
    _MAX_DELIVERED_VIDEO_BYTES,
    OMNI_PREVIEW_MODEL,
    generate_video_omni,
)


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
            model=OMNI_PREVIEW_MODEL,
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
            model=OMNI_PREVIEW_MODEL,
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
    # The preview model is the one whose cap is this constant; 1.1 can emit
    # 4K and 40s chains and gets a larger one, so patching this and then
    # calling the default model would exercise neither boundary.
    monkeypatch.setattr("src.omni._MAX_DELIVERED_VIDEO_BYTES", 64)
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    at_cap = b"\x01" * 64
    result = await generate_video_omni(
        client=_uri_delivery_client(download_bytes=at_cap),  # type: ignore[arg-type]
        model=OMNI_PREVIEW_MODEL,
        prompt="p",
        videos_dir=videos_dir,
    )
    assert Path(result["video_url"][7:]).read_bytes() == at_cap

    # One byte over the (shrunk) cap must be refused, proving the boundary is
    # where the docstring says it is.
    with pytest.raises(ValueError, match="exceeds cap"):
        await generate_video_omni(
            client=_uri_delivery_client(download_bytes=b"\x01" * 65),  # type: ignore[arg-type]
            model=OMNI_PREVIEW_MODEL,
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
            model=OMNI_PREVIEW_MODEL,
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
        model=OMNI_PREVIEW_MODEL,
        client=client,  # type: ignore[arg-type]
        prompt="p",
        videos_dir=videos_dir,
    )
    assert Path(result["video_url"][7:]).read_bytes() == b"REAL"


# ============================================================================
# A returned model ID must round-trip: an agent that feeds result["model"]
# from a Gemini-API render back into the next call must not hit a rejection.
# ============================================================================


def test_every_returnable_video_model_is_also_accepted() -> None:
    """On the Gemini API a render reports a `-preview` model; the type it is
    validated against rejected those, so chaining that ID into loop_extend or
    generate_video failed. Every ID a run can return must be an accepted
    input."""
    import typing

    from src.video import TranslatedVideoModel, VideoModel, _GEMINI_API_MODEL_IDS

    # The tools accept the union: the canonical catalogue the planner ranks
    # over, plus the translated spellings a Gemini-API render reports back.
    accepted = set(typing.get_args(VideoModel)) | set(
        typing.get_args(TranslatedVideoModel)
    )
    returnable: set[str] = set()
    for canonical in (
        "veo-3.1-generate-001",
        "veo-3.1-fast-generate-001",
        "veo-3.1-lite-generate-preview",
    ):
        returnable.add(canonical)  # Vertex reports the canonical id
        returnable.add(_GEMINI_API_MODEL_IDS.get(canonical, canonical))  # Gemini API
    assert returnable <= accepted, f"not accepted: {returnable - accepted}"

    # But the planner's candidate catalogue must stay canonical-only — folding
    # the aliases into VideoModel made it rank each model twice.
    assert set(typing.get_args(VideoModel)).isdisjoint(
        typing.get_args(TranslatedVideoModel)
    )


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_a_returned_preview_id_normalizes_before_the_backend_translation(
    tmp_path: Path,
) -> None:
    """A `-preview` id passed back must resolve to canonical first: raw on
    Vertex it would 404, and the reported model must be identical to what a
    caller who named the canonical id would have gotten."""
    from tests.test_video import (
        FakeGeneratedVideo,
        FakeGenaiClient,
        FakeOperation,
        FakeVideoObject,
        FakeVideoResult,
    )
    from src.video import generate_video

    async def report(vertexai: bool, model_in: str) -> str:
        op = FakeOperation(
            done=True,
            result=FakeVideoResult(
                [FakeGeneratedVideo(FakeVideoObject(video_bytes=b"v"))]
            ),
        )
        client = FakeGenaiClient(operation=op, vertexai=vertexai)
        out = tmp_path / f"{vertexai}-{model_in}"
        out.mkdir()
        result = await generate_video(
            client=client,  # type: ignore[arg-type]
            prompt="p",
            videos_dir=out,
            model=model_in,  # type: ignore[arg-type]
        )
        return result["model"]

    # On Vertex a fed-back preview id must not be sent raw (it would 404) — it
    # is normalized, so the reported model is the canonical one.
    assert await report(True, "veo-3.1-generate-preview") == "veo-3.1-generate-001"
    # On the Gemini API, naming the canonical id or feeding back the preview id
    # both report the same served spelling.
    assert await report(False, "veo-3.1-generate-001") == "veo-3.1-generate-preview"
    assert await report(False, "veo-3.1-generate-preview") == "veo-3.1-generate-preview"


def test_the_video_tool_schemas_accept_the_translated_ids() -> None:
    """The round-trip only works if the TOOL signatures accept the alias, not
    just the impl — a signature left as bare VideoModel would have MCP reject
    a fed-back -preview id before the tool ran. Guards the __main__ wiring."""
    import asyncio

    import src.__main__ as main_mod

    tools = {t.name: t for t in asyncio.run(main_mod.mcp.list_tools())}
    aliases = {"veo-3.1-generate-preview", "veo-3.1-fast-generate-preview"}
    for name in (
        "generate_video",
        "generate_transition",
        "generate_bridge",
        "generate_clip",
        "loop_extend",
    ):
        prop = tools[name].inputSchema["properties"]["model"]
        accepted = set(prop.get("enum") or [])
        for branch in prop.get("anyOf", []):
            accepted |= set(branch.get("enum") or [])
        assert aliases <= accepted, (
            f"{name} rejects the translated ids: {aliases - accepted}"
        )
