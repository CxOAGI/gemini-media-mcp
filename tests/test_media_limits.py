"""Resource limits and output safety for image input and storyboard rendering.

Everything here guards a path where a small input buys a large cost, or where a
detail dropped in passing changes the answer downstream:

  * decoded-pixel budgets on generate_image's input images — the byte caps
    upstream bound the wire, not the bitmap a decoder expands it into;
  * the per-modality token breakdown surviving into the usage dict, which is
    what decides whether a completed call is priced or guessed;
  * bounded frame embeds in the HTML storyboard, where the page holds several
    copies of every inlined image while it is built;
  * the two rendering hazards that only show up under concurrency (shared
    FreeType faces) or a future caller-supplied URL (href schemes).
"""

import base64
import json
import random
import re
import struct
import threading
import zlib
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

import src.image
import src.storyboard
from src.image import _usage_dict, generate_image  # pyright: ignore[reportPrivateUsage]
from src.pricing import actual_image_cost
from src.storyboard import (
    StoryboardFrame,
    _load_font,  # pyright: ignore[reportPrivateUsage]
    render_html,
)
from tests.test_followups_tools import (  # pyright: ignore[reportPrivateUsage]
    _app_ctx,
    _ctx,
)

# ============================================================================
# Helpers
# ============================================================================


def _encode(image: Image.Image, fmt: str = "PNG") -> bytes:
    buffer = BytesIO()
    image.save(buffer, format=fmt)
    image.close()
    return buffer.getvalue()


def _solid(width: int, height: int, color: str = "red", fmt: str = "PNG") -> bytes:
    return _encode(Image.new("RGB", (width, height), color=color), fmt)


def _header_only_png(width: int, height: int) -> bytes:
    """A PNG carrying a valid IHDR of ``width``x``height`` and no pixel data.

    Pillow reports ``size`` straight from the header, so a guard that runs
    before ``load()`` sees the real dimensions here while a guard that runs
    after one only ever sees a decode failure. That difference is the point:
    these 45 bytes are exactly the shape of the attack — a wire payload far too
    small to be capped by any byte limit, claiming a bitmap that is not.
    """

    def chunk(kind: bytes, payload: bytes) -> bytes:
        body = kind + payload
        return (
            struct.pack(">I", len(payload))
            + body
            + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IEND", b"")


def _is_closed(image: Image.Image) -> bool:
    """Whether ``image`` has released its decoded bitmap."""
    try:
        image.getpixel((0, 0))
    except ValueError:
        return True
    except Exception:  # pragma: no cover - any other failure is not closure
        return False
    return False


# ============================================================================
# Test doubles
# ============================================================================


class FakeInlineData:
    """Test double for inline data."""

    def __init__(self, mime_type: str, data: bytes) -> None:
        self.mime_type = mime_type
        self.data = data


class FakePart:
    """Test double for a response part."""

    def __init__(self, inline_data: FakeInlineData | None = None) -> None:
        self.text = None
        self.inline_data = inline_data
        self.thought = False


class FakeContent:
    """Test double for response content."""

    def __init__(self, parts: list[FakePart]) -> None:
        self.parts = parts


class FakeCandidate:
    """Test double for a response candidate."""

    def __init__(self, content: FakeContent) -> None:
        self.content = content


class FakeModalityTokenCount:
    """Test double for one entry of a ``*_tokens_details`` list."""

    def __init__(self, modality: Any, token_count: Any) -> None:
        self.modality = modality
        self.token_count = token_count


class FakeMediaModality:
    """Test double for the SDK's MediaModality enum."""

    def __init__(self, value: str) -> None:
        self.value = value


class FakeUsageMetadata:
    """Test double for a response's usage_metadata."""

    def __init__(
        self,
        prompt_token_count: int | None = None,
        candidates_token_count: int | None = None,
        total_token_count: int | None = None,
        prompt_tokens_details: list[Any] | None = None,
        candidates_tokens_details: list[Any] | None = None,
    ) -> None:
        self.prompt_token_count = prompt_token_count
        self.candidates_token_count = candidates_token_count
        self.total_token_count = total_token_count
        self.prompt_tokens_details = prompt_tokens_details
        self.candidates_tokens_details = candidates_tokens_details


class FakeGeminiResponse:
    """Test double for a generate_content response.

    ``candidates`` is a property so the moment the caller reads it can be
    observed: it is the first thing generate_image touches after the API call
    returns, which is where the decoded inputs are supposed to already be gone.
    """

    def __init__(self, image_bytes: bytes, usage_metadata: Any = None) -> None:
        part = FakePart(FakeInlineData("image/png", image_bytes))
        self._candidates = [FakeCandidate(FakeContent([part]))]
        self.usage_metadata = usage_metadata
        self.watched: list[Image.Image] = []
        self.inputs_closed_on_read: bool | None = None

    @property
    def candidates(self) -> list[FakeCandidate]:
        self.inputs_closed_on_read = all(_is_closed(i) for i in self.watched)
        return self._candidates


class FakeModels:
    """Test double for genai models."""

    def __init__(self, response: FakeGeminiResponse) -> None:
        self._response = response
        self.contents: list[Any] = []

    def generate_content(self, **kwargs: Any) -> FakeGeminiResponse:
        self.contents = list(kwargs.get("contents") or [])
        self._response.watched = [
            item for item in self.contents if isinstance(item, Image.Image)
        ]
        return self._response


class FakeApiClient:
    """Test double for the internal API client."""

    def __init__(self) -> None:
        self.vertexai = False


class FakeGenaiClient:
    """Test double for the Google GenAI client."""

    def __init__(self, response: FakeGeminiResponse) -> None:
        self.models = FakeModels(response)
        self._api_client = FakeApiClient()


def _client(usage_metadata: Any = None) -> FakeGenaiClient:
    return FakeGenaiClient(FakeGeminiResponse(_solid(8, 8), usage_metadata))


# ============================================================================
# Input image pixel budgets
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(10.0)
async def test_oversized_input_image_is_rejected_before_it_is_decoded(
    tmp_path: Path,
) -> None:
    """A bomb must be refused from its header, not after it costs the memory."""
    client = _client()

    with pytest.raises(ValueError) as excinfo:
        await generate_image(
            client=client,  # type: ignore[arg-type]
            prompt="Edit this",
            images_dir=tmp_path,
            image_bytes=_header_only_png(7000, 7000),
        )

    message = str(excinfo.value)
    assert "7000x7000" in message
    assert "49.0MP" in message
    assert "input image" in message.lower()


@pytest.mark.asyncio
@pytest.mark.timeout(10.0)
async def test_rejected_reference_image_is_named_by_position(tmp_path: Path) -> None:
    """The error has to say which reference to fix, not just that one is bad."""
    client = _client()

    with pytest.raises(ValueError) as excinfo:
        await generate_image(
            client=client,  # type: ignore[arg-type]
            prompt="Blend these",
            images_dir=tmp_path,
            reference_images=[
                _solid(64, 64),
                _solid(64, 64),
                _header_only_png(9000, 5000),
            ],
        )

    message = str(excinfo.value)
    assert "Reference image 3" in message
    assert "9000x5000" in message


@pytest.mark.asyncio
@pytest.mark.timeout(20.0)
async def test_large_input_image_is_downscaled_before_the_request(
    tmp_path: Path,
) -> None:
    """A legal-but-large frame is shrunk to the retained budget, not held whole."""
    client = _client()

    await generate_image(
        client=client,  # type: ignore[arg-type]
        prompt="Edit this",
        images_dir=tmp_path,
        image_bytes=_solid(4000, 3000),
    )

    sent = [item for item in client.models.contents if isinstance(item, Image.Image)]
    assert len(sent) == 1
    width, height = sent[0].size
    assert width * height <= src.image._MAX_DECODED_PIXELS
    # Aspect ratio is preserved: a squashed reference would change the render.
    assert width / height == pytest.approx(4000 / 3000, rel=0.01)


@pytest.mark.asyncio
@pytest.mark.timeout(20.0)
async def test_batch_budget_stops_a_run_of_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Per-image budgets alone do not bound a batch; the running total does."""
    monkeypatch.setattr(src.image, "_MAX_TOTAL_DECODED_PIXELS", 20_000)
    client = _client()

    with pytest.raises(ValueError) as excinfo:
        await generate_image(
            client=client,  # type: ignore[arg-type]
            prompt="Blend these",
            images_dir=tmp_path,
            reference_images=[_solid(100, 100) for _ in range(3)],
        )

    assert "Reference image 3" in str(excinfo.value)
    assert "budget" in str(excinfo.value)


def test_documented_reference_count_fits_the_batch_budget() -> None:
    """The batch cap must never reject the workflow the tool advertises.

    Fourteen references plus an edit input, each at the full per-image budget,
    is the largest legal request; if that does not fit, the cap is a bug.
    """
    largest_legal_request = 15 * src.image._MAX_DECODED_PIXELS
    assert largest_legal_request <= src.image._MAX_TOTAL_DECODED_PIXELS


@pytest.mark.asyncio
@pytest.mark.timeout(20.0)
async def test_images_decoded_before_a_rejection_are_released(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refused batch must not leak the frames it had already decoded."""
    opened: list[Image.Image] = []
    real_open = Image.open

    def spy_open(*args: Any, **kwargs: Any) -> Image.Image:
        image = real_open(*args, **kwargs)
        opened.append(image)
        return image

    monkeypatch.setattr(src.image.Image, "open", spy_open)
    client = _client()

    with pytest.raises(ValueError):
        await generate_image(
            client=client,  # type: ignore[arg-type]
            prompt="Blend these",
            images_dir=tmp_path,
            reference_images=[_solid(64, 64), _header_only_png(7000, 7000)],
        )

    assert opened, "expected the guard to have opened the good reference first"
    assert all(_is_closed(image) for image in opened)


@pytest.mark.asyncio
@pytest.mark.timeout(20.0)
async def test_decoded_inputs_are_released_once_the_response_arrives(
    tmp_path: Path,
) -> None:
    """Inputs are dead after the call; holding them spans the whole save path."""
    client = _client()

    await generate_image(
        client=client,  # type: ignore[arg-type]
        prompt="Blend these",
        images_dir=tmp_path,
        reference_images=[_solid(64, 64), _solid(64, 64)],
    )

    response = client.models._response
    assert response.watched, "expected the references to reach generate_content"
    assert response.inputs_closed_on_read is True


# ============================================================================
# Usage metadata -> pricing
# ============================================================================


def _usage_with_two_image_parts() -> FakeUsageMetadata:
    """A thinking model's response: an interim render plus the final one."""
    return FakeUsageMetadata(
        prompt_token_count=1500,
        candidates_token_count=2300,
        total_token_count=3800,
        prompt_tokens_details=[FakeModalityTokenCount(FakeMediaModality("TEXT"), 1500)],
        candidates_tokens_details=[
            FakeModalityTokenCount(FakeMediaModality("IMAGE"), 2240),
            FakeModalityTokenCount(FakeMediaModality("TEXT"), 60),
        ],
    )


def test_usage_dict_carries_the_modality_breakdown() -> None:
    """Pricing reads the breakdown from the dict; dropping it loses the branch."""

    class Response:
        usage_metadata = _usage_with_two_image_parts()

    usage = _usage_dict(Response())

    assert usage is not None
    assert usage["candidates_tokens_details"] == [
        {"modality": "IMAGE", "token_count": 2240},
        {"modality": "TEXT", "token_count": 60},
    ]
    assert usage["prompt_tokens_details"] == [{"modality": "TEXT", "token_count": 1500}]


def test_usage_dict_stays_json_serializable() -> None:
    """The dict goes through json.dumps in the MCP layer; SDK objects cannot."""

    class Response:
        usage_metadata = _usage_with_two_image_parts()

    usage = _usage_dict(Response())
    assert usage is not None

    round_tripped = json.loads(json.dumps(usage))

    assert round_tripped == usage
    # Named explicitly: a breakdown that vanished would round-trip perfectly.
    assert round_tripped["candidates_tokens_details"] == [
        {"modality": "IMAGE", "token_count": 2240},
        {"modality": "TEXT", "token_count": 60},
    ]


def test_metered_cost_uses_the_reported_image_split() -> None:
    """The exact-split branch has to be reachable from a real response.

    Without the breakdown, everything past the table's 1120 image tokens is
    billed at the $3/MTok text rate instead of $60 — a 20x understatement on
    every response that carried more than one image part.
    """

    class Response:
        usage_metadata = _usage_with_two_image_parts()

    usage = _usage_dict(Response())
    cost = actual_image_cost("gemini-3.1-flash-image", usage, "1K", 1)

    assert cost is not None
    assert cost.is_estimate is False
    assert cost.breakdown["output_image_tokens"] == 2240.0
    assert cost.breakdown["output_text_tokens"] == 60.0
    assert cost.usd == pytest.approx((1500 * 0.5 + 2240 * 60 + 60 * 3) / 1e6)


def test_usage_dict_skips_malformed_entries_and_keeps_the_rest() -> None:
    """A detail list the SDK reshapes must degrade, never raise or drop it all."""

    class Response:
        usage_metadata = FakeUsageMetadata(
            prompt_token_count=10,
            candidates_tokens_details=[
                FakeModalityTokenCount(None, 5),
                FakeModalityTokenCount("IMAGE", None),
                "not-a-detail",
                {"modality": "IMAGE", "tokenCount": 1120},
            ],
        )

    usage = _usage_dict(Response())

    assert usage == {
        "prompt_token_count": 10,
        "candidates_tokens_details": [{"modality": "IMAGE", "token_count": 1120}],
    }


# ============================================================================
# Storyboard HTML: embed size
# ============================================================================

_SRC_RE = re.compile(r'<figure><img alt="[^"]*" src="data:([^;]+);base64,([^"]+)"')


def _embedded_frames(document: str) -> list[tuple[str, bytes]]:
    return [
        (mime, base64.b64decode(payload)) for mime, payload in _SRC_RE.findall(document)
    ]


def test_html_embed_is_bounded_for_an_oversized_frame() -> None:
    """A 4K keyframe inlined whole is what made a 24-shot board a 673MB page."""
    frame_bytes = _solid(3840, 2160, color="teal")
    document = render_html(
        [StoryboardFrame(index=1, image_bytes=frame_bytes, prompt="A wide shot")]
    )

    embedded = _embedded_frames(document)
    assert len(embedded) == 1
    _, payload = embedded[0]
    with Image.open(BytesIO(payload)) as shrunk:
        assert max(shrunk.size) <= src.storyboard._HTML_EMBED_MAX_EDGE
        # Still the same picture, just smaller.
        assert shrunk.width / shrunk.height == pytest.approx(3840 / 2160, rel=0.01)


def test_html_embed_passes_a_normal_frame_through_untouched() -> None:
    """Boards that were never the problem must render byte-for-byte as before."""
    frame_bytes = _solid(320, 180)
    document = render_html(
        [StoryboardFrame(index=1, image_bytes=frame_bytes, prompt="A shot")]
    )

    assert _embedded_frames(document) == [("image/png", frame_bytes)]


def test_html_embed_keeps_transparency_when_it_shrinks() -> None:
    """Re-encoding an alpha frame as JPEG would matte it a colour the CSS picks."""
    source = Image.new("RGBA", (2400, 1600), (10, 20, 30, 0))
    document = render_html(
        [StoryboardFrame(index=1, image_bytes=_encode(source), prompt="A shot")]
    )

    mime, payload = _embedded_frames(document)[0]
    assert mime == "image/png"
    with Image.open(BytesIO(payload)) as shrunk:
        assert max(shrunk.size) <= src.storyboard._HTML_EMBED_MAX_EDGE
        assert shrunk.mode == "RGBA"
        pixel = shrunk.getpixel((0, 0))
        assert isinstance(pixel, tuple)
        assert pixel[3] == 0


def test_html_embed_falls_back_to_the_raw_bytes_when_undecodable() -> None:
    """A frame Pillow cannot read still belongs on the page."""
    garbage = b"\x89PNG\r\n\x1a\nnot really a png"
    document = render_html(
        [StoryboardFrame(index=1, image_bytes=garbage, prompt="A shot")]
    )

    assert _embedded_frames(document) == [("image/png", garbage)]


# ============================================================================
# Storyboard response: inline payload size
# ============================================================================


def _noise(width: int, height: int, seed: int) -> bytes:
    """A keyframe that does not compress away.

    ``_solid`` paints one colour, which composites into a sheet of a few
    kilobytes — it would fit the budget whatever the shot count and never show
    the bug. Real keyframes are photographic; noise is their worst end.
    """
    data = random.Random(seed).randbytes(width * height * 3)
    image = Image.frombytes("RGB", (width, height), data)
    try:
        return _encode(image)
    finally:
        image.close()


async def _board(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shots: int
) -> list[Any]:
    """Run generate_storyboard over ``shots`` incompressible keyframes."""
    import src.__main__ as main_mod
    from src.__main__ import generate_storyboard

    async def mock_image_impl(**kwargs: Any) -> dict[str, Any]:
        images_dir: Path = kwargs["images_dir"]
        idx = len(list(images_dir.glob("shot_*.png")))
        path = images_dir / f"shot_{idx}.png"
        path.write_bytes(_noise(320, 180, seed=idx))
        return {
            "message": "ok",
            "image_url": f"file://{path}",
            "prompt": kwargs["prompt"],
            "model": kwargs["model"],
        }

    monkeypatch.setattr(main_mod, "generate_image_impl", mock_image_impl)
    return await generate_storyboard(
        ctx=_ctx(_app_ctx(tmp_path)),
        shots=[
            {"prompt": f"shot {i}", "caption": f"SHOT {i}", "notes": "slow push in"}
            for i in range(shots)
        ],
    )


def _wire_bytes(blocks: list[Any]) -> int:
    """What the response costs on the wire, image blocks base64 and all."""
    total = 0
    for block in blocks:
        data = getattr(block, "data", None)
        total += len(base64.b64encode(data)) if data else len(block.text)
    return total


@pytest.mark.asyncio
@pytest.mark.timeout(120.0)
@pytest.mark.parametrize("shots", [12, 24])
async def test_storyboard_response_stays_under_the_client_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shots: int
) -> None:
    """The reproduction: a board too big to arrive is a board you never see.

    Returning the composited sheet verbatim put a 12-shot board at 1.11MB and a
    24-shot board at 1.25MB once base64'd — past what Claude Desktop accepts, so
    the whole result was dropped after every shot had been billed. Both counts
    are well inside the cap now, on frames that compress worse than real ones.
    """
    blocks = await _board(tmp_path, monkeypatch, shots)

    assert _wire_bytes(blocks) < 1024 * 1024


@pytest.mark.asyncio
@pytest.mark.timeout(120.0)
async def test_storyboard_inline_preview_shrinks_but_the_disk_sheet_does_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The board is bounded inline and full-resolution on disk — both at once.

    The inline copy is the only one with a size problem; shrinking the artifact
    the user actually opens would trade one defect for another.
    """
    blocks = await _board(tmp_path, monkeypatch, 24)
    inline = blocks[0].data
    payload = json.loads(blocks[1].text)

    with Image.open(BytesIO(inline)) as preview:
        assert preview.format == "JPEG"
        preview_w = preview.width
    with Image.open(Path(payload["sheet_url"][7:])) as sheet:
        assert sheet.format == "PNG"
        assert preview_w < sheet.width  # this board genuinely needed shrinking

    assert Path(payload["storyboard_url"][7:]).is_file()


@pytest.mark.asyncio
@pytest.mark.timeout(120.0)
async def test_storyboard_degrades_to_text_when_the_preview_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An undecodable sheet drops the image block, not the whole board.

    Every shot has been paid for by this point, and both artifacts are already
    on disk, so the URLs still have to come back.
    """

    def undecodable(sheet_png: bytes, *, max_bytes: int = 0) -> bytes:
        return b""

    # generate_storyboard imports the helper at call time, so the module
    # attribute is what it resolves.
    monkeypatch.setattr(src.storyboard, "render_sheet_preview", undecodable)
    blocks = await _board(tmp_path, monkeypatch, 2)

    assert len(blocks) == 1
    payload = json.loads(blocks[0].text)
    assert Path(payload["sheet_url"][7:]).is_file()
    assert Path(payload["storyboard_url"][7:]).is_file()


# ============================================================================
# Storyboard HTML: link safety
# ============================================================================


@pytest.mark.parametrize(
    "image_url",
    [
        pytest.param("javascript:alert(1)", id="javascript"),
        pytest.param("JaVaScRiPt:alert(1)", id="mixed_case"),
        pytest.param("java\tscript:alert(1)", id="embedded_control_char"),
        pytest.param(" javascript:alert(1)", id="leading_space"),
        pytest.param("data:text/html,<script>alert(1)</script>", id="data_document"),
        pytest.param("//evil.example/frame.png", id="scheme_relative"),
    ],
)
def test_html_does_not_link_an_unsafe_source_url(image_url: str) -> None:
    """html.escape makes an attribute parse safely; it does not defuse a scheme."""
    document = render_html(
        [
            StoryboardFrame(
                index=1,
                image_bytes=_solid(64, 64),
                prompt="A shot",
                image_url=image_url,
            )
        ]
    )

    assert "<a href=" not in document
    # The reviewer still gets to see where the frame claims to come from.
    assert "alert(1)" in document or "evil.example" in document


def test_html_links_a_server_generated_file_url() -> None:
    """The real URLs the server produces must keep working."""
    document = render_html(
        [
            StoryboardFrame(
                index=1,
                image_bytes=_solid(64, 64),
                prompt="A shot",
                image_url="file:///tmp/frame-1.png",
            )
        ]
    )

    assert '<a href="file:///tmp/frame-1.png">' in document


# ============================================================================
# Fonts under concurrency
# ============================================================================


def test_fonts_are_not_shared_between_threads() -> None:
    """Pillow's FreeTypeFont is not thread-safe, and boards render on workers."""
    _load_font.cache_clear()
    results: dict[str, Any] = {}
    barrier = threading.Barrier(2)

    def render(name: str) -> None:
        barrier.wait(timeout=5)
        results[name] = (_load_font(18, bold=True), _load_font(18, bold=True))

    threads = [threading.Thread(target=render, args=(name,)) for name in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert set(results) == {"a", "b"}
    # Memoized within a thread...
    assert results["a"][0] is results["a"][1]
    # ...and never handed to a second one.
    assert results["a"][0] is not results["b"][0]
