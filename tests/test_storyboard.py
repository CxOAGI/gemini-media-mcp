"""Tests for storyboard.py contact-sheet and HTML rendering."""

import base64
import re
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image, ImageFont

import src.storyboard
from src.storyboard import (
    StoryboardFrame,
    format_timecode,
    render_contact_sheet,
    render_html,
    write_storyboard,
)
from src.storyboard import (
    _default_columns,  # pyright: ignore[reportPrivateUsage]
)
from src.storyboard import (
    _load_font,  # pyright: ignore[reportPrivateUsage]
)
from src.storyboard import (
    _sniff_mime,  # pyright: ignore[reportPrivateUsage]
)
from src.storyboard import (
    _wrap_text,  # pyright: ignore[reportPrivateUsage]
)

_OUTER_PAD = src.storyboard._OUTER_PAD  # pyright: ignore[reportPrivateUsage]
_GUTTER = src.storyboard._GUTTER  # pyright: ignore[reportPrivateUsage]


@pytest.fixture(autouse=True)
def _clear_font_cache() -> Iterator[None]:
    """Drop the memoized fonts around every test.

    _load_font is lru_cached, so a test that patches font discovery would
    otherwise leak its fallback font into unrelated tests (and vice versa).
    """
    _load_font.cache_clear()
    yield
    _load_font.cache_clear()


# ============================================================================
# Helpers
# ============================================================================


def make_image(
    width: int = 320,
    height: int = 180,
    color: tuple[int, ...] | int = (90, 120, 200),
    mode: str = "RGB",
    fmt: str = "PNG",
) -> bytes:
    """Encode a solid test image."""
    image = Image.new(mode, (width, height), color)  # pyright: ignore[reportArgumentType]
    buffer = BytesIO()
    image.save(buffer, format=fmt)
    image.close()
    return buffer.getvalue()


def make_frames(count: int, *, notes: bool = False) -> list[StoryboardFrame]:
    """Build ``count`` well-formed frames."""
    return [
        StoryboardFrame(
            index=i + 1,
            image_bytes=make_image(),
            prompt=f"Shot {i + 1}: a wide establishing shot of somewhere moody.",
            caption=f"SHOT {i + 1}",
            duration_seconds=4.0,
            notes="Handheld, 35mm" if notes else None,
        )
        for i in range(count)
    ]


def open_png(data: bytes) -> tuple[int, int]:
    """Assert ``data`` is a real PNG and return its size."""
    with Image.open(BytesIO(data)) as image:
        image.load()
        assert image.format == "PNG"
        return image.size


# ============================================================================
# Input validation
# ============================================================================


def test_render_contact_sheet_rejects_empty_frames() -> None:
    """An empty board is a caller error, not an empty image."""
    with pytest.raises(ValueError, match="must not be empty"):
        render_contact_sheet([])


def test_render_html_rejects_empty_frames() -> None:
    """render_html raises the same clear error."""
    with pytest.raises(ValueError, match="must not be empty"):
        render_html([])


def test_write_storyboard_rejects_empty_frames(tmp_path: Path) -> None:
    """write_storyboard fails before creating any files."""
    with pytest.raises(ValueError, match="must not be empty"):
        write_storyboard([], tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_render_contact_sheet_rejects_bad_geometry() -> None:
    """Non-positive size limits are rejected up front."""
    frames = make_frames(1)
    with pytest.raises(ValueError, match="must be positive"):
        render_contact_sheet(frames, panel_width=0)
    with pytest.raises(ValueError, match="must be positive"):
        render_contact_sheet(frames, max_sheet_width=-10)


# ============================================================================
# Contact sheet
# ============================================================================


def test_single_frame_sheet_is_valid_png_of_expected_width() -> None:
    """One frame renders a one-column sheet sized from the panel width."""
    png = render_contact_sheet(make_frames(1), panel_width=300)
    width, height = open_png(png)
    assert width == 2 * _OUTER_PAD + 300
    assert height > 300  # image area + text block + header + footer


def test_grid_width_matches_column_count() -> None:
    """Explicit columns drive the sheet width deterministically."""
    png = render_contact_sheet(make_frames(6), columns=3, panel_width=200)
    width, _ = open_png(png)
    assert width == 2 * _OUTER_PAD + 3 * 200 + 2 * _GUTTER


def test_two_frames_render_side_by_side() -> None:
    """Two shots sit on one row, so the sheet is twice as wide as one shot."""
    one_w, _ = open_png(render_contact_sheet(make_frames(1), panel_width=240))
    two_w, _ = open_png(render_contact_sheet(make_frames(2), panel_width=240))
    assert two_w == one_w + 240 + _GUTTER


def test_many_frames_wrap_onto_multiple_rows() -> None:
    """A 24-shot board stays inside the width cap and grows downward."""
    png = render_contact_sheet(make_frames(24), max_sheet_width=1400)
    width, height = open_png(png)
    assert width <= 1400
    assert height > width


def _vertical_frames(count: int) -> list[StoryboardFrame]:
    """Frames with 9:16 source images, the tall worst case for sheet height."""
    return [
        StoryboardFrame(
            index=i + 1,
            image_bytes=make_image(540, 960),
            prompt="A vertical shot for a social reel.",
            duration_seconds=3.0,
        )
        for i in range(count)
    ]


def test_sheet_respects_max_sheet_height() -> None:
    """Tall vertical boards shrink their panels instead of running away."""
    frames = _vertical_frames(20)
    _, tall = open_png(render_contact_sheet(frames, max_sheet_height=100_000))
    _, capped = open_png(render_contact_sheet(frames, max_sheet_height=2400))
    assert tall > 2400
    assert capped <= 2400


def test_max_sheet_height_never_shrinks_below_legibility() -> None:
    """An impossible cap yields a legible sheet rather than a postage stamp."""
    frames = _vertical_frames(20)
    width, _ = open_png(render_contact_sheet(frames, max_sheet_height=200))
    # Five columns at the 200px panel floor.
    assert width == 2 * _OUTER_PAD + 5 * 200 + 4 * _GUTTER


def test_default_columns_prefers_balanced_rows() -> None:
    """Column derivation avoids orphan panels and caps at five."""
    assert _default_columns(1) == 1
    assert _default_columns(2) == 2
    assert _default_columns(3) == 3
    assert _default_columns(4) == 4
    assert _default_columns(5) == 5  # one strip beats 4 + 1 orphan
    assert _default_columns(6) == 3  # 3 + 3
    assert _default_columns(9) == 3  # 3 x 3
    assert _default_columns(22) == 5
    assert _default_columns(40) == 5


def test_failed_frame_still_renders_a_panel() -> None:
    """A failed shot gets a placeholder, not a missing panel or an exception."""
    frames = make_frames(3)
    frames[1] = StoryboardFrame(
        index=2,
        image_bytes=None,
        prompt="Close-up on the package changing hands.",
        duration_seconds=4.0,
        error="Safety filter blocked the prompt (PROHIBITED_CONTENT).",
    )
    png = render_contact_sheet(frames)
    ok_png = render_contact_sheet(make_frames(3))
    # Same grid geometry: the hole is filled, not skipped.
    assert open_png(png) == open_png(ok_png)
    assert png != ok_png


def test_all_frames_failed_still_renders() -> None:
    """A board where nothing generated is still a reviewable artifact."""
    frames = [
        StoryboardFrame(index=i + 1, image_bytes=None, prompt="p", error="boom")
        for i in range(3)
    ]
    open_png(render_contact_sheet(frames, title="Everything failed"))


def test_undecodable_bytes_are_treated_as_a_failed_shot() -> None:
    """Garbage image bytes degrade to a placeholder rather than raising."""
    frames = make_frames(2)
    frames[0] = StoryboardFrame(
        index=1, image_bytes=b"not an image at all", prompt="broken"
    )
    open_png(render_contact_sheet(frames))


@pytest.mark.parametrize(
    ("mode", "color"),
    [
        ("RGBA", (200, 30, 30, 128)),
        ("LA", (128, 200)),
        ("P", 5),
        ("L", 128),
        ("CMYK", (10, 20, 30, 40)),
    ],
)
def test_non_rgb_source_modes_are_converted(
    mode: str, color: tuple[int, ...] | int
) -> None:
    """RGBA / palette / grayscale / CMYK sources composite safely."""
    data = make_image(
        120, 90, color, mode=mode, fmt="TIFF" if mode == "CMYK" else "PNG"
    )
    frames = [StoryboardFrame(index=1, image_bytes=data, prompt="mode test")]
    open_png(render_contact_sheet(frames))


@pytest.mark.parametrize(("width", "height"), [(1, 1), (4, 3000), (4000, 2250)])
def test_extreme_source_sizes_are_letterboxed(width: int, height: int) -> None:
    """Tiny and huge frames both fit the fixed panel box."""
    frames = [
        StoryboardFrame(index=1, image_bytes=make_image(width, height), prompt="size")
    ]
    open_png(render_contact_sheet(frames, panel_width=240))


def test_odd_aspect_source_is_letterboxed_not_stretched() -> None:
    """A square frame in a 16:9 panel keeps its shape and gets matte bars."""
    red = (220, 30, 30)
    frames = [
        StoryboardFrame(index=1, image_bytes=make_image(640, 360), prompt="16:9"),
        StoryboardFrame(index=2, image_bytes=make_image(400, 400, red), prompt="1:1"),
    ]
    panel_w = 300
    png = render_contact_sheet(frames, panel_width=panel_w, columns=2)

    # The panel box is 16:9 (taken from the first frame), so the square is
    # scaled to the panel height and stays square. If it were stretched to the
    # panel width there would be panel_w * image_h red pixels instead.
    image_h = round(panel_w / (16 / 9))
    with Image.open(BytesIO(png)) as sheet:
        rgb = sheet.convert("RGB")
        try:
            counts = rgb.getcolors(maxcolors=1 << 20) or []
            red_pixels = sum(count for count, color in counts if color == red)
        finally:
            rgb.close()

    square = image_h * image_h
    stretched = panel_w * image_h
    assert 0.85 * square <= red_pixels <= square
    assert red_pixels < 0.8 * stretched


def test_light_and_dark_themes_differ() -> None:
    """The theme argument actually changes the rendered pixels."""
    frames = make_frames(2)
    assert render_contact_sheet(frames, theme="light") != render_contact_sheet(
        frames, theme="dark"
    )


def test_contact_sheet_is_deterministic() -> None:
    """No timestamps or randomness leak into the PNG."""
    frames = make_frames(4, notes=True)
    assert render_contact_sheet(frames, title="Repeatable") == render_contact_sheet(
        frames, title="Repeatable"
    )


def test_notes_add_a_reserved_text_row() -> None:
    """Panels grow to fit notes only when some shot has them."""
    _, plain_h = open_png(render_contact_sheet(make_frames(2)))
    _, noted_h = open_png(render_contact_sheet(make_frames(2, notes=True)))
    assert noted_h > plain_h


# ============================================================================
# Fonts
# ============================================================================


def test_font_fallback_when_no_truetype_paths_resolve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rendering must survive a host with no installed fonts."""
    monkeypatch.setattr(src.storyboard, "_REGULAR_FONT_PATHS", ())
    monkeypatch.setattr(src.storyboard, "_BOLD_FONT_PATHS", ())
    _load_font.cache_clear()

    open_png(render_contact_sheet(make_frames(2), title="No fonts here"))


def _simulate_no_freetype(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make font discovery behave like a Pillow build without FreeType.

    No TrueType path resolves, and ``load_default(size=...)`` raises the way it
    does when FreeType is missing — leaving only the bitmap font.
    """
    monkeypatch.setattr(src.storyboard, "_REGULAR_FONT_PATHS", ())
    monkeypatch.setattr(src.storyboard, "_BOLD_FONT_PATHS", ())

    def no_scalable_default(size: float | None = None) -> ImageFont.ImageFont:
        if size is not None:
            raise TypeError("this Pillow build has no FreeType support")
        return ImageFont.load_default_imagefont()

    # ImageFont is the same module object the renderer imported, so patching it
    # here removes the scalable default from the renderer's view too.
    monkeypatch.setattr(ImageFont, "load_default", no_scalable_default)
    _load_font.cache_clear()


def test_font_fallback_to_bitmap_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """With neither TrueType files nor FreeType, the bitmap font is used."""
    _simulate_no_freetype(monkeypatch)

    font = _load_font(18, bold=True)
    assert isinstance(font, ImageFont.ImageFont)
    assert not isinstance(font, ImageFont.FreeTypeFont)
    # And the full render still produces a valid sheet.
    open_png(render_contact_sheet(make_frames(3), title="Bitmap only"))


def test_bitmap_fallback_renders_html_and_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """write_storyboard survives the fontless path end to end."""
    _simulate_no_freetype(monkeypatch)
    result = write_storyboard(make_frames(2), tmp_path, title="Fontless")
    assert Path(result["sheet_path"]).is_file()


def test_truetype_font_is_preferred_when_available() -> None:
    """On a normal host a real face resolves (guarded: skip if none exist)."""
    font = _load_font(20)
    if not isinstance(font, ImageFont.FreeTypeFont):
        pytest.skip("no TrueType fonts installed on this host")
    assert font.size == 20


def test_bad_font_paths_are_skipped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A path that exists but is not a font does not abort discovery."""
    junk = tmp_path / "not-a-font.ttf"
    junk.write_bytes(b"definitely not a font")
    monkeypatch.setattr(src.storyboard, "_REGULAR_FONT_PATHS", (str(junk),))
    monkeypatch.setattr(src.storyboard, "_BOLD_FONT_PATHS", (str(junk),))
    _load_font.cache_clear()

    open_png(render_contact_sheet(make_frames(1)))


# ============================================================================
# Text wrapping
# ============================================================================


def test_wrap_text_wraps_to_the_measured_width() -> None:
    """Wrapped lines never exceed the available width."""
    font = _load_font(14)
    text = "The quick brown fox jumps over the lazy dog " * 4
    lines = _wrap_text(text, font, 180, 6)
    assert len(lines) <= 6
    assert all(font.getlength(line) <= 180 for line in lines)


def test_wrap_text_truncates_with_an_ellipsis() -> None:
    """Overflowing text is cut off with an ellipsis, not spilled."""
    font = _load_font(14)
    text = "word " * 200
    lines = _wrap_text(text, font, 160, 3)
    assert len(lines) == 3
    assert lines[-1].endswith(("…", "..."))
    assert font.getlength(lines[-1]) <= 160


def test_wrap_text_breaks_a_single_overlong_word() -> None:
    """A token wider than the panel is broken mid-word instead of overflowing."""
    font = _load_font(14)
    lines = _wrap_text("A" * 400, font, 120, 4)
    assert len(lines) == 4
    assert all(font.getlength(line) <= 120 for line in lines)
    assert lines[-1].endswith(("…", "..."))


def test_wrap_text_keeps_short_text_intact() -> None:
    """Text that fits is returned unchanged, with no ellipsis."""
    font = _load_font(14)
    assert _wrap_text("Short line", font, 400, 3) == ["Short line"]


def test_wrap_text_handles_degenerate_inputs() -> None:
    """Empty text and zero space produce no lines rather than raising."""
    font = _load_font(14)
    assert _wrap_text("", font, 200, 3) == []
    assert _wrap_text("text", font, 0, 3) == []
    assert _wrap_text("text", font, 200, 0) == []


def test_bitmap_font_uses_ascii_ellipsis(monkeypatch: pytest.MonkeyPatch) -> None:
    """The bitmap fallback has no U+2026 glyph, so it gets three dots."""
    _simulate_no_freetype(monkeypatch)

    font = _load_font(12)
    lines = _wrap_text("word " * 200, font, 100, 2)
    assert lines[-1].endswith("...")


# ============================================================================
# HTML
# ============================================================================


def test_html_is_a_complete_document() -> None:
    """render_html returns a standalone page, not a fragment."""
    doc = render_html(make_frames(2), title="My Board")
    assert doc.startswith("<!doctype html>")
    assert doc.rstrip().endswith("</html>")
    assert "<title>My Board</title>" in doc


def test_html_embeds_images_as_data_uris() -> None:
    """Frames are inlined so the page works offline."""
    png = make_image(64, 36, (10, 200, 90))
    frames = [StoryboardFrame(index=1, image_bytes=png, prompt="p")]
    doc = render_html(frames)
    encoded = base64.b64encode(png).decode("ascii")
    assert f'src="data:image/png;base64,{encoded}"' in doc


def test_html_makes_no_external_requests() -> None:
    """No CDN, no web fonts, no remote images — everything is inline."""
    frames = make_frames(3, notes=True)
    frames[0].image_url = "file:///data/images/shot-1.png"
    doc = render_html(frames, title="Offline", subtitle="sub")

    # Everything the browser loads automatically must be inline data.
    for value in re.findall(r'src="([^"]*)"', doc):
        assert value.startswith("data:"), f"external src: {value}"
    # Links may point at the local full-res frame, but never off-machine.
    for value in re.findall(r'href="([^"]*)"', doc):
        assert value.startswith(("data:", "file://")), f"external href: {value}"
    assert "http://" not in doc and "https://" not in doc
    assert "@import" not in doc and "url(" not in doc


def test_html_inlines_css_and_js() -> None:
    """Style and script are embedded, with no external link/script src."""
    doc = render_html(make_frames(1))
    assert "<style>" in doc and "</style>" in doc
    assert "<script>" in doc and "</script>" in doc
    assert "<link" not in doc
    assert "<script src=" not in doc


def test_html_supports_light_and_dark() -> None:
    """The page follows the reader's colour scheme."""
    doc = render_html(make_frames(1))
    assert "prefers-color-scheme: dark" in doc
    assert "color-scheme: light dark" in doc


def test_html_is_responsive() -> None:
    """Narrow windows get a single column and images never overflow."""
    doc = render_html(make_frames(2))
    assert 'name="viewport"' in doc
    assert "max-width: 100%" in doc
    assert "@media (max-width: 760px)" in doc


def test_html_escapes_hostile_prompt_text() -> None:
    """Prompts are untrusted input and must never inject markup."""
    hostile = '<script>alert("xss")</script> & <img src=x onerror="alert(1)">'
    frames = [
        StoryboardFrame(
            index=1,
            image_bytes=make_image(),
            prompt=hostile,
            caption=hostile,
            notes=hostile,
            error=hostile,
            image_url='file:///tmp/"onmouseover="alert(1)".png',
        )
    ]
    doc = render_html(frames, title=hostile, subtitle=hostile)

    assert "<script>alert(" not in doc
    assert "<img src=x" not in doc
    assert '"onmouseover="' not in doc
    assert "&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;" in doc
    assert "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;" in doc
    # The only script element is our own inline one, and the only img tag is
    # the frame itself — nothing was injected.
    assert doc.count("<script") == 1
    assert doc.count("<img") == 1


def test_html_escapes_hostile_text_on_failed_frames() -> None:
    """The failure placeholder escapes its error text too."""
    frames = [
        StoryboardFrame(
            index=1,
            image_bytes=None,
            prompt="p",
            error='<script>alert("boom")</script>',
        )
    ]
    doc = render_html(frames)
    assert "<script>alert(" not in doc
    assert "&lt;script&gt;" in doc


def test_html_shows_running_timecode() -> None:
    """Timecodes accumulate across shots."""
    frames = [
        StoryboardFrame(
            index=1, image_bytes=make_image(), prompt="a", duration_seconds=8
        ),
        StoryboardFrame(
            index=2, image_bytes=make_image(), prompt="b", duration_seconds=4
        ),
        StoryboardFrame(
            index=3, image_bytes=make_image(), prompt="c", duration_seconds=60
        ),
    ]
    doc = render_html(frames)
    assert "00:00 – 00:08" in doc
    assert "00:08 – 00:12" in doc
    assert "00:12 – 01:12" in doc
    assert "01:12 total runtime" in doc


def test_html_reports_failures_and_shot_counts() -> None:
    """The header and footer summarise the board."""
    frames = make_frames(3)
    frames[2] = StoryboardFrame(index=3, image_bytes=None, prompt="p", error="nope")
    doc = render_html(frames)
    assert "3 shots" in doc
    assert "1 failed" in doc
    assert 'class="shot failed"' in doc
    assert "nope" in doc


def test_html_embeds_the_contact_sheet_overview() -> None:
    """The overview strip is the same PNG, inlined."""
    frames = make_frames(2)
    sheet = render_contact_sheet(frames)
    doc = render_html(frames, contact_sheet_png=sheet)
    assert base64.b64encode(sheet).decode("ascii") in doc
    assert 'class="overview"' in doc


def test_html_is_deterministic() -> None:
    """Same frames, same document — no timestamps baked in."""
    frames = make_frames(3, notes=True)
    assert render_html(frames) == render_html(frames)


# ============================================================================
# Formatting helpers
# ============================================================================


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "00:00"),
        (8, "00:08"),
        (59.4, "00:59"),
        (75, "01:15"),
        (3600, "1:00:00"),
        (-5, "00:00"),
    ],
)
def test_format_timecode(seconds: float, expected: str) -> None:
    """Timecodes are MM:SS, widening to H:MM:SS past an hour."""
    assert format_timecode(seconds) == expected


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (b"\x89PNG\r\n\x1a\n rest", "image/png"),
        (b"\xff\xd8\xff\xe0 rest", "image/jpeg"),
        (b"GIF89a rest", "image/gif"),
        (b"RIFF____WEBPVP8 ", "image/webp"),
        (b"unknown", "image/png"),
    ],
)
def test_sniff_mime(data: bytes, expected: str) -> None:
    """Data URIs get the right MIME type from magic bytes."""
    assert _sniff_mime(data) == expected


def test_frame_failed_property() -> None:
    """`failed` is driven by the absence of image bytes."""
    assert StoryboardFrame(index=1, image_bytes=None, prompt="p").failed
    assert not StoryboardFrame(index=1, image_bytes=make_image(), prompt="p").failed


# ============================================================================
# write_storyboard
# ============================================================================


def test_write_storyboard_writes_both_artifacts(tmp_path: Path) -> None:
    """Both files land on disk and the returned URLs point at them."""
    frames = make_frames(4, notes=True)
    result = write_storyboard(frames, tmp_path / "boards", title="Reel 01")

    assert set(result) == {"sheet_path", "sheet_url", "html_path", "html_url"}
    sheet = Path(result["sheet_path"])
    page = Path(result["html_path"])
    assert sheet.is_file() and page.is_file()
    assert result["sheet_url"] == f"file://{sheet}"
    assert result["html_url"] == f"file://{page}"
    assert sheet.suffix == ".png" and page.suffix == ".html"

    open_png(sheet.read_bytes())
    doc = page.read_text(encoding="utf-8")
    assert "data:image/png;base64," in doc
    assert "<title>Reel 01</title>" in doc


def test_write_storyboard_honours_an_explicit_basename(tmp_path: Path) -> None:
    """A caller-supplied stem is used verbatim for both files."""
    result = write_storyboard(make_frames(1), tmp_path, basename="my-board")
    assert Path(result["sheet_path"]).name == "my-board.png"
    assert Path(result["html_path"]).name == "my-board.html"


def test_write_storyboard_default_names_do_not_collide(tmp_path: Path) -> None:
    """Two boards with the same title do not overwrite each other."""
    frames = make_frames(1)
    first = write_storyboard(frames, tmp_path, title="Same Title")
    second = write_storyboard(frames, tmp_path, title="Same Title")
    assert first["sheet_path"] != second["sheet_path"]
    assert "same-title" in Path(first["sheet_path"]).name
    assert len(list(tmp_path.glob("*.png"))) == 2


def test_write_storyboard_creates_missing_directories(tmp_path: Path) -> None:
    """The output directory is created on demand."""
    target = tmp_path / "a" / "b" / "c"
    result = write_storyboard(make_frames(1), target)
    assert Path(result["sheet_path"]).parent == target
