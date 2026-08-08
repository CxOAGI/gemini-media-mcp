"""Storyboard rendering: turn generated keyframes into a reviewable artifact.

This module is PURE RENDERING. It never calls a generation API and makes no
network requests — callers generate the keyframes and hand the bytes here.
That keeps the whole module synchronous and unit-testable.

Why two artifacts?
    MCP clients (Claude Desktop, Claude Code) render *inline images* returned
    as MCP image content, but they do NOT execute or render arbitrary HTML/JS
    inside the conversation. So a storyboard that only existed as HTML would be
    invisible where the user is actually reviewing it. Hence:

    1. ``render_contact_sheet`` composites a real PNG — a grid of numbered
       panels with shot text drawn under each frame. This is the artifact the
       user looks at in chat, so it has to stand on its own.
    2. ``render_html`` emits a fully self-contained HTML document (images as
       ``data:`` URIs, CSS/JS inlined, zero external requests) that the user
       opens in a browser for the richer review pass: full prompts, notes,
       running timecode, light/dark support.

    ``write_storyboard`` writes both and returns their URLs.

Output is deterministic: no timestamps, no randomness, and no ambient state is
baked into either artifact, so the same frames always render the same bytes.
"""

import base64
import html
import logging
import math
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Literal, NamedTuple

from PIL import Image, ImageDraw, ImageFont, ImageOps

logger = logging.getLogger(__name__)

# A loaded font is either a real TrueType face or PIL's built-in bitmap
# fallback; both expose getbbox/getlength, which is all the layout code needs.
Font = ImageFont.FreeTypeFont | ImageFont.ImageFont

Theme = Literal["dark", "light"]

# ============================================================================
# Fonts
# ============================================================================

# PIL's built-in bitmap font is tiny and unstyleable, so a real TrueType face is
# strongly preferred. There is no portable font-discovery API in Pillow, so we
# probe a list of paths that cover the environments this server actually runs
# in: Debian/Ubuntu containers (DejaVu, Liberation, Noto), macOS (Helvetica, SF,
# Arial), and Alpine. Every entry is optional — a slim container may ship none
# of them, and rendering a storyboard must never fail just because the host has
# no fonts installed. See _load_font's fallback chain.
_REGULAR_FONT_PATHS: tuple[str, ...] = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "C:/Windows/Fonts/arial.ttf",
)

_BOLD_FONT_PATHS: tuple[str, ...] = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
)


@lru_cache(maxsize=64)
def _load_font(size: int, bold: bool = False) -> Font:
    """Load a TrueType face at ``size``, falling back to PIL's bitmap font.

    Tries each candidate path in order and returns the first that opens. If
    none do (a container with no fonts, or a Pillow build without FreeType),
    it degrades to ``ImageFont.load_default`` rather than raising — an ugly
    storyboard beats no storyboard.

    Results are memoized because a single sheet asks for the same handful of
    (size, weight) pairs once per panel.

    Args:
        size: Requested point size.
        bold: Whether to prefer a bold face.

    Returns:
        A FreeTypeFont when a face resolved, otherwise PIL's default font.
    """
    for path in _BOLD_FONT_PATHS if bold else _REGULAR_FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, ValueError):
            continue

    # No TrueType face available. Modern Pillow's load_default() accepts a size
    # and returns a scalable bundled face when FreeType is present; older or
    # FreeType-less builds only offer the fixed bitmap font.
    try:
        return ImageFont.load_default(size=size)
    except (OSError, TypeError, ValueError):
        logger.debug("No scalable font available; using PIL's bitmap default")
        return ImageFont.load_default()


def _is_scalable(font: Font) -> bool:
    """Whether ``font`` is a real TrueType face (vs. the bitmap fallback)."""
    return isinstance(font, ImageFont.FreeTypeFont)


def _ellipsis_for(font: Font) -> str:
    """Return the widest ellipsis ``font`` can actually draw.

    PIL's bitmap fallback has no U+2026 glyph and would silently drop it, so
    that path gets three ASCII dots instead.
    """
    return "\u2026" if _has_typographic_glyphs(font) else "..."


# Typographic characters outside Latin-1, mapped to ASCII lookalikes. Pillow's
# bundled fallback face covers Latin-1 only, so these render as tofu boxes on a
# system with no fonts installed. Slug lines use them constantly
# ("EXT. ALLEY — NIGHT"), so folding beats a row of blank squares.
_ASCII_FOLD = str.maketrans(
    {
        "\u2014": "-",  # em dash
        "\u2013": "-",  # en dash
        "\u2012": "-",  # figure dash
        "\u2010": "-",  # hyphen
        "\u2011": "-",  # non-breaking hyphen
        "\u2026": "...",  # ellipsis
        "\u2018": "'",  # left single quote
        "\u2019": "'",  # right single quote
        "\u201a": "'",
        "\u201c": '"',  # left double quote
        "\u201d": '"',  # right double quote
        "\u201e": '"',
        "\u2022": "*",  # bullet
        "\u2192": "->",  # right arrow
        "\u2190": "<-",  # left arrow
        "\u2260": "!=",
        "\u00a0": " ",  # non-breaking space
        "\u200b": "",  # zero-width space
    }
)


def _has_typographic_glyphs(font: Font) -> bool:
    """Whether ``font`` is a real system face with punctuation beyond Latin-1.

    Pillow's bundled fallback is a FreeTypeFont too, so ``_is_scalable`` does
    not distinguish it — but it is loaded from an in-memory buffer, whereas a
    face resolved from ``_REGULAR_FONT_PATHS`` carries a string path. That is
    the only reliable signal available without a font-introspection library.
    """
    return _is_scalable(font) and isinstance(getattr(font, "path", None), str)


def _drawable(text: str, font: Font) -> str:
    """Fold ``text`` to characters ``font`` can actually render."""
    if not text or _has_typographic_glyphs(font):
        return text
    return text.translate(_ASCII_FOLD)


def _text_width(font: Font, text: str) -> float:
    """Measured advance width of ``text`` in pixels for ``font``.

    Measures the folded form, so wrapping and truncation agree with what is
    actually drawn.
    """
    if not text:
        return 0.0
    return font.getlength(_drawable(text, font))


def _line_height(font: Font, leading: float = 1.35) -> int:
    """Line box height for ``font``, including leading.

    Measured from a string with both ascenders and descenders so the value is
    stable across faces (the bitmap fallback reports very different metrics
    from a TrueType face).
    """
    bbox = font.getbbox("ÁQgjy")
    return max(1, round((bbox[3] - bbox[1]) * leading))


# ============================================================================
# Text layout
# ============================================================================


def _split_overlong(word: str, font: Font, max_width: float) -> tuple[str, str]:
    """Split a single word that cannot fit on one line.

    Returns:
        ``(head, rest)`` where ``head`` is the longest prefix that fits (at
        least one character, so the caller always makes progress).
    """
    for cut in range(len(word) - 1, 0, -1):
        if _text_width(font, word[:cut]) <= max_width:
            return word[:cut], word[cut:]
    return word[:1], word[1:]


def _ellipsize(line: str, font: Font, max_width: float) -> str:
    """Trim ``line`` so it plus an ellipsis fits inside ``max_width``."""
    ellipsis = _ellipsis_for(font)
    trimmed = line
    while trimmed and _text_width(font, trimmed + ellipsis) > max_width:
        trimmed = trimmed[:-1]
    return trimmed.rstrip() + ellipsis


def _wrap_text(text: str, font: Font, max_width: float, max_lines: int) -> list[str]:
    """Word-wrap ``text`` to ``max_width``, truncating with an ellipsis.

    Wrapping measures with the real font rather than guessing a character
    count, so text never spills outside its panel regardless of which face
    resolved. Words longer than the line box are broken mid-word.

    Args:
        text: Text to wrap. Whitespace is collapsed.
        font: Font the text will be drawn with.
        max_width: Available width in pixels.
        max_lines: Maximum number of lines to emit.

    Returns:
        Up to ``max_lines`` lines; the last carries an ellipsis if content
        was dropped. Empty list when there is nothing to draw.
    """
    if not text or max_lines <= 0 or max_width <= 0:
        return []

    lines: list[str] = []
    current = ""
    overflowed = False

    for word in text.split():
        candidate = f"{current} {word}" if current else word
        if _text_width(font, candidate) <= max_width:
            current = candidate
            continue

        if current:
            lines.append(current)
            current = ""

        if _text_width(font, word) <= max_width:
            current = word
        else:
            head, rest = _split_overlong(word, font, max_width)
            lines.append(head)
            while _text_width(font, rest) > max_width:
                head, rest = _split_overlong(rest, font, max_width)
                lines.append(head)
            current = rest

        if len(lines) >= max_lines:
            overflowed = True
            break

    if current and not overflowed:
        lines.append(current)

    if len(lines) > max_lines or (overflowed and len(lines) >= max_lines):
        kept = lines[:max_lines]
        kept[-1] = _ellipsize(kept[-1], font, max_width)
        return kept
    return lines


# ============================================================================
# Formatting helpers
# ============================================================================


def format_timecode(seconds: float) -> str:
    """Format ``seconds`` as ``MM:SS`` (or ``H:MM:SS`` past an hour)."""
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _format_duration(seconds: float) -> str:
    """Format a shot duration compactly, e.g. ``8s`` or ``7.5s``."""
    if abs(seconds - round(seconds)) < 0.05:
        return f"{round(seconds)}s"
    return f"{seconds:.1f}s"


def _slugify(text: str, fallback: str = "storyboard") -> str:
    """Reduce ``text`` to a lowercase filename-safe slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:48] or fallback


def _sniff_mime(data: bytes) -> str:
    """Guess an image MIME type from magic bytes.

    Only the formats Gemini image models emit are recognised; anything else
    falls back to PNG, which browsers sniff anyway.
    """
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def _data_uri(data: bytes) -> str:
    """Encode ``data`` as a ``data:`` URI so the HTML stays self-contained."""
    return f"data:{_sniff_mime(data)};base64,{base64.b64encode(data).decode('ascii')}"


# ============================================================================
# Public data model
# ============================================================================


@dataclass(slots=True)
class StoryboardFrame:
    """One shot in a storyboard.

    Attributes:
        index: 1-based shot number, used for the panel badge and headings.
            Panels are laid out in list order; this only labels them.
        image_bytes: Encoded keyframe. ``None`` means generation failed for
            this shot — it still gets a panel, marked as failed.
        prompt: The prompt the keyframe was generated from.
        caption: Short shot description / slug line shown under the panel.
        duration_seconds: Intended shot length, used for badges, timecodes
            and the total runtime.
        notes: Camera / lighting / action notes.
        image_url: ``file://`` or ``gs://`` URL of the full-resolution frame.
        error: Why this shot failed, if it did.
    """

    index: int
    image_bytes: bytes | None
    prompt: str
    caption: str | None = None
    duration_seconds: float | None = None
    notes: str | None = None
    image_url: str | None = None
    error: str | None = None

    @property
    def failed(self) -> bool:
        """Whether this shot has no usable image."""
        return self.image_bytes is None


# ============================================================================
# Palette
# ============================================================================


class _Palette(NamedTuple):
    """Flat colour set for one contact-sheet theme."""

    background: str
    card: str
    border: str
    matte: str
    text: str
    muted: str
    accent: str
    accent_text: str
    badge: str
    badge_text: str
    error: str
    error_bg: str
    rule: str


_PALETTES: dict[str, _Palette] = {
    "dark": _Palette(
        background="#0f1115",
        card="#191c22",
        border="#2a2f3a",
        matte="#0a0c10",
        text="#eceff4",
        muted="#98a1b3",
        accent="#4f8cff",
        accent_text="#ffffff",
        badge="#0d1015",
        badge_text="#dde3ec",
        error="#ff6b6b",
        error_bg="#241518",
        rule="#262b35",
    ),
    "light": _Palette(
        background="#f2f3f5",
        card="#ffffff",
        border="#d9dce2",
        matte="#e8eaee",
        text="#14161a",
        muted="#5d6572",
        accent="#2563eb",
        accent_text="#ffffff",
        badge="#ffffff",
        badge_text="#22262e",
        error="#b42318",
        error_bg="#fdf2f1",
        rule="#dfe2e7",
    ),
}


def _palette(theme: Theme) -> _Palette:
    """Look up a palette, defaulting to dark for an unknown theme name."""
    return _PALETTES.get(theme, _PALETTES["dark"])


# ============================================================================
# Contact sheet geometry
# ============================================================================

_OUTER_PAD = 28
_GUTTER = 20
_CARD_PAD = 12
_CARD_RADIUS = 10
_BADGE_INSET = 8

# Fixed per-panel line budgets. Every panel reserves the same text height so
# the grid stays aligned no matter how much text each shot carries.
_CAPTION_LINES = 2
_PROMPT_LINES = 3
_NOTE_LINES = 2
_ERROR_LINES = 4

# Clamp the derived panel aspect ratio. Source frames are 16:9, 9:16, 1:1 or
# similar; anything wilder (a stitched panorama, a 1px test image) would blow
# out the sheet, so the panel box stays inside a sane band and the frame is
# letterboxed into it.
_MIN_ASPECT = 0.5
_MAX_ASPECT = 2.4
_DEFAULT_ASPECT = 16 / 9

# Wider grids make panels too narrow to carry readable text.
_MAX_COLUMNS = 5

# Floor for panel width; below this the wrapped text stops being legible.
_MIN_CARD_WIDTH = 200


def _default_columns(count: int) -> int:
    """Pick a column count that reads well for ``count`` shots.

    Boards of four or fewer stay on a single strip. Beyond that the grid
    starts at roughly square and widens only when doing so fills the last row
    more completely — a 5-shot board reads better as one row of five than as
    4 + 1 orphan. Columns cap at ``_MAX_COLUMNS`` so panels never shrink below
    a legible width.
    """
    if count <= 4:
        return max(1, count)

    start = min(math.ceil(math.sqrt(count)), _MAX_COLUMNS)
    best, best_gap = start, count
    for cols in range(start, _MAX_COLUMNS + 1):
        gap = (cols - count % cols) % cols  # empty slots in the last row
        if gap < best_gap:
            best, best_gap = cols, gap
    return best


def _panel_aspect(frames: Sequence[StoryboardFrame]) -> float:
    """Derive the panel box aspect (w/h) from the first decodable frame.

    Using the source aspect means a vertical reel gets vertical panels instead
    of tiny letterboxed slivers. Undecodable or absurd frames fall back to
    16:9.
    """
    for frame in frames:
        if frame.image_bytes is None:
            continue
        try:
            with Image.open(BytesIO(frame.image_bytes)) as probe:
                width, height = probe.size
        except (OSError, ValueError):
            continue
        if width > 0 and height > 0:
            return min(max(width / height, _MIN_ASPECT), _MAX_ASPECT)
    return _DEFAULT_ASPECT


def _prepare_frame_image(data: bytes, box: tuple[int, int], matte: str) -> Image.Image:
    """Letterbox ``data`` into a fixed ``box`` without distorting it.

    Handles RGBA, palette and grayscale sources by flattening onto the matte
    colour, and scales in either direction so both a 4K frame and a 32px thumb
    fill the same panel.

    Args:
        data: Encoded source image bytes.
        box: ``(width, height)`` of the panel image area.
        matte: Fill colour for the letterbox bars.

    Returns:
        A new RGB image of exactly ``box``. The caller owns and must close it.
    """
    box_w, box_h = max(1, box[0]), max(1, box[1])
    canvas = Image.new("RGB", (box_w, box_h), matte)

    source = Image.open(BytesIO(data))
    try:
        source.load()
        if source.width <= 0 or source.height <= 0:
            raise ValueError("image has zero width or height")
        # Palette and LA images may carry transparency; go through RGBA so the
        # alpha is composited onto the matte instead of turning black.
        if source.mode in ("RGBA", "LA", "P", "PA"):
            rgba = source.convert("RGBA")
            try:
                flat = Image.new("RGB", rgba.size, matte)
                flat.paste(rgba, mask=rgba.split()[3])
            finally:
                rgba.close()
        else:
            # Covers L, I, F, CMYK, YCbCr and RGB itself; convert() always
            # returns a new image, so `source` can be closed either way.
            flat = source.convert("RGB")
    except BaseException:
        canvas.close()
        raise
    finally:
        source.close()

    try:
        fitted = ImageOps.contain(flat, (box_w, box_h), Image.Resampling.LANCZOS)
    finally:
        flat.close()

    try:
        canvas.paste(
            fitted, ((box_w - fitted.width) // 2, (box_h - fitted.height) // 2)
        )
    finally:
        fitted.close()
    return canvas


def _draw_pill(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: Font,
    *,
    fill: str,
    text_color: str,
    pad_x: int = 8,
    pad_y: int = 4,
    align_right: bool = False,
) -> tuple[int, int]:
    """Draw a rounded badge around ``text``.

    Args:
        draw: Target draw context.
        xy: Top-left corner, or top-right when ``align_right`` is set.
        text: Badge label.
        font: Font for the label.
        fill: Badge background colour.
        text_color: Label colour.
        pad_x: Horizontal padding inside the badge.
        pad_y: Vertical padding inside the badge.
        align_right: Treat ``xy`` as the badge's top-right corner.

    Returns:
        The ``(width, height)`` of the drawn badge.
    """
    text_w = _text_width(font, text)
    text_h = _line_height(font, leading=1.0)
    width = round(text_w) + 2 * pad_x
    height = text_h + 2 * pad_y
    x = xy[0] - width if align_right else xy[0]
    y = xy[1]
    draw.rounded_rectangle((x, y, x + width, y + height), radius=height // 2, fill=fill)
    draw.text((x + pad_x, y + pad_y), _drawable(text, font), font=font, fill=text_color)
    return width, height


def _draw_lines(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    lines: Sequence[str],
    font: Font,
    color: str,
    line_h: int,
) -> int:
    """Draw ``lines`` stacked downward and return the y after the last one."""
    x, y = xy
    for line in lines:
        draw.text((x, y), _drawable(line, font), font=font, fill=color)
        y += line_h
    return y


class _SheetFonts(NamedTuple):
    """Font set for one contact sheet.

    ``title``, ``subtitle`` and ``alert`` are header faces at fixed sizes; the
    rest scale with the panel width.
    """

    title: Font
    subtitle: Font
    alert: Font
    caption: Font
    body: Font
    badge: Font


def _sheet_fonts(panel_width: int) -> _SheetFonts:
    """Build the sheet's fonts, scaling body text to the panel width.

    Narrow panels (many columns) get proportionally smaller text so wrapping
    still yields readable lines instead of two words per row. The header faces
    stay fixed: the header is identical on every board, so letting it shrink
    with the grid would only make wide boards harder to read.
    """
    scale = panel_width / 480
    return _SheetFonts(
        title=_load_font(30, bold=True),
        subtitle=_load_font(15),
        alert=_load_font(13, bold=True),
        caption=_load_font(max(11, round(15 * scale)), bold=True),
        body=_load_font(max(10, round(13 * scale))),
        badge=_load_font(max(10, round(12 * scale)), bold=True),
    )


def _render_panel(
    frame: StoryboardFrame,
    *,
    card_size: tuple[int, int],
    image_h: int,
    fonts: _SheetFonts,
    colors: _Palette,
    note_lines: int,
) -> Image.Image:
    """Render one storyboard panel as a standalone card image.

    Args:
        frame: The shot to render.
        card_size: ``(width, height)`` of the card.
        image_h: Height of the image area at the top of the card.
        fonts: Sheet font set.
        colors: Sheet palette.
        note_lines: Reserved note lines (0 when no shot on the sheet has notes).

    Returns:
        An RGB card image. The caller owns and must close it.
    """
    card_w, card_h = card_size
    card = Image.new("RGB", (card_w, card_h), colors.card)
    draw = ImageDraw.Draw(card)
    text_w = card_w - 2 * _CARD_PAD

    if frame.image_bytes is None:
        _render_failed_area(
            draw, frame, card_w=card_w, image_h=image_h, fonts=fonts, colors=colors
        )
    else:
        try:
            panel_img = _prepare_frame_image(
                frame.image_bytes, (card_w, image_h), colors.matte
            )
        except (OSError, ValueError) as exc:
            # Bytes that Pillow cannot decode are treated exactly like a failed
            # shot: a partially broken board must still be reviewable.
            logger.warning("Shot %s: undecodable image bytes (%s)", frame.index, exc)
            _render_failed_area(
                draw,
                frame,
                card_w=card_w,
                image_h=image_h,
                fonts=fonts,
                colors=colors,
                error_override=f"Image could not be decoded: {exc}",
            )
        else:
            try:
                card.paste(panel_img, (0, 0))
            finally:
                panel_img.close()

    # Badges sit on top of the image so they read at thumbnail size.
    _draw_pill(
        draw,
        (_BADGE_INSET, _BADGE_INSET),
        str(frame.index),
        fonts.badge,
        fill=colors.error if frame.failed else colors.accent,
        text_color=colors.accent_text,
        pad_x=9,
    )
    if frame.duration_seconds is not None:
        badge_h = _line_height(fonts.badge, leading=1.0) + 8
        _draw_pill(
            draw,
            (card_w - _BADGE_INSET, image_h - _BADGE_INSET - badge_h),
            _format_duration(frame.duration_seconds),
            fonts.badge,
            fill=colors.badge,
            text_color=colors.badge_text,
            align_right=True,
        )

    # Hairline between the frame and its text block.
    draw.line((0, image_h, card_w, image_h), fill=colors.border)

    caption_lh = _line_height(fonts.caption)
    body_lh = _line_height(fonts.body)
    y = image_h + _CARD_PAD

    # The failure reason is already drawn across the image area, so the caption
    # slot only needs to label the shot rather than repeat it.
    caption = frame.caption or (
        f"Shot {frame.index} — not generated" if frame.failed else ""
    )
    caption_lines = _wrap_text(caption, fonts.caption, text_w, _CAPTION_LINES)
    _draw_lines(
        draw,
        (_CARD_PAD, y),
        caption_lines,
        fonts.caption,
        colors.error if frame.failed and not frame.caption else colors.text,
        caption_lh,
    )
    y += _CAPTION_LINES * caption_lh + 4

    prompt_lines = _wrap_text(frame.prompt, fonts.body, text_w, _PROMPT_LINES)
    _draw_lines(draw, (_CARD_PAD, y), prompt_lines, fonts.body, colors.muted, body_lh)
    y += _PROMPT_LINES * body_lh

    if note_lines and frame.notes:
        y += 4
        wrapped = _wrap_text(frame.notes, fonts.body, text_w, note_lines)
        _draw_lines(draw, (_CARD_PAD, y), wrapped, fonts.body, colors.muted, body_lh)

    draw.rounded_rectangle(
        (0, 0, card_w - 1, card_h - 1), radius=_CARD_RADIUS, outline=colors.border
    )
    return card


def _render_failed_area(
    draw: ImageDraw.ImageDraw,
    frame: StoryboardFrame,
    *,
    card_w: int,
    image_h: int,
    fonts: _SheetFonts,
    colors: _Palette,
    error_override: str | None = None,
) -> None:
    """Fill a panel's image area with a clearly-marked failure placeholder.

    ``generate_clip`` tolerates per-beat failures, so a storyboard with holes
    in it is a real case — the panel says what went wrong instead of leaving a
    silent gap.
    """
    draw.rectangle((0, 0, card_w, image_h), fill=colors.error_bg)
    # Diagonal rules mark the panel as intentionally empty at a glance.
    step = max(24, card_w // 10)
    for offset in range(-image_h, card_w, step):
        draw.line((offset, image_h, offset + image_h, 0), fill=colors.border)

    label = "SHOT NOT GENERATED"
    label_w = _text_width(fonts.badge, label)
    label_h = _line_height(fonts.badge, leading=1.0) + 8

    message = error_override or frame.error or "No image was returned for this shot."
    text_w = card_w - 2 * _CARD_PAD
    body_lh = _line_height(fonts.body)
    lines = _wrap_text(message, fonts.body, text_w, _ERROR_LINES)

    # Centre the label + message block vertically inside the image area.
    block_h = label_h + 10 + len(lines) * body_lh
    top = max(_CARD_PAD, (image_h - block_h) // 2)

    _draw_pill(
        draw,
        (round((card_w - label_w) / 2) - 10, top),
        label,
        fonts.badge,
        fill=colors.error,
        # Knockout in the palette's error surface colour, matching the header's
        # failure flag: white on the dark theme's light red is a weak contrast.
        text_color=colors.error_bg,
        pad_x=10,
    )
    _draw_lines(
        draw,
        (_CARD_PAD, top + label_h + 10),
        lines,
        fonts.body,
        colors.text,
        body_lh,
    )


class _Layout(NamedTuple):
    """Resolved contact-sheet geometry for one candidate panel width."""

    fonts: _SheetFonts
    card_w: int
    card_h: int
    image_h: int
    cols: int
    rows: int
    header_h: int
    sheet_w: int
    sheet_h: int


def _layout_sheet(
    *, card_w: int, cols: int, count: int, aspect: float, note_lines: int
) -> _Layout:
    """Compute full sheet geometry for a given panel width.

    Every panel reserves the same text height, so the grid stays aligned no
    matter how much text an individual shot carries.
    """
    fonts = _sheet_fonts(card_w)
    image_h = max(90, round(card_w / aspect))

    caption_lh = _line_height(fonts.caption)
    body_lh = _line_height(fonts.body)
    text_h = (
        _CARD_PAD
        + _CAPTION_LINES * caption_lh
        + 4
        + _PROMPT_LINES * body_lh
        + ((4 + note_lines * body_lh) if note_lines else 0)
        + _CARD_PAD
    )
    card_h = image_h + text_h
    rows = math.ceil(count / cols)

    header_h = _line_height(fonts.title) + _line_height(fonts.subtitle) + 14
    sheet_w = 2 * _OUTER_PAD + cols * card_w + (cols - 1) * _GUTTER
    grid_h = rows * card_h + (rows - 1) * _GUTTER
    # No footer band: the header already carries the whole board summary, so a
    # footer could only repeat it. The grid just ends on the outer padding.
    sheet_h = _OUTER_PAD + header_h + grid_h + _OUTER_PAD

    return _Layout(
        fonts=fonts,
        card_w=card_w,
        card_h=card_h,
        image_h=image_h,
        cols=cols,
        rows=rows,
        header_h=header_h,
        sheet_w=sheet_w,
        sheet_h=sheet_h,
    )


def render_contact_sheet(
    frames: Sequence[StoryboardFrame],
    *,
    title: str = "Storyboard",
    subtitle: str | None = None,
    columns: int | None = None,
    theme: Theme = "dark",
    panel_width: int = 480,
    max_sheet_width: int = 1760,
    max_sheet_height: int = 3000,
) -> bytes:
    """Composite ``frames`` into a single storyboard contact-sheet PNG.

    This is the artifact an MCP client can actually show inline, so it carries
    everything needed for a first-pass review: numbered panels, per-shot
    duration badges, captions, prompts, and a header that summarises the board
    and flags any shots that failed to generate.

    Args:
        frames: Shots in playback order. Must not be empty.
        title: Header title.
        subtitle: Optional line under the title (e.g. the source prompt).
        columns: Grid columns. Derived from the shot count when omitted.
        theme: ``"dark"`` (default) or ``"light"``.
        panel_width: Preferred panel width in pixels.
        max_sheet_width: Hard cap on total sheet width; panels shrink to fit
            so a 20-shot board stays a sane size for inline display.
        max_sheet_height: Soft cap on total sheet height. A tall board (many
            vertical 9:16 shots) shrinks its panels rather than emitting a
            multi-megabyte PNG that an MCP client has to downscale anyway.
            Soft because panels never shrink past a legible minimum width, so
            a very tall board with a very small cap can still exceed it.

    Returns:
        PNG bytes.

    Raises:
        ValueError: If ``frames`` is empty or the geometry arguments are
            non-positive.
    """
    if not frames:
        raise ValueError("frames must not be empty — nothing to render")
    if panel_width <= 0 or max_sheet_width <= 0 or max_sheet_height <= 0:
        raise ValueError("panel and sheet size limits must be positive")

    colors = _palette(theme)
    count = len(frames)
    cols = columns if columns and columns > 0 else _default_columns(count)
    cols = min(max(1, cols), count)

    # Shrink panels rather than overflow the width cap.
    available = max_sheet_width - 2 * _OUTER_PAD - (cols - 1) * _GUTTER
    card_w = max(_MIN_CARD_WIDTH, min(panel_width, available // cols))

    aspect = _panel_aspect(frames)
    note_lines = _NOTE_LINES if any(f.notes for f in frames) else 0

    def layout_for(width: int) -> _Layout:
        return _layout_sheet(
            card_w=width, cols=cols, count=count, aspect=aspect, note_lines=note_lines
        )

    layout = layout_for(card_w)
    if layout.sheet_h > max_sheet_height and card_w > _MIN_CARD_WIDTH:
        # Binary search the widest panel that fits the height budget. Height is
        # monotonic in panel width but not linear (text blocks scale by their
        # own font steps), so a search is more reliable than an estimate. If
        # even _MIN_CARD_WIDTH overflows, legibility wins over the cap.
        low, high = _MIN_CARD_WIDTH, card_w
        while low < high:
            mid = (low + high + 1) // 2
            if layout_for(mid).sheet_h <= max_sheet_height:
                low = mid
            else:
                high = mid - 1
        layout = layout_for(low)

    fonts = layout.fonts
    card_w, card_h, image_h = layout.card_w, layout.card_h, layout.image_h
    header_h = layout.header_h
    sheet_w, sheet_h = layout.sheet_w, layout.sheet_h

    title_lh = _line_height(fonts.title)
    sub_lh = _line_height(fonts.subtitle)

    sheet = Image.new("RGB", (sheet_w, sheet_h), colors.background)
    try:
        draw = ImageDraw.Draw(sheet)

        total_runtime = sum(f.duration_seconds or 0.0 for f in frames)
        failed = sum(1 for f in frames if f.failed)
        summary = f"{count} shot{'s' if count != 1 else ''}"
        # Durations drive every timecode downstream, so a board without them
        # says so rather than quietly dropping the runtime.
        summary += (
            f"  ·  {format_timecode(total_runtime)} runtime"
            if total_runtime > 0
            else "  ·  no durations set"
        )

        # Header: title left, summary right on the same baseline band. This is
        # the board's only summary — it used to be repeated in a footer, which
        # read as a rendering bug on a short board (both copies visible at
        # once) and spent a row of height saying nothing new.
        #
        # Laid out right to left, because the title is the elastic part: the
        # failure flag and the summary claim their space first and the title
        # gets what is left, so a long board name can never run under either.
        summary_y = _OUTER_PAD + title_lh - sub_lh - 2
        summary_text = _drawable(summary, fonts.subtitle)
        summary_w = _text_width(fonts.subtitle, summary)
        # Centre of the summary's *ink*, not of its line box: the line box
        # reserves descender space this text mostly does not use, so centring
        # the flag on it would hang the flag visibly low.
        summary_ink = fonts.subtitle.getbbox(summary_text)
        summary_mid = summary_y + round((summary_ink[1] + summary_ink[3]) / 2)
        right_edge = sheet_w - _OUTER_PAD

        if failed:
            # A board with holes in it is a diagnostic artifact, so the failure
            # count is a filled pill in the error colour instead of one more
            # muted phrase in the summary run. error_bg is the palette's
            # counterpart to error, so the label stays legible in both themes.
            flag = f"{failed} failed"
            flag_h = _line_height(fonts.alert, leading=1.0) + 8
            flag_w, _ = _draw_pill(
                draw,
                (right_edge, summary_mid - flag_h // 2),
                flag,
                fonts.alert,
                fill=colors.error,
                text_color=colors.error_bg,
                pad_x=10,
                align_right=True,
            )
            right_edge -= flag_w + 12

        summary_x = right_edge - summary_w
        draw.text(
            (summary_x, summary_y),
            summary_text,
            font=fonts.subtitle,
            fill=colors.muted,
        )
        # Titles are caller-supplied and unbounded, so ellipsize rather than
        # let one run under the summary. On a sheet too narrow to hold even the
        # ellipsis the title is dropped: a smear of glyphs over the summary is
        # worse than no title, and the summary is the part carrying meaning.
        title_text = title
        title_max = summary_x - 16 - _OUTER_PAD
        if _text_width(fonts.title, title_text) > title_max:
            title_text = _ellipsize(title_text, fonts.title, title_max)
        if _text_width(fonts.title, title_text) <= title_max:
            draw.text(
                (_OUTER_PAD, _OUTER_PAD),
                _drawable(title_text, fonts.title),
                font=fonts.title,
                fill=colors.text,
            )
        if subtitle:
            sub_lines = _wrap_text(
                subtitle, fonts.subtitle, sheet_w - 2 * _OUTER_PAD, 1
            )
            _draw_lines(
                draw,
                (_OUTER_PAD, _OUTER_PAD + title_lh),
                sub_lines,
                fonts.subtitle,
                colors.muted,
                sub_lh,
            )
        rule_y = _OUTER_PAD + header_h - 8
        draw.line((_OUTER_PAD, rule_y, sheet_w - _OUTER_PAD, rule_y), fill=colors.rule)

        grid_top = _OUTER_PAD + header_h
        for position, frame in enumerate(frames):
            col, row = position % cols, position // cols
            x = _OUTER_PAD + col * (card_w + _GUTTER)
            y = grid_top + row * (card_h + _GUTTER)
            card = _render_panel(
                frame,
                card_size=(card_w, card_h),
                image_h=image_h,
                fonts=fonts,
                colors=colors,
                note_lines=note_lines,
            )
            # Round the card's corners by pasting through a matching mask, so
            # the frame's own corners get clipped too.
            mask = Image.new("L", (card_w, card_h), 0)
            try:
                ImageDraw.Draw(mask).rounded_rectangle(
                    (0, 0, card_w - 1, card_h - 1), radius=_CARD_RADIUS, fill=255
                )
                sheet.paste(card, (x, y), mask)
            finally:
                mask.close()
                card.close()

        buffer = BytesIO()
        sheet.save(buffer, format="PNG", optimize=True)
        return buffer.getvalue()
    finally:
        sheet.close()


# ============================================================================
# HTML
# ============================================================================

# Inlined so the document works from a file:// URL with no network at all —
# no CDN, no web fonts, no remote images. Everything the browser needs is
# either here or a data: URI.
_HTML_STYLE = """
:root {
  color-scheme: light dark;
  --bg: #f4f5f7;
  --card: #ffffff;
  --border: #dcdfe5;
  --text: #14161a;
  --muted: #5d6572;
  --accent: #2563eb;
  --accent-text: #ffffff;
  --matte: #e8eaee;
  --error: #b42318;
  --error-bg: #fdf2f1;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f1115;
    --card: #191c22;
    --border: #2a2f3a;
    --text: #eceff4;
    --muted: #98a1b3;
    --accent: #4f8cff;
    --accent-text: #0b0e13;
    --matte: #0a0c10;
    --error: #ff6b6b;
    --error-bg: #241518;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 24px 20px 64px;
  background: var(--bg);
  color: var(--text);
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial,
        sans-serif;
}
.wrap { max-width: 1100px; margin: 0 auto; }
header.board { border-bottom: 1px solid var(--border); padding-bottom: 16px; }
h1 { font-size: 1.6rem; margin: 0 0 6px; letter-spacing: -0.01em; }
.sub { color: var(--muted); margin: 0 0 10px; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; list-style: none;
         margin: 0; padding: 0; }
.chip { border: 1px solid var(--border); border-radius: 999px;
        padding: 2px 10px; font-size: 0.8rem; color: var(--muted); }
.chip.bad { color: var(--error); border-color: var(--error); }
.overview { margin: 24px 0 8px; }
.overview img { width: 100%; border: 1px solid var(--border); border-radius: 10px; }
.shots { list-style: none; margin: 24px 0 0; padding: 0;
         display: grid; gap: 20px; }
.shot {
  display: grid;
  grid-template-columns: minmax(0, 440px) minmax(0, 1fr);
  gap: 20px;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 16px;
}
.shot.failed { border-color: var(--error); background: var(--error-bg); }
figure { margin: 0; }
figure img {
  display: block; width: 100%; max-width: 100%; height: auto;
  background: var(--matte); border-radius: 8px; cursor: zoom-in;
}
figure.zoom img { cursor: zoom-out; }
.placeholder {
  display: flex; align-items: center; justify-content: center;
  min-height: 180px; border: 1px dashed var(--error); border-radius: 8px;
  color: var(--error); font-weight: 600; text-align: center; padding: 16px;
}
.shot h2 { font-size: 1.05rem; margin: 0 0 4px; display: flex;
           align-items: center; gap: 10px; }
.num { background: var(--accent); color: var(--accent-text); border-radius: 6px;
       font-size: 0.8rem; padding: 2px 8px; font-variant-numeric: tabular-nums; }
.shot.failed .num { background: var(--error); color: #fff; }
.tc { color: var(--muted); font-size: 0.85rem; font-weight: 400;
      font-variant-numeric: tabular-nums; }
dl { margin: 12px 0 0; display: grid; grid-template-columns: 92px minmax(0, 1fr);
     gap: 6px 14px; }
dt { color: var(--muted); font-size: 0.8rem; text-transform: uppercase;
     letter-spacing: 0.04em; }
dd { margin: 0; overflow-wrap: anywhere; white-space: pre-wrap; }
dd.err { color: var(--error); }
a { color: var(--accent); }
footer.board { margin-top: 28px; padding-top: 14px; color: var(--muted);
               border-top: 1px solid var(--border); font-size: 0.85rem; }
@media (max-width: 760px) {
  .shot { grid-template-columns: minmax(0, 1fr); }
  dl { grid-template-columns: minmax(0, 1fr); gap: 2px; }
  dt { margin-top: 8px; }
}
.shot figure.zoom { grid-column: 1 / -1; }
"""

# Click a frame to blow it up to full width for a closer look. Deliberately
# tiny and data-free: it never touches user-supplied strings.
_HTML_SCRIPT = """
document.addEventListener('click', function (event) {
  var img = event.target.closest('figure img');
  if (img) { img.parentElement.classList.toggle('zoom'); }
});
"""


def _html_field(label: str, value: str, *, error: bool = False) -> str:
    """Render one escaped ``<dt>/<dd>`` pair."""
    css = ' class="err"' if error else ""
    return f"    <dt>{html.escape(label)}</dt><dd{css}>{html.escape(value)}</dd>"


def render_html(
    frames: Sequence[StoryboardFrame],
    *,
    title: str = "Storyboard",
    subtitle: str | None = None,
    contact_sheet_png: bytes | None = None,
) -> str:
    """Render a fully self-contained storyboard review page.

    Every image is embedded as a ``data:`` URI and the CSS/JS are inlined, so
    the document renders identically offline and issues zero external
    requests. All caller-supplied text is escaped — prompts are untrusted
    input and must never be able to inject markup.

    Args:
        frames: Shots in playback order. Must not be empty.
        title: Page title and heading.
        subtitle: Optional line under the heading.
        contact_sheet_png: Optional contact sheet to embed as an overview
            strip above the per-shot list.

    Returns:
        A complete HTML document as a string.

    Raises:
        ValueError: If ``frames`` is empty.
    """
    if not frames:
        raise ValueError("frames must not be empty — nothing to render")

    count = len(frames)
    total_runtime = sum(f.duration_seconds or 0.0 for f in frames)
    failed = sum(1 for f in frames if f.failed)

    parts: list[str] = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{html.escape(title)}</title>",
        f"<style>{_HTML_STYLE}</style>",
        "</head>",
        "<body>",
        '<div class="wrap">',
        '<header class="board">',
        f"<h1>{html.escape(title)}</h1>",
    ]
    if subtitle:
        parts.append(f'<p class="sub">{html.escape(subtitle)}</p>')
    chips = [f"{count} shot{'s' if count != 1 else ''}"]
    if total_runtime > 0:
        chips.append(f"{format_timecode(total_runtime)} total runtime")
    parts.append('<ul class="chips">')
    parts.extend(f'<li class="chip">{html.escape(chip)}</li>' for chip in chips)
    if failed:
        parts.append(f'<li class="chip bad">{failed} failed</li>')
    parts.append("</ul>")
    parts.append("</header>")

    if contact_sheet_png:
        parts.append(
            f'<section class="overview"><img alt="Contact sheet overview" '
            f'src="{_data_uri(contact_sheet_png)}"></section>'
        )

    parts.append('<ol class="shots">')
    elapsed = 0.0
    for frame in frames:
        duration = frame.duration_seconds or 0.0
        start, end = elapsed, elapsed + duration
        elapsed = end

        classes = "shot failed" if frame.failed else "shot"
        heading = html.escape(frame.caption or f"Shot {frame.index}")
        parts.append(f'<li class="{classes}">')

        if frame.image_bytes is None:
            note = html.escape(frame.error or "No image was returned for this shot.")
            parts.append(f'<figure><div class="placeholder">{note}</div></figure>')
        else:
            alt = html.escape(frame.caption or f"Shot {frame.index}")
            parts.append(
                f'<figure><img alt="{alt}" src="{_data_uri(frame.image_bytes)}">'
                "</figure>"
            )

        timecode = f"{format_timecode(start)} – {format_timecode(end)}"
        parts.append("<div>")
        parts.append(
            f'<h2><span class="num">{frame.index}</span>{heading}'
            f'<span class="tc">{html.escape(timecode)}</span></h2>'
        )
        parts.append("<dl>")
        parts.append(_html_field("Prompt", frame.prompt))
        if frame.duration_seconds is not None:
            parts.append(
                _html_field("Duration", _format_duration(frame.duration_seconds))
            )
        if frame.notes:
            parts.append(_html_field("Notes", frame.notes))
        if frame.error:
            parts.append(_html_field("Error", frame.error, error=True))
        if frame.image_url:
            url = html.escape(frame.image_url, quote=True)
            parts.append(
                f'    <dt>Source</dt><dd><a href="{url}">'
                f"{html.escape(frame.image_url)}</a></dd>"
            )
        parts.append("</dl></div></li>")

    parts.append("</ol>")
    footer = f"{count} shot{'s' if count != 1 else ''}"
    if total_runtime > 0:
        footer += f" · {format_timecode(total_runtime)} total runtime"
    if failed:
        footer += f" · {failed} failed"
    parts.append(f'<footer class="board">{html.escape(footer)}</footer>')
    parts.append("</div>")
    parts.append(f"<script>{_HTML_SCRIPT}</script>")
    parts.append("</body></html>")
    return "\n".join(parts)


# ============================================================================
# Disk output
# ============================================================================


def write_storyboard(
    frames: Sequence[StoryboardFrame],
    out_dir: Path,
    *,
    title: str = "Storyboard",
    subtitle: str | None = None,
    columns: int | None = None,
    theme: Theme = "dark",
    basename: str | None = None,
) -> dict[str, str]:
    """Render both artifacts to ``out_dir`` and return their locations.

    The PNG is the inline-displayable one; the HTML is the browser review
    page and embeds the same PNG as an overview strip.

    Args:
        frames: Shots in playback order. Must not be empty.
        out_dir: Directory to write into. Created if missing.
        title: Storyboard title used in both artifacts.
        subtitle: Optional line under the title.
        columns: Contact-sheet grid columns; derived when omitted.
        theme: Contact-sheet theme.
        basename: Filename stem for both files. Defaults to a slug of the
            title plus a short random suffix so repeated calls do not
            overwrite each other.

    Returns:
        ``{"sheet_path", "sheet_url", "html_path", "html_url"}`` — paths are
        absolute, URLs are ``file://``.

    Raises:
        ValueError: If ``frames`` is empty.
    """
    if not frames:
        raise ValueError("frames must not be empty — nothing to render")

    stem = basename or f"{_slugify(title)}-{uuid.uuid4().hex[:8]}"
    out_dir.mkdir(parents=True, exist_ok=True)

    sheet_png = render_contact_sheet(
        frames, title=title, subtitle=subtitle, columns=columns, theme=theme
    )
    sheet_path = out_dir / f"{stem}.png"
    sheet_path.write_bytes(sheet_png)

    document = render_html(
        frames, title=title, subtitle=subtitle, contact_sheet_png=sheet_png
    )
    html_path = out_dir / f"{stem}.html"
    html_path.write_text(document, encoding="utf-8")

    return {
        "sheet_path": str(sheet_path),
        "sheet_url": f"file://{sheet_path}",
        "html_path": str(html_path),
        "html_url": f"file://{html_path}",
    }
