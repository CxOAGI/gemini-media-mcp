"""Tests for video_utils.extract_frame_png."""

from io import BytesIO

import imageio.v3 as iio
import numpy as np
import pytest
from PIL import Image

from src.video_utils import extract_frame_png


def _make_video(frames: list[tuple[int, int, int]]) -> bytes:
    """Encode a tiny MP4 with the given per-frame RGB colors."""
    arrays = [np.full((64, 64, 3), color, dtype=np.uint8) for color in frames]
    buf = BytesIO()
    iio.imwrite(
        buf,
        np.stack(arrays),
        extension=".mp4",
        fps=8,
        codec="libx264",
        pixelformat="yuv420p",
    )
    return buf.getvalue()


def _dominant_color(png_bytes: bytes) -> tuple[int, int, int]:
    img = Image.open(BytesIO(png_bytes)).convert("RGB")
    arr = np.asarray(img)
    return tuple(int(c) for c in arr.reshape(-1, 3).mean(axis=0))


@pytest.mark.timeout(30.0)
def test_extract_first_frame() -> None:
    """extract_frame_png('start') returns the first frame."""
    video = _make_video([(200, 50, 50), (50, 200, 50), (50, 50, 200)])
    png = extract_frame_png(video, "start")
    r, g, b = _dominant_color(png)
    # First frame is dominantly red; h264 color coding is lossy, so allow
    # a generous tolerance.
    assert r > g and r > b
    assert r > 120


@pytest.mark.timeout(30.0)
def test_extract_last_frame() -> None:
    """extract_frame_png('end') returns the final frame."""
    video = _make_video([(200, 50, 50), (50, 200, 50), (50, 50, 200)])
    png = extract_frame_png(video, "end")
    r, g, b = _dominant_color(png)
    # Last frame is dominantly blue.
    assert b > r and b > g
    assert b > 120


@pytest.mark.timeout(30.0)
def test_extract_frame_returns_png() -> None:
    """Output is a valid PNG."""
    video = _make_video([(100, 100, 100)])
    png = extract_frame_png(video, "start")
    img = Image.open(BytesIO(png))
    assert img.format == "PNG"
    img.close()
