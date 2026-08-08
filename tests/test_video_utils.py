"""Tests for video_utils.extract_frame_png."""

from pathlib import Path
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


def test_frame_decoding_preflight_reports_a_usable_ffmpeg() -> None:
    """generate_bridge and generate_clip(add_bridges=True) decode frames out
    of videos, which needs an ffmpeg the host can run.

    imageio-ffmpeg bundles a glibc build that cannot execute on musl, so the
    published Alpine image had the dependency and no working binary — both
    paths were dead with an opaque decoder error, after the caller had been
    quoted a price. The check accepts an absolute path (bundled build) or a
    bare name resolvable on PATH (a system install, which is what the image
    now ships).
    """
    from src.video_utils import assert_frame_decoding_available

    assert_frame_decoding_available()


def test_frame_decoding_preflight_raises_an_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A host with no ffmpeg must be told what to install, not handed a
    decoder stack trace."""
    import imageio_ffmpeg

    from src.video_utils import assert_frame_decoding_available

    def no_ffmpeg() -> str:
        raise RuntimeError("No ffmpeg exe could be found")

    monkeypatch.setattr(imageio_ffmpeg, "get_ffmpeg_exe", no_ffmpeg)
    with pytest.raises(RuntimeError, match="ffmpeg is required"):
        assert_frame_decoding_available()


def test_measure_video_duration_reads_the_real_length(tmp_path: Path) -> None:
    """The only duration source that cannot drift.

    Every other figure is the caller's request or something the server wrote
    down earlier — and an edit's sidecar seeds the next edit's estimate, so a
    wrong assumption propagates down the chain. Five review rounds argued
    about inherited-vs-requested duration without anyone measuring a rendered
    file; this makes every render self-report.
    """
    import subprocess

    import imageio_ffmpeg

    from src.video_utils import measure_video_duration

    clip = tmp_path / "three_seconds.mp4"
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=3:size=160x90:rate=24",
            str(clip),
        ],
        capture_output=True,
        check=True,
    )
    assert measure_video_duration(clip) == pytest.approx(3.0, abs=0.2)


@pytest.mark.parametrize("name", ["not_a_video.mp4", "missing.mp4"])
def test_measure_video_duration_never_raises(name: str, tmp_path: Path) -> None:
    """A probe failure must not fail a render that already succeeded and was
    already billed — it falls back to the caller's existing reporting."""
    from src.video_utils import measure_video_duration

    target = tmp_path / name
    if "not_a_video" in name:
        target.write_bytes(b"definitely not an mp4")
    assert measure_video_duration(target) is None
