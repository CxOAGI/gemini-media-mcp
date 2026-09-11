"""Video decoding helpers for building transitions between existing clips."""

import logging
import math
import shutil
from io import BytesIO
from pathlib import Path
from typing import Literal

import imageio.v3 as iio
from PIL import Image

logger = logging.getLogger(__name__)

FramePosition = Literal["start", "end"]


def measure_video_duration(path: Path) -> float | None:
    """Read a rendered file's real duration in seconds, or None.

    The only duration source that cannot drift. Every other figure in this
    server is either the caller's request or a value the server previously
    wrote down, so a wrong assumption propagates: each edit's sidecar seeds the
    next one's estimate, and an edit's rendered length is chosen by the
    service rather than predictable from the request or the source.
    Measuring the artifact settles what actually rendered, independent of what
    was asked for or recorded.

    Never raises: a probe failure falls back to the caller's existing
    reporting rather than failing a render that already succeeded and was
    already billed.

    Args:
        path: A local video file.

    Returns:
        Duration in seconds, or None when it cannot be determined.
    """
    try:
        import imageio.v3 as iio

        meta = iio.immeta(path, plugin="FFMPEG")
    except Exception:
        logger.debug("Could not probe duration of %s", path, exc_info=True)
        return None
    duration = meta.get("duration")
    if isinstance(duration, (int, float)) and math.isfinite(duration) and duration > 0:
        return float(duration)
    return None


def measure_video_dimensions(path: Path) -> tuple[int, int] | None:
    """Read a rendered file's real (width, height), or None.

    The counterpart to measure_video_duration, and needed for the same reason:
    a response that echoes the resolution it ASKED for cannot be told apart
    from one reporting what rendered. Omni bills per second and the per-second
    rate differs threefold across its resolution tiers, so "the request said
    360p" is not evidence the bill is a 360p bill.

    Never raises: a probe failure degrades to "not measured" rather than
    failing a render that already succeeded.

    Args:
        path: A local video file.

    Returns:
        (width, height) in pixels, or None when it cannot be determined.
    """
    try:
        import imageio.v3 as iio

        meta = iio.immeta(path, plugin="FFMPEG")
    except Exception:
        logger.debug("Could not probe dimensions of %s", path, exc_info=True)
        return None
    size = meta.get("size")
    if (
        isinstance(size, (tuple, list))
        and len(size) == 2
        and all(isinstance(v, int) and v > 0 for v in size)
    ):
        return (int(size[0]), int(size[1]))
    return None


# Nominal frame heights of the resolutions omni can emit, used to name what a
# measured height actually is. Tolerant bands, because a container may report a
# height a few pixels off a nominal one (encoder alignment) and an aspect ratio
# of 9:16 swaps the axes.
_RESOLUTION_HEIGHTS: tuple[tuple[str, int], ...] = (
    ("360p", 360),
    ("720p", 720),
    ("1080p", 1080),
    ("4K", 2160),
)


def classify_video_resolution(size: tuple[int, int] | None) -> str | None:
    """Name the resolution tier a measured (width, height) falls in, or None.

    Uses the SHORTER side, so a 9:16 portrait render classifies the same as
    the 16:9 landscape one — the tier is about pixel count per frame, and omni
    prices it that way.
    """
    if size is None:
        return None
    short_side = min(size)
    best: tuple[str, int] | None = None
    for name, nominal in _RESOLUTION_HEIGHTS:
        if best is None or abs(short_side - nominal) < abs(short_side - best[1]):
            best = (name, nominal)
    if best is None:
        return None
    # Only claim a tier when the measurement is genuinely near it; a 500px
    # render belongs to neither 360p nor 720p and saying otherwise would be
    # inventing evidence.
    return best[0] if abs(short_side - best[1]) <= best[1] * 0.15 else None


def measure_video_duration_bytes(data: bytes, extension: str = ".mp4") -> float | None:
    """Read an in-memory clip's duration in seconds, or None.

    The same probe as ``measure_video_duration`` for a video that has not been
    written to disk — an input the caller handed us, which the omni tools have
    to measure BEFORE spending anything on it: Google documents a 10s ceiling
    on an uploaded edit/extension source and a 3s ceiling on a video
    reference, and both are properties of the bytes, not of the request.

    Never raises, for the same reason: a probe failure must degrade to "length
    unknown" rather than refuse a clip that may well be fine.

    Args:
        data: Raw video bytes.
        extension: Container hint for the decoder (".mp4", ".mov", ".webm").

    Returns:
        Duration in seconds, or None when it cannot be determined.
    """
    try:
        import imageio.v3 as iio

        meta = iio.immeta(BytesIO(data), extension=extension, plugin="FFMPEG")
    except Exception:
        logger.debug("Could not probe duration of an in-memory clip", exc_info=True)
        return None
    duration = meta.get("duration")
    if isinstance(duration, (int, float)) and math.isfinite(duration) and duration > 0:
        return float(duration)
    return None


def assert_frame_decoding_available() -> None:
    """Raise if no usable ffmpeg binary is present.

    Frame extraction needs an ffmpeg the host can execute. The imageio-ffmpeg
    dependency bundles one, but it is a glibc build: on musl (Alpine, and so
    the published image) it is absent or unrunnable, which left generate_bridge
    and generate_clip(add_bridges=True) dead with an opaque decoder error —
    after the caller had been quoted a price.

    This is an environment fact, not a model capability, so no table can catch
    it; the tools call this in their pre-flight so a quote and a run agree.
    """
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError(
            "ffmpeg is required to read frames out of a video, and none was "
            "found. Install ffmpeg (the Docker image ships it; for a local "
            "install use your package manager) or set IMAGEIO_FFMPEG_EXE to a "
            f"working binary. Underlying error: {exc}"
        ) from exc
    # imageio-ffmpeg returns either an absolute path (its bundled build) or a
    # bare name it expects to resolve on PATH (a system install, which is what
    # the Alpine image now provides). Accept both.
    if exe and (Path(exe).exists() or shutil.which(exe)):
        return
    raise RuntimeError(
        "ffmpeg is required to read frames out of a video. imageio-ffmpeg "
        f"reported {exe!r}, which is not executable here. Install ffmpeg or "
        "set IMAGEIO_FFMPEG_EXE to a working binary."
    )


def extract_frame_png(
    video_bytes: bytes,
    position: FramePosition = "end",
    extension: str = ".mp4",
) -> bytes:
    """Decode one frame from `video_bytes` and return it as PNG bytes.

    `position="start"` returns the first frame; `position="end"` returns
    the last. `extension` gives imageio a hint about the container format
    (pass ".mov", ".webm", etc. if needed). The bundled `imageio-ffmpeg`
    package supplies the decoder, so no system ffmpeg is required.
    """
    buf = BytesIO(video_bytes)
    if position == "start":
        frame = iio.imread(buf, index=0, extension=extension)
    else:
        # Scan all frames, keep the last. imiter streams frames lazily so
        # memory stays bounded to one frame at a time.
        frame = None
        for f in iio.imiter(buf, extension=extension):
            frame = f
        if frame is None:
            raise ValueError("video has no decodable frames")

    img = Image.fromarray(frame)
    out = BytesIO()
    img.save(out, format="PNG")
    img.close()
    return out.getvalue()
