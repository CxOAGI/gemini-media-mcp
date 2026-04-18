"""Video decoding helpers for building transitions between existing clips."""

from io import BytesIO
from typing import Literal

import imageio.v3 as iio
from PIL import Image

FramePosition = Literal["start", "end"]


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
