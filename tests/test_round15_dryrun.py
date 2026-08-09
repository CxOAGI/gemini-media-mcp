"""Round-15 follow-up: the dry_run source-refusal invariant everywhere.

A prior round wired ``_assert_local_source`` into loop_extend, generate_bridge,
generate_clip and generate_video_omni so a quote refuses a local file source
the real run would refuse (missing, or outside DATA_FOLDER). These tests pin
the same behaviour for the three remaining file-consuming tools —
generate_video, generate_image and generate_transition — on the dry_run path:

  * a missing in-sandbox source and an out-of-sandbox source each return an
    error (not a price) carrying the same detailed reason the render emits;
  * a gs:// source is uncheckable offline, so it still prices; and
  * for generate_video, a Veo-Lite mode restriction still takes precedence
    over the source error when both apply — the source check must not preempt
    the more specific validation.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from tests.test_main import _video_ctx

VEO = "veo-3.1-generate-001"
LITE = "veo-3.1-lite-generate-preview"
IMAGE_MODEL = "gemini-3.1-flash-image"


def _payload(result: Any) -> dict[str, Any]:
    """generate_image returns a list[TextContent]; the others return a str."""
    text = result if isinstance(result, str) else result[0].text
    return json.loads(text)


def _out_of_sandbox() -> str:
    """A local file:// URI that resolves outside any test's DATA_FOLDER."""
    return "file:///etc/hosts"


def _missing_in_sandbox(tmp_path: Path) -> str:
    """A local file:// URI inside DATA_FOLDER that does not exist."""
    return f"file://{tmp_path / 'images' / 'nope.png'}"


# ---------------------------------------------------------------------------
# generate_video
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
@pytest.mark.parametrize(
    ("mode_kwargs", "named"),
    [
        pytest.param({"image_uri": None}, "image_uri", id="image_uri"),
        pytest.param(
            {"image_uri": "gs://b/first.png", "last_frame_uri": None},
            "last_frame_uri",
            id="last_frame_uri",
        ),
        pytest.param(
            {"extend_video_uri": None}, "extend_video_uri", id="extend_video_uri"
        ),
        pytest.param(
            {"reference_image_uris": [None]}, "reference image", id="reference_image"
        ),
    ],
)
async def test_generate_video_dry_run_refuses_an_out_of_sandbox_source(
    tmp_path: Path, mode_kwargs: dict[str, Any], named: str
) -> None:
    """Every file-consuming param the render fetches is refused on the quote
    when it points outside DATA_FOLDER, with the render's own reason."""
    from src.__main__ import generate_video

    bad = _out_of_sandbox()
    kwargs: dict[str, Any] = {}
    for key, value in mode_kwargs.items():
        kwargs[key] = [bad] if isinstance(value, list) else (value if value else bad)

    payload = _payload(
        await generate_video(
            ctx=_video_ctx(tmp_path), prompt="x", model=VEO, dry_run=True, **kwargs
        )
    )
    assert "estimated_cost" not in payload
    assert "outside the permitted data folder" in payload["error"]
    assert "DATA_FOLDER" in payload["error"]
    assert named in payload["error"]


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_generate_video_dry_run_refuses_a_missing_in_sandbox_source(
    tmp_path: Path,
) -> None:
    """A path inside DATA_FOLDER that does not exist is refused with the
    'File not found' reason, distinct from the confinement message."""
    from src.__main__ import generate_video

    payload = _payload(
        await generate_video(
            ctx=_video_ctx(tmp_path),
            prompt="x",
            model=VEO,
            image_uri=_missing_in_sandbox(tmp_path),
            dry_run=True,
        )
    )
    assert "estimated_cost" not in payload
    assert "image_uri" in payload["error"]
    assert "File not found" in payload["error"]


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_generate_video_dry_run_still_prices_a_gs_source(tmp_path: Path) -> None:
    """A gs:// source is uncheckable offline, so the quote still prices it."""
    from src.__main__ import generate_video

    payload = _payload(
        await generate_video(
            ctx=_video_ctx(tmp_path),
            prompt="x",
            model=VEO,
            image_uri="gs://b/a.png",
            dry_run=True,
        )
    )
    assert "error" not in payload
    assert payload["estimated_cost"]["usd"] > 0


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_generate_video_dry_run_lite_restriction_precedes_source_error(
    tmp_path: Path,
) -> None:
    """The source check must run AFTER validate_render_options: a Lite model
    asked for first/last-frame with two bad local frames must report the mode
    restriction, not the source error — the ordering pitfall this round fixed."""
    from src.__main__ import generate_video

    payload = _payload(
        await generate_video(
            ctx=_video_ctx(tmp_path),
            prompt="x",
            model=LITE,
            image_uri=_out_of_sandbox(),
            last_frame_uri=_out_of_sandbox(),
            dry_run=True,
        )
    )
    assert "estimated_cost" not in payload
    assert "does not support first_last_frame" in payload["error"]
    assert "outside the permitted data folder" not in payload["error"]


# ---------------------------------------------------------------------------
# generate_image
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
@pytest.mark.parametrize(
    ("mode_kwargs", "named"),
    [
        pytest.param({"image_uri": None}, "image_uri", id="image_uri"),
        pytest.param(
            {"reference_image_uris": [None]}, "reference image", id="reference_image"
        ),
    ],
)
async def test_generate_image_dry_run_refuses_an_out_of_sandbox_source(
    tmp_path: Path, mode_kwargs: dict[str, Any], named: str
) -> None:
    from src.__main__ import generate_image

    bad = _out_of_sandbox()
    kwargs: dict[str, Any] = {}
    for key, value in mode_kwargs.items():
        kwargs[key] = [bad] if isinstance(value, list) else bad

    payload = _payload(
        await generate_image(
            ctx=_video_ctx(tmp_path),
            prompt="x",
            model=IMAGE_MODEL,
            dry_run=True,
            **kwargs,
        )
    )
    assert "estimated_cost" not in payload
    assert "outside the permitted data folder" in payload["error"]
    assert named in payload["error"]


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_generate_image_dry_run_refuses_a_missing_in_sandbox_source(
    tmp_path: Path,
) -> None:
    from src.__main__ import generate_image

    payload = _payload(
        await generate_image(
            ctx=_video_ctx(tmp_path),
            prompt="x",
            model=IMAGE_MODEL,
            image_uri=_missing_in_sandbox(tmp_path),
            dry_run=True,
        )
    )
    assert "estimated_cost" not in payload
    assert "image_uri" in payload["error"]
    assert "File not found" in payload["error"]


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_generate_image_dry_run_still_prices_a_gs_source(tmp_path: Path) -> None:
    from src.__main__ import generate_image

    payload = _payload(
        await generate_image(
            ctx=_video_ctx(tmp_path),
            prompt="x",
            model=IMAGE_MODEL,
            image_uri="gs://b/a.png",
            dry_run=True,
        )
    )
    assert "error" not in payload
    assert payload["estimated_cost"]["usd"] > 0


# ---------------------------------------------------------------------------
# generate_transition
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
@pytest.mark.parametrize(
    ("bad_key", "good_key"),
    [
        pytest.param("first_frame_uri", "last_frame_uri", id="first_frame_uri"),
        pytest.param("last_frame_uri", "first_frame_uri", id="last_frame_uri"),
    ],
)
async def test_generate_transition_dry_run_refuses_an_out_of_sandbox_source(
    tmp_path: Path, bad_key: str, good_key: str
) -> None:
    from src.__main__ import generate_transition

    kwargs = {bad_key: _out_of_sandbox(), good_key: "gs://b/ok.png"}
    payload = _payload(
        await generate_transition(
            ctx=_video_ctx(tmp_path), model=VEO, dry_run=True, **kwargs
        )
    )
    assert "estimated_cost" not in payload
    assert "outside the permitted data folder" in payload["error"]
    assert bad_key in payload["error"]


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_generate_transition_dry_run_refuses_a_missing_in_sandbox_source(
    tmp_path: Path,
) -> None:
    from src.__main__ import generate_transition

    payload = _payload(
        await generate_transition(
            ctx=_video_ctx(tmp_path),
            model=VEO,
            first_frame_uri=_missing_in_sandbox(tmp_path),
            last_frame_uri="gs://b/ok.png",
            dry_run=True,
        )
    )
    assert "estimated_cost" not in payload
    assert "first_frame_uri" in payload["error"]
    assert "File not found" in payload["error"]


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_generate_transition_dry_run_still_prices_gs_sources(
    tmp_path: Path,
) -> None:
    from src.__main__ import generate_transition

    payload = _payload(
        await generate_transition(
            ctx=_video_ctx(tmp_path),
            model=VEO,
            first_frame_uri="gs://b/a.png",
            last_frame_uri="gs://b/b.png",
            dry_run=True,
        )
    )
    assert "error" not in payload
    assert payload["estimated_cost"]["usd"] > 0
