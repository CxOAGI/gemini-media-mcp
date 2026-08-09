"""Round-10 deep-pass follow-ups: dry_run/real parity and input validation.

A quote must refuse what the render refuses (empty prompt, conflicting inputs),
disclose what the render warns about (over-count references), report the model
the render will report, and, for omni, disclose the clamped duration.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from tests.test_main import _video_ctx

VEO = "veo-3.1-fast-generate-001"


async def _gv(ctx: Any, **kw: Any) -> dict[str, Any]:
    from src.__main__ import generate_video

    return json.loads(await generate_video(ctx=ctx, dry_run=True, **kw))


async def _gi(ctx: Any, **kw: Any) -> dict[str, Any]:
    from src.__main__ import generate_image

    out = await generate_image(ctx=ctx, dry_run=True, **kw)
    return json.loads(out if isinstance(out, str) else out[0].text)


@pytest.mark.asyncio
async def test_dry_run_refuses_conflicting_video_inputs(tmp_path: Path) -> None:
    """The impl raises on references+frames or extend+images; the quote priced
    them and silently dropped half — it now refuses identically."""
    ctx = _video_ctx(tmp_path, vertexai=True)
    r = await _gv(ctx, prompt="x", model=VEO, reference_image_uris=["u"], image_uri="a")
    assert "error" in r and "reference images" in r["error"]
    r = await _gv(
        ctx, prompt="x", model=VEO, extend_video_uri="gs://b/v.mp4", image_uri="a"
    )
    assert "error" in r and "extend_video_uri" in r["error"]


@pytest.mark.asyncio
async def test_dry_run_refuses_empty_prompts(tmp_path: Path) -> None:
    """Both primitives priced an empty-prompt render; the composites already
    refused a blank beat/shot. Now the primitives refuse too."""
    ctx = _video_ctx(tmp_path, vertexai=True)
    assert "error" in await _gv(ctx, prompt="   ", model=VEO)
    assert "error" in await _gi(ctx, prompt="", model="gemini-3.1-flash-image")


@pytest.mark.asyncio
async def test_dry_run_warns_on_over_count_references(tmp_path: Path) -> None:
    """The render truncates over-count references with a warning; the quote was
    silent about it. Now both disclose it."""
    ctx = _video_ctx(tmp_path, vertexai=True)
    r = await _gv(ctx, prompt="x", model=VEO, reference_image_uris=list("abcde"))
    assert any("Veo 3.1 accepts 3" in w for w in r.get("warnings", []))
    r = await _gi(
        ctx,
        prompt="x",
        model="gemini-3.1-flash-image",
        reference_image_uris=[str(i) for i in range(16)],
    )
    assert any("accept 14" in w for w in r.get("warnings", []))


@pytest.mark.asyncio
async def test_dry_run_reports_the_served_model(tmp_path: Path) -> None:
    """A caller who feeds back a -preview id saw the quote say -preview while
    the Vertex render reported the resolved -001 name. They now agree."""
    ctx = _video_ctx(tmp_path, vertexai=True)
    r = await _gv(ctx, prompt="x", model="veo-3.1-fast-generate-preview")
    assert r["model"] == "veo-3.1-fast-generate-001"


@pytest.mark.asyncio
async def test_omni_dry_run_reports_the_clamped_request(tmp_path: Path) -> None:
    """generate_video and edit_video report requested + effective on a snap;
    omni's dry_run was the only one that hid the clamp."""
    from src.__main__ import generate_video_omni

    ctx = _video_ctx(tmp_path, vertexai=True)
    r = json.loads(
        await generate_video_omni(ctx=ctx, prompt="x", duration_seconds=2, dry_run=True)
    )
    assert r["requested_duration_seconds"] == 2
    assert r["duration_seconds"] == 3


@pytest.mark.asyncio
async def test_a_non_conflicting_video_request_still_prices(tmp_path: Path) -> None:
    """The new validation must not refuse a legitimate single-input call."""
    ctx = _video_ctx(tmp_path, vertexai=True)
    r = await _gv(ctx, prompt="x", model=VEO, image_uri="a")
    assert r["generation_mode"] == "image_to_video"
    assert r["estimated_cost"]["usd"] > 0
