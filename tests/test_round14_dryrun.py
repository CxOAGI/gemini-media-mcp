"""Round 14: a dry run must validate local file sources.

The project invariant is that a quote refuses everything the real run refuses.
A local file source that is missing or outside DATA_FOLDER is exactly what
fetch() rejects on the real run, yet dry_run used to price it — so a quote
succeeded for a call guaranteed to fail. These pin that a nonexistent or
out-of-sandbox local source is refused on a quote, with the same detailed
confinement / not-found message the real run emits, while a gs:// source
(uncheckable offline) still prices.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.__main__ import AppContext

VEO = "veo-3.1-generate-001"
VEO_FAST = "veo-3.1-fast-generate-001"


def _make_ctx(tmp_path: Path) -> MagicMock:
    """A mock MCP context whose AppContext points at real directories.

    tmp_path is DATA_FOLDER, so a file written under it is a legitimate local
    source and a path elsewhere is out-of-sandbox.
    """
    images_dir = tmp_path / "images"
    videos_dir = tmp_path / "videos"
    images_dir.mkdir(exist_ok=True)
    videos_dir.mkdir(exist_ok=True)

    client = MagicMock()
    client._api_client.vertexai = False

    ctx = MagicMock()
    ctx.info = AsyncMock()
    ctx.error = AsyncMock()
    ctx.warning = AsyncMock()
    ctx.request_context.lifespan_context = AppContext(
        data_folder=tmp_path,
        images_dir=images_dir,
        videos_dir=videos_dir,
        client=client,
    )
    return ctx


# ============================================================================
# loop_extend
# ============================================================================


@pytest.mark.asyncio
async def test_loop_extend_dry_run_refuses_a_missing_local_source(
    tmp_path: Path,
) -> None:
    """A local video_uri inside DATA_FOLDER that does not exist is refused with
    the not-found form, not priced."""
    from src.__main__ import loop_extend

    missing = f"file://{tmp_path / 'videos' / 'missing.mp4'}"
    payload = json.loads(
        await loop_extend(
            ctx=_make_ctx(tmp_path), video_uri=missing, times=2, dry_run=True
        )
    )
    assert "estimated_cost" not in payload
    assert "File not found" in payload["error"]
    assert "video_uri" in payload["error"]


@pytest.mark.asyncio
async def test_loop_extend_dry_run_refuses_an_out_of_sandbox_source(
    tmp_path: Path,
) -> None:
    """A local video_uri outside DATA_FOLDER is refused with the confinement
    form (names DATA_FOLDER and the remedy), not priced."""
    from src.__main__ import loop_extend

    payload = json.loads(
        await loop_extend(
            ctx=_make_ctx(tmp_path),
            video_uri="file:///outside.mp4",
            times=2,
            dry_run=True,
        )
    )
    assert "estimated_cost" not in payload
    assert "DATA_FOLDER" in payload["error"]
    assert "outside the permitted data folder" in payload["error"]


@pytest.mark.asyncio
async def test_loop_extend_dry_run_still_prices_a_gs_source(tmp_path: Path) -> None:
    """A gs:// source cannot be checked offline, so a quote still prices it."""
    from src.__main__ import loop_extend

    payload = json.loads(
        await loop_extend(
            ctx=_make_ctx(tmp_path),
            video_uri="gs://bucket/x.mp4",
            times=2,
            dry_run=True,
        )
    )
    assert "error" not in payload
    assert payload["estimated_cost"]["usd"] > 0


# ============================================================================
# generate_bridge
# ============================================================================


@pytest.mark.asyncio
async def test_generate_bridge_dry_run_refuses_a_missing_local_source(
    tmp_path: Path,
) -> None:
    """A local from_clip_uri inside DATA_FOLDER that does not exist is refused
    with the not-found form, not priced."""
    from src.__main__ import generate_bridge

    missing = f"file://{tmp_path / 'videos' / 'missing.mp4'}"
    payload = json.loads(
        await generate_bridge(
            ctx=_make_ctx(tmp_path),
            from_clip_uri=missing,
            to_clip_uri="gs://bucket/ok.mp4",
            model=VEO_FAST,
            dry_run=True,
        )
    )
    assert "estimated_cost" not in payload
    assert "File not found" in payload["error"]
    assert "from_clip_uri" in payload["error"]


@pytest.mark.asyncio
async def test_generate_bridge_dry_run_refuses_an_out_of_sandbox_source(
    tmp_path: Path,
) -> None:
    """A local to_clip_uri outside DATA_FOLDER is refused with the confinement
    form, not priced."""
    from src.__main__ import generate_bridge

    payload = json.loads(
        await generate_bridge(
            ctx=_make_ctx(tmp_path),
            from_clip_uri="gs://bucket/ok.mp4",
            to_clip_uri="file:///outside.mp4",
            model=VEO_FAST,
            dry_run=True,
        )
    )
    assert "estimated_cost" not in payload
    assert "DATA_FOLDER" in payload["error"]
    assert "outside the permitted data folder" in payload["error"]
    assert "to_clip_uri" in payload["error"]


@pytest.mark.asyncio
async def test_generate_bridge_dry_run_still_prices_gs_sources(
    tmp_path: Path,
) -> None:
    """gs:// clips cannot be checked offline, so a quote still prices them."""
    from src.__main__ import generate_bridge

    payload = json.loads(
        await generate_bridge(
            ctx=_make_ctx(tmp_path),
            from_clip_uri="gs://bucket/a.mp4",
            to_clip_uri="gs://bucket/b.mp4",
            model=VEO_FAST,
            dry_run=True,
        )
    )
    assert "error" not in payload
    assert payload["estimated_cost"]["usd"] > 0


# ============================================================================
# generate_clip (per-beat first_frame_uri)
# ============================================================================


@pytest.mark.asyncio
async def test_generate_clip_dry_run_refuses_a_missing_beat_frame(
    tmp_path: Path,
) -> None:
    """A beat's local first_frame_uri inside DATA_FOLDER that does not exist is
    refused with the not-found form, naming the beat — not priced."""
    from src.__main__ import generate_clip

    missing = f"file://{tmp_path / 'images' / 'missing.png'}"
    payload = json.loads(
        await generate_clip(
            ctx=_make_ctx(tmp_path),
            beats=[{"prompt": "a", "first_frame_uri": missing}],
            model=VEO_FAST,
            dry_run=True,
        )
    )
    assert "estimated_cost" not in payload
    assert "File not found" in payload["error"]
    assert "beats[0].first_frame_uri" in payload["error"]


@pytest.mark.asyncio
async def test_generate_clip_dry_run_refuses_an_out_of_sandbox_beat_frame(
    tmp_path: Path,
) -> None:
    """A beat's local first_frame_uri outside DATA_FOLDER is refused with the
    confinement form, not priced."""
    from src.__main__ import generate_clip

    payload = json.loads(
        await generate_clip(
            ctx=_make_ctx(tmp_path),
            beats=[{"prompt": "a", "first_frame_uri": "file:///outside.png"}],
            model=VEO_FAST,
            dry_run=True,
        )
    )
    assert "estimated_cost" not in payload
    assert "DATA_FOLDER" in payload["error"]
    assert "outside the permitted data folder" in payload["error"]


@pytest.mark.asyncio
async def test_generate_clip_dry_run_still_prices_a_gs_beat_frame(
    tmp_path: Path,
) -> None:
    """A beat's gs:// first_frame_uri cannot be checked offline, so a quote
    still prices the reel."""
    from src.__main__ import generate_clip

    payload = json.loads(
        await generate_clip(
            ctx=_make_ctx(tmp_path),
            beats=[{"prompt": "a", "first_frame_uri": "gs://bucket/f.png"}],
            model=VEO_FAST,
            dry_run=True,
        )
    )
    assert "error" not in payload
    assert payload["estimated_cost"]["usd"] > 0
