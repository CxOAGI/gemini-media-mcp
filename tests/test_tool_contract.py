"""Contract tests for the MCP tool surface in src/__main__.py.

Every test here defends a promise the tools make to an agent driving them:
a quote never under-states the bill, a quote refuses whatever the real run
refuses, a dry-run payload describes the render it prices, all input is
validated before any money is spent, no key is silently dropped, a billed
call always reports its cost, the documented keys are the emitted keys, and
work that was paid for stays recorded even when the caller disconnects.
"""

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.__main__ import AppContext
from src.omni import OMNI_MODEL

VEO = "veo-3.1-generate-001"
VEO_FAST = "veo-3.1-fast-generate-001"


def _make_ctx(tmp_path: Path) -> MagicMock:
    """A mock MCP context whose AppContext points at real directories."""
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


def _fake_mp4_bytes() -> bytes:
    """Enough bytes to be a file on disk; nothing here decodes it."""
    return b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64


# ============================================================================
# (1) A dry run must never under-state what the render will bill
# ============================================================================


@pytest.mark.asyncio
async def test_reference_render_quote_covers_the_forced_eight_seconds(
    tmp_path: Path,
) -> None:
    """Veo forces reference-to-video to 8s. Quoting the requested 4s billed the
    caller twice what they were shown, which is the one thing a pre-flight may
    not do."""
    from src.__main__ import generate_video
    from src.pricing import actual_video_cost

    payload = json.loads(
        await generate_video(
            ctx=_make_ctx(tmp_path),
            prompt="a dog",
            model=VEO,
            duration_seconds=4,
            reference_image_uris=["file:///a.png", "file:///b.png"],
            dry_run=True,
        )
    )

    billed = actual_video_cost(VEO, 8, "720p", False, snap_duration=False)
    assert billed is not None
    assert payload["generation_mode"] == "reference_to_video"
    assert payload["requested_duration_seconds"] == 4
    assert payload["duration_seconds"] == 8
    # Compared as the tools report it: cost_to_dict rounds at the presentation
    # boundary, so a raw float tail (6 x $0.40 is 2.4000000000000004) would
    # otherwise read as a fraction of a picocent of under-quoting.
    assert payload["estimated_cost"]["usd"] >= round(billed.usd, 6)


@pytest.mark.asyncio
async def test_extension_quote_matches_the_seven_seconds_veo_renders(
    tmp_path: Path,
) -> None:
    """An extension outputs exactly 7s; the generic 4/6/8 snap quoted 8s for a
    render that bills 7."""
    from src.__main__ import generate_video
    from src.pricing import actual_video_cost

    payload = json.loads(
        await generate_video(
            ctx=_make_ctx(tmp_path),
            prompt="keep going",
            model=VEO,
            duration_seconds=8,
            extend_video_uri="file:///clip.mp4",
            dry_run=True,
        )
    )

    billed = actual_video_cost(VEO, 7, "720p", False, snap_duration=False)
    assert billed is not None
    assert payload["generation_mode"] == "extend_video"
    assert payload["duration_seconds"] == 7
    assert payload["estimated_cost"]["usd"] == pytest.approx(billed.usd)


@pytest.mark.asyncio
async def test_plain_text_to_video_quote_still_snaps_the_ordinary_way(
    tmp_path: Path,
) -> None:
    """Threading the mode through must not disturb the default path: 5s still
    snaps down to 4s, ties included."""
    from src.__main__ import generate_video
    from src.pricing import actual_video_cost

    payload = json.loads(
        await generate_video(
            ctx=_make_ctx(tmp_path),
            prompt="a dog",
            model=VEO,
            duration_seconds=5,
            dry_run=True,
        )
    )

    billed = actual_video_cost(VEO, 4, "720p", False, snap_duration=False)
    assert billed is not None
    assert payload["generation_mode"] == "text_to_video"
    assert payload["duration_seconds"] == 4
    assert payload["estimated_cost"]["usd"] == pytest.approx(billed.usd)


# ============================================================================
# (2) A quote must refuse everything the real run refuses
# ============================================================================


@pytest.mark.asyncio
async def test_omni_quote_refuses_an_aspect_ratio_the_render_refuses(
    tmp_path: Path,
) -> None:
    """generate_video_omni quoted 1:1 cleanly and then raised on the real run,
    so an agent budgeted for a render it can never get."""
    from src.__main__ import generate_video_omni

    payload = json.loads(
        await generate_video_omni(
            ctx=_make_ctx(tmp_path),
            prompt="a dog",
            aspect_ratio="1:1",
            dry_run=True,
        )
    )

    assert "estimated_cost" not in payload
    assert "Unsupported aspect_ratio" in payload["error"]


@pytest.mark.asyncio
async def test_edit_video_quote_refuses_an_aspect_ratio_the_render_refuses(
    tmp_path: Path,
) -> None:
    """src/omni.py rejects an unsupported ratio on an edit before it decides
    what to send, so edit_video's quote has to reject it too."""
    from src.__main__ import edit_video

    payload = json.loads(
        await edit_video(
            ctx=_make_ctx(tmp_path),
            previous_interaction_id="interactions/1",
            prompt="make it stormy",
            aspect_ratio="4:3",
            dry_run=True,
        )
    )

    assert "estimated_cost" not in payload
    assert "Unsupported aspect_ratio" in payload["error"]


@pytest.mark.asyncio
async def test_supported_ratios_still_quote_on_both_omni_tools(
    tmp_path: Path,
) -> None:
    """The guard must reject only what omni rejects — a 9:16 quote is exactly
    what these tools exist to answer."""
    from src.__main__ import edit_video, generate_video_omni

    quote = json.loads(
        await generate_video_omni(
            ctx=_make_ctx(tmp_path), prompt="a dog", aspect_ratio="9:16", dry_run=True
        )
    )
    assert quote["estimated_cost"]["usd"] > 0

    edit_quote = json.loads(
        await edit_video(
            ctx=_make_ctx(tmp_path),
            previous_interaction_id="interactions/1",
            prompt="make it stormy",
            aspect_ratio="9:16",
            dry_run=True,
        )
    )
    assert edit_quote["estimated_cost"]["usd"] > 0


# ============================================================================
# (3) A dry-run payload must describe the render it prices
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["generate_transition", "generate_bridge"])
async def test_sibling_quotes_report_the_duration_they_price(
    tmp_path: Path, tool_name: str
) -> None:
    """Both reported the raw request (5) beside a cost detailing "4s of video",
    so the payload contradicted both the price and its own docstring."""
    import src.__main__ as main_module

    tool = getattr(main_module, tool_name)
    kwargs: dict[str, Any] = (
        {"first_frame_uri": "file:///a.png", "last_frame_uri": "file:///b.png"}
        if tool_name == "generate_transition"
        else {"from_clip_uri": "file:///a.mp4", "to_clip_uri": "file:///b.mp4"}
    )

    payload = json.loads(
        await tool(
            ctx=_make_ctx(tmp_path),
            model=VEO_FAST,
            duration_seconds=5,
            dry_run=True,
            **kwargs,
        )
    )

    assert payload["requested_duration_seconds"] == 5
    assert payload["duration_seconds"] == 4
    assert "4s of video" in payload["estimated_cost"]["detail"]


# ============================================================================
# (4) Every statically checkable beat field is validated before any render
# ============================================================================


@pytest.mark.asyncio
async def test_a_bad_seed_in_a_later_beat_costs_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-integer seed only failed inside src/video.py's seed comparison —
    after every earlier beat had rendered and billed."""
    from src.__main__ import generate_clip

    rendered: list[str] = []

    async def mock_impl(**kwargs: Any) -> dict[str, Any]:
        rendered.append(kwargs.get("prompt", ""))
        raise AssertionError("no beat may render when a later beat is invalid")

    monkeypatch.setattr("src.__main__.generate_video_impl", mock_impl)

    payload = json.loads(
        await generate_clip(
            ctx=_make_ctx(tmp_path),
            beats=[
                {"prompt": "one"},
                {"prompt": "two"},
                {"prompt": "three", "seed": "not-a-number"},
            ],
        )
    )

    assert rendered == []
    assert "beats[2].seed" in payload["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("bad_beat", "expected"),
    [
        pytest.param(
            {"prompt": "x", "first_frame_uri": 5}, "first_frame_uri", id="uri_type"
        ),
        pytest.param(
            {"prompt": "x", "first_frame_uri": "  "}, "first_frame_uri", id="uri_empty"
        ),
        pytest.param(
            {"prompt": "x", "negative_prompt": ["a"]}, "negative_prompt", id="negative"
        ),
        pytest.param({"prompt": "x", "audio_prompt": 3}, "audio_prompt", id="audio"),
        pytest.param({"prompt": "x", "seed": 1.5}, "seed", id="fractional_seed"),
    ],
)
async def test_statically_invalid_beat_fields_are_refused_up_front(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_beat: dict[str, Any],
    expected: str,
) -> None:
    """Each of these is decidable without the network, so none of them may
    surface after a render has been paid for."""
    from src.__main__ import generate_clip

    async def mock_impl(**kwargs: Any) -> dict[str, Any]:
        raise AssertionError("no beat may render when a beat is statically invalid")

    monkeypatch.setattr("src.__main__.generate_video_impl", mock_impl)

    payload = json.loads(
        await generate_clip(ctx=_make_ctx(tmp_path), beats=[{"prompt": "ok"}, bad_beat])
    )

    assert f"beats[1].{expected}" in payload["error"]


@pytest.mark.asyncio
async def test_a_negative_seed_is_still_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """src/video.py drops a negative seed with a warning rather than failing,
    so the pre-flight must not turn a working call into an error."""
    from src.__main__ import generate_clip

    videos_dir = tmp_path / "videos"

    async def mock_impl(**kwargs: Any) -> dict[str, Any]:
        out = videos_dir / "beat.mp4"
        out.write_bytes(_fake_mp4_bytes())
        return {
            "video_url": f"file://{out}",
            "model": kwargs.get("model", VEO_FAST),
            "duration_seconds": 4,
            "generation_mode": "text_to_video",
        }

    monkeypatch.setattr("src.__main__.generate_video_impl", mock_impl)

    payload = json.loads(
        await generate_clip(
            ctx=_make_ctx(tmp_path), beats=[{"prompt": "ok", "seed": -1}]
        )
    )

    assert payload["kind"] == "clip"
    assert len(payload["segments"]) == 1


# ============================================================================
# (5) Unknown keys in a beat / shot are refused, not dropped
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("wrong_key", ["image_url", "image_uri", "first_frame"])
async def test_a_misnamed_frame_key_is_refused_with_the_right_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, wrong_key: str
) -> None:
    """Chaining generate_storyboard's image_url (or copying generate_video's
    image_uri) into a beat bought a full-price text-to-video render with no
    conditioning and no diagnostic."""
    from src.__main__ import generate_clip

    async def mock_impl(**kwargs: Any) -> dict[str, Any]:
        raise AssertionError("an unknown beat key must not reach a render")

    monkeypatch.setattr("src.__main__.generate_video_impl", mock_impl)

    payload = json.loads(
        await generate_clip(
            ctx=_make_ctx(tmp_path),
            beats=[{"prompt": "x", wrong_key: "file:///frame.png"}],
        )
    )

    assert wrong_key in payload["error"]
    assert "first_frame_uri" in payload["error"]


@pytest.mark.asyncio
async def test_a_storyboard_shot_list_still_feeds_generate_clip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """generate_storyboard documents its shots as feedable straight in as
    beats, so caption/notes have to stay acceptable."""
    from src.__main__ import generate_clip

    videos_dir = tmp_path / "videos"

    async def mock_impl(**kwargs: Any) -> dict[str, Any]:
        out = videos_dir / "beat.mp4"
        out.write_bytes(_fake_mp4_bytes())
        return {
            "video_url": f"file://{out}",
            "model": kwargs.get("model", VEO_FAST),
            "duration_seconds": 4,
            "generation_mode": "text_to_video",
        }

    monkeypatch.setattr("src.__main__.generate_video_impl", mock_impl)

    payload = json.loads(
        await generate_clip(
            ctx=_make_ctx(tmp_path),
            beats=[
                {
                    "prompt": "wide establishing shot",
                    "caption": "INT. KITCHEN - DAY",
                    "notes": "slow push in",
                    "duration_seconds": 4,
                }
            ],
        )
    )

    assert payload["kind"] == "clip"
    assert len(payload["segments"]) == 1


@pytest.mark.asyncio
async def test_an_unknown_shot_key_is_refused_before_any_frame_is_billed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A storyboard bills per shot, so a dropped field is a paid-for board that
    silently ignored what was asked."""
    from src.__main__ import generate_storyboard

    async def mock_impl(**kwargs: Any) -> dict[str, Any]:
        raise AssertionError("an unknown shot key must not reach a render")

    monkeypatch.setattr("src.__main__.generate_image_impl", mock_impl)

    result = await generate_storyboard(
        ctx=_make_ctx(tmp_path),
        shots=[{"prompt": "a kitchen", "duration": 4}],
    )
    payload = json.loads(result[0].text)

    assert "duration" in payload["error"]
    assert "duration_seconds" in payload["error"]


# ============================================================================
# (6) A billed call always reports what it cost
# ============================================================================


@pytest.mark.asyncio
async def test_a_text_only_image_response_reports_its_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A safety refusal or clarifying question is metered and charged. The
    early return skipped pricing entirely, so an agent totalling spend
    under-counted every refusal."""
    from src.__main__ import generate_image
    from src.pricing import actual_image_cost

    usage = {
        "prompt_token_count": 14,
        "candidates_token_count": 30,
        "candidates_tokens_details": [{"modality": "TEXT", "token_count": 30}],
    }

    async def mock_impl(**kwargs: Any) -> dict[str, Any]:
        return {
            "message": "Model returned text only",
            "generated_text": "I can't generate that.",
            "model": "gemini-3.1-flash-image",
            "usage": usage,
        }

    monkeypatch.setattr("src.__main__.generate_image_impl", mock_impl)

    result = await generate_image(
        ctx=_make_ctx(tmp_path),
        prompt="something refused",
        model="gemini-3.1-flash-image",
    )
    payload = json.loads(result[0].text)

    expected = actual_image_cost("gemini-3.1-flash-image", usage, "1K", 0)
    assert expected is not None
    assert payload["usage"] == usage
    assert payload["cost"]["usd"] == pytest.approx(round(expected.usd, 6))
    assert payload["cost"]["usd"] > 0


# ============================================================================
# (7) The documented keys are the emitted keys
# ============================================================================


@pytest.mark.asyncio
async def test_a_draft_render_reports_its_generation_mode_and_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """generate_video's Returns block names generation_mode, and every
    non-draft response carries prompt; the omni-backed draft path emitted
    neither."""
    from src.__main__ import generate_video

    async def mock_omni(app_ctx: Any, ctx: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "message": "Video generated successfully",
            "video_url": "file:///draft.mp4",
            "model": OMNI_MODEL,
            "duration_seconds": 6,
        }

    monkeypatch.setattr("src.__main__._omni_generate_and_manifest", mock_omni)

    payload = json.loads(
        await generate_video(
            ctx=_make_ctx(tmp_path),
            prompt="a dog",
            model=VEO,
            audio_prompt="barking",
            draft=True,
        )
    )

    assert payload["generation_mode"] == "draft"
    assert payload["prompt"] == "a dog\nAudio: barking"


# ============================================================================
# (8) Work already paid for survives the caller disconnecting
# ============================================================================


@pytest.mark.asyncio
async def test_a_cancelled_clip_still_records_what_it_rendered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CancelledError is a BaseException, so the tool's except-Exception
    handler never saw it and the clip-level total — the only record of what N
    billed beats cost — died with the request."""
    from src.__main__ import generate_clip

    videos_dir = tmp_path / "videos"
    calls = {"n": 0}

    async def mock_impl(**kwargs: Any) -> dict[str, Any]:
        calls["n"] += 1
        if calls["n"] > 1:
            raise asyncio.CancelledError()
        out = videos_dir / "beat1.mp4"
        out.write_bytes(_fake_mp4_bytes())
        return {
            "video_url": f"file://{out}",
            "model": VEO_FAST,
            "duration_seconds": 4,
            "generation_mode": "text_to_video",
        }

    monkeypatch.setattr("src.__main__.generate_video_impl", mock_impl)

    with pytest.raises(asyncio.CancelledError):
        await generate_clip(
            ctx=_make_ctx(tmp_path),
            beats=[{"prompt": "one"}, {"prompt": "two"}],
        )

    written = list(videos_dir.glob("*_clip.json"))
    assert len(written) == 1
    manifest = json.loads(written[0].read_text())
    assert manifest["cancelled"] is True
    assert len(manifest["segments"]) == 1
    assert manifest["cost"]["usd"] > 0


@pytest.mark.asyncio
async def test_a_cancelled_extension_chain_still_records_its_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """loop_extend lost the whole chain's manifest on a disconnect, including
    the cost of extensions that had already rendered."""
    from src.__main__ import loop_extend

    videos_dir = tmp_path / "videos"
    calls = {"n": 0}

    async def mock_impl(**kwargs: Any) -> dict[str, Any]:
        calls["n"] += 1
        if calls["n"] > 2:
            raise asyncio.CancelledError()
        out = videos_dir / f"ext{calls['n']}.mp4"
        out.write_bytes(_fake_mp4_bytes())
        return {"video_url": f"file://{out}", "model": VEO, "duration_seconds": 7}

    monkeypatch.setattr("src.__main__.generate_video_impl", mock_impl)

    with pytest.raises(asyncio.CancelledError):
        await loop_extend(
            ctx=_make_ctx(tmp_path), video_uri="file:///source.mp4", times=4
        )

    manifest = json.loads((videos_dir / "ext2.json").read_text())
    assert manifest["cancelled"] is True
    assert manifest["times"] == 2
    assert len(manifest["extension_steps"]) == 2
    assert manifest["cost"]["usd"] > 0


# ============================================================================
# The invariant all of the above serves
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("requested", [0, 3, 4, 5, 6, 7, 8, 12])
@pytest.mark.parametrize(
    ("mode", "mode_kwargs"),
    [
        pytest.param("text_to_video", {}, id="text"),
        pytest.param("image_to_video", {"image_uri": "file:///a.png"}, id="image"),
        pytest.param(
            "first_last_frame",
            {"image_uri": "file:///a.png", "last_frame_uri": "file:///b.png"},
            id="first_last",
        ),
        pytest.param(
            "reference_to_video",
            {"reference_image_uris": ["file:///a.png"]},
            id="reference",
        ),
        pytest.param(
            "extend_video", {"extend_video_uri": "file:///c.mp4"}, id="extend"
        ),
    ],
)
async def test_a_quote_never_falls_below_the_metered_bill(
    tmp_path: Path, mode: str, mode_kwargs: dict[str, Any], requested: float
) -> None:
    """The invariant the whole dry_run surface rests on, checked across every
    mode and every duration the snap treats differently."""
    from src.__main__ import generate_video
    from src.pricing import actual_video_cost, snap_video_duration

    payload = json.loads(
        await generate_video(
            ctx=_make_ctx(tmp_path),
            prompt="a dog",
            model=VEO,
            duration_seconds=requested,
            dry_run=True,
            **mode_kwargs,
        )
    )

    # The expected render length is written out from the DOCUMENTED rule, not
    # computed with snap_video_duration — deriving it from the same function
    # the quote uses makes the assertion self-referential: a wrong snap rule
    # moves both sides together and the test passes anyway.
    if mode == "reference_to_video":
        rendered = 8.0  # reference mode always renders 8s
    elif mode == "extend_video":
        rendered = 7.0  # an extension always adds 7s
    else:
        rendered = min((4.0, 6.0, 8.0), key=lambda s: (abs(s - requested), s))
    assert snap_video_duration(VEO, requested, mode) == rendered, (
        "the documented snap rule and the implementation have diverged"
    )

    billed = actual_video_cost(VEO, rendered, "720p", False, snap_duration=False)
    assert billed is not None
    assert payload["generation_mode"] == mode
    assert payload["duration_seconds"] == rendered
    assert payload["estimated_cost"]["usd"] >= round(billed.usd, 6)


# ============================================================================
# The planner and the tools must still agree to the cent
# ============================================================================


@pytest.mark.asyncio
async def test_the_planner_still_agrees_with_generate_videos_quote(
    tmp_path: Path,
) -> None:
    """Threading the mode through the quote must not move the plain
    text-to-video figure the planner recommends against."""
    from src.__main__ import generate_video
    from src.routing import RoutingConstraints, plan_generation

    plan = plan_generation(
        "an 8 second video of a dog",
        RoutingConstraints(media_kind="video", duration_seconds=8.0),
    )
    route = next(r for r in plan.routes if r.tool == "generate_video")
    assert route.cost is not None

    payload = json.loads(
        await generate_video(
            ctx=_make_ctx(tmp_path),
            prompt="an 8 second video of a dog",
            model=route.model,
            duration_seconds=route.params["duration_seconds"],
            include_audio=route.params.get("include_audio", False),
            resolution=route.params.get("resolution"),
            dry_run=True,
        )
    )

    assert payload["estimated_cost"]["usd"] == pytest.approx(round(route.cost.usd, 6))
