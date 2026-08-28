"""gemini-omni-1.1-flash: the capabilities the preview model does not have.

The wire shapes are pinned in tests/test_wire_contract_omni.py. This file
covers everything either side of the wire: which requests are refused before
anything is spent, how the role declarations are built, what the new
resolutions cost, and what the tools quote for a render nobody has paid for
yet.

The recurring hazard is a SILENT downgrade. Every argument here exists only on
1.1, so the failure mode for each is the same: the preview model accepts the
call, renders something else, and the caller is billed for what they asked for
rather than what they got. Hence a refusal — never a dropped field — for every
one of them.
"""

import json
from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.__main__ import AppContext
from src.omni import (
    OMNI_1_1_MODEL,
    OMNI_PREVIEW_MODEL,
    OMNI_RESOLUTIONS,
    _build_media_role_prompt,  # pyright: ignore[reportPrivateUsage]
    _prepare_video_part,  # pyright: ignore[reportPrivateUsage]
    is_omni_model,
    normalize_omni_resolution,
    omni_spec,
    prompt_declares_media_roles,
    validate_omni_request,
)

_MP4_HEADER = b"\x00\x00\x00\x18ftypmp42"


# ============================================================================
# The model table
# ============================================================================


def test_both_models_are_omni_and_nothing_else_is() -> None:
    assert is_omni_model(OMNI_PREVIEW_MODEL)
    assert is_omni_model(OMNI_1_1_MODEL)
    assert not is_omni_model("veo-3.1-fast-generate-001")
    assert not is_omni_model(None)


def test_the_preview_model_advertises_none_of_the_new_capabilities() -> None:
    """The table is what every refusal below reads, so pin it directly."""
    spec = omni_spec(OMNI_PREVIEW_MODEL)
    assert not spec.supports_resolution
    assert not spec.supports_extend
    assert not spec.supports_keyframes
    assert not spec.supports_role_tags
    assert spec.max_reference_videos == 0
    assert spec.rendered_resolution == "720p"


def test_the_1_1_spec_matches_the_published_limits() -> None:
    """Every number here is quoted from the reference, not chosen."""
    spec = omni_spec(OMNI_1_1_MODEL)
    assert spec.resolutions == OMNI_RESOLUTIONS == ("360p", "720p", "1080p", "4K")
    assert spec.rendered_resolution == "720p"
    # "Video references support a maximum of 3 clips, up to 3 seconds each."
    assert spec.max_reference_videos == 3
    assert spec.max_reference_video_seconds == 3.0
    # "You can extend videos by 10s, up to a total length of 40s."
    assert spec.extension_step_seconds == 10
    assert spec.max_extended_seconds == 40
    # "Input videos for editing and extension must be 10 seconds or less when
    # uploading."
    assert spec.max_uploaded_source_seconds == 10.0


def test_an_unknown_model_is_named_not_defaulted() -> None:
    with pytest.raises(ValueError, match="Unsupported.*|Unknown omni model"):
        _ = omni_spec("gemini-omni-2-flash")


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("360p", "360p"),
        ("360P", "360p"),
        ("4k", "4K"),
        ("4K", "4K"),
        ("2160p", "4K"),
        ("1080P", "1080p"),
        ("bogus", None),
        (None, None),
    ],
)
def test_resolution_spellings_normalize(
    given: str | None, expected: str | None
) -> None:
    assert normalize_omni_resolution(given) == expected


# ============================================================================
# Refusals: every 1.1-only argument on the preview model
# ============================================================================


@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        pytest.param({"resolution": "4K"}, "no resolution parameter", id="resolution"),
        pytest.param({"task": "extend"}, "extend task", id="extend_task"),
        pytest.param(
            {"has_first_frame": True}, "explicit first-frame", id="first_frame"
        ),
        pytest.param(
            {"reference_image_count": 2}, "explicit first-frame", id="image_refs"
        ),
        pytest.param(
            {"reference_video_count": 1},
            "does not accept video references",
            id="video_refs",
        ),
    ],
)
def test_the_preview_model_refuses_what_it_cannot_do(
    kwargs: dict[str, Any], fragment: str
) -> None:
    """A dropped field would render 720p and bill 4K. Refuse instead."""
    with pytest.raises(ValueError, match=fragment):
        _ = validate_omni_request(OMNI_PREVIEW_MODEL, **kwargs)
    # And the message points at the model that CAN do it.
    try:
        _ = validate_omni_request(OMNI_PREVIEW_MODEL, **kwargs)
    except ValueError as exc:
        assert OMNI_1_1_MODEL in str(exc)


def test_a_fourth_reference_video_is_refused() -> None:
    """Three clips is a hard documented ceiling, so a fourth has nowhere to go."""
    with pytest.raises(ValueError, match="at most 3"):
        _ = validate_omni_request(OMNI_1_1_MODEL, reference_video_count=4)
    assert validate_omni_request(OMNI_1_1_MODEL, reference_video_count=3)


def test_a_last_frame_without_a_first_is_refused() -> None:
    """<LAST_FRAME> is documented as only usable with <FIRST_FRAME>."""
    with pytest.raises(ValueError, match="requires a first frame"):
        _ = validate_omni_request(OMNI_1_1_MODEL, has_last_frame=True)


def test_inferred_and_explicit_image_roles_cannot_be_mixed() -> None:
    """The two describe the same images two ways; one of them has to lose."""
    with pytest.raises(ValueError, match="cannot be combined"):
        _ = validate_omni_request(
            OMNI_1_1_MODEL, inferred_image_count=1, reference_image_count=1
        )


def test_an_unsupported_resolution_is_refused_on_1_1_too() -> None:
    with pytest.raises(ValueError, match="Unsupported resolution '8K'"):
        _ = validate_omni_request(OMNI_1_1_MODEL, resolution="8K")


def test_an_unknown_task_is_refused() -> None:
    with pytest.raises(ValueError, match="Unknown task 'upscale'"):
        _ = validate_omni_request(OMNI_1_1_MODEL, task="upscale")


# ============================================================================
# The generated role declarations
# ============================================================================


def test_one_inferred_image_gets_no_declaration() -> None:
    """The reference's advice: prompt normally, reach for tags when that fails.

    A single image's role is unambiguous, so a declaration would add
    constraints without resolving anything.
    """
    assert (
        _build_media_role_prompt(
            "a cat",
            has_first_frame=True,
            has_last_frame=False,
            reference_image_count=0,
            has_source_video=False,
            reference_video_count=0,
        )
        == "a cat"
    )


def test_a_keyframe_pair_declares_both_roles_and_closes_with_an_instruction() -> None:
    built = _build_media_role_prompt(
        "sunrise to snowfall",
        has_first_frame=True,
        has_last_frame=True,
        reference_image_count=0,
        has_source_video=False,
        reference_video_count=0,
    )
    assert built == (
        "[# Sources <FIRST_FRAME>@Image1 <LAST_FRAME>@Image2] sunrise to snowfall "
        "Use Image1 as the first frame and Image2 as the last frame."
    )


def test_image_numbering_runs_frames_then_references() -> None:
    """@ImageN is a position in the input list, so the order is the contract."""
    built = _build_media_role_prompt(
        "a woman walks",
        has_first_frame=True,
        has_last_frame=False,
        reference_image_count=2,
        has_source_video=False,
        reference_video_count=0,
    )
    assert "[# Sources <FIRST_FRAME>@Image1]" in built
    assert "[# References <IMAGE_REF_0>@Image2 <IMAGE_REF_1>@Image3]" in built


def test_a_source_video_is_video_0_and_references_follow_it() -> None:
    built = _build_media_role_prompt(
        "make the mirror ripple",
        has_first_frame=False,
        has_last_frame=False,
        reference_image_count=0,
        has_source_video=True,
        reference_video_count=2,
    )
    assert "[# Sources <VIDEO_0>@Video1]" in built
    assert "[# References <VIDEO_REF_0>@Video2 <VIDEO_REF_1>@Video3]" in built
    assert "Do not use them as a source for video editing." in built


@pytest.mark.parametrize(
    "prompt",
    [
        "<FIRST_FRAME> a woman is walking",
        "the person in <VIDEO_REF_0> plays violin",
        "[# Sources <FIRST_FRAME>@Image1] a woman",
        "in the style of <IMAGE_REF_0> a woman",
    ],
)
def test_a_prompt_that_already_binds_its_media_is_detected(prompt: str) -> None:
    assert prompt_declares_media_roles(prompt)


def test_an_ordinary_prompt_is_not_mistaken_for_a_declaration() -> None:
    assert not prompt_declares_media_roles("a cat < a dog, in a [bracketed] scene")


# ============================================================================
# Input videos: inline vs the Files API
# ============================================================================


class _FakeFiles:
    """Records what the Files API was asked to do."""

    def __init__(self, states: list[str]) -> None:
        self.states = states
        self.uploaded: bytes | None = None
        self.mime_type: str | None = None

    def upload(self, *, file: BytesIO, config: dict[str, str]) -> Any:
        self.uploaded = file.read()
        self.mime_type = config["mime_type"]
        return MagicMock(name_="files/abc", uri="files/abc", state=self.states.pop(0))

    def get(self, *, name: str) -> Any:
        return MagicMock(uri="files/abc", name=name, state=self.states.pop(0))


async def _prepare(data: bytes, model: str, files: Any) -> dict[str, Any]:
    client = MagicMock()
    client.files = files
    warnings: list[str] = []

    async def run(func: Any, /, **kwargs: Any) -> Any:
        return func(**kwargs)

    part = await _prepare_video_part(
        client,
        data,
        "input_video",
        spec=omni_spec(model),
        log_callback=None,
        run_within_deadline=run,
        deadline_expired=lambda: False,
        warnings=warnings,
    )
    part["_warnings"] = warnings
    return part


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_a_small_video_still_rides_inline() -> None:
    files = _FakeFiles([])
    part = await _prepare(_MP4_HEADER + b"tiny", OMNI_1_1_MODEL, files)
    assert part["type"] == "video"
    assert files.uploaded is None
    assert part["_warnings"] == []


@pytest.mark.asyncio
@pytest.mark.timeout(10.0)
async def test_a_large_video_is_uploaded_and_referenced_as_a_document() -> None:
    """Base64 inflates by a third, so a big clip cannot ride in the body.

    The reference's own answer is the Files API and a `document` part; without
    it the request is simply too large and the extension feature is unusable
    on real footage.
    """
    big = _MP4_HEADER + b"x" * (9 * 1024 * 1024)
    files = _FakeFiles(["PROCESSING", "ACTIVE"])
    part = await _prepare(big, OMNI_1_1_MODEL, files)
    assert part["type"] == "document"
    assert part["uri"] == "files/abc"
    assert files.uploaded == big
    assert files.mime_type == "video/mp4"


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_the_preview_model_warns_instead_of_uploading() -> None:
    """It documents no `document` part, so sending one would be a guess.

    The call is left exactly as it was and the caller is told why it may be
    refused — and which model does handle it.
    """
    big = _MP4_HEADER + b"x" * (9 * 1024 * 1024)
    files = _FakeFiles([])
    part = await _prepare(big, OMNI_PREVIEW_MODEL, files)
    assert part["type"] == "video"
    assert files.uploaded is None
    assert OMNI_1_1_MODEL in part["_warnings"][0]


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_a_failed_upload_is_not_silently_referenced() -> None:
    files = _FakeFiles(["PROCESSING", "FAILED"])
    with pytest.raises(ValueError, match="FAILED"):
        _ = await _prepare(
            _MP4_HEADER + b"x" * (9 * 1024 * 1024), OMNI_1_1_MODEL, files
        )


# ============================================================================
# Pricing
# ============================================================================


def test_360p_is_quoted_at_a_third_of_720p() -> None:
    """The launch post's figure, and the only statement about the draft tier."""
    from src.pricing import estimate_video_cost

    draft = estimate_video_cost(OMNI_1_1_MODEL, 6, "360p")
    standard = estimate_video_cost(OMNI_1_1_MODEL, 6, "720p")
    assert draft is not None and standard is not None
    assert draft.usd == pytest.approx(standard.usd / 3.0)


def test_the_360p_quote_says_where_its_ratio_came_from() -> None:
    """It is not on the pricing page, so the estimate has to admit that.

    Every other rate in the module is read off a pricing table; presenting a
    figure derived from a blog post the same way would launder it.
    """
    from src.pricing import estimate_video_cost

    draft = estimate_video_cost(OMNI_1_1_MODEL, 6, "360p")
    assert draft is not None
    note = draft.source_note or ""
    assert "launch post" in note
    assert "not from the pricing page" in note


def test_upscaled_tiers_bill_at_the_published_720p_rate_and_say_so() -> None:
    """Google publishes one omni token rate and pins it to 720p.

    Inventing a multiplier for the upscaled tiers would be fabrication, so
    the published rate is applied and its limits are stated on the figure.
    """
    from src.pricing import estimate_video_cost

    base = estimate_video_cost(OMNI_1_1_MODEL, 6, "720p")
    assert base is not None
    for resolution in ("1080p", "4K"):
        quote = estimate_video_cost(OMNI_1_1_MODEL, 6, resolution)
        assert quote is not None
        assert quote.usd == pytest.approx(base.usd)
        assert "upscaled" in (quote.source_note or "")


def test_the_preview_model_still_reprices_a_resolution_it_cannot_render() -> None:
    """1.1 gaining real resolutions must not change what the preview quotes."""
    from src.pricing import estimate_video_cost

    quote = estimate_video_cost(OMNI_PREVIEW_MODEL, 6, "1080p")
    assert quote is not None
    assert "renders 720p" in quote.detail
    assert "360p" not in quote.detail


def test_the_preview_model_has_no_360p_price_to_hand_out() -> None:
    """A 360p ask there is meaningless, not cheap: it renders 720p regardless."""
    from src.pricing import estimate_video_cost

    draft = estimate_video_cost(OMNI_PREVIEW_MODEL, 6, "360p")
    standard = estimate_video_cost(OMNI_PREVIEW_MODEL, 6, "720p")
    assert draft is not None and standard is not None
    assert draft.usd == pytest.approx(standard.usd)


def test_the_new_model_carries_the_omni_encoder_allowance() -> None:
    """Omni renders land a frame over nominal; 1.1 must not skip the allowance."""
    from src.pricing import OMNI_ENCODER_ALLOWANCE_SECONDS, quote_duration_for

    assert quote_duration_for(OMNI_1_1_MODEL, 6.0) == pytest.approx(
        6.0 + OMNI_ENCODER_ALLOWANCE_SECONDS
    )


# ============================================================================
# The tools
# ============================================================================


def _app_ctx(tmp_path: Path) -> AppContext:
    (tmp_path / "images").mkdir(exist_ok=True)
    (tmp_path / "videos").mkdir(exist_ok=True)
    client = MagicMock()
    client._api_client.vertexai = False
    return AppContext(
        data_folder=tmp_path,
        images_dir=tmp_path / "images",
        videos_dir=tmp_path / "videos",
        client=client,
    )


def _ctx(tmp_path: Path) -> Any:
    ctx = MagicMock()
    ctx.info = AsyncMock()
    ctx.error = AsyncMock()
    ctx.warning = AsyncMock()
    ctx.request_context.lifespan_context = _app_ctx(tmp_path)
    return ctx


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_a_360p_dry_run_quotes_the_draft_tier(tmp_path: Path) -> None:
    from src.__main__ import generate_video_omni

    payload = json.loads(
        await generate_video_omni(
            ctx=_ctx(tmp_path),
            prompt="a robot",
            omni_model=OMNI_1_1_MODEL,
            resolution="360p",
            duration_seconds=6,
            dry_run=True,
        )
    )
    assert payload["model"] == OMNI_1_1_MODEL
    assert payload["resolution"] == "360p"
    at_720 = json.loads(
        await generate_video_omni(
            ctx=_ctx(tmp_path),
            prompt="a robot",
            omni_model=OMNI_1_1_MODEL,
            duration_seconds=6,
            dry_run=True,
        )
    )
    # The payload rounds to cents-of-a-cent, so compare at that precision.
    assert payload["estimated_cost"]["usd"] == pytest.approx(
        at_720["estimated_cost"]["usd"] / 3.0, abs=1e-6
    )


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_a_resolution_on_the_preview_model_is_an_error_not_a_720p_render(
    tmp_path: Path,
) -> None:
    """The whole point of the refusal: no quote for a render that cannot happen."""
    from src.__main__ import generate_video_omni

    payload = json.loads(
        await generate_video_omni(
            ctx=_ctx(tmp_path),
            prompt="a robot",
            omni_model=OMNI_PREVIEW_MODEL,
            resolution="4K",
            dry_run=True,
        )
    )
    assert "estimated_cost" not in payload
    assert "no resolution parameter" in payload["error"]


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_an_unknown_omni_model_is_refused_before_it_is_quoted(
    tmp_path: Path,
) -> None:
    from src.__main__ import generate_video_omni

    payload = json.loads(
        await generate_video_omni(
            ctx=_ctx(tmp_path), prompt="x", omni_model="gemini-omni-9", dry_run=True
        )
    )
    assert "Unsupported omni_model" in payload["error"]


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_extend_needs_exactly_one_source(tmp_path: Path) -> None:
    """Two sources is ambiguous and none is unrunnable; both fail the same way."""
    from src.__main__ import extend_video_omni

    neither = json.loads(
        await extend_video_omni(ctx=_ctx(tmp_path), prompt="Continue.", dry_run=True)
    )
    assert "exactly one source" in neither["error"]

    both = json.loads(
        await extend_video_omni(
            ctx=_ctx(tmp_path),
            prompt="Continue.",
            previous_interaction_id="i-1",
            input_video_uri="file:///nope.mp4",
            dry_run=True,
        )
    )
    assert "exactly one source" in both["error"]


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_extend_quotes_each_turn_at_the_assembled_clip(
    tmp_path: Path,
) -> None:
    """A turn renders the ASSEMBLED clip, so every turn re-bills the rest.

    MEASURED: a 3.01s source extended once returned 13.01s. Two documentary
    readings said otherwise — `duration` is "the length of the generated video
    files" (3-10s), and Vertex's sample response bills 28,832 video tokens,
    i.e. 4.98s — and neither survived a real render. Quoting the increment
    under-billed a 2-turn chain 36s against a 20s quote.
    """
    from src.__main__ import extend_video_omni
    from src.pricing import estimate_video_cost

    payload = json.loads(
        await extend_video_omni(
            ctx=_ctx(tmp_path),
            prompt="Continue the scene.",
            previous_interaction_id="i-1",
            times=3,
            dry_run=True,
        )
    )
    assert payload["times"] == 3
    # Source unknown here, so the documented 10s maximum is assumed: the turns
    # render 20s, 30s and 40s of assembled clip.
    assert payload["turn_output_seconds"] == [20.0, 30.0, 40.0]
    assert payload["billed_seconds"] == 90.0
    assert payload["assembled_seconds"] == 40.0
    assert payload["cumulative_cap_seconds"] == 40

    # Priced at the published 720p rate, per turn, each carrying the one-frame
    # encoder allowance every omni quote carries. Note estimate_video_cost
    # cannot be used to build this figure directly: it snaps a duration into
    # omni's [3, 10]s per-RENDER range, which an extension chain's output
    # legitimately exceeds.
    from src.pricing import OMNI_ENCODER_ALLOWANCE_SECONDS

    probe = estimate_video_cost(OMNI_1_1_MODEL, 6, "720p")
    assert probe is not None
    rate = probe.breakdown["usd_per_second"]
    assert payload["estimated_cost"]["usd"] == pytest.approx(
        sum((n + OMNI_ENCODER_ALLOWANCE_SECONDS) * rate for n in (20, 30, 40)),
        abs=1e-6,
    )
    # The increment-only quote this replaced would have been ~3x under.
    assert payload["estimated_cost"]["usd"] > 3 * 10 * rate * 2.5


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_a_measured_source_bounds_how_many_turns_fit(
    tmp_path: Path,
) -> None:
    """The 40s ceiling counts the source, so a long source leaves fewer turns.

    Extending a 35s clip has room for one 5s turn, not four 10s ones — and
    quoting four would bill for footage the service will not produce.
    """
    from src.__main__ import extend_video_omni

    sidecar = tmp_path / "videos" / "prior.json"
    sidecar.parent.mkdir(exist_ok=True)
    sidecar.write_text(
        json.dumps(
            {
                "kind": "omni_video",
                "interaction_id": "i-long",
                "duration_seconds": 35.0,
                "duration_source": "measured from the rendered video",
            }
        )
    )

    payload = json.loads(
        await extend_video_omni(
            ctx=_ctx(tmp_path),
            prompt="Continue the scene.",
            previous_interaction_id="i-long",
            times=4,
            dry_run=True,
        )
    )
    assert payload["source_duration_seconds"] == 35.0
    # 35s + 10s clamps at the 40s ceiling; the other three turns have no room.
    assert payload["turn_output_seconds"] == [40.0]
    assert payload["planned_turns"] == 1
    assert payload["assembled_seconds"] == 40.0
    assert payload["appended_seconds"] == 5.0


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_extend_refuses_more_turns_than_the_40s_ceiling_allows(
    tmp_path: Path,
) -> None:
    from src.__main__ import extend_video_omni

    payload = json.loads(
        await extend_video_omni(
            ctx=_ctx(tmp_path),
            prompt="Continue.",
            previous_interaction_id="i-1",
            times=5,
            dry_run=True,
        )
    )
    assert "at most 4 turns" in payload["error"]


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_extend_is_refused_outright_on_the_preview_model(
    tmp_path: Path,
) -> None:
    from src.__main__ import extend_video_omni

    payload = json.loads(
        await extend_video_omni(
            ctx=_ctx(tmp_path),
            prompt="Continue.",
            previous_interaction_id="i-1",
            omni_model=OMNI_PREVIEW_MODEL,
            dry_run=True,
        )
    )
    assert "does not support the extend task" in payload["error"]


@pytest.mark.asyncio
@pytest.mark.timeout(20.0)
async def test_extend_chains_each_turn_into_the_next(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Turn 2 continues turn 1's interaction; the source video is sent once.

    Re-sending the uploaded clip on every turn would extend the ORIGINAL each
    time instead of the growing result — three renders, none of them longer
    than one extension.
    """
    from src.__main__ import extend_video_omni

    source = tmp_path / "clip.mp4"
    source.write_bytes(_MP4_HEADER + b"clip")
    calls: list[dict[str, Any]] = []

    async def mock_impl(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        out = tmp_path / "videos" / f"turn{len(calls)}.mp4"
        out.write_bytes(b"mp4")
        return {
            "message": "ok",
            "video_url": f"file://{out}",
            "interaction_id": f"i-{len(calls)}",
            "model": OMNI_1_1_MODEL,
            "task": "extend",
            "duration_seconds": None,
            "aspect_ratio": None,
            "resolution": None,
            "rendered_resolution": "720p",
        }

    monkeypatch.setattr("src.__main__.generate_video_omni_impl", mock_impl)

    payload = json.loads(
        await extend_video_omni(
            ctx=_ctx(tmp_path),
            prompt="Continue the scene.",
            input_video_uri=f"file://{source}",
            times=2,
        )
    )

    assert len(calls) == 2
    # Turn 1 uploads the clip and names the task; turn 2 continues turn 1.
    assert calls[0]["input_video_bytes"] is not None
    assert calls[0]["task"] == "extend"
    assert calls[0]["previous_interaction_id"] is None
    assert calls[1]["input_video_bytes"] is None
    assert calls[1]["previous_interaction_id"] == "i-1"
    # Every turn names the task it is. _build_create_kwargs drops it from the
    # wire whenever a previous_interaction_id is present (the API rejects the
    # pair), but passing None here instead made the impl infer "edit" and
    # every turn after the first was recorded as an edit of a chain the caller
    # had asked to extend.
    assert all(call["task"] == "extend" for call in calls)
    # No turn carries a duration: the service rejects it on an extend request
    # and the increment is fixed, so there is nothing to send.
    assert all(call["duration_seconds"] is None for call in calls)
    assert payload["interaction_id"] == "i-2"
    assert len(payload["segments"]) == 2


@pytest.mark.asyncio
@pytest.mark.timeout(20.0)
async def test_an_over_long_uploaded_source_is_refused_before_it_is_uploaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """10s is a documented hard limit on an UPLOADED extension source.

    It is a property of the bytes in hand, so it is measured here rather than
    discovered from a 400 after the transfer — and the message points at the
    multi-turn path, which has no such limit.
    """
    import numpy as np
    import imageio.v3 as iio

    from src.__main__ import extend_video_omni

    frames = [
        np.full((32, 32, 3), (i * 3) % 256, dtype=np.uint8) for i in range(24 * 12)
    ]
    buf = BytesIO()
    iio.imwrite(buf, frames, extension=".mp4", fps=24)
    source = tmp_path / "long.mp4"
    source.write_bytes(buf.getvalue())

    async def should_not_run(**kwargs: Any) -> dict[str, Any]:
        raise AssertionError("nothing may be sent for an over-long source")

    monkeypatch.setattr("src.__main__.generate_video_omni_impl", should_not_run)

    payload = json.loads(
        await extend_video_omni(
            ctx=_ctx(tmp_path),
            prompt="Continue.",
            input_video_uri=f"file://{source}",
        )
    )
    assert "10s or shorter" in payload["error"]
    assert "previous_interaction_id" in payload["error"]


@pytest.mark.asyncio
@pytest.mark.timeout(20.0)
async def test_a_chain_that_would_pass_the_40s_ceiling_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Warned, not refused: each turn's real contribution is the service's.

    "source + turns x 10s" is an upper bound on the finished length, so it is
    not a figure worth failing a call over — but it is worth saying before the
    later turns are paid for.
    """
    import imageio.v3 as iio
    import numpy as np

    from src.__main__ import extend_video_omni

    frames = [
        np.full((32, 32, 3), (i * 3) % 256, dtype=np.uint8) for i in range(24 * 9)
    ]
    buf = BytesIO()
    iio.imwrite(buf, frames, extension=".mp4", fps=24)
    source = tmp_path / "nine.mp4"
    source.write_bytes(buf.getvalue())

    async def mock_impl(**kwargs: Any) -> dict[str, Any]:
        out = tmp_path / "videos" / "x.mp4"
        out.write_bytes(b"mp4")
        return {
            "message": "ok",
            "video_url": f"file://{out}",
            "interaction_id": "i-1",
            "model": OMNI_1_1_MODEL,
            "task": "extend",
            "duration_seconds": None,
            "aspect_ratio": None,
            "resolution": None,
            "rendered_resolution": "720p",
        }

    monkeypatch.setattr("src.__main__.generate_video_omni_impl", mock_impl)

    payload = json.loads(
        await extend_video_omni(
            ctx=_ctx(tmp_path),
            prompt="Continue.",
            input_video_uri=f"file://{source}",
            times=4,
        )
    )
    assert "error" not in payload
    assert any("40s" in warning for warning in payload["warnings"])


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_edit_video_can_choose_a_resolution_on_1_1(tmp_path: Path) -> None:
    from src.__main__ import edit_video

    payload = json.loads(
        await edit_video(
            ctx=_ctx(tmp_path),
            previous_interaction_id="i-1",
            prompt="make it anime",
            omni_model=OMNI_1_1_MODEL,
            resolution="1080p",
            dry_run=True,
        )
    )
    assert payload["model"] == OMNI_1_1_MODEL
    assert payload["resolution"] == "1080p"
    assert payload["duration_seconds"] == 10.0


@pytest.mark.asyncio
@pytest.mark.timeout(20.0)
async def test_a_real_run_bills_the_resolution_that_rendered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The metered cost follows the RENDER, not the request.

    The preview model renders 720p whatever it is asked for, so billing from
    the request would charge a 4K price for a 720p file. 1.1 reports back what
    it rendered, and that is what the cost, the response and the sidecar all
    have to agree on.
    """
    import imageio.v3 as iio
    import numpy as np

    from src.__main__ import generate_video_omni
    from src.pricing import actual_video_cost

    frames = [
        np.full((32, 32, 3), (i * 3) % 256, dtype=np.uint8) for i in range(24 * 6)
    ]
    buf = BytesIO()
    iio.imwrite(buf, frames, extension=".mp4", fps=24)
    rendered = tmp_path / "videos" / "draft.mp4"
    rendered.parent.mkdir(exist_ok=True)
    rendered.write_bytes(buf.getvalue())

    captured: dict[str, Any] = {}

    async def mock_impl(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "message": "ok",
            "video_url": f"file://{rendered}",
            "interaction_id": "i-1",
            "model": OMNI_1_1_MODEL,
            "task": "text_to_video",
            "duration_seconds": 6,
            "aspect_ratio": "16:9",
            "resolution": "360p",
            "rendered_resolution": "360p",
        }

    monkeypatch.setattr("src.__main__.generate_video_omni_impl", mock_impl)

    payload = json.loads(
        await generate_video_omni(
            ctx=_ctx(tmp_path),
            prompt="a robot",
            omni_model=OMNI_1_1_MODEL,
            resolution="360p",
        )
    )

    assert captured["model"] == OMNI_1_1_MODEL
    assert captured["resolution"] == "360p"
    # Measured from the file that rendered, priced at the tier that rendered.
    metered = actual_video_cost(
        OMNI_1_1_MODEL, payload["duration_seconds"], "360p", False
    )
    assert metered is not None
    assert payload["cost"]["usd"] == pytest.approx(metered.usd, abs=1e-6)
    assert payload["cost"]["is_estimate"] is False
    # And a 720p bill would be three times that, so the two are not confusable.
    at_720 = actual_video_cost(
        OMNI_1_1_MODEL, payload["duration_seconds"], "720p", False
    )
    assert at_720 is not None
    assert payload["cost"]["usd"] < at_720.usd / 2


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_too_many_input_images_is_refused(tmp_path: Path) -> None:
    from src.__main__ import MAX_OMNI_INPUT_IMAGES, generate_video_omni

    payload = json.loads(
        await generate_video_omni(
            ctx=_ctx(tmp_path),
            prompt="x",
            omni_model=OMNI_1_1_MODEL,
            reference_image_uris=["gs://b/i.png"] * (MAX_OMNI_INPUT_IMAGES + 1),
            dry_run=True,
        )
    )
    assert "Too many input images" in payload["error"]


# ============================================================================
# The SDK floor
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_a_resolution_is_refused_when_the_sdk_would_drop_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """google-genai below 2.20.0 strips `resolution` during serialization.

    No error, no warning, nothing in the request body — the render comes back
    720p while this server reports and bills 4K. A dependency pin is the fix,
    but a pin cannot be checked at runtime and this failure is invisible on
    the wire, so the call refuses rather than trusting it.
    """
    import src.omni as omni

    monkeypatch.setattr(omni, "RESOLUTION_REACHES_THE_WIRE", False)

    with pytest.raises(RuntimeError, match="google-genai>=2.20.0"):
        _ = await omni.generate_video_omni(
            client=MagicMock(),
            prompt="a robot",
            videos_dir=tmp_path,
            model=OMNI_1_1_MODEL,
            resolution="4K",
        )


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_a_request_without_a_resolution_is_unaffected_by_the_sdk_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing to drop, nothing to refuse: the default path keeps working."""
    import src.omni as omni

    monkeypatch.setattr(omni, "RESOLUTION_REACHES_THE_WIRE", False)

    client = MagicMock()
    client.interactions.create.return_value = {
        "id": "i-1",
        "status": "completed",
        "steps": [
            {"content": [{"type": "video", "mime_type": "video/mp4", "data": "AAAA"}]}
        ],
    }
    result = await omni.generate_video_omni(
        client=client,
        prompt="a robot",
        videos_dir=tmp_path,
        model=OMNI_1_1_MODEL,
    )
    assert result["rendered_resolution"] == "720p"


def test_the_installed_sdk_actually_serializes_resolution() -> None:
    """The pin in pyproject.toml, checked rather than assumed.

    This is the assertion that would have caught the whole feature rendering
    at 720p on an SDK a third of a version behind.
    """
    from src.omni import RESOLUTION_REACHES_THE_WIRE

    assert RESOLUTION_REACHES_THE_WIRE


# ============================================================================
# Review fixes: one test per defect, each failing before its fix
# ============================================================================


def test_a_declared_reference_is_never_demoted_to_a_first_frame() -> None:
    """One reference image is still a reference, not an opening shot.

    _select_task_type used to take a single merged image count, so
    `reference_image_bytes_list=[x]` was indistinguishable from "one image,
    role unstated" and went out as image_to_video — the model told to open on
    a picture the caller had explicitly called a likeness, under a task field
    the reference itself says "adds strict constraints".
    """
    from src.omni import _select_task_type  # pyright: ignore[reportPrivateUsage]

    def task(**kwargs: Any) -> str | None:
        return _select_task_type(
            previous_interaction_id=None, input_video_bytes=None, **kwargs
        )

    assert task(reference_image_count=1) == "reference_to_video"
    assert task(reference_image_count=2) == "reference_to_video"
    assert task(reference_video_count=1) == "reference_to_video"
    assert task(has_first_frame=True) == "image_to_video"
    # Interpolation has a documented task after all — the GA release note
    # names image_to_video for "up to 2 images". Genuinely mixed roles still
    # map to none.
    assert task(has_first_frame=True, has_last_frame=True) == "image_to_video"
    assert task(has_first_frame=True, reference_image_count=1) is None
    # The inferred-role path the preview model uses is untouched.
    assert task(inferred_image_count=0) == "text_to_video"
    assert task(inferred_image_count=1) == "image_to_video"
    assert task(inferred_image_count=2) == "reference_to_video"


def test_a_lone_declared_reference_still_gets_its_tag() -> None:
    """The tag is the ONLY thing separating a likeness from a first frame.

    The "one obvious role needs no declaration" shortcut used to swallow a
    single reference, dropping both <IMAGE_REF_0> and the "should not be used
    as literal initial frames" instruction for exactly the request that needed
    them most.
    """
    one_image = _build_media_role_prompt(
        "a woman walks",
        has_first_frame=False,
        has_last_frame=False,
        reference_image_count=1,
        has_source_video=False,
        reference_video_count=0,
    )
    assert "<IMAGE_REF_0>@Image1" in one_image
    assert "should not be used as literal initial frames" in one_image

    one_video = _build_media_role_prompt(
        "the dog jumps onto the sofa",
        has_first_frame=False,
        has_last_frame=False,
        reference_image_count=0,
        has_source_video=False,
        reference_video_count=1,
    )
    assert "<VIDEO_REF_0>@Video1" in one_video

    # The two genuinely unambiguous cases stay bare, as the reference advises.
    bare: dict[str, Any] = {
        "has_first_frame": False,
        "has_last_frame": False,
        "reference_image_count": 0,
        "has_source_video": False,
        "reference_video_count": 0,
    }
    for role in ("has_first_frame", "has_source_video"):
        assert _build_media_role_prompt("a cat", **{**bare, role: True}) == "a cat"


def test_a_continuation_is_sent_to_the_backend_that_minted_it() -> None:
    """An interaction id resolves on ONE backend and fails on the other.

    _client_for_omni picked from the CURRENT call's needs, so a chain whose
    last turn wanted GCS output jumped to Vertex for that turn alone, carrying
    a previous_interaction_id the Gemini Developer API had minted. Every
    earlier turn billed; the last one dead.
    """
    import src.__main__ as server

    vertex_primary = MagicMock()
    vertex_primary._api_client.vertexai = True
    gemini_client = MagicMock()
    gemini_client._api_client.vertexai = False
    global_vertex = MagicMock()
    global_vertex._api_client.vertexai = True

    app = MagicMock()
    app.client = vertex_primary
    app.gemini_api_client = gemini_client

    original = server._omni_vertex_global_client
    server._omni_vertex_global_client = global_vertex
    try:
        # Without a pin, a GCS-bound turn goes to Vertex — the old behaviour.
        assert server._client_for_omni(app, need_gcs=True) is global_vertex
        # Pinned to the interaction's own backend, it does not, because a
        # dropped output_gcs_uri still renders and a wrong backend does not.
        assert (
            server._client_for_omni(app, need_gcs=True, prefer_backend="gemini_api")
            is gemini_client
        )
        assert server._client_for_omni(app, prefer_backend="vertex") is global_vertex
    finally:
        server._omni_vertex_global_client = original


def test_an_unreachable_pinned_backend_falls_back_rather_than_failing() -> None:
    """A recorded backend this deployment cannot reach is not a reason to stop.

    Better to issue the call and surface the service's own error than to
    invent a refusal from a sidecar written by some earlier configuration.
    """
    import src.__main__ as server

    gemini_primary = MagicMock()
    gemini_primary._api_client.vertexai = False
    app = MagicMock()
    app.client = gemini_primary
    app.gemini_api_client = None

    assert server._client_for_omni(app, prefer_backend="vertex") is gemini_primary


def test_the_manifest_records_which_backend_minted_the_interaction(
    tmp_path: Path,
) -> None:
    """The pin above is only as good as what was written down."""
    import src.__main__ as server

    (tmp_path / "videos").mkdir(exist_ok=True)
    sidecar = tmp_path / "videos" / "prior.json"
    sidecar.write_text(json.dumps({"interaction_id": "i-42", "backend": "vertex"}))
    recorded = server._prior_interaction(tmp_path / "videos", "i-42")
    assert recorded.backend == "vertex"
    # An older sidecar with no backend recorded yields None, not a guess.
    sidecar.write_text(json.dumps({"interaction_id": "i-43"}))
    assert server._prior_interaction(tmp_path / "videos", "i-43").backend is None
    # And nothing at all for an id this server never wrote down.
    assert server._prior_interaction(tmp_path / "videos", "nope").backend is None


@pytest.mark.asyncio
@pytest.mark.timeout(20.0)
async def test_a_mid_chain_failure_keeps_the_turns_already_billed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Turn 2 failing must not strand turn 1's render.

    A bare {"error": ...} left the caller unable to resume from — or reconcile
    — an interaction they had already paid for. loop_extend and generate_clip
    both carry handlers for exactly this; the omni chain reintroduced the hole.
    """
    from src.__main__ import extend_video_omni

    calls: list[dict[str, Any]] = []

    async def mock_impl(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        if len(calls) == 2:
            raise RuntimeError("service said no")
        out = tmp_path / "videos" / "turn1.mp4"
        out.write_bytes(b"mp4")
        return {
            "message": "ok",
            "video_url": f"file://{out}",
            "interaction_id": "i-1",
            "model": OMNI_1_1_MODEL,
            "task": "extend",
            "duration_seconds": 20.0,
            "aspect_ratio": None,
            "resolution": None,
            "rendered_resolution": "720p",
        }

    monkeypatch.setattr("src.__main__.generate_video_omni_impl", mock_impl)

    payload = json.loads(
        await extend_video_omni(
            ctx=_ctx(tmp_path),
            prompt="Continue.",
            previous_interaction_id="i-0",
            times=3,
        )
    )
    assert payload["completed_turns"] == 1
    assert payload["interaction_id"] == "i-1"
    assert len(payload["segments"]) == 1
    assert payload["cost"] is not None
    assert "resume" in payload["error"]


@pytest.mark.asyncio
@pytest.mark.timeout(20.0)
async def test_the_chain_separates_billed_assembled_and_appended(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three different numbers, because a turn renders the assembled clip.

    Summing the turns and calling it "appended" overstated a 2-turn chain off
    a 3s source as 36s of new footage when it made 20s — and 36s is the BILL,
    which is the number that has to be right.
    """
    from src.__main__ import extend_video_omni

    lengths = iter([13.0, 23.0])

    async def mock_impl(**kwargs: Any) -> dict[str, Any]:
        out = tmp_path / "videos" / f"{next(iter(['a']))}.mp4"
        out.write_bytes(b"mp4")
        return {
            "message": "ok",
            "video_url": f"file://{out}",
            "interaction_id": "i-x",
            "model": OMNI_1_1_MODEL,
            "task": "extend",
            "duration_seconds": next(lengths),
            "aspect_ratio": None,
            "resolution": None,
            "rendered_resolution": "720p",
        }

    monkeypatch.setattr("src.__main__.generate_video_omni_impl", mock_impl)

    payload = json.loads(
        await extend_video_omni(
            ctx=_ctx(tmp_path),
            prompt="Continue.",
            previous_interaction_id="i-0",
            times=2,
        )
    )
    assert payload["billed_seconds"] == 36.0
    assert payload["assembled_seconds"] == 23.0
    assert payload["completed_turns"] == 2
    assert "final_duration_seconds" not in payload


@pytest.mark.asyncio
@pytest.mark.timeout(20.0)
async def test_generate_video_omni_also_enforces_the_10s_source_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ "Editing AND extension" is one documented limit, checked on both paths.

    Enforced only on extend_video_omni, a 30s clip handed to the edit path was
    fetched whole and uploaded through the Files API before the service
    refused it for a length that was measurable from the bytes in hand.
    """
    import imageio.v3 as iio
    import numpy as np

    from src.__main__ import generate_video_omni

    frames = [
        np.full((32, 32, 3), (i * 3) % 256, dtype=np.uint8) for i in range(24 * 12)
    ]
    buf = BytesIO()
    iio.imwrite(buf, frames, extension=".mp4", fps=24)
    source = tmp_path / "long.mp4"
    source.write_bytes(buf.getvalue())

    async def should_not_run(**kwargs: Any) -> dict[str, Any]:
        raise AssertionError("nothing may be sent for an over-long source")

    monkeypatch.setattr("src.__main__.generate_video_omni_impl", should_not_run)

    payload = json.loads(
        await generate_video_omni(
            ctx=_ctx(tmp_path),
            prompt="make the mirror ripple",
            omni_model=OMNI_1_1_MODEL,
            input_video_uri=f"file://{source}",
        )
    )
    assert "10s or shorter" in payload["error"]


def test_the_reference_clip_probe_runs_off_the_event_loop() -> None:
    """Each probe spawns ffmpeg; three of them inline block every request.

    Its sibling _check_omni_source_video already wraps the identical call in a
    thread, and _source_duration_or_none goes further with a dedicated pool
    precisely because one blocked coroutine blocks the whole server.
    """
    import inspect

    import src.__main__ as server

    assert inspect.iscoroutinefunction(server._omni_reference_video_warnings)


def test_the_published_token_rate_reconciles_with_every_resolution_price() -> None:
    """tokens_per_second is the 720p figure, not a constant.

    Emitted alone beside per-resolution dollar rates derived from a different
    token count, it let a caller multiply it out and get 3x what the same
    dict's own 360p rate says.
    """
    from src.pricing import describe_model_pricing

    record = describe_model_pricing(OMNI_1_1_MODEL)
    assert record is not None
    per_resolution = record["tokens_per_second_by_resolution"]
    assert per_resolution["720p"] == record["tokens_per_second"]
    assert per_resolution["360p"] == pytest.approx(record["tokens_per_second"] / 3)
    for resolution, tokens in per_resolution.items():
        assert tokens * record["output_video_usd_per_mtok"] / 1e6 == pytest.approx(
            record["usd_per_second"][resolution]
        )


# ============================================================================
# Retrieving a delivered render
# ============================================================================


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        pytest.param("files/abc123", "files/abc123", id="already_a_name"),
        pytest.param(
            "https://generativelanguage.googleapis.com/v1beta/files/abc123"
            ":download?alt=media",
            "files/abc123",
            id="download_url",
        ),
        pytest.param(
            "https://generativelanguage.googleapis.com/v1beta/files/abc-1_2",
            "files/abc-1_2",
            id="plain_url",
        ),
    ],
)
def test_a_delivered_uri_resolves_to_a_files_resource_name(
    uri: str, expected: str
) -> None:
    """files.get takes the NAME, not the download URL handed back.

    The reference's Python snippet takes the URI's last path segment, which on
    a download URL yields "abc123:download?alt=media" — a name no lookup will
    ever match. Its JavaScript snippet uses a regex, which is the form that
    works, and is what this mirrors.
    """
    from src.omni import _delivered_file_name  # pyright: ignore[reportPrivateUsage]

    assert _delivered_file_name(uri) == expected


@pytest.mark.asyncio
@pytest.mark.timeout(10.0)
async def test_a_delivered_render_is_downloaded_only_once_it_is_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Downloading a PROCESSING file returns nothing useful for a paid render.

    The reference's retrieval example polls until ACTIVE before downloading,
    and this path did not — it went straight from the interaction's URI to a
    download, using the URI as a resource name into the bargain.
    """
    import src.omni as omni

    monkeypatch.setattr(omni, "_FILE_POLL_INTERVAL", 0)

    seen: list[str] = []
    states = iter(["PROCESSING", "PROCESSING", "ACTIVE"])

    class _Files:
        def get(self, *, name: str) -> Any:
            seen.append(name)
            return MagicMock(name_=name, state=next(states), size_bytes=11)

        def download(self, *, file: Any) -> bytes:
            return b"video-bytes"

    client = MagicMock()
    client.files = _Files()

    async def run(func: Any, /, **kwargs: Any) -> Any:
        return func(**kwargs)

    data = await omni._resolve_video_bytes(
        client,
        None,
        "https://generativelanguage.googleapis.com/v1beta/files/xyz:download?alt=media",
        None,
        run,
        lambda: False,
    )
    assert data == b"video-bytes"
    # Polled by resource name, three times, until it left PROCESSING.
    assert seen == ["files/xyz"] * 3


@pytest.mark.asyncio
@pytest.mark.timeout(10.0)
async def test_a_failed_delivery_is_not_downloaded_as_if_it_worked() -> None:
    import src.omni as omni

    class _Files:
        def get(self, *, name: str) -> Any:
            return MagicMock(state="FAILED")

        def download(self, *, file: Any) -> bytes:  # pragma: no cover
            raise AssertionError("a FAILED file must not be downloaded")

    client = MagicMock()
    client.files = _Files()

    async def run(func: Any, /, **kwargs: Any) -> Any:
        return func(**kwargs)

    with pytest.raises(ValueError, match="FAILED"):
        _ = await omni._resolve_video_bytes(
            client, None, "files/xyz", None, run, lambda: False
        )


def test_the_uri_delivery_rule_matches_the_documented_threshold() -> None:
    """ ">720p", plus an extension, which renders the whole growing clip."""
    from src.omni import wants_uri_delivery

    one_one = omni_spec(OMNI_1_1_MODEL)
    assert wants_uri_delivery(one_one, "4K", "text_to_video")
    assert wants_uri_delivery(one_one, "1080p", "text_to_video")
    assert wants_uri_delivery(one_one, None, "extend")
    assert not wants_uri_delivery(one_one, "720p", "text_to_video")
    assert not wants_uri_delivery(one_one, "360p", "text_to_video")
    assert not wants_uri_delivery(one_one, None, "edit")
    # The preview model cannot render anything that large.
    assert not wants_uri_delivery(omni_spec(OMNI_PREVIEW_MODEL), None, "edit")


# ============================================================================
# Final-review fixes
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(20.0)
async def test_the_source_cap_does_not_fire_on_a_model_with_no_such_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cap the model does not document is not a cap of zero seconds.

    gemini-omni-flash-preview records max_uploaded_source_seconds=0.0 because
    it documents no ceiling. Reading that as a limit refused EVERY input video
    on the DEFAULT model — a documented path that had always worked — with
    "must be 0s or shorter".
    """
    import imageio.v3 as iio
    import numpy as np

    from src.__main__ import generate_video_omni

    frames = [
        np.full((32, 32, 3), (i * 3) % 256, dtype=np.uint8) for i in range(24 * 3)
    ]
    buf = BytesIO()
    iio.imwrite(buf, frames, extension=".mp4", fps=24)
    source = tmp_path / "three.mp4"
    source.write_bytes(buf.getvalue())
    rendered = tmp_path / "videos" / "out.mp4"
    rendered.parent.mkdir(exist_ok=True)
    rendered.write_bytes(b"mp4")

    async def mock_impl(**kwargs: Any) -> dict[str, Any]:
        return {
            "message": "ok",
            "video_url": f"file://{rendered}",
            "interaction_id": "i-1",
            "model": OMNI_PREVIEW_MODEL,
            "task": "edit",
            "duration_seconds": None,
            "aspect_ratio": None,
            "resolution": None,
            "rendered_resolution": "720p",
        }

    monkeypatch.setattr("src.__main__.generate_video_omni_impl", mock_impl)

    payload = json.loads(
        await generate_video_omni(
            ctx=_ctx(tmp_path),
            prompt="make the sky stormy",
            input_video_uri=f"file://{source}",
        )
    )
    assert "error" not in payload, payload.get("error")
    assert payload["interaction_id"] == "i-1"


def test_an_unmeasurable_extension_is_bounded_by_what_it_can_render() -> None:
    """Not by the 10s per-render maximum, which under-bills a chain ~3x.

    An extension returns the whole growing clip: turn four renders 40s. An
    edit is the other way round — the one measurement in hand put a 3s
    source's edit at the per-render maximum — and a source longer than that is
    only reachable by extending here, which writes a sidecar. So an
    unrecorded edit source is a short one.
    """
    from src.omni import omni_continuation_upper_bound

    one_one = omni_spec(OMNI_1_1_MODEL)
    preview = omni_spec(OMNI_PREVIEW_MODEL)

    # A turn renders the assembled clip, so the bound is source + one
    # increment, and the documented ceiling when the source is unknown.
    assert omni_continuation_upper_bound(one_one, "extend", 10.0) == 20.0
    assert omni_continuation_upper_bound(one_one, "extend", 35.0) == 40.0
    assert omni_continuation_upper_bound(one_one, "extend", None) == 40.0
    assert omni_continuation_upper_bound(one_one, "edit", 3.0) == 10.0
    assert omni_continuation_upper_bound(one_one, "edit", 30.0) == 30.0
    assert omni_continuation_upper_bound(one_one, "edit", None) == 10.0
    # The preview model cannot extend, so nothing it renders exceeds 10s.
    assert omni_continuation_upper_bound(preview, "extend", 10.0) == 10.0
    assert omni_continuation_upper_bound(preview, "edit", None) == 10.0


@pytest.mark.asyncio
@pytest.mark.timeout(20.0)
async def test_an_unmeasurable_extension_bills_the_assembled_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rule above, reached through the tool: not one 10s increment."""
    from src.__main__ import extend_video_omni

    async def mock_impl(**kwargs: Any) -> dict[str, Any]:
        return {
            "message": "ok",
            # gs:// cannot be opened to measure, which is the whole point.
            "video_url": "gs://bucket/out/clip.mp4",
            "interaction_id": "i-1",
            "model": OMNI_1_1_MODEL,
            "task": "extend",
            "duration_seconds": None,
            "aspect_ratio": None,
            "resolution": None,
            "rendered_resolution": "720p",
        }

    monkeypatch.setattr("src.__main__.generate_video_omni_impl", mock_impl)

    payload = json.loads(
        await extend_video_omni(
            ctx=_ctx(tmp_path),
            prompt="Continue.",
            previous_interaction_id="i-0",
            times=1,
        )
    )
    segment = payload["segments"][0]
    assert segment["duration_seconds"] == 40.0
    assert "upper bound" in segment["duration_source"]


@pytest.mark.asyncio
@pytest.mark.timeout(20.0)
async def test_a_chain_pins_its_backend_from_the_result_not_the_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pin was dead code: `backend` only ever reached the sidecar.

    So a chain starting from an uploaded video re-scanned the filesystem every
    turn and, when no sidecar could be written at all, still let the final
    GCS-bound turn drift onto another backend.
    """
    from src.__main__ import _omni_generate_and_manifest

    rendered = tmp_path / "videos" / "out.mp4"
    rendered.parent.mkdir(exist_ok=True)
    rendered.write_bytes(b"mp4")

    async def mock_impl(**kwargs: Any) -> dict[str, Any]:
        return {
            "message": "ok",
            "video_url": f"file://{rendered}",
            "interaction_id": "i-1",
            "model": OMNI_1_1_MODEL,
            "task": "text_to_video",
            "duration_seconds": 6,
            "aspect_ratio": "16:9",
            "resolution": None,
            "rendered_resolution": "720p",
        }

    monkeypatch.setattr("src.__main__.generate_video_omni_impl", mock_impl)

    ctx = _ctx(tmp_path)
    result = await _omni_generate_and_manifest(
        ctx.request_context.lifespan_context,
        ctx,
        prompt="a robot",
        model=OMNI_1_1_MODEL,
    )
    assert result["backend"] == "gemini_api"


@pytest.mark.asyncio
@pytest.mark.timeout(10.0)
async def test_extending_another_models_interaction_is_refused(
    tmp_path: Path,
) -> None:
    """The task is dropped on a continuation, so this would bill as an edit.

    Passing a preview-model interaction to extend_video_omni sent a bare
    conversational edit and reported it as an extension of a clip that was
    never extended.
    """
    from src.__main__ import extend_video_omni

    (tmp_path / "videos").mkdir(exist_ok=True)
    (tmp_path / "videos" / "prior.json").write_text(
        json.dumps(
            {
                "interaction_id": "i-preview",
                "model": OMNI_PREVIEW_MODEL,
                "backend": "gemini_api",
                "duration_seconds": 6.0,
                "duration_source": "measured from the rendered video",
            }
        )
    )

    payload = json.loads(
        await extend_video_omni(
            ctx=_ctx(tmp_path),
            prompt="Continue.",
            previous_interaction_id="i-preview",
            dry_run=True,
        )
    )
    assert OMNI_PREVIEW_MODEL in payload["error"]
    assert "cannot change models" in payload["error"]


@pytest.mark.asyncio
@pytest.mark.timeout(10.0)
async def test_the_extend_path_caps_its_reference_images(tmp_path: Path) -> None:
    """Its sibling front-loads this cap; every image is buffered whole."""
    from src.__main__ import MAX_OMNI_INPUT_IMAGES, extend_video_omni

    payload = json.loads(
        await extend_video_omni(
            ctx=_ctx(tmp_path),
            prompt="Continue.",
            previous_interaction_id="i-1",
            reference_image_uris=["gs://b/i.png"] * (MAX_OMNI_INPUT_IMAGES + 1),
            dry_run=True,
        )
    )
    assert "Too many reference images" in payload["error"]


@pytest.mark.asyncio
@pytest.mark.timeout(20.0)
async def test_a_large_render_on_vertex_is_disclosed_even_though_it_cannot_help(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Vertex has no bare delivery='uri', so silence was the only outcome.

    A 4K request with no output_gcs_uri went out inline, in violation of the
    reference's own >4MB rule, with nothing said about it.
    """
    from src.__main__ import _omni_generate_and_manifest

    rendered = tmp_path / "videos" / "out.mp4"
    rendered.parent.mkdir(exist_ok=True)
    rendered.write_bytes(b"mp4")

    async def mock_impl(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["allow_uri_delivery"] is False
        return {
            "message": "ok",
            "video_url": f"file://{rendered}",
            "interaction_id": "i-1",
            "model": OMNI_1_1_MODEL,
            "task": "text_to_video",
            "duration_seconds": 6,
            "aspect_ratio": "16:9",
            "resolution": "4K",
            "rendered_resolution": "4K",
        }

    monkeypatch.setattr("src.__main__.generate_video_omni_impl", mock_impl)

    app_ctx = _app_ctx(tmp_path)
    app_ctx.client._api_client.vertexai = True  # pyright: ignore[reportAttributeAccessIssue]
    ctx = MagicMock()
    ctx.info = AsyncMock()
    ctx.error = AsyncMock()
    ctx.warning = AsyncMock()
    ctx.request_context.lifespan_context = app_ctx
    monkeypatch.setattr(
        "src.__main__._get_omni_vertex_global_client", lambda: app_ctx.client
    )

    result = await _omni_generate_and_manifest(
        app_ctx, ctx, prompt="a drone shot", model=OMNI_1_1_MODEL, resolution="4K"
    )
    assert any("output_gcs_uri" in w for w in result["warnings"])


def test_the_download_cap_clears_what_the_new_model_can_emit() -> None:
    """50 MB was defence-in-depth for 720p/<=10s and a real ceiling for 4K/40s.

    Hitting it raises AFTER the render is billed, which is the one place a cap
    must not sit.
    """
    from src.omni import (
        _delivered_video_cap,  # pyright: ignore[reportPrivateUsage]
    )

    assert _delivered_video_cap(omni_spec(OMNI_PREVIEW_MODEL)) == 50 * 1024 * 1024
    assert _delivered_video_cap(omni_spec(OMNI_1_1_MODEL)) > 200 * 1024 * 1024


# ============================================================================
# Facts confirmed against Google's own reference pages
# ============================================================================


def test_the_model_is_spelled_differently_on_each_backend() -> None:
    """Vertex publishes 1.1 as `gemini-omni-1.1-flash-preview`.

    Confirmed on four Gemini Enterprise Agent Platform model pages
    (generate-videos-from-text / -from-an-image / -from-references and
    extend-videos), none of which lists the bare name. Sending the Developer
    API spelling to Vertex reaches no model at all — the same split
    src/video.py already carries for Veo.
    """
    from src.omni import canonical_omni_model, served_omni_model

    assert (
        served_omni_model(OMNI_1_1_MODEL, vertexai=True)
        == "gemini-omni-1.1-flash-preview"
    )
    assert served_omni_model(OMNI_1_1_MODEL, vertexai=False) == OMNI_1_1_MODEL
    # The preview model is spelled the same on both.
    assert served_omni_model(OMNI_PREVIEW_MODEL, vertexai=True) == OMNI_PREVIEW_MODEL
    # And a Vertex ID coming back resolves to the same spec, so it still routes.
    assert canonical_omni_model("gemini-omni-1.1-flash-preview") == OMNI_1_1_MODEL
    assert is_omni_model("gemini-omni-1.1-flash-preview")
    assert omni_spec("gemini-omni-1.1-flash-preview").model == OMNI_1_1_MODEL


def test_the_vertex_spelling_prices_as_the_same_model() -> None:
    """Otherwise a render reported back by Vertex looks unpriced."""
    from src.pricing import estimate_video_cost, resolve_model_id, unpriced_models

    assert resolve_model_id("gemini-omni-1.1-flash-preview") == OMNI_1_1_MODEL
    served = estimate_video_cost("gemini-omni-1.1-flash-preview", 6, "360p")
    canonical = estimate_video_cost(OMNI_1_1_MODEL, 6, "360p")
    assert served is not None and canonical is not None
    assert served.usd == pytest.approx(canonical.usd)
    assert unpriced_models() == ()


@pytest.mark.asyncio
@pytest.mark.timeout(20.0)
async def test_the_served_model_goes_on_the_wire_and_is_reported_separately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wire gets the backend's spelling; the result keeps the canonical one.

    Swapping the reported ID would break every downstream lookup keyed on it
    (pricing, sidecars, a follow-up call); not swapping the wire one would
    reach no model.
    """
    import src.omni as omni

    sent: dict[str, Any] = {}

    client = MagicMock()
    client._api_client.vertexai = True

    def create(**kwargs: Any) -> Any:
        sent.update(kwargs)
        return {
            "id": "i-1",
            "status": "completed",
            "steps": [
                {
                    "content": [
                        {"type": "video", "mime_type": "video/mp4", "data": "AAAA"}
                    ]
                }
            ],
        }

    client.interactions.create = create

    result = await omni.generate_video_omni(
        client=client, prompt="a robot", videos_dir=tmp_path, model=OMNI_1_1_MODEL
    )
    assert sent["model"] == "gemini-omni-1.1-flash-preview"
    assert result["model"] == OMNI_1_1_MODEL
    assert result["served_model"] == "gemini-omni-1.1-flash-preview"


def test_the_uploaded_source_ceiling_is_the_backends() -> None:
    """10s on the Developer API "when uploading"; 1-30s on Vertex.

    Reading the stricter figure onto both refused a legitimate 20s Vertex
    source before anything was even sent.
    """
    from src.omni import omni_source_limit_seconds

    one_one = omni_spec(OMNI_1_1_MODEL)
    assert omni_source_limit_seconds(one_one, vertexai=False) == 10.0
    assert omni_source_limit_seconds(one_one, vertexai=True) == 30.0
    # The preview model documents no ceiling at all, on either backend.
    preview = omni_spec(OMNI_PREVIEW_MODEL)
    assert omni_source_limit_seconds(preview, vertexai=False) == 0.0
    assert omni_source_limit_seconds(preview, vertexai=True) == 0.0


@pytest.mark.asyncio
@pytest.mark.timeout(20.0)
async def test_a_20s_source_is_accepted_on_vertex_and_refused_on_the_dev_api(
    tmp_path: Path,
) -> None:
    """The two backends' limits, reached through the real check."""
    import imageio.v3 as iio
    import numpy as np

    from src.__main__ import _check_omni_source_video

    frames = [
        np.full((32, 32, 3), (i * 3) % 256, dtype=np.uint8) for i in range(24 * 20)
    ]
    buf = BytesIO()
    iio.imwrite(buf, frames, extension=".mp4", fps=24)
    clip = buf.getvalue()
    spec = omni_spec(OMNI_1_1_MODEL)

    assert await _check_omni_source_video(spec, clip, vertexai=True) == []
    with pytest.raises(ValueError, match="10s or shorter"):
        _ = await _check_omni_source_video(spec, clip, vertexai=False)


def test_duration_is_no_longer_warned_about() -> None:
    """The Interactions API reference documents it, so the caveat was wrong.

    It lists `duration` among VideoResponseFormat's fields ("integers between
    3 and 10, followed by 's'"), and Vertex's extend request sends it.
    """
    assert omni_spec(OMNI_1_1_MODEL).duration_is_documented


@pytest.mark.asyncio
@pytest.mark.timeout(20.0)
async def test_a_fresh_1_1_render_carries_no_duration_caveat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.omni as omni

    client = MagicMock()
    client._api_client.vertexai = False
    client.interactions.create.return_value = {
        "id": "i-1",
        "status": "completed",
        "steps": [
            {"content": [{"type": "video", "mime_type": "video/mp4", "data": "AAAA"}]}
        ],
    }

    result = await omni.generate_video_omni(
        client=client,
        prompt="a robot",
        videos_dir=tmp_path,
        model=OMNI_1_1_MODEL,
        duration_seconds=6,
    )
    assert result["duration_seconds"] == 6
    assert not any("never duration" in w for w in result.get("warnings") or [])


# ============================================================================
# The 360p tier reaches the composite tools
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(10.0)
async def test_a_draft_resolution_changes_what_the_draft_costs(
    tmp_path: Path,
) -> None:
    """generate_video(draft=True) was locked to the preview model's 720p.

    So the cheapest render this server can issue was unreachable from the
    draft shortcut — the one place a caller most wants it. The default omni
    model has resolutions now, so this is no longer about which model runs;
    it is about 360p being a third of the bill.
    """
    from src.__main__ import generate_video

    at_720 = json.loads(
        await generate_video(
            ctx=_ctx(tmp_path),
            prompt="a robot",
            model="veo-3.1-fast-generate-001",
            draft=True,
            dry_run=True,
        )
    )
    at_360 = json.loads(
        await generate_video(
            ctx=_ctx(tmp_path),
            prompt="a robot",
            model="veo-3.1-fast-generate-001",
            draft=True,
            draft_resolution="360p",
            dry_run=True,
        )
    )
    assert at_720["model"] == OMNI_1_1_MODEL
    assert at_720["resolution"] == "720p"
    assert at_360["model"] == OMNI_1_1_MODEL
    assert at_360["resolution"] == "360p"
    assert at_360["estimated_cost"]["usd"] == pytest.approx(
        at_720["estimated_cost"]["usd"] / 3.0, abs=1e-6
    )


@pytest.mark.asyncio
@pytest.mark.timeout(10.0)
async def test_an_animatic_can_preview_the_whole_reel_at_360p(
    tmp_path: Path,
) -> None:
    """A 20-beat animatic at 720p costs about what the Veo render costs.

    At 360p it is a third — which is what makes the preview worth recommending
    rather than merely worth explaining.
    """
    from src.__main__ import generate_clip

    beats = [{"prompt": f"beat {i}", "duration_seconds": 8} for i in range(5)]
    at_720 = json.loads(
        await generate_clip(
            ctx=_ctx(tmp_path), beats=beats, animatic=True, dry_run=True
        )
    )
    at_360 = json.loads(
        await generate_clip(
            ctx=_ctx(tmp_path),
            beats=beats,
            animatic=True,
            animatic_resolution="360p",
            dry_run=True,
        )
    )
    assert at_720["model"] == OMNI_1_1_MODEL
    assert at_360["model"] == OMNI_1_1_MODEL
    assert at_360["estimated_cost"]["usd"] == pytest.approx(
        at_720["estimated_cost"]["usd"] / 3.0, abs=1e-6
    )


@pytest.mark.asyncio
@pytest.mark.timeout(10.0)
async def test_a_draft_resolution_the_model_cannot_render_is_refused(
    tmp_path: Path,
) -> None:
    """Refused, not silently rendered at something else and billed for it."""
    from src.__main__ import generate_video

    payload = json.loads(
        await generate_video(
            ctx=_ctx(tmp_path),
            prompt="a robot",
            model="veo-3.1-fast-generate-001",
            draft=True,
            draft_resolution="8K",
            dry_run=True,
        )
    )
    assert "Unsupported resolution '8K'" in payload["error"]


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
async def test_the_animatic_resolution_reaches_the_beat_renders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A quote that promises 360p and a render that sends 720p bill differently.

    The dry_run path and the render path read the resolution separately, so a
    test that only priced the quote would pass while every beat came back at
    the default and cost three times what was quoted.
    """
    from src.__main__ import generate_clip

    calls: list[dict[str, Any]] = []

    async def mock_impl(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        out = tmp_path / "videos" / f"beat{len(calls)}.mp4"
        out.write_bytes(b"mp4")
        return {
            "message": "ok",
            "video_url": f"file://{out}",
            "interaction_id": f"i-{len(calls)}",
            "model": OMNI_1_1_MODEL,
            "task": "text_to_video",
            "duration_seconds": 4,
            "aspect_ratio": "9:16",
            "resolution": "360p",
            "rendered_resolution": "360p",
        }

    monkeypatch.setattr("src.__main__.generate_video_omni_impl", mock_impl)

    payload = json.loads(
        await generate_clip(
            ctx=_ctx(tmp_path),
            beats=[{"prompt": "beat 1", "duration_seconds": 4}],
            animatic=True,
            animatic_resolution="360p",
        )
    )
    assert "error" not in payload, payload.get("error")
    assert calls, "the animatic never rendered"
    assert calls[0]["model"] == OMNI_1_1_MODEL
    assert calls[0]["resolution"] == "360p"
    # And the segment records what rendered, so the clip total prices 360p.
    segment = payload["segments"][0]
    assert segment["resolution"] == "360p"


# ============================================================================
# The GA release notes (2026-08-27)
# ============================================================================


def test_resolution_stays_in_response_format_despite_the_release_note() -> None:
    """The note calls it a "new `resolution` parameter in `video_config`".

    Three generated or executable sources disagree, and they win: the task
    guide's runnable sample, the Interactions API reference schema, and
    google-genai's own types — where VideoConfig has exactly one field and
    VideoResponseFormat is the one carrying `resolution`. Moving it would put
    the field somewhere the SDK drops on serialization, which is the silent
    720p-render/4K-bill failure this module exists to prevent.
    """
    from google.genai._gaos.types.interactions.videoconfig import (  # pyright: ignore[reportMissingImports]
        VideoConfig,
    )
    from google.genai._gaos.types.interactions.videoresponseformat import (  # pyright: ignore[reportMissingImports]
        VideoResponseFormat,
    )

    assert set(VideoConfig.model_fields) == {"task"}
    assert "resolution" in VideoResponseFormat.model_fields


def test_interpolation_uses_the_task_the_release_note_names() -> None:
    """ "Generate a video transitioning between two images using the
    image_to_video task with up to 2 images." The task guide describes
    interpolation without naming a task, which is why this was sending none.
    """
    from src.omni import _select_task_type  # pyright: ignore[reportPrivateUsage]

    assert (
        _select_task_type(
            previous_interaction_id=None,
            input_video_bytes=None,
            has_first_frame=True,
            has_last_frame=True,
        )
        == "image_to_video"
    )


def test_the_deprecated_model_says_when_it_stops() -> None:
    """A caller who reads a warning has time to move.

    gemini-omni-flash-preview is switched off on 2026-09-30, so it can no
    longer be the silent default and cannot be pinned without being told.
    """
    from src.omni import (
        DEFAULT_OMNI_MODEL,
        OMNI_PREVIEW_SUNSET,
    )

    assert DEFAULT_OMNI_MODEL == OMNI_1_1_MODEL, "the default must be the GA model"
    assert OMNI_PREVIEW_SUNSET == "2026-09-30"


@pytest.mark.asyncio
@pytest.mark.timeout(10.0)
async def test_pinning_the_deprecated_model_warns_with_its_end_date(
    tmp_path: Path,
) -> None:
    import src.omni as omni

    client = MagicMock()
    client._api_client.vertexai = False
    client.interactions.create.return_value = {
        "id": "i-1",
        "status": "completed",
        "steps": [
            {"content": [{"type": "video", "mime_type": "video/mp4", "data": "AAAA"}]}
        ],
    }

    result = await omni.generate_video_omni(
        client=client,
        prompt="a robot",
        videos_dir=tmp_path,
        model=OMNI_PREVIEW_MODEL,
    )
    warnings = result.get("warnings") or []
    assert any("2026-09-30" in w for w in warnings), warnings
    assert any(OMNI_1_1_MODEL in w for w in warnings), warnings
    # And the GA model carries no such warning.
    fresh = await omni.generate_video_omni(
        client=client, prompt="a robot", videos_dir=tmp_path, model=OMNI_1_1_MODEL
    )
    assert not any("deprecated" in w for w in fresh.get("warnings") or [])


def test_a_chain_grows_from_its_source_and_reproduces_the_measurement() -> None:
    """The projection has to land on the one figure that was measured.

    3.01s source, one turn, 13.01s output. Everything else follows from that:
    the turns grow, the sum is what is billed, and the ceiling is on the
    assembled clip rather than on the footage added.
    """
    from src.omni import (
        omni_extension_appended_seconds,
        omni_extension_output_lengths,
    )

    spec = omni_spec(OMNI_1_1_MODEL)
    assert omni_extension_output_lengths(spec, 3.01, 1) == [13.01]
    two = omni_extension_output_lengths(spec, 3.01, 2)
    assert two == pytest.approx([13.01, 23.01])
    # The bill for 20s of new footage is 36s of rendered output.
    assert sum(two) == pytest.approx(36.02)
    assert omni_extension_appended_seconds(3.01, two) == pytest.approx(20.0)
    # The ceiling is on the assembled clip, so a long source leaves one turn.
    assert omni_extension_output_lengths(spec, 35.0, 4) == [40.0]
    # A per-turn duration is not a thing the service accepts; passing one
    # cannot change the projection.
    assert omni_extension_output_lengths(spec, 10.0, 2, 5.0) == [20.0, 30.0]


# ============================================================================
# Live-test findings (measured, 2026-08-28)
# ============================================================================


def test_the_quote_covers_the_measured_bill_for_the_reported_render() -> None:
    """The exact case that broke the invariant: 3.01s source -> 13.01s output.

    Reported quote $0.3393 against a $0.4396 bill at 360p. The quote priced the
    10s increment; the service billed the assembled clip.
    """
    from src.pricing import actual_video_cost
    from src.omni import omni_extension_output_lengths

    spec = omni_spec(OMNI_1_1_MODEL)
    projected = omni_extension_output_lengths(spec, 3.01, 1)
    assert projected == [13.01]

    billed = actual_video_cost(
        OMNI_1_1_MODEL, 13.01, "360p", False, snap_duration=False
    )
    quoted = actual_video_cost(
        OMNI_1_1_MODEL, 13.01 + (1.0 / 24), "360p", False, snap_duration=False
    )
    assert billed is not None and quoted is not None
    assert quoted.usd >= billed.usd, "a quote may over-state, never under"
    # And the figure that was wrong is now unreachable from the projection.
    assert quoted.usd == pytest.approx(0.4410, abs=5e-4)


def test_the_snapping_price_path_no_longer_clamps_an_extension() -> None:
    """estimate_video_cost clamps omni to [3, 10] — right for a fresh render.

    An extension's output legitimately exceeds it, and clamping there reported
    $0.3393 for a render that billed $0.4396: a silent 30% under-quote on the
    one path that runs past the range.
    """
    from src.pricing import actual_video_cost, estimate_video_cost

    metered = actual_video_cost(
        OMNI_1_1_MODEL, 13.01, "360p", False, snap_duration=False
    )
    extend_mode = estimate_video_cost(
        OMNI_1_1_MODEL, 13.01, "360p", False, generation_mode="extend_video"
    )
    fresh = estimate_video_cost(OMNI_1_1_MODEL, 13.01, "360p", False)
    assert metered is not None and extend_mode is not None and fresh is not None
    assert extend_mode.usd >= metered.usd
    # A fresh render still clamps, because the API clamps it too.
    assert fresh.usd < metered.usd


@pytest.mark.asyncio
@pytest.mark.timeout(10.0)
async def test_the_extension_dry_run_uses_the_source_it_already_measured(
    tmp_path: Path,
) -> None:
    """The source length was in hand and discarded, which caused the under-quote."""
    from src.__main__ import extend_video_omni

    (tmp_path / "videos").mkdir(exist_ok=True)
    (tmp_path / "videos" / "prior.json").write_text(
        json.dumps(
            {
                "interaction_id": "i-src",
                "duration_seconds": 3.01,
                "duration_source": "measured from the rendered video",
            }
        )
    )
    payload = json.loads(
        await extend_video_omni(
            ctx=_ctx(tmp_path),
            prompt="Continue.",
            previous_interaction_id="i-src",
            times=2,
            resolution="360p",
            dry_run=True,
        )
    )
    assert payload["source_duration_seconds"] == 3.01
    assert payload["turn_output_seconds"] == pytest.approx([13.01, 23.01])
    assert payload["billed_seconds"] == pytest.approx(36.02)
    assert payload["assembled_seconds"] == pytest.approx(23.01)
    # 20s of new footage for a 36s bill — the compounding, stated.
    assert payload["appended_seconds"] == pytest.approx(20.0)
    assert "re-bills" in payload["duration_source"]


def test_a_measured_frame_size_names_its_resolution_tier() -> None:
    """`rendered_resolution` was a request echo with no way to check it.

    Omni's per-second rate differs threefold across tiers, so "the request said
    360p" is not evidence the bill is a 360p bill.
    """
    from src.video_utils import classify_video_resolution

    assert classify_video_resolution((640, 360)) == "360p"
    # Portrait classifies the same: the tier is pixels per frame.
    assert classify_video_resolution((360, 640)) == "360p"
    assert classify_video_resolution((3840, 2160)) == "4K"
    # Nothing near a tier is claimed as one.
    assert classify_video_resolution((500, 890)) is None
    assert classify_video_resolution(None) is None


def test_a_timeout_names_the_render_it_abandoned() -> None:
    """A render that outlives the deadline is billed and keeps going.

    Without the id in the message the spend cannot be retrieved, resumed or
    reconciled — and a host ceiling shorter than timeout_seconds makes that the
    common case. The default sits under the usual ceiling for the same reason.
    """
    from src.omni import OMNI_DEFAULT_TIMEOUT_SECONDS

    assert OMNI_DEFAULT_TIMEOUT_SECONDS < 240, "must fit a ~4 minute host ceiling"


@pytest.mark.asyncio
@pytest.mark.timeout(15.0)
async def test_a_timeout_after_create_carries_the_interaction_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.omni as omni

    monkeypatch.setattr(omni, "_POLL_INTERVAL", 0)

    client = MagicMock()
    client._api_client.vertexai = False
    client.interactions.create.return_value = {"id": "i-abandoned", "status": "queued"}
    client.interactions.get.return_value = {"id": "i-abandoned", "status": "queued"}

    with pytest.raises(TimeoutError, match="i-abandoned"):
        _ = await omni.generate_video_omni(
            client=client,
            prompt="a robot",
            videos_dir=tmp_path,
            model=OMNI_1_1_MODEL,
            timeout_seconds=1,
        )


# ============================================================================
# Wedge hardening and pre-flight accuracy
# ============================================================================


def test_asking_which_backend_never_builds_a_client() -> None:
    """A yes/no question must not construct a Vertex client on the event loop.

    Client construction resolves credentials, which can reach for the metadata
    server, and a failed construction is not memoized — so every subsequent
    call retries and blocks again. A blocked event loop hangs every request in
    the process, not just the one that asked.
    """
    import src.__main__ as server

    built: list[str] = []

    def _explode() -> Any:  # pragma: no cover - must never run
        built.append("constructed")
        raise AssertionError("a pre-flight built a client")

    original = server._get_omni_vertex_global_client
    server._get_omni_vertex_global_client = _explode  # pyright: ignore[reportAttributeAccessIssue]
    try:
        vertex_primary = MagicMock()
        vertex_primary._api_client.vertexai = True
        app = MagicMock()
        app.client = vertex_primary
        app.gemini_api_client = None

        assert server._omni_backend_is_vertex(app) is True
        assert server._omni_backend_is_vertex(app, True) is True
        assert server._omni_backend_is_vertex(app, False, "gemini_api") is True
    finally:
        server._get_omni_vertex_global_client = original  # pyright: ignore[reportAttributeAccessIssue]
    assert built == [], "the pre-flight constructed a client"


def test_the_backend_decision_and_the_client_picker_cannot_disagree() -> None:
    """Splitting them is only safe while they read the same rule."""
    import src.__main__ as server

    gemini = MagicMock()
    gemini._api_client.vertexai = False
    vertex_primary = MagicMock()
    vertex_primary._api_client.vertexai = True
    global_vertex = MagicMock()
    global_vertex._api_client.vertexai = True

    original = server._omni_vertex_global_client
    server._omni_vertex_global_client = global_vertex
    try:
        for primary, api_client in (
            (vertex_primary, gemini),
            (vertex_primary, None),
            (gemini, None),
            (gemini, gemini),
        ):
            app = MagicMock()
            app.client = primary
            app.gemini_api_client = api_client
            for need_gcs in (False, True):
                for prefer in (None, "vertex", "gemini_api"):
                    decided = server._omni_backend_choice(
                        app, need_gcs=need_gcs, prefer_backend=prefer
                    )
                    client = server._client_for_omni(
                        app, need_gcs=need_gcs, prefer_backend=prefer
                    )
                    actual = (
                        "vertex"
                        if getattr(client._api_client, "vertexai", False)
                        else "gemini_api"
                    )
                    assert decided == actual, (primary, api_client, need_gcs, prefer)
    finally:
        server._omni_vertex_global_client = original


def test_no_ffmpeg_work_runs_on_the_shared_default_executor() -> None:
    """asyncio.to_thread draws on the loop's single shared pool.

    That is the same twelve threads every fetch, image render and frame
    extraction use, and a worker blocked in a subprocess cannot be cancelled —
    which this module's own comment says wedges the whole server. Media work
    gets its own bounded pool.
    """
    import re

    # Scanned across line breaks: ruff wraps these calls, so a line-by-line
    # grep silently misses `asyncio.to_thread(\n    measure_video_duration, ...)`
    # — which is the exact shape they are written in.
    source = Path("src/__main__.py").read_text()
    offenders = re.findall(
        r"asyncio\.to_thread\(\s*(measure_\w+|extract_frame\w*)", source
    )
    assert offenders == [], offenders


@pytest.mark.asyncio
@pytest.mark.timeout(20.0)
async def test_a_local_extend_source_is_measured_by_the_pre_flight(
    tmp_path: Path,
) -> None:
    """The quote was assuming the documented maximum for a file it could open.

    $0.6771 quoted against $0.4396 billed — over-stating, so the invariant
    held, but by 54% on a figure that was free to measure.
    """
    import imageio.v3 as iio
    import numpy as np

    from src.__main__ import extend_video_omni

    frames = [
        np.full((32, 32, 3), (i * 3) % 256, dtype=np.uint8) for i in range(24 * 3)
    ]
    buf = BytesIO()
    iio.imwrite(buf, frames, extension=".mp4", fps=24)
    source = tmp_path / "three.mp4"
    source.write_bytes(buf.getvalue())

    payload = json.loads(
        await extend_video_omni(
            ctx=_ctx(tmp_path),
            prompt="Continue.",
            input_video_uri=f"file://{source}",
            resolution="360p",
            dry_run=True,
        )
    )
    assert payload["source_duration_seconds"] == pytest.approx(3.0, abs=0.2)
    # 3s source + one 10s increment, not the 10s-maximum assumption.
    assert payload["turn_output_seconds"][0] == pytest.approx(13.0, abs=0.2)
    assert payload["assembled_seconds"] == pytest.approx(13.0, abs=0.2)


@pytest.mark.asyncio
@pytest.mark.timeout(20.0)
async def test_the_pre_flight_refuses_an_over_length_local_source(
    tmp_path: Path,
) -> None:
    """A quote must not price a call the real run refuses."""
    import imageio.v3 as iio
    import numpy as np

    from src.__main__ import extend_video_omni

    frames = [
        np.full((32, 32, 3), (i * 3) % 256, dtype=np.uint8) for i in range(24 * 13)
    ]
    buf = BytesIO()
    iio.imwrite(buf, frames, extension=".mp4", fps=24)
    source = tmp_path / "thirteen.mp4"
    source.write_bytes(buf.getvalue())

    payload = json.loads(
        await extend_video_omni(
            ctx=_ctx(tmp_path),
            prompt="Continue.",
            input_video_uri=f"file://{source}",
            dry_run=True,
        )
    )
    assert "10s or shorter" in payload["error"]
    assert "estimated_cost" not in payload
