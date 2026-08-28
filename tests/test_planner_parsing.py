"""Tests for the planner's text parsing and its quotes.

Two families live here:

* Parsing — negation scope and duration recognition. Both are pure functions
  of the intent string, so every test is a plain call.
* Quote agreement — the planner's cost for a route must equal what that
  tool's own dry_run reports for the same parameters. Those tests call the
  real tools with dry_run=True (nothing is generated, no network is touched)
  because a mirrored calculation would drift with the thing it mirrors.
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.omni import OMNI_MODELS
from src.routing import (
    MAX_CLIP_BEATS,
    RoutedCall,
    RoutingConstraints,
    infer_signals,
    plan_generation,
)

# ============================================================================
# Negation scope
# ============================================================================


@pytest.mark.parametrize(
    "intent",
    [
        pytest.param("a silent video of rain", id="silent_before_the_kind"),
        pytest.param("mute video of a waterfall", id="mute_before_the_kind"),
        pytest.param("no audio, a video of rain", id="across_a_comma"),
        pytest.param("no logo, a clip of the ocean", id="clip_across_a_comma"),
        pytest.param("skip the intro, video of a sunrise", id="skip_across_a_comma"),
        pytest.param(
            "avoid text. video montage of ocean waves", id="across_a_sentence"
        ),
        pytest.param("without music and video of a river", id="across_a_conjunction"),
    ],
)
def test_a_negated_extra_does_not_cancel_the_media_kind(intent: str) -> None:
    """Every one of these is an explicit video request that planned a still
    image: the backward scan for a negator ran through commas, full stops and
    conjunctions, and "silent"/"mute" negated whatever word followed them.
    """
    assert infer_signals(intent).media_kind == "video"
    assert plan_generation(intent).media_kind == "video"


@pytest.mark.parametrize(
    ("intent", "expect_audio"),
    [
        pytest.param("a short video with audio", True, id="plain_request"),
        pytest.param("a video of steam, no audio", False, id="no_audio"),
        pytest.param("a clip without dialogue", False, id="without"),
        pytest.param("a silent video with no music", False, id="silent_and_no"),
        pytest.param("a clip of traffic with muted sound", False, id="muted"),
        pytest.param("no dialogue, but ambient music", True, id="mixed_keeps_positive"),
    ],
)
def test_negation_still_reads_a_refused_soundtrack_as_refused(
    intent: str, expect_audio: bool
) -> None:
    """The original reason the negation scan exists: "no audio" must not be
    read as a request FOR audio, and "no dialogue, but ambient music" must
    keep the music. Narrowing the scan must not cost either of those.
    """
    assert infer_signals(intent).wants_audio is expect_audio


# ============================================================================
# Duration recognition
# ============================================================================


@pytest.mark.parametrize(
    "intent",
    [
        pytest.param("a retro montage set in the 1970s", id="decade_four_digit"),
        pytest.param("an 80s style video of a neon diner", id="decade_two_digit"),
        pytest.param(
            "the 90s called, they want their aesthetic back", id="decade_as_a_subject"
        ),
        pytest.param("a video celebrating our 100s of customers", id="plural_quantity"),
        pytest.param("product shot of the iPhone 15s in a studio", id="product_name"),
        pytest.param("a video of a 3 m tall robot", id="metres_not_minutes"),
    ],
)
def test_a_bare_unit_without_duration_context_is_not_a_runtime(intent: str) -> None:
    """ "1970s" was read as 1970 seconds and planned a 247-beat clip quoted at
    $197.60; "80s", "100s", "15s" and "3 m" were read the same way.
    """
    assert infer_signals(intent).duration_seconds is None


@pytest.mark.parametrize(
    ("intent", "expected"),
    [
        pytest.param("an 8 second clip", 8.0, id="spelled_out"),
        pytest.param("an 8-second clip", 8.0, id="hyphenated"),
        pytest.param("8s clip", 8.0, id="bare_before_a_noun"),
        pytest.param("30 seconds of footage", 30.0, id="plural_spelled_out"),
        pytest.param("a 2 minute video", 120.0, id="minutes"),
        pytest.param("make it 8s", 8.0, id="bare_after_a_cue"),
        pytest.param("a clip", None, id="no_duration_at_all"),
    ],
)
def test_a_real_runtime_still_parses(intent: str, expected: float | None) -> None:
    assert infer_signals(intent).duration_seconds == expected


def test_a_decade_does_not_inflate_the_plan() -> None:
    """The cost of the misread: a style reference quoted as a 247-render clip."""
    plan = plan_generation("a retro montage set in the 1970s")
    top = plan.recommended
    assert top is not None
    assert len(top.params.get("beats", ())) <= MAX_CLIP_BEATS
    assert top.cost is not None
    assert top.cost.usd < 5.0


def test_an_added_duration_is_not_matched_inside_a_word() -> None:
    """ "lullaby 30 seconds long" contains "by 30 seconds", so the extension
    delta fired on a request that never asked to extend anything.
    """
    signals = infer_signals("a lullaby 30 seconds long")
    assert signals.added_duration_seconds is None
    assert signals.duration_seconds == 30.0


def test_a_real_extension_delta_still_parses() -> None:
    assert (
        infer_signals("extend this by another 30 seconds").added_duration_seconds
        == 30.0
    )


# ============================================================================
# The tool's beat ceiling
# ============================================================================


def test_a_clip_route_never_exceeds_the_tools_beat_limit() -> None:
    """ "a 3 minute montage" planned 23 beats and was presented ready to call
    with a full quote; generate_clip hard-errors above 20.
    """
    plan = plan_generation("a 3 minute montage of city life")
    top = plan.recommended
    assert top is not None
    assert top.tool == "generate_clip"
    assert len(top.params["beats"]) == MAX_CLIP_BEATS
    assert any(str(MAX_CLIP_BEATS) in caveat for caveat in top.caveats)
    assert any("rejects more than" in caveat for caveat in top.caveats)


# ============================================================================
# Quote agreement with the tools themselves
# ============================================================================


def _tool_ctx(tmp_path: Path) -> MagicMock:
    """A context the dry_run paths can read, with a non-Vertex client.

    A bare MagicMock reads as Vertex (every attribute is truthy), which sends
    loop_extend down the branch that demands a GCS target.
    """
    from src.__main__ import AppContext

    videos_dir = tmp_path / "videos"
    videos_dir.mkdir(exist_ok=True)
    client = MagicMock()
    client._api_client.vertexai = False
    ctx = MagicMock()
    ctx.info = AsyncMock()
    ctx.error = AsyncMock()
    ctx.warning = AsyncMock()
    ctx.request_context.lifespan_context = AppContext(
        data_folder=tmp_path,
        images_dir=tmp_path / "images",
        videos_dir=videos_dir,
        client=client,
    )
    return ctx


# Half of the tool payload's last reported decimal: the only difference a
# quote comparison may show.
_PAYLOAD_ROUNDING_USD = 5e-7

# URIs the planner deliberately leaves out (it was never given them) but the
# tool signature requires. Dry runs never fetch them, but they now validate
# local file sources, so every stub uses gs:// (uncheckable offline, still
# priced) rather than a dummy file:// path that would be refused as outside
# DATA_FOLDER. generate_transition frames are now source-validated too.
_MISSING_URI_STUBS: dict[str, dict[str, Any]] = {
    "generate_transition": {
        "first_frame_uri": "gs://bucket/a.png",
        "last_frame_uri": "gs://bucket/b.png",
    },
    "generate_bridge": {
        "from_clip_uri": "gs://bucket/a.mp4",
        "to_clip_uri": "gs://bucket/b.mp4",
    },
    "loop_extend": {"video_uri": "gs://bucket/a.mp4"},
}


async def _tool_quote(route: RoutedCall, tmp_path: Path) -> float:
    """What the tool itself quotes for the parameters the planner emitted."""
    import src.__main__ as server

    kwargs = dict(_MISSING_URI_STUBS.get(route.tool, {}))
    kwargs.update(route.params)
    payload = json.loads(
        await getattr(server, route.tool)(
            ctx=_tool_ctx(tmp_path), dry_run=True, **kwargs
        )
    )
    assert payload.get("dry_run") is True, payload
    # The tool rounds its payload to 6 decimals (cost_to_dict), so the
    # comparison below is to that precision, not to the last float bit.
    return float(payload["estimated_cost"]["usd"])


@pytest.mark.asyncio
@pytest.mark.timeout(10.0)
async def test_bridges_are_quoted_at_the_length_they_render(tmp_path: Path) -> None:
    """generate_clip renders every bridge at 4s; the planner charged them at
    the beat length, quoting $4.00 against the tool's $3.20.
    """
    plan = plan_generation(
        "a 3 shot commercial with crossfade transitions, 24 seconds total"
    )
    top = plan.recommended
    assert top is not None
    assert top.params["add_bridges"] is True
    assert top.cost is not None
    assert top.cost.usd == pytest.approx(
        await _tool_quote(top, tmp_path), abs=_PAYLOAD_ROUNDING_USD
    )


@pytest.mark.asyncio
@pytest.mark.timeout(10.0)
async def test_a_resolution_the_route_cannot_send_is_not_priced(
    tmp_path: Path,
) -> None:
    """generate_clip takes no resolution parameter and always renders 720p,
    yet a 4K ask was priced at the 4K rate — $5.40 against the tool's $1.80 —
    and dropped without a word.
    """
    plan = plan_generation("a 3 shot 4K commercial for sneakers")
    top = plan.recommended
    assert top is not None
    assert top.tool == "generate_clip"
    assert top.cost is not None
    assert top.cost.usd == pytest.approx(
        await _tool_quote(top, tmp_path), abs=_PAYLOAD_ROUNDING_USD
    )
    assert any("720p" in caveat and "4K" in caveat for caveat in top.caveats)


@pytest.mark.asyncio
@pytest.mark.timeout(10.0)
async def test_the_edit_route_quotes_the_same_upper_bound_as_the_tool(
    tmp_path: Path,
) -> None:
    """An edit renders at Omni's maximum whatever the source was, which is
    why edit_video's own dry_run quotes 10s. The planner quoted the request's
    duration instead — $0.61 against $1.02, i.e. an under-stated quote.
    """
    plan = plan_generation(
        "make the sky stormy", RoutingConstraints(previous_interaction_id="int-1")
    )
    top = plan.recommended
    assert top is not None
    assert top.tool == "edit_video"
    # Conversational editing is omni-only; which omni model wins is a ranking
    # decision, but the emitted params must name the same one the route was
    # priced for, or the caller's call and the plan's quote describe
    # different renders.
    assert top.model in OMNI_MODELS
    assert top.params["omni_model"] == top.model
    assert top.cost is not None
    assert top.cost.usd == pytest.approx(
        await _tool_quote(top, tmp_path), abs=_PAYLOAD_ROUNDING_USD
    )
    # The route still shows the (unsent) duration, so the quote's basis has
    # to be stated or the two numbers look like a contradiction — including
    # the one case where the tool quotes HIGHER than the planner, an already-
    # extended source whose length only the tool can read.
    assert any("per-render maximum" in caveat for caveat in top.caveats)
    assert any("quotes higher" in caveat for caveat in top.caveats)


_INVARIANT_INTENTS: list[tuple[str, RoutingConstraints | None]] = [
    ("a video of a cat", None),
    ("an 8 second clip of steam rising", None),
    ("a 3 shot reel with music", None),
    ("a 3 shot commercial with crossfade transitions, 24 seconds total", None),
    ("a 3 shot 4K commercial for sneakers", None),
    ("a 3 minute montage of city life", None),
    ("a 5 shot montage with bridges between every shot", None),
    ("a 1080p video of a skyline", None),
    ("a cheap 4 second clip of steam", RoutingConstraints(budget="cheap")),
    ("a draft video of a robot", RoutingConstraints(is_draft=True)),
    (
        # gs:// frames so the transition route the planner emits carries
        # sources the tool's dry_run prices; local dummy paths would now be
        # refused as outside DATA_FOLDER, and the scheme does not change the
        # route or quote.
        "a crossfade between these frames",
        RoutingConstraints(
            first_frame_uri="gs://bucket/a.png", last_frame_uri="gs://bucket/b.png"
        ),
    ),
    (
        "a 4K crossfade between these frames",
        RoutingConstraints(
            first_frame_uri="gs://bucket/a.png",
            last_frame_uri="gs://bucket/b.png",
            resolution="4K",
        ),
    ),
    (
        # gs:// so the loop_extend route the planner emits carries a source the
        # tool's dry_run prices; a local dummy path would now be refused as
        # outside DATA_FOLDER, and the scheme does not change the route or quote.
        "make it 30 seconds long",
        RoutingConstraints(media_kind="video", source_video_uri="gs://bucket/a.mp4"),
    ),
    ("make the sky stormy", RoutingConstraints(previous_interaction_id="int-1")),
]


@pytest.mark.asyncio
@pytest.mark.timeout(60.0)
@pytest.mark.parametrize(
    ("intent", "constraints"),
    [
        pytest.param(intent, constraints, id=intent.replace(" ", "_")[:40])
        for intent, constraints in _INVARIANT_INTENTS
    ],
)
async def test_every_planned_route_agrees_with_the_tools_own_dry_run(
    intent: str, constraints: RoutingConstraints | None, tmp_path: Path
) -> None:
    """The planner's headline promise: a route it hands over is ready to call
    AND correctly priced. A quote that disagrees with the tool it recommends
    is worse than no quote, and one that under-states is worst of all.
    """
    plan = plan_generation(intent, constraints)
    assert plan.routes
    for route in plan.routes:
        assert route.cost is not None
        assert route.cost.usd == pytest.approx(
            await _tool_quote(route, tmp_path), abs=_PAYLOAD_ROUNDING_USD
        )
