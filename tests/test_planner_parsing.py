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


def _tool_ctx(tmp_path: Path, *, vertexai: bool = True) -> MagicMock:
    """A context the dry_run paths can read, with a non-Vertex client.

    A bare MagicMock reads as Vertex (every attribute is truthy), which sends
    loop_extend down the branch that demands a GCS target.
    """
    from src.__main__ import AppContext

    videos_dir = tmp_path / "videos"
    videos_dir.mkdir(exist_ok=True)
    client = MagicMock()
    # Vertex: plan_generation here runs backend-agnostic (backend="unknown"),
    # so its routes must be validated on the backend that can serve them. Veo
    # refuses first/last-frame and extension on the Gemini Developer API.
    client._api_client.vertexai = vertexai
    lite_client = MagicMock()
    lite_client._api_client.vertexai = False
    ctx = MagicMock()
    ctx.info = AsyncMock()
    ctx.error = AsyncMock()
    ctx.warning = AsyncMock()
    ctx.request_context.lifespan_context = AppContext(
        data_folder=tmp_path,
        images_dir=tmp_path / "images",
        videos_dir=videos_dir,
        client=client,
        # Lite is served only by the Gemini API even on a Vertex deployment,
        # so the planner's Lite routes need a client to land on.
        gemini_api_client=lite_client,
        video_gcs_bucket="gs://bkt/out/" if vertexai else None,
        allowed_gcs_buckets=frozenset({"bkt", "bucket"}),
    )
    return ctx


# Half of the tool payload's last reported decimal: the only difference a
# quote comparison may show. A hair over, because a figure that sits exactly
# on the half (a 3s 360p draft, $0.0704815) differs from its rounded payload
# by 5e-7 plus float noise.
_PAYLOAD_ROUNDING_USD = 5e-7 * (1 + 1e-6)

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
    # Nothing is recorded for int-1, so the route leaves omni_model unset and
    # the tool continues on whatever model holds the interaction -- the
    # default when it has no record, which is the model the route is priced
    # for. Naming a model here was how a plan handed over edits the tool
    # refused.
    assert top.model in OMNI_MODELS
    assert "omni_model" not in top.params
    assert top.cost is not None
    assert top.cost.usd == pytest.approx(
        await _tool_quote(top, tmp_path), abs=_PAYLOAD_ROUNDING_USD
    )
    # The route still shows the (unsent) duration, so the quote's basis has
    # to be stated or the two numbers look like a contradiction. It used to
    # add that the tool "quotes higher" for an already-extended source; 1.2
    # edits sources of at most 15s, so the tool refuses one instead.
    assert any("per-render maximum" in caveat for caveat in top.caveats)
    assert any(
        "edits clips up to 15s; a longer one is refused" in caveat
        for caveat in top.caveats
    ), top.caveats
    assert not any("quotes higher" in caveat for caveat in top.caveats)


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
    # One render on Omni 1.2, a seed plus loop_extend on Veo: both shapes in
    # one plan, each priced as its tool prices it.
    ("a 12 second video of a dog", None),
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


@pytest.mark.asyncio
@pytest.mark.timeout(120.0)
@pytest.mark.parametrize("backend", ["gemini_api", "vertex"])
@pytest.mark.parametrize(
    ("intent", "constraints"),
    [
        pytest.param(intent, constraints, id=intent.replace(" ", "_")[:40])
        for intent, constraints in _INVARIANT_INTENTS
    ],
)
async def test_the_planner_never_silently_hands_over_a_route_the_tool_refuses(
    intent: str, constraints: RoutingConstraints | None, backend: str, tmp_path: Path
) -> None:
    """Planner-tool agreement, on the backend the request will actually run on.

    The agreement test above runs the planner backend-agnostic and validates
    on Vertex, so it could not see the class of bug that has now recurred
    three times: the planner recommending, on the Gemini Developer API, a
    Veo mode that backend refuses — first/last-frame, extension, references,
    4K — with a clean quote beside it. Here the planner is told the backend,
    the tool runs on the same one, and for every route either the tool quotes
    and the two agree, or the plan carries a conflict that says why it won't.
    A route the tool refuses with no conflict in the plan is the failure.
    """
    import dataclasses

    given = constraints or RoutingConstraints()
    # _tool_ctx always carries a Gemini API client, and the server sends omni
    # there whenever a key is present -- so the planner is told, as the MCP
    # tool tells it. Without this the planner priced Vertex's 30s upload
    # ceiling for a chain the tool quoted on the Developer API.
    plan = plan_generation(
        intent,
        dataclasses.replace(given, backend=backend, gemini_api_key_available=True),
    )
    assert plan.routes or plan.conflicts, "no routes and no explanation"
    import src.__main__ as server

    for route in plan.routes:
        kwargs = dict(_MISSING_URI_STUBS.get(route.tool, {}))
        kwargs.update(route.params)
        payload = json.loads(
            await getattr(server, route.tool)(
                ctx=_tool_ctx(tmp_path, vertexai=backend == "vertex"),
                dry_run=True,
                **kwargs,
            )
        )
        if "error" in payload:
            assert plan.conflicts, (
                f"{route.tool}/{route.model} on {backend}: the tool refuses "
                f"({payload['error'][:90]}) and the plan says nothing about it"
            )
            continue
        assert route.cost is not None
        assert route.cost.usd == pytest.approx(
            float(payload["estimated_cost"]["usd"]), abs=_PAYLOAD_ROUNDING_USD
        ), f"{route.tool}/{route.model} on {backend}"


@pytest.mark.asyncio
@pytest.mark.timeout(60.0)
@pytest.mark.parametrize("recorded", [None, *OMNI_MODELS])
@pytest.mark.parametrize(
    ("intent", "extra"),
    [
        pytest.param("make it night", {}, id="edit"),
        pytest.param(
            "extend this clip by 10 seconds", {"media_kind": "video"}, id="extend"
        ),
        pytest.param("upscale it to 1080p", {"backend": "vertex"}, id="upscale"),
    ],
)
async def test_every_continuation_route_is_one_the_tool_accepts(
    recorded: str | None,
    intent: str,
    extra: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """plan_generation(intent, previous_interaction_id) offered a continuation
    per omni model, each naming its own omni_model, and the tools refuse every
    model but the one that recorded the interaction: the top route was refused
    whenever the interaction lived on the other model. Whatever is recorded,
    each route the plan hands over now runs, and anything it cannot hand over
    is a conflict instead."""
    import src.__main__ as server

    vertex = extra.get("backend") == "vertex"

    async def prior(videos_dir: Path, interaction_id: str | None) -> Any:
        return server.PriorInteraction(
            model=recorded,
            backend="vertex" if vertex else "gemini_api",
            resolution="720p",
            duration_seconds=6.0,
        )

    monkeypatch.setattr(server, "_prior_interaction_or_empty", prior)
    plan = plan_generation(
        intent,
        RoutingConstraints(
            previous_interaction_id="abc",
            previous_interaction_model=recorded,
            gemini_api_key_available=not vertex,
            **extra,
        ),
    )
    assert plan.routes or plan.conflicts
    for route in plan.routes:
        kwargs = dict(route.params)
        payload = json.loads(
            await getattr(server, route.tool)(
                ctx=_tool_ctx(tmp_path, vertexai=vertex), dry_run=True, **kwargs
            )
        )
        assert "error" not in payload, (route.tool, route.model, payload["error"])


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
@pytest.mark.parametrize(
    ("intent", "extra", "tool"),
    [
        pytest.param("make it night", {}, "edit_video", id="edit"),
        pytest.param(
            "extend this clip by 10 seconds",
            {"media_kind": "video"},
            "extend_video_omni",
            id="extend",
        ),
        pytest.param(
            "upscale it to 1080p",
            {"backend": "vertex"},
            "upscale_video_omni",
            id="upscale",
        ),
    ],
)
async def test_a_retired_interaction_gets_the_same_way_out_from_plan_and_tool(
    intent: str,
    extra: dict[str, Any],
    tool: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Continuing an interaction recorded on gemini-omni-1.1-flash is refused
    by the tool; the plan refuses it too, and names the same tool to pass the
    saved clip to as input_video_uri."""
    import src.__main__ as server
    from src.omni import OMNI_1_1_MODEL

    vertex = extra.get("backend") == "vertex"

    async def prior(videos_dir: Path, interaction_id: str | None) -> Any:
        return server.PriorInteraction(
            model=OMNI_1_1_MODEL,
            backend="vertex" if vertex else "gemini_api",
            video_url="gs://bucket/old11.mp4",
        )

    monkeypatch.setattr(server, "_prior_interaction_or_empty", prior)
    plan = plan_generation(
        intent,
        RoutingConstraints(
            previous_interaction_id="abc",
            previous_interaction_model=OMNI_1_1_MODEL,
            previous_interaction_video_url="gs://bucket/old11.mp4",
            gemini_api_key_available=not vertex,
            **extra,
        ),
    )
    assert plan.routes == ()
    [conflict] = plan.conflicts
    params: dict[str, Any] = {"previous_interaction_id": "abc"}
    if tool != "upscale_video_omni":
        params["prompt"] = intent
    payload = json.loads(
        await getattr(server, tool)(
            ctx=_tool_ctx(tmp_path, vertexai=vertex), dry_run=True, **params
        )
    )
    refusal = payload["error"]
    way_out = conflict.resolution.split(" call ")[1].split(" with ")[0]
    assert f"call {way_out} with input_video_uri=" in refusal, (way_out, refusal)
    assert conflict.resolution in refusal, (conflict.resolution, refusal)
    assert "input_video_uri='gs://bucket/old11.mp4'" in conflict.resolution


def test_on_the_gemini_api_the_planner_routes_4k_and_references_to_omni() -> None:
    """The capability table was per model; four Veo capabilities are per backend.

    Veo 4K on the Gemini Developer API 400s and reference images return an
    empty result that still bills. The planner offered Veo for both. Omni
    renders 4K and takes references on that backend, so it is not a conflict
    to report — it is a different right answer.
    """
    for intent, kwargs in (
        ("a 4K video of a skyline", {}),
        ("a video of this mascot walking", {"num_reference_images": 1}),
    ):
        plan = plan_generation(
            intent, RoutingConstraints(backend="gemini_api", **kwargs)
        )
        assert plan.routes, intent
        assert plan.routes[0].tool == "generate_video_omni", (intent, plan.routes[0])
        assert not any(r.model.startswith("veo") for r in plan.routes), intent
    # And Veo stays the 4K answer where it can render 4K.
    vertex = plan_generation(
        "a 4K video of a skyline", RoutingConstraints(backend="vertex")
    )
    assert vertex.routes[0].model.startswith("veo")


def test_a_gemini_api_clip_plan_hands_over_no_veo_clip() -> None:
    """generate_clip renders every beat with Veo, which the Gemini Developer API
    no longer serves (switched off 2026-10-22). Handing one over there would
    be a plan that cannot run, so the plan says what to do instead."""
    plan = plan_generation(
        "a 3 shot commercial with crossfade transitions",
        RoutingConstraints(backend="gemini_api"),
    )
    assert not any(r.tool == "generate_clip" for r in plan.routes)
    conflict = next(c for c in plan.conflicts if c.code == "no_model_can_serve_request")
    assert "gemini-omni-1.2-flash" in conflict.resolution
    assert "Vertex" in conflict.resolution


def test_an_all_excluded_plan_explains_itself_instead_of_returning_nothing() -> None:
    """Zero routes and zero conflicts is silence, and silence reads as a bug.

    A 4K multi-shot brief on the Gemini Developer API: Veo cannot render 4K
    there and generate_clip cannot run omni, so every candidate is excluded.
    The rules had computed the fix for each exclusion and the plan threw them
    away on the way out.
    """
    plan = plan_generation(
        "a 3 shot 4K commercial for sneakers", RoutingConstraints(backend="gemini_api")
    )
    assert not plan.routes
    codes = [c.code for c in plan.conflicts]
    assert "no_model_can_serve_request" in codes
    conflict = next(c for c in plan.conflicts if c.code == "no_model_can_serve_request")
    assert "gemini-omni-1.2-flash" in conflict.resolution
    assert "Vertex" in conflict.resolution


# ============================================================================
# Upscale: a follow-on to a render, or an operation on an existing video
# ============================================================================

_VERTEX = RoutingConstraints(backend="vertex", gemini_api_key_available=False)


@pytest.mark.parametrize(
    ("intent", "target"),
    [
        pytest.param(
            "Generate a 6 second video of a red fox running through snow, "
            "then upscale it to 4K",
            "4K",
            id="generate_then_upscale_to_4k",
        ),
        pytest.param(
            "Make a cinematic video of a sunrise and upscale to 1080p",
            "1080p",
            id="make_and_upscale_to_1080p",
        ),
        pytest.param(
            "a video of a cat, then upscale it to 4K", "4K", id="verbless_brief"
        ),
        pytest.param(
            "Create a video of a fox, upscale it to 4K, and keep it moody",
            "4K",
            id="upscale_clause_mid_brief",
        ),
    ],
)
def test_a_generate_then_upscale_intent_plans_the_render_first(
    intent: str, target: str
) -> None:
    """The upscale verb used to replace video planning outright: these came
    back as ONE upscale_video_omni route with no source and no render."""
    plan = plan_generation(intent, _VERTEX)
    tools = {route.tool for route in plan.routes}
    assert "upscale_video_omni" not in tools
    assert tools <= {"generate_video", "generate_video_omni"}, tools

    render = plan.recommended
    assert render is not None
    # The render is prompted and sized without the upscale clause: left in,
    # "4K" made the render itself 4K and the prompt asked the model to upscale.
    assert "upscale" not in render.params["prompt"].lower()
    assert render.params.get("resolution") in (None, "720p")
    assert "moody" in render.params["prompt"] or "moody" not in intent

    *_, source_step, upscale = plan.workflow
    assert upscale.tool == "upscale_video_omni"
    assert upscale.order == source_step.order + 1
    assert upscale.params["resolution"] == target
    assert upscale.params["source_resolution"] == "720p"
    source = upscale.params.get("input_video_uri") or upscale.params.get(
        "previous_interaction_id"
    )
    assert source is not None and f"step {source_step.order}" in source
    assert any("workflow step" in note for note in plan.notes)


@pytest.mark.parametrize(
    "intent",
    [
        pytest.param("Upscale my video to 4K", id="possessive"),
        pytest.param("Take the video I generated and upscale it to 4K", id="past"),
        pytest.param("Render my draft video and upscale it to 4K", id="render_my"),
        pytest.param("A clip I shot yesterday, upscale it to 4K", id="a_clip_i_shot"),
        pytest.param(
            "Don't generate a new video, just upscale my clip to 1080p",
            id="negated_creation",
        ),
    ],
)
def test_upscaling_an_existing_video_is_still_a_standalone_upscale(
    intent: str,
) -> None:
    """Only a request for a NEW video ahead of the upscale makes it a
    follow-on; a clip the caller already has is upscaled directly."""
    plan = plan_generation(intent, _VERTEX)
    assert [route.tool for route in plan.routes] == ["upscale_video_omni"]
    assert plan.workflow == ()


@pytest.mark.parametrize(
    ("constraints", "source_key"),
    [
        pytest.param(
            RoutingConstraints(
                backend="vertex",
                gemini_api_key_available=False,
                pinned_model="gemini-omni-1.2-flash",
            ),
            "previous_interaction_id",
            id="omni_1_2_on_vertex_upscales_in_place",
        ),
        pytest.param(
            RoutingConstraints(
                backend="vertex",
                gemini_api_key_available=True,
                pinned_model="gemini-omni-1.2-flash",
            ),
            "input_video_uri",
            id="omni_on_the_gemini_api_hands_over_the_file",
        ),
        pytest.param(
            RoutingConstraints(
                backend="vertex", pinned_model="veo-3.1-fast-generate-001"
            ),
            "input_video_uri",
            id="veo_hands_over_the_file",
        ),
    ],
)
def test_the_follow_on_upscale_names_a_source_the_tool_accepts(
    constraints: RoutingConstraints, source_key: str
) -> None:
    """upscale_video_omni takes a previous interaction only from 1.2 on
    Vertex; anything else must be passed as the rendered file."""
    plan = plan_generation(
        "Generate a 6 second video of a fox, then upscale it to 4K", constraints
    )
    upscale = plan.workflow[-1]
    assert upscale.tool == "upscale_video_omni"
    assert set(upscale.params) & {"previous_interaction_id", "input_video_uri"} == {
        source_key
    }


def test_a_render_too_long_to_upscale_is_planned_at_the_target_instead() -> None:
    """upscale_video_omni takes one video of at most 15s; a 30s render is an
    extension chain, so the 4K is asked of the render and the plan says so."""
    plan = plan_generation(
        "Generate a 30 second video of a fox and upscale it to 4K", _VERTEX
    )
    assert all(step.tool != "upscale_video_omni" for step in plan.workflow)
    assert plan.request.resolution == "4K"
    assert any("No upscale step is planned" in note for note in plan.notes)


def test_on_the_gemini_api_the_target_goes_to_the_render() -> None:
    """Upscaling is Vertex-only; on the Gemini API rendering at the target is
    the one way to reach it."""
    plan = plan_generation(
        "Make a cinematic video of a sunrise and upscale to 1080p",
        RoutingConstraints(backend="gemini_api"),
    )
    assert plan.recommended is not None
    assert plan.recommended.params.get("resolution") == "1080p"
    assert all(step.tool != "upscale_video_omni" for step in plan.workflow)
    assert any("Vertex AI only" in note for note in plan.notes)


def test_an_upscale_to_720p_renders_its_source_at_360p() -> None:
    """360p is the only source an upscale to 720p accepts, so the render is
    planned there rather than at 720p and then refused as a 720p->720p pair."""
    plan = plan_generation("Generate a video of a fox then upscale it to 720p", _VERTEX)
    assert plan.recommended is not None
    assert plan.recommended.params["resolution"] == "360p"
    upscale = plan.workflow[-1]
    assert upscale.tool == "upscale_video_omni"
    assert upscale.params["source_resolution"] == "360p"
    assert not plan.conflicts


# Upscale outputs past the 4 MB inline limit still come back inline on Vertex
# when no output_gcs_uri is given; the plan recommends a bucket for them, in
# the words of the policy the tool itself applies, rather than calling it a
# requirement -- unless the server is set to refuse them.


def _upscale_policy(target: str, seconds: float, *, bound: bool) -> str | None:
    """What upscale_video_omni's own pre-render policy says for this upscale."""
    from src.omni import OMNI_1_2_MODEL, omni_inline_delivery_policy, omni_spec

    return omni_inline_delivery_policy(
        omni_spec(OMNI_1_2_MODEL),
        vertexai=True,
        output_gcs_uri=None,
        resolution=target,
        task="upscale",
        output_seconds=seconds,
        seconds_are_bound=bound,
    )


@pytest.mark.parametrize("target", ["1080p", "4K"])
def test_a_large_standalone_upscale_recommends_output_gcs_uri(
    target: str,
) -> None:
    plan = plan_generation(f"Upscale my video from 720p to {target}", _VERTEX)
    route = plan.recommended
    assert route is not None and route.tool == "upscale_video_omni"
    caveat = next(c for c in route.caveats if "output_gcs_uri" in c)
    # The tool's own warning for the same upscale, then why a bucket is the
    # safer choice. It used to say the tool refused the call.
    policy = _upscale_policy(target, 15.0, bound=True)
    assert policy is not None and caveat.startswith(policy)
    assert "billed" in caveat[len(policy) :]
    assert "refuse" not in caveat
    # A RoutedCall names a key it cannot fill rather than inventing a value.
    assert "output_gcs_uri" not in route.params


def test_a_720p_upscale_of_unknown_length_names_the_10s_condition() -> None:
    """Over 10s a 1.2 output is past the inline limit too; with the source
    length unknown, the plan states the condition rather than asserting it."""
    route = plan_generation("Upscale my video from 360p to 720p", _VERTEX).recommended
    assert route is not None
    caveat = next(c for c in route.caveats if "output_gcs_uri" in c)
    assert caveat.startswith("If the source runs longer than 10s")
    assert "refuse" not in caveat


def test_a_short_720p_follow_on_upscale_needs_no_output_gcs_uri() -> None:
    plan = plan_generation(
        "Generate a 6 second video of a fox then upscale it to 720p", _VERTEX
    )
    upscale = plan.workflow[-1]
    assert upscale.tool == "upscale_video_omni"
    assert "output_gcs_uri" not in upscale.params
    assert "output_gcs_uri" not in upscale.rationale


def test_a_large_follow_on_upscale_recommends_output_gcs_uri_without_a_placeholder() -> (
    None
):
    """The step carried output_gcs_uri='<gs:// destination ...>', which reads as
    a value the step cannot run without; the tool delivers inline without one,
    so the recommendation rides in the rationale and the params run as given."""
    plan = plan_generation(
        "Generate a 6 second video of a fox, then upscale it to 4K", _VERTEX
    )
    upscale = plan.workflow[-1]
    assert upscale.tool == "upscale_video_omni"
    assert "output_gcs_uri" not in upscale.params
    assert "output_gcs_uri" in upscale.rationale
    assert "refuse" not in upscale.rationale


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
@pytest.mark.parametrize(
    ("intent", "extra", "tool"),
    [
        pytest.param("a 4K video of a dog", {}, "generate_video_omni", id="4k"),
        pytest.param("a 12 second video of a dog", {}, "generate_video_omni", id="12s"),
        pytest.param(
            "make it night", {"previous_interaction_id": "abc"}, "edit_video", id="edit"
        ),
        pytest.param(
            "extend this clip by 10 seconds",
            {"previous_interaction_id": "abc", "media_kind": "video"},
            "extend_video_omni",
            id="extend",
        ),
        pytest.param(
            "Upscale my video from 720p to 4K",
            {"source_video_uri": "gs://bucket/a.mp4"},
            "upscale_video_omni",
            id="upscale",
        ),
    ],
)
async def test_the_plans_delivery_advice_is_the_tools_own_warning(
    intent: str, extra: dict[str, Any], tool: str, tmp_path: Path
) -> None:
    """One policy for the four Omni tools, and the plan states it in the words
    each tool's dry run uses for the same call: a large output on Vertex AI
    without output_gcs_uri is delivered inline, with a warning."""
    import dataclasses

    import src.__main__ as server

    plan = plan_generation(intent, dataclasses.replace(_VERTEX, **extra))
    route = next(r for r in plan.routes if r.tool == tool)
    ctx = _tool_ctx(tmp_path)
    # No Gemini API key, so omni runs on Vertex AI, as the plan was told.
    ctx.request_context.lifespan_context = dataclasses.replace(
        ctx.request_context.lifespan_context, gemini_api_client=None
    )
    payload = json.loads(
        await getattr(server, tool)(ctx=ctx, dry_run=True, **route.params)
    )
    assert "error" not in payload, payload
    [warning] = [w for w in payload.get("warnings", []) if "output_gcs_uri" in w]
    assert any(caveat.startswith(warning) for caveat in route.caveats), (
        warning,
        route.caveats,
    )


def test_a_server_that_refuses_large_inline_output_gets_a_required_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The policy's one switch, flipped: the tools refuse a large inline
    output before rendering, so the plan says so and the follow-on step
    carries the placeholder it cannot run without."""
    import src.omni as omni

    monkeypatch.setattr(omni, "OMNI_REFUSE_LARGE_INLINE_ON_VERTEX", True)
    standalone = plan_generation("Upscale my video from 720p to 4K", _VERTEX)
    route = standalone.recommended
    assert route is not None
    caveat = next(c for c in route.caveats if "output_gcs_uri" in c)
    assert caveat.startswith("Before rendering, upscale_video_omni refuses")
    assert "Nothing was rendered" not in caveat

    chained = plan_generation(
        "Generate a 6 second video of a fox, then upscale it to 4K", _VERTEX
    )
    step = chained.workflow[-1]
    assert step.params["output_gcs_uri"].startswith("<")
    assert "refuses" in step.rationale


# ============================================================================
# Integration follow-ups: the render/upscale split
# ============================================================================


@pytest.mark.parametrize(
    "intent",
    [
        pytest.param(
            "Create a sharper version of the video and upscale it to 4K",
            id="the_before_the_noun",
        ),
        pytest.param(
            "Make an upscaled copy of the video: upscale it to 4K",
            id="copy_of_the_video",
        ),
        pytest.param(
            "A video of my dog that I recorded, upscale it to 4K",
            id="verbless_recorded",
        ),
        pytest.param("A video from my phone: upscale it to 4K", id="verbless_phone"),
        pytest.param(
            "Generate a video of a fox. Upscale my video of a cat to 4K.",
            id="upscale_names_its_own_source",
        ),
    ],
)
def test_an_existing_clip_is_not_turned_into_a_paid_render(intent: str) -> None:
    """Each of these describes a clip the caller has; reading it as a request
    for a new one planned a render nobody asked to pay for, and chained the
    cat upscale onto a fox render."""
    plan = plan_generation(intent, _VERTEX)
    assert [route.tool for route in plan.routes] == ["upscale_video_omni"]
    assert all(step.tool != "generate_video" for step in plan.workflow)


@pytest.mark.parametrize(
    ("intent", "prompt"),
    [
        pytest.param(
            "Generate a video of a fox. Then upscale it to 4K.",
            "Generate a video of a fox.",
            id="sentence_end",
        ),
        pytest.param(
            "Generate a video of a fox! And then upscale it to 4K!",
            "Generate a video of a fox!",
            id="exclamation",
        ),
        pytest.param(
            "Create a video of a fox (then upscale it to 4K)",
            "Create a video of a fox",
            id="parenthesised",
        ),
        pytest.param(
            "Generate a video of a fox, then upscale it to 4K (keep it moody)",
            "Generate a video of a fox (keep it moody)",
            id="brackets_in_the_brief_kept",
        ),
        pytest.param(
            "Generate a video of a fox, then upscale it to 4K resolution",
            "Generate a video of a fox",
            id="target_qualifier",
        ),
        pytest.param(
            "Generate a video of a fox, then upscale it to 4K, and keep it moody",
            "Generate a video of a fox, and keep it moody",
            id="brief_continues",
        ),
    ],
)
def test_the_render_prompt_carries_no_debris_from_the_upscale_clause(
    intent: str, prompt: str
) -> None:
    """Splitting left "a fox..", "a fox!!" and "a fox (" in the prompt sent
    to the video model."""
    render = plan_generation(intent, _VERTEX).recommended
    assert render is not None and render.tool != "upscale_video_omni"
    assert render.params["prompt"] == prompt


def test_a_modifier_after_the_upscale_target_stays_with_the_render() -> None:
    """The upscale clause ran to the next clause break, so "with audio" left
    with it and the render was planned silent."""
    plan = plan_generation(
        "Make a video of a fox, then upscale it to 4K with audio", _VERTEX
    )
    render = plan.recommended
    assert render is not None
    assert render.params["prompt"] == "Make a video of a fox with audio"
    assert render.params.get("include_audio") is True
    assert plan.workflow[-1].tool == "upscale_video_omni"


def test_a_10s_omni_render_upscaled_to_720p_recommends_output_gcs_uri() -> None:
    """An Omni render lands about one frame long, and upscale_video_omni judges
    inline delivery on the measured ~10.04s: judged on the nominal 10s, the
    plan said nothing about a step the tool flags as large."""
    plan = plan_generation(
        "Generate a 10 second video of a fox then upscale it to 720p",
        RoutingConstraints(
            backend="vertex",
            gemini_api_key_available=False,
            pinned_model="gemini-omni-1.2-flash",
        ),
    )
    upscale = plan.workflow[-1]
    assert upscale.tool == "upscale_video_omni"
    assert "output_gcs_uri" not in upscale.params
    from src.pricing import quote_duration_for

    policy = _upscale_policy(
        "720p", quote_duration_for("gemini-omni-1.2-flash", 10.0), bound=False
    )
    assert policy is not None and policy in upscale.rationale


def test_a_standalone_upscale_of_an_unknown_source_is_quoted_as_an_upper_bound() -> (
    None
):
    """With no source resolution the route carried no cost, and the plan said
    the pricing module was not installed, which it was."""
    from src.pricing import estimate_omni_upscale_cost

    plan = plan_generation("Upscale my video to 1080p", _VERTEX)
    route = plan.recommended
    assert route is not None and route.cost is not None
    # 360p to 1080p is the dearer of the two transitions that reach 1080p.
    assert route.cost.usd == pytest.approx(
        estimate_omni_upscale_cost(15, "360p", "1080p").usd
    )
    assert any("dearer" in caveat for caveat in route.caveats)
    assert not any("not installed" in note for note in plan.notes)


def test_a_route_the_installed_price_book_cannot_price_is_not_blamed_on_a_missing_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.pricing as pricing

    def _cannot_price(*args: object, **kwargs: object) -> None:
        raise RuntimeError("no rate")

    monkeypatch.setattr(pricing, "estimate_image_cost", _cannot_price)
    plan = plan_generation("a picture of a cat")
    assert all(route.cost is None for route in plan.routes)
    assert not any("not installed" in note for note in plan.notes)
    assert any("could not price" in note for note in plan.notes)


# ============================================================================
# Render-then-upscale: the bare verb, and a 4K ask in both clauses
# ============================================================================


@pytest.mark.parametrize(
    ("intent", "prompt"),
    [
        pytest.param(
            "Make a video of a dog then upscale", "Make a video of a dog", id="bare"
        ),
        pytest.param(
            "Make a video of a dog, then upscale.",
            "Make a video of a dog.",
            id="period",
        ),
        pytest.param(
            "Make a video of a dog, then upscale, and keep it moody",
            "Make a video of a dog, and keep it moody",
            id="brief_continues",
        ),
        pytest.param(
            "Generate a video of a fox, then upscale the result to 4K",
            "Generate a video of a fox",
            id="the_result",
        ),
    ],
)
def test_a_bare_or_result_upscale_is_planned_after_the_render(
    intent: str, prompt: str
) -> None:
    """ "then upscale" with no object was not read as an upscale at all: the
    render was prompted with "then upscale" and no upscale step followed."""
    plan = plan_generation(intent, _VERTEX)
    render = plan.recommended
    assert render is not None and render.tool != "upscale_video_omni"
    assert render.params["prompt"] == prompt
    upscale = plan.workflow[-1]
    assert upscale.tool == "upscale_video_omni"
    assert upscale.params["source_resolution"] == "720p"
    assert not plan.conflicts


@pytest.mark.parametrize(
    "intent",
    [
        pytest.param("Make a 4K video of a dog, then upscale it to 4K", id="both"),
        pytest.param("Make a 4K video of a dog, then upscale it", id="render_only"),
    ],
)
def test_a_4k_render_clause_is_the_upscales_target(intent: str) -> None:
    """The render clause was re-read on its own, inferred 4K from "a 4K video",
    planned a 4K render, and reported its 4K-to-4K upscale as an
    upscale_resolution conflict beside it."""
    plan = plan_generation(intent, _VERTEX)
    assert not plan.conflicts, plan.conflicts
    render = plan.recommended
    assert render is not None
    assert render.params.get("resolution") in (None, "720p")
    upscale = plan.workflow[-1]
    assert upscale.tool == "upscale_video_omni"
    assert upscale.params["resolution"] == "4K"
    assert upscale.params["source_resolution"] == "720p"


@pytest.mark.parametrize(
    "intent",
    [
        pytest.param(
            "Make a video of a hotel lobby that feels modern and upscale",
            id="adjective_after_and",
        ),
        pytest.param("Make a video of an upscale restaurant", id="adjective"),
        pytest.param(
            "Make a video of a street that was upscale then", id="then_after_it"
        ),
    ],
)
def test_the_adjective_is_still_not_an_upscale(intent: str) -> None:
    plan = plan_generation(intent, _VERTEX)
    render = plan.recommended
    assert render is not None and render.params["prompt"] == intent
    assert all(step.tool != "upscale_video_omni" for step in plan.workflow)


@pytest.mark.parametrize("backend", ["vertex", "gemini_api"])
@pytest.mark.parametrize(
    "intent, marker",
    [
        ("A video of a street that is shabby at first, then upscale", "first"),
        (
            "Make a time-lapse video of a neighborhood, first gritty, then upscale.",
            "first",
        ),
        (
            "A video of a dog walking through a park that turns rustic then upscale",
            "turns",
        ),
        ("Make a video of a restaurant going from cheap then upscale", "going"),
        ("Make a video of a cafe that becomes elegant then upscale", "becomes"),
    ],
)
def test_then_upscale_after_a_change_of_state_is_the_adjective(
    intent: str, marker: str, backend: str
) -> None:
    """A "then" that orders two looks makes "upscale" the second look. Read
    as the verb, the render lost the brief's second half, Vertex gained a
    paid upscale step, and the Gemini API's render was raised to 1080p "as if
    the brief had asked for" it. The note says how to ask for the verb."""
    plan = plan_generation(intent, RoutingConstraints(backend=backend))
    render = plan.recommended
    assert render is not None and render.params["prompt"] == intent
    assert all(step.tool != "upscale_video_omni" for step in plan.workflow)
    assert render.params.get("resolution") in (None, "720p")
    [note] = [n for n in plan.notes if "adjective" in n]
    assert f"('{marker}')" in note
    assert "say 'then upscale it'" in note


@pytest.mark.parametrize(
    "intent",
    [
        "Make a video of a dog then upscale",
        "Make a video of a dog then upscale, and keep it moody",
        # A change of state in an earlier sentence does not reach this one.
        "Make a video of a dog. It starts calm. Then upscale.",
        # With its object it is the verb, whatever came before.
        "Make a video of a dog that turns around, then upscale it",
    ],
)
def test_then_upscale_with_no_change_of_state_is_still_the_verb(intent: str) -> None:
    plan = plan_generation(intent, _VERTEX)
    assert [step.tool for step in plan.workflow][-1] == "upscale_video_omni"
    assert not any("adjective" in note for note in plan.notes)


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
@pytest.mark.parametrize(
    "intent, constraints",
    [
        ("Create a 12 second video of a dog", RoutingConstraints(budget="best")),
        (
            "Create a 12 second video of a dog",
            RoutingConstraints(pinned_model="veo-3.1-generate-001"),
        ),
        ("Create a 30 second video of a dog with dialogue", None),
    ],
    ids=["best-12s", "pinned-12s", "audio-30s"],
)
async def test_a_veo_chain_says_what_its_seed_costs(
    intent: str, constraints: RoutingConstraints | None, tmp_path: Path
) -> None:
    """A seed-and-extend chain's quote is loop_extend's own, which covers the
    extensions alone: a 12s Veo standard chain read as $2.80 and costs $6.00
    with the seed its own caveat asks for. The seed step names the figure
    generate_video quotes for it, and the route and the step say the total."""
    import src.__main__ as server

    plan = plan_generation(intent, constraints)
    seed, extend = plan.workflow
    assert (seed.tool, extend.tool) == ("generate_video", "loop_extend")
    top = plan.recommended
    assert top is not None and top.tool == "loop_extend" and top.cost is not None
    payload = json.loads(
        await server.generate_video(
            ctx=_tool_ctx(tmp_path), dry_run=True, **seed.params
        )
    )
    seed_usd = float(payload["estimated_cost"]["usd"])
    assert f"(est. ${seed_usd:.2f})" in seed.rationale
    total = f"${seed_usd + top.cost.usd:.2f}"
    assert f"about {total} for the chain" in extend.rationale
    assert any(
        f"The seed costs about ${seed_usd:.2f}, so the chain costs about {total}"
        in caveat
        for caveat in top.caveats
    ), top.caveats
    # The seed renders what the chain extends, audio included.
    assert seed.params["include_audio"] == top.params["include_audio"]


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
@pytest.mark.parametrize("bucket", [True, False], ids=["bucket", "no-bucket"])
async def test_a_vertex_veo_chain_says_whether_it_needs_output_gcs_uri(
    tmp_path: Path, bucket: bool
) -> None:
    """loop_extend on Vertex AI is refused, dry run included, with neither an
    output_gcs_uri nor the server's VIDEO_GCS_BUCKET. The plan only ever said
    "requires output_gcs_uri" -- false where a bucket is configured, and on a
    server with none its workflow step had no key to fill in."""
    import src.__main__ as server

    ctx = _tool_ctx(tmp_path)
    if not bucket:
        ctx.request_context.lifespan_context.video_gcs_bucket = None
    fn = getattr(server.plan_generation, "fn", server.plan_generation)
    plan = json.loads(
        (
            await fn(
                ctx,
                "Create a 12 second video of a dog",
                pinned_model="veo-3.1-fast-generate-001",
            )
        )[0].text
    )
    [route] = plan["routes"]
    assert route["tool"] == "loop_extend"
    seed, extend = plan["workflow"]
    stubbed = {**route["params"], "video_uri": "gs://bucket/a.mp4"}
    payload = json.loads(await server.loop_extend(ctx=ctx, dry_run=True, **stubbed))
    if bucket:
        assert "error" not in payload, payload
        assert any("this server's VIDEO_GCS_BUCKET" in c for c in route["caveats"])
        assert "output_gcs_uri" not in extend["params"]
        return
    assert "requires output_gcs_uri" in payload["error"]
    assert any(
        c.startswith("Add output_gcs_uri (a prefix such as gs://bucket/renders/)")
        and "no VIDEO_GCS_BUCKET" in c
        for c in route["caveats"]
    ), route["caveats"]
    assert extend["params"]["output_gcs_uri"].startswith("<gs://")
    # Filled in, the step the plan lays out runs.
    filled = {**stubbed, "output_gcs_uri": "gs://bucket/out.mp4"}
    assert "error" not in json.loads(
        await server.loop_extend(ctx=ctx, dry_run=True, **filled)
    )


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
@pytest.mark.parametrize("backend", ["gemini_api", "vertex"])
async def test_an_implied_omni_extension_opens_with_its_seed(
    tmp_path: Path, backend: str
) -> None:
    """An omni chain past one render had no seed workflow or caveat: its top
    route asked for a previous_interaction_id the caller does not have yet,
    with nothing saying where it comes from or what it costs."""
    import src.__main__ as server

    plan = plan_generation(
        "Create a 20 second video of a dog",
        RoutingConstraints(
            backend=backend,
            gemini_api_key_available=backend == "gemini_api",
            pinned_model="gemini-omni-1.2-flash",
        ),
    )
    [route] = plan.routes
    assert route.tool == "extend_video_omni"
    assert any(
        c.startswith(
            "No clip exists yet: render the opening 15s with generate_video_omni"
        )
        for c in route.caveats
    ), route.caveats
    seed, extend = plan.workflow
    assert (seed.tool, extend.tool) == ("generate_video_omni", "extend_video_omni")
    assert extend.params["previous_interaction_id"] == "<interaction_id from step 1>"
    payload = json.loads(
        await server.generate_video_omni(
            ctx=_tool_ctx(tmp_path, vertexai=backend == "vertex"),
            dry_run=True,
            **seed.params,
        )
    )
    assert "error" not in payload, payload
    assert f"(est. ${float(payload['estimated_cost']['usd']):.2f})" in seed.rationale
