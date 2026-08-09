"""Regression tests for three planner follow-up fixes in src/routing.py.

Each test here fails on the pre-fix code and passes after it:

* FINDING 1 — plan_generation now offers a generate_storyboard previz route
  for a storyboard-flavoured multi-shot brief, priced to match the tool's own
  dry_run.
* FINDING 2 — caption/slug-line/panel/label vocabulary (including plurals)
  now drives needs_text_rendering.
* FINDING 3 — a pinned Veo `-preview` id is normalized to its canonical
  `-001` name before the live-model check, so it is planned rather than
  rejected.

Like test_routing.py, the router is pure/deterministic, so every test is a
plain call — no clients, no network, no clock.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.pricing import estimate_image_cost
from src.routing import (
    RoutedCall,
    RoutingConstraints,
    RoutingPlan,
    infer_signals,
    plan_generation,
)

FLASH = "gemini-3.1-flash-image"
LITE_IMAGE = "gemini-3.1-flash-lite-image"
PRO_IMAGE = "gemini-3-pro-image"
VEO_FAST = "veo-3.1-fast-generate-001"
VEO = "veo-3.1-generate-001"


def _storyboard_route(plan: RoutingPlan) -> RoutedCall | None:
    """The generate_storyboard route on a plan, or None if there is none."""
    for route in plan.routes:
        if route.tool == "generate_storyboard":
            return route
    return None


def _conflict_codes(plan: RoutingPlan) -> list[str]:
    return [conflict.code for conflict in plan.conflicts]


# ============================================================================
# FINDING 1 — the storyboard previz route
# ============================================================================


def test_storyboard_signal_adds_a_recommended_previz_route() -> None:
    """The reproduction: a storyboard brief now offers generate_storyboard as
    the cheap first step, ahead of the per-second clip render."""
    plan = plan_generation(
        "I want to storyboard a 6 shot sequence for a coffee commercial "
        "before committing to renders",
        RoutingConstraints(num_beats=6),
    )
    assert plan.media_kind == "video"

    board = _storyboard_route(plan)
    assert board is not None, "expected a generate_storyboard route"
    # The whole point: it is discoverable AND the recommended first step.
    assert plan.recommended is board
    assert len(board.params["shots"]) == 6

    clip = next(r for r in plan.routes if r.tool == "generate_clip")
    assert board.cost is not None and clip.cost is not None
    # An order-of-magnitude cheaper preview, as the finding claims.
    assert board.cost.usd < clip.cost.usd
    assert "generate_clip" in board.rationale


def test_no_storyboard_signal_leaves_the_video_plan_unchanged() -> None:
    """A plain multi-shot brief with no previz vocabulary gets no storyboard
    route — the ~200 existing planner cases must be undisturbed."""
    plan = plan_generation("a 3 shot reel about shoes with music")
    assert _storyboard_route(plan) is None
    recommended = plan.recommended
    assert recommended is not None
    assert recommended.tool == "generate_clip"


def test_storyboard_route_leads_the_workflow_over_the_animatic() -> None:
    """When the storyboard previz out-ranks the delivery clip it leads the
    workflow, replacing the animatic: recommending a top storyboard route AND an
    animatic-first workflow was two contradictory previz steps in one answer."""
    plan = plan_generation(
        "storyboard a 4 shot commercial", RoutingConstraints(num_beats=4)
    )
    board = _storyboard_route(plan)
    assert board is not None and board is plan.recommended
    assert [(w.order, w.tool) for w in plan.workflow] == [
        (1, "generate_storyboard"),
        (2, "generate_clip"),
    ]
    # The previz step is the storyboard, not an animatic clip render.
    step1, step2 = plan.workflow
    assert "animatic" not in step1.params
    assert step1.params["shots"] == board.params["shots"]
    # The delivery clip is the top VIDEO render (not the storyboard).
    clip = next(r for r in plan.routes if r.tool == "generate_clip")
    assert step2.params == clip.params
    assert step2.params.get("animatic") is not True


@pytest.mark.parametrize(
    ("budget", "expected_model"),
    [
        # A throwaway storyboard defaults to cheap -> the cheapest renderer.
        ("cheap", LITE_IMAGE),
        # A balanced board -> the flash default, exactly as the tool defaults.
        ("balanced", FLASH),
    ],
)
def test_storyboard_route_cost_equals_estimate_image_cost(
    budget: str, expected_model: str
) -> None:
    """Planner<->tool cost parity: the route quote is estimate_image_cost for
    (model, 1K, shot count) — the exact path generate_storyboard's dry_run
    prices through (src/__main__.py: _image_cost(resolved, size, n=shots))."""
    plan = plan_generation(
        "storyboard a 6 shot commercial",
        RoutingConstraints(num_beats=6, budget=budget),  # type: ignore[arg-type]
    )
    board = _storyboard_route(plan)
    assert board is not None
    assert board.model == expected_model
    assert board.cost is not None

    expected = estimate_image_cost(expected_model, "1K", 6)
    assert expected is not None
    # Exact equality, not approx: both go through the same arithmetic.
    assert board.cost.usd == expected.usd


@pytest.mark.parametrize("budget", ["cheap", "balanced"])
@pytest.mark.asyncio
async def test_storyboard_route_cost_equals_the_tools_dry_run(
    tmp_path: Any, budget: str
) -> None:
    """End-to-end parity: the route's quote equals what generate_storyboard's
    own dry_run reports for the same shots, model and size."""
    from src.__main__ import generate_storyboard

    plan = plan_generation(
        "storyboard a 6 shot commercial",
        RoutingConstraints(num_beats=6, budget=budget),  # type: ignore[arg-type]
    )
    board = _storyboard_route(plan)
    assert board is not None and board.cost is not None

    ctx = MagicMock()
    ctx.info = AsyncMock()
    ctx.error = AsyncMock()
    ctx.warning = AsyncMock()

    import json

    blocks = await generate_storyboard(
        ctx=ctx,
        shots=board.params["shots"],
        model=board.model,
        image_size=board.params["image_size"],
        dry_run=True,
    )
    payload = json.loads(blocks[0].text)
    assert payload["dry_run"] is True
    assert payload["estimated_cost"]["usd"] == pytest.approx(round(board.cost.usd, 6))


def test_storyboard_with_on_panel_text_prefers_a_stronger_renderer() -> None:
    """A board whose panels carry slug lines/captions must not previz on the
    weakest text renderer — this is where FINDING 1 and FINDING 2 meet."""
    plan = plan_generation(
        "storyboard 6 panels with slug lines and captions",
        RoutingConstraints(num_beats=6),
    )
    board = _storyboard_route(plan)
    assert board is not None
    assert board.model != LITE_IMAGE


# ============================================================================
# FINDING 2 — caption/panel/slug-line vocabulary drives text rendering
# ============================================================================


@pytest.mark.parametrize(
    "intent",
    [
        "add slug lines to each panel",
        "a single slug line at the bottom",
        "sluglines on every board",
        "captions on every panel",
        "one caption per shot",
        "labels on the diagram",
        "a lower-third for the speaker",
        "a lower third strap",
        "panels with legible text notes",
    ],
)
def test_caption_and_panel_vocab_infer_text_rendering(intent: str) -> None:
    assert infer_signals(intent).wants_text_rendering is True


@pytest.mark.parametrize(
    "intent",
    [
        # None of the specific words -> must stay off; the fix must not
        # over-trigger on unrelated prompts.
        "a serene mountain lake at dawn",
        "a running dog in a field",
        "signature style lighting",
        # "panel" is a false friend: solar/control/wood panels carry no text,
        # so the bare word must not request a text renderer (a storyboard's
        # panels are covered by the storyboard signal + caption/slug/label).
        "a solar panel on a roof",
        "a control panel in a cockpit",
    ],
)
def test_text_rendering_vocab_does_not_over_fire(intent: str) -> None:
    assert infer_signals(intent).wants_text_rendering is False


# ============================================================================
# FINDING 3 — a pinned Veo -preview id is normalized to its canonical name
# ============================================================================


@pytest.mark.parametrize(
    ("preview_id", "canonical_id"),
    [
        ("veo-3.1-fast-generate-preview", VEO_FAST),
        ("veo-3.1-generate-preview", VEO),
    ],
)
def test_pinned_preview_veo_id_is_planned_as_its_canonical_model(
    preview_id: str, canonical_id: str
) -> None:
    """A returned Gemini-API `-preview` id is the same model as its `-001`
    name, so pinning it must plan that model, not error."""
    plan = plan_generation(
        "a video of a dog", RoutingConstraints(pinned_model=preview_id)
    )
    assert "pinned_model_not_routable" not in _conflict_codes(plan)
    recommended = plan.recommended
    assert recommended is not None
    assert recommended.model == canonical_id
    # The pin is a requirement: it is the only planned model.
    assert {route.model for route in plan.routes} == {canonical_id}


def test_pinned_unknown_video_model_is_still_a_conflict() -> None:
    """Normalization is scoped to the two known preview spellings; a genuinely
    unknown id still conflicts (control against over-broad matching)."""
    plan = plan_generation(
        "a video of a dog", RoutingConstraints(pinned_model="veo-9-imaginary")
    )
    assert "pinned_model_not_routable" in _conflict_codes(plan)


# ============================================================================
# FOLLOW-UP FINDING A — the workflow leads with the storyboard, not an animatic
# ============================================================================


def test_storyboard_previz_leads_the_workflow_when_it_out_ranks_the_clip() -> None:
    """Reproduction: a board that out-ranks the clip is the plan's cheap first
    pass, so the workflow leads with generate_storyboard, then the delivery
    clip — not the animatic the pre-fix code emitted while the board was the
    recommended route. The two previz recommendations now agree."""
    plan = plan_generation(
        "storyboard a 6 shot sequence before committing to renders",
        RoutingConstraints(num_beats=6),
    )
    board = _storyboard_route(plan)
    assert board is not None and board is plan.recommended

    step1, step2 = plan.workflow
    assert step1.tool == "generate_storyboard"
    assert "animatic" not in step1.params
    assert step1.params["shots"] == board.params["shots"]

    clip = next(r for r in plan.routes if r.tool == "generate_clip")
    assert step2.tool == "generate_clip"
    assert step2.params == clip.params
    assert step2.params.get("animatic") is not True
    # The board is genuinely the cheaper first step it is now recommended as.
    assert board.cost is not None and clip.cost is not None
    assert board.cost.usd < clip.cost.usd


def test_non_storyboard_clip_keeps_the_animatic_workflow() -> None:
    """Guard: the storyboard-first workflow must not disturb plans with no
    board. A plain multi-beat clip still gets the animatic-first workflow."""
    plan = plan_generation("a 4 shot reel about shoes")
    assert _storyboard_route(plan) is None
    assert [(w.order, w.tool) for w in plan.workflow] == [
        (1, "generate_clip"),
        (2, "generate_clip"),
    ]
    assert plan.workflow[0].params["animatic"] is True


# ============================================================================
# FOLLOW-UP FINDING B — the storyboard route is reachable for any media_kind
# ============================================================================

_CONTACT_SHEET = "a storyboard contact sheet with 4 keyframe panels and slug lines"


@pytest.mark.parametrize("media_kind", ["image", "video", None])
def test_board_deliverable_offers_the_storyboard_for_any_media_kind(
    media_kind: str | None,
) -> None:
    """A contact-sheet deliverable reaches generate_storyboard whether the
    request resolves to image, video or is left unset — its output IS images,
    so the board must not be hidden behind media_kind=video."""
    constraints = (
        RoutingConstraints(media_kind=media_kind)  # type: ignore[arg-type]
        if media_kind is not None
        else None
    )
    plan = plan_generation(_CONTACT_SHEET, constraints)
    board = _storyboard_route(plan)
    assert board is not None, f"expected a storyboard route (media_kind={media_kind})"
    # A request that names a contact sheet wants the board, not the renders.
    assert board is plan.recommended
    assert len(board.params["shots"]) == 4


def test_board_deliverable_out_ranks_the_video_renders() -> None:
    """Reproduction: the board previz sank below three Veo renders once slug
    lines forced the pricier text model. It must now out-score the clip."""
    plan = plan_generation(_CONTACT_SHEET, RoutingConstraints(media_kind="video"))
    board = _storyboard_route(plan)
    assert board is not None and board is plan.recommended
    clip = next(r for r in plan.routes if r.tool == "generate_clip")
    assert board.score > clip.score


def test_plain_image_request_gets_no_storyboard_route() -> None:
    """Guard: an image brief with no board vocabulary is untouched — only
    generate_image routes, and no storyboard route sneaks in."""
    plan = plan_generation(
        "a high-res poster of a mountain", RoutingConstraints(media_kind="image")
    )
    assert _storyboard_route(plan) is None
    assert {route.tool for route in plan.routes} == {"generate_image"}


def test_plain_video_request_is_unchanged_by_the_reachability_fix() -> None:
    """Guard: a plain multi-shot video brief still gets no storyboard route and
    still recommends the clip."""
    plan = plan_generation("a 3 shot reel about shoes with music")
    assert _storyboard_route(plan) is None
    assert plan.recommended is not None
    assert plan.recommended.tool == "generate_clip"


def test_board_deliverable_cost_matches_estimate_for_the_text_model() -> None:
    """Planner<->tool parity on the path FINDING B makes hot: a slug-line board
    picks the strong text renderer, and the route quote must still equal
    estimate_image_cost(model, 1K, shots) — generate_storyboard's dry_run path."""
    plan = plan_generation(_CONTACT_SHEET, RoutingConstraints(media_kind="image"))
    board = _storyboard_route(plan)
    assert board is not None
    # Slug lines demand the strongest text renderer even though it is priciest.
    assert board.model == PRO_IMAGE
    assert board.params["image_size"] == "1K"
    expected = estimate_image_cost(PRO_IMAGE, "1K", 4)
    assert expected is not None and board.cost is not None
    assert board.cost.usd == expected.usd


# ============================================================================
# FOLLOW-UP FINDING C — panel/keyframe/frame counts drive the shot count
# ============================================================================


@pytest.mark.parametrize(
    ("intent", "expected"),
    [
        ("a storyboard with 4 keyframe panels", 4),
        ("6 panels", 6),
        ("8 keyframes", 8),
        ("a 5 frame animation", 5),
        # The vocabulary it already parsed must keep working.
        ("6 shots of the product", 6),
        ("a 3 beat reel", 3),
    ],
)
def test_panel_and_keyframe_counts_infer_shot_count(intent: str, expected: int) -> None:
    assert infer_signals(intent).beat_count == expected


@pytest.mark.parametrize(
    "intent",
    [
        # "4k" is a resolution; the digit is not a panel count.
        "a 4k panel on a wall",
        # A frame rate, not a shot count.
        "render at 24 frames per second",
        # Nothing countable -> falls back to the default at plan time.
        "a storyboard of a chase scene",
    ],
)
def test_shot_count_vocab_does_not_over_fire(intent: str) -> None:
    assert infer_signals(intent).beat_count is None


def test_panel_count_flows_through_to_the_board_shot_count() -> None:
    """End to end: '4 keyframe panels' plans a 4-shot board, not the 3-shot
    default it silently fell back to before the vocabulary was parsed."""
    plan = plan_generation(
        "a storyboard contact sheet with 4 keyframe panels",
        RoutingConstraints(media_kind="image"),
    )
    board = _storyboard_route(plan)
    assert board is not None
    assert len(board.params["shots"]) == 4


def test_storyboard_workflow_previews_in_the_delivery_aspect_ratio() -> None:
    """The previz must be reviewed in the aspect the clip will render.

    generate_clip defaults to 9:16 while the storyboard defaults to 16:9, so
    the chained workflow reviewed a landscape board then delivered a vertical
    clip — reviewing the wrong frame defeats the previz. Both steps must carry
    one aspect ratio.
    """
    plan = plan_generation(
        "storyboard a 4 shot commercial before renders",
        RoutingConstraints(num_beats=4),
    )
    aspects = {w.params.get("aspect_ratio") for w in plan.workflow}
    assert len(plan.workflow) == 2
    assert len(aspects) == 1, f"workflow steps disagree on aspect: {aspects}"

    # And an explicit request is carried through both steps unchanged.
    vertical = plan_generation(
        "storyboard a 4 shot vertical tiktok ad before renders",
        RoutingConstraints(num_beats=4),
    )
    assert {w.params.get("aspect_ratio") for w in vertical.workflow} == {"9:16"}
