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


def test_storyboard_route_leaves_the_animatic_workflow_in_place() -> None:
    """The storyboard sits alongside the animatic, it does not replace it: the
    animatic workflow is keyed off the top VIDEO render, not routes[0]."""
    plan = plan_generation(
        "storyboard a 4 shot commercial", RoutingConstraints(num_beats=4)
    )
    assert _storyboard_route(plan) is plan.recommended
    assert plan.workflow, "the animatic workflow should still be recommended"
    assert plan.workflow[0].params.get("animatic") is True


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
