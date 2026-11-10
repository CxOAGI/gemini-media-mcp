"""Tests for routing.py — the intent router.

The router is pure, deterministic, rule-based logic, so everything here is a
plain call: no fixtures for clients, no network, no filesystem.
"""

import sys
from types import ModuleType
from typing import Any, Literal, cast

import pytest

from src.image import _IMAGE_SIZE_SUPPORT, ImageModel, ImageSize
from src.omni import (
    DEFAULT_OMNI_MODEL,
    OMNI_1_1_MODEL,
    OMNI_1_2_MODEL,
    OMNI_MODEL,
    OMNI_MODELS,
    is_omni_model,
)
from src.routing import (
    _VIDEO_CAPABILITY_RULES,
    _VIDEO_CAPABILITIES,
    _VIDEO_PROFILES,
    DEFAULT_IMAGE_MODEL,
    DEFAULT_VIDEO_MODEL,
    LIVE_IMAGE_MODELS,
    LIVE_VIDEO_MODELS,
    VEO_EXTENSION_SECONDS,
    VEO_MAX_EXTENDED_SECONDS,
    VEO_MAX_EXTENSIONS,
    WEIGHTS,
    IntentSignals,
    RoutingConstraints,
    RoutingPlan,
    infer_signals,
    plan_generation,
    resolve_request,
)

FLASH = "gemini-3.1-flash-image"
LITE_IMAGE = "gemini-3.1-flash-lite-image"
PRO_IMAGE = "gemini-3-pro-image"
VEO = "veo-3.1-generate-001"
VEO_FAST = "veo-3.1-fast-generate-001"

# The smallest score gap that counts as "capability genuinely outranked
# budget" rather than "the tie-break happened to pick the right model".
CAPABILITY_MARGIN = 0.05


def _models(plan: RoutingPlan) -> list[str]:
    """Model IDs of a plan's routes, in ranked order."""
    return [route.model for route in plan.routes]


def _rejected_models(plan: RoutingPlan) -> list[str]:
    """Model IDs of a plan's rejected options."""
    return [rejected.model for rejected in plan.rejected]


def _reason_for(plan: RoutingPlan, model: str) -> str:
    """The rejection reason recorded for ``model`` (fails if it was not rejected)."""
    for rejected in plan.rejected:
        if rejected.model == model:
            return rejected.reason
    raise AssertionError(f"{model} was not rejected; rejected={_rejected_models(plan)}")


def _conflict_codes(plan: RoutingPlan) -> list[str]:
    """Codes of every conflict on a plan."""
    return [conflict.code for conflict in plan.conflicts]


def _leads_by_a_margin(plan: RoutingPlan, winner: str, runner_up: str) -> bool:
    """Whether ``winner`` out-scores ``runner_up`` by at least the margin.

    Compared with a float tolerance because a weight table that produces
    exactly 0.05 lands on 0.049999999999999996 once the terms are summed.
    """
    scores = {route.model: route.score for route in plan.routes}
    return scores[winner] - scores[runner_up] >= CAPABILITY_MARGIN - 1e-9


# ============================================================================
# Catalog / table consistency
# ============================================================================


def test_image_profiles_cover_the_live_catalog_exactly() -> None:
    """Every live image model is routable, and nothing retired is."""
    from src.routing import _IMAGE_PROFILES

    assert set(_IMAGE_PROFILES) == set(LIVE_IMAGE_MODELS)
    assert set(LIVE_IMAGE_MODELS) == set(ImageModel.__args__)


def test_video_profiles_cover_the_live_catalog_plus_omni() -> None:
    """Veo models plus both omni models are routable; capabilities exist."""
    assert set(_VIDEO_PROFILES) == set(LIVE_VIDEO_MODELS) | set(OMNI_MODELS)
    assert set(_VIDEO_CAPABILITIES) == set(_VIDEO_PROFILES)


def test_weights_sum_to_one_so_scores_read_as_confidence() -> None:
    total = (
        WEIGHTS.capability_fit
        + WEIGHTS.budget_alignment
        + WEIGHTS.quality_ceiling
        + WEIGHTS.speed_fit
        + WEIGHTS.default_affinity
    )
    assert total == pytest.approx(1.0)


def test_image_size_support_is_imported_not_copied() -> None:
    """The router must share image.py's table, not keep a copy that can drift."""
    import src.routing

    assert src.routing._IMAGE_SIZE_SUPPORT is _IMAGE_SIZE_SUPPORT
    assert _IMAGE_SIZE_SUPPORT[LITE_IMAGE] == frozenset({"1K"})


# ============================================================================
# Intent inference
# ============================================================================


@pytest.mark.parametrize(
    ("intent", "expected"),
    [
        ("a 15 second video of a cat", "video"),
        ("an animation of a rocket", "video"),
        ("a tiktok about coffee", "video"),
        ("b-roll of a city at night", "video"),
        ("a logo for a bakery", "image"),
        ("a photo of a mountain", "image"),
        ("a poster in a cinematic style", "image"),
        # Motion words win over incidental image words.
        ("a video with a poster in the background", "video"),
        # Time-shaped vocabulary is weaker evidence but still decisive.
        ("a smooth transition between the frames", "video"),
        ("make it longer", "video"),
        # Nothing matched at all.
        ("a dog running through a field", None),
    ],
)
def test_media_kind_inference(intent: str, expected: str | None) -> None:
    assert infer_signals(intent).media_kind == expected


@pytest.mark.parametrize(
    ("intent", "attribute"),
    [
        ("a poster with a headline", "wants_text_rendering"),
        ("a sign that reads OPEN", "wants_text_rendering"),
        ("wordmark for a startup", "wants_text_rendering"),
        ("a 4k render", "wants_high_resolution"),
        ("print ready artwork", "wants_high_resolution"),
        ("a quick draft", "wants_draft"),
        ("a rough storyboard", "wants_draft"),
        ("make it warmer", "wants_iteration"),
        ("tweak the colours", "wants_iteration"),
        ("a reel about shoes", "wants_multi_shot"),
        ("a 3 shot sequence", "wants_multi_shot"),
        ("extend the clip", "wants_extension"),
        ("a seamless loop", "wants_extension"),
        ("with background music", "wants_audio"),
        ("add a voiceover", "wants_audio"),
        ("the cheapest option", "wants_cheap"),
        ("hero shot, production ready", "wants_best"),
        ("a crossfade", "wants_transition"),
        ("splice the two clips", "wants_bridge"),
        ("keep the same character", "wants_reference_consistency"),
        ("reproducible output", "wants_seed"),
        ("avoid text in the frame", "wants_negative_prompt"),
        ("write it to a gcs bucket", "wants_gcs_output"),
    ],
)
def test_boolean_signals_fire(intent: str, attribute: str) -> None:
    assert getattr(infer_signals(intent), attribute) is True


@pytest.mark.parametrize(
    ("intent", "attribute"),
    [
        # No stemming: a keyword must appear as a whole word.
        ("there is a shortage of coffee", "wants_multi_shot"),
        ("a drafting table", "wants_draft"),
        ("signature style", "wants_text_rendering"),
        # Audio words alone must not make something a video.
        ("a photo of a music festival", "wants_multi_shot"),
    ],
)
def test_signals_do_not_over_fire(intent: str, attribute: str) -> None:
    assert getattr(infer_signals(intent), attribute) is False


@pytest.mark.parametrize(
    ("intent", "expected"),
    [
        ("a vertical clip", "9:16"),
        ("shoot it 9:16", "9:16"),
        ("a tiktok", "9:16"),
        ("widescreen footage", "16:9"),
        ("16:9 please", "16:9"),
        ("a square image", "1:1"),
        ("no framing hints here", None),
    ],
)
def test_aspect_ratio_inference(intent: str, expected: str | None) -> None:
    assert infer_signals(intent).aspect_ratio == expected


@pytest.mark.parametrize(
    ("intent", "expected"),
    [
        ("an 8 second clip", 8.0),
        ("an 8-second clip", 8.0),
        ("8s clip", 8.0),
        ("30 seconds of footage", 30.0),
        ("a 2 minute video", 120.0),
        ("a clip", None),
    ],
)
def test_duration_inference(intent: str, expected: float | None) -> None:
    assert infer_signals(intent).duration_seconds == expected


@pytest.mark.parametrize(
    ("intent", "expected"),
    [
        ("a 3 beat reel", 3),
        ("5 shots of the product", 5),
        ("4 scenes", 4),
        ("a single clip", None),
    ],
)
def test_beat_count_inference(intent: str, expected: int | None) -> None:
    assert infer_signals(intent).beat_count == expected


def test_reference_count_inference() -> None:
    signals = infer_signals("use 6 reference images of the same person")
    assert signals.reference_image_count == 6
    assert signals.wants_reference_consistency is True


def test_matched_terms_are_sorted_and_deduplicated() -> None:
    signals = infer_signals("a cheap cheap poster with a logo and text")
    assert list(signals.matched_terms) == sorted(set(signals.matched_terms))
    assert "cheap" in signals.matched_terms
    assert "logo" in signals.matched_terms


def test_inference_is_case_insensitive() -> None:
    assert infer_signals("A 4K POSTER").wants_high_resolution is True


# ============================================================================
# Constraints override inference
# ============================================================================


@pytest.mark.parametrize(
    ("intent", "constraints", "attribute", "expected"),
    [
        # Explicit False beats a keyword that fired.
        (
            "a poster with a headline",
            RoutingConstraints(needs_text_rendering=False),
            "needs_text_rendering",
            False,
        ),
        (
            "a quick rough draft",
            RoutingConstraints(is_draft=False),
            "is_draft",
            False,
        ),
        # Explicit True beats silence.
        (
            "a picture of a cat",
            RoutingConstraints(needs_text_rendering=True),
            "needs_text_rendering",
            True,
        ),
        # Explicit budget beats the cheap/best keywords.
        (
            "the cheapest possible poster",
            RoutingConstraints(budget="best"),
            "budget",
            "best",
        ),
        # Explicit media kind beats the vocabulary.
        (
            "a tiktok about coffee",
            RoutingConstraints(media_kind="image"),
            "media_kind",
            "image",
        ),
        # Explicit sizes/ratios beat inferred ones.
        (
            "a 4k poster",
            RoutingConstraints(image_size="1K"),
            "image_size",
            "1K",
        ),
        (
            "a vertical clip",
            RoutingConstraints(aspect_ratio="16:9"),
            "aspect_ratio",
            "16:9",
        ),
        (
            "a 3 beat reel",
            RoutingConstraints(num_beats=7),
            "num_beats",
            7,
        ),
        (
            "an 8 second clip",
            RoutingConstraints(duration_seconds=4),
            "total_duration_seconds",
            4.0,
        ),
        (
            "extend the clip",
            RoutingConstraints(needs_extension=False),
            "needs_extension",
            False,
        ),
    ],
)
def test_explicit_constraints_override_inference(
    intent: str, constraints: RoutingConstraints, attribute: str, expected: Any
) -> None:
    request = resolve_request(intent, infer_signals(intent), constraints)
    assert getattr(request, attribute) == expected


def test_inference_applies_when_constraints_are_silent() -> None:
    request = resolve_request(
        "a poster with a headline",
        infer_signals("a poster with a headline"),
        RoutingConstraints(),
    )
    assert request.needs_text_rendering is True


@pytest.mark.parametrize(
    ("intent", "expected"),
    [
        ("a quick draft of a cat", "cheap"),
        ("the cheapest poster", "cheap"),
        ("a hero image, production ready", "best"),
        ("a picture of a cat", "balanced"),
    ],
)
def test_budget_defaults(intent: str, expected: str) -> None:
    """A throwaway render should not default to hero pricing."""
    request = resolve_request(intent, infer_signals(intent), None)
    assert request.budget == expected


@pytest.mark.parametrize(
    ("intent", "constraints", "expected"),
    [
        ("a 4k poster", None, "4K"),
        ("a print ready poster", None, "2K"),
        ("a picture of a cat", None, "1K"),
        ("a picture of a cat", RoutingConstraints(needs_4k=True), "4K"),
        ("a picture of a cat", RoutingConstraints(image_size="2K"), "2K"),
    ],
)
def test_image_size_resolution(
    intent: str, constraints: RoutingConstraints | None, expected: str
) -> None:
    request = resolve_request(intent, infer_signals(intent), constraints)
    assert request.image_size == expected


def test_video_only_constraints_imply_video_media_kind() -> None:
    """Frames to interpolate between can only mean a video request."""
    plan = plan_generation(
        "something smooth", RoutingConstraints(first_frame_uri="a", last_frame_uri="b")
    )
    assert plan.media_kind == "video"


def test_ambiguous_intent_defaults_to_image_with_a_note() -> None:
    plan = plan_generation("a dog running through a field")
    assert plan.media_kind == "image"
    assert any("planned as an image" in note for note in plan.notes)


# ============================================================================
# Image planning
# ============================================================================


@pytest.mark.parametrize(
    ("intent", "constraints", "expected_top"),
    [
        ("a picture of a cat", None, DEFAULT_IMAGE_MODEL),
        ("a picture of a cat", RoutingConstraints(budget="cheap"), LITE_IMAGE),
        ("a picture of a cat", RoutingConstraints(budget="best"), PRO_IMAGE),
        # Legible text is the single strongest reason to pay for pro.
        ("a poster with a bold headline", None, PRO_IMAGE),
        # A throwaway render goes to the cheap, fast tier.
        ("a quick rough sketch of a cat", None, LITE_IMAGE),
        # 4K removes lite from the race entirely.
        ("a 4k picture of a cat", None, FLASH),
    ],
)
def test_image_ranking(
    intent: str, constraints: RoutingConstraints | None, expected_top: str
) -> None:
    plan = plan_generation(intent, constraints)
    recommended = plan.recommended
    assert recommended is not None
    assert recommended.model == expected_top
    assert recommended.tool == "generate_image"


def test_image_routes_are_ranked_by_descending_score() -> None:
    plan = plan_generation("a picture of a cat")
    scores = [route.score for route in plan.routes]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize("image_size", ["2K", "4K"])
def test_flash_lite_rejected_above_1k_with_a_reason(image_size: str) -> None:
    plan = plan_generation(
        "a picture of a cat",
        RoutingConstraints(image_size=cast(ImageSize, image_size)),
    )
    assert LITE_IMAGE not in _models(plan)
    reason = _reason_for(plan, LITE_IMAGE)
    assert f"cannot produce {image_size}" in reason
    assert "1K" in reason


def test_flash_lite_survives_at_1k() -> None:
    plan = plan_generation("a picture of a cat", RoutingConstraints(image_size="1K"))
    assert LITE_IMAGE in _models(plan)
    assert not plan.rejected


def test_pinned_flash_lite_at_4k_is_a_conflict() -> None:
    plan = plan_generation(
        "a picture of a cat",
        RoutingConstraints(image_size="4K", pinned_model=LITE_IMAGE),
    )
    assert "image_size_unsupported_by_pinned_model" in _conflict_codes(plan)
    # A conflict is not a dead end: the workable alternatives are still ranked.
    assert plan.is_satisfiable


def test_pinned_retired_image_model_plans_against_its_replacement() -> None:
    """A superseded pin is not unroutable: generate_image reroutes it, so the
    plan must describe the model that would actually run."""
    plan = plan_generation(
        "a picture of a cat", RoutingConstraints(pinned_model="imagen-4.0-generate-001")
    )
    assert "pinned_model_not_routable" not in _conflict_codes(plan)
    assert [route.model for route in plan.routes] == ["gemini-3.1-flash-image"]
    assert any("superseded" in note for note in plan.notes)


def test_pinned_unknown_image_model_is_still_a_conflict() -> None:
    """An ID that resolves to nothing real remains unroutable."""
    plan = plan_generation(
        "a picture of a cat", RoutingConstraints(pinned_model="not-a-real-model")
    )
    assert "pinned_model_not_routable" in _conflict_codes(plan)


def test_a_surviving_pin_is_the_only_route() -> None:
    """pinned_model is a requirement, not a preference — planning an
    alternative would answer a question the caller did not ask."""
    plan = plan_generation(
        "a picture of a cat", RoutingConstraints(pinned_model="gemini-3-pro-image")
    )
    assert [route.model for route in plan.routes] == ["gemini-3-pro-image"]
    # The alternatives are still listed, with the pin named as the reason.
    reasons = {r.model: r.reason for r in plan.rejected}
    assert "gemini-3.1-flash-image" in reasons
    assert "pinned_model=gemini-3-pro-image" in reasons["gemini-3.1-flash-image"]


def test_a_pin_that_trips_a_rule_still_offers_alternatives() -> None:
    """When the pin cannot work, the conflict explains why and the ranked
    alternatives remain — a dead end would be less useful than a fix."""
    plan = plan_generation(
        "a 4k image",
        RoutingConstraints(pinned_model="gemini-3.1-flash-lite-image", needs_4k=True),
    )
    assert "image_size_unsupported_by_pinned_model" in _conflict_codes(plan)
    assert [route.model for route in plan.routes] == [
        "gemini-3.1-flash-image",
        "gemini-3-pro-image",
    ]


def test_a_surviving_video_pin_is_the_only_route() -> None:
    plan = plan_generation(
        "a video of a cat",
        RoutingConstraints(pinned_model="veo-3.1-generate-001"),
    )
    assert [route.model for route in plan.routes] == ["veo-3.1-generate-001"]


def test_vertex_backend_surfaces_the_global_location_requirement() -> None:
    plan = plan_generation("a picture of a cat", RoutingConstraints(backend="vertex"))
    recommended = plan.recommended
    assert recommended is not None
    assert any("global" in caveat for caveat in recommended.caveats)


def test_text_rendering_warns_on_every_route_that_is_not_pro() -> None:
    """A non-pro route for a text brief must say what it is trading away."""
    plan = plan_generation("a poster with a headline")
    for route in plan.routes:
        has_warning = any("gemini-3-pro-image" in caveat for caveat in route.caveats)
        assert has_warning is (route.model != PRO_IMAGE)


def test_capability_outranks_budget_for_a_text_brief() -> None:
    """'cheap' does not buy a model that cannot render the words legibly.

    Strengthened from "pro ranks first" to "pro ranks first by a margin":
    ordering alone was satisfied even when the scores tied, in which case the
    documented guarantee was really being supplied by the fidelity tie-break
    in ``_rank``.
    """
    plan = plan_generation(
        "a poster with a headline", RoutingConstraints(budget="cheap", image_size="2K")
    )
    recommended = plan.recommended
    assert recommended is not None
    assert recommended.model == PRO_IMAGE
    assert _leads_by_a_margin(plan, PRO_IMAGE, FLASH)


def test_image_params_are_ready_to_use() -> None:
    plan = plan_generation(
        "a 4k picture of a cat", RoutingConstraints(aspect_ratio="16:9")
    )
    recommended = plan.recommended
    assert recommended is not None
    assert recommended.params == {
        "prompt": "a 4k picture of a cat",
        "model": FLASH,
        "image_size": "4K",
        "aspect_ratio": "16:9",
    }


def test_image_routes_carry_cost_estimates() -> None:
    plan = plan_generation("a picture of a cat")
    for route in plan.routes:
        assert route.cost is not None
        assert route.cost.is_estimate is True
        assert route.cost.usd > 0


# ============================================================================
# Video tool selection
# ============================================================================


@pytest.mark.parametrize(
    ("intent", "constraints", "expected_tool"),
    [
        (
            "make it stormier",
            RoutingConstraints(previous_interaction_id="int-1"),
            "edit_video",
        ),
        (
            "keep the action going",
            RoutingConstraints(media_kind="video", needs_extension=True),
            "loop_extend",
        ),
        (
            "a smooth transition",
            RoutingConstraints(first_frame_uri="a", last_frame_uri="b"),
            "generate_transition",
        ),
        (
            "splice the two clips together",
            RoutingConstraints(has_first_frame=True, has_last_frame=True),
            "generate_bridge",
        ),
        (
            "a 3 shot reel about shoes",
            None,
            "generate_clip",
        ),
        (
            "a video of a cat",
            None,
            "generate_video",
        ),
    ],
)
def test_video_tool_ladder(
    intent: str, constraints: RoutingConstraints | None, expected_tool: str
) -> None:
    plan = plan_generation(intent, constraints)
    recommended = plan.recommended
    assert recommended is not None
    assert recommended.tool == expected_tool


def test_omni_is_reached_through_its_own_tool() -> None:
    plan = plan_generation("a video of a cat", RoutingConstraints(budget="cheap"))
    omni_routes = [route for route in plan.routes if route.model == OMNI_MODEL]
    assert omni_routes
    assert omni_routes[0].tool == "generate_video_omni"


@pytest.mark.parametrize(
    ("intent", "constraints", "expected_top"),
    [
        ("a video of a cat", None, DEFAULT_VIDEO_MODEL),
        ("a video of a cat", RoutingConstraints(budget="best"), VEO),
        # Omni 1.2 is now the cheapest 720p renderer at $0.069504/s.
        ("a video of a cat", RoutingConstraints(budget="cheap"), DEFAULT_OMNI_MODEL),
    ],
)
def test_video_ranking(
    intent: str, constraints: RoutingConstraints | None, expected_top: str
) -> None:
    plan = plan_generation(intent, constraints)
    recommended = plan.recommended
    assert recommended is not None
    assert recommended.model == expected_top


# ============================================================================
# Impossibility rules
# ============================================================================

# One case per hard video capability rule, so the rule table cannot grow a rule
# that nothing exercises (see test_every_capability_rule_is_covered).
_RULE_CASES: tuple[tuple[str, str, RoutingConstraints, str, str], ...] = (
    (
        "first_last_frame_unsupported",
        "the omni preview cannot interpolate between frames",
        RoutingConstraints(first_frame_uri="a", last_frame_uri="b"),
        OMNI_MODEL,
        "first+last-frame",
    ),
    (
        "extension_unsupported",
        "the omni preview cannot extend",
        RoutingConstraints(media_kind="video", needs_extension=True),
        OMNI_MODEL,
        "cannot extend",
    ),
    (
        "4k_unsupported",
        "the omni preview has no 4K",
        RoutingConstraints(media_kind="video", resolution="4K"),
        OMNI_MODEL,
        "cannot produce 4K",
    ),
    (
        "360p_unsupported",
        "veo has no 360p draft tier",
        RoutingConstraints(media_kind="video", resolution="360p"),
        "veo-3.1-fast-generate-001",
        "360p",
    ),
    (
        "1080p_unsupported",
        "omni renders 720p only",
        RoutingConstraints(media_kind="video", resolution="1080p"),
        OMNI_MODEL,
        "720p",
    ),
    (
        "seed_unsupported",
        "omni has no seed",
        RoutingConstraints(media_kind="video", needs_seed=True),
        OMNI_MODEL,
        "seed",
    ),
    (
        "negative_prompt_unsupported",
        "omni has no negative prompt",
        RoutingConstraints(media_kind="video", needs_negative_prompt=True),
        OMNI_MODEL,
        "negative_prompt",
    ),
    (
        "audio_unsupported",
        "omni previews carry no audio",
        RoutingConstraints(media_kind="video", needs_audio=True),
        OMNI_MODEL,
        "no audio track",
    ),
    (
        "conversational_edit_unsupported",
        "veo cannot conversationally edit",
        RoutingConstraints(previous_interaction_id="int-1"),
        VEO_FAST,
        "no conversational editing",
    ),
)


@pytest.mark.parametrize(
    ("code", "description", "constraints", "model", "expected_fragment"),
    _RULE_CASES,
    ids=[case[0] for case in _RULE_CASES],
)
def test_capability_rules_reject_with_an_explanation(
    code: str,
    description: str,
    constraints: RoutingConstraints,
    model: str,
    expected_fragment: str,
) -> None:
    plan = plan_generation("a video of a product", constraints)
    assert model not in _models(plan), description
    assert expected_fragment in _reason_for(plan, model)


def test_every_capability_rule_is_covered_by_a_test_case() -> None:
    """The rule table and the test table must not drift apart.

    reference_images_unsupported has no model in today's catalog to violate
    it, so it is proven on a patched capability set instead (below).
    """
    assert {case[0] for case in _RULE_CASES} | {"reference_images_unsupported"} == {
        rule.code for rule in _VIDEO_CAPABILITY_RULES
    }


def test_the_reference_images_rule_still_fires(monkeypatch: pytest.MonkeyPatch) -> None:
    """No model in the catalog lacks reference images since Veo 3.1 Lite left
    the server, but the rule must keep firing for the next model that does."""
    import dataclasses

    import src.routing as routing

    patched = dict(routing._VIDEO_CAPABILITIES)  # pyright: ignore[reportPrivateUsage]
    patched[VEO_FAST] = dataclasses.replace(
        patched[VEO_FAST], supports_reference_images=False
    )
    monkeypatch.setattr(routing, "_VIDEO_CAPABILITIES", patched)

    plan = plan_generation(
        "a video of a product",
        RoutingConstraints(media_kind="video", num_reference_images=2),
    )
    assert VEO_FAST not in _models(plan)
    assert "reference images" in _reason_for(plan, VEO_FAST)


def test_veo_is_not_offered_on_the_gemini_api() -> None:
    """Google switched off every Veo 3.1 model the Gemini Developer API served
    on 2026-10-22, so a plan there routes video to omni and says why Veo is
    out, with the fix."""
    plan = plan_generation("a video of a cat", RoutingConstraints(backend="gemini_api"))
    offered = _models(plan)
    assert VEO not in offered and VEO_FAST not in offered
    assert DEFAULT_OMNI_MODEL in offered
    for model in (VEO, VEO_FAST):
        reason = _reason_for(plan, model)
        assert "Gemini Developer API" in reason
        assert "2026-10-22" in reason


def test_veo_is_offered_on_vertex() -> None:
    plan = plan_generation("a video of a cat", RoutingConstraints(backend="vertex"))
    assert {VEO, VEO_FAST} <= set(_models(plan))


def test_a_pinned_veo_model_on_the_gemini_api_is_a_conflict() -> None:
    """A pin on a model the backend no longer serves must say why the plan has
    nothing for it, and how to get it."""
    plan = plan_generation(
        "a video of a cat",
        RoutingConstraints(backend="gemini_api", pinned_model=VEO_FAST),
    )
    conflicts = {conflict.code: conflict for conflict in plan.conflicts}
    assert "pinned_model_backend_unsupported" in conflicts
    conflict = conflicts["pinned_model_backend_unsupported"]
    assert "Gemini Developer API" in conflict.detail
    assert DEFAULT_OMNI_MODEL in conflict.resolution
    assert "Vertex AI" in conflict.resolution


def test_gcs_output_on_the_gemini_api_is_a_conflict() -> None:
    plan = plan_generation(
        "a video of a cat",
        RoutingConstraints(backend="gemini_api", wants_gcs_output=True),
    )
    assert "gcs_output_on_gemini_api" in _conflict_codes(plan)
    recommended = plan.recommended
    assert recommended is not None
    assert "output_gcs_uri" not in recommended.params
    assert any("Vertex-only" in caveat for caveat in recommended.caveats)


@pytest.mark.parametrize("duration", [1.0, 2.0])
def test_omni_rejected_below_its_minimum_duration(duration: float) -> None:
    plan = plan_generation(
        "a video of a cat",
        RoutingConstraints(duration_seconds=duration, budget="cheap"),
    )
    assert OMNI_MODEL not in _models(plan)
    assert "not supported" in _reason_for(plan, OMNI_MODEL)


def test_runtime_beyond_the_extension_ceiling_is_a_conflict() -> None:
    plan = plan_generation("a 10 minute video of a city")
    conflicts = {conflict.code: conflict for conflict in plan.conflicts}
    assert "duration_exceeds_extension_ceiling" in conflicts
    assert (
        str(VEO_MAX_EXTENDED_SECONDS)
        in conflicts["duration_exceeds_extension_ceiling"].detail
    )
    assert "generate_clip" in conflicts["duration_exceeds_extension_ceiling"].resolution


def test_transition_without_both_endpoints_is_a_conflict() -> None:
    plan = plan_generation(
        "a crossfade", RoutingConstraints(media_kind="video", has_first_frame=True)
    )
    assert "transition_requires_two_endpoints" in _conflict_codes(plan)


def test_conversational_edit_with_a_seed_requirement_is_a_conflict() -> None:
    plan = plan_generation(
        "make it stormier",
        RoutingConstraints(previous_interaction_id="int-1", needs_seed=True),
    )
    assert "conversational_edit_without_seed_support" in _conflict_codes(plan)
    # Every model is out: omni has no seed, Veo cannot edit conversationally.
    assert not plan.is_satisfiable
    assert plan.recommended is None


def test_pinned_model_violating_a_rule_becomes_a_conflict() -> None:
    plan = plan_generation(
        "a video of a cat",
        RoutingConstraints(
            media_kind="video", resolution="4K", pinned_model=OMNI_MODEL
        ),
    )
    assert "pinned_model_4k_unsupported" in _conflict_codes(plan)


# ============================================================================
# Video parameters and cost
# ============================================================================


def test_veo_duration_is_snapped_and_the_snap_is_disclosed() -> None:
    plan = plan_generation(
        "a video of a cat", RoutingConstraints(media_kind="video", duration_seconds=5)
    )
    recommended = plan.recommended
    assert recommended is not None
    assert recommended.params["duration_seconds"] == 4.0
    assert any("snaps to 4s" in caveat for caveat in recommended.caveats)


def test_clip_params_carry_one_beat_per_shot() -> None:
    plan = plan_generation("a 3 shot reel about shoes with music")
    recommended = plan.recommended
    assert recommended is not None
    assert recommended.tool == "generate_clip"
    assert len(recommended.params["beats"]) == 3
    assert recommended.params["aspect_ratio"] == "9:16"
    assert recommended.params["include_audio"] is True
    assert any("beat prompts" in caveat for caveat in recommended.caveats)


def test_loop_extend_computes_the_number_of_extensions() -> None:
    plan = plan_generation(
        "make it 30 seconds long",
        RoutingConstraints(media_kind="video", source_video_uri="file:///d/a.mp4"),
    )
    recommended = plan.recommended
    assert recommended is not None
    assert recommended.tool == "loop_extend"
    # 8s base clip + ceil(22 / 7) = 4 extensions.
    assert recommended.params["times"] == 4
    assert recommended.params["video_uri"] == "file:///d/a.mp4"
    assert any(str(VEO_EXTENSION_SECONDS) in caveat for caveat in recommended.caveats)


def test_loop_extend_caps_at_the_documented_maximum() -> None:
    plan = plan_generation("a 10 minute video of a city")
    recommended = plan.recommended
    assert recommended is not None
    assert recommended.params["times"] == VEO_MAX_EXTENSIONS


def test_missing_frame_uris_are_called_out_rather_than_invented() -> None:
    plan = plan_generation(
        "a crossfade",
        RoutingConstraints(
            media_kind="video", has_first_frame=True, has_last_frame=True
        ),
    )
    recommended = plan.recommended
    assert recommended is not None
    assert "first_frame_uri" not in recommended.params
    assert any("Add first_frame_uri" in caveat for caveat in recommended.caveats)


def test_unsupported_video_aspect_ratio_is_corrected_with_a_caveat() -> None:
    plan = plan_generation(
        "a video of a cat",
        RoutingConstraints(media_kind="video", aspect_ratio="1:1"),
    )
    recommended = plan.recommended
    assert recommended is not None
    assert recommended.params["aspect_ratio"] == "16:9"
    assert any("not supported for video" in caveat for caveat in recommended.caveats)


def test_multi_render_cost_sums_every_segment() -> None:
    """A 3-beat clip must not be quoted as a single clip. duration_seconds is
    the whole clip's runtime, so 18s across 3 beats is three 6s renders."""
    single = plan_generation(
        "a video of a cat", RoutingConstraints(media_kind="video", duration_seconds=6)
    ).recommended
    clip = plan_generation(
        "a 3 shot reel", RoutingConstraints(duration_seconds=18)
    ).recommended
    assert single is not None and clip is not None
    assert single.cost is not None and clip.cost is not None
    assert clip.cost.usd == pytest.approx(single.cost.usd * 3)
    assert clip.cost.breakdown["renders"] == 3.0


@pytest.mark.parametrize(
    "intent, constraints, beat, caveat",
    [
        # Each beat's share lands on the ladder: no caveat.
        ("Create a 12 second video of a dog", {"num_beats": 2}, 6.0, None),
        ("a 3 shot commercial, 12 seconds total", {}, 4.0, None),
        ("a 3 shot commercial, 24 seconds total", {}, 8.0, None),
        (
            "a 3 shot commercial, 20 seconds total",
            {},
            6.0,
            "Each of the 3 beats is 6.67s of the 20s asked; Veo renders "
            "4/6/8s clips, so each snaps to 6s (18s in all).",
        ),
        (
            "a 2 shot reel, 30 seconds total",
            {},
            8.0,
            "Each of the 2 beats is 15s of the 30s asked; Veo renders 4/6/8s "
            "clips, so each snaps to 8s (16s in all).",
        ),
    ],
)
def test_a_multi_beat_runtime_is_shared_across_its_beats(
    intent: str, constraints: dict[str, Any], beat: float, caveat: str | None
) -> None:
    """Every beat took the whole runtime, capped at one render: a 2-beat
    12s brief planned two 8s beats (16s) and said "12s snaps to 8s", as if
    the clip had been cut short; a 3-shot 12s brief planned 24s."""
    plan = plan_generation(intent, RoutingConstraints(backend="vertex", **constraints))
    route = plan.recommended
    assert route is not None and route.tool == "generate_clip"
    assert {b["duration_seconds"] for b in route.params["beats"]} == {beat}
    snaps = [c for c in route.caveats if "snaps to" in c]
    assert snaps == ([caveat] if caveat else [])


def test_a_storyboard_shot_is_its_beats_share_of_the_runtime() -> None:
    """The board's shot list feeds generate_clip, so its durations are the
    beats' own."""
    plan = plan_generation("storyboard a 3 shot commercial, 12 seconds total")
    board = next(r for r in plan.routes if r.tool == "generate_storyboard")
    assert {s["duration_seconds"] for s in board.params["shots"]} == {4.0}


# ============================================================================
# Workflow recommendations
# ============================================================================


def test_expensive_multi_beat_clips_recommend_an_animatic_first() -> None:
    plan = plan_generation("a 4 shot reel about shoes")
    assert len(plan.workflow) == 2
    first, second = plan.workflow
    assert first.tool == "generate_clip"
    assert first.params["animatic"] is True
    assert first.params["add_bridges"] is False
    assert second.params.get("animatic") is not True
    # The preview renders on the model with a 360p tier, which is what makes it
    # cheaper than the delivery render rather than level with it.
    assert DEFAULT_OMNI_MODEL in first.rationale
    assert "360p" in first.rationale


def test_a_draft_clip_gets_the_animatic_as_the_deliverable() -> None:
    # No storyboard vocabulary here: a board previz would out-rank the clip and
    # lead the workflow instead (see test_planner_followups), so this keeps the
    # intent a plain draft clip to exercise the animatic-as-deliverable path.
    plan = plan_generation("a rough 4 shot reel")
    assert len(plan.workflow) == 1
    assert plan.workflow[0].params["animatic"] is True


def test_single_shot_video_gets_no_workflow() -> None:
    plan = plan_generation("a video of a cat")
    assert plan.workflow == ()


def test_image_plans_get_no_workflow() -> None:
    plan = plan_generation("a picture of a cat")
    assert plan.workflow == ()


# ============================================================================
# Determinism and purity
# ============================================================================


@pytest.mark.parametrize(
    ("intent", "constraints"),
    [
        ("a picture of a cat", None),
        ("a 4k poster with a bold headline", RoutingConstraints(budget="best")),
        ("a 3 shot reel with music", RoutingConstraints(backend="vertex")),
        (
            "make it stormier",
            RoutingConstraints(previous_interaction_id="int-1"),
        ),
        ("a 10 minute video", RoutingConstraints(wants_gcs_output=True)),
    ],
)
def test_same_input_produces_an_identical_plan(
    intent: str, constraints: RoutingConstraints | None
) -> None:
    first = plan_generation(intent, constraints)
    second = plan_generation(intent, constraints)
    assert first == second
    assert [route.score for route in first.routes] == [
        route.score for route in second.routes
    ]


def test_default_constraints_match_an_explicitly_empty_one() -> None:
    assert plan_generation("a picture of a cat") == plan_generation(
        "a picture of a cat", RoutingConstraints()
    )


def test_scores_stay_within_the_unit_interval() -> None:
    for intent in ("a picture of a cat", "a 4k poster", "a 3 shot reel with music"):
        for route in plan_generation(intent).routes:
            assert 0.0 <= route.score <= 1.0


# ============================================================================
# Graceful degradation when pricing is unavailable
# ============================================================================


def test_routes_survive_a_missing_pricing_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing price book must not stop the router from answering."""
    monkeypatch.setitem(sys.modules, "src.pricing", None)
    plan = plan_generation("a 4k poster with a headline")
    assert plan.is_satisfiable
    assert all(route.cost is None for route in plan.routes)
    assert any("Cost estimates are unavailable" in note for note in plan.notes)


def test_video_routes_survive_a_missing_pricing_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "src.pricing", None)
    plan = plan_generation("a 3 shot reel with music")
    assert plan.is_satisfiable
    assert all(route.cost is None for route in plan.routes)
    # The animatic advice is beat-count driven, so it survives without pricing.
    assert plan.workflow


def test_routes_survive_a_failing_price_book(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pricing module that raises is treated exactly like a missing one."""
    broken = ModuleType("src.pricing")

    def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("price book unavailable")

    broken.__dict__["estimate_image_cost"] = _boom
    broken.__dict__["estimate_video_cost"] = _boom
    monkeypatch.setitem(sys.modules, "src.pricing", broken)

    plan = plan_generation("a picture of a cat")
    assert plan.is_satisfiable
    assert all(route.cost is None for route in plan.routes)


# ============================================================================
# Input validation
# ============================================================================


@pytest.mark.parametrize("intent", ["", "   ", "\n"])
def test_empty_intent_is_rejected(intent: str) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        plan_generation(intent)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"budget": "free"}, "budget"),
        ({"backend": "azure"}, "backend"),
        ({"media_kind": "audio"}, "media_kind"),
        ({"image_size": "8K"}, "image_size"),
        ({"resolution": "480p"}, "resolution"),
        ({"num_beats": -1}, "num_beats"),
        ({"duration_seconds": 0}, "duration_seconds"),
    ],
)
def test_invalid_constraints_are_rejected(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        RoutingConstraints(**kwargs)


def test_signals_default_to_no_opinion() -> None:
    """An all-default IntentSignals must not assert anything."""
    signals = IntentSignals()
    assert signals.media_kind is None
    assert signals.matched_terms == ()
    assert signals.wants_text_rendering is False


# ============================================================================
# The emitted parameters really are the tools' parameters
# ============================================================================


def test_emitted_params_match_the_real_tool_signatures() -> None:
    """Every key the router emits must exist on the tool it names.

    This is the guarantee that makes ``params`` pass-through: a renamed tool
    parameter has to break this test rather than a caller's request.
    """
    import inspect

    import src.__main__ as server

    plans = [
        plan_generation("a 4k poster with a headline"),
        plan_generation("a video of a cat", RoutingConstraints(backend="vertex")),
        plan_generation("a 3 shot reel with music"),
        plan_generation(
            "a crossfade", RoutingConstraints(first_frame_uri="a", last_frame_uri="b")
        ),
        plan_generation(
            "make it 30 seconds",
            RoutingConstraints(media_kind="video", source_video_uri="file:///d/a.mp4"),
        ),
        plan_generation(
            "make it stormier", RoutingConstraints(previous_interaction_id="int-1")
        ),
    ]
    calls = [(route.tool, route.params) for plan in plans for route in plan.routes] + [
        (step.tool, step.params) for plan in plans for step in plan.workflow
    ]
    assert calls

    for tool_name, params in calls:
        signature = inspect.signature(getattr(server, tool_name))
        allowed = set(signature.parameters)
        assert set(params) <= allowed, f"{tool_name} got unknown params {set(params)}"

    # Beat specs are their own mini-schema, documented on generate_clip.
    beat_keys = {
        "prompt",
        "duration_seconds",
        "seed",
        "first_frame_uri",
        "negative_prompt",
        "audio_prompt",
    }
    for tool_name, params in calls:
        for beat in params.get("beats", []):
            assert set(beat) <= beat_keys


def test_profile_accessors_cover_the_live_catalogs() -> None:
    """image_profile/video_profile/video_capabilities are the module's public
    lookup API and had no direct tests — a rename would only have surfaced
    through downstream callers."""
    from typing import get_args

    from src.image import ImageModel
    from src.routing import image_profile, video_capabilities, video_profile
    from src.video import VideoModel

    for model in get_args(ImageModel):
        assert image_profile(model) is not None, model
    for model in get_args(VideoModel):
        assert video_profile(model) is not None, model
        assert video_capabilities(model) is not None, model
    assert image_profile("not-a-model") is None
    assert video_profile("not-a-model") is None
    assert video_capabilities("not-a-model") is None


def test_an_implied_extension_plans_the_seed_render_first() -> None:
    """A fresh 30-second request implies extension (a continuous shot longer
    than one Veo render can only be made by extending), but the caller has no
    video yet — so the workflow must lead with the seed render. This used to
    return loop_extend alone with an empty workflow: a top route whose
    required video_uri could not exist."""
    plan = plan_generation("a 30 second product video")
    top = plan.recommended
    assert top is not None
    assert top.tool == "loop_extend"

    assert [(w.order, w.tool) for w in plan.workflow] == [
        (1, "generate_video"),
        (2, "loop_extend"),
    ]
    seed, extend = plan.workflow
    assert seed.params["duration_seconds"] == 8.0
    # 8s seed + 4 extensions of ~7s covers the requested 30s.
    assert extend.params["times"] == 4
    assert "step 1" in extend.params["video_uri"]


def test_an_explicit_extension_gets_no_seed_workflow() -> None:
    """A caller who asked to extend has a clip already: with the URI supplied
    the plan is a single call, and without it the caveat asks for the URI
    instead of proposing a misleading fresh render."""
    with_source = plan_generation(
        "make this longer",
        RoutingConstraints(
            needs_extension=True,
            source_video_uri="file:///v.mp4",
            duration_seconds=30,
            media_kind="video",
        ),
    )
    assert with_source.workflow == ()

    without_source = plan_generation(
        "make this longer",
        RoutingConstraints(
            needs_extension=True, duration_seconds=30, media_kind="video"
        ),
    )
    assert without_source.workflow == ()
    assert without_source.recommended is not None
    assert any("video_uri" in c for c in without_source.recommended.caveats)


# ============================================================================
# Live-test findings
# ============================================================================


@pytest.mark.parametrize(
    ("intent", "expect_audio"),
    [
        pytest.param("a short video with audio", True, id="plain_request"),
        pytest.param("a video of steam, no audio", False, id="no_audio"),
        pytest.param("a clip without dialogue", False, id="without"),
        pytest.param("silent b-roll of a city", False, id="silent"),
        pytest.param("no dialogue, but ambient music", True, id="mixed_keeps_positive"),
    ],
)
def test_negated_keywords_are_not_read_as_requests(
    intent: str, expect_audio: bool
) -> None:
    """ "no audio" was keyword-matched as a request FOR audio, which flipped
    both the emitted include_audio and the model ranking. A term counts only
    when every occurrence is un-negated, so a sentence mentioning it both ways
    keeps the positive signal."""
    assert infer_signals(intent).wants_audio is expect_audio


def test_a_brief_single_shot_request_is_not_inflated_into_a_clip() -> None:
    """ "short video" means brief, not "a short". Reading it as multi-shot
    turned one 4s render into a 3-beat clip at three times the price, with
    placeholder prompts for shots the caller never described."""
    plan = plan_generation(
        "cheapest possible short video of steam rising from a coffee cup, no audio",
        RoutingConstraints(budget="cheap", duration_seconds=4),
    )
    top = plan.recommended
    assert top is not None
    assert top.params.get("beats") is None
    # The winning route may be omni, which has no audio control at all — the
    # requirement is that nothing asks FOR audio.
    assert top.params.get("include_audio") is not True
    assert plan.request.needs_audio is False


@pytest.mark.parametrize(
    "intent",
    ["a 3 beat tiktok reel about coffee", "a montage of city shots"],
)
def test_genuine_multi_shot_intents_still_plan_a_clip(intent: str) -> None:
    """The fix must not over-reach: real multi-shot briefs still get clips."""
    top = plan_generation(intent).recommended
    assert top is not None
    assert top.tool == "generate_clip"
    assert len(top.params["beats"]) > 1


def test_a_capability_demand_outweighs_a_budget_preference_by_construction() -> None:
    """The weight table has to make the documented promise arithmetically true.

    Two models one tier apart on the cost axis differ by at most half the
    budget weight, and the cheaper one may also be the documented default and
    collect default_affinity on top; the model that owns a capability leads
    the one that merely has some of it by half the capability weight. At
    0.35/0.25/0.05 that balance was exactly zero — "capability outranks
    budget" was a tie, resolved by ``_rank``'s fidelity tie-break.
    """
    balance = (
        0.5 * WEIGHTS.capability_fit
        - 0.5 * WEIGHTS.budget_alignment
        - WEIGHTS.default_affinity
    )
    assert balance >= CAPABILITY_MARGIN - 1e-9


def test_a_text_demand_beats_a_cheap_budget_by_a_real_margin() -> None:
    """The reported reproduction: pro and flash both scored 0.525 here, so the
    documented winner came out of the tie-break and any change to iteration
    order would have flipped it silently."""
    plan = plan_generation(
        "a poster with the words GRAND OPENING", RoutingConstraints(budget="cheap")
    )
    recommended = plan.recommended
    assert recommended is not None
    assert recommended.model == PRO_IMAGE

    scores = [route.score for route in plan.routes]
    # A strict descent: no two candidates may be separated by the tie-break.
    assert scores == sorted(scores, reverse=True)
    assert len(set(scores)) == len(scores)
    assert _leads_by_a_margin(plan, PRO_IMAGE, FLASH)


def test_a_size_demand_every_survivor_meets_still_resolves_on_budget() -> None:
    """The other half of the scoring fix: 4K is a hard rule, and the models
    that pass it satisfy it equally. Scoring the survivors on fidelity as well
    double-counted quality_ceiling, so a plain 4K brief at a balanced budget
    would have drifted off the documented default onto the priciest model."""
    plan = plan_generation("a 4k picture of a cat")
    recommended = plan.recommended
    assert recommended is not None
    assert recommended.model == DEFAULT_IMAGE_MODEL
    assert _leads_by_a_margin(plan, FLASH, PRO_IMAGE)
    # The size is still the reason the plan exists, so it stays in the prose.
    assert "4K output" in recommended.rationale


def _omni_animatic_usd(
    beats: int,
    beat_seconds: float,
    model: str = DEFAULT_OMNI_MODEL,
    resolution: str = "360p",
) -> float:
    """What an omni animatic of ``beats`` x ``beat_seconds`` really costs.

    Computed from the price book rather than restated, so a test cannot pass
    against a rationale that hard-codes a number the pricing no longer says.
    """
    from src.pricing import estimate_video_cost, quote_duration_for

    probe = estimate_video_cost(model, beat_seconds, resolution, False)
    assert probe is not None
    # Each render carries one frame of encoder allowance (omni lands ~0.01s
    # over nominal), so the plan's figure is per-render quoted seconds — the
    # same arithmetic the tool's per-beat dry_run performs.
    return (
        probe.breakdown["usd_per_second"]
        * beats
        * quote_duration_for(model, beat_seconds)
    )


@pytest.mark.parametrize(
    ("budget", "expected_model", "animatic_is_cheaper"),
    [
        # At 360p on omni 1.2 the preview is a third of omni's own 720p rate,
        # which puts it under every Veo tier. That is the point of previewing
        # at 360p.
        pytest.param("best", VEO, True, id="standard_tier_saves"),
        pytest.param(None, VEO_FAST, True, id="fast_tier_saves"),
    ],
)
def test_the_animatic_claims_a_saving_only_when_it_saves(
    budget: str | None, expected_model: str, animatic_is_cheaper: bool
) -> None:
    """The rationale may only claim a saving the arithmetic supports.

    It used to oversell an animatic that cost about what the delivery render
    cost: omni bills $0.10136/s against Veo Fast's $0.10/s, so 3 x 8s beats
    were ~$2.45 as a preview and $2.40 as the real thing. Previewing at 360p on
    omni 1.2 is a third of its 720p rate, so the saving is real against every tier —
    and the assertion that it be *stated* only when true is unchanged."""
    plan = plan_generation(
        "a 3 beat reel about coffee",
        RoutingConstraints(
            budget=cast(Any, budget),
            num_beats=3,
            # The whole clip's runtime: three 8s beats.
            duration_seconds=24,
            media_kind="video",
        ),
    )
    top = plan.recommended
    assert top is not None
    assert top.model == expected_model
    assert top.cost is not None

    # The advice survives in both directions: catching the mistake early has
    # value even when it is not the cheaper path.
    assert len(plan.workflow) == 2
    animatic = plan.workflow[0]
    assert animatic.params["animatic"] is True
    # Naming the resolution is what selects the model that has one.
    assert animatic.params["animatic_resolution"] == "360p"

    animatic_usd = _omni_animatic_usd(3, 8.0)
    render_usd = top.cost.usd
    assert (animatic_usd < render_usd) is animatic_is_cheaper

    rationale = animatic.rationale
    assert f"${animatic_usd:.2f}" in rationale
    assert f"${render_usd:.2f}" in rationale
    if animatic_is_cheaper:
        assert f"saving ~${render_usd - animatic_usd:.2f}" in rationale
        assert "does NOT save money" not in rationale
    else:
        assert "does NOT save money" in rationale
        assert "sav" not in rationale.replace("save money", "")
        assert f"{animatic_usd / render_usd:.2f}x" in rationale


def test_the_animatic_makes_no_economic_claim_when_pricing_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no price book there is no honest comparison to make, so the step is
    recommended on its creative merit and says nothing about money."""
    monkeypatch.setitem(sys.modules, "src.pricing", None)
    plan = plan_generation("a 4 shot reel about shoes")
    assert plan.workflow
    rationale = plan.workflow[0].rationale
    assert "$" not in rationale
    assert "sav" not in rationale
    assert DEFAULT_OMNI_MODEL in rationale


def test_gcs_advice_is_never_offered_on_the_gemini_api() -> None:
    """output_gcs_uri does not exist on the Gemini Developer API, so offering
    it as an option is advice that cannot be taken."""
    plan = plan_generation(
        "a video of a cat",
        RoutingConstraints(backend="gemini_api", wants_gcs_output=True),
    )
    for route in plan.routes:
        for caveat in route.caveats:
            assert "Set output_gcs_uri" not in caveat
        assert any("Vertex-only" in caveat for caveat in route.caveats)

    # ...and it is still offered where the parameter really exists.
    on_vertex = plan_generation(
        "a video of a cat",
        RoutingConstraints(backend="vertex", wants_gcs_output=True),
    )
    top = on_vertex.recommended
    assert top is not None
    assert any("Set output_gcs_uri" in caveat for caveat in top.caveats)


def test_planner_and_tool_agree_on_extension_price() -> None:
    """The planner billed a base render loop_extend never performs: $1.50 for
    the same call the tool quotes at $0.70. loop_extend extends a clip the
    caller already has."""
    from src.pricing import actual_video_cost

    plan = plan_generation(
        "extend this clip",
        RoutingConstraints(needs_extension=True, source_video_uri="file:///v.mp4"),
    )
    top = plan.recommended
    assert top is not None and top.cost is not None
    times = top.params["times"]

    # What loop_extend's own dry_run quotes: times x 7s with no re-snap. The
    # generic estimator would snap 7s down to 6s (ties go down), which is why
    # both the tool and the planner bypass it for extension chains.
    tool_quote = actual_video_cost(
        top.model, times * 7, "720p", True, snap_duration=False
    )
    assert tool_quote is not None
    assert top.cost.usd == pytest.approx(tool_quote.usd)


def test_video_cost_indices_track_the_published_rates() -> None:
    """The routing profiles' cost_index must order the same way real money
    does, or budget=cheap recommends the dearer route.

    omni sat at 0.0 — the cheapest slot in the family — while actually costing
    $0.10136/s against Veo Fast's $0.10/s, so a cheap budget ranked a $0.4054
    route above a $0.40 one and called it "cheap".
    """
    from src.pricing import estimate_video_cost
    from src.routing import _VIDEO_PROFILES

    by_rate: list[tuple[float, float, str]] = []
    for model, profile in _VIDEO_PROFILES.items():
        cost = estimate_video_cost(model, 8, "720p", True)
        assert cost is not None, model
        by_rate.append((cost.breakdown["usd_per_second"], profile.cost_index, model))

    ranked = sorted(by_rate)
    indices = [cost_index for _rate, cost_index, _model in ranked]
    assert indices == sorted(indices), (
        "cost_index must be non-decreasing in real $/s; got "
        + ", ".join(f"{m}={c} at ${r}/s" for r, c, m in ranked)
    )


def test_a_cheap_budget_never_recommends_a_dearer_route() -> None:
    """The end-to-end version of the above: the top route under budget=cheap
    must not cost more than a route it outranked."""
    from src.pricing import estimate_video_cost

    plan = plan_generation(
        "a video of steam", RoutingConstraints(budget="cheap", duration_seconds=4)
    )
    costs = [
        (route.model, estimate_video_cost(route.model, 4, "720p", True))
        for route in plan.routes
    ]
    priced = [(m, c.usd) for m, c in costs if c is not None]
    assert priced
    top_model, top_cost = priced[0]
    assert top_cost == min(usd for _m, usd in priced), (
        f"{top_model} won on a cheap budget at ${top_cost}, but a cheaper "
        f"route was available: {priced}"
    )


@pytest.mark.parametrize(
    ("intent", "expected_times"),
    [
        pytest.param(
            "extend my existing 8 second clip by another 30 seconds",
            5,
            id="delta_after_a_source_length",
        ),
        pytest.param("extend this by another 30 seconds", 5, id="delta_only"),
        pytest.param("extend this by 2 minutes", 18, id="delta_in_minutes"),
        pytest.param("extend this clip", 1, id="no_amount_named"),
    ],
)
def test_an_extension_amount_is_read_as_a_delta_not_a_total(
    intent: str, expected_times: int
) -> None:
    """ "extend my 8 second clip by another 30 seconds" planned ONE 7s
    extension. The duration pattern takes the first number in the sentence,
    which here is the SOURCE clip's length, not the amount to add. An amount
    named with "by"/"add"/"another" now drives the extension count directly.
    """
    plan = plan_generation(
        intent,
        RoutingConstraints(needs_extension=True, source_video_uri="file:///v.mp4"),
    )
    top = plan.recommended
    assert top is not None
    assert top.params["times"] == expected_times


def test_a_single_render_is_not_described_in_the_plural() -> None:
    """Cosmetic, but it is the first line a caller reads: "1 renders"."""
    plan = plan_generation(
        "extend this clip",
        RoutingConstraints(needs_extension=True, source_video_uri="file:///v.mp4"),
    )
    top = plan.recommended
    assert top is not None and top.cost is not None
    assert "1 render totalling" in top.cost.detail


# ============================================================================
# gemini-omni-1.2-flash routing
# ============================================================================


def test_the_new_omni_model_is_routable_and_names_itself_in_the_params() -> None:
    """A route the caller cannot reproduce is worse than no route.

    A plan has to emit omni_model or the call it hands over renders on the
    tool's default, which need not be the model it priced.
    """
    plan = plan_generation("a video of a cat", RoutingConstraints(media_kind="video"))
    omni_routes = [r for r in plan.routes if r.tool == "generate_video_omni"]
    assert {r.model for r in omni_routes} == set(OMNI_MODELS)
    for route in omni_routes:
        assert route.params["omni_model"] == route.model


def test_a_360p_request_reaches_only_the_model_that_has_360p() -> None:
    """360p is 1.2's draft tier alone.

    Veo publishes no 360p output and no 360p rate, and the preview model
    renders 720p whatever it is asked — so letting either through would quote
    a render that cannot happen, at a price that is not its own.
    """
    plan = plan_generation(
        "a draft clip of a robot",
        RoutingConstraints(media_kind="video", resolution="360p"),
    )
    assert {route.model for route in plan.routes} == {DEFAULT_OMNI_MODEL}
    assert plan.routes[0].params["resolution"] == "360p"
    rejected = {r.model for r in plan.rejected}
    assert OMNI_MODEL in rejected
    assert "veo-3.1-fast-generate-001" in rejected


def test_the_360p_route_is_the_cheapest_video_render_planned() -> None:
    """The draft tier's whole point: a third of 720p, under every Veo tier."""
    draft = plan_generation(
        "a clip of a robot",
        RoutingConstraints(media_kind="video", resolution="360p"),
    ).routes[0]
    others = plan_generation(
        "a clip of a robot", RoutingConstraints(media_kind="video")
    ).routes
    assert draft.cost is not None
    assert all(
        route.cost is None or draft.cost.usd < route.cost.usd for route in others
    )


def test_extending_from_an_interaction_offers_the_omni_extension_tool() -> None:
    """ "Make it longer" is an extension, even when an interaction id is present.

    An interaction id is a source, not a verb — omni can both edit AND extend
    from one. Ranking the id first routed every "extend this, from interaction
    X" to edit_video, which does not lengthen anything.
    """
    plan = plan_generation(
        "extend this clip by 20 seconds",
        RoutingConstraints(media_kind="video", previous_interaction_id="int-9"),
    )
    omni = [r for r in plan.routes if r.tool == "extend_video_omni"]
    assert omni, [r.tool for r in plan.routes]
    assert omni[0].model == DEFAULT_OMNI_MODEL
    assert omni[0].params["previous_interaction_id"] == "int-9"
    # 20s of continuation at 10s a turn.
    assert omni[0].params["times"] == 2
    # The preview model is excluded by capability, not silently absent.
    assert any(
        r.model == OMNI_MODEL and "cannot extend" in r.reason for r in plan.rejected
    )


def test_an_omni_extension_is_capped_at_the_turns_that_actually_fit() -> None:
    """The cumulative ceiling is on the ASSEMBLED clip, so it caps the chain.

    On the Gemini API 1.2 assembles at most 30s, which is three full 10s
    turns. Planning more turns than fit priced fewer than it told the caller
    to run.
    """
    plan = plan_generation(
        "extend this clip by 5 minutes",
        RoutingConstraints(media_kind="video", previous_interaction_id="int-9"),
    )
    omni = next(r for r in plan.routes if r.tool == "extend_video_omni")
    assert omni.params["times"] == 3
    assert any("30s" in caveat for caveat in omni.caveats)


def test_a_plain_edit_still_routes_to_edit_video() -> None:
    """The extension check must not swallow ordinary conversational edits."""
    plan = plan_generation(
        "make the sky stormy", RoutingConstraints(previous_interaction_id="int-1")
    )
    assert plan.routes[0].tool == "edit_video"


def test_the_extension_route_sends_the_resolution_it_is_priced_at() -> None:
    """It was the one omni branch that quoted a resolution it never asked for.

    A 360p extension plan handed over params with no `resolution` key at all:
    running them verbatim renders 720p and bills three times the quote — the
    under-quote this module's docstring says must never happen.
    """
    plan = plan_generation(
        "extend this clip so it is 30 seconds long",
        RoutingConstraints(
            media_kind="video", resolution="360p", previous_interaction_id="abc123"
        ),
    )
    route = next(r for r in plan.routes if r.tool == "extend_video_omni")
    assert route.params["resolution"] == "360p"
    assert route.cost is not None
    assert "@ 360p" in route.cost.detail


def test_an_extension_chain_is_priced_at_the_clip_each_turn_assembles() -> None:
    """A turn renders the ASSEMBLED clip, so the turns grow and so does the bill.

    MEASURED: a 3.01s source extended once returned 13.01s. Pricing the 10s
    increment instead under-billed a 2-turn chain 36s against a 20s quote.
    The planner cannot see the source, so it prices every turn at the
    assembled ceiling — assuming a shorter source would under-quote.
    """
    from src.omni import omni_extension_priced_lengths, omni_spec
    from src.pricing import OMNI_ENCODER_ALLOWANCE_SECONDS

    plan = plan_generation(
        "extend this clip by 40 seconds",
        RoutingConstraints(media_kind="video", previous_interaction_id="abc123"),
    )
    route = next(r for r in plan.routes if r.tool == "extend_video_omni")
    turns = route.params["times"]
    assert route.cost is not None
    expected = omni_extension_priced_lengths(omni_spec(route.model), None, turns)
    # The plan may not ask for more turns than it prices.
    assert len(expected) == turns, f"planned {turns} turns, priced {len(expected)}"
    assert route.cost.breakdown["seconds"] == pytest.approx(
        sum(expected) + turns * OMNI_ENCODER_ALLOWANCE_SECONDS
    )
    # The assembled clip, not the increment — the defect this replaced
    # quoted turns * 10s.
    assert sum(expected) > 10.0 * turns
    assert any("assembled output" in caveat for caveat in route.caveats)


def test_an_interaction_id_excludes_every_model_that_cannot_read_one() -> None:
    """An interaction id is a source only the omni family can resolve.

    Gating the conversational capability on `tool == "edit_video"` alone let an
    extension-flavoured request top out with a Veo `loop_extend` route that had
    silently dropped the interaction id and told the caller to supply a clip
    they do not have.
    """
    plan = plan_generation(
        "make this clip longer, continue the scene",
        RoutingConstraints(media_kind="video", previous_interaction_id="abc123"),
    )
    top = plan.recommended
    assert top is not None
    assert top.tool == "extend_video_omni"
    assert top.model == DEFAULT_OMNI_MODEL
    assert top.params["previous_interaction_id"] == "abc123"
    # No route may survive that cannot use the only source the caller gave.
    assert all(is_omni_model(route.model) for route in plan.routes)
    veo = {r.model for r in plan.rejected if r.model.startswith("veo")}
    assert veo, "the Veo tiers must be excluded, not merely out-ranked"


# ============================================================================
# Enumeration: a capability nobody can find is not exposed
# ============================================================================


def test_two_stills_offer_the_omni_interpolation_route_with_its_frames() -> None:
    """1.2 interpolates between a first and last frame; the planner never said so.

    A transition-shaped request produced Veo routes only, because omni was a
    candidate for four tools and generate_transition was not one of them — so a
    capability reachable on the tool was undiscoverable through the planner.
    And routing omni for two stills without EMITTING the stills would hand over
    a plan that renders something else entirely.
    """
    plan = plan_generation(
        "a smooth transition from sunrise to snowfall",
        RoutingConstraints(
            media_kind="video",
            first_frame_uri="gs://b/a.png",
            last_frame_uri="gs://b/b.png",
        ),
    )
    omni = [r for r in plan.routes if r.tool == "generate_video_omni"]
    assert omni, [(r.tool, r.model) for r in plan.routes]
    route = omni[0]
    assert route.model == DEFAULT_OMNI_MODEL
    assert route.params["first_frame_uri"] == "gs://b/a.png"
    assert route.params["last_frame_uri"] == "gs://b/b.png"
    assert any("FIRST_FRAME" in caveat for caveat in route.caveats)
    # Veo still competes; this adds an option rather than replacing one.
    assert any(r.tool == "generate_transition" for r in plan.routes)
    # And the preview model is excluded by capability, not merely out-ranked.
    assert any(
        r.model == OMNI_MODEL and "first+last-frame" in r.reason for r in plan.rejected
    )


def test_a_dear_single_render_is_offered_a_360p_draft_first() -> None:
    """ "Draft at 360p, finalize at 4K" is the workflow the tier exists for.

    Nothing enumerated it: plan.workflow was empty for every single-render
    plan, so the cheapest preview this server can issue was reachable only by
    a caller who already knew to ask for it.
    """
    plan = plan_generation(
        "a cinematic drone shot over mountains",
        RoutingConstraints(media_kind="video", resolution="4K"),
    )
    assert len(plan.workflow) == 2
    draft, delivery = plan.workflow
    assert draft.tool == "generate_video_omni"
    assert draft.params["omni_model"] == DEFAULT_OMNI_MODEL
    assert draft.params["resolution"] == "360p"
    # Reviewed in the deliverable's framing, or the review is of the wrong frame.
    assert draft.params["aspect_ratio"] == delivery.params["aspect_ratio"]
    assert "saving ~$" in draft.rationale
    assert delivery.params.get("resolution") == "4K"


def test_the_draft_step_is_not_offered_when_it_would_not_help() -> None:
    """No preview of a preview, and none where it saves nothing."""
    # Already a throwaway render.
    assert (
        plan_generation(
            "a clip of a robot",
            RoutingConstraints(media_kind="video", is_draft=True, resolution="4K"),
        ).workflow
        == ()
    )
    # A 360p omni render IS the draft; it needs no draft of its own.
    assert (
        plan_generation(
            "a clip of a robot",
            RoutingConstraints(media_kind="video", resolution="360p"),
        ).workflow
        == ()
    )


def test_the_animatic_previews_on_the_model_with_a_360p_tier() -> None:
    """The saving is the whole reason to recommend the step.

    At 720p on the preview model the animatic cost about what the delivery
    render cost, and the rationale had to concede it. At 360p it is a third of
    that, under every Veo tier.
    """
    plan = plan_generation(
        "a 5 beat reel about coffee",
        RoutingConstraints(media_kind="video", num_beats=5, duration_seconds=8),
    )
    animatic = plan.workflow[0]
    assert animatic.params["animatic"] is True
    assert animatic.params["animatic_resolution"] == "360p"
    assert DEFAULT_OMNI_MODEL in animatic.rationale
    assert "does NOT save money" not in animatic.rationale


# ============================================================================
# Planner defects found by live harness testing
# ============================================================================


@pytest.mark.parametrize(
    "intent",
    [
        "a cinematic drone shot over a coastline",
        "a drone shot",
        "a tracking shot of a runner",
        "slow motion water droplet",
        "an establishing shot of the city",
        "the camera pans across a desk",
    ],
)
def test_camera_vocabulary_settles_a_brief_that_names_no_medium(intent: str) -> None:
    """ "A cinematic drone shot over a coastline" planned as an IMAGE.

    The video vocabulary had no cinematography in it, so a brief written the
    way a director writes one matched nothing at all.
    """
    assert plan_generation(intent).media_kind == "video"


@pytest.mark.parametrize(
    "intent",
    [
        "a cinematic portrait of a chef",
        "a wide shot photo of the logo",
        "a poster with a drone shot aesthetic",
        "a photograph, cinematic lighting",
        "a product shot with cinematic lighting",
    ],
)
def test_camera_vocabulary_does_not_overrule_someone_who_said_photo(
    intent: str,
) -> None:
    """Camera words sit in the weaker tier for exactly this reason.

    A strong video hit wins outright over an image word, so putting
    "cinematic" there would have turned every cinematic still into a video.
    """
    assert plan_generation(intent).media_kind == "image"


def test_a_transition_brief_excludes_models_that_cannot_interpolate() -> None:
    """The module docstring names this as an example it catches.

    It once ranked a model that cannot interpolate anyway: the need was keyed
    on the chosen tool, and with no endpoint URIs yet — which is how an agent
    asks before it has the stills — the tool was generate_video and nothing
    needed anything.
    """
    plan = plan_generation("a smooth transition from sunrise to snowfall")
    rejected = {r.model for r in plan.rejected}
    assert OMNI_MODEL in rejected, [r.model for r in plan.routes]
    assert all(
        "first+last-frame" in r.reason for r in plan.rejected if r.model == OMNI_MODEL
    )


def test_the_missing_endpoint_conflict_is_a_sentence() -> None:
    """It read "but only neither frame is available"."""
    both = plan_generation("a smooth transition from sunrise to snowfall")
    detail = next(
        c.detail
        for c in both.conflicts
        if c.code == "transition_requires_two_endpoints"
    )
    assert "but neither frame is available" in detail
    assert "only neither" not in detail

    one = plan_generation("a crossfade", RoutingConstraints(first_frame_uri="a"))
    detail = next(
        c.detail for c in one.conflicts if c.code == "transition_requires_two_endpoints"
    )
    assert "only the first frame is available" in detail


def test_a_brief_naming_a_prior_interaction_excludes_veo() -> None:
    """loop_extend needs a Veo video_uri and cannot take an interaction_id.

    It out-ranked the omni route 0.5875 to 0.5425 on a brief that says the
    source IS an interaction, so the top-ranked route could not run at all.
    """
    plan = plan_generation(
        "continuing from the interaction I already rendered, make it longer"
    )
    top = plan.recommended
    assert top is not None
    assert top.tool == "extend_video_omni"
    assert all(is_omni_model(route.model) for route in plan.routes)
    assert any(r.model.startswith("veo") for r in plan.rejected)


def test_a_file_based_extension_still_offers_veo() -> None:
    """The interaction signal must not exclude Veo from ordinary extensions."""
    plan = plan_generation(
        "make this clip longer",
        RoutingConstraints(media_kind="video", source_video_uri="file:///d/a.mp4"),
    )
    assert any(r.tool == "loop_extend" for r in plan.routes)


def test_a_deprecated_model_never_outranks_its_replacement() -> None:
    """They scored level, so the dying endpoint was offered as an equal choice.

    33 days from shutdown at the time this was found.
    """
    plan = plan_generation("a video of a cat", RoutingConstraints(media_kind="video"))
    omni = [r for r in plan.routes if is_omni_model(r.model)]
    assert [r.model for r in omni][0] == DEFAULT_OMNI_MODEL
    deprecated = next(r for r in omni if r.model == OMNI_MODEL)
    assert any("deprecated" in c and "2026-10-22" in c for c in deprecated.caveats)
    # The GA model carries no such caveat.
    ga = next(r for r in omni if r.model == DEFAULT_OMNI_MODEL)
    assert not any("deprecated" in c for c in ga.caveats)


def test_the_deprecation_caveat_states_the_successors_rate_as_priced() -> None:
    """It said the successor "renders the same 720p at the same published
    rate", true of 1.1 and false of 1.2, which bills less: the plan
    contradicted the quotes printed beside it."""
    from src.pricing import estimate_video_cost

    plan = plan_generation("a video of a cat", RoutingConstraints(media_kind="video"))
    deprecated = next(r for r in plan.routes if r.model == OMNI_MODEL)
    [caveat] = [c for c in deprecated.caveats if "GA successor" in c]
    old = estimate_video_cost(OMNI_MODEL, 10, resolution="720p")
    new = estimate_video_cost(DEFAULT_OMNI_MODEL, 10, resolution="720p")
    assert old is not None and new is not None
    assert new.usd < old.usd
    assert "lower published rate" in caveat, caveat
    assert "same published rate" not in caveat, caveat


def test_the_deprecation_demotion_is_what_orders_a_tie(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercised with synthetic ids, so neither alphabetical order nor a score
    can be what puts the replacement first.

    "gemini-omni-1.2-flash" already sorts before "gemini-omni-flash-preview",
    so the real pair cannot show the ID half of this. And a tie break is not
    enough on its own: it held only while the preview had a GA twin that
    scored level with it (1.1); the balanced budget scores the preview ABOVE
    1.2, so the demotion has to outrank the score too.
    """
    import src.routing as routing
    from src.routing import ModelProfile, RoutedCall, _rank

    deprecated, replacement = "aaa-old-model", "zzz-new-model"
    monkeypatch.setattr(routing, "DEPRECATED_VIDEO_MODELS", {deprecated: "2026-09-30"})

    def profile(model: str) -> ModelProfile:
        return ModelProfile(
            model=model,
            media_kind="video",
            cost_index=0.5,
            fidelity_index=0.5,
            speed_index=0.5,
            text_rendering_index=0.0,
            summary=model,
        )

    # Alphabetically the deprecated id comes FIRST, so only the demotion can
    # put the replacement on top.
    assert deprecated < replacement
    routes = [
        RoutedCall(
            tool="generate_video_omni",
            model=m,
            params={},
            score=0.5,
            rationale="",
            caveats=(),
            cost=None,
        )
        for m in (deprecated, replacement)
    ]
    profiles = {deprecated: profile(deprecated), replacement: profile(replacement)}
    # Identical fidelity, identical score: the only separator left is the
    # shutdown date.
    ranked = _rank(routes, profiles)
    assert ranked[0].model == replacement
    assert ranked[1].model == deprecated

    # A deprecated model that scores HIGHER still ranks below the live one.
    from dataclasses import replace

    ranked = _rank([replace(routes[0], score=0.9), routes[1]], profiles)
    assert [r.model for r in ranked] == [replacement, deprecated]


# ============================================================================
# Signal-parsing defects found reviewing this PR
#
# Every case below was a wrong number or a wrong claim handed to a calling
# agent, and none of them raised: the plan looked well-formed and said
# something false. They are grouped here because they share one shape --
# a pattern or a note that was right about the case it was written for and
# never checked the case next to it.
# ============================================================================


@pytest.mark.parametrize(
    ("intent", "expected"),
    [
        # The bug: the digit after the colon read as a shot count, so a
        # one-render vertical brief planned a 16-beat clip quoted at $9.60.
        ("a 9:16 shot of a dancer", None),
        ("a 16:9 shot", None),
        ("a 1920x1080 panel", None),
        ("2.5 scenes", None),
        # Still counted, or the guard has gone too far.
        ("3 beats", 3),
        ("5 shots", 5),
        ("4 keyframe panels", 4),
        ("10 shots", 10),
        ("render 9:16 with 3 shots", 3),
        # Pre-existing guards that must survive.
        ("a 4k panel", None),
        ("24 frames per second", None),
    ],
)
def test_a_digit_inside_another_number_is_not_a_shot_count(
    intent: str, expected: int | None
) -> None:
    """Only a digit that is not part of a larger number counts segments."""
    assert infer_signals(intent).beat_count == expected


@pytest.mark.parametrize(
    ("intent", "expected"),
    [
        # The bug: seconds won outright, so the minutes were dropped and a
        # 2.5-minute brief was planned as one sub-8s render.
        ("a 2 minute 30 second trailer", 150.0),
        ("a 2 min 30 sec trailer", 150.0),
        ("a 1 minute 5 second clip", 65.0),
        # Single-unit runtimes are unchanged.
        ("a 90 second promo", 90.0),
        ("a 2 minute trailer", 120.0),
        ("a 6 second clip", 6.0),
    ],
)
def test_a_runtime_naming_both_units_sums_them(intent: str, expected: float) -> None:
    """ "2 minute 30 second" is 150s, not 30s."""
    assert infer_signals(intent).duration_seconds == expected


def test_a_gcs_uri_in_prose_is_recognised() -> None:
    """The "gs://" term could never match a real URI.

    Every term was wrapped in a trailing ``(?!\\w)``, so ``gs://`` required a
    non-word character after the final slash -- and a real URI has a letter
    there. The conflict and the Vertex advice that depend on this signal were
    both silently unreachable; it only appeared to work when the URI happened
    to contain the separate term ``bucket``.
    """
    assert infer_signals("render the video to gs://renders/out.mp4").wants_gcs_output
    assert infer_signals("render a video of a cat").wants_gcs_output is False


def test_punctuated_terms_still_anchor_on_the_side_that_has_a_word_char() -> None:
    """The fix must not turn the boundary check off wholesale."""
    from src.routing import _term_pattern

    assert _term_pattern("gs://").search("upload to gs://renders/out.mp4")
    assert _term_pattern("gs://").search("xgs://nope") is None
    assert _term_pattern("b-roll").search("some b-roll here")
    assert _term_pattern("b-roll").search("ab-roll") is None
    assert _term_pattern("9:16").search("shoot 9:16 please")
    assert _term_pattern("9:16").search("19:164") is None


def test_a_ten_second_ask_is_one_omni_render_not_an_extension_chain() -> None:
    """Omni renders natively up to 10s; the planner measured against Veo's 8s.

    A "10 second video" was capped to an 8s clip, flagged needs_extension and
    routed to loop_extend -- which needs a video_uri the caller does not have
    -- plus a two-step seed+extend workflow, when
    generate_video_omni(duration_seconds=10) is a single call.
    """
    plan = plan_generation("a 10 second video of a sunset over the ocean")
    assert plan.request.needs_extension is False
    assert plan.request.clip_duration_seconds == 10.0
    tools = [route.tool for route in plan.routes]
    assert "generate_video_omni" in tools
    assert "loop_extend" not in tools
    omni = next(r for r in plan.routes if r.tool == "generate_video_omni")
    assert omni.params["duration_seconds"] == 10.0


def test_a_runtime_past_every_single_render_still_plans_an_extension() -> None:
    """The cap moved from 8s to 10s; it did not go away."""
    plan = plan_generation("a 30 second video of a sunset over the ocean")
    assert plan.request.needs_extension is True
    assert plan.request.clip_duration_seconds == 8.0


def test_the_no_keyword_note_names_the_kind_that_was_planned() -> None:
    """The note said "an image" even when a video was planned.

    It was gated only on "no keyword matched", ignoring resolve_request's
    structurally_video fallback -- and the MCP tool forwards `notes` verbatim,
    so a calling agent was handed the opposite of what happened.
    """
    video = plan_generation(
        "a golden retriever running on a beach",
        RoutingConstraints(duration_seconds=6.0),
    )
    assert video.request.media_kind == "video"
    assert any("planned as a video" in note for note in video.notes)
    assert not any("planned as an image" in note for note in video.notes)

    image = plan_generation("a golden retriever running on a beach")
    assert image.request.media_kind == "image"
    assert any("planned as an image" in note for note in image.notes)


def test_a_reference_route_is_priced_and_parameterised_for_the_mode_it_runs() -> None:
    """Veo forces reference_to_video to 8s, and the route priced the request.

    src/video.py sets `effective_duration = 8` for this mode, so 3 references
    at duration_seconds=4 was quoted $0.40 (4s) while generate_video's own
    dry_run quoted $0.80 (8s) for the same intent. The emitted params also
    omitted reference_image_uris entirely while the rationale read "picked for
    subject consistency from reference images" -- a plan recommending a model
    for inputs it never asks for.
    """
    plan = plan_generation(
        "a product spinning on a turntable",
        RoutingConstraints(
            media_kind="video",
            num_reference_images=3,
            duration_seconds=4.0,
            backend="vertex",
        ),
    )
    route = next(r for r in plan.routes if r.tool == "generate_video")
    assert route.params["duration_seconds"] == 8.0
    assert route.cost is not None
    assert route.cost.usd == pytest.approx(0.8)
    assert any("reference_image_uris" in c for c in route.caveats)
    assert any("exactly 8s" in c for c in route.caveats)


def test_a_veo_route_on_an_unknown_backend_says_it_needs_vertex() -> None:
    """When the backend is not known the route is still offered, but Veo is
    refused on the Gemini Developer API, so the route must say so — a caveat
    that may not apply is cheaper than a refused call."""
    plan = plan_generation(
        "a product spinning on a turntable",
        RoutingConstraints(
            media_kind="video",
            num_reference_images=3,
            backend="unknown",
        ),
    )
    route = next(r for r in plan.routes if r.tool == "generate_video")
    assert any("Vertex AI only" in c and "2026-10-22" in c for c in route.caveats), (
        route.caveats
    )


def test_a_plain_text_to_video_route_gains_no_reference_caveats() -> None:
    """The new caveats must be scoped to the mode that earns them."""
    plan = plan_generation(
        "a sunset over the ocean",
        RoutingConstraints(media_kind="video", duration_seconds=6.0, backend="vertex"),
    )
    route = next(r for r in plan.routes if r.tool == "generate_video")
    assert route.params["duration_seconds"] == 6.0
    assert not any("reference_image_uris" in c for c in route.caveats)
    assert not any("reference_to_video" in c for c in route.caveats)


@pytest.mark.parametrize(
    ("intent", "expected"),
    [
        # The bug: two bare words with a common non-format meaning silently
        # chose a vertical frame.
        ("an illustration telling the story of a rabbit", None),
        ("a short video of a cat", None),
        # The platform senses still count.
        ("an instagram story of a cat", "9:16"),
        ("the stories format for ig", "9:16"),
        ("a youtube shorts clip", "9:16"),
        ("a tiktok of a cat", "9:16"),
        ("a reel of a cat", "9:16"),
        ("a vertical video", "9:16"),
        ("a portrait photo", "9:16"),
        ("a 9:16 video", "9:16"),
        # And the other orientations are untouched.
        ("a widescreen landscape shot", "16:9"),
        ("a square logo", "1:1"),
    ],
)
def test_a_format_word_is_not_read_out_of_a_narrative(
    intent: str, expected: str | None
) -> None:
    """ "story" is a narrative unless a platform is named; "short" is a length."""
    assert infer_signals(intent).aspect_ratio == expected


@pytest.mark.parametrize(
    ("intent", "expected"),
    [
        # One runtime in two units: summed.
        ("a 2 minute 30 second trailer", 150.0),
        ("a 2 min 30 sec trailer", 150.0),
        ("2 minutes and 30 seconds of footage", 150.0),
        ("a 1 minute, 5 second clip", 65.0),
        # A minute figure that is NOT part of the runtime: seconds win, as
        # they did before the first fix over-corrected. "I need it in 5
        # minutes" read as 306s and planned a $14 extension chain for a 6s clip.
        ("a 6 second clip of a cat, I need it in 5 minutes", 6.0),
        ("a 2 minute video made of 8 second clips", 8.0),
        ("a 2 minute trailer and a 30 second teaser", 30.0),
        ("a 30 second loop, keep it under 2 minutes of render time", 30.0),
        # Single units unchanged.
        ("a 2 minute trailer", 120.0),
        ("a 90 second promo", 90.0),
    ],
)
def test_two_duration_units_are_summed_only_when_adjacent(
    intent: str, expected: float
) -> None:
    """ "2 minute 30 second" is one number; a deadline elsewhere is not."""
    assert infer_signals(intent).duration_seconds == expected


@pytest.mark.asyncio
async def test_the_planners_extension_quote_matches_the_tools_on_vertex(
    tmp_path: Any,
) -> None:
    """The plan priced the model's 10s fallback list on both backends, so on a
    Vertex-only server it said $9.14 for three turns while extend_video_omni's
    own dry_run said $12.18 -- and a max_cost_usd taken from the plan was
    refused. Parity is asserted against the tool itself, on a matching
    deployment, for both backends."""
    import json
    from unittest.mock import AsyncMock, MagicMock

    from src.__main__ import AppContext, extend_video_omni

    def tool_ctx(*, vertex: bool) -> Any:
        (tmp_path / "images").mkdir(exist_ok=True)
        (tmp_path / "videos").mkdir(exist_ok=True)
        client = MagicMock()
        client._api_client.vertexai = vertex
        ctx = MagicMock()
        ctx.info = AsyncMock()
        ctx.error = AsyncMock()
        ctx.warning = AsyncMock()
        ctx.request_context.lifespan_context = AppContext(
            data_folder=tmp_path,
            images_dir=tmp_path / "images",
            videos_dir=tmp_path / "videos",
            client=client,
            gemini_api_client=None if vertex else client,
        )
        return ctx

    span = "extend my omni clip from interaction abc by another 30 seconds"
    for backend in ("vertex", "gemini_api"):
        plan = plan_generation(
            span,
            RoutingConstraints(
                backend=backend,
                gemini_api_key_available=backend != "vertex",
                needs_extension=True,
                previous_interaction_id="abc",
            ),
        )
        route = next(
            r
            for r in plan.routes
            if r.tool == "extend_video_omni" and r.model == OMNI_1_2_MODEL
        )
        quote = json.loads(
            await extend_video_omni(
                ctx=tool_ctx(vertex=backend == "vertex"),
                prompt=span,
                previous_interaction_id="abc",
                times=int(route.params["times"]),
                omni_model=OMNI_1_2_MODEL,
                dry_run=True,
            )
        )
        assert "error" not in quote, quote
        assert route.cost is not None
        assert route.cost.usd == pytest.approx(
            quote["estimated_cost"]["usd"], abs=0.005
        ), (
            backend,
            quote["turn_output_seconds"],
        )
        if backend == "vertex":
            assert quote["turn_output_seconds"] == [40.0] * int(route.params["times"])


@pytest.mark.parametrize(
    ("intent", "expected_added"),
    [
        ("extend my 8 second clip by another 2 minutes 30 seconds", 150.0),
        ("extend it by 2 min 30 sec", 150.0),
        ("extend it by 30 seconds", 30.0),
        ("add another 2 minutes", 120.0),
        # A seconds delta followed by an unrelated seconds figure is not a
        # compound.
        ("extend by 10 seconds and 5 seconds of credits", 10.0),
    ],
)
def test_the_delta_reads_a_compound_the_way_the_total_does(
    intent: str, expected_added: float
) -> None:
    """duration and added_duration came from one phrase and disagreed:
    "by another 2 minutes 30 seconds" -> total 150s, delta 120s."""
    signals = infer_signals(intent)
    assert signals.added_duration_seconds == expected_added
    if "2 min" in intent and "30" in intent:
        assert signals.duration_seconds == signals.added_duration_seconds


# ============================================================================
# Second review pass: planner inputs
# ============================================================================


def test_a_fully_negated_intent_is_linear_in_its_length() -> None:
    """_is_negated handed the ENTIRE prefix to _negator_precedes on every match
    and re-split it each time, so a negated intent cost O(matches x length):
    "no video " x 6000 took 2.6s, x 12000 took 9.9s, inline on the loop with
    no length cap anywhere. Only the last clause's last three words are ever
    read, so a bounded window is exact -- and this pins the complexity, since
    an assertion on the result passes just as well when it takes ten seconds.
    """
    import time

    text = "no video " * 12000  # 108k chars, every term negated
    start = time.perf_counter()
    signals = infer_signals(text)
    elapsed = time.perf_counter() - start
    assert signals.media_kind != "video"
    assert elapsed < 1.5, f"{elapsed:.2f}s"


def test_the_negation_window_is_counted_in_words_not_characters() -> None:
    """The first bounded window was 120 characters, and a single token longer
    than that between the negator and the term cut the negator off: "no " +
    130-char hashtag + " video" stopped reading as negated. Counting back in
    whitespace-delimited words is exact whatever the token lengths, and a cut
    token can never read as a negator ("techno" -> "no") because the window
    always starts on a boundary.
    """
    from src.routing import _is_negated

    assert _is_negated("no " + "x" * 130 + " video please", "video") is True
    assert (
        _is_negated("without " + "y" * 60 + " " + "z" * 60 + " video", "video") is True
    )
    assert _is_negated("x techno beat video", "video") is False
    assert _is_negated("a loud track with no video", "video") is True
    # A clause break inside the window still ends the negator's reach.
    assert _is_negated("no audio, a video of rain", "video") is False


def test_an_intent_past_the_length_cap_is_refused() -> None:
    from src.routing import MAX_INTENT_CHARS

    with pytest.raises(ValueError, match=str(MAX_INTENT_CHARS)):
        plan_generation("a" * (MAX_INTENT_CHARS + 1))
    # At the cap is fine.
    plan_generation("a video of a cat " + "x" * (MAX_INTENT_CHARS - 40))


@pytest.mark.parametrize(
    "intent",
    [
        "a 24 frames per second animation",
        "a 24 frames per sec animation",
        "a 24 frames/second animation",
        "a 24 frames-per-second animation",
        "a 24 frames / sec animation",
        "24 frames/s of a running horse",
        # The "per" spelled another way: one word from the guard's list still
        # planned 20 paid beats.
        "a 24 frames a second animation",
        "a 24 frames every second animation",
        "24 frames each second",
        "24 frames per minute",
    ],
)
def test_a_frame_rate_is_not_a_shot_count_however_it_is_spelled(intent: str) -> None:
    """The guard knew only "per second"; "24 frames per sec" planned 20 beats
    at $12.00 in place of one $0.60 render."""
    assert infer_signals(intent).beat_count is None


def test_a_frame_count_still_counts() -> None:
    assert infer_signals("4 frames").beat_count == 4
    assert infer_signals("4 keyframes of a sunrise").beat_count == 4


@pytest.mark.parametrize(
    ("intent", "expected"),
    [
        # The comment's own example, which failed: _word_after saw "of".
        ("30s of video of a beach", 30.0),
        ("an 8s clip", 8.0),
        ("8s of footage", 8.0),
        # Bare units without a runtime noun still do not count.
        ("the 1970s diner", None),
        ("a 3 m tall robot", None),
    ],
)
def test_a_bare_unit_may_sit_one_link_word_from_its_noun(
    intent: str, expected: float | None
) -> None:
    assert infer_signals(intent).duration_seconds == expected


def test_the_planner_honours_a_server_wide_omni_backend_pin() -> None:
    """With OMNI_BACKEND=vertex the server keeps omni on Vertex even though a
    key is present; the planner must price Vertex's ceiling for the same
    deployment, or the plan and the tool disagree again."""
    from src.omni import omni_extension_priced_lengths, omni_spec

    base = dict(
        backend="vertex",
        gemini_api_key_available=True,
        needs_extension=True,
        previous_interaction_id="abc",
    )
    span = "extend my omni clip from interaction abc by another 30 seconds"
    auto = plan_generation(span, RoutingConstraints(**base))
    pinned = plan_generation(span, RoutingConstraints(**base, omni_backend="vertex"))
    pick = lambda plan: next(  # noqa: E731
        r
        for r in plan.routes
        if r.tool == "extend_video_omni" and r.model == OMNI_1_2_MODEL
    )
    a, v = pick(auto), pick(pinned)
    assert a.cost is not None and v.cost is not None
    # Both plan the three turns 30 more seconds takes. Auto (a key is
    # present) prices each at the Developer API's 30s assembled ceiling;
    # pinned to Vertex, each is priced at Vertex's 40s -- exactly what the
    # tool quotes on each deployment.
    assert int(a.params["times"]) == int(v.params["times"]) == 3
    spec = omni_spec(OMNI_1_2_MODEL)
    assert omni_extension_priced_lengths(spec, None, 3, vertexai=False) == [30.0] * 3
    assert omni_extension_priced_lengths(spec, None, 3, vertexai=True) == [40.0] * 3
    assert v.cost.usd > a.cost.usd  # 120s against 90s over three turns


# ============================================================================
# Omni 1.2: cost position and per-model ceilings
# ============================================================================


def test_video_cost_indices_sit_on_the_dollar_per_second_scale() -> None:
    """Every video cost_index is its 720p $/s over Veo standard's, the scale
    the profile comments describe. Ordering alone did not catch Omni 1.2 at
    0.0: it IS the cheapest, so 0.0 ordered correctly while sitting 0.17 off
    the scale, and the balanced budget then ranked it last."""
    from src.pricing import estimate_video_cost

    def per_second(model: str) -> float:
        cost = estimate_video_cost(model, 8, "720p", True)
        assert cost is not None, model
        return cost.breakdown["usd_per_second"]

    top = per_second(VEO)
    for model, profile in _VIDEO_PROFILES.items():
        assert profile.cost_index == pytest.approx(per_second(model) / top, abs=0.03), (
            f"{model}: cost_index {profile.cost_index} is off the $/s scale "
            f"({per_second(model)}/s over {top}/s)"
        )


def test_a_balanced_budget_ranks_omni_1_2_above_the_dearest_tier() -> None:
    """At cost_index 0.0 the budget term put the cheapest renderer level with
    the $0.40/s tier, and the fidelity tie-break then ranked it last."""
    plan = plan_generation("Create a video of waves crashing")
    models = _models(plan)
    assert DEFAULT_OMNI_MODEL in models
    assert models.index(DEFAULT_OMNI_MODEL) < models.index(VEO), models


@pytest.mark.parametrize(
    ("recorded", "ceiling"), [(OMNI_MODEL, 10), (DEFAULT_OMNI_MODEL, 15)]
)
def test_an_edit_quotes_and_names_its_own_models_maximum(
    recorded: str, ceiling: int
) -> None:
    """The edit caveat read a flat 15s for every model, so a 10s route said
    "Omni's 15s per-render maximum" beside a quote priced at 10s."""
    from src.pricing import estimate_video_cost

    plan = plan_generation(
        "make the sky purple in the clip",
        RoutingConstraints(
            previous_interaction_id="abc",
            previous_interaction_model=recorded,
            backend="gemini_api",
        ),
    )
    [route] = [route for route in plan.routes if route.tool == "edit_video"]
    assert route.model == recorded
    caveat = next(c for c in route.caveats if "per-render maximum" in c)
    assert f"{ceiling}s per-render maximum" in caveat, caveat
    expected = estimate_video_cost(recorded, ceiling, "720p", False)
    assert route.cost is not None and expected is not None
    assert route.cost.usd == pytest.approx(expected.usd)


def test_a_pinned_omni_model_too_short_for_the_ask_is_pointed_at_1_2() -> None:
    """Omni ceilings differ by model. A pinned 10s model asked for 13s was
    told only to use Veo, when 1.2 renders the full 13s in one call."""
    plan = plan_generation(
        "a 13 second video of waves",
        RoutingConstraints(pinned_model=OMNI_MODEL, backend="vertex"),
    )
    conflict = next(
        c for c in plan.conflicts if c.code == "pinned_model_duration_out_of_range"
    )
    assert conflict.resolution.startswith(f"Use {DEFAULT_OMNI_MODEL}")
    assert "15s in one call" in conflict.resolution
    assert "Veo" in conflict.resolution


def test_a_cheap_budget_is_not_outvoted_by_the_default_model_nudge() -> None:
    """On the $/s scale Omni 1.2 (0.17) and Veo Fast (0.25) are close, so the
    budget term separates them by 0.02 while the default nudge gave Fast 0.05:
    budget=cheap recommended the dearer default."""
    plan = plan_generation(
        "a video of steam", RoutingConstraints(budget="cheap", backend="vertex")
    )
    assert plan.recommended is not None
    assert plan.recommended.model == DEFAULT_OMNI_MODEL
    # Under the balanced budget, with no stated preference, the nudge still
    # applies and the documented default leads.
    balanced = plan_generation("a video of steam", RoutingConstraints(backend="vertex"))
    assert balanced.recommended is not None
    assert balanced.recommended.model == DEFAULT_VIDEO_MODEL


# ============================================================================
# Integration follow-ups
# ============================================================================


def test_a_multi_render_1_2_quote_carries_its_tables_date() -> None:
    """The hand-built multi-render estimate dropped ``as_of``, so a 1.2
    extension plan was dated by the full re-read, before 1.2 was verified."""
    from src.pricing import PRICING_AS_OF, cost_to_dict, estimate_video_cost

    plan = plan_generation(
        "a 40 second video of rain",
        RoutingConstraints(pinned_model=DEFAULT_OMNI_MODEL),
    )
    route = next(r for r in plan.routes if r.tool == "extend_video_omni")
    single = cost_to_dict(estimate_video_cost(DEFAULT_OMNI_MODEL, 6.0))
    assert single is not None and single["pricing_as_of"] > PRICING_AS_OF
    quoted = cost_to_dict(route.cost)
    assert quoted is not None
    assert quoted["pricing_as_of"] == single["pricing_as_of"]


def test_a_360p_rejection_points_at_the_default_omni_model() -> None:
    """It sent the draft to gemini-omni-1.1-flash, which this server has
    since retired."""
    plan = plan_generation(
        "a draft clip of a robot",
        RoutingConstraints(
            media_kind="video",
            resolution="360p",
            pinned_model="veo-3.1-fast-generate-001",
        ),
    )
    conflict = next(c for c in plan.conflicts if c.code.endswith("360p_unsupported"))
    assert DEFAULT_OMNI_MODEL in conflict.resolution
    assert OMNI_1_1_MODEL not in conflict.resolution


# ============================================================================
# Continuations: planned on the model that holds the interaction
# ============================================================================

_RETIRED_SPELLINGS = (OMNI_1_1_MODEL, "gemini-omni-1.1-flash-preview")


def _continuation(intent: str, **given: Any) -> RoutingPlan:
    return plan_generation(
        intent, RoutingConstraints(previous_interaction_id="abc", **given)
    )


@pytest.mark.parametrize("recorded", [OMNI_MODEL, DEFAULT_OMNI_MODEL])
def test_an_edit_is_one_route_on_the_model_that_recorded_the_interaction(
    recorded: str,
) -> None:
    """Every omni model got its own edit_video route, each naming its own
    omni_model, and the tool refuses any model but the recorded one -- so for
    an interaction on the model that ranked second, the top route was a call
    the tool refused."""
    plan = _continuation("make it night", previous_interaction_model=recorded)
    [route] = plan.routes
    assert route.tool == "edit_video"
    assert route.model == recorded
    assert route.params["omni_model"] == recorded
    assert route.params["previous_interaction_id"] == "abc"
    other = next(m for m in (OMNI_MODEL, DEFAULT_OMNI_MODEL) if m != recorded)
    assert "cannot change models" in _reason_for(plan, other)
    assert not plan.conflicts


def test_an_extension_is_one_route_on_the_recorded_model() -> None:
    plan = _continuation(
        "extend this clip by 10 seconds",
        media_kind="video",
        previous_interaction_model=DEFAULT_OMNI_MODEL,
    )
    [route] = plan.routes
    assert route.tool == "extend_video_omni"
    assert route.params["omni_model"] == DEFAULT_OMNI_MODEL
    assert any("model this server recorded" in c for c in route.caveats)


@pytest.mark.parametrize(
    ("intent", "tool"),
    [
        ("make it night", "edit_video"),
        ("extend this clip by 10 seconds", "extend_video_omni"),
    ],
)
def test_an_unrecorded_continuation_leaves_omni_model_to_the_tool(
    intent: str, tool: str
) -> None:
    """With nothing recorded, the plan cannot know which model holds the
    interaction; the tool can. One route, omni_model unset, priced on the
    model the tool falls back to."""
    plan = _continuation(intent, media_kind="video")
    routes = [route for route in plan.routes if route.tool == tool]
    assert len(routes) == 1
    [route] = routes
    assert route.model == DEFAULT_OMNI_MODEL
    assert "omni_model" not in route.params
    assert any("omni_model is left unset" in c for c in route.caveats)
    # The preview is accounted for, not silently dropped.
    assert OMNI_MODEL in _rejected_models(plan)


@pytest.mark.parametrize("recorded", _RETIRED_SPELLINGS)
@pytest.mark.parametrize(
    ("intent", "extra", "tool", "verb"),
    [
        ("make it night", {}, "generate_video_omni", "edit"),
        (
            "extend this clip by 10 seconds",
            {"media_kind": "video"},
            "extend_video_omni",
            "extend",
        ),
        (
            "upscale it to 1080p",
            {"backend": "vertex", "gemini_api_key_available": False},
            "upscale_video_omni",
            "upscale",
        ),
    ],
)
def test_an_interaction_recorded_on_a_retired_model_is_not_planned(
    recorded: str, intent: str, extra: dict[str, Any], tool: str, verb: str
) -> None:
    """A request naming 1.1 runs on 1.2, but an interaction cannot change
    models, so continuing one recorded on 1.1 is refused by the tools. The
    plan says so and gives the same way out: the saved clip, as
    input_video_uri, to the tool that takes it."""
    plan = _continuation(intent, previous_interaction_model=recorded, **extra)
    assert plan.routes == ()
    assert not plan.is_satisfiable
    conflict = next(c for c in plan.conflicts if c.code == "interaction_model_retired")
    assert recorded in conflict.detail and "cannot change models" in conflict.detail
    assert conflict.resolution.startswith(
        f"To {verb} that clip on {DEFAULT_OMNI_MODEL}, call {tool} with the clip "
        "it saved"
    )
    assert "input_video_uri" in conflict.resolution
    assert "in place of previous_interaction_id" in conflict.resolution


def test_an_interaction_on_the_switched_off_preview_is_not_planned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import datetime

    import src.omni as omni

    monkeypatch.setattr(omni, "_today", lambda: datetime.date(2026, 11, 9))
    plan = _continuation("make it night", previous_interaction_model=OMNI_MODEL)
    assert plan.routes == ()
    conflict = next(
        c for c in plan.conflicts if c.code == "interaction_model_switched_off"
    )
    assert "generate_video_omni" in conflict.resolution
    assert "input_video_uri" in conflict.resolution


def test_extending_a_preview_interaction_points_at_the_saved_clip() -> None:
    """The preview cannot extend, and nothing else can continue its
    interaction; the rule's own fix (Veo's loop_extend) drops the
    interaction the request is about."""
    plan = _continuation(
        "extend this clip by 10 seconds",
        media_kind="video",
        previous_interaction_model=OMNI_MODEL,
    )
    assert plan.routes == ()
    [conflict] = plan.conflicts
    assert conflict.resolution == (
        "An interaction cannot change models. To extend that clip on "
        f"{DEFAULT_OMNI_MODEL}, call extend_video_omni with the clip it saved "
        "(the video_url in its manifest) as input_video_uri in place of "
        "previous_interaction_id."
    )


def test_upscaling_a_preview_interaction_is_a_conflict() -> None:
    plan = _continuation(
        "upscale it to 1080p",
        backend="vertex",
        gemini_api_key_available=False,
        previous_interaction_model=OMNI_MODEL,
    )
    assert plan.routes == ()
    conflict = next(c for c in plan.conflicts if c.code == "upscale_interaction_model")
    assert "upscale_video_omni" in conflict.resolution
    assert "input_video_uri" in conflict.resolution


def test_a_pin_on_another_model_than_the_interactions_is_a_conflict() -> None:
    plan = _continuation(
        "make it night",
        previous_interaction_model=OMNI_MODEL,
        pinned_model=DEFAULT_OMNI_MODEL,
    )
    conflict = next(
        c
        for c in plan.conflicts
        if c.code == "pinned_model_cannot_continue_interaction"
    )
    assert OMNI_MODEL in conflict.detail
    assert "input_video_uri" in conflict.resolution
    # The one route that can run is still the recorded model's.
    assert [route.model for route in plan.routes] == [OMNI_MODEL]


def test_a_pin_on_the_retired_id_continues_a_1_2_interaction() -> None:
    """A pinned 1.1 plans as 1.2, which is the model that holds a 1.2
    interaction, so there is nothing to refuse."""
    plan = _continuation(
        "make it night",
        previous_interaction_model=DEFAULT_OMNI_MODEL,
        pinned_model=OMNI_1_1_MODEL,
    )
    [route] = plan.routes
    assert route.params["omni_model"] == DEFAULT_OMNI_MODEL
    assert not plan.conflicts


def test_previous_interaction_model_must_be_an_omni_model() -> None:
    with pytest.raises(ValueError, match="previous_interaction_model"):
        RoutingConstraints(
            previous_interaction_id="abc", previous_interaction_model=VEO_FAST
        )


def test_a_recorded_model_without_an_interaction_id_is_ignored() -> None:
    plan = plan_generation(
        "a video of a cat",
        RoutingConstraints(previous_interaction_model=OMNI_1_1_MODEL),
    )
    assert plan.request.previous_interaction_model is None
    assert plan.routes and not plan.conflicts


@pytest.mark.parametrize(
    "field, value",
    [
        ("previous_interaction_backend", "mars"),
        ("previous_interaction_resolution", "8K"),
    ],
)
def test_the_rest_of_the_record_is_validated(field: str, value: str) -> None:
    with pytest.raises(ValueError, match=field):
        RoutingConstraints(previous_interaction_id="abc", **{field: value})


def test_the_rest_of_the_record_without_an_interaction_id_is_ignored() -> None:
    plan = plan_generation(
        "upscale this video to 720p",
        RoutingConstraints(
            backend="vertex",
            source_video_uri="gs://bucket/a.mp4",
            previous_interaction_backend="gemini_api",
            previous_interaction_resolution="720p",
            previous_interaction_video_url="gs://bucket/old.mp4",
        ),
    )
    assert plan.request.previous_interaction_backend is None
    assert plan.request.previous_interaction_resolution is None
    assert plan.request.previous_interaction_video_url is None
    assert plan.routes and not plan.conflicts


def test_a_lost_interaction_names_the_clip_the_server_saved() -> None:
    """D2: the way out names the saved file when one is recorded, as the
    tool's refusal does, and says where to look when none is."""
    base = {
        "previous_interaction_id": "abc",
        "previous_interaction_model": OMNI_1_1_MODEL,
    }
    named = plan_generation(
        "make the sky stormy",
        RoutingConstraints(
            **base, previous_interaction_video_url="file:///data/videos/old.mp4"
        ),
    )
    [conflict] = named.conflicts
    assert conflict.resolution == (
        f"To edit that clip on {DEFAULT_OMNI_MODEL}, call generate_video_omni "
        "with input_video_uri='file:///data/videos/old.mp4' (the clip it saved) "
        "in place of previous_interaction_id."
    )
    [unnamed] = plan_generation(
        "make the sky stormy", RoutingConstraints(**base)
    ).conflicts
    assert "the video_url in its manifest" in unnamed.resolution


# ============================================================================
# Runtimes one Veo clip cannot reach
# ============================================================================


def _delivered_seconds(route: Any) -> float:
    """The runtime a route delivers, as the plan states it."""
    if route.tool == "loop_extend":
        return 8.0 + VEO_EXTENSION_SECONDS * route.params["times"]
    return float(route.params["duration_seconds"])


@pytest.mark.parametrize("seconds", [9, 10])
def test_a_short_overshoot_still_snaps_veo_to_8s(seconds: int) -> None:
    """Up to 10s a Veo route renders one 8s clip and says so, as it always
    has; no route chains an extension for a 2s shortfall."""
    plan = plan_generation(f"Create a {seconds} second video of a dog")
    veo = [route for route in plan.routes if not is_omni_model(route.model)]
    assert veo and all(route.tool == "generate_video" for route in veo)
    for route in veo:
        assert route.params["duration_seconds"] == 8.0
        assert any(f"{seconds}s snaps to 8s" in c for c in route.caveats)
    assert plan.workflow == ()


@pytest.mark.parametrize("seconds", [11, 12, 14, 15])
def test_a_runtime_past_one_veo_clip_is_never_planned_short(seconds: int) -> None:
    """MAX_SINGLE_RENDER_SECONDS moved from 10s to 15s with Omni 1.2, so a 12s
    ask stopped implying an extension: Veo Fast was planned at 8s, ranked
    first, with only a "12s snaps to 8s" caveat. Every route now reaches the
    runtime, and the one that renders it whole in one call leads."""
    plan = plan_generation(f"Create a {seconds} second video of a dog")
    top = plan.recommended
    assert top is not None
    assert top.tool == "generate_video_omni"
    assert top.params["duration_seconds"] == float(seconds)
    for route in plan.routes:
        assert _delivered_seconds(route) >= seconds, (route.tool, route.model)
        assert not any("snaps to" in c for c in route.caveats)
    veo = [route for route in plan.routes if not is_omni_model(route.model)]
    assert {route.model for route in veo} == {VEO, VEO_FAST}
    for route in veo:
        assert route.tool == "loop_extend"
        assert route.params["times"] == 1
        assert any("No clip exists yet" in c for c in route.caveats)
    assert f"the {seconds}s runtime in one render" in top.rationale


@pytest.mark.parametrize("seconds", [16, 20, 30])
def test_a_runtime_past_every_single_render_chains_from_a_seed(seconds: int) -> None:
    plan = plan_generation(f"Create a {seconds} second video of a dog")
    top = plan.recommended
    assert top is not None and top.tool == "loop_extend"
    assert _delivered_seconds(top) >= seconds
    assert [step.tool for step in plan.workflow] == ["generate_video", "loop_extend"]


def test_a_pinned_veo_model_reaches_12s_by_extending_a_seed() -> None:
    """Pinned to a model whose clip tops out at 8s, the 12s ask gets the
    extension workflow it had before 1.2, not one short clip."""
    plan = plan_generation(
        "Create a 12 second video of a dog", RoutingConstraints(pinned_model=VEO_FAST)
    )
    [route] = plan.routes
    assert route.tool == "loop_extend" and route.model == VEO_FAST
    assert [step.tool for step in plan.workflow] == ["generate_video", "loop_extend"]
    seed, extend = plan.workflow
    assert seed.params["duration_seconds"] == 8.0
    assert "step 1" in extend.params["video_uri"]


def test_a_12s_ask_omni_cannot_serve_still_reaches_12s_on_veo() -> None:
    """With a seed required, Omni is excluded; the Veo chain is the plan."""
    plan = plan_generation(
        "Create a 12 second video of a dog", RoutingConstraints(needs_seed=True)
    )
    assert plan.recommended is not None
    assert plan.recommended.tool == "loop_extend"
    assert all(_delivered_seconds(route) >= 12 for route in plan.routes)
    assert [step.tool for step in plan.workflow] == ["generate_video", "loop_extend"]


_FRAMES = {"first_frame_uri": "gs://b/a.png", "last_frame_uri": "gs://b/b.png"}


@pytest.mark.parametrize("seconds", [9, 10])
def test_a_transition_up_to_10s_is_planned_as_before(seconds: int) -> None:
    """Omni 1.2 interpolates up to 10s; Veo snaps a 9-10s ask to 8s and
    says so, as a plain render does."""
    plan = plan_generation(
        f"a {seconds} second transition between these two frames",
        RoutingConstraints(backend="vertex", **_FRAMES),
    )
    assert not plan.conflicts
    by_model = {route.model: route for route in plan.routes}
    assert by_model[DEFAULT_OMNI_MODEL].params["duration_seconds"] == seconds
    assert by_model[VEO_FAST].tool == "generate_transition"
    assert any("snaps to 8s" in c for c in by_model[VEO_FAST].caveats)


@pytest.mark.parametrize("seconds", [11, 12, 15, 20, 30])
@pytest.mark.parametrize("framed", [True, False], ids=["frames", "prose"])
def test_a_transition_no_model_interpolates_is_a_conflict(
    seconds: int, framed: bool
) -> None:
    """A 12s transition was ranked first as an 8s Veo transition, 4s short
    with only a snap caveat; past 15s it became a seed-and-extend chain whose
    seed ignored both frames. No model interpolates more than 10s between two
    endpoints, and an extension carries the shot past its last frame."""
    plan = plan_generation(
        f"a {seconds} second transition between these two frames",
        RoutingConstraints(backend="vertex", **(_FRAMES if framed else {})),
    )
    assert plan.routes == ()
    assert plan.workflow == ()
    [conflict] = [c for c in plan.conflicts if c.code == "transition_too_long"]
    assert f"transition {seconds}s long" in conflict.detail
    assert f"10s, on {DEFAULT_OMNI_MODEL}" in conflict.detail
    assert conflict.resolution.startswith("Ask for 10s or less.")
    reasons = {r.model: r.reason for r in plan.rejected}
    assert f"at most 8s ({seconds}s requested)" in reasons[VEO_FAST]
    assert f"at most 10s ({seconds}s requested)" in reasons[DEFAULT_OMNI_MODEL]


def test_a_pinned_model_too_short_for_the_transition_says_so() -> None:
    plan = plan_generation(
        "a 12 second transition between these two frames",
        RoutingConstraints(backend="vertex", pinned_model=VEO_FAST, **_FRAMES),
    )
    codes = {c.code for c in plan.conflicts}
    assert {"transition_too_long", "pinned_model_transition_too_long"} <= codes


@pytest.mark.parametrize("budget", ["cheap", "balanced", "best"])
def test_no_budget_plans_a_12s_ask_short(budget: str) -> None:
    plan = plan_generation(
        "Create a 12 second video of a dog", RoutingConstraints(budget=budget)
    )
    assert plan.routes
    assert all(_delivered_seconds(route) >= 12 for route in plan.routes)


# ============================================================================
# Omni 1.1 is gone from every plan
# ============================================================================


@pytest.mark.parametrize(
    ("intent", "constraints"),
    [
        ("a video of a cat", None),
        ("a 12 second video of waves", None),
        ("a 40 second video of rain", None),
        ("a 360p draft of a robot", RoutingConstraints(resolution="360p")),
        ("a 4K video of a skyline", RoutingConstraints(backend="vertex")),
        ("a crossfade", RoutingConstraints(first_frame_uri="a", last_frame_uri="b")),
        ("make it night", RoutingConstraints(previous_interaction_id="abc")),
        ("a video of a cat", RoutingConstraints(backend="gemini_api")),
    ],
)
def test_no_plan_offers_the_retired_model(
    intent: str, constraints: RoutingConstraints | None
) -> None:
    plan = plan_generation(intent, constraints)
    named = [route.model for route in plan.routes] + [
        str(value) for route in plan.routes for value in route.params.values()
    ]
    named += [str(step.params) for step in plan.workflow]
    named += [rejected.model for rejected in plan.rejected]
    assert not any("1.1" in value for value in named), named


def test_a_pin_on_the_retired_id_is_described_as_planned() -> None:
    """The rejected routes read "pinned_model=gemini-omni-1.2-flash was
    requested" when the caller had pinned 1.1."""
    plan = plan_generation(
        "a video of a cat", RoutingConstraints(pinned_model=OMNI_1_1_MODEL)
    )
    assert plan.notes[0].startswith(f"pinned_model={OMNI_1_1_MODEL}:")
    pinned_out = [r.reason for r in plan.rejected if "not planned" in r.reason]
    assert pinned_out
    for reason in pinned_out:
        assert "was requested" not in reason
        assert f"pinned to {DEFAULT_OMNI_MODEL}" in reason


def test_the_360p_caveat_cites_no_retired_source_and_holds() -> None:
    """It credited 1.1's launch post and said the pricing page published only
    the 720p rate; 1.2's 360p rate is on the pricing page, and is a third."""
    from src.pricing import estimate_video_cost

    plan = plan_generation("a video of a cat", RoutingConstraints(media_kind="video"))
    route = next(r for r in plan.routes if r.model == DEFAULT_OMNI_MODEL)
    caveat = next(c for c in route.caveats if "resolution='360p'" in c)
    assert "launch post" not in caveat
    assert "a third of the 720p price" in caveat
    draft = estimate_video_cost(DEFAULT_OMNI_MODEL, 6.0, "360p", True)
    full = estimate_video_cost(DEFAULT_OMNI_MODEL, 6.0, "720p", True)
    assert draft is not None and full is not None
    assert draft.usd / full.usd == pytest.approx(1 / 3, abs=0.01)


# ============================================================================
# Inline delivery of large Omni outputs on Vertex AI
# ============================================================================

_OMNI_ON_VERTEX = RoutingConstraints(backend="vertex", gemini_api_key_available=False)


@pytest.mark.parametrize(
    ("intent", "extra", "tool"),
    [
        ("a 4K video of a dog", {}, "generate_video_omni"),
        ("a 12 second video of a dog", {}, "generate_video_omni"),
        (
            "extend this clip by 10 seconds",
            {"previous_interaction_id": "abc", "media_kind": "video"},
            "extend_video_omni",
        ),
        (
            "make it night",
            {"previous_interaction_id": "abc", "resolution": "1080p"},
            "edit_video",
        ),
        # At 720p too: the service picks an edit's length, and edit_video
        # judges it at the 15s it could run to, as it quotes it.
        ("make it night", {"previous_interaction_id": "abc"}, "edit_video"),
    ],
)
def test_a_large_omni_output_on_vertex_recommends_a_bucket(
    intent: str, extra: dict[str, Any], tool: str
) -> None:
    """The tools deliver a large output inline on Vertex unless given
    output_gcs_uri, and warn before rendering; the plan says the same."""
    import dataclasses

    plan = plan_generation(intent, dataclasses.replace(_OMNI_ON_VERTEX, **extra))
    route = next(r for r in plan.routes if r.tool == tool)
    caveat = next(c for c in route.caveats if "output_gcs_uri" in c)
    assert "inline" in caveat and "billed" in caveat
    assert "refuse" not in caveat
    assert "output_gcs_uri" not in route.params


def test_a_4k_omni_route_quotes_the_tools_own_delivery_warning() -> None:
    """One policy, one wording: the plan's caveat is the warning the tool's
    dry run gives for the same call, plus why a bucket is the safer choice."""
    from src.omni import omni_inline_delivery_policy, omni_spec

    plan = plan_generation("a 4K video of a dog", _OMNI_ON_VERTEX)
    route = next(r for r in plan.routes if r.tool == "generate_video_omni")
    expected = omni_inline_delivery_policy(
        omni_spec(route.model),
        vertexai=True,
        output_gcs_uri=None,
        resolution="4K",
        task=None,
        output_seconds=route.params["duration_seconds"],
    )
    assert expected is not None
    assert any(c.startswith(expected) for c in route.caveats)


def test_a_server_that_refuses_large_inline_output_says_so_on_omni_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.omni as omni

    monkeypatch.setattr(omni, "OMNI_REFUSE_LARGE_INLINE_ON_VERTEX", True)
    plan = plan_generation("a 4K video of a dog", _OMNI_ON_VERTEX)
    route = next(r for r in plan.routes if r.tool == "generate_video_omni")
    assert any(
        c.startswith("Before rendering, generate_video_omni refuses")
        for c in route.caveats
    )


@pytest.mark.parametrize(
    ("intent", "constraints"),
    [
        pytest.param("a 6 second video of a dog", _OMNI_ON_VERTEX, id="small"),
        pytest.param(
            "a 4K video of a dog",
            RoutingConstraints(backend="gemini_api"),
            id="gemini_api_asks_for_uri_itself",
        ),
        pytest.param(
            "a 4K video of a dog in my bucket",
            RoutingConstraints(
                backend="vertex", gemini_api_key_available=False, wants_gcs_output=True
            ),
            id="bucket_already_asked_for",
        ),
    ],
)
def test_no_inline_delivery_advice_where_it_does_not_apply(
    intent: str, constraints: RoutingConstraints
) -> None:
    plan = plan_generation(intent, constraints)
    for route in plan.routes:
        assert not any("requested inline" in c for c in route.caveats), route.tool


def test_a_veo_pin_on_a_continuation_points_at_the_default_omni_model() -> None:
    """The conversational-edit fix and the seed conflict named the preview
    model, switched off on 2026-10-22, as the place edits run."""
    plan = plan_generation(
        "make it night",
        RoutingConstraints(
            previous_interaction_id="abc", pinned_model=VEO_FAST, needs_seed=True
        ),
    )
    by_code = {c.code: c for c in plan.conflicts}
    pinned = by_code["pinned_model_conversational_edit_unsupported"]
    assert DEFAULT_OMNI_MODEL in pinned.resolution
    assert OMNI_MODEL not in pinned.resolution
    assert OMNI_MODEL not in by_code["conversational_edit_without_seed_support"].detail


# ---------------------------------------------------------------------------
# A plan never offers an Omni route its tool refuses
# ---------------------------------------------------------------------------


def _recorded(
    backend: Literal["vertex", "gemini_api"],
    seconds: float,
    resolution: str = "720p",
    **extra: Any,
) -> RoutingConstraints:
    return RoutingConstraints(
        backend=backend,
        previous_interaction_id="i-1",
        previous_interaction_backend=backend,
        previous_interaction_model=DEFAULT_OMNI_MODEL,
        previous_interaction_resolution=resolution,
        previous_interaction_seconds=seconds,
        gemini_api_key_available=backend == "gemini_api",
        **extra,
    )


@pytest.mark.parametrize(
    "intent, seconds, tool, fragment",
    [
        pytest.param(
            "make it night in this video",
            25.0,
            "edit_video",
            "edits clips up to 15s",
            id="edit-of-a-25s-clip",
        ),
        pytest.param(
            "extend this video",
            30.0,
            "extend_video_omni",
            "at or over",
            id="extension-at-the-ceiling",
        ),
    ],
)
def test_a_recorded_source_the_tool_refuses_is_a_conflict_not_a_route(
    intent: str, seconds: float, tool: str, fragment: str
) -> None:
    """The planner offered an edit of a recorded 25s clip (1.2 edits up to
    15s), and an extension of a 30s clip on the Gemini API priced at nothing
    and blamed on the price book; both tools refuse before rendering."""
    plan = plan_generation(intent, _recorded("gemini_api", seconds))
    assert not [r for r in plan.routes if r.tool == tool], plan.routes
    [conflict] = plan.conflicts
    assert fragment in conflict.detail, conflict.detail
    assert "15s or less" in conflict.resolution, conflict.resolution


@pytest.mark.parametrize(
    "intent, seconds, tool",
    [
        ("make it night in this video", 15.0, "edit_video"),
        ("extend this video", 20.0, "extend_video_omni"),
    ],
)
def test_a_recorded_source_the_tool_takes_is_still_planned(
    intent: str, seconds: float, tool: str
) -> None:
    plan = plan_generation(intent, _recorded("gemini_api", seconds))
    assert [r for r in plan.routes if r.tool == tool], (plan.routes, plan.conflicts)


def test_vertex_keeps_a_window_so_a_long_clip_still_extends() -> None:
    plan = plan_generation("extend this video", _recorded("vertex", 30.0))
    assert [r for r in plan.routes if r.tool == "extend_video_omni"], plan.conflicts


def test_an_upscale_quote_uses_the_recorded_length() -> None:
    """A recorded 6s clip was quoted at the 15s bound ($0.52 against the
    tool's own $0.21), with a delivery caveat the tool never gives."""
    plan = plan_generation(
        "upscale this video to 1080p", _recorded("vertex", 6.0, "720p")
    )
    [route] = [r for r in plan.routes if r.tool == "upscale_video_omni"]
    assert route.cost is not None
    assert route.cost.detail.startswith("6.0"), route.cost.detail
    small = plan_generation(
        "upscale this video to 720p", _recorded("vertex", 6.0, "360p")
    )
    [route] = [r for r in small.routes if r.tool == "upscale_video_omni"]
    assert not [c for c in route.caveats if "4 MB" in c], route.caveats


def test_an_upscale_of_a_recorded_clip_over_15s_is_a_conflict() -> None:
    plan = plan_generation("upscale this video to 1080p", _recorded("vertex", 20.0))
    assert not plan.routes
    [conflict] = plan.conflicts
    assert conflict.code == "upscale_source_too_long", conflict


@pytest.mark.parametrize(
    "intent, constraints, planned",
    [
        pytest.param(
            "a video of a cat",
            RoutingConstraints(media_kind="video", num_reference_images=7),
            False,
            id="7-references",
        ),
        pytest.param(
            "a video of a cat",
            RoutingConstraints(media_kind="video", num_reference_images=6),
            True,
            id="6-references",
        ),
        pytest.param(
            "make it night in this video",
            _recorded("vertex", 6.0, num_reference_images=3),
            False,
            id="vertex-edit-with-3",
        ),
        pytest.param(
            "make it night in this video",
            _recorded("vertex", 6.0, num_reference_images=2),
            True,
            id="vertex-edit-with-2",
        ),
        pytest.param(
            "make it night in this video",
            _recorded("gemini_api", 6.0, num_reference_images=3),
            True,
            id="gemini-api-edit-with-3",
        ),
    ],
)
def test_reference_image_limits_are_the_tools(
    intent: str, constraints: RoutingConstraints, planned: bool
) -> None:
    plan = plan_generation(intent, constraints)
    omni = [r for r in plan.routes if r.model == DEFAULT_OMNI_MODEL]
    assert bool(omni) is planned, (plan.routes, plan.rejected, plan.conflicts)


def test_an_extension_caveat_says_when_the_source_length_was_recorded() -> None:
    """It said "The source length is unknown" beside a quote priced from the
    recorded 6s."""
    plan = plan_generation(
        "extend this video by 10 seconds", _recorded("gemini_api", 6.0)
    )
    [route] = [r for r in plan.routes if r.tool == "extend_video_omni"]
    assert route.cost is not None
    assert route.cost.detail.endswith(
        "16.0417s @ 720p on gemini-omni-1.2-flash ($0.069504/s)"
    ), route.cost.detail
    joined = " ".join(route.caveats)
    assert "unknown" not in joined, joined
    assert "from the 6s this server recorded" in joined, joined


def test_the_extension_ceiling_refusal_gives_advice_that_can_be_followed() -> None:
    """It said "edit the clip rather than extending it" about a clip of 30s
    or more, which 1.2 refuses to edit (it edits up to 15s)."""
    from src.omni import omni_extension_refusal, omni_spec

    refusal = omni_extension_refusal(omni_spec(DEFAULT_OMNI_MODEL), 30.0)
    assert refusal is not None
    assert "trim it to 15s or less and edit it" in refusal, refusal


@pytest.mark.parametrize(
    ("intent", "prompt"),
    [
        (
            "Generate a video of a fox, but don't upscale it",
            "Generate a video of a fox",
        ),
        (
            "Generate a video of a fox, but don’t upscale it",
            "Generate a video of a fox",
        ),
        (
            "Generate a video of a fox, but DON'T upscale it",
            "Generate a video of a fox",
        ),
        (
            "Generate a video of a fox, but do not upscale it",
            "Generate a video of a fox",
        ),
        (
            "Generate a video of a fox but do not upscale it",
            "Generate a video of a fox",
        ),
        ("a fox video, we won't need to upscale it", "a fox video"),
        ("a fox video, and you shouldn't upscale it", "a fox video"),
        ("a 4K video of a dog, don't upscale it", "a 4K video of a dog"),
        (
            "a 4K video of a dog but don't upscale it, keep it moody",
            "a 4K video of a dog, keep it moody",
        ),
        ("Please don't upscale it, a video of a cat", "a video of a cat"),
        ("a video of a fox. Don't upscale it.", "a video of a fox."),
    ],
)
def test_a_negated_upscale_plans_no_upscale(intent: str, prompt: str) -> None:
    """The upscale check knew "no" and "not" but no contraction, so "but
    don't upscale it" chained a paid upscale onto the render and cut the
    render's prompt to "...but don't"; "we won't need to upscale it" planned
    a standalone upscale of a clip nobody had. Then the declined clause went
    to the video model verbatim, asking it not to upscale; it leaves the
    prompt now, with the words that joined it."""
    plan = plan_generation(
        intent, RoutingConstraints(media_kind="video", backend="vertex")
    )
    assert not [r for r in plan.routes if r.tool == "upscale_video_omni"], plan.routes
    assert not [s for s in plan.workflow or () if s.tool == "upscale_video_omni"]
    top = plan.recommended
    assert top is not None
    assert top.params.get("prompt") == prompt, top.params
    assert plan.intent == intent
    assert any("declines it" in note for note in plan.notes), plan.notes


@pytest.mark.parametrize(
    "intent",
    [
        "a video of a dog, then upscale it. actually don't upscale it",
        "a video of a dog, then upscale it to 4K. Actually, don't upscale it",
    ],
)
def test_a_retracted_upscale_plans_no_upscale(intent: str) -> None:
    """Only a mention every occurrence of which was negated counted, so a
    later "actually don't" left the first "then upscale it" standing and
    planned the paid step. The last word on upscaling decides."""
    plan = plan_generation(
        intent, RoutingConstraints(media_kind="video", backend="vertex")
    )
    assert not [r for r in plan.routes if r.tool == "upscale_video_omni"]
    assert not [s for s in plan.workflow or () if s.tool == "upscale_video_omni"]
    top = plan.recommended
    assert top is not None
    assert "upscale" not in top.params["prompt"].lower(), top.params
    assert top.params.get("resolution") in (None, "720p"), top.params


def test_a_correction_back_to_upscaling_still_plans_it() -> None:
    plan = plan_generation(
        "a video of a dog, don't upscale it. Actually, upscale it to 4K",
        RoutingConstraints(media_kind="video", backend="vertex"),
    )
    [step] = [s for s in plan.workflow or () if s.tool == "upscale_video_omni"]
    assert step.params["resolution"] == "4K"
    render = plan.workflow[0]
    assert "upscale" not in render.params["prompt"].lower(), render.params


@pytest.mark.parametrize(
    "intent, chained",
    [
        ("Generate a video of a fox, then upscale it to 1080p", True),
        ("Upscale my video to 4K", False),
        ("Don't make a new video, upscale my clip to 4K", False),
    ],
)
def test_an_upscale_that_is_asked_for_is_still_planned(
    intent: str, chained: bool
) -> None:
    plan = plan_generation(
        intent, RoutingConstraints(media_kind="video", backend="vertex")
    )
    steps = [s.tool for s in plan.workflow or ()]
    if chained:
        assert "upscale_video_omni" in steps, steps
    else:
        top = plan.recommended
        assert top is not None and top.tool == "upscale_video_omni", plan.routes


@pytest.mark.parametrize(
    "intent, backend, tool",
    [
        ("make it night in this video", "gemini_api", "edit_video"),
        ("upscale this video to 1080p", "vertex", "upscale_video_omni"),
    ],
)
@pytest.mark.parametrize("seconds, planned", [(15.03, True), (15.5, False)])
def test_the_source_limit_allows_the_tools_one_frame(
    intent: str,
    backend: Literal["vertex", "gemini_api"],
    tool: str,
    seconds: float,
    planned: bool,
) -> None:
    """A recorded 15.03s clip -- a 15s render, measured a frame long -- was
    refused for an edit and an upscale that both tools accept, since each
    allows one frame over the limit."""
    plan = plan_generation(intent, _recorded(backend, seconds))
    routes = [r for r in plan.routes if r.tool == tool]
    assert bool(routes) is planned, (plan.routes, plan.conflicts)
