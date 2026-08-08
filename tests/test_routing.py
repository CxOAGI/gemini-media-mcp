"""Tests for routing.py — the intent router.

The router is pure, deterministic, rule-based logic, so everything here is a
plain call: no fixtures for clients, no network, no filesystem.
"""

import sys
from types import ModuleType
from typing import Any, cast

import pytest

from src.image import _IMAGE_SIZE_SUPPORT, ImageModel, ImageSize
from src.omni import OMNI_MODEL
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
VEO_LITE = "veo-3.1-lite-generate-preview"

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
    """Veo models plus omni are routable; capabilities exist for each."""
    assert set(_VIDEO_PROFILES) == set(LIVE_VIDEO_MODELS) | {OMNI_MODEL}
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
        ("a video of a cat", RoutingConstraints(budget="cheap"), OMNI_MODEL),
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
        "veo lite cannot interpolate between frames",
        RoutingConstraints(first_frame_uri="a", last_frame_uri="b"),
        VEO_LITE,
        "first+last-frame",
    ),
    (
        "extension_unsupported",
        "veo lite cannot extend",
        RoutingConstraints(media_kind="video", needs_extension=True),
        VEO_LITE,
        "cannot extend",
    ),
    (
        "reference_images_unsupported",
        "veo lite takes no reference images",
        RoutingConstraints(media_kind="video", num_reference_images=2),
        VEO_LITE,
        "reference images",
    ),
    (
        "4k_unsupported",
        "veo lite has no 4K",
        RoutingConstraints(media_kind="video", resolution="4K"),
        VEO_LITE,
        "cannot produce 4K",
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
    """The rule table and the test table must not drift apart."""
    assert {case[0] for case in _RULE_CASES} == {
        rule.code for rule in _VIDEO_CAPABILITY_RULES
    }


def test_veo_lite_is_not_offered_on_vertex() -> None:
    """Lite is published on the Gemini Developer API only."""
    plan = plan_generation("a video of a cat", RoutingConstraints(backend="vertex"))
    assert VEO_LITE not in _models(plan)
    reason = _reason_for(plan, VEO_LITE)
    assert "Gemini Developer API only" in reason
    assert "GEMINI_API_KEY" in reason


def test_veo_lite_is_offered_on_the_gemini_api() -> None:
    plan = plan_generation("a video of a cat", RoutingConstraints(backend="gemini_api"))
    assert VEO_LITE in _models(plan)


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


def test_gcs_output_excludes_veo_lite_even_on_an_unknown_backend() -> None:
    """Lite runs on the Gemini API, which has no GCS output at all."""
    plan = plan_generation(
        "a video of a cat", RoutingConstraints(wants_gcs_output=True)
    )
    assert VEO_LITE not in _models(plan)
    assert "output_gcs_uri" in _reason_for(plan, VEO_LITE)


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
        RoutingConstraints(media_kind="video", resolution="4K", pinned_model=VEO_LITE),
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
    """A 3-beat clip must not be quoted as a single clip."""
    single = plan_generation(
        "a video of a cat", RoutingConstraints(media_kind="video", duration_seconds=6)
    ).recommended
    clip = plan_generation(
        "a 3 shot reel", RoutingConstraints(duration_seconds=6)
    ).recommended
    assert single is not None and clip is not None
    assert single.cost is not None and clip.cost is not None
    assert clip.cost.usd == pytest.approx(single.cost.usd * 3)
    assert clip.cost.breakdown["renders"] == 3.0


def test_lite_route_discloses_its_gemini_api_constraints() -> None:
    plan = plan_generation("a video of a cat", RoutingConstraints(backend="gemini_api"))
    lite = [route for route in plan.routes if route.model == VEO_LITE]
    assert lite
    assert any("GEMINI_API_KEY" in caveat for caveat in lite[0].caveats)


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
    assert OMNI_MODEL in first.rationale


def test_a_draft_clip_gets_the_animatic_as_the_deliverable() -> None:
    plan = plan_generation("a rough 4 shot storyboard reel")
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


def _omni_animatic_usd(beats: int, beat_seconds: float) -> float:
    """What an omni animatic of ``beats`` x ``beat_seconds`` really costs.

    Computed from the price book rather than restated, so a test cannot pass
    against a rationale that hard-codes a number the pricing no longer says.
    """
    from src.pricing import estimate_video_cost

    probe = estimate_video_cost(OMNI_MODEL, beat_seconds, "720p", False)
    assert probe is not None
    return probe.breakdown["usd_per_second"] * beats * beat_seconds


@pytest.mark.parametrize(
    ("budget", "expected_model", "animatic_is_cheaper"),
    [
        # Only the standard tier is dearer than the animatic; against Fast the
        # preview costs slightly more, and against Lite about double.
        pytest.param("best", VEO, True, id="standard_tier_saves"),
        pytest.param(None, VEO_FAST, False, id="fast_tier_costs_more"),
        pytest.param("cheap", VEO_LITE, False, id="lite_tier_costs_double"),
    ],
)
def test_the_animatic_claims_a_saving_only_when_it_saves(
    budget: str | None, expected_model: str, animatic_is_cheaper: bool
) -> None:
    """The animatic was sold as the cheap preflight for every Veo tier, but
    omni bills $0.10136/s against Veo Fast's $0.10/s: 3 x 8s beats cost $2.43
    as an animatic and $2.40 as the real render. The step is still worth
    recommending — it catches a bad creative call before the delivery render —
    but it may not claim a saving the numbers contradict."""
    plan = plan_generation(
        "a 3 beat reel about coffee",
        RoutingConstraints(
            budget=cast(Any, budget),
            num_beats=3,
            duration_seconds=8,
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
    assert OMNI_MODEL in rationale


@pytest.mark.parametrize("backend", ["vertex", "unknown", "gemini_api"])
def test_veo_lite_is_rejected_when_the_server_has_no_gemini_api_key(
    backend: str,
) -> None:
    """Lite is Gemini-API-only, so with no key the call fails with exactly that
    message. The planner knew the rule and recommended Lite anyway because
    nothing ever told it the credential state."""
    plan = plan_generation(
        "a video of a cat",
        RoutingConstraints(backend=cast(Any, backend), gemini_api_key_available=False),
    )
    assert VEO_LITE not in _models(plan)
    reason = _reason_for(plan, VEO_LITE)
    assert "GEMINI_API_KEY" in reason
    # The models the deployment can actually run are still planned.
    assert plan.is_satisfiable


@pytest.mark.parametrize("key_state", [None, True])
def test_veo_lite_survives_when_a_key_is_available_or_unknown(
    key_state: bool | None,
) -> None:
    """None means "not stated", which must keep today's behaviour."""
    plan = plan_generation(
        "a video of a cat",
        RoutingConstraints(backend="gemini_api", gemini_api_key_available=key_state),
    )
    assert VEO_LITE in _models(plan)


def test_pinned_veo_lite_without_a_key_is_a_conflict_that_names_the_key() -> None:
    plan = plan_generation(
        "a video of a cat",
        RoutingConstraints(gemini_api_key_available=False, pinned_model=VEO_LITE),
    )
    conflicts = {conflict.code: conflict for conflict in plan.conflicts}
    assert "pinned_model_backend_unsupported" in conflicts
    conflict = conflicts["pinned_model_backend_unsupported"]
    assert "GEMINI_API_KEY" in conflict.detail
    assert "GEMINI_API_KEY" in conflict.resolution


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
