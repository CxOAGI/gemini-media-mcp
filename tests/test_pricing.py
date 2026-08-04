"""Tests for pricing.py cost estimation and accounting."""

from typing import Any

import pytest

from src.image import _RETIRED_MODELS, _SUNSET_MODELS
from src.omni import OMNI_MODEL
from src.pricing import (
    _IMAGE_PRICING,
    _VIDEO_PRICING,
    PRICING_AS_OF,
    PRICING_SOURCES,
    CostEstimate,
    actual_image_cost,
    actual_video_cost,
    cost_to_dict,
    describe_model_pricing,
    estimate_image_cost,
    estimate_video_cost,
    format_cost,
    format_cost_line,
    is_priced,
    known_models,
    priced_models,
    pricing_coverage,
    resolve_model_id,
    snap_video_duration,
    sum_costs,
    unpriced_models,
)

# ============================================================================
# Test Doubles
# ============================================================================


class FakeModalityTokenCount:
    """Test double for google.genai types.ModalityTokenCount."""

    def __init__(self, modality: Any, token_count: int) -> None:
        self.modality = modality
        self.token_count = token_count


class FakeMediaModality:
    """Test double for the MediaModality enum (has a .value like the real one)."""

    def __init__(self, value: str) -> None:
        self.value = value


class FakeUsageMetadata:
    """Test double for GenerateContentResponseUsageMetadata.

    Only the attributes a real response carries are set, so tests can also
    exercise the "field missing entirely" path by omitting arguments.
    """

    def __init__(
        self,
        prompt_token_count: int | None = None,
        candidates_token_count: int | None = None,
        total_token_count: int | None = None,
        thoughts_token_count: int | None = None,
        prompt_tokens_details: list[FakeModalityTokenCount] | None = None,
        candidates_tokens_details: list[FakeModalityTokenCount] | None = None,
    ) -> None:
        self.prompt_token_count = prompt_token_count
        self.candidates_token_count = candidates_token_count
        self.total_token_count = total_token_count
        self.thoughts_token_count = thoughts_token_count
        self.prompt_tokens_details = prompt_tokens_details
        self.candidates_tokens_details = candidates_tokens_details


class BareObject:
    """An object with no usage fields at all (worst-case duck typing)."""


# ============================================================================
# Table sanity
# ============================================================================


def test_pricing_as_of_is_an_iso_date() -> None:
    assert len(PRICING_AS_OF) == 10
    year, month, day = PRICING_AS_OF.split("-")
    assert year.isdigit() and month.isdigit() and day.isdigit()


def test_every_source_is_an_official_google_url() -> None:
    for url in PRICING_SOURCES.values():
        assert url.startswith("https://")
        assert "google.dev" in url or "cloud.google.com" in url

    for pricing in (*_IMAGE_PRICING.values(), *_VIDEO_PRICING.values()):
        assert pricing.source in PRICING_SOURCES.values()


def test_every_priced_model_has_positive_rates() -> None:
    for pricing in _IMAGE_PRICING.values():
        assert pricing.input_usd_per_mtok > 0
        assert pricing.output_image_usd_per_mtok > 0
        assert pricing.output_tokens_by_size
        assert all(t > 0 for t in pricing.output_tokens_by_size.values())

    for pricing in _VIDEO_PRICING.values():
        assert pricing.usd_per_second_by_resolution
        assert all(v > 0 for v in pricing.usd_per_second_by_resolution.values())


# ============================================================================
# Coverage
# ============================================================================


def test_every_model_the_server_accepts_is_priced() -> None:
    assert unpriced_models() == ()
    assert pricing_coverage()["fully_covered"] is True


def test_known_models_includes_current_retired_and_video_ids() -> None:
    models = known_models()
    assert "gemini-3.1-flash-image" in models
    assert "imagen-4.0-generate-001" in models
    assert "veo-3.1-generate-001" in models
    # Gemini-API Veo spelling, which generate_video reports back to callers.
    assert "veo-3.1-generate-preview" in models
    assert OMNI_MODEL in models


def test_priced_models_are_the_canonical_ids_only() -> None:
    assert priced_models() == (
        "gemini-3-pro-image",
        "gemini-3.1-flash-image",
        "gemini-3.1-flash-lite-image",
        OMNI_MODEL,
        "veo-3.1-fast-generate-001",
        "veo-3.1-generate-001",
        "veo-3.1-lite-generate-preview",
    )


def test_unpriced_models_reports_unknown_ids() -> None:
    assert unpriced_models(["totally-made-up-model"]) == ("totally-made-up-model",)
    assert is_priced("totally-made-up-model") is False


def test_pricing_coverage_reports_date_and_sources() -> None:
    coverage = pricing_coverage()
    assert coverage["pricing_as_of"] == PRICING_AS_OF
    assert coverage["sources"] == PRICING_SOURCES
    assert coverage["tier"] == "standard-paid"


# ============================================================================
# estimate_image_cost
# ============================================================================


@pytest.mark.parametrize("model", list(_IMAGE_PRICING))
def test_every_image_model_returns_a_positive_estimate(model: str) -> None:
    estimate = estimate_image_cost(model)
    assert estimate is not None
    assert estimate.usd > 0
    assert estimate.is_estimate is True
    assert estimate.unit == "image"
    assert model in estimate.detail


def test_image_estimate_matches_googles_published_per_image_figures() -> None:
    # From the Gemini API pricing footnotes: 1120 tokens @ $60/1M for 1K,
    # 1680 @ $60/1M for 2K, 2520 @ $60/1M for 4K.
    flash = "gemini-3.1-flash-image"
    for size, expected in (("1K", 0.0672), ("2K", 0.1008), ("4K", 0.1512)):
        estimate = estimate_image_cost(flash, size)
        assert estimate is not None
        assert estimate.usd == pytest.approx(expected)

    lite = estimate_image_cost("gemini-3.1-flash-lite-image", "1K")
    assert lite is not None
    assert lite.usd == pytest.approx(0.0336)

    pro_1k = estimate_image_cost("gemini-3-pro-image", "1K")
    pro_2k = estimate_image_cost("gemini-3-pro-image", "2K")
    pro_4k = estimate_image_cost("gemini-3-pro-image", "4K")
    assert pro_1k is not None and pro_2k is not None and pro_4k is not None
    assert pro_1k.usd == pytest.approx(0.1344)
    # 1K and 2K bill identically on Pro.
    assert pro_2k.usd == pytest.approx(pro_1k.usd)
    assert pro_4k.usd == pytest.approx(0.24)


def test_image_resolution_tiers_are_monotonic() -> None:
    sizes = ["512", "1K", "2K", "4K"]
    costs: list[float] = []
    for size in sizes:
        estimate = estimate_image_cost("gemini-3.1-flash-image", size)
        assert estimate is not None
        costs.append(estimate.usd)
    assert costs == sorted(costs)
    assert costs[0] < costs[-1]


def test_image_size_is_case_insensitive_and_aliased() -> None:
    canonical = estimate_image_cost("gemini-3.1-flash-image", "2K")
    lowercase = estimate_image_cost("gemini-3.1-flash-image", "2k")
    assert canonical is not None and lowercase is not None
    assert canonical.usd == lowercase.usd

    half_k = estimate_image_cost("gemini-3.1-flash-image", "0.5K")
    explicit = estimate_image_cost("gemini-3.1-flash-image", "512")
    assert half_k is not None and explicit is not None
    assert half_k.usd == explicit.usd


def test_multi_image_scaling_is_linear() -> None:
    one = estimate_image_cost("gemini-3.1-flash-image", "1K", 1)
    five = estimate_image_cost("gemini-3.1-flash-image", "1K", 5)
    assert one is not None and five is not None
    assert five.usd == pytest.approx(one.usd * 5)
    assert five.breakdown["images"] == 5.0
    assert "5 images" in five.detail
    assert "1 image " in one.detail


def test_zero_images_costs_nothing() -> None:
    estimate = estimate_image_cost("gemini-3.1-flash-image", "1K", 0)
    assert estimate is not None
    assert estimate.usd == 0.0


def test_negative_image_count_is_rejected() -> None:
    assert estimate_image_cost("gemini-3.1-flash-image", "1K", -1) is None


def test_reference_images_and_prompt_tokens_add_input_cost() -> None:
    plain = estimate_image_cost("gemini-3.1-flash-image", "1K")
    with_refs = estimate_image_cost(
        "gemini-3.1-flash-image", "1K", reference_images=3, input_text_tokens=200
    )
    assert plain is not None and with_refs is not None
    # 3 * 1120 + 200 tokens @ $0.50/1M.
    assert with_refs.usd == pytest.approx(plain.usd + (3 * 1120 + 200) * 0.5 / 1e6)
    assert with_refs.breakdown["input_tokens"] == pytest.approx(3560.0)
    assert "reference image" in with_refs.detail


def test_unsupported_size_prices_at_the_model_default() -> None:
    # gemini-3.1-flash-lite-image is 1K-only; src/image.py drops the request
    # and the model renders its default, which is what gets billed.
    at_4k = estimate_image_cost("gemini-3.1-flash-lite-image", "4K")
    at_1k = estimate_image_cost("gemini-3.1-flash-lite-image", "1K")
    assert at_4k is not None and at_1k is not None
    assert at_4k.usd == at_1k.usd
    assert "unsupported" in at_4k.detail


def test_unknown_image_size_returns_none() -> None:
    assert estimate_image_cost("gemini-3.1-flash-image", "8K") is None
    assert estimate_image_cost("gemini-3.1-flash-image", "huge") is None


def test_unknown_image_model_returns_none_and_never_raises() -> None:
    assert estimate_image_cost("dall-e-9") is None
    assert estimate_image_cost("") is None
    assert actual_image_cost("dall-e-9", FakeUsageMetadata(1, 2)) is None


# ============================================================================
# Superseded model IDs
# ============================================================================


@pytest.mark.parametrize("retired,replacement", sorted(_RETIRED_MODELS.items()))
def test_retired_ids_price_as_their_replacement(
    retired: str, replacement: tuple[str, str]
) -> None:
    target = replacement[0]
    assert resolve_model_id(retired) == target
    superseded = estimate_image_cost(retired, "1K")
    current = estimate_image_cost(target, "1K")
    assert superseded is not None and current is not None
    assert superseded.usd == current.usd
    # The detail names the model that actually runs, not the dead one.
    assert target in superseded.detail


@pytest.mark.parametrize("sunset,replacement", sorted(_SUNSET_MODELS.items()))
def test_sunset_ids_price_as_their_replacement(
    sunset: str, replacement: tuple[str, str]
) -> None:
    target = replacement[0]
    superseded = estimate_image_cost(sunset, "2K")
    current = estimate_image_cost(target, "2K")
    assert superseded is not None and current is not None
    assert superseded.usd == current.usd


def test_unlisted_imagen_id_falls_back_like_src_image_does() -> None:
    # src/image.py reroutes any imagen-* ID it does not recognize.
    assert resolve_model_id("imagen-5.0-hypothetical-001") == "gemini-3.1-flash-image"
    estimate = estimate_image_cost("imagen-5.0-hypothetical-001")
    reference = estimate_image_cost("gemini-3.1-flash-image")
    assert estimate is not None and reference is not None
    assert estimate.usd == reference.usd


def test_gemini_api_veo_spelling_prices_like_the_vertex_spelling() -> None:
    assert resolve_model_id("veo-3.1-generate-preview") == "veo-3.1-generate-001"
    preview = estimate_video_cost("veo-3.1-generate-preview", 8)
    vertex = estimate_video_cost("veo-3.1-generate-001", 8)
    assert preview is not None and vertex is not None
    assert preview.usd == vertex.usd


# ============================================================================
# estimate_video_cost
# ============================================================================


@pytest.mark.parametrize("model", list(_VIDEO_PRICING))
def test_every_video_model_returns_a_positive_estimate(model: str) -> None:
    estimate = estimate_video_cost(model, 8)
    assert estimate is not None
    assert estimate.usd > 0
    assert estimate.is_estimate is True
    assert estimate.unit == "second-of-video"


def test_video_estimate_matches_published_per_second_rates() -> None:
    cases = [
        ("veo-3.1-generate-001", "720p", 0.40),
        ("veo-3.1-generate-001", "1080p", 0.40),
        ("veo-3.1-generate-001", "4K", 0.60),
        ("veo-3.1-fast-generate-001", "720p", 0.10),
        ("veo-3.1-fast-generate-001", "1080p", 0.12),
        ("veo-3.1-fast-generate-001", "4K", 0.30),
        ("veo-3.1-lite-generate-preview", "720p", 0.05),
        ("veo-3.1-lite-generate-preview", "1080p", 0.08),
    ]
    for model, resolution, per_second in cases:
        estimate = estimate_video_cost(model, 8, resolution)
        assert estimate is not None, f"{model} @ {resolution}"
        assert estimate.usd == pytest.approx(8 * per_second)
        assert estimate.breakdown["usd_per_second"] == pytest.approx(per_second)


def test_omni_per_second_derives_from_its_token_rate() -> None:
    # 5,792 tokens per second of 720p video @ $17.50 per 1M tokens.
    estimate = estimate_video_cost(OMNI_MODEL, 6)
    assert estimate is not None
    assert estimate.usd == pytest.approx(6 * 5792 * 17.50 / 1e6)
    assert estimate.breakdown["usd_per_second"] == pytest.approx(0.10136)


def test_video_resolution_tiers_increase_with_resolution() -> None:
    fast_720 = estimate_video_cost("veo-3.1-fast-generate-001", 8, "720p")
    fast_1080 = estimate_video_cost("veo-3.1-fast-generate-001", 8, "1080p")
    fast_4k = estimate_video_cost("veo-3.1-fast-generate-001", 8, "4K")
    assert fast_720 is not None and fast_1080 is not None and fast_4k is not None
    assert fast_720.usd < fast_1080.usd < fast_4k.usd


def test_resolution_spelling_is_normalized() -> None:
    upper = estimate_video_cost("veo-3.1-generate-001", 8, "4K")
    lower = estimate_video_cost("veo-3.1-generate-001", 8, "4k")
    assert upper is not None and lower is not None
    assert upper.usd == lower.usd


def test_lite_has_no_published_4k_rate_so_returns_none() -> None:
    # src/video.py rejects 4K on Lite outright; there is no price to quote.
    assert estimate_video_cost("veo-3.1-lite-generate-preview", 8, "4K") is None


def test_unknown_resolution_returns_none() -> None:
    assert estimate_video_cost("veo-3.1-generate-001", 8, "480p") is None


def test_unknown_video_model_returns_none() -> None:
    assert estimate_video_cost("sora-99", 8) is None
    assert actual_video_cost("sora-99", 8) is None


def test_audio_flag_does_not_change_the_price_but_is_explained() -> None:
    with_audio = estimate_video_cost("veo-3.1-generate-001", 8, include_audio=True)
    without_audio = estimate_video_cost("veo-3.1-generate-001", 8, include_audio=False)
    assert with_audio is not None and without_audio is not None
    # Google publishes one Veo 3.1 rate that already bundles audio.
    assert with_audio.usd == without_audio.usd
    assert "no discount" in without_audio.detail
    assert "no discount" not in with_audio.detail


def test_omni_prices_a_non_720p_request_at_what_it_actually_renders() -> None:
    at_1080 = estimate_video_cost(OMNI_MODEL, 6, "1080p")
    at_720 = estimate_video_cost(OMNI_MODEL, 6, "720p")
    assert at_1080 is not None and at_720 is not None
    assert at_1080.usd == at_720.usd
    assert "720p" in at_1080.detail


# ============================================================================
# Duration snapping
# ============================================================================


@pytest.mark.parametrize(
    "requested,expected",
    [
        (0, 4),
        (3, 4),
        (4, 4),
        (5, 4),
        (5.4, 6),
        (5.6, 6),
        (6, 6),
        (7, 6),
        (8, 8),
        (30, 8),
    ],
)
def test_veo_durations_snap_to_the_allowed_set(requested: float, expected: int) -> None:
    # Mirrors src/video.py exactly, ties included (5s snaps down to 4s).
    assert snap_video_duration("veo-3.1-generate-001", requested) == expected
    estimate = estimate_video_cost("veo-3.1-generate-001", requested)
    assert estimate is not None
    assert estimate.breakdown["seconds"] == float(expected)
    assert estimate.usd == pytest.approx(expected * 0.40)


def test_reference_and_extend_modes_use_their_forced_durations() -> None:
    assert snap_video_duration("veo-3.1-generate-001", 4, "reference_to_video") == 8
    assert snap_video_duration("veo-3.1-generate-001", 4, "extend_video") == 7

    extend = estimate_video_cost(
        "veo-3.1-generate-001", 4, generation_mode="extend_video"
    )
    assert extend is not None
    assert extend.breakdown["seconds"] == 7.0
    assert extend.usd == pytest.approx(7 * 0.40)


@pytest.mark.parametrize(
    "requested,expected", [(0, 3), (2, 3), (3, 3), (6.4, 6), (10, 10), (99, 10)]
)
def test_omni_durations_clamp_to_the_documented_range(
    requested: float, expected: int
) -> None:
    assert snap_video_duration(OMNI_MODEL, requested) == expected


# ============================================================================
# actual_image_cost
# ============================================================================


def test_actual_image_cost_with_modality_breakdown_is_exact() -> None:
    usage = FakeUsageMetadata(
        prompt_token_count=1500,
        candidates_token_count=1180,
        total_token_count=2680,
        candidates_tokens_details=[
            FakeModalityTokenCount(FakeMediaModality("IMAGE"), 1120),
            FakeModalityTokenCount(FakeMediaModality("TEXT"), 60),
        ],
    )
    cost = actual_image_cost("gemini-3.1-flash-image", usage, "1K", 1)
    assert cost is not None
    assert cost.is_estimate is False
    assert cost.unit == "token"
    expected = (1500 * 0.5 + 1120 * 60 + 60 * 3) / 1e6
    assert cost.usd == pytest.approx(expected)
    assert cost.breakdown["output_image_tokens"] == 1120.0
    assert cost.breakdown["output_text_tokens"] == 60.0


def test_actual_image_cost_accepts_a_plain_dict() -> None:
    usage: dict[str, Any] = {
        "prompt_token_count": 1500,
        "candidates_token_count": 1180,
        "candidates_tokens_details": [
            {"modality": "IMAGE", "token_count": 1120},
            {"modality": "TEXT", "token_count": 60},
        ],
    }
    from_dict = actual_image_cost("gemini-3.1-flash-image", usage, "1K", 1)
    from_object = actual_image_cost(
        "gemini-3.1-flash-image",
        FakeUsageMetadata(
            prompt_token_count=1500,
            candidates_token_count=1180,
            candidates_tokens_details=[
                FakeModalityTokenCount("IMAGE", 1120),
                FakeModalityTokenCount("TEXT", 60),
            ],
        ),
        "1K",
        1,
    )
    assert from_dict is not None and from_object is not None
    assert from_dict.usd == pytest.approx(from_object.usd)


def test_actual_image_cost_accepts_camel_case_rest_keys() -> None:
    usage = {
        "promptTokenCount": 1500,
        "candidatesTokenCount": 1180,
        "candidatesTokensDetails": [{"modality": "IMAGE", "tokenCount": 1120}],
    }
    cost = actual_image_cost("gemini-3.1-flash-image", usage, "1K", 1)
    assert cost is not None
    assert cost.breakdown["input_tokens"] == 1500.0
    assert cost.breakdown["output_image_tokens"] == 1120.0


def test_actual_image_cost_splits_a_bare_total_using_the_documented_counts() -> None:
    # No modality detail: the documented 1120 image tokens are attributed at
    # the image rate and the remainder at the (much cheaper) text rate.
    usage = FakeUsageMetadata(prompt_token_count=100, candidates_token_count=1200)
    cost = actual_image_cost("gemini-3.1-flash-image", usage, "1K", 1)
    assert cost is not None
    assert cost.is_estimate is False
    expected = (100 * 0.5 + 1120 * 60 + 80 * 3) / 1e6
    assert cost.usd == pytest.approx(expected)


def test_actual_image_cost_never_over_attributes_image_tokens() -> None:
    # Fewer output tokens than a full image: everything is image, none text.
    usage = FakeUsageMetadata(prompt_token_count=0, candidates_token_count=500)
    cost = actual_image_cost("gemini-3.1-flash-image", usage, "1K", 1)
    assert cost is not None
    assert cost.breakdown["output_image_tokens"] == 500.0
    assert "output_text_tokens" not in cost.breakdown


def test_actual_image_cost_bills_thinking_tokens_at_the_text_rate() -> None:
    usage = FakeUsageMetadata(
        prompt_token_count=0,
        candidates_token_count=1120,
        thoughts_token_count=800,
        candidates_tokens_details=[FakeModalityTokenCount("IMAGE", 1120)],
    )
    cost = actual_image_cost("gemini-3-pro-image", usage, "1K", 1)
    assert cost is not None
    expected = (1120 * 120 + 800 * 12) / 1e6
    assert cost.usd == pytest.approx(expected)


def test_actual_image_cost_falls_back_to_unit_pricing_without_usage() -> None:
    for usage in (None, BareObject(), {}, FakeUsageMetadata()):
        cost = actual_image_cost("gemini-3.1-flash-image", usage, "2K", 2)
        assert cost is not None
        assert cost.is_estimate is True
        assert cost.unit == "image"
        assert cost.usd == pytest.approx(2 * 0.1008)


def test_actual_image_cost_tolerates_none_and_garbage_fields() -> None:
    usage = FakeUsageMetadata(
        prompt_token_count=None,
        candidates_token_count=1120,
        candidates_tokens_details=[
            FakeModalityTokenCount(None, 5),
            FakeModalityTokenCount("IMAGE", None),  # type: ignore[arg-type]
            FakeModalityTokenCount("UNRECOGNIZED_MODALITY", 42),
        ],
    )
    cost = actual_image_cost("gemini-3.1-flash-image", usage, "1K", 1)
    assert cost is not None
    # No usable modality rows, so it fell back to the reported total.
    assert cost.breakdown["input_tokens"] == 0.0
    assert cost.breakdown["output_image_tokens"] == 1120.0


def test_actual_image_cost_ignores_nonsense_token_counts() -> None:
    usage = {"prompt_token_count": float("nan"), "candidates_token_count": -5}
    cost = actual_image_cost("gemini-3.1-flash-image", usage, "1K", 1)
    assert cost is not None
    # Both fields are unusable, so this degrades to the unit-priced fallback.
    assert cost.is_estimate is True
    assert cost.usd == pytest.approx(0.0672)


def test_actual_image_cost_resolves_superseded_ids() -> None:
    usage = FakeUsageMetadata(prompt_token_count=0, candidates_token_count=1120)
    superseded = actual_image_cost("gemini-2.5-flash-image", usage, "1K", 1)
    current = actual_image_cost("gemini-3.1-flash-image", usage, "1K", 1)
    assert superseded is not None and current is not None
    assert superseded.usd == current.usd


# ============================================================================
# actual_video_cost
# ============================================================================


def test_actual_video_cost_uses_the_reported_duration_verbatim() -> None:
    # 7s is a real extend_video output; re-snapping it would corrupt the cost.
    cost = actual_video_cost("veo-3.1-generate-001", 7, "1080p")
    assert cost is not None
    assert cost.is_estimate is False
    assert cost.breakdown["seconds"] == 7.0
    assert cost.usd == pytest.approx(7 * 0.40)


def test_actual_video_cost_can_snap_a_requested_duration_on_demand() -> None:
    cost = actual_video_cost("veo-3.1-generate-001", 5, snap_duration=True)
    assert cost is not None
    assert cost.breakdown["seconds"] == 4.0


def test_actual_video_cost_prefers_reported_video_tokens_for_omni() -> None:
    usage = FakeUsageMetadata(
        prompt_token_count=2000,
        candidates_token_count=34752,
        candidates_tokens_details=[
            FakeModalityTokenCount(FakeMediaModality("VIDEO"), 34752),
        ],
    )
    cost = actual_video_cost(OMNI_MODEL, 6, usage_metadata=usage)
    assert cost is not None
    assert cost.is_estimate is False
    assert cost.unit == "token"
    expected = (2000 * 1.50 + 34752 * 17.50) / 1e6
    assert cost.usd == pytest.approx(expected)


def test_actual_video_cost_ignores_video_tokens_for_veo_which_bills_per_second() -> (
    None
):
    usage = {"candidates_tokens_details": [{"modality": "VIDEO", "token_count": 99999}]}
    cost = actual_video_cost("veo-3.1-generate-001", 8, usage_metadata=usage)
    assert cost is not None
    assert cost.unit == "second-of-video"
    assert cost.usd == pytest.approx(8 * 0.40)


def test_actual_video_cost_without_duration_or_usage_returns_none() -> None:
    assert actual_video_cost("veo-3.1-generate-001") is None
    assert actual_video_cost("veo-3.1-generate-001", None, usage_metadata=None) is None
    assert actual_video_cost(OMNI_MODEL, None, usage_metadata=BareObject()) is None


def test_actual_video_cost_returns_none_for_an_unpriced_resolution() -> None:
    assert actual_video_cost("veo-3.1-lite-generate-preview", 8, "4K") is None


# ============================================================================
# sum_costs
# ============================================================================


def test_sum_costs_totals_a_multi_beat_clip() -> None:
    beats = [
        estimate_video_cost("veo-3.1-generate-001", 8, "720p"),
        estimate_video_cost("veo-3.1-generate-001", 6, "720p"),
        estimate_video_cost("veo-3.1-fast-generate-001", 4, "720p"),
    ]
    total = sum_costs(beats, label="reel")
    assert total.usd == pytest.approx(8 * 0.40 + 6 * 0.40 + 4 * 0.10)
    assert total.unit == "second-of-video"
    assert total.is_estimate is True
    assert total.breakdown["seconds"] == 18.0
    assert total.breakdown["components"] == 3.0
    assert "reel" in total.detail


def test_sum_costs_does_not_accumulate_rounding_error() -> None:
    beats = [estimate_image_cost("gemini-3.1-flash-image", "1K") for _ in range(1000)]
    total = sum_costs(beats)
    assert total.usd == pytest.approx(1000 * 0.0672, rel=1e-12)


def test_sum_costs_marks_units_mixed_when_components_differ() -> None:
    total = sum_costs(
        [
            estimate_image_cost("gemini-3.1-flash-image", "1K"),
            estimate_video_cost("veo-3.1-generate-001", 8),
        ]
    )
    assert total.unit == "mixed"


def test_sum_costs_flags_unpriced_components_rather_than_hiding_them() -> None:
    total = sum_costs(
        [
            estimate_video_cost("veo-3.1-generate-001", 8),
            estimate_video_cost("sora-9", 8),
        ]
    )
    assert total.usd == pytest.approx(8 * 0.40)
    assert total.is_estimate is True
    assert total.breakdown["unpriced_components"] == 1.0
    assert "lower bound" in total.detail


def test_sum_costs_of_nothing_is_zero() -> None:
    total = sum_costs([])
    assert total.usd == 0.0
    assert total.breakdown["components"] == 0.0


def test_sum_costs_of_actual_costs_stays_actual() -> None:
    total = sum_costs(
        [
            actual_video_cost("veo-3.1-generate-001", 8),
            actual_video_cost("veo-3.1-generate-001", 6),
        ]
    )
    assert total.is_estimate is False


# ============================================================================
# format_cost
# ============================================================================


@pytest.mark.parametrize(
    "usd,expected",
    [
        (0.0, "$0.00"),
        (12.5, "$12.50"),
        (1234.567, "$1,234.57"),
        (0.40, "$0.40"),
        (0.0672, "$0.0672"),
        # The case that must never render as "$0.00".
        (0.0043, "$0.0043"),
        (0.00056, "$0.00056"),
        (0.000001, "$0.000001"),
        (-0.0043, "-$0.0043"),
    ],
)
def test_format_cost_keeps_enough_precision(usd: float, expected: str) -> None:
    assert format_cost(usd) == expected


def test_format_cost_never_renders_a_real_charge_as_free() -> None:
    for usd in (0.004, 0.0001, 0.00001):
        assert format_cost(usd) != "$0.00"


def test_format_cost_accepts_a_cost_estimate_and_none() -> None:
    estimate = estimate_image_cost("gemini-3.1-flash-image", "1K")
    assert estimate is not None
    assert format_cost(estimate) == "$0.0672"
    assert format_cost(None) == "unpriced"
    assert format_cost(float("inf")) == "unpriced"


def test_format_cost_line_marks_estimates_with_a_tilde() -> None:
    estimate = estimate_video_cost("veo-3.1-generate-001", 8)
    actual = actual_video_cost("veo-3.1-generate-001", 8)
    assert estimate is not None and actual is not None
    assert format_cost_line(estimate).startswith("~$3.20")
    assert format_cost_line(actual).startswith("$3.20")
    assert format_cost_line(None) == "unpriced"


# ============================================================================
# Reporting helpers
# ============================================================================


def test_cost_to_dict_is_json_shaped_and_carries_the_as_of_date() -> None:
    estimate = estimate_image_cost("gemini-3.1-flash-image", "2K", 2)
    payload = cost_to_dict(estimate)
    assert payload is not None
    assert payload["usd"] == pytest.approx(0.2016)
    assert payload["usd_display"] == "$0.2016"
    assert payload["is_estimate"] is True
    assert payload["pricing_as_of"] == PRICING_AS_OF
    assert isinstance(payload["breakdown"], dict)
    assert cost_to_dict(None) is None


def test_describe_model_pricing_exposes_rates_and_source() -> None:
    image = describe_model_pricing("imagen-4.0-generate-001")
    assert image is not None
    assert image["kind"] == "image"
    # Reports the model that will really be billed, plus what was asked for.
    assert image["model"] == "gemini-3.1-flash-image"
    assert image["requested_model"] == "imagen-4.0-generate-001"
    assert image["usd_per_image"]["1K"] == pytest.approx(0.0672)
    assert image["source"] in PRICING_SOURCES.values()

    video = describe_model_pricing("veo-3.1-fast-generate-001")
    assert video is not None
    assert video["kind"] == "video"
    assert video["usd_per_second"]["1080p"] == pytest.approx(0.12)
    assert video["audio_included_in_price"] is True

    assert describe_model_pricing("nope") is None


def test_cost_estimate_is_immutable() -> None:
    estimate = estimate_image_cost("gemini-3.1-flash-image")
    assert isinstance(estimate, CostEstimate)
    with pytest.raises(Exception):
        estimate.usd = 1.0  # type: ignore[misc]


def test_module_is_deterministic() -> None:
    first = estimate_image_cost("gemini-3.1-flash-image", "4K", 3)
    second = estimate_image_cost("gemini-3.1-flash-image", "4K", 3)
    assert first == second
