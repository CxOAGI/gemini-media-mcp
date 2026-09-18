"""Embedded pricing and cost accounting for every model this server can call.

The point of this module is that a caller can ask "what will this cost?"
*before* spending money (``estimate_image_cost`` / ``estimate_video_cost``),
and "what did that cost?" *after* the call returns
(``actual_image_cost`` / ``actual_video_cost``), without the server ever
touching the network. Prices are embedded as constants, verified against
Google's published pricing pages on ``PRICING_AS_OF``.

Sourcing and accuracy
---------------------
Every number below comes from an official Google pricing page; the URL is
recorded on each pricing record's ``source`` field and collected in
``PRICING_SOURCES``. Nothing is inferred from third-party summaries. A model
with no confirmed price is simply absent from the tables, and every public
entry point returns ``None`` for it rather than guessing.

That provenance travels with the money: every ``CostEstimate`` carries the
``source`` URL it was priced from, plus a ``source_note`` for the rates whose
sourcing needs a sentence of explanation (see the Vertex assumption below).
``cost_to_dict`` publishes both, so a caller who is quoted a figure over MCP
can open the page it came from instead of taking the number on trust.

Documented assumptions (each one is a place where the sources are thinner
than we would like):

* **Vertex AI vs Gemini Developer API.** The image-model token rates are
  published identically on both the Gemini Developer API pricing page and the
  Vertex AI generative-AI pricing page, so one table covers both. The Veo
  per-second rates could only be confirmed on the Gemini Developer API
  pricing page — the Vertex page defers to a Veo section that is rendered
  client-side and is not retrievable as text. Veo prices are therefore
  assumed to be identical on Vertex AI. If that assumption ever breaks, it
  breaks in the direction of a wrong number, so treat Veo figures on Vertex
  as "best published rate" rather than a billing guarantee.
* **Tier.** Everything here is the *Standard* paid tier. Google's Batch tier
  is exactly half price for these models (``BATCH_PRICE_MULTIPLIER``), but
  this server never issues batch requests, so batch rates are not applied
  automatically.
* **Free tier.** There is none for any model in this module: the Gemini
  Developer API pricing page lists "Not available" in the free-tier column
  for all three image models, all three Veo 3.1 variants and both omni
  models (``gemini-omni-flash-preview`` and ``gemini-omni-1.1-flash``). So a
  paid-tier price is always the right price here, and there is no free-tier
  branch to model.
* **Audio.** Veo 3.1 publishes a single per-second rate that already
  includes natively generated audio; Google publishes no audio-free
  discount. ``include_audio=False`` therefore does not reduce the estimate —
  it only changes the wording of ``CostEstimate.detail``, so a caller can see
  that turning audio off saves nothing.
* **Input tokens.** A pre-flight estimate cannot know how long the prompt
  will be, so ``estimate_image_cost`` prices output only unless the caller
  passes ``reference_images``/``input_text_tokens``. The post-hoc
  ``actual_image_cost`` prices reported input tokens exactly.

Money handling
--------------
All arithmetic is done in full float precision and rounding happens only in
``format_cost``, at the presentation boundary. Summing many small beats
(``sum_costs``) therefore does not accumulate rounding error.

Superseded model IDs
--------------------
``src.image`` reroutes retired/sunset image IDs to a live replacement before
calling the API, which means the *replacement* is what actually gets billed.
This module imports that same table (``_MODEL_SHUTDOWNS``) instead of
duplicating it, so a price quoted for
``imagen-4.0-generate-001`` is the price of the model that really runs. The
same applies to the Veo ID translation in ``src.video``: the Gemini Developer
API serves Veo under ``-preview`` IDs, and ``generate_video`` reports the
translated ID back to the caller, so both spellings resolve to one price.
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, get_args

from .image import (
    _MODEL_SHUTDOWNS,
    _SUPERSEDED_DEFAULT_TARGET,
    ImageModel,
    RetiredImageModel,
)
from .omni import (
    OMNI_1_1_MODEL,
    OMNI_MODEL,
    OMNI_MODEL_ALIASES,
    canonical_omni_model,
    is_omni_model,
)
from .video import _GEMINI_API_MODEL_IDS, VideoModel

# Date on which every price in this module was verified against the sources
# below. Bump it (and re-check the tables) whenever these are refreshed —
# callers surface it so a stale figure is visible rather than silent.
#
# Re-read in full on 2026-08-28, the day after gemini-omni-1.1-flash went GA
# and republished the video table: all three image rows, all three Veo tiers
# and both omni rows still carry the figures below. Anything automated reads
# THIS field, so a per-record note claiming a later verification than this
# date is a contradiction, not a refinement — bump here or not at all.
PRICING_AS_OF = "2026-08-28"

PRICING_SOURCES: dict[str, str] = {
    "gemini_api": "https://ai.google.dev/gemini-api/docs/pricing",
    "vertex_ai": "https://cloud.google.com/vertex-ai/generative-ai/pricing",
    # Not a pricing page, but the only Google source that states how Omni
    # 1.1's 360p draft tier is charged relative to 720p. Cited on the rate it
    # backs, so a caller can see exactly which figure rests on it.
    "omni_1_1_launch": (
        "https://blog.google/innovation-and-ai/technology/developers-tools/"
        "build-with-gemini-omni-1-1-flash/"
    ),
}

# Every embedded rate below was read from the Gemini Developer API pricing
# page; the image rates were additionally cross-checked against the Vertex AI
# page (PRICING_SOURCES["vertex_ai"]), which publishes the same numbers.
_SRC_GEMINI_API = PRICING_SOURCES["gemini_api"]
_SRC_VERTEX_AI = PRICING_SOURCES["vertex_ai"]
_SRC_OMNI_1_1_LAUNCH = PRICING_SOURCES["omni_1_1_launch"]

# Per-rate provenance notes. ``source`` answers "where did this number come
# from?"; these answer the follow-up question a caller on the *other* backend
# has to ask — "does it hold for me too?" — using only what the pages
# themselves say. A rate whose source page tells the whole story carries no
# note, so a note always means "read this before you trust the figure".
_IMAGE_SOURCE_NOTE = (
    "The same token rates are published on the Vertex AI pricing page "
    f"({_SRC_VERTEX_AI}), so this rate applies on either backend."
)
_VEO_SOURCE_NOTE = (
    "Confirmed on the Gemini Developer API pricing page. The Vertex AI page "
    "renders its Veo section client-side and publishes no retrievable table, "
    "so the same rate is assumed — not verified — for Veo on Vertex AI."
)

# Google's Batch tier is half the Standard tier for every model priced here.
# Exposed for callers that batch on their own; this server does not.
BATCH_PRICE_MULTIPLIER = 0.5

# Tokens are quoted per million everywhere in Google's tables.
_TOKENS_PER_MILLION = 1_000_000.0


@dataclass(frozen=True)
class CostEstimate:
    """A priced media operation.

    Attributes:
        usd: Total cost in US dollars, unrounded.
        is_estimate: True for a pre-flight estimate, False when the figure was
            derived from usage the API actually reported (token counts, or the
            effective duration of a rendered clip).
        unit: The billing unit the figure is built from — "image",
            "second-of-video", "token", or "mixed" for an aggregate.
        detail: Human-readable one-liner, e.g.
            "2 images @ 2K on gemini-3.1-flash-image".
        breakdown: Component values. Keys ending in ``_usd`` are dollars, keys
            ending in ``_tokens`` are token counts, and ``seconds`` /
            ``images`` are plain quantities. Never rounded.
        source: URL of the pricing page the rate behind ``usd`` was read from.
            ``None`` means "no single honest answer": an aggregate whose
            components were priced from different pages, or a cost assembled
            outside this module. Never guessed — an unpriced model yields
            ``None`` for the whole estimate, not a sourceless one.
        source_note: One-line qualifier a reader needs before checking the
            figure against ``source`` (e.g. that the other backend's rate is
            assumed rather than confirmed). ``None`` when the source page
            stands on its own, or when components disagree.
    """

    usd: float
    is_estimate: bool
    unit: str
    detail: str
    breakdown: dict[str, float]
    # Defaulted so callers that build a CostEstimate positionally or by
    # keyword (src.routing assembles multi-render plan quotes itself) keep
    # working; they simply report no provenance rather than a borrowed one.
    source: str | None = None
    source_note: str | None = None


@dataclass(frozen=True)
class ImageModelPricing:
    """Token rates and per-image token counts for one image model.

    Google bills image models purely per token; the familiar "$0.067 per 1K
    image" figures are derived from a fixed output-token count per resolution.
    Storing the token counts rather than the rounded dollar figures keeps this
    table exactly consistent with what the API meters.

    Attributes:
        input_usd_per_mtok: USD per 1M input tokens (text and image).
        output_text_usd_per_mtok: USD per 1M text/thinking output tokens.
        output_image_usd_per_mtok: USD per 1M image output tokens.
        tokens_per_input_image: Tokens charged for each input/reference image.
        output_tokens_by_size: Output image tokens per supported size.
        default_size: Size the model falls back to when asked for one it
            cannot produce (``src.image`` warns and sends at the default).
        source: URL the numbers were read from.
        source_note: Optional one-liner about how far ``source`` can be
            trusted for a caller on the other backend; travels onto every
            ``CostEstimate`` priced from this record.
    """

    input_usd_per_mtok: float
    output_text_usd_per_mtok: float
    output_image_usd_per_mtok: float
    tokens_per_input_image: int
    output_tokens_by_size: Mapping[str, int]
    default_size: str
    source: str
    source_note: str | None = None

    def usd_per_image(self, image_size: str) -> float | None:
        """Return the output cost of one image at ``image_size``, or None."""
        tokens = self.output_tokens_by_size.get(image_size)
        if tokens is None:
            return None
        return tokens * self.output_image_usd_per_mtok / _TOKENS_PER_MILLION


@dataclass(frozen=True)
class VideoModelPricing:
    """Per-second (and, where published, per-token) rates for one video model.

    Attributes:
        usd_per_second_by_resolution: USD per second of output video, keyed by
            normalized resolution. A resolution the model cannot produce is
            absent rather than priced.
        audio_included: True when the published rate already covers natively
            generated audio and no audio-free rate exists.
        output_video_usd_per_mtok: USD per 1M video output tokens, when Google
            publishes token-level billing for the model (Omni does; Veo does
            not).
        tokens_per_second: Video output tokens metered per second, when
            published. Google publishes exactly one such figure per omni model
            and pins it to 720p ("5,792 tokens per second of 720p video"), so
            on a model with a real resolution parameter this is the 720p rate
            specifically, NOT a constant across resolutions — see
            ``tokens_per_second_by_resolution`` on the published record.
        output_text_usd_per_mtok: USD per 1M text output tokens, when the
            model can also emit text.
        input_usd_per_mtok: USD per 1M input tokens, when published.
        fixed_resolution: Set when the model only ever emits one resolution,
            so a differing request is priced at what is actually rendered.
        resolution_notes: Per-resolution caveats appended to the estimate's
            detail. A model whose published rate covers one resolution but
            whose API accepts four (Omni 1.1) has to say which of its four
            figures Google actually printed and which were derived.
        source: URL the numbers were read from.
        source_note: Optional one-liner about how far ``source`` can be
            trusted for a caller on the other backend; travels onto every
            ``CostEstimate`` priced from this record.
    """

    usd_per_second_by_resolution: Mapping[str, float]
    audio_included: bool
    source: str
    source_note: str | None = None
    output_video_usd_per_mtok: float | None = None
    tokens_per_second: int | None = None
    output_text_usd_per_mtok: float | None = None
    input_usd_per_mtok: float | None = None
    fixed_resolution: str | None = None
    resolution_notes: Mapping[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Image pricing
# ---------------------------------------------------------------------------
#
# Source: https://ai.google.dev/gemini-api/docs/pricing (image models) and the
# matching rows + footnotes on
# https://cloud.google.com/vertex-ai/generative-ai/pricing, verified
# 2026-08-03. Both pages quote the same token rates. The footnotes give the
# per-resolution output token counts reproduced below, e.g.: "Gemini 3.1 Flash
# Image charges 1120 tokens per input image, with output image costs scaling
# by resolution: 747 tokens ($0.045) for 512 ..., 1120 tokens ($0.067) for
# 1K ..., 1680 tokens ($0.101) for 2K ..., and 2,520 tokens ($0.15) for 4K".
_IMAGE_PRICING: dict[str, ImageModelPricing] = {
    "gemini-3.1-flash-image": ImageModelPricing(
        input_usd_per_mtok=0.50,
        output_text_usd_per_mtok=3.00,
        output_image_usd_per_mtok=60.00,
        tokens_per_input_image=1120,
        # 747/1120/1680/2520 tokens -> $0.0448/$0.0672/$0.1008/$0.1512.
        output_tokens_by_size={"512": 747, "1K": 1120, "2K": 1680, "4K": 2520},
        default_size="1K",
        source=_SRC_GEMINI_API,
        source_note=_IMAGE_SOURCE_NOTE,
    ),
    "gemini-3.1-flash-lite-image": ImageModelPricing(
        input_usd_per_mtok=0.25,
        output_text_usd_per_mtok=1.50,
        output_image_usd_per_mtok=30.00,
        tokens_per_input_image=1120,
        # 1K-only model (mirrors _IMAGE_SIZE_SUPPORT in src/image.py):
        # 1120 tokens -> $0.0336.
        output_tokens_by_size={"1K": 1120},
        default_size="1K",
        source=_SRC_GEMINI_API,
        source_note=_IMAGE_SOURCE_NOTE,
    ),
    "gemini-3-pro-image": ImageModelPricing(
        input_usd_per_mtok=2.00,
        output_text_usd_per_mtok=12.00,
        output_image_usd_per_mtok=120.00,
        tokens_per_input_image=560,
        # 1K and 2K bill identically (1120 tokens -> $0.1344); 4K is 2000
        # tokens -> $0.24.
        output_tokens_by_size={"1K": 1120, "2K": 1120, "4K": 2000},
        default_size="1K",
        source=_SRC_GEMINI_API,
        source_note=_IMAGE_SOURCE_NOTE,
    ),
}

# ---------------------------------------------------------------------------
# Video pricing
# ---------------------------------------------------------------------------
#
# Source: https://ai.google.dev/gemini-api/docs/pricing (Veo 3.1 table),
# verified 2026-08-03: 720p/1080p/4K at $0.40/$0.40/$0.60 for Veo 3.1,
# $0.10/$0.12/$0.30 for Veo 3.1 Fast, and $0.05/$0.08/(unsupported) for
# Veo 3.1 Lite, all "include audio by default". The Vertex AI pricing page
# does not expose a retrievable Veo table, so these rates are assumed to hold
# on Vertex as well (see the module docstring).
_OMNI_VIDEO_TOKENS_PER_SECOND = 5792
_OMNI_VIDEO_USD_PER_MTOK = 17.50

# Both omni models sit on the same row shape of the pricing page: input $1.50
# / 1M tokens, text output $9.00, video output $17.50, and the footnote
# "Billing is based on total output token consumption, calculated at a rate of
# 5,792 tokens per second of 720p video."
_OMNI_INPUT_USD_PER_MTOK = 1.50
_OMNI_OUTPUT_TEXT_USD_PER_MTOK = 9.00

# gemini-omni-1.1-flash's 360p tier. The pricing page publishes ONE token rate
# and pins it to 720p; the only statement about 360p is in the launch post —
# "lightweight previews in 360p resolution up to 60% faster and at a third of
# the cost compared to Omni 1.1's standard 720p resolution". A third is
# therefore what is quoted, sourced to that post rather than to the pricing
# page, and said so on every estimate that uses it. The two sources agree in
# kind: the footnote's "per second of 720p video" only makes sense if the
# token count moves with resolution.
_OMNI_360P_COST_DIVISOR = 3.0


def _omni_usd_per_second(tokens_per_second: float) -> float:
    """USD per second of omni output video at a given token rate.

    Derived rather than hard-coded so the per-second and per-token paths can
    never disagree.
    """
    return tokens_per_second * _OMNI_VIDEO_USD_PER_MTOK / _TOKENS_PER_MILLION


# Why 1080p and 4K bill at the 720p rate here: both are documented as
# *upscaled* from the base render, and Google publishes no second token rate
# for them. Applying the one published figure is the only honest arithmetic
# available; the estimate says outright that an upscaled render may meter
# differently, so nobody reads the number as a confirmed tier.
# STANDING EXPOSURE, re-check whenever PRICING_AS_OF moves: if Google ever
# publishes a separate upscale rate, every 1080p and 4K omni quote is wrong by
# the delta, silently and in the under-quoting direction if the tier is dearer.
# There is no way to detect that from here — only the pricing page says.
_OMNI_1_1_UPSCALE_NOTE = (
    "1080p and 4K are upscaled from the base render and Google publishes no "
    "separate token rate for them, so this prices them at the published 720p "
    "rate ({tokens:,} tokens/s). Treat it as the best published figure, not a "
    "confirmed tier: if a separate upscale rate is ever published, this quote "
    "is wrong by the difference."
).format(tokens=_OMNI_VIDEO_TOKENS_PER_SECOND)

_OMNI_1_1_360P_NOTE = (
    "360p is priced at one third of the published 720p token rate. That ratio "
    "comes from Google's launch post ("
    + _SRC_OMNI_1_1_LAUNCH
    + "), not from the pricing page, which publishes only the 720p rate — so "
    "this figure is the vendor's stated ratio rather than a metered one. The "
    "real run bills the video output tokens the response reports."
)

_VIDEO_PRICING: dict[str, VideoModelPricing] = {
    "veo-3.1-generate-001": VideoModelPricing(
        usd_per_second_by_resolution={"720p": 0.40, "1080p": 0.40, "4K": 0.60},
        audio_included=True,
        source=_SRC_GEMINI_API,
        source_note=_VEO_SOURCE_NOTE,
    ),
    "veo-3.1-fast-generate-001": VideoModelPricing(
        usd_per_second_by_resolution={"720p": 0.10, "1080p": 0.12, "4K": 0.30},
        audio_included=True,
        source=_SRC_GEMINI_API,
        source_note=_VEO_SOURCE_NOTE,
    ),
    "veo-3.1-lite-generate-preview": VideoModelPricing(
        # 4K is deliberately absent: Lite cannot render it (src/video.py
        # rejects the combination) and Google publishes no 4K Lite rate.
        usd_per_second_by_resolution={"720p": 0.05, "1080p": 0.08},
        audio_included=True,
        source=_SRC_GEMINI_API,
        source_note=_VEO_SOURCE_NOTE,
    ),
    OMNI_MODEL: VideoModelPricing(
        # Omni is billed per output token: "5,792 tokens per second of 720p
        # video" at $17.50 per 1M video output tokens, i.e. ~$0.10/second.
        # Derived rather than hard-coded so the per-second and per-token
        # paths can never disagree.
        usd_per_second_by_resolution={
            "720p": _omni_usd_per_second(_OMNI_VIDEO_TOKENS_PER_SECOND)
        },
        audio_included=True,
        output_video_usd_per_mtok=_OMNI_VIDEO_USD_PER_MTOK,
        tokens_per_second=_OMNI_VIDEO_TOKENS_PER_SECOND,
        output_text_usd_per_mtok=_OMNI_OUTPUT_TEXT_USD_PER_MTOK,
        input_usd_per_mtok=_OMNI_INPUT_USD_PER_MTOK,
        # Omni always renders 720p (see src/omni.py), so any other request is
        # still billed at the 720p rate.
        fixed_resolution="720p",
        source=_SRC_GEMINI_API,
        # Stated explicitly rather than left null: the other rates carry a
        # cross-backend note, and a silent gap here reads as an oversight.
        # Neither pricing page says how omni's token rates apply on the other
        # backend, so no such claim is made.
        source_note=(
            "Rates confirmed on the Gemini Developer API pricing page. Neither "
            "page states how omni's token rates apply on the other backend, so "
            "no cross-backend equivalence is claimed."
        ),
    ),
    OMNI_1_1_MODEL: VideoModelPricing(
        # Same published row as the preview model — input $1.50, text $9.00,
        # video $17.50 per 1M tokens, 5,792 tokens per second of 720p — but
        # 1.1 has a real resolution parameter, so the table carries one rate
        # per resolution instead of a single fixed one. See the notes above
        # for which of these four figures Google actually printed.
        usd_per_second_by_resolution={
            "360p": _omni_usd_per_second(
                _OMNI_VIDEO_TOKENS_PER_SECOND / _OMNI_360P_COST_DIVISOR
            ),
            "720p": _omni_usd_per_second(_OMNI_VIDEO_TOKENS_PER_SECOND),
            "1080p": _omni_usd_per_second(_OMNI_VIDEO_TOKENS_PER_SECOND),
            "4K": _omni_usd_per_second(_OMNI_VIDEO_TOKENS_PER_SECOND),
        },
        audio_included=True,
        output_video_usd_per_mtok=_OMNI_VIDEO_USD_PER_MTOK,
        tokens_per_second=_OMNI_VIDEO_TOKENS_PER_SECOND,
        output_text_usd_per_mtok=_OMNI_OUTPUT_TEXT_USD_PER_MTOK,
        input_usd_per_mtok=_OMNI_INPUT_USD_PER_MTOK,
        # Deliberately unset: unlike the preview model, 1.1 renders what it is
        # asked for, so there is no substitution to disclose.
        fixed_resolution=None,
        resolution_notes={
            "360p": _OMNI_1_1_360P_NOTE,
            "1080p": _OMNI_1_1_UPSCALE_NOTE,
            "4K": _OMNI_1_1_UPSCALE_NOTE,
        },
        source=_SRC_GEMINI_API,
        source_note=(
            "Confirmed on the Gemini Developer API pricing page: "
            "gemini-omni-1.1-flash carries the same published token rates as "
            "the preview model ($1.50 input, $9.00 text output, $17.50 video "
            "output per 1M tokens, 5,792 tokens per second of 720p). Neither "
            "page states how omni's token rates apply on the other backend, "
            "so no cross-backend equivalence is claimed."
        ),
    ),
}

# Veo IDs are spelled differently per backend: the Gemini Developer API serves
# `-preview`, Vertex serves `-001`, and generate_video reports back whichever
# it used. Invert src/video.py's table so both spellings price identically
# instead of one of them looking like an unknown model.
_VIDEO_MODEL_ALIASES: dict[str, str] = {
    api_id: canonical for canonical, api_id in _GEMINI_API_MODEL_IDS.items()
}

# Omni splits the same way: Vertex serves 1.1 as `gemini-omni-1.1-flash-preview`
# and the Developer API as the bare name. Both spellings must price
# identically, or a render reported back by Vertex looks like an unknown model.
_VIDEO_MODEL_ALIASES.update(
    {
        alias: canonical_omni_model(alias)
        for alias in OMNI_MODEL_ALIASES
        if canonical_omni_model(alias) != alias
    }
)

# Durations Veo actually accepts, mirroring src/video.py. Anything else is
# snapped to the nearest of these, so an estimate matches the clip that is
# really billed.
_VEO_ALLOWED_DURATIONS = (4, 6, 8)
_VEO_REFERENCE_DURATION = 8  # reference_to_video is 8s only
_VEO_EXTEND_DURATION = 7  # extend_video outputs exactly 7s

# Omni clamps to [3, 10]s (src/omni.py) — both models.
_OMNI_MIN_DURATION = 3
_OMNI_MAX_DURATION = 10

# Image sizes with a published price somewhere in the table. A size outside
# this set is not "unpriced for this model", it is meaningless — so it yields
# None rather than a fallback.
_KNOWN_IMAGE_SIZES = frozenset({"512", "1K", "2K", "4K"})

# Spellings callers plausibly use for the 512px tier, plus the tolerated
# lowercase 'k' that the API itself rejects.
_IMAGE_SIZE_ALIASES: dict[str, str] = {
    "0.5K": "512",
    "512PX": "512",
    "512P": "512",
}

# Resolution spellings seen across src/video.py ("4K"), the Veo docs ("4k")
# and casual callers.
_RESOLUTION_ALIASES: dict[str, str] = {
    # 360p arrived with gemini-omni-1.1-flash's draft tier.
    "360P": "360p",
    "720P": "720p",
    "1080P": "1080p",
    "4K": "4K",
    "2160P": "4K",
}


# ---------------------------------------------------------------------------
# Normalization and model resolution
# ---------------------------------------------------------------------------


def resolve_model_id(model: str) -> str:
    """Return the model ID that will actually be billed for ``model``.

    ``src.image`` reroutes retired and sunset image IDs to a live replacement
    before issuing the call, and ``src.video`` translates Veo IDs per backend.
    Pricing has to follow those rewrites, otherwise a caller pinned to
    ``imagen-4.0-generate-001`` would be quoted nothing at all while their
    account is charged for ``gemini-3.1-flash-image``.

    Args:
        model: Any model ID the server accepts, current or superseded.

    Returns:
        The canonical ID whose price applies. Unrecognized IDs are returned
        unchanged so the caller-facing lookup can report them as unpriced.
    """
    model_id = str(model)

    # Same three-way test src/image.py uses, including the imagen-* catch-all
    # for regional or newly-surfaced variants that never made the table.
    if model_id in _MODEL_SHUTDOWNS:
        return _MODEL_SHUTDOWNS[model_id][0]
    if model_id.startswith("imagen"):
        return _SUPERSEDED_DEFAULT_TARGET

    return _VIDEO_MODEL_ALIASES.get(model_id, model_id)


def _normalize_image_size(image_size: str | None) -> str | None:
    """Normalize an image-size spelling to a table key, or None if unknown."""
    if image_size is None:
        return None
    key = str(image_size).strip().upper()
    key = _IMAGE_SIZE_ALIASES.get(key, key)
    return key if key in _KNOWN_IMAGE_SIZES else None


def _normalize_resolution(resolution: str | None) -> str | None:
    """Normalize a video-resolution spelling to a table key, or None."""
    if resolution is None:
        return None
    key = str(resolution).strip().upper()
    return _RESOLUTION_ALIASES.get(key)


# Omni's container duration lands a fraction over the nominal length: a 3s
# request measured 3.01s and a 10s edit measured 10.01s, so every omni quote
# came in just under the metered cost. Tiny per call, but the tools state the
# invariant that a pre-flight may over-state and must never under-state, and a
# 20-beat animatic accrues the shortfall twenty times. One frame at omni's
# 24fps is the smallest principled allowance that clears the observed
# overhang. Veo measured exactly on nominal, so it gets no allowance.
_OMNI_FPS = 24
OMNI_ENCODER_ALLOWANCE_SECONDS = 1.0 / _OMNI_FPS


def quote_duration_for(model: str, duration_seconds: float) -> float:
    """Duration to price a quote at, allowing for encoder overhang.

    Estimates only. A metered cost uses the duration actually measured from
    the rendered file and needs no allowance.

    Args:
        model: Model the quote is for; only the Omni models carry one.
        duration_seconds: Nominal duration being quoted.

    Returns:
        The duration to price, at or above ``duration_seconds``.
    """
    if is_omni_model(resolve_model_id(model)):
        return duration_seconds + OMNI_ENCODER_ALLOWANCE_SECONDS
    return duration_seconds


def snap_video_duration(
    model: str,
    duration_seconds: float,
    generation_mode: str = "text_to_video",
) -> int:
    """Return the duration that will really be rendered (and billed).

    Veo does not honor arbitrary lengths: ``src.video`` snaps the request to
    the nearest of 4/6/8 seconds, forces 8s for reference-to-video and 7s for
    extensions. Omni clamps to [3, 10]. Estimating against the *requested*
    duration would therefore quote a price nobody is ever charged, so this
    mirrors that logic exactly.

    Args:
        model: Model ID (superseded/aliased spellings are resolved first).
        duration_seconds: Requested duration.
        generation_mode: One of ``src.video``'s modes.
            ``reference_to_video`` and ``extend_video`` override the Veo snap,
            and ``extend_video`` also lifts Omni's per-render clamp, since an
            extension's output is the assembled clip rather than one render.

    Returns:
        The effective duration in whole seconds.
    """
    model_id = resolve_model_id(model)

    if is_omni_model(model_id):
        if generation_mode == "extend_video":
            # An extension renders the ASSEMBLED clip, which legitimately runs
            # past the per-render maximum — measured: a 3.01s source came back
            # 13.01s, and a chain reaches 40s. Clamping it to 10 here reported
            # $0.3393 for a render that billed $0.4396, a silent 30%
            # under-quote on the one path that exceeds the range.
            return max(_OMNI_MIN_DURATION, round(duration_seconds))
        # src/omni.py rounds then clamps into the documented range.
        clamped = round(duration_seconds)
        return max(_OMNI_MIN_DURATION, min(_OMNI_MAX_DURATION, clamped))

    if generation_mode == "reference_to_video":
        return _VEO_REFERENCE_DURATION
    if generation_mode == "extend_video":
        return _VEO_EXTEND_DURATION

    # Identical expression to src/video.py, ties included: min() keeps the
    # first candidate, so 5.0s snaps down to 4s exactly as the real call does.
    return min(_VEO_ALLOWED_DURATIONS, key=lambda x: abs(x - duration_seconds))


# ---------------------------------------------------------------------------
# Pre-flight estimates
# ---------------------------------------------------------------------------


def estimate_image_cost(
    model: str,
    image_size: str = "1K",
    n: int = 1,
    *,
    reference_images: int = 0,
    input_text_tokens: int = 0,
) -> CostEstimate | None:
    """Estimate the cost of generating ``n`` images before making the call.

    Output image tokens are fully determined by model and resolution, so the
    output half of this estimate is exact. Input tokens are not knowable
    up front, so they are only included when the caller declares them.

    Args:
        model: Image model ID; retired/sunset IDs are priced as the
            replacement that ``src.image`` actually calls.
        image_size: "512", "1K", "2K" or "4K" (case-insensitive). A size the
            model cannot produce is priced at the model's default size,
            because ``src.image`` drops the unsupported request and lets the
            model render its default.
        n: Number of images.
        reference_images: Input/reference images that will be sent, priced at
            the model's per-input-image token count.
        input_text_tokens: Prompt tokens the caller expects to send.

    Returns:
        A ``CostEstimate`` with ``is_estimate=True``, or None when the model
        or the size is unknown — never a guessed figure.
    """
    model_id = resolve_model_id(model)
    pricing = _IMAGE_PRICING.get(model_id)
    if pricing is None or n < 0 or reference_images < 0 or input_text_tokens < 0:
        return None

    size = _normalize_image_size(image_size)
    if size is None:
        return None

    # A size the model cannot render is not an error here: src/image.py warns
    # and sends the request anyway, so the model's default size is what gets
    # billed. Say so in the detail rather than silently repricing.
    substituted = size not in pricing.output_tokens_by_size
    effective_size = pricing.default_size if substituted else size

    per_image = pricing.usd_per_image(effective_size)
    if per_image is None:
        return None

    output_usd = per_image * n
    input_image_tokens = reference_images * pricing.tokens_per_input_image
    input_usd = (
        (input_image_tokens + input_text_tokens)
        * pricing.input_usd_per_mtok
        / _TOKENS_PER_MILLION
    )

    breakdown: dict[str, float] = {
        "images": float(n),
        "output_image_tokens": float(pricing.output_tokens_by_size[effective_size] * n),
        "output_image_usd": output_usd,
    }
    if input_image_tokens or input_text_tokens:
        breakdown["input_tokens"] = float(input_image_tokens + input_text_tokens)
        breakdown["input_usd"] = input_usd

    detail = f"{n} image{'s' if n != 1 else ''} @ {effective_size} on {model_id}"
    if substituted:
        detail += f" ({image_size} unsupported; priced at the model default)"
    if reference_images:
        detail += f", {reference_images} reference image(s)"

    return CostEstimate(
        usd=output_usd + input_usd,
        is_estimate=True,
        unit="image",
        detail=detail,
        breakdown=breakdown,
        source=pricing.source,
        source_note=pricing.source_note,
    )


def estimate_video_cost(
    model: str,
    duration_seconds: float,
    resolution: str = "720p",
    include_audio: bool = True,
    *,
    generation_mode: str = "text_to_video",
) -> CostEstimate | None:
    """Estimate the cost of a video generation before making the call.

    The duration is snapped exactly as ``src.video``/``src.omni`` snap it, so
    the estimate reflects the clip that will really be rendered — asking for
    5 seconds of Veo is quoted (and billed) as 4.

    Args:
        model: Veo or Omni model ID; per-backend Veo spellings both work.
        duration_seconds: Requested duration; snapped before pricing.
        resolution: "360p" (gemini-omni-1.1-flash only), "720p", "1080p" or
            "4K" (case-insensitive).
        include_audio: Kept for interface symmetry and reporting. Veo 3.1 and
            Omni publish a single rate that already includes audio, so this
            does not change the price — only the detail string, so a caller
            can see that disabling audio saves nothing.
        generation_mode: One of ``src.video``'s modes; affects only the
            duration snap (reference-to-video is 8s, extensions are 7s).

    Returns:
        A ``CostEstimate`` with ``is_estimate=True``, or None when the model
        is unknown, the duration is negative, or the model has no published
        rate for that resolution (e.g. 4K on Veo 3.1 Lite).
    """
    # A negative duration is not a short clip, it is a bad call. Snapping it
    # to the 4s floor would quote a real price for an impossible request, so
    # decline it the same way an unknown model is declined.
    try:
        value = float(duration_seconds)
    except (TypeError, ValueError):
        return None
    # NaN passes a bare < 0 check and would be quoted at the 4s floor.
    if not math.isfinite(value) or value < 0:
        return None

    model_id = resolve_model_id(model)
    pricing = _VIDEO_PRICING.get(model_id)
    if pricing is None:
        return None

    requested = _normalize_resolution(resolution)
    if requested is None:
        return None

    # Omni only ever renders 720p, so a 1080p request still bills at 720p.
    effective_resolution = pricing.fixed_resolution or requested
    usd_per_second = pricing.usd_per_second_by_resolution.get(effective_resolution)
    if usd_per_second is None:
        # No published rate for this pairing (Veo Lite at 4K). Refusing to
        # answer beats inventing a rate for a combination that would be
        # rejected anyway.
        return None

    # Quote at or above what will be billed: omni renders a fraction over
    # nominal, so a bare snapped duration under-quotes every omni call.
    seconds = quote_duration_for(
        model_id, snap_video_duration(model_id, duration_seconds, generation_mode)
    )
    return _build_video_estimate(
        model_id=model_id,
        pricing=pricing,
        seconds=float(seconds),
        requested_resolution=requested,
        effective_resolution=effective_resolution,
        usd_per_second=usd_per_second,
        include_audio=include_audio,
        is_estimate=True,
    )


def _build_video_estimate(
    *,
    model_id: str,
    pricing: VideoModelPricing,
    seconds: float,
    requested_resolution: str,
    effective_resolution: str,
    usd_per_second: float,
    include_audio: bool,
    is_estimate: bool,
) -> CostEstimate:
    """Assemble a per-second video ``CostEstimate`` (shared by both paths)."""
    usd = seconds * usd_per_second

    detail = (
        f"{seconds:g}s of video @ {effective_resolution} on {model_id} "
        f"(${usd_per_second:g}/s)"
    )
    if effective_resolution != requested_resolution:
        detail += f"; {requested_resolution} was requested but {model_id} renders "
        detail += f"{effective_resolution}"
    if pricing.audio_included and not include_audio:
        detail += "; audio disabled but the published rate bundles audio, so "
        detail += "there is no discount"

    # A rate that is not the one Google printed says so on the figure itself,
    # not only in the module's comments — this string is what a caller sees
    # next to the money.
    resolution_note = pricing.resolution_notes.get(effective_resolution)
    source_note = pricing.source_note
    if resolution_note:
        source_note = (
            f"{source_note} {resolution_note}" if source_note else resolution_note
        )

    return CostEstimate(
        usd=usd,
        is_estimate=is_estimate,
        unit="second-of-video",
        detail=detail,
        breakdown={
            "seconds": seconds,
            "usd_per_second": usd_per_second,
            "video_usd": usd,
        },
        source=pricing.source,
        source_note=source_note,
    )


# ---------------------------------------------------------------------------
# Usage-metadata extraction
# ---------------------------------------------------------------------------


def _get_field(obj: Any, *names: str) -> Any:
    """Read the first present field from an SDK object or a plain dict.

    Deliberately duck-typed: the caller may hand us a
    ``GenerateContentResponseUsageMetadata``, a REST-shaped dict with
    camelCase keys, or a hand-rolled test double. Hard-depending on an SDK
    class would make this module break whenever google-genai reshuffles its
    types, and a missing field must never raise.
    """
    if obj is None:
        return None
    for name in names:
        if isinstance(obj, Mapping):
            value = obj.get(name)
        else:
            value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


def _as_token_count(value: Any) -> float | None:
    """Coerce a reported token count to a float, or None if it is not one."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        # NaN/inf would poison every downstream sum; treat them as unreported.
        return number if math.isfinite(number) and number >= 0 else None
    return None


def _modality_tokens(details: Any) -> dict[str, float]:
    """Sum reported token counts per modality from a *_tokens_details list.

    Modality is matched on its string form (``MediaModality.IMAGE``, "IMAGE",
    "MODALITY_IMAGE" have all appeared) so no enum import is needed.
    """
    totals: dict[str, float] = {}
    if not isinstance(details, Sequence) or isinstance(details, (str, bytes)):
        return totals
    for entry in details:
        modality = _get_field(entry, "modality")
        tokens = _as_token_count(_get_field(entry, "token_count", "tokenCount"))
        if modality is None or tokens is None:
            continue
        name = str(getattr(modality, "value", modality)).upper()
        for known in ("IMAGE", "VIDEO", "AUDIO", "TEXT", "DOCUMENT"):
            if known in name:
                totals[known] = totals.get(known, 0.0) + tokens
                break
    return totals


@dataclass(frozen=True)
class _Usage:
    """Normalized view of a response's usage metadata."""

    prompt_tokens: float | None
    candidates_tokens: float | None
    thoughts_tokens: float | None
    total_tokens: float | None
    prompt_modalities: dict[str, float]
    candidate_modalities: dict[str, float]

    @property
    def is_empty(self) -> bool:
        """True when nothing usable was reported."""
        return (
            self.prompt_tokens is None
            and self.candidates_tokens is None
            and self.thoughts_tokens is None
            and self.total_tokens is None
            and not self.prompt_modalities
            and not self.candidate_modalities
        )


def _extract_usage(usage_metadata: Any) -> _Usage:
    """Normalize an SDK usage object, a dict, or None into a ``_Usage``."""
    return _Usage(
        prompt_tokens=_as_token_count(
            _get_field(usage_metadata, "prompt_token_count", "promptTokenCount")
        ),
        candidates_tokens=_as_token_count(
            _get_field(usage_metadata, "candidates_token_count", "candidatesTokenCount")
        ),
        thoughts_tokens=_as_token_count(
            _get_field(usage_metadata, "thoughts_token_count", "thoughtsTokenCount")
        ),
        total_tokens=_as_token_count(
            _get_field(usage_metadata, "total_token_count", "totalTokenCount")
        ),
        prompt_modalities=_modality_tokens(
            _get_field(usage_metadata, "prompt_tokens_details", "promptTokensDetails")
        ),
        candidate_modalities=_modality_tokens(
            _get_field(
                usage_metadata, "candidates_tokens_details", "candidatesTokensDetails"
            )
        ),
    )


# ---------------------------------------------------------------------------
# Post-hoc actual costs
# ---------------------------------------------------------------------------


def actual_image_cost(
    model: str,
    usage_metadata: Any = None,
    image_size: str = "1K",
    n: int = 1,
) -> CostEstimate | None:
    """Derive the real cost of a completed image call from reported usage.

    Args:
        model: Model ID from the response (already rerouted by ``src.image``,
            but superseded IDs are resolved again here for safety).
        usage_metadata: The response's ``usage_metadata`` — an SDK object, a
            dict, or None. Missing or partial fields degrade gracefully.
        image_size: Size that was requested, used to price the output when
            the API did not break usage down by modality.
        n: Number of images returned.

    Returns:
        A ``CostEstimate``, or None for an unknown model/size.
        ``is_estimate`` is False when output tokens were actually reported and
        True when the figure fell back to unit pricing.
    """
    model_id = resolve_model_id(model)
    pricing = _IMAGE_PRICING.get(model_id)
    if pricing is None or n < 0:
        return None

    size = _normalize_image_size(image_size)
    if size is None:
        return None
    if size not in pricing.output_tokens_by_size:
        size = pricing.default_size

    usage = _extract_usage(usage_metadata)
    if usage.is_empty:
        # Nothing was reported: fall back to the pre-flight table so the
        # caller still gets a number, honestly flagged as an estimate.
        return estimate_image_cost(model_id, size, n)

    expected_image_tokens = float(pricing.output_tokens_by_size[size] * n)

    reported_image = usage.candidate_modalities.get("IMAGE")
    reported_text = usage.candidate_modalities.get("TEXT")
    if reported_image is not None:
        # Best case: the API told us exactly how the output split.
        image_tokens = reported_image
        text_tokens = reported_text or 0.0
        output_reported = True
    elif usage.candidates_tokens is not None:
        # Only a total. Image models emit a fixed, documented number of image
        # tokens per rendered image, so attribute those first and treat the
        # remainder as text/thinking rather than over-billing the whole total
        # at the (20x higher) image rate.
        image_tokens = min(expected_image_tokens, usage.candidates_tokens)
        text_tokens = usage.candidates_tokens - image_tokens
        output_reported = True
    else:
        image_tokens = expected_image_tokens
        text_tokens = 0.0
        output_reported = False

    # Thinking tokens (gemini-3-pro-image) are billed at the text output rate
    # and are reported separately from candidates_token_count.
    text_tokens += usage.thoughts_tokens or 0.0

    prompt_tokens = usage.prompt_tokens or 0.0
    input_usd = prompt_tokens * pricing.input_usd_per_mtok / _TOKENS_PER_MILLION
    image_usd = image_tokens * pricing.output_image_usd_per_mtok / _TOKENS_PER_MILLION
    text_usd = text_tokens * pricing.output_text_usd_per_mtok / _TOKENS_PER_MILLION

    breakdown: dict[str, float] = {
        "images": float(n),
        "input_tokens": prompt_tokens,
        "input_usd": input_usd,
        "output_image_tokens": image_tokens,
        "output_image_usd": image_usd,
    }
    if text_tokens:
        breakdown["output_text_tokens"] = text_tokens
        breakdown["output_text_usd"] = text_usd

    if output_reported:
        detail = (
            f"{n} image{'s' if n != 1 else ''} @ {size} on {model_id} — "
            f"{image_tokens:g} image + {text_tokens:g} text output tokens, "
            f"{prompt_tokens:g} input tokens"
        )
    else:
        detail = (
            f"{n} image{'s' if n != 1 else ''} @ {size} on {model_id} — "
            f"output tokens not reported, priced from the per-image table"
        )

    return CostEstimate(
        usd=input_usd + image_usd + text_usd,
        is_estimate=not output_reported,
        unit="token" if output_reported else "image",
        detail=detail,
        breakdown=breakdown,
        source=pricing.source,
        source_note=pricing.source_note,
    )


def actual_video_cost(
    model: str,
    duration_seconds: float | None = None,
    resolution: str = "720p",
    include_audio: bool = True,
    usage_metadata: Any = None,
    *,
    snap_duration: bool = False,
) -> CostEstimate | None:
    """Derive the real cost of a completed video generation.

    Seconds of video are Veo's billing unit and ``generate_video`` reports the
    duration that was actually rendered, so the per-second path here is an
    actual cost, not an estimate — hence ``snap_duration`` defaults to False:
    the caller is expected to pass the effective duration straight from the
    response (re-snapping a 7s extension would corrupt it).

    Args:
        model: Model ID from the response; both Veo spellings resolve.
        duration_seconds: Effective duration reported by the response. May be
            omitted when ``usage_metadata`` carries video output tokens.
        resolution: Resolution that was rendered.
        include_audio: Whether audio was generated; reporting only, since the
            published rate bundles audio.
        usage_metadata: Optional usage object/dict. Used when the model bills
            per token (both Omni models do; Veo does not).
        snap_duration: Set True to snap ``duration_seconds`` to what the model
            would really render — only useful if the caller has a *requested*
            rather than an effective duration.

    Returns:
        A ``CostEstimate`` with ``is_estimate=False`` when a real duration or
        real token counts were available, or None if the model is unknown, the
        resolution has no published rate, or no usage information was given.
    """
    model_id = resolve_model_id(model)
    pricing = _VIDEO_PRICING.get(model_id)
    if pricing is None:
        return None

    requested = _normalize_resolution(resolution)
    if requested is None:
        return None
    effective_resolution = pricing.fixed_resolution or requested

    usage = _extract_usage(usage_metadata)

    # Token-billed models (Omni) report video output tokens directly; that is
    # strictly more accurate than multiplying a rounded duration.
    video_tokens = usage.candidate_modalities.get("VIDEO")
    if video_tokens is not None and pricing.output_video_usd_per_mtok is not None:
        text_tokens = (usage.candidate_modalities.get("TEXT") or 0.0) + (
            usage.thoughts_tokens or 0.0
        )
        prompt_tokens = usage.prompt_tokens or 0.0
        video_usd = (
            video_tokens * pricing.output_video_usd_per_mtok / _TOKENS_PER_MILLION
        )
        text_usd = (
            text_tokens * (pricing.output_text_usd_per_mtok or 0.0)
        ) / _TOKENS_PER_MILLION
        input_usd = (
            prompt_tokens * (pricing.input_usd_per_mtok or 0.0)
        ) / _TOKENS_PER_MILLION
        breakdown: dict[str, float] = {
            "input_tokens": prompt_tokens,
            "input_usd": input_usd,
            "output_video_tokens": video_tokens,
            "output_video_usd": video_usd,
        }
        if text_tokens:
            breakdown["output_text_tokens"] = text_tokens
            breakdown["output_text_usd"] = text_usd
        return CostEstimate(
            usd=input_usd + video_usd + text_usd,
            is_estimate=False,
            unit="token",
            detail=(
                f"{video_tokens:g} video output tokens on {model_id} "
                f"(${pricing.output_video_usd_per_mtok:g}/1M)"
            ),
            breakdown=breakdown,
            source=pricing.source,
            source_note=pricing.source_note,
        )

    if duration_seconds is None:
        # No duration and no token counts: there is nothing to price from.
        return None

    # Same guard estimate_video_cost applies: a negative duration would bill a
    # negative dollar amount, and NaN survives a bare < 0 check and serializes
    # as invalid-JSON `NaN`. Current callers snap to finite ints upstream, so
    # this hardens the public entry point rather than fixing a live path.
    try:
        duration_seconds = float(duration_seconds)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(duration_seconds) or duration_seconds < 0:
        return None

    usd_per_second = pricing.usd_per_second_by_resolution.get(effective_resolution)
    if usd_per_second is None:
        return None

    seconds = (
        float(snap_video_duration(model_id, duration_seconds))
        if snap_duration
        else float(duration_seconds)
    )
    return _build_video_estimate(
        model_id=model_id,
        pricing=pricing,
        seconds=seconds,
        requested_resolution=requested,
        effective_resolution=effective_resolution,
        usd_per_second=usd_per_second,
        include_audio=include_audio,
        is_estimate=False,
    )


# ---------------------------------------------------------------------------
# Aggregation and presentation
# ---------------------------------------------------------------------------


# Breakdown entries that are rates, not quantities: adding them across
# components produces a number that means nothing.
_RATE_BREAKDOWN_KEYS = frozenset({"usd_per_second"})


def sum_costs(
    estimates: Iterable[CostEstimate | None],
    label: str = "total",
) -> CostEstimate:
    """Add up many costs, e.g. every beat of a multi-shot ``generate_clip``.

    Components are summed at full precision (no intermediate rounding), and a
    ``None`` component — an unpriced model — is counted rather than silently
    dropped, so a reel total can never look complete when part of it is
    unknown.

    Args:
        estimates: Costs to combine; ``None`` entries mean "unpriced".
        label: Name for the aggregate, used in ``detail``.

    Returns:
        A ``CostEstimate`` whose ``unit`` is the shared unit of the components
        or "mixed", and whose ``is_estimate`` is True if any component was an
        estimate or could not be priced. ``source``/``source_note`` are carried
        only when every priced component agrees on them.
    """
    total = 0.0
    breakdown: dict[str, float] = {}
    units: set[str] = set()
    sources: set[str | None] = set()
    source_notes: set[str | None] = set()
    counted = 0
    unpriced = 0
    any_estimate = False

    for estimate in estimates:
        if estimate is None:
            unpriced += 1
            continue
        counted += 1
        total += estimate.usd
        units.add(estimate.unit)
        sources.add(estimate.source)
        source_notes.add(estimate.source_note)
        any_estimate = any_estimate or estimate.is_estimate
        for key, value in estimate.breakdown.items():
            if key in _RATE_BREAKDOWN_KEYS:
                # A rate is not additive. Summing $0.40/s across 39 components
                # reported usd_per_second: 15.6 — a nonsense figure for any
                # client that renders it. Carry it through when every
                # component agrees, and drop it when they differ.
                if breakdown.get(key, value) != value:
                    breakdown[key] = float("nan")
                else:
                    breakdown[key] = value
            else:
                breakdown[key] = breakdown.get(key, 0.0) + value

    # A mixed-rate aggregate has no single rate to report; omit it rather
    # than emit NaN, which is not valid JSON.
    for key in _RATE_BREAKDOWN_KEYS:
        if key in breakdown and breakdown[key] != breakdown[key]:
            del breakdown[key]

    breakdown["components"] = float(counted)
    if unpriced:
        breakdown["unpriced_components"] = float(unpriced)

    unit = units.pop() if len(units) == 1 else "mixed"

    # Provenance obeys the same rule as usd_per_second above: an aggregate may
    # claim only what every one of its components actually shares. Picking one
    # component's page would point a caller at a table that does not contain
    # part of the total, and there is no "mixed" URL to fall back on — so the
    # honest answer for a disagreement is no answer. A component with no
    # source of its own (one built outside this module) is a disagreement too,
    # which is why None participates in the set. Unpriced components add no
    # dollars and so cannot contradict a source; `unpriced_components` already
    # tells the caller the total is incomplete.
    source = sources.pop() if len(sources) == 1 else None
    source_note = source_notes.pop() if len(source_notes) == 1 else None

    detail = f"{label}: {counted} priced component{'s' if counted != 1 else ''}"
    if unpriced:
        detail += f", {unpriced} unpriced (total is a lower bound)"

    return CostEstimate(
        usd=total,
        is_estimate=any_estimate or bool(unpriced),
        unit=unit,
        detail=detail,
        breakdown=breakdown,
        source=source,
        source_note=source_note,
    )


def format_cost(estimate: CostEstimate | float | None) -> str:
    """Render a cost as a dollar string with enough precision to be useful.

    Two decimal places would print "$0.00" for a real charge of $0.0043, which
    reads as free. Precision therefore scales with magnitude, and trailing
    zeros beyond cents are trimmed so common figures still look like money.

    Args:
        estimate: A ``CostEstimate``, a bare USD float, or None.

    Returns:
        e.g. "$1.20", "$0.0672", "$0.00056", or "unpriced" for None.
    """
    if estimate is None:
        return "unpriced"

    usd = estimate.usd if isinstance(estimate, CostEstimate) else float(estimate)
    if not math.isfinite(usd):
        return "unpriced"

    sign = "-" if usd < 0 else ""
    value = abs(usd)

    if value == 0:
        return f"{sign}$0.00"
    if value >= 1:
        decimals = 2
    elif value >= 0.01:
        decimals = 4
    elif value >= 0.0001:
        decimals = 6
    else:
        decimals = 8

    text = f"{value:,.{decimals}f}"
    if "." in text:
        # Trim trailing zeros, but never below cents: "$0.10", not "$0.1".
        integer, _, fraction = text.partition(".")
        fraction = fraction.rstrip("0")
        fraction = fraction.ljust(2, "0")
        text = f"{integer}.{fraction}"
    return f"{sign}${text}"


def format_cost_line(estimate: CostEstimate | None) -> str:
    """Render a cost as "<amount> (<detail>)" for logs and tool responses."""
    if estimate is None:
        return "unpriced"
    prefix = "~" if estimate.is_estimate else ""
    return f"{prefix}{format_cost(estimate)} ({estimate.detail})"


def cost_to_dict(estimate: CostEstimate | None) -> dict[str, Any] | None:
    """Convert a cost to a JSON-safe dict for an MCP tool response.

    Args:
        estimate: The cost to serialize, or None for an unpriced operation.

    Returns:
        A dict of JSON primitives, or None when ``estimate`` is None.
        ``pricing_as_of`` plus ``pricing_source``/``pricing_source_note`` make
        the figure checkable: a caller can open the page the rate came from
        rather than trusting a bare number. Both provenance keys are always
        present, and are ``None`` when the cost has no single honest source.
    """
    if estimate is None:
        return None
    # The tool response is a presentation boundary: intermediate arithmetic
    # stays full-precision, but 3 * $0.80 must not reach a caller as
    # $2.4000000000000004. Six decimals keeps sub-cent precision for the
    # smallest real charges (a 512px image is ~$0.045).
    return {
        "usd": round(estimate.usd, 6),
        "usd_display": format_cost(estimate),
        "is_estimate": estimate.is_estimate,
        "unit": estimate.unit,
        "detail": estimate.detail,
        "breakdown": {k: round(v, 6) for k, v in estimate.breakdown.items()},
        "pricing_as_of": PRICING_AS_OF,
        # The date alone is unfalsifiable: it says when someone checked, not
        # what they checked against. The URL is what lets a caller who doubts
        # a quote go and verify it, and the note is what stops them checking
        # a Veo figure against a Vertex page that never published one.
        "pricing_source": estimate.source,
        "pricing_source_note": estimate.source_note,
    }


# ---------------------------------------------------------------------------
# Coverage reporting
# ---------------------------------------------------------------------------


def is_priced(model: str) -> bool:
    """Whether ``model`` (or the model it reroutes to) has an embedded price."""
    model_id = resolve_model_id(model)
    return model_id in _IMAGE_PRICING or model_id in _VIDEO_PRICING


def priced_models() -> tuple[str, ...]:
    """Canonical model IDs with an embedded price, sorted."""
    return tuple(sorted({*_IMAGE_PRICING, *_VIDEO_PRICING}))


def known_models() -> tuple[str, ...]:
    """Every model ID this server accepts, current and superseded.

    Read from the sibling modules' own type annotations so a model added there
    shows up here (as unpriced) instead of quietly escaping coverage checks.
    """
    models: set[str] = {
        *get_args(ImageModel),
        *get_args(RetiredImageModel),
        *get_args(VideoModel),
        *_GEMINI_API_MODEL_IDS.values(),
        *OMNI_MODEL_ALIASES,
    }
    return tuple(sorted(models))


def unpriced_models(models: Iterable[str] | None = None) -> tuple[str, ...]:
    """Model IDs with no embedded price, sorted.

    Args:
        models: IDs to check; defaults to the server's full catalog.

    Returns:
        The subset that ``estimate_*``/``actual_*`` would answer None for.
    """
    candidates = known_models() if models is None else models
    return tuple(sorted({m for m in candidates if not is_priced(m)}))


def describe_model_pricing(model: str) -> dict[str, Any] | None:
    """Return the embedded pricing record for ``model``, or None.

    Useful for a "why does it cost that?" tool response: it exposes the rates,
    the resolution tiers, the resolved model ID, the source URL and any caveat
    attached to that source, so a user can check the figure against Google's
    page themselves.
    """
    model_id = resolve_model_id(model)

    image = _IMAGE_PRICING.get(model_id)
    if image is not None:
        return {
            "model": model_id,
            "requested_model": str(model),
            "kind": "image",
            "pricing_as_of": PRICING_AS_OF,
            "source": image.source,
            "source_note": image.source_note,
            "input_usd_per_mtok": image.input_usd_per_mtok,
            "output_text_usd_per_mtok": image.output_text_usd_per_mtok,
            "output_image_usd_per_mtok": image.output_image_usd_per_mtok,
            "tokens_per_input_image": image.tokens_per_input_image,
            "usd_per_image": {
                size: image.usd_per_image(size)
                for size in sorted(image.output_tokens_by_size)
            },
        }

    video = _VIDEO_PRICING.get(model_id)
    if video is not None:
        return {
            "model": model_id,
            "requested_model": str(model),
            "kind": "video",
            "pricing_as_of": PRICING_AS_OF,
            "source": video.source,
            "source_note": video.source_note,
            "usd_per_second": dict(video.usd_per_second_by_resolution),
            "audio_included_in_price": video.audio_included,
            "output_video_usd_per_mtok": video.output_video_usd_per_mtok,
            # The published scalar is the 720p figure. Emitting it beside
            # per-resolution dollar rates derived from a different token count
            # let a caller multiply it out and get 3x what the same dict's own
            # 360p rate says, so the per-resolution counts are published too
            # and the two can be reconciled.
            "tokens_per_second": video.tokens_per_second,
            **(
                {
                    "tokens_per_second_by_resolution": {
                        resolution: usd
                        * _TOKENS_PER_MILLION
                        / video.output_video_usd_per_mtok
                        for resolution, usd in sorted(
                            video.usd_per_second_by_resolution.items()
                        )
                    }
                }
                if video.tokens_per_second is not None
                and video.output_video_usd_per_mtok
                else {}
            ),
            "fixed_resolution": video.fixed_resolution,
            # Only present when some resolution's figure needs a sentence:
            # an empty dict would read as "all four rates are published".
            **(
                {"resolution_notes": dict(video.resolution_notes)}
                if video.resolution_notes
                else {}
            ),
        }

    return None


def pricing_coverage() -> dict[str, Any]:
    """Summarize what this module can and cannot price.

    Returns:
        A dict with the as-of date, source URLs, the priced canonical models,
        and any accepted model ID left unpriced — intended to be surfaced
        verbatim so stale or missing coverage is visible to users.
    """
    unpriced = unpriced_models()
    return {
        "pricing_as_of": PRICING_AS_OF,
        "sources": dict(PRICING_SOURCES),
        "priced_models": priced_models(),
        "known_models": known_models(),
        "unpriced_models": unpriced,
        "fully_covered": not unpriced,
        "tier": "standard-paid",
        "batch_price_multiplier": BATCH_PRICE_MULTIPLIER,
        "notes": (
            "Standard paid-tier prices; no free tier exists for these media "
            "models. Batch requests bill at half these rates. Veo per-second "
            "rates were confirmed on the Gemini Developer API pricing page "
            "and are assumed to match Vertex AI."
        ),
    }
