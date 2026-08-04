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
  for all three image models, all three Veo 3.1 variants and
  ``gemini-omni-flash-preview``. So a paid-tier price is always the right
  price here, and there is no free-tier branch to model.
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
This module imports those same tables (``_RETIRED_MODELS``,
``_SUNSET_MODELS``) instead of duplicating them, so a price quoted for
``imagen-4.0-generate-001`` is the price of the model that really runs. The
same applies to the Veo ID translation in ``src.video``: the Gemini Developer
API serves Veo under ``-preview`` IDs, and ``generate_video`` reports the
translated ID back to the caller, so both spellings resolve to one price.
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, get_args

from .image import (
    _RETIRED_DEFAULT_TARGET,
    _RETIRED_MODELS,
    _SUNSET_MODELS,
    ImageModel,
    RetiredImageModel,
)
from .omni import OMNI_MODEL
from .video import _GEMINI_API_MODEL_IDS, VideoModel

# Date on which every price in this module was verified against the sources
# below. Bump it (and re-check the tables) whenever these are refreshed —
# callers surface it so a stale figure is visible rather than silent.
PRICING_AS_OF = "2026-08-03"

PRICING_SOURCES: dict[str, str] = {
    "gemini_api": "https://ai.google.dev/gemini-api/docs/pricing",
    "vertex_ai": "https://cloud.google.com/vertex-ai/generative-ai/pricing",
}

# Every embedded rate below was read from the Gemini Developer API pricing
# page; the image rates were additionally cross-checked against the Vertex AI
# page (PRICING_SOURCES["vertex_ai"]), which publishes the same numbers.
_SRC_GEMINI_API = PRICING_SOURCES["gemini_api"]

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
    """

    usd: float
    is_estimate: bool
    unit: str
    detail: str
    breakdown: dict[str, float]


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
    """

    input_usd_per_mtok: float
    output_text_usd_per_mtok: float
    output_image_usd_per_mtok: float
    tokens_per_input_image: int
    output_tokens_by_size: Mapping[str, int]
    default_size: str
    source: str

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
            published.
        output_text_usd_per_mtok: USD per 1M text output tokens, when the
            model can also emit text.
        input_usd_per_mtok: USD per 1M input tokens, when published.
        fixed_resolution: Set when the model only ever emits one resolution,
            so a differing request is priced at what is actually rendered.
        source: URL the numbers were read from.
    """

    usd_per_second_by_resolution: Mapping[str, float]
    audio_included: bool
    source: str
    output_video_usd_per_mtok: float | None = None
    tokens_per_second: int | None = None
    output_text_usd_per_mtok: float | None = None
    input_usd_per_mtok: float | None = None
    fixed_resolution: str | None = None


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

_VIDEO_PRICING: dict[str, VideoModelPricing] = {
    "veo-3.1-generate-001": VideoModelPricing(
        usd_per_second_by_resolution={"720p": 0.40, "1080p": 0.40, "4K": 0.60},
        audio_included=True,
        source=_SRC_GEMINI_API,
    ),
    "veo-3.1-fast-generate-001": VideoModelPricing(
        usd_per_second_by_resolution={"720p": 0.10, "1080p": 0.12, "4K": 0.30},
        audio_included=True,
        source=_SRC_GEMINI_API,
    ),
    "veo-3.1-lite-generate-preview": VideoModelPricing(
        # 4K is deliberately absent: Lite cannot render it (src/video.py
        # rejects the combination) and Google publishes no 4K Lite rate.
        usd_per_second_by_resolution={"720p": 0.05, "1080p": 0.08},
        audio_included=True,
        source=_SRC_GEMINI_API,
    ),
    OMNI_MODEL: VideoModelPricing(
        # Omni is billed per output token: "5,792 tokens per second of 720p
        # video" at $17.50 per 1M video output tokens, i.e. ~$0.10/second.
        # Derived rather than hard-coded so the per-second and per-token
        # paths can never disagree.
        usd_per_second_by_resolution={
            "720p": _OMNI_VIDEO_TOKENS_PER_SECOND
            * _OMNI_VIDEO_USD_PER_MTOK
            / _TOKENS_PER_MILLION
        },
        audio_included=True,
        output_video_usd_per_mtok=_OMNI_VIDEO_USD_PER_MTOK,
        tokens_per_second=_OMNI_VIDEO_TOKENS_PER_SECOND,
        output_text_usd_per_mtok=9.00,
        input_usd_per_mtok=1.50,
        # Omni always renders 720p (see src/omni.py), so any other request is
        # still billed at the 720p rate.
        fixed_resolution="720p",
        source=_SRC_GEMINI_API,
    ),
}

# Veo IDs are spelled differently per backend: the Gemini Developer API serves
# `-preview`, Vertex serves `-001`, and generate_video reports back whichever
# it used. Invert src/video.py's table so both spellings price identically
# instead of one of them looking like an unknown model.
_VIDEO_MODEL_ALIASES: dict[str, str] = {
    api_id: canonical for canonical, api_id in _GEMINI_API_MODEL_IDS.items()
}

# Durations Veo actually accepts, mirroring src/video.py. Anything else is
# snapped to the nearest of these, so an estimate matches the clip that is
# really billed.
_VEO_ALLOWED_DURATIONS = (4, 6, 8)
_VEO_REFERENCE_DURATION = 8  # reference_to_video is 8s only
_VEO_EXTEND_DURATION = 7  # extend_video outputs exactly 7s

# Omni clamps to [3, 10]s (src/omni.py).
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
    if model_id in _SUNSET_MODELS:
        return _SUNSET_MODELS[model_id][0]
    if model_id in _RETIRED_MODELS:
        return _RETIRED_MODELS[model_id][0]
    if model_id.startswith("imagen"):
        return _RETIRED_DEFAULT_TARGET

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
        generation_mode: One of ``src.video``'s modes; only
            ``reference_to_video`` and ``extend_video`` override the snap.

    Returns:
        The effective duration in whole seconds.
    """
    model_id = resolve_model_id(model)

    if model_id == OMNI_MODEL:
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
        resolution: "720p", "1080p" or "4K" (case-insensitive).
        include_audio: Kept for interface symmetry and reporting. Veo 3.1 and
            Omni publish a single rate that already includes audio, so this
            does not change the price — only the detail string, so a caller
            can see that disabling audio saves nothing.
        generation_mode: One of ``src.video``'s modes; affects only the
            duration snap (reference-to-video is 8s, extensions are 7s).

    Returns:
        A ``CostEstimate`` with ``is_estimate=True``, or None when the model
        is unknown or the model has no published rate for that resolution
        (e.g. 4K on Veo 3.1 Lite).
    """
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

    seconds = snap_video_duration(model_id, duration_seconds, generation_mode)
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
            per token (``gemini-omni-flash-preview``).
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
        )

    if duration_seconds is None:
        # No duration and no token counts: there is nothing to price from.
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
        estimate or could not be priced.
    """
    total = 0.0
    breakdown: dict[str, float] = {}
    units: set[str] = set()
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
        any_estimate = any_estimate or estimate.is_estimate
        for key, value in estimate.breakdown.items():
            breakdown[key] = breakdown.get(key, 0.0) + value

    breakdown["components"] = float(counted)
    if unpriced:
        breakdown["unpriced_components"] = float(unpriced)

    unit = units.pop() if len(units) == 1 else "mixed"
    detail = f"{label}: {counted} priced component{'s' if counted != 1 else ''}"
    if unpriced:
        detail += f", {unpriced} unpriced (total is a lower bound)"

    return CostEstimate(
        usd=total,
        is_estimate=any_estimate or bool(unpriced),
        unit=unit,
        detail=detail,
        breakdown=breakdown,
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
    """Convert a cost to a JSON-safe dict for an MCP tool response."""
    if estimate is None:
        return None
    return {
        "usd": estimate.usd,
        "usd_display": format_cost(estimate),
        "is_estimate": estimate.is_estimate,
        "unit": estimate.unit,
        "detail": estimate.detail,
        "breakdown": dict(estimate.breakdown),
        "pricing_as_of": PRICING_AS_OF,
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
        OMNI_MODEL,
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
    the resolution tiers, the resolved model ID and the source URL, so a user
    can check the figure against Google's page themselves.
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
            "usd_per_second": dict(video.usd_per_second_by_resolution),
            "audio_included_in_price": video.audio_included,
            "output_video_usd_per_mtok": video.output_video_usd_per_mtok,
            "tokens_per_second": video.tokens_per_second,
            "fixed_resolution": video.fixed_resolution,
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
