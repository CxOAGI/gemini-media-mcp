"""Intent routing: turn "what I want to make" into concrete generation plans.

This module is the server's model-selection expertise, expressed as data and
rules instead of prose in a docstring. Given a natural-language description of
the desired output (plus optional structured constraints) it returns a ranked
set of ready-to-call plans: which tool, which model, which parameters, why,
roughly what it costs, and what to watch out for.

It is deliberately ADDITIVE: the explicit generate_* tools keep working exactly
as before. Nothing here generates media, makes a network call, touches the
filesystem, or consults the clock — routing is pure, deterministic, rule-based
logic, so the same input always produces the same plan.

Capability facts that already exist in the implementation modules (image size
support, Veo's backend, omni's duration bounds) are IMPORTED
rather than restated: a second copy would drift the first time Google changes
a model. Facts with no existing home (relative cost/fidelity of each model,
tool-level routing rules) are defined here.
"""

import math
import re
from dataclasses import dataclass, replace
from typing import Any, Literal, Protocol, get_args

from .image import (
    _IMAGE_SIZE_SUPPORT,  # pyright: ignore[reportPrivateUsage]
    _supports_image_size,  # pyright: ignore[reportPrivateUsage]
    ImageModel,
    ImageSize,
    resolve_image_model,
)
from .omni import (
    _MIN_DURATION as OMNI_MIN_DURATION,  # pyright: ignore[reportPrivateUsage]
)
from .omni import (
    _SUPPORTED_ASPECT_RATIOS as OMNI_ASPECT_RATIOS,  # pyright: ignore[reportPrivateUsage]
)
from .omni import (
    _TASK_UPSCALE as OMNI_TASK_UPSCALE,  # pyright: ignore[reportPrivateUsage]
)
from .omni import (
    _TASK_EXTEND as OMNI_TASK_EXTEND,  # pyright: ignore[reportPrivateUsage]
)

from .omni import (
    OMNI_1_2_MODEL,
    DEFAULT_OMNI_MODEL,
    OMNI_DEFAULT_RESOLUTION,
    omni_preview_is_sunset,
    OMNI_MODEL,
    OMNI_PREVIEW_MODEL,
    OMNI_PREVIEW_SUNSET,
    OMNI_RESOLUTIONS,
    OMNI_SPECS,
    OmniModelSpec,
    is_omni_model,
    canonical_omni_model,
    omni_continuation_upper_bound,
    omni_extension_duration_refusal,
    omni_extension_priced_lengths,
    omni_extension_refusal,
    omni_inline_delivery_policy,
    omni_model_is_retired,
    omni_model_reroute_warning,
    omni_source_limit_seconds,
    omni_spec,
)
from .video import (
    _CANONICAL_VIDEO_MODEL_IDS,  # pyright: ignore[reportPrivateUsage]
    VEO_GEMINI_API_SHUTDOWN,
    VideoModel,
)

# Reference-to-video renders exactly this long whatever the request asks for
# (the `effective_duration = 8` force in src/video.py). Mirrored rather than
# imported because it is a function-local literal there, like the 4/6/8 ladder
# above it.
VEO_REFERENCE_MODE_SECONDS = 8.0

# ============================================================================
# Public vocabulary
# ============================================================================

MediaKind = Literal["image", "video"]
BudgetPreference = Literal["cheap", "balanced", "best"]
Backend = Literal["vertex", "gemini_api", "unknown"]
ToolName = Literal[
    "generate_image",
    "generate_storyboard",
    "generate_video",
    "generate_video_omni",
    "extend_video_omni",
    "upscale_video_omni",
    "generate_clip",
    "generate_transition",
    "generate_bridge",
    "edit_video",
    "loop_extend",
]

# The live catalogs, read straight off the Literal types the tools validate
# against, so a model added to (or dropped from) image.py/video.py shows up
# here without a second edit. The retired/superseded IDs are intentionally NOT
# routable: RetiredImageModel exists only so pinned callers keep working, and
# recommending one would be recommending a reroute.
LIVE_IMAGE_MODELS: tuple[str, ...] = get_args(ImageModel)
LIVE_VIDEO_MODELS: tuple[str, ...] = get_args(VideoModel)
IMAGE_SIZES: tuple[str, ...] = get_args(ImageSize)

# Video models with a published end date, and when they stop. A route on one
# of these is still offered — pinning it works until the date — but it is
# demoted below any equally-scored replacement and says when it stops.
DEPRECATED_VIDEO_MODELS: dict[str, str] = {OMNI_PREVIEW_MODEL: OMNI_PREVIEW_SUNSET}

# Tool defaults, mirrored from src/__main__.py so a route that matches the
# documented default can say so (and win ties against equally-scored models).
DEFAULT_IMAGE_MODEL = "gemini-3.1-flash-image"
DEFAULT_VIDEO_MODEL = "veo-3.1-fast-generate-001"
HIGHEST_FIDELITY_VIDEO_MODEL = "veo-3.1-generate-001"

# ============================================================================
# Capability facts with no existing home
# ============================================================================

# Veo accepts only these clip lengths; anything else is snapped to the nearest
# one (mirrors the `allowed = [4, 6, 8]` snap inside src/video.py, which is a
# function-local list and therefore cannot be imported).
VEO_DURATIONS_SECONDS: tuple[int, ...] = (4, 6, 8)
VEO_MAX_CLIP_SECONDS = max(VEO_DURATIONS_SECONDS)

# Each Veo extension appends ~7s and at most 20 can be chained, so a single
# extended Veo video tops out here. Longer runtimes need a multi-beat clip
# spliced downstream.
VEO_EXTENSION_SECONDS = 7
VEO_MAX_EXTENSIONS = 20
VEO_MAX_EXTENDED_SECONDS = (
    VEO_MAX_CLIP_SECONDS + VEO_EXTENSION_SECONDS * VEO_MAX_EXTENSIONS
)

# The longest clip ANY model here renders in one call. Veo tops out at 8s but
# omni renders natively up to 10s, and measuring every request against Veo's
# ceiling hid that: a "10 second video" was capped to an 8s clip, flagged
# needs_extension, and routed to loop_extend -- which needs a video_uri the
# caller does not have -- plus a two-step seed+extend workflow, when
# generate_video_omni(duration_seconds=10) is one call. Read from each omni
# model's own record (1.2 reaches 15s) rather than a flat constant, so it
# stays "the longest any model renders" if one model's limit moves; a model
# that cannot reach a given length is rejected per route by _duration_rejection.
MAX_SINGLE_RENDER_SECONDS = max(
    float(VEO_MAX_CLIP_SECONDS),
    *(float(spec.max_duration_seconds) for spec in OMNI_SPECS.values()),
)

# The longest single-shot ask a Veo route still answers with one clip snapped
# down to 8s, at most 2s short and said in a caveat. Past it, a Veo route
# reaches the runtime the way it always has, by extending an 8s seed with
# loop_extend. This is where every model's single render topped out before
# 1.2 reached 15s; once that moved MAX_SINGLE_RENDER_SECONDS, a 12s or 15s
# ask planned Veo at 8s and ranked it first, a third or more short.
VEO_SNAP_MAX_SECONDS = 10.0

# Veo output resolutions (src/video.py validates against exactly this set).
VEO_RESOLUTIONS: tuple[str, ...] = ("720p", "1080p", "4K")

# What an omni render comes back as when nothing asks otherwise, and the only
# thing gemini-omni-flash-preview can produce at all. gemini-omni-1.2-flash
# takes a real resolution parameter (OMNI_RESOLUTIONS) on top of this default.
OMNI_RESOLUTION = OMNI_DEFAULT_RESOLUTION

# The Omni draft tier: a third of the 720p price, and the cheapest render
# this server can issue. Named here because the workflow builder recommends it
# by name as the preview pass.
OMNI_DRAFT_RESOLUTION = "360p"

# Every resolution some video model here can render, which is what a request
# is validated against: Veo's three plus Omni's 360p draft tier.
VIDEO_RESOLUTIONS: tuple[str, ...] = tuple(
    dict.fromkeys((*VEO_RESOLUTIONS, *OMNI_RESOLUTIONS))
)

# The backends an omni interaction can be recorded on; an interaction lives
# on one of them and resolves on no other.
_INTERACTION_BACKENDS: tuple[str, ...] = ("vertex", "gemini_api")

# Aspect ratios the video path accepts. src/video.py hard-errors on anything
# else, and omni documents the same pair.
VIDEO_ASPECT_RATIOS: tuple[str, ...] = OMNI_ASPECT_RATIOS

# Gemini 3.x image models accept up to 14 reference images (see image.py's
# max_refs) — beyond that the extras are dropped by the SDK call.
MAX_IMAGE_REFERENCE_IMAGES = 14

# Veo accepts at most 3 reference images (asset type), per src/video.py, which
# sends the first 3 of a longer list; _veo_reference_rejection keeps the
# planner from offering that route.
MAX_VIDEO_REFERENCE_IMAGES = 3

# generate_clip hard-errors above this beat count. Mirrored from
# src/__main__.py's MAX_CLIP_BEATS (the source of truth) rather than imported:
# importing it would pull the whole MCP server — FastMCP construction and all —
# into a module whose entire promise is that it only computes. Planning more
# beats than this emitted a fully quoted route the tool refuses outright.
MAX_CLIP_BEATS = 20

# generate_storyboard hard-errors above this many shots, because every shot is
# a billed image generation. Mirrored from src/__main__.py's
# MAX_STORYBOARD_SHOTS (the source of truth) for the same reason MAX_CLIP_BEATS
# is mirrored rather than imported. A storyboard route beyond this count would
# be a fully priced plan the tool refuses outright.
MAX_STORYBOARD_SHOTS = 24

# generate_clip renders every bridge at exactly 4.0s regardless of how long the
# beats are (the duration_seconds=4.0 bridge render in src/__main__.py), and
# its own dry_run prices them at 4.0s. Pricing a bridge at the beat length
# quoted $4.00 for a plan the tool itself quotes at $3.20.
CLIP_BRIDGE_SECONDS = 4.0

# Tools with no resolution parameter at all: they always render 720p, so a 4K
# or 1080p ask cannot reach them and must not be priced as if it had. Quoting
# generate_clip at 4K reported $5.40 against the tool's own $1.80.
FIXED_720P_TOOLS: frozenset[str] = frozenset(
    {"generate_clip", "generate_transition", "generate_bridge"}
)

# Defaults applied when neither the caller nor the intent text says otherwise.
DEFAULT_VIDEO_DURATION_SECONDS = 6.0
DEFAULT_BEAT_COUNT = 3

# An animatic pass is worth recommending once a clip is big enough that a
# wrong creative call is expensive: either many beats, or a cost estimate over
# this threshold. Both are checked so the advice still fires when pricing is
# unavailable.
ANIMATIC_MIN_BEATS = 3
ANIMATIC_MIN_COST_USD = 1.00


# ============================================================================
# Cost integration (src/pricing.py, imported lazily and optionally)
# ============================================================================


class CostEstimateLike(Protocol):
    """Structural view of ``pricing.CostEstimate``.

    Declared structurally so this module type-checks and imports cleanly even
    when src/pricing.py is absent; routing degrades to ``cost=None`` rather
    than failing.
    """

    # Declared as read-only properties so a frozen dataclass (which is what
    # pricing.CostEstimate is) satisfies the protocol.
    @property
    def usd(self) -> float:
        """Total estimated USD."""
        ...

    @property
    def is_estimate(self) -> bool:
        """True for pre-flight estimates."""
        ...

    @property
    def unit(self) -> str:
        """Billing unit, e.g. "image" or "second-of-video"."""
        ...

    @property
    def detail(self) -> str:
        """Human-readable one-liner."""
        ...

    @property
    def breakdown(self) -> dict[str, float]:
        """Component costs behind the total."""
        ...


def _estimate_image_cost(
    model: str, image_size: str, n: int = 1
) -> CostEstimateLike | None:
    """Best-effort image cost estimate, or None when pricing is unavailable.

    Pricing is a separate, optional module: a missing (or failing) price book
    must never stop the router from answering, so every failure mode collapses
    to None.
    """
    try:
        from .pricing import estimate_image_cost
    except ImportError:
        return None
    try:
        return estimate_image_cost(model, image_size, n)
    except Exception:  # pragma: no cover - defensive, pricing is advisory only
        return None


def _estimate_video_cost(
    model: str,
    duration_seconds: float,
    resolution: str = "720p",
    include_audio: bool = True,
) -> CostEstimateLike | None:
    """Best-effort video cost estimate, or None when pricing is unavailable.

    Video pricing is per second, so multi-segment plans (a clip's beats, a
    chain of extensions) pass their SUMMED duration and describe the split in
    the route's caveats.
    """
    try:
        from .pricing import estimate_video_cost
    except ImportError:
        return None
    try:
        return estimate_video_cost(model, duration_seconds, resolution, include_audio)
    except Exception:  # pragma: no cover - defensive, pricing is advisory only
        return None


# ============================================================================
# Scoring weights
# ============================================================================


@dataclass(frozen=True)
class ScoringWeights:
    """Relative importance of each scoring term.

    The weights sum to 1.0 and every term is normalised to [0, 1], so a route
    score is directly readable as a 0-1 confidence. A term that the request
    does not express an opinion about returns ``NEUTRAL_TERM`` for every
    candidate, so it cancels out instead of distorting the ranking.

    The spread between capability_fit and budget_alignment is the module's
    headline promise ("a demanded capability outranks a budget habit") and is
    therefore sized, not guessed. Two models one tier apart on the cost axis
    differ by at most ``0.5 * budget_alignment``, and the cheaper of them may
    also be the documented default and collect ``default_affinity`` on top;
    the capability gap between the model that owns a capability and the one
    that merely has some of it is ``0.5 * capability_fit``. So the guarantee
    holds only while::

        0.5 * capability_fit - 0.5 * budget_alignment - default_affinity > 0

    At the previous 0.35/0.25/0.05 that expression was exactly 0.0: a "cheap"
    poster brief scored gemini-3-pro-image and gemini-3.1-flash-image at an
    identical 0.525 and the documented winner emerged only from the fidelity
    tie-break in ``_rank``, i.e. by luck. At 0.45/0.25/0.05 it is 0.05 — a
    margin that survives a re-ordering of the candidate list.

    Attributes:
        capability_fit: Heaviest weight — an explicitly demanded capability
            (legible text, character consistency) is the whole reason the
            caller asked for advice, so it must outrank cost habits and the
            default-model nudge by a visible margin, not a tie-break.
        budget_alignment: How closely the model's price tier matches the
            requested budget. Second-heaviest: cost is the most common reason
            a caller regrets a model choice.
        quality_ceiling: Raw output fidelity, consulted when the request is
            quality-led (budget="best", 4K, print; a video 4K ask under
            budget="cheap" is not, since the caller ranked cost first). This is where a
            size/resolution demand is scored — every model still in the race
            already passed the hard size rule, so re-scoring it as a
            capability would only tilt a satisfied demand toward the priciest
            option.
        speed_fit: Turnaround, consulted when the request is draft-led.
        default_affinity: A deliberate thumb on the scale for each tool's
            documented default, so an otherwise-even race resolves to the
            option the rest of the server already assumes. Applied only under
            the balanced budget: a caller who said cheap or best has stated
            the preference the nudge stands in for.
    """

    capability_fit: float = 0.45
    budget_alignment: float = 0.25
    quality_ceiling: float = 0.15
    speed_fit: float = 0.10
    default_affinity: float = 0.05


WEIGHTS = ScoringWeights()

# Value a scoring term returns when the request expresses no preference about
# it. Identical for every candidate, so the term contributes nothing to the
# ordering while keeping scores comparable across requests.
NEUTRAL_TERM = 0.5

# Where each budget preference sits on the 0.0 (cheapest) .. 1.0 (priciest)
# cost axis used by the model profiles.
_BUDGET_TARGET_COST: dict[str, float] = {
    "cheap": 0.0,
    "balanced": 0.5,
    "best": 1.0,
}


# ============================================================================
# Model profiles
# ============================================================================


@dataclass(frozen=True)
class ModelProfile:
    """Relative standing of one model on the axes the router scores.

    The indices are ordinal, not absolute: they encode "pro renders text
    better than flash, which beats lite" rather than any measured number.
    Actual money comes from src/pricing.py.

    Attributes:
        model: Model ID as passed to the tools.
        media_kind: Which planner considers this model.
        cost_index: 0.0 cheapest .. 1.0 priciest.
        fidelity_index: 0.0 lowest .. 1.0 highest output quality.
        speed_index: 0.0 slowest .. 1.0 fastest turnaround.
        text_rendering_index: How reliably it renders legible text (images).
        summary: One-line description used to build route rationales.
    """

    model: str
    media_kind: MediaKind
    cost_index: float
    fidelity_index: float
    speed_index: float
    text_rendering_index: float
    summary: str


@dataclass(frozen=True)
class VideoCapabilities:
    """Hard yes/no capabilities of a video model.

    Every False here is a documented restriction, not a preference, and each
    is turned into an explicit rejection (never a silent drop) by
    ``_VIDEO_CAPABILITY_RULES``.
    """

    supports_first_last_frame: bool
    supports_extension: bool
    supports_reference_images: bool
    supports_4k: bool
    supports_1080p: bool
    # Omni's draft tier. Veo publishes no 360p rate and src/video.py
    # rejects the value, so a 360p ask has to exclude every Veo model rather
    # than reach one and be dropped (or, worse, be quoted at the 720p rate).
    supports_360p: bool
    supports_seed: bool
    supports_negative_prompt: bool
    supports_audio: bool
    supports_conversational_edit: bool
    min_duration_seconds: float
    max_duration_seconds: float


_IMAGE_PROFILES: dict[str, ModelProfile] = {
    "gemini-3.1-flash-lite-image": ModelProfile(
        model="gemini-3.1-flash-lite-image",
        media_kind="image",
        cost_index=0.0,
        fidelity_index=0.35,
        speed_index=1.0,
        # 1K-only output caps how much detail is available for glyphs, so its
        # text rendering is the weakest of the three.
        text_rendering_index=0.15,
        summary="cheapest and fastest Gemini image model, 1K output only",
    ),
    "gemini-3.1-flash-image": ModelProfile(
        model="gemini-3.1-flash-image",
        media_kind="image",
        cost_index=0.5,
        fidelity_index=0.70,
        speed_index=0.70,
        text_rendering_index=0.50,
        summary="balanced default: 1K/2K/4K, up to 14 reference images",
    ),
    "gemini-3-pro-image": ModelProfile(
        model="gemini-3-pro-image",
        media_kind="image",
        cost_index=1.0,
        fidelity_index=1.0,
        speed_index=0.30,
        # The reasoning pass is what makes this the model for precise text.
        text_rendering_index=1.0,
        summary=(
            "most capable: reasoning, precise text rendering, 4K, "
            "thought_signature multi-turn editing"
        ),
    ),
}

_VIDEO_PROFILES: dict[str, ModelProfile] = {
    OMNI_MODEL: ModelProfile(
        model=OMNI_MODEL,
        media_kind="video",
        # Grounded in the published rates, not vibes: the preview is
        # $0.10136/s against Fast's $0.10/s, so it is marginally the DEARER of
        # the two. (Its 1.2 successor, below, is cheaper than both.)
        # It sat at 0.0 — the cheapest slot in the family — which made a
        # budget=cheap request rank a $0.4054 route above a $0.40 one.
        cost_index=0.27,
        fidelity_index=0.30,
        speed_index=1.0,
        text_rendering_index=0.0,
        summary="fastest 720p/24fps drafts with conversational editing",
    ),
    OMNI_1_2_MODEL: ModelProfile(
        model=OMNI_1_2_MODEL,
        media_kind="video",
        # On the same $/s-at-720p scale as every row here, where Veo
        # standard's $0.40/s is 1.0 and Fast's $0.10/s is 0.25: 1.2 bills
        # 5,792 tokens/s at $12/M, $0.069504/s, so 0.069504 / 0.40 = 0.17. It
        # sat at 0.0 once, and the budget term scores distance from the
        # target, so under the default balanced budget (0.5) the cheapest
        # renderer here ranked LAST, level with the $0.40/s tier. Its 360p
        # draft tier is cheaper still, but that is a resolution the caller
        # chooses per call, not a property of the model's standing.
        cost_index=0.17,
        # Above the preview model on every axis that is not price: real 1080p
        # and 4K output, keyframe interpolation, video references and native
        # audio. Below full-fat Veo, which is still the finishing renderer.
        fidelity_index=0.75,
        speed_index=1.0,
        text_rendering_index=0.0,
        summary="3-15s video with native audio, stronger references, conversational editing and upscaling",
    ),
    "veo-3.1-fast-generate-001": ModelProfile(
        model="veo-3.1-fast-generate-001",
        media_kind="video",
        # $0.10/s — half the standard tier, a hair under omni.
        cost_index=0.25,
        fidelity_index=0.85,
        speed_index=0.60,
        text_rendering_index=0.0,
        summary="faster, cheaper Veo 3.1 — the default in the composite tools",
    ),
    "veo-3.1-generate-001": ModelProfile(
        model="veo-3.1-generate-001",
        media_kind="video",
        cost_index=1.0,
        fidelity_index=1.0,
        speed_index=0.30,
        text_rendering_index=0.0,
        summary="highest-fidelity Veo 3.1",
    ),
}

_VIDEO_CAPABILITIES: dict[str, VideoCapabilities] = {
    # Omni is the draft tier: no seed, no negative prompt, 720p only, 3-10s,
    # but it is the ONLY model with conversational multi-turn editing.
    OMNI_MODEL: VideoCapabilities(
        supports_first_last_frame=False,
        supports_extension=False,
        supports_reference_images=True,
        supports_4k=False,
        supports_1080p=False,
        supports_360p=False,
        supports_seed=False,
        supports_negative_prompt=False,
        # 720p preview renders carry no usable audio track; Veo is the model
        # family with native audio.
        supports_audio=False,
        supports_conversational_edit=True,
        min_duration_seconds=float(OMNI_MIN_DURATION),
        max_duration_seconds=float(OMNI_SPECS[OMNI_MODEL].max_duration_seconds),
    ),
    # Omni 1.2: everything the preview model does, plus the four capabilities
    # that used to force a Veo route — resolution control up to 4K, video
    # extension, first/last-frame interpolation and native audio — while
    # keeping conversational editing, which no Veo tier has. Still no seed and
    # no negative_prompt: the reference lists both as unsupported, along with
    # system instructions, temperature and top_p.
    OMNI_1_2_MODEL: VideoCapabilities(
        supports_first_last_frame=True,
        supports_extension=True,
        supports_reference_images=True,
        supports_4k=True,
        supports_1080p=True,
        supports_360p=True,
        supports_seed=False,
        supports_negative_prompt=False,
        # 1.2 generates an audio track natively ("The model generates a video
        # with audio based on your text description") and the prompt guide has
        # a section on directing it. What it has no switch for is turning it
        # OFF — a caveat, not a reason to exclude it when audio is wanted.
        supports_audio=True,
        supports_conversational_edit=True,
        min_duration_seconds=float(OMNI_MIN_DURATION),
        max_duration_seconds=float(OMNI_SPECS[OMNI_1_2_MODEL].max_duration_seconds),
    ),
}

# Both full-fat Veo tiers share one capability set.
_FULL_VEO_CAPABILITIES = VideoCapabilities(
    supports_first_last_frame=True,
    supports_extension=True,
    supports_reference_images=True,
    supports_4k=True,
    supports_1080p=True,
    supports_360p=False,
    supports_seed=True,
    supports_negative_prompt=True,
    # Veo 3.1 generates audio natively, switchable via the generate_audio flag.
    supports_audio=True,
    supports_conversational_edit=False,
    min_duration_seconds=float(min(VEO_DURATIONS_SECONDS)),
    max_duration_seconds=float(VEO_MAX_CLIP_SECONDS),
)
for _model in LIVE_VIDEO_MODELS:
    if _model not in _VIDEO_CAPABILITIES:
        _VIDEO_CAPABILITIES[_model] = _FULL_VEO_CAPABILITIES


def image_profile(model: str) -> ModelProfile | None:
    """Return the scoring profile for an image model, or None if unknown."""
    return _IMAGE_PROFILES.get(model)


def _served_video_model(model: str) -> str:
    """The model a call naming ``model`` runs on.

    Folds a retired omni ID onto its replacement, which is what the omni
    tools render, price and report for it.
    """
    return canonical_omni_model(model) if is_omni_model(model) else model


def video_profile(model: str) -> ModelProfile | None:
    """Return the scoring profile for the video model a call naming ``model``
    runs on, or None if unknown."""
    return _VIDEO_PROFILES.get(_served_video_model(model))


def video_capabilities(model: str) -> VideoCapabilities | None:
    """Return the hard capability set for the video model a call naming
    ``model`` runs on, or None if unknown.

    A call naming the retired gemini-omni-1.1-flash runs on 1.2, so 1.2's
    limits are the ones that apply to it.
    """
    return _VIDEO_CAPABILITIES.get(_served_video_model(model))


# ============================================================================
# Intent inference
# ============================================================================

# Every table below is matched case-insensitively with word-ish boundaries, so
# "short" does not fire on "shortage" and "4k" does not fire on "24k".
# Deliberately ABSENT from these tables: "film", "movie", "scene", "shot" —
# they appear just as often in still-image prompts ("film grain", "a scene of
# a forest") and a false video classification is an expensive mistake.
_VIDEO_TERMS: frozenset[str] = frozenset(
    {
        "animate",
        "animated",
        "animation",
        "b-roll",
        "broll",
        "clip",
        "commercial",
        "cinemagraph",
        "footage",
        "montage",
        "moving image",
        "reel",
        "reels",
        "short",
        "shorts",
        "tiktok",
        "timelapse",
        "time-lapse",
        "trailer",
        "video",
        "vlog",
    }
)

_IMAGE_TERMS: frozenset[str] = frozenset(
    {
        "album cover",
        "artwork",
        "avatar",
        "banner",
        "cover art",
        "drawing",
        "headshot",
        "icon",
        "illustration",
        "image",
        "logo",
        "painting",
        "photo",
        "photograph",
        "picture",
        "portrait",
        "poster",
        "product shot",
        "render",
        "sticker",
        "thumbnail",
        "wallpaper",
    }
)

# Anything that has to come out with readable glyphs. This is the single
# strongest reason to pay for gemini-3-pro-image.
#
# Term matching is whole-word (see _term_pattern), so plurals are listed
# explicitly — "captions"/"labels"/"panels" do NOT match the singular forms,
# which is exactly why a storyboard/caption brief whose whole point is legible
# on-panel text was routing to the weakest text renderer.
_TEXT_RENDERING_TERMS: frozenset[str] = frozenset(
    {
        "banner",
        "brand",
        "branding",
        "caption",
        "captions",
        "chart",
        "diagram",
        "flyer",
        "headline",
        "infographic",
        "label",
        "labels",
        "lettering",
        "logo",
        "lower third",
        "lower-third",
        "lower thirds",
        "lower-thirds",
        "menu",
        "packaging",
        # "panel"/"panels" is deliberately NOT here: a solar panel, control
        # panel or wood panel carries no text, and it over-triggered a stronger
        # (pricier) text renderer for unrelated images. A storyboard's panels
        # are covered by the storyboard signal plus caption/slug-line/label.
        "poster",
        "sign",
        "signage",
        "slide",
        "slug line",
        "slug lines",
        "slugline",
        "sluglines",
        "subtitle",
        "text",
        "title card",
        "typography",
        "ui",
        "wordmark",
        "words",
    }
)

# Output that will be viewed large or printed, i.e. 2K/4K territory.
_HIGH_RESOLUTION_TERMS: frozenset[str] = frozenset(
    {
        "2k",
        "4k",
        "billboard",
        "high resolution",
        "high-res",
        "hi-res",
        "hires",
        "large format",
        "print",
        "printable",
        "retina",
        "wallpaper",
    }
)

# Cheap-and-disposable output: the caller is exploring, not delivering.
_DRAFT_TERMS: frozenset[str] = frozenset(
    {
        "animatic",
        "concept test",
        "draft",
        "first pass",
        "mock up",
        "mockup",
        "placeholder",
        "preview",
        "prototype",
        "quick",
        "rough",
        "scratch",
        "sketch",
        "storyboard",
        "wip",
    }
)

_ITERATION_TERMS: frozenset[str] = frozenset(
    {
        "adjust",
        "again",
        "change it",
        "iterate",
        "iterating",
        "make it",
        "refine",
        "revise",
        "tweak",
        "variation",
        "variations",
    }
)

# Requests that are inherently several shots cut together.
_MULTI_SHOT_TERMS: frozenset[str] = frozenset(
    {
        "ad",
        "advert",
        "beats",
        "commercial",
        "explainer",
        "montage",
        "multi-shot",
        "reel",
        "sequence",
        "series of",
        # NB: bare "short" is deliberately absent. It reads as the noun ("a
        # short") only rarely; "short video" / "short clip" means brief, and
        # treating it as multi-shot turned a single 4s render into a 3-beat
        # clip at 3x the price. "shorts" (the format) is unambiguous and stays.
        "shorts",
        "storyboard",
        "tiktok",
        "trailer",
    }
)

# Previsualization before committing to renders: the caller wants to see the
# shots as cheap stills first. These drive the generate_storyboard route — the
# intended step between an idea and generate_clip. Multi-word phrases anchor as
# whole phrases (see _term_pattern), so "before committing" matches "before
# committing to renders" but not "committing".
_STORYBOARD_TERMS: frozenset[str] = frozenset(
    {
        "before committing",
        "before rendering",
        "keyframe",
        "keyframes",
        "mock up the shots",
        "mockup the shots",
        "panel",
        "panels",
        "plan the shots",
        "pre-vis",
        "previsualize",
        "previz",
        "shot sheet",
        "shot sheets",
        "shotlist",
        "shotlists",
        "storyboard",
        "storyboards",
    }
)

_EXTENSION_TERMS: frozenset[str] = frozenset(
    {
        "continue",
        "extend",
        "keep going",
        "lengthen",
        "longer",
        "loop",
        "looping",
        "seamless loop",
    }
)

_AUDIO_TERMS: frozenset[str] = frozenset(
    {
        "ambience",
        "audio",
        "dialogue",
        "music",
        "narration",
        "sfx",
        "sound",
        "sound effects",
        "soundtrack",
        "speech",
        "voice",
        "voiceover",
    }
)

_CHEAP_TERMS: frozenset[str] = frozenset(
    {
        "as cheap as possible",
        "budget",
        "cheap",
        "cheapest",
        "cost-effective",
        "economical",
        "inexpensive",
        "low cost",
        "low-cost",
        "save money",
    }
)

_BEST_TERMS: frozenset[str] = frozenset(
    {
        "best quality",
        "final",
        "flagship",
        "hero",
        "highest quality",
        "master",
        "premium",
        "production ready",
        "production-ready",
        "top quality",
    }
)

# Camera and cinematography vocabulary. Deliberately NOT in _VIDEO_TERMS: a
# strong video hit wins outright over an image word, and "a cinematic portrait"
# or "a wide shot of the logo" are stills. These sit in the weaker tier with
# transitions and runtimes — enough to settle a brief that names no medium at
# all ("a cinematic drone shot over a coastline", which planned as an IMAGE),
# and not enough to overrule someone who said "photo".
#
# Every entry is a phrase rather than a bare word for the same reason: "shot"
# alone matches "a shot of espresso" and "headshot".
_CAMERA_TERMS: frozenset[str] = frozenset(
    {
        "aerial shot",
        "camera move",
        "camera moves",
        "camera pan",
        "camera pans",
        "cinematic",
        "close-up shot",
        "continuous shot",
        "crane shot",
        "dolly in",
        "dolly out",
        "dolly shot",
        "drone shot",
        "establishing shot",
        "handheld shot",
        "one take",
        "pans across",
        "pov shot",
        "single take",
        "slow motion",
        "slow-motion",
        "tracking shot",
        "unbroken shot",
        "zoom in",
        "zoom out",
    }
)

# Phrases that name the SOURCE of a continuation as a prior interaction
# rather than a file. Only omni can resolve one: loop_extend needs a Veo
# video_uri and cannot take an interaction_id, so a brief that says "continuing
# from the interaction I already rendered" was ranking a top route that could
# not run at all.
_INTERACTION_SOURCE_TERMS: frozenset[str] = frozenset(
    {
        "already rendered",
        "already generated",
        "interaction",
        "interaction_id",
        "i just generated",
        "i just made",
        "i just rendered",
        "previous interaction",
        "previously generated",
        "the clip you made",
        "the video you made",
    }
)

_TRANSITION_TERMS: frozenset[str] = frozenset(
    {
        "bridge",
        "cross-fade",
        "crossfade",
        "in-between",
        "morph",
        "transition",
        "tween",
    }
)

_BRIDGE_TERMS: frozenset[str] = frozenset(
    {
        "between the clips",
        "between two clips",
        "between the two clips",
        "bridge",
        "join the clips",
        "splice",
    }
)

_REFERENCE_TERMS: frozenset[str] = frozenset(
    {
        "character consistency",
        "consistent character",
        "reference image",
        "reference images",
        "same character",
        "same person",
        "same product",
    }
)

_SEED_TERMS: frozenset[str] = frozenset(
    {"deterministic", "reproducible", "reproducibility", "same seed", "seed"}
)

_NEGATIVE_PROMPT_TERMS: frozenset[str] = frozenset(
    {"avoid", "negative prompt", "no text", "without any"}
)

_GCS_TERMS: frozenset[str] = frozenset(
    {"bucket", "cloud storage", "gcs", "gs://", "output_gcs_uri"}
)

# Vertical-format vocabulary. Two entries used to be bare words with a common
# non-format meaning, and both silently set aspect_ratio="9:16":
#
# * "story"/"stories" -- "an illustration telling the story of a rabbit" is a
#   narrative, not an Instagram Story, so the platform sense has to be spelled
#   out to count;
# * "short" -- "a short video of a cat" is a duration. "shorts" (YouTube
#   Shorts) stays, because the plural has no such reading.
#
# "reel"/"reels", "shorts" and "tiktok" already carry most of this signal, so
# qualifying these costs almost nothing and stops a wrong frame shape being
# chosen from a word the caller did not mean as a format.
_VERTICAL_TERMS: frozenset[str] = frozenset(
    {
        "9:16",
        "portrait",
        "reel",
        "reels",
        "shorts",
        "tiktok",
        "vertical",
        "facebook stories",
        "facebook story",
        "ig stories",
        "ig story",
        "instagram stories",
        "instagram story",
        "snapchat stories",
        "snapchat story",
        "stories format",
        "story format",
    }
)

_HORIZONTAL_TERMS: frozenset[str] = frozenset(
    {"16:9", "horizontal", "landscape", "widescreen", "youtube"}
)

_SQUARE_TERMS: frozenset[str] = frozenset({"1:1", "square"})

# "8s", "8 sec", "8-second", "30 seconds" — the number of seconds of output.
# The unit is captured because the single-letter spelling is ambiguous and has
# to earn its reading (see _duration_value).
_DURATION_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:-|\s)?\s*(s|sec|secs|second|seconds)\b"
)

# "2 minutes", "90-min" — the same fact in the other unit. Worth parsing
# precisely because a runtime in minutes is exactly the request that cannot be
# served by one Veo clip, and saying so is more useful than quietly planning 8
# seconds of video.
_MINUTES_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:-|\s)?\s*(m|min|mins|minute|minutes)\b"
)

# "2 minute 30 second", "2 min 30 sec", "2 minutes and 30 seconds" -- one
# runtime spelled in two units, which is the ONLY case in which the two should
# be summed. The first fix for the dropped-minutes bug summed whenever both
# units appeared anywhere in the intent, and any unrelated minute mention
# became hundreds of seconds of planned runtime: "a 6 second clip of a cat, I
# need it in 5 minutes" read as 306s and planned a 20-turn extension chain at
# $14.00 for a 6s clip. Adjacency is what makes "2 minute 30 second" one
# number; a deadline, a render-time budget or a second deliverable is not.
_COMPOUND_DURATION_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:-|\s)?\s*(?:m|min|mins|minute|minutes)\b"
    r"\s*(?:,\s*|and\s+)?"
    r"(\d+(?:\.\d+)?)\s*(?:-|\s)?\s*(?:s|sec|secs|second|seconds)\b"
)

# "by another 30 seconds", "add 20s", "extend by 2 minutes" — an amount to
# ADD, not a total. Without this, the plain duration pattern takes the first
# number in the sentence, so "extend my 8 second clip by another 30 seconds"
# read the SOURCE length and planned one 7s extension instead of five.
# The leading lookbehind is load-bearing: without it "a lullaby 30 seconds
# long" matched the "by 30 seconds" buried inside "lullaby" and planned a 30s
# extension of a clip nobody asked to extend.
_ADDED_DURATION_PATTERN = re.compile(
    r"(?<!\w)(?:by|add|plus|another)\s+(?:another\s+)?(\d+(?:\.\d+)?)\s*"
    # "by 3 more seconds" is the same delta; without this it matched nothing,
    # and the extension went out at the tool's 10s default.
    r"(?:(?:more|extra|additional)\s+)?"
    r"(?:-|\s)?\s*(s\b|sec\b|secs\b|second\b|seconds\b"
    r"|m\b|min\b|mins\b|minute\b|minutes\b)"
    # An optional adjacent seconds part, so "by another 2 minutes 30 seconds"
    # is 150s. Without it the delta read the minutes alone (120s) while the
    # total read the compound (150s): two signals from one phrase disagreeing,
    # and loop_extend planning 18 turns where 20 were asked for.
    r"(?:\s*(?:,\s*|and\s+)?(\d+(?:\.\d+)?)\s*(?:-|\s)?\s*"
    r"(?:s|sec|secs|second|seconds)\b)?"
)

# The single-letter units are the ambiguous ones: "8s" is a runtime but "the
# 1970s", "an 80s diner", "our 100s of customers", "the iPhone 15s" and "a 3 m
# tall robot" are not, and every one of them was being planned as a duration
# (1970s quoted a 247-beat clip at $197.60). A bare unit therefore only counts
# when the surrounding words make it a runtime; the spelled-out units
# (sec/second/min/minute) are unambiguous and need no such proof.
_BARE_DURATION_UNITS: frozenset[str] = frozenset({"s", "m"})

# Nouns that can only be describing the thing being timed, so a bare unit in
# front of one is a runtime: "8s clip", "30s of video". The second example is
# the one the lookup used to fail -- _word_after saw "of" and stopped -- so a
# single linking word may sit between the unit and the noun.
_DURATION_LINK_WORDS: frozenset[str] = frozenset({"of"})

_DURATION_CONTEXT_WORDS: frozenset[str] = frozenset(
    {
        "ad",
        "advert",
        "animatic",
        "animation",
        "b-roll",
        "beat",
        "beats",
        "broll",
        "clip",
        "clips",
        "commercial",
        "cut",
        "cuts",
        "film",
        "footage",
        "loop",
        "long",
        "montage",
        "movie",
        "reel",
        "reels",
        "render",
        "scene",
        "scenes",
        "segment",
        "sequence",
        "shot",
        "shots",
        "spot",
        "take",
        "trailer",
        "video",
        "videos",
        "vlog",
    }
)

# Words that announce a duration is coming, so a bare unit behind one is a
# runtime even with no noun after it: "for 15s", "make it 8s".
_DURATION_CUE_WORDS: frozenset[str] = frozenset(
    {
        "about",
        "approximately",
        "around",
        "duration",
        "exactly",
        "for",
        "it",
        "lasting",
        "length",
        "over",
        "roughly",
        "runtime",
        "under",
    }
)

# A four-digit number carrying a trailing "s" is a decade, never a runtime:
# "1970s footage" is a look, not 1970 seconds of output.
_DECADE_PATTERN = re.compile(r"^(?:19|20)\d{2}$")

_WORD_BEFORE_PATTERN = re.compile(r"([\w'-]+)\W*$")
_WORD_AFTER_PATTERN = re.compile(r"^\W*([\w'-]+)")


def _word_before(text: str, index: int) -> str:
    """The word ending immediately before ``index``, or "" if there is none."""
    match = _WORD_BEFORE_PATTERN.search(text[:index])
    return match.group(1) if match else ""


def _word_after(text: str, index: int) -> str:
    """The word starting immediately after ``index``, or "" if there is none."""
    match = _WORD_AFTER_PATTERN.search(text[index:])
    return match.group(1) if match else ""


def _context_word_after(text: str, index: int) -> str:
    """The noun a bare duration unit is describing, skipping one link word.

    "30s of video": the word after the unit is "of", and the noun that makes
    it a runtime is the one after that.
    """
    tail = text[index:]
    first = _WORD_AFTER_PATTERN.search(tail)
    if not first:
        return ""
    if first.group(1) not in _DURATION_LINK_WORDS:
        return first.group(1)
    second = _WORD_AFTER_PATTERN.search(tail[first.end() :])
    return second.group(1) if second else ""


def _duration_value(text: str, pattern: re.Pattern[str]) -> float | None:
    """First number in ``text`` that is really a duration, in its own unit.

    The spelled-out units are taken at face value. A bare "s"/"m" has to be
    corroborated by the words around it, because the same characters spell
    decades, product names and plural quantities — and every one of those was
    being planned as a runtime.
    """
    for match in pattern.finditer(text):
        if match.group(2) in _BARE_DURATION_UNITS:
            if _DECADE_PATTERN.match(match.group(1)):
                continue
            if not (
                _context_word_after(text, match.end()) in _DURATION_CONTEXT_WORDS
                or _word_before(text, match.start()) in _DURATION_CUE_WORDS
            ):
                continue
        return float(match.group(1))
    return None


# "3 beats", "5 shots", "4 scenes", "4 keyframe panels" — how many segments the
# output has. A storyboard names its segments panels/keyframes/frames exactly as
# a clip names them beats/shots, so all count the same thing; without them "4
# keyframe panels" fell through to the 3-beat default and planned the wrong
# board size. The digit must sit immediately before the noun (so "a 4k panel" is
# a resolution, not 4 shots), and "N frames per second" is a frame rate, not a
# shot count.
#
# The leading lookbehind is the other half of that rule, and it was missing: a
# digit that is part of a LARGER number is not a count either. Guarding only
# the right-hand side caught "a 4k panel" (the trailing `k` fails the noun) but
# not "a 9:16 shot of a dancer", which matched the `16` and planned a 16-beat
# clip quoted at $9.60 in place of one 6s render (~$0.60). "16:9 shot" read as
# 9. Text is lower-cased before matching, so `x` covers "1920x1080" too.
_BEAT_PATTERN = re.compile(
    r"(?<![\dx:.])(\d+)\s*(?:beats?|shots?|scenes?|cuts?|segments?|panels?"
    # A frame RATE is not a shot count, however it is spelled: "per second",
    # "per sec", "/second", "-per-second", "/s". The first guard knew only the
    # first spelling, so "a 24 frames per sec animation" planned 20 beats at
    # $12.00 in place of one $0.60 render.
    # ... and "a second", "every second", "each second", "per minute": the
    # guard was an enumeration of one word, and "24 frames a second" still
    # planned 20 paid beats one word away from the spelling it knew.
    r"|keyframes?|frames?(?!\s*(?:-\s*)?(?:(?:per|a|every|each)[\s\-]*|/\s*)"
    r"(?:sec(?:ond)?s?|s|min(?:ute)?s?)\b))\b"
)

# "up to 6 reference images", "3 reference photos". Same lookbehind as the beat
# count, and for the same reason: "9:16 reference images" is one aspect ratio,
# not 16 references.
_REFERENCE_COUNT_PATTERN = re.compile(
    r"(?<![\dx:.])(\d+)\s*(?:reference|ref)\s*(?:images?|photos?)"
)

_TERM_PATTERNS: dict[str, re.Pattern[str]] = {}


def _term_pattern(term: str) -> re.Pattern[str]:
    """Compile (and memoise) a boundary-anchored pattern for one keyword.

    Lookarounds rather than ``\\b`` so terms that start or end with
    punctuation (``gs://``, ``b-roll``, ``9:16``) still anchor correctly.

    Each assertion is applied only where the term's OWN edge is a word
    character, which is the half that was missing. Wrapping every term in
    ``(?!\\w)`` unconditionally meant ``gs://`` required a non-word character
    after the final slash -- and a real URI has a letter there, so the term
    could never match one. ``infer_signals("render to gs://renders/out.mp4")``
    reported ``wants_gcs_output=False``, which silently disabled the
    ``gcs_output_on_gemini_api`` conflict and, on Vertex, the ``output_gcs_uri``
    advice; it only looked like it worked when the URI happened to contain the
    separate term ``bucket``. ``b-roll`` and ``9:16`` begin and end with word
    characters, so they still anchor on both sides.
    """
    pattern = _TERM_PATTERNS.get(term)
    if pattern is None:
        lead = r"(?<!\w)" if term[:1] and re.match(r"\w", term[0]) else ""
        trail = r"(?!\w)" if term[-1:] and re.match(r"\w", term[-1]) else ""
        pattern = re.compile(rf"{lead}{re.escape(term)}{trail}")
        _TERM_PATTERNS[term] = pattern
    return pattern


# Words that flip the meaning of a keyword immediately following them.
# "no audio" must not be read as a request FOR audio.
_NEGATORS: tuple[str, ...] = (
    "no",
    "not",
    "without",
    "never",
    "avoid",
    "exclude",
    "sans",
    "minus",
    "drop",
    "remove",
    "skip",
)

# A verb is negated by an auxiliary as often as by "no": "but don't upscale
# it", "we won't need to upscale it". Without these the upscale check read
# both as requests, chained a paid upscale onto the render and cut the
# render's prompt at "but don't". ("do not" is already caught by "not".)
_VERB_NEGATORS: tuple[str, ...] = _NEGATORS + (
    "don't",
    "dont",
    "doesn't",
    "doesnt",
    "won't",
    "wont",
    "shouldn't",
    "shouldnt",
    "can't",
    "cant",
    "cannot",
    "mustn't",
    "mustnt",
    "needn't",
    "neednt",
)

# Negators that can only be talking about the soundtrack. "a silent video of
# rain" is a video request with the sound off, but a negator scoped to nothing
# read "silent" as cancelling "video" and planned a still image — 5 of 15
# natural "video without X" phrasings lost their modality this way.
_AUDIO_NEGATORS: tuple[str, ...] = _NEGATORS + ("silent", "mute", "muted")

# How far back to look for a negator. Short on purpose: "no dialogue, but
# ambient music" should keep the music hit, so only the words immediately
# before a term can negate it.
_NEGATION_WINDOW = 3

# How much text is handed to that check, in WORDS. _negator_precedes reads
# only the last _NEGATION_WINDOW words of the current clause, but it was handed
# the ENTIRE prefix and re-split it on every match -- so a fully negated intent
# cost O(matches x length): "no video " x 6000 took 2.6s, x 12000 took 9.9s,
# inline on the event loop. A first fix handed it a 120-character window, and a
# single token longer than that between the negator and the term cut the
# negator off ("no " + 130-char hashtag + " video" stopped reading as
# negated). Stepping back one whitespace at a time is exact whatever the token
# lengths, and still linear: each step costs the length of one word.
_NEGATION_LOOKBACK_WORDS = _NEGATION_WINDOW + 1

# The longest intent the planner will read. It is pure string work, but it is
# not free -- see _NEGATION_LOOKBACK_CHARS -- and nothing above this length is
# a description of something to generate.
MAX_INTENT_CHARS = 20_000

# A negator only reaches a term while nothing separates them. Clause and
# sentence punctuation ends the scan — "no audio, a video of rain" and "avoid
# text. video montage of ocean waves" are asking FOR video, and reading the
# negator across the break planned stills for both.
_CLAUSE_BREAK_PATTERN = re.compile(r"[,.;:!?()\[\]\"]|\s[-–—]+\s")

# Conjunctions start a new clause even without punctuation, so "without music
# and video of a river" must not lose its video.
_CLAUSE_BREAK_WORDS: frozenset[str] = frozenset(
    {"and", "but", "or", "nor", "then", "so", "yet", "while", "plus"}
)


def _negator_precedes(prefix: str, negators: tuple[str, ...]) -> bool:
    """Whether a negator sits in the words directly before a term.

    ``prefix`` is everything ahead of the term. Only the current clause is
    read, and only its last ``_NEGATION_WINDOW`` words.
    """
    clause = _CLAUSE_BREAK_PATTERN.split(prefix)[-1]
    for word in reversed(clause.split()[-_NEGATION_WINDOW:]):
        # A typographic apostrophe ("don’t") is folded to a plain one.
        cleaned = word.replace("\u2019", "'").strip("-\"'")
        if cleaned in _CLAUSE_BREAK_WORDS:
            return False
        if cleaned in negators:
            return True
    return False


def _is_negated(text: str, term: str, negators: tuple[str, ...] | None = None) -> bool:
    """Whether every occurrence of ``term`` in ``text`` sits under a negator.

    Keyword matching alone read "no audio" as an audio request, which flipped
    both the emitted include_audio and the model ranking. Requiring EVERY
    occurrence to be negated is the conservative reading: a sentence that
    mentions the term both ways keeps the positive signal.
    """
    if negators is None:
        negators = _AUDIO_NEGATORS if term in _AUDIO_TERMS else _NEGATORS
    pattern = _term_pattern(term)
    found = False
    for match in pattern.finditer(text):
        found = True
        start = match.start()
        low = start
        for _ in range(_NEGATION_LOOKBACK_WORDS):
            # One whitespace back per word; stop at the start of the text.
            # Always lands on a boundary, so no token is ever cut mid-way.
            boundary = text.rfind(" ", 0, max(0, low - 1))
            if boundary < 0:
                low = 0
                break
            low = boundary
        if not _negator_precedes(text[low:start], negators):
            return False
    return found


def _matched_terms(text: str, terms: frozenset[str]) -> tuple[str, ...]:
    """Return the sorted subset of ``terms`` present in ``text``.

    Sorted so the result is reproducible regardless of set iteration order —
    the whole module's determinism guarantee depends on details like this.
    """
    return tuple(
        sorted(
            term
            for term in terms
            if _term_pattern(term).search(text) and not _is_negated(text, term)
        )
    )


@dataclass(frozen=True)
class IntentSignals:
    """What the free-text intent appears to be asking for.

    Every field is a guess. Explicit ``RoutingConstraints`` always win; see
    ``resolve_request``. ``matched_terms`` is kept so a caller (or a test) can
    see exactly which words drove a decision.
    """

    media_kind: MediaKind | None = None
    wants_text_rendering: bool = False
    wants_high_resolution: bool = False
    wants_draft: bool = False
    wants_iteration: bool = False
    wants_multi_shot: bool = False
    wants_storyboard: bool = False
    wants_extension: bool = False
    wants_audio: bool = False
    wants_transition: bool = False
    names_interaction_source: bool = False
    wants_bridge: bool = False
    wants_reference_consistency: bool = False
    wants_seed: bool = False
    wants_negative_prompt: bool = False
    wants_gcs_output: bool = False
    wants_cheap: bool = False
    wants_best: bool = False
    aspect_ratio: str | None = None
    duration_seconds: float | None = None
    # Seconds to ADD, when the intent asked for an amount rather than a total
    # ("by another 30 seconds"). Distinct from duration_seconds because an
    # extension request names a delta, and reading it as a total plans one
    # 7s extension for a 30s ask.
    added_duration_seconds: float | None = None
    beat_count: int | None = None
    reference_image_count: int | None = None
    matched_terms: tuple[str, ...] = ()


def infer_signals(intent: str) -> IntentSignals:
    """Derive routing signals from a free-text description.

    Matching is deliberately conservative: exact keywords with word
    boundaries, no stemming and no fuzzy matching, because a wrong inference
    silently sends the caller to the wrong model. Ambiguous words are omitted
    from the tables entirely rather than guessed at.

    Args:
        intent: Natural-language description of what the caller wants to make.

    Returns:
        The inferred signals, with ``matched_terms`` listing every keyword
        that fired.
    """
    text = intent.lower()

    video_hits = _matched_terms(text, _VIDEO_TERMS)
    image_hits = _matched_terms(text, _IMAGE_TERMS)
    text_hits = _matched_terms(text, _TEXT_RENDERING_TERMS)
    highres_hits = _matched_terms(text, _HIGH_RESOLUTION_TERMS)
    draft_hits = _matched_terms(text, _DRAFT_TERMS)
    iteration_hits = _matched_terms(text, _ITERATION_TERMS)
    multi_hits = _matched_terms(text, _MULTI_SHOT_TERMS)
    storyboard_hits = _matched_terms(text, _STORYBOARD_TERMS)
    extension_hits = _matched_terms(text, _EXTENSION_TERMS)
    audio_hits = _matched_terms(text, _AUDIO_TERMS)
    cheap_hits = _matched_terms(text, _CHEAP_TERMS)
    best_hits = _matched_terms(text, _BEST_TERMS)
    transition_hits = _matched_terms(text, _TRANSITION_TERMS)
    camera_hits = _matched_terms(text, _CAMERA_TERMS)
    interaction_hits = _matched_terms(text, _INTERACTION_SOURCE_TERMS)
    bridge_hits = _matched_terms(text, _BRIDGE_TERMS)
    reference_hits = _matched_terms(text, _REFERENCE_TERMS)
    seed_hits = _matched_terms(text, _SEED_TERMS)
    negative_hits = _matched_terms(text, _NEGATIVE_PROMPT_TERMS)
    gcs_hits = _matched_terms(text, _GCS_TERMS)
    vertical_hits = _matched_terms(text, _VERTICAL_TERMS)
    horizontal_hits = _matched_terms(text, _HORIZONTAL_TERMS)
    square_hits = _matched_terms(text, _SQUARE_TERMS)

    seconds_value = _duration_value(text, _DURATION_PATTERN)
    minutes_value = _duration_value(text, _MINUTES_PATTERN)
    added_match = _ADDED_DURATION_PATTERN.search(text)
    added_duration: float | None = None
    if added_match is not None:
        added_value = float(added_match.group(1))
        added_duration = (
            added_value * 60.0 if added_match.group(2).startswith("m") else added_value
        )
        if added_match.group(3) is not None and added_match.group(2).startswith("m"):
            # "2 minutes 30 seconds": the trailing part only exists after a
            # minutes unit -- "30 seconds 5 seconds" is not a compound.
            added_duration += float(added_match.group(3))
    # A runtime can name BOTH units, and seconds winning outright dropped the
    # minutes on the floor: "a 2 minute 30 second trailer" read as 30s, so a
    # 2.5-minute brief was planned as one sub-8s render with no extension
    # ceiling handling at all. Summed only when the two are ADJACENT (see
    # _COMPOUND_DURATION_PATTERN); otherwise seconds keep winning, since a
    # stray minute figure elsewhere is a deadline or a budget, not a runtime.
    compound = _COMPOUND_DURATION_PATTERN.search(text)
    if compound is not None:
        duration = float(compound.group(1)) * 60.0 + float(compound.group(2))
    elif seconds_value is not None:
        duration = seconds_value
    elif minutes_value is not None:
        duration = minutes_value * 60.0
    else:
        duration = None

    # Motion words settle the media kind: the video vocabulary is specific
    # enough that a hit is almost never incidental, whereas image words
    # ("poster style", "photographic") show up inside video briefs all the
    # time. Failing those, vocabulary that only makes sense over time —
    # transitions, extensions, a runtime in seconds — is treated as weaker
    # video evidence. Audio words are deliberately NOT evidence: "a photo of a
    # music festival" is still a photo. No hits at all leaves the kind
    # undecided for the caller (or the default) to fill in.
    media_kind: MediaKind | None = None
    if video_hits:
        media_kind = "video"
    elif image_hits:
        media_kind = "image"
    elif (
        transition_hits
        or bridge_hits
        or extension_hits
        or camera_hits
        or duration is not None
    ):
        media_kind = "video"

    aspect_ratio: str | None = None
    if vertical_hits:
        aspect_ratio = "9:16"
    elif horizontal_hits:
        aspect_ratio = "16:9"
    elif square_hits:
        aspect_ratio = "1:1"

    beat_match = _BEAT_PATTERN.search(text)
    beat_count = int(beat_match.group(1)) if beat_match else None

    reference_match = _REFERENCE_COUNT_PATTERN.search(text)
    reference_count = int(reference_match.group(1)) if reference_match else None

    matched = tuple(
        sorted(
            set(
                video_hits
                + camera_hits
                + image_hits
                + text_hits
                + highres_hits
                + draft_hits
                + iteration_hits
                + multi_hits
                + storyboard_hits
                + extension_hits
                + audio_hits
                + cheap_hits
                + best_hits
                + transition_hits
                + bridge_hits
                + reference_hits
                + seed_hits
                + negative_hits
                + gcs_hits
                + vertical_hits
                + horizontal_hits
                + square_hits
            )
        )
    )

    return IntentSignals(
        media_kind=media_kind,
        wants_text_rendering=bool(text_hits),
        wants_high_resolution=bool(highres_hits),
        wants_draft=bool(draft_hits),
        wants_iteration=bool(iteration_hits),
        # An explicit beat/shot count is itself a multi-shot signal.
        wants_multi_shot=bool(multi_hits)
        or (beat_count is not None and beat_count > 1),
        wants_storyboard=bool(storyboard_hits),
        wants_extension=bool(extension_hits),
        wants_audio=bool(audio_hits),
        wants_transition=bool(transition_hits),
        names_interaction_source=bool(interaction_hits),
        wants_bridge=bool(bridge_hits),
        wants_reference_consistency=bool(reference_hits) or reference_count is not None,
        wants_seed=bool(seed_hits),
        wants_negative_prompt=bool(negative_hits),
        wants_gcs_output=bool(gcs_hits),
        wants_cheap=bool(cheap_hits),
        wants_best=bool(best_hits),
        aspect_ratio=aspect_ratio,
        duration_seconds=duration,
        added_duration_seconds=added_duration,
        beat_count=beat_count,
        reference_image_count=reference_count,
        matched_terms=matched,
    )


# ============================================================================
# Constraints and the resolved request
# ============================================================================


@dataclass(frozen=True)
class RoutingConstraints:
    """Structured facts the caller already knows.

    Every field defaults to None meaning "no opinion, infer it". A value that
    IS supplied always beats whatever the intent text suggested — the caller
    knows things the prose does not say.

    Attributes:
        budget: Cost sensitivity: "cheap", "balanced" or "best".
        media_kind: Force image or video routing.
        needs_text_rendering: Output must contain legible text.
        needs_4k: Output must be 4K (image size or video resolution).
        image_size: Exact image size ("1K"/"2K"/"4K").
        aspect_ratio: Exact aspect ratio, e.g. "16:9", "9:16", "1:1".
        num_reference_images: How many reference images will be supplied.
        needs_audio: The video must carry a controllable audio track.
        duration_seconds: Total desired runtime of the video.
        has_first_frame / has_last_frame: Whether frame stills are available.
            Inferred from first_frame_uri/last_frame_uri when not given.
        first_frame_uri / last_frame_uri: Frame URIs, folded into the emitted
            parameter dicts when present.
        source_video_uri: Existing clip to extend or bridge from.
        needs_extension: The output must continue an existing video.
        wants_gcs_output: Output must land in Cloud Storage.
        backend: Which backend the server is running against. Several rules
            are backend-specific (GCS output, Veo, which runs on Vertex AI
            only).
        gemini_api_key_available: Whether the server holds a GEMINI_API_KEY.
            None means unknown. On a Vertex deployment it decides which
            backend omni runs on, and so which upload ceiling an omni
            extension is priced against.
        video_gcs_bucket_configured: Whether the server has a
            VIDEO_GCS_BUCKET, which a Veo extension on Vertex AI is written
            to when no output_gcs_uri is given; without either it is
            refused. None means unknown.
        num_beats: How many distinct shots the output has.
        is_draft: This render is throwaway.
        is_iterating: The caller is refining an existing result.
        needs_seed: Reproducibility via a fixed seed is required.
        needs_negative_prompt: A negative prompt is required.
        resolution: Exact video resolution ("360p"/"720p"/"1080p"/"4K");
            360p exists on Omni 1.2 only.
        num_images: How many images to generate (cost estimation only).
        pinned_model: A model the caller insists on. Rules still apply, but a
            violation is reported as a conflict rather than a quiet swap.
        previous_interaction_id: An omni interaction to keep editing.
        previous_interaction_model: The omni model recorded for
            previous_interaction_id, as the server's sidecar has it; None when
            nothing is recorded. An interaction cannot change models, so a
            continuation is planned on this model alone, and one recorded on
            a retired model is not planned at all.
        previous_interaction_backend: The backend the interaction was
            recorded on ("vertex" / "gemini_api"); None when unknown. An
            interaction lives on one backend, so a continuation is planned
            there, and named there when the server's OMNI_BACKEND pin is
            the other one. Ignored without previous_interaction_id, as are
            the two below and the model.
        previous_interaction_resolution: The resolution the interaction
            rendered at, as recorded; an upscale starts from it.
        previous_interaction_video_url: Where the interaction's clip was
            saved, as recorded: the input_video_uri a conflict offers when
            the interaction itself cannot be continued.
        previous_interaction_seconds: The interaction's clip length, as
            recorded; None when unknown. An extension measures a runtime
            total against it and prices its turns from it, as
            extend_video_omni's own quote does.
    """

    budget: BudgetPreference | None = None
    media_kind: MediaKind | None = None
    needs_text_rendering: bool | None = None
    needs_4k: bool | None = None
    image_size: ImageSize | None = None
    aspect_ratio: str | None = None
    num_reference_images: int | None = None
    needs_audio: bool | None = None
    duration_seconds: float | None = None
    has_first_frame: bool | None = None
    has_last_frame: bool | None = None
    first_frame_uri: str | None = None
    last_frame_uri: str | None = None
    source_video_uri: str | None = None
    needs_extension: bool | None = None
    wants_gcs_output: bool | None = None
    backend: Backend = "unknown"
    gemini_api_key_available: bool | None = None
    video_gcs_bucket_configured: bool | None = None
    # The server's OMNI_BACKEND pin, when it has one ("vertex" / "gemini_api");
    # None means "auto". Omni's backend is otherwise inferred from the two
    # fields above, and a pinned server would price the wrong ceiling.
    omni_backend: str | None = None
    num_beats: int | None = None
    is_draft: bool | None = None
    is_iterating: bool | None = None
    needs_seed: bool | None = None
    needs_negative_prompt: bool | None = None
    resolution: str | None = None
    num_images: int | None = None
    pinned_model: str | None = None
    previous_interaction_id: str | None = None
    previous_interaction_model: str | None = None
    previous_interaction_backend: str | None = None
    previous_interaction_resolution: str | None = None
    previous_interaction_video_url: str | None = None
    previous_interaction_seconds: float | None = None

    def __post_init__(self) -> None:
        """Reject impossible constraint values up front.

        A bad enum value here would otherwise surface much later as a mystery
        route, so it is a hard error in the style of the impl modules.
        """
        if self.budget is not None and self.budget not in get_args(BudgetPreference):
            raise ValueError(
                f"Unsupported budget '{self.budget}'. "
                f"Supported values are {', '.join(get_args(BudgetPreference))}."
            )
        if self.backend not in get_args(Backend):
            raise ValueError(
                f"Unsupported backend '{self.backend}'. "
                f"Supported values are {', '.join(get_args(Backend))}."
            )
        if self.media_kind is not None and self.media_kind not in get_args(MediaKind):
            raise ValueError(
                f"Unsupported media_kind '{self.media_kind}'. "
                f"Supported values are {', '.join(get_args(MediaKind))}."
            )
        if self.image_size is not None and self.image_size not in IMAGE_SIZES:
            raise ValueError(
                f"Unsupported image_size '{self.image_size}'. "
                f"Supported values are {', '.join(IMAGE_SIZES)}."
            )
        if self.resolution is not None and self.resolution not in VIDEO_RESOLUTIONS:
            raise ValueError(
                f"Unsupported resolution '{self.resolution}'. "
                f"Supported values are {', '.join(VIDEO_RESOLUTIONS)}."
            )
        for name in ("num_reference_images", "num_beats", "num_images"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}.")
        if self.duration_seconds is not None and (
            not math.isfinite(self.duration_seconds) or self.duration_seconds <= 0
        ):
            # Non-finite values need an explicit check: every comparison with
            # NaN is False, so it sailed past `<= 0`, reached the cost math,
            # and serialized as bare NaN — invalid JSON — in the plan.
            # Infinity got further still, overflowing the loop_extend
            # times calculation. Same rule as _validate_duration_seconds in
            # the generation tools.
            raise ValueError(
                "duration_seconds must be a positive finite number, got "
                f"{self.duration_seconds!r}."
            )
        if self.previous_interaction_model is not None and not is_omni_model(
            self.previous_interaction_model
        ):
            # Only omni models hold interactions. Anything else would plan a
            # continuation on a model the tools cannot call.
            raise ValueError(
                "previous_interaction_model must be an omni model, got "
                f"'{self.previous_interaction_model}'."
            )
        if self.previous_interaction_backend not in (None, *_INTERACTION_BACKENDS):
            raise ValueError(
                "previous_interaction_backend must be one of "
                f"{', '.join(_INTERACTION_BACKENDS)}, got "
                f"'{self.previous_interaction_backend}'."
            )
        if (
            self.previous_interaction_resolution is not None
            and self.previous_interaction_resolution not in VIDEO_RESOLUTIONS
        ):
            raise ValueError(
                "previous_interaction_resolution must be one of "
                f"{', '.join(VIDEO_RESOLUTIONS)}, got "
                f"'{self.previous_interaction_resolution}'."
            )
        if self.previous_interaction_seconds is not None and not (
            math.isfinite(self.previous_interaction_seconds)
            and self.previous_interaction_seconds > 0
        ):
            raise ValueError(
                "previous_interaction_seconds must be a positive finite number, "
                f"got {self.previous_interaction_seconds!r}."
            )


@dataclass(frozen=True)
class ResolvedRequest:
    """Constraints merged over inferences — the input the planners actually see.

    Nothing here is optional-because-unknown any more: every field is the
    final decision, so the planners never have to re-apply the override rules.
    """

    intent: str
    media_kind: MediaKind
    budget: BudgetPreference
    backend: Backend
    omni_backend: str | None
    # None when the caller did not say, i.e. a Gemini-API-only model might
    # work; False is a positive statement that it cannot.
    gemini_api_key_available: bool | None
    video_gcs_bucket_configured: bool | None
    needs_text_rendering: bool
    needs_4k: bool
    image_size: str
    num_images: int
    aspect_ratio: str | None
    num_reference_images: int
    needs_audio: bool
    clip_duration_seconds: float
    total_duration_seconds: float | None
    has_first_frame: bool
    has_last_frame: bool
    first_frame_uri: str | None
    last_frame_uri: str | None
    source_video_uri: str | None
    needs_extension: bool
    # Seconds the caller asked to ADD, when they named an amount rather than
    # a total runtime. Drives the extension count directly.
    added_duration_seconds: float | None
    # True when extension was implied by a long runtime rather than asked
    # for. An implied extension has no source video yet, so the plan must
    # lead with a seed render; an explicit one references a clip the
    # caller already has.
    extension_implied: bool
    wants_bridge: bool
    wants_transition: bool
    names_interaction_source: bool
    wants_gcs_output: bool
    wants_storyboard: bool
    num_beats: int
    is_draft: bool
    is_iterating: bool
    needs_seed: bool
    needs_negative_prompt: bool
    resolution: str | None
    pinned_model: str | None
    previous_interaction_id: str | None
    # As recorded, a retired ID included: folding it onto its replacement
    # would plan a continuation on a model that never held the interaction.
    previous_interaction_model: str | None = None
    previous_interaction_backend: str | None = None
    previous_interaction_resolution: str | None = None
    previous_interaction_video_url: str | None = None
    previous_interaction_seconds: float | None = None
    # Whether budget=cheap was asked for (the budget argument, or a draft)
    # rather than read off a word in the brief: "a restaurant going from
    # cheap then upscale" describes the restaurant, and must not drop the
    # render to the 360p tier.
    budget_stated: bool = False


def _first_not_none(*values: Any) -> Any:
    """Return the first argument that is not None (None if all are)."""
    for value in values:
        if value is not None:
            return value
    return None


def _snap_veo_duration(seconds: float) -> int:
    """Snap a duration to the nearest length Veo accepts (4/6/8s).

    Mirrors the snap in src/video.py, including its tie behaviour (``min``
    keeps the first of two equally-near options), so the router reports the
    duration the API will actually produce.
    """
    return min(VEO_DURATIONS_SECONDS, key=lambda option: abs(option - seconds))


def resolve_request(
    intent: str,
    signals: IntentSignals,
    constraints: RoutingConstraints | None,
) -> ResolvedRequest:
    """Merge explicit constraints over inferred signals.

    Explicit always wins, including explicit False: a caller that says
    ``needs_text_rendering=False`` about a poster has overruled the keyword,
    and quietly re-inferring it would make constraints untrustworthy.

    Args:
        intent: The original free-text description (kept for prompt fields).
        signals: Inferences from ``infer_signals``.
        constraints: Caller-supplied facts, or None.

    Returns:
        The fully resolved request.
    """
    given = constraints or RoutingConstraints()

    # A Gemini-API render reports its model under a `-preview` Veo id, and the
    # generation tools accept that spelling as an alias of the canonical `-001`
    # model so a returned id round-trips. Normalize a pinned preview id to its
    # canonical name up front — BEFORE the live-model check in _plan_video —
    # so pinning a preview id is planned as its canonical model instead of
    # rejected as "not a live video model". Source of truth for the mapping is
    # video._CANONICAL_VIDEO_MODEL_IDS; a non-preview id passes through
    # unchanged.
    pinned_model = given.pinned_model
    if pinned_model is not None:
        pinned_model = _CANONICAL_VIDEO_MODEL_IDS.get(pinned_model, pinned_model)
        # A retired omni ID likewise plans as the model the omni tools run it
        # on; plan_generation notes the redirect.
        if is_omni_model(pinned_model):
            pinned_model = canonical_omni_model(pinned_model)

    # Media kind: explicit, then inferred from the text, then inferred from
    # constraints that only exist for video (frames to interpolate between, a
    # clip to extend, an omni interaction to edit), then image — the cheaper
    # and faster failure mode when the request is genuinely ambiguous.
    structurally_video = any(
        (
            given.has_first_frame,
            given.has_last_frame,
            given.first_frame_uri is not None,
            given.last_frame_uri is not None,
            given.source_video_uri is not None,
            given.previous_interaction_id is not None,
            given.needs_extension,
            given.resolution is not None,
            given.duration_seconds is not None,
            given.num_beats is not None and given.num_beats > 1,
            given.needs_audio,
        )
    )
    media_kind: MediaKind = _first_not_none(given.media_kind, signals.media_kind) or (
        "video" if structurally_video else "image"
    )

    is_draft = _first_not_none(given.is_draft, signals.wants_draft) or False
    is_iterating = _first_not_none(given.is_iterating, signals.wants_iteration) or False

    # Budget: explicit, then the cheap/best keywords, then a draft-aware
    # default — a throwaway render should not be billed at hero prices.
    budget: BudgetPreference
    if given.budget is not None:
        budget = given.budget
    elif signals.wants_best:
        budget = "best"
    elif signals.wants_cheap or is_draft:
        budget = "cheap"
    else:
        budget = "balanced"

    needs_text_rendering = (
        _first_not_none(given.needs_text_rendering, signals.wants_text_rendering)
        or False
    )
    needs_4k = _first_not_none(
        given.needs_4k,
        given.image_size == "4K" if given.image_size is not None else None,
        given.resolution == "4K" if given.resolution is not None else None,
        # "4k" in the text is a direct request; other high-res words only
        # justify 2K, which is handled below.
        "4k" in signals.matched_terms or None,
    )
    needs_4k = bool(needs_4k)

    # Image size: explicit wins; else 4K when asked for by name; else 2K for
    # the softer print/large-format words; else the 1K default.
    if given.image_size is not None:
        image_size = str(given.image_size)
    elif needs_4k:
        image_size = "4K"
    elif signals.wants_high_resolution:
        image_size = "2K"
    else:
        image_size = "1K"

    aspect_ratio = _first_not_none(given.aspect_ratio, signals.aspect_ratio)

    num_reference_images = (
        _first_not_none(given.num_reference_images, signals.reference_image_count) or 0
    )

    needs_audio = _first_not_none(given.needs_audio, signals.wants_audio) or False

    # Duration: the caller's number is the TOTAL runtime. Per-clip length is
    # that value capped at the longest single render available — anything
    # longer has to come from extensions or several beats.
    #
    # Capped at MAX_SINGLE_RENDER_SECONDS rather than Veo's 8s so an ask some
    # model renders in one call stays a one-render request. A Veo route does
    # not render it in one call: up to VEO_SNAP_MAX_SECONDS _video_params
    # snaps it into the 4/6/8s ladder and says so in a caveat, and the ranker
    # does not re-rank for that shortfall (a 9-10s ask can still lead with a
    # Veo route at 8s). Past it, _plan_video plans the Veo route as a seed
    # render extended with loop_extend (_veo_chains_runtime), and ranks a
    # route that renders the whole runtime in one call above that chain.
    total_duration = _first_not_none(given.duration_seconds, signals.duration_seconds)
    if total_duration is None:
        clip_duration = DEFAULT_VIDEO_DURATION_SECONDS
    elif total_duration > MAX_SINGLE_RENDER_SECONDS:
        clip_duration = float(VEO_MAX_CLIP_SECONDS)
    else:
        clip_duration = float(total_duration)

    num_beats = _first_not_none(given.num_beats, signals.beat_count)
    if num_beats is None:
        if total_duration is not None and total_duration <= VEO_MAX_CLIP_SECONDS:
            # A runtime one render already covers. Splitting it into beats
            # multiplies the bill and invents placeholder prompts for shots
            # the caller never described; if they want cuts inside 8s they
            # can say how many.
            num_beats = 1
        elif signals.wants_multi_shot:
            # A multi-shot brief with no stated count: enough beats to cover
            # the requested runtime, or the default storyboard length.
            if total_duration is not None and total_duration > VEO_MAX_CLIP_SECONDS:
                num_beats = math.ceil(total_duration / VEO_MAX_CLIP_SECONDS)
            else:
                num_beats = DEFAULT_BEAT_COUNT
        else:
            num_beats = 1

    has_first_frame = _first_not_none(
        given.has_first_frame, given.first_frame_uri is not None or None, False
    )
    has_last_frame = _first_not_none(
        given.has_last_frame, given.last_frame_uri is not None or None, False
    )

    # Extension: explicit, else the "longer/loop/continue" words, else implied
    # by a single-shot request longer than any single render can cover.
    #
    # Measured against MAX_SINGLE_RENDER_SECONDS, not Veo's 8s: an implied
    # extension routes to loop_extend / extend_video_omni, both of which need
    # an existing clip, so implying one for a 10s ask offered the caller
    # nothing they could run. A caller who does want a chain can still ask for
    # it -- `needs_extension` and the "longer/continue" vocabulary both still
    # win here.
    #
    # Never for a transition: its last frame has to end the shot, and an
    # extension carries the shot on past it, so a seed-and-extend chain is
    # not a longer transition but a different video. A transition no model
    # interpolates in one render is reported as such (_plan_video).
    transition = (
        has_first_frame
        and has_last_frame
        or (num_beats <= 1 and (signals.wants_transition or signals.wants_bridge))
    )
    implied_extension = (
        total_duration is not None
        and total_duration > MAX_SINGLE_RENDER_SECONDS
        and num_beats <= 1
        and not transition
    )
    needs_extension = _first_not_none(
        given.needs_extension,
        signals.wants_extension or implied_extension or None,
        False,
    )

    wants_gcs_output = (
        _first_not_none(given.wants_gcs_output, signals.wants_gcs_output) or False
    )

    resolution = given.resolution
    if resolution is None and needs_4k and media_kind == "video":
        resolution = "4K"
    continuing = given.previous_interaction_id is not None

    return ResolvedRequest(
        intent=intent,
        media_kind=media_kind,
        budget=budget,
        backend=given.backend,
        omni_backend=given.omni_backend,
        gemini_api_key_available=given.gemini_api_key_available,
        video_gcs_bucket_configured=given.video_gcs_bucket_configured,
        needs_text_rendering=bool(needs_text_rendering),
        needs_4k=needs_4k,
        image_size=image_size,
        num_images=given.num_images if given.num_images is not None else 1,
        aspect_ratio=aspect_ratio,
        num_reference_images=num_reference_images,
        needs_audio=bool(needs_audio),
        clip_duration_seconds=clip_duration,
        total_duration_seconds=(
            float(total_duration) if total_duration is not None else None
        ),
        has_first_frame=bool(has_first_frame),
        has_last_frame=bool(has_last_frame),
        first_frame_uri=given.first_frame_uri,
        last_frame_uri=given.last_frame_uri,
        source_video_uri=given.source_video_uri,
        needs_extension=bool(needs_extension),
        added_duration_seconds=signals.added_duration_seconds,
        extension_implied=bool(
            needs_extension
            and implied_extension
            and not (given.needs_extension or signals.wants_extension)
        ),
        wants_bridge=signals.wants_bridge,
        wants_transition=signals.wants_transition,
        names_interaction_source=signals.names_interaction_source,
        wants_gcs_output=bool(wants_gcs_output),
        wants_storyboard=signals.wants_storyboard,
        num_beats=int(num_beats),
        is_draft=bool(is_draft),
        is_iterating=bool(is_iterating),
        needs_seed=_first_not_none(given.needs_seed, signals.wants_seed) or False,
        needs_negative_prompt=(
            _first_not_none(given.needs_negative_prompt, signals.wants_negative_prompt)
            or False
        ),
        resolution=resolution,
        pinned_model=pinned_model,
        previous_interaction_id=given.previous_interaction_id,
        # Meaningless without the id they describe.
        previous_interaction_model=(
            given.previous_interaction_model if continuing else None
        ),
        previous_interaction_backend=(
            given.previous_interaction_backend if continuing else None
        ),
        previous_interaction_resolution=(
            given.previous_interaction_resolution if continuing else None
        ),
        previous_interaction_video_url=(
            given.previous_interaction_video_url if continuing else None
        ),
        previous_interaction_seconds=(
            given.previous_interaction_seconds if continuing else None
        ),
        budget_stated=given.budget is not None or is_draft,
    )


# ============================================================================
# Plan result types
# ============================================================================


@dataclass(frozen=True)
class RoutedCall:
    """One concrete, ready-to-issue generation call.

    Attributes:
        tool: MCP tool to call.
        model: Model ID to pass to it.
        params: Parameter dict the caller can forward as-is. Keys the router
            cannot know (frame/video URIs it was not given) are omitted and
            called out in ``caveats`` rather than filled with placeholders.
        score: Confidence in [0, 1] — see ``ScoringWeights``.
        rationale: Why this route was chosen, in one sentence.
        caveats: Things that will bite the caller if ignored.
        cost: Pre-flight cost estimate, or None when pricing is unavailable.
    """

    tool: ToolName
    model: str
    params: dict[str, Any]
    score: float
    rationale: str
    caveats: tuple[str, ...] = ()
    cost: CostEstimateLike | None = None


@dataclass(frozen=True)
class RejectedRoute:
    """A model that was considered and excluded, with the reason.

    The reason is the product here: "cannot produce 4K" teaches the caller
    something a silently shortened candidate list never would.
    """

    model: str
    reason: str
    tool: ToolName | None = None


@dataclass(frozen=True)
class RoutingConflict:
    """A contradiction in the request itself.

    Surfaced instead of returning a plan that is guaranteed to fail at call
    time.

    Attributes:
        code: Stable machine-readable identifier (see the module's rule list).
        detail: What is contradictory.
        resolution: The concrete change that would make the request work.
    """

    code: str
    detail: str
    resolution: str


@dataclass(frozen=True)
class WorkflowStep:
    """One step of a recommended multi-call workflow (e.g. animatic first)."""

    order: int
    tool: ToolName
    params: dict[str, Any]
    rationale: str


@dataclass(frozen=True)
class RoutingPlan:
    """The router's answer.

    Attributes:
        intent: The original description.
        media_kind: Image or video, after constraints and inference.
        signals: What was inferred from the text.
        request: The resolved request the planners used.
        routes: Candidate calls, best first.
        rejected: Every excluded option, with its reason.
        conflicts: Contradictions in the request as stated.
        workflow: Recommended call sequence when one call is not the whole
            answer (e.g. a fast animatic pass before an expensive clip).
        notes: Non-blocking observations about the routing itself.
    """

    intent: str
    media_kind: MediaKind
    signals: IntentSignals
    request: ResolvedRequest
    routes: tuple[RoutedCall, ...]
    rejected: tuple[RejectedRoute, ...] = ()
    conflicts: tuple[RoutingConflict, ...] = ()
    workflow: tuple[WorkflowStep, ...] = ()
    notes: tuple[str, ...] = ()
    # A step the request needs was refused, though routes for the rest
    # survived: a render that its upscale cannot take.
    workflow_blocked: bool = False

    @property
    def recommended(self) -> RoutedCall | None:
        """The top-ranked route, or None when nothing can satisfy the request."""
        return self.routes[0] if self.routes else None

    @property
    def is_satisfiable(self) -> bool:
        """Whether a route survived every rule, and no step the request needs
        was refused."""
        return bool(self.routes) and not self.workflow_blocked


# ============================================================================
# Scoring
# ============================================================================


def _score_route(
    profile: ModelProfile,
    request: ResolvedRequest,
    *,
    demanded_capabilities: tuple[float, ...],
    quality_is_demanded: bool,
    default_model: str,
    cost_index_override: float | None = None,
) -> float:
    """Score one candidate model against the resolved request.

    Each term is normalised to [0, 1] and multiplied by its weight; the
    weights sum to 1.0 so the result reads as a confidence. Terms the request
    has no opinion about return ``NEUTRAL_TERM`` for every candidate and
    therefore cannot change the ordering.

    Args:
        profile: The candidate's scoring profile.
        request: The resolved request.
        demanded_capabilities: Capability indices this request actually cares
            about (empty means "no opinion").
        quality_is_demanded: Whether the request is quality-led.
        default_model: The documented default for this media kind.
        cost_index_override: Cost position to score budget alignment against
            instead of ``profile.cost_index``. Used only for a storyboard
            route, whose 1K keyframe board is cheap on any image model, so the
            keyframe model's delivery-cost tier must not be what places the
            previz in the ranking.

    Returns:
        The score, rounded to 4 decimals so equal inputs compare equal.
    """
    target_cost = _BUDGET_TARGET_COST[request.budget]
    cost_index = (
        profile.cost_index if cost_index_override is None else cost_index_override
    )
    budget_term = 1.0 - abs(cost_index - target_cost)

    capability_term = (
        sum(demanded_capabilities) / len(demanded_capabilities)
        if demanded_capabilities
        else NEUTRAL_TERM
    )

    quality_term = profile.fidelity_index if quality_is_demanded else NEUTRAL_TERM

    # Speed only matters when the caller said they are drafting or iterating;
    # otherwise a fast model has no advantage worth ranking on.
    speed_is_demanded = request.is_draft or request.is_iterating
    speed_term = profile.speed_index if speed_is_demanded else NEUTRAL_TERM

    # The nudge settles an otherwise-even race, and a stated budget is not
    # one. On the $/s scale the cheap video models sit close together (Omni
    # 1.2 at 0.17, Veo Fast at 0.25), so a 30% cheaper route gains only 0.02
    # on budget, and the default's 0.05 made budget=cheap recommend the
    # dearer default.
    default_term = (
        1.0 if profile.model == default_model and request.budget == "balanced" else 0.0
    )

    score = (
        WEIGHTS.capability_fit * capability_term
        + WEIGHTS.budget_alignment * budget_term
        + WEIGHTS.quality_ceiling * quality_term
        + WEIGHTS.speed_fit * speed_term
        + WEIGHTS.default_affinity * default_term
    )
    return round(min(max(score, 0.0), 1.0), 4)


def _rank(
    routes: list[RoutedCall], profiles: dict[str, ModelProfile]
) -> tuple[RoutedCall, ...]:
    """Order routes best-first, deterministically.

    A model with a published shutdown date ranks after every model without
    one, whatever it scores. A deprecated model offered as an equal choice is
    how a caller ends up building on an endpoint with weeks left, and a tie
    break was not enough: it held only while the preview had a GA twin that
    scored level with it (gemini-omni-1.1-flash), and the balanced budget
    scores the preview's price ABOVE its real replacement's, 1.2. Then the
    score, then fidelity (the more capable model is the safer default when
    the score cannot separate them) and then the model ID, so the ordering is
    a total order that never depends on insertion or set iteration order.
    """

    def sort_key(route: RoutedCall) -> tuple[bool, float, float, str, str]:
        profile = profiles.get(route.model)
        fidelity = profile.fidelity_index if profile else 0.0
        return (
            route.model in DEPRECATED_VIDEO_MODELS,
            -route.score,
            -fidelity,
            route.model,
            route.tool,
        )

    return tuple(sorted(routes, key=sort_key))


# ============================================================================
# Image planning
# ============================================================================


def _plan_image(
    request: ResolvedRequest,
) -> tuple[
    tuple[RoutedCall, ...],
    tuple[RejectedRoute, ...],
    tuple[RoutingConflict, ...],
    tuple[str, ...],
]:
    """Build the ranked image routes plus rejections, conflicts and notes."""
    routes: list[RoutedCall] = []
    rejected: list[RejectedRoute] = []
    conflicts: list[RoutingConflict] = []
    notes: list[str] = []

    # A pinned ID may be superseded rather than invalid. generate_image would
    # reroute it and serve the request, so honour the pin against the model
    # that would actually run instead of calling the request unroutable.
    pinned = request.pinned_model
    if pinned is not None and pinned not in LIVE_IMAGE_MODELS:
        resolved, resolve_warnings, _ = resolve_image_model(pinned)
        if resolved in LIVE_IMAGE_MODELS:
            notes.extend(resolve_warnings)
            notes.append(
                f"pinned_model={pinned} is superseded; planned against "
                f"{resolved}, which is what generate_image would call."
            )
            pinned = resolved
        else:
            conflicts.append(
                RoutingConflict(
                    code="pinned_model_not_routable",
                    detail=(
                        f"pinned_model={pinned} is not a live image model "
                        f"({', '.join(LIVE_IMAGE_MODELS)})."
                    ),
                    resolution=(
                        "Superseded IDs are still accepted by generate_image and "
                        f"rerouted, but plan against {DEFAULT_IMAGE_MODEL} instead."
                    ),
                )
            )
            pinned = None

    for model in sorted(LIVE_IMAGE_MODELS):
        profile = _IMAGE_PROFILES[model]

        # RULE image-size: the only hard image capability rule. Imported from
        # image.py so it cannot drift from what generate_image enforces.
        if not _supports_image_size(model, request.image_size):
            supported = ", ".join(sorted(_IMAGE_SIZE_SUPPORT[model]))
            reason = (
                f"{model} excluded: cannot produce {request.image_size} "
                f"(supported: {supported})."
            )
            rejected.append(
                RejectedRoute(model=model, reason=reason, tool="generate_image")
            )
            if pinned == model:
                conflicts.append(
                    RoutingConflict(
                        code="image_size_unsupported_by_pinned_model",
                        detail=reason,
                        resolution=(
                            f"Request image_size={supported} on {model}, or switch "
                            f"to {DEFAULT_IMAGE_MODEL}/gemini-3-pro-image for "
                            f"{request.image_size}."
                        ),
                    )
                )
            continue

        # Capability terms the request actually asked about. Text rendering is
        # a soft preference, not a filter: every Gemini 3.x image model can
        # put words on a canvas, they just differ in how legible the result is.
        demanded: list[float] = []
        reasons: list[str] = []
        if request.needs_text_rendering:
            demanded.append(profile.text_rendering_index)
            reasons.append("text has to be legible")
        if request.num_reference_images > 6:
            # All three take up to 14 references, but keeping many subjects
            # coherent is a fidelity problem.
            demanded.append(profile.fidelity_index)
            reasons.append(f"{request.num_reference_images} reference images")
        if request.image_size != "1K":
            # Named as a reason, but deliberately NOT a capability term: the
            # hard size rule above already dropped every model that cannot
            # render this size, so each survivor satisfies the demand in full.
            # Scoring the survivors on fidelity here double-counted
            # quality_ceiling (quality_is_demanded switches on for exactly this
            # case) and handed the priciest model an advantage on a demand both
            # models meet — which is why raising capability_fit would otherwise
            # have moved a plain 4K brief off the documented default.
            reasons.append(f"{request.image_size} output")

        quality_is_demanded = (
            request.budget == "best" or request.needs_4k or request.image_size != "1K"
        )
        score = _score_route(
            profile,
            request,
            demanded_capabilities=tuple(demanded),
            quality_is_demanded=quality_is_demanded,
            default_model=DEFAULT_IMAGE_MODEL,
        )

        caveats: list[str] = []
        if request.backend == "vertex":
            caveats.append(
                "Gemini 3.x image models require Vertex location='global'; the "
                "server swaps in a global-location client automatically."
            )
        if request.num_reference_images > MAX_IMAGE_REFERENCE_IMAGES:
            caveats.append(
                f"Only the first {MAX_IMAGE_REFERENCE_IMAGES} reference images are "
                f"used; {request.num_reference_images} were declared."
            )
        if request.needs_text_rendering and model != "gemini-3-pro-image":
            caveats.append(
                "gemini-3-pro-image is the model documented for precise text "
                "rendering; expect to re-roll glyphs on this one."
            )
        if model == "gemini-3-pro-image":
            caveats.append(
                "Returns a thought_signature_url — pass it back to continue "
                "editing in the same session."
            )

        params: dict[str, Any] = {
            "prompt": request.intent,
            "model": model,
            "image_size": request.image_size,
        }
        if request.aspect_ratio is not None:
            params["aspect_ratio"] = request.aspect_ratio

        rationale = f"{model}: {profile.summary}"
        if reasons:
            rationale += f"; picked for {', '.join(reasons)}"
        rationale += f"; budget={request.budget}"

        routes.append(
            RoutedCall(
                tool="generate_image",
                model=model,
                params=params,
                score=score,
                rationale=rationale,
                caveats=tuple(caveats),
                cost=_estimate_image_cost(
                    model, request.image_size, request.num_images
                ),
            )
        )

    if request.is_iterating and not request.needs_text_rendering:
        notes.append(
            "Iterating: generate_image returns a thought_signature_url — reuse it "
            "for multi-turn edits instead of regenerating from scratch."
        )

    # A storyboard's deliverable — keyframe stills plus a PNG contact sheet — is
    # itself images, so a board brief that resolved to media_kind=image (a
    # "storyboard contact sheet with N panels") must still be able to reach
    # generate_storyboard; without this it saw only generate_image and the board
    # tool was unreachable for the exact request it was built for. Gated inside
    # the helper on the storyboard signal, so a plain image request (no board
    # vocabulary) gets no storyboard route.
    storyboard_route = _plan_storyboard_route(request)
    if storyboard_route is not None:
        routes.append(storyboard_route)

    ranked = _rank(routes, _IMAGE_PROFILES)

    # A pin is a requirement, not a preference: when the pinned model survived
    # the rules, it is the only plan. Recommending something else would answer
    # a question the caller did not ask. If the pin was excluded, the ranked
    # alternatives stay — the conflict already explains why the pin failed.
    if pinned is not None:
        pinned_routes = tuple(r for r in ranked if r.model == pinned)
        if pinned_routes:
            for other in ranked:
                if other.model != pinned:
                    rejected.append(
                        RejectedRoute(
                            model=other.model,
                            reason=(
                                f"{other.model} not planned: pinned_model="
                                f"{pinned} was requested."
                            ),
                            tool=other.tool,
                        )
                    )
            ranked = pinned_routes

    return (
        ranked,
        tuple(rejected),
        tuple(conflicts),
        tuple(notes),
    )


# ============================================================================
# Video planning
# ============================================================================


@dataclass(frozen=True)
class VideoNeeds:
    """The hard capabilities a video request requires of a model."""

    first_last_frame: bool
    extension: bool
    reference_images: bool
    four_k: bool
    hd_1080p: bool
    draft_360p: bool
    seed: bool
    negative_prompt: bool
    audio: bool
    conversational_edit: bool
    clip_duration_seconds: float


@dataclass(frozen=True)
class _CapabilityRule:
    """One "the model simply cannot do this" rule.

    Table-driven so every rule has exactly one place to live and one message,
    and so the whole rule set is enumerable by tests.
    """

    code: str
    need_attr: str
    capability_attr: str
    reason: str
    resolution: str


# The complete set of hard video capability rules. Each fires only when the
# request needs the feature AND the model lacks it, and every firing produces
# a RejectedRoute (plus a RoutingConflict when the caller pinned that model).
_VIDEO_CAPABILITY_RULES: tuple[_CapabilityRule, ...] = (
    _CapabilityRule(
        code="first_last_frame_unsupported",
        need_attr="first_last_frame",
        capability_attr="supports_first_last_frame",
        reason=(
            "{model} excluded: no first+last-frame control, which "
            "generate_transition/generate_bridge are built on"
        ),
        resolution=(
            "Use veo-3.1-fast-generate-001 or veo-3.1-generate-001 for "
            "transitions and bridges."
        ),
    ),
    _CapabilityRule(
        code="extension_unsupported",
        need_attr="extension",
        capability_attr="supports_extension",
        reason="{model} excluded: cannot extend an existing video",
        resolution=(
            "Extend with veo-3.1-fast-generate-001 or veo-3.1-generate-001 via "
            "loop_extend."
        ),
    ),
    _CapabilityRule(
        code="reference_images_unsupported",
        need_attr="reference_images",
        capability_attr="supports_reference_images",
        reason="{model} excluded: does not accept reference images",
        resolution=(
            "Use veo-3.1-fast-generate-001 or veo-3.1-generate-001 (up to 3 "
            "reference images) for subject preservation."
        ),
    ),
    _CapabilityRule(
        code="4k_unsupported",
        need_attr="four_k",
        capability_attr="supports_4k",
        reason="{model} excluded: cannot produce 4K video",
        resolution="Request 4K on veo-3.1-fast-generate-001 or veo-3.1-generate-001.",
    ),
    _CapabilityRule(
        code="360p_unsupported",
        need_attr="draft_360p",
        capability_attr="supports_360p",
        reason=(
            "{model} excluded: 360p is the Omni models' draft tier; "
            "this model publishes neither the output nor a rate for it"
        ),
        resolution=(
            f"Render the 360p draft on {DEFAULT_OMNI_MODEL} via "
            "generate_video_omni, then finalize at 720p or above."
        ),
    ),
    _CapabilityRule(
        code="1080p_unsupported",
        need_attr="hd_1080p",
        capability_attr="supports_1080p",
        reason=("{model} excluded: renders 720p/24fps only, so 1080p is not available"),
        resolution="Use a Veo model for 1080p or 4K output.",
    ),
    _CapabilityRule(
        code="seed_unsupported",
        need_attr="seed",
        capability_attr="supports_seed",
        reason="{model} excluded: has no seed parameter, so runs are not reproducible",
        resolution="Use a Veo model when a fixed seed is required.",
    ),
    _CapabilityRule(
        code="negative_prompt_unsupported",
        need_attr="negative_prompt",
        capability_attr="supports_negative_prompt",
        reason="{model} excluded: has no negative_prompt parameter",
        resolution="Use a Veo model when a negative prompt is required.",
    ),
    _CapabilityRule(
        code="audio_unsupported",
        need_attr="audio",
        capability_attr="supports_audio",
        reason="{model} excluded: its 720p preview renders carry no audio track",
        resolution=(
            "Veo 3.1 generates audio natively; on Vertex AI include_audio maps "
            "to the generate_audio flag."
        ),
    ),
    _CapabilityRule(
        code="conversational_edit_unsupported",
        need_attr="conversational_edit",
        capability_attr="supports_conversational_edit",
        reason=(
            "{model} excluded: no conversational editing — Veo re-renders from "
            "the prompt instead of amending the previous result"
        ),
        resolution=(
            f"Conversational edits run on the Omni models ({DEFAULT_OMNI_MODEL} "
            "by default) via edit_video with the prior interaction_id."
        ),
    ),
)


def _capability_rejection(
    model: str, needs: VideoNeeds
) -> tuple[_CapabilityRule, str] | None:
    """Return the first hard rule ``model`` violates, or None.

    Only the first is reported: once a model is out, listing every additional
    reason is noise rather than expertise. The backend is not a capability:
    Veo on the Gemini Developer API is refused whatever it is asked for, which
    _backend_rejection reports with the fix that applies.
    """
    capabilities = _VIDEO_CAPABILITIES.get(model)
    if capabilities is None:
        return None
    for rule in _VIDEO_CAPABILITY_RULES:
        if getattr(needs, rule.need_attr) and not getattr(
            capabilities, rule.capability_attr
        ):
            return rule, rule.reason.format(model=model) + "."
    return None


def _duration_rejection(model: str, needs: VideoNeeds) -> str | None:
    """Return a reason when the clip length is outside the model's range.

    The omni models are the only ones with a real range here (3-10s, 3-15s on
    1.2); Veo snaps instead of failing, so a Veo mismatch is a caveat rather
    than a rejection.
    """
    capabilities = _VIDEO_CAPABILITIES.get(model)
    if capabilities is None or not is_omni_model(model):
        return None
    if needs.extension:
        # An extension's per-turn length is the service's choice, not the
        # caller's, and the duration on the request describes the FINISHED
        # clip. Measuring a 30s target against omni's 10s per-render cap
        # rejected the one model that can actually reach 30s by extending.
        return None
    duration = needs.clip_duration_seconds
    if duration < capabilities.min_duration_seconds:
        return (
            f"{model} excluded: clips shorter than "
            f"{capabilities.min_duration_seconds:g}s are not supported "
            f"({duration:g}s requested)."
        )
    if duration > capabilities.max_duration_seconds:
        return (
            f"{model} excluded: clips are capped at "
            f"{capabilities.max_duration_seconds:g}s "
            f"({duration:g}s requested)."
        )
    return None


# The tools that render one shot between two endpoints.
_TRANSITION_TOOLS: tuple[ToolName, ...] = ("generate_transition", "generate_bridge")


def _transition_limit(model: str) -> tuple[float, float]:
    """(longest transition ``model`` renders, longest ask it is planned for).

    An omni model interpolates up to its keyframe ceiling. Veo renders 8s and
    snaps an ask up to VEO_SNAP_MAX_SECONDS onto it with a caveat, as a plain
    render does.
    """
    if is_omni_model(model):
        longest = float(omni_spec(model).max_keyframe_seconds)
        return longest, longest
    return float(VEO_MAX_CLIP_SECONDS), VEO_SNAP_MAX_SECONDS


def _transition_too_long(
    tool: ToolName, model: str, request: ResolvedRequest
) -> str | None:
    """A reason when ``model`` cannot render this transition at its length.

    A transition cannot be lengthened by extending it: the extension carries
    the shot on past its last frame. So a model whose longest interpolation
    falls short of the ask is excluded, not planned short -- a 12s ask was
    ranked first as an 8s Veo transition.
    """
    seconds = request.total_duration_seconds
    if tool not in _TRANSITION_TOOLS or seconds is None:
        return None
    longest, allowed = _transition_limit(model)
    if seconds <= allowed:
        return None
    return (
        f"{model} excluded: it interpolates between two endpoints in one "
        f"render of at most {longest:g}s ({seconds:g}s requested), and "
        "extending that render would carry the shot past its last frame."
    )


def _longest_transition_conflict(
    tool: ToolName, request: ResolvedRequest, candidates: list[str]
) -> RoutingConflict | None:
    """A conflict when no candidate renders the transition at its length."""
    seconds = request.total_duration_seconds
    if tool not in _TRANSITION_TOOLS or seconds is None:
        return None
    live = [
        model
        for model in candidates
        if not (model in DEPRECATED_VIDEO_MODELS and omni_preview_is_sunset())
        and _VIDEO_CAPABILITIES[model].supports_first_last_frame
    ]
    if not live or any(seconds <= _transition_limit(m)[1] for m in live):
        return None
    longest = max(live, key=lambda model: _transition_limit(model)[0])
    limit = _transition_limit(longest)[0]
    return RoutingConflict(
        code="transition_too_long",
        detail=(
            f"No model renders a transition {seconds:g}s long: the longest "
            f"interpolation between two endpoints is {limit:g}s, on {longest}, "
            "and extending one would carry the shot past its last frame."
        ),
        resolution=(
            f"Ask for {limit:g}s or less. For a longer shot, render the "
            "transition and then extend it (extend_video_omni, or loop_extend "
            "on Veo); the last frame then falls inside the shot, not at its end."
        ),
    )


def _omni_model_rendering(seconds: float, excluding: str) -> str | None:
    """Another omni model that renders ``seconds`` in one call today, or None.

    Skips a model that is switched off, so the fix offered for one exclusion
    is never a model the same plan excludes.
    """
    for other in sorted(_VIDEO_CAPABILITIES):
        capabilities = _VIDEO_CAPABILITIES[other]
        if (
            other != excluding
            and is_omni_model(other)
            and not (other in DEPRECATED_VIDEO_MODELS and omni_preview_is_sunset())
            and capabilities.min_duration_seconds
            <= seconds
            <= capabilities.max_duration_seconds
        ):
            return other
    return None


# What a clip that can no longer be continued is passed to instead, per
# continuation: the tool that takes it as input_video_uri, and the verb. An
# edit goes to generate_video_omni, because edit_video takes no input video.
# Mirrors src/__main__._retired_interaction_refusal, so the plan and the
# tool's refusal give the same way out.
_SAVED_CLIP_TOOLS: dict[str, tuple[ToolName, str]] = {
    "edit": ("generate_video_omni", "edit"),
    "extend": ("extend_video_omni", "extend"),
    "upscale": ("upscale_video_omni", "upscale"),
}


def _saved_clip_resolution(task: str, request: ResolvedRequest) -> str:
    """How to work on a clip whose interaction cannot be continued.

    Names the saved clip when the server recorded where it went, as the
    tool's own refusal does, so the fix is a value to pass rather than one
    to go and find.
    """
    tool, verb = _SAVED_CLIP_TOOLS[task]
    saved = request.previous_interaction_video_url
    source = (
        f"input_video_uri='{saved}' (the clip it saved)"
        if saved
        else "the clip it saved (the video_url in its manifest) as input_video_uri"
    )
    return (
        f"To {verb} that clip on {DEFAULT_OMNI_MODEL}, call {tool} with "
        f"{source} in place of previous_interaction_id."
    )


def _lost_interaction_conflict(
    request: ResolvedRequest, task: str
) -> RoutingConflict | None:
    """A conflict when the interaction to continue lives on a model nothing calls.

    An interaction cannot change models, so one recorded on a retired model
    (redirected to its replacement, which never held it) or on a switched-off
    one has no continuation at all. The tools refuse it before spending
    anything; planning it would hand over a route they refuse.
    """
    recorded = request.previous_interaction_model
    if request.previous_interaction_id is None or recorded is None:
        return None
    if omni_model_is_retired(recorded):
        code = "interaction_model_retired"
        where = f"{recorded}, which this server has retired"
    elif recorded in DEPRECATED_VIDEO_MODELS and omni_preview_is_sunset():
        code = "interaction_model_switched_off"
        where = (
            f"{recorded}, whose endpoint was switched off on "
            f"{DEPRECATED_VIDEO_MODELS[recorded]}"
        )
    else:
        return None
    return RoutingConflict(
        code=code,
        detail=(
            f"Interaction {request.previous_interaction_id} was created on "
            f"{where}, and an interaction cannot change models, so it cannot "
            f"be continued on {DEFAULT_OMNI_MODEL}."
        ),
        resolution=_saved_clip_resolution(task, request),
    )


def _backend_label(backend: str) -> str:
    return "Vertex AI" if backend == "vertex" else "the Gemini Developer API"


def _omni_backend_reachable(backend: str, request: ResolvedRequest) -> bool | None:
    """Whether this server can send an omni call to ``backend``; None if unknown.

    Mirrors the server's own test (_omni_backend_decision): Vertex AI needs a
    Vertex primary client, and the Gemini Developer API needs a key, which a
    Gemini-API primary client is.
    """
    if request.backend == "unknown":
        return None
    if backend == "vertex":
        return request.backend == "vertex"
    if request.backend == "gemini_api":
        return True
    return request.gemini_api_key_available


def _interaction_backend_conflict(
    request: ResolvedRequest, task: str
) -> RoutingConflict | None:
    """A conflict when the interaction lives on a backend this server cannot reach.

    An interaction resolves on the backend that minted it and on no other, so
    a continuation sent anywhere else fails -- after the tool has refused the
    other backend's pin, or with the service's own error.
    """
    recorded = request.previous_interaction_backend
    if request.previous_interaction_id is None or recorded is None:
        return None
    if _omni_backend_reachable(recorded, request) is not False:
        return None
    where = _backend_label(recorded)
    reach = (
        "Vertex AI credentials (GOOGLE_GENAI_USE_VERTEXAI=true)"
        if recorded == "vertex"
        else "a GEMINI_API_KEY"
    )
    fix = _saved_clip_resolution(task, request)
    return RoutingConflict(
        code="interaction_backend_unreachable",
        detail=(
            f"Interaction {request.previous_interaction_id} was created on "
            f"{where}, and an interaction lives on the backend that created "
            f"it, which this server cannot reach."
        ),
        resolution=(
            f"Give the server {reach} to continue it there. Or: "
            f"{fix[0].lower()}{fix[1:]}"
        ),
    )


def _continuation_backend_param(request: ResolvedRequest) -> str | None:
    """The omni_backend a continuation route sends, or None to leave it unset.

    Named only when the server pins OMNI_BACKEND to the other backend: left
    unset, the tool then refuses the call, while with no pin it continues on
    the recorded backend by itself.
    """
    recorded = request.previous_interaction_backend
    if recorded is None or request.omni_backend in (None, recorded):
        return None
    return recorded


def _continuation_backend_caveat(request: ResolvedRequest) -> str:
    """Why a continuation route names omni_backend."""
    recorded = str(request.previous_interaction_backend)
    return (
        f"omni_backend is {recorded}: interaction "
        f"{request.previous_interaction_id} lives on {_backend_label(recorded)}, "
        f"and this server's OMNI_BACKEND default, {request.omni_backend}, "
        "cannot continue it."
    )


def _continuation_model(request: ResolvedRequest) -> str:
    """The one omni model a continuation of previous_interaction_id runs on.

    The recorded model when the server has one, else a pinned omni model,
    else the default: the model the tools take when nothing is recorded.
    """
    if request.previous_interaction_model is not None:
        return canonical_omni_model(request.previous_interaction_model)
    if request.pinned_model is not None and is_omni_model(request.pinned_model):
        return request.pinned_model
    return DEFAULT_OMNI_MODEL


def _continuation_omni_model_param(request: ResolvedRequest) -> str | None:
    """The omni_model a continuation route sends, or None to leave it unset.

    Sent only when it is known to be the model that recorded the interaction
    (or the caller pinned it). Otherwise the tool resolves an unset omni_model
    to the recorded model itself; naming one here made every route but the
    recorded model's a call the tool refuses, and the top route was one of
    them whenever the interaction lived on the other model.
    """
    if request.previous_interaction_model is not None or (
        request.pinned_model is not None and is_omni_model(request.pinned_model)
    ):
        return _continuation_model(request)
    return None


def _continuation_model_caveat(
    tool: ToolName, model: str, request: ResolvedRequest
) -> str:
    """What a continuation route's omni_model (or its absence) means."""
    if request.previous_interaction_model is not None:
        return (
            f"omni_model is {model}, the model this server recorded for "
            "previous_interaction_id: an interaction's video context lives "
            "with the model that made it, so a continuation cannot switch "
            "models."
        )
    if _continuation_omni_model_param(request) is not None:
        return (
            f"omni_model is the pinned {model}. It must be the model that "
            "made previous_interaction_id, since an interaction cannot change "
            f"models; {tool} refuses the call otherwise."
        )
    return (
        f"omni_model is left unset, so {tool} continues on the model that "
        "made previous_interaction_id (an interaction cannot change models). "
        f"The quote is {model}'s, the model the tool uses when this server "
        "has no record of the interaction."
    )


def _backend_rejection(model: str, request: ResolvedRequest) -> tuple[str, str] | None:
    """Return (reason, resolution) when the backend cannot serve ``model``.

    Veo runs on Vertex AI only: Google switched off every Veo 3.1 model the
    Gemini Developer API served on VEO_GEMINI_API_SHUTDOWN, and the server
    refuses the call there (src/__main__._client_for_video_model). Omni is
    served on both backends.

    Args:
        model: Candidate model ID.
        request: The resolved request, including the backend.

    Returns:
        The rejection reason paired with the concrete fix, or None when the
        backend can serve the model.
    """
    if (
        model in _VIDEO_CAPABILITIES
        and not is_omni_model(model)
        and request.backend == "gemini_api"
    ):
        return (
            f"{model} excluded: Veo is not available on the Gemini Developer "
            f"API, where Google switched off its Veo 3.1 models on "
            f"{VEO_GEMINI_API_SHUTDOWN}.",
            f"Use {DEFAULT_OMNI_MODEL}, Google's named replacement, or run the "
            "server with Vertex AI credentials, where Veo 3.1 stays generally "
            "available.",
        )
    return None


def _select_video_tool(request: ResolvedRequest) -> ToolName:
    """Pick the tool for a video request.

    The ladder is ordered by how specific the signal is, most specific first:
    "make it longer" can only mean an extension; an interaction id otherwise
    means an edit; frames can only mean a transition; and so on down to plain
    text-to-video.

    Extension outranks the interaction id because an interaction id is a
    source, not a verb — omni can both edit AND extend from one, and "extend
    this clip" asked for the latter. Ranking the id first routed every
    "make it longer, from interaction X" to edit_video, which does not
    lengthen anything.
    """
    if request.needs_extension:
        return "loop_extend"
    if request.previous_interaction_id is not None:
        return "edit_video"
    if request.has_first_frame and request.has_last_frame:
        # Bridges and transitions are the same Veo primitive; the difference
        # is only whether the endpoints are stills or clips to sample.
        return "generate_bridge" if request.wants_bridge else "generate_transition"
    if request.wants_bridge and request.source_video_uri is not None:
        return "generate_bridge"
    if request.num_beats <= 1 and (request.wants_bridge or request.wants_transition):
        # A single-shot brief that ASKS for a transition gets the tool that
        # renders one, even with no endpoints yet — _video_params then names
        # the URIs it still needs. Falling through to generate_video here made
        # the plan incoherent: the conflict block and the capability
        # rejections both named generate_transition while every route offered
        # was generate_video. Guarded on the beat count so a multi-shot brief
        # ("a 3 shot reel with crossfade transitions") still routes to
        # generate_clip, which renders its own bridges.
        return "generate_bridge" if request.wants_bridge else "generate_transition"
    if request.num_beats > 1:
        return "generate_clip"
    return "generate_video"


def _veo_chains_runtime(request: ResolvedRequest) -> bool:
    """Whether a single-shot Veo route reaches this runtime by extending a seed.

    True for a runtime past VEO_SNAP_MAX_SECONDS that is still one render on
    some other model; anything longer implies an extension for every model
    in resolve_request already. A Veo route there renders an 8s seed and
    extends it, as every route did before Omni 1.2 rendered 15s in one call,
    rather than snapping the whole ask to 8s.
    """
    return (
        _select_video_tool(request) == "generate_video"
        and request.total_duration_seconds is not None
        and VEO_SNAP_MAX_SECONDS
        < request.total_duration_seconds
        <= MAX_SINGLE_RENDER_SECONDS
    )


def _route_tool(tool: ToolName, model: str) -> ToolName:
    """Return the tool a given model is actually reached through.

    Omni never runs behind the Veo-shaped tools: it is called via
    generate_video_omni (or edit_video, which is omni-only by definition, or
    extend_video_omni for a continuation), so the emitted parameters match the
    signature the caller will use.
    """
    if tool == "edit_video":
        return "edit_video"
    if is_omni_model(model):
        # Extension is its own verb on omni: it appends to existing footage
        # rather than rendering a new clip, and loop_extend is Veo's chaining
        # tool, which cannot drive an interaction.
        if tool == "loop_extend":
            return "extend_video_omni"
        return "generate_video_omni"
    return tool


def _omni_serves_on_vertex(request: ResolvedRequest) -> bool:
    """Whether an omni call from this request will run on Vertex AI.

    Mirrors the server's own choice (_omni_backend_decision): an explicit
    OMNI_BACKEND pin wins; otherwise omni goes to the Gemini Developer API
    whenever a key is present, even on a Vertex-primary deployment, so
    "backend is vertex" alone is not the question. Pricing the
    Vertex upload ceiling off that alone had the plan saying $12.18 for a
    chain the tool quoted at $9.14 on the same server. An UNKNOWN key
    (None) is read as absent, which prices the higher Vertex ceiling -- the
    direction a quote may err in.

    A continuation runs where its interaction was recorded, whatever the
    pin: the route names that backend when the pin is the other one
    (_continuation_backend_param), so its quote and delivery advice are that
    backend's too.
    """
    recorded = request.previous_interaction_backend
    if request.previous_interaction_id is not None and recorded is not None:
        return recorded == "vertex"
    if request.omni_backend == "vertex":
        return True
    if request.omni_backend == "gemini_api":
        return False
    return request.backend == "vertex" and request.gemini_api_key_available is not True


# Why the plan recommends a bucket, said beside the tools' own warning: what
# a large inline delivery risks is paying for a render that never arrives.
_WHY_A_BUCKET = (
    "If the response arrives without the video, the render has still been "
    "billed; written to a bucket, it is kept either way."
)


def _inline_delivery_caveat(
    tool: ToolName,
    spec: OmniModelSpec,
    *,
    vertexai: bool,
    resolution: str | None,
    task: str | None,
    output_seconds: float | None,
    seconds_are_bound: bool = False,
    chain_turns: int = 1,
) -> tuple[str | None, bool]:
    """What a route with no output_gcs_uri says about a large Omni output.

    The tools decide this before rendering with omni_inline_delivery_policy,
    so the plan asks the same function, with what the route sends, rather
    than restating the policy: by default a large output on Vertex AI is
    requested inline with a warning, and when the server is set to refuse
    such a request the function raises, which the plan reports as a
    requirement. The router is never given a bucket, so it never sends one.

    Returns:
        The caveat (None when there is nothing to say), and whether the tool
        refuses the call without output_gcs_uri.
    """
    try:
        warning = omni_inline_delivery_policy(
            spec,
            vertexai=vertexai,
            output_gcs_uri=None,
            resolution=resolution,
            task=task,
            output_seconds=output_seconds,
            seconds_are_bound=seconds_are_bound,
            chain_turns=chain_turns,
        )
    except ValueError as refusal:
        # The tool's own refusal, less its "Nothing was rendered or billed",
        # which a plan has no call to report.
        return (
            f"Before rendering, {tool} refuses this call without "
            "output_gcs_uri. "
            + str(refusal).removesuffix(" Nothing was rendered or billed."),
            True,
        )
    if warning is None:
        return None, False
    return f"{warning} {_WHY_A_BUCKET}", False


def _omni_route_delivery_caveat(
    tool: ToolName, model: str, request: ResolvedRequest, params: dict[str, Any]
) -> str | None:
    """The inline-delivery caveat for an Omni render route, or None.

    Fed what each tool feeds the policy for the same params: a fresh render
    its requested length, an edit the upper bound of a source this planner
    cannot see, and an extension chain its turns.
    """
    if request.wants_gcs_output:
        # The caller already means to send one.
        return None
    vertexai = _omni_serves_on_vertex(request)
    spec = omni_spec(model)
    resolution = params.get("resolution")
    if tool == "edit_video":
        caveat, _ = _inline_delivery_caveat(
            tool,
            spec,
            vertexai=vertexai,
            resolution=resolution,
            task="edit",
            output_seconds=omni_continuation_upper_bound(
                spec, "edit", None, vertexai=vertexai
            ),
            seconds_are_bound=True,
        )
    elif tool == "extend_video_omni":
        caveat, _ = _inline_delivery_caveat(
            tool,
            spec,
            vertexai=vertexai,
            resolution=resolution,
            task=OMNI_TASK_EXTEND,
            output_seconds=None,
            chain_turns=int(params.get("times", 1)),
        )
    else:
        caveat, _ = _inline_delivery_caveat(
            tool,
            spec,
            vertexai=vertexai,
            resolution=resolution,
            task=None,
            output_seconds=params.get("duration_seconds"),
        )
    return caveat


def _clip_bridges_possible(request: ResolvedRequest) -> bool:
    """Whether generate_clip can render the bridges a brief asks for.

    A bridge is a first/last-frame render, which Veo refuses on the Gemini
    Developer API. The clip itself still runs there, so bridges are a
    preference the tool degrades around (add_bridges off, with a caveat), not
    a capability to demand of the model — demanding it excluded every Veo
    model and returned a plan with no routes and no explanation.
    """
    return request.backend != "gemini_api"


def _video_needs(request: ResolvedRequest, tool: ToolName) -> VideoNeeds:
    """Translate the resolved request plus chosen tool into hard requirements."""
    # A brief that ASKS for a transition needs first/last-frame support
    # whether or not the endpoint URIs have been handed over yet. Keying this
    # on the chosen tool alone meant "a smooth transition from sunrise to
    # snowfall" — with no URIs, which is how an agent asks before it has the
    # stills — selected generate_video, needed nothing, and ranked a model
    # that cannot interpolate (Veo 3.1 Lite, since removed): the exact example
    # this module's docstring gives for a request it catches "before you pay
    # for the failure".
    first_last = (
        tool in ("generate_transition", "generate_bridge")
        or (request.has_first_frame and request.has_last_frame)
        or (
            request.wants_transition
            and not (tool == "generate_clip" and not _clip_bridges_possible(request))
        )
    )
    return VideoNeeds(
        first_last_frame=first_last,
        extension=tool == "loop_extend" or request.needs_extension,
        reference_images=request.num_reference_images > 0,
        four_k=request.resolution == "4K" or request.needs_4k,
        hd_1080p=request.resolution == "1080p",
        draft_360p=request.resolution == "360p",
        seed=request.needs_seed,
        negative_prompt=request.needs_negative_prompt,
        audio=request.needs_audio,
        # An interaction id is a source only the omni family can read, so
        # ANY route built on one needs the conversational capability — not
        # just edit_video. Keying on the tool alone let an extension-flavoured
        # request ("make this longer, continue the scene, from interaction X")
        # top out with a Veo loop_extend route that had silently dropped the
        # id and told the caller to supply a clip they do not have.
        conversational_edit=(
            tool == "edit_video"
            or request.previous_interaction_id is not None
            # A brief that names a prior interaction as its source needs a
            # model that can read one, even before the id has been supplied.
            or request.names_interaction_source
        ),
        clip_duration_seconds=request.clip_duration_seconds,
    )


def _veo_generation_mode(tool: ToolName, request: ResolvedRequest) -> str | None:
    """The generation mode a Veo ``generate_video`` call will run in.

    Mirrors the derivation inside ``generate_video`` and its dry_run, because
    the mode decides both the billed length (reference_to_video is forced to
    8s) and whether the chosen backend can serve the call at all.

    Only ``generate_video`` is derived here. Every other Veo tool has a mode
    fixed by the tool itself and already emits its own gating caveats, so
    returning a mode for those would double up on advice they give already.
    """
    if tool != "generate_video":
        return None
    if request.source_video_uri is not None:
        return "extend_video"
    if request.num_reference_images:
        return "reference_to_video"
    if request.has_first_frame and request.has_last_frame:
        return "first_last_frame"
    if request.has_first_frame:
        return "image_to_video"
    return "text_to_video"


def _splits_runtime(tool: ToolName, request: ResolvedRequest) -> bool:
    """Whether a runtime the caller named is shared out across beats."""
    return (
        tool in ("generate_clip", "generate_storyboard")
        and request.num_beats > 1
        and request.total_duration_seconds is not None
    )


def _beat_seconds(request: ResolvedRequest) -> float:
    """One beat's share of the runtime asked for, before any snap.

    The runtime a multi-shot brief names is the whole clip's, so each beat
    gets its share of it. Every beat took the whole runtime instead (capped at
    one render): "a 3 shot commercial, 12 seconds total" planned three 8s
    beats, 24s, and a 2-beat 12s brief said "12s snaps to 8s".
    """
    if _splits_runtime("generate_clip", request):
        return float(request.total_duration_seconds or 0) / request.num_beats
    return request.clip_duration_seconds


def _veo_extension_gcs_caveat(request: ResolvedRequest) -> str | None:
    """Where a Veo extension on Vertex AI is delivered, or None off Vertex.

    The tool refuses one with neither an output_gcs_uri nor the server's
    VIDEO_GCS_BUCKET, its dry run included, so a server known to have no
    bucket gets a caveat that asks for the key rather than describes it.
    """
    if request.backend != "vertex":
        return None
    if request.video_gcs_bucket_configured is False:
        return (
            "Add output_gcs_uri (a prefix such as gs://bucket/renders/): a Veo extension on "
            "Vertex AI is delivered only to Cloud Storage, and this server has "
            "no VIDEO_GCS_BUCKET to fall back on, so the call is refused "
            "without one."
        )
    if request.video_gcs_bucket_configured:
        return (
            "On Vertex AI the extended video is written to Cloud Storage: to "
            "this server's VIDEO_GCS_BUCKET, or to output_gcs_uri if given."
        )
    return (
        "On Vertex AI, extension requires output_gcs_uri (or a VIDEO_GCS_BUCKET "
        "set on the server) — the combined video exceeds the inline response "
        "limit."
    )


def _omni_seed_seconds(spec: OmniModelSpec) -> float:
    """How long an omni extension chain's seed render is.

    The longest source the chain's turn count assumes (_video_params measures
    what is left to render from it), within one render.
    """
    return float(min(spec.max_uploaded_source_seconds, spec.max_duration_seconds))


def _omni_extension_turn_seconds(
    spec: OmniModelSpec,
    added_seconds: float | None,
    times: int,
    *,
    continues_interaction: bool,
) -> int | None:
    """The per-turn duration_seconds an omni extension route sends, or None.

    gemini-omni-1.2-flash sets each turn's length (a whole second from 3 up
    to the step), so an ask for fewer seconds than whole turns would add goes
    out as a shorter turn. Without it, "extend this by 6 seconds" planned one
    default turn, which renders and bills 10s. Only an explicit delta counts:
    a runtime total is measured against a source the planner cannot see.
    None leaves the tool's default step, as when the ask already fills every
    turn, or when extend_video_omni would refuse the duration
    (omni_extension_duration_refusal), so the route stays callable as given.
    """
    step = spec.extension_step_seconds
    if added_seconds is None or step <= 0 or times < 1:
        return None
    seconds = min(step, max(OMNI_MIN_DURATION, math.ceil(added_seconds / times)))
    if seconds >= step:
        return None
    if omni_extension_duration_refusal(
        spec,
        duration_seconds=seconds,
        continues_interaction=continues_interaction,
        times=times,
    ):
        return None
    return int(seconds)


def _omni_seed_cost(model: str, request: ResolvedRequest) -> CostEstimateLike | None:
    """What the generate_video_omni seed of an omni extension chain costs."""
    return _estimate_video_cost(
        model,
        _omni_seed_seconds(omni_spec(model)),
        (request.resolution if omni_spec(model).supports_resolution else None)
        or OMNI_RESOLUTION,
        request.needs_audio,
    )


def _veo_seed_cost(model: str, request: ResolvedRequest) -> CostEstimateLike | None:
    """What the 8s generate_video seed of a Veo extension chain costs.

    Priced as the seed step's own params render it (_build_workflow): 8s at
    720p, with the chain's audio choice.
    """
    return _estimate_video_cost(
        model, float(VEO_MAX_CLIP_SECONDS), OMNI_RESOLUTION, request.needs_audio
    )


def _veo_chain_total(
    model: str, request: ResolvedRequest, params: dict[str, Any]
) -> str:
    """The seed's cost and the chain's total, said beside a quote that has neither.

    A chain route's cost is loop_extend's own quote, which covers the
    extensions alone -- the figure its dry run gives for the same params. Said
    without the seed, a 12s Veo Fast chain read as $0.70 beside Omni 1.2's
    $0.84 single render, and really costs $1.50.
    """
    seed = _veo_seed_cost(model, request)
    extensions = _video_cost("loop_extend", model, request, params)
    if seed is None or extensions is None:
        return ""
    return (
        f" The seed costs about ${seed.usd:.2f}, so the chain costs about "
        f"${seed.usd + extensions.usd:.2f} in all; this route's quote covers "
        "the extensions alone."
    )


def _omni_source_over_limit(limit: float, seconds: float) -> bool:
    """Whether a recorded source is over an Omni source limit, as the tools judge it.

    With the tools' one frame of tolerance (_omni_source_length_refusal):
    this server's renders measure a frame over their nominal length, so a
    recorded 15.03s clip that edit_video and upscale_video_omni both take was
    refused here.
    """
    from .pricing import OMNI_ENCODER_ALLOWANCE_SECONDS

    return seconds > limit + OMNI_ENCODER_ALLOWANCE_SECONDS


def _omni_input_rejection(
    tool: ToolName, model: str, request: ResolvedRequest
) -> tuple[str, str, str] | None:
    """(code, reason, fix) when an Omni tool would refuse this route's inputs.

    The planner offered routes the tools refuse before they render: an edit
    of a recorded 25s clip (1.2 edits up to 15s), an extension of a clip
    already at the ceiling (priced at nothing, blamed on the price book), and
    7 reference images where 1.2 takes 6, or 3 on a Vertex AI edit, which
    carries at most 2 images.
    """
    if not is_omni_model(model):
        return None
    # The loop carries the generic tool (generate_video, loop_extend); the
    # rules below are the Omni tool's.
    tool = _route_tool(tool, model)
    vertexai = _omni_serves_on_vertex(request)
    spec = omni_spec(model, vertexai=vertexai)
    references = request.num_reference_images or 0
    if tool in ("generate_video_omni", "edit_video", "extend_video_omni"):
        if references > spec.max_reference_images:
            return (
                "too_many_reference_images",
                f"{model} takes at most {spec.max_reference_images} reference "
                f"images, and {references} were asked for.",
                _reference_limit_fix(references),
            )
        if (
            tool == "edit_video"
            and vertexai
            and spec.max_vertex_edit_images
            and references > spec.max_vertex_edit_images
        ):
            return (
                "too_many_edit_images",
                f"A Vertex AI edit on {model} carries at most "
                f"{spec.max_vertex_edit_images} images, and {references} "
                "reference images were asked for.",
                f"Pass at most {spec.max_vertex_edit_images}, or edit an "
                "interaction created on the Gemini API, which takes up to "
                f"{omni_spec(model).max_reference_images}.",
            )
    seconds = (
        request.previous_interaction_seconds
        if request.previous_interaction_id is not None
        else None
    )
    if seconds is None:
        return None
    if tool == "edit_video":
        cap = omni_source_limit_seconds(spec, task="edit")
        if cap and _omni_source_over_limit(cap, seconds):
            return (
                "source_too_long",
                f"{model} edits clips up to {cap:g}s, and interaction "
                f"{request.previous_interaction_id}'s clip is {seconds:g}s, as "
                "this server recorded it.",
                f"Trim the clip it saved to {cap:g}s or less and pass it as "
                "input_video_uri to generate_video_omni.",
            )
    if tool == "extend_video_omni":
        refusal = omni_extension_refusal(spec, seconds)
        if refusal is not None:
            return (
                "source_at_extension_ceiling",
                refusal.split(" Nothing was rendered", 1)[0],
                refusal.split("Nothing was rendered or billed. ", 1)[-1],
            )
    return None


def _veo_reference_rejection(
    model: str, request: ResolvedRequest
) -> tuple[str, str, str] | None:
    """(code, reason, fix) when a Veo route would drop reference images.

    Veo takes 3 reference images, and generate_video sends the first 3 of a
    longer list with a warning. A planner that offers that route for 4 to 6
    references ranks a render that ignores some of them beside an Omni route
    that uses all of them, so it is excluded rather than offered.
    """
    references = request.num_reference_images or 0
    if is_omni_model(model) or references <= MAX_VIDEO_REFERENCE_IMAGES:
        return None
    return (
        "too_many_reference_images",
        f"{model} takes at most {MAX_VIDEO_REFERENCE_IMAGES} reference images, "
        f"and {references} were asked for; the rest would not be sent.",
        _reference_limit_fix(references),
    )


def _reference_limit_fix(references: int) -> str:
    """The one fix for a reference count over a video model's limit.

    Omni and Veo routes refuse the same brief for the same reason, so they
    share one sentence: two ("Pass at most 6 ... Pass at most 3 ...") read
    as contradictory advice when every model was excluded.
    """
    omni_max = omni_spec(DEFAULT_OMNI_MODEL).max_reference_images
    if references <= omni_max:
        return (
            f"Use {DEFAULT_OMNI_MODEL}, which takes up to {omni_max} reference "
            f"images, or pass at most {MAX_VIDEO_REFERENCE_IMAGES} for Veo."
        )
    return (
        f"Pass at most {omni_max} reference images for {DEFAULT_OMNI_MODEL}, "
        f"or at most {MAX_VIDEO_REFERENCE_IMAGES} for Veo."
    )


def _video_params(
    tool: ToolName, model: str, request: ResolvedRequest
) -> tuple[dict[str, Any], list[str]]:
    """Build the ready-to-use parameter dict for a video route.

    Returns the params plus any caveats generated while building them (e.g.
    URIs the router was never given, so it left the key out rather than
    inventing a value).
    """
    caveats: list[str] = []
    aspect_ratio = request.aspect_ratio
    if aspect_ratio is not None and aspect_ratio not in VIDEO_ASPECT_RATIOS:
        caveats.append(
            f"aspect_ratio={aspect_ratio} is not supported for video "
            f"(only {', '.join(VIDEO_ASPECT_RATIOS)}); planned as 16:9."
        )
        aspect_ratio = "16:9"
    if aspect_ratio is None:
        aspect_ratio = "9:16" if tool == "generate_clip" else "16:9"

    duration = (
        _beat_seconds(request)
        if tool == "generate_clip"
        else request.clip_duration_seconds
    )
    if is_omni_model(model) and request.has_first_frame and request.has_last_frame:
        if duration > omni_spec(model).max_keyframe_seconds:
            caveats.append(
                f"First/last-frame interpolation is limited to {omni_spec(model).max_keyframe_seconds}s; "
                f"this route shortens the requested {duration:g}s clip to that limit."
            )
        duration = min(duration, float(omni_spec(model).max_keyframe_seconds))
    if not is_omni_model(model):
        # The mode decides the length before the ladder does. generate_video
        # derives it from the same inputs, and src/video.py then FORCES 8s for
        # reference_to_video -- so pricing every Veo route as text_to_video
        # under-quoted a reference render: 3 references at duration_seconds=4
        # was quoted $0.40 (4s) while the tool's own dry_run said $0.80 (8s).
        veo_mode = _veo_generation_mode(tool, request)
        if veo_mode == "reference_to_video":
            if duration != VEO_REFERENCE_MODE_SECONDS:
                caveats.append(
                    f"Veo renders reference-guided clips at exactly "
                    f"{VEO_REFERENCE_MODE_SECONDS:g}s; the {duration:g}s request "
                    f"is not honored and the quote is for "
                    f"{VEO_REFERENCE_MODE_SECONDS:g}s."
                )
            duration = VEO_REFERENCE_MODE_SECONDS
        elif tool != "loop_extend":
            # loop_extend takes no duration: it appends ~7s steps to a clip,
            # and the runtime asked for sets how many.
            snapped = _snap_veo_duration(duration)
            ladder = "/".join(str(d) for d in VEO_DURATIONS_SECONDS)
            if float(snapped) != duration and _splits_runtime(tool, request):
                # Said per beat: "12s snaps to 8s" read as the whole clip cut
                # short, beside beats that add up to more than was asked.
                beats = min(request.num_beats, MAX_CLIP_BEATS)
                caveats.append(
                    f"Each of the {beats} beats is {round(duration, 2):g}s of "
                    f"the {request.total_duration_seconds:g}s asked; Veo renders "
                    f"{ladder}s clips, so each snaps to {snapped}s "
                    f"({snapped * beats}s in all)."
                )
            elif float(snapped) != duration:
                caveats.append(
                    f"Veo renders {ladder}s clips; {duration:g}s snaps to {snapped}s."
                )
            duration = float(snapped)
        if request.backend == "unknown":
            # A caveat that may not apply is cheaper than a refused call.
            caveats.append(
                f"{model} runs on Vertex AI only; on the Gemini Developer API "
                f"Veo was switched off on {VEO_GEMINI_API_SHUTDOWN} and the "
                "call is refused."
            )

    params: dict[str, Any] = {}
    if tool == "edit_video":
        params = {
            "previous_interaction_id": request.previous_interaction_id,
            "prompt": request.intent,
            "aspect_ratio": aspect_ratio,
            "duration_seconds": duration,
        }
        continued_on = _continuation_omni_model_param(request)
        if continued_on is not None:
            params["omni_model"] = continued_on
        backend_param = _continuation_backend_param(request)
        if backend_param is not None:
            params["omni_backend"] = backend_param
        if omni_spec(model).supports_resolution and request.resolution is not None:
            params["resolution"] = request.resolution
        caveats.append(_continuation_model_caveat(tool, model, request))
        if backend_param is not None:
            caveats.append(_continuation_backend_caveat(request))
        caveats.append(
            "An edit does not send duration or aspect ratio — the API rejects "
            "them on an edit task — and the rendered length is chosen by the "
            "service, predictable from neither the request nor the source. "
            "Cost is measured from the rendered file; a quote is an upper bound."
        )
        # The route model's own ceiling, which is also what _video_cost prices
        # this edit at. A flat 15s here would tell a preview route (10s) that
        # its quote was 15s, beside a quote computed at 10s.
        render_max = float(omni_spec(model).max_duration_seconds)
        edit_cap = omni_source_limit_seconds(omni_spec(model), task="edit")
        caveats.append(
            f"The quote is {model}'s {render_max:g}s per-render maximum, "
            "the same upper bound edit_video's own dry_run reports — not the "
            f"{duration:g}s above, which is never sent."
            + (
                f" {model} edits clips up to {edit_cap:g}s; a longer one is "
                "refused before it renders."
                if edit_cap
                else ""
            )
        )
    elif tool == "loop_extend":
        times = 1
        if request.added_duration_seconds is not None:
            # The caller named an amount to add, so use it directly rather
            # than deriving a delta from a total that may be the SOURCE
            # clip's length.
            times = max(
                1,
                math.ceil(request.added_duration_seconds / VEO_EXTENSION_SECONDS),
            )
        elif request.total_duration_seconds is not None:
            extra = request.total_duration_seconds - VEO_MAX_CLIP_SECONDS
            times = max(1, math.ceil(extra / VEO_EXTENSION_SECONDS))
        if times > VEO_MAX_EXTENSIONS:
            caveats.append(
                f"{times} extensions would be needed but Veo allows "
                f"{VEO_MAX_EXTENSIONS}; planned at the maximum, which reaches "
                f"{VEO_MAX_EXTENDED_SECONDS}s."
            )
        params = {
            "prompt": request.intent,
            "times": min(times, VEO_MAX_EXTENSIONS),
            "model": model,
            "aspect_ratio": aspect_ratio,
            "include_audio": request.needs_audio,
        }
        if request.source_video_uri is not None:
            params["video_uri"] = request.source_video_uri
        elif request.extension_implied or _veo_chains_runtime(request):
            # No clip exists yet, and only the top route gets the seed step
            # laid out as a workflow; a lower-ranked chain has to say it too.
            planned = int(params["times"])
            reached = VEO_MAX_CLIP_SECONDS + VEO_EXTENSION_SECONDS * planned
            caveats.append(
                f"No clip exists yet: render the opening {VEO_MAX_CLIP_SECONDS}s "
                f"with generate_video on {model}, then pass its video_url as "
                f"video_uri. With {planned} extension{'s' if planned != 1 else ''} "
                f"it runs about {reached}s against the "
                f"{request.total_duration_seconds:g}s asked."
                + _veo_chain_total(model, request, params)
            )
        else:
            caveats.append(
                "Add video_uri: loop_extend needs the existing Veo clip to continue."
            )
        caveats.append(
            f"Each extension adds ~{VEO_EXTENSION_SECONDS}s and output is 720p; "
            f"Veo allows at most {VEO_MAX_EXTENSIONS} extensions."
        )
        gcs = _veo_extension_gcs_caveat(request)
        if gcs is not None:
            caveats.append(gcs)
    elif tool == "extend_video_omni":
        spec = omni_spec(model, vertexai=_omni_serves_on_vertex(request))
        step = float(spec.extension_step_seconds)
        wanted = request.added_duration_seconds
        # The delta a per-turn duration may be cut to: an explicit one, or a
        # total measured against a source whose length was recorded.
        exact_delta = wanted
        if wanted is None and request.total_duration_seconds is not None:
            # A total was named instead of a delta. Omni extends footage that
            # already exists, so what has to be rendered is the difference
            # above the source — measured from the recorded clip when this
            # server rendered it, else from the longest source omni will
            # take, rather than from zero, which would bill the caller for
            # seconds they already have.
            source = request.previous_interaction_seconds
            wanted = request.total_duration_seconds - (
                source or spec.max_uploaded_source_seconds
            )
            if source is not None and wanted > 0:
                exact_delta = wanted
        times = max(1, math.ceil((wanted or step) / step))
        # The most full-step turns the cumulative ceiling allows. An unknown
        # source is priced conservatively (see _video_cost), but does not
        # decide how many turns fit: the tool measures the source and trims
        # the chain on the Developer API, and Vertex keeps a rolling window.
        ceiling = spec.max_extended_seconds // spec.extension_step_seconds
        if times > ceiling:
            caveats.append(
                f"{times} extension turns would be needed but omni caps a clip "
                f"at {spec.max_extended_seconds}s in "
                f"{spec.extension_step_seconds}s steps; planned at the "
                f"{ceiling}-turn maximum."
            )
        # A continuation names its model only when it is known to be the one
        # that holds the interaction; any other source is new to the model.
        omni_model_param = (
            _continuation_omni_model_param(request)
            if request.previous_interaction_id is not None
            else model
        )
        params = {"prompt": request.intent}
        if omni_model_param is not None:
            params["omni_model"] = omni_model_param
        params["times"] = min(times, ceiling)
        turn_seconds = _omni_extension_turn_seconds(
            spec,
            exact_delta,
            int(params["times"]),
            continues_interaction=request.previous_interaction_id is not None,
        )
        if turn_seconds is not None:
            params["duration_seconds"] = turn_seconds
        if spec.supports_resolution and request.resolution is not None:
            # Priced at request.resolution below, so it has to be SENT. It was
            # the one omni branch that quoted a resolution its own params
            # never asked for: a 360p plan handed over params that render 720p
            # and bill 3x the quote.
            params["resolution"] = request.resolution
        if request.previous_interaction_id is not None:
            params["previous_interaction_id"] = request.previous_interaction_id
            caveats.append(_continuation_model_caveat(tool, model, request))
            backend_param = _continuation_backend_param(request)
            if backend_param is not None:
                params["omni_backend"] = backend_param
                caveats.append(_continuation_backend_caveat(request))
        elif request.source_video_uri is not None:
            params["input_video_uri"] = request.source_video_uri
        elif request.extension_implied:
            # No clip exists yet, as on the Veo chain: the seed is rendered
            # first, and only the top route gets it laid out as a workflow.
            seed = _omni_seed_seconds(spec)
            reached = min(seed + step * int(params["times"]), spec.max_extended_seconds)
            seed_cost = _omni_seed_cost(model, request)
            priced = (
                f" (est. ${seed_cost.usd:.2f}, on top of this route's quote)"
                if seed_cost is not None
                else ""
            )
            caveats.append(
                f"No clip exists yet: render the opening {seed:g}s with "
                f"generate_video_omni on {model}{priced}, then pass its "
                "interaction_id as previous_interaction_id. With "
                f"{params['times']} turn{'s' if params['times'] != 1 else ''} it "
                f"runs about {reached:g}s against the "
                f"{request.total_duration_seconds:g}s asked."
            )
        else:
            caveats.append(
                "Add previous_interaction_id (a clip this server generated) or "
                "input_video_uri (a clip to upload): extend_video_omni needs "
                "existing footage to continue."
            )
        caveats.append(
            f"Each turn appends {OMNI_MIN_DURATION}-{spec.extension_step_seconds}s "
            "using the final 10s as context. "
            + (
                f"Vertex retains up to the last {spec.extension_window_seconds}s "
                "of the source on each turn; longer input videos are supported."
                if spec.extension_window_seconds
                else "The Gemini API limits the assembled clip to "
                f"{spec.max_extended_seconds}s."
            )
        )
        recorded_source = (
            request.previous_interaction_seconds
            if request.previous_interaction_id is not None
            else None
        )
        caveats.append(
            "Each turn bills its assembled output. "
            + (
                f"The quote projects each turn from the {recorded_source:g}s "
                "this server recorded for the source, as the tool's own quote "
                "does."
                if recorded_source is not None
                else "The source length is unknown, so the quote uses the "
                f"{spec.max_extended_seconds}s output bound per turn. Use the "
                "tool's dry_run with a local source for a measured projection."
            )
        )
    elif tool in ("generate_transition", "generate_bridge"):
        params = {
            "prompt": request.intent,
            "model": model,
            "duration_seconds": duration,
            "aspect_ratio": aspect_ratio,
            "include_audio": request.needs_audio,
        }
        if tool == "generate_transition":
            if request.first_frame_uri is not None:
                params["first_frame_uri"] = request.first_frame_uri
            if request.last_frame_uri is not None:
                params["last_frame_uri"] = request.last_frame_uri
            if request.first_frame_uri is None or request.last_frame_uri is None:
                caveats.append(
                    "Add first_frame_uri and last_frame_uri: generate_transition "
                    "renders the motion between two stills."
                )
        else:
            caveats.append(
                "Add from_clip_uri and to_clip_uri: generate_bridge samples the "
                "last frame of one clip and the first frame of the next."
            )
        caveats.append(
            "aspect_ratio must match the surrounding clips or the cut will jump."
        )
    elif tool == "generate_clip":
        # generate_clip refuses more beats than MAX_CLIP_BEATS outright, so a
        # longer plan was a ready-to-call route that could only ever return an
        # error ("a 3 minute montage" planned 23). Truncate to what the tool
        # accepts and say what was lost — a silent truncation would quote a
        # runtime the caller is not getting.
        beat_count = request.num_beats
        if beat_count > MAX_CLIP_BEATS:
            caveats.append(
                f"{request.num_beats} beats would be needed but generate_clip "
                f"rejects more than {MAX_CLIP_BEATS}; planned at "
                f"{MAX_CLIP_BEATS} beats, which covers "
                f"{MAX_CLIP_BEATS * duration:g}s of the "
                f"{request.num_beats * duration:g}s asked for. Split the "
                "sequence across several clips to cover the rest."
            )
            beat_count = MAX_CLIP_BEATS
        params = {
            "beats": [
                {"prompt": request.intent, "duration_seconds": duration}
                for _ in range(beat_count)
            ],
            "aspect_ratio": aspect_ratio,
            "model": model,
            "include_audio": request.needs_audio,
            "add_bridges": request.wants_transition or request.wants_bridge,
        }
        if params["add_bridges"] and not _clip_bridges_possible(request):
            # A bridge is a first/last-frame render, which Veo refuses on this
            # backend. The clip itself runs here; with bridges on, the beats
            # rendered and billed and the first bridge then failed — so the
            # planner must not hand over the parameter that springs that trap.
            params["add_bridges"] = False
            caveats.append(
                "add_bridges disabled: Veo cannot render first/last-frame "
                "bridges on the Gemini Developer API (it refuses the mode after "
                "the beats have billed). Run against Vertex AI for bridged "
                "clips, or cut the beats together as-is."
            )
        caveats.append(
            f"Replace each of the {beat_count} beat prompts with that "
            "shot's own description — they are seeded from the intent."
        )
    elif tool == "generate_video_omni":
        spec = omni_spec(model)
        if request.is_draft and request.total_duration_seconds is None:
            # A throwaway pass with no runtime asked for renders the shortest
            # clip the model takes; the 6s default is a deliverable's length.
            duration = float(OMNI_MIN_DURATION)
            caveats.append(
                f"Planned at {OMNI_MIN_DURATION}s, the shortest render, because "
                "this is a draft and no duration was given; pass "
                "duration_seconds for the length the deliverable will run."
            )
        params = {
            "prompt": request.intent,
            "omni_model": model,
            "aspect_ratio": aspect_ratio,
            "duration_seconds": duration,
        }
        # Frames the caller supplied, bound to their roles. Emitted here
        # because a transition-shaped request now reaches this tool: routing
        # omni for two stills and then dropping the stills would hand over a
        # plan that renders something else entirely.
        if spec.supports_keyframes and request.first_frame_uri is not None:
            params["first_frame_uri"] = request.first_frame_uri
            if request.last_frame_uri is not None:
                params["last_frame_uri"] = request.last_frame_uri
                caveats.append(
                    "The two stills are bound as <FIRST_FRAME> and "
                    "<LAST_FRAME>, so omni interpolates between them — the "
                    "same shape generate_transition renders on Veo. Passing "
                    "one URI for both loops the clip."
                )
        elif (
            request.has_first_frame
            and request.has_last_frame
            and spec.supports_keyframes
        ):
            caveats.append(
                "Add first_frame_uri and last_frame_uri: this route "
                "interpolates between two stills and the router was given "
                "neither URI."
            )
        if spec.supports_resolution:
            cheap_tier = (
                request.resolution is None
                and request.budget == "cheap"
                and request.budget_stated
                and OMNI_DRAFT_RESOLUTION in spec.resolutions
            )
            if request.resolution is not None:
                params["resolution"] = request.resolution
            elif cheap_tier:
                # budget=cheap (a draft implies it) gets the cheapest tier
                # rather than a caveat pointing at it.
                params["resolution"] = OMNI_DRAFT_RESOLUTION
                caveats.append(
                    f"Planned at {OMNI_DRAFT_RESOLUTION}, a third of the 720p "
                    "price, because budget is cheap; pass resolution='720p' or "
                    "higher for the deliverable."
                )
            caveats.append(
                f"{model} renders {'/'.join(spec.resolutions)} (1080p and 4K "
                "upscaled from the base render) and has no seed or "
                "negative_prompt; keep the returned interaction_id to edit or "
                "extend it conversationally."
            )
            if not cheap_tier and request.resolution in (None, OMNI_RESOLUTION):
                # The one saving worth naming unprompted: a 360p pass is a
                # third of the 720p price (1,931 against 5,792 tokens/s on the
                # pricing page; src/pricing.py), which makes it the cheapest
                # video render this server can issue.
                caveats.append(_OMNI_360P_TIP)
        else:
            caveats.append(
                f"{model} output is {spec.rendered_resolution}/24fps with no "
                "seed or negative_prompt; keep the returned interaction_id to "
                "edit it conversationally. Pass "
                f"omni_model='{DEFAULT_OMNI_MODEL}' for resolution control, video "
                "extension and first/last-frame interpolation."
            )
    else:  # generate_video
        params = {
            "prompt": request.intent,
            "model": model,
            "aspect_ratio": aspect_ratio,
            "duration_seconds": duration,
            "include_audio": request.needs_audio,
        }
        if request.resolution is not None:
            params["resolution"] = request.resolution
        if request.num_reference_images:
            # The rationale for these models reads "picked for subject
            # consistency from reference images", but the router is given a
            # COUNT and never the URIs -- so the key is named rather than
            # invented, the way the transition/bridge routes name theirs. A
            # plan that recommends a model for inputs it never asks for is a
            # plan the caller cannot run.
            caveats.append(
                f"Add reference_image_uris ({request.num_reference_images} "
                "image(s)): this model was chosen for subject consistency "
                "from reference images, and without them the call is a plain "
                "text-to-video render."
            )

    # These tools have no resolution parameter at all, so an HD/4K ask cannot
    # reach them. It was previously priced as if it had — three times the
    # tool's own quote — and dropped without a word.
    requested_resolution = request.resolution or ("4K" if request.needs_4k else None)
    if tool in FIXED_720P_TOOLS and requested_resolution not in (None, "720p"):
        caveats.append(
            f"{tool} takes no resolution parameter and always renders "
            f"{OMNI_RESOLUTION}; the {requested_resolution} request cannot be "
            f"honored here and is neither sent nor priced. Render the shots "
            "individually with generate_video if the resolution matters."
        )

    if is_omni_model(model):
        delivery = _omni_route_delivery_caveat(tool, model, request, params)
        if delivery is not None:
            caveats.append(delivery)

    if request.wants_gcs_output and tool != "edit_video":
        if request.backend == "gemini_api":
            caveats.append(
                "output_gcs_uri omitted: GCS output is Vertex-only and the Gemini "
                "API rejects an explicit one."
            )
        elif request.backend in ("vertex", "unknown"):
            # Offered only where the parameter can exist. Stated as a positive
            # backend test rather than an `else`: advising a Vertex-only field
            # on a Gemini API deployment is advice that cannot be taken, and a
            # fourth Backend value must not inherit it by accident.
            caveats.append(
                "Set output_gcs_uri (Vertex AI only) to land the render in Cloud "
                "Storage."
            )

    sunset = DEPRECATED_VIDEO_MODELS.get(model)
    if sunset:
        caveats.append(
            f"{model} is deprecated: its endpoint is switched off on {sunset}. "
            "It ranks below any equally-scored replacement and should not be "
            f"built on. The GA successor is {DEFAULT_OMNI_MODEL}, which renders "
            "720p at a lower published rate and adds resolution control, "
            "video extension and first/last-frame interpolation."
        )

    return params, caveats


def _aggregate_video_cost(
    model: str,
    segment_seconds: float,
    total_seconds: float,
    segments: int,
    resolution: str,
    include_audio: bool,
) -> CostEstimateLike | None:
    """Price a multi-render plan (a clip's beats, a chain of extensions).

    ``estimate_video_cost`` snaps the duration it is given to a length the
    model can actually render, so handing it a summed duration would quote a
    single 8s clip for a 36s plan. Instead one segment is priced to obtain the
    published per-second rate, and that rate is applied to the real total.

    Returns None when pricing is unavailable or publishes no rate for the
    model/resolution pairing.
    """
    probe = _estimate_video_cost(model, segment_seconds, resolution, include_audio)
    if probe is None:
        return None
    rate = probe.breakdown.get("usd_per_second")
    if rate is None:
        # Unexpected breakdown shape — the single-segment estimate is still
        # better than nothing, and it is honestly labelled as an estimate.
        return probe
    try:
        from .pricing import CostEstimate
    except ImportError:  # pragma: no cover - probe already proved it imports
        return probe
    # Each omni render carries one frame of encoder overhang (see
    # pricing.quote_duration_for); a multi-render plan accrues it per render.
    # quote_duration_for(model, 0.0) is the per-render allowance itself —
    # zero for Veo, one frame for omni — so this stays a no-op for Veo and
    # keeps the planner's figure identical to the tool's per-beat quote.
    try:
        from .pricing import quote_duration_for

        per_render_allowance = quote_duration_for(model, 0.0)
    except ImportError:  # pragma: no cover - pricing is a sibling module
        per_render_allowance = 0.0
    quoted_seconds = total_seconds + segments * per_render_allowance
    usd = rate * quoted_seconds
    return CostEstimate(
        usd=usd,
        is_estimate=True,
        unit=probe.unit,
        # Carry the probe's provenance: this estimate is the probe's published
        # rate applied to a longer total, so it came off the same page. Built
        # by hand here, it would otherwise report no source at all.
        source=getattr(probe, "source", None),
        source_note=getattr(probe, "source_note", None),
        # And its date: a 1.2 multi-render plan was dated by the full
        # re-read, before the 1.2 table was verified.
        as_of=getattr(probe, "as_of", None),
        detail=(
            f"{segments} render{'s' if segments != 1 else ''} totalling "
            f"{quoted_seconds:g}s @ {resolution} on "
            f"{model} (${rate:g}/s)"
        ),
        breakdown={
            "renders": float(segments),
            "seconds": quoted_seconds,
            "usd_per_second": rate,
            "video_usd": usd,
        },
    )


def _video_cost(
    tool: ToolName, model: str, request: ResolvedRequest, params: dict[str, Any]
) -> CostEstimateLike | None:
    """Estimate the cost of a video route.

    Single-render routes are priced directly (so the quote reflects the
    duration snap); multi-render routes go through ``_aggregate_video_cost``.

    Every figure here must equal what the tool's own dry_run quotes for the
    same parameters, and must never under-state — a planner that disagrees
    with the tool it is recommending is worse than no quote at all.
    """

    # What an omni route is actually rendered at: the preview model has no
    # resolution parameter, so a 4K ask cannot reach it and must not be priced
    # as if it had; 1.2 renders what it is given.
    def _omni_resolution(candidate: str) -> str:
        spec = omni_spec(candidate)
        if not spec.supports_resolution:
            return OMNI_RESOLUTION
        # A continuation the caller names no resolution for sends the one the
        # interaction was recorded at, as the tools do; a 360p clip's
        # extension was quoted at 720p.
        inherited = (
            request.previous_interaction_resolution
            if "previous_interaction_id" in params
            and request.previous_interaction_resolution in spec.resolutions
            else None
        )
        # The route's own resolution first: a budget=cheap route is planned at
        # 360p without the request naming one, and was quoted at 720p.
        return (
            params.get("resolution")
            or request.resolution
            or inherited
            or OMNI_RESOLUTION
        )

    if tool == "edit_video":
        # An edit's rendered length is chosen by the service: two measurements
        # showed it rendering at Omni's maximum regardless of the source, and
        # edit_video's own dry_run quotes that maximum for exactly that reason.
        # Pricing the request's duration under-quoted a ~6s edit by 40%, which
        # is the one thing a pre-flight may not do. include_audio mirrors the
        # tool's own call so the two quotes are identical down to the detail.
        return _estimate_video_cost(
            model,
            float(omni_spec(model).max_duration_seconds),
            _omni_resolution(model),
            False,
        )

    if tool == "extend_video_omni":
        # Every turn is its own interaction and its own render, and each
        # renders the ASSEMBLED clip — measured: a 3.01s source extended once
        # returned 13.01s — so every turn re-bills the footage before it. The
        # source length is not knowable here, so the documented maximum is
        # assumed; a shorter real source quotes lower, and assuming one would
        # under-quote. The tool's own dry_run measures it and is sharper.
        turns = int(params.get("times", 1))
        # The same list the tool's dry_run prices, for the backend omni will
        # actually run on. This quoted the model's 10s fallback list on both
        # backends, so on a Vertex-only server the plan said $9.14 for three
        # turns while extend_video_omni's own quote said $12.18 -- and a
        # max_cost_usd taken from the plan was refused by the tool.
        lengths = omni_extension_priced_lengths(
            omni_spec(model),
            # The recorded clip, as the tool's own quote reads it; anything
            # else is unknown and priced at the bound.
            request.previous_interaction_seconds
            if "previous_interaction_id" in params
            else None,
            turns,
            vertexai=_omni_serves_on_vertex(request),
            per_turn_seconds=params.get("duration_seconds"),
        )
        if not lengths:
            return None
        return _aggregate_video_cost(
            model,
            lengths[0],
            sum(lengths),
            len(lengths),
            _omni_resolution(model),
            request.needs_audio,
        )

    resolution = (
        _omni_resolution(model)
        if is_omni_model(model)
        else (
            OMNI_RESOLUTION
            if tool in FIXED_720P_TOOLS
            else (request.resolution or "720p")
        )
    )
    duration = float(params.get("duration_seconds", request.clip_duration_seconds))

    if tool == "generate_clip":
        beats = len(params.get("beats", []))
        beat_duration = float(
            params["beats"][0].get("duration_seconds", duration) if beats else duration
        )
        bridges = beats - 1 if (params.get("add_bridges") and beats > 1) else 0
        # A bridge is its own render, but always a 4s one — charging it at the
        # beat length quoted $4.00 against the tool's $3.20 for identical
        # params.
        return _aggregate_video_cost(
            model,
            beat_duration,
            beat_duration * beats + CLIP_BRIDGE_SECONDS * bridges,
            beats + bridges,
            resolution,
            request.needs_audio,
        )

    if tool == "loop_extend":
        times = int(params.get("times", 1))
        # loop_extend extends a clip the caller already has: it renders the
        # extensions and nothing else. Billing a base render here quoted
        # $1.50 for a call the tool itself quotes at $0.70. When the planner
        # also proposes the seed render (an implied extension), that step
        # carries its own cost.
        total = VEO_EXTENSION_SECONDS * times
        return _aggregate_video_cost(
            model,
            float(VEO_EXTENSION_SECONDS),
            total,
            times,
            # Extended output is 720p regardless of what the base clip asked
            # for (documented on loop_extend).
            OMNI_RESOLUTION,
            request.needs_audio,
        )

    return _estimate_video_cost(model, duration, resolution, request.needs_audio)


# Size a storyboard previews at. generate_storyboard leaves image_size unset by
# default (its docstring: "storyboard frames rarely need to be large"), which
# prices at the model default of 1K; fixing 1K here keeps the board cheap and
# is the only size flash-lite can render at all. Cost parity with the tool's
# dry_run holds because both price the same (model, 1K, shot count).
_STORYBOARD_IMAGE_SIZE = "1K"


def _plan_storyboard_route(request: ResolvedRequest) -> RoutedCall | None:
    """A generate_storyboard previz route for a storyboard-flavoured request.

    generate_storyboard is the intended step between an idea and generate_clip:
    one keyframe still per shot, so a multi-shot sequence can be reviewed for
    framing and pacing before any per-second video spend — typically an order
    of magnitude cheaper than rendering the clip. It is otherwise undiscoverable
    through the planner, which is documented as the tool to call first.

    Offered only when the intent actually signals previsualization AND the
    sequence is genuinely multi-shot, so requests without that signal are
    unchanged. Priced through the same estimate_image_cost path
    generate_storyboard's own dry_run uses (one keyframe per shot), so the quote
    beside this route equals what the tool reports for the same shots and model.

    Returns None when the request is not a storyboard previz.
    """
    if not request.wants_storyboard or request.num_beats < 2:
        return None

    shots = min(request.num_beats, MAX_STORYBOARD_SHOTS)

    # Choose the keyframe model with the image planner's own scoring, so the
    # board respects budget and — the point of the paired text-rendering fix —
    # a captions/slug-line brief prefers a stronger text renderer. Size is fixed
    # at 1K, so quality counts as demanded only when the caller wants "best".
    quality_is_demanded = request.budget == "best"
    scored: list[tuple[float, ModelProfile]] = []
    for model in sorted(LIVE_IMAGE_MODELS):
        profile = _IMAGE_PROFILES[model]
        demanded = (
            (profile.text_rendering_index,) if request.needs_text_rendering else ()
        )
        score = _score_route(
            profile,
            request,
            demanded_capabilities=demanded,
            quality_is_demanded=quality_is_demanded,
            default_model=DEFAULT_IMAGE_MODEL,
        )
        scored.append((score, profile))
    # Same total order as _rank: score desc, then fidelity desc, then id asc.
    scored.sort(key=lambda sp: (-sp[0], -sp[1].fidelity_index, sp[1].model))
    _, profile = scored[0]
    model = profile.model

    # Rank the route on what the board actually costs, not the keyframe model's
    # delivery tier. Model selection above is budget-aware (cheap picks lite,
    # "best" picks pro), but the board is a set of cheap 1K stills whatever the
    # model — so when a slug-line/caption brief forces the pricier text renderer
    # the route must not sink BELOW the very per-second video renders it exists
    # to precede. Scoring budget alignment from the cheap end (cost_index 0.0)
    # keeps the previz at the top of the plan for a board deliverable while
    # leaving cheap boards (already lite) unchanged.
    demanded = (profile.text_rendering_index,) if request.needs_text_rendering else ()
    score = _score_route(
        profile,
        request,
        demanded_capabilities=demanded,
        quality_is_demanded=quality_is_demanded,
        default_model=DEFAULT_IMAGE_MODEL,
        cost_index_override=0.0,
    )

    cost = _estimate_image_cost(model, _STORYBOARD_IMAGE_SIZE, shots)
    aspect_ratio = request.aspect_ratio or "16:9"

    params: dict[str, Any] = {
        "shots": [
            {
                "prompt": request.intent,
                # The beat's share: the shot list feeds generate_clip.
                "duration_seconds": _beat_seconds(request),
            }
            for _ in range(shots)
        ],
        "model": model,
        "aspect_ratio": aspect_ratio,
        "image_size": _STORYBOARD_IMAGE_SIZE,
    }

    caveats: list[str] = [
        f"Replace each of the {shots} shot prompts with that shot's own "
        "description — they are seeded from the intent.",
        "Keyframes are stills, not the clip: the board previews framing and "
        "pacing, then its shot list feeds straight into generate_clip.",
    ]
    if request.num_beats > MAX_STORYBOARD_SHOTS:
        caveats.append(
            f"{request.num_beats} shots were asked for but generate_storyboard "
            f"renders at most {MAX_STORYBOARD_SHOTS}; planned at "
            f"{MAX_STORYBOARD_SHOTS}. Split the sequence across several boards."
        )
    if request.needs_text_rendering and model != "gemini-3-pro-image":
        caveats.append(
            "gemini-3-pro-image renders on-panel text (slug lines, captions) "
            "most legibly; expect to re-roll glyphs on this one."
        )

    rationale = (
        f"generate_storyboard on {model}: render {shots} keyframe stills to "
        "previsualize the sequence"
    )
    if cost is not None:
        rationale += f" for about ${cost.usd:.2f}"
    rationale += (
        " before committing to the per-second video render; the shot list then "
        f"feeds straight into generate_clip. budget={request.budget}"
    )

    return RoutedCall(
        tool="generate_storyboard",
        model=model,
        params=params,
        score=score,
        rationale=rationale,
        caveats=tuple(caveats),
        cost=cost,
    )


def _plan_video(
    request: ResolvedRequest,
) -> tuple[
    tuple[RoutedCall, ...],
    tuple[RejectedRoute, ...],
    tuple[RoutingConflict, ...],
    tuple[str, ...],
]:
    """Build the ranked video routes plus rejections, conflicts and notes."""
    routes: list[RoutedCall] = []
    rejected: list[RejectedRoute] = []
    conflicts: list[RoutingConflict] = []
    notes: list[str] = []

    tool = _select_video_tool(request)
    needs = _video_needs(request, tool)

    # Backend-level contradiction: no model choice can fix this one.
    if request.wants_gcs_output and request.backend == "gemini_api":
        conflicts.append(
            RoutingConflict(
                code="gcs_output_on_gemini_api",
                detail=(
                    "output_gcs_uri was requested but the server is on the Gemini "
                    "Developer API, which has no Cloud Storage destination and "
                    "rejects an explicit GCS target."
                ),
                resolution=(
                    "Run against Vertex AI for GCS output, or accept the inline "
                    "result and upload it yourself."
                ),
            )
        )

    # Runtime beyond what a Veo extension chain can reach.
    if (
        request.total_duration_seconds is not None
        and request.total_duration_seconds > VEO_MAX_EXTENDED_SECONDS
        and request.num_beats <= 1
    ):
        conflicts.append(
            RoutingConflict(
                code="duration_exceeds_extension_ceiling",
                detail=(
                    f"{request.total_duration_seconds:g}s exceeds the "
                    f"{VEO_MAX_EXTENDED_SECONDS}s ceiling of an 8s Veo clip plus "
                    f"{VEO_MAX_EXTENSIONS} x {VEO_EXTENSION_SECONDS}s extensions."
                ),
                resolution=(
                    "Render several beats with generate_clip and splice them "
                    "downstream (e.g. with a cutting MCP)."
                ),
            )
        )

    # A transition/bridge needs two endpoints. This fires both when the ladder
    # chose one of those tools and when the caller asked for a transition in
    # prose without supplying the frames — the second case would otherwise
    # quietly become a plain image-to-video render. Multi-beat requests are
    # exempt: generate_clip(add_bridges=True) makes its own endpoints.
    endpoint_tool_requested = tool in ("generate_transition", "generate_bridge") or (
        (request.wants_transition or request.wants_bridge) and request.num_beats <= 1
    )
    if endpoint_tool_requested and not (
        request.has_first_frame and request.has_last_frame
    ):
        named_tool = (
            tool
            if tool in ("generate_transition", "generate_bridge")
            else ("generate_bridge" if request.wants_bridge else "generate_transition")
        )
        if request.has_first_frame:
            missing = "only the first frame is available"
        elif request.has_last_frame:
            missing = "only the last frame is available"
        else:
            missing = "neither frame is available"
        conflicts.append(
            RoutingConflict(
                code="transition_requires_two_endpoints",
                detail=f"{named_tool} needs both endpoints, but {missing}.",
                resolution=(
                    "Supply first_frame_uri and last_frame_uri (or two clip URIs "
                    "for generate_bridge)."
                ),
            )
        )

    if request.previous_interaction_id is not None and request.needs_seed:
        conflicts.append(
            RoutingConflict(
                code="conversational_edit_without_seed_support",
                detail=(
                    "Conversational editing only exists on the Omni models, "
                    "which have no seed parameter."
                ),
                resolution=(
                    "Drop the seed requirement, or re-render on Veo with a fixed "
                    "seed instead of editing."
                ),
            )
        )

    if request.pinned_model is not None and request.pinned_model not in _VIDEO_PROFILES:
        conflicts.append(
            RoutingConflict(
                code="pinned_model_not_routable",
                detail=(
                    f"pinned_model={request.pinned_model} is not a live video model "
                    f"({', '.join(sorted(_VIDEO_PROFILES))})."
                ),
                resolution=f"Plan against {DEFAULT_VIDEO_MODEL} instead.",
            )
        )

    # Candidate models for the chosen tool. Omni is only ever a candidate for
    # the tools that actually run it — including loop_extend, which _route_tool
    # turns into extend_video_omni for an omni model (the preview model is
    # then rejected by the extension capability rule, as it should be).
    if tool in (
        "generate_video",
        "generate_video_omni",
        "edit_video",
        "loop_extend",
        # Two stills and a prompt is exactly omni 1.2's interpolation, so it
        # belongs in the running for a transition. The preview model is then
        # excluded by the first/last-frame capability rule, as it should be.
        # generate_bridge stays Veo-only: it SAMPLES its endpoints out of two
        # clips, which is that tool's own decoding work rather than a model
        # capability.
        "generate_transition",
    ):
        candidates = sorted(_VIDEO_PROFILES)
    elif tool == "generate_clip":
        # generate_clip's animatic mode runs on omni, but that is a preview
        # workflow step (see _build_workflow), not a way to deliver the clip.
        candidates = sorted(LIVE_VIDEO_MODELS)
    else:
        candidates = sorted(LIVE_VIDEO_MODELS)

    # A continuation runs on the model that holds the interaction and on no
    # other: an interaction cannot change models, and the tools refuse any
    # omni_model but that one. One route per tool, on that model -- a route
    # per omni model handed over calls the tool refuses, and the top-ranked
    # one was refused whenever the interaction lived on the other model.
    continuing = request.previous_interaction_id is not None
    continuation_task = "extend" if tool == "loop_extend" else "edit"
    lost = (
        _lost_interaction_conflict(request, continuation_task)
        or _interaction_backend_conflict(request, continuation_task)
        if continuing
        else None
    )
    continued_on = _continuation_model(request) if continuing else None
    recorded = request.previous_interaction_model
    if lost is not None:
        conflicts.append(lost)
    elif (
        recorded is not None
        and request.pinned_model is not None
        and is_omni_model(request.pinned_model)
        and request.pinned_model != continued_on
    ):
        conflicts.append(
            RoutingConflict(
                code="pinned_model_cannot_continue_interaction",
                detail=(
                    f"pinned_model={request.pinned_model} cannot continue "
                    f"interaction {request.previous_interaction_id}, which was "
                    f"created on {recorded}: an interaction cannot change models."
                ),
                resolution=(
                    f"Drop pinned_model (or pin {continued_on}) to continue it on "
                    f"{continued_on}, or start a new interaction from the clip it "
                    "saved by passing it as input_video_uri to "
                    f"{_SAVED_CLIP_TOOLS[continuation_task][0]}."
                ),
            )
        )

    chains_runtime = _veo_chains_runtime(request)
    too_long = _longest_transition_conflict(tool, request, candidates)
    if too_long is not None:
        conflicts.append(too_long)

    # The fix for each exclusion, kept: when EVERY candidate is excluded the
    # plan used to return no routes and no conflicts, and the resolutions the
    # rules had computed were discarded on the way out.
    exclusion_fixes: list[str] = []
    for model in candidates:
        profile = _VIDEO_PROFILES[model]

        sunset = DEPRECATED_VIDEO_MODELS.get(model)
        if sunset and omni_preview_is_sunset():
            # Demoted until the day; excluded from it. A plan that offers a
            # switched-off endpoint, however low it ranks, is a plan with a
            # route that cannot run.
            rejected.append(
                RejectedRoute(
                    model=model,
                    reason=f"{model} excluded: its endpoint was switched off on {sunset}.",
                    tool=_route_tool(tool, model),
                )
            )
            exclusion_fixes.append(f"Use {DEFAULT_OMNI_MODEL}, the GA successor.")
            continue

        # Past VEO_SNAP_MAX_SECONDS a Veo route chains a seed and extensions
        # instead of rendering one clip short; omni routes render the whole
        # runtime, or are excluded for their ceiling.
        model_tool: ToolName = (
            "loop_extend" if chains_runtime and not is_omni_model(model) else tool
        )
        model_needs = needs if model_tool == tool else _video_needs(request, model_tool)
        capability_hit = _capability_rejection(model, model_needs)
        duration_reason = _duration_rejection(model, model_needs)
        transition_reason = _transition_too_long(tool, model, request)
        backend_hit = _backend_rejection(model, request)
        source_hit = _omni_input_rejection(
            model_tool, model, request
        ) or _veo_reference_rejection(model, request)
        reason = None
        code = None
        resolution = None
        if capability_hit is not None:
            rule, reason = capability_hit
            code, resolution = rule.code, rule.resolution
            if (
                recorded is not None
                and model == continued_on
                and _capability_rejection(DEFAULT_OMNI_MODEL, needs) is None
            ):
                # The interaction's own model cannot do this, and no other
                # model can continue it; the clip it saved can still go to
                # one that can. The rule's own fix (a Veo route, say) would
                # drop the interaction the request is about.
                resolution = (
                    "An interaction cannot change models. "
                    + _saved_clip_resolution(continuation_task, request)
                )
        elif duration_reason is not None:
            reason = duration_reason
            code = "duration_out_of_range"
            resolution = (
                f"Use a Veo model, which renders "
                f"{'/'.join(str(d) for d in VEO_DURATIONS_SECONDS)}s clips and can "
                "be extended."
            )
            longer = _omni_model_rendering(needs.clip_duration_seconds, model)
            if longer is not None:
                # Omni ceilings differ by model (10s on the preview, 15s on
                # 1.2), so "use Veo" is not the nearest fix for a pinned
                # preview asked for 13s: 1.2 renders it in one call, and for
                # less.
                resolution = (
                    f"Use {longer}, which renders up to "
                    f"{_VIDEO_CAPABILITIES[longer].max_duration_seconds:g}s in "
                    f"one call. Or: {resolution[0].lower()}{resolution[1:]}"
                )
        elif transition_reason is not None:
            reason = transition_reason
            code = "transition_too_long"
            resolution = too_long.resolution if too_long is not None else None
        elif backend_hit is not None:
            reason, resolution = backend_hit
            code = "backend_unsupported"
        elif source_hit is not None:
            code, reason, resolution = source_hit

        if (
            reason is None
            and continuing
            and is_omni_model(model)
            and (lost is not None or model != continued_on)
        ):
            # Capable, but not the model holding the interaction. A model
            # that cannot do this at all keeps its own reason above.
            if (
                lost is not None
                and lost.code == "interaction_backend_unreachable"
                and model == continued_on
            ):
                why = (
                    f"interaction {request.previous_interaction_id} lives on "
                    f"{_backend_label(str(request.previous_interaction_backend))}"
                    ", which this server cannot reach"
                )
            elif recorded is not None:
                why = (
                    f"interaction {request.previous_interaction_id} was created "
                    f"on {recorded}, and an interaction cannot change models"
                )
            elif _continuation_omni_model_param(request) is not None:
                why = f"the plan is pinned to {continued_on}"
            else:
                why = (
                    f"one {_route_tool(tool, model)} route covers the "
                    "interaction, with omni_model left unset so the tool "
                    "continues on the model that made it"
                )
            rejected.append(
                RejectedRoute(
                    model=model,
                    reason=f"{model} not planned: {why}.",
                    tool=_route_tool(tool, model),
                )
            )
            continue

        if reason is not None:
            route_tool = _route_tool(model_tool, model)
            rejected.append(RejectedRoute(model=model, reason=reason, tool=route_tool))
            # A model that cannot read an interaction has no fix to offer a
            # continuation of one: "use Omni" is what the request already is.
            if resolution and not (continuing and not is_omni_model(model)):
                exclusion_fixes.append(resolution)
            if request.pinned_model == model:
                conflicts.append(
                    RoutingConflict(
                        code=f"pinned_model_{code}",
                        detail=reason,
                        resolution=resolution or "Choose a different model.",
                    )
                )
            continue

        route_tool = _route_tool(model_tool, model)

        params, caveats = _video_params(route_tool, model, request)

        demanded: list[float] = []
        reasons: list[str] = []
        # Every route still here meets the size rule, so under budget=cheap
        # the size is not a capability to prefer the higher-fidelity (and
        # dearer) model for: scored as one on top of quality_ceiling, it
        # ranked Veo Fast ($1.80) above Omni 1.2 ($1.26) for a cheap 4K brief.
        if (
            request.needs_4k or request.resolution in ("1080p", "4K")
        ) and request.budget != "cheap":
            demanded.append(profile.fidelity_index)
            reasons.append(f"{request.resolution or '4K'} output")
        if request.num_reference_images > 0:
            demanded.append(profile.fidelity_index)
            reasons.append("subject consistency from reference images")
        if request.needs_audio:
            demanded.append(profile.fidelity_index)
            reasons.append("a controllable audio track")
        if chains_runtime:
            # The runtime is a stated demand, and a route that renders all of
            # it in one call owns it; a seed-and-extend chain reaches it in
            # two paid renders and overshoots to the next ~7s step, so it
            # only has some of it. Without this the default-model nudge kept
            # the Veo chain on top of a 12s ask that Omni 1.2 renders whole.
            if route_tool == "loop_extend":
                demanded.append(NEUTRAL_TERM)
            else:
                demanded.append(1.0)
                reasons.append(
                    f"the {request.total_duration_seconds:g}s runtime in one render"
                )

        score = _score_route(
            profile,
            request,
            demanded_capabilities=tuple(demanded),
            # Under budget=cheap a 4K ask is a met requirement, not a call for
            # fidelity; scored as one, Omni led a cheap 4K brief by 0.005.
            quality_is_demanded=request.budget == "best"
            or (request.needs_4k and request.budget != "cheap"),
            default_model=DEFAULT_VIDEO_MODEL,
        )

        rationale = f"{model} via {route_tool}: {profile.summary}"
        if reasons:
            rationale += f"; picked for {', '.join(reasons)}"
        rationale += f"; budget={request.budget}"

        routes.append(
            RoutedCall(
                tool=route_tool,
                model=model,
                params=params,
                score=score,
                rationale=rationale,
                caveats=tuple(caveats),
                cost=_video_cost(route_tool, model, request, params),
            )
        )

    if tool == "generate_video" and request.is_draft:
        notes.append(
            "generate_video(draft=True) is a shortcut to the omni route: it "
            f"renders the draft on {DEFAULT_OMNI_MODEL} at the length the Veo "
            "render will run (4, 6 or 8s) and reports which Veo-only "
            "parameters it ignored."
        )
    if request.num_beats > 1:
        # Reported at the count the emitted route actually carries, which
        # generate_clip's beat ceiling may have truncated.
        planned_beats = min(request.num_beats, MAX_CLIP_BEATS)
        notes.append(
            f"{planned_beats} beats render sequentially; the returned manifest "
            "is an ordered segment list for a downstream cutting MCP."
        )

    # A storyboard previz sits alongside the video routes: for a
    # previsualization-flavoured multi-shot brief it is the cheap first step,
    # ranked on the image-model scale so it lands ahead of the per-second
    # renders. Gated inside the helper, so non-storyboard requests are
    # untouched. A pinned video model still wins outright (the filter below
    # drops every non-pinned route, storyboard included).
    storyboard_route = _plan_storyboard_route(request)
    if storyboard_route is not None:
        routes.append(storyboard_route)

    ranked = _rank(routes, _VIDEO_PROFILES)

    # Same rule as the image planner: a surviving pin is the only plan.
    if request.pinned_model is not None:
        pinned_routes = tuple(r for r in ranked if r.model == request.pinned_model)
        if pinned_routes:
            for other in ranked:
                if other.model != request.pinned_model:
                    rejected.append(
                        RejectedRoute(
                            model=other.model,
                            # Named as planned, not as asked: a pin on a
                            # retired ID is planned as its replacement, and
                            # the notes say why.
                            reason=(
                                f"{other.model} not planned: the plan is "
                                f"pinned to {request.pinned_model}."
                            ),
                            tool=other.tool,
                        )
                    )
            ranked = pinned_routes

    if not ranked and rejected and not conflicts:
        # Every candidate was excluded and nothing else in the request was
        # contradictory, so the exclusions ARE the answer. Without this a
        # 4K multi-shot brief on the Gemini Developer API — where Veo cannot
        # render 4K and generate_clip cannot run omni — came back as an empty
        # plan with no explanation. The planner's own agreement test calls
        # that silence, and it is right to.
        reasons = list(dict.fromkeys(r.reason for r in rejected))
        # A fix that points at a model this same plan rejected is no fix: the
        # preview's "use 1.2" sat beside 1.2's own refusal at the ceiling.
        rejected_models = {r.model for r in rejected}
        fixes = [
            fix
            for fix in dict.fromkeys(exclusion_fixes)
            if not any(fix.startswith(f"Use {m},") for m in rejected_models)
        ]
        # Named by kind, not by the generic tool: an Omni extension's route is
        # extend_video_omni, and "every candidate for loop_extend" named a
        # Veo tool the request was never routed to.
        conflicts.append(
            RoutingConflict(
                code="no_model_can_serve_request",
                detail=("Every candidate model was excluded: " + " ".join(reasons)),
                resolution=(
                    " ".join(fixes)
                    if fixes
                    else "Relax one of the constraints named above."
                ),
            )
        )

    return (
        ranked,
        tuple(rejected),
        tuple(conflicts),
        tuple(notes),
    )


# ============================================================================
# Workflow recommendations
# ============================================================================


def _animatic_rationale(
    beats: int,
    animatic_cost: CostEstimateLike | None,
    render_cost: CostEstimateLike | None,
    model: str = OMNI_MODEL,
    resolution: str = OMNI_RESOLUTION,
) -> str:
    """Explain the animatic preflight without claiming a saving that is not there.

    Live testing caught this rationale selling the animatic as the cheap
    preview of an expensive render, which was only true against the standard
    Veo tier: the preview model billed $0.10136/s against Veo 3.1 Fast's
    $0.10/s, so 3 x 8s beats cost $2.43 as an animatic against $2.40 for the
    real thing. The animatic now renders on 1.2 at 360p, which is cheaper than
    every Veo tier, but rates move, so the saving is still read off the two
    quotes rather than assumed. The preview keeps its place either way
    (catching a bad creative call before the delivery render is paid for is
    worth something on its own), but the claim has to match the arithmetic.

    Args:
        beats: How many beats the animatic renders.
        animatic_cost: Estimated cost of the omni preview, or None when
            pricing is unavailable.
        render_cost: Estimated cost of the delivery render it precedes, or
            None. This is the route's own quote, itself produced by
            ``_aggregate_video_cost``, so the workflow can never contradict
            the cost printed next to the route.

    Returns:
        The rationale for the animatic step: a saving only when the numbers
        show one, an explicit "does not save money" when they do not, and no
        economic claim at all when either side is unpriced.
    """
    lead = f"Preview all {beats} beats on {model} at {resolution} first"
    why = (
        "an animatic surfaces pacing and continuity problems before the "
        "delivery render is paid for."
    )

    if animatic_cost is None or render_cost is None or render_cost.usd <= 0:
        # Nothing to compare, so recommend the step on its creative merit
        # alone. A guessed saving is exactly the failure being fixed here.
        return f"{lead}: {why}"

    animatic_usd = animatic_cost.usd
    render_usd = render_cost.usd
    if animatic_usd < render_usd:
        return (
            f"{lead} (est. ${animatic_usd:.2f} against ${render_usd:.2f} for the "
            f"full render, saving ~${render_usd - animatic_usd:.2f}): {why}"
        )
    return (
        f"{lead} (est. ${animatic_usd:.2f}, about "
        f"{animatic_usd / render_usd:.2f}x the ${render_usd:.2f} full render — "
        "this does NOT save money): run it to catch a bad creative call before "
        "committing to the delivery render, not as a way to spend less."
    )


def _build_workflow(
    request: ResolvedRequest, routes: tuple[RoutedCall, ...]
) -> tuple[WorkflowStep, ...]:
    """Recommend a call sequence when one call is not the whole answer.

    The animatic-first pattern is the important one: a multi-beat clip is
    exactly the case where discovering a bad creative call after the delivery
    render is the costly mistake, and generate_clip(animatic=True) renders
    every beat on omni for the same review. Whether that preview is also
    *cheaper* depends on the Veo tier it precedes and is decided per plan by
    ``_animatic_rationale`` — omni is not categorically the cheap option.
    """
    if not routes:
        return ()
    # The animatic/seed-clip workflows are keyed off the primary VIDEO render.
    # A storyboard previz may now outrank it (it is the cheap first step, not
    # the delivery render), so skip past it to the top video route — for every
    # non-storyboard plan this is still routes[0], leaving existing behaviour
    # unchanged.
    render_routes = [route for route in routes if route.tool != "generate_storyboard"]
    if not render_routes:
        return ()
    best = render_routes[0]

    # When the storyboard previz is the top route, IT is the plan's cheap first
    # pass — so the workflow must lead with it, not with an animatic. Proposing a
    # top storyboard route AND an animatic-first workflow put two contradictory
    # previz steps in one answer, and the animatic's own rationale concedes it is
    # ~1x the delivery cost while the board previsualizes the same shots for a
    # fraction of it. Only fires when the storyboard out-ranks the delivery clip;
    # a plan where it does not keeps the animatic path untouched.
    storyboard_route = next(
        (route for route in routes if route.tool == "generate_storyboard"), None
    )
    if (
        storyboard_route is not None
        and routes[0] is storyboard_route
        and best.tool == "generate_clip"
    ):
        shots = len(storyboard_route.params.get("shots", ()))
        board_lead = f"Storyboard all {shots} shots on {storyboard_route.model} first"
        if storyboard_route.cost is not None and best.cost is not None:
            board_lead += (
                f" (est. ${storyboard_route.cost.usd:.2f} against "
                f"${best.cost.usd:.2f} for the full clip render)"
            )
        # Preview in the aspect ratio of the DELIVERABLE. generate_clip defaults
        # to 9:16 while the storyboard defaults to 16:9, so without this the
        # board reviewed landscape while the clip rendered vertical — reviewing
        # the wrong frame defeats the point of the previz. When the request
        # names an aspect both already carry it; this only reconciles the
        # split defaults.
        board_params = storyboard_route.params
        clip_aspect = best.params.get("aspect_ratio")
        if clip_aspect and board_params.get("aspect_ratio") != clip_aspect:
            board_params = {**board_params, "aspect_ratio": clip_aspect}
        return (
            WorkflowStep(
                order=1,
                tool="generate_storyboard",
                params=board_params,
                rationale=(
                    f"{board_lead}: review framing and pacing as cheap keyframe "
                    "stills before paying for any per-second video."
                ),
            ),
            WorkflowStep(
                order=2,
                tool=best.tool,
                params=best.params,
                rationale=(
                    "Once the keyframes read correctly, render the delivery clip "
                    f"on {best.model}; the board's shot list feeds generate_clip."
                ),
            ),
        )

    if (
        best.tool == "loop_extend"
        and (request.extension_implied or _veo_chains_runtime(request))
        and request.source_video_uri is None
    ):
        # A continuous shot longer than one Veo render can only be made by
        # generating a seed clip and extending it — a multi-beat clip would
        # have cuts. The caller has no video yet, so the sequence IS the plan;
        # without it the top route's required video_uri cannot exist.
        seed_params: dict[str, Any] = {
            "prompt": request.intent,
            "model": best.params.get("model", best.model),
            "duration_seconds": float(VEO_MAX_CLIP_SECONDS),
            # The chain's audio choice: generate_video defaults to none, so a
            # seed left at the default went silent under extensions with audio.
            "include_audio": request.needs_audio,
        }
        if "aspect_ratio" in best.params:
            seed_params["aspect_ratio"] = best.params["aspect_ratio"]
        extend_params = dict(best.params)
        extend_params["video_uri"] = "<video_url from step 1>"
        if request.backend == "vertex" and request.video_gcs_bucket_configured is False:
            # The step cannot run without one here; workflow params carry
            # placeholders for what the caller must fill in.
            extend_params["output_gcs_uri"] = "<gs:// destination for the extension>"
        # Each step says what it costs, since the route's quote is the
        # extensions' alone.
        seed_cost = _veo_seed_cost(str(seed_params["model"]), request)
        seed_priced = f" (est. ${seed_cost.usd:.2f})" if seed_cost is not None else ""
        extend_priced = (
            f" (est. ${best.cost.usd:.2f}; about "
            f"${seed_cost.usd + best.cost.usd:.2f} for the chain)"
            if seed_cost is not None and best.cost is not None
            else ""
        )
        return (
            WorkflowStep(
                order=1,
                tool="generate_video",
                params=seed_params,
                rationale=(
                    f"Render the opening {VEO_MAX_CLIP_SECONDS}s seed clip"
                    f"{seed_priced} — a continuous shot longer than one Veo "
                    "render can only be made by extending an existing clip."
                ),
            ),
            WorkflowStep(
                order=2,
                tool="loop_extend",
                params=extend_params,
                rationale=(
                    f"Extend the seed clip {extend_params.get('times', 1)} "
                    f"time(s) (~7s each){extend_priced} to reach the requested "
                    "runtime, using step 1's video_url."
                ),
            ),
        )

    if (
        best.tool == "extend_video_omni"
        and request.extension_implied
        and request.source_video_uri is None
        and request.previous_interaction_id is None
    ):
        # The omni twin of the Veo chain above: no clip exists yet, so the
        # chain opens with the seed render it continues conversationally.
        model = str(best.params.get("omni_model", best.model))
        seed_params: dict[str, Any] = {
            "prompt": request.intent,
            "omni_model": model,
            "duration_seconds": _omni_seed_seconds(omni_spec(model)),
            "aspect_ratio": (
                request.aspect_ratio
                if request.aspect_ratio in VIDEO_ASPECT_RATIOS
                else "16:9"
            ),
        }
        if "resolution" in best.params:
            seed_params["resolution"] = best.params["resolution"]
        extend_params = dict(best.params)
        extend_params["previous_interaction_id"] = "<interaction_id from step 1>"
        seed_cost = _omni_seed_cost(model, request)
        seed_priced = f" (est. ${seed_cost.usd:.2f})" if seed_cost is not None else ""
        extend_priced = (
            f" (est. ${best.cost.usd:.2f}, an upper bound until the tool "
            "measures the seed)"
            if best.cost is not None
            else ""
        )
        return (
            WorkflowStep(
                order=1,
                tool="generate_video_omni",
                params=seed_params,
                rationale=(
                    f"Render the opening {seed_params['duration_seconds']:g}s "
                    f"seed clip on {model}{seed_priced}: extend_video_omni "
                    "continues existing footage, and none exists yet."
                ),
            ),
            WorkflowStep(
                order=2,
                tool="extend_video_omni",
                params=extend_params,
                rationale=(
                    f"Extend it {extend_params.get('times', 1)} time(s)"
                    f"{extend_priced} to reach the requested runtime, "
                    "continuing step 1's interaction_id."
                ),
            ),
        )

    if best.tool in ("generate_video", "generate_video_omni"):
        # Draft-then-finalize for a single render. Only worth a step when the
        # delivery render is dear enough that discovering a bad creative call
        # after paying for it hurts — and only when the preview is genuinely
        # cheaper, which is exactly what omni's 360p draft tier makes possible.
        # A caller who has already said the render is throwaway needs no
        # preview of a preview.
        if (
            request.is_draft
            or best.model == DEFAULT_OMNI_MODEL
            and (best.params.get("resolution") == OMNI_DRAFT_RESOLUTION)
        ):
            return ()
        delivery_resolution = best.params.get("resolution") or request.resolution
        dear = (best.cost is not None and best.cost.usd >= ANIMATIC_MIN_COST_USD) or (
            delivery_resolution in ("1080p", "4K")
        )
        if not dear:
            return ()
        draft_params: dict[str, Any] = {
            "prompt": request.intent,
            "omni_model": DEFAULT_OMNI_MODEL,
            "resolution": OMNI_DRAFT_RESOLUTION,
            "duration_seconds": best.params.get(
                "duration_seconds", request.clip_duration_seconds
            ),
        }
        if "aspect_ratio" in best.params:
            # Preview in the deliverable's framing; reviewing the wrong frame
            # defeats the point.
            draft_params["aspect_ratio"] = best.params["aspect_ratio"]
        for frame in ("first_frame_uri", "last_frame_uri"):
            if frame in best.params:
                draft_params[frame] = best.params[frame]
        draft_cost = _estimate_video_cost(
            DEFAULT_OMNI_MODEL,
            float(draft_params["duration_seconds"]),
            OMNI_DRAFT_RESOLUTION,
            False,
        )
        if draft_cost is None or (
            best.cost is not None and draft_cost.usd >= best.cost.usd
        ):
            return ()
        lead = f"Preview on {DEFAULT_OMNI_MODEL} at {OMNI_DRAFT_RESOLUTION} first"
        if best.cost is not None:
            lead += (
                f" (est. ${draft_cost.usd:.2f} against ${best.cost.usd:.2f} for "
                f"the delivery render, saving ~${best.cost.usd - draft_cost.usd:.2f})"
            )
        return (
            WorkflowStep(
                order=1,
                tool="generate_video_omni",
                params=draft_params,
                rationale=(
                    f"{lead}: 360p is a third of omni's 720p price and the "
                    "cheapest render available here, so the composition, "
                    "motion and timing can be judged before the "
                    f"{delivery_resolution or 'delivery'} render is paid for. "
                    "Keep the returned interaction_id — the same clip can be "
                    "edited conversationally or extended from there."
                ),
            ),
            WorkflowStep(
                order=2,
                tool=best.tool,
                params=best.params,
                rationale=(
                    "Once the draft reads correctly, render the deliverable on "
                    f"{best.model}"
                    + (f" at {delivery_resolution}." if delivery_resolution else ".")
                ),
            ),
        )

    if best.tool != "generate_clip":
        return ()

    beats = len(best.params.get("beats", []))
    expensive = beats >= ANIMATIC_MIN_BEATS or (
        best.cost is not None and best.cost.usd >= ANIMATIC_MIN_COST_USD
    )
    if not expensive:
        return ()

    animatic_params = dict(best.params)
    animatic_params["animatic"] = True
    # An animatic previews the beats only; generate_clip ignores bridges
    # in animatic mode, so the preview step should not ask for them.
    animatic_params["add_bridges"] = False
    # Preview at 360p on the model that has resolutions. This is what turns
    # the animatic from a pass that costs about what the delivery render costs
    # — which is what its own rationale had to concede — into a third of it.
    # Naming the resolution is what selects the model; generate_clip's
    # animatic_resolution is documented that way.
    animatic_params["animatic_resolution"] = OMNI_DRAFT_RESOLUTION
    animatic_model = DEFAULT_OMNI_MODEL
    beat_duration = float(best.params["beats"][0].get("duration_seconds", 0.0))
    animatic_cost = _aggregate_video_cost(
        animatic_model,
        beat_duration,
        beat_duration * beats,
        beats,
        OMNI_DRAFT_RESOLUTION,
        False,
    )

    animatic_step = WorkflowStep(
        order=1,
        tool="generate_clip",
        params=animatic_params,
        # Both sides of the comparison come from _aggregate_video_cost:
        # best.cost is the generate_clip route's own aggregate quote.
        rationale=_animatic_rationale(
            beats, animatic_cost, best.cost, animatic_model, OMNI_DRAFT_RESOLUTION
        ),
    )
    if request.is_draft:
        # The caller already said this render is throwaway, so the animatic is
        # the deliverable, not a preflight step.
        return (animatic_step,)

    return (
        animatic_step,
        WorkflowStep(
            order=2,
            tool=best.tool,
            params=best.params,
            rationale=(
                "Re-run with animatic=False once the beats read correctly, to "
                f"render the delivery clip on {best.model}."
            ),
        ),
    )


# ============================================================================
# Upscaling
# ============================================================================

# upscale_video_omni refuses a longer source ("Standalone upscaling supports
# videos up to 15 seconds"); mirrored here so a plan never chains an upscale
# onto a render the tool will refuse.
OMNI_UPSCALE_MAX_SOURCE_SECONDS = omni_source_limit_seconds(
    omni_spec(OMNI_1_2_MODEL), task=OMNI_TASK_UPSCALE
)

# The resolutions an upscale can produce, and the ones only an upscale's
# output (never its source) can be.
_UPSCALE_TARGETS: tuple[str, ...] = ("720p", "1080p", "4K")
_UPSCALE_ONLY_RESOLUTIONS: tuple[str, ...] = ("1080p", "4K")

# Upscaling as an operation on a video, not the adjective ("an upscale hotel
# lobby"): the verb followed by its object or its destination, or the bare
# verb ending its clause after "then" ("make a video of a dog then upscale").
# The bare form takes "then" alone: after "and" or "finally" a clause-final
# "upscale" is as often the adjective ("a lobby that feels modern and
# upscale"), and misreading it plans an upscale nobody asked to pay for.
# Even after "then" it is the adjective when the "then" orders two looks,
# which _upscale_request checks.
_UPSCALE_PATTERN = re.compile(
    r"\b(?:upscale|upscaling|upsample|upsampling)\s+"
    r"(?:(?:my|our|this|that|the|a|an|existing|saved|approved|source|input|generated|previous|rendered)\s+)*"
    r"(?:video|clip|footage|draft|render|result|output|it|to|into|from|360p|720p|1080p|4k)\b"
    r"|\b(?:video|clip|footage)\s+(?:upscaling|upsampling)\b"
    r"|(?<=\bthen\s)(?:upscale|upsample)\b(?=\s*(?:[,.;:!?)\]]|$))",
    re.IGNORECASE,
)
_UPSCALE_DESTINATION_PATTERN = re.compile(
    r"\b(?:to|into)\s+(360p|720p|1080p|4k)\b", re.IGNORECASE
)

# A change of state in the words before a bare "then upscale": there the
# "then" orders two looks, and "upscale" is the second ("a street that is
# shabby at first, then upscale"; "a restaurant going from cheap then
# upscale"). Read as the verb, the brief lost its second half and gained a
# paid upscale step. Erring this way costs a caller who meant the verb
# nothing but a rephrase, which the plan's note asks for.
_STATE_CHANGE_PATTERN = re.compile(
    r"\b(?:first|initially|start(?:s|ing)?|begin(?:s|ning)?|turn(?:s|ing)?"
    r"|becom(?:e|es|ing)|go(?:es|ing)|get(?:s|ting)|grow(?:s|ing)"
    r"|chang(?:e|es|ing)|transform(?:s|ing)?|shift(?:s|ing)?|evolv(?:e|es|ing)"
    r"|morph(?:s|ing)?)\b",
    re.IGNORECASE,
)
_SENTENCE_END_PATTERN = re.compile(r"[.!?;]")


def _upscale_request(intent: str) -> tuple[re.Match[str] | None, str | None]:
    """The first _UPSCALE_PATTERN match that asks for an upscale, and a note.

    A bare clause-final "then upscale" is skipped where the words before it
    in its sentence describe a change of state, which makes it the
    adjective; the note says so, and how to ask for the upscale instead.
    """
    note = None
    for match in _UPSCALE_PATTERN.finditer(intent):
        word = match.group(0)
        if not word.isalpha():
            # The verb with its object or destination, or "video upscaling".
            return match, note
        before = _SENTENCE_END_PATTERN.split(intent[: match.start()])[-1]
        state = _STATE_CHANGE_PATTERN.search(before)
        if state is None:
            return match, note
        note = note or (
            f"'then {word}' is read as the adjective here, since the brief "
            f"describes a change of state ('{state.group(0)}'), so no upscale "
            f"step is planned. To upscale the render, say 'then {word} it'."
        )
    return None, note


_UPSCALE_SOURCE_PATTERN = re.compile(
    r"\b(?:from\s+)?(360p|720p|1080p|4k)\s+(?:to|into)\s+", re.IGNORECASE
)

# A request for a NEW video: an imperative creation verb whose object, within
# a few words, is a video noun, or a brief that opens by describing one ("a
# 4 shot reel about shoes, then upscale it"), the planner's usual verbless
# form. Base-form verbs only, so "the clip I generated" reads as a source
# rather than a request. A possessive, a demonstrative or "the" anywhere on
# the way to the noun ("render my draft video", "render the video I
# uploaded", "create a sharper version of the video"), or "I shot"/"you made"
# straight after it marks a clip that already exists. Erring that way is the
# safe direction: misreading an existing clip as a new one plans a render
# nobody asked to pay for, while the reverse leaves an upscale route that asks
# for its source.
_NEW_VIDEO_PATTERN = re.compile(
    r"(?:(?P<verb>\b(?:generate|create|make|render|produce|shoot|film|animate))\s+"
    r"(?!sure\b)|^\W*(?=an?\s))"
    r"(?:(?!(?:the|my|our|your|this|that|these|those|existing|saved|approved"
    r"|previous|uploaded)\b)[\w'-]+\s+){0,8}?"
    r"(?:video|clip|footage|animation|reel|movie)s?\b"
    r"(?!\s+(?:i|we|you)\s+(?:\w+ed|shot|made|took|had|have|got|already)\b)",
    re.IGNORECASE,
)

# What marks a clip as one the caller already has: an owner, or a word for
# capturing or keeping footage. Read in the words that follow a verbless
# brief ("a video of my dog that I recorded", "a video from my phone"), which
# has no imperative to say it asks for something new, and in the upscale's
# own object ("upscale my video of a cat"), which names its source outright.
_EXISTING_CLIP_PATTERN = re.compile(
    r"\b(?:my|our|your|his|her|their|i|we|existing|saved|uploaded|previous"
    r"|original|recorded|filmed|captured|shot|took|taken|downloaded)\b",
    re.IGNORECASE,
)

# _NEGATORS is written for nouns ("no audio"); a creation verb is negated by
# an auxiliary too ("don't make a video").
_CREATION_NEGATORS: tuple[str, ...] = _VERB_NEGATORS

# What joins a render clause to the upscale that follows it ("..., then
# upscale it", "... and upscale to 1080p", "... (then upscale it)"), stripped
# from the render's prompt.
_UPSCALE_JOINER_PATTERN = re.compile(
    r"(?:[\s,;:(\[\u2013\u2014-]|\b(?:and|then|afterwards|finally)\b)*$",
    re.IGNORECASE,
)
_CLAUSE_END_PATTERN = re.compile(r"[,.;!?)\]]")
# A word that only qualifies the upscale's target ("to 4K resolution"), so it
# leaves the render's prompt with the target.
_UPSCALE_TARGET_QUALIFIER = re.compile(
    r"\s+(?:resolution|res|quality|output|format)\b", re.IGNORECASE
)


def _normalize_upscale_resolution(value: str) -> str:
    """Spell an upscale resolution as this server does publicly: 4K, 1080p."""
    return "4K" if value.lower() == "4k" else value.lower()


def _split_render_and_upscale(
    intent: str, upscale: re.Match[str]
) -> tuple[str, str] | None:
    """Split a generate-then-upscale intent into (render, upscale) text.

    "Generate a video of a fox, then upscale it to 4K" upscales a video this
    plan has to make first; "Upscale my video to 4K" upscales one the caller
    already has. Only the first has a creation request ahead of the upscale,
    and for it this returns the intent without the upscale clause (and the
    word joining it) — what the render is planned and prompted from. Left in,
    "4K" made the render itself 4K and the prompt asked the video model to
    upscale. None when there is no creation request: a standalone upscale.
    """
    head = intent[: upscale.start()]
    # "Upscale my video of a cat" names a clip the caller has, whatever came
    # before it: chained onto an earlier render, that clip was never upscaled.
    if _EXISTING_CLIP_PATTERN.search(upscale.group(0)):
        return None
    creation = None
    for match in _NEW_VIDEO_PATTERN.finditer(head):
        if _negator_precedes(head[: match.start()].lower(), _CREATION_NEGATORS):
            continue
        # A verbless brief has no imperative asking for something new, so
        # the words after its noun decide: "a video of a cat, then upscale
        # it" is a brief, "a video from my phone: upscale it" is a source.
        if match.group("verb") is None and _EXISTING_CLIP_PATTERN.search(
            head[match.end() :]
        ):
            continue
        creation = match
    if creation is None:
        return None
    # The upscale clause ends at its target ("to 4K", plus a word qualifying
    # it), so a modifier after it stays with the render ("..., then upscale
    # it to 4K with audio" still asks for audio). With no target named it
    # runs to the next clause break; anything after that is still part of
    # the brief ("..., upscale it, and keep it moody").
    clause_end = _CLAUSE_END_PATTERN.search(intent, upscale.end())
    end = clause_end.start() if clause_end else len(intent)
    target = _UPSCALE_DESTINATION_PATTERN.search(intent, upscale.start(), end)
    if target is not None:
        end = target.end()
        qualifier = _UPSCALE_TARGET_QUALIFIER.match(intent, end)
        if qualifier is not None:
            end = qualifier.end()
    render = _UPSCALE_JOINER_PATTERN.sub("", head)
    # A parenthesised upscale clause takes its closing bracket with it ("a fox
    # (then upscale it to 4K)" left "a fox ("), and the clause's own end
    # punctuation is dropped where the render text already ends a sentence:
    # "a fox. Then upscale it to 4K." left "a fox..".
    tail = intent[end:]
    joiner = head[len(render) :]
    for opener, closer in (("(", ")"), ("[", "]")):
        if opener in joiner and closer in tail:
            at = tail.index(closer)
            tail = tail[:at] + tail[at + 1 :]
    if tail[:1] in (".", "!", "?", ";", ",") and render[-1:] in (".", "!", "?"):
        tail = tail[1:]
    render = (render + tail).strip()
    if not render:
        return None
    return render, intent[upscale.start() : end]


# What joins a declined upscale clause to the words before it ("..., but
# don't upscale it", "... it. Actually, don't upscale it"), removed with it.
_DECLINE_JOINER_PATTERN = re.compile(
    r"(?:[\s,;:(\[\u2013\u2014-]|\b(?:and|but|then|so|actually|also|please"
    r"|just|though|however|ok|okay)\b)*$",
    re.IGNORECASE,
)
_WORD_PATTERN = re.compile(r"\S+")


def _without_declined_upscales(intent: str) -> tuple[str, bool]:
    """The intent minus every upscale clause it declines, and whether it declines.

    "a 4K video of a dog, don't upscale it" planned the render with the whole
    text as its prompt, asking the video model not to upscale; and "..., then
    upscale it. actually don't upscale it" planned the upscale, because a
    mention counted as declined only when every mention was negated. The last
    mention decides, as a correction does, and a declined clause leaves the
    render's prompt from its negator to the end of its clause.

    Returns:
        The intent without its declined clauses, and True when the last
        mention of an upscale is a declined one. An intent that would be left
        empty is returned unchanged.
    """
    declined: list[tuple[int, int]] = []
    last_declines = False
    for match in _UPSCALE_PATTERN.finditer(intent):
        head = intent[: match.start()]
        clause = _CLAUSE_BREAK_PATTERN.split(head)[-1]
        clause_start = len(head) - len(clause)
        negator_at = None
        for word in reversed(
            list(_WORD_PATTERN.finditer(head, clause_start))[-_NEGATION_WINDOW:]
        ):
            cleaned = word.group(0).replace("\u2019", "'").strip("-\"'").lower()
            if cleaned in _CLAUSE_BREAK_WORDS:
                break
            if cleaned in _VERB_NEGATORS:
                negator_at = word.start()
                break
        last_declines = negator_at is not None
        if negator_at is None:
            continue
        # The declined clause runs from its own start -- a clause break, or a
        # conjunction like "but" -- when that is a few words back: cut at the
        # negator, "but do not upscale it" left "but do" in the prompt.
        lead = list(_WORD_PATTERN.finditer(intent, clause_start, negator_at))
        for at in range(len(lead) - 1, -1, -1):
            if lead[at].group(0).lower() in _CLAUSE_BREAK_WORDS:
                lead = lead[at:]
                break
        if lead and len(lead) <= _NEGATION_WINDOW + 1:
            negator_at = lead[0].start()
        end = len(intent)
        clause_end = _CLAUSE_END_PATTERN.search(intent, match.end())
        if clause_end is not None:
            end = clause_end.start()
        for word in _WORD_PATTERN.finditer(intent, match.end(), end):
            if word.group(0).lower() in _CLAUSE_BREAK_WORDS:
                end = word.start()
                break
        declined.append((negator_at, end))
    text = intent
    for start, end in reversed(declined):
        head = _DECLINE_JOINER_PATTERN.sub("", text[:start])
        tail = text[end:]
        if not head or head[-1:] in (".", "!", "?"):
            # The clause's own punctuation goes with it where the words before
            # it already end a sentence ("a fox. Don't upscale it." left "a
            # fox..").
            tail = tail.lstrip(" ,;:.!?")
        text = head + (" " if head and tail[:1].isalnum() else "") + tail
    text = " ".join(text.split())
    if not text:
        return intent, last_declines
    return text, last_declines


@dataclass(frozen=True)
class _PlannedSource:
    """The video an upscale follow-on runs on, when this plan renders it.

    Attributes:
        params: The source parameter, as a placeholder for the step's output.
        resolution: What that render comes back as.
        seconds: How long it was asked to be.
        delivered_seconds: How long upscale_video_omni will measure it: an
            Omni render comes back about one frame over its nominal length,
            and the tool prices and judges inline delivery on what it
            measures. None means the same as ``seconds``.
    """

    params: dict[str, Any]
    resolution: str
    seconds: float
    delivered_seconds: float | None = None


def _upscale_delivery(target: str, delivered: float | None) -> tuple[str | None, bool]:
    """The inline-delivery caveat for an upscale, and whether it is required.

    Judged as upscale_video_omni judges it: on Vertex AI, at the target, on
    the length the tool will measure, or at the 15s bound for a source it
    cannot measure.
    """
    return _inline_delivery_caveat(
        "upscale_video_omni",
        omni_spec(OMNI_1_2_MODEL),
        vertexai=True,
        resolution=target,
        task=OMNI_TASK_UPSCALE,
        output_seconds=(
            OMNI_UPSCALE_MAX_SOURCE_SECONDS if delivered is None else delivered
        ),
        seconds_are_bound=delivered is None,
    )


def _upscale_conflict(
    code: str, detail: str, resolution: str
) -> tuple[
    tuple[RoutedCall, ...],
    tuple[RejectedRoute, ...],
    tuple[RoutingConflict, ...],
    tuple[str, ...],
]:
    """An upscale plan with no route, only the conflict that explains why."""
    return ((), (), (RoutingConflict(code, detail, resolution),), ())


def _plan_upscale(
    request: ResolvedRequest, source: _PlannedSource | None = None
) -> tuple[
    tuple[RoutedCall, ...],
    tuple[RejectedRoute, ...],
    tuple[RoutingConflict, ...],
    tuple[str, ...],
]:
    """Route an explicit upscale to the operation that preserves the video.

    Args:
        request: The resolved request: backend, pin, target resolution, any
            source the caller supplied, and the words describing the upscale.
        source: The render this plan makes first, for a generate-then-upscale
            intent. Its output is the source, its resolution and length are
            known, and a pinned model governs that render, not the upscale.
    """
    # upscale_video_omni runs on Vertex AI whatever OMNI_BACKEND says (it
    # names the backend itself), so only a server with no Vertex client at
    # all cannot upscale.
    if request.backend == "gemini_api":
        return _upscale_conflict(
            "upscale_backend",
            "Standalone upscaling requires Vertex AI.",
            "Use a Vertex server and a saved video or Vertex interaction.",
        )
    if source is None:
        lost = _lost_interaction_conflict(request, OMNI_TASK_UPSCALE)
        if lost is not None:
            return ((), (), (lost,), ())
        if request.previous_interaction_backend == "gemini_api":
            # The upscale is a turn on the interaction, on Vertex AI, where an
            # interaction the Developer API minted does not exist.
            return _upscale_conflict(
                "upscale_interaction_backend",
                f"Interaction {request.previous_interaction_id} was created on "
                "the Gemini Developer API, and standalone upscaling runs only "
                "on Vertex AI, where that interaction does not exist.",
                _saved_clip_resolution(OMNI_TASK_UPSCALE, request),
            )
        recorded = request.previous_interaction_model
        if recorded is not None and canonical_omni_model(recorded) != OMNI_1_2_MODEL:
            # upscale_video_omni refuses it: the upscale is a turn on the
            # interaction, and only 1.2 upscales.
            return _upscale_conflict(
                "upscale_interaction_model",
                f"Interaction {request.previous_interaction_id} was created on "
                f"{recorded}; standalone upscaling runs on {OMNI_1_2_MODEL}, and "
                "an interaction cannot change models.",
                _saved_clip_resolution(OMNI_TASK_UPSCALE, request),
            )
    if (
        source is None
        and request.pinned_model
        and request.pinned_model != OMNI_1_2_MODEL
    ):
        return _upscale_conflict(
            "upscale_model",
            "Standalone upscaling requires Omni 1.2.",
            f"Use {OMNI_1_2_MODEL}.",
        )
    destination = _UPSCALE_DESTINATION_PATTERN.search(request.intent)
    target = _normalize_upscale_resolution(
        request.resolution or (destination.group(1) if destination else "1080p")
    )
    if target not in _UPSCALE_TARGETS:
        return _upscale_conflict(
            "upscale_resolution",
            "Upscaling targets must be 720p, 1080p or 4K.",
            "Select a higher output resolution than the source.",
        )
    params: dict[str, Any] = {"resolution": target}
    caveats: list[str] = []
    seconds: float | None = None
    # What the tool will measure, which is what it prices and what decides
    # its delivery; the nominal ``seconds`` is what the caveats name.
    delivered: float | None = None
    source_resolution: str | None = None
    if source is not None:
        params.update(source.params)
        source_resolution = source.resolution
        params["source_resolution"] = source_resolution
        seconds = source.seconds
        delivered = (
            source.delivered_seconds
            if source.delivered_seconds is not None
            else seconds
        )
        caveats.append("Vertex AI only.")
    else:
        recorded_seconds = (
            request.previous_interaction_seconds
            if request.previous_interaction_id
            else None
        )
        if recorded_seconds is not None:
            # The tool reads this same record, and prices and judges delivery
            # on it; the 15s bound here quoted a 6s clip at 2.5x the tool.
            if _omni_source_over_limit(
                OMNI_UPSCALE_MAX_SOURCE_SECONDS, recorded_seconds
            ):
                return _upscale_conflict(
                    "upscale_source_too_long",
                    f"Interaction {request.previous_interaction_id}'s clip is "
                    f"{recorded_seconds:g}s, as this server recorded it, and "
                    "standalone upscaling takes at most "
                    f"{OMNI_UPSCALE_MAX_SOURCE_SECONDS:g}s.",
                    f"Trim the clip it saved to {OMNI_UPSCALE_MAX_SOURCE_SECONDS:g}s "
                    "or less and pass it as input_video_uri.",
                )
            seconds = delivered = recorded_seconds
            caveats.append(
                f"Vertex AI only. The quote uses the {recorded_seconds:g}s this "
                f"server recorded for interaction {request.previous_interaction_id}, "
                "as the tool's own quote does."
            )
        else:
            caveats.append(
                "Vertex AI only; source video must be at most "
                f"{OMNI_UPSCALE_MAX_SOURCE_SECONDS:g}s. A quote uses the "
                f"{OMNI_UPSCALE_MAX_SOURCE_SECONDS:g}s upper bound until the tool "
                "measures the source."
            )
        if request.previous_interaction_id:
            params["previous_interaction_id"] = request.previous_interaction_id
        elif request.source_video_uri:
            params["input_video_uri"] = request.source_video_uri
        else:
            caveats.append(
                "Supply input_video_uri or previous_interaction_id for the "
                "source video."
            )
        named = _UPSCALE_SOURCE_PATTERN.search(request.intent)
        recorded_resolution = request.previous_interaction_resolution
        if recorded_resolution is not None:
            # The tool reads the same record, and refuses a source_resolution
            # that disagrees with it, so none is sent; quoted for "a 360p
            # source" instead, a recorded 720p clip's upscale to 720p was a
            # route the tool refused.
            source_resolution = recorded_resolution
            if named and _normalize_upscale_resolution(named.group(1)) != (
                recorded_resolution
            ):
                caveats.append(
                    f"The intent names a {named.group(1)} source, but this "
                    "server recorded interaction "
                    f"{request.previous_interaction_id} at {recorded_resolution}: "
                    f"the upscale starts from {recorded_resolution}."
                )
        elif named:
            source_resolution = named.group(1).lower()
            params["source_resolution"] = source_resolution
        else:
            caveats.append(
                "Supply source_resolution (360p or 720p) if the source has no "
                "recorded metadata."
            )

    # Past the 4 MB inline response limit (above 720p, or past 10s on 1.2)
    # upscale_video_omni settles delivery with the shared policy before it
    # renders, so the plan asks the same policy, with the length the tool
    # will measure or the 15s bound it takes for a source it cannot. The
    # router is never given a bucket, so the key is named, not invented.
    spec = omni_spec(OMNI_1_2_MODEL)
    delivery, _ = _upscale_delivery(target, delivered)
    if delivery is not None:
        if delivered is None and target not in _UPSCALE_ONLY_RESOLUTIONS:
            # Large only if the source is: a measured one of 10s or less is
            # not, and the tool's real run measures it when it can.
            delivery = (
                f"If the source runs longer than {spec.max_inline_output_seconds:g}s, "
                "or upscale_video_omni cannot download it to measure it: "
                f"{delivery[0].lower()}{delivery[1:]}"
            )
        caveats.append(delivery)

    cost = None
    try:
        from .pricing import estimate_omni_upscale_cost
    except ImportError:
        # Pricing is optional here, as everywhere in the planner; the
        # tool still validates the resolution pair before it renders.
        pass
    else:
        priced_seconds = (
            OMNI_UPSCALE_MAX_SOURCE_SECONDS if delivered is None else delivered
        )
        if source_resolution is not None:
            try:
                cost = estimate_omni_upscale_cost(
                    priced_seconds, source_resolution, target
                )
            except ValueError as exc:
                recorded_at = (
                    f"Interaction {request.previous_interaction_id} rendered "
                    f"at {source_resolution}, as this server recorded it. "
                    if source is None
                    and request.previous_interaction_resolution is not None
                    else ""
                )
                return _upscale_conflict(
                    "upscale_resolution",
                    f"{recorded_at}{exc}",
                    "Upscale to a resolution above the source's: 360p goes to "
                    "720p or 1080p, 720p to 1080p or 4K. For 360p to 4K, "
                    "upscale to 720p first, then upscale that output to 4K.",
                )
        else:
            # The source's resolution is not known here, but the target
            # narrows it to the transitions that reach it. Quoting the
            # dearest keeps the figure an upper bound; a missing quote read
            # as "pricing module not installed", which it was not.
            quotes = []
            for candidate in ("360p", "720p"):
                try:
                    quotes.append(
                        (
                            estimate_omni_upscale_cost(
                                priced_seconds, candidate, target
                            ),
                            candidate,
                        )
                    )
                except ValueError:
                    continue
            if quotes:
                cost, dearest = max(quotes, key=lambda quote: quote[0].usd)
                caveats.append(
                    f"Quoted for a {dearest} source, the only one an upscale "
                    f"to {target} takes."
                    if len(quotes) == 1
                    else f"Quoted for a {dearest} source, the dearer of the "
                    f"transitions to {target}, because the source's "
                    "resolution is not known here; a 720p source costs less."
                )
    return (
        (
            RoutedCall(
                tool="upscale_video_omni",
                model=OMNI_1_2_MODEL,
                params=params,
                score=1.0,
                rationale=(
                    "Upscale the approved video while preserving its timing and audio."
                ),
                caveats=tuple(caveats),
                cost=cost,
            ),
        ),
        (),
        (),
        (),
    )


# The tools whose output is one video an upscale can take as its source. A
# generate_clip manifest is several, and an extension chain outgrows the 15s
# source ceiling.
_UPSCALABLE_RENDER_TOOLS: frozenset[str] = frozenset(
    {"generate_video", "generate_video_omni"}
)


# The one source resolution an upscale to each of these targets takes.
_UPSCALE_SOURCE_RESOLUTION: dict[str, str] = {"720p": "360p", "4K": "720p"}

# The Omni route's unprompted tip, kept as one string so a render that must
# not be 360p (an upscale to 4K's source) can leave it out.
_OMNI_360P_TIP = (
    "resolution='360p' renders a draft at a third of the 720p price — the "
    "cheapest preview available here."
)


@dataclass(frozen=True)
class _RenderThenUpscale:
    """A generate-then-upscale plan: the render ranked, the upscale chained."""

    request: ResolvedRequest
    routes: tuple[RoutedCall, ...]
    rejected: tuple[RejectedRoute, ...]
    conflicts: tuple[RoutingConflict, ...]
    workflow: tuple[WorkflowStep, ...]
    notes: tuple[str, ...]
    # The upscale the brief asks for could not be planned on the render.
    workflow_blocked: bool = False


def _plan_render_then_upscale(
    request: ResolvedRequest,
    constraints: RoutingConstraints | None,
    render_intent: str,
    upscale_text: str,
) -> _RenderThenUpscale:
    """Plan "generate X, then upscale it": the render first, the upscale after.

    The routes are the render's, ranked as any render is; the upscale is a
    workflow step on the render's output, because it is a follow-on to the
    render rather than an alternative to it. Routing the whole intent to the
    upscale planner returned one upscale route with no source and no render.

    When the render cannot feed an upscale (upscaling is Vertex-only, takes
    one video of at most 15s, and this render is not that), the target
    resolution is planned as the render's own instead, and a note says why.
    """
    named = _UPSCALE_DESTINATION_PATTERN.search(upscale_text)
    if named:
        target = _normalize_upscale_resolution(named.group(1))
    elif request.resolution in _UPSCALE_ONLY_RESOLUTIONS:
        target = str(request.resolution)
    else:
        target = "1080p"

    # The output-resolution ask belongs to the upscale. Left on the render,
    # "then upscale it to 4K" rendered 4K and left nothing to upscale. A
    # 360p/720p resolution the caller set is the render's own and stays;
    # otherwise the render takes the default, except that the only source an
    # upscale to 720p accepts is 360p.
    given = constraints or RoutingConstraints()
    render_resolution = (
        None if given.resolution in _UPSCALE_ONLY_RESOLUTIONS else given.resolution
    )
    if render_resolution is None:
        # The cheapest source the target takes, where it is the only one or
        # where budget is cheap: an upscale to 720p takes only 360p, one to
        # 4K only 720p. Left unset, budget="cheap" planned a 4K brief's
        # render at 360p, which no upscale takes to 4K, and the plan came
        # back as a conflict. A 1080p target takes either, so it is left to
        # the render: 360p on a cheap budget, else the 720p default, at which
        # the Veo routes stay in the running.
        render_resolution = _UPSCALE_SOURCE_RESOLUTION.get(target)
    # media_kind stays what the whole intent resolved to: the render clause on
    # its own may carry no video keyword at all. needs_4k is False rather than
    # unset because the render clause is re-read: "make a 4K video of a dog,
    # then upscale it to 4K" inferred 4K from the render's own words, planned
    # a 4K render, and then reported its 4K-to-4K upscale as a conflict. A 4K
    # ask in either clause is the deliverable's, which the upscale reaches.
    render_constraints = replace(
        given, media_kind="video", needs_4k=False, resolution=render_resolution
    )
    render_request = resolve_request(
        render_intent, infer_signals(render_intent), render_constraints
    )
    routes, rejected, conflicts, notes = _plan_video(render_request)
    if target == "4K" and given.resolution is None:
        # No upscale takes a 360p render to 4K, so the route's 360p tip would
        # point at a source this plan cannot use.
        routes = tuple(
            replace(
                route,
                caveats=tuple(c for c in route.caveats if c != _OMNI_360P_TIP),
            )
            for route in routes
        )
        notes = (
            *notes,
            "The render is planned at 720p, the only resolution an upscale to "
            "4K takes; a 360p render costs less, but cannot be upscaled to 4K.",
        )
    workflow = _build_workflow(render_request, routes)

    renders = [route for route in routes if route.tool != "generate_storyboard"]
    if not renders:
        # Nothing renders, so there is nothing to upscale; the render's own
        # rejections and conflicts already say why.
        return _RenderThenUpscale(
            render_request, routes, rejected, conflicts, workflow, notes
        )
    # The upscale's source is what the plan finally delivers: the last
    # workflow step when there is a sequence, else the top render.
    if workflow:
        last = workflow[-1]
        delivery_tool, delivery_params, step = last.tool, last.params, last.order
    else:
        delivery_tool, delivery_params, step = renders[0].tool, renders[0].params, 1
    seconds = float(
        delivery_params.get("duration_seconds", render_request.clip_duration_seconds)
    )
    fallback = None
    # upscale_video_omni runs on Vertex AI whatever OMNI_BACKEND says, so a
    # render the pin sends to the Gemini Developer API is still upscaled,
    # from the file it saved (see source_params below).
    if request.backend == "gemini_api":
        fallback = "upscale_video_omni runs on Vertex AI only"
    elif delivery_tool not in _UPSCALABLE_RENDER_TOOLS:
        fallback = (
            f"upscale_video_omni takes one video, and this plan delivers "
            f"through {delivery_tool}"
        )
    elif seconds > OMNI_UPSCALE_MAX_SOURCE_SECONDS:
        fallback = (
            f"upscale_video_omni takes at most "
            f"{OMNI_UPSCALE_MAX_SOURCE_SECONDS:g}s, and this render is "
            f"{seconds:g}s"
        )

    if fallback is not None:
        direct = replace(
            render_request,
            resolution=target,
            needs_4k=render_request.needs_4k or target == "4K",
        )
        routes, rejected, conflicts, notes = _plan_video(direct)
        return _RenderThenUpscale(
            direct,
            routes,
            rejected,
            conflicts,
            _build_workflow(direct, routes),
            (
                *notes,
                f"No upscale step is planned: {fallback}. The {target} ask is "
                f"planned on the render itself, as if the brief had asked for "
                f"a {target} video; each route's caveats say whether it can "
                "reach it.",
            ),
        )

    model = delivery_params.get("omni_model") or delivery_params.get("model")
    # What upscale_video_omni will measure: an Omni render lands about one
    # frame over its nominal length, so a 10s render is a 10.04s source, past
    # the 10s inline line the tool judges on. Judged on the nominal 10s, the
    # plan said nothing about delivery for a step the tool warns about.
    try:
        from .pricing import quote_duration_for
    except ImportError:  # pragma: no cover - pricing is a sibling module
        delivered = seconds
    else:
        delivered = quote_duration_for(str(model), seconds) if model else seconds
    if model == OMNI_1_2_MODEL and _omni_serves_on_vertex(render_request):
        # A 1.2 interaction on Vertex upscales in place: its resolution and
        # length are on record, and nothing is re-uploaded.
        source_params: dict[str, Any] = {
            "previous_interaction_id": f"<interaction_id from step {step}>"
        }
    else:
        # upscale_video_omni takes a previous interaction only from 1.2 on
        # Vertex; any other render is passed as the file it produced.
        source_params = {"input_video_uri": f"<video_url from step {step}>"}
    source_resolution = str(delivery_params.get("resolution") or OMNI_RESOLUTION)
    upscale_routes, _, upscale_conflicts, _ = _plan_upscale(
        replace(request, intent=upscale_text, resolution=target),
        _PlannedSource(source_params, source_resolution, seconds, delivered),
    )
    if not upscale_routes:
        # The render routes still stand, and are kept for what they say, but
        # the brief asked for the upscale too: is_satisfiable read true
        # beside an upscale conflict and an empty workflow.
        return _RenderThenUpscale(
            render_request,
            routes,
            rejected,
            (*conflicts, *upscale_conflicts),
            workflow,
            notes,
            workflow_blocked=True,
        )
    if not workflow:
        workflow = (
            WorkflowStep(
                order=1,
                tool=renders[0].tool,
                params=renders[0].params,
                rationale=(
                    f"Render the {source_resolution} source on "
                    f"{renders[0].model}; step 2 upscales its output."
                ),
            ),
        )

    upscale = upscale_routes[0]
    upscale_params = dict(upscale.params)
    if _upscale_delivery(target, delivered)[1]:
        # Only where the server refuses a large inline upscale. By default the
        # tool delivers inline without a bucket, so a placeholder would read
        # as a value the step cannot run without; here it is one. Workflow
        # params carry placeholders for what the caller must fill in, as the
        # source above does for an earlier step's output.
        upscale_params["output_gcs_uri"] = "<gs:// destination for the upscale>"
    # A step has no caveats of its own, so what the route would have said
    # beyond "Vertex AI only" (the delivery advice) rides in the rationale.
    said = tuple(caveat for caveat in upscale.caveats if caveat != "Vertex AI only.")
    lead = (
        f"Once step {step}'s {source_resolution} render is approved, upscale "
        f"it to {target} on {OMNI_1_2_MODEL}"
    )
    if upscale.cost is not None:
        lead += f" (est. ${upscale.cost.usd:.2f})"
    return _RenderThenUpscale(
        render_request,
        routes,
        rejected,
        conflicts,
        (
            *workflow,
            WorkflowStep(
                order=step + 1,
                tool="upscale_video_omni",
                params=upscale_params,
                rationale=" ".join(
                    (
                        f"{lead}: frames, timing and audio are kept, and the "
                        "upscale is paid for only on a take worth keeping.",
                        *said,
                    )
                ),
            ),
        ),
        (
            *notes,
            f"The upscale to {target} is workflow step {step + 1}, after the "
            "render; the routes rank the render alone.",
        ),
    )


# ============================================================================
# Public entry point
# ============================================================================


def _pricing_installed() -> bool:
    """Whether the price book can be imported at all.

    So a route without a quote is explained truthfully: "pricing module not
    installed" was said whenever any route lacked one, including a price
    book that was installed and could not price that route.
    """
    try:
        # The ``from .pricing import`` form, as the planner's own imports use:
        # it goes through sys.modules, where a missing module shows.
        from .pricing import estimate_video_cost  # noqa: F401
    except ImportError:
        return False
    return True


def plan_generation(
    intent: str, constraints: RoutingConstraints | None = None
) -> RoutingPlan:
    """Plan how to generate what ``intent`` describes.

    Pure and deterministic: no network, no filesystem. The one clock read is
    Google's published model calendar (the preview switch-off), checked per
    call so a long-running server changes its plans on the day; on any one
    date the same arguments produce an identical plan.

    Args:
        intent: Natural-language description of the desired output, e.g.
            "a 9:16 product reel with three shots and music".
        constraints: Structured facts the caller already knows. Any field that
            is set overrides whatever the intent text implied.

    Returns:
        A RoutingPlan whose ``routes`` are ranked best-first, whose
        ``rejected`` entries explain every excluded model, and whose
        ``conflicts`` name any contradiction that would fail at call time.

    Raises:
        ValueError: If ``intent`` is empty or whitespace.
    """
    if not intent or not intent.strip():
        raise ValueError("intent must be a non-empty description of what to generate.")
    if len(intent) > MAX_INTENT_CHARS:
        raise ValueError(
            f"intent is {len(intent)} characters; the limit is {MAX_INTENT_CHARS}. "
            "Describe what to generate, not the whole brief."
        )

    signals = infer_signals(intent)
    request = resolve_request(intent, signals, constraints)

    # Settled before the request is split: a declined upscale ("but don't
    # upscale it") is no upscale at all, and its clause is no part of the
    # render's prompt.
    render_intent, declined = _without_declined_upscales(intent)
    upscale, adjective_note = _upscale_request(render_intent)
    if declined and upscale is not None:
        # Asked for, then retracted ("then upscale it. actually don't"): the
        # retracted clause leaves the prompt too.
        retracted = _split_render_and_upscale(render_intent, upscale)
        if retracted is not None:
            render_intent = retracted[0]
    if render_intent != intent:
        signals = infer_signals(render_intent)
        request = resolve_request(render_intent, signals, constraints)
    if upscale is not None and (request.media_kind != "video" or declined):
        upscale = None
    split = _split_render_and_upscale(render_intent, upscale) if upscale else None
    workflow: tuple[WorkflowStep, ...] | None = None
    workflow_blocked = False
    if split is not None:
        # "Generate X, then upscale it": the upscale needs the render first.
        chained = _plan_render_then_upscale(request, constraints, *split)
        request = chained.request
        routes, rejected, conflicts = (
            chained.routes,
            chained.rejected,
            chained.conflicts,
        )
        notes, workflow = chained.notes, chained.workflow
        workflow_blocked = chained.workflow_blocked
    elif upscale is not None:
        routes, rejected, conflicts, notes = _plan_upscale(request)
    elif request.media_kind == "image":
        routes, rejected, conflicts, notes = _plan_image(request)
    else:
        routes, rejected, conflicts, notes = _plan_video(request)

    extra_notes = list(notes)
    if upscale is None and adjective_note is not None and request.media_kind == "video":
        extra_notes.append(adjective_note)
    if declined and request.media_kind == "video":
        extra_notes.append(
            "No upscale step is planned: the brief's last word on upscaling "
            "declines it, and the render's prompt leaves that clause out."
        )
    pinned = constraints.pinned_model if constraints is not None else None
    redirect = omni_model_reroute_warning(pinned)
    if redirect:
        # resolve_request already planned the pin as the model that serves
        # it, so every route and quote above is that model's. First, because
        # it explains why the routes name a model the caller did not pin.
        extra_notes.insert(
            0,
            f"pinned_model={pinned}: {redirect} This plan is for "
            f"{request.pinned_model}.",
        )
    if signals.media_kind is None and (
        constraints is None or constraints.media_kind is None
    ):
        # Names the kind that was ACTUALLY planned. The note used to say "an
        # image" unconditionally, because it checked only that no keyword
        # matched -- ignoring resolve_request's structurally_video fallback.
        # plan_generation("a golden retriever running on a beach",
        # duration_seconds=6.0) planned a VIDEO and then told the caller it had
        # planned an image, and the MCP tool forwards `notes` verbatim, so a
        # calling agent was handed the opposite of what happened.
        planned_as = "a video" if request.media_kind == "video" else "an image"
        extra_notes.append(
            f"No image or video keyword matched; planned as {planned_as}. Set "
            "media_kind explicitly if that is wrong."
        )
    if any(route.cost is None for route in routes):
        extra_notes.append(
            "Cost estimates are unavailable (pricing module not installed); "
            "routes are ranked on capability and relative cost tier only."
            if not _pricing_installed()
            else "Some routes carry no cost estimate: the price book could not "
            "price them. Those are ranked on capability and relative cost "
            "tier only."
        )

    if workflow is None:
        workflow = _build_workflow(request, routes)

    return RoutingPlan(
        intent=intent,
        media_kind=request.media_kind,
        signals=signals,
        request=request,
        routes=routes,
        rejected=rejected,
        conflicts=conflicts,
        workflow=workflow,
        notes=tuple(extra_notes),
        workflow_blocked=workflow_blocked,
    )
