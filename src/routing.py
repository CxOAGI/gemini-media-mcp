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
support, the Veo Lite restriction set, omni's duration bounds) are IMPORTED
rather than restated: a second copy would drift the first time Google changes
a model. Facts with no existing home (relative cost/fidelity of each model,
tool-level routing rules) are defined here.
"""

import math
import re
from dataclasses import dataclass
from typing import Any, Literal, Protocol, get_args

from .image import (
    _IMAGE_SIZE_SUPPORT,  # pyright: ignore[reportPrivateUsage]
    _supports_image_size,  # pyright: ignore[reportPrivateUsage]
    ImageModel,
    ImageSize,
    resolve_image_model,
)
from .omni import (
    _MAX_DURATION as OMNI_MAX_DURATION,  # pyright: ignore[reportPrivateUsage]
)
from .omni import (
    _MIN_DURATION as OMNI_MIN_DURATION,  # pyright: ignore[reportPrivateUsage]
)
from .omni import (
    _SUPPORTED_ASPECT_RATIOS as OMNI_ASPECT_RATIOS,  # pyright: ignore[reportPrivateUsage]
)
from .omni import (
    OMNI_1_1_MODEL,
    OMNI_DEFAULT_RESOLUTION,
    OMNI_MODEL,
    OMNI_RESOLUTIONS,
    is_omni_model,
    omni_extension_output_lengths,
    omni_spec,
)
from .video import (
    _CANONICAL_VIDEO_MODEL_IDS,  # pyright: ignore[reportPrivateUsage]
    _VEO_LITE_MODELS,  # pyright: ignore[reportPrivateUsage]
    VideoModel,
)

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

# Veo output resolutions (src/video.py validates against exactly this set).
VEO_RESOLUTIONS: tuple[str, ...] = ("720p", "1080p", "4K")

# What an omni render comes back as when nothing asks otherwise, and the only
# thing gemini-omni-flash-preview can produce at all. gemini-omni-1.1-flash
# takes a real resolution parameter (OMNI_RESOLUTIONS) on top of this default.
OMNI_RESOLUTION = OMNI_DEFAULT_RESOLUTION

# Omni 1.1's draft tier: a third of the 720p price, and the cheapest render
# this server can issue. Named here because the workflow builder recommends it
# by name as the preview pass.
OMNI_DRAFT_RESOLUTION = "360p"

# Every resolution some video model here can render, which is what a request
# is validated against: Veo's three plus omni 1.1's 360p draft tier.
VIDEO_RESOLUTIONS: tuple[str, ...] = tuple(
    dict.fromkeys((*VEO_RESOLUTIONS, *OMNI_RESOLUTIONS))
)

# Aspect ratios the video path accepts. src/video.py hard-errors on anything
# else, and omni documents the same pair.
VIDEO_ASPECT_RATIOS: tuple[str, ...] = OMNI_ASPECT_RATIOS

# Gemini 3.x image models accept up to 14 reference images (see image.py's
# max_refs) — beyond that the extras are dropped by the SDK call.
MAX_IMAGE_REFERENCE_IMAGES = 14

# Veo accepts at most 3 reference images (asset type), per src/video.py.
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
            quality-led (budget="best", 4K, print). This is where a
            size/resolution demand is scored — every model still in the race
            already passed the hard size rule, so re-scoring it as a
            capability would only tilt a satisfied demand toward the priciest
            option.
        speed_fit: Turnaround, consulted when the request is draft-led.
        default_affinity: A deliberate thumb on the scale for each tool's
            documented default, so an otherwise-even race resolves to the
            option the rest of the server already assumes.
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
    # Omni 1.1's draft tier. Veo publishes no 360p rate and src/video.py
    # rejects the value, so a 360p ask has to exclude every Veo model rather
    # than reach one and be dropped (or, worse, be quoted at the 720p rate).
    supports_360p: bool
    supports_seed: bool
    supports_negative_prompt: bool
    supports_audio: bool
    supports_conversational_edit: bool
    gemini_api_only: bool
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
        # Grounded in the published rates, not vibes: omni is $0.10136/s
        # against Fast's $0.10/s, so it is marginally the DEARER of the two.
        # It sat at 0.0 — the cheapest slot in the family — which made a
        # budget=cheap request rank a $0.4054 route above a $0.40 one.
        cost_index=0.27,
        fidelity_index=0.30,
        speed_index=1.0,
        text_rendering_index=0.0,
        summary="fastest 720p/24fps drafts with conversational editing",
    ),
    OMNI_1_1_MODEL: ModelProfile(
        model=OMNI_1_1_MODEL,
        media_kind="video",
        # Identical published token rate to the preview model at 720p
        # ($0.10136/s), so it sits on exactly the same rung — the index tracks
        # real $/s at 720p and nothing else, which is what keeps a
        # budget=cheap request from ranking a dearer route as the cheap one.
        # Its 360p draft tier IS a third of that ($0.0338/s), the cheapest
        # video render in the catalog, but that is a resolution the caller
        # chooses per call, not a property of the model's standing.
        cost_index=0.27,
        # Above the preview model on every axis that is not price: real 1080p
        # and 4K output, keyframe interpolation, video references and native
        # audio. Below full-fat Veo, which is still the finishing renderer.
        fidelity_index=0.65,
        speed_index=0.95,
        text_rendering_index=0.0,
        summary=(
            "fast 360p-4K renders with conversational editing, video "
            "extension, keyframe interpolation and video references"
        ),
    ),
    "veo-3.1-lite-generate-preview": ModelProfile(
        model="veo-3.1-lite-generate-preview",
        media_kind="video",
        # $0.05/s — genuinely the cheapest video route.
        cost_index=0.0,
        fidelity_index=0.60,
        speed_index=0.80,
        text_rendering_index=0.0,
        summary="cheapest Veo tier, text-to-video and image-to-video only",
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
        gemini_api_only=False,
        min_duration_seconds=float(OMNI_MIN_DURATION),
        max_duration_seconds=float(OMNI_MAX_DURATION),
    ),
    # Omni 1.1: everything the preview model does, plus the four capabilities
    # that used to force a Veo route — resolution control up to 4K, video
    # extension, first/last-frame interpolation and native audio — while
    # keeping conversational editing, which no Veo tier has. Still no seed and
    # no negative_prompt: the reference lists both as unsupported, along with
    # system instructions, temperature and top_p.
    OMNI_1_1_MODEL: VideoCapabilities(
        supports_first_last_frame=True,
        supports_extension=True,
        supports_reference_images=True,
        supports_4k=True,
        supports_1080p=True,
        supports_360p=True,
        supports_seed=False,
        supports_negative_prompt=False,
        # 1.1 generates an audio track natively ("The model generates a video
        # with audio based on your text description") and the prompt guide has
        # a section on directing it. What it has no switch for is turning it
        # OFF — a caveat, not a reason to exclude it when audio is wanted.
        supports_audio=True,
        supports_conversational_edit=True,
        gemini_api_only=False,
        min_duration_seconds=float(OMNI_MIN_DURATION),
        max_duration_seconds=float(OMNI_MAX_DURATION),
    ),
    # Veo 3.1 Lite: Gemini-API-only, no 4K, no extension, no first/last frame,
    # no reference images (src/video.py raises on the last three).
    "veo-3.1-lite-generate-preview": VideoCapabilities(
        supports_first_last_frame=False,
        supports_extension=False,
        supports_reference_images=False,
        supports_4k=False,
        supports_1080p=True,
        supports_360p=False,
        supports_seed=True,
        supports_negative_prompt=True,
        # Lite generates audio like the other Veo tiers; what it cannot do is
        # let you switch it OFF (include_audio maps to the Vertex-only
        # generate_audio flag and Lite never runs on Vertex) — a caveat, not a
        # reason to exclude it when audio is wanted.
        supports_audio=True,
        supports_conversational_edit=False,
        gemini_api_only=True,
        min_duration_seconds=float(min(VEO_DURATIONS_SECONDS)),
        max_duration_seconds=float(VEO_MAX_CLIP_SECONDS),
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
    # Veo 3.1 generates audio natively. On Vertex it is switchable via the
    # generate_audio flag; on the Gemini API it is always on (src/video.py
    # warns about exactly this).
    supports_audio=True,
    supports_conversational_edit=False,
    gemini_api_only=False,
    min_duration_seconds=float(min(VEO_DURATIONS_SECONDS)),
    max_duration_seconds=float(VEO_MAX_CLIP_SECONDS),
)
for _model in LIVE_VIDEO_MODELS:
    if _model not in _VIDEO_CAPABILITIES:
        _VIDEO_CAPABILITIES[_model] = _FULL_VEO_CAPABILITIES


def image_profile(model: str) -> ModelProfile | None:
    """Return the scoring profile for an image model, or None if unknown."""
    return _IMAGE_PROFILES.get(model)


def video_profile(model: str) -> ModelProfile | None:
    """Return the scoring profile for a video model, or None if unknown."""
    return _VIDEO_PROFILES.get(model)


def video_capabilities(model: str) -> VideoCapabilities | None:
    """Return the hard capability set for a video model, or None if unknown."""
    return _VIDEO_CAPABILITIES.get(model)


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
        "poster",
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

_VERTICAL_TERMS: frozenset[str] = frozenset(
    {
        "9:16",
        "portrait",
        "reel",
        "reels",
        "short",
        "shorts",
        "stories",
        "story",
        "tiktok",
        "vertical",
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

# "by another 30 seconds", "add 20s", "extend by 2 minutes" — an amount to
# ADD, not a total. Without this, the plain duration pattern takes the first
# number in the sentence, so "extend my 8 second clip by another 30 seconds"
# read the SOURCE length and planned one 7s extension instead of five.
# The leading lookbehind is load-bearing: without it "a lullaby 30 seconds
# long" matched the "by 30 seconds" buried inside "lullaby" and planned a 30s
# extension of a clip nobody asked to extend.
_ADDED_DURATION_PATTERN = re.compile(
    r"(?<!\w)(?:by|add|plus|another)\s+(?:another\s+)?(\d+(?:\.\d+)?)\s*"
    r"(?:-|\s)?\s*(s\b|sec\b|secs\b|second\b|seconds\b"
    r"|m\b|min\b|mins\b|minute\b|minutes\b)"
)

# The single-letter units are the ambiguous ones: "8s" is a runtime but "the
# 1970s", "an 80s diner", "our 100s of customers", "the iPhone 15s" and "a 3 m
# tall robot" are not, and every one of them was being planned as a duration
# (1970s quoted a 247-beat clip at $197.60). A bare unit therefore only counts
# when the surrounding words make it a runtime; the spelled-out units
# (sec/second/min/minute) are unambiguous and need no such proof.
_BARE_DURATION_UNITS: frozenset[str] = frozenset({"s", "m"})

# Nouns that can only be describing the thing being timed, so a bare unit in
# front of one is a runtime: "8s clip", "30s of video".
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
                _word_after(text, match.end()) in _DURATION_CONTEXT_WORDS
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
_BEAT_PATTERN = re.compile(
    r"(\d+)\s*(?:beats?|shots?|scenes?|cuts?|segments?|panels?|keyframes?"
    r"|frames?(?!\s+per\s+second))\b"
)

# "up to 6 reference images", "3 reference photos".
_REFERENCE_COUNT_PATTERN = re.compile(
    r"(\d+)\s*(?:reference|ref)\s*(?:images?|photos?)"
)

_TERM_PATTERNS: dict[str, re.Pattern[str]] = {}


def _term_pattern(term: str) -> re.Pattern[str]:
    """Compile (and memoise) a boundary-anchored pattern for one keyword.

    Lookarounds rather than ``\\b`` so terms that start or end with
    punctuation (``gs://``, ``b-roll``, ``9:16``) still anchor correctly.
    """
    pattern = _TERM_PATTERNS.get(term)
    if pattern is None:
        pattern = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)")
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

# Negators that can only be talking about the soundtrack. "a silent video of
# rain" is a video request with the sound off, but a negator scoped to nothing
# read "silent" as cancelling "video" and planned a still image — 5 of 15
# natural "video without X" phrasings lost their modality this way.
_AUDIO_NEGATORS: tuple[str, ...] = _NEGATORS + ("silent", "mute", "muted")

# How far back to look for a negator. Short on purpose: "no dialogue, but
# ambient music" should keep the music hit, so only the words immediately
# before a term can negate it.
_NEGATION_WINDOW = 3

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
        cleaned = word.strip("-\"'")
        if cleaned in _CLAUSE_BREAK_WORDS:
            return False
        if cleaned in negators:
            return True
    return False


def _is_negated(text: str, term: str) -> bool:
    """Whether every occurrence of ``term`` in ``text`` sits under a negator.

    Keyword matching alone read "no audio" as an audio request, which flipped
    both the emitted include_audio and the model ranking. Requiring EVERY
    occurrence to be negated is the conservative reading: a sentence that
    mentions the term both ways keeps the positive signal.
    """
    negators = _AUDIO_NEGATORS if term in _AUDIO_TERMS else _NEGATORS
    pattern = _term_pattern(term)
    found = False
    for match in pattern.finditer(text):
        found = True
        if not _negator_precedes(text[: match.start()], negators):
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
    if seconds_value is not None:
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
    elif transition_hits or bridge_hits or extension_hits or duration is not None:
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
            are backend-specific (GCS output, Veo Lite availability).
        gemini_api_key_available: Whether the server holds a GEMINI_API_KEY.
            None means unknown, so a Gemini-API-only model stays on the table;
            False is the server stating it has no key, which makes those
            models unrunnable however the request is phrased. Live testing
            found the planner recommending Veo 3.1 Lite on a Vertex
            deployment with no key — the routing rule was already here, only
            the credential fact was missing.
        num_beats: How many distinct shots the output has.
        is_draft: This render is throwaway.
        is_iterating: The caller is refining an existing result.
        needs_seed: Reproducibility via a fixed seed is required.
        needs_negative_prompt: A negative prompt is required.
        resolution: Exact video resolution ("360p"/"720p"/"1080p"/"4K");
            360p exists on gemini-omni-1.1-flash only.
        num_images: How many images to generate (cost estimation only).
        pinned_model: A model the caller insists on. Rules still apply, but a
            violation is reported as a conflict rather than a quiet swap.
        previous_interaction_id: An omni interaction to keep editing.
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
    num_beats: int | None = None
    is_draft: bool | None = None
    is_iterating: bool | None = None
    needs_seed: bool | None = None
    needs_negative_prompt: bool | None = None
    resolution: str | None = None
    num_images: int | None = None
    pinned_model: str | None = None
    previous_interaction_id: str | None = None

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
    # None when the caller did not say, i.e. a Gemini-API-only model might
    # work; False is a positive statement that it cannot.
    gemini_api_key_available: bool | None
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
    # that value snapped into Veo's 4/6/8s ladder, capped at 8s — anything
    # longer has to come from extensions or several beats.
    total_duration = _first_not_none(given.duration_seconds, signals.duration_seconds)
    if total_duration is None:
        clip_duration = DEFAULT_VIDEO_DURATION_SECONDS
    elif total_duration > VEO_MAX_CLIP_SECONDS:
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
    # by a single-shot request longer than Veo's 8s ceiling.
    implied_extension = (
        total_duration is not None
        and total_duration > VEO_MAX_CLIP_SECONDS
        and num_beats <= 1
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

    return ResolvedRequest(
        intent=intent,
        media_kind=media_kind,
        budget=budget,
        backend=given.backend,
        gemini_api_key_available=given.gemini_api_key_available,
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

    @property
    def recommended(self) -> RoutedCall | None:
        """The top-ranked route, or None when nothing can satisfy the request."""
        return self.routes[0] if self.routes else None

    @property
    def is_satisfiable(self) -> bool:
        """Whether at least one route survived every capability rule."""
        return bool(self.routes)


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

    default_term = 1.0 if profile.model == default_model else 0.0

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

    Ties break on fidelity (the more capable model is the safer default when
    the score cannot separate them) and then on the model ID, so the ordering
    is a total order that never depends on insertion or set iteration order.
    """

    def sort_key(route: RoutedCall) -> tuple[float, float, str, str]:
        profile = profiles.get(route.model)
        fidelity = profile.fidelity_index if profile else 0.0
        return (-route.score, -fidelity, route.model, route.tool)

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
            "{model} excluded: 360p is gemini-omni-1.1-flash's draft tier; "
            "this model publishes neither the output nor a rate for it"
        ),
        resolution=(
            "Render the 360p draft on gemini-omni-1.1-flash via "
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
            "Conversational edits run on gemini-omni-flash-preview via "
            "edit_video with the prior interaction_id."
        ),
    ),
)


def _capability_rejection(
    model: str, needs: VideoNeeds
) -> tuple[_CapabilityRule, str] | None:
    """Return the first hard rule ``model`` violates, or None.

    Only the first is reported: once a model is out, listing every additional
    reason is noise rather than expertise.
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

    The omni models are the only ones with a real range here (3-10s); Veo
    snaps instead of failing, so a Veo mismatch is a caveat rather than a
    rejection.
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


def _backend_rejection(model: str, request: ResolvedRequest) -> tuple[str, str] | None:
    """Return (reason, resolution) when the backend cannot serve ``model``.

    Three documented facts drive this: Veo 3.1 Lite is published on the Gemini
    Developer API only, reaching it needs a GEMINI_API_KEY whatever the
    backend, and GCS output is a Vertex-only config field.

    Args:
        model: Candidate model ID.
        request: The resolved request, including the backend and (when the
            server told us) whether a Gemini API key exists.

    Returns:
        The rejection reason paired with the concrete fix, or None when the
        backend can serve the model.
    """
    capabilities = _VIDEO_CAPABILITIES.get(model)
    if capabilities is None:
        return None
    if capabilities.gemini_api_only and request.gemini_api_key_available is False:
        # Checked before the backend rules because it is the stronger fact: on
        # a Vertex deployment the server can still reach a Gemini-API-only
        # model through a Gemini API client, but not without the key. Live
        # testing had the planner recommending Lite into exactly this failure,
        # so the reason is worded as the error the real call raises.
        return (
            f"{model} excluded: it is served by the Gemini Developer API only "
            "and this server has no GEMINI_API_KEY, so the call would fail with "
            "that error instead of rendering.",
            "Set GEMINI_API_KEY to reach this model, or pick a Veo tier the "
            "configured backend publishes.",
        )
    if (
        capabilities.gemini_api_only
        and request.backend == "vertex"
        and not request.gemini_api_key_available
    ):
        # Only when the key is absent or unknown. The server genuinely routes
        # these models through a dedicated Gemini API client on a Vertex
        # deployment (see _client_for_video_model), so once it has told us a
        # key exists, excluding the model would refuse a call that works —
        # the mirror of the bug this rejection was added to prevent.
        return (
            f"{model} excluded: served by the Gemini Developer API only — Vertex "
            "AI has not published it. The server can route it through a Gemini "
            "API client, but only when GEMINI_API_KEY is set.",
            "Switch backend, or pick a model published on the backend in use.",
        )
    if capabilities.gemini_api_only and request.wants_gcs_output:
        return (
            f"{model} excluded: it runs on the Gemini API, which does not support "
            "output_gcs_uri (GCS output is Vertex-only).",
            "Switch backend, or pick a model published on the backend in use.",
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
    if request.num_beats > 1:
        return "generate_clip"
    return "generate_video"


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


def _video_needs(request: ResolvedRequest, tool: ToolName) -> VideoNeeds:
    """Translate the resolved request plus chosen tool into hard requirements."""
    first_last = tool in ("generate_transition", "generate_bridge") or (
        request.has_first_frame and request.has_last_frame
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
            tool == "edit_video" or request.previous_interaction_id is not None
        ),
        clip_duration_seconds=request.clip_duration_seconds,
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

    duration = request.clip_duration_seconds
    if not is_omni_model(model):
        snapped = _snap_veo_duration(duration)
        if float(snapped) != duration:
            caveats.append(
                f"Veo renders {'/'.join(str(d) for d in VEO_DURATIONS_SECONDS)}s "
                f"clips; {duration:g}s snaps to {snapped}s."
            )
        duration = float(snapped)

    params: dict[str, Any] = {}
    if tool == "edit_video":
        params = {
            "previous_interaction_id": request.previous_interaction_id,
            "prompt": request.intent,
            # Emitted so the call runs on the model this route was priced
            # for. Both omni models quote identically at 720p, but the plan
            # naming one and the tool defaulting to the other is exactly the
            # drift this pass-through exists to prevent.
            "omni_model": model,
            "aspect_ratio": aspect_ratio,
            "duration_seconds": duration,
        }
        if omni_spec(model).supports_resolution and request.resolution is not None:
            params["resolution"] = request.resolution
        caveats.append(
            "omni_model must be the model that produced "
            "previous_interaction_id — the interaction's video context lives "
            "with it, so an edit cannot switch models mid-conversation."
        )
        caveats.append(
            "An edit does not send duration or aspect ratio — the API rejects "
            "them on an edit task — and the rendered length is chosen by the "
            "service, predictable from neither the request nor the source. "
            "Cost is measured from the rendered file; a quote is an upper bound."
        )
        caveats.append(
            f"The quote is Omni's {OMNI_MAX_DURATION:g}s per-render maximum, "
            "the same upper bound edit_video's own dry_run reports for a clip "
            f"this planner cannot measure — not the {duration:g}s above, which "
            "is never sent. If the interaction is a clip you already extended "
            f"past {OMNI_MAX_DURATION:g}s, the tool's dry_run reads its real "
            "length from the sidecar and quotes higher; this figure assumes a "
            "single-render source."
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
        else:
            caveats.append(
                "Add video_uri: loop_extend needs the existing Veo clip to continue."
            )
        caveats.append(
            f"Each extension adds ~{VEO_EXTENSION_SECONDS}s and output is 720p; "
            f"Veo allows at most {VEO_MAX_EXTENSIONS} extensions."
        )
        if request.backend == "vertex":
            caveats.append(
                "On Vertex AI, extension requires output_gcs_uri — the combined "
                "video exceeds the inline response limit."
            )
    elif tool == "extend_video_omni":
        spec = omni_spec(model)
        step = float(spec.extension_step_seconds)
        wanted = request.added_duration_seconds
        if wanted is None and request.total_duration_seconds is not None:
            # A total was named instead of a delta. Omni extends footage that
            # already exists, so what has to be rendered is the difference
            # above the source — measured from the longest source omni will
            # take, rather than from zero, which would bill the caller for
            # seconds they already have.
            wanted = request.total_duration_seconds - spec.max_uploaded_source_seconds
        times = max(1, math.ceil((wanted or step) / step))
        ceiling = int(spec.max_extended_seconds // spec.extension_step_seconds)
        if times > ceiling:
            caveats.append(
                f"{times} extension turns would be needed but omni caps a clip "
                f"at {spec.max_extended_seconds}s in "
                f"{spec.extension_step_seconds}s steps; planned at the "
                f"{ceiling}-turn maximum."
            )
        params = {
            "prompt": request.intent,
            "omni_model": model,
            "times": min(times, ceiling),
        }
        if spec.supports_resolution and request.resolution is not None:
            # Priced at request.resolution below, so it has to be SENT. It was
            # the one omni branch that quoted a resolution its own params
            # never asked for: a 360p plan handed over params that render 720p
            # and bill 3x the quote.
            params["resolution"] = request.resolution
        if request.previous_interaction_id is not None:
            params["previous_interaction_id"] = request.previous_interaction_id
        elif request.source_video_uri is not None:
            params["input_video_uri"] = request.source_video_uri
        else:
            caveats.append(
                "Add previous_interaction_id (a clip this server generated) or "
                "input_video_uri (a clip to upload): extend_video_omni needs "
                "existing footage to continue."
            )
        caveats.append(
            f"Each turn appends up to {spec.extension_step_seconds}s using the "
            f"last {spec.extension_step_seconds}s of the source as context, to "
            f"a cumulative {spec.max_extended_seconds}s. An uploaded source "
            f"must be {spec.max_uploaded_source_seconds:g}s or shorter; a "
            "multi-turn source has no such limit."
        )
        caveats.append(
            "Each turn returns the footage it appends, not the assembled "
            "clip, so the response is a sequence of segments to join "
            f"downstream. The planner cannot see the source's length, so it "
            f"does not know how many turns fit under the "
            f"{spec.max_extended_seconds}s total; the tool's own dry_run "
            "measures the source and says."
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
        caveats.append(
            f"Replace each of the {beat_count} beat prompts with that "
            "shot's own description — they are seeded from the intent."
        )
    elif tool == "generate_video_omni":
        params = {
            "prompt": request.intent,
            "omni_model": model,
            "aspect_ratio": aspect_ratio,
            "duration_seconds": duration,
        }
        spec = omni_spec(model)
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
            if request.resolution is not None:
                params["resolution"] = request.resolution
            caveats.append(
                f"{model} renders {'/'.join(spec.resolutions)} (1080p and 4K "
                "upscaled from the base render) and has no seed or "
                "negative_prompt; keep the returned interaction_id to edit or "
                "extend it conversationally."
            )
            if request.resolution in (None, OMNI_RESOLUTION):
                # The one saving worth naming unprompted: a 360p pass is a
                # third of the 720p price, which makes it the cheapest video
                # render this server can issue.
                caveats.append(
                    "resolution='360p' renders a draft at a third of the 720p "
                    "price (Google's launch post; the pricing page publishes "
                    "only the 720p rate) — the cheapest preview available here."
                )
        else:
            caveats.append(
                f"{model} output is {spec.rendered_resolution}/24fps with no "
                "seed or negative_prompt; keep the returned interaction_id to "
                "edit it conversationally. Pass "
                f"omni_model='{OMNI_1_1_MODEL}' for resolution control, video "
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

    if (
        not is_omni_model(model)
        and request.backend == "gemini_api"
        and not request.needs_audio
    ):
        caveats.append(
            "include_audio=False cannot be honored on the Gemini API — Veo 3.1 "
            "always generates audio there."
        )

    # Lite is published on the Gemini Developer API only, so the server routes
    # it through a Gemini API client even in Vertex mode — which also means
    # its audio is always on and its output can never go to GCS.
    if model in _VEO_LITE_MODELS:
        caveats.append(
            f"{model} is served by the Gemini Developer API only; GEMINI_API_KEY "
            "must be set, audio is always generated, and output_gcs_uri is "
            "unavailable."
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
    # as if it had; 1.1 renders what it is given.
    def _omni_resolution(candidate: str) -> str:
        if not omni_spec(candidate).supports_resolution:
            return OMNI_RESOLUTION
        return request.resolution or OMNI_RESOLUTION

    if tool == "edit_video":
        # An edit's rendered length is chosen by the service: two measurements
        # showed it rendering at Omni's maximum regardless of the source, and
        # edit_video's own dry_run quotes that maximum for exactly that reason.
        # Pricing the request's duration under-quoted a ~6s edit by 40%, which
        # is the one thing a pre-flight may not do. include_audio mirrors the
        # tool's own call so the two quotes are identical down to the detail.
        return _estimate_video_cost(
            model, float(OMNI_MAX_DURATION), _omni_resolution(model), False
        )

    if tool == "extend_video_omni":
        # Every turn is its own interaction and its own render, and each
        # renders the CONTINUATION it appends — 3-10s, set by duration — not
        # the whole growing clip. The Interactions API reference gives
        # duration's range for exactly this request, and the documented sample
        # response bills 28,832 video tokens, which is 4.98s at the published
        # rate. The source length is not knowable here, so the projection does
        # not clamp against the cumulative ceiling; the tool's own dry_run
        # measures the source and does.
        turns = int(params.get("times", 1))
        lengths = omni_extension_output_lengths(
            omni_spec(model),
            None,
            turns,
            params.get("duration_seconds"),
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
                "duration_seconds": request.clip_duration_seconds,
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
                    "Developer API, which returns media inline and rejects an "
                    "explicit GCS target."
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
        conflicts.append(
            RoutingConflict(
                code="transition_requires_two_endpoints",
                detail=(
                    f"{named_tool} needs both endpoints, but only "
                    f"{'a first' if request.has_first_frame else 'a last' if request.has_last_frame else 'neither'} "
                    "frame is available."
                ),
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
                    "Conversational editing only exists on "
                    f"{OMNI_MODEL}, which has no seed parameter."
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
        # Two stills and a prompt is exactly omni 1.1's interpolation, so it
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

    for model in candidates:
        profile = _VIDEO_PROFILES[model]

        capability_hit = _capability_rejection(model, needs)
        duration_reason = _duration_rejection(model, needs)
        backend_hit = _backend_rejection(model, request)
        reason = None
        code = None
        resolution = None
        if capability_hit is not None:
            rule, reason = capability_hit
            code, resolution = rule.code, rule.resolution
        elif duration_reason is not None:
            reason = duration_reason
            code = "duration_out_of_range"
            resolution = (
                f"Use a Veo model, which renders "
                f"{'/'.join(str(d) for d in VEO_DURATIONS_SECONDS)}s clips and can "
                "be extended."
            )
        elif backend_hit is not None:
            reason, resolution = backend_hit
            code = "backend_unsupported"

        if reason is not None:
            route_tool = _route_tool(tool, model)
            rejected.append(RejectedRoute(model=model, reason=reason, tool=route_tool))
            if request.pinned_model == model:
                conflicts.append(
                    RoutingConflict(
                        code=f"pinned_model_{code}",
                        detail=reason,
                        resolution=resolution or "Choose a different model.",
                    )
                )
            continue

        route_tool = _route_tool(tool, model)

        params, caveats = _video_params(route_tool, model, request)

        demanded: list[float] = []
        reasons: list[str] = []
        if request.needs_4k or request.resolution in ("1080p", "4K"):
            demanded.append(profile.fidelity_index)
            reasons.append(f"{request.resolution or '4K'} output")
        if request.num_reference_images > 0:
            demanded.append(profile.fidelity_index)
            reasons.append("subject consistency from reference images")
        if request.needs_audio:
            demanded.append(profile.fidelity_index)
            reasons.append("a controllable audio track")

        score = _score_route(
            profile,
            request,
            demanded_capabilities=tuple(demanded),
            quality_is_demanded=request.budget == "best" or request.needs_4k,
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
            "generate_video(draft=True) is an equivalent shortcut to the omni "
            "route: it renders the draft on gemini-omni-flash and reports which "
            "Veo-only parameters it ignored."
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
                            reason=(
                                f"{other.model} not planned: pinned_model="
                                f"{request.pinned_model} was requested."
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
    preview of an expensive render, which is only true against the standard
    Veo tier. Omni bills $0.10136/s and Veo 3.1 Fast $0.10/s, so 3 x 8s beats
    cost $2.43 as an animatic against $2.40 for the real thing — and $1.20 on
    Veo 3.1 Lite, which the animatic then costs double. The preview keeps its
    place (catching a bad creative call before the delivery render is paid for
    is worth something on its own), but the claim has to match the arithmetic.

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
        and request.extension_implied
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
        }
        if "aspect_ratio" in best.params:
            seed_params["aspect_ratio"] = best.params["aspect_ratio"]
        extend_params = dict(best.params)
        extend_params["video_uri"] = "<video_url from step 1>"
        return (
            WorkflowStep(
                order=1,
                tool="generate_video",
                params=seed_params,
                rationale=(
                    f"Render the opening {VEO_MAX_CLIP_SECONDS}s seed clip — a "
                    "continuous shot longer than one Veo render can only be "
                    "made by extending an existing clip."
                ),
            ),
            WorkflowStep(
                order=2,
                tool="loop_extend",
                params=extend_params,
                rationale=(
                    f"Extend the seed clip {extend_params.get('times', 1)} "
                    "time(s) (~7s each) to reach the requested runtime, using "
                    "step 1's video_url."
                ),
            ),
        )

    if best.tool in ("generate_video", "generate_video_omni"):
        # Draft-then-finalize for a single render. Only worth a step when the
        # delivery render is dear enough that discovering a bad creative call
        # after paying for it hurts — and only when the preview is genuinely
        # cheaper, which is exactly what omni 1.1's 360p tier made possible.
        # A caller who has already said the render is throwaway needs no
        # preview of a preview.
        if (
            request.is_draft
            or best.model == OMNI_1_1_MODEL
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
            "omni_model": OMNI_1_1_MODEL,
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
            OMNI_1_1_MODEL,
            float(draft_params["duration_seconds"]),
            OMNI_DRAFT_RESOLUTION,
            False,
        )
        if draft_cost is None or (
            best.cost is not None and draft_cost.usd >= best.cost.usd
        ):
            return ()
        lead = f"Preview on {OMNI_1_1_MODEL} at {OMNI_DRAFT_RESOLUTION} first"
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
    # Bridges are a Veo first/last-frame feature; generate_clip ignores them
    # in animatic mode, so the preview step should not ask for them.
    animatic_params["add_bridges"] = False
    # Preview at 360p on the model that has resolutions. This is what turns
    # the animatic from a pass that costs about what the delivery render costs
    # — which is what its own rationale had to concede — into a third of it.
    # Naming the resolution is what selects the model; generate_clip's
    # animatic_resolution is documented that way.
    animatic_params["animatic_resolution"] = OMNI_DRAFT_RESOLUTION
    animatic_model = OMNI_1_1_MODEL
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
# Public entry point
# ============================================================================


def plan_generation(
    intent: str, constraints: RoutingConstraints | None = None
) -> RoutingPlan:
    """Plan how to generate what ``intent`` describes.

    Pure and deterministic: no network, no filesystem, no clock. The same
    arguments always produce an identical plan.

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

    signals = infer_signals(intent)
    request = resolve_request(intent, signals, constraints)

    if request.media_kind == "image":
        routes, rejected, conflicts, notes = _plan_image(request)
    else:
        routes, rejected, conflicts, notes = _plan_video(request)

    extra_notes = list(notes)
    if signals.media_kind is None and (
        constraints is None or constraints.media_kind is None
    ):
        extra_notes.append(
            "No image or video keyword matched; planned as an image. Set "
            "media_kind explicitly if that is wrong."
        )
    if any(route.cost is None for route in routes):
        extra_notes.append(
            "Cost estimates are unavailable (pricing module not installed); "
            "routes are ranked on capability and relative cost tier only."
        )

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
    )
