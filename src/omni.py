"""Omni video generation helpers (Interactions API).

Support for Google's Omni video models. Unlike the VEO models in ``video.py``
(which use the long-running ``generate_videos`` operation), the Omni models are
driven through the **Interactions API**: ``client.interactions.create(...,
background=True)`` starts a server-side render, ``client.interactions.get(...)``
is polled until the interaction completes, and multi-turn conversational
editing is done by threading a ``previous_interaction_id`` (the server holds
the prior video context).

Each model has its own capability record:

``gemini-omni-1.2-flash`` (default, GA on both backends)
    Generates 3-15s clips and edits sources up to 15s
    (https://ai.google.dev/gemini-api/docs/omni). On top of the preview it
    has:
      * ``resolution`` in ``response_format`` — 360p / 720p (default) /
        1080p / 4k, the last two upsampled from the base render;
      * an ``extend`` task, which appends a 3-10s continuation using the final
        10s of the source as context, with backend-specific cumulative
        limits;
      * first/last-frame interpolation via the ``<FIRST_FRAME>`` /
        ``<LAST_FRAME>`` prompt tags, which stays 3-10s;
      * video references — up to 3 clips of up to 3s each (one, on an 11-15s
        generation) — via ``<VIDEO_REF_N>``, alongside ``<IMAGE_REF_N>``
        image references;
      * the ``[# Sources ...] [# References ...]`` prefix syntax that binds
        each uploaded item to one of those roles unambiguously;
      * on Vertex AI, standalone upscaling.
    Audio references are not an API input.

``gemini-omni-flash-preview`` (``OMNI_PREVIEW_MODEL``)
    The original preview. 720p/24fps only, no resolution control, no video
    extension, no explicit keyframe roles. Everything about the request it
    receives is live-verified and is deliberately left untouched.

``gemini-omni-1.1-flash`` (``OMNI_1_1_MODEL``) is retired on this server and
has no record. Google redirects that ID to 1.2 from ``OMNI_1_1_REDIRECT_DATE``;
this server applies the same redirect now, with a warning, so a client that
hard-codes it keeps working (``canonical_omni_model``). An interaction created
on 1.1 cannot be continued, because an interaction cannot change models.

Request/response shapes follow the Vertex AI "Use Gemini Omni Flash …to
generate videos" REST reference and the Interactions API docs:
  * media rides INSIDE ``input`` as flattened parts
    ({type: 'text'|'image'|'video'|'document', text|data|uri, mime_type}) —
    there are no separate image/video kwargs;
  * ``response_format`` is a LIST of one object
    ``[{'type': 'video', 'aspect_ratio': ..., 'duration': 'Ns',
    'resolution': ..., 'delivery': 'uri', 'gcs_uri': ...}]``. aspect_ratio
    ("16:9"/"9:16"), duration ("3s" up to the model's maximum: "15s" on 1.2,
    "10s" on the preview) and resolution live here;
    ``delivery='uri'`` + ``gcs_uri`` sends output to Cloud Storage, otherwise
    the video bytes come back inline. The Gemini API docs show a bare object
    rather than a list; the SDK normalizes both, and the list is what is
    live-verified here;
  * ``generation_config={'video_config': {'task': ...}}`` carries the task
    type (text_to_video / image_to_video / reference_to_video / edit, plus
    extend and upscale on 1.2) — and NOTHING ELSE. The GA release
    note of 2026-08-27 describes resolution as a "new ``resolution`` parameter in
    ``video_config``"; that is loose prose in a hand-written note. Three
    generated or executable sources say response_format: the task guide's
    own runnable sample, the Interactions API reference schema, and
    google-genai's types, where ``VideoConfig`` has exactly one field
    (``task``) and ``VideoResponseFormat`` is the one carrying
    ``resolution``. Moving it would put the field somewhere the SDK drops on
    serialization, which is the silent 720p-render/4K-bill failure this
    module already guards against. Do not move it without a live 4K render
    proving the note right;
  * ``background=True`` runs the (minute-plus) render asynchronously; the
    interaction id is polled until ``status == 'completed'``;
  * a finished interaction carries the clip in a ``model_output`` step's
    ``content[]`` video part (inline base64 ``data`` or a hosted ``uri``);
    ``thought`` and ``user_input`` steps are skipped; newer SDKs also expose
    a convenience ``output_video``.

On Vertex the interactions collection is location ``global``
(…/locations/global/interactions) — the caller must supply a global-location
client. NOTE: the Vertex REST example places ``background`` inside the first
``input`` part; that reads as a doc artifact (it is a request-level flag), so
it is sent top-level here, matching the Interactions API convention. If a
live call rejects it, move it into the input part.

SDK FLOOR. ``resolution`` is a field google-genai only learned in 2.20.0.
Older SDKs parse the request into a ``VideoResponseFormat`` that has no such
field and drop it on serialization *silently* — the render comes back at 720p
while this server quotes and bills the resolution that was asked for. Because
that failure is invisible on the wire, it is checked at import
(``RESOLUTION_REACHES_THE_WIRE``) and refused per-request rather than trusted
to the dependency pin alone.
"""

import asyncio
import base64
import functools
import io
import math
import re
import datetime
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from google import genai

from .video import http_status, refused_before_work, retry_after_seconds, run_off_loop

# Type for async log callback from MCP context
LogCallback = Callable[[str], Awaitable[None]]

OMNI_PREVIEW_MODEL = "gemini-omni-flash-preview"
OMNI_1_2_MODEL = "gemini-omni-1.2-flash"

# Retired here: no spec, no price, no route. Kept as a name because clients
# hard-code it and manifests written before the retirement record it -- a
# request naming it is redirected (canonical_omni_model), and an interaction
# recorded on it is recognised so it can be refused rather than sent to a
# model that does not hold it.
OMNI_1_1_MODEL = "gemini-omni-1.1-flash"
# The date Google itself starts serving the 1.1 ID on 1.2. This server does
# not wait for it; the date is quoted in the redirect warning so a caller
# knows the change is Google's and not a quirk of this server.
OMNI_1_1_REDIRECT_DATE = "2026-12-03"

# Both spellings of the retired model: Vertex AI published 1.1 as
# `gemini-omni-1.1-flash-preview`, the Developer API as the bare name, and
# renders from either backend recorded theirs. Each runs on 1.2 now.
_RETIRED_OMNI_MODELS: dict[str, str] = {
    OMNI_1_1_MODEL: OMNI_1_2_MODEL,
    "gemini-omni-1.1-flash-preview": OMNI_1_2_MODEL,
}

# The Gemini API deprecations page lists gemini-omni-flash-preview with an
# earliest shutdown of October 22, 2026, the same day as the Veo preview IDs
# (the 2026-08-27 release notes had said September 30). Surfaced on every call
# that still pins it, because a caller who reads a warning has time to move
# and a caller who reads nothing does not.
OMNI_PREVIEW_SUNSET = "2026-10-22"


def _today() -> datetime.date:
    """Today — a function so a test can pin the calendar."""
    return datetime.date.today()


def omni_preview_is_sunset(today: datetime.date | None = None) -> bool:
    """Whether the preview endpoint's switch-off date has arrived."""
    return (today or _today()) >= datetime.date.fromisoformat(OMNI_PREVIEW_SUNSET)


def omni_model_refusal(model: str | None) -> str | None:
    """Why ``model`` must not be called at all today, or None.

    Until the sunset the preview model is carried with a warning on every
    call. On the day, a warning becomes a lie: the endpoint is gone, and a
    quote or a render attempted against it is a 404 with a price beside it.
    Refused at the pre-flight instead, so a dry run says so for free.
    """
    if (
        canonical_omni_model(model) != OMNI_PREVIEW_MODEL
        or not omni_preview_is_sunset()
    ):
        return None
    # The facts below are the default's, 1.2 (OMNI_SPECS, pricing.py).
    return (
        f"{OMNI_PREVIEW_MODEL} was switched off on {OMNI_PREVIEW_SUNSET} and can "
        f"no longer be called. Use {DEFAULT_OMNI_MODEL}: generally available on "
        "both the Gemini Developer API and Vertex AI, at a lower published "
        "720p rate than the preview, with resolution control, clips up to 15s, "
        "extension and first/last-frame interpolation."
    )


def _access_refusal(
    exc: BaseException, served_model: Any, *, vertexai: bool
) -> ValueError | None:
    """Turn a 403/404 from interactions.create into advice, or None.

    The raw error names an endpoint and a project; it does not say what to
    check, or, for the preview model, that the way off an endpoint being
    switched off is the generally available default rather than an access
    request.
    """
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    text = str(exc)
    if isinstance(code, int) and code not in (403, 404):
        # The service SAID what this is. A 400 whose body happens to mention a
        # 404 (an expired Files-API upload reports NOT_FOUND inside a 400), or
        # a 504 reading "timed out after 403 seconds", is not an access
        # problem, and the text matchers below were rewrapping both as
        # allowlist advice. Those matchers exist for exceptions that carry no
        # code at all, so they run only then.
        return None
    # Deliberately narrow. A first cut matched "404"/"403" as substrings and
    # "is not supported for" anywhere, which would have rewrapped a duration
    # 400 or a timeout carrying "4034ms" as an allowlist problem — advice that
    # sends the caller to fix credentials that were never the issue.
    looks_like_access = (
        code in (403, 404)
        or re.search(r"\b(?:404|403)\b", text) is not None
        or "NOT_FOUND" in text
        or "PERMISSION_DENIED" in text
        or "does not have access" in text
        or "allowlist" in text.lower()
        or ("model" in text.lower() and "not found" in text.lower())
    )
    if not looks_like_access:
        return None
    backend = "Vertex AI" if vertexai else "The Gemini Developer API"
    if canonical_omni_model(served_model) == OMNI_1_2_MODEL:
        return ValueError(
            f"{backend} refused model '{served_model}': {text[:200]}. Check "
            "project access, API enablement and quota for Gemini Omni 1.2 Flash "
            "(generally available)."
        )
    # The preview model, the only other one this server calls.
    return ValueError(
        f"{backend} refused model '{served_model}': {text[:200]}. Check the "
        "model id, and that the project has access to the Interactions API, or "
        f"use {DEFAULT_OMNI_MODEL}, which is generally available on both backends."
    )


# Kept as the preview model's name, not as "the default" — it is imported all
# over the tree as an identity (which price, which capability set) and renaming
# it would churn every call site for nothing.
OMNI_MODEL = OMNI_PREVIEW_MODEL

# What every omni entry point defaults to. This WAS the preview model, on the
# reasoning that flipping it would move existing callers onto a different
# render without asking. The deprecation settles it the other way: the preview
# endpoint is switched off on OMNI_PREVIEW_SUNSET, so defaulting to it means
# defaulting new callers onto a dead endpoint, and 1.2 is GA on both backends
# at a lower published 720p rate with strictly more capability.
DEFAULT_OMNI_MODEL = OMNI_1_2_MODEL

# The models this server has a record for. Both are spelled the same on both
# backends, so the ID sent is the ID named.
OMNI_MODELS: tuple[str, ...] = (OMNI_PREVIEW_MODEL, OMNI_1_2_MODEL)

# Every spelling this server accepts, including the retired ones it redirects.
OMNI_MODEL_ALIASES: tuple[str, ...] = (*OMNI_MODELS, *_RETIRED_OMNI_MODELS)


def omni_model_is_retired(model: str | None) -> bool:
    """Whether ``model`` names an omni model this server has retired."""
    return str(model) in _RETIRED_OMNI_MODELS


def canonical_omni_model(model: str | None) -> str:
    """The omni model a call naming ``model`` runs on.

    A retired ID folds onto its replacement; anything else passes through
    unchanged, so is_omni_model and omni_spec still judge an unknown name.
    """
    name = str(model)
    return _RETIRED_OMNI_MODELS.get(name, name)


def omni_model_reroute_warning(requested: str | None) -> str | None:
    """The warning a request naming a retired omni ID carries, or None.

    The request runs on the replacement rather than being refused, so a
    client that hard-codes the old ID keeps working -- Google makes the same
    move on OMNI_1_1_REDIRECT_DATE. Running it silently would render and bill
    a model the caller never named. Shared by every quote, render and plan so
    none of them can describe the redirect differently.
    """
    if requested is None or not omni_model_is_retired(requested):
        return None
    target = canonical_omni_model(requested)
    return (
        f"{requested} is retired on this server; requests naming it run on "
        f"{target}, with {target}'s capabilities and pricing (Google applies "
        f"the same redirect to that ID from {OMNI_1_1_REDIRECT_DATE}). Name "
        f"{target} to drop this warning."
    )


# Output spec limits documented for both omni models.
_SUPPORTED_ASPECT_RATIOS = ("16:9", "9:16")

# Documented output duration floor (sent as "Ns" in response_format). The
# ceiling is per model -- OmniModelSpec.max_duration_seconds, 10s on the
# preview and 15s on 1.2 -- so it lives on the spec, not here.
_MIN_DURATION = 3

# Resolutions gemini-omni-1.2-flash accepts, in the spelling this server uses
# publicly ("4K" matches src/video.py and the pricing tables). The API spells
# the top tier lowercase, hence the translation table below.
OMNI_RESOLUTIONS: tuple[str, ...] = ("360p", "720p", "1080p", "4K")
OMNI_DEFAULT_RESOLUTION = "720p"

# Public spelling -> the literal the Interactions API expects.
_API_RESOLUTIONS: dict[str, str] = {
    "360p": "360p",
    "720p": "720p",
    "1080p": "1080p",
    "4K": "4k",
}

# Spellings a caller might reasonably send, normalized to the public set.
_RESOLUTION_ALIASES: dict[str, str] = {
    "360P": "360p",
    "720P": "720p",
    "1080P": "1080p",
    "4K": "4K",
    "2160P": "4K",
}

# Documented task types. ``extend`` and ``upscale`` are 1.2's; the preview has
# neither.
_TASK_TEXT_TO_VIDEO = "text_to_video"
_TASK_IMAGE_TO_VIDEO = "image_to_video"
_TASK_REFERENCE_TO_VIDEO = "reference_to_video"
_TASK_EDIT = "edit"
_TASK_EXTEND = "extend"
_TASK_UPSCALE = "upscale"

_ALL_TASKS = (
    _TASK_TEXT_TO_VIDEO,
    _TASK_IMAGE_TO_VIDEO,
    _TASK_REFERENCE_TO_VIDEO,
    _TASK_EDIT,
    _TASK_EXTEND,
    _TASK_UPSCALE,
)

# Tasks that continue existing footage rather than rendering something new.
_CONTINUATION_TASKS = (_TASK_EDIT, _TASK_EXTEND)


def _strips_output_spec(
    task_type: str | None, previous_interaction_id: str | None
) -> bool:
    """Whether this request must leave ``aspect_ratio`` and ``duration`` off.

    Every continuation must. Three separate live 400s say so:

    * an ``edit`` rejects duration — "Duration cannot be set in response
      format for edit task";
    * an ``extend`` rejects aspect ratio — "Aspect ratio cannot be set in
      response format for extend task". Vertex's own documented
      extend-videos request DOES send it, so the documentation and the
      service disagree and the service wins;
    * a turn carrying ``previous_interaction_id`` cannot say which task it is
      (the API rejects a task alongside it -- upscale on 1.2 is the one
      documented exception, see _build_create_kwargs), so the service infers
      one and there is no way to tell from here which of the two rejections
      applies.

    An earlier version of this function narrowed to the edit case alone on
    the strength of Vertex's sample request, which 400s every uploaded
    extension.

    This decides ``aspect_ratio``, and ``duration`` everywhere but an
    extension: there ``duration`` is the per-turn length the caller chose,
    documented for 1.2, and _sends_duration sends it.
    """
    return previous_interaction_id is not None or task_type in (
        *_CONTINUATION_TASKS,
        _TASK_UPSCALE,
    )


def _sends_duration(
    model: str, task_type: str | None, previous_interaction_id: str | None
) -> bool:
    """Whether this request carries ``response_format.duration`` when one is given.

    The one place that decides it, so the request builder, the validation of
    the value and the warning describing it cannot disagree. Validating a
    duration an edit never sends refused a render over a value the service
    would never see, while the dry run, which knows the edit ignores it,
    quoted that same call.

    Everything _strips_output_spec strips, except an extension. The
    Gemini API's Omni guide (https://ai.google.dev/gemini-api/docs/omni,
    "Video extension" and "Extension and total duration limits") documents
    the per-turn length as "a 3 to 10 second continuation (10 seconds by
    default, or set using ``duration`` in ``response_format``)" for
    "uploaded or model-generated videos", and lists
    ``previous_interaction_id`` as the way to extend a model-generated one.
    Vertex's extend-videos page sends ``duration`` with ``task: extend`` on
    an uploaded source and has no conversational example. A conversational
    turn still cannot declare its task, so the service infers extension from
    the prompt; if it reads the turn as an edit it refuses the duration
    before rendering (generate_video_omni turns that 400 into advice).
    Whether a conversational turn sends it at all is
    OMNI_CONVERSATIONAL_EXTENSION_SENDS_DURATION's call, pending a live check.
    """
    spec = OMNI_SPECS.get(canonical_omni_model(model))
    if spec is not None and spec.supports_extend and task_type == _TASK_EXTEND:
        return (
            previous_interaction_id is None
            or OMNI_CONVERSATIONAL_EXTENSION_SENDS_DURATION
        )
    return not _strips_output_spec(task_type, previous_interaction_id)


def omni_duration_is_sent(
    model: str,
    *,
    previous_interaction_id: str | None = None,
    has_input_video: bool = False,
    inferred_image_count: int = 0,
    reference_image_count: int = 0,
    has_first_frame: bool = False,
    has_last_frame: bool = False,
    reference_video_count: int = 0,
    task: str | None = None,
) -> bool:
    """Whether a request of this shape carries ``response_format.duration``.

    For a caller that has counts and URIs rather than bytes: the tool layer
    validates a duration only when generate_video_omni will send it, and it
    asks here, through the same task selection and the same _sends_duration,
    so the two layers cannot disagree. The tool used to validate every
    duration, which refused (on both its dry run and its real run) an edit
    over a value the service never receives -- 15s beside a first and last
    frame on a continuation drew "3 to 10" while the impl would have sent no
    duration at all.
    """
    task_type = _select_task_type(
        previous_interaction_id=previous_interaction_id,
        input_video_bytes=b"" if has_input_video else None,
        inferred_image_count=inferred_image_count,
        reference_image_count=reference_image_count,
        has_first_frame=has_first_frame,
        has_last_frame=has_last_frame,
        reference_video_count=reference_video_count,
        requested_task=task,
    )
    return _sends_duration(model, task_type, previous_interaction_id)


@dataclass(frozen=True)
class OmniModelSpec:
    """What one omni model accepts, as data rather than scattered branches.

    Every field is a documented restriction of that specific model, so a
    capability check reads the same way for both and a third model is a table
    entry rather than a new code path.

    Attributes:
        model: Model ID as sent to the API.
        resolutions: Output resolutions the model accepts, empty when it has
            no resolution parameter at all.
        rendered_resolution: What comes back when no resolution is requested —
            for the preview model, the only thing it ever renders.
        duration_is_documented: True when the model's own reference documents
            ``duration`` in ``response_format``. False means the field is
            still in the SDK schema and still sent, but a caller who relies on
            it is told the docs do not back it.
        supports_extend: Whether the ``extend`` task exists.
        supports_keyframes: Whether ``<FIRST_FRAME>``/``<LAST_FRAME>``
            interpolation exists.
        supports_role_tags: Whether the ``[# Sources ...]`` prefix syntax and
            ``<IMAGE_REF_N>``/``<VIDEO_REF_N>`` tags are understood.
        max_reference_videos: How many reference clips may ride along.
        max_reference_video_seconds: Documented length cap per reference clip.
        extension_step_seconds: The longest, and the default, length one
            extension turn appends; the caller may ask for anything from 3s
            up to it, sent as ``duration``.
        max_extended_seconds: Documented cumulative ceiling for a chain of
            extensions.
        max_uploaded_source_seconds: Documented cap, on both backends, on a
            video to edit or upscale -- uploaded or continued through
            previous_interaction_id alike. An extension reads only the
            source's final seconds, so it takes a source of any length.
        media_before_text: Whether media parts precede the text part, matching
            the ordering that model's own reference shows.
        max_inline_output_seconds: Output length past which a render is
            expected to exceed the 4 MB inline response limit at any
            resolution; 0 when only the resolution and task decide.
        max_vertex_edit_images: How many images a Vertex AI edit may carry,
            whatever their roles; 0 when no backend-specific cap is
            documented.
        reads_gcs_source_by_uri: Whether a gs:// input video is sent to
            Vertex AI as a URI part rather than fetched and inlined.
        sends_inferred_task: Whether a task the server infers is sent. 1.2
            recommends prompting first, and a forced task restricts its
            references, so it sends a task only when the caller names one.
        vertex_max_extended_seconds: Vertex AI's cumulative ceiling for an
            extended clip when it differs from max_extended_seconds; 0 when
            it does not.
        vertex_extension_window_seconds: How much of the input Vertex AI
            retains on each extension turn; 0 when it keeps all of it.
        max_seconds_with_video_references: The longest render that may
            carry more than one video reference; 0 when no such limit is
            documented.
    """

    model: str
    resolutions: tuple[str, ...]
    rendered_resolution: str
    duration_is_documented: bool
    supports_extend: bool
    supports_keyframes: bool
    supports_role_tags: bool
    max_reference_videos: int
    max_reference_video_seconds: float
    extension_step_seconds: int
    max_extended_seconds: int
    max_uploaded_source_seconds: float
    media_before_text: bool
    max_duration_seconds: int = 10
    max_keyframe_seconds: int = 10
    max_reference_images: int = 8
    supports_upscale: bool = False
    extension_window_seconds: int = 0
    # The fields below default to what the preview does, so a model entry
    # only names where it differs, and the code reads the field instead of
    # asking which model it is holding.
    max_inline_output_seconds: float = 0.0
    max_vertex_edit_images: int = 0
    reads_gcs_source_by_uri: bool = False
    sends_inferred_task: bool = True
    vertex_max_extended_seconds: int = 0
    vertex_extension_window_seconds: int = 0
    max_seconds_with_video_references: int = 0

    @property
    def supports_resolution(self) -> bool:
        """Whether ``resolution`` can be sent at all for this model."""
        return bool(self.resolutions)


OMNI_SPECS: dict[str, OmniModelSpec] = {
    OMNI_PREVIEW_MODEL: OmniModelSpec(
        model=OMNI_PREVIEW_MODEL,
        # No resolution parameter: the preview renders 720p/24fps, full stop.
        resolutions=(),
        rendered_resolution="720p",
        duration_is_documented=True,
        supports_extend=False,
        supports_keyframes=False,
        supports_role_tags=False,
        max_reference_videos=0,
        max_reference_video_seconds=0.0,
        extension_step_seconds=0,
        max_extended_seconds=0,
        max_uploaded_source_seconds=0.0,
        # Text-first is what the live-verified preview requests send.
        media_before_text=False,
    ),
    OMNI_1_2_MODEL: OmniModelSpec(
        model=OMNI_1_2_MODEL,
        resolutions=OMNI_RESOLUTIONS,
        rendered_resolution=OMNI_DEFAULT_RESOLUTION,
        # The Interactions API reference lists duration among
        # VideoResponseFormat's fields, and Vertex's own extend-videos request
        # sends it. An earlier read of the task-guide page alone concluded
        # otherwise and warned callers off it; the reference settles it.
        duration_is_documented=True,
        supports_extend=True,
        supports_keyframes=True,
        supports_role_tags=True,
        # "Video references support a maximum of 3 clips, up to 3 seconds each."
        max_reference_videos=3,
        max_reference_video_seconds=3.0,
        # "a 3 to 10 second continuation (10 seconds by default, or set using
        # duration in response_format)".
        extension_step_seconds=10,
        max_extended_seconds=30,
        # "Maximum input video length (upscaling): 15 seconds", and editing
        # takes the same 15s on both backends and for a multi-turn source as
        # well as an upload; an extension reads only the final 10s of an
        # input of any length.
        max_uploaded_source_seconds=15.0,
        # Every example in the reference puts the media parts first.
        media_before_text=True,
        max_duration_seconds=15,
        max_reference_images=6,
        supports_upscale=True,
        # A 1.2 render longer than 10s is treated as able to exceed the 4 MB
        # inline response limit, even at 720p.
        max_inline_output_seconds=10.0,
        # Vertex AI's 1.2 editing takes at most 2 images in one request.
        max_vertex_edit_images=2,
        reads_gcs_source_by_uri=True,
        sends_inferred_task=False,
        # Vertex appends up to 10s to the final 30s of the source; the
        # Developer API documents a 30s cumulative limit instead.
        vertex_max_extended_seconds=40,
        vertex_extension_window_seconds=30,
        # "Videos from 11 to 15 seconds support at most 1 reference video."
        max_seconds_with_video_references=10,
    ),
}

# The longest single render ANY omni model produces (1.2's 15s), derived from
# the specs so it cannot drift from them. It is a model-agnostic ceiling for a
# caller that does not yet know which model will serve a render, and it
# over-states for the preview (10s) by design. It is NOT a per-model limit:
# code that knows the model must read omni_spec(model).max_duration_seconds
# instead, as everything in this module does. It used to be exactly 10s,
# every model's limit, and was read that way; at 15s that reading over-quotes
# a preview render by 50% and lets through a duration the preview cannot
# render.
OMNI_MAX_DURATION_SECONDS = max(
    spec.max_duration_seconds for spec in OMNI_SPECS.values()
)


# The one switch for whether an Omni 1.2 extension turn that continues an
# interaction (it carries previous_interaction_id) sends
# response_format.duration. The Gemini API's Omni guide documents the 3-10s
# per-turn duration for "uploaded or model-generated videos" and names
# previous_interaction_id as the way to extend a model-generated one; Vertex's
# extend-videos page shows duration only on an uploaded source, and no live
# render has yet sent it on a conversational turn. tests/test_live_omni_1_2.py's
# L2 tests settle it, one per backend: if a conversational turn rejects
# duration (or ignores it), set this to False. Such turns then send none and
# append the documented 10s default, and extend_video_omni refuses a
# duration_seconds it could not honour (omni_extension_duration_refusal)
# instead of failing on the service, possibly after a first turn has billed;
# extend_video_omni's docstring should then say so too.
OMNI_CONVERSATIONAL_EXTENSION_SENDS_DURATION = True


def omni_extension_duration_refusal(
    spec: OmniModelSpec,
    *,
    duration_seconds: float | None,
    continues_interaction: bool,
    times: int,
) -> str | None:
    """Why an extension cannot honour duration_seconds, or None when it can.

    Only when OMNI_CONVERSATIONAL_EXTENSION_SENDS_DURATION is off: then only
    a turn that uploads its source sends a duration, which in one call is the
    first turn of an input_video_uri extension, and only when it is the only
    turn. Read at call time, so a dry run and a render agree whichever way
    the switch is set.
    """
    if (
        duration_seconds is None
        or OMNI_CONVERSATIONAL_EXTENSION_SENDS_DURATION
        or (not continues_interaction and times <= 1)
    ):
        return None
    which = (
        "every turn of this call does"
        if continues_interaction
        else "every turn after the first does"
    )
    return (
        "duration_seconds can set an extension turn's length only on a turn "
        f"that uploads its source: {spec.model} turns that continue an "
        f"interaction take no duration, and {which}. Leave duration_seconds "
        f"unset to append the {spec.extension_step_seconds}s default on every "
        "turn, or extend an uploaded clip one turn per call "
        "(input_video_uri, times=1) to choose that turn's length."
    )


def omni_source_limit_seconds(spec: OmniModelSpec, *, task: str | None = None) -> float:
    """Documented ceiling on an edit, extension or upscale source; 0 for none.

    The one home of the source-length rule, read off the spec. The figure is
    the same on both backends and binds a source continued through
    previous_interaction_id as well as an upload. An extension has none: it
    reads only the source's final seconds, and its assembled output has the
    backend's cumulative ceiling instead.

    Args:
        spec: The model's capability record.
        task: The operation; only ``extend`` changes the answer.
    """
    if task == _TASK_EXTEND:
        return 0.0
    return spec.max_uploaded_source_seconds


def is_omni_model(model: str | None) -> bool:
    """Whether a call naming ``model`` is served through the Interactions API.

    True for a retired ID too, so a hard-coded or recorded one still routes,
    and prices as the model that serves it.
    """
    return canonical_omni_model(model) in OMNI_SPECS


def omni_spec(model: str | None, *, vertexai: bool = False) -> OmniModelSpec:
    """Return the capability record for the model a call naming ``model`` runs on.

    Raising (rather than falling back to the default spec) keeps a typo from
    quietly buying the preview model's restrictions on a 1.2 request.
    """
    spec = OMNI_SPECS.get(canonical_omni_model(model))
    if spec is None:
        raise ValueError(
            f"Unknown omni model '{model}'. Supported: {', '.join(OMNI_MODELS)}."
        )
    if vertexai and spec.vertex_extension_window_seconds:
        return replace(
            spec,
            max_extended_seconds=spec.vertex_max_extended_seconds
            or spec.max_extended_seconds,
            extension_window_seconds=spec.vertex_extension_window_seconds,
        )
    return spec


def normalize_omni_resolution(resolution: str | None) -> str | None:
    """Normalize a resolution spelling to this server's public set, or None."""
    if resolution is None:
        return None
    key = str(resolution).strip().upper()
    return _RESOLUTION_ALIASES.get(key)


def _sdk_serializes_resolution() -> bool:
    """Whether the installed google-genai puts ``resolution`` on the wire.

    google-genai models ``response_format`` as a typed object and serializes
    only its declared fields, so an SDK older than 2.20.0 drops ``resolution``
    without an error, a warning, or a trace of it in the request body. The
    render then comes back 720p while this server reports (and bills) the
    resolution the caller asked for. Detecting it here turns an invisible
    billing lie into a refusal at the call site.
    """
    try:
        from google.genai._gaos.types.interactions.videoresponseformat import (  # pyright: ignore[reportPrivateUsage]
            VideoResponseFormat,
        )
    except Exception:  # pragma: no cover - SDK layout changed
        # Unknown layout: assume the field travels rather than blocking every
        # render that names a resolution on a probe that no longer applies.
        return True
    return "resolution" in VideoResponseFormat.model_fields


RESOLUTION_REACHES_THE_WIRE = _sdk_serializes_resolution()

# Cap on the delivered-video download. Mirrors MAX_FETCH_BYTES (50 MB) in
# src/__main__.py — defined here rather than imported because this module does
# not own that file. files.download buffers the whole response body, so without
# this the one delivered-file path would be the only fetch in the server that
# reads an untrusted body uncapped; a 720p/<=10s clip is far under it in
# practice, so this is defence-in-depth matching the rest of the codebase.
# The preview model could only ever emit 720p/<=10s, for which 50 MB was
# pure defence-in-depth. 1.2 reaches 4K and 40s, and the URI-delivery path
# exists precisely for those — a 40s 4K clip plausibly clears 50 MB, so the
# old cap would have failed a render that had already been billed, mid-chain.
# 256 MB clears the largest output the model documents at a generous bitrate
# while still bounding what one response can allocate here. GCS delivery
# (Vertex + output_gcs_uri) avoids the buffer entirely and has no cap.
_MAX_DELIVERED_VIDEO_BYTES = 50 * 1024 * 1024
_MAX_LARGE_DELIVERED_VIDEO_BYTES = 256 * 1024 * 1024


def _delivered_video_cap(spec: OmniModelSpec) -> int:
    """Download cap for a render this model could produce."""
    if spec.supports_resolution or spec.supports_extend:
        return _MAX_LARGE_DELIVERED_VIDEO_BYTES
    return _MAX_DELIVERED_VIDEO_BYTES


# Above this, an input video goes through the Files API instead of riding as
# inline base64. Google's own guidance ("since videos can be quite big we
# recommend using the File API") has no number on it; 8 MB is chosen because
# base64 inflates by a third, so this keeps a single request body under ~11 MB
# — comfortably inside the JSON payload limits — while leaving the short
# clips omni actually takes on the simpler inline path.
_MAX_INLINE_VIDEO_BYTES = 8 * 1024 * 1024
OMNI_VERTEX_INLINE_VIDEO_BYTES = 20_000_000

# Default deadline for a whole interaction (create + polling). Deliberately
# under the ~4 minute ceiling common MCP hosts impose on a single tool call: a
# render that outlives the HOST's limit is billed with no result, no id and no
# way to retrieve or cancel it, so the spend cannot be reconciled at all. Under
# this limit the call returns a TimeoutError naming the interaction_id instead,
# which is recoverable. Raise it only if the host's own ceiling is higher.
OMNI_DEFAULT_TIMEOUT_SECONDS = 210

# MEASURED, not estimated: Vertex AI, 2026-11-09, submit to downloaded file
# (tests/test_live_omni_1_2.py L3b and L3). A 3.03s 4K render took 153.8s and
# a 10.03s 720p->4K upscale took 210.0s, as (output seconds, wall seconds).
# 4K is the slowest tier on both paths; nothing below 4K has come near the
# default deadline (a 10s 720p render took 46s).
OMNI_MEASURED_4K_RENDER = (3.03, 153.8)
OMNI_MEASURED_4K_UPSCALE = (10.03, 210.0)
# Headroom over the measured rate before a deadline counts as enough, as for
# Veo's 4K tier (src.__main__._veo_slow_tier_warning).
_SLOW_TIER_MARGIN = 1.2


def omni_slow_tier_warning(
    *,
    resolution: str | None,
    kind: str,
    output_seconds: float | None,
    timeout_seconds: float,
    seconds_are_bound: bool = False,
) -> str | None:
    """Warn, before anything renders, when a 4K output will likely outlast the deadline.

    A render that outlives the deadline is billed and returns a timeout
    naming its interaction_id, and nothing here can fetch a finished render by
    that id: the money is spent and the video has to be rendered again. The
    expected time scales the measured 4K figure above by the output's length,
    a straight line through the one measurement there is for each path, so it
    is a rough guide and is described as one.

    Args:
        resolution: The normalized output resolution; only "4K" is judged.
        kind: "upscale" for upscale_video_omni, "extend" for an extension
            turn (which renders the whole assembled clip), anything else for a
            render or an edit.
        output_seconds: The output's length, or a bound on it. None counts as
            the model's longest single render (15s).
        timeout_seconds: The deadline the call will run under.
        seconds_are_bound: output_seconds is an upper bound, not a length.
    """
    if resolution != "4K":
        return None
    measured_seconds, measured_wall = (
        OMNI_MEASURED_4K_UPSCALE if kind == _TASK_UPSCALE else OMNI_MEASURED_4K_RENDER
    )
    if not output_seconds or output_seconds <= 0:
        output_seconds = float(omni_spec(OMNI_1_2_MODEL).max_duration_seconds)
        seconds_are_bound = True
    expected = measured_wall * output_seconds / measured_seconds
    needed = expected * _SLOW_TIER_MARGIN
    if timeout_seconds >= needed:
        return None
    if kind == _TASK_UPSCALE:
        what = (
            f"A 4K upscale of a source of up to {output_seconds:g}s"
            if seconds_are_bound
            else f"A 4K upscale of a {output_seconds:g}s source"
        )
        measured = (
            f"a {measured_seconds:g}s 720p-to-4K upscale MEASURED "
            f"{measured_wall:.0f}s end to end"
        )
        smaller = "upscale to 1080p, or upscale a shorter clip"
    else:
        if kind == _TASK_EXTEND:
            what = (
                "A 4K extension turn, which renders the whole assembled clip "
                f"(at least {output_seconds:g}s here),"
            )
        elif seconds_are_bound:
            what = f"A 4K render that could run up to {output_seconds:g}s"
        else:
            what = f"A {output_seconds:g}s 4K render"
        measured = (
            f"a {measured_seconds:g}s 4K render MEASURED {measured_wall:.0f}s "
            "end to end"
        )
        smaller = "render at 1080p or shorter"
    return (
        f"{what} is likely to take about {expected:.0f}s: {measured} on "
        f"Vertex AI. This call's deadline is {timeout_seconds:g}s, so it will "
        "likely be cut short after the render has been billed, and the "
        "timeout's interaction_id cannot be used to fetch the finished video "
        f"here. Raise timeout_seconds past {needed:.0f}s; many MCP hosts cap a "
        "single tool call near 240s, in which case no timeout_seconds value "
        f"will help: {smaller}, or call the tool from a client without that "
        "ceiling."
    )


# Interval (seconds) between polls of an in-flight background interaction.
_POLL_INTERVAL = 5

# Interval between polls of a Files API upload waiting to leave PROCESSING.
_FILE_POLL_INTERVAL = 2

# A create refused with 429 started nothing (refused_before_work), so the same
# request is sent again: after the wait the refusal asked for, else 10s and
# then 20s, each capped at a minute. Only while the deadline leaves a render's
# worth of time after the wait, since a resend that then times out leaves a
# billed render behind.
_QUOTA_RETRY_WAITS_SECONDS: tuple[float, ...] = (10.0, 20.0)
_QUOTA_RETRY_MAX_WAIT_SECONDS = 60.0
_QUOTA_RETRY_RENDER_SECONDS = 60.0

# A status poll that fails with a 429 or a 5xx is a read of something that
# already exists; it is skipped, and the next poll reads it again. This many
# in a row ends the wait with the error.
_POLL_ERROR_LIMIT = 3

# How often an unchanged status is worth repeating. Logging every poll turned
# one routine render into 15 notifications, 13 of them the identical
# "in_progress" line, and an animatic multiplies that by its beat count.
# src/video.py polls for up to 1800s and logs twice; this keeps the state
# changes plus a sparse heartbeat so a long render still shows liveness.
_POLL_LOG_INTERVAL_SECONDS = 60.0

# Interaction statuses that mean "still rendering — keep polling". Everything
# else (failed / cancelled / budget_exceeded / incomplete / requires_action /
# unknown) is terminal: fail fast instead of polling to the full timeout.
_IN_FLIGHT_STATUSES = ("in_progress", "queued")

# The statuses an interaction ends on without completing, from google-genai's
# InteractionStatus. Any other status the poll stops on is not known to be
# final, so the render may yet complete and bill.
OMNI_ENDED_STATUSES = frozenset(
    {"failed", "cancelled", "incomplete", "budget_exceeded"}
)


def _ftyp_brand(data: bytes) -> str | None:
    """Return the ISO-BMFF major brand (the token after the `ftyp` box) or None.

    PNG/JPEG/WebP don't use ftyp; MP4/MOV/HEIC/HEIF all do, distinguished by
    this brand.
    """
    if len(data) >= 12 and data[4:8] == b"ftyp":
        return data[8:12].decode("ascii", "ignore").strip().lower()
    return None


# HEIF-family ftyp brands → still images (HEIC/HEIF).
_HEIF_BRANDS = {"heic", "heix", "heim", "heis", "hevc", "hevx"}
_HEIF_SEQUENCE_BRANDS = {"mif1", "msf1"}


def _detect_image_mime(data: bytes) -> str:
    """Return the exact image MIME type from magic bytes, or raise.

    Detects the formats omni accepts as image input. Rejects unknown bytes
    rather than mislabeling them (e.g. defaulting HEIC to image/png) — an
    accurate label lets the API accept supported formats and reject the rest
    with a clear error, instead of us silently sending the wrong type.
    """
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    brand = _ftyp_brand(data)
    if brand in _HEIF_BRANDS:
        return "image/heic"
    if brand in _HEIF_SEQUENCE_BRANDS:
        return "image/heif"
    raise ValueError(
        "Unrecognized image input format; could not detect a supported image "
        "MIME type from the data. Supported: PNG, JPEG, WebP, HEIC/HEIF, GIF."
    )


# The SDK's VideoContentMimeType literal set
# (google/genai/_gaos/types/interactions/videocontent.py) spells several
# containers differently from the registered names that `mimetypes` and most
# tooling produce, and a spelling outside that set rides as UnrecognizedStr.
# One table for every path that labels a video part, so a .mov read from its
# bytes and a .mov named by a gs:// URI go out under the same label.
_SDK_VIDEO_MIME_TYPES: dict[str, str] = {
    "video/quicktime": "video/mov",
    "video/x-msvideo": "video/avi",
    # WebM is the Matroska profile the SDK names; magic-byte detection cannot
    # tell the two apart either (both open with the same EBML header).
    "video/x-matroska": "video/webm",
    "video/x-ms-wmv": "video/wmv",
    # .3gp is registered as audio; it reaches here as an input VIDEO.
    "audio/3gpp": "video/3gpp",
}


def _sdk_video_mime(mime_type: str) -> str:
    """Translate a registered video MIME spelling into the SDK's own."""
    return _SDK_VIDEO_MIME_TYPES.get(mime_type.lower(), mime_type)


# A video known only by its URI is labelled from its extension, in the SDK's
# spelling. A table rather than mimetypes.guess_type, whose answers depend on
# the host's mime.types: on one machine it named .ts a Qt translation file
# (text/vnd.trolltech.linguist) and .ogv video/ogg, and either went out as a
# video part's mime_type, outside the SDK's VideoContentMimeType set.
_VIDEO_EXTENSION_MIME_TYPES: dict[str, str] = {
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".mov": "video/mov",
    ".qt": "video/mov",
    ".avi": "video/avi",
    ".mkv": "video/webm",
    ".webm": "video/webm",
    ".wmv": "video/wmv",
    ".3gp": "video/3gpp",
    ".mpg": "video/mpeg",
    ".mpeg": "video/mpeg",
    ".flv": "video/x-flv",
}


def _guess_video_mime_from_uri(uri: str) -> str:
    """The MIME type for a video known only by its URI (e.g. a gs:// source).

    The bytes are never seen here, so the extension is all there is, labelled
    in the same SDK spelling the magic-byte path uses. Any other extension,
    or none, is labelled MP4, the container Omni renders and documents: the
    label is a hint, and one outside the SDK's set is not one it lists. The
    content is the service's to judge, since it reads the object itself; a
    file that is not a video is refused there, not here.
    """
    path = uri.split("?", 1)[0].rsplit("/", 1)[-1]
    suffix = path[path.rfind(".") :].lower() if "." in path else ""
    return _VIDEO_EXTENSION_MIME_TYPES.get(suffix, "video/mp4")


def _detect_video_mime(data: bytes) -> str:
    """Return the exact video MIME type from magic bytes, or raise.

    Distinguishes MP4 vs QuickTime/MOV via the ftyp brand and recognizes
    WebM/Matroska, MPEG program/video streams, and AVI, rather than labeling
    every input video as MP4. Spelled the SDK's way (_sdk_video_mime).
    """
    brand = _ftyp_brand(data)
    if brand is not None:
        if brand.startswith("qt"):
            return _sdk_video_mime("video/quicktime")
        if brand in _HEIF_BRANDS or brand in _HEIF_SEQUENCE_BRANDS:
            raise ValueError(
                "Input labeled as video but the data is a HEIC/HEIF image."
            )
        return "video/mp4"
    if data[:4] == b"\x1a\x45\xdf\xa3":
        return _sdk_video_mime("video/x-matroska")
    if data[:3] == b"\x00\x00\x01" and data[3:4] in (b"\xba", b"\xb3"):
        return "video/mpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"AVI ":
        return _sdk_video_mime("video/x-msvideo")
    raise ValueError(
        "Unrecognized video input format; could not detect a supported video "
        "MIME type from the data. Supported: MP4, QuickTime/MOV, WebM, MPEG, AVI."
    )


# ---------------------------------------------------------------------------
# Media parts and the role-tag prompt prefix
# ---------------------------------------------------------------------------


def _image_part(data: bytes) -> dict[str, Any]:
    """One flattened image part for ``input``."""
    return {
        "type": "image",
        "data": base64.b64encode(data).decode("ascii"),
        "mime_type": _detect_image_mime(data),
    }


def _inline_video_part(data: bytes) -> dict[str, Any]:
    """One flattened, inline-base64 video part for ``input``."""
    return {
        "type": "video",
        "data": base64.b64encode(data).decode("ascii"),
        "mime_type": _detect_video_mime(data),
    }


def _build_input_parts(
    prompt: str,
    image_parts: Sequence[dict[str, Any]],
    video_parts: Sequence[dict[str, Any]],
    *,
    media_before_text: bool = False,
) -> list[dict[str, Any]]:
    """Assemble the flattened ``input`` parts for interactions.create.

    The Interactions API takes flattened media parts ({type, data, mime_type})
    rather than generateContent's inlineData/fileData nesting.

    ``media_before_text`` follows each model's own reference: every 1.2 example
    puts the media ahead of the prompt, and the preview model's verified
    requests put the text first. Image order is load-bearing on 1.2 — a
    ``@Image2`` in the role prefix means the second image part — so callers
    must not reorder them afterwards.
    """
    text_part: dict[str, Any] = {"type": "text", "text": prompt}
    media: list[dict[str, Any]] = [*image_parts, *video_parts]
    if media_before_text:
        return [*media, text_part]
    return [text_part, *media]


# A prompt that already carries any role tag (or a "[# " declaration block) is
# the caller driving the binding themselves; adding a second, generated block
# would fight it.
_ROLE_TAG_PATTERN = re.compile(
    r"\[#\s|<(?:FIRST_FRAME|LAST_FRAME|PREVIOUS_VIDEO"
    r"|IMAGE_REF_\d+|VIDEO_REF_\d+|VIDEO_\d+)>"
)


def prompt_declares_media_roles(prompt: str) -> bool:
    """Whether the caller's prompt already binds media to roles itself."""
    return bool(_ROLE_TAG_PATTERN.search(prompt or ""))


def _build_media_role_prompt(
    prompt: str,
    *,
    has_first_frame: bool,
    has_last_frame: bool,
    reference_image_count: int,
    has_source_video: bool,
    reference_video_count: int,
    last_frame_is_first: bool = False,
) -> str:
    """Prefix ``prompt`` with the documented source/reference declarations.

    gemini-omni-1.2-flash infers each uploaded item's role from the prompt
    unless it is told, which is fine for one image and ambiguous for anything
    else: two images are either two subject references or a first/last frame
    pair, and a video is either the thing being edited or a likeness to copy.
    The reference documents an explicit syntax for exactly this —
    ``[# Sources <FIRST_FRAME>@Image1 <LAST_FRAME>@Image2]
    [# References <IMAGE_REF_0>@Image3 <VIDEO_REF_0>@Video1]`` plus a closing
    instruction — and the roles are already known here, so the declaration is
    built rather than left to a guess.

    ``@ImageN``/``@VideoN`` are 1-based positions within their own media type,
    matching the order ``_build_input_parts`` emits: first frame, last frame,
    then reference images; source video, then reference videos.
    ``last_frame_is_first`` declares a loop: the first frame's image is also
    the last frame, sent once, so both tags name it and the references number
    on from the second image.

    Returns the prompt unchanged when there is no ambiguity to resolve (a bare
    prompt, or one image whose role the model infers correctly on its own).
    """
    roles_in_play = sum(
        (
            bool(has_first_frame or has_last_frame),
            bool(reference_image_count),
            bool(has_source_video),
            bool(reference_video_count),
        )
    )
    if roles_in_play == 0:
        return prompt
    # One image the model will read as the opening shot anyway, or one video
    # that is obviously the thing being edited: the reference's own advice is
    # to prompt normally and only reach for tags when that fails.
    #
    # A single DECLARED reference is NOT such a case. "One reference image"
    # previously fell in here and went out as a bare prompt, so the one signal
    # separating a likeness from a starting frame — the <IMAGE_REF_0> tag and
    # its "should not be used as literal initial frames" instruction — was
    # dropped for exactly the request that needed it most.
    if (
        roles_in_play == 1
        and not has_last_frame
        and (has_first_frame or has_source_video)
    ):
        return prompt

    if has_last_frame and not has_first_frame:
        # validate_omni_request already refuses this, and the numbering below
        # depends on it: a lone last frame would still be sent as an image
        # part while no tag claimed it, so the references after it would all
        # name the wrong picture.
        raise ValueError(
            "A last frame requires a first frame: <LAST_FRAME> is documented "
            "as only usable together with <FIRST_FRAME>."
        )

    image_index = 0
    first_label = ""
    sources: list[str] = []
    references: list[str] = []
    instructions: list[str] = []

    if has_first_frame:
        image_index += 1
        first_label = f"Image{image_index}"
        sources.append(f"<FIRST_FRAME>@{first_label}")
    if has_last_frame and last_frame_is_first:
        sources.append(f"<LAST_FRAME>@{first_label}")
        instructions.append(
            f"Use {first_label} as both the first frame and the last frame."
        )
    elif has_last_frame:
        image_index += 1
        last_label = f"Image{image_index}"
        sources.append(f"<LAST_FRAME>@{last_label}")
        instructions.append(
            f"Use {first_label} as the first frame and {last_label} as the last frame."
        )
    elif has_first_frame:
        instructions.append(f"Use {first_label} as the starting frame.")

    for slot in range(reference_image_count):
        image_index += 1
        references.append(f"<IMAGE_REF_{slot}>@Image{image_index}")
    if reference_image_count:
        instructions.append(
            "Use the given image(s) as references for video generation. The "
            "images should not be used as literal initial frames."
        )

    video_index = 0
    if has_source_video:
        video_index += 1
        sources.append(f"<VIDEO_0>@Video{video_index}")
    for slot in range(reference_video_count):
        video_index += 1
        references.append(f"<VIDEO_REF_{slot}>@Video{video_index}")
    if reference_video_count:
        instructions.append(
            "Use the given video(s) as references. Do not use them as a "
            "source for video editing."
        )

    prefix = ""
    if sources:
        prefix += f"[# Sources {' '.join(sources)}] "
    if references:
        prefix += f"[# References {' '.join(references)}] "
    tail = (" " + " ".join(instructions)) if instructions else ""
    return f"{prefix}{prompt}{tail}"


# ---------------------------------------------------------------------------
# Request assembly
# ---------------------------------------------------------------------------


# Resolutions whose render the reference expects to exceed the 4 MB inline
# response limit: "For videos larger than 4MB (>720p when available), use
# delivery='uri' in response_format to avoid payload size limits."
_LARGE_OUTPUT_RESOLUTIONS = ("1080p", "4K")


def wants_uri_delivery(
    spec: OmniModelSpec,
    resolution: str | None,
    task: str | None,
    duration_seconds: float | None = None,
) -> bool:
    """Whether this request's output should be delivered by URI, not inline.

    The reference gives one rule — anything over 720p — and an extension earns
    the same treatment for the same reason: it renders the whole growing clip,
    up to 40s, which is far past 4 MB. Both are outputs the new model can
    produce and the preview model never could, so the inline default that has
    always been fine is no longer fine for them.

    Only meaningful on a model that can render above 720p at all. What a
    large output gets differs by backend: on the Gemini API a bare
    delivery='uri' returns a Google-hosted URI; Vertex AI has no bare URI
    delivery (it needs a gcs_uri beside it), so without the caller's
    output_gcs_uri the render is requested inline, and
    omni_inline_delivery_policy decides before it renders whether that is
    warned about or refused.
    """
    if not spec.supports_resolution:
        return False
    return (
        resolution in _LARGE_OUTPUT_RESOLUTIONS
        or task == _TASK_EXTEND
        or (
            bool(spec.max_inline_output_seconds)
            and duration_seconds is not None
            and duration_seconds > spec.max_inline_output_seconds
        )
    )


def _output_seconds_for_delivery(
    spec: OmniModelSpec,
    task_type: str | None,
    previous_interaction_id: str | None,
    *,
    sent_seconds: float | None,
    source_seconds: float | None,
    vertexai: bool = False,
) -> tuple[float | None, bool]:
    """The output length a delivery choice is judged by, and whether it is a bound.

    A duration the request sends is the length. An upscale keeps its
    source's length, taken at the longest source upscaling accepts when the
    source was never measured. Any other continuation (an edit, a turn on a
    previous interaction) renders a length the service picks, so it is
    judged at omni_continuation_upper_bound, as the cost and the Vertex
    delivery policy already are.
    """
    if sent_seconds is not None:
        return float(sent_seconds), False
    if task_type == _TASK_UPSCALE:
        if source_seconds:
            return float(source_seconds), False
        return float(omni_source_limit_seconds(spec, task=_TASK_UPSCALE)), True
    if _strips_output_spec(task_type, previous_interaction_id):
        return (
            omni_continuation_upper_bound(
                spec, task_type or _TASK_EDIT, source_seconds, vertexai=vertexai
            ),
            True,
        )
    return None, False


# The one switch in the large-inline-delivery policy below
# (omni_inline_delivery_policy).
#
# The 4 MB inline response limit comes from the Gemini API reference, where a
# bare delivery='uri' is the answer and hands back a Google-hosted file.
# Vertex AI has no bare URI delivery -- it needs a gcs_uri beside it -- and
# its "Upscale videos" page says of gcs_uri, "If not provided, video bytes
# are returned in the response", stating no size limit
# (https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/video/upscale-videos).
# So by default a large Vertex render with no output_gcs_uri is requested
# inline, with a warning before it renders. tests/test_live_omni_1_2.py's L3
# test settles it live: if a large render's bytes do not arrive inline, set
# this to True and every omni tool refuses such a request before rendering,
# dry run included, asking for output_gcs_uri.
OMNI_REFUSE_LARGE_INLINE_ON_VERTEX = False


def omni_uri_delivery_note(
    resolution: str | None,
    task: str | None,
    output_seconds: float | None,
    *,
    seconds_are_bound: bool = False,
) -> str:
    """The warning for a render the Gemini API delivered as a Google-hosted URI.

    URI delivery is chosen BEFORE the render, by wants_uri_delivery's rule for
    outputs that can pass the 4 MB inline limit, so the note says which rule
    applied rather than that this render did pass it: a live 9s 360p
    extension (2026-11-09) came back at 2.64 MB under a note that said it
    exceeded 4 MB.
    """
    return (
        f"This render was requested as a Google-hosted URI rather than inline, "
        f"because {_large_output(resolution, task, output_seconds, seconds_are_bound)} "
        "can exceed the 4 MB inline response limit, and it was downloaded here. On Vertex "
        "AI, pass output_gcs_uri to have the service write it straight to a "
        "bucket you control instead."
    )


def _large_output(
    resolution: str | None,
    task: str | None,
    output_seconds: float | None,
    seconds_are_bound: bool = False,
) -> str:
    """What makes an output large, as a noun phrase, for wants_uri_delivery's cases."""
    if resolution in _LARGE_OUTPUT_RESOLUTIONS:
        return f"a {resolution} render"
    if task == _TASK_EXTEND:
        return "an extension turn, which returns the whole assembled clip,"
    if output_seconds is None:
        return "this render"
    if seconds_are_bound:
        return f"a render that could run up to {output_seconds:g}s"
    return f"a {output_seconds:g}s render"


def _no_video_returned(
    interaction_id: str,
    spec: OmniModelSpec,
    *,
    inline_requested: bool,
    vertexai: bool,
    resolution: str | None,
    task: str | None,
    sent_seconds: float | None,
    source_seconds: float | None = None,
) -> str:
    """Why a completed interaction handed back no video, and what to do next.

    A response with neither bytes nor a URI used to fail with a bare "no
    inline video data and no file URI", which on the one path the delivery
    policy lets through -- a large render requested inline on Vertex AI --
    hid the likely cause and the fix. The interaction completed, so the
    render ran; the caller needs to know why it did not arrive.

    Size is judged on the length the tools' pre-flight judged it on
    (omni_inline_delivery_policy), not only on a sent duration: an edit and
    an upscale send none, so judged on that alone, the edit the pre-flight
    had warned about at its 15s bound, and an upscale of a 12s source, were
    never blamed on size here. ``source_seconds`` is the source's recorded
    or measured length, when the caller knows it.
    """
    message = (
        f"Interaction {interaction_id} completed, but its response carried no video"
    )
    if not inline_requested:
        return message + ": no inline bytes and no file URI."
    message += (
        ": the video was requested inline (no output_gcs_uri was sent) and no "
        "bytes came back."
    )
    output_seconds, seconds_are_bound = sent_seconds, False
    if task == _TASK_UPSCALE:
        # Upscaling keeps the source's length; unmeasured, it is the bound.
        output_seconds = source_seconds or omni_source_limit_seconds(
            spec, task=_TASK_UPSCALE
        )
        seconds_are_bound = not source_seconds
    elif task == _TASK_EDIT:
        output_seconds = omni_continuation_upper_bound(
            spec, _TASK_EDIT, source_seconds, vertexai=vertexai
        )
        seconds_are_bound = True
    if wants_uri_delivery(spec, resolution, task, output_seconds):
        large = _large_output(resolution, task, output_seconds, seconds_are_bound)
        message += (
            f" {large[0].upper()}{large[1:]} can exceed the 4 MB inline "
            "response limit, which is the likely reason."
        )
    return message + (
        " Pass output_gcs_uri (a prefix such as gs://bucket/renders/) to have Vertex AI write "
        "the render to Cloud Storage instead of the response."
        if vertexai
        else " On Vertex AI, output_gcs_uri (a prefix such as gs://bucket/renders/) delivers "
        "the render to Cloud Storage instead of the response."
    )


def omni_inline_delivery_policy(
    spec: OmniModelSpec,
    *,
    vertexai: bool,
    output_gcs_uri: str | None,
    resolution: str | None,
    task: str | None,
    output_seconds: float | None,
    seconds_are_bound: bool = False,
    chain_turns: int = 1,
    takes_output_gcs_uri: bool = True,
) -> str | None:
    """Decide, before anything renders, what a large inline output on Vertex gets.

    The whole policy, in one place, for generate_video_omni, edit_video,
    extend_video_omni and upscale_video_omni and their dry runs -- and for
    generate_video's draft and generate_clip's animatic, which render on omni
    too -- each calling it with what it knows before rendering. It speaks
    only about Vertex AI: the Gemini API delivers a large render as a
    Google-hosted URI instead (wants_uri_delivery, _build_create_kwargs),
    which needs no warning ahead of the render.

    Args:
        spec: The model that renders.
        vertexai: Whether the call runs on Vertex AI.
        output_gcs_uri: The caller's Cloud Storage destination, if any.
        resolution: The output resolution (None is the 720p default).
        task: The task, where it changes the size: an extension returns the
            whole assembled clip.
        output_seconds: The longest the output could run, as the call's own
            quote prices it -- a fresh render's clamped duration, an edit's
            upper bound, an upscale's source length (or its bound when the
            source is unmeasured). None when only resolution and task decide.
        seconds_are_bound: Whether output_seconds is a bound rather than a
            known length, so the message does not state it as fact.
        chain_turns: How many turns the call renders. An extension chain
            sends output_gcs_uri with its final turn only, so the turns
            before it come back inline even when one is given.
        takes_output_gcs_uri: Whether the calling tool can deliver to Cloud
            Storage at all. A draft and an animatic cannot, so the advice
            names what they can do instead.

    Returns:
        None when there is nothing to say: off Vertex AI, an output inside
        the limit, or every large turn bound for output_gcs_uri. Otherwise the
        warning a quote and a response both carry.

    Raises:
        ValueError: Instead of returning the warning, when
            OMNI_REFUSE_LARGE_INLINE_ON_VERTEX is set. Raised before anything
            renders, and identically on a dry run, since both call here first.
    """
    inline_turns = chain_turns - 1 if output_gcs_uri else chain_turns
    if not vertexai or inline_turns < 1:
        return None
    if not wants_uri_delivery(spec, resolution, task, output_seconds):
        return None
    large = _large_output(resolution, task, output_seconds, seconds_are_bound)
    # "can exceed", not "is past": whether a render is large is decided by
    # rule before it renders, and live renders the rule catches came back
    # under 4 MB (a 9s 360p extension was 2.67 MB, 2026-11-09).
    limit = "can exceed the 4 MB inline response limit the Gemini API reference gives"
    if output_gcs_uri:
        # Only an extension chain gets here: its final turn goes to the
        # bucket, and every turn before it comes back in the response.
        one_per_call = (
            "extend one turn per call (times=1) with output_gcs_uri, "
            "continuing from the interaction_id each call returns"
        )
        if OMNI_REFUSE_LARGE_INLINE_ON_VERTEX:
            raise ValueError(
                f"To deliver every turn to Cloud Storage, {one_per_call}: "
                "output_gcs_uri receives only a chain's final turn, so the "
                f"turns before it would come back inline, and {large} {limit}. "
                "This server delivers a Vertex AI render that large only to "
                "Cloud Storage. Nothing was rendered or billed."
            )
        return (
            "Every turn but the last of this extension comes back inline: "
            f"output_gcs_uri receives only the final turn, and {large} "
            f"{limit}. Vertex AI returns those bytes in the response and "
            "documents no size limit for that, so the chain runs; to deliver "
            f"every turn to Cloud Storage instead, {one_per_call}."
        )
    only_to_gcs = (
        "this server delivers a Vertex AI render that large only to Cloud Storage"
    )
    if takes_output_gcs_uri:
        if OMNI_REFUSE_LARGE_INLINE_ON_VERTEX:
            raise ValueError(
                f"Pass output_gcs_uri (a prefix such as gs://bucket/renders/): {large} {limit}, "
                f"and {only_to_gcs}. Nothing was rendered or billed."
            )
        return (
            f"{large[0].upper()}{large[1:]} {limit}, and with no output_gcs_uri "
            "it is requested inline: Vertex AI then returns the video bytes in "
            "the response, and documents no size limit for that. Pass "
            "output_gcs_uri (a prefix such as gs://bucket/renders/) to have Vertex write it to "
            "Cloud Storage instead, the safer delivery for an output this large."
        )
    instead = (
        "ask for 720p or below"
        if resolution in _LARGE_OUTPUT_RESOLUTIONS
        else f"keep it to {spec.max_inline_output_seconds:g}s or shorter"
    ) + (
        ", or render it with generate_video_omni, whose output_gcs_uri "
        "delivers to Cloud Storage"
    )
    if OMNI_REFUSE_LARGE_INLINE_ON_VERTEX:
        raise ValueError(
            f"{large[0].upper()}{large[1:]} {limit}, and {only_to_gcs}, which "
            f"this call cannot deliver to: {instead}. Nothing was rendered or "
            "billed."
        )
    return (
        f"{large[0].upper()}{large[1:]} {limit}, and it is requested inline: "
        "Vertex AI then returns the video bytes in the response, and documents "
        "no size limit for that. Cloud Storage is the safer delivery for an "
        f"output this large, and this call cannot deliver there: {instead}."
    )


def _select_task_type(
    *,
    previous_interaction_id: str | None,
    input_video_bytes: bytes | None,
    input_video_uri: str | None = None,
    inferred_image_count: int = 0,
    reference_image_count: int = 0,
    has_first_frame: bool = False,
    has_last_frame: bool = False,
    reference_video_count: int = 0,
    requested_task: str | None = None,
) -> str | None:
    """Deterministic task-type selection, mirroring the documented semantics.

    An explicit ``requested_task`` always wins — it is how a caller asks for
    ``extend`` rather than ``edit``, which are otherwise the same request
    shape. Failing that: editing (a prior interaction or an input video) wins;
    anything the caller DECLARED a reference is reference_to_video whatever
    its count; a first frame is image_to_video; and images whose role the
    model is left to infer keep the old rule — several are references, one is
    a first frame.

    The declared-vs-inferred split is the whole point of taking two counts.
    Merging them made a single ``reference_image_bytes_list=[x]`` indexed as
    "one image" and sent as ``image_to_video`` — the model told to open on a
    picture the caller had explicitly said was a likeness reference, under a
    task field this module's own docs call strict.

    Returns None when the role mix maps to no single documented task: a
    first/last frame pair is not ``image_to_video`` (the reference lists no
    task for interpolation), and mixing frames with references is not
    ``reference_to_video``. The reference's own advice is that the task field
    "adds strict constraints", so declining to invent one leaves the model to
    read the role tags — which say exactly what the mix is.
    """
    if requested_task:
        return requested_task
    # Footage to continue makes it an edit whatever images ride along: the
    # docstring always said so, but this check sat below the keyframe ones,
    # so a source video with a first and last frame was classed
    # image_to_video and sent the duration and aspect ratio an edit rejects,
    # while the tool layer quoted it as an edit that sends neither.
    if previous_interaction_id or input_video_bytes is not None or input_video_uri:
        return _TASK_EDIT
    declared_references = reference_image_count + reference_video_count
    if has_last_frame:
        # The GA release note names the task for this outright: "Interpolation
        # (first + last frame): Generate a video transitioning between two
        # images using the image_to_video task with up to 2 images." An
        # earlier reading of the task guide alone — which describes
        # interpolation without naming a task — concluded there was none and
        # sent no task at all, leaving the model to infer the mode from the
        # role tags.
        if not declared_references:
            return _TASK_IMAGE_TO_VIDEO
        return None
    if has_first_frame and declared_references:
        return None
    if declared_references and inferred_image_count:
        return None
    if declared_references:
        return _TASK_REFERENCE_TO_VIDEO
    if has_first_frame:
        return _TASK_IMAGE_TO_VIDEO
    if inferred_image_count > 1:
        return _TASK_REFERENCE_TO_VIDEO
    if inferred_image_count == 1:
        return _TASK_IMAGE_TO_VIDEO
    return _TASK_TEXT_TO_VIDEO


def _build_create_kwargs(
    *,
    model: str,
    prompt: str,
    image_parts: Sequence[dict[str, Any]],
    video_parts: Sequence[dict[str, Any]],
    previous_interaction_id: str | None,
    task_type: str | None,
    aspect_ratio: str,
    duration_seconds_int: int | None,
    resolution: str | None,
    output_gcs_uri: str | None,
    uri_delivery: bool = False,
    media_before_text: bool = False,
) -> dict[str, Any]:
    """Assemble the ``interactions.create`` request body.

    Pure and side-effect-free so it can be validated against the SDK's own
    request normalizer in tests. ``response_format`` is a LIST of one object;
    ``background`` is top-level (the Vertex REST example nests it in input[0],
    which is a doc artifact — the SDK's create schema has it top-level).

    Requests that continue existing footage carry FEWER fields (live-verified
    against the API, which 400s otherwise):
      * ``previous_interaction_id`` conflicts with ``video_config.task``
        ("previous_interaction_id is not allowed when video task is set"),
        so conversational turns send NO generation_config. The one exception
        is an Omni 1.2 upscale, whose documented stateful request carries
        both (see the comment at the generation_config below);
      * edit tasks reject ``duration`` in response_format ("Duration cannot
        be set in response format for edit task") — duration and aspect
        ratio cannot be sent for an edit-type request, so neither is. What
        the service then renders is undocumented and is NOT the source's
        length — a measured 3s source came back at 10.01s;
      * ``extend`` sends no aspect ratio (the source fixes it). Before 1.2 it
        sent no duration either, because the continuation's length was the
        model's fixed step; Omni 1.2 documents a per-turn duration of 3-10s,
        so a 1.2 extension sends the one the caller chose (_sends_duration);
      * ``upscale`` sends neither: the documented requests carry only the
        target ``resolution``, and upscaling keeps the source's length and
        framing. It sends no text either (the service 400s it): the input is
        the source clip, or empty on a conversational turn.

    ``resolution`` rides in response_format on every task that sends one, and
    only when the caller asked for a specific one — an omitted field is the
    documented 720p default, which is what the reference's own examples send.
    """
    response_format_item: dict[str, Any] = {"type": "video"}
    if not _strips_output_spec(task_type, previous_interaction_id):
        response_format_item["aspect_ratio"] = aspect_ratio
    if duration_seconds_int is not None and _sends_duration(
        model, task_type, previous_interaction_id
    ):
        response_format_item["duration"] = f"{duration_seconds_int}s"
    if resolution is not None:
        response_format_item["resolution"] = _API_RESOLUTIONS[resolution]
    if output_gcs_uri:
        response_format_item["delivery"] = "uri"
        response_format_item["gcs_uri"] = output_gcs_uri
    elif uri_delivery:
        # A Google-hosted URI instead of inline base64, for a render the
        # reference expects to exceed the 4 MB inline response limit. Only
        # ever sent WITHOUT a gcs_uri: on Vertex the API requires one
        # alongside delivery='uri', and that is the output_gcs_uri branch
        # above. The interaction then carries a URI rather than bytes, which
        # _resolve_video_bytes waits for and downloads.
        response_format_item["delivery"] = "uri"

    parts = _build_input_parts(
        prompt, image_parts, video_parts, media_before_text=media_before_text
    )
    if task_type == _TASK_UPSCALE:
        # An upscale carries no text: Vertex 400s "Text input is not supported
        # for upscale task." (live, 2026-11-09), even though the "Upscale
        # videos" page's stateful sample sends "Upsample video.". The input is
        # the source clip, or, on a conversational turn, empty: an empty input
        # beside previous_interaction_id was accepted live and returned a 720p
        # upscale of the prior 3.03s 360p clip at its own 3.03s.
        parts = list(video_parts)
    create_kwargs: dict[str, Any] = {
        "model": model,
        "input": parts,
        "background": True,
        "response_format": [response_format_item],
    }
    if previous_interaction_id is not None:
        # Conversational turn: the server holds the video context; sending a
        # task alongside previous_interaction_id is rejected.
        create_kwargs["previous_interaction_id"] = previous_interaction_id
    # An upscale declares its task even on a conversational turn. Vertex's
    # "Upscale videos" page
    # (https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/video/upscale-videos)
    # documents exactly this pair in its "Synchronous: stateful" request --
    # previous_interaction_id beside generation_config.video_config.task
    # "upscale" -- and says "You must set generation_config.video_config.task
    # to 'upscale'" because "If you omit task: 'upscale', the API interprets
    # the request as a video editing ('edit') operation". The live rejection
    # recorded above was measured on the earlier models' edit and extend
    # turns; for a 1.2 upscale, leaving the task off is what is documented to
    # go wrong: the request is read as an edit.
    if task_type is not None and (
        previous_interaction_id is None or task_type == _TASK_UPSCALE
    ):
        create_kwargs["generation_config"] = {"video_config": {"task": task_type}}
    return create_kwargs


def _field(obj: Any, name: str) -> Any:
    """Read a field from an SDK object or a plain dict interchangeably."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


# Codes a failed interaction carries when the service, not the request, ran
# out of room.
_CAPACITY_ERROR_CODES = frozenset({"too_many_requests", "resource_exhausted", "429"})


def _interaction_failure_detail(interaction: Any) -> str | None:
    """Why a terminal interaction failed, from ``error`` or an ``errors`` list.

    Vertex AI reports a failed render's reason in an ``errors`` LIST of
    {code, message}, not a singular ``error``: a live extension turn that
    failed on capacity (2026-11-09) carried only
    ``errors: [{code: "too_many_requests", message: "Resource exhausted.
    Please try again later. ..."}]``, which reading ``error`` alone reported
    as "terminal or unrecognized status". Both shapes are read.
    """
    entries: list[Any] = []
    single = _field(interaction, "error")
    if single is not None:
        entries.append(single)
    entries.extend(_field(interaction, "errors") or [])
    reasons: list[str] = []
    capacity = False
    for entry in entries:
        message = entry if isinstance(entry, str) else _field(entry, "message")
        code = None if isinstance(entry, str) else _field(entry, "code")
        if code is not None and str(code).lower() in _CAPACITY_ERROR_CODES:
            capacity = True
        if message:
            reasons.append(f"{message} (code {code})" if code else str(message))
        elif code:
            reasons.append(f"code {code}")
    if not reasons:
        return None
    detail = "; ".join(reasons)
    if capacity:
        detail += (
            ". This is the service's capacity or the project's quota, not this "
            "request: it can be sent again unchanged once capacity frees up."
        )
    return detail


def _extract_video_payload(interaction: Any) -> tuple[str | bytes | None, str | None]:
    """Return (inline_data, uri) for the interaction's video output.

    Prefers the SDK's convenience ``output_video`` when present, then scans
    ``steps[].content[]`` for a video part (the documented REST shape).
    """
    output_video = _field(interaction, "output_video")
    if output_video is not None:
        data = _field(output_video, "data")
        uri = _field(output_video, "uri") or _field(output_video, "name")
        if data or uri:
            return data, uri

    for step in _field(interaction, "steps") or []:
        for part in _field(step, "content") or []:
            mime = _field(part, "mime_type") or _field(part, "mimeType") or ""
            is_video = _field(part, "type") == "video" or mime.startswith("video/")
            if not is_video:
                continue
            data = _field(part, "data")
            uri = (
                _field(part, "uri")
                or _field(part, "file_uri")
                or _field(part, "fileUri")
            )
            if data or uri:
                return data, uri
    return None, None


# A delivered video's URI is a full download URL
# (…/v1beta/files/<id>:download?alt=media), but files.get wants the resource
# NAME (files/<id>). The reference's Python snippet takes the URI's last path
# segment, which on that URL yields "<id>:download?alt=media"; its JavaScript
# snippet uses a regex instead, which is the form that actually works. Both a
# bare name and a full URL go in, one resource name comes out.
_FILE_RESOURCE_PATTERN = re.compile(r"files/([A-Za-z0-9_-]+)")


def _delivered_file_name(uri: str) -> str:
    """Resolve a delivered-video URI to a ``files/<id>`` resource name."""
    match = _FILE_RESOURCE_PATTERN.search(uri)
    if match:
        return f"files/{match.group(1)}"
    return uri


def _file_state(file_obj: Any) -> str:
    """Read a File's state as a bare string, object-or-enum-or-dict alike."""
    state = _field(file_obj, "state")
    return str(_field(state, "name") or state)


async def _wait_for_file_active(
    client: genai.Client,
    name: str,
    label: str,
    run_within_deadline: Callable[..., Awaitable[Any]],
    deadline_expired: Callable[[], bool],
    state: str | None = None,
    file_obj: Any = None,
) -> Any:
    """Poll a Files API resource until it leaves PROCESSING, or raise.

    Both directions need this. An UPLOAD lands in PROCESSING and is unusable
    as an input until it is ACTIVE. A DELIVERED render is the same the other
    way round — the reference's own retrieval example polls until ACTIVE
    before downloading — and downloading early is how you get an empty or
    truncated file back from a render that was already billed.

    Bounded by the caller's deadline for the same reason every other call in
    this module is: a resource that never becomes ACTIVE must not hang the
    request forever.
    """
    if state is None:
        file_obj = await run_within_deadline(client.files.get, name=name)
        state = _file_state(file_obj)
    while state == "PROCESSING":
        if deadline_expired():
            raise TimeoutError(f"Timed out waiting for {label} to become ACTIVE.")
        await asyncio.sleep(_FILE_POLL_INTERVAL)
        file_obj = await run_within_deadline(client.files.get, name=name)
        state = _file_state(file_obj)
    if state == "FAILED":
        raise ValueError(f"The Files API reported {label} as FAILED.")
    return file_obj


async def _resolve_video_bytes(
    client: genai.Client,
    inline_data: str | bytes | None,
    uri: str | None,
    log_callback: LogCallback | None,
    run_within_deadline: Callable[..., Awaitable[Any]],
    deadline_expired: Callable[[], bool],
    max_bytes: int = _MAX_DELIVERED_VIDEO_BYTES,
) -> bytes:
    """Materialize mp4 bytes from an inline payload or a Files API uri.

    The download runs through the caller's deadline runner like every other
    call in this module: an interaction that completes at t=590s of a 600s
    budget would otherwise hang here forever on an untimed transfer.
    """
    if inline_data:
        if isinstance(inline_data, str):
            return base64.b64decode(inline_data)
        return bytes(inline_data)

    if not uri:
        raise ValueError("Interaction returned no inline video data and no file URI.")

    if log_callback:
        await log_callback(f"Downloading delivered video: {uri}")

    # files.get takes the resource NAME, not the download URL the interaction
    # hands back, and the render may still be PROCESSING when the interaction
    # completes — the reference's own retrieval example polls for ACTIVE
    # before downloading.
    file_obj = await _wait_for_file_active(
        client,
        _delivered_file_name(uri),
        "the delivered video",
        run_within_deadline,
        deadline_expired,
    )

    # Reject an oversize clip before buffering when the resource advertises its
    # size, so the cap can hold without first allocating the whole body.
    declared_size = _field(file_obj, "size_bytes")
    if declared_size is not None and declared_size > max_bytes:
        raise ValueError(
            f"Delivered video size {declared_size} exceeds cap "
            f"{max_bytes}: {uri}. Deliver to Cloud Storage instead "
            "(output_gcs_uri, Vertex AI only), which is not buffered here."
        )

    data = await run_within_deadline(client.files.download, file=file_obj)
    # files.download returns bytes, so a zero-length body is b"" not None: the
    # None-only guard let an empty download write a 0-byte .mp4 and report
    # success. `not data` catches both None and b"".
    if not data:
        raise ValueError(f"Downloaded video file was empty: {uri}")
    # Hard enforcement of the cap even when no size was advertised (or it lied);
    # every other fetch in the server bounds its body the same way.
    if len(data) > max_bytes:
        raise ValueError(
            f"Downloaded video ({len(data)} bytes) exceeds cap "
            f"{max_bytes}: {uri}. Deliver to Cloud Storage instead "
            "(output_gcs_uri, Vertex AI only), which is not buffered here."
        )
    return bytes(data)


async def _upload_video_via_files_api(
    client: genai.Client,
    data: bytes,
    label: str,
    log_callback: LogCallback | None,
    run_within_deadline: Callable[..., Awaitable[Any]],
    deadline_expired: Callable[[], bool],
) -> str:
    """Upload video bytes through the Files API and return the file URI.

    The documented path for anything sizeable ("since videos can be quite big
    we recommend using the File API"), and the only path for a clip whose
    base64 form would blow the request body. An upload lands in PROCESSING and
    is unusable until it reaches ACTIVE, so this waits — the same bounded poll
    the interaction itself gets, because an upload that never becomes ACTIVE
    would otherwise hang the render.

    Vertex has no Files API; callers route large-video requests to a Gemini API
    client before getting here, and a Vertex client surfaces the SDK's own
    error rather than a silent fallback to an oversize inline body.
    """
    mime_type = _detect_video_mime(data)
    if log_callback:
        await log_callback(
            f"Uploading {label} ({len(data)} bytes, {mime_type}) via the Files API"
        )
    file_obj = await run_within_deadline(
        client.files.upload,
        file=io.BytesIO(data),
        config={"mime_type": mime_type},
    )

    name = _field(file_obj, "name")
    file_obj = await _wait_for_file_active(
        client,
        name,
        f"the uploaded {label}",
        run_within_deadline,
        deadline_expired,
        state=_file_state(file_obj),
        file_obj=file_obj,
    )

    uri = _field(file_obj, "uri") or name
    if not uri:
        raise ValueError(f"Files API upload of {label} returned no URI.")
    return str(uri)


async def _prepare_video_part(
    client: genai.Client,
    data: bytes,
    label: str,
    *,
    spec: OmniModelSpec,
    log_callback: LogCallback | None,
    run_within_deadline: Callable[..., Awaitable[Any]],
    deadline_expired: Callable[[], bool],
    warnings: list[str],
) -> dict[str, Any]:
    """Return the input part for one input video, uploading it when large.

    Small clips ride inline exactly as they always have. Anything over
    ``_MAX_INLINE_VIDEO_BYTES`` goes through the Files API and is referenced
    by URI as a ``video`` part, the shape 1.2's reference documents; on the
    preview model, which documents no Files API input at all, the request is
    left inline and the caller is told why it may be refused, rather than
    quietly sending an unverified field.
    """
    if len(data) <= _MAX_INLINE_VIDEO_BYTES:
        return _inline_video_part(data)
    if bool(getattr(getattr(client, "_api_client", None), "vertexai", False)):
        if len(data) > OMNI_VERTEX_INLINE_VIDEO_BYTES:
            raise ValueError(
                "Vertex inline video inputs must be at most 20 MB. "
                "Save the video in Cloud Storage and pass its gs:// URI."
            )
        return _inline_video_part(data)
    if not spec.supports_role_tags:
        warnings.append(
            f"{label} is {len(data)} bytes, over the {_MAX_INLINE_VIDEO_BYTES}-byte "
            f"inline threshold, and {spec.model} documents no Files API input "
            "shape — it is sent inline and the API may reject the request. "
            # True of the default off Vertex, which is the only way to get
            # here: a Vertex client returned above.
            f"Use {DEFAULT_OMNI_MODEL}, which uploads a clip this size through "
            "the Files API instead, or a shorter clip."
        )
        return _inline_video_part(data)
    uri = await _upload_video_via_files_api(
        client, data, label, log_callback, run_within_deadline, deadline_expired
    )
    return {"type": "video", "uri": uri, "mime_type": _detect_video_mime(data)}


def _quota_retry_wait(exc: BaseException, retries_so_far: int) -> float | None:
    """Seconds to wait before resending a create refused with 429, or None.

    None for any other failure, and once the resends are spent.
    """
    if http_status(exc) != 429 or retries_so_far >= len(_QUOTA_RETRY_WAITS_SECONDS):
        return None
    asked = retry_after_seconds(exc)
    wait = _QUOTA_RETRY_WAITS_SECONDS[retries_so_far] if asked is None else asked
    return min(wait, _QUOTA_RETRY_MAX_WAIT_SECONDS)


def _poll_error_is_transient(exc: BaseException) -> bool:
    """Whether a failed status read is worth reading again.

    A 429, a 5xx, or a connection that failed: a read changes nothing, so
    reading again is safe whatever happened to the first one.
    """
    status = http_status(exc)
    if status is not None:
        return status == 429 or status >= 500
    return _is_connection_error(exc)


# By name, as google-genai's own retry code tells them apart: its
# Interactions client raises APIConnectionError over the httpx error, and
# httpx is its dependency, not this server's.
_CONNECTION_ERROR_NAMES = frozenset({"APIConnectionError", "TransportError"})


def _is_connection_error(exc: BaseException) -> bool:
    """Whether a call failed on the connection rather than with a response."""
    for error in (exc, exc.__cause__):
        if error is not None and _CONNECTION_ERROR_NAMES & {
            base.__name__ for base in type(error).__mro__
        }:
            return True
    return False


# The statuses google-genai may resend an Interactions call on by itself.
# Its default adds 409 and every 5xx, and it may resend after a connection
# failure; a create that failed that way may already have started a render,
# which the resend bills a second time. A 408 or a 429 started nothing.
_SDK_RESEND_STATUSES = ("408", "429")


def _interactions(client: Any) -> Any:
    """``client.interactions``, resending only refusals that started nothing.

    google-genai's Interactions resource resends a 408, 409, 429 or 5xx up to
    3 more times, 0.5s, 1s and 2s apart. Measured against a stub, a create
    answered 503 went out 4 times. Narrowed here on this client's
    Interactions resource alone: the models.* calls the Veo and image tools
    make have their own retry settings, which default to none. Status reads
    lose the SDK's 5xx resends with it, and _poll_error_is_transient reads
    again instead.
    """
    resource = client.interactions
    retry = getattr(getattr(resource, "sdk_configuration", None), "retry_config", None)
    if retry is not None and hasattr(retry, "status_codes_override"):
        retry.status_codes_override = list(_SDK_RESEND_STATUSES)
        retry.retry_connection_errors = False
    return resource


def _times(count: int) -> str:
    return "once" if count == 1 else "twice" if count == 2 else f"{count} times"


@dataclass
class OmniSubmission:
    """How far an omni render got, for an error that interrupts it.

    The service renders an interaction once it exists and bills it whether or
    not this process is still waiting, so a failure must say whether one was
    created. The caller passes an instance in; the impl marks it just before
    the create call, again when create returns an id, and again when the
    interaction completes -- after which it has rendered and billed, whatever
    goes wrong fetching or saving the video.
    """

    create_sent: bool = False
    interaction_id: str | None = None
    completed: bool = False
    # The status an interaction ended on without completing ("failed", say):
    # it will not complete later, so a note saying it may would mislead.
    final_status: str | None = None
    # How many times a create refused with 429 was sent again. A caller that
    # retries on 429 itself (the director) reads it so the waits do not stack.
    quota_retries: int = 0

    def state(self, exc: BaseException) -> Literal["submitted", "unknown", "none"]:
        """``submitted`` once an interaction exists; ``unknown`` when the
        create call failed in a way that leaves open whether the service
        received it (a timeout of either SDK's kind, a dropped connection, a
        5xx); ``none`` when it was not sent or was refused outright."""
        if self.interaction_id:
            return "submitted"
        if self.create_sent and not refused_before_work(exc):
            return "unknown"
        return "none"


def _omni_1_2_extension_note(
    spec: OmniModelSpec, *, sent_duration: int | None, continues_interaction: bool
) -> str:
    """What an Omni 1.2 extension request actually sent, and what follows."""
    note = (
        f"This extension sends duration={sent_duration}s, the per-turn length "
        f"{spec.model} documents as {_MIN_DURATION} to "
        f"{spec.extension_step_seconds}s"
        if sent_duration is not None
        else "This extension sends no duration, so the service appends its "
        f"documented default of {spec.extension_step_seconds}s"
    ) + "; aspect_ratio is not sent and follows the source."
    if continues_interaction:
        note += (
            " This turn continues a previous interaction, so it cannot declare "
            "its task and the service infers extension from the prompt"
            + (
                "; if it reads the turn as an edit it refuses the duration "
                "before rendering anything."
                if sent_duration is not None
                else "; a prompt it reads as an edit is rendered as one."
            )
        )
    return note + (
        " The response reports the length measured from the rendered file, "
        "which includes the footage before the extension. A render delivered "
        "to Cloud Storage is downloaded to measure it; only one that cannot be "
        "read reports its bound, as duration_upper_bound_seconds."
    )


# The service's wording for a duration on a request it treats as an edit
# (live-verified; quoted in _build_create_kwargs).
_EDIT_REJECTS_DURATION = "Duration cannot be set in response format for edit task"


def _extension_read_as_edit(
    exc: BaseException, previous_interaction_id: str | None
) -> ValueError:
    """Advice for a conversational extension the service took for an edit.

    That turn cannot name its task, so the service infers one from the
    prompt, and the duration Omni 1.2 documents for extension is what an
    edit refuses. The raw 400 names neither the cause nor the fix, and it
    arrives mid-chain after earlier turns rendered and billed. It is refused
    before any work, so this turn costs nothing and the chain can resume
    from the same id.
    """
    return ValueError(
        f"The service read this continuation of interaction "
        f"{previous_interaction_id} as an edit, not an extension, and an edit "
        f"cannot carry a duration ({str(exc)[:200]}). Nothing was rendered or "
        "billed for this turn. A turn that continues an interaction cannot "
        "declare its task, so the prompt is what asks for an extension: say "
        "so explicitly (for example 'Extend the video: ...') and continue "
        f"from {previous_interaction_id} again."
    )


async def generate_video_omni(
    client: genai.Client,
    prompt: str,
    videos_dir: Path,
    *,
    model: str = DEFAULT_OMNI_MODEL,
    image_bytes_list: list[bytes] | None = None,
    first_frame_bytes: bytes | None = None,
    last_frame_bytes: bytes | None = None,
    reference_image_bytes_list: list[bytes] | None = None,
    input_video_bytes: bytes | None = None,
    input_video_uri: str | None = None,
    reference_video_bytes_list: list[bytes] | None = None,
    previous_interaction_id: str | None = None,
    task: str | None = None,
    aspect_ratio: str = "16:9",
    duration_seconds: float | None = 6.0,
    resolution: str | None = None,
    output_gcs_uri: str | None = None,
    allow_uri_delivery: bool = False,
    timeout_seconds: int = OMNI_DEFAULT_TIMEOUT_SECONDS,
    log_callback: LogCallback | None = None,
    submission: OmniSubmission | None = None,
    source_duration_seconds: float | None = None,
) -> dict[str, Any]:
    """Generate, edit or extend a video with one of the Omni models.

    Args:
        client: Google GenAI client (for Vertex, a global-location client).
        prompt: Text description of the video, the edit to apply, or the
            continuation to append. Tags the caller writes themselves
            (``<FIRST_FRAME>``, ``<IMAGE_REF_0>``, ``[# Sources ...]``) are
            left alone; otherwise the role declarations are generated from the
            media arguments below.
        videos_dir: Directory to save the generated video.
        model: ``DEFAULT_OMNI_MODEL`` (``gemini-omni-1.2-flash``, the
            default) or the switched-off ``gemini-omni-flash-preview``. The
            retired ``gemini-omni-1.1-flash`` runs on 1.2 with a warning.
            Everything below marked "1.2" says so, and is refused rather
            than dropped on the preview model.
        image_bytes_list: Optional input images whose role the model infers.
            One image is treated as a first frame (image_to_video); several
            are treated as references (reference_to_video). Mutually exclusive
            with the explicit-role image arguments.
        first_frame_bytes: 1.2. Image to start the video on.
        last_frame_bytes: 1.2. Image to end the video on; requires
            ``first_frame_bytes`` (documented as ``<LAST_FRAME>`` must be used
            with ``<FIRST_FRAME>``). The pair renders an interpolation between
            them, and using the same image for both loops the clip.
        reference_image_bytes_list: 1.2. Subject/style references,
            bound to ``<IMAGE_REF_0>``, ``<IMAGE_REF_1>``, … in order.
        input_video_bytes: Optional source video (raw bytes) to edit or
            extend. Sent inline, or, off Vertex AI, uploaded through the Files
            API on 1.2 when it is too large to inline.
        reference_video_bytes_list: 1.2. Up to 3 clips of up to 3s each,
            bound to ``<VIDEO_REF_0>``, … Any audio in them is ignored.
        previous_interaction_id: Optional id of a prior interaction to
            continue a multi-turn edit or extension (server holds the video
            context).
        task: Optional explicit task
            (text_to_video / image_to_video / reference_to_video / edit, plus
            extend and upscale on 1.2). Needed to ask for
            ``extend`` rather than ``edit``, since both are "a prompt plus a
            source video". Not sent alongside ``previous_interaction_id``,
            which the API rejects, except for a 1.2 upscale, whose documented
            request carries both.
        aspect_ratio: "16:9" (default) or "9:16". Not sent when continuing
            existing footage, whose source already fixes it.
        duration_seconds: Desired clip length, clamped to the model's
            supported range ([3, 10]s on the preview, [3, 15]s on 1.2) and
            sent as "Ns" in response_format; None omits it and lets
            the service choose. Zero snaps to the 3s minimum; negative or
            non-finite values are refused. Never sent for an edit or an
            upscale, and then not checked at all; on an extension it is sent
            as the per-turn length (3-10s).
        resolution: 1.2. "360p", "720p", "1080p" or "4K" (1080p and 4K
            are upscaled from the base render). Omitted means the documented
            720p default.
        output_gcs_uri: Optional gs:// destination. When set, the video is
            delivered to Cloud Storage (delivery='uri') and video_url is the
            gs:// URI; otherwise the bytes come back inline and are written
            locally as a file:// URL.
        allow_uri_delivery: Permit ``delivery='uri'`` without a gcs_uri, which
            is a Gemini-Developer-API-only shape (Vertex requires the gcs_uri
            alongside it). Set by the caller that knows the backend; when it
            is on, a render the reference expects to exceed the 4 MB inline
            limit — above 720p, or an extension — is fetched from a
            Google-hosted URI instead of arriving as base64.
        timeout_seconds: Overall deadline covering any input upload, the
            create call and the background polling loop.
        log_callback: Async callback for progress logging.
        source_duration_seconds: The source's recorded or measured length,
            when the caller knows it. Nothing is sent from it; it sizes an
            edit's or an upscale's output when explaining a response that
            came back without a video.

    Returns:
        Dict with message, video_url (file:// or gs://), interaction_id, model,
        task, duration_seconds (clamped int), aspect_ratio, resolution, the
        requested_* originals, effective_prompt (when role declarations were
        generated), and warnings (only when non-empty). duration_seconds and
        aspect_ratio are both None for an edit or an extension — neither was
        sent, so neither can be reported as fact.
    """
    # Non-fatal warnings surfaced back to the caller.
    warnings: list[str] = []
    spec = omni_spec(model)

    refusal = omni_model_refusal(spec.model)
    if refusal:
        raise ValueError(refusal)

    reroute = omni_model_reroute_warning(model)
    if reroute:
        warnings.append(reroute)
    if spec.model == OMNI_PREVIEW_MODEL:
        # A caller who reads this has time to move; one who reads nothing gets
        # a dead endpoint. Said on every call rather than once at startup,
        # because the pin that matters is the one in the call.
        warnings.append(
            f"{OMNI_PREVIEW_MODEL} is deprecated and its endpoint is switched "
            f"off on {OMNI_PREVIEW_SUNSET}. Pass model='{DEFAULT_OMNI_MODEL}' "
            "— it is generally available on both the Gemini Developer API and "
            "Vertex AI at a lower published 720p rate, and adds resolution "
            "control, clips up to 15s, video extension and first/last-frame "
            "interpolation."
        )

    # Aspect ratio is a hard error rather than a silent coercion, matching
    # the style in src/video.py.
    if aspect_ratio not in _SUPPORTED_ASPECT_RATIOS:
        raise ValueError(
            f"Unsupported aspect_ratio '{aspect_ratio}'. "
            "Supported values are '16:9' and '9:16'."
        )

    reference_images = list(reference_image_bytes_list or [])
    reference_videos = list(reference_video_bytes_list or [])
    legacy_images = list(image_bytes_list or [])

    # Selected before validation because whether a duration is SENT depends
    # on it. Pure: an unknown task passes through here and is refused just
    # below.
    task_type = _select_task_type(
        previous_interaction_id=previous_interaction_id,
        input_video_bytes=input_video_bytes,
        input_video_uri=input_video_uri,
        inferred_image_count=len(legacy_images),
        reference_image_count=len(reference_images),
        has_first_frame=first_frame_bytes is not None,
        has_last_frame=last_frame_bytes is not None,
        reference_video_count=len(reference_videos),
        requested_task=task,
    )
    # An edit never sends a duration, so its duration_seconds is not checked
    # against anything: refusing the render over a value the service would
    # never see made edit_video's dry run, which knows the edit ignores it,
    # quote a call the real run then refused.
    sends_duration = _sends_duration(spec.model, task_type, previous_interaction_id)

    validate_omni_request(
        spec.model,
        resolution=resolution,
        task=task,
        has_first_frame=first_frame_bytes is not None,
        has_last_frame=last_frame_bytes is not None,
        reference_image_count=len(reference_images),
        reference_video_count=len(reference_videos),
        inferred_image_count=len(legacy_images),
        duration_seconds=duration_seconds if sends_duration else None,
        is_continuation=bool(
            previous_interaction_id or input_video_bytes is not None or input_video_uri
        ),
        vertexai=bool(getattr(getattr(client, "_api_client", None), "vertexai", False)),
    )

    normalized_resolution = normalize_omni_resolution(resolution)
    if resolution is not None:
        if not RESOLUTION_REACHES_THE_WIRE:
            # Refusing beats rendering 720p while quoting 4K: the field is
            # dropped during serialization, so nothing downstream could tell.
            raise RuntimeError(
                "The installed google-genai is too old to send `resolution`: it "
                "drops the field during serialization, so the render would come "
                "back at the 720p default while this server reported "
                f"'{normalized_resolution}'. Upgrade to google-genai>=2.20.0."
            )

    is_continuation = (
        task_type in _CONTINUATION_TASKS or previous_interaction_id is not None
    )
    # Identical to `is_continuation` today, and kept as its own name because
    # the two ask different questions and could diverge: one is "is this turn
    # a continuation", the other is "must this request omit aspect_ratio and
    # duration". This comment used to claim the second was NARROWER -- that an
    # extend with an uploaded source still sends both and can report them as
    # fact -- which _strips_output_spec's own docstring contradicts: an
    # uploaded extend 400s on aspect ratio, and duration is stripped there
    # too. Two of the response fields below are None on that path, so the
    # claim was falsified by the code three lines away.
    withholds_output_spec = _strips_output_spec(task_type, previous_interaction_id)

    if task_type == _TASK_EXTEND and previous_interaction_id is None:
        if input_video_bytes is None and not input_video_uri:
            raise ValueError(
                "task='extend' needs something to extend: pass "
                "input_video_bytes (an uploaded clip) or "
                "previous_interaction_id (a clip this server generated)."
            )
    if task_type == _TASK_UPSCALE:
        if (
            not previous_interaction_id
            and not input_video_uri
            and input_video_bytes is None
        ):
            raise ValueError(
                "Upscaling needs a source video or previous_interaction_id."
            )
        if normalized_resolution is None:
            raise ValueError("Upscaling requires an explicit target resolution.")

    # Clamp duration into the model's documented [3, max_duration_seconds]
    # range (an extension turn into its 3-10s step) and send it as
    # "Ns". A duration that is not sent is still reported back as
    # requested_duration_seconds, but its clamp is not announced: a warning
    # about a value that never reaches the wire describes nothing.
    clamped_duration: int | None = None
    if duration_seconds is not None and math.isfinite(duration_seconds):
        rounded = round(duration_seconds)
        clamped_duration = min(spec.max_duration_seconds, max(_MIN_DURATION, rounded))
        if task_type == _TASK_EXTEND:
            clamped_duration = min(clamped_duration, spec.extension_step_seconds)
        clamp_warning = omni_duration_clamp_warning(
            spec, duration_seconds, extension=task_type == _TASK_EXTEND
        )
        if sends_duration and clamp_warning:
            warnings.append(clamp_warning)

    if task_type == _TASK_EXTEND:
        # Says what THIS request sent. It used to read "Extension duration is
        # sent" on every 1.2 extension, including those that sent none.
        warnings.append(
            _omni_1_2_extension_note(
                spec,
                sent_duration=clamped_duration if sends_duration else None,
                continues_interaction=previous_interaction_id is not None,
            )
        )
    elif (
        _strips_output_spec(task_type, previous_interaction_id)
        and task_type != _TASK_UPSCALE
    ):
        # Whichever of the two reasons applies (see _strips_output_spec), the
        # consequence for the caller is the same: the length is the service's
        # and is NOT predictable from the request or from the source — a
        # measured 3s source edited with duration_seconds=4 rendered 10.01s.
        # One message for one behaviour. Two were emitted before, naming
        # different reasons for the same pair of suppressed fields, which read
        # as though one of them were a guess. Both reasons are real and both
        # are stated, with the one that applies here named first.
        why = (
            "this turn continues a previous interaction, so it cannot declare "
            "its task and the service infers one"
            if previous_interaction_id is not None
            # Names the task that actually applies. This said "edit" flatly,
            # so an uploaded extend was told the wrong reason for its own
            # suppressed fields.
            else f"this is an {task_type} task"
        )
        warnings.append(
            "This request sends neither duration_seconds nor aspect_ratio: "
            f"{why}, and a continuation rejects both — live-verified as "
            '"Duration cannot be set in response format for edit task" and '
            '"Aspect ratio cannot be set in response format for extend task". '
            "The rendered length is therefore the service's choice. The "
            "response reports the duration measured from the rendered file, "
            "including one delivered to Cloud Storage, which is downloaded to "
            "measure it; only a render that cannot be read reports its bound, "
            "as duration_upper_bound_seconds."
        )

    effective_prompt = prompt
    declares_roles = spec.supports_role_tags and not prompt_declares_media_roles(prompt)
    # A loop: one picture opens and closes the clip. It goes on the wire once,
    # with both role tags naming it, instead of twice in base64. Only when the
    # declaration is built here: a caller's own tags number the images
    # themselves, and dropping one would shift every @ImageN after it.
    loop_frame = (
        declares_roles
        and first_frame_bytes is not None
        and first_frame_bytes == last_frame_bytes
    )
    if declares_roles:
        effective_prompt = _build_media_role_prompt(
            prompt,
            has_first_frame=first_frame_bytes is not None,
            has_last_frame=last_frame_bytes is not None,
            last_frame_is_first=loop_frame,
            reference_image_count=len(reference_images),
            has_source_video=input_video_bytes is not None or bool(input_video_uri),
            reference_video_count=len(reference_videos),
        )

    if log_callback:
        mode = (
            "extending"
            if task_type == _TASK_EXTEND
            else "editing"
            if is_continuation
            else "generating"
        )
        await log_callback(
            f"Starting {mode} interaction with {spec.model} (task={task_type})"
        )

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds

    expired = f"Omni video interaction timed out after {timeout_seconds}s."

    def _timed_out(interaction_id: Any = None) -> TimeoutError:
        """A timeout that carries the interaction id, when there is one.

        The render keeps going on the service and gets billed whether or not
        this process is still waiting. Without the id in the message a caller
        who times out has no way to retrieve it, resume from it, or reconcile
        the charge — the spend simply vanishes. An MCP host with a shorter
        ceiling than timeout_seconds makes this the common case, not the edge.
        """
        if interaction_id:
            return TimeoutError(
                f"{expired} The render is still in progress on the service and "
                f"will be billed: interaction_id={interaction_id}. Collect it "
                "with get_omni_interaction rather than rendering again."
            )
        return TimeoutError(expired)

    # Set as soon as create returns, so every timeout after that point can
    # name the render it abandoned.
    interaction_id: Any = None

    async def _run_within_deadline(func: Any, /, **kwargs: Any) -> Any:
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise _timed_out(interaction_id)
        # run_off_loop, not asyncio.to_thread: a timed-out call cannot be
        # cancelled, and abandoning it on the loop's shared default executor
        # burns a worker that every other request also draws from.
        return await run_off_loop(
            functools.partial(func, **kwargs),
            timeout=remaining,
            message=str(_timed_out(interaction_id)),
        )

    def _deadline_expired() -> bool:
        return loop.time() >= deadline

    # Image parts, in the order the generated role declarations name them.
    image_parts: list[dict[str, Any]] = []
    if first_frame_bytes is not None:
        image_parts.append(_image_part(first_frame_bytes))
    if last_frame_bytes is not None and not loop_frame:
        image_parts.append(_image_part(last_frame_bytes))
    image_parts.extend(_image_part(img) for img in reference_images)
    image_parts.extend(_image_part(img) for img in legacy_images)

    # Video parts: the source first (it is <VIDEO_0>), then the references.
    video_parts: list[dict[str, Any]] = []
    if input_video_uri:
        video_parts.append(
            {
                "type": "video",
                "uri": input_video_uri,
                "mime_type": _guess_video_mime_from_uri(input_video_uri),
            }
        )
    if input_video_bytes is not None:
        video_parts.append(
            await _prepare_video_part(
                client,
                input_video_bytes,
                "input_video",
                spec=spec,
                log_callback=log_callback,
                run_within_deadline=_run_within_deadline,
                deadline_expired=_deadline_expired,
                warnings=warnings,
            )
        )
    for index, ref in enumerate(reference_videos):
        video_parts.append(
            await _prepare_video_part(
                client,
                ref,
                f"reference_video[{index}]",
                spec=spec,
                log_callback=log_callback,
                run_within_deadline=_run_within_deadline,
                deadline_expired=_deadline_expired,
                warnings=warnings,
            )
        )

    # Judged on the length the output can reach, not the caller's
    # duration_seconds: an edit or upscale sends none, so that value said
    # nothing about the output, and a 15s edit at the 6s default was
    # requested inline past the rule's own 10s line.
    delivery_seconds, delivery_seconds_are_bound = _output_seconds_for_delivery(
        spec,
        task_type,
        previous_interaction_id,
        sent_seconds=clamped_duration if sends_duration else None,
        source_seconds=source_duration_seconds,
        vertexai=bool(getattr(getattr(client, "_api_client", None), "vertexai", False)),
    )
    uri_delivery = (
        allow_uri_delivery
        and not output_gcs_uri
        and wants_uri_delivery(spec, normalized_resolution, task_type, delivery_seconds)
    )
    if uri_delivery:
        # Said here, where the choice is made, so the note cannot name a
        # different length from the one that decided it.
        warnings.append(
            omni_uri_delivery_note(
                normalized_resolution,
                task_type,
                delivery_seconds,
                seconds_are_bound=delivery_seconds_are_bound,
            )
        )
    create_kwargs = _build_create_kwargs(
        model=spec.model,
        prompt=effective_prompt,
        image_parts=image_parts,
        video_parts=video_parts,
        previous_interaction_id=previous_interaction_id,
        task_type=task_type,
        aspect_ratio=aspect_ratio,
        duration_seconds_int=clamped_duration,
        resolution=normalized_resolution,
        output_gcs_uri=output_gcs_uri,
        uri_delivery=uri_delivery,
        media_before_text=spec.media_before_text,
    )

    if not spec.sends_inferred_task and task is None:
        # 1.2 recommends prompting first; a forced task restricts references.
        create_kwargs.pop("generation_config", None)
    # Read off the request rather than re-derived, so task_source below
    # cannot describe a different request from the one sent: it said "NOT
    # sent" for a conversational upscale, which sends its task.
    task_sent = "generation_config" in create_kwargs

    # A deadline that has already passed sends nothing, so it is checked
    # before the request is marked as sent.
    if loop.time() >= deadline:
        raise _timed_out(None)
    if submission is not None:
        submission.create_sent = True
    quota_retries = 0
    create_error: Exception | None = None
    interaction: Any = None
    while True:
        try:
            interaction = await _run_within_deadline(
                _interactions(client).create, **create_kwargs
            )
            break
        except Exception as exc:
            wait = _quota_retry_wait(exc, quota_retries)
            if wait is None or deadline - loop.time() < (
                wait + _QUOTA_RETRY_RENDER_SECONDS
            ):
                create_error = exc
                break
        quota_retries += 1
        if submission is not None:
            submission.quota_retries = quota_retries
        if log_callback:
            await log_callback(
                f"The service refused the request with 429; sending it again "
                f"in {wait:g}s ({quota_retries} of "
                f"{len(_QUOTA_RETRY_WAITS_SECONDS)})."
            )
        await asyncio.sleep(wait)
    if quota_retries and create_error is None:
        warnings.append(
            "The service refused the request with 429 (quota or capacity) "
            f"{_times(quota_retries)} before starting anything; it was sent "
            "again and accepted."
        )
    if create_error is not None:
        exc = create_error
        if (
            previous_interaction_id is not None
            and task_type == _TASK_EXTEND
            and "duration" in create_kwargs["response_format"][0]
            and _EDIT_REJECTS_DURATION in str(exc)
        ):
            raise _extension_read_as_edit(exc, previous_interaction_id) from exc
        advice = _access_refusal(
            exc,
            create_kwargs.get("model"),
            vertexai=bool(
                getattr(getattr(client, "_api_client", None), "vertexai", False)
            ),
        )
        if advice is None:
            raise exc
        raise advice from exc

    interaction_id = _field(interaction, "id") or _field(interaction, "name")
    if not interaction_id:
        raise ValueError("Interaction create response carried no interaction id.")
    if submission is not None:
        submission.interaction_id = str(interaction_id)

    # Poll the background interaction until it leaves the in-flight statuses.
    # `completed` proceeds to extraction; any other terminal status fails fast
    # with the raw status and any error message.
    status = _field(interaction, "status") or _field(interaction, "state")
    # Report state CHANGES and a sparse heartbeat, never one line per poll.
    logged_status: Any = object()
    last_logged_at = loop.time()
    poll_errors = 0
    while status in _IN_FLIGHT_STATUSES:
        if _deadline_expired():
            raise _timed_out(interaction_id)
        if log_callback and (
            status != logged_status
            or loop.time() - last_logged_at >= _POLL_LOG_INTERVAL_SECONDS
        ):
            await log_callback(f"Interaction {interaction_id}: {status}")
            logged_status = status
            last_logged_at = loop.time()
        await asyncio.sleep(_POLL_INTERVAL)
        # google-genai's interactions.get takes the id as `id` (positional-or-
        # keyword), NOT `interaction_id`.
        try:
            interaction = await _run_within_deadline(
                _interactions(client).get, id=interaction_id
            )
        except Exception as exc:
            poll_errors += 1
            if not _poll_error_is_transient(exc) or poll_errors >= _POLL_ERROR_LIMIT:
                raise
            if log_callback:
                await log_callback(
                    f"Interaction {interaction_id}: status read failed "
                    f"({http_status(exc)}); reading it again."
                )
            continue
        poll_errors = 0
        status = _field(interaction, "status") or _field(interaction, "state")

    if status is not None and status != "completed":
        if submission is not None and status in OMNI_ENDED_STATUSES:
            submission.final_status = str(status)
        raise ValueError(
            f"Omni interaction {interaction_id} ended with status "
            f"'{status}': "
            f"{_interaction_failure_detail(interaction) or 'terminal or unrecognized status'}"
        )

    if submission is not None:
        submission.completed = True
    if log_callback:
        await log_callback("Interaction complete; resolving video output")

    inline_data, uri = _extract_video_payload(interaction)
    if not inline_data and not uri:
        raise ValueError(
            _no_video_returned(
                interaction_id,
                spec,
                inline_requested="delivery" not in create_kwargs["response_format"][0],
                vertexai=bool(
                    getattr(getattr(client, "_api_client", None), "vertexai", False)
                ),
                resolution=normalized_resolution,
                task=task_type,
                sent_seconds=clamped_duration if sends_duration else None,
                source_seconds=source_duration_seconds,
            )
        )

    if uri and uri.startswith("gs://"):
        # GCS-delivered output: pass the gs:// URI through unchanged (no
        # download, no local write), mirroring the Veo gs:// output path.
        video_url = uri
    else:
        video_bytes = await _resolve_video_bytes(
            client,
            inline_data,
            uri,
            log_callback,
            _run_within_deadline,
            _deadline_expired,
            max_bytes=_delivered_video_cap(spec),
        )
        filename = f"{uuid.uuid4()}.mp4"
        filepath = videos_dir / filename
        filepath.write_bytes(video_bytes)
        video_url = f"file://{filepath}"

    result: dict[str, Any] = {
        "message": "Video generated successfully",
        "video_url": video_url,
        "interaction_id": interaction_id,
        # The model that rendered, which is also the ID sent: one name for a
        # manifest, a price lookup and a follow-up call. A retired ID the
        # caller named is reported beside it rather than in its place.
        "model": spec.model,
        **({"requested_model": str(model)} if str(model) != spec.model else {}),
        "task": task_type,
        # A mixed role set maps to no single documented task, so none is sent
        # and the role tags carry the intent (see _select_task_type). That is
        # a decision, and `task` is the field a caller reads to learn what the
        # server decided — reporting a bare null told them nothing about it.
        "task_source": (
            "inferred from prompt and media roles; no task field sent, as recommended for Omni 1.2"
            if not spec.sends_inferred_task and task is None
            else f"classified as {task_type}, but NOT sent: a turn carrying "
            "previous_interaction_id cannot also carry a task (the API "
            "rejects the pair), so the service infers it from the context"
            if previous_interaction_id is not None and task_type and not task_sent
            else "requested by the caller"
            if task
            else f"classified from the role mix as {task_type}"
            if task_type
            else "not sent: this role mix maps to no single documented task, "
            "so the role declarations in the prompt carry it instead. The "
            "reference warns that the task field adds strict constraints, so "
            "inventing one here would over-constrain the render."
        ),
        # A continuation's rendered length is not the request: an edit sends
        # no duration, and the duration a 1.2 extension does send is only the
        # appended segment, while the render returns the whole assembled
        # clip. Reporting the request here would describe a render that did
        # not happen, and the caller bills from this field. None means
        # "unknown here, resolve upstream"; the request is kept separately so
        # nothing is lost.
        "duration_seconds": None if withholds_output_spec else clamped_duration,
        # What went on the wire, like aspect_ratio: an edit sends none, so
        # edit_video's 6.0 default was reported as if it had been asked for.
        "requested_duration_seconds": clamped_duration if sends_duration else None,
        # Same property as duration above: _build_create_kwargs omits
        # aspect_ratio on a continuation, so reporting the request here would
        # state a ratio the service never received — editing a 9:16 source
        # under the 16:9 default renders at the source's ratio, not the
        # request's.
        "aspect_ratio": None if withholds_output_spec else aspect_ratio,
        "requested_aspect_ratio": aspect_ratio,
        # What was asked for, or None when the request carried no resolution
        # and the service applied its own default.
        "resolution": normalized_resolution,
        "rendered_resolution": normalized_resolution or spec.rendered_resolution,
    }
    if effective_prompt != prompt:
        # The generated role declarations changed what the model was asked, so
        # the changed text travels back rather than staying an invisible
        # rewrite of the caller's prompt.
        result["effective_prompt"] = effective_prompt

    # Include warnings only when non-empty, matching src/video.py.
    if warnings:
        result["warnings"] = warnings

    return result


# How long a collect may spend fetching a completed render's video, on top of
# the time it waits for the render to finish.
OMNI_COLLECT_DOWNLOAD_SECONDS = 300


async def collect_omni_interaction(
    *,
    client: genai.Client,
    interaction_id: str,
    videos_dir: Path,
    wait_seconds: float = 0,
    log_callback: LogCallback | None = None,
) -> dict[str, Any]:
    """Read an interaction by id and, once it has completed, save its video.

    For a render this process stopped waiting for (a timeout, a dropped
    connection, a host that cut the tool call off): the service keeps
    rendering and bills it, and the id is the only handle on it. Nothing is
    created or sent but reads, so this never bills.

    Polls for up to ``wait_seconds`` while the interaction is in flight. The
    result always carries ``interaction_id``, ``status`` and ``model`` (None
    when the service does not echo it), and then, by status:

    * still in flight: nothing more; nothing was saved.
    * ``completed``: ``video_url``, a file:// it saved under ``videos_dir``,
      or the gs:// object it was delivered to.
    * anything else: ``error``, with the service's reason.
    """
    loop = asyncio.get_running_loop()
    wait_until = loop.time() + max(0.0, float(wait_seconds))
    deadline = wait_until + OMNI_COLLECT_DOWNLOAD_SECONDS

    async def _run(func: Any, /, **kwargs: Any) -> Any:
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise TimeoutError(f"Reading interaction {interaction_id} ran out of time.")
        return await run_off_loop(
            functools.partial(func, **kwargs),
            timeout=remaining,
            message=f"Reading interaction {interaction_id} ran out of time.",
        )

    async def _read() -> Any:
        # A 429 or a 5xx on a read is read again, as the render's own poll
        # does; the first read fails outright on anything else (a 404 on the
        # wrong backend, say), which the caller explains.
        errors = 0
        while True:
            try:
                return await _run(_interactions(client).get, id=interaction_id)
            except Exception as exc:
                errors += 1
                if not _poll_error_is_transient(exc) or errors >= _POLL_ERROR_LIMIT:
                    raise
                await asyncio.sleep(_POLL_INTERVAL)

    interaction = await _read()
    status = _field(interaction, "status") or _field(interaction, "state")
    while status in _IN_FLIGHT_STATUSES and loop.time() + _POLL_INTERVAL <= wait_until:
        if log_callback:
            await log_callback(f"Interaction {interaction_id}: {status}")
        await asyncio.sleep(_POLL_INTERVAL)
        interaction = await _read()
        status = _field(interaction, "status") or _field(interaction, "state")

    model = _field(interaction, "model")
    result: dict[str, Any] = {
        "interaction_id": interaction_id,
        "status": status,
        "model": str(model) if model else None,
    }
    previous = _field(interaction, "previous_interaction_id")
    if previous:
        result["previous_interaction_id"] = str(previous)
    if status in _IN_FLIGHT_STATUSES:
        return result
    if status != "completed":
        result["error"] = (
            f"Omni interaction {interaction_id} ended with status '{status}': "
            f"{_interaction_failure_detail(interaction) or 'terminal or unrecognized status'}"
        )
        return result

    inline_data, uri = _extract_video_payload(interaction)
    if not inline_data and not uri:
        result["error"] = (
            f"Omni interaction {interaction_id} completed, but carried no video."
        )
        return result
    if uri and uri.startswith("gs://"):
        result["video_url"] = uri
        return result
    video_bytes = await _resolve_video_bytes(
        client,
        inline_data,
        uri,
        log_callback,
        _run,
        lambda: loop.time() >= deadline,
        max_bytes=_delivered_video_cap(
            omni_spec(model if is_omni_model(model) else None)
        ),
    )
    filepath = videos_dir / f"{uuid.uuid4()}.mp4"
    filepath.write_bytes(video_bytes)
    result["video_url"] = f"file://{filepath}"
    return result


def omni_extension_output_lengths(
    spec: OmniModelSpec,
    source_seconds: float | None,
    times: int,
    per_turn_seconds: float | None = None,
    *,
    vertexai: bool = False,
) -> list[float]:
    """Billable OUTPUT length of each turn of an extension chain.

    MEASURED: a 3.01s source extended once returned a 13.01s file. A turn
    renders the ASSEMBLED clip — the source plus the appended increment — not
    the increment alone. Omni bills per second of output, so each turn of a
    chain re-bills every second that came before it, and the cost grows
    quadratically in the number of turns rather than linearly.

    That measurement overrules two documentary readings that pointed the other
    way, and both are recorded here because they are the trap: the Interactions
    API reference calls ``duration`` "the length of the generated video files"
    with a 3-10s range, and Vertex's documented sample response for an omni
    extension bills 28,832 video output tokens, which is 4.98s at the
    published rate. Neither survives a real render.

    Args:
        spec: The model's capability record.
        source_seconds: Length of the clip being extended. This is now
            load-bearing, not decorative: it is the base every turn is billed
            on top of. None is taken at the cumulative ceiling: an extension
            reads a source of any length, so there is no upload cap to fall
            back on, and assuming a SHORTER source than the real one
            under-quotes, which is the one direction a pre-flight may not err
            in.
        vertexai: Whether Vertex AI serves the chain, whose cumulative
            ceiling and retained window are its own (see omni_spec).
        times: How many turns are planned.
        per_turn_seconds: The appended duration, 3s up to the model's
            extension step; None means the step itself (10s by default).

    Returns:
        The assembled length after each turn, in order — which is what each
        turn bills.
    """
    if vertexai:
        spec = omni_spec(spec.model, vertexai=True)
    ceiling = float(spec.max_extended_seconds or spec.max_duration_seconds)
    assembled = float(source_seconds) if source_seconds is not None else ceiling
    lengths: list[float] = []
    for _ in range(max(0, times)):
        if spec.extension_window_seconds:
            assembled = min(assembled, float(spec.extension_window_seconds))
        appended = omni_extension_turn_seconds(spec, assembled, per_turn_seconds)
        if appended is None:
            # The documented ceiling is on the assembled clip, so a chain with
            # less than one turn's minimum left has nothing it can render.
            break
        # Rounded: 3.03 + 3 summed to 6.029999999999999 in the output.
        assembled = round(assembled + appended, 6)
        lengths.append(assembled)
    return lengths


def omni_duration_clamp_warning(
    spec: OmniModelSpec, duration_seconds: float | None, *, extension: bool = False
) -> str | None:
    """What a sent duration was changed to, or None when it went as asked.

    A render takes whole seconds from 3 to the model's maximum (an extension
    turn, to its step), so 2 becomes 3, 16 becomes 15 and 5.5 becomes 6. The
    real run said so for the first two only, and the dry run for none.
    """
    if duration_seconds is None or not math.isfinite(duration_seconds):
        return None
    longest = spec.extension_step_seconds if extension else spec.max_duration_seconds
    rounded = round(duration_seconds)
    clamped = min(longest, max(_MIN_DURATION, rounded))
    if clamped == duration_seconds:
        return None
    if rounded < _MIN_DURATION:
        why = f"below the {_MIN_DURATION}s minimum"
    elif rounded > longest:
        why = f"above the {longest}s maximum"
    else:
        why = "not a whole second"
    return f"duration_seconds={duration_seconds:g} is {why}; it renders {clamped}s."


def omni_extension_turn_seconds(
    spec: OmniModelSpec,
    assembled_seconds: float,
    per_turn_seconds: float | None = None,
    *,
    vertexai: bool = False,
) -> int | None:
    """Whole seconds the next turn appends to a clip, or None when none fit.

    The step asked for (3s up to the model's extension step, 10s by default)
    while it fits under the cumulative ceiling, and past that the whole
    seconds that still fit, since a turn's duration is a whole second from 3.
    A 3.03s source extended three times on the Gemini API was quoted at
    13.03, 23.03 and 30.0s: 6.97s appended by a turn that asked for 10s, a
    length no request can name. That turn now asks for 6s (29.03s).

    Args:
        spec: The model's capability record.
        assembled_seconds: Length of the clip the turn extends.
        per_turn_seconds: The appended duration asked for; None means the
            model's extension step.
        vertexai: Whether Vertex AI serves the turn, whose ceiling and
            retained window are its own.

    Returns:
        The seconds to append, or None when fewer than 3 fit.
    """
    if vertexai:
        spec = omni_spec(spec.model, vertexai=True)
    step = int(spec.extension_step_seconds or spec.max_duration_seconds)
    if per_turn_seconds is not None and spec.supports_extend:
        step = min(step, max(_MIN_DURATION, round(per_turn_seconds)))
    ceiling = float(spec.max_extended_seconds or spec.max_duration_seconds)
    assembled = float(assembled_seconds)
    if spec.extension_window_seconds:
        assembled = min(assembled, float(spec.extension_window_seconds))
    # The epsilon keeps float noise (6.999999999) from costing a whole second.
    room = math.floor(ceiling - assembled + 1e-6)
    seconds = min(step, room)
    return seconds if seconds >= _MIN_DURATION else None


def omni_extension_priced_lengths(
    spec: OmniModelSpec,
    source_seconds: float | None,
    times: int,
    *,
    vertexai: bool = False,
    per_turn_seconds: float | None = None,
) -> list[float]:
    """The per-turn lengths a quote and a cost cap must PRICE.

    Identical to omni_extension_output_lengths when the source is measured:
    the chain is clamped at the ceiling and only those turns run. When it is
    NOT measured the render loop cannot clamp -- the fallback assumes a
    source already at the ceiling, and cutting a short real source's chain on
    that guess would render fewer turns than it can take -- so every one of
    ``times`` turns runs, and every one of them must be priced. The fallback
    list stops emitting at the ceiling, which is exactly what made the quote
    and the cap disagree with the loop: a 10s-fallback list has three entries
    for times=4, the loop ran four, and a $10 cap that passed a $9.14
    projection billed at least $10.15.

    Turns past the fallback's ceiling are priced AT the ceiling, which is the
    most any turn can render, so the result is a true upper bound whatever the
    real source turns out to be.
    """
    if vertexai:
        spec = omni_spec(spec.model, vertexai=True)
    lengths = omni_extension_output_lengths(
        spec, source_seconds, times, per_turn_seconds, vertexai=vertexai
    )
    if source_seconds is not None or len(lengths) >= times:
        return lengths
    # The same ceiling omni_extension_output_lengths stops at. This fell back
    # to a flat 10s while that one used the model's own maximum.
    ceiling = float(spec.max_extended_seconds or spec.max_duration_seconds)
    return lengths + [ceiling] * (times - len(lengths))


def omni_extension_refusal(
    spec: OmniModelSpec, source_seconds: float | None
) -> str | None:
    """Why a chain cannot append anything, or None.

    A source already at the documented ceiling plans zero turns. The quote
    reported planned_turns: 0 with estimated_cost: null, the render attempted
    every turn anyway, and max_cost_usd could not stop it because an unpriced
    plan reads as $0.00 — the cap compared 0.0 against the limit and passed.
    A call the quote cannot price is a call that must not run.
    """
    if source_seconds is None or spec.max_extended_seconds <= 0:
        return None
    if spec.extension_window_seconds:
        return None
    if omni_extension_turn_seconds(spec, source_seconds) is not None:
        return None
    # "Edit it instead" could not be followed as it stood: this fires at 30s
    # or more, and 1.2 edits sources of at most 15s.
    edit_cap = omni_source_limit_seconds(spec, task=_TASK_EDIT)
    edit = (
        f"trim it to {edit_cap:g}s or less and edit it"
        if edit_cap and source_seconds > edit_cap
        else "edit it"
    )
    where = (
        "at or over"
        if source_seconds >= spec.max_extended_seconds
        else f"within {_MIN_DURATION}s of"
    )
    return (
        f"The source is {source_seconds:g}s, {where} {spec.model}'s documented "
        f"{spec.max_extended_seconds}s ceiling for an extended video, and a turn "
        f"appends at least {_MIN_DURATION}s, so no turn can run. Nothing was "
        f"rendered or billed. Start a new generation, or {edit} rather than "
        "extending it."
    )


def omni_extension_appended_seconds(
    source_seconds: float | None,
    turn_outputs: list[float],
    *,
    spec: OmniModelSpec | None = None,
) -> float | None:
    """New footage a chain produced, given each turn's assembled length.

    The turns' own lengths cannot be summed to get this: every one of them
    contains all the footage before it. The answer is the last turn's length
    minus the source, and it is unknowable without the source — reporting a
    sum as "appended" overstated a 2-turn chain off a 3s source as 36s of new
    footage when it made 20s.
    """
    if not turn_outputs or source_seconds is None:
        return None
    # Rounded: a measured 29.03s chain off a 23.04s source reported
    # 5.990000000000002.
    if spec and spec.extension_window_seconds:
        appended = 0.0
        previous = source_seconds
        for output in turn_outputs:
            appended += max(0.0, output - min(previous, spec.extension_window_seconds))
            previous = output
        return round(appended, 6)
    return round(max(0.0, turn_outputs[-1] - float(source_seconds)), 6)


def omni_continuation_upper_bound(
    spec: OmniModelSpec,
    task: str | None,
    source_seconds: float | None,
    *,
    vertexai: bool = False,
    per_turn_seconds: float | None = None,
) -> float:
    """Longest clip a continuation could render, for a cost that cannot be measured.

    A continuation's length is the service's choice, so a render this
    process cannot read (a gs:// delivery over the fetch cap, or outside its
    credentials) has to be billed at a bound. Which bound depends on the kind:

    * ``extend`` returns the whole growing clip, so it is the source plus one
      step, clamped at the backend's cumulative ceiling. Billing every
      extension at the per-turn step under-billed a chain roughly by its turn
      count -- from a 10s source with 10s steps, the Gemini API renders 20s
      then 30s, not 10s twice, and Vertex (which keeps the last 30s and
      appends to it) renders 20s, 30s, 40s, 40s.
    * an ``edit`` re-renders the clip it is given. The one measurement in hand
      put a 3s source's edit at 10.01s, i.e. the per-render maximum, so that is
      the floor; a source longer than it (only reachable through an extension
      chain) raises the bound to its own length.

    With no source length to hand, the model's own ceiling is the only honest
    answer — over-stating in a corner that needs both an unmeasurable delivery
    and a source this server never recorded.
    """
    if vertexai:
        spec = omni_spec(spec.model, vertexai=True)
    ceiling = float(spec.max_extended_seconds or spec.max_duration_seconds)
    if task == _TASK_EXTEND and spec.supports_extend:
        # A turn renders the assembled clip (measured: a 3.01s source came
        # back 13.01s), so the bound is the source plus one increment, and the
        # documented ceiling when the source is unknown.
        step = (
            per_turn_seconds
            if per_turn_seconds is not None
            else spec.extension_step_seconds
        )
        if source_seconds is None:
            return (
                min(spec.extension_window_seconds + step, ceiling)
                if spec.extension_window_seconds
                else ceiling
            )
        if spec.extension_window_seconds:
            source_seconds = min(source_seconds, float(spec.extension_window_seconds))
        return min(source_seconds + step, ceiling)
    # An edit re-renders the clip it is given, and the per-render maximum is
    # what the one measurement in hand showed it producing (a 3s source
    # rendered 10.01s). A source longer than that raises the bound to its own
    # length, but never past the model's edit-source cap, which binds an
    # upload and a multi-turn source alike -- so an unrecorded source is taken
    # at that cap, and the extension ceiling is never assumed for an edit.
    if source_seconds is None:
        return min(
            max(float(spec.max_duration_seconds), omni_source_limit_seconds(spec)),
            ceiling,
        )
    edit_cap = omni_source_limit_seconds(spec, task=_TASK_EDIT)
    if edit_cap:
        source_seconds = min(source_seconds, edit_cap)
    return min(max(float(spec.max_duration_seconds), source_seconds), ceiling)


def validate_omni_request(
    model: str,
    *,
    resolution: str | None = None,
    task: str | None = None,
    has_first_frame: bool = False,
    has_last_frame: bool = False,
    reference_image_count: int = 0,
    reference_video_count: int = 0,
    inferred_image_count: int = 0,
    duration_seconds: float | None = None,
    is_continuation: bool = False,
    vertexai: bool = False,
) -> OmniModelSpec:
    """Check a request shape against one model's capabilities, or raise.

    Counts rather than bytes, so the tool layer can run exactly these rules
    before it fetches a single URI — a dry run that quoted a keyframe render
    the preview model cannot do would be a price for an impossible call — and
    the impl can run them again on what it actually received.

    Silently dropping any of these would render something the caller did not
    ask for and bill them for it: a 4K request coming back 720p, a keyframe
    pair coming back as two subject references. Each rejection names the model
    that does support the argument.

    Args:
        model: The omni model ID being called.
        resolution: Requested output resolution, if any.
        task: Explicitly requested task, if any.
        has_first_frame: Whether a ``<FIRST_FRAME>`` image was supplied.
        has_last_frame: Whether a ``<LAST_FRAME>`` image was supplied.
        reference_image_count: Number of explicit ``<IMAGE_REF_N>`` images.
        reference_video_count: Number of ``<VIDEO_REF_N>`` clips.
        inferred_image_count: Number of images passed in the legacy form,
            whose role the model infers.
        duration_seconds: The duration that will be SENT, or None when the
            request carries none. A caller whose request strips duration (an
            edit) passes None: a value that never reaches the wire must not be
            what refuses the render.
        is_continuation: Whether the request continues existing footage.
        vertexai: Whether Vertex AI serves the request, for the limits that
            backend sets on its own.

    Returns:
        The model's spec, so callers do not look it up twice.
    """
    spec = omni_spec(model)
    # Zero is allowed and snaps to the model's minimum, as it does in the tool
    # layer's _validate_duration_seconds. Refusing it here while the tool
    # layer quoted it made a dry run price a call the real run then refused.
    if duration_seconds is not None and (
        not math.isfinite(duration_seconds) or duration_seconds < 0
    ):
        raise ValueError(
            "duration_seconds must be a finite number of seconds, not negative; "
            f"got {duration_seconds!r}. Zero snaps to the {_MIN_DURATION}s minimum."
        )
    # Vertex AI's edit image cap. Checked here, the one place every caller
    # passes through, because it was hand-rolled three times (this module's
    # impl, generate_video_omni and edit_video) and the copies counted
    # differently. Every image the request carries counts, whatever its
    # role: each is an image part of the same edit request, and the impl,
    # the last check before the wire, always counted that way. Counting
    # fewer in a tool let a dry run quote an edit the render then refused.
    image_count = (
        reference_image_count
        + inferred_image_count
        + int(has_first_frame)
        + int(has_last_frame)
    )
    if (
        vertexai
        and spec.max_vertex_edit_images
        and is_continuation
        and task not in (_TASK_EXTEND, _TASK_UPSCALE)
        and image_count > spec.max_vertex_edit_images
    ):
        raise ValueError(
            f"Vertex AI editing on {spec.model} supports at most "
            f"{spec.max_vertex_edit_images} images per request, counting "
            "reference, inferred-role and first/last-frame images together; "
            f"this request has {image_count}. Send at most "
            f"{spec.max_vertex_edit_images}."
        )
    # Each limit is gated on the capability it limits, not on which model
    # this is: a model that gains one is checked by its spec entry alone.
    # An unsent duration (an edit's) is judged at the extension's documented
    # default turn, as the extend tool sends it.
    duration = min(
        spec.max_duration_seconds,
        max(
            _MIN_DURATION,
            round(
                duration_seconds
                if duration_seconds is not None
                else spec.extension_step_seconds or spec.max_duration_seconds
            ),
        ),
    )
    if (
        spec.supports_keyframes
        and has_last_frame
        and duration > spec.max_keyframe_seconds
    ):
        raise ValueError(
            "First/last-frame interpolation supports "
            f"{_MIN_DURATION} to {spec.max_keyframe_seconds} seconds, "
            f"not {duration}s."
        )
    if spec.supports_role_tags:
        references = reference_image_count + (
            inferred_image_count if inferred_image_count > 1 else 0
        )
        if references > spec.max_reference_images:
            raise ValueError(
                f"Too many reference images ({references}); {spec.model} "
                f"accepts at most {spec.max_reference_images}."
            )
    if spec.max_reference_videos and reference_video_count:
        if is_continuation or task in _CONTINUATION_TASKS:
            raise ValueError(
                "Video references are supported only for new generation, not "
                "editing or extension."
            )
        longest = spec.max_seconds_with_video_references
        if longest and reference_video_count > 1 and duration > longest:
            raise ValueError(
                f"Videos from {longest + 1} to {spec.max_duration_seconds} "
                "seconds support at most 1 reference video (up to "
                f"{spec.max_reference_video_seconds:g} seconds)."
            )
    if (
        spec.supports_extend
        and task == _TASK_EXTEND
        and duration_seconds is not None
        and duration_seconds > spec.extension_step_seconds
    ):
        raise ValueError(
            f"An extension adds {_MIN_DURATION} to "
            f"{spec.extension_step_seconds} seconds per turn."
        )
    if task == _TASK_UPSCALE and not spec.supports_upscale:
        raise ValueError(
            f"{spec.model} does not support standalone upscaling. Use {OMNI_1_2_MODEL}."
        )
    upgrade = f"Pass model='{DEFAULT_OMNI_MODEL}' to use it."
    explicit_roles = has_first_frame or has_last_frame or bool(reference_image_count)

    if inferred_image_count and explicit_roles:
        raise ValueError(
            "Images whose role the model infers cannot be combined with "
            "first-frame, last-frame or reference images — the first lets the "
            "model decide each image's role and the others state it, so "
            "together they would describe the same images two ways. Use the "
            "explicit-role arguments alone."
        )
    if has_last_frame and not has_first_frame:
        raise ValueError(
            "A last frame requires a first frame: <LAST_FRAME> is documented "
            "as only usable together with <FIRST_FRAME>."
        )
    if resolution is not None and not spec.supports_resolution:
        raise ValueError(
            f"{spec.model} has no resolution parameter — it renders "
            f"{spec.rendered_resolution} only. {upgrade}"
        )
    if resolution is not None:
        normalized = normalize_omni_resolution(resolution)
        if normalized is None or normalized not in spec.resolutions:
            raise ValueError(
                f"Unsupported resolution '{resolution}' for {spec.model}. "
                f"Supported values are {', '.join(spec.resolutions)}."
            )
    if task == _TASK_EXTEND and not spec.supports_extend:
        raise ValueError(f"{spec.model} does not support the extend task. {upgrade}")
    if task is not None and task not in _ALL_TASKS:
        # Read off the spec, so a task the model gains is listed: this named
        # extend but never upscale, which 1.2 accepts.
        supported = [
            name
            for name in _ALL_TASKS
            if (name != _TASK_EXTEND or spec.supports_extend)
            and (name != _TASK_UPSCALE or spec.supports_upscale)
        ]
        raise ValueError(f"Unknown task '{task}'. Supported: {', '.join(supported)}.")
    if explicit_roles and not spec.supports_role_tags:
        raise ValueError(
            f"{spec.model} cannot bind images to explicit first-frame, "
            f"last-frame or reference roles. {upgrade}"
        )
    if reference_video_count:
        if not spec.max_reference_videos:
            raise ValueError(
                f"{spec.model} does not accept video references. {upgrade}"
            )
        if reference_video_count > spec.max_reference_videos:
            raise ValueError(
                f"Too many reference videos ({reference_video_count}); "
                f"{spec.model} accepts at most {spec.max_reference_videos}, "
                f"of up to {spec.max_reference_video_seconds:g}s each."
            )
    return spec
