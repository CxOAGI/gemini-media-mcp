"""Omni video generation helpers (Interactions API).

Support for Google's Omni video models. Unlike the VEO models in ``video.py``
(which use the long-running ``generate_videos`` operation), the Omni models are
driven through the **Interactions API**: ``client.interactions.create(...,
background=True)`` starts a server-side render, ``client.interactions.get(...)``
is polled until the interaction completes, and multi-turn conversational
editing is done by threading a ``previous_interaction_id`` (the server holds
the prior video context).

Two models live here, and they are NOT interchangeable:

``gemini-omni-flash-preview`` (``OMNI_PREVIEW_MODEL``)
    The original preview. 720p/24fps only, no resolution control, no video
    extension, no explicit keyframe roles. Everything about the request it
    receives is live-verified and is deliberately left untouched.

``gemini-omni-1.1-flash`` (``OMNI_1_1_MODEL``)
    The GA successor (https://ai.google.dev/gemini-api/docs/omni). It adds, on
    top of everything above:
      * ``resolution`` in ``response_format`` — 360p / 720p (default) /
        1080p / 4k, the last two upscaled from the base render;
      * a fifth task, ``extend``, which appends a seamless continuation using
        the last 10s of the source as context, in 10s steps to a 40s total;
      * first/last-frame interpolation via the ``<FIRST_FRAME>`` /
        ``<LAST_FRAME>`` prompt tags;
      * video references — up to 3 clips of up to 3s each — via
        ``<VIDEO_REF_N>``, alongside ``<IMAGE_REF_N>`` image references;
      * the ``[# Sources ...] [# References ...]`` prefix syntax that binds
        each uploaded item to one of those roles unambiguously.

Request/response shapes follow the Vertex AI "Use Gemini Omni Flash …to
generate videos" REST reference and the Interactions API docs:
  * media rides INSIDE ``input`` as flattened parts
    ({type: 'text'|'image'|'video'|'document', text|data|uri, mime_type}) —
    there are no separate image/video kwargs;
  * ``response_format`` is a LIST of one object
    ``[{'type': 'video', 'aspect_ratio': ..., 'duration': 'Ns',
    'resolution': ..., 'delivery': 'uri', 'gcs_uri': ...}]``. aspect_ratio
    ("16:9"/"9:16"), duration ("3s".."10s") and resolution live here;
    ``delivery='uri'`` + ``gcs_uri`` sends output to Cloud Storage, otherwise
    the video bytes come back inline. The Gemini API docs show a bare object
    rather than a list; the SDK normalizes both, and the list is what is
    live-verified here;
  * ``generation_config={'video_config': {'task': ...}}`` carries the task
    type (text_to_video / image_to_video / reference_to_video / edit, plus
    extend on 1.1);
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
import re
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google import genai

from .video import run_off_loop

# Type for async log callback from MCP context
LogCallback = Callable[[str], Awaitable[None]]

OMNI_PREVIEW_MODEL = "gemini-omni-flash-preview"
OMNI_1_1_MODEL = "gemini-omni-1.1-flash"

# The model every omni entry point still defaults to. Kept at the preview ID
# on purpose: 1.1 is the better model, but flipping the default would move
# every existing caller onto a different render (and a different set of
# accepted request fields) without them asking. It is one keyword away.
OMNI_MODEL = OMNI_PREVIEW_MODEL
DEFAULT_OMNI_MODEL = OMNI_PREVIEW_MODEL

OMNI_MODELS: tuple[str, ...] = (OMNI_PREVIEW_MODEL, OMNI_1_1_MODEL)

# Output spec limits documented for both omni models.
_SUPPORTED_ASPECT_RATIOS = ("16:9", "9:16")

# Documented output duration bounds (sent as "Ns" in response_format).
_MIN_DURATION = 3
_MAX_DURATION = 10

# Public alias: the tool layer quotes this as the worst case for an edit,
# whose rendered length the service chooses and does not document.
OMNI_MAX_DURATION_SECONDS = _MAX_DURATION

# Resolutions gemini-omni-1.1-flash accepts, in the spelling this server uses
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

# Documented task types. ``extend`` is 1.1-only.
_TASK_TEXT_TO_VIDEO = "text_to_video"
_TASK_IMAGE_TO_VIDEO = "image_to_video"
_TASK_REFERENCE_TO_VIDEO = "reference_to_video"
_TASK_EDIT = "edit"
_TASK_EXTEND = "extend"

_ALL_TASKS = (
    _TASK_TEXT_TO_VIDEO,
    _TASK_IMAGE_TO_VIDEO,
    _TASK_REFERENCE_TO_VIDEO,
    _TASK_EDIT,
    _TASK_EXTEND,
)

# Tasks that continue existing footage. Both reject ``duration`` (live-
# verified for edit; extend's length is documented as service-chosen) and
# neither is given an aspect ratio, which the source already fixes.
_CONTINUATION_TASKS = (_TASK_EDIT, _TASK_EXTEND)


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
        extension_step_seconds: Length one extension turn appends.
        max_extended_seconds: Documented cumulative ceiling for a chain of
            extensions.
        max_uploaded_source_seconds: Documented cap on an UPLOADED video used
            as an edit or extension source (multi-turn sources are exempt).
        media_before_text: Whether media parts precede the text part, matching
            the ordering that model's own reference shows.
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
    OMNI_1_1_MODEL: OmniModelSpec(
        model=OMNI_1_1_MODEL,
        resolutions=OMNI_RESOLUTIONS,
        rendered_resolution=OMNI_DEFAULT_RESOLUTION,
        # The 1.1 reference documents aspect_ratio, resolution and delivery in
        # response_format and never mentions duration. The field is still in
        # the SDK's VideoResponseFormat (shared across models) and the preview
        # model accepts it, so it is still sent — but a caller leaning on it
        # is warned rather than left to discover the gap from a 400.
        duration_is_documented=False,
        supports_extend=True,
        supports_keyframes=True,
        supports_role_tags=True,
        # "Video references support a maximum of 3 clips, up to 3 seconds each."
        max_reference_videos=3,
        max_reference_video_seconds=3.0,
        # "You can extend videos by 10s, up to a total length of 40s."
        extension_step_seconds=10,
        max_extended_seconds=40,
        # "Input videos for editing and extension must be 10 seconds or less
        # when uploading (unless extending videos generated by the model in
        # multi-turn)."
        max_uploaded_source_seconds=10.0,
        # Every example in the 1.1 reference puts the media parts first.
        media_before_text=True,
    ),
}


def is_omni_model(model: str | None) -> bool:
    """Whether ``model`` is served through the Interactions API here."""
    return str(model) in OMNI_SPECS


def omni_spec(model: str | None) -> OmniModelSpec:
    """Return the capability record for ``model``, or raise.

    Raising (rather than falling back to the default spec) keeps a typo from
    quietly buying the preview model's restrictions on a 1.1 request.
    """
    spec = OMNI_SPECS.get(str(model))
    if spec is None:
        raise ValueError(
            f"Unknown omni model '{model}'. Supported: {', '.join(OMNI_MODELS)}."
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
        # 1.1 render on a probe that no longer applies.
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
# pure defence-in-depth. 1.1 reaches 4K and 40s, and the URI-delivery path
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

# Interval (seconds) between polls of an in-flight background interaction.
_POLL_INTERVAL = 5

# Interval between polls of a Files API upload waiting to leave PROCESSING.
_FILE_POLL_INTERVAL = 2

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


def _detect_video_mime(data: bytes) -> str:
    """Return the exact video MIME type from magic bytes, or raise.

    Distinguishes MP4 vs QuickTime/MOV via the ftyp brand and recognizes
    WebM/Matroska, MPEG program/video streams, and AVI, rather than labeling
    every input video as MP4.
    """
    brand = _ftyp_brand(data)
    if brand is not None:
        if brand.startswith("qt"):
            # The SDK's VideoContentMimeType literal set spells MOV "video/mov"
            # (google/genai/_gaos/types/interactions/videocontent.py); the
            # RFC "video/quicktime" is not in it and rides as UnrecognizedStr.
            return "video/mov"
        if brand in _HEIF_BRANDS or brand in _HEIF_SEQUENCE_BRANDS:
            raise ValueError(
                "Input labeled as video but the data is a HEIC/HEIF image."
            )
        return "video/mp4"
    if data[:4] == b"\x1a\x45\xdf\xa3":
        return "video/webm"
    if data[:3] == b"\x00\x00\x01" and data[3:4] in (b"\xba", b"\xb3"):
        return "video/mpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"AVI ":
        # SDK spells AVI "video/avi", not the RFC "video/x-msvideo" — the
        # latter is absent from VideoContentMimeType and rides as
        # UnrecognizedStr. See the MOV note above.
        return "video/avi"
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


def _document_part(uri: str) -> dict[str, Any]:
    """A Files API reference, the shape the docs use for uploaded videos."""
    return {"type": "document", "uri": uri}


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

    ``media_before_text`` follows each model's own reference: every 1.1 example
    puts the media ahead of the prompt, and the preview model's verified
    requests put the text first. Image order is load-bearing on 1.1 — a
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
) -> str:
    """Prefix ``prompt`` with the documented source/reference declarations.

    gemini-omni-1.1-flash infers each uploaded item's role from the prompt
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
    if has_last_frame:
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
    spec: OmniModelSpec, resolution: str | None, task: str | None
) -> bool:
    """Whether this request's output should be delivered by URI, not inline.

    The reference gives one rule — anything over 720p — and an extension earns
    the same treatment for the same reason: it renders the whole growing clip,
    up to 40s, which is far past 4 MB. Both are outputs the new model can
    produce and the preview model never could, so the inline default that has
    always been fine is no longer fine for them.

    Only meaningful on a model that can render above 720p at all; a caller on
    the Gemini API gets a Google-hosted URI, and one on Vertex needs a
    gcs_uri instead (the API requires it there), which the caller supplies.
    """
    if not spec.supports_resolution:
        return False
    return resolution in _LARGE_OUTPUT_RESOLUTIONS or task == _TASK_EXTEND


def _select_task_type(
    *,
    previous_interaction_id: str | None,
    input_video_bytes: bytes | None,
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
    declared_references = reference_image_count + reference_video_count
    if has_last_frame:
        return None
    if has_first_frame and declared_references:
        return None
    if declared_references and inferred_image_count:
        return None
    if previous_interaction_id or input_video_bytes is not None:
        return _TASK_EDIT
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
        so conversational turns send NO generation_config;
      * edit tasks reject ``duration`` in response_format ("Duration cannot
        be set in response format for edit task") — duration and aspect
        ratio cannot be sent for an edit-type request, so neither is. What
        the service then renders is undocumented and is NOT the source's
        length — a measured 3s source came back at 10.01s;
      * ``extend`` is treated the same way: its reference describes the
        continuation's length as the model's choice ("a 3-10 second
        continuation", "extend videos by 10s"), and the source already fixes
        the aspect ratio.

    ``resolution`` rides in response_format on every task that sends one, and
    only when the caller asked for a specific one — an omitted field is the
    documented 720p default, which is what the reference's own examples send.
    """
    is_continuation = (
        task_type in _CONTINUATION_TASKS or previous_interaction_id is not None
    )

    response_format_item: dict[str, Any] = {"type": "video"}
    if not is_continuation:
        response_format_item["aspect_ratio"] = aspect_ratio
        if duration_seconds_int is not None:
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

    create_kwargs: dict[str, Any] = {
        "model": model,
        "input": _build_input_parts(
            prompt,
            image_parts,
            video_parts,
            media_before_text=media_before_text,
        ),
        "background": True,
        "response_format": [response_format_item],
    }
    if previous_interaction_id is not None:
        # Conversational turn: the server holds the video context; sending a
        # task alongside previous_interaction_id is rejected.
        create_kwargs["previous_interaction_id"] = previous_interaction_id
    elif task_type is not None:
        create_kwargs["generation_config"] = {"video_config": {"task": task_type}}
    return create_kwargs


def _field(obj: Any, name: str) -> Any:
    """Read a field from an SDK object or a plain dict interchangeably."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


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
    ``_MAX_INLINE_VIDEO_BYTES`` goes through the Files API on a model whose
    reference documents the ``document`` part; on the preview model, which
    documents no such shape, the request is left inline and the caller is told
    why it may be refused, rather than quietly sending an unverified field.
    """
    if len(data) <= _MAX_INLINE_VIDEO_BYTES:
        return _inline_video_part(data)
    if not spec.supports_role_tags:
        warnings.append(
            f"{label} is {len(data)} bytes, over the {_MAX_INLINE_VIDEO_BYTES}-byte "
            f"inline threshold, and {spec.model} documents no Files API input "
            "shape — it is sent inline and the API may reject the request. "
            f"Use {OMNI_1_1_MODEL} (which uploads instead) or a shorter clip."
        )
        return _inline_video_part(data)
    uri = await _upload_video_via_files_api(
        client, data, label, log_callback, run_within_deadline, deadline_expired
    )
    return _document_part(uri)


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
    reference_video_bytes_list: list[bytes] | None = None,
    previous_interaction_id: str | None = None,
    task: str | None = None,
    aspect_ratio: str = "16:9",
    duration_seconds: float | None = 6.0,
    resolution: str | None = None,
    output_gcs_uri: str | None = None,
    allow_uri_delivery: bool = False,
    timeout_seconds: int = 600,
    log_callback: LogCallback | None = None,
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
        model: ``gemini-omni-flash-preview`` (default) or
            ``gemini-omni-1.1-flash``. Everything below that is 1.1-only says
            so, and is refused rather than dropped on the preview model.
        image_bytes_list: Optional input images whose role the model infers.
            One image is treated as a first frame (image_to_video); several
            are treated as references (reference_to_video). Mutually exclusive
            with the explicit-role image arguments.
        first_frame_bytes: 1.1 only. Image to start the video on.
        last_frame_bytes: 1.1 only. Image to end the video on; requires
            ``first_frame_bytes`` (documented as ``<LAST_FRAME>`` must be used
            with ``<FIRST_FRAME>``). The pair renders an interpolation between
            them, and using the same image for both loops the clip.
        reference_image_bytes_list: 1.1 only. Subject/style references, bound
            to ``<IMAGE_REF_0>``, ``<IMAGE_REF_1>``, … in order.
        input_video_bytes: Optional source video (raw bytes) to edit or
            extend. Sent inline, or uploaded through the Files API on 1.1 when
            it is too large to inline.
        reference_video_bytes_list: 1.1 only. Up to 3 clips of up to 3s each,
            bound to ``<VIDEO_REF_0>``, … Any audio in them is ignored.
        previous_interaction_id: Optional id of a prior interaction to
            continue a multi-turn edit or extension (server holds the video
            context).
        task: Optional explicit task
            (text_to_video / image_to_video / reference_to_video / edit, plus
            extend on 1.1). Needed to ask for ``extend`` rather than ``edit``,
            since both are "a prompt plus a source video". Never sent
            alongside ``previous_interaction_id``, which the API rejects.
        aspect_ratio: "16:9" (default) or "9:16". Not sent when continuing
            existing footage, whose source already fixes it.
        duration_seconds: Desired clip length, clamped to the supported
            [3, 10] seconds and sent as "Ns" in response_format; None omits
            it and lets the service choose. Never sent for an edit or an
            extension.
        resolution: 1.1 only. "360p", "720p", "1080p" or "4K" (1080p and 4K
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

    validate_omni_request(
        spec.model,
        resolution=resolution,
        task=task,
        has_first_frame=first_frame_bytes is not None,
        has_last_frame=last_frame_bytes is not None,
        reference_image_count=len(reference_images),
        reference_video_count=len(reference_videos),
        inferred_image_count=len(legacy_images),
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

    task_type = _select_task_type(
        previous_interaction_id=previous_interaction_id,
        input_video_bytes=input_video_bytes,
        inferred_image_count=len(legacy_images),
        reference_image_count=len(reference_images),
        has_first_frame=first_frame_bytes is not None,
        has_last_frame=last_frame_bytes is not None,
        reference_video_count=len(reference_videos),
        requested_task=task,
    )
    is_continuation = (
        task_type in _CONTINUATION_TASKS or previous_interaction_id is not None
    )

    if task_type == _TASK_EXTEND and previous_interaction_id is None:
        if input_video_bytes is None:
            raise ValueError(
                "task='extend' needs something to extend: pass "
                "input_video_bytes (an uploaded clip) or "
                "previous_interaction_id (a clip this server generated)."
            )

    # Clamp duration into the documented [3, 10]s range and send it as "Ns".
    clamped_duration: int | None = None
    if duration_seconds is not None:
        clamped_duration = round(duration_seconds)
        if clamped_duration < _MIN_DURATION:
            clamped_duration = _MIN_DURATION
            warnings.append(
                f"duration_seconds={duration_seconds} below the {_MIN_DURATION}s "
                f"minimum; clamped to {_MIN_DURATION}s."
            )
        elif clamped_duration > _MAX_DURATION:
            clamped_duration = _MAX_DURATION
            warnings.append(
                f"duration_seconds={duration_seconds} above the {_MAX_DURATION}s "
                f"maximum; clamped to {_MAX_DURATION}s."
            )
        if not is_continuation and not spec.duration_is_documented:
            warnings.append(
                f"{spec.model}'s reference documents aspect_ratio, resolution "
                "and delivery in response_format but never duration, so "
                f"duration_seconds={clamped_duration} is sent on the strength "
                "of the SDK schema and the preview model's behaviour, not the "
                "docs. Pass duration_seconds=None to omit it and let the "
                "service pick the length."
            )

    if is_continuation:
        # The API rejects duration (and task alongside previous_interaction_id)
        # on a continuation, so neither duration nor aspect ratio is sent. What
        # the service then renders is NOT the source's length — a measured 3s
        # source came back at 10.01s — and is undocumented, so the warning
        # promises nothing and points at the measured figure instead.
        verb = "Extension" if task_type == _TASK_EXTEND else "Edit"
        warnings.append(
            f"{verb} requests do not send duration_seconds or aspect_ratio — the "
            "API rejects them on a continuation task — so the rendered length is "
            "chosen by the service and is NOT predictable from the request or "
            "from the source video's length. A measured 3s source edited with "
            "duration_seconds=4 rendered 10.01s. The response reports the "
            "duration measured from the rendered file, or that same 10s "
            "maximum as a labelled upper bound when the render is delivered "
            "somewhere it cannot be opened to measure (a gs:// URI)."
        )

    effective_prompt = prompt
    if spec.supports_role_tags and not prompt_declares_media_roles(prompt):
        effective_prompt = _build_media_role_prompt(
            prompt,
            has_first_frame=first_frame_bytes is not None,
            has_last_frame=last_frame_bytes is not None,
            reference_image_count=len(reference_images),
            has_source_video=input_video_bytes is not None,
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

    async def _run_within_deadline(func: Any, /, **kwargs: Any) -> Any:
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise TimeoutError(expired)
        # run_off_loop, not asyncio.to_thread: a timed-out call cannot be
        # cancelled, and abandoning it on the loop's shared default executor
        # burns a worker that every other request also draws from.
        return await run_off_loop(
            functools.partial(func, **kwargs),
            timeout=remaining,
            message=expired,
        )

    def _deadline_expired() -> bool:
        return loop.time() >= deadline

    # Image parts, in the order the generated role declarations name them.
    image_parts: list[dict[str, Any]] = []
    if first_frame_bytes is not None:
        image_parts.append(_image_part(first_frame_bytes))
    if last_frame_bytes is not None:
        image_parts.append(_image_part(last_frame_bytes))
    image_parts.extend(_image_part(img) for img in reference_images)
    image_parts.extend(_image_part(img) for img in legacy_images)

    # Video parts: the source first (it is <VIDEO_0>), then the references.
    video_parts: list[dict[str, Any]] = []
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
        uri_delivery=allow_uri_delivery
        and not output_gcs_uri
        and wants_uri_delivery(spec, normalized_resolution, task_type),
        media_before_text=spec.media_before_text,
    )

    interaction = await _run_within_deadline(
        client.interactions.create, **create_kwargs
    )

    interaction_id = _field(interaction, "id") or _field(interaction, "name")
    if not interaction_id:
        raise ValueError("Interaction create response carried no interaction id.")

    # Poll the background interaction until it leaves the in-flight statuses.
    # `completed` proceeds to extraction; any other terminal status fails fast
    # with the raw status and any error message.
    status = _field(interaction, "status") or _field(interaction, "state")
    # Report state CHANGES and a sparse heartbeat, never one line per poll.
    logged_status: Any = object()
    last_logged_at = loop.time()
    while status in _IN_FLIGHT_STATUSES:
        if _deadline_expired():
            raise TimeoutError(expired)
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
        interaction = await _run_within_deadline(
            client.interactions.get, id=interaction_id
        )
        status = _field(interaction, "status") or _field(interaction, "state")

    if status is not None and status != "completed":
        error = _field(interaction, "error")
        detail = _field(error, "message") if error is not None else None
        raise ValueError(
            f"Omni interaction {interaction_id} ended with status "
            f"'{status}': {detail or 'terminal or unrecognized status'}"
        )

    if log_callback:
        await log_callback("Interaction complete; resolving video output")

    inline_data, uri = _extract_video_payload(interaction)

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
        "model": spec.model,
        "task": task_type,
        # For a continuation the duration was never sent, so reporting the
        # request here would describe a render that did not happen — and the
        # caller bills from this field. None means "unknown here, resolve
        # upstream"; the request is kept separately so nothing is lost.
        "duration_seconds": None if is_continuation else clamped_duration,
        "requested_duration_seconds": clamped_duration,
        # Same property as duration above: _build_create_kwargs omits
        # aspect_ratio on a continuation, so reporting the request here would
        # state a ratio the service never received — editing a 9:16 source
        # under the 16:9 default renders at the source's ratio, not the
        # request's.
        "aspect_ratio": None if is_continuation else aspect_ratio,
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


def omni_extension_output_lengths(
    spec: OmniModelSpec, source_seconds: float | None, times: int
) -> list[float]:
    """Projected OUTPUT length of each turn of an extension chain.

    An extension turn does NOT render only the new tail. Three statements in
    the reference say so together: the cap is on "a total length of 40s"; a
    prompt saying "after 2s cut to a new scene" applied to a 10s source lands
    the cut "after 12s"; and "some of the final frames in your input video
    will be edited to make the transition seamless", which the output can only
    contain if the output contains the input. So turn N comes back as roughly
    source + N steps, growing, until the 40s ceiling clamps it.

    Omni bills per second of OUTPUT video, so quoting every turn at the 10s
    step under-quoted a chain by up to 4x — a 10s source extended twice
    renders 20s and then 30s, not 10s and 10s. This is what a pre-flight
    prices instead. A real run never depends on it: each turn's cost is taken
    from the length measured off the file it produced.

    Args:
        spec: The model's capability record.
        source_seconds: Length of the clip being extended, when known. None
            falls back to the longest source the model documents accepting,
            which is the assumption that quotes lowest without under-quoting a
            chain whose source is already at the limit.
        times: How many turns are planned.

    Returns:
        One projected length per turn, in order.
    """
    current = (
        float(source_seconds)
        if source_seconds is not None
        else float(spec.max_uploaded_source_seconds)
    )
    ceiling = float(spec.max_extended_seconds)
    lengths: list[float] = []
    for _ in range(max(0, times)):
        current = min(current + spec.extension_step_seconds, ceiling)
        lengths.append(current)
    return lengths


def omni_continuation_upper_bound(
    spec: OmniModelSpec, task: str | None, source_seconds: float | None
) -> float:
    """Longest clip a continuation could render, for a cost that cannot be measured.

    A continuation's length is the service's choice, so a render delivered
    somewhere this process cannot open (a gs:// URI) has to be billed at a
    bound. Which bound depends on the kind:

    * ``extend`` returns the whole growing clip, so it is the source plus one
      step, clamped at the model's cumulative ceiling. Billing every extension
      at the 10s per-render maximum under-billed a 4-turn chain roughly 3x —
      that chain renders 20s, 30s, 40s, 40s, not 10s four times.
    * an ``edit`` re-renders the clip it is given. The one measurement in hand
      put a 3s source's edit at 10.01s, i.e. the per-render maximum, so that is
      the floor; a source longer than it (only reachable through an extension
      chain) raises the bound to its own length.

    With no source length to hand, the model's own ceiling is the only honest
    answer — over-stating in a corner that needs both an unmeasurable delivery
    and a source this server never recorded.
    """
    ceiling = float(spec.max_extended_seconds or _MAX_DURATION)
    if task == _TASK_EXTEND and spec.supports_extend:
        # An extension grows the clip, and its source may well be one this
        # server never recorded (an uploaded file), so an unknown source has
        # to assume the ceiling.
        if source_seconds is None:
            return ceiling
        return min(source_seconds + spec.extension_step_seconds, ceiling)
    # An edit re-renders the clip it is given, and the per-render maximum is
    # what the one measurement in hand showed it producing (a 3s source
    # rendered 10.01s). A source longer than that is only reachable by
    # extending, which happens HERE and writes a sidecar — so an unrecorded
    # source is a short one, and assuming the 40s ceiling for it would
    # over-quote every ordinary edit fourfold.
    if source_seconds is None:
        return float(_MAX_DURATION)
    return min(max(float(_MAX_DURATION), source_seconds), ceiling)


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

    Returns:
        The model's spec, so callers do not look it up twice.
    """
    spec = omni_spec(model)
    upgrade = f"Pass model='{OMNI_1_1_MODEL}' to use it."
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
        raise ValueError(
            f"Unknown task '{task}'. Supported: text_to_video, image_to_video, "
            "reference_to_video, edit" + (", extend." if spec.supports_extend else ".")
        )
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
