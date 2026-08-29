"""Video generation helpers."""

import asyncio
import functools
import threading
import uuid
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from stat import S_ISREG
from typing import Any, Literal

from google import genai
from google.genai import types
from PIL import Image

# Type for async log callback from MCP context
LogCallback = Callable[[str], Awaitable[None]]

# Blocking SDK and filesystem work runs here instead of the event loop's shared
# default executor. `asyncio.wait_for` hands control back to the caller but
# cannot cancel a thread already inside a blocking syscall, so every timeout
# parks a worker until the OS releases it. On the shared executor those
# abandoned workers accumulate against the same small pool that image
# generation, ffmpeg and every fetch draw from, and twelve of them wedge the
# whole server. Isolated here, a stalled Veo poll or a dead network mount
# degrades video generation only, and once the pool is saturated new work
# queues behind a deadline instead of hanging the request forever.
_SDK_POOL_MAX_WORKERS = 8
_sdk_pool_lock = threading.Lock()
_sdk_pool: ThreadPoolExecutor | None = None


def _sdk_executor() -> ThreadPoolExecutor:
    """Return the process-wide bounded pool for blocking media SDK calls."""
    global _sdk_pool
    with _sdk_pool_lock:
        if _sdk_pool is None:
            _sdk_pool = ThreadPoolExecutor(
                max_workers=_SDK_POOL_MAX_WORKERS,
                thread_name_prefix="media-sdk",
            )
        return _sdk_pool


async def run_off_loop(
    call: Callable[[], Any],
    *,
    timeout: float,
    message: str,
) -> Any:
    """Run a blocking call in the bounded pool under a hard deadline.

    Shared with ``omni.py`` so both video backends bound their SDK calls the
    same way. ``call`` must be argument-free — bind arguments with
    ``functools.partial`` — so nothing it is given can collide with the
    deadline parameters.

    Raises TimeoutError with ``message`` when the deadline passes. The worker
    thread is NOT cancelled (nothing can cancel a thread blocked in a syscall);
    the point is that the caller and the event loop are released regardless.
    """
    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(_sdk_executor(), call), timeout
        )
    except asyncio.TimeoutError as exc:
        raise TimeoutError(message) from exc


VideoModel = Literal[
    "veo-3.1-generate-001",
    "veo-3.1-fast-generate-001",
    "veo-3.1-lite-generate-preview",
]

# The IDs the Gemini API backend serves under, and the ones a real run REPORTS
# back in its `model` field on that backend. Accepted as tool input purely so a
# returned model round-trips: an agent that reads result["model"] from a
# Gemini-API render and feeds it into loop_extend / generate_video must not hit
# a schema rejection. Kept SEPARATE from VideoModel on purpose — the planner
# enumerates VideoModel as its candidate catalogue (routing.LIVE_VIDEO_MODELS),
# and folding these spellings in would make it rank each model twice. They
# resolve to the canonical `-001` id before anything downstream sees them (see
# _CANONICAL_VIDEO_MODEL_IDS). Same split as ImageModel vs RetiredImageModel.
TranslatedVideoModel = Literal[
    "veo-3.1-generate-preview",
    "veo-3.1-fast-generate-preview",
]

# Veo 3.1 Lite does not support 4K output or video extension.
_VEO_LITE_MODELS = {"veo-3.1-lite-generate-preview"}

# The Gemini Developer API serves Veo 3.1 under `-preview` IDs, while Vertex
# AI uses the `-001` IDs (live-verified: a `-001` call on the Gemini API 404s
# with "not found for API version v1beta"). The public VideoModel values stay
# the `-001` names; they are translated per backend at call time.
_GEMINI_API_MODEL_IDS = {
    "veo-3.1-generate-001": "veo-3.1-generate-preview",
    "veo-3.1-fast-generate-001": "veo-3.1-fast-generate-preview",
}

# Reverse of the per-backend map: a caller may pass back a `-preview` ID that a
# prior Gemini-API render reported. Normalize it to the canonical `-001` name
# BEFORE the per-backend translation runs, so the request behaves identically
# to one that named the canonical model — on Vertex it stays `-001` (a raw
# `-preview` would 404 there), on the Gemini API it re-translates to `-preview`.
_CANONICAL_VIDEO_MODEL_IDS = {
    preview: canonical for canonical, preview in _GEMINI_API_MODEL_IDS.items()
}

# Generation mode for VEO 3.1
GenerationMode = Literal[
    "text_to_video",  # Text-only generation
    "image_to_video",  # First frame image input
    "first_last_frame",  # First and last frame control
    "reference_to_video",  # Reference images for style/character
    "extend_video",  # Extend existing video
]


def _prepare_image_input(image_bytes: bytes) -> types.Image:
    """Convert image bytes to types.Image for API input."""
    pil_img = Image.open(BytesIO(image_bytes))
    fmt = "PNG" if pil_img.mode in ("RGB", "RGBA") else "JPEG"
    if fmt == "JPEG" and pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    buf = BytesIO()
    pil_img.save(buf, format=fmt)
    pil_img.close()
    return types.Image(image_bytes=buf.getvalue(), mime_type=f"image/{fmt.lower()}")


# Generation modes Veo 3.1 Lite cannot serve. It handles text-to-video and
# image-to-video only.
_LITE_UNSUPPORTED_MODES = ("extend_video", "reference_to_video", "first_last_frame")

# MEASURED against the live service: on the Gemini Developer API, Veo refuses
# every mode that conditions on existing footage. first_last_frame comes back
# "Your use case is currently not supported" (generate_transition,
# generate_bridge) and extend_video comes back "encodedVideo isn't supported by
# this model" (loop_extend). The identical calls succeed on Vertex.
#
# reference_to_video joined these after a live test: it does not answer with a
# usable error at all, just an empty result the tool surfaced as "No videos
# returned", and it appears to have billed for the attempt — the worst possible
# failure shape, and the one most worth gating. image_to_video was tested in the
# same round and RENDERS FINE, so the caution once emitted for it is gone: this
# backend is not "text-to-video only", and a warning that fires on a working
# call is noise that teaches callers to ignore warnings.
_GEMINI_API_UNSUPPORTED_MODES = (
    "first_last_frame",
    "extend_video",
    "reference_to_video",
)

_GEMINI_API_MODE_ERRORS = {
    "first_last_frame": (
        'the service answers "Your use case is currently not supported"'
    ),
    "extend_video": (
        'the service answers "encodedVideo isn\'t supported by this model"'
    ),
    "reference_to_video": (
        "the service returns an empty result with no usable error, and bills "
        "for the attempt"
    ),
}

# Veo 3.1 accepts at most this many reference images; extras are not sent.
_MAX_REFERENCE_IMAGES = 3

# Size cap for a local clip handed to extend_video, matching the fetch helper's
# MAX_FETCH_BYTES — the whole file is read into memory and inlined in the
# request, so an unbounded read is an OOM the caller controls.
_EXTEND_MAX_BYTES = 50 * 1024 * 1024

# Deadline for reading that local clip. The media directory is caller-supplied
# and may live on a network mount, so the read is bounded like the sidecar scan.
_EXTEND_READ_TIMEOUT_SECONDS = 30.0

# Overall budget for one render: submit plus the whole polling loop. Measured
# against the monotonic clock so blocked time counts — the previous counter
# only added the sleep interval, so a stalled call could never reach it.
_VEO_TOTAL_TIMEOUT_SECONDS = 1800.0
_VEO_POLL_INTERVAL_SECONDS = 10.0

# Per-call ceilings. google-genai hands timeout=None straight to httpx when
# HttpOptions.timeout is unset, which disables transport timeouts entirely, so
# without these a single stalled connection outlives the request that made it.
_VEO_SUBMIT_TIMEOUT_SECONDS = 120.0
_VEO_POLL_TIMEOUT_SECONDS = 120.0
_VEO_DOWNLOAD_TIMEOUT_SECONDS = 600.0

# Transport deadline (milliseconds — HttpOptions.timeout is in ms) applied to
# the submit call itself. A real transport timeout makes the SDK call *return*
# rather than being abandoned, so unlike run_off_loop it leaks no worker.
# Only the submit call can carry it without changing an SDK call signature;
# see the note on run_off_loop for the rest.
_VEO_SUBMIT_HTTP_TIMEOUT_MS = int(_VEO_SUBMIT_TIMEOUT_SECONDS * 1000)


def _load_extend_video(location: Path, allowed_dir: Path | None) -> types.Video:
    """Read a local clip for extend_video into an inline types.Video.

    Replaces ``types.Video.from_file``, which is a bare
    ``Path(location).read_bytes()`` with no guards at all: a FIFO in the media
    directory blocks that read forever, and a large file is inlined into the
    request whole. The media directory is caller-controlled and may sit on a
    network mount, so this applies the same rules the rest of the server
    already uses — containment in allowed_dir, regular files only, and a size
    cap — before anything is opened.
    """
    resolved = location.resolve()
    if allowed_dir is not None:
        allowed = allowed_dir.resolve()
        if resolved != allowed and not resolved.is_relative_to(allowed):
            # Same wording the fetch-based inputs use, so one confinement
            # violation does not read two different ways depending on which
            # parameter tripped it.
            raise ValueError(
                f"'{location}' is outside the permitted data folder "
                f"(DATA_FOLDER at {allowed}); copy the file under that folder, "
                "or pass a gs:// or https:// URI instead."
            )
    try:
        stat = resolved.stat()
    except OSError as exc:
        raise ValueError(f"Could not read video to extend '{location}': {exc}") from exc
    # A FIFO, socket or device node would block the read indefinitely, and a
    # directory would raise a less obvious error.
    if not S_ISREG(stat.st_mode):
        raise ValueError(f"Video to extend '{location}' is not a regular file.")
    if stat.st_size > _EXTEND_MAX_BYTES:
        raise ValueError(
            f"Video to extend '{location}' is {stat.st_size} bytes, over the "
            f"{_EXTEND_MAX_BYTES} byte limit."
        )
    return types.Video(video_bytes=resolved.read_bytes(), mime_type="video/mp4")


def validate_render_options(
    model: str,
    resolution: str | None = None,
    generation_mode: str | None = None,
    backend: str | None = None,
) -> None:
    """Raise for a model/resolution/mode combination Veo cannot render.

    Shared by the generation path and the tools' dry_run quotes, so a quote
    can never succeed for a call that would be refused — the same
    single-source rule as resolve_image_model on the image side. Every Lite
    restriction lives here; enforcing them per tool is what let a dry run
    price an extension on a model that cannot extend.

    ``backend`` extends that rule from the model to the deployment. Veo on the
    Gemini Developer API refuses every mode that conditions on existing
    footage, and the tools were quoting those calls cleanly — printing the very
    backend that made them impossible in the same response as the price.
    """
    if resolution is not None:
        valid_resolutions = ("720p", "1080p", "4K")
        if resolution not in valid_resolutions:
            raise ValueError(
                f"Unsupported resolution '{resolution}'. "
                f"Supported values are {', '.join(valid_resolutions)}."
            )
        if resolution == "4K" and model in _VEO_LITE_MODELS:
            raise ValueError(
                f"Model {model} does not support 4K resolution. "
                "Use veo-3.1-generate-001 or veo-3.1-fast-generate-001 instead."
            )
        if resolution == "4K" and backend == "gemini_api":
            # Gating covered modes but not resolutions, so this quoted $1.20 at
            # the 4K rate and then 400'd: the highest-value false quote in the
            # server. Veo-specific — omni renders 4K on this backend fine.
            raise ValueError(
                "Veo cannot render 4K on the Gemini Developer API: the service "
                "answers \"The string value 4K for resolution is invalid\". "
                "Render at 720p or 1080p here, or run against Vertex AI. "
                "(Omni does render 4K on this backend.)"
            )
    if (
        generation_mode is not None
        and model in _VEO_LITE_MODELS
        and generation_mode in _LITE_UNSUPPORTED_MODES
    ):
        raise ValueError(
            f"Model {model} does not support {generation_mode}. "
            "Veo 3.1 Lite supports only text-to-video and image-to-video; "
            "use veo-3.1-generate-001 or veo-3.1-fast-generate-001 instead."
        )
    if (
        backend == "gemini_api"
        and generation_mode is not None
        and generation_mode in _GEMINI_API_UNSUPPORTED_MODES
    ):
        detail = _GEMINI_API_MODE_ERRORS.get(generation_mode, "")
        raise ValueError(
            f"Veo cannot render {generation_mode} on the Gemini Developer API"
            + (f": {detail}" if detail else "")
            + ". This mode works on Vertex AI — configure Vertex credentials "
            "(GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION, with "
            "output_gcs_uri or VIDEO_GCS_BUCKET for delivery) and retry. "
            "Text-to-video works on this backend."
        )


def validate_video_request_inputs(
    *,
    prompt: str,
    has_first_frame: bool,
    has_last_frame: bool,
    reference_count: int,
    has_extend: bool,
) -> None:
    """Raise for a request no render could honor, before any work is spent.

    Shared by the generation impl and the tools' dry_run quotes so a quote can
    never succeed for a call the real run would refuse — the mode ladder below
    silently picks one input and drops the rest, so an unenforced conflict
    billed a full render that ignored half of what the caller supplied.
    ``has_*`` are presence flags (URI or inline bytes) since the tool front
    door has URIs, not bytes.
    """
    if not prompt or not prompt.strip():
        raise ValueError("prompt must be a non-empty description of the video.")

    frame_names = []
    if has_first_frame:
        frame_names.append("image_uri/image_base64")
    if has_last_frame:
        frame_names.append("last_frame_uri/last_frame_base64")

    if has_extend and (has_first_frame or has_last_frame or reference_count):
        others = list(frame_names)
        if reference_count:
            others.append("reference_image_uris")
        joined = others[0] if len(others) == 1 else ", ".join(others)
        raise ValueError(
            f"extend_video_uri cannot be combined with {joined}. An extension "
            "continues the source clip and sends no image inputs; drop the "
            "image inputs or drop extend_video_uri."
        )
    if reference_count and (has_first_frame or has_last_frame):
        joined = frame_names[0] if len(frame_names) == 1 else ", ".join(frame_names)
        raise ValueError(
            f"reference images cannot be combined with {joined}. "
            "Reference-to-video sends neither a first nor a last frame and "
            "always renders 8 seconds; drop one or the other."
        )
    if has_last_frame and not has_first_frame:
        raise ValueError(
            "A last frame was provided without a first frame. First+last "
            "frame mode requires both; provide image_uri/image_base64 too."
        )


def resolve_served_video_model(model: str, is_vertexai: bool) -> str:
    """The model id a real run reports for ``model`` on this backend.

    A caller may pin a canonical `-001` id or feed back the `-preview` id a
    prior Gemini-API render returned; either way the served id is derived the
    same way the impl derives it, so a dry_run's `model` field matches what the
    real run will report instead of echoing the raw input.
    """
    canonical = _CANONICAL_VIDEO_MODEL_IDS.get(model, model)
    if not is_vertexai:
        return _GEMINI_API_MODEL_IDS.get(canonical, canonical)
    return canonical


async def generate_video(
    client: genai.Client,
    prompt: str,
    videos_dir: Path,
    model: VideoModel | TranslatedVideoModel = "veo-3.1-generate-001",
    image_bytes: bytes | None = None,
    allowed_dir: Path | None = None,
    aspect_ratio: str = "16:9",
    duration_seconds: float = 5.0,
    include_audio: bool = False,
    audio_prompt: str | None = None,
    negative_prompt: str | None = None,
    seed: int | None = None,
    log_callback: LogCallback | None = None,
    last_frame_bytes: bytes | None = None,
    reference_images: list[bytes] | None = None,
    extend_video_uri: str | None = None,
    resolution: str | None = None,
    person_generation: str | None = None,
    output_gcs_uri: str | None = None,
) -> dict[str, Any]:
    """Generate a video using VEO models.

    Args:
        client: Google GenAI client
        prompt: Text description of the video to generate
        videos_dir: Directory to save generated videos
        model: VEO model to use
        image_bytes: First frame image bytes for image-to-video
        aspect_ratio: Video aspect ratio (16:9 or 9:16)
        duration_seconds: Video duration (4/6/8s)
        include_audio: Enable audio generation
        audio_prompt: Audio description
        negative_prompt: Things to avoid in the video
        seed: Random seed for reproducibility
        log_callback: Async callback for progress logging
        last_frame_bytes: Last frame image bytes for first+last frame control
        reference_images: List of reference image bytes (up to 3) for style/character
        allowed_dir: Directory that file:// / bare-path extend sources must
            resolve inside. Security boundary, not a convenience: without it
            any local path readable by the server could be uploaded to the
            API as "video to extend".
        extend_video_uri: URI of existing VEO video to extend. On Vertex AI
            this requires output_gcs_uri; on the Gemini API the extended
            clip is returned inline and GCS output is not supported.
        resolution: Output resolution ("720p" or "1080p"; "4K" only for non-Lite
            models). When None, the API default is used.
        person_generation: Person generation policy ("allow_adult" or "allow_all").
            Passed through to the API for validation. When None, the API default
            is used.
        output_gcs_uri: GCS URI for output (required for extensions and large
            videos). Only supported in Vertex AI mode.

    Returns:
        Dictionary with video_url and generation metadata
    """
    model_id = str(model)

    # Normalize a served `-preview` ID a caller fed back from a prior render to
    # its canonical `-001` name first, so the per-backend translation below is
    # the single source of the served spelling.
    model_id = _CANONICAL_VIDEO_MODEL_IDS.get(model_id, model_id)

    # Translate model IDs per backend: the Gemini Developer API serves Veo
    # under `-preview` IDs and 404s on the Vertex `-001` names.
    is_vertexai = getattr(client._api_client, "vertexai", False)
    if not is_vertexai:
        model_id = _GEMINI_API_MODEL_IDS.get(model_id, model_id)

    # Non-fatal warnings surfaced back to the caller (e.g. a request that could
    # not be honored but should not abort the whole generation).
    warnings: list[str] = []

    # Inputs that cannot all be honored are a hard error, not a silent pick:
    # the mode ladder below chooses exactly one and drops the rest after the
    # caller has paid to fetch them. Shared with the tools' dry_run so a quote
    # refuses exactly what the render refuses.
    validate_video_request_inputs(
        prompt=prompt,
        has_first_frame=bool(image_bytes),
        has_last_frame=bool(last_frame_bytes),
        reference_count=len(reference_images) if reference_images else 0,
        has_extend=bool(extend_video_uri),
    )

    # Determine generation mode based on inputs
    generation_mode: str = "text_to_video"
    if extend_video_uri:
        generation_mode = "extend_video"
    elif reference_images:
        generation_mode = "reference_to_video"
    elif image_bytes and last_frame_bytes:
        generation_mode = "first_last_frame"
    elif image_bytes:
        generation_mode = "image_to_video"

    # Veo 3.1 Lite (served via the Gemini API) does not support video
    # extension, reference images, or first/last-frame control — fail fast
    # with a clear message instead of an opaque API error. Shared with the
    # tools' dry_run quotes so both refuse the same combinations.
    validate_render_options(
        model,
        generation_mode=generation_mode,
        backend="vertex" if is_vertexai else "gemini_api",
    )

    # Prepare image inputs
    first_frame_input: types.Image | None = None
    last_frame_input: types.Image | None = None
    reference_image_inputs: list[types.VideoGenerationReferenceImage] = []

    if generation_mode == "image_to_video" and image_bytes:
        first_frame_input = _prepare_image_input(image_bytes)
    elif generation_mode == "first_last_frame":
        if image_bytes:
            first_frame_input = _prepare_image_input(image_bytes)
        if last_frame_bytes:
            last_frame_input = _prepare_image_input(last_frame_bytes)
    elif generation_mode == "reference_to_video" and reference_images:
        # VEO 3.1 supports up to 3 reference images (asset type)
        # Must wrap in VideoGenerationReferenceImage with reference_type="asset"
        if len(reference_images) > _MAX_REFERENCE_IMAGES:
            # A truncation, not a conflict: the render the caller asked for
            # still happens, just without the extras. Say so rather than
            # letting a reference the caller explicitly supplied vanish and
            # quietly change the generation result.
            warnings.append(
                f"{len(reference_images)} reference images were supplied but "
                f"Veo 3.1 accepts {_MAX_REFERENCE_IMAGES}; the last "
                f"{len(reference_images) - _MAX_REFERENCE_IMAGES} were not "
                "sent and did not influence this render."
            )
        for ref_bytes in reference_images[:_MAX_REFERENCE_IMAGES]:
            ref_image = _prepare_image_input(ref_bytes)
            reference_image_inputs.append(
                types.VideoGenerationReferenceImage(
                    image=ref_image,
                    reference_type="asset",  # asset for subject preservation
                )
            )

    # Aspect ratio must match source clips for transitions/bridges, so an
    # unsupported value is a hard error rather than a silent coercion.
    if aspect_ratio not in ("16:9", "9:16"):
        raise ValueError(
            f"Unsupported aspect_ratio '{aspect_ratio}'. "
            "Supported values are '16:9' and '9:16'."
        )

    config_kwargs: dict[str, Any] = {
        "number_of_videos": 1,
        "aspect_ratio": aspect_ratio,
    }

    # Compute the effective (snapped/forced) duration so it can be both sent to
    # the API and reported back to callers.
    if generation_mode == "reference_to_video":
        # Reference-to-video only supports 8 seconds.
        effective_duration = 8
    elif generation_mode == "extend_video":
        # Extend video requires exactly 7 seconds output.
        effective_duration = 7
    else:
        allowed = [4, 6, 8]
        effective_duration = min(allowed, key=lambda x: abs(x - duration_seconds))
    config_kwargs["duration_seconds"] = effective_duration
    # generate_audio is only supported on Vertex AI. Veo 3.1 already applies prompt
    # rewriting automatically, so `enhance_prompt` is Veo-2-only and must not be sent.
    # (is_vertexai computed at the top alongside model-ID translation.)
    if is_vertexai:
        # Send the flag explicitly BOTH ways: omitting it lets the API apply
        # its own default (audio on for Veo 3.1), which would silently
        # contradict include_audio=False and the reported audio_enabled.
        config_kwargs["generate_audio"] = include_audio

    # On the Gemini API path, generate_audio is never sent and Veo 3.1 always
    # generates audio natively. A caller who explicitly asked for NO audio
    # (include_audio=False, the default) cannot have that honored here, so warn
    # rather than silently returning a clip with baked-in audio. (include_audio=True
    # is satisfied since audio is produced, so no warning is needed there.)
    if not is_vertexai and not include_audio:
        warnings.append(
            "include_audio=False was not honored: Veo 3.1 on the Gemini API "
            "always generates audio. Use Vertex AI mode to control audio."
        )

    # Add last frame to config for first+last frame mode
    if last_frame_input:
        config_kwargs["last_frame"] = last_frame_input

    # Add reference images to config for VEO 3.1
    if reference_image_inputs:
        config_kwargs["reference_images"] = reference_image_inputs

    if negative_prompt:
        config_kwargs["negative_prompt"] = negative_prompt
    if seed is not None and seed >= 0:
        config_kwargs["seed"] = seed
    elif seed is not None:
        # Dropping it is right (Veo seeds are non-negative) but doing it
        # silently costs the caller the one thing a seed buys: they would
        # re-run the same request expecting the same clip and never learn why
        # it differs.
        warnings.append(
            f"seed={seed} was not sent: Veo seeds must be non-negative, so "
            "this render is not reproducible."
        )

    # output_gcs_uri is a Vertex-only config field. The Gemini API (e.g. for
    # Veo Lite) does not support GCS output, so only forward it on Vertex.
    # Callers may pass a default bucket (VIDEO_GCS_BUCKET) that applies to
    # Vertex runs; silently ignoring it on the Gemini API path keeps Lite and
    # plain text-to-video working. The generate_video tool separately rejects
    # an *explicit* output_gcs_uri on a Gemini-API-routed call.
    if output_gcs_uri and is_vertexai:
        config_kwargs["output_gcs_uri"] = output_gcs_uri

    if resolution is not None:
        validate_render_options(model, resolution)
        config_kwargs["resolution"] = resolution

    if person_generation is not None:
        # Pass through as-is and let the API validate the value.
        config_kwargs["person_generation"] = person_generation

    # A real transport deadline on the submit call, so a stalled connection
    # raises instead of parking a worker thread forever. Merged field-wise
    # over the client's own HttpOptions, so base_url/api_version survive.
    config_kwargs["http_options"] = types.HttpOptions(
        timeout=_VEO_SUBMIT_HTTP_TIMEOUT_MS
    )

    prompt_for_api = prompt
    if audio_prompt:
        prompt_for_api = f"{prompt}\nAudio: {audio_prompt}"

    video_config = types.GenerateVideosConfig(**config_kwargs)

    if log_callback:
        mode_desc = generation_mode.replace("_", " ")
        await log_callback(f"Starting {mode_desc} with {model_id}")

    # Build API call based on generation mode
    api_kwargs: dict[str, Any] = {
        "model": model_id,
        "prompt": prompt_for_api,
        "config": video_config,
    }

    if generation_mode == "image_to_video" and first_frame_input:
        api_kwargs["image"] = first_frame_input
    elif generation_mode == "first_last_frame" and first_frame_input:
        # First frame as image param, last frame in config (already added above)
        api_kwargs["image"] = first_frame_input
    elif generation_mode == "extend_video" and extend_video_uri:
        # Video extension for VEO 3.1
        # For file:// URIs, load from local file to get proper mime type
        if extend_video_uri.startswith("file://"):
            local_path = Path(extend_video_uri[7:])
            api_kwargs["video"] = await run_off_loop(
                functools.partial(_load_extend_video, local_path, allowed_dir),
                timeout=_EXTEND_READ_TIMEOUT_SECONDS,
                message=(
                    f"Timed out reading the video to extend '{local_path}' "
                    f"after {_EXTEND_READ_TIMEOUT_SECONDS:.0f}s."
                ),
            )
        else:
            # Remote URI - pass with mime type
            api_kwargs["video"] = types.Video(
                uri=extend_video_uri, mimeType="video/mp4"
            )

    # The whole render is measured against the monotonic clock, so time spent
    # blocked inside an SDK call counts toward the budget. The old counter only
    # accumulated the sleep interval, which meant a stalled call could never
    # reach the limit and the request hung indefinitely.
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _VEO_TOTAL_TIMEOUT_SECONDS
    expired = f"Video generation timed out after {_VEO_TOTAL_TIMEOUT_SECONDS:.0f}s."

    def _remaining() -> float:
        left = deadline - loop.time()
        if left <= 0:
            raise TimeoutError(expired)
        return left

    operation = await run_off_loop(
        functools.partial(client.models.generate_videos, **api_kwargs),
        timeout=min(_VEO_SUBMIT_TIMEOUT_SECONDS, _remaining()),
        message="Video generation timed out submitting the request.",
    )

    if log_callback:
        await log_callback(f"Polling operation: {operation.name}")
    while not operation.done:
        await asyncio.sleep(min(_VEO_POLL_INTERVAL_SECONDS, _remaining()))
        operation = await run_off_loop(
            functools.partial(client.operations.get, operation),
            timeout=min(_VEO_POLL_TIMEOUT_SECONDS, _remaining()),
            message="Video generation timed out polling the operation.",
        )

    if operation.error:
        raise ValueError(f"VEO error: {operation.error}")

    result = getattr(operation, "response", None) or getattr(operation, "result", None)
    if not result or not getattr(result, "generated_videos", None):
        raise ValueError("No videos returned")

    video = result.generated_videos[0].video

    if hasattr(video, "uri") and video.uri and video.uri.startswith("gs://"):
        video_url = video.uri
    elif hasattr(video, "video_bytes") and video.video_bytes:
        filename = f"{uuid.uuid4()}.mp4"
        filepath = videos_dir / filename
        filepath.write_bytes(video.video_bytes)
        video_url = f"file://{filepath}"
    else:
        await run_off_loop(
            functools.partial(client.files.download, file=video),
            timeout=min(_VEO_DOWNLOAD_TIMEOUT_SECONDS, _remaining()),
            message="Timed out downloading the generated video.",
        )
        filename = f"{uuid.uuid4()}.mp4"
        filepath = videos_dir / filename
        await run_off_loop(
            functools.partial(video.save, str(filepath)),
            timeout=min(_VEO_DOWNLOAD_TIMEOUT_SECONDS, _remaining()),
            message="Timed out writing the generated video to disk.",
        )
        video_url = f"file://{filepath}"

    # Report audio truthfully: on Vertex AI it depends on the generate_audio flag
    # (== include_audio), but on the Gemini API path Veo 3.1 always generates
    # audio natively regardless of the include_audio input.
    audio_enabled = include_audio if is_vertexai else True

    result = {
        "message": "Video generated successfully",
        "video_url": video_url,
        "prompt": prompt_for_api,
        "model": model_id,
        "audio_enabled": audio_enabled,
        "duration_seconds": effective_duration,
        "generation_mode": generation_mode,
    }

    # For extend_video mode, also return the extended video URI
    if generation_mode == "extend_video" and extend_video_uri:
        result["extended_from"] = extend_video_uri

    # Include warnings only when non-empty, so clean runs keep tidy manifests.
    if warnings:
        result["warnings"] = warnings

    return result


VEO_EXTENSION_STEP_SECONDS = 7.0
# Veo caps a chain at 20 extensions; the tool exposes the same range.
VEO_MAX_EXTENSIONS = 20


def veo_extension_output_lengths(
    source_seconds: float | None,
    times: int,
    step: float = VEO_EXTENSION_STEP_SECONDS,
) -> list[float]:
    """Billable OUTPUT length of each turn of a Veo extension chain.

    MEASURED: a 4.0s source extended once returned an 11.0s file billed at
    11s, not at the 7s it appended. Veo re-bills the assembled clip on every
    turn, exactly as omni was measured to do, so turn i outputs
    ``source + step*i`` and the cost of a chain grows QUADRATICALLY in the
    number of turns.

    This was quoted as ``times * 7`` — the appended footage alone — which
    under-billed a single extension of a 4s source by 57% and a 20-turn chain
    by an order of magnitude ($56 quoted against $620 actual on
    veo-3.1-generate-001). The three numbers are distinct and conflating them
    is the whole bug: what gets BILLED is the sum of these lengths, what the
    caller ENDS UP WITH is the last one, and what is NEW is only ``times *
    step``.

    Returns [] when the source length is unknown, because there is no honest
    point estimate without it — the caller must say so rather than quoting the
    floor as if it were the price.
    """
    if source_seconds is None or source_seconds <= 0:
        return []
    return [round(source_seconds + step * i, 3) for i in range(1, times + 1)]
