"""Image generation helpers."""

import asyncio
import base64
import logging
import math
import uuid
from collections.abc import Mapping, Sequence
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

from google import genai
from google.auth import exceptions as google_auth_exceptions
from google.genai import types
from PIL import Image

logger = logging.getLogger(__name__)

ImageModel = Literal[
    # Gemini 3.x image models, GA under suffix-less IDs
    "gemini-3-pro-image",
    "gemini-3.1-flash-image",
    "gemini-3.1-flash-lite-image",
]

# IDs whose endpoints are gone (or imminently going). Calls to these fail with
# 404, so generate_image() reroutes them rather than letting the request die —
# see _MODEL_SHUTDOWNS. They are deliberately split out of ImageModel, the live
# catalog, but MUST stay in the MCP tool annotation: pydantic validates the
# model argument against it, so dropping them from the schema would reject a
# pinned caller's request with a validation error before the shim could run.
RetiredImageModel = Literal[
    # Imagen — every image endpoint discontinued 2026-08-17
    "imagen-3.0-capability-001",
    "imagen-3.0-capability-002",
    "imagen-3.0-fast-generate-001",
    "imagen-3.0-generate-001",
    "imagen-3.0-generate-002",
    "imagen-4.0-fast-generate-001",
    "imagen-4.0-generate-001",
    "imagen-4.0-ultra-generate-001",
    # Nano Banana 2 / Pro preview aliases — retired 2026-06-25, already dead
    "gemini-3-pro-image-preview",
    "gemini-3.1-flash-image-preview",
    # Nano Banana 1 — still served, shutdown scheduled 2026-10-02
    "gemini-2.5-flash-image",
]

# Superseded ID -> (replacement, shutdown date). One table, because whether a
# shutdown is past or future is a function of today's date, not a property to
# hard-code: classifying imagen-4.0-* as "retired" was correct on 2026-08-18
# and wrong on 2026-08-07, when it still had ten days of service left. Dates
# come from Google's published deprecation table.
_MODEL_SHUTDOWNS: dict[str, tuple[str, str]] = {
    # Imagen — every image endpoint discontinued 2026-08-17...
    "imagen-3.0-capability-001": ("gemini-3.1-flash-image", "2026-08-17"),
    "imagen-3.0-capability-002": ("gemini-3.1-flash-image", "2026-08-17"),
    "imagen-3.0-fast-generate-001": ("gemini-3.1-flash-image", "2026-08-17"),
    "imagen-3.0-generate-001": ("gemini-3.1-flash-image", "2026-08-17"),
    # ...except this one, which Google shut down early.
    "imagen-3.0-generate-002": ("gemini-3.1-flash-image", "2025-11-10"),
    "imagen-4.0-fast-generate-001": ("gemini-3.1-flash-image", "2026-08-17"),
    "imagen-4.0-generate-001": ("gemini-3.1-flash-image", "2026-08-17"),
    # Ultra is the top Imagen tier, so it maps to the top Gemini image model
    # rather than dropping to flash — Google's table permits either.
    "imagen-4.0-ultra-generate-001": ("gemini-3-pro-image", "2026-08-17"),
    # Nano Banana 2 / Pro preview aliases
    "gemini-3-pro-image-preview": ("gemini-3-pro-image", "2026-06-25"),
    "gemini-3.1-flash-image-preview": ("gemini-3.1-flash-image", "2026-06-25"),
    # Nano Banana 1
    "gemini-2.5-flash-image": ("gemini-3.1-flash-image", "2026-10-02"),
}

# Fallback for a superseded-looking ID absent from the table (e.g. a regional
# or newly-surfaced Imagen variant) — better than a guaranteed 404.
_SUPERSEDED_DEFAULT_TARGET = "gemini-3.1-flash-image"
_SUPERSEDED_DEFAULT_SHUTDOWN = "2026-08-17"


def _shutdown_phrase(shutdown: str, today: date | None = None) -> str:
    """Describe a shutdown in the right tense for today.

    A model with a future shutdown date still serves; saying it "was retired
    and no longer exists" is wrong, and would tell a Provisioned Throughput
    holder their model is already dead when they still have time to migrate.
    """
    now = today or date.today()
    try:
        when = date.fromisoformat(shutdown)
    except ValueError:  # pragma: no cover - table is hand-maintained
        return f"is superseded (shutdown {shutdown})"
    if when <= now:
        return f"was retired on {shutdown} and no longer exists"
    return f"is scheduled for shutdown on {shutdown}"


# Output sizes a model can actually produce. Only models with a restriction are
# listed; anything absent accepts the full ImageSize range. gemini-3.1-flash-
# lite-image is 1K-only — 2K and 4K are documented as unsupported.
_IMAGE_SIZE_SUPPORT: dict[str, frozenset[str]] = {
    "gemini-3.1-flash-lite-image": frozenset({"1K"}),
}


def _field(obj: Any, *names: str) -> Any:
    """Read the first present field from an SDK object or a plain mapping.

    Duck-typed on purpose: the SDK returns objects, but a REST-shaped dict and
    a hand-rolled double must read the same way, and a missing field must never
    raise.
    """
    for name in names:
        value = obj.get(name) if isinstance(obj, Mapping) else getattr(obj, name, None)
        if value is not None:
            return value
    return None


def _modality_details(details: Any) -> list[dict[str, Any]]:
    """Flatten a ``*_tokens_details`` list into plain, JSON-safe entries.

    ``src.pricing`` prices an image call exactly when the response says how the
    output split across modalities. Dropping these lists forced every live call
    onto the fallback heuristic, which attributes only the table's image-token
    count and bills every remaining candidate token at the TEXT rate — a 20x
    understatement whenever a response carries more than one image part (a
    thinking model's interim renders do exactly that). The shape emitted here
    (``modality`` as a string, ``token_count`` as an int) is what
    ``pricing._modality_tokens`` reads.
    """
    if not isinstance(details, Sequence) or isinstance(details, (str, bytes)):
        return []
    entries: list[dict[str, Any]] = []
    for entry in details:
        modality = _field(entry, "modality")
        tokens = _field(entry, "token_count", "tokenCount")
        if modality is None or not isinstance(tokens, int) or isinstance(tokens, bool):
            continue
        # Modality arrives as an enum on the SDK path; str(enum) would serialize
        # as "MediaModality.IMAGE", which pricing matches but reads badly in the
        # response payload, so unwrap .value when it has one.
        entries.append(
            {
                "modality": str(getattr(modality, "value", modality)),
                "token_count": tokens,
            }
        )
    return entries


def _usage_dict(response: Any) -> dict[str, Any] | None:
    """Extract token counts from a response's usage_metadata as a plain dict.

    Returned to the caller so cost can be computed from what the API actually
    metered rather than from a pre-flight estimate. Kept as plain data (ints
    and lists of small dicts) because the result travels through
    ``json.dumps`` in the MCP layer, which cannot serialize the SDK's usage
    object.
    """
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return None
    fields = (
        "prompt_token_count",
        "candidates_token_count",
        "total_token_count",
        "thoughts_token_count",
        "cached_content_token_count",
    )
    out: dict[str, Any] = {}
    for field in fields:
        value = getattr(usage, field, None)
        if isinstance(value, int):
            out[field] = value
    # The per-modality breakdown is the only thing that lets pricing bill image
    # output at the image rate instead of guessing, so it has to survive the
    # trip to plain data.
    for field, alias in (
        ("prompt_tokens_details", "promptTokensDetails"),
        ("candidates_tokens_details", "candidatesTokensDetails"),
    ):
        entries = _modality_details(_field(usage, field, alias))
        if entries:
            out[field] = entries
    return out or None


def _supports_image_size(model_id: str, image_size: str) -> bool:
    """Whether ``model_id`` can produce ``image_size``.

    Models absent from ``_IMAGE_SIZE_SUPPORT`` have no documented restriction
    and are assumed to accept the full range.
    """
    supported = _IMAGE_SIZE_SUPPORT.get(model_id)
    return supported is None or image_size in supported


# Gemini 3.x image models that share enhanced capabilities (image_config, up to
# 14 reference images, global-location Vertex). The retired -preview aliases are
# absent by design: reroute rewrites them to these GA IDs before any lookup.
_GEMINI3_IMAGE_MODELS = {
    "gemini-3-pro-image",
    "gemini-3.1-flash-image",
    "gemini-3.1-flash-lite-image",
}

# Decompression-bomb guards for input images. The byte caps upstream bound what
# arrives on the wire, which says nothing about what it costs to DECODE: a solid
# 9999x9999 PNG is 316KB encoded and ~500MB in memory, and fourteen references
# like that took the server to 2.8GB RSS because every decoded frame stays alive
# until generate_content returns. Pillow's own MAX_IMAGE_PIXELS does not bound
# this either — it only warns until twice its limit.
#
# Ceiling on the SOURCE pixel count, checked from the header before any pixels
# are decoded. Roughly 6300x6300, and deliberately under the 50MP Pillow limit
# the server installs so a permitted image never trips the bomb warning.
_MAX_SOURCE_PIXELS = 40_000_000
# What is RETAINED per image. Gemini downsamples image input to a few hundred
# thousand pixels before tokenizing it, so shrinking a larger source costs
# nothing the model can see while capping each held frame at ~12MB.
_MAX_DECODED_PIXELS = 4_000_000
# Ceiling on the retained pixels of a whole request. An edit input plus the
# fourteen references Gemini 3.x allows, each at the full per-image budget,
# stay under this — the documented workflow cannot trip it. It is here so that
# raising the per-image budget cannot silently reintroduce the batch blow-up.
_MAX_TOTAL_DECODED_PIXELS = 64_000_000


def _megapixels(pixels: float) -> str:
    """Format a pixel count for an error message."""
    return f"{pixels / 1_000_000:.1f}MP"


def _open_input_image(data: bytes, label: str, budget: int) -> Image.Image:
    """Decode one input image within the decompression-bomb budget.

    Args:
        data: Encoded image bytes.
        label: How this image is named in an error, e.g. "Reference image 3".
        budget: Retained-pixel allowance left for the rest of the request.

    Returns:
        A decoded image, downscaled when the source exceeds the per-image
        budget. The caller owns it and must close it.

    Raises:
        ValueError: If the source is too large to decode, or the request's
            total decoded budget is exhausted.
    """
    image = Image.open(BytesIO(data))
    try:
        width, height = image.size
        pixels = width * height
        # Checked before .load(): the point is to never materialize the bitmap.
        if pixels > _MAX_SOURCE_PIXELS:
            raise ValueError(
                f"{label} is {width}x{height} ({_megapixels(pixels)}), above the "
                f"{_megapixels(_MAX_SOURCE_PIXELS)} limit for input images. "
                "Downscale it before sending."
            )
        if pixels > _MAX_DECODED_PIXELS:
            scale = math.sqrt(_MAX_DECODED_PIXELS / pixels)
            target = (max(1, int(width * scale)), max(1, int(height * scale)))
            # thumbnail() drafts JPEG decoding down inside the decoder and
            # replaces the bitmap in place, so the oversized copy never sits
            # alongside the small one.
            image.thumbnail(target, Image.Resampling.LANCZOS)
        else:
            image.load()
        retained = image.width * image.height
        if retained > budget:
            raise ValueError(
                f"{label} ({image.width}x{image.height}) does not fit this "
                f"request's remaining decoded-image budget of "
                f"{_megapixels(budget)} (total "
                f"{_megapixels(_MAX_TOTAL_DECODED_PIXELS)}). Send fewer or "
                "smaller reference images."
            )
    except BaseException:
        image.close()
        raise
    return image


def _prepare_input_images(
    image_bytes: bytes | None,
    reference_images: list[bytes] | None,
    max_refs: int,
) -> list[Image.Image]:
    """Decode the edit input and reference images under the pixel budgets.

    Raises:
        ValueError: If any image blows a budget. Everything already decoded is
            closed first, so a rejected batch does not leak the frames that
            preceded it.
    """
    sources: list[tuple[str, bytes]] = []
    if image_bytes:
        sources.append(("The input image", image_bytes))
    if reference_images:
        for position, ref_bytes in enumerate(reference_images[:max_refs], start=1):
            sources.append((f"Reference image {position}", ref_bytes))

    images: list[Image.Image] = []
    remaining = _MAX_TOTAL_DECODED_PIXELS
    try:
        for label, data in sources:
            image = _open_input_image(data, label, remaining)
            images.append(image)
            remaining -= image.width * image.height
    except BaseException:
        for image in images:
            image.close()
        raise
    return images


# Lazily-created, module-level cached Vertex AI client pinned to the "global"
# location. Gemini 3.x image models require the global location on Vertex; this
# avoids re-creating a brand-new client on every call.
_vertex_global_client: genai.Client | None = None


def _get_vertex_global_client() -> genai.Client:
    """Return a memoized Vertex AI client pinned to the global location.

    Gemini 3.x image models require ``location="global"`` on Vertex AI. The
    client is created lazily on first use and reused on subsequent calls.
    """
    global _vertex_global_client
    if _vertex_global_client is None:
        _vertex_global_client = genai.Client(vertexai=True, location="global")
    return _vertex_global_client


# Output image size options for Gemini 3.x Image models
# Must use uppercase K (1K, 2K, 4K)
ImageSize = Literal["1K", "2K", "4K"]

# Media resolution options for input processing
# Valid values are the enum values from google.genai.types.MediaResolution
MediaResolution = Literal[
    "MEDIA_RESOLUTION_LOW",
    "MEDIA_RESOLUTION_MEDIUM",
    "MEDIA_RESOLUTION_HIGH",
]


def resolve_image_model(
    model: str,
    image_size: ImageSize | None = None,
) -> tuple[str, list[str], ImageSize | None]:
    """Resolve a requested model to the one that will actually be called.

    Single source of truth for model substitution, shared by the real
    generation path, the ``dry_run`` estimate and the intent router, so a
    quoted price always describes the call that would really be issued.

    Pure: it reports substitutions through the returned warnings and does not
    log, because most callers never issue a request.

    Args:
        model: The requested model ID, which may be retired or superseded.
        image_size: Requested output size, if any.

    Returns:
        ``(model_id, warnings, effective_image_size)``. ``effective_image_size``
        is None when the requested size cannot be produced by the resolved
        model, in which case a warning explains the drop.
    """
    model_id = str(model)
    warnings: list[str] = []

    # Reroute retired endpoints to their replacement. These IDs 404, so issuing
    # the call as-is is never the right behaviour; the substitution is reported
    # so callers can update their own configuration. An unlisted imagen-* ID
    # falls back to the GA replacement rather than a guaranteed failure.
    if model_id in _MODEL_SHUTDOWNS or model_id.startswith("imagen"):
        target, shutdown = _MODEL_SHUTDOWNS.get(
            model_id, (_SUPERSEDED_DEFAULT_TARGET, _SUPERSEDED_DEFAULT_SHUTDOWN)
        )
        state = _shutdown_phrase(shutdown)
        warnings.append(
            f"Model {model_id} {state}; {target} served this request instead. "
            f"Update your configuration to request {target} directly."
        )
        model_id = target

    # Drop an output size the resolved model cannot produce. Warn rather than
    # fail: the caller picked this model explicitly, so substituting a
    # different one would be the bigger surprise.
    if image_size and not _supports_image_size(model_id, image_size):
        supported_sizes = _IMAGE_SIZE_SUPPORT[model_id]
        warnings.append(
            f"{model_id} does not support image_size={image_size} "
            f"(supported: {', '.join(sorted(supported_sizes))}); the "
            "request was sent at the model's default size. Use "
            "gemini-3.1-flash-image or gemini-3-pro-image for "
            "higher-resolution output."
        )
        image_size = None

    return model_id, warnings, image_size


async def generate_image(
    client: genai.Client,
    prompt: str,
    images_dir: Path,
    model: ImageModel | RetiredImageModel | str = "gemini-3.1-flash-image",
    image_bytes: bytes | None = None,
    reference_images: list[bytes] | None = None,
    image_size: ImageSize | None = None,
    media_resolution: MediaResolution | None = None,
    aspect_ratio: str | None = None,
    person_generation: str | None = None,
    thought_signature: str | None = None,
    conversation_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Generate an image using Gemini image models.

    Supported models (``ImageModel``):
      - gemini-3.1-flash-image (default), gemini-3.1-flash-lite-image,
        gemini-3-pro-image

    Superseded IDs (``RetiredImageModel``) are still accepted so pinned callers
    keep working, and are rerouted to the replacement Google published:
      - the Imagen family — endpoints discontinued 2026-08-17
      - the -preview image aliases — gone 2026-06-25
      - gemini-2.5-flash-image — still served, shutdown 2026-10-02
        (see ``_MODEL_SHUTDOWNS``)

    Every substitution is reported in the returned ``warnings`` list and
    logged at WARNING.

    Args:
        client: Google GenAI client
        prompt: Text description of the image to generate
        images_dir: Directory to save generated images
        model: Model to use for generation. Typed to include ``str`` because
            an unrecognised ``imagen-*`` variant is deliberately rerouted
            rather than rejected; the MCP tool keeps the strict union so the
            published schema still advertises only real models.
        image_bytes: Input image bytes for editing
        reference_images: List of reference image bytes (up to 14 for Gemini 3.x image models)
        image_size: Output image size (1K, 2K, 4K) - must use uppercase K
        media_resolution: Input image resolution processing (low/medium/high)
        aspect_ratio: Desired output aspect ratio (e.g. "1:1", "16:9", "9:16").
            Applied via ImageConfig. Passed through as-is; the API validates it.
        person_generation: Policy for generating people. Valid values:
            "dont_allow", "allow_adult", "allow_all". Applied via ImageConfig.
            Passed through as-is; the API validates it.
        thought_signature: Thought signature from previous turn for multi-turn editing
        conversation_history: Previous conversation history for multi-turn editing

    Returns:
        Dictionary with image_url, image_preview, and generation metadata
    """
    model_id, warnings, image_size = resolve_image_model(model, image_size)

    # Log here rather than inside the resolver. Dry-run estimates and the
    # intent router resolve models too, and a WARNING saying a request "was
    # rerouted" is untrue when nothing was ever sent. A caller that never
    # inspects the returned warnings still needs to learn it is pinned to a
    # dying model, so this fires on the path that really issues the call.
    for warning in warnings:
        logger.warning("%s", warning)

    # Gemini 3.x image models require the global location when using Vertex AI.
    if model_id in _GEMINI3_IMAGE_MODELS:
        if getattr(client._api_client, "vertexai", False):
            # Reuse a memoized global-location client instead of recreating one
            # on every call.
            client = _get_vertex_global_client()

    # Prepare input images (up to 14 references for Gemini 3.x image models).
    # Decoding is budgeted: see _open_input_image.
    max_refs = 14 if model_id in _GEMINI3_IMAGE_MODELS else 1
    pil_images = _prepare_input_images(image_bytes, reference_images, max_refs)

    try:
        # Build contents for Gemini models
        contents: list[Any] = []

        # Handle conversation history for multi-turn editing
        if conversation_history:
            contents.extend(conversation_history)

        # Add current turn content
        current_turn: list[Any] = [prompt]
        for pil_img in pil_images:
            current_turn.append(pil_img)
        contents.extend(current_turn)

        # Build config with new Gemini 3 parameters
        config_kwargs: dict[str, Any] = {
            "response_modalities": ["TEXT", "IMAGE"],
        }

        # Build ImageConfig for Gemini image models. aspect_ratio and
        # person_generation are accepted by all of them (including
        # gemini-2.5-flash-image); image_size (1K/2K/4K) is Gemini
        # 3.x-only, so it is gated separately.
        image_config_kwargs: dict[str, Any] = {}
        # resolve_image_model() already cleared image_size if the resolved
        # model cannot produce it, so reaching here means the size is valid.
        if image_size and model_id in _GEMINI3_IMAGE_MODELS:
            image_config_kwargs["image_size"] = image_size
        if aspect_ratio:
            image_config_kwargs["aspect_ratio"] = aspect_ratio
        if person_generation:
            image_config_kwargs["person_generation"] = person_generation
        if image_config_kwargs:
            config_kwargs["image_config"] = types.ImageConfig(**image_config_kwargs)

        # Add media_resolution for input processing
        if media_resolution:
            config_kwargs["media_resolution"] = media_resolution

        # Add thought_signature for multi-turn editing continuity
        # It's a Part field expecting bytes, decode from base64 string
        if thought_signature:
            sig_bytes = base64.b64decode(thought_signature)
            contents.insert(0, types.Part(thought_signature=sig_bytes))

        config = types.GenerateContentConfig(**config_kwargs)
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=model_id,
            contents=contents,
            config=config,
        )

        # The decoded inputs are dead once the request has been serialized and
        # answered; holding them until the outer finally kept every reference
        # frame alive through the file write and thumbnailing too.
        for pil_img in pil_images:
            pil_img.close()
        pil_images.clear()

        output_bytes = None
        fallback_image_bytes = None
        text_parts: list[str] = []
        response_thought_signature = None
        usage = _usage_dict(response)

        candidates = response.candidates if response else None
        if candidates:
            content = candidates[0].content
            parts = content.parts if content else None
            if parts:
                for part in parts:
                    if part.text:
                        text_parts.append(part.text)
                    elif (
                        part.inline_data
                        and part.inline_data.data
                        and part.inline_data.mime_type
                        and part.inline_data.mime_type.startswith("image/")
                    ):
                        # Prefer the LAST non-thought image part. Thinking
                        # image models (e.g. gemini-3-pro-image) can emit
                        # interim sketch images (thought=True) before the
                        # final render; those must not win when a real
                        # output part exists. But keep any image as a
                        # fallback so a response containing only thought
                        # images (e.g. truncated before the final render)
                        # still returns the image the API produced rather
                        # than "no image".
                        fallback_image_bytes = part.inline_data.data
                        if not getattr(part, "thought", False):
                            output_bytes = part.inline_data.data
                    # Capture thought signature for multi-turn editing
                    if hasattr(part, "thought_signature") and part.thought_signature:
                        sig = part.thought_signature
                        # Convert bytes to string if needed for JSON serialization
                        if isinstance(sig, bytes):
                            sig = base64.b64encode(sig).decode("utf-8")
                        response_thought_signature = sig

        # Fall back to a thought/interim image if that was the only image
        # the model returned — better than discarding real image bytes.
        if not output_bytes and fallback_image_bytes:
            output_bytes = fallback_image_bytes

        if not output_bytes:
            if text_parts:
                result: dict[str, Any] = {
                    "message": "Model returned text only",
                    "generated_text": " ".join(text_parts),
                    "model": model_id,
                }
                if response_thought_signature:
                    sig_filename = f"{uuid.uuid4()}_thought.txt"
                    sig_path = images_dir / sig_filename
                    sig_path.write_text(response_thought_signature)
                    result["thought_signature_url"] = f"file://{sig_path}"
                if usage:
                    result["usage"] = usage
                if warnings:
                    result["warnings"] = warnings
                return result
            raise ValueError("Gemini returned no image")

        filename = f"{uuid.uuid4()}.png"
        filepath = images_dir / filename
        filepath.write_bytes(output_bytes)

        # Create thumbnail for inline preview (256px, balanced quality)
        thumb_image = Image.open(BytesIO(output_bytes))
        thumb_image.thumbnail((256, 256))
        if thumb_image.mode in ("RGBA", "P"):
            thumb_image = thumb_image.convert("RGB")
        thumb_buffer = BytesIO()
        thumb_image.save(thumb_buffer, format="JPEG", quality=70)
        thumb_bytes = thumb_buffer.getvalue()
        thumb_base64 = base64.b64encode(thumb_bytes).decode("utf-8")
        thumb_image.close()

        file_url = f"file://{filepath}"
        result = {
            "message": "Image generated successfully",
            "image_url": file_url,
            "image_preview": f"data:image/jpeg;base64,{thumb_base64}",
            "prompt": prompt,
            "model": model_id,
            # The size actually requested of the API, which is None when the
            # resolved model could not honour what the caller asked for.
            "image_size": image_size,
        }

        # Token counts the API actually metered, so the MCP layer can report
        # real cost instead of a pre-flight estimate.
        if usage:
            result["usage"] = usage

        # Save thought signature to file for multi-turn editing workflows
        # (can be 1MB+, too large for MCP response)
        if response_thought_signature:
            sig_filename = f"{filepath.stem}_thought.txt"
            sig_path = images_dir / sig_filename
            sig_path.write_text(response_thought_signature)
            result["thought_signature_url"] = f"file://{sig_path}"

        # Include warnings only when non-empty, matching video/omni.
        if warnings:
            result["warnings"] = warnings

        return result

    except google_auth_exceptions.RefreshError:
        raise ValueError("Authentication error - check API key or credentials")
    finally:
        # Clean up all PIL images
        for pil_img in pil_images:
            pil_img.close()
