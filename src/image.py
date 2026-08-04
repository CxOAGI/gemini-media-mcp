"""Image generation helpers."""

import asyncio
import base64
import logging
import uuid
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
# see _RETIRED_MODELS. They are deliberately split out of ImageModel, the live
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

# Retired ID -> (replacement, shutdown date). Sourced from Google's published
# deprecation table. The Imagen rows all name gemini-3.1-flash-image as the
# replacement; the preview aliases map to their own GA promotion.
_RETIRED_MODELS: dict[str, tuple[str, str]] = {
    "imagen-3.0-capability-001": ("gemini-3.1-flash-image", "2026-08-17"),
    "imagen-3.0-capability-002": ("gemini-3.1-flash-image", "2026-08-17"),
    "imagen-3.0-fast-generate-001": ("gemini-3.1-flash-image", "2026-08-17"),
    "imagen-3.0-generate-001": ("gemini-3.1-flash-image", "2026-08-17"),
    "imagen-3.0-generate-002": ("gemini-3.1-flash-image", "2026-08-17"),
    "imagen-4.0-fast-generate-001": ("gemini-3.1-flash-image", "2026-08-17"),
    "imagen-4.0-generate-001": ("gemini-3.1-flash-image", "2026-08-17"),
    "imagen-4.0-ultra-generate-001": ("gemini-3.1-flash-image", "2026-08-17"),
    "gemini-3-pro-image-preview": ("gemini-3-pro-image", "2026-06-25"),
    "gemini-3.1-flash-image-preview": ("gemini-3.1-flash-image", "2026-06-25"),
}

# Fallback for a retired-looking ID that is not in the table (e.g. a regional
# or newly-surfaced Imagen variant) — still better than a guaranteed 404.
_RETIRED_DEFAULT_TARGET = "gemini-3.1-flash-image"

# Still served, but with a published shutdown date. Rerouted like the retired
# IDs — a caller left on one of these has a hard deadline and no upside, since
# the replacement is strictly more capable. Only the wording differs, so the
# warning does not claim a model is already gone when it is not.
_SUNSET_MODELS: dict[str, tuple[str, str]] = {
    "gemini-2.5-flash-image": ("gemini-3.1-flash-image", "2026-10-02"),
}

# Output sizes a model can actually produce. Only models with a restriction are
# listed; anything absent accepts the full ImageSize range. gemini-3.1-flash-
# lite-image is 1K-only — 2K and 4K are documented as unsupported.
_IMAGE_SIZE_SUPPORT: dict[str, frozenset[str]] = {
    "gemini-3.1-flash-lite-image": frozenset({"1K"}),
}


def _usage_dict(response: Any) -> dict[str, int] | None:
    """Extract token counts from a response's usage_metadata as a plain dict.

    Returned to the caller so cost can be computed from what the API actually
    metered rather than from a pre-flight estimate. Kept as a plain dict of
    ints because the result travels through ``json.dumps`` in the MCP layer,
    which cannot serialize the SDK's usage object.
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
    out: dict[str, int] = {}
    for field in fields:
        value = getattr(usage, field, None)
        if isinstance(value, int):
            out[field] = value
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
    generation path and the ``dry_run`` estimate so a quoted price always
    describes the call that would really be issued.

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
    if (
        model_id in _RETIRED_MODELS
        or model_id in _SUNSET_MODELS
        or model_id.startswith("imagen")
    ):
        if model_id in _SUNSET_MODELS:
            target, shutdown = _SUNSET_MODELS[model_id]
            state = f"is scheduled for shutdown on {shutdown}"
        else:
            target, shutdown = _RETIRED_MODELS.get(
                model_id, (_RETIRED_DEFAULT_TARGET, "2026-08-17")
            )
            state = f"was retired on {shutdown} and no longer exists"
        warnings.append(
            f"Model {model_id} {state}; {target} served this request instead. "
            f"Update your configuration to request {target} directly."
        )
        # Also log it: a caller that never inspects the returned warnings still
        # needs to find out it is pinned to a model that is going away.
        logger.warning(
            "Rerouted model %s to %s (shutdown %s); update the caller's configuration",
            model_id,
            target,
            shutdown,
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
    model: ImageModel | RetiredImageModel = "gemini-3.1-flash-image",
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
      - the Imagen family — endpoints gone 2026-08-17 (``_RETIRED_MODELS``)
      - the -preview image aliases — gone 2026-06-25 (``_RETIRED_MODELS``)
      - gemini-2.5-flash-image — still served, shutdown 2026-10-02
        (``_SUNSET_MODELS``)

    Every substitution is reported in the returned ``warnings`` list and
    logged at WARNING.

    Args:
        client: Google GenAI client
        prompt: Text description of the image to generate
        images_dir: Directory to save generated images
        model: Model to use for generation
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

    # Gemini 3.x image models require the global location when using Vertex AI.
    if model_id in _GEMINI3_IMAGE_MODELS:
        if getattr(client._api_client, "vertexai", False):
            # Reuse a memoized global-location client instead of recreating one
            # on every call.
            client = _get_vertex_global_client()

    # Prepare input images
    pil_images: list[Image.Image] = []
    if image_bytes:
        pil_image = Image.open(BytesIO(image_bytes))
        pil_image.load()
        pil_images.append(pil_image)

    # Process reference images (up to 14 for Gemini 3.x image models)
    if reference_images:
        max_refs = 14 if model_id in _GEMINI3_IMAGE_MODELS else 1
        for ref_bytes in reference_images[:max_refs]:
            ref_image = Image.open(BytesIO(ref_bytes))
            ref_image.load()
            pil_images.append(ref_image)

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
