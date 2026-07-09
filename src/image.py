"""Image generation helpers."""

import asyncio
import base64
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

from google import genai
from google.auth import exceptions as google_auth_exceptions
from google.genai import types
from PIL import Image

ImageModel = Literal[
    # Gemini image models (Nano Banana family)
    "gemini-2.5-flash-image",
    # Gemini 3.x image models, now GA under suffix-less IDs
    "gemini-3-pro-image",
    "gemini-3.1-flash-image",
    "gemini-3.1-flash-lite-image",
    # -preview aliases retained for accounts still pinned to them
    "gemini-3-pro-image-preview",
    "gemini-3.1-flash-image-preview",
    # Imagen 4.x models (deprecated by Google, shutdown 2026-08-17)
    "imagen-4.0-generate-001",
    "imagen-4.0-ultra-generate-001",
    "imagen-4.0-fast-generate-001",
]

# NOTE: imagen-3.0-generate-002 was shut down by Google on 2025-11-10 and has
# been removed from ImageModel because every call now fails.

# Gemini 3.x image models (both GA and -preview) that share enhanced
# capabilities (image_config, up to 14 reference images, global-location Vertex).
_GEMINI3_IMAGE_MODELS = {
    "gemini-3-pro-image",
    "gemini-3.1-flash-image",
    "gemini-3.1-flash-lite-image",
    "gemini-3-pro-image-preview",
    "gemini-3.1-flash-image-preview",
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


async def generate_image(
    client: genai.Client,
    prompt: str,
    images_dir: Path,
    model: ImageModel = "gemini-2.5-flash-image",
    image_bytes: bytes | None = None,
    reference_images: list[bytes] | None = None,
    image_size: ImageSize | None = None,
    media_resolution: MediaResolution | None = None,
    aspect_ratio: str | None = None,
    person_generation: str | None = None,
    thought_signature: str | None = None,
    conversation_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Generate an image using Gemini or Imagen models.

    Supported models (``ImageModel``):
      - gemini-2.5-flash-image
      - gemini-3-pro-image, gemini-3.1-flash-image, gemini-3.1-flash-lite-image
        (GA IDs; -preview aliases also accepted)
      - imagen-4.0-generate-001, imagen-4.0-ultra-generate-001,
        imagen-4.0-fast-generate-001 (deprecated, see note below)

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
            Applied to Imagen via GenerateImagesConfig and to Gemini 3.x image
            models via ImageConfig. Passed through as-is; the API validates it.
        person_generation: Policy for generating people. Valid values:
            "dont_allow", "allow_adult", "allow_all". Applied to Imagen via
            GenerateImagesConfig and to Gemini 3.x image models via ImageConfig.
            Passed through as-is; the API validates it.
        thought_signature: Thought signature from previous turn for multi-turn editing
        conversation_history: Previous conversation history for multi-turn editing

    Returns:
        Dictionary with image_url, image_preview, and generation metadata
    """
    model_id = str(model)

    # Gemini 3.x image models require the global location when using Vertex AI.
    if model in _GEMINI3_IMAGE_MODELS:
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
        max_refs = 14 if model in _GEMINI3_IMAGE_MODELS else 1
        for ref_bytes in reference_images[:max_refs]:
            ref_image = Image.open(BytesIO(ref_bytes))
            ref_image.load()
            pil_images.append(ref_image)

    try:
        if model_id.startswith("imagen"):
            # NOTE: Imagen 4.x is deprecated by Google with shutdown scheduled
            # for 2026-08-17. The replacement is the Nano Banana / gemini-3.x
            # image family (gemini-3-pro-image, gemini-3.1-flash-image, etc.).
            imagen_config_kwargs: dict[str, Any] = {"number_of_images": 1}
            if aspect_ratio:
                imagen_config_kwargs["aspect_ratio"] = aspect_ratio
            if person_generation:
                imagen_config_kwargs["person_generation"] = person_generation
            config = types.GenerateImagesConfig(**imagen_config_kwargs)
            response = await asyncio.to_thread(
                client.models.generate_images,
                model=model_id,
                prompt=prompt,
                config=config,
            )
            generated_images = response.generated_images
            if not generated_images:
                raise ValueError("Imagen returned no image")
            image_obj = generated_images[0].image
            if image_obj is None or image_obj.image_bytes is None:
                raise ValueError("Imagen returned no image bytes")
            output_bytes = image_obj.image_bytes
            response_thought_signature = None
        else:
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
            if image_size and model in _GEMINI3_IMAGE_MODELS:
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
                        if (
                            hasattr(part, "thought_signature")
                            and part.thought_signature
                        ):
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

        # Save thought signature to file for multi-turn editing workflows
        # (can be 1MB+, too large for MCP response)
        if response_thought_signature:
            sig_filename = f"{filepath.stem}_thought.txt"
            sig_path = images_dir / sig_filename
            sig_path.write_text(response_thought_signature)
            result["thought_signature_url"] = f"file://{sig_path}"

        return result

    except google_auth_exceptions.RefreshError:
        raise ValueError("Authentication error - check API key or credentials")
    finally:
        # Clean up all PIL images
        for pil_img in pil_images:
            pil_img.close()
