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
    "gemini-2.5-flash-image",
    "gemini-3-pro-image-preview",
    "imagen-3.0-generate-002",
    "imagen-4.0-generate-001",
    "imagen-4.0-ultra-generate-001",
    "imagen-4.0-fast-generate-001",
]

# Output image size options for Gemini 3 Pro Image
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
    thought_signature: str | None = None,
    conversation_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Generate an image using Gemini or Imagen models.

    Args:
        client: Google GenAI client
        prompt: Text description of the image to generate
        images_dir: Directory to save generated images
        model: Model to use for generation
        image_bytes: Input image bytes for editing
        reference_images: List of reference image bytes (up to 14 for Gemini 3 Pro)
        image_size: Output image size (1K, 2K, 4K) - must use uppercase K
        media_resolution: Input image resolution processing (low/medium/high)
        thought_signature: Thought signature from previous turn for multi-turn editing
        conversation_history: Previous conversation history for multi-turn editing

    Returns:
        Dictionary with image_url, image_preview, and generation metadata
    """
    model_id = str(model)

    # Gemini 3 Pro Image requires global location when using Vertex AI
    if model == "gemini-3-pro-image-preview":
        if getattr(client._api_client, 'vertexai', False):
            # Re-create client with global location for Vertex AI
            client = genai.Client(vertexai=True, location="global")

    # Prepare input images
    pil_images: list[Image.Image] = []
    if image_bytes:
        pil_image = Image.open(BytesIO(image_bytes))
        pil_image.load()
        pil_images.append(pil_image)

    # Process reference images (up to 14 for Gemini 3 Pro)
    if reference_images:
        max_refs = 14 if model == "gemini-3-pro-image-preview" else 1
        for ref_bytes in reference_images[:max_refs]:
            ref_image = Image.open(BytesIO(ref_bytes))
            ref_image.load()
            pil_images.append(ref_image)

    try:
        if model_id.startswith("imagen"):
            config = types.GenerateImagesConfig(number_of_images=1)
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

            # Add image_size for Gemini 3 Pro (1K, 2K, 4K)
            if image_size and model == "gemini-3-pro-image-preview":
                config_kwargs["image_config"] = types.ImageConfig(
                    image_size=image_size
                )

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
                            and part.inline_data.mime_type
                            and part.inline_data.mime_type.startswith("image/")
                        ):
                            output_bytes = part.inline_data.data
                        # Capture thought signature for multi-turn editing
                        if hasattr(part, "thought_signature") and part.thought_signature:
                            sig = part.thought_signature
                            # Convert bytes to string if needed for JSON serialization
                            if isinstance(sig, bytes):
                                sig = base64.b64encode(sig).decode("utf-8")
                            response_thought_signature = sig

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
