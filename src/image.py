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


async def generate_image(
    client: genai.Client,
    prompt: str,
    images_dir: Path,
    model: ImageModel = "gemini-2.5-flash-image",
    image_bytes: bytes | None = None,
) -> dict[str, Any]:
    """Generate an image using Gemini or Imagen models."""
    model_id = str(model)

    # Gemini 3 Pro Image requires global location
    if model == "gemini-3-pro-image-preview":
        client = genai.Client(vertexai=True, location="global")

    pil_image = None
    if image_bytes:
        pil_image = Image.open(BytesIO(image_bytes))
        pil_image.load()

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
        else:
            contents: list[Any] = [prompt]
            if pil_image:
                contents.append(pil_image)

            config = types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"])
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=model_id,
                contents=contents,
                config=config,
            )

            output_bytes = None
            text_parts: list[str] = []
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
                            break

            if not output_bytes:
                if text_parts:
                    return {
                        "message": "Model returned text only",
                        "generated_text": " ".join(text_parts),
                        "model": model_id,
                    }
                raise ValueError("Gemini returned no image")

        filename = f"{uuid.uuid4()}.png"
        filepath = images_dir / filename
        filepath.write_bytes(output_bytes)

        # Create thumbnail for inline preview (max 512px, JPEG for size)
        thumb_image = Image.open(BytesIO(output_bytes))
        thumb_image.thumbnail((512, 512))
        if thumb_image.mode in ("RGBA", "P"):
            thumb_image = thumb_image.convert("RGB")
        thumb_buffer = BytesIO()
        thumb_image.save(thumb_buffer, format="JPEG", quality=80)
        thumb_bytes = thumb_buffer.getvalue()
        thumb_base64 = base64.b64encode(thumb_bytes).decode("utf-8")
        thumb_image.close()

        file_url = f"file://{filepath}"
        return {
            "message": "Image generated successfully",
            "image_url": file_url,
            "image_preview": f"data:image/jpeg;base64,{thumb_base64}",
            "prompt": prompt,
            "model": model_id,
        }

    except google_auth_exceptions.RefreshError:
        raise ValueError("Authentication error - check API key or credentials")
    finally:
        if pil_image:
            pil_image.close()
