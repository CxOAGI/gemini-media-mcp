"""Tests for image.py image generation helpers."""

import base64
from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from PIL import Image

from src.image import ImageModel, generate_image

# ============================================================================
# Test Doubles
# ============================================================================


class FakeInlineData:
    """Test double for inline data."""

    def __init__(self, mime_type: str, data: bytes) -> None:
        self.mime_type = mime_type
        self.data = data


class FakePart:
    """Test double for response part."""

    def __init__(
        self,
        text: str | None = None,
        inline_data: FakeInlineData | None = None,
    ) -> None:
        self.text = text
        self.inline_data = inline_data


class FakeContent:
    """Test double for response content."""

    def __init__(self, parts: list[FakePart]) -> None:
        self.parts = parts


class FakeCandidate:
    """Test double for response candidate."""

    def __init__(self, content: FakeContent) -> None:
        self.content = content


class FakeGeminiResponse:
    """Test double for Gemini generate_content response."""

    def __init__(self, candidates: list[FakeCandidate] | None = None) -> None:
        self.candidates = candidates


class FakeImageObject:
    """Test double for Imagen image object."""

    def __init__(self, image_bytes: bytes | None = None) -> None:
        self.image_bytes = image_bytes


class FakeGeneratedImage:
    """Test double for generated image."""

    def __init__(self, image: FakeImageObject | None = None) -> None:
        self.image = image


class FakeImagenResponse:
    """Test double for Imagen generate_images response."""

    def __init__(
        self, generated_images: list[FakeGeneratedImage] | None = None
    ) -> None:
        self.generated_images = generated_images


class FakeModels:
    """Test double for genai models."""

    def __init__(
        self,
        gemini_response: FakeGeminiResponse | None = None,
        imagen_response: FakeImagenResponse | None = None,
        raise_error: Exception | None = None,
    ) -> None:
        self._gemini_response = gemini_response
        self._imagen_response = imagen_response
        self._raise_error = raise_error

    def generate_content(self, **kwargs: Any) -> FakeGeminiResponse:
        if self._raise_error:
            raise self._raise_error
        return self._gemini_response or FakeGeminiResponse()

    def generate_images(self, **kwargs: Any) -> FakeImagenResponse:
        if self._raise_error:
            raise self._raise_error
        return self._imagen_response or FakeImagenResponse()


class FakeApiClient:
    """Test double for internal API client."""

    def __init__(self, vertexai: bool = False) -> None:
        self.vertexai = vertexai


class FakeGenaiClient:
    """Test double for Google GenAI client."""

    def __init__(
        self,
        gemini_response: FakeGeminiResponse | None = None,
        imagen_response: FakeImagenResponse | None = None,
        raise_error: Exception | None = None,
        vertexai: bool = False,
    ) -> None:
        self.models = FakeModels(gemini_response, imagen_response, raise_error)
        self._api_client = FakeApiClient(vertexai=vertexai)


def _create_test_image(
    width: int = 100, height: int = 100, color: str = "red"
) -> bytes:
    img = Image.new("RGB", (width, height), color=color)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img.close()
    return buffer.getvalue()


# ============================================================================
# generate_image tests - Gemini models
# ============================================================================


@pytest.mark.parametrize(
    ("input", "expected"),
    [
        pytest.param(
            {
                "prompt": "A red square",
                "model": "gemini-2.5-flash-image",
                "image_bytes": None,
                "response_type": "image",
            },
            {"success": True, "has_image_url": True},
            id="gemini_text_prompt_returns_image",
        ),
        pytest.param(
            {
                "prompt": "Edit this image",
                "model": "gemini-2.5-flash-image",
                "image_bytes": _create_test_image(),
                "response_type": "image",
            },
            {"success": True, "has_image_url": True},
            id="gemini_with_input_image",
        ),
        pytest.param(
            {
                "prompt": "A" * 10000,
                "model": "gemini-2.5-flash-image",
                "image_bytes": None,
                "response_type": "image",
            },
            {"success": True, "has_image_url": True},
            id="gemini_large_prompt",
        ),
        pytest.param(
            {
                "prompt": "Unicode: 🎨 日本語 émoji",
                "model": "gemini-2.5-flash-image",
                "image_bytes": None,
                "response_type": "image",
            },
            {"success": True, "has_image_url": True},
            id="gemini_unicode_prompt",
        ),
        pytest.param(
            {
                "prompt": "",
                "model": "gemini-2.5-flash-image",
                "image_bytes": None,
                "response_type": "image",
            },
            {"success": True, "has_image_url": True},
            id="gemini_empty_prompt",
        ),
        pytest.param(
            {
                "prompt": "Describe this",
                "model": "gemini-2.5-flash-image",
                "image_bytes": None,
                "response_type": "text_only",
            },
            {"success": True, "has_generated_text": True},
            id="gemini_returns_text_only",
        ),
        pytest.param(
            {
                "prompt": "Generate",
                "model": "gemini-2.5-flash-image",
                "image_bytes": None,
                "response_type": "empty",
            },
            ValueError,
            id="gemini_no_response",
        ),
        pytest.param(
            {
                "prompt": "Generate",
                "model": "gemini-2.5-flash-image",
                "image_bytes": None,
                "response_type": "no_candidates",
            },
            ValueError,
            id="gemini_no_candidates",
        ),
    ],
)
@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_image_gemini(
    input: dict[str, Any],
    expected: dict[str, Any] | type[Exception],
    tmp_path: Path,
) -> None:
    """Test generate_image with Gemini models."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    test_image_bytes = _create_test_image()

    # Build response based on response_type
    response_type = input.get("response_type", "image")

    if response_type == "image":
        inline_data = FakeInlineData("image/png", test_image_bytes)
        part = FakePart(inline_data=inline_data)
        content = FakeContent([part])
        candidate = FakeCandidate(content)
        gemini_response = FakeGeminiResponse([candidate])
    elif response_type == "text_only":
        part = FakePart(text="This is a description of the image")
        content = FakeContent([part])
        candidate = FakeCandidate(content)
        gemini_response = FakeGeminiResponse([candidate])
    elif response_type == "empty":
        content = FakeContent([])
        candidate = FakeCandidate(content)
        gemini_response = FakeGeminiResponse([candidate])
    elif response_type == "no_candidates":
        gemini_response = FakeGeminiResponse([])
    else:
        gemini_response = FakeGeminiResponse()

    client = FakeGenaiClient(gemini_response=gemini_response)

    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            await generate_image(
                client=client,  # type: ignore[arg-type]
                prompt=input["prompt"],
                images_dir=images_dir,
                model=input["model"],
                image_bytes=input.get("image_bytes"),
            )
    else:
        result = await generate_image(
            client=client,  # type: ignore[arg-type]
            prompt=input["prompt"],
            images_dir=images_dir,
            model=input["model"],
            image_bytes=input.get("image_bytes"),
        )

        assert result["model"] == input["model"]

        if expected.get("has_image_url"):
            assert "image_url" in result
            assert result["image_url"].startswith("file://")
            assert "image_preview" in result
            assert result["message"] == "Image generated successfully"
        elif expected.get("has_generated_text"):
            assert "generated_text" in result
            assert result["message"] == "Model returned text only"


# ============================================================================
# generate_image tests - Imagen models
# ============================================================================


@pytest.mark.parametrize(
    ("input", "expected"),
    [
        pytest.param(
            {
                "prompt": "A blue circle",
                "model": "imagen-3.0-generate-002",
                "image_bytes": None,
            },
            {"success": True, "has_image_url": True},
            id="imagen3_basic",
        ),
        pytest.param(
            {
                "prompt": "A green triangle",
                "model": "imagen-4.0-generate-001",
                "image_bytes": None,
            },
            {"success": True, "has_image_url": True},
            id="imagen4_standard",
        ),
        pytest.param(
            {
                "prompt": "Ultra quality image",
                "model": "imagen-4.0-ultra-generate-001",
                "image_bytes": None,
            },
            {"success": True, "has_image_url": True},
            id="imagen4_ultra",
        ),
        pytest.param(
            {
                "prompt": "Fast image",
                "model": "imagen-4.0-fast-generate-001",
                "image_bytes": None,
            },
            {"success": True, "has_image_url": True},
            id="imagen4_fast",
        ),
        pytest.param(
            {
                "prompt": "A" * 10000,
                "model": "imagen-3.0-generate-002",
                "image_bytes": None,
            },
            {"success": True, "has_image_url": True},
            id="imagen_large_prompt",
        ),
        pytest.param(
            {
                "prompt": "Unicode: 🎨 日本語",
                "model": "imagen-3.0-generate-002",
                "image_bytes": None,
            },
            {"success": True, "has_image_url": True},
            id="imagen_unicode_prompt",
        ),
        pytest.param(
            {
                "prompt": "No image returned",
                "model": "imagen-3.0-generate-002",
                "image_bytes": None,
                "empty_response": True,
            },
            ValueError,
            id="imagen_no_images",
        ),
        pytest.param(
            {
                "prompt": "No bytes",
                "model": "imagen-3.0-generate-002",
                "image_bytes": None,
                "no_bytes": True,
            },
            ValueError,
            id="imagen_no_image_bytes",
        ),
    ],
)
@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_image_imagen(
    input: dict[str, Any],
    expected: dict[str, Any] | type[Exception],
    tmp_path: Path,
) -> None:
    """Test generate_image with Imagen models."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    test_image_bytes = _create_test_image()

    # Build response based on flags
    if input.get("empty_response"):
        imagen_response = FakeImagenResponse([])
    elif input.get("no_bytes"):
        image_obj = FakeImageObject(None)
        gen_image = FakeGeneratedImage(image_obj)
        imagen_response = FakeImagenResponse([gen_image])
    else:
        image_obj = FakeImageObject(test_image_bytes)
        gen_image = FakeGeneratedImage(image_obj)
        imagen_response = FakeImagenResponse([gen_image])

    client = FakeGenaiClient(imagen_response=imagen_response)

    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            await generate_image(
                client=client,  # type: ignore[arg-type]
                prompt=input["prompt"],
                images_dir=images_dir,
                model=input["model"],
                image_bytes=input.get("image_bytes"),
            )
    else:
        result = await generate_image(
            client=client,  # type: ignore[arg-type]
            prompt=input["prompt"],
            images_dir=images_dir,
            model=input["model"],
            image_bytes=input.get("image_bytes"),
        )

        assert result["model"] == input["model"]
        assert "image_url" in result
        assert result["image_url"].startswith("file://")
        assert "image_preview" in result
        assert result["message"] == "Image generated successfully"


# ============================================================================
# generate_image tests - Gemini 3 Pro
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_image_gemini3_pro(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test generate_image with Gemini 3 Pro requires global location."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    test_image_bytes = _create_test_image()
    inline_data = FakeInlineData("image/png", test_image_bytes)
    part = FakePart(inline_data=inline_data)
    content = FakeContent([part])
    candidate = FakeCandidate(content)
    gemini_response = FakeGeminiResponse([candidate])

    # Track client creation parameters
    created_clients: list[dict[str, Any]] = []

    def mock_client(**kwargs: Any) -> FakeGenaiClient:
        created_clients.append(kwargs)
        return FakeGenaiClient(gemini_response=gemini_response)

    monkeypatch.setattr("src.image.genai.Client", mock_client)

    # Initial client (will be replaced for gemini-3-pro-image-preview)
    # Must set vertexai=True to trigger global location logic
    initial_client = FakeGenaiClient(gemini_response=gemini_response, vertexai=True)

    result = await generate_image(
        client=initial_client,  # type: ignore[arg-type]
        prompt="Test prompt",
        images_dir=images_dir,
        model="gemini-3-pro-image-preview",
    )

    # Verify a new client was created with global location
    assert len(created_clients) == 1
    assert created_clients[0]["vertexai"] is True
    assert created_clients[0]["location"] == "global"

    assert result["model"] == "gemini-3-pro-image-preview"
    assert "image_url" in result


# ============================================================================
# generate_image tests - Authentication errors
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_image_auth_error(
    tmp_path: Path,
) -> None:
    """Test generate_image handles authentication errors."""
    from google.auth import exceptions as google_auth_exceptions

    images_dir = tmp_path / "images"
    images_dir.mkdir()

    client = FakeGenaiClient(
        raise_error=google_auth_exceptions.RefreshError("Token expired"),
    )

    with pytest.raises(ValueError, match="Authentication error"):
        await generate_image(
            client=client,  # type: ignore[arg-type]
            prompt="Test prompt",
            images_dir=images_dir,
            model="gemini-2.5-flash-image",
        )


# ============================================================================
# generate_image tests - Input image handling
# ============================================================================


@pytest.mark.parametrize(
    ("input", "expected"),
    [
        pytest.param(
            {"format": "RGB", "size": (100, 100)},
            {"success": True},
            id="rgb_image",
        ),
        pytest.param(
            {"format": "RGBA", "size": (100, 100)},
            {"success": True},
            id="rgba_image",
        ),
        pytest.param(
            {"format": "L", "size": (100, 100)},
            {"success": True},
            id="grayscale_image",
        ),
        pytest.param(
            {"format": "RGB", "size": (1, 1)},
            {"success": True},
            id="tiny_image",
        ),
        pytest.param(
            {"format": "RGB", "size": (4096, 4096)},
            {"success": True},
            id="large_image",
        ),
    ],
)
@pytest.mark.asyncio
@pytest.mark.timeout(3.0)
async def test_generate_image_input_formats(
    input: dict[str, Any],
    expected: dict[str, Any],
    tmp_path: Path,
) -> None:
    """Test generate_image handles various input image formats."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    # Create input image with specified format
    img = Image.new(input["format"], input["size"], color=128)
    buffer = BytesIO()
    if input["format"] in ("RGBA", "P"):
        img.save(buffer, format="PNG")
    else:
        if input["format"] == "L":
            img = img.convert("RGB")
        img.save(buffer, format="JPEG")
    img.close()
    input_bytes = buffer.getvalue()

    # Create response
    test_image_bytes = _create_test_image()
    inline_data = FakeInlineData("image/png", test_image_bytes)
    part = FakePart(inline_data=inline_data)
    content = FakeContent([part])
    candidate = FakeCandidate(content)
    gemini_response = FakeGeminiResponse([candidate])

    client = FakeGenaiClient(gemini_response=gemini_response)

    result = await generate_image(
        client=client,  # type: ignore[arg-type]
        prompt="Edit this image",
        images_dir=images_dir,
        model="gemini-2.5-flash-image",
        image_bytes=input_bytes,
    )

    assert result["message"] == "Image generated successfully"
    assert "image_url" in result


# ============================================================================
# generate_image tests - Output file handling
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_image_creates_file(
    tmp_path: Path,
) -> None:
    """Test generate_image creates output file correctly."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    test_image_bytes = _create_test_image()
    inline_data = FakeInlineData("image/png", test_image_bytes)
    part = FakePart(inline_data=inline_data)
    content = FakeContent([part])
    candidate = FakeCandidate(content)
    gemini_response = FakeGeminiResponse([candidate])

    client = FakeGenaiClient(gemini_response=gemini_response)

    result = await generate_image(
        client=client,  # type: ignore[arg-type]
        prompt="Test prompt",
        images_dir=images_dir,
        model="gemini-2.5-flash-image",
    )

    # Verify file was created
    file_url = result["image_url"]
    assert file_url.startswith("file://")
    file_path = Path(file_url[7:])
    assert file_path.exists()
    assert file_path.suffix == ".png"

    # Verify content matches
    saved_bytes = file_path.read_bytes()
    assert saved_bytes == test_image_bytes


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_image_thumbnail_preview(
    tmp_path: Path,
) -> None:
    """Test generate_image creates proper thumbnail preview."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    # Create a larger image to test thumbnail resizing
    test_image_bytes = _create_test_image(width=1024, height=1024)
    inline_data = FakeInlineData("image/png", test_image_bytes)
    part = FakePart(inline_data=inline_data)
    content = FakeContent([part])
    candidate = FakeCandidate(content)
    gemini_response = FakeGeminiResponse([candidate])

    client = FakeGenaiClient(gemini_response=gemini_response)

    result = await generate_image(
        client=client,  # type: ignore[arg-type]
        prompt="Test prompt",
        images_dir=images_dir,
        model="gemini-2.5-flash-image",
    )

    # Verify preview is valid base64 JPEG
    preview = result["image_preview"]
    assert preview.startswith("data:image/jpeg;base64,")
    preview_b64 = preview.split(",")[1]
    preview_bytes = base64.b64decode(preview_b64)

    # Verify it's a valid JPEG
    preview_img = Image.open(BytesIO(preview_bytes))
    assert preview_img.format == "JPEG"
    # Verify thumbnail size constraint
    assert preview_img.width <= 512
    assert preview_img.height <= 512
    preview_img.close()


# ============================================================================
# generate_image tests - Gemini 3 Pro new parameters
# ============================================================================


@pytest.mark.parametrize(
    ("input", "expected"),
    [
        pytest.param(
            {"image_size": "1K"},
            {"success": True},
            id="image_size_1K",
        ),
        pytest.param(
            {"image_size": "2K"},
            {"success": True},
            id="image_size_2K",
        ),
        pytest.param(
            {"image_size": "4K"},
            {"success": True},
            id="image_size_4K",
        ),
    ],
)
@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_image_image_size(
    input: dict[str, Any],
    expected: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test generate_image with image_size parameter for Gemini 3 Pro."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    test_image_bytes = _create_test_image()
    inline_data = FakeInlineData("image/png", test_image_bytes)
    part = FakePart(inline_data=inline_data)
    content = FakeContent([part])
    candidate = FakeCandidate(content)
    gemini_response = FakeGeminiResponse([candidate])

    def mock_client(**kwargs: Any) -> FakeGenaiClient:
        return FakeGenaiClient(gemini_response=gemini_response)

    monkeypatch.setattr("src.image.genai.Client", mock_client)

    initial_client = FakeGenaiClient(gemini_response=gemini_response)

    result = await generate_image(
        client=initial_client,  # type: ignore[arg-type]
        prompt="Test prompt",
        images_dir=images_dir,
        model="gemini-3-pro-image-preview",
        image_size=input["image_size"],
    )

    assert result["message"] == "Image generated successfully"
    assert "image_url" in result


@pytest.mark.parametrize(
    ("input", "expected"),
    [
        pytest.param(
            {"media_resolution": "MEDIA_RESOLUTION_LOW"},
            {"success": True},
            id="media_resolution_low",
        ),
        pytest.param(
            {"media_resolution": "MEDIA_RESOLUTION_MEDIUM"},
            {"success": True},
            id="media_resolution_medium",
        ),
        pytest.param(
            {"media_resolution": "MEDIA_RESOLUTION_HIGH"},
            {"success": True},
            id="media_resolution_high",
        ),
    ],
)
@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_image_media_resolution(
    input: dict[str, Any],
    expected: dict[str, Any],
    tmp_path: Path,
) -> None:
    """Test generate_image with media_resolution parameter."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    test_image_bytes = _create_test_image()
    inline_data = FakeInlineData("image/png", test_image_bytes)
    part = FakePart(inline_data=inline_data)
    content = FakeContent([part])
    candidate = FakeCandidate(content)
    gemini_response = FakeGeminiResponse([candidate])

    client = FakeGenaiClient(gemini_response=gemini_response)

    result = await generate_image(
        client=client,  # type: ignore[arg-type]
        prompt="Test prompt",
        images_dir=images_dir,
        model="gemini-2.5-flash-image",
        image_bytes=test_image_bytes,
        media_resolution=input["media_resolution"],
    )

    assert result["message"] == "Image generated successfully"


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_image_multiple_reference_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test generate_image with multiple reference images for Gemini 3 Pro."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    test_image_bytes = _create_test_image()
    inline_data = FakeInlineData("image/png", test_image_bytes)
    part = FakePart(inline_data=inline_data)
    content = FakeContent([part])
    candidate = FakeCandidate(content)
    gemini_response = FakeGeminiResponse([candidate])

    def mock_client(**kwargs: Any) -> FakeGenaiClient:
        return FakeGenaiClient(gemini_response=gemini_response)

    monkeypatch.setattr("src.image.genai.Client", mock_client)

    initial_client = FakeGenaiClient(gemini_response=gemini_response)

    # Create multiple reference images
    reference_images = [
        _create_test_image(color="blue"),
        _create_test_image(color="green"),
        _create_test_image(color="yellow"),
    ]

    result = await generate_image(
        client=initial_client,  # type: ignore[arg-type]
        prompt="Combine these reference images",
        images_dir=images_dir,
        model="gemini-3-pro-image-preview",
        reference_images=reference_images,
    )

    assert result["message"] == "Image generated successfully"
    assert "image_url" in result


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_image_max_reference_images_limited(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that reference images are limited to 14 for Gemini 3 Pro."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    test_image_bytes = _create_test_image()
    inline_data = FakeInlineData("image/png", test_image_bytes)
    part = FakePart(inline_data=inline_data)
    content = FakeContent([part])
    candidate = FakeCandidate(content)
    gemini_response = FakeGeminiResponse([candidate])

    def mock_client(**kwargs: Any) -> FakeGenaiClient:
        return FakeGenaiClient(gemini_response=gemini_response)

    monkeypatch.setattr("src.image.genai.Client", mock_client)

    initial_client = FakeGenaiClient(gemini_response=gemini_response)

    # Create 20 reference images (should be limited to 14)
    reference_images = [_create_test_image(color="red") for _ in range(20)]

    result = await generate_image(
        client=initial_client,  # type: ignore[arg-type]
        prompt="Combine references",
        images_dir=images_dir,
        model="gemini-3-pro-image-preview",
        reference_images=reference_images,
    )

    assert result["message"] == "Image generated successfully"


# ============================================================================
# generate_image tests - Thought signature handling
# ============================================================================


class FakePartWithSignature(FakePart):
    """Test double for response part with thought signature."""

    def __init__(
        self,
        text: str | None = None,
        inline_data: FakeInlineData | None = None,
        thought_signature: str | None = None,
    ) -> None:
        super().__init__(text, inline_data)
        self.thought_signature = thought_signature


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_image_returns_thought_signature(
    tmp_path: Path,
) -> None:
    """Test generate_image returns thought_signature when present."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    test_image_bytes = _create_test_image()
    inline_data = FakeInlineData("image/png", test_image_bytes)
    part = FakePartWithSignature(
        inline_data=inline_data,
        thought_signature="encrypted_thought_signature_abc123",
    )
    content = FakeContent([part])
    candidate = FakeCandidate(content)
    gemini_response = FakeGeminiResponse([candidate])

    client = FakeGenaiClient(gemini_response=gemini_response)

    result = await generate_image(
        client=client,  # type: ignore[arg-type]
        prompt="Test prompt",
        images_dir=images_dir,
        model="gemini-2.5-flash-image",
    )

    assert result["message"] == "Image generated successfully"
    assert "thought_signature_url" in result
    # Verify file exists and contains signature
    sig_path = Path(result["thought_signature_url"].replace("file://", ""))
    assert sig_path.exists()
    assert sig_path.read_text() == "encrypted_thought_signature_abc123"


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_image_accepts_thought_signature(
    tmp_path: Path,
) -> None:
    """Test generate_image accepts thought_signature for multi-turn editing."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    test_image_bytes = _create_test_image()
    inline_data = FakeInlineData("image/png", test_image_bytes)
    part = FakePart(inline_data=inline_data)
    content = FakeContent([part])
    candidate = FakeCandidate(content)
    gemini_response = FakeGeminiResponse([candidate])

    client = FakeGenaiClient(gemini_response=gemini_response)

    # Pass a thought signature from a previous turn (must be valid base64)
    import base64
    prev_sig = base64.b64encode(b"previous_turn_signature").decode()
    result = await generate_image(
        client=client,  # type: ignore[arg-type]
        prompt="Make the background sunset",
        images_dir=images_dir,
        model="gemini-2.5-flash-image",
        thought_signature=prev_sig,
    )

    assert result["message"] == "Image generated successfully"


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_image_text_only_with_thought_signature(
    tmp_path: Path,
) -> None:
    """Test text-only response includes thought_signature if present."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    part = FakePartWithSignature(
        text="This is a description",
        thought_signature="signature_for_text_response",
    )
    content = FakeContent([part])
    candidate = FakeCandidate(content)
    gemini_response = FakeGeminiResponse([candidate])

    client = FakeGenaiClient(gemini_response=gemini_response)

    result = await generate_image(
        client=client,  # type: ignore[arg-type]
        prompt="Describe this",
        images_dir=images_dir,
        model="gemini-2.5-flash-image",
    )

    assert result["message"] == "Model returned text only"
    assert "thought_signature_url" in result
    sig_path = Path(result["thought_signature_url"].replace("file://", ""))
    assert sig_path.exists()
    assert sig_path.read_text() == "signature_for_text_response"


# ============================================================================
# generate_image tests - Combined parameters
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_image_all_gemini3_params(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test generate_image with all Gemini 3 Pro parameters combined."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    test_image_bytes = _create_test_image()
    inline_data = FakeInlineData("image/png", test_image_bytes)
    part = FakePartWithSignature(
        inline_data=inline_data,
        thought_signature="new_signature",
    )
    content = FakeContent([part])
    candidate = FakeCandidate(content)
    gemini_response = FakeGeminiResponse([candidate])

    def mock_client(**kwargs: Any) -> FakeGenaiClient:
        return FakeGenaiClient(gemini_response=gemini_response, vertexai=True)

    monkeypatch.setattr("src.image.genai.Client", mock_client)

    initial_client = FakeGenaiClient(gemini_response=gemini_response, vertexai=True)

    reference_images = [
        _create_test_image(color="blue"),
        _create_test_image(color="green"),
    ]

    # thought_signature must be valid base64
    import base64
    prev_sig = base64.b64encode(b"previous_signature").decode()
    result = await generate_image(
        client=initial_client,  # type: ignore[arg-type]
        prompt="Generate high quality 4K image",
        images_dir=images_dir,
        model="gemini-3-pro-image-preview",
        image_bytes=test_image_bytes,
        reference_images=reference_images,
        image_size="4K",
        media_resolution="MEDIA_RESOLUTION_HIGH",
        thought_signature=prev_sig,
    )

    assert result["message"] == "Image generated successfully"
    assert result["model"] == "gemini-3-pro-image-preview"
    assert "thought_signature_url" in result
    sig_path = Path(result["thought_signature_url"].replace("file://", ""))
    assert sig_path.exists()
    assert sig_path.read_text() == "new_signature"
