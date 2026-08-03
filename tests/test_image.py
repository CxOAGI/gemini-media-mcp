"""Tests for image.py image generation helpers."""

import base64
import logging
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import pytest
from PIL import Image

import src.image
from src.image import LegacyImagenModel, generate_image


@pytest.fixture(autouse=True)
def _reset_vertex_global_client() -> Any:
    """Reset the module-level memoized Vertex global client around each test.

    The client is cached at module scope for efficiency; resetting keeps each
    test's client-creation assertions independent.
    """
    src.image._vertex_global_client = None
    yield
    src.image._vertex_global_client = None


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
        thought: bool = False,
    ) -> None:
        self.text = text
        self.inline_data = inline_data
        self.thought = thought


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


class FakeModels:
    """Test double for genai models."""

    def __init__(
        self,
        gemini_response: FakeGeminiResponse | None = None,
        raise_error: Exception | None = None,
    ) -> None:
        self._gemini_response = gemini_response
        self._raise_error = raise_error
        self.last_generate_content_kwargs: dict[str, Any] | None = None
        self.last_generate_images_kwargs: dict[str, Any] | None = None

    def generate_content(self, **kwargs: Any) -> FakeGeminiResponse:
        self.last_generate_content_kwargs = kwargs
        if self._raise_error:
            raise self._raise_error
        return self._gemini_response or FakeGeminiResponse()

    def generate_images(self, **kwargs: Any) -> None:
        """Tripwire: generate_images is the Imagen-only endpoint Google
        discontinues on 2026-08-17. Nothing may call it any more."""
        self.last_generate_images_kwargs = kwargs
        raise AssertionError(
            "generate_images (Imagen) was called; it is discontinued on 2026-08-17"
        )


class FakeApiClient:
    """Test double for internal API client."""

    def __init__(self, vertexai: bool = False) -> None:
        self.vertexai = vertexai


class FakeGenaiClient:
    """Test double for Google GenAI client."""

    def __init__(
        self,
        gemini_response: FakeGeminiResponse | None = None,
        raise_error: Exception | None = None,
        vertexai: bool = False,
    ) -> None:
        self.models = FakeModels(gemini_response, raise_error)
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
# generate_image tests - legacy Imagen IDs reroute to Gemini GA
# ============================================================================


@pytest.mark.parametrize(
    ("legacy_model", "expected_target"),
    [
        ("imagen-3.0-capability-001", "gemini-3.1-flash-image"),
        ("imagen-3.0-capability-002", "gemini-3.1-flash-image"),
        ("imagen-3.0-fast-generate-001", "gemini-3.1-flash-lite-image"),
        ("imagen-3.0-generate-001", "gemini-3.1-flash-image"),
        ("imagen-3.0-generate-002", "gemini-3.1-flash-image"),
        ("imagen-4.0-fast-generate-001", "gemini-3.1-flash-lite-image"),
        ("imagen-4.0-generate-001", "gemini-3.1-flash-image"),
        ("imagen-4.0-ultra-generate-001", "gemini-3.1-flash-image"),
    ],
)
@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_legacy_imagen_reroutes_to_gemini_ga(
    legacy_model: LegacyImagenModel,
    expected_target: str,
    tmp_path: Path,
) -> None:
    """A discontinued Imagen ID is served by its Gemini GA replacement via
    generate_content, never by the retired generate_images endpoint."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    part = FakePart(inline_data=FakeInlineData("image/png", _create_test_image()))
    gemini_response = FakeGeminiResponse([FakeCandidate(FakeContent([part]))])
    client = FakeGenaiClient(gemini_response=gemini_response)

    result = await generate_image(
        client=client,  # type: ignore[arg-type]
        prompt="A blue circle",
        images_dir=images_dir,
        model=legacy_model,
    )

    assert client.models.last_generate_content_kwargs is not None
    assert client.models.last_generate_content_kwargs["model"] == expected_target
    # The discontinued endpoint must not be called at all.
    assert client.models.last_generate_images_kwargs is None
    # The reported model is the one actually served, not the dead alias.
    assert result["model"] == expected_target
    assert result["image_url"].startswith("file://")
    assert result["message"] == "Image generated successfully"

    joined = " ".join(result["warnings"])
    assert legacy_model in joined
    assert expected_target in joined
    assert "2026-08-17" in joined


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_legacy_imagen_reroute_is_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The reroute is logged at WARNING, so an operator whose caller ignores the
    returned warnings still learns it is pinned to a discontinued model."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    part = FakePart(inline_data=FakeInlineData("image/png", _create_test_image()))
    gemini_response = FakeGeminiResponse([FakeCandidate(FakeContent([part]))])
    client = FakeGenaiClient(gemini_response=gemini_response)

    with caplog.at_level(logging.WARNING, logger="src.image"):
        await generate_image(
            client=client,  # type: ignore[arg-type]
            prompt="a photo",
            images_dir=images_dir,
            model="imagen-4.0-generate-001",
        )

    records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(records) == 1
    message = records[0].getMessage()
    assert "imagen-4.0-generate-001" in message
    assert "gemini-3.1-flash-image" in message
    assert "2026-08-17" in message


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_gemini_image_reroute_is_not_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A normal Gemini request logs nothing — the warning must not be noise."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    part = FakePart(inline_data=FakeInlineData("image/png", _create_test_image()))
    gemini_response = FakeGeminiResponse([FakeCandidate(FakeContent([part]))])
    client = FakeGenaiClient(gemini_response=gemini_response)

    with caplog.at_level(logging.WARNING, logger="src.image"):
        await generate_image(
            client=client,  # type: ignore[arg-type]
            prompt="a photo",
            images_dir=images_dir,
            model="gemini-3.1-flash-image",
        )

    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_legacy_imagen_accepts_input_images(tmp_path: Path) -> None:
    """Input and reference images are no longer dropped for a legacy Imagen
    request — the Gemini replacement accepts them."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    part = FakePart(inline_data=FakeInlineData("image/png", _create_test_image()))
    gemini_response = FakeGeminiResponse([FakeCandidate(FakeContent([part]))])
    client = FakeGenaiClient(gemini_response=gemini_response)

    result = await generate_image(
        client=client,  # type: ignore[arg-type]
        prompt="Edit this",
        images_dir=images_dir,
        model="imagen-4.0-generate-001",
        image_bytes=_create_test_image(),
        reference_images=[_create_test_image()],
    )

    assert result["message"] == "Image generated successfully"
    assert client.models.last_generate_content_kwargs is not None
    contents = client.models.last_generate_content_kwargs["contents"]
    # prompt + input image + reference image all reached the model.
    assert len(contents) == 3
    joined = " ".join(result["warnings"])
    assert "ignored" not in joined


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_unknown_imagen_id_falls_back_to_flash_image(tmp_path: Path) -> None:
    """An Imagen ID missing from the published table still reroutes rather than
    hitting a discontinued endpoint."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    part = FakePart(inline_data=FakeInlineData("image/png", _create_test_image()))
    gemini_response = FakeGeminiResponse([FakeCandidate(FakeContent([part]))])
    client = FakeGenaiClient(gemini_response=gemini_response)

    result = await generate_image(
        client=client,  # type: ignore[arg-type]
        prompt="A blue circle",
        images_dir=images_dir,
        # Deliberately off-table: not a LegacyImagenModel member.
        model=cast(Any, "imagen-9.9-generate-999"),
    )

    assert result["model"] == "gemini-3.1-flash-image"
    assert client.models.last_generate_images_kwargs is None


# ============================================================================
# generate_image tests - Gemini 3.1 Flash Image Preview
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_image_gemini31_flash(
    tmp_path: Path,
) -> None:
    """Test generate_image with Gemini 3.1 Flash model."""
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
        prompt="A red square",
        images_dir=images_dir,
        model="gemini-3.1-flash-image-preview",
    )

    assert result["model"] == "gemini-3.1-flash-image-preview"
    assert result["message"] == "Image generated successfully"
    assert "image_url" in result
    assert result["image_url"].startswith("file://")


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_image_gemini31_flash_vertex_global_location(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test generate_image with Gemini 3.1 Flash requires global location on Vertex AI."""
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

    # Initial client (will be replaced for gemini-3.1-flash-image-preview)
    # Must set vertexai=True to trigger global location logic
    initial_client = FakeGenaiClient(gemini_response=gemini_response, vertexai=True)

    result = await generate_image(
        client=initial_client,  # type: ignore[arg-type]
        prompt="Test prompt",
        images_dir=images_dir,
        model="gemini-3.1-flash-image-preview",
    )

    # Verify a new client was created with global location
    assert len(created_clients) == 1
    assert created_clients[0]["vertexai"] is True
    assert created_clients[0]["location"] == "global"

    assert result["model"] == "gemini-3.1-flash-image-preview"
    assert "image_url" in result


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_image_gemini31_flash_image_size(
    tmp_path: Path,
) -> None:
    """Test generate_image with Gemini 3.1 Flash supports image_size parameter."""
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
        model="gemini-3.1-flash-image-preview",
        image_size="4K",
    )

    assert result["message"] == "Image generated successfully"
    assert "image_url" in result


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_image_gemini31_flash_reference_images(
    tmp_path: Path,
) -> None:
    """Test generate_image with Gemini 3.1 Flash supports multiple reference images."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    test_image_bytes = _create_test_image()
    inline_data = FakeInlineData("image/png", test_image_bytes)
    part = FakePart(inline_data=inline_data)
    content = FakeContent([part])
    candidate = FakeCandidate(content)
    gemini_response = FakeGeminiResponse([candidate])

    client = FakeGenaiClient(gemini_response=gemini_response)

    reference_images = [
        _create_test_image(color="blue"),
        _create_test_image(color="green"),
        _create_test_image(color="yellow"),
    ]

    result = await generate_image(
        client=client,  # type: ignore[arg-type]
        prompt="Combine these reference images",
        images_dir=images_dir,
        model="gemini-3.1-flash-image-preview",
        reference_images=reference_images,
    )

    assert result["message"] == "Image generated successfully"
    assert "image_url" in result


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


# ============================================================================
# generate_image tests - ImageModel catalog
# ============================================================================


def test_image_model_excludes_all_imagen_ids() -> None:
    """Every Imagen endpoint is discontinued on 2026-08-17, so ImageModel — the
    catalog offered to callers — must contain no Imagen ID at all."""
    from typing import get_args

    from src.image import ImageModel

    assert not [m for m in get_args(ImageModel) if str(m).startswith("imagen")]


def test_image_model_includes_new_ga_ids() -> None:
    """New GA Gemini 3.x IDs must be present in ImageModel."""
    from typing import get_args

    from src.image import ImageModel

    models = set(get_args(ImageModel))
    expected = {
        "gemini-2.5-flash-image",
        "gemini-3-pro-image",
        "gemini-3.1-flash-image",
        "gemini-3.1-flash-lite-image",
        "gemini-3-pro-image-preview",
        "gemini-3.1-flash-image-preview",
    }
    assert expected <= models


def test_legacy_imagen_catalog_covers_googles_discontinued_list() -> None:
    """Every endpoint on Google's discontinuation table is accepted as a legacy
    alias and has a published migration target."""
    from typing import get_args

    from src.image import _IMAGEN_MIGRATION, LegacyImagenModel

    discontinued = {
        "imagen-3.0-capability-001",
        "imagen-3.0-capability-002",
        "imagen-3.0-fast-generate-001",
        "imagen-3.0-generate-001",
        "imagen-3.0-generate-002",
        "imagen-4.0-fast-generate-001",
        "imagen-4.0-generate-001",
        "imagen-4.0-ultra-generate-001",
    }
    assert set(get_args(LegacyImagenModel)) == discontinued
    assert set(_IMAGEN_MIGRATION) == discontinued
    # Google's table only permits these two targets.
    assert set(_IMAGEN_MIGRATION.values()) <= {
        "gemini-3.1-flash-image",
        "gemini-3.1-flash-lite-image",
    }


# ============================================================================
# generate_image tests - aspect_ratio and person_generation
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_legacy_imagen_aspect_ratio_and_person_generation(
    tmp_path: Path,
) -> None:
    """aspect_ratio and person_generation survive the reroute and land on the
    Gemini ImageConfig."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    part = FakePart(inline_data=FakeInlineData("image/png", _create_test_image()))
    gemini_response = FakeGeminiResponse([FakeCandidate(FakeContent([part]))])
    client = FakeGenaiClient(gemini_response=gemini_response)

    result = await generate_image(
        client=client,  # type: ignore[arg-type]
        prompt="A landscape",
        images_dir=images_dir,
        model="imagen-4.0-generate-001",
        aspect_ratio="16:9",
        person_generation="allow_adult",
    )

    assert result["message"] == "Image generated successfully"
    config = client.models.last_generate_content_kwargs["config"]
    assert config.image_config is not None
    assert config.image_config.aspect_ratio == "16:9"
    # The SDK coerces the pass-through string into a PersonGeneration enum.
    assert "allow_adult" in str(config.image_config.person_generation).lower()


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_image_gemini_aspect_ratio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """aspect_ratio is merged into the Gemini ImageConfig alongside image_size."""
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
        prompt="A portrait",
        images_dir=images_dir,
        model="gemini-3-pro-image",
        image_size="2K",
        aspect_ratio="9:16",
    )

    assert result["message"] == "Image generated successfully"
    config = client.models.last_generate_content_kwargs["config"]
    assert config.image_config is not None
    assert config.image_config.aspect_ratio == "9:16"
    assert config.image_config.image_size == "2K"


# ============================================================================
# generate_image tests - multi-image response
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_image_multi_image_keeps_last_non_thought(
    tmp_path: Path,
) -> None:
    """Among real image parts keep the LAST (final render); ignore thought parts.

    Thinking image models can emit interim sketch images (thought=True) before
    the final image, and may emit more than one real image part; the final
    non-thought image must win.
    """
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    sketch_bytes = _create_test_image(color="green")
    draft_bytes = _create_test_image(color="red")
    final_bytes = _create_test_image(color="blue")
    assert len({sketch_bytes, draft_bytes, final_bytes}) == 3

    # Order: interim thought sketch, then a draft, then the final render.
    thought_part = FakePart(
        inline_data=FakeInlineData("image/png", sketch_bytes), thought=True
    )
    draft_part = FakePart(inline_data=FakeInlineData("image/png", draft_bytes))
    final_part = FakePart(inline_data=FakeInlineData("image/png", final_bytes))
    content = FakeContent([thought_part, draft_part, final_part])
    candidate = FakeCandidate(content)
    gemini_response = FakeGeminiResponse([candidate])

    client = FakeGenaiClient(gemini_response=gemini_response)

    result = await generate_image(
        client=client,  # type: ignore[arg-type]
        prompt="Generate",
        images_dir=images_dir,
        model="gemini-3-pro-image",
    )

    file_path = Path(result["image_url"][7:])
    # Final non-thought image wins; the thought sketch is never selected.
    assert file_path.read_bytes() == final_bytes


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_image_only_thought_images_falls_back(
    tmp_path: Path,
) -> None:
    """If every image part is a thought part, fall back to it rather than fail.

    A truncated thinking-model response may contain only interim (thought=True)
    images; returning that image beats discarding the bytes the API produced.
    """
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    sketch_bytes = _create_test_image(color="green")
    thought_part = FakePart(
        inline_data=FakeInlineData("image/png", sketch_bytes), thought=True
    )
    content = FakeContent([thought_part])
    candidate = FakeCandidate(content)
    gemini_response = FakeGeminiResponse([candidate])

    client = FakeGenaiClient(gemini_response=gemini_response)

    result = await generate_image(
        client=client,  # type: ignore[arg-type]
        prompt="Generate",
        images_dir=images_dir,
        model="gemini-3-pro-image",
    )

    # An image is returned (the fallback), not a "no image" / text-only result.
    assert "image_url" in result
    file_path = Path(result["image_url"][7:])
    assert file_path.read_bytes() == sketch_bytes


# ============================================================================
# generate_image tests - memoized Vertex global client reuse
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_image_vertex_global_client_reused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The memoized Vertex global client is created once and reused across calls."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    test_image_bytes = _create_test_image()
    inline_data = FakeInlineData("image/png", test_image_bytes)
    part = FakePart(inline_data=inline_data)
    content = FakeContent([part])
    candidate = FakeCandidate(content)
    gemini_response = FakeGeminiResponse([candidate])

    created_clients: list[dict[str, Any]] = []

    def mock_client(**kwargs: Any) -> FakeGenaiClient:
        created_clients.append(kwargs)
        return FakeGenaiClient(gemini_response=gemini_response)

    monkeypatch.setattr("src.image.genai.Client", mock_client)

    # Two separate calls, each supplying a Vertex-mode client that triggers the
    # global-location swap.
    for _ in range(2):
        initial_client = FakeGenaiClient(gemini_response=gemini_response, vertexai=True)
        result = await generate_image(
            client=initial_client,  # type: ignore[arg-type]
            prompt="Test prompt",
            images_dir=images_dir,
            model="gemini-3-pro-image",
        )
        assert result["message"] == "Image generated successfully"

    # The global client should have been created only once and reused.
    assert len(created_clients) == 1
    assert created_clients[0]["vertexai"] is True
    assert created_clients[0]["location"] == "global"


# ============================================================================
# Warnings channel (legacy Imagen reroute)
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_legacy_imagen_text_only_response_keeps_warning(
    tmp_path: Path,
) -> None:
    """The reroute warning survives the text-only return path too."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    part = FakePart(text="I cannot draw that")
    gemini_response = FakeGeminiResponse([FakeCandidate(FakeContent([part]))])
    client = FakeGenaiClient(gemini_response=gemini_response)

    result = await generate_image(
        client=client,  # type: ignore[arg-type]
        prompt="a photo",
        images_dir=images_dir,
        model="imagen-4.0-generate-001",
    )
    assert result["message"] == "Model returned text only"
    assert any("2026-08-17" in w for w in result["warnings"])


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_gemini_image_result_has_no_warnings(tmp_path: Path) -> None:
    """Gemini image results omit the warnings key entirely on a clean run."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    part = FakePart(inline_data=FakeInlineData("image/png", _create_test_image()))
    gemini_response = FakeGeminiResponse([FakeCandidate(FakeContent([part]))])
    client = FakeGenaiClient(gemini_response=gemini_response)

    result = await generate_image(
        client=client,  # type: ignore[arg-type]
        prompt="a cat",
        images_dir=images_dir,
        model="gemini-2.5-flash-image",
    )
    assert "warnings" not in result
