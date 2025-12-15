"""Tests for __main__.py MCP server."""

import asyncio
import base64
import json
import tempfile
from collections.abc import AsyncIterator
from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from src.__main__ import (
    AppContext,
    app_lifespan,
    check_credentials,
    cleanup_credentials,
    create_client,
    fetch,
    is_running_in_container,
    setup_vertex_credentials,
)

# ============================================================================
# Test Doubles
# ============================================================================


class FakeGCSBlob:
    """Test double for GCS blob."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def download_as_bytes(self) -> bytes:
        return self._data


class FakeGCSBucket:
    """Test double for GCS bucket."""

    def __init__(self, blobs: dict[str, bytes]) -> None:
        self._blobs = blobs

    def blob(self, name: str) -> FakeGCSBlob:
        if name not in self._blobs:
            raise ValueError(f"Blob not found: {name}")
        return FakeGCSBlob(self._blobs[name])


class FakeGCSClient:
    """Test double for GCS client."""

    def __init__(self, buckets: dict[str, dict[str, bytes]]) -> None:
        self._buckets = buckets

    def bucket(self, name: str) -> FakeGCSBucket:
        if name not in self._buckets:
            raise ValueError(f"Bucket not found: {name}")
        return FakeGCSBucket(self._buckets[name])


class FakeGenaiClient:
    """Test double for Google GenAI client."""

    def __init__(self, vertexai: bool = False, api_key: str | None = None) -> None:
        self.vertexai = vertexai
        self.api_key = api_key


class FakeFastMCP:
    """Test double for FastMCP server."""

    pass


class FakeResponse:
    """Test double for aiohttp response."""

    def __init__(self, status: int, data: bytes) -> None:
        self.status = status
        self._data = data

    async def read(self) -> bytes:
        return self._data


class FakeClientSession:
    """Test double for aiohttp ClientSession."""

    def __init__(self, responses: dict[str, tuple[int, bytes]]) -> None:
        self._responses = responses

    async def __aenter__(self) -> "FakeClientSession":
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass

    def get(self, url: str) -> "FakeContextManager":
        status, data = self._responses.get(url, (404, b"Not found"))
        return FakeContextManager(FakeResponse(status, data))


class FakeContextManager:
    """Test double for async context manager."""

    def __init__(self, response: FakeResponse) -> None:
        self._response = response

    async def __aenter__(self) -> FakeResponse:
        return self._response

    async def __aexit__(self, *args: Any) -> None:
        pass


# ============================================================================
# setup_vertex_credentials tests
# ============================================================================


@pytest.mark.parametrize(
    ("input", "expected"),
    [
        pytest.param(
            {"GOOGLE_GENAI_USE_VERTEXAI": "false"},
            None,
            id="vertexai_disabled",
        ),
        pytest.param(
            {"GOOGLE_GENAI_USE_VERTEXAI": ""},
            None,
            id="vertexai_empty",
        ),
        pytest.param(
            {"GOOGLE_GENAI_USE_VERTEXAI": "true", "GOOGLE_SERVICE_ACCOUNT_JSON": ""},
            None,
            id="vertexai_true_no_json",
        ),
        pytest.param(
            {
                "GOOGLE_GENAI_USE_VERTEXAI": "true",
                "GOOGLE_SERVICE_ACCOUNT_JSON": '{"type": "service_account", "project_id": "test"}',
            },
            Path,
            id="vertexai_with_sa_json",
        ),
        pytest.param(
            {
                "GOOGLE_GENAI_USE_VERTEXAI": "true",
                "GOOGLE_APPLICATION_CREDENTIALS": '{"type": "service_account", "project_id": "test2"}',
            },
            Path,
            id="vertexai_with_gac_json",
        ),
        pytest.param(
            {
                "GOOGLE_GENAI_USE_VERTEXAI": "true",
                "GOOGLE_SERVICE_ACCOUNT_JSON": "not valid json",
            },
            None,
            id="invalid_json",
        ),
    ],
)
@pytest.mark.timeout(1.0)
def test_setup_vertex_credentials(
    input: dict[str, str],
    expected: type[Path] | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test setup_vertex_credentials function."""
    # Clear environment
    for key in [
        "GOOGLE_GENAI_USE_VERTEXAI",
        "GOOGLE_SERVICE_ACCOUNT_JSON",
        "GOOGLE_APPLICATION_CREDENTIALS",
    ]:
        monkeypatch.delenv(key, raising=False)

    # Set input environment
    for key, value in input.items():
        monkeypatch.setenv(key, value)

    result = setup_vertex_credentials()

    if expected is None:
        assert result is None
    else:
        assert isinstance(result, Path)
        assert result.exists()
        # Cleanup
        result.unlink()


# ============================================================================
# cleanup_credentials tests
# ============================================================================


@pytest.mark.parametrize(
    ("input", "expected"),
    [
        pytest.param(None, None, id="none_path"),
        pytest.param("nonexistent", None, id="nonexistent_path"),
        pytest.param("existing", None, id="existing_path"),
    ],
)
@pytest.mark.timeout(1.0)
def test_cleanup_credentials(
    input: str | None,
    expected: None,
    tmp_path: Path,
) -> None:
    """Test cleanup_credentials function."""
    if input == "existing":
        path = tmp_path / "creds.json"
        path.write_text('{"type": "service_account"}')
        cleanup_credentials(path)
        assert not path.exists()
    elif input == "nonexistent":
        path = tmp_path / "nonexistent.json"
        cleanup_credentials(path)
        assert not path.exists()
    else:
        cleanup_credentials(None)
        assert expected is None


# ============================================================================
# check_credentials tests
# ============================================================================


@pytest.mark.parametrize(
    ("input", "expected"),
    [
        pytest.param({}, False, id="no_credentials"),
        pytest.param({"GOOGLE_GENAI_USE_VERTEXAI": "true"}, True, id="vertexai_enabled"),
        pytest.param(
            {"GOOGLE_GENAI_USE_VERTEXAI": "true"}, True, id="vertexai_enabled"
        ),
        pytest.param(
            {"GOOGLE_GENAI_USE_VERTEXAI": "TRUE"}, True, id="vertexai_uppercase"
        ),
        pytest.param({"GEMINI_API_KEY": "test-key"}, True, id="api_key_set"),
        pytest.param(
            {"GOOGLE_GENAI_USE_VERTEXAI": "true", "GEMINI_API_KEY": "test-key"},
            True,
            id="both_credentials",
        ),
        pytest.param({"GEMINI_API_KEY": ""}, False, id="empty_api_key"),
        pytest.param(
            {"GOOGLE_GENAI_USE_VERTEXAI": "false"}, False, id="vertexai_false"
        ),
    ],
)
@pytest.mark.timeout(1.0)
def test_check_credentials(
    input: dict[str, str],
    expected: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test check_credentials function."""
    # Clear environment
    for key in ["GOOGLE_GENAI_USE_VERTEXAI", "GEMINI_API_KEY"]:
        monkeypatch.delenv(key, raising=False)

    # Set input environment
    for key, value in input.items():
        monkeypatch.setenv(key, value)

    result = check_credentials()
    assert result == expected


# ============================================================================
# create_client tests
# ============================================================================


@pytest.mark.parametrize(
    ("input", "expected"),
    [
        pytest.param(
            {"GOOGLE_GENAI_USE_VERTEXAI": "true"},
            "vertexai",
            id="vertexai_client",
        ),
        pytest.param(
            {"GEMINI_API_KEY": "test-api-key"},
            "api_key",
            id="api_key_client",
        ),
        pytest.param(
            {},
            RuntimeError,
            id="no_credentials",
        ),
    ],
)
@pytest.mark.timeout(1.0)
def test_create_client(
    input: dict[str, str],
    expected: str | type[Exception],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test create_client function."""
    # Clear environment
    for key in ["GOOGLE_GENAI_USE_VERTEXAI", "GEMINI_API_KEY"]:
        monkeypatch.delenv(key, raising=False)

    # Set input environment
    for key, value in input.items():
        monkeypatch.setenv(key, value)

    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            create_client()
    else:
        # Mock genai.Client to avoid actual API calls
        mock_client = MagicMock()
        monkeypatch.setattr("src.__main__.genai.Client", lambda **kwargs: mock_client)

        result = create_client()
        assert result == mock_client


# ============================================================================
# fetch tests
# ============================================================================


@pytest.mark.parametrize(
    ("input", "expected"),
    [
        pytest.param(
            {"uri": "file://existing", "file_data": b"test content"},
            b"test content",
            id="file_uri_existing",
        ),
        pytest.param(
            {"uri": "file://nonexistent"},
            None,
            id="file_uri_nonexistent",
        ),
        pytest.param(
            {"uri": "local_path", "file_data": b"local content"},
            b"local content",
            id="local_path_existing",
        ),
        pytest.param(
            {"uri": "nonexistent_local"},
            None,
            id="local_path_nonexistent",
        ),
        pytest.param(
            {
                "uri": "http://example.com/image.png",
                "http_status": 200,
                "http_data": b"http data",
            },
            b"http data",
            id="http_uri_success",
        ),
        pytest.param(
            {
                "uri": "https://example.com/image.png",
                "http_status": 200,
                "http_data": b"https data",
            },
            b"https data",
            id="https_uri_success",
        ),
        pytest.param(
            {
                "uri": "http://example.com/notfound",
                "http_status": 404,
                "http_data": b"",
            },
            None,
            id="http_uri_404",
        ),
        pytest.param(
            {"uri": "gs://invalid"},
            None,
            id="invalid_gcs_uri",
        ),
        pytest.param(
            {"uri": "unknown://something"},
            None,
            id="unsupported_scheme",
        ),
    ],
)
@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_fetch(
    input: dict[str, Any],
    expected: bytes | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test fetch function."""
    uri = input["uri"]

    # Handle file:// URIs
    if uri.startswith("file://"):
        if "file_data" in input:
            file_path = tmp_path / "testfile"
            file_path.write_bytes(input["file_data"])
            uri = f"file://{file_path}"
        else:
            uri = f"file://{tmp_path / 'nonexistent'}"

    # Handle local paths
    elif not uri.startswith(("http://", "https://", "gs://", "unknown://")):
        if "file_data" in input:
            file_path = tmp_path / input["uri"]
            file_path.write_bytes(input["file_data"])
            uri = str(file_path)
        else:
            uri = str(tmp_path / "nonexistent_local_file")

    # Handle HTTP/HTTPS URIs
    if uri.startswith(("http://", "https://")):
        responses = {uri: (input.get("http_status", 404), input.get("http_data", b""))}

        async def fake_client_session() -> FakeClientSession:
            return FakeClientSession(responses)

        monkeypatch.setattr(
            "aiohttp.ClientSession", lambda: FakeClientSession(responses)
        )
    result = await fetch(uri)
    assert result == expected


# ============================================================================
# app_lifespan tests
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_app_lifespan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test app_lifespan context manager."""
    images_dir = tmp_path / "images"
    videos_dir = tmp_path / "videos"

    monkeypatch.setenv("DATA_FOLDER", str(tmp_path))
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    # Mock genai.Client
    mock_client = MagicMock()
    monkeypatch.setattr("src.__main__.genai.Client", lambda **kwargs: mock_client)

    server = FakeFastMCP()

    async with app_lifespan(server) as ctx:  # type: ignore[arg-type]
        assert isinstance(ctx, AppContext)
        assert ctx.images_dir == images_dir
        assert ctx.videos_dir == videos_dir
        assert ctx.client == mock_client
        assert images_dir.exists()
        assert videos_dir.exists()


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_app_lifespan_default_dirs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test app_lifespan with default directories."""
    # Clear environment
    monkeypatch.delenv("DATA_FOLDER", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    # Mock genai.Client
    mock_client = MagicMock()
    monkeypatch.setattr("src.__main__.genai.Client", lambda **kwargs: mock_client)

    server = FakeFastMCP()

    async with app_lifespan(server) as ctx:  # type: ignore[arg-type]
        assert ctx.images_dir == Path("data/images")
        assert ctx.videos_dir == Path("data/videos")


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_app_lifespan_cleanup_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test app_lifespan cleans up temporary credentials."""
    images_dir = tmp_path / "images"
    videos_dir = tmp_path / "videos"

    monkeypatch.setenv("DATA_FOLDER", str(tmp_path))
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.setenv(
        "GOOGLE_SERVICE_ACCOUNT_JSON",
        '{"type": "service_account", "project_id": "test"}',
    )

    # Mock genai.Client
    mock_client = MagicMock()
    monkeypatch.setattr("src.__main__.genai.Client", lambda **kwargs: mock_client)

    server = FakeFastMCP()
    temp_creds_path: Path | None = None

    async with app_lifespan(server) as ctx:  # type: ignore[arg-type]
        temp_creds_path = ctx.temp_creds_path
        if temp_creds_path:
            assert temp_creds_path.exists()

    # After context exit, temp credentials should be cleaned up
    if temp_creds_path:
        assert not temp_creds_path.exists()


# ============================================================================
# Container detection tests
# ============================================================================


def test_is_running_in_container_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test container detection via RUNNING_IN_CONTAINER env var."""
    monkeypatch.setenv("RUNNING_IN_CONTAINER", "true")
    assert is_running_in_container() is True

    monkeypatch.setenv("RUNNING_IN_CONTAINER", "TRUE")
    assert is_running_in_container() is True

    monkeypatch.setenv("RUNNING_IN_CONTAINER", "false")
    assert is_running_in_container() is False


def test_is_running_in_container_dockerenv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test container detection via /.dockerenv file."""
    monkeypatch.delenv("RUNNING_IN_CONTAINER", raising=False)
    # Without /.dockerenv, should return False (unless actually in container)
    # We can't easily test the /.dockerenv path without mocking Path


def test_is_running_in_container_not_in_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test container detection returns False when not in container."""
    monkeypatch.delenv("RUNNING_IN_CONTAINER", raising=False)
    # This will check /.dockerenv which shouldn't exist on host
    # Result depends on actual environment, but env var takes precedence
    monkeypatch.setenv("RUNNING_IN_CONTAINER", "false")
    assert is_running_in_container() is False


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_app_lifespan_container_requires_data_folder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test app_lifespan raises error in container without DATA_FOLDER."""
    monkeypatch.setenv("RUNNING_IN_CONTAINER", "true")
    monkeypatch.delenv("DATA_FOLDER", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    mock_client = MagicMock()
    monkeypatch.setattr("src.__main__.genai.Client", lambda **kwargs: mock_client)

    server = FakeFastMCP()

    with pytest.raises(ValueError, match="DATA_FOLDER must be set"):
        async with app_lifespan(server) as ctx:  # type: ignore[arg-type]
            pass


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_app_lifespan_container_with_data_folder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test app_lifespan works in container with DATA_FOLDER set."""
    monkeypatch.setenv("RUNNING_IN_CONTAINER", "true")
    monkeypatch.setenv("DATA_FOLDER", str(tmp_path))
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    mock_client = MagicMock()
    monkeypatch.setattr("src.__main__.genai.Client", lambda **kwargs: mock_client)

    server = FakeFastMCP()

    async with app_lifespan(server) as ctx:  # type: ignore[arg-type]
        assert ctx.images_dir == tmp_path / "images"
        assert ctx.videos_dir == tmp_path / "videos"
        assert ctx.images_dir.exists()
        assert ctx.videos_dir.exists()


# ============================================================================
# generate_image tool tests
# ============================================================================


def _create_test_image(width: int = 100, height: int = 100) -> bytes:
    """Create a test image and return bytes."""
    img = Image.new("RGB", (width, height), color="red")
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img.close()
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("input", "expected"),
    [
        pytest.param(
            {
                "prompt": "A red square",
                "model": "gemini-2.5-flash-image",
                "image_uri": None,
                "image_base64": None,
            },
            {"success": True, "has_image": True},
            id="text_prompt_only",
        ),
        pytest.param(
            {
                "prompt": "Edit this image",
                "model": "gemini-2.5-flash-image",
                "image_uri": None,
                "image_base64": "base64_image",
            },
            {"success": True, "has_image": True},
            id="with_base64_image",
        ),
        pytest.param(
            {
                "prompt": "A" * 10000,
                "model": "gemini-2.5-flash-image",
                "image_uri": None,
                "image_base64": None,
            },
            {"success": True, "has_image": True},
            id="large_prompt",
        ),
        pytest.param(
            {
                "prompt": "Unicode test: 🎨 日本語 émoji",
                "model": "gemini-2.5-flash-image",
                "image_uri": None,
                "image_base64": None,
            },
            {"success": True, "has_image": True},
            id="unicode_prompt",
        ),
        pytest.param(
            {
                "prompt": "",
                "model": "gemini-2.5-flash-image",
                "image_uri": None,
                "image_base64": None,
            },
            {"success": True, "has_image": True},
            id="empty_prompt",
        ),
        pytest.param(
            {
                "prompt": "Test imagen",
                "model": "imagen-3.0-generate-002",
                "image_uri": None,
                "image_base64": None,
            },
            {"success": True, "has_image": True},
            id="imagen_model",
        ),
        pytest.param(
            {
                "prompt": "Generate fail",
                "model": "gemini-2.5-flash-image",
                "image_uri": None,
                "image_base64": None,
                "should_fail": True,
            },
            {"success": False, "error": True},
            id="generation_error",
        ),
    ],
)
@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_image(
    input: dict[str, Any],
    expected: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test generate_image tool."""
    from src.__main__ import generate_image

    images_dir = tmp_path / "images"
    images_dir.mkdir()

    # Create mock context
    mock_ctx = MagicMock()
    mock_ctx.info = AsyncMock()
    mock_ctx.error = AsyncMock()

    mock_app_ctx = AppContext(
        images_dir=images_dir,
        videos_dir=tmp_path / "videos",
        client=MagicMock(),
    )
    mock_ctx.request_context.lifespan_context = mock_app_ctx

    # Create test image for base64 input
    test_image_bytes = _create_test_image()
    if input.get("image_base64") == "base64_image":
        input["image_base64"] = base64.b64encode(test_image_bytes).decode("utf-8")

    # Mock generate_image_impl
    async def mock_generate_impl(**kwargs: Any) -> dict[str, Any]:
        if input.get("should_fail"):
            raise ValueError("Generation failed")

        # Create a real image file
        filename = "test_output.png"
        filepath = images_dir / filename
        filepath.write_bytes(test_image_bytes)

        # Create thumbnail preview
        thumb_b64 = base64.b64encode(test_image_bytes).decode("utf-8")

        return {
            "message": "Image generated successfully",
            "image_url": f"file://{filepath}",
            "image_preview": f"data:image/jpeg;base64,{thumb_b64}",
            "prompt": kwargs.get("prompt", ""),
            "model": kwargs.get("model", ""),
        }

    monkeypatch.setattr("src.__main__.generate_image_impl", mock_generate_impl)

    result = await generate_image(
        ctx=mock_ctx,
        prompt=input["prompt"],
        model=input["model"],
        image_uri=input.get("image_uri"),
        image_base64=input.get("image_base64"),
    )

    if expected.get("success"):
        assert len(result) == 2
        # Check Image is returned
        from mcp.server.fastmcp import Image as MCPImage

        assert isinstance(result[0], MCPImage)
    else:
        assert len(result) == 1
        assert "error" in result[0].text


# ============================================================================
# generate_video tool tests
# ============================================================================


@pytest.mark.parametrize(
    ("input", "expected"),
    [
        pytest.param(
            {
                "prompt": "A cat walking",
                "model": "veo-2.0-generate-001",
                "aspect_ratio": "16:9",
                "duration_seconds": 5.0,
            },
            {"success": True},
            id="basic_video_generation",
        ),
        pytest.param(
            {
                "prompt": "A dog running",
                "model": "veo-3.1-generate-preview",
                "aspect_ratio": "9:16",
                "duration_seconds": 8.0,
                "include_audio": True,
                "audio_prompt": "Barking sounds",
            },
            {"success": True},
            id="veo3_with_audio",
        ),
        pytest.param(
            {
                "prompt": "A" * 10000,
                "model": "veo-2.0-generate-001",
                "aspect_ratio": "16:9",
                "duration_seconds": 5.0,
            },
            {"success": True},
            id="large_prompt",
        ),
        pytest.param(
            {
                "prompt": "Negative test",
                "model": "veo-2.0-generate-001",
                "aspect_ratio": "16:9",
                "duration_seconds": 5.0,
                "negative_prompt": "blurry, distorted",
            },
            {"success": True},
            id="with_negative_prompt",
        ),
        pytest.param(
            {
                "prompt": "Seeded video",
                "model": "veo-2.0-generate-001",
                "aspect_ratio": "16:9",
                "duration_seconds": 5.0,
                "seed": 42,
            },
            {"success": True},
            id="with_seed",
        ),
        pytest.param(
            {
                "prompt": "Fail video",
                "model": "veo-2.0-generate-001",
                "aspect_ratio": "16:9",
                "duration_seconds": 5.0,
                "should_fail": True,
            },
            {"success": False, "error": True},
            id="generation_error",
        ),
    ],
)
@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_video(
    input: dict[str, Any],
    expected: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test generate_video tool."""
    from src.__main__ import generate_video

    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    # Create mock context
    mock_ctx = MagicMock()
    mock_ctx.info = AsyncMock()
    mock_ctx.error = AsyncMock()

    mock_app_ctx = AppContext(
        images_dir=tmp_path / "images",
        videos_dir=videos_dir,
        client=MagicMock(),
    )
    mock_ctx.request_context.lifespan_context = mock_app_ctx

    # Mock generate_video_impl
    async def mock_generate_impl(**kwargs: Any) -> dict[str, Any]:
        if input.get("should_fail"):
            raise ValueError("Video generation failed")

        return {
            "message": "Video generated successfully",
            "video_url": f"file://{videos_dir}/test.mp4",
            "prompt": kwargs.get("prompt", ""),
            "model": kwargs.get("model", ""),
            "audio_enabled": kwargs.get("include_audio", False),
        }

    monkeypatch.setattr("src.__main__.generate_video_impl", mock_generate_impl)

    result = await generate_video(
        ctx=mock_ctx,
        prompt=input["prompt"],
        model=input["model"],
        aspect_ratio=input.get("aspect_ratio", "16:9"),
        duration_seconds=input.get("duration_seconds", 5.0),
        include_audio=input.get("include_audio", False),
        audio_prompt=input.get("audio_prompt"),
        negative_prompt=input.get("negative_prompt"),
        seed=input.get("seed"),
        image_uri=input.get("image_uri"),
        image_base64=input.get("image_base64"),
    )

    result_json = json.loads(result)

    if expected.get("success"):
        assert "error" not in result_json
        assert result_json["message"] == "Video generated successfully"
    else:
        assert "error" in result_json


# ============================================================================
# main function tests
# ============================================================================


@pytest.mark.parametrize(
    ("input", "expected"),
    [
        pytest.param(
            {"args": [], "has_credentials": True},
            {"exit_code": None},
            id="default_stdio_transport",
        ),
        pytest.param(
            {"args": ["stdio"], "has_credentials": True},
            {"exit_code": None},
            id="explicit_stdio",
        ),
        pytest.param(
            {"args": ["sse"], "has_credentials": True},
            {"exit_code": None},
            id="sse_transport",
        ),
        pytest.param(
            {"args": ["streamable-http"], "has_credentials": True},
            {"exit_code": None},
            id="http_transport",
        ),
        pytest.param(
            {"args": [], "has_credentials": False},
            {"exit_code": 1},
            id="no_credentials_exits",
        ),
        pytest.param(
            {"args": ["--log-level", "DEBUG"], "has_credentials": True},
            {"exit_code": None},
            id="custom_log_level",
        ),
    ],
)
@pytest.mark.timeout(2.0)
def test_main(
    input: dict[str, Any],
    expected: dict[str, int | None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test main function."""
    import sys

    from src.__main__ import main

    # Set up sys.argv
    argv = ["gemini-media-mcp"] + input.get("args", [])
    monkeypatch.setattr(sys, "argv", argv)

    # Set credentials
    if input.get("has_credentials"):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    else:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)

    # Mock mcp.run to prevent actual server startup
    mock_run = MagicMock()
    monkeypatch.setattr("src.__main__.mcp.run", mock_run)

    if expected.get("exit_code") is not None:
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == expected["exit_code"]
    else:
        main()
        mock_run.assert_called_once()
