"""Tests for __main__.py MCP server."""

import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from src.__main__ import (
    AppContext,
    _client_for_video_model,
    app_lifespan,
    check_credentials,
    cleanup_credentials,
    create_client,
    create_gemini_api_client,
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


class FakeContentStream:
    """Test double for aiohttp response content with iter_chunked."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    async def _iter(self, n: int) -> Any:
        if self._data:
            yield self._data

    def iter_chunked(self, n: int) -> Any:
        return self._iter(n)


class FakeResponse:
    """Test double for aiohttp response."""

    def __init__(self, status: int, data: bytes) -> None:
        self.status = status
        self._data = data
        self.content_length = len(data) if data else 0
        self.content = FakeContentStream(data)

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
# Veo Lite routing tests
# ============================================================================


def _make_fake_client(vertexai: bool) -> MagicMock:
    client = MagicMock()
    client._api_client = MagicMock()
    client._api_client.vertexai = vertexai
    return client


def test_create_gemini_api_client_returns_none_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert create_gemini_api_client() is None


def test_create_gemini_api_client_builds_client_with_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    fake = MagicMock()
    monkeypatch.setattr("src.__main__.genai.Client", lambda **kwargs: fake)
    assert create_gemini_api_client() is fake


def test_client_for_video_model_routes_lite_through_api_client(
    tmp_path: Path,
) -> None:
    vertex_client = _make_fake_client(vertexai=True)
    api_client = _make_fake_client(vertexai=False)
    ctx = AppContext(
        data_folder=tmp_path,
        images_dir=tmp_path,
        videos_dir=tmp_path,
        client=vertex_client,
        gemini_api_client=api_client,
    )
    picked = _client_for_video_model(ctx, "veo-3.1-lite-generate-preview")
    assert picked is api_client


def test_client_for_video_model_uses_main_client_for_non_lite(
    tmp_path: Path,
) -> None:
    vertex_client = _make_fake_client(vertexai=True)
    api_client = _make_fake_client(vertexai=False)
    ctx = AppContext(
        data_folder=tmp_path,
        images_dir=tmp_path,
        videos_dir=tmp_path,
        client=vertex_client,
        gemini_api_client=api_client,
    )
    picked = _client_for_video_model(ctx, "veo-3.1-fast-generate-001")
    assert picked is vertex_client


def test_client_for_video_model_lite_without_api_key_raises(
    tmp_path: Path,
) -> None:
    vertex_client = _make_fake_client(vertexai=True)
    ctx = AppContext(
        data_folder=tmp_path,
        images_dir=tmp_path,
        videos_dir=tmp_path,
        client=vertex_client,
        gemini_api_client=None,
    )
    with pytest.raises(RuntimeError, match="only available via the Gemini API"):
        _client_for_video_model(ctx, "veo-3.1-lite-generate-preview")


def test_client_for_video_model_lite_on_api_only_passes_through(
    tmp_path: Path,
) -> None:
    """When primary client is already the Gemini API, don't require a second one."""
    api_client = _make_fake_client(vertexai=False)
    ctx = AppContext(
        data_folder=tmp_path,
        images_dir=tmp_path,
        videos_dir=tmp_path,
        client=api_client,
        gemini_api_client=None,
    )
    picked = _client_for_video_model(ctx, "veo-3.1-lite-generate-preview")
    assert picked is api_client


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
            "aiohttp.ClientSession",
            lambda *args, **kwargs: FakeClientSession(responses),
        )
        # Bypass SSRF host-resolution check (no DNS in unit tests)
        monkeypatch.setattr(
            "src.__main__._assert_http_host_public", lambda url: None
        )
    result = await fetch(uri, allowed_dir=tmp_path)
    assert result == expected


# ============================================================================
# Security: path traversal / LFI prevention
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_fetch_rejects_path_traversal(tmp_path: Path) -> None:
    """fetch() must reject paths outside allowed_dir."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_bytes(b"secret")

    # Absolute path outside allowed_dir
    result = await fetch(f"file://{outside}", allowed_dir=data_dir)
    assert result is None

    # Traversal via ../ inside file://
    result = await fetch(f"file://{data_dir}/../secret.txt", allowed_dir=data_dir)
    assert result is None


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_fetch_rejects_local_without_allowed_dir(tmp_path: Path) -> None:
    """fetch() must reject local file access when allowed_dir is not provided."""
    target = tmp_path / "f.txt"
    target.write_bytes(b"data")
    result = await fetch(f"file://{target}")
    assert result is None


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_fetch_allows_file_inside_allowed_dir(tmp_path: Path) -> None:
    """fetch() must allow files inside allowed_dir."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    target = data_dir / "f.txt"
    target.write_bytes(b"ok")
    result = await fetch(f"file://{target}", allowed_dir=data_dir)
    assert result == b"ok"


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_fetch_rejects_private_ip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fetch() must reject http URLs resolving to private / loopback IPs."""
    import socket as _socket

    # Simulate hostname resolving to a loopback address.
    def fake_getaddrinfo(host: str, *args: Any, **kwargs: Any) -> list[Any]:
        return [(0, 0, 0, "", ("127.0.0.1", 0))]

    monkeypatch.setattr(_socket, "getaddrinfo", fake_getaddrinfo)
    result = await fetch("http://evil.example.com/x", allowed_dir=tmp_path)
    assert result is None


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_fetch_rejects_metadata_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fetch() must block cloud-metadata link-local IPs."""
    import socket as _socket

    def fake_getaddrinfo(host: str, *args: Any, **kwargs: Any) -> list[Any]:
        return [(0, 0, 0, "", ("169.254.169.254", 0))]

    monkeypatch.setattr(_socket, "getaddrinfo", fake_getaddrinfo)
    result = await fetch(
        "http://metadata.google.internal/", allowed_dir=tmp_path
    )
    assert result is None


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_fetch_enforces_size_cap_http(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fetch() must cap HTTP response size."""
    big = b"x" * 1024
    responses = {"https://example.com/big": (200, big)}
    monkeypatch.setattr(
        "aiohttp.ClientSession",
        lambda *a, **kw: FakeClientSession(responses),
    )
    monkeypatch.setattr("src.__main__._assert_http_host_public", lambda url: None)
    # Cap smaller than payload
    result = await fetch(
        "https://example.com/big", allowed_dir=tmp_path, max_bytes=100
    )
    assert result is None


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_fetch_enforces_size_cap_local(tmp_path: Path) -> None:
    """fetch() must cap local file size."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    target = data_dir / "big.bin"
    target.write_bytes(b"x" * 1024)
    result = await fetch(
        f"file://{target}", allowed_dir=data_dir, max_bytes=100
    )
    assert result is None


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_fetch_gcs_allowlist_rejects_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fetch() must reject gs:// buckets outside the allowlist."""
    from src.__main__ import fetch as fetch_fn

    # Storage client should never be touched.
    def fail_client() -> Any:
        raise AssertionError("storage.Client should not be constructed")

    monkeypatch.setattr("src.__main__.storage.Client", fail_client)
    result = await fetch_fn(
        "gs://forbidden/path.mp4",
        allowed_dir=tmp_path,
        allowed_gcs_buckets=frozenset({"mybucket"}),
    )
    assert result is None


def test_compute_allowed_gcs_buckets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Allowlist combines GCS_ALLOWED_BUCKETS and VIDEO_GCS_BUCKET."""
    from src.__main__ import _compute_allowed_gcs_buckets

    monkeypatch.setenv("GCS_ALLOWED_BUCKETS", "a, b ,c")
    monkeypatch.setenv("VIDEO_GCS_BUCKET", "gs://d/sub/")
    result = _compute_allowed_gcs_buckets()
    assert result == frozenset({"a", "b", "c", "d"})


def test_compute_allowed_gcs_buckets_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty env yields empty allowlist."""
    from src.__main__ import _compute_allowed_gcs_buckets

    monkeypatch.delenv("GCS_ALLOWED_BUCKETS", raising=False)
    monkeypatch.delenv("VIDEO_GCS_BUCKET", raising=False)
    assert _compute_allowed_gcs_buckets() == frozenset()


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_fetch_rejects_symlink_escape(tmp_path: Path) -> None:
    """fetch() must reject symlinks that escape allowed_dir."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_bytes(b"secret")
    link = data_dir / "link.txt"
    try:
        link.symlink_to(secret)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported")
    result = await fetch(f"file://{link}", allowed_dir=data_dir)
    assert result is None


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
    # Clear environment and ensure not detected as container
    monkeypatch.delenv("DATA_FOLDER", raising=False)
    monkeypatch.delenv("RUNNING_IN_CONTAINER", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    # Mock Path.exists to return False for /.dockerenv
    original_exists = Path.exists
    def mock_exists(self: Path) -> bool:
        if str(self) == "/.dockerenv":
            return False
        return original_exists(self)
    monkeypatch.setattr(Path, "exists", mock_exists)

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


def test_is_running_in_container_dockerenv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test container detection via /.dockerenv file."""
    monkeypatch.delenv("RUNNING_IN_CONTAINER", raising=False)

    # Mock Path.exists to control /.dockerenv detection
    original_exists = Path.exists
    def mock_exists_true(self: Path) -> bool:
        if str(self) == "/.dockerenv":
            return True
        return original_exists(self)
    def mock_exists_false(self: Path) -> bool:
        if str(self) == "/.dockerenv":
            return False
        return original_exists(self)

    monkeypatch.setattr(Path, "exists", mock_exists_true)
    assert is_running_in_container() is True

    monkeypatch.setattr(Path, "exists", mock_exists_false)
    assert is_running_in_container() is False


def test_is_running_in_container_not_in_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test container detection returns False when not in container."""
    monkeypatch.delenv("RUNNING_IN_CONTAINER", raising=False)

    # Mock Path.exists to return False for /.dockerenv
    original_exists = Path.exists
    def mock_exists(self: Path) -> bool:
        if str(self) == "/.dockerenv":
            return False
        return original_exists(self)
    monkeypatch.setattr(Path, "exists", mock_exists)

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
        async with app_lifespan(server) as _ctx:  # type: ignore[arg-type]
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
        data_folder=tmp_path,
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
                "model": "veo-3.1-generate-001",
                "aspect_ratio": "16:9",
                "duration_seconds": 5.0,
            },
            {"success": True},
            id="basic_video_generation",
        ),
        pytest.param(
            {
                "prompt": "A dog running",
                "model": "veo-3.1-generate-001",
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
                "model": "veo-3.1-generate-001",
                "aspect_ratio": "16:9",
                "duration_seconds": 5.0,
            },
            {"success": True},
            id="large_prompt",
        ),
        pytest.param(
            {
                "prompt": "Negative test",
                "model": "veo-3.1-generate-001",
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
                "model": "veo-3.1-generate-001",
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
                "model": "veo-3.1-generate-001",
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
        data_folder=tmp_path,
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
# Sidecar manifest tests
# ============================================================================


def test_write_sidecar_writes_json_next_to_media(tmp_path: Path) -> None:
    """Sidecar is written as <stem>.json next to a file:// media URL."""
    from src.__main__ import _write_sidecar

    media = tmp_path / "abc.mp4"
    media.write_bytes(b"x")
    sidecar_url = _write_sidecar(f"file://{media}", {"kind": "video", "a": 1})
    assert sidecar_url == f"file://{tmp_path / 'abc.json'}"
    assert (tmp_path / "abc.json").exists()
    import json as _json

    content = _json.loads((tmp_path / "abc.json").read_text())
    assert content["kind"] == "video"
    assert content["a"] == 1


def test_write_sidecar_skips_remote(tmp_path: Path) -> None:
    """Remote URIs (gs://) do not yield a local sidecar."""
    from src.__main__ import _write_sidecar

    assert _write_sidecar("gs://bucket/obj.mp4", {"kind": "video"}) is None


# ============================================================================
# generate_transition tool tests
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_transition_happy_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """generate_transition fetches both frames and produces a video."""
    from src.__main__ import generate_transition

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    # Two frames stored inside DATA_FOLDER so fetch() accepts them.
    frame_a = images_dir / "a.png"
    frame_b = images_dir / "b.png"
    frame_a.write_bytes(_create_test_image())
    frame_b.write_bytes(_create_test_image())

    mock_ctx = MagicMock()
    mock_ctx.info = AsyncMock()
    mock_ctx.error = AsyncMock()
    mock_ctx.request_context.lifespan_context = AppContext(
        data_folder=tmp_path,
        images_dir=images_dir,
        videos_dir=videos_dir,
        client=MagicMock(),
    )

    async def mock_impl(**kwargs: Any) -> dict[str, Any]:
        out = videos_dir / "out.mp4"
        out.write_bytes(b"mp4")
        return {
            "message": "Video generated successfully",
            "video_url": f"file://{out}",
            "prompt": kwargs.get("prompt", ""),
            "model": kwargs.get("model", ""),
            "audio_enabled": False,
            "generation_mode": "first_last_frame",
        }

    monkeypatch.setattr("src.__main__.generate_video_impl", mock_impl)

    result_json = await generate_transition(
        ctx=mock_ctx,
        first_frame_uri=f"file://{frame_a}",
        last_frame_uri=f"file://{frame_b}",
        prompt="crossfade",
    )
    result = json.loads(result_json)
    assert result["generation_mode"] == "first_last_frame"
    assert result["first_frame_uri"] == f"file://{frame_a}"
    assert result["last_frame_uri"] == f"file://{frame_b}"
    assert result["sidecar_url"].endswith("out.json")

    # Sidecar content is the manifest.
    sidecar = Path(result["sidecar_url"][7:])
    assert sidecar.exists()
    manifest = json.loads(sidecar.read_text())
    assert manifest["kind"] == "transition"
    assert manifest["first_frame_uri"] == f"file://{frame_a}"


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_transition_missing_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """generate_transition returns an error JSON when a frame cannot be fetched."""
    from src.__main__ import generate_transition

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    mock_ctx = MagicMock()
    mock_ctx.info = AsyncMock()
    mock_ctx.error = AsyncMock()
    mock_ctx.request_context.lifespan_context = AppContext(
        data_folder=tmp_path,
        images_dir=images_dir,
        videos_dir=videos_dir,
        client=MagicMock(),
    )

    async def should_not_run(**kwargs: Any) -> dict[str, Any]:
        raise AssertionError("impl should not be called when a frame is missing")

    monkeypatch.setattr("src.__main__.generate_video_impl", should_not_run)

    result_json = await generate_transition(
        ctx=mock_ctx,
        first_frame_uri=f"file://{tmp_path / 'missing_a.png'}",
        last_frame_uri=f"file://{tmp_path / 'missing_b.png'}",
    )
    result = json.loads(result_json)
    assert "error" in result


# ============================================================================
# generate_bridge tool tests
# ============================================================================


def _make_fake_mp4() -> bytes:
    """Encode a minimal MP4 with two solid-color frames."""
    import imageio.v3 as iio
    import numpy as np

    frames = np.stack([
        np.full((64, 64, 3), (200, 30, 30), dtype=np.uint8),
        np.full((64, 64, 3), (30, 30, 200), dtype=np.uint8),
    ])
    buf = BytesIO()
    iio.imwrite(
        buf,
        frames,
        extension=".mp4",
        fps=8,
        codec="libx264",
        pixelformat="yuv420p",
    )
    return buf.getvalue()


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
async def test_generate_bridge_happy_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """generate_bridge extracts frames from two clips, produces a video."""
    from src.__main__ import generate_bridge

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    clip_a = videos_dir / "a.mp4"
    clip_b = videos_dir / "b.mp4"
    clip_a.write_bytes(_make_fake_mp4())
    clip_b.write_bytes(_make_fake_mp4())

    mock_ctx = MagicMock()
    mock_ctx.info = AsyncMock()
    mock_ctx.error = AsyncMock()
    mock_ctx.request_context.lifespan_context = AppContext(
        data_folder=tmp_path,
        images_dir=images_dir,
        videos_dir=videos_dir,
        client=MagicMock(),
    )

    captured: dict[str, Any] = {}

    async def mock_impl(**kwargs: Any) -> dict[str, Any]:
        # Record what the tool passed through so we can assert frames were
        # extracted and supplied.
        captured["image_bytes_len"] = len(kwargs.get("image_bytes") or b"")
        captured["last_frame_bytes_len"] = len(
            kwargs.get("last_frame_bytes") or b""
        )
        out = videos_dir / "bridge.mp4"
        out.write_bytes(b"bridge-mp4")
        return {
            "message": "Video generated successfully",
            "video_url": f"file://{out}",
            "prompt": kwargs.get("prompt", ""),
            "model": kwargs.get("model", ""),
            "audio_enabled": False,
            "generation_mode": "first_last_frame",
        }

    monkeypatch.setattr("src.__main__.generate_video_impl", mock_impl)

    result_json = await generate_bridge(
        ctx=mock_ctx,
        from_clip_uri=f"file://{clip_a}",
        to_clip_uri=f"file://{clip_b}",
        prompt="dissolve",
    )
    result = json.loads(result_json)

    assert result["generation_mode"] == "first_last_frame"
    assert result["from_clip_uri"] == f"file://{clip_a}"
    assert result["to_clip_uri"] == f"file://{clip_b}"
    assert result["sidecar_url"].endswith("bridge.json")

    # Confirm we actually extracted frames and fed them through.
    assert captured["image_bytes_len"] > 0
    assert captured["last_frame_bytes_len"] > 0

    sidecar = Path(result["sidecar_url"][7:])
    manifest = json.loads(sidecar.read_text())
    assert manifest["kind"] == "bridge"
    assert manifest["from_clip_uri"] == f"file://{clip_a}"


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_generate_bridge_missing_clip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing source clip yields an error JSON, no impl call."""
    from src.__main__ import generate_bridge

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    mock_ctx = MagicMock()
    mock_ctx.info = AsyncMock()
    mock_ctx.error = AsyncMock()
    mock_ctx.request_context.lifespan_context = AppContext(
        data_folder=tmp_path,
        images_dir=images_dir,
        videos_dir=videos_dir,
        client=MagicMock(),
    )

    async def should_not_run(**kwargs: Any) -> dict[str, Any]:
        raise AssertionError("impl should not be called when a clip is missing")

    monkeypatch.setattr("src.__main__.generate_video_impl", should_not_run)

    result_json = await generate_bridge(
        ctx=mock_ctx,
        from_clip_uri=f"file://{tmp_path / 'missing_a.mp4'}",
        to_clip_uri=f"file://{tmp_path / 'missing_b.mp4'}",
    )
    assert "error" in json.loads(result_json)


# ============================================================================
# generate_clip tool tests
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(10.0)
async def test_generate_clip_three_beats_no_bridges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three beats without bridges produce three segments and correct duration."""
    from src.__main__ import generate_clip

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    mock_ctx = MagicMock()
    mock_ctx.info = AsyncMock()
    mock_ctx.error = AsyncMock()
    mock_ctx.request_context.lifespan_context = AppContext(
        data_folder=tmp_path,
        images_dir=images_dir,
        videos_dir=videos_dir,
        client=MagicMock(),
    )

    call_index = {"n": 0}

    async def mock_impl(**kwargs: Any) -> dict[str, Any]:
        call_index["n"] += 1
        out = videos_dir / f"beat{call_index['n']}.mp4"
        out.write_bytes(_make_fake_mp4())
        return {
            "message": "Video generated successfully",
            "video_url": f"file://{out}",
            "prompt": kwargs.get("prompt", ""),
            "model": kwargs.get("model", ""),
            "audio_enabled": kwargs.get("include_audio", False),
            "generation_mode": "text_to_video",
            "duration_seconds": kwargs.get("duration_seconds"),
        }

    monkeypatch.setattr("src.__main__.generate_video_impl", mock_impl)

    result_json = await generate_clip(
        ctx=mock_ctx,
        beats=[
            {"prompt": "hook", "duration_seconds": 4},
            {"prompt": "body", "duration_seconds": 6},
            {"prompt": "outro", "duration_seconds": 4},
        ],
        add_bridges=False,
    )
    result = json.loads(result_json)

    assert result["kind"] == "clip"
    assert result["aspect_ratio"] == "9:16"
    assert result["beat_count"] == 3
    assert len(result["segments"]) == 3
    assert all(seg["kind"] == "beat" for seg in result["segments"])
    assert result["total_duration_seconds"] == 14.0


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
async def test_generate_clip_with_bridges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bridges are inserted between consecutive beats."""
    from src.__main__ import generate_clip

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    mock_ctx = MagicMock()
    mock_ctx.info = AsyncMock()
    mock_ctx.error = AsyncMock()
    mock_ctx.request_context.lifespan_context = AppContext(
        data_folder=tmp_path,
        images_dir=images_dir,
        videos_dir=videos_dir,
        client=MagicMock(),
    )

    call_index = {"n": 0}
    fake_mp4 = _make_fake_mp4()

    async def mock_impl(**kwargs: Any) -> dict[str, Any]:
        call_index["n"] += 1
        out = videos_dir / f"seg{call_index['n']}.mp4"
        out.write_bytes(fake_mp4)
        return {
            "message": "Video generated successfully",
            "video_url": f"file://{out}",
            "prompt": kwargs.get("prompt", ""),
            "model": kwargs.get("model", ""),
            "audio_enabled": False,
            "generation_mode": (
                "first_last_frame"
                if kwargs.get("last_frame_bytes") is not None
                else "text_to_video"
            ),
            "duration_seconds": kwargs.get("duration_seconds"),
        }

    monkeypatch.setattr("src.__main__.generate_video_impl", mock_impl)

    result_json = await generate_clip(
        ctx=mock_ctx,
        beats=[
            {"prompt": "hook", "duration_seconds": 4},
            {"prompt": "body", "duration_seconds": 4},
        ],
        add_bridges=True,
    )
    result = json.loads(result_json)

    # 2 beats + 1 bridge = 3 segments, ordered [beat, bridge, beat].
    assert result["beat_count"] == 2
    assert len(result["segments"]) == 3
    kinds = [seg["kind"] for seg in result["segments"]]
    assert kinds == ["beat", "bridge", "beat"]
    # beat(4) + bridge(4) + beat(4) = 12
    assert result["total_duration_seconds"] == 12.0


@pytest.mark.asyncio
@pytest.mark.timeout(10.0)
async def test_generate_clip_partial_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If one beat fails the clip still returns a manifest for the successful
    beats, plus an `errors` entry identifying the failed beat index."""
    from src.__main__ import generate_clip

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    mock_ctx = MagicMock()
    mock_ctx.info = AsyncMock()
    mock_ctx.error = AsyncMock()
    mock_ctx.request_context.lifespan_context = AppContext(
        data_folder=tmp_path,
        images_dir=images_dir,
        videos_dir=videos_dir,
        client=MagicMock(),
    )

    call_index = {"n": 0}

    async def mock_impl(**kwargs: Any) -> dict[str, Any]:
        call_index["n"] += 1
        if call_index["n"] == 2:
            raise RuntimeError("simulated API failure on beat 2")
        out = videos_dir / f"beat{call_index['n']}.mp4"
        out.write_bytes(_make_fake_mp4())
        return {
            "message": "Video generated successfully",
            "video_url": f"file://{out}",
            "prompt": kwargs.get("prompt", ""),
            "model": kwargs.get("model", ""),
            "audio_enabled": False,
            "generation_mode": "text_to_video",
            "duration_seconds": kwargs.get("duration_seconds"),
        }

    monkeypatch.setattr("src.__main__.generate_video_impl", mock_impl)

    result_json = await generate_clip(
        ctx=mock_ctx,
        beats=[
            {"prompt": "a", "duration_seconds": 4},
            {"prompt": "b_fails", "duration_seconds": 4},
            {"prompt": "c", "duration_seconds": 4},
        ],
        add_bridges=False,
    )
    result = json.loads(result_json)

    # Two successful beats land in the manifest, one error is recorded.
    assert result["beat_count"] == 3
    assert len(result["segments"]) == 2
    assert [s["prompt"] for s in result["segments"]] == ["a", "c"]
    assert result["total_duration_seconds"] == 8.0
    assert result["errors"] == [
        {"beat_index": 1, "error": "simulated API failure on beat 2"}
    ]


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_generate_clip_strict_first_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A beat with an unfetchable first_frame_uri fails rather than silently
    falling back to text-to-video."""
    from src.__main__ import generate_clip

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    mock_ctx = MagicMock()
    mock_ctx.info = AsyncMock()
    mock_ctx.error = AsyncMock()
    mock_ctx.request_context.lifespan_context = AppContext(
        data_folder=tmp_path,
        images_dir=images_dir,
        videos_dir=videos_dir,
        client=MagicMock(),
    )

    async def mock_impl(**kwargs: Any) -> dict[str, Any]:
        raise AssertionError("impl should not be called when first_frame fetch fails")

    monkeypatch.setattr("src.__main__.generate_video_impl", mock_impl)

    result_json = await generate_clip(
        ctx=mock_ctx,
        beats=[
            {
                "prompt": "a",
                "first_frame_uri": f"file://{tmp_path / 'nope.png'}",
            },
        ],
        add_bridges=False,
    )
    result = json.loads(result_json)
    assert result["beat_count"] == 1
    assert result["segments"] == []
    assert len(result["errors"]) == 1
    assert "first_frame_uri" in result["errors"][0]["error"]


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_generate_clip_empty_beats(tmp_path: Path) -> None:
    """Empty beats list returns an error."""
    from src.__main__ import generate_clip

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    mock_ctx = MagicMock()
    mock_ctx.info = AsyncMock()
    mock_ctx.error = AsyncMock()
    mock_ctx.request_context.lifespan_context = AppContext(
        data_folder=tmp_path,
        images_dir=images_dir,
        videos_dir=videos_dir,
        client=MagicMock(),
    )

    result_json = await generate_clip(ctx=mock_ctx, beats=[])
    assert "error" in json.loads(result_json)


# ============================================================================
# End-to-end: image -> transition (vfx-mcp workflow)
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_e2e_image_to_transition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Chain: generate two images, read their sidecars, feed URLs into
    generate_transition, verify the transition's sidecar references both.

    This is the agent-style workflow a vfx-mcp would orchestrate.
    """
    from src.__main__ import generate_image, generate_transition

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    app_ctx = AppContext(
        data_folder=tmp_path,
        images_dir=images_dir,
        videos_dir=videos_dir,
        client=MagicMock(),
    )

    def _make_ctx() -> MagicMock:
        ctx = MagicMock()
        ctx.info = AsyncMock()
        ctx.error = AsyncMock()
        ctx.request_context.lifespan_context = app_ctx
        return ctx

    # Mock image impl: write a file and return its url + preview.
    image_bytes = _create_test_image()
    call_index = {"n": 0}

    async def mock_image_impl(**kwargs: Any) -> dict[str, Any]:
        call_index["n"] += 1
        fname = images_dir / f"img{call_index['n']}.png"
        fname.write_bytes(image_bytes)
        thumb_b64 = base64.b64encode(image_bytes).decode("utf-8")
        return {
            "message": "Image generated successfully",
            "image_url": f"file://{fname}",
            "image_preview": f"data:image/jpeg;base64,{thumb_b64}",
            "prompt": kwargs["prompt"],
            "model": kwargs["model"],
        }

    monkeypatch.setattr("src.__main__.generate_image_impl", mock_image_impl)

    # Generate two frames.
    r_a = await generate_image(
        ctx=_make_ctx(),
        prompt="opening frame: sunrise",
        model="gemini-2.5-flash-image",
    )
    r_b = await generate_image(
        ctx=_make_ctx(),
        prompt="closing frame: sunset",
        model="gemini-2.5-flash-image",
    )

    # Each result is [Image, TextContent]; parse JSON from the text part.
    text_a = r_a[1].text
    text_b = r_b[1].text
    data_a = json.loads(text_a)
    data_b = json.loads(text_b)
    assert "sidecar_url" in data_a
    assert "sidecar_url" in data_b

    # Sidecar files exist and carry the original prompt.
    sidecar_a = Path(data_a["sidecar_url"][7:])
    manifest_a = json.loads(sidecar_a.read_text())
    assert manifest_a["kind"] == "image"
    assert manifest_a["prompt"] == "opening frame: sunrise"

    # Mock video impl for the transition step.
    async def mock_video_impl(**kwargs: Any) -> dict[str, Any]:
        out = videos_dir / "transition.mp4"
        out.write_bytes(b"mp4-bytes")
        return {
            "message": "Video generated successfully",
            "video_url": f"file://{out}",
            "prompt": kwargs["prompt"],
            "model": kwargs["model"],
            "audio_enabled": False,
            "generation_mode": "first_last_frame",
        }

    monkeypatch.setattr("src.__main__.generate_video_impl", mock_video_impl)

    # Feed both image urls into generate_transition.
    result_json = await generate_transition(
        ctx=_make_ctx(),
        first_frame_uri=data_a["image_url"],
        last_frame_uri=data_b["image_url"],
        prompt="dissolve from sunrise to sunset",
    )
    result = json.loads(result_json)

    assert result["generation_mode"] == "first_last_frame"
    assert result["first_frame_uri"] == data_a["image_url"]
    assert result["last_frame_uri"] == data_b["image_url"]

    # Transition sidecar cross-references the two source frames.
    sidecar_t = Path(result["sidecar_url"][7:])
    manifest_t = json.loads(sidecar_t.read_text())
    assert manifest_t["kind"] == "transition"
    assert manifest_t["first_frame_uri"] == data_a["image_url"]
    assert manifest_t["last_frame_uri"] == data_b["image_url"]


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
