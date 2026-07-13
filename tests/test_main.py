"""Tests for __main__.py MCP server."""

import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import urlparse

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

    def __init__(
        self,
        status: int,
        data: bytes,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status = status
        self._data = data
        self.content_length = len(data) if data else 0
        self.content = FakeContentStream(data)
        self.headers = headers or {}

    async def read(self) -> bytes:
        return self._data


class FakeClientSession:
    """Test double for aiohttp ClientSession.

    Responses may be (status, data) or (status, data, headers) tuples so
    tests can drive the manual-redirect path via a Location header.
    """

    def __init__(self, responses: dict[str, tuple[Any, ...]]) -> None:
        self._responses = responses

    async def __aenter__(self) -> "FakeClientSession":
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass

    def get(self, url: str, **kwargs: Any) -> "FakeContextManager":
        entry = self._responses.get(url, (404, b"Not found"))
        status, data = entry[0], entry[1]
        headers = entry[2] if len(entry) > 2 else {}
        return FakeContextManager(FakeResponse(status, data, headers))


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
        pytest.param(
            {"GOOGLE_GENAI_USE_VERTEXAI": "true"}, True, id="vertexai_enabled"
        ),
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
        monkeypatch.setattr("src.__main__._assert_http_host_public", lambda url: None)
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
    result = await fetch("http://metadata.google.internal/", allowed_dir=tmp_path)
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
    result = await fetch("https://example.com/big", allowed_dir=tmp_path, max_bytes=100)
    assert result is None


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_fetch_enforces_size_cap_local(tmp_path: Path) -> None:
    """fetch() must cap local file size."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    target = data_dir / "big.bin"
    target.write_bytes(b"x" * 1024)
    result = await fetch(f"file://{target}", allowed_dir=data_dir, max_bytes=100)
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
                "model": "imagen-4.0-generate-001",
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

    frames = np.stack(
        [
            np.full((64, 64, 3), (200, 30, 30), dtype=np.uint8),
            np.full((64, 64, 3), (30, 30, 200), dtype=np.uint8),
        ]
    )
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
        captured["last_frame_bytes_len"] = len(kwargs.get("last_frame_bytes") or b"")
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


# ============================================================================
# Security: SSRF via HTTP redirect
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_fetch_rejects_redirect_to_private_ip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A public URL that 302-redirects to a private/metadata host must be
    rejected — the SSRF guard has to re-run on the redirect target."""
    responses = {
        "http://public.example.com/x": (
            302,
            b"",
            {"Location": "http://169.254.169.254/latest/meta-data/"},
        ),
        # Would leak metadata if the redirect were followed blindly.
        "http://169.254.169.254/latest/meta-data/": (200, b"SECRET"),
    }
    monkeypatch.setattr(
        "aiohttp.ClientSession", lambda *a, **kw: FakeClientSession(responses)
    )

    def fake_guard(url: str) -> None:
        if urlparse(url).hostname == "169.254.169.254":
            raise ValueError("Refusing to fetch non-public address")

    monkeypatch.setattr("src.__main__._assert_http_host_public", fake_guard)

    result = await fetch("http://public.example.com/x", allowed_dir=tmp_path)
    assert result is None


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_fetch_follows_redirect_to_public(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A redirect to another public URL is followed and its body returned."""
    responses = {
        "http://a.example.com/x": (
            302,
            b"",
            {"Location": "http://b.example.com/y"},
        ),
        "http://b.example.com/y": (200, b"final-body"),
    }
    monkeypatch.setattr(
        "aiohttp.ClientSession", lambda *a, **kw: FakeClientSession(responses)
    )
    monkeypatch.setattr("src.__main__._assert_http_host_public", lambda url: None)

    result = await fetch("http://a.example.com/x", allowed_dir=tmp_path)
    assert result == b"final-body"


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_fetch_redirect_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A redirect chain longer than the hop limit is rejected (returns None)."""

    class LoopSession(FakeClientSession):
        def get(self, url: str, **kwargs: Any) -> "FakeContextManager":
            # Always redirect back to itself -> hop limit trips.
            return FakeContextManager(FakeResponse(302, b"", {"Location": url}))

    monkeypatch.setattr("aiohttp.ClientSession", lambda *a, **kw: LoopSession({}))
    monkeypatch.setattr("src.__main__._assert_http_host_public", lambda url: None)

    result = await fetch("http://loop.example.com/x", allowed_dir=tmp_path)
    assert result is None


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_fetch_redirect_missing_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A redirect status with no Location header is rejected."""
    responses = {"http://a.example.com/x": (302, b"")}
    monkeypatch.setattr(
        "aiohttp.ClientSession", lambda *a, **kw: FakeClientSession(responses)
    )
    monkeypatch.setattr("src.__main__._assert_http_host_public", lambda url: None)

    result = await fetch("http://a.example.com/x", allowed_dir=tmp_path)
    assert result is None


# ============================================================================
# base64 size cap
# ============================================================================


def test_decode_base64_capped_ok() -> None:
    """Small payloads decode normally."""
    from src.__main__ import _decode_base64_capped

    raw = b"hello world"
    encoded = base64.b64encode(raw).decode()
    assert _decode_base64_capped(encoded, max_bytes=1024) == raw


def test_decode_base64_capped_rejects_oversize() -> None:
    """Oversize payloads raise, and are rejected BEFORE the buffer is decoded."""
    import src.__main__ as main_mod
    from src.__main__ import _decode_base64_capped

    encoded = base64.b64encode(b"x" * 2048).decode()

    # The guard must reject based on encoded length before allocating the
    # decoded buffer — otherwise the memory-exhaustion protection is moot.
    called = False
    orig_decode = base64.b64decode

    def _tracking_decode(*args: Any, **kwargs: Any) -> bytes:
        nonlocal called
        called = True
        return orig_decode(*args, **kwargs)

    main_mod.base64.b64decode = _tracking_decode  # type: ignore[attr-defined]
    try:
        with pytest.raises(ValueError, match="exceeds"):
            _decode_base64_capped(encoded, max_bytes=1024)
    finally:
        main_mod.base64.b64decode = orig_decode  # type: ignore[attr-defined]

    assert not called, "oversize input must be rejected before decoding"


def test_decode_base64_capped_accepts_within_cap() -> None:
    """A payload within the cap decodes normally."""
    from src.__main__ import _decode_base64_capped

    encoded = base64.b64encode(b"x" * 512).decode()
    assert _decode_base64_capped(encoded, max_bytes=1024) == b"x" * 512


# ============================================================================
# Path validation ordering (no existence oracle outside allowed_dir)
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_fetch_bare_path_outside_allowed_dir_rejected(
    tmp_path: Path,
) -> None:
    """A bare path outside allowed_dir is rejected regardless of whether it
    exists — validation runs before any stat()/exists() probe."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Existing file outside the allowed dir.
    outside_existing = tmp_path / "exists.txt"
    outside_existing.write_bytes(b"secret")
    assert await fetch(str(outside_existing), allowed_dir=data_dir) is None

    # Non-existent file outside the allowed dir.
    outside_missing = tmp_path / "missing.txt"
    assert await fetch(str(outside_missing), allowed_dir=data_dir) is None


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_fetch_bare_path_inside_allowed_dir(tmp_path: Path) -> None:
    """A bare path inside allowed_dir still works after the reorder."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    target = data_dir / "ok.txt"
    target.write_bytes(b"data")
    assert await fetch(str(target), allowed_dir=data_dir) == b"data"


# ============================================================================
# Tool call sites: fail loud + param plumbing + extend validation
# ============================================================================


def _image_ctx(tmp_path: Path) -> MagicMock:
    images_dir = tmp_path / "images"
    images_dir.mkdir(exist_ok=True)
    ctx = MagicMock()
    ctx.info = AsyncMock()
    ctx.error = AsyncMock()
    ctx.request_context.lifespan_context = AppContext(
        data_folder=tmp_path,
        images_dir=images_dir,
        videos_dir=tmp_path / "videos",
        client=MagicMock(),
    )
    return ctx


def _video_ctx(
    tmp_path: Path, allowed_gcs_buckets: frozenset[str] = frozenset()
) -> MagicMock:
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir(exist_ok=True)
    ctx = MagicMock()
    ctx.info = AsyncMock()
    ctx.error = AsyncMock()
    ctx.request_context.lifespan_context = AppContext(
        data_folder=tmp_path,
        images_dir=tmp_path / "images",
        videos_dir=videos_dir,
        client=MagicMock(),
        allowed_gcs_buckets=allowed_gcs_buckets,
    )
    return ctx


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_image_raises_on_unfetchable_uri(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A provided image_uri that can't be fetched yields an error, not a
    silent downgrade to text-to-image."""
    from src.__main__ import generate_image

    async def should_not_run(**kwargs: Any) -> dict[str, Any]:
        raise AssertionError("impl must not run when image_uri fetch fails")

    monkeypatch.setattr("src.__main__.generate_image_impl", should_not_run)

    result = await generate_image(
        ctx=_image_ctx(tmp_path),
        prompt="edit",
        model="gemini-3.1-flash-image",
        image_uri=f"file://{tmp_path / 'nope.png'}",
    )
    assert len(result) == 1
    payload = json.loads(result[0].text)
    assert "error" in payload
    assert "nope.png" in payload["error"]


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_image_raises_on_unfetchable_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A provided reference image that can't be fetched is not silently
    dropped — it errors."""
    from src.__main__ import generate_image

    async def should_not_run(**kwargs: Any) -> dict[str, Any]:
        raise AssertionError("impl must not run when a reference fetch fails")

    monkeypatch.setattr("src.__main__.generate_image_impl", should_not_run)

    result = await generate_image(
        ctx=_image_ctx(tmp_path),
        prompt="edit",
        model="gemini-3.1-flash-image",
        reference_image_uris=[f"file://{tmp_path / 'missing_ref.png'}"],
    )
    payload = json.loads(result[0].text)
    assert "error" in payload
    assert "missing_ref.png" in payload["error"]


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_image_passes_new_params(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """aspect_ratio and person_generation reach the impl and the manifest."""
    from src.__main__ import generate_image

    captured: dict[str, Any] = {}
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    async def mock_impl(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        out = images_dir / "out.png"
        out.write_bytes(_create_test_image())
        thumb = base64.b64encode(_create_test_image()).decode()
        return {
            "message": "ok",
            "image_url": f"file://{out}",
            "image_preview": f"data:image/jpeg;base64,{thumb}",
            "prompt": kwargs["prompt"],
            "model": kwargs["model"],
        }

    monkeypatch.setattr("src.__main__.generate_image_impl", mock_impl)

    result = await generate_image(
        ctx=_image_ctx(tmp_path),
        prompt="a cat",
        model="gemini-3.1-flash-image",
        aspect_ratio="16:9",
        person_generation="allow_adult",
    )
    assert captured["aspect_ratio"] == "16:9"
    assert captured["person_generation"] == "allow_adult"

    # Manifest carries the new fields too.
    text = result[1].text
    data = json.loads(text)
    sidecar = Path(data["sidecar_url"][7:])
    manifest = json.loads(sidecar.read_text())
    assert manifest["aspect_ratio"] == "16:9"
    assert manifest["person_generation"] == "allow_adult"


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_image_rejects_oversize_base64(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An oversize inline base64 image is rejected before generation."""
    import src.__main__ as main_mod
    from src.__main__ import generate_image

    monkeypatch.setattr(main_mod, "MAX_FETCH_BYTES", 16)

    async def should_not_run(**kwargs: Any) -> dict[str, Any]:
        raise AssertionError("impl must not run on oversize base64")

    monkeypatch.setattr("src.__main__.generate_image_impl", should_not_run)

    big_b64 = base64.b64encode(b"x" * 1024).decode()
    result = await generate_image(
        ctx=_image_ctx(tmp_path),
        prompt="edit",
        model="gemini-3.1-flash-image",
        image_base64=big_b64,
    )
    payload = json.loads(result[0].text)
    assert "error" in payload


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_video_raises_on_unfetchable_image_uri(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A provided image_uri that can't be fetched errors instead of silently
    becoming text-to-video."""
    from src.__main__ import generate_video

    async def should_not_run(**kwargs: Any) -> dict[str, Any]:
        raise AssertionError("impl must not run when image_uri fetch fails")

    monkeypatch.setattr("src.__main__.generate_video_impl", should_not_run)

    result = await generate_video(
        ctx=_video_ctx(tmp_path),
        prompt="p",
        model="veo-3.1-generate-001",
        image_uri=f"file://{tmp_path / 'nope.png'}",
    )
    payload = json.loads(result)
    assert "error" in payload
    assert "nope.png" in payload["error"]


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_video_raises_on_unfetchable_last_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A provided last_frame_uri that can't be fetched errors."""
    from src.__main__ import generate_video

    async def should_not_run(**kwargs: Any) -> dict[str, Any]:
        raise AssertionError("impl must not run when last_frame fetch fails")

    monkeypatch.setattr("src.__main__.generate_video_impl", should_not_run)

    result = await generate_video(
        ctx=_video_ctx(tmp_path),
        prompt="p",
        model="veo-3.1-generate-001",
        last_frame_uri=f"file://{tmp_path / 'nope.png'}",
    )
    payload = json.loads(result)
    assert "error" in payload
    assert "nope.png" in payload["error"]


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_video_passes_new_params(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """resolution and person_generation reach the impl and the manifest."""
    from src.__main__ import generate_video

    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    captured: dict[str, Any] = {}

    async def mock_impl(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        out = videos_dir / "v.mp4"
        out.write_bytes(b"mp4")
        return {
            "message": "Video generated successfully",
            "video_url": f"file://{out}",
            "prompt": kwargs["prompt"],
            "model": kwargs["model"],
            "audio_enabled": False,
            "duration_seconds": 8.0,
        }

    monkeypatch.setattr("src.__main__.generate_video_impl", mock_impl)

    result = await generate_video(
        ctx=_video_ctx(tmp_path),
        prompt="p",
        model="veo-3.1-generate-001",
        resolution="1080p",
        person_generation="allow_all",
    )
    assert captured["resolution"] == "1080p"
    assert captured["person_generation"] == "allow_all"

    data = json.loads(result)
    sidecar = Path(data["sidecar_url"][7:])
    manifest = json.loads(sidecar.read_text())
    assert manifest["resolution"] == "1080p"
    assert manifest["person_generation"] == "allow_all"


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_video_extend_rejects_disallowed_bucket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """extend_video_uri pointing at a bucket outside the allowlist errors."""
    from src.__main__ import generate_video

    async def should_not_run(**kwargs: Any) -> dict[str, Any]:
        raise AssertionError("impl must not run for a disallowed extend bucket")

    monkeypatch.setattr("src.__main__.generate_video_impl", should_not_run)

    result = await generate_video(
        ctx=_video_ctx(tmp_path, allowed_gcs_buckets=frozenset({"good"})),
        prompt="p",
        model="veo-3.1-generate-001",
        extend_video_uri="gs://evil/clip.mp4",
        output_gcs_uri="gs://good/out/",
    )
    payload = json.loads(result)
    assert "error" in payload
    assert "allowlist" in payload["error"]


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_generate_video_extend_requires_gcs_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Video extension with no resolvable GCS output target errors before
    starting generation."""
    from src.__main__ import generate_video

    async def should_not_run(**kwargs: Any) -> dict[str, Any]:
        raise AssertionError("impl must not run without a GCS output target")

    monkeypatch.setattr("src.__main__.generate_video_impl", should_not_run)

    # allowlist empty and no output_gcs_uri / VIDEO_GCS_BUCKET.
    result = await generate_video(
        ctx=_video_ctx(tmp_path),
        prompt="p",
        model="veo-3.1-generate-001",
        extend_video_uri="gs://anything/clip.mp4",
    )
    payload = json.loads(result)
    assert "error" in payload
    assert "output_gcs_uri" in payload["error"]


# ============================================================================
# Regression tests for the post-review fixes
# ============================================================================


def _gemini_api_app_ctx(tmp_path: Path, video_gcs_bucket: str | None = None) -> Any:
    """AppContext whose client reports the Gemini API (non-Vertex) backend.

    A bare MagicMock has a truthy `_api_client.vertexai`, so it must be pinned
    to False to exercise the Gemini-API code paths.
    """
    client = MagicMock()
    client._api_client.vertexai = False
    return AppContext(
        data_folder=tmp_path,
        images_dir=tmp_path / "images",
        videos_dir=tmp_path / "videos",
        client=client,
        video_gcs_bucket=video_gcs_bucket,
    )


def _ctx_wrapping(app_ctx: Any) -> Any:
    """Wrap a pre-built AppContext in a mock MCP context."""
    mock_ctx = MagicMock()
    mock_ctx.info = AsyncMock()
    mock_ctx.error = AsyncMock()
    mock_ctx.request_context.lifespan_context = app_ctx
    return mock_ctx


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_generate_video_explicit_gcs_on_gemini_api_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit output_gcs_uri on a Gemini-API-routed call is a clear error."""
    from src.__main__ import generate_video

    (tmp_path / "images").mkdir()
    (tmp_path / "videos").mkdir()
    ctx = _ctx_wrapping(_gemini_api_app_ctx(tmp_path))

    async def should_not_run(**kwargs: Any) -> dict[str, Any]:
        raise AssertionError("impl must not run when explicit GCS is rejected")

    monkeypatch.setattr("src.__main__.generate_video_impl", should_not_run)

    result = json.loads(
        await generate_video(
            ctx=ctx,
            prompt="x",
            model="veo-3.1-generate-001",
            output_gcs_uri="gs://bucket/out.mp4",
        )
    )
    assert "error" in result
    assert "Vertex AI" in result["error"]


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_generate_video_env_gcs_default_dropped_on_gemini_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A VIDEO_GCS_BUCKET default must not break Gemini-API generation.

    Regression: the env default was funneled to the impl and the Vertex-only
    gate then hard-raised, breaking all Veo Lite / text-to-video runs.
    """
    from src.__main__ import generate_video

    (tmp_path / "images").mkdir()
    (tmp_path / "videos").mkdir()
    ctx = _ctx_wrapping(
        _gemini_api_app_ctx(tmp_path, video_gcs_bucket="gs://default-bucket/out/")
    )

    captured: dict[str, Any] = {}

    async def mock_impl(**kwargs: Any) -> dict[str, Any]:
        captured["output_gcs_uri"] = kwargs.get("output_gcs_uri")
        return {
            "message": "Video generated successfully",
            "video_url": f"file://{tmp_path}/videos/out.mp4",
            "model": kwargs.get("model", ""),
        }

    monkeypatch.setattr("src.__main__.generate_video_impl", mock_impl)

    result = json.loads(
        await generate_video(ctx=ctx, prompt="x", model="veo-3.1-generate-001")
    )
    assert result["message"] == "Video generated successfully"
    # The env-default bucket is dropped on the Gemini API path.
    assert captured["output_gcs_uri"] is None


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_generate_video_extend_on_gemini_api_without_gcs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Video extension works on the Gemini API without a GCS target.

    Regression: two independently-added checks (extend-requires-GCS and
    GCS-requires-Vertex) deadlocked extension on Gemini-API deployments.
    """
    from src.__main__ import generate_video

    (tmp_path / "images").mkdir()
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    src_clip = videos_dir / "src.mp4"
    src_clip.write_bytes(b"fake-mp4")
    ctx = _ctx_wrapping(_gemini_api_app_ctx(tmp_path))

    captured: dict[str, Any] = {}

    async def mock_impl(**kwargs: Any) -> dict[str, Any]:
        captured["extend_video_uri"] = kwargs.get("extend_video_uri")
        return {
            "message": "Video generated successfully",
            "video_url": f"file://{videos_dir}/out.mp4",
            "model": kwargs.get("model", ""),
            "generation_mode": "extend_video",
        }

    monkeypatch.setattr("src.__main__.generate_video_impl", mock_impl)

    result = json.loads(
        await generate_video(
            ctx=ctx,
            prompt="continue",
            model="veo-3.1-generate-001",
            extend_video_uri=f"file://{src_clip}",
        )
    )
    assert result["message"] == "Video generated successfully"
    assert captured["extend_video_uri"] == f"file://{src_clip}"


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_generate_image_text_only_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A text-only model response is surfaced, not crashed on a missing key.

    Regression: the tool indexed result["image_url"] unconditionally, so a
    text-only response (refusal / clarifying question) raised KeyError.
    """
    from src.__main__ import generate_image

    (tmp_path / "images").mkdir()
    (tmp_path / "videos").mkdir()
    ctx = _ctx_wrapping(
        AppContext(
            data_folder=tmp_path,
            images_dir=tmp_path / "images",
            videos_dir=tmp_path / "videos",
            client=MagicMock(),
        )
    )

    async def mock_impl(**kwargs: Any) -> dict[str, Any]:
        return {
            "message": "Model returned text only",
            "generated_text": "I can't create that image.",
            "model": kwargs.get("model", ""),
        }

    monkeypatch.setattr("src.__main__.generate_image_impl", mock_impl)

    result = await generate_image(
        ctx=ctx, prompt="something", model="gemini-2.5-flash-image"
    )
    # Single TextContent with the model's text — no KeyError, no image part.
    assert len(result) == 1
    payload = json.loads(result[0].text)
    assert payload["generated_text"] == "I can't create that image."
    assert "error" not in payload


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_generate_clip_invalid_aspect_ratio_top_level_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An invalid clip aspect ratio fails top-level, not as a per-beat error.

    Regression: the impl's per-value ValueError fired inside each beat handler,
    producing a success-shaped clip manifest with zero segments.
    """
    from src.__main__ import generate_clip

    (tmp_path / "images").mkdir()
    (tmp_path / "videos").mkdir()
    ctx = _ctx_wrapping(
        AppContext(
            data_folder=tmp_path,
            images_dir=tmp_path / "images",
            videos_dir=tmp_path / "videos",
            client=MagicMock(),
        )
    )

    async def should_not_run(**kwargs: Any) -> dict[str, Any]:
        raise AssertionError("impl must not run for an invalid clip aspect ratio")

    monkeypatch.setattr("src.__main__.generate_video_impl", should_not_run)

    result = json.loads(
        await generate_clip(
            ctx=ctx,
            beats=[{"prompt": "a"}, {"prompt": "b"}],
            aspect_ratio="1:1",
        )
    )
    assert "error" in result
    assert "aspect_ratio" in result["error"]
    assert result.get("kind") != "clip"


@pytest.mark.asyncio
@pytest.mark.timeout(2.0)
async def test_fetch_unsupported_scheme_message(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An unsupported URI scheme reports the scheme, not a directory error.

    Regression: ftp://, s3://, data: fell into local-path validation and were
    misreported as 'outside the allowed directory'.
    """
    import logging

    with caplog.at_level(logging.ERROR):
        result = await fetch("ftp://host/pic.png", allowed_dir=tmp_path)
    assert result is None
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "Unsupported URI scheme" in joined
    assert "outside the allowed directory" not in joined


# ============================================================================
# Round: GCS gating across transition/bridge/clip, clip client routing,
# base64 wrap/pad cap, up-front aspect validation, warnings propagation.
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_generate_transition_explicit_gcs_on_gemini_api_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit output_gcs_uri on a Gemini-API-routed transition is a clear
    top-level error, not a silent drop by the impl."""
    from src.__main__ import generate_transition

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (tmp_path / "videos").mkdir()
    frame_a = images_dir / "a.png"
    frame_b = images_dir / "b.png"
    frame_a.write_bytes(_create_test_image())
    frame_b.write_bytes(_create_test_image())
    ctx = _ctx_wrapping(_gemini_api_app_ctx(tmp_path))

    async def should_not_run(**kwargs: Any) -> dict[str, Any]:
        raise AssertionError("impl must not run when explicit GCS is rejected")

    monkeypatch.setattr("src.__main__.generate_video_impl", should_not_run)

    result = json.loads(
        await generate_transition(
            ctx=ctx,
            first_frame_uri=f"file://{frame_a}",
            last_frame_uri=f"file://{frame_b}",
            output_gcs_uri="gs://bucket/out.mp4",
        )
    )
    assert "error" in result
    assert "Vertex AI" in result["error"]


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_generate_transition_env_gcs_default_dropped_on_gemini_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A VIDEO_GCS_BUCKET default is silently dropped (not raised) for a
    Gemini-API transition, and the call succeeds inline."""
    from src.__main__ import generate_transition

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    frame_a = images_dir / "a.png"
    frame_b = images_dir / "b.png"
    frame_a.write_bytes(_create_test_image())
    frame_b.write_bytes(_create_test_image())
    ctx = _ctx_wrapping(
        _gemini_api_app_ctx(tmp_path, video_gcs_bucket="gs://default-bucket/out/")
    )

    captured: dict[str, Any] = {}

    async def mock_impl(**kwargs: Any) -> dict[str, Any]:
        captured["output_gcs_uri"] = kwargs.get("output_gcs_uri")
        out = videos_dir / "out.mp4"
        out.write_bytes(b"mp4")
        return {
            "message": "Video generated successfully",
            "video_url": f"file://{out}",
            "model": kwargs.get("model", ""),
            "audio_enabled": False,
            "generation_mode": "first_last_frame",
        }

    monkeypatch.setattr("src.__main__.generate_video_impl", mock_impl)

    result = json.loads(
        await generate_transition(
            ctx=ctx,
            first_frame_uri=f"file://{frame_a}",
            last_frame_uri=f"file://{frame_b}",
        )
    )
    assert result["message"] == "Video generated successfully"
    assert captured["output_gcs_uri"] is None


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
async def test_generate_bridge_explicit_gcs_on_gemini_api_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit output_gcs_uri on a Gemini-API-routed bridge errors."""
    from src.__main__ import generate_bridge

    (tmp_path / "images").mkdir()
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    clip_a = videos_dir / "a.mp4"
    clip_b = videos_dir / "b.mp4"
    clip_a.write_bytes(_make_fake_mp4())
    clip_b.write_bytes(_make_fake_mp4())
    ctx = _ctx_wrapping(_gemini_api_app_ctx(tmp_path))

    async def should_not_run(**kwargs: Any) -> dict[str, Any]:
        raise AssertionError("impl must not run when explicit GCS is rejected")

    monkeypatch.setattr("src.__main__.generate_video_impl", should_not_run)

    result = json.loads(
        await generate_bridge(
            ctx=ctx,
            from_clip_uri=f"file://{clip_a}",
            to_clip_uri=f"file://{clip_b}",
            output_gcs_uri="gs://bucket/out.mp4",
        )
    )
    assert "error" in result
    assert "Vertex AI" in result["error"]


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
async def test_generate_bridge_env_gcs_default_dropped_on_gemini_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A VIDEO_GCS_BUCKET default is dropped (not raised) for a Gemini-API
    bridge, and the call succeeds inline."""
    from src.__main__ import generate_bridge

    (tmp_path / "images").mkdir()
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    clip_a = videos_dir / "a.mp4"
    clip_b = videos_dir / "b.mp4"
    clip_a.write_bytes(_make_fake_mp4())
    clip_b.write_bytes(_make_fake_mp4())
    ctx = _ctx_wrapping(
        _gemini_api_app_ctx(tmp_path, video_gcs_bucket="gs://default-bucket/out/")
    )

    captured: dict[str, Any] = {}

    async def mock_impl(**kwargs: Any) -> dict[str, Any]:
        captured["output_gcs_uri"] = kwargs.get("output_gcs_uri")
        out = videos_dir / "bridge.mp4"
        out.write_bytes(b"mp4")
        return {
            "message": "Video generated successfully",
            "video_url": f"file://{out}",
            "model": kwargs.get("model", ""),
            "audio_enabled": False,
            "generation_mode": "first_last_frame",
        }

    monkeypatch.setattr("src.__main__.generate_video_impl", mock_impl)

    result = json.loads(
        await generate_bridge(
            ctx=ctx,
            from_clip_uri=f"file://{clip_a}",
            to_clip_uri=f"file://{clip_b}",
        )
    )
    assert result["message"] == "Video generated successfully"
    assert captured["output_gcs_uri"] is None


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_generate_clip_explicit_gcs_on_gemini_api_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit output_gcs_uri on a Gemini-API-routed clip errors top-level
    before any beat runs."""
    from src.__main__ import generate_clip

    (tmp_path / "images").mkdir()
    (tmp_path / "videos").mkdir()
    ctx = _ctx_wrapping(_gemini_api_app_ctx(tmp_path))

    async def should_not_run(**kwargs: Any) -> dict[str, Any]:
        raise AssertionError("impl must not run when explicit GCS is rejected")

    monkeypatch.setattr("src.__main__.generate_video_impl", should_not_run)

    result = json.loads(
        await generate_clip(
            ctx=ctx,
            beats=[{"prompt": "a"}, {"prompt": "b"}],
            output_gcs_uri="gs://bucket/out.mp4",
        )
    )
    assert "error" in result
    assert "Vertex AI" in result["error"]
    assert result.get("kind") != "clip"


@pytest.mark.asyncio
@pytest.mark.timeout(10.0)
async def test_generate_clip_env_gcs_default_dropped_on_gemini_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A VIDEO_GCS_BUCKET default is dropped (not raised) for a Gemini-API
    clip, and beats succeed inline."""
    from src.__main__ import generate_clip

    (tmp_path / "images").mkdir()
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    ctx = _ctx_wrapping(
        _gemini_api_app_ctx(tmp_path, video_gcs_bucket="gs://default-bucket/out/")
    )

    captured: dict[str, Any] = {}

    async def mock_impl(**kwargs: Any) -> dict[str, Any]:
        captured["output_gcs_uri"] = kwargs.get("output_gcs_uri")
        out = videos_dir / "beat.mp4"
        out.write_bytes(_make_fake_mp4())
        return {
            "message": "Video generated successfully",
            "video_url": f"file://{out}",
            "model": kwargs.get("model", ""),
            "audio_enabled": kwargs.get("include_audio", False),
            "generation_mode": "text_to_video",
            "duration_seconds": kwargs.get("duration_seconds"),
        }

    monkeypatch.setattr("src.__main__.generate_video_impl", mock_impl)

    result = json.loads(
        await generate_clip(
            ctx=ctx,
            beats=[{"prompt": "a", "duration_seconds": 4}],
            add_bridges=False,
        )
    )
    assert result["kind"] == "clip"
    assert len(result["segments"]) == 1
    assert captured["output_gcs_uri"] is None


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_generate_clip_lite_without_api_client_top_level_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Lite model that can't route (Vertex primary, no gemini_api_client)
    fails top-level, not as an empty-segments success manifest.

    Regression: the client was resolved inside the per-beat try/except, so the
    RuntimeError was recorded once per beat and the clip returned zero
    segments while looking successful.
    """
    from src.__main__ import generate_clip

    (tmp_path / "images").mkdir()
    (tmp_path / "videos").mkdir()
    vertex_client = MagicMock()
    vertex_client._api_client.vertexai = True
    app_ctx = AppContext(
        data_folder=tmp_path,
        images_dir=tmp_path / "images",
        videos_dir=tmp_path / "videos",
        client=vertex_client,
        gemini_api_client=None,
    )
    ctx = _ctx_wrapping(app_ctx)

    async def should_not_run(**kwargs: Any) -> dict[str, Any]:
        raise AssertionError("impl must not run when the model can't be routed")

    monkeypatch.setattr("src.__main__.generate_video_impl", should_not_run)

    result = json.loads(
        await generate_clip(
            ctx=ctx,
            beats=[{"prompt": "a"}, {"prompt": "b"}],
            model="veo-3.1-lite-generate-preview",
        )
    )
    assert "error" in result
    assert "Gemini API" in result["error"]
    assert result.get("kind") != "clip"
    assert "segments" not in result


def test_decode_base64_capped_accepts_wrapped_under_cap() -> None:
    """A 76-column MIME-wrapped payload whose true decoded size is under the
    cap must NOT be rejected (whitespace is stripped before the estimate)."""
    from src.__main__ import _decode_base64_capped

    raw = b"y" * 1024
    # encodebytes wraps at 76 columns and adds newlines — the pre-decode
    # estimate must ignore this whitespace.
    encoded = base64.encodebytes(raw).decode()
    assert "\n" in encoded
    assert _decode_base64_capped(encoded, max_bytes=1030) == raw


def test_decode_base64_capped_accepts_exactly_cap() -> None:
    """A payload decoding to exactly max_bytes must NOT be pre-rejected.

    Regression: the old `len(data) // 4 * 3` estimate overshot on padded
    input and falsely rejected an at-cap payload.
    """
    from src.__main__ import _decode_base64_capped

    raw = b"z" * 1024
    encoded = base64.b64encode(raw).decode()
    assert _decode_base64_capped(encoded, max_bytes=1024) == raw


def test_decode_base64_capped_rejects_oversize_before_decoding() -> None:
    """A clearly-oversize payload is rejected BEFORE base64.b64decode runs."""
    import src.__main__ as main_mod
    from src.__main__ import _decode_base64_capped

    encoded = base64.encodebytes(b"x" * (64 * 1024)).decode()

    called = False
    orig_decode = base64.b64decode

    def _tracking_decode(*args: Any, **kwargs: Any) -> bytes:
        nonlocal called
        called = True
        return orig_decode(*args, **kwargs)

    main_mod.base64.b64decode = _tracking_decode  # type: ignore[attr-defined]
    try:
        with pytest.raises(ValueError, match="exceeds"):
            _decode_base64_capped(encoded, max_bytes=1024)
    finally:
        main_mod.base64.b64decode = orig_decode  # type: ignore[attr-defined]

    assert not called, "oversize input must be rejected before decoding"


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_video_tools_reject_bad_aspect_ratio_before_impl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """generate_video / _transition / _bridge reject an unsupported aspect
    ratio up front, before the impl (and before any fetch)."""
    from src.__main__ import generate_bridge, generate_transition, generate_video

    (tmp_path / "images").mkdir()
    (tmp_path / "videos").mkdir()

    async def should_not_run(**kwargs: Any) -> dict[str, Any]:
        raise AssertionError("impl must not run for an invalid aspect ratio")

    monkeypatch.setattr("src.__main__.generate_video_impl", should_not_run)

    v = json.loads(
        await generate_video(
            ctx=_video_ctx(tmp_path),
            prompt="p",
            model="veo-3.1-generate-001",
            aspect_ratio="4:3",
        )
    )
    assert "error" in v
    assert "aspect_ratio" in v["error"]

    t = json.loads(
        await generate_transition(
            ctx=_video_ctx(tmp_path),
            first_frame_uri=f"file://{tmp_path / 'a.png'}",
            last_frame_uri=f"file://{tmp_path / 'b.png'}",
            aspect_ratio="4:3",
        )
    )
    assert "error" in t
    assert "aspect_ratio" in t["error"]

    b = json.loads(
        await generate_bridge(
            ctx=_video_ctx(tmp_path),
            from_clip_uri=f"file://{tmp_path / 'a.mp4'}",
            to_clip_uri=f"file://{tmp_path / 'b.mp4'}",
            aspect_ratio="4:3",
        )
    )
    assert "error" in b
    assert "aspect_ratio" in b["error"]


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_generate_video_propagates_warnings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Warnings from the impl surface in both the response and the manifest."""
    from src.__main__ import generate_video

    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    (tmp_path / "images").mkdir()
    warning = "include_audio=False ignored on the Gemini API path"

    async def mock_impl(**kwargs: Any) -> dict[str, Any]:
        out = videos_dir / "v.mp4"
        out.write_bytes(b"mp4")
        return {
            "message": "Video generated successfully",
            "video_url": f"file://{out}",
            "model": kwargs["model"],
            "audio_enabled": False,
            "warnings": [warning],
        }

    monkeypatch.setattr("src.__main__.generate_video_impl", mock_impl)

    result = json.loads(
        await generate_video(
            ctx=_video_ctx(tmp_path), prompt="p", model="veo-3.1-generate-001"
        )
    )
    assert result["warnings"] == [warning]
    sidecar = Path(result["sidecar_url"][7:])
    manifest = json.loads(sidecar.read_text())
    assert manifest["warnings"] == [warning]


@pytest.mark.asyncio
@pytest.mark.timeout(10.0)
async def test_generate_clip_propagates_beat_warnings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Per-beat impl warnings appear in the beat manifest and aggregate into a
    clip-level warnings list."""
    from src.__main__ import generate_clip

    (tmp_path / "images").mkdir()
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    ctx = _ctx_wrapping(
        AppContext(
            data_folder=tmp_path,
            images_dir=tmp_path / "images",
            videos_dir=videos_dir,
            client=MagicMock(),
        )
    )
    warning = "include_audio ignored on the Gemini API path"
    call_index = {"n": 0}

    async def mock_impl(**kwargs: Any) -> dict[str, Any]:
        call_index["n"] += 1
        out = videos_dir / f"beat{call_index['n']}.mp4"
        out.write_bytes(_make_fake_mp4())
        return {
            "message": "Video generated successfully",
            "video_url": f"file://{out}",
            "model": kwargs.get("model", ""),
            "audio_enabled": False,
            "generation_mode": "text_to_video",
            "duration_seconds": kwargs.get("duration_seconds"),
            "warnings": [warning],
        }

    monkeypatch.setattr("src.__main__.generate_video_impl", mock_impl)

    result = json.loads(
        await generate_clip(
            ctx=ctx,
            beats=[{"prompt": "a", "duration_seconds": 4}],
            add_bridges=False,
        )
    )
    assert result["segments"][0]["warnings"] == [warning]
    assert result["warnings"] == [warning]


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_generate_clip_dedupes_repeated_warnings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Identical warnings from many beats collapse to one at the clip level.

    Regression: the same 'audio not honored' warning was appended per beat and
    per bridge, flooding the clip manifest with byte-identical copies.
    """
    from src.__main__ import generate_clip

    (tmp_path / "images").mkdir()
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    ctx = _ctx_wrapping(
        AppContext(
            data_folder=tmp_path,
            images_dir=tmp_path / "images",
            videos_dir=videos_dir,
            client=MagicMock(),
        )
    )
    warning = "include_audio=False was not honored: ..."
    call_index = {"n": 0}

    async def mock_impl(**kwargs: Any) -> dict[str, Any]:
        call_index["n"] += 1
        out = videos_dir / f"seg{call_index['n']}.mp4"
        out.write_bytes(_make_fake_mp4())
        return {
            "message": "Video generated successfully",
            "video_url": f"file://{out}",
            "model": kwargs.get("model", ""),
            "audio_enabled": False,
            "generation_mode": "text_to_video",
            "duration_seconds": kwargs.get("duration_seconds"),
            "warnings": [warning],
        }

    monkeypatch.setattr("src.__main__.generate_video_impl", mock_impl)

    result = json.loads(
        await generate_clip(
            ctx=ctx,
            beats=[{"prompt": "a"}, {"prompt": "b"}, {"prompt": "c"}],
            add_bridges=False,
        )
    )
    # Three beats each emit the identical warning; the clip carries it once.
    assert result["warnings"] == [warning]
    # But each beat manifest still records its own warning.
    beat_segs = [s for s in result["segments"] if s.get("kind") == "beat"]
    assert len(beat_segs) == 3
    assert all(s["warnings"] == [warning] for s in beat_segs)


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_generate_image_malformed_thought_signature_url_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-file:// thought_signature_url errors instead of being silently
    dropped (which would turn an edit into an unrelated fresh generation)."""
    from src.__main__ import generate_image

    (tmp_path / "images").mkdir()
    (tmp_path / "videos").mkdir()
    ctx = _ctx_wrapping(
        AppContext(
            data_folder=tmp_path,
            images_dir=tmp_path / "images",
            videos_dir=tmp_path / "videos",
            client=MagicMock(),
        )
    )

    async def should_not_run(**kwargs: Any) -> dict[str, Any]:
        raise AssertionError("impl must not run with a malformed signature URL")

    monkeypatch.setattr("src.__main__.generate_image_impl", should_not_run)

    result = await generate_image(
        ctx=ctx,
        prompt="make it orange",
        model="gemini-3-pro-image",
        thought_signature_url="/data/images/x_thought.txt",
    )
    payload = json.loads(result[0].text)
    assert "error" in payload
    assert "file://" in payload["error"]


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_generate_video_bare_env_bucket_dropped_on_gemini_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed (bare, non-gs://) VIDEO_GCS_BUCKET env default must not
    fail calls on the Gemini API path, where it would be dropped anyway."""
    from src.__main__ import generate_video

    (tmp_path / "images").mkdir()
    (tmp_path / "videos").mkdir()
    ctx = _ctx_wrapping(_gemini_api_app_ctx(tmp_path, video_gcs_bucket="my-bucket"))

    captured: dict[str, Any] = {}

    async def mock_impl(**kwargs: Any) -> dict[str, Any]:
        captured["output_gcs_uri"] = kwargs.get("output_gcs_uri")
        return {
            "message": "Video generated successfully",
            "video_url": f"file://{tmp_path}/videos/out.mp4",
            "model": kwargs.get("model", ""),
        }

    monkeypatch.setattr("src.__main__.generate_video_impl", mock_impl)

    result = json.loads(
        await generate_video(ctx=ctx, prompt="x", model="veo-3.1-generate-001")
    )
    assert result["message"] == "Video generated successfully"
    assert captured["output_gcs_uri"] is None


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_generate_video_gcs_output_includes_inline_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When output lands in GCS (no local sidecar possible), the manifest is
    included inline in the response so generation params aren't lost."""
    from src.__main__ import generate_video

    (tmp_path / "images").mkdir()
    (tmp_path / "videos").mkdir()
    client = MagicMock()
    client._api_client.vertexai = True
    ctx = _ctx_wrapping(
        AppContext(
            data_folder=tmp_path,
            images_dir=tmp_path / "images",
            videos_dir=tmp_path / "videos",
            client=client,
        )
    )

    async def mock_impl(**kwargs: Any) -> dict[str, Any]:
        return {
            "message": "Video generated successfully",
            "video_url": "gs://bucket/out.mp4",
            "model": kwargs.get("model", ""),
            "duration_seconds": 8,
        }

    monkeypatch.setattr("src.__main__.generate_video_impl", mock_impl)

    result = json.loads(
        await generate_video(
            ctx=ctx,
            prompt="x",
            model="veo-3.1-generate-001",
            seed=42,
            resolution="1080p",
            output_gcs_uri="gs://bucket/out/",
        )
    )
    assert "sidecar_url" not in result
    manifest = result["manifest"]
    assert manifest["seed"] == 42
    assert manifest["resolution"] == "1080p"


# ============================================================================
# Omni / draft / animatic / loop_extend tools
# ============================================================================


def _omni_result(video_url: str, interaction_id: str = "int-1") -> dict[str, Any]:
    return {
        "message": "Video generated successfully",
        "video_url": video_url,
        "interaction_id": interaction_id,
        "model": "gemini-omni-flash-preview",
        "duration_seconds": 6,
        "aspect_ratio": "16:9",
    }


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_generate_video_omni_returns_interaction_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.__main__ import generate_video_omni

    (tmp_path / "images").mkdir()
    (tmp_path / "videos").mkdir()
    ctx = _ctx_wrapping(_gemini_api_app_ctx(tmp_path))
    out = tmp_path / "videos" / "o.mp4"
    out.write_bytes(b"mp4")

    captured: dict[str, Any] = {}

    async def mock_impl(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return _omni_result(f"file://{out}")

    monkeypatch.setattr("src.__main__.generate_video_omni_impl", mock_impl)

    result = json.loads(await generate_video_omni(ctx=ctx, prompt="a marble rolling"))
    assert result["interaction_id"] == "int-1"
    assert result["video_url"] == f"file://{out}"
    assert result["sidecar_url"].endswith(".json")
    assert captured["prompt"] == "a marble rolling"


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_edit_video_forwards_previous_interaction_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.__main__ import edit_video

    (tmp_path / "images").mkdir()
    (tmp_path / "videos").mkdir()
    ctx = _ctx_wrapping(_gemini_api_app_ctx(tmp_path))
    out = tmp_path / "videos" / "e.mp4"
    out.write_bytes(b"mp4")

    captured: dict[str, Any] = {}

    async def mock_impl(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return _omni_result(f"file://{out}", interaction_id="int-2")

    monkeypatch.setattr("src.__main__.generate_video_omni_impl", mock_impl)

    result = json.loads(
        await edit_video(
            ctx=ctx,
            previous_interaction_id="int-1",
            prompt="make the sky stormy",
        )
    )
    assert captured["previous_interaction_id"] == "int-1"
    assert result["interaction_id"] == "int-2"


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_generate_video_draft_routes_to_omni(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.__main__ import generate_video

    (tmp_path / "images").mkdir()
    (tmp_path / "videos").mkdir()
    ctx = _ctx_wrapping(_gemini_api_app_ctx(tmp_path))
    out = tmp_path / "videos" / "d.mp4"
    out.write_bytes(b"mp4")

    omni_called = {"n": 0}

    async def mock_omni(**kwargs: Any) -> dict[str, Any]:
        omni_called["n"] += 1
        return _omni_result(f"file://{out}")

    async def veo_should_not_run(**kwargs: Any) -> dict[str, Any]:
        raise AssertionError("draft mode must not call the Veo impl")

    monkeypatch.setattr("src.__main__.generate_video_omni_impl", mock_omni)
    monkeypatch.setattr("src.__main__.generate_video_impl", veo_should_not_run)

    result = json.loads(
        await generate_video(
            ctx=ctx,
            prompt="draft this",
            model="veo-3.1-generate-001",
            draft=True,
            seed=42,
            negative_prompt="blurry",
        )
    )
    assert omni_called["n"] == 1
    assert result["interaction_id"] == "int-1"
    # Veo-only params that were passed are reported as ignored.
    joined = " ".join(result.get("warnings", []))
    assert "seed" in joined and "negative_prompt" in joined


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_generate_clip_animatic_uses_omni_and_skips_bridges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.__main__ import generate_clip

    (tmp_path / "images").mkdir()
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    ctx = _ctx_wrapping(_gemini_api_app_ctx(tmp_path))

    n = {"i": 0}

    async def mock_omni(**kwargs: Any) -> dict[str, Any]:
        n["i"] += 1
        out = videos_dir / f"a{n['i']}.mp4"
        out.write_bytes(b"mp4")
        return _omni_result(f"file://{out}", interaction_id=f"int-{n['i']}")

    async def veo_should_not_run(**kwargs: Any) -> dict[str, Any]:
        raise AssertionError("animatic mode must not call the Veo impl")

    monkeypatch.setattr("src.__main__.generate_video_omni_impl", mock_omni)
    monkeypatch.setattr("src.__main__.generate_video_impl", veo_should_not_run)

    result = json.loads(
        await generate_clip(
            ctx=ctx,
            beats=[{"prompt": "a"}, {"prompt": "b"}],
            animatic=True,
            add_bridges=True,
        )
    )
    assert result["animatic"] is True
    assert result["model"] == "gemini-omni-flash-preview"
    beat_segs = [s for s in result["segments"] if s.get("kind") == "beat"]
    assert len(beat_segs) == 2
    assert all(s["generation_mode"] == "animatic" for s in beat_segs)
    # No bridge segments; add_bridges was ignored with a warning.
    assert not any(s.get("kind") == "bridge" for s in result["segments"])
    assert any("add_bridges is ignored" in w for w in result.get("warnings", []))


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_loop_extend_chains_extensions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.__main__ import loop_extend

    (tmp_path / "images").mkdir()
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    ctx = _ctx_wrapping(_gemini_api_app_ctx(tmp_path))

    extend_uris: list[str] = []
    n = {"i": 0}

    async def mock_impl(**kwargs: Any) -> dict[str, Any]:
        n["i"] += 1
        extend_uris.append(kwargs.get("extend_video_uri"))
        out = videos_dir / f"ext{n['i']}.mp4"
        out.write_bytes(b"mp4")
        return {
            "message": "Video generated successfully",
            "video_url": f"file://{out}",
            "model": kwargs.get("model"),
            "generation_mode": "extend_video",
        }

    monkeypatch.setattr("src.__main__.generate_video_impl", mock_impl)

    start = videos_dir / "start.mp4"
    start.write_bytes(b"mp4")
    result = json.loads(
        await loop_extend(
            ctx=ctx,
            video_uri=f"file://{start}",
            times=3,
            model="veo-3.1-generate-001",
        )
    )
    assert n["i"] == 3
    # First extension uses the source; each subsequent uses the prior output.
    assert extend_uris[0] == f"file://{start}"
    assert extend_uris[1] == f"file://{videos_dir}/ext1.mp4"
    assert extend_uris[2] == f"file://{videos_dir}/ext2.mp4"
    assert result["video_url"] == f"file://{videos_dir}/ext3.mp4"
    assert len(result["extension_steps"]) == 3


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_loop_extend_rejects_lite_and_bad_times(
    tmp_path: Path,
) -> None:
    from src.__main__ import loop_extend

    (tmp_path / "images").mkdir()
    (tmp_path / "videos").mkdir()
    ctx = _ctx_wrapping(_gemini_api_app_ctx(tmp_path))

    lite = json.loads(
        await loop_extend(
            ctx=ctx,
            video_uri="file:///x.mp4",
            model="veo-3.1-lite-generate-preview",
        )
    )
    assert "error" in lite and "Lite" in lite["error"]

    bad = json.loads(await loop_extend(ctx=ctx, video_uri="file:///x.mp4", times=99))
    assert "error" in bad and "between 1 and 20" in bad["error"]


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_generate_video_omni_caps_image_uris(tmp_path: Path) -> None:
    """More than 8 image URIs is rejected before any fetch."""
    from src.__main__ import generate_video_omni

    (tmp_path / "images").mkdir()
    (tmp_path / "videos").mkdir()
    ctx = _ctx_wrapping(_gemini_api_app_ctx(tmp_path))

    result = json.loads(
        await generate_video_omni(
            ctx=ctx,
            prompt="p",
            image_uris=[f"file:///img{i}.png" for i in range(9)],
        )
    )
    assert "error" in result and "at most 8" in result["error"]


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_draft_reports_seed_zero_and_inlines_audio_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """seed=0 is reported as ignored (no truthiness bug) and audio_prompt is
    inlined into the omni prompt rather than dropped."""
    from src.__main__ import generate_video

    (tmp_path / "images").mkdir()
    (tmp_path / "videos").mkdir()
    ctx = _ctx_wrapping(_gemini_api_app_ctx(tmp_path))
    out = tmp_path / "videos" / "d.mp4"
    out.write_bytes(b"mp4")

    captured: dict[str, Any] = {}

    async def mock_omni(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return _omni_result(f"file://{out}")

    monkeypatch.setattr("src.__main__.generate_video_omni_impl", mock_omni)

    result = json.loads(
        await generate_video(
            ctx=ctx,
            prompt="scene",
            model="veo-3.1-generate-001",
            draft=True,
            seed=0,
            audio_prompt="soft piano",
            include_audio=True,
        )
    )
    joined = " ".join(result.get("warnings", []))
    assert "seed" in joined
    assert "include_audio" in joined
    # audio_prompt was honored via inlining, so it is NOT in the ignored list.
    assert "audio_prompt" not in joined
    assert captured["prompt"] == "scene\nAudio: soft piano"


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_animatic_warns_on_dropped_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Animatic mode surfaces dropped per-beat controls and ignored
    output_gcs_uri/include_audio instead of silently discarding them."""
    from src.__main__ import generate_clip

    (tmp_path / "images").mkdir()
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    ctx = _ctx_wrapping(_gemini_api_app_ctx(tmp_path))

    n = {"i": 0}
    captured_prompts: list[str] = []

    async def mock_omni(**kwargs: Any) -> dict[str, Any]:
        n["i"] += 1
        captured_prompts.append(kwargs["prompt"])
        out = videos_dir / f"a{n['i']}.mp4"
        out.write_bytes(b"mp4")
        return _omni_result(f"file://{out}", interaction_id=f"int-{n['i']}")

    monkeypatch.setattr("src.__main__.generate_video_omni_impl", mock_omni)

    result = json.loads(
        await generate_clip(
            ctx=ctx,
            beats=[
                {"prompt": "a", "seed": 7, "negative_prompt": "text overlays"},
                {"prompt": "b", "audio_prompt": "rain"},
            ],
            animatic=True,
            include_audio=True,
            output_gcs_uri="gs://bucket/x/",
        )
    )
    joined = " ".join(result.get("warnings", []))
    assert "output_gcs_uri is ignored" in joined
    assert "include_audio is ignored" in joined
    assert "negative_prompt" in joined and "seed" in joined
    # audio_prompt inlined, not dropped.
    assert captured_prompts[1] == "b\nAudio: rain"


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_loop_extend_passes_audio_and_propagates_warnings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """include_audio reaches each extension step and per-step impl warnings
    surface (deduped) in the loop_extend result."""
    from src.__main__ import loop_extend

    (tmp_path / "images").mkdir()
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    ctx = _ctx_wrapping(_gemini_api_app_ctx(tmp_path))

    audio_flags: list[bool] = []
    n = {"i": 0}

    async def mock_impl(**kwargs: Any) -> dict[str, Any]:
        n["i"] += 1
        audio_flags.append(kwargs.get("include_audio"))
        out = videos_dir / f"e{n['i']}.mp4"
        out.write_bytes(b"mp4")
        return {
            "message": "Video generated successfully",
            "video_url": f"file://{out}",
            "model": kwargs.get("model"),
            "warnings": ["same warning each step"],
        }

    monkeypatch.setattr("src.__main__.generate_video_impl", mock_impl)

    start = videos_dir / "s.mp4"
    start.write_bytes(b"mp4")
    result = json.loads(
        await loop_extend(
            ctx=ctx,
            video_uri=f"file://{start}",
            times=2,
            include_audio=True,
        )
    )
    assert audio_flags == [True, True]
    assert result["warnings"] == ["same warning each step"]


def test_client_for_omni_prefers_gemini_api_then_falls_back_to_vertex(
    tmp_path: Path,
) -> None:
    """Omni routing: prefer a dedicated Gemini API client (Interactions is GA
    there); otherwise use the primary client even in Vertex mode (omni is
    documented on Vertex too) rather than refusing."""
    from src.__main__ import _client_for_omni

    vertex_primary = MagicMock()
    vertex_primary._api_client.vertexai = True

    # No dedicated Gemini API client → fall back to the Vertex primary
    # (previously this raised).
    ctx_vertex = AppContext(
        data_folder=tmp_path,
        images_dir=tmp_path / "images",
        videos_dir=tmp_path / "videos",
        client=vertex_primary,
    )
    assert _client_for_omni(ctx_vertex) is vertex_primary

    # Dedicated Gemini API client present → prefer it.
    gemini_client = MagicMock()
    ctx_both = AppContext(
        data_folder=tmp_path,
        images_dir=tmp_path / "images",
        videos_dir=tmp_path / "videos",
        client=vertex_primary,
        gemini_api_client=gemini_client,
    )
    assert _client_for_omni(ctx_both) is gemini_client
