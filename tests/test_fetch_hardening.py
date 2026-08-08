"""Hardening tests for the fetch / network / filesystem / threading layer.

Covers the SSRF address classifier, DNS-rebinding pinning, the gs:// allowlist
applied uniformly to outputs, GCS client construction, the thought-signature
read cap, and the dedicated filesystem-probe thread pool.
"""

import asyncio
import ipaddress
import json
import logging
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from aiohttp import web

import src.__main__ as main_mod
from src.__main__ import (
    MAX_THOUGHT_SIGNATURE_BYTES,
    AppContext,
    _assert_http_host_public,
    _is_public_ip,
    _read_thought_signature,
    _resolve_video_gcs,
    _source_duration_or_none,
    _VettedAddressResolver,
    fetch,
)

# ============================================================================
# (1) SSRF address classification
# ============================================================================


@pytest.mark.parametrize(
    "addr",
    [
        "100.64.0.0",  # RFC 6598 shared address space, first address
        "100.64.1.1",  # standard EKS/GKE pod + internal-LB addressing
        "100.127.255.255",  # RFC 6598, last address
        "::ffff:100.64.1.1",  # the same range reached via a mapped address
        "fec0::1",  # deprecated IPv6 site-local, still is_global on 3.13
        "feff::1",  # fec0::/10, last block
        "169.254.169.254",  # cloud metadata
        "127.0.0.1",
        "10.0.0.1",
        "::1",
        "fe80::1",
        "0.0.0.0",
        "ff02::1",  # multicast scopes that report is_global True
        "::ffff:127.0.0.1",
    ],
)
def test_is_public_ip_rejects_non_routable(addr: str) -> None:
    """Every non-routable range the fetch guard must refuse."""
    assert _is_public_ip(ipaddress.ip_address(addr)) is False


@pytest.mark.parametrize(
    "addr",
    [
        "8.8.8.8",
        "93.184.216.34",
        "100.63.255.255",  # immediately below RFC 6598
        "100.128.0.0",  # immediately above RFC 6598
        "2606:4700:4700::1111",
        "::ffff:8.8.8.8",
    ],
)
def test_is_public_ip_allows_routable(addr: str) -> None:
    """Genuinely public addresses stay fetchable."""
    assert _is_public_ip(ipaddress.ip_address(addr)) is True


@pytest.mark.parametrize(
    "url",
    [
        "http://100.64.1.1/x",
        "http://[::ffff:100.64.1.1]/x",
    ],
)
def test_assert_http_host_public_rejects_shared_address_space(url: str) -> None:
    """RFC 6598 literals are refused, by either spelling."""
    with pytest.raises(ValueError, match="Refusing to fetch non-public address"):
        _assert_http_host_public(url)


def test_assert_http_host_public_rejects_site_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A name resolving into fec0::/10 is refused."""

    def fake_getaddrinfo(host: str, *args: Any, **kwargs: Any) -> list[Any]:
        return [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("fec0::1", 0, 0, 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(ValueError, match="Refusing to fetch non-public address"):
        _assert_http_host_public("http://internal.example.com/x")


def test_assert_http_host_public_rejects_mixed_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A record mixing a public and an internal answer is refused outright.

    Allowing it would leave the choice of address to the connector.
    """

    def fake_getaddrinfo(host: str, *args: Any, **kwargs: Any) -> list[Any]:
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("100.64.1.1", 0)),
        ]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(ValueError, match="Refusing to fetch non-public address"):
        _assert_http_host_public("http://rebind.example.com/x")


def test_assert_http_host_public_returns_vetted_addresses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard hands back what it checked, so the caller can pin it."""

    def fake_getaddrinfo(host: str, *args: Any, **kwargs: Any) -> list[Any]:
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0)),
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("2606:2800::1", 0, 0, 0)),
        ]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    assert _assert_http_host_public("https://example.com/x") == [
        "93.184.216.34",
        "2606:2800::1",
    ]


# ============================================================================
# (2) DNS-rebinding: the vetted address is the one connected to
# ============================================================================


@pytest.mark.asyncio
async def test_vetted_resolver_fails_closed_for_unpinned_host() -> None:
    """An unpinned name must not fall back to a fresh lookup."""
    resolver = _VettedAddressResolver()
    with pytest.raises(OSError, match="No vetted address"):
        await resolver.resolve("unvetted.example.com", 443)


@pytest.mark.asyncio
async def test_vetted_resolver_preserves_hostname_for_sni() -> None:
    """The pinned entry carries the hostname so TLS still validates by name."""
    resolver = _VettedAddressResolver()
    resolver.pin("example.com", ["93.184.216.34"])
    results = await resolver.resolve("example.com", 443)
    assert [r["host"] for r in results] == ["93.184.216.34"]
    assert results[0]["hostname"] == "example.com"
    assert results[0]["family"] == socket.AF_INET
    assert results[0]["port"] == 443


@pytest.mark.asyncio
async def test_vetted_resolver_replaces_pins_per_hop() -> None:
    """A later hop's pin replaces the earlier one rather than accumulating."""
    resolver = _VettedAddressResolver()
    resolver.pin("example.com", ["93.184.216.34"])
    resolver.pin("example.com", ["203.0.113.7"])
    results = await resolver.resolve("example.com", 80)
    assert [r["host"] for r in results] == ["203.0.113.7"]


class _PinnedFetchServer:
    """Local HTTP server used to prove the connection follows the pin.

    The hostnames the tests fetch do not exist in DNS, so a body can only come
    back if the vetted address was the one dialled.
    """

    def __init__(self) -> None:
        self.runner: web.AppRunner | None = None
        self.port = 0

    async def start(self) -> None:
        app = web.Application()

        async def ok(request: web.Request) -> web.Response:
            return web.Response(body=b"pinned-body")

        async def start_redirect(request: web.Request) -> web.Response:
            return web.Response(
                status=302,
                headers={"Location": f"http://second.invalid.test:{self.port}/final"},
            )

        async def final(request: web.Request) -> web.Response:
            return web.Response(body=b"final-body")

        app.router.add_get("/ok", ok)
        app.router.add_get("/start", start_redirect)
        app.router.add_get("/final", final)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        sockets = site._server.sockets  # type: ignore[union-attr]
        self.port = sockets[0].getsockname()[1]

    async def stop(self) -> None:
        if self.runner is not None:
            await self.runner.cleanup()


@pytest_asyncio.fixture
async def pinned_server() -> Any:
    server = _PinnedFetchServer()
    await server.start()
    try:
        yield server
    finally:
        await server.stop()


@pytest.mark.asyncio
@pytest.mark.timeout(20)
async def test_fetch_connects_to_the_vetted_address(
    pinned_server: _PinnedFetchServer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """fetch dials the address the guard vetted, not a second lookup's answer.

    ``pinned.invalid.test`` has no DNS record at all, so a body coming back
    proves the connector used the guard's answer.
    """

    def guard(url: str) -> list[str]:
        return ["127.0.0.1"]

    monkeypatch.setattr(main_mod, "_assert_http_host_public", guard)

    result = await fetch(
        f"http://pinned.invalid.test:{pinned_server.port}/ok",
        allowed_dir=tmp_path,
    )
    assert result == b"pinned-body"


@pytest.mark.asyncio
@pytest.mark.timeout(20)
async def test_fetch_repins_on_every_redirect_hop(
    pinned_server: _PinnedFetchServer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each redirect hop is re-vetted AND re-pinned to its own address."""
    seen: list[str] = []

    def guard(url: str) -> list[str]:
        from urllib.parse import urlparse

        seen.append(urlparse(url).hostname or "")
        return ["127.0.0.1"]

    monkeypatch.setattr(main_mod, "_assert_http_host_public", guard)

    result = await fetch(
        f"http://first.invalid.test:{pinned_server.port}/start",
        allowed_dir=tmp_path,
    )
    assert result == b"final-body"
    assert seen == ["first.invalid.test", "second.invalid.test"]


@pytest.mark.asyncio
@pytest.mark.timeout(20)
async def test_fetch_refuses_unpinned_host(
    pinned_server: _PinnedFetchServer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With nothing pinned the connector fails closed instead of resolving."""

    def guard(url: str) -> list[str]:
        return []

    monkeypatch.setattr(main_mod, "_assert_http_host_public", guard)

    result = await fetch(
        f"http://localhost:{pinned_server.port}/ok",
        allowed_dir=tmp_path,
    )
    assert result is None


# ============================================================================
# (3) gs:// allowlist applied uniformly, including output_gcs_uri
# ============================================================================


def test_resolve_video_gcs_warns_when_no_allowlist(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An output bucket with no allowlist warns like every gs:// input does.

    The README documents an unset allowlist as warn-and-defer, so the URI is
    still returned; what must not happen is silence.
    """
    with caplog.at_level(logging.WARNING, logger="src.__main__"):
        result = _resolve_video_gcs("gs://attacker-bucket/out", None, frozenset(), True)
    assert result == "gs://attacker-bucket/out"
    assert any("attacker-bucket" in record.message for record in caplog.records)
    assert any("GCS_ALLOWED_BUCKETS" in record.message for record in caplog.records)


def test_resolve_video_gcs_warns_for_env_default_bucket(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The env default takes the same path as an explicit destination."""
    with caplog.at_level(logging.WARNING, logger="src.__main__"):
        result = _resolve_video_gcs(None, "gs://env-bucket/out", frozenset(), True)
    assert result == "gs://env-bucket/out"
    assert any("env-bucket" in record.message for record in caplog.records)


def test_resolve_video_gcs_rejects_bucket_outside_allowlist() -> None:
    """A configured allowlist still hard-refuses an unlisted output bucket."""
    with pytest.raises(ValueError, match="not in the allowlist"):
        _resolve_video_gcs("gs://attacker-bucket/out", None, frozenset({"good"}), True)


def test_resolve_video_gcs_quiet_for_allowlisted_bucket(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A bucket that passes a configured allowlist produces no warning."""
    with caplog.at_level(logging.WARNING, logger="src.__main__"):
        result = _resolve_video_gcs("gs://good/out", None, frozenset({"good"}), True)
    assert result == "gs://good/out"
    assert caplog.records == []


# ============================================================================
# (4) GCS client: off the event loop, and built once
# ============================================================================


class _FakeBlob:
    size = 4

    def reload(self) -> None:
        return None

    def download_as_bytes(self, start: int = 0, end: int | None = None) -> bytes:
        return b"blob"


class _FakeBucket:
    def blob(self, path: str) -> _FakeBlob:
        return _FakeBlob()


class _FakeStorageClient:
    def bucket(self, name: str) -> _FakeBucket:
        return _FakeBucket()


@pytest.fixture(autouse=True)
def _reset_storage_client() -> Any:
    main_mod._storage_client = None
    yield
    main_mod._storage_client = None


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_storage_client_is_built_off_the_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Credential discovery blocks for seconds; it must not run on the loop."""
    threads: list[str] = []

    def fake_client() -> _FakeStorageClient:
        threads.append(threading.current_thread().name)
        return _FakeStorageClient()

    monkeypatch.setattr("src.__main__.storage.Client", fake_client)

    result = await fetch(
        "gs://mybucket/obj", allowed_gcs_buckets=frozenset({"mybucket"})
    )
    assert result == b"blob"
    assert threads and threading.current_thread().name not in threads


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_storage_client_is_memoised_across_fetches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two fetches pay the credential-discovery cost once, not twice."""
    calls = 0

    def fake_client() -> _FakeStorageClient:
        nonlocal calls
        calls += 1
        return _FakeStorageClient()

    monkeypatch.setattr("src.__main__.storage.Client", fake_client)

    assert await fetch("gs://b/one", allowed_gcs_buckets=frozenset({"b"})) == b"blob"
    assert await fetch("gs://b/two", allowed_gcs_buckets=frozenset({"b"})) == b"blob"
    assert calls == 1


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_storage_client_failure_is_not_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Credentials can appear after startup, so a failure must be retried."""
    attempts = 0

    def fake_client() -> _FakeStorageClient:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("no credentials yet")
        return _FakeStorageClient()

    monkeypatch.setattr("src.__main__.storage.Client", fake_client)

    assert await fetch("gs://b/one", allowed_gcs_buckets=frozenset({"b"})) is None
    assert await fetch("gs://b/two", allowed_gcs_buckets=frozenset({"b"})) == b"blob"


# ============================================================================
# (5) thought_signature_url is size-capped and read off the loop
# ============================================================================


def test_read_thought_signature_rejects_oversize(tmp_path: Path) -> None:
    """A big file inside DATA_FOLDER is refused, not slurped into memory."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    big = data_dir / "sig.txt"
    big.write_bytes(b"x" * (MAX_THOUGHT_SIGNATURE_BYTES + 1))
    with pytest.raises(ValueError, match="over the"):
        _read_thought_signature(big, data_dir)


def test_read_thought_signature_reads_small_file(tmp_path: Path) -> None:
    """A real signature is well under the cap and still round-trips."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    sig = data_dir / "sig.txt"
    sig.write_text("opaque-signature")
    assert _read_thought_signature(sig, data_dir) == "opaque-signature"


def test_read_thought_signature_still_enforces_containment(tmp_path: Path) -> None:
    """The size cap does not replace the DATA_FOLDER containment check."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("nope")
    with pytest.raises(ValueError, match="outside the allowed directory"):
        _read_thought_signature(outside, data_dir)


@pytest.mark.asyncio
@pytest.mark.timeout(20)
async def test_generate_image_rejects_oversize_thought_signature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """generate_image must not read an unbounded signature file."""
    from src.__main__ import generate_image

    monkeypatch.setattr(main_mod, "MAX_THOUGHT_SIGNATURE_BYTES", 16)

    async def should_not_run(**kwargs: Any) -> dict[str, Any]:
        raise AssertionError("impl must not run on an oversize signature")

    monkeypatch.setattr("src.__main__.generate_image_impl", should_not_run)

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    sig = tmp_path / "sig.txt"
    sig.write_bytes(b"x" * 4096)

    ctx = MagicMock()
    ctx.info = AsyncMock()
    ctx.error = AsyncMock()
    ctx.warning = AsyncMock()
    ctx.request_context.lifespan_context = AppContext(
        data_folder=tmp_path,
        images_dir=images_dir,
        videos_dir=tmp_path / "videos",
        client=MagicMock(),
    )

    result = await generate_image(
        ctx=ctx,
        prompt="edit",
        model="gemini-3.1-flash-image",
        thought_signature_url=f"file://{sig}",
    )
    payload = json.loads(result[0].text)
    assert "error" in payload
    assert "cap" in payload["error"]


# ============================================================================
# (6) Filesystem probes get their own pool and cannot starve renders
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(20)
async def test_sidecar_probe_runs_on_the_probe_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The scan must not land on the loop's shared default executor."""

    def record(videos_dir: Path, interaction_id: str) -> Any:
        return threading.current_thread().name

    monkeypatch.setattr(main_mod, "_source_duration_for_interaction", record)

    name = await _source_duration_or_none(tmp_path, "abc")
    assert isinstance(name, str)
    assert name.startswith("fs-probe")


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_wedged_probes_do_not_starve_render_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Render work still gets a thread while probe threads are wedged.

    This is the whole point of the dedicated pool: an uncancellable scan on a
    stale mount used to park a thread of the SHARED default executor, and
    enough of them stopped every generation path from ever running again while
    each request still reported a clean timeout.
    """
    release = threading.Event()
    started = threading.Semaphore(0)
    wedged_count = 2

    def wedged(videos_dir: Path, interaction_id: str) -> Any:
        started.release()
        release.wait(30)
        return None

    monkeypatch.setattr(main_mod, "_source_duration_for_interaction", wedged)

    loop = asyncio.get_running_loop()
    render_pool = ThreadPoolExecutor(
        max_workers=wedged_count, thread_name_prefix="render-pool"
    )
    loop.set_default_executor(render_pool)

    probes = [
        asyncio.create_task(_source_duration_or_none(tmp_path, f"i{n}"))
        for n in range(wedged_count)
    ]
    try:
        deadline = time.monotonic() + 10
        acquired = 0
        while acquired < wedged_count and time.monotonic() < deadline:
            if started.acquire(blocking=False):
                acquired += 1
            else:
                await asyncio.sleep(0.01)
        assert acquired == wedged_count, "probe threads never started"

        rendered = await asyncio.wait_for(
            asyncio.to_thread(lambda: "rendered"), timeout=5
        )
        assert rendered == "rendered"
    finally:
        release.set()
        await asyncio.gather(*probes, return_exceptions=True)
        render_pool.shutdown(wait=True)
