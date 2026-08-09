"""Process-lifecycle and transport hardening for the MCP server.

These cover the startup/transport surface that had no coverage: the
DNS-rebinding settings must track the resolved bind host, a failed client
build must not strand a service-account key file, and inline credentials
supplied through GOOGLE_APPLICATION_CREDENTIALS must survive the per-connection
lifespan cycle an HTTP transport runs.
"""

from __future__ import annotations

import contextlib
import glob
import json
import tempfile
from pathlib import Path

import pytest

import src.__main__ as main_mod
from src.__main__ import (
    _transport_security_for,
    app_lifespan,
    cleanup_credentials,
    setup_vertex_credentials,
)


def test_a_loopback_bind_keeps_the_localhost_rebinding_allowlist() -> None:
    """A local run must keep DNS-rebinding protection: a browser page must not
    be able to drive a localhost-bound server."""
    for host in ("127.0.0.1", "localhost", "::1"):
        settings = _transport_security_for(host)
        assert settings.enable_dns_rebinding_protection is True
        assert "127.0.0.1:*" in settings.allowed_hosts
        assert "localhost:*" in settings.allowed_hosts


@pytest.mark.parametrize("host", ["0.0.0.0", "10.0.0.5", "192.168.1.20"])
def test_a_public_bind_does_not_freeze_a_localhost_host_allowlist(host: str) -> None:
    """The SDK derives rebinding protection from the host AT CONSTRUCTION, and
    the server is built at import with the default 127.0.0.1 — so without a
    re-derivation a container bound to 0.0.0.0 rejected its own service Host
    header with 421. A deliberately public bind must not carry a localhost
    allowlist it can never satisfy."""
    settings = _transport_security_for(host)
    assert settings.enable_dns_rebinding_protection is False


def test_the_middleware_accepts_a_foreign_host_only_on_a_public_bind() -> None:
    """End to end through the SDK's own gate: a container service name passes
    on a 0.0.0.0 bind, and a foreign host is still refused on a loopback
    bind."""
    from mcp.server.transport_security import TransportSecurityMiddleware

    public = TransportSecurityMiddleware(_transport_security_for("0.0.0.0"))
    # Protection off -> the middleware never reaches host validation.
    assert public.settings.enable_dns_rebinding_protection is False

    loopback = TransportSecurityMiddleware(_transport_security_for("127.0.0.1"))
    assert loopback._validate_host("gemini-media-mcp:8000") is False
    assert loopback._validate_host("127.0.0.1:8000") is True


@pytest.mark.asyncio
async def test_a_failed_startup_does_not_strand_the_credentials_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """setup_vertex_credentials writes the service-account key BEFORE the
    client is built, and building a Vertex client can fail. With the write
    outside the try, that key file was left in /tmp — one per connection on an
    HTTP transport, since the lifespan re-runs per connection."""
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.setenv(
        "GOOGLE_SERVICE_ACCOUNT_JSON",
        json.dumps({"type": "service_account", "project_id": "x", "private_key": "K"}),
    )
    monkeypatch.setenv("DATA_FOLDER", str(tmp_path))
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)

    def boom() -> object:
        raise RuntimeError("DefaultCredentialsError")

    monkeypatch.setattr(main_mod, "create_client", boom)

    before = set(glob.glob(str(Path(tempfile.gettempdir()) / "gcp_sa_*.json")))
    with contextlib.suppress(RuntimeError):
        async with app_lifespan(main_mod.mcp):
            pass
    leaked = set(glob.glob(str(Path(tempfile.gettempdir()) / "gcp_sa_*.json"))) - before
    for stale in leaked:  # never reached on success; keep the temp dir clean
        Path(stale).unlink()
    assert leaked == set()


def test_inline_credentials_survive_a_lifespan_cleanup_cycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inline JSON passed via GOOGLE_APPLICATION_CREDENTIALS was overwritten
    with the temp file path, which teardown then deleted — so a second
    connection found GOOGLE_APPLICATION_CREDENTIALS pointing at a deleted file
    and no JSON to rebuild from. Every connection after the first failed."""
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.delenv("GOOGLE_SERVICE_ACCOUNT_JSON", raising=False)
    monkeypatch.setenv(
        "GOOGLE_APPLICATION_CREDENTIALS",
        json.dumps({"type": "service_account", "project_id": "x", "private_key": "K"}),
    )

    first = setup_vertex_credentials()
    assert first is not None and first.exists()
    cleanup_credentials(first)
    assert not first.exists()

    # The second connection must rebuild a real, existing key file.
    second = setup_vertex_credentials()
    try:
        assert second is not None, "second lifespan could not rebuild credentials"
        assert second.exists()
    finally:
        if second is not None:
            cleanup_credentials(second)
