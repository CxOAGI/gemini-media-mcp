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
import os
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
    monkeypatch: pytest.MonkeyPatch, service_account_json: str
) -> None:
    """Inline JSON passed via GOOGLE_APPLICATION_CREDENTIALS used to be written
    to a temp file that teardown deleted. There is no file now: the same
    credentials object serves every lifespan, and nothing is repointed."""
    import glob
    import tempfile

    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.delenv("GOOGLE_SERVICE_ACCOUNT_JSON", raising=False)
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", service_account_json)

    before = set(glob.glob(str(Path(tempfile.gettempdir()) / "gcp_sa_*.json")))
    first = setup_vertex_credentials()
    assert first is None
    cleanup_credentials(first)  # a no-op on None, and must stay one
    second = setup_vertex_credentials()
    assert second is None
    assert set(glob.glob(str(Path(tempfile.gettempdir()) / "gcp_sa_*.json"))) == before

    # The inline JSON is kept -- under the variable that means "inline". The
    # path variable is cleared, because google-auth reads it as a FILE PATH and
    # its error for a missing file quotes the variable's whole value: with key
    # text there, the private key was returned to the MCP caller.
    assert "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ
    assert os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"] == service_account_json
    found = main_mod._service_account_credentials()
    assert found is not None
    creds, project = found
    assert project == "proj-x"
    # Memoised: every lifespan sees the one object.
    assert main_mod._service_account_credentials()[0] is creds  # type: ignore[index]


@pytest.mark.asyncio
async def test_one_session_disconnecting_does_not_break_another(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, service_account_json: str
) -> None:
    """The defect, end to end: on sse/streamable-http the MCP SDK enters
    app_lifespan once PER SESSION. Each lifespan wrote GOOGLE_SERVICE_ACCOUNT_
    JSON to its own temp file, repointed the process-wide GOOGLE_APPLICATION_
    CREDENTIALS at it, and deleted its file on teardown -- so a probe client
    connecting and disconnecting deleted the file every other live session's
    lazily-loaded Vertex credentials pointed at, and their first tool call
    failed with DefaultCredentialsError naming the deleted path. Verified
    against mcp 1.28.1 / google-genai 2.20.0; the Dockerfile ships exactly
    this configuration.

    With a credentials object there is no file to delete and no variable to
    repoint, so session B's teardown leaves session A exactly as it was."""
    import glob
    import tempfile

    monkeypatch.setenv("DATA_FOLDER", str(tmp_path))
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_JSON", service_account_json)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)

    built: list[dict[str, object]] = []

    def fake_client(**kwargs: object) -> object:
        built.append(kwargs)
        return object()

    monkeypatch.setattr(main_mod.genai, "Client", fake_client)
    before = set(glob.glob(str(Path(tempfile.gettempdir()) / "gcp_sa_*.json")))

    async with app_lifespan(main_mod.mcp):  # session A
        async with app_lifespan(main_mod.mcp):  # a probe: session B
            pass
        # B has torn down. A must still hold live, file-free credentials.
        creds_a = [k for k in built if k.get("vertexai")][0]["credentials"]
        assert creds_a is not None
        assert main_mod._service_account_credentials()[0] is creds_a  # type: ignore[index]
        assert "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ
        # The module-global Vertex omni client is built the same way.
        main_mod._omni_vertex_global_client = None
        main_mod._get_omni_vertex_global_client()
        assert built[-1]["credentials"] is creds_a and built[-1]["location"] == "global"

    assert set(glob.glob(str(Path(tempfile.gettempdir()) / "gcp_sa_*.json"))) == before
    main_mod._omni_vertex_global_client = None


def test_an_unusable_service_account_key_fails_loudly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Valid JSON that is not a key must not fall through to ambient ADC."""
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.setenv(
        "GOOGLE_SERVICE_ACCOUNT_JSON",
        json.dumps({"type": "service_account", "project_id": "x", "private_key": "K"}),
    )
    with pytest.raises(ValueError, match="not a usable service-account key"):
        main_mod.create_client()


def test_both_calendars_are_pinned_for_the_suite() -> None:
    """src.image reads date.today() too, and the first pin covered only omni.

    Without this, test_sunset_model_warning_does_not_claim_it_is_already_gone
    would have failed on 2026-10-02 on its own. The class is swapped rather
    than the module function, because image.py imports `date` by name.
    """
    import datetime

    import src.image as image
    import src.omni as omni

    assert image.date is not datetime.date, "image clock is not pinned"
    assert image.date.today() == omni._today()
    assert isinstance(image.date.today(), datetime.date)


def test_the_image_models_vertex_client_gets_the_same_credentials(
    monkeypatch: pytest.MonkeyPatch, service_account_json: str
) -> None:
    """image.py builds its own Vertex client for the Gemini-3 image models, and
    it was the one client the credentials redesign missed: on an inline-JSON
    deployment it fell into ADC, found nothing, and every default-model image
    render failed with DefaultCredentialsError while Veo and omni worked."""
    import src.image as image

    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_JSON", service_account_json)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)

    built: list[dict[str, object]] = []
    monkeypatch.setattr(image.genai, "Client", lambda **kw: built.append(kw) or object())
    monkeypatch.setattr(image, "_vertex_global_client", None)

    image._get_vertex_global_client()
    assert built and built[0]["vertexai"] is True
    assert built[0]["location"] == "global"
    assert built[0]["credentials"] is not None
    assert built[0]["project"] == "proj-x"
    # Same object the server's own clients hold.
    assert built[0]["credentials"] is main_mod._service_account_credentials()[0]  # type: ignore[index]


def test_a_key_without_a_project_fails_with_the_missing_name_not_adc(
    monkeypatch: pytest.MonkeyPatch, service_account_json: str
) -> None:
    """google-genai calls google.auth.default() whenever project is None,
    whatever credentials it was handed -- so a key with no project_id and no
    GOOGLE_CLOUD_PROJECT tumbled into ADC discovery (and, in a container, a
    metadata-server probe) instead of saying what was missing."""
    info = json.loads(service_account_json)
    info.pop("project_id")
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_JSON", json.dumps(info))
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)

    with pytest.raises(ValueError, match="GOOGLE_CLOUD_PROJECT"):
        main_mod.create_client()


def test_key_text_never_stays_in_the_path_variable(
    monkeypatch: pytest.MonkeyPatch, service_account_json: str
) -> None:
    """Whichever spelling the operator used, after setup no variable that
    google-auth reads as a path holds key text -- the precondition for its
    "File {...} was not found" message ever containing a private key."""
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.delenv("GOOGLE_SERVICE_ACCOUNT_JSON", raising=False)
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", service_account_json)

    setup_vertex_credentials()
    assert "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ
    assert "BEGIN PRIVATE KEY" in os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]
    # And a real path is left alone.
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/etc/gcp/key.json")
    setup_vertex_credentials()
    assert os.environ["GOOGLE_APPLICATION_CREDENTIALS"] == "/etc/gcp/key.json"
