"""Suite-wide fixtures.

The one thing here is the calendar -- both of them. `src.omni` asks the real clock whether the
preview endpoint's switch-off date has passed, and 17 tests across
`test_omni.py`, `test_wire_contract_omni.py` and `test_routing.py` pass
`model=OMNI_PREVIEW_MODEL` and expect it to work. On 2026-10-22 the real clock
starts answering "yes" and every one of them fails with the refusal — 38 tests
in all, on a date nobody would connect to a code change. Only
`test_budget_and_lifecycle.py` pinned the calendar, and only for its own cases.

So the clock is pinned for the whole suite, and the tests that are ABOUT the
sunset pin their own date on top (a function-scoped `monkeypatch.setattr` runs
after this fixture, so it wins) and assert both sides of it deliberately.
"""

from __future__ import annotations

import datetime
import functools
import json
import os

import pytest
from collections.abc import Iterator

import src.__main__ as _server
import src.image as image
import src.omni as omni
import src.pricing as pricing


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "live: a paid test against the real API; runs only with OMNI_LIVE_SPEND=1",
    )


def live_spend_enabled(node: pytest.Item) -> bool:
    """Whether this is a paid live test the caller explicitly switched on.

    Such a test talks to the real services on purpose, so the fixtures that
    keep the hermetic suite off the network stand aside for it.
    """
    return (
        node.get_closest_marker("live") is not None
        and os.environ.get("OMNI_LIVE_SPEND") == "1"
    )


# Before OMNI_PREVIEW_SUNSET (2026-10-22), so the preview model still serves and
# the suite tests the pre-sunset behaviour it was written for. Any date before
# the sunset would do; this one is fixed so a failure is never about "when".
PINNED_TODAY = datetime.date(2026, 9, 15)


@pytest.fixture(autouse=True)
def _pin_the_calendar(monkeypatch: pytest.MonkeyPatch) -> None:
    """Freeze `src.omni`'s idea of today for every test.

    Autouse rather than opt-in: the failure mode this prevents is a test that
    never mentioned the date breaking because the date moved, so requiring
    tests to remember to ask for it would not have prevented it.
    """
    monkeypatch.setattr(omni, "_today", lambda: PINNED_TODAY)
    # Understanding-model rates step up on published dates (Argon on
    # 2026-12-01, 3.8 Flash on 2027-01-01); unpinned, a quote assertion would
    # change on those days.
    monkeypatch.setattr(pricing, "_today", lambda: PINNED_TODAY)

    # The second clock. src.image reads `date.today()` directly for the image
    # model shutdown phrasing, and the first version of this fixture pinned
    # only omni's -- so test_sunset_model_warning_does_not_claim_it_is_already_
    # gone would have failed on 2026-10-02 on its own. `date` is imported by
    # name there, so the class itself is swapped for one whose today() is
    # pinned; every other date method is inherited unchanged.
    class _PinnedDate(datetime.date):
        @classmethod
        def today(cls) -> datetime.date:  # type: ignore[override]
            return PINNED_TODAY

    monkeypatch.setattr(image, "date", _PinnedDate)


# ---------------------------------------------------------------------------
# Service-account credentials
#
# src.__main__ builds Vertex credentials once per process from the inline
# service-account JSON and memoises them. Tests set different environments, so
# the memo is cleared for every test; and a test that wants the real
# from_service_account_info path gets a real-shaped key, minted once.
# ---------------------------------------------------------------------------

_CREDENTIAL_ENV = (
    "GOOGLE_SERVICE_ACCOUNT_JSON",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "GOOGLE_CLOUD_PROJECT",
)


@pytest.fixture(autouse=True)
def _restore_credential_environment() -> Iterator[None]:
    """Snapshot the credential variables and put them back after every test.

    src.credentials.normalize_environment writes GOOGLE_SERVICE_ACCOUNT_JSON
    and deletes GOOGLE_APPLICATION_CREDENTIALS directly -- the point of it, in
    production. But monkeypatch.delenv on a variable that was ABSENT records
    nothing to restore, so a placeholder key one test normalised into the
    environment leaked into the next, which then failed with "not a usable
    service-account key" having set no key at all.
    """
    before = {name: os.environ.get(name) for name in _CREDENTIAL_ENV}
    yield
    for name, value in before.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


@pytest.fixture(autouse=True)
def _forget_service_account_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    # The memo lives in src.credentials now (shared with image.py). raising=
    # False: the suite is sometimes run against an older src/ to prove a new
    # test fails there, and the module may not exist yet.
    import src.credentials as creds

    monkeypatch.setattr(creds, "_creds", None, raising=False)
    monkeypatch.setattr(creds, "_project", None, raising=False)
    monkeypatch.setattr(_server, "_service_account_creds", None, raising=False)
    monkeypatch.setattr(_server, "_service_account_project", None, raising=False)


@pytest.fixture(autouse=True)
def _no_real_cloud_storage(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Make an unstubbed gs:// fetch fail at once instead of reaching GCS.

    fetch() builds a real google-cloud-storage client for a gs:// URI. In a
    test that did not stub it, that meant ~12s of credential discovery per
    call offline, and on a machine with Application Default Credentials a
    real download from whatever bucket the test named. A test that wants the
    client stubs ``src.__main__.storage.Client`` itself (function-scoped, so
    it runs after this and wins) or stubs fetch().

    An enabled live test is the exception: L8 reads back the object it just
    rendered to the caller's bucket, and with this stub in place the upscale
    could not measure its source and failed after the first render billed.
    """
    if live_spend_enabled(request.node):
        return

    def _refuse(*args: object, **kwargs: object) -> None:
        raise RuntimeError(
            "This test reached real Cloud Storage: stub src.__main__.fetch or "
            "src.__main__.storage.Client."
        )

    monkeypatch.setattr(_server.storage, "Client", _refuse)
    monkeypatch.setattr(_server, "_storage_client", None)


@functools.lru_cache(maxsize=1)
def fake_service_account_json(project_id: str = "proj-x") -> str:
    """A syntactically valid service-account key with a freshly minted RSA key.

    google-auth parses the PEM when it builds the credentials object, so the
    placeholder keys the older tests used ("private_key": "K") no longer get
    past construction. Nothing here can authenticate to anything.
    """
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    return json.dumps(
        {
            "type": "service_account",
            "project_id": project_id,
            "private_key_id": "test-key",
            "private_key": pem,
            "client_email": f"svc@{project_id}.iam.gserviceaccount.com",
            "client_id": "1",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    )


@pytest.fixture
def service_account_json() -> str:
    """The minted key above, as a fixture."""
    return fake_service_account_json()
