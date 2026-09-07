"""Suite-wide fixtures.

The one thing here is the calendar -- both of them. `src.omni` asks the real clock whether the
preview endpoint's switch-off date has passed, and 17 tests across
`test_omni.py`, `test_wire_contract_omni.py` and `test_routing.py` pass
`model=OMNI_PREVIEW_MODEL` and expect it to work. On 2026-09-30 the real clock
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

import pytest

import src.__main__ as _server
import src.image as image
import src.omni as omni

# Before OMNI_PREVIEW_SUNSET (2026-09-30), so the preview model still serves and
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

@pytest.fixture(autouse=True)
def _forget_service_account_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    # raising=False: the suite is sometimes run against an older src/ to prove
    # a new test fails there, and these attributes may not exist yet.
    monkeypatch.setattr(_server, "_service_account_creds", None, raising=False)
    monkeypatch.setattr(_server, "_service_account_project", None, raising=False)


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
