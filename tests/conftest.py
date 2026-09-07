"""Suite-wide fixtures.

The one thing here is the calendar. `src.omni` asks the real clock whether the
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

import pytest

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
