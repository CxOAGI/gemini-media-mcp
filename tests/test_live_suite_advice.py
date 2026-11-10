"""Hermetic checks of the live suite's own plumbing (tests/test_live_omni_1_2.py).

The live tests spend money and run only on request, so what they conclude
from a failure has to be right the first time: each one's failure message
names a switch to flip, and flipping it on the wrong evidence changes what
every omni call does.
"""

import json
from types import SimpleNamespace
from typing import Any

import pytest

from tests.test_live_omni_1_2 import (
    _LIVE_CALL_TIMEOUT_SECONDS,
    _Live,
    _l2_verdict,
    _l3_verdict,
)

_FLIP_L3 = "set src.omni.OMNI_REFUSE_LARGE_INLINE_ON_VERTEX = True"
_FLIP_L2 = "set src.omni.OMNI_CONVERSATIONAL_EXTENSION_SENDS_DURATION = False"
# A Vertex capacity failure as the tools now report it.
_CAPACITY_FAILURE = (
    "Omni interaction i-9 ended with status 'failed': Resource exhausted. "
    "Please try again later. (code too_many_requests). This is the service's "
    "capacity or the project's quota, not this request: it can be sent again "
    "unchanged once capacity frees up."
)


@pytest.mark.parametrize(
    "error, flips",
    [
        (
            "Interaction i-9 completed, but its response carried no video: the "
            "video was requested inline (no output_gcs_uri was sent) and no "
            "bytes came back.",
            True,
        ),
        ("413 Request Entity Too Large", True),
        ("Response payload size exceeds the limit", True),
        ("RemoteProtocolError: Server disconnected without sending a response", True),
        # None of these says anything about inline delivery.
        ("Omni video interaction timed out after 210s.", False),
        ("429 RESOURCE_EXHAUSTED: Quota exceeded for aiplatform", False),
        ("403 PERMISSION_DENIED: aiplatform.endpoints.predict denied", False),
        ("Interaction i-4135 ended with status 'failed': internal error", False),
        (_CAPACITY_FAILURE, False),
    ],
)
def test_only_an_inline_delivery_failure_tells_l3_to_flip_its_switch(
    error: str, flips: bool
) -> None:
    """L3 told the reader to refuse large inline renders for ANY upscale
    error -- a timeout at the tools' 210s default deadline included."""
    verdict = _l3_verdict(error)
    assert (_FLIP_L3 in verdict) is flips
    if not flips:
        assert "does not settle D3" in verdict


@pytest.mark.parametrize(
    "error, flips",
    [
        (
            "Duration cannot be set in response format for edit task",
            True,
        ),
        (
            "Pass output_gcs_uri (a prefix such as gs://bucket/renders/): an extension turn, "
            "which returns the whole assembled clip, can exceed the 4 MB inline "
            "response limit. Nothing was rendered or billed.",
            False,
        ),
        ("Omni video interaction timed out after 600s.", False),
        # What L2's first Vertex run failed with.
        (_CAPACITY_FAILURE, False),
    ],
)
def test_only_a_duration_failure_tells_l2_to_flip_its_switch(
    error: str, flips: bool
) -> None:
    """An inline-delivery refusal on Vertex was reported as a reason to stop
    sending duration on conversational turns."""
    verdict = _l2_verdict(error)
    assert (_FLIP_L2 in verdict) is flips
    if not flips:
        assert "does not settle F11" in verdict


@pytest.mark.asyncio
async def test_every_live_call_carries_a_deadline_past_the_tools_default() -> None:
    """With no timeout_seconds, a live call kept the tools' 210s default under
    a 900-1800s pytest timeout, so a slow render failed as a timeout and was
    read as the answer to the question the test asks."""
    seen: list[dict[str, Any]] = []

    async def tool(ctx: Any, **kwargs: Any) -> str:
        seen.append(kwargs)
        return json.dumps({"model": "m"})

    live = _Live(SimpleNamespace(), [])
    await live.call(tool, prompt="p")
    await live.call(tool, prompt="p", timeout_seconds=900)
    assert seen[0]["timeout_seconds"] == _LIVE_CALL_TIMEOUT_SECONDS
    assert _LIVE_CALL_TIMEOUT_SECONDS > 210
    assert seen[1]["timeout_seconds"] == 900


@pytest.mark.asyncio
async def test_a_tool_without_a_deadline_is_not_handed_one() -> None:
    """generate_clip takes no timeout_seconds; passing one made L12b fail as
    an unexpected argument instead of rendering."""
    seen: list[dict[str, Any]] = []

    async def generate_clip(ctx: Any, beats: list[Any]) -> str:
        seen.append({"beats": beats})
        return json.dumps({"model": "m"})

    live = _Live(SimpleNamespace(), [])
    await live.call(generate_clip, beats=[])
    assert seen == [{"beats": []}]


@pytest.mark.live
@pytest.mark.parametrize("spend, real", [("1", True), (None, False), ("0", False)])
def test_only_an_enabled_live_test_gets_real_cloud_storage(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    spend: str | None,
    real: bool,
) -> None:
    """The storage stub ran in every test, so L8's upscale could not measure
    the gs:// object L8 had just rendered and failed with "Supply
    source_resolution" after the first render billed."""
    from tests.conftest import live_spend_enabled

    if spend is None:
        monkeypatch.delenv("OMNI_LIVE_SPEND", raising=False)
    else:
        monkeypatch.setenv("OMNI_LIVE_SPEND", spend)
    assert live_spend_enabled(request.node) is real


def test_a_hermetic_test_keeps_the_storage_stub_whatever_the_environment(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.conftest import live_spend_enabled

    monkeypatch.setenv("OMNI_LIVE_SPEND", "1")
    assert not live_spend_enabled(request.node)


def test_every_live_test_carries_the_live_marker() -> None:
    import tests.test_live_omni_1_2 as live

    assert any(mark.name == "live" for mark in live.pytestmark)


@pytest.mark.live
def test_a_live_test_not_switched_on_still_gets_the_storage_stub() -> None:
    """This file runs without OMNI_LIVE_SPEND in the hermetic suite, so the
    fixture has already applied the stub here."""
    import os

    import src.__main__ as server

    if os.environ.get("OMNI_LIVE_SPEND") == "1":
        pytest.skip("live spending is switched on for this run")
    with pytest.raises(RuntimeError, match="reached real Cloud Storage"):
        server.storage.Client()
