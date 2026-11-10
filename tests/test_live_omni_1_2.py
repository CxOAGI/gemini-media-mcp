"""Live, paid checks of the questions about gemini-omni-1.2-flash that only a real render can settle.

Every test here spends money, so none runs by accident. Each one skips unless
the credentials it needs are set AND ``OMNI_LIVE_SPEND=1``. L3 and L3b, the
4K ones, also need ``OMNI_LIVE_EXPENSIVE=1``. With neither set, the whole module
is collected and skipped. A plain ``uv run pytest`` never pays for anything.

How to run (from the repo root; ``-s`` shows what each test measured)::

    # Vertex AI: L1, L2, L4-L7, L9-L13 (Vertex)
    GOOGLE_GENAI_USE_VERTEXAI=true GOOGLE_CLOUD_PROJECT=<project> \\
    OMNI_LIVE_SPEND=1 uv run pytest tests/test_live_omni_1_2.py -v -s

    # Gemini API: L2, L4, L5, L7, L9-L13 (Gemini API)
    GEMINI_API_KEY=<key> OMNI_LIVE_SPEND=1 \\
    uv run pytest tests/test_live_omni_1_2.py -v -s

    # Both backends at once: set both credentials. L3 and L3b need
    # OMNI_LIVE_EXPENSIVE=1 as well, and L8 a bucket in OMNI_LIVE_GCS_BUCKET:
    GOOGLE_GENAI_USE_VERTEXAI=true GOOGLE_CLOUD_PROJECT=<project> \\
    GEMINI_API_KEY=<key> OMNI_LIVE_SPEND=1 OMNI_LIVE_EXPENSIVE=1 \\
    OMNI_LIVE_GCS_BUCKET=<bucket> \\
    uv run pytest tests/test_live_omni_1_2.py -v -s

Vertex credentials are whatever the server takes (ADC, or
GOOGLE_SERVICE_ACCOUNT_JSON); GOOGLE_API_KEY is accepted for GEMINI_API_KEY.
The server is the real one, run in-process through its own lifespan, so each
test can print the exact ``interactions.create`` body it sent next to what came
back.

Expected cost per test, at pricing.py's published 1.2 rates:

* L1: about $0.18. A 3s 360p render ($0.07), then that interaction upscaled
  to 720p ($0.11).
* L1b: about $0.28. The same, then that upscale's interaction upscaled to
  1080p ($0.10), which is sent the clip it saved.
* L2: about $0.42 per backend. A 3s 360p render ($0.07), then two 3s
  extension turns, which bill the assembled 6s and 9s ($0.14 + $0.21). If the
  service ignores the duration and appends its 10s default, the turns bill
  the assembled 13s and 23s instead, about $0.84, for about $0.91 in all:
  max_cost_usd checks the estimate, which assumes 3s turns, not the bill.
* L3: about $2.10. A 10s 720p render ($0.70), then that render upscaled to
  4K ($1.40).
* L3b: about $0.63. A 3s 4K render.
* L4: about $0.07. A 3s 360p render.
* L5: about $0.21 per backend. A 3s 360p render, then that file extended by
  3s, billed as the assembled 6s ($0.14).
* L6: about $0.18. A 3s 360p render, then that file upscaled to 720p.
* L7: about $0.28 per backend. A 12s 360p render.
* L8: about $0.18, Vertex AI with OMNI_LIVE_GCS_BUCKET only. A 3s 360p render
  to that bucket, then an upscale read from it.
* L9a, L9b, L9c, L10a, L10b: about $0.07 each per backend. One 3s 360p render
  each.
* L11a, L11b: about $0.14 each per backend (up to $0.42). A 3s 360p render,
  then an edit whose length the service picks, up to 15s. Live, on both
  backends, it kept the source's 3s (2026-11-10).
* L12a, L12b: about $0.07 each, on the backend the server's default picks.
* L12c: about $0.28, on the backend the server's default picks. Two 4s 360p
  beats and the 4s bridge between them.
* L13: about $0.07 per backend. A 3s 360p render that times out after 15s and
  is collected by id; collecting bills nothing.

All of L5-L13 on both backends, with a bucket: about $3.16. L1b: about $0.28.

What each outcome means for the code:

* L1 (F5): a conversational upscale is sent with ``background=True``,
  ``previous_interaction_id``, ``generation_config.video_config.task=
  'upscale'`` and an empty ``input``. Vertex's documented example uses
  ``background: false`` and the text "Upsample video."; the first L1 run
  (2026-11-09) sent that text and got a 400, "Text input is not supported for
  upscale task." A follow-up probe was accepted with ``input: []`` and
  returned a 720p upscale at the source's 3.03s. The same probe accepted a
  stateless upscale of the saved clip sent as an inline video part.
  - Passes: keep that request shape. Nothing changes. It also checks that
    the upscale kept the source's length, since an edit read of the same
    request comes back at 720p too, but re-rendered at about 10s.
  - Fails with an API error about the task or ``background``, or with the
    upscale's length off the source's: change the upscale exception in
    ``src/omni.py::_build_create_kwargs`` to match what the API accepts. The
    error text is in the failure message.
* L2 (F11): does a turn that carries ``previous_interaction_id`` accept
  ``response_format.duration``? This runs once per backend.
  - Passes on both: keep
    ``src.omni.OMNI_CONVERSATIONAL_EXTENSION_SENDS_DURATION = True``.
  - Fails on either, by rejecting the duration or by measuring about 10s per
    turn instead of 3s: set it to False. Such turns then append the 10s
    default, and ``extend_video_omni`` refuses a ``duration_seconds`` it
    cannot honour.
* L3 (D3): do a large render's bytes arrive inline on Vertex AI with no
  ``output_gcs_uri``? The test prints the size that arrived.
  - Passes: keep ``src.omni.OMNI_REFUSE_LARGE_INLINE_ON_VERTEX = False``.
    The four omni tools keep warning and rendering inline.
  - Fails with "completed, but its response carried no video", or with a
    transport or size error: set it to True. Every omni tool then refuses
    such a request before rendering, dry run included, and asks for
    ``output_gcs_uri``. Any other failure (a timeout, quota or permission
    error) does not settle the question, and the failure message says so.
* L3b (D3): the same question for a fresh 4K render, since L3's evidence is
  Vertex's upscaling page but the switch governs all four tools. Read it as
  L3: both must pass to keep the switch False; a delivery failure on either
  sets it True.
* L4 (D1): a call naming ``gemini-omni-1.1-flash`` reports
  ``gemini-omni-1.2-flash``, sends 1.2, and says why.
  - Fails: the redirect in ``src/omni.py`` (``canonical_omni_model`` and
    ``omni_model_reroute_warning``) is broken.

L5-L13 run every other request shape that is new on 1.2 end to end, with no
switch riding on them: a failure is a bug, and its message carries the API's
own error and the request that drew it. Each checks the request sent and
something measured from the result:

* L5: an uploaded clip extended by 3s (``task: extend``, a duration, the
  video). Measures the source plus 3s; about 3s would mean the service
  returns only the new footage, which the extension pricing does not assume.
* L6: an uploaded file upscaled by the tool (the clip inline, no text).
* L7: a 12s render. URI delivery on the Gemini API, inline on Vertex AI.
* L8: delivery to ``output_gcs_uri``, then an upscale whose source is that
  ``gs://`` object, sent as a URI part.
* L9a/L9b: keyframes. The render's first frame (and, for the pair, its last)
  must look more like the image given for it than like the other one: a
  bright day scene and a dark night scene, drawn here.
* L9c: a loop. One image as both first and last frame is sent once, with both
  tags naming it, and both ends of the render must look like it. This is the
  only check of that request shape; if it fails, send the image twice again.
* L10a/L10b: two reference images bound as ``<IMAGE_REF_n>``, and a 2s
  reference clip bound as ``<VIDEO_REF_0>``.
* L11a/L11b: a night edit, of an interaction and of an uploaded clip; no
  duration or aspect ratio is sent, and the result must be darker.
* L12a/L12b: ``generate_video(draft=True)`` and ``generate_clip(animatic=True)``
  render on 1.2; the animatic, at generate_clip's default 9:16, must measure
  portrait.
* L12c: an animatic with ``add_bridges``; the bridge is a first/last-frame
  render on 1.2 between the beats' frames, measured like a beat.
* L13: a render whose call times out after the create is collected with
  ``get_omni_interaction``: read on the backend that made it, saved, measured
  and recorded, with nothing created.
"""

import inspect
import io
import json
import os
import re
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import src.__main__ as server
import src.omni as omni

# Each live call's own deadline, well past the tools' 210s default: a slow
# render must fail as a slow render, with the interaction id it billed under,
# not as the answer to the question the test asks. Every test's pytest
# timeout covers its calls' deadlines in sequence; an extension chain's
# applies per turn.
_LIVE_CALL_TIMEOUT_SECONDS = 600


def _gemini_api_key() -> str | None:
    """GEMINI_API_KEY, or GOOGLE_API_KEY in its place, as the live suite reads it."""
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def _is_vertex_ai() -> bool:
    return os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() == "true"


# Spending is opt-in, separately from having credentials: a developer with
# credentials in their shell must not pay for a routine test run.
requires_spend = pytest.mark.skipif(
    os.environ.get("OMNI_LIVE_SPEND") != "1",
    reason="paid live test: set OMNI_LIVE_SPEND=1 to run it",
)
requires_expensive = pytest.mark.skipif(
    os.environ.get("OMNI_LIVE_EXPENSIVE") != "1",
    reason=(
        "L3 costs about $2.10 and L3b about $0.63: set OMNI_LIVE_EXPENSIVE=1 "
        "(and OMNI_LIVE_SPEND=1)"
    ),
)
requires_vertex = pytest.mark.skipif(
    not _is_vertex_ai(),
    reason="needs Vertex AI: set GOOGLE_GENAI_USE_VERTEXAI=true and its project",
)
requires_gemini_api = pytest.mark.skipif(
    not _gemini_api_key(),
    reason="needs GEMINI_API_KEY (or GOOGLE_API_KEY)",
)
requires_api = pytest.mark.skipif(
    not _gemini_api_key() and not _is_vertex_ai(),
    reason="GEMINI_API_KEY or Vertex AI credentials not set",
)

# `live` lets the hermetic suite's network fixtures stand aside when
# OMNI_LIVE_SPEND=1 (tests/conftest.py, live_spend_enabled).
pytestmark = [pytest.mark.asyncio, pytest.mark.live]


def _redact(value: Any) -> Any:
    """The request as sent, minus the media bytes, which would drown it out."""
    if isinstance(value, dict):
        return {key: _redact(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    if isinstance(value, (bytes, bytearray)):
        return f"<{len(value)} bytes>"
    if isinstance(value, str) and len(value) > 300:
        return f"<{len(value)} chars>"
    return value


def _file_bytes(video_url: Any) -> int | None:
    if isinstance(video_url, str) and video_url.startswith("file://"):
        path = Path(video_url[len("file://") :])
        if path.is_file():
            return path.stat().st_size
    return None


class _Live:
    """The real server's tools, with every request body recorded."""

    def __init__(self, app_ctx: Any, requests: list[dict[str, Any]]) -> None:
        self.requests = requests
        self.ctx = SimpleNamespace(
            request_context=SimpleNamespace(lifespan_context=app_ctx),
            info=self._say("info"),
            warning=self._say("warning"),
            error=self._say("error"),
        )

    @staticmethod
    def _say(level: str) -> Callable[[str], Awaitable[None]]:
        async def say(message: str) -> None:
            print(f"    [{level}] {message}")

        return say

    async def call(
        self, tool: Callable[..., Awaitable[str]], **kwargs: Any
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Call a tool, print what it measured, and return (payload, requests sent)."""
        # Not every tool takes a deadline (generate_clip has none), and one
        # it does not declare would be refused as an unexpected argument.
        parameters = inspect.signature(tool).parameters
        if "timeout_seconds" in parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        ):
            kwargs.setdefault("timeout_seconds", _LIVE_CALL_TIMEOUT_SECONDS)
        first = len(self.requests)
        started = time.monotonic()
        payload = json.loads(await tool(self.ctx, **kwargs))
        elapsed = time.monotonic() - started
        sent = self.requests[first:]
        _report(tool.__name__, kwargs, payload, sent, elapsed)
        return payload, sent


def _report(
    name: str,
    kwargs: dict[str, Any],
    payload: dict[str, Any],
    sent: list[dict[str, Any]],
    elapsed: float,
) -> None:
    """Print what a live call measured, so a run is readable without a debugger."""
    print(f"\n=== {name}({json.dumps(_redact(kwargs), default=str)}) in {elapsed:.1f}s")
    if "error" in payload:
        print(f"  error: {payload['error']}")
    for key in (
        "model",
        "requested_model",
        "backend",
        "interaction_id",
        "task",
        "duration_seconds",
        "duration_source",
        "resolution",
        "rendered_resolution",
        "resolution_source",
        "video_url",
        "turn_output_seconds",
        "billed_seconds",
        "assembled_seconds",
        "submission",
    ):
        if key in payload:
            print(f"  {key}: {payload[key]}")
    size = _file_bytes(payload.get("video_url"))
    if size is not None:
        print(f"  bytes: {size} ({size / 1_000_000:.2f} MB)")
    for index, segment in enumerate(payload.get("segments") or [], start=1):
        print(
            f"  segment {index}: {segment.get('duration_seconds')}s "
            f"({segment.get('duration_source')}), "
            f"bytes={_file_bytes(segment.get('video_url'))}, "
            f"interaction={segment.get('interaction_id')}"
        )
    cost = payload.get("cost") or payload.get("estimated_cost")
    if isinstance(cost, dict):
        print(f"  cost: ${cost.get('usd')} ({cost.get('detail')})")
    for warning in payload.get("warnings") or []:
        print(f"  warning: {warning}")
    for index, request in enumerate(sent, start=1):
        print(f"  request {index}: {json.dumps(_redact(request), default=str)}")


def _ok(payload: dict[str, Any], sent: list[dict[str, Any]]) -> None:
    """Fail with the API's own error text and the request that drew it."""
    assert "error" not in payload, (
        f"the API refused or failed: {payload['error']}\n"
        f"requests sent: {json.dumps(_redact(sent), default=str)}"
    )


# What a large inline delivery that failed looks like: a completed response
# with no video in it (src.omni._no_video_returned), or a response the
# transport could not carry. A timeout, a quota or a permission error says
# nothing either way, and flipping the switch on one would be a wrong answer.
_INLINE_DELIVERY_FAILURE = re.compile(
    r"carried no video|too large|payload|\b413\b|response size|incomplete"
    r"|connection reset|server disconnected|remoteprotocolerror",
    re.IGNORECASE,
)


def _l3_verdict(error: str) -> str:
    """What L3's failure means for OMNI_REFUSE_LARGE_INLINE_ON_VERTEX."""
    if _INLINE_DELIVERY_FAILURE.search(error):
        return (
            "Large inline delivery did not work on Vertex AI: set "
            "src.omni.OMNI_REFUSE_LARGE_INLINE_ON_VERTEX = True."
        )
    return (
        "This is not a failure large inline delivery explains (a timeout, "
        "quota or permission error, say), so it does not settle D3: leave "
        "src.omni.OMNI_REFUSE_LARGE_INLINE_ON_VERTEX as it is, fix the cause, "
        "and rerun L3."
    )


def _l2_verdict(error: str) -> str:
    """What L2's failure means for OMNI_CONVERSATIONAL_EXTENSION_SENDS_DURATION."""
    if "duration" in error.lower():
        return (
            "The API refused the duration on a conversational turn: set "
            "src.omni.OMNI_CONVERSATIONAL_EXTENSION_SENDS_DURATION = False."
        )
    return (
        "This failure does not concern the duration, so it does not settle "
        "F11: leave src.omni.OMNI_CONVERSATIONAL_EXTENSION_SENDS_DURATION as "
        "it is, fix the cause, and rerun L2."
    )


@asynccontextmanager
async def _live_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[_Live]:
    """The server's own lifespan, on a scratch DATA_FOLDER."""
    monkeypatch.setenv("DATA_FOLDER", str(tmp_path))
    key = _gemini_api_key()
    if key and not os.environ.get("GEMINI_API_KEY"):
        monkeypatch.setenv("GEMINI_API_KEY", key)
    requests: list[dict[str, Any]] = []
    build = omni._build_create_kwargs

    def recording_build(**kwargs: Any) -> dict[str, Any]:
        # The dict the impl sends, recorded by reference, so any field it
        # drops after building (an inferred task) is dropped here too.
        built = build(**kwargs)
        requests.append(built)
        return built

    monkeypatch.setattr(omni, "_build_create_kwargs", recording_build)
    async with server.app_lifespan(server.mcp) as app_ctx:
        yield _Live(app_ctx, requests)


async def _render_cheapest(
    live: _Live, backend: str | None, prompt: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The cheapest 1.2 clip there is: 3s at 360p."""
    kwargs: dict[str, Any] = {
        "prompt": prompt,
        "duration_seconds": 3,
        "resolution": "360p",
    }
    if backend:
        kwargs["omni_backend"] = backend
    clip, sent = await live.call(server.generate_video_omni, **kwargs)
    _ok(clip, sent)
    assert clip["model"] == omni.OMNI_1_2_MODEL, clip
    if backend:
        assert clip["backend"] == backend, clip
    return clip, sent


# ---------------------------------------------------------------------------
# L1: upscaling a conversational interaction on Vertex AI (F5)
# ---------------------------------------------------------------------------


@requires_spend
@requires_vertex
@pytest.mark.timeout(1500)
async def test_L1_vertex_upscales_an_interaction_by_previous_interaction_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F5: background=True + previous_interaction_id + task='upscale' together.

    Expected cost: about $0.18 (3s 360p render $0.07, 360p->720p upscale of
    3s $0.11). Pass: keep the request shape. Fail: the API error printed here
    says which field _build_create_kwargs must change.
    """
    async with _live_server(tmp_path, monkeypatch) as live:
        clip, _ = await _render_cheapest(
            live,
            "vertex",
            "A red ball rolls slowly across a wooden table. Static camera.",
        )
        upscaled, sent = await live.call(
            server.upscale_video_omni,
            previous_interaction_id=clip["interaction_id"],
            resolution="720p",
            max_cost_usd=0.5,
        )
    _ok(upscaled, sent)
    [request] = sent
    assert request["previous_interaction_id"] == clip["interaction_id"]
    # No text: Vertex 400s "Text input is not supported for upscale task."
    assert request["input"] == []
    assert request["background"] is True
    assert request["generation_config"] == {"video_config": {"task": "upscale"}}
    assert upscaled["rendered_resolution"] == "720p", upscaled
    assert upscaled.get("resolution_source", "").startswith("measured"), upscaled
    # An edit read of this request would also come back at 720p, re-rendered
    # at about 10s; only an upscale keeps the source's length.
    assert upscaled.get("duration_source", "").startswith("measured"), upscaled
    assert upscaled["duration_seconds"] == pytest.approx(
        clip["duration_seconds"], abs=0.5
    ), (
        f"the upscale measured {upscaled['duration_seconds']}s from a "
        f"{clip['duration_seconds']}s source: the service did not run "
        "task='upscale' (an edit re-renders at about 10s). Change the upscale "
        "exception in src/omni.py::_build_create_kwargs."
    )


@requires_spend
@requires_vertex
@pytest.mark.timeout(1800)
async def test_L1b_vertex_upscales_an_upscale_interaction_from_its_saved_clip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """360p -> 720p -> 1080p, the second upscale given the first's interaction id.

    By id, the service refused it after submission: "Exactly one input video
    is required for the upscale task". The tool sends the first upscale's
    saved clip instead. Expected cost: about $0.28 (3s 360p render $0.07,
    360p->720p $0.10, 720p->1080p $0.10).
    """
    async with _live_server(tmp_path, monkeypatch) as live:
        clip, _ = await _render_cheapest(
            live,
            "vertex",
            "A red ball rolls slowly across a wooden table. Static camera.",
        )
        first, sent = await live.call(
            server.upscale_video_omni,
            previous_interaction_id=clip["interaction_id"],
            resolution="720p",
            max_cost_usd=0.5,
        )
        _ok(first, sent)
        second, sent = await live.call(
            server.upscale_video_omni,
            previous_interaction_id=first["interaction_id"],
            resolution="1080p",
            max_cost_usd=0.5,
        )
    _ok(second, sent)
    [request] = sent
    assert "previous_interaction_id" not in request, _redact(request)
    assert _parts(request) == ["video"], _redact(request)
    assert request["generation_config"] == {"video_config": {"task": "upscale"}}
    assert second["rendered_resolution"] == "1080p", second
    assert second["duration_seconds"] == pytest.approx(
        first["duration_seconds"], abs=0.5
    ), second
    assert any("is an upscale" in w for w in second.get("warnings", [])), second


# ---------------------------------------------------------------------------
# L2: duration on a conversational extension turn, per backend (F11)
# ---------------------------------------------------------------------------


async def _extend_twice_by_three_seconds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    # The question is whether duration is accepted, so it is sent whatever
    # the switch is currently set to -- and the turns are rendered whatever
    # L3 settled: every extension turn counts as a large output on Vertex AI,
    # which that switch refuses inline before rendering.
    monkeypatch.setattr(omni, "OMNI_CONVERSATIONAL_EXTENSION_SENDS_DURATION", True)
    monkeypatch.setattr(omni, "OMNI_REFUSE_LARGE_INLINE_ON_VERTEX", False)
    async with _live_server(tmp_path, monkeypatch) as live:
        clip, _ = await _render_cheapest(
            live, backend, "A paper boat drifts down a narrow stream. Slow pan."
        )
        source = clip["duration_seconds"]
        chain, sent = await live.call(
            server.extend_video_omni,
            prompt="Extend this video: the paper boat keeps drifting downstream.",
            previous_interaction_id=clip["interaction_id"],
            omni_backend=backend,
            times=2,
            duration_seconds=3,
            resolution="360p",
            max_cost_usd=0.75,
        )
    advice = (
        "\nThe duration on a conversational turn was not honoured: set "
        "src.omni.OMNI_CONVERSATIONAL_EXTENSION_SENDS_DURATION = False."
    )
    assert "error" not in chain, (
        f"{chain['error']}\n{_l2_verdict(chain['error'])}\nrequests sent: "
        f"{json.dumps(_redact(sent), default=str)}"
    )
    assert len(sent) == 2, sent
    # The chain's own backend, not the server's default: the first Vertex
    # chain to pass reported gemini_api.
    assert chain["backend"] == backend, chain
    for request in sent:
        # Each turn continues an interaction and asks for 3s.
        assert request["previous_interaction_id"], request
        assert request["response_format"][0].get("duration") == "3s", request
    measured = [segment["duration_seconds"] for segment in chain["segments"]]
    print(
        f"  source {source}s; turns measured {measured}; expected about "
        f"{[round(source + 3 * turn, 1) for turn in (1, 2)]}"
    )
    for turn, seconds in enumerate(measured, start=1):
        # A turn returns the assembled clip: the source plus 3s per turn so
        # far. About 10s per turn instead means the duration was ignored.
        assert seconds == pytest.approx(source + 3 * turn, abs=1.0), (
            f"turn {turn} measured {seconds}s, not about {source + 3 * turn:g}s: "
            f"the duration was not honoured.{advice}"
        )


@requires_spend
@requires_vertex
@pytest.mark.timeout(2000)
async def test_L2_vertex_conversational_extension_turns_honour_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F11 on Vertex AI: response_format.duration on previous_interaction_id turns.

    Expected cost: about $0.42 (3s 360p render $0.07; turns bill the
    assembled 6s and 9s at 360p, $0.14 + $0.21), or about $0.91 if the
    duration is ignored. Pass: keep
    OMNI_CONVERSATIONAL_EXTENSION_SENDS_DURATION = True. Fail: set it False.
    """
    await _extend_twice_by_three_seconds(tmp_path, monkeypatch, "vertex")


@requires_spend
@requires_gemini_api
@pytest.mark.timeout(2000)
async def test_L2_gemini_api_conversational_extension_turns_honour_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F11 on the Gemini API: response_format.duration on previous_interaction_id turns.

    Expected cost: about $0.42 (3s 360p render $0.07; turns bill the
    assembled 6s and 9s at 360p, $0.14 + $0.21), or about $0.91 if the
    duration is ignored. Pass: keep
    OMNI_CONVERSATIONAL_EXTENSION_SENDS_DURATION = True. Fail: set it False.
    """
    await _extend_twice_by_three_seconds(tmp_path, monkeypatch, "gemini_api")


# ---------------------------------------------------------------------------
# L3: a large render delivered inline on Vertex AI (D3)
# ---------------------------------------------------------------------------


@requires_spend
@requires_expensive
@requires_vertex
@pytest.mark.timeout(1800)
async def test_L3_vertex_returns_a_4k_upscale_inline_without_output_gcs_uri(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D3: Vertex's upscale page says bytes come back inline with no gcs_uri.

    Expected cost: about $2.10 (10s 720p render $0.70, 720p->4K upscale of
    10s $1.40). Pass: keep OMNI_REFUSE_LARGE_INLINE_ON_VERTEX = False and
    note the size printed. Fail with "carried no video", or a transport or
    size error: set it True.
    """
    # The question is whether inline delivery works, so it is attempted
    # whatever the switch is currently set to.
    monkeypatch.setattr(omni, "OMNI_REFUSE_LARGE_INLINE_ON_VERTEX", False)
    async with _live_server(tmp_path, monkeypatch) as live:
        # 10s at 720p is the longest render the policy does not count as
        # large, so only the upscale is in question.
        clip, sent = await live.call(
            server.generate_video_omni,
            prompt="A lighthouse on a cliff at dusk, waves below. Slow push in.",
            omni_backend="vertex",
            duration_seconds=10,
            resolution="720p",
        )
        _ok(clip, sent)
        upscaled, sent = await live.call(
            server.upscale_video_omni,
            previous_interaction_id=clip["interaction_id"],
            resolution="4K",
            max_cost_usd=2.0,
            # A 4K upscale is the slowest render here.
            timeout_seconds=900,
        )
    assert "error" not in upscaled, (
        f"{upscaled['error']}\n{_l3_verdict(upscaled['error'])}\nrequests "
        f"sent: {json.dumps(_redact(sent), default=str)}"
    )
    [request] = sent
    assert "delivery" not in request["response_format"][0], request
    # The policy said so before rendering, in the response.
    assert any(
        "can exceed the 4 MB inline response limit" in warning
        for warning in upscaled.get("warnings") or []
    ), upscaled.get("warnings")
    size = _file_bytes(upscaled["video_url"])
    assert size, upscaled
    print(f"  4K inline delivery arrived: {size / 1_000_000:.2f} MB")
    assert upscaled["rendered_resolution"] == "4K", upscaled


@requires_spend
@requires_expensive
@requires_vertex
@pytest.mark.timeout(900)
async def test_L3b_vertex_returns_a_4k_render_inline_without_output_gcs_uri(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D3 for a fresh render: L3's evidence is Vertex's upscaling page, but the
    one switch also governs generate_video_omni, edit_video and
    extend_video_omni, so a large generation is checked as well.

    Expected cost: about $0.63 (a 3s 4K render). Pass: inline delivery holds
    for generation too. Fail with "carried no video", or a transport or size
    error: set OMNI_REFUSE_LARGE_INLINE_ON_VERTEX = True, even if L3 passed.
    """
    monkeypatch.setattr(omni, "OMNI_REFUSE_LARGE_INLINE_ON_VERTEX", False)
    async with _live_server(tmp_path, monkeypatch) as live:
        clip, sent = await live.call(
            server.generate_video_omni,
            prompt="A lighthouse on a cliff at dusk, waves below. Slow push in.",
            omni_backend="vertex",
            duration_seconds=3,
            resolution="4K",
        )
    assert "error" not in clip, (
        f"{clip['error']}\n{_l3_verdict(clip['error'])}\nrequests "
        f"sent: {json.dumps(_redact(sent), default=str)}"
    )
    [request] = sent
    assert "delivery" not in request["response_format"][0], request
    assert any(
        "can exceed the 4 MB inline response limit" in warning
        for warning in clip.get("warnings") or []
    ), clip.get("warnings")
    size = _file_bytes(clip["video_url"])
    assert size, clip
    print(f"  4K inline delivery arrived: {size / 1_000_000:.2f} MB")


# ---------------------------------------------------------------------------
# L4: a call naming the retired 1.1 runs on 1.2 (D1)
# ---------------------------------------------------------------------------


@requires_spend
@requires_api
@pytest.mark.timeout(900)
async def test_L4_a_call_naming_1_1_runs_on_1_2_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D1 on whichever backend this server's omni default picks.

    Expected cost: about $0.07 (a 3s 360p render on 1.2).
    """
    async with _live_server(tmp_path, monkeypatch) as live:
        result, sent = await live.call(
            server.generate_video_omni,
            prompt="A kite rises over a grassy hill. Wide shot.",
            omni_model=omni.OMNI_1_1_MODEL,
            duration_seconds=3,
            resolution="360p",
        )
    _ok(result, sent)
    [request] = sent
    assert request["model"] == omni.OMNI_1_2_MODEL, request
    assert result["model"] == omni.OMNI_1_2_MODEL, result
    assert result["requested_model"] == omni.OMNI_1_1_MODEL, result
    assert any(
        "retired" in warning and omni.OMNI_1_2_MODEL in warning
        for warning in result.get("warnings") or []
    ), result.get("warnings")


# ---------------------------------------------------------------------------
# L5-L13: every other request shape new on 1.2, run end to end
# ---------------------------------------------------------------------------

# Each backend that has credentials, as its own test.
_BACKENDS = [
    pytest.param("vertex", marks=requires_vertex, id="vertex"),
    pytest.param("gemini_api", marks=requires_gemini_api, id="gemini_api"),
]

requires_gcs_bucket = pytest.mark.skipif(
    not os.environ.get("OMNI_LIVE_GCS_BUCKET"),
    reason=(
        "needs OMNI_LIVE_GCS_BUCKET: a bucket the Vertex AI credentials can "
        "write to and read from"
    ),
)


def _local(video_url: Any) -> Path:
    assert isinstance(video_url, str) and video_url.startswith("file://"), video_url
    return Path(video_url[len("file://") :])


def _measured(payload: dict[str, Any]) -> float:
    """The length measured from the rendered file, failing if it was not measured."""
    assert str(payload.get("duration_source", "")).startswith("measured"), payload
    return float(payload["duration_seconds"])


def _parts(request: dict[str, Any]) -> list[str]:
    """The input's part types, in the order sent."""
    return [part.get("type") for part in request.get("input") or []]


def _text(request: dict[str, Any]) -> str:
    return " ".join(
        str(part.get("text"))
        for part in request.get("input") or []
        if part.get("type") == "text"
    )


# Two scenes that differ everywhere, so which one a frame resembles is
# unambiguous: a bright day with the sun on the left, a dark night with the
# moon on the right. (sky, ground, orb, side the orb is on)
_DAY = ((135, 206, 235), (34, 139, 34), (255, 223, 0), "left")
_NIGHT = ((12, 16, 52), (6, 38, 14), (235, 235, 250), "right")


def _scene(
    path: Path,
    sky: tuple[int, int, int],
    ground: tuple[int, int, int],
    orb: tuple[int, int, int],
    side: str,
) -> str:
    """A 640x360 PNG of a hill under a sky with one sun or moon, as a file:// URI."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (640, 360), sky)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 230, 640, 360), fill=ground)
    x = 120 if side == "left" else 520
    draw.ellipse((x - 40, 50, x + 40, 130), fill=orb)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    return f"file://{path}"


def _object_image(path: Path, shape: str, color: tuple[int, int, int]) -> str:
    """One plain object on a white ground, as a file:// URI."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (512, 512), (250, 250, 250))
    draw = ImageDraw.Draw(image)
    box = (136, 136, 376, 376)
    if shape == "ball":
        draw.ellipse(box, fill=color)
    else:
        draw.rectangle(box, fill=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    return f"file://{path}"


def _moving_ball_mp4(path: Path, seconds: float = 2.0) -> str:
    """A red ball crossing a white frame left to right, as a file:// URI.

    512x288 keeps both sides multiples of 16, which the encoder would
    otherwise pad, and 2s is inside the 3s reference-video limit.
    """
    import imageio.v3 as iio
    import numpy as np

    fps, width, height = 24, 512, 288
    count = int(seconds * fps)
    rows, cols = np.ogrid[:height, :width]
    frames = []
    for index in range(count):
        frame = np.full((height, width, 3), 245, dtype=np.uint8)
        x = 50 + int((width - 100) * index / (count - 1))
        frame[(rows - height // 2) ** 2 + (cols - x) ** 2 <= 32**2] = (220, 30, 30)
        frames.append(frame)
    buffer = io.BytesIO()
    iio.imwrite(buffer, frames, extension=".mp4", fps=fps)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buffer.getvalue())
    return f"file://{path}"


def _thumbnail(png: bytes) -> list[tuple[int, int, int]]:
    from PIL import Image

    image = Image.open(io.BytesIO(png)).convert("RGB").resize((32, 18))
    pixels: list[tuple[int, int, int]] = []
    # getdata() is deprecated in Pillow 12 in favour of this.
    for pixel in image.get_flattened_data():
        assert isinstance(pixel, tuple)  # RGB: three bands per pixel
        pixels.append((pixel[0], pixel[1], pixel[2]))
    return pixels


def _distance(a: bytes, b: bytes) -> float:
    """Mean per-channel difference of two images, 0 (same) to 255."""
    pa, pb = _thumbnail(a), _thumbnail(b)
    total = sum(abs(x - y) for p, q in zip(pa, pb) for x, y in zip(p, q))
    return total / (len(pa) * 3)


def _brightness(png: bytes) -> float:
    pixels = _thumbnail(png)
    return sum(sum(pixel) for pixel in pixels) / (len(pixels) * 3)


def _frame(video_url: Any, position: str) -> bytes:
    """The first ("start") or last ("end") frame of a saved render, as PNG."""
    from src.video_utils import extract_frame_png

    return extract_frame_png(_local(video_url).read_bytes(), position)  # pyright: ignore[reportArgumentType]


# ---------------------------------------------------------------------------
# L5: extending an uploaded clip (input_video_uri), per backend
# ---------------------------------------------------------------------------


@requires_spend
@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.timeout(1500)
async def test_L5_extends_an_uploaded_clip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    """An uploaded source declares task='extend' beside the video and a duration.

    L2 covered only turns that continue an interaction, which send neither.
    Expected cost: about $0.21 per backend (3s 360p render $0.07; the turn
    bills the assembled 6s, $0.14). The extension is priced as returning the
    whole assembled clip; a turn measuring about 3s instead means the service
    returns only the new footage, and the quote overstates it.
    """
    monkeypatch.setattr(omni, "OMNI_REFUSE_LARGE_INLINE_ON_VERTEX", False)
    async with _live_server(tmp_path, monkeypatch) as live:
        clip, _ = await _render_cheapest(
            live, backend, "A paper boat drifts down a narrow stream. Slow pan."
        )
        chain, sent = await live.call(
            server.extend_video_omni,
            prompt="Extend this video: the paper boat keeps drifting downstream.",
            input_video_uri=clip["video_url"],
            omni_backend=backend,
            times=1,
            duration_seconds=3,
            resolution="360p",
            max_cost_usd=0.5,
        )
    _ok(chain, sent)
    [request] = sent
    assert "previous_interaction_id" not in request, _redact(request)
    assert request["generation_config"] == {"video_config": {"task": "extend"}}
    assert "video" in _parts(request), _redact(request)
    assert request["response_format"][0].get("duration") == "3s", request
    assert "aspect_ratio" not in request["response_format"][0], request
    assert chain["backend"] == backend, chain
    [segment] = chain["segments"]
    source, seconds = clip["duration_seconds"], _measured(segment)
    print(
        f"  source {source}s; extension measured {seconds}s; expected about {source + 3:g}s"
    )
    assert seconds == pytest.approx(source + 3, abs=1.0), (
        f"an uploaded {source}s clip extended by 3s measured {seconds}s. About "
        "3s means the service returned only the new footage: the extension "
        "pricing (omni_extension_priced_lengths) bills it as the assembled clip."
    )


# ---------------------------------------------------------------------------
# L6: upscaling an uploaded file through the tool (Vertex AI)
# ---------------------------------------------------------------------------


@requires_spend
@requires_vertex
@pytest.mark.timeout(1500)
async def test_L6_vertex_upscales_an_uploaded_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stateless upscale: the clip inline, no interaction, no text.

    The request shape was accepted by a raw probe on 2026-11-09; this runs it
    through upscale_video_omni, which measures the source's resolution and
    length to choose the pair and price it. Expected cost: about $0.18 (3s
    360p render $0.07, 360p->720p upscale of 3s $0.11).
    """
    async with _live_server(tmp_path, monkeypatch) as live:
        clip, _ = await _render_cheapest(
            live,
            "vertex",
            "A red ball rolls slowly across a wooden table. Static camera.",
        )
        upscaled, sent = await live.call(
            server.upscale_video_omni,
            input_video_uri=clip["video_url"],
            resolution="720p",
            max_cost_usd=0.5,
        )
    _ok(upscaled, sent)
    [request] = sent
    assert "previous_interaction_id" not in request, _redact(request)
    assert _parts(request) == ["video"], _redact(request)
    assert request["input"][0].get("data"), "the clip should ride inline"
    assert request["generation_config"] == {"video_config": {"task": "upscale"}}
    assert upscaled["rendered_resolution"] == "720p", upscaled
    assert _measured(upscaled) == pytest.approx(clip["duration_seconds"], abs=0.5)
    assert "360p to 720p" in upscaled["cost"]["detail"], upscaled["cost"]


# ---------------------------------------------------------------------------
# L7: one 12s render, longer than any earlier Omni model could make, per backend
# ---------------------------------------------------------------------------


@requires_spend
@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.timeout(900)
async def test_L7_renders_12_seconds_in_one_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    """duration '12s', and the over-10s delivery rule on each backend.

    Expected cost: about $0.28 per backend (12s at 360p). The Gemini API asks
    for a URI over 10s; Vertex AI renders inline with a warning.
    """
    monkeypatch.setattr(omni, "OMNI_REFUSE_LARGE_INLINE_ON_VERTEX", False)
    async with _live_server(tmp_path, monkeypatch) as live:
        clip, sent = await live.call(
            server.generate_video_omni,
            prompt="Waves roll onto a sandy beach at golden hour. Slow pan.",
            omni_backend=backend,
            duration_seconds=12,
            resolution="360p",
        )
    _ok(clip, sent)
    [request] = sent
    response_format = request["response_format"][0]
    assert response_format["duration"] == "12s", request
    if backend == "gemini_api":
        assert response_format.get("delivery") == "uri", request
    else:
        assert "delivery" not in response_format, request
        assert any(
            "can exceed the 4 MB inline response limit" in warning
            for warning in clip.get("warnings") or []
        ), clip.get("warnings")
    assert _measured(clip) == pytest.approx(12, abs=1.0), clip


# ---------------------------------------------------------------------------
# L8: delivery to output_gcs_uri, then an upscale read from gs:// (Vertex AI)
# ---------------------------------------------------------------------------


@requires_spend
@requires_vertex
@requires_gcs_bucket
@pytest.mark.timeout(1500)
async def test_L8_vertex_delivers_to_cloud_storage_and_upscales_from_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The delivery every large-output warning recommends, and a gs:// source.

    Needs OMNI_LIVE_GCS_BUCKET. Leaves two small objects under omni-live/ in
    that bucket. Expected cost: about $0.18 (3s 360p render $0.07, 360p->720p
    upscale $0.11), plus storage.
    """
    bucket = os.environ["OMNI_LIVE_GCS_BUCKET"].removeprefix("gs://").strip("/")
    monkeypatch.setenv("GCS_ALLOWED_BUCKETS", bucket)
    run = uuid.uuid4().hex
    # A prefix, as the tools ask for: Vertex AI writes a new object under it.
    target = f"gs://{bucket}/omni-live/{run}/"
    async with _live_server(tmp_path, monkeypatch) as live:
        clip, sent = await live.call(
            server.generate_video_omni,
            prompt="A red ball rolls slowly across a wooden table. Static camera.",
            omni_backend="vertex",
            duration_seconds=3,
            resolution="360p",
            output_gcs_uri=target,
        )
        _ok(clip, sent)
        [request] = sent
        response_format = request["response_format"][0]
        assert response_format.get("delivery") == "uri", request
        assert response_format.get("gcs_uri") == target, request
        stored = clip["video_url"]
        assert str(stored).startswith(target), clip
        assert not any("ends in a file name" in w for w in clip.get("warnings", []))
        # Downloaded from the bucket and measured, not reported as the request.
        assert clip["duration_source"] == "measured from the rendered video", clip
        assert clip["resolution_source"] == "measured from the rendered video", clip
        upscaled, sent = await live.call(
            server.upscale_video_omni,
            input_video_uri=stored,
            resolution="720p",
            output_gcs_uri=f"gs://{bucket}/omni-live/{run}-720p/",
            # Above the $0.52 a remote source is priced at before it is
            # measured (counted as 15s); the measured 3s bills about $0.11.
            max_cost_usd=0.6,
        )
    _ok(upscaled, sent)
    [request] = sent
    [part] = request["input"]
    assert part.get("type") == "video" and part.get("uri") == stored, part
    assert request["generation_config"] == {"video_config": {"task": "upscale"}}
    assert str(upscaled["video_url"]).startswith(f"gs://{bucket}/"), upscaled
    assert upscaled["duration_source"] == "measured from the rendered video", upscaled
    # A measured length is priced as measured, with no encoder allowance.
    assert "encoder_allowance_seconds" not in upscaled["cost"]["breakdown"], upscaled
    assert upscaled["duration_seconds"] == pytest.approx(
        clip["duration_seconds"], abs=0.05
    )
    print(f"  stored {stored}; upscaled to {upscaled['video_url']}")


# ---------------------------------------------------------------------------
# L9: keyframes. A first frame, and a first/last pair, per backend
# ---------------------------------------------------------------------------


@requires_spend
@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.timeout(900)
async def test_L9a_opens_on_the_first_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    """The render's first frame looks like first_frame_uri, not like a control.

    Expected cost: about $0.07 per backend (3s at 360p).
    """
    day = _scene(tmp_path / "images" / "day.png", *_DAY)
    night = _scene(tmp_path / "images" / "night.png", *_NIGHT)
    async with _live_server(tmp_path, monkeypatch) as live:
        clip, sent = await live.call(
            server.generate_video_omni,
            prompt="Clouds drift over a sunny grassy hill. Static camera.",
            omni_backend=backend,
            first_frame_uri=day,
            duration_seconds=3,
            resolution="360p",
        )
    _ok(clip, sent)
    [request] = sent
    # 1.2 infers the task from the prompt and the media roles, so none is
    # sent; the result reports what it was classified as.
    assert "generation_config" not in request, request["generation_config"]
    assert {
        "task": clip["task"],
        "inferred": clip["task_source"].startswith("inferred"),
    } == {
        "task": "image_to_video",
        "inferred": True,
    }, clip["task_source"]
    assert _parts(request) == ["image", "text"], _redact(request)
    opening = _frame(clip["video_url"], "start")
    to_day = _distance(opening, _local(day).read_bytes())
    to_night = _distance(opening, _local(night).read_bytes())
    print(
        f"  first frame differs from the day image by {to_day:.1f}, night {to_night:.1f}"
    )
    assert to_day < to_night, "the render did not open on first_frame_uri"


@requires_spend
@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.timeout(900)
async def test_L9b_interpolates_between_a_first_and_a_last_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    """Day to night: the first frame looks like the day image, the last like the night.

    The pair is bound with <FIRST_FRAME>@Image1 <LAST_FRAME>@Image2; swapped
    or ignored roles show as the wrong end resembling the wrong image.
    Expected cost: about $0.07 per backend (3s at 360p).
    """
    day = _scene(tmp_path / "images" / "day.png", *_DAY)
    night = _scene(tmp_path / "images" / "night.png", *_NIGHT)
    async with _live_server(tmp_path, monkeypatch) as live:
        clip, sent = await live.call(
            server.generate_video_omni,
            prompt="The sun sets behind the hill and night falls as the moon rises.",
            omni_backend=backend,
            first_frame_uri=day,
            last_frame_uri=night,
            duration_seconds=3,
            resolution="360p",
        )
    _ok(clip, sent)
    [request] = sent
    assert "generation_config" not in request, request["generation_config"]
    assert clip["task"] == "image_to_video", clip
    assert _parts(request) == ["image", "image", "text"], _redact(request)
    assert "<FIRST_FRAME>@Image1" in _text(request), _text(request)
    assert "<LAST_FRAME>@Image2" in _text(request), _text(request)
    day_png, night_png = _local(day).read_bytes(), _local(night).read_bytes()
    opening = _frame(clip["video_url"], "start")
    closing = _frame(clip["video_url"], "end")
    first = (_distance(opening, day_png), _distance(opening, night_png))
    last = (_distance(closing, day_png), _distance(closing, night_png))
    print(
        f"  first frame vs day/night: {first[0]:.1f}/{first[1]:.1f}; "
        f"last frame vs day/night: {last[0]:.1f}/{last[1]:.1f}"
    )
    assert first[0] < first[1], "the render did not open on the first frame"
    assert last[1] < last[0], "the render did not close on the last frame"


@requires_spend
@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.timeout(900)
async def test_L9c_loops_on_one_image_sent_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    """The same image as first and last frame goes on the wire once, and the clip loops.

    Sent as one image part with <FIRST_FRAME>@Image1 <LAST_FRAME>@Image1. If
    the service refuses that, or renders it without closing on the image,
    revert to two parts: drop ``loop_frame`` in src/omni.py's
    generate_video_omni. Expected cost: about $0.07 per backend (3s at 360p).
    """
    day = _scene(tmp_path / "images" / "day.png", *_DAY)
    night_png = _local(_scene(tmp_path / "images" / "night.png", *_NIGHT)).read_bytes()
    async with _live_server(tmp_path, monkeypatch) as live:
        clip, sent = await live.call(
            server.generate_video_omni,
            prompt="Clouds drift across the sky above the hill, and the scene settles back.",
            omni_backend=backend,
            first_frame_uri=day,
            last_frame_uri=day,
            duration_seconds=3,
            resolution="360p",
        )
    _ok(clip, sent)
    [request] = sent
    assert _parts(request) == ["image", "text"], _redact(request)
    assert "<FIRST_FRAME>@Image1 <LAST_FRAME>@Image1" in _text(request), _text(request)
    day_png = _local(day).read_bytes()
    opening = _frame(clip["video_url"], "start")
    closing = _frame(clip["video_url"], "end")
    first = (_distance(opening, day_png), _distance(opening, night_png))
    last = (_distance(closing, day_png), _distance(closing, night_png))
    print(
        f"  first frame vs image/other: {first[0]:.1f}/{first[1]:.1f}; "
        f"last frame vs image/other: {last[0]:.1f}/{last[1]:.1f}; "
        f"first vs last: {_distance(opening, closing):.1f}"
    )
    assert first[0] < first[1], "the render did not open on the image"
    assert last[0] < last[1], "the render did not close on the image"


# ---------------------------------------------------------------------------
# L10: references. Images, and a video, per backend
# ---------------------------------------------------------------------------


@requires_spend
@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.timeout(900)
async def test_L10a_renders_from_two_reference_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    """reference_to_video with each image bound as <IMAGE_REF_n>.

    Expected cost: about $0.07 per backend (3s at 360p).
    """
    ball = _object_image(tmp_path / "images" / "ball.png", "ball", (220, 30, 30))
    cube = _object_image(tmp_path / "images" / "cube.png", "cube", (30, 60, 220))
    async with _live_server(tmp_path, monkeypatch) as live:
        clip, sent = await live.call(
            server.generate_video_omni,
            prompt=(
                "The red ball rolls across a wooden table and bumps into the "
                "blue cube. Static camera."
            ),
            omni_backend=backend,
            reference_image_uris=[ball, cube],
            duration_seconds=3,
            resolution="360p",
        )
    _ok(clip, sent)
    [request] = sent
    # 1.2 infers the task from the prompt and the media roles, so none is
    # sent; the result reports what it was classified as.
    assert "generation_config" not in request, request["generation_config"]
    assert {
        "task": clip["task"],
        "inferred": clip["task_source"].startswith("inferred"),
    } == {
        "task": "reference_to_video",
        "inferred": True,
    }, clip["task_source"]
    assert _parts(request) == ["image", "image", "text"], _redact(request)
    assert "<IMAGE_REF_0>@Image1" in _text(request), _text(request)
    assert "<IMAGE_REF_1>@Image2" in _text(request), _text(request)
    assert _measured(clip) == pytest.approx(3, abs=1.0), clip
    # A reference is a likeness, not a starting frame: print how far the
    # opening is from the plain reference, for a reader to judge.
    opening = _frame(clip["video_url"], "start")
    print(
        f"  first frame differs from the ball reference by "
        f"{_distance(opening, _local(ball).read_bytes()):.1f}"
    )


@requires_spend
@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.timeout(900)
async def test_L10b_renders_from_a_reference_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    """reference_to_video with a 2s clip bound as <VIDEO_REF_0>.

    Expected cost: about $0.07 per backend (3s at 360p).
    """
    reference = _moving_ball_mp4(tmp_path / "inputs" / "ball.mp4")
    async with _live_server(tmp_path, monkeypatch) as live:
        clip, sent = await live.call(
            server.generate_video_omni,
            prompt=(
                "A red ball rolls from left to right across a white tabletop, "
                "moving like the reference video. Static camera."
            ),
            omni_backend=backend,
            reference_video_uris=[reference],
            duration_seconds=3,
            resolution="360p",
        )
    _ok(clip, sent)
    [request] = sent
    # 1.2 infers the task from the prompt and the media roles, so none is
    # sent; the result reports what it was classified as.
    assert "generation_config" not in request, request["generation_config"]
    assert {
        "task": clip["task"],
        "inferred": clip["task_source"].startswith("inferred"),
    } == {
        "task": "reference_to_video",
        "inferred": True,
    }, clip["task_source"]
    assert _parts(request) == ["video", "text"], _redact(request)
    assert "<VIDEO_REF_0>@Video1" in _text(request), _text(request)
    assert _measured(clip) == pytest.approx(3, abs=1.0), clip


# ---------------------------------------------------------------------------
# L11: editing on 1.2. An interaction, and an uploaded clip, per backend
# ---------------------------------------------------------------------------

_NIGHT_EDIT = "Make it night: a dark sky, the scene lit only by moonlight."


@requires_spend
@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.timeout(1500)
async def test_L11a_edits_an_interaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    """edit_video continues the interaction and sends no duration or aspect ratio.

    The service picks the edit's length (up to 15s); live it kept the
    source's 3s on both backends (2026-11-10). Expected cost: about $0.14 per
    backend (3s 360p render $0.07; the 3s edit $0.07), up to $0.42 if it
    runs 15s.
    """
    monkeypatch.setattr(omni, "OMNI_REFUSE_LARGE_INLINE_ON_VERTEX", False)
    async with _live_server(tmp_path, monkeypatch) as live:
        clip, _ = await _render_cheapest(
            live, backend, "A quiet village street on a sunny afternoon. Static camera."
        )
        edited, sent = await live.call(
            server.edit_video,
            previous_interaction_id=clip["interaction_id"],
            prompt=_NIGHT_EDIT,
            omni_backend=backend,
            resolution="360p",
        )
    _ok(edited, sent)
    [request] = sent
    assert request["previous_interaction_id"] == clip["interaction_id"]
    assert "generation_config" not in request, request
    assert "duration" not in request["response_format"][0], request
    assert "aspect_ratio" not in request["response_format"][0], request
    assert edited["rendered_resolution"] == "360p", edited
    seconds = _measured(edited)
    before = _brightness(_frame(clip["video_url"], "end"))
    after = _brightness(_frame(edited["video_url"], "end"))
    print(f"  edit measured {seconds}s; brightness {before:.0f} -> {after:.0f}")
    assert 2.5 <= seconds <= 15.5, edited
    assert after < before, "the night edit did not darken the clip"


@requires_spend
@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.timeout(1500)
async def test_L11b_edits_an_uploaded_clip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    """generate_video_omni(input_video_uri=...) edits the uploaded clip.

    Expected cost: about $0.14 per backend, as L11a.
    """
    monkeypatch.setattr(omni, "OMNI_REFUSE_LARGE_INLINE_ON_VERTEX", False)
    async with _live_server(tmp_path, monkeypatch) as live:
        clip, _ = await _render_cheapest(
            live, backend, "A quiet village street on a sunny afternoon. Static camera."
        )
        edited, sent = await live.call(
            server.generate_video_omni,
            prompt=_NIGHT_EDIT,
            input_video_uri=clip["video_url"],
            omni_backend=backend,
            resolution="360p",
        )
    _ok(edited, sent)
    [request] = sent
    assert "previous_interaction_id" not in request, _redact(request)
    # 1.2 infers the task from the prompt and the media roles, so none is
    # sent; the result reports what it was classified as.
    assert "generation_config" not in request, request["generation_config"]
    assert {
        "task": edited["task"],
        "inferred": edited["task_source"].startswith("inferred"),
    } == {
        "task": "edit",
        "inferred": True,
    }, edited["task_source"]
    assert "video" in _parts(request), _redact(request)
    assert "duration" not in request["response_format"][0], request
    assert "aspect_ratio" not in request["response_format"][0], request
    seconds = _measured(edited)
    before = _brightness(_frame(clip["video_url"], "end"))
    after = _brightness(_frame(edited["video_url"], "end"))
    print(f"  edit measured {seconds}s; brightness {before:.0f} -> {after:.0f}")
    assert 2.5 <= seconds <= 15.5, edited
    assert after < before, "the night edit did not darken the clip"


# ---------------------------------------------------------------------------
# L12: the Veo tools' Omni passes. A draft, and an animatic beat
# ---------------------------------------------------------------------------


@requires_spend
@requires_api
@pytest.mark.timeout(900)
async def test_L12a_a_draft_renders_on_1_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """generate_video(draft=True) renders on gemini-omni-1.2-flash at the draft resolution.

    Expected cost: about $0.07 (3s at 360p).
    """
    async with _live_server(tmp_path, monkeypatch) as live:
        result, sent = await live.call(
            server.generate_video,
            prompt="A kite rises over a grassy hill. Wide shot.",
            model="veo-3.1-generate-001",
            draft=True,
            draft_resolution="360p",
            duration_seconds=3,
        )
    _ok(result, sent)
    [request] = sent
    assert request["model"] == omni.OMNI_1_2_MODEL, request
    assert result["model"] == omni.OMNI_1_2_MODEL, result
    assert result["rendered_resolution"] == "360p", result
    assert _measured(result) == pytest.approx(3, abs=1.0), result


@requires_spend
@requires_api
@pytest.mark.timeout(900)
async def test_L12b_an_animatic_beat_renders_on_1_2_in_portrait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """generate_clip(animatic=True), one beat, at generate_clip's default 9:16.

    Expected cost: about $0.07 (one 3s beat at 360p). Also the one live check
    of a portrait render.
    """
    from src.video_utils import measure_video_dimensions

    async with _live_server(tmp_path, monkeypatch) as live:
        result, sent = await live.call(
            server.generate_clip,
            beats=[
                {
                    "prompt": "A kite rises over a grassy hill. Wide shot.",
                    "duration_seconds": 3,
                }
            ],
            animatic=True,
            animatic_resolution="360p",
        )
    _ok(result, sent)
    [request] = sent
    assert request["model"] == omni.OMNI_1_2_MODEL, request
    assert request["response_format"][0].get("aspect_ratio") == "9:16", request
    beat = next(
        (
            segment
            for segment in result.get("segments") or []
            if segment.get("video_url")
        ),
        None,
    )
    assert beat, result
    size = measure_video_dimensions(_local(beat["video_url"]))
    print(f"  animatic beat measured {size}")
    assert size and size[1] > size[0], f"not portrait: {size}"


@requires_spend
@requires_api
@pytest.mark.timeout(1500)
async def test_L12c_an_animatic_bridges_its_beats_on_1_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """generate_clip(animatic=True, add_bridges=True), two beats and the bridge between them.

    The bridge is a first/last-frame render on 1.2 from beat 1's last frame
    to beat 2's first. Expected cost: about $0.28 (three 4s renders at 360p).
    """
    async with _live_server(tmp_path, monkeypatch) as live:
        result, sent = await live.call(
            server.generate_clip,
            beats=[
                {
                    "prompt": "A kite rises over a grassy hill. Wide shot.",
                    "duration_seconds": 4,
                },
                {
                    "prompt": "The kite drifts over a lake at dusk. Wide shot.",
                    "duration_seconds": 4,
                },
            ],
            animatic=True,
            animatic_resolution="360p",
            add_bridges=True,
        )
    assert not result.get("errors"), result["errors"]
    assert [s["kind"] for s in result["segments"]] == ["beat", "bridge", "beat"], result
    assert len(sent) == 3, [_parts(r) for r in sent]
    bridge_request = sent[2]
    assert bridge_request["model"] == omni.OMNI_1_2_MODEL
    assert _parts(bridge_request)[:2] == ["image", "image"], _redact(bridge_request)
    assert "<LAST_FRAME>@Image2" in _text(bridge_request), _text(bridge_request)
    bridge = result["segments"][1]
    print(f"  bridge measured {bridge['duration_seconds']}s at {bridge['resolution']}")
    assert bridge["duration_source"] == "measured from the rendered video", bridge
    assert bridge["duration_seconds"] == pytest.approx(4, abs=1.0), bridge
    assert bridge["resolution"] == "360p", bridge


# ---------------------------------------------------------------------------
# L13: a render that outlives its call, collected by id
# ---------------------------------------------------------------------------


@requires_spend
@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.timeout(900)
async def test_L13_a_render_that_outlives_its_call_is_collected_by_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    """generate_video_omni times out after the create; get_omni_interaction collects it.

    Settles that an interaction is read back by id on each backend, with its
    video, and that the record the timeout leaves names the backend. Expected
    cost: about $0.07 per backend (3s at 360p); collecting bills nothing.
    """
    async with _live_server(tmp_path, monkeypatch) as live:
        late, sent = await live.call(
            server.generate_video_omni,
            prompt="A paper boat floats down a gutter stream. Close shot.",
            omni_backend=backend,
            duration_seconds=3,
            resolution="360p",
            timeout_seconds=15,
        )
        assert late.get("timed_out") is True, late
        assert late["submission"] == "submitted", late
        interaction_id = late["interaction_id"]
        assert "get_omni_interaction" in late["note"], late
        collected, more = await live.call(
            server.get_omni_interaction, interaction_id=interaction_id, wait_seconds=600
        )
    _ok(collected, more)
    assert more == [], "collecting sent a create"
    assert collected["status"] == "completed", collected
    assert collected["backend"] == backend, collected
    assert collected["model"] == omni.OMNI_1_2_MODEL, collected
    print(
        f"  collected {collected['duration_seconds']}s at {collected['resolution']}, "
        f"{collected['cost']['usd']}"
    )
    assert collected["duration_source"] == "measured from the rendered video"
    assert collected["duration_seconds"] == pytest.approx(3, abs=1.0), collected
    assert collected["resolution"] == "360p", collected
    assert collected.get("sidecar_url"), collected
    # The timeout recorded the render, so it is not priced as unrecorded.
    assert "no record" not in collected["cost"]["detail"], collected["cost"]
