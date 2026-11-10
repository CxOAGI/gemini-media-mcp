"""Tool-level contracts for the Omni 1.2 default.

Each test pins one behaviour a tool has to keep now that gemini-omni-1.2-flash
is the default: continuing a recorded interaction on its own model, measuring
a gs:// source that is sent by URI, applying one inline-delivery policy to a
large output on Vertex AI across all four omni tools (in both positions of its
switch), pricing an upscale from what rendered, and surfacing the bucket
allowlist warning for every gs:// URI a call touches.
"""

import base64
import contextlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from google import genai
from google.genai.types import HttpOptions

import src.__main__ as server
import src.omni as omni
from src.pricing import estimate_omni_upscale_cost
from tests.test_omni_1_2 import context
from tests.test_wire_contract_omni import _FAKE_MP4, _OmniStub, _run, _serve

# A PNG signature is all omni's MIME sniffing reads.
_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
_ONE_FRAME = 1 / 24


def _record(
    videos_dir: Path,
    interaction_id: str,
    model: str,
    *,
    backend: str = "gemini_api",
    duration: float = 6.0,
    resolution: str = "720p",
) -> None:
    """Write the sidecar a render leaves behind, so lookups read a real record."""
    (videos_dir / f"{interaction_id}.json").write_text(
        json.dumps(
            {
                "interaction_id": interaction_id,
                "model": model,
                "backend": backend,
                "duration_seconds": duration,
                "duration_source": "measured from the rendered video",
                "resolution": resolution,
            }
        )
    )


@contextlib.contextmanager
def _stubbed_network(tmp_path: Path, stub: _OmniStub) -> Iterator[MagicMock]:
    """A Gemini-API tool context whose client talks to a local Interactions stub."""
    httpd, base_url = _serve(stub)
    ctx = context(tmp_path)
    ctx.request_context.lifespan_context.client = genai.Client(
        api_key="stub-key-not-real", http_options=HttpOptions(base_url=base_url)
    )
    try:
        yield ctx
    finally:
        httpd.shutdown()
        httpd.server_close()


def _probe(duration: float | None, dimensions: tuple[int, int] | None = None) -> Any:
    """A _probe_media stand-in that reports fixed measurements."""

    async def probe(function: Any, *args: Any) -> Any:
        if function is server.measure_video_metadata_bytes:
            return duration, dimensions
        if function in (
            server.measure_video_duration_bytes,
            server.measure_video_duration,
        ):
            return duration
        if function is server.measure_video_dimensions:
            return dimensions
        return None

    return probe


def _render(**extra: Any) -> AsyncMock:
    return AsyncMock(
        return_value={
            "interaction_id": "rendered",
            "video_url": "gs://bucket/output.mp4",
            **extra,
        }
    )


_CONTINUATIONS = {
    "edit_video": lambda ctx, **kw: server.edit_video(
        ctx, "prior", "make it night", **kw
    ),
    "extend_video_omni": lambda ctx, **kw: server.extend_video_omni(
        ctx, "Extend this video.", previous_interaction_id="prior", **kw
    ),
    "generate_video_omni": lambda ctx, **kw: server.generate_video_omni(
        ctx, "make it night", previous_interaction_id="prior", **kw
    ),
}


# ---------------------------------------------------------------------------
# A continuation stays on the model that created it
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
@pytest.mark.parametrize("tool", sorted(_CONTINUATIONS))
@pytest.mark.parametrize("dry_run", [True, False])
async def test_an_unpinned_continuation_runs_on_the_recorded_model(
    tmp_path: Path, tool: str, dry_run: bool
) -> None:
    """Moving the default to 1.2 refused every default-args continuation of an
    interaction recorded on another model; the quote and the render must both
    land on the recorded one. The preview is the one other model left (the
    suite's calendar is before its sunset), and it cannot extend -- so an
    extension of its interaction is refused for THAT, which is only reachable
    by resolving to the recorded model rather than to the default."""
    stub = _OmniStub()
    with _stubbed_network(tmp_path, stub) as ctx:
        _record(tmp_path / "videos", "prior", omni.OMNI_PREVIEW_MODEL)
        result = json.loads(await _CONTINUATIONS[tool](ctx, dry_run=dry_run))
    if tool == "extend_video_omni":
        assert "extend task" in result["error"], result
        assert "cannot change models" not in result["error"]
        assert not stub.requests
        return
    assert "error" not in result, result
    assert result["model"] == omni.OMNI_PREVIEW_MODEL
    if dry_run:
        assert not stub.requests
    else:
        assert stub.created()["model"] == omni.OMNI_PREVIEW_MODEL
        assert stub.created()["previous_interaction_id"] == "prior"


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", sorted(_CONTINUATIONS))
@pytest.mark.parametrize("dry_run", [True, False])
async def test_an_explicit_other_model_is_refused_with_the_way_out(
    tmp_path: Path, tool: str, dry_run: bool
) -> None:
    ctx = context(tmp_path)
    _record(tmp_path / "videos", "prior", omni.OMNI_PREVIEW_MODEL)
    result = json.loads(
        await _CONTINUATIONS[tool](ctx, omni_model=omni.OMNI_1_2_MODEL, dry_run=dry_run)
    )
    error = result["error"]
    assert "cannot change models" in error
    assert f"Leave omni_model unset (or pass '{omni.OMNI_PREVIEW_MODEL}')" in error
    # extend's old text said only one model can extend, telling a caller
    # holding that model's clip to re-render it on the same model.
    assert "can extend" not in error
    ctx.request_context.lifespan_context.client.interactions.create.assert_not_called()


# ---------------------------------------------------------------------------
# gs:// sources on Vertex
# ---------------------------------------------------------------------------


_SOURCE_TOOLS = {
    "generate_video_omni": lambda ctx, **kw: server.generate_video_omni(
        ctx, "make it night", **kw
    ),
    "extend_video_omni": lambda ctx, **kw: server.extend_video_omni(
        ctx, "Extend this video.", **kw
    ),
}


@pytest.mark.asyncio
async def test_a_gcs_source_on_vertex_is_fetched_and_inlined_for_the_preview(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The preview keeps the wire shape it was verified with (inline bytes);
    only 1.2 reads a gs:// source by URI."""
    ctx = context(tmp_path, vertex=True)
    fetched = AsyncMock(return_value=b"source-bytes")
    monkeypatch.setattr(server, "_fetch_omni_media", fetched)
    render = _render(duration_seconds=10.0)
    monkeypatch.setattr(server, "_omni_generate_and_manifest", render)
    monkeypatch.setattr(server, "_probe_media", _probe(8.0))
    result = json.loads(
        await _SOURCE_TOOLS["generate_video_omni"](
            ctx,
            omni_model=omni.OMNI_PREVIEW_MODEL,
            input_video_uri="gs://bucket/src.mp4",
        )
    )
    assert "error" not in result, result
    assert render.call_args.kwargs["input_video_bytes"] == b"source-bytes"
    assert render.call_args.kwargs["input_video_uri"] is None


@pytest.mark.asyncio
async def test_a_1_2_gcs_extension_source_is_measured_but_sent_by_uri(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unmeasured, every turn priced at the 40s ceiling: this $2 cap refused
    a chain that costs $1.74 (15s source + 10s = 25s at $0.069504/s)."""
    ctx = context(tmp_path, vertex=True)
    downloaded = AsyncMock(return_value=b"source-bytes")
    monkeypatch.setattr(server, "fetch", downloaded)
    monkeypatch.setattr(
        server,
        "_fetch_omni_media",
        AsyncMock(side_effect=AssertionError("a URI source is not fetched to send")),
    )
    monkeypatch.setattr(server, "_probe_media", _probe(15.0))
    render = _render(duration_seconds=25.0, model=omni.OMNI_1_2_MODEL)
    monkeypatch.setattr(server, "_omni_generate_and_manifest", render)

    result = json.loads(
        await server.extend_video_omni(
            ctx,
            "Extend this video.",
            input_video_uri="gs://bucket/src.mp4",
            max_cost_usd=2.0,
        )
    )
    assert "error" not in result, result
    assert downloaded.await_args is not None
    assert downloaded.await_args.args[0] == "gs://bucket/src.mp4"
    kwargs = render.call_args.kwargs
    assert kwargs["input_video_uri"] == "gs://bucket/src.mp4"
    assert kwargs["input_video_bytes"] is None
    assert kwargs["source_duration_seconds"] == 15.0


@pytest.mark.asyncio
async def test_a_1_2_gcs_edit_source_gets_its_15s_check_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = context(tmp_path, vertex=True)
    monkeypatch.setattr(server, "fetch", AsyncMock(return_value=b"source-bytes"))
    monkeypatch.setattr(server, "_probe_media", _probe(16.0))
    render = _render()
    monkeypatch.setattr(server, "_omni_generate_and_manifest", render)
    result = json.loads(
        await server.generate_video_omni(
            ctx, "make it night", input_video_uri="gs://bucket/src.mp4"
        )
    )
    assert "15s or shorter" in result["error"]
    render.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_unreadable_gcs_source_proceeds_by_uri_and_says_it_was_not_measured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = context(tmp_path, vertex=True)
    monkeypatch.setattr(server, "fetch", AsyncMock(return_value=None))
    render = _render(duration_seconds=40.0, model=omni.OMNI_1_2_MODEL)
    monkeypatch.setattr(server, "_omni_generate_and_manifest", render)
    result = json.loads(
        await server.extend_video_omni(
            ctx, "Extend this video.", input_video_uri="gs://bucket/src.mp4"
        )
    )
    assert "error" not in result, result
    assert render.call_args.kwargs["input_video_uri"] == "gs://bucket/src.mp4"
    assert render.call_args.kwargs["source_duration_seconds"] is None
    assert any("could not be downloaded to measure" in w for w in result["warnings"])


# ---------------------------------------------------------------------------
# Upscale delivery and pricing
# ---------------------------------------------------------------------------


_LIMIT = "can exceed the 4 MB inline response limit"


def _delivery_warnings(payload: dict[str, Any]) -> list[str]:
    """The inline-delivery policy's warnings in a quote or a response."""
    return [w for w in payload.get("warnings") or [] if _LIMIT in w]


def _capital(phrase: str) -> str:
    return phrase[0].upper() + phrase[1:]


@pytest.fixture(params=[False, True], ids=["warns", "refuses"])
def refuse_large_inline(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> bool:
    """Both positions of the policy's one switch."""
    monkeypatch.setattr(omni, "OMNI_REFUSE_LARGE_INLINE_ON_VERTEX", request.param)
    return request.param


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "target", [None, "1080p", "4K"], ids=["default", "1080p", "4K"]
)
@pytest.mark.parametrize("dry_run", [True, False])
async def test_a_large_upscale_follows_the_inline_delivery_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    refuse_large_inline: bool,
    target: str | None,
    dry_run: bool,
) -> None:
    """It was refused outright -- even at the tool's own default
    resolution='1080p' -- although Vertex's "Upscale videos" page says a
    request with no gcs_uri gets its bytes back in the response."""
    ctx = context(tmp_path, vertex=True)
    monkeypatch.setattr(server, "fetch", AsyncMock(return_value=b"source-bytes"))
    monkeypatch.setattr(server, "_probe_media", _probe(6.0, (1280, 720)))
    render = _render()
    monkeypatch.setattr(server, "_omni_generate_and_manifest", render)
    kwargs: dict[str, Any] = dict(
        input_video_uri="gs://bucket/draft.mp4", source_resolution="720p"
    )
    if target:
        kwargs["resolution"] = target
    shown = target or "1080p"
    result = json.loads(await server.upscale_video_omni(ctx, **kwargs, dry_run=dry_run))
    if refuse_large_inline:
        assert result["error"].startswith(
            "Pass output_gcs_uri (a prefix such as gs://bucket/renders/)"
        )
        assert f"a {shown} render {_LIMIT}" in result["error"]
        assert "Nothing was rendered or billed." in result["error"]
        render.assert_not_awaited()
    else:
        assert "error" not in result, result
        [warning] = _delivery_warnings(result)
        assert warning.startswith(f"A {shown} render {_LIMIT}")
        assert "Pass output_gcs_uri (a prefix such as gs://bucket/renders/)" in warning
        ctx.warning.assert_any_await(warning)
        if not dry_run:
            assert render.call_args.kwargs["output_gcs_uri"] is None
            assert render.call_args.kwargs["resolution"] == shown

    # With a destination, both positions run and have nothing to say.
    monkeypatch.setattr(server, "_omni_generate_and_manifest", _render())
    accepted = json.loads(
        await server.upscale_video_omni(
            ctx, **kwargs, output_gcs_uri="gs://bucket/up.mp4", dry_run=dry_run
        )
    )
    assert "error" not in accepted, accepted
    assert not _delivery_warnings(accepted)


@pytest.mark.asyncio
@pytest.mark.parametrize("seconds, large", [(6.0, False), (10.0, False), (12.0, True)])
@pytest.mark.parametrize("dry_run", [True, False])
async def test_a_720p_upscale_is_large_only_past_ten_seconds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    refuse_large_inline: bool,
    seconds: float,
    large: bool,
    dry_run: bool,
) -> None:
    ctx = context(tmp_path, vertex=True)
    monkeypatch.setattr(server, "_omni_generate_and_manifest", _render())
    _record(
        tmp_path / "videos",
        "prior",
        omni.OMNI_1_2_MODEL,
        backend="vertex",
        duration=seconds,
        resolution="360p",
    )
    result = json.loads(
        await server.upscale_video_omni(
            ctx, previous_interaction_id="prior", resolution="720p", dry_run=dry_run
        )
    )
    if not large:
        assert "error" not in result, result
        assert not _delivery_warnings(result)
    elif refuse_large_inline:
        assert f"a {seconds:g}s render {_LIMIT}" in result["error"]
    else:
        [warning] = _delivery_warnings(result)
        assert warning.startswith(f"A {seconds:g}s render {_LIMIT}")


@pytest.mark.asyncio
async def test_an_unmeasured_720p_upscale_is_judged_at_its_fifteen_second_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, refuse_large_inline: bool
) -> None:
    """A source the real run cannot download to measure is taken at the 15s
    bound the estimate prices, and the message does not state it as fact."""
    ctx = context(tmp_path, vertex=True)
    monkeypatch.setattr(server, "fetch", AsyncMock(return_value=None))
    render = _render()
    monkeypatch.setattr(server, "_omni_generate_and_manifest", render)
    result = json.loads(
        await server.upscale_video_omni(
            ctx,
            input_video_uri="gs://bucket/draft.mp4",
            source_resolution="360p",
            resolution="720p",
        )
    )
    phrase = f"a render that could run up to 15s {_LIMIT}"
    if refuse_large_inline:
        assert phrase in result["error"]
        render.assert_not_awaited()
    else:
        [warning] = _delivery_warnings(result)
        assert warning.startswith(_capital(phrase))
        assert render.call_args.kwargs["source_duration_seconds"] is None


@pytest.mark.asyncio
async def test_a_remote_720p_upscale_is_judged_at_its_bound_until_it_is_measured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, refuse_large_inline: bool
) -> None:
    """A dry run does not download a remote source, so it judges the 15s
    bound it quotes; the real run downloads and measures it, and a 6s source
    is small. One policy for both -- no separate remote-source branch."""
    ctx = context(tmp_path, vertex=True)
    downloaded = AsyncMock(return_value=b"short source")
    monkeypatch.setattr(server, "fetch", downloaded)
    monkeypatch.setattr(server, "_probe_media", _probe(6.0, (640, 360)))
    render = _render()
    monkeypatch.setattr(server, "_omni_generate_and_manifest", render)
    kwargs = dict(
        input_video_uri="gs://bucket/draft.mp4",
        source_resolution="360p",
        resolution="720p",
    )
    quote = json.loads(await server.upscale_video_omni(ctx, **kwargs, dry_run=True))
    downloaded.assert_not_awaited()
    phrase = f"a render that could run up to 15s {_LIMIT}"
    if refuse_large_inline:
        assert phrase in quote["error"]
    else:
        assert quote["duration_seconds"] == 15
        [warning] = _delivery_warnings(quote)
        assert warning.startswith(_capital(phrase))

    result = json.loads(await server.upscale_video_omni(ctx, **kwargs))
    assert "error" not in result, result
    assert not _delivery_warnings(result)
    assert render.call_args.kwargs["source_duration_seconds"] == 6.0


# ---------------------------------------------------------------------------
# One inline-delivery policy for all four omni tools
# ---------------------------------------------------------------------------

# Each tool asked, on Vertex with no output_gcs_uri, for an output past the
# 4 MB inline limit, beside what the policy says made it large.
_LARGE_ON_VERTEX = {
    "generate_video_omni-4K": (
        lambda ctx, **kw: server.generate_video_omni(
            ctx, "a cat", resolution="4K", **kw
        ),
        "a 4K render",
    ),
    "generate_video_omni-12s": (
        lambda ctx, **kw: server.generate_video_omni(
            ctx, "a cat", duration_seconds=12, **kw
        ),
        "a 12s render",
    ),
    "generate_video_omni-edit": (
        lambda ctx, **kw: server.generate_video_omni(
            ctx, "make it night", previous_interaction_id="prior", **kw
        ),
        "a render that could run up to 15s",
    ),
    "edit_video": (
        lambda ctx, **kw: server.edit_video(ctx, "prior", "make it night", **kw),
        "a render that could run up to 15s",
    ),
    "extend_video_omni": (
        lambda ctx, **kw: server.extend_video_omni(
            ctx, "Extend this video.", previous_interaction_id="prior", **kw
        ),
        "an extension turn, which returns the whole assembled clip,",
    ),
    "upscale_video_omni": (
        lambda ctx, **kw: server.upscale_video_omni(
            ctx, previous_interaction_id="prior", **kw
        ),
        "a 1080p render",
    ),
}


def _vertex_with_prior(tmp_path: Path) -> MagicMock:
    ctx = context(tmp_path, vertex=True)
    _record(
        tmp_path / "videos",
        "prior",
        omni.OMNI_1_2_MODEL,
        backend="vertex",
        duration=6.0,
        resolution="360p",
    )
    return ctx


@pytest.mark.asyncio
@pytest.mark.parametrize("call", sorted(_LARGE_ON_VERTEX))
async def test_every_omni_tool_applies_the_one_inline_delivery_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    refuse_large_inline: bool,
    call: str,
) -> None:
    """Warned (by default) or refused (switch flipped) BEFORE anything
    renders: the dry run says exactly what the real run says. Before this,
    upscale refused, the other three warned only after the render had been
    billed, and their quotes said nothing."""
    tool, phrase = _LARGE_ON_VERTEX[call]
    ctx = _vertex_with_prior(tmp_path)
    render = _render()
    monkeypatch.setattr(server, "_omni_generate_and_manifest", render)
    quote = json.loads(await tool(ctx, dry_run=True))
    result = json.loads(await tool(ctx))
    if refuse_large_inline:
        assert quote["error"] == result["error"]
        assert f"{phrase} {_LIMIT}" in quote["error"]
        assert "output_gcs_uri" in quote["error"]
        assert "Nothing was rendered or billed." in quote["error"]
        render.assert_not_awaited()
        return
    for payload in (quote, result):
        assert "error" not in payload, payload
        [warning] = _delivery_warnings(payload)
        assert warning.startswith(f"{_capital(phrase)} {_LIMIT}")
        assert "Pass output_gcs_uri (a prefix such as gs://bucket/renders/)" in warning
        ctx.warning.assert_any_await(warning)
    assert _delivery_warnings(quote) == _delivery_warnings(result)
    render.assert_awaited()
    for rendered in render.call_args_list:
        assert rendered.kwargs["output_gcs_uri"] is None


# ---------------------------------------------------------------------------
# A 4K output that will likely outlast the call's deadline is warned about
# ---------------------------------------------------------------------------

_SLOW = "is likely to take about"


def _slow_tier_warnings(payload: dict[str, Any]) -> list[str]:
    return [w for w in payload.get("warnings") or [] if _SLOW in w]


@pytest.mark.parametrize(
    "kind, seconds, timeout, warns",
    [
        # Measured: a 3.03s 4K render took 153.8s, inside the 210s default.
        pytest.param("render", 3, 210, False, id="3s-render-fits"),
        pytest.param("render", 6, 210, True, id="6s-render-does-not"),
        pytest.param("render", 6, 900, False, id="6s-render-with-a-raised-deadline"),
        # Measured: a 10.03s 720p->4K upscale took 210.0s, the default itself.
        pytest.param("upscale", 3, 210, False, id="3s-upscale-fits"),
        pytest.param("upscale", 10, 210, True, id="10s-upscale-does-not"),
        pytest.param("upscale", None, 210, True, id="unknown-source-counts-as-15s"),
        pytest.param("extend", 6, 210, True, id="extension-turn"),
    ],
)
def test_the_4k_deadline_warning_follows_the_measured_times(
    kind: str, seconds: float | None, timeout: float, warns: bool
) -> None:
    warning = omni.omni_slow_tier_warning(
        resolution="4K", kind=kind, output_seconds=seconds, timeout_seconds=timeout
    )
    assert (warning is not None) is warns, warning
    if warning:
        assert "MEASURED" in warning
        assert f"deadline is {timeout:g}s" in warning
        assert "Raise timeout_seconds past" in warning


@pytest.mark.parametrize("resolution", ["360p", "720p", "1080p", None])
def test_nothing_below_4k_gets_the_deadline_warning(resolution: str | None) -> None:
    """Nothing below 4K has come near the deadline: a 10s 720p render took 46s."""
    assert (
        omni.omni_slow_tier_warning(
            resolution=resolution,
            kind="render",
            output_seconds=15,
            timeout_seconds=210,
        )
        is None
    )


_SLOW_ON_VERTEX = {
    "generate_video_omni": (
        lambda ctx, **kw: server.generate_video_omni(
            ctx, "a cat", resolution="4K", duration_seconds=6, **kw
        ),
        "A 6s 4K render",
    ),
    "generate_video_omni-edit": (
        lambda ctx, **kw: server.generate_video_omni(
            ctx, "make it night", previous_interaction_id="prior", resolution="4K", **kw
        ),
        "A 4K render that could run up to 15s",
    ),
    "edit_video": (
        lambda ctx, **kw: server.edit_video(
            ctx, "prior", "make it night", resolution="4K", **kw
        ),
        "A 4K render that could run up to 15s",
    ),
    "extend_video_omni": (
        lambda ctx, **kw: server.extend_video_omni(
            ctx,
            "Extend this video.",
            previous_interaction_id="prior",
            resolution="4K",
            duration_seconds=3,
            **kw,
        ),
        "A 4K extension turn, which renders the whole assembled clip (at least 13s",
    ),
    "upscale_video_omni": (
        lambda ctx, **kw: server.upscale_video_omni(
            ctx, previous_interaction_id="prior", resolution="4K", **kw
        ),
        "A 4K upscale of a 10s source",
    ),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("call", sorted(_SLOW_ON_VERTEX))
async def test_every_omni_tool_warns_before_a_4k_output_outlasts_its_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, call: str
) -> None:
    """Said on the quote and before the render, and not said once the deadline fits.

    Live L3 (2026-11-09): a 10s 720p-to-4K upscale took 210.0s, the default
    deadline itself. Past the deadline the render is billed and its video
    cannot be fetched here, so the caller has to know before paying.
    """
    tool, phrase = _SLOW_ON_VERTEX[call]
    ctx = context(tmp_path, vertex=True)
    _record(
        tmp_path / "videos",
        "prior",
        omni.OMNI_1_2_MODEL,
        backend="vertex",
        duration=10.0,
        resolution="720p",
    )
    render = _render()
    monkeypatch.setattr(server, "_omni_generate_and_manifest", render)
    quote = json.loads(await tool(ctx, dry_run=True))
    result = json.loads(await tool(ctx))
    for payload in (quote, result):
        assert "error" not in payload, payload
        [warning] = _slow_tier_warnings(payload)
        assert warning.startswith(phrase), warning
        assert "deadline is 210s" in warning
        ctx.warning.assert_any_await(warning)
    render.assert_awaited()
    raised = json.loads(await tool(ctx, dry_run=True, timeout_seconds=1200))
    assert "error" not in raised, raised
    assert not _slow_tier_warnings(raised), raised


@pytest.mark.asyncio
@pytest.mark.parametrize("dry_run", [True, False])
async def test_a_4k_draft_warns_before_it_outlasts_its_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dry_run: bool
) -> None:
    """generate_video(draft=True) renders on Omni under the call's own deadline."""
    ctx = context(tmp_path, vertex=True)
    monkeypatch.setattr(server, "_omni_generate_and_manifest", _render())
    payload = json.loads(
        await server.generate_video(
            ctx,
            "a cat",
            model="veo-3.1-generate-001",
            draft=True,
            draft_resolution="4K",
            duration_seconds=6,
            dry_run=dry_run,
        )
    )
    assert "error" not in payload, payload
    [warning] = _slow_tier_warnings(payload)
    assert warning.startswith("A 6s 4K render"), warning


@pytest.mark.asyncio
@pytest.mark.parametrize("call", sorted(_LARGE_ON_VERTEX))
@pytest.mark.parametrize("dry_run", [True, False])
async def test_an_output_gcs_uri_satisfies_the_policy_in_both_positions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    refuse_large_inline: bool,
    call: str,
    dry_run: bool,
) -> None:
    tool, _ = _LARGE_ON_VERTEX[call]
    ctx = _vertex_with_prior(tmp_path)
    monkeypatch.setattr(server, "_omni_generate_and_manifest", _render())
    payload = json.loads(
        await tool(ctx, output_gcs_uri="gs://bucket/out.mp4", dry_run=dry_run)
    )
    assert "error" not in payload, payload
    assert not _delivery_warnings(payload)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call",
    sorted(name for name in _LARGE_ON_VERTEX if name != "upscale_video_omni"),
)
@pytest.mark.parametrize("dry_run", [True, False])
async def test_the_gemini_api_path_is_not_the_policys_business(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    refuse_large_inline: bool,
    call: str,
    dry_run: bool,
) -> None:
    """The Gemini API delivers a large render as a Google-hosted URI
    (delivery='uri'), so there is nothing to warn about or refuse up front."""
    tool, _ = _LARGE_ON_VERTEX[call]
    ctx = context(tmp_path)
    _record(tmp_path / "videos", "prior", omni.OMNI_1_2_MODEL, duration=6.0)
    monkeypatch.setattr(server, "_omni_generate_and_manifest", _render())
    payload = json.loads(await tool(ctx, dry_run=dry_run))
    assert "error" not in payload, payload
    assert not _delivery_warnings(payload)


@pytest.mark.asyncio
@pytest.mark.parametrize("dry_run", [True, False])
async def test_a_small_vertex_render_is_not_the_policys_business(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    refuse_large_inline: bool,
    dry_run: bool,
) -> None:
    ctx = context(tmp_path, vertex=True)
    monkeypatch.setattr(server, "_omni_generate_and_manifest", _render())
    payload = json.loads(
        await server.generate_video_omni(
            ctx, "a cat", resolution="720p", duration_seconds=10, dry_run=dry_run
        )
    )
    assert "error" not in payload, payload
    assert not _delivery_warnings(payload)


@pytest.mark.asyncio
@pytest.mark.parametrize("dry_run", [True, False])
async def test_a_chain_delivers_only_its_last_turn_to_output_gcs_uri_and_says_so(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    refuse_large_inline: bool,
    dry_run: bool,
) -> None:
    """output_gcs_uri rides on a chain's final turn, so the turns before it
    come back inline even when one is given -- the policy applies to them."""
    ctx = context(tmp_path, vertex=True)
    _record(
        tmp_path / "videos",
        "prior",
        omni.OMNI_1_2_MODEL,
        backend="vertex",
        duration=3.0,
    )
    render = _render()
    monkeypatch.setattr(server, "_omni_generate_and_manifest", render)

    async def extend(times: int) -> dict[str, Any]:
        return json.loads(
            await server.extend_video_omni(
                ctx,
                "Extend this video.",
                previous_interaction_id="prior",
                times=times,
                output_gcs_uri="gs://bucket/out.mp4",
                dry_run=dry_run,
            )
        )

    chain = await extend(2)
    if refuse_large_inline:
        assert (
            "extend one turn per call (times=1) with output_gcs_uri" in chain["error"]
        )
        assert "Nothing was rendered or billed." in chain["error"]
        render.assert_not_awaited()
    else:
        [warning] = _delivery_warnings(chain)
        assert warning.startswith(
            "Every turn but the last of this extension comes back inline"
        )
        assert "extend one turn per call (times=1) with output_gcs_uri" in warning
    # One turn per call is the way to deliver every turn to the bucket.
    single = await extend(1)
    assert "error" not in single, single
    assert not _delivery_warnings(single)


def _vertex_impl(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """The real render helper on a Vertex client, with only the impl stubbed."""
    vertex_client = MagicMock()
    vertex_client._api_client.vertexai = True
    monkeypatch.setattr(server, "_get_omni_vertex_global_client", lambda: vertex_client)
    turns = iter(range(1, 100))

    async def impl(**kwargs: Any) -> dict[str, Any]:
        turn = next(turns)
        return {
            "message": "ok",
            "video_url": kwargs.get("output_gcs_uri") or f"gs://bucket/turn{turn}.mp4",
            "interaction_id": f"i-{turn}",
            "model": omni.OMNI_1_2_MODEL,
            "task": kwargs.get("task") or "text_to_video",
            "duration_seconds": None,
            "aspect_ratio": None,
            "resolution": kwargs.get("resolution"),
            "rendered_resolution": kwargs.get("resolution") or "720p",
        }

    mock = AsyncMock(side_effect=impl)
    monkeypatch.setattr(server, "generate_video_omni_impl", mock)
    return mock


@pytest.mark.asyncio
async def test_a_turn_the_tool_already_judged_is_not_judged_again_by_the_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The chain's first turn goes out with no output_gcs_uri. Judged again
    per turn, it drew a second, contradictory warning ("with no
    output_gcs_uri it is requested inline") beside the chain's own, which says
    output_gcs_uri was given and receives the final turn."""
    impl = _vertex_impl(monkeypatch)
    ctx = context(tmp_path, vertex=True)
    _record(
        tmp_path / "videos",
        "prior",
        omni.OMNI_1_2_MODEL,
        backend="vertex",
        duration=3.0,
    )
    result = json.loads(
        await server.extend_video_omni(
            ctx,
            "Extend this video.",
            previous_interaction_id="prior",
            times=2,
            output_gcs_uri="gs://bucket/out.mp4",
        )
    )
    assert "error" not in result, result
    assert impl.await_count == 2
    [warning] = _delivery_warnings(result)
    assert warning.startswith("Every turn but the last of this extension")


_NO_DESTINATION = (
    "ask for 720p or below, or render it with generate_video_omni, whose "
    "output_gcs_uri delivers to Cloud Storage"
)


@pytest.mark.asyncio
@pytest.mark.parametrize("dry_run", [True, False])
async def test_a_large_draft_on_vertex_follows_the_policy_with_advice_it_can_take(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    refuse_large_inline: bool,
    dry_run: bool,
) -> None:
    """A draft renders on omni, so the policy applies -- but a draft ignores
    output_gcs_uri, so advice to pass one would be advice it cannot take."""
    impl = _vertex_impl(monkeypatch)
    result = json.loads(
        await server.generate_video(
            context(tmp_path, vertex=True),
            "a lighthouse at dusk",
            model="veo-3.1-fast-generate-001",
            draft=True,
            draft_resolution="4K",
            duration_seconds=6,
            dry_run=dry_run,
        )
    )
    if refuse_large_inline:
        assert result["error"].startswith(f"A 4K render {_LIMIT}")
        assert _NO_DESTINATION in result["error"]
        assert "Nothing was rendered or billed." in result["error"]
    else:
        assert "error" not in result, result
        [warning] = _delivery_warnings(result)
        assert warning.startswith(f"A 4K render {_LIMIT}")
        assert _NO_DESTINATION in warning
        assert "Pass output_gcs_uri" not in warning
    assert impl.await_count == (0 if dry_run or refuse_large_inline else 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("dry_run", [True, False])
async def test_a_large_animatic_on_vertex_is_judged_before_any_beat_renders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    refuse_large_inline: bool,
    dry_run: bool,
) -> None:
    """Every beat is judged up front: refused per beat as it came up, a
    later beat's refusal would land after the earlier ones had billed. Large
    by resolution: a beat runs as long as its Veo delivery, at most 8s, so no
    animatic beat passes 10s any more."""
    impl = _vertex_impl(monkeypatch)
    result = json.loads(
        await server.generate_clip(
            context(tmp_path, vertex=True),
            beats=[
                {"prompt": "beat 1", "duration_seconds": 4},
                {"prompt": "beat 2", "duration_seconds": 8},
            ],
            animatic=True,
            animatic_resolution="1080p",
            dry_run=dry_run,
        )
    )
    if refuse_large_inline:
        assert result["error"].startswith(f"A 1080p render {_LIMIT}")
        assert _NO_DESTINATION in result["error"]
        assert impl.await_count == 0
    else:
        assert "error" not in result, result
        [warning] = _delivery_warnings(result)
        assert warning.startswith(f"A 1080p render {_LIMIT}")
        assert impl.await_count == (0 if dry_run else 2)


@pytest.mark.timeout(30.0)
@pytest.mark.parametrize(
    "request_kwargs, reason",
    [
        pytest.param(
            {"resolution": "4K", "duration_seconds": 4},
            "because a 4K render can exceed",
            id="4k",
        ),
        pytest.param(
            {
                "task": "extend",
                "resolution": "360p",
                "duration_seconds": 3,
                "input_video_bytes": _FAKE_MP4,
            },
            "because an extension turn, which returns the whole assembled clip, "
            "can exceed",
            id="extension",
        ),
        pytest.param(
            {"resolution": "360p", "duration_seconds": 12},
            "because a 12s render can exceed",
            id="long-render",
        ),
        # An edit sends no duration, so the 6s default said nothing about
        # its output: it was requested inline though it can run 15s.
        pytest.param(
            {"resolution": "720p", "input_video_bytes": _FAKE_MP4},
            "because a render that could run up to 15s can exceed",
            id="edit-at-the-default-duration",
        ),
        pytest.param(
            {
                "resolution": "720p",
                "previous_interaction_id": "prior",
                "duration_seconds": 3,
            },
            "because a render that could run up to 15s can exceed",
            id="conversational-edit",
        ),
        pytest.param(
            {
                "task": "upscale",
                "resolution": "720p",
                "input_video_bytes": _FAKE_MP4,
                "source_duration_seconds": 12.0,
                "duration_seconds": None,
            },
            "because a 12s render can exceed",
            id="upscale-of-a-12s-source",
        ),
    ],
)
def test_a_gemini_api_uri_delivery_says_why_it_was_asked_for_not_that_it_was_large(
    tmp_path: Path, request_kwargs: dict[str, Any], reason: str
) -> None:
    """URI delivery is chosen before the render, by rule, not by the file's size.

    Live L2 on the Gemini API (2026-11-09) delivered a 9s 360p extension of
    2.64 MB by URI under a warning that "This render exceeds the 4 MB inline
    response limit". The warning names the rule that asked for a URI instead,
    judged on the length the output can reach.
    """
    stub = _OmniStub()
    result = _run(
        stub,
        tmp_path,
        "Keep going.",
        allow_uri_delivery=True,
        **request_kwargs,
    )
    assert "error" not in result, result
    assert stub.response_format().get("delivery") == "uri", stub.response_format()
    [note] = [w for w in result.get("warnings", []) if "Google-hosted URI" in w]
    assert note.startswith(
        "This render was requested as a Google-hosted URI rather than inline, " + reason
    ), note
    assert "exceeds" not in note


@pytest.mark.timeout(30.0)
@pytest.mark.parametrize(
    "request_kwargs",
    [
        pytest.param({"resolution": "720p", "duration_seconds": 6}, id="6s-720p"),
        pytest.param({"resolution": "360p", "duration_seconds": 10}, id="10s-360p"),
        pytest.param(
            {
                "task": "upscale",
                "resolution": "720p",
                "input_video_bytes": _FAKE_MP4,
                "source_duration_seconds": 6.0,
                "duration_seconds": None,
            },
            id="upscale-of-a-6s-source",
        ),
    ],
)
def test_a_small_gemini_api_output_comes_back_inline(
    tmp_path: Path, request_kwargs: dict[str, Any]
) -> None:
    stub = _OmniStub()
    result = _run(
        stub, tmp_path, "Keep going.", allow_uri_delivery=True, **request_kwargs
    )
    assert "error" not in result, result
    assert "delivery" not in stub.response_format(), stub.response_format()
    assert not [w for w in result.get("warnings", []) if "Google-hosted URI" in w]


@pytest.mark.asyncio
async def test_a_measured_upscale_is_metered_at_the_pair_that_rendered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 4K request that measured 1080p and 6.0s bills 720p->1080p for 6.0s,
    with no encoder allowance and as a metered figure."""
    ctx = context(tmp_path, vertex=True)
    app_ctx = ctx.request_context.lifespan_context
    rendered = tmp_path / "videos" / "upscaled.mp4"
    rendered.write_bytes(b"video")
    monkeypatch.setattr(server, "_client_for_omni", lambda *a, **kw: app_ctx.client)
    monkeypatch.setattr(
        server,
        "generate_video_omni_impl",
        AsyncMock(
            return_value={
                "interaction_id": "upscaled",
                "video_url": f"file://{rendered}",
                "model": omni.OMNI_1_2_MODEL,
                "task": "upscale",
                "rendered_resolution": "4K",
                "duration_seconds": None,
            }
        ),
    )
    monkeypatch.setattr(server, "_probe_media", _probe(6.0, (1920, 1080)))
    result = await server._omni_generate_and_manifest(
        app_ctx,
        ctx,
        prompt="Upsample video.",
        model=omni.OMNI_1_2_MODEL,
        input_video_uri="gs://bucket/draft.mp4",
        task="upscale",
        duration_seconds=None,
        resolution="4K",
        upscale_source_resolution="720p",
        source_duration_seconds=6.0,
        output_gcs_uri="gs://bucket/upscaled.mp4",
    )
    metered = estimate_omni_upscale_cost(6.0, "720p", "1080p", actual=True)
    assert result["cost"]["usd"] == pytest.approx(round(metered.usd, 6))
    assert result["cost"]["is_estimate"] is False
    assert result["cost"]["breakdown"]["seconds"] == 6.0
    assert "720p to 1080p" in result["cost"]["detail"]


def test_an_upscale_that_rendered_outside_the_published_pairs_prices_the_request() -> (
    None
):
    cost = server._omni_upscale_cost(
        6.0, "720p", rendered="720p", requested="1080p", actual=True
    )
    assert cost is not None
    assert cost["usd"] == pytest.approx(
        round(estimate_omni_upscale_cost(6.0, "720p", "1080p", actual=True).usd, 6)
    )
    assert "prices the 1080p that was requested" in cost["detail"]


@pytest.mark.asyncio
async def test_upscale_on_a_gemini_api_server_says_it_needs_vertex(
    tmp_path: Path,
) -> None:
    ctx = context(tmp_path, vertex=False)
    result = json.loads(
        await server.upscale_video_omni(
            ctx,
            input_video_uri="gs://bucket/draft.mp4",
            source_resolution="720p",
            output_gcs_uri="gs://bucket/up.mp4",
            dry_run=True,
        )
    )
    assert "runs only on Vertex AI" in result["error"]
    assert "generate_video_omni(resolution=...)" in result["error"]
    # Not the shared resolver's wording, which names a parameter this tool
    # does not have.
    assert "omni_backend" not in result["error"]


@pytest.mark.asyncio
async def test_upscale_of_a_gemini_api_interaction_says_where_it_lives(
    tmp_path: Path,
) -> None:
    ctx = context(tmp_path, vertex=True)
    _record(tmp_path / "videos", "prior", omni.OMNI_1_2_MODEL, backend="gemini_api")
    result = json.loads(
        await server.upscale_video_omni(
            ctx, previous_interaction_id="prior", resolution="720p", dry_run=True
        )
    )
    assert "created on the Gemini Developer API" in result["error"]
    assert "input_video_uri" in result["error"]
    assert "omni_backend" not in result["error"]


# ---------------------------------------------------------------------------
# The bucket allowlist warning reaches the caller
# ---------------------------------------------------------------------------


def _no_allowlist(ctx: MagicMock) -> MagicMock:
    ctx.request_context.lifespan_context.allowed_gcs_buckets = frozenset()
    return ctx


def _bucket_warned(result: dict[str, Any], bucket: str) -> bool:
    return any(f"bucket '{bucket}'" in w for w in result.get("warnings") or [])


@pytest.mark.asyncio
@pytest.mark.parametrize("dry_run", [True, False])
async def test_edit_video_surfaces_the_allowlist_warning_for_inputs_and_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dry_run: bool
) -> None:
    ctx = _no_allowlist(context(tmp_path, vertex=True))
    monkeypatch.setattr(server, "_fetch_omni_media", AsyncMock(return_value=_PNG))
    monkeypatch.setattr(server, "_omni_generate_and_manifest", _render())
    result = json.loads(
        await server.edit_video(
            ctx,
            "prior",
            "Swap the product",
            reference_image_uris=["gs://refs/product.png"],
            output_gcs_uri="gs://out/edit.mp4",
            dry_run=dry_run,
        )
    )
    assert "error" not in result, result
    assert _bucket_warned(result, "refs")
    assert _bucket_warned(result, "out")
    warned = [call.args[0] for call in ctx.warning.await_args_list]
    assert any("bucket 'refs'" in w for w in warned)


@pytest.mark.asyncio
@pytest.mark.parametrize("dry_run", [True, False])
async def test_upscale_surfaces_the_allowlist_warning_for_input_and_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dry_run: bool
) -> None:
    ctx = _no_allowlist(context(tmp_path, vertex=True))
    monkeypatch.setattr(server, "fetch", AsyncMock(return_value=None))
    monkeypatch.setattr(server, "_omni_generate_and_manifest", _render())
    result = json.loads(
        await server.upscale_video_omni(
            ctx,
            input_video_uri="gs://drafts/draft.mp4",
            source_resolution="720p",
            output_gcs_uri="gs://out/up.mp4",
            dry_run=dry_run,
        )
    )
    assert "error" not in result, result
    assert _bucket_warned(result, "drafts")
    assert _bucket_warned(result, "out")


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", sorted(_SOURCE_TOOLS))
async def test_a_gcs_source_is_resolved_once_and_its_warning_surfaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool: str
) -> None:
    """Evaluated three times per call, it logged the same warning three times
    and returned it to nobody."""
    ctx = _no_allowlist(context(tmp_path, vertex=True))
    monkeypatch.setattr(server, "fetch", AsyncMock(return_value=b"source-bytes"))
    monkeypatch.setattr(server, "_probe_media", _probe(5.0))
    monkeypatch.setattr(
        server, "_omni_generate_and_manifest", _render(duration_seconds=15.0)
    )
    calls: list[Any] = []
    resolve = server._omni_gcs_source

    def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append(args)
        return resolve(*args, **kwargs)

    monkeypatch.setattr(server, "_omni_gcs_source", spy)
    result = json.loads(
        await _SOURCE_TOOLS[tool](ctx, input_video_uri="gs://sources/src.mp4")
    )
    assert "error" not in result, result
    assert len(calls) == 1
    assert _bucket_warned(result, "sources")


# ---------------------------------------------------------------------------
# Reference images on edit_video
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("dry_run", [True, False])
async def test_edit_video_caps_reference_images_before_fetching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dry_run: bool
) -> None:
    """1.2 has its own limit of 6, backed by the server-wide cap for any model
    without one; before either, any number of images was fetched whole into
    memory before the request was even built."""
    fetch = AsyncMock(side_effect=AssertionError("nothing is fetched past the cap"))
    monkeypatch.setattr(server, "_fetch_omni_media", fetch)
    render = _render()
    monkeypatch.setattr(server, "_omni_generate_and_manifest", render)
    result = json.loads(
        await server.edit_video(
            context(tmp_path),
            "prior",
            "Swap the product",
            omni_model=omni.OMNI_1_2_MODEL,
            reference_image_uris=["https://example.com/ref.png"]
            * (server.MAX_OMNI_INPUT_IMAGES + 1),
            dry_run=dry_run,
        )
    )
    assert "Too many reference images" in result["error"]
    fetch.assert_not_awaited()
    render.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
async def test_edit_video_reference_images_reach_the_wire(tmp_path: Path) -> None:
    """Through the real impl: fetched from DATA_FOLDER, inlined as image
    parts, sent on the continuation of the named interaction."""
    (tmp_path / "product.png").write_bytes(_PNG)
    stub = _OmniStub()
    with _stubbed_network(tmp_path, stub) as ctx:
        result = json.loads(
            await server.edit_video(
                ctx,
                "prior",
                "Swap the product for <IMAGE_REF_0>",
                reference_image_uris=[f"file://{tmp_path / 'product.png'}"],
            )
        )
    assert "error" not in result, result
    request = stub.created()
    assert request["model"] == omni.DEFAULT_OMNI_MODEL
    assert request["previous_interaction_id"] == "prior"
    parts = [part for turn in request["input"] for part in turn["content"]]
    images = [part for part in parts if part["type"] == "image"]
    assert images == [
        {
            "type": "image",
            "data": base64.b64encode(_PNG).decode("ascii"),
            "mime_type": "image/png",
        }
    ]


# ---------------------------------------------------------------------------
# Per-model fallbacks and messages
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model, ceiling", [(omni.OMNI_PREVIEW_MODEL, 10), (omni.OMNI_1_2_MODEL, 15)]
)
def test_an_unmeasured_segment_is_priced_at_its_own_models_maximum(
    model: str, ceiling: int
) -> None:
    cost = server._sum_omni_segment_costs(
        [{"model": model, "video_url": "gs://bucket/turn.mp4"}],
        model=model,
        resolution="720p",
    )
    assert cost is not None
    assert cost["breakdown"]["seconds"] == pytest.approx(ceiling + _ONE_FRAME)


@pytest.mark.asyncio
async def test_an_unmeasured_render_with_no_length_is_priced_at_its_models_maximum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = context(tmp_path)
    app_ctx = ctx.request_context.lifespan_context
    monkeypatch.setattr(server, "_client_for_omni", lambda *a, **kw: app_ctx.client)
    monkeypatch.setattr(
        server,
        "generate_video_omni_impl",
        AsyncMock(
            return_value={
                "interaction_id": "fresh",
                "video_url": "gs://bucket/fresh.mp4",
                "model": omni.OMNI_PREVIEW_MODEL,
                "task": "text_to_video",
                "rendered_resolution": "720p",
                "duration_seconds": None,
            }
        ),
    )
    result = await server._omni_generate_and_manifest(
        app_ctx,
        ctx,
        prompt="rain",
        model=omni.OMNI_PREVIEW_MODEL,
        duration_seconds=None,
    )
    assert result["cost"]["breakdown"]["seconds"] == pytest.approx(10 + _ONE_FRAME)


def test_a_model_without_resolution_names_the_default_as_the_way_out() -> None:
    with pytest.raises(ValueError) as refused:
        server._validate_omni_resolution(omni.omni_spec(omni.OMNI_PREVIEW_MODEL), "4K")
    assert f"omni_model='{omni.DEFAULT_OMNI_MODEL}'" in str(refused.value)


# ---------------------------------------------------------------------------
# Integration follow-ups
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
@pytest.mark.parametrize("dry_run", [True, False])
async def test_a_duration_the_continuation_never_sends_is_not_validated(
    tmp_path: Path, dry_run: bool
) -> None:
    """generate_video_omni validated every duration, so a first/last-frame
    continuation with duration_seconds=15 drew "3 to 10" on both runs, though
    the impl sends no duration on a turn carrying previous_interaction_id."""
    stub = _OmniStub()
    with _stubbed_network(tmp_path, stub) as ctx:
        _record(tmp_path / "videos", "prior", omni.OMNI_1_2_MODEL)
        for name in ("first.png", "last.png"):
            (tmp_path / name).write_bytes(_PNG)
        result = json.loads(
            await server.generate_video_omni(
                ctx,
                "move from the first frame to the last",
                previous_interaction_id="prior",
                first_frame_uri=f"file://{tmp_path}/first.png",
                last_frame_uri=f"file://{tmp_path}/last.png",
                duration_seconds=15,
                dry_run=dry_run,
            )
        )
    assert "error" not in result, result
    if not dry_run:
        assert "duration" not in stub.response_format()


@pytest.mark.asyncio
async def test_a_duration_a_fresh_keyframe_render_sends_is_still_validated(
    tmp_path: Path,
) -> None:
    """The same 15s with no continuation is sent, so it is still refused."""
    result = json.loads(
        await server.generate_video_omni(
            context(tmp_path),
            "move from the first frame to the last",
            first_frame_uri="https://example.com/first.png",
            last_frame_uri="https://example.com/last.png",
            duration_seconds=15,
            dry_run=True,
        )
    )
    assert "3 to 10" in result["error"]


@pytest.mark.asyncio
async def test_a_chain_that_stops_mid_way_keeps_the_failing_turns_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mid-chain handler replaced the failing turn's error with a generic
    line, dropping the advice omni attaches to a conversational turn the
    service read as an edit, in the multi-turn case it was written for."""
    ctx = context(tmp_path)
    _record(tmp_path / "videos", "prior", omni.OMNI_1_2_MODEL)
    advice = (
        "The service read this continuation of prior as an edit and refused "
        "its duration. Nothing was rendered or billed."
    )
    render = AsyncMock(
        side_effect=[
            {
                "interaction_id": "turn-1",
                "video_url": "gs://bucket/turn-1.mp4",
                "duration_seconds": 16.0,
            },
            ValueError(advice),
        ]
    )
    monkeypatch.setattr(server, "_omni_generate_and_manifest", render)
    result = json.loads(
        await server.extend_video_omni(
            ctx,
            "Extend this video.",
            previous_interaction_id="prior",
            times=2,
            duration_seconds=10,
        )
    )
    assert result["completed_turns"] == 1
    assert "Extension stopped after 1 of 2 turn(s)" in result["error"]
    assert f"Turn 2 failed: {advice}" in result["error"]
    assert result["interaction_id"] == "turn-1"


@pytest.mark.asyncio
@pytest.mark.parametrize("second_turn_fails", [False, True])
async def test_an_extension_chain_reports_the_backend_its_turns_ran_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, second_turn_fails: bool
) -> None:
    """A live Vertex chain (2026-11-09) reported ``backend: gemini_api``.

    The chain's response never set ``backend``, so _respond stamped the
    server's default -- the Gemini API, wherever a key is configured -- over a
    chain whose every turn ran on Vertex. A caller continuing from its
    interaction_id is routed by that field, to a backend that cannot see the
    interaction. A chain cut short mid-way reports through the same builder.
    """
    ctx = _vertex_with_prior(tmp_path)
    # The server-wide default where a GEMINI_API_KEY is configured.
    monkeypatch.setattr(server, "_backend_of", lambda *a, **kw: "gemini_api")
    turn = {"video_url": "gs://bucket/turn.mp4", "backend": "vertex"}
    render = AsyncMock(
        side_effect=[
            {**turn, "interaction_id": "turn-1", "duration_seconds": 9.0},
            ValueError("Resource exhausted.")
            if second_turn_fails
            else {**turn, "interaction_id": "turn-2", "duration_seconds": 12.0},
        ]
    )
    monkeypatch.setattr(server, "_omni_generate_and_manifest", render)
    result = json.loads(
        await server.extend_video_omni(
            ctx,
            "Extend this video.",
            previous_interaction_id="prior",
            omni_backend="vertex",
            times=2,
            duration_seconds=3,
            resolution="360p",
        )
    )
    assert result["completed_turns"] == (1 if second_turn_fails else 2), result
    assert result["backend"] == "vertex", result
    assert result["backend_reason"], result


@pytest.mark.asyncio
async def test_an_over_long_upscale_source_gets_upscale_advice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A measured 16s source was refused with the edit/extend advice ("extend a
    clip this server generated instead"), which an upscale cannot follow."""
    ctx = context(tmp_path, vertex=True)
    monkeypatch.setattr(server, "fetch", AsyncMock(return_value=b"source-bytes"))
    monkeypatch.setattr(server, "_probe_media", _probe(16.0, (1280, 720)))
    render = _render()
    monkeypatch.setattr(server, "_omni_generate_and_manifest", render)
    result = json.loads(
        await server.upscale_video_omni(
            ctx,
            input_video_uri="gs://bucket/draft.mp4",
            resolution="1080p",
            output_gcs_uri="gs://bucket/up.mp4",
        )
    )
    error = result["error"]
    assert "Standalone upscaling supports videos up to 15 seconds" in error
    assert "16.00s" in error
    assert "edit or extend" not in error and "Extend a clip" not in error
    render.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_upscale_source_is_probed_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three ffmpeg probes ran on the same bytes: the source-length check, the
    duration and the dimensions. One read answers all three."""
    ctx = context(tmp_path, vertex=True)
    monkeypatch.setattr(server, "fetch", AsyncMock(return_value=b"source-bytes"))
    answers = _probe(6.0, (1280, 720))
    probed: list[Any] = []

    async def counting(function: Any, *args: Any) -> Any:
        probed.append(function)
        return await answers(function, *args)

    monkeypatch.setattr(server, "_probe_media", counting)
    render = _render()
    monkeypatch.setattr(server, "_omni_generate_and_manifest", render)
    result = json.loads(
        await server.upscale_video_omni(
            ctx,
            input_video_uri="gs://bucket/draft.mp4",
            resolution="1080p",
            output_gcs_uri="gs://bucket/up.mp4",
        )
    )
    assert "error" not in result, result
    assert probed == [server.measure_video_metadata_bytes]
    assert render.call_args.kwargs["source_duration_seconds"] == 6.0
    assert render.call_args.kwargs["upscale_source_resolution"] == "720p"


@pytest.mark.asyncio
async def test_a_1_2_gcs_source_over_the_inline_cap_is_sent_by_uri(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Vertex's 20 MB inline cap is for bytes in the request; a 1.2 gs://
    source is sent by URI and downloaded only to measure, so it must not be
    held to it."""
    ctx = context(tmp_path, vertex=True)
    large = b"\x00" * (server.OMNI_VERTEX_INLINE_VIDEO_BYTES + 1)
    monkeypatch.setattr(server, "fetch", AsyncMock(return_value=large))
    monkeypatch.setattr(server, "_probe_media", _probe(8.0))
    render = _render()
    monkeypatch.setattr(server, "_omni_generate_and_manifest", render)
    result = json.loads(
        await server.generate_video_omni(
            ctx, "make it night", input_video_uri="gs://bucket/src.mp4"
        )
    )
    assert "error" not in result, result
    assert render.call_args.kwargs["input_video_uri"] == "gs://bucket/src.mp4"
    assert render.call_args.kwargs["input_video_bytes"] is None


def test_a_rebuilt_cost_keeps_its_pricing_date() -> None:
    """_cost_estimate_from_dict dropped pricing_as_of, so a directed 1.2
    render whose director calls reported no usage was dated before the 1.2
    table was verified."""
    from src.pricing import PRICING_AS_OF, cost_to_dict, estimate_video_cost

    render = cost_to_dict(estimate_video_cost(omni.OMNI_1_2_MODEL, 6.0))
    assert render is not None and render["pricing_as_of"] > PRICING_AS_OF
    payload: dict[str, Any] = {}
    server._add_total_cost(payload, [render], [None])
    assert payload["cost"]["pricing_as_of"] == render["pricing_as_of"]


@pytest.mark.asyncio
async def test_the_mcp_planner_carries_an_upscale_source_and_target(
    tmp_path: Path,
) -> None:
    """The MCP tool exposed no resolution or source, so a planned upscale
    could never carry either and always asked the caller for them."""
    fn = getattr(server.plan_generation, "fn", server.plan_generation)
    result = await fn(
        context(tmp_path, vertex=True),
        "Upscale my video",
        resolution="4K",
        source_video_uri="gs://bucket/draft.mp4",
    )
    plan = json.loads(result[0].text)
    route = plan["routes"][0]
    assert route["tool"] == "upscale_video_omni"
    assert route["params"]["resolution"] == "4K"
    assert route["params"]["input_video_uri"] == "gs://bucket/draft.mp4"
    assert not any("Supply input_video_uri" in c for c in route["caveats"])


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", ["generate_video_omni", "edit_video"])
async def test_the_vertex_edit_image_cap_counts_every_image_one_way(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool: str
) -> None:
    """One rule, one count, every caller: reference, inferred-role and
    first/last-frame images all ride in the same edit request."""
    monkeypatch.setattr(
        server,
        "_fetch_omni_media",
        AsyncMock(side_effect=AssertionError("refused before fetching")),
    )
    ctx = context(tmp_path, vertex=True)
    if tool == "generate_video_omni":
        # A first frame, a last frame and one reference: three images.
        call = server.generate_video_omni(
            ctx,
            "change it",
            previous_interaction_id="prior",
            first_frame_uri="https://example.com/first.png",
            last_frame_uri="https://example.com/last.png",
            reference_image_uris=["https://example.com/ref.png"],
            dry_run=True,
        )
    else:
        call = server.edit_video(
            ctx,
            "prior",
            "change it",
            reference_image_uris=["https://example.com/a.png"] * 3,
            dry_run=True,
        )
    result = json.loads(await call)
    assert "at most 2 images" in result["error"]
    assert "this request has 3" in result["error"]


@pytest.mark.asyncio
async def test_an_over_cap_quote_of_a_remote_source_says_the_real_run_measures_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The quote does not download a gs:// source and prices every turn at
    the ceiling; the real run downloads and measures it first. The warning
    said the source "could not be measured", which the quote never tried."""
    ctx = context(tmp_path, vertex=True)
    downloaded = AsyncMock(return_value=b"source-bytes")
    monkeypatch.setattr(server, "fetch", downloaded)
    kwargs = dict(input_video_uri="gs://bucket/src.mp4", max_cost_usd=2.0)
    quote = json.loads(
        await server.extend_video_omni(
            ctx, "Extend this video.", **kwargs, dry_run=True
        )
    )
    downloaded.assert_not_awaited()
    assert quote["would_be_refused"] is True
    warning = next(w for w in quote["warnings"] if "max_cost_usd" in w)
    assert "the real run measures the source first" in warning
    assert "could not be measured" not in warning

    # And the real run does: a 15s source plus one 10s turn is under $2.
    monkeypatch.setattr(server, "_probe_media", _probe(15.0))
    render = _render(duration_seconds=25.0, model=omni.OMNI_1_2_MODEL)
    monkeypatch.setattr(server, "_omni_generate_and_manifest", render)
    result = json.loads(
        await server.extend_video_omni(ctx, "Extend this video.", **kwargs)
    )
    assert "error" not in result, result


# ---------------------------------------------------------------------------
# A conversational extension turn's duration: one switch, pending a live check
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("sends", [True, False], ids=["sends", "withholds"])
@pytest.mark.parametrize(
    "source, times, duration, honoured",
    [
        ("interaction", 1, 3.0, False),
        ("upload", 2, 3.0, False),
        ("upload", 1, 3.0, True),
        ("interaction", 2, None, True),
    ],
)
@pytest.mark.parametrize("dry_run", [True, False])
async def test_a_duration_a_conversational_turn_cannot_carry_is_refused_up_front(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sends: bool,
    source: str,
    times: int,
    duration: float | None,
    honoured: bool,
    dry_run: bool,
) -> None:
    """With the switch off, a turn that continues an interaction sends no
    duration and appends the 10s default. A call whose duration_seconds would
    be dropped on such a turn is refused before anything renders -- for an
    uploaded source's multi-turn chain, before turn 1 bills -- on the dry run
    alike. With it on (the default) every one of these runs."""
    monkeypatch.setattr(omni, "OMNI_CONVERSATIONAL_EXTENSION_SENDS_DURATION", sends)
    ctx = context(tmp_path)
    _record(tmp_path / "videos", "prior", omni.OMNI_1_2_MODEL, duration=3.0)
    src_clip = tmp_path / "videos" / "src.mp4"
    src_clip.write_bytes(b"\x00\x00\x00\x18ftypmp42clip")
    monkeypatch.setattr(server, "_probe_media", _probe(3.0))
    render = _render(duration_seconds=6.0)
    monkeypatch.setattr(server, "_omni_generate_and_manifest", render)
    kwargs: dict[str, Any] = {"times": times, "dry_run": dry_run}
    if duration is not None:
        kwargs["duration_seconds"] = duration
    if source == "interaction":
        kwargs["previous_interaction_id"] = "prior"
    else:
        kwargs["input_video_uri"] = f"file://{src_clip}"
    result = json.loads(
        await server.extend_video_omni(ctx, "Extend this video.", **kwargs)
    )
    if sends or honoured:
        assert "error" not in result, result
        return
    error = result["error"]
    assert "duration_seconds can set an extension turn's length only on a turn" in error
    assert "Leave duration_seconds unset to append the 10s default" in error
    assert "(input_video_uri, times=1)" in error
    assert (
        "every turn of this call does"
        if source == "interaction"
        else "every turn after the first does"
    ) in error
    render.assert_not_awaited()


class _NoVideoStub(_OmniStub):
    """An interaction that completes with nothing in its response."""

    def _body(self, status: str) -> dict[str, Any]:
        body = super()._body(status)
        body.pop("steps", None)
        return body


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
@pytest.mark.parametrize("tool", ["generate_video_omni", "generate_video_draft"])
async def test_a_completed_interaction_with_no_video_is_reported_as_billed(
    tmp_path: Path, tool: str
) -> None:
    """The error body said the interaction "was submitted and may still
    complete and bill" -- of one that had completed, and so had rendered and
    billed already. A caller waiting for it would wait for nothing."""
    stub = _NoVideoStub()
    with _stubbed_network(tmp_path, stub) as ctx:
        if tool == "generate_video_omni":
            raw = await server.generate_video_omni(ctx, "a dog", duration_seconds=4)
        else:
            raw = await server.generate_video(
                ctx,
                "a dog",
                model="veo-3.1-fast-generate-001",
                draft=True,
                duration_seconds=4,
            )
    body = json.loads(raw)
    assert "carried no video" in body["error"], body
    assert body["submission"] == "submitted"
    assert body["interaction_id"] == "i-1"
    assert "may still complete" not in body["note"]
    assert body["note"] == (
        "Interaction i-1 completed, so it has rendered and been billed, but "
        "its video did not reach this server; rendering again bills again."
    )


# ---------------------------------------------------------------------------
# The plan continues an interaction where it was recorded
# ---------------------------------------------------------------------------


async def _plan(ctx: Any, intent: str, **kwargs: Any) -> dict[str, Any]:
    fn = getattr(server.plan_generation, "fn", server.plan_generation)
    return json.loads((await fn(ctx, intent, **kwargs))[0].text)


async def _dry_run(ctx: Any, route: dict[str, Any]) -> dict[str, Any]:
    return json.loads(
        await getattr(server, route["tool"])(ctx, dry_run=True, **route["params"])
    )


def _vertex_server_with_a_key(tmp_path: Path, default: str) -> Any:
    """A Vertex server that also holds a GEMINI_API_KEY, so both backends are
    reachable, with ``default`` as its OMNI_BACKEND."""
    ctx = context(tmp_path, vertex=True)
    app = ctx.request_context.lifespan_context
    app.omni_backend_default = default
    app.gemini_api_client = MagicMock()
    app.gemini_api_client._api_client.vertexai = False
    return ctx


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
@pytest.mark.parametrize(
    "intent", ["edit this video to make it night", "extend this video"]
)
@pytest.mark.parametrize(
    "default, recorded",
    [
        ("gemini_api", "vertex"),
        ("vertex", "gemini_api"),
        ("auto", "vertex"),
        ("auto", "gemini_api"),
    ],
)
async def test_a_planned_continuation_runs_on_the_backend_it_was_recorded_on(
    tmp_path: Path, intent: str, default: str, recorded: str
) -> None:
    """The planner never learnt the recorded backend, so on a server whose
    OMNI_BACKEND pins the other one every continuation route it handed over
    was refused by the tool ("previous_interaction_id was minted on ...")."""
    ctx = _vertex_server_with_a_key(tmp_path, default)
    _record(tmp_path / "videos", "prior", omni.OMNI_1_2_MODEL, backend=recorded)
    plan = await _plan(ctx, intent, previous_interaction_id="prior")
    assert plan["routes"], plan
    for route in plan["routes"]:
        assert route["params"].get("omni_backend") == (
            recorded if default not in ("auto", recorded) else None
        ), route
        payload = await _dry_run(ctx, route)
        assert "error" not in payload, (route, payload)
        assert payload["backend"] == recorded
        # The plan judged delivery on the backend the call runs on, as the
        # tool does: a large output warns on Vertex AI only.
        tool_warns = any("4 MB" in w for w in payload.get("warnings") or [])
        plan_warns = any("4 MB" in c for c in route["caveats"])
        assert plan_warns is tool_warns is (recorded == "vertex"), route["caveats"]


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
@pytest.mark.parametrize(
    "default, recorded, planned",
    [
        # upscale_video_omni names Vertex itself, whatever OMNI_BACKEND says.
        ("gemini_api", "vertex", True),
        ("auto", "vertex", True),
        # A Developer API interaction does not exist on Vertex AI.
        ("vertex", "gemini_api", False),
        ("auto", "gemini_api", False),
    ],
)
async def test_an_upscale_is_planned_exactly_when_the_tool_can_run_it(
    tmp_path: Path, default: str, recorded: str, planned: bool
) -> None:
    """With OMNI_BACKEND=gemini_api the plan called upscaling unsatisfiable
    though the tool would run it; with a Developer API interaction it
    planned an upscale the tool refuses."""
    ctx = _vertex_server_with_a_key(tmp_path, default)
    _record(tmp_path / "videos", "prior", omni.OMNI_1_2_MODEL, backend=recorded)
    plan = await _plan(
        ctx, "upscale this video to 1080p", previous_interaction_id="prior"
    )
    tool = json.loads(
        await server.upscale_video_omni(
            ctx, previous_interaction_id="prior", resolution="1080p", dry_run=True
        )
    )
    if planned:
        [route] = plan["routes"]
        payload = await _dry_run(ctx, route)
        assert "error" not in payload, payload
        # The plan quotes the 15s bound its caveat names; the tool reads the
        # recorded length. Never below the tool's figure.
        assert route["cost"]["usd"] >= payload["estimated_cost"]["usd"]
    else:
        assert plan["routes"] == []
        [conflict] = plan["conflicts"]
        assert conflict["code"] == "upscale_interaction_backend"
        assert "input_video_uri" in conflict["resolution"]
        assert "created on the Gemini Developer API" in tool["error"]


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
@pytest.mark.parametrize(
    "intent", ["edit this video to make it night", "extend this video"]
)
async def test_a_continuation_on_a_backend_the_server_cannot_reach_is_a_conflict(
    tmp_path: Path, intent: str
) -> None:
    """An interaction resolves only on the backend that minted it; a Gemini
    API server cannot reach Vertex AI, so no route can continue one there."""
    ctx = context(tmp_path, vertex=False)
    _record(tmp_path / "videos", "prior", omni.OMNI_1_2_MODEL, backend="vertex")
    plan = await _plan(ctx, intent, previous_interaction_id="prior")
    assert plan["routes"] == []
    [conflict] = plan["conflicts"]
    assert conflict["code"] == "interaction_backend_unreachable"
    assert "created on Vertex AI" in conflict["detail"]
    assert "GOOGLE_GENAI_USE_VERTEXAI=true" in conflict["resolution"]
    assert "as input_video_uri" in conflict["resolution"]


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
@pytest.mark.parametrize(
    "recorded, intent, planned",
    [
        ("720p", "upscale this video to 720p", False),
        ("360p", "upscale this video to 720p", True),
        ("720p", "upscale this video to 4K", True),
        ("4K", "upscale this video to 4K", False),
        # The record wins over the brief's word for it, as the tool's does.
        ("720p", "upscale this 360p video to 1080p", True),
    ],
)
async def test_an_upscale_starts_from_the_recorded_resolution(
    tmp_path: Path, recorded: str, intent: str, planned: bool
) -> None:
    """The planner never learnt the recorded resolution: a 720p clip's
    upscale to 720p was quoted "for a 360p source" and handed over as a
    route the tool refuses."""
    ctx = context(tmp_path, vertex=True)
    _record(
        tmp_path / "videos",
        "prior",
        omni.OMNI_1_2_MODEL,
        backend="vertex",
        resolution=recorded,
    )
    plan = await _plan(ctx, intent, previous_interaction_id="prior")
    if not planned:
        assert plan["routes"] == []
        [conflict] = plan["conflicts"]
        assert conflict["code"] == "upscale_resolution"
        assert (
            f"rendered at {recorded}, as this server recorded it"
            in (conflict["detail"])
        )
        target = intent.rsplit(" ", 1)[1]
        tool = json.loads(
            await server.upscale_video_omni(
                ctx, previous_interaction_id="prior", resolution=target, dry_run=True
            )
        )
        assert "Upscaling supports" in tool["error"]
        return
    [route] = plan["routes"]
    assert "source_resolution" not in route["params"]
    assert not any("Supply source_resolution" in c for c in route["caveats"])
    assert not any("Quoted for a" in c for c in route["caveats"])
    payload = await _dry_run(ctx, route)
    assert "error" not in payload, payload
    assert payload["source_resolution"] == recorded


# ---------------------------------------------------------------------------
# A planned extension asks for the seconds it was asked for
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
@pytest.mark.parametrize("vertex", [False, True])
@pytest.mark.parametrize(
    "intent, kwargs, turns, turn_seconds, appended",
    [
        # 1.2 sets each turn's length, so a short ask is one short turn. The
        # route sent no duration_seconds, and the tool appended its 10s
        # default: 4s more footage, and 4s more billed, than asked.
        ("extend this video by 6 seconds", {}, 1, 6, 6.0),
        # A runtime total is measured against the recorded 4s clip. It was
        # measured against the 15s longest source instead, so this planned one
        # 10s turn and delivered 14s of the 20s asked.
        ("make this video 20 seconds long", {}, 2, 8, 16.0),
        ("extend this video", {"duration_seconds": 20}, 2, 8, 16.0),
        # Whole steps need no duration: the tool's default is the step.
        ("extend this video by 20 seconds", {}, 2, None, 20.0),
    ],
)
async def test_a_planned_extension_appends_what_was_asked_and_quotes_what_the_tool_does(
    tmp_path: Path,
    vertex: bool,
    intent: str,
    kwargs: dict[str, Any],
    turns: int,
    turn_seconds: int | None,
    appended: float,
) -> None:
    ctx = context(tmp_path, vertex=vertex)
    _record(
        tmp_path / "videos",
        "prior",
        omni.OMNI_1_2_MODEL,
        backend="vertex" if vertex else "gemini_api",
        duration=4.0,
    )
    plan = await _plan(ctx, intent, previous_interaction_id="prior", **kwargs)
    [route] = plan["routes"]
    assert route["tool"] == "extend_video_omni"
    assert route["params"]["times"] == turns
    assert route["params"].get("duration_seconds") == turn_seconds
    payload = await _dry_run(ctx, route)
    assert "error" not in payload, payload
    assert payload["appended_seconds"] == appended
    # The plan prices the recorded clip, as the tool's quote does; it priced
    # an unknown source at the ceiling, quoting $2.78 for a $0.70 call.
    assert route["cost"]["usd"] == payload["estimated_cost"]["usd"]


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
async def test_a_planned_extension_sends_no_duration_the_tool_would_refuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With OMNI_CONVERSATIONAL_EXTENSION_SENDS_DURATION off (L2 failed), a
    turn that continues an interaction takes no duration and the tool refuses
    one, so the route leaves it unset and stays callable as given."""
    monkeypatch.setattr(omni, "OMNI_CONVERSATIONAL_EXTENSION_SENDS_DURATION", False)
    ctx = context(tmp_path)
    _record(tmp_path / "videos", "prior", omni.OMNI_1_2_MODEL, duration=4.0)
    plan = await _plan(
        ctx, "extend this video by 6 seconds", previous_interaction_id="prior"
    )
    [route] = plan["routes"]
    assert "duration_seconds" not in route["params"]
    payload = await _dry_run(ctx, route)
    assert "error" not in payload, payload
    assert payload["appended_seconds"] == 10.0


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
@pytest.mark.parametrize("saved", [True, False])
@pytest.mark.parametrize("dry_run", [True, False])
async def test_an_upscale_of_a_gemini_api_interaction_names_the_clip_to_pass_instead(
    tmp_path: Path, saved: bool, dry_run: bool
) -> None:
    """The refusal said "Pass the saved video as input_video_uri instead"
    without saying where that video is, though the manifest records it, as the
    retired-model refusal already says."""
    ctx = context(tmp_path, vertex=True)
    url = f"file://{tmp_path / 'videos' / 'clip.mp4'}"
    (tmp_path / "videos" / "prior.json").write_text(
        json.dumps(
            {
                "interaction_id": "prior",
                "model": omni.OMNI_1_2_MODEL,
                "backend": "gemini_api",
                "duration_seconds": 4.0,
                "resolution": "720p",
                **({"video_url": url} if saved else {}),
            }
        )
    )
    payload = json.loads(
        await server.upscale_video_omni(
            ctx, previous_interaction_id="prior", dry_run=dry_run
        )
    )
    expected = (
        f"input_video_uri='{url}' (the clip it saved)"
        if saved
        else "(no saved copy is recorded here)"
    )
    assert expected in payload["error"], payload
    assert "Nothing was rendered or billed." in payload["error"]


# ---------------------------------------------------------------------------
# The task 1.2 leaves to the service
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
@pytest.mark.parametrize(
    "media, task",
    [
        ({"first_frame_uri": "first.png"}, "image_to_video"),
        (
            {"first_frame_uri": "first.png", "last_frame_uri": "last.png"},
            "image_to_video",
        ),
        ({"reference_image_uris": ["a.png", "b.png"]}, "reference_to_video"),
    ],
)
async def test_a_task_the_service_can_infer_is_reported_but_not_sent(
    tmp_path: Path, media: dict[str, Any], task: str
) -> None:
    """1.2 infers the task from the prompt and the media roles, and a forced
    task restricts its references, so a task the caller did not name stays
    off the wire. The live suite asserted the opposite and failed four
    completed renders on both backends (2026-11-10)."""
    for name in ("first.png", "last.png", "a.png", "b.png"):
        (tmp_path / name).write_bytes(_PNG)
    uris = {
        key: (
            [f"file://{tmp_path / name}" for name in value]
            if isinstance(value, list)
            else f"file://{tmp_path / value}"
        )
        for key, value in media.items()
    }
    stub = _OmniStub()
    with _stubbed_network(tmp_path, stub) as ctx:
        result = json.loads(
            await server.generate_video_omni(
                ctx, "clouds drift over a hill", duration_seconds=3, **uris
            )
        )
    assert "error" not in result, result
    assert "generation_config" not in stub.created()
    assert result["task"] == task
    assert result["task_source"].startswith("inferred"), result["task_source"]


@pytest.mark.timeout(30.0)
def test_a_task_the_caller_names_is_sent(tmp_path: Path) -> None:
    """The contrast: a named task is declared, as extend and upscale do."""
    stub = _OmniStub()
    result = _run(
        stub,
        tmp_path,
        "clouds drift over a hill",
        first_frame_bytes=_PNG,
        task="image_to_video",
        duration_seconds=3,
    )
    assert "error" not in result, result
    assert stub.created()["generation_config"] == {
        "video_config": {"task": "image_to_video"}
    }
    assert result["task_source"] == "requested by the caller"


# ---------------------------------------------------------------------------
# A source video with images is an edit
# ---------------------------------------------------------------------------

_PNG_2 = b"\x89PNG\r\n\x1a\n" + b"\x01" * 32


@pytest.mark.timeout(30.0)
@pytest.mark.parametrize(
    "images",
    [
        pytest.param(
            {"first_frame_bytes": _PNG, "last_frame_bytes": _PNG_2},
            id="first-and-last-frame",
        ),
        pytest.param({"first_frame_bytes": _PNG}, id="first-frame"),
        pytest.param(
            {"first_frame_bytes": _PNG, "reference_image_bytes_list": [_PNG_2]},
            id="first-frame-and-reference",
        ),
    ],
)
def test_a_source_video_with_keyframes_is_sent_as_an_edit(
    tmp_path: Path, images: dict[str, Any]
) -> None:
    """The keyframe checks ran before the source-video check, so a source
    video with a first and last frame was classed image_to_video and sent
    the duration and aspect ratio an edit rejects, while the tool quoted it
    as an edit that sends neither."""
    stub = _OmniStub()
    result = _run(
        stub, tmp_path, "make it night", input_video_bytes=_FAKE_MP4, **images
    )
    assert "error" not in result, result
    assert result["task"] == "edit", result["task"]
    assert "duration" not in stub.response_format(), stub.response_format()
    assert "aspect_ratio" not in stub.response_format(), stub.response_format()
    assert not omni.omni_duration_is_sent(
        omni.OMNI_1_2_MODEL,
        has_input_video=True,
        has_first_frame="first_frame_bytes" in images,
        has_last_frame="last_frame_bytes" in images,
        reference_image_count=len(images.get("reference_image_bytes_list", [])),
    )


# ---------------------------------------------------------------------------
# Tool descriptions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tool",
    ["generate_video_omni", "edit_video", "extend_video_omni", "upscale_video_omni"],
)
def test_every_omni_tool_argument_is_described(tool: str) -> None:
    """An MCP client learns how to call a tool from its description. The 1.2
    rewrite dropped the Args sections, so director_notes and max_retakes
    still drove the directed render but were described nowhere."""
    import inspect

    function = getattr(server, tool)
    doc = inspect.getdoc(function) or ""
    assert "\nArgs:\n" in doc, tool
    args = doc.split("\nArgs:\n", 1)[1].split("\nReturns:", 1)[0]
    described = {
        line.split(":", 1)[0].strip()
        for line in args.splitlines()
        if line.startswith("    ") and not line.startswith("        ") and ":" in line
    }
    assert described == set(inspect.signature(function).parameters), (
        set(inspect.signature(function).parameters) ^ described
    )


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
async def test_a_chain_bounds_an_unmeasured_turn_from_the_turn_before_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Turn 2 was handed the projection (40s on Vertex for an unmeasurable
    source) even after turn 1 measured 16s, so it billed a 40s upper bound
    where the turn can render at most 16 + 10 = 26s."""
    ctx = context(tmp_path, vertex=True)
    app_ctx = ctx.request_context.lifespan_context
    vertex_client = MagicMock()
    vertex_client._api_client.vertexai = True
    monkeypatch.setattr(server, "_get_omni_vertex_global_client", lambda: vertex_client)
    source = tmp_path / "source.mp4"
    source.write_bytes(_FAKE_MP4)
    calls: list[dict[str, Any]] = []

    async def impl(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        if len(calls) == 1:
            rendered = app_ctx.videos_dir / "turn1.mp4"
            rendered.write_bytes(b"video")
            url = f"file://{rendered}"
        else:
            url = "gs://bucket/turn2.mp4"
        return {
            "video_url": url,
            "interaction_id": f"i-{len(calls)}",
            "model": omni.OMNI_1_2_MODEL,
            "task": "extend",
            "duration_seconds": None,
            "rendered_resolution": "720p",
        }

    async def probe(function: Any, *args: Any) -> Any:
        return 16.0 if function is server.measure_video_duration else None

    monkeypatch.setattr(server, "generate_video_omni_impl", impl)
    monkeypatch.setattr(server, "_probe_media", probe)
    result = json.loads(
        await server.extend_video_omni(
            ctx,
            "Extend the video.",
            input_video_uri=str(source),
            times=2,
            output_gcs_uri="gs://bucket/out.mp4",
        )
    )
    assert "error" not in result, result
    first, second = result["segments"]
    assert first["duration_seconds"] == 16.0
    assert calls[1]["source_duration_seconds"] == 16.0
    assert second["duration_seconds"] is None, second
    assert second["duration_upper_bound_seconds"] == 26.0, second
    assert second["duration_source"].startswith("upper bound"), second


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
@pytest.mark.parametrize(
    "default, expected", [("vertex", "vertex"), ("auto", "gemini_api")]
)
async def test_a_draft_honours_the_servers_omni_backend(
    tmp_path: Path, default: str, expected: str
) -> None:
    """The draft and animatic paths asked for a backend with no request at
    all, so on a Vertex server with a Gemini key and OMNI_BACKEND=vertex a
    draft ran on the Gemini API, under a note telling the operator to set
    OMNI_BACKEND=vertex."""
    ctx = context(tmp_path, vertex=True)
    app_ctx = ctx.request_context.lifespan_context
    app_ctx.gemini_api_client = MagicMock()
    app_ctx.gemini_api_client._api_client.vertexai = False
    app_ctx.omni_backend_default = default
    assert server._omni_backend_is_vertex(
        app_ctx, requested=server._omni_server_backend(app_ctx)
    ) is (expected == "vertex")
    quote = json.loads(
        await server.generate_video(
            ctx, "a fox", model="veo-3.1-fast-generate-001", draft=True, dry_run=True
        )
    )
    assert "error" not in quote, quote
    assert quote["backend"] == expected, quote
    if expected == "vertex":
        assert "backend_note" not in quote, quote["backend_note"]

    # And the render goes where the quote said.
    vertex_client = MagicMock()
    vertex_client._api_client.vertexai = True
    clients: list[Any] = []

    async def impl(**kwargs: Any) -> dict[str, Any]:
        clients.append(kwargs["client"])
        return {
            "video_url": "gs://bucket/draft.mp4",
            "interaction_id": "draft",
            "model": omni.OMNI_1_2_MODEL,
            "task": "text_to_video",
            "duration_seconds": 6,
            "rendered_resolution": "720p",
        }

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(server, "_get_omni_vertex_global_client", lambda: vertex_client)
        patch.setattr(server, "generate_video_omni_impl", impl)
        rendered = json.loads(
            await server.generate_video(
                ctx, "a fox", model="veo-3.1-fast-generate-001", draft=True
            )
        )
    assert "error" not in rendered, rendered
    [client] = clients
    assert client is (
        vertex_client if expected == "vertex" else app_ctx.gemini_api_client
    )
    assert rendered["backend"] == expected, rendered


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
@pytest.mark.parametrize(
    "tool, kwargs",
    [
        ("edit_video", {"prompt": "make it night"}),
        ("generate_video_omni", {"prompt": "make it night"}),
        ("extend_video_omni", {"prompt": "Extend this video."}),
        ("upscale_video_omni", {"resolution": "720p"}),
    ],
)
@pytest.mark.parametrize("dry_run", [True, False])
async def test_an_omni_call_reads_the_interactions_record_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tool: str,
    kwargs: dict[str, Any],
    dry_run: bool,
) -> None:
    """Each tool scanned the sidecars up to three times for the same
    interaction (resolving the backend, in its own body, and again before
    rendering), though PriorInteraction exists so the scan runs once; each
    scan is executor-bound filesystem work with its own timeout."""
    vertex = tool == "upscale_video_omni"
    ctx = context(tmp_path, vertex=vertex)
    _record(
        tmp_path / "videos",
        "prior",
        omni.OMNI_1_2_MODEL,
        backend="vertex" if vertex else "gemini_api",
        duration=6.0,
        resolution="360p",
    )
    scans: list[str] = []
    for name in ("_prior_interaction", "_source_duration_for_interaction"):
        real = getattr(server, name)

        def counted(videos_dir: Path, interaction_id: str, _real: Any = real) -> Any:
            scans.append(interaction_id)
            return _real(videos_dir, interaction_id)

        monkeypatch.setattr(server, name, counted)
    if vertex:
        monkeypatch.setattr(
            server, "_get_omni_vertex_global_client", lambda: MagicMock()
        )
    monkeypatch.setattr(
        server,
        "generate_video_omni_impl",
        AsyncMock(
            return_value={
                "video_url": "gs://bucket/out.mp4",
                "interaction_id": "next",
                "model": omni.OMNI_1_2_MODEL,
                "task": "edit",
                "duration_seconds": None,
                "rendered_resolution": "720p",
            }
        ),
    )
    result = json.loads(
        await getattr(server, tool)(
            ctx,
            previous_interaction_id="prior",
            dry_run=dry_run,
            **({"output_gcs_uri": "gs://bucket/out.mp4"} if vertex else {}),
            **kwargs,
        )
    )
    assert "error" not in result, result
    assert scans == ["prior"], scans


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
@pytest.mark.parametrize("tool", ["edit_video", "upscale_video_omni"])
@pytest.mark.parametrize("seconds, accepted", [(15.03, True), (15.5, False)])
async def test_the_tools_and_the_planner_agree_on_the_source_limit(
    tmp_path: Path, tool: str, seconds: float, accepted: bool
) -> None:
    """The planner refused a recorded 15.03s clip the tools accept, by
    comparing strictly where the tools allow one frame over 15s."""
    from src.routing import RoutingConstraints, plan_generation

    vertex = tool == "upscale_video_omni"
    backend = "vertex" if vertex else "gemini_api"
    ctx = context(tmp_path, vertex=vertex)
    _record(
        tmp_path / "videos",
        "prior",
        omni.OMNI_1_2_MODEL,
        backend=backend,
        duration=seconds,
        resolution="720p",
    )
    kwargs: dict[str, Any] = (
        {"resolution": "1080p", "output_gcs_uri": "gs://bucket/out.mp4"}
        if vertex
        else {"prompt": "make it night"}
    )
    quote = json.loads(
        await getattr(server, tool)(
            ctx, previous_interaction_id="prior", dry_run=True, **kwargs
        )
    )
    assert ("error" not in quote) is accepted, quote
    plan = plan_generation(
        "upscale this video to 1080p" if vertex else "make it night in this video",
        RoutingConstraints(
            backend=backend,
            previous_interaction_id="prior",
            previous_interaction_backend=backend,
            previous_interaction_model=omni.OMNI_1_2_MODEL,
            previous_interaction_resolution="720p",
            previous_interaction_seconds=seconds,
            gemini_api_key_available=not vertex,
        ),
    )
    assert bool([r for r in plan.routes if r.tool == tool]) is accepted, plan.conflicts
