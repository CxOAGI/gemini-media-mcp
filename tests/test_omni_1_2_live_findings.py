"""Fixes for what the paid live runs and their dry-run passes found.

Each test pins one finding: a chain whose last turn asked for more than the
ceiling left (refused by the service after the dry run approved it), a
continuation quoted at 720p for a 360p source, an upscale that padded a
measured source and hid its quote behind a max_cost_usd error, a backend note
that described a choice nobody made, a gs:// output whose bound was recorded as
its length, a remote record that lost the resolution an upscale needs, a draft
shorter than the render it previews, and the planner's cheap budget, which
named its cheaper settings without applying them.
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import src.__main__ as server
import src.omni as omni
from src.omni import (
    OMNI_1_2_MODEL,
    omni_duration_clamp_warning,
    omni_extension_output_lengths,
    omni_extension_refusal,
    omni_extension_turn_seconds,
    omni_spec,
)
from src.pricing import estimate_omni_upscale_cost, estimate_video_cost, sum_costs
from src.routing import RoutingConstraints, plan_generation
from tests.test_omni_1_2 import context
from tests.test_omni_1_2_tools import _record, _stubbed_network
from tests.test_wire_contract_omni import _OmniStub

_SPEC = omni_spec(OMNI_1_2_MODEL)
_ONE_FRAME = 1 / 24


def _posts(stub: _OmniStub) -> list[dict[str, Any]]:
    return [r["body"] for r in stub.requests if r["method"] == "POST"]


def _sidecar(videos: Path, interaction_id: str, **fields: Any) -> None:
    (videos / f"{interaction_id}.json").write_text(
        json.dumps(
            {"interaction_id": interaction_id, "model": OMNI_1_2_MODEL, **fields}
        )
    )


# ---------------------------------------------------------------------------
# An extension turn near the ceiling asks for the whole seconds that fit
# ---------------------------------------------------------------------------


def test_the_last_turn_appends_the_whole_seconds_that_fit() -> None:
    """23.03s + 10 passes the 30s ceiling. The quote clamped that turn to
    30.0s, a 6.97s append no request can name; the service refused the 10s
    turn live (INVALID_ARGUMENT), and a 6s turn ran (29.03s)."""
    assert omni_extension_output_lengths(_SPEC, 3.03, 3) == [13.03, 23.03, 29.03]
    assert omni_extension_turn_seconds(_SPEC, 23.03) == 6
    assert omni_extension_turn_seconds(_SPEC, 3.03) == 10
    assert omni_extension_turn_seconds(_SPEC, 23.03, 4) == 4
    # Vertex keeps the last 30s, so a 10s turn always fits its 40s.
    assert omni_extension_turn_seconds(_SPEC, 35.0, vertexai=True) == 10


@pytest.mark.parametrize("source", [27.01, 28.0, 29.03, 30.0, 31.0])
def test_a_source_with_less_than_one_turn_left_is_refused(source: float) -> None:
    """29.03s was quoted as a 0.97s append instead of refused."""
    refusal = omni_extension_refusal(_SPEC, source)
    assert refusal is not None
    assert "no turn can run" in refusal
    assert omni_extension_output_lengths(_SPEC, source, 1) == []


def test_a_source_with_a_turn_left_is_not_refused() -> None:
    assert omni_extension_refusal(_SPEC, 26.99) is None
    assert omni_extension_turn_seconds(_SPEC, 26.99) == 3


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
async def test_the_chain_sends_the_shorter_last_turn_and_quotes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(omni, "_POLL_INTERVAL", 0)
    stub = _OmniStub()
    with _stubbed_network(tmp_path, stub) as ctx:
        _record(tmp_path / "videos", "prior", OMNI_1_2_MODEL, duration=3.03)
        quote = json.loads(
            await server.extend_video_omni(
                ctx,
                "Extend this video.",
                previous_interaction_id="prior",
                times=3,
                dry_run=True,
            )
        )
        assert quote["turn_output_seconds"] == [13.03, 23.03, 29.03]
        assert quote["appended_seconds"] == 26.0
        breakdown = quote["estimated_cost"]["breakdown"]
        # The priced seconds reconcile with the billed seconds beside them.
        assert breakdown["seconds"] == pytest.approx(
            quote["billed_seconds"] + breakdown["encoder_allowance_seconds"]
        )
        result = json.loads(
            await server.extend_video_omni(
                ctx, "Extend this video.", previous_interaction_id="prior", times=3
            )
        )
    assert "error" not in result, result
    sent = [body["response_format"][0].get("duration") for body in _posts(stub)]
    assert sent == [None, None, "6s"]


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
async def test_a_chain_stops_when_a_measured_turn_leaves_no_room(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Projected 23.03s, measured 27.5s: the third turn has under 3s left and
    is not sent, rather than refused by the service after two turns billed."""
    ctx = context(tmp_path)
    _record(tmp_path / "videos", "prior", OMNI_1_2_MODEL, duration=3.03)
    lengths = iter([13.03, 27.5])
    render = AsyncMock(
        side_effect=lambda *a, **k: {
            "interaction_id": f"i-{k['duration_seconds']}",
            "video_url": "file:///x.mp4",
            "backend": "gemini_api",
            "duration_seconds": next(lengths),
            "duration_source": "measured from the rendered video",
        }
    )
    monkeypatch.setattr(server, "_omni_generate_and_manifest", render)
    result = json.loads(
        await server.extend_video_omni(
            ctx, "Extend this video.", previous_interaction_id="prior", times=3
        )
    )
    assert render.await_count == 2
    assert result["completed_turns"] == 2
    assert any("Stopped after 2 turn(s)" in w for w in result["warnings"])


# ---------------------------------------------------------------------------
# A continuation keeps the resolution its source was recorded at
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
async def test_an_edit_of_a_360p_clip_is_sent_and_quoted_at_360p(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Live: with resolution unset the service kept the source's 360p, while
    the dry run quoted 720p (3x) and the response warned "Requested 720p"."""
    monkeypatch.setattr(omni, "_POLL_INTERVAL", 0)
    stub = _OmniStub()
    with _stubbed_network(tmp_path, stub) as ctx:
        _record(tmp_path / "videos", "prior", OMNI_1_2_MODEL, resolution="360p")
        quote = json.loads(
            await server.edit_video(ctx, "prior", "make it night", dry_run=True)
        )
        result = json.loads(await server.edit_video(ctx, "prior", "make it night"))
        fresh = json.loads(
            await server.generate_video_omni(
                ctx, "a fox", duration_seconds=3, dry_run=True
            )
        )
    assert quote["resolution"] == "360p"
    assert "360p" in quote["estimated_cost"]["detail"]
    assert quote["resolution_source"].startswith("the source's recorded resolution")
    assert _posts(stub)[0]["response_format"][0]["resolution"] == "360p"
    assert result["resolution_source"].startswith("the source's recorded resolution")
    assert not any("Requested" in w for w in result.get("warnings", []))
    assert result["requested_duration_seconds"] is None
    assert fresh["resolution_source"].startswith("the default")


@pytest.mark.asyncio
async def test_a_named_resolution_still_wins_on_a_continuation(tmp_path: Path) -> None:
    ctx = context(tmp_path)
    _record(tmp_path / "videos", "prior", OMNI_1_2_MODEL, resolution="360p")
    quote = json.loads(
        await server.edit_video(
            ctx, "prior", "make it night", resolution="1080p", dry_run=True
        )
    )
    assert quote["resolution"] == "1080p"
    assert quote["resolution_source"].startswith("the request")


def test_the_planner_prices_a_continuation_at_the_recorded_resolution() -> None:
    plan = plan_generation(
        "extend it by 3 more seconds",
        RoutingConstraints(
            previous_interaction_id="prior",
            previous_interaction_model=OMNI_1_2_MODEL,
            previous_interaction_backend="gemini_api",
            previous_interaction_resolution="360p",
            previous_interaction_seconds=23.04,
        ),
    )
    route = plan.recommended
    assert route is not None and route.tool == "extend_video_omni"
    # "3 more seconds" was dropped, so the tool's 10s default applied.
    assert route.params["duration_seconds"] == 3
    assert route.cost is not None
    assert "@ 360p" in route.cost.detail
    assert route.cost.breakdown["seconds"] == pytest.approx(26.04 + _ONE_FRAME)


def test_the_planner_refuses_an_extension_within_a_turn_of_the_ceiling() -> None:
    plan = plan_generation(
        "extend it by 3 more seconds",
        RoutingConstraints(
            previous_interaction_id="prior",
            previous_interaction_model=OMNI_1_2_MODEL,
            previous_interaction_backend="gemini_api",
            previous_interaction_seconds=29.03,
        ),
    )
    assert not plan.is_satisfiable
    assert any("no turn can run" in c.detail for c in plan.conflicts), plan.conflicts


# ---------------------------------------------------------------------------
# Upscale quoting
# ---------------------------------------------------------------------------


def test_a_measured_upscale_source_is_priced_as_measured() -> None:
    measured = estimate_omni_upscale_cost(3.03, "360p", "720p", measured=True)
    assert measured.breakdown["seconds"] == pytest.approx(3.03)
    assert "encoder_allowance_seconds" not in measured.breakdown
    assert measured.is_estimate
    unmeasured = estimate_omni_upscale_cost(3.0, "360p", "720p")
    assert unmeasured.breakdown["seconds"] == pytest.approx(3.0 + _ONE_FRAME)
    assert unmeasured.breakdown["encoder_allowance_seconds"] == pytest.approx(
        _ONE_FRAME
    )
    for cost in (measured, unmeasured):
        assert cost.breakdown["usd_per_second"] == pytest.approx(0.034752)
        assert cost.breakdown["video_output_tokens"] == round(
            cost.breakdown["video_output_tokens"]
        )
        assert cost.source_note and "Vertex AI" in cost.source_note


def _vertex(tmp_path: Path) -> MagicMock:
    return context(tmp_path, vertex=True)


@pytest.mark.asyncio
async def test_an_upscale_of_a_measured_take_quotes_its_measured_length(
    tmp_path: Path,
) -> None:
    ctx = _vertex(tmp_path)
    _record(
        tmp_path / "videos",
        "take",
        OMNI_1_2_MODEL,
        backend="vertex",
        duration=3.03,
        resolution="360p",
    )
    quote = json.loads(
        await server.upscale_video_omni(
            ctx, "720p", previous_interaction_id="take", dry_run=True
        )
    )
    assert quote["estimated_cost"]["breakdown"]["seconds"] == pytest.approx(3.03)
    assert quote["duration_source"] == "measured source"


@pytest.mark.asyncio
async def test_an_upscale_of_an_unmeasured_take_keeps_the_allowance(
    tmp_path: Path,
) -> None:
    ctx = _vertex(tmp_path)
    _sidecar(
        tmp_path / "videos",
        "take",
        backend="vertex",
        duration_seconds=3.0,
        duration_source="the clamped request: the rendered file could not be opened",
        resolution="360p",
    )
    quote = json.loads(
        await server.upscale_video_omni(
            ctx, "720p", previous_interaction_id="take", dry_run=True
        )
    )
    assert quote["estimated_cost"]["breakdown"]["seconds"] == pytest.approx(
        3.0 + _ONE_FRAME
    )


@pytest.mark.asyncio
async def test_an_upscale_over_its_cap_quotes_on_a_dry_run_and_refuses_for_real(
    tmp_path: Path,
) -> None:
    """The dry run hard-errored and hid the quote; extend_video_omni returns
    it with would_be_refused, and now both do."""
    ctx = _vertex(tmp_path)
    _record(
        tmp_path / "videos",
        "take",
        OMNI_1_2_MODEL,
        backend="vertex",
        duration=3.03,
        resolution="360p",
    )
    quote = json.loads(
        await server.upscale_video_omni(
            ctx,
            "720p",
            previous_interaction_id="take",
            max_cost_usd=0.01,
            dry_run=True,
        )
    )
    assert quote["would_be_refused"] is True
    assert quote["estimated_cost"]["usd"] > 0.01
    assert any("exceeds max_cost_usd" in w for w in quote["warnings"])
    refused = json.loads(
        await server.upscale_video_omni(
            ctx, "720p", previous_interaction_id="take", max_cost_usd=0.01
        )
    )
    assert "exceeds max_cost_usd" in refused["error"]
    ctx.request_context.lifespan_context.client.interactions.create.assert_not_called()


@pytest.mark.asyncio
async def test_an_upscale_names_its_own_reason_for_vertex(tmp_path: Path) -> None:
    ctx = _vertex(tmp_path)
    _record(
        tmp_path / "videos",
        "take",
        OMNI_1_2_MODEL,
        backend="vertex",
        duration=3.03,
        resolution="360p",
    )
    quote = json.loads(
        await server.upscale_video_omni(
            ctx, "720p", previous_interaction_id="take", dry_run=True
        )
    )
    assert quote["backend_reason"] == "standalone upscaling runs only on Vertex AI"


# ---------------------------------------------------------------------------
# backend_note describes only the auto default's split
# ---------------------------------------------------------------------------


def _vertex_with_key(tmp_path: Path) -> MagicMock:
    ctx = context(tmp_path, vertex=True)
    gemini = MagicMock()
    gemini._api_client.vertexai = False
    ctx.request_context.lifespan_context.gemini_api_client = gemini
    return ctx


@pytest.mark.asyncio
async def test_the_backend_note_is_for_the_auto_default_only(tmp_path: Path) -> None:
    ctx = _vertex_with_key(tmp_path)
    auto = json.loads(
        await server.generate_video_omni(ctx, "a fox", duration_seconds=3, dry_run=True)
    )
    assert auto["backend"] == "gemini_api"
    # Nothing ran on a dry run.
    assert auto["backend_note"].startswith("Omni would run on the Gemini Developer API")
    explicit = json.loads(
        await server.generate_video_omni(
            ctx, "a fox", duration_seconds=3, omni_backend="gemini_api", dry_run=True
        )
    )
    assert "backend_note" not in explicit
    _record(tmp_path / "videos", "prior", OMNI_1_2_MODEL, backend="gemini_api")
    continued = json.loads(
        await server.edit_video(ctx, "prior", "make it night", dry_run=True)
    )
    assert continued["backend"] == "gemini_api"
    assert "backend_note" not in continued


# ---------------------------------------------------------------------------
# Cloud Storage outputs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_gcs_output_is_downloaded_and_measured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A gs:// edit reported the 15s bound as its length; every edit measured
    locally kept its source's 3.03s."""
    app = context(tmp_path, vertex=True).request_context.lifespan_context
    fetched = AsyncMock(return_value=b"rendered-bytes")
    monkeypatch.setattr(server, "fetch", fetched)

    async def probe(function: Any, *args: Any) -> Any:
        assert function is server.measure_video_metadata_bytes
        return 3.03, (640, 360)

    monkeypatch.setattr(server, "_probe_media", probe)
    result: dict[str, Any] = {
        "video_url": "gs://bucket/out/1.mp4",
        "aspect_ratio": None,
    }
    facts = await server._measure_rendered_video(
        result,
        resolution_key="rendered_resolution",
        requested_resolution=None,
        remote_app_ctx=app,
    )
    assert fetched.await_args is not None
    assert fetched.await_args.args[0] == "gs://bucket/out/1.mp4"
    assert facts.duration_seconds == 3.03
    assert facts.duration_source == "measured from the rendered video"
    assert result["rendered_resolution"] == "360p"
    assert result["aspect_ratio"] == "16:9"


def test_a_gcs_take_keeps_its_resolution_for_an_upscale(tmp_path: Path) -> None:
    """The remote record kept duration but not resolution, so an upscale by
    id of a gs:// take was refused as unrecorded."""
    videos = tmp_path / "videos"
    videos.mkdir()
    server._write_remote_interaction_record(
        videos,
        {
            "interaction_id": "gcs-take",
            "backend": "vertex",
            "model": OMNI_1_2_MODEL,
            "duration_seconds": 3.03,
            "duration_source": "measured from the rendered video",
            "resolution": "360p",
            "resolution_source": "measured from the rendered video",
            "video_url": "gs://bucket/out/1.mp4",
        },
    )
    prior = server._prior_interaction(videos, "gcs-take")
    assert prior.resolution == "360p"
    assert prior.duration_seconds == 3.03
    assert prior.duration_measured


@pytest.mark.parametrize(
    ("uri", "warned"),
    [
        ("gs://bucket/omni-tests/mug.mp4", True),
        ("gs://bucket/omni-tests/", False),
        ("gs://bucket/omni-tests", False),
    ],
)
def test_an_object_style_output_uri_is_warned_about(uri: str, warned: bool) -> None:
    warnings = server._gcs_prefix_warning(uri)
    assert bool(warnings) is warned
    if warned:
        assert f"{uri}/<id>.mp4" in warnings[0]


@pytest.mark.asyncio
async def test_every_omni_tool_warns_about_an_object_style_output_uri(
    tmp_path: Path,
) -> None:
    ctx = _vertex(tmp_path)
    _record(
        tmp_path / "videos",
        "take",
        OMNI_1_2_MODEL,
        backend="vertex",
        duration=3.03,
        resolution="360p",
    )
    uri = "gs://bucket/omni-tests/mug.mp4"
    calls = [
        server.generate_video_omni(
            ctx, "a fox", duration_seconds=3, output_gcs_uri=uri, dry_run=True
        ),
        server.edit_video(
            ctx, "take", "make it night", output_gcs_uri=uri, dry_run=True
        ),
        server.extend_video_omni(
            ctx,
            "Extend this video.",
            previous_interaction_id="take",
            output_gcs_uri=uri,
            dry_run=True,
        ),
        server.upscale_video_omni(
            ctx,
            "720p",
            previous_interaction_id="take",
            output_gcs_uri=uri,
            dry_run=True,
        ),
    ]
    for call in calls:
        payload = json.loads(await call)
        assert "error" not in payload, payload
        assert any("ends in a file name" in w for w in payload["warnings"]), payload


# ---------------------------------------------------------------------------
# Durations: snaps are announced, and a draft runs as long as its final
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("asked", "renders"), [(2, 3), (16, 15), (5.5, 6)])
def test_a_changed_duration_is_announced(asked: float, renders: int) -> None:
    warning = omni_duration_clamp_warning(_SPEC, asked)
    assert warning is not None and warning.endswith(f"it renders {renders}s.")
    assert omni_duration_clamp_warning(_SPEC, 6) is None


@pytest.mark.asyncio
async def test_a_quote_announces_the_snap_the_render_will_make(tmp_path: Path) -> None:
    ctx = context(tmp_path)
    quote = json.loads(
        await server.generate_video_omni(ctx, "a fox", duration_seconds=2, dry_run=True)
    )
    assert quote["duration_seconds"] == 3
    assert any(w.endswith("it renders 3s.") for w in quote["warnings"])


@pytest.mark.asyncio
@pytest.mark.parametrize(("asked", "renders"), [(3, 4), (2, 4), (16, 8), (6, 6)])
async def test_a_draft_runs_as_long_as_the_veo_render_it_previews(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, asked: float, renders: int
) -> None:
    ctx = context(tmp_path)
    quote = json.loads(
        await server.generate_video(
            ctx,
            "a fox",
            model="veo-3.1-fast-generate-001",
            duration_seconds=asked,
            draft=True,
            dry_run=True,
        )
    )
    assert quote["duration_seconds"] == renders
    assert quote["estimated_cost"]["breakdown"]["seconds"] == pytest.approx(
        renders + _ONE_FRAME
    )
    render = AsyncMock(
        return_value={
            "interaction_id": "d",
            "video_url": "file:///d.mp4",
            "duration_seconds": float(renders),
        }
    )
    monkeypatch.setattr(server, "_omni_generate_and_manifest", render)
    await server.generate_video(
        ctx,
        "a fox",
        model="veo-3.1-fast-generate-001",
        duration_seconds=asked,
        draft=True,
    )
    assert render.await_args is not None
    assert render.await_args.kwargs["duration_seconds"] == renders
    if renders != asked:
        assert any(f"The draft renders {renders}s" in w for w in quote["warnings"])


# ---------------------------------------------------------------------------
# Planner budget=cheap applies its own settings
# ---------------------------------------------------------------------------


def test_a_cheap_budget_plans_omni_at_360p_and_prices_it() -> None:
    plan = plan_generation(
        "a fox running through snow",
        RoutingConstraints(budget="cheap", media_kind="video", backend="vertex"),
    )
    route = plan.recommended
    assert route is not None and route.tool == "generate_video_omni"
    assert route.params["resolution"] == "360p"
    assert route.cost is not None and "@ 360p" in route.cost.detail
    assert not any("resolution='360p' renders a draft" in c for c in route.caveats)


def test_a_draft_plans_the_shortest_render() -> None:
    plan = plan_generation(
        "a fox running through snow",
        RoutingConstraints(is_draft=True, media_kind="video", backend="vertex"),
    )
    route = plan.recommended
    assert route is not None and route.tool == "generate_video_omni"
    assert route.params["duration_seconds"] == 3.0
    assert route.params["resolution"] == "360p"
    named = plan_generation(
        "a fox running through snow",
        RoutingConstraints(
            is_draft=True, media_kind="video", backend="vertex", duration_seconds=6
        ),
    )
    assert named.recommended is not None
    assert named.recommended.params["duration_seconds"] == 6.0


def test_a_cheap_word_in_the_brief_does_not_drop_the_resolution() -> None:
    plan = plan_generation(
        "a video of a cheap diner at night",
        RoutingConstraints(media_kind="video", backend="vertex"),
    )
    for route in plan.routes:
        assert route.params.get("resolution") in (None, "720p"), route.params


def test_a_cheap_4k_brief_ranks_the_cheaper_omni_route_first() -> None:
    plan = plan_generation(
        "a fox running through snow",
        RoutingConstraints(
            budget="cheap", media_kind="video", backend="vertex", needs_4k=True
        ),
    )
    first, second = plan.routes[0], plan.routes[1]
    assert first.tool == "generate_video_omni", [r.model for r in plan.routes]
    assert first.cost is not None and second.cost is not None
    assert first.cost.usd < second.cost.usd
    # By the budget term's margin, not a hair: with fidelity still counted
    # for the 4K ask, Omni led by 0.005.
    assert first.score - second.score >= 0.015, [r.score for r in plan.routes]


def test_the_planner_names_the_draft_model() -> None:
    plan = plan_generation(
        "a draft video of a robot",
        RoutingConstraints(
            media_kind="video",
            backend="vertex",
            pinned_model="veo-3.1-fast-generate-001",
        ),
    )
    notes = " ".join(plan.notes)
    assert "gemini-omni-flash " not in notes
    if "draft=True" in notes:
        assert OMNI_1_2_MODEL in notes


# ---------------------------------------------------------------------------
# Cost breakdowns reconcile
# ---------------------------------------------------------------------------


def test_a_quote_reports_the_allowance_inside_its_seconds() -> None:
    one = estimate_video_cost(OMNI_1_2_MODEL, 3, "360p")
    assert one is not None
    assert one.breakdown["encoder_allowance_seconds"] == pytest.approx(_ONE_FRAME)
    total = sum_costs([one, one, one], label="three")
    assert total.breakdown["seconds"] == pytest.approx(9.125)
    assert total.breakdown["encoder_allowance_seconds"] == pytest.approx(0.125)


def test_the_aspect_ratio_comes_from_measured_dimensions() -> None:
    assert server._aspect_ratio_of((640, 360)) == "16:9"
    assert server._aspect_ratio_of((720, 1280)) == "9:16"
    assert server._aspect_ratio_of((1024, 1024)) == "1:1"
    assert server._aspect_ratio_of((800, 600)) == "4:3"


# ---------------------------------------------------------------------------
# Regression round: wording that outlived the fixes, and two gaps
# ---------------------------------------------------------------------------


def test_the_ceiling_warning_names_the_turn_that_actually_runs() -> None:
    """It said the clip "stops at 30s" and that the quote "clamps each turn
    at 30s" while the chain appended 6s to 29.03s."""
    [shorter] = server._omni_cumulative_cap_warning(_SPEC, 3.03, 3, 3, None)
    assert "the last turn asks for 6s" in shorter, shorter
    assert "finishes at 29.03s" in shorter, shorter
    assert "stops at" not in shorter and "clamps each turn" not in shorter
    [dropped] = server._omni_cumulative_cap_warning(_SPEC, 23.03, 3, 1, None)
    assert "only 1 of the 3" in dropped and "finishes at 29.03s" in dropped, dropped
    assert "less than 3s" in dropped


def test_appended_seconds_carry_no_float_noise() -> None:
    assert omni.omni_extension_appended_seconds(23.04, [29.03]) == 5.99


def test_the_planner_conflict_names_no_wrong_tool_and_no_rejected_fix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Past the preview's switch-off, as the live run was: its exclusion's
    # "Use gemini-omni-1.2-flash" fix sat beside 1.2's own refusal.
    import src.routing as routing

    monkeypatch.setattr(routing, "omni_preview_is_sunset", lambda: True)
    plan = plan_generation(
        "extend it by 3 more seconds",
        RoutingConstraints(
            previous_interaction_id="prior",
            previous_interaction_model=OMNI_1_2_MODEL,
            previous_interaction_backend="gemini_api",
            previous_interaction_seconds=29.03,
        ),
    )
    [conflict] = [c for c in plan.conflicts if c.code == "no_model_can_serve_request"]
    assert "loop_extend" not in conflict.detail, conflict.detail
    assert f"Use {OMNI_1_2_MODEL}" not in conflict.resolution, conflict.resolution
    assert "Start a new generation" in conflict.resolution


@pytest.mark.asyncio
async def test_an_animatic_beat_runs_as_long_as_its_veo_delivery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 3s beat previewed at 3s finalized at 4s on Veo, unlike draft=true."""
    ctx = context(tmp_path)
    beats = [
        {"prompt": "a fox", "duration_seconds": 3},
        {"prompt": "a hare", "duration_seconds": 6},
    ]
    quote = json.loads(
        await server.generate_clip(
            ctx,
            beats,
            model="veo-3.1-fast-generate-001",
            animatic=True,
            animatic_resolution="360p",
            dry_run=True,
        )
    )
    assert any("beat 1 3s -> 4s" in w for w in quote["warnings"]), quote["warnings"]
    assert not any("beat 2" in w for w in quote["warnings"])
    assert quote["estimated_cost"]["breakdown"]["seconds"] == pytest.approx(
        10 + 2 * _ONE_FRAME
    )
    impl = AsyncMock(
        return_value={
            "interaction_id": "b",
            "video_url": "file:///b.mp4",
            "model": OMNI_1_2_MODEL,
        }
    )
    monkeypatch.setattr(server, "generate_video_omni_impl", impl)
    monkeypatch.setattr(server, "_client_for_omni", lambda app_ctx, **_: MagicMock())
    await server.generate_clip(
        ctx,
        beats,
        model="veo-3.1-fast-generate-001",
        animatic=True,
        animatic_resolution="360p",
    )
    assert [c.kwargs["duration_seconds"] for c in impl.await_args_list] == [4.0, 6.0]


@pytest.mark.asyncio
async def test_an_older_gcs_take_is_upscaled_from_its_measured_clip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Records written before the remote record kept a resolution refused an
    upscale by id with "Supply source_resolution"."""
    ctx = _vertex(tmp_path)
    _sidecar(
        tmp_path / "videos",
        "old",
        backend="vertex",
        duration_seconds=3.0,
        duration_source="the clamped request: the rendered file could not be opened",
        video_url="gs://bucket/omni-tests/old.mp4/1.mp4",
    )
    quote = json.loads(
        await server.upscale_video_omni(
            ctx, "720p", previous_interaction_id="old", dry_run=True
        )
    )
    assert "a real run measures it" in quote["error"], quote
    fetched = AsyncMock(return_value=b"saved-clip")
    monkeypatch.setattr(server, "fetch", fetched)

    async def probe(function: Any, *args: Any) -> Any:
        return 3.03, (640, 360)

    monkeypatch.setattr(server, "_probe_media", probe)
    render = AsyncMock(
        return_value={"interaction_id": "u", "video_url": "gs://bucket/u.mp4"}
    )
    monkeypatch.setattr(server, "_omni_generate_and_manifest", render)
    result = json.loads(
        await server.upscale_video_omni(ctx, "720p", previous_interaction_id="old")
    )
    assert "error" not in result, result
    assert fetched.await_args is not None
    assert fetched.await_args.args[0] == "gs://bucket/omni-tests/old.mp4/1.mp4"
    assert render.await_args is not None
    assert render.await_args.kwargs["upscale_source_resolution"] == "360p"
    assert render.await_args.kwargs["source_duration_seconds"] == 3.03
    assert render.await_args.kwargs["source_duration_measured"] is True


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
async def test_the_continuation_note_says_gcs_outputs_are_measured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(omni, "_POLL_INTERVAL", 0)
    stub = _OmniStub()
    with _stubbed_network(tmp_path, stub) as ctx:
        _record(tmp_path / "videos", "prior", OMNI_1_2_MODEL)
        result = json.loads(await server.edit_video(ctx, "prior", "make it night"))
    [note] = [w for w in result["warnings"] if "sends neither duration_seconds" in w]
    assert "which is downloaded to measure it" in note, note
    assert "labelled upper bound" not in note


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
async def test_the_extension_note_says_gcs_outputs_are_measured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """edit_video's note was fixed and the extension turn's, a separate
    string, still promised gs:// outputs "a labelled upper bound"."""
    monkeypatch.setattr(omni, "_POLL_INTERVAL", 0)
    stub = _OmniStub()
    with _stubbed_network(tmp_path, stub) as ctx:
        _record(tmp_path / "videos", "prior", OMNI_1_2_MODEL, duration=3.03)
        result = json.loads(
            await server.extend_video_omni(
                ctx, "Extend this video.", previous_interaction_id="prior"
            )
        )
    notes = [
        w for w in _strings(result) if "includes the footage before the extension" in w
    ]
    assert notes, result
    for note in notes:
        assert "is downloaded to measure it" in note, note
        assert "labelled upper bound" not in note


def test_no_source_string_still_says_gcs_outputs_go_unmeasured() -> None:
    """The stale wording survived once because it was split across string
    literals; read the source with adjacent literals joined."""
    import re

    join = re.compile(r'"\s*\n\s*(?:f|rf|fr)?"')
    stale = (
        "labelled upper bound when",
        "cannot be opened to measure (a gs:// URI)",
        "An output delivered to Cloud Storage is not opened here",
    )
    root = Path(__file__).resolve().parents[1]
    for path in [
        *sorted((root / "src").glob("*.py")),
        root / "README.md",
        root / "DOCKER_HUB_README.md",
    ]:
        text = join.sub("", path.read_text())
        for phrase in stale:
            assert phrase not in text, f"{path.name}: {phrase}"
