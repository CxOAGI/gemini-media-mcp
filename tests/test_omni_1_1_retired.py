"""gemini-omni-1.1-flash is retired on this server.

Two rules, one per way a caller can still meet the old model:

* A request that NAMES it runs on gemini-omni-1.2-flash -- the redirect
  Google applies to the ID from 2026-12-03, applied here now -- and says so,
  so a client that hard-codes the ID keeps working and is told why its
  response names another model. The planner never offers it, and plans a
  pinned 1.1 as 1.2 with a note.
* An interaction RECORDED on it cannot be redirected, because an interaction
  cannot change models. Continuing one is refused before anything is spent,
  identically on a dry run, with the way forward: the saved video, passed as
  input_video_uri to the tool that takes it.
"""

import datetime
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

import src.__main__ as server
import src.omni as omni
from src.pricing import (
    estimate_video_cost,
    priced_models,
    resolve_model_id,
    unpriced_models,
)
from src.routing import (
    RoutingConstraints,
    plan_generation,
    video_capabilities,
    video_profile,
)
from tests.test_omni_1_2 import context
from tests.test_omni_1_2_tools import (
    _CONTINUATIONS,
    _probe,
    _record,
    _stubbed_network,
)
from tests.test_wire_contract_omni import _OmniStub

# Both spellings: the Developer API's, and the one Vertex AI published 1.1 as,
# which renders made there recorded.
_RETIRED = (omni.OMNI_1_1_MODEL, "gemini-omni-1.1-flash-preview")
_NEW = omni.OMNI_1_2_MODEL


def _redirect_warning(payload: dict[str, Any], requested: str) -> str:
    """The one warning that discloses the redirect, asserted to say enough."""
    found = [w for w in payload.get("warnings") or [] if "is retired" in w]
    assert len(found) == 1, payload.get("warnings")
    warning = found[0]
    assert requested in warning
    assert f"run on {_NEW}" in warning
    assert omni.OMNI_1_1_REDIRECT_DATE in warning
    return warning


def _mentions_retired_model(value: Any) -> bool:
    return omni.OMNI_1_1_MODEL in json.dumps(value, default=str)


# ===========================================================================
# D1: a request naming 1.1 runs on 1.2, and says so
# ===========================================================================


@pytest.mark.parametrize("requested", _RETIRED)
@pytest.mark.parametrize("day", ["2026-09-15", "2026-12-02", "2026-12-03"])
def test_the_retired_id_runs_on_1_2_before_and_after_googles_date(
    requested: str, day: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not date-gated: Google's own redirect date no longer decides anything."""
    monkeypatch.setattr(omni, "_today", lambda: datetime.date.fromisoformat(day))
    assert omni.omni_model_is_retired(requested)
    assert omni.canonical_omni_model(requested) == _NEW
    assert omni.is_omni_model(requested)
    assert omni.omni_spec(requested) is omni.OMNI_SPECS[_NEW]
    assert requested not in omni.OMNI_SPECS
    assert requested not in omni.OMNI_MODELS


def test_only_a_retired_id_carries_the_redirect_warning() -> None:
    for model in (None, _NEW, omni.OMNI_PREVIEW_MODEL, "gemini-omni-9"):
        assert omni.omni_model_reroute_warning(model) is None, model
        assert not omni.omni_model_is_retired(model)
    warning = omni.omni_model_reroute_warning(omni.OMNI_1_1_MODEL)
    assert warning is not None
    assert "retired on this server" in warning
    assert _NEW in warning


@pytest.mark.parametrize("requested", _RETIRED)
@pytest.mark.parametrize("resolution", ["360p", "720p", "1080p", "4K"])
def test_the_retired_id_prices_as_1_2(requested: str, resolution: str) -> None:
    """A quote must price what will be billed, and 1.2 is what renders."""
    assert resolve_model_id(requested) == _NEW
    retired = estimate_video_cost(requested, 6, resolution)
    new = estimate_video_cost(_NEW, 6, resolution)
    assert retired is not None and new is not None
    assert retired.usd == pytest.approx(new.usd)
    # Still accepted, so still priced -- but not a price book of its own.
    assert unpriced_models([requested]) == ()
    assert requested not in priced_models()


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
@pytest.mark.parametrize("requested", _RETIRED)
@pytest.mark.parametrize("dry_run", [True, False])
async def test_generate_video_omni_runs_a_retired_id_on_1_2_and_says_so(
    tmp_path: Path, requested: str, dry_run: bool
) -> None:
    stub = _OmniStub()
    with _stubbed_network(tmp_path, stub) as ctx:
        result = json.loads(
            await server.generate_video_omni(
                ctx, "a lighthouse at dusk", omni_model=requested, dry_run=dry_run
            )
        )
    assert "error" not in result, result
    assert result["model"] == _NEW
    assert result["requested_model"] == requested
    _redirect_warning(result, requested)
    if dry_run:
        assert not stub.requests
        # Quoted at 1.2's rate, for 1.2's render.
        quote = estimate_video_cost(_NEW, 6, "720p")
        assert quote is not None
        assert result["estimated_cost"]["usd"] == pytest.approx(quote.usd, abs=1e-6)
    else:
        assert stub.created()["model"] == _NEW


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
@pytest.mark.parametrize("tool", sorted(_CONTINUATIONS))
@pytest.mark.parametrize("dry_run", [True, False])
async def test_a_continuation_naming_1_1_runs_on_1_2_and_says_so(
    tmp_path: Path, tool: str, dry_run: bool
) -> None:
    """edit_video, extend_video_omni and generate_video_omni continuing a 1.2
    interaction under a hard-coded 1.1 ID: the redirect, not a refusal for a
    model mismatch the caller cannot see."""
    stub = _OmniStub()
    with _stubbed_network(tmp_path, stub) as ctx:
        _record(tmp_path / "videos", "prior", _NEW)
        result = json.loads(
            await _CONTINUATIONS[tool](
                ctx, omni_model=omni.OMNI_1_1_MODEL, dry_run=dry_run
            )
        )
    assert "error" not in result, result
    assert result["model"] == _NEW
    assert result["requested_model"] == omni.OMNI_1_1_MODEL
    _redirect_warning(result, omni.OMNI_1_1_MODEL)
    if dry_run:
        assert not stub.requests
    else:
        assert stub.created()["model"] == _NEW
        assert stub.created()["previous_interaction_id"] == "prior"


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
@pytest.mark.parametrize("dry_run", [True, False])
async def test_extending_an_upload_under_1_1_runs_on_1_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dry_run: bool
) -> None:
    """The other extension source, with 1.2's per-turn duration honoured --
    something 1.1's fixed step refused."""
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"clip")
    monkeypatch.setattr(server, "_probe_media", _probe(5.0))
    stub = _OmniStub()
    with _stubbed_network(tmp_path, stub) as ctx:
        result = json.loads(
            await server.extend_video_omni(
                ctx,
                "Extend this video.",
                input_video_uri=f"file://{source}",
                omni_model=omni.OMNI_1_1_MODEL,
                duration_seconds=4,
                dry_run=dry_run,
            )
        )
    assert "error" not in result, result
    assert result["model"] == _NEW
    _redirect_warning(result, omni.OMNI_1_1_MODEL)
    if dry_run:
        assert result["turn_output_seconds"] == [9.0]
    else:
        assert stub.created()["model"] == _NEW
        assert stub.response_format()["duration"] == "4s"


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
@pytest.mark.parametrize("dry_run", [True, False])
async def test_the_generate_video_draft_runs_on_1_2_never_1_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dry_run: bool
) -> None:
    """The draft shortcut picks its own omni model; it must be the live one."""
    render = AsyncMock(
        return_value={
            "message": "ok",
            "video_url": "gs://bucket/draft.mp4",
            "interaction_id": "i-draft",
            "model": _NEW,
            "task": "text_to_video",
            "duration_seconds": 6,
            "aspect_ratio": "16:9",
            "resolution": None,
            "rendered_resolution": "720p",
        }
    )
    monkeypatch.setattr(server, "generate_video_omni_impl", render)
    result = json.loads(
        await server.generate_video(
            context(tmp_path),
            "a lighthouse at dusk",
            model="veo-3.1-fast-generate-001",
            draft=True,
            duration_seconds=6,
            dry_run=dry_run,
        )
    )
    assert "error" not in result, result
    assert result["model"] == _NEW
    assert not _mentions_retired_model(result)
    if dry_run:
        render.assert_not_awaited()
    else:
        assert render.call_args.kwargs["model"] == _NEW


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
@pytest.mark.parametrize("dry_run", [True, False])
async def test_the_generate_clip_animatic_runs_on_1_2_never_1_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dry_run: bool
) -> None:
    calls: list[dict[str, Any]] = []

    async def render(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        out = tmp_path / "videos" / f"beat{len(calls)}.mp4"
        out.write_bytes(b"mp4")
        return {
            "message": "ok",
            "video_url": f"file://{out}",
            "interaction_id": f"i-{len(calls)}",
            "model": kwargs["model"],
            "task": "text_to_video",
            "duration_seconds": 4,
            "aspect_ratio": "16:9",
            "resolution": None,
            "rendered_resolution": "720p",
        }

    monkeypatch.setattr(server, "generate_video_omni_impl", render)
    result = json.loads(
        await server.generate_clip(
            context(tmp_path),
            beats=[
                {"prompt": "beat 1", "duration_seconds": 4},
                {"prompt": "beat 2", "duration_seconds": 4},
            ],
            animatic=True,
            dry_run=dry_run,
        )
    )
    assert "error" not in result, result
    assert result["model"] == _NEW
    assert not _mentions_retired_model(result)
    if dry_run:
        assert calls == []
    else:
        assert [call["model"] for call in calls] == [_NEW, _NEW]


_PLANNED = [
    ("a video of a cat", RoutingConstraints()),
    ("a 13 second video of waves crashing", RoutingConstraints()),
    ("a draft clip of a robot", RoutingConstraints(resolution="360p")),
    ("a 3 beat reel about coffee", RoutingConstraints(num_beats=3)),
    (
        "a smooth transition from sunrise to snowfall",
        RoutingConstraints(
            first_frame_uri="gs://b/a.png", last_frame_uri="gs://b/b.png"
        ),
    ),
    ("make the sky purple", RoutingConstraints(previous_interaction_id="abc")),
    (
        "extend this clip by 20 seconds",
        RoutingConstraints(previous_interaction_id="abc"),
    ),
    (
        "upscale my video to 1080p",
        RoutingConstraints(backend="vertex", source_video_uri="gs://b/v.mp4"),
    ),
]


@pytest.mark.parametrize(("intent", "constraints"), _PLANNED)
@pytest.mark.parametrize("backend", ["vertex", "gemini_api"])
def test_the_planner_never_offers_1_1(
    intent: str, constraints: RoutingConstraints, backend: str
) -> None:
    """Not as a route, not as a rejection, not in a workflow step's params."""
    from dataclasses import asdict, replace

    plan = plan_generation(intent, replace(constraints, backend=backend))
    assert not _mentions_retired_model(
        {
            "routes": [asdict(r) for r in plan.routes],
            "rejected": [asdict(r) for r in plan.rejected],
            "workflow": [asdict(w) for w in plan.workflow],
            "notes": list(plan.notes),
            "conflicts": [asdict(c) for c in plan.conflicts],
        }
    ), intent


@pytest.mark.parametrize("pinned", _RETIRED)
def test_a_pinned_1_1_is_planned_on_1_2_with_a_note(pinned: str) -> None:
    """13s is past 1.1's old 10s ceiling: the plan is 1.2's in every respect."""
    plan = plan_generation(
        "a 13 second video of waves crashing",
        RoutingConstraints(pinned_model=pinned, backend="vertex"),
    )
    assert plan.routes, plan.conflicts
    assert {route.model for route in plan.routes} == {_NEW}
    assert all(route.params["omni_model"] == _NEW for route in plan.routes)
    assert plan.routes[0].params["duration_seconds"] == 13.0
    assert not plan.conflicts
    note = plan.notes[0]
    assert f"pinned_model={pinned}" in note
    assert "retired on this server" in note
    assert f"This plan is for {_NEW}." in note
    # A call naming 1.1 is served by 1.2, so 1.2's limits and standing apply.
    assert video_capabilities(pinned) == video_capabilities(_NEW)
    assert video_profile(pinned) == video_profile(_NEW)


def test_an_unpinned_plan_carries_no_retirement_note() -> None:
    plan = plan_generation("a video of a cat", RoutingConstraints(pinned_model=_NEW))
    assert not any("retired" in note for note in plan.notes)


@pytest.mark.asyncio
@pytest.mark.timeout(10.0)
async def test_the_plan_generation_tool_carries_the_note(tmp_path: Path) -> None:
    out = await server.plan_generation(
        context(tmp_path),
        "a 13 second video of waves crashing",
        pinned_model=omni.OMNI_1_1_MODEL,
    )
    payload = json.loads(out[0].text)
    assert {route["model"] for route in payload["routes"]} == {_NEW}
    assert "retired on this server" in payload["notes"][0]


@pytest.mark.asyncio
@pytest.mark.timeout(10.0)
@pytest.mark.parametrize(
    "interaction_id, expected",
    [
        ("old", omni.OMNI_1_1_MODEL),
        ("old-vertex-spelling", "gemini-omni-1.1-flash-preview"),
        ("new", _NEW),
        ("never-recorded", None),
        (None, None),
    ],
)
async def test_the_plan_tool_tells_the_planner_which_model_holds_the_interaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interaction_id: str | None,
    expected: str | None,
) -> None:
    """The planner is pure and cannot read a sidecar; only the tool wrapper
    can, through the same lookup the tools make. Without it a continuation of
    a 1.1 interaction was planned as a route every tool refuses."""
    import src.routing as routing

    seen: list[RoutingConstraints] = []
    real = routing.plan_generation

    def spy(intent: str, constraints: RoutingConstraints) -> Any:
        seen.append(constraints)
        return real(intent, constraints)

    monkeypatch.setattr(routing, "plan_generation", spy)
    videos = tmp_path / "videos"
    _record_retired(videos, "old", omni.OMNI_1_1_MODEL)
    _record_retired(videos, "old-vertex-spelling", "gemini-omni-1.1-flash-preview")
    _record_retired(videos, "new", _NEW)
    await server.plan_generation(
        context(tmp_path),
        "edit this video to make it night",
        previous_interaction_id=interaction_id,
    )
    assert [c.previous_interaction_model for c in seen] == [expected]
    [given] = seen
    if expected is None:
        assert given.previous_interaction_backend is None
        assert given.previous_interaction_resolution is None
        assert given.previous_interaction_video_url is None
    else:
        # The rest of the record rides along: the backend a continuation must
        # run on, the resolution an upscale starts from, and the clip a
        # conflict names as the way forward.
        assert given.previous_interaction_backend == "gemini_api"
        assert given.previous_interaction_resolution == "720p"
        assert given.previous_interaction_video_url == (
            f"file://{videos / interaction_id}.mp4"
        )


@pytest.mark.asyncio
@pytest.mark.timeout(10.0)
@pytest.mark.parametrize(
    "intent", ["edit this video to make it night", "extend this video"]
)
async def test_the_plan_tool_plans_no_route_for_a_1_1_interaction(
    tmp_path: Path, intent: str
) -> None:
    _record_retired(tmp_path / "videos", "old", omni.OMNI_1_1_MODEL)
    _record_retired(tmp_path / "videos", "new", _NEW)
    refused = json.loads(
        (
            await server.plan_generation(
                context(tmp_path), intent, previous_interaction_id="old"
            )
        )[0].text
    )
    assert refused["routes"] == []
    assert not refused["is_satisfiable"]
    [conflict] = [c for c in refused["conflicts"] if "retired" in c["detail"]]
    # D2: the way forward names the saved file, as the tool's refusal does.
    tool = "extend_video_omni" if "extend" in intent else "generate_video_omni"
    saved = f"file://{tmp_path / 'videos' / 'old.mp4'}"
    assert (
        f"call {tool} with input_video_uri='{saved}' (the clip it saved) in "
        "place of previous_interaction_id."
    ) in conflict["resolution"]
    # The same request on a 1.2 interaction plans normally, on 1.2.
    planned = json.loads(
        (
            await server.plan_generation(
                context(tmp_path), intent, previous_interaction_id="new"
            )
        )[0].text
    )
    assert planned["routes"] and planned["is_satisfiable"]
    assert {r["params"].get("omni_model") for r in planned["routes"]} == {_NEW}


# ===========================================================================
# D2: an interaction recorded on 1.1 cannot be continued -- refused, with the
# saved video named as the way forward
# ===========================================================================


def _record_retired(
    videos_dir: Path,
    interaction_id: str,
    model: str,
    *,
    backend: str = "gemini_api",
    video_url: str | None = "",
) -> str | None:
    """Write the sidecar a 1.1 render left behind, and return its video_url.

    ``video_url=""`` means "next to the sidecar, as a real render saves it";
    None leaves it out, as an older sidecar may.
    """
    videos_dir.mkdir(parents=True, exist_ok=True)
    if video_url == "":
        clip = videos_dir / f"{interaction_id}.mp4"
        clip.write_bytes(b"mp4")
        video_url = f"file://{clip}"
    manifest: dict[str, Any] = {
        "interaction_id": interaction_id,
        "model": model,
        "backend": backend,
        "duration_seconds": 8.0,
        "duration_source": "measured from the rendered video",
        "resolution": "720p",
    }
    if video_url is not None:
        manifest["video_url"] = video_url
    (videos_dir / f"{interaction_id}.json").write_text(json.dumps(manifest))
    return video_url


def _assert_refusal_says_how_to_proceed(
    error: str, *, recorded: str, tool: str, video_url: str | None
) -> None:
    assert f"was created on {recorded}" in error
    assert "retired" in error
    assert "cannot change models" in error
    assert f"call {tool} with" in error
    if video_url:
        assert f"input_video_uri='{video_url}'" in error
    else:
        assert "no saved copy is recorded here" in error
    assert "Nothing was rendered or billed." in error
    # Never the generic mismatch advice, which would send the caller back to
    # the retired model.
    assert "Leave omni_model unset" not in error


# The tool that takes the saved video, for each tool that was asked to
# continue: edit_video has no input_video_uri, so an edit goes through
# generate_video_omni.
_TAKES_THE_SAVED_VIDEO = {
    "edit_video": "generate_video_omni",
    "extend_video_omni": "extend_video_omni",
    "generate_video_omni": "generate_video_omni",
}


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
@pytest.mark.parametrize("tool", sorted(_CONTINUATIONS))
@pytest.mark.parametrize("dry_run", [True, False])
@pytest.mark.parametrize("recorded", _RETIRED)
@pytest.mark.parametrize("omni_model", [None, _NEW, omni.OMNI_1_1_MODEL])
async def test_continuing_a_1_1_interaction_is_refused_with_the_saved_video(
    tmp_path: Path,
    tool: str,
    dry_run: bool,
    recorded: str,
    omni_model: str | None,
) -> None:
    """Whatever the caller pins -- nothing, 1.2, or 1.1 itself -- the
    interaction is 1.1's and no model this server calls can continue it."""
    stub = _OmniStub()
    with _stubbed_network(tmp_path, stub) as ctx:
        video_url = _record_retired(tmp_path / "videos", "prior", recorded)
        kwargs: dict[str, Any] = {"dry_run": dry_run}
        if omni_model is not None:
            kwargs["omni_model"] = omni_model
        result = json.loads(await _CONTINUATIONS[tool](ctx, **kwargs))
    _assert_refusal_says_how_to_proceed(
        result["error"],
        recorded=recorded,
        tool=_TAKES_THE_SAVED_VIDEO[tool],
        video_url=video_url,
    )
    assert not stub.requests, "a refused continuation reached the service"


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
@pytest.mark.parametrize("tool", sorted(_CONTINUATIONS))
@pytest.mark.parametrize("dry_run", [True, False])
@pytest.mark.parametrize(
    "recorded_on, conflict",
    [
        ("vertex", "server_default"),
        ("vertex", "explicit"),
        ("gemini_api", "server_default"),
        ("gemini_api", "explicit"),
    ],
)
async def test_a_backend_conflict_does_not_mask_the_retired_refusal(
    tmp_path: Path, tool: str, dry_run: bool, recorded_on: str, conflict: str
) -> None:
    """The backend check ran first, so a 1.1 interaction recorded on the
    other backend from the server's OMNI_BACKEND default drew "Drop
    omni_backend" -- an argument the caller never passed -- and never named
    the saved video, which is the only way forward."""
    other = "gemini_api" if recorded_on == "vertex" else "vertex"
    stub = _OmniStub()
    with _stubbed_network(tmp_path, stub) as ctx:
        video_url = _record_retired(
            tmp_path / "videos", "prior", omni.OMNI_1_1_MODEL, backend=recorded_on
        )
        kwargs: dict[str, Any] = {"dry_run": dry_run}
        if conflict == "explicit":
            kwargs["omni_backend"] = other
        else:
            ctx.request_context.lifespan_context.omni_backend_default = other
        result = json.loads(await _CONTINUATIONS[tool](ctx, **kwargs))
    _assert_refusal_says_how_to_proceed(
        result["error"],
        recorded=omni.OMNI_1_1_MODEL,
        tool=_TAKES_THE_SAVED_VIDEO[tool],
        video_url=video_url,
    )
    assert "minted on" not in result["error"]
    assert not stub.requests, "a refused continuation reached the service"


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
async def test_a_conflict_with_the_server_default_says_which_backend_to_name(
    tmp_path: Path,
) -> None:
    """A 1.2 interaction really can continue, on its own backend; a caller who
    passed no omni_backend cannot "drop" it, but can name the right one."""
    ctx = context(tmp_path)
    ctx.request_context.lifespan_context.omni_backend_default = "gemini_api"
    _record_retired(tmp_path / "videos", "prior", _NEW, backend="vertex")
    result = json.loads(
        await server.edit_video(ctx, "prior", "make it night", dry_run=True)
    )
    assert "minted on Vertex AI" in result["error"]
    assert "OMNI_BACKEND default is gemini_api" in result["error"]
    assert 'pass omni_backend="vertex"' in result["error"]
    assert "Drop omni_backend" not in result["error"]


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
@pytest.mark.parametrize(
    "default, drop_helps",
    [("gemini_api", False), ("auto", True), ("vertex", True)],
)
async def test_an_explicit_conflict_says_whether_dropping_omni_backend_helps(
    tmp_path: Path, default: str, drop_helps: bool
) -> None:
    """omni_backend="gemini_api" on a server whose default is also gemini_api
    was told to drop the argument -- and dropping it fell back to the same
    default and was refused again. Every message names the backend that works,
    and offers dropping it only where that would."""
    ctx = context(tmp_path)
    ctx.request_context.lifespan_context.omni_backend_default = default
    _record_retired(tmp_path / "videos", "prior", _NEW, backend="vertex")
    result = json.loads(
        await server.edit_video(
            ctx, "prior", "make it night", omni_backend="gemini_api", dry_run=True
        )
    )
    error = result["error"]
    assert "minted on Vertex AI" in error
    assert 'omni_backend="vertex"' in error
    assert ("Drop omni_backend" in error) is drop_helps
    if not drop_helps:
        assert "OMNI_BACKEND default, gemini_api" in error


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
@pytest.mark.parametrize("dry_run", [True, False])
@pytest.mark.parametrize("recorded", _RETIRED)
async def test_upscaling_a_1_1_interaction_is_refused_with_the_saved_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dry_run: bool, recorded: str
) -> None:
    ctx = context(tmp_path, vertex=True)
    render = AsyncMock(side_effect=AssertionError("nothing may render"))
    monkeypatch.setattr(server, "_omni_generate_and_manifest", render)
    video_url = _record_retired(
        tmp_path / "videos", "prior", recorded, backend="vertex"
    )
    result = json.loads(
        await server.upscale_video_omni(
            ctx, previous_interaction_id="prior", dry_run=dry_run
        )
    )
    _assert_refusal_says_how_to_proceed(
        result["error"],
        recorded=recorded,
        tool="upscale_video_omni",
        video_url=video_url,
    )
    render.assert_not_awaited()
    ctx.request_context.lifespan_context.client.interactions.create.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
@pytest.mark.parametrize(
    "video_url", [None, "gs://bucket/renders/old.mp4"], ids=["unrecorded", "gcs"]
)
async def test_the_refusal_names_whatever_was_saved(
    tmp_path: Path, video_url: str | None
) -> None:
    """A Vertex render delivered to a bucket saved it there; an older sidecar
    may not say where it went at all, and the refusal must not invent one."""
    ctx = context(tmp_path)
    _record_retired(
        tmp_path / "videos", "prior", omni.OMNI_1_1_MODEL, video_url=video_url
    )
    result = json.loads(
        await server.extend_video_omni(
            ctx, "Extend this video.", previous_interaction_id="prior", dry_run=True
        )
    )
    _assert_refusal_says_how_to_proceed(
        result["error"],
        recorded=omni.OMNI_1_1_MODEL,
        tool="extend_video_omni",
        video_url=video_url,
    )


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
@pytest.mark.parametrize("tool", [*sorted(_CONTINUATIONS), "upscale_video_omni"])
async def test_the_dry_run_refuses_word_for_word_what_the_render_refuses(
    tmp_path: Path, tool: str
) -> None:
    async def call(dry_run: bool) -> str:
        if tool == "upscale_video_omni":
            out = await server.upscale_video_omni(
                context(tmp_path, vertex=True),
                previous_interaction_id="prior",
                dry_run=dry_run,
            )
        else:
            out = await _CONTINUATIONS[tool](context(tmp_path), dry_run=dry_run)
        return json.loads(out)["error"]

    _record_retired(
        tmp_path / "videos",
        "prior",
        omni.OMNI_1_1_MODEL,
        backend="vertex" if tool == "upscale_video_omni" else "gemini_api",
    )
    quoted, rendered = await call(True), await call(False)
    assert "retired" in quoted
    assert quoted == rendered


@pytest.mark.parametrize("recorded", _RETIRED)
def test_a_recorded_1_1_interaction_is_read_back_as_1_1(
    tmp_path: Path, recorded: str
) -> None:
    """Folding the record onto 1.2, as a request is folded, would claim the
    interaction lives on a model that never saw it -- and send it there."""
    video_url = _record_retired(tmp_path / "videos", "prior", recorded)
    prior = server._prior_interaction(tmp_path / "videos", "prior")
    assert prior.model == recorded
    assert prior.video_url == video_url
    # A 1.2 record still reads back as 1.2.
    _record_retired(tmp_path / "videos", "fresh", _NEW)
    assert server._prior_interaction(tmp_path / "videos", "fresh").model == _NEW


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
async def test_the_saved_video_the_refusal_names_is_accepted_by_the_tool_it_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Following the advice works: the same clip, as input_video_uri, extends
    on 1.2 with no refusal."""
    monkeypatch.setattr(server, "_probe_media", _probe(8.0))
    ctx = context(tmp_path)
    video_url = _record_retired(tmp_path / "videos", "prior", omni.OMNI_1_1_MODEL)
    refused = json.loads(
        await server.extend_video_omni(
            ctx, "Extend this video.", previous_interaction_id="prior", dry_run=True
        )
    )
    assert f"input_video_uri='{video_url}'" in refused["error"]
    followed = json.loads(
        await server.extend_video_omni(
            ctx, "Extend this video.", input_video_uri=video_url, dry_run=True
        )
    )
    assert "error" not in followed, followed
    assert followed["model"] == _NEW
    assert followed["turn_output_seconds"] == [18.0]
