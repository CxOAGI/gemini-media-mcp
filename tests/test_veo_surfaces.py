"""Veo behaviour this PR touches, even though it is an Omni PR.

src/video.py — the Veo impl — is untouched. But five Veo-facing tools in
src/__main__.py have changed lines, plus the router and the price book, and
each change is one an Omni-focused reviewer would not think to re-check:

  * generate_video's GCS resolution moved AHEAD of the dry_run return, so a
    Veo quote now applies four rejections it previously skipped;
  * the model a Veo quote reports now resolves through the pricing resolver;
  * generate_clip's segments gained resolution provenance, and its total is
    priced per segment rather than at a hardcoded 720p;
  * generate_bridge's frame extraction moved off the shared thread pool;
  * the router gained a media-kind vocabulary, a 360p capability, two new
    capability needs and a ranking tie-break, all of which decide whether a
    Veo model is offered at all.

These are the regressions an Omni PR is most likely to ship, because nothing
about them looks like Omni.
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.__main__ import AppContext

VEO_FAST = "veo-3.1-fast-generate-001"
VEO = "veo-3.1-generate-001"
VEO_LITE = "veo-3.1-lite-generate-preview"

# Veo serves these modes only on Vertex. MEASURED: the Gemini Developer API
# answers "Your use case is currently not supported" for first/last-frame and
# "encodedVideo isn't supported by this model" for extension.
_VERTEX_ONLY_TOOLS = frozenset(
    {"generate_transition", "generate_bridge", "loop_extend"}
)


def _ctx(
    tmp_path: Path,
    *,
    vertexai: bool = False,
    bucket: str | None = None,
    allowed: frozenset[str] = frozenset(),
) -> Any:
    for sub in ("images", "videos"):
        (tmp_path / sub).mkdir(exist_ok=True)
    client = MagicMock()
    client._api_client.vertexai = vertexai
    ctx = MagicMock()
    ctx.info = AsyncMock()
    ctx.error = AsyncMock()
    ctx.warning = AsyncMock()
    ctx.request_context.lifespan_context = AppContext(
        data_folder=tmp_path,
        images_dir=tmp_path / "images",
        videos_dir=tmp_path / "videos",
        client=client,
        video_gcs_bucket=bucket,
        allowed_gcs_buckets=allowed,
    )
    return ctx


async def _quote(ctx: Any, **kwargs: Any) -> dict[str, Any]:
    from src.__main__ import generate_video

    return json.loads(
        await generate_video(
            ctx=ctx, prompt="x", model=VEO_FAST, dry_run=True, **kwargs
        )
    )


# ============================================================================
# generate_video: the GCS resolution moved ahead of the dry_run return
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(10.0)
async def test_vertex_still_quotes_an_explicit_gcs_destination(
    tmp_path: Path,
) -> None:
    """The working path must survive the hoist.

    Moving a rejection earlier is only safe if it still lets through
    everything it used to.
    """
    payload = await _quote(
        _ctx(tmp_path, vertexai=True, allowed=frozenset({"bucket"})),
        output_gcs_uri="gs://bucket/out/",
    )
    assert "error" not in payload, payload.get("error")
    assert payload["estimated_cost"]["usd"] > 0


@pytest.mark.asyncio
@pytest.mark.timeout(10.0)
async def test_an_env_bucket_default_is_dropped_on_the_gemini_api_not_rejected(
    tmp_path: Path,
) -> None:
    """The documented asymmetry, and the easiest thing for a hoist to break.

    An EXPLICIT output_gcs_uri on the Gemini API is rejected — the caller
    asked for the impossible. A VIDEO_GCS_BUCKET env default is silently
    dropped instead, so Lite and text-to-video still succeed inline on a
    deployment that has the variable set for its Vertex work.
    """
    payload = await _quote(_ctx(tmp_path, bucket="gs://envbucket/out/"))
    assert "error" not in payload, payload.get("error")
    assert payload["estimated_cost"]["usd"] > 0


@pytest.mark.asyncio
@pytest.mark.timeout(10.0)
async def test_the_quote_now_applies_the_bucket_allowlist(tmp_path: Path) -> None:
    """Newly enforced at dry_run: it used to fire only on the real call."""
    payload = await _quote(
        _ctx(tmp_path, vertexai=True, allowed=frozenset({"allowed-one"})),
        output_gcs_uri="gs://other/out/",
    )
    assert "not in the allowlist" in payload["error"]
    assert "estimated_cost" not in payload


@pytest.mark.asyncio
@pytest.mark.timeout(10.0)
async def test_the_quote_now_rejects_a_malformed_gcs_uri(tmp_path: Path) -> None:
    payload = await _quote(_ctx(tmp_path, vertexai=True), output_gcs_uri="not-a-gs-uri")
    assert "must start with gs://" in payload["error"]
    assert "estimated_cost" not in payload


# ============================================================================
# generate_video: the reported model
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(10.0)
@pytest.mark.parametrize(
    ("model", "vertexai"),
    [
        (VEO_FAST, True),
        (VEO_FAST, False),
        (VEO, True),
        (VEO, False),
        # Lite is Gemini-API-only; a Vertex context without a key refuses it
        # rather than quoting — see the test below.
        (VEO_LITE, False),
    ],
)
async def test_a_veo_quote_names_one_model_on_both_backends(
    tmp_path: Path, vertexai: bool, model: str
) -> None:
    """`model` and `cost.detail` must never name different models.

    The reported id used to be the backend-specific spelling while the detail
    beside it priced the canonical one.
    """
    from src.__main__ import generate_video

    payload = json.loads(
        await generate_video(
            ctx=_ctx(tmp_path, vertexai=vertexai),
            prompt="x",
            model=model,
            dry_run=True,
        )
    )
    assert "error" not in payload, payload.get("error")
    assert payload["model"] == model
    assert payload["model"] in payload["estimated_cost"]["detail"]


@pytest.mark.asyncio
@pytest.mark.timeout(10.0)
async def test_lite_on_vertex_without_a_key_is_refused_by_the_quote_too(
    tmp_path: Path,
) -> None:
    """A third rejection the hoist brought forward.

    Choosing the client moved ahead of the dry_run return with the GCS
    resolution, so a quote for a Gemini-API-only model on a Vertex deployment
    with no key now fails the way the render fails, instead of pricing a call
    that cannot be made.
    """
    from src.__main__ import generate_video

    payload = json.loads(
        await generate_video(
            ctx=_ctx(tmp_path, vertexai=True),
            prompt="x",
            model=VEO_LITE,
            dry_run=True,
        )
    )
    assert "only available via the Gemini API" in payload["error"]
    assert "estimated_cost" not in payload


# ============================================================================
# generate_clip: segments and totals
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(20.0)
async def test_a_veo_clip_segment_states_why_its_resolution_is_fixed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """generate_clip takes no resolution parameter, so a Veo beat is 720p.

    The animatic path measures its beats now; the Veo path cannot and must say
    so rather than leaving the same unattributed echo behind.
    """
    from src.__main__ import generate_clip

    calls: list[dict[str, Any]] = []

    async def veo_impl(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        out = tmp_path / "videos" / f"beat{len(calls)}.mp4"
        out.write_bytes(b"mp4")
        return {
            "video_url": f"file://{out}",
            "model": kwargs.get("model"),
            "duration_seconds": kwargs.get("duration_seconds", 4),
        }

    monkeypatch.setattr("src.__main__.generate_video_impl", veo_impl)

    payload = json.loads(
        await generate_clip(
            ctx=_ctx(tmp_path),
            beats=[{"prompt": "a"}, {"prompt": "b"}],
            model=VEO_FAST,
        )
    )
    assert "error" not in payload, payload.get("error")
    beats = [s for s in payload["segments"] if s.get("kind") == "beat"]
    assert len(beats) == 2
    for segment in beats:
        assert segment["model"] == VEO_FAST
        assert segment["resolution"] == "720p"
        assert "no resolution parameter" in segment["resolution_source"]
        # Nothing was measured, so nothing may claim to have been.
        assert "rendered_dimensions" not in segment


@pytest.mark.asyncio
@pytest.mark.timeout(20.0)
async def test_a_veo_clip_total_is_unchanged_by_per_segment_pricing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The total used to be priced at a hardcoded 720p for every segment.

    It reads each segment's recorded resolution now, which must produce the
    identical figure for Veo — two 4s beats on the fast tier at $0.10/s.
    """
    from src.__main__ import generate_clip

    calls: list[dict[str, Any]] = []

    async def veo_impl(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        out = tmp_path / "videos" / f"beat{len(calls)}.mp4"
        out.write_bytes(b"mp4")
        return {
            "video_url": f"file://{out}",
            "model": kwargs.get("model"),
            "duration_seconds": 4,
        }

    monkeypatch.setattr("src.__main__.generate_video_impl", veo_impl)

    payload = json.loads(
        await generate_clip(
            ctx=_ctx(tmp_path),
            beats=[
                {"prompt": "a", "duration_seconds": 4},
                {"prompt": "b", "duration_seconds": 4},
            ],
            model=VEO_FAST,
        )
    )
    assert payload["cost"]["usd"] == pytest.approx(2 * 4 * 0.10)


# ============================================================================
# The router still offers Veo where Veo is the answer
# ============================================================================


def test_a_plain_video_brief_still_leads_with_veo() -> None:
    """The vocabulary and capability changes must not have displaced it."""
    from src.routing import RoutingConstraints, plan_generation

    plan = plan_generation("a video of a cat", RoutingConstraints(media_kind="video"))
    top = plan.recommended
    assert top is not None
    assert top.model.startswith("veo"), [r.model for r in plan.routes]


def test_a_4k_brief_still_routes_to_veo() -> None:
    """Omni 1.1 can render 4K now, but Veo is still the finishing tier."""
    from src.routing import RoutingConstraints, plan_generation

    plan = plan_generation(
        "a cinematic drone shot over a coastline",
        RoutingConstraints(media_kind="video", resolution="4K"),
    )
    top = plan.recommended
    assert top is not None
    assert top.model.startswith("veo")
    assert top.params["resolution"] == "4K"


def test_a_supplied_first_and_last_frame_still_routes_to_veo_transition() -> None:
    """Omni now competes for interpolation; it must not have taken it over."""
    from src.routing import RoutingConstraints, plan_generation

    plan = plan_generation(
        "a crossfade between these two stills",
        RoutingConstraints(
            media_kind="video",
            first_frame_uri="gs://b/a.png",
            last_frame_uri="gs://b/b.png",
        ),
    )
    assert any(r.tool == "generate_transition" for r in plan.routes)
    transition = next(r for r in plan.routes if r.tool == "generate_transition")
    assert transition.params["first_frame_uri"] == "gs://b/a.png"
    assert transition.params["last_frame_uri"] == "gs://b/b.png"


def test_veo_records_carry_no_resolution_notes() -> None:
    """The field was added for Omni's derived tiers; Veo's rates are published.

    A note on a Veo record would be claiming a caveat that does not exist.
    """
    from src.pricing import _VIDEO_PRICING  # pyright: ignore[reportPrivateUsage]

    for model, record in _VIDEO_PRICING.items():
        if model.startswith("veo"):
            assert not record.resolution_notes, model


@pytest.mark.parametrize("model", [VEO_FAST, VEO, VEO_LITE])
def test_veo_has_no_360p_rate_and_is_not_given_one(model: str) -> None:
    """360p entered the resolution vocabulary for Omni's draft tier.

    Veo publishes no 360p rate, so the answer must stay "unpriced" rather than
    quietly falling back to the 720p one.
    """
    from src.pricing import estimate_video_cost

    assert estimate_video_cost(model, 6, "360p") is None
    assert estimate_video_cost(model, 6, "720p") is not None


@pytest.mark.parametrize(
    ("model", "duration", "expected"),
    [
        (VEO_FAST, 5.0, 4),
        # min() keeps the FIRST candidate on a tie, so 5 snaps down to 4 and
        # 7 snaps down to 6 — the documented behaviour, mirrored from
        # src/video.py so a quote matches the clip that is really billed.
        (VEO_FAST, 7.0, 6),
        (VEO, 6.0, 6),
    ],
)
def test_the_veo_duration_snap_is_untouched_by_the_omni_extend_branch(
    model: str, duration: float, expected: int
) -> None:
    """snap_video_duration gained an extend_video branch for Omni.

    It is guarded on the model family, so Veo's 4/6/8 snap — and its own
    extend_video override — must be exactly as they were.
    """
    from src.pricing import snap_video_duration

    assert snap_video_duration(model, duration) == expected
    assert snap_video_duration(model, 4.0, "extend_video") == 7
    assert snap_video_duration(model, 4.0, "reference_to_video") == 8


# ============================================================================
# Provenance parity: every rendered number says where it came from
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(20.0)
async def test_a_veo_clip_segment_states_where_its_duration_came_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It carried resolution_source but no duration_source — the one gap left.

    Veo renders exactly the length it is sent, which is the fact
    _segment_is_metered already relies on to price a Veo segment without a
    probe. So the number has a provenance; it just is not a measurement, and
    saying nothing left it indistinguishable from one.
    """
    from src.__main__ import generate_clip

    calls: list[dict[str, Any]] = []

    async def veo_impl(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        out = tmp_path / "videos" / f"beat{len(calls)}.mp4"
        out.write_bytes(b"mp4")
        return {
            "video_url": f"file://{out}",
            "model": kwargs.get("model"),
            "duration_seconds": kwargs.get("duration_seconds", 4),
        }

    monkeypatch.setattr("src.__main__.generate_video_impl", veo_impl)

    payload = json.loads(
        await generate_clip(ctx=_ctx(tmp_path), beats=[{"prompt": "a"}], model=VEO_FAST)
    )
    segment = payload["segments"][0]
    assert "Veo renders exactly the length it is sent" in segment["duration_source"]
    assert "no resolution parameter" in segment["resolution_source"]


@pytest.mark.asyncio
@pytest.mark.timeout(20.0)
async def test_a_bridge_measures_its_render_and_labels_both_numbers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bridge reported the snapped request as an unlabelled integer.

    It was the one rendered path with neither a duration_source nor a
    resolution_source, so its number could not be told apart from a measured
    one — and the integer was the tell.
    """
    import imageio.v3 as iio
    import numpy as np

    from src.__main__ import generate_bridge

    frames = [
        np.full((64, 64, 3), (i * 5) % 256, dtype=np.uint8) for i in range(24 * 3)
    ]
    buf = __import__("io").BytesIO()
    iio.imwrite(buf, frames, extension=".mp4", fps=24)
    clip = buf.getvalue()
    for name in ("from.mp4", "to.mp4"):
        (tmp_path / name).write_bytes(clip)

    async def veo_impl(**kwargs: Any) -> dict[str, Any]:
        out = tmp_path / "videos" / "bridge.mp4"
        out.write_bytes(clip)
        return {
            "video_url": f"file://{out}",
            "model": kwargs.get("model"),
            "duration_seconds": 4,
        }

    monkeypatch.setattr("src.__main__.generate_video_impl", veo_impl)

    payload = json.loads(
        await generate_bridge(
            ctx=_ctx(tmp_path),
            from_clip_uri=f"file://{tmp_path / 'from.mp4'}",
            to_clip_uri=f"file://{tmp_path / 'to.mp4'}",
            model=VEO_FAST,
        )
    )
    assert "error" not in payload, payload.get("error")
    # Measured off the file the stub wrote: 3s, not the 4s the request snapped
    # to — which is exactly the difference the label exists to expose.
    assert payload["duration_source"] == "measured from the rendered video"
    assert payload["duration_seconds"] == pytest.approx(3.0, abs=0.2)
    assert "no resolution parameter" in payload["resolution_source"]
    assert payload["resolution"] == "720p"


@pytest.mark.asyncio
@pytest.mark.timeout(10.0)
@pytest.mark.parametrize("vertexai", [True, False])
async def test_a_veo_response_says_which_backend_ran_it(
    tmp_path: Path, vertexai: bool
) -> None:
    """Omni's responses carry it; Veo's did not.

    A session spent three calls working out a backend confusion that this
    field answers in one.
    """
    payload = await _quote(_ctx(tmp_path, vertexai=vertexai))
    assert payload["backend"] == ("vertex" if vertexai else "gemini_api")


def test_the_planner_routes_to_the_tool_its_reasons_name() -> None:
    """A brief asking for a transition offered only generate_video routes.

    The conflict block and the capability rejections both named
    generate_transition, so the plan talked about a tool it never handed over.
    """
    from src.routing import plan_generation

    plan = plan_generation("a smooth transition from sunrise to snowfall")
    named = {
        "generate_transition"
        for r in plan.rejected
        if "generate_transition" in r.reason
    }
    if named:
        assert any(r.tool == "generate_transition" for r in plan.routes), [
            r.tool for r in plan.routes
        ]
    top = plan.recommended
    assert top is not None
    assert top.tool == "generate_transition"
    # And it says what it still needs.
    assert any("Add first_frame_uri" in c for c in top.caveats)


def test_a_multi_shot_brief_mentioning_transitions_still_routes_to_clip() -> None:
    """generate_clip renders its own bridges; the guard is the beat count."""
    from src.routing import plan_generation

    plan = plan_generation(
        "a 3 shot commercial with crossfade transitions, 24 seconds total"
    )
    assert plan.routes[0].tool == "generate_clip"


def _seed_media(tmp_path: Path) -> tuple[str, str]:
    """A real mp4 and png on disk, since several tools probe their sources."""
    import imageio.v3 as iio
    import numpy as np

    for sub in ("images", "videos"):
        (tmp_path / sub).mkdir(exist_ok=True)
    mp4 = tmp_path / "videos" / "seed.mp4"
    iio.imwrite(
        mp4,
        [np.full((64, 64, 3), (i * 5) % 256, dtype=np.uint8) for i in range(48)],
        extension=".mp4",
        fps=24,
    )
    png = tmp_path / "images" / "a.png"
    png.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
            "1f15c4890000000a49444154789c6360000002000100ffff030000060005"
            "57bfabd40000000049454e44ae426082"
        )
    )
    return f"file://{mp4}", f"file://{png}"


def _render_tool_calls(video: str, image: str) -> dict[str, dict[str, Any]]:
    return {
        "generate_video": dict(prompt="x", model=VEO_FAST),
        "generate_video_omni": dict(prompt="x"),
        "edit_video": dict(prompt="x", previous_interaction_id="i-1"),
        "extend_video_omni": dict(prompt="c", previous_interaction_id="i-1"),
        "generate_clip": dict(
            beats=[{"prompt": "a"}, {"prompt": "b"}], model=VEO_FAST, add_bridges=True
        ),
        "generate_transition": dict(
            prompt="x", first_frame_uri=image, last_frame_uri=image, model=VEO_FAST
        ),
        "loop_extend": dict(prompt="c", video_uri=video, model=VEO_FAST, times=1),
        "generate_storyboard": dict(shots=[{"prompt": "a"}, {"prompt": "b"}]),
    }


@pytest.mark.asyncio
@pytest.mark.timeout(60.0)
@pytest.mark.parametrize("vertexai", [False, True])
async def test_every_render_tool_reports_its_backend(
    tmp_path: Path, vertexai: bool
) -> None:
    """`backend` is a contract, not a per-tool courtesy.

    It was added one tool at a time and reached exactly one of eight. A
    caller cannot reconcile a bill, or reproduce a render, without knowing
    which API served it — and "check the field" is useless advice if seven
    tools omit it. Sweeping every tool at once is the only version of this
    check that stays true as tools are added.
    """
    import src.__main__ as main_mod

    video, image = _seed_media(tmp_path)
    expected = "vertex" if vertexai else "gemini_api"
    missing: list[str] = []
    for tool, kwargs in _render_tool_calls(video, image).items():
        if not vertexai and tool in _VERTEX_ONLY_TOOLS:
            # Veo refuses these on the Gemini Developer API, so there is no
            # response to stamp — the refusal is the correct behaviour and is
            # covered by test_a_gemini_api_quote_refuses_a_mode_veo_cannot_serve.
            continue
        if not vertexai and kwargs.get("add_bridges"):
            # A bridge is a first/last-frame render, so a clip that builds them
            # inherits the restriction. The clip itself runs here; the bridges
            # do not, so drop them rather than skipping the tool.
            kwargs = {**kwargs, "add_bridges": False}
        # A bucket, because Vertex refuses to extend without a destination;
        # on the Gemini API the env default is dropped rather than rejected.
        raw = await getattr(main_mod, tool)(
            ctx=_ctx(
                tmp_path,
                vertexai=vertexai,
                bucket="gs://bkt/out/",
                allowed=frozenset({"bkt"}),
            ),
            dry_run=True,
            **kwargs,
        )
        text = raw if isinstance(raw, str) else raw[0].text
        payload = json.loads(text)
        assert "error" not in payload, f"{tool}: {payload['error']}"
        if payload.get("backend") != expected:
            missing.append(f"{tool}: {payload.get('backend')!r} != {expected!r}")
        # A quoted number is the figure a caller decides on, so it needs a
        # source at least as much as a billed one does.
        for field in ("duration_seconds", "resolution"):
            if payload.get(field) is not None and not payload.get(
                f"{field.split('_')[0]}_source"
            ):
                missing.append(f"{tool}: {field} quoted with no source")
    assert not missing, "quotes missing provenance:\n" + "\n".join(missing)


@pytest.mark.asyncio
@pytest.mark.timeout(20.0)
async def test_planning_does_not_claim_a_backend_it_never_calls(
    tmp_path: Path,
) -> None:
    """The contract is "every response that renders", not "every response".

    plan_generation recommends a tool; it places no call. Stamping a backend
    on it would report a decision that was never made — the same class of
    error as a quoted number with no source, pointed the other way.
    """
    from src.__main__ import plan_generation

    result = await plan_generation(ctx=_ctx(tmp_path), intent="a 5 second clip")
    payload = json.loads(result[0].text)
    assert "backend" not in payload


@pytest.mark.asyncio
@pytest.mark.timeout(60.0)
@pytest.mark.parametrize("vertexai", [False, True])
async def test_every_rendered_response_reports_backend_and_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, vertexai: bool
) -> None:
    """The REAL run is the one that reports a bill.

    The dry-run sweep passed while four rendered paths returned a bare
    duration with no source and no backend — the fields stamped on the
    quote and missing from the invoice, which is exactly backwards.
    generate_transition is the case in point: its sibling generate_bridge
    was given a measurement and provenance block, and the identically
    shaped tool beside it was not.
    """
    import src.__main__ as main_mod

    video, image = _seed_media(tmp_path)
    rendered = tmp_path / "videos" / "out.mp4"

    async def veo_impl(**kwargs: Any) -> dict[str, Any]:
        rendered.write_bytes((tmp_path / "videos" / "seed.mp4").read_bytes())
        return {
            "video_url": f"file://{rendered}",
            "model": kwargs.get("model"),
            "duration_seconds": kwargs.get("duration_seconds", 4),
        }

    monkeypatch.setattr("src.__main__.generate_video_impl", veo_impl)

    calls = {
        "generate_transition": dict(
            prompt="x", first_frame_uri=image, last_frame_uri=image, model=VEO_FAST
        ),
        "generate_bridge": dict(
            prompt="x", from_clip_uri=video, to_clip_uri=video, model=VEO_FAST
        ),
        "loop_extend": dict(prompt="c", video_uri=video, model=VEO_FAST, times=1),
        "generate_video": dict(prompt="x", model=VEO_FAST),
    }
    expected = "vertex" if vertexai else "gemini_api"
    bad: list[str] = []
    for tool, kwargs in calls.items():
        if not vertexai and tool in _VERTEX_ONLY_TOOLS:
            continue
        raw = await getattr(main_mod, tool)(
            ctx=_ctx(
                tmp_path,
                vertexai=vertexai,
                bucket="gs://bkt/out/",
                allowed=frozenset({"bkt"}),
            ),
            **kwargs,
        )
        payload = json.loads(raw if isinstance(raw, str) else raw[0].text)
        assert "error" not in payload, f"{tool}: {payload['error']}"
        if payload.get("backend") != expected:
            bad.append(f"{tool}: backend {payload.get('backend')!r} != {expected!r}")
        # A number a caller reconciles a bill against must say where it came
        # from; an unlabelled one cannot be told apart from a measured one.
        for field in ("duration_seconds", "resolution"):
            if field in payload and not payload.get(f"{field.split('_')[0]}_source"):
                bad.append(f"{tool}: {field} present with no source")
    assert not bad, "rendered responses missing provenance:\n" + "\n".join(bad)


@pytest.mark.asyncio
@pytest.mark.timeout(20.0)
async def test_a_loop_extend_quote_measures_the_source_it_can_open(
    tmp_path: Path,
) -> None:
    """added_seconds is growth over a base, and the base was never reported.

    extend_video_omni opens its local source in the pre-flight; this one
    quoted `added_seconds: 7` against nothing, so the number could not be
    reconciled against the file that produced it.
    """
    from src.__main__ import loop_extend

    video, _ = _seed_media(tmp_path)
    payload = json.loads(
        await loop_extend(
            ctx=_ctx(
                tmp_path,
                vertexai=True,
                bucket="gs://bkt/out/",
                allowed=frozenset({"bkt"}),
            ),
            prompt="c",
            video_uri=video,
            model=VEO,
            dry_run=True,
        )
    )
    assert "error" not in payload, payload.get("error")
    # The seeded clip is 48 frames at 24fps.
    assert payload["source_duration_seconds"] == pytest.approx(2.0, abs=0.05)
    assert "measured" in payload["source_duration_source"]
    assert payload["final_duration_seconds"] == pytest.approx(9.0, abs=0.05)
    assert payload["added_seconds"] == 7


@pytest.mark.asyncio
@pytest.mark.timeout(20.0)
async def test_a_loop_extend_quote_says_why_a_remote_source_is_unmeasured(
    tmp_path: Path,
) -> None:
    """A dry run is documented offline, so it must not download a gs:// source.

    Silence would be indistinguishable from a failed probe.
    """
    from src.__main__ import loop_extend

    payload = json.loads(
        await loop_extend(
            ctx=_ctx(
                tmp_path,
                vertexai=True,
                bucket="gs://bkt/out/",
                allowed=frozenset({"bkt"}),
            ),
            prompt="c",
            video_uri="gs://bkt/in.mp4",
            model=VEO,
            dry_run=True,
        )
    )
    assert "error" not in payload, payload.get("error")
    assert payload["source_duration_seconds"] is None
    assert "does not download" in payload["source_duration_source"]


def _write_video(path: Path, width: int, height: int, frames: int = 24) -> None:
    import imageio.v3 as iio
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(
        path,
        [
            np.full((height, width, 3), (i * 7) % 256, dtype=np.uint8)
            for i in range(frames)
        ],
        extension=".mp4",
        fps=24,
    )


def _impl_rendering(path: Path, width: int, height: int) -> Any:
    async def veo_impl(**kwargs: Any) -> dict[str, Any]:
        _write_video(path, width, height)
        return {
            "video_url": f"file://{path}",
            "model": kwargs.get("model"),
            "duration_seconds": kwargs.get("duration_seconds", 4),
        }

    return veo_impl


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
async def test_generate_video_reports_the_resolution_it_billed_at(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Veo bills by resolution and the response named one nowhere.

    The manifest recorded the raw request — null whenever the caller
    defaulted — while the cost line priced 720p. Two fields that must agree,
    neither of them measured.
    """
    from src.__main__ import generate_video

    out = tmp_path / "videos" / "r.mp4"
    monkeypatch.setattr(
        "src.__main__.generate_video_impl", _impl_rendering(out, 1280, 720)
    )

    payload = json.loads(
        await generate_video(ctx=_ctx(tmp_path), prompt="x", model=VEO_FAST)
    )
    assert "error" not in payload, payload.get("error")
    assert payload["resolution"] == "720p"
    assert payload["resolution_source"] == "measured from the rendered video"
    assert payload["rendered_dimensions"] == [1280, 720]


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
async def test_a_veo_render_that_defies_the_request_is_priced_as_rendered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Asking for 1080p and getting 360p must not bill 1080p.

    The request was the only resolution this path ever knew, so a render that
    came back smaller was billed at the price of the one that was asked for.
    """
    from src.__main__ import generate_video

    out = tmp_path / "videos" / "small.mp4"
    monkeypatch.setattr(
        "src.__main__.generate_video_impl", _impl_rendering(out, 640, 360)
    )

    payload = json.loads(
        await generate_video(
            ctx=_ctx(tmp_path), prompt="x", model=VEO_FAST, resolution="1080p"
        )
    )
    assert "error" not in payload, payload.get("error")
    assert payload["resolution"] == "360p"
    assert payload["resolution_source"] == "measured from the rendered video"
    assert any("measured 360p" in w for w in payload.get("warnings", [])), payload.get(
        "warnings"
    )
    priced = json.dumps(payload.get("cost", {}))
    assert "1080p" not in priced, priced


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
async def test_an_unmeasurable_veo_render_says_its_resolution_is_assumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tier the classifier will not claim must not be reported as measured.

    64x64 is near no tier, and inventing one would be exactly the evidence
    the classifier refuses to fabricate.
    """
    from src.__main__ import generate_transition

    out = tmp_path / "videos" / "odd.mp4"
    monkeypatch.setattr(
        "src.__main__.generate_video_impl", _impl_rendering(out, 64, 64)
    )
    img = tmp_path / "images" / "a.png"
    _seed_media(tmp_path)

    payload = json.loads(
        await generate_transition(
            ctx=_ctx(tmp_path),
            prompt="x",
            first_frame_uri=f"file://{img}",
            last_frame_uri=f"file://{img}",
            model=VEO_FAST,
        )
    )
    assert "error" not in payload, payload.get("error")
    assert payload["resolution"] == "720p"
    assert payload["resolution_source"].startswith("assumed:")


@pytest.mark.asyncio
@pytest.mark.timeout(20.0)
@pytest.mark.parametrize(
    ("tool", "kwargs", "mode"),
    [
        (
            "generate_transition",
            {"first_frame_uri": None, "last_frame_uri": None},
            "first_last_frame",
        ),
        (
            "generate_bridge",
            {"from_clip_uri": None, "to_clip_uri": None},
            "first_last_frame",
        ),
        ("loop_extend", {"video_uri": None}, "extend_video"),
    ],
)
async def test_a_gemini_api_quote_refuses_a_mode_veo_cannot_serve(
    tmp_path: Path, tool: str, kwargs: dict[str, Any], mode: str
) -> None:
    """A quote must never price a call the backend cannot run.

    MEASURED against the live service: on the Gemini Developer API, Veo
    refuses every mode that conditions on existing footage. These three tools
    were returning a clean price AND the very backend that made the call
    impossible, in the same response — the signal was already in hand and
    simply was not wired to a gate.
    """
    import src.__main__ as main_mod

    video, image = _seed_media(tmp_path)
    filled = {k: (video if "clip" in k or "video" in k else image) for k in kwargs}
    payload = json.loads(
        await getattr(main_mod, tool)(
            ctx=_ctx(tmp_path), prompt="x", model=VEO_FAST, dry_run=True, **filled
        )
    )
    assert "estimated_cost" not in payload, f"{tool} priced an impossible call"
    assert mode in payload["error"]
    assert "Vertex AI" in payload["error"]


def test_the_planner_does_not_route_to_a_tool_this_backend_refuses() -> None:
    """plan_generation ranked generate_transition top on a backend that 400s.

    The planner already had a backend field and backend-specific rules, so
    the mechanism existed; this mode simply was not among them.
    """
    from src.routing import RoutingConstraints, plan_generation

    intent = "a crossfade between these two frames"
    blocked = plan_generation(intent, RoutingConstraints(backend="gemini_api"))
    codes = [c.code for c in blocked.conflicts]
    assert "veo_mode_unsupported_on_gemini_api" in codes
    conflict = next(
        c for c in blocked.conflicts if c.code == "veo_mode_unsupported_on_gemini_api"
    )
    assert "Vertex AI" in conflict.resolution

    # And the same brief on Vertex must stay clean, or the gate is just noise.
    allowed = plan_generation(intent, RoutingConstraints(backend="vertex"))
    assert "veo_mode_unsupported_on_gemini_api" not in [
        c.code for c in allowed.conflicts
    ]


@pytest.mark.asyncio
@pytest.mark.timeout(20.0)
@pytest.mark.parametrize(
    ("times", "expected_seconds"), [(1, 9.0), (2, 25.0), (3, 48.0)]
)
async def test_a_loop_extend_quote_bills_every_turns_assembled_length(
    tmp_path: Path, times: int, expected_seconds: float
) -> None:
    """MEASURED: a 4.0s source extended once rendered 11s and billed 11s.

    Veo re-bills the assembled clip on every turn, exactly as omni does, so
    turn i outputs source + 7i and the chain costs the SUM of those lengths.
    Quoting `times * 7` charged only the appended footage: 57% under on a
    single extension, and an order of magnitude on a long chain ($56 quoted
    against $620 actual at veo-3.1-generate-001 rates).

    Pricing the FINAL length instead would fix times=1 and still under-quote
    every chain, which is why this is parametrized past 1.
    """
    from src.__main__ import loop_extend

    video, _ = _seed_media(tmp_path)  # 48 frames at 24fps = 2.0s
    payload = json.loads(
        await loop_extend(
            ctx=_ctx(
                tmp_path,
                vertexai=True,
                bucket="gs://bkt/out/",
                allowed=frozenset({"bkt"}),
            ),
            prompt="c",
            video_uri=video,
            model=VEO,
            times=times,
            dry_run=True,
        )
    )
    assert "error" not in payload, payload.get("error")
    assert payload["billed_seconds"] == pytest.approx(expected_seconds, abs=0.2)
    # The appended footage is what it is NOT: conflating them was the bug.
    assert payload["appended_seconds"] == times * 7
    assert payload["estimated_cost"]["usd"] == pytest.approx(
        expected_seconds * 0.40, abs=0.1
    )


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
async def test_a_remotely_delivered_chain_is_never_reported_as_metered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A GCS-delivered render can never be is_estimate: false.

    The chain returned is_estimate: false for a delivery the server never
    opened — wrong twice over, since the basis was also the appended footage
    rather than the assembled length.
    """
    from src.__main__ import loop_extend

    async def gcs_impl(**kwargs: Any) -> dict[str, Any]:
        return {
            "video_url": "gs://bkt/out/ext.mp4",
            "model": kwargs.get("model"),
            "duration_seconds": 7,
        }

    monkeypatch.setattr("src.__main__.generate_video_impl", gcs_impl)

    payload = json.loads(
        await loop_extend(
            ctx=_ctx(
                tmp_path,
                vertexai=True,
                bucket="gs://bkt/out/",
                allowed=frozenset({"bkt"}),
            ),
            prompt="c",
            video_uri="gs://bkt/in.mp4",
            model=VEO,
            times=1,
        )
    )
    assert "error" not in payload, payload.get("error")
    manifest = payload.get("manifest") or {}
    cost = payload.get("cost") or manifest.get("cost") or {}
    assert cost, payload
    assert cost["is_estimate"] is True, cost
    assert "FLOOR" in manifest.get("duration_source", "")


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
async def test_a_remote_delivery_still_projects_from_a_local_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real run must never report less than its own pre-flight.

    Identical call, local 4s source: the dry run measured it and quoted 11.0s,
    then the real run delivered to GCS, could not open the turns, and fell all
    the way back to appended-only — $0.70 against its own $1.10 quote. Only
    the OUTPUT went remote; the source was still sitting there.
    """
    from src.__main__ import loop_extend

    src_video = tmp_path / "videos" / "src.mp4"
    _write_video(src_video, 1280, 720, frames=96)  # 4.0s

    async def gcs_impl(**kwargs: Any) -> dict[str, Any]:
        return {
            "video_url": "gs://bkt/out/ext.mp4",
            "model": kwargs.get("model"),
            "duration_seconds": 7,
        }

    monkeypatch.setattr("src.__main__.generate_video_impl", gcs_impl)
    ctx = _ctx(
        tmp_path, vertexai=True, bucket="gs://bkt/out/", allowed=frozenset({"bkt"})
    )
    kwargs: dict[str, Any] = dict(
        prompt="c", video_uri=f"file://{src_video}", model=VEO_FAST, times=1
    )
    quote = json.loads(await loop_extend(ctx=ctx, dry_run=True, **kwargs))
    real = json.loads(await loop_extend(ctx=ctx, **kwargs))
    manifest = real.get("manifest") or {}

    assert quote["billed_seconds"] == pytest.approx(11.0, abs=0.2)
    assert manifest["billed_seconds"] == pytest.approx(quote["billed_seconds"], abs=0.2)
    assert manifest["duration_source"].startswith("PROJECTED")
    # Projected is not metered, however good the projection is.
    assert manifest["cost"]["is_estimate"] is True


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
@pytest.mark.parametrize("vertexai", [False, True])
async def test_one_response_names_one_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, vertexai: bool
) -> None:
    """`model` and `cost.detail` must never name different models.

    The impl returns the SERVED id — the Gemini API is fed `-preview`
    spellings — and reporting that beside a cost line priced on the canonical
    name put two model names in one response. This was fixed once for the
    quote and grew back on the rendered path, which is why it is settled in
    one place now rather than per tool.
    """
    from src.__main__ import generate_video
    from src.video import _GEMINI_API_MODEL_IDS  # pyright: ignore[reportPrivateUsage]

    out = tmp_path / "videos" / "m.mp4"

    async def veo_impl(**kwargs: Any) -> dict[str, Any]:
        _write_video(out, 1280, 720)
        served = _GEMINI_API_MODEL_IDS.get(kwargs["model"], kwargs["model"])
        return {"video_url": f"file://{out}", "model": served, "duration_seconds": 4}

    monkeypatch.setattr("src.__main__.generate_video_impl", veo_impl)
    payload = json.loads(
        await generate_video(
            ctx=_ctx(tmp_path, vertexai=vertexai), prompt="x", model=VEO_FAST
        )
    )
    assert "error" not in payload, payload.get("error")
    assert payload["model"] == VEO_FAST
    assert payload["model"] in payload["cost"]["detail"]
    assert "-preview" not in payload["cost"]["detail"]
    if payload.get("served_model"):
        # The wire spelling is kept, not dropped.
        assert payload["served_model"].endswith("-preview")


@pytest.mark.asyncio
@pytest.mark.timeout(20.0)
async def test_reference_images_are_refused_on_the_gemini_api(
    tmp_path: Path,
) -> None:
    """It returned an empty result with no usable error, and billed for it.

    "No videos returned" told a caller nothing, cost roughly $0.80, and left
    reference_to_video looking like a transient failure rather than a mode
    this backend does not serve.
    """
    from src.__main__ import generate_video

    _, image = _seed_media(tmp_path)
    payload = json.loads(
        await generate_video(
            ctx=_ctx(tmp_path),
            prompt="x",
            model=VEO_FAST,
            reference_image_uris=[image],
            dry_run=True,
        )
    )
    assert "estimated_cost" not in payload
    assert "reference_to_video" in payload["error"]
    assert "Vertex AI" in payload["error"]


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
async def test_image_to_video_is_not_warned_about_on_the_gemini_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It renders fine there, and a warning on a working call is noise.

    The caution was written when the backend was believed text-to-video only.
    It was not, and warnings that fire on success teach callers to skip them.
    """
    from src.__main__ import generate_video

    _, image = _seed_media(tmp_path)
    out = tmp_path / "videos" / "i2v.mp4"

    async def veo_impl(**kwargs: Any) -> dict[str, Any]:
        _write_video(out, 1280, 720)
        return {
            "video_url": f"file://{out}",
            "model": kwargs.get("model"),
            "duration_seconds": 4,
        }

    monkeypatch.setattr("src.__main__.generate_video_impl", veo_impl)
    payload = json.loads(
        await generate_video(
            ctx=_ctx(tmp_path), prompt="x", model=VEO_FAST, image_uri=image
        )
    )
    assert "error" not in payload, payload.get("error")
    assert not [w for w in payload.get("warnings", []) if "text-to-video only" in w], (
        payload.get("warnings")
    )


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
async def test_generate_videos_extend_mode_bills_the_assembled_clip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The second extension entry point had none of loop_extend's fixes.

    Same service, same rule, separate implementation: it billed the ~7s it
    appends rather than the assembled clip, asserted is_estimate=False for a
    gs:// file it never opened, and labelled the figure with the TEXT-TO-VIDEO
    rule ("Veo renders exactly the length it is sent") in the one mode where
    that rule is known to be false. A confidently wrong source is worse than
    an absent one, because it will be trusted.
    """
    from src.__main__ import generate_video

    src_video = tmp_path / "videos" / "src.mp4"
    _write_video(src_video, 1280, 720, frames=96)  # 4.0s

    async def gcs_impl(**kwargs: Any) -> dict[str, Any]:
        return {
            "video_url": "gs://bkt/out/sample_0.mp4",
            "model": kwargs.get("model"),
            "duration_seconds": 7,
            "generation_mode": "extend_video",
        }

    monkeypatch.setattr("src.__main__.generate_video_impl", gcs_impl)
    payload = json.loads(
        await generate_video(
            ctx=_ctx(
                tmp_path,
                vertexai=True,
                bucket="gs://bkt/out/",
                allowed=frozenset({"bkt"}),
            ),
            prompt="c",
            model=VEO_FAST,
            extend_video_uri=f"file://{src_video}",
        )
    )
    assert "error" not in payload, payload.get("error")
    assert payload["billed_seconds"] == pytest.approx(11.0, abs=0.2)
    assert payload["appended_seconds"] == 7.0
    assert payload["cost"]["is_estimate"] is True
    assert payload["cost"]["usd"] == pytest.approx(1.10, abs=0.05)
    # The text-to-video rule must not be asserted over an extension.
    assert "exactly the length it is sent" not in payload["duration_source"]


@pytest.mark.asyncio
@pytest.mark.timeout(20.0)
async def test_veo_4k_is_refused_on_the_gemini_api_before_it_is_priced(
    tmp_path: Path,
) -> None:
    """Gating covered modes but not resolutions.

    This quoted $1.20 at the 4K rate and then 400'd — the highest-value false
    quote in the server. Veo-specific: omni renders 4K on this backend.
    """
    from src.__main__ import generate_video

    payload = json.loads(
        await generate_video(
            ctx=_ctx(tmp_path),
            prompt="x",
            model=VEO_FAST,
            resolution="4K",
            dry_run=True,
        )
    )
    assert "estimated_cost" not in payload
    assert "4K" in payload["error"]
    assert "Vertex" in payload["error"] or "720p" in payload["error"]


@pytest.mark.asyncio
@pytest.mark.timeout(20.0)
async def test_a_clip_with_bridges_is_refused_before_any_beat_is_billed(
    tmp_path: Path,
) -> None:
    """A partial-spend trap, made worse by a pre-flight that reassured first.

    On the Gemini API the quote came back clean with bridge_count: 1 and a
    positive ffmpeg check, the beats then rendered and billed ~$0.80, and only
    the first bridge failed. Standalone bridges already refused on that
    backend; the composite that builds them did not.
    """
    from src.__main__ import generate_clip

    payload = json.loads(
        await generate_clip(
            ctx=_ctx(tmp_path),
            beats=[{"prompt": "a"}, {"prompt": "b"}],
            model=VEO_FAST,
            add_bridges=True,
            dry_run=True,
        )
    )
    assert "estimated_cost" not in payload
    assert "first_last_frame" in payload["error"]


def test_a_thought_signature_the_server_wrote_can_be_read_back() -> None:
    """The cap refused every signature the server itself had just written.

    MEASURED: 2,828,040 bytes from gemini-3-pro-image and 1,634,388 from
    gemini-3.1-flash-image, against a 256 KB cap — so the docstring's own
    two-step editing example could not execute on either model. The cap still
    has to refuse the 157 MB render that motivated it.
    """
    from src.__main__ import MAX_THOUGHT_SIGNATURE_BYTES

    assert MAX_THOUGHT_SIGNATURE_BYTES >= 2_828_040
    assert MAX_THOUGHT_SIGNATURE_BYTES < 157 * 1024 * 1024


def test_no_docstring_claims_extension_is_served_inline_on_the_gemini_api() -> None:
    """The mode is gated there and the service refuses it.

    A docstring that contradicts a gate sends a caller at a wall the server
    already knows about.
    """
    import pathlib

    for path in sorted(pathlib.Path("src").glob("*.py")):
        text = path.read_text()
        assert "the extended clip is" not in text or "returned inline" not in text, path


@pytest.mark.asyncio
@pytest.mark.timeout(20.0)
async def test_generate_videos_extend_quote_bills_the_assembled_clip(
    tmp_path: Path,
) -> None:
    """The real run was corrected and the quote beside it was not.

    A 4s source, extend_video_uri, dry_run: 7s / $0.70 against a real run
    that now reports 11s / $1.10 — the quote BELOW its own render. Both ends
    go through veo_extension_billing now.
    """
    from src.__main__ import generate_video

    src_video = tmp_path / "videos" / "src.mp4"
    _write_video(src_video, 1280, 720, frames=96)  # 4.0s
    payload = json.loads(
        await generate_video(
            ctx=_ctx(
                tmp_path,
                vertexai=True,
                bucket="gs://bkt/out/",
                allowed=frozenset({"bkt"}),
            ),
            prompt="c",
            model=VEO_FAST,
            extend_video_uri=f"file://{src_video}",
            dry_run=True,
        )
    )
    assert "error" not in payload, payload.get("error")
    assert payload["billed_seconds"] == pytest.approx(11.0, abs=0.2)
    assert payload["duration_seconds"] == pytest.approx(11.0, abs=0.2)
    assert payload["estimated_cost"]["usd"] == pytest.approx(1.10, abs=0.05)
    assert "nothing to measure" not in payload["duration_source"]


@pytest.mark.asyncio
@pytest.mark.timeout(20.0)
@pytest.mark.parametrize("tool", ["generate_transition", "generate_bridge"])
async def test_first_last_frame_quotes_apply_the_gcs_allowlist(
    tmp_path: Path, tool: str
) -> None:
    """generate_video's quote had GCS resolution hoisted above it; its two
    siblings did not, so their quotes skipped the allowlist and backend checks
    the real call applies — a price for a destination the render refuses."""
    import src.__main__ as main_mod

    video, image = _seed_media(tmp_path)
    kwargs: dict[str, Any] = (
        dict(first_frame_uri=image, last_frame_uri=image)
        if tool == "generate_transition"
        else dict(from_clip_uri=video, to_clip_uri=video)
    )
    payload = json.loads(
        await getattr(main_mod, tool)(
            ctx=_ctx(tmp_path, vertexai=True, allowed=frozenset({"trusted"})),
            prompt="x",
            model=VEO_FAST,
            output_gcs_uri="gs://other/out/",
            dry_run=True,
            **kwargs,
        )
    )
    assert "estimated_cost" not in payload, f"{tool} priced a refused destination"
    assert "not in the allowlist" in payload["error"]


@pytest.mark.asyncio
@pytest.mark.timeout(20.0)
async def test_the_impl_refuses_veo_4k_on_the_gemini_api_before_the_wire() -> None:
    """The dry run refused it; the real call let it through to a 400.

    Same rule, both ends — the impl's resolution check now sees the backend.
    """
    from unittest.mock import MagicMock

    from src.video import generate_video as impl

    client = MagicMock()
    client._api_client.vertexai = False
    with pytest.raises(ValueError, match="4K"):
        await impl(
            client=client, prompt="x", videos_dir=Path("."), model=VEO, resolution="4K"
        )
    client.models.generate_videos.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.timeout(20.0)
async def test_a_local_probe_reads_the_header_and_respects_the_sandbox(
    tmp_path: Path,
) -> None:
    """Probing went through the bytes fetch, loading up to 50 MB to read a
    header. Path-based now, with the same containment as every other
    caller-supplied path."""
    from src.__main__ import _probe_local_video_seconds

    video, _ = _seed_media(tmp_path)  # 2.0s
    ctx = _ctx(tmp_path)
    assert await _probe_local_video_seconds(ctx, video) == pytest.approx(2.0, abs=0.05)
    outside = tmp_path.parent / "outside.mp4"
    _write_video(outside, 64, 64)
    try:
        assert await _probe_local_video_seconds(ctx, f"file://{outside}") is None
        assert await _probe_local_video_seconds(ctx, "gs://bkt/a.mp4") is None
    finally:
        outside.unlink(missing_ok=True)
