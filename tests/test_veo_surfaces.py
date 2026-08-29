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
    assert not missing, "tools not reporting their backend:\n" + "\n".join(missing)


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
