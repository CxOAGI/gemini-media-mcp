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
