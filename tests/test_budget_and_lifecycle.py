"""Guards that keep a call from spending what its caller did not sign up for.

Three of them, from one review:

  * a cost cap on the two extension chains, because their bills grow with
    the square of `times` and `times` accepts 20;
  * a date-aware refusal of gemini-omni-flash-preview on its sunset, because
    from that day a deprecation warning is a lie beside a dead endpoint;
  * an actionable error when Vertex refuses omni 1.1, because the raw 404 says
    nothing about the allowlist, the GA route one credential away, or the
    preview model that still answers until the sunset.

Plus the omni docs' own prompting advice: negatives inline, not dropped.
"""

import datetime
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.__main__ import AppContext

VEO = "veo-3.1-generate-001"


def _ctx(tmp_path: Path, *, vertexai: bool = True) -> Any:
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
        gemini_api_client=client,
        video_gcs_bucket="gs://bkt/out/" if vertexai else None,
        allowed_gcs_buckets=frozenset({"bkt"}),
    )
    return ctx


def _write_video(path: Path, seconds: float = 2.0) -> str:
    import imageio.v3 as iio
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(24 * seconds)
    iio.imwrite(
        path,
        [np.full((360, 640, 3), (i * 7) % 256, dtype=np.uint8) for i in range(frames)],
        extension=".mp4",
        fps=24,
    )
    return f"file://{path}"


# --------------------------------------------------------------------------
# max_cost_usd
# --------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
async def test_loop_extend_refuses_before_the_first_turn_bills(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 2s source, three turns: 9 + 16 + 23 = 48s at $0.40 is $19.20.

    With a $5 cap the tool must refuse BEFORE the impl is called — a refusal
    after turn one has already spent the money it was meant to protect.
    """
    from src.__main__ import loop_extend

    video = _write_video(tmp_path / "videos" / "src.mp4")
    calls: list[dict[str, Any]] = []

    async def impl(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"video_url": "gs://bkt/out/x.mp4", "model": VEO, "duration_seconds": 7}

    monkeypatch.setattr("src.__main__.generate_video_impl", impl)
    payload = json.loads(
        await loop_extend(
            ctx=_ctx(tmp_path),
            prompt="c",
            video_uri=video,
            model=VEO,
            times=3,
            max_cost_usd=5.0,
        )
    )
    assert "Refused before rendering" in payload["error"]
    assert "$19.20" in payload["error"]
    assert calls == [], "the cap fired after a turn had billed"

    # Under the cap the chain runs.
    ok = json.loads(
        await loop_extend(
            ctx=_ctx(tmp_path),
            prompt="c",
            video_uri=video,
            model=VEO,
            times=1,
            max_cost_usd=5.0,
        )
    )
    assert "error" not in ok, ok.get("error")
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.timeout(20.0)
async def test_a_dry_run_reports_would_be_refused_rather_than_refusing(
    tmp_path: Path,
) -> None:
    """The quote's whole point is to show the number; hiding it behind an
    error defeats it. It says the call would be refused, and why."""
    from src.__main__ import loop_extend

    video = _write_video(tmp_path / "videos" / "src.mp4")
    payload = json.loads(
        await loop_extend(
            ctx=_ctx(tmp_path),
            prompt="c",
            video_uri=video,
            model=VEO,
            times=3,
            max_cost_usd=5.0,
            dry_run=True,
        )
    )
    assert "error" not in payload, payload.get("error")
    assert payload["would_be_refused"] is True
    assert payload["estimated_cost"]["usd"] == pytest.approx(19.20, abs=0.05)
    assert any("max_cost_usd" in w for w in payload["warnings"])


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
async def test_extend_video_omni_refuses_before_the_first_turn_bills(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same quadratic hazard on the omni chain, same cap, same ordering."""
    from src.__main__ import extend_video_omni

    video = _write_video(tmp_path / "videos" / "src.mp4", seconds=3.0)
    calls: list[dict[str, Any]] = []

    async def impl(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {
            "video_url": "gs://bkt/out/x.mp4",
            "model": kwargs.get("model"),
            "duration_seconds": 13,
            "interaction_id": "i-1",
        }

    monkeypatch.setattr("src.__main__._omni_generate_and_manifest", impl)
    payload = json.loads(
        await extend_video_omni(
            ctx=_ctx(tmp_path, vertexai=False),
            prompt="c",
            input_video_uri=video,
            times=3,
            max_cost_usd=0.50,
        )
    )
    assert "Refused before rendering" in payload["error"], payload
    assert calls == []

    quote = json.loads(
        await extend_video_omni(
            ctx=_ctx(tmp_path, vertexai=False),
            prompt="c",
            input_video_uri=video,
            times=3,
            max_cost_usd=0.50,
            dry_run=True,
        )
    )
    assert quote["would_be_refused"] is True


# --------------------------------------------------------------------------
# sunset
# --------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(20.0)
async def test_the_preview_model_is_refused_from_its_sunset_day(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Until the day, a warning; from the day, a refusal at the pre-flight.

    A dry run against a switched-off endpoint used to quote a price for a
    404. The refusal names the successor and where it is GA.
    """
    import src.omni as omni_mod
    from src.__main__ import generate_video_omni

    monkeypatch.setattr(omni_mod, "_today", lambda: datetime.date(2026, 9, 29))
    before = json.loads(
        await generate_video_omni(
            ctx=_ctx(tmp_path, vertexai=False),
            prompt="x",
            omni_model="gemini-omni-flash-preview",
            dry_run=True,
        )
    )
    assert "error" not in before, before.get("error")

    monkeypatch.setattr(omni_mod, "_today", lambda: datetime.date(2026, 9, 30))
    after = json.loads(
        await generate_video_omni(
            ctx=_ctx(tmp_path, vertexai=False),
            prompt="x",
            omni_model="gemini-omni-flash-preview",
            dry_run=True,
        )
    )
    assert "switched off on 2026-09-30" in after["error"]
    assert "gemini-omni-1.1-flash" in after["error"]


def test_the_planner_stops_offering_the_preview_model_from_its_sunset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Demoted until the day; excluded from it, with the successor named."""
    import src.omni as omni_mod
    from src.routing import RoutingConstraints, plan_generation

    monkeypatch.setattr(omni_mod, "_today", lambda: datetime.date(2026, 10, 1))
    plan = plan_generation(
        "a 6 second video of rain", RoutingConstraints(backend="gemini_api")
    )
    offered = {r.model for r in plan.routes}
    assert "gemini-omni-flash-preview" not in offered
    assert "gemini-omni-1.1-flash" in offered
    assert any(
        r.model == "gemini-omni-flash-preview" and "switched off" in r.reason
        for r in plan.rejected
    )


# --------------------------------------------------------------------------
# access refusal
# --------------------------------------------------------------------------


def test_a_vertex_404_for_omni_says_allowlist_and_names_the_ga_route() -> None:
    """The raw error names an endpoint and a project. The useful facts are
    that the model is Preview-gated on Vertex, GA one credential away, and
    that the preview model still answers until its sunset."""
    from src.omni import _access_refusal  # pyright: ignore[reportPrivateUsage]

    exc = RuntimeError(
        "404 NOT_FOUND: Publisher Model `gemini-omni-1.1-flash-preview` not found"
    )
    advice = _access_refusal(exc, "gemini-omni-1.1-flash-preview", vertexai=True)
    assert advice is not None
    text = str(advice)
    assert "allowlist" in text
    assert "GEMINI_API_KEY" in text
    assert "gemini-omni-flash-preview" in text  # still an option before the sunset

    assert (
        _access_refusal(RuntimeError("deadline exceeded"), "m", vertexai=True) is None
    )
    on_gemini = _access_refusal(exc, "gemini-omni-1.1-flash", vertexai=False)
    assert on_gemini is not None and "Interactions API" in str(on_gemini)


# --------------------------------------------------------------------------
# draft negatives
# --------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
async def test_a_draft_folds_the_negative_prompt_inline_instead_of_dropping_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Omni has no negative_prompt field; its docs say to write negatives
    inline ("No dialogue"). A draft that ignored the caller's exclusions
    previewed the wrong shot and then said so in a warning."""
    from src.__main__ import generate_video

    sent: list[dict[str, Any]] = []
    out = tmp_path / "videos" / "d.mp4"

    async def impl(*args: Any, **kwargs: Any) -> dict[str, Any]:
        sent.append(kwargs)
        _write_video(out)
        return {
            "video_url": f"file://{out}",
            "model": kwargs.get("model"),
            "duration_seconds": 6,
            "interaction_id": "i-1",
        }

    monkeypatch.setattr("src.__main__._omni_generate_and_manifest", impl)
    payload = json.loads(
        await generate_video(
            ctx=_ctx(tmp_path, vertexai=False),
            prompt="a quiet street",
            model=VEO,
            draft=True,
            negative_prompt="rain",
        )
    )
    assert "error" not in payload, payload.get("error")
    assert sent and sent[0]["prompt"].endswith("No rain.")
    assert payload["negative_prompt_inlined"] == "rain"
    assert "negative_prompt" not in payload.get("ignored_veo_params", [])
    assert any("folded into the prompt" in w for w in payload["warnings"])
