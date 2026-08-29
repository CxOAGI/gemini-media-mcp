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


# --------------------------------------------------------------------------
# a remote extension source
# --------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
@pytest.mark.parametrize(
    ("tool", "uri_kwarg"),
    [("generate_video", "extend_video_uri"), ("loop_extend", "video_uri")],
)
async def test_a_real_run_measures_a_remote_extension_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool: str, uri_kwarg: str
) -> None:
    """On Vertex the source is usually gs://, and that was the loose case.

    Extension REQUIRES a GCS destination on Vertex, so the source is normally
    remote too — and a remote source fell straight to the appended-only floor:
    7s / $0.70 against a 4s source whose extension bills 11s / $1.10. A real
    run is already committing to a render, so a capped read of the source is
    what turns that floor into a projection. Both chains, one helper.
    """
    import src.__main__ as main_mod

    real = _write_video(tmp_path / "videos" / "src.mp4", seconds=4.0)
    source_bytes = Path(real[7:]).read_bytes()

    async def fake_fetch(uri: str, **kwargs: Any) -> bytes | None:
        assert uri.startswith("gs://")
        return source_bytes

    async def impl(**kwargs: Any) -> dict[str, Any]:
        return {
            "video_url": "gs://bkt/out/x.mp4",
            "model": kwargs.get("model"),
            "duration_seconds": 7,
            "generation_mode": "extend_video",
        }

    monkeypatch.setattr(main_mod, "fetch", fake_fetch)
    monkeypatch.setattr(main_mod, "generate_video_impl", impl)

    kwargs: dict[str, Any] = {uri_kwarg: "gs://bkt/in/src.mp4", "model": VEO}
    if tool == "generate_video":
        kwargs["prompt"] = "c"
    else:
        kwargs["prompt"] = "c"
        kwargs["times"] = 1
    payload = json.loads(await getattr(main_mod, tool)(ctx=_ctx(tmp_path), **kwargs))
    assert "error" not in payload, payload.get("error")
    manifest = payload.get("manifest") or {}
    billed = payload.get("billed_seconds", manifest.get("billed_seconds"))
    # The BILLING basis has its own key: duration_source describes
    # duration_seconds, which is the artifact's length, not what was charged.
    source = payload.get(
        "billed_seconds_source", manifest.get("billed_seconds_source", "")
    )
    assert billed == pytest.approx(11.0, abs=0.2), payload
    assert source.startswith("PROJECTED"), source
    cost = payload.get("cost") or manifest.get("cost") or {}
    assert cost["usd"] == pytest.approx(4.40, abs=0.05)  # 11s at veo-3.1 $0.40/s
    assert cost["is_estimate"] is True


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
async def test_a_dry_run_stays_offline_and_says_the_real_run_will_measure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A quote is documented free, instant and offline, so it must not
    download a remote source — but it must not pass its floor off as a price
    either. It says FLOOR, and the real run measures."""
    import src.__main__ as main_mod

    called: list[str] = []

    async def fake_fetch(uri: str, **kwargs: Any) -> bytes | None:
        called.append(uri)
        return None

    monkeypatch.setattr(main_mod, "fetch", fake_fetch)
    payload = json.loads(
        await main_mod.generate_video(
            ctx=_ctx(tmp_path),
            prompt="c",
            model=VEO,
            extend_video_uri="gs://bkt/in/src.mp4",
            dry_run=True,
        )
    )
    assert called == [], "a dry run downloaded a remote source"
    assert payload["billed_seconds_source"].startswith("FLOOR")


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
async def test_extension_billing_does_not_depend_on_the_impl_echoing_a_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rule was keyed on generation_mode coming back in the response.

    A caller who passed extend_video_uri IS extending; if that field were
    ever absent the response silently reverted to the pre-fix numbers —
    7s billed AND is_estimate: false, which is exactly the pair reported
    from the field. Keyed on the request now.
    """
    import src.__main__ as main_mod

    real = _write_video(tmp_path / "videos" / "src.mp4", seconds=4.0)

    async def impl_without_mode(**kwargs: Any) -> dict[str, Any]:
        return {
            "video_url": "gs://bkt/out/x.mp4",
            "model": kwargs.get("model"),
            "duration_seconds": 7,
        }

    monkeypatch.setattr(main_mod, "generate_video_impl", impl_without_mode)
    payload = json.loads(
        await main_mod.generate_video(
            ctx=_ctx(tmp_path), prompt="c", model=VEO, extend_video_uri=real
        )
    )
    assert payload["billed_seconds"] == pytest.approx(11.0, abs=0.2)
    assert payload["cost"]["is_estimate"] is True


def test_no_docstring_still_prices_a_chain_at_times_times_seven() -> None:
    """The basis moved to the assembled clip; the prose has to move with it.

    A docstring that still says "times x ~7s" teaches the caller the number
    the code was corrected away from.
    """
    import pathlib

    for path in sorted(pathlib.Path("src").glob("*.py")):
        text = " ".join(path.read_text().split())
        assert "times x ~7s at the model" not in text, path
        assert "1527 input tokens" not in text, path


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
async def test_an_unmeasurable_extension_never_borrows_the_text_to_video_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ "Veo renders exactly the length it is sent" is text-to-video's rule.

    An extension delivers the assembled clip, so that sentence is false in the
    one mode it was being applied to — and a confidently wrong source gets
    trusted where an absent one would not. It reappeared through the fallback
    when neither the render nor its source could be measured.
    """
    import src.__main__ as main_mod

    async def unreadable(uri: str, **kwargs: Any) -> bytes | None:
        return None

    async def impl(**kwargs: Any) -> dict[str, Any]:
        return {
            "video_url": "gs://bkt/out/x.mp4",
            "model": kwargs.get("model"),
            "duration_seconds": 7,
            "generation_mode": "extend_video",
        }

    monkeypatch.setattr(main_mod, "fetch", unreadable)
    monkeypatch.setattr(main_mod, "generate_video_impl", impl)
    payload = json.loads(
        await main_mod.generate_video(
            ctx=_ctx(tmp_path),
            prompt="c",
            model=VEO,
            extend_video_uri="gs://bkt/in/src.mp4",
        )
    )
    assert "exactly the length it is sent" not in payload["duration_source"]
    assert "not measured" in payload["duration_source"]
    assert payload["cost"]["is_estimate"] is True


# --------------------------------------------------------------------------
# the invariant, and its one documented exception
# --------------------------------------------------------------------------


def test_the_one_place_a_quote_can_under_state_is_documented_in_prose() -> None:
    """The campaign's invariant is "may over-state, never under-state".

    An extension from a remote source breaks it — deliberately, because a dry
    run is offline and Veo publishes no maximum source length to assume in
    place of measuring. That exception was only inferable from a runtime
    field (billed_seconds_source), which is not where a caller looks before
    committing to a call. It belongs in the docstring, and it has to stay
    there: this is the check that notices if it is edited away.
    """
    import inspect

    from src.__main__ import generate_video, loop_extend

    for tool in (generate_video, loop_extend):
        doc = " ".join((inspect.getdoc(tool) or "").split())
        assert "FLOOR" in doc, tool.__name__
        assert "under-state" in doc, tool.__name__
        # The concrete measurement, so the size of the gap is not abstract.
        assert "$1.80" in doc or "61%" in doc, tool.__name__
        # And the way out, so the note is actionable rather than a caveat.
        assert "local copy" in doc, tool.__name__


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
async def test_both_provenance_fields_name_the_measurement_they_share(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """duration_source said "projected"; billed_seconds_source said
    "PROJECTED from a measured 11.01s source". They agreed on the number and
    only one said where it came from, which reads like disagreement."""
    import src.__main__ as main_mod

    real = _write_video(tmp_path / "videos" / "src.mp4", seconds=4.0)
    source_bytes = Path(real[7:]).read_bytes()

    async def fake_fetch(uri: str, **kwargs: Any) -> bytes | None:
        return source_bytes

    async def impl(**kwargs: Any) -> dict[str, Any]:
        return {
            "video_url": "gs://bkt/out/x.mp4",
            "model": kwargs.get("model"),
            "duration_seconds": 7,
            "generation_mode": "extend_video",
        }

    monkeypatch.setattr(main_mod, "fetch", fake_fetch)
    monkeypatch.setattr(main_mod, "generate_video_impl", impl)
    payload = json.loads(
        await main_mod.generate_video(
            ctx=_ctx(tmp_path),
            prompt="c",
            model=VEO,
            extend_video_uri="gs://bkt/in/src.mp4",
        )
    )
    for field in ("duration_source", "billed_seconds_source"):
        assert "measured 4s source" in payload[field], (field, payload[field])

    # And the quote, from a local source, does the same.
    quote = json.loads(
        await main_mod.generate_video(
            ctx=_ctx(tmp_path),
            prompt="c",
            model=VEO,
            extend_video_uri=real,
            dry_run=True,
        )
    )
    assert "measured 4s source" in quote["duration_source"], quote["duration_source"]
