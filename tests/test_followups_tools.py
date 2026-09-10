"""Regression tests for the video-tool warning channel, dry-run disclosure,
storyboard sheet de-duplication, and the malformed-credential guard.

Grouped here (rather than in test_main.py) so the follow-up fixes stay
self-contained: each test fails against the pre-fix code and passes after.
"""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from src.__main__ import AppContext, setup_vertex_credentials


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _png_bytes(w: int = 320, h: int = 180, color: str = "steelblue") -> bytes:
    buf = BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="PNG")
    return buf.getvalue()


def _app_ctx(tmp_path: Path, *, vertexai: bool = False) -> AppContext:
    (tmp_path / "images").mkdir(exist_ok=True)
    (tmp_path / "videos").mkdir(exist_ok=True)
    client = MagicMock()
    client._api_client.vertexai = vertexai
    return AppContext(
        data_folder=tmp_path,
        images_dir=tmp_path / "images",
        videos_dir=tmp_path / "videos",
        client=client,
    )


def _ctx(app_ctx: AppContext) -> Any:
    ctx = MagicMock()
    ctx.info = AsyncMock()
    ctx.error = AsyncMock()
    ctx.warning = AsyncMock()
    ctx.request_context.lifespan_context = app_ctx
    return ctx


def _emitted(ctx: Any) -> list[str]:
    """The distinct warning strings pushed onto the notification channel."""
    return [call.args[0] for call in ctx.warning.await_args_list]


def _video_result(video_url: str, warnings: list[str] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "message": "Video generated successfully",
        "video_url": video_url,
        "model": "veo-3.1-fast-generate-001",
        "audio_enabled": False,
        "duration_seconds": 4,
    }
    if warnings is not None:
        result["warnings"] = warnings
    return result


def _omni_result(video_url: str, warnings: list[str] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "message": "Video generated successfully",
        "video_url": video_url,
        "interaction_id": "int-1",
        "model": "gemini-omni-flash-preview",
        "duration_seconds": 6,
        "aspect_ratio": "16:9",
    }
    if warnings is not None:
        result["warnings"] = warnings
    return result


_AUDIO_WARNING = (
    "include_audio was not honored: the Gemini API path does not expose audio."
)


# ===========================================================================
# Defect 1 — warnings must reach the MCP notification channel on video tools
# ===========================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_generate_video_emits_warnings_to_channel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.__main__ import generate_video

    ctx = _ctx(_app_ctx(tmp_path))
    out = tmp_path / "videos" / "v.mp4"
    out.write_bytes(b"mp4")

    async def mock_impl(**kwargs: Any) -> dict[str, Any]:
        return _video_result(f"file://{out}", warnings=[_AUDIO_WARNING])

    monkeypatch.setattr("src.__main__.generate_video_impl", mock_impl)

    result = json.loads(
        await generate_video(ctx=ctx, prompt="a cat", model="veo-3.1-fast-generate-001")
    )
    assert result["warnings"] == [_AUDIO_WARNING]  # still in the body
    assert _AUDIO_WARNING in _emitted(ctx)  # and on the channel


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_generate_transition_emits_warnings_to_channel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.__main__ import generate_transition

    ctx = _ctx(_app_ctx(tmp_path))
    (tmp_path / "first.png").write_bytes(_png_bytes())
    (tmp_path / "last.png").write_bytes(_png_bytes(color="tomato"))
    out = tmp_path / "videos" / "t.mp4"
    out.write_bytes(b"mp4")

    async def mock_impl(**kwargs: Any) -> dict[str, Any]:
        return _video_result(f"file://{out}", warnings=[_AUDIO_WARNING])

    monkeypatch.setattr("src.__main__.generate_video_impl", mock_impl)

    await generate_transition(
        ctx=ctx,
        first_frame_uri=f"file://{tmp_path}/first.png",
        last_frame_uri=f"file://{tmp_path}/last.png",
        include_audio=True,
    )
    assert _AUDIO_WARNING in _emitted(ctx)


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_generate_bridge_emits_warnings_to_channel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.__main__ import generate_bridge

    ctx = _ctx(_app_ctx(tmp_path))
    (tmp_path / "a.mp4").write_bytes(b"clipA")
    (tmp_path / "b.mp4").write_bytes(b"clipB")
    out = tmp_path / "videos" / "br.mp4"
    out.write_bytes(b"mp4")

    monkeypatch.setattr("src.__main__.assert_frame_decoding_available", lambda: None)
    monkeypatch.setattr("src.__main__.extract_frame_png", lambda *a, **k: _png_bytes())

    async def mock_impl(**kwargs: Any) -> dict[str, Any]:
        return _video_result(f"file://{out}", warnings=[_AUDIO_WARNING])

    monkeypatch.setattr("src.__main__.generate_video_impl", mock_impl)

    await generate_bridge(
        ctx=ctx,
        from_clip_uri=f"file://{tmp_path}/a.mp4",
        to_clip_uri=f"file://{tmp_path}/b.mp4",
        include_audio=True,
    )
    assert _AUDIO_WARNING in _emitted(ctx)


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_loop_extend_emits_warnings_to_channel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.__main__ import loop_extend

    # Vertex: Veo refuses extension on the Gemini Developer API outright
    # ("encodedVideo isn't supported by this model"), so a chain cannot reach
    # the point of emitting anything there. What this test is about — a
    # warning from a chained impl reaching the notification channel — is not
    # about a backend, so it runs where the chain can run.
    app_ctx = _app_ctx(tmp_path, vertexai=True)
    object.__setattr__(app_ctx, "video_gcs_bucket", "gs://bkt/out/")
    object.__setattr__(app_ctx, "allowed_gcs_buckets", frozenset({"bkt"}))
    ctx = _ctx(app_ctx)
    src_video = tmp_path / "videos" / "src.mp4"
    src_video.write_bytes(b"mp4")
    out = tmp_path / "videos" / "ext.mp4"
    out.write_bytes(b"mp4")

    async def mock_impl(**kwargs: Any) -> dict[str, Any]:
        return _video_result(f"file://{out}", warnings=[_AUDIO_WARNING])

    monkeypatch.setattr("src.__main__.generate_video_impl", mock_impl)

    await loop_extend(ctx=ctx, video_uri=f"file://{src_video}", times=1)
    assert _AUDIO_WARNING in _emitted(ctx)


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_generate_clip_emits_beat_warnings_to_channel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.__main__ import generate_clip

    ctx = _ctx(_app_ctx(tmp_path))
    out = tmp_path / "videos" / "beat.mp4"
    out.write_bytes(b"mp4")

    async def mock_impl(**kwargs: Any) -> dict[str, Any]:
        return _video_result(f"file://{out}", warnings=[_AUDIO_WARNING])

    monkeypatch.setattr("src.__main__.generate_video_impl", mock_impl)

    await generate_clip(
        ctx=ctx,
        beats=[{"prompt": "a"}],
        model="veo-3.1-fast-generate-001",
    )
    assert _AUDIO_WARNING in _emitted(ctx)


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_generate_video_omni_emits_warnings_to_channel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.__main__ import generate_video_omni

    ctx = _ctx(_app_ctx(tmp_path))
    out = tmp_path / "videos" / "o.mp4"
    out.write_bytes(b"mp4")

    async def mock_impl(**kwargs: Any) -> dict[str, Any]:
        return _omni_result(f"file://{out}", warnings=[_AUDIO_WARNING])

    monkeypatch.setattr("src.__main__.generate_video_omni_impl", mock_impl)

    await generate_video_omni(ctx=ctx, prompt="a marble")
    assert _AUDIO_WARNING in _emitted(ctx)


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_edit_video_emits_warnings_to_channel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.__main__ import edit_video

    ctx = _ctx(_app_ctx(tmp_path))
    out = tmp_path / "videos" / "e.mp4"
    out.write_bytes(b"mp4")

    async def mock_impl(**kwargs: Any) -> dict[str, Any]:
        return _omni_result(f"file://{out}", warnings=[_AUDIO_WARNING])

    monkeypatch.setattr("src.__main__.generate_video_omni_impl", mock_impl)

    await edit_video(ctx=ctx, previous_interaction_id="int-0", prompt="stormy sky")
    assert _AUDIO_WARNING in _emitted(ctx)


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_generate_video_draft_emits_ignored_params_to_channel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.__main__ import generate_video

    ctx = _ctx(_app_ctx(tmp_path))
    out = tmp_path / "videos" / "d.mp4"
    out.write_bytes(b"mp4")

    async def mock_impl(**kwargs: Any) -> dict[str, Any]:
        return _omni_result(f"file://{out}")

    monkeypatch.setattr("src.__main__.generate_video_omni_impl", mock_impl)

    result = json.loads(
        await generate_video(
            ctx=ctx,
            prompt="a cat",
            model="veo-3.1-fast-generate-001",
            draft=True,
            seed=7,
            negative_prompt="blurry",
        )
    )
    warning = next(w for w in result["warnings"] if "ignored Veo-only" in w)
    assert "seed" in warning and "negative_prompt" not in warning
    assert warning in _emitted(ctx)
    # The negative is no longer dropped: omni's docs say to state it inline.
    assert any("folded into the prompt" in w for w in result["warnings"])


# ===========================================================================
# Defect 2 — a dry run must disclose the warnings its real run would emit
# ===========================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_generate_video_dry_run_draft_discloses_ignored_params(
    tmp_path: Path,
) -> None:
    from src.__main__ import generate_video

    ctx = _ctx(_app_ctx(tmp_path))
    result = json.loads(
        await generate_video(
            ctx=ctx,
            prompt="a cat",
            model="veo-3.1-fast-generate-001",
            draft=True,
            dry_run=True,
            seed=7,
            negative_prompt="blurry",
        )
    )
    assert result["dry_run"] is True
    assert "seed" in result["ignored_veo_params"]
    # negative_prompt is no longer dropped: omni's docs say to state negatives
    # inline, so a draft folds it into the prompt as "No <x>." instead.
    assert "negative_prompt" not in result["ignored_veo_params"]
    assert any("ignored Veo-only" in w for w in result["warnings"])
    assert any("ignored Veo-only" in w for w in _emitted(ctx))


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_generate_video_dry_run_draft_without_veo_params_is_clean(
    tmp_path: Path,
) -> None:
    """No Veo-only params passed → no ignored list and no warnings (no spam)."""
    from src.__main__ import generate_video

    ctx = _ctx(_app_ctx(tmp_path))
    result = json.loads(
        await generate_video(
            ctx=ctx,
            prompt="a cat",
            model="veo-3.1-fast-generate-001",
            draft=True,
            dry_run=True,
        )
    )
    assert "ignored_veo_params" not in result
    assert "warnings" not in result
    assert _emitted(ctx) == []


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_generate_clip_dry_run_animatic_discloses_warnings(
    tmp_path: Path,
) -> None:
    from src.__main__ import generate_clip

    ctx = _ctx(_app_ctx(tmp_path))
    result = json.loads(
        await generate_clip(
            ctx=ctx,
            beats=[{"prompt": "a", "seed": 3}],
            animatic=True,
            add_bridges=True,
            include_audio=True,
            output_gcs_uri="gs://bucket/out/",
            dry_run=True,
        )
    )
    warnings = result["warnings"]
    assert any("add_bridges is ignored in animatic mode" in w for w in warnings)
    assert any("output_gcs_uri is ignored in animatic mode" in w for w in warnings)
    assert any("include_audio is ignored in animatic mode" in w for w in warnings)
    assert any("Veo-only beat params" in w for w in warnings)
    assert any("add_bridges is ignored in animatic mode" in w for w in _emitted(ctx))


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_generate_clip_dry_run_animatic_matches_real_run_warnings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The quote's warnings are exactly what a real animatic run reports."""
    from src.__main__ import generate_clip

    beats = [{"prompt": "a", "seed": 3}, {"prompt": "b", "negative_prompt": "x"}]

    ctx = _ctx(_app_ctx(tmp_path))
    quote = json.loads(
        await generate_clip(
            ctx=ctx,
            beats=beats,
            animatic=True,
            add_bridges=True,
            include_audio=True,
            output_gcs_uri="gs://bucket/out/",
            dry_run=True,
        )
    )

    out = tmp_path / "videos" / "beat.mp4"
    out.write_bytes(b"mp4")

    async def mock_omni(**kwargs: Any) -> dict[str, Any]:
        return _omni_result(f"file://{out}")

    monkeypatch.setattr("src.__main__.generate_video_omni_impl", mock_omni)
    monkeypatch.setattr("src.__main__._client_for_omni", lambda *a, **k: MagicMock())

    ctx2 = _ctx(_app_ctx(tmp_path))
    real = json.loads(
        await generate_clip(
            ctx=ctx2,
            beats=beats,
            animatic=True,
            add_bridges=True,
            include_audio=True,
            output_gcs_uri="gs://bucket/out/",
        )
    )
    assert set(quote["warnings"]) == set(real["warnings"])


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_generate_storyboard_dry_run_emits_plan_warnings(
    tmp_path: Path,
) -> None:
    from src.__main__ import generate_storyboard

    ctx = _ctx(_app_ctx(tmp_path))
    blocks = await generate_storyboard(
        ctx=ctx,
        shots=[{"prompt": "a shot"}],
        model="gemini-2.5-flash-image",
        dry_run=True,
    )
    payload = json.loads(blocks[0].text)
    assert payload["warnings"]  # the reroute warning is in the body
    assert _emitted(ctx) == payload["warnings"]  # and mirrored on the channel


# ===========================================================================
# Defect 3 — the contact sheet is rendered once, and what goes inline is a
# bounded preview of it rather than every byte
# ===========================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(20.0)
async def test_generate_storyboard_composites_once_and_previews_the_sheet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The board is composited once, and the inline copy is a bounded preview.

    Both halves matter and pull against each other. Compositing a second time
    at a smaller width re-decoded every frame and re-ran a LANCZOS pass per
    panel — seconds on a 24-shot board. Returning the on-disk sheet verbatim
    instead was fast but shipped a 1.1MB result that no MCP client accepts.
    Downscaling the one finished sheet satisfies both.
    """
    import src.__main__ as main_mod
    from src import storyboard as sb
    from src.__main__ import generate_storyboard
    from src.storyboard import INLINE_PREVIEW_MAX_BYTES

    ctx = _ctx(_app_ctx(tmp_path))

    async def mock_image_impl(**kwargs: Any) -> dict[str, Any]:
        images_dir: Path = kwargs["images_dir"]
        idx = len(list(images_dir.glob("shot_*.png")))
        path = images_dir / f"shot_{idx}.png"
        path.write_bytes(_png_bytes())
        return {
            "message": "ok",
            "image_url": f"file://{path}",
            "prompt": kwargs["prompt"],
            "model": kwargs["model"],
        }

    monkeypatch.setattr(main_mod, "generate_image_impl", mock_image_impl)

    # Count how many times the board is actually composited.
    calls = {"n": 0}
    original = sb.render_contact_sheet

    def counting_render(*args: Any, **kwargs: Any) -> bytes:
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(sb, "render_contact_sheet", counting_render)

    # Six shots so the inline copy is meaningfully narrower than the 1760-wide
    # on-disk sheet — a one-shot board would fit the budget either way.
    shots = [{"prompt": f"shot {i}", "notes": "wide"} for i in range(6)]
    blocks = await generate_storyboard(ctx=ctx, shots=shots)

    inline = blocks[0].data
    payload = json.loads(blocks[1].text)
    sheet_path = Path(payload["sheet_url"][7:])

    assert calls["n"] == 1  # composited exactly once
    assert inline.startswith(b"\xff\xd8")  # a JPEG preview, not the PNG sheet
    assert len(inline) <= INLINE_PREVIEW_MAX_BYTES

    # Derived from the sheet, not the sheet: byte size is the wrong comparison
    # here, because these frames are flat colour and PNG compresses that better
    # than JPEG can. What holds for any board is that the preview never exceeds
    # the sheet's own width, and the full-resolution sheet is still on disk.
    # That a *large* board is actually scaled down is pinned by
    # tests/test_media_limits.py, on frames that do not compress away.
    assert inline != sheet_path.read_bytes()
    with Image.open(BytesIO(inline)) as preview_img:
        preview_w = preview_img.width
    with Image.open(sheet_path) as sheet_img:
        assert sheet_img.format == "PNG"
        assert preview_w <= sheet_img.width


# ===========================================================================
# Defect 4 — malformed service-account JSON must raise, not swap in ADC
# ===========================================================================


def test_setup_vertex_credentials_raises_on_malformed_sa_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in (
        "GOOGLE_GENAI_USE_VERTEXAI",
        "GOOGLE_SERVICE_ACCOUNT_JSON",
        "GOOGLE_APPLICATION_CREDENTIALS",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_JSON", "not valid json")

    with pytest.raises(ValueError, match="GOOGLE_SERVICE_ACCOUNT_JSON"):
        setup_vertex_credentials()


def test_setup_vertex_credentials_raises_on_malformed_inline_gac(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in (
        "GOOGLE_GENAI_USE_VERTEXAI",
        "GOOGLE_SERVICE_ACCOUNT_JSON",
        "GOOGLE_APPLICATION_CREDENTIALS",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    # Inline JSON in GOOGLE_APPLICATION_CREDENTIALS (starts with "{") that does
    # not parse must also raise rather than fall through to ambient ADC.
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", '{"type": broken')

    with pytest.raises(ValueError, match="GOOGLE_APPLICATION_CREDENTIALS"):
        setup_vertex_credentials()


def test_setup_vertex_credentials_no_sa_json_still_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The legitimate ADC path (no explicit SA JSON) must not be disturbed."""
    for key in (
        "GOOGLE_GENAI_USE_VERTEXAI",
        "GOOGLE_SERVICE_ACCOUNT_JSON",
        "GOOGLE_APPLICATION_CREDENTIALS",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")

    assert setup_vertex_credentials() is None


# ===========================================================================
# A draft is not a Veo call
#
# generate_video resolves the Veo client and the GCS destination in a
# pre-flight so a dry_run refuses everything the render refuses. Hoisting that
# above the draft branch applied Veo's rejections to a call that routes to
# omni, and both of them fire on parameters the draft documents as IGNORED.
# ===========================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_a_draft_reports_output_gcs_uri_as_ignored_instead_of_refusing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`_resolve_video_gcs` ran before the draft branch on a Gemini-API ctx.

    It raises "output_gcs_uri requires Vertex AI mode" for a non-Vertex
    client, so `draft=True, output_gcs_uri=...` errored -- even though
    `_draft_ignored_veo_params` lists output_gcs_uri as ignored and builds a
    warning naming it, which sat downstream and had become unreachable.
    """
    from src.__main__ import generate_video

    ctx = _ctx(_app_ctx(tmp_path))
    out = tmp_path / "videos" / "draft.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"mp4")

    async def mock_impl(**kwargs: Any) -> dict[str, Any]:
        return _omni_result(f"file://{out}")

    monkeypatch.setattr("src.__main__.generate_video_omni_impl", mock_impl)

    result = json.loads(
        await generate_video(
            ctx=ctx,
            prompt="a cat",
            model="veo-3.1-fast-generate-001",
            draft=True,
            output_gcs_uri="gs://bucket/out/",
        )
    )
    assert "error" not in result, result
    warning = next(w for w in result["warnings"] if "ignored Veo-only" in w)
    assert "output_gcs_uri" in warning


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_a_draft_dry_run_prices_a_gcs_request_it_will_ignore(
    tmp_path: Path,
) -> None:
    """The quote must not refuse what the draft render happily ignores."""
    from src.__main__ import generate_video

    result = json.loads(
        await generate_video(
            ctx=_ctx(_app_ctx(tmp_path)),
            prompt="a cat",
            model="veo-3.1-fast-generate-001",
            draft=True,
            dry_run=True,
            output_gcs_uri="gs://bucket/out/",
        )
    )
    assert "error" not in result, result
    assert result["dry_run"] is True
    assert "output_gcs_uri" in result["ignored_veo_params"]


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_a_draft_never_resolves_a_client_for_the_veo_model_it_skips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`_client_for_video_model` raises for a Lite model on Vertex with no key.

    The draft renders on omni and never uses that client, so resolving it
    turned a working draft into a RuntimeError about a model it does not
    touch. Asserted by making the resolver fail outright: if the draft path
    calls it at all, this test fails.
    """
    from src.__main__ import generate_video

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("the draft path must not resolve a Veo client")

    monkeypatch.setattr("src.__main__._client_for_video_model", boom)

    result = json.loads(
        await generate_video(
            ctx=_ctx(_app_ctx(tmp_path)),
            prompt="a cat",
            model="veo-3.1-lite-generate-preview",
            draft=True,
            dry_run=True,
        )
    )
    assert "error" not in result, result
    assert result["dry_run"] is True


# ===========================================================================
# An ignored parameter must be ignored, and a failure must be a body
# ===========================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_animatic_resolution_is_ignored_when_animatic_is_false(
    tmp_path: Path,
) -> None:
    """It was validated above the try, and unconditionally.

    So generate_clip(animatic=False, animatic_resolution="9000p") raised
    ValueError straight out of the tool -- past the handler that turns every
    other failure into an {"error": ...} body -- on a parameter the docstring
    calls "Ignored unless animatic is True".
    """
    from src.__main__ import generate_clip

    result = json.loads(
        await generate_clip(
            ctx=_ctx(_app_ctx(tmp_path)),
            beats=[{"prompt": "a cat"}],
            animatic=False,
            animatic_resolution="9000p",
            dry_run=True,
        )
    )
    assert "error" not in result, result
    assert result["dry_run"] is True


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_a_bad_animatic_resolution_is_an_error_body_not_a_raise(
    tmp_path: Path,
) -> None:
    """When it DOES apply, it still fails the way every other input fails."""
    from src.__main__ import generate_clip

    result = json.loads(
        await generate_clip(
            ctx=_ctx(_app_ctx(tmp_path)),
            beats=[{"prompt": "a cat"}],
            animatic=True,
            animatic_resolution="9000p",
            dry_run=True,
        )
    )
    assert "9000p" in result["error"]


# ===========================================================================
# An interaction must stay findable past the sidecar read cap
# ===========================================================================


def test_an_interaction_is_found_past_the_sidecar_read_limit(tmp_path: Path) -> None:
    """Only the newest 200 sidecars were ever READ.

    So past 200 renders an older interaction returned None from a directory
    that plainly contained it, and every fact PriorInteraction carries went
    with it: prefer_backend fell back to None, so a chain's last turn carrying
    output_gcs_uri could be routed to Vertex holding a Gemini-API-minted id --
    the exact failure PriorInteraction was added to fix -- and
    extend_video_omni's `prior.model != spec.model` refusal was disabled.
    """
    import time

    from src.__main__ import (
        _manifest_for_interaction,
        _prior_interaction,
        _write_sidecar,
    )
    from src.omni import OMNI_1_1_MODEL

    videos_dir = tmp_path / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    # Written first, so it sorts oldest and falls outside the read window.
    buried = videos_dir / "buried.mp4"
    buried.write_bytes(b"mp4")
    _write_sidecar(
        f"file://{buried}",
        {
            "interaction_id": "i-buried",
            "backend": "gemini_api",
            "model": OMNI_1_1_MODEL,
            "duration_seconds": 7.5,
        },
    )
    time.sleep(0.01)

    for i in range(400):  # well past the old 200-read cap
        media = videos_dir / f"r{i}.mp4"
        media.write_bytes(b"mp4")
        _write_sidecar(
            f"file://{media}",
            {
                "interaction_id": f"i-{i}",
                "backend": "vertex",
                "model": OMNI_1_1_MODEL,
                "duration_seconds": 1.0,
            },
        )

    manifest = _manifest_for_interaction(videos_dir, "i-buried")
    assert manifest is not None, "the interaction is in the directory"
    assert manifest["interaction_id"] == "i-buried"

    prior = _prior_interaction(videos_dir, "i-buried")
    assert prior.backend == "gemini_api"
    assert prior.model == OMNI_1_1_MODEL
    assert prior.duration_seconds == 7.5


def test_the_interaction_index_does_not_invent_a_match(tmp_path: Path) -> None:
    """An id that was never recorded still resolves to nothing."""
    from src.__main__ import _manifest_for_interaction, _write_sidecar

    videos_dir = tmp_path / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    media = videos_dir / "one.mp4"
    media.write_bytes(b"mp4")
    _write_sidecar(f"file://{media}", {"interaction_id": "i-real"})

    assert _manifest_for_interaction(videos_dir, "i-nope") is None
    assert _manifest_for_interaction(videos_dir, "i-real") is not None


def test_a_recent_interaction_still_resolves_without_an_index(
    tmp_path: Path,
) -> None:
    """The scan remains the fallback for sidecars written before the index."""
    import shutil

    from src.__main__ import (
        _INTERACTION_INDEX_DIRNAME,
        _manifest_for_interaction,
        _write_sidecar,
    )

    videos_dir = tmp_path / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    media = videos_dir / "recent.mp4"
    media.write_bytes(b"mp4")
    _write_sidecar(f"file://{media}", {"interaction_id": "i-recent"})

    shutil.rmtree(videos_dir / _INTERACTION_INDEX_DIRNAME)
    assert _manifest_for_interaction(videos_dir, "i-recent") is not None


def test_the_index_is_not_mistaken_for_a_sidecar(tmp_path: Path) -> None:
    """The `*.json` glob must not see index entries.

    They live in a subdirectory for exactly this reason: an index entry read
    as a manifest would be a record with no model, backend or duration.
    """
    from src.__main__ import _write_sidecar

    videos_dir = tmp_path / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    media = videos_dir / "one.mp4"
    media.write_bytes(b"mp4")
    _write_sidecar(f"file://{media}", {"interaction_id": "i-1"})

    assert [p.name for p in videos_dir.glob("*.json")] == ["one.json"]


@pytest.mark.asyncio
@pytest.mark.timeout(10.0)
async def test_storyboard_text_fields_are_bounded(tmp_path: Path) -> None:
    """A generous cap on the text a board draws.

    Not the fix for the quadratic layout -- `_split_overlong` and `_ellipsize`
    binary-search their cuts now -- but a bound on memory and wasted work for
    input that cannot be meant seriously. Deliberately far above any real
    prompt, since `prompt` is also what the image model receives.
    """
    from src.__main__ import MAX_STORYBOARD_TEXT_CHARS, generate_storyboard

    over = "x" * (MAX_STORYBOARD_TEXT_CHARS + 1)

    blocks = await generate_storyboard(
        ctx=_ctx(_app_ctx(tmp_path)),
        shots=[{"prompt": over}],
        dry_run=True,
    )
    payload = json.loads(blocks[0].text)
    assert str(MAX_STORYBOARD_TEXT_CHARS) in payload["error"]

    blocks = await generate_storyboard(
        ctx=_ctx(_app_ctx(tmp_path)),
        shots=[{"prompt": "a cat"}],
        title=over,
        dry_run=True,
    )
    payload = json.loads(blocks[0].text)
    assert "title" in payload["error"]

    # A long-but-plausible prompt is still accepted.
    blocks = await generate_storyboard(
        ctx=_ctx(_app_ctx(tmp_path)),
        shots=[{"prompt": "a cat " * 300}],
        dry_run=True,
    )
    payload = json.loads(blocks[0].text)
    assert "error" not in payload, payload


def test_a_pre_index_interaction_past_the_old_cap_is_found_and_backfilled(
    tmp_path: Path,
) -> None:
    """An upgrade with >200 existing renders: no index, old cap, lost records.

    The first version of the index left the fallback scan reading only the
    newest 200 sidecars while its docstring claimed it was "still correct for
    everything written before the index existed". Position 201 returned None.
    The scan reads every candidate now and writes the index entry the record
    never had, so the full walk is paid once per id.
    """
    import shutil
    import time

    from src.__main__ import (
        _INTERACTION_INDEX_DIRNAME,
        _manifest_for_interaction,
        _write_sidecar,
    )

    videos_dir = tmp_path / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    old = videos_dir / "old.mp4"
    old.write_bytes(b"mp4")
    _write_sidecar(f"file://{old}", {"interaction_id": "i-preindex", "backend": "vertex"})
    time.sleep(0.01)
    for i in range(250):  # past the old 200-read cap
        media = videos_dir / f"r{i}.mp4"
        media.write_bytes(b"mp4")
        _write_sidecar(f"file://{media}", {"interaction_id": f"i-{i}"})
    # Simulate the upgrade: these sidecars predate the index entirely.
    shutil.rmtree(videos_dir / _INTERACTION_INDEX_DIRNAME)

    found = _manifest_for_interaction(videos_dir, "i-preindex")
    assert found is not None and found["backend"] == "vertex"
    # And the hit was backfilled, so the next lookup is one read.
    index_dir = videos_dir / _INTERACTION_INDEX_DIRNAME
    assert index_dir.exists() and any(index_dir.iterdir())


# ===========================================================================
# Second review pass: two tool-contract gaps
# ===========================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(5.0)
async def test_a_draft_quote_refuses_the_source_its_render_refuses(
    tmp_path: Path,
) -> None:
    """Every local-source check sat under `if not draft`, so
    generate_video(draft=True, dry_run=True, image_uri=<missing>) quoted
    $0.815 for a render that cannot fetch its input -- the exact invariant
    the pre-flight's own docstring states."""
    from src.__main__ import generate_video

    result = json.loads(
        await generate_video(
            ctx=_ctx(_app_ctx(tmp_path)),
            prompt="a cat",
            model="veo-3.1-fast-generate-001",
            draft=True,
            dry_run=True,
            image_uri=f"file://{tmp_path / 'does-not-exist.png'}",
        )
    )
    assert "error" in result, result
    assert "does-not-exist.png" in result["error"] or "image_uri" in result["error"]


@pytest.mark.asyncio
@pytest.mark.timeout(10.0)
async def test_a_cancelled_storyboard_records_the_shots_it_paid_for(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """generate_storyboard had no CancelledError handler, unlike loop_extend,
    generate_clip and extend_video_omni: cancel after 2 of 4 paid shots left
    two orphan PNGs with no sidecar and no cost record. It now says what was
    rendered and billed before letting the cancellation continue."""
    import asyncio
    import base64
    import logging

    from src.__main__ import generate_storyboard

    app_ctx = _app_ctx(tmp_path)
    images_dir = app_ctx.images_dir
    calls = {"n": 0}
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (32, 32), (9, 9, 9)).save(buf, "PNG")
    image_bytes = buf.getvalue()

    async def two_then_cancel(**kwargs: Any) -> dict[str, Any]:
        calls["n"] += 1
        if calls["n"] > 2:
            raise asyncio.CancelledError()
        path = images_dir / f"shot{calls['n']}.png"
        path.write_bytes(image_bytes)
        return {
            "message": "ok",
            "image_url": f"file://{path}",
            "image_preview": "data:image/png;base64,"
            + base64.b64encode(image_bytes).decode(),
            "prompt": kwargs["prompt"],
            "model": kwargs["model"],
        }

    monkeypatch.setattr("src.__main__.generate_image_impl", two_then_cancel)

    with caplog.at_level(logging.WARNING, logger="src.__main__"):
        with pytest.raises(asyncio.CancelledError):
            await generate_storyboard(
                ctx=_ctx(app_ctx),
                shots=[{"prompt": f"shot {i}"} for i in range(4)],
            )
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "cancelled after 2 of 4" in joined, joined
    assert "shot1.png" in joined and "shot2.png" in joined


# ===========================================================================
# P0: a Veo render that outlives the deadline must answer, structured
# ===========================================================================


def _small_mp4(path: Path, *, size: tuple[int, int] = (64, 64), frames: int = 8) -> Path:
    import imageio.v3 as iio
    import numpy as np
    from io import BytesIO

    h, w = size[1], size[0]
    buf = BytesIO()
    iio.imwrite(
        buf,
        [np.full((h, w, 3), (i * 20) % 255, dtype=np.uint8) for i in range(frames)],
        extension=".mp4",
        fps=2,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.getvalue())
    return path


@pytest.mark.asyncio
@pytest.mark.timeout(10.0)
async def test_generate_video_hands_its_deadline_to_the_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """generate_video exposed no timeout at all while the omni tools default to
    210s to stay under the host's ceiling. The default and an explicit value
    both reach the impl."""
    from src.__main__ import generate_video
    from src.video import VEO_DEFAULT_TIMEOUT_SECONDS

    seen: list[dict[str, Any]] = []
    out = _small_mp4(tmp_path / "videos" / "v.mp4")

    async def mock_impl(**kwargs: Any) -> dict[str, Any]:
        seen.append(kwargs)
        return {
            "message": "ok",
            "video_url": f"file://{out}",
            "model": kwargs["model"],
            "aspect_ratio": "16:9",
            "duration_seconds": 4,
            "resolution": kwargs.get("resolution"),
            "generation_mode": "text_to_video",
            "audio_enabled": False,
        }

    monkeypatch.setattr("src.__main__.generate_video_impl", mock_impl)
    ctx = _ctx(_app_ctx(tmp_path))
    await generate_video(ctx=ctx, prompt="a leaf", model="veo-3.1-fast-generate-001")
    assert seen[-1]["timeout_seconds"] == VEO_DEFAULT_TIMEOUT_SECONDS == 210.0
    await generate_video(
        ctx=ctx, prompt="a leaf", model="veo-3.1-fast-generate-001", timeout_seconds=30
    )
    assert seen[-1]["timeout_seconds"] == 30


@pytest.mark.asyncio
@pytest.mark.timeout(10.0)
async def test_a_veo_timeout_names_the_mode_the_cost_and_the_operation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 4K render on Vertex came back as a bare "Tool execution failed": no
    message, no cost, no backend, no operation -- and $1.20 billed, twice. The
    server's own deadline now fires first and the error is a document."""
    from src.__main__ import generate_video
    from src.video import VeoTimeoutError

    async def mock_impl(**kwargs: Any) -> dict[str, Any]:
        raise VeoTimeoutError(
            "Video generation timed out after 210s. Operation operations/abc123 was "
            "submitted and may still complete and bill; reconcile it in the console.",
            operation_name="operations/abc123",
        )

    monkeypatch.setattr("src.__main__.generate_video_impl", mock_impl)
    app_ctx = _app_ctx(tmp_path)
    app_ctx.client._api_client.vertexai = True
    body = json.loads(
        await generate_video(
            ctx=_ctx(app_ctx),
            prompt="A single red maple leaf on wet pavement",
            model="veo-3.1-fast-generate-001",
            duration_seconds=4,
            resolution="4K",
        )
    )
    assert body["timed_out"] is True
    assert body["operation_name"] == "operations/abc123"
    assert body["generation_mode"] == "text_to_video"
    assert body["model"] == "veo-3.1-fast-generate-001"
    assert body["resolution"] == "4K"
    assert body["attempted_cost"]["usd"] == pytest.approx(1.2)  # 4s @ $0.30/s
    assert body["backend"] == "vertex"
    assert "may still complete and bill" in body["note"]
    assert "operations/abc123" in body["error"]


@pytest.mark.asyncio
@pytest.mark.timeout(10.0)
async def test_any_veo_failure_names_what_was_attempted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not only timeouts: every failure on the paid path carries the mode and
    the attempted cost, so it can be reconciled against the console."""
    from src.__main__ import generate_video

    async def mock_impl(**kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("VEO error: internal")

    monkeypatch.setattr("src.__main__.generate_video_impl", mock_impl)
    body = json.loads(
        await generate_video(
            ctx=_ctx(_app_ctx(tmp_path)),
            prompt="x",
            model="veo-3.1-fast-generate-001",
            duration_seconds=8,
            reference_image_uris=[f"file://{_small_mp4(tmp_path / 'r.mp4')}"],
        )
    )
    assert body["error"] == "VEO error: internal"
    assert body["generation_mode"] == "reference_to_video"
    assert body["attempted_cost"]["usd"] == pytest.approx(0.8)  # forced 8s @ $0.10/s
    assert "timed_out" not in body


@pytest.mark.asyncio
@pytest.mark.timeout(20.0)
async def test_a_4k_render_assembles_and_reports_its_dimensions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hypothesis (b) from the triage -- an exception in response assembly that
    is 4K-specific -- ruled out: a real 3840x2160 file goes through the
    tool's assembly and is measured, dimensioned and metered."""
    from src.__main__ import generate_video

    out = _small_mp4(tmp_path / "videos" / "fourk.mp4", size=(3840, 2160), frames=8)

    async def mock_impl(**kwargs: Any) -> dict[str, Any]:
        return {
            "message": "ok",
            "video_url": f"file://{out}",
            "model": kwargs["model"],
            "aspect_ratio": "16:9",
            "duration_seconds": 4,
            "resolution": kwargs.get("resolution"),
            "generation_mode": "text_to_video",
            "audio_enabled": False,
        }

    monkeypatch.setattr("src.__main__.generate_video_impl", mock_impl)
    app_ctx = _app_ctx(tmp_path)
    app_ctx.client._api_client.vertexai = True
    body = json.loads(
        await generate_video(
            ctx=_ctx(app_ctx),
            prompt="a leaf",
            model="veo-3.1-fast-generate-001",
            duration_seconds=4,
            resolution="4K",
        )
    )
    assert "error" not in body, body
    assert body["rendered_dimensions"] == [3840, 2160]
    assert body["resolution"] == "4K"
    assert body["resolution_source"] == "measured from the rendered video"
    assert body["cost"]["usd"] == pytest.approx(1.2)
    assert body["cost"]["is_estimate"] is False


# ===========================================================================
# P2: in-clip bridges carry the provenance beats carry
# ===========================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(30.0)
async def test_clip_bridges_carry_the_same_provenance_as_beats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Beats reported resolution, resolution_source and duration_source; the
    bridge between them carried a bare `duration_seconds: 4` and none of the
    three, while standalone generate_bridge reported all of them. And the
    top-level total was a sum of snapped requests with no source label."""
    from src.__main__ import generate_clip

    monkeypatch.setattr("src.__main__.assert_frame_decoding_available", lambda: None)
    monkeypatch.setattr("src.__main__.extract_frame_png", lambda *a, **k: _png_bytes())
    calls = {"n": 0}

    async def mock_impl(**kwargs: Any) -> dict[str, Any]:
        calls["n"] += 1
        out = _small_mp4(tmp_path / "videos" / f"seg{calls['n']}.mp4")
        return {
            "message": "ok",
            "video_url": f"file://{out}",
            "model": kwargs["model"],
            "aspect_ratio": kwargs.get("aspect_ratio", "16:9"),
            "duration_seconds": kwargs.get("duration_seconds", 4),
            "resolution": kwargs.get("resolution"),
            "generation_mode": "text_to_video",
            "audio_enabled": False,
        }

    monkeypatch.setattr("src.__main__.generate_video_impl", mock_impl)
    # Bridges are first/last-frame renders, which Veo serves on Vertex only.
    app_ctx = _app_ctx(tmp_path)
    app_ctx.client._api_client.vertexai = True
    body = json.loads(
        await generate_clip(
            ctx=_ctx(app_ctx),
            beats=[{"prompt": "a"}, {"prompt": "b"}],
            add_bridges=True,
            model="veo-3.1-fast-generate-001",
        )
    )
    assert "error" not in body, body
    bridges = [s for s in body["segments"] if s.get("kind") == "bridge"]
    assert bridges, body["segments"]
    for bridge in bridges:
        for key in ("resolution", "resolution_source", "duration_source"):
            assert bridge.get(key), (key, bridge)
        assert isinstance(bridge["duration_seconds"], float)
    assert body["total_duration_source"]
    assert body["total_duration_source"].startswith("sum of")


@pytest.mark.asyncio
@pytest.mark.timeout(10.0)
async def test_a_vertex_iam_refusal_names_the_grant_that_fixes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 403 from Veo arrived as a wall of SDK traceback ending in a
    troubleshooter URL, saying nothing about which identity was refused --
    the one fact the operator needs, since the server may be running as a
    service account they did not expect. omni already translated its own
    allowlist refusals; Veo did not."""
    from src.__main__ import generate_video

    class Denied(Exception):
        def __init__(self) -> None:
            super().__init__(
                "403 PERMISSION_DENIED. {'error': {'code': 403, 'message': "
                "\"Permission 'aiplatform.endpoints.predict' denied on resource "
                "'//aiplatform.googleapis.com/projects/p/locations/us-central1/"
                "publishers/google/models/veo-3.1-fast-generate-001'\"}}"
            )
            self.code = 403

    async def denied(**kwargs: Any) -> dict[str, Any]:
        raise Denied()

    monkeypatch.setattr("src.__main__.generate_video_impl", denied)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "cxo-agi")
    app_ctx = _app_ctx(tmp_path)
    app_ctx.client._api_client.vertexai = True

    body = json.loads(
        await generate_video(
            ctx=_ctx(app_ctx),
            prompt="a leaf",
            model="veo-3.1-fast-generate-001",
            duration_seconds=4,
            resolution="4K",
        )
    )
    assert "aiplatform.endpoints.predict" in body["advice"]
    assert "roles/aiplatform.user" in body["advice"]
    assert "cxo-agi" in body["advice"]
    assert "Nothing was rendered or billed" in body["advice"]
    # The attempt facts are still there, so the refusal is fully described.
    assert body["generation_mode"] == "text_to_video"
    assert body["resolution"] == "4K"
    assert body["attempted_cost"]["usd"] == pytest.approx(1.2)


@pytest.mark.asyncio
@pytest.mark.timeout(10.0)
async def test_a_non_iam_failure_is_left_exactly_as_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The translator must not attach IAM advice to unrelated errors."""
    from src.__main__ import generate_video

    class Other(Exception):
        def __init__(self) -> None:
            super().__init__("403 quota exceeded for requests")
            self.code = 403

    async def failing(**kwargs: Any) -> dict[str, Any]:
        raise Other()

    monkeypatch.setattr("src.__main__.generate_video_impl", failing)
    body = json.loads(
        await generate_video(
            ctx=_ctx(_app_ctx(tmp_path)), prompt="a leaf",
            model="veo-3.1-fast-generate-001", duration_seconds=4,
        )
    )
    assert "advice" not in body
    assert body["error"] == "403 quota exceeded for requests"
