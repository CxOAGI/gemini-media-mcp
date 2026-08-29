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
