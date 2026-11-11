"""Omni 1.2 request serialization, preflight limits and billing contracts."""

import asyncio
import datetime
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import src.__main__ as server
import src.omni as omni
from src.pricing import (
    estimate_omni_upscale_cost,
    estimate_video_cost,
    resolve_model_id,
)
from src.routing import RoutingConstraints, plan_generation
from tests.test_wire_contract_omni import _OmniStub, _run


def context(tmp_path: Path, *, vertex: bool = False) -> MagicMock:
    ctx = MagicMock()
    ctx.info = AsyncMock()
    ctx.warning = AsyncMock()
    ctx.error = AsyncMock()
    client = MagicMock()
    client._api_client.vertexai = vertex
    images, videos = tmp_path / "images", tmp_path / "videos"
    images.mkdir(exist_ok=True)
    videos.mkdir(exist_ok=True)
    ctx.request_context.lifespan_context = server.AppContext(
        data_folder=tmp_path,
        images_dir=images,
        videos_dir=videos,
        client=client,
        allowed_gcs_buckets=frozenset({"bucket"}),
    )
    return ctx


@pytest.mark.parametrize("duration", [3, 10, 11, 14, 15])
def test_duration_survives_sdk_serialization(tmp_path: Path, duration: int) -> None:
    stub = _OmniStub()
    result = _run(stub, tmp_path, duration_seconds=duration)
    assert stub.created()["model"] == omni.OMNI_1_2_MODEL
    assert stub.response_format()["duration"] == f"{duration}s"
    assert result["duration_seconds"] == duration
    assert "generation_config" not in stub.created()


@pytest.mark.parametrize(
    "kwargs, fragment",
    [
        (
            {"duration_seconds": 15, "has_first_frame": True, "has_last_frame": True},
            "3 to 10",
        ),
        ({"duration_seconds": 15, "reference_video_count": 2}, "at most 1"),
        ({"reference_image_count": 7}, "at most 6"),
        ({"inferred_image_count": 7}, "at most 6"),
        ({"task": "edit", "reference_video_count": 1}, "only for new generation"),
        (
            {"is_continuation": True, "reference_video_count": 1},
            "only for new generation",
        ),
        ({"task": "extend", "duration_seconds": 15}, "3 to 10"),
    ],
)
def test_invalid_combinations_are_refused(kwargs: dict, fragment: str) -> None:
    with pytest.raises(ValueError, match=fragment):
        omni.validate_omni_request(omni.OMNI_1_2_MODEL, **kwargs)


def test_short_generation_accepts_three_video_references() -> None:
    omni.validate_omni_request(
        omni.OMNI_1_2_MODEL, duration_seconds=10, reference_video_count=3
    )
    omni.validate_omni_request(
        omni.OMNI_1_2_MODEL,
        duration_seconds=15,
        reference_image_count=6,
        reference_video_count=1,
    )


@pytest.mark.parametrize(
    "resolution,tokens",
    [("360p", 1931), ("720p", 5792), ("1080p", 8688), ("4K", 17376)],
)
def test_quotes_use_new_token_rates(resolution: str, tokens: int) -> None:
    cost = estimate_video_cost(omni.OMNI_1_2_MODEL, 15, resolution)
    assert cost is not None
    assert cost.usd == pytest.approx((15 + 1 / 24) * tokens * 12 / 1_000_000)


def test_the_1_1_redirect_changes_capabilities_and_price_before_googles_date(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This server retired 1.1 ahead of Google's own redirect: on either side
    of OMNI_1_1_REDIRECT_DATE, both spellings run, price and limit as 1.2."""
    for day in (datetime.date(2026, 12, 2), datetime.date(2026, 12, 3)):
        monkeypatch.setattr(omni, "_today", lambda day=day: day)
        for spelling in (omni.OMNI_1_1_MODEL, "gemini-omni-1.1-flash-preview"):
            assert resolve_model_id(spelling) == omni.OMNI_1_2_MODEL
            assert omni.omni_spec(spelling).max_duration_seconds == 15
            assert omni.canonical_omni_model(spelling) == omni.OMNI_1_2_MODEL


@pytest.mark.asyncio
async def test_tool_preflight_quotes_fifteen_seconds(tmp_path: Path) -> None:
    ctx = context(tmp_path)
    result = json.loads(
        await server.generate_video_omni(
            ctx, "continuous orbit", duration_seconds=15, dry_run=True
        )
    )
    assert result["duration_seconds"] == 15
    assert result["model"] == omni.OMNI_1_2_MODEL
    assert result["estimated_cost"]["usd"] == pytest.approx(1.045456)


@pytest.mark.asyncio
async def test_reference_limit_is_checked_before_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fetch = AsyncMock(side_effect=AssertionError("no fetching an invalid request"))
    monkeypatch.setattr(server, "_fetch_omni_media", fetch)
    result = json.loads(
        await server.generate_video_omni(
            context(tmp_path),
            "orbit",
            duration_seconds=15,
            reference_video_uris=["https://example.com/a", "https://example.com/b"],
        )
    )
    assert "at most 1" in result["error"]
    fetch.assert_not_called()


def test_extension_duration_reaches_wire(tmp_path: Path) -> None:
    stub = _OmniStub()
    _run(
        stub,
        tmp_path,
        prompt="Extend this video.",
        previous_interaction_id="prior",
        task="extend",
        duration_seconds=4,
    )
    assert stub.response_format()["duration"] == "4s"
    assert "aspect_ratio" not in stub.response_format()
    assert "generation_config" not in stub.created()


def test_extension_cost_handles_custom_steps_and_unknown_sources() -> None:
    spec = omni.omni_spec(omni.OMNI_1_2_MODEL)
    assert omni.omni_extension_priced_lengths(spec, 15, 3, per_turn_seconds=4) == [
        19,
        23,
        27,
    ]
    assert omni.omni_extension_priced_lengths(spec, None, 2) == [30, 30]


@pytest.mark.asyncio
async def test_custom_extension_preflight_matches_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        server,
        "_prior_interaction_or_empty",
        AsyncMock(
            return_value=server.PriorInteraction(
                model=omni.OMNI_1_2_MODEL, backend="gemini_api", duration_seconds=15
            )
        ),
    )
    result = json.loads(
        await server.extend_video_omni(
            context(tmp_path),
            "Continue the scene",
            previous_interaction_id="prior",
            duration_seconds=4,
            times=3,
            dry_run=True,
        )
    )
    assert "error" not in result, result
    assert result["turn_output_seconds"] == [19, 23, 27]
    assert not result.get("warnings")


def test_upscale_wire_is_a_task_without_generation_controls(tmp_path: Path) -> None:
    stub = _OmniStub()
    _run(
        stub,
        tmp_path,
        task="upscale",
        input_video_uri="gs://bucket/draft.mp4",
        resolution="1080p",
        duration_seconds=None,
    )
    request = stub.created()
    assert request["generation_config"]["video_config"]["task"] == "upscale"
    assert request["input"][0]["content"] == [
        {"type": "video", "uri": "gs://bucket/draft.mp4", "mime_type": "video/mp4"}
    ]
    assert "duration" not in stub.response_format()
    assert "aspect_ratio" not in stub.response_format()


def test_upscale_conversation_keeps_explicit_task(tmp_path: Path) -> None:
    """The documented stateful upscale carries the task beside the id.

    Every other conversational turn leaves its task off (the earlier models
    400 on the pair). Vertex's "Upscale videos" page shows this exact pair in
    its "Synchronous: stateful" request and warns that without the task the
    request is read as an edit. The result must say the task was sent, not
    repeat the conversational-turn caveat that it was not.

    The same sample's "Upsample video." text is NOT sent: Vertex 400s it
    ("Text input is not supported for upscale task.", live 2026-11-09), and an
    empty input beside the id was accepted live and kept the source's length.
    Whatever prompt the caller passes stays off the wire.
    """
    stub = _OmniStub()
    result = _run(
        stub,
        tmp_path,
        prompt="Upsample video.",
        task="upscale",
        previous_interaction_id="prior",
        resolution="1080p",
        duration_seconds=None,
    )
    assert stub.created()["previous_interaction_id"] == "prior"
    assert stub.created()["generation_config"]["video_config"]["task"] == "upscale"
    assert stub.created()["input"] == []
    assert "Upsample video." not in json.dumps(stub.created())
    assert result["task"] == "upscale"
    assert "NOT sent" not in result["task_source"], result["task_source"]


@pytest.mark.parametrize(
    "source",
    [
        {"input_video_uri": "gs://bucket/draft.mp4"},
        {"previous_interaction_id": "prior"},
    ],
    ids=["uploaded", "conversational"],
)
@pytest.mark.parametrize(
    "resolution,wire", [("720p", "720p"), ("1080p", "1080p"), ("4K", "4k")]
)
def test_upscale_sends_the_requested_resolution(
    tmp_path: Path, source: dict, resolution: str, wire: str
) -> None:
    """The target resolution is the one thing an upscale request is for.

    Vertex documents it as required in response_format ("Omitting the target
    resolution returns an error"), and the bill is priced from it, so a
    request that lost or respelled it would render one tier and bill another.
    """
    stub = _OmniStub()
    result = _run(
        stub,
        tmp_path,
        task="upscale",
        resolution=resolution,
        duration_seconds=None,
        **source,
    )
    assert stub.response_format()["resolution"] == wire
    assert result["resolution"] == resolution


@pytest.mark.parametrize(
    "source,target,tokens",
    [
        ("360p", "720p", 2896),
        ("360p", "1080p", 5792),
        ("720p", "1080p", 2896),
        ("720p", "4K", 11584),
    ],
)
def test_upscale_bills_resolution_pair_with_zero_input(
    source: str, target: str, tokens: int
) -> None:
    quote = estimate_omni_upscale_cost(15, source, target)
    assert quote.usd == pytest.approx((15 + 1 / 24) * tokens * 12 / 1_000_000)
    assert quote.breakdown["input_usd"] == 0


@pytest.mark.asyncio
async def test_upscale_preflight_and_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = context(tmp_path, vertex=True)
    # A real run downloads a gs:// source to measure it; one that cannot be
    # read is quoted at the 15s bound, which is what this test prices.
    monkeypatch.setattr(server, "fetch", AsyncMock(return_value=None))
    kwargs = {
        "input_video_uri": "gs://bucket/draft.mp4",
        "source_resolution": "720p",
        "resolution": "4K",
        # A 4K render is past the inline response limit, and Vertex delivers
        # it only to Cloud Storage; this test is about the quote and the cap.
        "output_gcs_uri": "gs://bucket/upscaled.mp4",
    }
    quote = json.loads(await server.upscale_video_omni(ctx, **kwargs, dry_run=True))
    assert "error" not in quote, quote
    assert quote["estimated_cost"]["usd"] == pytest.approx(
        estimate_omni_upscale_cost(15, "720p", "4K").usd
    )
    refused = json.loads(
        await server.upscale_video_omni(ctx, **kwargs, max_cost_usd=0.01)
    )
    assert "exceeds max_cost_usd" in refused["error"]
    ctx.request_context.lifespan_context.client.interactions.create.assert_not_called()


@pytest.mark.asyncio
async def test_upscale_refuses_unsupported_transition(tmp_path: Path) -> None:
    result = json.loads(
        await server.upscale_video_omni(
            context(tmp_path, vertex=True),
            input_video_uri="gs://bucket/draft.mp4",
            source_resolution="360p",
            resolution="4K",
            dry_run=True,
        )
    )
    assert "360p to 720p/1080p" in result["error"]


def test_planner_can_recommend_one_fifteen_second_render() -> None:
    plan = plan_generation(
        "15 second continuous video of a cat",
        RoutingConstraints(
            media_kind="video",
            pinned_model=omni.OMNI_1_2_MODEL,
            duration_seconds=15,
        ),
    )
    assert plan.routes
    assert plan.routes[0].params["duration_seconds"] == 15


def test_vertex_extension_uses_its_own_ceiling() -> None:
    spec = omni.omni_spec(omni.OMNI_1_2_MODEL)
    assert omni.omni_extension_priced_lengths(spec, 15, 3, vertexai=True) == [
        25,
        35,
        40,
    ]
    assert omni.omni_extension_priced_lengths(spec, None, 2, vertexai=True) == [40, 40]


def test_upscale_intent_preserves_the_source_in_the_plan() -> None:
    plan = plan_generation(
        "Upscale my video from 720p to 4K",
        RoutingConstraints(
            media_kind="video",
            backend="vertex",
            source_video_uri="gs://bucket/draft.mp4",
            resolution="4K",
        ),
    )
    assert plan.routes[0].tool == "upscale_video_omni"
    assert plan.routes[0].params["input_video_uri"] == "gs://bucket/draft.mp4"
    assert plan.routes[0].params["source_resolution"] == "720p"
    assert plan.routes[0].cost is not None


@pytest.mark.asyncio
async def test_upscale_result_and_manifest_use_the_upscale_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = context(tmp_path, vertex=True)
    client = ctx.request_context.lifespan_context.client
    monkeypatch.setattr(server, "_client_for_omni", lambda *args, **kwargs: client)
    # The measuring download finds nothing, so the source is billed at the
    # 15s bound; unstubbed, it reached for real Cloud Storage credentials.
    monkeypatch.setattr(server, "fetch", AsyncMock(return_value=None))
    client.interactions.create.return_value = {
        "id": "upscaled",
        "status": "completed",
        "output_video": {"uri": "gs://bucket/approved.mp4"},
    }
    result = json.loads(
        await server.upscale_video_omni(
            ctx,
            input_video_uri="gs://bucket/draft.mp4",
            source_resolution="720p",
            resolution="1080p",
            output_gcs_uri="gs://bucket/approved.mp4",
        )
    )
    assert "error" not in result, result
    assert result["cost"]["usd"] == pytest.approx(
        estimate_omni_upscale_cost(15, "720p", "1080p").usd
    )
    assert result["cost"]["breakdown"]["input_usd"] == 0
    assert result["manifest"]["cost"] == result["cost"]
    request = client.interactions.create.call_args.kwargs
    assert request["model"] == omni.OMNI_1_2_MODEL
    assert request["generation_config"]["video_config"]["task"] == "upscale"


@pytest.mark.asyncio
async def test_edit_keeps_reference_images_across_a_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = context(tmp_path)
    fetch = AsyncMock(return_value=b"image")
    render = AsyncMock(
        return_value={"video_url": "gs://bucket/edit.mp4", "interaction_id": "edited"}
    )
    monkeypatch.setattr(server, "_fetch_omni_media", fetch)
    monkeypatch.setattr(server, "_omni_generate_and_manifest", render)
    result = json.loads(
        await server.edit_video(
            ctx,
            "prior",
            "Swap the product",
            reference_image_uris=["https://example.com/product.png"],
        )
    )
    assert "error" not in result, result
    assert render.call_args.kwargs["reference_image_bytes_list"] == [b"image"]
    assert render.call_args.kwargs["previous_interaction_id"] == "prior"


@pytest.mark.parametrize("source", [35, 40, 60])
def test_vertex_extension_rolls_the_source_window(source: int) -> None:
    spec = omni.omni_spec(omni.OMNI_1_2_MODEL, vertexai=True)
    outputs = omni.omni_extension_priced_lengths(
        spec, source, 3, per_turn_seconds=3, vertexai=True
    )
    assert outputs == [33, 33, 33]
    assert omni.omni_extension_refusal(spec, source) is None
    assert omni.omni_extension_appended_seconds(source, outputs, spec=spec) == 9
    assert (
        omni.omni_continuation_upper_bound(
            spec, "extend", source, vertexai=True, per_turn_seconds=3
        )
        == 33
    )
    assert (
        omni.omni_continuation_upper_bound(
            spec, "extend", None, vertexai=True, per_turn_seconds=3
        )
        == 33
    )


@pytest.mark.parametrize("backend", ["vertex", "gemini_api"])
def test_planner_preserves_multiple_extension_turns(backend: str) -> None:
    plan = plan_generation(
        "Extend my existing video by 20 seconds",
        RoutingConstraints(
            media_kind="video",
            backend=backend,
            pinned_model=omni.OMNI_1_2_MODEL,
            source_video_uri="gs://bucket/source.mp4",
        ),
    )
    assert plan.routes[0].params["times"] == 2


@pytest.mark.parametrize("backend", ["vertex", "gemini_api"])
def test_upscale_adjective_keeps_generation_route(backend: str) -> None:
    plan = plan_generation(
        "Create a video of an upscale hotel lobby",
        RoutingConstraints(media_kind="video", backend=backend),
    )
    assert plan.routes
    assert all(route.tool != "upscale_video_omni" for route in plan.routes)


@pytest.mark.parametrize("target", ["720p", "1080p", "4K"])
def test_upscale_planner_reads_destination(target: str) -> None:
    source = "360p" if target == "720p" else "720p"
    plan = plan_generation(
        f"Upscale my video from {source} to {target}",
        RoutingConstraints(media_kind="video", backend="vertex"),
    )
    assert plan.routes[0].params["resolution"] == target
    assert plan.routes[0].cost.usd == pytest.approx(
        estimate_omni_upscale_cost(15, source, target).usd
    )


@pytest.mark.asyncio
async def test_vertex_long_extension_quotes_and_renders_all_turns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = context(tmp_path, vertex=True)
    prior = server.PriorInteraction(
        model=omni.OMNI_1_2_MODEL,
        backend="vertex",
        duration_seconds=60,
    )
    monkeypatch.setattr(
        server, "_prior_interaction_or_empty", AsyncMock(return_value=prior)
    )
    monkeypatch.setattr(server, "_source_duration_or_none", AsyncMock(return_value=60))
    render = AsyncMock(
        return_value={
            "interaction_id": "extended",
            "video_url": "gs://bucket/extended.mp4",
            "duration_seconds": 33,
        }
    )
    monkeypatch.setattr(server, "_omni_generate_and_manifest", render)
    args = dict(previous_interaction_id="prior", times=3, duration_seconds=3)
    quote = json.loads(
        await server.extend_video_omni(ctx, "Extend the scene", **args, dry_run=True)
    )
    assert "error" not in quote, quote
    assert quote["turn_output_seconds"] == [33, 33, 33]
    assert quote["appended_seconds"] == 9
    result = json.loads(await server.extend_video_omni(ctx, "Extend the scene", **args))
    assert "error" not in result, result
    assert render.await_count == 3
    assert result["appended_seconds"] == 9


@pytest.mark.asyncio
async def test_vertex_large_input_uses_inline_instead_of_files_api() -> None:
    from google import genai
    from google.auth.credentials import AnonymousCredentials

    client = genai.Client(
        vertexai=True,
        project="offline-test",
        location="global",
        credentials=AnonymousCredentials(),
    )
    try:
        part = await omni._prepare_video_part(
            client,
            b"\x00\x00\x00\x18ftypmp42" + b"0" * (9 * 1024 * 1024),
            "source",
            spec=omni.omni_spec(omni.OMNI_1_2_MODEL),
            log_callback=None,
            run_within_deadline=AsyncMock(side_effect=AssertionError("no upload")),
            deadline_expired=lambda: False,
            warnings=[],
        )
        assert part["type"] == "video"
        assert part["mime_type"] == "video/mp4"
        assert "data" in part
    finally:
        client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("remote", [False, True])
async def test_upscale_checks_buffered_dimensions_before_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    remote: bool,
) -> None:
    ctx = context(tmp_path, vertex=True)
    source = (
        "https://example.com/source.mp4" if remote else f"file://{tmp_path}/source.mp4"
    )
    if not remote:
        (tmp_path / "source.mp4").write_bytes(b"video")
    monkeypatch.setattr(
        server,
        "_fetch_local_source",
        AsyncMock(return_value=None if remote else b"video"),
    )
    monkeypatch.setattr(server, "_fetch_omni_media", AsyncMock(return_value=b"video"))

    async def probe(function, *args):
        if function is server.measure_video_metadata_bytes:
            return 6.0, (1280, 720)
        return 6.0

    monkeypatch.setattr(server, "_probe_media", probe)
    render = AsyncMock(return_value={"interaction_id": "upscaled"})
    monkeypatch.setattr(server, "_omni_generate_and_manifest", render)
    # The default 1080p target is delivered only to Cloud Storage on Vertex;
    # this test is about the dimension cross-check, not delivery.
    destination = "gs://bucket/upscaled.mp4"
    refused = json.loads(
        await server.upscale_video_omni(
            ctx,
            input_video_uri=source,
            source_resolution="360p",
            output_gcs_uri=destination,
        )
    )
    assert "disagrees" in refused["error"]
    render.assert_not_awaited()
    result = json.loads(
        await server.upscale_video_omni(
            ctx, input_video_uri=source, output_gcs_uri=destination
        )
    )
    assert "error" not in result, result
    assert render.call_args.kwargs["upscale_source_resolution"] == "720p"
    assert render.call_args.kwargs["source_duration_seconds"] == 6.0


@pytest.mark.asyncio
async def test_upscale_remote_delivery_keeps_known_source_duration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = context(tmp_path, vertex=True)
    app_ctx = ctx.request_context.lifespan_context
    monkeypatch.setattr(server, "_client_for_omni", lambda *a, **kw: app_ctx.client)
    monkeypatch.setattr(
        server,
        "_prior_interaction_or_empty",
        AsyncMock(
            return_value=server.PriorInteraction(
                model=omni.OMNI_1_2_MODEL,
                backend="vertex",
                duration_seconds=6,
                resolution="720p",
            )
        ),
    )
    app_ctx.client.interactions.create.return_value = {
        "id": "upscaled",
        "status": "completed",
        "output_video": {"uri": "gs://bucket/upscaled.mp4"},
    }
    result = json.loads(
        await server.upscale_video_omni(
            ctx,
            previous_interaction_id="prior",
            output_gcs_uri="gs://bucket/upscaled.mp4",
        )
    )
    assert "error" not in result, result
    assert result["duration_seconds"] == 6
    assert result["cost"]["usd"] == pytest.approx(
        estimate_omni_upscale_cost(6, "720p", "1080p").usd
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["edit", "extend"])
async def test_vertex_gcs_source_passes_directly_and_checks_allowlist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    ctx = context(tmp_path, vertex=True)
    fetch = AsyncMock(side_effect=AssertionError("GCS source should remain a URI"))
    monkeypatch.setattr(server, "_fetch_omni_media", fetch)
    # The real run downloads the source once to measure it (never to send
    # it); stubbed, so the test never reaches for Cloud Storage credentials.
    monkeypatch.setattr(server, "fetch", AsyncMock(return_value=None))
    render = AsyncMock(
        return_value={
            "interaction_id": "rendered",
            "video_url": "gs://bucket/output.mp4",
            "duration_seconds": 25,
        }
    )
    monkeypatch.setattr(server, "_omni_generate_and_manifest", render)
    tool = (
        server.generate_video_omni if operation == "edit" else server.extend_video_omni
    )
    for dry_run in (True, False):
        denied = json.loads(
            await tool(
                ctx,
                "Continue the scene",
                input_video_uri="gs://denied/source.mp4",
                dry_run=dry_run,
            )
        )
        assert "error" in denied
    result = json.loads(
        await tool(ctx, "Continue the scene", input_video_uri="gs://bucket/source.mp4")
    )
    assert "error" not in result, result
    assert render.call_args.kwargs["input_video_uri"] == "gs://bucket/source.mp4"
    assert render.call_args.kwargs["input_video_bytes"] is None
    fetch.assert_not_awaited()


def test_gcs_extension_serializes_source_and_duration(tmp_path: Path) -> None:
    stub = _OmniStub()
    _run(
        stub,
        tmp_path,
        task="extend",
        input_video_uri="gs://bucket/source.mp4",
        duration_seconds=3,
    )
    assert stub.response_format()["duration"] == "3s"
    assert "aspect_ratio" not in stub.response_format()
    parts = stub.created()["input"][0]["content"]
    assert parts[0]["type"] == "video"
    assert parts[0]["uri"] == "gs://bucket/source.mp4"


@pytest.mark.asyncio
async def test_gemini_api_large_input_keeps_video_file_type() -> None:
    from tests.test_omni_capabilities import _FakeFiles, _prepare

    data = b"\x00\x00\x00\x18ftypmp42" + b"0" * (9 * 1024 * 1024)
    part = await _prepare(data, omni.OMNI_1_2_MODEL, _FakeFiles(["ACTIVE"]))
    assert part["type"] == "video"
    assert part["uri"] == "files/abc"


def test_upscale_planner_rejects_unsupported_source_and_target() -> None:
    for intent in ("Upscale my video from 1080p to 4K", "Upscale my video to 360p"):
        plan = plan_generation(
            intent, RoutingConstraints(media_kind="video", backend="vertex")
        )
        assert not plan.routes
        assert plan.conflicts


@pytest.mark.asyncio
@pytest.mark.parametrize("dry_run", [True, False])
async def test_vertex_edit_inferred_images_obey_the_backend_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dry_run: bool,
) -> None:
    fetch = AsyncMock(side_effect=AssertionError("invalid images must not be fetched"))
    monkeypatch.setattr(server, "_fetch_omni_media", fetch)
    result = json.loads(
        await server.generate_video_omni(
            context(tmp_path, vertex=True),
            "Change the product",
            previous_interaction_id="prior",
            image_uris=["https://example.com/a.png"] * 3,
            dry_run=dry_run,
        )
    )
    assert "at most 2" in result["error"]
    fetch.assert_not_awaited()


def test_zero_duration_snaps_to_the_minimum_negative_and_nan_do_not() -> None:
    """Zero is allowed in the tool layer, so it must be allowed here too.

    __main__._validate_duration_seconds lets zero through on purpose (it snaps
    to the minimum in both layers). Refusing it here made a dry run quote a
    call the real run refused.
    """
    assert omni.validate_omni_request(omni.OMNI_1_2_MODEL, duration_seconds=0)
    for bad in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite"):
            omni.validate_omni_request(omni.OMNI_1_2_MODEL, duration_seconds=bad)


def test_a_zero_duration_generation_sends_the_minimum(tmp_path: Path) -> None:
    stub = _OmniStub()
    result = _run(stub, tmp_path, duration_seconds=0)
    assert stub.response_format()["duration"] == "3s"
    assert result["duration_seconds"] == 3
    assert any("it renders 3s" in w for w in result["warnings"])


@pytest.mark.parametrize("duration", [0, float("nan")])
def test_an_edit_is_not_refused_over_a_duration_it_never_sends(
    tmp_path: Path, duration: float
) -> None:
    """An edit strips duration from the request, so it cannot be what refuses it."""
    stub = _OmniStub()
    result = _run(
        stub, tmp_path, previous_interaction_id="abc", duration_seconds=duration
    )
    assert "duration" not in stub.response_format()
    assert result["duration_seconds"] is None
    # No clamp is announced for a value that never reached the wire.
    assert not any("clamped" in w for w in result.get("warnings", []))


def test_edit_video_dry_run_and_real_run_agree_on_zero_duration(
    tmp_path: Path,
) -> None:
    """edit_video documents duration_seconds as ignored; both runs must agree.

    The dry run quoted $1.05 for duration_seconds=0 while the real run raised
    "duration_seconds must be finite and greater than zero". Synchronous
    because _run drives its own event loop.
    """
    fn = getattr(server.edit_video, "fn", server.edit_video)
    quote = json.loads(
        asyncio.run(
            fn(
                context(tmp_path),
                previous_interaction_id="abc",
                prompt="x",
                duration_seconds=0,
                dry_run=True,
            )
        )
    )
    assert "error" not in quote, quote
    assert quote["estimated_cost"]["usd"] > 0
    videos = tmp_path / "render"
    videos.mkdir()
    result = _run(
        _OmniStub(), videos, previous_interaction_id="abc", duration_seconds=0
    )
    assert result["interaction_id"] == "i-1"


@pytest.mark.parametrize(
    "uri,mime",
    [
        ("gs://bucket/clip.mov", "video/mov"),
        ("gs://bucket/CLIP.MOV", "video/mov"),
        ("gs://bucket/clip.avi", "video/avi"),
        ("gs://bucket/clip.mkv", "video/webm"),
        ("gs://bucket/clip.mp4", "video/mp4"),
        ("gs://bucket/clip", "video/mp4"),
        ("gs://bucket/clip.wmv", "video/wmv"),
        ("gs://bucket/clip.3gp", "video/3gpp"),
        # Outside the SDK's set, or not a video extension at all: the host's
        # mime.types said text/vnd.trolltech.linguist for .ts, video/ogg for
        # .ogv and image/png for .png, and each went out as the label.
        ("gs://bucket/clip.ts", "video/mp4"),
        ("gs://bucket/clip.ogv", "video/mp4"),
        ("gs://bucket/clip.png", "video/mp4"),
        ("gs://bucket/v1.2/clip", "video/mp4"),
    ],
)
def test_a_gcs_source_is_labelled_in_the_sdks_spelling(
    tmp_path: Path, uri: str, mime: str
) -> None:
    """A gs:// source is labelled from its extension, the same way bytes are.

    mimetypes says video/quicktime, video/x-msvideo and video/x-matroska,
    which are outside the SDK's VideoContentMimeType set; the magic-byte path
    already used the SDK's spellings, so the same .mov went out two ways.
    """
    stub = _OmniStub()
    _run(stub, tmp_path, input_video_uri=uri)
    parts = stub.created()["input"][0]["content"]
    video = next(p for p in parts if p["type"] == "video")
    assert video["mime_type"] == mime


def test_both_video_labelling_paths_agree() -> None:
    mov = b"\x00\x00\x00\x14ftypqt  " + b"\x00" * 8
    avi = b"RIFF\x00\x00\x00\x00AVI " + b"\x00" * 4
    mkv = b"\x1a\x45\xdf\xa3" + b"\x00" * 8
    for data, uri in ((mov, "a.mov"), (avi, "a.avi"), (mkv, "a.mkv")):
        assert omni._detect_video_mime(data) == omni._guess_video_mime_from_uri(
            f"gs://bucket/{uri}"
        )


@pytest.mark.parametrize("continues", [True, False])
def test_an_extension_warning_says_what_was_sent(
    tmp_path: Path, continues: bool
) -> None:
    """It said "Extension duration is sent" on extensions that sent none."""
    source = (
        {"previous_interaction_id": "prior"}
        if continues
        else {"input_video_uri": "gs://bucket/source.mp4"}
    )

    def duration_note(result: dict) -> str:
        notes = [w for w in result.get("warnings", []) if "duration" in w.lower()]
        assert len(notes) == 1, notes
        return notes[0]

    sent = _OmniStub()
    result = _run(sent, tmp_path, task="extend", duration_seconds=4, **source)
    assert sent.response_format()["duration"] == "4s"
    note = duration_note(result)
    assert "sends duration=4s" in note
    assert ("cannot declare its task" in note) is continues

    unsent = _OmniStub()
    result = _run(unsent, tmp_path, task="extend", duration_seconds=None, **source)
    assert "duration" not in unsent.response_format()
    note = duration_note(result)
    assert "sends no duration" in note
    assert "duration is sent" not in note.lower()


def test_a_conversational_extension_read_as_an_edit_gets_advice(
    tmp_path: Path,
) -> None:
    """The one 400 a documented per-turn duration can meet, made actionable.

    A conversational turn cannot declare its task; if the service reads an
    extension prompt as an edit it refuses the duration. The raw message names
    neither the cause nor the fix, mid-chain, after earlier turns billed.
    """
    stub = _OmniStub(
        create_error=(
            400,
            "Duration cannot be set in response format for edit task",
        )
    )
    submission = omni.OmniSubmission()
    with pytest.raises(ValueError, match="read this continuation") as raised:
        _run(
            stub,
            tmp_path,
            prompt="Continue",
            previous_interaction_id="prior",
            task="extend",
            duration_seconds=4,
            submission=submission,
        )
    assert "Nothing was rendered or billed" in str(raised.value)
    assert "prior" in str(raised.value)
    # What was wrapped is the SDK's own typed 4xx, not a failure to parse.
    assert getattr(raised.value.__cause__, "status_code", None) == 400
    # Refused before any work, so nothing is reported as possibly billing.
    assert submission.state(raised.value) == "none"


def test_omni_fallbacks_use_the_models_own_maximum() -> None:
    """OMNI_MAX_DURATION_SECONDS is the longest of ANY model, not each one's.

    It became a flat 15s with 1.2, so the module's own fallbacks read the
    spec: a model whose maximum is not 10s is priced at its own ceiling.
    """
    from dataclasses import replace

    assert omni.OMNI_MAX_DURATION_SECONDS == max(
        spec.max_duration_seconds for spec in omni.OMNI_SPECS.values()
    )
    assert omni.omni_spec(omni.OMNI_PREVIEW_MODEL).max_duration_seconds == 10
    twelve = replace(omni.omni_spec(omni.OMNI_PREVIEW_MODEL), max_duration_seconds=12)
    assert omni.omni_extension_priced_lengths(twelve, None, 2) == [12.0, 12.0]


def test_the_sunset_refusal_points_at_the_default_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Not at 1.1, which this server has retired."""
    monkeypatch.setattr(omni, "_today", lambda: datetime.date(2026, 11, 9))
    refusal = omni.omni_model_refusal(omni.OMNI_PREVIEW_MODEL)
    assert refusal is not None
    # The facts in the message are 1.2's, so the default must be 1.2.
    assert omni.DEFAULT_OMNI_MODEL == omni.OMNI_1_2_MODEL
    assert f"Use {omni.DEFAULT_OMNI_MODEL}" in refusal
    assert omni.OMNI_1_1_MODEL not in refusal
    assert "Vertex AI" in refusal and "15s" in refusal


# ---------------------------------------------------------------------------
# 1.2 behaviour is read off the spec, not off the model name
# ---------------------------------------------------------------------------


def test_a_model_with_1_2s_fields_behaves_like_1_2() -> None:
    """OmniModelSpec promises a new model is "a table entry rather than a new
    code path". A copy of 1.2's record under another name must get 1.2's
    rules everywhere they are read off the spec; a branch on the model name
    would leave the copy with the preview's."""
    import dataclasses

    clone = dataclasses.replace(
        omni.OMNI_SPECS[omni.OMNI_1_2_MODEL], model="gemini-omni-next"
    )
    # Source limits: 15s for an edit and an upscale, on both backends and for
    # a multi-turn source too; none for an extension.
    assert omni.omni_source_limit_seconds(clone) == 15.0
    assert omni.omni_source_limit_seconds(clone, task="upscale") == 15.0
    assert omni.omni_source_limit_seconds(clone, task="extend") == 0.0
    # Past 10s the output is over the inline limit at any resolution.
    assert omni.wants_uri_delivery(clone, "720p", None, 12)
    assert not omni.wants_uri_delivery(clone, "720p", None, 10)
    # The per-turn length is the caller's, and an unknown source is taken at
    # the ceiling rather than at an upload cap.
    assert omni.omni_extension_output_lengths(clone, 20.0, 1, per_turn_seconds=3) == [
        23.0
    ]
    assert omni.omni_extension_priced_lengths(clone, None, 1) == [30.0]
    # An edit never renders past the model's maximum, whatever the source.
    assert omni.omni_continuation_upper_bound(clone, "edit", 14.0) == 15.0
    # The Vertex edit image cap.
    with pytest.raises(ValueError, match="at most 2 images"):
        omni.validate_omni_request(
            omni.OMNI_1_2_MODEL,
            reference_image_count=3,
            is_continuation=True,
            vertexai=True,
        )
    assert clone.max_vertex_edit_images == 2


@pytest.mark.parametrize(
    "kwargs, refused",
    [
        # Every role counts toward the same cap.
        (
            {
                "has_first_frame": True,
                "has_last_frame": True,
                "reference_image_count": 1,
            },
            True,
        ),
        ({"inferred_image_count": 2, "reference_image_count": 0}, False),
        ({"inferred_image_count": 3}, True),
        # Not an edit: an extension and a fresh render are not capped.
        ({"reference_image_count": 3, "task": "extend"}, False),
        ({"reference_image_count": 3, "is_continuation": False}, False),
        # Not Vertex.
        ({"reference_image_count": 3, "vertexai": False}, False),
        # Not a model with the cap.
        ({"inferred_image_count": 3, "model": omni.OMNI_PREVIEW_MODEL}, False),
    ],
)
def test_the_vertex_edit_image_cap_has_one_rule(kwargs: dict, refused: bool) -> None:
    args = {"is_continuation": True, "vertexai": True, **kwargs}
    model = args.pop("model", omni.OMNI_1_2_MODEL)
    if refused:
        with pytest.raises(ValueError, match="at most 2 images"):
            omni.validate_omni_request(model, **args)
    else:
        omni.validate_omni_request(model, **args)


def test_an_over_long_1_2_interaction_edit_says_how_to_proceed(
    tmp_path: Path,
) -> None:
    prior = server.PriorInteraction(
        model=omni.OMNI_1_2_MODEL, backend="vertex", duration_seconds=20.0
    )
    spec = omni.omni_spec(omni.OMNI_1_2_MODEL)
    with pytest.raises(ValueError) as refused:
        server._validate_omni_continuation_source(spec, prior, "prior")
    message = str(refused.value)
    assert "up to 15 seconds" in message and "20.00s" in message
    assert "input_video_uri" in message
    # Extension and upscale check the length where they measure the source.
    server._validate_omni_continuation_source(spec, prior, "prior", task="extend")


@pytest.mark.parametrize(
    "kwargs, fragment",
    [
        ({"reference_image_count": 7}, "at most 6"),
        (
            {"has_first_frame": True, "has_last_frame": True, "duration_seconds": 15},
            "3 to 10 seconds",
        ),
        ({"reference_video_count": 2, "duration_seconds": 12}, "11 to 15 seconds"),
        ({"reference_video_count": 1, "task": "edit"}, "only for new generation"),
        ({"task": "extend", "duration_seconds": 12}, "3 to 10 seconds per turn"),
    ],
)
def test_the_limits_follow_the_spec_not_the_model_name(
    monkeypatch: pytest.MonkeyPatch, kwargs: dict[str, Any], fragment: str
) -> None:
    """validate_omni_request checked these only when spec.model was 1.2's id,
    with the numbers written in, so a model with the same capabilities under
    another name was never checked at all."""
    import dataclasses

    twin = dataclasses.replace(
        omni.omni_spec(omni.OMNI_1_2_MODEL), model="gemini-omni-twin-flash"
    )
    monkeypatch.setitem(omni.OMNI_SPECS, twin.model, twin)
    for model in (omni.OMNI_1_2_MODEL, twin.model):
        with pytest.raises(ValueError, match=fragment):
            omni.validate_omni_request(model, **kwargs)
