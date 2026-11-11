"""Reference limits, 429s, collecting a render by id, animatic bridges, loops.

1. Veo routes are not planned for more than 3 reference images.
2. A create refused with 429 is sent again; a status read that fails with a
   429 or a 5xx is read again.
3. get_omni_interaction collects a render by its interaction_id.
4. An animatic bridges its beats on Omni 1.2.
5. A loop (one image as first and last frame) sends the image once.

The network tests run the real google-genai client against a local stub, so
a 429 or a 503 reaches the code as the SDK's own typed error.
"""

import asyncio
import base64
import contextlib
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from google import genai
from google.genai.types import HttpOptions, HttpRetryOptions

import src.__main__ as server
import src.omni as omni
from src.routing import RoutingConstraints, plan_generation
from tests.test_omni_1_2 import context
from tests.test_omni_1_2_tools import _probe

_MP4 = b"\x00\x00\x00\x18ftypmp42" + b"LOOP" * 16


class _Scripted:
    """An Interactions endpoint whose every call answers from a script.

    ``creates`` and ``gets`` are consumed in order, one entry per call: an
    int is an HTTP error status, ``"in_progress"`` an in-flight body, and
    anything else (or an exhausted script) the final body.
    """

    def __init__(
        self,
        creates: list[Any] | None = None,
        gets: list[Any] | None = None,
        *,
        final: str = "completed",
        retry_after: str | None = None,
        gcs_uri: str | None = None,
    ) -> None:
        self.creates = list(creates or [])
        self.gets = list(gets or [])
        self.final = final
        self.retry_after = retry_after
        self.gcs_uri = gcs_uri
        self.requests: list[tuple[str, str, dict[str, Any]]] = []

    def body(self, status: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": "i-9",
            "name": "interactions/i-9",
            "status": status,
            "model": omni.OMNI_1_2_MODEL,
        }
        if status == "failed":
            payload["errors"] = [{"code": "internal", "message": "the render broke"}]
        if status == "completed":
            part: dict[str, Any] = {"type": "video", "mimeType": "video/mp4"}
            if self.gcs_uri:
                part["uri"] = self.gcs_uri
            else:
                part["data"] = base64.b64encode(_MP4).decode()
            payload["steps"] = [{"content": [part]}]
        return payload

    def posts(self) -> list[dict[str, Any]]:
        return [body for method, _, body in self.requests if method == "POST"]

    def reads(self) -> int:
        return sum(1 for method, _, _ in self.requests if method == "GET")


def _handler(stub: _Scripted) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            """Silence the access log."""

        def _send(self, payload: dict[str, Any], status: int = 200) -> None:
            raw = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            if status == 429 and stub.retry_after is not None:
                self.send_header("Retry-After", stub.retry_after)
            self.end_headers()
            _ = self.wfile.write(raw)

        def _answer(self, step: Any) -> None:
            if isinstance(step, int):
                self._send(
                    {
                        "error": {
                            "code": "resource_exhausted",
                            "message": f"HTTP {step}",
                        }
                    },
                    status=step,
                )
            elif step == "in_progress":
                self._send(stub.body("in_progress"))
            else:
                self._send(stub.body(stub.final))

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            length = int(self.headers.get("Content-Length", 0))
            stub.requests.append(
                ("POST", self.path, json.loads(self.rfile.read(length) or b"{}"))
            )
            step = stub.creates.pop(0) if stub.creates else None
            if step is None:
                # Accepted: in flight when reads are scripted, else done.
                step = "in_progress" if stub.gets else "final"
            self._answer(step)

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            stub.requests.append(("GET", self.path, {}))
            self._answer(stub.gets.pop(0) if stub.gets else "final")

    return Handler


@contextlib.contextmanager
def _client(stub: _Scripted, *, sdk_retries: bool = False) -> Iterator[genai.Client]:
    """A real client on the stub, with the SDK's own retries off unless asked.

    google-genai's Interactions client retries a 408, 409, 429 or 5xx itself:
    up to 3 more attempts, 0.5s, 1s and 2s apart. Off, each scripted status
    reaches the code under test as it is.
    """
    httpd = HTTPServer(("127.0.0.1", 0), _handler(stub))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    options = HttpOptions(base_url=f"http://127.0.0.1:{httpd.server_address[1]}")
    if not sdk_retries:
        options.retry_options = HttpRetryOptions(attempts=0)
    try:
        yield genai.Client(api_key="stub-key-not-real", http_options=options)
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.fixture
def waits(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Every asyncio.sleep the code asks for, returned at once."""
    asked: list[float] = []
    real_sleep = asyncio.sleep

    async def instant(seconds: float) -> None:
        asked.append(seconds)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", instant)
    return asked


def _render(
    stub: _Scripted,
    tmp_path: Path,
    submission: omni.OmniSubmission | None = None,
    *,
    sdk_retries: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    videos = tmp_path / "videos"
    videos.mkdir(exist_ok=True)
    with _client(stub, sdk_retries=sdk_retries) as client:
        return asyncio.run(
            omni.generate_video_omni(
                client=client,
                prompt="a fox in snow",
                videos_dir=videos,
                duration_seconds=3,
                submission=submission,
                **kwargs,
            )
        )


# ---------------------------------------------------------------------------
# 1. Veo routes and reference images
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("references", [4, 6])
def test_veo_routes_are_not_planned_for_more_than_3_references(references: int) -> None:
    plan = plan_generation(
        "a video of a cat",
        RoutingConstraints(media_kind="video", num_reference_images=references),
    )
    models = [r.model for r in plan.routes]
    assert omni.OMNI_1_2_MODEL in models
    assert not [m for m in models if m.startswith("veo-")], models
    veo = [r for r in plan.rejected if r.model.startswith("veo-")]
    assert len(veo) == 2
    assert all(
        f"takes at most 3 reference images, and {references} were asked for" in r.reason
        for r in veo
    ), veo


def test_veo_routes_are_still_planned_for_3_references() -> None:
    plan = plan_generation(
        "a video of a cat",
        RoutingConstraints(media_kind="video", num_reference_images=3),
    )
    assert any(r.model.startswith("veo-") for r in plan.routes), plan.routes


def test_a_pinned_veo_model_with_5_references_points_at_omni() -> None:
    plan = plan_generation(
        "a video of a cat",
        RoutingConstraints(
            media_kind="video",
            num_reference_images=5,
            pinned_model="veo-3.1-fast-generate-001",
        ),
    )
    [conflict] = plan.conflicts
    assert conflict.code == "pinned_model_too_many_reference_images", conflict
    assert conflict.resolution == (
        "Use gemini-omni-1.2-flash, which takes up to 6 reference images, or "
        "pass at most 3 for Veo."
    )


def test_7_references_give_one_fix_not_two(monkeypatch: pytest.MonkeyPatch) -> None:
    """Omni and Veo refuse the same brief; their fixes read as one sentence."""
    import src.routing as routing

    monkeypatch.setattr(routing, "omni_preview_is_sunset", lambda: True)
    plan = plan_generation(
        "a video of a cat",
        RoutingConstraints(media_kind="video", num_reference_images=7),
    )
    [conflict] = plan.conflicts
    assert conflict.code == "no_model_can_serve_request"
    assert conflict.resolution == (
        "Pass at most 6 reference images for gemini-omni-1.2-flash, or at most 3 for Veo."
    )


# ---------------------------------------------------------------------------
# 2. 429s
# ---------------------------------------------------------------------------


class _Refused(Exception):
    """An SDK HTTP error as the code reads it: a status, and maybe Retry-After."""

    def __init__(self, status: int, retry_after: str | None = None) -> None:
        super().__init__(f"HTTP {status}")
        self.status_code = status
        self.response = SimpleNamespace(
            headers={"retry-after": retry_after} if retry_after else {}
        )


def _interaction(status: str) -> dict[str, Any]:
    body: dict[str, Any] = {"id": "i-9", "status": status}
    if status == "completed":
        body["steps"] = [
            {
                "content": [
                    {
                        "type": "video",
                        "mime_type": "video/mp4",
                        "data": base64.b64encode(_MP4).decode(),
                    }
                ]
            }
        ]
    return body


def _fake(creates: list[Any], gets: list[Any] | None = None) -> MagicMock:
    """A client whose create and get answer in turn: an exception is raised."""
    client = MagicMock()
    client._api_client.vertexai = False
    client.interactions.create.side_effect = creates
    client.interactions.get.side_effect = gets or []
    return client


def _render_on(
    client: MagicMock,
    tmp_path: Path,
    submission: omni.OmniSubmission | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    videos = tmp_path / "videos"
    videos.mkdir(exist_ok=True)
    return asyncio.run(
        omni.generate_video_omni(
            client=client,
            prompt="a fox in snow",
            videos_dir=videos,
            duration_seconds=3,
            submission=submission,
            **kwargs,
        )
    )


@pytest.mark.timeout(20.0)
def test_a_create_refused_with_429_is_sent_again(
    tmp_path: Path, waits: list[float]
) -> None:
    client = _fake([_Refused(429), _interaction("completed")])
    submission = omni.OmniSubmission()
    result = _render_on(client, tmp_path, submission)
    first, second = client.interactions.create.call_args_list
    assert first == second, "the resend is the same request"
    assert waits[0] == 10.0, waits
    assert submission.quota_retries == 1
    assert any("429" in w and "accepted" in w for w in result["warnings"]), result


@pytest.mark.timeout(30.0)
def test_the_sdks_own_retries_run_first(tmp_path: Path, waits: list[float]) -> None:
    """google-genai resends a 429 within seconds; the tool's longer wait follows.

    Four refusals use up the SDK's attempts (0.5s, 1s and 2s apart, in real
    time), and only then does the tool wait its 10s and send it once more.
    """
    stub = _Scripted(creates=[429, 429, 429, 429])
    submission = omni.OmniSubmission()
    _render(stub, tmp_path, submission, sdk_retries=True)
    assert len(stub.posts()) == 5
    assert submission.quota_retries == 1
    assert waits[0] == 10.0, waits


@pytest.mark.timeout(20.0)
def test_a_429_waits_as_long_as_it_asks(tmp_path: Path, waits: list[float]) -> None:
    _render_on(_fake([_Refused(429, "3"), _interaction("completed")]), tmp_path)
    assert waits[0] == 3.0, waits


@pytest.mark.timeout(20.0)
def test_a_429_asking_too_long_waits_a_minute(
    tmp_path: Path, waits: list[float]
) -> None:
    client = _fake([_Refused(429, "900"), _interaction("completed")])
    _render_on(client, tmp_path, timeout_seconds=600)
    assert waits[0] == 60.0, waits


@pytest.mark.timeout(20.0)
def test_a_create_still_refused_after_two_resends_raises_the_429(
    tmp_path: Path, waits: list[float]
) -> None:
    client = _fake([_Refused(429), _Refused(429), _Refused(429)])
    submission = omni.OmniSubmission()
    with pytest.raises(_Refused) as caught:
        _render_on(client, tmp_path, submission)
    assert server.http_status(caught.value) == 429
    assert client.interactions.create.call_count == 3
    assert waits[:2] == [10.0, 20.0], waits
    assert submission.quota_retries == 2
    assert submission.state(caught.value) == "none", "a refusal billed nothing"


@pytest.mark.timeout(20.0)
def test_no_resend_without_a_renders_worth_of_deadline(
    tmp_path: Path, waits: list[float]
) -> None:
    client = _fake([_Refused(429), _interaction("completed")])
    submission = omni.OmniSubmission()
    with pytest.raises(_Refused):
        _render_on(client, tmp_path, submission, timeout_seconds=60)
    assert client.interactions.create.call_count == 1
    assert submission.quota_retries == 0


@pytest.mark.timeout(20.0)
def test_any_other_refusal_is_not_sent_again(
    tmp_path: Path, waits: list[float]
) -> None:
    client = _fake([_Refused(400), _interaction("completed")])
    with pytest.raises(_Refused):
        _render_on(client, tmp_path)
    assert client.interactions.create.call_count == 1


@pytest.mark.timeout(20.0)
@pytest.mark.parametrize("status", [429, 503])
def test_a_failed_status_read_is_read_again(
    tmp_path: Path, waits: list[float], status: int
) -> None:
    client = _fake(
        [_interaction("in_progress")],
        [_Refused(status), _interaction("completed")],
    )
    result = _render_on(client, tmp_path)
    assert result["interaction_id"] == "i-9"
    assert client.interactions.get.call_count == 2
    assert client.interactions.create.call_count == 1, "a read never re-creates"


@pytest.mark.timeout(20.0)
def test_three_failed_reads_in_a_row_end_the_wait(
    tmp_path: Path, waits: list[float]
) -> None:
    client = _fake(
        [_interaction("in_progress")],
        [_Refused(503), _Refused(503), _Refused(503), _interaction("completed")],
    )
    submission = omni.OmniSubmission()
    with pytest.raises(_Refused) as caught:
        _render_on(client, tmp_path, submission)
    assert client.interactions.get.call_count == 3
    assert submission.state(caught.value) == "submitted"


@pytest.mark.timeout(20.0)
def test_a_read_refused_outright_is_not_read_again(
    tmp_path: Path, waits: list[float]
) -> None:
    client = _fake(
        [_interaction("in_progress")], [_Refused(404), _interaction("completed")]
    )
    with pytest.raises(_Refused):
        _render_on(client, tmp_path)
    assert client.interactions.get.call_count == 1


@pytest.mark.timeout(20.0)
def test_a_create_answered_with_a_server_error_goes_out_once(
    tmp_path: Path, waits: list[float]
) -> None:
    """A 5xx may follow a render the service already started; a resend bills twice.

    google-genai resends a 503'd create 3 more times by default (4 requests,
    measured against this stub). The Interactions resource is narrowed to
    resend only a 408 or a 429.
    """
    stub = _Scripted(creates=[503, 503, 503, 503])
    submission = omni.OmniSubmission()
    with pytest.raises(Exception) as caught:
        _render(stub, tmp_path, submission, sdk_retries=True)
    assert server.http_status(caught.value) == 503
    assert len(stub.posts()) == 1
    assert submission.state(caught.value) == "unknown", "it may have reached Google"


@pytest.mark.timeout(20.0)
def test_a_status_read_answered_with_a_server_error_is_read_again_by_the_tool(
    tmp_path: Path, waits: list[float]
) -> None:
    """With the SDK's 5xx resends off, the tool's own read-again covers reads."""
    stub = _Scripted(gets=["in_progress", 503, 503])
    result = _render(stub, tmp_path, sdk_retries=True)
    assert result["interaction_id"] == "i-9"
    assert stub.reads() == 4
    assert len(stub.posts()) == 1


@pytest.mark.timeout(20.0)
def test_a_dropped_connection_on_a_read_is_read_again(
    tmp_path: Path, waits: list[float]
) -> None:
    class APIConnectionError(Exception):
        """Named as google-genai's Interactions client names it."""

    client = _fake(
        [_interaction("in_progress")],
        [APIConnectionError("Connection error."), _interaction("completed")],
    )
    result = _render_on(client, tmp_path)
    assert result["interaction_id"] == "i-9"
    assert client.interactions.get.call_count == 2


@pytest.mark.timeout(20.0)
def test_only_the_interactions_resource_is_narrowed(
    tmp_path: Path, waits: list[float]
) -> None:
    """The models.* calls Veo and the image tools make keep their own default."""
    stub = _Scripted()
    videos = tmp_path / "videos"
    videos.mkdir()
    with _client(stub, sdk_retries=True) as client:
        asyncio.run(
            omni.generate_video_omni(
                client=client, prompt="a fox", videos_dir=videos, duration_seconds=3
            )
        )
        retry: Any = client.interactions.sdk_configuration.retry_config
        assert retry.status_codes_override == ["408", "429"]
        assert retry.retry_connection_errors is False
        assert client._api_client._http_options.retry_options is None  # pyright: ignore[reportPrivateUsage]


def test_the_director_does_not_stack_its_waits_on_the_tools() -> None:
    refused = {"error": "429", "status_code": 429, "submission": "none"}
    assert server._render_retry_reason(refused) == "quota"  # pyright: ignore[reportPrivateUsage]
    assert server._render_retry_reason({**refused, "quota_retries": 2}) is None  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
@pytest.mark.timeout(20.0)
async def test_the_tool_reports_how_often_it_resent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Refused(Exception):
        code = 429

    async def refused(**kwargs: Any) -> dict[str, Any]:
        kwargs["submission"].quota_retries = 2
        raise Refused("quota")

    monkeypatch.setattr(server, "generate_video_omni_impl", refused)
    body = json.loads(
        await server.generate_video_omni(
            ctx=context(tmp_path), prompt="a fox", duration_seconds=3
        )
    )
    assert body["status_code"] == 429
    assert body["quota_retries"] == 2


# ---------------------------------------------------------------------------
# 3. get_omni_interaction
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _tool_context(tmp_path: Path, stub: _Scripted) -> Iterator[Any]:
    ctx = context(tmp_path)
    with _client(stub) as client:
        ctx.request_context.lifespan_context.client = client
        yield ctx


async def _collect(ctx: Any, **kwargs: Any) -> dict[str, Any]:
    return json.loads(await server.get_omni_interaction(ctx=ctx, **kwargs))


@pytest.mark.asyncio
@pytest.mark.timeout(20.0)
async def test_a_render_in_progress_is_reported_and_nothing_saved(
    tmp_path: Path, waits: list[float]
) -> None:
    stub = _Scripted(gets=["in_progress"])
    with _tool_context(tmp_path, stub) as ctx:
        body = await _collect(ctx, interaction_id="i-9")
    assert body["status"] == "in_progress", body
    assert "video_url" not in body
    assert "nothing was saved" in body["message"]
    assert not list((tmp_path / "videos").glob("*.mp4"))
    assert stub.posts() == [], "collecting never creates anything"


@pytest.mark.asyncio
@pytest.mark.timeout(20.0)
async def test_a_completed_render_is_saved_measured_priced_and_recorded(
    tmp_path: Path, waits: list[float], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server, "_probe_media", _probe(3.03, (640, 360)))
    stub = _Scripted(gets=["in_progress", "in_progress"])
    with _tool_context(tmp_path, stub) as ctx:
        body = await _collect(ctx, interaction_id="i-9", wait_seconds=60)
    assert body["status"] == "completed", body
    saved = Path(body["video_url"][7:])
    assert saved.read_bytes() == _MP4
    assert body["duration_seconds"] == 3.03
    assert body["resolution"] == "360p"
    assert body["model"] == omni.OMNI_1_2_MODEL
    assert body["cost"]["detail"].count("360p"), body["cost"]
    assert body["cost"]["is_estimate"] is False
    assert "no record of the request" in body["cost"]["detail"]
    # Recorded like any render: a sidecar beside the clip, so a continuation
    # finds its backend.
    sidecar = json.loads(Path(body["sidecar_url"][7:]).read_text())
    assert sidecar["collected_by"] == "get_omni_interaction"
    assert sidecar["interaction_id"] == "i-9"
    assert sidecar["cost"] == body["cost"]
    prior = server._prior_interaction(tmp_path / "videos", "i-9")  # pyright: ignore[reportPrivateUsage]
    assert prior.backend == "gemini_api"
    assert prior.duration_seconds == 3.03 and prior.duration_measured
    assert prior.resolution == "360p"
    assert stub.posts() == []


@pytest.mark.asyncio
@pytest.mark.timeout(20.0)
async def test_a_saved_render_is_returned_from_its_record(
    tmp_path: Path, waits: list[float], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server, "_probe_media", _probe(3.03, (640, 360)))
    stub = _Scripted()
    with _tool_context(tmp_path, stub) as ctx:
        first = await _collect(ctx, interaction_id="i-9")
        reads = stub.reads()
        again = await _collect(ctx, interaction_id="i-9")
    assert stub.reads() == reads, "the second call read nothing"
    assert again["video_url"] == first["video_url"]
    assert again["status"] == "completed"
    assert "not read" in again["message"]


@pytest.mark.asyncio
@pytest.mark.timeout(20.0)
async def test_a_failed_render_reports_the_services_reason(
    tmp_path: Path, waits: list[float]
) -> None:
    stub = _Scripted(final="failed")
    with _tool_context(tmp_path, stub) as ctx:
        body = await _collect(ctx, interaction_id="i-9")
    assert body["status"] == "failed"
    assert "the render broke" in body["error"], body


@pytest.mark.asyncio
@pytest.mark.timeout(20.0)
async def test_an_unknown_id_says_which_backend_was_read(
    tmp_path: Path, waits: list[float]
) -> None:
    stub = _Scripted(gets=[404])
    with _tool_context(tmp_path, stub) as ctx:
        body = await _collect(ctx, interaction_id="i-9")
    assert "could not be read on the Gemini Developer API" in body["error"], body
    assert 'omni_backend="vertex"' in body["error"]


@pytest.mark.asyncio
async def test_a_backend_contradicting_the_record_is_refused_unread(
    tmp_path: Path,
) -> None:
    ctx = context(tmp_path)
    server._write_remote_interaction_record(  # pyright: ignore[reportPrivateUsage]
        tmp_path / "videos",
        {"interaction_id": "i-9", "backend": "vertex", "status": "pending"},
    )
    body = await _collect(ctx, interaction_id="i-9", omni_backend="gemini_api")
    assert "created on Vertex AI" in body["error"], body
    ctx.request_context.lifespan_context.client.interactions.get.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.timeout(20.0)
async def test_an_upscale_recorded_as_pending_is_priced_as_an_upscale(
    tmp_path: Path, waits: list[float], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server, "_probe_media", _probe(3.03, (1280, 720)))
    server._write_remote_interaction_record(  # pyright: ignore[reportPrivateUsage]
        tmp_path / "videos",
        {
            "interaction_id": "i-9",
            "backend": "gemini_api",
            "status": "pending",
            "task": "upscale",
            "upscale_source_resolution": "360p",
        },
    )
    stub = _Scripted()
    with _tool_context(tmp_path, stub) as ctx:
        body = await _collect(ctx, interaction_id="i-9")
    assert body["task"] == "upscale"
    assert body["cost"]["breakdown"]["usd_per_second"] == pytest.approx(0.034752)


@pytest.mark.asyncio
@pytest.mark.timeout(20.0)
async def test_a_pending_render_with_no_task_is_priced_as_the_render_it_was(
    tmp_path: Path, waits: list[float], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new render records no task (it is inferred); that is not "no record".

    Live, L13 collected its own timed-out render and said this server had no
    record of the request, beside the pending record the timeout had written.
    """
    monkeypatch.setattr(server, "_probe_media", _probe(3.03, (640, 360)))
    server._write_remote_interaction_record(  # pyright: ignore[reportPrivateUsage]
        tmp_path / "videos",
        {
            "interaction_id": "i-9",
            "backend": "gemini_api",
            "status": "pending",
            "task": None,
        },
    )
    stub = _Scripted()
    with _tool_context(tmp_path, stub) as ctx:
        body = await _collect(ctx, interaction_id="i-9")
    assert body["cost"]["usd"] == pytest.approx(0.023172 * 3.03, abs=1e-6)
    assert "no record" not in body["cost"]["detail"], body["cost"]


@pytest.mark.asyncio
@pytest.mark.timeout(20.0)
async def test_a_render_that_outlives_its_call_is_recorded_as_pending(
    tmp_path: Path, waits: list[float], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The record a timeout leaves tells get_omni_interaction where to read."""

    async def created_then_timed_out(**kwargs: Any) -> dict[str, Any]:
        kwargs["submission"].interaction_id = "i-late"
        raise TimeoutError("Omni video interaction timed out after 210s.")

    monkeypatch.setattr(server, "generate_video_omni_impl", created_then_timed_out)
    ctx = context(tmp_path)
    body = json.loads(
        await server.generate_video_omni(ctx=ctx, prompt="a fox", duration_seconds=3)
    )
    assert body["interaction_id"] == "i-late"
    assert "get_omni_interaction" in body["note"], body
    manifest = server._manifest_for_interaction(tmp_path / "videos", "i-late")  # pyright: ignore[reportPrivateUsage]
    assert manifest is not None
    assert manifest["status"] == "pending"
    assert manifest["backend"] == "gemini_api"
    assert manifest["model"] == omni.OMNI_1_2_MODEL


def test_the_timeout_error_names_the_collecting_tool() -> None:
    import inspect

    assert "Collect it with get_omni_interaction" in " ".join(
        inspect.getsource(omni.generate_video_omni).split()
    ).replace('" "', "")


# ---------------------------------------------------------------------------
# 4. Animatic bridges
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_animatic_quote_prices_its_bridges_at_the_animatic_resolution(
    tmp_path: Path,
) -> None:
    ctx = context(tmp_path)
    beats = [
        {"prompt": "a", "duration_seconds": 4},
        {"prompt": "b", "duration_seconds": 4},
    ]
    with_bridges = json.loads(
        await server.generate_clip(
            ctx=ctx,
            beats=beats,
            animatic=True,
            animatic_resolution="360p",
            add_bridges=True,
            dry_run=True,
        )
    )
    without = json.loads(
        await server.generate_clip(
            ctx=ctx,
            beats=beats,
            animatic=True,
            animatic_resolution="360p",
            dry_run=True,
        )
    )
    assert with_bridges["bridge_count"] == 1
    assert without["bridge_count"] == 0
    added = with_bridges["estimated_cost"]["usd"] - without["estimated_cost"]["usd"]
    # One 4s bridge at 360p, quoted a frame high like the beats.
    assert added == pytest.approx(0.023172 * (4 + 1 / 24), abs=1e-5)
    assert not any("add_bridges" in w for w in with_bridges.get("warnings", []))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "passed, warned",
    [
        pytest.param({}, False, id="unset"),
        pytest.param({"include_audio": True}, True, id="true"),
        pytest.param({"include_audio": False}, True, id="false"),
    ],
)
async def test_an_animatic_warns_about_include_audio_only_when_it_was_passed(
    tmp_path: Path, passed: dict[str, Any], warned: bool
) -> None:
    """Its True default drew "include_audio is ignored" on every animatic."""
    quote = json.loads(
        await server.generate_clip(
            ctx=context(tmp_path),
            beats=[{"prompt": "a", "duration_seconds": 4}],
            animatic=True,
            dry_run=True,
            **passed,
        )
    )
    said = any("include_audio" in w for w in quote.get("warnings", []))
    assert said is warned, quote.get("warnings")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "passed, sent",
    [
        pytest.param({}, True, id="unset"),
        pytest.param({"include_audio": False}, False, id="false"),
    ],
)
async def test_a_veo_clip_still_has_audio_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, passed: dict[str, Any], sent: bool
) -> None:
    seen: list[Any] = []

    async def veo(**kwargs: Any) -> dict[str, Any]:
        seen.append(kwargs["include_audio"])
        raise RuntimeError("stop after the request is built")

    monkeypatch.setattr(server, "generate_video_impl", veo)
    await server.generate_clip(
        ctx=context(tmp_path, vertex=True),
        beats=[{"prompt": "a", "duration_seconds": 4}],
        **passed,
    )
    assert seen == [sent]


# ---------------------------------------------------------------------------
# 5. A loop sends its image once
# ---------------------------------------------------------------------------


_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02"
    b"\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe"
    b"\x02\xfe\xa7\x9a\xa0\xa3\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _images_and_text(request: dict[str, Any]) -> tuple[list[str], str]:
    # The SDK sends the parts inside one user_input turn.
    parts = [
        part for turn in request["input"] for part in turn.get("content") or [turn]
    ]
    text = " ".join(p["text"] for p in parts if p["type"] == "text")
    return [p["data"] for p in parts if p["type"] == "image"], text


@pytest.mark.timeout(20.0)
def test_a_loop_sends_its_image_once(tmp_path: Path, waits: list[float]) -> None:
    stub = _Scripted()
    result = _render(stub, tmp_path, first_frame_bytes=_PNG, last_frame_bytes=_PNG)
    images, text = _images_and_text(stub.posts()[0])
    assert len(images) == 1
    assert "[# Sources <FIRST_FRAME>@Image1 <LAST_FRAME>@Image1]" in text
    assert "Use Image1 as both the first frame and the last frame." in text
    assert result["effective_prompt"] == text


@pytest.mark.timeout(20.0)
def test_a_loop_numbers_its_references_from_the_second_image(
    tmp_path: Path, waits: list[float]
) -> None:
    other = _PNG.replace(b"\x90wS\xde", b"\x90wS\xdf")
    stub = _Scripted()
    _render(
        stub,
        tmp_path,
        first_frame_bytes=_PNG,
        last_frame_bytes=_PNG,
        reference_image_bytes_list=[other],
    )
    images, text = _images_and_text(stub.posts()[0])
    assert len(images) == 2
    assert "<IMAGE_REF_0>@Image2" in text


@pytest.mark.timeout(20.0)
def test_two_different_frames_still_send_two_images(
    tmp_path: Path, waits: list[float]
) -> None:
    other = _PNG.replace(b"\x90wS\xde", b"\x90wS\xdf")
    stub = _Scripted()
    _render(stub, tmp_path, first_frame_bytes=_PNG, last_frame_bytes=other)
    images, text = _images_and_text(stub.posts()[0])
    assert len(images) == 2
    assert "<LAST_FRAME>@Image2" in text


@pytest.mark.timeout(20.0)
def test_a_loop_under_the_callers_own_tags_sends_both_images(
    tmp_path: Path, waits: list[float]
) -> None:
    """The caller's tags number the images; dropping one would shift them."""
    stub = _Scripted()
    videos = tmp_path / "videos"
    videos.mkdir()
    with _client(stub) as client:
        asyncio.run(
            omni.generate_video_omni(
                client=client,
                prompt="[# Sources <FIRST_FRAME>@Image1 <LAST_FRAME>@Image2] a loop",
                videos_dir=videos,
                duration_seconds=3,
                first_frame_bytes=_PNG,
                last_frame_bytes=_PNG,
            )
        )
    images, _ = _images_and_text(stub.posts()[0])
    assert len(images) == 2


@pytest.mark.asyncio
async def test_a_loop_fetches_its_image_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fetched: list[str] = []

    async def fetch(ctx: Any, uri: str | None, label: str) -> bytes | None:
        if uri is None:
            return None
        fetched.append(uri)
        return _PNG

    captured: dict[str, Any] = {}

    async def impl(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        raise RuntimeError("stop after the request is built")

    monkeypatch.setattr(server, "_fetch_optional_omni_media", fetch)
    monkeypatch.setattr(server, "generate_video_omni_impl", impl)
    uri = f"file://{tmp_path}/images/still.png"
    await server.generate_video_omni(
        ctx=context(tmp_path),
        prompt="a loop",
        first_frame_uri=uri,
        last_frame_uri=uri,
        duration_seconds=3,
    )
    assert fetched == [uri]
    assert captured["first_frame_bytes"] is captured["last_frame_bytes"]


def test_the_rename_left_no_reference_to_the_old_test_file() -> None:
    root = Path(__file__).parent
    assert not (root / "test_omni_1_1.py").exists()
    assert (root / "test_omni_capabilities.py").exists()


_ = AsyncMock  # kept for parity with the sibling test modules' imports


# ---------------------------------------------------------------------------
# The cheap-budget upscale regression, and a blocked workflow
# ---------------------------------------------------------------------------

_MUG = "a 3 second clip of a coffee mug, then upscale it to {}"


@pytest.mark.parametrize(
    "target, budget, render",
    [
        pytest.param("4K", "cheap", "720p", id="4K-cheap"),
        pytest.param("4K", "balanced", "720p", id="4K-balanced"),
        pytest.param("1080p", "cheap", "360p", id="1080p-cheap"),
        pytest.param("720p", "cheap", "360p", id="720p-cheap"),
        pytest.param("720p", "balanced", "360p", id="720p-balanced"),
    ],
)
def test_an_upscale_plans_its_render_at_a_resolution_the_target_takes(
    target: str, budget: str, render: str
) -> None:
    """budget=cheap planned a 4K brief's render at 360p, which nothing upscales to 4K."""
    plan = plan_generation(_MUG.format(target), RoutingConstraints(budget=budget))  # type: ignore[arg-type]
    assert plan.is_satisfiable, plan.conflicts
    assert not plan.conflicts, plan.conflicts
    first, second = plan.workflow
    assert second.tool == "upscale_video_omni"
    assert second.params["resolution"] == target
    assert str(first.params.get("resolution") or "720p") == render, first.params


def test_a_cheap_4k_upscale_says_why_its_render_is_not_360p() -> None:
    plan = plan_generation(_MUG.format("4K"), RoutingConstraints(budget="cheap"))
    top = plan.routes[0]
    assert top.params["resolution"] == "720p"
    assert not [c for c in top.caveats if "resolution='360p'" in c], top.caveats
    assert any("the only resolution an upscale to 4K takes" in n for n in plan.notes)


def test_a_conflict_that_empties_the_workflow_is_unsatisfiable() -> None:
    """A 360p render named outright cannot reach 4K: routes survive, the plan does not."""
    plan = plan_generation(
        _MUG.format("4K"), RoutingConstraints(budget="cheap", resolution="360p")
    )
    assert plan.routes, "the render routes are kept for what they say"
    assert [c.code for c in plan.conflicts] == ["upscale_resolution"]
    assert plan.workflow == ()
    assert plan.is_satisfiable is False


@pytest.mark.asyncio
async def test_plan_generation_reports_the_blocked_workflow(tmp_path: Path) -> None:
    # On Vertex AI, where upscaling runs; on the Gemini API the 4K goes on
    # the render itself instead, and there is no upscale step to block.
    out = await server.plan_generation(
        ctx=context(tmp_path, vertex=True),
        intent=_MUG.format("4K"),
        budget="cheap",
        resolution="360p",
    )
    body = json.loads(out[0].text)  # type: ignore[union-attr]
    assert body["is_satisfiable"] is False
    assert body["workflow"] == []


# ---------------------------------------------------------------------------
# A render that ended without completing is not "may still complete"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(20.0)
@pytest.mark.parametrize(
    "status", ["failed", "cancelled", "incomplete", "budget_exceeded"]
)
async def test_an_ended_interactions_note_says_it_will_not_complete(
    tmp_path: Path, waits: list[float], status: str
) -> None:
    stub = _Scripted(final=status)
    with _tool_context(tmp_path, stub) as ctx:
        body = json.loads(
            await server.generate_video_omni(
                ctx=ctx, prompt="a fox", duration_seconds=3
            )
        )
    assert body["submission"] == "submitted", body
    assert body["note"] == (
        f"Interaction i-9 ended with status '{status}' and will not complete; "
        "rendering again starts a new render."
    )
    assert "get_omni_interaction" not in body["note"]


@pytest.mark.asyncio
@pytest.mark.timeout(20.0)
async def test_a_status_not_known_to_be_final_keeps_the_collect_note(
    tmp_path: Path, waits: list[float]
) -> None:
    stub = _Scripted(final="requires_action")
    with _tool_context(tmp_path, stub) as ctx:
        body = json.loads(
            await server.generate_video_omni(
                ctx=ctx, prompt="a fox", duration_seconds=3
            )
        )
    assert "may still complete" in body["note"], body
    assert "get_omni_interaction" in body["note"]


# ---------------------------------------------------------------------------
# An upscale interaction is continued from the clip it saved
# ---------------------------------------------------------------------------


def _upscale_record(
    tmp_path: Path, *, saved: bool = True, resolution: str = "720p"
) -> str:
    """The sidecar an upscale leaves, beside its clip; returns the clip's URI."""
    videos = tmp_path / "videos"
    videos.mkdir(exist_ok=True)
    clip = videos / "up.mp4"
    if saved:
        clip.write_bytes(_MP4)
    (videos / "up.json").write_text(
        json.dumps(
            {
                "kind": "omni_upscale",
                "task": "upscale",
                "interaction_id": "up-1",
                "model": omni.OMNI_1_2_MODEL,
                "backend": "vertex",
                "duration_seconds": 3.03,
                "duration_source": "measured from the rendered video",
                "resolution": resolution,
                "video_url": f"file://{clip}",
            }
        )
    )
    return f"file://{clip}"


def _capture_renders(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    sent: list[dict[str, Any]] = []

    async def render(app_ctx: Any, ctx: Any, **kwargs: Any) -> dict[str, Any]:
        sent.append(kwargs)
        return {
            "message": "Video generated successfully",
            "video_url": "file:///nowhere/out.mp4",
            "interaction_id": f"new-{len(sent)}",
            "model": omni.OMNI_1_2_MODEL,
            "backend": "vertex",
            "duration_seconds": 6.03,
            "duration_source": "measured from the rendered video",
            "resolution": kwargs.get("resolution"),
            "rendered_resolution": kwargs.get("resolution"),
        }

    monkeypatch.setattr(server, "_omni_generate_and_manifest", render)
    return sent


def test_an_upscale_is_recorded_as_one() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        _upscale_record(Path(d))
        prior = server._prior_interaction(Path(d) / "videos", "up-1")  # pyright: ignore[reportPrivateUsage]
    assert prior.task == "upscale"


@pytest.mark.asyncio
async def test_an_upscale_interaction_is_upscaled_from_its_saved_clip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """By id the service refused it: "Exactly one input video is required"."""
    clip = _upscale_record(tmp_path)
    monkeypatch.setattr(server, "_probe_media", _probe(3.03, (1280, 720)))
    sent = _capture_renders(monkeypatch)
    body = json.loads(
        await server.upscale_video_omni(
            context(tmp_path, vertex=True), "4K", previous_interaction_id="up-1"
        )
    )
    [request] = sent
    assert request["previous_interaction_id"] is None
    assert request["input_video_bytes"] == _MP4
    assert request["upscale_source_resolution"] == "720p"
    assert request["manifest_extra"]["source_interaction_id"] == "up-1"
    assert any(
        "up-1 is an upscale" in w and clip in w and "was sent" in w
        for w in body["warnings"]
    ), body.get("warnings")


@pytest.mark.asyncio
async def test_an_upscale_interactions_quote_says_what_it_would_send(
    tmp_path: Path,
) -> None:
    _upscale_record(tmp_path)
    quote = json.loads(
        await server.upscale_video_omni(
            context(tmp_path, vertex=True),
            "4K",
            previous_interaction_id="up-1",
            dry_run=True,
        )
    )
    assert "error" not in quote, quote
    assert any("would be sent as input_video_uri" in w for w in quote["warnings"])


@pytest.mark.asyncio
@pytest.mark.parametrize("saved", [True, False], ids=["file-gone", "no-record"])
async def test_an_upscale_interaction_with_no_clip_is_refused_before_sending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, saved: bool
) -> None:
    clip = _upscale_record(tmp_path, saved=False)
    if not saved:
        record = tmp_path / "videos" / "up.json"
        manifest = json.loads(record.read_text())
        del manifest["video_url"]
        record.write_text(json.dumps(manifest))
    sent = _capture_renders(monkeypatch)
    for call in (
        server.upscale_video_omni(
            context(tmp_path, vertex=True), "4K", previous_interaction_id="up-1"
        ),
        server.edit_video(
            context(tmp_path, vertex=True),
            previous_interaction_id="up-1",
            prompt="night",
        ),
        server.extend_video_omni(
            context(tmp_path, vertex=True),
            prompt="Extend this video.",
            previous_interaction_id="up-1",
            duration_seconds=3,
        ),
    ):
        body = json.loads(await call)
        assert "up-1 is an upscale" in body["error"], body
        assert "Nothing was rendered or billed" in body["error"]
        if saved:
            assert clip in body["error"]
    assert sent == []


@pytest.mark.asyncio
async def test_an_upscale_interaction_is_edited_as_its_saved_clip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = _upscale_record(tmp_path, resolution="1080p")
    monkeypatch.setattr(server, "_probe_media", _probe(3.03, (1920, 1080)))
    sent = _capture_renders(monkeypatch)
    body = json.loads(
        await server.edit_video(
            context(tmp_path, vertex=True),
            previous_interaction_id="up-1",
            prompt="night",
        )
    )
    [request] = sent
    assert request["previous_interaction_id"] is None
    assert request["input_video_bytes"] == _MP4
    # Kept at the upscale's resolution, not an upload's 720p default.
    assert request["resolution"] == "1080p"
    assert clip in body["warnings"][0]


@pytest.mark.asyncio
async def test_an_upscale_interactions_edit_quote_keeps_its_resolution(
    tmp_path: Path,
) -> None:
    _upscale_record(tmp_path, resolution="1080p")
    quote = json.loads(
        await server.edit_video(
            context(tmp_path, vertex=True),
            previous_interaction_id="up-1",
            prompt="night",
            dry_run=True,
        )
    )
    assert quote["resolution"] == "1080p", quote
    assert quote["resolution_source"].startswith("the source's recorded resolution")
    assert "would be sent as input_video_uri" in quote["warnings"][0]


@pytest.mark.asyncio
async def test_an_upscale_interaction_is_extended_from_its_saved_clip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = _upscale_record(tmp_path, resolution="1080p")
    monkeypatch.setattr(server, "_probe_media", _probe(3.03, (1920, 1080)))
    sent = _capture_renders(monkeypatch)
    body = json.loads(
        await server.extend_video_omni(
            context(tmp_path, vertex=True),
            prompt="Extend this video.",
            previous_interaction_id="up-1",
            duration_seconds=3,
        )
    )
    first = sent[0]
    assert first["previous_interaction_id"] is None
    assert first["input_video_bytes"] == _MP4 or first["input_video_uri"] == clip
    assert first["resolution"] == "1080p"
    assert any("up-1 is an upscale" in w for w in body.get("warnings", [])), body


@pytest.mark.asyncio
async def test_a_render_interaction_is_still_continued_by_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _upscale_record(tmp_path)
    record = tmp_path / "videos" / "up.json"
    manifest = json.loads(record.read_text())
    manifest.update(kind="omni_video", task="text_to_video")
    record.write_text(json.dumps(manifest))
    sent = _capture_renders(monkeypatch)
    await server.edit_video(
        context(tmp_path, vertex=True), previous_interaction_id="up-1", prompt="night"
    )
    [request] = sent
    assert request["previous_interaction_id"] == "up-1"


def test_a_directed_total_says_why_its_note_and_date_are_not_its_renders() -> None:
    """A 1.2 render (verified 2026-11-06) plus director calls (2026-10-26)."""
    from src.pricing import (
        actual_understanding_cost,
        cost_to_dict,
        estimate_video_cost,
        sum_costs,
    )

    render = estimate_video_cost(omni.OMNI_1_2_MODEL, 3.03, resolution="360p")
    assert render is not None
    review = actual_understanding_cost(
        "gemini-4-argon", {"prompt_token_count": 1000, "candidates_token_count": 200}
    )
    total = cost_to_dict(sum_costs([render, review], label="directed render"))
    assert total is not None
    assert total["pricing_as_of"] == "2026-10-26"
    note = str(total["pricing_source_note"])
    assert "different notes" in note
    assert "oldest of its components' verification dates" in note
    assert str(render.source_note) not in note


def test_a_total_whose_components_agree_keeps_their_note() -> None:
    from src.pricing import estimate_video_cost, sum_costs

    beat = estimate_video_cost(omni.OMNI_1_2_MODEL, 4, resolution="360p")
    assert beat is not None
    total = sum_costs([beat, beat])
    assert total.source_note == beat.source_note
