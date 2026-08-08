"""What the omni Interactions calls actually put on the wire.

The last untested wire shape. Omni is a third API style — neither
``generateContent`` nor Veo's long-running operations — so nothing the other
two contract files pin applies here: a background interaction is created, then
polled by id until it leaves the in-flight statuses.

The specific hazard is that duration and aspect ratio do NOT travel in
``generation_config`` where a reader would expect them; they are serialized
into ``response_format`` as ``"10s"`` / ``"9:16"``. A refactor that moved them
to the more obvious place would still return a plausible-looking result while
the API silently rendered its own defaults.
"""

import asyncio
import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest
from google import genai
from google.genai.types import HttpOptions

from src.omni import OMNI_MODEL, generate_video_omni

_FAKE_MP4 = b"\x00\x00\x00\x18ftypmp42" + b"OMNI" * 16


class _OmniStub:
    """An Interactions endpoint that stays in flight for a fixed number of polls."""

    def __init__(
        self, in_flight_polls: int = 0, final_status: str = "completed"
    ) -> None:
        self.in_flight_polls = in_flight_polls
        self.final_status = final_status
        self.requests: list[dict[str, Any]] = []
        self.polls = 0

    def _body(self, status: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": "interactions/i-1",
            "id": "i-1",
            "status": status,
        }
        if status == "completed":
            payload["steps"] = [
                {
                    "content": [
                        {
                            "type": "video",
                            "mimeType": "video/mp4",
                            "data": base64.b64encode(_FAKE_MP4).decode(),
                        }
                    ]
                }
            ]
        return payload

    def create_response(self) -> dict[str, Any]:
        # background=True means create returns before the render finishes.
        return self._body("in_progress" if self.in_flight_polls else self.final_status)

    def poll_response(self) -> dict[str, Any]:
        self.polls += 1
        if self.polls <= self.in_flight_polls:
            return self._body("in_progress")
        return self._body(self.final_status)

    def created(self) -> dict[str, Any]:
        return next(r for r in self.requests if r["method"] == "POST")["body"]

    def response_format(self) -> dict[str, Any]:
        formats = self.created().get("response_format") or []
        return formats[0] if formats else {}


def _make_handler(stub: _OmniStub) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            """Silence the default stderr access log."""

        def _reply(self, obj: dict[str, Any]) -> None:
            raw = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            _ = self.wfile.write(raw)

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            stub.requests.append({"method": "POST", "path": self.path, "body": body})
            self._reply(stub.create_response())

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            stub.requests.append({"method": "GET", "path": self.path, "body": {}})
            self._reply(stub.poll_response())

    return Handler


def _serve(stub: _OmniStub) -> Any:
    server = HTTPServer(("127.0.0.1", 0), _make_handler(stub))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


@pytest.fixture
def no_poll_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    """Collapse the 5s inter-poll sleep so the wait loop is testable."""
    real_sleep = asyncio.sleep

    async def instant(_seconds: float) -> None:
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", instant)


def _run(stub: _OmniStub, videos_dir: Path, **kwargs: Any) -> dict[str, Any]:
    server, base_url = _serve(stub)
    client = genai.Client(
        api_key="stub-key-not-real", http_options=HttpOptions(base_url=base_url)
    )
    try:
        return asyncio.run(
            generate_video_omni(
                client=client, prompt="a cat walking", videos_dir=videos_dir, **kwargs
            )
        )
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.timeout(20.0)
def test_the_interaction_is_created_as_a_background_render(
    tmp_path: Path, no_poll_delay: None
) -> None:
    """The whole polling design depends on background=True; without it the
    create call would block and the poll loop would be dead code."""
    stub = _OmniStub()
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    result = _run(stub, videos_dir)

    created = stub.created()
    assert created["model"] == OMNI_MODEL
    assert created["background"] is True
    assert created["generation_config"]["video_config"]["task"] == "text_to_video"
    assert result["model"] == OMNI_MODEL
    assert Path(result["video_url"][7:]).read_bytes() == _FAKE_MP4
    # The id is what makes multi-turn editing possible.
    assert result["interaction_id"] == "i-1"


@pytest.mark.parametrize(
    ("requested", "expected_wire", "expected_reported"),
    [
        pytest.param(6, "6s", 6, id="in_range"),
        pytest.param(3, "3s", 3, id="at_minimum"),
        pytest.param(10, "10s", 10, id="at_maximum"),
        pytest.param(1, "3s", 3, id="below_minimum_clamps_up"),
        pytest.param(99, "10s", 10, id="above_maximum_clamps_down"),
        pytest.param(6.4, "6s", 6, id="rounded_to_whole_seconds"),
    ],
)
@pytest.mark.timeout(20.0)
def test_duration_is_clamped_and_sent_in_response_format(
    requested: float,
    expected_wire: str,
    expected_reported: int,
    tmp_path: Path,
    no_poll_delay: None,
) -> None:
    """Duration rides in response_format as "Ns", not in generation_config.

    If it stopped reaching the API the result would still report a clamped
    duration while the service rendered its own default, so pin both the
    serialized value and what the caller is told.
    """
    stub = _OmniStub()
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    result = _run(stub, videos_dir, duration_seconds=requested)

    assert stub.response_format()["duration"] == expected_wire
    assert result["duration_seconds"] == expected_reported
    # generation_config is the tempting-but-wrong home for this.
    assert "duration" not in stub.created().get("generation_config", {})


@pytest.mark.parametrize("aspect", ["16:9", "9:16"])
@pytest.mark.timeout(20.0)
def test_aspect_ratio_is_sent_in_response_format(
    aspect: str, tmp_path: Path, no_poll_delay: None
) -> None:
    stub = _OmniStub()
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    result = _run(stub, videos_dir, aspect_ratio=aspect)

    assert stub.response_format()["aspect_ratio"] == aspect
    assert stub.response_format()["type"] == "video"
    assert result["aspect_ratio"] == aspect


@pytest.mark.timeout(20.0)
def test_a_clamped_duration_is_reported_as_a_warning(
    tmp_path: Path, no_poll_delay: None
) -> None:
    """Silently rendering 10s for a 99s request would be a surprise."""
    stub = _OmniStub()
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    result = _run(stub, videos_dir, duration_seconds=99)

    assert any("clamped to 10s" in w for w in result["warnings"])


@pytest.mark.timeout(20.0)
def test_an_in_flight_interaction_is_polled_until_it_completes(
    tmp_path: Path, no_poll_delay: None
) -> None:
    """Treating the create response as final would return a video that does
    not exist yet."""
    stub = _OmniStub(in_flight_polls=3)
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    result = _run(stub, videos_dir)

    assert stub.polls == 4, "expected 3 in-flight polls then a completed one"
    assert Path(result["video_url"][7:]).exists()


@pytest.mark.parametrize("status", ["failed", "cancelled"])
@pytest.mark.timeout(20.0)
def test_a_terminal_failure_status_raises(
    status: str, tmp_path: Path, no_poll_delay: None
) -> None:
    """A failed interaction leaves the in-flight set like a successful one, so
    it must be distinguished rather than read as done."""
    stub = _OmniStub(in_flight_polls=1, final_status=status)
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    with pytest.raises(ValueError, match=status):
        _run(stub, videos_dir)


@pytest.mark.timeout(20.0)
def test_an_edit_turn_omits_the_fields_the_api_rejects(
    tmp_path: Path, no_poll_delay: None
) -> None:
    """A conversational edit has a different wire shape from a fresh render.

    The API 400s if an edit turn carries duration ("Duration cannot be set in
    response format for edit task") or a video_config task alongside
    previous_interaction_id. Duration and aspect are inherited from the source
    video instead. A refactor that made every request look the same would
    break every edit, so pin the omissions.
    """
    stub = _OmniStub()
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    result = _run(
        stub,
        videos_dir,
        previous_interaction_id="i-0",
        duration_seconds=8,
        aspect_ratio="9:16",
    )

    created = stub.created()
    response_format = stub.response_format()
    assert created["previous_interaction_id"] == "i-0"
    assert response_format["type"] == "video"
    assert "duration" not in response_format
    assert "aspect_ratio" not in response_format
    assert "generation_config" not in created
    assert result["interaction_id"] == "i-1"


@pytest.mark.timeout(20.0)
def test_a_create_response_without_an_id_fails_loud(
    tmp_path: Path, no_poll_delay: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without an interaction id there is nothing to poll and nothing for a
    later edit turn to reference — proceeding would hang or edit nothing."""
    stub = _OmniStub()
    original = stub.create_response

    def no_id() -> dict[str, Any]:
        body = original()
        body.pop("id", None)
        body.pop("name", None)
        return body

    monkeypatch.setattr(stub, "create_response", no_id)
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    with pytest.raises(ValueError, match="no interaction id"):
        _run(stub, videos_dir)


@pytest.mark.timeout(20.0)
def test_an_interaction_that_never_finishes_times_out(
    tmp_path: Path, no_poll_delay: None
) -> None:
    """The deadline covers the whole create-and-poll sequence; without it a
    stuck render would poll forever inside a tool call."""
    stub = _OmniStub(in_flight_polls=10_000)
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    with pytest.raises(TimeoutError, match="timed out"):
        _run(stub, videos_dir, timeout_seconds=0)


@pytest.mark.timeout(20.0)
def test_a_completed_interaction_with_no_video_output_fails_loud(
    tmp_path: Path, no_poll_delay: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """status=completed does not guarantee a video part exists; returning a
    result without one would hand back a video_url pointing at nothing."""
    stub = _OmniStub()
    original = stub.poll_response

    def no_video() -> dict[str, Any]:
        body = original()
        body.pop("steps", None)
        return body

    monkeypatch.setattr(stub, "poll_response", no_video)
    monkeypatch.setattr(
        stub, "create_response", lambda: {"id": "i-1", "status": "in_progress"}
    )
    stub.in_flight_polls = 1
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    with pytest.raises(ValueError, match="video"):
        _run(stub, videos_dir)


@pytest.mark.timeout(20.0)
def test_an_edit_does_not_report_a_duration_it_never_sent(
    tmp_path: Path, no_poll_delay: None
) -> None:
    """An edit sends no duration, so it must not report one.

    The impl echoed the caller's clamped request into `duration_seconds`, and
    the tool layer bills from that field — so a 3s source edited with the
    default 6 billed $0.6082 for a render that was 3s. Reporting the request
    describes a render that did not happen. None means "inherited, resolve it
    upstream"; the request is preserved separately.
    """
    stub = _OmniStub()
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    result = _run(stub, videos_dir, previous_interaction_id="i-0", duration_seconds=3)

    # Nothing about duration reached the API...
    assert "duration" not in stub.response_format()
    # ...so nothing about duration is claimed on the way back.
    assert result["duration_seconds"] is None
    assert result["requested_duration_seconds"] == 3
    assert any("not the value requested" in w for w in result["warnings"])


@pytest.mark.timeout(20.0)
def test_a_fresh_render_still_reports_the_duration_it_sent(
    tmp_path: Path, no_poll_delay: None
) -> None:
    """The other side of the same rule: a fresh render DOES send a duration,
    so it must keep reporting it — the fix must not blank the normal path."""
    stub = _OmniStub()
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    result = _run(stub, videos_dir, duration_seconds=3)

    assert stub.response_format()["duration"] == "3s"
    assert result["duration_seconds"] == 3
