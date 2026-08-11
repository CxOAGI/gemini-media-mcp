"""What the Veo calls actually put on the wire.

Video is half this server's tools and had no wire-level coverage. The risk is
the same one the Imagen work was about: ``src/video.py`` rewrites the model ID
on the way out, because the Gemini Developer API serves Veo under ``-preview``
IDs and 404s on the Vertex ``-001`` names. A translation that updated an
internal variable but built the old URL would pass every fake-client test and
fail every real call.

These drive the real SDK against a stateful local stub covering the full
long-running-operation flow: submit, poll while pending, then read the result.
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

from src.video import generate_video

_FAKE_MP4 = b"\x00\x00\x00\x18ftypmp42" + b"FAKE" * 64


class _VeoStub:
    """A Veo endpoint that reports pending for a fixed number of polls.

    ``pending_polls`` exercises the wait loop; ``fail_with`` exercises the
    operation-error path, which is otherwise unreachable offline.
    """

    def __init__(self, pending_polls: int = 1, fail_with: str | None = None) -> None:
        self.pending_polls = pending_polls
        self.fail_with = fail_with
        self.requests: list[dict[str, Any]] = []
        self.polls = 0

    def submit_response(self) -> dict[str, Any]:
        return {"name": "models/veo/operations/op-1", "done": False}

    def poll_response(self) -> dict[str, Any]:
        self.polls += 1
        if self.polls <= self.pending_polls:
            return {"name": "models/veo/operations/op-1", "done": False}
        if self.fail_with:
            return {
                "name": "models/veo/operations/op-1",
                "done": True,
                "error": {"code": 3, "message": self.fail_with},
            }
        return {
            "name": "models/veo/operations/op-1",
            "done": True,
            "response": {
                "generateVideoResponse": {
                    "generatedSamples": [
                        {
                            "video": {
                                # mldev spells these encodedVideo/encoding.
                                "encodedVideo": base64.b64encode(_FAKE_MP4).decode(),
                                "encoding": "video/mp4",
                            }
                        }
                    ]
                }
            },
        }

    def model_on_wire(self) -> str:
        """The model ID from the submit URL — the thing under test."""
        submit = next(r for r in self.requests if "predictLongRunning" in r["path"])
        return submit["path"].split("/models/")[1].split(":")[0]

    def parameters(self) -> dict[str, Any]:
        submit = next(r for r in self.requests if "predictLongRunning" in r["path"])
        return submit["body"].get("parameters", {}) or {}


def _make_handler(stub: _VeoStub) -> type[BaseHTTPRequestHandler]:
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
            stub.requests.append({"path": self.path, "body": body})
            # Vertex polls operations with POST; mldev submits with POST.
            if "operations" in self.path or ":fetchPredictOperation" in self.path:
                self._reply(stub.poll_response())
            else:
                self._reply(stub.submit_response())

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            stub.requests.append({"path": self.path, "body": {}})
            self._reply(stub.poll_response())

    return Handler


def _serve(stub: _VeoStub) -> Any:
    server = HTTPServer(("127.0.0.1", 0), _make_handler(stub))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


@pytest.fixture
def no_poll_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    """Collapse the 10s inter-poll sleep so the wait loop is testable."""
    real_sleep = asyncio.sleep

    async def instant(_seconds: float) -> None:
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", instant)


def _run(client: Any, videos_dir: Path, **kwargs: Any) -> dict[str, Any]:
    return asyncio.run(
        generate_video(
            client=client, prompt="a cat walking", videos_dir=videos_dir, **kwargs
        )
    )


def _client(base_url: str) -> Any:
    return genai.Client(
        api_key="stub-key-not-real", http_options=HttpOptions(base_url=base_url)
    )


@pytest.mark.parametrize(
    ("requested", "expected_on_wire"),
    [
        pytest.param("veo-3.1-generate-001", "veo-3.1-generate-preview", id="standard"),
        pytest.param(
            "veo-3.1-fast-generate-001", "veo-3.1-fast-generate-preview", id="fast"
        ),
        pytest.param(
            "veo-3.1-lite-generate-preview",
            "veo-3.1-lite-generate-preview",
            id="lite_already_preview",
        ),
    ],
)
@pytest.mark.timeout(20.0)
def test_gemini_api_receives_the_preview_spelling(
    requested: str,
    expected_on_wire: str,
    tmp_path: Path,
    no_poll_delay: None,
) -> None:
    """The Gemini Developer API 404s on the Vertex -001 names, so the submit
    URL must carry the -preview spelling."""
    stub = _VeoStub()
    server, base_url = _serve(stub)
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    try:
        result = _run(_client(base_url), videos_dir, model=requested)
    finally:
        server.shutdown()
        server.server_close()

    assert stub.model_on_wire() == expected_on_wire
    assert result["model"] == expected_on_wire
    assert Path(result["video_url"][7:]).read_bytes() == _FAKE_MP4


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        pytest.param(4.0, 4, id="exact_4"),
        # 5 is equidistant from 4 and 6, and 7 from 6 and 8. min() keeps the
        # first candidate, so both ties resolve downward — pinned because a
        # cost estimate is built from the value that actually ships.
        pytest.param(5.0, 4, id="5_ties_down_to_4"),
        pytest.param(5.4, 6, id="5.4_rounds_up"),
        pytest.param(7.0, 6, id="7_ties_down_to_6"),
        pytest.param(8.0, 8, id="exact_8"),
        pytest.param(99.0, 8, id="clamped_to_max"),
    ],
)
@pytest.mark.timeout(20.0)
def test_duration_is_snapped_before_it_reaches_the_api(
    requested: float, expected: int, tmp_path: Path, no_poll_delay: None
) -> None:
    """Veo only accepts 4/6/8s. The snapped value must be what is sent, and
    what the caller is told, so a cost estimate built from it is right."""
    stub = _VeoStub()
    server, base_url = _serve(stub)
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    try:
        result = _run(
            _client(base_url),
            videos_dir,
            model="veo-3.1-generate-001",
            duration_seconds=requested,
        )
    finally:
        server.shutdown()
        server.server_close()

    assert stub.parameters().get("durationSeconds") == expected
    assert result["duration_seconds"] == expected


@pytest.mark.timeout(20.0)
def test_the_operation_is_polled_until_it_reports_done(
    tmp_path: Path, no_poll_delay: None
) -> None:
    """The wait loop must actually re-fetch: returning the first pending
    operation as the result would hand back a video that does not exist."""
    stub = _VeoStub(pending_polls=3)
    server, base_url = _serve(stub)
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    try:
        result = _run(_client(base_url), videos_dir, model="veo-3.1-generate-001")
    finally:
        server.shutdown()
        server.server_close()

    assert stub.polls == 4, "expected 3 pending polls then a completed one"
    assert result["message"] == "Video generated successfully"


@pytest.mark.timeout(20.0)
def test_an_operation_error_is_raised_not_silently_returned(
    tmp_path: Path, no_poll_delay: None
) -> None:
    """A failed operation completes with done=true, so it must be inspected for
    an error rather than treated as success."""
    stub = _VeoStub(fail_with="content policy violation")
    server, base_url = _serve(stub)
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    try:
        with pytest.raises(ValueError, match="content policy violation"):
            _run(_client(base_url), videos_dir, model="veo-3.1-generate-001")
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.timeout(20.0)
def test_aspect_ratio_and_sample_count_reach_the_api(
    tmp_path: Path, no_poll_delay: None
) -> None:
    """Pin the serialized parameter names, which differ from our argument
    names and are easy to break silently."""
    stub = _VeoStub()
    server, base_url = _serve(stub)
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    try:
        _run(
            _client(base_url),
            videos_dir,
            model="veo-3.1-generate-001",
            aspect_ratio="9:16",
        )
    finally:
        server.shutdown()
        server.server_close()

    parameters = stub.parameters()
    assert parameters.get("aspectRatio") == "9:16"
    assert parameters.get("sampleCount") == 1


@pytest.mark.timeout(20.0)
def test_gemini_api_reports_audio_as_always_on(
    tmp_path: Path, no_poll_delay: None
) -> None:
    """Veo 3.1 on the Gemini API always generates audio, whatever the caller
    asked for. Reporting the request back would be a lie."""
    stub = _VeoStub()
    server, base_url = _serve(stub)
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    try:
        result = _run(
            _client(base_url),
            videos_dir,
            model="veo-3.1-generate-001",
            include_audio=False,
        )
    finally:
        server.shutdown()
        server.server_close()

    assert result["audio_enabled"] is True
