"""Tests for the blocking SDK/filesystem paths in video.py and omni.py.

Covers the ways a single caller-controlled input used to hang the whole
server: an unguarded local read for extend_video, SDK calls with no deadline
at all, and a download that escaped the deadline every sibling call honoured.
Also covers the paid parameters those paths used to discard without a word.
"""

import asyncio
import os
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from src import video as video_mod
from src.omni import generate_video_omni
from src.video import generate_video, run_off_loop

# ============================================================================
# Test doubles
# ============================================================================


class FakeVideoObject:
    def __init__(
        self, uri: str | None = None, video_bytes: bytes | None = None
    ) -> None:
        self.uri = uri
        self.video_bytes = video_bytes

    def save(self, path: str) -> None:
        Path(path).write_bytes(b"saved")


class FakeGeneratedVideo:
    def __init__(self, video: FakeVideoObject) -> None:
        self.video = video


class FakeVideoResult:
    def __init__(self, generated_videos: list[FakeGeneratedVideo]) -> None:
        self.generated_videos = generated_videos


class FakeOperation:
    def __init__(self, done: bool = True, name: str = "op-1") -> None:
        self.done = done
        self.name = name
        self.error = None
        result = FakeVideoResult([FakeGeneratedVideo(FakeVideoObject(b"", b"bytes"))])
        self.result = result
        self.response = result


class FakeModels:
    def __init__(self, operation: FakeOperation) -> None:
        self._operation = operation
        self.calls: list[dict[str, Any]] = []

    def generate_videos(self, **kwargs: Any) -> FakeOperation:
        self.calls.append(kwargs)
        return self._operation


class FakeOperations:
    def __init__(self, operation: FakeOperation) -> None:
        self._operation = operation

    def get(self, op: FakeOperation) -> FakeOperation:
        return self._operation


class FakeFiles:
    def download(self, file: Any) -> None:
        return None


class FakeApiClient:
    def __init__(self, vertexai: bool = False) -> None:
        self.vertexai = vertexai


class FakeVeoClient:
    def __init__(
        self, operation: FakeOperation | None = None, vertexai: bool = False
    ) -> None:
        op = operation or FakeOperation()
        self.models = FakeModels(op)
        self.operations = FakeOperations(op)
        self.files = FakeFiles()
        self._api_client = FakeApiClient(vertexai=vertexai)


class FakeOmniOutputVideo:
    def __init__(self, uri: str) -> None:
        self.uri = uri
        self.data = None


class FakeOmniInteraction:
    def __init__(
        self,
        interaction_id: str = "int-1",
        status: str = "completed",
        output_video: FakeOmniOutputVideo | None = None,
    ) -> None:
        self.id = interaction_id
        self.status = status
        self.output_video = output_video
        self.steps: list[Any] = []
        self.error = None


class FakeOmniInteractions:
    def __init__(
        self,
        create_result: FakeOmniInteraction,
        get_results: list[FakeOmniInteraction] | None = None,
    ) -> None:
        self._create_result = create_result
        self._get_results = list(get_results or [])

    def create(self, **kwargs: Any) -> FakeOmniInteraction:
        return self._create_result

    def get(self, id: str, **kwargs: Any) -> FakeOmniInteraction:
        if self._get_results:
            return self._get_results.pop(0)
        raise AssertionError("unexpected interactions.get call")


class FakeOmniFiles:
    def __init__(self, block: threading.Event | None = None) -> None:
        self._block = block

    def get(self, **kwargs: Any) -> Any:
        if self._block is not None:
            self._block.wait()
        return object()

    def download(self, **kwargs: Any) -> bytes:
        return b"mp4"


class FakeOmniClient:
    def __init__(
        self,
        interactions: FakeOmniInteractions,
        files: FakeOmniFiles | None = None,
    ) -> None:
        self.interactions = interactions
        self.files = files or FakeOmniFiles()


def _png_bytes() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (8, 8), color="blue").save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def videos_dir(tmp_path: Path) -> Path:
    path = tmp_path / "videos"
    path.mkdir()
    return path


# ============================================================================
# The extend_video local read
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(10.0)
async def test_extend_rejects_non_regular_file(
    tmp_path: Path, videos_dir: Path
) -> None:
    """A FIFO in the media directory is refused, not read.

    Reading one blocks forever with no writer, and the media directory is
    caller-controlled — that read used to be made straight on the event loop.
    """
    fifo = tmp_path / "clip.mp4"
    os.mkfifo(fifo)

    with pytest.raises(ValueError, match="not a regular file"):
        await generate_video(
            client=FakeVeoClient(vertexai=True),  # type: ignore[arg-type]
            prompt="extend it",
            videos_dir=videos_dir,
            model="veo-3.1-generate-001",
            allowed_dir=tmp_path,
            extend_video_uri=f"file://{fifo}",
        )


@pytest.mark.asyncio
@pytest.mark.timeout(10.0)
async def test_extend_rejects_oversize_file(
    tmp_path: Path, videos_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A local clip over the cap is refused before it is read into memory."""
    monkeypatch.setattr(video_mod, "_EXTEND_MAX_BYTES", 16)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x" * 128)

    with pytest.raises(ValueError, match="over the 16 byte limit"):
        await generate_video(
            client=FakeVeoClient(vertexai=True),  # type: ignore[arg-type]
            prompt="extend it",
            videos_dir=videos_dir,
            model="veo-3.1-generate-001",
            allowed_dir=tmp_path,
            extend_video_uri=f"file://{clip}",
        )


@pytest.mark.asyncio
@pytest.mark.timeout(10.0)
async def test_extend_read_does_not_block_the_event_loop(
    tmp_path: Path, videos_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A slow local read leaves the loop free to serve other requests."""
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"data")

    def slow_load(location: Path, allowed_dir: Path | None) -> Any:
        time.sleep(0.3)
        return object()

    monkeypatch.setattr(video_mod, "_load_extend_video", slow_load)

    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.02)
            ticks += 1

    tick_task = asyncio.create_task(ticker())
    try:
        await generate_video(
            client=FakeVeoClient(vertexai=True),  # type: ignore[arg-type]
            prompt="extend it",
            videos_dir=videos_dir,
            model="veo-3.1-generate-001",
            allowed_dir=tmp_path,
            extend_video_uri=f"file://{clip}",
        )
    finally:
        tick_task.cancel()

    assert ticks >= 5


@pytest.mark.asyncio
@pytest.mark.timeout(10.0)
async def test_extend_read_times_out_instead_of_hanging(
    tmp_path: Path, videos_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wedged mount surfaces as a TimeoutError, not a hung request."""
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"data")

    release = threading.Event()

    def wedged_load(location: Path, allowed_dir: Path | None) -> Any:
        release.wait(30)
        return object()

    monkeypatch.setattr(video_mod, "_load_extend_video", wedged_load)
    monkeypatch.setattr(video_mod, "_EXTEND_READ_TIMEOUT_SECONDS", 0.2)

    try:
        with pytest.raises(TimeoutError, match="Timed out reading the video"):
            await generate_video(
                client=FakeVeoClient(vertexai=True),  # type: ignore[arg-type]
                prompt="extend it",
                videos_dir=videos_dir,
                model="veo-3.1-generate-001",
                allowed_dir=tmp_path,
                extend_video_uri=f"file://{clip}",
            )
    finally:
        release.set()


# ============================================================================
# Veo call deadlines
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(10.0)
async def test_run_off_loop_uses_the_dedicated_pool() -> None:
    """Blocking SDK work never lands on the loop's shared default executor.

    A timed-out call cannot be cancelled, so the worker is abandoned; on the
    shared pool enough of those starve every other request in the server.
    """
    name = await run_off_loop(
        lambda: threading.current_thread().name, timeout=5.0, message="unused"
    )
    assert name.startswith("media-sdk")


@pytest.mark.asyncio
@pytest.mark.timeout(10.0)
async def test_veo_submit_sends_a_transport_timeout(videos_dir: Path) -> None:
    """The submit call carries HttpOptions.timeout.

    Unset, google-genai passes timeout=None to httpx, which disables transport
    timeouts entirely and lets a stalled connection outlive the request.
    """
    client = FakeVeoClient()
    await generate_video(
        client=client,  # type: ignore[arg-type]
        prompt="a clip",
        videos_dir=videos_dir,
        model="veo-3.1-generate-001",
    )
    config = client.models.calls[0]["config"]
    assert config.http_options is not None
    assert config.http_options.timeout == video_mod._VEO_SUBMIT_HTTP_TIMEOUT_MS


@pytest.mark.asyncio
@pytest.mark.timeout(10.0)
async def test_veo_submit_stall_raises_instead_of_hanging(
    videos_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A submit call that never returns fails the request on a deadline."""
    release = threading.Event()
    client = FakeVeoClient()

    def wedged(**kwargs: Any) -> FakeOperation:
        release.wait(30)
        return FakeOperation()

    monkeypatch.setattr(client.models, "generate_videos", wedged)
    monkeypatch.setattr(video_mod, "_VEO_SUBMIT_TIMEOUT_SECONDS", 0.2)

    try:
        with pytest.raises(TimeoutError, match="timed out submitting"):
            await generate_video(
                client=client,  # type: ignore[arg-type]
                prompt="a clip",
                videos_dir=videos_dir,
                model="veo-3.1-generate-001",
            )
    finally:
        release.set()


@pytest.mark.asyncio
@pytest.mark.timeout(10.0)
async def test_veo_poll_stall_raises_instead_of_hanging(
    videos_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A poll that never returns fails the request on a deadline."""
    release = threading.Event()
    client = FakeVeoClient(operation=FakeOperation(done=False))

    def wedged(op: FakeOperation) -> FakeOperation:
        release.wait(30)
        return FakeOperation()

    monkeypatch.setattr(client.operations, "get", wedged)
    monkeypatch.setattr(video_mod, "_VEO_POLL_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(video_mod, "_VEO_POLL_TIMEOUT_SECONDS", 0.2)

    try:
        with pytest.raises(TimeoutError, match="timed out polling"):
            await generate_video(
                client=client,  # type: ignore[arg-type]
                prompt="a clip",
                videos_dir=videos_dir,
                model="veo-3.1-generate-001",
            )
    finally:
        release.set()


@pytest.mark.asyncio
@pytest.mark.timeout(20.0)
async def test_veo_total_budget_counts_time_spent_blocked(
    videos_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Time inside a slow SDK call counts against the render budget.

    The old counter only accumulated the sleep interval, so a render that
    spent all its time blocked in the SDK could never reach the limit.
    """
    client = FakeVeoClient(operation=FakeOperation(done=False))

    def slow_poll(op: FakeOperation) -> FakeOperation:
        time.sleep(0.3)
        return FakeOperation(done=False)

    # A poll interval far shorter than the poll itself: a budget that only
    # accrues the sleep interval would need ~100 rounds to notice, by which
    # point 30s of real time has gone by.
    monkeypatch.setattr(client.operations, "get", slow_poll)
    monkeypatch.setattr(video_mod, "_VEO_POLL_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(video_mod, "_VEO_TOTAL_TIMEOUT_SECONDS", 1.0)

    started = time.monotonic()
    with pytest.raises(TimeoutError, match="timed out"):
        await generate_video(
            client=client,  # type: ignore[arg-type]
            prompt="a clip",
            videos_dir=videos_dir,
            model="veo-3.1-generate-001",
        )
    assert time.monotonic() - started < 10.0


# ============================================================================
# Paid parameters that used to be discarded silently
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(10.0)
async def test_extend_with_image_inputs_raises(videos_dir: Path) -> None:
    """Extension plus image inputs names the conflict instead of dropping it."""
    with pytest.raises(ValueError, match="extend_video_uri cannot be combined"):
        await generate_video(
            client=FakeVeoClient(),  # type: ignore[arg-type]
            prompt="a clip",
            videos_dir=videos_dir,
            model="veo-3.1-generate-001",
            image_bytes=_png_bytes(),
            extend_video_uri="gs://bucket/clip.mp4",
        )


@pytest.mark.asyncio
@pytest.mark.timeout(10.0)
async def test_references_with_frames_raise_and_name_both(
    videos_dir: Path,
) -> None:
    """References plus frames raise: neither frame is sent and the bill doubles."""
    with pytest.raises(ValueError) as excinfo:
        await generate_video(
            client=FakeVeoClient(),  # type: ignore[arg-type]
            prompt="a clip",
            videos_dir=videos_dir,
            model="veo-3.1-generate-001",
            image_bytes=_png_bytes(),
            last_frame_bytes=_png_bytes(),
            reference_images=[_png_bytes()],
        )
    message = str(excinfo.value)
    assert "reference images cannot be combined" in message
    assert "image_uri/image_base64" in message
    assert "last_frame_uri/last_frame_base64" in message


@pytest.mark.asyncio
@pytest.mark.timeout(10.0)
async def test_extra_reference_images_warn(videos_dir: Path) -> None:
    """Truncating to three references is reported, not silent."""
    result = await generate_video(
        client=FakeVeoClient(),  # type: ignore[arg-type]
        prompt="a clip",
        videos_dir=videos_dir,
        model="veo-3.1-generate-001",
        reference_images=[_png_bytes() for _ in range(5)],
    )
    warnings = result.get("warnings", [])
    assert any(
        "5 reference images were supplied" in w and "were not sent" in w
        for w in warnings
    ), warnings


@pytest.mark.asyncio
@pytest.mark.timeout(10.0)
async def test_negative_seed_warns(videos_dir: Path) -> None:
    """A dropped seed is reported — reproducibility is the whole point of it."""
    client = FakeVeoClient()
    result = await generate_video(
        client=client,  # type: ignore[arg-type]
        prompt="a clip",
        videos_dir=videos_dir,
        model="veo-3.1-generate-001",
        seed=-7,
    )
    assert client.models.calls[0]["config"].seed is None
    warnings = result.get("warnings", [])
    assert any("seed=-7 was not sent" in w for w in warnings), warnings


# ============================================================================
# Omni
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(15.0)
async def test_omni_download_honours_the_deadline(videos_dir: Path) -> None:
    """The delivered-file download cannot outlive the interaction's budget.

    It used to run outside the deadline every other call in the module honours,
    so an interaction completing at t=590s of a 600s budget hung forever here.
    """
    release = threading.Event()
    interaction = FakeOmniInteraction(
        interaction_id="int-dl",
        status="completed",
        output_video=FakeOmniOutputVideo(uri="files/abc"),
    )
    client = FakeOmniClient(
        FakeOmniInteractions(create_result=interaction),
        files=FakeOmniFiles(block=release),
    )

    try:
        with pytest.raises(TimeoutError, match="Omni video interaction timed out"):
            await generate_video_omni(
                client=client,  # type: ignore[arg-type]
                prompt="p",
                videos_dir=videos_dir,
                timeout_seconds=1,
            )
    finally:
        release.set()


@pytest.mark.asyncio
@pytest.mark.timeout(15.0)
async def test_omni_logs_state_changes_not_every_poll(
    videos_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unchanged status is logged once, not once per poll."""
    pending = FakeOmniInteraction(interaction_id="int-log", status="in_progress")
    done = FakeOmniInteraction(
        interaction_id="int-log",
        status="completed",
        output_video=FakeOmniOutputVideo(uri="gs://bucket/out.mp4"),
    )
    in_flight = [
        FakeOmniInteraction(interaction_id="int-log", status="in_progress")
        for _ in range(11)
    ]
    client = FakeOmniClient(
        FakeOmniInteractions(create_result=pending, get_results=[*in_flight, done])
    )

    real_sleep = asyncio.sleep

    async def instant_sleep(_seconds: float) -> None:
        await real_sleep(0)

    monkeypatch.setattr("src.omni.asyncio.sleep", instant_sleep)

    messages: list[str] = []

    async def log_callback(message: str) -> None:
        messages.append(message)

    await generate_video_omni(
        client=client,  # type: ignore[arg-type]
        prompt="p",
        videos_dir=videos_dir,
        log_callback=log_callback,
    )

    status_lines = [m for m in messages if m.startswith("Interaction int-log:")]
    assert status_lines == ["Interaction int-log: in_progress"], messages


@pytest.mark.asyncio
@pytest.mark.timeout(15.0)
async def test_omni_logs_each_state_transition(
    videos_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A genuine state change is still reported."""
    pending = FakeOmniInteraction(interaction_id="int-t", status="queued")
    done = FakeOmniInteraction(
        interaction_id="int-t",
        status="completed",
        output_video=FakeOmniOutputVideo(uri="gs://bucket/out.mp4"),
    )
    client = FakeOmniClient(
        FakeOmniInteractions(
            create_result=pending,
            get_results=[
                FakeOmniInteraction(interaction_id="int-t", status="in_progress"),
                FakeOmniInteraction(interaction_id="int-t", status="in_progress"),
                done,
            ],
        )
    )

    real_sleep = asyncio.sleep

    async def instant_sleep(_seconds: float) -> None:
        await real_sleep(0)

    monkeypatch.setattr("src.omni.asyncio.sleep", instant_sleep)

    messages: list[str] = []

    async def log_callback(message: str) -> None:
        messages.append(message)

    await generate_video_omni(
        client=client,  # type: ignore[arg-type]
        prompt="p",
        videos_dir=videos_dir,
        log_callback=log_callback,
    )

    status_lines = [m for m in messages if m.startswith("Interaction int-t:")]
    assert status_lines == [
        "Interaction int-t: queued",
        "Interaction int-t: in_progress",
    ], messages


@pytest.mark.asyncio
@pytest.mark.timeout(15.0)
async def test_omni_edit_reports_no_aspect_ratio(videos_dir: Path) -> None:
    """An edit never sends aspect_ratio, so it must not report one as fact."""
    done = FakeOmniInteraction(
        interaction_id="int-edit",
        status="completed",
        output_video=FakeOmniOutputVideo(uri="gs://bucket/out.mp4"),
    )
    client = FakeOmniClient(FakeOmniInteractions(create_result=done))

    result = await generate_video_omni(
        client=client,  # type: ignore[arg-type]
        prompt="make it warmer",
        videos_dir=videos_dir,
        previous_interaction_id="int-prev",
        aspect_ratio="16:9",
    )

    assert result["aspect_ratio"] is None
    assert result["requested_aspect_ratio"] == "16:9"
    # The companion field is already handled this way; both must agree.
    assert result["duration_seconds"] is None


@pytest.mark.asyncio
@pytest.mark.timeout(15.0)
async def test_omni_generation_still_reports_its_aspect_ratio(
    videos_dir: Path,
) -> None:
    """A fresh generation does send aspect_ratio, so it keeps reporting it."""
    done = FakeOmniInteraction(
        interaction_id="int-gen",
        status="completed",
        output_video=FakeOmniOutputVideo(uri="gs://bucket/out.mp4"),
    )
    client = FakeOmniClient(FakeOmniInteractions(create_result=done))

    result = await generate_video_omni(
        client=client,  # type: ignore[arg-type]
        prompt="a clip",
        videos_dir=videos_dir,
        aspect_ratio="9:16",
    )

    assert result["aspect_ratio"] == "9:16"
    assert result["requested_aspect_ratio"] == "9:16"
