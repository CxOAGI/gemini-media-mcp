"""What actually goes on the wire.

Every other test asserts against a fake client object, which proves what the
code *intended* to send. These drive the real google-genai SDK against a local
HTTP stub and assert on the request it actually produced — the URL path (which
carries the model ID) and the serialized config.

This is the layer where the deprecation work lives or dies: a reroute that
updates an internal variable but still calls the dead endpoint would pass every
fake-client test and 404 in production.
"""

import asyncio
import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from google import genai
from google.genai.types import HttpOptions
from PIL import Image

from src.image import generate_image


def _png(color: str = "#1f6feb", size: tuple[int, int] = (64, 36)) -> bytes:
    buffer = BytesIO()
    image = Image.new("RGB", size, color)
    image.save(buffer, "PNG")
    image.close()
    return buffer.getvalue()


class _Capture:
    """Requests the stub received, newest last."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []


def _make_handler(capture: _Capture) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            """Silence the default stderr access log."""

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(length) or b"{}"
            capture.requests.append({"path": self.path, "body": json.loads(raw_body)})
            payload = json.dumps(
                {
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": [
                                    {
                                        "inlineData": {
                                            "mimeType": "image/png",
                                            "data": base64.b64encode(_png()).decode(),
                                        }
                                    }
                                ],
                            },
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 17,
                        "candidatesTokenCount": 1120,
                        "totalTokenCount": 1137,
                    },
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            _ = self.wfile.write(payload)

    return Handler


@pytest.fixture
def stub() -> Any:
    """A local stand-in for the Gemini endpoint, plus a client pointed at it."""
    capture = _Capture()
    server = HTTPServer(("127.0.0.1", 0), _make_handler(capture))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    client = genai.Client(
        api_key="stub-key-not-real", http_options=HttpOptions(base_url=base_url)
    )
    try:
        yield client, capture
    finally:
        server.shutdown()
        server.server_close()


def _sent(capture: _Capture) -> tuple[str, dict[str, Any]]:
    """The model ID and imageConfig from the most recent request."""
    request = capture.requests[-1]
    model = request["path"].split("/models/")[1].split(":")[0]
    image_config = (
        request["body"].get("generationConfig", {}).get("imageConfig", {}) or {}
    )
    return model, image_config


async def _call(client: Any, images_dir: Path, **kwargs: Any) -> dict[str, Any]:
    return await generate_image(
        client=client, prompt="a test frame", images_dir=images_dir, **kwargs
    )


@pytest.mark.parametrize(
    ("requested", "expected_on_wire"),
    [
        pytest.param(
            "imagen-4.0-generate-001", "gemini-3.1-flash-image", id="imagen_retired"
        ),
        pytest.param(
            "imagen-3.0-fast-generate-001",
            "gemini-3.1-flash-image",
            id="imagen_fast_retired",
        ),
        pytest.param(
            "gemini-3.1-flash-image-preview",
            "gemini-3.1-flash-image",
            id="flash_preview_retired",
        ),
        pytest.param(
            "gemini-3-pro-image-preview", "gemini-3-pro-image", id="pro_preview_retired"
        ),
        pytest.param(
            "gemini-2.5-flash-image", "gemini-3.1-flash-image", id="nano_banana_sunset"
        ),
        pytest.param(
            "gemini-3.1-flash-image", "gemini-3.1-flash-image", id="live_unchanged"
        ),
    ],
)
@pytest.mark.timeout(15.0)
def test_superseded_ids_never_reach_the_network(
    requested: str, expected_on_wire: str, stub: Any, tmp_path: Path
) -> None:
    """The URL the SDK builds must name the replacement, not the dead alias.

    A reroute that only rewrote an internal variable would still 404 here.
    """
    client, capture = stub
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    asyncio.run(_call(client, images_dir, model=requested))

    model_on_wire, _ = _sent(capture)
    assert model_on_wire == expected_on_wire
    assert not model_on_wire.startswith("imagen")
    assert not model_on_wire.endswith("-preview")


@pytest.mark.parametrize("size", ["2K", "4K"])
@pytest.mark.timeout(15.0)
def test_flash_lite_never_receives_an_unsupported_size(
    size: str, stub: Any, tmp_path: Path
) -> None:
    """gemini-3.1-flash-lite-image is 1K-only, so imageSize must be absent from
    the serialized config rather than sent and rejected."""
    client, capture = stub
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    result = asyncio.run(
        _call(client, images_dir, model="gemini-3.1-flash-lite-image", image_size=size)
    )

    model_on_wire, image_config = _sent(capture)
    assert model_on_wire == "gemini-3.1-flash-lite-image"
    assert "imageSize" not in image_config
    assert any(f"image_size={size}" in w for w in result["warnings"])


@pytest.mark.parametrize(
    ("model", "size"),
    [("gemini-3.1-flash-image", "4K"), ("gemini-3-pro-image", "2K")],
)
@pytest.mark.timeout(15.0)
def test_supported_sizes_do_reach_the_api(
    model: str, size: str, stub: Any, tmp_path: Path
) -> None:
    """The guard must not over-reach: a size the model supports still ships."""
    client, capture = stub
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    asyncio.run(_call(client, images_dir, model=model, image_size=size))

    model_on_wire, image_config = _sent(capture)
    assert model_on_wire == model
    assert image_config.get("imageSize") == size


@pytest.mark.timeout(15.0)
def test_aspect_ratio_reaches_the_api_as_image_config(
    stub: Any, tmp_path: Path
) -> None:
    """aspect_ratio belongs on ImageConfig — it moved there when the Imagen
    GenerateImagesConfig path was removed, so pin the serialized name."""
    client, capture = stub
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    asyncio.run(
        _call(
            client,
            images_dir,
            model="gemini-3.1-flash-image",
            aspect_ratio="16:9",
            image_size="2K",
        )
    )

    _, image_config = _sent(capture)
    assert image_config == {"aspectRatio": "16:9", "imageSize": "2K"}


@pytest.mark.timeout(15.0)
def test_usage_metadata_is_parsed_from_a_real_response(
    stub: Any, tmp_path: Path
) -> None:
    """Cost reporting depends on usage surviving SDK deserialization."""
    client, capture = stub
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    result = asyncio.run(
        _call(client, images_dir, model="gemini-3.1-flash-image", image_size="1K")
    )

    assert result["usage"] == {
        "prompt_token_count": 17,
        "candidates_token_count": 1120,
        "total_token_count": 1137,
    }
    assert Path(result["image_url"][7:]).exists()


@pytest.mark.timeout(15.0)
def test_only_generate_content_is_ever_called(stub: Any, tmp_path: Path) -> None:
    """generate_images (the Imagen endpoint) is discontinued. Even a request
    that names an Imagen model must go out as generateContent."""
    client, capture = stub
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    asyncio.run(_call(client, images_dir, model="imagen-4.0-ultra-generate-001"))

    assert capture.requests[-1]["path"].endswith(":generateContent")
    assert ":predict" not in capture.requests[-1]["path"]
