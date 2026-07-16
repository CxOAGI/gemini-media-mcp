"""Tests for setup_wizard.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from src.setup_wizard import (
    build_claude_config,
    macos_config_path,
    run_wizard,
)

# ============================================================================
# Test Doubles
# ============================================================================


class FakeInput:
    """Test double for input(): yields scripted answers in order."""

    def __init__(self, answers: list[str]) -> None:
        self._answers = list(answers)
        self.prompts: list[str] = []

    def __call__(self, prompt: str = "") -> str:
        self.prompts.append(prompt)
        if not self._answers:
            raise AssertionError(
                f"Ran out of scripted answers; pending prompt: {prompt!r}"
            )
        return self._answers.pop(0)


class FakePrint:
    """Test double for print(): captures all calls."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        self.lines.append(" ".join(str(a) for a in args))


class FakeModels:
    """Test double for client.models, exposing a live-call list()."""

    def __init__(self) -> None:
        self.list_calls = 0

    def list(self) -> list[str]:
        """Mimic genai's models.list(); return an iterable of fake models."""
        self.list_calls += 1
        return ["fake-model-a", "fake-model-b"]


class FakeGenaiClient:
    """Test double for google.genai.Client."""

    instances: list[FakeGenaiClient] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.models = FakeModels()
        FakeGenaiClient.instances.append(self)


class FakeGenaiModule:
    """Test double for the google.genai module."""

    Client = FakeGenaiClient


@pytest.fixture(autouse=True)
def _reset_fake_genai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a stub google.genai module for every test."""
    FakeGenaiClient.instances = []
    monkeypatch.setitem(sys.modules, "google.genai", FakeGenaiModule)
    # google.genai is typically accessed via ``from google import genai``;
    # the real google package supports this, so our stub module only needs
    # to intercept the attribute lookup.
    import google  # type: ignore[import-not-found]

    monkeypatch.setattr(google, "genai", FakeGenaiModule, raising=False)


# ============================================================================
# build_claude_config tests
# ============================================================================


@pytest.mark.parametrize(
    ("env", "expected_keys"),
    [
        pytest.param(
            {"GEMINI_API_KEY": "k"},
            ["GEMINI_API_KEY"],
            id="gemini_only",
        ),
        pytest.param(
            {
                "GOOGLE_GENAI_USE_VERTEXAI": "true",
                "GOOGLE_CLOUD_PROJECT": "p",
                "GOOGLE_CLOUD_LOCATION": "us-central1",
                "GOOGLE_APPLICATION_CREDENTIALS": "/tmp/sa.json",
            },
            [
                "GOOGLE_GENAI_USE_VERTEXAI",
                "GOOGLE_CLOUD_PROJECT",
                "GOOGLE_CLOUD_LOCATION",
                "GOOGLE_APPLICATION_CREDENTIALS",
            ],
            id="vertex_file",
        ),
    ],
)
@pytest.mark.timeout(1.0)
def test_build_claude_config(env: dict[str, str], expected_keys: list[str]) -> None:
    """Config block wraps env under mcpServers.gemini-media."""
    cfg = build_claude_config(env)
    assert cfg["mcpServers"]["gemini-media"]["command"] == "uvx"
    assert cfg["mcpServers"]["gemini-media"]["args"] == ["gemini-media-mcp"]
    server_env = cfg["mcpServers"]["gemini-media"]["env"]
    for key in expected_keys:
        assert key in server_env


# ============================================================================
# Gemini-API path
# ============================================================================


@pytest.mark.timeout(1.0)
def test_gemini_interactive_path() -> None:
    """Gemini mode via interactive prompts returns GEMINI_API_KEY in env."""
    fake_input = FakeInput(
        answers=[
            "g",  # mode
            "my-key",  # api key
            "",  # data folder (accept default)
            "",  # video gcs bucket (skip)
        ]
    )
    fake_print = FakePrint()

    config = run_wizard(
        interactive=True,
        input_fn=fake_input,
        print_fn=fake_print,
    )

    env = config["mcpServers"]["gemini-media"]["env"]
    assert env["GEMINI_API_KEY"] == "my-key"
    assert "GOOGLE_GENAI_USE_VERTEXAI" not in env
    # Default data folder should be expanded from ~
    assert env["DATA_FOLDER"].endswith("gemini-media")
    assert not env["DATA_FOLDER"].startswith("~")
    assert len(FakeGenaiClient.instances) == 1
    # http_options carries the validation timeout; its value depends on
    # whether the (stubbed) SDK exposes HttpOptions, so only pin api_key.
    assert FakeGenaiClient.instances[0].kwargs["api_key"] == "my-key"


# ============================================================================
# Vertex + service-account file path
# ============================================================================


@pytest.mark.timeout(1.0)
def test_vertex_with_sa_file(tmp_path: Path) -> None:
    """Vertex flow with an on-disk SA file stores the path in env."""
    sa_file = tmp_path / "sa.json"
    sa_file.write_text(json.dumps({"type": "service_account", "project_id": "p"}))

    fake_input = FakeInput(
        answers=[
            "v",  # mode
            "my-project",  # project id
            "",  # location (default)
            str(sa_file),  # sa path
            "",  # data folder (default)
            "",  # video gcs bucket (skip)
        ]
    )
    fake_print = FakePrint()

    config = run_wizard(
        interactive=True,
        input_fn=fake_input,
        print_fn=fake_print,
    )

    env = config["mcpServers"]["gemini-media"]["env"]
    assert env["GOOGLE_GENAI_USE_VERTEXAI"] == "true"
    assert env["GOOGLE_CLOUD_PROJECT"] == "my-project"
    assert env["GOOGLE_CLOUD_LOCATION"] == "us-central1"
    assert env["GOOGLE_APPLICATION_CREDENTIALS"] == str(sa_file)
    assert "GOOGLE_SERVICE_ACCOUNT_JSON" not in env


# ============================================================================
# Vertex + inline paste
# ============================================================================


@pytest.mark.timeout(1.0)
def test_vertex_with_inline_paste() -> None:
    """Vertex flow with inline JSON paste stores JSON string in env."""
    sa = {"type": "service_account", "project_id": "inline"}
    sa_json = json.dumps(sa)

    fake_input = FakeInput(
        answers=[
            "v",  # mode
            "inline",  # project id
            "us-west1",  # location
            "",  # no path given: triggers paste mode
            sa_json,  # first line of paste (ends with })
            "",  # blank line terminates paste
            "",  # data folder default
            "",  # video gcs bucket skip
        ]
    )
    fake_print = FakePrint()

    config = run_wizard(
        interactive=True,
        input_fn=fake_input,
        print_fn=fake_print,
    )

    env = config["mcpServers"]["gemini-media"]["env"]
    assert env["GOOGLE_CLOUD_LOCATION"] == "us-west1"
    assert "GOOGLE_APPLICATION_CREDENTIALS" not in env
    # Inline JSON is stored serialized in GOOGLE_SERVICE_ACCOUNT_JSON
    parsed = json.loads(env["GOOGLE_SERVICE_ACCOUNT_JSON"])
    assert parsed == sa


# ============================================================================
# Non-interactive mode
# ============================================================================


@pytest.mark.parametrize(
    ("overrides", "expected_env_subset"),
    [
        pytest.param(
            {"mode": "gemini", "api_key": "xyz", "data_folder": "/out"},
            {"GEMINI_API_KEY": "xyz", "DATA_FOLDER": "/out"},
            id="gemini_non_interactive",
        ),
        pytest.param(
            {
                "mode": "vertex",
                "project_id": "proj",
                "location": "us-central1",
                "sa_json": json.dumps({"type": "service_account"}),
                "data_folder": "/tmp/out",
                "video_gcs_bucket": "gs://my-bucket/path",
                "populate_allowed_buckets": True,
            },
            {
                "GOOGLE_GENAI_USE_VERTEXAI": "true",
                "GOOGLE_CLOUD_PROJECT": "proj",
                "GOOGLE_CLOUD_LOCATION": "us-central1",
                "VIDEO_GCS_BUCKET": "gs://my-bucket/path",
                "GCS_ALLOWED_BUCKETS": "my-bucket",
            },
            id="vertex_non_interactive_with_bucket",
        ),
    ],
)
@pytest.mark.timeout(1.0)
def test_non_interactive_builds_env_without_input(
    overrides: dict[str, Any],
    expected_env_subset: dict[str, str],
) -> None:
    """Non-interactive mode should never call input()."""

    def forbidden_input(_: str = "") -> str:
        raise AssertionError("input() must not be called in non-interactive mode")

    fake_print = FakePrint()

    config = run_wizard(
        interactive=False,
        input_fn=forbidden_input,
        print_fn=fake_print,
        **overrides,
    )

    env = config["mcpServers"]["gemini-media"]["env"]
    for key, value in expected_env_subset.items():
        assert env[key] == value, f"Expected {key}={value!r}, got {env.get(key)!r}"


@pytest.mark.timeout(1.0)
def test_non_interactive_requires_mode() -> None:
    """Non-interactive mode without 'mode' raises ValueError."""

    def forbidden_input(_: str = "") -> str:
        raise AssertionError("input() must not be called")

    with pytest.raises(ValueError, match="mode"):
        run_wizard(interactive=False, input_fn=forbidden_input, print_fn=FakePrint())


# ============================================================================
# macOS config merge
# ============================================================================


@pytest.mark.timeout(1.0)
def test_macos_config_merge_preserves_other_servers(tmp_path: Path) -> None:
    """Writing the config merges with existing mcpServers and backs up the old file."""
    config_path = tmp_path / "claude_desktop_config.json"
    existing = {
        "mcpServers": {
            "other": {"command": "node", "args": ["server.js"]},
        },
        "theme": "dark",
    }
    config_path.write_text(json.dumps(existing))

    fake_print = FakePrint()
    run_wizard(
        interactive=False,
        mode="gemini",
        api_key="abc",
        data_folder="/d",
        input_fn=lambda _="": "",
        print_fn=fake_print,
        write_config=True,
        config_path=config_path,
    )

    result = json.loads(config_path.read_text())
    assert "other" in result["mcpServers"], "existing server must be preserved"
    assert result["mcpServers"]["other"] == existing["mcpServers"]["other"]
    assert "gemini-media" in result["mcpServers"]
    assert result["mcpServers"]["gemini-media"]["env"]["GEMINI_API_KEY"] == "abc"
    # Unrelated top-level keys preserved
    assert result["theme"] == "dark"

    backup = config_path.with_suffix(config_path.suffix + ".bak")
    assert backup.exists(), "backup file must be created"
    assert json.loads(backup.read_text()) == existing


@pytest.mark.timeout(1.0)
def test_macos_config_write_creates_file(tmp_path: Path) -> None:
    """Writing to a nonexistent path creates the file without a backup."""
    config_path = tmp_path / "nested" / "claude_desktop_config.json"
    fake_print = FakePrint()

    run_wizard(
        interactive=False,
        mode="gemini",
        api_key="abc",
        data_folder="/d",
        input_fn=lambda _="": "",
        print_fn=fake_print,
        write_config=True,
        config_path=config_path,
    )

    assert config_path.exists()
    result = json.loads(config_path.read_text())
    assert "gemini-media" in result["mcpServers"]
    assert not config_path.with_suffix(config_path.suffix + ".bak").exists()


# ============================================================================
# Validation failure path
# ============================================================================


@pytest.mark.timeout(1.0)
def test_validation_failure_can_continue(monkeypatch: pytest.MonkeyPatch) -> None:
    """When genai.Client raises, user can opt to continue."""

    def boom(**_: Any) -> None:
        raise RuntimeError("nope")

    monkeypatch.setattr(FakeGenaiModule, "Client", boom)

    fake_print = FakePrint()
    config = run_wizard(
        interactive=False,
        mode="gemini",
        api_key="bad",
        data_folder="/d",
        continue_on_validation_error=True,
        input_fn=lambda _="": "",
        print_fn=fake_print,
    )
    assert config["mcpServers"]["gemini-media"]["env"]["GEMINI_API_KEY"] == "bad"


@pytest.mark.timeout(1.0)
def test_validation_performs_live_models_list() -> None:
    """Successful validation makes a live models.list() call."""
    fake_input = FakeInput(answers=["g", "good-key", "", ""])
    fake_print = FakePrint()

    run_wizard(interactive=True, input_fn=fake_input, print_fn=fake_print)

    assert len(FakeGenaiClient.instances) == 1
    assert FakeGenaiClient.instances[0].models.list_calls == 1
    assert any("validated successfully" in line for line in fake_print.lines)


@pytest.mark.timeout(1.0)
def test_validation_fails_when_live_call_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A working construction but failing live call is a validation failure."""

    def bad_list(self: Any) -> list[str]:
        raise RuntimeError("PERMISSION_DENIED: bad api key")

    monkeypatch.setattr(FakeModels, "list", bad_list)

    with pytest.raises(RuntimeError, match="Setup aborted"):
        run_wizard(
            interactive=False,
            mode="gemini",
            api_key="bad",
            data_folder="/d",
            continue_on_validation_error=False,
            input_fn=lambda _="": "",
            print_fn=FakePrint(),
        )


@pytest.mark.timeout(1.0)
def test_validation_failure_aborts_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without continue flag in non-interactive mode, abort."""

    def boom(**_: Any) -> None:
        raise RuntimeError("nope")

    monkeypatch.setattr(FakeGenaiModule, "Client", boom)

    with pytest.raises(RuntimeError, match="Setup aborted"):
        run_wizard(
            interactive=False,
            mode="gemini",
            api_key="bad",
            data_folder="/d",
            continue_on_validation_error=False,
            input_fn=lambda _="": "",
            print_fn=FakePrint(),
        )


# ============================================================================
# macos_config_path
# ============================================================================


@pytest.mark.timeout(1.0)
def test_macos_config_path_points_to_application_support() -> None:
    """Default macOS path lives under ~/Library/Application Support/Claude."""
    path = macos_config_path()
    assert path.name == "claude_desktop_config.json"
    assert "Application Support" in str(path)
    assert "Claude" in str(path)
