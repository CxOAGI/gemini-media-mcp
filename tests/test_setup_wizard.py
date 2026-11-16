"""Tests for setup_wizard.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from src.client_config import build_config, default_config_path, merge_config
from src.setup_wizard import run_wizard

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
    for name in (
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GOOGLE_GENAI_USE_VERTEXAI",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_SERVICE_ACCOUNT_JSON",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
    ):
        monkeypatch.delenv(name, raising=False)
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
    cfg = build_config("claude-desktop", env)
    assert Path(cfg["mcpServers"]["gemini-media"]["command"]).name in {"uvx", "uvx.exe"}
    assert cfg["mcpServers"]["gemini-media"]["args"][-2:] == [
        "gemini-media-mcp",
        "stdio",
    ]
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
        write_config=False,
        input_fn=fake_input,
        print_fn=fake_print,
    )

    env = config["mcpServers"]["gemini-media"]["env"]
    assert env["GEMINI_API_KEY"] == "my-key"
    assert env["GOOGLE_GENAI_USE_VERTEXAI"] == "false"
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
        write_config=False,
        input_fn=fake_input,
        print_fn=fake_print,
    )

    env = config["mcpServers"]["gemini-media"]["env"]
    assert env["GOOGLE_GENAI_USE_VERTEXAI"] == "true"
    assert env["GOOGLE_CLOUD_PROJECT"] == "my-project"
    assert env["GOOGLE_CLOUD_LOCATION"] == "us-central1"
    assert env["GOOGLE_APPLICATION_CREDENTIALS"] == str(sa_file)
    assert not env["GOOGLE_SERVICE_ACCOUNT_JSON"]


# ============================================================================
# Vertex + inline paste
# ============================================================================


@pytest.mark.timeout(1.0)
def test_vertex_with_inline_paste(service_account_json: str) -> None:
    """Vertex flow with inline JSON paste stores JSON string in env."""
    sa = json.loads(service_account_json)
    sa_json = json.dumps(sa)

    fake_input = FakeInput(
        answers=[
            "v",  # mode
            "inline",  # project id
            "us-west1",  # location
            sa_json,  # first line of paste (ends with })
            "",  # data folder default
            "",  # video gcs bucket skip
        ]
    )
    fake_print = FakePrint()

    config = run_wizard(
        interactive=True,
        write_config=False,
        input_fn=fake_input,
        print_fn=fake_print,
    )

    env = config["mcpServers"]["gemini-media"]["env"]
    assert env["GOOGLE_CLOUD_LOCATION"] == "us-west1"
    assert not env["GOOGLE_APPLICATION_CREDENTIALS"]
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
    service_account_json: str,
) -> None:
    """Non-interactive mode should never call input()."""

    def forbidden_input(_: str = "") -> str:
        raise AssertionError("input() must not be called in non-interactive mode")

    fake_print = FakePrint()
    if "sa_json" in overrides:
        overrides = {**overrides, "sa_json": service_account_json}

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

    run_wizard(
        interactive=True, input_fn=fake_input, print_fn=fake_print, write_config=False
    )

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
def test_macos_config_path_points_to_application_support(monkeypatch) -> None:
    """Default macOS path lives under ~/Library/Application Support/Claude."""
    monkeypatch.setattr(sys, "platform", "darwin")
    path = default_config_path("claude-desktop")
    assert path.name == "claude_desktop_config.json"
    assert "Application Support" in str(path)
    assert "Claude" in str(path)


def test_written_config_is_not_readable_by_other_accounts(tmp_path: Path) -> None:
    """The wizard writes live credentials, so the file must be owner-only.

    In Vertex mode the block carries GOOGLE_SERVICE_ACCOUNT_JSON — an entire
    service-account private key. Left to the umask it landed at 0664, readable
    by every other account on the host.
    """
    import stat

    config_path = tmp_path / "claude_desktop_config.json"
    block = build_config("claude-desktop", {"GEMINI_API_KEY": "AIza-secret"})

    merge_config(config_path, block)

    mode = stat.S_IMODE(config_path.stat().st_mode)
    assert not mode & (stat.S_IRGRP | stat.S_IROTH | stat.S_IWGRP | stat.S_IWOTH)
    assert "AIza-secret" in config_path.read_text()


def test_backup_of_a_wide_open_config_is_narrowed(tmp_path: Path) -> None:
    """copy2 preserves the source mode, so a pre-existing world-readable
    config would otherwise be duplicated exactly as permissive."""
    import stat

    config_path = tmp_path / "claude_desktop_config.json"
    config_path.write_text('{"mcpServers": {"old": {}}}')
    config_path.chmod(0o644)

    backup = merge_config(
        config_path, build_config("claude-desktop", {"GEMINI_API_KEY": "AIza-secret"})
    )

    assert backup is not None
    for path in (config_path, backup):
        mode = stat.S_IMODE(path.stat().st_mode)
        assert not mode & (stat.S_IRGRP | stat.S_IROTH), f"{path.name} is {oct(mode)}"


def test_an_existing_config_is_narrowed_before_the_secret_is_written(
    tmp_path: Path,
) -> None:
    """O_CREAT does not change an existing file's mode, so a config that was
    already world-readable would keep those bits while gaining a fresh key."""
    import stat

    config_path = tmp_path / "claude_desktop_config.json"
    config_path.write_text("{}")
    config_path.chmod(0o666)

    merge_config(
        config_path, build_config("claude-desktop", {"GEMINI_API_KEY": "AIza-new"})
    )

    mode = stat.S_IMODE(config_path.stat().st_mode)
    assert not mode & (stat.S_IRGRP | stat.S_IROTH)
    assert "AIza-new" in config_path.read_text()


@pytest.mark.parametrize("client", ["claude-desktop", "vscode", "claude-code"])
def test_unavailable_install_fails_before_collecting_credentials(monkeypatch, client):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr("src.setup_wizard.shutil.which", lambda _: None)
    prompts = []
    with pytest.raises(ValueError):
        run_wizard(
            client=client, write_config=True, input_fn=lambda p: prompts.append(p)
        )
    assert not prompts
    assert not FakeGenaiClient.instances


def test_linux_default_offers_export_before_credentials(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # expanduser uses the OS account environment, so provide an explicit export.
    export = tmp_path / "export.json"
    prompts = FakeInput([str(export), "g", "test-key", ""])
    result = run_wizard(input_fn=prompts, print_fn=FakePrint())
    assert prompts.prompts[0].startswith("Export configuration to")
    assert json.loads(export.read_text()) == result
    assert result["mcpServers"]["gemini-media"]["env"]["GEMINI_API_KEY"] == "test-key"


@pytest.mark.parametrize("option", ["output", "config_path"])
@pytest.mark.parametrize("dangling", [False, True])
def test_wizard_refuses_symlinks_before_credentials(tmp_path, option, dangling):
    victim = tmp_path / "victim"
    if not dangling:
        victim.write_text("keep me")
    link = tmp_path / "config.json"
    link.symlink_to(victim)
    prompts = []
    with pytest.raises(ValueError, match="symlink"):
        run_wizard(
            write_config=True, input_fn=lambda p: prompts.append(p), **{option: link}
        )
    assert not prompts
    assert link.is_symlink()
    assert not victim.exists() if dangling else victim.read_text() == "keep me"


@pytest.mark.parametrize(
    "options", [{}, {"sa_json": ""}, {"sa_path": ""}, {"sa_json": " "}]
)
def test_scripted_vertex_requires_explicit_nonempty_credential_source(
    monkeypatch, options
):
    monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_JSON", '{"project_id":"ambient"}')
    with pytest.raises(ValueError, match="--adc"):
        run_wizard(
            interactive=False,
            mode="vertex",
            project_id="p",
            skip_validation=True,
            **options,
        )


@pytest.mark.parametrize(
    "payload", ["null", " \nnull\t", "[]", "false", "0", '"private-value"']
)
@pytest.mark.parametrize("skip_validation", [False, True])
def test_vertex_rejects_non_object_inline_credentials_before_validation_or_write(
    tmp_path, monkeypatch, payload, skip_validation
):
    monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_JSON", '{"project_id":"ambient"}')
    output = tmp_path / "export.json"
    original = '{"keep":"existing configuration"}\n'
    output.write_text(original)
    with pytest.raises(ValueError, match="Service-account JSON must be an object"):
        run_wizard(
            interactive=False,
            mode="vertex",
            project_id="p",
            sa_json=payload,
            skip_validation=skip_validation,
            output=output,
            print_fn=FakePrint(),
        )
    assert not FakeGenaiClient.instances
    assert output.read_text() == original


def test_prompted_service_account_path_is_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "key.json").write_text('{"type":"service_account","project_id":"p"}')
    result = run_wizard(
        mode="vertex",
        project_id="p",
        location="us-central1",
        write_config=False,
        input_fn=FakeInput(["key.json", "", ""]),
        skip_validation=True,
        print_fn=FakePrint(),
    )
    assert result["mcpServers"]["gemini-media"]["env"][
        "GOOGLE_APPLICATION_CREDENTIALS"
    ] == str(tmp_path / "key.json")


@pytest.mark.parametrize("reuse", ["", "y"])
def test_ambient_inline_key_requires_disclosed_choice(monkeypatch, reuse, capsys):
    secret = '{"project_id":"ambient","private_key":"keep-private"}'
    monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_JSON", secret)
    answers = [reuse, "", ""] if reuse else ["", "", "", ""]
    prompts = FakeInput(answers)
    result = run_wizard(
        mode="vertex",
        project_id="p",
        location="us-central1",
        write_config=False,
        input_fn=prompts,
        skip_validation=True,
        print_fn=FakePrint(),
    )
    selected = result["mcpServers"]["gemini-media"]["env"][
        "GOOGLE_SERVICE_ACCOUNT_JSON"
    ]
    assert bool(selected) == bool(reuse)
    assert "embeds its private key" in prompts.prompts[0]
    output = capsys.readouterr().out
    assert "keep-private" not in output
    assert (
        "inline service-account JSON" in output
        if reuse
        else "Selected credentials: Application Default Credentials" in output
    )


def test_adc_validation_gives_safe_recovery_and_defaults_to_abort(monkeypatch):
    from google.auth.exceptions import DefaultCredentialsError

    def missing_adc(**kwargs):
        raise DefaultCredentialsError("provider text with secret")

    monkeypatch.setattr(FakeGenaiModule, "Client", missing_adc)
    printed = FakePrint()
    with pytest.raises(RuntimeError, match="gcloud auth application-default login"):
        run_wizard(
            mode="vertex",
            adc=True,
            project_id="p",
            location="us-central1",
            write_config=False,
            input_fn=FakeInput(["", "", ""]),
            print_fn=printed,
        )
    assert "provider text with secret" not in str(printed.lines)
    assert "some valid" not in str(printed.lines)


@pytest.mark.parametrize(
    "payload,message",
    [("", "sa_json is empty"), ("null", "Service-account JSON must be an object")],
)
def test_cli_rejects_invalid_explicit_sa_json(monkeypatch, capsys, payload, message):
    from src.__main__ import main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gemini-media-mcp",
            "setup",
            "--non-interactive",
            "--mode",
            "vertex",
            "--project-id",
            "p",
            "--sa-json",
            payload,
            "--skip-validation",
            "--preview",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert message in error
    assert "Traceback" not in error


def test_jsonc_is_rejected_before_credentials_or_validation(tmp_path):
    path = tmp_path / "mcp.json"
    original = '{\n// keep my settings\n"servers": {}\n}'
    path.write_text(original)
    prompts = FakeInput([])
    with pytest.raises(ValueError, match="not strict JSON"):
        run_wizard(
            client="vscode", config_path=path, write_config=True, input_fn=prompts
        )
    assert not prompts.prompts
    assert not FakeGenaiClient.instances
    assert path.read_text() == original
    assert not path.with_suffix(".json.bak").exists()


def test_invalid_api_key_error_gives_key_advice_without_provider_details():
    from src.setup_wizard import _validation_error

    class InvalidKey(Exception):
        code = 400

    message = _validation_error(InvalidKey("provider echoed private-key-value"))
    assert "API key" in message and "Google AI Studio" in message
    assert "private-key-value" not in message


@pytest.mark.parametrize("write", [None, False, True])
def test_claude_inline_json_is_rejected_before_any_prompt(write, capsys):
    prompts = FakeInput([])
    with pytest.raises(ValueError, match="--adc or --sa-path"):
        run_wizard(
            client="claude-code",
            sa_json='{"private_key":"secret"}',
            write_config=write,
            input_fn=prompts,
        )
    assert not prompts.prompts
    assert not FakeGenaiClient.instances
    assert "Selected credentials" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "answers,expected",
    [
        (["", ""], ("old-project", "asia-east1")),
        (["new-project", "us-central1"], ("new-project", "us-central1")),
    ],
)
def test_interactive_vertex_confirms_inherited_project_and_location(
    monkeypatch, answers, expected
):
    from src.setup_wizard import _collect_vertex_env

    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "old-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "asia-east1")
    prompts = FakeInput(answers)
    env = _collect_vertex_env(True, {"adc": True}, prompts)
    assert "[old-project]" in prompts.prompts[0]
    assert "[asia-east1]" in prompts.prompts[1]
    assert (env["GOOGLE_CLOUD_PROJECT"], env["GOOGLE_CLOUD_LOCATION"]) == expected


def test_old_scripted_cli_requires_an_explicit_output_choice(monkeypatch, capsys):
    from src.__main__ import main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gemini-media-mcp",
            "setup",
            "--non-interactive",
            "--mode",
            "gemini",
            "--api-key",
            "private-value",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    output = capsys.readouterr()
    assert "--output PATH" in output.err and "--preview" in output.err
    assert "private-value" not in output.err + output.out
    assert not FakeGenaiClient.instances


@pytest.mark.parametrize("vertex", [False, True])
def test_sdk_backend_switch_is_neutralized_and_restored(monkeypatch, vertex):
    import os
    from src.setup_wizard import _validate_client

    ambient = "false" if vertex else "true"
    monkeypatch.setenv("GOOGLE_GENAI_USE_ENTERPRISE", ambient)

    class CheckClient(FakeGenaiClient):
        def __init__(self, **kwargs):
            assert os.environ["GOOGLE_GENAI_USE_ENTERPRISE"] == str(vertex).lower()
            super().__init__(**kwargs)

    monkeypatch.setattr(FakeGenaiModule, "Client", CheckClient)
    ok, error = _validate_client(
        {
            "GOOGLE_GENAI_USE_VERTEXAI": str(vertex).lower(),
            "GEMINI_API_KEY": "dummy",
            "GOOGLE_CLOUD_PROJECT": "p",
        }
    )
    assert ok, error
    assert os.environ["GOOGLE_GENAI_USE_ENTERPRISE"] == ambient
