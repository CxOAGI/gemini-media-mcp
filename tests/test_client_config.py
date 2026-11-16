"""Configuration changes must preserve clients' settings and keep keys private."""

import json
import os
import stat
import subprocess
import tomllib
from pathlib import Path

import pytest

from src.client_config import build_config, configure_claude_code, merge_config
from src.setup_wizard import _validate_client, run_wizard


@pytest.mark.parametrize(
    "client,root",
    [
        ("cursor", "mcpServers"),
        ("gemini-cli", "mcpServers"),
        ("cline", "mcpServers"),
        ("vscode", "servers"),
        ("copilot", "mcpServers"),
    ],
)
def test_merge_twice_preserves_settings_and_protects_secrets(tmp_path, client, root):
    path = tmp_path / "settings.json"
    original = {
        "theme": "dark",
        root: {"other": {"command": "node"}},
        "inputs": [{"id": "other", "type": "promptString"}],
    }
    path.write_text(json.dumps(original))
    block = build_config(client, {"GEMINI_API_KEY": "a-secret"})
    merge_config(path, block)
    merge_config(path, block)
    value = json.loads(path.read_text())
    assert value["theme"] == "dark"
    assert value[root]["other"] == original[root]["other"]
    assert len(value[root]) == 2
    if client == "vscode":
        assert "a-secret" not in path.read_text()
        assert len(value["inputs"]) == 2
        assert value["inputs"][-1]["password"] is True
    if os.name != "nt":
        for target in (path, path.with_suffix(".json.bak")):
            assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_codex_preserves_comments_and_other_tables(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        '# Keep my settings\nmodel = "test"\n[mcp_servers.other]\ncommand = "node"\n'
    )
    block = build_config("codex", {"GEMINI_API_KEY": 'quote" and \\ slash'})
    merge_config(path, block, toml=True)
    merge_config(path, block, toml=True)
    text = path.read_text()
    parsed = tomllib.loads(text)
    assert "# Keep my settings" in text
    assert parsed["model"] == "test"
    assert parsed["mcp_servers"]["other"]["command"] == "node"
    assert parsed["mcp_servers"]["gemini-media"] == block["mcp_servers"]["gemini-media"]


@pytest.mark.parametrize(
    "content", ["{bad", '{"mcpServers": []}', "[]", "// comment\n{}"]
)
def test_invalid_config_is_never_replaced(tmp_path, content):
    path = tmp_path / "mcp.json"
    path.write_text(content)
    with pytest.raises(ValueError):
        merge_config(path, build_config("cursor", {}))
    assert path.read_text() == content
    assert not path.with_suffix(".json.bak").exists()


def test_setup_export_masks_key_and_expands_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "never-print-me")
    output = tmp_path / "export.json"
    printed = []
    result = run_wizard(
        interactive=False,
        client="cursor",
        mode="gemini",
        data_folder="relative media",
        skip_validation=True,
        output=output,
        print_fn=lambda *v: printed.append(str(v)),
    )
    assert "never-print-me" not in "".join(printed)
    assert json.loads(output.read_text()) == result
    assert Path(
        result["mcpServers"]["gemini-media"]["env"]["DATA_FOLDER"]
    ).is_absolute()


@pytest.mark.parametrize("profile", [None, "", "isolated profile"])
def test_gemini_wizard_installs_into_the_selected_profile(
    tmp_path, monkeypatch, profile
):
    normal_home = tmp_path / "normal-home"
    monkeypatch.setattr(Path, "home", lambda: normal_home)
    if profile is None:
        monkeypatch.delenv("GEMINI_CLI_HOME", raising=False)
    else:
        monkeypatch.setenv(
            "GEMINI_CLI_HOME", str(tmp_path / profile) if profile else ""
        )
    selected_home = tmp_path / profile if profile else normal_home
    target = selected_home / ".gemini/settings.json"
    target.parent.mkdir(parents=True)
    original = '{"theme":"dark","mcpServers":{"other":{"command":"node"}}}\n'
    target.write_text(original)
    normal_config = normal_home / ".gemini/settings.json"
    if selected_home != normal_home:
        normal_config.parent.mkdir(parents=True)
        normal_config.write_text(original)

    run_wizard(
        interactive=False,
        client="gemini-cli",
        mode="gemini",
        api_key="profile-specific-key",
        data_folder=str(tmp_path / "media"),
        skip_validation=True,
        write_config=True,
        print_fn=lambda *a: None,
    )
    saved = json.loads(target.read_text())
    assert (
        saved["mcpServers"]["gemini-media"]["env"]["GEMINI_API_KEY"]
        == "profile-specific-key"
    )
    assert saved["theme"] == "dark"
    assert saved["mcpServers"]["other"] == {"command": "node"}
    assert target.with_suffix(".json.bak").read_text() == original
    if selected_home != normal_home:
        assert normal_config.read_text() == original
        assert not normal_config.with_suffix(".json.bak").exists()


def test_vertex_adc_does_not_require_a_key_file(tmp_path):
    result = run_wizard(
        interactive=False,
        client="codex",
        mode="vertex",
        project_id="project",
        adc=True,
        skip_validation=True,
        output=tmp_path / "config.toml",
        print_fn=lambda *a: None,
    )
    env = result["mcp_servers"]["gemini-media"]["env"]
    assert env["GOOGLE_GENAI_USE_VERTEXAI"] == "true"
    assert env["OMNI_BACKEND"] == "vertex"
    assert env["GEMINI_API_KEY"] == env["GOOGLE_APPLICATION_CREDENTIALS"] == ""


def test_claude_uses_stdin_for_sensitive_config(monkeypatch):
    calls = []
    monkeypatch.setattr("src.client_config.shutil.which", lambda _: "/bin/claude")

    def run(command, **kwargs):
        calls.append((command, kwargs))
        output = {"configured": ["gemini_api_key"]}
        if command[2] == "list":
            output = (
                [{"id": "gemini-media@cxoagi", "scope": "user", "enabled": True}]
                if any(cmd[2] == "install" for cmd, _ in calls)
                else []
            )
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(output))

    monkeypatch.setattr("src.client_config.subprocess.run", run)
    configure_claude_code(
        {
            "DATA_FOLDER": "/media",
            "GEMINI_API_KEY": "secret-value",
            "OMNI_BACKEND": "gemini_api",
            "GOOGLE_CLOUD_LOCATION": "",
        }
    )
    assert all("secret-value" not in str(cmd) for cmd, _ in calls)
    supplied = next(
        json.loads(kwargs["input"]) for _, kwargs in calls if kwargs.get("input")
    )
    assert supplied["gemini_api_key"] == "secret-value"
    assert supplied["google_cloud_location"] == "us-central1"
    install = next(cmd for cmd, _ in calls if cmd[2] == "install")
    assert "data_folder=/media" in install
    assert calls[-1][0][2] == "list"


def test_validation_restores_environment_and_does_not_leak_errors(monkeypatch):
    from google import genai

    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.setenv("GOOGLE_API_KEY", "ambient")

    def broken(**kwargs):
        assert kwargs["vertexai"] is False
        assert "GOOGLE_API_KEY" not in os.environ
        raise ValueError("secret-value")

    monkeypatch.setattr(genai, "Client", broken)
    ok, error = _validate_client({"GEMINI_API_KEY": "secret-value"})
    assert not ok and "secret-value" not in error
    assert os.environ["GOOGLE_GENAI_USE_VERTEXAI"] == "true"
    assert os.environ["GOOGLE_API_KEY"] == "ambient"


@pytest.mark.parametrize("toml", [False, True])
def test_merge_accepts_bom_and_preserves_exact_backup(tmp_path, toml):
    path = tmp_path / ("config.toml" if toml else "config.json")
    body = (
        '# keep comment\r\nmodel = "example"\r\n'
        if toml
        else '{\r\n  "theme": "dark"\r\n}\r\n'
    )
    original = b"\xef\xbb\xbf" + body.encode()
    path.write_bytes(original)
    backup = merge_config(
        path, build_config("codex" if toml else "cursor", {}), toml=toml
    )
    assert backup.read_bytes() == original
    written = path.read_bytes()
    assert written.startswith(b"\xef\xbb\xbf")
    assert b"\n" not in written.replace(b"\r\n", b"")
    parsed = (tomllib.loads if toml else json.loads)(written.decode("utf-8-sig"))
    assert parsed["model" if toml else "theme"] == ("example" if toml else "dark")


def test_backup_symlink_is_never_followed(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{}")
    victim = tmp_path / "victim"
    victim.write_text("keep")
    path.with_suffix(".json.bak").symlink_to(victim)
    with pytest.raises(ValueError, match="symlink"):
        merge_config(path, build_config("cursor", {}))
    assert victim.read_text() == "keep"
    assert path.read_text() == "{}"


def test_vscode_config_has_no_machine_specific_paths_or_credentials(monkeypatch):
    monkeypatch.setattr(
        "src.client_config.shutil.which", lambda _: "/home/alice/.local/bin/uvx"
    )
    config = build_config(
        "vscode",
        {
            "DATA_FOLDER": "/home/alice/media",
            "GOOGLE_APPLICATION_CREDENTIALS": "/home/alice/key.json",
            "GEMINI_API_KEY": "private-value",
        },
    )
    rendered = json.dumps(config)
    assert "/home/alice" not in rendered
    assert "private-value" not in rendered
    server = config["servers"]["gemini-media"]
    assert server["command"] == "uvx"
    assert server["env"]["DATA_FOLDER"].startswith("${input:")


def test_release_wizard_installs_its_own_release(monkeypatch):
    monkeypatch.setattr("src.client_config.metadata.version", lambda _: "0.1.15")
    config = build_config("cursor", {})
    assert "gemini-media-mcp==0.1.15" in config["mcpServers"]["gemini-media"]["args"]


def test_development_pin_matches_distribution_manifests(monkeypatch):
    monkeypatch.setattr(
        "src.client_config.metadata.version", lambda _: "0.1.15.dev1+test"
    )
    root = Path(__file__).resolve().parents[1]
    release = json.loads((root / "src/client_release.json").read_text())["version"]
    pin = f"gemini-media-mcp=={release}"
    assert pin in build_config("cursor", {})["mcpServers"]["gemini-media"]["args"]
    for file in ("plugins/gemini-media/.mcp.json", "gemini-extension.json"):
        manifest = json.loads((root / file).read_text())
        assert pin in manifest["mcpServers"]["gemini-media"]["args"]
    for file in ("README.md", "llms-install.md", "plugins/gemini-media/README.md"):
        assert pin in (root / file).read_text()


def test_codex_split_tables_survive_repeated_setup(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        '# top\n[mcp_servers.gemini-media]\ncommand="old"\n\n[other]\ny=1\n\n[mcp_servers.gemini-media.env]\nOLD="1"\n\n[mcp_servers.b]\n# keep b\ncommand="node"\n'
    )
    block = build_config("codex", {"GEMINI_API_KEY": "new"})
    for _ in range(2):
        merge_config(path, block, toml=True)
        parsed = tomllib.loads(path.read_text())
        assert parsed["mcp_servers"]["b"] == {"command": "node"}
        assert (
            parsed["mcp_servers"]["gemini-media"]
            == block["mcp_servers"]["gemini-media"]
        )
        assert parsed["other"] == {"y": 1}
        assert "# keep b" in path.read_text()
        assert tomllib.loads(path.with_suffix(".toml.bak").read_text())["mcp_servers"][
            "b"
        ] == {"command": "node"}


def test_codex_merge_refuses_unrelated_changes_before_backup(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    original = '[mcp_servers.other]\ncommand="node"\n'
    path.write_text(original)
    monkeypatch.setattr("tomlkit.dumps", lambda _: "")
    with pytest.raises(ValueError, match="unrelated settings"):
        merge_config(path, build_config("codex", {}), toml=True)
    assert path.read_text() == original
    assert not path.with_suffix(".toml.bak").exists()


@pytest.mark.parametrize("version", ["0.0.0", "0.1.0"])
def test_docker_placeholder_version_uses_public_pin(monkeypatch, version):
    from src.client_config import server_package

    monkeypatch.setattr("src.client_config.metadata.version", lambda _: version)
    release = json.loads(
        (Path(__file__).resolve().parents[1] / "src/client_release.json").read_text()
    )["version"]
    assert server_package() == f"gemini-media-mcp=={release}"


def test_claude_preview_and_export_are_native_options(tmp_path):
    from src.client_config import claude_config_values

    path = tmp_path / "claude-options.json"
    result = run_wizard(
        interactive=False,
        client="claude-code",
        mode="gemini",
        api_key="secret",
        data_folder=str(tmp_path / "media"),
        output=path,
        skip_validation=True,
        print_fn=lambda *a: None,
    )
    assert result == claude_config_values(
        {
            "GEMINI_API_KEY": "secret",
            "DATA_FOLDER": str(tmp_path / "media"),
            "OMNI_BACKEND": "gemini_api",
        }
    )
    assert json.loads(path.read_text()) == result
    assert "command" not in result and "GCS_ALLOWED_BUCKETS" not in result


@pytest.mark.parametrize("action", ["install", "configure"])
@pytest.mark.parametrize("timeout", [False, True])
def test_claude_failed_setup_disables_plugin_without_leaking_output(
    monkeypatch, action, timeout
):
    calls = []
    monkeypatch.setattr("src.client_config.shutil.which", lambda _: "/bin/claude")

    def run(command, **kwargs):
        calls.append(command)
        if command[2] == action:
            if timeout:
                raise subprocess.TimeoutExpired(command, 180, output="secret-value")
            return subprocess.CompletedProcess(
                command, 1, stdout="secret-value", stderr="secret-value"
            )
        return subprocess.CompletedProcess(command, 0, stdout="[]")

    monkeypatch.setattr("src.client_config.subprocess.run", run)
    with pytest.raises(RuntimeError, match="disabled") as exc:
        configure_claude_code(
            {
                "GEMINI_API_KEY": "secret-value",
                "DATA_FOLDER": "/media",
                "OMNI_BACKEND": "gemini_api",
            }
        )
    assert "secret-value" not in str(exc.value)
    assert calls[-1][2] == "disable"
    assert not any(cmd[2] == "enable" for cmd in calls)


def test_claude_does_not_reset_secrets_shared_with_other_scopes(monkeypatch):
    calls = []
    monkeypatch.setattr("src.client_config.shutil.which", lambda _: "/bin/claude")

    def run(command, **kwargs):
        calls.append(command)
        data = (
            [
                {"id": "gemini-media@cxoagi", "scope": "user"},
                {"id": "gemini-media@cxoagi", "scope": "project"},
            ]
            if command[2] == "list"
            else {"configured": ["gemini_api_key"]}
        )
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(data))

    monkeypatch.setattr("src.client_config.subprocess.run", run)
    with pytest.raises(ValueError, match="other scopes"):
        configure_claude_code(
            {
                "DATA_FOLDER": "/media",
                "GOOGLE_GENAI_USE_VERTEXAI": "true",
                "OMNI_BACKEND": "vertex",
            }
        )
    assert all(
        cmd[2] in {"list", "configure"} and "--values-stdin" not in cmd for cmd in calls
    )
