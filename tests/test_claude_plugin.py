"""Opt-in, free smoke checks of the server release shipped by the Claude plugin.

These need PyPI access on a cold uv cache, so normal unit tests skip them.
The CLI validator and isolated install in CI cover Claude's own config loading;
this test substitutes dummy values and exercises the actual MCP launch command.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        os.environ.get("CLAUDE_PLUGIN_SMOKE") != "1",
        reason="set CLAUDE_PLUGIN_SMOKE=1 to test the pinned PyPI plugin server",
    ),
    pytest.mark.timeout(180),
]

PLUGIN = Path(__file__).resolve().parents[1] / "plugins" / "gemini-media"


@pytest.mark.parametrize("mode", ["gemini-api", "vertex-ai", "gemini-extension"])
async def test_plugin_server(
    tmp_path: Path, service_account_json: str, mode: str
) -> None:
    vertex = mode == "vertex-ai"
    manifest = json.loads((PLUGIN / ".claude-plugin" / "plugin.json").read_text())
    server = json.loads((PLUGIN / ".mcp.json").read_text())["mcpServers"][
        "gemini-media"
    ]
    values = {
        key: field["default"]
        for key, field in manifest["userConfig"].items()
        if "default" in field
    }
    # Spaces exercise argv/env handling; no shell should interpret these paths.
    media = tmp_path / "generated media"
    credentials = tmp_path / "dummy credentials.json"
    credentials.write_text(service_account_json)
    values.update(
        data_folder=str(media),
        gemini_api_key="" if vertex else "plugin-smoke-placeholder",
        use_vertex=vertex,
        google_cloud_project="proj-x" if vertex else "",
        google_application_credentials=str(credentials) if vertex else "",
        omni_backend="vertex" if vertex else "auto",
    )

    def substitute(match: re.Match[str]) -> str:
        value = values[match.group(1)]
        return str(value).lower() if isinstance(value, bool) else str(value)

    # Do not copy the caller's cloud credentials into the child process.
    env = {
        key: os.environ[key]
        for key in ("PATH", "SYSTEMROOT", "UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR")
        if key in os.environ
    }
    env["GOOGLE_GENAI_USE_ENTERPRISE"] = "false" if vertex else "true"
    if mode == "gemini-extension":
        extension = json.loads(
            (PLUGIN.parents[1] / "gemini-extension.json").read_text()
        )
        server = extension["mcpServers"]["gemini-media"]
        settings = {
            "GEMINI_MEDIA_API_KEY": "plugin-smoke-placeholder",
            "GEMINI_MEDIA_DATA_FOLDER": str(media),
        }
        env.update(
            {
                key: re.sub(
                    r"\$\{([A-Z_]+)\}", lambda match: settings[match.group(1)], value
                )
                for key, value in server["env"].items()
            }
        )
    else:
        env.update(
            {
                key: re.sub(r"\$\{user_config\.([a-z_]+)\}", substitute, value)
                for key, value in server["env"].items()
            }
        )
    # Keep local ADC fallback out of the developer's gcloud configuration.
    assert env["GOOGLE_GENAI_USE_ENTERPRISE"] == str(vertex).lower()
    env["CLOUDSDK_CONFIG"] = str(tmp_path / "empty-gcloud")
    params = StdioServerParameters(
        command=server["command"], args=server["args"], env=env, cwd=str(tmp_path)
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            names = {tool.name for tool in (await session.list_tools()).tools}
            assert {
                "plan_generation",
                "generate_image",
                "generate_video",
                "generate_video_omni",
            } <= names
            result = await session.call_tool(
                "plan_generation", {"intent": "Generate an image of a ceramic mug"}
            )
            assert not result.isError
            payload = json.loads(
                next(c.text for c in result.content if c.type == "text")
            )
            assert "error" not in payload, payload
            assert payload["is_satisfiable"], payload
            assert payload["routes"][0]["tool"] == "generate_image", payload
    assert (media / "images").is_dir()
    assert (media / "videos").is_dir()
    assert not any(p.is_file() for p in media.rglob("*")), (
        "a planning call must not generate files"
    )


async def test_native_claude_key_reset_and_failed_configuration(tmp_path, monkeypatch):
    """Exercise native secret storage and failure recovery, not a mocked store."""
    from src import client_config

    claude = shutil.which("claude")
    if not claude:
        pytest.skip("Install Claude Code to exercise its native credential store")
    checkout = PLUGIN.parents[1]
    marketplace = tmp_path / "marketplace"
    shutil.copytree(checkout / ".claude-plugin", marketplace / ".claude-plugin")
    plugin = marketplace / "plugins/gemini-media"
    shutil.copytree(PLUGIN, plugin)
    # A unique marketplace also isolates secure-storage entries on platforms
    # where CLAUDE_CONFIG_DIR alone might not isolate the system keychain.
    name = "review-" + uuid.uuid4().hex
    plugin_id = "gemini-media@" + name
    catalog = marketplace / ".claude-plugin/marketplace.json"
    manifest = json.loads(catalog.read_text())
    manifest["name"] = name
    catalog.write_text(json.dumps(manifest))
    capture = tmp_path / "launched-env.json"
    mcp_file = plugin / ".mcp.json"
    mcp = json.loads(mcp_file.read_text())
    server = mcp["mcpServers"]["gemini-media"]
    server.update(
        command=sys.executable,
        args=[
            "-c",
            "import json,os;from pathlib import Path;Path(os.environ['TEST_CAPTURE']).write_text(json.dumps({'key_present':bool(os.environ.get('GEMINI_API_KEY')),'vertex':os.environ.get('GOOGLE_GENAI_USE_VERTEXAI'),'enterprise':os.environ.get('GOOGLE_GENAI_USE_ENTERPRISE')}))",
        ],
    )
    server["env"]["TEST_CAPTURE"] = str(capture)
    mcp_file.write_text(json.dumps(mcp))
    profile = tmp_path / "claude-profile"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(profile))
    monkeypatch.setattr(client_config, "MARKETPLACE", str(marketplace))
    monkeypatch.setattr(client_config, "CLAUDE_PLUGIN", plugin_id)

    def native(*args):
        result = subprocess.run(
            [claude, *args], capture_output=True, text=True, timeout=60, cwd=tmp_path
        )
        assert result.returncode == 0, result.stderr
        return result.stdout

    def launched_env():
        capture.unlink(missing_ok=True)
        # This probe only records presence of the dummy key, then exits. A
        # failed MCP health check is expected; it never talks to Google.
        native("mcp", "list")
        return json.loads(capture.read_text())

    try:
        client_config.configure_claude_code(
            {
                "GEMINI_API_KEY": "dummy-old-key",
                "DATA_FOLDER": str(tmp_path / "media"),
                "OMNI_BACKEND": "gemini_api",
            }
        )
        assert launched_env()["key_present"] is True
        data = profile / "plugins/data" / re.sub(r"[^a-zA-Z0-9_-]", "-", plugin_id)
        data.mkdir(parents=True, exist_ok=True)
        (data / "keep.txt").write_text("preserve plugin data")
        vertex = {
            "DATA_FOLDER": str(tmp_path / "media"),
            "GOOGLE_GENAI_USE_VERTEXAI": "true",
            "GOOGLE_CLOUD_PROJECT": "p",
            "OMNI_BACKEND": "vertex",
        }
        client_config.configure_claude_code(vertex)
        assert launched_env() == {
            "key_present": False,
            "vertex": "true",
            "enterprise": "true",
        }
        assert (data / "keep.txt").read_text() == "preserve plugin data"
        configured = json.loads(native("plugin", "configure", plugin_id, "--json"))
        assert "gemini_api_key" not in configured["configured"]

        real_run = client_config._claude_run
        with monkeypatch.context() as failure:

            def fail_configure(executable, action, *args, **kwargs):
                if action == "configure" and "--values-stdin" in args:
                    raise RuntimeError("Simulated configure failure")
                return real_run(executable, action, *args, **kwargs)

            failure.setattr(client_config, "_claude_run", fail_configure)
            with pytest.raises(RuntimeError, match="plugin was disabled"):
                client_config.configure_claude_code(vertex)
        installed = json.loads(native("plugin", "list", "--json"))
        assert next(p for p in installed if p["id"] == plugin_id)["enabled"] is False
        client_config.configure_claude_code(vertex)
        installed = json.loads(native("plugin", "list", "--json"))
        assert next(p for p in installed if p["id"] == plugin_id)["enabled"] is True
    finally:
        subprocess.run(
            [
                claude,
                "plugin",
                "uninstall",
                plugin_id,
                "--scope",
                "user",
                "--keep-data",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=tmp_path,
        )
