"""Local MCP client adapters. No credentials are passed on command lines."""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from importlib import metadata
from typing import Any

CLIENTS = (
    "claude-desktop",
    "claude-code",
    "codex",
    "cursor",
    "vscode",
    "copilot",
    "gemini-cli",
    "cline",
)
MARKETPLACE = "https://github.com/CxOAGI/gemini-media-mcp"
CLAUDE_PLUGIN = "gemini-media@cxoagi"
SECRET_KEYS = frozenset(
    {
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GOOGLE_SERVICE_ACCOUNT_JSON",
        "gemini_api_key",
    }
)


def server_package() -> str:
    """Released wizards install their own version; checkouts use the public pin."""
    release = json.loads(Path(__file__).with_name("client_release.json").read_text())
    try:
        installed = metadata.version("gemini-media-mcp")
    except metadata.PackageNotFoundError:
        installed = ""
    public = release["version"]
    version = public
    if re.fullmatch(r"\d+\.\d+\.\d+", installed) and tuple(
        map(int, installed.split("."))
    ) >= tuple(map(int, public.split("."))):
        version = installed
    return f"gemini-media-mcp=={version}"


def redacted(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: "<redacted>" if k in SECRET_KEYS and v else redacted(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [redacted(v) for v in value]
    return value


def build_config(client: str, env: dict[str, str]) -> dict[str, Any]:
    if client not in CLIENTS:
        raise ValueError(f"Unknown client: {client}")
    if client == "claude-code":
        return claude_config_values(env)
    server: dict[str, Any] = {
        "command": shutil.which("uvx") or "uvx",
        "args": [
            "--python",
            "3.13",
            "--from",
            server_package(),
            "gemini-media-mcp",
            "stdio",
        ],
        "env": dict(env),
    }
    if client == "codex":
        server.update(startup_timeout_sec=120, tool_timeout_sec=1800)
        return {"mcp_servers": {"gemini-media": server}}
    if client == "gemini-cli":
        server["timeout"] = 1_800_000  # Gemini CLI uses milliseconds.
    if client == "cline":
        server["timeout"] = 1800  # Cline uses seconds.
    if client == "copilot":
        server.update(type="stdio", tools=["*"])
    if client == "vscode":
        server["command"] = "uvx"
        server["type"] = "stdio"
        block: dict[str, Any] = {"servers": {"gemini-media": server}}
        # Workspace configs travel between machines. Prompt for local paths too.
        inputs = []
        for key in sorted(
            SECRET_KEYS | {"DATA_FOLDER", "GOOGLE_APPLICATION_CREDENTIALS"}
        ):
            if env.get(key):
                name = f"gemini-media-{key.lower()}"
                server["env"][key] = "${input:" + name + "}"
                inputs.append(
                    {
                        "id": name,
                        "type": "promptString",
                        "description": key
                        if key in SECRET_KEYS
                        else f"Absolute path for {key} on this machine",
                        "password": key in SECRET_KEYS,
                    }
                )
        if inputs:
            block["inputs"] = inputs
        return block
    return {"mcpServers": {"gemini-media": server}}


def default_config_path(client: str) -> Path:
    home = Path.home()
    paths = {
        "codex": Path(os.environ.get("CODEX_HOME", str(home / ".codex")))
        / "config.toml",
        "cursor": home / ".cursor" / "mcp.json",
        "gemini-cli": Path(os.environ.get("GEMINI_CLI_HOME") or home)
        / ".gemini"
        / "settings.json",
        "copilot": Path(os.environ.get("COPILOT_HOME", str(home / ".copilot")))
        / "mcp-config.json",
        "cline": Path(
            os.environ.get(
                "CLINE_MCP_SETTINGS_PATH",
                str(home / ".cline/data/settings/cline_mcp_settings.json"),
            )
        ),
    }
    if client == "claude-desktop":
        if sys.platform == "darwin":
            return (
                home / "Library/Application Support/Claude/claude_desktop_config.json"
            )
        if sys.platform == "win32" and os.environ.get("APPDATA"):
            return Path(os.environ["APPDATA"]) / "Claude/claude_desktop_config.json"
        raise ValueError("Specify --config-path for Claude Desktop on this platform.")
    if client == "vscode":
        raise ValueError(
            "Use --config-path /path/to/project/.vscode/mcp.json, or the path from MCP: Open User Configuration."
        )
    return paths[client]


def checked_path(path: str | Path) -> Path:
    """Make a path absolute without dereferencing its final symlink."""
    target = Path(path).expanduser().absolute()
    if target.is_symlink():
        raise ValueError(f"Refusing to replace symlink: {target}")
    if target.exists() and not target.is_file():
        raise ValueError(f"Configuration destination is not a file: {target}")
    return target


def write_private(path: Path, text: str | bytes) -> None:
    """Replace atomically; a partially written config must never strand a client."""
    path = checked_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(text.encode("utf-8") if isinstance(text, str) else text)
        checked_path(path)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_config(path: Path, *, toml: bool, root: str) -> tuple[bytes, Any]:
    """Validate the destination before credential entry and again before writing."""
    path = checked_path(path)
    original_bytes = path.read_bytes() if path.exists() else b""
    original = original_bytes.decode("utf-8-sig")
    if toml:
        import tomlkit

        existing = tomlkit.parse(original)
    else:
        try:
            existing = json.loads(original) if original.strip() else {}
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Existing config is not strict JSON (possibly JSONC). Export with --output and merge it in the client's editor."
            ) from exc
        if not isinstance(existing, dict):
            raise ValueError("Existing config must be an object")
    if not isinstance(existing.get(root, {}), dict):
        raise ValueError(f"{root} must be an object/table")
    if root == "servers":
        prior = existing.get("inputs", [])
        if not isinstance(prior, list) or any(not isinstance(v, dict) for v in prior):
            raise ValueError("inputs must be an array of objects")
    return original_bytes, existing


def preflight_config(path: Path, client: str) -> None:
    root = (
        "mcp_servers"
        if client == "codex"
        else "servers"
        if client == "vscode"
        else "mcpServers"
    )
    _read_config(path, toml=client == "codex", root=root)
    checked_path(path.with_suffix(path.suffix + ".bak"))


def merge_config(
    path: Path, block: dict[str, Any], *, toml: bool = False
) -> Path | None:
    """Preserve unrelated settings and refuse malformed or ambiguous input."""
    path = checked_path(path)
    root = "mcp_servers" if toml else "servers" if "servers" in block else "mcpServers"
    original_bytes, existing = _read_config(path, toml=toml, root=root)
    original = original_bytes.decode("utf-8-sig")
    servers = existing.setdefault(root, {})
    if toml:
        import tomlkit

        expected = tomllib.loads(original)
        expected.setdefault(root, {}).update(block[root])
        # Replacing a split table in place can drop siblings in tomlkit's proxy.
        for name, server in block[root].items():
            if name in servers:
                del servers[name]
            servers[name] = server
        rendered = tomlkit.dumps(existing)
        if tomllib.loads(rendered) != expected:
            raise ValueError(
                "Config merge would change unrelated settings. Export with --output and merge manually; the existing file was not changed."
            )
    else:
        servers.update(copy.deepcopy(block[root]))
        if "inputs" in block:
            prior = existing.get("inputs", [])
            if not isinstance(prior, list) or any(
                not isinstance(v, dict) for v in prior
            ):
                raise ValueError("inputs must be an array of objects")
            ids = {v["id"] for v in block["inputs"]}
            existing["inputs"] = [v for v in prior if v.get("id") not in ids] + block[
                "inputs"
            ]
        rendered = json.dumps(existing, indent=2) + "\n"
    if path.is_symlink():
        raise ValueError(f"Refusing to replace symlink: {path}")
    backup = None
    if path.exists():
        backup = path.with_suffix(path.suffix + ".bak")
        write_private(backup, original_bytes)
    if "\r\n" in original and "\n" not in original.replace("\r\n", ""):
        rendered = rendered.replace("\r\n", "\n").replace("\n", "\r\n")
    if original_bytes.startswith(b"\xef\xbb\xbf"):
        rendered = "\ufeff" + rendered
    write_private(path, rendered)
    return backup


def claude_config_values(env: dict[str, str]) -> dict[str, str]:
    """The same native options are previewed, exported, and installed."""
    if env.get("GOOGLE_SERVICE_ACCOUNT_JSON"):
        raise ValueError(
            "Claude Code setup accepts --adc or --sa-path; use a file instead of inline service-account JSON."
        )
    return {
        "data_folder": env["DATA_FOLDER"],
        "gemini_api_key": env.get("GEMINI_API_KEY", ""),
        "use_vertex": env.get("GOOGLE_GENAI_USE_VERTEXAI", "false"),
        "google_cloud_project": env.get("GOOGLE_CLOUD_PROJECT", ""),
        "google_cloud_location": env.get("GOOGLE_CLOUD_LOCATION") or "us-central1",
        "google_application_credentials": env.get("GOOGLE_APPLICATION_CREDENTIALS", ""),
        "video_gcs_bucket": env.get("VIDEO_GCS_BUCKET", ""),
        "omni_backend": env["OMNI_BACKEND"],
    }


def _claude_run(claude: str, action: str, *args: str, stdin: str | None = None) -> str:
    try:
        result = subprocess.run(
            [claude, "plugin", action, *args],
            input=stdin,
            text=True,
            capture_output=True,
            timeout=180,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"Claude plugin {action} timed out after 180 seconds. Check connectivity and retry setup."
        ) from None
    except OSError:
        raise RuntimeError(
            f"Could not run Claude plugin {action}. Check Claude Code is on PATH and up to date."
        ) from None
    if result.returncode:
        # Native output might contain a supplied value. Do not echo it.
        raise RuntimeError(
            f"Claude plugin {action} failed. Check Claude Code is up to date and the marketplace is reachable; then rerun setup."
        )
    return result.stdout


def _claude_json(claude: str, action: str, *args: str) -> Any:
    output = _claude_run(claude, action, *args, "--json")
    try:
        return json.loads(output)
    except ValueError:
        raise RuntimeError(
            f"Claude plugin {action} returned unreadable status. Update Claude Code and retry."
        ) from None


def _claude_configured(claude: str) -> set[str]:
    status = _claude_json(claude, "configure", CLAUDE_PLUGIN)
    configured = status.get("configured") if isinstance(status, dict) else None
    if not isinstance(configured, list) or any(
        not isinstance(v, str) for v in configured
    ):
        raise RuntimeError(
            "Claude did not report which plugin options are configured. Update Claude Code and retry."
        )
    return set(configured)


def _claude_records(claude: str) -> list[dict[str, Any]]:
    installed = _claude_json(claude, "list")
    if not isinstance(installed, list) or any(
        not isinstance(v, dict) for v in installed
    ):
        raise RuntimeError("Claude returned unreadable plugin installation records.")
    return [v for v in installed if v.get("id") == CLAUDE_PLUGIN]


def configure_claude_code(env: dict[str, str]) -> None:
    claude = shutil.which("claude")
    if not claude:
        raise ValueError("Install Claude Code and ensure 'claude' is on PATH first.")
    values = claude_config_values(env)
    records = _claude_records(claude)
    clear_key = (
        bool(records)
        and not values["gemini_api_key"]
        and "gemini_api_key" in _claude_configured(claude)
    )
    if clear_key and any(v.get("scope") != "user" for v in records):
        raise ValueError(
            "Claude keeps secrets shared across plugin scopes. Remove gemini-media@cxoagi from its other scopes with --keep-data before switching to Vertex-only setup; no configuration was changed."
        )

    try:
        if clear_key:
            # Empty sensitive values are ignored by Claude. Its supported reset
            # removes saved options/secrets only at the last installed scope.
            removed = _claude_json(
                claude, "uninstall", CLAUDE_PLUGIN, "--scope", "user", "--keep-data"
            )
            if (
                not isinstance(removed, dict)
                or removed.get("outcome") != "ok"
                or removed.get("savedKept")
            ):
                raise RuntimeError(
                    "Claude could not confirm removal of the saved plugin credentials. Check its installation records before retrying."
                )
        args = ["gemini-media", "--marketplace", MARKETPLACE, "--scope", "user"]
        for key, value in values.items():
            if key != "gemini_api_key":
                args.extend(["--config", f"{key}={value}"])
        # All non-secret settings, including the required folder, accompany the
        # install. The secret is supplied separately on stdin, never in argv.
        _claude_run(claude, "install", *args)
        _claude_run(
            claude,
            "configure",
            CLAUDE_PLUGIN,
            "--values-stdin",
            stdin=json.dumps(values),
        )
        key_is_set = "gemini_api_key" in _claude_configured(claude)
        if key_is_set != bool(values["gemini_api_key"]):
            raise RuntimeError(
                "Claude did not apply the requested API-key state. The saved credential needs attention before this plugin can run."
            )
        current = [v for v in _claude_records(claude) if v.get("scope") == "user"]
        if len(current) != 1 or not isinstance(current[0].get("enabled"), bool):
            raise RuntimeError(
                "Claude could not confirm the user-scope plugin's enabled state."
            )
        if not current[0]["enabled"]:
            _claude_run(claude, "enable", CLAUDE_PLUGIN, "--scope", "user")
    except RuntimeError as exc:
        try:
            _claude_run(claude, "disable", CLAUDE_PLUGIN, "--scope", "user")
        except RuntimeError:
            raise RuntimeError(
                f"{exc} Could not confirm the user-scope plugin is disabled. Check /plugin and run 'claude plugin disable {CLAUDE_PLUGIN} --scope user' before using it."
            ) from None
        raise RuntimeError(
            f"{exc} The user-scope plugin was disabled to avoid running a partial configuration. Rerun setup to finish and enable it."
        ) from None
