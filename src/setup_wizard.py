"""Interactive setup wizard for gemini-media-mcp.

Walks the user through credential selection (Gemini API or Vertex AI),
output directory, optional GCS configuration, validates the credentials,
and emits a Claude Desktop configuration block.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

__all__ = ["run_wizard", "build_claude_config", "macos_config_path"]


def _prompt(
    message: str,
    default: str | None = None,
    input_fn: Any = input,
) -> str:
    """Prompt the user for a value, returning default on empty input."""
    suffix = f" [{default}]" if default else ""
    raw = input_fn(f"{message}{suffix}: ")
    value = raw.strip()
    if not value and default is not None:
        return default
    return value


def _prompt_yes_no(
    message: str,
    default: bool = False,
    input_fn: Any = input,
) -> bool:
    """Prompt the user for a yes/no answer."""
    suffix = "Y/n" if default else "y/N"
    raw = input_fn(f"{message} [{suffix}]: ").strip().lower()
    if not raw:
        return default
    return raw.startswith("y")


def _prompt_choice(
    message: str,
    choices: dict[str, str],
    input_fn: Any = input,
) -> str:
    """Prompt for one of a set of single-letter choices."""
    valid = set(choices)
    while True:
        parts = " / ".join(f"[{key}] {label}" for key, label in choices.items())
        raw = input_fn(f"{message} ({parts}): ").strip().lower()
        if raw in valid:
            return raw
        print(f"Please enter one of: {', '.join(sorted(valid))}")


def _read_sa_json(
    path_or_paste: str,
    input_fn: Any = input,
) -> tuple[dict[str, Any], str | None]:
    """Resolve a service-account JSON from a file path or inline paste.

    Returns a tuple of (parsed dict, file path or None). The file path is
    returned when the user supplied a path on disk that parsed successfully.
    """
    candidate = path_or_paste.strip()
    if candidate and not candidate.startswith("{"):
        expanded = Path(candidate).expanduser()
        if not expanded.is_file():
            raise FileNotFoundError(f"Service account file not found: {expanded}")
        try:
            data = json.loads(expanded.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Service account file is not valid JSON: {exc}") from exc
        return data, str(expanded)

    # Treat as an inline JSON paste, possibly spanning multiple lines.
    buffer = [candidate] if candidate else []
    if not candidate or not candidate.rstrip().endswith("}"):
        print("Paste the service account JSON. End with a blank line:")
        while True:
            try:
                line = input_fn("")
            except EOFError:
                break
            if line == "" and buffer:
                break
            buffer.append(line)
    blob = "\n".join(buffer).strip()
    if not blob:
        raise ValueError("No service account JSON provided.")
    try:
        data = json.loads(blob)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Pasted text is not valid JSON: {exc}") from exc
    return data, None


def _validate_client(env: dict[str, str]) -> tuple[bool, str | None]:
    """Attempt to construct a genai.Client using the collected env.

    Returns (ok, error_message). Imports are done lazily so that tests that
    do not touch validation do not need to stub genai.
    """
    try:
        from google import genai  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, f"Failed to import google.genai: {exc}"

    import os

    saved: dict[str, str | None] = {}
    try:
        for key, value in env.items():
            saved[key] = os.environ.get(key)
            os.environ[key] = value
        try:
            if env.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() == "true":
                genai.Client(vertexai=True)
            elif env.get("GEMINI_API_KEY"):
                genai.Client(api_key=env["GEMINI_API_KEY"])
            else:
                return False, "No credentials in env to validate."
        except Exception as exc:
            return False, str(exc)
        return True, None
    finally:
        for key, prior in saved.items():
            if prior is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prior


def build_claude_config(env: dict[str, str]) -> dict[str, Any]:
    """Build the Claude Desktop mcpServers block for gemini-media."""
    return {
        "mcpServers": {
            "gemini-media": {
                "command": "uvx",
                "args": ["gemini-media-mcp"],
                "env": dict(env),
            }
        }
    }


def macos_config_path() -> Path:
    """Return the default Claude Desktop config path on macOS."""
    return (
        Path.home()
        / "Library"
        / "Application Support"
        / "Claude"
        / "claude_desktop_config.json"
    )


def _merge_and_write(
    config_path: Path,
    new_block: dict[str, Any],
) -> Path | None:
    """Merge mcpServers from new_block into config_path, backing up the old file.

    Returns the backup path if one was created, otherwise None.
    """
    backup: Path | None = None
    existing: dict[str, Any] = {}
    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text() or "{}")
        except json.JSONDecodeError:
            existing = {}
        backup = config_path.with_suffix(config_path.suffix + ".bak")
        shutil.copy2(config_path, backup)

    servers = dict(existing.get("mcpServers") or {})
    servers.update(new_block.get("mcpServers") or {})
    existing["mcpServers"] = servers

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(existing, indent=2) + "\n")
    return backup


def _collect_gemini_env(
    interactive: bool,
    overrides: dict[str, Any],
    input_fn: Any,
) -> dict[str, str]:
    """Collect env vars for the Gemini API flow."""
    api_key = overrides.get("api_key")
    if not api_key:
        if not interactive:
            raise ValueError("Non-interactive Gemini mode requires api_key override.")
        print("Get a Gemini API key at: https://aistudio.google.com/apikey")
        api_key = _prompt("Gemini API key", input_fn=input_fn)
    if not api_key:
        raise ValueError("Gemini API key is required.")
    return {"GEMINI_API_KEY": api_key}


def _collect_vertex_env(
    interactive: bool,
    overrides: dict[str, Any],
    input_fn: Any,
) -> dict[str, str]:
    """Collect env vars for the Vertex AI flow."""
    if interactive:
        print("Vertex AI setup. You will need a Google Cloud project with the")
        print("Vertex AI API enabled and a service account key.")
        print("  Cloud Console:     https://console.cloud.google.com")
        print("  IAM > Service Accounts:")
        print("    https://console.cloud.google.com/iam-admin/serviceaccounts")

    project_id = overrides.get("project_id")
    if not project_id:
        if not interactive:
            raise ValueError(
                "Non-interactive Vertex mode requires project_id override."
            )
        project_id = _prompt("Google Cloud project ID", input_fn=input_fn)
    if not project_id:
        raise ValueError("Project ID is required for Vertex AI.")

    location = overrides.get("location")
    if not location:
        if interactive:
            location = _prompt(
                "Google Cloud location",
                default="us-central1",
                input_fn=input_fn,
            )
        else:
            location = "us-central1"

    sa_path_override = overrides.get("sa_path")
    sa_json_override = overrides.get("sa_json")
    sa_data: dict[str, Any] | None = None
    sa_file_path: str | None = None

    if sa_path_override:
        expanded = Path(str(sa_path_override)).expanduser()
        if not expanded.is_file():
            raise FileNotFoundError(f"Service account file not found: {expanded}")
        sa_data = json.loads(expanded.read_text())
        sa_file_path = str(expanded)
    elif sa_json_override:
        if isinstance(sa_json_override, dict):
            sa_data = sa_json_override
        else:
            sa_data = json.loads(str(sa_json_override))
    elif interactive:
        raw = _prompt(
            "Service account: path to JSON file, or leave blank to paste",
            input_fn=input_fn,
        )
        sa_data, sa_file_path = _read_sa_json(raw, input_fn=input_fn)
    else:
        raise ValueError(
            "Non-interactive Vertex mode requires sa_path or sa_json override."
        )

    env: dict[str, str] = {
        "GOOGLE_GENAI_USE_VERTEXAI": "true",
        "GOOGLE_CLOUD_PROJECT": project_id,
        "GOOGLE_CLOUD_LOCATION": location,
    }
    if sa_file_path:
        env["GOOGLE_APPLICATION_CREDENTIALS"] = sa_file_path
    else:
        assert sa_data is not None
        env["GOOGLE_SERVICE_ACCOUNT_JSON"] = json.dumps(sa_data)
    return env


def run_wizard(
    interactive: bool = True,
    *,
    input_fn: Any = None,
    print_fn: Any = None,
    write_config: bool | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Run the interactive setup wizard.

    Args:
        interactive: When False, values must come from overrides.
        input_fn: Injectable input callable for testing.
        print_fn: Injectable print callable for testing.
        write_config: Force-enable or disable writing to the macOS config file.
            When None (default), offer to write in interactive mode on macOS.
        **overrides: Pre-filled values. Recognized keys:
            mode ("gemini"|"vertex"), api_key, project_id, location,
            sa_path, sa_json, data_folder, video_gcs_bucket,
            populate_allowed_buckets (bool), continue_on_validation_error (bool).

    Returns:
        The Claude Desktop config dict that was produced.
    """
    _input = input_fn if input_fn is not None else input

    def _print(*args: Any, **kwargs: Any) -> None:
        if print_fn is not None:
            print_fn(*args, **kwargs)
        else:
            print(*args, **kwargs)

    # Step 1: mode
    mode = overrides.get("mode")
    if mode is None:
        if not interactive:
            raise ValueError("Non-interactive mode requires 'mode' override.")
        choice = _prompt_choice(
            "Which credential mode?",
            {"g": "Gemini API (images only)", "v": "Vertex AI (images + video)"},
            input_fn=_input,
        )
        mode = "gemini" if choice == "g" else "vertex"
    mode = str(mode).lower()
    if mode in {"g", "gemini", "gemini-api"}:
        mode = "gemini"
    elif mode in {"v", "vertex", "vertex-ai", "vertexai"}:
        mode = "vertex"
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    # Step 2/3: collect credentials
    if mode == "gemini":
        env = _collect_gemini_env(interactive, overrides, _input)
    else:
        env = _collect_vertex_env(interactive, overrides, _input)

    # Step 4: data folder
    data_folder = overrides.get("data_folder")
    if data_folder is None and interactive:
        data_folder = _prompt(
            "Output folder for generated media",
            default="~/gemini-media",
            input_fn=_input,
        )
    if data_folder:
        env["DATA_FOLDER"] = str(Path(str(data_folder)).expanduser())

    # Step 5: optional VIDEO_GCS_BUCKET
    video_gcs_bucket = overrides.get("video_gcs_bucket")
    if video_gcs_bucket is None and interactive:
        video_gcs_bucket = _prompt(
            "Optional VIDEO_GCS_BUCKET (gs://...), leave blank to skip",
            default="",
            input_fn=_input,
        )
    if video_gcs_bucket:
        env["VIDEO_GCS_BUCKET"] = str(video_gcs_bucket)
        populate = overrides.get("populate_allowed_buckets")
        if populate is None and interactive:
            populate = _prompt_yes_no(
                "Populate GCS_ALLOWED_BUCKETS from this bucket?",
                default=True,
                input_fn=_input,
            )
        if populate:
            bucket_uri = str(video_gcs_bucket)
            bucket_name = (
                bucket_uri[5:] if bucket_uri.startswith("gs://") else bucket_uri
            )
            bucket_name = bucket_name.split("/", 1)[0]
            env["GCS_ALLOWED_BUCKETS"] = bucket_name

    # Step 6: validate credentials
    ok, error = _validate_client(env)
    if not ok:
        _print(f"Credential validation failed: {error}")
        cont = overrides.get("continue_on_validation_error")
        if cont is None and interactive:
            cont = _prompt_yes_no(
                "Continue anyway and finish writing the config?",
                default=True,
                input_fn=_input,
            )
        if not cont:
            raise RuntimeError(f"Setup aborted: {error}")
    else:
        _print("Credentials validated successfully.")

    # Step 7/8: build and print config
    config = build_claude_config(env)
    _print("")
    _print("Paste the following into your Claude Desktop config:")
    _print(json.dumps(config, indent=2))

    # Step 9: macOS auto-write
    should_write = write_config
    if should_write is None and interactive and sys.platform == "darwin":
        should_write = _prompt_yes_no(
            "Write this directly to your Claude Desktop config?",
            default=False,
            input_fn=_input,
        )
    if should_write:
        target = overrides.get("config_path") or macos_config_path()
        target = Path(target)
        backup = _merge_and_write(target, config)
        _print(f"Wrote config to {target}")
        if backup:
            _print(f"Backed up prior config to {backup}")

    return config
