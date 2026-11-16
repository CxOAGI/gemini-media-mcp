"""Interactive setup wizard for gemini-media-mcp.

Walks the user through credential selection (Gemini API or Vertex AI),
output directory and optional GCS configuration, validates the credentials,
and installs or exports configuration for the selected client.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import getpass
from pathlib import Path
from typing import Any

from .client_config import (
    CLIENTS,
    build_config,
    checked_path,
    configure_claude_code,
    default_config_path,
    preflight_config,
    merge_config,
    redacted,
    write_private,
)
from .credentials import isolated_google_env

__all__ = ["run_wizard"]


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
    if not candidate:
        raise ValueError("No service account JSON provided.")
    if not candidate.startswith("{"):
        expanded = Path(candidate).expanduser().resolve()
        if not expanded.is_file():
            raise FileNotFoundError(f"Service account file not found: {expanded}")
        try:
            data = json.loads(expanded.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Service account file is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("Service-account JSON must be an object.")
        return data, str(expanded)

    # Treat as an inline JSON paste, possibly spanning multiple lines.
    buffer = [candidate]
    if not candidate.rstrip().endswith("}"):
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
    try:
        data = json.loads(blob)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Pasted text is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Service-account JSON must be an object.")
    return data, None


def _validation_error(exc: Exception) -> str:
    """Give actionable guidance without echoing provider errors containing keys."""
    from google.auth.exceptions import DefaultCredentialsError, RefreshError
    from .credentials import CredentialConfigurationError

    if isinstance(exc, CredentialConfigurationError):
        return str(exc)  # Our own messages contain no credential values.
    if isinstance(exc, DefaultCredentialsError):
        return "Application Default Credentials are missing or unusable. Run gcloud auth application-default login, or select a valid --sa-path file."
    if isinstance(exc, RefreshError):
        return "Google could not refresh these credentials. Log in again with gcloud auth application-default login, or replace the service-account key."
    code = getattr(exc, "code", None)
    if code == 401:
        return "Google rejected the credentials (401). Check the selected API key or renew your ADC login."
    if code == 403:
        return "Google denied model-list access (403). Check the project, enabled Vertex AI API, API-key restrictions and IAM permissions. Some generation-only accounts lack model-list permission; continue only if that restriction is expected."
    if code == 429:
        return "Google rate-limited the check (429). Check project quota and billing, then retry later."
    if code == 400:
        return "Google rejected the request (400). An invalid Gemini API key can cause this: check the key in Google AI Studio and its restrictions. For Vertex AI, also check the project, location and enabled API."
    if code == 404:
        return "Google rejected the request. Check the selected backend, project and location."
    if isinstance(code, int) and code >= 500:
        return "Google's service could not complete the check. Retry later."
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)) or type(
        exc
    ).__name__ in {"ConnectError", "ConnectTimeout", "ReadTimeout"}:
        return "Could not reach Google. Check your network, proxy and firewall, then retry."
    return f"Credential check failed ({type(exc).__name__}). Check the selected credentials, project and network; provider details are withheld to protect secrets."


def _validate_client(env: dict[str, str]) -> tuple[bool, str | None]:
    """Validate the collected credentials against the live GenAI backend.

    Constructs a ``genai.Client`` from the collected env, then performs a
    cheap live call (``client.models.list()``, consuming one item) to prove
    the credentials actually authenticate. Client construction alone is lazy
    and would happily accept an invalid API key, so the live call is what
    makes this validation meaningful.

    Returns (ok, error_message). Imports are done lazily so that tests that
    do not touch validation do not need to stub genai. Any failure -- import,
    construction, auth/permission error, or being offline -- is reported as
    (False, message); the wizard stops unless the user explicitly continues.
    """
    try:
        from google import genai  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - environment dependent
        return (
            False,
            f"Could not import google.genai ({type(exc).__name__}). Reinstall gemini-media-mcp in this Python environment.",
        )

    saved: dict[str, str | None] = {}
    client: Any = None
    env = isolated_google_env(env)
    try:
        # Match the selected backend even if the shell has another account set.
        for key in env:
            saved[key] = os.environ.get(key)
            if env.get(key):
                os.environ[key] = env[key]
            else:
                os.environ.pop(key, None)
        # Bound the live check so the wizard can't hang forever on a network
        # that silently drops packets — a timeout falls through to the same
        # soft-fail / "continue anyway" path as any other validation error.
        try:
            from google.genai import types as genai_types

            http_options = genai_types.HttpOptions(timeout=30_000)  # ms
        except Exception:  # pragma: no cover - very old SDK
            http_options = None
        try:
            if env.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() == "true":
                from .credentials import vertex_client_kwargs

                client = genai.Client(
                    **vertex_client_kwargs(
                        cache_credentials=False, http_options=http_options
                    )
                )
            elif env.get("GEMINI_API_KEY"):
                client = genai.Client(
                    vertexai=False,
                    api_key=env["GEMINI_API_KEY"],
                    http_options=http_options,
                )
            else:
                return False, "No credentials in env to validate."
        except Exception as exc:
            return False, _validation_error(exc)

        # Client construction is lazy and never touches the network, so an
        # invalid key would pass. Make a cheap live call and consume a single
        # item to force real authentication.
        try:
            for _ in client.models.list():
                break
        except Exception as exc:
            return False, _validation_error(exc)
        return True, None
    finally:
        try:
            if client is not None and hasattr(client, "close"):
                client.close()
        finally:
            for key, prior in saved.items():
                if prior is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = prior


def _collect_gemini_env(
    interactive: bool,
    overrides: dict[str, Any],
    input_fn: Any,
) -> dict[str, str]:
    """Collect env vars for the Gemini API flow."""
    api_key = overrides.get("api_key")
    if api_key is not None and not str(api_key).strip():
        raise ValueError("The supplied api_key is empty.")
    if api_key is None and os.environ.get("GEMINI_API_KEY"):
        api_key = os.environ["GEMINI_API_KEY"]
        print(
            "Using GEMINI_API_KEY from the environment. Installation/export will save it in the selected client's credential configuration."
        )
    if not api_key:
        if not interactive:
            raise ValueError("Non-interactive Gemini mode requires api_key override.")
        print("Get a Gemini API key at: https://aistudio.google.com/apikey")
        if not overrides.get("secret_input_fn") and not sys.stdin.isatty():
            raise ValueError(
                "Run setup in an interactive terminal for hidden key entry, or set GEMINI_API_KEY in the environment."
            )
        secret_input = overrides.get("secret_input_fn") or getpass.getpass
        api_key = secret_input("Gemini API key (hidden): ").strip()
    if not api_key:
        raise ValueError("Gemini API key is required.")
    return {"GEMINI_API_KEY": api_key}


def _collect_vertex_env(
    interactive: bool,
    overrides: dict[str, Any],
    input_fn: Any,
) -> dict[str, str]:
    """Collect env vars for the Vertex AI flow."""
    sa_path_override = overrides.get("sa_path")
    sa_json_override = overrides.get("sa_json")
    for name, value in (("sa_path", sa_path_override), ("sa_json", sa_json_override)):
        if value is not None and (
            not value or (isinstance(value, str) and not value.strip())
        ):
            raise ValueError(
                f"The supplied {name} is empty. Choose --adc explicitly to use Application Default Credentials."
            )
    if (
        sum(
            (
                bool(overrides.get("adc")),
                sa_path_override is not None,
                sa_json_override is not None,
            )
        )
        > 1
    ):
        raise ValueError(
            "Choose --adc, --sa-path, or --sa-json, not multiple credential sources."
        )
    if not interactive and not (
        overrides.get("adc")
        or sa_path_override is not None
        or sa_json_override is not None
    ):
        raise ValueError(
            "Non-interactive Vertex mode requires --adc, --sa-path, or --sa-json. Ambient credentials are not selected implicitly."
        )
    if interactive:
        print("Vertex AI setup. You will need a Google Cloud project with the")
        print("Vertex AI API enabled. Use existing Application Default Credentials")
        print("or first run: gcloud auth application-default login")
        print("  Cloud Console:     https://console.cloud.google.com")
        print("  IAM > Service Accounts:")
        print("    https://console.cloud.google.com/iam-admin/serviceaccounts")

    project_id = overrides.get("project_id")
    if not project_id and interactive:
        project_id = _prompt(
            "Google Cloud project ID",
            default=os.environ.get("GOOGLE_CLOUD_PROJECT"),
            input_fn=input_fn,
        )
    elif not project_id:
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        if not interactive:
            raise ValueError(
                "Non-interactive Vertex mode requires project_id override."
            )
    if not project_id:
        raise ValueError("Project ID is required for Vertex AI.")

    location = overrides.get("location")
    if not location:
        inherited_location = os.environ.get("GOOGLE_CLOUD_LOCATION") or "us-central1"
        location = (
            _prompt(
                "Google Cloud location", default=inherited_location, input_fn=input_fn
            )
            if interactive
            else inherited_location
        )

    if not overrides.get("adc") and not (sa_path_override or sa_json_override):
        from .credentials import inline_service_account_json

        inline = inline_service_account_json()
        ambient_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if inline and overrides.get("client") == "claude-code":
            print(
                "Claude Code uses a credential file or ADC; inherited inline service-account JSON will not be reused."
            )
        elif inline and _prompt_yes_no(
            f"Reuse {inline[1]} from the environment? This embeds its private key in the client configuration",
            input_fn=input_fn,
        ):
            sa_json_override = inline[0]
        elif (
            not inline
            and ambient_path
            and _prompt_yes_no(
                "Reuse the GOOGLE_APPLICATION_CREDENTIALS file from the environment?",
                input_fn=input_fn,
            )
        ):
            sa_path_override = ambient_path
    if sa_path_override and str(sa_path_override).lstrip().startswith("{"):
        sa_json_override, sa_path_override = sa_path_override, None
    sa_data: dict[str, Any] | None = None
    sa_file_path: str | None = None

    if sa_path_override:
        expanded = Path(str(sa_path_override)).expanduser().resolve()
        if not expanded.is_file():
            raise FileNotFoundError(f"Service account file not found: {expanded}")
        sa_data, sa_file_path = _read_sa_json(str(expanded), input_fn=input_fn)
    elif sa_json_override:
        if isinstance(sa_json_override, dict):
            sa_data = sa_json_override
        else:
            sa_data = json.loads(str(sa_json_override))
        # JSON null is an explicit invalid credential, not a request for ADC.
        if not isinstance(sa_data, dict):
            raise ValueError("Service-account JSON must be an object.")
    elif interactive and not overrides.get("adc"):
        raw = _prompt(
            "Service account file (leave blank for Application Default Credentials)",
            input_fn=input_fn,
        )
        if raw:
            if overrides.get("client") == "claude-code" and raw.lstrip().startswith(
                "{"
            ):
                raise ValueError(
                    "Claude Code requires a service-account file or ADC; inline JSON cannot be installed or exported as native plugin options."
                )
            sa_data, sa_file_path = _read_sa_json(raw, input_fn=input_fn)

    env: dict[str, str] = {
        "GOOGLE_GENAI_USE_VERTEXAI": "true",
        "GOOGLE_CLOUD_PROJECT": project_id,
        "GOOGLE_CLOUD_LOCATION": location,
    }
    if sa_file_path:
        env["GOOGLE_APPLICATION_CREDENTIALS"] = sa_file_path
        print(
            "Selected credentials: service-account file (absolute path will be used)."
        )
    elif sa_data is not None:
        if not isinstance(sa_data, dict):
            raise ValueError("Service-account JSON must be an object.")
        env["GOOGLE_SERVICE_ACCOUNT_JSON"] = json.dumps(sa_data)
        print(
            "Selected credentials: inline service-account JSON (private key will be saved on install/export)."
        )
    else:
        print("Selected credentials: Application Default Credentials (ADC).")
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
        interactive: When False, values must come from overrides or environment.
        input_fn: Injectable input callable for testing.
        print_fn: Injectable print callable for testing.
        write_config: Force-enable or disable client installation.
            When None (default), offer to install in interactive mode.
        **overrides: Pre-filled values. Recognized keys:
            mode ("gemini"|"vertex"), api_key, project_id, location,
            sa_path, sa_json, data_folder, video_gcs_bucket,
            populate_allowed_buckets (bool), continue_on_validation_error (bool).

    Returns:
        The selected client's configuration dict (contains credentials).
    """
    _input = input_fn if input_fn is not None else input
    # Injectable prompt doubles remain useful without exposing terminal secrets.
    if input_fn is not None and "secret_input_fn" not in overrides:
        overrides["secret_input_fn"] = input_fn
    client = overrides.get("client", "claude-desktop")

    def _print(*args: Any, **kwargs: Any) -> None:
        if print_fn is not None:
            print_fn(*args, **kwargs)
        else:
            print(*args, **kwargs)

    # Resolve installation availability before asking for any credentials.
    if client not in CLIENTS:
        raise ValueError(f"Unknown client: {client}")
    if client == "claude-code" and (
        overrides.get("sa_json") is not None
        or str(overrides.get("sa_path", "")).lstrip().startswith("{")
    ):
        raise ValueError(
            "Claude Code setup accepts --adc or --sa-path; inline JSON cannot be installed or exported as native plugin options."
        )
    output = overrides.get("output")
    target: Path | None = None
    if output:
        target = checked_path(output)
    elif write_config or (interactive and write_config is None):
        try:
            if client == "claude-code":
                if not shutil.which("claude"):
                    raise ValueError(
                        "Install Claude Code and ensure 'claude' is on PATH first."
                    )
            else:
                destination = overrides.get("config_path") or default_config_path(
                    client
                )
        except ValueError as exc:
            if write_config:
                raise
            _print(str(exc))
            output = _prompt(
                "Export configuration to",
                default="~/gemini-media-config.json",
                input_fn=_input,
            )
            target = checked_path(output)
        else:
            if client != "claude-code":
                target = checked_path(destination)
                preflight_config(target, client)

    # Step 1: mode
    mode = overrides.get("mode")
    if mode is None:
        if not interactive:
            raise ValueError("Non-interactive mode requires 'mode' override.")
        choice = _prompt_choice(
            "Which credential mode?",
            {
                "g": "Gemini API (images + video)",
                "v": "Vertex AI (images + video, GCS output)",
            },
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
    env["DATA_FOLDER"] = str(
        Path(str(data_folder or "~/gemini-media")).expanduser().resolve()
    )
    env["OMNI_BACKEND"] = "vertex" if mode == "vertex" else "gemini_api"
    # Prevent inherited credentials from silently changing the chosen backend.
    env = isolated_google_env(env)

    # Step 5: optional VIDEO_GCS_BUCKET
    video_gcs_bucket = overrides.get("video_gcs_bucket")
    if video_gcs_bucket is None and interactive and mode == "vertex":
        video_gcs_bucket = _prompt(
            "Optional VIDEO_GCS_BUCKET (gs://...), leave blank to skip",
            default="",
            input_fn=_input,
        )
    if video_gcs_bucket:
        env["VIDEO_GCS_BUCKET"] = str(video_gcs_bucket)
        populate = overrides.get("populate_allowed_buckets")
        if populate is None and interactive and client != "claude-code":
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
    ok, error = (
        (True, None) if overrides.get("skip_validation") else _validate_client(env)
    )
    if not ok:
        _print(f"Credential validation failed: {error}")
        cont = overrides.get("continue_on_validation_error")
        if cont is None and interactive:
            cont = _prompt_yes_no(
                "Continue anyway and finish writing the config?",
                default=False,
                input_fn=_input,
            )
        if not cont:
            raise RuntimeError(f"Setup aborted: {error}")
    else:
        _print(
            "Credential validation skipped."
            if overrides.get("skip_validation")
            else "Credentials validated successfully (model listing; generation access and quota are not tested)."
        )

    # Step 7/8: build and print config
    config = build_config(client, env)
    _print("")
    _print(
        "Claude Code native plugin options for gemini-media@cxoagi (secrets redacted):"
        if client == "claude-code"
        else f"{client} configuration preview (secrets redacted):"
    )
    _print(json.dumps(redacted(config), indent=2))

    if output:
        assert target is not None
        if client == "codex":
            import tomlkit

            write_private(target, tomlkit.dumps(config))
        else:
            write_private(target, json.dumps(config, indent=2) + "\n")
        if client == "claude-code":
            _print(
                f"Exported private plugin options to {target}. Supply the file to claude plugin configure gemini-media@cxoagi --values-stdin after installing. To clear a saved API key, use setup --write; Claude ignores empty sensitive values."
            )
        else:
            _print(
                f"Exported private config to {target}. Merge it with your client settings."
            )
        return config

    # Step 9: client installation
    should_write = write_config
    if should_write is None and interactive:
        should_write = _prompt_yes_no(
            f"Install into {client}?",
            default=True,
            input_fn=_input,
        )
    if should_write:
        if client == "claude-code":
            configure_claude_code(env)
            _print("Configured Claude Code. Reload plugins, then check /mcp.")
            return config
        assert target is not None
        backup = merge_config(target, config, toml=client == "codex")
        _print(f"Wrote config to {target}")
        if backup:
            _print(f"Backed up prior config to {backup}")
    else:
        _print(
            "Preview only. Use --write to install, or --output PATH to export a usable private config."
        )

    return config
