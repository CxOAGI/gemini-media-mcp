"""Local diagnostics; live authentication is explicit and never renders media."""

from __future__ import annotations

import os
import json
import shutil
import sys
import tempfile
from pathlib import Path

from .runtime_config import parse_omni_backend_env


def diagnose(*, live: bool = False) -> list[dict[str, str]]:
    report: list[dict[str, str]] = []

    def add(check: str, status: str, detail: str) -> None:
        report.append({"check": check, "status": status, "detail": detail})

    add(
        "python",
        "ok" if sys.version_info >= (3, 13) else "error",
        sys.version.split()[0],
    )
    uvx = shutil.which("uvx")
    add(
        "uvx",
        "ok" if uvx else "unchecked",
        uvx
        or "uvx is optional for pip and Docker runtimes. Install uv only if your client configuration launches uvx.",
    )
    # Match the server's actual default; setup normally supplies an absolute path.
    try:
        container = (
            os.environ.get("RUNNING_IN_CONTAINER", "").lower() == "true"
            or Path("/.dockerenv").exists()
        )
        if container and not os.environ.get("DATA_FOLDER"):
            raise ValueError(
                "Set DATA_FOLDER to a persistent mounted directory in a container."
            )
        if os.environ.get("DATA_FOLDER", "").startswith("~"):
            raise ValueError(
                "Use an absolute DATA_FOLDER path; the server does not expand ~. Run setup to resolve it."
            )
        folder = Path(os.environ.get("DATA_FOLDER", "data")).resolve()
        if not folder.exists() and not os.environ.get("DATA_FOLDER"):
            add(
                "output",
                "unchecked",
                f"DATA_FOLDER is unset; the server would create {folder}. Set an explicit output folder. No directory was created.",
            )
        elif not folder.is_dir():
            raise ValueError(
                "DATA_FOLDER does not name an existing directory. Check the path or create it explicitly; doctor does not create directories."
            )
        else:
            with tempfile.TemporaryFile(dir=folder):
                pass
            add("output", "ok", str(folder))
    except OSError:
        add("output", "error", "DATA_FOLDER is not writable.")
    except ValueError as exc:
        add("output", "error", str(exc))
    vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() == "true"
    add("backend", "ok", "Vertex AI" if vertex else "Gemini API")
    try:
        omni = parse_omni_backend_env(
            primary_is_vertex=vertex,
            gemini_api_available=bool(os.environ.get("GEMINI_API_KEY")),
        )
        add("omni_backend", "ok", omni)
    except ValueError as exc:
        add("omni_backend", "error", str(exc))
    if vertex:
        from .credentials import parse_service_account_json

        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        inline = None
        try:
            inline = parse_service_account_json()
            if inline is not None:
                project = project or inline.get("project_id")
                add(
                    "credentials",
                    "ok",
                    "Inline service-account JSON selected ahead of any credential file; use --live to authenticate.",
                )
            elif path := os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
                info = json.loads(Path(path).read_text(encoding="utf-8-sig"))
                if not isinstance(info, dict):
                    raise ValueError("Credential file must contain an object.")
                project = project or info.get("project_id")
                add(
                    "credentials",
                    "ok",
                    "Credential file configured; use --live to authenticate.",
                )
            else:
                add(
                    "credentials",
                    "unchecked",
                    "ADC discovery; use --live to authenticate. For local ADC: gcloud auth application-default login.",
                )
        except (ValueError, OSError):
            add(
                "credentials",
                "error",
                "The selected inline JSON or credential file is invalid or unreadable. Fix that source; the server will not fall back to a different identity.",
            )
        add(
            "project",
            "ok" if project else "error" if inline is not None else "unchecked",
            "Project configured."
            if project
            else "Set GOOGLE_CLOUD_PROJECT; inline credentials have no project_id."
            if inline is not None
            else "Project may be discovered through ADC. Set GOOGLE_CLOUD_PROJECT explicitly or use --live to check discovery.",
        )
    else:
        add(
            "credentials",
            "ok" if os.environ.get("GEMINI_API_KEY") else "error",
            "API key configured (not printed)."
            if os.environ.get("GEMINI_API_KEY")
            else "Set GEMINI_API_KEY or run setup.",
        )
    if live:
        from .setup_wizard import _validate_client

        ok, error = _validate_client(dict(os.environ))
        add(
            "authentication",
            "ok" if ok else "error",
            "Model listing succeeded; generation permissions and quota remain untested."
            if ok
            else error or "Authentication check failed.",
        )
    else:
        add(
            "authentication",
            "unchecked",
            "Offline checks only. Run doctor --live for a model-list check.",
        )
    return report
