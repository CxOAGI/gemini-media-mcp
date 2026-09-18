"""Vertex AI credentials, built once from inline service-account JSON.

Shared by ``__main__`` and ``image`` because both build Vertex clients and
``image`` cannot import ``__main__`` (under ``python -m src`` that would load
the server module a second time).

Why an object and not a file. The previous design wrote the JSON to
``/tmp/gcp_sa_*.json``, pointed the process-wide GOOGLE_APPLICATION_CREDENTIALS
at it, and deleted the file on lifespan teardown -- and on the sse and
streamable-http transports the MCP SDK enters the lifespan once PER SESSION.
Every connecting client wrote its own file and repointed the shared variable;
every disconnecting client deleted the file every other live session's
lazily-loaded credentials pointed at. A credentials object handed to each
client has no file to delete and mutates no environment.

Two consequences of the first cut of that redesign are handled here too:

* every Vertex client must receive the object. ``image.py``'s Gemini-3 client
  was missed, so on an inline-JSON deployment it fell into Application
  Default Credentials, found nothing, and every default-model image render
  failed;
* nothing may ever reach ``google.auth.default()`` while a path variable holds
  key text. google-auth's error for that is ``File {<the whole JSON, private
  key included>} was not found``, and the tool returned it to the caller. The
  accepted "JSON pasted into GOOGLE_APPLICATION_CREDENTIALS" spelling is moved
  into GOOGLE_SERVICE_ACCOUNT_JSON once, at startup, and the SDK is given a
  project explicitly, since google-genai 2.20.0 calls ``google.auth.default()``
  whenever ``project`` is None regardless of the credentials it was handed.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any

from google.oauth2 import service_account

_SCOPES = ("https://www.googleapis.com/auth/cloud-platform",)

_creds: Any | None = None
_project: str | None = None
_lock = threading.Lock()


def inline_service_account_json() -> tuple[str, str] | None:
    """The inline service-account JSON the operator set, and which variable.

    Two spellings are accepted: GOOGLE_SERVICE_ACCOUNT_JSON, or the JSON pasted
    into GOOGLE_APPLICATION_CREDENTIALS in place of a path (it starts with
    "{"). A path in GOOGLE_APPLICATION_CREDENTIALS is not inline JSON and is
    left to Application Default Credentials.
    """
    sa_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if sa_json:
        return sa_json, "GOOGLE_SERVICE_ACCOUNT_JSON"
    gac = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if gac.strip().startswith("{"):
        return gac, "GOOGLE_APPLICATION_CREDENTIALS"
    return None


def normalize_environment() -> None:
    """Move key text out of the path variable, once.

    GOOGLE_APPLICATION_CREDENTIALS is read by google-auth as a FILE PATH. When
    it holds JSON instead, any code that reaches Application Default
    Credentials raises an error whose message is the variable's entire value
    -- the private key included -- and that message was being returned to the
    MCP caller. The JSON is kept, under the variable that means "inline", and
    the path variable is cleared. Idempotent; a real path is left alone.
    """
    gac = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if not gac.strip().startswith("{"):
        return
    os.environ.setdefault("GOOGLE_SERVICE_ACCOUNT_JSON", gac)
    del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]


def parse_service_account_json() -> dict[str, Any] | None:
    """Parse the inline service-account JSON, failing loudly if it is broken.

    A configured service-account JSON that does not parse must fail. Falling
    through to ``genai.Client(vertexai=True)`` discovered whatever ambient ADC
    happened to exist -- so a typo'd credential silently ran the server as a
    DIFFERENT identity than configured, with one log line as the only signal.
    """
    inline = inline_service_account_json()
    if inline is None:
        return None
    sa_json, source = inline
    try:
        data = json.loads(sa_json)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"{source} is set but is not valid JSON: {e}. "
            "Refusing to fall back to ambient application-default "
            "credentials — that would run the server as a different "
            "identity than configured. Fix the JSON or unset the variable."
        ) from e
    if not isinstance(data, dict):
        raise ValueError(f"{source} must be a JSON object, got {type(data).__name__}.")
    return data


def service_account_credentials() -> tuple[Any, str | None] | None:
    """Process-wide credentials from the inline JSON, or None for ADC.

    Returns ``(credentials, project_id)``, memoised for the life of the
    process; every lifespan and every client sees the one object.
    """
    global _creds, _project
    with _lock:
        if _creds is not None:
            return _creds, _project
        info = parse_service_account_json()
        if info is None:
            return None
        normalize_environment()
        try:
            creds = service_account.Credentials.from_service_account_info(
                info, scopes=list(_SCOPES)
            )
        except (ValueError, KeyError) as e:
            raise ValueError(
                "The inline service-account JSON parsed but is not a usable "
                f"service-account key: {e}."
            ) from e
        _creds = creds
        _project = os.environ.get("GOOGLE_CLOUD_PROJECT") or info.get("project_id") or None
        return creds, _project


def vertex_client_kwargs(**extra: Any) -> dict[str, Any]:
    """Keyword arguments for a Vertex ``genai.Client`` on this deployment.

    With inline credentials the project is always passed explicitly: google-genai
    2.20.0 calls ``google.auth.default()`` whenever ``project`` is None, whatever
    ``credentials`` it was given, so a key without ``project_id`` and no
    GOOGLE_CLOUD_PROJECT tumbled into ADC discovery instead of saying what was
    missing.
    """
    kwargs: dict[str, Any] = {"vertexai": True, **extra}
    found = service_account_credentials()
    if found is not None:
        creds, project = found
        if not project:
            raise ValueError(
                "The inline service-account JSON has no project_id and "
                "GOOGLE_CLOUD_PROJECT is unset, so the Vertex project is unknown. "
                "Set GOOGLE_CLOUD_PROJECT."
            )
        kwargs["credentials"] = creds
        kwargs.setdefault("project", project)
    return kwargs
