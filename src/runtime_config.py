"""Runtime configuration rules shared by startup and offline diagnostics."""

import os

OMNI_BACKENDS = ("auto", "vertex", "gemini_api")


def parse_omni_backend_env(
    *, primary_is_vertex: bool, gemini_api_available: bool
) -> str:
    """Reject a backend that the configured credentials cannot serve."""
    raw = os.environ.get("OMNI_BACKEND", "auto").strip().lower() or "auto"
    if raw not in OMNI_BACKENDS:
        raise ValueError(
            f"OMNI_BACKEND={raw!r} is not one of {', '.join(OMNI_BACKENDS)}."
        )
    if raw == "vertex" and not primary_is_vertex:
        raise ValueError(
            "OMNI_BACKEND=vertex, but this server's primary client is not "
            "Vertex AI (GOOGLE_GENAI_USE_VERTEXAI is not true)."
        )
    if raw == "gemini_api" and primary_is_vertex and not gemini_api_available:
        raise ValueError(
            "OMNI_BACKEND=gemini_api, but this Vertex server has no "
            "GEMINI_API_KEY to reach the Gemini Developer API with."
        )
    return raw
