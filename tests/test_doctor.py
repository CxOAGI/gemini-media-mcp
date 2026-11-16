from src.doctor import diagnose
import json
import pytest


def test_offline_diagnostics_never_authenticate_or_echo_keys(tmp_path, monkeypatch):
    (tmp_path / "media").mkdir()
    monkeypatch.setenv("DATA_FOLDER", str(tmp_path / "media"))
    monkeypatch.setenv("GEMINI_API_KEY", "private-test-value")
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "false")

    def forbidden(*args):
        raise AssertionError("Offline doctor must not authenticate")

    monkeypatch.setattr("src.setup_wizard._validate_client", forbidden)
    result = diagnose()
    checks = {row["check"]: row for row in result}
    assert checks["output"]["status"] == "ok"
    assert checks["authentication"]["status"] == "unchecked"
    assert "private-test-value" not in str(result)
    assert not list((tmp_path / "media").iterdir())


def test_doctor_reports_invalid_output_and_missing_credentials(tmp_path, monkeypatch):
    path = tmp_path / "file"
    path.write_text("occupied")
    monkeypatch.setenv("DATA_FOLDER", str(path))
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "false")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    checks = {row["check"]: row for row in diagnose()}
    assert checks["output"]["status"] == "error"
    assert checks["credentials"]["status"] == "error"
    assert path.read_text() == "occupied"


def test_container_requires_explicit_data_folder(monkeypatch):
    monkeypatch.setenv("RUNNING_IN_CONTAINER", "true")
    monkeypatch.delenv("DATA_FOLDER", raising=False)
    checks = {row["check"]: row for row in diagnose()}
    assert checks["output"]["status"] == "error"


def test_pip_or_container_runtime_does_not_require_uvx(tmp_path, monkeypatch):
    monkeypatch.setattr("src.doctor.shutil.which", lambda _: None)
    monkeypatch.setenv("DATA_FOLDER", str(tmp_path))
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "false")
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    assert not [r for r in diagnose() if r["status"] == "error"]


@pytest.mark.parametrize(
    "source", ["GOOGLE_SERVICE_ACCOUNT_JSON", "GOOGLE_APPLICATION_CREDENTIALS"]
)
def test_doctor_infers_inline_project_and_uses_runtime_precedence(
    tmp_path, monkeypatch, source
):
    monkeypatch.setenv("DATA_FOLDER", str(tmp_path))
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_SERVICE_ACCOUNT_JSON", raising=False)
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/not/a/real/file")
    monkeypatch.setenv(
        source, json.dumps({"project_id": "p", "private_key": "dont-print"})
    )
    checks = {row["check"]: row for row in diagnose()}
    assert checks["project"]["status"] == "ok"
    assert checks["credentials"]["status"] == "ok"
    assert "dont-print" not in str(checks)


def test_doctor_rejects_broken_inline_json_without_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_FOLDER", str(tmp_path))
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_JSON", "{broken-secret}")
    path = tmp_path / "adc.json"
    path.write_text('{"project_id":"p"}')
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(path))
    checks = {row["check"]: row for row in diagnose()}
    assert checks["credentials"]["status"] == "error"
    assert "broken-secret" not in str(checks)


@pytest.mark.parametrize("explicit", [False, True])
def test_doctor_never_creates_missing_output_directories(
    tmp_path, monkeypatch, explicit
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RUNNING_IN_CONTAINER", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    if explicit:
        monkeypatch.setenv("DATA_FOLDER", str(tmp_path / "mistyped" / "media"))
    else:
        monkeypatch.delenv("DATA_FOLDER", raising=False)
    checks = {row["check"]: row for row in diagnose()}
    assert checks["output"]["status"] != "ok"
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "vertex,key,backend",
    [(False, "dummy", "vertex"), (True, "", "gemini_api"), (False, "dummy", "unknown")],
)
def test_doctor_rejects_omni_settings_that_startup_rejects(
    tmp_path, monkeypatch, vertex, key, backend
):
    from src.runtime_config import parse_omni_backend_env

    monkeypatch.setenv("DATA_FOLDER", str(tmp_path))
    monkeypatch.setenv("GEMINI_API_KEY", key)
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", str(vertex).lower())
    monkeypatch.setenv("OMNI_BACKEND", backend)
    checks = {row["check"]: row for row in diagnose()}
    with pytest.raises(ValueError) as exc:
        parse_omni_backend_env(primary_is_vertex=vertex, gemini_api_available=bool(key))
    assert checks["omni_backend"]["status"] == "error"
    assert checks["omni_backend"]["detail"] == str(exc.value)
