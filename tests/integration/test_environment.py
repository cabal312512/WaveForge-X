"""CLI diagnostics and project-local environment behavior in fresh processes."""

import json
import os
import subprocess
import sys
from pathlib import Path


def run_cli(project: Path, *arguments: str) -> subprocess.CompletedProcess:
    environment = os.environ.copy()
    environment["WAVEFORGE_PROJECT_ROOT"] = str(project)
    environment["PYTHONPYCACHEPREFIX"] = str(project / ".cache" / "python")
    return subprocess.run(
        [sys.executable, "-m", "waveforge6g", *arguments],
        cwd=project, env=environment, capture_output=True, text=True, timeout=30,
        check=False,
    )


def test_doctor_reports_local_paths_dependencies_and_headless_backend(tmp_path):
    result = run_cli(tmp_path, "doctor")
    assert result.returncode == 0, result.stderr
    status = json.loads(result.stdout)
    assert status["python_supported"]
    assert status["virtual_environment"]
    assert Path(status["project_root"]).resolve() == tmp_path.resolve()
    assert all(status["dependencies"].values())
    assert all(status["writable_directories"].values())
    assert status["matplotlib_backend"].lower() == "agg"
    for value in status["cache_paths"].values():
        assert Path(value).resolve().is_relative_to(tmp_path.resolve())
    assert isinstance(status["numba_available"], bool)


def test_cli_help_and_invalid_configuration_are_actionable(tmp_path):
    help_result = run_cli(tmp_path, "--help")
    assert help_result.returncode == 0
    assert "doctor" in help_result.stdout
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text("system:\n  frame_size: 1\n", encoding="utf-8")
    result = run_cli(tmp_path, "run", str(bad_config), "--no-plot")
    assert result.returncode == 2
    assert "system.frame_size" in result.stderr
    assert "Traceback" not in result.stderr


def test_validation_cli_checks_all_three_waveforms(tmp_path):
    result = run_cli(tmp_path, "validate")
    assert result.returncode == 0, result.stderr
    status = json.loads(result.stdout)
    for name in ("ofdm", "otfs", "afdm"):
        assert status[f"{name}_roundtrip_max_error"] < 1e-12
