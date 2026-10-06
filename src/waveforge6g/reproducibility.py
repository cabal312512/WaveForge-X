"""Project-local runtime paths and independent, order-invariant random streams."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__


def project_root() -> Path:
    """Resolve explicit root, checkout root, or current working directory."""
    if os.environ.get("WAVEFORGE_PROJECT_ROOT"):
        return Path(os.environ["WAVEFORGE_PROJECT_ROOT"]).resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


def configure_local_environment() -> Path:
    """Confine runtime cache/temp paths before importing plotting libraries."""
    root = project_root()
    locations = {
        "PIP_CACHE_DIR": ".cache/pip", "MPLCONFIGDIR": ".cache/matplotlib",
        "NUMBA_CACHE_DIR": ".cache/numba", "XDG_CACHE_HOME": ".cache",
        "XDG_CONFIG_HOME": ".cache/config", "XDG_DATA_HOME": ".cache/data",
        "PYTHONPYCACHEPREFIX": ".cache/python", "TMPDIR": ".cache/tmp",
        "TMP": ".cache/tmp", "TEMP": ".cache/tmp",
    }
    for key, relative in locations.items():
        path = root / relative
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    sys.pycache_prefix = str(root / ".cache/python")
    tempfile.tempdir = str(root / ".cache/tmp")
    os.environ["MPLBACKEND"] = "Agg"
    return root


def stable_digest(value: Any) -> str:
    """Stable SHA256 for JSON-serializable metadata."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def trial_rng(seed: int, point: dict[str, Any], trial: int, stream: int) -> np.random.Generator:
    """Streams depend on physical point/trial, never scheduling or waveform order."""
    digest = bytes.fromhex(stable_digest(point))
    words = np.frombuffer(digest, dtype="<u4").astype(int).tolist()
    return np.random.default_rng(np.random.SeedSequence([int(seed), *words, trial, stream]))


def collect_metadata(config_hash: str, seed: int) -> dict[str, Any]:
    """Capture software and Git provenance without requiring a Git checkout."""
    root = project_root()

    def git(*args: str) -> str | None:
        try:
            return subprocess.check_output(["git", "-C", str(root), *args],
                                           stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = git("status", "--porcelain")
    versions = {}
    for package in ("numpy", "scipy", "matplotlib", "pandas", "PyYAML", "tqdm", "numba"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python_version": sys.version, "platform": platform.platform(),
        "project_version": __version__, "dependencies": versions,
        "utc_timestamp": datetime.now(UTC).isoformat(),
        "git_commit": git("rev-parse", "HEAD"),
        "dirty_working_tree": None if status is None else bool(status),
        "random_seed": seed, "configuration_hash": config_hash,
        "noise_policy": "shared unit-variance complex time-domain noise per paired trial",
        "snr_definition": "Es/N0 at transmitter, ensemble useful-symbol Es=1; not received SNR",
        "seed_policy": "SeedSequence(seed, SHA256(physical point), trial, stream)",
        "thread_environment": {key: os.environ.get(key) for key in
                               ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
    }
