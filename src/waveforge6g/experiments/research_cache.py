"""Content-keyed local research cache with code-change invalidation."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from uuid import uuid4

from .. import __version__
from ..config import ExperimentConfig
from ..reproducibility import collect_metadata, project_root, stable_digest

_ARTIFACTS = (
    "metadata.json",
    "config.yaml",
    "trajectory.csv",
    "oracle_outcomes.csv",
    "summary.json",
)


def implementation_hash(source: Path | None = None) -> str:
    """Hash all package Python sources, including uncommitted edits."""
    digest = hashlib.sha256()
    source = Path(__file__).resolve().parents[1] if source is None else Path(source)
    for path in sorted(source.rglob("*.py")):
        digest.update(path.relative_to(source).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def cache_key(config: ExperimentConfig, seed: int) -> str:
    """Include complete config, Git revision, implementation, scenario and selectors."""
    metadata = collect_metadata(config.config_hash, seed)
    models = []
    for spec in config["selectors"]:
        if spec.get("type") == "offline":
            path = Path(spec["model"])
            path = path if path.is_absolute() else project_root() / path
            models.append(
                {
                    "selector": spec["name"],
                    "path": str(path.resolve()),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
                    if path.is_file()
                    else None,
                }
            )
    return stable_digest(
        {
            "configuration_hash": config.config_hash,
            "git_commit": metadata["git_commit"],
            "implementation_hash": implementation_hash(),
            "version": __version__,
            "dependencies": metadata["dependencies"],
            "python": metadata["python_version"],
            "seed": seed,
            "scenario": config["dynamic"],
            "selectors": config["selectors"],
            "offline_models": models,
            "platform": metadata.get("platform"),
            "thread_environment": metadata.get("thread_environment"),
        }
    )


def _valid_key(key: str) -> bool:
    return isinstance(key, str) and re.fullmatch(r"[0-9a-f]{64}", key) is not None


def _artifact_hashes(directory: Path) -> dict[str, str]:
    return {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in _ARTIFACTS
    }


def lookup(key: str) -> Path | None:
    """Return an existing complete run only when all recorded hashes match."""
    if not _valid_key(key):
        return None
    pointer = project_root() / ".cache/experiments" / f"{key}.json"
    if not pointer.is_file():
        return None
    try:
        record = json.loads(pointer.read_text(encoding="utf-8"))
        directory = Path(record["directory"]).resolve()
        directory.relative_to((project_root() / "results").resolve())
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        valid = metadata.get("cache_key") == key and metadata.get("status") == "complete"
        return (
            directory if valid and record.get("artifacts") == _artifact_hashes(directory) else None
        )
    except (ValueError, KeyError, OSError, TypeError):
        return None


def remember(key: str, directory: Path) -> None:
    """Atomic small pointer; large CSV outputs remain under results/."""
    if not _valid_key(key):
        raise ValueError("research cache keys must be 64 lowercase hexadecimal characters")
    directory = directory.resolve()
    directory.relative_to((project_root() / "results").resolve())
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("cache_key") != key or metadata.get("status") != "complete":
        raise ValueError("only complete runs with the matching cache key can be cached")
    artifacts = _artifact_hashes(directory)
    root = project_root() / ".cache/experiments"
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / f"{key}_{uuid4().hex}.tmp"
    temporary.write_text(
        json.dumps({"directory": str(directory), "artifacts": artifacts}), encoding="utf-8"
    )
    temporary.replace(root / f"{key}.json")
