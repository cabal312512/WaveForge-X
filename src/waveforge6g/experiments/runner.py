"""Experiment persistence. Raw CSV is independent from figure generation."""

from __future__ import annotations

import importlib.metadata
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import yaml

from ..config import ExperimentConfig
from ..reproducibility import collect_metadata, configure_local_environment
from .monte_carlo import MonteCarloRunner


def run_experiment(config: ExperimentConfig, *, plot: bool = True,
                   output_root: Path | None = None) -> Path:
    """Execute and save complete configuration, counts, raw frames and provenance."""
    root = configure_local_environment()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    name = f"{timestamp}_{config['experiment']['name']}_{uuid4().hex[:6]}"
    directory = (output_root or root / "results") / name
    directory.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir(exist_ok=True)
    handler = logging.FileHandler(root / "logs" / f"{name}.log", encoding="utf-8")
    logger = logging.getLogger("waveforge6g")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    metadata = collect_metadata(config.config_hash, config["experiment"]["seed"])
    metadata["status"] = "running"
    (directory / "config.yaml").write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8")
    (directory / "seed.txt").write_text(str(config["experiment"]["seed"]) + "\n", encoding="utf-8")
    (directory / "git_commit.txt").write_text((metadata["git_commit"] or "unversioned") + "\n", encoding="utf-8")
    environment = sorted(f"{dist.metadata['Name']}=={dist.version}" for dist in importlib.metadata.distributions())
    (directory / "environment.txt").write_text("\n".join(environment) + "\n", encoding="utf-8")
    metadata_path = directory / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    try:
        logger.info("Starting %s config=%s", name, config.config_hash)
        metrics, trials, examples = MonteCarloRunner(config).run()
        metrics.to_csv(directory / "metrics.csv", index=False)
        trials.to_csv(directory / "trials.csv", index=False)
        (directory / "channel_examples.json").write_text(json.dumps(examples, indent=2), encoding="utf-8")
        metadata.update(status="complete", metric_rows=len(metrics), trial_rows=len(trials))
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if plot and config["plotting"]["enabled"]:
            from ..plotting.publication import plot_results
            plot_results(directory)
        logger.info("Completed %s", name)
    except Exception as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        logger.exception("Failed %s", name)
        raise
    finally:
        logger.removeHandler(handler)
        handler.close()
    return directory
