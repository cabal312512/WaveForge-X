"""Paired trials, stopping and saved evidence across complete physical links."""

import json

import numpy as np
import pandas as pd
import pytest

from waveforge6g.config import load_config, parse_config
from waveforge6g.experiments.monte_carlo import MonteCarloRunner
from waveforge6g.experiments.runner import run_experiment


def tiny_config(**changes):
    raw = {
        "experiment": {"name": "integration", "seed": 7331},
        "system": {"frame_size": 16, "cp_length": 4, "subcarriers": 4},
        "channel": {"model": "doubly_selective", "max_doppler_hz": 300,
                    "paths": [{"delay_samples": 0, "power_db": 0, "doppler_hz": 300},
                              {"delay_samples": 2, "power_db": -5, "doppler_hz": -90}]},
        "receiver": {"detectors": ["zf", "lmmse"], "solver": "dense"},
        "monte_carlo": {"max_frames": 4, "min_frames": 4, "min_bit_errors": 0,
                        "progress": False},
        "sweep": {"snr_db": [0, 8]},
        "plotting": {"enabled": False, "papr_oversampling": 2},
    }
    for section, values in changes.items():
        raw.setdefault(section, {}).update(values)
    return parse_config(raw)


def test_every_candidate_reuses_bits_noise_channel_and_budget():
    metrics, trials, _ = MonteCarloRunner(tiny_config()).run()
    for _, group in trials.groupby(["point_id", "trial"]):
        assert len(group) == 6
        for column in ("bits_sha256", "noise_sha256", "channel_sha256"):
            assert group[column].nunique() == 1
    assert (metrics.n_frames == 4).all()
    assert (metrics.n_bits == 4 * 16 * 2).all()
    np.testing.assert_allclose(metrics.ber, metrics.bit_errors / metrics.n_bits)
    assert set(metrics.stopping_reason) == {"max_frames"}


def test_waveform_order_does_not_change_trial_data_or_counts():
    config = tiny_config()
    reversed_config = tiny_config(system={"waveforms": ["afdm", "otfs", "ofdm"]})
    _, first, _ = MonteCarloRunner(config).run()
    _, reordered, _ = MonteCarloRunner(reversed_config).run()
    columns = ["point_id", "trial", "waveform", "detector", "bit_errors", "symbol_errors",
               "block_errors", "papr_db", "bits_sha256", "noise_sha256", "channel_sha256"]
    keys = ["point_id", "trial", "waveform", "detector"]
    pd.testing.assert_frame_equal(
        first[columns].sort_values(keys).reset_index(drop=True),
        reordered[columns].sort_values(keys).reset_index(drop=True),
    )


def test_joint_error_target_waits_for_minimum_frames_and_all_candidates():
    config = tiny_config(
        channel={"model": "awgn", "paths": None, "max_doppler_hz": 0},
        sweep={"snr_db": [-20]},
        monte_carlo={"min_frames": 2, "max_frames": 20, "min_bit_errors": 1},
    )
    metrics, trials, _ = MonteCarloRunner(config).run()
    assert metrics.n_frames.nunique() == 1
    assert (metrics.n_frames == 2).all()
    assert (metrics.bit_errors >= 1).all()
    assert set(metrics.stopping_reason) == {"all_error_targets"}
    assert len(trials) == 2 * 6


def test_result_files_preserve_counts_configuration_and_provenance(tmp_path, monkeypatch):
    monkeypatch.setenv("WAVEFORGE_PROJECT_ROOT", str(tmp_path))
    config = tiny_config()
    directory = run_experiment(config, plot=False)
    expected = {"config.yaml", "metadata.json", "metrics.csv", "trials.csv", "environment.txt",
                "git_commit.txt", "seed.txt", "channel_examples.json"}
    assert expected <= {path.name for path in directory.iterdir()}
    assert directory.parent == tmp_path / "results"
    assert load_config(directory / "config.yaml").config_hash == config.config_hash
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["configuration_hash"] == config.config_hash
    assert metadata["status"] == "complete"
    assert metadata["random_seed"] == 7331
    assert metadata["dependencies"]["numpy"]
    assert "dirty_working_tree" in metadata
    metrics = pd.read_csv(directory / "metrics.csv")
    assert set(metrics.configuration_hash) == {config.config_hash}
    assert metadata["metric_rows"] == len(metrics)
    assert (directory / "seed.txt").read_text(encoding="utf-8").strip() == "7331"


def test_independent_plotting_writes_pdf_png_and_report(tmp_path, monkeypatch):
    monkeypatch.setenv("WAVEFORGE_PROJECT_ROOT", str(tmp_path))
    directory = run_experiment(tiny_config(), plot=False)
    csv_before = (directory / "metrics.csv").read_bytes()
    from waveforge6g.plotting.publication import plot_results

    destination = plot_results(directory, tmp_path / "figures" / "redrawn")
    assert list(destination.glob("ber_*.pdf"))
    assert list(destination.glob("papr_*.png"))
    assert (destination / "channel.pdf").read_bytes().startswith(b"%PDF")
    assert (destination / "channel.png").read_bytes().startswith(b"\x89PNG")
    assert "integration" in (destination / "report.html").read_text(encoding="utf-8")
    assert (directory / "metrics.csv").read_bytes() == csv_before


def test_failed_run_records_failure_instead_of_complete_metrics(tmp_path, monkeypatch):
    monkeypatch.setenv("WAVEFORGE_PROJECT_ROOT", str(tmp_path))

    def fail(self):
        raise RuntimeError("deliberate numerical failure")

    monkeypatch.setattr(MonteCarloRunner, "run", fail)
    with pytest.raises(RuntimeError, match="deliberate"):
        run_experiment(tiny_config(), plot=False)
    directory = next((tmp_path / "results").iterdir())
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "failed"
    assert "deliberate numerical failure" in metadata["error"]
    assert not (directory / "metrics.csv").exists()
