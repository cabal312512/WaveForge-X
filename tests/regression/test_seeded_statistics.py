"""Small measured integer-count reference; timing is deliberately excluded."""

import json
from pathlib import Path

import pytest

from waveforge6g.config import parse_config
from waveforge6g.experiments.monte_carlo import MonteCarloRunner


@pytest.mark.regression
def test_seeded_awgn_error_counts_match_measured_reference():
    path = Path(__file__).parents[1] / "reference" / "seeded_smoke.json"
    reference = json.loads(path.read_text(encoding="utf-8"))
    metrics, _, _ = MonteCarloRunner(parse_config(reference["configuration"])).run()
    columns = reference["compared_columns"]
    observed = metrics.sort_values(["snr_db", "waveform"])[columns].to_dict(orient="records")
    assert observed == reference["expected"]
