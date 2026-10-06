"""Transparent synthetic presets; no waveform performance labels are encoded."""

from copy import deepcopy

SCENARIO_NAMES = (
    "stable_low_mobility",
    "acceleration",
    "urban_transition",
    "abrupt_change",
    "cluster_birth_death",
    "mixed_extreme_mobility",
)


def scenario_config(config: dict) -> dict:
    """Merge supplied controls over a deterministic scenario definition."""
    supplied = deepcopy(config)
    name = supplied.get("scenario", "mixed_extreme_mobility")
    if name not in SCENARIO_NAMES:
        raise ValueError(f"unknown nonstationary scenario {name!r}")
    epochs = int(supplied.get("epochs", 120))
    interval = int(supplied.get("change_interval", 30))
    if epochs < 1 or interval < 1:
        raise ValueError("epochs and change_interval must be positive")
    base = {
        "snr_db": 12.0,
        "max_doppler_hz": 800.0,
        "delay_samples": 5.0,
        "k_factor_db": 3.0,
        "correlation": 0.95,
        "path_count": 3,
        "fractional_doppler_severity": 1.0,
    }
    base.update({key: supplied[key] for key in base if key in supplied})
    end = max(1, epochs - 1)
    changes = list(range(0, epochs, interval))
    changes = changes or [0]

    def piecewise(values: list[float]) -> dict:
        return {
            "type": "piecewise_constant",
            "points": [[t, values[i % len(values)]] for i, t in enumerate(changes)],
        }

    trajectories: dict = dict(base)
    dynamics = {
        "birth_probability": 0.0,
        "death_probability": 0.0,
        "cluster_birth_probability": 0.0,
        "cluster_death_probability": 0.0,
        "lifetime": None,
        "power_ramp_frames": 3,
        "max_paths": 10,
        "cluster_size": 2,
        "delay_drift_std": 0.0,
        "doppler_drift_std": 0.0,
        "power_drift_std": 0.0,
    }
    if name == "stable_low_mobility":
        trajectories["max_doppler_hz"] = supplied.get("max_doppler_hz", 15.0)
        trajectories["correlation"] = supplied.get("correlation", 0.995)
    elif name == "acceleration":
        trajectories["max_doppler_hz"] = {
            "type": "linear",
            "start": 15.0,
            "end": base["max_doppler_hz"],
        }
        trajectories["correlation"] = {"type": "linear", "start": 0.995, "end": base["correlation"]}
    elif name == "urban_transition":
        # Fit recovery into a full default run while preserving faster requested changes.
        urban_interval = min(interval, max(1, (epochs - 1) // 4))
        changes = list(range(0, epochs, urban_interval))
        trajectories.update(
            {
                "max_doppler_hz": piecewise(
                    [20, 0.5 * base["max_doppler_hz"], base["max_doppler_hz"], 80, 20]
                ),
                "path_count": piecewise([2, 3, 6, 2, 3]),
                "snr_db": piecewise(
                    [
                        base["snr_db"],
                        base["snr_db"] - 2,
                        base["snr_db"] - 3,
                        base["snr_db"] - 10,
                        base["snr_db"],
                    ]
                ),
                "delay_samples": {
                    "type": "smooth",
                    "before": 1,
                    "after": base["delay_samples"],
                    "start_time": 0,
                    "end_time": end,
                },
            }
        )
    elif name == "abrupt_change":
        trajectories.update(
            {
                "max_doppler_hz": piecewise([20, base["max_doppler_hz"]]),
                "snr_db": piecewise([base["snr_db"], base["snr_db"] - 7]),
                "delay_samples": piecewise([1, base["delay_samples"]]),
                "correlation": piecewise([0.995, max(0.1, base["correlation"] - 0.2)]),
            }
        )
    elif name == "cluster_birth_death":
        dynamics.update(
            {
                "birth_probability": 0.18,
                "death_probability": 0.035,
                "cluster_birth_probability": 0.08,
                "cluster_death_probability": 0.025,
                "lifetime": [12, 45],
                "delay_drift_std": 0.02,
                "doppler_drift_std": 0.02,
            }
        )
    else:
        # First third accelerates, middle has high Doppler and blockage, then recovers.
        knots = sorted(
            set(
                [
                    0,
                    max(1, end // 5),
                    max(2, 2 * end // 5),
                    max(3, 3 * end // 5),
                    max(4, 4 * end // 5),
                    max(5, end),
                ]
            )
        )
        trajectories.update(
            {
                "max_doppler_hz": {
                    "type": "piecewise_linear",
                    "points": [
                        [t, value]
                        for t, value in zip(
                            knots,
                            [
                                20,
                                80,
                                base["max_doppler_hz"],
                                base["max_doppler_hz"],
                                0.6 * base["max_doppler_hz"],
                                40,
                            ],
                            strict=True,
                        )
                    ],
                },
                "snr_db": piecewise(
                    [base["snr_db"], base["snr_db"] - 2, base["snr_db"] - 9, base["snr_db"] + 1]
                ),
                "k_factor_db": piecewise([6, 1, -5, 3]),
                "delay_samples": piecewise([1, base["delay_samples"], base["delay_samples"], 2]),
            }
        )
        dynamics.update(
            {
                "birth_probability": 0.12,
                "death_probability": 0.025,
                "cluster_birth_probability": 0.04,
                "cluster_death_probability": 0.015,
                "lifetime": [15, 60],
                "delay_drift_std": 0.01,
                "doppler_drift_std": 0.025,
            }
        )
    trajectories.update(supplied.get("trajectories", {}))
    dynamics.update(supplied.get("cluster_dynamics", supplied.get("dynamics", {})))
    return {
        "scenario": name,
        "epochs": epochs,
        "change_interval": interval,
        "trajectories": trajectories,
        "cluster_dynamics": dynamics,
    }
