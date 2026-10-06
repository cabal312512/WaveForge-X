"""Causal online experiments: decisions precede all counterfactual evaluation."""

from __future__ import annotations

import importlib.metadata
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml

from ..channels.nonstationary import NonStationaryChannel
from ..channels.observation import ObservationModel
from ..config import ExperimentConfig
from ..decision.base import ACTIONS, BaseSelector, DecisionContext
from ..decision.objective import Objective
from ..decision.oracle import OracleSelector
from ..decision.regret import best_fixed_action, hindsight_optimal_sequence
from ..reproducibility import collect_metadata, configure_local_environment, stable_digest
from .link import evaluate_waveforms
from .research_cache import cache_key, implementation_hash, lookup, remember


def make_context(
    observation: Any, previous: int | None, recent_ber: float, system: dict
) -> DecisionContext:
    """Construct a value-only context. No engine, truth, oracle or future reference."""
    onehot = tuple(float(previous == action) for action in range(3))
    features = (
        1.0,
        observation.estimated_snr_db / 30,
        observation.estimated_doppler_hz * system["frame_size"] / system["sample_rate_hz"],
        observation.estimated_delay_spread_s
        * system["sample_rate_hz"]
        / max(system["cp_length"], 1),
        observation.estimated_variation,
        recent_ber,
        *onehot,
    )
    return DecisionContext(features=features, previous_action=previous, time=observation.time)


def switching_statistics(actions: list[int] | np.ndarray) -> dict:
    """Include censored first/last dwell lengths; first selection is not a switch."""
    sequence = np.asarray(actions, dtype=int)
    if sequence.ndim != 1 or len(sequence) == 0 or np.any((sequence < 0) | (sequence > 2)):
        raise ValueError("actions must be a nonempty vector of indices 0,1,2")
    boundaries = np.flatnonzero(np.diff(sequence) != 0) + 1
    lengths = np.diff(np.r_[0, boundaries, len(sequence)])
    return {
        "total_switch_count": len(boundaries),
        "switches_per_100_frames": 100 * len(boundaries) / len(sequence),
        "average_dwell_time": float(lengths.mean()),
        "minimum_dwell_time": int(lengths.min()),
        "occupancy": {
            action: float(np.mean(sequence == index)) for index, action in enumerate(ACTIONS)
        },
    }


def adaptation_statistics(
    table: pd.DataFrame, window: int = 5, regret_threshold: float = 0.05
) -> dict:
    """Latency until a full post-change window has mean conditional regret <= threshold.

    Search is censored at the next true change. Missing recovery stays null.
    Detection matches the first alarm after each change and before the next.
    Thresholds are specified in advance, not fitted to test outcomes.
    """
    changes = table.loc[table.change_point, "time"].astype(int).tolist()
    alarms = table.loc[table.detected_change, "time"].astype(int).tolist()
    episodes = []
    for index, change in enumerate(changes):
        end = changes[index + 1] if index + 1 < len(changes) else len(table)
        latency = None
        for start in range(change, end - window + 1):
            if table.iloc[start : start + window].instantaneous_regret.mean() <= regret_threshold:
                latency = start + window - 1 - change
                break
        following = [alarm for alarm in alarms if change <= alarm < end]
        episodes.append(
            {
                "change_time": change,
                "adaptation_latency": latency,
                "detection_delay": following[0] - change if following else None,
                "pre_change_mean_loss": float(
                    table.iloc[max(0, change - window) : change].objective.mean()
                )
                if change > 0
                else None,
                "post_change_mean_loss": float(
                    table.iloc[change : min(end, change + window)].objective.mean()
                ),
            }
        )
    matched = {
        change["change_time"] + change["detection_delay"]
        for change in episodes
        if change["detection_delay"] is not None
    }
    # Additional alarms are unmatched; this is an operational event measure,
    # not a calibrated false-positive probability under a null distribution.
    false_alarms = len(set(alarms) - matched)
    return {
        "episodes": episodes,
        "unmatched_alarms": false_alarms,
        "unmatched_alarms_per_100_epochs": 100 * false_alarms / len(table),
        "window": window,
        "regret_threshold": regret_threshold,
    }


class DynamicExperimentRunner:
    """Each policy receives only current noisy context and its selected base reward."""

    def __init__(
        self, config: ExperimentConfig, seed: int, selectors: dict[str, BaseSelector] | None = None
    ):
        self.config, self.seed = config, seed
        self._simulated = False
        self.objective = Objective(config["objective"])
        for spec in config["selectors"]:
            if spec["type"] == "offline":
                from ..decision.offline import check_offline_protocol

                check_offline_protocol(spec, config, seed)
        if selectors is None:
            from ..decision import create_selector

            selectors = {
                spec["name"]: create_selector(
                    spec, seed=seed, objective=self.objective, context_dimension=9
                )
                for spec in config["selectors"]
            }
        self.selectors = selectors

    def simulate(self) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        """Return policy trajectories, complete oracle outcomes, and summaries."""
        if self._simulated:
            raise RuntimeError("create a new DynamicExperimentRunner for an independent trajectory")
        self._simulated = True
        config, system = self.config, self.config["system"]
        engine = NonStationaryChannel(
            config["dynamic"],
            self.seed,
            system["frame_size"],
            system["cp_length"],
            system["sample_rate_hz"],
        )
        observation_config = config["observation"]
        observation_config.setdefault("reference_bandwidth_hz", system["sample_rate_hz"])
        observation_model = ObservationModel(observation_config, self.seed + 100003)
        oracle = OracleSelector(self.objective)
        previous = dict.fromkeys(self.selectors)
        recent = dict.fromkeys(self.selectors, 0.0)
        cumulative = dict.fromkeys(self.selectors, 0.0)
        rows, outcomes, losses = [], [], []
        for epoch in range(config["dynamic"]["epochs"]):
            state, channel = engine.step(epoch)
            observation = observation_model.observe(state)
            contexts, decisions = {}, {}
            # All policies commit before any candidate link is evaluated.
            for name, selector in self.selectors.items():
                context = make_context(observation, previous[name], recent[name], system)
                contexts[name] = context
                selector.observe(context)
                decisions[name] = selector.select(context)
            measured = evaluate_waveforms(config, channel, state.snr_db, self.seed, epoch)
            base_losses = np.asarray([self.objective.base_loss(metric) for metric in measured])
            losses.append(base_losses)
            for action, metric in enumerate(measured):
                outcomes.append(
                    {
                        "time": epoch,
                        "seed": self.seed,
                        "action": action,
                        "base_loss": base_losses[action],
                        **metric,
                    }
                )
            for name, decision in decisions.items():
                action, before = decision.action, previous[name]
                selected = measured[action]
                switch_cost = self.objective.switch_cost(before, action)
                selected_loss = float(base_losses[action] + switch_cost)
                best = oracle.evaluate(base_losses, before)
                oracle_loss = float(best.diagnostics["oracle_loss"])
                regret = selected_loss - oracle_loss
                if regret < -1e-12:
                    raise RuntimeError("conditional oracle regret became negative")
                regret = max(0.0, regret)
                cumulative[name] += regret
                truth = {f"true_{key}": value for key, value in state.to_dict().items()}
                for key, value in truth.items():
                    if isinstance(value, list | tuple | dict):
                        truth[key] = json.dumps(value)
                rows.append(
                    {
                        "time": epoch,
                        "seed": self.seed,
                        "selector": name,
                        "scenario": config["dynamic"]["scenario"],
                        "split": config["experiment"]["split"],
                        **truth,
                        **observation.to_dict(),
                        **selected,
                        "estimated_source_time": observation_model.last_source_time,
                        "selected_waveform": ACTIONS[action],
                        "action": action,
                        "previous_waveform": "none" if before is None else ACTIONS[before],
                        "oracle_waveform": ACTIONS[best.action],
                        "oracle_action": best.action,
                        "base_loss": float(base_losses[action]),
                        "switching_cost": switch_cost,
                        "objective": selected_loss,
                        "oracle_objective": oracle_loss,
                        "learner_reward": -float(base_losses[action]),
                        "deployment_reward": -selected_loss,
                        "instantaneous_regret": regret,
                        "cumulative_regret": cumulative[name],
                        "uncertainty": decision.uncertainty,
                        "confidence": decision.confidence,
                        "scores": json.dumps(np.asarray(decision.scores).tolist()),
                        "context_features": json.dumps(contexts[name].features),
                             "decision_diagnostics": json.dumps(decision.diagnostics, default=_json_default,
                                                                 allow_nan=False),
                        "change_point": bool(getattr(state, "change_point", False)),
                        "detected_change": bool(decision.diagnostics.get("detected_change", False)),
                        "channel_hash": stable_digest(channel.to_dict()),
                        "configuration_hash": config.config_hash,
                    }
                )
                self.selectors[name].update(contexts[name], action, -float(base_losses[action]))
                previous[name], recent[name] = action, selected["ber"]
        table, counterfactuals = pd.DataFrame(rows), pd.DataFrame(outcomes)
        base = np.asarray(losses)
        hindsight = hindsight_optimal_sequence(base, self.objective)
        fixed = best_fixed_action(base, self.objective)
        summary: dict[str, Any] = {
            "seed": self.seed,
            "scenario": config["dynamic"]["scenario"],
            "split": config["experiment"]["split"],
            "selectors": {},
        }
        for name, group in table.groupby("selector", sort=False):
            indices = group.index
            cumulative_loss = group.objective.cumsum().to_numpy()
            table.loc[indices, "hindsight_oracle_action"] = hindsight.actions
            table.loc[indices, "hindsight_oracle_loss"] = hindsight.losses
            table.loc[indices, "hindsight_dynamic_regret"] = cumulative_loss - np.cumsum(
                hindsight.losses
            )
            # Prefix differences against a whole-horizon optimum can be negative;
            # final total is the valid cost-aware sequence comparison.
            switches = switching_statistics(group.action.to_numpy())
            regret = group.instantaneous_regret
            summary["selectors"][name] = {
                "n_epochs": len(group),
                "n_bits": int(group.n_bits.sum()),
                "bit_errors": int(group.bit_errors.sum()),
                "mean_ber": float(group.bit_errors.sum() / group.n_bits.sum()),
                "mean_bler": float(group.bler.mean()),
                "mean_objective": float(group.objective.mean()),
                "final_cumulative_regret": float(regret.sum()),
                "mean_instantaneous_regret": float(regret.mean()),
                "median_regret": float(regret.median()),
                "p95_regret": float(regret.quantile(0.95)),
                "dynamic_regret": float(group.objective.sum() - hindsight.total_loss),
                "static_regret": float(group.objective.sum() - fixed.total_loss),
                "hindsight_oracle_total_loss": hindsight.total_loss,
                "best_fixed_total_loss": fixed.total_loss,
                "mean_uncertainty": float(group.uncertainty.mean()),
                "mean_runtime_s": float(group.runtime_s.mean()),
                **switches,
                "adaptation": adaptation_statistics(group.reset_index(drop=True)),
            }
        summary["selector_states"] = {
            name: selector.state_dict() for name, selector in self.selectors.items()
        }
        return table, counterfactuals, summary

    def run(self, *, plot: bool = True, use_cache: bool = True, force: bool = False) -> Path:
        """Save one seed's complete paired trajectory; reuse only matching source hash."""
        root = configure_local_environment()
        key = cache_key(self.config, self.seed)
        if use_cache and not force and self.config["cache"]["enabled"]:
            hit = lookup(key)
            if hit is not None:
                if plot:
                    from ..plotting.dynamic import plot_dynamic

                    plot_dynamic(hit)
                return hit
        name = f"{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}_{self.config['experiment']['name']}_seed{self.seed}_{uuid4().hex[:6]}"
        target = root / "results" / name
        target.mkdir(parents=True)
        metadata = collect_metadata(self.config.config_hash, self.seed)
        metadata.update(
            cache_key=key,
            implementation_hash=implementation_hash(),
            status="running",
            oracle_definition="conditional one-step with learner predecessor; independent hindsight DP sequence",
            observation_scope="decision context noisy/delayed; physical receiver uses perfect CSI",
            seed_policy="channel: SeedSequence(seed) child streams; observations: seed+100003; bits/noise: SeedSequence([seed,epoch,9173]); selectors: same supplied replicate seed per selector",
            snr_definition="transmitter Es/N0, ensemble Es=1; not post-fading received SNR",
        )
        (target / "config.yaml").write_text(
            yaml.safe_dump(self.config.to_dict(), sort_keys=False), encoding="utf-8"
        )
        (target / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (target / "git_commit.txt").write_text(
            (metadata["git_commit"] or "unversioned") + "\n", encoding="utf-8"
        )
        (target / "seed.txt").write_text(f"{self.seed}\n", encoding="utf-8")
        versions = sorted(
            f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()
        )
        (target / "environment.txt").write_text("\n".join(versions), encoding="utf-8")
        try:
            trajectory, outcomes, summary = self.simulate()
            trajectory.to_csv(target / "trajectory.csv", index=False)
            outcomes.to_csv(target / "oracle_outcomes.csv", index=False)
            (target / "summary.json").write_text(
                json.dumps(summary, indent=2, default=_json_default, allow_nan=False),
                encoding="utf-8",
            )
            metadata["status"] = "complete"
            (target / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            if use_cache and self.config["cache"]["enabled"]:
                remember(key, target)
            if plot:
                from ..plotting.dynamic import plot_dynamic

                plot_dynamic(target)
        except Exception as error:
            metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
            (target / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            raise
        return target


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")
