# Experiment record

This edition brings together five linked lines of investigation. They share
waveform transforms and propagation operators, but their objectives, channel
models and statistical units differ. Their numbers must not be pooled as one
performance experiment.

| Study | Question and procedure | Retained evidence | Statistical unit |
|---|---|---|---|
| Waveform selection | Common bits/noise; static and nonstationary channels; fixed, oracle, contextual and switching policies | `data/tables/waveform-selection/`, configuration YAML, aggregate and seed summaries | Independent trajectory seed |
| Adaptation reliability | Separate feedback selection from independent reevaluation; calibrate before 80 frozen 120-epoch confirmation trajectories; test CSI/context mismatch | `data/tables/adaptation/`, frozen configuration, policy feedback and paired contrasts | Trajectory; four scenario families |
| Receiver budget | 72 development states; six probe orders; choose configurations on development feedback; evaluate 60 new paired trajectories | `data/tables/receiver-budget/`, `method-development.zip`, channel/resource settings | Trajectory |
| Anytime reception and structured refinement | Diagnose error sets and decision checks; freeze 48-cluster/288-frame profile-based comparison with quality-matched heuristics | Existing `certiphy-evidence.zip`, `method-development.zip`, `data/tables/`, five original figures | Independent channel cluster |
| Reusable preparation and operator validation | Freeze reuse dependencies and fallback; paired 100-frame episodes; then measure actual coefficient changes, phase alignment and selected payload bits | `data/tables/reuse-*`, `reuse-evidence.zip`, frozen physical/method settings | Channel episode; three waveforms are correlated runs |

## How to interpret the record

Development, validation and confirmation are separate. A tuned residual threshold
is an empirical quality control, not the same deterministic guarantee as a valid
error-set calculation. The tested differences include negative and failed cases.
Frame rows remain in the record when a certificate request was not completed;
that status is neither a proof of wrong output nor a successful certificate.

Work units count preparation, transforms, checks and iterative operations. They
are a transparent cost model, not CPU cycles or a hardware latency claim. CPU
measurements reflect the recorded software environment; old timings were not
rerun under a newly controlled machine workload for this release.

The initial adaptation results contain a probe-order bias identified later. The
six-order receiver-budget study uses the corrected commitment logic and a simple
Explore-Then-Commit competitor. Earlier numbers are retained as historical
observations, not silently relabeled as corrected experiments.

## Raw evidence and size

Existing archives retain the full compact standardized receiver study and earlier
structured-bound/budget development. The new reuse archive retains every
confirmation outcome row in a fixed field schema. For each reuse study it also
includes received signals, truth/reference outputs and method outputs for the
first four frames of eight representative episodes across three waveforms.
These fixture prefixes do not constitute complete 100-frame timing evidence.
Complete episode tables and physical seeds permit fresh replay with the supplied
experiment runners. Large redundant full-array dumps and local caches are not
included. Archive/member SHA-256 checks distinguish this packaging from the
original experiment-source hashes retained inside frozen protocols.

See [adaptation](adaptation.md), [receiver budgets](receiver-budget.md),
[anytime reception](research.md), and [state reuse](reuse.md) for model-specific
results and limitations.
