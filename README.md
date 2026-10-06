<p align="center"><img src="docs/assets/wordmark.svg" width="560" alt="WaveForge-X"></p>

<p align="center">Waveform adaptation, iterative receivers and reusable computation</p>
<p align="center"><a href="docs/experiment-record.md">Experiment record</a> · <a href="docs/methods.md">Receiver methods</a> · <a href="docs/reproduce.md">Reproduce</a> · <a href="https://github.com/cabal312512/WaveForge-X/releases/latest">Release & data</a> · <a href="README.zh-CN.md">中文</a></p>

WaveForge-X is a CPU research platform for OFDM, OTFS and AFDM. It brings together
nonstationary waveform selection, finite receiver budgets, decision-aware PCG
stopping and reusable sparse time-domain preparation. NumPy/SciPy implementations,
frozen settings and saved experimental outcomes make the comparisons inspectable.

## Research record

| Topic | What the evidence establishes | Record |
|---|---|---|
| Waveform adaptation | Less switching can help; the tested controller does not beat the best fixed waveform. Corrected six-order probing matches Explore-Then-Commit. | [Adaptation](docs/adaptation.md) |
| Compute budgets | Receiver iterations create BER-cost tradeoffs. The tested joint selector is worse than a validation-selected fixed configuration at 2.49× work. | [Receiver budgets](docs/receiver-budget.md) |
| Anytime reception | Valid error sets constrain disagreement with a specified LMMSE hard-decision reference. Preparation can outweigh iteration savings. | [Receiver study](docs/research.md) |
| Reuse and changing CSI | Legal static reuse saves preparation. Against strong cached baselines, further phase/refinement work does not show a general end-to-end advantage. | [State reuse](docs/reuse.md) |

These are distinct experiments, including negative results. Comparisons retain
their channel assumptions, statistical units and cost conventions rather than
pooling everything into one claimed speedup.

<img src="docs/figures/receiver-budget/pareto.png" width="820" alt="BER and complete modeled computational cost">

## Implemented scope

- Gray BPSK/QPSK/16/64-QAM; unitary OFDM, reduced-prefix rectangular OTFS and
  chirp-prefix AFDM with complete useful-time CP/CPP channel operators.
- AWGN/fading, synthetic nonstationary multipath and finite-FIR TR 38.901
  profile-based TDL. The latter is not a full standard-conformance claim.
- Dense LMMSE references, sparse time-domain PCG, symbol-domain sparsification,
  posterior decision checks, structured refinement and checked reuse/fallback.
- Paired Monte Carlo, noisy selected-action feedback, independent reevaluation,
  uncertainty/switching controls, cost decomposition and episode-cluster analysis.

Certificates concern the supplied model's reference hard decisions, not zero BER,
ML/MAP optimality or reliability on an unknown physical channel.
`floating_point_certified` remains **False**. Modeled work and measured CPU time
are reported separately. No hardware, CUDA or deep-learning framework is needed.

## Repository

| Location | Contents |
|---|---|
| `src/waveforge6g/` | Physical layer, policy experiments, iterative receivers and reuse |
| `configs/` | Channel values, scenario grids and frozen method/analysis parameters |
| `scripts/` | Fresh study runners, diagnostics, artifact checks and figure regeneration |
| `data/tables/` | Seed/trajectory/episode summaries, paired results and ablations |
| `data/*.zip` | Checksummed synthetic evidence, including recorded failures |
| `docs/` | Consolidated English methods, experiments, limitations and reproduction |
| `tests/` | Channel/transform, policy, solver, decision-boundary and reuse checks |

Internal phase identifiers appear where required to trace original evidence.
The public project remains one integrated edition. [Experiment record](docs/experiment-record.md)
explains which raw arrays are complete and which are representative prefixes.

## Run

Python 3.11 or 3.12. In Bash:

```bash
source scripts/env.sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m pytest -q
.venv/bin/python scripts/verify_evidence.py --extract
.venv/bin/python scripts/make_figures.py
```

On Windows use `. ./scripts/env.ps1` and `.venv/Scripts/python.exe`.
The import and CLI name `waveforge6g` is retained for numerical compatibility;
the public package is `waveforge-x`.

```bash
.venv/bin/python -m waveforge6g run configs/smoke.yaml
.venv/bin/python -m waveforge6g research configs/research/smoke.yaml
```

Environments, dependencies, caches and generated outputs stay in the checkout.
Reading evidence and regenerating figures does not repeat the PHY experiments.
See [reproduction](docs/reproduce.md) for expensive fresh studies.

## License and attribution

Copyright © 2026 **cabal312512**. Code, original artwork and synthetic experiment
artifacts are under the [MIT License](LICENSE). Third-party papers/standards are
cited rather than redistributed. Cite the software version using [CITATION.cff](CITATION.cff).
AI assistance contributed to implementation, diagnostics and technical documentation;
the published numerical evidence records actual executed experiments.
