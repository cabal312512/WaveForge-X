# Reproduction and data provenance

## Public edition

The repository provides one runnable software edition. Original internal phase
names survive inside machine-readable evidence and source snapshots so that
previous results can be traced; they do not imply multiple installable releases.
Publication changed paths, documentation and evaluation-helper organization, not
the PCG/Gray/structured-bound equations. Core receiver source is preserved, and
the public implementation is checked against saved received frames and outputs.

Use the project-local installation commands in the README. The pinned dependency
file records the numerical runtime used by the study; Python 3.11/3.12 are the
supported CI targets. NumPy/SciPy floating behavior and timing can differ across
platforms. Exact SHA/bitwise replication of floating-point internals is not promised
across arbitrary numerical libraries.

## Verify and regenerate without simulation

```bash
.venv/bin/python scripts/verify_evidence.py --extract
.venv/bin/python scripts/make_figures.py
```

Verification checks all three archive SHA-256 values and every member against its
manifest. Extraction is path-checked, refuses differing existing files, and places
only the main recorded study under `results/certiphy`. The original source snapshot
stays in the archive as provenance. Figure generation reads saved records and
creates `results/certiphy/analysis/*.png`, not new PHY observations.

`scripts/check_publication.py` checks public receiver outputs/work against six
saved confirmation cases (all three waveforms, two channel conditions). It is a
packaging regression, not another statistical confirmation or a retuning step.
The integrated receiver charges an additional `4*N*bits_per_symbol` work units
for explicit output selection. Outputs and iteration counts must match the
original fixtures; total work must differ by exactly this documented setup fee.
Historical saved costs remain unchanged.

## Fresh, separate experiments

To avoid overwriting recorded evidence, set a new output name:

```bash
export WAVEFORGE_RUN=fresh-study
.venv/bin/python scripts/run_study.py develop
.venv/bin/python scripts/run_study.py validate
.venv/bin/python scripts/run_study.py confirm
.venv/bin/python scripts/check_numerics.py
.venv/bin/python scripts/measure_memory.py
.venv/bin/python scripts/measure_filters.py
.venv/bin/python scripts/make_figures.py
```

On PowerShell use `$env:WAVEFORGE_RUN='fresh-study'`. Development precedes
validation; validation freezes source, gate, targets, selected baselines and test
seeds. Confirmation checks the source hash. Completed stages refuse overwrite;
interrupted stages retain completed frame records and can resume with the same
command. Fresh experiments use the same published seed plan, so they are
reproduction, not a new independent confirmation cohort.

The cumulative experiment limit is four hours and five GiB per output directory.
Caps retain failures and completed evidence. High-precision references are only
computed by the evaluator. The receiver never receives true bits or reference
answers. No high-precision fallback is called by the online receiver.

## Evidence schema

| File | Meaning |
|---|---|
| `<seed>_<frame>_<waveform>.npz` | `received`, `truth`, reference soft symbols, `reference_known`, ordered `methods`, output bit rows, channel seed and start index |
| Matching `.json` | Per-method costs/times, iterations/checks, BER/disagreement, status, coverage, preparation/check breakdown and refinement diagnostics |
| `complete.json` | Stage definition, method configurations, original code hash and seed plan |
| `analysis/develop.csv`, `validate.csv`, `confirm.csv` | Flat records for all development, validation and confirmation calls |
| `analysis/freeze.json` | Frozen main comparison and selected method specifications |
| `analysis/confirmation_lock.json` | Original pre-confirmation file hashes; historical path strings are provenance |
| `analysis/quality_test.csv` | Actual test quality, both target ceilings, equivalence intervals and eligible cost ratios |
| `MANIFEST.json` | SHA-256 of every included evidence member |

NPZ files contain synthetic numeric/string arrays and are read with
`allow_pickle=False`. The archives include neither credentials nor private input
prompts, installed environments, caches, third-party standard PDFs or downloaded
papers. They are licensed under MIT with the project.

`method-development.zip` contains separate compute-budget and structured-bound
development cohorts, including source snapshots where available. Its historical
protocols apply only to their own experiments; their sample counts cannot be added
to the current study's 48 independent clusters. The compact evolution narrative
is in [Research](research.md).

## Reading the tables

- `work` is the modeled real-equivalent operation count; `cpu` and `wall` are
  measured receiver time in seconds. Corrected CPU includes separately timed
  common FIR preparation amortized across two frames.
- `met` is a mathematical certificate request status, not heuristic solver
  success. Heuristics' certificate completion is reported as not applicable.
- `coverage` describes individually identified bits. `bound` can be smaller than
  the unknown-bit fraction because aggregate Gray checks exploit shared energy.
- Reference-ambiguous bits count against empirical quality selection. Failures
  remain in every relevant BER/cost denominator.
- The inference unit is a complete channel cluster, including correlated
  waveforms/frames. Per-bit binomial confidence intervals are not substituted.

Raw observed CPU values are preserved; reproducing plots does not remeasure time.

## Adaptation, budgets and reuse

The integrated CLI supports link and nonstationary waveform-policy experiments:
`python -m waveforge6g run configs/smoke.yaml` and
`python -m waveforge6g research configs/research/smoke.yaml`.
Use `scripts/research_v2.py --help` and `scripts/research_v3.py --help` for the
independent-feedback and receiver-budget study interfaces. These are fresh
simulation entry points, not required to inspect saved evidence.

For reusable preparation, `scripts/research_v6.py --help` and
`scripts/research_v7.py --help` expose the phased runners. They preserve original
output/config identifiers for provenance. Use a separate clone for full fresh
development/validation/confirmation so that saved data are not overwritten.
The original frozen hashes describe the original experiment sources; a changed
public checkout has a different packaging identity. Fresh development and
validation regenerate the appropriate source freeze before confirmation.

`reuse-evidence.zip` contains all 403,200 recorded confirmation method/frame
outcomes, complete episode tables being in `data/tables/reuse-*`. Its 48 received
fixtures each contain four initial frames, with explicit indices and all three
waveforms; they are diagnostic prefixes rather than full 100-frame timing runs.
They retain actual received data, transmitted truth, reference information and
algorithm output bits. Receivers themselves do not read the offline truth or
reference arrays. See [experiment record](experiment-record.md) for coverage.
The package's successful tests do not change `floating_point_certified=False`.
