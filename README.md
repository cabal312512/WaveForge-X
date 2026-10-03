<p align="center"><img src="docs/assets/wordmark.svg" width="560" alt="WaveForge-X"></p>

<p align="center">Decision-aware computation for wireless receivers</p>
<p align="center"><a href="docs/research.md">Research results</a> · <a href="docs/methods.md">Methods</a> · <a href="docs/reproduce.md">Reproduce</a> · <a href="https://github.com/cabal312512/WaveForge-X/releases/latest">Release & data</a> · <a href="README.zh-CN.md">中文</a></p>

WaveForge-X studies how much computation a wireless receiver needs **before its
final bit decisions are sufficiently determined**. Its CertiPHY receiver combines
time-domain PCG, posterior error bounds and Gray-QAM decision geometry to bound
disagreement with a specified full LMMSE hard-decision reference.

The implementation covers OFDM, OTFS and AFDM, complete CP/CPP channel operators,
QPSK/16/64-QAM, structured spectral refinement and a cost-aware refinement gate.
It runs on CPU with NumPy and SciPy. No neural network, hardware or external
measurement dataset is required.

## What the experiments show

The frozen confirmation study uses **48 independent channel clusters, 288 received
frames and 12 TDL-A/C/E profile-based conditions**. All methods use the same received
data and model; preparation, transforms, checks and iterations are charged.

| Finding | Result |
|---|---|
| Selective refinement vs Gray, disagreement budget 0.01 | **0.47% less modeled work**; paired 95% cluster interval for the cost ratio: **[0.98943, 0.99922]** |
| Full refinement vs Gray | **20.7% fewer iterations**, but **2.42% more total work** |
| Empirically quality-matched residual stopping | Selective refinement costs **4.58% more** at the tight target |
| End-to-end timing | No CPU improvement at the primary 0.01 budget |
| Scope of the gain | Concentrated in one four-cluster, low-Doppler TDL-C condition |

<img src="docs/figures/ablation_cost.png" width="900" alt="Preparation cost offsets savings in PCG iterations">

These are research results, including negative findings—not a claim of a universally
faster receiver. Bounds target the **receiver's same-model LMMSE decisions**, not
transmitted truth or ML/MAP optimality. Proofs assume exact arithmetic;
`floating_point_certified` remains `False`. The finite-FIR channel is
**TR 38.901 profile-based**, not a full conformance implementation.

## Repository

| Location | Contents |
|---|---|
| `src/waveforge6g/` | Waveforms, exact sparse channel operators, PCG and certificate methods |
| `scripts/` | Frozen study, numerical diagnostics, figure generation and evidence verification |
| `configs/` | Official profile values and predeclared statistical analysis |
| `data/tables/` | Compact numerical results, selection records and protocols |
| `data/*.zip` | Received data, output bits, references, manifests and supporting development evidence |
| `docs/` | Consolidated English research record, mathematics, reproduction and prior-art assessment |
| `tests/` | Decision boundaries, error bounds, Gray-region enumeration, waveform and channel checks |

The public package is `waveforge-x`; the Python import name remains `waveforge6g`
to preserve numerical code compatibility. Internal phase identifiers survive only
where needed to trace frozen evidence. They are not separate public editions.

## Run

Python 3.11 or 3.12. From a clone on Linux/macOS:

```bash
source scripts/env.sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m pytest -q
.venv/bin/python scripts/verify_evidence.py --extract
.venv/bin/python scripts/make_figures.py
```

On Windows, use `. ./scripts/env.ps1` and `.venv/Scripts/python.exe`.
The environment scripts keep dependencies, caches and generated outputs inside
the checkout. Figure regeneration reads saved evidence and does not repeat PHY
experiments. Fresh experiments and exact provenance are documented in
[Reproduction](docs/reproduce.md).

## License and attribution

Copyright © 2026 **cabal312512**. Code, original artwork and synthetic experiment
artifacts are provided under the [MIT License](LICENSE). Third-party papers and
standards are cited, not redistributed. See [source attribution](docs/related-work.md)
and [CITATION.cff](CITATION.cff).
