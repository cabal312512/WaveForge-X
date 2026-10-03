# WaveForge-X 1.0.0 — CertiPHY research snapshot

This release consolidates the working receiver and its experimental record into
one reproducible public edition.

- OFDM/OTFS/AFDM, sparse full-channel time-domain PCG, Gray energy checks,
  conditional spectral bounds, directional refinement and a selective gate.
- A frozen 48-cluster/288-frame confirmation study with actual quality-matched
  baselines, complete preparation cost accounting and five core figures.
- Received signals, transmitted bits, reference solutions, receiver outputs,
  development/validation selection records and SHA-256 manifests.
- Supporting method-development evidence, mathematical assumptions, numerical
  tests and a compact prior-art assessment.
- Original vector identity, English documentation and MIT licensing to cabal312512.

At the primary 0.01 disagreement budget, selective refinement reduces modeled
work by 0.47% relative to Gray, without a CPU improvement. Full refinement costs
2.42% more overall. At the tight empirical quality target it is 4.58% more expensive
than tuned residual stopping. The benefit is conditional; negative findings remain
part of this release.

This is a software/evidence release, not a peer-reviewed paper or a validated
floating-point certification claim. The finite-FIR channels are TR 38.901
profile-based and their approximation error is reported explicitly.

Assets: `certiphy-evidence.zip`, `method-development.zip`, and `SHA256SUMS`.
The repository's automatic source archives contain the runnable public edition.
