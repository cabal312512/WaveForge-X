# Prior-art assessment and attribution

Assessment cutoff: **2026-10-04**. The purpose is to identify overlap, not to
claim priority. Numerical bounds, safe regions, circulating approximations,
Lagrangian relaxation and multiple-choice knapsack DP are established tools.
The remaining project-specific object is the aggregate Gray-decision disagreement
budget for a specified LMMSE reference, combined with a conditional sparse receiver.

| Work / primary source | Object and closest overlap | Distinction / reading scope |
|---|---|---|
| [Yin et al., CG soft-output detection (2014)](https://arxiv.org/html/1404.0424v2) | MIMO regularized Gram solve; fixed-K CG and SINR/LLRs | §III-B/C read. No aggregate hard-Gray reference-disagreement certificate in those methods. CG detection is not a contribution here. |
| [Meurant–Tichý, Gauss–Radau behavior (2023)](https://arxiv.org/html/2209.14601v2) | SPD CG A-energy bounds with a valid spectral underestimate | §3.1–3.2 read, including recurrence/integral remainder. Scalar energy bounds and their late-iteration limitations are existing results. |
| [Dolejší–Tichý, goal-oriented algebraic systems (2020)](https://arxiv.org/pdf/2001.01929) | Primal/dual solves control linear output-functional error | §5.2–5.3/6.2 read. Output-aware stopping and infrequent expensive checks are established; some estimates are explicitly not upper bounds. |
| [Fercoq–Gramfort–Salmon, Mind the duality gap (2015)](https://proceedings.mlr.press/v37/fercoq15.pdf) | Safe solution regions and support-function tests for Lasso | §2 and sphere tests read. Individual safe-bit logic parallels safe screening; Lasso variable deletion cannot justify deleting coupled receiver variables. |
| [Ndiaye et al., Gap Safe (2017)](https://www.jmlr.org/papers/volume18/16-577/16-577.pdf) | Group screening, strong-convexity/gap spheres | Theorem 6 and associated region construction read. Group-safe reasoning is not new; groups here represent axis decision regions, not sparse supports. |
| [El Ghaoui et al., SAFE (2010/2012)](https://arxiv.org/abs/1009.4219) | Safe elimination for sparse learning | Abstract checked; finer extensions not excluded by this reading. |
| [Li–Yu, sparsified MMSE OTFS (2023)](https://arxiv.org/html/2207.00866v4) | GMRES, FSPAI and decoder-dependent sparsification | §IV–V read. Turbo priors change the system and objective; this project neither implements turbo decoding nor reproduces the complete paper. |
| [Hucker–Reiss, early CG stopping (2024)](https://arxiv.org/abs/2406.15001) | Statistical inverse-problem risk/regularization | Abstract checked. Statistical reconstruction risk differs from deterministic disagreement with a specified linear reference. |
| [Dietl et al., CG MMSE DFE (2008)](https://mediatum.ub.tum.de/doc/976439/296962.pdf) | Krylov-reduced filter design and hard decision feedback | §III/Algorithm 1 read. DFE's reference and feedback assumptions differ; hard feedback alone is not a consistency certificate. |
| [Iterative receivers with channel estimation (2012)](https://link.springer.com/article/10.1186/1687-1499-2012-75) | Multiuser MIMO-OFDM, Krylov order/residual stopping | Relevant receiver discussion checked. Fixed-budget and residual methods belong among strong baselines. |
| [Pisinger, minimal MCKP algorithm (1995)](https://www.sciencedirect.com/science/article/pii/037722179500015I) | Multiple-choice capacity, core reduction and DP | Publisher abstract/author bibliography checked; full algorithm not obtained. This is a substantial overlap risk for any claim about target DP. |
| [Dudziński–Walukiewicz, hybrid DP/B&B (1994)](https://www.sciencedirect.com/science/article/pii/0377042793E0264M) | Multiple-choice knapsack with Lagrangian reduction | Full text unavailable in the assessment. Reduced-cost screening plus a small exact subproblem must not be renamed as a new general solver. |
| [T. F. Chan, optimal circulant preconditioner (1988)](https://epubs.siam.org/doi/10.1137/0909051) | Circular projection and Fourier diagonalization | Full text not obtained. Circulant averaging is established; an approximation still needs an independent Loewner-bound proof. |
| [Block CG error estimates (2025)](https://arxiv.org/abs/2502.14979) | Per-system A-error estimates from block Lanczos | Abstract checked. Multi-direction error tools already exist; their strongest variants were not excluded. |
| [Mixed-precision PCG error bounds (2025)](https://arxiv.org/abs/2510.11379) | Forward/backward rounding-error analysis | Abstract checked. Its assumptions were not transferred; this receiver does not claim a complete floating-point proof. |
| [Tichý, normwise backward error (2016)](https://www.karlin.mff.cuni.cz/~ptichy/download/public/Ti2016.pdf) | Incremental Jacobi norm estimates for CG stopping | Relevant introduction/estimation scope checked. Ritz/norm estimates are not automatically deterministic full-spectrum lower bounds. |

## Closest-method comparison

CG-D already supplies an iterative LMMSE detector; CertiPHY's stopping output is
different, but its linear solver is not new. Radau already bounds a shared error
energy; the additional information must come from a valid operator envelope and
decision geometry. Goal-oriented PDE solvers already focus effort on outputs and
schedule expensive checks; their continuous functional estimates cannot simply
be transferred to discontinuous Gray decisions. Gap Safe already separates an
enclosed optimum from an inactive decision region; aggregate unidentified bit
changes, rather than individual screening, are the relevant distinction. OTFS
GMRES/FSPAI already uses structure and staged cost reduction; different decoder
priors prevent a direct like-for-like performance superiority claim.

The most serious unresolved comparison is the general multiple-choice knapsack
core/reduction literature. No fully equivalent end-to-end receiver was confirmed,
but that is not evidence that none exists. Searches included arXiv, IEEE author
copies, Springer, Elsevier, SIAM and PMLR/JMLR. ACM DL and Google Scholar direct
queries did not return usable pages; several full texts were unavailable.
Those are substantive limits, not completed exhaustive coverage.

The project claims neither first use nor state of the art. Its mathematical
derivations are explicit adaptations under stated assumptions. Third-party papers
and standards remain at the linked sources and are not redistributed with the
MIT-licensed code, original artwork or synthetic evidence.
