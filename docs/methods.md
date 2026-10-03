# Receiver model and bounds

## Reference and decision convention

For each actual CP/CPP, let `C=RHP` be the useful time-domain channel and `T` the
unitary waveform synthesis. With complex noise variance `N0>0`,

\[
A=C^HC+N_0I,\quad b=C^Hy,\quad z_* = A^{-1}b,\quad s_* = T^Hz_*.
\]

The reference bits are the existing Gray slicer's decisions on `s_*`, with no
additional gain normalization. An estimated CSI model defines its own reference;
matching that reference does not establish correctness under an unknown channel.
All propagation uses the complete `C`, including interference from all paths.

For `r=b-Az` and a valid `0<alpha<=lambda_min(A)`, the standard residual bound is
`||z_*-z|| <= ||r||/alpha`. Unitarity preserves this Euclidean error radius.
A bit is individually safe only when its distance to a boundary changing that
Gray bit is **strictly greater** than the applicable radius. Equality is unknown.
Real and imaginary axes are separate; BPSK uses the real axis only.

Certified bit values are retained; other bits use the current iterate's slicer
output. The soft system is never clamped to constellation values or reduced by
deleting certified variables. With `u` individually unknown bits out of `B`,
reference disagreement and the absolute difference between the two true BERs
are each bounded by `u/B`. This elementary relation is not a new theorem.

## Shared Gray energy

Individual unknown counts can be loose. A reference inside an error ball cannot
spend the full radius independently on every bit. For binary axes, changing
`q+1` bits costs at least the sum of the `q+1` smallest squared boundary margins.
If that sum exceeds the energy budget, at most `q=floor(delta B)` bits can differ.

For higher-order QAM, an axis is a **group**: multiple Gray-bit changes share one
displacement. For axis `i` and target region `t`, let `c_it` be squared distance to
the region's closure and `h_it` its Hamming change on currently uncertified bits.
For every nonnegative multiplier `lambda`,

\[
H\leq \lambda E+\sum_i\max_t(h_{it}-\lambda c_{it}).
\]

This is a standard multiple-choice knapsack relaxation specialized to the actual
Gray regions. Five nonnegative trial multipliers provide a cheap upper bound;
their data-dependent selection affects tightness, not exact-arithmetic validity.
Closed regions conservatively include slicer ties. Independent bitwise energy
addition within a multi-bit axis would be invalid.

The target formulation instead computes a lower bound on

\[
D(Q)=\min_{\sum_i h_i\geq Q}\sum_i c_i(h_i),\qquad Q=q+1.
\]

Only `D(Q)>E` rules out the dangerous event. A feasible perturbation gives an upper
bound and cannot establish safety. Saturating the DP count at `Q` retains all
dangerous combinations, yielding the exact closed-region minimum in exact
arithmetic. Complexity is `O(groups * Q * bits_per_axis)` and memory `O(Q)`.

For `b_i=min_h(c_i(h)-lambda h)`, the dual lower bound is
`L(lambda)=lambda Q+sum_i b_i`. Any feasible dangerous combination must have total
nonnegative reduced cost at most `E-L(lambda)`. Larger options may therefore be
discarded. Singleton groups become constants; remaining groups undergo target DP.
The implementation uses at most ten multiplier updates and 180,000 DP cells.
Exhaustion returns unknown. This screens a combinatorial check, not PCG variables.

## Conditional spectral information

Circular diagonal averaging gives a circulant `C0=F^H diag(H) F` and the complete
sparse defect `E_C=C-C0`, including locations absent from `C`. Set
`epsilon=sqrt(||E_C||_1 ||E_C||_infinity)` and `a=min|H|`.
If `a>epsilon`, choose `tau=epsilon/a`. Young's inequality gives

\[
C^HC\succeq(1-\tau)C_0^HC_0-(1/\tau-1)\epsilon^2I,
\]
\[
0\prec M=F^H\operatorname{diag}(m)F\preceq A,\quad
m_j=N_0+(1-\tau)|H_j|^2-(1/\tau-1)\epsilon^2.
\]

In exact arithmetic `min(m)=N0+(a-epsilon)^2>0`. The zero-defect case has the direct
circulant limit. Failed applicability/positivity checks disable this refinement.
Neither `diag(A)`, a smallest Ritz value, nor a subset of physical paths is
assumed to be a matrix lower bound: path cross terms can cancel.

Inverse monotonicity and a full residual give an additional energy bound,

\[
e^HAe=r^HA^{-1}r\leq \eta=
\min\{\eta_{\rm Radau},\sum_j |(Fr)_j|^2/m_j\},\quad e^HMe\leq\eta.
\]

The existing fixed-Jacobi PCG supplies the standard Gauss–Radau term; it requires
a valid spectral underestimate. Recomputed residuals detect recurrence drift and
trigger conservative fallback/restart. These measures do not prove all floating
roundoff is enclosed.

For symbol direction `f_j=T e_j`, Cauchy–Schwarz yields
`|(T^H e)_j|^2 <= eta v_j`, with `v_j=f_j^H M^-1 f_j`.
The complex magnitude bounds each real axis without an extra factor of two.

| Transform | Cheap direction quantity |
|---|---|
| OFDM | `v_j=1/m_j` |
| OTFS, N=K*Mt, column-major symbols | `v(l,d)=mean_t 1/m[d+t*Mt]` |
| AFDM, T=Lambda1^H F^H Lambda2^H | circular correlation of `1/m` with `abs(FFT(conj(chirp1))/N)^2` |

Direction quantities are individual bounds. Their reciprocals are **not** joint
separable weights for OTFS/AFDM. OFDM can use the diagonal weighted-axis model;
other transforms retain the appropriate conservative joint relaxation.
Every bound is combined only for the same current output. On an unchecked final
iterate, the implementation falls back to retained individual certificates.

## Anytime computation and gate

PCG always solves the complete sparse normal equation with the same fixed SPD
Jacobi preconditioner. Periodic/adaptive checks either establish the requested
integer disagreement budget or continue until an iteration/work cap. A capped
frame returns its actual current output, certificate set and status; there is
no free high-precision rescue.

Selective refinement charges 80 modeled work units and reads only available
`fd,N,fs,alpha,beta,nnz` and step cost. It predicts steps by
`0.5*sqrt(beta/alpha)`, estimates potential savings as 20% of that count times
step cost, and compares against refinement preparation. Block Doppler must not
exceed 0.01 and predicted steps must reach 12. The gate decides whether to pay
for the existing refiner; it **cannot certify**. Disabled behavior equals Gray
apart from the fee and possible hard-budget interaction.

The energy ablation changes individual radii through `eta/alpha`, leaving the
baseline joint Gray ball unchanged. The direction ablation adds the inverse
direction quantities. Full refinement also enables weighted region/target checks;
its incremental effect cannot be attributed solely to DP.

## Numerical scope

Local region-distance, summation and DP/dual kernels use outward scalar rounding;
strict comparisons reject ambiguous boundaries. Upstream FFTs, sparse products,
spectral envelopes, PCG history and input construction do not have a complete
verified enclosure. **Every receiver reports `floating_point_certified=False`.**
Decimal and exact enumeration are independent small-system evaluators, not online
decision inputs. Observing no violations cannot replace a floating-point proof.
