# Source attribution

The platform applies established numerical and communication tools. Its use of
them does not assert first use or state-of-the-art performance.

- [Yin et al., CG-based soft-output detection and precoding](https://arxiv.org/abs/1404.0424): iterative regularized linear reception.
- [Meurant and Tichý, Gauss–Radau upper bounds in CG](https://arxiv.org/abs/2209.14601): posterior energy estimates under a valid spectral underestimate.
- [Dolejší and Tichý, goal-oriented algebraic error control](https://arxiv.org/abs/2001.01929): output-aware error control in a different linear-functional setting.
- [Li and Yu, sparsified MMSE OTFS equalization](https://arxiv.org/abs/2207.00866): structural low-complexity reception; its turbo-decoder assumptions are not implemented here.
- [Fercoq et al., Mind the duality gap](https://proceedings.mlr.press/v37/fercoq15.html): safe solution-region screening in sparse learning. This does not authorize deleting coupled receiver variables.
- [3GPP TR 38.901](https://www.3gpp.org/ftp/Specs/archive/38_series/38.901/): source of the named TDL profile values; finite-FIR approximation error is separately measured.

General conjugate gradients, circulant projection, Lagrangian relaxation, dynamic
programming and resource accounting are existing tools. The project documents
their assumptions and actual results in controlled software experiments.
Original standards, papers and publisher PDFs are not redistributed.
