"""Asymptotic costs and transparent complex128 working-memory estimates."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ComplexityEstimate:
    """Model costs, not measured resident memory, FLOP counts, or timing."""

    transform_time: str
    detector_time: str
    detector_memory: str
    working_memory_bytes: int
    assumptions: str

    def to_dict(self) -> dict[str, str | int]:
        """Return a serializable description for result metadata."""
        return asdict(self)


def estimate_complexity(
    n_symbols: int,
    n_paths: int = 1,
    solver: str = "matrix_free",
    iterations: int | None = None,
    cp_length: int = 0,
) -> ComplexityEstimate:
    """Estimate a single frame's detector workspace and asymptotic operations.

    Dense includes H, Gram matrix, and an approximate factorization workspace.
    Matrix-free includes path phase arrays and roughly twelve work vectors.
    Python/BLAS overhead and allocator peaks are excluded; this is a planning
    estimate, not measured RSS. N=n_symbols, P=n_paths, K=solver iterations.
    """
    if n_symbols < 1 or n_paths < 1 or not 0 <= cp_length <= n_symbols:
        raise ValueError("Require N>=1, paths>=1, and 0<=CP<=N")
    if iterations is not None and iterations < 1:
        raise ValueError("iterations must be positive if given")
    if solver == "dense":
        return ComplexityEstimate(
            "O(N log N)",
            "O(N^3 + N^2(log N + P)) including operator construction",
            "O(N^2 + P(N+CP))",
            16 * (3 * n_symbols**2 + 4 * n_symbols + n_paths * (n_symbols + cp_length)),
            "complex128 dense H, Gram and factorization; transform structure retained in construction",
        )
    if solver in ("matrix_free", "iterative", "lsmr"):
        budget = "unspecified" if iterations is None else str(iterations)
        return ComplexityEstimate(
            "O(N log N)",
            "O(K (N log N + P(N+CP)))",
            "O(N + P(N+CP))",
            16 * (12 * (n_symbols + cp_length) + n_paths * (n_symbols + cp_length)),
            f"complex128 work vectors and path phases; K={budget}; convergence depends on conditioning",
        )
    if solver in ("one_tap", "one_tap_zf", "one_tap_mmse"):
        return ComplexityEstimate(
            "O(N log N)", "O(N)", "O(N)", 16 * 4 * n_symbols,
            "only a diagonal effective channel; excludes common time-domain path storage",
        )
    raise ValueError("solver must be dense, matrix_free, lsmr, or one_tap")


def working_memory_bytes(n_symbols: int, n_paths: int = 1, solver: str = "matrix_free") -> int:
    """Convenience access to the explicitly approximate detector workspace."""
    return estimate_complexity(n_symbols, n_paths, solver).working_memory_bytes
