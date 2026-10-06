"""Confidence intervals with explicit sampling assumptions.

Bit errors within a fading frame are dependent. A Wilson interval on pooled
bit counts is only an iid-binomial diagnostic in that setting, not a valid
cluster-adjusted BER interval. Fixed-frame simulations may instead use the
frame-mean interval below, with independent channel realizations per frame.
Neither interval corrects repeated looks/optional error-count stopping.
"""

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm, t


def wilson_interval(errors: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for independent Bernoulli observations.

    Zero errors give a positive upper bound. Zero observations return (NaN,
    NaN), distinguishing missing data from a measured zero error probability.
    """
    if not isinstance(errors, int | np.integer) or not isinstance(total, int | np.integer):
        raise ValueError("errors and total must be integers")
    if not 0 <= errors <= total or not 0 < confidence < 1:
        raise ValueError("Require 0 <= errors <= total and 0 < confidence < 1")
    if total == 0:
        return float("nan"), float("nan")
    z = float(norm.ppf(0.5 + confidence / 2))
    proportion = errors / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half_width = z * np.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total**2))
    half_width /= denominator
    return max(0.0, float(center - half_width)), min(1.0, float(center + half_width))


def frame_mean_interval(frame_error_rates: ArrayLike, confidence: float = 0.95) -> tuple[float, float]:
    """Approximate t interval for mean error rate across independent equal-size frames.

    This treats frames as clusters and allows arbitrary within-frame error
    dependence. It is asymptotic for bounded nonnormal rates; fewer than two
    frames or constant observed rates return (NaN,NaN), since there is no
    evidence to estimate uncertainty. It is not an anytime-valid interval.
    """
    rates = np.asarray(frame_error_rates, dtype=float)
    if rates.ndim != 1 or not np.all(np.isfinite(rates)) or np.any((rates < 0) | (rates > 1)):
        raise ValueError("frame_error_rates must be a vector with values in [0, 1]")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")
    if rates.size < 2 or np.ptp(rates) == 0:
        return float("nan"), float("nan")
    margin = float(t.ppf((1 + confidence) / 2, rates.size - 1)) * rates.std(ddof=1) / np.sqrt(rates.size)
    average = float(np.mean(rates))
    return max(0.0, float(average - margin)), min(1.0, float(average + margin))
