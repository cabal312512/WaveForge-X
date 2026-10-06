"""Two-sided Page-Hinkley-style mean shift monitor on causal observations."""

import numpy as np

from .base import BaseSelector, DecisionContext, DecisionResult


class PageHinkley:
    """Monitor running-mean residual cumulative excursions in both directions.

    S+ += x-mean-delta, S- += mean-x-delta; detect when either sum minus its
    historical minimum exceeds threshold after min_instances. On detection,
    restart the reference mean and cumulative sums. This declared two-sided
    variant is an empirical detector, not a calibrated false-alarm test for
    correlated drifting channel features.
    """

    def __init__(self, threshold: float = 0.2, delta: float = 0.005,
                 min_instances: int = 10) -> None:
        if not np.isfinite(threshold) or threshold <= 0 or not np.isfinite(delta) or delta < 0:
            raise ValueError("Page-Hinkley requires threshold>0 and delta>=0")
        if isinstance(min_instances, bool) or not isinstance(min_instances, int) or min_instances < 1:
            raise ValueError("min_instances must be a positive integer")
        self.threshold, self.delta, self.min_instances = float(threshold), float(delta), min_instances
        self.reset()

    def _clear_statistics(self) -> None:
        self.count = 0
        self.mean = self.upper = self.lower = self.min_upper = self.min_lower = 0.0

    def reset(self) -> None:
        self._clear_statistics()
        self.total_samples = self.alarm_count = 0
        self.last_statistic = 0.0

    def update(self, value: float) -> bool:
        """Consume exactly one observable scalar; return whether a shift was detected."""
        if not np.isfinite(value):
            raise ValueError("change detector input must be finite")
        self.total_samples += 1
        self.count += 1
        self.mean += (value - self.mean) / self.count
        residual = value - self.mean
        self.upper += residual - self.delta
        self.lower += -residual - self.delta
        self.min_upper = min(self.min_upper, self.upper)
        self.min_lower = min(self.min_lower, self.lower)
        self.last_statistic = max(self.upper - self.min_upper, self.lower - self.min_lower)
        detected = self.count >= self.min_instances and self.last_statistic > self.threshold
        if detected:
            self.alarm_count += 1
            self._clear_statistics()
        return bool(detected)

    def state_dict(self) -> dict:
        return {"threshold": self.threshold, "delta": self.delta, "min_instances": self.min_instances,
                "count": self.count, "mean": self.mean, "upper": self.upper, "lower": self.lower,
                "min_upper": self.min_upper, "min_lower": self.min_lower,
                "last_statistic": self.last_statistic, "alarm_count": self.alarm_count,
                "total_samples": self.total_samples}


class ChangeAwareSelector(BaseSelector):
    """Reset a wrapped learner on a shift in a present estimated context feature.

    feature_index=2 is estimated normalized Doppler in the dynamic runner.
    observe/select on the same epoch are idempotent for detection, so runner
    lifecycle calls cannot accidentally consume the same signal twice.
    """

    def __init__(self, base: BaseSelector, threshold: float = 0.2, delta: float = 0.005,
                 feature_index: int = 2, min_instances: int = 10) -> None:
        if isinstance(feature_index, bool) or not isinstance(feature_index, int) or feature_index < 0:
            raise ValueError("feature_index must be a nonnegative integer")
        self.base, self.feature_index = base, feature_index
        self.detector = PageHinkley(threshold, delta, min_instances)
        self.last_observed_time = None
        self.detected_change = False
        self.detection_times: list[int] = []

    def observe(self, context: DecisionContext) -> None:
        if self.last_observed_time == context.time:
            return
        if self.last_observed_time is not None and context.time < self.last_observed_time:
            raise ValueError("change-aware contexts must be chronological")
        if self.feature_index >= len(context.features):
            raise ValueError("change detector feature_index exceeds context dimension")
        self.detected_change = self.detector.update(context.features[self.feature_index])
        self.last_observed_time = context.time
        if self.detected_change:
            self.base.reset()
            self.detection_times.append(int(context.time))
        self.base.observe(context)

    def select(self, context: DecisionContext) -> DecisionResult:
        self.observe(context)
        result = self.base.select(context)
        diagnostics = {**result.diagnostics, "detected_change": self.detected_change,
                       "change_statistic": self.detector.last_statistic,
                       "change_signal_feature": self.feature_index}
        return DecisionResult(result.action, result.scores, result.confidence, result.uncertainty, diagnostics)

    def update(self, context: DecisionContext, action: int, reward: float) -> None:
        self.base.update(context, action, reward)

    def reset(self) -> None:
        self.base.reset()
        self.detector.reset()
        self.last_observed_time = None
        self.detected_change = False
        self.detection_times = []

    def state_dict(self) -> dict:
        return {"selector": type(self).__name__, "feature_index": self.feature_index,
                "detection_times": self.detection_times.copy(), "detector": self.detector.state_dict(),
                "base": self.base.state_dict()}
