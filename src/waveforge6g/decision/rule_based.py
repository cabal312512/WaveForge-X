"""Declared illustrative thresholds; no claimed universal waveform ordering."""

import numpy as np

from .base import BaseSelector, DecisionContext, DecisionResult


class RuleBasedSelector(BaseSelector):
    """Heuristic using normalized estimated SNR, Doppler and RMS delay.

    Features are [intercept,SNR/30,Doppler/(fs/N),RMSdelay*fs/CP,...].
    Thresholds are explicit configuration choices, never inferred from test
    outcomes. Fixed scores are unavailable; diagnostics mark this limitation.
    """

    def __init__(self, snr_threshold: float = 0, doppler_low: float = 0.05,
                 doppler_high: float = 0.35, delay_threshold: float = 0.5) -> None:
        values = (snr_threshold, doppler_low, doppler_high, delay_threshold)
        if not np.all(np.isfinite(values)) or not 0 <= doppler_low < doppler_high or delay_threshold < 0:
            raise ValueError("rule thresholds must be finite with 0<=doppler_low<doppler_high")
        self.snr_threshold, self.doppler_low = snr_threshold, doppler_low
        self.doppler_high, self.delay_threshold = doppler_high, delay_threshold

    def select(self, context: DecisionContext) -> DecisionResult:
        if len(context.features) < 4:
            raise ValueError("rule selector requires at least four normalized context features")
        _, snr, doppler, delay = context.features[:4]
        if snr < self.snr_threshold or doppler < self.doppler_low:
            action = 0
        elif doppler < self.doppler_high or delay >= self.delay_threshold:
            action = 1
        else:
            action = 2
        return DecisionResult(action, np.full(3, -0.5), 0, 1,
                              {"policy": "explicit_threshold_heuristic", "scores_available": False})

    def state_dict(self) -> dict:
        return {"selector": type(self).__name__, "snr_threshold": self.snr_threshold,
                "doppler_low": self.doppler_low, "doppler_high": self.doppler_high,
                "delay_threshold": self.delay_threshold, "threshold_source": "explicit_configuration"}
