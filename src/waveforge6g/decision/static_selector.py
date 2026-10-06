"""Fixed-waveform controls; preferences are not calibrated reward predictions."""

import numpy as np

from .base import BaseSelector, DecisionContext, DecisionResult, validate_action


class StaticSelector(BaseSelector):
    """Always select one configured waveform; no inference or tuning."""

    def __init__(self, action: int = 0) -> None:
        self.action = validate_action(action)

    def select(self, context: DecisionContext) -> DecisionResult:
        return DecisionResult(self.action, np.full(3, -0.5), 0.0, 1.0,
                              {"policy": "fixed", "scores_available": False})

    def state_dict(self) -> dict:
        return {"selector": type(self).__name__, "action": self.action}
