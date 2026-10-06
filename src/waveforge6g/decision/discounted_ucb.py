"""Exponentially discounted counts/reward sums with elapsed-epoch aging."""

import numpy as np

from .ucb import UCBSelector


class DiscountedUCBSelector(UCBSelector):
    """Discount both numerator and denominator; bonuses use effective counts."""

    def __init__(self, discount: float = 0.97, alpha: float = 1.0) -> None:
        if not np.isfinite(discount) or not 0 < discount <= 1:
            raise ValueError("discount must lie in (0,1]")
        self.discount = float(discount)
        super().__init__(alpha)

    def reset(self) -> None:
        super().reset()
        self.last_time = None

    def _prepare(self, time: int) -> None:
        if self.last_time is not None:
            if time < self.last_time:
                raise ValueError("discounted bandit contexts must be chronological")
            factor = self.discount ** (time - self.last_time)
            self.counts *= factor
            self.reward_sums *= factor
        self.last_time = time

    def state_dict(self) -> dict:
        return {**super().state_dict(), "discount": self.discount, "last_time": self.last_time}
