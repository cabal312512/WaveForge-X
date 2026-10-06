"""Strict selector specifications shared by CLI configuration and the runner."""

import math

from .base import BaseSelector
from .discounted_ucb import DiscountedUCBSelector
from .objective import Objective
from .offline import OfflineSelector
from .random_selector import RandomSelector
from .rule_based import RuleBasedSelector
from .sliding_window_ucb import SlidingWindowUCBSelector
from .static_selector import StaticSelector
from .switching import SwitchingSelector
from .thompson import ThompsonSelector
from .ucb import UCBSelector

_OPTIONS = {
    "static": {"action"}, "random": set(),
    "rule": {"snr_threshold", "doppler_low", "doppler_high", "delay_threshold"},
    "offline": {"model"}, "ucb": {"alpha"}, "sliding_window_ucb": {"alpha", "window"},
    "discounted_ucb": {"alpha", "discount"}, "thompson": {"prior"},
    "linucb": {"alpha", "window", "regularization"},
}
_COMMON = {"name", "type", "policy", "base_threshold", "uncertainty_alpha", "max_hold",
           "change_detection", "ignore_switching_cost"}


def _number(value, name: str, low: float = 0, *, positive: bool = False, integer: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if (positive and value <= low) or (not positive and value < low) or (integer and not isinstance(value, int)):
        raise ValueError(f"invalid {name}")


def validate_selector_spec(spec: dict) -> None:
    """Validate keys/types/ranges without loading an offline model or touching files."""
    if not isinstance(spec, dict) or not isinstance(spec.get("type"), str) or spec["type"] not in _OPTIONS:
        raise ValueError(f"selector.type must be one of {sorted(_OPTIONS)}")
    kind = spec["type"]
    unknown = set(spec) - (_COMMON | _OPTIONS[kind])
    if unknown:
        raise ValueError(f"unknown {kind} selector options: {sorted(unknown)}")
    if "name" in spec and (not isinstance(spec["name"], str) or not spec["name"].strip()):
        raise ValueError("selector name must be a nonempty string")
    if spec.get("policy", "greedy") not in ("greedy", "sticky", "uncertainty", "uncertainty_aware"):
        raise ValueError("selector policy must be greedy, sticky, or uncertainty")
    if not isinstance(spec.get("ignore_switching_cost", False), bool):
        raise ValueError("ignore_switching_cost must be boolean")
    wrapper_options = {"base_threshold", "uncertainty_alpha", "max_hold", "ignore_switching_cost"}
    if "policy" not in spec and set(spec) & wrapper_options:
        raise ValueError("switching wrapper options require an explicit policy")
    if kind in ("static", "random", "rule") and spec.get("policy", "greedy") != "greedy":
        raise ValueError("non-greedy switching requires a learner with predicted reward scores")
    for key in ("alpha", "base_threshold", "uncertainty_alpha", "doppler_low", "doppler_high", "delay_threshold"):
        if key in spec:
            _number(spec[key], key)
    for key in ("prior", "regularization"):
        if key in spec:
            _number(spec[key], key, positive=True)
    if "snr_threshold" in spec:
        _number(spec["snr_threshold"], "snr_threshold", -float("inf"))
    if "discount" in spec:
        _number(spec["discount"], "discount", positive=True)
        if spec["discount"] > 1:
            raise ValueError("discount must be <=1")
    for key in ("max_hold", "window"):
        if key in spec and spec[key] is not None:
            _number(spec[key], key, 3 if key == "window" else 1, integer=True)
    if "max_hold" in spec and spec["max_hold"] is None:
        raise ValueError("max_hold must be a positive integer")
    if "window" in spec and spec["window"] is None and kind != "linucb":
        raise ValueError("only linucb accepts window=None")
    if kind == "static":
        _number(spec.get("action", 0), "action", integer=True)
        if spec.get("action", 0) > 2:
            raise ValueError("action must be 0,1,2")
    if kind == "rule" and spec.get("doppler_low", 0.05) >= spec.get("doppler_high", 0.35):
        raise ValueError("rule requires doppler_low<doppler_high")
    if kind == "offline" and (not isinstance(spec.get("model"), str) or not spec["model"].strip()):
        raise ValueError("offline model must be a nonempty path string")
    if "change_detection" in spec:
        change = spec["change_detection"]
        if not isinstance(change, dict) or set(change) - {"threshold", "delta", "feature_index", "min_instances"}:
            raise ValueError("change_detection has unknown options or is not a mapping")
        for key in ("threshold", "delta"):
            if key in change:
                _number(change[key], key, positive=key == "threshold")
        for key in ("feature_index", "min_instances"):
            if key in change:
                _number(change[key], key, 1 if key == "min_instances" else 0, integer=True)


def create_selector(spec: dict, seed: int, objective: Objective, context_dimension: int = 9) -> BaseSelector:
    """Construct base learner then optional switching and causal-change wrappers."""
    validate_selector_spec(spec)
    kind = spec["type"]
    options = {key: spec[key] for key in _OPTIONS[kind] if key in spec}
    constructors = {"static": StaticSelector, "random": RandomSelector, "rule": RuleBasedSelector,
                    "offline": OfflineSelector, "ucb": UCBSelector,
                    "sliding_window_ucb": SlidingWindowUCBSelector,
                    "discounted_ucb": DiscountedUCBSelector, "thompson": ThompsonSelector}
    if kind == "linucb":
        from .linucb import LinUCBSelector
        base = LinUCBSelector(context_dimension=context_dimension, **options)
    else:
        if kind in ("random", "thompson"):
            options["seed"] = seed
        if kind == "offline":
            options["objective"] = objective
        base = constructors[kind](**options)
    if "policy" in spec:
        base = SwitchingSelector(base, objective, policy=spec["policy"],
                                 base_threshold=spec.get("base_threshold", 0.01),
                                 alpha=spec.get("uncertainty_alpha", 0.1),
                                 max_hold=spec.get("max_hold", 20),
                                 cost_aware=not spec.get("ignore_switching_cost", False))
    if "change_detection" in spec:
        from .change_detection import ChangeAwareSelector
        if spec["change_detection"].get("feature_index", 2) >= context_dimension:
            raise ValueError("change detector feature_index exceeds context dimension")
        base = ChangeAwareSelector(base, **spec["change_detection"])
    return base
