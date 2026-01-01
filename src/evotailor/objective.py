from __future__ import annotations

import math
from collections.abc import Iterable

from evotailor.contracts import TrialResult, TrialStatus


def cost(value: float, direction: str) -> float:
    """Convert either objective convention to a minimization cost."""
    return value if direction == "minimize" else -value


def improvement(value: float, incumbent: float | None, direction: str) -> float:
    if incumbent is None:
        return 0.0
    return incumbent - value if direction == "minimize" else value - incumbent


def best_trial(history: Iterable[TrialResult], direction: str) -> TrialResult | None:
    valid = [
        item
        for item in history
        if item.status == TrialStatus.SUCCEEDED
        and item.objective is not None
        and math.isfinite(item.objective)
        and not any(
            item.validity.get(gate) is False
            for gate in (
                "workload_completed",
                "served_model_matches_all_requests",
                "cleanup_success",
            )
        )
    ]
    if not valid:
        return None
    chooser = min if direction == "minimize" else max
    return chooser(valid, key=lambda item: float(item.objective))


def valid_rate(history: Iterable[TrialResult]) -> float:
    """Valid-evaluation rate eta_n (paper §3).

    The denominator is the number of *initiated deployments*, including failed ones, so a
    method that crashes often is penalised even when its best trial looks good.
    """
    items = list(history)
    if not items:
        return 0.0
    valid = sum(
        1
        for item in items
        if item.status == TrialStatus.SUCCEEDED
        and not any(
            item.validity.get(gate) is False
            for gate in (
                "workload_completed",
                "served_model_matches_all_requests",
                "cleanup_success",
            )
        )
    )
    return valid / len(items)


def best_verified(history: Iterable[TrialResult], direction: str = "minimize") -> float | None:
    """J_n: the lowest verified end-to-end time found within the budget (paper Eq. 2).

    Returns ``None`` when no trial is valid, which the caller must report as "no valid
    configuration" rather than as an infinite time.
    """
    best = best_trial(history, direction)
    return None if best is None else float(best.objective)
