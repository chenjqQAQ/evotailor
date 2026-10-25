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


def is_valid_trial(item: TrialResult, *, allow_simulated: bool = False) -> bool:
    """Apply Section 3 validity, including explicit quality and measurement gates."""
    if item.measurement_kind != "real":
        return allow_simulated and item.status == TrialStatus.SUCCEEDED and item.objective is not None
    gates = ("deployment_succeeded", "workload_completed", "quality_passed", "measurement_complete")
    if any(item.validity.get(gate) is False for gate in ("cleanup_success", "served_model_matches_all_requests")):
        return False
    if item.status != TrialStatus.SUCCEEDED or any(item.validity.get(gate) is not True for gate in gates):
        return False
    if item.objective is None or not math.isfinite(item.objective):
        return False
    try:
        deploy = item.metrics["deployment_time_s"]
        task = item.metrics["task_makespan_s"]
        total = item.metrics["deployment_plus_task_time_s"]
    except KeyError:
        return False
    return (
        all(math.isfinite(value) and value >= 0 for value in (deploy, task, total))
        and math.isclose(total, deploy + task, rel_tol=1e-8, abs_tol=1e-6)
        and math.isclose(item.objective, total, rel_tol=1e-8, abs_tol=1e-6)
    )


def best_trial(
    history: Iterable[TrialResult], direction: str = "minimize", *, allow_simulated: bool = False
) -> TrialResult | None:
    valid = [item for item in history if is_valid_trial(item, allow_simulated=allow_simulated)]
    if not valid:
        return None
    chooser = min if direction == "minimize" else max
    return chooser(valid, key=lambda item: float(item.objective))


def valid_rate(history: Iterable[TrialResult]) -> float:
    """Section 3: valid real evaluations divided by all real evaluations, including failures."""
    items = [item for item in history if item.measurement_kind == "real"]
    return sum(is_valid_trial(item) for item in items) / len(items) if items else 0.0


def objective_value(history: Iterable[TrialResult]) -> float:
    """Equation 2, with positive infinity when no valid real configuration exists."""
    best = best_trial(history)
    return float(best.objective) if best is not None else math.inf


def best_verified(history: Iterable[TrialResult], direction: str = "minimize") -> float | None:
    """J_n: the lowest verified end-to-end time found within the budget (paper Eq. 2).

    Returns ``None`` when no trial is valid, which the caller must report as "no valid
    configuration" rather than as an infinite time.
    """
    best = best_trial(history, direction)
    return None if best is None else float(best.objective)
