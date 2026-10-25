"""Refuse only the configurations that provably cannot start the engine.

This is deliberately narrower than "configurations that look bad". A candidate is
refused here solely when the deployer cannot even reach the point of measuring:

* the configuration asks for more GPUs than the allocation holds, or
* the model weights alone exceed the memory the configuration reserves.

Both conditions have been observed to end a Trial at engine start with nothing
measured, so spending search budget on them buys no information. Refusing them is
not a preference and must never be used to steer the search: a configuration that is
merely expensive, or that the priors warn about, still runs.

The baselines do not consult this module. They must search the same space unaided, so
a refusal here is recorded in the Campaign as a portfolio-only decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Dimensions this check reads. Missing values make the check decline to judge.
TP = "deployment.tensor_parallel_size"
PP = "deployment.pipeline_parallel_size"
DP = "deployment.data_parallel_size"
GMU = "runtime.gpu_memory_utilization"


@dataclass(frozen=True)
class Infeasible:
    """Why a configuration cannot be attempted, in a form the proposers can read."""

    code: str
    detail: str
    parameters: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "detail": self.detail, "parameters": self.parameters}


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = int(value)
    return number if number >= 1 else None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def weight_memory_verdict(
    parameters: dict[str, Any],
    *,
    weights_bytes: int | None,
    gpu_memory_bytes: int | None,
    gpu_count: int | None,
    available_memory_bytes: int | None = None,
) -> Infeasible | None:
    """Return the reason a configuration cannot start, or ``None`` when it may.

    Silence is the default: if a number is missing, the candidate is allowed through
    rather than refused on a guess.
    """
    tp = _as_int(parameters.get(TP, 1))
    pp = _as_int(parameters.get(PP, 1))
    dp = _as_int(parameters.get(DP, 1))
    if tp is None or pp is None or dp is None:
        return None

    if available_memory_bytes is not None and weights_bytes is not None and weights_bytes / (tp * pp) > available_memory_bytes:
        return Infeasible("INSUFFICIENT_LIVE_MEMORY", "Model weights exceed currently free memory per device", dict(parameters))

    if gpu_count is not None and tp * pp * dp > gpu_count:
        return Infeasible(
            code="PARALLEL_EXCEEDS_ALLOCATION",
            detail=(
                f"TP * PP * DP = {tp} * {pp} * {dp} = {tp * pp * dp} requires {tp * pp * dp} GPUs; "
                f"only {gpu_count} are allocated, so the engine cannot launch."
            ),
            parameters=dict(parameters),
        )

    gmu = _as_float(parameters.get(GMU))
    if weights_bytes is None or gpu_memory_bytes is None or gmu is None:
        return None
    # gpu_memory_utilization caps what the engine may use on each GPU, and tensor and
    # pipeline parallelism divide the weights across tp*pp GPUs. If the weights alone
    # already exceed that cap, the engine cannot reserve any cache and exits during
    # startup with "No available memory for the cache blocks".
    budget = int(gmu * gpu_memory_bytes) * tp * pp
    if weights_bytes > budget:
        return Infeasible(
            code="WEIGHTS_EXCEED_MEMORY_BUDGET",
            detail=(
                f"Weight size {weights_bytes / 1024**3:.1f} GiB exceeds "
                f"the budget for gpu_memory_utilization={gmu:.3f} across {tp * pp} devices, "
                f"{budget / 1024**3:.1f} GiB, leaving no KV-cache capacity; the engine cannot launch."
            ),
            parameters=dict(parameters),
        )
    return None
