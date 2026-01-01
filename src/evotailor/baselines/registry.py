"""Baseline interface and registry (paper §5.1).

All methods, including EvoTailor, are driven through the same contract: given the current
stage space, the history of real evaluations and the remaining budget, produce the next
configuration to deploy.  A method never sees a measurement it did not pay for.

This keeps the comparison about *search behaviour*: the space, the budget accounting and
the timing decomposition are shared, so a difference in end-to-end time is attributable to
proposal and selection.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from evotailor.spaces.base import SearchSpace


@dataclass
class TrialRecord:
    """One real evaluation, as seen by a method."""

    configuration: Mapping[str, Any]
    valid: bool
    deploy_s: float | None = None
    task_s: float | None = None
    failure: str | None = None
    stage: int = 0

    @property
    def e2e(self) -> float | None:
        if self.deploy_s is None or self.task_s is None:
            return None
        return self.deploy_s + self.task_s


@dataclass
class BaselineContext:
    space: SearchSpace
    history: Sequence[TrialRecord]
    budget_remaining: int
    seed: int = 0
    rng: random.Random = field(default_factory=random.Random)

    def valid(self) -> list[TrialRecord]:
        return [r for r in self.history if r.valid and r.e2e is not None]

    def best(self) -> TrialRecord | None:
        rows = self.valid()
        return min(rows, key=lambda r: r.e2e) if rows else None


class Baseline(Protocol):
    name: str

    def next_configuration(self, ctx: BaselineContext) -> dict[str, Any]:
        ...


_REGISTRY: dict[str, type] = {}


def register(cls: type) -> type:
    _REGISTRY[getattr(cls, "name", cls.__name__)] = cls
    return cls


def build(name: str, **kwargs: Any) -> "Baseline":
    key = name.lower().replace("-", "_")
    if key not in _REGISTRY:
        raise KeyError(f"unknown method {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[key](**kwargs)  # type: ignore[call-arg]


def available() -> list[str]:
    return sorted(_REGISTRY)


# Importing the modules populates the registry.
from evotailor.baselines import llm_tuna, scoot, slo_guard  # noqa: E402,F401
