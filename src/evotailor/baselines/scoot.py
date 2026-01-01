"""SCOOT: SLO-oriented tuning through constraint learning (paper §5.1).

SCOOT builds an implicit feasible region from the trials it observes and spends the
remaining budget inside it.  The adaptation here keeps that behaviour while sharing the
stage space and budget accounting of the harness:

* a *constraint model* accumulates, per dimension, the values seen in valid trials and the
  values that produced a failure;
* proposals are drawn from the intersection, so the method narrows quickly after a failure
  but cannot escape a space the outer loop has fixed.
"""
from __future__ import annotations

import random
from typing import Any

from evotailor.baselines.registry import BaselineContext, TrialRecord, register


@register
class ScootBaseline:
    name = "scoot"

    def __init__(self, exploration: float = 0.15) -> None:
        self.exploration = exploration
        self._failed_values: dict[str, set[Any]] = {}
        self._valid_values: dict[str, set[Any]] = {}

    def _observe(self, record: TrialRecord) -> None:
        for name, value in record.configuration.items():
            bucket = self._valid_values if record.valid else self._failed_values
            bucket.setdefault(name, set()).add(value)

    def next_configuration(self, ctx: BaselineContext) -> dict[str, Any]:
        for record in ctx.history:
            self._observe(record)
        rng = ctx.rng
        proposal: dict[str, Any] = {}
        for dim in ctx.space.dimensions:
            candidates = [
                v for v in self._valid_values.get(dim.name, set())
                if v not in self._failed_values.get(dim.name, set())
            ]
            if candidates and rng.random() > self.exploration:
                proposal[dim.name] = rng.choice(sorted(candidates, key=str))
            else:
                proposal[dim.name] = dim.sample(rng)
        ok, _ = ctx.space.admit(proposal)
        return proposal if ok else ctx.space.sample(rng)
