"""SLO-Guard: failure-aware, budget-consistent tuning (paper §5.1).

The method treats a failed deployment as the primary signal.  After a failure it retreats
along the dimension that is most likely responsible — the one whose value changed with
respect to the last valid trial — instead of resampling the whole space.  That behaviour is
reproduced here so that the valid-rate comparison in the resource-constrained experiment is
attributable to the policy rather than to the harness.
"""
from __future__ import annotations

from typing import Any

from evotailor.baselines.registry import BaselineContext, register


@register
class SloGuardBaseline:
    name = "slo_guard"

    def __init__(self, retreat_fraction: float = 0.25) -> None:
        self.retreat_fraction = retreat_fraction

    def _blame(self, ctx: BaselineContext) -> list[str]:
        """Dimensions whose value differs from the last valid trial."""
        last_valid = None
        for record in reversed(ctx.history):
            if record.valid:
                last_valid = record
                break
        if last_valid is None or not ctx.history:
            return [d.name for d in ctx.space.dimensions]
        failed = ctx.history[-1]
        return [
            name for name, value in failed.configuration.items()
            if last_valid.configuration.get(name) != value
        ]

    def next_configuration(self, ctx: BaselineContext) -> dict[str, Any]:
        rng = ctx.rng
        if not ctx.history or ctx.history[-1].valid:
            return ctx.space.sample(rng)

        proposal = dict(ctx.history[-1].configuration)
        for name in self._blame(ctx):
            dim = ctx.space.dimension(name)
            # Retreat toward the low end of the domain, which is the conservative side for
            # every resource-related dimension (memory, batching, sequences).
            if dim.kind in ("int", "float") and dim.domain:
                low, high = dim.domain[0], dim.domain[1]
                proposal[name] = dim.snap(low + (high - low) * self.retreat_fraction)
            else:
                proposal[name] = dim.sample(rng)
        ok, _ = ctx.space.admit(proposal)
        return proposal if ok else ctx.space.sample(rng)
