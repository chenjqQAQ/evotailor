"""LLM-Tuna: hyper-parameter tuning for LLM inference (paper §5.1).

LLM-Tuna samples several candidates in parallel and keeps the best-ranked arm.  In this
harness the parallel trials are represented as a batch of proposals: the method draws a
batch, ranks it with a surrogate built from its own history, and returns the first point
whose predicted value is not dominated by the incumbent.

The batch size is the only free parameter; it is exposed so that the budget comparison is
fair (a wider batch consumes the same number of real evaluations, not more).
"""
from __future__ import annotations

from typing import Any

from evotailor.baselines.registry import BaselineContext, register


@register
class LlmTunaBaseline:
    name = "llm_tuna"

    def __init__(self, batch_size: int = 4, jitter: float = 0.1) -> None:
        self.batch_size = batch_size
        self.jitter = jitter

    def _score(self, ctx: BaselineContext, candidate: dict[str, Any]) -> float:
        """Cheap surrogate: distance to the best configuration seen so far."""
        best = ctx.best()
        if best is None:
            return 0.0
        total = 0.0
        for name, value in candidate.items():
            reference = best.configuration.get(name)
            if isinstance(value, (int, float)) and isinstance(reference, (int, float)):
                scale = abs(reference) or 1.0
                total += abs(value - reference) / scale
        return total

    def next_configuration(self, ctx: BaselineContext) -> dict[str, Any]:
        batch = [ctx.space.sample(ctx.rng) for _ in range(self.batch_size)]
        batch.sort(key=lambda c: self._score(ctx, c))
        # Small jitter keeps the arm from collapsing onto the incumbent.
        chosen = batch[0]
        if ctx.rng.random() < self.jitter and len(batch) > 1:
            chosen = batch[1]
        return chosen
