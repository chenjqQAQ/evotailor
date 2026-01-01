"""Inner optimization loop (paper §4.2).

Within a stage the loop repeats:

1. build a candidate set ``Q_t = ⋃_{g∈G_k} g(S_k, H_{t-1})`` from the generators;
2. reject candidates that are infeasible from resource evidence (no budget consumed);
3. let the selector rank the survivors with predictions and history, and pick one;
4. deploy it, run the task, record ``(T_deploy, T_task, validity, diagnostics)``;
5. update the history, the task model and the AIC calibration;
6. update the best verified result.

The loop only depends on the abstract interfaces below, which is what allows the same
driver to run EvoTailor and every baseline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from evotailor.baselines.registry import TrialRecord
from evotailor.execution.timing import TimingBreakdown
from evotailor.spaces.base import SearchSpace, Stage


@dataclass
class Candidate:
    """A proposed configuration plus the provenance the selector needs."""

    configuration: dict[str, Any]
    generator: str
    rationale: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrialOutcome:
    """What one real evaluation produced."""

    configuration: dict[str, Any]
    timing: TimingBreakdown | None
    valid: bool
    failure: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    telemetry: dict[str, Any] = field(default_factory=dict)


class Generator(Protocol):
    name: str

    def propose(
        self, space: SearchSpace, history: Sequence[TrialRecord], rng: Any
    ) -> list[Candidate]:
        ...


class Selector(Protocol):
    def choose(
        self, candidates: Sequence[Candidate], context: Any
    ) -> Candidate | None:
        ...


class Executor(Protocol):
    def run(self, configuration: dict[str, Any]) -> TrialOutcome:
        ...


class Feasibility(Protocol):
    def check(self, configuration: dict[str, Any]) -> tuple[bool, str]:
        ...


@dataclass
class InnerLoopResult:
    stage: int
    trials: list[TrialOutcome]
    rejected: int
    best: TrialOutcome | None

    @property
    def valid_trials(self) -> list[TrialOutcome]:
        return [t for t in self.trials if t.valid]


def run_stage(
    stage: Stage,
    *,
    generators: Sequence[Generator],
    selector: Selector,
    executor: Executor,
    feasibility: Feasibility | None = None,
    history: Sequence[TrialRecord] | None = None,
    rng: Any = None,
    max_rounds: int = 64,
) -> InnerLoopResult:
    """Execute one stage until its trial budget is spent.

    ``budget_trials`` counts *initiated deployments*.  Candidates rejected by the
    feasibility gate or declined by the selector are not charged, matching the budget
    accounting in the paper.
    """
    history = list(history or [])
    trials: list[TrialOutcome] = []
    rejected = 0

    while len(trials) < stage.budget_trials and max_rounds > 0:
        max_rounds -= 1
        space = stage.space

        # 1. proposal
        pool: list[Candidate] = []
        for generator in generators:
            pool.extend(generator.propose(space, history, rng))
        if not pool:
            break

        # 2. explicit feasibility (free)
        if feasibility is not None:
            survivors = []
            for candidate in pool:
                ok, _reason = feasibility.check(candidate.configuration)
                if ok:
                    survivors.append(candidate)
                else:
                    rejected += 1
            pool = survivors
            if not pool:
                continue

        # 3. selection
        chosen = selector.choose(pool, context=None)
        if chosen is None:
            rejected += len(pool)
            continue

        # 4-5. deploy, measure, record
        outcome = executor.run(chosen.configuration)
        trials.append(outcome)
        history.append(
            TrialRecord(
                configuration=chosen.configuration,
                valid=outcome.valid,
                deploy_s=outcome.timing.deploy if outcome.timing else None,
                task_s=outcome.timing.task if outcome.timing else None,
                failure=outcome.failure,
                stage=stage.index,
            )
        )

    valid = [t for t in trials if t.valid and t.timing is not None]
    best = min(valid, key=lambda t: t.timing.e2e) if valid else None  # type: ignore[union-attr]
    return InnerLoopResult(stage=stage.index, trials=trials, rejected=rejected, best=best)


def should_evolve(stage: Stage, result: InnerLoopResult, *, min_trials: int = 6) -> bool:
    """Trigger for the outer loop (paper §4.2: updates happen at stage boundaries).

    Two conditions, both cheap: the stage must have produced enough evidence, and either
    the trial budget is spent or the last trials stopped improving.
    """
    if len(result.trials) < min_trials:
        return False
    if len(result.trials) >= stage.budget_trials:
        return True
    valid = result.valid_trials
    if len(valid) < 3:
        return True  # mostly failing stage: reflect rather than keep sampling
    tail = valid[-3:]
    best_so_far = min(v.timing.e2e for v in valid)  # type: ignore[union-attr]
    return all(abs(v.timing.e2e - best_so_far) < 1e-9 for v in tail)  # type: ignore[union-attr]
