from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import os
import random
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from evotailor.contracts import Candidate, Phase, TrialResult, TrialStatus
from evotailor.objective import cost
from evotailor.spaces.search_space import SearchSpace


def candidate_id(name: str, values: dict[str, Any]) -> str:
    raw = json.dumps(values, sort_keys=True, default=str, separators=(",", ":"))
    return f"{name}-{hashlib.sha1(raw.encode()).hexdigest()[:12]}"


def make_candidate(name: str, values: dict[str, Any], rationale: str, **metadata: Any) -> Candidate:
    return Candidate(
        candidate_id=candidate_id(name, values),
        proposer=name,
        parameters=values,
        rationale=rationale,
        metadata=metadata,
    )


class Proposer(ABC):
    name: str
    version = "2.0.0"

    @abstractmethod
    def propose(
        self,
        space: SearchSpace,
        history: list[TrialResult],
        incumbent: TrialResult | None,
        phase: Phase,
        rng: random.Random,
        policy: Any | None = None,
    ) -> Candidate: ...


class CoverageProposer(Proposer):
    name = "coverage"

    def propose(self, space, history, incumbent, phase, rng, policy=None):
        # The search policy may widen or narrow how far this proposer reaches for
        # a diverse point; the phase default applies when no policy is supplied.
        default_pool = 256 if phase == Phase.MODELING else 96
        pool_size = int(getattr(policy, "exploration_pool_size", default_pool) or default_pool)
        pool = [space.sample(rng) for _ in range(pool_size)]
        seen = [trial.candidate.parameters for trial in history]
        values = max(
            pool,
            key=lambda item: min(
                (space.distance(item, previous) for previous in seen), default=1.0
            ),
        )
        return make_candidate(
            self.name,
            values,
            "Maximin coverage over the frozen normalized search space.",
            implementation="native_maximin",
            pool_size=pool_size,
        )


class NeighborhoodProposer(Proposer):
    name = "neighborhood"

    def propose(self, space, history, incumbent, phase, rng, policy=None):
        if incumbent is None:
            return make_candidate(
                self.name,
                space.sample(rng),
                "No incumbent exists; use a feasible seed.",
                implementation="native_incumbent_neighborhood",
            )
        values = dict(incumbent.candidate.parameters)
        names = list(space.dimensions)
        changes_by_phase = {
            Phase.MODELING: max(2, len(names) // 3),
            Phase.MIXED: max(1, len(names) // 4),
            Phase.LOCAL: 1,
            Phase.VALIDATION: 1,
        }
        changes = changes_by_phase[phase]
        override = getattr(policy, "neighborhood_changes", None)
        if override:
            # A policy may set the perturbation radius explicitly; it is still
            # bounded by the number of dimensions that exist.
            changes = min(int(override), len(names))
        for name in rng.sample(names, k=min(changes, len(names))):
            domain = space.dimensions[name].values()
            current = values[name]
            index = domain.index(current)
            neighbors = [i for i in (index - 1, index + 1) if 0 <= i < len(domain)]
            values[name] = domain[rng.choice(neighbors)] if neighbors else current
        try:
            values = space.normalize(values)
        except ValueError:
            values = space.sample(rng)
        return make_candidate(
            self.name,
            values,
            "Perturb the incumbent in a phase-dependent feasible neighborhood.",
            implementation="native_incumbent_neighborhood",
            parent=incumbent.trial_id,
        )


def _legacy_space(space: SearchSpace, root: Path):
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from evotailor.domain.enums import ApplyMethod, ParameterKind, RedeployLevel
    from evotailor.domain.search_space import ParameterDimension, SearchSpaceSpec

    dimensions = []
    for name, dimension in space.dimensions.items():
        values = dimension.values()
        if dimension.kind in {"integer", "float"} and dimension.step is not None:
            # A stepped numeric dimension is a discrete ordered grid, and it is handed over
            # as one. Passing lower/upper/step instead lets the other side regenerate the
            # values arithmetically, where 0.3 + 6*0.05 is 0.6000000000000001 rather than
            # 0.6: a configuration this project considers legal is then rejected there, and
            # SLO-Guard ended after 14 Trials because its own ask() would not accept an
            # observed value. Listing the values removes the disagreement entirely.
            kind = ParameterKind.ORDINAL
            kwargs = {"choices": list(values)}
        elif dimension.kind == "integer":
            kind = ParameterKind.INTEGER
            kwargs = {
                "lower": int(dimension.lower),
                "upper": int(dimension.upper),
                "step": int(dimension.step or 1),
            }
        elif dimension.kind == "float":
            kind = ParameterKind.FLOAT
            kwargs = {
                "lower": float(dimension.lower),
                "upper": float(dimension.upper),
                "step": None,
            }
        else:
            if all(isinstance(value, bool) for value in values):
                kind = ParameterKind.BOOLEAN
            elif all(
                isinstance(value, (int, float)) and not isinstance(value, bool) for value in values
            ):
                kind = ParameterKind.ORDINAL
            else:
                kind = ParameterKind.CATEGORICAL
            kwargs = {"choices": values}
        dimensions.append(
            ParameterDimension(
                name=name,
                category=name.split(".", 1)[0],
                kind=kind,
                default_value=space.safe_baseline[name],
                provider=space.backend,
                apply_method=ApplyMethod.CLI_ARGUMENT,
                redeploy_level=RedeployLevel.SERVICE_RESTART,
                supported_backends=[space.backend],
                **kwargs,
            )
        )
    return SearchSpaceSpec(
        dimensions=dimensions,
        global_constraints=space.global_constraints,
    )


def _history_observations(history: list[TrialResult], direction: str) -> list[dict[str, Any]]:
    observations = []
    for item in history:
        value = (
            cost(float(item.objective), direction)
            if item.status == TrialStatus.SUCCEEDED and item.objective is not None
            else None
        )
        observations.append(
            {
                "trial_id": item.trial_id,
                "configuration_id": item.candidate.candidate_id,
                "parameters": item.candidate.parameters,
                "observed": value is not None,
                "metrics": {} if value is None else {"task_makespan_s": value},
                "error": item.error,
            }
        )
    return observations


class OptunaTPEProposer(Proposer):
    """Real Optuna TPESampler rebuilt from durable Trial facts on every ask."""

    name = "tpe"

    def __init__(self, direction: str, policy: dict[str, Any] | None = None) -> None:
        self.direction = direction
        self.policy = policy or {}

    def propose(self, space, history, incumbent, phase, rng, policy=None):
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler_options = {
            "seed": rng.randrange(2**31),
            "n_startup_trials": int(self.policy.get("n_startup_trials", 10)),
            "multivariate": bool(self.policy.get("multivariate", True)),
        }
        prior_weight = float(self.policy.get("prior_weight", 1.0))
        if prior_weight != 1.0:
            sampler_options["prior_weight"] = prior_weight
        sampler = optuna.samplers.TPESampler(**sampler_options)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        distributions = {}
        for name, dim in space.dimensions.items():
            if dim.kind == "integer":
                distributions[name] = optuna.distributions.IntDistribution(
                    int(dim.lower), int(dim.values()[-1]), step=int(dim.step or 1)
                )
            elif dim.kind == "float":
                distributions[name] = optuna.distributions.FloatDistribution(
                    float(dim.lower),
                    float(dim.values()[-1]),
                    step=float(dim.step) if dim.step else None,
                )
            else:
                distributions[name] = optuna.distributions.CategoricalDistribution(
                    tuple(dim.values())
                )
        for item in history:
            try:
                if item.status == TrialStatus.SUCCEEDED and item.objective is not None:
                    trial = optuna.trial.create_trial(
                        params=item.candidate.parameters,
                        distributions=distributions,
                        value=cost(float(item.objective), self.direction),
                    )
                else:
                    trial = optuna.trial.create_trial(
                        params=item.candidate.parameters,
                        distributions=distributions,
                        state=optuna.trial.TrialState.FAIL,
                    )
                study.add_trial(trial)
            except (TypeError, ValueError):
                continue
        seen = {
            json.dumps(item.candidate.parameters, sort_keys=True, default=str) for item in history
        }
        for _ in range(200):
            trial = study.ask(fixed_distributions=distributions)
            values = {name: trial.params[name] for name in space.dimensions}
            key = json.dumps(values, sort_keys=True, default=str)
            if key not in seen and not space.validate_candidate(values):
                return make_candidate(
                    self.name,
                    values,
                    "Optuna TPESampler ask over all durable real observations.",
                    implementation="optuna.tpe",
                    optuna_version=optuna.__version__,
                    phase=phase.value,
                )
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
        raise RuntimeError("Optuna TPE could not produce a new feasible candidate")


@contextlib.contextmanager
def _isolated_optimizer_environment(name: str):
    """Prevent V2's Python 3.12 dependency path leaking into legacy workers."""
    if name != "scoot":
        yield
        return
    saved = {key: os.environ.get(key) for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")}
    try:
        for key in saved:
            os.environ.pop(key, None)
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class LegacyOptimizerProposer(Proposer):
    """Thin runtime bridge to the tested SLO-Guard and SCOOT adapters."""

    adapter_names = {
        "slo_guard": (
            "evotailor.optimizer_tools.adapters.slo_guard",
            "SloGuardOptimizer",
        ),
        "scoot": ("evotailor.optimizer_tools.adapters.scoot", "ScootOptimizer"),
    }

    def __init__(
        self,
        name: str,
        root: Path,
        direction: str,
        max_trials: int,
        policy: dict[str, Any] | None = None,
    ) -> None:
        if name not in self.adapter_names:
            raise ValueError(f"unsupported mature optimizer: {name}")
        self.name = name
        self.root = root
        self.direction = direction
        self.max_trials = max_trials
        self.policy = policy or {}

    def propose(self, space, history, incumbent, phase, rng, policy=None):
        legacy = _legacy_space(space, self.root)
        module_name, class_name = self.adapter_names[self.name]
        optimizer_type = getattr(importlib.import_module(module_name), class_name)
        policy = {"budget": self.max_trials, "batch_size": 1, **self.policy}
        optimizer = optimizer_type(**policy)
        observations = _history_observations(history, self.direction)
        task_spec = {
            "search_space": legacy.model_dump(mode="json"),
            "seed": rng.randrange(2**31),
            "max_trials": self.max_trials,
            **policy,
        }
        try:
            with _isolated_optimizer_environment(self.name):
                if self.name == "scoot":
                    optimizer.initialize(task_spec, history=observations, seed=task_spec["seed"])
                else:
                    optimizer.initialize(task_spec, history=(), seed=task_spec["seed"])
                    if observations:
                        optimizer.observe(observations)
                for _ in range(40):
                    result = optimizer.recommend(1)
                    if not result.suggestions:
                        break
                    suggestion = result.suggestions[0]
                    try:
                        values = space.normalize(dict(suggestion.parameters))
                    except ValueError:
                        continue
                    return make_candidate(
                        self.name,
                        values,
                        suggestion.rationale,
                        implementation=f"upstream_{self.name}",
                        provenance=dict(suggestion.provenance),
                        phase=phase.value,
                    )
                raise RuntimeError(f"{self.name} produced no new feasible candidate")
        finally:
            with contextlib.suppress(Exception):
                optimizer.finalize()


def default_proposers(
    *,
    direction: str,
    max_trials: int,
    mature_optimizer_root: Path,
    policies: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Proposer]:
    policies = policies or {}
    return {
        "tpe": OptunaTPEProposer(direction, policies.get("tpe")),
        "slo_guard": LegacyOptimizerProposer(
            "slo_guard",
            mature_optimizer_root,
            direction,
            max_trials,
            policies.get("slo_guard"),
        ),
        "scoot": LegacyOptimizerProposer(
            "scoot",
            mature_optimizer_root,
            direction,
            max_trials,
            policies.get("scoot"),
        ),
        "coverage": CoverageProposer(),
        "neighborhood": NeighborhoodProposer(),
    }
