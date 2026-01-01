"""Deterministic attribution evidence for the outer meta-agent.

Self-evolution is supposed to adjust search policy from what the Campaign
actually observed, which requires evidence. The evidence has to be computed the
same way every time from real Trials, rather than inferred by a model from a
table of parameter values.

Replication inside the search is deliberately not used, so attribution rests on
two other things:

* an effect estimate per dimension over the whole pooled Trial set, together
  with an explicit statement of whether the observation count actually supports
  attributing an effect to a value; and
* an exposure matrix, which makes a confound visible instead of leaving the
  model to notice it by luck.

Nothing here is guessed. When the data cannot support a statement, the field
says so and carries the count that shows why.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from statistics import mean, median
from typing import Any

from evotailor.contracts import TrialResult

#: Two observations per value are the minimum that lets a difference between
#: values be attributed to the value rather than to run-to-run variation.
MIN_OBSERVATIONS_PER_VALUE = 2


def _observed(trials: list[TrialResult]) -> list[TrialResult]:
    return [item for item in trials if item.objective is not None]


def _dimension_names(trials: list[TrialResult]) -> list[str]:
    names: set[str] = set()
    for trial in trials:
        names.update(trial.candidate.parameters)
    return sorted(names)


def dimension_effects(trials: list[TrialResult], direction: str) -> dict[str, dict[str, Any]]:
    """Per-dimension spread, per-value means, and whether attribution is supported."""
    observed = _observed(trials)
    effects: dict[str, dict[str, Any]] = {}
    for name in _dimension_names(observed):
        pairs = [
            (trial.candidate.parameters.get(name), float(trial.objective))
            for trial in observed
            if name in trial.candidate.parameters
        ]
        groups: dict[str, list[float]] = defaultdict(list)
        for value, objective in pairs:
            groups[repr(value)].append(objective)
        value_means = {
            value: {
                "observations": len(objectives),
                "mean": round(mean(objectives), 6),
                "best": round((min if direction == "minimize" else max)(objectives), 6),
            }
            for value, objectives in sorted(groups.items())
        }
        entry: dict[str, Any] = {
            "observations": len(pairs),
            "distinct_values_observed": len(groups),
            "value_means": value_means,
        }
        # Marginal association is descriptive only. Other parameters may co-vary.
        replicated = [v["mean"] for v in value_means.values() if v["observations"] >= 2]
        entry["association_gap"] = (
            max(replicated) - min(replicated) if len(replicated) >= 2 else None
        )
        strata = defaultdict(lambda: defaultdict(list))
        for trial in observed:
            parameters = trial.candidate.parameters
            if name not in parameters:
                continue
            key = json.dumps({k: v for k, v in parameters.items() if k != name}, sort_keys=True)
            strata[key][repr(parameters[name])].append(float(trial.objective))
        controlled = []
        for groups_at_fixed_others in strata.values():
            repeated = {v: xs for v, xs in groups_at_fixed_others.items() if len(xs) >= 2}
            values = sorted(repeated)
            for i, left in enumerate(values):
                for right in values[i + 1 :]:
                    controlled.append(
                        {
                            "left": left,
                            "right": right,
                            "left_n": len(repeated[left]),
                            "right_n": len(repeated[right]),
                            "median_difference_right_minus_left": median(repeated[right])
                            - median(repeated[left]),
                        }
                    )
        entry["controlled_comparisons"] = controlled
        entry["attributable"] = bool(controlled)
        entry["attribution_note"] = (
            "Repeated comparisons holding all other configured parameters fixed exist; "
            "time drift and environmental confounding still require checks."
            if controlled
            else "No replicated controlled comparison; parameter differences cannot be separated from run-to-run variation and co-varying parameters."
        )
        entry["preferred_observed_value"] = (
            (min if direction == "minimize" else max)(
                value_means, key=lambda v: value_means[v]["mean"]
            )
            if len(value_means) > 1
            else None
        )
        effects[name] = entry
    return effects


def exposure_matrix(trials: list[TrialResult]) -> dict[str, Any]:
    """Which proposer ever used which value of which dimension.

    This is the confound map. A dimension where each proposer used only one value,
    and different proposers used different values, cannot be attributed to either
    the proposer or the dimension.
    """
    matrix: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(Counter))
    for trial in _observed(trials):
        proposer = trial.candidate.proposer
        for name, value in trial.candidate.parameters.items():
            matrix[name][proposer][repr(value)] += 1

    confounded: list[dict[str, Any]] = []
    for name, by_proposer in matrix.items():
        shared = False
        seen: dict[str, str] = {}
        for proposer, counts in by_proposer.items():
            for value in counts:
                if value in seen and seen[value] != proposer:
                    shared = True
                seen.setdefault(value, proposer)
        if not shared and len(by_proposer) > 1:
            confounded.append(
                {
                    "dimension": name,
                    "reason": (
                        "no value of this dimension was ever tried by more than one "
                        "proposer, so its effect is confounded with the proposer"
                    ),
                    "proposers": sorted(by_proposer),
                }
            )
    return {
        "by_dimension": {
            name: {proposer: dict(counts) for proposer, counts in sorted(by_proposer.items())}
            for name, by_proposer in sorted(matrix.items())
        },
        "confounded_dimensions": confounded,
    }


def proposer_table(
    trials: list[TrialResult],
    direction: str,
    proposed_counts: dict[str, int] | None = None,
) -> dict[str, dict[str, Any]]:
    """Selected counts, success and improvement rates, and mean objective rank."""
    observed = _observed(trials)
    ordered = sorted(
        observed, key=lambda item: float(item.objective), reverse=direction != "minimize"
    )
    rank = {trial.trial_id: index + 1 for index, trial in enumerate(ordered)}

    table: dict[str, dict[str, Any]] = {}
    for trial in trials:
        name = trial.candidate.proposer
        entry = table.setdefault(
            name,
            {
                "proposed": int((proposed_counts or {}).get(name, 0)),
                "selected": 0,
                "succeeded": 0,
                "objective_ranks": [],
                "objectives": [],
            },
        )
        entry["selected"] += 1
        if trial.objective is not None:
            entry["succeeded"] += 1
            entry["objective_ranks"].append(rank[trial.trial_id])
            entry["objectives"].append(float(trial.objective))

    for entry in table.values():
        ranks = entry.pop("objective_ranks")
        objectives = entry.pop("objectives")
        entry["success_rate"] = (
            round(entry["succeeded"] / entry["selected"], 4) if entry["selected"] else None
        )
        entry["mean_rank"] = round(mean(ranks), 3) if ranks else None
        entry["best_objective"] = round(min(objectives), 6) if objectives else None
    return dict(sorted(table.items()))


def phase_budget(phases_by_trial: dict[str, str] | None) -> dict[str, Any]:
    if not phases_by_trial:
        return {
            "available": False,
            "note": "the caller did not supply the phase of each Trial",
        }
    counts = Counter(phases_by_trial.values())
    return {"available": True, "trials_per_phase": dict(sorted(counts.items()))}


def failure_log(trials: list[TrialResult]) -> list[dict[str, Any]]:
    """Failed Trials, with the label the kernel recorded rather than a guess."""
    entries = []
    for trial in trials:
        if trial.status.value == "succeeded" and trial.objective is not None:
            continue
        entries.append(
            {
                "trial_id": trial.trial_id,
                "proposer": trial.candidate.proposer,
                "status": trial.status.value,
                "failure_type": trial.failure_type,
                "error": (trial.error or "")[:300] or None,
            }
        )
    return entries


def calibration_trend(pairs: list[tuple[float, float]] | None) -> dict[str, Any]:
    """How far the prediction was from the measurement, over the Campaign."""
    if not pairs:
        return {"available": False, "note": "no prediction/measurement pairs were supplied"}
    ratios = [
        (measured / predicted)
        for predicted, measured in pairs
        if predicted not in (None, 0) and measured is not None
    ]
    if not ratios:
        return {"available": False, "note": "no usable prediction/measurement pair"}
    ordered = sorted(ratios)
    middle = ordered[len(ordered) // 2]
    return {
        "available": True,
        "pairs": len(ratios),
        "median_relative_error": median(
            [abs(a - p) / abs(a) for p, a in pairs if p is not None and a not in (None, 0)]
        ),
        "median_ratio": round(middle, 4),
        "mean_ratio": round(mean(ratios), 4),
        "min_ratio": round(ordered[0], 4),
        "max_ratio": round(ordered[-1], 4),
    }


def build_evidence(
    trials: list[TrialResult],
    *,
    direction: str,
    proposed_counts: dict[str, int] | None = None,
    phases_by_trial: dict[str, str] | None = None,
    calibration_pairs: list[tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Assemble everything the outer agent may reason from, and nothing else."""
    observed = _observed(trials)
    return {
        "observed_trials": len(observed),
        "total_trials": len(trials),
        "objective_direction": direction,
        "dimension_effects": dimension_effects(trials, direction),
        "exposure": exposure_matrix(trials),
        "proposers": proposer_table(trials, direction, proposed_counts),
        "phase_budget": phase_budget(phases_by_trial),
        "failures": failure_log(trials),
        "calibration": calibration_trend(calibration_pairs),
        "parameter_analysis_note": "Marginal means are associations, not causal importance. Controlled comparisons hold other configuration values fixed.",
        "replication_policy": (
            "The search does not repeat a configuration inside its budget, so an effect "
            "requires repeated measurements of at least two values with all other configuration values held fixed. Final validation repeats the winning configuration separately "
            "and outside the search budget."
        ),
    }
