from __future__ import annotations

from typing import Any

from evotailor.contracts import Candidate, Phase, Prediction, TrialResult
from evotailor.objective import best_trial


class ContextBuilder:
    @staticmethod
    def selector_payload(
        phase: Phase,
        candidates: list[Candidate],
        predictions: list[Prediction],
        history: list[TrialResult],
        reputation: dict[str, Any],
        objective_direction: str = "maximize",
        proposer_weights: dict[str, float] | None = None,
        *,
        search_space: dict[str, Any] | None = None,
        discovery: dict[str, Any] | None = None,
        experiment: dict[str, Any] | None = None,
        prior_knowledge: dict[str, Any] | None = None,
        uncertainty_bonus: float = 0.15,
    ) -> dict[str, Any]:
        successful = [x for x in history if x.objective is not None]
        incumbent = best_trial(successful, objective_direction)
        return {
            "decision_goal": "Select exactly one candidate for the configured trial backend. Balance expected objective, uncertainty, information gain, source evidence, and diversity. Simulated results are workflow-test data, not real performance evidence.",
            "uncertainty_bonus": uncertainty_bonus,
            "phase": phase.value,
            "search_space": search_space,
            "discovery": discovery,
            "experiment": experiment,
            # Measured and derived facts about this model, allocation and the workload
            # this Campaign has already seen. Advisory: it narrows the region the agent
            # reasons about but never removes a candidate.
            "prior_knowledge": prior_knowledge,
            "dimension_coverage": {
                name: {
                    str(value): sum(t.candidate.parameters.get(name) == value for t in history)
                    for value in sorted({c.parameters[name] for c in candidates}, key=str)
                }
                for name in candidates[0].parameters
            }
            if candidates
            else {},
            "objective_direction": objective_direction,
            "incumbent": None
            if incumbent is None
            else {
                "trial_id": incumbent.trial_id,
                "objective": incumbent.objective,
                "parameters": incumbent.candidate.parameters,
            },
            "candidates": [
                c.model_dump(mode="json")
                | {
                    "prediction": (
                        chosen := next(p for p in predictions if p.candidate_id == c.candidate_id)
                    ).model_dump(mode="json")
                    # Lifted out of diagnostics so the decision can see, without
                    # digging, whether the number above is an estimate or the prior
                    # showing through. A candidate outside the measured region has no
                    # calibrated value at all, and that is what should drive the choice
                    # during modeling.
                    | {
                        "support": ((chosen.diagnostics or {}).get("objective_model") or {}).get(
                            "support"
                        )
                    }
                }
                for c in candidates
            ],
            "proposer_evidence": reputation,
            "proposer_weights": proposer_weights or {},
            "recent_trials": [
                {
                    "trial_id": x.trial_id,
                    "proposer": x.candidate.proposer,
                    "objective": x.objective,
                    "status": x.status.value,
                    "measurement_kind": x.measurement_kind,
                    "parameters": x.candidate.parameters,
                    "failure_type": x.failure_type,
                    "error": x.error,
                    "validity": x.validity,
                    "measured_request_shape": (
                        ((x.telemetry.get("evidence") or {}).get("request") or {}).get(
                            "request_shape"
                        )
                    ),
                    "task_fixed_environment": (x.telemetry.get("evidence") or {}).get(
                        "task_fixed_environment"
                    ),
                }
                for x in history[-24:]
            ],
        }
