from __future__ import annotations

import json
from typing import Any

from evotailor.contracts import Phase, TrialResult

SYSTEM_PROMPT = """You plan the search phases for exactly one EvoTailor Episode. Choose one phase for every Trial in the Episode from: modeling, mixed, local, validation. Use modeling to improve broad search-space understanding, mixed to balance exploration and exploitation, local to refine the incumbent, and validation to repeat/check the best configuration. Read learning_context for explicit source observations and retrieved experience; source observations do not consume target budget. Do not claim cold start when compatible source measurements exist. For short budgets use only enough modeling to resolve key unknowns, then exploit evidence. Base the allocation on observed evidence, remaining budget, calibration maturity, and proposer coverage. The first-ever Trial must be modeling. Only use allowed_phases. When require_final_validation is true and total_trials is greater than one, the final Episode must include at least one validation Trial. Episode length is an upper bound: a boundary agent can close it earlier; unused phases are discarded. Return JSON with phase_plan, rationale, evidence, and risks. phase_plan must have exactly the requested number of items. evidence and risks must be arrays of complete sentences. Keep phase values in English. Write rationale, evidence, and risks in English."""


class EpisodePlannerAgent:
    def __init__(self, client=None) -> None:
        self.client = client

    def plan(
        self,
        *,
        episode: int,
        episode_trials: int,
        completed_trials: int,
        total_trials: int,
        history: list[TrialResult],
        proposer_evidence: dict[str, Any],
        calibration_samples: int,
        current_policy: dict[str, Any],
        deterministic_plan: list[Phase],
        use_llm: bool,
        allow_validation: bool = True,
        learning_context: dict | None = None,
    ) -> dict[str, Any]:
        fallback = {
            "phase_plan": [item.value for item in deterministic_plan],
            "rationale": "No model was used; retaining the deterministic phase policy.",
            "evidence": [],
            "risks": ["Phase allocation did not use an agent decision."],
            "used_fallback": True,
        }
        if not use_llm or self.client is None:
            return fallback
        payload = {
            "learning_context": learning_context or {},
            "allowed_phases": [
                phase.value for phase in Phase if allow_validation or phase != Phase.VALIDATION
            ],
            "require_final_validation": allow_validation,
            "episode": episode,
            "episode_trials": episode_trials,
            "completed_trials": completed_trials,
            "remaining_trials_including_this_episode": total_trials - completed_trials,
            "total_trials": total_trials,
            "is_final_episode": completed_trials + episode_trials >= total_trials,
            "objective_history": [
                {
                    "trial_id": item.trial_id,
                    "phase": item.candidate.metadata.get("phase"),
                    "proposer": item.candidate.proposer,
                    "objective": item.objective,
                    "status": item.status.value,
                }
                for item in history
            ],
            "proposer_evidence": proposer_evidence,
            "calibration_samples": calibration_samples,
            "current_policy": current_policy,
            "deterministic_reference_plan": [item.value for item in deterministic_plan],
        }
        try:
            answer = self.client.complete_json(
                SYSTEM_PROMPT, json.dumps(payload, ensure_ascii=False, default=str)
            )
            raw_plan = answer.get("phase_plan")
            if not isinstance(raw_plan, list) or len(raw_plan) != episode_trials:
                raise ValueError("phase_plan length does not match Episode budget")
            phases = [Phase(str(item)) for item in raw_plan]
            if not allow_validation and Phase.VALIDATION in phases:
                raise ValueError("validation is disabled by the Campaign budget")
            if completed_trials == 0 and phases[0] != Phase.MODELING:
                raise ValueError("the first Trial must use modeling")
            if (
                allow_validation
                and total_trials > 1
                and payload["is_final_episode"]
                and Phase.VALIDATION not in phases
            ):
                raise ValueError("the final Episode must include validation")

            def text_list(value: Any) -> list[str]:
                if value is None:
                    return []
                if isinstance(value, list):
                    return [str(item) for item in value]
                return [str(value)]

            return {
                "phase_plan": [item.value for item in phases],
                "rationale": str(answer.get("rationale", "The agent generated the phase plan.")),
                "evidence": text_list(answer.get("evidence")),
                "risks": text_list(answer.get("risks")),
                "used_fallback": False,
            }
        except Exception as exc:
            fallback["rationale"] = f"Phase-planning agent call failed: {type(exc).__name__}: {exc}"
            return fallback
