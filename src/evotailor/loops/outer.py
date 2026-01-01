from __future__ import annotations

import json
from typing import Any

from evotailor.contracts import TrialResult
from evotailor.objective import best_trial

SYSTEM_PROMPT = """You are the EvoTailor outer meta-agent. Analyze evidence at an episode boundary. Separate facts from hypotheses.

You may only adjust the search policy, and only in small steps. policy_patch may contain these keys:
  phase_boundaries: three strictly increasing numbers in (0,1)
  uncertainty_bonus: number in [0,1]
  proposer_weights: map of proposer name to number in [0,1]; a weight of 0 stops calls only in adaptive pools; with fixed_candidate_pool true it affects selection preference only
  exploration_pool_size: integer in [16,512], the sample pool the coverage proposer draws from
  neighborhood_changes: integer in [1,8], how many dimensions the neighborhood proposer perturbs

Each Episode may move a field by at most: phase_boundaries 0.05, uncertainty_bonus 0.10, proposer_weights 0.25, exploration_pool_size 32, neighborhood_changes 1. Setting a proposer weight to exactly 0 or 1 is a discrete enable/disable decision and is allowed in one step. A larger patch is rejected outright, not clamped.

Only propose policy fields listed in policy_parameter_scope.active_fields. In particular exploration_pool_size affects only the coverage proposer, and neighborhood_changes affects only the neighborhood proposer: do not tune either when its proposer is inactive. A source can remain available in the code and appear in current_policy without being enabled in this Campaign.

You must not alter the frozen search space, the objective, completed trials, or fairness rules.

The evidence package is computed deterministically from recorded trials. Check their measurement_kind: simulated results validate workflow only and must not be presented as real hardware/model performance. Read it carefully:
  dimension_effects[name].attributable is true only when replicated comparisons hold every other configuration parameter fixed. Marginal means are associations, not causal effects. When it is false, a difference between values cannot be separated from run-to-run variation, so do not claim one value is better.
  exposure.confounded_dimensions lists dimensions whose effect is confounded with the proposer that proposed them.
  calibration reports how far predictions were from measurements.

Return JSON with facts, observations, hypotheses, proposed_changes, no_change_reason, evidence_refs, validation_plan, and policy_patch, plus parameter_experiments (an optional list of {dimension, reference_trial, value, evidence_ref, rationale}). Each experiment must cite evidence.dimension_effects.<dimension>, reference a successful real target trial, and change exactly one parameter to an allowed value. The kernel schedules two real anchor/variant pairs, all four measurements consuming trial budget. Do not fabricate outcomes. Cite the evidence fields you relied on in evidence_refs. Use an empty policy_patch and explain why in no_change_reason when the evidence does not support a change. Keep IDs, parameter names, numeric values, and policy keys unchanged. Write the user-facing text in facts, observations, hypotheses, proposed_changes, no_change_reason, and validation_plan in Simplified Chinese."""


class OuterAgent:
    def __init__(self, client=None):
        self.client = client

    def review(
        self,
        episode: int,
        trials: list[TrialResult],
        evidence: dict[str, Any],
        calibration_samples: int,
        use_llm: bool,
        objective_direction: str,
        current_policy: dict[str, Any],
        completed_trials: int,
        total_trials: int,
        episode_phase_plan: list[str],
    ) -> dict[str, Any]:
        active = evidence.get("active_proposers")
        if active is None:
            active = list((evidence.get("proposers") or {}).keys())
        fields = ["phase_boundaries", "uncertainty_bonus", "proposer_weights"]
        if "coverage" in active:
            fields.append("exploration_pool_size")
        if "neighborhood" in active:
            fields.append("neighborhood_changes")
        payload = {
            "policy_parameter_scope": {
                "active_proposers": active,
                "active_fields": fields,
                "fixed_candidate_pool": evidence.get("fixed_candidate_pool", False),
            },
            "episode": episode,
            "trials": [x.model_dump(mode="json") for x in trials],
            "evidence": evidence,
            "calibration_samples": calibration_samples,
            "objective_direction": objective_direction,
            "current_policy": current_policy,
            "completed_trials": completed_trials,
            "remaining_trials": max(total_trials - completed_trials, 0),
            "total_trials": total_trials,
            "episode_phase_plan": episode_phase_plan,
        }
        review_error = None
        if use_llm and self.client:
            try:
                return self.client.complete_json(SYSTEM_PROMPT, json.dumps(payload, default=str))
            except Exception as exc:
                raise RuntimeError("Episode review failed; boundary awaits recovery") from exc
        best = best_trial(trials, objective_direction)
        return {
            "facts": [
                f"本 Episode 共完成 {len(trials)} 个 Trial。",
                f"本 Episode 最优Target value：{best.objective if best else 'unavailable'}。",
            ],
            "observations": ["Backing model未成功返回，本复盘由确定性逻辑生成。"],
            "hypotheses": [],
            "proposed_changes": [],
            "evidence_refs": [x.trial_id for x in trials],
            "validation_plan": ["下一个 Episode 继续使用Current policy。"],
            "policy_patch": {},
            "no_change_reason": "当前证据不足以支持安全的策略修改。",
            "review_error": review_error,
        }
