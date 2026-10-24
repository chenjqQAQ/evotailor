"""The post-Campaign review: the one place the agent's remit is deliberately wide.

Inside a Campaign the agent may only nudge search policy, because the search
space, the objective, and the completed Trials have to stay fixed for the results
to mean anything. Once a Campaign has finished, none of those constraints apply
to the *next* Campaign: the space can be redesigned, the policy can move further,
and code can be proposed.

Two boundaries survive, and both are enforced by the applier rather than by this
prompt: the root-trust layer is never patched, and a change to what is measured
is recorded rather than applied, because silently redefining the objective would
make the new Campaign incomparable with the old one.
"""

from __future__ import annotations

import json
from typing import Any

from evotailor.loops.policy import SearchPolicy
from evotailor.spaces.search_space import Dimension

SYSTEM_PROMPT = """You are the EvoTailor campaign reviewer. A complete Campaign has finished, and you decide what the NEXT Campaign should do differently. Your remit here is wide: you may redesign the search space, change the search policy by any amount, and propose source-code patches. Read source_context before proposing a patch. Only modify exact supplied source; never invent unseen code. A functional test pass alone is not performance evidence; patches must be measured on real paired searches before adoption.

Read experience.search_space for the complete current domains and constraints, and experience.discovery for task, hardware, and backend evidence. Preserve those evidence grades. Never infer the full domain from only the values observed in Trials.

Read the experience record. Check measurement_kinds first: simulated observations validate workflow only and do not establish real hardware/model performance. Keep that distinction explicit in every conclusion. The record contains:
  trials: how many Trials ran, how many produced a measurement, and the best objective.
  evidence.dimension_effects: per dimension, the observed values, their means, and whether attribution is supported. attributable is true only for replicated comparisons holding other parameters fixed. Marginal means are associations, not causal effects. When it is false you must not claim a value is better.
  evidence.exposure.confounded_dimensions: dimensions whose effect is confounded with the proposer that proposed them.
  evidence.proposers: exposure, success rate, and mean objective rank per proposer.
  evidence.failures: Trials that failed, with the kernel's own failure label.
  evidence.calibration: how far predictions were from measurements.
  evidence.phase_budget: how the search budget was distributed.
  final_validation: repeated measurements of the winning configuration, taken outside the search budget.
  workload_signature: the workload facts, so you can tell whether a lesson transfers.

Return JSON with exactly these keys:
  summary: 3-8 concise English sentences on what this Campaign showed.
  space_changes: {"add": {name: {kind, choices} or {kind, lower, upper, step}}, "retune": {name: dimension}, "remove": [name], "evidence": {name: fact_id}}. A dimension may only be added or retuned when you cite a workload fact id from the experience record. The search space may only expand: no existing dimension may be removed or lose any of its existing values. Retuning may expand its domain only.
  policy_changes: only fields in policy_parameter_scope.active_fields, using policy_limits and policy_schema. There is no per-Episode step cap here, but the absolute domains still apply. All proposer weights must be in [0,1], and target only configured_proposers. Never tune coverage or neighborhood parameters when those sources are not configured.
  Each dimension kind must be exactly choice, integer, or float. For choices (including booleans) use {"kind":"choice","choices":[...]}. The word categorical is not a valid kind. Use dimension_schema, and preserve all existing domain values.
  objective_changes: {"metric": ..., "direction": ..., "note": ...} describing a change to what is measured. This is recorded but not applied automatically, because it would make the new Campaign incomparable with the old one.
  code_proposals: a list of {"path", "rationale", "expected_effect", "patch"}, where patch is a unified diff that applies with `git apply` from the repository root. Patches touching src/evotailor/preflight.py, scripts/trial_wrapper.py, or scripts/vllm_fpm_sitecustomize/sitecustomize.py are refused. Every patch is applied to a clean tree and must pass the full test suite, or it is reversed.
  no_change_reason: English explanation when you propose nothing.

Keep IDs, parameter names, numeric values, and file paths unchanged. Write the user-facing text in the operator's configured language."""


class CampaignReviewAgent:
    def __init__(self, client=None) -> None:
        self.client = client

    def review(
        self,
        experience: dict[str, Any],
        *,
        use_llm: bool,
        current_policy: dict[str, Any],
        protected_dimensions: list[str],
        source_context: dict | None = None,
    ) -> dict[str, Any]:
        configured = list(
            (experience.get("evidence") or {}).get("configured_proposers")
            or (experience.get("evidence", {}).get("proposers") or {}).keys()
        )
        active_fields = ["phase_boundaries", "uncertainty_bonus", "proposer_weights"]
        if "coverage" in configured:
            active_fields.append("exploration_pool_size")
        if "neighborhood" in configured:
            active_fields.append("neighborhood_changes")
        payload = {
            "source_context": source_context or experience.get("source_context", {}),
            "dimension_schema": Dimension.model_json_schema(),
            "policy_schema": SearchPolicy.model_json_schema(),
            "policy_limits": {
                "proposer_weights": [0, 1],
                "uncertainty_bonus": [0, 1],
                "phase_boundaries": "three strictly increasing values inside (0,1)",
                "exploration_pool_size": [16, 512],
                "neighborhood_changes": [1, 8],
            },
            "policy_parameter_scope": {
                "configured_proposers": configured,
                "active_fields": active_fields,
            },
            "experience": experience,
            "current_policy": current_policy,
            "protected_dimensions": protected_dimensions,
            "note": (
                "protected_dimensions are defined by the shared base search space and "
                "may only expand their domains and must never be removed or lose existing values."
            ),
        }
        if use_llm and self.client:
            try:
                answer = self.client.complete_json(
                    SYSTEM_PROMPT, json.dumps(payload, ensure_ascii=False, default=str)
                )
                return self._normalise(answer)
            except Exception as exc:
                raise RuntimeError(
                    "Campaign review failed; completed experiment awaits review"
                ) from exc
        return self._fallback(experience, "The configuration disables model calls.")

    @staticmethod
    def _normalise(answer: Any) -> dict[str, Any]:
        if not isinstance(answer, dict):
            raise ValueError("campaign reviewer returned a non-object")
        result = dict(answer)
        for key in ("space_changes", "policy_changes", "objective_changes"):
            value = result.get(key)
            if not isinstance(value, dict):
                result[key] = {}
        proposals = result.get("code_proposals")
        result["code_proposals"] = [
            item
            for item in (proposals if isinstance(proposals, list) else [])
            if isinstance(item, dict) and str(item.get("patch") or "").strip()
        ]
        if not isinstance(result.get("summary"), list):
            result["summary"] = [str(result.get("summary") or "")]
        return result

    @staticmethod
    def _fallback(experience: dict[str, Any], reason: str) -> dict[str, Any]:
        return {
            "summary": [
                f"Campaign {experience.get('campaign_id')} completed "
                f"{(experience.get('trials') or {}).get('total', 0)} trials.",
                reason,
            ],
            "space_changes": {},
            "policy_changes": {},
            "objective_changes": {},
            "code_proposals": [],
            "no_change_reason": "The review model is unavailable; no changes are proposed.",
        }
