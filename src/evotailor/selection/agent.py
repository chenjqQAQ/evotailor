from __future__ import annotations

import json
import math
from typing import Any

from evotailor.context import ContextBuilder
from evotailor.contracts import Candidate, Phase, Prediction, Selection, TrialResult

SYSTEM_PROMPT = """You are the bounded candidate selector for a configuration-search experiment. Select only a candidate_id present in the input. Treat predictions as uncertain estimates, not facts. A null task prediction means no fitted task estimate exists. The native forward_ms estimate is only compute time, not the task objective. In modeling phase prioritize measured-space coverage and useful comparisons; use historical configurations and failures as evidence. Read each candidate's support block: when support.extrapolated is true the candidate lies farther from every measurement than the measurements lie from each other, its calibrated value is null, and the fitted model has no estimate for it. A null calibrated value is not a low objective and must never be treated as one; in modeling prefer such candidates, because measuring them is the only way the model can learn that region, and a region never measured can never be compared. Do not let a predicted value for a well-sampled candidate veto an unmeasured one. When every candidate is extrapolated the flag no longer separates them, so order them by support.distance_to_nearest and take the largest: that candidate sits farthest from anything measured, which is the region whose measurement teaches the most. Never resolve that tie by falling back to the incumbent, because repeating the neighbourhood already measured is what leaves a region unmeasured. Use discovery workload facts, hardware and vLLM capabilities, the complete search space and experiment budget alongside calibrated objective, uncertainty, diversity, phase objective, and proposer evidence. Read uncertainty_bonus and experiment.search_policy: larger uncertainty_bonus favors information gain, lower favors measured objective. Retrieved experience contains conditional hypotheses, not target measurements; cite source campaign IDs when using it. Use prior_knowledge: each fact carries a statement, an implication, a source and a grade. Prefer candidates the facts support and avoid the regions they warn about, but the facts are advisory and you may select outside them when the other evidence justifies it. Preserve measured/declared/inferred/unavailable evidence grades; missing discovery is not evidence that a workload or hardware fact is false. When experiment.fixed_candidate_pool is true, all supplied sources remain available even if their policy weight is zero; weights affect preference only. Do not invent parameters or candidates. Return JSON with candidate_id, reason, confidence, and risk_factors. Keep candidate IDs, parameter names, numeric values, and evidence references unchanged. Write only the user-facing values of reason and risk_factors in English."""


def _confidence(value: Any) -> float:
    """Read the model's confidence without letting its wording end the Campaign.

    The field is an annotation, not the decision: the candidate id is what matters. A
    model that answers "moderately low" instead of a number used to raise out of here and kill
    a run that had already measured Trials, so anything unparsable falls back to the
    neutral default rather than propagating.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.5
    if not math.isfinite(number):
        return 0.5
    return min(max(number, 0.0), 1.0)


def _risk_factors(value: Any) -> list[str]:
    """Accept a list, a single string, or nothing; never raise over presentation."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


class SelectorAgent:
    def __init__(self, client=None):
        self.client = client

    def select(
        self,
        phase: Phase,
        candidates: list[Candidate],
        predictions: list[Prediction],
        history: list[TrialResult],
        reputation: dict,
        use_llm: bool = True,
        uncertainty_bonus: float = 0.15,
        objective_direction: str = "maximize",
        proposer_weights: dict[str, float] | None = None,
        *,
        search_space: dict | None = None,
        discovery: dict | None = None,
        experiment: dict | None = None,
        prior_knowledge: dict | None = None,
    ) -> Selection:
        if use_llm and self.client:
            payload = ContextBuilder.selector_payload(
                phase,
                candidates,
                predictions,
                history,
                reputation,
                objective_direction,
                proposer_weights,
                search_space=search_space,
                discovery=discovery,
                experiment=experiment,
                prior_knowledge=prior_knowledge,
                uncertainty_bonus=uncertainty_bonus,
            )
            try:
                answer = self.client.complete_json(
                    SYSTEM_PROMPT, json.dumps(payload, ensure_ascii=False)
                )
                chosen = str(answer["candidate_id"])
                if chosen not in {x.candidate_id for x in candidates}:
                    raise ValueError("model selected an unknown candidate")
                return Selection(
                    candidate_id=chosen,
                    reason=str(answer.get("reason") or "The model selected a candidate."),
                    confidence=_confidence(answer.get("confidence")),
                    risk_factors=_risk_factors(answer.get("risk_factors")),
                )
            except Exception as exc:
                raise RuntimeError(
                    "Selector model decision failed; campaign checkpoint preserved"
                ) from exc
        return self._fallback(
            candidates,
            predictions,
            "Configured deterministic selector.",
            uncertainty_bonus,
            objective_direction,
            proposer_weights,
        )

    @staticmethod
    def _fallback(
        candidates: list[Candidate],
        predictions: list[Prediction],
        reason: str,
        uncertainty_bonus: float,
        objective_direction: str,
        proposer_weights: dict[str, float] | None = None,
    ) -> Selection:
        by_id = {x.candidate_id: x for x in predictions}
        weights = proposer_weights or {}

        def score(candidate: Candidate) -> float:
            prediction = by_id[candidate.candidate_id]
            value = prediction.calibrated
            if value is None or prediction.uncertainty is None:
                raise ValueError(
                    "Deterministic objective selection needs a measured-data prediction"
                )
            # A proposer's weight scales how much uncertainty credit its
            # candidates receive, so lowering it genuinely changes the choice
            # rather than merely annotating it.
            weight = float(weights.get(candidate.proposer, 1.0))
            bonus = uncertainty_bonus * weight * prediction.uncertainty
            return value + bonus if objective_direction == "maximize" else value - bonus

        chosen = (
            max(candidates, key=score)
            if objective_direction == "maximize"
            else min(candidates, key=score)
        )
        return Selection(
            candidate_id=chosen.candidate_id,
            reason=reason + " Selected a candidate using calibrated predictions and uncertainty scores.",
            confidence=0.45,
            risk_factors=["This trial used the deterministic numerical selector rather than a model decision."],
            used_fallback=True,
        )
