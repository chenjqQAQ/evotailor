from __future__ import annotations

import json
from typing import Any

from evotailor.workloads.discovery.workload import flatten_facts
from evotailor.spaces.search_space import SearchSpace
from evotailor.spaces.builder import adapt_space

SYSTEM_PROMPT = """Construct the search space for the next fixed stage from workload, model, hardware, framework, and reusable experience. Select relevant dimensions and ranges under the evaluation budget. Add uncovered dimensions, narrow unproductive ranges, or remove irrelevant searched dimensions only when a supplied evidence fact supports the decision. Use supported engine parameters. Never change an active stage or completed records. Removed dimensions retain predetermined deployment defaults.
Return strict JSON with project_summary (English sentences), space_changes ({add, retune, remove, evidence}), rationale, and evidence. Every changed dimension must cite exactly one available workload_evidence fact ID in space_changes.evidence. Dimension schemas use kind=choice with choices, or kind=integer/float with lower, upper, step. No evidence means no change. Keep parameter names and evidence IDs unchanged. Write all explanations in English."""


class SpaceDesignAgent:
    def __init__(self, client) -> None:
        self.client = client

    def design(
        self,
        base: SearchSpace,
        discovery: dict[str, Any],
        max_trials: int,
    ) -> tuple[SearchSpace, dict[str, Any]]:
        evidence = flatten_facts(discovery)
        payload = {
            "base_space": base.model_dump(mode="json"),
            "discovery": discovery,
            # The addressable view of the profile: the model cites these ids, and
            # the kernel refuses any citation that is not in here.
            "workload_evidence": {
                fact_id: {
                    "value": record.get("value"),
                    "grade": record.get("grade"),
                    "source": record.get("source"),
                    "note": record.get("note"),
                }
                for fact_id, record in sorted(evidence.items())
            },
            "max_trials": max_trials,
        }
        answer = self.client.complete_json(
            SYSTEM_PROMPT, json.dumps(payload, ensure_ascii=False, default=str)
        )
        for key in ("project_summary", "evidence"):
            value = answer.get(key)
            if value is None:
                answer[key] = []
            elif not isinstance(value, list):
                answer[key] = [str(value)]
            else:
                answer[key] = [str(item) for item in value]
        changes = answer.get("space_changes") or {
            "add": answer.get("extra_dimensions") or {},
            "evidence": answer.get("dimension_evidence") or {},
        }
        available = {k for k, v in evidence.items() if v.get("grade") != "unavailable"}
        extended = adapt_space(base, changes, available_evidence=available)
        citations = changes.get("evidence") or {}
        answer["dimension_evidence"] = citations
        answer["cited_facts"] = {
            name: {
                "fact_id": fact_id,
                "value": evidence[fact_id].get("value"),
                "grade": evidence[fact_id].get("grade"),
                "source": evidence[fact_id].get("source"),
            }
            for name, fact_id in citations.items()
            if fact_id in evidence
        }
        return extended, answer

    @staticmethod
    def _citations(raw: Any, extra: dict[str, Any]) -> dict[str, str]:
        """Keep only well-formed citations; the gate judges whether they resolve."""
        if not isinstance(raw, dict):
            return {}
        return {
            str(name): str(fact_id)
            for name, fact_id in raw.items()
            if name in extra and isinstance(fact_id, str) and fact_id
        }
