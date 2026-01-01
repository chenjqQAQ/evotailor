from __future__ import annotations

import json
from typing import Any

from evotailor.workloads.discovery.workload import flatten_facts
from evotailor.spaces.search_space import SearchSpace
from evotailor.spaces.builder import extend_base_space

SYSTEM_PROMPT = """You design an initial configuration-search space before a Campaign starts. You may add any number of well-supported dimensions to the supplied base space. You must never remove, narrow, rename, or redefine a base dimension. Use only flags supported by the discovered vLLM installation.

The payload carries workload_evidence: a map of fact ids to facts measured from the upper task's own code and configuration. Each fact records its value, how it was obtained (grade: measured | declared | inferred | unavailable), and the artifact it came from.

For every dimension you add you must return dimension_evidence mapping that dimension's name to exactly one fact id from workload_evidence. A dimension whose citation is missing, or does not name a real fact id, is rejected by the kernel and the whole addition is discarded. Cite the fact that makes the dimension worth tuning: concurrency, context length, arrival shape, or call depth. Do not cite a fact whose grade is unavailable. If no fact justifies an addition, return empty additions and say so in rationale.

Return JSON with project_summary, extra_dimensions, extra_constraints, dimension_evidence, rationale, and evidence. project_summary must contain 3-6 concise Simplified Chinese sentences describing task-code, hardware, and vLLM characteristics relevant to tuning. Write rationale and evidence in Simplified Chinese. Each added dimension must use the schema {kind, choices} or {kind, lower, upper, step}. Do not make any change after trials start."""


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
        extra = answer.get("extra_dimensions") or {}
        constraints = answer.get("extra_constraints") or []
        if not isinstance(extra, dict):
            raise ValueError("space designer returned invalid extra_dimensions")
        if not isinstance(constraints, list):
            raise ValueError("space designer returned invalid constraints")
        citations = self._citations(answer.get("dimension_evidence"), extra)
        extended = extend_base_space(
            base,
            extra,
            constraints,
            dimension_evidence=citations,
            available_evidence={k for k, v in evidence.items() if v.get("grade") != "unavailable"},
        )
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
