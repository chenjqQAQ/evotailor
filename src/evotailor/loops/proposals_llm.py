from __future__ import annotations

import copy
import json
import math

from evotailor.loops.proposals import Proposer, make_candidate

SYSTEM_PROMPT = """You propose the next configuration to TEST in a configuration-search experiment, not a guaranteed optimum. Return exactly one complete configuration inside the supplied frozen search space. Use all historical successes and failures, measured workload facts, hardware allocation and backend capabilities, the current phase, budget and search policy to choose an informative experiment. In modeling favor coverage and informative comparisons; in mixed/local phases balance improvement with information gain. Preserve measured/declared/inferred/unavailable evidence grades. Use prior_knowledge to spend the budget where it can still teach something: each fact carries a statement, an implication, a source and a grade, and the refusals already excluded are listed in excluded_parameters. The facts are advisory; a configuration they warn about is still a legal proposal if you can justify it. Obey every domain, step and global constraint. Source historical configurations may be remeasured on the target; they are not completed target trials. Do not repeat any excluded target or currently offered configuration. Do not change the space, objective, budgets or completed trials. Do not claim an unmeasured configuration is faster. Return JSON with parameters (all and only search-space dimensions), rationale, expected_information, and evidence_refs (a list of references to supplied trial IDs or discovery fields). Write rationale and expected_information in Simplified Chinese. Keep parameter names, numeric values, booleans and evidence references unchanged."""


class LLMProposalError(RuntimeError):
    """An enabled model proposer failed; no replacement configuration is invented."""


class LLMProposer(Proposer):
    name = "llm"

    def __init__(self, client) -> None:
        self.client = client
        self.context: dict = {}

    def set_context(
        self, *, discovery, experiment, excluded_parameters, prior_knowledge=None
    ) -> None:
        self.context = copy.deepcopy(
            dict(
                discovery=discovery,
                experiment=experiment,
                excluded_parameters=excluded_parameters,
                prior_knowledge=prior_knowledge,
            )
        )

    def propose(self, space, history, incumbent, phase, rng, policy=None):
        payload = self.context | {
            "phase": phase.value,
            "search_space": space.model_dump(mode="json"),
            "space_hash": space.digest,
            "search_policy": policy.model_dump(mode="json") if policy is not None else None,
            "incumbent": None
            if incumbent is None
            else {
                "trial_id": incumbent.trial_id,
                "objective": incumbent.objective,
                "parameters": incumbent.candidate.parameters,
            },
            "history": [
                {
                    "trial_id": t.trial_id,
                    "parameters": t.candidate.parameters,
                    "status": t.status.value,
                    "objective": t.objective,
                    "metrics": t.metrics,
                    "measurement_kind": t.measurement_kind,
                    "failure_type": t.failure_type,
                    "error": t.error,
                    "validity": t.validity,
                    "evidence": t.telemetry.get("evidence"),
                }
                for t in history
            ],
        }
        try:
            answer = self.client.complete_json(
                SYSTEM_PROMPT, json.dumps(payload, ensure_ascii=False, allow_nan=False)
            )
            values = answer["parameters"]
            if not isinstance(values, dict):
                raise ValueError("parameters must be an object")
            issues = space.validate_candidate(values)
            if issues:
                raise ValueError("; ".join(issues))
            # normalize() snaps numbers to a grid. A model proposal must pass
            # unchanged, including types, instead of silently being repaired.
            canonical = {}
            for name, dim in space.dimensions.items():
                value = values[name]
                if dim.kind in {"integer", "float"}:
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        raise ValueError(f"{name} must be numeric")
                    if not math.isfinite(value) or (dim.kind == "integer" and int(value) != value):
                        raise ValueError(f"invalid numeric value for {name}")
                    canonical[name] = int(value) if dim.kind == "integer" else float(value)
                else:
                    match = [v for v in dim.values() if type(v) is type(value) and v == value]
                    if not match:
                        raise ValueError(f"invalid choice type for {name}")
                    canonical[name] = value
            excluded = [
                t.candidate.parameters
                for t in history
                if not t.candidate.metadata.get("source_observation_id")
            ]
            excluded.extend(self.context.get("excluded_parameters", []))
            if canonical in excluded:
                raise ValueError("model proposed an already tested or offered configuration")
            rationale = answer.get("rationale")
            information = answer.get("expected_information")
            refs = answer.get("evidence_refs")
            if not isinstance(rationale, str) or not rationale.strip():
                raise ValueError("rationale must be a nonempty string")
            if not isinstance(information, str) or not information.strip():
                raise ValueError("expected_information must be a nonempty string")
            if not isinstance(refs, list) or any(not isinstance(x, str) for x in refs):
                raise ValueError("evidence_refs must be a list of strings")
            return make_candidate(
                self.name,
                canonical,
                rationale,
                implementation="llm_next_experiment",
                expected_information=information,
                evidence_refs=refs,
                model=getattr(self.client, "model", None),
                phase=phase.value,
                space_hash=space.digest,
            )
        except Exception as exc:
            raise LLMProposalError(
                f"LLM next-experiment proposal failed: {type(exc).__name__}: {exc}"
            ) from exc
