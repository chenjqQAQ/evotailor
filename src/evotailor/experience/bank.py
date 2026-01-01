"""Everything one finished Campaign learned, in a form the next one can use.

The intra-run review sees a single Episode. This record sees the whole Campaign:
every real Trial, the deterministic attribution evidence, the calibration trend,
the failures, the policy it ended on, and the final validation that measured the
winning configuration outside the search budget.

It also carries the workload profile, because "what improved on this workload"
is only reusable if the next Campaign can tell whether it is the same workload.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evotailor.contracts import TrialResult
from evotailor.workloads.discovery.workload import flatten_facts


def workload_signature(discovery: dict[str, Any] | None) -> dict[str, Any]:
    """The facts that say whether two Campaigns ran the same kind of workload."""
    if not discovery:
        return {"available": False, "note": "no discovery payload was supplied"}
    # Flatten the whole payload so the ids match the ones the design agent cites.
    facts = flatten_facts(discovery)
    keys = (
        "workload.scenario.kind",
        "workload.scenario.agent_count",
        "workload.scenario.max_react_turns",
        "workload.scenario.total_steps",
        "workload.arrival.shape",
        "workload.arrival.peak_concurrent_requests",
        "workload.context.model_context_limit",
        "workload.context.prompt_tokens_max",
        "workload.task.inference_stack.client_libraries",
    )
    signature = {
        key: {"value": facts[key].get("value"), "grade": facts[key].get("grade")}
        for key in keys
        if key in facts
    }
    return signature or {"available": False, "note": "the workload profile exposed no facts"}


def build_experience(
    *,
    campaign_id: str,
    run_dir: Path,
    direction: str,
    objective_metric: str,
    trials: list[TrialResult],
    evidence: dict[str, Any],
    policy: dict[str, Any],
    space_hash: str,
    discovery: dict[str, Any] | None = None,
    final_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    observed = [item for item in trials if item.objective is not None]
    return {
        "campaign_id": campaign_id,
        "measurement_kinds": {
            kind: sum(item.measurement_kind == kind for item in trials)
            for kind in sorted({item.measurement_kind for item in trials})
        },
        "run_dir": str(run_dir),
        "objective": {"metric": objective_metric, "direction": direction},
        "search_space_hash": space_hash,
        "policy_at_end": policy,
        "workload_signature": workload_signature(discovery),
        "trials": {
            "total": len(trials),
            "observed": len(observed),
            "best": min((float(x.objective) for x in observed), default=None)
            if direction == "minimize"
            else max((float(x.objective) for x in observed), default=None),
        },
        "evidence": evidence,
        "final_validation": final_validation
        or {"available": False, "note": "final validation did not run for this Campaign"},
    }


def write_experience(path: Path, record: dict[str, Any]) -> Path:
    """Append one record to the cross-Campaign experience log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, default=str) + "\n")
    return path


def read_experience(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
