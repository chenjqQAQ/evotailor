"""Propose real target remeasurements from explicit source facts, never fill measurements."""

import hashlib
import json
import statistics

from evotailor.contracts import Candidate


def source_reference(records, space):
    result = []
    for record in records:
        parameters = record.get("candidate_parameters", {})
        common = {k: v for k, v in parameters.items() if k in space.dimensions}
        if not common:
            continue
        proposed = dict(space.safe_baseline) | common
        if space.validate_candidate(proposed):
            continue
        missing = sorted(set(space.dimensions) - set(parameters))
        result.append(
            {
                "observation_id": record["observation_id"],
                "parameters": parameters,
                "objective": record["objective"],
                "finished_at": record["finished_at"],
                "source": record.get("import_source"),
                "missing_target_dimensions": missing,
                "usage": "Actual source facts; missing values are unknown, not recorded baseline values.",
            }
        )
    return result


def seed_candidate(records, space, history, direction, limit, refused=()):
    completed = sum(t.candidate.metadata.get("source_seed", False) for t in history)
    if completed >= limit:
        return None
    excluded = [t.candidate.parameters for t in history] + [v.parameters for v in refused]
    groups = {}
    for record in source_reference(records, space):
        known = {k: v for k, v in record["parameters"].items() if k in space.dimensions}
        parameters = dict(space.safe_baseline) | known
        if parameters in excluded:
            continue
        key = json.dumps(parameters, sort_keys=True)
        groups.setdefault(key, []).append(record)
    ranked = sorted(
        groups.items(),
        key=lambda item: statistics.median(r["objective"] for r in item[1]),
        reverse=direction != "minimize",
    )
    if not ranked:
        return None
    key, rows = ranked[0]
    unknown = rows[0]["missing_target_dimensions"]
    return Candidate(
        candidate_id="source-seed-" + hashlib.sha256(key.encode()).hexdigest()[:16],
        proposer="historical_seed",
        parameters=json.loads(key),
        rationale="Remeasure a promising historical configuration on target. New dimension values are proposed, never imputed historical measurements.",
        metadata={
            "source_seed": True,
            "source_observation_ids": [r["observation_id"] for r in rows],
            "source_objective_median": statistics.median(r["objective"] for r in rows),
            "unmeasured_source_dimensions": unknown,
            "proposed_new_dimension_values": {k: space.safe_baseline[k] for k in unknown},
            "target_objective_known": False,
        },
    )
