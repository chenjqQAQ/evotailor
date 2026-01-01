"""Budgeted controlled measurements proposed at an Episode boundary."""

import hashlib
import json

from evotailor.contracts import Candidate
from evotailor.experiment import atomic_json


def schedule(path, proposals, history, space, remaining):
    queue = json.loads(path.read_text()) if path.exists() else []
    rejected = []
    completed = {t.candidate.metadata.get("parameter_experiment_id") for t in history}
    pending = sum(item["id"] not in completed for item in queue)
    for proposal in proposals:
        if not isinstance(proposal, dict):
            rejected.append({"proposal": proposal, "reason": "proposal must be an object"})
            continue
        ident = hashlib.sha256(json.dumps(proposal, sort_keys=True).encode()).hexdigest()[:16]
        if any(q["id"].startswith(ident + ":") for q in queue):
            continue
        dimension = proposal.get("dimension")
        reference = next(
            (t for t in history if t.trial_id == proposal.get("reference_trial")), None
        )
        reason = None
        if (
            dimension not in space.dimensions
            or proposal.get("evidence_ref") != f"evidence.dimension_effects.{dimension}"
        ):
            reason = "unknown dimension or missing evidence reference"
        elif (
            reference is None
            or reference.status != "succeeded"
            or reference.measurement_kind != "real"
        ):
            reason = "reference must be a successful real target measurement"
        elif proposal.get("value") not in space.dimensions[dimension].values():
            reason = "value outside frozen domain"
        elif proposal["value"] == reference.candidate.parameters[dimension]:
            reason = "variant must change the dimension"
        elif pending + 4 > remaining:
            reason = "insufficient budget for two repeated anchor/variant pairs"
        if reason:
            rejected.append({"proposal": proposal, "reason": reason})
            continue
        anchor = dict(reference.candidate.parameters)
        variant = anchor | {dimension: proposal["value"]}
        if space.validate_candidate(variant):
            rejected.append({"proposal": proposal, "reason": "variant violates frozen constraints"})
            continue
        for role, parameters in (
            ("anchor-01", anchor),
            ("variant-01", variant),
            ("anchor-02", anchor),
            ("variant-02", variant),
        ):
            item = {
                "id": ident + ":" + role,
                "role": role,
                "parameters": parameters,
                "proposal": proposal,
            }
            if not any(q["id"] == item["id"] for q in queue):
                queue.append(item)
        pending += 4
    atomic_json(path, queue)
    return {"pending": pending, "rejected": rejected}


def next_candidate(path, history):
    if not path.exists():
        return None
    completed = {t.candidate.metadata.get("parameter_experiment_id") for t in history}
    for item in json.loads(path.read_text()):
        if item["id"] not in completed:
            return Candidate(
                candidate_id="controlled-" + item["id"],
                proposer="parameter_validation",
                parameters=item["parameters"],
                rationale=item["proposal"].get("rationale", "Controlled real measurement"),
                metadata={
                    "parameter_experiment_id": item["id"],
                    "role": item["role"],
                    "evidence_ref": item["proposal"]["evidence_ref"],
                    "repeat": True,
                },
            )
    return None
