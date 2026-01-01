"""Explicit, frozen provenance for real observations and transferable experience."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from evotailor.contracts import Candidate, TrialResult
from evotailor.experience.retrieval import cross_protocol_hypothesis, reorder_for_target
from evotailor.experiment import atomic_json
from evotailor.loops.history_search import source_reference


@dataclass(frozen=True)
class HistoryView:
    """The one history view every decision in a Campaign must read.

    Prediction, planning and candidate selection each used to assemble their own
    view, so the same Trial could be explained by a different evidence set
    depending on which component asked. This object is built exactly once per
    runner and projected into every consumer.
    """

    manifest: dict
    source_records: list[dict]
    source_trials: list[TrialResult]
    target_trials: list[TrialResult]
    experience: dict

    @property
    def learning(self) -> list[TrialResult]:
        """Source measurements followed by target measurements."""
        return [*self.source_trials, *self.target_trials]

    def context(self, space) -> dict:
        """The shared envelope handed to the predictor, planner and selector."""
        return {
            "manifest": self.manifest,
            "compatible_source_measurements": len(self.source_trials),
            "source_observations": source_reference(self.source_records, space),
            "target_observations": [
                {
                    "trial_id": t.trial_id,
                    "candidate_id": t.candidate.candidate_id,
                    "parameters": t.candidate.parameters,
                    "status": str(t.status),
                    "objective": t.objective,
                    "measurement_kind": t.measurement_kind,
                    "finished_at": t.finished_at.isoformat()
                    if hasattr(t.finished_at, "isoformat")
                    else t.finished_at,
                }
                for t in self.target_trials
            ],
            "scope": {
                "source_measurements": len(self.source_trials),
                "target_measurements": len(self.target_trials),
                "unmapped_source_records": len(self.source_records) - len(self.source_trials),
                "rule": "Both parts above are projections of this single view; "
                "prediction, planning and selection read it and nothing else.",
            },
            # The schema-3 source observations the predictor may fit on. Target
            # observations become predictor records only after the predictor's own
            # observe path has accepted them, so they are not claimed here.
            "predictor_records": list(self.source_records),
            "search_focus": "Use promising source configurations as real target starting points; prefer improvement around measured good regions under a small budget."
            if self.source_records
            else "cold exploration",
            "experience": self.experience,
            "note": "Source measurements never count as target trials or target scores.",
        }


def objective_contract(config):
    return {
        "metric": config.objective_metric,
        "direction": config.objective_direction,
        "unit": config.objective_unit,
        "protocol": config.measurement_protocol,
        "scope": config.measurement_scope,
    }


def runtime_identity(profile):
    value = {k: profile[k] for k in ("engine_config", "runtime_identity") if k in profile}
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_artifacts(paths, root):
    records, sources = [], []
    for value in paths:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = root / path
        raw = path.read_bytes()
        sources.append({"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()})
        if path.suffix == ".jsonl":
            items = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
        else:
            item = json.loads(raw)
            items = item if isinstance(item, list) else item.get("records", [item])
        for record in items:
            records.append(record | {"import_source": sources[-1]})
    return records, sources


def prepare_history(config, root, run_dir, profile):
    """Never inherit the mutable state_path of the input profile implicitly."""
    contract = objective_contract(config)
    records, sources = load_artifacts(config.history_sources, root)
    ident = runtime_identity(profile)
    accepted, excluded = [], []
    cutoff = datetime.now(UTC)
    for record in records:
        reason = None
        if record.get("schema_version") != 3 or record.get("measurement_kind") != "real":
            reason = "explicit schema-3 real observation required; use the checked importer for legacy data"
        elif record.get("objective_contract") != contract:
            reason = "objective contract differs"
        elif record.get("runtime_identity") != ident:
            reason = "runtime differs: transfer lessons, not raw target durations"
        elif record.get("status") != "succeeded" or not record.get("finished_at"):
            reason = "no completed successful observation"
        elif record.get("metrics", {}).get(contract["metric"]) != record.get("objective"):
            reason = "objective does not match measured metric"
        if reason is None:
            try:
                start = datetime.fromisoformat(record["started_at"].replace("Z", "+00:00"))
                finish = datetime.fromisoformat(record["finished_at"].replace("Z", "+00:00"))
                if start.tzinfo is None or finish.tzinfo is None or not start <= finish <= cutoff:
                    reason = "invalid or non-past source timestamps"
                elif isinstance(record["objective"], bool) or not math.isfinite(
                    float(record["objective"])
                ):
                    reason = "non-finite measured objective"
                elif not record.get("observation_id") or not record.get("iterations"):
                    reason = "missing source identity or real forward-pass observations"
            except (ValueError, TypeError, KeyError):
                reason = "malformed measured observation"
        if reason:
            excluded.append({"observation_id": record.get("observation_id"), "reason": reason})
        else:
            accepted.append(record)
    if records and not accepted and config.history_mode == "warm":
        raise ValueError("No compatible warm observations: " + json.dumps(excluded[:5]))
    seen = {}
    for record in accepted:
        identity = record["observation_id"]
        if identity in seen and {
            k: v for k, v in seen[identity].items() if k != "import_source"
        } != {k: v for k, v in record.items() if k != "import_source"}:
            raise ValueError("Conflicting source observation " + identity)
        seen[identity] = record
    accepted = list(seen.values())
    manifest = {
        "schema_version": 1,
        "mode": config.history_mode,
        "objective_contract": contract,
        "runtime_identity": ident,
        "sources": sources,
        "accepted_ids": [r["observation_id"] for r in accepted],
        "excluded": excluded,
    }
    manifest_path = run_dir / "history_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("History identity changed; use a new campaign_id")
    else:
        atomic_json(manifest_path, manifest)
        (run_dir / "observations.jsonl").write_text("".join(json.dumps(r) + "\n" for r in accepted))
    prepared = profile | {
        "state_path": str((run_dir / "observations.jsonl").resolve()),
        "objective_contract": contract,
    }
    atomic_json(run_dir / "aic_profile.json", prepared)
    return accepted, manifest


def historical_trials(records, space):
    """Real compatible source facts for planning; not target scores or budget."""
    output = []
    for record in records:
        parameters = record.get("candidate_parameters", {})
        if set(parameters) != set(space.dimensions) or space.validate_candidate(parameters):
            continue
        output.append(
            TrialResult(
                trial_id="source:" + record["observation_id"],
                candidate=Candidate(
                    candidate_id="source:" + record["observation_id"],
                    proposer="historical",
                    parameters=parameters,
                    metadata={"source_observation_id": record["observation_id"]},
                ),
                status="succeeded",
                objective=record["objective"],
                metrics=record["metrics"],
                measurement_kind="real",
                started_at=record["started_at"],
                finished_at=record["finished_at"],
                telemetry={"historical_source": record.get("import_source")},
            )
        )
    return output


def _real_provenance(record):
    """Whether this record is backed by real, observed trials.

    Two acceptable forms of the same evidence. A record may declare a
    measurement_kinds mapping in which every counted kind is real, or -- for
    records written before that field existed -- carry a trial block showing a
    nonzero count of observations. The second form is not a weaker claim: a
    record with no observed trial has no evidence to transfer at all.
    """
    kinds = record.get("measurement_kinds")
    if kinds:
        return not any(kind != "real" and count for kind, count in kinds.items())
    trials = record.get("trials") or {}
    if not isinstance(trials, dict):
        return False
    try:
        return int(trials.get("observed") or 0) > 0 and int(trials.get("total") or 0) > 0
    except (TypeError, ValueError):
        return False


def retrieve_experience(paths, root, discovery, contract, target_dimensions=()):
    """Explicit source-only retrieval. Different runtimes transfer hypotheses, not timings."""
    from evotailor.experience.bank import workload_signature

    records, sources = load_artifacts(paths, root)
    target = workload_signature(discovery)
    # The target facts a cross-protocol hypothesis is conditionalized against.
    target_context = {
        **target,
        "objective_metric": contract.get("metric"),
        "dimensions": sorted(target_dimensions),
    }
    matches, rejected, hypotheses = [], [], []
    for record in records:
        if not _real_provenance(record):
            rejected.append(
                {
                    "campaign_id": record.get("campaign_id"),
                    "reason": "non-real or missing provenance",
                }
            )
            continue
        if record.get("objective_contract") != contract:
            # A Campaign run under a different measurement protocol cannot supply
            # a target duration: the two numbers are different quantities. What it
            # can supply is the *direction* of a parameter effect, which is a
            # statement about a relation rather than a timing. That is kept in a
            # separate channel so no consumer can mistake it for a measurement,
            # and so a same-protocol match is always preferred over it.
            hypothesis = cross_protocol_hypothesis(record, target_context)
            if hypothesis["directional_hypotheses"]:
                hypotheses.append(hypothesis)
            else:
                rejected.append(
                    {
                        "campaign_id": record.get("campaign_id"),
                        "reason": "objective contract differs and no directional effect is supported",
                        "source_objective_metric": (record.get("objective_contract") or {}).get(
                            "metric"
                        ),
                    }
                )
            continue
        signature = record.get("workload_signature", {})
        common = [
            k
            for k in target
            if k in signature
            and isinstance(target[k], dict)
            and isinstance(signature[k], dict)
            and target[k].get("grade") != "unavailable"
            and signature[k].get("grade") != "unavailable"
        ]
        agreement = [k for k in common if target[k].get("value") == signature[k].get("value")]
        # Never invent similarity when the target has not been characterized.
        if not common or not agreement:
            rejected.append(
                {
                    "campaign_id": record.get("campaign_id"),
                    "reason": "no evidenced shared workload features",
                }
            )
            continue
        matches.append(
            {
                "campaign_id": record["campaign_id"],
                "source": record["import_source"],
                "similarity": len(agreement) / len(common),
                "matching_features": agreement,
                "different_features": [k for k in common if k not in agreement],
                "dimension_effects": record.get("evidence", {}).get("dimension_effects", {}),
                "lessons": record.get("lessons", []),
                "usage": "hypotheses only; validate on target before treating as a target effect",
            }
        )
    ranked = reorder_for_target(
        {name: entry for item in hypotheses for name, entry in item["directional_hypotheses"].items()},
        {
            "dimensions": target_context["dimensions"],
            "signature": target,
            "matching_features": sorted(
                {key for item in hypotheses for key in item["matching_workload_features"]}
            ),
        },
    )
    return {
        "sources": sources,
        "matches": sorted(matches, key=lambda x: -x["similarity"]),
        "hypotheses": {
            "transfer_kind": "cross-protocol-hypothesis",
            "records": hypotheses,
            "conditional_ranking": ranked,
            "rule": (
                "Directional hypotheses from Campaigns measured under a different objective "
                "protocol. No absolute duration crosses this boundary, and these must never "
                "be ranked as target measurements."
            ),
        }
        if hypotheses
        else {
            "transfer_kind": "cross-protocol-hypothesis",
            "records": [],
            "conditional_ranking": {"conditional_on": {}, "ranked": []},
            "rule": "No compatible cross-protocol source supplied directional evidence.",
        },
        "rejected": rejected,
        "target_signature": target,
    }
