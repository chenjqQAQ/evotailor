"""Cross-protocol experience transfer: direction only, never target durations.

A finished Campaign on a different workload reports its objective in a different
protocol (for example an image-generation Campaign reporting total deployment
plus task time). ``retrieve_experience`` refuses those records, and it must: the
two numbers are not the same quantity, so copying one as if it predicted the
other would fabricate a target measurement.

What does survive a protocol change is the *direction* of a parameter effect --
"raising this dimension lowered the objective in that Campaign" -- because it is
a statement about a relation, not about a duration. This module extracts exactly
that, and nothing else:

* only per-dimension association between observed parameter values and the
  source objective, expressed as an ordering (better/worse relative to the
  source Campaign's own median), never as seconds or any absolute magnitude;
* an explicit confidence derived from the source observation counts, so a
  dimension seen twice is not presented as a law;
* a stated source protocol, so a reader can always see the hypothesis did not
  come from the target's own measurement protocol.

No field here may carry a source objective value, a mean, a best, or any other
absolute duration. That restriction is enforced by ``_ordering_only`` and
covered by tests.
"""

from __future__ import annotations

from typing import Any

#: Keys that would leak an absolute source duration into a target decision.
FORBIDDEN_KEYS = frozenset(
    {
        "mean",
        "best",
        "worst",
        "objective",
        "objectives",
        "value_means",
        "association_gap",
        "median",
    }
)

#: Observations per value needed before a direction is offered at all.
MIN_OBSERVATIONS_PER_VALUE = 2

#: Grade vocabulary, weakest to strongest. A direction is never claimed as
#: measured on the target.
_GRADES = ("unsupported", "hinted", "supported")


def _strip_absolutes(value: Any) -> Any:
    """Remove every absolute-duration leaf, keeping decisions about ordering."""
    if isinstance(value, dict):
        return {
            key: _strip_absolutes(item)
            for key, item in value.items()
            if key not in FORBIDDEN_KEYS
        }
    if isinstance(value, list):
        return [_strip_absolutes(item) for item in value]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    # A bare number inside an effect record is an objective magnitude. The only
    # numbers this module keeps are counts, which it re-adds explicitly.
    return None


def _grade(observations: int) -> str:
    if observations >= 4:
        return "supported"
    if observations >= MIN_OBSERVATIONS_PER_VALUE:
        return "hinted"
    return "unsupported"


def dimensional_hypotheses(effects: dict, direction: str) -> dict[str, dict[str, Any]]:
    """Turn source dimension effects into orderings that survive a protocol change.

    ``effects`` is the source Campaign's ``dimension_effects`` mapping. Only the
    ordering of values by the source objective is kept. The source objective may
    have been minimized or maximized, so ``direction`` is required to interpret
    it: a value that produced a *lower* source objective is "better" only when
    the source minimized.
    """
    hypotheses: dict[str, dict[str, Any]] = {}
    for name, entry in sorted((effects or {}).items()):
        if not isinstance(entry, dict):
            continue
        value_means = entry.get("value_means") or {}
        if not isinstance(value_means, dict):
            continue
        repeated = {
            value: rows
            for value, rows in value_means.items()
            if isinstance(rows, dict) and int(rows.get("observations") or 0) >= MIN_OBSERVATIONS_PER_VALUE
        }
        # One repeated value cannot order anything against another.
        if len(repeated) < 2:
            continue

        def source_score(value: str, _rows=value_means) -> float:
            # Only used to sort; the value itself is never published.
            return float(_rows[value].get("mean") or 0.0)

        ordered = sorted(repeated, key=source_score, reverse=direction != "minimize")
        observations = sum(int(repeated[v].get("observations") or 0) for v in repeated)
        # The source's own verdict on whether a value difference can be separated
        # from run-to-run variation. Carried through because it is the strongest
        # available signal for deciding which direction deserves attention first.
        attributable = entry.get("attributable")
        hypotheses[name] = {
            "preferred_values": ordered,
            "disfavoured_values": list(reversed(ordered)),
            "source_observations": observations,
            "source_distinct_values": len(repeated),
            "source_attributable": bool(attributable) if attributable is not None else None,
            "association": "marginal ordering of values by the source objective",
            "grade": _grade(observations) if attributable else "hinted",
            "direction_basis": f"source minimized={direction == 'minimize'}",
            "caveat": (
                "Direction only. The source protocol differs from the target's, so no "
                "source duration is transferable; confirm on the target before trusting it."
            ),
        }
    return hypotheses


def cross_protocol_hypothesis(record: dict, target: dict) -> dict[str, Any]:
    """Describe what a different-protocol Campaign can and cannot contribute."""
    # Records written before `objective_contract` existed still state their
    # protocol in the `objective` block. Reading that fallback is what lets an
    # old record be labelled honestly instead of reporting a null protocol.
    source_contract = record.get("objective_contract") or {}
    stated = record.get("objective") or {}
    source_metric = source_contract.get("metric") or stated.get("metric")
    source_unit = source_contract.get("unit") or stated.get("unit")
    source_protocol = source_contract.get("protocol") or stated.get("protocol")
    source_signature = record.get("workload_signature") or {}
    direction = stated.get("direction") or source_contract.get("direction") or "minimize"
    effects = (record.get("evidence") or {}).get("dimension_effects") or {}
    hypotheses = dimensional_hypotheses(effects, direction)

    comparable = sorted(
        key
        for key in target
        if key in source_signature
        and isinstance(target[key], dict)
        and isinstance(source_signature[key], dict)
        and target[key].get("grade") != "unavailable"
        and source_signature[key].get("grade") != "unavailable"
    )
    agreed = [
        key
        for key in comparable
        if target[key].get("value") == source_signature[key].get("value")
    ]
    return {
        "campaign_id": record.get("campaign_id"),
        "source": record.get("import_source"),
        "transfer_kind": "cross-protocol-hypothesis",
        "source_objective_metric": source_metric,
        "source_objective_unit": source_unit,
        "source_objective_protocol": source_protocol,
        "source_objective_contract_recorded": record.get("objective_contract") is not None,
        "target_objective_metric": target.get("objective_metric"),
        "directional_hypotheses": hypotheses,
        "comparable_workload_features": comparable,
        "matching_workload_features": agreed,
        "workload_agreement": (len(agreed) / len(comparable)) if comparable else None,
        "grade": "hinted" if hypotheses else "unsupported",
        "not_transferable": [
            "source objective values or any absolute duration",
            "source best/mean per-value magnitudes",
            "source timing under the source protocol",
        ],
        "usage": (
            "Directional hypotheses only. Validate on the target; the source protocol "
            "differs, so its timings are not target predictions and must not be ranked as such."
        ),
    }


def project_for_mode(result: dict, mode: str) -> dict:
    """Project the retrieval result onto one of the three transfer arms.

    The arms differ *only* in how history is injected; the underlying evidence is
    the same object, so a difference between arms is a difference in injection,
    never in what was measured.

    * ``cold``   -- nothing is injected. The key is present and empty so a
      consumer cannot mistake absence for "not computed".
    * ``direct`` -- the source facts are injected verbatim: every hypothesis in
      source order, with the source's own grading untouched.
    * ``v2``     -- the same hypotheses, reordered and annotated by target
      workload signature and resource condition, so the target-expressible and
      better-supported directions come first. The hypothesis bodies are carried
      through unchanged; only order and conditioning metadata differ.
    """
    if mode not in {"cold", "direct", "v2"}:
        raise ValueError(f"unknown transfer mode {mode!r}")
    if mode == "cold":
        return {
            "transfer_mode": "cold",
            "hypotheses": {
                "transfer_kind": "cold-start",
                "records": [],
                "conditional_ranking": {"conditional_on": {}, "ranked": []},
                "rule": "Cold start: no history is injected by construction.",
            },
        }

    hypotheses = result.get("hypotheses") or {}
    records = hypotheses.get("records") or []
    if mode == "direct":
        return {
            "transfer_mode": "direct",
            "hypotheses": {
                **hypotheses,
                "rule": (
                    "Direct reuse: source hypothesis records injected as-is, in source "
                    "order. No target conditioning is applied."
                ),
                "records": records,
                "conditional_ranking": {
                    "conditional_on": {},
                    "ranked": sorted(
                        (
                            {"dimension": name, "hypothesis": entry}
                            for record in records
                            for name, entry in record["directional_hypotheses"].items()
                        ),
                        key=lambda item: item["dimension"],
                    ),
                },
            },
        }

    return {
        "transfer_mode": "v2",
        "hypotheses": {
            **hypotheses,
            "rule": (
                "V2 conditional transfer: the same source hypotheses, ranked by target "
                "workload signature and resource condition. Hypothesis bodies are unchanged; "
                "only their order and conditioning metadata differ from direct reuse."
            ),
            "records": records,
            "conditional_ranking": result["hypotheses"]["conditional_ranking"],
        },
    }


def reorder_for_target(hypotheses: dict, target: dict, resource: dict | None = None) -> dict:
    """Rank hypotheses by how well the source context matches the target's.

    This is the conditionalization step: identical inputs give an identical
    order, and a hypothesis whose dimensions the target cannot express is moved
    last rather than silently dropped.
    """
    resource = resource or {}
    target_dimensions = set(target.get("dimensions") or {})
    target_signature = target.get("signature") or {}
    gpu_count = resource.get("gpu_count")
    weight_bytes = resource.get("model_weight_bytes")
    gpu_memory = resource.get("gpu_memory_bytes")

    ranked = []
    for name, entry in hypotheses.items():
        score = 0.0
        reasons = []
        if not target_dimensions or name in target_dimensions:
            score += 2.0
            reasons.append("target search space contains this dimension")
        else:
            reasons.append("target search space does not contain this dimension")

        # Strength of the source ordering, in the source's own terms. These are
        # orderings, not magnitudes, so this ranks attention rather than speed.
        if entry.get("source_attributable") is True:
            score += 3.0
            reasons.append("source separated this value difference from noise")
        elif entry.get("grade") == "supported":
            score += 1.0
            reasons.append("source ordering backed by repeated source observations")
        distinct = int(entry.get("source_distinct_values") or 0)
        if distinct >= 2:
            score += 0.5 * min(distinct, 4)
            reasons.append(f"source observed {distinct} distinct values")

        # Resource conditions: a hypothesis that trades breadth for memory is
        # only plausible when the target is actually memory constrained.
        constraint = None
        if weight_bytes is not None and gpu_memory and gpu_count:
            per_card = float(gpu_memory) * float(gpu_count)
            constraint = "memory-bound" if float(weight_bytes) > 0.5 * per_card else "capacity-rich"
        if constraint:
            reasons.append(f"target classified as {constraint} at proposal time")

        ranked.append(
            {
                "dimension": name,
                "rank_score": round(score, 4),
                "requirements_met": bool(target_dimensions and name in target_dimensions),
                "reason": "; ".join(reasons),
                "hypothesis": entry | ({"resource_condition": constraint} if constraint else {}),
                "target_features_matched": sorted(
                    key
                    for key in target_signature
                    if key in (target.get("matching_features") or [])
                ),
            }
        )
    # Ties fall back to the source's own preferred ordering, never to alphabetical
    # order, so a reordering always reflects evidence rather than a name sort.
    ranked.sort(
        key=lambda item: (
            -item["rank_score"],
            -int(item["hypothesis"].get("source_observations") or 0),
            item["dimension"],
        )
    )
    return {
        "conditional_on": {
            "target_dimensions": sorted(target_dimensions),
            "target_workload_features": sorted(target_signature),
            "resource_condition": constraint if ranked else None,
        },
        "ranked": ranked,
    }
