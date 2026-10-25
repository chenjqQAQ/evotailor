from __future__ import annotations

from typing import Any
import random

from evotailor.spaces.search_space import Dimension, SearchSpace



# The designer names a dimension's kind in prose, and writes "choices" for a choice
# dimension often enough that rejecting the whole design over the plural would discard an
# otherwise usable space. Only the spelling is repaired: a kind that is not one of the
# three real shapes is still refused, and the values themselves are never adjusted.
_KIND_ALIASES = {
    "choices": "choice",
    "categorical": "choice",
    "enum": "choice",
    "int": "integer",
    "ints": "integer",
    "integers": "integer",
    "float": "float",
    "floats": "float",
    "number": "float",
    "numeric": "float",
    "continuous": "float",
}


def _dimension(spec: dict[str, Any]) -> Dimension:
    if isinstance(spec, dict) and "kind" in spec:
        raw = spec["kind"]
        if isinstance(raw, str):
            normalised = _KIND_ALIASES.get(raw.strip().lower(), raw.strip().lower())
            spec = {**spec, "kind": normalised}
    return Dimension.model_validate(spec)

def extend_base_space(
    base: SearchSpace,
    extra_dimensions: dict[str, dict[str, Any]],
    extra_constraints: list[dict[str, Any]] | None = None,
    *,
    dimension_evidence: dict[str, str] | None = None,
    available_evidence: set[str] | None = None,
) -> SearchSpace:
    """Add dimensions without changing or removing any base dimension.

    Every added dimension must cite a workload-profile fact that really exists.
    A dimension the profile cannot justify is exactly the kind of invention this
    project must not accept, so an unresolved citation is refused rather than
    recorded and ignored. ``available_evidence`` is the set of fact ids the
    profile actually exposes; when it is empty nothing can be added, which is
    the honest outcome for a task that produced no evidence at all.
    """
    overlap = set(base.dimensions) & set(extra_dimensions)
    if overlap:
        raise ValueError(
            "Campaign discovery may add dimensions but cannot redefine base dimensions: "
            f"{sorted(overlap)}"
        )

    if extra_dimensions:
        citations = dimension_evidence or {}
        known = available_evidence or set()
        missing = sorted(name for name in extra_dimensions if not citations.get(name))
        if missing:
            raise ValueError(
                "every added dimension must cite a workload-profile fact; "
                f"these cite nothing: {missing}"
            )
        unknown = sorted(
            f"{name}->{citations[name]}"
            for name in extra_dimensions
            if citations.get(name) not in known
        )
        if unknown:
            raise ValueError(
                "added dimensions must cite an existing workload-profile fact id; "
                f"these citations do not resolve: {unknown}"
            )

    dimensions = dict(base.dimensions)
    dimensions.update(
        {name: _dimension(spec) for name, spec in extra_dimensions.items()}
    )
    baseline = dict(base.safe_baseline)
    for name, dimension in dimensions.items():
        if name not in baseline:
            baseline[name] = dimension.values()[0]
    return SearchSpace(
        backend=base.backend,
        dimensions=dimensions,
        global_constraints=[*base.global_constraints, *(extra_constraints or [])],
        safe_baseline=baseline,
        fixed_parameters=dict(base.fixed_parameters),
    )


def adapt_space(
    base: SearchSpace,
    changes: dict[str, Any],
    *,
    available_evidence: set[str],
) -> SearchSpace:
    """Construct S_(k+1) without mutating S_k (paper Section 4.3).

    Adding, narrowing, and removing searched dimensions all require explicit
    evidence. Removed dimensions retain their predetermined deployment defaults.
    Old observations are not rewritten or automatically mapped to this space.
    """
    add, retune, remove = (changes.get(key) or default for key, default in
                           (("add", {}), ("retune", {}), ("remove", [])))
    citations = changes.get("evidence") or {}
    if not isinstance(add, dict) or not isinstance(retune, dict) or not isinstance(remove, list):
        raise ValueError("add and retune must be mappings; remove must be a list")
    if set(add) & set(base.dimensions):
        raise ValueError("An added dimension already exists; use retune")
    if set(retune) - set(base.dimensions) or set(remove) - set(base.dimensions):
        raise ValueError("Retuned and removed dimensions must exist")
    if set(remove) & (set(add) | set(retune)):
        raise ValueError("A dimension cannot be removed and changed together")
    for name in set(add) | set(retune) | set(remove):
        if citations.get(name) not in available_evidence:
            raise ValueError(f"Dimension {name} needs an existing evidence reference")
    dimensions = dict(base.dimensions)
    fixed = dict(base.fixed_parameters)
    baseline = dict(base.safe_baseline)
    for name in remove:
        fixed[name] = baseline.pop(name)
        del dimensions[name]
    for name, spec in (add | retune).items():
        dimension = _dimension(spec)
        dimensions[name] = dimension
        baseline[name] = baseline.get(name, fixed.get(name))
        if baseline[name] not in dimension.values():
            baseline[name] = dimension.values()[0]
        fixed.pop(name, None)
    pending = SearchSpace.model_construct(
        backend=base.backend, dimensions=dimensions, safe_baseline=baseline,
        fixed_parameters=fixed, global_constraints=list(base.global_constraints),
    )
    if pending.validate_candidate(baseline):
        pending.safe_baseline = pending.sample(random.Random(0))
    return SearchSpace.model_validate(pending.model_dump())
