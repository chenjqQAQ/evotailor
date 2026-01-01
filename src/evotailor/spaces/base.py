"""Adaptive search-space primitives (paper §4.3).

A :class:`SearchSpace` is the stage-level object the inner loop samples from.  It holds
the dimensions that are actually searched, their domains, and the conditional
constraints that make a sampled configuration deployable.  The space is *fixed* for the
duration of a stage; only the outer loop may produce the next one.

Design notes
------------
* Dimensions are named after the engine flag they reach, using a dotted canonical form
  (``runtime.max_num_seqs``), so that historical samples keep their meaning when a space
  is rebuilt (see :mod:`evotailor.spaces.invariant`).
* A dimension can be *framework scoped*: ``--enable-prefix-caching`` exists in vLLM but is
  spelled differently in SGLang.  Scoping keeps one space definition usable per scenario.
* Constraints are predicates over a full assignment.  They are evaluated before a
  candidate is ever handed to the selector, so infeasible points are never charged to the
  evaluation budget.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

ParameterKind = Literal["int", "float", "bool", "enum"]

#: A constraint is ``(predicate, explanation)``.  ``predicate`` returns ``True`` when the
#: assignment is *admissible*.
Constraint = tuple[Callable[[Mapping[str, Any]], bool], str]


@dataclass(frozen=True)
class Dimension:
    """One searched parameter.

    ``domain`` is interpreted according to ``kind``:

    ==========  ==========================================================
    kind        domain
    ==========  ==========================================================
    ``int``     ``(low, high)`` or ``(low, high, step)`` — both inclusive
    ``float``   ``(low, high)`` — sampled uniformly unless ``log`` is set
    ``bool``    ignored
    ``enum``    an explicit sequence of admissible values
    ==========  ==========================================================
    """

    name: str
    kind: ParameterKind
    domain: Sequence[Any] = ()
    unit: str = ""
    log: bool = False
    frameworks: frozenset[str] = frozenset()
    #: Free-form provenance, e.g. ``{"source": "scenario", "evidence": "concurrency>=16"}``.
    rationale: Mapping[str, Any] = field(default_factory=dict)

    def supports(self, framework: str) -> bool:
        return not self.frameworks or framework in self.frameworks

    def sample(self, rng: random.Random) -> Any:
        if self.kind == "bool":
            return rng.random() < 0.5
        if self.kind == "enum":
            return rng.choice(list(self.domain))
        if self.kind == "int":
            low, high = int(self.domain[0]), int(self.domain[1])
            step = int(self.domain[2]) if len(self.domain) > 2 else 1
            return rng.randrange(low, high + 1, step)
        low, high = float(self.domain[0]), float(self.domain[1])
        if self.log and low > 0:
            import math

            return math.exp(rng.uniform(math.log(low), math.log(high)))
        return rng.uniform(low, high)

    def snap(self, value: Any) -> Any:
        """Clamp / round a proposed value into this dimension's domain."""
        if self.kind == "bool":
            return bool(value)
        if self.kind == "enum":
            values = list(self.domain)
            return value if value in values else values[0]
        low, high = self.domain[0], self.domain[1]
        value = min(max(value, low), high)
        if self.kind == "int":
            step = int(self.domain[2]) if len(self.domain) > 2 else 1
            offset = int(round((value - low) / step)) * step
            return int(min(max(low + offset, low), high))
        return float(value)


@dataclass
class SearchSpace:
    """The stage-level space ``S_k``."""

    stage_index: int
    framework: str
    dimensions: list[Dimension]
    constraints: list[Constraint] = field(default_factory=list)
    #: Human-readable note on why this space differs from the full engine surface.
    origins: list[str] = field(default_factory=list)
    version: str = "s0"

    # -- construction -------------------------------------------------------
    @classmethod
    def from_dimensions(
        cls,
        dimensions: Iterable[Dimension],
        *,
        stage_index: int = 0,
        framework: str = "vllm",
        constraints: Iterable[Constraint] = (),
        origins: Iterable[str] = (),
        version: str = "s0",
    ) -> "SearchSpace":
        dims = [d for d in dimensions if d.supports(framework)]
        return cls(
            stage_index=stage_index,
            framework=framework,
            dimensions=dims,
            constraints=list(constraints),
            origins=list(origins),
            version=version,
        )

    # -- access -------------------------------------------------------------
    @property
    def names(self) -> list[str]:
        return [d.name for d in self.dimensions]

    def dimension(self, name: str) -> Dimension:
        for d in self.dimensions:
            if d.name == name:
                return d
        raise KeyError(name)

    def __contains__(self, name: object) -> bool:
        return any(d.name == name for d in self.dimensions)

    def __len__(self) -> int:
        return len(self.dimensions)

    # -- sampling -----------------------------------------------------------
    def sample(self, rng: random.Random) -> dict[str, Any]:
        """Sample one admissible assignment (rejection sampling on constraints)."""
        for _ in range(64):
            candidate = {d.name: d.sample(rng) for d in self.dimensions}
            ok, _reason = self.admit(candidate)
            if ok:
                return candidate
        # Fall back to the centroid of the space: always admissible in practice.
        return {d.name: d.snap(d.domain[0] if d.domain else False) for d in self.dimensions}

    def project(self, configuration: Mapping[str, Any]) -> dict[str, Any]:
        """Restrict a full configuration to the dimensions of this space."""
        return {n: configuration[n] for n in self.names if n in configuration}

    # -- admissibility ------------------------------------------------------
    def admit(self, configuration: Mapping[str, Any]) -> tuple[bool, str]:
        for predicate, explanation in self.constraints:
            try:
                if not predicate(configuration):
                    return False, explanation
            except Exception as exc:  # a malformed constraint must not crash a run
                return False, f"constraint error: {exc}"
        return True, ""

    # -- evolution ----------------------------------------------------------
    def evolve(
        self,
        *,
        add: Iterable[Dimension] = (),
        drop: Iterable[str] = (),
        narrow: Mapping[str, Sequence[Any]] | None = None,
        reason: str = "",
    ) -> "SearchSpace":
        """Produce ``S_{k+1}``.

        The outer loop either adds a promising dimension, drops one that never
        discriminated, or narrows a range around the observed optimum.  Adding a
        dimension invalidates unconditional comparability with earlier samples, so the
        previous version is recorded in ``origins``.
        """
        dropped = set(drop)
        dims: list[Dimension] = []
        for d in self.dimensions:
            if d.name in dropped:
                continue
            if narrow and d.name in narrow:
                dims.append(
                    Dimension(**{**d.__dict__, "domain": tuple(narrow[d.name])})
                )
            else:
                dims.append(d)
        seen = {d.name for d in dims}
        for d in add:
            if d.name not in seen and d.supports(self.framework):
                dims.append(d)
                seen.add(d.name)
        origins = [*self.origins, f"{self.version}: {reason}"]
        return SearchSpace(
            stage_index=self.stage_index + 1,
            framework=self.framework,
            dimensions=dims,
            constraints=self.constraints,
            origins=origins,
            version=f"s{self.stage_index + 1}",
        )


@dataclass
class Stage:
    """One outer-loop stage: a space, a generator set, a policy and a code version."""

    index: int
    space: SearchSpace
    generators: list[Any] = field(default_factory=list)
    selector: Any | None = None
    code_version: str = "v0"
    budget_trials: int = 12

    def describe(self) -> dict[str, Any]:
        return {
            "stage": self.index,
            "space_version": self.space.version,
            "dimensions": self.space.names,
            "code_version": self.code_version,
            "budget_trials": self.budget_trials,
        }
