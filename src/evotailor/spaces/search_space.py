from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class Dimension(BaseModel):
    kind: Literal["choice", "integer", "float"]
    choices: list[Any] | None = None
    lower: float | int | None = None
    upper: float | int | None = None
    step: float | int | None = None

    @model_validator(mode="after")
    def valid_shape(self) -> Dimension:
        if self.kind == "choice" and not self.choices:
            raise ValueError("choice dimension requires choices")
        if self.kind in {"integer", "float"} and (self.lower is None or self.upper is None):
            raise ValueError(f"{self.kind} dimension requires lower and upper")
        return self

    def values(self) -> list[Any]:
        if self.kind == "choice":
            return list(self.choices or [])
        assert self.lower is not None and self.upper is not None
        step = self.step or (
            1 if self.kind == "integer" else (float(self.upper) - float(self.lower)) / 20
        )
        count = int(round((float(self.upper) - float(self.lower)) / float(step)))
        items = [float(self.lower) + i * float(step) for i in range(count + 1)]
        return (
            [int(round(x)) for x in items]
            if self.kind == "integer"
            else [round(x, 12) for x in items]
        )


class SearchSpace(BaseModel):
    backend: str = "vllm"
    dimensions: dict[str, Dimension]
    global_constraints: list[dict[str, Any]] = Field(default_factory=list)
    safe_baseline: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> SearchSpace:
        return cls.model_validate_json(path.read_text())

    @property
    def digest(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def freeze(self, target: Path) -> str:
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.model_dump(mode="json") | {"space_hash": self.digest}
        if target.exists():
            existing = json.loads(target.read_text())
            if existing.get("space_hash") != self.digest:
                raise ValueError(
                    "the Campaign search space is already frozen with a different hash"
                )
            return self.digest
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return self.digest

    def sample(self, rng: random.Random) -> dict[str, Any]:
        for _ in range(500):
            candidate = {name: rng.choice(dim.values()) for name, dim in self.dimensions.items()}
            if not self.validate_candidate(candidate):
                return candidate
        raise RuntimeError("unable to sample a feasible candidate")

    def normalize(self, values: dict[str, Any]) -> dict[str, Any]:
        result = {}
        for name, dim in self.dimensions.items():
            if name not in values:
                raise ValueError(f"missing parameter: {name}")
            value = values[name]
            domain = dim.values()
            if value in domain:
                result[name] = value
            elif (
                dim.kind in {"integer", "float"}
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            ):
                result[name] = min(domain, key=lambda x: abs(float(x) - float(value)))
            else:
                raise ValueError(f"invalid value for {name}: {value!r}")
        unknown = set(values) - set(self.dimensions)
        if unknown:
            raise ValueError(f"unknown parameters: {sorted(unknown)}")
        issues = self.validate_candidate(result)
        if issues:
            raise ValueError("; ".join(issues))
        return result

    @staticmethod
    def _centre(dimension: Dimension) -> Any:
        """The middle value of a dimension, by position in its allowed list."""
        domain = dimension.values()
        return domain[len(domain) // 2]

    def neutralise(self, values: dict[str, Any]) -> dict[str, Any]:
        """Fit a given parameter set to this space.

        A configuration chosen in advance cannot know how the space design will resolve,
        and the design is re-run per campaign. This reconciles the two: a dimension the
        space has but the values do not gets the dimension's centre, and a dimension the
        values have but the space does not is dropped. Snapping is the same rule
        :meth:`normalize` uses, so a value already in the space is never moved.

        Without this, a recording run would depend on the design agent happening to
        produce the same dimensions twice, which it does not.
        """
        result: dict[str, Any] = {}
        for name, dimension in self.dimensions.items():
            if name in values:
                value = values[name]
                domain = dimension.values()
                if value in domain:
                    result[name] = value
                    continue
                if (
                    dimension.kind in {"integer", "float"}
                    and isinstance(value, (int, float))
                    and not isinstance(value, bool)
                ):
                    result[name] = min(domain, key=lambda x: abs(float(x) - float(value)))
                    continue
            result[name] = self._centre(dimension)
        issues = self.validate_candidate(result)
        if issues:
            raise ValueError("neutralised candidate is invalid: " + "; ".join(issues))
        return result

    def validate_candidate(self, values: dict[str, Any]) -> list[str]:
        issues = []
        if set(values) != set(self.dimensions):
            missing = set(self.dimensions) - set(values)
            unknown = set(values) - set(self.dimensions)
            if missing:
                issues.append(f"missing={sorted(missing)}")
            if unknown:
                issues.append(f"unknown={sorted(unknown)}")
            return issues
        for name, dim in self.dimensions.items():
            if values[name] not in dim.values():
                issues.append(f"{name} is outside its domain")
        for rule in self.global_constraints:
            if rule.get("kind") == "product_lte":
                if math.prod(float(values[n]) for n in rule["parameters"]) > float(rule["limit"]):
                    issues.append(str(rule.get("code", "product_lte")))
            elif rule.get("kind") == "conditional_gte":
                c = rule["if"]
                if values[c["parameter"]] == c["equals"] and float(values[rule["left"]]) < float(
                    values[rule["right"]]
                ):
                    issues.append(str(rule.get("code", "conditional_gte")))
            else:
                issues.append(f"unsupported constraint: {rule.get('kind')}")
        return issues

    def distance(self, left: dict[str, Any], right: dict[str, Any]) -> float:
        pieces = []
        for name, dim in self.dimensions.items():
            if dim.kind == "choice":
                pieces.append(0.0 if left[name] == right[name] else 1.0)
            else:
                assert dim.lower is not None and dim.upper is not None
                width = max(float(dim.upper) - float(dim.lower), 1e-12)
                pieces.append(abs(float(left[name]) - float(right[name])) / width)
        return sum(pieces) / max(len(pieces), 1)
