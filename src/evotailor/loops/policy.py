from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from evotailor.experiment import atomic_json

#: Proposers the kernel can ask for a candidate. A weight of zero means the
#: proposer is not asked at all, which is a hard, observable effect.
PROPOSER_NAMES = frozenset({"tpe", "slo_guard", "scoot", "coverage", "neighborhood", "llm", "llm_tuna"})

#: Tolerance for the per-Episode step limit. A patch that sits exactly on the
#: limit must be accepted, and binary floating point puts e.g. |0.85 - 0.90| a
#: hair above 0.05.
_STEP_TOLERANCE = 1e-9


class SearchPolicy(BaseModel):
    """The knobs an Episode boundary may adjust while a Campaign is running.

    Every field here is wired to something the kernel really does: phase
    boundaries decide which phase a Trial runs in, the uncertainty bonus and the
    proposer weights reach the selector, and the exploration pool and
    neighborhood size are read by the proposers at call time. A knob that no
    code reads would be a capability on paper only, so none is declared.

    What is deliberately absent matters too: the objective, the search space,
    completed Trials, and the fairness rules are not patchable, because changing
    them mid-Campaign would make the results incomparable with themselves.
    """

    version: int = Field(default=1, ge=1)
    #: Where MODELING hands over to MIXED, and MIXED to LOCAL. The search used
    #: to carry a third boundary and a trailing VALIDATION phase; that phase is
    #: no longer scheduled, so LOCAL now runs to the end of the Campaign.
    phase_boundaries: tuple[float, float] = (0.30, 0.70)
    uncertainty_bonus: float = Field(default=0.15, ge=0.0, le=1.0)
    proposer_weights: dict[str, float] = Field(
        default_factory=lambda: {name: 1.0 for name in sorted(PROPOSER_NAMES)}
    )
    #: Sample pool the coverage proposer draws from: larger explores more widely.
    exploration_pool_size: int = Field(default=96, ge=16, le=512)
    #: How many dimensions the neighborhood proposer perturbs around the incumbent.
    neighborhood_changes: int = Field(default=2, ge=1, le=8)

    @model_validator(mode="before")
    @classmethod
    def accept_retired_validation_boundary(cls, data: Any) -> Any:
        """Read the three-boundary form that earlier Campaigns recorded.

        Boundaries used to be (modeling_end, mixed_end, local_end) with a
        fourth VALIDATION phase after local_end. Retiring that phase leaves
        the third boundary with nothing left to separate, so it is dropped on
        read: the modeling -> mixed switch stays exactly where the old policy
        put it, and MIXED simply runs for longer. Stored records are never
        rewritten; only the policy loaded into memory is normalized.
        """
        if isinstance(data, dict) and "phase_boundaries" in data:
            boundaries = data["phase_boundaries"]
            if isinstance(boundaries, (list, tuple)) and len(boundaries) == 3:
                return {**data, "phase_boundaries": tuple(boundaries[:2])}
        return data

    @model_validator(mode="after")
    def validate_policy(self) -> SearchPolicy:
        first, second = self.phase_boundaries
        if not 0.0 < first < second < 1.0:
            raise ValueError("phase boundaries must be strictly increasing inside (0, 1)")
        unknown = set(self.proposer_weights) - PROPOSER_NAMES
        if unknown:
            raise ValueError(f"unknown proposer in weights: {sorted(unknown)}")
        for name, weight in self.proposer_weights.items():
            if not 0.0 <= float(weight) <= 1.0:
                raise ValueError(f"proposer weight for {name} must be inside [0, 1]")
        return self

    def active_proposers(self, configured: list[str]) -> list[str]:
        """Drop any configured proposer whose weight has been driven to zero."""
        return [name for name in configured if self.proposer_weights.get(name, 1.0) > 0.0]


class PolicyManager:
    allowed_patch_fields = frozenset(
        {
            "phase_boundaries",
            "uncertainty_bonus",
            "proposer_weights",
            "exploration_pool_size",
            "neighborhood_changes",
        }
    )

    #: The largest change a single Episode may make. The brief for an intra-run
    #: patch is "small adjustment", so a patch that would redefine the search
    #: mid-Campaign is refused outright rather than silently clamped -- clamping
    #: would hide what the agent actually asked for.
    max_step: dict[str, float] = {
        "phase_boundaries": 0.05,
        "uncertainty_bonus": 0.10,
        "proposer_weights": 0.25,
        "exploration_pool_size": 32,
        "neighborhood_changes": 1,
    }

    def __init__(self, path: Path, seed_path: Path | None = None) -> None:
        """Load the policy for a Campaign.

        ``seed_path`` is where a previous Campaign's review wrote its evolved
        defaults. It is only consulted when this Campaign has no policy file yet,
        so resuming a Campaign never silently swaps its policy underneath it.
        """
        self.path = path
        if path.exists():
            self.current = SearchPolicy.model_validate_json(path.read_text())
        elif seed_path is not None and seed_path.exists():
            self.current = SearchPolicy.model_validate_json(seed_path.read_text())
        else:
            self.current = SearchPolicy()
        self._write()

    def promote(self, patch: dict[str, Any]) -> SearchPolicy:
        unknown = set(patch) - self.allowed_patch_fields
        if unknown:
            raise ValueError(f"policy patch contains unsupported fields: {sorted(unknown)}")
        payload = self.current.model_dump()
        for field, value in patch.items():
            # A map-valued field is merged, not replaced: a patch that names one
            # proposer must not silently drop the weights of the others.
            if isinstance(value, dict) and isinstance(payload.get(field), dict):
                merged = dict(payload[field])
                merged.update(value)
                payload[field] = merged
            else:
                payload[field] = value
        payload["version"] = self.current.version + 1
        # Validate the resulting policy shape first, so a malformed patch is
        # reported as malformed rather than as "moved too far", and only then
        # apply the per-Episode magnitude limit.
        candidate = SearchPolicy.model_validate(payload)
        self._check_step(patch)
        self.current = candidate
        self._write()
        return self.current

    def _check_step(self, patch: dict[str, Any]) -> None:
        """Refuse an intra-run patch larger than the allowed per-Episode step."""
        violations: list[str] = []
        for field, value in patch.items():
            limit = self.max_step[field]
            current = getattr(self.current, field)
            if field == "phase_boundaries":
                # A patch written against the retired three-boundary form carries
                # one value more than the loaded policy holds. Only the boundaries
                # that still exist can move, so only those are compared.
                deltas = [
                    abs(float(new) - float(old))
                    for new, old in zip(tuple(value)[: len(current)], current, strict=True)
                ]
                worst = max(deltas) if deltas else 0.0
            elif field == "proposer_weights":
                # Enabling or disabling a proposer is a discrete decision, not a
                # magnitude, so a full flip is allowed in one step. Only an
                # intermediate nudge is capped.
                deltas = [
                    abs(float(weight) - float(current.get(name, 1.0)))
                    for name, weight in value.items()
                    if float(weight) not in (0.0, 1.0)
                ]
                worst = max(deltas) if deltas else 0.0
            else:
                worst = abs(float(value) - float(current))
            # A patch sitting exactly on the limit must be accepted, and binary
            # floating point puts e.g. |0.85 - 0.90| a hair above 0.05.
            if worst > limit + _STEP_TOLERANCE:
                violations.append(f"{field} moves {worst:g} but the per-Episode limit is {limit:g}")
        if violations:
            raise ValueError("an intra-run policy patch must stay small; " + "; ".join(violations))

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(self.path, self.current.model_dump(mode="json"))
