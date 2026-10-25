"""Optional controlled-space checks for explicitly declared shared-space comparisons.

This is not a universal paper requirement. Section 4.3 and the space experiment
permit scenario-adapted spaces with different dimensions and ranges. This gate
is used only when common_space_path explicitly requests the stricter legacy
superset/equality control; ordinary adaptive stages leave that field empty.
Every active stage still freezes its own space hash independently.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evotailor.spaces.search_space import SearchSpace

#: The two arms a Campaign can belong to.
BASELINE_ARMS = frozenset({"baseline"})
PORTFOLIO_ARMS = frozenset({"portfolio_agent"})


class SpaceInvariantError(RuntimeError):
    """A Campaign's space violates the S_common relationship."""


def canonical_space_hash(space: SearchSpace) -> str:
    """The frozen identity of a space: its own digest, not a file's bytes."""
    return space.digest


def _dimension_payload(space: SearchSpace, name: str) -> str:
    return json.dumps(
        space.dimensions[name].model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    )


@dataclass(frozen=True)
class SpaceRelationship:
    """The measured relationship between a Campaign space and S_common."""

    common_hash: str
    campaign_hash: str
    common_dimensions: tuple[str, ...]
    extra_dimensions: tuple[str, ...]
    removed_dimensions: tuple[str, ...]
    redefined_dimensions: tuple[str, ...]
    arm: str
    is_baseline_exact: bool
    is_superset: bool

    @property
    def ok(self) -> bool:
        if self.arm in BASELINE_ARMS:
            return self.is_baseline_exact
        if self.arm in PORTFOLIO_ARMS:
            return self.is_superset
        return False

    def as_report(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "common_space_hash": self.common_hash,
            "campaign_space_hash": self.campaign_hash,
            "common_dimensions": list(self.common_dimensions),
            "extra_dimensions": list(self.extra_dimensions),
            "removed_dimensions": list(self.removed_dimensions),
            "redefined_dimensions": list(self.redefined_dimensions),
            "baseline_space_equals_common": self.is_baseline_exact,
            "campaign_space_is_superset": self.is_superset,
            "verdict": "ok" if self.ok else "violation",
        }


def relate(
    common: SearchSpace,
    campaign: SearchSpace,
    arm: str,
) -> SpaceRelationship:
    """Measure a Campaign space against S_common without changing either."""
    common_names = set(common.dimensions)
    campaign_names = set(campaign.dimensions)
    shared = common_names & campaign_names
    removed = tuple(sorted(common_names - campaign_names))
    extra = tuple(sorted(campaign_names - common_names))
    # A shared dimension whose definition differs is a redefinition, which the
    # invariant forbids even though the name is still present.
    redefined = tuple(
        sorted(name for name in shared if _dimension_payload(common, name) != _dimension_payload(campaign, name))
    )
    exact = not removed and not extra and not redefined
    superset = not removed and not redefined
    return SpaceRelationship(
        common_hash=canonical_space_hash(common),
        campaign_hash=canonical_space_hash(campaign),
        common_dimensions=tuple(sorted(common_names)),
        extra_dimensions=extra,
        removed_dimensions=removed,
        redefined_dimensions=redefined,
        arm=arm,
        is_baseline_exact=exact,
        is_superset=superset,
    )


def enforce(
    common: SearchSpace,
    campaign: SearchSpace,
    arm: str,
    *,
    same_common_hash: str | None = None,
) -> SpaceRelationship:
    """Refuse a Campaign whose space breaks the invariant.

    ``same_common_hash`` is the S_common hash a Campaign family was opened with.
    Passing it makes "the public space did not move between campaigns" an
    enforced fact rather than an assumption.
    """
    if arm not in BASELINE_ARMS | PORTFOLIO_ARMS:
        raise SpaceInvariantError(f"unknown arm {arm!r}")
    relationship = relate(common, campaign, arm)
    if same_common_hash is not None and same_common_hash != relationship.common_hash:
        raise SpaceInvariantError(
            "S_common changed between Campaigns: "
            f"{same_common_hash} -> {relationship.common_hash}"
        )
    if arm in BASELINE_ARMS and not relationship.is_baseline_exact:
        detail = {
            "removed": relationship.removed_dimensions,
            "added": relationship.extra_dimensions,
            "redefined": relationship.redefined_dimensions,
        }
        raise SpaceInvariantError(
            f"baseline space must equal S_common exactly; differences: {detail}"
        )
    if arm in PORTFOLIO_ARMS and not relationship.is_superset:
        detail = {
            "removed": relationship.removed_dimensions,
            "redefined": relationship.redefined_dimensions,
        }
        raise SpaceInvariantError(
            f"portfolio space must be a superset of S_common; differences: {detail}"
        )
    return relationship


@dataclass
class SpaceFreezeLedger:
    """Records the one space hash a Campaign family is allowed to have.

    The point is to make "the space never changed" checkable after the fact
    instead of asserted in prose.
    """

    common_hash: str
    campaigns: dict[str, dict[str, Any]] = field(default_factory=dict)

    def record(self, campaign_id: str, relationship: SpaceRelationship) -> None:
        existing = self.campaigns.get(campaign_id)
        entry = {
            "common_space_hash": relationship.common_hash,
            "campaign_space_hash": relationship.campaign_hash,
            "extra_dimensions": list(relationship.extra_dimensions),
            "verdict": relationship.as_report()["verdict"],
        }
        if existing is not None and existing != entry:
            raise SpaceInvariantError(
                f"Campaign {campaign_id} already froze a different space: {existing} -> {entry}"
            )
        self.campaigns[campaign_id] = entry

    def assert_unchanged(self, campaign_id: str, space: SearchSpace) -> None:
        """Refuse a changed space for a Campaign that already froze one."""
        entry = self.campaigns.get(campaign_id)
        if entry is None:
            return
        if entry["campaign_space_hash"] != canonical_space_hash(space):
            raise SpaceInvariantError(
                f"Campaign {campaign_id} space changed after freeze: "
                f"{entry['campaign_space_hash']} -> {canonical_space_hash(space)}"
            )

    def digest(self) -> str:
        payload = json.dumps(
            {"common": self.common_hash, "campaigns": self.campaigns},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()


def enforce_common_space(config, root, campaign_space: SearchSpace):
    """Apply the invariant for one Campaign, or return None when it is not in use.

    Returning None is deliberate and is recorded by the caller: a Campaign with
    no declared S_common has not passed this gate, and must not be reported as
    though it had.
    """
    path_value = getattr(config, "common_space_path", "") or ""
    if not path_value:
        return None
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = Path(root) / path
    if not path.exists():
        raise SpaceInvariantError(f"S_common file does not exist: {path}")
    common = SearchSpace.load(path)
    arm = str(config.run_mode)
    return enforce(common, campaign_space, arm)


def write_common_space_report(run_dir, relationship: SpaceRelationship) -> Path:
    """Persist the gate's verdict beside the frozen space."""
    target = Path(run_dir) / "common_space_report.json"
    target.write_text(json.dumps(relationship.as_report(), indent=2, sort_keys=True) + "\n")
    return target

