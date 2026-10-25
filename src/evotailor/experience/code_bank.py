"""Apply a Campaign review to the next Campaign, without a human in the loop.

The brief is that the agent may change things by itself, so nothing here asks for
approval. What replaces approval is mechanism:

* a change to the search space is validated as a search space, and a dimension
  can be added, narrowed, or removed at this stage boundary with evidence;
* a change to the policy is validated as a policy, with no per-Episode step cap,
  because it applies to a Campaign that has not started;
* a code proposal is applied to a clean tree and must pass the full test suite,
  and is reversed with its own patch when it does not;
* the root-trust layer is refused by path, and the refusal is recorded.

Every outcome, including every refusal, is written to an evolution event so the
decision trail survives the run.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from evotailor.experience.guard import WorkspaceGuard, is_root_trust, patch_paths
from evotailor.loops.policy import SearchPolicy
from evotailor.spaces.search_space import SearchSpace
from evotailor.spaces.builder import adapt_space


@dataclass
class EvolutionOutcome:
    applied: list[str] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    code: list[dict[str, Any]] = field(default_factory=list)
    next_space_path: str | None = None
    next_policy_path: str | None = None
    event_path: str | None = None


class EvolutionApplier:
    def __init__(
        self,
        repo_root: Path,
        evolution_root: Path,
        *,
        guard: WorkspaceGuard | None = None,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.evolution_root = Path(evolution_root)
        self.guard = guard or WorkspaceGuard(self.repo_root)

    # ------------------------------------------------------------------ apply

    def apply(
        self,
        plan: dict[str, Any],
        *,
        base_space: SearchSpace,
        current_policy: dict[str, Any],
        protected_dimensions: list[str],
        available_evidence: set[str] | None = None,
        campaign_id: str = "",
    ) -> EvolutionOutcome:
        outcome = EvolutionOutcome()
        self.evolution_root.mkdir(parents=True, exist_ok=True)

        self._apply_space(
            plan.get("space_changes") or {},
            base_space=base_space,
            protected=set(protected_dimensions),
            available=available_evidence or set(),
            outcome=outcome,
        )
        self._apply_policy(
            plan.get("policy_changes") or {}, current_policy=current_policy, outcome=outcome
        )
        self._apply_code(plan.get("code_proposals") or [], campaign_id=campaign_id, outcome=outcome)

        if plan.get("objective_changes"):
            # Redefining what is measured changes the meaning of every earlier
            # number, so it is recorded for a deliberate decision rather than
            # applied behind the comparison it would invalidate.
            outcome.rejected.append(
                {
                    "kind": "objective_changes",
                    "requested": plan["objective_changes"],
                    "reason": (
                        "a change to the measured objective is recorded but not applied, "
                        "because it would make this Campaign incomparable with the previous one"
                    ),
                }
            )

        outcome.event_path = str(self._write_event(plan, outcome, campaign_id))
        return outcome

    # ------------------------------------------------------------------ parts

    def _apply_space(
        self,
        changes: dict[str, Any],
        *,
        base_space: SearchSpace,
        protected: set[str],
        available: set[str],
        outcome: EvolutionOutcome,
    ) -> None:
        if not any(changes.get(key) for key in ("add", "retune", "remove")):
            return
        try:
            next_space = adapt_space(base_space, changes, available_evidence=available)
        except (TypeError, ValueError, RuntimeError, KeyError) as exc:
            outcome.rejected.append({"kind": "space_changes", "reason": str(exc)})
            return
        add = changes.get("add") or {}
        retune = changes.get("retune") or {}
        remove = changes.get("remove") or []
        path = self.evolution_root / "next_search_space.json"
        path.write_text(next_space.model_dump_json(indent=2) + "\n")
        outcome.next_space_path = str(path)
        outcome.applied.append(
            f"search space for the next Campaign written to {path.name} "
            f"(+{sorted(add)} ~{sorted(retune)} -{sorted(remove)})"
        )

    def _apply_policy(
        self, changes: dict[str, Any], *, current_policy: dict[str, Any], outcome: EvolutionOutcome
    ) -> None:
        if not changes:
            return
        payload = dict(current_policy)
        for field_name, value in changes.items():
            if isinstance(value, dict) and isinstance(payload.get(field_name), dict):
                merged = dict(payload[field_name])
                merged.update(value)
                payload[field_name] = merged
            else:
                payload[field_name] = value
        try:
            policy = SearchPolicy.model_validate(payload)
        except (TypeError, ValueError) as exc:
            outcome.rejected.append({"kind": "policy_changes", "reason": str(exc)})
            return
        path = self.evolution_root / "next_policy.json"
        path.write_text(json.dumps(policy.model_dump(mode="json"), indent=2, sort_keys=True) + "\n")
        outcome.next_policy_path = str(path)
        outcome.applied.append(f"policy defaults for the next Campaign written to {path.name}")

    def _apply_code(
        self, proposals: list[Any], *, campaign_id: str, outcome: EvolutionOutcome
    ) -> None:
        for proposal in proposals:
            if not isinstance(proposal, dict):
                continue
            patch = str(proposal.get("patch") or "")
            if not patch.strip():
                continue
            paths = patch_paths(patch)
            blocked = sorted(path for path in paths if is_root_trust(path))
            if blocked:
                outcome.code.append(
                    {
                        "path": proposal.get("path"),
                        "status": "refused",
                        "reason": f"touches the root-trust layer: {blocked}",
                    }
                )
                outcome.rejected.append(
                    {
                        "kind": "code_proposal",
                        "path": proposal.get("path"),
                        "reason": f"touches the root-trust layer: {blocked}",
                    }
                )
                continue
            verification = self.guard.apply_and_verify(
                patch,
                message=(
                    f"Apply an evolved change proposed by the Campaign review\n\n"
                    f"Campaign: {campaign_id}\n"
                    f"Rationale: {proposal.get('rationale')}\n"
                    f"Expected effect: {proposal.get('expected_effect')}"
                ),
            )
            outcome.code.append(
                {
                    "path": proposal.get("path"),
                    "status": "applied" if verification.ok else "reverted",
                    "reason": verification.detail,
                    "commit": verification.commit,
                    "paths": verification.paths,
                }
            )
            if verification.ok:
                outcome.applied.append(f"code patch committed: {verification.commit}")
            else:
                outcome.rejected.append(
                    {
                        "kind": "code_proposal",
                        "path": proposal.get("path"),
                        "reason": verification.detail,
                    }
                )

    # ------------------------------------------------------------------ event

    def _write_event(
        self, plan: dict[str, Any], outcome: EvolutionOutcome, campaign_id: str
    ) -> Path:
        existing = sorted(self.evolution_root.glob("evolution_event_*.json"))
        path = self.evolution_root / f"evolution_event_{len(existing) + 1:03d}.json"
        path.write_text(
            json.dumps(
                {
                    "campaign_id": campaign_id,
                    "plan": plan,
                    "outcome": asdict(outcome) | {"event_path": None},
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            )
            + "\n"
        )
        return path
