from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import subprocess
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from evotailor.loops.episode_boundary import EpisodeBoundaryAgent
from evotailor.loops.episode_planner import EpisodePlannerAgent
from evotailor.loops.outer import OuterAgent
from evotailor.selection.agent import SelectorAgent
from evotailor.contracts import (
    CampaignConfig,
    Candidate,
    ExecutionBackend,
    Phase,
    Prediction,
    RunMode,
    Selection,
    TrialResult,
    TrialStatus,
)
from evotailor.experience.retrieval import project_for_mode
from evotailor.workloads.discovery.workload import flatten_facts
from evotailor.selection.evidence import build_evidence
from evotailor.experience.code_bank import EvolutionApplier
from evotailor.experience.bank import build_experience, write_experience
from evotailor.experience.guard import WorkspaceGuard
from evotailor.experience.review import CampaignReviewAgent
from evotailor.experience.source_context import source_context
from evotailor.spaces.invariant import (
    enforce_common_space,
    write_common_space_report,
)
from evotailor.execution.backends import CommandExecutor, Executor, SimulatedExecutor
from evotailor.experiment import atomic_json, freeze_experiment, serialized_campaign
from evotailor.selection.feasibility import Infeasible, weight_memory_verdict
from evotailor.loops.history import (
    historical_trials,
    objective_contract,
    prepare_history,
    retrieve_experience,
)
from evotailor.loops.history_search import seed_candidate, source_reference
from evotailor.selection.resources import CampaignResourceLease, capture_resource_snapshot
from evotailor.llm import client_from_config
from evotailor.objective import best_trial, improvement, valid_rate
from evotailor.experiment.parameter_experiments import next_candidate, schedule
from evotailor.loops.policy import PolicyManager
from evotailor.loops.proposals_llm import LLMProposalError, LLMProposer
from evotailor.loops.proposals import Proposer, default_proposers
from evotailor.selection.prediction.aiconfigurator import AIConfiguratorAdapter
from evotailor.selection.priors import (
    build_prior_knowledge,
    gpu_count,
    gpu_memory_bytes,
    model_weight_bytes,
)
from evotailor.selection.reputation import ReputationTracker
from evotailor.spaces.search_space import SearchSpace
from evotailor.storage import EventStore
from evotailor.tracing import ConsoleTrace

#: How many times the proposers may be re-asked within one Trial when every candidate
#: they offered could not start the engine. A refusal spends no budget, so the loop is
#: bounded only to surface a pathological proposer as an error instead of a hang.
MAX_PROPOSAL_ROUNDS = 50

PROPOSER_LABELS = {
    "tpe": "Optuna TPE",
    "slo_guard": "SLO-Guard",
    "scoot": "SCOOT/HEBO",
    "llm_tuna": "LLM-Tuna",
    "coverage": "Space coverage",
    "neighborhood": "Neighbourhood search around the incumbent",
    "llm": "LLM next-trial proposal",
    "validation": "Repeat validation of the best configuration",
}


PROPOSER_ACTIONS = {
    "tpe": "Fit a TPE model to previous trials and propose a candidate",
    "slo_guard": "Request a candidate from SLO-Guard",
    "scoot": "Request a candidate from SCOOT/HEBO",
    "llm_tuna": "Request a candidate from LLM-Tuna",
    "coverage": "Find the candidate with the largest coverage distance in the frozen space",
    "neighborhood": "Search for feasible candidates near the incumbent",
    "llm": "Propose a test configuration using history, workload, hardware, phase, and search-space evidence",
}


PHASE_LABELS = {
    Phase.MODELING: "Global modeling",
    Phase.MIXED: "Mixed global and local search",
    Phase.LOCAL: "Neighbourhood search around the incumbent",
    Phase.VALIDATION: "Repeat validation of the best configuration",
}


STATUS_LABELS = {
    TrialStatus.SUCCEEDED: "succeeded",
    TrialStatus.FAILED: "failed",
}


class CampaignRunner:
    def __init__(self, config: CampaignConfig, project_root: Path) -> None:
        self.config = config
        self.root = project_root
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project_root, capture_output=True, text=True, check=False)
        self.code_version = revision.stdout.strip() if revision.returncode == 0 else "unversioned"
        self.run_dir = project_root / config.run_root / config.campaign_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        # The evolution root must be shared across Campaigns, not scoped to this
        # one, or the next Campaign would never find what this one evolved.
        self.evolution_root = Path(
            config.evolution_root if config.evolution_root else Path(config.run_root) / "evolution"
        )
        if not self.evolution_root.is_absolute():
            self.evolution_root = project_root / self.evolution_root
        space_path = Path(config.search_space)
        space_path = space_path if space_path.is_absolute() else project_root / space_path
        # A previous Campaign's review may have redesigned the space and moved the
        # policy. Using them is opt-in, and the fact that they were used is
        # recorded so a result is never silently attributed to the wrong setup.
        self.evolved_space_path: Path | None = None
        self.evolved_policy_path: Path | None = None
        if config.use_evolved_defaults:
            candidate_space = self.evolution_root / "next_search_space.json"
            if candidate_space.exists():
                space_path = candidate_space
                self.evolved_space_path = candidate_space
            candidate_policy = self.evolution_root / "next_policy.json"
            if candidate_policy.exists():
                self.evolved_policy_path = candidate_policy
        self.space = SearchSpace.load(space_path)
        # The S_common relationship is checked before anything is frozen or
        # scheduled, so a violating Campaign cannot reach a Trial or a budget.
        self.space_relationship = enforce_common_space(config, self.root, self.space)
        self.trace = ConsoleTrace(self.run_dir, config.console_progress)
        freeze_experiment(config, project_root, self.run_dir, self.space.digest)
        self.store = EventStore(self.run_dir / "campaign.sqlite3")
        self.policy_manager = PolicyManager(
            self.run_dir / "policy.json", seed_path=self.evolved_policy_path
        )
        self.space.freeze(self.run_dir / "frozen_search_space.json")
        if self.space_relationship is not None:
            write_common_space_report(self.run_dir, self.space_relationship)
        if (
            config.execution_backend != ExecutionBackend.SIMULATED
            and config.run_mode == RunMode.PORTFOLIO_AGENT
            and not config.aiconfigurator_command
        ):
            raise ValueError(
                "A real execution backend requires aiconfigurator_command; "
                "the deterministic proxy is allowed only in simulated mode."
            )
        self.source_trials = []
        self.source_records = []
        self.history_manifest = {"mode": config.history_mode, "accepted_ids": []}
        prediction_environment = {}
        if config.aiconfigurator_command or config.history_sources:
            profile_value = os.environ.get("EVOTAILOR_AIC_PROFILE")
            if config.history_sources and not profile_value:
                raise ValueError(
                    "Explicit history requires EVOTAILOR_AIC_PROFILE for runtime identity"
                )
            if profile_value:
                profile_path = Path(profile_value).expanduser()
                if not profile_path.is_absolute():
                    profile_path = self.root / profile_path
                records, self.history_manifest = prepare_history(
                    config, self.root, self.run_dir, json.loads(profile_path.read_text())
                )
                self.source_records = records
                self.source_trials = historical_trials(records, self.space)
                prediction_environment["EVOTAILOR_AIC_PROFILE"] = str(
                    self.run_dir / "aic_profile.json"
                )
                self.store.append_once(
                    "HistoryLoaded",
                    self.history_manifest | {"campaign_id": config.campaign_id},
                    "campaign_id",
                )
        retrieved = retrieve_experience(
            config.experience_sources,
            self.root,
            self._selector_discovery(),
            objective_contract(config),
            target_dimensions=self.space.dimensions,
        )
        # The arm is applied here, once, so every downstream consumer sees the
        # same projection and cold/direct/v2 differ only in injection.
        self.experience_memory = project_for_mode(retrieved, config.experience_transfer_mode)
        self.experience_memory["retrieval"] = {
            "sources": retrieved["sources"],
            "same_protocol_matches": retrieved["matches"],
            "rejected": retrieved["rejected"],
        }
        atomic_json(self.run_dir / "retrieved_experience.json", self.experience_memory)
        self.aic = AIConfiguratorAdapter(
            config.aiconfigurator_command,
            environment=prediction_environment,
            allow_proxy=config.execution_backend == ExecutionBackend.SIMULATED,
        )
        self.proposers = default_proposers(
            direction=config.objective_direction,
            max_trials=config.max_trials,
            mature_optimizer_root=(self.root / Path(config.mature_optimizer_root)).resolve(),
            policies=config.mature_optimizer_policies,
        )
        self.reputation = ReputationTracker()
        self._restore_derived_state()
        client = client_from_config(config)

        self.client = client
        self.proposers["llm"] = LLMProposer(client)
        if (
            config.use_llm_selector
            or config.use_outer_agent
            or config.use_episode_planner
            or (config.run_mode == RunMode.PORTFOLIO_AGENT and "llm" in config.enabled_proposers)
        ):
            self.trace.emit(
                "Launch check",
                "Checking the model endpoint before consuming trial budget.",
                base_url=config.base_url,
                model=config.model,
            )
            connectivity = client.check_connectivity()
            self.trace.emit(
                "Launch check",
                "The model endpoint is reachable.",
                base_url=connectivity["base_url"],
                model=connectivity["model"],
                model_listed=connectivity["listed"],
            )
        self.selector = SelectorAgent(client)
        self.outer = OuterAgent(client)
        self.episode_planner = EpisodePlannerAgent(client)
        self.episode_boundary = EpisodeBoundaryAgent(client)
        self.executor = self._executor()
        #: (weights, per-GPU memory, GPU count) read once from the frozen runtime.
        self._live_resources: dict[str, Any] = {}
        self._feasibility_facts: tuple[int | None, int | None, int | None] | None = None

    def _learning_history(self, history):
        return self.source_trials + history

    def _history_context(self):
        return {
            "manifest": self.history_manifest,
            "compatible_source_measurements": len(self.source_trials),
            "source_observations": source_reference(self.source_records, self.space),
            "search_focus": "Use promising source configurations as real target starting points; prefer improvement around measured good regions under a small budget."
            if self.config.history_guided_search and self.source_records
            else "cold exploration",
            "experience": self.experience_memory,
            "note": "Source measurements never count as target trials or target scores.",
        }

    def _restore_derived_state(self) -> None:
        """Rebuild mutable statistics from durable facts when a campaign resumes."""
        for candidate in self.store.candidate_records():
            self.reputation.stats[candidate.proposer].proposed += 1
        observed_history: list[TrialResult] = []
        stored_trials = self.store.trials()
        for result in stored_trials:
            prediction = self.store.prediction(
                next(
                    i["trial_index"]
                    for i in self.store.connection.execute(
                        "SELECT trial_index FROM trials WHERE trial_id=?", (result.trial_id,)
                    )
                ),
                result.candidate.candidate_id,
            )
            if (
                self.config.run_mode == RunMode.PORTFOLIO_AGENT
                and prediction is not None
                and result.objective is not None
            ):
                prior = (
                    prediction.calibrated if self.config.aiconfigurator_command else prediction.raw
                )
                if prior is not None:
                    self.aic.calibrator.observe(prior, result.objective)
            incumbent = best_trial(observed_history, self.config.objective_direction)
            delta = 0.0
            if result.objective is not None:
                delta = improvement(
                    float(result.objective),
                    None if incumbent is None else float(incumbent.objective),
                    self.config.objective_direction,
                )
            self.reputation.observed(
                result.candidate.proposer,
                result.status == TrialStatus.SUCCEEDED,
                delta,
            )
            observed_history.append(result)
        if self.config.run_mode == RunMode.PORTFOLIO_AGENT:
            self.aic.restore_external(stored_trials)

    def _derived_parameter_arguments(self) -> dict[str, Any]:
        """Flag mappings derived for dimensions the design step added.

        The runtime config only declares mappings for the shared base space, so a
        designed dimension needs its mapping derived from the installed vLLM's own
        flag list. Without this the wrapper refuses the candidate and the Trial
        fails before a container starts.
        """
        path = self.run_dir / "derived_parameter_arguments.json"
        payload = json.loads(path.read_text()) if path.exists() else {}
        if not isinstance(payload, dict):
            raise ValueError("Derived parameter arguments must be an object")
        if self.evolved_space_path and self.config.execution_backend != ExecutionBackend.SIMULATED:
            from evotailor.spaces.engine_flags import derive_argument

            base_path = Path(self.config.search_space)
            base_path = base_path if base_path.is_absolute() else self.root / base_path
            base = SearchSpace.load(base_path)
            runtime_arguments = self._runtime_payload().get("parameter_arguments") or {}
            flags = set((self._selector_discovery().get("vllm") or {}).get("supported_flags") or [])
            for name in sorted(set(self.space.dimensions) - set(base.dimensions)):
                if name in payload or name in runtime_arguments:
                    continue
                mapping, reason = derive_argument(name, self.space.dimensions[name], flags)
                if mapping is None:
                    raise ValueError(
                        f"Evolved dimension cannot reach installed engine: {name}: {reason}"
                    )
                payload[name] = mapping | {"derived_from": reason, "startup_verified": False}
            atomic_json(path, payload)
        return payload

    def _executor(self) -> Executor:
        if self.config.execution_backend == ExecutionBackend.SIMULATED:
            return SimulatedExecutor(
                self.space,
                self.config.seed,
                self.config.simulated_noise,
                self.config.objective_direction,
            )
        return CommandExecutor(
            self.config.command,
            self.config.command_timeout_s,
            self.root,
            {
                "EVOTAILOR_CAMPAIGN_ID": self.config.campaign_id,
                "EVOTAILOR_FIXED_PARAMETERS": json.dumps(self.space.fixed_parameters),
                "EVOTAILOR_EXTRA_PARAMETER_ARGUMENTS": json.dumps(
                    self._derived_parameter_arguments()
                ),
            },
        )

    def _phase(self, index: int) -> Phase:
        ratio = index / self.config.max_trials
        if self.config.history_guided_search and self.source_records and index > 0:
            return Phase.MIXED if ratio < 0.75 else Phase.LOCAL
        modeling_end, mixed_end = self.policy_manager.current.phase_boundaries
        if ratio < modeling_end:
            return Phase.MODELING
        if ratio < mixed_end:
            return Phase.MIXED
        return Phase.LOCAL

    def _phases_by_trial(self) -> dict[str, str]:
        """Map each recorded Trial id to the phase it actually ran in.

        The phase is a durable kernel decision, so it is read back from the event
        log rather than recomputed from the current policy, which may have moved
        since the Trial ran.
        """
        phases: dict[str, str] = {}
        for event in self.store.events():
            if event.get("event_type") != "TrialPhaseDecided":
                continue
            payload = event.get("payload") or {}
            index = payload.get("trial_index")
            phase = payload.get("phase")
            if isinstance(index, int) and phase:
                phases[f"trial-{index + 1:04d}"] = str(phase)
        return phases

    def _episode_plan(self, episode: int, history: list[TrialResult]) -> dict[str, Any]:
        path = self.run_dir / f"episode_{episode:03d}_plan.json"
        episode_start = len(history)
        episode_end = min(episode_start + self.config.episode_max_trials, self.config.max_trials)
        count = episode_end - episode_start
        deterministic = [self._phase(index) for index in range(episode_start, episode_end)]
        if (
            self.config.max_trials > 1
            and self.config.final_validation_trials > 0
            and episode_end == self.config.max_trials
            and Phase.VALIDATION not in deterministic
        ):
            deterministic[-1] = Phase.VALIDATION
        if episode_start == 0:
            deterministic[0] = Phase.MODELING
        if path.exists():
            payload = json.loads(path.read_text())
            phases = payload.get("phase_plan") or []
            if not phases or len(phases) > self.config.episode_max_trials:
                raise ValueError(f"stored Episode {episode} plan has the wrong length")
            [Phase(str(item)) for item in phases]
            return payload
        payload = self.episode_planner.plan(
            episode=episode,
            episode_trials=count,
            completed_trials=len(history),
            total_trials=self.config.max_trials,
            history=history,
            proposer_evidence=self.reputation.snapshot(),
            calibration_samples=len(self.source_trials) + len(self.aic.calibrator.pairs),
            learning_context=self._history_context(),
            current_policy=self.policy_manager.current.model_dump(mode="json"),
            deterministic_plan=deterministic,
            use_llm=bool(self.config.use_episode_planner),
            allow_validation=self.config.final_validation_trials > 0,
        )
        if self.config.history_guided_search and self.source_records:
            original = list(payload["phase_plan"])
            payload["phase_plan"] = [
                Phase.MIXED.value
                if phase == Phase.MODELING.value and episode_start + i > 0
                else phase
                for i, phase in enumerate(original)
            ]
            if payload["phase_plan"] != original:
                payload["history_guided_adjustment"] = {
                    "original": original,
                    "reason": "Compatible source facts already exist; reserve only the initial target trial for modeling and use the remaining budget for improvement.",
                }
        if self.config.max_modeling_trials is not None:
            original = list(payload["phase_plan"])
            payload["phase_plan"] = [
                Phase.MIXED.value
                if phase == Phase.MODELING.value
                and episode_start + i >= self.config.max_modeling_trials
                else phase
                for i, phase in enumerate(original)
            ]
            if original != payload["phase_plan"]:
                payload["modeling_budget_adjustment"] = {
                    "original": original,
                    "limit": self.config.max_modeling_trials,
                }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        self.store.append("EpisodePlanned", {"episode": episode, "plan": payload})
        return payload

    def _best(self, history: list[TrialResult]) -> TrialResult | None:
        return best_trial(history, self.config.objective_direction, allow_simulated=self.config.execution_backend == ExecutionBackend.SIMULATED)

    def _rng(self, trial_index: int, proposer: str, attempt: int) -> random.Random:
        material = f"{self.config.seed}:{trial_index}:{proposer}:{attempt}".encode()
        seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
        return random.Random(seed)

    def _unique_candidates(
        self,
        trial_index: int,
        proposers: list[Proposer],
        history: list[TrialResult],
        phase: Phase,
        refused: list[Infeasible] | None = None,
    ) -> tuple[list, dict[str, str]]:
        seeded = seed_candidate(
            self.source_records,
            self.space,
            history,
            self.config.objective_direction,
            self.config.history_seed_trials,
            refused or [],
        )
        if seeded is not None:
            return [seeded], {}
        controlled = next_candidate(self.run_dir / "parameter_experiments.json", history)
        if controlled is not None and self.config.run_mode == RunMode.PORTFOLIO_AGENT:
            return [controlled], {}
        candidates = []
        errors: dict[str, str] = {}
        seen = {json.dumps(x.candidate.parameters, sort_keys=True) for x in history}
        # A refused configuration is as unusable as one already measured, so it joins
        # the excluded set. Without this the proposers would keep offering the same
        # fatal combination and the Campaign would never reach an attemptable one.
        for verdict in refused or []:
            seen.add(json.dumps(verdict.parameters, sort_keys=True))
        incumbent = self._best(history)
        if phase == Phase.VALIDATION and incumbent is not None:
            candidate = incumbent.candidate.model_copy(
                update={
                    "candidate_id": f"validation-{trial_index:04d}-{incumbent.candidate.candidate_id}",
                    "proposer": incumbent.candidate.proposer,
                    "rationale": "Repeat the incumbent during the validation phase.",
                    "metadata": {
                        "parent_trial": incumbent.trial_id,
                        "repeat": True,
                        "phase": Phase.VALIDATION.value,
                        "proposed_by": "validation",
                    },
                }
            )
            candidates.append(candidate)
            self.trace.emit(
                "Step 1/6: validate candidate",
                f"Repeat the incumbent candidate {candidate.candidate_id} to assess stability.",
                original_trial=incumbent.trial_id,
                candidate_parameters=candidate.parameters,
            )
        for proposer in proposers:
            if isinstance(proposer, LLMProposer):
                proposer.set_context(
                    discovery=self._selector_discovery(),
                    experiment=self._experiment_context(history),
                    excluded_parameters=[json.loads(key) for key in sorted(seen)],
                    prior_knowledge=self._prior_knowledge(history),
                )
            self.trace.emit(
                "Step 1/6: call proposer",
                f"Calling {PROPOSER_LABELS.get(proposer.name, proposer.name)}.",
                current_stage=PHASE_LABELS[phase],
                proposer_action=PROPOSER_ACTIONS.get(proposer.name, "Generate a candidate configuration"),
            )
            candidate = None
            for attempt in range(20):
                try:
                    proposed = proposer.propose(
                        self.space,
                        self._learning_history(history),
                        incumbent,
                        phase,
                        self._rng(trial_index, proposer.name, attempt),
                        self.policy_manager.current,
                    )
                except Exception as exc:
                    errors[proposer.name] = f"{type(exc).__name__}: {exc}"
                    self.trace.emit(
                        "Proposer error",
                        f"{PROPOSER_LABELS.get(proposer.name, proposer.name)} failed.",
                        error=errors[proposer.name],
                    )
                    if isinstance(exc, LLMProposalError):
                        # The model proposer is one source among several. A relay that
                        # answers 503, or a proposal the space rejects, leaves TPE, SCOOT
                        # and SLO-Guard able to continue, so it must not end a Campaign
                        # that may already hold hours of measurements. It is recorded and
                        # the round carries on without it; the round only fails when no
                        # source can offer anything, which the check below still catches.
                        self.store.append(
                            "ProposerErrors", {"trial_index": trial_index, "errors": dict(errors)}
                        )
                    break
                key = json.dumps(proposed.parameters, sort_keys=True)
                if key not in seen and not self.space.validate_candidate(proposed.parameters):
                    candidate = proposed
                    seen.add(key)
                    break
            if candidate:
                candidates.append(candidate)
                self.trace.emit(
                    "Step 1/6: candidates generated",
                    f"{PROPOSER_LABELS.get(proposer.name, proposer.name)} proposed candidate {candidate.candidate_id}.",
                    candidate_parameters=candidate.parameters,
                    proposer_metadata=candidate.metadata,
                )
            elif proposer.name not in errors:
                self.trace.emit(
                    "Step 1/6: candidate skipped",
                    f"{PROPOSER_LABELS.get(proposer.name, proposer.name)} found no new unique candidate.",
                )
        if not candidates:
            raise RuntimeError("all proposers exhausted or duplicated the known space")
        return candidates, errors

    def _dynamic_episode_state(
        self, history: list[TrialResult]
    ) -> tuple[int, int, list[TrialResult]]:
        boundaries = [e for e in self.store.events() if e["event_type"] == "EpisodeBoundaryDecided"]
        episode, start = 1, 0
        for event in boundaries:
            payload = event["payload"]
            if payload.get("decision") == "end_episode":
                episode = int(payload.get("episode", episode)) + 1
                start = int(payload.get("completed_trials", start))
        return episode, len(history) - start, history[start:]

    def _decide_episode_boundary(self, episode: int, history: list[TrialResult]) -> dict[str, Any]:
        existing = self.store.event("EpisodeBoundaryDecided", "completed_trials", len(history))
        if existing is not None:
            return existing
        _, episode_trials, current = self._dynamic_episode_state(history)
        decision = self.episode_boundary.decide(
            episode=episode,
            episode_trials=episode_trials,
            completed_trials=len(history),
            total_trials=self.config.max_trials,
            min_trials=self.config.episode_min_trials,
            max_trials=self.config.episode_max_trials,
            episode_count=episode,
            min_episodes=self.config.episode_min_count,
            max_episodes=self.config.episode_max_count,
            history=history,
            current_episode=current,
            use_llm=bool(self.config.use_episode_boundary_agent),
        )
        decision.update(
            {"episode": episode, "completed_trials": len(history), "episode_trials": episode_trials}
        )
        self.store.append("EpisodeBoundaryDecided", decision)
        self.trace.emit(
            "Episode boundary decision",
            f"Episode {episode} boundary decision: {decision['decision']}.",
            **decision,
        )
        return decision

    @serialized_campaign
    def _acquire_campaign_lease(self):
        """Take the Campaign-scoped hold on this run's GPUs and endpoint.

        Returns None for simulated and forced-candidate runs, which deploy
        nothing, and for a baseline that names no GPU.
        """
        if self.config.execution_backend == ExecutionBackend.SIMULATED:
            return None
        runtime = self._runtime_payload() or {}
        gpu_uuids = list(runtime.get("gpu_uuids") or [])
        if not gpu_uuids:
            return None
        # GPU only. The Trial wrapper takes its own lease on the GPU and the
        # endpoint while a service runs; if the Campaign also held the endpoint
        # the wrapper would block on its own Campaign's lease and every Trial
        # would fail before starting. Holding the GPU for the whole Campaign is
        # what stops two Campaigns colliding between Trials.
        lease = CampaignResourceLease(
            gpu_uuids, None, self.run_dir, self.config.campaign_id
        )
        self.trace.emit(
            "Resource lease",
            "Requesting an exclusive GPU lease for the full campaign.",
            resources=lease.resources,
        )
        lease.acquire()
        # The Trial wrapper is a subprocess and takes its own lease; it cannot
        # inherit this hold through the lock file, so it is told which resources
        # this Campaign already owns. The value is bound to the campaign id when
        # the wrapper checks it.
        os.environ["EVOTAILOR_CAMPAIGN_HELD_RESOURCES"] = json.dumps(
            {"campaign_id": self.config.campaign_id, "resources": lease.resources}
        )
        self.trace.emit(
            "Resource lease",
            "Acquired the campaign GPU lease; it remains held until validation finishes.",
            resources=lease.resources,
        )
        return lease

    def _release_campaign_lease(self):
        """Free the Campaign's GPUs; safe to call more than once."""
        lease = getattr(self, "campaign_lease", None)
        if lease is None:
            return
        lease.release()
        os.environ.pop("EVOTAILOR_CAMPAIGN_HELD_RESOURCES", None)
        self.campaign_lease = None
        self.trace.emit("Resource lease", "The campaign finished and its exclusive GPU lease was released.")

    def run(self, stop_after: int | None = None) -> dict[str, Any]:
        """Run or resume a campaign.

        ``stop_after`` is an operational checkpoint used by supervisors and tests. It
        does not change the campaign budget or phase schedule, so resuming later is
        equivalent to an uninterrupted run with the same configuration.
        """
        completed = self.store.event("CampaignCompleted", "campaign_id", self.config.campaign_id)
        if completed and self.store.event(
            "CampaignFinalized", "campaign_id", self.config.campaign_id
        ):
            self._learn_sealed_validation()
            return json.loads((self.run_dir / "summary.json").read_text())
        history = self.store.trials()
        self._recover_measured_trials(history)
        # The Campaign holds its GPUs for its whole life, not per Trial: a
        # per-Trial lease is released between Trials and lets a second Campaign
        # collide with this one's next deployment.
        self.campaign_lease = self._acquire_campaign_lease()
        start = max((int(x["trial_index"]) for x in self.store.trial_rows()), default=-1) + 1
        end = self.config.max_trials
        if stop_after is not None:
            if stop_after < 0:
                raise ValueError("stop_after must be non-negative")
            end = min(end, stop_after)
        self.trace.emit(
            "Task launch",
            "Starting or resuming configuration search.",
            task_id=self.config.campaign_id,
            run_mode=self.config.run_mode.value,
            execution_backend=self.config.execution_backend.value,
            backing_model=self.config.model
            if (
                self.config.use_llm_selector
                or self.config.use_outer_agent
                or self.config.use_episode_planner
            )
            else "disabled",
            completed_trials=len(history),
            run_until=end,
            total_rounds=self.config.max_trials,
            episode_count_range=f"{self.config.episode_min_count}-{self.config.episode_max_count}",
            episode_min_trials=self.config.episode_min_trials,
            episode_max_trials=self.config.episode_max_trials,
            episode_budget_policy="Decided by the boundary agent after each trial",
            run_dir=str(self.run_dir),
        )
        if not any(x["event_type"] == "CampaignStarted" for x in self.store.events()):
            self.store.append(
                "CampaignStarted",
                {
                    "campaign_id": self.config.campaign_id,
                    "space_hash": self.space.digest,
                    "mode": self.config.run_mode.value,
                    "backend": self.config.execution_backend.value,
                    "objective_direction": self.config.objective_direction,
                },
            )
        cached_plan: dict[str, Any] | None = None
        for index in range(start, end):
            episode, episode_trial, _ = self._dynamic_episode_state(history)
            episode_trial += 1
            phase = self._phase(index)
            if index == 0:
                phase = Phase.MODELING
            if index + 1 > self.config.max_trials - self.config.final_validation_trials and history:
                phase = Phase.VALIDATION
            if self.config.run_mode != RunMode.BASELINE and self.config.use_episode_planner:
                cached_plan = self._episode_plan(episode, history)
                if cached_plan.get("used_fallback"):
                    raise RuntimeError(cached_plan["rationale"])
                phase = Phase(cached_plan["phase_plan"][episode_trial - 1])
                if (
                    index + 1 > self.config.max_trials - self.config.final_validation_trials
                    and history
                ):
                    phase = Phase.VALIDATION
            else:
                cached_plan = {
                    "phase_plan": [phase.value],
                    "rationale": "Use the configured deterministic phase policy.",
                    "used_fallback": False,
                }
            self.trace.emit(
                "Episode plan",
                f"Trial {index + 1} uses the recorded phase decision.",
                Episode=episode,
                current_stage=PHASE_LABELS[phase],
                stage_plan=[PHASE_LABELS[Phase(value)] for value in cached_plan["phase_plan"]],
                trials_per_stage={
                    value: cached_plan["phase_plan"].count(value)
                    for value in dict.fromkeys(cached_plan["phase_plan"])
                },
                planning_rationale=cached_plan.get("rationale", ""),
                remaining_budget=self.config.max_trials - index - 1,
            )
            names = (
                []
                if phase == Phase.VALIDATION and self._best(history) is not None
                else [self.config.baseline_proposer]
                if self.config.run_mode == RunMode.BASELINE
                else self.config.enabled_proposers
            )
            # A policy may drive a proposer's weight to zero, which removes it
            # from this Trial entirely rather than merely deprioritising it.
            active = (
                list(names)
                if self.config.fixed_candidate_pool
                else self.policy_manager.current.active_proposers(list(names))
            )
            if len(active) != len(names):
                self.trace.emit(
                    "Policy applied",
                    "The current policy disabled some proposers.",
                    configured_proposers=names,
                    active_proposers=active,
                )
                names = active
            # Record which tools this Trial really asked, so the effect of a
            # policy change is a durable fact rather than an inference.
            self.store.append(
                "TrialPhaseDecided",
                {
                    "trial_index": index,
                    "episode": episode,
                    "phase": phase.value,
                    "proposers": list(names),
                    "remaining_trials": self.config.max_trials - index - 1,
                },
            )
            proposers = [self.proposers[name] for name in names]
            self.trace.emit(
                "New round",
                f"Starting trial {index + 1}/{self.config.max_trials}; phase: {PHASE_LABELS[phase]}.",
                current_episode=f"{episode}",
                episode_trial=f"{episode_trial}",
                proposers=[PROPOSER_LABELS.get(name, name) for name in names]
                or [PROPOSER_LABELS["validation"]],
            )
            candidates, predictions, selection, chosen = self._prepare_trial(
                index, proposers, history, phase
            )
            incumbent = self._best(history)
            before = incumbent.objective if incumbent and incumbent.objective is not None else None
            execution_action = (
                "SimulatedExecutor generates a simulated result without deploying a model or running a real workload."
                if self.config.execution_backend == ExecutionBackend.SIMULATED
                else "Calling the configured host or Docker trial executor."
            )
            self.trace.emit(
                "Step 4/6: execute trial",
                execution_action,
                executor_command=self.config.command
                if self.config.execution_backend != ExecutionBackend.SIMULATED
                else "SimulatedExecutor.execute",
            )
            trial_id = f"trial-{index + 1:04d}"
            prediction = next(p for p in predictions if p.candidate_id == chosen.candidate_id)
            execution = self.store.event("TrialExecutionStarted", "trial_id", trial_id)
            if execution is None:
                execution = {
                    "trial_id": trial_id,
                    "trial_index": index,
                    "attempt_id": uuid.uuid4().hex,
                    "candidate": chosen.model_dump(mode="json"),
                    "prediction": prediction.model_dump(mode="json"),
                }
                self.store.append("TrialExecutionStarted", execution)
            if isinstance(self.executor, CommandExecutor):
                self.executor.environment.update(
                    {
                        "EVOTAILOR_ATTEMPT_ID": execution["attempt_id"],
                        "EVOTAILOR_RECEIPT_PATH": str(
                            self.run_dir / "execution_receipts" / f"{trial_id}.json"
                        ),
                    }
                )
            result = self.executor.execute(trial_id, chosen)
            result.stage_id = self.config.campaign_id
            result.space_hash = self.space.digest
            result.code_version = self.code_version
            result.resource_snapshot = dict(self._live_resources)
            result.resource_snapshot["execution_hardware"] = result.telemetry.get("hardware_summary")
            result.telemetry["campaign_id"] = self.config.campaign_id
            result.telemetry["prediction_before_execution"] = prediction.model_dump(mode="json") | {
                "native_ms": prediction.diagnostics.get("native_ms")
            }
            result.telemetry["episode"] = episode
            result.telemetry["episode_trial"] = episode_trial
            result.telemetry["phase"] = phase.value
            self.store.save_trial(index, result)
            history.append(result)
            self._finish_trial(
                index, result, history, phase, episode, episode_trial, cached_plan, before
            )
        best = self._best(history)
        final_validation = (
            self._run_final_validation(best)
            if len(history) >= self.config.max_trials
            else {"available": False, "note": "Search checkpoint; final validation has not started"}
        )
        # The repeats have finished, so this Campaign no longer needs the card.
        # Releasing here rather than after the last search Trial is what makes
        # the lease cover final validation as well as search.
        self._release_campaign_lease()
        self._learn_sealed_validation()
        summary = {
            "campaign_id": self.config.campaign_id,
            "completed_trials": len(history),
            "valid_rate": valid_rate(history),
            "best": None if best is None else best.model_dump(mode="json"),
            "space_hash": self.space.digest,
            "code_version": self.code_version,
            "measurement_kind": "simulated" if self.config.execution_backend == ExecutionBackend.SIMULATED else "real",
            "proposer_evidence": self.reputation.snapshot(),
            "learning_context": self._history_context(),
            "recommendation": self._recommendation(history, final_validation),
            "calibration": {
                "method": "configuration_model_prequential_errors"
                if self.aic.command
                else self.aic.calibrator.method,
                "samples": len(self.aic.calibrator.pairs),
            },
            "policy": self.policy_manager.current.model_dump(mode="json"),
            "final_validation": final_validation,
            "episode_plans": [
                json.loads(path.read_text())
                for path in sorted(self.run_dir.glob("episode_*_plan.json"))
            ],
        }
        (self.run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, default=str) + "\n"
        )
        self.trace.emit(
            "Task finished",
            "Configuration search completed."
            if len(history) >= self.config.max_trials
            else "Saved a campaign checkpoint.",
            completed_trials=len(history),
            best_objective=None if best is None else best.objective,
            best_candidate=None if best is None else best.candidate.candidate_id,
            summary_path=str(self.run_dir / "summary.json"),
            console_log=str(self.trace.console_path),
            trace_path=str(self.trace.jsonl_path),
        )
        if len(history) >= self.config.max_trials:
            self.store.append_once("CampaignCompleted", summary, "campaign_id")
            self._evolve_after_campaign(summary, history)
            self.store.append_once(
                "CampaignFinalized", {"campaign_id": self.config.campaign_id}, "campaign_id"
            )
        else:
            self.store.append(
                "CampaignCheckpointed",
                {
                    "completed_trials": len(history),
                    "max_trials": self.config.max_trials,
                },
            )
            # A checkpointed Campaign keeps its lease only if it intends to
            # continue immediately; a stopped one must free the card.
            if stop_after is not None:
                self._release_campaign_lease()
        return summary

    def _selector_discovery(self) -> dict:
        path = (
            Path(self.config.discovery_artifact)
            if self.config.discovery_artifact
            else self.run_dir / "discovery.json"
        )
        if not path.is_absolute():
            path = self.root / path
        if not path.is_file():
            return {"available": False, "reason": "No task discovery artifact was supplied", "live_resources": getattr(self, "_live_resources", {})}
        raw = path.read_bytes()
        result = json.loads(raw)
        # All discovered task facts and hardware results are retained. Raw CLI
        # help is represented by its artifact reference; supported flags remain.
        vllm = dict(result.get("vllm") or {})
        if "help" in vllm:
            vllm.pop("help")
            vllm["help_artifact"] = {"path": str(path), "field": "vllm.help"}
        result["vllm"] = vllm
        import hashlib

        result["artifact"] = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
        result["available"] = True
        result["live_resources"] = dict(getattr(self, "_live_resources", {}))
        return result

    def _experiment_context(self, history):
        return {
            "campaign_id": self.config.campaign_id,
            "objective_metric": self.config.objective_metric,
            "objective_direction": self.config.objective_direction,
            "execution_backend": self.config.execution_backend.value,
            "fixed_candidate_pool": self.config.fixed_candidate_pool,
            "completed_trials": len(history),
            "remaining_trials": max(self.config.max_trials - len(history), 0),
            "max_trials": self.config.max_trials,
            "space_hash": self.space.digest,
            "learning_context": self._history_context(),
            "search_policy": self.policy_manager.current.model_dump(mode="json"),
        }

    # -- measured background ------------------------------------------------

    def _recommendation(self, history, final_validation):
        best = self._best(history)
        if best is None:
            return {"available": False, "reason": "No successful target measurement"}
        repeats = [t for t in history if t.candidate.parameters == best.candidate.parameters]
        values = [
            t.objective
            for t in repeats
            if t.status == TrialStatus.SUCCEEDED and t.objective is not None
        ]
        return {
            "parameters": best.candidate.parameters,
            "fastest_observed": best.objective,
            "median_observed": statistics.median(values) if values else None,
            "attempts": len(repeats),
            "succeeded": len(values),
            "available": True,
            "verification_status": "repeated_validation_available"
            if final_validation.get("available")
            else "not_independently_validated",
            "verified_stable": False,
            "final_validation": final_validation,
            "note": "Fastest is not a stability guarantee. Inspect independent validation and failures.",
        }

    def _runtime_payload(self) -> dict:
        """The frozen runtime this Campaign executes under, for the priors."""
        value = os.environ.get("EVOTAILOR_RUNTIME_CONFIG")
        if not value:
            return {}
        path = Path(value)
        if not path.is_absolute():
            path = self.root / path
        try:
            import yaml

            payload = yaml.safe_load(path.read_text())
        except Exception:  # noqa: BLE001 - priors are advisory, never fatal
            return {}
        return payload if isinstance(payload, dict) else {}

    def _experience_available(self) -> bool:
        """Whether either experience channel carries something to inject.

        Same-protocol matches are measurements; cross-protocol hypotheses are
        directions. Both are usable by a proposer, so neither may be ignored
        when deciding whether experience is worth handing over.
        """
        memory = self.experience_memory or {}
        matches = (memory.get("retrieval") or {}).get("same_protocol_matches")
        hypotheses = (memory.get("hypotheses") or {}).get("records") or []
        return bool(matches) or bool(hypotheses)

    def _prior_knowledge(self, history) -> dict | None:
        """Facts the proposers and selector may use, or None when disabled."""
        if not self.config.use_prior_knowledge:
            return (
                {"experience": self.experience_memory}
                if self._experience_available()
                else None
            )
        result = build_prior_knowledge(
            runtime=self._runtime_payload(),
            discovery=self._selector_discovery(),
            history=history,
        )
        result["experience"] = self.experience_memory
        return result

    # -- provably-fatal configurations --------------------------------------

    def _refusal_payload(self) -> tuple[int | None, int | None, int | None]:
        """The model size and allocation the feasibility check needs, read once."""
        if self._feasibility_facts is None:
            runtime = self._runtime_payload()
            discovery = self._selector_discovery()
            model_path = Path(str(runtime.get("model_path") or ""))
            self._feasibility_facts = (
                model_weight_bytes(model_path),
                gpu_memory_bytes(discovery),
                gpu_count(discovery, runtime),
            )
        return self._feasibility_facts

    def _refuse_infeasible(self, candidates: list) -> tuple[list, list[Infeasible]]:
        """Split proposals into attemptable and provably-fatal.

        A refused configuration costs no search budget, so an empty attemptable set is
        not a reason to stop searching: the refusal is recorded, handed back to the
        proposers as an excluded configuration, and they are asked again.
        """
        if not self.config.enforce_weight_memory_feasibility:
            return list(candidates), []
        if self.config.run_mode != RunMode.PORTFOLIO_AGENT:
            # The baselines must search the same space unaided.
            return list(candidates), []
        weights, memory, count = self._refusal_payload()
        if self.config.execution_backend != ExecutionBackend.SIMULATED:
            self._live_resources = capture_resource_snapshot(self._runtime_payload())
            self.store.append("ResourceSnapshot", self._live_resources)
            memory = self._live_resources.get("gpu_total_bytes", memory)
            count = self._live_resources.get("gpu_count", count)
        attemptable: list = []
        refused: list[Infeasible] = []
        for candidate in candidates:
            verdict = weight_memory_verdict(
                self.space.deployment_parameters(candidate.parameters),
                weights_bytes=weights,
                gpu_memory_bytes=memory,
                gpu_count=count,
                available_memory_bytes=self._live_resources.get("gpu_free_bytes"),
            )
            if verdict is None:
                attemptable.append(candidate)
            else:
                refused.append(verdict)
        return attemptable, refused

    def _propose_attemptable(
        self, index: int, proposers: list[Proposer], history: list[TrialResult], phase: Phase
    ) -> tuple[list, dict[str, str]]:
        """Ask the proposers until at least one candidate can actually be attempted.

        A configuration that cannot start the engine measures nothing, so it never
        becomes a Trial here. Because refusing one spends no budget, exhausting a round
        is not a failure: the refusals accumulate into the excluded set the proposers
        already respect, and the round repeats. The loop is bounded only to turn a
        pathological proposer into a loud error instead of a silent hang.
        """
        refused: list[Infeasible] = []
        for _ in range(MAX_PROPOSAL_ROUNDS):
            try:
                candidates, errors = self._unique_candidates(
                    index, proposers, history, phase, refused=refused
                )
            except RuntimeError as exc:
                if refused:
                    raise RuntimeError(
                        "every candidate the proposers could still offer is provably "
                        f"unable to start the engine; refused {len(refused)} without "
                        "spending search budget"
                    ) from exc
                raise RuntimeError(
                    "Candidate generation failed before execution; no Trial budget consumed"
                ) from exc
            attemptable, newly_refused = self._refuse_infeasible(candidates)
            if newly_refused:
                refused.extend(newly_refused)
                self.store.append(
                    "InfeasibleRefused",
                    {
                        "trial_index": index,
                        "refused": [item.as_dict() for item in newly_refused],
                        "budget_consumed": False,
                    },
                )
            if attemptable:
                if newly_refused:
                    self.trace.emit(
                        "Step 1/6: hard-constraint rejection",
                        f"Rejected {len(newly_refused)} candidates that cannot launch the engine and requested replacements; "
                        "no trial budget was consumed.",
                        rejection_codes=sorted({item.code for item in newly_refused}),
                        rejection_examples=[item.detail for item in newly_refused][:2],
                    )
                return attemptable, errors
            self.trace.emit(
                "Step 1/6: all candidates rejected",
                f"All {len(newly_refused)} candidates cannot launch the engine; proposers were notified to generate replacements.",
                rejection_codes=sorted({item.code for item in newly_refused}),
                rejection_count=len(refused),
            )
        raise RuntimeError(
            f"proposers offered only provably-fatal configurations for "
            f"{MAX_PROPOSAL_ROUNDS} rounds; refused {len(refused)} without spending budget"
        )

    # -- modeling-phase exploration ----------------------------------------

    def _maximin_selection(self, index, candidates, history, phase):
        """Choose the candidate farthest from anything measured, during modeling.

        The modeling phase exists to build a model of the space, which only works if
        the space is actually sampled. Leaving that to the selector did not hold: with
        a pool of five, one of them at max_num_seqs=256 and a fitted estimate of 394 s
        against 417 s for a low-concurrency one, the model still chose max_num_seqs=8
        and justified it as a cleaner controlled comparison. A 23-second predicted
        advantage is not evidence, and the region it defers is never measured. So during
        modeling the choice is made on coverage, which is what the phase name promises,
        and the selector decides in every other phase.
        """
        if self.config.history_guided_search and (
            self.source_records or self._experience_available()
        ):
            # Defer to the selector. Real source observations and injected
            # experience both reach the search through it; taking the choice
            # here would discard exactly the signal the arm is meant to test.
            return None
        if phase != Phase.MODELING or len(candidates) < 2:
            return None
        measured = [
            t.candidate.parameters
            for t in self._learning_history(history)
            if t.candidate is not None
        ]
        if not measured:
            # Nothing measured yet: the coverage source is maximin over the space by
            # construction, so prefer it; otherwise leave the choice to the selector.
            fallback = [c for c in candidates if c.proposer == "coverage"]
            if len(fallback) == 1:
                return Selection(
                    candidate_id=fallback[0].candidate_id,
                    reason=(
                        "No measured configuration exists in the modeling phase. The coverage candidate maximizes "
                        "the minimum distance to known points, so it is selected for the first real observation."
                    ),
                    confidence=1.0,
                    risk_factors=["No prediction exists for this configuration; this trial establishes the first measured anchor."],
                    used_fallback=True,
                )
            return None
        scored = []
        for candidate in candidates:
            distance = min(self.space.distance(candidate.parameters, params) for params in measured)
            scored.append((distance, candidate))
        scored.sort(key=lambda item: (-item[0], item[1].candidate_id))
        best_distance, chosen = scored[0]
        return Selection(
            candidate_id=chosen.candidate_id,
            reason=(
                f"Modeling prioritizes coverage: this candidate has a minimum distance of "
                f"{best_distance:.4f} to measured configurations, the largest among the candidates, so it covers an unexplored region."
                "This phase measures the wider space rather than refining the incumbent."
            ),
            confidence=1.0,
            risk_factors=["Selection uses coverage distance rather than a predicted objective; this region has no fitted estimate."],
            used_fallback=True,
        )

    def _prepare_trial(self, index, proposers, history, phase):
        selection = self.store.selection(index)
        if selection is not None:
            row = self.store.connection.execute(
                "SELECT payload FROM candidates WHERE trial_index=? AND candidate_id=?",
                (index, selection.candidate_id),
            ).fetchone()
            if row is None:
                raise ValueError("Saved selection has no candidate")
            candidate = Candidate.model_validate_json(row["payload"])
            prediction = self.store.prediction(index, candidate.candidate_id)
            if prediction is None:
                raise ValueError("Saved selection has no prediction record")
            prior = prediction.calibrated if self.config.aiconfigurator_command else prediction.raw
            if prior is not None:
                self.aic.raw_by_candidate[candidate.candidate_id] = prior
            return [candidate], [prediction], selection, candidate
        saved = self.store.connection.execute(
            "SELECT payload FROM candidates WHERE trial_index=? ORDER BY rowid", (index,)
        ).fetchall()
        if saved:
            # A selector interruption must resume its recorded pool, not create
            # a second set of proposals for the same Trial.
            candidates = [Candidate.model_validate_json(row["payload"]) for row in saved]
            predictions = [self.store.prediction(index, c.candidate_id) for c in candidates]
            if any(p is None for p in predictions):
                raise ValueError("Saved candidate pool has incomplete predictions")
            for prediction in predictions:
                prior = (
                    prediction.calibrated if self.config.aiconfigurator_command else prediction.raw
                )
                if prior is not None:
                    self.aic.raw_by_candidate[prediction.candidate_id] = prior
        elif self.config.forced_candidates:
            # Execute the configuration fixed before the run, skipping proposers and the
            # selector entirely. A recording run needs one exact configuration — the
            # centre of the search space — because a trace recorded from whatever a
            # proposer offered would make the frozen workload depend on that proposer.
            forced = self.config.forced_candidates
            if index >= len(forced):
                raise RuntimeError(
                    f"forced_candidates holds {len(forced)} entries but Trial "
                    f"{index + 1} was requested"
                )
            parameters = self.space.neutralise(dict(forced[index]))
            candidates = [
                Candidate(
                    candidate_id=f"forced-{index:04d}",
                    parameters=parameters,
                    proposer="forced",
                    rationale="The configuration fixed before the run.",
                    metadata={"phase": phase.value, "proposed_by": "forced"},
                )
            ]
            proposer_errors = []
            self.reputation.proposed([candidate.proposer for candidate in candidates])
            for candidate in candidates:
                candidate.metadata["coverage_distance"] = min(
                    (
                        self.space.distance(candidate.parameters, t.candidate.parameters)
                        for t in history
                    ),
                    default=1.0,
                )
                candidate.metadata["nearest_real_trials"] = []
            predictions = [
                Prediction(
                    candidate_id=candidate.candidate_id,
                    raw=0.0,
                    calibrated=0.0,
                    uncertainty=0.0,
                    method="forced_candidate_not_scored",
                )
                for candidate in candidates
            ]
        else:
            candidates, proposer_errors = self._propose_attemptable(
                index, proposers, history, phase
            )
            if proposer_errors:
                self.store.append(
                    "ProposerErrors", {"trial_index": index, "errors": proposer_errors}
                )
            self.reputation.proposed([x.proposer for x in candidates])
            for candidate in candidates:
                candidate.metadata["coverage_distance"] = min(
                    (
                        self.space.distance(candidate.parameters, t.candidate.parameters)
                        for t in history
                    ),
                    default=1.0,
                )
                candidate.metadata["nearest_real_trials"] = [
                    t.trial_id
                    for t in sorted(
                        history,
                        key=lambda t: self.space.distance(
                            candidate.parameters, t.candidate.parameters
                        ),
                    )[:3]
                ]
            predictions = (
                self.aic.predict_many(self.space, candidates)
                if self.config.run_mode == RunMode.PORTFOLIO_AGENT
                else [
                    Prediction(
                        candidate_id=candidate.candidate_id,
                        raw=0.0,
                        calibrated=0.0,
                        uncertainty=0.0,
                        method="baseline_not_scored",
                    )
                    for candidate in candidates
                ]
            )
        prediction_mode = (
            "AIConfigurator test proxy + residual calibration"
            if self.config.execution_backend == ExecutionBackend.SIMULATED
            else "AIConfigurator forward prediction + configuration-dependent task model"
        )
        if self.config.run_mode == RunMode.BASELINE:
            prediction_mode = "Baseline mode does not use performance prediction"
        self.trace.emit(
            "Step 2/6: performance prediction",
            f"Completed performance predictions for {len(predictions)} candidates.",
            prediction_method=prediction_mode,
            predicted_result=[
                {
                    "candidate_id": item.candidate_id,
                    "raw_prediction": item.raw,
                    "Calibrated prediction": item.calibrated,
                    "Uncertainty": item.uncertainty,
                    "calibration_samples": item.samples,
                    "prediction_method": item.method,
                }
                for item in predictions
            ],
        )
        self.store.save_candidates(index, candidates)
        self.store.save_predictions(index, predictions)
        self.trace.emit(
            "Step 3/6: agent decision",
            f"Calling model {self.config.model} to compare candidates."
            if self.config.use_llm_selector
            else "The selector has no model enabled; using the deterministic numerical selector.",
            current_stage=PHASE_LABELS[phase],
            candidate_id=[item.candidate_id for item in candidates],
            historical_trial_count=len(history),
            objective_direction="Lower objective values are better"
            if self.config.objective_direction == "minimize"
            else "Higher objective values are better",
        )
        selection = None
        if len(candidates) == 1 and candidates[0].metadata.get("source_seed"):
            selection = Selection(
                candidate_id=candidates[0].candidate_id,
                reason="Remeasure the explicit historical seed through the actual target executor; this consumes target trial budget.",
                confidence=1.0,
                risk_factors=[
                    "Source performance is not target performance; new dimension values are untested hypotheses."
                ],
            )
        if self.config.forced_candidates:
            forced_candidate = candidates[0]
            selection = Selection(
                candidate_id=forced_candidate.candidate_id,
                reason=(
                    "This run uses a fixed configuration without proposers or the selector agent."
                    "The configuration was chosen from the geometric center before recording, without using experimental results."
                ),
                confidence=1.0,
                risk_factors=["Forced configuration: this result creates a frozen trace and is not a search result."],
                used_fallback=True,
            )
        if (
            selection is None
            and len(candidates) == 1
            and candidates[0].metadata.get("parameter_experiment_id")
        ):
            selection = Selection(
                candidate_id=candidates[0].candidate_id,
                reason="Execute the recorded controlled experiment; measured through the real executor and charged to the trial budget.",
                confidence=1.0,
                risk_factors=[
                    "A single controlled pair is a hypothesis test, not proof of causality."
                ],
            )
        if selection is None and self.config.run_mode == RunMode.PORTFOLIO_AGENT:
            selection = self._maximin_selection(index, candidates, history, phase)
        if selection is None:
            selection = self.selector.select(
                phase,
                candidates,
                predictions,
                history,
                self.reputation.snapshot(),
                self.config.use_llm_selector,
                self.policy_manager.current.uncertainty_bonus,
                self.config.objective_direction,
                self.policy_manager.current.proposer_weights,
                search_space=self.space.model_dump(mode="json"),
                discovery=self._selector_discovery(),
                experiment=self._experiment_context(history),
                prior_knowledge=self._prior_knowledge(history),
            )
        self.store.save_selection(index, selection)
        chosen = next(x for x in candidates if x.candidate_id == selection.candidate_id)
        self.trace.emit(
            "Step 3/6: agent verdict",
            f"Selected candidate {chosen.candidate_id} from {PROPOSER_LABELS.get(chosen.proposer, chosen.proposer)}.",
            decision_rationale=selection.reason,
            confidence=selection.confidence,
            risk_factors=selection.risk_factors,
            fallback_used=selection.used_fallback,
            final_parameters=chosen.parameters,
        )
        return candidates, predictions, selection, chosen

    def _promote_episode_policy(self, episode: int, patch: dict):
        existing = self.store.event("PolicyPromoted", "episode", episode)
        if existing is not None:
            return self.policy_manager.current
        intent = self.store.event("PolicyPromotionStarted", "episode", episode)
        if intent is None:
            intent = {
                "episode": episode,
                "before": self.policy_manager.current.version,
                "patch": patch,
            }
            self.store.append("PolicyPromotionStarted", intent)
        if self.policy_manager.current.version == intent["before"]:
            return self.policy_manager.promote(intent["patch"])
        if self.policy_manager.current.version != intent["before"] + 1:
            raise ValueError("Policy recovery version does not match recorded transition")
        return self.policy_manager.current

    def _rebuild_reputation(self):
        self.reputation = ReputationTracker()
        for c in self.store.candidate_records():
            self.reputation.proposed([c.proposer])
        prior = []
        for result in self.store.trials():
            best = self._best(prior)
            delta = (
                improvement(
                    result.objective,
                    None if best is None else best.objective,
                    self.config.objective_direction,
                )
                if result.objective is not None
                else 0.0
            )
            self.reputation.observed(
                result.candidate.proposer, result.status == TrialStatus.SUCCEEDED, delta
            )
            prior.append(result)

    def _recover_measured_trials(self, history):
        for index, result in enumerate(history):
            if self.store.event("TrialFinalized", "trial_id", result.trial_id):
                continue
            phase = Phase(result.telemetry.get("phase", self._phase(index).value))
            episode = int(
                result.telemetry.get("episode", self._dynamic_episode_state(history[:index])[0])
            )
            episode_trial = int(
                result.telemetry.get(
                    "episode_trial", self._dynamic_episode_state(history[:index])[1] + 1
                )
            )
            best = self._best(history[:index])
            before = None if best is None else best.objective
            self._finish_trial(
                index,
                result,
                history[: index + 1],
                phase,
                episode,
                episode_trial,
                {"phase_plan": [phase.value]},
                before,
            )

    def _finish_trial(
        self, index, result, history, phase, episode, episode_trial, cached_plan, before
    ):
        chosen = result.candidate
        selection = self.store.selection(index)
        if selection is None:
            raise ValueError(f"Measured Trial {result.trial_id} has no durable selection")
        self.trace.emit(
            "Step 4/6: test result",
            f"Trial {result.trial_id} completed with status {STATUS_LABELS[result.status]}.",
            target_value=result.objective,
            metrics=result.metrics,
            error=result.error,
        )
        if (
            self.config.run_mode == RunMode.PORTFOLIO_AGENT
            and self.store.event("TrialObserved", "trial_id", result.trial_id) is None
        ):
            try:
                calibration_feedback = self.aic.observe(result)
                native_feedback = calibration_feedback.get("native") or {}
                # aic_feedback_accepted is a kernel fact: only this call knows
                # whether AIConfigurator accepted real FPM for this Trial. It
                # requires native FPM iterations, not merely a residual update.
                accepted_iterations = int(
                    native_feedback.get("accepted_iterations")
                    or native_feedback.get("recorded_iterations")
                    or 0
                )
                result.telemetry["aic_feedback_accepted"] = accepted_iterations > 0
                result.telemetry["aic_accepted_iterations"] = accepted_iterations
                result.telemetry["aic_calibration_context_key"] = native_feedback.get(
                    "calibration_context_key"
                )
                self.store.save_trial(index, result)
                self.store.append_once("TrialObserved", {"trial_id": result.trial_id}, "trial_id")
                self.trace.emit(
                    "Step 5/6: calibration feedback",
                    "Sent this trial result to the performance calibration pipeline.",
                    residual_accepted=calibration_feedback.get("residual_accepted", False),
                    accepted_fpm_iterations=native_feedback.get("accepted_iterations", 0),
                    aic_calibration_context=native_feedback.get("calibration_context_key"),
                    fpm_rejection_reason=native_feedback.get("reason"),
                )
            except Exception as exc:
                result.telemetry["aic_feedback_accepted"] = False
                self.store.save_trial(index, result)
                self.store.append(
                    "AIConfiguratorObservationFailed",
                    {
                        "trial_index": index,
                        "trial_id": result.trial_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                raise RuntimeError(
                    "Real AIC observation failed; saved Trial awaits recovery"
                ) from exc
        delta = (
            improvement(
                float(result.objective),
                None if before is None else float(before),
                self.config.objective_direction,
            )
            if result.objective is not None
            else 0.0
        )
        self._rebuild_reputation()
        self.trace.emit(
            "Step 5/6: update history",
            "Recorded the result and prediction error in history, and updated learning data and proposer statistics.",
            improvement=delta,
            best_objective=None if self._best(history) is None else self._best(history).objective,
            calibration_samples=len(self.aic.calibrator.pairs),
        )
        self.store.append_once(
            "TrialCompleted",
            {
                "trial_index": index,
                "phase": phase.value,
                "candidate_id": chosen.candidate_id,
                "proposer": chosen.proposer,
                "selection": selection.model_dump(mode="json"),
                "result": result.model_dump(mode="json"),
            },
            "trial_index",
        )
        boundary = self._decide_episode_boundary(episode, history)
        if (
            boundary["decision"] == "end_episode" or index + 1 == self.config.max_trials
        ) and self.store.event("EpisodeCompleted", "episode", episode) is None:
            episode_start = index + 1 - episode_trial
            episode_results = history[episode_start : index + 1]
            self.trace.emit(
                "Step 6/6: outer-agent review",
                f"Episode {episode} finished; calling the outer agent to review its evidence.",
                episode_trials=[item.trial_id for item in episode_results],
                use_backing_model=self.config.use_outer_agent,
            )
            # The outer agent reasons from a deterministic evidence package
            # rather than a raw reputation table, because the search does not
            # repeat configurations and attribution has to be quantified.
            evidence = build_evidence(
                history,
                direction=self.config.objective_direction,
                proposed_counts={
                    name: int(stats.get("proposed", 0))
                    for name, stats in self.reputation.snapshot().items()
                },
                phases_by_trial=self._phases_by_trial(),
                calibration_pairs=list(self.aic.calibrator.pairs),
            )
            evidence["active_proposers"] = (
                list(self.config.enabled_proposers)
                if self.config.fixed_candidate_pool
                else self.policy_manager.current.active_proposers(
                    list(self.config.enabled_proposers)
                )
            )
            evidence["fixed_candidate_pool"] = self.config.fixed_candidate_pool
            (self.run_dir / f"episode_{episode:03d}_evidence.json").write_text(
                json.dumps(evidence, indent=2, default=str) + "\n"
            )
            review_path = self.run_dir / f"episode_{episode:03d}_review.json"
            review = (
                json.loads(review_path.read_text())
                if review_path.exists()
                else self.outer.review(
                    episode,
                    episode_results,
                    evidence,
                    len(self.aic.calibrator.pairs),
                    self.config.use_outer_agent,
                    self.config.objective_direction,
                    self.policy_manager.current.model_dump(mode="json"),
                    len(history),
                    self.config.max_trials,
                    list(cached_plan["phase_plan"]),
                )
            )
            (self.run_dir / f"episode_{episode:03d}_review.json").write_text(
                json.dumps(review, indent=2, default=str) + "\n"
            )
            self.trace.emit(
                "Step 6/6: reflection complete",
                f"The outer agent completed the review of Episode {episode}.",
                facts=review.get("facts", []),
                observations=review.get("observations", []),
                hypotheses=review.get("hypotheses", []),
                proposed_changes=review.get("proposed_changes", []),
                validation_plan=review.get("validation_plan", []),
                policy_patch=review.get("policy_patch", {}),
                no_change_reason=review.get("no_change_reason"),
                model_error=review.get("review_error"),
            )
            proposals = (
                (review.get("parameter_experiments") or [])
                if self.config.enable_parameter_experiments
                else []
            )
            if proposals and self.config.run_mode == RunMode.PORTFOLIO_AGENT:
                outcome = schedule(
                    self.run_dir / "parameter_experiments.json",
                    proposals,
                    history,
                    self.space,
                    self.config.max_trials - len(history),
                )
                self.store.append("ParameterExperimentsScheduled", outcome)
            patch = review.get("policy_patch", {})
            if self.config.auto_promote_policy and isinstance(patch, dict) and patch:
                try:
                    promoted = self._promote_episode_policy(episode, patch)
                    self.store.append(
                        "PolicyPromoted",
                        {"episode": episode, "policy": promoted.model_dump(mode="json")},
                    )
                    self.trace.emit(
                        "Policy evolution",
                        f"Applied the policy changes for Episode {episode}.",
                        policy_version=promoted.version,
                        Change=patch,
                        full_policy=promoted.model_dump(mode="json"),
                    )
                except (TypeError, ValueError) as exc:
                    self.store.append(
                        "PolicyRejected",
                        {"episode": episode, "patch": patch, "reason": str(exc)},
                    )
                    self.trace.emit(
                        "Policy evolution failed",
                        f"Policy changes for Episode {episode} failed validation.",
                        Change=patch,
                        Reason=str(exc),
                    )
            else:
                self.trace.emit(
                    "Policy unchanged",
                    f"The Episode {episode} review left the search policy unchanged.",
                    Reason=review.get("no_change_reason") or "The outer agent found insufficient evidence for a change.",
                    current_policy=self.policy_manager.current.model_dump(mode="json"),
                )
            self.store.append(
                "EpisodeCompleted",
                {
                    "episode": episode,
                    "review": review,
                    "policy_version": self.policy_manager.current.version,
                },
            )
        self.store.append_once("TrialFinalized", {"trial_id": result.trial_id}, "trial_id")

    def _learn_sealed_validation(self) -> None:
        """Admit held-out measurements for future prediction only after sealing the report."""
        if self.config.run_mode != RunMode.PORTFOLIO_AGENT or not self.aic.command:
            return
        if not (self.run_dir / "final_validation.json").is_file():
            return
        for path in sorted((self.run_dir / "final_validation_trials").glob("*.json")):
            result = TrialResult.model_validate_json(path.read_text())
            if self.store.event("FinalValidationObserved", "trial_id", result.trial_id):
                continue
            feedback = self.aic.observe(result)
            self.store.append_once(
                "FinalValidationObserved",
                {"trial_id": result.trial_id, "feedback": feedback},
                "trial_id",
            )

    # ------------------------------------------------------- final validation

    def _run_final_validation(self, best: TrialResult | None) -> dict[str, Any]:
        """Re-measure each arm's final configuration outside the search budget.

        The search deliberately does not repeat a configuration, so one
        measurement of the winning point cannot separate a real improvement from
        run-to-run variation -- the same algorithm has been observed to differ by
        a factor of two between runs. Each arm is therefore measured the same
        number of times and the arms alternate, so host drift cannot favour one
        of them. These Trials are verification, not search, so they do not consume
        the search budget and they do not feed the search policy.
        """
        repeats = self.config.final_validation_repeats
        if repeats <= 0:
            return {
                "available": False,
                "note": "final_validation_repeats is 0, so no verification was run",
            }

        arms: list[tuple[str, dict[str, Any]]] = []
        for name in self.config.final_validation_arms:
            if name == "best":
                if best is None:
                    continue
                arms.append((name, dict(best.candidate.parameters)))
            elif name == "baseline":
                arms.append((name, dict(self.space.safe_baseline)))
            else:
                self.trace.emit("Final validation", f"Skipped unknown validation arm {name}.")
        if not arms:
            return {
                "available": False,
                "note": "no validation arm could be resolved for this Campaign",
            }

        schedule = [
            (name, parameters, repeat + 1) for repeat in range(repeats) for name, parameters in arms
        ]
        self.trace.emit(
            "Final validation",
            "Repeating each arm's final configuration for validation; these trials are outside the search budget.",
            repeats=repeats,
            validation_arms=[name for name, _ in arms],
            validation_schedule=[f"{name}#{repeat}" for name, _, repeat in schedule],
            search_budget=self.config.max_trials,
        )

        records: list[dict[str, Any]] = []
        for name, parameters, repeat in schedule:
            trial_id = f"final-{name}-{repeat:02d}"
            candidate = Candidate(
                candidate_id=trial_id,
                proposer="validation",
                parameters=parameters,
                rationale=f"Repeat the {name} configuration to verify the final result.",
                metadata={
                    "arm": name,
                    "repeat": repeat,
                    "outside_search_budget": True,
                    "phase": Phase.VALIDATION.value,
                },
            )
            path = self.run_dir / "final_validation_trials" / f"{trial_id}.json"
            if path.exists():
                result = TrialResult.model_validate_json(path.read_text())
                if result.candidate.parameters != parameters:
                    raise ValueError("Final validation configuration changed")
            else:
                if isinstance(self.executor, CommandExecutor):
                    self.executor.environment.update(
                        {
                            "EVOTAILOR_ATTEMPT_ID": trial_id,
                            "EVOTAILOR_RECEIPT_PATH": str(
                                self.run_dir / "execution_receipts" / f"{trial_id}.json"
                            ),
                        }
                    )
                result = self.executor.execute(trial_id, candidate)
                result.telemetry["campaign_id"] = self.config.campaign_id
                atomic_json(path, result.model_dump(mode="json"))
            self.store.append_once(
                "FinalValidationTrial",
                {
                    "trial_id": trial_id,
                    "arm": name,
                    "repeat": repeat,
                    "objective": result.objective,
                    "status": result.status.value,
                    "failure_type": result.failure_type,
                },
                "trial_id",
            )
            records.append(
                {
                    "trial_id": trial_id,
                    "arm": name,
                    "repeat": repeat,
                    "objective": result.objective,
                    "status": result.status.value,
                    "failure_type": result.failure_type,
                    "metrics": result.metrics,
                }
            )
            self.trace.emit(
                "Final validation",
                f"Validation trial {trial_id} completed with status {STATUS_LABELS[result.status]}.",
                validation_arms=name,
                repeat_index=repeat,
                target_value=result.objective,
                error=result.error,
            )

        summary: dict[str, Any] = {
            "available": True,
            "repeats": repeats,
            "counted_in_search_budget": False,
            "objective_metric": self.config.objective_metric,
            "objective_direction": self.config.objective_direction,
            "schedule": [f"{name}#{repeat}" for name, _, repeat in schedule],
            "arms": {},
            "trials": records,
        }
        for name, _ in arms:
            values = [
                float(item["objective"])
                for item in records
                if item["arm"] == name
                and item["objective"] is not None
                and item["status"] == "succeeded"
            ]
            summary["arms"][name] = self._arm_statistics(values, len(arms) and repeats)
        path = self.run_dir / "final_validation.json"
        atomic_json(path, summary)
        summary["path"] = str(path)
        self.trace.emit(
            "Final validation",
            "Repeated validation of the final configurations completed.",
            result_path=str(path),
            arm_results={
                name: {
                    "count": stats["observations"],
                    "mean": stats["mean"],
                    "stdev": stats["stdev"],
                }
                for name, stats in summary["arms"].items()
            },
        )
        return summary

    @staticmethod
    def _arm_statistics(values: list[float], attempts: int) -> dict[str, Any]:
        if not values:
            return {
                "attempts": attempts,
                "observations": 0,
                "mean": None,
                "stdev": None,
                "note": "no attempt produced a measurement, so no statistic is reported",
            }
        return {
            "attempts": attempts,
            "observations": len(values),
            "mean": round(statistics.fmean(values), 6),
            "stdev": round(statistics.stdev(values), 6) if len(values) > 1 else 0.0,
            "min": round(min(values), 6),
            "max": round(max(values), 6),
        }

    # ------------------------------------------------------- post-campaign review

    def _evolve_after_campaign(self, summary: dict[str, Any], history: list[TrialResult]) -> None:
        """Review the finished Campaign and apply what the next one should change.

        This is the one point where the agent's remit is wide: the Campaign is
        over, so the search space, the policy, and the source code may all change
        for the *next* run. The review runs whether or not applying is enabled, so
        the reasoning is on record even when nothing is applied.
        """
        discovery = self._selector_discovery() or None
        discovery_path = self.run_dir / "discovery.json"
        if discovery_path.exists():
            try:
                discovery = json.loads(discovery_path.read_text())
            except ValueError:
                discovery = None

        evidence = build_evidence(
            history,
            direction=self.config.objective_direction,
            proposed_counts={
                name: int(stats.get("proposed", 0))
                for name, stats in self.reputation.snapshot().items()
            },
            phases_by_trial=self._phases_by_trial(),
            calibration_pairs=list(self.aic.calibrator.pairs),
        )
        evidence["configured_proposers"] = list(self.config.enabled_proposers)
        evidence["fixed_candidate_pool"] = self.config.fixed_candidate_pool
        final_validation = summary.get("final_validation") or None
        experience = build_experience(
            campaign_id=self.config.campaign_id,
            run_dir=self.run_dir,
            direction=self.config.objective_direction,
            objective_metric=self.config.objective_metric,
            trials=history,
            evidence=evidence,
            policy=self.policy_manager.current.model_dump(mode="json"),
            space_hash=self.space.digest,
            discovery=discovery,
            final_validation=final_validation,
        )
        experience["objective_contract"] = objective_contract(self.config)
        experience["history_manifest"] = self.history_manifest
        experience["lessons"] = [
            json.loads(p.read_text()) for p in sorted(self.run_dir.glob("episode_*_review.json"))
        ]
        experience["search_space"] = self.space.model_dump(mode="json")
        experience["discovery"] = self._selector_discovery()
        experience_path = self.run_dir / "campaign_experience.json"
        if not experience_path.exists():
            write_experience(self.evolution_root / "experience.jsonl", experience)
        (self.run_dir / "campaign_experience.json").write_text(
            json.dumps(experience, indent=2, ensure_ascii=False, default=str) + "\n"
        )

        base_space_path = Path(self.config.search_space)
        if not base_space_path.is_absolute():
            base_space_path = self.root / base_space_path
        protected = (
            sorted(SearchSpace.load(base_space_path).dimensions) if base_space_path.exists() else []
        )
        experience["source_context"] = (
            source_context(self.root) if self.config.use_outer_agent else {}
        )
        review_path = self.run_dir / "campaign_review.json"
        plan = (
            json.loads(review_path.read_text())
            if review_path.exists()
            else CampaignReviewAgent(self.client).review(
                experience,
                use_llm=self.config.use_outer_agent,
                current_policy=self.policy_manager.current.model_dump(mode="json"),
                protected_dimensions=protected,
            )
        )
        (self.run_dir / "campaign_review.json").write_text(
            json.dumps(plan, indent=2, ensure_ascii=False, default=str) + "\n"
        )
        self.store.append_once(
            "CampaignReviewed",
            {"campaign_id": self.config.campaign_id, "plan": plan},
            "campaign_id",
        )
        self.trace.emit(
            "Overall summary",
            "The outer agent completed its campaign review.",
            summary=plan.get("summary", []),
            space_changes=plan.get("space_changes"),
            policy_patch=plan.get("policy_changes"),
            objective_changes=plan.get("objective_changes"),
            code_proposal_count=len(plan.get("code_proposals") or []),
            no_change_reason=plan.get("no_change_reason"),
        )

        if self.store.event("EvolutionApplied", "campaign_id", self.config.campaign_id):
            return
        if not self.config.auto_apply_evolution:
            self.trace.emit(
                "Overall summary",
                "Automatic application is disabled; the review is recorded only.",
                result_path=str(self.run_dir / "campaign_review.json"),
            )
            return

        available = set(flatten_facts(discovery.get("workload") or {})) if discovery else set()
        available.update(f"evidence.dimension_effects.{name}" for name in evidence.get("dimension_effects", {}))
        available.add("evidence.failures")
        applier = EvolutionApplier(
            self.root,
            self.evolution_root,
            guard=WorkspaceGuard(
                self.root,
                test_command=self.config.evolution_test_command or None,
                performance_command=self.config.evolution_performance_command,
                min_repeats=self.config.evolution_min_repeats,
                performance_timeout_s=self.config.evolution_performance_timeout_s,
                isolated=True,
            ),
        )
        outcome = applier.apply(
            plan,
            base_space=self.space,
            current_policy=self.policy_manager.current.model_dump(mode="json"),
            protected_dimensions=protected,
            available_evidence=available,
            campaign_id=self.config.campaign_id,
        )
        self.store.append_once(
            "EvolutionApplied",
            asdict(outcome) | {"campaign_id": self.config.campaign_id},
            "campaign_id",
        )
        self.trace.emit(
            "Apply evolution",
            "Applied the review results to the next run.",
            applied=outcome.applied,
            rejected=outcome.rejected,
            code_results=outcome.code,
            evolution_event=outcome.event_path,
            next_search_space=outcome.next_space_path,
            next_policy=outcome.next_policy_path,
        )
