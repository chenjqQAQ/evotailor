from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, FiniteFloat, model_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


class RunMode(StrEnum):
    PORTFOLIO_AGENT = "portfolio_agent"
    BASELINE = "baseline"


class ExecutionBackend(StrEnum):
    SIMULATED = "simulated"
    DOCKER = "docker"
    HOST = "host"


class Phase(StrEnum):
    MODELING = "modeling"
    MIXED = "mixed"
    LOCAL = "local"
    VALIDATION = "validation"


class TrialStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class FailureType(StrEnum):
    PREFLIGHT = "FAILED_PREFLIGHT"
    #: The harness itself is misconfigured: a dimension has no way to reach the
    #: engine, so the Trial failed before it could measure anything.
    CONFIGURATION = "FAILED_CONFIGURATION"
    DOCKER = "FAILED_DOCKER"
    GPU = "FAILED_GPU"
    VLLM_START = "FAILED_VLLM_START"
    VLLM_READY = "FAILED_VLLM_READY"
    WORKLOAD = "FAILED_WORKLOAD"
    FPM = "FAILED_FPM"
    AIC_FEEDBACK = "FAILED_AIC_FEEDBACK"
    CLEANUP = "FAILED_CLEANUP"
    TIMEOUT = "FAILED_TIMEOUT"
    UNKNOWN = "FAILED_UNKNOWN"


class Candidate(BaseModel):
    candidate_id: str
    proposer: str
    proposer_version: str = "1.0.0"
    parameters: dict[str, Any]
    rationale: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class Prediction(BaseModel):
    candidate_id: str
    raw: float | None = Field(default=None, allow_inf_nan=False)
    calibrated: float | None = Field(default=None, allow_inf_nan=False)
    uncertainty: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    method: str
    samples: int = 0
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class Selection(BaseModel):
    candidate_id: str
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)
    risk_factors: list[str] = Field(default_factory=list)
    used_fallback: bool = False


class TrialResult(BaseModel):
    trial_id: str
    candidate: Candidate
    status: TrialStatus
    objective: float | None = Field(default=None, allow_inf_nan=False)
    metrics: dict[str, FiniteFloat] = Field(default_factory=dict)
    telemetry: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    failure_type: str | None = None
    measurement_kind: str = "real"
    stage_id: str = ""
    space_hash: str = ""
    code_version: str = ""
    resource_snapshot: dict[str, Any] = Field(default_factory=dict)
    validity: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime = Field(default_factory=utc_now)


class ProposerStats(BaseModel):
    proposed: int = 0
    selected: int = 0
    succeeded: int = 0
    improved: int = 0
    cumulative_improvement: float = 0.0

    @property
    def smoothed_success(self) -> float:
        return (self.improved + 1.0) / (self.succeeded + 2.0)


class CampaignConfig(BaseModel):
    campaign_id: str = "demo"
    run_mode: RunMode = RunMode.PORTFOLIO_AGENT
    execution_backend: ExecutionBackend = ExecutionBackend.SIMULATED
    max_trials: int = Field(default=12, ge=1)
    episode_size: int = Field(default=4, ge=1)
    episode_min_trials: int = Field(default=6, ge=1)
    episode_max_trials: int = Field(default=15, ge=1)
    episode_min_count: int = Field(default=3, ge=1)
    episode_max_count: int = Field(default=6, ge=1)
    final_validation_trials: int = Field(default=0, ge=0)
    #: Repeats per arm of the post-Campaign validation. These Trials measure the
    #: winning configuration and are deliberately outside the search budget,
    #: because their job is to verify the result rather than to search.
    final_validation_repeats: int = Field(default=0, ge=0)
    #: Which configurations the final validation re-measures. "best" is this
    #: Campaign's winner and "baseline" is the frozen safe baseline.
    final_validation_arms: list[str] = Field(default_factory=lambda: ["best", "baseline"])
    seed: int = 42
    objective_direction: str = "minimize"
    search_space: str = "configs/search_spaces/common_v2.json"
    run_root: str = "runs"
    baseline_proposer: str = "tpe"
    enabled_proposers: list[str] = Field(
        default_factory=lambda: [
            "tpe",
            "slo_guard",
            "scoot",
            "coverage",
            "neighborhood",
        ]
    )
    mature_optimizer_root: str = "${EVOTAILOR_MATURE_ROOT}/src"
    mature_optimizer_policies: dict[str, dict[str, Any]] = Field(default_factory=dict)
    use_llm_selector: bool = True
    use_outer_agent: bool = True
    use_episode_planner: bool = True
    use_episode_boundary_agent: bool | None = None
    fixed_candidate_pool: bool = False
    # An explicit list of complete parameter dicts, executed instead of any proposer or
    # selector. This exists so a recording run can execute one configuration fixed in
    # advance — the centre of the search space. A trace recorded from whatever a
    # proposer happened to offer would make the frozen workload depend on that proposer,
    # which is exactly the bias a frozen trace exists to prevent.
    forced_candidates: list[dict] | None = None
    auto_promote_policy: bool = False
    model: str = "gpt-5.6-sol"
    base_url: str = "${EVOTAILOR_V2_BASE_URL}"
    api_key_env: str = "EVOTAILOR_PROPOSER_API_KEY"
    model_timeout_s: float = 60.0
    model_retries: int = 2
    #: Stand-in relay serving the same model. The primary relay rate-limits under
    #: load and a rate limit used to end the Campaign; with a fallback configured the
    #: client abandons the primary for the rest of the run instead. Empty means no
    #: fallback is configured and a primary failure stays fatal.
    fallback_base_url: str | None = None
    fallback_api_key_env: str | None = None
    #: Attempts against one relay before it is considered failed.
    model_attempts_per_endpoint: int = Field(default=3, ge=1)
    model_max_backoff_s: float = Field(default=300.0, gt=0)
    #: Give the proposers and the selector measured facts about this model and engine
    #: so they do not have to rediscover known-bad regions. Advisory only.
    use_prior_knowledge: bool = False
    #: Refuse a candidate whose weights cannot fit the memory the configuration
    #: reserves, without spending search budget on it. Portfolio mode only, because
    #: the baselines must search the same space without help.
    enforce_weight_memory_feasibility: bool = True
    #: Run the upper-layer workload inside the serving container so both compete for
    #: the same frozen CPU, memory and pid budget. False keeps it on the host.
    workload_in_container: bool = False
    simulated_noise: float = 0.01
    command: list[str] = Field(default_factory=list)
    command_timeout_s: float = 900.0
    aiconfigurator_command: list[str] = Field(default_factory=list)
    discovery_artifact: str | None = None
    console_progress: bool = True
    #: Name of the objective the runtime actually reports, carried into the
    #: Campaign review so a change to it can be described precisely.
    objective_metric: str = "deployment_plus_task_time_s"
    #: Run the post-Campaign review and apply its plan without human review.
    auto_apply_evolution: bool = False
    #: Where the next-Campaign artifacts live. Defaults to ``<run_dir>/evolution``.
    evolution_root: str | None = None
    #: Start from the search space and policy a previous Campaign evolved.
    use_evolved_defaults: bool = False
    #: Verification command for an evolved code patch; the project default is used
    #: when empty.
    evolution_test_command: list[str] = Field(default_factory=list)
    history_seed_trials: int = Field(default=1, ge=0, le=2)
    history_guided_search: bool = True
    max_modeling_trials: int | None = Field(default=None, ge=1)
    enable_parameter_experiments: bool = True
    history_mode: str = "cold"
    history_sources: list[str] = Field(default_factory=list)
    experience_sources: list[str] = Field(default_factory=list)
    #: How experience is injected: cold (nothing), direct (verbatim source
    #: order) or v2 (ranked by target workload signature and resource condition).
    #: The arm lives here, not in `experience_sources`, so every arm is measured
    #: on the same declared source artifacts.
    experience_transfer_mode: str = "cold"
    #: The public search space S_common every baseline must search exactly and
    #: the portfolio must extend. Empty means the invariant is not being applied
    #: to this Campaign, which is recorded rather than assumed.
    common_space_path: str = ""
    objective_unit: str = "s"
    measurement_protocol: str = "fixed-workload-v1"
    measurement_scope: str = "configured objective metric"
    evolution_performance_command: list[str] = Field(default_factory=list)
    evolution_min_repeats: int = Field(default=3, ge=2)
    evolution_performance_timeout_s: int = Field(default=28800, ge=60)

    @model_validator(mode="after")
    def validate_mode(self) -> CampaignConfig:
        if self.auto_promote_policy:
            raise ValueError("Apply policy changes at the next stage boundary, not within an active stage")
        if self.objective_metric != "deployment_plus_task_time_s" or self.objective_direction != "minimize":
            raise ValueError("The paper objective is deployment_plus_task_time_s, minimized")
        if self.history_mode not in {"cold", "warm", "transfer"}:
            raise ValueError("history_mode must be cold, warm or transfer")
        if self.experience_transfer_mode not in {"cold", "direct", "v2"}:
            raise ValueError("experience_transfer_mode must be cold, direct or v2")
        if self.common_space_path and self.run_mode not in {
            RunMode.BASELINE,
            RunMode.PORTFOLIO_AGENT,
        }:
            raise ValueError("S_common applies to baseline or portfolio_agent Campaigns only")
        # Raw history is what `history_mode` governs. Experience injection is
        # governed by `experience_transfer_mode`, so a declared-but-cold arm is
        # explicit rather than an error.
        if self.history_mode == "cold" and self.history_sources:
            raise ValueError("cold mode cannot import raw history")
        if self.history_mode != "cold" and not (self.history_sources or self.experience_sources):
            raise ValueError("warm/transfer mode requires explicit source artifacts")
        if self.experience_transfer_mode != "cold" and not self.experience_sources:
            raise ValueError("direct/v2 transfer requires explicit experience sources")
        if self.objective_direction not in {"maximize", "minimize"}:
            raise ValueError("objective_direction must be 'maximize' or 'minimize'")
        if self.use_episode_boundary_agent is None:
            self.use_episode_boundary_agent = (
                self.use_episode_planner and self.run_mode == RunMode.PORTFOLIO_AGENT
            )
        if self.run_mode == RunMode.BASELINE:
            if self.baseline_proposer == "llm":
                raise ValueError("Baseline mode cannot use the LLM proposer")
            self.use_llm_selector = False
            self.use_outer_agent = False
            self.use_episode_planner = False
            self.use_episode_boundary_agent = False
        if "aiconfigurator" in self.enabled_proposers:
            raise ValueError("AIConfigurator is prediction-only and cannot be an enabled proposer")
        return self
