import json
import random
import sys

import pytest

from evotailor.baselines.registry import build
from evotailor.contracts import Candidate, Phase, TrialStatus
from evotailor.execution.backends import CommandExecutor
from evotailor.loops.inner import allocate_stage_budgets
from evotailor.selection.feasibility import weight_memory_verdict
from evotailor.spaces.search_space import Dimension, SearchSpace


def test_multiple_stages_share_one_budget_instead_of_resetting_it():
    assert allocate_stage_budgets(12, 3) == [4, 4, 4]
    assert allocate_stage_budgets(10, 3) == [4, 3, 3]
    with pytest.raises(ValueError):
        allocate_stage_budgets(12, 3, [12, 12, 12])


def test_unknown_resources_do_not_imply_failure_and_dp_is_not_globally_blacklisted():
    parameters = {"deployment.data_parallel_size": 2, "runtime.gpu_memory_utilization": 0.5}
    assert weight_memory_verdict(parameters, weights_bytes=None, gpu_memory_bytes=None, gpu_count=None) is None
    assert weight_memory_verdict(parameters, weights_bytes=4, gpu_memory_bytes=20, gpu_count=2) is None
    assert weight_memory_verdict(parameters, weights_bytes=4, gpu_memory_bytes=20, gpu_count=1).code
    assert weight_memory_verdict(parameters, weights_bytes=4, gpu_memory_bytes=20,
                                 gpu_count=2, available_memory_bytes=2).code == "INSUFFICIENT_LIVE_MEMORY"


def test_missing_upstream_baseline_is_an_error_not_an_approximation(tmp_path):
    proposer = build("llm_tuna", root=tmp_path, max_trials=12)
    space = SearchSpace(dimensions={"batch": Dimension(kind="choice", choices=[8, 16])},
                        safe_baseline={"batch": 8})
    with pytest.raises(RuntimeError, match="upstream|Upstream"):
        proposer.propose(space, [], None, Phase.MODELING, random.Random(0))


def test_executor_requires_quality_and_consistent_timing():
    payload = {"objective": 10, "metrics": {"deployment_time_s": 3, "task_makespan_s": 7,
                "deployment_plus_task_time_s": 10}, "validity": {key: True for key in
                ("deployment_succeeded", "workload_completed", "quality_passed", "measurement_complete",
                 "cleanup_success", "served_model_matches_all_requests")}}
    candidate = Candidate(candidate_id="test", proposer="fixture", parameters={})

    def execute(value):
        command = [sys.executable, "-c", "import sys;sys.stdin.read();print(" + repr(json.dumps(value)) + ")"]
        return CommandExecutor(command).execute("test", candidate)

    assert execute(payload).status == TrialStatus.SUCCEEDED
    payload["validity"]["quality_passed"] = False
    assert execute(payload).status == TrialStatus.FAILED
