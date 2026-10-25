import math

import pytest

from evotailor.contracts import CampaignConfig, Candidate, TrialResult, TrialStatus
from evotailor.metrics.serving import RequestTiming, from_requests, from_scrape
from evotailor.objective import best_trial, objective_value, valid_rate
from evotailor.storage import EventStore


def trial(identity, total=10.0, **overrides):
    data = dict(trial_id=identity, candidate=Candidate(candidate_id=identity, proposer="test", parameters={}),
                status=TrialStatus.SUCCEEDED, objective=total,
                metrics={"deployment_time_s": 3.0, "task_makespan_s": total-3,
                         "deployment_plus_task_time_s": total},
                validity={key: True for key in ("deployment_succeeded", "workload_completed",
                                                "quality_passed", "measurement_complete")})
    data.update(overrides)
    return TrialResult(**data)


def test_primary_result_uses_the_same_valid_real_trial_for_all_metrics():
    failed = trial("failed", status=TrialStatus.FAILED, objective=None)
    rejected_quality = trial("bad-quality", 4, validity={"quality_passed": False})
    simulated = trial("synthetic", 3, measurement_kind="simulated")
    best, slower = trial("best", 8), trial("slower", 12)
    history = [failed, rejected_quality, simulated, best, slower]
    assert best_trial(history).trial_id == "best"
    assert valid_rate(history) == 2/4
    assert math.isinf(objective_value([failed, rejected_quality, simulated]))


def test_inconsistent_timing_and_missing_measurements_are_not_valid():
    inconsistent = trial("bad", metrics={"deployment_time_s": 3, "task_makespan_s": 4,
                                         "deployment_plus_task_time_s": 10})
    assert best_trial([inconsistent, trial("missing", metrics={})]) is None
    with pytest.raises(ValueError, match="paper objective"):
        CampaignConfig(objective_direction="maximize")


def test_request_tpot_is_not_token_weighted_and_throughput_uses_serving_window():
    records = [RequestTiming(0, 1, 2, 2), RequestTiming(0, 2, 20, 10), RequestTiming(0, 3, 3, 1)]
    result = from_requests(records, serving_window_s=25)
    assert result.tpot_mean_s == 1.5
    assert result.ttft_mean_s == 2
    assert result.throughput_tok_s == 13/25


def test_histogram_is_a_separate_diagnostic_and_uses_counter_deltas():
    def metrics(count, total):
        return "\n".join(f"vllm:{name}_{suffix} {value}" for name in
            ("time_to_first_token_seconds", "inter_token_latency_seconds")
            for suffix, value in (("count", count), ("sum", total)))
    result = from_scrape(metrics(3, 8), before_text=metrics(1, 2), completion_tokens=12,
                         request_count=2, serving_window_s=6)
    assert result.ttft_mean_s == 3
    assert "TPOT" not in result.as_dict()
    assert result.as_dict()["inter_token_histogram_mean_s"] == 3


def test_completed_stage_provenance_and_validity_cannot_be_rewritten(tmp_path):
    store = EventStore(tmp_path / "trials.sqlite3")
    record = trial("one", stage_id="stage-1", space_hash="old-space", code_version="v1")
    store.save_trial(0, record)
    changed = record.model_copy(deep=True)
    changed.space_hash = "new-space"
    with pytest.raises(ValueError, match="immutable"):
        store.save_trial(0, changed)
    changed = record.model_copy(deep=True)
    changed.validity["quality_passed"] = False
    with pytest.raises(ValueError, match="immutable"):
        store.save_trial(0, changed)
    assert store.trials()[0].space_hash == "old-space"
