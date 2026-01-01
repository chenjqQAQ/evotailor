from evotailor.execution.timing import Span, TaskComposition, TimingBreakdown


def test_breakdown_splits_and_sums():
    tb = TimingBreakdown(launch_ts=10.0, ready_ts=42.5, finish_ts=100.0)
    assert tb.deploy == 32.5
    assert tb.task == 57.5
    assert abs(tb.e2e - (tb.deploy + tb.task)) < 1e-9


def test_non_monotonic_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        TimingBreakdown(launch_ts=1.0, ready_ts=0.5, finish_ts=2.0)


def test_overlap_is_not_double_counted():
    # planning 0-10, execution 8-20, judging 20-25, task window 25
    spans = [Span("planning", 0.0, 10.0), Span("execution", 8.0, 20.0), Span("judging", 20.0, 25.0)]
    comp = TaskComposition.from_spans(spans, task_window=25.0)
    assert abs(comp.planning_only - 8.0) < 1e-9
    assert abs(comp.overlap - 2.0) < 1e-9
    assert abs(comp.execution_only - 10.0) < 1e-9
    assert abs(comp.judging - 5.0) < 1e-9
    assert abs(comp.total - 25.0) < 1e-9


def test_shares_sum_to_100():
    comp = TaskComposition(planning_only=60, overlap=10, execution_only=20, judging=8, other=2)
    assert abs(sum(comp.shares().values()) - 100.0) < 1e-9
