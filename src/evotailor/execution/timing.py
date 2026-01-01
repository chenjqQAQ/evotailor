"""Timing decomposition (paper §3, Eq. 1).

For a valid trial ``t``:

    T_deploy,t = b_t - a_t      (launch -> service ready)
    T_task,t   = e_t - b_t      (ready -> task finished)
    T_e2e,t    = e_t - a_t

The task interval is *wall clock*: overlapping sub-activities are not double counted.  The
functions here are deliberately pure so that the same decomposition is used by the inner
loop, by the metric aggregation, and by offline analysis of a campaign database.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence


@dataclass(frozen=True)
class TimingBreakdown:
    """Absolute timestamps of one trial, in seconds on a monotonic clock."""

    launch_ts: float
    ready_ts: float
    finish_ts: float

    def __post_init__(self) -> None:
        if not (self.launch_ts <= self.ready_ts <= self.finish_ts):
            raise ValueError(
                f"non-monotonic timestamps: {self.launch_ts} {self.ready_ts} {self.finish_ts}"
            )

    @property
    def deploy(self) -> float:
        return self.ready_ts - self.launch_ts

    @property
    def task(self) -> float:
        return self.finish_ts - self.ready_ts

    @property
    def e2e(self) -> float:
        return self.finish_ts - self.launch_ts

    def as_dict(self) -> dict[str, float]:
        return {"deploy_s": self.deploy, "task_s": self.task, "e2e_s": self.e2e}


@dataclass
class Span:
    """A labelled interval inside the task window."""

    label: str
    start: float
    end: float

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"span {self.label!r} ends before it starts")

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class TaskComposition:
    """Mutually exclusive partition of the task window.

    ``planning_only``, ``execution_only``, ``judging`` and ``other`` are disjoint;
    ``overlap`` is the interval where planning of step *i+1* runs concurrently with the
    execution of step *i*.  The five shares sum to the task duration, which is what makes
    the stacked attribution in the paper's Figure 3 well defined.
    """

    planning_only: float = 0.0
    overlap: float = 0.0
    execution_only: float = 0.0
    judging: float = 0.0
    other: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.planning_only
            + self.overlap
            + self.execution_only
            + self.judging
            + self.other
        )

    def shares(self) -> dict[str, float]:
        total = self.total or 1.0
        return {
            "planning_only_pct": 100.0 * self.planning_only / total,
            "overlap_pct": 100.0 * self.overlap / total,
            "execution_only_pct": 100.0 * self.execution_only / total,
            "judging_pct": 100.0 * self.judging / total,
            "other_pct": 100.0 * self.other / total,
        }

    @classmethod
    def from_spans(cls, spans: Sequence[Span], task_window: float) -> "TaskComposition":
        """Partition ``[0, task_window]`` from possibly overlapping spans.

        The sweep is done over interval endpoints, so the result is exact for any overlap
        pattern and never double counts.
        """
        cut_points = {0.0, task_window}
        for span in spans:
            cut_points.add(min(max(span.start, 0.0), task_window))
            cut_points.add(min(max(span.end, 0.0), task_window))
        ordered = sorted(cut_points)
        composition = cls()
        for left, right in zip(ordered, ordered[1:]):
            if right <= left:
                continue
            mid = 0.5 * (left + right)
            active = {s.label for s in spans if s.start <= mid < s.end}
            dt = right - left
            if {"planning", "execution"} <= active:
                composition.overlap += dt
            elif "planning" in active:
                composition.planning_only += dt
            elif "execution" in active:
                composition.execution_only += dt
            elif "judging" in active:
                composition.judging += dt
            else:
                composition.other += dt
        # Anything the spans do not cover is "other".
        covered = composition.planning_only + composition.overlap + composition.execution_only + composition.judging
        composition.other += max(0.0, task_window - covered - composition.other)
        return composition


def aggregate(timings: Iterable[TimingBreakdown]) -> dict[str, float]:
    """Mean / min / max of the decomposition over a set of valid trials."""
    rows = list(timings)
    if not rows:
        return {}
    out: dict[str, float] = {}
    for key in ("deploy", "task", "e2e"):
        values = [getattr(r, key) for r in rows]
        out[f"{key}_mean"] = sum(values) / len(values)
        out[f"{key}_min"] = min(values)
        out[f"{key}_max"] = max(values)
    out["n"] = float(len(rows))
    return out


@dataclass
class ServiceLifecycle:
    """Small helper used by the backend adapters to stamp the three timestamps."""

    launch_ts: float | None = None
    ready_ts: float | None = None
    finish_ts: float | None = None
    marks: dict[str, float] = field(default_factory=dict)

    def mark(self, name: str, ts: float) -> None:
        self.marks[name] = ts
        if name == "launch":
            self.launch_ts = ts
        elif name == "ready":
            self.ready_ts = ts
        elif name == "finish":
            self.finish_ts = ts

    def breakdown(self) -> TimingBreakdown | None:
        if None in (self.launch_ts, self.ready_ts, self.finish_ts):
            return None
        return TimingBreakdown(self.launch_ts, self.ready_ts, self.finish_ts)  # type: ignore[arg-type]

    def deployment_phases(self) -> dict[str, float]:
        """Relative deploy sub-phases, used for attribution figures.

        Recognised marks: ``import``, ``engine_init``, ``weights``, ``warmup``, ``serve``.
        Missing marks are skipped, so partial instrumentation still yields a usable split.
        """
        order = ["launch", "import", "engine_init", "weights", "warmup", "serve", "ready"]
        present = [m for m in order if m in self.marks]
        return {
            f"{a}->{b}": self.marks[b] - self.marks[a]
            for a, b in zip(present, present[1:])
        }
