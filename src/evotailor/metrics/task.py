"""Task-time composition from application events (paper §5.3, Figure 3, Appendix C).

The workload emits a JSON-lines event stream.  Each line is a span:

    {"label": "planning", "start": 0.0, "end": 12.4, "task": 1}

``label`` is one of ``planning``, ``execution``, ``judging`` (anything else is folded into
``other``).  Spans may overlap: planning of step *i+1* runs while step *i* executes.  The
composition therefore partitions the wall clock instead of summing durations, which is the
property that makes the stacked attribution in the paper's Figure 3 valid.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from evotailor.execution.timing import Span, TaskComposition

KNOWN_LABELS = ("planning", "execution", "judging")


@dataclass
class TaskEvent:
    label: str
    start: float
    end: float
    task: int = 0
    meta: dict | None = None

    def to_span(self) -> Span:
        label = self.label if self.label in KNOWN_LABELS else "other"
        return Span(label=label, start=self.start, end=self.end)


def read_events(path: str | Path) -> list[TaskEvent]:
    events: list[TaskEvent] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        events.append(
            TaskEvent(
                label=str(raw.get("label", "other")),
                start=float(raw["start"]),
                end=float(raw["end"]),
                task=int(raw.get("task", 0)),
                meta=raw.get("meta"),
            )
        )
    return events


def compose(events: Iterable[TaskEvent], task_window: float) -> TaskComposition:
    spans = [e.to_span() for e in events]
    return TaskComposition.from_spans(spans, task_window)


def compose_per_task(events: Iterable[TaskEvent], n_tasks: int) -> list[TaskComposition]:
    """Compose each task separately; task indices start at 1."""
    grouped: dict[int, list[TaskEvent]] = {}
    for event in events:
        grouped.setdefault(event.task, []).append(event)
    out: list[TaskComposition] = []
    for index in range(1, n_tasks + 1):
        chunk = grouped.get(index, [])
        window = max((e.end for e in chunk), default=0.0)
        out.append(compose(chunk, window))
    return out


def summarise(compositions: Iterable[TaskComposition]) -> dict[str, float]:
    """Aggregate composition across tasks into seconds and percentages."""
    items = list(compositions)
    if not items:
        return {}
    total = TaskComposition(
        planning_only=sum(c.planning_only for c in items),
        overlap=sum(c.overlap for c in items),
        execution_only=sum(c.execution_only for c in items),
        judging=sum(c.judging for c in items),
        other=sum(c.other for c in items),
    )
    result = {
        "planning_only_s": total.planning_only,
        "overlap_s": total.overlap,
        "execution_only_s": total.execution_only,
        "judging_s": total.judging,
        "other_s": total.other,
        "task_total_s": total.total,
    }
    result.update(total.shares())
    return result


def pipeline_duration(events: Iterator[TaskEvent]) -> tuple[float, float]:
    """Return ``(pipeline_end, execution_start)`` for the timeline figure.

    The pipeline starts at the first execution span and stops at the last event, which is
    the interval the paper uses when it compares two methods on the same workflow.
    """
    events = list(events)
    if not events:
        return 0.0, 0.0
    start = min((e.start for e in events if e.label == "execution"), default=events[0].start)
    end = max(e.end for e in events)
    return end - start, start
