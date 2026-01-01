"""ComfyBench workload adapter (paper §5.1, Table 3, Figure 3).

ComfyBench tasks are not request traces: each task is an agentic workflow that plans a
ComfyUI graph, executes it on a resident diffusion backend, and is scored by an external
judge.  The task therefore contributes three span types to the task window —
``planning``, ``execution`` and ``judging`` — and the optimizer's objective covers all of
them plus deployment.

Only the four ``sample4`` tasks are used in the paper.  A task is considered *passed* when
the produced image passes the judge threshold; a task that fails to produce an image is
still counted in the task window, because the wall clock the user waits for is what the
objective measures.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TASK_KINDS = ("generate", "edit", "extend", "upscale")


@dataclass
class ComfyTask:
    """One ComfyBench instruction."""

    task_id: str
    instruction: str
    kind: str = "generate"
    inputs: list[str] = field(default_factory=list)
    expected_outputs: int = 1

    def is_valid(self) -> bool:
        return bool(self.instruction.strip()) and self.kind in TASK_KINDS


@dataclass
class TaskOutcome:
    task_id: str
    passed: bool
    planning_s: float
    execution_s: float
    judging_s: float
    error: str | None = None

    @property
    def total_s(self) -> float:
        return self.planning_s + self.execution_s + self.judging_s


def load_tasks(path: str | Path) -> list[ComfyTask]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    items = raw if isinstance(raw, list) else raw.get("tasks", [])
    tasks: list[ComfyTask] = []
    for index, item in enumerate(items):
        tasks.append(
            ComfyTask(
                task_id=str(item.get("task_id") or item.get("id") or f"task-{index + 1:02d}"),
                instruction=str(item.get("instruction") or item.get("prompt") or ""),
                kind=str(item.get("kind") or item.get("type") or "generate"),
                inputs=list(item.get("inputs") or []),
                expected_outputs=int(item.get("expected_outputs", 1)),
            )
        )
    return tasks


def summarise(outcomes: list[TaskOutcome]) -> dict[str, Any]:
    """Aggregate per-task outcomes into the quantity the optimizer minimizes.

    ``task_window_s`` is the wall clock of the whole task batch: tasks are dispatched by one
    agent, so their planning phases overlap with earlier executions.  Summing per-task
    durations would overcount, hence the batch window is taken from the caller.
    """
    n = len(outcomes)
    passed = sum(1 for o in outcomes if o.passed)
    return {
        "tasks": n,
        "passed": passed,
        "pass_rate": (passed / n) if n else 0.0,
        "planning_s": sum(o.planning_s for o in outcomes),
        "execution_s": sum(o.execution_s for o in outcomes),
        "judging_s": sum(o.judging_s for o in outcomes),
    }


def validate_environment(required_nodes: list[str], installed_nodes: set[str]) -> list[str]:
    """Return the missing custom nodes.

    ComfyUI resolves node names at graph build time and reports a generic failure when one
    is absent, so the adapter checks explicitly before a trial is charged to the budget.
    """
    return [node for node in required_nodes if node not in installed_nodes]
