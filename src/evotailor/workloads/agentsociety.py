"""AgentSociety workload adapters: PD, PG and SM (paper §5.1, Table A1).

A workload is a frozen request trace: an ordered list of LLM calls with their prompt
payload, a dependency chain identifier, and a relative arrival time.  Replaying it against
a serving engine reproduces the application's request pattern without re-running the
simulation policy, which is what makes configuration comparisons reproducible.

Trace schema (``workload.json``)::

    {
      "workload_id": "pd",
      "agents": ["a1", ...],              # agent chain owners
      "chains": {"a1": [...]},            # optional per-chain op ordering
      "operations": [
        {"op_id": ..., "chain": "a1", "start_s": 0.0,
         "request": {"model": ..., "messages": [...], "temperature": 0,
                     "max_tokens": 128, "tools": [...]}}
      ],
      "concurrency": {"peak_concurrent_llm": 4},
      "expected_input_tokens": 123,
      "expected_output_tokens": 45
    }

The trace is executed in fixed-output mode: generation length is pinned to the recorded
value so that two configurations process exactly the same number of tokens and therefore
compare on latency rather than on sampling luck.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

#: Scenario sizes used in the paper (Table A1).  Kept as data so that a run can assert that
#: it is replaying the expected workload.
SCENARIO_SIZES: dict[str, int] = {"PD": 271, "PG": 255, "SM": 263}
#: Larger windows used by the resource and transfer experiments.
WINDOW_SIZES: dict[str, int] = {"PD": 1040, "PG": 1016, "SM": 1670}


@dataclass
class Operation:
    op_id: str
    chain: str
    start_s: float
    request: dict[str, Any]

    @property
    def tool_count(self) -> int:
        return len(self.request.get("tools") or [])

    @property
    def prompt_chars(self) -> int:
        total = 0
        for message in self.request.get("messages") or []:
            content = message.get("content")
            if isinstance(content, str):
                total += len(content)
            elif isinstance(content, list):
                total += sum(len(part.get("text", "")) for part in content if isinstance(part, dict))
        return total


@dataclass
class Workload:
    workload_id: str
    operations: list[Operation]
    agents: list[str] = field(default_factory=list)
    peak_concurrency: int = 0
    expected_input_tokens: int = 0
    expected_output_tokens: int = 0
    source: str = ""

    @property
    def size(self) -> int:
        return len(self.operations)

    @property
    def tool_bearing(self) -> int:
        """Number of operations that carry a tool schema.

        An engine must be launched with tool-call support when this is non-zero; otherwise
        those requests are rejected and the replay silently measures a smaller workload.
        """
        return sum(1 for op in self.operations if op.tool_count)

    def __iter__(self) -> Iterator[Operation]:
        return iter(self.operations)

    def assert_size(self, scenario: str, *, window: bool = False) -> None:
        expected = (WINDOW_SIZES if window else SCENARIO_SIZES)[scenario]
        if self.size != expected:
            raise ValueError(
                f"workload {self.workload_id} has {self.size} operations, "
                f"expected {expected} for scenario {scenario}"
            )

    def summary(self) -> dict[str, Any]:
        return {
            "workload_id": self.workload_id,
            "operations": self.size,
            "agents": len(self.agents),
            "peak_concurrency": self.peak_concurrency,
            "tool_bearing": self.tool_bearing,
            "expected_input_tokens": self.expected_input_tokens,
            "expected_output_tokens": self.expected_output_tokens,
        }


def load(path: str | Path, *, scenario: str | None = None) -> Workload:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    operations: list[Operation] = []
    for index, item in enumerate(raw.get("operations") or []):
        request = item.get("request") or item.get("request_parameters") or {}
        operations.append(
            Operation(
                op_id=str(item.get("op_id", f"op-{index:04d}")),
                chain=str(item.get("chain") or item.get("agent_id") or "default"),
                start_s=float(item.get("start_s", item.get("arrival_s", 0.0))),
                request=request,
            )
        )
    concurrency = raw.get("concurrency") or {}
    workload = Workload(
        workload_id=str(raw.get("workload_id") or Path(path).parent.name),
        operations=operations,
        agents=list(raw.get("agents") or sorted({o.chain for o in operations})),
        peak_concurrency=int(concurrency.get("peak_concurrent_llm", 0)),
        expected_input_tokens=int(raw.get("expected_input_tokens", 0)),
        expected_output_tokens=int(raw.get("expected_output_tokens", 0)),
        source=str(path),
    )
    if scenario:
        workload.assert_size(scenario)
    return workload


def fixed_output_request(op: Operation, *, max_tokens: int | None = None) -> dict[str, Any]:
    """Pin the generation length so that two configurations process the same tokens."""
    request = dict(op.request)
    if max_tokens is not None:
        request["max_tokens"] = max_tokens
    request.setdefault("temperature", 0)
    request["stream"] = False
    return request
