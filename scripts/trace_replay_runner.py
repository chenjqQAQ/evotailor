#!/usr/bin/env python3
"""Replay a frozen trace as one EvoTailor Trial's workload.

This is the workload side of a replay Trial, the counterpart to
``scripts/agentsociety_runner.py``. The wrapper has already started vLLM with the
candidate configuration and waited for health; this runner sends the frozen requests to
it and reports how long the fixed workload took.

What it deliberately does not do:

* It does not let the model's answers steer anything. The next request is whatever the
  frozen graph says it is, so every configuration is asked to serve exactly the same
  work and the only thing that varies is how fast it is served.
* It does not execute tools. In the recording the costly tools were ``ask_env`` and
  ``observe``, whose work *is* LLM calls that appear as their own entries in this same
  graph, and every other tool was a sub-millisecond local file read.

A Trial that fails, times out, or issues the wrong number of requests is reported as
invalid rather than as a fast result.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path

import httpx

# The replay's objective is the wall clock of the fixed workload, not the
# deployment-plus-task time an agent simulation reports. The wrapper checks that this name
# matches the runtime config's objective_metric.
OBJECTIVE_METRIC = "trace_execution_s"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workload", required=True, type=Path, help="frozen traces/<trace_id>/workload.json"
    )
    parser.add_argument("--result-path", required=True, type=Path)
    parser.add_argument("--api-base", required=True, help="OpenAI-compatible base URL")
    parser.add_argument("--served-model", required=True)
    parser.add_argument(
        "--output-mode",
        choices=("natural", "fixed"),
        default="natural",
        help="fixed enforces each recorded completion token count using vLLM min_tokens/ignore_eos",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--extra-body",
        default=None,
        help="JSON merged into every request; must match the recording",
    )
    parser.add_argument(
        "--extra-body-env",
        default=None,
        help="environment variable holding that JSON, used because a "
        "literal body contains braces the runtime template check "
        "reads as an unsubstituted placeholder",
    )
    parser.add_argument(
        "--timeout", type=float, default=1800.0, help="hard cap; hitting it marks the Trial invalid"
    )
    parser.add_argument("--max-connections", type=int, default=64)
    parser.add_argument("--experiment-id", default="evotailor-replay")
    return parser.parse_args(argv)


def _write_result(result_path: Path, payload: dict) -> None:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


class Replay:
    def __init__(self, workload: dict, args: argparse.Namespace) -> None:
        self.workload = workload
        self.args = args
        raw_body = args.extra_body
        if raw_body is None and args.extra_body_env:
            raw_body = os.environ.get(args.extra_body_env)
        self.extra_body = json.loads(raw_body) if raw_body else {}
        self.samples: list[dict] = []
        self.timed_out = False
        self.deadline: float | None = None

    def chains(self) -> dict:
        grouped: dict = defaultdict(list)
        for operation in self.workload["operations"]:
            grouped[operation.get("agent_id")].append(operation)
        for operations in grouped.values():
            operations.sort(key=lambda item: item.get("arrival_index") or 0)
        return grouped

    async def call(self, client: httpx.AsyncClient, operation: dict) -> None:
        # Carry the recorded request parameters, tool definitions included. They are
        # most of the prompt, so omitting them silently shrinks the workload.
        body = dict(operation.get("request_parameters") or {})
        body.update(
            {
                "model": self.args.served_model,
                "messages": operation["messages"],
                "temperature": self.args.temperature,
                "stream": False,
            }
        )
        body.update(self.extra_body)
        expected_output = operation.get("expected_output_tokens")
        if self.args.output_mode == "fixed":
            if not isinstance(expected_output, int) or expected_output <= 0:
                raise ValueError(
                    "Fixed mode requires positive per-request recorded completion lengths"
                )
            body.update(max_tokens=expected_output, min_tokens=expected_output, ignore_eos=True)
            body.pop("max_completion_tokens", None)
            body.pop("stop", None)
        started = time.perf_counter()
        record = {
            "op_id": operation["op_id"],
            "expected_output_tokens": operation.get("expected_output_tokens"),
        }
        try:
            response = await client.post(
                f"{self.args.api_base.rstrip('/')}/chat/completions",
                headers={"Authorization": "Bearer EMPTY"},
                json=body,
            )
            response.raise_for_status()
            payload = response.json()
            usage = payload.get("usage") or {}
            record.update(
                ok=True,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                model_matches=payload.get("model") == self.args.served_model,
                output_matches=(usage.get("completion_tokens") == expected_output),
            )
        except Exception as exc:  # noqa: BLE001
            record.update(ok=False, error=f"{type(exc).__name__}: {exc}")
        record["latency_s"] = time.perf_counter() - started
        self.samples.append(record)

    async def run_chain(self, client: httpx.AsyncClient, operations: list[dict]) -> None:
        for operation in operations:
            if self.deadline is not None and time.perf_counter() > self.deadline:
                self.timed_out = True
                return
            await self.call(client, operation)

    async def run(self) -> dict:
        chains = self.chains()
        started = time.perf_counter()
        if self.args.timeout:
            self.deadline = started + self.args.timeout
        limits = httpx.Limits(max_connections=self.args.max_connections)
        async with httpx.AsyncClient(timeout=self.args.timeout or 3600.0, limits=limits) as client:
            try:
                async with asyncio.timeout(self.args.timeout or 3600.0):
                    await asyncio.gather(*(self.run_chain(client, ops) for ops in chains.values()))
            except TimeoutError:
                self.timed_out = True
        return self.report(time.perf_counter() - started)

    def report(self, elapsed: float) -> dict:
        ok = [s for s in self.samples if s.get("ok")]
        failed = [s for s in self.samples if not s.get("ok")]
        expected = int(self.workload.get("llm_count") or 0)
        prompt_tokens = sum(int(s.get("prompt_tokens") or 0) for s in ok)
        completion_tokens = sum(int(s.get("completion_tokens") or 0) for s in ok)
        latencies = sorted(s["latency_s"] for s in ok)
        expected_input = int(self.workload.get("expected_input_tokens") or 0)
        expected_output = int(self.workload.get("expected_output_tokens") or 0)
        served_ok = bool(ok) and all(
            s.get("prompt_tokens") is not None and s.get("model_matches") for s in ok
        )
        output_matches = all(s.get("output_matches") for s in ok)
        # The workload is only the recorded one if every request was issued and the
        # prompt volume matches. A run that quietly sent less must not look faster.
        volume_matches = expected_input == 0 or abs(prompt_tokens - expected_input) <= max(
            1, int(expected_input * 0.001)
        )
        return {
            "complete": (not self.timed_out)
            and not failed
            and len(self.samples) == expected
            and (self.args.output_mode != "fixed" or output_matches),
            "output_mode": self.args.output_mode,
            "output_volume_matches": output_matches,
            "output_mismatched_requests": sum(not s.get("output_matches") for s in ok),
            "timed_out": self.timed_out,
            "elapsed_s": elapsed,
            "requests_expected": expected,
            "requests_issued": len(self.samples),
            "requests_ok": len(ok),
            "requests_failed": len(failed),
            "agent_chains": len(self.chains()),
            "prompt_tokens": prompt_tokens,
            "expected_input_tokens": expected_input,
            "completion_tokens": completion_tokens,
            "expected_output_tokens": expected_output,
            "output_token_ratio": (completion_tokens / expected_output)
            if expected_output
            else None,
            "latency_p50_s": statistics.median(latencies) if latencies else None,
            "latency_p95_s": latencies[int(len(latencies) * 0.95)] if latencies else None,
            "served_model_ok": served_ok,
            "volume_matches": volume_matches,
            "failures": failed[:10],
        }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result_path = args.result_path.expanduser().resolve()
    workload_path = args.workload.expanduser().resolve()

    payload: dict = {
        "metrics": {},
        "validity": {
            "workload_completed": False,
            "served_model_matches_all_requests": False,
            "cleanup_success": True,
        },
        "trial_validity_score": 0.0,
        "telemetry": {
            "task_runner": "evotailor_v2_trace_replay",
            "task_runner_version": 1,
            "workload": str(workload_path),
            "experiment_id": args.experiment_id,
        },
    }

    try:
        workload = json.loads(workload_path.read_text())
    except Exception as exc:  # noqa: BLE001
        payload["failure_reason"] = f"FAILED_WORKLOAD: cannot read workload: {exc}"
        _write_result(result_path, payload)
        return 0

    replay = Replay(workload, args)
    try:
        report = asyncio.run(replay.run())
    except Exception as exc:  # noqa: BLE001
        payload["failure_reason"] = f"FAILED_WORKLOAD: {type(exc).__name__}: {exc}"
        _write_result(result_path, payload)
        return 0

    payload["telemetry"]["replay"] = {k: v for k, v in report.items() if k != "failures"}
    # The wrapper takes this gate from the workload's own telemetry, at the path the
    # AgentSociety runner fills. The replay proves the same thing a different way: it
    # issued every request to one endpoint that serves one model, and every response came
    # back with usage, so no request was answered by anything else.
    payload["telemetry"]["evidence"] = {
        "request": {
            "served_model_matches_all_requests": bool(report["served_model_ok"]),
            "served_model": args.served_model,
            "requests_issued": report["requests_issued"],
            "requests_ok": report["requests_ok"],
        }
    }
    payload["validity"]["served_model_matches_all_requests"] = bool(report["served_model_ok"])
    payload["validity"]["workload_completed"] = bool(
        report["complete"] and report["volume_matches"]
    )
    payload["trial_validity_score"] = 1.0 if payload["validity"]["workload_completed"] else 0.0

    if not payload["validity"]["workload_completed"]:
        reasons = []
        if report["timed_out"]:
            reasons.append("timed out")
        if report["requests_failed"]:
            reasons.append(f"{report['requests_failed']} failed requests")
        if report["requests_issued"] != report["requests_expected"]:
            reasons.append(
                f"issued {report['requests_issued']} of {report['requests_expected']} requests"
            )
        if not report["volume_matches"]:
            reasons.append(
                f"prompt volume {report['prompt_tokens']} does not match recorded "
                f"{report['expected_input_tokens']}"
            )
        if report["output_mode"] == "fixed" and not report["output_volume_matches"]:
            reasons.append("completion token counts differ from fixed recording")
        payload["failure_reason"] = "FAILED_WORKLOAD: " + "; ".join(reasons or ["incomplete"])
        payload["objective_metric"] = OBJECTIVE_METRIC
        # Deliberately no value under OBJECTIVE_METRIC. A run that timed out stopped early,
        # so its elapsed time is shorter than a complete run's and would be read as the best
        # result if it were published. The wrapper refuses a Trial whose configured metric is
        # not a number, which is what a timeout has to be: a failure, never a fast result.
        payload["metrics"] = {
            "request_count": report["requests_issued"],
            "elapsed_s_at_failure": report["elapsed_s"],
        }
        _write_result(result_path, payload)
        return 0

    # The wrapper resolves the objective by name: it reads metrics[objective_metric], and
    # when a top-level objective is present it insists that the payload also declares which
    # metric that number is. Declaring both keeps the contract explicit rather than relying
    # on the fallback.
    payload["objective"] = report["elapsed_s"]
    payload["objective_metric"] = OBJECTIVE_METRIC
    payload["metrics"] = {
        OBJECTIVE_METRIC: report["elapsed_s"],
        "request_count": report["requests_issued"],
        "prompt_tokens": report["prompt_tokens"],
        "completion_tokens": report["completion_tokens"],
        "latency_p50_s": report["latency_p50_s"],
        "latency_p95_s": report["latency_p95_s"],
    }
    payload["telemetry"]["readings"] = report
    _write_result(result_path, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
