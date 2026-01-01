#!/usr/bin/env python3
"""Run the AgentSociety upper-layer task for one EvoTailor Trial.

This is the workload side of the real Docker runtime. ``scripts/trial_wrapper.py``
starts vLLM, waits for health, then invokes this runner. The runner:

1. creates a per-Trial run directory inside the Trial directory;
2. points AgentSociety at the vLLM endpoint through its environment variables;
3. executes ``agentsociety2.society.cli``;
4. judges the result from the run artifacts only;
5. writes ``workload_result.json`` for the wrapper.

The runner always exits 0 and reports failure through the result file, so the
kernel can classify the failure precisely instead of guessing from stderr text.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from task_adapters.agentsociety_n8 import (  # noqa: E402
    FAILED_TIMEOUT,
    FAILED_WORKLOAD,
    AgentSocietyContract,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="scenario init_config.json")
    parser.add_argument("--steps", required=True, type=Path, help="scenario steps.yaml")
    parser.add_argument(
        "--python-executable",
        type=Path,
        default=Path("${PYTHON}"),
        help="interpreter that has agentsociety2 installed",
    )
    parser.add_argument("--result-path", required=True, type=Path)
    parser.add_argument("--api-base", required=True, help="OpenAI-compatible base URL")
    parser.add_argument("--served-model", required=True, help="served model name")
    parser.add_argument("--experiment-id", default="evotailor-trial")
    parser.add_argument("--expected-agents", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=3000)
    parser.add_argument("--run-dir-name", default="agentsociety_run")
    return parser.parse_args(argv)


def _write_result(result_path: Path, payload: dict) -> None:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result_path = args.result_path.expanduser().resolve()
    trial_dir = result_path.parent
    run_dir = trial_dir / args.run_dir_name

    started = time.monotonic()
    base_payload: dict = {
        "metrics": {},
        "telemetry": {
            "task_runner": "evotailor_v2_native_agentsociety",
            "task_runner_version": 1,
            "scenario": {
                "config_path": str(args.config),
                "steps_path": str(args.steps),
                "api_base": args.api_base,
                "served_model": args.served_model,
            },
        },
    }

    try:
        contract = AgentSocietyContract.from_paths(
            config_path=args.config,
            steps_path=args.steps,
            python_executable=args.python_executable,
            timeout_s=args.timeout,
            expected_agents=args.expected_agents,
        )
    except Exception as exc:
        payload = base_payload | {
            "ok": False,
            "failure_type": FAILED_WORKLOAD,
            "reason": f"invalid AgentSociety contract: {type(exc).__name__}: {exc}",
        }
        _write_result(result_path, payload)
        print(json.dumps({"ok": False, "reason": payload["reason"]}))
        return 0

    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    log_file = run_dir / "agentsociety.stdout.log"

    command, endpoint_environment = contract.build_launch(
        run_dir=run_dir,
        experiment_id=args.experiment_id,
        api_base=args.api_base,
        served_model=args.served_model,
        log_file=log_file,
    )
    environment = os.environ.copy()
    environment.update(endpoint_environment)
    # The society must reach the local vLLM, never a relay inherited from a shell.
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        environment.pop(name, None)

    (trial_dir / "workload_command.json").write_text(
        json.dumps({"argv": command, "endpoint": endpoint_environment}, indent=2) + "\n"
    )

    timed_out = False
    exit_code = 0
    stdout_tail = ""
    try:
        completed = subprocess.run(
            command,
            cwd=str(contract.project_root),
            env=environment,
            capture_output=True,
            text=True,
            timeout=contract.timeout_s,
            check=False,
        )
        exit_code = completed.returncode
        stdout_tail = (completed.stdout or "")[-4000:]
        (run_dir / "runner.stdout.log").write_text(completed.stdout or "")
        (run_dir / "runner.stderr.log").write_text(completed.stderr or "")
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        exit_code = -1
        stdout_tail = f"timed out after {contract.timeout_s}s"
        (run_dir / "runner.stdout.log").write_text(str(exc.stdout or ""))
        (run_dir / "runner.stderr.log").write_text(str(exc.stderr or ""))

    runner_wall = time.monotonic() - started
    base_payload["metrics"]["task_runner_wall_time_s"] = float(runner_wall)

    if timed_out:
        payload = base_payload | {
            "ok": False,
            "failure_type": FAILED_TIMEOUT,
            "reason": f"AgentSociety exceeded {contract.timeout_s}s",
            "telemetry": base_payload["telemetry"]
            | {"run_dir": str(run_dir), "stdout_tail": stdout_tail},
        }
        _write_result(result_path, payload)
        print(json.dumps({"ok": False, "reason": payload["reason"]}))
        return 0

    verdict = contract.evaluate(
        run_dir=run_dir,
        exit_code=exit_code,
        served_model=args.served_model,
        stdout_tail=stdout_tail,
    )
    payload = base_payload | {
        "ok": bool(verdict["ok"]),
        "failure_type": verdict["failure_type"],
        "reason": verdict["reason"],
        "metrics": base_payload["metrics"] | dict(verdict["metrics"]),
        "telemetry": base_payload["telemetry"]
        | {
            "run_dir": str(run_dir),
            "evidence": verdict["evidence"],
            "stdout_tail": stdout_tail,
        },
    }
    _write_result(result_path, payload)
    print(
        json.dumps(
            {
                "ok": payload["ok"],
                "failure_type": payload["failure_type"],
                "reason": payload["reason"],
                "metrics": payload["metrics"],
            },
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
