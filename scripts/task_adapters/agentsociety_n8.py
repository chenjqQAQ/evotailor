"""Native AgentSociety contract for EvoTailor.

This module owns the launch semantics that EvoTailor needs in order to run
the upper-layer task. Its launch contract follows the AgentSociety CLI and keeps endpoint selection
separate from trial validity checks.

Two facts drive the design:

* ``agentsociety2.society.cli`` validates ``AGENTSOCIETY_LLM_API_KEY`` while it
  is being imported, so the endpoint environment must be complete before the
  interpreter starts.
* The OpenAI-compatible endpoint is selected purely through the
  ``AGENTSOCIETY_LLM_*`` and ``AGENTSOCIETY_CODER_LLM_*`` variables, which is
  how a Trial points the society at the vLLM instance that EvoTailor started.

Evaluation reads the run artifacts only. It never trusts the runner's own
summary, and it independently proves that the workload really called the
served model by inspecting the AgentSociety OpenTelemetry trace.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Failure labels shared with the EvoTailor kernel (see contracts.FailureType).
FAILED_WORKLOAD = "FAILED_WORKLOAD"
FAILED_TIMEOUT = "FAILED_TIMEOUT"

#: Artifacts AgentSociety must leave behind for a Trial to count as complete.
DEFAULT_REQUIRED_ARTIFACTS = ("pid.json", "SOCIETY.json", "SOCIETY_STEP.json")
#: Per-step questionnaire results; one file per questionnaire step.
DEFAULT_RESULT_GLOBS = ("artifacts/questionnaire_step_*.json",)
#: The span name AgentSociety uses for one model request.
LLM_SPAN_NAME = "llm.completion"

#: AgentSociety records the served model under different attribute keys
#: depending on the call site (memory consolidation uses the underscore form),
#: so every known spelling must be checked before concluding a model is absent.
MODEL_ATTRIBUTE_KEYS = ("llm.model", "llm_model")


def span_model(attributes: dict[str, Any]) -> str:
    for key in MODEL_ATTRIBUTE_KEYS:
        value = attributes.get(key)
        if value:
            return str(value)
    return ""


#: AgentSociety runtime requirements on this machine.
#:
#: Without ``AGENTSOCIETY_LLM_RAY_MAX_WORKERS`` and the concurrency cap, Ray
#: spawns workers far beyond what the host tolerates and they all die during
#: registration; ``RAY_worker_register_timeout_seconds`` then has to be raised
#: so a slow first registration is not treated as a crash. The thinking flag is
#: required because Qwen3 emits a reasoning block that breaks tool-call
#: parsing. These were taken from the last configuration that really worked on
#: this host and are re-verified by running the workload.
#: Sampling settings pinned on every call the simulated agents make to the served
#: model. Without them vLLM applies its own default temperature and no seed, so two
#: runs of the same scenario make different numbers of model calls and the same
#: configuration can look fast in one Trial and slow in the next. Greedy decoding
#: with a fixed seed removes most of that spread; it cannot remove all of it, because
#: continuous batching and eight concurrent agents still change batch composition.
SERVED_MODEL_SEED = 20260927
SERVED_MODEL_SAMPLING: dict[str, Any] = {
    "temperature": 0,
    "seed": SERVED_MODEL_SEED,
    "chat_template_kwargs": {"enable_thinking": False},
}

#: Thread caps. The container reports the host's 128 cores through ``nproc`` even
#: though its CFS quota is 6, so torch/OMP size their pools from 128 and exhaust the
#: frozen pids limit; that failure has nothing to do with the candidate. The workload
#: runs beside vLLM, so both sides must cap at the same number.
THREAD_LIMIT = "6"
THREAD_ENVIRONMENT: dict[str, str] = {
    "OMP_NUM_THREADS": THREAD_LIMIT,
    "MKL_NUM_THREADS": THREAD_LIMIT,
    "OPENBLAS_NUM_THREADS": THREAD_LIMIT,
    "NUMEXPR_NUM_THREADS": THREAD_LIMIT,
    "RAYON_NUM_THREADS": THREAD_LIMIT,
}

#: Kept as a pure literal on purpose. The discovery step reads this name out of the
#: module with ``ast.literal_eval`` to record the execution contract, and a computed
#: value would make that read fail. ``test_agentsociety_contract_literal`` checks that
#: the literal still matches SERVED_MODEL_SAMPLING and THREAD_ENVIRONMENT below.
DEFAULT_FIXED_ENVIRONMENT: dict[str, str] = {
    "MEM0_TELEMETRY": "False",
    "ANONYMIZED_TELEMETRY": "False",
    "AGENTSOCIETY_LLM_DEFAULT_EXTRA_BODY": (
        '{"chat_template_kwargs":{"enable_thinking":false},"seed":20260927,"temperature":0}'
    ),
    "AGENTSOCIETY_LLM_RAY_MAX_WORKERS": "1",
    "AGENTSOCIETY_LLM_RAY_CONCURRENCY": "1",
    "AGENTSOCIETY_BATCH_SIZE": "256",
    "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    "RAY_worker_register_timeout_seconds": "300",
    "OMP_NUM_THREADS": "6",
    "MKL_NUM_THREADS": "6",
    "OPENBLAS_NUM_THREADS": "6",
    "NUMEXPR_NUM_THREADS": "6",
    "RAYON_NUM_THREADS": "6",
}


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_agent_count(config_path: Path) -> int:
    """Read the society size from the scenario instead of hard-coding it."""
    payload = _read_json(config_path)
    agents = payload.get("agents")
    if not isinstance(agents, list) or not agents:
        raise ValueError(f"{config_path} does not declare a non-empty agents list")
    return len(agents)


def read_step_count(steps_path: Path) -> int:
    payload = _read_json(steps_path) if steps_path.suffix == ".json" else None
    if payload is not None:
        steps = payload.get("steps")
    else:
        import yaml

        steps = (yaml.safe_load(steps_path.read_text()) or {}).get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"{steps_path} does not declare a non-empty steps list")
    return len(steps)


#: Marker that identifies the AgentSociety repository root, which is the
#: working directory the society expects (the scenario lives inside it).
PROJECT_ROOT_MARKER = Path("packages") / "agentsociety2"


def find_project_root(start: Path) -> Path:
    """Walk up from a scenario path to the AgentSociety repository root."""
    for candidate in [start, *start.parents]:
        if (candidate / PROJECT_ROOT_MARKER).is_dir():
            return candidate
    raise ValueError(
        f"could not locate an AgentSociety project root above {start}; "
        f"expected a directory containing {PROJECT_ROOT_MARKER}"
    )


@dataclass(frozen=True)
class AgentSocietyContract:
    """Everything needed to launch and judge one AgentSociety run."""

    project_root: Path
    config_path: Path
    steps_path: Path
    python_executable: Path
    expected_agents: int
    expected_steps: int
    timeout_s: int
    required_artifacts: tuple[str, ...] = DEFAULT_REQUIRED_ARTIFACTS
    result_globs: tuple[str, ...] = DEFAULT_RESULT_GLOBS
    questionnaire_requirements: dict[str, list[str]] = field(default_factory=dict)
    questionnaire_targets: dict[str, list[int]] = field(default_factory=dict)
    fixed_environment: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_FIXED_ENVIRONMENT)
    )

    @classmethod
    def from_paths(
        cls,
        config_path: Path,
        steps_path: Path,
        python_executable: Path,
        timeout_s: int,
        expected_agents: int | None = None,
    ) -> AgentSocietyContract:
        config_path = config_path.expanduser().resolve()
        steps_path = steps_path.expanduser().resolve()
        python_executable = python_executable.expanduser().resolve()
        if not config_path.is_file():
            raise ValueError(f"scenario init config not found: {config_path}")
        if not steps_path.is_file():
            raise ValueError(f"scenario steps file not found: {steps_path}")
        if not python_executable.is_file():
            raise ValueError(f"AgentSociety interpreter not found: {python_executable}")
        declared = read_agent_count(config_path)
        if expected_agents is not None and expected_agents != declared:
            raise ValueError(
                f"expected_agents={expected_agents} disagrees with the scenario's "
                f"{declared} agents in {config_path}"
            )
        import yaml

        steps = (
            json.loads(steps_path.read_text())
            if steps_path.suffix == ".json"
            else yaml.safe_load(steps_path.read_text())
        )
        requirements = {
            str(step["questionnaire_id"]): [
                str(question["id"]) for question in step.get("questions", [])
            ]
            for step in steps["steps"]
            if step.get("type") == "questionnaire"
        }
        return cls(
            questionnaire_requirements=requirements,
            questionnaire_targets={
                str(step["questionnaire_id"]): step["target_agent_ids"]
                for step in steps["steps"]
                if step.get("type") == "questionnaire" and step.get("target_agent_ids") is not None
            },
            project_root=find_project_root(config_path),
            config_path=config_path,
            steps_path=steps_path,
            python_executable=python_executable,
            expected_agents=declared,
            expected_steps=read_step_count(steps_path),
            timeout_s=timeout_s,
        )

    def build_launch(
        self,
        run_dir: Path,
        experiment_id: str,
        api_base: str,
        served_model: str,
        log_file: Path,
    ) -> tuple[list[str], dict[str, str]]:
        """Return the argv and environment for one society run.

        The endpoint variables are the only mechanism AgentSociety offers for
        selecting a model server, so they are set explicitly rather than
        inherited.
        """
        argv = [
            str(self.python_executable),
            "-m",
            "agentsociety2.society.cli",
            "--config",
            str(self.config_path),
            "--steps",
            str(self.steps_path),
            "--run-dir",
            str(run_dir),
            "--experiment-id",
            experiment_id,
            "--log-level",
            "INFO",
            "--log-file",
            str(log_file),
        ]
        environment = {
            "AGENTSOCIETY_LLM_API_KEY": "not-needed",
            "AGENTSOCIETY_LLM_API_BASE": api_base,
            "AGENTSOCIETY_LLM_MODEL": served_model,
            "AGENTSOCIETY_CODER_LLM_API_KEY": "not-needed",
            "AGENTSOCIETY_CODER_LLM_API_BASE": api_base,
            "AGENTSOCIETY_CODER_LLM_MODEL": served_model,
        }
        environment.update(self.fixed_environment)
        return argv, environment

    # ── evidence ────────────────────────────────────────────────────────────

    def _missing_artifacts(self, run_dir: Path) -> list[str]:
        missing = [name for name in self.required_artifacts if not (run_dir / name).is_file()]
        for pattern in self.result_globs:
            if not list(run_dir.glob(pattern)):
                missing.append(pattern)
        return missing

    @staticmethod
    def read_trace(run_dir: Path) -> list[dict[str, Any]]:
        """Read every OpenTelemetry span AgentSociety wrote for this run."""
        spans: list[dict[str, Any]] = []
        trace_dir = run_dir / "trace"
        if not trace_dir.is_dir():
            return spans
        for path in sorted(trace_dir.glob("*.jsonl")):
            for line in path.read_text(errors="ignore").splitlines():
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    spans.append(item)
        return spans

    @classmethod
    def request_evidence(cls, spans: list[dict[str, Any]], served_model: str) -> dict[str, Any]:
        """Derive request-level metrics and prove the served model was used.

        ``llm.model`` is written by AgentSociety itself, so it is independent
        evidence that the society really called the vLLM instance EvoTailor
        started rather than an external API or a stub.
        """
        completions = [span for span in spans if span.get("name") == LLM_SPAN_NAME]
        input_tokens = 0
        output_tokens = 0
        latencies_ms: list[float] = []
        models: dict[str, int] = {}
        error_count = 0
        input_lengths: list[int] = []
        output_lengths: list[int] = []
        intervals: list[tuple[int, int]] = []
        for span in completions:
            attributes = span.get("attributes") or {}
            status = span.get("status") or {}
            if str(status.get("code", status.get("status_code", ""))).lower() == "error":
                error_count += 1
            try:
                incoming = int(attributes.get("llm.input_tokens") or 0)
                outgoing = int(attributes.get("llm.output_tokens") or 0)
                input_tokens += incoming
                output_tokens += outgoing
                if attributes.get("llm.input_tokens") is not None:
                    input_lengths.append(incoming)
                if attributes.get("llm.output_tokens") is not None:
                    output_lengths.append(outgoing)
            except (TypeError, ValueError):
                pass
            model = span_model(attributes)
            models[model] = models.get(model, 0) + 1
            started = span.get("start_time_unix_nano")
            ended = span.get("end_time_unix_nano")
            if isinstance(started, int) and isinstance(ended, int) and ended >= started:
                latencies_ms.append((ended - started) / 1e6)
                intervals.append((started, ended))
        matching = models.get(served_model, 0)
        events = sorted(
            (timestamp, delta)
            for start, end in intervals
            for timestamp, delta in ((start, 1), (end, -1))
            if end > start
        )
        active = peak = 0
        for _, delta in events:
            active += delta
            peak = max(peak, active)
        starts = sorted(start for start, _ in intervals)
        gaps_ms = [(b - a) / 1e6 for a, b in zip(starts, starts[1:], strict=False)]

        def lengths(values):
            return (
                {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "p50": _percentile(values, 0.5),
                    "p95": _percentile(values, 0.95),
                }
                if values
                else None
            )

        return {
            "request_count": len(completions),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "mean_request_latency_ms": statistics.fmean(latencies_ms) if latencies_ms else 0.0,
            "p99_request_latency_ms": _percentile(latencies_ms, 0.99),
            "llm_models_seen": models,
            "explicit_error_completion_spans": error_count,
            "request_shape": {
                "source": "AgentSociety llm.completion trace timestamps and usage attributes",
                "input_tokens": lengths(input_lengths),
                "output_tokens": lengths(output_lengths),
                "timed_request_count": len(intervals),
                "peak_in_flight_completion_spans": peak if intervals else None,
                "measurement_scope": "client_completion_span_including_client_queue_wait",
                "median_interarrival_ms": _percentile(gaps_ms, 0.5) if gaps_ms else None,
            },
            "requests_to_served_model": matching,
            "served_model_matches_all_requests": bool(completions) and matching == len(completions),
        }

    def questionnaire_evidence(self, run_dir: Path) -> dict[str, Any]:
        files = sorted((run_dir / "artifacts").glob("questionnaire_step_*.json"))
        seen = set()
        answers = 0
        for path in files:
            payload = _read_json(path)
            if not isinstance(payload, dict):
                raise ValueError(f"Questionnaire is not an object: {path.name}")
            identity = payload.get("questionnaire_id")
            if not identity or identity in seen:
                raise ValueError(f"Missing or duplicate questionnaire identity: {path.name}")
            seen.add(identity)
            targets = payload.get("target_agent_ids") or []
            responses = payload.get("responses") or []
            expected_targets = self.questionnaire_targets.get(identity)
            incorrect_target_count = (
                len(targets) != self.expected_agents
                if expected_targets is None
                else set(targets) != set(expected_targets)
            )
            if incorrect_target_count or len(set(targets)) != len(targets):
                raise ValueError(
                    f"Questionnaire does not target all configured agents: {path.name}"
                )
            responded = [r.get("agent_id") for r in responses]
            if len(responded) != len(targets) or set(responded) != set(targets):
                raise ValueError(f"Questionnaire agent responses are incomplete: {path.name}")
            required = self.questionnaire_requirements.get(identity) or [
                q["id"] for q in payload.get("questions", [])
            ]
            if not required or len(set(required)) != len(required):
                raise ValueError(f"Questionnaire question identities are invalid: {path.name}")
            for response in responses:
                items = response.get("answers") or []
                if len(items) != len(required) or {
                    item.get("question_id") for item in items
                } != set(required):
                    raise ValueError(f"Questionnaire answers are incomplete: {path.name}")
                if any(
                    item.get("parse_success") is not True
                    or item.get("parse_error") is not None
                    or not isinstance(item.get("raw_text"), str)
                    or not item["raw_text"].strip()
                    for item in items
                ):
                    raise ValueError(f"Questionnaire contains failed or empty answers: {path.name}")
                answers += len(items)
        missing = set(self.questionnaire_requirements) - seen
        if missing:
            raise ValueError(f"Configured questionnaires missing: {sorted(missing)}")
        return {
            "questionnaire_count": len(files),
            "questionnaire_ids": sorted(seen),
            "complete_answer_count": answers,
        }

    def evaluate(
        self, run_dir: Path, exit_code: int, served_model: str, stdout_tail: str = ""
    ) -> dict[str, Any]:
        """Judge one finished run from its artifacts alone."""
        evidence: dict[str, Any] = {
            "exit_code": exit_code,
            "run_dir": str(run_dir),
            "task_fixed_environment": self.fixed_environment,
        }
        missing = self._missing_artifacts(run_dir)
        evidence["missing_artifacts"] = missing
        if exit_code != 0:
            return {
                "ok": False,
                "failure_type": FAILED_WORKLOAD,
                "reason": f"AgentSociety exited {exit_code}: {stdout_tail[-400:]}",
                "metrics": {},
                "evidence": evidence,
            }
        if missing:
            return {
                "ok": False,
                "failure_type": FAILED_WORKLOAD,
                "reason": f"AgentSociety did not produce: {', '.join(missing)}",
                "metrics": {},
                "evidence": evidence,
            }

        pid = _read_json(run_dir / "pid.json")
        society = _read_json(run_dir / "SOCIETY.json")
        step = _read_json(run_dir / "SOCIETY_STEP.json")
        evidence["pid"] = pid
        evidence["agent_count"] = len(society.get("agent_specs") or [])
        evidence["terminated"] = step.get("terminated")
        evidence["expected_steps"] = self.expected_steps
        evidence["completed_step_count"] = step.get("completed_step_count")

        if str(pid.get("status")) != "completed":
            return {
                "ok": False,
                "failure_type": FAILED_WORKLOAD,
                "reason": f"AgentSociety pid.json status is {pid.get('status')!r}, not 'completed'",
                "metrics": {},
                "evidence": evidence,
            }
        if evidence["agent_count"] != self.expected_agents:
            return {
                "ok": False,
                "failure_type": FAILED_WORKLOAD,
                "reason": (
                    f"society ran {evidence['agent_count']} agents, expected {self.expected_agents}"
                ),
                "metrics": {},
                "evidence": evidence,
            }
        # ``terminated`` is false on a normal completion (verified against a
        # known-good historical run), so completion is judged by the number of
        # finished steps instead.
        completed_steps = step.get("completed_step_count")
        if not isinstance(completed_steps, int) or completed_steps < self.expected_steps:
            return {
                "ok": False,
                "failure_type": FAILED_WORKLOAD,
                "reason": (
                    f"SOCIETY_STEP.json reports {completed_steps} completed steps, "
                    f"expected {self.expected_steps}"
                ),
                "metrics": {},
                "evidence": evidence,
            }
        simulation_time = pid.get("simulation_time")
        if simulation_time is not None and step.get("current_time") != simulation_time:
            return {
                "ok": False,
                "failure_type": FAILED_WORKLOAD,
                "reason": (
                    "the society did not reach the configured end time: "
                    f"current_time={step.get('current_time')!r} "
                    f"simulation_time={simulation_time!r}"
                ),
                "metrics": {},
                "evidence": evidence,
            }

        try:
            evidence["questionnaires"] = self.questionnaire_evidence(run_dir)
        except (ValueError, KeyError, TypeError) as exc:
            return {
                "ok": False,
                "failure_type": FAILED_WORKLOAD,
                "reason": str(exc),
                "metrics": {},
                "evidence": evidence,
            }

        spans = self.read_trace(run_dir)
        request = self.request_evidence(spans, served_model)
        evidence["span_count"] = len(spans)
        evidence["request"] = request
        if request["request_count"] == 0:
            return {
                "ok": False,
                "failure_type": FAILED_WORKLOAD,
                "reason": "the society produced no llm.completion spans; it never called a model",
                "metrics": {},
                "evidence": evidence,
            }
        if not request["served_model_matches_all_requests"]:
            return {
                "ok": False,
                "failure_type": FAILED_WORKLOAD,
                "reason": (
                    "the society did not talk exclusively to the served model "
                    f"{served_model!r}: saw {request['llm_models_seen']}"
                ),
                "metrics": {},
                "evidence": evidence,
            }

        makespan = None
        try:
            from datetime import datetime

            started = datetime.fromisoformat(str(pid["start_time"]))
            finished = datetime.fromisoformat(str(pid["end_time"]))
            makespan = (finished - started).total_seconds()
        except Exception:
            makespan = None

        metrics: dict[str, float] = {
            "request_count": float(request["request_count"]),
            "input_tokens": float(request["input_tokens"]),
            "output_tokens": float(request["output_tokens"]),
            "mean_request_latency_ms": float(request["mean_request_latency_ms"]),
            "p99_request_latency_ms": float(request["p99_request_latency_ms"]),
            "agentsociety_agent_count": float(evidence["agent_count"]),
            "agentsociety_completed_step_count": float(step.get("completed_step_count") or 0),
        }
        if makespan is not None:
            metrics["agentsociety_simulation_makespan_s"] = float(makespan)
        return {
            "ok": True,
            "failure_type": None,
            "reason": "AgentSociety completed and called the served model for every request",
            "metrics": metrics,
            "evidence": evidence,
        }
