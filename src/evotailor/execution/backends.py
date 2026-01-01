from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import signal
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

from evotailor.contracts import Candidate, FailureType, TrialResult, TrialStatus, utc_now
from evotailor.spaces.search_space import SearchSpace


class Executor(ABC):
    @abstractmethod
    def execute(self, trial_id: str, candidate: Candidate) -> TrialResult: ...


class SimulatedExecutor(Executor):
    def __init__(
        self,
        space: SearchSpace,
        seed: int = 42,
        noise: float = 0.01,
        objective_direction: str = "maximize",
    ) -> None:
        self.space = space
        self.seed = seed
        self.noise = noise
        self.objective_direction = objective_direction

    def execute(self, trial_id: str, candidate: Candidate) -> TrialResult:
        started = utc_now()
        x = candidate.parameters
        score = 1100.0
        for name, dim in self.space.dimensions.items():
            domain = dim.values()
            value = x[name]
            if name == "runtime.gpu_memory_utilization":
                part = 1 - (float(value) - 0.875) ** 2 / 0.08
            elif name == "runtime.max_num_seqs":
                part = math.exp(-(((float(value) - 112) / 100) ** 2))
            elif name == "runtime.max_num_batched_tokens":
                part = math.exp(-(((float(value) - 18432) / 15000) ** 2))
            elif name == "runtime.enable_prefix_caching":
                part = 1.0 if value else 0.55
            elif name == "deployment.tensor_parallel_size":
                part = 0.85 if value == 2 else 0.62
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                part = 0.3 + 0.7 * domain.index(value) / max(len(domain) - 1, 1)
            else:
                part = 0.65
            score += 95 * part
        fingerprint = int(hashlib.sha1(json.dumps(x, sort_keys=True).encode()).hexdigest()[:8], 16)
        stable = (fingerprint % 1000) / 1000 - 0.5
        noise_seed = int.from_bytes(
            hashlib.sha256(f"{self.seed}:{trial_id}:{candidate.candidate_id}".encode()).digest()[
                :8
            ],
            "big",
        )
        noise_rng = random.Random(noise_seed)
        score *= 1 + self.noise * (stable + noise_rng.uniform(-0.25, 0.25))
        objective = score if self.objective_direction == "maximize" else 2000.0 - score
        return TrialResult(
            trial_id=trial_id,
            candidate=candidate,
            status=TrialStatus.SUCCEEDED,
            objective=round(objective, 6),
            metrics={"throughput": round(score, 6), "simulated": 1.0},
            measurement_kind="simulated",
            validity={"trial_validity_score": 0.0, "simulated": True},
            started_at=started,
            finished_at=utc_now(),
        )


class CommandExecutor(Executor):
    """Runs a configured host/docker wrapper. The wrapper receives candidate JSON on stdin and returns result JSON on stdout."""

    def __init__(
        self,
        command: list[str],
        timeout_s: float = 900,
        cwd: Path | None = None,
        environment: dict[str, str] | None = None,
    ) -> None:
        if not command:
            raise ValueError("command backend requires a command")
        self.command = command
        self.timeout_s = timeout_s
        self.cwd = cwd
        # Extra variables for the wrapper. The Campaign id travels this way so
        # the wrapper can scope its artifacts per Campaign instead of letting
        # two Campaigns overwrite each other's Trial directories.
        self.environment = dict(environment or {})

    def execute(self, trial_id: str, candidate: Candidate) -> TrialResult:
        started = utc_now()
        try:
            environment = os.environ.copy()
            environment.update(self.environment)
            receipt_value = environment.get("EVOTAILOR_RECEIPT_PATH")
            receipt = Path(receipt_value) if receipt_value else None
            if receipt is not None and receipt.exists():
                from types import SimpleNamespace

                saved = json.loads(receipt.read_text())
                if saved["trial_id"] != trial_id or saved["parameters"] != candidate.parameters:
                    raise ValueError("Execution receipt identity mismatch")
                completed = SimpleNamespace(
                    returncode=0, stdout=json.dumps(saved["result"]), stderr=""
                )
                started = __import__("datetime").datetime.fromisoformat(saved["started_at"])
            else:
                process = subprocess.Popen(
                    self.command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=self.cwd,
                    env=environment,
                    start_new_session=True,
                )
                try:
                    stdout, stderr = process.communicate(
                        json.dumps(
                            {"trial_id": trial_id, "candidate": candidate.model_dump(mode="json")}
                        ),
                        timeout=self.timeout_s,
                    )
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.communicate(timeout=45)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.communicate()
                    raise
                from types import SimpleNamespace

                completed = SimpleNamespace(
                    returncode=process.returncode, stdout=stdout, stderr=stderr
                )
            if completed.returncode:
                raise RuntimeError(
                    f"command exited {completed.returncode}: {completed.stderr[-1000:]}"
                )
            payload = json.loads(completed.stdout)
            objective = float(payload["objective"])
            if not math.isfinite(objective):
                raise ValueError("FAILED_WORKLOAD: objective must be finite")
            validity = dict(payload.get("validity") or {})
            required = (
                "workload_completed",
                "served_model_matches_all_requests",
                "cleanup_success",
            )
            failed = [gate for gate in required if validity.get(gate) is not True]
            if failed:
                raise ValueError(f"FAILED_WORKLOAD: required measurement gates failed: {failed}")
            metrics = {
                str(k): float(v)
                for k, v in payload.get("metrics", {}).items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            validity = dict(payload.get("validity") or {})
            validity.setdefault(
                "trial_validity_score", float(payload.get("trial_validity_score", 0.0))
            )
            return TrialResult(
                trial_id=trial_id,
                candidate=candidate,
                status=TrialStatus.SUCCEEDED,
                objective=objective,
                metrics=metrics,
                telemetry=dict(payload.get("telemetry") or {}),
                measurement_kind="real",
                validity=validity,
                started_at=started,
                finished_at=utc_now(),
            )
        except subprocess.TimeoutExpired as exc:
            return TrialResult(
                trial_id=trial_id,
                candidate=candidate,
                status=TrialStatus.FAILED,
                failure_type=FailureType.TIMEOUT.value,
                error=f"{type(exc).__name__}: command timed out",
                started_at=started,
                finished_at=utc_now(),
            )
        except Exception as exc:
            text = str(exc)
            # A structured label from the wrapper or the workload always wins:
            # keyword guessing is only a fallback for unstructured failures.
            explicit = re.search(r"\bFAILED_[A-Z_]+\b", text)
            if explicit is not None:
                return TrialResult(
                    trial_id=trial_id,
                    candidate=candidate,
                    status=TrialStatus.FAILED,
                    failure_type=explicit.group(0),
                    error=f"{type(exc).__name__}: {text}",
                    started_at=started,
                    finished_at=utc_now(),
                )
            lowered = text.lower()
            failure = FailureType.UNKNOWN.value
            if "fpm" in lowered or "forwardpass" in lowered:
                failure = FailureType.FPM.value
            elif "workload" in lowered:
                failure = FailureType.WORKLOAD.value
            elif "docker" in lowered or "container" in lowered:
                failure = FailureType.DOCKER.value
            elif "ready" in lowered or "health" in lowered:
                failure = FailureType.VLLM_READY.value
            return TrialResult(
                trial_id=trial_id,
                candidate=candidate,
                status=TrialStatus.FAILED,
                failure_type=failure,
                error=f"{type(exc).__name__}: {text}",
                started_at=started,
                finished_at=utc_now(),
            )
