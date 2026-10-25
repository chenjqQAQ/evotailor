from __future__ import annotations

import json
import math
import os
import subprocess
from typing import Any

from evotailor.selection.prediction.calibration import ResidualCalibrator
from evotailor.contracts import Candidate, Prediction, TrialResult
from evotailor.objective import is_valid_trial
from evotailor.spaces.search_space import SearchSpace


class AIConfiguratorError(RuntimeError):
    pass


class AIConfiguratorAdapter:
    """Prediction-only AIC boundary.

    Real runs require an external bridge and never fall back to the deterministic
    test proxy. The bridge supports `predict` and `observe`; observe may feed
    collected ForwardPassMetrics to AIC's native `tune_with_fpms` implementation.
    """

    def __init__(
        self,
        command: list[str] | None = None,
        *,
        allow_proxy: bool = False,
        environment: dict[str, str] | None = None,
    ) -> None:
        self.command = command or []
        self.environment = environment or {}
        self.allow_proxy = allow_proxy
        self.calibrator = ResidualCalibrator()
        self.raw_by_candidate: dict[str, float] = {}

    def _proxy_score(self, space: SearchSpace, values: dict[str, Any]) -> float:
        if not self.allow_proxy:
            raise AIConfiguratorError("the AIConfigurator test proxy is disabled")
        score = 0.0
        weights = 0.0
        preferences = {
            "runtime.gpu_memory_utilization": 0.88,
            "runtime.max_num_seqs": 96,
            "runtime.max_num_batched_tokens": 16384,
            "vllm.max_cudagraph_capture_size": 128,
        }
        for name, dimension in space.dimensions.items():
            domain = dimension.values()
            value = values[name]
            if name in preferences and isinstance(value, (int, float)):
                width = max(float(domain[-1]) - float(domain[0]), 1.0)
                part = 1 - abs(float(value) - preferences[name]) / width
            elif isinstance(value, bool):
                part = 0.65 if value else 0.45
            elif isinstance(value, (int, float)):
                part = 0.35 + 0.55 * (domain.index(value) / max(len(domain) - 1, 1))
            else:
                part = 0.5
            score += part
            weights += 1
        return 1000.0 + 1000.0 * score / max(weights, 1)

    def _external(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.command:
            raise AIConfiguratorError("aiconfigurator_command is not configured")
        completed = subprocess.run(
            [*self.command, operation],
            input=json.dumps(payload, default=str),
            text=True,
            capture_output=True,
            timeout=600,
            check=False,
            env=os.environ | self.environment,
        )
        if completed.returncode:
            detail = completed.stderr[-2000:] or completed.stdout[-2000:]
            raise AIConfiguratorError(f"bridge exited {completed.returncode}: {detail.strip()}")
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise AIConfiguratorError("bridge returned invalid JSON") from exc
        if not isinstance(result, dict) or result.get("ok") is False:
            raise AIConfiguratorError(str(result.get("error", "bridge prediction failed")))
        return result

    def predict_many(self, space: SearchSpace, candidates: list[Candidate]) -> list[Prediction]:
        if self.command:
            response = self._external(
                "predict",
                {
                    "space": space.model_dump(mode="json"),
                    "space_hash": space.digest,
                    "candidates": [item.model_dump(mode="json") for item in candidates],
                },
            )
            records = response.get("predictions", [])
            by_id = {record["candidate_id"]: record for record in records}
            if len(by_id) != len(records) or set(by_id) != {c.candidate_id for c in candidates}:
                raise AIConfiguratorError(
                    "bridge returned incomplete or duplicate candidate identities"
                )
            output = []
            for candidate in candidates:
                record = by_id[candidate.candidate_id]
                prediction = Prediction(
                    candidate_id=candidate.candidate_id,
                    raw=record.get("prediction"),
                    calibrated=record.get("calibrated"),
                    uncertainty=record.get("uncertainty"),
                    method=record["method"],
                    samples=record.get("samples", 0),
                    diagnostics=record.get("diagnostics", {}),
                )
                if prediction.calibrated is not None:
                    self.raw_by_candidate[candidate.candidate_id] = prediction.calibrated
                else:
                    self.raw_by_candidate.pop(candidate.candidate_id, None)
                output.append(prediction)
            return output
        if not self.allow_proxy:
            raise AIConfiguratorError("real prediction requires aiconfigurator_command")
        output = []
        for candidate in candidates:
            raw = self._proxy_score(space, candidate.parameters)
            self.raw_by_candidate[candidate.candidate_id] = raw
            center, uncertainty, method = self.calibrator.predict(raw)
            output.append(
                Prediction(
                    candidate_id=candidate.candidate_id,
                    raw=raw,
                    calibrated=center,
                    uncertainty=uncertainty,
                    method=f"aiconfigurator_proxy+{method}",
                    samples=len(self.calibrator.pairs),
                )
            )
        return output

    def observe(self, result: TrialResult) -> dict[str, Any]:
        raw = self.raw_by_candidate.get(result.candidate.candidate_id)
        residual_accepted = False
        if (
            is_valid_trial(result, allow_simulated=self.allow_proxy)
            and raw is not None
            and result.objective is not None
            and math.isfinite(result.objective)
        ):
            self.calibrator.observe(raw, result.objective)
            residual_accepted = True
        native = {"accepted_iterations": 0, "reason": "external bridge disabled"}
        if self.command:
            native = self._external("observe", {"trial": result.model_dump(mode="json")})
        return {
            "residual_accepted": residual_accepted,
            "native": native,
        }

    def restore_external(self, results: list[TrialResult]) -> None:
        if not self.command:
            return
        for result in results:
            self._external("observe", {"trial": result.model_dump(mode="json")})
