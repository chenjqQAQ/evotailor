#!/usr/bin/env python3
"""AIC forward prediction + configuration learning over auditable real Trials."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evotailor.selection.prediction.alignment import candidate_workload, engine_config, mapping_report
from evotailor.selection.prediction.configuration import predict_objective


def _load_profile() -> tuple[dict, Path]:
    value = os.environ.get("EVOTAILOR_AIC_PROFILE")
    if not value:
        raise ValueError("EVOTAILOR_AIC_PROFILE is required")
    path = Path(value).expanduser().resolve()
    profile = json.loads(path.read_text())
    state = Path(profile.get("state_path") or path.with_suffix(".observations.jsonl"))
    if not state.is_absolute():
        state = path.parent / state
    return profile, state


def _normalise_iterations(value: Any) -> list:
    if isinstance(value, dict):
        return [[value]]
    if isinstance(value, list):
        return [[x] if isinstance(x, dict) else x for x in value]
    return []


def _identity(profile: dict) -> str:
    payload = {k: profile[k] for k in ("engine_config", "runtime_identity") if k in profile}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _mapped_engine_config(profile, parameters):
    return engine_config(profile, parameters)


def _calibration_context(profile, parameters):
    config = engine_config(profile, parameters)
    # Hardware, software, model and every candidate knob define correction identity.
    encoded = json.dumps(
        {"runtime": _identity(profile), "config": config, "parameters": parameters},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode()).hexdigest(), config, dict(parameters)


def _native_transfer_key(profile, parameters):
    # These are admission/capacity limits. The observed scheduled FPM shape
    # carries their realized workload; they are not independent kernel regimes.
    descriptor_limits = {
        "runtime.gpu_memory_utilization",
        "runtime.max_model_len",
        "runtime.max_num_seqs",
        "runtime.max_num_batched_tokens",
    }
    # KV dtype already participates through resolved engine_config. Omitting
    # vLLM's default and explicitly choosing auto are the same native regime.
    descriptor_limits.add("model.kv_cache_dtype")
    regime = {key: value for key, value in parameters.items() if key not in descriptor_limits}
    payload = {
        "runtime": _identity(profile),
        "engine_config": engine_config(profile, parameters),
        "execution_regime": regime,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _load_records(path: Path) -> list:
    if not path.exists():
        return []
    with path.open() as handle:
        fcntl.flock(handle, fcntl.LOCK_SH)
        return [json.loads(line) for line in handle if line.strip()]


def _read_fpm_jsonl(path: Path) -> list:
    from trial_wrapper import _validate_fpm

    result = []
    seen = set()
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            item = _validate_fpm(json.loads(line))
            key = (item.get("worker_id"), item.get("counter_id"))
            if None in key:
                raise ValueError("FPM artifact lacks worker/counter identity")
            if key in seen:
                raise ValueError(f"Duplicate FPM identity {key}")
            seen.add(key)
            # Replica counters are independent; coincident counters are not a synchronized step.
            result.append([item])
    return result


def _kind(iteration):
    shape = iteration[0]["scheduled_requests"]
    pre = int(shape.get("sum_prefill_tokens", 0))
    dec = int(shape.get("num_decode_requests", 0))
    return "mixed" if pre and dec else "prefill" if pre else "decode" if dec else "empty"


def _sample(iterations: list, maximum: int) -> list:
    """Deterministic bounded workload-stratified sample, keeping measured values intact."""
    by_kind = {k: [] for k in ("prefill", "decode", "mixed")}
    for item in iterations:
        kind = _kind(item)
        if kind != "empty":
            by_kind[kind].append(item)
    result = []
    for items in by_kind.values():
        # Order by workload features so sampling includes the full observed range.
        items = sorted(
            items,
            key=lambda x: (
                x[0]["scheduled_requests"].get("sum_prefill_tokens", 0),
                x[0]["scheduled_requests"].get("num_decode_requests", 0),
                x[0]["scheduled_requests"].get("sum_decode_kv_tokens", 0),
                str(x[0].get("worker_id")),
                x[0].get("counter_id", 0),
            ),
        )
        if len(items) > maximum:
            indices = (
                [round(i * (len(items) - 1) / (maximum - 1)) for i in range(maximum)]
                if maximum > 1
                else [len(items) // 2]
            )
            items = [items[i] for i in indices]
        result.extend(items)
    return result


def _options(profile, parameters):
    options = dict(profile.get("options") or {})
    options.setdefault("max_observations", 128)
    options["max_num_tokens"] = max(
        int(options.get("max_num_tokens", 8192)),
        int(parameters.get("runtime.max_num_batched_tokens", 8192)),
    )
    options["max_batch_size"] = max(
        int(options.get("max_batch_size", 512)), int(parameters.get("runtime.max_num_seqs", 512))
    )
    options["max_kv_tokens"] = max(
        int(options.get("max_kv_tokens", 2000000)),
        int(parameters.get("runtime.max_model_len", 32768))
        * int(parameters.get("runtime.max_num_seqs", 512)),
    )
    return options


def _corrections(model):
    return {
        "min": model.get_min_correction_factor(),
        "max": model.get_max_correction_factor(),
        "avg": model.get_avg_correction_factor(),
    }


def _forward(profile, parameters, workload, observations):
    from aiconfigurator_core.sdk.rust_engine_step import RustForwardPassPerfModel

    options = _options(profile, parameters)
    shapes = [
        row["scheduled_requests"] for iteration in [*workload, *observations] for row in iteration
    ]
    if shapes:
        options["max_num_tokens"] = max(
            options["max_num_tokens"],
            max(s.get("sum_prefill_tokens", 0) + s.get("num_decode_requests", 0) for s in shapes),
        )
    maximum = int(options["max_observations"])
    observations = _sample(observations, maximum)
    mode = profile.get("estimator_mode", "native")
    if mode not in ("native", "regression"):
        raise ValueError("estimator_mode must be explicitly native or regression")
    model = (
        RustForwardPassPerfModel.from_native(engine_config(profile, parameters), options)
        if mode == "native"
        else RustForwardPassPerfModel.from_regression(options)
    )
    try:
        base_values = (
            [model.estimate_forward_pass_time_ms(item) for item in workload]
            if mode == "native"
            else []
        )
        base_native_ms = (
            sum(float(x) for x in base_values)
            if base_values and all(x is not None for x in base_values)
            else None
        )
        if observations:
            model.tune_with_fpms(observations)
        values = [model.estimate_forward_pass_time_ms(item) for item in workload]
        diagnostics = model.diagnostics() | {
            "corrections": _corrections(model),
            "estimator_mode": mode,
            "tuning_sample_count": len(observations),
            "base_native_ms": base_native_ms,
        }
        if not workload or any(x is None for x in values):
            return None, diagnostics | {
                "available": False,
                "reason": "No estimate until matching real FPM workload data are sufficient",
            }
        if any(not math.isfinite(float(x)) or float(x) < 0 for x in values):
            raise ValueError("AIC returned an invalid latency")
        return sum(float(x) for x in values), diagnostics | {"available": True}
    finally:
        model.close()


def _capacity(profile, parameters):
    from aiconfigurator_core.sdk.memory import KVCacheEstimator

    config = engine_config(profile, parameters)
    capacity = profile.get("runtime_identity", {}).get("gpu_memory_capacity_bytes")
    if capacity is None:
        return {
            "available": False,
            "reason": "No measured per-GPU memory capacity in runtime identity",
        }
    context = {
        "source": "aic_native_kv_cache_estimator",
        "memory_fraction_kind": "of_total",
        "candidate_max_model_len": parameters["runtime.max_model_len"],
        "candidate_max_num_seqs": parameters["runtime.max_num_seqs"],
        "candidate_max_num_batched_tokens": parameters["runtime.max_num_batched_tokens"],
    }
    try:
        estimator = KVCacheEstimator.from_request(
            config["model_name"],
            config["system_name"],
            config["backend"],
            config.get("backend_version"),
            max_num_tokens=int(parameters["runtime.max_num_batched_tokens"]),
            max_batch_size=int(parameters["runtime.max_num_seqs"]),
            tp_size=int(config["tp_size"]),
            pp_size=int(config["pp_size"]),
            attention_dp_size=1,
            kvcache_quant_mode="fp8" if config.get("kv_cache_dtype") == "fp8" else None,
        )
        result = estimator.estimate(
            is_of_free=False,
            fraction=float(parameters["runtime.gpu_memory_utilization"]),
            gpu_memory_capacity_bytes_override=int(capacity),
        )
    except Exception as exc:  # noqa: BLE001
        # This estimate is a diagnostic that is stored beside the prediction; nothing
        # gates a Trial on it. The estimator refuses configurations whose activation
        # memory it cannot fit, and it refuses architectures it has no kernel data for.
        # Either way the candidate is still worth MEASURING, so a failure here is
        # recorded as "no estimate" instead of ending the Campaign before any Trial.
        return {
            "available": False,
            "reason": f"{type(exc).__name__}: {exc}",
            **context,
        }
    result.pop("tokens_from_kv_bytes", None)
    return {
        "available": True,
        "estimate": result,
        **context,
    }


def predict(payload: dict, profile: dict, state: Path) -> dict:
    records = [
        r
        for r in _load_records(state)
        if r.get("schema_version") == 3
        and r.get("runtime_identity") == _identity(profile)
        and r.get("objective_contract") == profile.get("objective_contract")
    ]
    initial = _normalise_iterations(profile.get("workload_iterations"))
    source = initial or (_sample(records[0]["iterations"], 4) if records else [])
    fitting_records = records
    if profile.get("estimator_mode", "native") == "native":
        fitting_records = []
        for record in records:
            reference, _ = candidate_workload(source, record["candidate_parameters"])
            base, _ = _forward(profile, record["candidate_parameters"], reference, [])
            # This is a table-derived feature, not a revised historical prediction.
            fitting_records.append(
                record | {"native_base_reference_ms": base, "native_base_reference_mode": "native"}
            )
    normalized_defaults = 0
    if "model.kv_cache_dtype" not in payload["space"]["dimensions"]:
        compatible = []
        for record in fitting_records:
            values = record["candidate_parameters"]
            if values.get("model.kv_cache_dtype") == "auto":
                record = record | {
                    "candidate_parameters": {
                        key: value for key, value in values.items() if key != "model.kv_cache_dtype"
                    }
                }
                normalized_defaults += 1
            compatible.append(record)
        fitting_records = compatible
    predictions = []
    for candidate in payload["candidates"]:
        parameters = candidate["parameters"]
        context, config, _ = _calibration_context(profile, parameters)
        transfer_key = _native_transfer_key(profile, parameters)
        matching = [
            r
            for r in records
            if _native_transfer_key(profile, r["candidate_parameters"]) == transfer_key
        ]
        observed = [it for r in matching for it in r["iterations"]]
        workload, transform = candidate_workload(source, parameters)
        native, diagnostics = _forward(profile, parameters, workload, observed)
        capacity = _capacity(profile, parameters)
        objective = predict_objective(
            payload["space"],
            parameters,
            fitting_records,
            diagnostics.get("base_native_ms"),
            prior_mode=profile.get("estimator_mode", "native"),
        )
        predictions.append(
            {
                "candidate_id": candidate["candidate_id"],
                "prediction": native,
                "calibrated": objective["prediction"],
                "uncertainty": objective["uncertainty"],
                "method": objective["method"],
                "samples": objective["samples"],
                "diagnostics": {
                    "history_provenance": {
                        "training_count": len(records),
                        "source_count": sum(bool(r.get("import_source")) for r in records),
                        "target_count": sum(not r.get("import_source") for r in records),
                        "exact_source_match": any(
                            r.get("import_source") and r.get("candidate_parameters") == parameters
                            for r in records
                        ),
                        "exact_target_match": any(
                            not r.get("import_source")
                            and r.get("candidate_parameters") == parameters
                            for r in records
                        ),
                        "evaluation_group": (
                            "target_repeat"
                            if any(
                                not r.get("import_source")
                                and r.get("candidate_parameters") == parameters
                                for r in records
                            )
                            else "warm_seen"
                            if any(
                                r.get("import_source")
                                and r.get("candidate_parameters") == parameters
                                for r in records
                            )
                            else "warm_unseen"
                            if any(r.get("import_source") for r in records)
                            else "cold_unseen"
                        ),
                        "observation_ids": [r["observation_id"] for r in records],
                    },
                    "prediction_implementation_sha256": hashlib.sha256(
                        Path(__file__).read_bytes()
                        + (
                            Path(__file__).resolve().parents[1]
                            / "src/evotailor/prediction/configuration.py"
                        ).read_bytes()
                        + (
                            Path(__file__).resolve().parents[1]
                            / "src/evotailor/prediction/alignment.py"
                        ).read_bytes()
                    ).hexdigest(),
                    "forward": diagnostics,
                    "objective_model": objective,
                    "native_ms": native,
                    "forward_unit": "ms",
                    "objective_unit": profile.get("objective_unit", "s"),
                    "objective_metric": profile.get("objective_metric"),
                    "calibration_context_key": context,
                    "native_transfer_key": transfer_key,
                    "native_training_observation_ids": [r["observation_id"] for r in matching],
                    "historical_default_kv_normalization_count": normalized_defaults,
                    "parameter_alignment": mapping_report(parameters, profile),
                    "workload_transform": transform,
                    "capacity": capacity,
                    "runtime_identity": _identity(profile),
                    "model_data_provenance": profile.get("provenance", {}),
                },
            }
        )
    return {
        "ok": True,
        "method": "aic_forward_and_configuration_objective",
        "predictions": predictions,
    }


def observe(payload: dict, profile: dict, state: Path) -> dict:
    trial = payload["trial"]
    telemetry = trial.get("telemetry") or {}
    if trial.get("measurement_kind") != "real" or trial.get("status") != "succeeded":
        return {
            "ok": True,
            "accepted_iterations": 0,
            "reason": "Only successful real Trials enter the learning dataset",
        }
    objective = trial.get("objective")
    if not isinstance(objective, (int, float)) or not math.isfinite(objective) or objective <= 0:
        raise ValueError("A duration objective must be finite and positive")
    required = ("workload_completed", "served_model_matches_all_requests", "cleanup_success")
    if any(trial.get("validity", {}).get(k) is not True for k in required):
        raise ValueError("Real measurement validity gates are incomplete")
    expected_version = (profile.get("runtime_identity") or {}).get("serving_backend_version")
    if expected_version:
        status_path = telemetry.get("collector_status_path")
        if not status_path or not Path(status_path).is_file():
            raise ValueError("No actual collector backend version evidence")
        statuses = [
            json.loads(line) for line in Path(status_path).read_text().splitlines() if line.strip()
        ]
        installed = [r for r in statuses if r.get("event") == "collector_installed"]
        if not installed or any(r.get("backend_version") != expected_version for r in installed):
            raise ValueError("Actual collector backend version differs from the runtime identity")
    artifact = telemetry.get("forward_pass_metrics_path")
    if not artifact:
        raise ValueError("Real Trial is missing its FPM artifact")
    artifact = Path(artifact).resolve()
    iterations = _read_fpm_jsonl(artifact)
    if not iterations:
        raise ValueError("Real FPM artifact is empty")
    contract = profile.get("objective_contract")
    if not contract or trial["metrics"].get(contract["metric"]) != objective:
        raise ValueError("Missing objective contract or measured objective mismatch")
    parameters = trial["candidate"]["parameters"]
    context, config, _ = _calibration_context(profile, parameters)
    observation_id = f"{telemetry.get('campaign_id')}:{trial['trial_id']}@{trial['started_at']}"
    sampled = _sample(iterations, int(profile.get("options", {}).get("max_observations", 128)))
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    before = telemetry.get("prediction_before_execution") or {}
    record = {
        "schema_version": 3,
        "measurement_kind": "real",
        "objective_contract": profile.get("objective_contract"),
        "started_at": trial["started_at"],
        "finished_at": trial["finished_at"],
        "observation_id": observation_id,
        "trial_id": trial["trial_id"],
        "runtime_identity": _identity(profile),
        "context_key": context,
        "native_transfer_key": _native_transfer_key(profile, parameters),
        "engine_config": config,
        "candidate_parameters": parameters,
        "iterations": sampled,
        "source_path": str(artifact),
        "source_sha256": digest,
        "collector_source_sha256": telemetry.get("collector_source_sha256"),
        "source_iterations": len(iterations),
        "status": trial["status"],
        "objective": objective,
        "metrics": trial["metrics"],
        "request_shape": ((telemetry.get("evidence") or {}).get("request") or {}).get(
            "request_shape"
        ),
        "prediction_before_execution": before,
        "prequential_error": None
        if before.get("calibrated") is None
        else objective - before["calibrated"],
    }
    state.parent.mkdir(parents=True, exist_ok=True)
    with state.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        existing = [json.loads(line) for line in handle if line.strip()]
        duplicate = next((r for r in existing if r.get("observation_id") == observation_id), None)
        if duplicate:
            if duplicate.get("source_sha256") != digest:
                raise ValueError("Previously recorded FPM artifact changed")
            return {
                "ok": True,
                "accepted_iterations": 0,
                "already_recorded": True,
                "recorded_iterations": duplicate["source_iterations"],
                "observation_id": observation_id,
                "calibration_context_key": context,
            }
        handle.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "ok": True,
        "accepted_iterations": len(iterations),
        "retained_observations": len(sampled),
        "observation_id": observation_id,
        "calibration_context_key": context,
        "calibration_pending": True,
        "prequential_error": record["prequential_error"],
        "source_sha256": digest,
        "objective_accepted": True,
    }


def main():
    try:
        payload = json.load(sys.stdin)
        profile, state = _load_profile()
        operation = sys.argv[1]
        if operation == "predict":
            result = predict(payload, profile, state)
        elif operation == "observe":
            result = observe(payload, profile, state)
        else:
            raise ValueError(f"Unknown operation {operation}")
        print(json.dumps(result, allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
