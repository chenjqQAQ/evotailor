#!/usr/bin/env python3
"""Production host/docker Trial wrapper used by CommandExecutor.

The wrapper accepts one Trial JSON object on stdin and prints one result JSON on
stdout. All commands are argv arrays from a trusted YAML file; no shell is used.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evotailor.selection.resources import serving_lease

FPM_VERSION = 1


def _validity_score(validity: dict[str, Any]) -> float:
    """Share of boolean gates that passed.

    ``trial_validity_score`` is excluded: counting the score as one of its own
    gates silently shifts every result and makes the number unexplainable.
    """
    gates = [value for key, value in validity.items() if key != "trial_validity_score"]
    return sum(bool(value) for value in gates) / len(gates) if gates else 0.0


def _validate_fpm(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise TypeError("ForwardPassMetrics record must be a JSON object")
    if int(item.get("version", FPM_VERSION)) != FPM_VERSION:
        raise ValueError(f"unsupported ForwardPassMetrics version: {item.get('version')}")
    wall_time = item.get("wall_time")
    if (
        not isinstance(wall_time, (int, float))
        or isinstance(wall_time, bool)
        or wall_time <= 0
        or not math.isfinite(wall_time)
    ):
        raise ValueError("ForwardPassMetrics wall_time must be positive")
    scheduled = item.get("scheduled_requests")
    if not isinstance(scheduled, dict):
        raise ValueError("ForwardPassMetrics scheduled_requests is required")
    integer_fields = (
        "num_prefill_requests",
        "sum_prefill_tokens",
        "sum_prefill_kv_tokens",
        "num_decode_requests",
        "sum_decode_kv_tokens",
    )
    for name in integer_fields:
        value = scheduled.get(name, 0)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"ForwardPassMetrics {name} must be a non-negative integer")
    if scheduled.get("num_prefill_requests", 0) == 0 and (
        scheduled.get("sum_prefill_tokens", 0) > 0 or scheduled.get("sum_prefill_kv_tokens", 0) > 0
    ):
        raise ValueError("prefill token sums require num_prefill_requests > 0")
    if (
        scheduled.get("num_decode_requests", 0) == 0
        and scheduled.get("sum_decode_kv_tokens", 0) > 0
    ):
        raise ValueError("decode KV sum requires num_decode_requests > 0")
    return item


def _flatten_fpm(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [_validate_fpm(value)]
    if not isinstance(value, list):
        return []
    output: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, list):
            output.extend(_validate_fpm(rank) for rank in item)
        else:
            output.append(_validate_fpm(item))
    return output


def _validate_fpm_file(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            _validate_fpm(json.loads(line))
        except Exception as exc:
            raise ValueError(f"invalid FPM record at {path}:{line_number}: {exc}") from exc
        count += 1
    return count


GPU_QUERY_ARGS = [
    "--query-gpu=timestamp,index,uuid,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
    "--format=csv,noheader,nounits",
]


class HardwareSampler:
    """Sample GPU telemetry while a Trial runs.

    On a platform that hides the GPUs from the host (this one sets
    NVIDIA_VISIBLE_DEVICES=none) a bare nvidia-smi returns nothing, so the
    runtime config may supply a probe command that runs inside the service
    container instead.
    """

    def __init__(self, path: Path, interval_s: float, command: list[str] | None = None) -> None:
        self.path = path
        self.interval_s = max(interval_s, 0.2)
        self.command = list(command) if command else None
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.samples = 0
        self.unavailable_reason: str | None = None

    def start(self) -> None:
        if self.command is None:
            if not shutil.which("nvidia-smi"):
                self.unavailable_reason = "nvidia-smi is not on PATH"
                return
            self.command = ["nvidia-smi", *GPU_QUERY_ARGS]
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        command = self.command or ["nvidia-smi", *GPU_QUERY_ARGS]
        while not self.stop_event.is_set():
            measured_at = time.time()
            completed = subprocess.run(command, text=True, capture_output=True, check=False)
            if completed.returncode == 0:
                with self.path.open("a", encoding="utf-8") as handle:
                    for row in completed.stdout.splitlines():
                        fields = [field.strip() for field in row.split(",")]
                        if len(fields) != 8:
                            continue
                        payload = {
                            "measured_at": measured_at,
                            "timestamp": fields[0],
                            "index": fields[1],
                            "uuid": fields[2],
                            "gpu_utilization_percent": fields[3],
                            "memory_used_mib": fields[4],
                            "memory_total_mib": fields[5],
                            "power_draw_w": fields[6],
                            "temperature_c": fields[7],
                        }
                        handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
                        self.samples += 1
            self.stop_event.wait(self.interval_s)

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=max(self.interval_s * 2, 2.0))
        if self.samples == 0 and self.unavailable_reason is None:
            self.unavailable_reason = (
                "the probe command produced no samples; on this platform the GPUs "
                "are only visible inside the service container"
            )

    def summary(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        groups: dict[str, list[dict[str, Any]]] = {}
        for line in self.path.read_text().splitlines():
            if line.strip():
                item = json.loads(line)
                groups.setdefault(str(item["uuid"]), []).append(item)

        def numbers(items: list[dict[str, Any]], name: str) -> list[float]:
            output = []
            for item in items:
                try:
                    output.append(float(item[name]))
                except (KeyError, TypeError, ValueError):
                    continue
            return output

        per_gpu = {}
        for uuid, items in groups.items():
            utilization = numbers(items, "gpu_utilization_percent")
            memory = numbers(items, "memory_used_mib")
            power = numbers(items, "power_draw_w")
            temperature = numbers(items, "temperature_c")
            per_gpu[uuid] = {
                "samples": len(items),
                "gpu_utilization_mean_percent": sum(utilization) / len(utilization)
                if utilization
                else None,
                "memory_used_max_mib": max(memory) if memory else None,
                "power_draw_mean_w": sum(power) / len(power) if power else None,
                "temperature_max_c": max(temperature) if temperature else None,
            }
        return {"per_gpu": per_gpu}


def _render(items: list[str], variables: dict[str, str]) -> list[str]:
    return [str(item).format_map(variables) for item in items]


def _parameter_args(parameters: dict[str, Any], specs: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for name, value in parameters.items():
        spec = specs.get(name)
        if not isinstance(spec, dict):
            # A configuration error in the harness, not a fault of the candidate.
            # Labelling it lets the kernel record why the Trial never measured
            # anything instead of filing it as an unexplained unknown.
            raise ValueError(
                f"FAILED_CONFIGURATION: missing parameter_arguments mapping for {name}"
            )
        if isinstance(value, bool):
            flag = spec.get("true_flag") if value else spec.get("false_flag")
            if flag:
                result.append(str(flag))
            continue
        flag = spec.get("flag")
        if not flag:
            raise ValueError(f"FAILED_CONFIGURATION: missing flag for {name}")
        result.extend([str(flag), str(value)])
    return result


def _merged_parameter_arguments(config: dict[str, Any]) -> dict[str, Any]:
    """Runtime mappings plus any derived for dimensions the design step added.

    The runtime config only declares the shared base space. A designed dimension
    arrives through ``EVOTAILOR_EXTRA_PARAMETER_ARGUMENTS``, which the kernel
    fills from the flag list the installed vLLM actually reported.
    """
    merged = dict(config.get("parameter_arguments") or {})
    raw = os.environ.get("EVOTAILOR_EXTRA_PARAMETER_ARGUMENTS")
    if raw:
        try:
            extra = json.loads(raw)
        except ValueError as exc:
            raise ValueError(
                f"FAILED_CONFIGURATION: EVOTAILOR_EXTRA_PARAMETER_ARGUMENTS is not JSON: {exc}"
            ) from exc
        if isinstance(extra, dict):
            merged.update(extra)
    return merged


def _wait_ready(url: str, timeout_s: float, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + timeout_s
    last = ""
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"service exited early with code {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if 200 <= response.status < 500:
                    return
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(1)
    raise TimeoutError(f"service health check timed out: {last}")


def _terminate(process: subprocess.Popen[str] | None, timeout_s: float = 20) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=timeout_s)
    except Exception:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def _run_cleanup(
    cleanup_command: list[str],
    working_directory: str,
    environment: dict[str, str],
    timeout_s: float,
) -> bool:
    """Run the cleanup command and report whether it succeeded.

    The same command is used before the service starts and after it stops: a
    Trial that was killed hard cannot clean up after itself, so the next Trial
    must reap the stale container instead of failing on a name or port clash.
    """
    if not cleanup_command:
        return True
    completed = subprocess.run(
        cleanup_command,
        cwd=working_directory,
        env=environment,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    return completed.returncode == 0


def _inherited_resources() -> set[str]:
    """Resources this Trial's own Campaign already holds.

    Returns an empty set unless the token names this very Campaign, so a value
    left in the environment cannot excuse a Trial from locking a card that some
    other Campaign is using.
    """
    raw = os.environ.get("EVOTAILOR_CAMPAIGN_HELD_RESOURCES")
    if not raw:
        return set()
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return set()
    if payload.get("campaign_id") != (os.environ.get("EVOTAILOR_CAMPAIGN_ID") or "unscoped"):
        return set()
    return set(payload.get("resources") or [])


def main() -> int:
    payload = json.load(sys.stdin)
    config_path = os.environ.get("EVOTAILOR_RUNTIME_CONFIG")
    if not config_path:
        raise RuntimeError("EVOTAILOR_RUNTIME_CONFIG is not set")
    config_file = Path(config_path).expanduser().resolve()
    config = yaml.safe_load(config_file.read_text()) or {}
    mode = str(config.get("mode") or "host")
    if mode not in {"host", "docker"}:
        raise ValueError("runtime mode must be host or docker")

    trial_id = str(payload["trial_id"])
    candidate = payload["candidate"]
    parameters = candidate["parameters"]
    started_wall = time.time()
    started_at = datetime.now(UTC).isoformat()

    def interrupted(signum, frame):
        raise TimeoutError("FAILED_TIMEOUT: execution interrupted; cleaning up service")

    signal.signal(signal.SIGTERM, interrupted)
    # A resume reuses the Trial id, so a service name derived only from it can
    # collide with a stale container left by an attempt that was killed before
    # it could clean up. The token makes each attempt's name unique; cleanup
    # reaps by prefix so those stale containers are still collected.
    run_token = f"{os.getpid()}-{int(started_wall)}"
    # run_root is rendered first because it may be scoped per Campaign. Without
    # that, two Campaigns that both number their Trials from trial-0001 would
    # overwrite each other's artifacts.
    identity_variables = {
        "trial_id": trial_id,
        "run_token": run_token,
        "campaign_id": os.environ.get("EVOTAILOR_CAMPAIGN_ID") or "unscoped",
        "config_dir": str(config_file.parent),
    }
    run_root = Path(
        _render([str(config.get("run_root") or "runs/production_trials")], identity_variables)[0]
    ).expanduser()
    if not run_root.is_absolute():
        run_root = config_file.parent / run_root
    attempt_id = os.environ.get("EVOTAILOR_ATTEMPT_ID") or f"{os.getpid()}-{time.time_ns()}"
    trial_dir = run_root / trial_id / attempt_id
    lease = serving_lease(
        config.get("gpu_uuids") or [],
        config.get("health_url"),
        inherited=_inherited_resources(),
    )
    lease.__enter__()
    trial_dir.mkdir(parents=True, exist_ok=True)
    result_path = trial_dir / "workload_result.json"
    fpm_path = trial_dir / "forward_pass_metrics.jsonl"
    hardware_path = trial_dir / "hardware_samples.jsonl"
    result_path.unlink(missing_ok=True)
    fpm_path.unlink(missing_ok=True)
    hardware_path.unlink(missing_ok=True)
    Path(str(fpm_path) + ".collector.jsonl").unlink(missing_ok=True)
    instrumentation_path = Path(__file__).resolve().parent / "vllm_fpm_sitecustomize"
    variables = identity_variables | {
        "trial_dir": str(trial_dir),
        "result_path": str(result_path),
        "fpm_output_path": str(fpm_path),
        "hardware_samples_path": str(hardware_path),
        "instrumentation_path": str(instrumentation_path),
    }
    service_command = _render(list(config.get("service_command") or []), variables)
    workload_command = _render(list(config.get("workload_command") or []), variables)
    cleanup_command = _render(list(config.get("cleanup_command") or []), variables)
    if not service_command or not workload_command:
        raise ValueError("service_command and workload_command are required")
    service_command.extend(_parameter_args(parameters, _merged_parameter_arguments(config)))
    environment = os.environ.copy()
    environment.update({str(k): str(v) for k, v in (config.get("environment") or {}).items()})
    environment["EVOTAILOR_TRIAL_ID"] = trial_id
    environment["EVOTAILOR_CANDIDATE_JSON"] = json.dumps(candidate)
    environment["EVOTAILOR_FPM_OUTPUT_PATH"] = str(fpm_path)
    telemetry_config = dict(config.get("telemetry") or {})
    collect_vllm_fpm = bool(telemetry_config.get("collect_vllm_fpm", True))
    require_fpm = bool(telemetry_config.get("require_forward_pass_metrics", False))
    service_environment = environment.copy()
    if collect_vllm_fpm and mode == "host":
        current_pythonpath = service_environment.get("PYTHONPATH", "")
        service_environment["PYTHONPATH"] = os.pathsep.join(
            item for item in (str(instrumentation_path), current_pythonpath) if item
        )
    hardware_sampler = HardwareSampler(
        hardware_path,
        float(telemetry_config.get("hardware_sample_interval_s", 2.0)),
        _render(list(config.get("hardware_probe_command") or []), variables) or None,
    )

    service: subprocess.Popen[str] | None = None
    cleanup_success = True
    main_result: dict[str, Any] | None = None
    started = time.monotonic()
    working_directory = config.get("working_directory") or config_file.parent
    cleanup_timeout = float(config.get("cleanup_timeout_s") or 120)
    # Reap anything a previous crashed Trial left behind before claiming names.
    _run_cleanup(cleanup_command, working_directory, environment, cleanup_timeout)
    try:
        hardware_sampler.start()
        log = (trial_dir / "service.log").open("w")
        service = subprocess.Popen(
            service_command,
            cwd=config.get("working_directory") or config_file.parent,
            env=service_environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        _wait_ready(
            str(config.get("health_url") or "http://127.0.0.1:8000/health"),
            float(config.get("startup_timeout_s") or 300),
            service,
        )
        deployment_time = time.monotonic() - started
        workload_started = time.monotonic()
        completed = subprocess.run(
            workload_command,
            cwd=config.get("working_directory") or config_file.parent,
            env=environment,
            capture_output=True,
            text=True,
            timeout=float(config.get("workload_timeout_s") or 3600),
            check=False,
        )
        (trial_dir / "workload.stdout.log").write_text(completed.stdout)
        (trial_dir / "workload.stderr.log").write_text(completed.stderr)
        if completed.returncode:
            raise RuntimeError(
                f"workload exited {completed.returncode}: {completed.stderr[-1000:]}"
            )
        workload_time = time.monotonic() - workload_started
        hardware_sampler.stop()
        if result_path.exists():
            workload = json.loads(result_path.read_text())
        else:
            workload = json.loads(completed.stdout)
        if workload.get("ok") is False:
            # Carry the workload's own failure label so the kernel classifies
            # the Trial precisely instead of guessing from stderr text.
            failure = str(workload.get("failure_type") or "FAILED_WORKLOAD")
            raise RuntimeError(f"workload reported failure [{failure}]: {workload.get('reason')}")
        metrics = {
            str(key): float(value)
            for key, value in (workload.get("metrics") or {}).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        metrics.setdefault("deployment_time_s", deployment_time)
        metrics.setdefault("task_makespan_s", workload_time)
        metrics.setdefault("deployment_plus_task_time_s", deployment_time + workload_time)
        telemetry = dict(workload.get("telemetry") or {})
        instrumented_fpm_count = _validate_fpm_file(fpm_path)
        inline_fpms = _flatten_fpm(telemetry.pop("forward_pass_metrics", []))
        if inline_fpms:
            with fpm_path.open("a", encoding="utf-8") as handle:
                for record in inline_fpms:
                    handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        fpm_count = _validate_fpm_file(fpm_path)
        if require_fpm and fpm_count == 0:
            raise RuntimeError(
                "no real ForwardPassMetrics were collected; verify vLLM v1 "
                "instrumentation or the workload telemetry producer"
            )
        hardware_summary = hardware_sampler.summary()
        telemetry.update(
            {
                "forward_pass_metrics_path": str(fpm_path) if fpm_count else None,
                "forward_pass_metrics_count": fpm_count,
                "forward_pass_metrics_source": (
                    "vllm_v1_scheduler_instrumentation+workload"
                    if instrumented_fpm_count and inline_fpms
                    else "vllm_v1_scheduler_instrumentation"
                    if instrumented_fpm_count
                    else "workload"
                    if inline_fpms
                    else "unavailable"
                ),
                "hardware_samples_path": str(hardware_path) if hardware_sampler.samples else None,
                "hardware_samples_count": hardware_sampler.samples,
                "hardware_probe": hardware_sampler.command,
                "hardware_unavailable_reason": hardware_sampler.unavailable_reason,
                "hardware_summary": hardware_summary,
                "telemetry_provenance": {
                    "fpm_schema_version": FPM_VERSION,
                    "collector": "evotailor_v2_vllm_v1_sitecustomize",
                    "runtime_mode": mode,
                },
            }
        )
        metrics["forward_pass_metrics_count"] = float(fpm_count)
        metrics["hardware_samples_count"] = float(hardware_sampler.samples)
        validity = {
            "preflight": True,
            "vllm_started": service is not None,
            "vllm_ready": True,
            "workload_completed": True,
            "task_output_complete": result_path.exists(),
            "request_metrics_complete": bool(
                metrics.get("request_count", 0) or metrics.get("input_tokens", 0)
            ),
            "fpm_collected": fpm_count > 0,
            # Independent proof that the workload really talked to the vLLM
            # instance this wrapper started, taken from the workload's own trace.
            "served_model_matches_all_requests": bool(
                ((telemetry.get("evidence") or {}).get("request") or {}).get(
                    "served_model_matches_all_requests", False
                )
            ),
            # aic_feedback_accepted is deliberately absent: only the kernel knows
            # whether AIConfigurator accepted this Trial's FPM, so the kernel
            # records that fact after observing the Trial.
            "cleanup_success": cleanup_success,
        }
        (trial_dir / "telemetry.json").write_text(
            json.dumps(telemetry, indent=2, default=str) + "\n"
        )
        # The score and the objective are finalised after cleanup below, so the
        # objective can never be computed from a placeholder cleanup result.
        main_result = {"metrics": metrics, "telemetry": telemetry, "validity": validity}
        return 0
    finally:
        hardware_sampler.stop()
        _terminate(service)
        if cleanup_command:
            cleanup_success = _run_cleanup(
                cleanup_command, working_directory, environment, cleanup_timeout
            )
        lease.__exit__(None, None, None)
        if main_result is not None:
            validity = main_result["validity"]
            metrics = main_result["metrics"]
            validity["cleanup_success"] = cleanup_success
            validity["trial_validity_score"] = _validity_score(validity)
            # Expose the validity gate as a metric so a validation Campaign can
            # make it the search objective while performance stays auxiliary.
            metrics["trial_validity_score"] = float(validity["trial_validity_score"])
            objective_metric = str(config.get("objective_metric") or "deployment_plus_task_time_s")
            objective = metrics.get(objective_metric)
            if "objective" in workload and workload.get("objective_metric") != objective_metric:
                raise ValueError(
                    "FAILED_WORKLOAD: workload objective does not name the configured metric"
                )
            if "objective" in workload:
                objective = workload["objective"]
            if not isinstance(objective, (int, float)) or isinstance(objective, bool):
                print(
                    f"ValueError: workload did not provide numeric objective {objective_metric}",
                    file=sys.stderr,
                )
                raise SystemExit(1)
            if not math.isfinite(float(objective)):
                raise ValueError("FAILED_WORKLOAD: non-finite objective")
            if not cleanup_success:
                raise RuntimeError("FAILED_CLEANUP: service cleanup failed")
            main_result["objective"] = float(objective)
            main_result["trial_validity_score"] = validity["trial_validity_score"]
            (trial_dir / "telemetry.json").write_text(
                json.dumps(main_result["telemetry"], indent=2, default=str) + "\n"
            )
            main_result["telemetry"]["attempt_id"] = attempt_id
            main_result["telemetry"]["campaign_id"] = identity_variables["campaign_id"]
            main_result["telemetry"]["collector_status_path"] = str(fpm_path) + ".collector.jsonl"
            main_result["telemetry"]["collector_source_sha256"] = hashlib.sha256(
                (instrumentation_path / "sitecustomize.py").read_bytes()
            ).hexdigest()
            receipt = os.environ.get("EVOTAILOR_RECEIPT_PATH")
            if receipt:
                destination = Path(receipt)
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(".tmp")
                with temporary.open("w") as handle:
                    json.dump(
                        {
                            "trial_id": trial_id,
                            "parameters": parameters,
                            "started_at": started_at,
                            "result": main_result,
                        },
                        handle,
                        allow_nan=False,
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
                temporary.replace(destination)
            print(json.dumps(main_result, allow_nan=False))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
