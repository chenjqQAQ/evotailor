"""Opt-in vLLM v1 scheduler instrumentation for AIConfigurator FPM telemetry.

Python imports ``sitecustomize`` during interpreter startup.  The Trial wrapper
adds this directory to the vLLM service's PYTHONPATH and sets
``EVOTAILOR_FPM_OUTPUT_PATH``.  No vLLM installation files are modified.
"""

from __future__ import annotations

import json
import math
import os
import socket
import statistics
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def _population_variance(values: list[int]) -> float:
    return float(statistics.pvariance(values)) if len(values) > 1 else 0.0


def _rank() -> int:
    for name in ("VLLM_DP_RANK", "DP_RANK", "RANK"):
        try:
            return max(int(os.environ.get(name, "")), 0)
        except ValueError:
            continue
    return 0


def _build_fpm(scheduler_output: Any, wall_time: float, counter_id: int) -> dict[str, Any]:
    new_by_id = {item.req_id: item for item in scheduler_output.scheduled_new_reqs}
    cached = scheduler_output.scheduled_cached_reqs
    cached_index = {req_id: index for index, req_id in enumerate(cached.req_ids)}
    prefill_lengths: list[int] = []
    decode_kv_lengths: list[int] = []
    prefill_tokens = 0
    prefill_kv_tokens = 0
    prefix_cache_hit_tokens = 0

    for req_id, scheduled_tokens in scheduler_output.num_scheduled_tokens.items():
        scheduled_tokens = max(int(scheduled_tokens), 0)
        if req_id in new_by_id:
            request = new_by_id[req_id]
            computed = max(int(getattr(request, "num_computed_tokens", 0)), 0)
            prefill_tokens += scheduled_tokens
            prefill_kv_tokens += computed
            prefix_cache_hit_tokens += computed
            prompt = getattr(request, "prompt_token_ids", None)
            prompt_length = len(prompt) if prompt is not None else computed + scheduled_tokens
            prefill_lengths.append(max(int(prompt_length), 0))
            continue

        index = cached_index.get(req_id)
        if index is None:
            raise RuntimeError(f"FPM collector cannot classify scheduled request {req_id}")
        computed = max(int(cached.num_computed_tokens[index]), 0)
        if cached.is_context_phase(req_id):
            prefill_tokens += scheduled_tokens
            prefill_kv_tokens += computed
            prefill_lengths.append(computed + scheduled_tokens)
        else:
            decode_kv_lengths.append(computed)

    worker_id = f"{socket.gethostname()}:{os.getpid()}:dp{_rank()}"
    return {
        "version": 1,
        "observed_prefix_cache_hit_tokens": prefix_cache_hit_tokens,
        "collection_interval_end_ns": time.time_ns(),
        "measurement_scope": "engine_iteration_wall_time_including_scheduler_executor_synchronization",
        "worker_id": worker_id,
        "dp_rank": _rank(),
        "counter_id": counter_id,
        "wall_time": float(wall_time),
        "scheduled_requests": {
            "num_prefill_requests": len(prefill_lengths),
            "sum_prefill_tokens": prefill_tokens,
            "var_prefill_length": _population_variance(prefill_lengths),
            "sum_prefill_kv_tokens": prefill_kv_tokens,
            "num_decode_requests": len(decode_kv_lengths),
            "sum_decode_kv_tokens": sum(decode_kv_lengths),
            "var_decode_kv_tokens": _population_variance(decode_kv_lengths),
        },
        "queued_requests_measured": False,
        "queued_requests_source": "not_collected_schema_placeholders_ignored_by_aic",
        "queued_requests": {
            "num_prefill_requests": 0,
            "sum_prefill_tokens": 0,
            "var_prefill_length": 0.0,
            "num_decode_requests": 0,
            "sum_decode_kv_tokens": 0,
            "var_decode_kv_tokens": 0.0,
        },
    }


def _append(path: Path, payload: dict[str, Any]) -> None:
    line = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    # One append-mode write keeps each JSONL record intact for the normal
    # single scheduler process and for small records from multiple DP ranks.
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _install() -> None:
    output = os.environ.get("EVOTAILOR_FPM_OUTPUT_PATH")
    if not output:
        return
    try:
        from vllm.v1.engine.core import EngineCore
    except Exception:
        return

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    original = EngineCore.log_iteration_details
    original_init = EngineCore.__init__

    def initialise(self: Any, *args: Any, **kwargs: Any):
        original_init(self, *args, **kwargs)
        self._evotailor_fpm_launches = {}
        execute = self.model_executor.execute_model

        def measured_execute(scheduler_output, *call_args, **call_kwargs):
            self._evotailor_fpm_launches[id(scheduler_output)] = time.monotonic()
            return execute(scheduler_output, *call_args, **call_kwargs)

        self.model_executor.execute_model = measured_execute
        _append(
            Path(str(output_path) + ".collector.jsonl"),
            {
                "event": "collector_installed",
                "worker_id": f"{socket.gethostname()}:{os.getpid()}",
                "clock": "monotonic",
                "wall_time_unit": "s",
                "scope": "execute_model_launch_to_engine_output",
                "backend": "vllm",
                "backend_version": __import__("vllm").__version__,
                "dp_rank": int(self.vllm_config.parallel_config.data_parallel_rank),
            },
        )

    EngineCore.__init__ = initialise

    @contextmanager
    def instrumented(self: Any, scheduler_output: Any):
        started = self._evotailor_fpm_launches.pop(id(scheduler_output), None)
        if started is None:
            raise RuntimeError("FPM collector did not observe execute_model launch")
        with original(self, scheduler_output):
            yield
        elapsed = time.monotonic() - started
        if (
            elapsed <= 0
            or not math.isfinite(elapsed)
            or scheduler_output.total_num_scheduled_tokens <= 0
        ):
            return
        counter = int(getattr(self, "_evotailor_fpm_counter", 0))
        self._evotailor_fpm_counter = counter + 1
        record = _build_fpm(scheduler_output, elapsed, counter)
        record["dp_rank"] = int(self.vllm_config.parallel_config.data_parallel_rank)
        record["measurement_scope"] = "execute_model_launch_to_engine_output"
        _append(output_path, record)

    EngineCore.log_iteration_details = instrumented


_install()
