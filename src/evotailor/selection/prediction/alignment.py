"""Semantic vLLM -> AIC alignment, with explicit scheduling transformations."""

from __future__ import annotations

import copy
import math

DIRECT = {
    "deployment.tensor_parallel_size": ("tp_size", "Per-replica tensor parallelism"),
    "deployment.pipeline_parallel_size": ("pp_size", "Pipeline stages"),
    "model.kv_cache_dtype": ("kv_cache_dtype", "KV element representation"),
    "vllm.block_size": ("kv_block_size", "KV allocation block size"),
}
DERIVED = {
    "runtime.gpu_memory_utilization": "AIC native KV capacity with fraction of TOTAL device memory",
    "runtime.max_model_len": "Context admission limit checked against AIC KV token capacity",
    "runtime.max_num_seqs": "Upper bound on active requests, not the observed batch size",
    "runtime.max_num_batched_tokens": "Upper bound on scheduled tokens per iteration",
    "runtime.enable_chunked_prefill": "Whether token-budget splitting of prefill is permitted",
    "runtime.enable_prefix_caching": "Reuse observed prefill KV only when cache is enabled",
}
LEARNED = {
    "deployment.data_parallel_size": "Independent vLLM replicas; does not map to AIC attention-DP",
    "runtime.gpu_memory_utilization": "Fraction of total device memory; learned capacity/runtime effects",
    "runtime.max_model_len": "Request context limit; learned admission and allocation effects",
    "vllm.max_cudagraph_capture_size": "Graph execution regime learned from real observations",
}


def mapping_report(parameters: dict, profile: dict) -> list[dict]:
    result = []
    for name, value in sorted(parameters.items()):
        if name in DIRECT:
            path, description = DIRECT[name]
            mode = "direct"
        elif name in DERIVED:
            path = (
                "KVCacheEstimator.of_total"
                if name == "runtime.gpu_memory_utilization"
                else "KVCacheEstimator.context_limit"
                if name == "runtime.max_model_len"
                else "scheduled_requests"
            )
            description = DERIVED[name]
            mode = "derived_workload"
        else:
            path = "configuration_features"
            description = LEARNED.get(
                name, "Configuration feature learned from measured task results"
            )
            mode = "learned"
        result.append(
            {
                "vllm_parameter": name,
                "value": value,
                "aic_input": path,
                "mode": mode,
                "meaning": description,
                "task_feature": True,
            }
        )
    return result


def engine_config(profile: dict, parameters: dict) -> dict:
    config = copy.deepcopy(profile["engine_config"])
    for name, (path, _) in DIRECT.items():
        if name not in parameters:
            continue
        value = parameters[name]
        if name == "model.kv_cache_dtype":
            value = {"auto": None, "fp8": "fp8", "fp8_e4m3": "fp8"}.get(value, value)
        config[path] = value
    # vLLM data-parallel replicas each have their own KV cache and schedule.
    config["attention_dp_size"] = 1
    return config


def candidate_workload(iterations: list, parameters: dict) -> tuple[list, dict]:
    output = []
    splits = 0
    removed_reuse = 0
    seq_cap = int(parameters.get("runtime.max_num_seqs", 2**31 - 1))
    token_cap = int(parameters.get("runtime.max_num_batched_tokens", 2**31 - 1))
    prefix = bool(parameters.get("runtime.enable_prefix_caching", False))
    chunked = bool(parameters.get("runtime.enable_chunked_prefill", True))
    for ranks in iterations:
        for item in ranks:
            record = copy.deepcopy(item)
            record["wall_time"] = 0.0
            shape = record["scheduled_requests"]
            if not prefix and shape.get("num_prefill_requests", 0):
                reused = int(record.get("observed_prefix_cache_hit_tokens", 0))
                shape["sum_prefill_tokens"] = int(shape.get("sum_prefill_tokens", 0)) + reused
                shape["sum_prefill_kv_tokens"] = max(
                    int(shape.get("sum_prefill_kv_tokens", 0)) - reused, 0
                )
                removed_reuse += reused
            prefill = int(shape.get("sum_prefill_tokens", 0))
            decode = int(shape.get("num_decode_requests", 0))
            partitions = max(
                math.ceil((decode + int(shape.get("num_prefill_requests", 0))) / seq_cap),
                math.ceil((prefill + decode) / token_cap),
                1,
            )
            if partitions > 1 and prefill and not chunked:
                return [], {
                    "available": False,
                    "reason": "Measured prefill exceeds candidate scheduling caps with chunking disabled",
                }
            # This is a scheduling estimate over measured workload descriptors,
            # not newly measured telemetry. Preserve total token and request work.
            assigned_decode_kv = 0
            for index in range(partitions):
                piece = copy.deepcopy(record)
                s = piece["scheduled_requests"]
                for field in (
                    "sum_prefill_tokens",
                    "sum_prefill_kv_tokens",
                    "num_decode_requests",
                    "sum_decode_kv_tokens",
                    "num_prefill_requests",
                ):
                    total = int(shape.get(field, 0))
                    q, r = divmod(total, partitions)
                    s[field] = q + int(index < r)
                if s["num_prefill_requests"] == 0 and s["sum_prefill_tokens"] > 0:
                    s["num_prefill_requests"] = 1
                if decode:
                    active = min(decode, partitions)
                    kv = int(shape.get("sum_decode_kv_tokens", 0))
                    s["sum_decode_kv_tokens"] = (
                        (
                            kv - assigned_decode_kv
                            if index == active - 1
                            else round(kv * s["num_decode_requests"] / decode)
                        )
                        if s["num_decode_requests"]
                        else 0
                    )
                    assigned_decode_kv += s["sum_decode_kv_tokens"]
                if s["sum_prefill_tokens"] or s["num_decode_requests"]:
                    output.append([piece])
            splits += partitions - 1
    return output, {
        "available": True,
        "source": "estimated_schedule_from_measured_fpm",
        "measured_source_iterations": len(iterations),
        "estimated_iterations": len(output),
        "extra_partitions": splits,
        "observed_reuse_tokens_removed": removed_reuse,
        "limitations": [
            "FPM aggregate descriptors do not reconstruct individual arrival times or cache hit decisions. Task-level configuration learning models remaining effects."
        ],
    }
