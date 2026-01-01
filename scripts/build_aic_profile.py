#!/usr/bin/env python3
"""Build a real AIConfigurator profile from measured ForwardPassMetrics.

The profile is the contract between the served vLLM instance and
AIConfigurator's native tuner. Every value here is measured, never templated:

* ``engine_config`` comes from the model's own ``config.json`` and the vLLM
  version that really served it;
* ``workload_iterations`` is a stratified sample of iterations the real
  upper-layer task produced, so prediction is scored on the workload that
  actually runs;
* the correction-grid bounds in ``options`` are derived from the measured
  distribution. Bounds taken from a template are the reason a tuner can retain
  samples while no correction bucket ever becomes ready.

Usage:
  build_aic_profile.py --fpm <fpm.jsonl> --model-path <dir> --vllm-version 0.19.1 \
      --tp-size 2 --kv-block-size 16 --output <profile.json> --state-path <fpms.jsonl>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PREFILL = "prefill"
DECODE = "decode"
MIXED = "mixed"


def classify(iteration: dict[str, Any]) -> str:
    scheduled = iteration.get("scheduled_requests") or {}
    has_prefill = int(scheduled.get("sum_prefill_tokens") or 0) > 0
    has_decode = int(scheduled.get("num_decode_requests") or 0) > 0
    if has_prefill and has_decode:
        return MIXED
    if has_prefill:
        return PREFILL
    if has_decode:
        return DECODE
    return "empty"


def load_iterations(path: Path) -> list[dict[str, Any]]:
    iterations: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise TypeError(f"FPM record {line_number} is not an object")
        if float(item.get("wall_time") or 0) <= 0:
            continue
        iterations.append(item)
    if not iterations:
        raise ValueError(f"no usable FPM iterations in {path}")
    return iterations


def stratified_sample(
    iterations: list[dict[str, Any]], per_kind: dict[str, int]
) -> list[list[dict[str, Any]]]:
    """Pick evenly spaced representatives of each workload kind.

    Even spacing keeps the sample deterministic and reproducible from the same
    artifact, which a random sample would not be.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for iteration in iterations:
        grouped.setdefault(classify(iteration), []).append(iteration)
    chosen: list[dict[str, Any]] = []
    for kind in (PREFILL, DECODE, MIXED):
        wanted = per_kind.get(kind, 0)
        pool = grouped.get(kind, [])
        if wanted <= 0 or not pool:
            continue
        if len(pool) <= wanted:
            chosen.extend(pool)
            continue
        step = len(pool) / wanted
        chosen.extend(pool[int(index * step)] for index in range(wanted))
    if not chosen:
        raise ValueError("no representative iterations could be selected")
    # Each chosen iteration becomes one attention-DP-rank list, which is the
    # canonical shape AIConfigurator's tuner expects.
    return [[iteration] for iteration in chosen]


def distribution(iterations: list[dict[str, Any]], field: str) -> dict[str, float]:
    values = sorted(float((i.get("scheduled_requests") or {}).get(field) or 0) for i in iterations)
    if not values:
        return {"min": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    def quantile(fraction: float) -> float:
        return values[min(int(fraction * len(values)), len(values) - 1)]
    return {
        "min": values[0],
        "p50": quantile(0.5),
        "p95": quantile(0.95),
        "p99": quantile(0.99),
        "max": values[-1],
    }


def bound(maximum: float, floor: int) -> int:
    """Round a measured maximum up to a power of two, never below ``floor``."""
    value = max(int(maximum), floor)
    power = 1
    while power < value:
        power *= 2
    return power


def build_profile(args: argparse.Namespace) -> dict[str, Any]:
    fpm_path = args.fpm.expanduser().resolve()
    iterations = load_iterations(fpm_path)
    model_config = json.loads((args.model_path / "config.json").read_text())

    prefill = distribution(iterations, "sum_prefill_tokens")
    decode_requests = distribution(iterations, "num_decode_requests")
    decode_kv = distribution(iterations, "sum_decode_kv_tokens")

    workload = stratified_sample(
        iterations,
        {
            PREFILL: args.prefill_samples,
            DECODE: args.decode_samples,
            MIXED: args.mixed_samples,
        },
    )

    profile = {
        "engine_config": {
            "schema_version": 1,
            # AIConfigurator resolves this as a HuggingFace id or a local
            # directory. This host has no route to HuggingFace, so the local
            # directory is the only value that can be modelled.
            "model_name": args.engine_model_name or str(args.model_path.expanduser().resolve()),
            "system_name": args.system_name,
            "backend": "vllm",
            # This is the AIConfigurator perf-database lookup key, not a claim
            # about what is serving. AIConfigurator ships a100_sxm vLLM data
            # only for 0.14.0 and 0.22.0, and cross-version use is the documented
            # practice, so the nearest measured version is used and the gap is
            # recorded in provenance.
            "backend_version": args.estimator_version,
            "tp_size": args.tp_size,
            "pp_size": args.pp_size,
            "attention_dp_size": args.dp_size,
            "cp_size": None,
            "moe_tp_size": None,
            "moe_ep_size": None,
            "weight_dtype": model_config.get("torch_dtype"),
            "activation_dtype": model_config.get("torch_dtype"),
            "moe_dtype": None,
            "kv_cache_dtype": None,
            "kv_block_size": args.kv_block_size,
            "nextn": None,
            "nextn_accept_rates": None,
            "extra": {},
        },
        # Bounds come from the measured distribution, not from a template: a
        # grid far wider than the real workload parks every observation in one
        # bucket, so no correction ever becomes ready.
        "options": {
            "min_observations": args.min_observations,
            "max_num_tokens": bound(prefill["max"], args.min_num_tokens),
            "max_batch_size": bound(decode_requests["max"], args.min_batch_size),
            "max_kv_tokens": bound(decode_kv["max"], args.min_kv_tokens),
            "min_faster_correction_factor": 0.5,
            "max_slower_correction_factor": 2.0,
        },
        "parameter_map": {
            "deployment.tensor_parallel_size": "tp_size",
            "deployment.pipeline_parallel_size": "pp_size",
            "deployment.data_parallel_size": "attention_dp_size",
            "model.kv_cache_dtype": {"path": "kv_cache_dtype", "values": {"auto": None}},
            "vllm.block_size": "kv_block_size",
        },
        "calibration_context_parameters": ["vllm.max_cudagraph_capture_size"],
        "allow_legacy_global_observations": False,
        "workload_iterations": workload,
        "state_path": str(args.state_path.expanduser().resolve()),
        "provenance": {
            "source_fpm": str(fpm_path),
            "source_fpm_iterations": len(iterations),
            "workload_kind_counts": {
                kind: sum(1 for i in iterations if classify(i) == kind)
                for kind in (PREFILL, DECODE, MIXED, "empty")
            },
            "measured_distribution": {
                "sum_prefill_tokens": prefill,
                "num_decode_requests": decode_requests,
                "sum_decode_kv_tokens": decode_kv,
            },
            "sampled_iterations": len(workload),
            "model_display_name": args.model_name,
            "serving_backend_version": args.vllm_version,
            "estimator_backend_version": args.estimator_version or args.vllm_version,
            "backend_version_gap": (
                None
                if (args.estimator_version or args.vllm_version) == args.vllm_version
                else (
                    "AIConfigurator ships no a100_sxm vLLM perf database for the serving "
                    "version; the nearest measured version is used as the estimator key"
                )
            ),
            "model_head_dim": model_config.get("head_dim"),
        },
    }
    return profile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fpm", required=True, type=Path, help="measured FPM JSONL")
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument(
        "--engine-model-name",
        default=None,
        help="Value AIConfigurator resolves as the model; defaults to the local model directory.",
    )
    parser.add_argument("--model-name", default="Qwen3-4B", help="Human-readable model name.")
    parser.add_argument("--system-name", default="a100_sxm")
    parser.add_argument("--vllm-version", required=True, help="The version that really served.")
    parser.add_argument(
        "--estimator-version",
        default=None,
        help="AIConfigurator perf-database version to estimate with; defaults to --vllm-version.",
    )
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--pp-size", type=int, default=1)
    parser.add_argument("--dp-size", type=int, default=1)
    parser.add_argument("--kv-block-size", type=int, default=16)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--state-path", required=True, type=Path)
    parser.add_argument("--prefill-samples", type=int, default=4)
    parser.add_argument("--decode-samples", type=int, default=6)
    parser.add_argument("--mixed-samples", type=int, default=2)
    parser.add_argument("--min-observations", type=int, default=3)
    parser.add_argument("--min-num-tokens", type=int, default=2048)
    parser.add_argument("--min-batch-size", type=int, default=4)
    parser.add_argument("--min-kv-tokens", type=int, default=4096)
    args = parser.parse_args()

    if args.estimator_version is None:
        args.estimator_version = args.vllm_version
    profile = build_profile(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "engine_config": profile["engine_config"],
                "options": profile["options"],
                "sampled_iterations": len(profile["workload_iterations"]),
                "provenance": profile["provenance"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
