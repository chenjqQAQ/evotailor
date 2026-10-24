"""Measured facts that help the proposers skip known-bad regions of the space.

Everything here is advisory. Nothing in this module removes a candidate or refuses a
Trial: the proposers and the selector receive these statements, and they are free to
disagree. The one condition that is provably fatal lives in ``feasibility`` instead.

Each fact carries the source it was read from and a grade, so a decision in the
Campaign record can be traced back to the measurement that informed it. No number in
this module is estimated by hand: weights come from the files on disk, the KV figure
is computed from the model's own config, and the observed shape comes from Trials
this Campaign actually ran.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

#: dtype widths for the cache. AIC and vLLM both express this as bytes per element.
_DTYPE_BYTES = {
    "bfloat16": 2,
    "float16": 2,
    "half": 2,
    "float32": 4,
    "float": 4,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
    "fp8": 1,
    "int8": 1,
}


@dataclass(frozen=True)
class PriorFact:
    id: str
    statement: str
    implication: str
    source: str
    grade: str

    def as_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "statement": self.statement,
            "implication": self.implication,
            "source": self.source,
            "grade": self.grade,
        }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def model_weight_bytes(model_path: Path) -> int | None:
    """Bytes of model weights on disk, preferring the exact index total."""
    if not model_path.is_dir():
        return None
    index = _read_json(model_path / "model.safetensors.index.json")
    total = ((index.get("metadata") or {}).get("total_size")) if index else None
    if isinstance(total, int) and total > 0:
        return total
    shards = sorted(model_path.glob("*.safetensors"))
    if not shards:
        return None
    return sum(path.stat().st_size for path in shards if path.is_file()) or None


def kv_bytes_per_token(architecture: dict[str, Any]) -> tuple[int, int] | None:
    """(bytes per token, number of layers that actually hold a growing cache).

    A hybrid model keeps a full KV cache only on its full-attention layers. The
    linear-attention layers carry a constant-size state, so counting all layers
    would overstate the cache by roughly three times for this architecture.
    """
    layers = int(architecture.get("num_hidden_layers") or 0)
    kv_heads = int(architecture.get("num_key_value_heads") or 0)
    head_dim = int(architecture.get("head_dim") or 0)
    dtype = str(architecture.get("dtype") or architecture.get("torch_dtype") or "bfloat16")
    layer_types = architecture.get("layer_types")
    if isinstance(layer_types, list) and layer_types:
        full_layers = sum(1 for item in layer_types if str(item) == "full_attention")
    else:
        interval = int(architecture.get("full_attention_interval") or 0)
        full_layers = layers // interval if interval else layers
    if not (full_layers and kv_heads and head_dim):
        return None
    width = _DTYPE_BYTES.get(dtype.lower(), 2)
    # K and V are each kv_heads * head_dim elements per token per full layer.
    return full_layers * kv_heads * head_dim * 2 * width, full_layers


def gpu_memory_bytes(discovery: dict[str, Any]) -> int | None:
    """Per-GPU memory from the probe that actually ran on this allocation."""
    stdout = ((discovery.get("hardware") or {}).get("gpus") or {}).get("stdout") or ""
    for line in str(stdout).splitlines():
        parts = [item.strip() for item in line.split(",")]
        for part in parts:
            if "MiB" in part:
                try:
                    return int(float(part.replace("MiB", "").strip())) * 1024 * 1024
                except ValueError:
                    continue
    return None


def gpu_count(discovery: dict[str, Any], runtime: dict[str, Any] | None = None) -> int | None:
    """How many GPUs this run may use.

    The account's allocation is an upper bound, not the answer: an arm may be
    deliberately narrowed to fewer devices, as the protocol's single-GPU arms were.
    When the runtime names its devices, that is the number that matters, because it is
    the one the container will actually see.
    """
    declared = (runtime or {}).get("gpu_uuids")
    if isinstance(declared, list) and declared:
        return len(declared)
    allocation = (discovery.get("hardware") or {}).get("allocation") or {}
    value = allocation.get("gpu_count")
    return int(value) if isinstance(value, int) and value > 0 else None


def _architecture(discovery: dict[str, Any]) -> tuple[dict[str, Any], str]:
    context = ((discovery.get("workload") or {}).get("context") or {})
    entry = context.get("model_architecture") or {}
    value = entry.get("value") if isinstance(entry, dict) else None
    source = str(entry.get("source") or "discovery") if isinstance(entry, dict) else "discovery"
    return (value if isinstance(value, dict) else {}), source


def model_facts(runtime: dict[str, Any], discovery: dict[str, Any]) -> list[PriorFact]:
    facts: list[PriorFact] = []
    model_path = Path(str(runtime.get("model_path") or ""))
    weights = model_weight_bytes(model_path)
    memory = gpu_memory_bytes(discovery)
    count = gpu_count(discovery, runtime)
    arch, arch_source = _architecture(discovery)
    dtype = str(arch.get("dtype") or arch.get("torch_dtype") or "bfloat16")

    if weights:
        facts.append(
            PriorFact(
                id="prior.model.weight_bytes",
                statement=(
                    f"Served-model weights occupy {weights / 1024**3:.1f} GiB ({dtype}), "
                    f"measured from files at {model_path}."
                ),
                implication=(
                    "Weights must fit the configured memory budget. Larger TP distributes weights across more devices; "
                    "lower memory utilization leaves less space for the KV cache."
                ),
                source=f"{model_path} safetensors index",
                grade="measured",
            )
        )
    if memory and count:
        facts.append(
            PriorFact(
                id="prior.hardware.gpu",
                statement=(
                    f"Allocated {count} GPUs with {memory / 1024**3:.0f} GiB available memory each."
                ),
                implication=(
                    "TP * PP * DP cannot exceed the allocated device count; "
                    "more parallelism reduces weights per device but increases communication overhead."
                ),
                source="nvidia-smi hardware probe + /certs/docker/nvidia_visible_devices.txt",
                grade="measured",
            )
        )
    kv = kv_bytes_per_token(arch)
    if kv:
        per_token, full_layers = kv
        facts.append(
            PriorFact(
                id="prior.model.kv_bytes_per_token",
                statement=(
                    f"The KV cache uses approximately {per_token} bytes per token "
                    f"({full_layers} full-attention layers; linear-attention layers use constant state)."
                ),
                implication=(
                    "The product of `max_model_len` and `max_num_seqs` determines maximum KV demand; "
                    "compare it with available KV capacity to check whether a configuration fits."
                    "Insufficient capacity causes launch failure. Fitting in memory does not imply better or worse performance; "
                    "only real measurements can establish that, so do not reject larger values on that basis alone."
                ),
                source=f"{arch_source} num_hidden_layers/layer_types/num_key_value_heads/head_dim",
                grade="derived",
            )
        )
        if memory:
            for tp in (1, 2):
                # gmu is applied per GPU; the weights are split across tp GPUs.
                for gmu in (0.6, 0.75, 0.9):
                    budget = int(gmu * memory * tp)
                    if weights and weights > budget:
                        facts.append(
                            PriorFact(
                                id=f"prior.feasible.tp{tp}.gmu{gmu:.2f}",
                                statement=(
                                    f"With TP={tp} and gpu_memory_utilization={gmu:.2f}, "
                                    f"the memory budget of {budget / 1024**3:.1f} GiB "
                                    f"is below the weight size of {weights / 1024**3:.1f} GiB."
                                ),
                                implication=(
                                    "This configuration leaves no space for the KV cache and cannot launch; "
                                    "it should not be proposed."
                                ),
                                source="Derived from prior.model.weight_bytes and the measured memory budget",
                                grade="derived",
                            )
                        )
    return facts


def workload_facts(history: Iterable[Any]) -> list[PriorFact]:
    """What the Campaign's own Trials already showed about this workload."""
    trials = list(history)
    if not trials:
        return []
    facts: list[PriorFact] = []
    succeeded = [t for t in trials if t.status.value == "succeeded" and t.objective is not None]
    if succeeded:
        requests = [
            v
            for t in succeeded
            for v in [t.metrics.get("request_count")]
            if isinstance(v, (int, float)) and v > 0
        ]
        deployments = [
            v
            for t in succeeded
            for v in [t.metrics.get("deployment_time_s")]
            if isinstance(v, (int, float)) and v > 0
        ]
        tasks = [
            v
            for t in succeeded
            for v in [t.metrics.get("task_makespan_s")]
            if isinstance(v, (int, float)) and v > 0
        ]
        if requests:
            facts.append(
                PriorFact(
                    id="prior.observed.request_count",
                    statement=(
                        f"Across {len(succeeded)} successful campaign trials, request counts "
                        f"range from {min(requests):.0f} to {max(requests):.0f}."
                    ),
                    implication=(
                        "The workload issues a finite number of requests; raising concurrency does not add work, "
                        "it only changes scheduling."
                    ),
                    source="Real trials completed in this campaign",
                    grade="measured",
                )
            )
        if deployments and tasks:
            mean_deploy = sum(deployments) / len(deployments)
            mean_task = sum(tasks) / len(tasks)
            facts.append(
                PriorFact(
                    id="prior.observed.time_split",
                    statement=(
                        f"Mean deployment time is {mean_deploy:.0f} s and mean task time is {mean_task:.0f} s; "
                        f"deployment accounts for {(mean_deploy / (mean_deploy + mean_task)) * 100:.0f}%."
                    ),
                    implication=(
                        "The objective includes both components. Weight loading contributes to deployment time, "
                        "while engine configuration and scheduling affect task time; improving one should not degrade the other."
                    ),
                    source="Real trials completed in this campaign",
                    grade="measured",
                )
            )
    return facts


def failure_facts(history: Iterable[Any]) -> list[PriorFact]:
    """Which configurations this Campaign already tried and why they failed."""
    trials = [t for t in history if t.status.value != "succeeded"]
    if not trials:
        return []
    grouped: dict[str, list[Any]] = {}
    for trial in trials:
        grouped.setdefault(str(trial.failure_type or "FAILED_UNKNOWN"), []).append(trial)
    facts: list[PriorFact] = []
    for failure_type, items in sorted(grouped.items()):
        sample = items[0].candidate.parameters
        facts.append(
            PriorFact(
                id=f"prior.failure.{failure_type.lower()}",
                statement=(
                    f"{len(items)} trials ended with {failure_type}; "
                    f"for example, {json.dumps(sample, ensure_ascii=False)}."
                ),
                implication=(
                    "Failures concentrate in these configurations; avoid similar configurations unless evidence justifies retrying."
                    if failure_type.startswith("FAILED_VLLM") or failure_type == "FAILED_GPU"
                    else "These failures may be unrelated to configuration; distinguish candidate failures from environment failures first."
                ),
                source="Real trials completed in this campaign",
                grade="measured",
            )
        )
    return facts


def build_prior_knowledge(
    *,
    runtime: dict[str, Any],
    discovery: dict[str, Any],
    history: Iterable[Any] = (),
) -> dict[str, Any]:
    """Everything the proposers and selector may use as background.

    The statement text is written so an agent can act on it; the grades keep the
    distinction between a measured file size and a derived capacity limit, which the
    selector is instructed to preserve.
    """
    trials = list(history)
    facts = [
        *model_facts(runtime, discovery),
        *workload_facts(trials),
        *failure_facts(trials),
    ]
    return {
        "note": (
            "The following facts guide search and are advisory rather than hard filters. "
            "Measured facts come from this machine; derived facts are computed from measurements."
        ),
        "facts": [fact.as_dict() for fact in facts],
    }
