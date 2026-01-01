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
                    f"服务模型权重共 {weights / 1024**3:.1f} GiB（{dtype}），"
                    f"来自 {model_path} 的实际文件。"
                ),
                implication=(
                    "这个体积必须在配置声明的显存预算之内。TP 越大，每张卡分担的权重越少；"
                    "显存利用率越低，留给 KV cache 的空间越少。"
                ),
                source=f"{model_path} 的 safetensors 索引",
                grade="measured",
            )
        )
    if memory and count:
        facts.append(
            PriorFact(
                id="prior.hardware.gpu",
                statement=(
                    f"本次分配 {count} 张 GPU，每张 {memory / 1024**3:.0f} GiB 可用显存。"
                ),
                implication=(
                    "TP×PP×DP 的总和不能超过这个卡数；"
                    "并行度越高，单卡权重越小，但通信开销越大。"
                ),
                source="nvidia-smi 实机探测 + /certs/docker/nvidia_visible_devices.txt",
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
                    f"每 token 的 KV cache 约 {per_token} 字节"
                    f"（{full_layers} 个 full-attention 层；线性注意力层只占常数状态）。"
                ),
                implication=(
                    "`max_model_len` 与 `max_num_seqs` 的乘积决定最大 KV 需求，"
                    "把它与上面可用的 KV 容量作比较即可判断一个组合是否装得下。"
                    "装不下会让引擎在启动阶段失败；装得下并不说明它更快或更慢，"
                    "那只能由真实测量回答，因此不要据此排除大值组合。"
                ),
                source=f"{arch_source} 的 num_hidden_layers/layer_types/num_key_value_heads/head_dim",
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
                                    f"TP={tp} 且 gpu_memory_utilization={gmu:.2f} 时，"
                                    f"显存预算 {budget / 1024**3:.1f} GiB "
                                    f"小于权重 {weights / 1024**3:.1f} GiB。"
                                ),
                                implication=(
                                    "这一组合没有空间放 KV cache，引擎必然在启动阶段失败，"
                                    "不应作为候选。"
                                ),
                                source="由 prior.model.weight_bytes 与实测显存预算推导",
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
                        f"本 Campaign {len(succeeded)} 次成功 Trial 中，单次请求数 "
                        f"介于 {min(requests):.0f} 与 {max(requests):.0f}。"
                    ),
                    implication=(
                        "上层任务发出的请求量是有限的；提高并发上限不会增加工作负载，"
                        "只会改变调度形态。"
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
                        f"平均部署耗时 {mean_deploy:.0f} s，平均任务耗时 {mean_task:.0f} s，"
                        f"部署占 {(mean_deploy / (mean_deploy + mean_task)) * 100:.0f}%。"
                    ),
                    implication=(
                        "目标函数同时包含这两段。部署时间主要由权重加载决定，"
                        "任务时间由引擎配置与调度决定；优化其中一段不应恶化另一段。"
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
                    f"{len(items)} 次 Trial 以 {failure_type} 结束，"
                    f"例如 {json.dumps(sample, ensure_ascii=False)}。"
                ),
                implication=(
                    "失败集中在这些配置上。除非有明确理由，否则不要重复相近的组合。"
                    if failure_type.startswith("FAILED_VLLM") or failure_type == "FAILED_GPU"
                    else "这类失败可能与配置无关；先确认是候选问题还是环境问题。"
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
            "以下事实用于缩小搜索范围，仅供参考，不构成硬性过滤。"
            "measured 来自本机实测，derived 由实测值推导。"
        ),
        "facts": [fact.as_dict() for fact in facts],
    }
