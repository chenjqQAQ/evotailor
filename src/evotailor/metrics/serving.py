"""Serving-side metrics (paper §5.1, Appendix E).

TTFT, TPOT and throughput are read from the engine's Prometheus endpoint rather than
computed from request logs, because the endpoint keeps cumulative histograms that survive
client-side retries and partial failures.

Definitions used throughout the paper:

``TTFT``
    mean of ``vllm:time_to_first_token_seconds`` over the evaluated window.
``TPOT``
    mean inter-token latency, ``vllm:inter_token_latency_seconds``, *excluding* the first
    token (that token is already counted by TTFT).
``throughput``
    output tokens per second of decode time, ``completion_tokens / task_window``.  This is
    the definition used in the paper's tables; aggregate server throughput measured over a
    different interval can rank configurations differently, which is one of the observations
    motivating an end-to-end objective.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

_HIST_LINE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*?)(?:_bucket)?\{?(?P<labels>[^}]*)\}?\s+(?P<value>[-+0-9.eE]+)$"
)
_LE = re.compile(r'le="(?P<le>[^"]+)"')


@dataclass
class HistogramSnapshot:
    """Cumulative histogram: bucket upper bounds and cumulative counts."""

    buckets: dict[float, float]
    count: float
    total: float

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else float("nan")

    def quantile(self, q: float) -> float:
        """Linear interpolation inside the bucket that contains ``q``."""
        if not self.count:
            return float("nan")
        target = q * self.count
        upper = sorted(self.buckets)
        prev_le, prev_c = 0.0, 0.0
        for le in upper:
            c = self.buckets[le]
            if c >= target:
                if c == prev_c or math.isinf(le):
                    return prev_le if math.isinf(le) else le
                width = le - prev_le
                return prev_le + width * (target - prev_c) / (c - prev_c)
            prev_le, prev_c = le, c
        return upper[-1] if upper else float("nan")


def parse_prometheus(text: str, metric: str) -> HistogramSnapshot | None:
    """Parse one histogram family out of a ``/metrics`` exposition."""
    buckets: dict[float, float] = {}
    count = total = 0.0
    seen = False
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _HIST_LINE.match(line)
        if not match:
            continue
        name = match.group("name")
        if name not in (metric, f"{metric}_bucket", f"{metric}_count", f"{metric}_sum"):
            continue
        value = float(match.group("value"))
        labels = match.group("labels") or ""
        if name == f"{metric}_bucket":
            le_match = _LE.search(labels)
            if le_match:
                le = float("inf") if le_match.group("le") in ("+Inf", "Inf") else float(le_match.group("le"))
                buckets[le] = value
                seen = True
        elif name == f"{metric}_count":
            count = value
        elif name == f"{metric}_sum":
            total = value
    if not seen and not count:
        return None
    return HistogramSnapshot(buckets=buckets, count=count, total=total)


@dataclass
class ServingMetrics:
    ttft_mean_s: float
    ttft_p50_s: float
    ttft_p95_s: float
    tpot_mean_s: float
    request_count: int
    completion_tokens: int
    task_window_s: float

    @property
    def throughput_tok_s(self) -> float:
        return self.completion_tokens / self.task_window_s if self.task_window_s else float("nan")

    def as_dict(self) -> dict[str, float]:
        return {
            "TTFT": self.ttft_mean_s,
            "TTFT_p50": self.ttft_p50_s,
            "TTFT_p95": self.ttft_p95_s,
            "TPOT": self.tpot_mean_s,
            "throughput_tok_s": self.throughput_tok_s,
            "request_count": float(self.request_count),
        }


def from_scrape(
    metrics_text: str,
    *,
    completion_tokens: int,
    request_count: int,
    task_window_s: float,
) -> ServingMetrics | None:
    """Build :class:`ServingMetrics` from one ``/metrics`` scrape."""
    ttft = parse_prometheus(metrics_text, "vllm:time_to_first_token_seconds")
    tpot = parse_prometheus(metrics_text, "vllm:inter_token_latency_seconds")
    if ttft is None or tpot is None:
        return None
    return ServingMetrics(
        ttft_mean_s=ttft.mean,
        ttft_p50_s=ttft.quantile(0.50),
        ttft_p95_s=ttft.quantile(0.95),
        tpot_mean_s=tpot.mean,
        request_count=request_count,
        completion_tokens=completion_tokens,
        task_window_s=task_window_s,
    )


@dataclass
class PrefixCacheStats:
    """Prefix-cache effectiveness, reported alongside a configuration.

    Both arms of a comparison must be described by this record: a large end-to-end gap can
    be produced by the cache setting alone, so it is part of the configuration, not an
    implementation detail.
    """

    queries: float = 0.0
    hits: float = 0.0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.queries if self.queries else 0.0


def prefix_cache_stats(metrics_text: str) -> PrefixCacheStats:
    queries = hits = 0.0
    names = {
        "vllm:prefix_cache_queries_total": "q",
        "vllm:prefix_cache_hits_total": "h",
    }
    for line in metrics_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for metric, which in names.items():
            if line.startswith(metric):
                try:
                    value = float(line.rsplit(" ", 1)[-1])
                except ValueError:
                    continue
                if which == "q":
                    queries += value
                else:
                    hits += value
    return PrefixCacheStats(queries=queries, hits=hits)
