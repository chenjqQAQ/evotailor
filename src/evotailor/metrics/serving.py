"""Serving metrics from explicitly bounded measurement windows (Appendix E).

The paper TPOT is the mean of per-request (last-first)/(L-1), excluding
requests with L <= 1. A server inter-token histogram is token-weighted and is
reported under a separate diagnostic name, never silently as paper TPOT.
Throughput uses the serving measurement window, which need not equal task time.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

_HIST_LINE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+"
    r"(?P<value>[-+0-9.eE]+)(?:\s+[0-9]+)?$"
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
                buckets[le] = buckets.get(le, 0.0) + value
                seen = True
        elif name == f"{metric}_count":
            count += value
        elif name == f"{metric}_sum":
            total += value
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
    serving_window_s: float
    tpot_aggregation: str = "per_request_mean"

    @property
    def throughput_tok_s(self) -> float:
        return self.completion_tokens / self.serving_window_s if self.serving_window_s else float("nan")

    def as_dict(self) -> dict[str, float]:
        return {
            "TTFT": self.ttft_mean_s,
            "TTFT_p50": self.ttft_p50_s,
            "TTFT_p95": self.ttft_p95_s,
            ("TPOT" if self.tpot_aggregation == "per_request_mean" else "inter_token_histogram_mean_s"): self.tpot_mean_s,
            "throughput_tok_s": self.throughput_tok_s,
            "request_count": float(self.request_count),
        }


def histogram_delta(after: HistogramSnapshot, before: HistogramSnapshot | None) -> HistogramSnapshot:
    if before is None:
        raise ValueError("The initial histogram snapshot is missing")
    buckets = {key: value - before.buckets.get(key, 0.0) for key, value in after.buckets.items()}
    count, total = after.count - before.count, after.total - before.total
    if count < 0 or total < 0 or any(value < 0 for value in buckets.values()):
        raise ValueError("Histogram counters reset within the serving measurement window")
    return HistogramSnapshot(buckets, count, total)


@dataclass(frozen=True)
class RequestTiming:
    submit_ts: float
    first_token_ts: float
    last_token_ts: float
    output_tokens: int

    def __post_init__(self) -> None:
        if self.output_tokens < 1 or not all(math.isfinite(value) for value in
                (self.submit_ts, self.first_token_ts, self.last_token_ts)):
            raise ValueError("A completed request needs finite timestamps and at least one token")
        if not self.submit_ts <= self.first_token_ts <= self.last_token_ts:
            raise ValueError("Request timestamps must be monotonic")


def from_requests(records: list[RequestTiming], *, serving_window_s: float) -> ServingMetrics:
    """Appendix E.3, with arithmetic averaging over requests rather than tokens."""
    import statistics

    if not records or not math.isfinite(serving_window_s) or serving_window_s <= 0:
        raise ValueError("Completed requests and a positive serving window are required")
    first = sorted(row.first_token_ts - row.submit_ts for row in records)
    per_request = [(row.last_token_ts - row.first_token_ts) / (row.output_tokens - 1)
                   for row in records if row.output_tokens > 1]
    return ServingMetrics(
        ttft_mean_s=statistics.fmean(first),
        ttft_p50_s=first[min(len(first) - 1, math.ceil(0.50 * len(first)) - 1)],
        ttft_p95_s=first[min(len(first) - 1, math.ceil(0.95 * len(first)) - 1)],
        tpot_mean_s=statistics.fmean(per_request) if per_request else math.nan,
        request_count=len(records), completion_tokens=sum(row.output_tokens for row in records),
        serving_window_s=serving_window_s,
    )


def from_scrape(
    metrics_text: str,
    *,
    completion_tokens: int,
    request_count: int,
    serving_window_s: float | None = None,
    task_window_s: float | None = None,
    before_text: str | None = None,
) -> ServingMetrics | None:
    """Read server diagnostics; use before_text for a bounded counter delta.

    Without before_text the scrape must come from a freshly started service.
    task_window_s is a compatibility alias, and must denote the serving window.
    """
    window = serving_window_s if serving_window_s is not None else task_window_s
    if window is None or not math.isfinite(window) or window <= 0:
        raise ValueError("A positive serving measurement window is required")
    ttft = parse_prometheus(metrics_text, "vllm:time_to_first_token_seconds")
    tpot = parse_prometheus(metrics_text, "vllm:inter_token_latency_seconds")
    if ttft is None or tpot is None:
        return None
    if before_text is not None:
        ttft = histogram_delta(ttft, parse_prometheus(before_text, "vllm:time_to_first_token_seconds"))
        tpot = histogram_delta(tpot, parse_prometheus(before_text, "vllm:inter_token_latency_seconds"))
    return ServingMetrics(
        ttft_mean_s=ttft.mean,
        ttft_p50_s=ttft.quantile(0.50),
        ttft_p95_s=ttft.quantile(0.95),
        tpot_mean_s=tpot.mean,
        request_count=request_count,
        completion_tokens=completion_tokens,
        serving_window_s=window,
        tpot_aggregation="token_weighted_histogram",
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
