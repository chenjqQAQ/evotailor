from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field


@dataclass
class ResidualCalibrator:
    pairs: list[tuple[float, float]] = field(default_factory=list)
    min_affine_samples: int = 8

    def observe(self, predicted: float, actual: float) -> None:
        if math.isfinite(predicted) and math.isfinite(actual):
            self.pairs.append((predicted, actual))

    @property
    def method(self) -> str:
        return (
            "affine"
            if len(self.pairs) >= self.min_affine_samples
            else ("median_bias" if self.pairs else "none")
        )

    def predict(self, raw: float) -> tuple[float, float, str]:
        if not self.pairs:
            return raw, max(abs(raw) * 0.20, 1.0), "none"
        residuals = [a - p for p, a in self.pairs]
        if len(self.pairs) < self.min_affine_samples:
            center = raw + statistics.median(residuals)
        else:
            xs = [x for x, _ in self.pairs]
            ys = [y for _, y in self.pairs]
            xb, yb = statistics.mean(xs), statistics.mean(ys)
            den = sum((x - xb) ** 2 for x in xs)
            slope = sum((x - xb) * (y - yb) for x, y in self.pairs) / den if den else 1.0
            slope = min(max(slope, 0.25), 4.0)
            center = yb + slope * (raw - xb)
        med = statistics.median(residuals)
        maturity_floor = (
            abs(center) * 0.20
            if len(self.pairs) < 3
            else abs(center) * 0.12
            if len(self.pairs) < self.min_affine_samples
            else abs(center) * 0.05
        )
        uncertainty = max(
            1.4826 * statistics.median([abs(x - med) for x in residuals]),
            maturity_floor,
            1e-6,
        )
        return center, uncertainty, self.method
