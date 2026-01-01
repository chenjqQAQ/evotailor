from collections import defaultdict

from evotailor.contracts import ProposerStats


class ReputationTracker:
    def __init__(self) -> None:
        self.stats: dict[str, ProposerStats] = defaultdict(ProposerStats)

    def proposed(self, names: list[str]) -> None:
        for n in names:
            self.stats[n].proposed += 1

    def observed(self, name: str, succeeded: bool, improvement: float) -> None:
        s = self.stats[name]
        s.selected += 1
        s.succeeded += int(succeeded)
        s.improved += int(improvement > 0)
        s.cumulative_improvement += max(improvement, 0)

    def snapshot(self) -> dict:
        return {
            n: s.model_dump() | {"smoothed_success": round(s.smoothed_success, 4)}
            for n, s in sorted(self.stats.items())
        }
