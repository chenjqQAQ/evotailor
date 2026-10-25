"""Registry of upstream baseline adapters; no approximate stand-in algorithms."""
from pathlib import Path
from typing import Any

METHODS = ("scoot", "slo_guard", "llm_tuna")


def available() -> list[str]:
    """Supported adapter names; availability of upstream dependencies is checked on use."""
    return list(METHODS)


def build(name: str, *, root: Path, max_trials: int, direction: str = "minimize",
          policy: dict[str, Any] | None = None):
    from evotailor.loops.proposals import LegacyOptimizerProposer
    key = name.lower().replace("-", "_")
    if key not in METHODS:
        raise ValueError(f"Unknown baseline {name!r}; supported adapters: {METHODS}")
    return LegacyOptimizerProposer(key, root, direction, max_trials, policy)
