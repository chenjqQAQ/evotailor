"""One fixed-space stage of the paper inner loop.

CampaignRunner is the executable implementation, shared by the CLI and stage driver.
A campaign corresponds to one fixed-space stage; run_evolution_cycles.py connects
stages under one total budget and applies outer-loop changes only between them.
"""
from pathlib import Path
from evotailor.contracts import CampaignConfig
from evotailor.loops.runner import CampaignRunner


def allocate_stage_budgets(total: int, stages: int, explicit: list[int] | None = None) -> list[int]:
    """Partition the one real-evaluation budget B; never multiply it by stages."""
    if not 1 <= stages <= total:
        raise ValueError("The stage count must be between one and the total budget")
    quotient, remainder = divmod(total, stages)
    budgets = list(explicit) if explicit is not None else [quotient + (index < remainder) for index in range(stages)]
    if len(budgets) != stages or any(value < 1 for value in budgets) or sum(budgets) != total:
        raise ValueError("Stage budgets must be positive and sum to the total budget B")
    return budgets


def run_stage(config: CampaignConfig, project_root: Path) -> dict:
    return CampaignRunner(config, project_root).run()
