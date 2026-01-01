"""EvoTailor: self-evolving configuration optimization for LLM inference serving.

The package is organised around the two loops and the three mechanisms described in the
paper:

``spaces``      adaptive search spaces (dimensions, domains, constraints)
``loops``       inner optimization loop and outer reflection/evolution loop
``selection``   resource- and prediction-guided candidate selection
``experience``  cross-scenario experience and code reuse
``execution``   backend adapters and the timing decomposition of a trial
``workloads``   AgentSociety and ComfyBench adapters
``metrics``     serving metrics and task-time composition
``baselines``   SCOOT, SLO-Guard and LLM-Tuna under a shared interface
"""

from evotailor.objective import best_trial, best_verified, valid_rate

__version__ = "0.1.0"

__all__ = ["best_trial", "best_verified", "valid_rate", "__version__"]
