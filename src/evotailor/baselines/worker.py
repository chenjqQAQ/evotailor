"""Isolated JSON bridge to the configured upstream optimizer installation.

No approximate algorithm or generated observation is used when upstream code is missing.
The subprocess imports the upstream evotailor package rather than shadowing it with ours.
"""
from __future__ import annotations
import contextlib
import importlib
import json
import sys
from pathlib import Path

ADAPTERS = {
    "scoot": ("evotailor.optimizer_tools.adapters.scoot", "ScootOptimizer"),
    "slo_guard": ("evotailor.optimizer_tools.adapters.slo_guard", "SloGuardOptimizer"),
    "llm_tuna": ("evotailor.optimizer_tools.adapters.llm_tuna", "LlmTunaOptimizer"),
}

def legacy_space(space, root: Path):
    from evotailor.domain.enums import ApplyMethod, ParameterKind, RedeployLevel
    from evotailor.domain.search_space import ParameterDimension, SearchSpaceSpec

    dimensions = []
    for name, dimension in space["dimensions"].items():
        from decimal import Decimal
        kind_name = dimension["kind"]
        if kind_name == "choice":
            values = dimension["choices"]
        else:
            low, high = Decimal(str(dimension["lower"])), Decimal(str(dimension["upper"]))
            step = Decimal(str(dimension.get("step") or (1 if kind_name == "integer" else (high-low)/20)))
            values = [float(low + i*step) for i in range(int((high-low)/step)+1)]
            if kind_name == "integer": values = [int(value) for value in values]
        if kind_name in {"integer", "float"} and dimension.get("step") is not None:
            # A stepped numeric dimension is a discrete ordered grid, and it is handed over
            # as one. Passing lower/upper/step instead lets the other side regenerate the
            # values arithmetically, where 0.3 + 6*0.05 is 0.6000000000000001 rather than
            # 0.6: a configuration this project considers legal is then rejected there, and
            # SLO-Guard ended after 14 Trials because its own ask() would not accept an
            # observed value. Listing the values removes the disagreement entirely.
            kind = ParameterKind.ORDINAL
            kwargs = {"choices": list(values)}
        elif kind_name == "integer":
            kind = ParameterKind.INTEGER
            kwargs = {
                "lower": int(dimension["lower"]),
                "upper": int(dimension["upper"]),
                "step": int(dimension.get("step") or 1),
            }
        elif kind_name == "float":
            kind = ParameterKind.FLOAT
            kwargs = {
                "lower": float(dimension["lower"]),
                "upper": float(dimension["upper"]),
                "step": None,
            }
        else:
            if all(isinstance(value, bool) for value in values):
                kind = ParameterKind.BOOLEAN
            elif all(
                isinstance(value, (int, float)) and not isinstance(value, bool) for value in values
            ):
                kind = ParameterKind.ORDINAL
            else:
                kind = ParameterKind.CATEGORICAL
            kwargs = {"choices": values}
        dimensions.append(
            ParameterDimension(
                name=name,
                category=name.split(".", 1)[0],
                kind=kind,
                default_value=space["safe_baseline"][name],
                provider=space["backend"],
                apply_method=ApplyMethod.CLI_ARGUMENT,
                redeploy_level=RedeployLevel.SERVICE_RESTART,
                supported_backends=[space["backend"]],
                **kwargs,
            )
        )
    return SearchSpaceSpec(
        dimensions=dimensions,
        global_constraints=space.get("global_constraints", []),
    )



def recommend(payload):
    import copy
    payload = copy.deepcopy(payload)
    for name, value in payload["space"].get("fixed_parameters", {}).items():
        payload["space"]["dimensions"][name] = {"kind": "choice", "choices": [value]}
        payload["space"]["safe_baseline"][name] = value
    root = Path(payload["source_root"]).expanduser().resolve()
    if not (root / "evotailor" / "optimizer_tools").is_dir():
        raise RuntimeError("Configured upstream optimizer source_root is unavailable")
    sys.path.insert(0, str(root))
    name = payload["method"]
    module_name, class_name = ADAPTERS[name]
    optimizer_type = getattr(importlib.import_module(module_name), class_name)
    policy = {"budget": payload["budget"], "batch_size": 1, **payload.get("policy", {})}
    optimizer = optimizer_type(**policy)
    task_spec = {"search_space": legacy_space(payload["space"], root).model_dump(mode="json"),
                 "seed": payload["seed"], "max_trials": payload["budget"], **policy}
    history = payload.get("history", [])
    try:
        if name in {"scoot", "llm_tuna"}:
            optimizer.initialize(task_spec, history=history, seed=payload["seed"])
        else:
            optimizer.initialize(task_spec, history=(), seed=payload["seed"])
            if history: optimizer.observe(history)
        result = optimizer.recommend(1)
        if not result.suggestions:
            raise RuntimeError(f"Upstream {name} returned no candidate")
        suggestion = result.suggestions[0]
        return {"parameters": dict(suggestion.parameters), "rationale": suggestion.rationale,
                "provenance": dict(suggestion.provenance),
                "implementation": f"{module_name}.{class_name}"}
    finally:
        with contextlib.suppress(Exception): optimizer.finalize()


if __name__ == "__main__":
    payload = json.load(sys.stdin)
    with contextlib.redirect_stdout(sys.stderr):
        response = recommend(payload)
    print(json.dumps(response, allow_nan=False))
