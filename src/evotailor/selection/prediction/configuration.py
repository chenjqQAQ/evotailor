"""Configuration-aware learning from real task observations.

The kernel regression learns the measured objective, not an invented conversion
from milliseconds of forward compute to seconds of task execution. Its validation
errors are leave-one-out errors; campaign pre-execution errors are recorded separately.
"""

from __future__ import annotations

import math

import numpy as np


def encode(space: dict, parameters: dict) -> tuple[list[str], np.ndarray]:
    names, values = [], []
    for name, spec in sorted(space["dimensions"].items()):
        value = parameters[name]
        if spec["kind"] in ("integer", "float"):
            lo, hi = float(spec["lower"]), float(spec["upper"])
            if hi > lo:
                names.append(name)
                values.append(2 * (float(value) - lo) / (hi - lo) - 1)
        else:
            domain = spec["choices"]
            if len(domain) > 1:
                for category in domain[1:]:
                    names.append(f"{name}={category}")
                    values.append(float(value == category))
    return names, np.asarray(values, dtype=float)


def kernel(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    width = max(left.shape[1], 1)
    distances = ((left[:, None, :] - right[None, :, :]) ** 2).sum(axis=2) / width
    return 1 + left @ right.T / width + np.exp(-2 * distances)


def _prior(record):
    if "native_base_reference_ms" in record:
        return record["native_base_reference_ms"]
    before = record.get("prediction_before_execution") or {}
    forward = (before.get("diagnostics") or {}).get("forward") or {}
    # Keep the task feature independent of the progressively retuned forward model.
    # Legacy observations had no transferable tuning and used the native base estimate.
    return forward["base_native_ms"] if "base_native_ms" in forward else before.get("native_ms")


def support_report(space: dict, parameters: dict, samples: list[dict]) -> dict:
    """Say plainly whether this candidate lies inside the region the fit has seen.

    A kernel regression still returns a number far from its training points; that
    number is the prior showing through, not an estimate. Left unlabelled it looks
    like evidence, and a selector comparing numbers will reject an unexplored region
    on the strength of it — which is how a search locks itself into whichever corner
    it happened to sample first. The test here is scale free: a candidate is outside
    support when it sits farther from every measurement than the measurements sit
    from each other.
    """
    if not samples:
        return {
            "extrapolated": True,
            "distance_to_nearest": None,
            "measured_spread": None,
            "out_of_range_dimensions": [],
            "reason": "No measured samples exist; all candidates are outside measured coverage.",
        }
    names, x = encode(space, parameters)
    # Which dimensions step outside the values actually measured for them. This is
    # computed even with a single sample, because one measurement still defines what
    # has been seen.
    out_of_range = []
    for name in sorted(space["dimensions"]):
        spec = space["dimensions"][name]
        seen = [s["candidate_parameters"][name] for s in samples]
        value = parameters[name]
        if spec["kind"] in ("integer", "float"):
            low, high = min(seen), max(seen)
            if value < low or value > high:
                out_of_range.append(f"{name}={value} (measured range {low}–{high})")
        elif value not in seen:
            out_of_range.append(f"{name}={value} (measured values {sorted(set(seen))})")
    rows = [encode(space, s["candidate_parameters"])[1] for s in samples]
    width = max(len(x), 1)
    nearest = min(float(((x - row) ** 2).sum() / width) for row in rows)
    spread = 0.0
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            spread = max(spread, float(((rows[i] - rows[j]) ** 2).sum() / width))
    # Two ways to be outside support, and either is enough. The distance test catches a
    # candidate that is far from everything overall, but averaging across dimensions
    # lets a wide spread in one dimension mask a large step in another. A value outside
    # the range actually measured for its own dimension is extrapolation by definition,
    # so it is checked separately rather than being diluted into the average.
    # With one sample there is no spread, so any different point is outside support.
    extrapolated = nearest > spread or bool(out_of_range)
    return {
        "extrapolated": extrapolated,
        "distance_to_nearest": round(nearest, 6),
        "measured_spread": round(spread, 6),
        "out_of_range_dimensions": out_of_range,
        "reason": (
            "The candidate is farther from measured configurations than their maximum pairwise distance, "
            "or exceeds measured values in some dimensions. Its prediction is an extrapolation, "
            "which alone is not grounds for rejecting it."
            if extrapolated
            else "The candidate lies within the measured region."
        ),
    }


def predict_objective(
    space: dict,
    parameters: dict,
    records: list[dict],
    native_ms: float | None,
    *,
    prior_mode: str | None = None,
) -> dict:
    samples = [
        r
        for r in records
        if r.get("status") == "succeeded"
        and isinstance(r.get("objective"), (float, int))
        and math.isfinite(r["objective"])
        and r["objective"] > 0
        and set(r.get("candidate_parameters", {})) == set(space["dimensions"])
    ]
    names, x = encode(space, parameters)
    if len(samples) < 3:
        return {
            "prediction": None,
            "uncertainty": None,
            "method": "objective_model_not_ready",
            "samples": len(samples),
            "feature_names": names,
            "reason": "At least three real task observations are needed for fitting and leave-one-out validation.",
        }
    rows = [encode(space, r["candidate_parameters"])[1] for r in samples]
    use_native = (
        native_ms is not None
        and native_ms > 0
        and all(
            _prior(r) is not None
            and _prior(r) > 0
            and (
                prior_mode is None
                or r.get("native_base_reference_mode") == prior_mode
                or (
                    ((r.get("prediction_before_execution") or {}).get("diagnostics") or {}).get(
                        "forward"
                    )
                    or {}
                ).get("estimator_mode")
                == prior_mode
            )
            for r in samples
        )
    )
    if use_native:
        # AIC prior is an input feature, not a substitute objective measurement.
        # Use fixed physical units rather than a sample-dependent scale. A native
        # prior identical across observed configurations must not amplify a new
        # configuration through a near-zero fitted standard deviation.
        prior = np.log(np.asarray([_prior(r) for r in samples]) / 1000.0)
        rows = [np.append(row, p) for row, p in zip(rows, prior, strict=True)]
        x = np.append(x, math.log(native_ms / 1000.0))
        names.append("aic.log_base_forward_seconds")
    matrix = np.asarray(rows)
    target = np.log([r["objective"] for r in samples])
    center = float(target.mean())
    y = target - center
    gram = kernel(matrix, matrix)
    chosen = None
    for penalty in (0.01, 0.1, 1.0, 10.0):
        inverse = np.linalg.inv(gram + np.eye(len(samples)) * penalty)
        alpha = inverse @ y
        errors = []
        for held_out in range(len(samples)):
            keep = np.arange(len(samples)) != held_out
            training_center = float(target[keep].mean())
            training_gram = gram[np.ix_(keep, keep)]
            training_alpha = np.linalg.solve(
                training_gram + np.eye(int(keep.sum())) * penalty,
                target[keep] - training_center,
            )
            held_out_prediction = training_center + float(gram[held_out, keep] @ training_alpha)
            errors.append(float(target[held_out] - held_out_prediction))
        errors = np.asarray(errors)
        loss = float(np.mean(errors**2))
        if chosen is None or loss < chosen[0]:
            chosen = (loss, penalty, inverse, alpha, errors)
    loss, penalty, inverse, alpha, errors = chosen
    cross = kernel(x[None, :], matrix)[0]
    log_prediction = center + float(cross @ alpha)
    posterior = max(float(kernel(x[None, :], x[None, :])[0, 0] - cross @ inverse @ cross), 0.0)
    deviation = math.sqrt(loss * (1 + posterior / penalty))
    prediction = math.exp(log_prediction)
    uncertainty = prediction * math.expm1(deviation)
    support = support_report(space, parameters, samples)
    result = {
        "prediction": prediction,
        "uncertainty": uncertainty,
        "method": "configuration_kernel_with_aic_prior" if use_native else "configuration_kernel",
        "samples": len(samples),
        "feature_names": names,
        "regularization": penalty,
        "loo_log_rmse": math.sqrt(loss),
        "posterior_variance": posterior,
        "training_observation_ids": [r["observation_id"] for r in samples],
        "support": support,
    }
    if support["extrapolated"]:
        # Keep the number for reference, but stop presenting it as the objective
        # estimate. The selector must decide this candidate on coverage grounds.
        result["extrapolated_estimate"] = prediction
        result["prediction"] = None
        result["uncertainty"] = None
        result["method"] = "outside_fitted_support"
    return result
