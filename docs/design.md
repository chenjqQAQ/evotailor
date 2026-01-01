# Design walkthrough

One search, from a scenario descriptor to the reported number.  Section references are to
the paper.

## 0. Inputs

```
scenario   z = (workload, model, hardware, framework)      configs/scenarios.yaml
budget     B real evaluations                              Stage.budget_trials
envelope   T1..T4 (optional)                               configs/envelopes.yaml
experience an optional experience/code bank                 experience/
```

## 1. Build the initial space (paper §4.3)

[`spaces/builder.py`](../src/evotailor/spaces/builder.py) derives dimensions and ranges from
the scenario: the observed peak concurrency bounds the batched-token and sequence
dimensions, the model size bounds the memory-utilization range, and the framework decides
which flags exist at all ([`spaces/engine_flags.py`](../src/evotailor/spaces/engine_flags.py)).

The result is a [`SearchSpace`](../src/evotailor/spaces/base.py) — dimensions, domains and
conditional constraints.  Two invariants are checked before the stage starts
([`spaces/invariant.py`](../src/evotailor/spaces/invariant.py)):

* every dimension reaches the engine flag it claims to control;
* when the space is rebuilt, dimensions that keep their name keep their meaning, so old
  samples stay comparable.

## 2. Inner loop (paper §4.2)

```
repeat until budget exhausted:
    Q_t  = union over generators of g(S_k, H_{t-1})        loops/proposals.py
    keep = {c in Q_t : feasible(c)}                        selection/feasibility.py
    c_t  = selector(keep, resources, predictions, H)       selection/agent.py
    run  = deploy(c_t) + task(c_t)                         execution/trial.py
    H    = H + (c_t, run)                                  loops/history.py
```

Budget accounting: every *deployment* consumes one unit, including failures; a candidate
rejected by the feasibility gate consumes none.  This is why a method that screens well can
evaluate more distinct configurations within the same budget.

## 3. Selection (paper §4.4)

The selector sees three kinds of evidence:

1. **Resources** — current GPU memory headroom, device count, host memory; used to reject
   candidates that cannot be served now, not just nominally.
2. **Inference-forward prediction** `ŷ_AIC(c)` and its calibration state
   ([`selection/prediction/`](../src/evotailor/selection/prediction)).
3. **Task-level estimate** `T̂_task(c)` with an uncertainty band, fitted from the valid
   trials of the current stage.

It ranks survivors by expected end-to-end time *discounted by uncertainty*, so an
unexplored region can still win when the incumbent's prediction is confident and slow.

## 4. Timing (paper §3)

[`execution/timing.py`](../src/evotailor/execution/timing.py) stamps three instants —
launch, ready, finish — and derives `T_deploy`, `T_task`, `T_e2e`.  Optional intermediate
marks (`import`, `engine_init`, `weights`, `warmup`, `serve`) allow the deployment interval
to be attributed afterwards without changing the definition of the objective.

## 5. Metrics (paper §5.1, Appendix E)

Serving metrics are scraped from the engine and parsed as cumulative histograms
([`metrics/serving.py`](../src/evotailor/metrics/serving.py)).  The parse yields the mean
(the sum/count ratio) and interpolated quantiles.  Prefix-cache counters are recorded with
every trial: a configuration that enables prefix caching and one that does not are not
comparable, so the setting is part of the record.

Task time is partitioned into planning-only, planning/execution overlap, execution-only,
judging and other ([`metrics/task.py`](../src/evotailor/metrics/task.py)).  The partition is
computed over interval endpoints, so overlapping spans are never counted twice and the
shares always sum to the task duration.

## 6. Outer loop (paper §4.2, §4.5)

When the stage budget is spent — or the stage has stopped improving
(`loops/inner.py::should_evolve`) — the outer loop runs:

* **reflect**: rank dimensions by how much they explained the observed spread
  ([`experience/review.py`](../src/evotailor/experience/review.py));
* **evolve the space**: add a dimension, drop a dead one, or narrow a range
  (`SearchSpace.evolve`);
* **evolve the code**: the code bank holds versioned generators and selection policy
  fragments; a proposed change is admitted only after the guard accepts it
  ([`experience/guard.py`](../src/evotailor/experience/guard.py));
* **write back**: outcomes, prediction errors and reflected adjustments are appended to the
  experience bank, tagged with the stage and code version that produced them.

## 7. Reporting

`J_n` (best verified end-to-end time) and `η_n` (valid-evaluation rate) are computed from
the trial list, not from a running minimum, so a crash cannot inflate either
([`objective.py`](../src/evotailor/objective.py)).
