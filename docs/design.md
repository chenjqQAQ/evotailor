# One search, end to end

## Scenario and fixed stage

A scenario fixes the workload protocol, served model, hardware, and inference framework.
Discovery records evidence, supported engine flags, and task characteristics. The space
designer constructs a stage space from that evidence and reusable experience. Dimension
changes require traceable evidence. Unsearched parameters have predetermined defaults.

The implementation calls a fixed-space stage a campaign. `CampaignRunner` freezes its
space before the first evaluation; its episodes organize internal search phases and
collect review evidence. Episodes do not authorize changing an active stage's space or
code. The multi-stage driver is the outer search across campaigns.

## Inner loop

1. Configured generators propose candidates from the frozen space and history.
2. Before execution, capture live resource evidence. Explicit device-count and memory
   infeasibility can reject a candidate for free. Missing measurements do not imply a
   failure. If all candidates are rejected, record their reasons and generate another set.
3. AIC provides inference-forward estimates. The configuration-level model uses valid
   measurements from a compatible space to estimate task performance and uncertainty.
   Missing predictions permit exploration; they do not establish infeasibility.
4. The selector combines resources, history, uncertainty, coverage, and generator evidence.
   Execute only the chosen candidate. Initiated evaluations consume budget, including
   deployment failures.
5. Record the immutable trial, its stage, space hash, code version and resources. Successful
   task measurements update the task model; inference-forward telemetry updates AIC.
6. Update the best verified real result. Its deployment/task components and auxiliary
   metrics belong to that same trial, rather than independently selected metric minima.

## Timing and validity

The wrapper records launch, readiness, and finish on one monotonic clock. Deployment and
task intervals exactly partition E2E. Cleanup and optimizer computation do not silently
change these boundaries. Completion and protocol quality are checked separately from
latency. A failed quality criterion or missing measurements cannot produce a valid winner.
AIC calibration acceptance is an annotation, not a reason to rewrite a completed trial's
quality or measured validity.

## Outer boundary

After a stage ends, reflection reads histories, prediction errors, resource rejections,
generator outcomes, and source context. The next stage may add, narrow, or remove searched
dimensions. Removed dimensions retain fixed deployment defaults. Completed records keep
their original space and are never silently relabelled as equivalent observations for a
changed schema. Compatible evidence may be reused; other records remain conditional
experience or explicitly mapped data.

Policies and code for the next stage are versioned. Code proposals run in isolation and
must pass functional checks plus the configured performance evaluation before adoption.
The objective and measurement/validity kernel are protected from self-modification.

The stage driver divides one total budget between stages and aggregates only target real
trials when selecting a winner. Source-scenario best configurations require new target
evaluations. Fixed-configuration retests are separate from search-budget accounting.
