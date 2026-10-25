# Experiment setup and validation boundaries

## Dependencies and assets

Install the Python package and the serving engine in appropriate environments. Supply
model weights, workload traces, ComfyBench tasks, container images, quality evaluators,
and any predictor profiles separately. No measured results or private data are bundled.

Set these values in `.env` as required by the chosen experiment:

| Variable | Purpose |
| --- | --- |
| `EVOTAILOR_DATA_ROOT`, `EVOTAILOR_MODEL_ROOT` | Benchmark assets and served checkpoints |
| `EVOTAILOR_TASK_ROOT` | Workload source inspected by discovery |
| `EVOTAILOR_V2_MODEL` | Search-decision agent; the paper uses gpt-5.6-sol |
| `EVOTAILOR_V2_BASE_URL`, `EVOTAILOR_PROPOSER_API_KEY` | Agent endpoint and credential |
| `EVOTAILOR_RUNTIME_CONFIG` | Fully configured host/container runtime YAML |
| `EVOTAILOR_AIC_PROFILE` | Measured model/engine profile for the external AIC bridge |
| `EVOTAILOR_MATURE_ROOT` | External optimizer adapter installation |
| `EVOTAILOR_BASELINE_PYTHON` | Interpreter with the upstream optimizer dependencies |

A relative `mature_optimizer_root` is resolved against the project root. Its
source directory must contain the `evotailor.optimizer_tools.adapters` package with
`ScootOptimizer`, `SloGuardOptimizer`, and `LlmTunaOptimizer` in modules `scoot`,
`slo_guard`, and `llm_tuna`. The worker uses the adapters' initialize/recommend/observe/
finalize lifecycle and preserves suggestion provenance. These adapters must load the
corresponding upstream algorithms, with their versions and licenses recorded. The code
fails explicitly when they are missing; it does not ship guessed baseline algorithms.

AIConfigurator is prediction-only. Real execution requires the configured external AIC
bridge; its deterministic proxy is allowed only for simulated orchestration checks.


## Connect an original upper-layer task

AgentSociety and ComfyBench are external projects, not source directories vendored into
EvoTailor. Install their original code and dependencies separately. EvoTailor includes
only integration glue: launching a configured command, passing the selected service
endpoint/model, checking task artifacts, and reading the measurement result.

Set `service_command`, `health_url`, `workload_command`, and `cleanup_command` in the
runtime YAML. Commands are argument arrays, without shell evaluation. The workload can
write its result to `{result_path}` or emit JSON on stdout. It must report `ok`, its
metrics and telemetry; when required, it must also report `quality_passed`. The request
evidence in `telemetry.evidence.request.served_model_matches_all_requests` must confirm
that requests reached the selected served model. Merely returning an exit code of zero
is not proof of a completed valid workload.

`agentsociety_runner.py` and `task_adapters/agentsociety_n8.py` are such glue for the
original AgentSociety CLI. `trace_replay_runner.py` supplies the frozen-request replay
path. Neither includes the AgentSociety simulator or its benchmark data. Other original
workloads can be connected through the same result contract, with their own quality
checks and measured forward-pass telemetry where needed.

## Workload and platform identity

Use the per-framework request counts in `configs/scenarios.yaml`, not one global request
count for all platforms. The paper's resource experiment uses the frozen 1040-request PD
trace, while the space experiment uses 271 requests. Supply the original task protocol,
request inputs and ordering for the corresponding comparison. The serving model is
separate from the search-decision model.

The ComfyBench runtime must include planning, ComfyUI execution and external quality
assessment. Set `quality_required: true`; the workload result must explicitly contain
`quality_passed: true`. For a protocol without an additional quality predicate, explicitly
set `quality_required: false`. Workload completion alone cannot satisfy a declared quality
criterion.

`configs/search_spaces/common_v2.json` and `fixed_7d.json` specify the 12D and fixed 7D
spaces. An adaptive 7D result is a particular evidence-dependent stage artifact, not a
synonym for the fixed seven-dimensional space. Preserve that stage's original space
artifact when reproducing an experimental row; do not invent a replacement from the
reported dimensionality alone.

## Resource conditions

`configs/envelopes.yaml` transcribes Table A2. It describes the imposed limits; it does not
by itself reserve GPU memory or configure containers. The runtime must establish and
record those limits before a comparison. CPU, RAM and I/O limits must cover the intended
engine/workload process scope consistently across methods.

Before selection, live resource probes refresh the available GPU count and per-device
free/total memory. Non-NVIDIA setups can provide `resource_probe_command` returning a
JSON object with `gpu_count`, `gpu_total_bytes`, and `gpu_free_bytes`. Optional CPU/RAM
fields may also be reported. Unknown values stay unavailable, rather than being inferred
from a nominal GPU model. Resource-based candidate rejection is recorded without
consuming evaluation budget. A started deployment that fails still counts.

The Table A2 scan pass rate concerns 33 specified memory-utilization points. It is not
used as the feasible volume of the full search space or as a candidate-success estimate.

## Search, stages and immutable output

A campaign is a fixed-space stage. The multi-stage driver partitions the total budget,
launches successive stages, and aggregates the target real trials. An internal episode
can produce diagnostic reviews, but policy/space/code changes are applied only for the
next stage. Code adoption requires isolated tests and a configured real performance
comparison. Without that evaluator, code proposals are recorded but not adopted.

Keep `campaign.sqlite3`, `frozen_search_space.json`, `experiment.json`, stage summaries,
experience records, and code revisions from an actual run. Each completed trial retains
its stage, space hash, code version and resource snapshot. A later space cannot silently
rewrite a completed trial or its validity. Source-scenario outcomes remain source
experience until a target evaluation verifies a configuration.

## Metrics and reporting

- Record monotonic launch/readiness/finish timestamps. Deployment and task intervals must
  sum to E2E; optimizer overhead and pre-run cleanup are outside those intervals.
- Select the best **valid real** E2E trial once. Its component and auxiliary values form
  that configuration-result row. Count failed real evaluations in the validity denominator.
- For TPOT, use the arithmetic mean of per-request `(last-first)/(output_tokens-1)`,
  excluding requests with at most one output token. Inter-token histograms use different
  weighting and are retained only as labelled diagnostics.
- Throughput is generated output tokens divided by the stated serving measurement window,
  which need not equal the complete application task window.
- Partition application wall time into disjoint planning-only, overlap, execution-only,
  judging and other intervals. Pipeline timelines with different start/end boundaries
  must not be treated as complete task-time partitions.
- Independent searches, fixed-configuration retests and candidate-window repetitions are
  different repetition units. Report their arithmetic means and E2E ranges separately.

`pytest -q` and `configs/default.yaml` validate contracts and orchestration. They are not a
reproduction of the numerical experimental tables. Reproduction requires the external
assets, original per-setting conditions and real measurements described above.
