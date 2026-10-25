# EvoTailor

Self-evolving agent-based configuration optimization for LLM inference serving.

EvoTailor minimizes the best **verified deployment-plus-task time** found under a fixed
real-evaluation budget. The served model runs the application; a separate search-decision
model proposes, selects, and reflects on deployment configurations.

## Objective and validity

For one real evaluation, deployment starts at `a`, the service becomes ready at `b`, and
the task finishes at `e`:

```
T_deploy = b - a
T_task  = e - b
T_e2e   = e - a = T_deploy + T_task
```

Task time is elapsed wall-clock time including model calls, tools, orchestration, and
waiting. Concurrent activities are not summed twice. Optimizer overhead is separate.
Every initiated evaluation, including a failed deployment, consumes one search unit;
a candidate rejected before execution consumes none. The winner and all its reported
component metrics come from the same valid real trial. A valid trial requires successful
deployment, workload completion, the declared quality criterion, and complete measurements.
Simulated observations are never reported as verified real configurations.

## Nested loops

One `CampaignRunner` invocation implements a **fixed-space stage**. Candidate generators
propose complete coordinates within that stage; the selector combines current resource
snapshots, trial history, AIC predictions, and configuration-level task estimates. Only the
selected candidate is executed. Rejections, failures, measurements, and prediction errors
are recorded in SQLite.

The stage driver connects stages under **one total budget**, not a fresh budget per stage.
At a stage boundary, reflection may revise the next stage's dimensions, ranges, generators,
policy, and code. An active stage's space and completed trials remain immutable. Removed
searched dimensions retain predetermined deployment defaults. Code changes require isolated
functional and configured performance validation before adoption.

Experience retrieval provides conditional source evidence. Source configurations must be
measured on the target before they can become target results. A changed space does not
silently turn old observations into equivalent training samples.

## Repository layout

```
src/evotailor/
  contracts.py            campaign, candidate, trial and measurement contracts
  objective.py            validity, best real trial, J_n and valid-evaluation rate
  spaces/                 scenario adaptation, constraints and frozen space identity
  loops/                  fixed-stage runner, generators, policies and reflection
  selection/              live feasibility, evidence and prediction-guided selection
  experience/             retrieval, records, code validation and next-stage evolution
  execution/              service/workload execution and timing decomposition
  workloads/              AgentSociety and ComfyBench input interfaces
  metrics/                request metrics and disjoint application-time partitions
  baselines/              isolated bridges to the upstream comparison implementations
configs/                  campaign templates, 12D/7D spaces and paper scenario settings
scripts/                  deployment wrapper, replay, AIC bridge and multi-stage driver
tests/                    objective, validity, stage adaptation, metrics and timing checks
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the paper-to-code mapping and
[docs/design.md](docs/design.md) for an execution walkthrough.

## Installation

Python 3.11 or later:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,optimizers]"
cp .env.example .env
```

Inference engines, AIConfigurator, upstream baseline implementations, model weights, and
benchmark assets are installed separately. Hardware and software settings from the paper
are listed in [docs/environment.md](docs/environment.md). Supply local paths and endpoints
through environment variables; no credentials are bundled.

## Usage

### Check the installation

```bash
evotailor --help
pytest -q
```

### Test orchestration without an engine or model endpoint

```bash
evotailor run --config configs/default.yaml --no-dashboard
```

This uses the explicitly marked simulated executor and local generators. It checks
orchestration only; its scores are not experimental measurements.

### Inspect a target workload

```bash
evotailor discover --task-root "$EVOTAILOR_TASK_ROOT" --output discovery.json
```

Inspect the discovered evidence and configure a real campaign and its runtime wrapper.
The runtime must explicitly declare `quality_required`. For ComfyBench with an external
quality criterion, set it to `true` and require the workload adapter to return
`quality_passed: true`; missing quality evidence fails validity.

### Run a real fixed-space stage

```bash
evotailor run --config configs/production_host.example.yaml --no-dashboard
```

Fill the environment-specific runtime and AIC settings first. Preflight rejects unresolved
paths, missing engines, or unavailable predictor settings. No simulated fallback is used
by a real execution backend. These commands start real evaluations when configured.

### Run the nested search across stages

```bash
python scripts/run_evolution_cycles.py \
  --config configs/production_host.example.yaml \
  --stages 3 --output runs/nested-search
```

Set `auto_apply_evolution: true` in the chosen campaign when using multiple stages. Its
`max_trials` is the total budget `B`; the driver divides that budget between stages.
Optionally give `--stage-budgets 4 4 4` when `B=12`. In-stage policy application is disabled.
`search_summary.json` selects the winner across valid target-stage trials. Additional
fixed-configuration retests are separate evaluations, never silently part of this search.

### Run a comparison method

Choose `baseline_scoot.yaml`, `baseline_slo_guard.yaml`, or `baseline_llm_tuna.yaml`, and
configure the same real executor, workload, completion criteria, resources, and budget as
the corresponding comparison. These files select algorithms; adjust their execution
backend and runtime commands for the experiment.

Baseline proposals call the configured upstream adapters in an isolated subprocess. Set
`EVOTAILOR_MATURE_ROOT` to the optimizer installation and optionally
`EVOTAILOR_BASELINE_PYTHON` to its interpreter. The required adapter classes are documented
in [docs/reproduction.md](docs/reproduction.md). Missing upstream code is an error, not a
reason to substitute a random or approximate algorithm with the same method name.

### Replay logs and compute metrics

```bash
evotailor replay --run-dir runs/your-stage
```

Use `metrics.serving.from_requests` for per-request TTFT/TPOT and specify the serving
measurement window for throughput. Prometheus inter-token histograms are token-weighted
diagnostics and are not labelled as the paper's mean per-request TPOT. Use before/after
counter snapshots for an already-running service.

`metrics.task` partitions planning-only, planning/execution overlap, execution-only,
judging, and other intervals without double-counting concurrent work.

## External assets

Model weights, benchmark request traces, ComfyBench task definitions, container images,
and measurement logs are not included. The 12D common and fixed 7D parameter definitions
are configuration files. The adaptive space is constructed from target evidence; do not
replace an unavailable adaptive-stage record with a guessed seven-parameter definition.

See [docs/reproduction.md](docs/reproduction.md) for setup, output contracts, and validation
boundaries. The code does not embed or manufacture the numerical results in paper tables.

## License

Copyright (c) 2026 The EvoTailor Authors.

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for the full terms.
External dependencies and upstream baseline sources retain their own licenses.
