# EvoTailor

**Self-evolving agent-based configuration optimization for LLM inference serving.**

EvoTailor tunes the deployment parameters of an inference engine (vLLM, SGLang, MUSA-vLLM) to
minimize the **end-to-end completion time** of an agentic Web task:

```
T_e2e = T_deploy + T_task
```

`T_deploy` is the time from launching the service to readiness; `T_task` is the time from readiness
to task completion, measured as wall clock so that overlapping sub-activities are never double
counted. The optimizer searches under a fixed budget of **real** evaluations — every deployment that
is started costs one unit, including failures.

This repository contains the complete implementation: both search loops, the three mechanisms, the
execution and measurement layer, the workload adapters, and the baseline methods used for
comparison. Model weights, request traces and measurement logs are not included; see
[Environment](#environment) for what has to be supplied.

---

## Contents

- [How it works](#how-it-works)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Environment](#environment)
- [Design notes](#design-notes)
- [License](#license)

---

## How it works

The search is organized as **two nested loops**.

| Loop | Scope | Entry point |
| --- | --- | --- |
| Inner | within a stage: propose → screen → select → deploy → measure → record | `src/evotailor/loops/inner.py` |
| Outer | at stage boundaries: reflect on the evidence, evolve the space, the policy and the code | `src/evotailor/loops/outer.py` |

Three mechanisms connect them.

**1. Adaptive search spaces** — `src/evotailor/spaces/`
Dimensions, domains and conditional constraints are constructed from the scenario: workload
concurrency and context length, model size, available memory, and the flag surface of the target
framework. They are revised only at stage boundaries. Inside a stage the space is frozen, so
historical samples keep their meaning and predictions stay comparable.

**2. Resource- and prediction-guided selection** — `src/evotailor/selection/`
Explicit resource evidence removes candidates that cannot be served right now, before they consume
budget. The remainder is ranked with an inference-forward prediction plus a task-level estimate that
carries an uncertainty band fitted from the valid trials of the current stage. The selector balances
exploitation and exploration rather than ranking by predicted time alone.

**3. Cross-scenario experience and code reuse** — `src/evotailor/experience/`
An experience bank records configurations, scenarios, resources, outcomes and prediction errors; a
code bank holds versioned generators, space construction and selection logic. Both are retrieved by
scenario similarity and verified in the target environment — a source scenario's best configuration
is never reported as a target result. An evolved policy is admitted only after the guard accepts it
(`experience/guard.py`), which keeps it from silently changing budget accounting or the objective.

## Repository layout

```
src/evotailor/
  contracts.py            scenario, candidate, trial record, validity, budget
  objective.py            J_n (best verified time) and eta_n (valid-evaluation rate)
  spaces/                 dimensions, domains, constraints, space evolution, engine flags
  loops/                  inner loop, outer loop, proposal generators, history
  selection/              feasibility, resources, prediction (AIC + task model), selector agent
  experience/             experience bank, code bank, retrieval, guard, review
  execution/              backend adapters (host / container / simulated) and timing decomposition
  workloads/              AgentSociety (PD, PG, SM) and ComfyBench adapters
  metrics/                serving metrics (TTFT, TPOT, throughput) and task-time composition
  baselines/              SCOOT, SLO-Guard and LLM-Tuna under a shared interface
configs/                  scenario descriptors, resource envelopes, baseline settings
scripts/                  command-line entry points
docs/                     architecture, design walkthrough, environment, reproduction notes
tests/                    unit tests for timing, spaces and metrics
```

`ARCHITECTURE.md` maps every mechanism to its module; `docs/design.md` walks through one search end
to end.

## Installation

Python 3.11+ and an inference engine you intend to tune.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional extras:

```bash
pip install -e ".[metrics]"   # prometheus-client, for scraping engine metrics
pip install -e ".[dev]"       # pytest, ruff
```

## Configuration

All machine-specific values are injected through environment variables; no absolute paths,
credentials or host names are stored in the repository.

```bash
cp .env.example .env
$EDITOR .env
```

| Variable | Meaning |
| --- | --- |
| `EVOTAILOR_DATA_ROOT` | directory holding request traces and task definitions |
| `EVOTAILOR_MODEL_ROOT` | directory holding served model weights |
| `EVOTAILOR_V2_MODEL` | model id used by the search-decision agent |
| `EVOTAILOR_V2_BASE_URL` | OpenAI-compatible endpoint for that model |
| `EVOTAILOR_V2_API_KEY` | key for that endpoint |
| `EVOTAILOR_SERVED_MODEL_PATH` | weights used by the serving engine |
| `EVOTAILOR_VLLM_BIN` | engine entry point (`vllm`, or a wrapper script) |
| `PYTHON` | interpreter that provides the engine |

Scenario descriptors live in `configs/scenarios.yaml`, the resource envelopes in
`configs/envelopes.yaml`, and baseline settings in `configs/baseline_*.yaml`.

## Usage

### 1. Verify the install

```bash
python -c "import evotailor; print(evotailor.__version__)"
```

### 2. Probe the engine

Records which flags the installed build actually accepts, so no stage is spent on a dimension that
does not reach the engine.

```python
from evotailor.selection.engine_probe import probe

caps = probe(engine="vllm")
print(sorted(caps.flags))
```

### 3. Load and check a trace

```python
from evotailor.workloads import agentsociety

workload = agentsociety.load(f"{DATA_ROOT}/traces/pd/workload.json", scenario="PD")
print(workload.summary())   # operations, agents, peak concurrency, tool-bearing count
```

A non-zero tool-bearing count means the engine must be launched with tool-call support; otherwise
those requests are rejected and the replay silently measures less work than intended.

### 4. Run a search

```bash
python scripts/run_evolution_cycles.py --config configs/default.yaml
```

A campaign writes one record per trial (configuration, resources, validity, `T_deploy`, `T_task`,
failure diagnostics) plus a summary with `J_n` and `eta_n`.

### 5. Run a baseline under the same budget

```bash
python scripts/run_evolution_cycles.py --config configs/baseline_scoot.yaml
```

Baselines share the space, the executor, the budget accounting and the output contract with the main
method, so a difference in the result is attributable to proposal and selection.

### 6. Replay a frozen trace

```bash
python scripts/trace_replay_runner.py --workload "$TRACE" --api-base "$API" \
    --served-model "$MODEL" --output-mode fixed --result-path out.json
```

`--output-mode fixed` pins the generation length to the recorded value, so two configurations
process exactly the same tokens and compare on latency rather than on sampling luck.

### 7. Read the metrics

```python
from evotailor.metrics.serving import from_scrape, prefix_cache_stats

metrics = from_scrape(scrape_text, completion_tokens=35459,
                      request_count=271, task_window_s=79.7)
print(metrics.as_dict())   # TTFT, TTFT_p50, TTFT_p95, TPOT, throughput
print(prefix_cache_stats(scrape_text).hit_rate)
```

Prefix-cache counters are recorded with every trial. A configuration with prefix caching enabled and
one with it disabled are not comparable, so the setting belongs to the trial record rather than to the
implementation.

### 8. Analyze a campaign

`src/evotailor/objective.py` computes `J_n` (best verified end-to-end time) and `eta_n`
(valid-evaluation rate) from the trial list rather than from a running minimum, so a crash cannot
inflate either. `src/evotailor/metrics/task.py` partitions the task window into planning-only,
planning/execution overlap, execution-only, judging and other.

## Environment

Tested combinations; the versions below are the ones used for the measurements reported with this
code. See `docs/environment.md` for the full matrix and `requirements.txt` for pinned dependencies.

| Component | Version / setting |
| --- | --- |
| Python | 3.11 |
| CUDA | 12.8 |
| vLLM | 0.19.1 |
| SGLang | 0.3.6.post3 |
| MUSA-vLLM | Moore Threads build, MTT S5000 |
| Served model (AgentSociety) | Qwen3-4B, BF16 |
| Served model (ComfyBench) | Qwen3.8-27B, BF16 |
| Resident stage (ComfyBench) | Stable Diffusion 1.5 |
| Hardware | 8 × A100 80GB; 8 × MTT S5000 80GB (MUSA); 2 × A100 80GB (SGLang) |
| Search-decision agent | an external LLM endpoint, distinct from the served model |

**Not included in this repository:** model weights, request traces, ComfyBench task definitions,
container images and measurement logs. Supply them through `EVOTAILOR_DATA_ROOT` and
`EVOTAILOR_MODEL_ROOT`.

## Design notes

- `ARCHITECTURE.md` — mechanism-by-mechanism map from the design to the code.
- `docs/design.md` — one complete search, step by step.
- `docs/environment.md` — versions, hardware matrix, asset inventory.
- `docs/reproduction.md` — order of operations to reproduce a campaign, and the invariants worth
  asserting.

Invariants the harness enforces or checks:

- a completed trial records `completed_replicas == replicas`;
- the recorded prompt/completion volume equals the trace's expected volume;
- `T_deploy`, `T_task` and `T_e2e` come from one monotonic three-instant decomposition;
- a candidate rejected by the feasibility gate consumes no budget;
- the prefix-cache setting is part of every trial record.

## Tests

```bash
pytest -q
```

## License

Copyright (c) 2026 The EvoTailor Authors.

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for the full terms.
