# Reproduction guide

The artifact is source-level: it does not ship model weights, request traces, container
images, or measurement logs.  This page lists what has to be supplied and in what order.

## 1. Environment variables

Copy `.env.example` to `.env` and set at least:

| Variable | Meaning |
| --- | --- |
| `EVOTAILOR_DATA_ROOT` | directory holding request traces and task definitions |
| `EVOTAILOR_MODEL_ROOT` | directory holding served model weights |
| `EVOTAILOR_V2_MODEL` | model id used by the search-decision agent |
| `EVOTAILOR_V2_BASE_URL` | OpenAI-compatible endpoint for that model |
| `EVOTAILOR_SERVED_MODEL_PATH` | weights used by the serving engine |
| `PYTHON` | interpreter that provides the serving engine |

## 2. Assets that must be supplied

| Asset | Used by | Shape |
| --- | --- | --- |
| Request traces (PD / PG / SM) | `workloads/agentsociety.py` | `workload.json` as documented in the module |
| ComfyBench `sample4` tasks | `workloads/comfybench.py` | JSON list of instructions |
| AIConfigurator profile | `selection/prediction/aiconfigurator.py` | see `configs/aiconfigurator_profile.example.json` |
| Served model weights | `execution/backends.py` | any HF-format checkpoint |

Trace sizes assumed by the paper: PD 271, PG 255, SM 263 operations for the base settings,
and 1040 / 1016 / 1670 for the larger windows.  `Workload.assert_size` fails fast if a trace
does not match, which prevents silently measuring a smaller workload.

## 3. Order of operations

1. **Probe the engine** — `selection/engine_probe.py` records which flags the installed
   build accepts.  Dimensions that do not reach a real flag are dropped before the search
   starts, otherwise a whole stage can be spent on a no-op dimension.
2. **Validate the trace** — load it, check the operation count and the number of operations
   that carry tool schemas.  If that number is non-zero, the engine must be launched with
   tool-call support; otherwise those requests are rejected and the replay measures less
   work than intended.
3. **Run one trial** — deploy, run the task, record the three timestamps and one
   `/metrics` scrape.  Confirm that `T_e2e = T_deploy + T_task` and that the prompt volume
   matches the trace.
4. **Run a campaign** — `loops/runner.py` with the chosen method (`evotailor` or a
   baseline from `baselines/registry.py`), a budget, and optionally an envelope.
5. **Aggregate** — `objective.py` for `J_n` and `η_n`; `metrics/` for the auxiliary
   columns.

## 4. Invariants worth asserting

* `completed_replicas == replicas` for every trial; otherwise the trial is invalid.
* the recorded prompt/completion volume equals the trace's expected volume.
* `T_deploy`, `T_task`, `T_e2e` are monotonic in their timestamps.
* the prefix-cache setting is recorded for both arms of every comparison.
* a repeated run of a fixed configuration stays within the dispersion reported in the
  appendix of the paper; larger swings mean the environment, not the configuration,
  changed.

## 5. Resource envelopes

`configs/envelopes.yaml` declares the four envelopes.  When running under an envelope the
harness must (a) reserve the GPU occupancy before deployment and (b) apply the CPU, RAM and
I/O limits to the process tree that hosts both the engine and the task.  A trial that fails
to allocate is *invalid* rather than slow, and still consumes budget.
