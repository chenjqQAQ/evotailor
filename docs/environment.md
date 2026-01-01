# Environment

Versions, hardware and external assets used with this code. Nothing here is bundled in the
repository: weights, traces and task definitions are supplied at run time through
`EVOTAILOR_MODEL_ROOT` and `EVOTAILOR_DATA_ROOT`.

## Software

| Component | Version | Notes |
| --- | --- | --- |
| Python | 3.11 | 3.12 also works for the analysis-only modules |
| CUDA | 12.8 | driver must match the engine build |
| vLLM | 0.19.1 | exposes `vllm:time_to_first_token_seconds`, `vllm:inter_token_latency_seconds`, `vllm:prefix_cache_queries_total`, `vllm:prefix_cache_hits_total` |
| SGLang | 0.3.6.post3 | separate adapter; server flag names differ from vLLM |
| MUSA-vLLM | Moore Threads build | same interface as vLLM, different device backend |
| PyTorch | bundled with the engine build | do not pin separately |

The engine is launched as a subprocess by `execution/backends.py`; `EVOTAILOR_VLLM_BIN` may point
at the `vllm` entry point of the engine environment or at a wrapper script.

## Hardware

| Setting | Devices | Notes |
| --- | --- | --- |
| AgentSociety on vLLM | 8 × A100 80GB | tensor parallelism 1–2 depending on the configuration under test |
| AgentSociety on SGLang | 2 × A100 80GB | |
| AgentSociety on MUSA-vLLM | 8 × MTT S5000 80GB | |
| ComfyBench | 8 × A100 80GB | 4 ComfyUI instances, 2 GPUs each |

Device visibility must be granted by the scheduler; the harness only sees the devices the
container/runtime exposes and never assumes a device count.

## Models and assets

| Asset | Used by | Supplied as |
| --- | --- | --- |
| Qwen3-4B (BF16) | AgentSociety workloads | `EVOTAILOR_SERVED_MODEL_PATH` |
| Qwen3.8-27B (BF16) | ComfyBench planning stage | `EVOTAILOR_SERVED_MODEL_PATH` |
| Stable Diffusion 1.5 | ComfyBench resident stage | ComfyUI model directory |
| AgentSociety traces (PD, PG, SM) | `workloads/agentsociety.py` | `EVOTAILOR_DATA_ROOT/traces/...` |
| ComfyBench `sample4` tasks | `workloads/comfybench.py` | `EVOTAILOR_DATA_ROOT/tasks/comfybench/sample4.json` |
| AIConfigurator profile | `selection/prediction/aiconfigurator.py` | `configs/aiconfigurator_profile.example.json` |

Trace sizes assumed by the defaults in `workloads/agentsociety.py`: PD 271, PG 255, SM 263
operations for the base settings and 1040 / 1016 / 1670 for the larger windows.
`Workload.assert_size` fails fast on a mismatch, which prevents silently measuring a smaller
workload.

## Search-decision agent

The agent that proposes candidates and reflects between stages is an external LLM served over an
OpenAI-compatible API; it is distinct from the model that serves application requests. Configure it
with `EVOTAILOR_V2_MODEL`, `EVOTAILOR_V2_BASE_URL` and `EVOTAILOR_V2_API_KEY`.

## Resource envelopes

`configs/envelopes.yaml` declares four envelopes. Under an envelope the harness reserves the
per-GPU occupancy before deployment and applies CPU, RAM and I/O limits to the process tree hosting
both the engine and the task. A trial that cannot allocate is *invalid*, not merely slow, and still
consumes budget.
