# Architecture

This document maps every part of the paper to the code that implements it.

## 1. Problem formulation (paper §3)

| Concept in the paper | Code |
| --- | --- |
| Scenario `z = (w, m, h, f)` — workload, model, hardware, framework | [`src/evotailor/contracts.py`](../src/evotailor/contracts.py) `Scenario`, `CampaignConfig` |
| Complete deployment configuration space `C_z` | [`spaces/search_space.py`](../src/evotailor/spaces/search_space.py) |
| Stage space `S_k ⊆ C_z`, fixed until an outer-loop boundary | [`spaces/base.py`](../src/evotailor/spaces/base.py) `SearchSpace`, `Stage` |
| Trial record `(c_t, r_t, s_t, T_deploy, T_task)` | [`contracts.py`](../src/evotailor/contracts.py) `TrialResult` |
| Validity `s_t ∈ {0,1}` (deploy ok, quality ok, measurements complete) | [`execution/trial.py`](../src/evotailor/execution/trial.py) |
| `T_deploy`, `T_task`, `T_e2e` | [`execution/timing.py`](../src/evotailor/execution/timing.py) |
| Budget `B`: every initiated deployment consumes one unit, rejected candidates consume none | [`loops/runner.py`](../src/evotailor/loops/runner.py), [`loops/inner.py`](../src/evotailor/loops/inner.py) |
| Objective `J_n` and valid-evaluation rate `η_n` | [`objective.py`](../src/evotailor/objective.py) |
| History `H_{t-1}` | [`loops/history.py`](../src/evotailor/loops/history.py) |

## 2. Nested loops (paper §4.2)

| Paper | Code |
| --- | --- |
| Inner loop: candidate set `Q_t = ⋃_{g∈G_k} g(S_k, H_{t-1})` | [`loops/proposals.py`](../src/evotailor/loops/proposals.py), [`loops/proposals_llm.py`](../src/evotailor/loops/proposals_llm.py) |
| Generator set `G_k` | same modules; generators are pluggable and may run concurrently or with different weights |
| Selection policy `π_k` | [`selection/agent.py`](../src/evotailor/selection/agent.py) |
| Outer loop: reflect and evolve at a stage boundary | [`loops/outer.py`](../src/evotailor/loops/outer.py), [`experience/review.py`](../src/evotailor/experience/review.py) |
| Code version `v_k` bound to a stage | [`experience/code_bank.py`](../src/evotailor/experience/code_bank.py) |
| Episode boundaries used to decide when the outer loop fires | [`loops/episode_boundary.py`](../src/evotailor/loops/episode_boundary.py), [`loops/episode_planner.py`](../src/evotailor/loops/episode_planner.py) |
| End-to-end driver that runs one campaign | [`loops/runner.py`](../src/evotailor/loops/runner.py) |

## 3. Adaptive search spaces (paper §4.3)

| Paper | Code |
| --- | --- |
| Dimension set, domain, conditional constraints | [`spaces/base.py`](../src/evotailor/spaces/base.py) `Dimension`, `SearchSpace` |
| Space construction from scenario + reusable experience | [`spaces/builder.py`](../src/evotailor/spaces/builder.py), [`spaces/designer.py`](../src/evotailor/spaces/designer.py) |
| Parameter meanings preserved across stages when dimensions are compatible | [`spaces/invariant.py`](../src/evotailor/spaces/invariant.py) |
| Engine-specific flag semantics (vLLM / SGLang / MUSA) | [`spaces/engine_flags.py`](../src/evotailor/spaces/engine_flags.py) |

## 4. Resource- and prediction-guided selection (paper §4.4)

| Paper | Code |
| --- | --- |
| Explicit infeasibility from resources (memory, devices, incompatible flags) | [`selection/feasibility.py`](../src/evotailor/selection/feasibility.py), [`selection/preflight.py`](../src/evotailor/selection/preflight.py) |
| Current resource snapshot | [`selection/resources.py`](../src/evotailor/selection/resources.py) |
| Inference-forward prediction `ŷ_AIC(c)` and calibration | [`selection/prediction/aiconfigurator.py`](../src/evotailor/selection/prediction/aiconfigurator.py), [`selection/prediction/calibration.py`](../src/evotailor/selection/prediction/calibration.py) |
| Task-level estimate `T̂_task(c)` with uncertainty | [`selection/prediction/configuration.py`](../src/evotailor/selection/prediction/configuration.py) |
| Balancing exploitation and exploration; rejection reasons logged | [`selection/agent.py`](../src/evotailor/selection/agent.py) |
| Engine capability probing | [`selection/engine_probe.py`](../src/evotailor/selection/engine_probe.py) |
| Priors, reputation, evidence bookkeeping | [`selection/priors.py`](../src/evotailor/selection/priors.py), [`selection/reputation.py`](../src/evotailor/selection/reputation.py), [`selection/evidence.py`](../src/evotailor/selection/evidence.py) |

## 5. Experience and code reuse (paper §4.5)

| Paper | Code |
| --- | --- |
| Experience bank: configurations, scenarios, resources, outcomes, prediction errors | [`experience/bank.py`](../src/evotailor/experience/bank.py) |
| Code bank: versioned generators, space design, selection, AIC adaptation | [`experience/code_bank.py`](../src/evotailor/experience/code_bank.py) |
| Retrieval by task/model/hardware/framework/space compatibility | [`experience/retrieval.py`](../src/evotailor/experience/retrieval.py) |
| Guardrails on evolved code before it is admitted | [`experience/guard.py`](../src/evotailor/experience/guard.py) |
| Reflected strategy adjustments stored alongside outcomes | [`experience/review.py`](../src/evotailor/experience/review.py), [`experience/performance.py`](../src/evotailor/experience/performance.py) |
| Source-context capture for later diagnosis | [`experience/source_context.py`](../src/evotailor/experience/source_context.py) |

## 6. Execution (paper §3, Appendix A)

| Paper | Code |
| --- | --- |
| Backend adapters (host / container / simulated) | [`execution/backends.py`](../src/evotailor/execution/backends.py) |
| One deployment + task run, with validity and failure taxonomy | [`execution/trial.py`](../src/evotailor/execution/trial.py) |
| Timing decomposition and overlap accounting | [`execution/timing.py`](../src/evotailor/execution/timing.py) |

## 7. Workloads and metrics (paper §5.1, Appendix C, Appendix E)

| Paper | Code |
| --- | --- |
| AgentSociety PD / PG / SM request traces | [`workloads/agentsociety.py`](../src/evotailor/workloads/agentsociety.py) |
| ComfyBench `sample4` workflow tasks | [`workloads/comfybench.py`](../src/evotailor/workloads/comfybench.py) |
| Trace discovery and workload inspection | [`workloads/discovery/`](../src/evotailor/workloads/discovery) |
| TTFT, TPOT, throughput from serving histograms | [`metrics/serving.py`](../src/evotailor/metrics/serving.py) |
| Task-time composition: planning / overlap / execution / judging / other | [`metrics/task.py`](../src/evotailor/metrics/task.py) |

## 8. Baselines (paper §5.1)

| Paper | Code |
| --- | --- |
| SCOOT | [`baselines/scoot.py`](../src/evotailor/baselines/scoot.py) |
| SLO-Guard | [`baselines/slo_guard.py`](../src/evotailor/baselines/slo_guard.py) |
| LLM-Tuna | [`baselines/llm_tuna.py`](../src/evotailor/baselines/llm_tuna.py) |
| Shared baseline interface and registry | [`baselines/registry.py`](../src/evotailor/baselines/registry.py) |

## 9. Resource envelopes (paper §5.5, Table A2)

Envelopes `T1`–`T4` are declared in
[`configs/envelopes.yaml`](../configs/envelopes.yaml) and consumed by
[`selection/resources.py`](../src/evotailor/selection/resources.py).

## Dependency direction

```
contracts ── objective
    │           │
    ├── spaces ─┼── loops ── selection ── experience
    │           │      │          │            │
    └───────────┴── execution ────┴── workloads ┴── metrics
                                     baselines
```

Lower layers never import upper layers: `execution` and `workloads` know nothing about
search, and `spaces`, `selection`, `experience` never start a process directly.
