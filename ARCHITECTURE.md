# Paper-to-code map

The manuscript is the specification. This map distinguishes executable mechanisms from
external serving, prediction, optimizer, and benchmark dependencies.

| Paper requirement | Implementation |
| --- | --- |
| Section 3, Eq. 1: deployment, task and end-to-end timing | [execution/timing.py](src/evotailor/execution/timing.py), [trial_wrapper.py](scripts/trial_wrapper.py) use one monotonic launch/ready/finish decomposition. |
| Section 3: deployment/completion/quality/measurement validity | [objective.py](src/evotailor/objective.py), [execution/backends.py](src/evotailor/execution/backends.py), and the wrapper enforce explicit gates and consistent component measurements. |
| Eq. 2 and Appendix E: best valid real trial, valid/attempted rate | [objective.py](src/evotailor/objective.py); a configuration-result row is selected by E2E once, with auxiliary metrics from that trial. |
| Section 4.2: candidate set, selection, one real execution | [loops/runner.py](src/evotailor/loops/runner.py), [loops/proposals.py](src/evotailor/loops/proposals.py), [selection/agent.py](src/evotailor/selection/agent.py). |
| One total budget across fixed-space stages | [run_evolution_cycles.py](scripts/run_evolution_cycles.py); a campaign is one stage, while an episode is an internal scheduling/review unit. |
| Section 4.3: add dimensions, narrow ranges or remove dimensions at a boundary | [spaces/builder.py](src/evotailor/spaces/builder.py), [spaces/designer.py](src/evotailor/spaces/designer.py), [experience/code_bank.py](src/evotailor/experience/code_bank.py). Removed dimensions become fixed deployment defaults. |
| Keep old stage records and parameter meanings intact | [spaces/search_space.py](src/evotailor/spaces/search_space.py), [storage.py](src/evotailor/storage.py), [contracts.py](src/evotailor/contracts.py); each trial records stage, space hash, code version, and resources. |
| Section 4.4: live resources, explicit infeasibility and free regeneration | [selection/resources.py](src/evotailor/selection/resources.py), [selection/feasibility.py](src/evotailor/selection/feasibility.py), [loops/runner.py](src/evotailor/loops/runner.py). Unknown resources are not guessed. |
| AIC prediction, task model, uncertainty and two-level feedback | [prediction/aiconfigurator.py](src/evotailor/selection/prediction/aiconfigurator.py), [prediction/configuration.py](src/evotailor/selection/prediction/configuration.py), [aiconfigurator_bridge.py](scripts/aiconfigurator_bridge.py). AIC is external and never serves as a proposer. |
| Section 4.5: experience retrieval and target verification | [loops/history.py](src/evotailor/loops/history.py), [experience/retrieval.py](src/evotailor/experience/retrieval.py), [experience/bank.py](src/evotailor/experience/bank.py). Source outcomes are conditional evidence, not target measurements. |
| Versioned search code with isolated validation | [experience/guard.py](src/evotailor/experience/guard.py), [experience/performance.py](src/evotailor/experience/performance.py), and [experience/code_bank.py](src/evotailor/experience/code_bank.py). No configured performance evaluator means no code adoption. |
| Section 5.1: SCOOT, SLO-Guard, LLM-Tuna | [baselines/registry.py](src/evotailor/baselines/registry.py), [baselines/worker.py](src/evotailor/baselines/worker.py); upstream algorithms execute in an isolated process. No simplified substitute is shipped under a baseline name. |
| Appendix A: frameworks, models, request counts and envelopes | [scenarios.yaml](configs/scenarios.yaml), [envelopes.yaml](configs/envelopes.yaml), [environment.md](docs/environment.md). Runtime enforcement requires the configured engine/container/resource setup. |
| Space comparison: Common 12D and Fixed 7D | [common_v2.json](configs/search_spaces/common_v2.json), [fixed_7d.json](configs/search_spaces/fixed_7d.json); adaptive selections are evidence-dependent stage artifacts. |
| Appendix E: per-request TPOT and serving-window throughput | [metrics/serving.py](src/evotailor/metrics/serving.py); inter-token histogram diagnostics remain separate from mean per-request TPOT. |
| Figure 3: disjoint task-time partition | [metrics/task.py](src/evotailor/metrics/task.py), [execution/timing.py](src/evotailor/execution/timing.py). |

Search-space equality is an optional experimental control in `spaces/invariant.py`, not
a universal restriction: the paper's adaptive variants may differ in dimensions and
ranges. Every stage independently freezes its actual space.

Engine execution, task completion, quality grading, and predictor accuracy must be
verified with the external assets and real environments. Unit and simulated checks do
not constitute a rerun of the paper's experimental tables.
