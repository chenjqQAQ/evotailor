# Architecture

1. `discover` records task entrypoints, hardware, installed vLLM version, and supported flags.
2. `prepare` lets a bounded English-prompt agent add up to two dimensions without redefining the base space.
3. `CampaignRunner` freezes the final space and derives deterministic per-Trial/per-proposer seeds.
4. Optuna TPE, SLO-Guard, SCOOT, coverage, and incumbent-neighborhood each emit one complete canonical candidate. One proposer failure does not discard the Trial when other candidates remain.
5. AIConfigurator predicts every candidate. Simulated runs use a marked deterministic proxy. Real runs require the production bridge and fail closed.
6. The selector chooses an existing candidate ID. Invalid model output retries in the model client and then uses a deterministic calibrated confidence-bound fallback.
7. `trial_wrapper.py` executes a host or Docker service/workload contract and returns objective, metrics, and telemetry.
8. Real observations update task-level residual calibration and are sent to the AIC bridge; FPM telemetry feeds AIConfigurator's native `tune_with_fpms()` path.
9. The validation phase includes a repeated incumbent candidate. At Episode boundaries the outer agent writes an evidence-linked review and may patch only versioned phase boundaries and uncertainty bonus.
10. SQLite is the durable fact log. Derived reputation and residual state are rebuilt after restart. A frozen-space hash mismatch stops resume.

The failure-avoidance model, advanced proposer credit assignment, and per-role LLM routing are intentionally deferred. Their interfaces remain separable from the experiment kernel.
