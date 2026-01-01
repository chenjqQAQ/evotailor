from __future__ import annotations

import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from evotailor.contracts import CampaignConfig, ExecutionBackend, RunMode
from evotailor.workloads.discovery.inspectors import _allocated_gpu_uuids
from evotailor.spaces.search_space import SearchSpace

#: Substrings that mean "this value was never filled in". A real run must never
#: start from a template, so any of these in a real path is a hard failure.
PLACEHOLDER_TOKENS = (
    "replace_with",
    "replace with",
    "placeholder",
    "<path",
    "todo",
    "template",
    "example",
)

#: The only keys an AIC profile may not omit.
REQUIRED_PROFILE_KEYS = ("engine_config",)


def _contains_placeholder(text: str) -> str | None:
    lowered = text.lower()
    for token in PLACEHOLDER_TOKENS:
        if token in lowered:
            return token
    return None


def _command_issue(command: list[str], root: Path, label: str) -> list[str]:
    if not command:
        return [f"{label} is empty"]
    issues = []
    executable = Path(command[0])
    if executable.is_absolute() and not executable.is_file():
        issues.append(f"{label} executable does not exist: {executable}")
    for argument in command[1:]:
        if argument.endswith(".py"):
            script = Path(argument)
            if not script.is_absolute():
                script = root / script
            if not script.is_file():
                issues.append(f"{label} script does not exist: {script}")
    placeholder = _contains_placeholder(" ".join(str(item) for item in command))
    if placeholder is not None:
        issues.append(f"{label} still contains a placeholder value ({placeholder!r})")
    return issues


def _model_weight_issues(model_path: Path) -> list[str]:
    """A directory of tokenizer files is not a model."""
    issues: list[str] = []
    weights = (
        list(model_path.glob("*.safetensors"))
        + list(model_path.glob("*.bin"))
        + list(model_path.glob("*.pt"))
    )
    if not weights:
        return [f"model path has no model weight files: {model_path}"]
    if any(re.search(r"-\d+-of-\d+\.safetensors$", path.name) for path in weights):
        index = model_path / "model.safetensors.index.json"
        if not index.is_file():
            issues.append(
                f"sharded safetensors require model.safetensors.index.json in {model_path}"
            )
    return issues


def _aic_profile_issues(profile_path: Path) -> list[str]:
    issues: list[str] = []
    if ".example." in profile_path.name:
        issues.append(f"AIC profile must not be an example template: {profile_path.name}")
    try:
        payload = json.loads(profile_path.read_text())
    except Exception as exc:
        return issues + [f"AIC profile is not valid JSON: {type(exc).__name__}: {exc}"]
    if not isinstance(payload, dict):
        return issues + ["AIC profile must be a JSON object"]

    text = json.dumps(payload, default=str)
    placeholder = _contains_placeholder(text)
    if placeholder is not None:
        issues.append(
            f"AIC profile still contains a placeholder value ({placeholder!r}); "
            "a real profile must be measured from the served model"
        )
    for key in REQUIRED_PROFILE_KEYS:
        if not payload.get(key):
            issues.append(f"AIC profile is missing required key: {key}")
    mode = payload.get("estimator_mode", "native")
    if mode not in ("native", "regression"):
        issues.append("AIC estimator_mode must be native or regression")
    iterations = payload.get("workload_iterations")
    if iterations is not None and not isinstance(iterations, list):
        issues.append("AIC workload_iterations must be a list of real observations")
    if isinstance(iterations, list) and iterations:
        flat: list[dict[str, Any]] = []
        for item in iterations:
            flat.extend(item if isinstance(item, list) else [item])
        positive = [
            record
            for record in flat
            if isinstance(record, dict) and float(record.get("wall_time") or 0) > 0
        ]
        if not positive:
            issues.append(
                "AIC profile workload_iterations contain no positive wall_time; "
                "tuning treats wall_time as the latency target"
            )
    engine = payload.get("engine_config")
    if isinstance(engine, dict):
        for key in ("system_name",):
            if not engine.get(key):
                issues.append(f"AIC profile engine_config missing required key: {key}")
        if not engine.get("model_name"):
            issues.append("AIC profile engine_config.model_name is empty")
        if not engine.get("backend_version"):
            issues.append("AIC profile engine_config.backend_version is empty")
    return issues


def _docker_runtime_checks(
    runtime_payload: dict[str, Any], root: Path, issues: list[str], warnings: list[str]
) -> None:
    """Check the runtime that will really run, not a lookalike command."""
    service = [str(item) for item in runtime_payload.get("service_command") or []]
    service_text = " ".join(service)
    image = runtime_payload.get("image") or (runtime_payload.get("docker") or {}).get("image")
    if not image:
        issues.append("docker runtime requires image")
        return
    if not service:
        issues.append("docker runtime requires service_command")
        return

    check = subprocess.run(
        ["docker", "image", "inspect", str(image)], capture_output=True, text=True, check=False
    )
    if check.returncode != 0:
        issues.append(f"docker image unavailable: {image}: {check.stderr.strip()[-500:]}")
    if str(image) not in service:
        issues.append(
            f"service_command does not use the declared image {image}; "
            "the preflight probe would not match the real run"
        )

    if "--gpus" in service:
        issues.append(
            "service_command must not use --gpus; request only this account's GPUs "
            "through NVIDIA_VISIBLE_DEVICES"
        )

    allocation = _allocated_gpu_uuids()
    configured = [str(item) for item in runtime_payload.get("gpu_uuids") or []]
    if allocation:
        if not configured:
            issues.append(
                "runtime config must declare gpu_uuids so the run cannot claim "
                "GPUs this account does not own"
            )
        elif not set(configured).issubset(set(allocation)):
            issues.append(
                "runtime gpu_uuids name devices this account does not own: "
                f"configured={sorted(configured)} allocated={sorted(allocation)}"
            )
    elif not configured:
        issues.append("runtime config does not select any GPU")
    if set(configured) < set(allocation):
        warnings.append(
            "runtime declares a subset of this account's GPUs on purpose: using "
            f"{sorted(configured)}, leaving {sorted(set(allocation) - set(configured))} "
            "unused. The protocol's single-GPU arms ran this way."
        )

    model = runtime_payload.get("model_path")
    container_model = runtime_payload.get("model_container_path")
    if not model:
        issues.append("runtime config requires model_path")
    else:
        model_path = Path(str(model))
        if not model_path.exists():
            issues.append(f"model path does not exist: {model_path}")
        else:
            issues.extend(_model_weight_issues(model_path))
        if container_model and f"{model}:{container_model}" not in service_text:
            issues.append(
                "service_command does not mount the model into the container at "
                f"{container_model}; the container cannot read {model}"
            )

    environment_path = runtime_payload.get("vllm_environment_path")
    if environment_path and f"{environment_path}:{environment_path}" not in service_text:
        issues.append(
            "service_command does not mount the vLLM environment at its own path "
            f"{environment_path}; the image entrypoint hard-codes that host path"
        )

    if not runtime_payload.get("health_url"):
        issues.append("runtime config requires health_url so readiness can be awaited")

    if runtime_payload.get("telemetry", {}).get("require_forward_pass_metrics") and (
        "{fpm_output_path}" not in service_text or "{instrumentation_path}" not in service_text
    ):
        issues.append(
            "runtime requires FPM telemetry but service_command does not forward "
            "{fpm_output_path} and {instrumentation_path} into the container"
        )

    # Probe with exactly the declared selection so the probe cannot disagree
    # with the real run.
    probe_command = [
        "docker",
        "run",
        "--rm",
        "--runtime=nvidia",
        "-e",
        f"NVIDIA_VISIBLE_DEVICES={','.join(configured)}",
        str(image),
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total",
        "--format=csv,noheader",
    ]
    try:
        probe = subprocess.run(
            probe_command, capture_output=True, text=True, timeout=300, check=False
        )
    except Exception as exc:
        issues.append(f"docker GPU probe failed to run: {type(exc).__name__}: {exc}")
        return
    if probe.returncode != 0 or not probe.stdout.strip():
        issues.append(f"docker GPU probe failed: {probe.stderr.strip()[-500:]}")
        return
    probed = [line for line in probe.stdout.strip().splitlines() if line.strip()]
    warnings.append(f"docker GPU probe succeeded: {len(probed)} GPU(s)")
    if configured and len(probed) != len(configured):
        issues.append(f"GPU probe saw {len(probed)} GPU(s) but {len(configured)} were requested")


def _runtime_checks(
    config: CampaignConfig, root: Path, issues: list[str], warnings: list[str]
) -> None:
    runtime = os.environ.get("EVOTAILOR_RUNTIME_CONFIG")
    if not runtime or not Path(runtime).expanduser().is_file():
        issues.append("EVOTAILOR_RUNTIME_CONFIG must point to a real runtime YAML")
        return
    runtime_path = Path(runtime).expanduser().resolve()
    placeholder = _contains_placeholder(runtime_path.name)
    if placeholder is not None:
        issues.append(
            f"EVOTAILOR_RUNTIME_CONFIG must not point at a template file "
            f"({runtime_path.name} contains {placeholder!r})"
        )
    runtime_payload = yaml.safe_load(runtime_path.read_text()) or {}
    runtime_objective = runtime_payload.get("objective_metric", "deployment_plus_task_time_s")
    if runtime_objective != config.objective_metric:
        issues.append(
            f"Runtime objective {runtime_objective!r} does not match Campaign objective {config.objective_metric!r}"
        )
    profile_value = os.environ.get("EVOTAILOR_AIC_PROFILE")
    if (
        config.run_mode == RunMode.PORTFOLIO_AGENT
        and profile_value
        and Path(profile_value).is_file()
    ):
        profile = json.loads(Path(profile_value).read_text())
        if (
            profile.get("objective_metric")
            and profile["objective_metric"] != config.objective_metric
        ):
            issues.append("AIC objective_metric does not match the Campaign")
        declared = profile.get("runtime_identity") or {}
        expected_sha = declared.get("runtime_config_sha256")
        if expected_sha:
            import hashlib

            if expected_sha != hashlib.sha256(runtime_path.read_bytes()).hexdigest():
                issues.append(
                    "AIC runtime identity does not match the current runtime configuration"
                )
        expected_model = declared.get("served_model_path")
        if (
            expected_model
            and Path(expected_model).resolve()
            != Path(runtime_payload.get("model_path", "")).resolve()
        ):
            issues.append("AIC model identity does not match the served model path")
    if runtime_payload.get("mode") != config.execution_backend.value:
        issues.append(
            "runtime YAML mode does not match execution_backend: "
            f"{runtime_payload.get('mode')!r} != {config.execution_backend.value!r}"
        )
    if config.execution_backend == ExecutionBackend.DOCKER:
        _docker_runtime_checks(runtime_payload, root, issues, warnings)

    telemetry = runtime_payload.get("telemetry") or {}
    if not telemetry.get("collect_vllm_fpm", False):
        warnings.append(
            "real runtime does not enable telemetry.collect_vllm_fpm; "
            "AIConfigurator native calibration will receive no automatic FPM data"
        )

    issues.extend(
        _command_issue(runtime_payload.get("workload_command") or [], root, "workload_command")
    )

    run_root = runtime_payload.get("run_root")
    if not run_root:
        issues.append("runtime config requires run_root for Trial artifacts")
    else:
        # run_root is normally scoped per Campaign, so the literal template never
        # exists as a directory. Render it the way the wrapper does before judging
        # whether the artifacts can actually be written.
        rendered = (
            str(run_root)
            .replace("{campaign_id}", config.campaign_id)
            .replace("{trial_id}", "trial-0001")
        )
        leftover_match = re.search(r"\{[a-z_]+\}", rendered)
        leftover = leftover_match.group(0) if leftover_match else None
        run_root_path = Path(rendered).expanduser()
        if leftover is not None:
            issues.append(f"run_root contains an unresolved placeholder {leftover!r}")
        else:
            # The leaf is created on demand, so what must exist and be writable is
            # its nearest existing ancestor.
            anchor = run_root_path
            while not anchor.exists() and anchor != anchor.parent:
                anchor = anchor.parent
            if not anchor.is_dir():
                issues.append(f"run_root has no existing directory: {run_root_path}")
            elif not os.access(anchor, os.W_OK):
                issues.append(f"run_root is not writable: {anchor}")
            elif anchor != run_root_path:
                warnings.append(
                    f"run_root {run_root_path} does not exist yet and will be created "
                    f"under {anchor}"
                )


def _engine_argument_issues(config: CampaignConfig, root: Path) -> list[str]:
    """Every search dimension must have a way to reach the engine.

    A dimension with no ``parameter_arguments`` mapping makes the Trial wrapper
    refuse the candidate, and the Trial then fails before any container starts.
    That is how 36 Trials once produced zero measurements, so it is checked here
    as well as derived at design time.
    """
    runtime = os.environ.get("EVOTAILOR_RUNTIME_CONFIG")
    declared: dict[str, Any] = {}
    if runtime and Path(runtime).expanduser().is_file():
        try:
            payload = yaml.safe_load(Path(runtime).expanduser().read_text()) or {}
        except (OSError, ValueError):
            payload = {}
        arguments = payload.get("parameter_arguments")
        if isinstance(arguments, dict):
            declared = arguments

    run_dir = Path(config.run_root)
    if not run_dir.is_absolute():
        run_dir = root / run_dir
    run_dir = run_dir / config.campaign_id
    space_path = run_dir / "prepared_search_space.json"
    if not space_path.is_file():
        space_path = Path(config.search_space)
        if not space_path.is_absolute():
            space_path = root / space_path

    derived: dict[str, Any] = {}
    derived_path = run_dir / "derived_parameter_arguments.json"
    if derived_path.is_file():
        try:
            payload = json.loads(derived_path.read_text())
        except ValueError:
            payload = {}
        if isinstance(payload, dict):
            derived = payload

    try:
        space = SearchSpace.load(space_path)
    except Exception:
        return []
    known = set(declared) | set(derived)
    return [
        f"search dimension {name} cannot reach the engine: the runtime config "
        "declares no parameter_arguments mapping for it and none could be derived"
        for name in sorted(space.dimensions)
        if name not in known
    ]


def run_preflight(config: CampaignConfig, root: Path) -> dict[str, Any]:
    issues: list[str] = []
    warnings: list[str] = []
    space_path = Path(config.search_space)
    if not space_path.is_absolute():
        space_path = root / space_path
    try:
        space = SearchSpace.load(space_path)
        issues.extend(
            f"safe_baseline: {item}" for item in space.validate_candidate(space.safe_baseline)
        )
    except Exception as exc:
        issues.append(f"search space: {type(exc).__name__}: {exc}")

    if config.execution_backend != ExecutionBackend.SIMULATED:
        issues.extend(_engine_argument_issues(config, root))

    try:
        import optuna

        versions = {"optuna": optuna.__version__}
    except Exception as exc:
        versions = {}
        issues.append(f"Optuna unavailable: {type(exc).__name__}: {exc}")

    legacy_root = Path(config.mature_optimizer_root)
    if not legacy_root.is_dir():
        issues.append(f"mature optimizer root does not exist: {legacy_root}")
    else:
        if str(legacy_root) not in sys.path:
            sys.path.insert(0, str(legacy_root))
        for name, module_name in {
            "slo_guard": "evotailor.optimizer_tools.adapters.slo_guard",
            "scoot": "evotailor.optimizer_tools.adapters.scoot",
        }.items():
            if name not in config.enabled_proposers and config.baseline_proposer != name:
                continue
            try:
                module = importlib.import_module(module_name)
                manifest = module.manifest_snapshot()
                if not manifest.available:
                    issues.append(f"{name} unavailable: {manifest.unavailable_reason}")
            except Exception as exc:
                issues.append(f"{name} import failed: {type(exc).__name__}: {exc}")

    if config.execution_backend != ExecutionBackend.SIMULATED:
        issues.extend(_command_issue(config.command, root, "command"))
        _runtime_checks(config, root, issues, warnings)
        if config.run_mode == RunMode.PORTFOLIO_AGENT:
            issues.extend(
                _command_issue(config.aiconfigurator_command, root, "aiconfigurator_command")
            )
            profile = os.environ.get("EVOTAILOR_AIC_PROFILE")
            if not profile or not Path(profile).expanduser().is_file():
                issues.append("EVOTAILOR_AIC_PROFILE must point to a real AIC profile JSON")
            else:
                issues.extend(_aic_profile_issues(Path(profile).expanduser().resolve()))
    else:
        warnings.append("simulated backend validates orchestration, not a real vLLM deployment")

    return {
        "ok": not issues,
        "backend": config.execution_backend.value,
        "issues": issues,
        "warnings": warnings,
        "versions": versions,
    }
