from __future__ import annotations

import ast
import fcntl
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from evotailor.workloads.discovery.workload import GRADE_DECLARED, GRADE_MEASURED, WorkloadProfiler, fact

# The platform publishes this host's GPU allocation as container UUIDs. Bare
# ``nvidia-smi`` sees nothing inside the sandbox because the platform sets
# ``NVIDIA_VISIBLE_DEVICES=none``, so this file is the authoritative source for
# "how many GPUs may this account actually use".
GPU_ALLOCATION_FILE = Path("/certs/docker/nvidia_visible_devices.txt")

# Leftover values that make a foreign interpreter import this project's private
# dependency directory instead of its own packages. ``.deps/numpy`` shadowing a
# conda environment's numpy breaks ``import vllm`` outright, so every call into
# a foreign interpreter must run with these removed.
PYTHON_ENV_OVERRIDES = ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")

GPU_QUERY = [
    "nvidia-smi",
    "--query-gpu=index,uuid,name,memory.total,driver_version",
    "--format=csv,noheader",
]


def _clean_python_env() -> dict[str, str]:
    """Environment for a foreign interpreter: no inherited Python dirt."""
    env = dict(os.environ)
    for name in PYTHON_ENV_OVERRIDES:
        env.pop(name, None)
    return env


def _run(
    command: list[str],
    timeout: int = 20,
    env: dict[str, str] | None = None,
    capture_limit: int = 20000,
) -> dict[str, Any]:
    try:
        done = subprocess.run(
            command, text=True, capture_output=True, timeout=timeout, check=False, env=env
        )
        return {
            "command": command,
            "returncode": done.returncode,
            "stdout": done.stdout[-capture_limit:],
            "stderr": done.stderr[-4000:],
        }
    except Exception as exc:
        return {"command": command, "error": f"{type(exc).__name__}: {exc}"}


def _allocated_gpu_uuids() -> list[str]:
    try:
        raw = GPU_ALLOCATION_FILE.read_text()
    except OSError:
        return []
    return [item.strip() for item in raw.replace("\n", ",").split(",") if item.strip()]


def _gpu_count(stdout: Any) -> int:
    return len([line for line in str(stdout or "").splitlines() if line.strip()])


class TaskInspector:
    def inspect(self, path: Path) -> dict[str, Any]:
        extensions = {".py", ".toml", ".yaml", ".yml", ".json", ".sh"}
        files = []
        ignored = {
            ".git",
            ".venv",
            ".deps",
            "node_modules",
            "runs",
            "__pycache__",
            ".pytest_cache",
        }
        for root, directories, names in os.walk(path):
            directories[:] = [name for name in directories if name not in ignored]
            root_path = Path(root)
            for name in names:
                item = root_path / name
                if item.suffix in extensions:
                    files.append(str(item.relative_to(path)))
                    if len(files) >= 500:
                        break
            if len(files) >= 500:
                break
        entrypoints = [
            x
            for x in files
            if any(
                key in x.lower() for key in ("main", "run", "launch", "serve", "docker", "compose")
            )
        ]
        return {
            "root": str(path.resolve()),
            "file_count_sampled": len(files),
            "entrypoint_candidates": entrypoints[:80],
            "files": files[:200],
        }


class HardwareInspector:
    """Report hardware facts without inventing any.

    ``source`` always names where a GPU fact came from, so a reader can tell a
    real allocation-aware probe from an empty host probe.
    """

    def _probe_gpus(self, allocation: list[str], gpu_probe_image: str | None) -> dict[str, Any]:
        if gpu_probe_image and allocation:
            result = _run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--runtime=nvidia",
                    "-e",
                    f"NVIDIA_VISIBLE_DEVICES={','.join(allocation)}",
                    gpu_probe_image,
                    *GPU_QUERY,
                ],
                timeout=180,
            )
            result["source"] = "docker_allocation"
            result["gpu_count"] = _gpu_count(result.get("stdout"))
            if result.get("returncode") == 0 and result["gpu_count"]:
                return result
            docker_failure = result
        else:
            docker_failure = None

        if shutil.which("nvidia-smi"):
            result = _run(GPU_QUERY)
            result["source"] = "host"
            result["gpu_count"] = _gpu_count(result.get("stdout"))
            if result.get("returncode") == 0 and result["gpu_count"]:
                return result
            host_result = result
        else:
            host_result = {"available": False, "source": "host", "gpu_count": 0}

        if allocation:
            fallback: dict[str, Any] = {
                "source": "allocation_file",
                "gpu_count": len(allocation),
                "gpu_uuids": allocation,
                "note": (
                    "GPU query commands could not enumerate devices; only this "
                    "account's allocated UUID list is authoritative."
                ),
            }
        else:
            fallback = {
                "source": "unavailable",
                "gpu_count": 0,
                "note": "No allocation file and no enumerable GPU.",
            }
        if docker_failure is not None:
            fallback["docker_probe"] = docker_failure
        fallback["host_probe"] = host_result
        return fallback

    def inspect(self, gpu_probe_image: str | None = None) -> dict[str, Any]:
        allocation = _allocated_gpu_uuids()
        profile: dict[str, Any] = {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "memory": _run(["sh", "-lc", "free -b 2>/dev/null || vm_stat 2>/dev/null"]),
            "allocation": {
                "source": str(GPU_ALLOCATION_FILE),
                "exists": GPU_ALLOCATION_FILE.exists(),
                "gpu_uuids": allocation,
                "gpu_count": len(allocation),
                "nvidia_visible_devices_env": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
            },
        }
        profile["gpus"] = self._probe_gpus(allocation, gpu_probe_image)
        if gpu_probe_image and allocation:
            profile["topology"] = _run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--runtime=nvidia",
                    "-e",
                    f"NVIDIA_VISIBLE_DEVICES={','.join(allocation)}",
                    gpu_probe_image,
                    "nvidia-smi",
                    "topo",
                    "-m",
                ],
                timeout=180,
            )
            profile["topology"]["source"] = "docker_allocation"
        elif shutil.which("nvidia-smi"):
            profile["topology"] = _run(["nvidia-smi", "topo", "-m"])
        else:
            profile["topology"] = {"available": False}
        return profile


class VLLMInspector:
    """Discover the installed vLLM and the flags it really accepts.

    vLLM refuses to build its CLI parser without a visible CUDA device, so the
    runtime probe only works inside the GPU container. The container probe is
    therefore preferred and is always allocation-aware; the source scan is a
    labelled last resort because it both misses flags and invents a few.
    """

    @staticmethod
    def _extract_flags(text: str) -> list[str]:
        found = set(re.findall(r"--[a-z0-9][a-z0-9-]+", text))
        return sorted(flag for flag in found if re.fullmatch(r"--[a-z0-9][a-z0-9-]*[a-z0-9]", flag))

    @staticmethod
    def _source_flags(python_executable: str) -> tuple[list[str], str | None]:
        location = _run(
            [
                python_executable,
                "-c",
                (
                    "import importlib.util; "
                    "s=importlib.util.find_spec('vllm'); "
                    "print(next(iter(s.submodule_search_locations)))"
                ),
            ],
            env=_clean_python_env(),
        )
        if location.get("returncode") != 0:
            return [], None
        root = Path(str(location.get("stdout", "")).strip())
        if not root.is_dir():
            return [], None
        flags: set[str] = set()
        for pattern in ("*arg*.py", "*cli*.py", "*config*.py"):
            for source in root.rglob(pattern):
                try:
                    flags.update(
                        re.findall(r"--[a-z0-9][a-z0-9-]+", source.read_text(errors="ignore"))
                    )
                except OSError:
                    continue
        return sorted(flags), str(root)

    @classmethod
    def _host_help(cls, binary: str) -> tuple[dict[str, Any] | None, list[str]]:
        environment = _clean_python_env()
        for arguments in (["serve", "--help=all"], ["serve", "--help"]):
            result = _run([binary, *arguments], timeout=120, env=environment, capture_limit=400000)
            flags = cls._extract_flags(str(result.get("stdout", "")))
            if result.get("returncode") == 0 and flags:
                return result, flags
        last = _run([binary, "serve", "--help=all"], timeout=120, env=environment)
        return last, []

    @classmethod
    def _docker_help(
        cls, binary: str, image: str, gpu_uuids: list[str]
    ) -> tuple[dict[str, Any] | None, list[str]]:
        allocation = gpu_uuids or _allocated_gpu_uuids()
        if not allocation:
            return {"available": False, "reason": "no allocated GPU UUIDs"}, []
        # The image's own ``vllm`` entrypoint hard-codes the host environment
        # path, so the environment must be mounted at the identical path.
        environment_root = Path(binary).resolve().parent.parent
        command = [
            "docker",
            "run",
            "--rm",
            "--runtime=nvidia",
            "-e",
            f"NVIDIA_VISIBLE_DEVICES={','.join(allocation)}",
            "-v",
            f"{environment_root}:{environment_root}:ro",
            image,
            binary,
            "serve",
            "--help=all",
        ]
        result = _run(command, timeout=420, env=_clean_python_env(), capture_limit=400000)
        return result, cls._extract_flags(str(result.get("stdout", "")))

    def inspect(
        self,
        python_executable: str = "python",
        vllm_binary: str | None = None,
        gpu_probe_image: str | None = None,
        gpu_uuids: list[str] | None = None,
    ) -> dict[str, Any]:
        binary = vllm_binary or os.environ.get("EVOTAILOR_VLLM_BIN") or shutil.which("vllm")
        if binary:
            adjacent_python = str(Path(binary).with_name("python"))
            if Path(adjacent_python).is_file():
                python_executable = adjacent_python
        version = _run(
            [python_executable, "-c", "import vllm; print(vllm.__version__)"],
            env=_clean_python_env(),
        )

        flags: list[str] = []
        flag_source = "installed_source"
        source_root: str | None = None
        help_result: dict[str, Any] | None = None

        if binary and gpu_probe_image:
            help_result, flags = self._docker_help(binary, gpu_probe_image, gpu_uuids or [])
            if flags:
                flag_source = "docker_runtime_help"
        elif binary:
            help_result, flags = self._host_help(binary)
            if flags:
                flag_source = "runtime_help"
        if not flags:
            flags, source_root = self._source_flags(python_executable)
            flag_source = "installed_source"

        return {
            "version": version,
            "binary": binary,
            "supported_flags": flags,
            "flag_source": flag_source,
            "source_root": source_root,
            "help": help_result if help_result is not None else {"available": False},
        }


def accepted_measurement_facts(served_model=None, scenario=None) -> dict[str, Any]:
    profile_value = os.environ.get("EVOTAILOR_AIC_PROFILE")
    if not profile_value or not served_model or not scenario:
        return {}
    profile_path = Path(profile_value).expanduser().resolve()
    profile = json.loads(profile_path.read_text())
    declared = profile.get("runtime_identity") or {}
    expected_model = declared.get("served_model_path")
    if (
        served_model
        and expected_model
        and Path(served_model).resolve() != Path(expected_model).resolve()
    ):
        return {}
    if scenario:
        for relative, key in (
            ("init/init_config.json", "scenario_config_sha256"),
            ("init/steps.yaml", "scenario_steps_sha256"),
        ):
            path = Path(scenario) / relative
            if declared.get(key) and (
                not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != declared[key]
            ):
                return {}
    state = Path(profile.get("state_path") or profile_path.with_suffix(".observations.jsonl"))
    if not state.is_absolute():
        state = profile_path.parent / state
    if not state.exists():
        return {}
    identity = hashlib.sha256(
        json.dumps(
            {key: profile[key] for key in ("engine_config", "runtime_identity") if key in profile},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    with state.open() as handle:
        fcntl.flock(handle, fcntl.LOCK_SH)
        records = [json.loads(line) for line in handle if line.strip()]
    records = [
        r
        for r in records
        if r.get("schema_version") == 2
        and r.get("runtime_identity") == identity
        and r.get("status") == "succeeded"
    ]
    if not records:
        return {}
    return {
        "available": fact(True, grade=GRADE_MEASURED, source=str(state)),
        "measured_task_records": fact(
            [
                {
                    "observation_id": r["observation_id"],
                    "parameters": r["candidate_parameters"],
                    "objective": r["objective"],
                    "metrics": r["metrics"],
                    "request_shape": r.get("request_shape"),
                    "fpm_source_sha256": r["source_sha256"],
                }
                for r in records[-24:]
            ],
            grade=GRADE_MEASURED,
            source=str(state),
            note="Only admitted real tasks matching the current model, workload and runtime identity.",
        ),
    }


def _literal_or_none(node: ast.AST) -> Any:
    """Evaluate a module-level literal, or None when the value is computed.

    The execution contract is read straight out of the adapter module so it can be
    recorded without importing and running it. A module that computes that value is
    still valid Python, and a discovery pass must not fail because of it; the fact is
    simply left unrecorded.
    """
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        return None


def write_discovery(
    target: Path,
    task_root: Path,
    vllm_binary: str | None = None,
    gpu_probe_image: str | None = None,
    model_path: Path | str | None = None,
    project_root: Path | str | None = None,
    scenario_path: Path | str | None = None,
    history_paths: Iterable[Path | str] = (),
) -> dict[str, Any]:
    image = gpu_probe_image or os.environ.get("EVOTAILOR_DISCOVERY_GPU_IMAGE") or None
    allocation = _allocated_gpu_uuids()
    served_model = model_path or os.environ.get("EVOTAILOR_SERVED_MODEL_PATH") or None
    scenario = scenario_path or os.environ.get("EVOTAILOR_SCENARIO_PATH") or None
    # The workload profile is what the space designer cites. It is built from the
    # task's own code and configuration, so it stands even when no earlier run
    # exists to learn from.
    workload_profile = WorkloadProfiler(
        model_path=Path(served_model) if served_model else None,
        project_root=Path(project_root) if project_root else None,
        scenario_path=Path(scenario) if scenario else None,
        history_paths=[Path(item) for item in history_paths],
    ).profile(task_root)
    workload_profile["workload"]["history"].update(
        accepted_measurement_facts(served_model, scenario)
    )
    runtime_path = os.environ.get("EVOTAILOR_RUNTIME_CONFIG")
    if runtime_path:
        import yaml

        runtime = yaml.safe_load(Path(runtime_path).read_text()) or {}
        command = runtime.get("workload_command") or []
        if any(Path(str(value)).name == "agentsociety_runner.py" for value in command):
            adapter = (
                Path(__file__).resolve().parents[3] / "scripts/task_adapters/agentsociety_n8.py"
            )
            tree = ast.parse(adapter.read_text())
            declared = next(
                (
                    value
                    for value in (
                        _literal_or_none(node.value)
                        for node in tree.body
                        if isinstance(node, ast.AnnAssign)
                        and isinstance(node.target, ast.Name)
                        and node.target.id == "DEFAULT_FIXED_ENVIRONMENT"
                    )
                    if isinstance(value, dict)
                ),
                {},
            )
            workload_profile["workload"]["execution_contract"] = {
                key: fact(
                    declared[key],
                    grade=GRADE_DECLARED,
                    source=f"{adapter}:DEFAULT_FIXED_ENVIRONMENT",
                )
                for key in (
                    "AGENTSOCIETY_LLM_RAY_MAX_WORKERS",
                    "AGENTSOCIETY_LLM_RAY_CONCURRENCY",
                    "AGENTSOCIETY_BATCH_SIZE",
                )
                if key in declared
            }
    result = {
        "task": TaskInspector().inspect(task_root),
        **workload_profile,
        "hardware": HardwareInspector().inspect(gpu_probe_image=image),
        "vllm": VLLMInspector().inspect(
            vllm_binary=vllm_binary,
            gpu_probe_image=image,
            gpu_uuids=allocation,
        ),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, default=str) + "\n")
    return result
