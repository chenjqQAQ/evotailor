"""Immutable experiment identity and atomic checkpoint artifacts."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
from functools import wraps
from pathlib import Path
from typing import Any


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        handle.write(
            json.dumps(payload, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_experiment(config, root: Path, run_dir: Path, space_hash: str) -> dict:
    fields = (
        "history_seed_trials",
        "history_guided_search",
        "max_modeling_trials",
        "enable_parameter_experiments",
        "history_mode",
        "history_sources",
        "experience_sources",
        "objective_unit",
        "measurement_protocol",
        "measurement_scope",
        "evolution_performance_command",
        "evolution_min_repeats",
        "evolution_performance_timeout_s",
        "run_mode",
        "execution_backend",
        "max_trials",
        "seed",
        "objective_metric",
        "objective_direction",
        "command",
        "aiconfigurator_command",
        "enabled_proposers",
        "baseline_proposer",
        "mature_optimizer_policies",
        "final_validation_trials",
        "final_validation_repeats",
        "final_validation_arms",
        "episode_min_trials",
        "episode_max_trials",
        "episode_min_count",
        "episode_max_count",
        "model",
        "base_url",
        "use_llm_selector",
        "use_outer_agent",
        "use_episode_boundary_agent",
        "auto_promote_policy",
        "use_episode_planner",
        "use_prior_knowledge",
        "use_evolved_defaults",
        "auto_apply_evolution",
        "evolution_root",
        "discovery_artifact",
        "enforce_weight_memory_feasibility",
    )
    # Preserve the implicit adaptive behavior of older manifests. Explicit
    # fixed pools add a frozen field, so switching availability semantics requires
    # a new Campaign identity.
    if config.fixed_candidate_pool:
        fields = (*fields, "fixed_candidate_pool")
    all_config = config.model_dump(mode="json")
    external = {}
    for name in ("EVOTAILOR_RUNTIME_CONFIG", "EVOTAILOR_AIC_PROFILE"):
        value = os.environ.get(name)
        if value:
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = root / path
            path = path.resolve()
            external[name] = {"path": str(path), "sha256": file_digest(path)}
    for name in config.history_sources + config.experience_sources:
        path = Path(name).expanduser()
        if not path.is_absolute():
            path = root / path
        external[str(path.resolve())] = {"sha256": file_digest(path)}
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=False
    )
    identity = {
        "source_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "schema_version": 1,
        "space_hash": space_hash,
        "configuration": {k: all_config[k] for k in fields},
        "external_configuration": external,
    }
    path = run_dir / "experiment.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous != identity:
            raise ValueError("Campaign experiment identity changed; start a new campaign_id")
    else:
        if (run_dir / "campaign.sqlite3").exists():
            import sqlite3

            with sqlite3.connect(run_dir / "campaign.sqlite3") as db:
                exists = db.execute("SELECT name FROM sqlite_master WHERE name='trials'").fetchone()
                if exists and db.execute("SELECT COUNT(*) FROM trials").fetchone()[0]:
                    raise ValueError(
                        "Existing Campaign has no immutable experiment manifest; explicit migration required"
                    )
        atomic_json(path, identity)
    return identity


def serialized_campaign(method):
    @wraps(method)
    def run(self, *args, **kwargs):
        with (self.run_dir / ".campaign.lock").open("a+") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Another process is already running this Campaign") from exc
            return method(self, *args, **kwargs)

    return run


def discovery_identity(task_root: Path, runtime: dict) -> dict:
    ignored = {
        "agentsociety_data",
        "artifacts",
        "logs",
        ".cache",
        ".git",
        ".venv",
        ".deps",
        "node_modules",
        "runs",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
    }
    suffixes = {".py", ".json", ".yaml", ".yml", ".toml", ".sh"}
    digest = hashlib.sha256()
    count = 0
    for base, folders, files in os.walk(task_root):
        folders[:] = sorted(x for x in folders if x not in ignored)
        for name in sorted(files):
            path = Path(base) / name
            if path.suffix not in suffixes:
                continue
            digest.update(str(path.relative_to(task_root)).encode())
            digest.update(path.read_bytes())
            count += 1
    paths = {}
    for key in ("model_path", "scenario_path"):
        if runtime.get(key):
            path = Path(runtime[key])
            selected = (
                [path / "config.json"]
                if key == "model_path"
                else [path / "init/init_config.json", path / "init/steps.yaml"]
            )
            paths[key] = {str(p): file_digest(p) for p in selected if p.is_file()}
    return {
        "task_root": str(task_root.resolve()),
        "source_sha256": digest.hexdigest(),
        "file_count": count,
        "inputs": paths,
        "runtime_configuration_sha256": hashlib.sha256(
            json.dumps(runtime, sort_keys=True).encode()
        ).hexdigest(),
    }
