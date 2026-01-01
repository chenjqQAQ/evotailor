#!/usr/bin/env python3
"""Run real Campaigns sequentially, consuming explicit previous-cycle experience."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evotailor.config import load_config, load_dotenv
from evotailor.experiment import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--cycles", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--task-root", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    base = load_config(args.config)
    if base.execution_backend not in ("host", "docker") or args.cycles < 2:
        raise ValueError("At least two cycles and a real backend are required")
    if not base.auto_apply_evolution:
        raise ValueError(
            "Enable auto_apply_evolution to execute a multi-cycle evolution experiment"
        )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "cycles.json"
    identity = {
        "configuration": base.model_dump(mode="json"),
        "cycles": args.cycles,
        "task_root": str(args.task_root.resolve()) if args.task_root else None,
    }
    if manifest.exists():
        state = json.loads(manifest.read_text())
        if state["identity"] != identity:
            raise ValueError(
                "Evolution series configuration changed; choose a new output directory"
            )
    else:
        state = {"identity": identity, "completed": []}
        atomic_json(manifest, state)
    memory = list(base.experience_sources)
    for index in range(args.cycles):
        campaign_id = f"{base.campaign_id}-cycle-{index + 1:02d}"
        run_dir = root / base.run_root / campaign_id
        if index < len(state["completed"]):
            memory.append(str(run_dir / "campaign_experience.json"))
            continue
        config = base.model_dump(mode="json")
        config.update(
            campaign_id=campaign_id,
            use_evolved_defaults=index > 0,
            evolution_root=str(output / "evolution"),
            experience_sources=list(memory),
            history_mode="transfer"
            if memory and base.history_mode == "cold"
            else base.history_mode,
        )
        config_path = output / f"cycle-{index + 1:02d}.json"
        atomic_json(config_path, config)
        command = [
            sys.executable,
            "-m",
            "evotailor.cli",
            "run",
            "--config",
            str(config_path),
            "--project-root",
            str(root),
            "--no-dashboard",
            "--json-output",
        ]
        if args.task_root:
            command.extend(["--task-root", str(args.task_root.resolve())])
        with (output / f"cycle-{index + 1:02d}.log").open("a") as log:
            completed = subprocess.run(
                command,
                cwd=root,
                env=os.environ | {"PYTHONPATH": str(root / "src")},
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode:
            raise RuntimeError(
                f"Real cycle {index + 1} failed; inspect its log. No fabricated continuation."
            )
        experience = run_dir / "campaign_experience.json"
        if not experience.is_file():
            raise RuntimeError("Finished cycle has no experience artifact")
        state["completed"].append(
            {
                "campaign_id": campaign_id,
                "experience": str(experience),
                "summary": str(run_dir / "summary.json"),
                "evolution_review": str(run_dir / "campaign_review.json"),
            }
        )
        atomic_json(manifest, state)
        memory.append(str(experience))
    print(json.dumps({"series": str(manifest), "completed": len(state["completed"])}))


if __name__ == "__main__":
    main()
