#!/usr/bin/env python3
"""Read preserved real measurements. Write a separate CPU replay dataset, never trial facts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
from pathlib import Path


def sample(items, cap=512):
    return (
        items
        if len(items) <= cap
        else [items[i * (len(items) - 1) // (cap - 1)] for i in range(cap)]
    )


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def read_real_fpms(path):
    # A fixed sample controls CPU replay cost; all source counts and hashes are retained.
    count = sum(1 for line in path.open() if line.strip())
    wanted = set(i * (count - 1) // 511 for i in range(512)) if count > 512 else set(range(count))
    rows = []
    with path.open() as f:
        for i, line in enumerate(f):
            if i in wanted:
                row = json.loads(line)
                if isinstance(row.get("wall_time"), (int, float)) and row["wall_time"] > 0:
                    rows.append([row])
    return rows, count


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--state", type=Path)
    p.add_argument("--run", type=Path)
    p.add_argument("--space", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    records = []
    provenance = {}
    if args.state:
        provenance = {"state_path": str(args.state), "state_sha256": digest(args.state)}
        with args.state.open() as f:
            for line in f:
                r = json.loads(line)
                if r.get("schema_version") != 2:
                    raise ValueError("State-only replay requires schema 2 real task observations")
                keep = {
                    k: v
                    for k, v in r.items()
                    if k not in {"iterations", "prediction_before_execution", "prequential_error"}
                }
                keep["iterations"] = sample(
                    [it for it in r["iterations"] if all(row.get("wall_time", 0) > 0 for row in it)]
                )
                keep["started_at"] = r["observation_id"].split("@", 1)[1]
                records.append(keep)
        space_path = args.space
    else:
        db = args.run / "campaign.sqlite3"
        provenance = {"database_path": str(db), "database_sha256": digest(db)}
        with sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True) as c:
            trials = [
                json.loads(x[0])
                for x in c.execute("SELECT payload FROM trials ORDER BY trial_index")
            ]
        for t in trials:
            if t.get("measurement_kind") != "real":
                raise ValueError("Refuse non-real trial data")
            artifact = Path(t.get("telemetry", {}).get("forward_pass_metrics_path", "/nonexistent"))
            # Only portfolio FPMs are sampled; every arm contributes real task facts.
            iterations, count = (
                read_real_fpms(artifact)
                if "portfolio" in args.run.name and artifact.is_file()
                else ([], 0)
            )
            records.append(
                dict(
                    observation_id=args.run.name + ":" + t["trial_id"] + "@" + t["started_at"],
                    trial_id=t["trial_id"],
                    started_at=t["started_at"],
                    candidate_parameters=t["candidate"]["parameters"],
                    status=t["status"],
                    objective=t.get("objective"),
                    metrics=t.get("metrics", {}),
                    validity=t.get("validity", {}),
                    iterations=iterations,
                    source_iterations=count,
                    source_path=str(artifact),
                    source_sha256=digest(artifact) if iterations else None,
                )
            )
        space_path = args.run / "frozen_search_space.json"
    records.sort(key=lambda r: r["started_at"])
    for r in records:
        if r.get("objective") is not None and not math.isfinite(r["objective"]):
            raise ValueError("nonfinite objective")
    result = dict(
        dataset=args.output.stem,
        space=json.loads(space_path.read_text()),
        records=records,
        provenance=provenance,
        notes=[
            "Separate replay copy. Original trials, objectives and calibration files were not modified.",
            "Forward-pass evaluation is conditional on held-out measured schedule descriptors; held-out wall_time is never given to the estimator.",
            "No GPU experiments or synthetic performance data.",
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
    print(
        json.dumps(
            dict(
                output=str(args.output),
                records=len(records),
                successful=sum(r["status"] == "succeeded" for r in records),
                sampled_fpm=sum(len(r["iterations"]) for r in records),
            )
        )
    )


if __name__ == "__main__":
    main()
