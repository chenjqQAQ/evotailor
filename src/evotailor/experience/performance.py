"""An independent promotion gate over measured real Campaign databases."""

import json
import math
import sqlite3
import statistics
from pathlib import Path


def assess(report_path, min_repeats=3):
    report = json.loads(Path(report_path).read_text())
    groups = []
    contracts = []
    seen = set()
    for label in ("baseline", "candidate"):
        entries = report.get(label) or []
        if len(entries) < min_repeats:
            raise ValueError("Insufficient independent real search campaigns")
        values, successes, fingerprints = [], [], []
        for entry in entries:
            db = Path(entry["database"]).resolve()
            if db in seen:
                raise ValueError("Repeated campaign database cannot count as independent evidence")
            seen.add(db)
            exp = json.loads((db.parent / "experiment.json").read_text())
            if exp.get("source_revision") != entry["revision"]:
                raise ValueError("Experiment is not bound to measured source revision")
            config = exp["configuration"]
            if config["execution_backend"] not in ("docker", "host"):
                raise ValueError("Simulated performance cannot promote evolved code")
            with sqlite3.connect(db.as_uri() + "?mode=ro", uri=True) as connection:
                rows = [
                    json.loads(r[0])
                    for r in connection.execute("SELECT payload FROM trials ORDER BY trial_index")
                ]
            if len(rows) != config["max_trials"] or any(
                r["measurement_kind"] != "real" for r in rows
            ):
                raise ValueError("Incomplete or non-real campaign")
            contract = {
                k: v
                for k, v in config.items()
                if k
                not in (
                    "campaign_id",
                    "run_root",
                    "auto_apply_evolution",
                    "evolution_root",
                    "evolution_performance_command",
                )
            }
            # Runtime paths can differ only as identifiers; their digests must match.
            external = sorted(v["sha256"] for v in exp["external_configuration"].values())
            fingerprints.append(
                {"config": contract, "space": exp["space_hash"], "external": external}
            )
            ok = [r for r in rows if r["status"] == "succeeded" and r.get("objective") is not None]
            if not ok:
                raise ValueError("No real successful measurement")
            if any(
                not all(
                    r.get("validity", {}).get(k) is True
                    for k in (
                        "workload_completed",
                        "cleanup_success",
                        "served_model_matches_all_requests",
                    )
                )
                or not math.isfinite(r["objective"])
                or r.get("metrics", {}).get(config["objective_metric"]) != r["objective"]
                for r in ok
            ):
                raise ValueError("Missing workload validity")
            direction = config["objective_direction"]
            best = (min if direction == "minimize" else max)(r["objective"] for r in ok)
            values.append(best)
            successes.append(len(ok) / len(rows))
            source = Path(entry["source_root"])
            digest = (
                __import__("subprocess")
                .run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=source,
                    text=True,
                    capture_output=True,
                    check=True,
                )
                .stdout.strip()
            )
            if digest != entry["revision"]:
                raise ValueError("Measured source revision changed")
        contracts.append(fingerprints)
        groups.append(
            {
                "best_median": statistics.median(values),
                "success_median": statistics.median(successes),
                "search_repeats": len(values),
            }
        )
    if any(a != b for a, b in zip(contracts[0], contracts[1], strict=True)):
        raise ValueError("Unmatched budget, seed, protocol, history or runtime")
    baseline, candidate = groups
    direction = contracts[0][0]["config"]["objective_direction"]
    improved = (
        candidate["best_median"] < baseline["best_median"]
        if direction == "minimize"
        else candidate["best_median"] > baseline["best_median"]
    )
    return {
        "accepted": improved and candidate["success_median"] >= baseline["success_median"],
        "baseline": baseline,
        "candidate": candidate,
        "note": "Measured median improvement with no median search-success regression; not a statistical significance claim.",
    }
