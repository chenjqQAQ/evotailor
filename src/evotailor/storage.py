from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from evotailor.contracts import Candidate, Prediction, Selection, TrialResult, utc_now


class EventStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript("""
        CREATE TABLE IF NOT EXISTS events(sequence INTEGER PRIMARY KEY AUTOINCREMENT,created_at TEXT NOT NULL,event_type TEXT NOT NULL,payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS candidates(candidate_id TEXT NOT NULL,trial_index INTEGER NOT NULL,payload TEXT NOT NULL,PRIMARY KEY(candidate_id,trial_index));
            CREATE TABLE IF NOT EXISTS predictions(candidate_id TEXT NOT NULL,trial_index INTEGER NOT NULL,payload TEXT NOT NULL,PRIMARY KEY(candidate_id,trial_index));
        CREATE TABLE IF NOT EXISTS selections(trial_index INTEGER PRIMARY KEY,payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS trials(trial_id TEXT PRIMARY KEY,trial_index INTEGER UNIQUE NOT NULL,payload TEXT NOT NULL);
        """)
        self.connection.commit()

    def append(self, event_type: str, payload: dict[str, Any]) -> None:
        self.connection.execute(
            "INSERT INTO events(created_at,event_type,payload) VALUES(?,?,?)",
            (utc_now().isoformat(), event_type, json.dumps(payload, default=str, sort_keys=True)),
        )
        self.connection.commit()

    def save_candidates(self, i: int, items: list[Candidate]) -> None:
        self.connection.executemany(
            "INSERT OR REPLACE INTO candidates(candidate_id,trial_index,payload) VALUES(?,?,?)",
            [(x.candidate_id, i, x.model_dump_json()) for x in items],
        )
        self.connection.commit()

    def save_predictions(self, i: int, items: list[Prediction]) -> None:
        self.connection.executemany(
            "INSERT OR REPLACE INTO predictions(candidate_id,trial_index,payload) VALUES(?,?,?)",
            [(x.candidate_id, i, x.model_dump_json()) for x in items],
        )
        self.connection.commit()

    def save_selection(self, i: int, item: Selection) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO selections(trial_index,payload) VALUES(?,?)",
            (i, item.model_dump_json()),
        )
        self.connection.commit()

    def save_trial(self, i: int, item: TrialResult) -> None:
        row = self.connection.execute(
            "SELECT payload,trial_index FROM trials WHERE trial_id=?", (item.trial_id,)
        ).fetchone()
        if row is not None:
            previous = TrialResult.model_validate_json(row["payload"])
            immutable = (
                "trial_id",
                "candidate",
                "status",
                "objective",
                "metrics",
                "started_at",
                "finished_at",
                "measurement_kind",
                "stage_id",
                "space_hash",
                "code_version",
                "resource_snapshot",
                "validity",
                "failure_type",
                "error",
            )
            if row["trial_index"] != i or any(
                getattr(previous, k) != getattr(item, k) for k in immutable
            ):
                raise ValueError(
                    "Completed trial facts and validity are immutable"
                )
        self.connection.execute(
            "INSERT OR REPLACE INTO trials(trial_id,trial_index,payload) VALUES(?,?,?)",
            (item.trial_id, i, item.model_dump_json()),
        )
        self.connection.commit()

    def candidate_records(self) -> list[Candidate]:
        rows = self.connection.execute(
            "SELECT payload FROM candidates ORDER BY trial_index, candidate_id"
        ).fetchall()
        return [Candidate.model_validate_json(row["payload"]) for row in rows]

    def prediction_map(self) -> dict[str, Prediction]:
        rows = self.connection.execute(
            "SELECT payload FROM predictions ORDER BY trial_index"
        ).fetchall()
        return {
            item.candidate_id: item
            for item in (Prediction.model_validate_json(row["payload"]) for row in rows)
        }

    def trials(self) -> list[TrialResult]:
        return [
            TrialResult.model_validate_json(r["payload"])
            for r in self.connection.execute("SELECT payload FROM trials ORDER BY trial_index")
        ]

    def trial_rows(self) -> list[sqlite3.Row]:
        return self.connection.execute(
            "SELECT trial_index FROM trials ORDER BY trial_index"
        ).fetchall()

    def events(self) -> list[dict[str, Any]]:
        return [
            dict(r) | {"payload": json.loads(r["payload"])}
            for r in self.connection.execute("SELECT * FROM events ORDER BY sequence")
        ]

    def event(self, event_type: str, key: str, value: Any) -> dict | None:
        return next(
            (
                e["payload"]
                for e in self.events()
                if e["event_type"] == event_type and e["payload"].get(key) == value
            ),
            None,
        )

    def append_once(self, event_type: str, payload: dict, key: str) -> None:
        if self.event(event_type, key, payload[key]) is None:
            self.append(event_type, payload)

    def prediction(self, index: int, candidate_id: str) -> Prediction | None:
        row = self.connection.execute(
            "SELECT payload FROM predictions WHERE trial_index=? AND candidate_id=?",
            (index, candidate_id),
        ).fetchone()
        return None if row is None else Prediction.model_validate_json(row["payload"])

    def selection(self, index: int) -> Selection | None:
        row = self.connection.execute(
            "SELECT payload FROM selections WHERE trial_index=?", (index,)
        ).fetchone()
        return None if row is None else Selection.model_validate_json(row["payload"])
