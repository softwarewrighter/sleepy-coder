"""Episode data types and loading."""

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator


@dataclass
class Episode:
    """A single learning episode from the agent."""

    task_id: str
    attempt_idx: int
    prompt_hash: str
    model_id: str
    error_signature: str | None
    diff_unified: str | None
    passed: bool
    steps_to_green: int
    wall_clock_ms: int
    tokens_in: int
    tokens_out: int
    timestamp: datetime

    @classmethod
    def from_dict(cls, data: dict) -> "Episode":
        """Create Episode from dictionary."""
        # Parse timestamp, handling 'Z' suffix for UTC
        timestamp = datetime.now()
        if "timestamp" in data:
            ts_str = data["timestamp"]
            # Python 3.10 fromisoformat doesn't handle 'Z', replace with +00:00
            if ts_str.endswith("Z"):
                ts_str = ts_str[:-1] + "+00:00"
            timestamp = datetime.fromisoformat(ts_str)

        return cls(
            task_id=data["task_id"],
            attempt_idx=data["attempt_idx"],
            prompt_hash=data.get("prompt_hash", ""),
            model_id=data.get("model_id", ""),
            error_signature=data.get("error_signature"),
            diff_unified=data.get("diff_unified"),
            passed=data.get("passed", False),
            steps_to_green=data.get("steps_to_green", 0),
            wall_clock_ms=data.get("wall_clock_ms", 0),
            tokens_in=data.get("tokens_in", 0),
            tokens_out=data.get("tokens_out", 0),
            timestamp=timestamp,
        )

    def to_dict(self) -> dict:
        """Convert Episode to dictionary."""
        return {
            "task_id": self.task_id,
            "attempt_idx": self.attempt_idx,
            "prompt_hash": self.prompt_hash,
            "model_id": self.model_id,
            "error_signature": self.error_signature,
            "diff_unified": self.diff_unified,
            "passed": self.passed,
            "steps_to_green": self.steps_to_green,
            "wall_clock_ms": self.wall_clock_ms,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "timestamp": self.timestamp.isoformat(),
        }


def load_episodes_jsonl(path: Path) -> list[Episode]:
    """Load episodes from a JSONL file."""
    episodes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                episodes.append(Episode.from_dict(data))
    return episodes


def iter_episodes_jsonl(path: Path) -> Iterator[Episode]:
    """Iterate over episodes from a JSONL file."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                yield Episode.from_dict(data)


def load_episodes_sqlite(db_path: Path, run_id: str | None = None) -> list[Episode]:
    """Load episodes from SQLite database.

    Args:
        db_path: Path to the SQLite database.
        run_id: Optional run ID to filter by.

    Returns:
        List of Episodes.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if run_id:
        cursor.execute(
            """
            SELECT task_id, attempt_idx, prompt_hash, model_id, error_signature,
                   diff_unified, passed, steps_to_green, wall_clock_ms,
                   tokens_in, tokens_out, timestamp
            FROM episodes
            WHERE model_id = ?
            ORDER BY timestamp
            """,
            (run_id,),
        )
    else:
        cursor.execute(
            """
            SELECT task_id, attempt_idx, prompt_hash, model_id, error_signature,
                   diff_unified, passed, steps_to_green, wall_clock_ms,
                   tokens_in, tokens_out, timestamp
            FROM episodes
            ORDER BY timestamp
            """
        )

    episodes = []
    for row in cursor.fetchall():
        # Parse timestamp, handling 'Z' suffix
        timestamp = datetime.now()
        if row["timestamp"]:
            ts_str = row["timestamp"]
            if ts_str.endswith("Z"):
                ts_str = ts_str[:-1] + "+00:00"
            timestamp = datetime.fromisoformat(ts_str)

        episodes.append(
            Episode(
                task_id=row["task_id"],
                attempt_idx=row["attempt_idx"],
                prompt_hash=row["prompt_hash"] or "",
                model_id=row["model_id"] or "",
                error_signature=row["error_signature"],
                diff_unified=row["diff_unified"],
                passed=bool(row["passed"]),
                steps_to_green=row["steps_to_green"] or 0,
                wall_clock_ms=row["wall_clock_ms"] or 0,
                tokens_in=row["tokens_in"] or 0,
                tokens_out=row["tokens_out"] or 0,
                timestamp=timestamp,
            )
        )

    conn.close()
    return episodes


def filter_successful(episodes: list[Episode]) -> list[Episode]:
    """Filter to only successful episodes."""
    return [e for e in episodes if e.passed]


def filter_failed(episodes: list[Episode]) -> list[Episode]:
    """Filter to only failed episodes."""
    return [e for e in episodes if not e.passed]


def group_by_task(episodes: list[Episode]) -> dict[str, list[Episode]]:
    """Group episodes by task ID."""
    groups: dict[str, list[Episode]] = {}
    for ep in episodes:
        if ep.task_id not in groups:
            groups[ep.task_id] = []
        groups[ep.task_id].append(ep)
    return groups
