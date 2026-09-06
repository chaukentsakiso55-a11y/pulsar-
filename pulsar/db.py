from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class Database:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def init(self) -> None:
        with self._lock, self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS api_keys (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    key_hash TEXT NOT NULL UNIQUE,
                    prefix TEXT NOT NULL,
                    permissions TEXT NOT NULL,
                    daily_limit INTEGER NOT NULL DEFAULT 1000,
                    created_at TEXT NOT NULL,
                    revoked_at TEXT
                );

                CREATE TABLE IF NOT EXISTS usage_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    model TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY(key_id) REFERENCES api_keys(id)
                );

                CREATE INDEX IF NOT EXISTS idx_usage_key_created
                ON usage_events(key_id, created_at);
                """
            )

    def insert_key(self, record: dict[str, Any]) -> None:
        with self._lock, self.connect() as conn:
            conn.execute(
                """
                INSERT INTO api_keys
                (id, name, key_hash, prefix, permissions, daily_limit, created_at, revoked_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    record["id"],
                    record["name"],
                    record["key_hash"],
                    record["prefix"],
                    json.dumps(record["permissions"]),
                    record["daily_limit"],
                    record["created_at"],
                ),
            )

    def find_key_by_hash(self, key_hash: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM api_keys WHERE key_hash = ?", (key_hash,)
            ).fetchone()
        return self._key_row(row) if row else None

    def list_keys(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM api_keys ORDER BY created_at DESC"
            ).fetchall()
        return [self._key_row(row) for row in rows]

    def revoke_key(self, key_id: str) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self.connect() as conn:
            cur = conn.execute(
                "UPDATE api_keys SET revoked_at = ? WHERE id = ? AND revoked_at IS NULL",
                (now, key_id),
            )
            return cur.rowcount > 0

    def record_usage(
        self,
        key_id: str,
        endpoint: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self.connect() as conn:
            conn.execute(
                """
                INSERT INTO usage_events
                (key_id, created_at, endpoint, model, prompt_tokens, completion_tokens)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (key_id, now, endpoint, model, prompt_tokens, completion_tokens),
            )

    def usage_today(self, key_id: str) -> int:
        day = datetime.now(timezone.utc).date().isoformat()
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS c FROM usage_events
                WHERE key_id = ? AND substr(created_at, 1, 10) = ?
                """,
                (key_id, day),
            ).fetchone()
        return int(row["c"])

    def usage_summary(self) -> dict[str, Any]:
        with self.connect() as conn:
            totals = conn.execute(
                """
                SELECT COUNT(*) AS requests,
                       COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                       COALESCE(SUM(completion_tokens), 0) AS completion_tokens
                FROM usage_events
                """
            ).fetchone()
            by_model = conn.execute(
                """
                SELECT model, COUNT(*) AS requests
                FROM usage_events GROUP BY model ORDER BY requests DESC
                """
            ).fetchall()
        return {
            "requests": int(totals["requests"]),
            "prompt_tokens": int(totals["prompt_tokens"]),
            "completion_tokens": int(totals["completion_tokens"]),
            "by_model": [dict(r) for r in by_model],
        }

    @staticmethod
    def _key_row(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["permissions"] = json.loads(data["permissions"])
        data.pop("key_hash", None)
        return data
