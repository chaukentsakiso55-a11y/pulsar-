from __future__ import annotations

import json
import re
import sqlite3
import threading
import uuid
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
                    provider_id TEXT,
                    reasoning_effort TEXT,
                    passes INTEGER NOT NULL DEFAULT 1,
                    FOREIGN KEY(key_id) REFERENCES api_keys(id)
                );

                CREATE TABLE IF NOT EXISTS knowledge_chunks (
                    id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tags TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS conversation_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_usage_key_created
                ON usage_events(key_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_conversation_id
                ON conversation_messages(conversation_id, id);
                """
            )
            # Lightweight migrations for databases created by Pulsar v0.2.
            columns = {r[1] for r in conn.execute("PRAGMA table_info(usage_events)").fetchall()}
            for name, ddl in [
                ("provider_id", "ALTER TABLE usage_events ADD COLUMN provider_id TEXT"),
                ("reasoning_effort", "ALTER TABLE usage_events ADD COLUMN reasoning_effort TEXT"),
                ("passes", "ALTER TABLE usage_events ADD COLUMN passes INTEGER NOT NULL DEFAULT 1"),
            ]:
                if name not in columns:
                    conn.execute(ddl)

    def insert_key(self, record: dict[str, Any]) -> None:
        with self._lock, self.connect() as conn:
            conn.execute(
                """
                INSERT INTO api_keys
                (id, name, key_hash, prefix, permissions, daily_limit, created_at, revoked_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    record["id"], record["name"], record["key_hash"], record["prefix"],
                    json.dumps(record["permissions"]), record["daily_limit"], record["created_at"],
                ),
            )

    def find_key_by_hash(self, key_hash: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM api_keys WHERE key_hash = ?", (key_hash,)).fetchone()
        return self._key_row(row) if row else None

    def list_keys(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM api_keys ORDER BY created_at DESC").fetchall()
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
        provider_id: str | None = None,
        reasoning_effort: str | None = None,
        passes: int = 1,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self.connect() as conn:
            conn.execute(
                """
                INSERT INTO usage_events
                (key_id, created_at, endpoint, model, prompt_tokens, completion_tokens, provider_id, reasoning_effort, passes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (key_id, now, endpoint, model, prompt_tokens, completion_tokens, provider_id, reasoning_effort, passes),
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
                       COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                       COALESCE(SUM(passes), 0) AS orchestration_passes
                FROM usage_events
                """
            ).fetchone()
            by_model = conn.execute(
                "SELECT model, COUNT(*) AS requests FROM usage_events GROUP BY model ORDER BY requests DESC"
            ).fetchall()
            by_provider = conn.execute(
                "SELECT COALESCE(provider_id, 'unknown') AS provider, COUNT(*) AS requests FROM usage_events GROUP BY provider_id ORDER BY requests DESC"
            ).fetchall()
        return {
            "requests": int(totals["requests"]),
            "prompt_tokens": int(totals["prompt_tokens"]),
            "completion_tokens": int(totals["completion_tokens"]),
            "orchestration_passes": int(totals["orchestration_passes"]),
            "by_model": [dict(r) for r in by_model],
            "by_provider": [dict(r) for r in by_provider],
        }

    def add_knowledge(self, source: str, content: str, tags: list[str] | None = None) -> str:
        chunk_id = f"kn_{uuid.uuid4().hex[:20]}"
        with self._lock, self.connect() as conn:
            conn.execute(
                "INSERT INTO knowledge_chunks (id, source, content, tags, created_at) VALUES (?, ?, ?, ?, ?)",
                (chunk_id, source, content, json.dumps(tags or []), datetime.now(timezone.utc).isoformat()),
            )
        return chunk_id

    @staticmethod
    def _terms(text: str) -> set[str]:
        return {w.lower() for w in re.findall(r"[A-Za-z0-9_'-]{3,}", text) if len(w) >= 3}

    def search_knowledge(self, query: str, limit: int = 4) -> list[dict[str, Any]]:
        terms = self._terms(query)
        if not terms or limit <= 0:
            return []
        with self.connect() as conn:
            rows = conn.execute("SELECT id, source, content, tags, created_at FROM knowledge_chunks ORDER BY created_at DESC LIMIT 500").fetchall()
        scored: list[tuple[int, sqlite3.Row]] = []
        for row in rows:
            hay = self._terms(f"{row['source']} {row['content']} {' '.join(json.loads(row['tags']))}")
            score = len(terms & hay)
            if score:
                scored.append((score, row))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [dict(row) | {"score": score, "tags": json.loads(row["tags"])} for score, row in scored[:limit]]

    def add_message(self, conversation_id: str, role: str, content: str) -> None:
        with self._lock, self.connect() as conn:
            conn.execute(
                "INSERT INTO conversation_messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (conversation_id, role, content, datetime.now(timezone.utc).isoformat()),
            )

    def get_conversation(self, conversation_id: str, limit: int = 12) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT role, content, created_at FROM conversation_messages WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
                (conversation_id, limit),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    @staticmethod
    def _key_row(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["permissions"] = json.loads(data["permissions"])
        data.pop("key_hash", None)
        return data
