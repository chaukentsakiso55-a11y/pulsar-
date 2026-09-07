from __future__ import annotations

import json
import re
import sqlite3
from typing import Any

from pulsar.db import Database


class KnowledgeRetriever:
    """FTS5/BM25 retrieval with an automatic lexical fallback."""

    def __init__(self, db: Database):
        self.db = db
        self._fts_available: bool | None = None

    @staticmethod
    def _query(text: str) -> str:
        terms = [t.lower() for t in re.findall(r"[A-Za-z0-9_'-]{2,}", text)]
        unique = list(dict.fromkeys(terms))[:24]
        return " OR ".join(f"{term}*" for term in unique)

    def _ensure_index(self, conn: sqlite3.Connection) -> bool:
        if self._fts_available is False:
            return False
        try:
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts
                USING fts5(
                    id UNINDEXED,
                    source,
                    content,
                    tags,
                    tokenize='porter unicode61'
                )
                """
            )
            conn.execute(
                """
                INSERT INTO knowledge_fts(id, source, content, tags)
                SELECT k.id, k.source, k.content, k.tags
                FROM knowledge_chunks AS k
                WHERE NOT EXISTS (
                    SELECT 1 FROM knowledge_fts AS f WHERE f.id = k.id
                )
                """
            )
            self._fts_available = True
            return True
        except sqlite3.OperationalError:
            self._fts_available = False
            return False

    def search(self, query: str, limit: int = 4) -> list[dict[str, Any]]:
        fts_query = self._query(query)
        if not fts_query or limit <= 0:
            return []

        with self.db.connect() as conn:
            if not self._ensure_index(conn):
                return self.db.search_knowledge(query, limit=limit)
            try:
                rows = conn.execute(
                    """
                    SELECT
                        k.id,
                        k.source,
                        k.content,
                        k.tags,
                        k.created_at,
                        bm25(knowledge_fts, 0.0, 3.0, 1.0, 0.5) AS rank
                    FROM knowledge_fts
                    JOIN knowledge_chunks AS k ON k.id = knowledge_fts.id
                    WHERE knowledge_fts MATCH ?
                    ORDER BY rank ASC, k.created_at DESC
                    LIMIT ?
                    """,
                    (fts_query, limit),
                ).fetchall()
            except sqlite3.OperationalError:
                self._fts_available = False
                return self.db.search_knowledge(query, limit=limit)

        return [
            dict(row)
            | {
                "score": float(-row["rank"]),
                "tags": json.loads(row["tags"]),
                "retrieval": "fts5-bm25",
            }
            for row in rows
        ]
