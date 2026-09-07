from __future__ import annotations

import json
import math
from dataclasses import dataclass

from pulsar.db import Database
from pulsar.embeddings import EmbeddingClient


@dataclass(slots=True)
class SemanticHit:
    id: str
    source: str
    content: str
    score: float
    created_at: str


class SemanticMemory:
    def __init__(self, db: Database, embeddings: EmbeddingClient):
        self.db = db
        self.embeddings = embeddings

    @staticmethod
    def cosine(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return -1.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if not na or not nb:
            return 0.0
        return dot / (na * nb)

    async def remember(self, namespace: str, source: str, content: str) -> str | None:
        content = content.strip()
        if not namespace or not content:
            return None
        batch = await self.embeddings.embed([content])
        return self.db.add_semantic_memory(
            namespace=namespace,
            source=source,
            content=content,
            vector=batch.vectors[0],
            model=batch.model,
        )

    async def search(self, namespace: str, query: str, limit: int = 4) -> list[SemanticHit]:
        if not namespace or not query.strip() or limit <= 0:
            return []
        candidates = self.db.list_semantic_memories(namespace, limit=1200)
        if not candidates:
            return []
        batch = await self.embeddings.embed([query])
        q = batch.vectors[0]
        scored: list[tuple[float, dict]] = []
        for item in candidates:
            try:
                vector = json.loads(item["vector"])
                if not isinstance(vector, list) or len(vector) != len(q):
                    continue
                score = self.cosine(q, [float(x) for x in vector])
                if score < -0.5:
                    continue
            except (ValueError, TypeError, json.JSONDecodeError):
                continue
            scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            SemanticHit(
                id=item["id"], source=item["source"], content=item["content"],
                score=float(score), created_at=item["created_at"],
            )
            for score, item in scored[: min(12, max(1, int(limit)))]
        ]
