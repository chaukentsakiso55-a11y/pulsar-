from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

import httpx


class EmbeddingError(RuntimeError):
    pass


@dataclass(slots=True)
class EmbeddingBatch:
    model: str
    vectors: list[list[float]]

    @property
    def dimensions(self) -> int:
        return len(self.vectors[0]) if self.vectors else 0


class EmbeddingClient:
    """OpenAI-compatible embeddings client with an offline deterministic fallback."""

    def __init__(
        self,
        base_url: str = "",
        api_key: str = "",
        model: str = "",
        *,
        timeout_seconds: float = 30.0,
        fallback_dimensions: int = 384,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = max(1.0, min(180.0, float(timeout_seconds)))
        self.fallback_dimensions = min(1536, max(64, int(fallback_dimensions)))

    @property
    def external_enabled(self) -> bool:
        return bool(self.base_url and self.model)

    @property
    def active_model(self) -> str:
        return self.model if self.external_enabled else "pulsar-embed-lite"

    @staticmethod
    def _features(text: str) -> list[str]:
        words = re.findall(r"[\w'-]{2,}", text.lower(), flags=re.UNICODE)
        features = list(words)
        features.extend(f"{a}_{b}" for a, b in zip(words, words[1:]))
        return features[:8192]

    def _lite_vector(self, text: str) -> list[float]:
        dims = self.fallback_dimensions
        vec = [0.0] * dims
        for feature in self._features(text):
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % dims
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[bucket] += sign
        norm = math.sqrt(sum(x * x for x in vec))
        if norm:
            vec = [x / norm for x in vec]
        return vec

    @staticmethod
    def _validate_vectors(data: object, expected: int) -> list[list[float]]:
        if not isinstance(data, dict) or not isinstance(data.get("data"), list):
            raise EmbeddingError("Embedding provider returned invalid JSON")
        rows = sorted(data["data"], key=lambda item: int(item.get("index", 0)) if isinstance(item, dict) else 0)
        vectors: list[list[float]] = []
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("embedding"), list):
                raise EmbeddingError("Embedding provider returned an invalid vector")
            vector = [float(v) for v in row["embedding"]]
            if not vector or any(not math.isfinite(v) for v in vector):
                raise EmbeddingError("Embedding provider returned a non-finite vector")
            vectors.append(vector)
        if len(vectors) != expected:
            raise EmbeddingError("Embedding provider returned the wrong number of vectors")
        dims = {len(v) for v in vectors}
        if len(dims) > 1:
            raise EmbeddingError("Embedding provider returned inconsistent vector dimensions")
        return vectors

    async def embed(self, inputs: list[str]) -> EmbeddingBatch:
        if not inputs:
            return EmbeddingBatch(model=self.active_model, vectors=[])
        if len(inputs) > 128:
            raise ValueError("A maximum of 128 embedding inputs is supported per request")
        cleaned = [str(item)[:50_000] for item in inputs]
        if not self.external_enabled:
            return EmbeddingBatch(model=self.active_model, vectors=[self._lite_vector(text) for text in cleaned])

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    f"{self.base_url}/embeddings",
                    json={"model": self.model, "input": cleaned, "encoding_format": "float"},
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            raise EmbeddingError(f"Embedding provider failed: {type(exc).__name__}") from exc
        return EmbeddingBatch(model=self.model, vectors=self._validate_vectors(data, len(cleaned)))
