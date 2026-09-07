from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class Settings:
    host: str = os.getenv("PULSAR_HOST", "0.0.0.0")
    port: int = int(os.getenv("PULSAR_PORT", "8000"))
    db_path: Path = Path(os.getenv("PULSAR_DB_PATH", "data/pulsar.db"))
    admin_token: str = os.getenv("PULSAR_ADMIN_TOKEN", "change-me-before-production")
    environment: str = os.getenv("PULSAR_ENV", "development")
    model_checkpoint: str = os.getenv("PULSAR_MODEL_CHECKPOINT", "")

    # Pulsar Max provider registry. The committed file never contains secrets;
    # provider configs reference environment-variable names for API keys.
    providers_file: Path = Path(os.getenv("PULSAR_PROVIDERS_FILE", "configs/providers.local.json"))
    provider_fallback_file: Path = Path(os.getenv("PULSAR_PROVIDER_FALLBACK_FILE", "configs/providers.example.json"))
    default_reasoning_effort: str = os.getenv("PULSAR_REASONING_EFFORT", "standard")
    enable_knowledge: bool = _bool("PULSAR_ENABLE_KNOWLEDGE", True)
    enable_memory: bool = _bool("PULSAR_ENABLE_MEMORY", True)
    retrieval_limit: int = int(os.getenv("PULSAR_RETRIEVAL_LIMIT", "4"))
    max_orchestration_passes: int = int(os.getenv("PULSAR_MAX_ORCHESTRATION_PASSES", "5"))

    # Optional read-only public web research through a SearXNG-compatible JSON API.
    web_search_url: str = os.getenv("PULSAR_WEB_SEARCH_URL", "")
    web_search_timeout: float = float(os.getenv("PULSAR_WEB_SEARCH_TIMEOUT", "15"))
    web_search_safesearch: int = int(os.getenv("PULSAR_WEB_SEARCH_SAFESEARCH", "1"))

    # OpenAI-compatible embeddings. When unset, Pulsar uses a deterministic
    # local fallback vectorizer so the API and semantic memory still work offline.
    embeddings_base_url: str = os.getenv("PULSAR_EMBEDDINGS_BASE_URL", "")
    embeddings_api_key: str = os.getenv("PULSAR_EMBEDDINGS_API_KEY", "")
    embeddings_model: str = os.getenv("PULSAR_EMBEDDINGS_MODEL", "")
    embeddings_timeout: float = float(os.getenv("PULSAR_EMBEDDINGS_TIMEOUT", "30"))
    embeddings_fallback_dimensions: int = int(os.getenv("PULSAR_EMBEDDINGS_FALLBACK_DIMENSIONS", "384"))
    enable_semantic_memory: bool = _bool("PULSAR_ENABLE_SEMANTIC_MEMORY", True)
    semantic_memory_limit: int = int(os.getenv("PULSAR_SEMANTIC_MEMORY_LIMIT", "4"))

    # Backward-compatible single upstream configuration from Pulsar v0.2.
    upstream_base_url: str = os.getenv("PULSAR_UPSTREAM_BASE_URL", "")
    upstream_api_key: str = os.getenv("PULSAR_UPSTREAM_API_KEY", "")
    upstream_model: str = os.getenv("PULSAR_UPSTREAM_MODEL", "")
    allow_upstream: bool = _bool("PULSAR_ALLOW_UPSTREAM", False)

    cors_origins: str = os.getenv("PULSAR_CORS_ORIGINS", "")

    @property
    def cors_list(self) -> list[str]:
        return [x.strip() for x in self.cors_origins.split(",") if x.strip()]
