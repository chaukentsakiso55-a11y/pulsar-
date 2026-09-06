from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from pathlib import Path


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
    upstream_base_url: str = os.getenv("PULSAR_UPSTREAM_BASE_URL", "")
    upstream_api_key: str = os.getenv("PULSAR_UPSTREAM_API_KEY", "")
    upstream_model: str = os.getenv("PULSAR_UPSTREAM_MODEL", "")
    allow_upstream: bool = _bool("PULSAR_ALLOW_UPSTREAM", False)
    cors_origins: str = os.getenv("PULSAR_CORS_ORIGINS", "")

    @property
    def cors_list(self) -> list[str]:
        return [x.strip() for x in self.cors_origins.split(",") if x.strip()]
