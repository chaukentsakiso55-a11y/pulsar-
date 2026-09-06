from __future__ import annotations

from pathlib import Path

from pulsar.config import Settings
from pulsar.providers.base import Provider
from pulsar.providers.fallback import FallbackProvider
from pulsar.providers.openai_compat import OpenAICompatibleProvider


class ModelRouter:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.fallback = FallbackProvider()
        self.pulsar1: Provider | None = None
        self.upstream: Provider | None = None

        if settings.model_checkpoint and Path(settings.model_checkpoint).exists():
            from pulsar.providers.pulsar1 import Pulsar1Provider

            self.pulsar1 = Pulsar1Provider(settings.model_checkpoint)

        if (
            settings.allow_upstream
            and settings.upstream_base_url
            and settings.upstream_api_key
            and settings.upstream_model
        ):
            self.upstream = OpenAICompatibleProvider(
                settings.upstream_base_url,
                settings.upstream_api_key,
                settings.upstream_model,
            )

    def resolve(self, model: str) -> Provider:
        if model == "pulsar-1":
            return self.pulsar1 or self.fallback
        if model in {"pulsar-1-fallback", "bootstrap"}:
            return self.fallback
        if model.startswith("upstream:"):
            if not self.upstream:
                raise LookupError("No upstream model is configured")
            return self.upstream
        raise LookupError(f"Unknown model: {model}")

    def models(self) -> list[dict]:
        items = [
            {
                "id": "pulsar-1",
                "object": "model",
                "owned_by": "pulsar-ai",
                "status": "ready" if self.pulsar1 else "checkpoint-required",
                "backend": self.pulsar1.name if self.pulsar1 else self.fallback.name,
            },
            {
                "id": "pulsar-1-fallback",
                "object": "model",
                "owned_by": "pulsar-ai",
                "status": "ready",
                "backend": self.fallback.name,
            },
        ]
        if self.upstream:
            items.append(
                {
                    "id": f"upstream:{self.settings.upstream_model}",
                    "object": "model",
                    "owned_by": "external",
                    "status": "ready",
                    "backend": "upstream",
                }
            )
        return items
