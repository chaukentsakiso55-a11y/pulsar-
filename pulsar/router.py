from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pulsar.config import Settings
from pulsar.provider_config import ProviderSpec, load_provider_specs
from pulsar.providers.base import Provider
from pulsar.providers.fallback import FallbackProvider
from pulsar.providers.openai_compat import OpenAICompatibleProvider


@dataclass(slots=True)
class RouteDecision:
    provider: Provider
    provider_id: str
    backend_model: str
    score: float


class ModelRouter:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.fallback = FallbackProvider()
        self.pulsar1: Provider | None = None
        self.providers: dict[str, Provider] = {}
        self.specs: dict[str, ProviderSpec] = {}

        if settings.model_checkpoint and Path(settings.model_checkpoint).exists():
            from pulsar.providers.pulsar1 import Pulsar1Provider

            self.pulsar1 = Pulsar1Provider(settings.model_checkpoint)

        config_path = settings.providers_file if settings.providers_file.exists() else settings.provider_fallback_file
        for spec in load_provider_specs(config_path):
            if not spec.enabled or spec.type != "openai-compatible" or not spec.base_url or not spec.model:
                continue
            # Local OpenAI-compatible servers can intentionally omit a key.
            if spec.api_key_env and not spec.api_key:
                continue
            provider = OpenAICompatibleProvider(
                spec.base_url,
                spec.api_key,
                spec.model,
                name=spec.id,
            )
            self.providers[spec.id] = provider
            self.specs[spec.id] = spec

        # v0.2 compatibility: translate the legacy single-upstream environment into a provider.
        if (
            settings.allow_upstream
            and settings.upstream_base_url
            and settings.upstream_model
            and "legacy-upstream" not in self.providers
        ):
            p = OpenAICompatibleProvider(
                settings.upstream_base_url,
                settings.upstream_api_key,
                settings.upstream_model,
                name="legacy-upstream",
            )
            self.providers["legacy-upstream"] = p
            self.specs["legacy-upstream"] = ProviderSpec(
                id="legacy-upstream",
                type="openai-compatible",
                base_url=settings.upstream_base_url,
                model=settings.upstream_model,
                quality=85,
                speed=60,
                cost=50,
            )

    def _native(self) -> RouteDecision:
        provider = self.pulsar1 or self.fallback
        return RouteDecision(provider, "pulsar-native", provider.name, 1.0)

    @staticmethod
    def _score(spec: ProviderSpec, effort: str) -> float:
        if effort == "fast":
            return spec.speed * 0.55 + spec.quality * 0.25 + (100 - spec.cost) * 0.20
        if effort == "standard":
            return spec.quality * 0.55 + spec.speed * 0.25 + (100 - spec.cost) * 0.20
        if effort == "think":
            return spec.quality * 0.72 + (15 if spec.supports_reasoning else 0) + spec.speed * 0.08 + (100 - spec.cost) * 0.05
        if effort == "deep":
            return spec.quality * 0.82 + (15 if spec.supports_reasoning else 0) + (100 - spec.cost) * 0.03
        return spec.quality * 0.9 + (15 if spec.supports_reasoning else 0)

    def route(self, model: str, effort: str = "standard") -> RouteDecision:
        if model in {"pulsar-1", "pulsar-native", "pulsar-1-fallback", "bootstrap"}:
            return self._native()

        if model.startswith("provider:"):
            provider_id = model.split(":", 1)[1]
            provider = self.providers.get(provider_id)
            spec = self.specs.get(provider_id)
            if not provider or not spec:
                raise LookupError(f"Provider is not configured or not ready: {provider_id}")
            return RouteDecision(provider, provider_id, spec.model, self._score(spec, effort))

        aliases = {"pulsar-fast", "pulsar-standard", "pulsar-think", "pulsar-deep", "pulsar-max", "pulsar-auto"}
        if model not in aliases:
            raise LookupError(f"Unknown model: {model}")
        if not self.providers:
            return self._native()

        alias_effort = {
            "pulsar-fast": "fast",
            "pulsar-standard": "standard",
            "pulsar-think": "think",
            "pulsar-deep": "deep",
            "pulsar-max": "max",
            "pulsar-auto": effort,
        }[model]
        ranked = sorted(
            self.specs.values(),
            key=lambda spec: self._score(spec, alias_effort),
            reverse=True,
        )
        spec = ranked[0]
        return RouteDecision(self.providers[spec.id], spec.id, spec.model, self._score(spec, alias_effort))

    def resolve(self, model: str) -> Provider:
        return self.route(model).provider

    def models(self) -> list[dict]:
        items = [
            {
                "id": alias,
                "object": "model",
                "owned_by": "pulsar-ai",
                "status": "ready" if self.providers or self.pulsar1 else "bootstrap",
                "backend": "auto-router",
            }
            for alias in ["pulsar-fast", "pulsar-standard", "pulsar-think", "pulsar-deep", "pulsar-max", "pulsar-auto"]
        ]
        items.append(
            {
                "id": "pulsar-1",
                "object": "model",
                "owned_by": "pulsar-ai",
                "status": "ready" if self.pulsar1 else "checkpoint-required",
                "backend": self.pulsar1.name if self.pulsar1 else self.fallback.name,
            }
        )
        for spec in sorted(self.specs.values(), key=lambda s: s.quality, reverse=True):
            items.append(
                {
                    "id": f"provider:{spec.id}",
                    "object": "model",
                    "owned_by": "external-or-local",
                    "status": "ready",
                    "backend": spec.model,
                    "quality": spec.quality,
                    "speed": spec.speed,
                    "privacy": spec.privacy,
                    "reasoning": spec.supports_reasoning,
                }
            )
        return items

    def status(self) -> dict:
        return {
            "native_checkpoint": self.pulsar1 is not None,
            "ready_providers": len(self.providers),
            "providers": [
                {
                    "id": spec.id,
                    "model": spec.model,
                    "quality": spec.quality,
                    "speed": spec.speed,
                    "privacy": spec.privacy,
                    "reasoning": spec.supports_reasoning,
                }
                for spec in sorted(self.specs.values(), key=lambda s: s.quality, reverse=True)
            ],
        }
