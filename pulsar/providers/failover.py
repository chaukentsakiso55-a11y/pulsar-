from __future__ import annotations

import time
from dataclasses import dataclass
from typing import MutableMapping

from pulsar.providers.base import Provider
from pulsar.providers.openai_compat import ProviderRequestError
from pulsar.schemas import Message


@dataclass(slots=True)
class ProviderHealth:
    failures: int = 0
    cooldown_until: float = 0.0
    last_error: str = ""


@dataclass(slots=True)
class FailoverCandidate:
    provider_id: str
    backend_model: str
    provider: Provider
    score: float


class FailoverProvider(Provider):
    """Try ranked providers in order and temporarily cool down unhealthy backends."""

    name = "pulsar-failover"

    def __init__(
        self,
        candidates: list[FailoverCandidate],
        health: MutableMapping[str, ProviderHealth],
        *,
        cooldown_seconds: float = 30.0,
        failure_threshold: int = 1,
    ):
        if not candidates:
            raise ValueError("FailoverProvider requires at least one candidate")
        self.candidates = candidates
        self.health = health
        self.cooldown_seconds = max(1.0, float(cooldown_seconds))
        self.failure_threshold = max(1, int(failure_threshold))
        self.last_provider_id = candidates[0].provider_id
        self.last_backend_model = candidates[0].backend_model
        self.failover_count = 0

    def _ordered_candidates(self) -> list[FailoverCandidate]:
        now = time.monotonic()
        ready = [
            candidate
            for candidate in self.candidates
            if self.health.setdefault(candidate.provider_id, ProviderHealth()).cooldown_until <= now
        ]
        return ready or list(self.candidates)

    async def generate(
        self,
        messages: list[Message],
        max_tokens: int,
        temperature: float,
    ) -> str:
        errors: list[str] = []
        ordered = self._ordered_candidates()

        for index, candidate in enumerate(ordered):
            state = self.health.setdefault(candidate.provider_id, ProviderHealth())
            try:
                text = await candidate.provider.generate(messages, max_tokens, temperature)
            except ProviderRequestError as exc:
                state.failures += 1
                state.last_error = str(exc)
                if state.failures >= self.failure_threshold:
                    state.cooldown_until = time.monotonic() + self.cooldown_seconds
                errors.append(f"{candidate.provider_id}: {exc}")
                if index < len(ordered) - 1:
                    self.failover_count += 1
                continue

            state.failures = 0
            state.cooldown_until = 0.0
            state.last_error = ""
            self.last_provider_id = candidate.provider_id
            self.last_backend_model = candidate.backend_model
            return text

        detail = "; ".join(errors) if errors else "no provider was available"
        raise ProviderRequestError(f"All Pulsar providers failed: {detail}")

    def status(self) -> list[dict]:
        now = time.monotonic()
        return [
            {
                "id": candidate.provider_id,
                "model": candidate.backend_model,
                "score": candidate.score,
                "cooldown_seconds_remaining": round(
                    max(0.0, self.health.setdefault(candidate.provider_id, ProviderHealth()).cooldown_until - now),
                    2,
                ),
                "failures": self.health[candidate.provider_id].failures,
                "last_error": self.health[candidate.provider_id].last_error,
            }
            for candidate in self.candidates
        ]
