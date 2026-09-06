from __future__ import annotations

from pulsar.providers.base import Provider
from pulsar.schemas import Message


class FallbackProvider(Provider):
    name = "pulsar-1-fallback"

    async def generate(
        self,
        messages: list[Message],
        max_tokens: int,
        temperature: float,
    ) -> str:
        user_text = next(
            (m.content for m in reversed(messages) if m.role == "user"),
            "",
        )
        return (
            "Pulsar AI is online and your API pipeline is working. "
            "Pulsar-1 does not have a trained checkpoint loaded yet, so this is the "
            "bootstrap provider rather than a learned-model answer. "
            f"Your last message was: {user_text[:500]}"
        )
