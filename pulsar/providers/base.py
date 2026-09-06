from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from pulsar.schemas import Message


class Provider(ABC):
    name: str

    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        max_tokens: int,
        temperature: float,
    ) -> str:
        raise NotImplementedError

    async def stream(
        self,
        messages: list[Message],
        max_tokens: int,
        temperature: float,
    ) -> AsyncIterator[str]:
        text = await self.generate(messages, max_tokens, temperature)
        words = text.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
