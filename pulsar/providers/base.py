from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from pulsar.schemas import Message


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class ProviderToolResult:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


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

    async def generate_with_tools(
        self,
        messages: list[Message],
        max_tokens: int,
        temperature: float,
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any] = "auto",
    ) -> ProviderToolResult:
        # Providers that do not implement native function calling remain fully
        # compatible; Pulsar simply receives a normal text response.
        return ProviderToolResult(content=await self.generate(messages, max_tokens, temperature))

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
