from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


ReasoningEffort = Literal["fast", "standard", "think", "deep", "max"]


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "pulsar-auto"
    messages: list[Message] = Field(min_length=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=16384)
    stream: bool = False
    reasoning_effort: ReasoningEffort = "standard"
    verify: bool = True
    conversation_id: str | None = Field(default=None, max_length=120)
    enable_tools: bool = False
    enable_web_search: bool = False


class ResponseRequest(BaseModel):
    model: str = "pulsar-auto"
    input: str | list[Message]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=1024, ge=1, le=16384)
    reasoning: ReasoningEffort = "standard"
    verify: bool = True
    conversation_id: str | None = Field(default=None, max_length=120)
    enable_tools: bool = False
    enable_web_search: bool = False

    def as_messages(self) -> list[Message]:
        if isinstance(self.input, str):
            return [Message(role="user", content=self.input)]
        return self.input


class CreateKeyRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    permissions: list[str] = Field(default_factory=lambda: ["chat"])
    daily_limit: int = Field(default=1000, ge=1, le=1_000_000)


class KnowledgeRequest(BaseModel):
    source: str = Field(default="manual", min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=200_000)
    tags: list[str] = Field(default_factory=list)


class ToolCallRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    arguments: dict[str, Any] = Field(default_factory=dict)


class EmbeddingRequest(BaseModel):
    model: str = "pulsar-embed"
    input: str | list[str]

    def inputs(self) -> list[str]:
        return [self.input] if isinstance(self.input, str) else self.input


class ResearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    limit: int = Field(default=5, ge=1, le=10)
    time_range: Literal["day", "month", "year"] | None = None


class SemanticMemoryRequest(BaseModel):
    namespace: str = Field(min_length=1, max_length=120)
    source: str = Field(default="user", min_length=1, max_length=80)
    content: str = Field(min_length=1, max_length=100_000)


class SemanticSearchRequest(BaseModel):
    namespace: str = Field(min_length=1, max_length=120)
    query: str = Field(min_length=1, max_length=20_000)
    limit: int = Field(default=4, ge=1, le=12)
