from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "pulsar-1"
    messages: list[Message] = Field(min_length=1)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    max_tokens: int = Field(default=256, ge=1, le=4096)
    stream: bool = False


class ResponseRequest(BaseModel):
    model: str = "pulsar-1"
    input: str | list[Message]
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=256, ge=1, le=4096)

    def as_messages(self) -> list[Message]:
        if isinstance(self.input, str):
            return [Message(role="user", content=self.input)]
        return self.input


class CreateKeyRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    permissions: list[str] = Field(default_factory=lambda: ["chat"])
    daily_limit: int = Field(default=1000, ge=1, le=1_000_000)
