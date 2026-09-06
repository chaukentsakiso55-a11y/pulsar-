from __future__ import annotations

import httpx

from pulsar.providers.base import Provider
from pulsar.schemas import Message


class OpenAICompatibleProvider(Provider):
    def __init__(self, base_url: str, api_key: str, model: str, name: str = "upstream"):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.name = name

    async def generate(
        self,
        messages: list[Message],
        max_tokens: int,
        temperature: float,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": [m.model_dump() for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        async with httpx.AsyncClient(timeout=180) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()
        content = data["choices"][0]["message"]["content"]
        if isinstance(content, str):
            return content
        return str(content)
