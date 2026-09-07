from __future__ import annotations

import asyncio
import json
from email.utils import parsedate_to_datetime
from time import time
from typing import Any

import httpx

from pulsar.providers.base import Provider, ProviderToolResult, ToolCall
from pulsar.schemas import Message

_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


class ProviderRequestError(RuntimeError):
    """Raised when an OpenAI-compatible provider cannot complete a request."""


class OpenAICompatibleProvider(Provider):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        name: str = "upstream",
        *,
        timeout_seconds: float = 180.0,
        max_retries: int = 2,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.name = name
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.max_retries = max(0, int(max_retries))

    @staticmethod
    def _retry_delay(response: httpx.Response | None, attempt: int) -> float:
        if response is not None:
            value = response.headers.get("Retry-After", "").strip()
            if value:
                try:
                    return min(5.0, max(0.0, float(value)))
                except ValueError:
                    try:
                        retry_at = parsedate_to_datetime(value).timestamp()
                        return min(5.0, max(0.0, retry_at - time()))
                    except (TypeError, ValueError, OverflowError):
                        pass
        return min(2.0, 0.25 * (2**attempt))

    async def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_error: Exception | None = None
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            for attempt in range(self.max_retries + 1):
                response: httpx.Response | None = None
                try:
                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        json=payload,
                        headers=headers,
                    )
                    if response.status_code not in _RETRYABLE_STATUS:
                        response.raise_for_status()
                        data = response.json()
                        if not isinstance(data, dict):
                            raise ProviderRequestError("Provider returned a non-object JSON response")
                        return data
                    last_error = httpx.HTTPStatusError(
                        f"retryable provider status {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                except (httpx.TimeoutException, httpx.NetworkError) as exc:
                    last_error = exc
                except httpx.HTTPStatusError as exc:
                    status = exc.response.status_code
                    raise ProviderRequestError(f"Provider {self.name} rejected the request with HTTP {status}") from exc
                except (ValueError, TypeError) as exc:
                    raise ProviderRequestError(f"Provider {self.name} returned invalid JSON") from exc

                if attempt < self.max_retries:
                    await asyncio.sleep(self._retry_delay(response, attempt))

        detail = type(last_error).__name__ if last_error else "unknown error"
        raise ProviderRequestError(
            f"Provider {self.name} failed after {self.max_retries + 1} attempt(s): {detail}"
        ) from last_error

    @staticmethod
    def _message(data: dict[str, Any]) -> dict[str, Any]:
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderRequestError("Provider returned an invalid chat-completions response") from exc
        if not isinstance(message, dict):
            raise ProviderRequestError("Provider returned an invalid message object")
        return message

    @classmethod
    def _extract_content(cls, data: dict[str, Any]) -> str:
        content = cls._message(data).get("content", "")
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        return str(content)

    @classmethod
    def _extract_tool_result(cls, data: dict[str, Any]) -> ProviderToolResult:
        message = cls._message(data)
        content = message.get("content") or ""
        calls: list[ToolCall] = []
        raw_calls = message.get("tool_calls") or []
        if not isinstance(raw_calls, list):
            raise ProviderRequestError("Provider returned invalid tool_calls")
        for raw in raw_calls:
            if not isinstance(raw, dict):
                continue
            function = raw.get("function") or {}
            if not isinstance(function, dict):
                continue
            name = str(function.get("name") or "").strip()
            if not name:
                continue
            raw_arguments = function.get("arguments", "{}")
            if isinstance(raw_arguments, dict):
                arguments = raw_arguments
            else:
                try:
                    parsed = json.loads(str(raw_arguments or "{}"))
                except json.JSONDecodeError as exc:
                    raise ProviderRequestError(f"Provider returned invalid JSON arguments for tool {name}") from exc
                if not isinstance(parsed, dict):
                    raise ProviderRequestError(f"Tool arguments for {name} must be a JSON object")
                arguments = parsed
            calls.append(ToolCall(id=str(raw.get("id") or f"call_{len(calls)+1}"), name=name, arguments=arguments))
        return ProviderToolResult(content=str(content), tool_calls=calls)

    async def generate(
        self,
        messages: list[Message],
        max_tokens: int,
        temperature: float,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": [m.model_dump(exclude_none=True) for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        return self._extract_content(await self._post_chat(payload))

    async def generate_with_tools(
        self,
        messages: list[Message],
        max_tokens: int,
        temperature: float,
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any] = "auto",
    ) -> ProviderToolResult:
        payload = {
            "model": self.model,
            "messages": [m.model_dump(exclude_none=True) for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
            "tools": tools,
            "tool_choice": tool_choice,
        }
        return self._extract_tool_result(await self._post_chat(payload))
