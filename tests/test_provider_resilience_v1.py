import asyncio

import httpx

from pulsar.providers.openai_compat import OpenAICompatibleProvider, ProviderRequestError
from pulsar.schemas import Message


class _ScriptedClient:
    responses: list[int] = []
    calls = 0

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def post(self, url, json, headers):
        type(self).calls += 1
        status = type(self).responses.pop(0)
        request = httpx.Request("POST", url)
        if status == 200:
            return httpx.Response(
                200,
                request=request,
                json={"choices": [{"message": {"content": "ok"}}]},
            )
        return httpx.Response(status, request=request, headers={"Retry-After": "0"})


def test_retries_transient_provider_failure(monkeypatch):
    _ScriptedClient.responses = [503, 200]
    _ScriptedClient.calls = 0
    monkeypatch.setattr(httpx, "AsyncClient", _ScriptedClient)
    provider = OpenAICompatibleProvider("http://localhost:8001/v1", "", "local", max_retries=2)
    result = asyncio.run(provider.generate([Message(role="user", content="hello")], 32, 0.2))
    assert result == "ok"
    assert _ScriptedClient.calls == 2


def test_does_not_retry_authentication_failure(monkeypatch):
    _ScriptedClient.responses = [401]
    _ScriptedClient.calls = 0
    monkeypatch.setattr(httpx, "AsyncClient", _ScriptedClient)
    provider = OpenAICompatibleProvider("http://localhost:8001/v1", "bad", "local", max_retries=2)
    try:
        asyncio.run(provider.generate([Message(role="user", content="hello")], 32, 0.2))
    except ProviderRequestError as exc:
        assert "HTTP 401" in str(exc)
    else:
        raise AssertionError("expected ProviderRequestError")
    assert _ScriptedClient.calls == 1
