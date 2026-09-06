import json
from pathlib import Path

import pytest

from pulsar.config import Settings
from pulsar.orchestrator import PulsarOrchestrator
from pulsar.providers.base import Provider
from pulsar.providers.openai_compat import ProviderRequestError
from pulsar.router import ModelRouter
from pulsar.schemas import Message


class BrokenProvider(Provider):
    name = "broken"

    async def generate(self, messages, max_tokens, temperature):
        raise ProviderRequestError("temporary outage")


class WorkingProvider(Provider):
    name = "working"

    async def generate(self, messages, max_tokens, temperature):
        return "recovered"


class EmptyDb:
    def get_conversation(self, conversation_id, limit=12):
        return []

    def search_knowledge(self, query, limit=4):
        return []

    def add_message(self, conversation_id, role, content):
        return None


@pytest.mark.asyncio
async def test_alias_fails_over_to_next_ranked_provider(tmp_path: Path):
    cfg = tmp_path / "providers.json"
    cfg.write_text(json.dumps({"providers": [
        {"id":"primary","type":"openai-compatible","base_url":"http://localhost:9/v1","api_key_env":"","model":"primary","quality":99,"speed":80,"cost":20,"supports_reasoning":True},
        {"id":"backup","type":"openai-compatible","base_url":"http://localhost:9/v1","api_key_env":"","model":"backup","quality":90,"speed":90,"cost":10,"supports_reasoning":True}
    ]}), encoding="utf-8")
    settings = Settings(db_path=tmp_path/"x.db", admin_token="x", environment="test", providers_file=cfg, provider_fallback_file=cfg)
    router = ModelRouter(settings)
    router.providers["primary"] = BrokenProvider()
    router.providers["backup"] = WorkingProvider()

    orchestrator = PulsarOrchestrator(router, EmptyDb(), retrieval_limit=0, max_passes=1)
    result = await orchestrator.run(
        [Message(role="user", content="hello")],
        model="pulsar-standard",
        effort="standard",
        max_tokens=50,
        temperature=0.2,
        verify=False,
    )

    assert result.text == "recovered"
    assert router.provider_health["primary"].failures == 1
    assert router.provider_health["primary"].cooldown_until > 0


def test_explicit_provider_does_not_silently_fail_over(tmp_path: Path):
    cfg = tmp_path / "providers.json"
    cfg.write_text(json.dumps({"providers": [
        {"id":"primary","type":"openai-compatible","base_url":"http://localhost:9/v1","api_key_env":"","model":"primary","quality":99,"speed":80,"cost":20,"supports_reasoning":True},
        {"id":"backup","type":"openai-compatible","base_url":"http://localhost:9/v1","api_key_env":"","model":"backup","quality":90,"speed":90,"cost":10,"supports_reasoning":True}
    ]}), encoding="utf-8")
    settings = Settings(db_path=tmp_path/"x.db", admin_token="x", environment="test", providers_file=cfg, provider_fallback_file=cfg)
    router = ModelRouter(settings)
    decision = router.route("provider:primary", "standard")
    assert decision.provider.name == "primary"
