from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi.testclient import TestClient

from pulsar.config import Settings
from pulsar.embeddings import EmbeddingClient
from pulsar.main import create_app
from pulsar.semantic import SemanticMemory
from pulsar.web_research import SearchResult, WebSearchClient


def build_client(tmp_path: Path, *, web_search_url: str = "") -> TestClient:
    settings = Settings(
        host="127.0.0.1",
        port=8000,
        db_path=tmp_path / "test.db",
        admin_token="admin",
        environment="test",
        model_checkpoint="",
        providers_file=tmp_path / "none.json",
        provider_fallback_file=tmp_path / "none2.json",
        upstream_base_url="",
        upstream_api_key="",
        upstream_model="",
        allow_upstream=False,
        cors_origins="",
        web_search_url=web_search_url,
        embeddings_base_url="",
        embeddings_api_key="",
        embeddings_model="",
        enable_semantic_memory=True,
    )
    return TestClient(create_app(settings))


def make_key(client: TestClient, permissions: list[str]) -> str:
    response = client.post(
        "/admin/api-keys",
        headers={"X-Pulsar-Admin": "admin"},
        json={"name": "v12-test", "permissions": permissions, "daily_limit": 100},
    )
    assert response.status_code == 200
    return response.json()["key"]


def test_local_embedding_api_and_semantic_memory(tmp_path):
    client = build_client(tmp_path)
    key = make_key(client, ["embeddings", "memory"])
    auth = {"Authorization": f"Bearer {key}"}

    embedded = client.post(
        "/v1/embeddings",
        headers=auth,
        json={"model": "pulsar-embed", "input": ["gravity planets", "banana bread"]},
    )
    assert embedded.status_code == 200
    body = embedded.json()
    assert body["model"] == "pulsar-embed-lite"
    assert len(body["data"]) == 2
    assert len(body["data"][0]["embedding"]) == 384

    for content in [
        "Gravity pulls objects toward planets and stars.",
        "Banana bread is made with ripe bananas and flour.",
    ]:
        stored = client.post(
            "/v1/memory",
            headers=auth,
            json={"namespace": "science-project", "source": "note", "content": content},
        )
        assert stored.status_code == 200
        assert stored.json()["stored"] is True

    searched = client.post(
        "/v1/memory/search",
        headers=auth,
        json={"namespace": "science-project", "query": "gravity planets", "limit": 2},
    )
    assert searched.status_code == 200
    hits = searched.json()["data"]
    assert hits
    assert "Gravity" in hits[0]["content"]


def test_proactive_web_research_is_opt_in_and_permissioned(tmp_path):
    client = build_client(tmp_path, web_search_url="http://search.local")

    async def fake_search(query: str, *, limit: int = 5, time_range: str | None = None):
        assert query
        return [
            SearchResult(
                title="Pulsar test source",
                url="https://example.com/pulsar",
                snippet="A current public search snippet about Pulsar.",
                engine="test",
            )
        ][:limit]

    client.app.state.web_search.search = fake_search

    chat_only = make_key(client, ["chat"])
    denied = client.post(
        "/v1/responses",
        headers={"Authorization": f"Bearer {chat_only}"},
        json={"model": "pulsar-standard", "input": "What is new?", "enable_web_search": True},
    )
    assert denied.status_code == 403

    key = make_key(client, ["chat", "research"])
    result = client.post(
        "/v1/responses",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": "pulsar-standard",
            "input": "What is new with Pulsar?",
            "enable_web_search": True,
            "conversation_id": "web-test",
        },
    )
    assert result.status_code == 200
    assert result.json()["pulsar"]["web_results"] == 1


def test_research_endpoint_requires_configuration(tmp_path):
    client = build_client(tmp_path)
    key = make_key(client, ["research"])
    response = client.post(
        "/v1/research/search",
        headers={"Authorization": f"Bearer {key}"},
        json={"query": "Pulsar AI", "limit": 3},
    )
    assert response.status_code == 503


def test_embedding_fallback_is_deterministic_and_dimension_safe(tmp_path):
    client = EmbeddingClient(fallback_dimensions=128)
    first = asyncio.run(client.embed(["same text"]))
    second = asyncio.run(client.embed(["same text"]))
    assert first.vectors == second.vectors
    assert len(first.vectors[0]) == 128

    db = build_client(tmp_path).app.state.db
    semantic = SemanticMemory(db, client)
    asyncio.run(semantic.remember("ns", "note", "same text about physics"))
    assert asyncio.run(semantic.search("ns", "physics", limit=1))


def test_web_search_url_filter_blocks_non_http_schemes():
    assert WebSearchClient._safe_url("javascript:alert(1)") == ""
    assert WebSearchClient._safe_url("file:///tmp/secret") == ""
    assert WebSearchClient._safe_url("https://example.com/a") == "https://example.com/a"
