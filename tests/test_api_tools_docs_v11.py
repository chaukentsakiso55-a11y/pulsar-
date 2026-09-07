from pathlib import Path

from fastapi.testclient import TestClient

from pulsar.config import Settings
from pulsar.main import create_app


def build_client(tmp_path: Path) -> TestClient:
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
    )
    return TestClient(create_app(settings))


def make_tool_key(client: TestClient) -> str:
    response = client.post(
        "/admin/api-keys",
        headers={"X-Pulsar-Admin": "admin"},
        json={"name": "tool-test", "permissions": ["chat", "tools"], "daily_limit": 20},
    )
    assert response.status_code == 200
    return response.json()["key"]


def test_tool_api_and_document_ingestion(tmp_path):
    client = build_client(tmp_path)
    key = make_tool_key(client)

    listed = client.get("/v1/tools", headers={"Authorization": f"Bearer {key}"})
    assert listed.status_code == 200
    assert {x["function"]["name"] for x in listed.json()["data"]} >= {"calculator", "knowledge_search"}

    calc = client.post(
        "/v1/tools/call",
        headers={"Authorization": f"Bearer {key}"},
        json={"name": "calculator", "arguments": {"expression": "21*2"}},
    )
    assert calc.status_code == 200
    assert calc.json()["output"] == "42"

    upload = client.post(
        "/admin/knowledge/file",
        headers={"X-Pulsar-Admin": "admin"},
        files={"file": ("gravity.md", b"Gravity attracts masses. Earth gravity accelerates falling objects.", "text/markdown")},
        data={"tags": "physics,science"},
    )
    assert upload.status_code == 200
    assert upload.json()["chunks"] >= 1

    search = client.post(
        "/v1/tools/call",
        headers={"Authorization": f"Bearer {key}"},
        json={"name": "knowledge_search", "arguments": {"query": "gravity masses", "limit": 3}},
    )
    assert search.status_code == 200
    assert "gravity.md" in search.json()["output"]
