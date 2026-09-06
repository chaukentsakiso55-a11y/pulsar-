from pathlib import Path

from fastapi.testclient import TestClient

from pulsar.config import Settings
from pulsar.main import create_app


def build_client(tmp_path: Path) -> TestClient:
    settings = Settings(
        host="127.0.0.1",
        port=8000,
        db_path=tmp_path / "test.db",
        admin_token="test-admin",
        environment="test",
        model_checkpoint="",
        upstream_base_url="",
        upstream_api_key="",
        upstream_model="",
        allow_upstream=False,
        cors_origins="",
    )
    return TestClient(create_app(settings))


def test_health(tmp_path):
    client = build_client(tmp_path)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_create_key_and_chat(tmp_path):
    client = build_client(tmp_path)
    created = client.post(
        "/admin/api-keys",
        headers={"X-Pulsar-Admin": "test-admin"},
        json={"name": "test", "permissions": ["chat"], "daily_limit": 10},
    )
    assert created.status_code == 200
    raw = created.json()["key"]

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {raw}"},
        json={
            "model": "pulsar-1",
            "messages": [{"role": "user", "content": "Hello Pulsar"}],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert "Pulsar AI is online" in data["choices"][0]["message"]["content"]


def test_revoked_key_is_rejected(tmp_path):
    client = build_client(tmp_path)
    created = client.post(
        "/admin/api-keys",
        headers={"X-Pulsar-Admin": "test-admin"},
        json={"name": "test", "permissions": ["chat"], "daily_limit": 10},
    ).json()
    client.post(
        f"/admin/api-keys/{created['id']}/revoke",
        headers={"X-Pulsar-Admin": "test-admin"},
    )
    response = client.get(
        "/v1/models", headers={"Authorization": f"Bearer {created['key']}"}
    )
    assert response.status_code == 401


def test_responses_endpoint(tmp_path):
    client = build_client(tmp_path)
    created = client.post(
        "/admin/api-keys",
        headers={"X-Pulsar-Admin": "test-admin"},
        json={"name": "responses-test", "permissions": ["chat"], "daily_limit": 10},
    ).json()
    response = client.post(
        "/v1/responses",
        headers={"Authorization": f"Bearer {created['key']}"},
        json={"model": "pulsar-1", "input": "Hello through Responses"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "response"
    assert data["status"] == "completed"
    assert "Pulsar AI is online" in data["output"][0]["content"][0]["text"]
