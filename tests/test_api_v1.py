from pathlib import Path

from fastapi.testclient import TestClient

from pulsar.config import Settings
from pulsar.main import create_app
from pulsar.version import __version__


def client(tmp_path: Path) -> TestClient:
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


def make_key(c: TestClient) -> str:
    return c.post("/admin/api-keys", headers={"X-Pulsar-Admin":"admin"}, json={"name":"test","permissions":["chat","models"],"daily_limit":10}).json()["key"]


def test_v1_health_and_models(tmp_path):
    c = client(tmp_path)
    assert c.get("/health").json()["version"] == __version__
    key = make_key(c)
    data = c.get("/v1/models", headers={"Authorization":f"Bearer {key}"}).json()["data"]
    ids = {x["id"] for x in data}
    assert "pulsar-max" in ids
    assert "pulsar-fast" in ids


def test_v1_knowledge_admin(tmp_path):
    c = client(tmp_path)
    r = c.post("/admin/knowledge", headers={"X-Pulsar-Admin":"admin"}, json={"source":"test","content":"Pulsar Max knows about routing and verification","tags":["pulsar"]})
    assert r.status_code == 200
    assert r.json()["stored"] is True
