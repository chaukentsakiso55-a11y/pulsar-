import json
import os
from pathlib import Path

from pulsar.config import Settings
from pulsar.router import ModelRouter


def test_router_prefers_quality_for_max(tmp_path: Path, monkeypatch):
    cfg = tmp_path / "providers.json"
    cfg.write_text(json.dumps({"providers": [
        {"id":"fast","type":"openai-compatible","base_url":"http://localhost:9/v1","api_key_env":"","model":"fast","quality":60,"speed":100,"cost":1,"supports_reasoning":False},
        {"id":"smart","type":"openai-compatible","base_url":"http://localhost:9/v1","api_key_env":"","model":"smart","quality":99,"speed":20,"cost":90,"supports_reasoning":True}
    ]}), encoding="utf-8")
    settings = Settings(db_path=tmp_path/"x.db", admin_token="x", environment="test", providers_file=cfg, provider_fallback_file=cfg)
    router = ModelRouter(settings)
    assert router.route("pulsar-fast", "fast").provider_id == "fast"
    assert router.route("pulsar-max", "max").provider_id == "smart"
