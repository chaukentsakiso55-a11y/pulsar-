from pathlib import Path

from pulsar.model_presets import get_model_preset, model_presets
from scripts.configure_models import apply_preset


def test_integrated_presets_include_small_and_stronger_local_models():
    ids = {preset.id for preset in model_presets()}
    assert "ollama-qwen3-1.7b" in ids
    assert "ollama-llama3.2-3b" in ids
    assert "ollama-deepseek-r1-1.5b" in ids
    assert "ollama-gemma3-1b" in ids
    assert "ollama-qwen3-8b" in ids
    assert "vllm-qwen3-8b" in ids


def test_presets_never_embed_secrets():
    for preset in model_presets():
        provider = preset.provider_config()
        assert provider["api_key_env"] == ""
        rendered = str(provider).lower()
        assert "sk-" not in rendered
        assert "api_key\": \"" not in rendered


def test_apply_preset_preserves_existing_providers(tmp_path: Path):
    config = tmp_path / "configs" / "providers.local.json"
    env = tmp_path / ".env"
    config.parent.mkdir(parents=True)
    config.write_text('{"providers":[{"id":"existing","enabled":true}]}', encoding="utf-8")

    provider = apply_preset(
        "ollama-qwen3-1.7b",
        config_path=config,
        env_path=env,
        provider_id="qwen-local",
    )
    text = config.read_text(encoding="utf-8")
    assert '"id": "existing"' in text
    assert '"id": "qwen-local"' in text
    assert provider["model"] == "qwen3:1.7b"
    assert "PULSAR_PROVIDERS_FILE=configs/providers.local.json" in env.read_text(encoding="utf-8")


def test_qwen3_presets_advertise_reasoning_and_tools():
    small = get_model_preset("ollama-qwen3-1.7b")
    gpu = get_model_preset("vllm-qwen3-8b")
    assert small.supports_reasoning and small.supports_tools
    assert gpu.supports_reasoning and gpu.supports_tools
