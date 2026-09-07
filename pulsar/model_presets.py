from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class ModelPreset:
    id: str
    display_name: str
    engine: str
    base_url: str
    model: str
    quality: int
    speed: int
    cost: int
    privacy: str
    supports_reasoning: bool
    supports_tools: bool
    hardware_tier: str
    approx_download: str
    notes: str

    def provider_config(self, provider_id: str | None = None) -> dict:
        return {
            "id": provider_id or self.id,
            "type": "openai-compatible",
            "base_url": self.base_url,
            "api_key_env": "",
            "model": self.model,
            "enabled": True,
            "quality": self.quality,
            "speed": self.speed,
            "cost": self.cost,
            "privacy": self.privacy,
            "supports_reasoning": self.supports_reasoning,
            "supports_tools": self.supports_tools,
            "notes": f"Pulsar integrated preset: {self.display_name}. {self.notes}",
        }

    def public_dict(self) -> dict:
        return asdict(self)


_PRESETS: tuple[ModelPreset, ...] = (
    ModelPreset(
        id="ollama-qwen3-1.7b",
        display_name="Qwen3 1.7B (Ollama)",
        engine="ollama",
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3:1.7b",
        quality=68,
        speed=90,
        cost=1,
        privacy="local",
        supports_reasoning=True,
        supports_tools=True,
        hardware_tier="light",
        approx_download="~1.4 GB",
        notes="Small local reasoning/tool model; a good starting point for limited hardware.",
    ),
    ModelPreset(
        id="ollama-llama3.2-3b",
        display_name="Llama 3.2 3B (Ollama)",
        engine="ollama",
        base_url="http://127.0.0.1:11434/v1",
        model="llama3.2:3b",
        quality=66,
        speed=86,
        cost=1,
        privacy="local",
        supports_reasoning=False,
        supports_tools=True,
        hardware_tier="light",
        approx_download="~2.0 GB",
        notes="Compact multilingual assistant with tool-use support.",
    ),
    ModelPreset(
        id="ollama-deepseek-r1-1.5b",
        display_name="DeepSeek-R1 1.5B (Ollama)",
        engine="ollama",
        base_url="http://127.0.0.1:11434/v1",
        model="deepseek-r1:1.5b",
        quality=69,
        speed=76,
        cost=1,
        privacy="local",
        supports_reasoning=True,
        supports_tools=True,
        hardware_tier="light",
        approx_download="~1.1 GB",
        notes="Small distilled reasoning model for math, code, and logic-heavy tasks.",
    ),
    ModelPreset(
        id="ollama-gemma3-1b",
        display_name="Gemma 3 1B (Ollama)",
        engine="ollama",
        base_url="http://127.0.0.1:11434/v1",
        model="gemma3:1b",
        quality=61,
        speed=94,
        cost=1,
        privacy="local",
        supports_reasoning=False,
        supports_tools=False,
        hardware_tier="very-light",
        approx_download="~0.8 GB",
        notes="Very small text model for fast offline/basic assistant tasks.",
    ),
    ModelPreset(
        id="ollama-qwen3-8b",
        display_name="Qwen3 8B (Ollama)",
        engine="ollama",
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3:8b",
        quality=84,
        speed=58,
        cost=2,
        privacy="local",
        supports_reasoning=True,
        supports_tools=True,
        hardware_tier="medium",
        approx_download="~5.2 GB",
        notes="Stronger local Qwen3 preset when the machine has enough RAM/VRAM.",
    ),
    ModelPreset(
        id="vllm-qwen3-8b",
        display_name="Qwen3 8B (vLLM)",
        engine="vllm",
        base_url="http://127.0.0.1:8000/v1",
        model="Qwen/Qwen3-8B",
        quality=85,
        speed=72,
        cost=2,
        privacy="local",
        supports_reasoning=True,
        supports_tools=True,
        hardware_tier="gpu",
        approx_download="depends on model precision/cache",
        notes="GPU-server preset; start vLLM separately with the matching Qwen3 model.",
    ),
)


def model_presets() -> list[ModelPreset]:
    return list(_PRESETS)


def get_model_preset(preset_id: str) -> ModelPreset:
    for preset in _PRESETS:
        if preset.id == preset_id:
            return preset
    raise LookupError(f"Unknown Pulsar model preset: {preset_id}")
