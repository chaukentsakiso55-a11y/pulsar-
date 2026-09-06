from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ProviderSpec:
    id: str
    type: str
    base_url: str
    model: str
    api_key_env: str = ""
    enabled: bool = True
    quality: int = 50
    speed: int = 50
    cost: int = 50
    privacy: str = "cloud"
    supports_reasoning: bool = True
    supports_tools: bool = False
    notes: str = ""

    @property
    def api_key(self) -> str:
        if not self.api_key_env:
            return ""
        return os.getenv(self.api_key_env, "")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProviderSpec":
        return cls(
            id=str(data["id"]),
            type=str(data.get("type", "openai-compatible")),
            base_url=str(data.get("base_url", "")),
            model=str(data.get("model", "")),
            api_key_env=str(data.get("api_key_env", "")),
            enabled=bool(data.get("enabled", True)),
            quality=max(0, min(100, int(data.get("quality", 50)))),
            speed=max(0, min(100, int(data.get("speed", 50)))),
            cost=max(0, min(100, int(data.get("cost", 50)))),
            privacy=str(data.get("privacy", "cloud")),
            supports_reasoning=bool(data.get("supports_reasoning", True)),
            supports_tools=bool(data.get("supports_tools", False)),
            notes=str(data.get("notes", "")),
        )


def load_provider_specs(path: Path) -> list[ProviderSpec]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("providers", []) if isinstance(data, dict) else []
    specs: list[ProviderSpec] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        spec = ProviderSpec.from_dict(item)
        if spec.id in seen:
            raise ValueError(f"Duplicate provider id: {spec.id}")
        seen.add(spec.id)
        specs.append(spec)
    return specs
