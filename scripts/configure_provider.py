from __future__ import annotations

import getpass
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "providers.local.json"
ENV = ROOT / ".env"


def read_env() -> list[str]:
    return ENV.read_text(encoding="utf-8").splitlines() if ENV.exists() else []


def set_env(key: str, value: str) -> None:
    lines = read_env()
    prefix = key + "="
    out = []
    replaced = False
    for line in lines:
        if line.startswith(prefix):
            out.append(prefix + value)
            replaced = True
        else:
            out.append(line)
    if not replaced:
        if out and out[-1] != "":
            out.append("")
        out.append(prefix + value)
    ENV.write_text("\n".join(out) + "\n", encoding="utf-8")


def load_config() -> dict:
    if CONFIG.exists():
        return json.loads(CONFIG.read_text(encoding="utf-8"))
    return {"providers": []}


def main() -> None:
    print("\nPulsar Max Provider Wizard")
    print("=" * 64)
    print("This connects Pulsar to an OpenAI-compatible model server.")
    print("The API key is stored only in your local .env, which is gitignored.")
    print()

    provider_id = input("Provider ID [primary]: ").strip() or "primary"
    provider_id = re.sub(r"[^a-zA-Z0-9_-]+", "-", provider_id).strip("-") or "primary"
    base_url = input("Base URL (example: http://127.0.0.1:11434/v1): ").strip().rstrip("/")
    if not base_url:
        raise SystemExit("Base URL is required.")
    model = input("Backend model ID: ").strip()
    if not model:
        raise SystemExit("Model ID is required.")

    local = base_url.startswith("http://127.0.0.1") or base_url.startswith("http://localhost")
    api_key_env = ""
    if not local:
        env_suffix = re.sub(r"[^A-Z0-9]+", "_", provider_id.upper())
        api_key_env = f"PULSAR_PROVIDER_{env_suffix}_API_KEY"
        secret = getpass.getpass("API key (hidden): ").strip()
        if not secret:
            raise SystemExit("An API key is required for this remote provider.")
        set_env(api_key_env, secret)
    else:
        maybe = getpass.getpass("API key if your local server requires one [blank for none]: ").strip()
        if maybe:
            env_suffix = re.sub(r"[^A-Z0-9]+", "_", provider_id.upper())
            api_key_env = f"PULSAR_PROVIDER_{env_suffix}_API_KEY"
            set_env(api_key_env, maybe)

    def number(prompt: str, default: int) -> int:
        raw = input(f"{prompt} [{default}]: ").strip()
        return max(0, min(100, int(raw or default)))

    quality = number("Quality score 0-100", 95 if not local else 75)
    speed = number("Speed score 0-100", 65)
    cost = number("Cost score 0-100 (higher = more expensive)", 50 if not local else 5)
    reasoning = (input("Supports strong reasoning? [Y/n]: ").strip().lower() or "y") in {"y", "yes"}

    config = load_config()
    providers = [p for p in config.get("providers", []) if p.get("id") != provider_id]
    providers.append(
        {
            "id": provider_id,
            "type": "openai-compatible",
            "base_url": base_url,
            "api_key_env": api_key_env,
            "model": model,
            "enabled": True,
            "quality": quality,
            "speed": speed,
            "cost": cost,
            "privacy": "local" if local else "cloud",
            "supports_reasoning": reasoning,
            "supports_tools": False,
            "notes": "Configured by Pulsar Max provider wizard",
        }
    )
    CONFIG.write_text(json.dumps({"providers": providers}, indent=2) + "\n", encoding="utf-8")
    set_env("PULSAR_PROVIDERS_FILE", "configs/providers.local.json")
    print(f"\n[OK] Provider '{provider_id}' configured.")
    print(f"[OK] Config: {CONFIG}")
    print("[OK] Restart Pulsar to load the provider.")


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
