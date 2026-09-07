from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import httpx

from pulsar.model_presets import ModelPreset, get_model_preset, model_presets

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "providers.local.json"
ENV = ROOT / ".env"


def _read_env(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


def _set_env(path: Path, key: str, value: str) -> None:
    lines = _read_env(path)
    prefix = key + "="
    out: list[str] = []
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def _load_config(path: Path) -> dict:
    if not path.exists():
        return {"providers": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {"providers": []}


def apply_preset(
    preset_id: str,
    *,
    config_path: Path = CONFIG,
    env_path: Path = ENV,
    provider_id: str | None = None,
) -> dict:
    preset = get_model_preset(preset_id)
    provider = preset.provider_config(provider_id)
    config = _load_config(config_path)
    providers = [p for p in config.get("providers", []) if p.get("id") != provider["id"]]
    providers.append(provider)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps({"providers": providers}, indent=2) + "\n", encoding="utf-8")
    _set_env(env_path, "PULSAR_PROVIDERS_FILE", "configs/providers.local.json")
    return provider


def probe_preset(preset: ModelPreset, timeout: float = 3.0) -> tuple[bool, str]:
    try:
        response = httpx.get(f"{preset.base_url.rstrip('/')}/models", timeout=timeout)
        response.raise_for_status()
        data = response.json()
        models = {
            str(item.get("id"))
            for item in (data.get("data", []) if isinstance(data, dict) else [])
            if isinstance(item, dict)
        }
        if preset.model in models:
            return True, f"{preset.model} is available"
        if models:
            return False, f"server is online but {preset.model} is not installed/served"
        return False, "server is online but returned no model IDs"
    except Exception as exc:
        return False, f"server probe failed: {type(exc).__name__}"


def pull_ollama(preset: ModelPreset) -> None:
    if preset.engine != "ollama":
        raise ValueError("Only Ollama presets can be pulled by this helper")
    executable = shutil.which("ollama")
    if not executable:
        raise RuntimeError("Ollama was not found on PATH")
    subprocess.run([executable, "pull", preset.model], check=True)


def _print_catalog() -> None:
    print("\nPulsar Integrated Model Presets")
    print("=" * 78)
    for index, preset in enumerate(model_presets(), start=1):
        print(
            f"[{index}] {preset.display_name:<30} {preset.hardware_tier:<10} "
            f"download={preset.approx_download}"
        )
        print(f"    {preset.notes}")
    print()


def interactive() -> None:
    _print_catalog()
    presets = model_presets()
    raw = input("Choose a preset number [1]: ").strip() or "1"
    try:
        preset = presets[int(raw) - 1]
    except (ValueError, IndexError):
        raise SystemExit("Invalid preset selection.")

    print(f"\nSelected: {preset.display_name}")
    print(f"Model:    {preset.model}")
    print(f"Server:   {preset.base_url}")
    print(f"Tier:     {preset.hardware_tier}")

    if preset.engine == "ollama":
        choice = input(f"Pull {preset.model} with Ollama now? [y/N]: ").strip().lower() or "n"
        if choice in {"y", "yes"}:
            try:
                pull_ollama(preset)
                print("[OK] Ollama model pull completed.")
            except Exception as exc:
                print(f"[WARN] Could not pull model automatically: {exc}")
                print(f"       You can run manually later: ollama pull {preset.model}")
    elif preset.engine == "vllm":
        print("Start a vLLM server separately before using this preset, for example:")
        print(f"  vllm serve {preset.model} --host 127.0.0.1 --port 8000")

    provider = apply_preset(preset.id)
    print(f"[OK] Added provider '{provider['id']}' to {CONFIG}")
    ok, detail = probe_preset(preset)
    print(f"[{'READY' if ok else 'INFO'}] {detail}")
    print("Restart Pulsar after the model server is ready. Multiple presets can be added; Pulsar Max will rank them automatically.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Configure integrated model presets for Pulsar AI")
    parser.add_argument("--list", action="store_true", help="List integrated model presets")
    parser.add_argument("--preset", help="Apply one preset non-interactively")
    parser.add_argument("--provider-id", help="Override the provider ID")
    parser.add_argument("--pull", action="store_true", help="Pull an Ollama model before applying")
    parser.add_argument("--skip-probe", action="store_true", help="Do not probe the local model server")
    args = parser.parse_args()

    os.chdir(ROOT)
    if args.list:
        _print_catalog()
        return
    if not args.preset:
        interactive()
        return

    preset = get_model_preset(args.preset)
    if args.pull:
        pull_ollama(preset)
    provider = apply_preset(preset.id, provider_id=args.provider_id)
    print(json.dumps(provider, indent=2))
    if not args.skip_probe:
        ok, detail = probe_preset(preset)
        print(f"[{'READY' if ok else 'INFO'}] {detail}")


if __name__ == "__main__":
    main()
