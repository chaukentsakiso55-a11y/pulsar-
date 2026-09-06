from __future__ import annotations

import argparse
import json
import os
import platform
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"
ENV_EXAMPLE = ROOT / ".env.example"


def _read_env(path: Path) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8").splitlines()


def _get_value(lines: list[str], key: str) -> str | None:
    prefix = key + "="
    for line in lines:
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    return None


def _set_value(lines: list[str], key: str, value: str) -> list[str]:
    prefix = key + "="
    out: list[str] = []
    replaced = False
    for line in lines:
        if line.startswith(prefix):
            out.append(f"{key}={value}")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        if out and out[-1] != "":
            out.append("")
        out.append(f"{key}={value}")
    return out


def prepare() -> None:
    if sys.version_info < (3, 10):
        raise SystemExit("Pulsar AI requires Python 3.10 or newer.")

    if not ENV_PATH.exists():
        if not ENV_EXAMPLE.exists():
            raise SystemExit(".env.example is missing.")
        ENV_PATH.write_text(ENV_EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
        print("Created .env from .env.example")

    lines = _read_env(ENV_PATH)
    current = _get_value(lines, "PULSAR_ADMIN_TOKEN")
    insecure = {None, "", "replace-with-a-long-random-secret", "change-me-before-production"}
    if current in insecure:
        token = "pulsar_admin_" + secrets.token_urlsafe(48)
        lines = _set_value(lines, "PULSAR_ADMIN_TOKEN", token)
        ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("Generated a strong PULSAR_ADMIN_TOKEN and saved it to .env")
    else:
        print("Existing PULSAR_ADMIN_TOKEN kept unchanged.")

    for directory in ["data", "checkpoints", "logs", "configs"]:
        (ROOT / directory).mkdir(parents=True, exist_ok=True)
    print("Pulsar directories are ready.")


def show_admin() -> None:
    lines = _read_env(ENV_PATH)
    token = _get_value(lines, "PULSAR_ADMIN_TOKEN")
    if not token:
        raise SystemExit("No admin token found. Run setup first.")
    print(token)


def set_checkpoint(value: str) -> None:
    if not ENV_PATH.exists():
        prepare()
    lines = _read_env(ENV_PATH)
    lines = _set_value(lines, "PULSAR_MODEL_CHECKPOINT", value)
    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"PULSAR_MODEL_CHECKPOINT set to {value}")


def _local_provider_config() -> Path:
    lines = _read_env(ENV_PATH)
    value = _get_value(lines, "PULSAR_PROVIDERS_FILE") or "configs/providers.local.json"
    return ROOT / value


def intelligence() -> None:
    config_path = _local_provider_config()
    print("\nPulsar Max Intelligence Status")
    print("=" * 64)
    if not config_path.exists():
        print("[WARN] No powerful provider is configured yet.")
        print("       Use PULSAR.bat option 4 to connect a strong local/cloud model.")
    else:
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
            providers = [p for p in data.get("providers", []) if p.get("enabled", True)]
        except Exception as exc:
            print(f"[FAIL] Provider config is invalid: {exc}")
            providers = []
        if providers:
            for p in sorted(providers, key=lambda x: int(x.get("quality", 50)), reverse=True):
                key_env = p.get("api_key_env", "")
                secret_ready = (not key_env) or bool(os.getenv(key_env)) or bool(_get_value(_read_env(ENV_PATH), key_env))
                print(
                    f"[{'READY' if secret_ready else 'KEY MISSING'}] {p.get('id')} -> {p.get('model')} "
                    f"quality={p.get('quality',50)} speed={p.get('speed',50)} privacy={p.get('privacy','cloud')}"
                )
        else:
            print("[WARN] Provider config contains no enabled providers.")

    lines = _read_env(ENV_PATH)
    checkpoint = _get_value(lines, "PULSAR_MODEL_CHECKPOINT") or ""
    if checkpoint:
        print(f"[{'READY' if (ROOT / checkpoint).exists() else 'MISSING'}] Native Pulsar-1 checkpoint: {checkpoint}")
    else:
        print("[INFO] Native Pulsar-1 checkpoint not configured.")
    print("\nPulsar Max modes: pulsar-fast, pulsar-standard, pulsar-think, pulsar-deep, pulsar-max")
    print("Note: frontier-level quality depends on the quality of the configured backend model(s).")


def doctor() -> None:
    checks: list[tuple[str, bool, str]] = []
    checks.append(("Python >= 3.10", sys.version_info >= (3, 10), platform.python_version()))
    checks.append((".env exists", ENV_PATH.exists(), str(ENV_PATH)))
    checks.append(("data directory", (ROOT / "data").exists(), str(ROOT / "data")))
    checks.append(("checkpoints directory", (ROOT / "checkpoints").exists(), str(ROOT / "checkpoints")))

    if ENV_PATH.exists():
        lines = _read_env(ENV_PATH)
        token = _get_value(lines, "PULSAR_ADMIN_TOKEN")
        checks.append((
            "admin token configured",
            bool(token and token not in {"replace-with-a-long-random-secret", "change-me-before-production"}),
            "configured" if token else "missing",
        ))
        checkpoint = _get_value(lines, "PULSAR_MODEL_CHECKPOINT") or ""
        if checkpoint:
            checks.append(("model checkpoint", (ROOT / checkpoint).exists(), checkpoint))
        else:
            checks.append(("model checkpoint", True, "optional; Pulsar Max can route to configured providers"))
        provider_path = _local_provider_config()
        checks.append(("provider config", True, str(provider_path) if provider_path.exists() else "not configured yet (optional)"))

    failed = False
    print("\nPulsar AI Doctor")
    print("=" * 60)
    for label, ok, detail in checks:
        failed |= not ok
        print(f"[{'OK' if ok else 'FAIL'}] {label}: {detail}")
    print("=" * 60)
    if failed:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pulsar AI Windows setup helper")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("show-admin")
    sub.add_parser("doctor")
    sub.add_parser("intelligence")
    checkpoint = sub.add_parser("set-checkpoint")
    checkpoint.add_argument("path")
    args = parser.parse_args()

    os.chdir(ROOT)
    if args.command == "prepare":
        prepare()
    elif args.command == "show-admin":
        show_admin()
    elif args.command == "doctor":
        doctor()
    elif args.command == "intelligence":
        intelligence()
    elif args.command == "set-checkpoint":
        set_checkpoint(args.path)


if __name__ == "__main__":
    main()
