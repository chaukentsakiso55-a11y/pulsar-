from __future__ import annotations

from pathlib import Path

ENV_PATH = Path('.env')
EXAMPLE_PATH = Path('.env.example')


def read_env() -> dict[str, str]:
    if not ENV_PATH.exists() and EXAMPLE_PATH.exists():
        ENV_PATH.write_text(EXAMPLE_PATH.read_text(encoding='utf-8'), encoding='utf-8')
    values: dict[str, str] = {}
    if ENV_PATH.exists():
        for raw in ENV_PATH.read_text(encoding='utf-8').splitlines():
            line = raw.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            values[key.strip()] = value.strip()
    return values


def update_env(changes: dict[str, str]) -> None:
    lines = ENV_PATH.read_text(encoding='utf-8').splitlines() if ENV_PATH.exists() else []
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        if '=' in line and not line.lstrip().startswith('#'):
            key = line.split('=', 1)[0].strip()
            if key in changes:
                out.append(f'{key}={changes[key]}')
                seen.add(key)
                continue
        out.append(line)
    if out and out[-1].strip():
        out.append('')
    for key, value in changes.items():
        if key not in seen:
            out.append(f'{key}={value}')
    ENV_PATH.write_text('\n'.join(out).rstrip() + '\n', encoding='utf-8')


def prompt(label: str, current: str = '') -> str:
    suffix = f' [{current}]' if current else ''
    value = input(f'{label}{suffix}: ').strip()
    return value if value else current


def main() -> int:
    env = read_env()
    print('\nPulsar AI v1.2 intelligence configuration')
    print('Secrets are written only to local .env, which is ignored by Git.\n')

    search_url = prompt('SearXNG base URL (blank disables web research)', env.get('PULSAR_WEB_SEARCH_URL', ''))
    embed_url = prompt('OpenAI-compatible embeddings base URL (blank = local fallback)', env.get('PULSAR_EMBEDDINGS_BASE_URL', ''))
    embed_model = prompt('Embedding model ID (blank = local fallback)', env.get('PULSAR_EMBEDDINGS_MODEL', ''))

    current_key = env.get('PULSAR_EMBEDDINGS_API_KEY', '')
    key_hint = 'configured' if current_key else 'not configured'
    new_key = input(f'Embedding API key [{key_hint}] (Enter keeps current, type CLEAR to remove): ').strip()
    if new_key.upper() == 'CLEAR':
        embed_key = ''
    elif new_key:
        embed_key = new_key
    else:
        embed_key = current_key

    update_env({
        'PULSAR_WEB_SEARCH_URL': search_url,
        'PULSAR_EMBEDDINGS_BASE_URL': embed_url,
        'PULSAR_EMBEDDINGS_MODEL': embed_model,
        'PULSAR_EMBEDDINGS_API_KEY': embed_key,
        'PULSAR_ENABLE_SEMANTIC_MEMORY': 'true',
    })

    print('\n[OK] Pulsar intelligence settings updated.')
    print('Restart Pulsar for configuration changes to take effect.')
    if not embed_url or not embed_model:
        print('Embeddings: using offline pulsar-embed-lite fallback.')
    else:
        print(f'Embeddings: external OpenAI-compatible model {embed_model}.')
    print('Web research:', 'enabled' if search_url else 'disabled')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
