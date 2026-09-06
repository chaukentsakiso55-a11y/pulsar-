# Pulsar AI v0.2 ⚡

Pulsar AI is a deployable AI platform consisting of **Pulsar Core**, **Pulsar API**, **Pulsar Console**, and the trainable **Pulsar-1** decoder-only Transformer.

## Fastest Windows setup

On Windows, extract the project and double-click:

```text
PULSAR.bat
```

The Pulsar Control Center automatically:

- finds Python 3
- creates `.venv`
- upgrades pip
- installs Pulsar Core + test dependencies
- creates `.env` from `.env.example`
- generates a strong random `PULSAR_ADMIN_TOKEN`
- prepares data/checkpoint/log folders
- gives you a menu to start/stop the server, open the console, create API keys, run tests, install model support, train checkpoints, or launch Docker

The generated admin token stays in your local `.env`. Use Control Center option **4** when you need to copy it into Pulsar Console.

## What works now

- FastAPI server with OpenAI-style `/v1/chat/completions`
- Responses-style `/v1/responses`
- Server-generated `pulsar_live_...` API keys
- Raw API keys shown once; only SHA-256 hashes are stored in the database
- Per-key permissions, revocation and daily request limits
- SQLite usage tracking
- SSE streaming chat responses
- Model registry and routing
- Pulsar Console at `/console`
- Built-in API test panel in Pulsar Console
- Trainable byte-level Pulsar-1 Transformer in PyTorch
- Optional OpenAI-compatible upstream provider
- Docker deployment
- Windows server process controller + PID tracking
- Pulsar Doctor configuration checks
- Automated tests for API keys, revocation, chat, Responses API and the model forward pass

> **Important:** Pulsar-1 source code is included, but a useful learned model requires training. Until `PULSAR_MODEL_CHECKPOINT` points at a trained checkpoint, the API uses a bootstrap provider that proves the complete server/API pipeline works without pretending random weights are intelligent.

## Windows Control Center

`PULSAR.bat` currently provides:

```text
[1] Start Pulsar locally + open Console
[2] Start Pulsar in server/LAN mode
[3] Create a Pulsar API key
[4] Show admin token
[5] Run Pulsar Doctor
[6] Run automated tests
[7] Install/update Pulsar-1 model support (PyTorch)
[8] Train quick dev checkpoint and activate it
[9] Train Pulsar-1 Tiny checkpoint and activate it
[10] Docker build and launch
[11] Stop Pulsar server
[0] Exit
```

### Local mode vs server mode

**Local mode** binds to `127.0.0.1`, so only the current computer can connect.

**Server/LAN mode** binds to `0.0.0.0`, allowing other devices that can reach the computer to connect on port `8000`. If Pulsar is exposed to the public Internet, place HTTPS and proper firewall/reverse-proxy protection in front of it.

## Manual setup (Windows/Linux/macOS)

Requires Python 3.10+.

```bash
python -m venv .venv
```

Windows:

```bat
.venv\Scripts\activate
pip install -e ".[dev]"
copy .env.example .env
python scripts\windows_setup.py prepare
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
python scripts/windows_setup.py prepare
```

Start the API:

```bash
python -m uvicorn pulsar.main:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000/console
```

## Create an API key

### Control Center

Use option **3** in `PULSAR.bat`.

### Console

Enter the `PULSAR_ADMIN_TOKEN`, name the client, and press **Create key**.

### CLI

```bash
python -m pulsar.cli key create --name "Star AI" --daily-limit 2000
```

Pulsar prints the raw key once:

```text
pulsar_live_<random-secret>
```

Pulsar stores only its hash. If the raw key is lost, revoke it and create another.

## Call the Chat Completions API

```python
import requests

API_KEY = "pulsar_live_YOUR_KEY"

r = requests.post(
    "http://127.0.0.1:8000/v1/chat/completions",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={
        "model": "pulsar-1",
        "messages": [{"role": "user", "content": "Hello Pulsar"}],
        "stream": False,
    },
    timeout=60,
)
print(r.json())
```

## Call the Responses API

```python
import requests

r = requests.post(
    "http://127.0.0.1:8000/v1/responses",
    headers={"Authorization": "Bearer pulsar_live_YOUR_KEY"},
    json={
        "model": "pulsar-1",
        "input": "Explain gravity simply.",
        "max_output_tokens": 256,
    },
    timeout=60,
)
print(r.json())
```

## Train Pulsar-1

The Windows Control Center can install model dependencies and run training for you.

Manual installation:

```bash
pip install -r requirements-model.txt
```

Included configs:

- `configs/pulsar-1-dev.json` — tiny smoke-test model
- `configs/pulsar-1-tiny.json` — approximately 11M parameters
- `configs/pulsar-1-small.json` — approximately 38M parameters

Replace `data/train.txt` with a **large, licensed/authorized training corpus** for meaningful training. The included data is deliberately tiny and is only for checking that the training pipeline works.

Quick smoke test:

```bash
python -m model.train --config configs/pulsar-1-dev.json --data data/train.txt --steps 10 --batch-size 1 --out checkpoints/pulsar-1-dev.pt
python scripts/windows_setup.py set-checkpoint checkpoints/pulsar-1-dev.pt
```

Restart the server to load the checkpoint.

## Docker

Prepare `.env`, then:

```bash
docker compose up -d --build
```

The standard Dockerfile runs Pulsar Core without PyTorch. `Dockerfile.model` installs model support for a trained Pulsar-1 checkpoint.

## Production security checklist

- Keep the admin token private.
- Use HTTPS for any network-accessible deployment.
- Never embed the admin token in Android APKs, browser JavaScript, or public desktop builds.
- Give each application its own restricted API key.
- Revoke leaked API keys immediately.
- Keep `.env`, databases, checkpoints and private logs out of public repositories.
- Use firewall/network rules for a public server.
- For public mobile/web apps, prefer short-lived authenticated user sessions in front of long-lived service API keys.

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Server and model health |
| GET | `/console` | Pulsar Console |
| GET | `/v1/models` | List available models |
| GET | `/v1/usage` | Current key's daily usage |
| POST | `/v1/chat/completions` | Chat completion, streaming or normal |
| POST | `/v1/responses` | Simpler Responses-style generation |
| GET | `/admin/api-keys` | List API-key metadata |
| POST | `/admin/api-keys` | Create a key |
| POST | `/admin/api-keys/{id}/revoke` | Revoke a key |
| GET | `/admin/usage` | Aggregate usage |

## Architecture

```text
Client apps
    │
    ▼
Pulsar API
    │
    ├── API-key authentication
    ├── permissions / limits
    ├── usage tracking
    ▼
Pulsar Core / Model Router
    │
    ├── Pulsar-1 checkpoint
    ├── Bootstrap provider
    └── Optional upstream model
```

## Version

Pulsar AI v0.2.0
