# Pulsar AI v1.0 — Pulsar Max

Pulsar AI is a self-hostable AI gateway and orchestration engine. It exposes one Pulsar API while routing work to a native Pulsar-1 checkpoint, local OpenAI-compatible model servers, or configured cloud model providers.

## What changed in v1.0

- **Pulsar Max orchestration** with `fast`, `standard`, `think`, `deep`, and `max` reasoning modes.
- **Multi-provider routing** using quality/speed/cost/reasoning metadata.
- **Expert synthesis + verification** for difficult requests.
- **Conversation memory** stored in SQLite when a `conversation_id` is supplied.
- **Knowledge retrieval** with an admin ingestion endpoint.
- **OpenAI-compatible API surface** through `/v1/chat/completions` and `/v1/responses`.
- **Secure Pulsar API keys** stored as hashes, with revocation and daily limits.
- **Provider wizard** that keeps provider secrets in local `.env` only.
- **Native Pulsar-1** transformer/training code remains available for research and future native models.

> Important: the orchestration software can make a strong backend more useful, but it cannot turn a tiny untrained model into a frontier model. Pulsar Max's ceiling depends heavily on the strongest model you connect or train.

## Windows quick start

1. Install Python 3.10+.
2. Double-click `PULSAR.bat`.
3. Let it create `.venv`, install Pulsar, and generate a local admin token.
4. Choose **Configure / add a powerful model provider** to connect a local or cloud OpenAI-compatible backend.
5. Choose **Create a Pulsar API key**.
6. Start Pulsar and open `http://127.0.0.1:8000/console`.

The provider wizard stores secrets in `.env`, which is ignored by Git. `configs/providers.local.json` is also ignored.

## Model aliases

| Alias | Goal |
|---|---|
| `pulsar-fast` | Lowest latency / cost-weighted routing |
| `pulsar-standard` | Balanced quality, speed, and cost |
| `pulsar-think` | Planning pass + final answer |
| `pulsar-deep` | Parallel expert drafts + synthesis + verification |
| `pulsar-max` | Highest-quality route + multiple expert and verification passes |
| `pulsar-auto` | Uses the request's reasoning setting |
| `pulsar-1` | Native Pulsar checkpoint (or bootstrap provider if no checkpoint is loaded) |

## Provider configuration

Run `PULSAR.bat` option **4**. The wizard asks for provider ID, OpenAI-compatible base URL, backend model ID, optional API key, quality/speed/cost scores, and reasoning capability. Secrets are stored only in `.env` and never in committed provider configuration.

## Create a Pulsar API key

From `PULSAR.bat`, choose **Create a Pulsar API key**, or run:

```bash
python -m pulsar.cli key create --name "Star AI" --permissions chat,models,usage --daily-limit 2000
```

The raw `pulsar_live_...` key is shown once; only its SHA-256 hash is stored.

## Pulsar Max API example

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/v1/responses",
    headers={"Authorization": "Bearer pulsar_live_YOUR_KEY"},
    json={
        "model": "pulsar-max",
        "input": "Design a reliable AI routing architecture.",
        "reasoning": "max",
        "verify": True,
        "conversation_id": "project-42",
        "max_output_tokens": 1600
    },
)
print(response.json())
```

Pulsar returns high-level routing metadata such as selected provider, backend model, reasoning effort, passes, and retrieved chunks. It does not expose private internal chain-of-thought.

## Endpoints

- `GET /health`
- `GET /console`
- `GET /v1/models`
- `GET /v1/usage`
- `POST /v1/chat/completions`
- `POST /v1/responses`
- `GET /admin/providers`
- `GET /admin/api-keys`
- `POST /admin/api-keys`
- `POST /admin/api-keys/{id}/revoke`
- `GET /admin/usage`
- `POST /admin/knowledge`

## Native Pulsar-1

Pulsar-1 is a real decoder-only Transformer implementation in `model/`. The bundled tiny/dev configurations are for learning and testing, not frontier-grade pretrained checkpoints. Serious native intelligence requires properly licensed datasets, much larger models, GPU infrastructure, post-training, evals, and safety testing.

## Production deployment

For public deployment, use HTTPS behind a reverse proxy, rotate admin/provider secrets, restrict CORS, add firewall rules, use per-client API keys and rate limits, back up the database, and monitor provider costs. Do not expose `PULSAR_ADMIN_TOKEN` to client apps.

Docker:

```bash
docker compose up -d --build
```

## Architecture

```text
Client
  |
  v
Pulsar API + API key auth
  |
  v
Pulsar Max Orchestrator
  |-- conversation memory
  |-- knowledge retrieval
  |-- effort controller
  |-- expert passes
  |-- verification / revision
  v
Model Router
  |-- strong cloud provider(s)
  |-- local OpenAI-compatible model(s)
  |-- native Pulsar-1
  v
Final answer + high-level route metadata
```
