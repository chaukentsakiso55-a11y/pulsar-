# Pulsar AI v1.2 — Pulsar Max

Pulsar AI is a self-hostable AI gateway and orchestration engine. It exposes one API while routing work to native Pulsar-1 checkpoints, local OpenAI-compatible model servers, or configured cloud model providers.

## Current capabilities

- `pulsar-fast`, `pulsar-standard`, `pulsar-think`, `pulsar-deep`, and `pulsar-max` reasoning modes
- quality/speed/cost-aware multi-provider routing
- health-aware provider failover and retry handling
- expert synthesis and verification passes
- conversation memory in SQLite
- FTS5/BM25 document and knowledge retrieval
- **semantic conversation memory** with embeddings
- **OpenAI-compatible `/v1/embeddings` endpoint**
- **opt-in public web research through SearXNG-compatible search**
- safe server-side tools: calculator, private knowledge search, and optional web search
- PDF/text/code/CSV/JSON document ingestion into RAG
- secure Pulsar API keys with permissions, revocation and daily limits
- Docker/server deployment and a Windows control center
- native Pulsar-1 transformer/training code for research

> Pulsar Max can orchestrate a strong backend, but orchestration cannot turn a tiny untrained native checkpoint into a frontier model. Native model quality still depends on model size, data, compute, post-training and evaluation.

## Windows quick start

1. Install Python 3.10+.
2. Double-click `PULSAR.bat`.
3. Let it create `.venv`, install Pulsar, and generate a local admin token.
4. Configure a strong model provider with option **4**.
5. Optionally configure web research + embeddings with option **14**.
6. Create a Pulsar API key.
7. Start Pulsar and open `http://127.0.0.1:8000/console`.

Secrets are stored in local `.env`. `configs/providers.local.json`, `.env`, databases and checkpoints are Git-ignored.

## Model aliases

| Alias | Goal |
|---|---|
| `pulsar-fast` | lowest latency / cost-weighted routing |
| `pulsar-standard` | balanced quality, speed and cost |
| `pulsar-think` | planning pass + final answer |
| `pulsar-deep` | parallel expert drafts + synthesis + verification |
| `pulsar-max` | highest-quality route + multiple expert/verification passes |
| `pulsar-auto` | uses request reasoning setting |
| `pulsar-1` | native Pulsar checkpoint or bootstrap fallback |

## Pulsar Max request

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/v1/responses",
    headers={"Authorization": "Bearer pulsar_live_YOUR_KEY"},
    json={
        "model": "pulsar-max",
        "input": "Research this topic and explain the strongest answer.",
        "reasoning": "max",
        "verify": True,
        "enable_tools": True,
        "enable_web_search": True,
        "conversation_id": "project-42",
        "max_output_tokens": 1600
    },
)
print(response.json())
```

`enable_web_search` is explicit because sending a prompt to a search service has different privacy implications from local-only inference.

## Embeddings

Pulsar exposes:

```text
POST /v1/embeddings
```

When `PULSAR_EMBEDDINGS_BASE_URL` and `PULSAR_EMBEDDINGS_MODEL` are configured, requests are sent to that OpenAI-compatible embedding server. Otherwise Pulsar uses the offline `pulsar-embed-lite` fallback.

## Semantic memory

With a `conversation_id`, Pulsar can combine:

- recent chronological messages;
- semantically related older messages;
- private knowledge/RAG results;
- optional current web-search snippets.

Direct memory endpoints are also available:

```text
POST /v1/memory
POST /v1/memory/search
```

## Web research

Configure a SearXNG-compatible instance:

```env
PULSAR_WEB_SEARCH_URL=http://127.0.0.1:8080
```

Then use:

```text
POST /v1/research/search
```

or set `enable_web_search: true` on Chat/Responses requests. Pulsar only consumes search result metadata/snippets; it does not automatically browse arbitrary result pages.

## Main endpoints

```text
GET  /health
GET  /console
GET  /v1/models
GET  /v1/usage
GET  /v1/tools
POST /v1/tools/call
POST /v1/chat/completions
POST /v1/responses
POST /v1/embeddings
POST /v1/research/search
POST /v1/memory
POST /v1/memory/search
GET  /admin/providers
GET  /admin/api-keys
POST /admin/api-keys
POST /admin/api-keys/{id}/revoke
GET  /admin/usage
POST /admin/knowledge
POST /admin/knowledge/file
```

## Security

For public deployment, use HTTPS behind a reverse proxy, rotate admin/provider secrets, restrict CORS and firewall access, rate-limit clients, back up the database, and monitor provider costs. Do not expose `PULSAR_ADMIN_TOKEN` to client apps.

Pulsar's built-in tool registry does **not** enable arbitrary shell commands, device control, arbitrary filesystem writes, or automatic browsing of result pages.

## Tests

```bash
python -m pytest -q
```

v1.2 adds regression coverage for embeddings, semantic memory, web-research permission boundaries and URL filtering in addition to the existing API/router/tool/RAG/model tests.

See `docs/V1.1.md` and `docs/V1.2.md` for release-specific details.
