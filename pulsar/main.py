from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse

from pulsar.config import Settings
from pulsar.db import Database
from pulsar.documents import MAX_DOCUMENT_BYTES, chunk_document, extract_document_text
from pulsar.embeddings import EmbeddingClient, EmbeddingError
from pulsar.orchestrator import PulsarOrchestrator
from pulsar.semantic import SemanticMemory
from pulsar.router import ModelRouter
from pulsar.schemas import (
    ChatCompletionRequest,
    CreateKeyRequest,
    EmbeddingRequest,
    KnowledgeRequest,
    ResearchRequest,
    ResponseRequest,
    SemanticMemoryRequest,
    SemanticSearchRequest,
    ToolCallRequest,
)
from pulsar.security import hash_api_key, new_api_key, secure_equal
from pulsar.version import __version__
from pulsar.web_research import WebResearchError, WebSearchClient


def approx_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4) if text else 0


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    if settings.environment not in {"development", "test"} and settings.admin_token == "change-me-before-production":
        raise RuntimeError("PULSAR_ADMIN_TOKEN must be changed before production deployment")

    db = Database(settings.db_path)
    db.init()
    router = ModelRouter(settings)
    web_search = WebSearchClient(
        settings.web_search_url,
        timeout_seconds=settings.web_search_timeout,
        safesearch=settings.web_search_safesearch,
    )
    embeddings = EmbeddingClient(
        settings.embeddings_base_url,
        settings.embeddings_api_key,
        settings.embeddings_model,
        timeout_seconds=settings.embeddings_timeout,
        fallback_dimensions=settings.embeddings_fallback_dimensions,
    )
    semantic_memory = SemanticMemory(db, embeddings) if settings.enable_semantic_memory else None
    orchestrator = PulsarOrchestrator(
        router,
        db,
        settings.retrieval_limit,
        settings.max_orchestration_passes,
        web_search=web_search,
        semantic_memory=semantic_memory,
        semantic_memory_limit=settings.semantic_memory_limit,
    )

    app = FastAPI(
        title="Pulsar AI API",
        version=__version__,
        description="Pulsar Max orchestration + web research + embeddings + semantic memory + safe tools + document RAG",
    )
    app.state.settings = settings
    app.state.db = db
    app.state.router = router
    app.state.orchestrator = orchestrator
    app.state.web_search = web_search
    app.state.embeddings = embeddings
    app.state.semantic_memory = semantic_memory

    if settings.cors_list:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_list,
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type", "X-Pulsar-Admin"],
        )

    def require_admin(x_pulsar_admin: str | None = Header(default=None)) -> None:
        if not x_pulsar_admin or not secure_equal(x_pulsar_admin, settings.admin_token):
            raise HTTPException(status_code=401, detail="Invalid admin token")

    def require_key(authorization: str | None = Header(default=None)) -> dict:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing API key")
        raw = authorization[7:].strip()
        record = db.find_key_by_hash(hash_api_key(raw))
        if not record or record.get("revoked_at"):
            raise HTTPException(status_code=401, detail="Invalid or revoked API key")
        if db.usage_today(record["id"]) >= record["daily_limit"]:
            raise HTTPException(status_code=429, detail="Daily request limit reached")
        return record

    def require_permission(key: dict, permission: str) -> None:
        if permission not in key["permissions"] and "*" not in key["permissions"]:
            raise HTTPException(status_code=403, detail=f"API key lacks {permission} permission")

    @app.get("/health")
    def health() -> dict:
        status = router.status()
        return {
            "status": "ok",
            "service": "Pulsar AI",
            "version": __version__,
            "environment": settings.environment,
            "pulsar_1_checkpoint_loaded": router.pulsar1 is not None,
            "pulsar_max_ready": bool(status["ready_providers"] or status["native_checkpoint"]),
            "ready_providers": status["ready_providers"],
            "tools": orchestrator.tools.names(),
            "web_research_ready": web_search.enabled,
            "embedding_model": embeddings.active_model,
            "embedding_external": embeddings.external_enabled,
            "semantic_memory_enabled": semantic_memory is not None,
        }

    @app.get("/", response_class=HTMLResponse)
    @app.get("/console", response_class=HTMLResponse)
    def console() -> str:
        from pathlib import Path
        return Path(__file__).with_name("static").joinpath("console.html").read_text(encoding="utf-8")

    @app.get("/v1/models")
    def models(key: dict = Depends(require_key)) -> dict:
        return {"object": "list", "data": router.models()}

    @app.get("/v1/usage")
    def usage(key: dict = Depends(require_key)) -> dict:
        return {"key_id": key["id"], "today_requests": db.usage_today(key["id"]), "daily_limit": key["daily_limit"]}

    @app.get("/v1/tools")
    def tools(key: dict = Depends(require_key)) -> dict:
        require_permission(key, "tools")
        return {"object": "list", "data": orchestrator.tools.definitions()}

    @app.post("/v1/tools/call")
    async def tool_call(body: ToolCallRequest, key: dict = Depends(require_key)) -> dict:
        require_permission(key, "tools")
        try:
            result = await orchestrator.tools.execute_async(body.name, body.arguments)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, TypeError, ArithmeticError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        db.record_usage(key["id"], "/v1/tools/call", body.name, 0, 0, "pulsar-tool", "tool", 1)
        return {"object": "tool.result", "name": result.name, "output": result.output}

    @app.post("/v1/embeddings")
    async def embeddings_endpoint(body: EmbeddingRequest, key: dict = Depends(require_key)) -> dict:
        require_permission(key, "embeddings")
        inputs = body.inputs()
        if not inputs:
            raise HTTPException(status_code=400, detail="At least one embedding input is required")
        try:
            batch = await embeddings.embed(inputs)
        except (EmbeddingError, ValueError) as exc:
            raise HTTPException(status_code=502 if isinstance(exc, EmbeddingError) else 400, detail=str(exc)) from exc
        prompt_tokens = sum(approx_tokens(text) for text in inputs)
        db.record_usage(
            key["id"], "/v1/embeddings", batch.model, prompt_tokens, 0,
            "embedding-provider" if embeddings.external_enabled else "pulsar-local", "embedding", 1,
        )
        return {
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": vector, "index": index}
                for index, vector in enumerate(batch.vectors)
            ],
            "model": batch.model,
            "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
        }

    @app.post("/v1/research/search")
    async def research_search(body: ResearchRequest, key: dict = Depends(require_key)) -> dict:
        require_permission(key, "research")
        if not web_search.enabled:
            raise HTTPException(status_code=503, detail="Web research is not configured")
        try:
            results = await web_search.search(body.query, limit=body.limit, time_range=body.time_range)
        except (WebResearchError, ValueError) as exc:
            raise HTTPException(status_code=502 if isinstance(exc, WebResearchError) else 400, detail=str(exc)) from exc
        db.record_usage(key["id"], "/v1/research/search", "pulsar-web", approx_tokens(body.query), 0, "web-search", "research", 1)
        return {"object": "search.results", "query": body.query, "data": [item.as_dict() for item in results]}

    @app.post("/v1/memory")
    async def memory_store(body: SemanticMemoryRequest, key: dict = Depends(require_key)) -> dict:
        require_permission(key, "memory")
        if semantic_memory is None:
            raise HTTPException(status_code=503, detail="Semantic memory is disabled")
        try:
            memory_id = await semantic_memory.remember(body.namespace, body.source, body.content)
        except EmbeddingError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        db.record_usage(key["id"], "/v1/memory", embeddings.active_model, approx_tokens(body.content), 0, "semantic-memory", "memory", 1)
        return {"object": "memory", "id": memory_id, "namespace": body.namespace, "stored": bool(memory_id)}

    @app.post("/v1/memory/search")
    async def memory_search(body: SemanticSearchRequest, key: dict = Depends(require_key)) -> dict:
        require_permission(key, "memory")
        if semantic_memory is None:
            raise HTTPException(status_code=503, detail="Semantic memory is disabled")
        try:
            hits = await semantic_memory.search(body.namespace, body.query, limit=body.limit)
        except EmbeddingError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        db.record_usage(key["id"], "/v1/memory/search", embeddings.active_model, approx_tokens(body.query), 0, "semantic-memory", "memory", 1)
        return {
            "object": "memory.search",
            "namespace": body.namespace,
            "data": [
                {"id": hit.id, "source": hit.source, "content": hit.content, "score": hit.score, "created_at": hit.created_at}
                for hit in hits
            ],
        }

    async def execute(
        messages, model, effort, max_tokens, temperature, conversation_id, verify, enable_tools, enable_web_search
    ):
        try:
            return await orchestrator.run(
                messages=messages,
                model=model,
                effort=effort,
                max_tokens=max_tokens,
                temperature=temperature,
                conversation_id=conversation_id,
                verify=verify,
                enable_tools=enable_tools,
                enable_web_search=enable_web_search,
            )
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Pulsar backend failed: {exc}") from exc

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest, key: dict = Depends(require_key)):
        require_permission(key, "chat")
        if body.enable_web_search:
            require_permission(key, "research")

        prompt_text = "\n".join(m.content for m in body.messages)
        prompt_tokens = approx_tokens(prompt_text)
        request_id = f"chatcmpl_{uuid.uuid4().hex}"
        created = int(time.time())
        result = await execute(
            body.messages, body.model, body.reasoning_effort, body.max_tokens,
            body.temperature, body.conversation_id, body.verify, body.enable_tools, body.enable_web_search,
        )
        completion_tokens = approx_tokens(result.text)
        db.record_usage(
            key["id"], "/v1/chat/completions", body.model, prompt_tokens, completion_tokens,
            result.provider_id, result.effort, result.passes,
        )

        if body.stream:
            async def event_stream() -> AsyncIterator[str]:
                first = {"id": request_id, "object": "chat.completion.chunk", "created": created, "model": body.model,
                         "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}
                yield f"data: {json.dumps(first)}\n\n"
                words = result.text.split(" ")
                for i, word in enumerate(words):
                    packet = {"id": request_id, "object": "chat.completion.chunk", "created": created, "model": body.model,
                              "choices": [{"index": 0, "delta": {"content": word + (' ' if i < len(words)-1 else '')}, "finish_reason": None}]}
                    yield f"data: {json.dumps(packet)}\n\n"
                final = {"id": request_id, "object": "chat.completion.chunk", "created": created, "model": body.model,
                         "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                yield f"data: {json.dumps(final)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(event_stream(), media_type="text/event-stream")

        return {
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": body.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": result.text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens},
            "pulsar": {
                "provider": result.provider_id,
                "backend_model": result.backend_model,
                "reasoning_effort": result.effort,
                "passes": result.passes,
                "retrieved_chunks": result.retrieved_chunks,
                "semantic_memories": result.semantic_memories,
                "web_results": result.web_results,
                "tools_used": result.tools_used,
            },
            "system_fingerprint": f"pulsar-ai-v{__version__}",
        }

    @app.post("/v1/responses")
    async def responses(body: ResponseRequest, key: dict = Depends(require_key)) -> dict:
        require_permission(key, "chat")
        if body.enable_web_search:
            require_permission(key, "research")
        messages = body.as_messages()
        prompt_text = "\n".join(m.content for m in messages)
        result = await execute(
            messages, body.model, body.reasoning, body.max_output_tokens,
            body.temperature, body.conversation_id, body.verify, body.enable_tools, body.enable_web_search,
        )
        input_tokens = approx_tokens(prompt_text)
        output_tokens = approx_tokens(result.text)
        db.record_usage(
            key["id"], "/v1/responses", body.model, input_tokens, output_tokens,
            result.provider_id, result.effort, result.passes,
        )
        return {
            "id": f"resp_{uuid.uuid4().hex}",
            "object": "response",
            "created_at": int(time.time()),
            "status": "completed",
            "model": body.model,
            "output": [{"id": f"msg_{uuid.uuid4().hex}", "type": "message", "role": "assistant", "content": [{"type": "output_text", "text": result.text}]}],
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": input_tokens + output_tokens},
            "pulsar": {
                "provider": result.provider_id,
                "backend_model": result.backend_model,
                "reasoning_effort": result.effort,
                "passes": result.passes,
                "retrieved_chunks": result.retrieved_chunks,
                "semantic_memories": result.semantic_memories,
                "web_results": result.web_results,
                "route_score": result.route_score,
                "tools_used": result.tools_used,
            },
        }

    @app.get("/admin/api-keys", dependencies=[Depends(require_admin)])
    def admin_keys() -> dict:
        return {"data": db.list_keys()}

    @app.post("/admin/api-keys", dependencies=[Depends(require_admin)])
    def admin_create_key(body: CreateKeyRequest) -> dict:
        allowed_permissions = {"chat", "models", "usage", "knowledge", "tools", "embeddings", "research", "memory", "*"}
        bad = [p for p in body.permissions if p not in allowed_permissions]
        if bad:
            raise HTTPException(status_code=400, detail=f"Unknown permissions: {bad}")
        raw, record = new_api_key(body.name, body.permissions, body.daily_limit)
        db.insert_key(record)
        return {
            "id": record["id"], "name": record["name"], "key": raw, "prefix": record["prefix"],
            "permissions": record["permissions"], "daily_limit": record["daily_limit"], "created_at": record["created_at"],
            "warning": "Copy this key now. Pulsar stores only its hash and cannot show the raw key again.",
        }

    @app.post("/admin/api-keys/{key_id}/revoke", dependencies=[Depends(require_admin)])
    def admin_revoke_key(key_id: str) -> dict:
        if not db.revoke_key(key_id):
            raise HTTPException(status_code=404, detail="Key not found or already revoked")
        return {"id": key_id, "revoked": True}

    @app.get("/admin/usage", dependencies=[Depends(require_admin)])
    def admin_usage() -> dict:
        return db.usage_summary()

    @app.get("/admin/providers", dependencies=[Depends(require_admin)])
    def admin_providers() -> dict:
        return router.status()

    @app.post("/admin/knowledge", dependencies=[Depends(require_admin)])
    def admin_knowledge(body: KnowledgeRequest) -> dict:
        chunk_id = db.add_knowledge(body.source, body.content, body.tags)
        return {"id": chunk_id, "stored": True, "source": body.source}

    @app.post("/admin/knowledge/file", dependencies=[Depends(require_admin)])
    async def admin_knowledge_file(
        file: UploadFile = File(...),
        tags: str = Form(default=""),
    ) -> dict:
        raw = await file.read(MAX_DOCUMENT_BYTES + 1)
        if len(raw) > MAX_DOCUMENT_BYTES:
            raise HTTPException(status_code=413, detail="Document exceeds the 10 MB ingestion limit")
        filename = file.filename or "document.txt"
        try:
            text = extract_document_text(filename, raw)
            chunks = chunk_document(text)
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except ValueError as exc:
            status = 415 if "Unsupported document type" in str(exc) else 400
            raise HTTPException(status_code=status, detail=str(exc)) from exc
        if not chunks:
            raise HTTPException(status_code=400, detail="Document produced no searchable text")

        parsed_tags = [item.strip() for item in tags.split(",") if item.strip()][:20]
        base_source = filename[:150]
        ids = [
            db.add_knowledge(f"{base_source}#chunk-{index}", chunk, parsed_tags)
            for index, chunk in enumerate(chunks, start=1)
        ]
        return {
            "stored": True,
            "filename": filename,
            "chunks": len(ids),
            "characters": len(text),
            "ids": ids,
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    s = app.state.settings
    uvicorn.run("pulsar.main:app", host=s.host, port=s.port, reload=False)
