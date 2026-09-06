from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse

from pulsar.config import Settings
from pulsar.db import Database
from pulsar.router import ModelRouter
from pulsar.schemas import ChatCompletionRequest, CreateKeyRequest, ResponseRequest
from pulsar.security import hash_api_key, new_api_key, secure_equal
from pulsar.version import __version__


def approx_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4) if text else 0


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    if settings.environment not in {"development", "test"} and settings.admin_token == "change-me-before-production":
        raise RuntimeError("PULSAR_ADMIN_TOKEN must be changed before production deployment")
    db = Database(settings.db_path)
    db.init()
    router = ModelRouter(settings)

    app = FastAPI(
        title="Pulsar AI API",
        version=__version__,
        description="Pulsar Core + Pulsar-1 API server",
    )
    app.state.settings = settings
    app.state.db = db
    app.state.router = router

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

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "service": "Pulsar AI",
            "version": __version__,
            "environment": settings.environment,
            "pulsar_1_checkpoint_loaded": router.pulsar1 is not None,
        }

    @app.get("/", response_class=HTMLResponse)
    @app.get("/console", response_class=HTMLResponse)
    def console() -> str:
        from pathlib import Path

        return Path(__file__).with_name("static").joinpath("console.html").read_text(
            encoding="utf-8"
        )

    @app.get("/v1/models")
    def models(key: dict = Depends(require_key)) -> dict:
        return {"object": "list", "data": router.models()}

    @app.get("/v1/usage")
    def usage(key: dict = Depends(require_key)) -> dict:
        return {
            "key_id": key["id"],
            "today_requests": db.usage_today(key["id"]),
            "daily_limit": key["daily_limit"],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(
        body: ChatCompletionRequest,
        key: dict = Depends(require_key),
    ):
        if "chat" not in key["permissions"] and "*" not in key["permissions"]:
            raise HTTPException(status_code=403, detail="API key lacks chat permission")
        try:
            provider = router.resolve(body.model)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        prompt_text = "\n".join(m.content for m in body.messages)
        prompt_tokens = approx_tokens(prompt_text)
        request_id = f"chatcmpl_{uuid.uuid4().hex}"
        created = int(time.time())

        if body.stream:
            async def event_stream() -> AsyncIterator[str]:
                completion_parts: list[str] = []
                first = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": body.model,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(first)}\n\n"
                async for chunk in provider.stream(
                    body.messages, body.max_tokens, body.temperature
                ):
                    completion_parts.append(chunk)
                    packet = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": body.model,
                        "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(packet)}\n\n"
                final = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": body.model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(final)}\n\n"
                yield "data: [DONE]\n\n"
                completion = "".join(completion_parts)
                db.record_usage(
                    key["id"],
                    "/v1/chat/completions",
                    body.model,
                    prompt_tokens,
                    approx_tokens(completion),
                )

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        text = await provider.generate(body.messages, body.max_tokens, body.temperature)
        completion_tokens = approx_tokens(text)
        db.record_usage(
            key["id"],
            "/v1/chat/completions",
            body.model,
            prompt_tokens,
            completion_tokens,
        )
        return {
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": body.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "system_fingerprint": f"pulsar-ai-v{__version__}",
        }

    @app.post("/v1/responses")
    async def responses(
        body: ResponseRequest,
        key: dict = Depends(require_key),
    ) -> dict:
        if "chat" not in key["permissions"] and "*" not in key["permissions"]:
            raise HTTPException(status_code=403, detail="API key lacks chat permission")
        try:
            provider = router.resolve(body.model)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        messages = body.as_messages()
        prompt_text = "\n".join(m.content for m in messages)
        text = await provider.generate(messages, body.max_output_tokens, body.temperature)
        input_tokens = approx_tokens(prompt_text)
        output_tokens = approx_tokens(text)
        db.record_usage(
            key["id"],
            "/v1/responses",
            body.model,
            input_tokens,
            output_tokens,
        )
        response_id = f"resp_{uuid.uuid4().hex}"
        return {
            "id": response_id,
            "object": "response",
            "created_at": int(time.time()),
            "status": "completed",
            "model": body.model,
            "output": [
                {
                    "id": f"msg_{uuid.uuid4().hex}",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                }
            ],
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        }

    @app.get("/admin/api-keys", dependencies=[Depends(require_admin)])
    def admin_keys() -> dict:
        return {"data": db.list_keys()}

    @app.post("/admin/api-keys", dependencies=[Depends(require_admin)])
    def admin_create_key(body: CreateKeyRequest) -> dict:
        allowed_permissions = {"chat", "models", "usage", "*"}
        bad = [p for p in body.permissions if p not in allowed_permissions]
        if bad:
            raise HTTPException(status_code=400, detail=f"Unknown permissions: {bad}")
        raw, record = new_api_key(body.name, body.permissions, body.daily_limit)
        db.insert_key(record)
        return {
            "id": record["id"],
            "name": record["name"],
            "key": raw,
            "prefix": record["prefix"],
            "permissions": record["permissions"],
            "daily_limit": record["daily_limit"],
            "created_at": record["created_at"],
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

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    s = app.state.settings
    uvicorn.run("pulsar.main:app", host=s.host, port=s.port, reload=False)
