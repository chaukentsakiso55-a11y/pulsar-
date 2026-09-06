from __future__ import annotations

import hashlib
import hmac
import secrets
import uuid
from datetime import datetime, timezone
from typing import Any


KEY_PREFIX = "pulsar_live_"


def hash_api_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def new_api_key(
    name: str,
    permissions: list[str] | None = None,
    daily_limit: int = 1000,
) -> tuple[str, dict[str, Any]]:
    token = secrets.token_urlsafe(32)
    raw = f"{KEY_PREFIX}{token}"
    record = {
        "id": f"key_{uuid.uuid4().hex[:20]}",
        "name": name,
        "key_hash": hash_api_key(raw),
        "prefix": raw[:20],
        "permissions": permissions or ["chat"],
        "daily_limit": daily_limit,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return raw, record


def secure_equal(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))
