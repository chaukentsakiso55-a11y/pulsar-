from __future__ import annotations

from dataclasses import dataclass, asdict
from urllib.parse import urlparse

import httpx


class WebResearchError(RuntimeError):
    pass


@dataclass(slots=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    engine: str = ""

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


class WebSearchClient:
    """Read-only SearXNG-compatible search client.

    It searches only. It never follows result URLs or downloads page contents.
    """

    def __init__(self, base_url: str = "", *, timeout_seconds: float = 15.0, safesearch: int = 1):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = max(1.0, min(60.0, float(timeout_seconds)))
        self.safesearch = min(2, max(0, int(safesearch)))

    @property
    def enabled(self) -> bool:
        return bool(self.base_url)

    @staticmethod
    def _safe_url(value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return ""
        return value[:2048]

    async def search(self, query: str, *, limit: int = 5, time_range: str | None = None) -> list[SearchResult]:
        query = query.strip()
        if not self.enabled:
            raise WebResearchError("Web research is not configured")
        if not query:
            raise ValueError("query is required")
        if len(query) > 500:
            raise ValueError("query is too long")
        limit = min(10, max(1, int(limit)))
        params: dict[str, str | int] = {
            "q": query,
            "format": "json",
            "safesearch": self.safesearch,
            "pageno": 1,
        }
        if time_range in {"day", "month", "year"}:
            params["time_range"] = time_range
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds, follow_redirects=True) as client:
                response = await client.get(f"{self.base_url}/search", params=params)
                response.raise_for_status()
                data = response.json()
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            raise WebResearchError(f"Web search failed: {type(exc).__name__}") from exc
        if not isinstance(data, dict) or not isinstance(data.get("results", []), list):
            raise WebResearchError("Search provider returned invalid JSON")

        results: list[SearchResult] = []
        for item in data.get("results", []):
            if not isinstance(item, dict):
                continue
            url = self._safe_url(str(item.get("url") or ""))
            if not url:
                continue
            title = str(item.get("title") or url).strip()[:300]
            snippet = str(item.get("content") or item.get("snippet") or "").strip()[:1200]
            engine = str(item.get("engine") or "").strip()[:80]
            results.append(SearchResult(title=title, url=url, snippet=snippet, engine=engine))
            if len(results) >= limit:
                break
        return results
