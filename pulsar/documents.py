from __future__ import annotations

import io
import json
from pathlib import Path

MAX_DOCUMENT_BYTES = 10 * 1024 * 1024
_TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".csv", ".json", ".jsonl", ".xml", ".html", ".htm",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".kt", ".kts", ".c", ".cpp", ".h",
    ".hpp", ".rs", ".go", ".rb", ".php", ".sql", ".sh", ".bat", ".ps1", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".log",
}


def extract_document_text(filename: str, data: bytes) -> str:
    if len(data) > MAX_DOCUMENT_BYTES:
        raise ValueError("Document exceeds the 10 MB ingestion limit")
    suffix = Path(filename or "document.txt").suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - dependency is packaged in normal installs
            raise RuntimeError("PDF support requires pypdf") from exc
        reader = PdfReader(io.BytesIO(data))
        text = "\n\n".join((page.extract_text() or "").strip() for page in reader.pages).strip()
        if not text:
            raise ValueError("The PDF contains no extractable text")
        return text
    if suffix and suffix not in _TEXT_EXTENSIONS:
        raise ValueError(f"Unsupported document type: {suffix}")
    text = data.decode("utf-8-sig", errors="replace").strip()
    if suffix in {".json", ".jsonl"} and suffix == ".json":
        try:
            text = json.dumps(json.loads(text), ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            pass
    if not text:
        raise ValueError("Document is empty")
    return text


def chunk_document(text: str, *, chunk_size: int = 2200, overlap: int = 240) -> list[str]:
    chunk_size = max(500, min(8000, int(chunk_size)))
    overlap = max(0, min(chunk_size // 3, int(overlap)))
    cleaned = "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").split("\n")).strip()
    if not cleaned:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + chunk_size)
        if end < len(cleaned):
            split_at = max(cleaned.rfind("\n\n", start, end), cleaned.rfind("\n", start, end), cleaned.rfind(" ", start, end))
            if split_at > start + chunk_size // 2:
                end = split_at
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(cleaned):
            break
        start = max(start + 1, end - overlap)
    return chunks
