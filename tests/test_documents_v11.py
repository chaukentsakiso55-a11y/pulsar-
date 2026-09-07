import pytest

from pulsar.documents import chunk_document, extract_document_text


def test_text_document_extraction_and_chunking():
    text = extract_document_text("notes.md", b"# Pulsar\n\n" + b"AI routing and retrieval. " * 300)
    chunks = chunk_document(text, chunk_size=700, overlap=80)
    assert len(chunks) > 1
    assert all(chunks)
    assert "Pulsar" in chunks[0]


def test_rejects_unsupported_document_type():
    with pytest.raises(ValueError, match="Unsupported document type"):
        extract_document_text("archive.exe", b"not a document")
