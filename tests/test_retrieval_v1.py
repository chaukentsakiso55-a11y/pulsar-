from pathlib import Path

from pulsar.db import Database
from pulsar.retrieval import KnowledgeRetriever


def test_bm25_retrieval_ranks_relevant_chunk_first(tmp_path: Path):
    db = Database(tmp_path / "p.db")
    db.init()
    db.add_knowledge(
        "physics notes",
        "Gravity attracts masses and gravitational acceleration near Earth is about 9.8 m/s^2",
        ["science"],
    )
    db.add_knowledge("cooking", "Rice can be cooked with water", ["food"])

    hits = KnowledgeRetriever(db).search("gravity acceleration", limit=2)

    assert hits
    assert hits[0]["source"] == "physics notes"
    assert hits[0].get("retrieval") in {None, "fts5-bm25"}


def test_retriever_indexes_new_chunks_lazily(tmp_path: Path):
    db = Database(tmp_path / "p.db")
    db.init()
    retriever = KnowledgeRetriever(db)

    db.add_knowledge("alpha", "Neural routing baseline", ["ai"])
    assert retriever.search("neural routing", limit=1)[0]["source"] == "alpha"

    db.add_knowledge("beta", "Sparse retrieval with BM25 ranking", ["rag"])
    assert retriever.search("sparse retrieval BM25", limit=2)[0]["source"] == "beta"
