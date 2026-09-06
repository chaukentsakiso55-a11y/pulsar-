from pathlib import Path

from pulsar.db import Database


def test_knowledge_retrieval(tmp_path: Path):
    db = Database(tmp_path / "p.db")
    db.init()
    db.add_knowledge("physics notes", "Gravity attracts masses and acceleration near Earth is commonly approximated as 9.8 m/s^2", ["science"])
    db.add_knowledge("cooking", "Rice can be cooked with water", ["food"])
    hits = db.search_knowledge("Explain gravity acceleration", limit=2)
    assert hits
    assert hits[0]["source"] == "physics notes"
