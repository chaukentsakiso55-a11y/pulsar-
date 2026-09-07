from pathlib import Path

import pytest

from pulsar.db import Database
from pulsar.tools import SafeToolRegistry, calculate


def test_calculator_is_safe_and_correct():
    assert calculate("2 * (3 + 4)").output == "14"
    assert calculate("sqrt(81) + 1").output == "10.0"
    with pytest.raises(ValueError):
        calculate("__import__('os').system('echo nope')")


def test_knowledge_search_tool(tmp_path: Path):
    db = Database(tmp_path / "pulsar.db")
    db.init()
    db.add_knowledge("physics", "Gravity attracts masses and causes objects to accelerate toward Earth.", ["science"])
    registry = SafeToolRegistry(db)
    result = registry.execute("knowledge_search", {"query": "gravity masses", "limit": 3})
    assert "physics" in result.output
    assert "Gravity" in result.output
