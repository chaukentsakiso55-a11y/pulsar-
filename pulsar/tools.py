from __future__ import annotations

import ast
import json
import math
import operator
from dataclasses import dataclass
from typing import Any

from pulsar.db import Database
from pulsar.retrieval import KnowledgeRetriever
from pulsar.web_research import WebSearchClient


@dataclass(slots=True)
class ToolResult:
    name: str
    output: str


_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_ALLOWED_NAMES = {"pi": math.pi, "e": math.e, "tau": math.tau}
_ALLOWED_FUNCS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "abs": abs,
    "round": round,
}


def _eval(node: ast.AST) -> float | int:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name) and node.id in _ALLOWED_NAMES:
        return _ALLOWED_NAMES[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(float(right)) > 12:
            raise ValueError("Exponent too large")
        value = _ALLOWED_BINOPS[type(node.op)](left, right)
        if isinstance(value, complex) or not math.isfinite(float(value)):
            raise ValueError("Calculator result is not finite")
        return value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        return _ALLOWED_UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_FUNCS:
        if node.keywords:
            raise ValueError("Keyword arguments are not supported")
        args = [_eval(a) for a in node.args]
        value = _ALLOWED_FUNCS[node.func.id](*args)
        if isinstance(value, complex) or not math.isfinite(float(value)):
            raise ValueError("Calculator result is not finite")
        return value
    raise ValueError("Unsupported calculator expression")


def calculate(expression: str) -> ToolResult:
    if not expression.strip():
        raise ValueError("Expression is required")
    if len(expression) > 256:
        raise ValueError("Expression too long")
    tree = ast.parse(expression, mode="eval")
    value = _eval(tree)
    return ToolResult(name="calculator", output=str(value))


class SafeToolRegistry:
    """Small, auditable server-side tool registry.

    Pulsar deliberately ships only non-destructive tools by default. Optional
    web search is read-only and must be configured explicitly. Shell execution,
    filesystem writes and device control are not enabled here.
    """

    def __init__(self, db: Database, web_search: WebSearchClient | None = None):
        self.retriever = KnowledgeRetriever(db)
        self.web_search = web_search

    def definitions(self) -> list[dict[str, Any]]:
        items = [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Evaluate a bounded arithmetic/scientific expression safely.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                        "required": ["expression"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "knowledge_search",
                    "description": "Search Pulsar's ingested private knowledge base for relevant passages.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "limit": {"type": "integer", "minimum": 1, "maximum": 8},
                        },
                        "required": ["query", "limit"],
                        "additionalProperties": False,
                    },
                },
            },
        ]
        if self.web_search and self.web_search.enabled:
            items.append(
                {
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "description": (
                            "Search the public web through Pulsar's configured SearXNG-compatible search service. "
                            "Returns result titles, URLs and snippets; it does not open result pages."
                        ),
                        "strict": True,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "limit": {"type": "integer", "minimum": 1, "maximum": 10},
                                "time_range": {
                                    "type": "string",
                                    "enum": ["day", "month", "year"],
                                },
                            },
                            "required": ["query"],
                            "additionalProperties": False,
                        },
                    },
                }
            )
        return items

    def names(self) -> list[str]:
        return [d["function"]["name"] for d in self.definitions()]

    async def execute_async(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if name == "web_search":
            if not self.web_search or not self.web_search.enabled:
                raise LookupError("Web search is not configured")
            query = str(arguments.get("query", "")).strip()
            if not query:
                raise ValueError("query is required")
            limit = min(10, max(1, int(arguments.get("limit", 5))))
            time_range = arguments.get("time_range")
            results = await self.web_search.search(query, limit=limit, time_range=time_range)
            return ToolResult(
                name="web_search",
                output=json.dumps([item.as_dict() for item in results], ensure_ascii=False),
            )
        return self.execute(name, arguments)

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if name == "calculator":
            expression = str(arguments.get("expression", ""))
            return calculate(expression)
        if name == "knowledge_search":
            query = str(arguments.get("query", "")).strip()
            if not query:
                raise ValueError("query is required")
            limit = min(8, max(1, int(arguments.get("limit", 4))))
            chunks = self.retriever.search(query, limit=limit)
            payload = [
                {
                    "source": c.get("source", "unknown"),
                    "content": c.get("content", ""),
                    "score": c.get("score"),
                }
                for c in chunks
            ]
            return ToolResult(name="knowledge_search", output=json.dumps(payload, ensure_ascii=False))
        raise LookupError(f"Unknown Pulsar tool: {name}")
