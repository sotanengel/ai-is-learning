"""MCP Server: search_past_experiences, recall_concepts, submit_reflection."""

from __future__ import annotations

from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from kolb_loop.memory.db import EpisodicDB
from kolb_loop.memory.vector_store import VectorStore


def _build_tools() -> list[Tool]:
    return [
        Tool(
            name="search_past_experiences",
            description="Search past inference experiences by keyword or topic",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="recall_concepts",
            description="Retrieve learned concepts relevant to the current context",
            inputSchema={
                "type": "object",
                "properties": {
                    "context": {"type": "string", "description": "Current task context"},
                    "top_k": {"type": "integer", "default": 3},
                },
                "required": ["context"],
            },
        ),
        Tool(
            name="submit_reflection",
            description="Submit a reflection for an experience from an external agent",
            inputSchema={
                "type": "object",
                "properties": {
                    "experience_id": {"type": "string"},
                    "verdict": {
                        "type": "string",
                        "enum": ["success", "partial", "failure"],
                    },
                    "causes": {"type": "array", "items": {"type": "string"}},
                    "improvement_hypotheses": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["experience_id", "verdict"],
            },
        ),
    ]


class KolbLoopMCPServer:
    def __init__(self, db: EpisodicDB, vector_store: VectorStore) -> None:
        self._db = db
        self._vs = vector_store
        self._server = Server("kolb-loop")
        self._register_handlers()

    def _register_handlers(self) -> None:
        @self._server.list_tools()  # type: ignore[no-untyped-call,untyped-decorator]
        async def list_tools() -> list[Tool]:
            return _build_tools()

        @self._server.call_tool()  # type: ignore[untyped-decorator]
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            if name == "search_past_experiences":
                return await self._search_experiences(arguments)
            if name == "recall_concepts":
                return await self._recall_concepts(arguments)
            if name == "submit_reflection":
                return await self._submit_reflection(arguments)
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    async def _search_experiences(self, args: dict[str, Any]) -> list[TextContent]:
        query = str(args.get("query", ""))
        limit = int(args.get("limit", 5))
        experiences = self._db.list_experiences(limit=limit)
        filtered = [e for e in experiences if query.lower() in str(e.request_messages).lower()][
            :limit
        ]
        if not filtered:
            return [TextContent(type="text", text="No matching experiences found.")]
        lines = [
            f"- [{e.id[:8]}] model={e.model} verdict=logged created={e.created_at.date()}"
            for e in filtered
        ]
        return [TextContent(type="text", text="\n".join(lines))]

    async def _recall_concepts(self, args: dict[str, Any]) -> list[TextContent]:
        top_k = int(args.get("top_k", 3))
        concepts = self._db.list_concepts(status="validated")[:top_k]
        if not concepts:
            concepts = self._db.list_concepts()[:top_k]
        if not concepts:
            return [TextContent(type="text", text="No concepts learned yet.")]
        lines = [f"- [{c.id[:8]}] {c.title}: when {c.condition} → {c.action}" for c in concepts]
        return [TextContent(type="text", text="\n".join(lines))]

    async def _submit_reflection(self, args: dict[str, Any]) -> list[TextContent]:
        from kolb_loop.memory.schemas import Reflection, Verdict

        exp_id = str(args["experience_id"])
        exp = self._db.get_experience(exp_id)
        if exp is None:
            return [TextContent(type="text", text=f"Experience {exp_id} not found.")]

        ref = Reflection(
            experience_id=exp_id,
            verdict=Verdict(args["verdict"]),
            causes=list(args.get("causes", [])),
            improvement_hypotheses=list(args.get("improvement_hypotheses", [])),
        )
        self._db.save_reflection(ref)
        return [TextContent(type="text", text=f"Reflection {ref.id} saved.")]

    async def run(self) -> None:
        async with stdio_server() as (read_stream, write_stream):
            await self._server.run(
                read_stream, write_stream, self._server.create_initialization_options()
            )
