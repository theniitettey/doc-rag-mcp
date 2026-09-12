"""Drives the doc-rag-mcp server as a real MCP client (stdio or HTTP), for
launching/testing the app -- not for production use. See SKILL.md.

    # stdio (spawns its own subprocess, same as Claude Code would)
    .venv/bin/python .claude/skills/run-doc-rag-mcp/driver.py stdio list_documents
    .venv/bin/python .claude/skills/run-doc-rag-mcp/driver.py stdio query_docs '{"query": "what is this project"}'

    # http (talks to an already-running server, e.g. `make up` / docker compose)
    .venv/bin/python .claude/skills/run-doc-rag-mcp/driver.py http list_documents
"""

import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

ROOT = Path(__file__).resolve().parent.parent.parent.parent
SERVER_PY = ROOT / "src" / "server.py"


async def run_stdio(tool: str, args: dict):
    env = {**os.environ}
    params = StdioServerParameters(command=str(ROOT / ".venv" / "bin" / "python"), args=[str(SERVER_PY)], env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool, args)
            return result


async def run_http(tool: str, args: dict):
    import httpx

    port = os.environ.get("RAG_PORT", "8743")
    token = os.environ.get("RAG_AUTH_TOKEN", "")
    url = f"http://localhost:{port}/mcp"
    http_client = httpx.AsyncClient(headers={"Authorization": f"Bearer {token}"})

    async with streamable_http_client(url, http_client=http_client) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool, args)
            return result


def main():
    if len(sys.argv) < 3:
        sys.exit(f"usage: {sys.argv[0]} <stdio|http> <tool_name> [json_args]")
    transport, tool = sys.argv[1], sys.argv[2]
    args = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}

    runner = run_stdio if transport == "stdio" else run_http
    result = asyncio.run(runner(tool, args))
    for item in result.content:
        print(getattr(item, "text", item))


if __name__ == "__main__":
    main()
