"""Terminal client for query_docs -- lets a person ask the knowledge base
questions directly, without going through an MCP client. Calls
server.query_docs() as-is (same cache, rerank/hybrid tiers) rather than
reimplementing retrieval. Run via `make query`.

    python src/scripts/query_cli.py                  # interactive REPL
    python src/scripts/query_cli.py "your question"   # single one-off query
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import server  # noqa: E402


async def ask(query: str, top_k: int) -> None:
    print(await server.query_docs(query, top_k))


def main():
    parser = argparse.ArgumentParser(description="Query the doc-rag-mcp knowledge base from the terminal.")
    parser.add_argument("query", nargs="?", default="", help="One-off question. Omit for an interactive prompt.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to return (default 5).")
    args = parser.parse_args()

    if args.query.strip():
        asyncio.run(ask(args.query, args.top_k))
        return

    print(f"doc-rag-mcp -- querying collection '{server.COLLECTION_NAME}'. "
          "Empty line or Ctrl-D to quit.\n")
    while True:
        try:
            query = input("> ").strip()
        except EOFError:
            print()
            break
        if not query:
            break
        asyncio.run(ask(query, args.top_k))
        print()


if __name__ == "__main__":
    main()
