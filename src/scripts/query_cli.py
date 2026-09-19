"""Terminal client for query_docs -- lets a person ask the knowledge base
questions directly, without going through an MCP client. Calls
server.query_docs() as-is (same cache, rerank/hybrid tiers) rather than
reimplementing retrieval. Run via `make query`.

    python src/scripts/query_cli.py                  # interactive REPL
    python src/scripts/query_cli.py "your question"   # single one-off query

The interactive REPL uses questionary for prompts/styling; the one-off mode
deliberately doesn't import it at all and only ever prints plain text, since
that mode is the one used from scripts/pipes where a TTY isn't guaranteed.
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import server  # noqa: E402


async def ask(query: str, top_k: int, collection: str) -> str:
    return await server.query_docs(query, top_k, collection=collection)


def list_collection_names() -> list:
    """[] on any DB error -- the REPL falls back to the default collection
    silently rather than failing to even start over a transient DB hiccup."""
    try:
        rows = server.run_query(lambda conn: conn.execute(
            "SELECT DISTINCT collection FROM doc_chunks ORDER BY collection"
        ).fetchall())
    except Exception:
        return []
    return [r[0] for r in rows]


def print_result(answer: str, style) -> None:
    """Pretty-prints query_docs()'s plain-text answer: bold/colored source
    headers, a dim rule between chunks, red for the empty/error-message
    cases (recognized by their fixed leading phrases -- see query_docs)."""
    import questionary

    if answer.startswith(("No relevant", "Query failed", "query must", "top_k must",
                           "collection and")):
        questionary.print(answer, style="fg:#ff5f5f")
        return

    for i, block in enumerate(answer.split("\n\n---\n\n")):
        if i > 0:
            questionary.print("─" * 60, style="fg:#585858")
        header, _, body = block.partition("\n")
        questionary.print(header, style="bold fg:#00afff")
        if body:
            print(body)


def run_repl(top_k: int, collection: str, collection_explicit: bool) -> None:
    import questionary
    from questionary import Style

    style = Style([
        ("qmark", "fg:#00d7af bold"),
        ("question", "bold"),
        ("answer", "fg:#00d7af bold"),
        ("pointer", "fg:#00d7af bold"),
        ("highlighted", "fg:#00d7af bold"),
    ])

    if not collection_explicit:
        names = list_collection_names()
        if len(names) > 1:
            picked = questionary.select(
                "Which collection?",
                choices=names,
                default=collection if collection in names else names[0],
                style=style,
            ).ask()
            if picked is None:  # Ctrl-C at the picker -- quit, don't guess
                return
            collection = picked

    questionary.print(f"doc-rag-mcp -- querying collection '{collection}'.", style="bold")
    questionary.print("Empty answer or Ctrl-C/Ctrl-D to quit.\n", style="fg:#888888")

    while True:
        try:
            query = questionary.text("Ask:", qmark="›", style=style).ask()
        except KeyboardInterrupt:
            print()
            break
        if query is None:  # questionary returns None on Ctrl-C/Ctrl-D too
            print()
            break
        query = query.strip()
        if not query:
            break
        answer = asyncio.run(ask(query, top_k, collection))
        print_result(answer, style)
        print()


def main():
    parser = argparse.ArgumentParser(description="Query the doc-rag-mcp knowledge base from the terminal.")
    parser.add_argument("query", nargs="?", default="", help="One-off question. Omit for an interactive prompt.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to return (default 5).")
    parser.add_argument("--collection", default=None,
                         help="Named collection to query (default from RAG_COLLECTION env var, "
                              f"currently '{server.COLLECTION_NAME}'). In the interactive REPL, "
                              "omitting this shows a picker if more than one collection exists.")
    args = parser.parse_args()
    collection_explicit = args.collection is not None
    collection = args.collection or server.COLLECTION_NAME

    if args.query.strip():
        print(asyncio.run(ask(args.query, args.top_k, collection)))
        return

    run_repl(args.top_k, collection, collection_explicit)


if __name__ == "__main__":
    main()
