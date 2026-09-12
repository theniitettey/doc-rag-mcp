---
name: search-docs
description: Use when the user asks a question that could be answered by this project's indexed knowledge base (the doc-rag-mcp MCP server) -- how to phrase queries, when to search multiple ways, and how to keep the index current. Covers query_docs, list_documents, reindex_docs, and usage_stats.
---

# Searching this project's knowledge base

This project runs a local MCP server (`doc-rag-mcp`) backed by Postgres/pgvector.
Its tools return short, relevant passages instead of you reading whole files --
use them instead of guessing at file contents or asking the user to paste docs in.

## Picking a query strategy

- **Narrow, specific question** ("what port does the server bind to by
  default?") -- one `query_docs` call with the default `top_k` is enough.
  Don't over-fetch.
- **Broad or exploratory question** ("what does this project do", "give me
  an overview of the architecture") -- call `list_documents()` first to see
  what's actually indexed, then issue a couple of targeted `query_docs`
  calls (per suspected topic/file) rather than one broad query and hoping.
- **Cross-cutting question spanning multiple docs** (ownership across
  modules, a concept that shows up in several files under different
  wording) -- issue 2-4 differently-phrased queries (synonyms, related
  terms, the specific names/codes involved) and synthesize across the
  results. A single query's `top_k` window can easily miss a relevant
  chunk that's phrased differently from the question.
- If results look incomplete or you suspect more relevant content exists,
  rephrase or raise `top_k` before concluding the answer isn't in the docs
  -- retrieval is chunk-based and imperfect, not a full-text guarantee.

## Keeping the index current

If the user mentions they added, edited, or removed docs, call
`reindex_docs()` yourself rather than telling them to run a command --
that's exactly what it's for. It's incremental (unchanged files are
skipped, no wasted API calls) and clears the query cache automatically
when it finishes.

If `reindex_docs()` reports refusing to remove a large fraction of
previously-indexed files, don't just retry with `confirm_large_removal=True`
to make the message go away -- that guard exists because the same pattern
already caused real data loss once. Tell the user what it flagged and let
them confirm the docs path is actually right before forcing it through.

## Cost awareness

Every `query_docs` call embeds the query (small, cheap) and, if
`RAG_RERANK_MODEL` is configured, also reranks candidates (meaningfully
more tokens, since it processes full chunk text). Check `usage_stats()` if
the user asks about API costs, or if you're about to run many queries in a
row and want to sanity-check the pattern -- it also shows cache hit rate,
which is often the easier lever (repeated/similar questions should be
cache hits, not fresh API calls).
