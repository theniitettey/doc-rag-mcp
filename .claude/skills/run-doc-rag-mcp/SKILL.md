---
name: run-doc-rag-mcp
description: Build, launch, and drive the doc-rag-mcp server as a real MCP client -- use when asked to run, start, test, or smoke-check doc-rag-mcp, verify a server.py/ingest.py/db.py change actually works, or call its tools (query_docs, list_documents, reindex_docs, cache_stats, usage_stats, clear_cache) end-to-end instead of just reading code. Paths below are relative to the repo root.
---

doc-rag-mcp is an MCP server (Python, `mcp`/FastMCP) over Postgres+pgvector.
It has no GUI -- the way to actually run it is to speak the MCP protocol to
it, which `driver.py` in this skill directory does for real, over either
transport it supports (stdio subprocess, or HTTP against an already-running
server). This is the agent path: prefer it over reading `server.py` and
guessing what a tool call would return.

## Prerequisites

```bash
cp .env.example .env   # then fill in VOYAGE_API_KEY at minimum
make install            # creates .venv, installs requirements.txt
make auth-token         # writes RAG_AUTH_TOKEN into .env (needed for HTTP transport)
```

## Build / start Postgres + ingest docs

```bash
make db                 # starts pgvector via docker compose, localhost:5433
make ingest              # chunks + embeds RAG_DOCS_DIR into Postgres (incremental)
```

## Run (agent path) -- drive it with driver.py

`driver.py` opens a real `mcp.ClientSession`, calls one tool, and prints the
result -- this is what a client (Claude Code, etc.) actually sees.

**stdio** (spawns its own `src/server.py` subprocess, no running server
needed beyond Postgres):

```bash
set -a; source .env; set +a
.venv/bin/python .claude/skills/run-doc-rag-mcp/driver.py stdio list_documents
.venv/bin/python .claude/skills/run-doc-rag-mcp/driver.py stdio query_docs '{"query": "what database does this use", "top_k": 2}'
.venv/bin/python .claude/skills/run-doc-rag-mcp/driver.py stdio cache_stats
.venv/bin/python .claude/skills/run-doc-rag-mcp/driver.py stdio usage_stats
```

Verified output (real run against this repo's indexed docs, filenames
redacted -- shape/format is what matters here):

```
N document(s) indexed, last reindexed <timestamp>.
Docs folder matches the index -- no reindex needed.

<source file 1>
<source file 2>
...
```

**HTTP** (talks to an already-running server -- `make up`, or `docker compose
up -d rag-mcp`; needs `RAG_AUTH_TOKEN` set, see Prerequisites):

```bash
set -a; source .env; set +a
.venv/bin/python .claude/skills/run-doc-rag-mcp/driver.py http list_documents
```

Any of the six tool names work as the second arg:
`query_docs`, `list_documents`, `reindex_docs`, `cache_stats`, `usage_stats`,
`clear_cache`. The third arg (optional) is a JSON object of tool arguments.

## Run (human path)

```bash
make serve        # stdio, for an MCP client config (.mcp.json) to spawn
make serve-http    # HTTP, for a network-reachable/tunneled setup
```

Both just block waiting for a client to connect -- not useful for a quick
smoke check by hand; use the driver above instead.

## Test

No automated test suite in this repo (per `Makefile`/`README.md`) -- the
driver above *is* the smoke check.

## Gotchas

- `mcp.client.streamable_http` exports `streamable_http_client` (underscore
  after `streamable`), not `streamablehttp_client` -- easy typo, fails at
  import time.
- `streamable_http_client(...)` yields a **2-tuple** (`read, write`), not 3 --
  unpacking a third `get_session_id` value raises
  `ValueError: not enough values to unpack`. (Some mcp SDK versions/examples
  show a 3-tuple; check the installed version's signature if this changes.)
- HTTP transport requires the `Authorization: Bearer <RAG_AUTH_TOKEN>` header
  set on the `httpx.AsyncClient` passed in as `http_client=`, not on the URL
  or as a separate kwarg to `streamable_http_client`.
- `query_docs` emits several `MCPDeprecationWarning`s about the logging
  capability (from the `mcp` SDK itself, SEP-2577) on every call -- harmless
  noise, not a bug in this project; the driver examples above pipe them
  through `grep -v Deprecation` to keep output readable.
- If `list_documents` reports every file as "removed" when you know they
  exist, check that `RAG_DOCS_DIR` is actually set in the environment the
  server process sees -- e.g. a `.mcp.json` stdio entry that omits it falls
  back to `./docs` (usually empty). `make gen-mcp-json` regenerates that
  file with the right value from `.env`.
