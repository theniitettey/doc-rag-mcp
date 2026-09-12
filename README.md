# doc-rag-mcp

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Turn any folder of docs (PDFs, markdown, text) into a local, queryable
knowledge base exposed as an [MCP](https://modelcontextprotocol.io) server
— so any MCP-compatible agent (Claude Code, or otherwise) can search your
docs for relevant passages instead of reading whole files into its context
window. There's a [dedicated section](#claude-code) below for Claude Code
specifics, since that's what this was built and tested against, but nothing
about the server itself is Claude-specific.

Embeddings run via the Voyage AI API (requires a `VOYAGE_API_KEY`, no
multi-GB local model download); storage is Postgres with the `pgvector`
extension. Runs entirely on your own machine (or wherever you deploy it) —
your documents and queries never go anywhere except to the embedding API.

## How it works

**Ingestion** — point it at a folder (or a single file), it chunks and
embeds anything new or changed, and skips the rest:

```mermaid
flowchart TD
    A[docs folder] --> B[chunk text]
    B --> C{content hash<br/>changed?}
    C -- no --> D[skip, no API call]
    C -- yes --> E[embed via Voyage AI]
    E --> F[(Postgres + pgvector)]
```

**Querying** — the agent calls the MCP tool, which embeds the question and
finds the closest chunks by cosine similarity:

```mermaid
flowchart TD
    A[MCP client] -- query_docs --> B[MCP server]
    B --> C{cached?}
    C -- yes --> G[return cached answer]
    C -- no --> D[embed query via Voyage AI]
    D --> E[(Postgres + pgvector<br/>cosine search)]
    E --> F[top-k matching chunks]
    F --> A
```

Both flows share one Postgres table (`doc_chunks`), namespaced by a
`collection` column so multiple doc sets can coexist in the same database.

## 1. Install

```bash
cd doc-rag-mcp
cp .env.example .env   # fill in VOYAGE_API_KEY and POSTGRES_PASSWORD at least
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Or just run `make install` (also used automatically by the other `make`
targets — see `Makefile` for the full list: `ingest`, `ingest-rebuild`,
`serve`, `serve-http` run things locally in a venv against Postgres; `db`
starts just Postgres in Docker for that local mode; `build`, `up`, `down`,
`restart` (same image, just restarts the process), `redeploy` (rebuild +
recreate — use this after changing `src/*.py`, `Dockerfile`, or
`docker-compose.yaml`), `reindex`, `reindex-rebuild`, `logs`, `clear-cache`
drive the fully containerized setup in `docker-compose.yaml` instead;
`db-reset` wipes all
indexed data).

## 2. Add your docs

Two ways to point at your docs, via `RAG_DOCS_DIR` in `.env`:

- **Drop them in `docs/`** (the default, subfolders are fine) — architecture
  docs, runbooks, research papers, whatever you've got:

  ```text
  docs/
    architecture-overview.md
    api-design.pdf
    adr/
      0001-use-postgres.md
  ```

- **Or point at any path already on your machine** — a folder or a single
  file, anywhere: `RAG_DOCS_DIR=/Users/you/Documents/some-project-docs`. No
  need to copy anything into this repo. (Docker only sees what's mounted
  into `docs/`, so this option is for running `ingest.py`/`make ingest`
  locally rather than via `make reindex`.)

Supported: `.pdf`, `.md`, `.markdown`, `.txt`.

## 3. Build the index

Needs Postgres reachable first — `make db` starts one in Docker (exposed on
`localhost:5433`) if you're running `ingest`/`serve` locally rather than
fully in Docker.

```bash
make db
python src/ingest.py --docs ./docs --collection project_docs
# or: make ingest
```

Re-run this any time the docs change — it's incremental: unchanged files are
skipped by content hash (no re-embedding, no API cost), changed/new files
are re-embedded, and files removed from the folder have their chunks
dropped. Pass `--rebuild` (or `make ingest-rebuild`) to force a full
re-embed of everything (this also happens automatically if
`RAG_EMBED_MODEL`, `RAG_CHUNK_SIZE`, or `RAG_CHUNK_OVERLAP` changed since
the last run).

## 4. Connect it to your MCP client

Any MCP client connects one of two ways — both are covered in more detail
in section 5 for the HTTP case:

- **stdio** (the common case for a local client): the client spawns
  `src/server.py` itself as a subprocess, using your venv's python binary
  so it picks up the installed packages:

  ```text
  /full/path/to/doc-rag-mcp/.venv/bin/python /full/path/to/doc-rag-mcp/src/server.py
  ```

  Most clients want that as a `command` + `args` pair in their own config
  format — see your client's docs for exactly where that goes (for Claude
  Code specifically, see the [dedicated section](#claude-code) below).

- **HTTP**: the client connects to a URL instead of spawning anything —
  see section 5 ("Run it on a network / behind a tunnel") for starting the
  server this way and the auth token it requires.

Once connected, it exposes six tools:

- `query_docs(query, top_k=5)` — semantic search over the indexed chunks,
  returns the matching passages with their source file and a relevance score
- `list_documents()` — lists every file currently indexed
- `reindex_docs(force_rebuild=False, confirm_large_removal=False)` —
  re-scan `RAG_DOCS_DIR` and embed anything new or changed, without leaving
  the chat to run `ingest.py` by hand. Incremental by default; blocks until
  done (can take a while for a large/changed doc set); clears the query
  cache automatically afterward. Refuses to remove a large fraction of
  previously-indexed files in one go unless `confirm_large_removal=True` —
  that pattern is much more often a misconfigured docs path than genuine
  bulk deletion, and the response explains what was left untouched if it
  trips.
- `cache_stats()` — hit/miss counts, hit rate, and current cache size
- `usage_stats()` — cumulative Voyage AI token usage (by model/operation)
  plus cache stats, combined. This is a local tally of calls made through
  this server/`ingest.py`, not your Voyage account's authoritative usage —
  check the [dashboard](https://dashboard.voyageai.com) for that.
- `clear_cache()` — drop the query cache (run after re-running `src/ingest.py`
  directly; `reindex_docs` does this for you)

Then just ask your agent things like "check the docs for how auth is
handled" and it'll call `query_docs` on its own.

### Caching

Query results are cached in-process (LRU, TTL-based) keyed by the normalized
`(query, top_k)` pair. A repeated question skips both the embedding call and
the Postgres round-trip. Tune it with env vars:

```json
"env": {
  "RAG_CACHE_MAX_SIZE": "256",
  "RAG_CACHE_TTL_SECONDS": "600"
}
```

The cache lives in the running server process's memory, so it resets
whenever the client restarts the server, and it will **not** know if you
re-run `src/ingest.py` while it's running — call `clear_cache()` afterward
so it doesn't keep serving answers from the old index (`reindex_docs` does
this automatically).

### Streaming

MCP tool calls return one final result — there's no token-by-token streaming
of the response body the way a chat completion streams. What the protocol
*does* support is progress notifications sent while a tool is still
running; a client that surfaces those (Claude Code does) shows activity
immediately rather than waiting on the final payload. `query_docs` reports
progress as each matching chunk is resolved (and logs cache hit/miss), so
you see activity right away instead of blocking until the whole joined
string is ready — useful mainly on a cache miss with a larger `top_k`, or
on the first call while the embedding model is loading.

## 5. Run it on a network / behind a tunnel

By default the server talks stdio and only exists as a subprocess of a local
MCP client. To let a *remote* agent reach it, run it as a standalone HTTP
server instead:

```bash
make auth-token   # generates a token and writes RAG_AUTH_TOKEN into .env
make serve-http    # or, for the Docker Compose setup: make up
```

`make auth-token` creates `.env` (from `.env.example`) if it doesn't exist
yet, and overwrites `RAG_AUTH_TOKEN` in place if it does — safe to re-run
any time you want to rotate it. Clients need that token. The server refuses
to start over `--transport http` without `RAG_AUTH_TOKEN` set, and every
request must send `Authorization: Bearer <token>`; requests without it get
a 401. Don't run this without a token — whatever's in `docs/` becomes
readable to anyone with the URL and token.

Keep `--host 127.0.0.1` (the default) unless you specifically want the port
open on your LAN — tunnel tools below connect *out* from your machine, so
the server itself never needs to bind `0.0.0.0`.

### Exposing it — pick one

The port below is whatever `RAG_PORT` is set to (default `8743` — deliberately
not 8000/5000/3000, since those are usually already taken by something else).

**Cloudflare Tunnel** (no account needed for a quick tunnel):
```bash
cloudflared tunnel --url http://localhost:8743
```
Gives you a `https://random-words.trycloudflare.com` URL. Full MCP endpoint
is `https://random-words.trycloudflare.com/mcp`.

**ngrok**:
```bash
ngrok http 8743
```
Endpoint: `https://<subdomain>.ngrok-free.app/mcp`.

**Tailscale Funnel** (if the querying agent is on your tailnet, use `serve`
instead of `funnel` to skip the public-internet exposure entirely):
```bash
tailscale funnel 8743     # public
tailscale serve 8743      # tailnet-only, no auth token strictly needed
```

**Plain SSH reverse tunnel** (if you have a VPS):
```bash
ssh -R 8743:localhost:8743 user@your-vps
```

### Connecting a remote client

Register it as an HTTP-type MCP server rather than a `command`/stdio one —
every client's config format is slightly different, so check your client's
docs; it needs the tunnel URL (`https://your-tunnel-url/mcp`) and an
`Authorization: Bearer YOUR_TOKEN` header. For Claude Code specifically,
see the [dedicated section](#claude-code) below.

Quick sanity check without any MCP client at all:

```bash
curl -i https://your-tunnel-url/mcp                       # expect 401
curl -i -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Accept: application/json, text/event-stream" \
     -H "Content-Type: application/json" \
     -X POST https://your-tunnel-url/mcp \
     -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"test","version":"0"}}}'
```

### Notes on running this exposed

- Quick tunnels (cloudflared/ngrok free tier) get a new random URL every
  restart — fine for testing, get a named/reserved tunnel if you want a
  stable address.
- The bearer token is a shared secret, not per-user auth — anyone with the
  token and URL has full read access to `query_docs`/`list_documents`.
  Rotate it (`RAG_AUTH_TOKEN`) if it leaks.
- The process needs to stay running for the tunnel to serve anything —
  run it under `systemd`, `tmux`, or `pm2` rather than a bare foreground
  shell if you want it up long-term.

## 6. Quick smoke test (optional, without any MCP client)

```bash
PYTHONPATH=src python -c "
import db
db.EMBED_MODEL, db.EMBED_DIM  # sanity check config
voyage = db.get_voyage_client()
conn = db.get_connection()
embedding = db.embed_texts(voyage, ['how does authentication work'], input_type='query')[0]
rows = conn.execute(
    'SELECT source, content FROM doc_chunks ORDER BY embedding <=> %s LIMIT 3',
    (embedding,),
).fetchall()
for source, content in rows:
    print(source, '-', content[:80])
"
```

## Notes / next steps

- **Chunking**: currently a simple ~800-char sliding window snapped to
  paragraph/sentence boundaries. If retrieval quality is off for long
  documents, consider chunking by markdown headers instead (keeps each
  section intact).
- **Embedding model**: `voyage-4-large` (1024 dims by default) out of the
  box — Voyage's best general-purpose/multilingual retrieval model as of
  this writing. For a cheaper/faster option try `voyage-4-lite` (same 1024
  default, same flexible 256/512/2048 options) — set `RAG_EMBED_MODEL` in
  `.env`. Since the dimension is baked into the Postgres `vector` column,
  changing `RAG_EMBED_DIM` (not needed for either of these two, since both
  default to 1024) requires `make db-reset` (wipes all data) and a full
  re-ingest; switching just the model name still forces a full re-embed
  automatically (see chunking/rebuild notes above) since embeddings from
  different models aren't comparable even at the same dimension.
- **Scaling**: ingestion is already incremental (keyed by file hash), so
  re-running it stays cheap as the doc set grows. The `doc_chunks` table has
  an HNSW index for approximate nearest-neighbor search, which scales well
  past what a project-sized doc set needs.
- **Multiple collections**: rows are namespaced by a `collection` column in
  the same table, so separate KBs (e.g. per project) just need a different
  `--collection` name at ingest time and matching `RAG_COLLECTION` per MCP
  server entry — no extra database needed. Note the embedding dimension
  (`RAG_EMBED_DIM`) is shared across all collections in one database.

## Claude Code

Everything above works with any MCP client. This section is just the
concrete Claude Code commands/config for the two connection modes from
section 4.

**stdio** (local, the default) — Claude Code launches the server itself:

```bash
claude mcp add local-docs -- /full/path/to/doc-rag-mcp/.venv/bin/python /full/path/to/doc-rag-mcp/src/server.py
```

Or generate `.mcp.json` automatically — `make gen-mcp-json` writes:

- `local-docs`: stdio, with this machine's real venv path, `src/server.py`
  path, and current `.env` values (`RAG_DATABASE_URL`, `RAG_COLLECTION`,
  `VOYAGE_API_KEY`) filled in.
- `local-docs-http`: only if `RAG_AUTH_TOKEN` is set (`make auth-token`) —
  points at `http://localhost:<RAG_PORT>/mcp` with the real token as a
  bearer header. This is for the server exposed *locally* by `make up`/
  `make serve-http`, not a tunnel — for genuine remote access over a
  tunnel (see below), add that entry by hand with the actual tunnel URL,
  since it can't be known in advance.

No manual editing either way, and it merges into an existing `.mcp.json`
rather than overwriting it, so other hand-added entries survive. Re-run it
any time `.env` changes (rotated password, different port, etc.).

Prefer to write it by hand instead? `cp .mcp.json.example .mcp.json` and
fill in the real venv path, password, and API key yourself (it ships with
both a stdio entry and the HTTP entry below; delete whichever you don't
need):

```json
{
  "mcpServers": {
    "local-docs": {
      "command": "/full/path/to/doc-rag-mcp/.venv/bin/python",
      "args": ["/full/path/to/doc-rag-mcp/src/server.py"],
      "env": {
        "RAG_DATABASE_URL": "postgresql://raguser:yourpassword@localhost:5433/ragdb",
        "RAG_COLLECTION": "project_docs",
        "VOYAGE_API_KEY": "your-voyage-api-key"
      }
    }
  }
}
```

**HTTP** (remote/tunnel, from section 5) — register it as an HTTP-type
server rather than a `command` entry:

```bash
claude mcp add --transport http local-docs-remote \
  https://your-tunnel-url/mcp \
  --header "Authorization: Bearer YOUR_TOKEN"
```

or in `.mcp.json`:

```json
{
  "mcpServers": {
    "local-docs-remote": {
      "type": "http",
      "url": "https://your-tunnel-url/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_TOKEN"
      }
    }
  }
}
```

Either way, restart Claude Code (or run `/mcp` to reconnect) to pick up new
or changed servers.

## Contributing

Issues and PRs welcome. This is a small, focused tool — keep additions in
that spirit rather than growing it into a framework.

## License

[MIT](LICENSE)
