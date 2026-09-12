# doc-rag-mcp

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Turn any folder of docs (PDFs, markdown, text) into a local, queryable
knowledge base exposed as an [MCP](https://modelcontextprotocol.io) server
— so any MCP-compatible agent (Claude Code, or otherwise) can search your
docs for relevant passages instead of reading whole files into its context
window. There's a [dedicated section](#claude-code) below for Claude Code
specifics, since that's what this was built and tested against, but nothing
about the server itself is Claude-specific.

Embeddings run via a pluggable API provider — Voyage, Azure OpenAI, or
OpenAI/OpenAI-compatible (see "Choosing an embedding provider" below) — no
multi-GB local model download either way. Storage is Postgres with the
`pgvector` extension. Runs entirely on your own machine (or wherever you
deploy it) — your documents and queries only ever leave the machine for
that one embedding call, to whichever provider you configured, plus one
more optional case: if you turn on reranking (`RAG_RERANK_MODEL`), chunk
text also goes to Voyage's rerank API specifically, *regardless* of which
provider embeds your documents — see the cross-provider note under
Reranking if you picked a non-Voyage provider for data-residency reasons.

## How it works

**Ingestion** — point it at a folder (or a single file), it chunks and
embeds anything new or changed (via whichever `RAG_EMBED_PROVIDER` is
configured — Voyage, Azure OpenAI, or OpenAI), and skips the rest:

```mermaid
flowchart TD
    A[docs folder] --> B[chunk text]
    B --> C{content hash<br/>changed?}
    C -- no --> D[skip, no API call]
    C -- yes --> E[embed via configured provider]
    E --> F[(Postgres + pgvector)]
```

**Querying** — the agent calls the MCP tool, which embeds the question,
runs a vector search, then improves on plain cosine-similarity ranking if
reranking or hybrid search is configured (rerank wins if both are):

```mermaid
flowchart TD
    A[MCP client] -- query_docs --> B[MCP server]
    B --> C{cached?}
    C -- yes --> Z[return cached answer]
    C -- no --> D[embed query via configured provider]
    D --> E[(vector search<br/>Postgres + pgvector)]
    E --> F{rerank or<br/>hybrid search set?}
    F -- rerank --> G[re-score with<br/>Voyage rerank API]
    F -- hybrid --> H[fuse with full-text search<br/>via reciprocal rank fusion]
    F -- neither --> I[top-k by<br/>cosine similarity]
    G --> J[top-k results]
    H --> J
    I --> J
    J --> A
```

**Staying fresh** — the agent doesn't need the user to say "I changed the
docs." `list_documents()` reports when the collection was last reindexed
and, via a cheap `stat()`-only check (file mtime + size, no file content
read — so this stays fast no matter how large the docs get), whether
anything on disk looks new/changed/removed since. Seeing that signal is
enough for the agent to call `reindex_docs()` on its own:

```mermaid
sequenceDiagram
    actor Agent
    participant Server as MCP server
    participant FS as Docs folder
    participant DB as Postgres

    Agent->>Server: list_documents()
    Server->>FS: stat() every file (mtime + size)
    Server->>DB: last indexed mtime/size per source
    Server-->>Agent: file list + "! N new, M changed, K removed -- reindex_docs()"
    Note over Agent: notices the staleness signal,<br/>no user prompt needed
    Agent->>Server: reindex_docs()
    Server->>FS: re-read changed/new files
    Server->>Server: content-hash diff
    Server->>DB: embed + upsert changed files,<br/>delete chunks for removed files
    Server-->>Agent: "X new, Y changed, Z removed"
```

The `stat()` check is a fast heuristic, not the authoritative one — a
touched-but-unedited file can show up as "changed" here. That's fine:
the content-hash diffing in the ingestion flow above is what actually
decides what gets re-embedded, and it correctly skips anything whose
content didn't really change.

Both flows share one Postgres table (`doc_chunks`), namespaced by a
`collection` column so multiple doc sets can coexist in the same database.

### Choosing an embedding provider

Set via `RAG_EMBED_PROVIDER` in `.env` before your first `ingest` — whichever
you pick, `ingest.py` and `server.py` must agree (same `.env`), since
switching later means the old embeddings are no longer comparable to new
ones (see the rebuild notes in the Notes/next-steps section).

- **`voyage`** (default) — Voyage AI's API. No local model download, just
  `VOYAGE_API_KEY`.
- **`azure-openai`** — your own Azure OpenAI embeddings deployment. Useful
  if you already have Azure credits and don't want another vendor
  subscription. Needs `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, and
  `AZURE_OPENAI_EMBED_DEPLOYMENT` (the deployment name, not the model name
  — Azure routes by deployment).
- **`openai`** — vanilla OpenAI, or any OpenAI-compatible endpoint (Ollama,
  vLLM, etc.) via `OPENAI_BASE_URL`. Needs `OPENAI_API_KEY` and
  `RAG_EMBED_MODEL` set to that endpoint's embedding model name.

Reranking (`RAG_RERANK_MODEL`, see below) always uses Voyage's rerank API
regardless of this choice — there's no Azure/OpenAI equivalent, and the two
settings are otherwise independent (embed via Azure, still rerank via
Voyage's free tier, if you want). If you didn't pick `voyage` above,
turning on reranking requires an extra explicit opt-in
(`RAG_ALLOW_CROSS_PROVIDER_RERANK=true`) precisely because it sends chunk
text to Voyage regardless — see the cross-provider note under Reranking.

## 1. Install

```bash
cd doc-rag-mcp
cp .env.example .env   # fill in credentials for your chosen RAG_EMBED_PROVIDER
                        # (see above) and POSTGRES_PASSWORD at least
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
  returns the matching passages with their source file and a relevance
  score. Ranks by rerank (`RAG_RERANK_MODEL`) if set, else hybrid search
  (`RAG_HYBRID_SEARCH=true`) if enabled, else plain cosine similarity — see
  "Improving on plain vector search" below.
- `list_documents()` — lists every file currently indexed, when the
  collection was last reindexed, and a fast staleness signal (new/changed/
  removed files since then, detected via file mtime + size) — see
  "Staying fresh" above
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
- `usage_stats()` — cumulative embedding/rerank API token usage (by
  model/operation, whichever `RAG_EMBED_PROVIDER` is active) plus cache
  stats, combined. This is a local tally of calls made through this
  server/`ingest.py`, not any provider's authoritative usage — check
  [Voyage's dashboard](https://dashboard.voyageai.com), the Azure portal,
  or [OpenAI's usage page](https://platform.openai.com/usage) for that,
  depending on your provider. Reranking always shows up under Voyage
  regardless of embed provider (see Reranking below).
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

### Improving on plain vector search (optional)

By default, `query_docs` ranks purely by embedding cosine similarity — fast
and cheap, but it scores the query and each chunk independently, so it can
miss subtleties a direct query/document comparison (or an exact term/code
match) would catch. Two ways to do better, in priority order if you set
both — reranking wins:

1. **Reranking** (`RAG_RERANK_MODEL`, e.g. `rerank-2.5`) — fetches a larger
   candidate pool by vector search (`top_k × 4`, capped at 100) and
   re-scores those against the actual query text with Voyage's rerank API,
   returning the best `top_k` after that second pass. Always uses Voyage
   regardless of `RAG_EMBED_PROVIDER` (no Azure/OpenAI equivalent exists).
   Costs its own tokens on every query, in addition to the query
   embedding — a rerank call processes full chunk text rather than just
   the short query, so it's typically the larger of the two costs. Check
   `usage_stats()` (it tracks rerank calls as their own `operation`) to see
   the actual cost for your usage pattern before deciding whether to leave
   it on.

   **Cross-provider note:** if `RAG_EMBED_PROVIDER` isn't `voyage`,
   `query_docs` *refuses* to rerank until you also set
   `RAG_ALLOW_CROSS_PROVIDER_RERANK=true`. This is deliberate — reranking
   sends chunk text to Voyage no matter which provider embeds your
   documents, which would otherwise be a silent, easy-to-miss surprise for
   anyone who picked Azure/OpenAI specifically for data-residency reasons.
   No such ack is needed when `RAG_EMBED_PROVIDER=voyage`, since that's the
   same provider for both, not a second one appearing.
2. **Hybrid search** (`RAG_HYBRID_SEARCH=true`) — free and provider-agnostic:
   fuses vector search with Postgres full-text search (`tsvector`/`ts_rank`)
   via [reciprocal rank fusion](https://en.wikipedia.org/wiki/Reciprocal_rank_fusion),
   no API calls at all. Helps most on queries with specific terms, codes, or
   names that cosine similarity alone tends to under-rank (an embedding
   captures meaning, not exact tokens). If the full-text side finds nothing
   for a given query (e.g. it's all stopwords), it degrades gracefully to
   vector-only ranking rather than erroring.

Only set one, or set `RAG_RERANK_MODEL` and treat `RAG_HYBRID_SEARCH` as a
free fallback for when you want to unset rerank temporarily.

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
  `.env`. See "Choosing an embedding provider" above if you'd rather use
  Azure OpenAI or another OpenAI-compatible endpoint instead of Voyage.
  Since the dimension is baked into the Postgres `vector` column, changing
  `RAG_EMBED_DIM` requires `make db-reset` (wipes all data) and a full
  re-ingest; switching the model (or provider) still forces a full
  re-embed automatically (see chunking/rebuild notes above) since
  embeddings from different models aren't comparable even at the same
  dimension.
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
claude mcp add doc-rag-mcp -- /full/path/to/doc-rag-mcp/.venv/bin/python /full/path/to/doc-rag-mcp/src/server.py
```

Or generate `.mcp.json` automatically — `make gen-mcp-json` writes:

- `doc-rag-mcp`: stdio, with this machine's real venv path, `src/server.py`
  path, and current `.env` values (`RAG_DATABASE_URL`, `RAG_COLLECTION`,
  `VOYAGE_API_KEY`) filled in.
- `doc-rag-mcp-http`: only if `RAG_AUTH_TOKEN` is set (`make auth-token`) —
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
    "doc-rag-mcp": {
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
claude mcp add --transport http doc-rag-mcp-remote \
  https://your-tunnel-url/mcp \
  --header "Authorization: Bearer YOUR_TOKEN"
```

or in `.mcp.json`:

```json
{
  "mcpServers": {
    "doc-rag-mcp-remote": {
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

### Included skill

This repo ships a [Claude Code Skill](.claude/skills/search-docs/SKILL.md)
(`search-docs`) that teaches Claude how to use these tools well: when a
single `query_docs` call is enough versus when to check `list_documents()`
first or issue several differently-phrased queries for a broad/cross-cutting
question, when to call `reindex_docs()` proactively instead of telling you
to run a command, and to respect the removal-safety guard rather than
blindly overriding it. It's picked up automatically once this repo is your
working directory — no separate install step.

## Contributing

Issues and PRs welcome. This is a small, focused tool — keep additions in
that spirit rather than growing it into a framework.

## License

[MIT](LICENSE)
