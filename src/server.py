"""MCP server over the Postgres/pgvector knowledge base: query_docs,
list_documents, reindex_docs, remove_document, cache_stats, usage_stats,
clear_cache. See README for transports, auth, caching, and reranking/hybrid
search.

    python src/server.py                   # stdio (default)
    python src/server.py --transport http  # network-reachable
"""

import argparse
import asyncio
import hmac
import os
import re
import sys
import time
from collections import OrderedDict
from pathlib import Path
from threading import Lock

import psycopg
import uvicorn
from mcp.server.mcpserver import Context, MCPServer
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

import db
import ingest

MAX_TOP_K = 50
# When reranking or hybrid-searching, fetch this many candidates by vector
# (and, for hybrid, full-text) search per requested result, capped at
# MAX_CANDIDATES -- gives the reranker/fusion step a real pool to choose
# from without an unbounded/expensive fetch.
CANDIDATE_MULTIPLIER = 4
MAX_CANDIDATES = 100
RRF_K = 60  # standard constant for reciprocal rank fusion

_RAG_COLLECTION_ENV = os.environ.get("RAG_COLLECTION")
COLLECTION_NAME = _RAG_COLLECTION_ENV or "project_docs"
if not _RAG_COLLECTION_ENV:
    # Falling back silently is how docs from unrelated projects end up
    # sharing one collection on a shared Postgres instance -- print loudly
    # (stderr, since stdio transport reserves stdout for JSON-RPC) so a
    # misconfigured deploy is obvious instead of surfacing later as
    # confusing cross-project "pending removal" noise in list_documents().
    print(f"WARNING: RAG_COLLECTION is not set -- falling back to the shared default "
          f"collection {COLLECTION_NAME!r}. If this Postgres instance is shared by multiple "
          f"docs sources, set RAG_COLLECTION explicitly in .env.", file=sys.stderr)

CACHE_MAX_SIZE = int(os.environ.get("RAG_CACHE_MAX_SIZE", "256"))
CACHE_TTL_SECONDS = int(os.environ.get("RAG_CACHE_TTL_SECONDS", "600"))  # 10 min default

# check_index_freshness() walks every file under RAG_DOCS_DIR and scans every
# chunk row in the collection -- too expensive to redo on every list_documents()
# call (an agent calling it repeatedly in one session is the common case this
# guards against). Cached for this many seconds; reindex_docs()/remove_document()
# invalidate it immediately so freshness never lies right after a real change.
FRESHNESS_CACHE_TTL_SECONDS = int(os.environ.get("RAG_FRESHNESS_CACHE_TTL_SECONDS", "30"))

AUTH_TOKEN = os.environ.get("RAG_AUTH_TOKEN")  # required for --transport http (see below)

mcp = MCPServer("doc-rag-mcp")

_embed_client = None
_rerank_client = None


def run_query(fn):
    """Check out a pooled connection, call fn(conn) -> result, return it to
    the pool. Each call gets its own connection -- required now that server
    calls can run concurrently (--transport http, multiple clients) and a
    single shared connection can't safely serve two in-flight queries at
    once. If the checked-out connection turns out to be dead (Postgres
    restarted, a network blip), the pool discards it and this retries once
    with a fresh one -- without it, a single dropped connection would break
    every tool call until the process itself restarts."""
    try:
        with db.get_pool().connection() as conn:
            return fn(conn)
    except psycopg.OperationalError:
        with db.get_pool().connection() as conn:
            return fn(conn)


def get_embed_client():
    """Client for whichever RAG_EMBED_PROVIDER is configured (Voyage, Azure
    OpenAI, or OpenAI-compatible) -- used for embedding, not reranking."""
    global _embed_client
    if _embed_client is None:
        _embed_client = db.get_embed_client()
    return _embed_client


def get_rerank_client():
    """Always Voyage -- reranking has no Azure/OpenAI equivalent, and is
    unrelated to which provider embeds your documents. Raises ValueError
    (caught by query_docs) if that's a cross-provider data-egress surprise
    the user hasn't explicitly acknowledged -- see db.get_rerank_client."""
    global _rerank_client
    if _rerank_client is None:
        _rerank_client = db.get_rerank_client()
    return _rerank_client


class QueryCache:
    """Thread-safe LRU cache with a TTL, plus hit/miss counters."""

    def __init__(self, max_size: int, ttl_seconds: int):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._store = OrderedDict()  # key -> (timestamp, value)
        self._lock = Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(query: str, top_k: int, collection) -> tuple:
        # Include which ranking mode produced the answer -- rerank/hybrid/
        # plain vector search are fixed at process startup from env vars
        # today, so this is currently redundant within a single process,
        # but keying on it avoids a stale-mode answer being served if that
        # ever stops being true.
        mode = "rerank" if db.RERANK_MODEL else ("hybrid" if db.HYBRID_SEARCH else "vector")
        # A 2-element scope tuple, not a plain string -- so a real
        # collection someday literally named "all"/"ALL" can never collide
        # with the all_collections=True merged-search cache entry.
        scope = ("all",) if collection is None else ("one", collection)
        return (query.strip().lower(), top_k, mode, scope)

    def get(self, query: str, top_k: int, collection):
        key = self.make_key(query, top_k, collection)
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self.misses += 1
                return None
            timestamp, value = entry
            if time.time() - timestamp > self.ttl_seconds:
                del self._store[key]
                self.misses += 1
                return None
            self._store.move_to_end(key)
            self.hits += 1
            return value

    def set(self, query: str, top_k: int, collection, value: str):
        key = self.make_key(query, top_k, collection)
        with self._lock:
            self._store[key] = (time.time(), value)
            self._store.move_to_end(key)
            while len(self._store) > self.max_size:
                self._store.popitem(last=False)

    def clear(self):
        with self._lock:
            self._store.clear()
            self.hits = 0
            self.misses = 0

    def stats(self) -> dict:
        with self._lock:
            total = self.hits + self.misses
            hit_rate = (self.hits / total) if total else 0.0
            return {
                "size": len(self._store),
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(hit_rate, 3),
            }


_cache = QueryCache(CACHE_MAX_SIZE, CACHE_TTL_SECONDS)


def fetch_vector_candidates(query_embedding, limit: int, collection) -> list:
    """[(id, collection, source, chunk_index, content, distance), ...]
    ordered by ascending cosine distance (closest first). collection=None
    searches every collection in the shared database (no WHERE filter) --
    callers must have already confirmed that's actually safe (see the
    embed-model guard in query_docs) before passing None."""
    where = "WHERE collection = %s" if collection is not None else ""
    params = [query_embedding]
    if collection is not None:
        params.append(collection)
    params += [query_embedding, limit]
    return run_query(lambda conn: conn.execute(
        f"""
        SELECT id, collection, source, chunk_index, content, embedding <=> %s AS distance
        FROM doc_chunks
        {where}
        ORDER BY embedding <=> %s
        LIMIT %s
        """,
        params,
    ).fetchall())


def build_or_tsquery(query: str):
    """Turns a natural-language query into an OR-combined tsquery input
    ('foo | bar | baz') so full-text search matches chunks containing ANY
    query term, ranked by how many/how well they match -- the AND-only
    default of plainto_tsquery/websearch_to_tsquery often matches nothing
    for a multi-word natural-language question. Returns None if the query
    has no usable words at all (pure punctuation, etc.)."""
    words = re.findall(r"\w+", query)
    return " | ".join(words) if words else None


def fetch_fulltext_candidates(query: str, limit: int, collection) -> list:
    """[(id, collection, source, chunk_index, content, rank), ...] ordered
    by descending ts_rank (best match first). [] if the query has no usable
    search terms, rather than erroring. collection=None searches every
    collection (see fetch_vector_candidates)."""
    tsquery_param = build_or_tsquery(query)
    if not tsquery_param:
        return []
    collection_clause = " AND collection = %s" if collection is not None else ""
    # Param order must follow the %s occurrences left-to-right in the SQL
    # below: ts_rank's to_tsquery, then the WHERE to_tsquery, then (if
    # present) collection, then LIMIT.
    params = [tsquery_param, tsquery_param]
    if collection is not None:
        params.append(collection)
    params.append(limit)
    return run_query(lambda conn: conn.execute(
        f"""
        SELECT id, collection, source, chunk_index, content,
               ts_rank(content_tsv, to_tsquery('english', %s)) AS rank
        FROM doc_chunks
        WHERE content_tsv @@ to_tsquery('english', %s){collection_clause}
        ORDER BY rank DESC
        LIMIT %s
        """,
        params,
    ).fetchall())


def reciprocal_rank_fusion(*ranked_lists, k: int = RRF_K) -> list:
    """Each ranked_list: [(id, row), ...] already ordered best-first (rank
    is positional, not carried in the row). Fuses any number of such lists
    by summing 1/(k + rank) per list an id appears in -- an id missing
    from a list simply doesn't get that list's contribution, so this
    degrades gracefully to single-list ranking if one list is empty.
    Returns [(id, row, fused_score), ...] sorted best-first."""
    scores: dict = {}
    row_by_id: dict = {}
    for ranked_list in ranked_lists:
        for rank, (item_id, row) in enumerate(ranked_list, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
            row_by_id.setdefault(item_id, row)
    return [(item_id, row_by_id[item_id], score)
            for item_id, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]


def fetch_embed_models_by_collection() -> dict:
    """{collection: embed_model}, limited to collections that currently have
    indexed chunks (excludes a collection_config row left behind by a
    force_rebuild that wiped doc_chunks but hasn't been re-ingested yet --
    that's not a real search candidate). One query, not N+1."""
    rows = run_query(lambda conn: conn.execute(
        "SELECT cc.collection, cc.embed_model FROM collection_config cc "
        "WHERE EXISTS (SELECT 1 FROM doc_chunks dc WHERE dc.collection = cc.collection)"
    ).fetchall())
    return dict(rows)


@mcp.tool()
async def query_docs(query: str, top_k: int = 5, collection: str = None,
                      all_collections: bool = False, ctx: Context = None) -> str:
    """Search the project's architecture docs and PDFs for relevant passages.

    Args:
        query: Natural-language question or search phrase.
        top_k: Number of chunks to return (default 5, max 50).
        collection: Search this named collection instead of the server's
            default (RAG_COLLECTION). Mutually exclusive with all_collections.
            See list_collections() for what's available.
        all_collections: Search across every collection in the shared
            database instead of just one -- an explicit opt-in for when you
            deliberately want the whole knowledge base, not the default
            (isolated to one collection). Refused if the collections being
            merged were indexed with different embedding models, since their
            vectors aren't comparable.
    """
    if not query or not query.strip():
        return "query must not be empty."
    if top_k <= 0:
        return "top_k must be a positive integer."
    top_k = min(top_k, MAX_TOP_K)
    if collection is not None and all_collections:
        return "collection and all_collections=True are mutually exclusive -- pass one or the other."
    effective_collection = None if all_collections else (collection or COLLECTION_NAME)

    cached = _cache.get(query, top_k, effective_collection)
    if cached is not None:
        if ctx:
            await ctx.info(f"cache hit for query '{query}' (top_k={top_k})")
            await ctx.report_progress(1, 1)
        return cached

    if ctx:
        await ctx.info(f"cache miss for query '{query}' (top_k={top_k}) -- querying index")

    if all_collections:
        embed_models = fetch_embed_models_by_collection()
        distinct_models = set(embed_models.values())
        if len(distinct_models) > 1:
            by_model = {}
            for coll, model in embed_models.items():
                by_model.setdefault(model, []).append(coll)
            detail = "; ".join(f"{m}: {', '.join(sorted(cs))}" for m, cs in sorted(by_model.items()))
            answer = ("Cannot search across all collections: they were indexed with different "
                      f"embedding models, so vector similarity isn't comparable. {detail}. "
                      "Query a specific collection with collection=<name> instead.")
            _cache.set(query, top_k, effective_collection, answer)
            return answer

    try:
        query_embedding = run_query(
            lambda conn: db.embed_texts(get_embed_client(), [query], input_type="query", conn=conn)
        )[0]

        # Three tiers, in priority order -- rerank wins if both it and
        # hybrid search are configured. Normalize to (collection, source,
        # chunk_idx, content, score) with score always "higher is better",
        # regardless of which tier produced it.
        if db.RERANK_MODEL:
            # Over-fetch a larger candidate pool by cheap vector search,
            # then let the (more accurate, more expensive) reranker pick
            # the real top_k from those -- asking pgvector for exactly
            # top_k up front would deny the reranker anything to work with.
            fetch_k = min(top_k * CANDIDATE_MULTIPLIER, MAX_CANDIDATES)
            candidates = fetch_vector_candidates(query_embedding, fetch_k, effective_collection)
            if candidates:
                documents = [c[4] for c in candidates]
                ranked = run_query(
                    lambda conn: db.rerank_texts(get_rerank_client(), query, documents, top_k, conn=conn)
                )
                rows = [(candidates[idx][1], candidates[idx][2], candidates[idx][3],
                         candidates[idx][4], score)
                        for idx, score in ranked]
            else:
                rows = []

        elif db.HYBRID_SEARCH:
            # Free, provider-agnostic alternative: fuse vector search with
            # Postgres full-text search via reciprocal rank fusion. Each
            # side over-fetches the same way reranking does, so fusion has
            # a real pool -- one side coming back empty (e.g. the query has
            # no full-text-matchable terms) just degrades to the other.
            fetch_k = min(top_k * CANDIDATE_MULTIPLIER, MAX_CANDIDATES)
            vector_rows = fetch_vector_candidates(query_embedding, fetch_k, effective_collection)
            text_rows = fetch_fulltext_candidates(query, fetch_k, effective_collection)
            fused = reciprocal_rank_fusion(
                [(r[0], r) for r in vector_rows],
                [(r[0], r) for r in text_rows],
            )[:top_k]
            rows = [(row[1], row[2], row[3], row[4], score) for _id, row, score in fused]

        else:
            candidates = fetch_vector_candidates(query_embedding, top_k, effective_collection)
            rows = [(coll, source, chunk_idx, content, 1 - distance)
                    for _id, coll, source, chunk_idx, content, distance in candidates]
    except (SystemExit, Exception) as e:
        # SystemExit included deliberately: get_embed_client()/
        # get_rerank_client() raise it for missing provider config, and
        # that's a per-request condition here (a misconfigured env var),
        # not something that should actually tear down the server process.
        if ctx:
            await ctx.info(f"query failed: {e}")
        return f"Query failed: {e}"

    if not rows:
        answer = "No relevant results found in the knowledge base."
        _cache.set(query, top_k, effective_collection, answer)
        return answer

    total = len(rows)
    parts = []
    for i, (coll, source, chunk_idx, content, score) in enumerate(rows, start=1):
        label = f"{coll}/{source}" if all_collections else source
        parts.append(
            f"[{i}] source: {label} (chunk {chunk_idx}, relevance score {score:.3f})\n{content}"
        )
        if ctx:
            await ctx.report_progress(i, total)
            await ctx.info(f"matched chunk {i}/{total} from {label}")

    answer = "\n\n---\n\n".join(parts)
    _cache.set(query, top_k, effective_collection, answer)
    return answer


_freshness_lock = Lock()
_freshness_cache = {"result": None, "checked_at": 0.0}


def invalidate_freshness_cache():
    """Called after reindex_docs()/remove_document() so the next
    list_documents() recomputes freshness instead of serving a cached
    answer from before the change."""
    with _freshness_lock:
        _freshness_cache["checked_at"] = 0.0


def check_index_freshness():
    """Cached wrapper around _compute_index_freshness() -- see that
    function for what this actually checks. Recomputing it on every
    list_documents() call means walking every file under RAG_DOCS_DIR and
    scanning every chunk row in the collection each time, which gets slow
    fast when a client calls list_documents() repeatedly (e.g. an agent
    checking it once per turn). Cached for FRESHNESS_CACHE_TTL_SECONDS;
    invalidate_freshness_cache() clears it early after a real change."""
    with _freshness_lock:
        if time.time() - _freshness_cache["checked_at"] <= FRESHNESS_CACHE_TTL_SECONDS:
            return _freshness_cache["result"]
    result = _compute_index_freshness()
    with _freshness_lock:
        _freshness_cache["result"] = result
        _freshness_cache["checked_at"] = time.time()
    return result


def _compute_index_freshness():
    """Cheap, read-only comparison between what's on disk (RAG_DOCS_DIR) and
    what's actually indexed -- stat() only (mtime + size), no file content
    reads at all, so this stays fast regardless of how large the doc set or
    individual files get (unlike content hashing, which is O(total bytes)).
    Lets a client notice the docs changed from the tool response itself,
    without the user having to mention it.

    This is a heuristic, not the authoritative check: a file touched but
    not actually changed could show up as "changed" here. That's fine --
    the real content-hash diffing that decides what to actually re-embed
    already lives in reindex_docs()/ingest.py, which will correctly skip
    anything whose content didn't change even if this flagged it.

    Returns None if the docs path or DB can't be read (deleted folder,
    permissions, etc.) -- list_documents() still works, it just won't
    report freshness in that case."""
    try:
        docs_path = Path(os.environ.get("RAG_DOCS_DIR", "./docs")).resolve()
        base_dir, files = ingest.collect_files(docs_path)
    except Exception:
        return None

    try:
        rows = run_query(lambda conn: conn.execute(
            "SELECT DISTINCT ON (source) source, file_mtime, file_size FROM doc_chunks WHERE collection = %s",
            (COLLECTION_NAME,),
        ).fetchall())
    except Exception:
        return None
    existing = {source: (mtime, size) for source, mtime, size in rows}

    current_sources = set()
    new_files, changed_files = [], []
    for path in files:
        rel_path = str(path.relative_to(base_dir))
        current_sources.add(rel_path)
        try:
            mtime, size = ingest.file_stat(path)
        except Exception:
            continue
        if rel_path not in existing:
            new_files.append(rel_path)
        else:
            stored_mtime, stored_size = existing[rel_path]
            # NULL means this row predates file_mtime/file_size existing --
            # treat as "changed" (safer to over-flag once than silently miss
            # it; the next real reindex will populate these fields).
            if stored_mtime is None or stored_size is None or mtime != stored_mtime or size != stored_size:
                changed_files.append(rel_path)

    removed_files = sorted(set(existing) - current_sources)
    return {"new": sorted(new_files), "changed": sorted(changed_files), "removed": removed_files}


@mcp.tool()
def list_documents(collection: str = None, all_collections: bool = False) -> str:
    """List every source document currently indexed, when the collection
    was last reindexed, and whether the docs folder has changed since
    (new/changed/removed files) -- so a client can notice staleness and
    call reindex_docs() on its own, without the user having to say so.

    Args:
        collection: List this named collection instead of the server's
            default (RAG_COLLECTION). Mutually exclusive with all_collections.
        all_collections: List documents across every collection in the
            shared database, labeled "collection/source". Staleness
            checking and the pending-removal notice are per-collection and
            are skipped in this mode -- see list_collections() for that
            detail broken out by collection.
    """
    if collection is not None and all_collections:
        return "collection and all_collections=True are mutually exclusive -- pass one or the other."
    effective_collection = None if all_collections else (collection or COLLECTION_NAME)

    try:
        if all_collections:
            rows = run_query(lambda conn: conn.execute(
                "SELECT DISTINCT collection, source FROM doc_chunks ORDER BY collection, source",
            ).fetchall())
        else:
            rows = run_query(lambda conn: conn.execute(
                "SELECT DISTINCT source FROM doc_chunks WHERE collection = %s ORDER BY source",
                (effective_collection,),
            ).fetchall())
    except Exception as e:
        return f"Failed to list documents: {e}"
    if not rows:
        return "No documents indexed yet. Run ingest.py first."

    if all_collections:
        sources = [f"{coll}/{source}" for coll, source in rows]
        collections_seen = {coll for coll, _source in rows}
        lines = [f"{len(sources)} document(s) indexed across {len(collections_seen)} collection(s). "
                 "Call list_collections() for per-collection detail."]
    else:
        sources = [source for (source,) in rows]
        try:
            meta = run_query(lambda conn: conn.execute(
                "SELECT last_indexed_at, pending_removal_sources FROM collection_config WHERE collection = %s",
                (effective_collection,),
            ).fetchone())
        except Exception:
            meta = None
        last_indexed_at = meta[0] if meta else None
        pending_removals = meta[1] if meta and meta[1] else None

        lines = [f"{len(sources)} document(s) indexed"
                 + (f", last reindexed {last_indexed_at}" if last_indexed_at else "") + "."]

        if pending_removals:
            lines.append(
                f"! {len(pending_removals)} previously-indexed file(s) no longer exist under the docs "
                f"path but were NOT removed -- the last reindex refused because that looked like a "
                f"misconfigured RAG_DOCS_DIR rather than intentional deletion: {', '.join(pending_removals)}. "
                "This stays flagged until RAG_DOCS_DIR is fixed and reindexed, or until you call "
                "reindex_docs(confirm_large_removal=True) to confirm the deletion is intentional."
            )

        # Freshness compares RAG_DOCS_DIR on disk against one collection's
        # rows -- only meaningful for this server's own configured
        # collection, since that's the only docs directory this process
        # actually knows about. Checking it against a different collection
        # (or merged across all of them) would compare the wrong directory
        # to the wrong rows and reproduce exactly the cross-collection
        # contamination bug this whole feature exists to prevent.
        if effective_collection == COLLECTION_NAME:
            freshness = check_index_freshness()
            if freshness is None:
                lines.append("(could not check the docs folder for changes)")
            elif freshness["new"] or freshness["changed"] or freshness["removed"]:
                lines.append(
                    f"! Docs folder has changed since last index: {len(freshness['new'])} new, "
                    f"{len(freshness['changed'])} changed, {len(freshness['removed'])} removed. "
                    "Call reindex_docs() to update."
                )
            else:
                lines.append("Docs folder matches the index -- no reindex needed.")
        else:
            lines.append(
                "(freshness check only applies to this server's own collection "
                f"{COLLECTION_NAME!r} -- omit collection/all_collections to check it)"
            )

    lines.append("")
    lines.extend(sources)
    return "\n".join(lines)


@mcp.tool()
def list_collections() -> str:
    """Enumerate every collection in the shared Postgres instance -- embed
    model, chunk settings, last reindex time, document count, and any
    pending-removal flag -- so you can see what's actually in the shared
    database before an all_collections=True search."""
    try:
        rows = run_query(lambda conn: conn.execute(
            """
            SELECT cc.collection, cc.embed_model, cc.chunk_size, cc.chunk_overlap,
                   cc.last_indexed_at, cc.pending_removal_sources,
                   COUNT(DISTINCT dc.source) AS doc_count
            FROM collection_config cc
            LEFT JOIN doc_chunks dc ON dc.collection = cc.collection
            GROUP BY cc.collection, cc.embed_model, cc.chunk_size, cc.chunk_overlap,
                     cc.last_indexed_at, cc.pending_removal_sources
            ORDER BY cc.collection
            """
        ).fetchall())
    except Exception as e:
        return f"Failed to list collections: {e}"
    if not rows:
        return "No collections indexed yet."

    lines = [f"{len(rows)} collection(s) in the shared database:"]
    for coll, model, chunk_size, chunk_overlap, last_indexed, pending, doc_count in rows:
        marker = " (this server's default)" if coll == COLLECTION_NAME else ""
        line = (f"- {coll}{marker}: {doc_count} document(s), embed_model={model}, "
                f"chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, last_indexed={last_indexed}")
        if pending:
            line += f", ! {len(pending)} pending removal(s)"
        lines.append(line)
    return "\n".join(lines)


@mcp.tool()
def cache_stats() -> str:
    """Report query cache hit/miss counts, hit rate, and current size."""
    s = _cache.stats()
    return (
        f"hits: {s['hits']}, misses: {s['misses']}, hit_rate: {s['hit_rate']}, "
        f"size: {s['size']}/{s['max_size']}, ttl_seconds: {s['ttl_seconds']}"
    )


@mcp.tool()
def usage_stats() -> str:
    """Report cumulative embedding/rerank API token usage and query cache
    performance for this knowledge base -- a local tally of calls made
    through ingest.py/this server, NOT any provider's authoritative usage
    or remaining balance (this can't see usage from a shared API key used
    elsewhere). Check your embedding provider's own dashboard for that:
    Voyage https://dashboard.voyageai.com, Azure the Azure portal's cost
    management, OpenAI https://platform.openai.com/usage. Reranking (if
    RAG_RERANK_MODEL is set) always uses Voyage regardless of embedding
    provider -- see https://dashboard.voyageai.com for that usage too."""
    try:
        rows = run_query(lambda conn: conn.execute(
            "SELECT model, operation, total_tokens, total_requests, updated_at "
            "FROM usage_totals ORDER BY model, operation"
        ).fetchall())
    except Exception as e:
        return f"Failed to read usage stats: {e}"

    lines = [f"-- API usage (embed provider: {db.EMBED_PROVIDER}; this knowledge base only) --"]
    if not rows:
        lines.append("No embedding calls recorded yet.")
    else:
        total_tokens = sum(r[2] for r in rows)
        total_requests = sum(r[3] for r in rows)
        for model, operation, tokens, requests, updated_at in rows:
            lines.append(f"{model} ({operation}): {tokens:,} tokens across {requests:,} "
                         f"request(s), last used {updated_at}")
        lines.append(f"Total: {total_tokens:,} tokens across {total_requests:,} request(s).")

    s = _cache.stats()
    lines.append("")
    lines.append("-- Query cache --")
    lines.append(f"hits: {s['hits']}, misses: {s['misses']}, hit_rate: {s['hit_rate']}, "
                 f"size: {s['size']}/{s['max_size']}, ttl_seconds: {s['ttl_seconds']}")

    return "\n".join(lines)


@mcp.tool()
def clear_cache() -> str:
    """Clear the query cache. Run this after re-running ingest.py so stale
    cached answers from the old index aren't served."""
    _cache.clear()
    return "Cache cleared."


@mcp.tool()
def remove_document(source: str, collection: str = None) -> str:
    """Remove a single document from the index on demand, without a full
    reindex_docs() run -- useful when a file is gone/renamed and you don't
    want to wait for (or trigger) a full reconciliation pass.

    Deletes every chunk for `source` in the target collection. Also clears
    it from that collection's pending-removal list (see list_documents())
    if it was flagged there, so it stops being surfaced as a refused mass
    removal. There's deliberately no all_collections option here -- removal
    is scoped to exactly one collection at a time, never a blanket delete
    across the shared database.

    Args:
        source: The document path exactly as shown by list_documents()
            (relative to RAG_DOCS_DIR), e.g. "guides/setup.md".
        collection: Remove from this named collection instead of the
            server's default (RAG_COLLECTION).
    """
    if not source or not source.strip():
        return "source must not be empty."
    source = source.strip()
    effective_collection = collection or COLLECTION_NAME

    def do_remove(conn):
        with conn.transaction():
            deleted = conn.execute(
                "DELETE FROM doc_chunks WHERE collection = %s AND source = %s",
                (effective_collection, source),
            ).rowcount

            row = conn.execute(
                "SELECT pending_removal_sources FROM collection_config WHERE collection = %s",
                (effective_collection,),
            ).fetchone()
            pending = row[0] if row and row[0] else []
            was_pending = source in pending
            if was_pending:
                remaining = [s for s in pending if s != source] or None
                conn.execute(
                    "UPDATE collection_config SET pending_removal_sources = %s WHERE collection = %s",
                    (remaining, effective_collection),
                )
        return deleted, was_pending

    try:
        deleted, was_pending = run_query(do_remove)
    except Exception as e:
        return f"Failed to remove document: {e}"

    if deleted == 0 and not was_pending:
        return f"No such document indexed: {source!r}"

    _cache.clear()
    invalidate_freshness_cache()

    parts = [f"Removed {deleted} chunk(s) for {source!r}."]
    if was_pending:
        parts.append("Also cleared it from the pending-removal list.")
    return " ".join(parts)


_reindex_lock = asyncio.Lock()


@mcp.tool()
async def reindex_docs(force_rebuild: bool = False, confirm_large_removal: bool = False,
                        ctx: Context = None) -> str:
    """Re-scan RAG_DOCS_DIR and embed any new or changed files into the
    knowledge base, without leaving the chat to run ingest.py by hand.

    Incremental by default: unchanged files are skipped (no embedding API
    cost); this can still take a while for a large or fully-changed doc set
    (one Voyage API call per batch of chunks) -- the call blocks until
    ingestion finishes. The query cache is cleared automatically afterward
    so query_docs doesn't keep serving stale answers from before this run.

    Args:
        force_rebuild: Ignore stored file hashes and re-embed every file,
            even ones that haven't changed. Also happens automatically if
            the embedding model or chunk settings changed since the last run.
        confirm_large_removal: Allow removing a large fraction of
            previously-indexed files in one run. Refused by default as a
            safety net: that pattern is much more often a misconfigured
            RAG_DOCS_DIR (wrong folder, an empty mount) than genuine bulk
            deletion -- the response explains what was left untouched if
            this trips, and small day-to-day removals are never affected.
    """
    if _reindex_lock.locked():
        return "A reindex is already running -- wait for it to finish before starting another."

    docs_path = Path(os.environ.get("RAG_DOCS_DIR", "./docs")).resolve()
    lines = []
    error = None

    async with _reindex_lock:
        if ctx:
            await ctx.info(f"reindexing {docs_path} (collection={COLLECTION_NAME}, "
                            f"force_rebuild={force_rebuild})")
        try:
            # Runs in a worker thread so the rest of the server (health
            # checks, other tool calls) stays responsive while this --
            # potentially multi-minute -- ingestion runs.
            await asyncio.to_thread(ingest.build_index, docs_path, COLLECTION_NAME, force_rebuild,
                                     lines.append, confirm_large_removal)
        except (SystemExit, Exception) as e:
            error = str(e)
        finally:
            _cache.clear()
            invalidate_freshness_cache()

    if error:
        return f"Reindex failed: {error}\n" + "\n".join(lines)
    lines.append("Query cache cleared automatically.")
    return "\n".join(lines)


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Rejects any request that doesn't carry the configured bearer token.
    Only mounted when running over --transport http. /health is exempt --
    it's used by the container HEALTHCHECK, which doesn't send a token, and
    it exposes nothing beyond DB reachability."""

    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/health":
            return await call_next(request)
        expected = f"Bearer {AUTH_TOKEN}"
        actual = request.headers.get("authorization") or ""
        # Constant-time compare -- a plain != leaks how many leading
        # characters matched via response timing, which a naive brute-force
        # could exploit over enough attempts.
        if not hmac.compare_digest(actual, expected):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)


async def health(request: Request) -> JSONResponse:
    try:
        run_query(lambda conn: conn.execute("SELECT 1"))
    except Exception as e:
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=503)
    return JSONResponse({"status": "ok"})


def run_http(host: str, port: int):
    if not AUTH_TOKEN:
        raise SystemExit(
            "RAG_AUTH_TOKEN is not set. Refusing to start an unauthenticated "
            "server over --transport http -- set RAG_AUTH_TOKEN=<some-secret> "
            "and have clients send 'Authorization: Bearer <same-secret>'."
        )
    app = mcp.streamable_http_app(host=host)
    app.add_route("/health", health, methods=["GET"])
    app.add_middleware(BearerAuthMiddleware)
    print(f"Serving MCP over Streamable HTTP at http://{host}:{port}/mcp "
          f"(auth required)")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default=COLLECTION_NAME)
    parser.add_argument("--transport", choices=["stdio", "http"],
                         default=os.environ.get("RAG_TRANSPORT", "stdio"))
    parser.add_argument("--host", default=os.environ.get("RAG_HOST", "127.0.0.1"),
                         help="Bind address for --transport http. Keep this "
                              "127.0.0.1 if a tunnel tool (ngrok/cloudflared/"
                              "tailscale) is forwarding to it locally; use "
                              "0.0.0.0 only for direct LAN exposure.")
    parser.add_argument("--port", type=int, default=int(os.environ.get("RAG_PORT", "8743")))
    args = parser.parse_args()
    COLLECTION_NAME = args.collection

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        run_http(args.host, args.port)

