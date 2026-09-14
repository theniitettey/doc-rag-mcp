"""Shared Postgres/pgvector connection, schema, and embedding calls used by
ingest.py and server.py. Embedding provider is pluggable (RAG_EMBED_PROVIDER)
and reranking is Voyage-only regardless of provider -- see README."""

import os

import psycopg
import voyageai
from pgvector import Vector
from pgvector.psycopg import register_vector

DATABASE_URL = os.environ.get("RAG_DATABASE_URL", "postgresql://raguser:ragpass@localhost:5432/ragdb")
# Both matter for the same failure mode: a connection or an embedding API
# call that goes quietly dead mid-request (network blip, Postgres/Voyage
# restart) with no timeout hangs forever from the caller's side -- a tool
# call that never errors and never returns, rather than one that fails fast
# and lets the caller retry.
DB_CONNECT_TIMEOUT_SECONDS = int(os.environ.get("RAG_DB_CONNECT_TIMEOUT", "10"))
DB_STATEMENT_TIMEOUT_MS = int(os.environ.get("RAG_DB_STATEMENT_TIMEOUT_MS", "30000"))
EMBED_TIMEOUT_SECONDS = float(os.environ.get("RAG_EMBED_TIMEOUT_SECONDS", "30"))
EMBED_PROVIDER = os.environ.get("RAG_EMBED_PROVIDER", "voyage").lower()
EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "voyage-4-large")
EMBED_DIM = int(os.environ.get("RAG_EMBED_DIM", "1024"))
VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY")
# Empty (default) = reranking disabled. Opt-in via RAG_RERANK_MODEL since it
# costs its own tokens on top of the embedding call, on every query.
RERANK_MODEL = os.environ.get("RAG_RERANK_MODEL", "")
# Free, provider-agnostic alternative/complement to reranking: fuse vector
# search with Postgres full-text search (no API calls at all). Only takes
# effect when RERANK_MODEL is unset -- rerank wins if both are configured.
HYBRID_SEARCH = os.environ.get("RAG_HYBRID_SEARCH", "").lower() in ("1", "true", "yes")
# Reranking always calls Voyage regardless of EMBED_PROVIDER (no Azure/OpenAI
# equivalent exists) -- if you picked azure-openai/openai specifically for
# data-residency reasons, RAG_RERANK_MODEL would otherwise silently send
# chunk text to Voyage anyway. Require this explicit ack in that case; no
# extra step needed when EMBED_PROVIDER=voyage, since that's not a
# cross-provider egress at all.
ALLOW_CROSS_PROVIDER_RERANK = os.environ.get("RAG_ALLOW_CROSS_PROVIDER_RERANK", "").lower() in ("1", "true", "yes")

# Azure OpenAI (RAG_EMBED_PROVIDER=azure-openai)
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")
AZURE_OPENAI_EMBED_DEPLOYMENT = os.environ.get("AZURE_OPENAI_EMBED_DEPLOYMENT")

# Vanilla OpenAI or an OpenAI-compatible endpoint (RAG_EMBED_PROVIDER=openai)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")  # unset = api.openai.com


def get_connection():
    conn = psycopg.connect(
        DATABASE_URL,
        autocommit=True,
        connect_timeout=DB_CONNECT_TIMEOUT_SECONDS,
        # TCP keepalives so a connection that died silently (container
        # restart, network blip) gets noticed -- raised as OperationalError,
        # which run_query() already catches and reconnects on -- instead of
        # a query just sitting there waiting for a response that will never
        # come.
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=3,
        options=f"-c statement_timeout={DB_STATEMENT_TIMEOUT_MS}",
    )
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    ensure_schema(conn)
    return conn


def ensure_schema(conn):
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS doc_chunks (
            id TEXT PRIMARY KEY,
            collection TEXT NOT NULL,
            source TEXT NOT NULL,
            chunk_index INT NOT NULL,
            file_hash TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding VECTOR({EMBED_DIM}) NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS doc_chunks_collection_source_idx
        ON doc_chunks (collection, source)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS doc_chunks_embedding_idx
        ON doc_chunks USING hnsw (embedding vector_cosine_ops)
    """)
    # Added via ALTER (not the CREATE TABLE above) so it backfills correctly
    # on a database that already has rows from before hybrid search existed,
    # the same way a fresh install gets it too.
    conn.execute("""
        ALTER TABLE doc_chunks ADD COLUMN IF NOT EXISTS content_tsv tsvector
        GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS doc_chunks_content_tsv_idx
        ON doc_chunks USING gin (content_tsv)
    """)
    # Fast (stat()-only, no file reads) staleness signal for list_documents()
    # -- separate from file_hash, which is the authoritative content hash
    # actual reindexing uses. NULL until the next real reindex populates
    # them for rows written before this existed.
    conn.execute("""
        ALTER TABLE doc_chunks ADD COLUMN IF NOT EXISTS file_mtime DOUBLE PRECISION
    """)
    conn.execute("""
        ALTER TABLE doc_chunks ADD COLUMN IF NOT EXISTS file_size BIGINT
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS collection_config (
            collection TEXT PRIMARY KEY,
            embed_model TEXT NOT NULL,
            chunk_size INT NOT NULL,
            chunk_overlap INT NOT NULL
        )
    """)
    # Added via ALTER for the same reason as content_tsv above -- backfills
    # existing collections instead of only applying to new ones.
    conn.execute("""
        ALTER TABLE collection_config ADD COLUMN IF NOT EXISTS last_indexed_at TIMESTAMPTZ
    """)
    # Set when a reindex refuses a large removal (see build_index's
    # removal_threshold guard) and cleared the next time a reindex completes
    # without tripping it -- so the refusal stays visible in list_documents
    # across calls instead of only appearing once, in that one reindex_docs
    # response, which is easy to miss.
    conn.execute("""
        ALTER TABLE collection_config ADD COLUMN IF NOT EXISTS pending_removal_sources TEXT[]
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS usage_totals (
            model TEXT NOT NULL,
            operation TEXT NOT NULL,
            total_tokens BIGINT NOT NULL DEFAULT 0,
            total_requests BIGINT NOT NULL DEFAULT 0,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (model, operation)
        )
    """)


def record_usage(conn, model: str, operation: str, tokens: int):
    """operation is Voyage's input_type ('document' or 'query'). Recorded as
    its own auto-committed statement (not inside any caller's transaction)
    so it persists even if a later step in that transaction rolls back --
    the API call happened and cost tokens regardless of what happens next."""
    conn.execute(
        """
        INSERT INTO usage_totals (model, operation, total_tokens, total_requests, updated_at)
        VALUES (%s, %s, %s, 1, now())
        ON CONFLICT (model, operation) DO UPDATE SET
            total_tokens = usage_totals.total_tokens + EXCLUDED.total_tokens,
            total_requests = usage_totals.total_requests + 1,
            updated_at = now()
        """,
        (model, operation, tokens),
    )


def get_voyage_client() -> voyageai.Client:
    """Always Voyage, regardless of RAG_EMBED_PROVIDER -- used internally
    when RAG_EMBED_PROVIDER=voyage, and by get_rerank_client() below. Not
    the right entry point for reranking specifically -- use
    get_rerank_client() so the cross-provider check below actually runs."""
    if not VOYAGE_API_KEY:
        raise SystemExit("VOYAGE_API_KEY is not set. Get one at https://dash.voyageai.com and set it in your .env.")
    # voyageai.Client defaults to timeout=None (no timeout at all) -- a
    # stalled request would otherwise hang the calling tool forever instead
    # of failing fast.
    return voyageai.Client(api_key=VOYAGE_API_KEY, timeout=EMBED_TIMEOUT_SECONDS)


def get_rerank_client() -> voyageai.Client:
    """Client for reranking -- always Voyage, since there's no Azure/OpenAI
    equivalent. If RAG_EMBED_PROVIDER isn't voyage, that means chunk text
    would go to Voyage even though you picked a different provider --
    possibly for data-residency reasons. Refuse by default rather than do
    that silently; RAG_ALLOW_CROSS_PROVIDER_RERANK=true opts in explicitly."""
    if EMBED_PROVIDER != "voyage" and not ALLOW_CROSS_PROVIDER_RERANK:
        raise ValueError(
            f"RAG_RERANK_MODEL is set but RAG_EMBED_PROVIDER={EMBED_PROVIDER!r} -- reranking "
            "always uses Voyage's API regardless of embed provider, so chunk text would be "
            "sent to Voyage even though you chose a different one (possibly for data-residency "
            "reasons). Set RAG_ALLOW_CROSS_PROVIDER_RERANK=true in .env to confirm that's fine, "
            "or unset RAG_RERANK_MODEL (RAG_HYBRID_SEARCH is a free, provider-agnostic "
            "alternative that never leaves Postgres)."
        )
    return get_voyage_client()


def get_embed_client():
    """Returns a client for whichever RAG_EMBED_PROVIDER is configured. Pass
    it to embed_texts() -- it dispatches on EMBED_PROVIDER, not the client's
    type, so this and embed_texts must agree on what provider is active
    (they both read the same module-level EMBED_PROVIDER)."""
    if EMBED_PROVIDER == "voyage":
        return get_voyage_client()

    if EMBED_PROVIDER == "azure-openai":
        if not (AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY and AZURE_OPENAI_EMBED_DEPLOYMENT):
            raise SystemExit(
                "RAG_EMBED_PROVIDER=azure-openai requires AZURE_OPENAI_ENDPOINT, "
                "AZURE_OPENAI_API_KEY, and AZURE_OPENAI_EMBED_DEPLOYMENT to be set."
            )
        import openai
        return openai.AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            timeout=EMBED_TIMEOUT_SECONDS,
        )

    if EMBED_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise SystemExit("RAG_EMBED_PROVIDER=openai requires OPENAI_API_KEY to be set.")
        import openai
        return openai.OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL, timeout=EMBED_TIMEOUT_SECONDS)

    raise SystemExit(
        f"Unknown RAG_EMBED_PROVIDER '{EMBED_PROVIDER}' -- expected 'voyage', 'azure-openai', or 'openai'."
    )


def embed_texts(client, texts: list, input_type: str, conn=None) -> list:
    """input_type is 'document' when embedding chunks to index, 'query' when
    embedding a search query -- meaningful for Voyage (asymmetric
    embeddings improve retrieval); ignored for azure-openai/openai, which
    have no such distinction. Returns pgvector.Vector objects, ready to bind
    as query parameters.

    Pass conn to record the real token usage this call reports (see
    record_usage) -- optional so callers that don't care about tracking
    (or don't have a connection handy) can skip it."""
    if EMBED_PROVIDER == "voyage":
        result = client.embed(texts, model=EMBED_MODEL, input_type=input_type, output_dimension=EMBED_DIM)
        vectors, tokens, model_label = result.embeddings, result.total_tokens, EMBED_MODEL
    else:
        # Azure routes by deployment name, passed as `model` here; vanilla
        # OpenAI (or an OpenAI-compatible endpoint) uses RAG_EMBED_MODEL.
        model_name = AZURE_OPENAI_EMBED_DEPLOYMENT if EMBED_PROVIDER == "azure-openai" else EMBED_MODEL
        response = client.embeddings.create(input=texts, model=model_name, dimensions=EMBED_DIM)
        ordered = sorted(response.data, key=lambda d: d.index)
        vectors = [d.embedding for d in ordered]
        tokens, model_label = response.usage.total_tokens, model_name

    if conn is not None:
        record_usage(conn, model_label, input_type, tokens)
    return [Vector(v) for v in vectors]


def rerank_texts(client: voyageai.Client, query: str, documents: list, top_k: int, conn=None) -> list:
    """Re-scores `documents` against `query` with Voyage's rerank API --
    more accurate than embedding-cosine-similarity alone, since it scores
    the actual query/document pair rather than comparing two independently
    computed vectors. Only call this when RERANK_MODEL is set.

    Returns (original_index, relevance_score) pairs, best match first."""
    result = client.rerank(query, documents, model=RERANK_MODEL, top_k=top_k)
    if conn is not None:
        record_usage(conn, RERANK_MODEL, "rerank", result.total_tokens)
    return [(r.index, r.relevance_score) for r in result.results]
