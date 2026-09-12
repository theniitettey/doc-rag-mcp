"""
src/db.py

Shared Postgres/pgvector access for ingest.py and server.py: connection,
schema, and the embedding call both scripts need identically.

All chunks live in one `doc_chunks` table, namespaced by `collection` so
multiple doc sets can share the same database. The embedding dimension is
fixed per database (RAG_EMBED_DIM) -- switching to a model with a different
output dimension requires wiping and recreating the table.

Embedding provider is pluggable via RAG_EMBED_PROVIDER:
  voyage        (default) Voyage AI's own API.
  azure-openai  Azure OpenAI embeddings deployment.
  openai        Vanilla OpenAI, or any OpenAI-compatible endpoint
                (Ollama, vLLM, ...) via OPENAI_BASE_URL.
Reranking (RAG_RERANK_MODEL) always uses Voyage's rerank API regardless of
the embedding provider -- there's no equivalent on Azure/OpenAI, and the two
are otherwise unrelated (you can embed via Azure and still rerank via
Voyage's free tier, or skip reranking entirely).
"""

import os

import psycopg
import voyageai
from pgvector import Vector
from pgvector.psycopg import register_vector

DATABASE_URL = os.environ.get("RAG_DATABASE_URL", "postgresql://raguser:ragpass@localhost:5432/ragdb")
EMBED_PROVIDER = os.environ.get("RAG_EMBED_PROVIDER", "voyage").lower()
EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "voyage-4-large")
EMBED_DIM = int(os.environ.get("RAG_EMBED_DIM", "1024"))
VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY")
# Empty (default) = reranking disabled. Opt-in via RAG_RERANK_MODEL since it
# costs its own tokens on top of the embedding call, on every query.
RERANK_MODEL = os.environ.get("RAG_RERANK_MODEL", "")

# Azure OpenAI (RAG_EMBED_PROVIDER=azure-openai)
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")
AZURE_OPENAI_EMBED_DEPLOYMENT = os.environ.get("AZURE_OPENAI_EMBED_DEPLOYMENT")

# Vanilla OpenAI or an OpenAI-compatible endpoint (RAG_EMBED_PROVIDER=openai)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")  # unset = api.openai.com


def get_connection():
    conn = psycopg.connect(DATABASE_URL, autocommit=True)
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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS collection_config (
            collection TEXT PRIMARY KEY,
            embed_model TEXT NOT NULL,
            chunk_size INT NOT NULL,
            chunk_overlap INT NOT NULL
        )
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
    """Always Voyage, regardless of RAG_EMBED_PROVIDER -- used directly for
    reranking, and internally when RAG_EMBED_PROVIDER=voyage."""
    if not VOYAGE_API_KEY:
        raise SystemExit("VOYAGE_API_KEY is not set. Get one at https://dash.voyageai.com and set it in your .env.")
    return voyageai.Client(api_key=VOYAGE_API_KEY)


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
        )

    if EMBED_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise SystemExit("RAG_EMBED_PROVIDER=openai requires OPENAI_API_KEY to be set.")
        import openai
        return openai.OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

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
