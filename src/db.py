"""
src/db.py

Shared Postgres/pgvector access for ingest.py and server.py: connection,
schema, and the embedding call (Voyage AI) both scripts need identically.

All chunks live in one `doc_chunks` table, namespaced by `collection` so
multiple doc sets can share the same database. The embedding dimension is
fixed per database (RAG_EMBED_DIM) -- switching to a model with a different
output dimension requires wiping and recreating the table.
"""

import os

import psycopg
import voyageai
from pgvector import Vector
from pgvector.psycopg import register_vector

DATABASE_URL = os.environ.get("RAG_DATABASE_URL", "postgresql://raguser:ragpass@localhost:5432/ragdb")
EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "voyage-4-large")
EMBED_DIM = int(os.environ.get("RAG_EMBED_DIM", "1024"))
VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY")


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
    if not VOYAGE_API_KEY:
        raise SystemExit("VOYAGE_API_KEY is not set. Get one at https://dash.voyageai.com and set it in your .env.")
    return voyageai.Client(api_key=VOYAGE_API_KEY)


def embed_texts(client: voyageai.Client, texts: list, input_type: str, conn=None) -> list:
    """input_type is 'document' when embedding chunks to index, 'query' when
    embedding a search query -- Voyage recommends both for best retrieval.
    Returns pgvector.Vector objects, ready to bind as query parameters.

    Pass conn to record the real token usage this call reports (see
    record_usage) -- optional so callers that don't care about tracking
    (or don't have a connection handy) can skip it."""
    result = client.embed(texts, model=EMBED_MODEL, input_type=input_type, output_dimension=EMBED_DIM)
    if conn is not None:
        record_usage(conn, EMBED_MODEL, input_type, result.total_tokens)
    return [Vector(e) for e in result.embeddings]
