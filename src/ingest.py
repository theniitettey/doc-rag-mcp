"""Chunks and embeds docs (.pdf, .md, .txt, or a single file) into
Postgres/pgvector. Incremental by default -- unchanged files are skipped by
content hash. See README for --rebuild, env vars, and details.

    python src/ingest.py --docs ./docs --collection project_docs
"""

import argparse
import hashlib
import os
import re
from pathlib import Path

import pymupdf

import db

CHUNK_SIZE = int(os.environ.get("RAG_CHUNK_SIZE", "800"))       # characters per chunk (rough proxy for tokens)
CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", "150"))  # overlap between consecutive chunks


def read_pdf(path: Path) -> str:
    text_parts = []
    with pymupdf.open(path) as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            if text.strip():
                text_parts.append(f"\n\n[page {page_num}]\n{text}")
    return "".join(text_parts)


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return read_pdf(path)
    return read_text_file(path)


def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Simple sliding-window chunker that tries to break on paragraph/sentence
    boundaries near the target size, rather than mid-word."""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            # try to snap to the nearest paragraph or sentence break
            window = text[start:end]
            break_pos = max(window.rfind("\n\n"), window.rfind(". "))
            if break_pos > chunk_size * 0.4:  # don't shrink chunks too much
                end = start + break_pos + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(end - overlap, end) if overlap >= (end - start) else end - overlap
        if start <= 0:
            start = end
    return chunks


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def insert_chunk_rows(conn, collection: str, rel_path: str, file_hash_value: str, chunk_rows: list):
    """chunk_rows: list of (chunk_index, content, embedding), already
    computed -- either freshly embedded or reused from identical content
    stored elsewhere (see find_duplicate_content)."""
    with conn.cursor() as cur:
        for chunk_index, content, embedding in chunk_rows:
            chunk_id = hashlib.sha1(f"{collection}::{rel_path}::{chunk_index}".encode("utf-8")).hexdigest()
            cur.execute(
                """
                INSERT INTO doc_chunks (id, collection, source, chunk_index, file_hash, content, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    file_hash = EXCLUDED.file_hash,
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding
                """,
                (chunk_id, collection, rel_path, chunk_index, file_hash_value, content, embedding),
            )


def embed_chunk_rows(conn, embed_client, chunks: list, batch_size: int = 128) -> list:
    """Embeds `chunks` via whichever provider RAG_EMBED_PROVIDER configures,
    batching. Returns (chunk_index, content, embedding) rows ready for
    insert_chunk_rows. Records token usage against conn as each batch call
    returns -- call this outside any transaction you plan to roll back on
    later failure, so the usage record (tokens already spent, real API
    calls already made) survives regardless."""
    rows = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        embeddings = db.embed_texts(embed_client, batch, input_type="document", conn=conn)
        rows.extend((i + j, chunk, embedding) for j, (chunk, embedding) in enumerate(zip(batch, embeddings)))
    return rows


def find_duplicate_content(conn, collection: str, file_hash_value: str):
    """Look for chunks already stored under this exact content hash --
    possibly a different file (byte-identical duplicate under another
    filename), including one inserted earlier in this same run. If found,
    their (chunk_index, content, embedding) can be reused verbatim instead
    of re-calling the embedding API for content we've already embedded."""
    rows = conn.execute(
        """
        SELECT DISTINCT ON (chunk_index) chunk_index, content, embedding
        FROM doc_chunks
        WHERE collection = %s AND file_hash = %s
        ORDER BY chunk_index
        """,
        (collection, file_hash_value),
    ).fetchall()
    return rows or None


SUPPORTED_EXT = {".pdf", ".md", ".markdown", ".txt"}


def collect_files(docs_path: Path):
    """docs_path may be a single file (index just that one doc) or a folder
    (the usual case: recursively index every supported file in it). Returns
    (base_dir, files) -- base_dir is what `source` paths get computed
    relative to, so a single file's source is just its own filename."""
    if docs_path.is_file():
        if docs_path.suffix.lower() not in SUPPORTED_EXT:
            raise SystemExit(f"{docs_path} is not a supported file type ({SUPPORTED_EXT})")
        return docs_path.parent, [docs_path]
    if docs_path.is_dir():
        files = [p for p in docs_path.rglob("*") if p.suffix.lower() in SUPPORTED_EXT]
        return docs_path, files
    raise SystemExit(f"{docs_path} does not exist -- set RAG_DOCS_DIR to an existing file or folder")


def build_index(docs_path: Path, collection: str, force_rebuild: bool = False, log=print,
                 confirm_large_removal: bool = False):
    """log: called with one line of progress text at a time -- defaults to
    print() for CLI use; the MCP reindex_docs tool passes a callback that
    captures lines to return as the tool result instead."""
    base_dir, files = collect_files(docs_path)

    log(f"Found {len(files)} document(s). Using {db.EMBED_PROVIDER} embedding model '{db.EMBED_MODEL}'...")
    embed_client = db.get_embed_client()
    conn = db.get_connection()

    # Stored per collection so a later run can tell whether the index was
    # built with settings that made these embeddings comparable to new ones.
    run_config = {
        "embed_model": db.EMBED_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    }
    stored_config = conn.execute(
        "SELECT embed_model, chunk_size, chunk_overlap FROM collection_config WHERE collection = %s",
        (collection,),
    ).fetchone()

    if stored_config is not None and not force_rebuild:
        if tuple(run_config.values()) != stored_config:
            log("Embedding model or chunk settings changed since the last run — "
                  "forcing a full rebuild (old embeddings aren't comparable to new ones).")
            force_rebuild = True

    if force_rebuild:
        conn.execute("DELETE FROM doc_chunks WHERE collection = %s", (collection,))
        stored_config = None

    conn.execute(
        """
        INSERT INTO collection_config (collection, embed_model, chunk_size, chunk_overlap)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (collection) DO UPDATE SET
            embed_model = EXCLUDED.embed_model,
            chunk_size = EXCLUDED.chunk_size,
            chunk_overlap = EXCLUDED.chunk_overlap
        """,
        (collection, run_config["embed_model"], run_config["chunk_size"], run_config["chunk_overlap"]),
    )

    existing_hashes = {}
    for source, file_hash_value in conn.execute(
        "SELECT DISTINCT ON (source) source, file_hash FROM doc_chunks WHERE collection = %s",
        (collection,),
    ).fetchall():
        existing_hashes[source] = file_hash_value

    current_sources = set()
    added_count, updated_count, skipped_count, failed_count = 0, 0, 0, 0

    for path in files:
        rel_path = str(path.relative_to(base_dir))
        current_sources.add(rel_path)
        try:
            hash_value = file_hash(path)
        except Exception as e:
            log(f"  ! Failed to hash {rel_path}: {e}")
            failed_count += 1
            continue

        if existing_hashes.get(rel_path) == hash_value:
            skipped_count += 1
            continue

        is_update = rel_path in existing_hashes

        # Byte-identical content under a different filename (this run or a
        # previous one) -- reuse its embeddings instead of re-calling the
        # API for content we've already embedded.
        duplicate_rows = find_duplicate_content(conn, collection, hash_value)
        rows = duplicate_rows  # None unless a duplicate was found

        if duplicate_rows is not None:
            log(f"  {'Updating' if is_update else 'Reading'} {rel_path} ... "
                  f"duplicate content of an already-indexed file -- reusing embeddings, no API call")
        else:
            log(f"  {'Updating' if is_update else 'Reading'} {rel_path} ...")
            try:
                text = load_document(path)
            except Exception as e:
                log(f"    ! Failed to read {rel_path}: {e}")
                failed_count += 1
                continue
            chunks = chunk_text(text)

        # Embedding happens outside the transaction below on purpose: those
        # tokens are spent (usage recorded) the moment the API call
        # returns, regardless of whether the DB write that follows
        # succeeds -- rolling back the write shouldn't roll back the fact
        # that the call happened and cost tokens.
        #
        # The write itself (delete + insert) IS atomic per file: if this is
        # interrupted (Ctrl+C, a network blip) partway through, the
        # transaction rolls back so no partial chunk set is left behind
        # under this file's hash -- otherwise a future run would see the
        # hash unchanged and skip it, silently leaving it half-indexed
        # forever.
        try:
            if rows is None:
                rows = embed_chunk_rows(conn, embed_client, chunks)
            with conn.transaction():
                if is_update:
                    conn.execute("DELETE FROM doc_chunks WHERE collection = %s AND source = %s",
                                 (collection, rel_path))
                insert_chunk_rows(conn, collection, rel_path, hash_value, rows)
        except Exception as e:
            log(f"    ! Failed to embed/store {rel_path}: {e}")
            failed_count += 1
            continue

        log(f"    -> {len(rows)} chunk(s)")
        updated_count += 1 if is_update else 0
        added_count += 0 if is_update else 1

    removed_sources = set(existing_hashes) - current_sources
    # A large fraction of previously-indexed files "disappearing" in one run
    # is much more often a misconfigured/mismatched RAG_DOCS_DIR (wrong
    # folder, an empty mount) than genuine mass deletion -- refuse to act on
    # it silently. Small removals (typical day-to-day doc churn) go through
    # as before.
    removal_threshold = max(3, len(existing_hashes) * 0.5)
    if removed_sources and len(removed_sources) >= removal_threshold and not confirm_large_removal:
        log(f"! Refusing to remove {len(removed_sources)} of {len(existing_hashes)} previously-indexed "
            f"file(s) automatically -- this usually means the docs path points somewhere unexpected "
            f"rather than genuine deletions. Re-run with --confirm-removal (CLI) or "
            f"confirm_large_removal=True (reindex_docs) if this is intentional. Not removed: "
            f"{', '.join(sorted(removed_sources))}")
        removed_sources = set()
    else:
        for rel_path in removed_sources:
            conn.execute("DELETE FROM doc_chunks WHERE collection = %s AND source = %s", (collection, rel_path))

    total_chunks = conn.execute(
        "SELECT count(*) FROM doc_chunks WHERE collection = %s", (collection,)
    ).fetchone()[0]

    log(f"Done. {added_count} new file(s), {updated_count} changed file(s), "
          f"{skipped_count} unchanged (skipped), {len(removed_sources)} removed"
          + (f", {failed_count} failed" if failed_count else "") + ".")
    log(f"Collection '{collection}' now has {total_chunks} chunks stored in Postgres.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a RAG index from a docs folder into Postgres/pgvector.")
    parser.add_argument("--docs", default=os.environ.get("RAG_DOCS_DIR", "./docs"),
                         help="Folder containing your docs/PDFs, or a path to a single doc file")
    parser.add_argument("--collection", default=os.environ.get("RAG_COLLECTION", "project_docs"),
                         help="Logical collection name (namespaces rows within the shared table)")
    parser.add_argument("--rebuild", action="store_true",
                         default=os.environ.get("RAG_FORCE_REBUILD", "").lower() in ("1", "true", "yes"),
                         help="Force a full re-embed of every file, ignoring stored file hashes")
    parser.add_argument("--confirm-removal", action="store_true",
                         default=os.environ.get("RAG_CONFIRM_REMOVAL", "").lower() in ("1", "true", "yes"),
                         help="Allow removing a large fraction of previously-indexed files in one run "
                              "(otherwise refused as a likely misconfigured docs path)")
    args = parser.parse_args()

    build_index(Path(args.docs).resolve(), args.collection, force_rebuild=args.rebuild,
                confirm_large_removal=args.confirm_removal)
