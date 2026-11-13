import time
import uuid
import re
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query

from app.database import get_db_connection
from app.schemas import (
    DocumentIngestRequest,
    DocumentListItemOut,
    DocumentDetailOut,
    DocumentChunkOut,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    AnalyticsStatsOut
)
from app.engine.search_engine import (
    compute_dense_embedding,
    bytes_to_vector,
    cosine_similarity,
    recursive_chunk_text,
    calculate_bm25_score,
    build_highlighted_snippet
)

router = APIRouter(prefix="/api", tags=["search-and-documents"])

@router.post("/documents/ingest", response_model=DocumentDetailOut, status_code=201)
def ingest_document(payload: DocumentIngestRequest):
    chunks = recursive_chunk_text(
        payload.content,
        target_size=payload.chunk_size or 500,
        overlap=payload.chunk_overlap or 80
    )

    if not chunks:
        raise HTTPException(status_code=400, detail="Document content could not be segmented into valid chunks.")

    doc_id = str(uuid.uuid4())
    raw_len = len(payload.content)
    total_chunks = len(chunks)

    conn = get_db_connection()
    with conn:
        conn.execute("""
            INSERT INTO documents (id, title, source_type, author, tags, total_chunks, raw_length)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (doc_id, payload.title, payload.source_type or "text", payload.author or "Author", payload.tags or "general", total_chunks, raw_len))

        for c in chunks:
            chunk_id = str(uuid.uuid4())
            conn.execute("""
                INSERT INTO document_chunks (id, document_id, chunk_index, content, embedding_blob, token_count)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (chunk_id, doc_id, c["chunk_index"], c["content"], c["embedding_blob"], c["token_count"]))

    return get_document_detail(doc_id)

@router.get("/documents", response_model=List[DocumentListItemOut])
def list_documents(q: Optional[str] = Query(None, description="Filter by title or tags")):
    conn = get_db_connection()
    query = "SELECT id, title, source_type, author, tags, total_chunks, raw_length, created_at FROM documents"
    params = []
    if q:
        query += " WHERE title LIKE ? OR tags LIKE ?"
        term = f"%{q}%"
        params.extend([term, term])
    query += " ORDER BY created_at DESC"

    rows = conn.execute(query, params).fetchall()
    return [DocumentListItemOut(**dict(r)) for r in rows]

@router.get("/documents/{document_id}", response_model=DocumentDetailOut)
def get_document_detail(document_id: str):
    conn = get_db_connection()
    doc = conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    chunk_rows = conn.execute("SELECT id, document_id, chunk_index, content, token_count, created_at FROM document_chunks WHERE document_id = ? ORDER BY chunk_index ASC", (document_id,)).fetchall()

    return DocumentDetailOut(
        id=doc["id"],
        title=doc["title"],
        source_type=doc["source_type"],
        author=doc["author"],
        tags=doc["tags"] or "",
        total_chunks=doc["total_chunks"],
        raw_length=doc["raw_length"],
        created_at=str(doc["created_at"]),
        chunks=[DocumentChunkOut(**dict(r)) for r in chunk_rows]
    )

@router.delete("/documents/{document_id}", status_code=204)
def delete_document(document_id: str):
    conn = get_db_connection()
    with conn:
        res = conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        if res.rowcount == 0:
            raise HTTPException(status_code=404, detail="Document not found")
    return None

@router.post("/search", response_model=SearchResponse)
def execute_search(payload: SearchRequest):
    start_time = time.perf_counter()
    conn = get_db_connection()

    # Retrieve all candidate chunks with parent document metadata
    query_sql = """
        SELECT c.id as chunk_id, c.document_id, c.chunk_index, c.content, c.embedding_blob, c.token_count,
               d.title as doc_title, d.author as doc_author, d.tags as doc_tags
        FROM document_chunks c
        JOIN documents d ON c.document_id = d.id
    """
    params = []
    if payload.tags_filter:
        query_sql += " WHERE d.tags LIKE ?"
        params.append(f"%{payload.tags_filter}%")

    candidates = conn.execute(query_sql, params).fetchall()
    total_evaluated = len(candidates)

    if total_evaluated == 0:
        latency = round((time.perf_counter() - start_time) * 1000, 2)
        return SearchResponse(
            query=payload.query,
            mode=payload.mode,
            alpha=payload.alpha,
            top_k=payload.top_k,
            total_evaluated=0,
            latency_ms=latency,
            results=[]
        )

    # 1. Query Vector Embedding
    q_vec = compute_dense_embedding(payload.query)
    q_tokens = re.findall(r'\b[a-z0-9_-]+\b', payload.query.lower())

    # 2. Build BM25 Corpus Statistics
    total_docs = total_evaluated
    total_len = sum(c["token_count"] for c in candidates)
    avgdl = total_len / max(total_docs, 1)

    df_dict = {}
    for c in candidates:
        tokens_in_chunk = set(re.findall(r'\b[a-z0-9_-]+\b', c["content"].lower()))
        for t in tokens_in_chunk:
            df_dict[t] = df_dict.get(t, 0) + 1

    scored_items = []
    raw_bm25_scores = []

    # First pass: calculate scores
    for c in candidates:
        # Semantic Score (Cosine similarity)
        c_vec = bytes_to_vector(c["embedding_blob"])
        sem_score = cosine_similarity(q_vec, c_vec)

        # Lexical Score (Okapi BM25)
        bm25_raw = calculate_bm25_score(q_tokens, c["content"], avgdl, total_docs, df_dict)
        raw_bm25_scores.append(bm25_raw)

        scored_items.append({
            "chunk_id": c["chunk_id"],
            "document_id": c["document_id"],
            "document_title": c["doc_title"],
            "document_author": c["doc_author"],
            "chunk_index": c["chunk_index"],
            "content": c["content"],
            "semantic_score": sem_score,
            "raw_lexical": bm25_raw
        })

    # Normalize BM25 scores to [0.0, 1.0]
    max_bm25 = max(raw_bm25_scores) if raw_bm25_scores else 0.0

    final_results = []
    alpha = payload.alpha if payload.alpha is not None else 0.7
    mode = payload.mode or "HYBRID"

    for item in scored_items:
        norm_lex = (item["raw_lexical"] / max_bm25) if max_bm25 > 1e-6 else (1.0 if item["raw_lexical"] > 0 else 0.0)
        sem = item["semantic_score"]

        if mode == "SEMANTIC":
            hybrid = sem
        elif mode == "LEXICAL":
            hybrid = norm_lex
        else: # HYBRID
            hybrid = (alpha * sem) + ((1.0 - alpha) * norm_lex)

        snippet = build_highlighted_snippet(item["content"], q_tokens)

        final_results.append(SearchResultItem(
            chunk_id=item["chunk_id"],
            document_id=item["document_id"],
            document_title=item["document_title"],
            document_author=item["document_author"],
            chunk_index=item["chunk_index"],
            content=item["content"],
            snippet=snippet,
            semantic_score=round(sem, 4),
            lexical_score=round(norm_lex, 4),
            hybrid_score=round(hybrid, 4)
        ))

    # Sort descending by hybrid_score
    final_results.sort(key=lambda x: x.hybrid_score, reverse=True)
    top_results = final_results[:payload.top_k]

    latency = round((time.perf_counter() - start_time) * 1000, 2)

    # Log query in search_queries table
    with conn:
        conn.execute("""
            INSERT INTO search_queries (id, query_text, search_mode, alpha, top_k, total_results, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (str(uuid.uuid4()), payload.query, mode, alpha, payload.top_k, len(top_results), latency))

    return SearchResponse(
        query=payload.query,
        mode=mode,
        alpha=alpha,
        top_k=payload.top_k,
        total_evaluated=total_evaluated,
        latency_ms=latency,
        results=top_results
    )

@router.get("/analytics/stats", response_model=AnalyticsStatsOut)
def get_analytics_stats():
    conn = get_db_connection()
    total_docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    total_chunks = conn.execute("SELECT COUNT(*) FROM document_chunks").fetchone()[0]
    total_searches = conn.execute("SELECT COUNT(*) FROM search_queries").fetchone()[0]
    avg_latency = conn.execute("SELECT AVG(latency_ms) FROM search_queries").fetchone()[0] or 0.0

    return AnalyticsStatsOut(
        total_documents=total_docs,
        total_chunks=total_chunks,
        total_searches=total_searches,
        avg_latency_ms=round(avg_latency, 2)
    )
