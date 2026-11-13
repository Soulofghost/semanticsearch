import numpy as np
import pytest
from app.engine.search_engine import (
    compute_dense_embedding,
    vector_to_bytes,
    bytes_to_vector,
    cosine_similarity,
    recursive_chunk_text,
    calculate_bm25_score,
    build_highlighted_snippet,
    VECTOR_DIM
)

def test_dense_embedding_properties():
    text1 = "Distributed consensus algorithm with Raft and Paxos"
    vec1 = compute_dense_embedding(text1)

    assert isinstance(vec1, np.ndarray)
    assert vec1.shape == (VECTOR_DIM,)
    assert vec1.dtype == np.float32

    # Verify L2 unit norm
    norm = np.linalg.norm(vec1)
    assert abs(norm - 1.0) < 1e-4

    # Identical texts must yield identical embeddings
    vec2 = compute_dense_embedding(text1)
    np.testing.assert_allclose(vec1, vec2)

def test_vector_bytes_roundtrip():
    vec = compute_dense_embedding("Test serialization")
    blob = vector_to_bytes(vec)
    assert isinstance(blob, bytes)
    assert len(blob) == VECTOR_DIM * 4  # 128 * 4 bytes for float32

    restored = bytes_to_vector(blob)
    np.testing.assert_allclose(vec, restored)

def test_cosine_similarity():
    v1 = compute_dense_embedding("Database replication and fault tolerance")
    v2 = compute_dense_embedding("Database replication and fault tolerance")
    v3 = compute_dense_embedding("A completely unrelated cooking recipe for chocolate cake")

    sim_identical = cosine_similarity(v1, v2)
    assert abs(sim_identical - 1.0) < 1e-4

    sim_diff = cosine_similarity(v1, v3)
    assert 0.0 <= sim_diff <= 1.0
    assert sim_identical > sim_diff

def test_recursive_chunk_text():
    sample_text = """
    The Raft consensus protocol was designed to be understandable and safe.
    It decomposes consensus into leader election, log replication, and safety invariants.

    In the leader election phase, nodes begin as followers. If no heartbeat is received
    within the election timeout, a follower transitions to candidate state and requests votes.

    Once a candidate receives a majority of votes, it transitions to leader state and begins
    servicing client requests, appending entries to its local log and replicating them to peers.
    """
    chunks = recursive_chunk_text(sample_text, target_size=200, overlap=40)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c["content"]) > 0
        assert "embedding" in c
        assert "embedding_blob" in c
        assert c["token_count"] > 0

def test_bm25_calculation():
    q_tokens = ["consensus", "raft"]
    chunk_hit = "The Raft protocol provides distributed consensus across clusters."
    chunk_miss = "Frontend styling with CSS grid and flexbox animations."

    df_dict = {"consensus": 1, "raft": 1}
    avgdl = 10.0
    total_docs = 2

    score_hit = calculate_bm25_score(q_tokens, chunk_hit, avgdl, total_docs, df_dict)
    score_miss = calculate_bm25_score(q_tokens, chunk_miss, avgdl, total_docs, df_dict)

    assert score_hit > 0.0
    assert score_miss == 0.0
    assert score_hit > score_miss

def test_highlighted_snippet():
    content = "The Paxos protocol guarantees safety under asynchronous network partitions."
    q_tokens = ["safety", "partitions"]
    snippet = build_highlighted_snippet(content, q_tokens)
    assert len(snippet) > 0
    assert "guarantees" in snippet or "partitions" in snippet
