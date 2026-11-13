import math
import re
import struct
import numpy as np
from typing import List, Dict, Any, Tuple

VECTOR_DIM = 128

def hash_token(token: str, seed: int = 0) -> int:
    h = 2166136261 ^ seed
    for ch in token:
        h = ((h * 16777619) ^ ord(ch)) & 0xFFFFFFFF
    return h

def compute_dense_embedding(text: str, dim: int = VECTOR_DIM) -> np.ndarray:
    """
    Computes a 128-dimensional dense semantic embedding using multi-scale
    word & character n-gram feature hashing with L2 unit normalization.
    """
    vec = np.zeros(dim, dtype=np.float32)
    clean_text = text.lower().strip()
    words = re.findall(r'\b[a-z0-9_-]+\b', clean_text)
    if not words:
        return vec

    # 1. Unigram & Bigram word hashing
    for i, w in enumerate(words):
        h1 = hash_token(w) % dim
        sign1 = 1.0 if (hash_token(w, 13) % 2 == 0) else -1.0
        vec[h1] += sign1 * 1.5

        if i < len(words) - 1:
            bigram = f"{w}_{words[i+1]}"
            h2 = hash_token(bigram) % dim
            sign2 = 1.0 if (hash_token(bigram, 17) % 2 == 0) else -1.0
            vec[h2] += sign2 * 2.0

    # 2. Subword character trigrams for semantic robustness to typos & morphology
    for w in words:
        if len(w) >= 3:
            for j in range(len(w) - 2):
                trigram = w[j:j+3]
                ht = hash_token(trigram, 41) % dim
                signt = 1.0 if (hash_token(trigram, 23) % 2 == 0) else -1.0
                vec[ht] += signt * 0.6

    # L2 Normalization
    norm = np.linalg.norm(vec)
    if norm > 1e-6:
        vec /= norm
    return vec

def vector_to_bytes(vec: np.ndarray) -> bytes:
    return vec.astype(np.float32).tobytes()

def bytes_to_vector(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    dot = float(np.dot(v1, v2))
    # Rescale [-1, 1] cosine range to [0, 1]
    sim = (dot + 1.0) / 2.0
    return max(0.0, min(1.0, sim))

def recursive_chunk_text(text: str, target_size: int = 500, overlap: int = 80) -> List[Dict[str, Any]]:
    """
    Recursively segments text into semantic paragraphs and sentence chunks.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    raw_blocks = []
    
    if len(paragraphs) > 1:
        raw_blocks = paragraphs
    else:
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        raw_blocks = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_content = ""

    for block in raw_blocks:
        if not current_content:
            current_content = block
        elif len(current_content) + len(block) + 1 <= target_size:
            current_content += "\n" + block
        else:
            chunks.append(current_content)
            # Retain overlap from end of previous chunk if possible
            if overlap > 0 and len(current_content) > overlap:
                overlap_text = current_content[-overlap:].strip()
                current_content = overlap_text + "\n" + block
            else:
                current_content = block

    if current_content:
        chunks.append(current_content)

    results = []
    for idx, c in enumerate(chunks):
        c_clean = c.strip()
        tokens = len(re.findall(r'\b\w+\b', c_clean))
        emb = compute_dense_embedding(c_clean)
        results.append({
            "chunk_index": idx,
            "content": c_clean,
            "token_count": tokens,
            "embedding": emb,
            "embedding_blob": vector_to_bytes(emb)
        })

    return results

def calculate_bm25_score(query_tokens: List[str], chunk_content: str, avgdl: float, total_docs: int, df_dict: Dict[str, int], k1: float = 1.5, b: float = 0.75) -> float:
    """
    Okapi BM25 term weighting score
    """
    chunk_tokens = re.findall(r'\b[a-z0-9_-]+\b', chunk_content.lower())
    doc_len = len(chunk_tokens)
    if doc_len == 0:
        return 0.0

    tf_dict = {}
    for t in chunk_tokens:
        tf_dict[t] = tf_dict.get(t, 0) + 1

    score = 0.0
    for q in query_tokens:
        if q not in tf_dict:
            continue
        tf = tf_dict[q]
        df = df_dict.get(q, 1)
        # Standard Lucene/BM25 IDF
        idf = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))
        numerator = tf * (k1 + 1.0)
        denominator = tf + k1 * (1.0 - b + b * (doc_len / max(avgdl, 1.0)))
        score += idf * (numerator / max(denominator, 1e-5))

    return max(0.0, score)

def build_highlighted_snippet(content: str, query_tokens: List[str], max_len: int = 220) -> str:
    """
    Extracts the highest relevance window and highlights query terms.
    """
    lower_content = content.lower()
    best_pos = 0
    max_hits = 0

    words = re.findall(r'\b[a-z0-9_-]+\b', lower_content)
    for q in query_tokens:
        pos = lower_content.find(q)
        if pos != -1:
            best_pos = max(0, pos - 40)
            break

    snippet = content[best_pos:best_pos + max_len].strip()
    if best_pos > 0:
        snippet = "..." + snippet
    if best_pos + max_len < len(content):
        snippet += "..."

    return snippet
