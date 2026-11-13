# SemanticSearch

[![CI Pipeline](https://github.com/Soulofghost/semanticsearch/actions/workflows/ci.yml/badge.svg)](https://github.com/Soulofghost/semanticsearch/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Database](https://img.shields.io/badge/SQLite-WAL%20Mode-003B57.svg)](https://www.sqlite.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **SemanticSearch** is a production-grade, hybrid semantic and lexical document search engine. It fuses dense subword n-gram vector embeddings with Okapi BM25 keyword ranking to deliver high-precision information retrieval with sub-millisecond query latency and zero external API dependencies.

---

## 1. System Architecture

SemanticSearch utilizes a dual-encoder hybrid retrieval pipeline that balances semantic conceptual understanding against exact term matching.

```
┌─────────────────────────────────────────────────────────────┐
│                    Search Console (SPA)                     │
│   • Multi-mode selector (Hybrid / Semantic / Lexical)       │
│   • Live Alpha Blend Slider (0.0 to 1.0)                    │
│   • Dynamic Score Breakdown Gauge                           │
│   • Document Ingestion Drawer & Preset Corpora              │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP / JSON
┌──────────────────────────────▼──────────────────────────────┐
│                   FastAPI Application Tier                  │
│   • Ingestion & Chunker Controller                          │
│   • Query Preprocessing & Vectorization                     │
│   • Hybrid Linear & Rank Fusion Pipeline                    │
│   • Real-Time Query Telemetry & Latency Profiling           │
└──────────────────────────────┬──────────────────────────────┘
                               │
       ┌───────────────────────┴───────────────────────┐
       ▼                                               ▼
┌─────────────────────────────┐         ┌─────────────────────────────┐
│   Dense Vector Space (D=128)│         │     Okapi BM25 Lexical      │
│   • Hashed word & subword   │         │     • Inverted document     │
│     n-grams                 │         │       frequency (IDF)       │
│   • L2 Unit Normalization   │         │     • Length normalization  │
│   • Cosine Similarity       │         │     • Term frequency (TF)   │
└──────────────┬──────────────┘         └──────────────┬──────────────┘
               │                                       │
               └───────────────────┬───────────────────┘
                                   ▼
                 ┌───────────────────────────────────┐
                 │        Hybrid Fusion Ranker       │
                 │ Score = α · Sem + (1 - α) · Lex   │
                 └─────────────────┬─────────────────┘
                                   ▼
                 ┌───────────────────────────────────┐
                 │   SQLite 3 High-Throughput WAL    │
                 │   • Documents, Chunks, Queries    │
                 │   • Serialized float32 BLOBs      │
                 └───────────────────────────────────┘
```

---

## 2. Core Retrieval Mechanics

### A. Dense Subword Vector Embeddings ($D = 128$)
To guarantee 100% offline, reproducible, and zero-latency vector generation, the engine utilizes multi-scale feature hashing:
- Unigram and bigram word representations capture conceptual meaning.
- Character trigrams capture morphological variants and provide robustness against typographical errors.
- Vectors are $L_2$-normalized:
$$\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2}$$
Cosine similarity between query vector $\mathbf{q}$ and chunk vector $\mathbf{d}$ is computed as:
$$\text{Sim}_{\text{sem}}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d} + 1}{2}$$

### B. Okapi BM25 Term Weighting
Lexical matching evaluates keyword specificity across the indexed corpus:
$$\text{Score}_{\text{BM25}}(D, Q) = \sum_{q \in Q} \ln\left(1 + \frac{N - n(q) + 0.5}{n(q) + 0.5}\right) \cdot \frac{f(q, D) \cdot (k_1 + 1)}{f(q, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$
Parameters: $k_1 = 1.5, b = 0.75$. Raw BM25 scores are normalized across candidates to $[0, 1]$.

### C. Hybrid Linear Fusion
Users can modulate the blend between conceptual search and keyword search via the $\alpha$ tuning parameter:
$$\text{Score}_{\text{hybrid}} = \alpha \cdot \text{Score}_{\text{sem}} + (1 - \alpha) \cdot \text{Score}_{\text{lex}}$$

---

## 3. Technology Stack

| Layer | Component | Purpose |
| :--- | :--- | :--- |
| **Backend Framework** | FastAPI 0.110+ | Asynchronous RESTful search API |
| **Vector Computation** | NumPy 1.24+ | Vector algebra and float32 array operations |
| **Data Serialization** | Pydantic v2 | Type validation and strict JSON schemas |
| **Persistence** | SQLite 3 | WAL mode with byte-level BLOB storage |
| **Frontend** | Vanilla JS / CSS3 | Bespoke deep indigo / electric violet search interface |
| **Testing** | Pytest + HTTPX | Integration and retrieval ranking unit tests |
| **DevOps** | Docker / Compose | Multi-stage containerized deployment |
| **CI/CD** | GitHub Actions | Automated build, test, and quality validation |

---

## 4. API Endpoints

### Document Management
- `POST /api/documents/ingest` — Ingest, segment, and vectorize document into chunks.
- `GET /api/documents` — List indexed documents with chunk count and character statistics.
- `GET /api/documents/{id}` — Fetch document detail with all serialized chunks.
- `DELETE /api/documents/{id}` — Cascade delete document and associated chunks.

### Search Engine
- `POST /api/search` — Execute hybrid semantic retrieval.
  ```json
  {
    "query": "How does write ahead logging provide crash recovery?",
    "mode": "HYBRID",
    "alpha": 0.7,
    "top_k": 5
  }
  ```

### Telemetry & Health
- `GET /api/analytics/stats` — Overall corpus metrics (total docs, chunks, searches, avg latency).
- `GET /health` — Readiness and health probe.

---

## 5. Local Setup & Testing

### Installation
```bash
# Clone the repository
git clone https://github.com/Soulofghost/semanticsearch.git
cd semanticsearch

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt
```

### Running the Search Engine
```bash
uvicorn app.main:app --app-dir backend --reload --port 8000
```
Visit `http://localhost:8000` to access the neural search console.

### Running Test Suite
```bash
pytest backend/tests -v
```

---

## 6. Docker Deployment

```bash
docker-compose up --build -d
docker-compose logs -f
```

---

## 7. License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
