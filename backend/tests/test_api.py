import os
import pytest
from fastapi.testclient import TestClient

TEST_DB = "file:semanticsearch_test?mode=memory&cache=shared"
os.environ["SEMANTICSEARCH_DB_PATH"] = TEST_DB

from app.database import init_db, get_db_connection
from app.main import app

client = TestClient(app)

holding_conn = None

@pytest.fixture(scope="session", autouse=True)
def setup_database_session():
    global holding_conn
    holding_conn = init_db(TEST_DB)
    yield
    if holding_conn:
        holding_conn.close()

SAMPLE_DOC = """
Modern high-performance storage engines rely on Write-Ahead Logging (WAL) to provide atomicity and durability.
Before changes are applied to the primary database page cache, a sequential write is committed to the append-only log.

During system recovery after an unexpected failure, the database replays log transactions starting from the last known checkpoint.
This guarantees zero data loss and prevents partial updates from corrupting tables.
"""

def test_health_check():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"
    assert res.json()["service"] == "SemanticSearch"

def test_document_ingest_and_crud_lifecycle():
    # 1. Ingest document
    payload = {
        "title": "Write-Ahead Logging Architecture",
        "content": SAMPLE_DOC,
        "author": "Systems Architect",
        "tags": "database,storage,durability",
        "chunk_size": 300,
        "chunk_overlap": 50
    }
    create_res = client.post("/api/documents/ingest", json=payload)
    assert create_res.status_code == 201
    doc = create_res.json()
    doc_id = doc["id"]
    assert doc["title"] == payload["title"]
    assert doc["total_chunks"] >= 2
    assert len(doc["chunks"]) == doc["total_chunks"]

    # 2. Get Document Detail
    get_res = client.get(f"/api/documents/{doc_id}")
    assert get_res.status_code == 200
    assert get_res.json()["id"] == doc_id

    # 3. List Documents
    list_res = client.get("/api/documents?q=Architecture")
    assert list_res.status_code == 200
    items = list_res.json()
    assert len(items) >= 1
    assert items[0]["id"] == doc_id

    # 4. Search - Hybrid Mode
    search_payload = {
        "query": "durability log checkpoint recovery",
        "mode": "HYBRID",
        "alpha": 0.7,
        "top_k": 3
    }
    search_res = client.post("/api/search", json=search_payload)
    assert search_res.status_code == 200
    s_data = search_res.json()
    assert s_data["total_evaluated"] >= 2
    assert len(s_data["results"]) >= 1
    top_hit = s_data["results"][0]
    assert top_hit["document_id"] == doc_id
    assert top_hit["hybrid_score"] > 0.0
    assert "recovery" in top_hit["content"].lower() or "log" in top_hit["content"].lower()

    # 5. Search - Semantic Mode
    semantic_search = client.post("/api/search", json={
        "query": "preventing corruption and system crashes",
        "mode": "SEMANTIC",
        "top_k": 2
    })
    assert semantic_search.status_code == 200
    assert len(semantic_search.json()["results"]) >= 1

    # 6. Analytics Stats
    stats_res = client.get("/api/analytics/stats")
    assert stats_res.status_code == 200
    stats = stats_res.json()
    assert stats["total_documents"] >= 1
    assert stats["total_chunks"] >= 2
    assert stats["total_searches"] >= 2

    # 7. Delete Document
    del_res = client.delete(f"/api/documents/{doc_id}")
    assert del_res.status_code == 204

    # Verify 404
    verify_del = client.get(f"/api/documents/{doc_id}")
    assert verify_del.status_code == 404

def test_validation_errors():
    # Content too short (< 20 characters)
    res = client.post("/api/documents/ingest", json={"title": "Test", "content": "Too short"})
    assert res.status_code == 422
