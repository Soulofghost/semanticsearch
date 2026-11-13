import sqlite3
import os

def get_db_path() -> str:
    return os.getenv("SEMANTICSEARCH_DB_PATH", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "semanticsearch.db")))

def get_db_connection(db_path: str = None):
    target_path = db_path or get_db_path()
    is_uri = target_path.startswith("file:") or "mode=memory" in target_path
    if is_uri or target_path == ":memory:":
        conn = sqlite3.connect(target_path, check_same_thread=False, uri=is_uri)
    else:
        conn = sqlite3.connect(target_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db(db_path: str = None):
    conn = get_db_connection(db_path)
    with conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                source_type TEXT DEFAULT 'text',
                author TEXT DEFAULT 'Anonymous',
                tags TEXT,
                total_chunks INTEGER DEFAULT 0,
                raw_length INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS document_chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding_blob BLOB NOT NULL,
                token_count INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS search_queries (
                id TEXT PRIMARY KEY,
                query_text TEXT NOT NULL,
                search_mode TEXT NOT NULL,
                alpha REAL NOT NULL,
                top_k INTEGER NOT NULL,
                total_results INTEGER NOT NULL,
                latency_ms REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_document ON document_chunks(document_id);
            CREATE INDEX IF NOT EXISTS idx_queries_created ON search_queries(created_at);
        """)
    return conn
