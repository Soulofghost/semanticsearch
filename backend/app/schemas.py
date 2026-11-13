from pydantic import BaseModel, Field
from typing import List, Optional

class DocumentIngestRequest(BaseModel):
    title: str = Field(..., min_length=2, max_length=150)
    content: str = Field(..., min_length=20)
    source_type: Optional[str] = "text"
    author: Optional[str] = "Author"
    tags: Optional[str] = "general"
    chunk_size: Optional[int] = 500
    chunk_overlap: Optional[int] = 80

class DocumentChunkOut(BaseModel):
    id: str
    document_id: str
    chunk_index: int
    content: str
    token_count: int
    created_at: str

class DocumentListItemOut(BaseModel):
    id: str
    title: str
    source_type: str
    author: str
    tags: str
    total_chunks: int
    raw_length: int
    created_at: str

class DocumentDetailOut(DocumentListItemOut):
    chunks: List[DocumentChunkOut]

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=250)
    mode: Optional[str] = Field("HYBRID", pattern="^(HYBRID|SEMANTIC|LEXICAL)$")
    alpha: Optional[float] = Field(0.7, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(5, ge=1, le=50)
    tags_filter: Optional[str] = None

class SearchResultItem(BaseModel):
    chunk_id: str
    document_id: str
    document_title: str
    document_author: str
    chunk_index: int
    content: str
    snippet: str
    semantic_score: float
    lexical_score: float
    hybrid_score: float

class SearchResponse(BaseModel):
    query: str
    mode: str
    alpha: float
    top_k: int
    total_evaluated: int
    latency_ms: float
    results: List[SearchResultItem]

class AnalyticsStatsOut(BaseModel):
    total_documents: int
    total_chunks: int
    total_searches: int
    avg_latency_ms: float
