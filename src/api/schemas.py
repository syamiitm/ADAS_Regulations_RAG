"""Pydantic API contracts."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    models_ready: bool
    openai_ready: bool
    chat_ready: bool
    chat_base_url: str | None
    embedding_base_url: str | None
    indexed_documents: int
    vector_count: int
    uptime_seconds: float
    embedding_model: str
    chat_model: str
    vision_model: str
    # Non-secret diagnostics (paths / booleans only)
    dotenv_path: str
    dotenv_found: bool
    openai_key_set: bool
    vision_key_set: bool
    openrouter_or_chat_key_set: bool


class IngestResponse(BaseModel):
    """Ingest result; ``chunker`` identifies the chunking backend (catch stale servers)."""

    filename: str
    processing_seconds: float
    chunk_counts: dict[str, int]
    chunks_indexed: int
    chunker: str = "pymupdf_langchain"


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=8000)
    top_k: int | None = Field(default=None, ge=1, le=50)


class SourceRef(BaseModel):
    filename: str
    page: int | str
    chunk_type: str
    excerpt: str
    distance: float | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceRef]


class DocumentChunkStats(BaseModel):
    """Chunk type counts for one ingested document (``source_file``)."""

    source_file: str
    by_type: dict[str, int]
    total: int


class IndexStatsResponse(BaseModel):
    """Aggregated chunk counts from the in-memory FAISS metadata."""

    vector_count: int
    documents: list[DocumentChunkStats]
