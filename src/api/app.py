"""FastAPI application: / → /docs, /health, /ingest, /query, /stats."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

# Project root: src/api/app.py -> parents[2] == repo root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.responses import Response

from src import config
from src.api import state as app_state
from src.api.schemas import (
    DocumentChunkStats,
    HealthResponse,
    IndexStatsResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceRef,
)
from src.models.openai_client import (
    chat_ready,
    embeddings_ready,
    get_chat_client,
    get_embedding_client,
    get_vision_client,
    models_ready,
)
from src.retrieval.query import run_query

logger = logging.getLogger(__name__)


def _http_exception_from_upstream(exc: Exception) -> HTTPException:
    """Map OpenAI-compatible SDK errors to clearer HTTP statuses (e.g. quota → 429)."""
    msg = str(exc)
    try:
        from openai import APIStatusError, RateLimitError

        if isinstance(exc, RateLimitError) or (
            isinstance(exc, APIStatusError)
            and getattr(exc, "status_code", None) == 429
        ):
            detail = (
                "Upstream API returned 429 (rate limit or insufficient quota). "
                "If you use OpenAI: add billing/credits or see "
                "https://platform.openai.com/docs/guides/error-codes/api-errors . "
                "If embeddings hit OpenAI, switch to OpenRouter Nemotron in .env "
                "(CHAT_BASE_URL + OPENROUTER_API_KEY, default EMBEDDING_MODEL). "
                "To ingest without figure captions, set SKIP_VLM=true. "
                f"Details: {msg}"
            )
            return HTTPException(status_code=429, detail=detail)
        if isinstance(exc, APIStatusError):
            sc = getattr(exc, "status_code", None)
            if sc == 404 and "No endpoints found" in msg:
                return HTTPException(
                    status_code=502,
                    detail=(
                        "Upstream has no provider for this VLM/chat model (404). "
                        "Set VISION_MODEL to a current OpenRouter vision model or remove it to use "
                        "the default openrouter/free. "
                        f"Details: {msg}"
                    ),
                )
            if sc == 401:
                return HTTPException(
                    status_code=401,
                    detail=f"Upstream API rejected the key (401). {msg}",
                )
    except ImportError:
        pass
    if "insufficient_quota" in msg or "Error code: 429" in msg:
        return HTTPException(
            status_code=429,
            detail=(
                "OpenAI or upstream returned quota/rate limit (429). "
                "Add credits, use OpenRouter for embeddings, or set SKIP_VLM=true. "
                f"Details: {msg}"
            ),
        )
    return HTTPException(status_code=500, detail=msg)


app = FastAPI(
    title="RAG Assignment API",
    version="0.2.0",
    description=(
        "RAG: PyMuPDF + LangChain recursive chunking; OpenRouter Nemotron embeddings by default; "
        "figure VLM uses OpenRouter chat when CHAT_BASE_URL is set, else OpenAI; FAISS."
    ),
)


@app.get("/", response_model=None)
def root(request: Request) -> Response:
    """Browsers → Swagger UI; curl / API clients → JSON (avoids empty 404 on Codespaces)."""
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return RedirectResponse(url="/docs")
    return JSONResponse(
        {
            "service": "RAG Assignment API",
            "docs": "/docs",
            "openapi": "/openapi.json",
            "health": "GET /health",
            "ingest": "POST /ingest (multipart PDF)",
            "query": "POST /query",
        }
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    config.reload_dotenv()
    base = os.getenv("CHAT_BASE_URL", "").strip().rstrip("/") or None
    emb_base = (
        os.getenv("EMBEDDING_BASE_URL", "").strip().rstrip("/") or base
    )
    or_key = os.getenv("OPENROUTER_API_KEY", "").strip() or os.getenv(
        "CHAT_API_KEY", ""
    ).strip()
    return HealthResponse(
        status="ok",
        models_ready=models_ready(),
        openai_ready=embeddings_ready(),
        chat_ready=chat_ready(),
        chat_base_url=base,
        embedding_base_url=emb_base or None,
        indexed_documents=app_state.indexed_document_count(),
        vector_count=app_state.vector_count(),
        uptime_seconds=app_state.uptime_seconds(),
        embedding_model=config.get_embedding_model(),
        chat_model=config.get_chat_model(),
        vision_model=config.get_vision_model(),
        dotenv_path=str(config.DOTENV_FILE),
        dotenv_found=config.DOTENV_FILE.is_file(),
        openai_key_set=bool(os.getenv("OPENAI_API_KEY", "").strip()),
        vision_key_set=get_vision_client() is not None,
        openrouter_or_chat_key_set=bool(or_key),
    )


@app.get("/stats", response_model=IndexStatsResponse)
def index_stats() -> IndexStatsResponse:
    """Chunk counts per document and by type (from FAISS sidecar metadata)."""
    store = app_state.get_store()
    raw = store.chunk_stats_by_document()
    docs: list[DocumentChunkStats] = []
    for source_file, by_type in raw.items():
        total = sum(by_type.values())
        docs.append(
            DocumentChunkStats(source_file=source_file, by_type=by_type, total=total)
        )
    return IndexStatsResponse(vector_count=store.ntotal, documents=docs)


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)) -> IngestResponse:
    config.reload_dotenv()
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF uploads are supported.",
        )
    embed_client = get_embedding_client()
    if embed_client is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Embedding client not configured: for OpenRouter Nemotron set "
                "CHAT_BASE_URL + OPENROUTER_API_KEY (or EMBEDDING_BASE_URL + EMBEDDING_API_KEY). "
                "For OpenAI text-embedding-* set OPENAI_API_KEY."
            ),
        )
    vision_client = get_vision_client()
    if config.get_skip_vlm():
        vision_client = None
    elif vision_client is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Figure summarization (VLM) needs CHAT_BASE_URL + OPENROUTER_API_KEY (or OPENAI_API_KEY "
                "when not using OpenRouter). Set SKIP_VLM=true to skip image captions."
            ),
        )

    from src.ingestion.pipeline import ingest_pdf_to_store

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".pdf", delete=False
        ) as tmp:
            tmp.write(raw)
            tmp_path = Path(tmp.name)

        with app_state.ingest_lock():
            summary = ingest_pdf_to_store(
                tmp_path,
                app_state.get_store(),
                embed_client,
                vision_client,
                source_name=file.filename,
            )
        app_state.register_document(summary["filename"])

        return IngestResponse(
            filename=summary["filename"],
            processing_seconds=summary["processing_seconds"],
            chunk_counts=summary["chunk_counts"],
            chunks_indexed=summary["chunks_indexed"],
            chunker=summary.get("chunker", "unknown"),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Ingest failed")
        raise _http_exception_from_upstream(e) from e
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


@app.post("/query", response_model=QueryResponse)
def query(body: QueryRequest) -> QueryResponse:
    config.reload_dotenv()
    embed_client = get_embedding_client()
    if embed_client is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Embedding client not configured: set OpenRouter keys for Nemotron or "
                "OPENAI_API_KEY for text-embedding-*."
            ),
        )
    chat_client = get_chat_client()
    if chat_client is None:
        raise HTTPException(
            status_code=503,
            detail="Chat is not configured: set OPENAI_API_KEY, or CHAT_BASE_URL + OPENROUTER_API_KEY (or CHAT_API_KEY).",
        )
    try:
        answer, sources_raw = run_query(
            app_state.get_store(),
            embed_client,
            chat_client,
            body.question,
            top_k=body.top_k,
            diversify_sources=body.diversify_sources,
        )
        sources = [
            SourceRef(
                filename=s["filename"],
                page=s["page"],
                chunk_type=s["chunk_type"],
                excerpt=s["excerpt"],
                distance=s.get("distance"),
            )
            for s in sources_raw
        ]
        return QueryResponse(answer=answer, sources=sources)
    except Exception as e:
        logger.exception("Query failed")
        raise _http_exception_from_upstream(e) from e


def run() -> None:
    import os

    host = os.getenv("APP_HOST", "127.0.0.1")
    port = int(os.getenv("APP_PORT", "8000"))
    uvicorn.run("src.api.app:app", host=host, port=port, reload=True)
