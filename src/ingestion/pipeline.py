"""PDF → SmartChunker (PyMuPDF + LangChain) → optional VLM → embeddings → FAISS."""

from __future__ import annotations

import logging
import time
import uuid
from collections import Counter
from pathlib import Path

from openai import OpenAI

from src.ingestion.chunks import ParsedChunk
from src.ingestion.smart_chunker import SmartChunker, documents_to_parsed_chunks
from src.models.embeddings import embed_texts
from src.models.vision import summarize_figure_png
from src.retrieval.vector_store import FaissVectorStore

logger = logging.getLogger(__name__)


def _enrich_images_with_vlm(
    chunks: list[ParsedChunk], client: OpenAI | None
) -> None:
    for ch in chunks:
        if ch.chunk_type != "image":
            continue
        if client is not None and ch.image_bytes:
            summary = summarize_figure_png(client, ch.image_bytes)
            ch.content = f"{ch.content}\n\n[VLM summary]\n{summary}".strip()
        ch.extra["vlm_applied"] = bool(client is not None and ch.image_bytes)


def ingest_pdf_to_store(
    pdf_path: Path,
    store: FaissVectorStore,
    embed_client: OpenAI,
    vision_client: OpenAI | None,
    *,
    source_name: str | None = None,
    converter: object | None = None,
) -> dict:
    """
    Chunk ``pdf_path`` with :class:`SmartChunker`, embed with ``embed_client``, index.

    ``converter`` is accepted for API compatibility and ignored.
    """
    _ = converter
    t0 = time.perf_counter()
    display_name = source_name or pdf_path.name

    chunker = SmartChunker()
    docs = chunker.process_pdf(pdf_path)
    chunks: list[ParsedChunk] = documents_to_parsed_chunks(docs, display_name)

    if not chunks:
        logger.warning("No chunks produced for %s", display_name)

    _enrich_images_with_vlm(chunks, vision_client)
    counts = Counter(c.chunk_type for c in chunks)
    count_dict = {
        "text": counts.get("text", 0),
        "table": counts.get("table", 0),
        "image": counts.get("image", 0),
    }

    if not chunks:
        elapsed = time.perf_counter() - t0
        return {
            "filename": display_name,
            "processing_seconds": round(elapsed, 3),
            "chunk_counts": count_dict,
            "chunks_indexed": 0,
            "chunker": "pymupdf_langchain",
        }

    texts = [c.content for c in chunks]
    vectors = embed_texts(embed_client, texts)

    metadatas: list[dict] = []
    for ch in chunks:
        metadatas.append(
            {
                "chunk_id": str(uuid.uuid4()),
                "source_file": ch.source_file,
                "page": ch.page,
                "chunk_type": ch.chunk_type,
                "text": ch.content,
                "docling_self_ref": ch.docling_self_ref,
                "section_title": str(ch.extra.get("title", "")),
                "chunk_index": int(ch.extra.get("chunk_id", -1)),
                "char_count": int(ch.extra.get("char_count", 0)),
                "has_numbers": bool(ch.extra.get("has_numbers", False)),
            }
        )

    store.add(vectors, metadatas)
    elapsed = time.perf_counter() - t0
    return {
        "filename": display_name,
        "processing_seconds": round(elapsed, 3),
        "chunk_counts": count_dict,
        "chunks_indexed": len(chunks),
        "chunker": "pymupdf_langchain",
    }
