"""Document ingestion: loading, chunking, embedding, indexing."""

from typing import Any

from src.ingestion.chunks import ChunkType, ParsedChunk

__all__ = ["ChunkType", "ParsedChunk", "SmartChunker", "documents_to_parsed_chunks"]


def __getattr__(name: str) -> Any:
    if name == "SmartChunker":
        from src.ingestion.smart_chunker import SmartChunker as SC

        return SC
    if name == "documents_to_parsed_chunks":
        from src.ingestion.smart_chunker import documents_to_parsed_chunks as dtp

        return dtp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
