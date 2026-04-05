"""Normalized chunks emitted by parsers (before embedding / VLM enrichment)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ChunkType = Literal["text", "table", "image"]


@dataclass
class ParsedChunk:
    """One retrievable unit from a document."""

    chunk_type: ChunkType
    content: str
    page: int
    source_file: str
    docling_self_ref: str = ""
    image_bytes: bytes | None = None
    extra: dict[str, Any] = field(default_factory=dict)
