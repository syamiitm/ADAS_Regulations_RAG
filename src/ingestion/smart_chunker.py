"""
Universal RAG chunking: PyMuPDF text + tables + embedded images, LangChain splitting.

- Text: per-page extract, recursive split (standards-friendly separators).
- Tables: ``Page.find_tables()`` → markdown / TSV fallback, split if oversized.
- Images: unique xrefs per PDF → PNG for VLM; tiny icons skipped.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import fitz
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.chunks import ChunkType, ParsedChunk

logger = logging.getLogger(__name__)


class SmartChunker:
    """Recursive split with separators tuned for standards / regulatory PDFs."""

    NUMBERS = re.compile(r"\d")

    def __init__(
        self,
        *,
        chunk_size: int = 1400,
        chunk_overlap: int = 280,
        min_page_chars: int = 50,
        min_image_pixels: int = 4096,
        min_image_side: int = 48,
    ) -> None:
        self.min_page_chars = min_page_chars
        self.chunk_size = chunk_size
        self.min_image_pixels = min_image_pixels
        self.min_image_side = min_image_side
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n## ",
                "\n### ",
                "\n1. ",
                "\n2. ",
                "\nAnnex ",
                "\nScope",
                "\n",
                ". ",
                " ",
            ],
            keep_separator=True,
        )

    def extract_title(self, page: fitz.Page) -> str:
        """Heuristic: first short line block as page/section title."""
        try:
            td = page.get_text("dict")
            blocks = td.get("blocks") or []
        except (RuntimeError, ValueError, KeyError):
            return "Untitled"
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines") or []:
                parts: list[str] = []
                for span in line.get("spans") or []:
                    parts.append(str(span.get("text") or ""))
                text = "".join(parts).strip()
                if 5 < len(text) < 100:
                    return text
        return "Untitled"

    @staticmethod
    def _table_to_text(table: Any) -> str:
        try:
            md = table.to_markdown()
            if isinstance(md, str) and md.strip():
                return md.strip()
        except Exception:
            pass
        try:
            rows = table.extract()
        except Exception:
            return ""
        if not rows:
            return ""
        lines: list[str] = []
        for row in rows:
            cells = ["" if c is None else str(c).strip() for c in row]
            lines.append("\t".join(cells))
        return "\n".join(lines).strip()

    def _collect_table_documents(
        self, doc: fitz.Document, path_str: str
    ) -> list[Document]:
        out: list[Document] = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            title = self.extract_title(page)
            try:
                tf = page.find_tables()
            except Exception as e:
                logger.debug("find_tables skip page %s: %s", page_num + 1, e)
                continue
            tables: list[Any] = []
            if tf is not None:
                raw = getattr(tf, "tables", None)
                tables = list(raw) if raw is not None else []
                if not tables:
                    try:
                        tables = list(tf)
                    except TypeError:
                        tables = []
            for ti, table in enumerate(tables):
                body = self._table_to_text(table)
                if len(body) < 3:
                    continue
                out.append(
                    Document(
                        page_content=body,
                        metadata={
                            "source": path_str,
                            "page": page_num + 1,
                            "title": title,
                            "chunk_type": "table",
                            "table_index": ti,
                        },
                    )
                )
        return out

    def _split_oversized(self, docs: list[Document]) -> list[Document]:
        merged: list[Document] = []
        for d in docs:
            if len(d.page_content) <= self.chunk_size:
                merged.append(d)
            else:
                merged.extend(self.splitter.split_documents([d]))
        return merged

    def _xref_to_png(self, doc: fitz.Document, xref: int) -> bytes | None:
        try:
            pix = fitz.Pixmap(doc, xref)
        except Exception:
            return None
        try:
            if pix.width < self.min_image_side or pix.height < self.min_image_side:
                return None
            if pix.width * pix.height < self.min_image_pixels:
                return None
            if pix.colorspace is None:
                return None
            # CMYK etc. → RGB
            if pix.n - pix.alpha >= 4:
                spix = fitz.Pixmap(fitz.csRGB, pix)
                pix.close()
                pix = spix
            elif pix.alpha:
                spix = fitz.Pixmap(fitz.csRGB, pix)
                pix.close()
                pix = spix
            return pix.tobytes("png")
        except Exception:
            return None
        finally:
            try:
                pix.close()
            except Exception:
                pass

    def _collect_image_documents(
        self, doc: fitz.Document, path_str: str
    ) -> list[Document]:
        out: list[Document] = []
        seen_xref: set[int] = set()
        for page_num in range(len(doc)):
            page = doc[page_num]
            title = self.extract_title(page)
            try:
                images = page.get_images(full=True) or []
            except Exception:
                continue
            for info in images:
                xref = int(info[0])
                if xref in seen_xref:
                    continue
                png = self._xref_to_png(doc, xref)
                if not png:
                    continue
                seen_xref.add(xref)
                out.append(
                    Document(
                        page_content=(
                            f"[Embedded figure / image on page {page_num + 1}, "
                            f"image id {xref}]"
                        ),
                        metadata={
                            "source": path_str,
                            "page": page_num + 1,
                            "title": title,
                            "chunk_type": "image",
                            "image_xref": xref,
                            "image_bytes": png,
                        },
                    )
                )
        return out

    def process_pdf(self, pdf_path: str | Path) -> list[Document]:
        """
        Text (split), tables (split if large), images (one chunk each, PNG for VLM).

        Raises:
            RuntimeError: PDF cannot be opened or read.
            FileNotFoundError: Path does not exist.
        """
        path = Path(pdf_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        path_str = str(path)
        page_docs: list[Document] = []
        try:
            doc = fitz.open(path_str)
        except Exception as e:
            raise RuntimeError(f"Failed to open PDF: {path_str}") from e
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text() or ""
                if len(text.strip()) < self.min_page_chars:
                    continue
                page_docs.append(
                    Document(
                        page_content=text.strip(),
                        metadata={
                            "source": path_str,
                            "page": page_num + 1,
                            "title": self.extract_title(page),
                            "chunk_type": "text",
                        },
                    )
                )
            table_docs = self._collect_table_documents(doc, path_str)
            image_docs = self._collect_image_documents(doc, path_str)
        finally:
            doc.close()

        combined: list[Document] = []
        if page_docs:
            combined.extend(self.splitter.split_documents(page_docs))
        if table_docs:
            combined.extend(self._split_oversized(table_docs))
        combined.extend(image_docs)

        if not combined:
            return []
        return self.enrich_chunks(combined)

    def enrich_chunks(self, chunks: list[Document]) -> list[Document]:
        """Add chunk_id, char_count, has_numbers for filtering / debugging."""
        out: list[Document] = []
        for i, chunk in enumerate(chunks):
            md: dict[str, Any] = dict(chunk.metadata)
            md.update(
                {
                    "chunk_id": i,
                    "char_count": len(chunk.page_content),
                    "has_numbers": bool(self.NUMBERS.search(chunk.page_content)),
                }
            )
            out.append(Document(page_content=chunk.page_content, metadata=md))
        return out


def documents_to_parsed_chunks(
    documents: list[Document],
    source_file: str,
) -> list[ParsedChunk]:
    """Convert LangChain ``Document``s to internal ``ParsedChunk`` rows."""
    rows: list[ParsedChunk] = []
    for d in documents:
        md = dict(d.metadata or {})
        raw_type = md.get("chunk_type", "text")
        ctype: ChunkType
        if raw_type in ("text", "table", "image"):
            ctype = raw_type
        else:
            ctype = "text"

        img_raw = md.get("image_bytes") if ctype == "image" else None
        img_b = (
            bytes(img_raw)
            if isinstance(img_raw, (bytes, bytearray))
            else None
        )
        extra: dict[str, Any] = {
            "title": md.get("title", ""),
            "chunk_id": md.get("chunk_id", -1),
            "char_count": md.get("char_count", 0),
            "has_numbers": md.get("has_numbers", False),
            "source_path": md.get("source", ""),
        }
        if md.get("table_index") is not None:
            extra["table_index"] = int(md["table_index"])
        if md.get("image_xref") is not None:
            extra["image_xref"] = int(md["image_xref"])
        rows.append(
            ParsedChunk(
                chunk_type=ctype,
                content=d.page_content,
                page=int(md.get("page", 0)),
                source_file=source_file,
                docling_self_ref="",
                image_bytes=img_b,
                extra=extra,
            )
        )
    return rows
