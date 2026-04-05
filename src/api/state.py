"""Process-wide vector index and ingest bookkeeping."""

from __future__ import annotations

import threading
import time
from src import config
from src.retrieval.vector_store import FaissVectorStore

_started_perf = time.perf_counter()
_started_wall = time.time()
_lock = threading.Lock()
_store = FaissVectorStore(config.EMBEDDING_DIMENSION)
_indexed_filenames: set[str] = set()


def get_store() -> FaissVectorStore:
    return _store


def register_document(filename: str) -> None:
    _indexed_filenames.add(filename)


def indexed_document_count() -> int:
    return len(_indexed_filenames)


def vector_count() -> int:
    return _store.ntotal


def uptime_seconds() -> float:
    return round(time.perf_counter() - _started_perf, 3)


def process_started_wall_time() -> float:
    return _started_wall


def ingest_lock() -> threading.Lock:
    return _lock
