"""FAISS-backed dense vector index with parallel metadata (FAISS stores vectors only)."""

from __future__ import annotations

import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import faiss
import numpy as np


class FaissVectorStore:
    """L2 distance on float32 vectors; metadata list index aligns with FAISS row order."""

    def __init__(self, dimension: int) -> None:
        if dimension < 1:
            raise ValueError("dimension must be positive")
        self.dimension = dimension
        self._index = faiss.IndexFlatL2(dimension)
        self._metadata: list[dict[str, Any]] = []

    @property
    def ntotal(self) -> int:
        return int(self._index.ntotal)

    def clear(self) -> None:
        self._index.reset()
        self._metadata.clear()

    def add(self, vectors: np.ndarray, metadatas: list[dict[str, Any]]) -> None:
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(
                f"vectors must be (n, {self.dimension}), got {vectors.shape}"
            )
        if vectors.shape[0] != len(metadatas):
            raise ValueError("vectors and metadatas must have same length")
        if vectors.shape[0] == 0:
            return
        v = np.ascontiguousarray(vectors, dtype=np.float32)
        self._index.add(v)
        self._metadata.extend(metadatas)

    def search(
        self, query_vector: np.ndarray, k: int
    ) -> list[tuple[dict[str, Any], float]]:
        if self.ntotal == 0:
            return []
        q = np.ascontiguousarray(query_vector, dtype=np.float32).reshape(1, -1)
        if q.shape[1] != self.dimension:
            raise ValueError(f"query dim must be {self.dimension}")
        kk = min(k, self.ntotal)
        distances, indices = self._index.search(q, kk)
        out: list[tuple[dict[str, Any], float]] = []
        for dist, idx in zip(distances[0], indices[0], strict=True):
            if idx < 0:
                continue
            out.append((self._metadata[idx], float(dist)))
        return out

    def chunk_stats_by_document(self) -> dict[str, dict[str, int]]:
        """Per ``source_file``: counts of ``chunk_type`` (text / table / image)."""
        by_file: dict[str, Counter[str]] = defaultdict(Counter)
        for m in self._metadata:
            fn = str(m.get("source_file") or "?")
            ct = str(m.get("chunk_type") or "?")
            by_file[fn][ct] += 1
        return {fn: dict(types) for fn, types in sorted(by_file.items())}

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(directory / "index.faiss"))
        with (directory / "metadata.pkl").open("wb") as f:
            pickle.dump(self._metadata, f)

    @classmethod
    def load(cls, directory: Path) -> FaissVectorStore:
        directory = Path(directory)
        path = directory / "index.faiss"
        if not path.is_file():
            raise FileNotFoundError(path)
        store = cls(dimension=int(faiss.read_index(str(path)).d))
        store._index = faiss.read_index(str(path))
        meta_path = directory / "metadata.pkl"
        if meta_path.is_file():
            with meta_path.open("rb") as f:
                store._metadata = pickle.load(f)
        else:
            store._metadata = []
        if len(store._metadata) != store.ntotal:
            raise ValueError("metadata length does not match FAISS index size")
        return store
