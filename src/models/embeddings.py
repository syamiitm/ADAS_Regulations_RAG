"""Batch text embeddings via OpenAI-compatible API (OpenAI or OpenRouter)."""

from __future__ import annotations

import numpy as np
from openai import OpenAI

from src import config


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Normalize vectors to unit length for cosine similarity via L2 norm."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True).astype(np.float64)
    norms = np.maximum(norms, 1e-12)  # avoid division by zero
    return (vectors / norms).astype(np.float32)


def embed_texts(
    client: OpenAI, texts: list[str], batch_size: int = 100
) -> np.ndarray:
    """
    Generate embeddings for a batch of texts using the provided client.
    Works with OpenAI or OpenRouter (via base_url + api_key in config).

    Args:
        client: An OpenAI client instance (configured for OpenAI or OpenRouter).
        texts: List of text strings to embed.
        batch_size: Number of texts per API call.

    Returns:
        np.ndarray: L2-normalized embeddings.
    """
    if not texts:
        return np.zeros((0, config.EMBEDDING_DIMENSION), dtype=np.float32)

    # Clean and truncate inputs
    safe = [(t.strip() if t.strip() else " ")[:32000] for t in texts]

    model = config.get_embedding_model()
    # OpenAI SDK defaults to encoding_format=base64; OpenRouter often returns float
    # vectors and the SDK parser can end up with empty `data` → "No embedding data received".
    extra: dict = {"encoding_format": "float"}
    if model.startswith("text-embedding-3"):
        max_dim = 3072 if "large" in model else 1536
        d = config.EMBEDDING_DIMENSION
        if d > max_dim:
            raise ValueError(
                f"EMBEDDING_DIMENSION={d} exceeds {max_dim} for {model}. "
                "Lower EMBEDDING_DIMENSION or use a model that matches (e.g. Nemotron 2048)."
            )
        extra["dimensions"] = d

    rows: list[np.ndarray] = []
    for i in range(0, len(safe), batch_size):
        batch = safe[i : i + batch_size]

        resp = client.embeddings.create(model=model, input=batch, **extra)

        # Ensure order is preserved
        ordered = sorted(resp.data, key=lambda d: d.index)
        arr = np.array([d.embedding for d in ordered], dtype=np.float32)

        # Dimension check
        if arr.shape[1] != config.EMBEDDING_DIMENSION:
            raise ValueError(
                f"Embedding dim {arr.shape[1]} != config {config.EMBEDDING_DIMENSION}"
            )
        rows.append(arr)

    combined = np.vstack(rows) if rows else np.zeros(
        (0, config.EMBEDDING_DIMENSION), dtype=np.float32
    )
    return l2_normalize(combined)