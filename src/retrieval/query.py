"""Retrieve + generate for POST /query."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from openai import OpenAI

from src import config
from src.models.embeddings import embed_texts
from src.models.llm import answer_with_context, rewrite_query_for_retrieval
from src.retrieval.vector_store import FaissVectorStore

logger = logging.getLogger(__name__)


def _diversify_hits(
    hits: Sequence[tuple[dict, float]],
    k: int,
) -> list[tuple[dict, float]]:
    """
    ``hits`` sorted by ascending distance. Prefer at least one chunk per ``source_file``,
    then fill remaining slots in global similarity order (dedupe by ``chunk_id``).
    """
    if k <= 0 or not hits:
        return []
    seen_chunk: set[str] = set()
    result: list[tuple[dict, float]] = []

    seen_file: set[str] = set()
    for meta, dist in hits:
        fn = str(meta.get("source_file", ""))
        cid = str(meta.get("chunk_id", ""))
        if fn in seen_file:
            continue
        seen_file.add(fn)
        if cid:
            seen_chunk.add(cid)
        result.append((meta, dist))
        if len(result) >= k:
            return result

    for meta, dist in hits:
        cid = str(meta.get("chunk_id", ""))
        if cid and cid in seen_chunk:
            continue
        if cid:
            seen_chunk.add(cid)
        result.append((meta, dist))
        if len(result) >= k:
            break
    return result


def _excerpt(text: str, max_len: int = 400) -> str:
    t = text.strip().replace("\n", " ")
    if len(t) <= max_len:
        return t
    return t[: max_len - 3] + "..."


def run_query(
    store: FaissVectorStore,
    embed_client: OpenAI,
    chat_client: OpenAI,
    question: str,
    top_k: int | None = None,
    *,
    diversify_sources: bool | None = None,
) -> tuple[str, list[dict]]:
    k = top_k if top_k is not None else config.QUERY_TOP_K_DEFAULT
    k = max(1, min(k, 50))
    diversify = (
        diversify_sources
        if diversify_sources is not None
        else config.get_query_diversify_sources()
    )

    if store.ntotal == 0:
        return (
            "The vector index is empty. Ingest at least one PDF via POST /ingest.",
            [],
        )

    q_embed = question.strip()
    if config.get_query_rewrite_enabled():
        try:
            q_embed = rewrite_query_for_retrieval(
                chat_client,
                question,
                max_tokens=config.QUERY_REWRITE_MAX_TOKENS,
            )
            logger.debug("Query rewrite: %r -> %r", question[:120], q_embed[:120])
        except Exception as e:
            logger.warning("Query rewrite failed, using original question: %s", e)
            q_embed = question.strip()

    qv = embed_texts(embed_client, [q_embed])
    pool_k = k
    if diversify:
        # Larger pool so a second PDF can appear even when one doc dominates similarity.
        pool_k = min(store.ntotal, max(k * 6, 40))
    hits = store.search(qv[0], k=pool_k)
    if diversify and pool_k > k:
        hits = _diversify_hits(hits, k)

    texts: list[str] = []
    metas: list[dict[str, str]] = []
    sources: list[dict] = []
    for meta, dist in hits:
        text = str(meta.get("text", ""))
        texts.append(text)
        m = {
            "source_file": str(meta.get("source_file", "")),
            "page": str(meta.get("page", "")),
            "chunk_type": str(meta.get("chunk_type", "")),
        }
        metas.append(m)
        sources.append(
            {
                "filename": m["source_file"],
                "page": int(meta["page"]) if str(meta.get("page", "")).isdigit() else meta.get("page"),
                "chunk_type": m["chunk_type"],
                "excerpt": _excerpt(text),
                "distance": round(dist, 6),
            }
        )

    answer = answer_with_context(
        chat_client, question, texts, metas, model=config.get_chat_model()
    )
    return answer, sources
