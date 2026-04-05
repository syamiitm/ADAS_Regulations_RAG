"""Grounded answer generation from retrieved chunks."""

from __future__ import annotations

from openai import OpenAI

from src import config


def build_context_block(texts: list[str], metas: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for i, (text, meta) in enumerate(zip(texts, metas, strict=True), start=1):
        head = (
            f"[{i}] source={meta['source_file']} page={meta['page']} "
            f"type={meta['chunk_type']}"
        )
        parts.append(f"{head}\n{text.strip()}")
    return "\n\n---\n\n".join(parts)


def rewrite_query_for_retrieval(
    client: OpenAI,
    question: str,
    *,
    model: str | None = None,
    max_tokens: int | None = None,
) -> str:
    """Turn a user question into a short search query for embedding (retrieval only; no answering)."""
    q = question.strip()
    if not q:
        return q
    model = model or config.get_chat_model()
    max_tokens = max_tokens if max_tokens is not None else config.QUERY_REWRITE_MAX_TOKENS
    sys = (
        "You rewrite questions for semantic search over regulatory and technical documents (e.g. automotive, ADAS). "
        "Output ONE concise search query in plain text. Do not answer the question. "
        "Do not invent article numbers, limits, or requirements. "
        "Expand obvious abbreviations only when safe. Prefer terms likely to appear in formal standards text."
    )
    user = f"User question:\n{q}\n\nSearch query:"
    r = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ],
        max_tokens=max(32, min(max_tokens, 512)),
        temperature=0.1,
    )
    out = (r.choices[0].message.content or "").strip()
    # Drop accidental quotes or a leading label line
    out = out.strip().strip('"').strip("'")
    for line in out.splitlines():
        t = line.strip()
        low = t.lower()
        if low.startswith(("search query:", "query:", "rewritten:", "output:")):
            t = t.split(":", 1)[-1].strip()
        if t:
            out = t
            break
    if len(out) < 2:
        return q
    return out[:4000]


def answer_with_context(
    client: OpenAI,
    question: str,
    chunk_texts: list[str],
    chunk_metas: list[dict[str, str]],
    *,
    model: str | None = None,
) -> str:
    if not chunk_texts:
        return (
            "No indexed content matches this query yet. "
            "Ingest a PDF via POST /ingest first."
        )
    model = model or config.get_chat_model()
    ctx = build_context_block(chunk_texts, chunk_metas)
    sys = (
        "You are a precise assistant for regulatory and technical documents. "
        "Answer ONLY using the provided context. If the context is insufficient, "
        "say so explicitly. Mention which source (filename, page, chunk type) "
        "supports each key claim when possible."
    )
    user = f"Question:\n{question}\n\nContext:\n{ctx}"
    r = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ],
        max_tokens=1024,
        temperature=0.7,
    )
    return (r.choices[0].message.content or "").strip()
