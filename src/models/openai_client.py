"""OpenAI SDK clients: embeddings (OpenAI or OpenRouter), vision (OpenRouter chat or OpenAI), chat."""

from __future__ import annotations

import os

from openai import OpenAI

from src import config

_openai_client: OpenAI | None = None
_chat_client: OpenAI | None = None
_embedding_client: OpenAI | None = None
_cached_openai_key: str | None = None
_cached_chat_signature: str | None = None
_cached_embedding_signature: str | None = None


def _openai_key() -> str:
    return os.getenv("OPENAI_API_KEY", "").strip()


def _chat_base_url() -> str:
    return os.getenv("CHAT_BASE_URL", "").strip().rstrip("/")


def _chat_key() -> str:
    return os.getenv("CHAT_API_KEY", "").strip() or os.getenv(
        "OPENROUTER_API_KEY", ""
    ).strip()


def _embedding_base_url() -> str:
    return (
        os.getenv("EMBEDDING_BASE_URL", "").strip().rstrip("/") or _chat_base_url()
    )


def _embedding_api_key() -> str:
    return (
        os.getenv("EMBEDDING_API_KEY", "").strip()
        or _chat_key()
    )


def get_openai_client() -> OpenAI | None:
    """OpenAI platform API (figure VLM / optional OpenAI embeddings)."""
    global _openai_client, _cached_openai_key
    key = _openai_key()
    if not key:
        _openai_client = None
        _cached_openai_key = None
        return None
    if _openai_client is not None and _cached_openai_key == key:
        return _openai_client
    _cached_openai_key = key
    _openai_client = OpenAI(api_key=key)
    return _openai_client


def get_embedding_client() -> OpenAI | None:
    """Embeddings: OpenAI for ``text-embedding-*``; otherwise OpenRouter-compatible base."""
    global _embedding_client, _cached_embedding_signature
    model = config.get_embedding_model()
    if model.startswith("text-embedding-"):
        return get_openai_client()

    base = _embedding_base_url()
    key = _embedding_api_key()
    if not base or not key:
        _embedding_client = None
        _cached_embedding_signature = None
        return None
    sig = f"{base}\n{key}"
    if _embedding_client is not None and _cached_embedding_signature == sig:
        return _embedding_client
    _cached_embedding_signature = sig
    headers: dict[str, str] = {}
    if "openrouter.ai" in base.lower():
        headers["HTTP-Referer"] = config.OPENROUTER_HTTP_REFERER
        headers["X-Title"] = f"{config.OPENROUTER_APP_NAME} Embeddings"
    _embedding_client = OpenAI(
        base_url=base,
        api_key=key,
        default_headers=headers or None,
    )
    return _embedding_client


def get_chat_client() -> OpenAI | None:
    """Chat completions: OpenRouter (or other OpenAI-compatible base) if set, else OpenAI."""
    global _chat_client, _cached_chat_signature
    base = _chat_base_url()
    if base:
        ck = _chat_key()
        if not ck:
            _chat_client = None
            _cached_chat_signature = None
            return None
        sig = f"{base}\n{ck}"
        if _chat_client is not None and _cached_chat_signature == sig:
            return _chat_client
        _cached_chat_signature = sig
        headers: dict[str, str] = {}
        if "openrouter.ai" in base.lower():
            headers["HTTP-Referer"] = config.OPENROUTER_HTTP_REFERER
            headers["X-Title"] = config.OPENROUTER_APP_NAME
        _chat_client = OpenAI(
            base_url=base,
            api_key=ck,
            default_headers=headers or None,
        )
        return _chat_client
    return get_openai_client()


def get_vision_client() -> OpenAI | None:
    """VLM (image → text): OpenRouter chat client when CHAT_BASE_URL is set; else OpenAI."""
    if _chat_base_url():
        return get_chat_client()
    return get_openai_client()


def embeddings_ready() -> bool:
    return get_embedding_client() is not None


def chat_ready() -> bool:
    return get_chat_client() is not None


def models_ready() -> bool:
    """Embedding + chat; plus VLM client unless SKIP_VLM."""
    if not embeddings_ready() or not chat_ready():
        return False
    if config.get_skip_vlm():
        return True
    return get_vision_client() is not None
