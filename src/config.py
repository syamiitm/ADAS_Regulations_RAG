"""Environment-driven settings (no pydantic-settings dependency)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

# Load `.env` before any reads. `state` and other modules import `config` before
# `src.api.app` runs its own load_dotenv, so this must happen here first.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOTENV_FILE = _PROJECT_ROOT / ".env"
PROJECT_ROOT = _PROJECT_ROOT


def reload_dotenv() -> bool:
    """Re-read `.env` into ``os.environ``. Handles BOM (utf-8-sig), UTF-16 exports from Notepad, etc."""
    if not DOTENV_FILE.is_file():
        return False

    encodings = ("utf-8-sig", "utf-8", "utf-16-le", "utf-16-be", "latin-1")
    best: dict | None = None
    best_count = 0
    for enc in encodings:
        try:
            vals = dotenv_values(str(DOTENV_FILE), encoding=enc)
        except (UnicodeError, UnicodeDecodeError):
            continue
        n = sum(1 for v in vals.values() if v and str(v).strip())
        if n > best_count:
            best_count = n
            best = vals

    if best is not None and best_count > 0:
        for k, v in best.items():
            if not k or v is None:
                continue
            s = str(v).strip()
            if not s:
                continue
            if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
                s = s[1:-1]
            os.environ[k] = s
        return True

    return bool(load_dotenv(DOTENV_FILE, override=True, encoding="utf-8-sig"))


reload_dotenv()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# Default: OpenRouter free NVIDIA multimodal embedder (2048-d). Override with text-embedding-* for OpenAI.
DEFAULT_EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip()
# Nemotron Embed VL 1B V2 outputs 2048-d vectors (see OpenRouter / NVIDIA docs).
EMBEDDING_DIMENSION = _env_int("EMBEDDING_DIMENSION", 2048)

# Chat completions: default OpenAI. Set CHAT_BASE_URL (e.g. OpenRouter) for Qwen etc.
CHAT_BASE_URL = os.getenv("CHAT_BASE_URL", "").strip().rstrip("/")
CHAT_API_KEY = (
    os.getenv("CHAT_API_KEY", "").strip()
    or os.getenv("OPENROUTER_API_KEY", "").strip()
)
CHAT_MODEL_RAW = os.getenv("CHAT_MODEL", "").strip()
CHAT_MODEL = CHAT_MODEL_RAW or (
    "qwen/qwen3.6-plus:free" if CHAT_BASE_URL else "gpt-4o-mini"
)


def get_embedding_model() -> str:
    """Embedding model id (OpenRouter or OpenAI ``text-embedding-*``)."""
    return os.getenv("EMBEDDING_MODEL", "").strip() or DEFAULT_EMBEDDING_MODEL


def get_chat_model() -> str:
    """Current chat model from the environment (use after reload_dotenv)."""
    raw = os.getenv("CHAT_MODEL", "").strip()
    url = os.getenv("CHAT_BASE_URL", "").strip().rstrip("/")
    if raw:
        return raw
    return "qwen/qwen3.6-plus:free" if url else "gpt-4o-mini"


# OpenRouter router: picks a free model that supports the request (e.g. vision when images are sent).
# Avoids hard-coded :free model IDs that OpenRouter may retire (404 "No endpoints found").
_DEFAULT_VISION_MODEL_OPENROUTER = "openrouter/free"


def get_vision_model() -> str:
    """Model for figure VLM; on OpenRouter defaults to ``openrouter/free`` if VISION_MODEL unset."""
    raw = os.getenv("VISION_MODEL", "").strip()
    if raw:
        return raw
    if os.getenv("CHAT_BASE_URL", "").strip().rstrip("/"):
        return _DEFAULT_VISION_MODEL_OPENROUTER
    return "gpt-4o-mini"


QUERY_TOP_K_DEFAULT = _env_int("QUERY_TOP_K_DEFAULT", 8)
QUERY_REWRITE_MAX_TOKENS = _env_int("QUERY_REWRITE_MAX_TOKENS", 256)


def get_query_rewrite_enabled() -> bool:
    """If true, /query runs one chat call to rewrite the question before embedding (retrieval only)."""
    return _env_bool("QUERY_REWRITE", False)


def get_skip_vlm() -> bool:
    """If true, /ingest skips VLM calls for image chunks (no vision client required)."""
    return _env_bool("SKIP_VLM", False)


# Optional OpenRouter attribution (recommended by OpenRouter)
OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER", "http://127.0.0.1:8000").strip()
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "RAG Assignment API").strip()
