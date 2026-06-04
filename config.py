"""
Configuration constants for the AI Study Assistant.

Supports two providers, selected via env var / secret ``LLM_PROVIDER``:
  • ``groq``   — cloud LLM (works on Streamlit Cloud; requires GROQ_API_KEY)
  • ``ollama`` — local LLM (works only when Ollama is running on the host)

Embeddings:
  • cloud mode  -> ``fastembed`` (in-process, no API key, ~33 MB ONNX model)
  • local mode  -> Ollama ``nomic-embed-text``
"""

import os

# Streamlit secrets are loaded automatically when running under Streamlit
try:
    import streamlit as st  # type: ignore
    _SECRETS = dict(st.secrets) if hasattr(st, "secrets") else {}
except Exception:
    _SECRETS = {}


def _get(name: str, default: str = "") -> str:
    """Read a config value from Streamlit secrets or environment variables."""
    return str(_SECRETS.get(name, os.environ.get(name, default)))


# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
VECTORSTORE_DIR = os.path.join(PROJECT_DIR, "vectorstore")

# ── Provider switch ──────────────────────────────────────────────────
LLM_PROVIDER = _get("LLM_PROVIDER", "groq").lower()

# ── Groq (cloud) Settings ─────────────────────────────────────────────
GROQ_API_KEY = _get("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = _get("GROQ_MODEL", "llama-3.1-8b-instant")

# ── Ollama (local) Settings ──────────────────────────────────────────
OLLAMA_MODEL = _get("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_BASE_URL = _get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBEDDING_MODEL = _get("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

# ── Active LLM (resolved from provider) ──────────────────────────────
if LLM_PROVIDER == "groq":
    LLM_MODEL = GROQ_MODEL
    LLM_BASE_URL = GROQ_BASE_URL
else:
    LLM_MODEL = OLLAMA_MODEL
    LLM_BASE_URL = OLLAMA_BASE_URL

LLM_TEMPERATURE = 0.3          # Low temperature for factual answers
LLM_TOP_P = 0.9
LLM_NUM_CTX = 4096             # Context window

# ── Payload Size Limits (prevent 413 errors on Groq) ──────────────────
MAX_CONTEXT_CHARS = 12000      # Max context size in prompt (chars)
MAX_CHUNK_DISPLAY = 400        # Truncate each chunk to this (chars)

# ── Embedding Settings ───────────────────────────────────────────────
EMBEDDING_MODEL = (
    "BAAI/bge-small-en-v1.5" if LLM_PROVIDER == "groq" else OLLAMA_EMBEDDING_MODEL
)
EMBEDDING_BASE_URL = OLLAMA_BASE_URL  # only used in ollama mode

# ── Chunking Settings ────────────────────────────────────────────────
CHUNK_SIZE = 1000              # Characters per chunk
CHUNK_OVERLAP = 200            # Overlap between chunks

# ── Retrieval Settings ───────────────────────────────────────────────
RETRIEVAL_TOP_K = 4            # Number of chunks to retrieve
