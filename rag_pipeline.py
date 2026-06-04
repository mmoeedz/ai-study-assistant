"""
Core RAG pipeline for the AI Study Assistant.

This implementation uses direct HTTP calls to the Ollama API and a pure-Python
vector store (NumPy-based cosine similarity) to avoid native DLL dependencies
that are blocked by Application Control policies.

Handles:
  - PDF text extraction (pypdf)
  - Text chunking (custom)
  - Embedding generation (Ollama REST API)
  - Vector similarity search (NumPy)
  - LLM answer generation (Ollama REST API)
"""

import io
import os
import json
import pickle
import requests
import numpy as np
from pypdf import PdfReader
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field

import config
from prompts import PROMPT_MAP


def _format_llm_request_error(exc: Exception) -> str:
    """Return a user-safe error message for provider/API failures."""
    if isinstance(exc, requests.exceptions.HTTPError):
        response = exc.response
        status = response.status_code if response is not None else "unknown"
        detail = ""
        if response is not None:
            try:
                payload = response.json()
                if isinstance(payload, dict):
                    err = payload.get("error", payload)
                    if isinstance(err, dict):
                        detail = str(
                            err.get("message")
                            or err.get("code")
                            or err
                        )
                    else:
                        detail = str(err)
            except Exception:
                detail = response.text[:500]
        detail = detail or str(exc)
        return (
            f"⚠️ LLM request failed (HTTP {status}).\n\n"
            f"{detail}\n\n"
            "Please check your Groq API key, model name, rate limits, and "
            "project permissions."
        )

    if isinstance(exc, requests.exceptions.Timeout):
        return "⚠️ The LLM request timed out. Please try again in a moment."

    if isinstance(exc, requests.exceptions.RequestException):
        return (
            "⚠️ Could not reach the LLM provider. Please check your internet "
            "connection and provider settings."
        )

    return f"⚠️ Something went wrong while generating the answer: {exc}"


# ── Data classes ──────────────────────────────────────────────────────

@dataclass
class Document:
    """A chunk of text with metadata."""
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── Text Splitter ─────────────────────────────────────────────────────

class RecursiveCharacterTextSplitter:
    """Pure-Python text splitter that recursively splits on separators."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: List[str] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        chunks = self._split_recursive(text, self.separators)
        # Merge small chunks and handle overlap
        return self._merge_chunks(chunks)

    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        sep = separators[0] if separators else ""
        remaining_separators = separators[1:] if len(separators) > 1 else []

        if sep == "":
            # Last resort: split by character count
            results = []
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                chunk = text[i : i + self.chunk_size]
                if chunk.strip():
                    results.append(chunk)
            return results

        parts = text.split(sep)
        results = []
        current = ""

        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current.strip():
                    results.append(current)
                if len(part) > self.chunk_size and remaining_separators:
                    results.extend(
                        self._split_recursive(part, remaining_separators)
                    )
                else:
                    current = part

        if current.strip():
            results.append(current)

        return results

    def _merge_chunks(self, chunks: List[str]) -> List[str]:
        """Add overlap between adjacent chunks."""
        if not chunks or self.chunk_overlap == 0:
            return chunks

        merged = []
        for i, chunk in enumerate(chunks):
            if i > 0 and self.chunk_overlap > 0:
                # Prepend overlap from previous chunk
                prev = chunks[i - 1]
                overlap_text = prev[-self.chunk_overlap :]
                chunk = overlap_text + chunk
                # Trim if too long
                if len(chunk) > self.chunk_size:
                    chunk = chunk[: self.chunk_size]
            merged.append(chunk)
        return merged

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split a list of Documents into smaller chunks."""
        result = []
        for doc in documents:
            texts = self.split_text(doc.page_content)
            for text in texts:
                result.append(
                    Document(page_content=text, metadata=doc.metadata.copy())
                )
        return result


# ── Simple Vector Store ───────────────────────────────────────────────

class SimpleVectorStore:
    """Pure-Python vector store using NumPy cosine similarity."""

    def __init__(self):
        self.embeddings: Optional[np.ndarray] = None  # shape: (n, dim)
        self.documents: List[Document] = []

    def add(self, docs: List[Document], vectors: List[List[float]]) -> None:
        """Add documents and their embeddings."""
        new_embeddings = np.array(vectors, dtype=np.float32)
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        self.documents.extend(docs)

    def search(self, query_vector: List[float], k: int = 4) -> List[Document]:
        """Find the k most similar documents using cosine similarity."""
        if self.embeddings is None or len(self.documents) == 0:
            return []

        q = np.array(query_vector, dtype=np.float32)
        # Cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(q)
        norms = np.where(norms == 0, 1e-10, norms)  # Avoid division by zero
        similarities = self.embeddings @ q / norms

        # Get top-k indices
        top_k = min(k, len(self.documents))
        indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.documents[i] for i in indices]

    def save(self, path: str) -> None:
        """Persist to disk using JSON + NumPy (no pickle anywhere)."""
        os.makedirs(path, exist_ok=True)

        # Force every document into plain Python primitives.
        docs_serialisable = []
        for d in self.documents:
            content = d.page_content
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="replace")
            content = str(content) if content is not None else ""

            meta_clean = {}
            for k, v in (d.metadata or {}).items():
                k = str(k)
                if v is None or isinstance(v, (bool, int, float)):
                    meta_clean[k] = v
                elif isinstance(v, bytes):
                    meta_clean[k] = v.decode("utf-8", errors="replace")
                else:
                    meta_clean[k] = str(v)
            docs_serialisable.append(
                {"page_content": content, "metadata": meta_clean}
            )

        with open(os.path.join(path, "documents.json"), "w", encoding="utf-8") as f:
            json.dump(docs_serialisable, f, ensure_ascii=False)

        # Save embeddings as a contiguous float32 array — explicitly disable
        # pickle so np.save NEVER falls back to it for any reason.
        if self.embeddings is not None:
            arr = np.asarray(self.embeddings, dtype=np.float32)
            np.save(
                os.path.join(path, "embeddings.npy"),
                arr,
                allow_pickle=False,
            )

    @classmethod
    def load(cls, path: str) -> Optional["SimpleVectorStore"]:
        """Load from disk; supports new JSON+npy format and falls back to old pickle."""
        json_path = os.path.join(path, "documents.json")
        npy_path = os.path.join(path, "embeddings.npy")
        if os.path.exists(json_path) and os.path.exists(npy_path):
            try:
                store = cls()
                with open(json_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                store.documents = [
                    Document(
                        page_content=d.get("page_content", ""),
                        metadata=d.get("metadata", {}) or {},
                    )
                    for d in raw
                ]
                store.embeddings = np.load(npy_path)
                return store
            except Exception:
                return None

        # Backwards-compat: old pickle format
        pkl_path = os.path.join(path, "vectorstore.pkl")
        if os.path.exists(pkl_path):
            try:
                store = cls()
                with open(pkl_path, "rb") as f:
                    data = pickle.load(f)
                store.embeddings = data["embeddings"]
                store.documents = data["documents"]
                return store
            except Exception:
                return None
        return None


# ── LLM Clients ──────────────────────────────────────────────────────

class OllamaClient:
    """Direct HTTP client for the Ollama API (local mode)."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")

    def embed(self, texts: List[str], model: str) -> List[List[float]]:
        embeddings = []
        for text in texts:
            resp = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": model, "input": text},
                timeout=120,
            )
            resp.raise_for_status()
            embeddings.append(resp.json()["embeddings"][0])
        return embeddings

    def embed_single(self, text: str, model: str) -> List[float]:
        return self.embed([text], model)[0]

    def generate(
        self, prompt: str, model: str,
        temperature: float = 0.3, num_ctx: int = 4096,
    ) -> str:
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": model, "prompt": prompt, "stream": False,
                "options": {"temperature": temperature, "num_ctx": num_ctx},
            },
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()["response"]


class GroqClient:
    """OpenAI-compatible HTTP client for the Groq API (cloud mode)."""

    def __init__(self, api_key: str, base_url: str = "https://api.groq.com/openai/v1"):
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to .streamlit/secrets.toml "
                "or set it as an environment variable."
            )
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def generate(
        self, prompt: str, model: str,
        temperature: float = 0.3, num_ctx: int = 4096,  # num_ctx kept for sig parity
    ) -> str:
        import time as _t  # local import to avoid name shadow
        last_exc = None
        for attempt in range(4):  # up to 4 attempts on 429 / transient errors
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                },
                timeout=120,
            )
            if resp.status_code == 429:
                # Honour Retry-After header if present, else exponential back-off
                wait = int(resp.headers.get("retry-after", "0")) or (2 ** attempt)
                _t.sleep(min(wait, 30))
                last_exc = requests.exceptions.HTTPError(
                    f"429 Too Many Requests (attempt {attempt + 1})", response=resp
                )
                continue
            if resp.status_code >= 400:
                # 5xx errors → retry once or twice with backoff
                if 500 <= resp.status_code < 600 and attempt < 2:
                    last_exc = requests.exceptions.HTTPError(
                        f"{resp.status_code}: {resp.text[:500]}", response=resp
                    )
                    _t.sleep(2 ** attempt)
                    continue
                # Surface the actual API error message for easier debugging
                try:
                    detail = resp.json()
                except Exception:
                    detail = resp.text[:500]
                raise requests.exceptions.HTTPError(
                    f"Groq API error {resp.status_code}: {detail}",
                    response=resp,
                )
            return resp.json()["choices"][0]["message"]["content"]
        # If we exhausted retries on 429s, raise the last one
        if last_exc:
            raise last_exc
        raise RuntimeError("Groq generate(): exhausted retries with no response")


# ── Embedding backends ───────────────────────────────────────────────

class FastEmbedEmbedder:
    """In-process embeddings via fastembed (no API key needed)."""

    _instance = None  # cached model handle

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        # Lazy import so installing fastembed isn't required for ollama mode
        from fastembed import TextEmbedding  # type: ignore
        if FastEmbedEmbedder._instance is None:
            FastEmbedEmbedder._instance = TextEmbedding(model_name=model_name)
        self.model = FastEmbedEmbedder._instance

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [list(map(float, v)) for v in self.model.embed(texts)]

    def embed_single(self, text: str) -> List[float]:
        return self.embed([text])[0]


# ── Study Assistant ──────────────────────────────────────────────────

class StudyAssistant:
    """End-to-end RAG study assistant.

    Provider is auto-selected from ``config.LLM_PROVIDER`` ("groq" or "ollama").
    """

    def __init__(self):
        self.provider = config.LLM_PROVIDER

        if self.provider == "groq":
            self.llm = GroqClient(config.GROQ_API_KEY, config.GROQ_BASE_URL)
            self.embedder = FastEmbedEmbedder(config.EMBEDDING_MODEL)
        else:
            self.llm = OllamaClient(config.LLM_BASE_URL)
            self.embedder = None  # ollama embeds via the same client

        self.vectorstore: Optional[SimpleVectorStore] = None
        self.indexed_files: List[str] = []
        self.total_chunks: int = 0

        # Try to load an existing vector store from disk
        self._try_load_vectorstore()

    # ── Internal embedding helpers ───────────────────────────────────
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        if self.provider == "groq":
            return self.embedder.embed(texts)
        return self.llm.embed(texts, config.EMBEDDING_MODEL)

    def _embed_single(self, text: str) -> List[float]:
        if self.provider == "groq":
            return self.embedder.embed_single(text)
        return self.llm.embed_single(text, config.EMBEDDING_MODEL)

    # ── PDF Extraction ────────────────────────────────────────────────

    @staticmethod
    def extract_text_from_pdf(file_bytes: bytes, filename: str) -> List[Document]:
        """Extract text from a PDF and return a list of Documents
        (one per page) with metadata. All text is forced to plain str so
        nothing strange (bytes, custom subclasses) sneaks into pickling/JSON."""
        documents = []
        reader = PdfReader(io.BytesIO(file_bytes))
        for page_num, page in enumerate(reader.pages, start=1):
            try:
                raw = page.extract_text()
            except Exception:
                raw = ""
            # Coerce bytes / custom string subclasses to plain str
            if raw is None:
                raw = ""
            elif isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            else:
                raw = str(raw)
            if raw.strip():
                documents.append(
                    Document(
                        page_content=raw,
                        metadata={
                            "source": str(filename),
                            "page": int(page_num),
                        },
                    )
                )
        return documents

    # ── Chunking ──────────────────────────────────────────────────────

    @staticmethod
    def chunk_documents(documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks for embedding."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return splitter.split_documents(documents)

    # ── Vector Store ──────────────────────────────────────────────────

    def build_vectorstore(
        self, chunks: List[Document], progress_callback=None
    ) -> None:
        """Create or extend the vector store from document chunks."""
        if self.vectorstore is None:
            self.vectorstore = SimpleVectorStore()

        # Embed chunks in batches for progress reporting
        texts = [c.page_content for c in chunks]
        batch_size = 5
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = self._embed_batch(batch)
            all_embeddings.extend(batch_embeddings)

            if progress_callback:
                pct = min(0.9, 0.4 + 0.5 * (i + len(batch)) / len(texts))
                progress_callback(
                    pct,
                    f"Embedding chunks {i + 1}–{min(i + len(batch), len(texts))} of {len(texts)}…",
                )

        self.vectorstore.add(chunks, all_embeddings)
        self.total_chunks += len(chunks)
        self._save_vectorstore()

    def _save_vectorstore(self) -> None:
        """Persist the vector store to disk."""
        if self.vectorstore:
            self.vectorstore.save(config.VECTORSTORE_DIR)

    def _try_load_vectorstore(self) -> None:
        """Load an existing vector store from disk if available."""
        store = SimpleVectorStore.load(config.VECTORSTORE_DIR)
        if store is not None:
            self.vectorstore = store
            self.total_chunks = len(store.documents)
            # Reconstruct indexed files from metadata
            files = set()
            for doc in store.documents:
                src = doc.metadata.get("source")
                if src:
                    files.add(src)
            self.indexed_files = list(files)

    def clear_vectorstore(self) -> None:
        """Delete the vector store and reset state."""
        self.vectorstore = None
        self.indexed_files = []
        self.total_chunks = 0
        # Remove persisted files
        if os.path.exists(config.VECTORSTORE_DIR):
            import shutil
            shutil.rmtree(config.VECTORSTORE_DIR, ignore_errors=True)

    # ── Ingest (Upload → Extract → Chunk → Index) ────────────────────

    def ingest_pdfs(
        self, uploaded_files: list, progress_callback=None
    ) -> Tuple[int, int]:
        """
        Full ingestion pipeline for uploaded PDF files.

        Returns:
            (num_documents, num_chunks) processed.
        """
        all_chunks: List[Document] = []
        total_pages = 0

        for i, uploaded_file in enumerate(uploaded_files):
            file_bytes = uploaded_file.read()
            filename = uploaded_file.name

            if progress_callback:
                progress_callback(
                    0.1 * (i + 1) / len(uploaded_files),
                    f"Extracting text from {filename}…",
                )

            # Extract pages
            pages = self.extract_text_from_pdf(file_bytes, filename)
            total_pages += len(pages)

            # Chunk
            chunks = self.chunk_documents(pages)
            all_chunks.extend(chunks)

            self.indexed_files.append(filename)

            if progress_callback:
                progress_callback(
                    0.1 + 0.3 * (i + 1) / len(uploaded_files),
                    f"Processed {filename} ({len(pages)} pages → {len(chunks)} chunks)",
                )

        if all_chunks:
            self.build_vectorstore(all_chunks, progress_callback)

            if progress_callback:
                progress_callback(0.95, "Saving index…")
            self._save_vectorstore()

        return len(uploaded_files), len(all_chunks)

    # ── Retrieval ────────────────────────────────────────────────────

    def retrieve(self, query: str, k: int = None) -> List[Document]:
        """Retrieve the top-k most relevant chunks for a query."""
        if self.vectorstore is None:
            return []
        k = k or config.RETRIEVAL_TOP_K
        query_vec = self._embed_single(query)
        return self.vectorstore.search(query_vec, k=k)

    # ── Generation ───────────────────────────────────────────────────

    def generate(self, query: str, mode: str = "qa") -> Tuple[str, List[Document]]:
        """
        Full RAG: retrieve context → format prompt → call LLM.

        Args:
            query: User's question or request.
            mode:  One of 'qa', 'summarize', 'mcq', 'eli5'.

        Returns:
            (answer_text, source_documents)
        """
        if self.vectorstore is None:
            return (
                "⚠️ No documents have been indexed yet. "
                "Please upload and process PDFs first.",
                [],
            )

        # Special path: SUMMARIZE uses map-reduce so we cover EVERY chunk
        # (the request body is too big to send everything in one call).
        if mode == "summarize" and self.vectorstore is not None:
            return self._summarize_map_reduce(query)

        # Retrieve — broader top-k for MCQ, focused for Q&A / ELI5.
        if mode == "mcq":
            broad_k = min(40, self.total_chunks or 40)
            docs = self.retrieve(query, k=broad_k)
        else:
            docs = self.retrieve(query)

        if not docs:
            return (
                "The answer is not available in the provided material.",
                [],
            )

        # Build context string
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "?")
            context_parts.append(
                f"[Chunk {i} | {source}, Page {page}]\n{doc.page_content}"
            )
        context = "\n\n---\n\n".join(context_parts)

        # Select prompt template
        prompt_template = PROMPT_MAP.get(mode, PROMPT_MAP["qa"])
        prompt = prompt_template.format(context=context, question=query)

        # Call LLM (provider-agnostic)
        try:
            response = self.llm.generate(
                prompt=prompt,
                model=config.LLM_MODEL,
                temperature=config.LLM_TEMPERATURE,
                num_ctx=config.LLM_NUM_CTX,
            )
        except Exception as exc:
            return _format_llm_request_error(exc), docs
        return response, docs

    # ── Map-reduce summarization (covers EVERY chunk for exam prep) ──
    def _summarize_map_reduce(self, query: str) -> Tuple[str, List[Document]]:
        """Summarise the entire indexed corpus in two passes:
           1) MAP — split chunks into batches, extract bullet points per batch
           2) REDUCE — merge all bullet points into a single exam-ready summary
        """
        all_docs = self.vectorstore.documents
        if not all_docs:
            return ("The answer is not available in the provided material.", [])

        # MAP step — chunk everything into batches that fit safely in one request
        BATCH_SIZE = 12  # ~12 chunks × ~1000 chars ≈ 12k chars per call (well under Groq's payload cap)
        batches = [all_docs[i : i + BATCH_SIZE] for i in range(0, len(all_docs), BATCH_SIZE)]

        partial_extracts: list[str] = []
        for batch_idx, batch in enumerate(batches, 1):
            batch_text = "\n\n---\n\n".join(
                f"[Chunk {j} | {d.metadata.get('source', '?')}, "
                f"Page {d.metadata.get('page', '?')}]\n{d.page_content}"
                for j, d in enumerate(batch, 1)
            )
            map_prompt = (
                "You are extracting study notes from part of a course PDF. "
                "List every distinct concept, technique, definition, formula, "
                "and named method that appears in the text below. "
                "Output ONLY a bullet list — no introduction, no conclusion. "
                "Use the exact terminology from the source.\n\n"
                f"TEXT:\n{batch_text}\n\n"
                "BULLET LIST OF EVERY CONCEPT IN THIS TEXT:"
            )
            try:
                bullets = self.llm.generate(
                    prompt=map_prompt,
                    model=config.LLM_MODEL,
                    temperature=0.1,  # very factual
                    num_ctx=config.LLM_NUM_CTX,
                )
                partial_extracts.append(bullets.strip())
            except Exception as e:
                # Don't fail the whole summary because one batch hit a transient error
                partial_extracts.append(f"[batch {batch_idx} failed: {e}]")

        merged_bullets = "\n\n".join(partial_extracts)

        # REDUCE step — collapse the merged bullets into the polished output
        reduce_prompt = (
            "You are an AI Study Assistant building EXAM-READY notes for a "
            "student. Below is a comprehensive list of bullet points extracted "
            "from EVERY chunk of their course PDF. This is for their EXAM — "
            "do NOT lose information.\n\n"
            "ABSOLUTE RULES:\n"
            "1. Every distinct technical term that appears in the bullets "
            "below MUST appear somewhere in your final output.\n"
            "2. Use the EXACT terminology from the bullets — do not rename "
            "concepts (e.g. keep 'stopwords' as 'stopwords', not 'common "
            "words').\n"
            "3. Do not invent anything not in the bullets.\n"
            "4. Do not merge two distinct concepts into one bullet.\n"
            "5. If a term appears in multiple bullets, list it ONCE in Key "
            "Concepts and ONCE in Techniques (if applicable).\n\n"
            f"STUDENT'S REQUEST: {query}\n\n"
            f"EXTRACTED BULLETS FROM ALL CHUNKS:\n{merged_bullets}\n\n"
            "Now produce the final summary in EXACTLY this structure:\n\n"
            "### 📝 Summary:\n"
            "[3–5 sentence overview naming the main topic and the high-level "
            "areas covered.]\n\n"
            "### 🎯 Main Themes:\n"
            "• Theme 1\n• Theme 2\n• Theme 3\n• …(more if needed)\n\n"
            "### 📌 Key Concepts:\n"
            "[ONE bullet per distinct concept above. NEVER drop a term. Use "
            "the source's wording. Aim for 15+ bullets.]\n"
            "• **Concept** — short definition\n"
            "• …\n\n"
            "### 🔧 Techniques / Methods:\n"
            "[ONE bullet per technique. Same rules — do not drop, do not "
            "rename. Aim for 10+ bullets.]\n"
            "• **Method** — what it does\n"
            "• …\n\n"
            "### 💡 Important Details for the Exam:\n"
            "[Formulas, edge-cases, examples, step-by-step procedures.]"
        )
        try:
            final = self.llm.generate(
                prompt=reduce_prompt,
                model=config.LLM_MODEL,
                temperature=0.2,
                num_ctx=config.LLM_NUM_CTX,
            )
        except Exception as exc:
            return _format_llm_request_error(exc), all_docs
        return final, all_docs
