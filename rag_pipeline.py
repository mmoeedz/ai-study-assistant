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
        """Add overlap between adjacent chunks, ensuring no chunk exceeds size."""
        if not chunks or self.chunk_overlap == 0:
            return chunks

        merged = []
        for i, chunk in enumerate(chunks):
            if i > 0 and self.chunk_overlap > 0:
                # Prepend the last N characters from the previous chunk for context
                prev = chunks[i - 1]
                # Only take as much overlap as exists in the previous chunk
                overlap_len = min(self.chunk_overlap, len(prev))
                overlap_text = prev[-overlap_len:] if overlap_len > 0 else ""
                if overlap_text:
                    chunk = overlap_text + chunk
                # Ensure chunk doesn't exceed size limit by trimming from the end
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
        # Normalize vectors to unit length before computing dot product (true cosine similarity)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []  # Query vector is all zeros, can't compute similarity
        
        q_normalized = q / q_norm
        
        # Normalize embeddings row-wise
        embedding_norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        # Avoid division by zero for zero vectors in the store
        embedding_norms = np.where(embedding_norms == 0, 1, embedding_norms)
        embeddings_normalized = self.embeddings / embedding_norms
        
        # Compute cosine similarities via dot product
        similarities = embeddings_normalized @ q_normalized

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
            try:
                resp = requests.post(
                    f"{self.base_url}/api/embed",
                    json={"model": model, "input": text},
                    timeout=120,
                )
                resp.raise_for_status()
                embeddings.append(resp.json()["embeddings"][0])
            except (requests.exceptions.RequestException, KeyError, IndexError) as e:
                raise RuntimeError(f"Failed to embed text with Ollama: {e}")
        return embeddings

    def embed_single(self, text: str, model: str) -> List[float]:
        return self.embed([text], model)[0]

    def generate(
        self, prompt: str, model: str,
        temperature: float = 0.3, num_ctx: int = 4096,
        image_base64: str = None,
        history: list = None,
    ) -> str:
        full_prompt = ""
        if history:
            for h in history:
                q = h.get("query", "")
                if q.startswith("📖 Upload:"):
                    continue
                a = h.get("answer", "")
                full_prompt += f"User: {q}\nAssistant: {a}\n"
        full_prompt += f"User: {prompt}\nAssistant:"

        payload = {
            "model": model, "prompt": full_prompt, "stream": False,
            "options": {"temperature": temperature, "num_ctx": num_ctx},
        }
        if image_base64:
            payload["images"] = [image_base64]
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
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
        image_base64: str = None,
        history: list = None,
    ) -> str:
        import time as _t  # local import to avoid name shadow
        last_exc = None
        
        has_any_image = bool(image_base64)
        if history:
            for h in history:
                if h.get("image_base64"):
                    has_any_image = True
                    break

        active_model = "meta-llama/llama-4-scout-17b-16e-instruct" if has_any_image else model
        
        for attempt in range(4):  # up to 4 attempts on 429 / transient errors
            try:
                messages = []
                if history:
                    for h in history:
                        q = h.get("query", "")
                        if q.startswith("📖 Upload:"):
                            continue
                        a = h.get("answer", "")
                        img = h.get("image_base64", None)
                        
                        if img:
                            messages.append({
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": q},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{img}"
                                        }
                                    }
                                ]
                            })
                        else:
                            messages.append({"role": "user", "content": q})
                        
                        if a:
                            messages.append({"role": "assistant", "content": a})

                # Append the current prompt
                if image_base64:
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    })
                else:
                    messages.append({"role": "user", "content": prompt})

                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": active_model,
                        "messages": messages,
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
                try:
                    resp.raise_for_status()
                    data = resp.json()
                    # Safely extract response with better error handling
                    if "choices" not in data or not data["choices"]:
                        raise ValueError("Empty response from API - no choices returned")
                    if "message" not in data["choices"][0]:
                        raise ValueError("Unexpected API response structure")
                    return data["choices"][0]["message"]["content"]
                except (ValueError, KeyError) as e:
                    raise RuntimeError(f"Failed to parse LLM response: {e}")
                except requests.exceptions.HTTPError as e:
                    # Handle specific error codes
                    if resp.status_code == 413:
                        raise RuntimeError(
                            "📦 Request too large. The document context exceeds API limits. "
                            "Try uploading fewer or shorter PDFs."
                        )
                    elif resp.status_code == 401:
                        raise RuntimeError(
                            "🔑 Invalid or expired Groq API key. "
                            "Please check your GROQ_API_KEY in .streamlit/secrets.toml"
                        )
                    elif resp.status_code == 403:
                        raise RuntimeError(
                            "🚫 Access denied. Your Groq API key may not have permission for this model."
                        )
                    elif resp.status_code == 400:
                        raise RuntimeError(
                            "⚠️ Bad request to Groq API. Check your prompt or model name."
                        )
                    elif 500 <= resp.status_code < 600:
                        # 5xx errors → retry once or twice with backoff
                        last_exc = e
                        if attempt < 2:
                            _t.sleep(2 ** attempt)
                            continue
                    last_exc = e
                    raise
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                # Network errors - retry with backoff
                last_exc = e
                if attempt < 3:
                    _t.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"Network error connecting to Groq API: {e}")
            except requests.exceptions.RequestException as e:
                # Catch other request errors
                last_exc = e
                if attempt < 2:
                    _t.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"Request error: {e}")
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

    @staticmethod
    def extract_text_from_file(file_bytes: bytes, filename: str) -> List[Document]:
        """Extract text from a file (PDF, DOCX, TXT, MD, etc.) and return a list of Documents.
        All text is forced to plain str."""
        ext = filename.split(".")[-1].lower() if "." in filename else ""
        documents = []

        if ext == "pdf":
            return StudyAssistant.extract_text_from_pdf(file_bytes, filename)
        
        elif ext == "docx":
            try:
                import zipfile
                import xml.etree.ElementTree as ET
                with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
                    xml_content = z.read('word/document.xml')
                    root = ET.fromstring(xml_content)
                    paragraphs = []
                    for para in root.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'):
                        texts = [node.text for node in para.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t') if node.text]
                        if texts:
                            paragraphs.append("".join(texts))
                    
                    full_text = "\n\n".join(paragraphs)
                    if full_text.strip():
                        documents.append(
                            Document(
                                page_content=full_text,
                                metadata={
                                    "source": str(filename),
                                    "page": 1,
                                },
                            )
                        )
            except Exception:
                pass
                
        else:
            try:
                raw = file_bytes.decode("utf-8", errors="replace")
                if raw.strip():
                    documents.append(
                        Document(
                            page_content=raw,
                            metadata={
                                "source": str(filename),
                                "page": 1,
                            },
                        )
                    )
            except Exception:
                pass
                
        return documents

    def generate_auto_summary(self) -> str:
        """Generate an automatic short summary of the uploaded files to welcome the user."""
        if not self.vectorstore or self.total_chunks == 0:
            return "Welcome! No documents have been indexed yet."
        
        docs = self.retrieve("main topics, overview, table of contents, introduction", k=4)
        if not docs:
            docs = self.vectorstore.documents[:4]
            
        context_parts = []
        for i, d in enumerate(docs, 1):
            context_parts.append(f"[Document Chunk {i}]: {d.page_content[:1000]}")
        context = "\n\n".join(context_parts)
        
        prompt = f"""You are an AI Study Assistant. The user has uploaded document(s) containing the following material:

{context}

Provide a concise, high-level short summary (4-6 sentences) of what these documents cover. 
Act like a normal advanced LLM that has just read these documents. Be professional, direct, and welcoming. 
Conclude with a brief note telling the user that they can now ask questions, generate MCQs, or request a detailed summary.
"""
        try:
            summary = self.llm.generate(
                prompt=prompt,
                model=config.LLM_MODEL,
                temperature=0.4,
                num_ctx=config.LLM_NUM_CTX,
            )
            return summary
        except Exception as e:
            return f"Welcome! I have successfully processed and indexed your documents. You can now start asking questions!"

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
            pages = self.extract_text_from_file(file_bytes, filename)
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
    
    def retrieve_all(self) -> List[Document]:
        """Retrieve ALL documents (for comprehensive summarization)."""
        if self.vectorstore is None:
            return []
        return self.vectorstore.documents

    # ── Generation ───────────────────────────────────────────────────

    def generate(self, query: str, mode: str = "qa", image_base64: str = None, history: list = None) -> Tuple[str, List[Document]]:
        """
        Full RAG: retrieve context → format prompt → call LLM.

        Args:
            query: User's question or request.
            mode:  One of 'qa', 'summarize', 'mcq', 'eli5'.
            image_base64: Optional base64 encoded image string for multimodal support.
            history: Optional conversation history.

        Returns:
            (answer_text, source_documents)
        """
        if self.vectorstore is None:
            if mode == "coding":
                context = "No study material uploaded."
                prompt_template = PROMPT_MAP.get(mode, PROMPT_MAP["qa"])
                prompt = prompt_template.format(context=context, question=query)
                response = self.llm.generate(
                    prompt=prompt,
                    model=config.LLM_MODEL,
                    temperature=config.LLM_TEMPERATURE,
                    num_ctx=config.LLM_NUM_CTX,
                    image_base64=image_base64,
                    history=history,
                )
                return response, []
            else:
                return (
                    "⚠️ No documents have been indexed yet. "
                    "Please upload and process documents first.",
                    [],
                )

        # Special handling for summarize and quiz: USE COMPREHENSIVE CONTEXT with batching
        if mode == "summarize":
            return self._generate_comprehensive_summary(query)
        
        if mode == "quiz":
            # For quiz generation, use comprehensive document coverage but with batch processing to avoid 413 errors
            return self._generate_quiz_batched(query)
        
        # For other modes: retrieve relevant chunks
        if mode == "mcq":
            k = min(8, self.total_chunks or 8)
            docs = self.retrieve(query, k=k)
        else:
            # Q&A and Coding: get top relevant chunks
            docs = self.retrieve(query)

        if not docs and mode != "coding":
            return (
                "The answer is not available in the provided material.",
                [],
            )

        # Build context string - NO TRUNCATION for content, let LLM see everything
        context_parts = []
        
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "?")
            
            # FULL content for non-summarize modes (still respect payload limit)
            content = doc.page_content
            
            chunk_text = f"[Chunk {i} | {source}, Page {page}]\n{content}"
            context_parts.append(chunk_text)

        if not context_parts and mode == "coding":
            context = "No relevant study material found."
        else:
            context = "\n\n---\n\n".join(context_parts)

        # Select prompt template
        prompt_template = PROMPT_MAP.get(mode, PROMPT_MAP["qa"])
        prompt = prompt_template.format(context=context, question=query)

        # Call LLM (provider-agnostic)
        response = self.llm.generate(
            prompt=prompt,
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            num_ctx=config.LLM_NUM_CTX,
            image_base64=image_base64,
            history=history,
        )
        return response, docs

    # ── Comprehensive Summary (includes ALL documents) ────────────────
    def _generate_comprehensive_summary(self, query: str) -> Tuple[str, List[Document]]:
        """
        Generate summary from ALL documents with full content (not just top chunks).
        Processes in batches to avoid 413 payload errors.
        
        Returns complete information from all pages.
        """
        all_docs = self.retrieve_all()
        if not all_docs:
            return ("The answer is not available in the provided material.", [])

        # Process all documents in batches to avoid 413 errors
        BATCH_SIZE = 30  # ~30 chunks × ~1000 chars = ~30KB per batch (safe)
        batches = [all_docs[i : i + BATCH_SIZE] for i in range(0, len(all_docs), BATCH_SIZE)]
        
        all_summaries = []
        
        for batch_idx, batch in enumerate(batches, 1):
            # Build context with FULL content (no truncation)
            context_parts = []
            for j, doc in enumerate(batch, 1):
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "?")
                # FULL content - no truncation
                chunk_text = f"[Chunk {j} | {source}, Page {page}]\n{doc.page_content}"
                context_parts.append(chunk_text)
            
            context = "\n\n---\n\n".join(context_parts)
            
            # Create batch-specific prompt
            prompt_template = PROMPT_MAP.get("summarize", PROMPT_MAP["qa"])
            prompt = prompt_template.format(context=context, question=query)
            
            try:
                summary = self.llm.generate(
                    prompt=prompt,
                    model=config.LLM_MODEL,
                    temperature=0.3,
                    num_ctx=config.LLM_NUM_CTX,
                )
                all_summaries.append(summary)
            except Exception as e:
                all_summaries.append(f"[Batch {batch_idx} processing error: {str(e)[:100]}]")
        
        # Merge all batch summaries into one comprehensive summary
        if len(all_summaries) == 1:
            final_summary = all_summaries[0]
        else:
            # Combine multiple batch summaries into comprehensive final summary
            merged_content = "\n\n---\n\n".join(all_summaries)
            merge_prompt = f"""You have received multiple summary sections from different batches of course material.
Your task is to merge them into ONE comprehensive, exam-ready summary that includes ALL information.

IMPORTANT RULES:
1. Do NOT drop any technical terms, concepts, methods, or formulas
2. Do NOT skip any information from any batch
3. Combine all Key Concepts, Techniques, and Details sections
4. Maintain the hierarchical structure (## Summary, ## Main Concepts, etc.)
5. Include EVERYTHING mentioned across all batches

BATCH SUMMARIES TO MERGE:
{merged_content}

Now produce the FINAL COMPREHENSIVE SUMMARY that includes ALL information from every batch above:

## Summary
[Comprehensive overview covering all batches]

## Main Concepts
[EVERY concept from every batch - do not skip any]

## Key Techniques & Methods
[EVERY technique from every batch]

## Important Details for the Exam
[ALL formulas, examples, edge cases from every batch]
"""
            try:
                final_summary = self.llm.generate(
                    prompt=merge_prompt,
                    model=config.LLM_MODEL,
                    temperature=0.2,
                    num_ctx=config.LLM_NUM_CTX,
                )
            except Exception as e:
                final_summary = f"Error merging summaries: {str(e)}\n\nPartial content:\n{merged_content[:5000]}"
        
        return final_summary, all_docs

    # ── Quiz Generation with Batch Processing ───────────────────────
    def _generate_quiz_batched(self, query: str) -> Tuple[str, List[Document]]:
        """
        Generate 10 multiple-choice questions from comprehensive document coverage.
        Uses batch processing to stay within API context limits.
        
        Returns JSON string with 10 questions and explanations.
        """
        all_docs = self.retrieve_all()
        if not all_docs:
            return (
                "⚠️ No documents have been indexed yet. "
                "Please upload and process documents first.",
                [],
            )

        # Smart batch sizing: aim for ~20-25KB per batch (safe margin below limits)
        BATCH_SIZE = 25  # ~25 chunks × ~1000 chars = ~25KB
        total_chunks = len(all_docs)
        
        # If we have few docs, send all at once
        if total_chunks <= BATCH_SIZE:
            context_parts = []
            for i, doc in enumerate(all_docs, 1):
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "?")
                chunk_text = f"[Chunk {i} | {source}, Page {page}]\n{doc.page_content}"
                context_parts.append(chunk_text)
            
            context = "\n\n---\n\n".join(context_parts)
            prompt_template = PROMPT_MAP.get("quiz", PROMPT_MAP["qa"])
            prompt = prompt_template.format(context=context, question=query)
            
            try:
                response = self.llm.generate(
                    prompt=prompt,
                    model=config.LLM_MODEL,
                    temperature=0.3,
                    num_ctx=config.LLM_NUM_CTX,
                )
                
                # Validate response is not empty
                if not response or not response.strip():
                    raise ValueError("LLM returned empty response")
                
                # Clean and parse response
                cleaned = response.strip()
                
                # Remove markdown code block wrappers if present
                if "```" in cleaned:
                    # Extract content between ``` markers
                    parts = cleaned.split("```")
                    if len(parts) >= 2:
                        # Find the JSON part (usually the middle section)
                        cleaned = parts[1].strip()
                        if cleaned.lower().startswith("json"):
                            cleaned = cleaned[4:].strip()
                
                # Find JSON array bounds
                start_idx = cleaned.find("[")
                end_idx = cleaned.rfind("]")
                if start_idx == -1 or end_idx == -1:
                    raise ValueError(f"No JSON array found in response: {cleaned[:100]}")
                
                cleaned = cleaned[start_idx:end_idx+1]
                
                # Parse JSON
                quiz_json = json.loads(cleaned)
                if not isinstance(quiz_json, list) or len(quiz_json) == 0:
                    raise ValueError("Response is not a non-empty JSON array")
                
                return json.dumps(quiz_json, ensure_ascii=False, indent=2), all_docs
            except Exception as e:
                raise RuntimeError(f"Quiz generation failed: {str(e)}")
        
        # For large document sets: generate questions batch-by-batch, then merge
        batches = [all_docs[i : i + BATCH_SIZE] for i in range(0, total_chunks, BATCH_SIZE)]
        
        all_questions = []
        
        for batch_idx, batch in enumerate(batches, 1):
            context_parts = []
            for j, doc in enumerate(batch, 1):
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "?")
                chunk_text = f"[Chunk {j} | {source}, Page {page}]\n{doc.page_content}"
                context_parts.append(chunk_text)
            
            context = "\n\n---\n\n".join(context_parts)
            
            # Generate questions for this batch (fewer per batch, combine later)
            batch_question_count = max(2, 10 // len(batches))  # Distribute 10 questions across batches
            batch_prompt = f"""You are an expert examiner creating multiple-choice questions.

From this material, create {batch_question_count} unique multiple-choice questions (each with exactly 4 options A-D and one correct answer).

Material:
{context}

Return ONLY valid JSON array with {batch_question_count} objects in this format (no markdown, no ```json wrapper):
[
  {{
    "question": "question text",
    "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
    "answer": "A",
    "explanation": "why this is correct"
  }}
]"""
            
            try:
                batch_response = self.llm.generate(
                    prompt=batch_prompt,
                    model=config.LLM_MODEL,
                    temperature=0.3,
                    num_ctx=config.LLM_NUM_CTX,
                )
                
                # Validate response is not empty
                if not batch_response or not batch_response.strip():
                    continue
                
                # Parse batch response with robust cleaning
                cleaned = batch_response.strip()
                
                # Remove markdown code block wrappers
                if "```" in cleaned:
                    parts = cleaned.split("```")
                    if len(parts) >= 2:
                        cleaned = parts[1].strip()
                        if cleaned.lower().startswith("json"):
                            cleaned = cleaned[4:].strip()
                
                # Find JSON array bounds
                start_idx = cleaned.find("[")
                end_idx = cleaned.rfind("]")
                if start_idx == -1 or end_idx == -1:
                    continue
                
                cleaned = cleaned[start_idx:end_idx+1]
                
                # Try to parse JSON
                batch_questions = json.loads(cleaned)
                if isinstance(batch_questions, list) and len(batch_questions) > 0:
                    all_questions.extend(batch_questions)
            except (json.JSONDecodeError, ValueError, IndexError):
                # Continue with next batch on parse errors
                continue
            except Exception:
                # Skip this batch on any other error
                continue
        
        # Return up to 10 questions (or all if fewer batches generated)
        final_questions = all_questions[:10]
        
        if not final_questions:
            # Fallback: Generate meaningful questions from document content
            final_questions = self._generate_fallback_quiz(all_docs)
        
        # Don't pad with generic placeholders - return actual questions from content
        # If we have fewer than 10, that's okay - better quality over quantity
        # Ensure at least some questions exist
        if not final_questions:
            return (
                "⚠️ Could not generate quiz from available documents. "
                "Please ensure your documents contain sufficient content.",
                all_docs,
            )
        
        return json.dumps(final_questions[:10], ensure_ascii=False, indent=2), all_docs
    
    # ── Fallback Quiz Generator ─────────────────────────────────────
    def _generate_fallback_quiz(self, docs: List[Document]) -> List[Dict]:
        """
        Generate meaningful quiz questions from document content as fallback.
        Creates diverse questions with actual content from documents.
        Ensures at least 8-10 real questions, even with small documents.
        """
        fallback_questions = []
        
        # Strategy 1: Extract individual sentences as questions
        all_sentences = []
        for doc in docs:
            content = doc.page_content
            source = doc.metadata.get("source", "Document")
            
            # Split into sentences
            sentences = [s.strip() for s in content.split(".") if 15 < len(s.strip()) < 300]
            for sent in sentences:
                all_sentences.append((sent, source))
        
        # Create questions from sentences
        for idx, (sentence, source) in enumerate(all_sentences[:10]):
            if idx >= 10:
                break
            
            # Create question asking about this fact
            words = sentence.split()
            key_phrase = " ".join(words[:min(7, len(words))])
            
            fallback_questions.append({
                "question": f"According to the material: {key_phrase}...?",
                "options": {
                    "A": sentence[:120].rstrip() + ("..." if len(sentence) > 120 else ""),
                    "B": "This is not correct based on the material",
                    "C": "Opposite of what is stated",
                    "D": "Not mentioned in the material"
                },
                "answer": "A",
                "explanation": f"From {source}: {sentence}"
            })
        
        # Strategy 2: If still need more, create comparison/contrast questions
        if len(fallback_questions) < 8 and len(docs) > 1:
            for i in range(min(2, len(docs) - 1)):
                if len(fallback_questions) >= 10:
                    break
                
                doc1_text = docs[i].page_content[:200]
                doc2_text = docs[i+1].page_content[:200] if i+1 < len(docs) else docs[i].page_content[200:400]
                doc1_source = docs[i].metadata.get("source", "Document 1")
                
                fallback_questions.append({
                    "question": f"Which statement is covered in the study material?",
                    "options": {
                        "A": doc1_text[:100],
                        "B": "Only mentioned outside the provided material",
                        "C": "Contradicted by the material",
                        "D": "Not applicable to this topic"
                    },
                    "answer": "A",
                    "explanation": f"From {doc1_source}: This concept is covered in your study materials."
                })
        
        # Strategy 3: If still need more, create definition/concept questions from keywords
        if len(fallback_questions) < 10:
            for doc in docs[:3]:
                if len(fallback_questions) >= 10:
                    break
                
                content = doc.page_content
                words = content.split()
                
                # Find noun phrases (potential concepts)
                for i in range(0, len(words)-2, 3):
                    if len(fallback_questions) >= 10:
                        break
                    
                    phrase = " ".join(words[i:i+3])
                    if len(phrase) > 8 and len(phrase) < 50:
                        fallback_questions.append({
                            "question": f"The material discusses the concept of '{phrase}'. What is its main purpose?",
                            "options": {
                                "A": "Central topic in the provided material",
                                "B": "Minor detail not worth studying",
                                "C": "Not mentioned in the material",
                                "D": "Only theoretical, not practical"
                            },
                            "answer": "A",
                            "explanation": f"'{phrase}' appears in the study material and represents an important concept."
                        })
        
        # Return unique questions (remove exact duplicates)
        seen = set()
        unique_questions = []
        for q in fallback_questions:
            q_key = q["question"]
            if q_key not in seen and len(unique_questions) < 10:
                seen.add(q_key)
                unique_questions.append(q)
        
        return unique_questions

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
        # Larger batch size to reduce API calls (safe: 15 chunks × 400 chars = 6KB << 30KB limit)
        BATCH_SIZE = 15  # ~15 chunks × ~400 chars ≈ 6KB per call (much faster, still safe)
        batches = [all_docs[i : i + BATCH_SIZE] for i in range(0, len(all_docs), BATCH_SIZE)]
        
        # Safety limit: don't process more than MAX_SUMMARIZE_BATCHES
        # (prevents hanging on huge documents)
        if len(batches) > config.MAX_SUMMARIZE_BATCHES:
            batches = batches[:config.MAX_SUMMARIZE_BATCHES]

        partial_extracts: list[str] = []
        for batch_idx, batch in enumerate(batches, 1):
            batch_text_parts = []
            for j, d in enumerate(batch, 1):
                chunk_content = d.page_content[:config.MAX_CHUNK_DISPLAY]
                if len(d.page_content) > config.MAX_CHUNK_DISPLAY:
                    chunk_content += "…"
                batch_text_parts.append(
                    f"[Chunk {j} | {d.metadata.get('source', '?')}, "
                    f"Page {d.metadata.get('page', '?')}]\n{chunk_content}"
                )
            batch_text = "\n\n---\n\n".join(batch_text_parts)
            
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
        # Simplified prompt for speed (no repetitive rules)
        reduce_prompt = (
            f"You are an AI Study Assistant. The student asked: '{query}'\n\n"
            "Below are bullet points extracted from course material.\n"
            "Organize them into an EXAM-READY summary with these sections:\n\n"
            "### 📝 Summary:\n"
            "(3-5 sentences overview)\n\n"
            "### 🎯 Main Themes:\n"
            "(Key topics covered)\n\n"
            "### 📌 Key Concepts:\n"
            "(Technical terms with brief definitions)\n\n"
            "### 🔧 Techniques / Methods:\n"
            "(Procedures and approaches)\n\n"
            "### 💡 Exam Tips:\n"
            "(Important formulas, edge cases, examples)\n\n"
            "---\n"
            f"EXTRACTED CONTENT:\n{merged_bullets}"
        )
        final = self.llm.generate(
            prompt=reduce_prompt,
            model=config.LLM_MODEL,
            temperature=0.2,
            num_ctx=config.LLM_NUM_CTX,
        )
        return final, all_docs